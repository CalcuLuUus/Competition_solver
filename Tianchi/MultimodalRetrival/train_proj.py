#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time, random, argparse
from typing import List, Dict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

# -----------------------
# Utils & Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def human_time():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

# -----------------------
# Dataset
# -----------------------
class RetrievalDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                item_ids = [iid for iid in ex["item_ids"]
                            if os.path.exists(os.path.join(image_root, f"{iid}.png"))]
                if item_ids:
                    self.samples.append({
                        "query_id": ex["query_id"],
                        "query_text": ex["query_text"],
                        "item_ids": item_ids
                    })
        assert len(self.samples) > 0, f"No valid samples in {jsonl_path}"

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        pos_item_id = random.choice(ex["item_ids"])
        img_path = os.path.join(args.train_image_root if self.is_train else args.val_image_root,
                                f"{pos_item_id}.png")
        # 训练与验证用不同根目录；这里由外部注入属性 is_train
        img = Image.open(img_path).convert("RGB")
        return {"query_text": ex["query_text"], "image": img, "pos_item_id": pos_item_id}

class Collator:
    def __init__(self, processor, max_length=64):
        self.processor = processor
        self.max_length = max_length
    def __call__(self, batch):
        texts = [b["query_text"] for b in batch]
        images = [b["image"] for b in batch]
        text_inputs = self.processor(text=texts, padding='max_length',
                                     max_length=64, return_tensors="pt")
        image_inputs = self.processor(images=images, return_tensors="pt")
        return text_inputs, image_inputs

# -----------------------
# Evaluation helpers
# -----------------------
@torch.no_grad()
def build_image_index(model, processor, image_root: str, image_ids: List[int],
                      batch_size: int, device: torch.device):
    paths = [os.path.join(image_root, f"{iid}.png") for iid in image_ids
             if os.path.exists(os.path.join(image_root, f"{iid}.png"))]
    feats = []
    for i in trange(0, len(paths), batch_size, desc="Indexing images (val)"):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        image_inputs = processor(images=imgs, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        feats_b = model.get_image_features(**image_inputs)
        feats_b = feats_b / feats_b.norm(dim=-1, keepdim=True)
        feats.append(feats_b)
    if len(feats) == 0:
        return torch.empty(0, model.text_model.head.out_features, device=device), []
    return torch.cat(feats, dim=0), paths

@torch.no_grad()
def evaluate(model, processor, val_queries: List[Dict], val_image_root: str,
             batch_size: int, device: torch.device, writer: SummaryWriter = None, global_step: int = 0):
    # 1) 收集所有验证集图片的去重 item_id
    all_item_ids = set()
    for q in val_queries:
        for iid in q["item_ids"]:
            if os.path.exists(os.path.join(val_image_root, f"{iid}.png")):
                all_item_ids.add(iid)
    all_item_ids = sorted(list(all_item_ids))
    # 2) 建索引（每次评估都重建，因为 vision head 会更新）
    image_index, image_paths = build_image_index(
        model, processor, val_image_root, all_item_ids, batch_size=batch_size, device=device
    )
    # 建立 path -> item_id 映射
    path2id = {p: int(os.path.basename(p)[:-4]) for p in image_paths}

    # 3) 文本编码并检索
    recalls = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    total = len(val_queries)
    for i in trange(0, total, batch_size, desc="Evaluating (val)"):
        chunk = val_queries[i:i+batch_size]
        texts = [c["query_text"] for c in chunk]
        text_inputs = processor(text=texts, padding='max_length', max_length=64, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = text_feats @ image_index.t()  # [B, N]
        ranks = torch.argsort(sims, dim=1, descending=True).cpu().tolist()

        for bi, rank_list in enumerate(ranks):
            pos_set = set([iid for iid in chunk[bi]["item_ids"]
                           if os.path.exists(os.path.join(val_image_root, f"{iid}.png"))])
            first_rank = None
            for r_idx, idx_img in enumerate(rank_list):
                item_id = path2id[image_paths[idx_img]]
                if item_id in pos_set:
                    first_rank = r_idx + 1
                    break
            if first_rank is None:
                continue
            if first_rank <= 1:  recalls[1]  += 1
            if first_rank <= 5:  recalls[5]  += 1
            if first_rank <= 10: recalls[10] += 1
            mrr += 1.0 / first_rank

    metrics = {
        "R@1":  recalls[1]  / total,
        "R@5":  recalls[5]  / total,
        "R@10": recalls[10] / total,
        "MRR":  mrr / total
    }
    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, global_step)
    return metrics

# -----------------------
# Training (Phase A)
# -----------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"tb_{human_time()}"))

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map=None, attn_implementation="sdpa"
    ).to(device)

    # ---- Freeze all, unfreeze heads + ln + logit_scale ----
    for p in model.parameters():
        p.requires_grad = False

    for p in model.text_model.head.parameters():
        p.requires_grad = True

    # vision head: MultiheadAttention(out_proj) + MLP
    for name, module in model.vision_model.head.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            for p in module.out_proj.parameters():
                p.requires_grad = True
        if hasattr(module, "fc1"):
            for p in module.fc1.parameters():
                p.requires_grad = True
        if hasattr(module, "fc2"):
            for p in module.fc2.parameters():
                p.requires_grad = True

    # final norms（可选）
    for p in model.text_model.final_layer_norm.parameters():
        p.requires_grad = True
    for p in model.vision_model.post_layernorm.parameters():
        p.requires_grad = True

    # learnable logit_scale
    if not hasattr(model, "logit_scale"):
        model.logit_scale = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device), requires_grad=True)

    # dataset & loader
    train_set = RetrievalDataset(args.train_jsonl, args.train_image_root)
    train_set.is_train = True
    val_set = RetrievalDataset(args.val_jsonl, args.val_image_root)
    val_set.is_train = False

    collator = Collator(processor, max_length=args.max_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collator, drop_last=True
    )

    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum))
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))
    ce = nn.CrossEntropyLoss()

    global_step = 0
    best_R5 = -1.0
    best_path = os.path.join(args.output_dir, "best_model")

    # ---- Eval before training ----
    val_queries = val_set.samples  # 直接复用
    print("Running initial validation...")
    base_metrics = evaluate(model, processor, val_queries, args.val_image_root,
                            batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
    print("Initial metrics:", base_metrics)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (text_inputs, image_inputs) in enumerate(pbar, start=1):
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            with torch.autocast(device_type="cuda", dtype=(torch.float16 if dtype==torch.float16 else torch.bfloat16), enabled=True):
                txt = model.get_text_features(**text_inputs)
                img = model.get_image_features(**image_inputs)
                txt = txt / txt.norm(dim=-1, keepdim=True)
                img = img / img.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp().clamp(max=100)
                logits_txt = logit_scale * (txt @ img.t())
                logits_img = logit_scale * (img @ txt.t())

                targets = torch.arange(logits_txt.size(0), device=device)
                loss = (ce(logits_txt, targets) + ce(logits_img, targets)) / 2

            scaler.scale(loss / args.grad_accum).backward()
            running_loss += loss.item()

            if step % args.grad_accum == 0:
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # logging
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/logit_scale", float(model.logit_scale.detach().exp().clamp(max=100).item()), global_step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

                # mid-epoch eval
                if args.eval_every_steps > 0 and (global_step % args.eval_every_steps == 0):
                    model.eval()
                    metrics = evaluate(model, processor, val_queries, args.val_image_root,
                                       batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
                    model.train()

                    # save best
                    if metrics["R@5"] > best_R5:
                        best_R5 = metrics["R@5"]
                        model.save_pretrained(best_path)
                        processor.save_pretrained(best_path)

        # end-of-epoch eval
        model.eval()
        metrics = evaluate(model, processor, val_queries, args.val_image_root,
                           batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
        model.train()
        if metrics["R@5"] > best_R5:
            best_R5 = metrics["R@5"]
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)

        # save checkpoint each epoch
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

    print("Training finished. Best model at:", best_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/xteam/.cache/modelscope/hub/models/google/siglip2-so400m-patch14-224/")
    parser.add_argument("--train_jsonl", type=str,
                        default="/home/xteam/Calculus/siglip/Multimodal_Retrieval/MR_train_queries.jsonl")
    parser.add_argument("--train_image_root", type=str, default="/home/xteam/Calculus/siglip/trainimg")
    parser.add_argument("--val_jsonl", type=str,
                        default="/home/xteam/Calculus/siglip/Multimodal_Retrieval/MR_valid_queries.jsonl")
    parser.add_argument("--val_image_root", type=str, default="/home/xteam/Calculus/siglip/validimg")

    parser.add_argument("--output_dir", type=str, default="/home/xteam/Calculus/siglip/finetune_phaseA")
    parser.add_argument("--seed", type=int, default=2025)

    # 适配 12GB：小 batch + 累积
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)     # 有效等价 batch = 128
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--eval_every_steps", type=int, default=0)  # 0=only epoch end
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
