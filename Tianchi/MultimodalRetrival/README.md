## 引言
本技术文档详细描述了参与【天池经典打榜赛】赛道五-多模态图文检索赛的技术实现方案。电商图文检索任务要求模型根据自然语言形式的检索query，从给定的商品图片池中检索出相关图片，衡量模型多模态理解与匹配的能力。在实际的电商业务中，多模态检索扮演重要的角色，是电商场景满足用户需求、促成点击交易不可缺少的一环。本报告利用Siglip2作为基座模型进行微调，赛季1的分数为75.8793，rank19。推理环境12G3060。

## 赛题说明
本次任务使用的电商检索数据涵盖服装、家居、电子等多个领域，由商品图片和搜索query两部分构成，划分为训练集、验证集和测试集。对于商品图片，原图统一缩放为224*224大小，以base64编码格式表示。训练集、验证集和测试集分别提供一个商品图片集合，
对于搜索query，训练集和验证集中给出了query的id、文本内容及其对应的商品图片id（分别来自的训练集和验证集各自的商品集合）。测试集则仅给出query的id和文本，需要模型从测试集商品集合中预测相关商品。训练集query一般对应1-2个商品图片，验证集和测试集query平均对应6个商品图片。训练集、验证集和测试集之间query没有交集。

本次评测训练集包含25w搜索query-相关商品对，覆盖12.9w商品图片。验证集和测试集各自包含5k搜索query，要求模型从各自的3w商品图片候选池中进行检索。
数据集下载文件为：Multimodal_Retrieval.zip，包括：
- MR_train_imgs.tsv：训练集图片集合（base64编码）
- MR_train_queries.jsonl：训练集搜索query及对应商品id
- MR_valid_imgs.tsv：验证集图片集合（base64编码）
- MR_valid_queries.jsonl：验证集搜索query及对应商品id
- MR_test_imgs.tsv：测试集图片集合（base64编码）
- MR_test_queries.jsonl：测试集搜索query，需预测的文件，选手需要补充"item_ids"字段，为list类型。
- example_pred.jsonl：测试集提交结果示例
- README.txt: 说明文件

参考MSCOCO、Flickr30K等英文检索数据集，用Recall@1/5/10作为评测指标。该指标统计预测结果topk(k=1/5/10)中含有至少1个ground truth商品图片的query数量，除以测试集query总数，最终以Recall@1/5/10的平均值(MeanRecall)作为该任务主指标。

## 技术方案
本方案使用Siglip2作为基座模型，使用给定数据集进行微调。具体包括
1. 图像解码，将给定的图像base64编码进行解码
2. 模型定义以及训练。下载基座模型权重，使用Lora进行训练。

以下是环境依赖
```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
from natsort import natsorted
from tqdm import tqdm, trange
import json
import os, json, math, time, random, argparse
from typing import List, Dict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

from peft import LoraConfig, get_peft_model

```
以下是杂项、dataset和特征处理定义
```python
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def human_time():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


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
        root = args.train_image_root if self.is_train else args.val_image_root
        img_path = os.path.join(root, f"{pos_item_id}.png")
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
```
以下是encode图像以及测评代码
```python
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
    all_item_ids = set()
    for q in val_queries:
        for iid in q["item_ids"]:
            if os.path.exists(os.path.join(val_image_root, f"{iid}.png")):
                all_item_ids.add(iid)
    all_item_ids = sorted(list(all_item_ids))
    image_index, image_paths = build_image_index(
        model, processor, val_image_root, all_item_ids, batch_size=batch_size, device=device
    )
    path2id = {p: int(os.path.basename(p)[:-4]) for p in image_paths}

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

        sims = text_feats @ image_index.t()
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
```
以下是LoRA代码，本次LoRA仅对最后4层,以及head，norm和投影层进行微调
```python

def inject_lora_lastN(model, last_n_layers: int, r: int, alpha: int, dropout: float):
    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]
    )
    model = get_peft_model(model, lora_cfg)


    total_layers = 27  
    text_prefixes = [f"base_model.model.text_model.encoder.layers.{i}." for i in range(total_layers - last_n_layers, total_layers)]
    vision_prefixes = [f"base_model.model.vision_model.encoder.layers.{i}." for i in range(total_layers - last_n_layers, total_layers)]
    print(text_prefixes)
    print(vision_prefixes)
    def is_in_lastN(name: str):
        return any(name.startswith(p) for p in (text_prefixes + vision_prefixes))


    for name, module in model.named_modules():
        has_lora = any(hasattr(module, attr) for attr in ["lora_A", "lora_B"])
        if has_lora and (not is_in_lastN(name)):
            for p in module.parameters():
                p.requires_grad = False

    return model

```
以下是训练代码，包含模型定义，优化器定义，损失函数定义，超参数设置等
```python
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

    for p in model.parameters():
        p.requires_grad = False
    for p in model.text_model.head.parameters():
        p.requires_grad = True
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
    for p in model.text_model.final_layer_norm.parameters():
        p.requires_grad = True
    for p in model.vision_model.post_layernorm.parameters():
        p.requires_grad = True
    if not hasattr(model, "logit_scale"):
        model.logit_scale = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device), requires_grad=True)


    model = inject_lora_lastN(model,
                              last_n_layers=args.last_n_layers,
                              r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

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

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))
    ce = nn.CrossEntropyLoss()

    global_step = 0
    best_R5 = -1.0
    best_path = os.path.join(args.output_dir, "best_model")

    val_queries = val_set.samples
    print("Running initial validation...")
    base_metrics = evaluate(model, processor, val_queries, args.val_image_root,
                            batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
    print("Initial metrics:", base_metrics)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
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

            if step % args.grad_accum == 0:
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/logit_scale", float(model.logit_scale.detach().exp().clamp(max=100).item()), global_step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if args.eval_every_steps > 0 and (global_step % args.eval_every_steps == 0):
                    model.eval()
                    metrics = evaluate(model, processor, val_queries, args.val_image_root,
                                       batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
                    model.train()
                    if metrics["R@5"] > best_R5:
                        best_R5 = metrics["R@5"]
                        model.save_pretrained(best_path); processor.save_pretrained(best_path)

        # end-of-epoch eval & save
        model.eval()
        metrics = evaluate(model, processor, val_queries, args.val_image_root,
                           batch_size=args.eval_batch_size, device=device, writer=writer, global_step=global_step)
        model.train()
        if metrics["R@5"] > best_R5:
            best_R5 = metrics["R@5"]
            model.save_pretrained(best_path); processor.save_pretrained(best_path)

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir); processor.save_pretrained(epoch_dir)

    print("Training finished. Best model at:", best_path)

```
```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--train_jsonl", type=str)
    parser.add_argument("--train_image_root", type=str)
    parser.add_argument("--val_jsonl", type=str)
    parser.add_argument("--val_image_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int, default=2025)


    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--grad_accum", type=int, default=6)         
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)            # 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--last_n_layers", type=int, default=4)      
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--eval_every_steps", type=int, default=4000)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
```

## 推理代码
```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
from natsort import natsorted
from tqdm import tqdm, trange
import json

model_path = "/google/siglip2-so400m-patch14-224/"


model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    model_path
)


image_paths = natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
image_index2path = {i: path for i, path in enumerate(image_paths)}


queries = []
with open(jsonl_path, 'r') as f:
    for line in f:
        queries.append(json.loads(line))
        
candidate_labels = [query['query_text'] for query in queries]
query_ids = [query['query_id'] for query in queries]
print(candidate_labels[:10])
# candidate_labels = ["金丝绒木床头罩", "黑金戒指男"]

image_embeds_list = []
batch_size = 128 
with torch.no_grad():
    for i in trange(0, len(image_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
        image_inputs = processor(
            images=batch_imgs,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(model.device)
        batch_embeds = model.get_image_features(**image_inputs)
        batch_embeds = batch_embeds / batch_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds_list.append(batch_embeds)

image_embeds = torch.cat(image_embeds_list, dim=0)  


batch_size = 128
topk_indices = []
for i in trange(0, len(queries), batch_size):
    texts = [f"{label}" for label in candidate_labels[i:i+batch_size]]

    text_inputs = processor(
        text=texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)  # shape: [num_texts, emb_dim]
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)  # L2 norm

    similarity = torch.matmul(text_embeds, image_embeds.T)  # [num_texts, num_images]

    topk_values, sub_topk_indices = torch.topk(similarity, k=10, dim=1)
    topk_indices.append(sub_topk_indices)


topk_indices = torch.cat(topk_indices, dim=0)


results = []
for i, query in enumerate(tqdm(queries, desc="Generating results")):
    top_indices_for_query = topk_indices[i].tolist()
    predicted_item_ids = [int(os.path.basename(image_index2path[idx])[:-4]) for idx in top_indices_for_query]
    
    results.append({
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "item_ids": predicted_item_ids
    })

with open('/home/xteam/Calculus/siglip/Multimodal_Retrieval/submission.jsonl', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
```

---

最后在finetune_lora，10个epoch，学习率2e-4，微调最后8层获得比较好的效果，分数77.6312。上giant后OOM而且感觉不是很值当。感觉在中文方面多模态检索可能还是不如CN-CLIP，目前看到排行榜都是CN-CLIP进行微调，方案大差不差。下次比赛还是可以在baseline上改改。