import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
from natsort import natsorted
from tqdm import tqdm, trange
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default="/mnt/sda/calculus/weight/siglip_giant")
args.add_argument("--valid", type=bool, default=False)
args.add_argument("--output_path", type=str, default="/home/xteam/Calculus/siglip/Multimodal_Retrieval/submission.jsonl")
args = args.parse_args()

model_path = args.model_path
#  "/home/xteam/.cache/modelscope/hub/models/google/siglip2-so400m-patch14-224/"

Valid = True
if Valid:
    image_folder = "/home/xteam/Calculus/siglip/validimg"
    jsonl_path = "/home/xteam/Calculus/siglip/Multimodal_Retrieval/MR_valid_queries.jsonl"
else:
    image_folder = "/home/xteam/Calculus/siglip/imgdata"
    jsonl_path = "/home/xteam/Calculus/siglip/Multimodal_Retrieval/MR_test_queries.jsonl"

# 载入模型与处理器
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    model_path
)

# 图片目录
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

# 批量编码图片向量
image_embeds_list = []
batch_size = 128  # 可根据显存调整
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

# 合并所有图片向量
image_embeds = torch.cat(image_embeds_list, dim=0)  # shape: [num_images, emb_dim]
# # 保存图片特征向量到本地文件
# torch.save(image_embeds, '/home/xteam/Calculus/siglip/image_features_tensor.pt')
# print("图片特征向量已保存到 /home/xteam/Calculus/siglip/image_features_tensor.pt")

# image_embeds = torch.load('/home/xteam/Calculus/siglip/image_features_tensor.pt')


batch_size = 128
topk_indices = []
for i in trange(0, len(queries), batch_size):
    texts = [f"{label}" for label in candidate_labels[i:i+batch_size]]

    # 编码文本向量（只需要运行一次）
    text_inputs = processor(
        text=texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)  # shape: [num_texts, emb_dim]
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)  # L2 norm

    # 计算相似度（cosine similarity）
    similarity = torch.matmul(text_embeds, image_embeds.T)  # [num_texts, num_images]

    # 取 Top-K
    topk_values, sub_topk_indices = torch.topk(similarity, k=10, dim=1)
    topk_indices.append(sub_topk_indices)


topk_indices = torch.cat(topk_indices, dim=0)


results = []
for i, query in enumerate(tqdm(queries, desc="Generating results")):
    # 获取top-10图片的索引
    top_indices_for_query = topk_indices[i].tolist()
    # 根据索引找到对应的item_id
    predicted_item_ids = [int(os.path.basename(image_index2path[idx])[:-4]) for idx in top_indices_for_query]
    
    results.append({
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "item_ids": predicted_item_ids
    })

# 保存为jsonl格式
with open(args.output_path, 'w', encoding='utf-8') as f:
    for result in results:
        # 确保 query_text 是中文字符串并以 utf-8 编码写入
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        