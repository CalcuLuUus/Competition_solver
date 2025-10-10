import os
from openai import OpenAI
from tqdm import tqdm
import json
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key='sk-37b14ac039af4503b26916c0eda25412',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


queries = []
with open('/Users/calculus/Competition_solver/Tianchi/MultimodalRetrival/MR_test_queries.jsonl', 'r') as f:
    for line in f:
        queries.append(json.loads(line))

# queries = queries[:10]      
candidate_labels = [query['query_text'] for query in queries]
query_ids = [query['query_id'] for query in queries]

for i, query in enumerate(tqdm(queries, desc="Translating queries")):
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "你现在是一个智能翻译官，我会给你中文的电商产品名称，你需要翻译成英文。请不要直接翻译，因为一些电商名称比较特殊，请思考后进行翻译。例如：黑金戒指男，翻译为：Black gold ring for men。你只需要给我英文翻译结果，不要给其他字符。"},
            {"role": "user", "content": candidate_labels[i]},
        ]
    )
    # for chunk in completion:
    #     print(chunk.choices[0].delta.content, end="", flush=True)
        
    result = completion.choices[0].message.content
    # print(result)
    
    new_item = {
        "query_id": query_ids[i],
        "query_text": result,
    }
    with open('./MR_test_queries_cn.jsonl', 'a') as f:
        f.write(json.dumps(new_item) + '\n')