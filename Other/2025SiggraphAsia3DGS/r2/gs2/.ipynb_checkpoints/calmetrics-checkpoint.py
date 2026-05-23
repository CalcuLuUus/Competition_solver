import os
import json

# 输入/输出路径自己改
txt_path = "log.txt"
json_path = "metrics.json"

result = {}

with open(txt_path, "r", encoding="utf-8") as f:
    current_name = None   # 用来做 JSON 的 key（文件名）
    current_time = None   # time 字段（这里用文件名）
    current_psnr = None

    for line in f:
        line = line.strip()

        # 解析 Scene 行
        if line.startswith("Scene:"):
            # 如果上一个场景已经拿到 PSNR，就先存进去
            if current_name is not None and current_psnr is not None:
                result[current_name] = {
                    "PSNR": current_psnr,
                    "time": current_time,
                }

            scene_path = line.split("Scene:", 1)[1].strip()
            filename = os.path.basename(scene_path)  # 取最后一段 1747834320424

            current_name = filename
            current_time = filename  # 如果你想用别的 time 含义，这里改
            current_psnr = None

        # 解析 PSNR 行
        elif line.startswith("PSNR"):
            # 例：'PSNR :   15.8707876'
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    current_psnr = float(parts[1].strip())
                except ValueError:
                    pass

    # 文件结束时，把最后一个场景存进去
    if current_name is not None and current_psnr is not None:
        result[current_name] = {
            "PSNR": current_psnr,
            "time": current_time,
        }

import re
results_path = "./results"
for x in result:
    result_time = os.path.join(results_path, x, "TRAIN_INFO")
    file_path = result_time 
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"Training Time:\s*([\d.]+)\s*seconds", text)
    if m:
        seconds = float(m.group(1))
        print("seconds:", seconds)
    else:
        print("未找到秒数")
    result[x]['time'] = seconds

# 写 JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
