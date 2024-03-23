import json
import random
import os

def generate_unique_id(existing_ids, max_value=99999):
    """生成一个在现有IDs中唯一的随机数ID"""
    while True:
        random_id = str(random.randint(1, max_value))
        if random_id not in existing_ids:
            return random_id

def convert_original_json_to_target(original_json_objects):
    target_dicts = []
    generated_ids = set()

    for original_dict in original_json_objects:
        if original_dict["traced_tactics"]:
            target_dict = {
                "id": generate_unique_id(generated_ids),
                "conversations": []
            }
            generated_ids.add(target_dict["id"])

            for item in original_dict["traced_tactics"]:
                target_dict["conversations"].append({
                    "from": "human",
                    "value": item["state_before"]
                })
                target_dict["conversations"].append({
                    "from": "gpt",
                    "value": item["tactic"]
                })
            
            target_dicts.append(target_dict)

    return target_dicts

# 读取原始JSON文件
with open('fine-tune\data\minif2f\\val.json', 'r') as file:
    original_json_objects = json.load(file)

# 转换
target_dicts = convert_original_json_to_target(original_json_objects)

# 输出到新的JSON文件
with open('target_json_file.json', 'w') as file:
    json.dump(target_dicts, file, indent=4)

print("转换完成，并保存到 'target_json_file.json'")
