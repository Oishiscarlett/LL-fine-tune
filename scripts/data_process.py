import json

# 转换为sharegpt格式
def convert_to_sharegpt(original_json_objects):
    target_dicts = []

    for original_dict in original_json_objects:
        # traced_tactics 非空
        if original_dict["traced_tactics"]:
            target_dict = {
                "conversations": []
            }

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


# 转换为alpaca格式
def convert_to_alpaca(original_json_objects):
    target_dicts = []

    for original_dict in original_json_objects:
        # traced_tactics 非空
        if original_dict["traced_tactics"]:
            target_dict = {
                "instruction": "",
                "input": "",
                "output": ""
            }

            for item in original_dict["traced_tactics"]:
                target_dict["instruction"] = item["state_before"]
                target_dict["output"] = item["tactic"]
            
            target_dicts.append(target_dict)

    return target_dicts


def main():
    # 读取原始JSON文件
    with open('data\\theorem\\random\\train.json', 'r') as file:
        original_json_objects = json.load(file)

    # 转换
    # target_dicts = convert_to_sharegpt(original_json_objects)
    target_dicts = convert_to_alpaca(original_json_objects)

    # 输出到新的JSON文件
    with open('train_processed.json', 'w') as file:
        json.dump(target_dicts, file, indent=4)

    print("转换完成，并保存到 'train_processed.json'")



if __name__ == "__main__":
    main()
