from transformers import AutoModelForCausalLM, AutoTokenizer
# import os
import torch

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

local_path = "/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base"
# input_text = "登鹳雀楼->王之涣\n夜雨寄北->"
input_text = "case inr\na\u271d b a : Ordinal\nh : b \u2264 b\n\u22a2 a \u266f b \u2264 a \u266f b"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    trust_remote_code = True
)

inputs = tokenizer(input_text , return_tensors='pt')

model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)

print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))