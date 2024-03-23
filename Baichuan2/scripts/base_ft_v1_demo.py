from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
# import os
import torch

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

local_path = "/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base-ft-v1"
# input_text = "case inr\na\u271d b a : Ordinal\nh : b \u2264 b\n\u22a2 a \u266f b \u2264 a \u266f b"
input_text = "\u03b1 : Type u\nm n o : \u2115\nm' : Type ?u.14351\nn' : Type ?u.14354\no' : Type ?u.14357\nx : \u03b1\nu : Fin m \u2192 \u03b1\ni : Fin m\n\u22a2 vecCons x u (Fin.succ i) = u i"
# input_text = "登鹳雀楼->王之涣\n夜雨寄北->"


# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code = True)
model = AutoPeftModelForCausalLM.from_pretrained(
    local_path,
    trust_remote_code = True
)

inputs = tokenizer(input_text , return_tensors='pt')


model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)

print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))