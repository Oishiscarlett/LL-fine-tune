import subprocess

hostfile=""
model_path="/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base"
# data_path="/data/home/zfyang/home/oishi/Baichuan2/fine-tune/data/minif2f/val_new.json"
# output_dir="/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base-ft-v1"
data_path="/data/home/zfyang/home/oishi/Baichuan2/fine-tune/data/theorem_v1-1/random/train_processed.json"
# output_dir="/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base-ft-v1-1"
output_dir="/data/home/zfyang/home/oishi/model/Baichuan2-7B-Base-ft-v1-2"
ds_path="/data/home/zfyang/home/oishi/Baichuan2/fine-tune/ds_config.json"

def run_fine_tune():
    command = [
        "deepspeed", "--hostfile", hostfile, "fine-tune/fine-tune.py",
        "--report_to", "none",
        "--data_path", data_path,
        "--model_name_or_path", model_path,
        "--output_dir", output_dir,
        "--model_max_length", "512",
        "--num_train_epochs", "4",
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "8",
        "--save_strategy", "epoch",
        "--learning_rate", "2e-5",
        "--lr_scheduler_type", "cosine",
        "--adam_beta1", "0.9",
        "--adam_beta2", "0.98",
        "--adam_epsilon", "1e-8",
        "--max_grad_norm", "0.5",
        "--weight_decay", "1e-4",
        "--warmup_ratio", "0.0",
        "--logging_steps", "10",
        "--gradient_checkpointing", "True",
        "--deepspeed", ds_path,
        "--bf16", "True",
        "--tf32", "True",
        "--use_lora", "True"
    ]
    subprocess.run(command)

# 使用函数
run_fine_tune()
