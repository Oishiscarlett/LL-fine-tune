import os,re,sys
import torch 
import torch.nn as nn 
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,AutoModelForCausalLM,get_scheduler,BitsAndBytesConfig,OPTForCausalLM,LlamaModel
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from trl import AutoModelForCausalLMWithValueHead
from torch.optim import AdamW
from peft.tuners.lora import LoraLayer 
# import deepspeed 

MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "opt": (AutoConfig, AutoTokenizer, OPTForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}


# oishi:这里是打印lora训练了多少参数的
def print_trainable_params(model) -> None:
    # Adopted from https://github.com/LLaMA-Efficient-Tuning-main/src/utils/other.py
    trainable_params, all_param = 0, 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))

"""oishi
actor 使用sft model初始化
critic 使用rm model初始化
"""
class PPOEngine():
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args

        self.config_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if self.model_args.load_in_4bit:
            self.config_kwargs["load_in_4bit"] = True
            self.config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        
        if model_args.sft_model_path is None:
            raise ValueError(
                f"you need to provide sft model path, but now model path is {model_args.sft_model_path}."
            )
        else:
            self.actor_model = self._init_actor()
            self.sft_model = None 

        if model_args.reward_model_path is None and model_args.reward_lora_path is None:
            raise ValueError(
                f"you need to provide reward model path, but now model path is {model_args.reward_model_path}."
            )
        else:
            self.critic_model = self._init_critic()
            self.reward_model = None


    # oishi：初始化actor模型
    def _init_actor(self):
        config_class, tokenizer_class, model_class = MODEL_CLASSES[self.model_args.model_type]
        
        """oishi
        加载模型
        line2 判断模型是否为TensorFlow格式的（感觉可以不用写）
        line3 这里**操作符表示将self.config_kwargs这个字典中的所有项作为关键字参数传递给from_pretrained方法。
        """
        model = model_class.from_pretrained(
            self.model_args.sft_model_path,
            from_tf=bool(".ckpt" in self.model_args.sft_model_path),
            **self.config_kwargs
        )
        
        # oishi：判断lora适配器是否已经训练好；训练好就直接加载，否则配置lora参数，训练一个新的
        if self.model_args.actor_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.model_args.actor_peft_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=self.training_args.actor_lora_target.split(','),
                r=self.training_args.actor_lora_rank,
                lora_alpha=self.training_args.actor_lora_alpha,
                lora_dropout=self.training_args.actor_lora_dropout,
                modules_to_save=self.training_args.actor_modules_to_save.split(',') if self.training_args.actor_modules_to_save is not None else None
            )

            model = get_peft_model(model, peft_config=lora_config)
        print('*********************actor*******************')
        print_trainable_params(model)
        
        return model 

    # oishi：初始化critic模型
    def _init_critic(self):
        
        confppoig_class, tokenizer_class, model_class = MODEL_CLASSES[self.model_args.model_type]
        # oishi：同上加载模型
        model = model_class.from_pretrained(
            self.model_args.sft_model_path,
            from_tf=bool(".ckpt" in self.model_args.sft_model_path),
            **self.config_kwargs
        )
        # oishi：加载RM的lora适配器
        model = PeftModel.from_pretrained(model, self.model_args.reward_lora_path)
        model = model.merge_and_unload()

        # oishi：判断，critic model是否已经有现成的lora，如果有就加载，否则训练一个
        if self.model_args.critic_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.model_args.critic_peft_path, adapter_name="default", is_trainable=True)

        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=self.training_args.critic_lora_target.split(','),
                r=self.training_args.critic_lora_rank,
                lora_alpha=self.training_args.critic_lora_alpha,
                lora_dropout=self.training_args.critic_lora_dropout,
            )

            model = get_peft_model(model, peft_config=lora_config)


        ## add value head 
        # oishi：这一行就是为model添加了新的一层网络：value head，但是这个参数还是初始参数
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        # load v_head 
        # oishi: 这里加载训练好的lora参数
        reward_model_state_dict = torch.load(os.path.join(self.model_args.reward_lora_path, 'adapter_model.bin'))
        
        # oishi：这个步骤就是选择使用rm的lora参数来更新value head还是用预先准备好的参数来更新。
        if self.model_args.critic_peft_path is not None:
            lora_state_dict = torch.load(os.path.join(self.model_args.critic_peft_path, 'adapter_model.bin'))
            model.v_head.load_state_dict({
                    "summary.weight": lora_state_dict["v_head.summary.weight"],
                    "summary.bias": lora_state_dict["v_head.summary.bias"]
                })
        else:
            model.v_head.load_state_dict({
                    "summary.weight": reward_model_state_dict["v_head.summary.weight"],
                    "summary.bias": reward_model_state_dict["v_head.summary.bias"]
                })
            
        """oishi
        注册缓冲区，缓冲区的数据会被加载，但是不会被更新；
        注册成缓冲区之后，该缓冲区就成为了model的一个属性，可以使用getattr来访问
        """
        model.register_buffer("reward_head_weight", reward_model_state_dict["v_head.summary.weight"])
        model.register_buffer("reward_head_bias", reward_model_state_dict["v_head.summary.bias"])
        model.register_buffer("critic_head_weight", reward_model_state_dict["v_head.summary.weight"])
        model.register_buffer("critic_head_bias", reward_model_state_dict["v_head.summary.bias"])

        print('*********************critic*******************')
        print_trainable_params(model)

        return model 

# oishi：这个部分是actor和critic共用一个model
class PPOEngine_CO():
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args

        self.config_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if self.model_args.load_in_4bit:
            self.config_kwargs["load_in_4bit"] = True
            self.config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.model = self.create_model()

    
    def create_model(self):
        config_class, tokenizer_class, model_class = MODEL_CLASSES[self.model_args.model_type]
        model = model_class.from_pretrained(
                pretrained_model_name_or_path=self.model_args.sft_model_path,
                from_tf=bool(".ckpt" in self.model_args.sft_model_path),
                **self.config_kwargs
            )
        model = PeftModel.from_pretrained(model, self.model_args.reward_lora_path)
        model = model.merge_and_unload()

        if self.model_args.peft_path is not None:
            model = PeftModel.from_pretrained(model, self.model_args.peft_path, is_trainable=True)
            
            if self.training_args.use_multi_adapters:
                model.load_adapter(self.model_args.critic_peft_path, adapter_name="critic", is_trainable=True)
                model.set_adapter("default")
                    
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=self.training_args.lora_target.split(','),
                r=self.training_args.lora_rank,
                lora_alpha=self.training_args.lora_alpha,
                lora_dropout=self.training_args.lora_dropout,
                modules_to_save=self.training_args.modules_to_save.split(',') if self.training_args.modules_to_save is not None else None
            )

            model = get_peft_model(model, peft_config=lora_config)
            
            if self.training_args.use_multi_adapters:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=self.training_args.critic_lora_target.split(','),
                    r=self.training_args.critic_lora_rank,
                    lora_alpha=self.training_args.critic_lora_alpha,
                    lora_dropout=self.training_args.critic_lora_dropout,
                    modules_to_save=self.training_args.critic_modules_to_save.split(',') if self.training_args.critic_modules_to_save is not None else None
                )
                ## add critic lora
                model.add_adapter(adapter_name="critic", peft_config=lora_config)
                # Set adapter back to default.
                model.set_adapter("default")

        ## add value head 
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        # load v_head 
        reward_model_state_dict = torch.load(os.path.join(self.model_args.reward_lora_path, 'adapter_model.bin'))
        
        if self.model_args.peft_path is not None:
            lora_state_dict = torch.load(os.path.join(self.model_args.peft_path, 'adapter_model.bin'))
            model.v_head.load_state_dict({
                    "summary.weight": lora_state_dict["v_head.summary.weight"],
                    "summary.bias": lora_state_dict["v_head.summary.bias"]
                })
        else:
            model.v_head.load_state_dict({
                    "summary.weight": reward_model_state_dict["v_head.summary.weight"],
                    "summary.bias": reward_model_state_dict["v_head.summary.bias"]
                })
            
        model.register_buffer("reward_head_weight", reward_model_state_dict["v_head.summary.weight"])
        model.register_buffer("reward_head_bias", reward_model_state_dict["v_head.summary.bias"])
        model.register_buffer("critic_head_weight", reward_model_state_dict["v_head.summary.weight"])
        model.register_buffer("critic_head_bias", reward_model_state_dict["v_head.summary.bias"])

        print('*********************model*******************')
        print_trainable_params(model)
        
        return model 
  
