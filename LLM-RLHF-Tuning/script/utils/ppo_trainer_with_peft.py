import re
from typing import List, Tuple
import torch, os, sys, math, logging, random, time, warnings, shutil, copy
from tqdm import tqdm

sys.path.append("..")
from transformers import Trainer,get_scheduler,PreTrainedModel,GenerationConfig
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from pathlib import Path
from peft import get_peft_model,get_peft_model_state_dict,PeftModel
import torch.nn as nn 
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PPOModel(nn.Module):
    # see from: https://github.com/huggingface/accelerate/issues/668
    def __init__(self, actor_model, critic_model):
        super().__init__()
        self.actor_model = actor_model 
        self.critic_model = critic_model 
    

    """oishi
    在强化学习中，extra_inputs可以用于多种目的，具体取决于训练的具体需求和实现的细节。这里有一些extra_inputs可能代表的情况：

    1. 辅助任务输入：extra_inputs可能用于一些辅助任务，这些任务旨在提升模型的泛化能力或提供额外的学习信号。
        例如，在自然语言处理任务中，除了主要的文本生成任务外，还可以有一个辅助的分类任务，如情感分析，用于提高模型对文本情感的敏感性。

    2. 探索增强输入：在强化学习中，探索（Exploration）是一个关键的概念，它鼓励模型尝试新的或少见的动作以发现可能获得更高奖励的策略。
        extra_inputs可以用于引入额外的信息或噪声，以鼓励模型探索状态空间的不同部分。

    3. 正则化项：extra_inputs可能包含用于计算正则化项的信息，正则化项可以添加到总损失中以防止过拟合。
        例如，这些输入可以用于计算一个正则化项，如权重衰减或者更复杂的结构化稀疏性正则化。

    4. 条件信息：在条件语言模型中，extra_inputs可以携带用于指导生成过程的额外信息，
        比如特定的风格指示、要求模型遵循的规则或其他上下文信息。

    5. 奖励或惩罚信号：在某些实现中，extra_inputs可以用于直接提供额外的奖励或惩罚信号，
        这些信号可以用于调整或增强模型基于主要任务接收的奖励。

    6. 对抗性训练输入：在对抗性训练场景中，extra_inputs可能包含生成的对抗样本或扰动，
        这些样本或扰动用于提高模型的鲁棒性和泛化能力。
    """


    def forward(self, sequences, extra_inputs=None):
        """oishi
        actor 输入序列，输出对应的动作对数概率
        critic 输入序列，输出状态的价值估计
        extra_inputs 这里应该指的是openai的RLHF方法中，为了缓解模型性能回归，
        在ppo过程中添加了预训练中使用的数据
        """
        actor_logits = self.actor_model(**sequences, return_dict=True).logits
        critic_values = self.critic_model(**sequences)[-1]
        if extra_inputs is not None:
            extra_loss = self.actor_model(**extra_inputs, return_dict=True).loss
        else:
            extra_loss = 0.0  
        return actor_logits, critic_values, extra_loss
    
    

class PPOPeftTrainer(Trainer):
    def __init__(
        self, 
        args = None, 
        ppo_engine = None, 
        data_collator = None,
        train_dataset = None,
        tokenizer = None,
        extra_train_dataset = None,
        extra_data_collator = None, 
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs
    ):
        self.args = args 
        if args.use_co_model:
            self.co_model = ppo_engine.model 
        else:
            self.actor_model = ppo_engine.actor_model 
            self.critic_model = ppo_engine.critic_model
            self.model = PPOModel(self.actor_model, self.critic_model)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision='fp16' if self.args.fp16 else None,
            log_with=self.args.report_to,
            
        )
        self.accelerator.init_trackers(
            project_name="ppo_train",
            config=self.args 
        )
        
        self.dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=self.args.per_device_train_batch_size,
                                    collate_fn=data_collator,
                                    num_workers=self.args.dataloader_num_workers,
                                    shuffle=True,
                                    )
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        if extra_train_dataset is not None:
            self.extra_train_dataloader = DataLoader(
                extra_train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=extra_data_collator,
                num_workers=self.args.dataloader_num_workers,
                shuffle=True,
            )
            self.extra_train_dataloader = self.accelerator.prepare(self.extra_train_dataloader)
        else:
            self.extra_train_dataloader = None 
        
        self.tokenizer = tokenizer
        
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.device = self.accelerator.device

        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        if self.is_deepspeed_enabled:
            
            if not self.args.use_co_model:
                raise PermissionError(
                    "if you use deepspeed_plugin, you need to provide one model."
                )
                
            if getattr(self.args, "hf_deepspeed_config", None) is None:
                from transformers.deepspeed import HfTrainerDeepSpeedConfig
                ds_plugin = self.accelerator.state.deepspeed_plugin

                ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
                ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
                ds_plugin.hf_ds_config.trainer_config_process(self.args)


        ## get max_update_steps for lr_scheduler 
        if self.extra_train_dataloader is None:
            self.max_dataloader_iters = len(self.dataloader)
        else:
            self.max_dataloader_iters = min(len(self.dataloader), len(self.extra_train_dataloader))
        self.num_update_steps_per_epoch, self.max_update_steps = self.get_max_update_steps(args, self.max_dataloader_iters)
        
        
        ## create optimizer and scheduler 
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = self.create_optimizer()
        
        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(self.optimizer, max_update_steps=self.max_update_steps)

        if self.args.use_co_model:
            self.co_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.co_model, self.optimizer, self.lr_scheduler)
        else:
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
            self.ator_model = self.accelerator.unwrap_model(self.model).actor_model 
            self.critic_model = self.accelerator.unwrap_model(self.model).critic_model
            

        
    def get_max_update_steps(self, args, dataloader_nums):
        num_update_steps_per_epoch = dataloader_nums * (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * args.ppo_epochs / args.gradient_accumulation_steps  
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        if args.max_steps > 0:
            max_update_steps = args.max_steps
        else:
            max_update_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        return num_update_steps_per_epoch, max_update_steps
        
        
    def get_parms(self, model, lr, weight_decay, eps=1e-8):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": lr,
                "eps": eps,
            }
        ]
        return params
    
    # oishi：创建一个优化器
    def create_optimizer(self):
        if self.args.use_co_model:
            params = self.get_parms(self.co_model, self.args.learning_rate, self.args.weight_decay)
        else:
            params = self.get_parms(self.actor_model, self.args.actor_lr, self.args.actor_weight_decay)
            params.extend(self.get_parms(self.critic_model, self.args.critic_lr, self.args.critic_weight_decay))

        optimizer = AdamW(params, betas=(0.9,0.95))
        
        return optimizer
    
    # oishi：创建一个动态学习率
    def create_scheduler(self, optimizer, max_update_steps):
        lr_scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=max_update_steps)
        return lr_scheduler

    """oishi
    掩码和使用-100来忽略某些目标：
    1. 掩码通常是和数据形状相同的张量，其中的数据是0和1
        0对应忽略数据，1对应有效数据；
    2. 掩码使用更加灵活，可以在各种计算中使用，如平均值、方差计算
    3. -100大多用于监督学习，在计算损失的时候，标记为-100的数据通常会被忽略
    
    这里3个与掩码相关的函数分别是在计算有效值的平均值、有效值的方差以及对有效数据进行白化

    白化：即将数据标准化为具有零均值和单位方差的形式

    dim是维度：
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    如果我们指定dim=0（有时也称为轴0），这意味着操作会沿着数组的第一个维度进行，即沿着行的方向。
    在上面的矩阵中，如果我们要计算每一列的和，我们会沿着dim=0方向进行，结果会是[12, 15, 18]（即1+4+7, 2+5+8, 3+6+9）。
    如果我们指定dim=1（有时也称为轴1），这意味着操作会沿着数组的第二个维度进行，即沿着列的方向。
    在同一个矩阵中，如果我们要计算每一行的和，我们会沿着dim=1方向进行，结果会是[6, 15, 24]（即1+2+3, 4+5+6, 7+8+9）。

    """
    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps) 
    
    def masked_var(self, data, mask, dim=None):
        mean = self.masked_mean(data, mask, dim=dim)
        centered_values = data - mean
        var = self.masked_mean(centered_values**2, mask, dim=dim)
        return var


    def masked_whiten(self, data, mask, dim=None, shift_mean=True):
        mean = data.sum() / mask.sum()
        var = torch.sum(((data - mean) ** 2).mul(mask)) / mask.sum()

        whitened = (data - mean) * torch.rsqrt(var + 1e-6)
        if not shift_mean:
            whitened += mean
        return whitened

    """oishi
    这个unwrap_model函数的作用是递归地将模型从其可能的容器中解包

    在分布式训练中，模型可能被装在类似torch.nn.DataParallel、
    torch.nn.parallel.DistributedDataParallel的容器中，来支持GPU或其他分布式设置，
    此时访问模型需要使用model.module

    unwrap_model函数的作用就是将模型从容器中解包。它会检查传入的模型是否有module属性，
    如果有就递归调用，直到没有（找到初始模型）。
    """
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).

        Args:
            model (`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    @torch.no_grad()
    def generate(
        self,
        prompts_ids,
        return_prompt: bool = True,
    ):

        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": self.args.max_response_length,
            "min_new_tokens": self.args.min_response_length, 
            "_from_model_config": False
        }
        
        if self.args.use_co_model:
            unwrapped_model = self.accelerator.unwrap_model(self.co_model)
            
            if self.unwrap_model(unwrapped_model) is not unwrapped_model:
                unwrapped_model = self.unwrap_model(unwrapped_model)

            if isinstance(unwrapped_model, AutoModelForCausalLMWithValueHead):
                unwrapped_model = unwrapped_model.pretrained_model
                
            if not hasattr(unwrapped_model, "generation_config"):
                raise AttributeError(
                    f" model object [{unwrapped_model.__class__.__name__}] has no attribute [generation_config] "
                )
                    
            if unwrapped_model.generation_config._from_model_config:
                unwrapped_model.generation_config._from_model_config = False
            sequences = unwrapped_model.generate(inputs=prompts_ids, **gen_kwargs)
        else:
            if self.actor_model.generation_config._from_model_config:   
                self.actor_model.generation_config._from_model_config = False

            sequences = self.actor_model.generate(inputs=prompts_ids, **gen_kwargs)
            
            
        if not return_prompt:
            return sequences[:, prompts_ids.shape[1] :]
        
        return sequences

    """oishi
    加工处理一批（batch）seq

    去除prompts responses中的padding

    """
    def process_sequences(self, prompts_ids, responses_ids):
        # seq: [0 0 0 0, prompt, response, 0 0 0 0] change to [prompt, response, 0 0 0 0]
        
        prompts_without_padding, responses_without_padding = [], []
        batch_size = prompts_ids.shape[0]
        for i in range(batch_size):
            response = responses_ids[i]
            prompt = prompts_ids[i] 
            prompt_left_padding_length = (prompt == self.tokenizer.pad_token_id).sum().item() #oishi：sum的结果是一个张量，item用于从单元素的张量中提取数值，转换成py数据类型
            response_length = (response != self.tokenizer.pad_token_id).sum().item()
            prompt_without_padding = prompt[prompt_left_padding_length:]
            response_without_padding = response[:response_length]
            
            prompts_without_padding.append(prompt_without_padding.to(self.device))
            responses_without_padding.append(response_without_padding.to(self.device))
        
        
        new_sequences = [torch.cat([q, r]) for q, r in zip(prompts_without_padding, responses_without_padding)]
        sequences = torch.nn.utils.rnn.pad_sequence(
            new_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        sequences = dict(
            input_ids=sequences.to(self.device),
            attention_mask=sequences.ne(self.tokenizer.pad_token_id).long().to(self.device)
        )
        
        return prompts_without_padding, responses_without_padding, sequences
      
    
    def get_last_reward_score(self, values, responses_mask):
        
        batch_size = values.shape[0]
        reward_score = []
        for i in range(batch_size):
            value = values[i]
            #oishi: nonzero() 返回的是非零元素的索引，[-1]取最后一个，detach()从图中分离出该张量（产生一个副本），不影响后续
            end_index = responses_mask[i].nonzero()[-1].detach().item()
            reward_score.append(value[end_index])
        
        rewards_score = torch.stack(reward_score)
        
        return rewards_score
    
    
    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)  
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        return log_probs_labels.squeeze(-1)


    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy 
        
    
    def compute_rewards_with_kl_penalty(self, ref_values, actor_log_probs, ref_log_probs, responses_mask):
        masks = responses_mask[:, 1:] 
        rewards_score = self.get_last_reward_score(ref_values, responses_mask)
        
        batch_size = rewards_score.shape[0]
        rewards_with_kl_penalty, kl_penalty_all = [], []
        for i in range(batch_size):
            mask = masks[i]
            
            kl = actor_log_probs[i] - ref_log_probs[i]
            if self.args.kl_penalty_method == 'abs':
                kl = torch.abs(kl)
            elif self.args.kl_penalty_method == 'mse':
                kl = kl ** 2 * 0.5 
                
            kl_penalty = - self.args.kl_penalty_beta * kl 
            kl_penalty_all.append(kl_penalty)

            if self.args.reward_score_clip is not None:
                rewards_score[i] = torch.clamp(rewards_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)
            
            end_index = mask.nonzero()[-1].detach().item()
            kl_penalty[end_index] += rewards_score[i]

            rewards_with_kl_penalty.append(kl_penalty)
        return torch.stack(rewards_with_kl_penalty), torch.stack(kl_penalty_all), rewards_score 
    
    
    def get_advantages_and_returns(self, values, rewards, responses_mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
        masks = responses_mask[:, 1:] 
        
        lastgaelam = 0 
        advantages_reversed = []
        length = rewards.size()[-1]

        for t in reversed(range(length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]  
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam        
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values     
        
        if self.args.use_advantage_norm:
            advantages = self.masked_whiten(advantages, masks)
            
        return advantages.detach(), returns


    def get_responses_mask(self, sequences_mask, prompts_without_padding):
        batch_size = sequences_mask.shape[0]
        responses_mask = []
        for i in range(batch_size):
            prompt = prompts_without_padding[i]
            response_mask = torch.zeros_like(sequences_mask[i]) #创建相同形状的全0张量
            response_mask[len(prompt):] = sequences_mask[i][len(prompt):] # 提示部分保留为0，其他用sequences_mask赋值
            responses_mask.append(response_mask)
        return torch.stack(responses_mask)



    @torch.no_grad()
    def get_co_model_output(self, sequences):
        unwrap_model = self.accelerator.unwrap_model(self.co_model)
        actor_logits, _, critic_values = unwrap_model(**sequences, return_dict=True)
 
        if self.args.use_multi_adapters:
            unwrap_model.pretrained_model.set_adapter("critic")
            critic_values = unwrap_model(**sequences, return_dict=True)[-1]
            unwrap_model.pretrained_model.set_adapter("default")

        with unwrap_model.pretrained_model.disable_adapter():
            ## the same as sft model 
            ref_logits, _, _ = unwrap_model(**sequences, return_dict=True)
            
            ## the same as reward model 
            ## save current critic model v_head 
            v_head_stat_dict = unwrap_model.v_head.state_dict()
            setattr(unwrap_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(unwrap_model, "critic_head_bias", v_head_stat_dict["summary.bias"])
            ## change to reward model v_head
            unwrap_model.v_head.load_state_dict({"summary.weight": getattr(unwrap_model, "reward_head_weight"), "summary.bias": getattr(unwrap_model, "reward_head_bias")})

            ref_values = unwrap_model(**sequences)[-1]
            ## back to critic model v_head 
            unwrap_model.v_head.load_state_dict({"summary.weight": getattr(unwrap_model, "critic_head_weight"), "summary.bias": getattr(unwrap_model, "critic_head_bias")})
        
        return actor_logits, critic_values, ref_logits, ref_values
        
        
        
    @torch.no_grad()
    def get_model_output(self, sequences):
        
        actor_logits, critic_values, _ = self.model(sequences)  
        with self.actor_model.disable_adapter():
            ## the same as sft model 
            ref_logits = self.actor_model(**sequences, return_dict=True).logits
            
        with self.critic_model.pretrained_model.disable_adapter():
            ## the same as reward model 
            ## save current critic model v_head 
            v_head_stat_dict = self.critic_model.v_head.state_dict()
            setattr(self.critic_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(self.critic_model, "critic_head_bias", v_head_stat_dict["summary.bias"])
            ## change to reward model v_head
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "reward_head_weight"), "summary.bias": getattr(self.critic_model, "reward_head_bias")})

            ref_values = self.critic_model(**sequences)[-1]
            ## back to critic model v_head 
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "critic_head_weight"), "summary.bias": getattr(self.critic_model, "critic_head_bias")})
        
        return actor_logits, critic_values, ref_logits, ref_values
        
        
    def get_experience_data(self, prompts_ids):

        responses_ids = self.generate(prompts_ids, return_prompt=False)
        prompts_without_padding, responses_without_padding, sequences = self.process_sequences(prompts_ids, responses_ids)
        
        ### 不同进程填充      
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            sequences["input_ids"] = self.accelerator.pad_across_processes(
                sequences["input_ids"], dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=pad_first
            )
            sequences["attention_mask"] = self.accelerator.pad_across_processes(
                sequences["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
        
        if self.args.use_co_model:
            actor_logits, critic_values, ref_logits, ref_values = self.get_co_model_output(sequences)
        else:
            actor_logits, critic_values, ref_logits, ref_values = self.get_model_output(sequences)
        
        """oishi
        优化问题中，最大化对数概率，最小化损失（所以有负号）
        """
        actor_log_probs = self.get_log_probs(actor_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        actor_ce_loss = -self.masked_mean(actor_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        ref_log_probs = self.get_log_probs(ref_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        ref_ce_loss = -self.masked_mean(ref_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)
        
        rewards_with_kl_penalty, kl_penalty, rewards_score = self.compute_rewards_with_kl_penalty(ref_values, actor_log_probs, ref_log_probs, responses_mask)

        critic_values = critic_values[:, :-1] * responses_mask[:, 1:] 
        rewards_with_kl_penalty = rewards_with_kl_penalty * responses_mask[:, 1:]  
        advantages, returns = self.get_advantages_and_returns(critic_values, rewards_with_kl_penalty, responses_mask)

        return dict(
            prompts_ids=prompts_without_padding,
            responses_ids=responses_without_padding,
            responses_mask=responses_mask,
            sequences_ids=sequences["input_ids"],
            sequences_mask=sequences["attention_mask"],
            actor_log_probs=actor_log_probs,
            ref_log_probs=ref_log_probs,
            rewards_with_kl_penalty=rewards_with_kl_penalty,
            rewards_score=rewards_score,
            kl_penalty=kl_penalty,
            critic_values=critic_values,
            advantages=advantages,
            returns=returns,
            actor_ce_loss=actor_ce_loss,
            ref_ce_loss=ref_ce_loss,
        )


    def get_mini_dataset(self, data_buffer):

        mini_dataset = []
        batch_size = data_buffer[0]["exp"]["sequences_ids"].shape[0]
        for item in data_buffer:
            experience_data, batch_extra_data = item['exp'], item['extra']
            index = 0 
            while index < batch_size:
                dic = {}
                for k, v in experience_data.items():
                    if k in ["prompts_ids", "responses_ids"]:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size]
                    else:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                        
                if batch_extra_data is not None:
                    for k, v in batch_extra_data.items():
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                
                mini_dataset.append(dic)
                index += self.args.per_device_mini_train_batch_size
 
        return mini_dataset 
        
        
    def actor_loss(self, actor_log_probs, mini_batch_actor_log_probs, advantages, mask):
        
        # oishi: 重要性采样 因为输入的是对数概率，所以还要进行指数计算
        ratio = torch.exp((mini_batch_actor_log_probs - actor_log_probs) * mask) 
        loss1 = -advantages * ratio
        loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.ratio_clip,
                                             1.0 + self.args.ratio_clip)

        loss = self.masked_mean(torch.max(loss1, loss2), mask)
        return loss, ratio 


    def critic_loss(self, critic_values, mini_batch_critic_values, returns, mask):
        
        # 值剪裁 mini_batch_critic_values是更新后的value
        # critic_values是更新前的value
        critic_values_clip = torch.clamp(
            mini_batch_critic_values,
            critic_values - self.args.value_clip,
            critic_values + self.args.value_clip,
        )
        values_error = (mini_batch_critic_values - returns)**2 
        values_clip_error = (critic_values_clip - returns)**2 
        loss = 0.5 * self.masked_mean(torch.max(values_error, values_clip_error), mask)
        
        return loss, values_error 
    
    # 组合预训练模型的参数和value head的参数
    # value head的参数以v_head.的形式加入pretrained_model_state_dict
    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict 


    def save_checkpoint(self, model, output_dir, step, adapter_name="default", state_dict=None):

        if self.unwrap_model(model) is not model:
            model = self.unwrap_model(model)
            
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            if hasattr(model, "v_head"):
                state_dict = self.get_state_dict(model)
            else:
                state_dict = model.state_dict()

        # 保存模型
        if isinstance(model, PreTrainedModel):  
            model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if hasattr(model, "peft_config"):
                adapter_state_dict = get_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                adapter_state_dict = get_peft_model_state_dict(model.pretrained_model, state_dict, adapter_name=adapter_name)

            if hasattr(model, "v_head"):
                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v 
            torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
        # 保存peft config
        try:
            if hasattr(model, "peft_config"):
                model.peft_config.save_pretrained(output_dir)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                model.pretrained_model.peft_config.save_pretrained(output_dir)

        except AttributeError:
            if hasattr(model, "peft_config"):
                model.peft_config[adapter_name].save_pretrained(output_dir)
            else:
                model.pretrained_model.peft_config[adapter_name].save_pretrained(output_dir)

        # 保存tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    
    def record_logs(self, batch):

        mask = batch["responses_mask"][:, 1:]
        prompt_lens = torch.tensor([len(prompt) for prompt in batch["prompts_ids"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in batch["responses_ids"]], dtype=torch.float)

        logs = dict()
        ## params
        logs["lr"] = self.optimizer.param_groups[0]['lr']
        
        ## loss
        logs["loss/actor"] = batch["actor_loss"]
        logs["loss/entropy"] = batch["entropy"]
        logs["loss/critic"] = batch["critic_loss"]
        logs["loss/extra"] = batch["extra_loss"]

        logs["exp_data/reward_score_mean"] = torch.mean(batch["rewards_score"])
        logs["exp_data/reward_score_var"] = torch.var(batch["rewards_score"]) 
        
        logs["exp_data/kl_penalty_mean"] = self.masked_mean(batch["kl_penalty"], mask)
        logs["exp_data/kl_penalty_var"] = self.masked_var(batch["kl_penalty"], mask)

        logs["exp_data/rewards_with_kl_penalty_mean"] = self.masked_mean(batch["rewards_with_kl_penalty"], mask)
        logs["exp_data/rewards_with_kl_penalty_var"] = self.masked_var(batch["rewards_with_kl_penalty"], mask)
        
        logs["exp_data/actor_perplexity"] = math.exp(torch.mean(batch["actor_ce_loss"]))
        logs["exp_data/ref_perplexity"] = math.exp(torch.mean(batch["ref_ce_loss"]))
        
        ## actor
        logs["actor/advantages_mean"] = self.masked_mean(batch["advantages"], mask)
        logs["actor/advantages_var"] = self.masked_var(batch["advantages"], mask)
        
        logs["actor/ratio_mean"] = self.masked_mean(batch["ratio"], mask)
        logs["actor/ratio_var"] = self.masked_var(batch["ratio"], mask)
        
        ## critic
        logs["critic/returns_mean"] = self.masked_mean(batch["returns"], mask)
        logs["critic/returns_var"] = self.masked_var(batch["returns"], mask)

        logs["critic/values_error_mean"] = self.masked_mean(batch["values_error"], mask)
        logs["critic/values_error_var"] = self.masked_var(batch["values_error"], mask)
        
        ## length
        logs["length/prompts_length_mean"] = torch.mean(prompt_lens)
        logs["length/prompts_length_var"] = torch.var(prompt_lens)
        
        logs["length/responses_length_mean"] = torch.mean(response_lens)
        logs["length/responses_length_var"] = torch.var(response_lens)
        
        return logs


    def print_logs(self, all_logs, update_steps):

        all_logs_merged = {}
        for key in all_logs[0]:
            all_logs_merged[key] = torch.mean(torch.tensor([log[key] for log in all_logs])).to(self.device)
        
        if self.is_distributed:
            logs = {}
            torch.distributed.barrier()
            for k, v in all_logs_merged.items():
                if not isinstance(v, torch.Tensor):
                    warnings.warn(f"the log of {k} need to be tensors")
                    continue
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
                v /= self.accelerator.num_processes
                logs[k] = v 
            all_logs_merged = copy.deepcopy(logs) 
        
        
        if self.accelerator.is_main_process:
            logs = {}
            for k, v in all_logs_merged.items():
                logs[k] = v.cpu().numpy().item()
            self.accelerator.log(logs, step=int(update_steps))

            if update_steps > 0 and update_steps % self.args.logging_steps == 0:
                actor_loss, critic_loss, extra_loss = logs["loss/actor"], logs["loss/critic"], logs["loss/extra"]
                rewards_with_kl_penalty_mean = logs["exp_data/rewards_with_kl_penalty_mean"]
                lr = logs["lr"]
                print(f'update_steps:{update_steps}|lr:{lr}|actor_loss:{actor_loss}, critic_loss:{critic_loss}, extra_loss:{extra_loss}, rewards_with_kl_penalty_mean:{rewards_with_kl_penalty_mean}')
    
    
    def train_step(self, batch_mini_data, extra_inputs, step):
        # oishi: 参数处理和损失权重调整
        extra_loss_weight_warmup = self.args.extra_loss_weight
        if self.args.extra_warmup_steps_ratio is not None:
            extra_warmup_steps = int(self.args.extra_warmup_steps_ratio * self.max_steps)
        ## get extra_loss_weight 
        if self.args.extra_warmup_steps_ratio is not None:
            if step < extra_warmup_steps:
                extra_loss_weight_warmup = step / extra_warmup_steps * self.args.extra_loss_weight
            else:
                extra_loss_weight_warmup = extra_loss_weight_warmup ** 1.001 

        # oishi：准备训练数据和额外输入的处理
        responses_mask = batch_mini_data["responses_mask"]
        sequences = {"input_ids": batch_mini_data["sequences_ids"], "attention_mask": batch_mini_data["sequences_mask"]}

        # oishi：前向传播
        if self.args.use_co_model:
            with self.accelerator.accumulate(self.co_model):
                unwrap_model = self.accelerator.unwrap_model(self.co_model)
                mini_batch_actor_logits, _, mini_batch_critic_values = unwrap_model(**sequences, return_dict=True)
                _, extra_loss, _ = unwrap_model(**extra_inputs, return_dict=True)
                
                if self.args.use_multi_adapters:
                    unwrap_model.pretrained_model.set_adapter("critic")
                    mini_batch_critic_values = unwrap_model(**sequences, return_dict=True)[-1]
                    unwrap_model.pretrained_model.set_adapter("default")
        else:
            with self.accelerator.accumulate(self.model):
                mini_batch_actor_logits, mini_batch_critic_values, extra_loss = self.model(sequences, extra_inputs)
            
        # oishi：损失计算
        mini_batch_actor_log_probs = self.get_log_probs(mini_batch_actor_logits[:, :-1, :], batch_mini_data["sequences_ids"][:, 1:]) 
        entropy = self.get_entropy(mini_batch_actor_logits[:, :-1, :], responses_mask[:, 1:])
        
        actor_loss, ratio = self.actor_loss(batch_mini_data["actor_log_probs"], mini_batch_actor_log_probs, batch_mini_data["advantages"], responses_mask[:, 1:])
        
        
        critic_loss, values_error = self.critic_loss(batch_mini_data["critic_values"], mini_batch_critic_values[:, :-1], batch_mini_data["returns"], responses_mask[:, 1:])
        
        if extra_inputs is not None:
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss + extra_loss_weight_warmup * extra_loss
        else:
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss
        
        # oishi：反向传播和更新参数
        self.accelerator.backward(loss) # oishi：计算loss
        
        if self.args.max_grad_norm is not None:
            if self.args.use_co_model:
                params = [p for n, p in self.co_model.named_parameters() if p.requires_grad]
            else:
                params = [p for n, p in self.actor_model.named_parameters() if p.requires_grad] + [p for n, p in self.critic_model.named_parameters() if p.requires_grad]
                
            torch.nn.utils.clip_grad_norm_(
                parameters=params,
                max_norm=self.args.max_grad_norm
            )
        
        self.optimizer.step() #oishi：这一步更新参数
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
   
        return dict(
            all_loss=loss.detach(),
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            extra_loss=extra_loss.detach() if extra_inputs is not None else 0.0,
            entropy=entropy.detach(),
            ratio=ratio.detach(),
            values_error=values_error.detach(),
            
        )
        
        
    def train(self):
        
        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        num_examples = self.num_examples(self.dataloader)
        if self.extra_train_dataloader is not None:
            extra_data_num_examples = self.num_examples(self.extra_train_dataloader)
        else:
            extra_data_num_examples = 0 
        
        # oishi：最大更新步骤；如果设置了那么不管epoch是多少，只要达到了，就停止训练
        if self.args.max_steps > 0:
            # oishi：如果设置了最大更新步，就根据最大更新步来挑战模型训练的epoch
            # // 是整除；如果有余数就额外加1
            self.num_train_epochs = self.args.max_steps // self.num_update_steps_per_epoch + int(
                self.args.max_steps % self.num_update_steps_per_epoch > 0
            )
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps 
        else:
            self.num_train_epochs = math.ceil(self.args.num_train_epochs)
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps 

        if self.is_world_process_zero():
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}, Extra task examples = {extra_data_num_examples}")
            logger.info(f"  Num Epochs = {self.num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total steps = {self.max_steps}, Total optimization steps = {self.max_update_steps}")


        progress_bar = tqdm(total=self.max_steps, disable=not self.is_world_process_zero())
        step = 0 
        data_buffer = list()
        all_logs = list()

        # 按照epoch训练
        for epoch in range(int(self.num_train_epochs)):
            if self.extra_train_dataloader is None:
                self.extra_train_dataloader = [None] * len(self.dataloader)
            
            # 取到每个batch的数据
            for i, (batch_data, batch_extra_data) in enumerate(zip(self.dataloader, self.extra_train_dataloader)):
                if i >= self.max_dataloader_iters:
                    break 
                
                
                prompts_ids = batch_data["input_ids"]
                experience_data = self.get_experience_data(prompts_ids)
                
                # 等待加载数据
                self.accelerator.wait_for_everyone()
                data_buffer.append({'exp': experience_data, 'extra': batch_extra_data})
                # if判断数据是否加载完成
                if len(data_buffer) == self.args.mini_data_buffer_nums:
                    mini_dataset = self.get_mini_dataset(data_buffer)
                    random.shuffle(mini_dataset) 
                    data_buffer.clear()

                    # ppo训练：在ppo周期中，对同一batch的数据进行多次训练
                    for ppo_epoch in range(self.args.ppo_epochs):

                        for j, batch_mini_data in enumerate(mini_dataset):
                            step += 1 

                            if batch_extra_data is not None:
                                extra_inputs = {"input_ids": batch_mini_data["input_ids"], "labels": batch_mini_data["labels"]}
                            else:
                                extra_inputs = None 
                        
                            # 训练，在train_step中更新参数
                            result = self.train_step(batch_mini_data, extra_inputs, step)
                            batch_mini_data.update(result)
                            
                            progress_bar.update(1)

                            logs = self.record_logs(batch_mini_data)
                            all_logs.append(logs)
                            
                            update_steps = step / self.args.gradient_accumulation_steps
                            
                            # 达到梯度积累步，打印log
                            if step > 0 and step % self.args.gradient_accumulation_steps == 0:
                                self.print_logs(all_logs, update_steps) 
                                all_logs.clear()
                            
                            # 检查是否达到模型保存步，达到就保存
                            if update_steps > 0 and (update_steps % self.args.save_steps) == 0:
                                
                                if self.is_world_process_zero():
                                    if self.args.use_co_model:
                                        unwrapped_model = self.accelerator.unwrap_model(self.co_model)
                                        self.save_checkpoint(unwrapped_model, self.args.output_dir, int(update_steps))
                                        if self.args.use_multi_adapters:
                                            self.save_checkpoint(unwrapped_model, self.args.critic_output_dir, int(update_steps), adapter_name="critic")
                                    else:
                                        self.save_checkpoint(self.actor_model, self.args.output_dir, int(update_steps))
                                        self.save_checkpoint(self.critic_model, self.args.critic_output_dir, int(update_steps))


                        random.shuffle(mini_dataset) 
                        torch.cuda.empty_cache()

        progress_bar.close()
        self.accelerator.end_training()

