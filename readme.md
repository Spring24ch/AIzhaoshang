# 项目说明

本项目旨在指导如何下载数据集、创建SFT和DPO数据集、清洗数据集以及使用llamafactory-cli进行训练。

## 目录
1. [下载数据集并创建SFT和DPO数据集](#1-下载数据集并创建sft和dpo数据集)
2. [清洗数据集](#2-清洗数据集)
3. [使用llamafactory-cli进行SFT和DPO训练](#3-使用llamafactory-cli进行sft和dpo训练)

## 1. 下载数据集并创建SFT和DPO数据集

要开始，请运行`datacreate.ipynb`。这个notebook将引导你完成下载数据集的过程，并教你如何制作SFT和DPO数据集。

## 2. 清洗数据集

我们提供了一个Python脚本`filtering.py`来帮助你清洗你的数据集。下面是一个示例命令，用于清洗指定的数据集：

```bash
python filtering.py \
    --dataset_name bigcode/the-stack-smol \
    --subset data/java \
    --filters basic,comments,stars,fertility \
    --hub_username loubnabnl \
    --remote_repo test_filter_pipeline_java



## 3. 使用llamafactory-cli进行SFT和DPO训练

### SFT 训练命令示例

以下是一个使用`llamafactory-cli`工具进行监督微调（Supervised Fine-Tuning, SFT）的命令示例。请根据您的实际需求调整参数。

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/deepseek-coder-1.3b-base \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset java_fim \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/DeepSeek-Coder-6.7B-Base/lora/train_2025-05-13-10-40-13 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --adapter_name_or_path saves/DeepSeek-Coder-6.7B-Base/lora/train_2025-05-12-09-33-32 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

###DPO训练示例
以下是一个使用llamafactory-cli工具进行直接偏好优化（Direct Preference Optimization, DPO）训练的命令示例。请根据您的实际需求调整参数。
```bash
llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/deepseek-coder-1.3b-base \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dataset_name \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/DeepSeek-Coder-6.7B-Base/lora/train_2025-05-13-09-48-03 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid
    