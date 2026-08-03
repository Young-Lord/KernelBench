# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

project_name='cudaLLM_sft'
experiment_name=Qwen-8B-SFT
default_hdfs_dir=$experiment_name

MODEL_PATH=Qwen/Qwen3-8B
TRAIN_FILE=sft_cuda_llm_r1.parquet

torchrun --nproc_per_node=8 --nnodes=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TRAIN_FILE} \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.prompt_key=prompts \
    data.response_key=responses \
    data.truncation=left \
    data.max_length=16384 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=64 \
    model.partial_pretrain=${MODEL_PATH} \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.01 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=${default_hdfs_dir}