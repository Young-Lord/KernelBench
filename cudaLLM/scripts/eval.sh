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

# gen + eval
set -x

PROJECT_NAME='cudaLLM_eval_only'
EXPERIMENT_NAME='level_1'

SFT_MODEL_PATH=actor
TEST_FILE=level_1.parquet
default_hdfs_dir=${EXPERIMENT_NAME}

max_prompt_length=8192
max_response_length=12288
use_last_response=lastcodeblock
last_response_sep=['python']
gen_tp=4
train_traj_micro_bsz_per_gpu=2

num_trainer_nodes=1
num_gpu_per_node=4

python3 main_val.py \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    +data.answer_key=answer \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation='left' \
    data.return_raw_chat=True \
    +reward_model.use_last_response=${use_last_response} \
    +reward_model.last_response_sep=${last_response_sep} \
    +reward_model.reward_executor_maxnum=1000 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    trainer.nnodes=${num_trainer_nodes} \
    trainer.n_gpus_per_node=${num_gpu_per_node} \
    val_config.add_sft_messages=True \
    val_config.save_log_per_iter=True \
    val_config.batch_size=100 \
    val_config.iters=1 \
    +val_config.dummy=False \
    +val_config.response_key=output \

