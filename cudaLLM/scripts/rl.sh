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

project_name='cudaLLM_RL_8B'
experiment_name=Qwen-8B-RL
default_hdfs_dir=$experiment_name

SFT_MODEL_PATH='Qwen-8B-SFT'
TRAIN_FILE='rl_cuda_llm_0424.parquet'
TEST_FILE=['level_1.parquet','level_2.parquet','level_3.parquet','level_4.parquet']


# train
max_prompt_length=8192
max_response_length=12288
total_epochs=100
test_freq=5
save_freq=10
bon=8
actor_lr=1e-6
lr_warmup_steps=10 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0
use_last_response=lastcodeblock
last_response_sep=['python']
gae_gamma=1.0
gae_lam=0.95
kl_penalty=low_var_kl
adv_estimator=grpo
use_dynamic_bsz=True
train_batch_size=256
ppo_mini_batch_size=256
infer_micro_batch_size=256
train_micro_batch_size=256
ref_sp_size=1
reward_sp_size=1
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
fsdp_size=-1
actor_sp_size=1
gen_tp=4
n_gpus=4
offload=True
n_nodes=32

python3 main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    +data.answer_key=answer \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=prompt \
    data.train_batch_size=${train_batch_size} \
    data.truncation='left' \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_penalty} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.89 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${bon} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gae_gamma} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.kl_penalty=${kl_penalty} \
    algorithm.lam=${gae_lam} \
    reward_model.enable=False \
    reward_model.mean=0.0 \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.model.input_tokenizer=null \
    reward_model.std=1.0 \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    reward_model.use_dynamic_bsz=${use_dynamic_bsz} \
    +reward_model.use_last_response=${use_last_response} \
    +reward_model.last_response_sep=${last_response_sep} \
    +reward_model.reward_executor_maxnum=1000 \
    +reward_model.extra_kwargs='{cuda_rm_enable_profiling: True}' \
    +reward_model.punish_no_answer=v0 \
    +reward_model.reward_0_for_overlong_rsp=False \
    trainer.critic_warmup=0 \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=${n_nodes} \
    trainer.num_cases_to_wandb=100 \
    trainer.project_name=${project_name} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.total_epochs=${total_epochs}