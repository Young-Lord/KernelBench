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
from typing import *
import hydra
from pprint import pprint
from omegaconf import OmegaConf, ListConfig
import uuid
import time
import asyncio
import sys
from codetiming import Timer

import copy
import os
import pandas as pd
import numpy as np
import ray
import torch
import tempfile
from torch.utils.data import Dataset

from transformers import AutoTokenizer, PreTrainedTokenizer
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.fs import copy_local_path_from_hdfs
import shutil

from verl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from validation_manager import ValidateManager
from main_ppo import RewardManager, RemoteClient


def init_ray():
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "0",
                "BPEX_NO_WARN_ON_UNTUNED_CASE": "1",
            }
        }

        ray.init(runtime_env=runtime_env)


@hydra.main(config_path="config", config_name="val_only", version_base=None)
def main(config):
    init_ray()
    main_task(config)


def text_to_messages(tokenizer, text) -> List:
    """Convert token ids to chat"""
    import re
    role_strings_to_role = dict()
    for role in ['assistant', 'user', 'system']:
        try:
            pattern = tokenizer.apply_chat_template([{'content': '', 'role': role}], tokenize=False)
            role_string = pattern.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
            role_strings_to_role[role_string] = role
        except Exception as e:
            print(f"[WARN]: role {role} cannot parse role_string")
    
    role_strings_to_role["tool\n"] = "tool"
    messages = []
    prev_role = None
    prev_pos = -1
    for i in range(len(text)):
        # try match a role
        for role_string, role in role_strings_to_role.items():
            if (len(role_string) > 0 and text[i:i + len(role_string)] == role_string) or (len(role_string) == 0 and i == 0):
                # match a role, append prev_role to 
                if prev_role is not None:
                    content = text[prev_pos: i].replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
                    if len(content) > 0:
                        messages.append({'role': prev_role, 'content': content})
                prev_role = role
                prev_pos = i + len(role_string)
                break
    if prev_role is not None and prev_pos < len(text):
        content = text[prev_pos:].replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
        messages.append({'role': prev_role, 'content': content})

    return messages


def tokens_to_messages(tokenizer, text) -> List:
    """Convert token ids to chat"""
    import re
    pattern = re.compile(
        re.escape(tokenizer.bos_token) + r"(tool|assistant|user|system)\n(.*?)" + re.escape(tokenizer.eos_token),
        re.DOTALL,
    )
    matches = pattern.findall(text)
    messages = []
    for match in matches:
        role, content = match
        loss_mask = 1 if role == "assistant" else 0
        messages.append({"content": content, "role": role, "loss_mask": loss_mask})
    return messages

class DummyDataSet(Dataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int,
        max_response_length: int,
        prompt_key: str = "prompt",
        response_key: str = "output",
        cache_dir="~/.cache/verl/rlhf",
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.prompt_key = prompt_key
        self.response_key = response_key
        self._download()
        self._read_files()
    
    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
    
    def _unpack_multi_responses(self, df: pd.DataFrame):
        extra_info = df.get("extra_info", np.array([{} for _ in range(len(df))], dtype=object))
        index = [extra_info.get("index", i) for i in range(len(extra_info))]
        df["index"] = index
        unpack_rows = []
        for i, row in df.iterrows():
            outputs = row[self.response_key]
            if isinstance(outputs, str):
                outputs = np.array([outputs], dtype=object)
            for output in outputs:
                new_row = row.copy()
                new_row[self.response_key] = output
                unpack_rows.append(new_row)
        return pd.DataFrame(unpack_rows).reset_index(drop=True)
    
    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframe = self._unpack_multi_responses(dataframe)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, item):
        import verl.utils.torch_functional as verl_F
        row_dict = self.dataframe.iloc[item].to_dict()
        prompt = row_dict.pop(self.prompt_key)
        prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="left",
        )
        response = row_dict.pop(self.response_key)
        response_ids, response_attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=response,
            tokenizer=self.tokenizer,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation="right",
        )
        raw_chat = tokens_to_messages(self.tokenizer, prompt)
        if len(raw_chat) == 0:
            raw_chat = [{'role': 'user', 'content': 'dummy chat because tokens_to_messages failed to parse'}]
        row_dict["raw_prompt"] = raw_chat
        row_dict["input_ids"] = torch.hstack((prompt_ids[0], response_ids[0]))
        row_dict["attention_mask"] = torch.hstack((prompt_attention_mask[0], response_attention_mask[0]))
        row_dict["prompt_names"] = [""]
        return row_dict

class DummyRollout:

    def __init__(self):
        self.world_size = 1
        self.eos_callback_fn = None

    def generate_sequences(self, batch):
        output_batch = {
            "input_ids": batch.batch['input_ids'].to(torch.int32),
            "attention_mask": batch.batch['attention_mask'].to(torch.int8),
        }
        non_tensors = {
            "reward_model": batch.non_tensor_batch['reward_model'],
        }
        output_batch = DataProto.from_dict(output_batch, non_tensors=non_tensors)
        return output_batch
    
    def release_param_and_cache(self):
        pass


def get_dataset(config, tokenizer):
    if config.val_config.dummy:
        val_dataset = DummyDataSet(
            parquet_files=config.data.val_files,
            tokenizer=tokenizer,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
            prompt_key=config.data.prompt_key,
            response_key=config.val_config.response_key,
        )
    else:
        from verl.utils.dataset.rl_dataset import RLHFDataset
        
        val_dataset = RLHFDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            config=config.data
        )
    if config.val_config.iters > 0:
        assert config.val_config.batch_size > 0
        assert config.val_config.iters * config.val_config.batch_size <= len(val_dataset)
    return val_dataset

def get_dataloader(config, dataset, batch_size: int, iter: int):
    from torch.utils.data import DataLoader, Subset
    from verl.utils.dataset.rl_dataset import collate_fn

    subset_index = range(batch_size * iter, batch_size * (iter + 1))
    subset = Subset(dataset, subset_index)

    val_dataloader = DataLoader(
        dataset=subset,
        batch_size=len(subset),
        shuffle=config.data.shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return val_dataloader


def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    tokenizer.padding_side = "left"

    logger = Tracking(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        default_backend=config.trainer.logger,
        config=OmegaConf.to_container(config, resolve=True),
    )
    val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

    val_dataset = get_dataset(config, tokenizer)

    use_rm = False
    if config.val_config.iters < 0:
        batch_size, iters = len(val_dataset), 1
    else:
        batch_size, iters = config.val_config.batch_size, config.val_config.iters

    if config.val_config.dummy:
        print("val from pre-gen dataset")
        rollout_wg = DummyRollout()
    else:
        print("use rollout actor for generation")
        rollout_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker), 
            config=config.actor_rollout_ref,
            role="rollout",
        )
        rollout_wg = RayWorkerGroup(resource_pool=rollout_pool, ray_cls_with_init=ray_cls_with_init)
        rollout_wg.init_model()
    remote_client = RemoteClient.options(name='remote_client').remote(config=config, tokenizer=tokenizer)

    log_dfs = []
    for iter_num in range(iters): # Renamed iter to iter_num
        metrics = dict()
        # Assuming get_dataloader and ValidateManager are defined
        val_dataloader = get_dataloader(config, val_dataset, batch_size, iter=iter_num)
        validation_manager = ValidateManager(config, logger, val_dataloader, tokenizer, use_rm, val_reward_fn)
        validation_manager.actor_rollout_wg = rollout_wg

        with Timer("val") as timer:
            seed = os.getenv('SEED', 0)
            tmp_file_name = f"{config.data.val_files.split('/')[-1]}/eval_{seed}.jsonl"
            directory_to_create = os.path.dirname(tmp_file_name)
            os.makedirs(directory_to_create, exist_ok=True)
            validation_manager.validate(val_epoch=1, global_step=iter_num, need_log=True, log_file=tmp_file_name)
            df = pd.read_json(tmp_file_name, lines=True)
        metrics['timing/val'] = timer.last

        logger.log(data=metrics, step=iter_num)

        if config.val_config.add_sft_messages:
            messages = []
            for i, row in df.iterrows():
                messages.append(text_to_messages(tokenizer, text=row.prompt + row.response))
            df["messages"] = messages
        
        log_dfs.append(df)

    log_df = pd.concat(log_dfs).reset_index(drop=True)

    if config.val_config.save_log_path is not None:
        tmp_parquet_file_name = None # Initialize to None
        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_parquet_file:
                tmp_parquet_file_name = tmp_parquet_file.name
            
            log_df.to_parquet(tmp_parquet_file_name)
            shutil.copy(tmp_parquet_file_name, config.val_config.save_log_path)
            shutil.copy(tmp_parquet_file_name, f"{config.val_config.save_log_path}.{int(time.time())}")
        finally:
            # Correctly check for existence and remove the file
            if tmp_parquet_file_name and os.path.exists(tmp_parquet_file_name): # <--- Corrected
                os.remove(tmp_parquet_file_name) # <--- Corrected

if __name__ == "__main__":
    main()
