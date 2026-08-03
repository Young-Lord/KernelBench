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
import tempfile
import os
import glob
import re
import json
import ray
import torch
from typing import *
import subprocess
from unittest.mock import patch

TEST_CODE_TMPL = r'''
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import sys
from contextlib import contextmanager

# op_dir, = list(Path('build').iterdir())
# sys.path.append(str(op_dir))

def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')

from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew


def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

with block_torch_functional():
    cuda_outputs = cuda_model(*cuda_inputs)
torch_outputs = torch_model(*torch_inputs)
check_equal(cuda_outputs, torch_outputs)
'''


def _compile_ext(cuda_code: str) -> Tuple[bool, Dict]:
    ret = {
        'ext_filename': None,
        'ext_content': None,
        'msg': None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), 'w') as fout:
            fout.write(cuda_code)

        compile_log = ''
        success = True
        try:
            compile_cmd = f"python3 model_new.py"
            with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0", "TORCH_EXTENSIONS_DIR": "build", "MAX_JOBS": "1"}):
                compile_result = subprocess.run(compile_cmd,
                                                timeout=180,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                shell=True,
                                                cwd=tmpdir)
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], 'rb') as fin:
                bin_content = fin.read()
            ret['ext_filename'] = os.path.basename(so_files[0])
            ret['ext_content'] = bin_content
            ret['msg'] = "compile success"
            success = True
        except subprocess.TimeoutExpired as e:
            success = False
            ret['msg'] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret['msg'] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret


def _exec_eval(ext_filename: str, ext_content: bytes, cuda_code: str, pytorch_module: str) -> Tuple[bool, str]:
    """Compile and execute test code which checks output with cuda implementation and pytorch module
    :param ext_filename: the cuda extension filename, in the format as "cuda_module.cpython-xxx.so"
    :param ext_content: file content of the extension file
    :param cuda_code: file content of the python file containing inline cuda code
    :param pytorch_module: pytorch baseline implementation. Should have Model.forward(...) and get_inputs() api
    :return (status,msg): (True,stdout) for success, (False,stderr) for error
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ext_filename), 'wb') as fout:
            fout.write(ext_content)
        with open(os.path.join(tmpdir, "model_new.py"), 'w') as fout:
            fout.write(cuda_code)
        with open(os.path.join(tmpdir, "model.py"), 'w') as fout:
            fout.write(pytorch_module)
        with open(os.path.join(tmpdir, "test.py"), 'w') as fout:
            fout.write(TEST_CODE_TMPL)

        test_log = ''
        try:
            test_cmd = f"python3 test.py"
            test_result = subprocess.run(test_cmd,
                                         timeout=60,
                                         stderr=subprocess.STDOUT,
                                         stdout=subprocess.PIPE,
                                         shell=True,
                                         cwd=tmpdir)
            test_log = test_result.stdout.decode()
        except subprocess.TimeoutExpired as e:
            return False, "failed: test timed out"
        except Exception as e:
            return False, f"failed: test error: [{e}] log: [{test_log}]"
        if test_result.returncode != 0:
            return False, f"failed: test error: [{test_log}]"

    return True, "test success"


def _extract_cuda_code(text: str):
    codeblock_seps = ['python']
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return code_content
    return None
    

def _validate_cuda_code(code: str):
    all_ops = set(torch.ops.aten.__dict__.keys())
    allowed_ops = set(['empty', 'empty_like', 'empty_strided', 'zeros', \
                       'zeros_like', 'ones', 'ones_like', 'numel', 'view',
                       'copy', 'dim', 'eye', 'full', 'full_like', 'mode',
                       'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_zeros',
                       'randn', 'rand'])
    forbidden_ops = all_ops - allowed_ops
    pattern = re.compile("(torch::|aten::|torch\.)(" + "|".join(forbidden_ops) + ")\(", flags=re.DOTALL)
    matched = re.search(pattern, code)
    if matched is not None:
        return False, f'Using {matched.group(0)[:-1]} is forbidden'
    return True, 'success'


def compute_score(solution_str, ground_truth, config, *args, **kwargs) -> dict[str, Any]:
    cuda_code = _extract_cuda_code(solution_str)
    if cuda_code is None:
        return dict(score=-1.0, msg='cannot extract cuda code from model response')
    else:
        validate_ret, validate_msg = _validate_cuda_code(cuda_code)
        if not validate_ret:
            return dict(score=-1.0, msg=validate_msg)
        remote_compile_ext = ray.remote(num_cpus=8)(_compile_ext)
        status, compile_res = ray.get(remote_compile_ext.remote(cuda_code=cuda_code))
        if not status:
            return dict(score=-1.0, msg=compile_res['msg'])

        gt_dict = json.loads(ground_truth)
        ext_filename = compile_res['ext_filename']
        ext_content = compile_res['ext_content']
        pytorch_module = gt_dict['pytorch_module']
        run_kwargs = dict(ext_filename=ext_filename, ext_content=ext_content, cuda_code=cuda_code, pytorch_module=pytorch_module)
        gpu_eval_task = ray.remote(num_gpus=1)(_exec_eval)
        eval_future = gpu_eval_task.remote(**run_kwargs)
        status, msg = ray.get(eval_future)

        if not status:
            return dict(score=-1.0, msg=msg)
        return dict(score=1.0, msg='success')

