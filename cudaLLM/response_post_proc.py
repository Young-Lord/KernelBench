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
import re


def summary_postprocess(input_text, last_response_sep=['<summarize>', '</summarize>'], last_response_strict=True):
    start_tag, end_tag = last_response_sep

    # Find the last occurrence of the start_tag
    start_index = input_text.rfind(start_tag)
    if start_index == -1:
        # Start tag not found
        if last_response_strict:
            return ''
        else:
            return input_text

    # Move the index to the end of the start_tag
    start_index += len(start_tag)

    # Find the end_tag starting from the end of start_tag
    end_index = input_text.rfind(end_tag, start_index)
    if end_index == -1:
        # End tag not found
        if last_response_strict:
            return ''
        else:
            return input_text

    # Extract and return the content between start_tag and end_tag
    return input_text[start_index:end_index].strip()


import re


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java'], last_response_strict=True):
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return f'```{language}\n{code_content}\n```'
    else:
        if last_response_strict:
            return ''
        else:
            return input_text
