# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation code for Life-Bench."""

from lifebench.constants import CATEGORIES, EASY_TASKS, RECALL_TASKS, TASK_NAMES, TASKS
from lifebench.judge import JUDGE_PROMPT, build_judge_prompt
from lifebench.metrics import clean_answer
from lifebench.metrics import exact_match
from lifebench.metrics import parse_answer
from lifebench.metrics import parse_binary
from lifebench.metrics import parse_multiple_choice
from lifebench.metrics import recall_at_k

__version__ = "1.0.0"
