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

"""Loading Life-Bench raw files, predictions and judgments."""

import glob
import json
import os
from typing import Dict, Iterable, List, Optional

from lifebench.constants import EASY_TASKS, TASKS

# Files needed for scoring (images are not needed).
METADATA_PATTERNS = ["questions/*.jsonl", "events/metadata.jsonl"]


def read_jsonl(path: str) -> List[dict]:
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(
    data_dir: str,
    tasks: Optional[List[str]] = None,
    vaccounts: Optional[List[str]] = None,
    include_easy: bool = False,
) -> List[dict]:
  """Loads `questions/<task>.jsonl` files from a Life-Bench raw directory."""
  if tasks is None:
    tasks = list(TASKS) + (EASY_TASKS if include_easy else [])
  questions = []
  for task in tasks:
    path = os.path.join(data_dir, "questions", f"{task}.jsonl")
    if not os.path.exists(path):
      raise FileNotFoundError(path)
    for q in read_jsonl(path):
      if vaccounts and q["vaccount"] not in vaccounts:
        continue
      questions.append(q)
  return questions


def load_image_to_event(data_dir: str) -> Dict[str, str]:
  """Maps every historical `image_id` to its `event_id`."""
  path = os.path.join(data_dir, "events", "metadata.jsonl")
  if not os.path.exists(path):
    raise FileNotFoundError(path)
  return {r["image_id"]: r["event_id"] for r in read_jsonl(path)}


def load_predictions(path: str) -> Dict[str, dict]:
  """Loads predictions keyed by `sample_id`."""
  preds = {}
  for row in read_jsonl(path):
    sid = row["sample_id"]
    if sid in preds:
      raise ValueError(f"Duplicate sample_id in predictions: {sid}")
    preds[sid] = row
  return preds


def load_judgments(path: Optional[str]) -> Dict[str, str]:
  """Loads judge results (`{"sample_id": ..., "Result": "Correct"|"Wrong"}`)."""
  if not path:
    return {}
  judgments = {}
  for row in read_jsonl(path):
    result = str(row["Result"]).strip().lower()
    if result not in ("correct", "wrong"):
      raise ValueError(
          f"Invalid Result for {row['sample_id']}: {row['Result']!r}"
      )
    judgments[row["sample_id"]] = result
  return judgments


def download_metadata(repo_id: str = "google/life-bench") -> str:
  """Downloads the files needed for scoring from the Hugging Face Hub."""
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top

  return snapshot_download(
      repo_id=repo_id, repo_type="dataset", allow_patterns=METADATA_PATTERNS
  )


def available_tasks(data_dir: str) -> List[str]:
  return sorted(
      os.path.splitext(os.path.basename(p))[0]
      for p in glob.glob(os.path.join(data_dir, "questions", "*.jsonl"))
  )
