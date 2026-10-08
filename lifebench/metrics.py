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

"""Accuracy and event-level recall for Life-Bench."""

import re
from typing import Dict, List, Optional, Sequence


def parse_multiple_choice(text: str) -> Optional[str]:
  """Extracts the option letter (A-D) from a model response."""
  text = _answer_line(text)
  m = re.search(r"\b([A-D])\b", text)
  return m.group(1) if m else None


def parse_binary(text: str) -> Optional[str]:
  """Extracts Yes/No from a model response."""
  m = re.search(r"\b(yes|no)\b", _answer_line(text), flags=re.IGNORECASE)
  return m.group(1).capitalize() if m else None


def _answer_line(text) -> str:
  text = "" if text is None else str(text).strip()
  for line in text.split("\n"):
    if "answer:" in line.lower():
      return line.split(":", 1)[-1].strip()
  return text


def exact_match(prediction, answer: str, answer_type: str) -> bool:
  """Exact match for multiple-choice and binary questions.

  Unparseable predictions count as wrong.
  """
  if answer_type == "multiple_choice":
    parsed = parse_multiple_choice(prediction)
    return parsed is not None and parsed.upper() == answer.strip().upper()
  if answer_type == "binary":
    parsed = parse_binary(prediction)
    return parsed is not None and parsed.lower() == answer.strip().lower()
  raise ValueError(f"exact_match does not apply to {answer_type}")


def recall_at_k(
    retrieved_ids: Sequence[str],
    reference_ids: Sequence[str],
    image_to_event: Dict[str, str],
    k: int,
) -> Optional[float]:
  """Event-level Recall@k.

  Gold references are `event_id`s and `concept_id`s. A gold event is hit if
  any of its images is among the top-k retrieved ids; a gold concept is hit
  if its `concept_id` is among the top-k retrieved ids.

  Returns None when the question has no gold references.
  """
  gold = list(dict.fromkeys(reference_ids))
  if not gold:
    return None
  hit = set()
  for rid in list(retrieved_ids)[:k]:
    hit.add(rid)
    if rid in image_to_event:
      hit.add(image_to_event[rid])
  return sum(1 for g in gold if g in hit) / len(gold)


def mean(values: List[float]) -> Optional[float]:
  values = [v for v in values if v is not None]
  return sum(values) / len(values) if values else None
