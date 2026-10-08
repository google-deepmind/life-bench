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


# Leading labels such as "Answer:", "Final Answer:", "The answer is", "Option:".
_LABEL = re.compile(
    r"^(?:the\s+)?(?:final|correct|predicted|my)?[\s_]*"
    r"(?:answer|option|choice|prediction|response)\s*(?:is|:|=|-)?\s*[:=]?\s*",
    flags=re.IGNORECASE,
)
_MARKDOWN = re.compile(r"[*_`#]+")
_WRAPPERS = "()\"'“”‘’"
_TRAILING = ".,;:!)"


def clean_answer(text) -> str:
  """Reduces a model response to its bare final answer.

  Takes the line holding the answer label (if any), then strips markdown
  emphasis, answer labels, surrounding brackets/quotes and trailing
  punctuation. The result is meant to be compared against an option letter
  or Yes/No; it is not an attempt to interpret free-form text.
  """
  text = "" if text is None else str(text).strip()
  for line in text.split("\n"):
    if "answer:" in line.lower():
      text = line.split(":", 1)[-1]
      break
  text = _MARKDOWN.sub("", text).strip()
  text = _LABEL.sub("", text, count=1).strip()
  text = text.strip(_WRAPPERS).rstrip(_TRAILING).strip(_WRAPPERS).strip()
  return text


def parse_multiple_choice(text) -> Optional[str]:
  """Returns the option letter (A-D) if the cleaned response is exactly one.

  Returns None when the response cannot be reduced to a single option letter;
  such responses are not exact-match scorable.
  """
  cleaned = clean_answer(text)
  return cleaned.upper() if re.fullmatch(r"[A-Da-d]", cleaned) else None


def parse_binary(text) -> Optional[str]:
  """Returns "Yes"/"No" if the cleaned response is exactly yes or no.

  Returns None when the response cannot be reduced to yes/no; such responses
  are not exact-match scorable.
  """
  cleaned = clean_answer(text).lower()
  return cleaned.capitalize() if cleaned in ("yes", "no") else None


def parse_answer(prediction, answer_type: str) -> Optional[str]:
  """Parses a multiple-choice or binary prediction; None if unparseable."""
  if answer_type == "multiple_choice":
    return parse_multiple_choice(prediction)
  if answer_type == "binary":
    return parse_binary(prediction)
  raise ValueError(f"parse_answer does not apply to {answer_type}")


def exact_match(prediction, answer: str, answer_type: str) -> Optional[bool]:
  """Exact match for multiple-choice and binary questions.

  Returns None when the prediction cannot be parsed to an option letter or
  Yes/No, i.e. the question cannot be scored by exact match.
  """
  parsed = parse_answer(prediction, answer_type)
  if parsed is None:
    return None
  return parsed.lower() == answer.strip().lower()


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
