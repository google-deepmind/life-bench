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

"""Tests for lifebench on examples/sample_data."""

import os

import pytest

from lifebench import data
from lifebench import evaluate
from lifebench import metrics
from lifebench.constants import EASY_TASKS, TASKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "examples", "sample_data")
PREDICTIONS = os.path.join(ROOT, "examples", "sample_predictions.jsonl")
JUDGMENTS = os.path.join(ROOT, "examples", "sample_judgments.jsonl")


def test_parse_multiple_choice():
  assert metrics.parse_multiple_choice("Answer: C") == "C"
  assert metrics.parse_multiple_choice("The answer is (A).") == "A"
  assert metrics.parse_multiple_choice("I think B because ...") == "B"
  assert metrics.parse_multiple_choice("Unsure") is None


def test_parse_binary():
  assert metrics.parse_binary("Yes, she is visible.") == "Yes"
  assert metrics.parse_binary("No.") == "No"
  assert metrics.parse_binary("Not visible") is None


def test_exact_match_unparseable_is_wrong():
  assert not metrics.exact_match("Unsure", "C", "multiple_choice")
  assert not metrics.exact_match(None, "Yes", "binary")


def test_recall_at_k():
  image_to_event = {"va-e-0-0": "va-e-0", "va-e-0-1": "va-e-0"}
  gold = ["va-e-0", "va-c-a", "va-c-b"]
  retrieved = ["va-e-7-0", "va-e-0-1", "va-c-a", "va-c-b"]
  assert metrics.recall_at_k(retrieved, gold, image_to_event, 3) == pytest.approx(2 / 3)
  assert metrics.recall_at_k(retrieved, gold, image_to_event, 5) == 1.0
  assert metrics.recall_at_k(retrieved, [], image_to_event, 5) is None


def _run(judgments):
  questions = data.load_questions(DATA_DIR, include_easy=True)
  rows = evaluate.score_questions(
      questions,
      data.load_predictions(PREDICTIONS),
      data.load_judgments(judgments),
      data.load_image_to_event(DATA_DIR),
  )
  return evaluate.aggregate(rows)


def test_sample_data_layout():
  questions = data.load_questions(DATA_DIR, include_easy=True)
  assert sorted(q["task"] for q in questions) == sorted(TASKS + EASY_TASKS)


def test_report_with_judgments():
  report = _run(JUDGMENTS)
  assert report["overall"]["accuracy"] == pytest.approx(0.7)
  assert report["overall"]["recall@3"] == pytest.approx((2 + 7 / 3) / 7)
  assert report["overall"]["recall@5"] == pytest.approx((6 + 2 / 3) / 7)
  cats = report["categories"]
  assert cats["Concept Identification"]["accuracy"] == pytest.approx(2 / 3)
  assert cats["Event Understanding"]["accuracy"] == pytest.approx(0.75)
  assert cats["Aggregated Reasoning"]["accuracy"] == pytest.approx(2 / 3)
  assert cats["Event Understanding"]["recall@3"] == pytest.approx(7 / 12)
  assert report["single_hop"]["concept_vqa_easy"]["accuracy"] == 0.0


def test_report_without_judgments():
  report = _run(None)
  assert report["overall"]["accuracy"] is None
  assert report["overall"]["recall@5"] == pytest.approx((6 + 2 / 3) / 7)
  assert report["tasks"]["text_concept_qa"]["accuracy"] == 1.0
  assert report["tasks"]["scene_and_activity"]["unjudged"] == 1


def test_cli(tmp_path):
  prompts = tmp_path / "prompts.jsonl"
  evaluate.main([
      "--predictions", PREDICTIONS,
      "--data_dir", DATA_DIR,
      "--export_judge_prompts", str(prompts),
  ])
  assert len(data.read_jsonl(str(prompts))) == 7
