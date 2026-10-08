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
  for text in ["C", "c", "C.", "(C)", "**C**", "Answer: C", "answer: c",
               "Final Answer: **C**", "**Answer:** C", "The answer is (C).",
               "Option: C", "Reasoning first.\nAnswer: C"]:
    assert metrics.parse_multiple_choice(text) == "C", text
  for text in ["C) A blue plaid shirt", "A blue plaid shirt",
               "I think B because ...", '{"answer": "C"}', '["C"]',
               "C or D", "Unsure", "", None]:
    assert metrics.parse_multiple_choice(text) is None, text


def test_parse_binary():
  for text in ["Yes", "yes", "No.", "**No**", "Answer: Yes",
               "The answer is No", "YES!"]:
    assert metrics.parse_binary(text) in ("Yes", "No"), text
  for text in ["Yes, she is visible.", "No, it's not", "Not visible",
               '{"Result": "Yes"}', "Yes/No", "", None]:
    assert metrics.parse_binary(text) is None, text


def test_exact_match_unparseable_is_none():
  assert metrics.exact_match("Answer: C", "C", "multiple_choice") is True
  assert metrics.exact_match("Answer: B", "C", "multiple_choice") is False
  assert metrics.exact_match("Unsure", "C", "multiple_choice") is None
  assert metrics.exact_match("Yes, she is.", "Yes", "binary") is None
  assert metrics.exact_match(None, "Yes", "binary") is None


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


def test_unparseable_exact_match_goes_to_judge(tmp_path):
  """An unparseable MC/binary answer is reported and judged, not scored."""
  preds = data.load_predictions(PREDICTIONS)
  preds["david_text_concept_qa_22"]["prediction"] = "C) Blue, like his father."
  preds["david_visual_concept_recognition_41"]["prediction"] = "Yes, she is."
  pred_path = tmp_path / "preds.jsonl"
  data.write_jsonl(str(pred_path), preds.values())

  questions = data.load_questions(DATA_DIR, include_easy=True)
  rows = evaluate.score_questions(
      questions, preds, {}, data.load_image_to_event(DATA_DIR)
  )
  report = evaluate.aggregate(rows)
  assert sorted(report["unparseable_samples"]) == [
      "david_text_concept_qa_22", "david_visual_concept_recognition_41"]
  assert report["tasks"]["text_concept_qa"]["unparseable"] == 1
  assert report["tasks"]["text_concept_qa"]["accuracy"] is None  # not wrong

  prompts = tmp_path / "prompts.jsonl"
  evaluate.main([
      "--predictions", str(pred_path),
      "--data_dir", DATA_DIR,
      "--export_judge_prompts", str(prompts),
  ])
  exported = {r["sample_id"] for r in data.read_jsonl(str(prompts))}
  assert {"david_text_concept_qa_22",
          "david_visual_concept_recognition_41"} <= exported
  assert len(exported) == 9

  judgments = {"david_text_concept_qa_22": "correct",
               "david_visual_concept_recognition_41": "wrong"}
  rows = evaluate.score_questions(
      questions, preds, judgments, data.load_image_to_event(DATA_DIR)
  )
  report = evaluate.aggregate(rows)
  assert report["tasks"]["text_concept_qa"]["accuracy"] == 1.0
  assert report["tasks"]["visual_concept_recognition"]["accuracy"] == 0.0
