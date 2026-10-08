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

"""Scores predictions on Life-Bench.

Usage:
  python -m lifebench.evaluate --predictions preds.jsonl --data_dir DATA_DIR \
      [--judgments judgments.jsonl] [--export_judge_prompts prompts.jsonl] \
      [--output report.json]
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

from lifebench import data
from lifebench.constants import CATEGORIES
from lifebench.constants import DEFAULT_RECALL_K
from lifebench.constants import EASY_TASKS
from lifebench.constants import RECALL_TASKS
from lifebench.constants import TASK_NAMES
from lifebench.constants import TASKS
from lifebench.judge import build_judge_prompt
from lifebench.metrics import exact_match
from lifebench.metrics import mean
from lifebench.metrics import recall_at_k


def score_questions(
    questions: List[dict],
    predictions: Dict[str, dict],
    judgments: Dict[str, str],
    image_to_event: Dict[str, str],
    recall_k=DEFAULT_RECALL_K,
) -> List[dict]:
  """Scores each question; `correct` is None for unjudged open questions."""
  rows = []
  for q in questions:
    sid = q["sample_id"]
    pred = predictions.get(sid)
    row = {
        "sample_id": sid,
        "vaccount": q["vaccount"],
        "task": q["task"],
        "answer_type": q["answer_type"],
        "has_prediction": pred is not None,
        "correct": None,
    }
    prediction = pred.get("prediction") if pred else None
    if q["answer_type"] in ("multiple_choice", "binary"):
      row["correct"] = exact_match(prediction, q["answer"], q["answer_type"])
    elif pred is None:
      row["correct"] = False
    elif sid in judgments:
      row["correct"] = judgments[sid] == "correct"

    if q["task"] in RECALL_TASKS and q.get("reference_ids"):
      retrieved = (pred or {}).get("retrieved_ids")
      row["has_retrieval"] = retrieved is not None
      for k in recall_k:
        row[f"recall@{k}"] = recall_at_k(
            retrieved or [], q["reference_ids"], image_to_event, k
        )
    rows.append(row)
  return rows


def _task_summary(rows: List[dict], recall_k) -> dict:
  n = len(rows)
  unjudged = sum(1 for r in rows if r["correct"] is None)
  summary = {
      "n": n,
      "missing_predictions": sum(1 for r in rows if not r["has_prediction"]),
      "unjudged": unjudged,
      "accuracy": None
      if unjudged or not n
      else sum(1 for r in rows if r["correct"]) / n,
  }
  if any(r.get("has_retrieval") for r in rows):
    for k in recall_k:
      summary[f"recall@{k}"] = mean([r.get(f"recall@{k}") for r in rows])
  return summary


def _macro(summaries: List[dict], key: str) -> Optional[float]:
  values = [s.get(key) for s in summaries]
  if not values or any(v is None for v in values):
    return None
  return sum(values) / len(values)


def aggregate(rows: List[dict], recall_k=DEFAULT_RECALL_K) -> dict:
  """Per-task, per-category, overall, and per-Vaccount summaries."""
  by_task = {}
  for r in rows:
    by_task.setdefault(r["task"], []).append(r)

  tasks = {t: _task_summary(by_task[t], recall_k) for t in TASKS if t in by_task}
  easy = {
      t: _task_summary(by_task[t], recall_k) for t in EASY_TASKS if t in by_task
  }

  categories = {}
  for cat, cat_tasks in CATEGORIES.items():
    present = [tasks[t] for t in cat_tasks if t in tasks]
    if len(present) != len(cat_tasks):
      continue
    categories[cat] = {"accuracy": _macro(present, "accuracy")}
    if cat != "Aggregated Reasoning":
      for k in recall_k:
        categories[cat][f"recall@{k}"] = _macro(present, f"recall@{k}")

  overall = {}
  if len(tasks) == len(TASKS):
    overall["accuracy"] = _macro(list(tasks.values()), "accuracy")
    recall_tasks = [tasks[t] for t in RECALL_TASKS]
    for k in recall_k:
      overall[f"recall@{k}"] = _macro(recall_tasks, f"recall@{k}")

  vaccounts = {}
  for va in sorted({r["vaccount"] for r in rows}):
    va_rows = [r for r in rows if r["vaccount"] == va and r["task"] in TASKS]
    va_tasks = {}
    for r in va_rows:
      va_tasks.setdefault(r["task"], []).append(r)
    va_summaries = [_task_summary(v, recall_k) for v in va_tasks.values()]
    vaccounts[va] = {
        "n": len(va_rows),
        "accuracy": _macro(va_summaries, "accuracy")
        if len(va_tasks) == len(TASKS)
        else None,
    }

  return {
      "overall": overall,
      "categories": categories,
      "tasks": tasks,
      "single_hop": easy,
      "vaccounts": vaccounts,
  }


def _fmt(v) -> str:
  return "–" if v is None else f"{v:.4f}"


def print_report(report: dict, recall_k) -> None:
  has_recall = any(
      f"recall@{recall_k[0]}" in s for s in report["tasks"].values()
  )
  cols = ["Accuracy"] + ([f"R@{k}" for k in recall_k] if has_recall else [])
  header = f"{'Task':<32}{'N':>7}" + "".join(f"{c:>10}" for c in cols)
  line = "-" * len(header)

  def row(name, n, s):
    vals = [s.get("accuracy")] + (
        [s.get(f"recall@{k}") for k in recall_k] if has_recall else []
    )
    return f"{name:<32}{n:>7}" + "".join(f"{_fmt(v):>10}" for v in vals)

  print(line)
  print(header)
  print(line)
  for cat, cat_tasks in CATEGORIES.items():
    for t in cat_tasks:
      if t in report["tasks"]:
        s = report["tasks"][t]
        print(row(TASK_NAMES[t], s["n"], s))
    if cat in report["categories"]:
      print(row(f"  {cat}", "", report["categories"][cat]))
  if report["overall"]:
    print(line)
    n = sum(s["n"] for s in report["tasks"].values())
    print(row("Overall", n, report["overall"]))
  if report["single_hop"]:
    print(line)
    print("Single-hop questions (not part of the benchmark score)")
    for t, s in report["single_hop"].items():
      print(row(t, s["n"], s))
  print(line)

  missing = sum(s["missing_predictions"] for s in report["tasks"].values())
  unjudged = sum(s["unjudged"] for s in report["tasks"].values())
  if missing:
    print(f"Missing predictions (counted as wrong): {missing}")
  if unjudged:
    print(
        f"Open-generation questions without a judgment: {unjudged}. "
        "Export prompts with --export_judge_prompts, run your judge, and pass "
        "the results via --judgments."
    )


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--predictions", required=True, help="Predictions JSONL.")
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--data_dir", help="Life-Bench raw directory.")
  src.add_argument(
      "--hf_repo",
      nargs="?",
      const="google/life-bench",
      help="Download the metadata needed for scoring from the Hugging Face "
      "Hub (default repo: google/life-bench).",
  )
  p.add_argument("--judgments", help="Judge results JSONL.")
  p.add_argument(
      "--export_judge_prompts",
      metavar="PATH",
      help="Write judge prompts for unjudged open-generation questions.",
  )
  p.add_argument("--output", help="Write the full report as JSON.")
  p.add_argument("--tasks", nargs="+", help="Subset of task keys.")
  p.add_argument("--vaccounts", nargs="+", help="Subset of Vaccounts.")
  p.add_argument(
      "--include_easy",
      action="store_true",
      help="Also score the single-hop `*_easy` questions.",
  )
  p.add_argument(
      "--recall_k", nargs="+", type=int, default=list(DEFAULT_RECALL_K)
  )
  args = p.parse_args(argv)

  data_dir = args.data_dir or data.download_metadata(args.hf_repo)
  questions = data.load_questions(
      data_dir, args.tasks, args.vaccounts, args.include_easy
  )
  image_to_event = data.load_image_to_event(data_dir)
  predictions = data.load_predictions(args.predictions)
  judgments = data.load_judgments(args.judgments)

  rows = score_questions(
      questions, predictions, judgments, image_to_event, args.recall_k
  )

  if args.export_judge_prompts:
    by_id = {q["sample_id"]: q for q in questions}
    prompts = []
    for r in rows:
      if r["correct"] is None:
        q = by_id[r["sample_id"]]
        prompts.append({
            "sample_id": q["sample_id"],
            "prompt": build_judge_prompt(
                q["question"],
                q["answer"],
                predictions[q["sample_id"]].get("prediction"),
            ),
        })
    data.write_jsonl(args.export_judge_prompts, prompts)
    print(f"Wrote {len(prompts)} judge prompts to {args.export_judge_prompts}")

  report = aggregate(rows, args.recall_k)
  print_report(report, args.recall_k)

  if args.output:
    report["samples"] = rows
    with open(args.output, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
