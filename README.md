<h2 align="center">Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2602.19001"><img src="https://img.shields.io/badge/arXiv-2602.19001-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/google/life-bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache%202.0-green.svg" alt="Code License"></a>
  <a href="https://creativecommons.org/licenses/by-sa/4.0/legalcode"><img src="https://img.shields.io/badge/Data-CC--BY--SA--4.0-lightgrey.svg" alt="Data License"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=1PT3EQoAAAAJ">Xia Hu</a>,
  <a href="https://scholar.google.com/citations?user=FxEDj4wAAAAJ">Honglei Zhuang</a>,
  <a href="https://scholar.google.com/citations?user=OwEFVw4AAAAJ">Brian Potetz</a>,
  <a href="https://scholar.google.com/citations?user=luv0xMIAAAAJ">Alireza Fathi</a>,
  Bo Hu,
  <a href="https://scholar.google.com/citations?user=5gS9W3wAAAAJ">Babak Samari</a>,
  <a href="https://scholar.google.com/citations?user=dJXeYCoAAAAJ">Howard Zhou</a>
  <br>
  Google DeepMind
</p>

## Overview

Life-Bench is a fully synthetic, human-verified multimodal benchmark of 11,811 question–answer pairs across 10 tasks, organized by required evidence scope: Concept Identification, Event Understanding, and Aggregated Reasoning. Underlying these tasks, the personal context comprises social networks and multimodal lifelogs organized into 10 isolated virtual accounts (Vaccounts) to simulate individual users' archives; every question and image has been verified through full-coverage human annotation. All data in Life-Bench (images, social networks, and event histories) is synthetically generated, enabling public release without exposing real users' data.

Please refer to the [paper](https://arxiv.org/abs/2602.19001) for the benchmark design, construction pipeline, and quality control. This repository contains the evaluation code; the dataset is hosted on the Hugging Face Hub at [google/life-bench](https://huggingface.co/datasets/google/life-bench).

## Tasks

| Category | Task | Task Key | Input Format | Answer Format | # Questions |
| :--- | :--- | :--- | :---: | :---: | ---: |
| Concept Identification | Text Concept QA | `text_concept_qa` | Text | Multiple-Choice | 163 |
| | Visual Concept Recognition | `visual_concept_recognition` | Text + Image | Binary | 1,613 |
| | Concept VQA | `concept_vqa` | Text + Image | Multiple-Choice | 885 |
| Event Understanding | Scene and Activity | `scene_and_activity` | Text | Open Generation | 2,265 |
| | Direct Person-Centric | `direct_person_centric` | Text | Open Generation | 2,245 |
| | Relational Person-Centric | `relational_person_centric` | Text | Open Generation | 1,812 |
| | Fine-Grained Scene | `fine_grained_scene` | Text | Open Generation | 1,981 |
| Aggregated Reasoning | Preference and Persona | `preference_and_persona` | Text | Open Gen (Option-guided) | 295 |
| | Frequency and Counting | `frequency_and_counting` | Text | Open Generation | 265 |
| | Relational Temporal Reasoning | `relational_temporal_reasoning` | Text | Open Generation | 287 |
| **Total** | | | | | **11,811** |

## Evaluation Results

Our primary metric is accuracy: multiple-choice and binary questions use exact match, while Gemini 3.6 Flash judges open-ended responses against the ground-truth answer using LLM-as-a-Judge. We additionally report recall to assess retrieval quality.

**Accuracy.** Retrieval-based methods are compared at context size $k = 5$, all using Gemma-3 12B as backbone; the best and second-best scores among them are in bold and underlined respectively. Reference settings contextualize retrieval performance and are generated with Gemini-3.5-Flash except Random: Close-book answers without personal context; Oracle receives only the gold supporting evidence (Oracle-Gemma: with the Gemma backbone); Full-context places an entire Vaccount history in the prompt.

| Method | Text Concept QA | Visual Concept Recognition | Concept VQA | Scene & Activity | Direct Person-Centric | Relational Person-Centric | Fine-Grained Scene | Preference & Persona | Relational Temporal | Frequency & Counting | Overall |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Random | 0.2500 | 0.5000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2003 | 0.0000 | 0.0000 | 0.1200 |
| Close-book | 0.2454 | 0.5111 | 0.2696 | 0.0848 | 0.0539 | 0.1341 | 0.1560 | 0.4373 | 0.0662 | 0.0755 | 0.2034 |
| Full-context | 0.9730 | 0.9722 | 0.9508 | 0.9143 | 0.9095 | 0.9141 | 0.9014 | 0.8856 | 0.8523 | 0.3823 | 0.8655 |
| Oracle | 0.9816 | 0.9826 | 0.9570 | 0.9324 | 0.9292 | 0.9269 | 0.9100 | – | – | – | 0.9457\* |
| Oracle-Gemma | 0.8527 | 0.8698 | 0.6170 | 0.8093 | 0.7007 | 0.7130 | 0.7410 | – | – | – | 0.7576\* |
| BM25 | **0.8650** | 0.8320 | <u>0.6249</u> | 0.6698 | 0.3648 | 0.4923 | 0.4962 | 0.5593 | 0.1080 | 0.1472 | 0.5159 |
| RAG-Cap | <u>0.8405</u> | 0.7650 | 0.5236 | 0.4843 | 0.2895 | 0.3786 | **0.5671** | <u>0.7051</u> | 0.1428 | 0.1623 | 0.4859 |
| RAP-Gemma | 0.7651 | 0.7498 | 0.4744 | 0.3043 | 0.1688 | 0.2467 | 0.5172 | **0.7119** | 0.1533 | 0.0755 | 0.4167 |
| R2P-Gemma | 0.7055 | 0.6943 | 0.3559 | 0.3244 | 0.2664 | 0.3488 | 0.4338 | 0.5831 | 0.1715 | 0.1094 | 0.3993 |
| HippoRAG2 | 0.8098 | **0.8494** | **0.6542** | 0.3978 | 0.2094 | 0.3201 | 0.4356 | 0.5559 | 0.1672 | 0.1623 | 0.4562 |
| LifeGraph ($d=2$) | 0.7587 | <u>0.8332</u> | 0.5318 | **0.7242** | **0.5092** | <u>0.5193</u> | 0.5157 | 0.6746 | <u>0.3240</u> | <u>0.1785</u> | <u>0.5569</u> |
| LifeGraph ($d=3$) | 0.7914 | 0.8278 | 0.5260 | <u>0.7228</u> | <u>0.4984</u> | **0.5348** | <u>0.5253</u> | 0.6524 | **0.3275** | **0.1976** | **0.5604** |

\* Oracle overall averages over seven concept and event tasks since aggregate does not have subset golden reference.

**Event-level Recall@3/5**, where retrieving any image from a gold event counts as a hit.

| Method | Concept R@3 | Concept R@5 | Event R@3 | Event R@5 | Overall R@3 | Overall R@5 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| BM25 | 0.4626 | **0.5330** | 0.4785 | 0.5371 | 0.4717 | 0.5353 |
| RAG-Cap | 0.4307 | 0.4808 | 0.3665 | 0.4580 | 0.3940 | 0.4678 |
| R2P | 0.4070 | 0.4238 | 0.3823 | 0.4299 | 0.3929 | 0.4273 |
| RAP | 0.4354 | 0.4653 | 0.4065 | 0.4572 | 0.4188 | 0.4607 |
| HippoRAG2 | 0.4465 | <u>0.5128</u> | 0.3042 | 0.4065 | 0.3652 | 0.4521 |
| LifeGraph ($d=2$) | <u>0.4748</u> | 0.5009 | **0.5309** | **0.6021** | <u>0.5068</u> | **0.5587** |
| LifeGraph ($d=3$) | **0.4844** | 0.4991 | <u>0.5271</u> | <u>0.5963</u> | **0.5088** | <u>0.5546</u> |

## Dataset

The dataset is hosted at [google/life-bench](https://huggingface.co/datasets/google/life-bench) as Parquet tables (`questions`, `history`, `concepts`) and as a raw file tree:

```
google/life-bench/
├── concepts/{vaccount}/
│   ├── concepts.json                    #   concepts of the Vaccount
│   └── portraits/*.jpg                  #   concept portraits
├── events/
│   ├── metadata.jsonl                   #   one record per historical image (382 events)
│   └── images/*.jpg                     #   2,479 historical images
├── questions/
│   ├── <task>.jsonl                     #   10 task files + 3 `<task>_easy.jsonl` files
│   └── images/*.jpg                     #   375 query images
└── metadata/statistics.json
```

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/life-bench", repo_type="dataset",
                  allow_patterns=["concepts/*", "events/*", "questions/*", "metadata/*"])
```

Each question record has the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | `string` | Question id, e.g. `"sun-ja_concept_vqa_120"`. |
| `vaccount` | `string` | Vaccount the question belongs to. |
| `task` | `string` | Task key (see the Tasks table and the `_easy` variants below). |
| `question` | `string` | Question text (including options for multiple-choice questions). |
| `answer` | `string` | Reference answer. |
| `answer_type` | `string` | `"multiple_choice"`, `"binary"`, or `"open_generation"`. |
| `question_image_id` | `string` | Query image id for text+image questions; empty for text-only questions. |
| `question_image_path` | `string` | Relative path of the query image, e.g. `"questions/images/sun-ja-s-7.jpg"`; empty for text-only questions. |
| `reference_ids` | `list[string]` | Gold supporting `event_id` and `concept_id` values for Concept Identification and Event Understanding questions; `[]` for Aggregated Reasoning questions. |

The `history` (`events/metadata.jsonl`) and `concepts` (`concepts/{vaccount}/concepts.json`) schemas, and loading with `datasets`, are documented on the [dataset card](https://huggingface.co/datasets/google/life-bench). A slice of one Vaccount in the same layout is included in [`examples/sample_data/`](examples/sample_data/).

### Additionally Released Single-hop Questions

In addition to the 11,811 questions in the Tasks table, we additionally provide 1,647 single-hop direct concept recognition questions (139 for Text Concept QA, 1,032 for Visual Concept Recognition, and 476 for Concept VQA) as diagnostic baselines to monitor model fallback, bringing the total released benchmark to 13,458 QA pairs. These are excluded from the primary evaluation because all evaluated retrieval approaches already near-saturate on single-hop concept lookup, limiting their discriminative value for method comparison; they are instead provided as a diagnostic set for verifying that more complex personalization methods do not regress on basic entity recognition.

They are stored separately with the `task` values `text_concept_qa_easy`, `visual_concept_recognition_easy`, and `concept_vqa_easy` (files `questions/<task>_easy.jsonl`). Results on Life-Bench should be reported on the 11,811 questions in the Tasks table.

## Installation

```bash
git clone https://github.com/google-deepmind/life-bench.git
cd life-bench
pip install -e .
```

No dependencies are required. `pip install -e ".[hf]"` adds `huggingface_hub` for `--hf_repo`.

## Evaluation

Multiple-choice and binary questions are scored by exact match: the response is cleaned (answer labels such as `Answer:`, markdown emphasis, brackets and trailing punctuation are removed) and must reduce to a single option letter or `Yes`/`No`. Responses that do not are not exact-match scorable; they are counted and listed in the report, and are judged with the same LLM-as-a-Judge prompt as open-generation questions rather than being scored as wrong. Open-generation questions are judged against the ground-truth answer with the LLM-as-a-Judge prompt in [`lifebench/judge.py`](lifebench/judge.py), using a judge model of your choice. The overall score is the average of the 10 per-task accuracies. Event-level Recall@k is computed over the 7 Concept Identification and Event Understanding tasks, where retrieving any image from a gold event counts as a hit.

**1. Predictions.** Run your system on each question and write one JSON record per line:

```json
{"sample_id": "david_scene_and_activity_0", "prediction": "The gathering was a trip with Zosime ...", "retrieved_ids": ["david-e-0-0", "david-e-0-3", "david-c-zosime", "david-c-david", "david-e-30-1"]}
```

| Field | Required | Description |
| :--- | :---: | :--- |
| `sample_id` | ✓ | Question id. |
| `prediction` | ✓ | Model response. For multiple-choice and binary questions it should reduce to the option letter (`A`–`D`) or `Yes`/`No` (e.g. `C`, `(C)`, `Answer: C`); otherwise it is judged like an open-generation response. |
| `retrieved_ids` | | Retrieved `image_id` and `concept_id` values in rank order. Required for Recall@k. |

**2. Export judge prompts.** Score exact-match questions and write the judge prompt of every question that needs a judgment (open-generation, and unparseable multiple-choice / binary) to `judge_prompts.jsonl` as `{"sample_id": ..., "prompt": ...}`:

```bash
python -m lifebench.evaluate --predictions predictions.jsonl --data_dir /path/to/life-bench \
    --export_judge_prompts judge_prompts.jsonl
```

`--data_dir` is the raw file tree; only `questions/*.jsonl` and `events/metadata.jsonl` are read. `--hf_repo` downloads these files from the Hub instead.

**3. Judge.** Send each prompt to your judge model. The prompt asks for a JSON reply `{"Result": "Correct" or "Wrong", "Reason": "..."}`; save the replies with their `sample_id`:

```json
{"sample_id": "david_scene_and_activity_0", "Result": "Correct", "Reason": "..."}
```

**4. Report.**

```bash
python -m lifebench.evaluate --predictions predictions.jsonl --data_dir /path/to/life-bench \
    --judgments judgments.jsonl --output report.json
```

```
---------------------------------------------------------------------------------------------
Category                 Task                                 N  Accuracy       R@3       R@5
---------------------------------------------------------------------------------------------
Concept Identification   Text Concept QA                      1    1.0000    1.0000    1.0000
                         Visual Concept Recognition           1    1.0000    0.5000    1.0000
                         Concept VQA                          1    0.0000    0.5000    1.0000
                         Average                                   0.6667    0.6667    1.0000
---------------------------------------------------------------------------------------------
Event Understanding      Scene and Activity                   1    1.0000    0.6667    1.0000
                         Direct Person-Centric                1    1.0000    0.6667    1.0000
                         Relational Person-Centric            1    1.0000    0.6667    0.6667
                         Fine-Grained Scene                   1    0.0000    0.3333    1.0000
                         Average                                   0.7500    0.5833    0.9167
---------------------------------------------------------------------------------------------
Aggregated Reasoning     Preference and Persona               1    1.0000         –         –
                         Frequency and Counting               1    0.0000         –         –
                         Relational Temporal Reasoning        1    1.0000         –         –
                         Average                                   0.6667         –         –
---------------------------------------------------------------------------------------------
Overall                                                      10    0.7000    0.6190    0.9524
---------------------------------------------------------------------------------------------
```

`report.json` contains the per-task, per-category, overall, and per-Vaccount scores, the per-question results, and the `sample_id`s of unparseable multiple-choice / binary predictions (`unparseable_samples`). `--tasks` and `--vaccounts` select a subset; `--include_easy` additionally scores the single-hop questions in a separate block; `--recall_k` sets the recall cutoffs (default `3 5`).

The output above is produced by the files in [`examples/`](examples/):

```bash
python -m lifebench.evaluate --predictions examples/sample_predictions.jsonl \
    --data_dir examples/sample_data --judgments examples/sample_judgments.jsonl
```

## Citation

```bibtex
@misc{hu2026lifebench,
      title={Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition},
      author={Xia Hu and Honglei Zhuang and Brian Potetz and Alireza Fathi and Bo Hu and Babak Samari and Howard Zhou},
      year={2026},
      eprint={2602.19001},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.19001},
}
```

## License & Disclaimer

Copyright 2026 Google LLC

All software in this repository is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this software except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

The Life-Bench data (images, personal-context records, and question–answer pairs), including the sample data in this repository, is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). You may obtain a copy of the CC BY-SA 4.0 license at: https://creativecommons.org/licenses/by-sa/4.0/legalcode. Life-Bench is released for evaluation purposes.

All data in Life-Bench is synthetically generated; the generated accounts map to no real individual. To retain realism, generation is seeded with textual captions of images from a CC-licensed subset of the YFCC100M dataset (Thomee et al., 2016). YFCC source images serve only as captioning inputs and never enter Vaccounts, and the captions are not released.

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC BY-SA 4.0 licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.
