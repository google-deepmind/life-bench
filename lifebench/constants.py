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

"""Task definitions for Life-Bench."""

# Task key -> display name, in the order of the Tasks table.
TASK_NAMES = {
    "text_concept_qa": "Text Concept QA",
    "visual_concept_recognition": "Visual Concept Recognition",
    "concept_vqa": "Concept VQA",
    "scene_and_activity": "Scene and Activity",
    "direct_person_centric": "Direct Person-Centric",
    "relational_person_centric": "Relational Person-Centric",
    "fine_grained_scene": "Fine-Grained Scene",
    "preference_and_persona": "Preference and Persona",
    "frequency_and_counting": "Frequency and Counting",
    "relational_temporal_reasoning": "Relational Temporal Reasoning",
}

TASKS = list(TASK_NAMES)

CATEGORIES = {
    "Concept Identification": [
        "text_concept_qa",
        "visual_concept_recognition",
        "concept_vqa",
    ],
    "Event Understanding": [
        "scene_and_activity",
        "direct_person_centric",
        "relational_person_centric",
        "fine_grained_scene",
    ],
    "Aggregated Reasoning": [
        "preference_and_persona",
        "frequency_and_counting",
        "relational_temporal_reasoning",
    ],
}

# Single-hop questions released in addition to the benchmark; excluded from
# the primary evaluation.
EASY_TASKS = [
    "text_concept_qa_easy",
    "visual_concept_recognition_easy",
    "concept_vqa_easy",
]

# Tasks with gold supporting evidence (`reference_ids`), used for recall.
RECALL_TASKS = (
    CATEGORIES["Concept Identification"] + CATEGORIES["Event Understanding"]
)

ANSWER_TYPES = ("multiple_choice", "binary", "open_generation")

DEFAULT_RECALL_K = (3, 5)
