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

"""LLM-as-a-Judge prompt for open-generation questions."""

JUDGE_PROMPT = """You are an expert evaluator for a retrieval-augmented generation system. Your primary goal is to determine if the system retrieved the correct information to answer the question, not to judge the completeness of the generated text.

**Evaluation Criteria:**
- An answer is "Correct" if the facts it presents are accurate and align with the ground truth.
- An answer is NOT "Wrong" simply because it is less detailed than the ground truth.
- An answer is "Wrong" if it contains factually incorrect information, contradicts the ground truth, or fails to cover the critical information required to answer the question.

**Instructions:**
1. Read the Question, Ground Truth Answer, and the Candidate Answer.
2. Compare the factual statements in the Candidate Answer against the Ground Truth.
3. Decide "Correct" or "Wrong". The key is accuracy, not completeness.

**Question:** {question}
**GroundTruth Answer:** {gt}
**Candidate Answer:** {candidate_answer}

Reply in JSON: {{ "Result": "Correct" or "Wrong", "Reason": "..." }}"""


def build_judge_prompt(question: str, answer: str, prediction) -> str:
  return JUDGE_PROMPT.format(
      question=question,
      gt=answer,
      candidate_answer="" if prediction is None else str(prediction),
  )
