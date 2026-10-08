# Sample Data

A slice of the `david` Vaccount from [google/life-bench](https://huggingface.co/datasets/google/life-bench), in the same file layout as the full dataset:

- `concepts/david/`: the 3 concepts of the Vaccount and their portraits.
- `events/`: one event (`david-e-0`, 6 historical images) and its `metadata.jsonl` records.
- `questions/`: one question per task (10 tasks + 3 `*_easy` files) and the query image `david-s-10.jpg`.

`../sample_predictions.jsonl` and `../sample_judgments.jsonl` are example inputs for `lifebench.evaluate` on this slice.
