"""pytest configuration of the solution's test suite: one GPU.

Kaggle's "T4 x2" machines show two GPUs. The measured runs use one (harness/run_task.py pins
it), and so must the tests: with two visible devices transformers' Trainer wraps the model in
DataParallel, which splits every batch across replicas, and the subprocess tests (test_parity,
test_switches, test_end_to_end) inherit whatever this process sees. The default is set before
torch initializes CUDA (pytest imports conftest.py first) and only when the caller has not
chosen a device; `CUDA_VISIBLE_DEVICES=1 python -m pytest solution` still works.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
