# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Configuration hyper-parameters for training, testing and inference
"""

import torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
CUDNN_BENCHMARK_ENABLED = True
CUDNN_DETERMINISTIC = True
SEED = 42

DATA_ROOT = "../data/"
IMAGE_SIZE = (64, 64)
NUM_OF_CLASSES = 6
TEST_INTERVAL = 1
LOG_INTERVAL = 5
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 10

INIT_LEARNING_RATE = 0.001
FACTOR = 0.1
PATIENCE = 3
THRESHOLD = 0.0001
VERBOSE = True
MIN_LR = 0.00001
