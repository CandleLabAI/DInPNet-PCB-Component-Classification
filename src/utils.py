# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utilities class and functions for train, test and inference.
"""

import os
from dataclasses import dataclass
import torch
import config

@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed = config.SEED # seed number to set the state of all random number generators
    cudnn_benchmark_enabled = config.CUDNN_BENCHMARK_ENABLED # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic = config.CUDNN_DETERMINISTIC  # make cudnn deterministic (reproducible training)

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS
    init_learning_rate = config.INIT_LEARNING_RATE # initial learning rate for lr scheduler
    log_interval = config.LOG_INTERVAL
    test_interval = config.TEST_INTERVAL
    data_root = config.DATA_ROOT
    num_workers = config.NUM_WORKERS
    device = config.DEVICE

def setup_system(system_config):
    """
    Setup function for reproducibility
    """
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

def save_model(model, device, model_dir='../weights', model_file_name='best.pt'):
    """
    Helper function to save model
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_file_name)
    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')
    # save the state_dict
    torch.save(model.state_dict(), model_path)
    if device == 'cuda':
        model.to('cuda')

def load_model(model, model_dir='../weights', model_file_name='best.pt'):
    """
    Helper function to load model
    """
    model_path = os.path.join(model_dir, model_file_name)
    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path))
    return model
