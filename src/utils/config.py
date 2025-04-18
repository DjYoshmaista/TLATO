# src/utils/config.py
"""
Central Configuration File

Consolidates settings, paths, hyperparameters, and other configuration
variables used throughout the application.
"""

import os
from pathlib import Path
import torch

# --- Project Root ---
# Assumes this file is in project_root/src/utils
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# --- Base Data Path ---
# Root directory for discovering raw data files recursively
BASE_DATA_DIR = PROJECT_ROOT / 'data' # Default to the main 'data' folder

# --- Data Paths (Subdirectories within BASE_DATA_DIR or elsewhere if needed) ---
RAW_DATA_DIR = BASE_DATA_DIR / 'raw' # Can still be used if structure is flat
PROCESSED_DATA_DIR = BASE_DATA_DIR / 'processed'
TOKENIZED_DATA_DIR = BASE_DATA_DIR / 'tokenized'
SYNTHETIC_DATA_DIR = BASE_DATA_DIR / 'synthetic'
DATA_REPO_FILE = BASE_DATA_DIR / 'data_repository.csv.zst' # Repository file location

# --- Checkpoint & Log Paths ---
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
LOG_DIR = PROJECT_ROOT / 'logs' # Consistent with logger.py
TEST_DATA_OUTPUT_DIR = PROJECT_ROOT / 'tests' / 'test_data_output' # For test artifacts

# --- Hardware Configuration ---
# Default device selection, can be overridden by specific components
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MIN_COMPUTE_CAPABILITY = 7.0 # From gpu_switch.py usage

# --- Compression Configuration ---
COMPRESSION_ENABLED = True # Global flag to enable/disable zstd compression output
COMPRESSION_LEVEL = 22 # Max compression level for zstandard (1-22)

# --- Data Processing Configuration ---
class DataProcessingConfig:
    SCAN_FILE_EXTENSIONS = ['.txt', '.csv', '.json', '.jsonl', '.md', '.py', '.log', '.pdf', '.xlsx', '.xls', '.zst']
    PROCESSING_CHUNK_SIZE = 1024 * 1024
    MAX_WORKERS = os.cpu_count() or 16
    REPO_FILE = DATA_REPO_FILE # Use the centrally defined path
    # Add specific processing options if needed, e.g., text cleaning regex
    TEXT_CLEANING_REGEX = r'[^\w\s\-\.]' # Keep word chars, space, hyphen, dot

# --- Data Loader Configuration ---
class DataLoaderConfig:
    ENHANCED_LOADER_DATA_DIR = TOKENIZED_DATA_DIR
    ENHANCED_LOADER_BATCH_SIZE = 1024
    ENHANCED_LOADER_FILE_PATTERN = '*.pt.zst' if COMPRESSION_ENABLED else '*.pt'
    SYNTHETIC_LOADER_DATA_DIR = SYNTHETIC_DATA_DIR
    SYNTHETIC_LOADER_BATCH_SIZE = 512
    SYNTHETIC_LOADER_FILE_PATTERN = '*.jsonl'

# --- Synthetic Data Generation Configuration (from synthetic_data.py) ---
# Assuming synthetic.py handles its own output format for now
class SyntheticDataConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else DEFAULT_DEVICE)
    OLLAMA_ENDPOINT = 'http://0.0.0.0:11434/api/generate'
    GENERATION_MODEL_NAME = 'qwen2.5-coder' # Example, adjust as needed
    CORRECTION_MODEL_NAME = 'gemma3:4b'
    ENABLE_API_CORRECTION = True
    JSON_CORRECTION_RETRIES = 1
    LIST_CORRECTION_RETRIES = 1
    API_TIMEOUT_SECONDS = 4200
    TARGET_SAMPLES = 100
    BATCH_SIZE = 10
    DATA_FORMAT = 'jsonl'
    MAX_WORKERS = 16
    SAVE_RETRIES = 6
    RETRY_DELAY_SECONDS = 2
    ADJUSTED_SUBDIR = 'adjusted_samples'

# --- Training Configuration (from training.txt) ---
class TrainingConfig:
    MAX_EPOCHS = 50 # Reduced from 100
    INITIAL_LR = 3e-4
    WEIGHT_DECAY = 0.01
    # Example Model/Loss Params (adjust based on actual model)
    MODEL_INPUT_DIM = 128 # Placeholder
    MODEL_NUM_CLASSES = 6 # Placeholder
    LOSS_FUNCTION = 'MSELoss' # Example: 'MSELoss' or 'CrossEntropyLoss'
    # Pruning settings
    PRUNE_INTERVAL_EPOCHS = 5
    PRUNE_AMOUNT = 0.20
    # Checkpointing
    CHECKPOINT_DIR = CHECKPOINT_DIR
    CHECKPOINT_INTERVAL_BATCH_PERCENT = 0.10
    PRUNE_METHOD = 'magnitude' # Example method, adjust as needed
    METRICS_FILENAME_PREFIX = "training_metrics"

# --- Neural Zone Configuration (from neural_zones.txt) ---
class ZoneConfig:
    BASE_ACTIVATION = 0.1
    MAX_CONNECTIONS = 5
    ACTIVATION_THRESHOLD = 0.5
    LINK_STRENGTH = 0.8

# --- Semantic Labeler Configuration (from semantic_labeler.txt) ---
class LabelerConfig:
    SIMILARITY_THRESHOLD = 0.7
    MAX_RECURSION_DEPTH = 5
    LABELING_MODE = 'hierarchical'
    TOKENIZER_MODEL = 'bert-base-uncased'
    EMBEDDING_MODEL = 'bert-base-uncased'

# --- Core/Attention Configuration (placeholders) ---
class CoreConfig:
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.1
    INPUT_DIM = 128
    NUM_CLASSES = 6

# --- Test Configuration ---
class TestConfig:
    TEST_ARTIFACT_DIR = TEST_DATA_OUTPUT_DIR

# --- Environment Variables ---
# Example: Load sensitive info like API keys from environment variables
API_KEY = os.getenv('API_KEY_1', "api key not set")

# --- Helper Function (Optional) ---
def get_config_value(key_path, default=None):
    """
    Retrieves a nested configuration value.
    Example: get_config_value('DataLoaderConfig.ENHANCED_LOADER_BATCH_SIZE')
    """
    try:
        parts = key_path.split('.')
        value = globals()
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part)
            if value is None:
                return default
        return value
    except (AttributeError, KeyError):
        return default