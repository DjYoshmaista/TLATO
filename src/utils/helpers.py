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

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TOKENIZED_DATA_DIR = DATA_DIR / 'tokenized'
SYNTHETIC_DATA_DIR = DATA_DIR / 'synthetic'
DATA_REPO_FILE = DATA_DIR / 'data_repository.csv.zst' # From data_processing.py

# --- Checkpoint & Log Paths ---
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
LOG_DIR = PROJECT_ROOT / 'logs' # Consistent with logger.py
TEST_DATA_OUTPUT_DIR = PROJECT_ROOT / 'tests' / 'test_data_output' # For test artifacts

# --- Hardware Configuration ---
# Default device selection, can be overridden by specific components
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_MIN_COMPUTE_CAPABILITY = 7.0 # From gpu_switch.py usage

# --- Data Processing Configuration (from data_processing.py) ---
class DataProcessingConfig:
    SUPPORTED_FORMATS = ['.zst', '.csv', '.xlsx', '.xls', '.jsonl', '.txt', '.pdf'] # Added pdf from file_reader
    PROCESSING_CHUNK_SIZE = 1024 * 1024 # 1MB
    MAX_WORKERS = os.cpu_count() or 8 # Default to CPU count or 8
    REPO_FILE = DATA_REPO_FILE # Centralized path
    # NLTK data path can be configured if needed, default usually works
    # NLTK_DATA_PATH = PROJECT_ROOT / 'nltk_data'

# --- Data Loader Configuration (from data_loaders.py) ---
class DataLoaderConfig:
    # Path for EnhancedDataLoader - should point to processed/tokenized data
    ENHANCED_LOADER_DATA_DIR = TOKENIZED_DATA_DIR # Or PROCESSED_DATA_DIR depending on workflow
    ENHANCED_LOADER_BATCH_SIZE = 1024
    ENHANCED_LOADER_FILE_PATTERN = '*.pt' # Assuming tokenized data is saved as .pt

    # Path for SyntheticDataLoader
    SYNTHETIC_LOADER_DATA_DIR = SYNTHETIC_DATA_DIR
    SYNTHETIC_LOADER_BATCH_SIZE = 512
    SYNTHETIC_LOADER_FILE_PATTERN = '*.jsonl'

# --- Synthetic Data Generation Configuration (from synthetic_data.py) ---
class SyntheticDataConfig:
    OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
    MODEL_NAME = 'qwen2.5-coder' # Or other suitable model
    TARGET_SAMPLES = 1_000_000 # Reduced for practicality, original was 1B
    BATCH_SIZE = 500 # Reduced for practicality
    DATA_FORMAT = 'jsonl'
    MAX_WORKERS = 4 # ThreadPoolExecutor workers

# --- Training Configuration (from training.txt) ---
class TrainingConfig:
    MAX_EPOCHS = 50 # Reduced from 100
    INITIAL_LR = 3e-4
    WEIGHT_DECAY = 0.01
    # Pruning settings
    PRUNE_INTERVAL_EPOCHS = 5
    PRUNE_AMOUNT = 0.20 # 20%
    # Checkpointing
    CHECKPOINT_DIR = CHECKPOINT_DIR # Centralized path
    CHECKPOINT_INTERVAL_BATCH_PERCENT = 0.10 # Save every 10% of batches in an epoch
    # Metrics filename prefix/pattern could be added here
    METRICS_FILENAME_PREFIX = "training_metrics"

# --- Neural Zone Configuration (from neural_zones.txt) ---
class ZoneConfig:
    BASE_ACTIVATION = 0.1
    MAX_CONNECTIONS = 5
    ACTIVATION_THRESHOLD = 0.5 # Note: This wasn't used in the original update_activation
    LINK_STRENGTH = 0.8

# --- Semantic Labeler Configuration (from semantic_labeler.txt) ---
class LabelerConfig:
    SIMILARITY_THRESHOLD = 0.7
    MAX_RECURSION_DEPTH = 5
    LABELING_MODE = 'hierarchical' # Options: 'hierarchical', 'linear', 'hybrid' (as per original menu)
    # Model can also be configured
    TOKENIZER_MODEL = 'bert-base-uncased'
    EMBEDDING_MODEL = 'bert-base-uncased'

# --- Core/Attention Configuration (from test_core.txt - implied CONFIG) ---
# This was less defined, adding placeholders if needed
class CoreConfig:
    # Example: Default attention heads, dropout rates, etc.
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.1

# --- Test Configuration ---
class TestConfig:
    # Configuration specific to running tests
    # e.g., paths for test data fixtures, specific thresholds
    TEST_ARTIFACT_DIR = TEST_DATA_OUTPUT_DIR

# --- Environment Variables ---
# Example: Load sensitive info like API keys from environment variables
# API_KEY = os.getenv('MY_API_KEY')
API_KEY = os.getenv('API_KEY_1', "api key not set")

# --- Dummy Data Loader Class ---
class DummyDataLoader:
    """A simple iterable that yields dummy batches."""
    def __init__(self, batch_size=4, num_batches=10, input_dim=128, num_classes=6, device=DEFAULT_DEVICE):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device

    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate dummy input based on model expectation
            inputs = torch.randn(self.batch_size, self.input_dim, device=self.device)
            # Generate dummy targets (shape depends on loss function)
            targets = torch.randn(self.batch_size, self.num_classes, device=self.device)
            yield inputs, targets

    def __len__(self):
        return self.num_batches

    # --- Save and Load State Functions ---
    def save_state(self, obj, file_path):
        """
        Saves the state of an object to a file using torch serialization.
        Args:
            obj: The object to save (e.g., model, configuration, etc.).
            file_path: The file path where the state will be saved.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        torch.save(obj, file_path)
        print(f"State saved to {file_path}")

    def load_state(self, file_path):
        """
        Loads the state of an object from a file using torch serialization.
        Args:
            file_path: The file path from which the state will be loaded.
        Returns:
            The loaded object.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        obj = torch.load(file_path)
        print(f"State loaded from {file_path}")
        return obj

    # --- Dummy Input Function ---
    def dummy_input(self, input_shape, device=None):
        """
        Generates a dummy input tensor for testing or debugging purposes.
        Args:
            input_shape: Tuple specifying the shape of the input tensor.
            device: The device to place the tensor on (e.g., 'cpu', 'cuda').
                    Defaults to the DEFAULT_DEVICE.
        Returns:
            A dummy input tensor.
        """
        if device is None:
            device = DEFAULT_DEVICE
        return torch.randn(*input_shape, device=device)
    
# # Example usage of DummyDataLoader
# dummy_loader = DummyDataLoader(batch_size=4, num_batches=10, input_dim=128, num_classes=6)
# for inputs, targets in dummy_loader:
#     print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
#     print(f"Inputs: {inputs}, Targets: {targets}")

import io
import pandas as pd
from ..utils.logger import configure_logging, log_statement
import zstandard as zstd
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.hashing import generate_data_hash
from ..utils.compression import stream_compress_lines, ZSTD_COMPRESSION_LEVEL, ZSTD_THREADS
import os
import sys
from ..data.constants import (COL_FILEPATH, COL_STATUS, COL_ERROR, COL_DATA_HASH,
                                STATUS_PROCESSING, STATUS_PROCESSED, STATUS_ERROR)
from ..data.readers import (FileReader, PDFReader, CSVReader, TXTReader, JSONReader, JSONLReader, ExcelReader)
from ..utils.config import PROCESSED_DATA_DIR # Assuming self.output_dir = PROCESSED_DATA_DIR

configure_logging()

# --- Helper Function for Saving DataFrame to Compressed Parquet (Needs Integration) ---
# This function is added based on the comment in the user prompt suggesting parquet output
# It should be placed within the DataProcessor class or imported if defined elsewhere.

def save_dataframe_to_parquet_zst(self, df: pd.DataFrame, output_path: Path):
    """Saves a pandas DataFrame to a Zstandard-compressed Parquet file."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if using cuDF DataFrame and convert if necessary before saving with pandas
        if 'cudf' in sys.modules and isinstance(df, sys.modules['cudf'].DataFrame):
            log_statement(loglevel='debug', logstatement="Converting cuDF DataFrame to Pandas before Parquet save.", main_logger=str(__name__))
            df_pandas = df.to_pandas()
        else:
            df_pandas = df # Assume it's already a pandas DataFrame

        # Save using pandas to_parquet with zstd compression
        df_pandas.to_parquet(output_path, compression='zstd', engine='pyarrow') # engine='fastparquet' is another option

        log_statement(loglevel='info', logstatement=f"DataFrame saved to compressed Parquet: {output_path}", main_logger=str(__name__))
        return True
    except ImportError:
        log_statement(loglevel='error', logstatement="Error saving to Parquet: 'pyarrow' or 'fastparquet' library not found. Please install one.", main_logger=str(__name__))
        return False
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to save DataFrame to compressed Parquet {output_path}: {e}", main_logger=str(__name__), exc_info=True)
        # Clean up potentially incomplete file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return False

# --- Helper Function for Saving String to Compressed File (Needs Integration) ---
def compress_string_to_file(self, text_content: str, output_path: Path):
    """Compresses a string content into a Zstandard file using streaming."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def text_generator():
            # Yield the content line by line or in chunks if very large
            # For simplicity, yielding the whole string (ensure it fits memory)
            # For very large strings, consider splitting into lines or chunks
            yield text_content

        # Use stream_compress_lines utility function
        stream_compress_lines(str(output_path), text_generator())

        log_statement(loglevel='debug', logstatement=f"Stream compressed string to '{output_path}'", main_logger=str(__name__))
        return True
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error during streaming compression of string to '{output_path}': {e}", main_logger=str(__name__), exc_info=True)
        # Clean up potentially incomplete output file
        if output_path.exists():
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

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

def save_state(obj, file_path):
    DummyDataLoader.save_state(obj, file_path)
def load_state(file_path):
    DummyDataLoader.load_state(file_path)
def dummy_input(input_shape, device=None):
    """
    Generates a dummy input tensor for testing or debugging purposes.
    Args:
        input_shape: Tuple specifying the shape of the input tensor.
        device: The device to place the tensor on (e.g., 'cpu', 'cuda').
                Defaults to the DEFAULT_DEVICE.
    Returns:
        A dummy input tensor.
    """
    return DummyDataLoader.dummy_input(input_shape, device)

# def save_state(obj, file_path):
#     """
#     Saves the state of an object to a file using torch serialization.
#     Args:
#         obj: The object to save (e.g., model, configuration, etc.).
#         file_path: The file path where the state will be saved.
#     """
#     file_path = Path(file_path)
#     file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
#     torch.save(obj, file_path)
#     print(f"State saved to {file_path}")

# def load_state(file_path):
#     """
#     Loads the state of an object from a file using torch serialization.
#     Args:
#         file_path: The file path from which the state will be loaded.
#     Returns:
#         The loaded object.
#     """
#     file_path = Path(file_path)
#     if not file_path.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")
#     obj = torch.load(file_path)
#     print(f"State loaded from {file_path}")
#     return obj

# def dummy_input(input_shape, device=None):
#     """
#     Generates a dummy input tensor for testing or debugging purposes.
#     Args:
#         input_shape: Tuple specifying the shape of the input tensor.
#         device: The device to place the tensor on (e.g., 'cpu', 'cuda').
#                 Defaults to the DEFAULT_DEVICE.
#     Returns:
#         A dummy input tensor.
#     """
#     if device is None:
#         device = DEFAULT_DEVICE
#     return torch.randn(*input_shape, device=device)

# # Example usage of dummy_input
# # input_shape = (4, 10, 128) # Example shape
# # dummy_tensor = dummy_input(input_shape)
# # print(f"Dummy tensor shape: {dummy_tensor.shape}")
# #     # Save the state
# #     save_state(self.model, self.test_filepath)
# #     save_state(self.optimizer, self.test_filepath)
# #     save_state(self.scheduler, self.test_filepath, extra_meta=extra_meta)
# #
# #     # Load the state
# #     loaded_model = load_state(self.test_filepath)
# #     loaded_optimizer = load_state(self.test_filepath)
# #     loaded_scheduler = load_state(self.test_filepath)
# #     # Check if the loaded state matches the saved state
# #     self.assertEqual(self.model.state_dict(), loaded_model.state_dict())
# #     self.assertEqual(self.optimizer.state_dict(), loaded_optimizer.state_dict())
#     self.assertEqual(self.scheduler.state_dict(), loaded_scheduler.state_dict())
#         self.assertEqual(loaded_meta['epoch'], epoch_to_save)
#         self.assertEqual(loaded_meta['info'], extra_meta['info'])
#         log_statement(loglevel=str("debug"), logstatement=str("Save and load cycle completed successfully."), main_logger=str(__name__))
#         # Clean up
#         if self.test_filepath.exists():
#             os.remove(self.test_filepath)
#         log_statement(loglevel=str("debug"), logstatement=str(f"Removed test checkpoint file: {self.test_filepath}"), main_logger=str(__name__))
#         log_statement(loglevel=str("debug"), logstatement=str("Save and load cycle completed successfully."), main_logger=str(__name__))