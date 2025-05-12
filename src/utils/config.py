# src/utils/config.py
"""
Central Configuration File

Consolidates settings, paths, hyperparameters, and other configuration
variables used throughout the application.
"""

import os
from pathlib import Path
import torch
from src.utils.logger import log_statement
import pandas as pd
import sys
import zstandard as zstd
from src.data.constants import *
from src.utils.compression import *

# --- Project Root ---
# Assumes this file is in project_root/src/utils
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# --- Base Data Path ---
# Root directory for discovering raw data files recursively
BASE_DATA_DIR = PROJECT_ROOT / 'data' # Default to the main 'data' folder

# --- Data Paths (Subdirectories within BASE_DATA_DIR or elsewhere if needed) ---
RAW_DATA_DIR = BASE_DATA_DIR / 'raw' # Can still be used if structure is flat
DATA_REPO_DIR = BASE_DATA_DIR / 'repositories'
PROCESSED_DATA_DIR = BASE_DATA_DIR / 'processed'
TOKENIZED_DATA_DIR = BASE_DATA_DIR / 'tokenized'
SYNTHETIC_DATA_DIR = BASE_DATA_DIR / 'synthetic'
DATA_REPO_FILE = DATA_REPO_DIR / 'data_repository.csv.zst' # Repository file location
INDEX_FILE = DATA_REPO_DIR / 'repository_index.json' # Central index file

# --- Define Keys For Index File Structure ---
INDEX_KEY_PATH = "path"
INDEX_KEY_METADATA = "metadata"
INDEX_KEY_CHILDREN = "children"

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
    SCAN_FILE_EXTENSIONS = ['.txt', '.html', '.rtf', '.doc', '.docx', '.zst', '.zstd', '.zip', '.gz', '.tar', '.tsv', '.yaml', '.xml', '.yml', '.csv', '.json', '.jsonl', '.md', '.py', '.log', '.pdf', '.xlsx', '.xls', '.zst']
    SUPPORTED_FORMATS = SCAN_FILE_EXTENSIONS
    PROCESSING_CHUNK_SIZE = 1024 * 1024 * 1024
    MAX_WORKERS = os.cpu_count() or 16
    REPO_FILE = DATA_REPO_FILE # Use the centrally defined path
    # Add specific processing options if needed, e.g., text cleaning regex
    TEXT_CLEANING_REGEX = r'[^\w\s\-\.]' # Keep word chars, space, hyphen, dot

# --- Data Loader Configuration ---
class DataLoaderConfig:
    ENHANCED_LOADER_DATA_DIR = TOKENIZED_DATA_DIR
    ENHANCED_LOADER_BATCH_SIZE = 1024
    ENHANCED_LOADER_FILE_PATTERN = ['*.pt' or '*.csv' or '*.json']
    ENHANCED_LOADER_NUM_WORKERS = 16
    SYNTHETIC_LOADER_DATA_DIR = SYNTHETIC_DATA_DIR
    SYNTHETIC_LOADER_BATCH_SIZE = 4096
    SYNTHETIC_LOADER_FILE_PATTERN = '*.jsonl'

# --- Synthetic Data Generation Configuration (from synthetic_data.py) ---
# Assuming synthetic.py handles its own output format for now
class SyntheticDataConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else DEFAULT_DEVICE)
    OLLAMA_ENDPOINT = 'http://0.0.0.0:11434/api/generate'
    GENERATION_MODEL_NAME = 'qwen2.5-coder' # Example, adjust as needed
    CORRECTION_MODEL_NAME = 'gemma3:4b'
    MODEL_NAME = 'deepseek-r1:7b'
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
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHT_DECAY = 0.01
    # Example Model/Loss Params (adjust based on actual model)
    MODEL_INPUT_DIM = 128 # Placeholder
    MODEL_NUM_CLASSES = 6 # Placeholder
    LOSS_FUNCTION = 'MSELoss' # Example: 'MSELoss' or 'CrossEntropyLoss'
    # Pruning settings
    PRUNE_INTERVAL_EPOCHS = 5
    PRUNE_AMOUNT = 0.05
    # Checkpointing
    CHECKPOINT_DIR = CHECKPOINT_DIR
    CHECKPOINT_INTERVAL_BATCH_PERCENT = 0.10
    PRUNE_METHOD = 'magnitude' # Example method, adjust as needed
    METRICS_FILENAME_PREFIX = "training_metrics"

# --- Model Configuration ---
class ModelConfig:
    NAME = 'bert-base-uncased'
    NUM_LABELS = 2

# --- Tokenizer Config ---
class TokenizerConfig:
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 2048

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
    LABELING_MODE = 'hierarchical' # Options: 'hierarchical', 'linear', 'hybrid'
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
    """Retrieves a nested configuration value."""
    try:
        parts = key_path.split('.')
        value = globals() # Start search from globals of this module
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                 # Check if it's a class before trying getattr
                 import inspect
                 if inspect.isclass(value):
                     # Try getting class attribute
                     try: value = getattr(value, part)
                     except AttributeError:
                         # Handle case where part might refer to nested dict/class within config dict
                         if isinstance(app_state.get('config'), dict):
                              temp_val = app_state['config']
                              for p in key_path.split('.'): temp_val = temp_val.get(p)
                              if temp_val is not None: return temp_val
                         return default # Fallback to default if not found via class or config dict
                 else: # It's likely an instance or module
                     value = getattr(value, part)

            if value is None: return default
        return value
    except (AttributeError, KeyError, TypeError): # Added TypeError
        # If initial lookup fails, try searching within the loaded app_state config directly
        try:
            if isinstance(app_state.get('config'), dict):
                value = app_state['config']
                for part in key_path.split('.'): value = value.get(part)
                if value is not None: return value
        except (AttributeError, KeyError, TypeError): pass # Ignore errors during fallback search
        return default # Return default if primary and fallback searches fail
    
app_state = {
    'config': None,
    'main_repo_path': None,
    'main_repo_df': None,
    'processed_repo_path': None,
    'processed_repo_df': None,
    'tokenized_repo_path': None,
    'tokenized_repo_df': None,
    'loaded_model_path': None,
    'loaded_model': None,
    'loaded_tokenizer': None
}

def load_config():
    """Loads configuration settings into a dictionary."""
    # This could be expanded to load from YAML/JSON if preferred
    conf = {
        'base_data_dir': str(BASE_DATA_DIR),
        'raw_data_dir': str(RAW_DATA_DIR),
        'processed_data_dir': str(PROCESSED_DATA_DIR),
        'tokenized_data_dir': str(TOKENIZED_DATA_DIR),
        'synthetic_data_dir': str(SYNTHETIC_DATA_DIR),
        'data_repo_dir': str(DATA_REPO_DIR), # Added central repo dir
        'checkpoint_dir': str(CHECKPOINT_DIR),
        'log_dir': str(LOG_DIR),
        'test_data_output_dir': str(TEST_DATA_OUTPUT_DIR),
        'default_device': str(DEFAULT_DEVICE),
        'gpu_min_compute_capability': GPU_MIN_COMPUTE_CAPABILITY,
        'data_processing': {
            'supported_formats': DataProcessingConfig.SCAN_FILE_EXTENSIONS,
            'chunk_size': DataProcessingConfig.PROCESSING_CHUNK_SIZE,
            'max_workers': DataProcessingConfig.MAX_WORKERS,
        },
        'data_loader': {
            'enhanced_dir': str(DataLoaderConfig.ENHANCED_LOADER_DATA_DIR),
            'enhanced_batch_size': DataLoaderConfig.ENHANCED_LOADER_BATCH_SIZE,
            'enhanced_pattern': DataLoaderConfig.ENHANCED_LOADER_FILE_PATTERN,
            'num_workers': DataLoaderConfig.ENHANCED_LOADER_NUM_WORKERS,
            'synthetic_dir': str(DataLoaderConfig.SYNTHETIC_LOADER_DATA_DIR),
            'synthetic_batch_size': DataLoaderConfig.SYNTHETIC_LOADER_BATCH_SIZE,
            'synthetic_pattern': DataLoaderConfig.SYNTHETIC_LOADER_FILE_PATTERN,
        },
        'synthetic_data': {
            'ollama_endpoint': SyntheticDataConfig.OLLAMA_ENDPOINT,
            'model_name': SyntheticDataConfig.MODEL_NAME,
            'target_samples': SyntheticDataConfig.TARGET_SAMPLES,
            'batch_size': SyntheticDataConfig.BATCH_SIZE,
            'data_format': SyntheticDataConfig.DATA_FORMAT,
            'max_workers': SyntheticDataConfig.MAX_WORKERS,
        },
        'training': {
            'max_epochs': TrainingConfig.MAX_EPOCHS,
            'learning_rate': TrainingConfig.INITIAL_LR,
            'weight_decay': TrainingConfig.WEIGHT_DECAY,
            'batch_size': TrainingConfig.BATCH_SIZE,
            'device': str(TrainingConfig.DEVICE),
            'prune_interval': TrainingConfig.PRUNE_INTERVAL_EPOCHS,
            'prune_amount': TrainingConfig.PRUNE_AMOUNT,
            'checkpoint_dir': str(TrainingConfig.CHECKPOINT_DIR),
            'checkpoint_interval_percent': TrainingConfig.CHECKPOINT_INTERVAL_BATCH_PERCENT,
            'metrics_prefix': TrainingConfig.METRICS_FILENAME_PREFIX,
        },
        'model': {
             'name': ModelConfig.NAME,
             'num_labels': ModelConfig.NUM_LABELS,
        },
        'tokenizer': {
            'model_name': TokenizerConfig.MODEL_NAME,
        },
        'zone': vars(ZoneConfig), # Example if ZoneConfig was simpler
        'labeler': vars(LabelerConfig), # Example
        'core': vars(CoreConfig),
        'test': vars(TestConfig),
        'api_key': API_KEY,
        'max_workers': DataProcessingConfig.MAX_WORKERS, # General max workers if needed elsewhere
    }
    # Flatten class vars into dict for easier access if needed elsewhere
    for key, value in conf.items():
        if isinstance(value, type):
            conf[key] = {k: v for k, v in vars(value).items if not k.startswith('_')}
    return conf

def _generate_file_paths(start_path):
    """Generator to yield file paths using os.scandir."""
    try:
        for entry in os.scandir(start_path):
            try:
                if entry.is_dir(follow_symlinks=False): yield from _generate_file_paths(entry.path)
                elif entry.is_file(follow_symlinks=False): yield Path(entry.path)
            except OSError as e: log_statement(loglevel='warning', logstatement=f"Permission error accessing {entry.path}: {e}", main_logger=str(__name__))
    except OSError as e: log_statement(loglevel='warning', logstatement=f"Permission error scanning {start_path}: {e}", main_logger=str(__name__))

# --- Save and Load State Functions ---
def save_state(obj, file_path):
    """Saves the state of an object to a file using torch serialization."""
    file_path = Path(file_path); file_path.parent.mkdir(parents=True, exist_ok=True); torch.save(obj, file_path); print(f"State saved to {file_path}")
def load_state(file_path):
    """Loads the state of an object from a file using torch serialization."""
    file_path = Path(file_path);
    if not file_path.exists(): raise FileNotFoundError(f"File not found: {file_path}")
    obj = torch.load(file_path); print(f"State loaded from {file_path}"); return obj

# --- Dummy Input Function ---
def dummy_input(input_shape, device=None):
    """Generates a dummy input tensor."""
    if device is None: device = DEFAULT_DEVICE
    return torch.randn(*input_shape, device=device)

# --- Helper Function to get Max Workers ---
def _get_max_workers(config=None):
    """Gets the appropriate number of workers from config or defaults."""
    # (Implementation remains the same)
    if config is None: config = app_state.get('config', {})
    proc_workers = config.get('data_processing', {}).get('max_workers')
    general_workers = config.get('max_workers')
    if isinstance(general_workers, int) and general_workers > 0: return general_workers
    if isinstance(proc_workers, int) and proc_workers > 0: return proc_workers
    cpu_count = os.cpu_count(); return min(cpu_count if cpu_count else 4, 32)


def save_dataframe_to_parquet_zst(df: pd.DataFrame, output_path: Path):
    """Saves a pandas DataFrame to a Zstandard-compressed Parquet file."""
    func_logger = str(__name__) # Use module name for logger
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if using cuDF DataFrame and convert if necessary before saving with pandas
        # Ensure cudf is imported safely if GPU checks are used elsewhere
        cudf = None
        if 'cudf' in sys.modules:
             try: cudf = sys.modules['cudf']
             except ImportError: pass # cudf not actually available

        if cudf is not None and isinstance(df, cudf.DataFrame):
            log_statement(loglevel='debug', logstatement="Converting cuDF DataFrame to Pandas before Parquet save.", main_logger=func_logger)
            df_pandas = df.to_pandas()
        elif isinstance(df, pd.DataFrame):
             df_pandas = df # Assume it's already a pandas DataFrame
        else:
             log_statement(loglevel='error', logstatement=f"Unsupported DataFrame type for Parquet saving: {type(df)}", main_logger=func_logger)
             return False

        # Save using pandas to_parquet with zstd compression
        # Requires 'pyarrow' (recommended) or 'fastparquet' engine
        df_pandas.to_parquet(output_path, compression='zstd', engine='pyarrow', index=False)

        log_statement(loglevel='info', logstatement=f"DataFrame saved to compressed Parquet: {output_path}", main_logger=func_logger)
        return True
    except ImportError:
        log_statement(loglevel='error', logstatement="Error saving to Parquet: 'pyarrow' or 'fastparquet' library not found. Please install one (e.g., pip install pyarrow).", main_logger=func_logger)
        return False
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to save DataFrame to compressed Parquet {output_path}: {e}", main_logger=func_logger, exc_info=True)
        # Clean up potentially incomplete file
        if output_path.exists():
            try: output_path.unlink()
            except OSError: pass
        return False

def compress_string_to_file(content_string: str, output_filepath: Path, encoding='utf-8'):
    """Compresses a string directly to a zstandard file."""
    func_logger = str(__name__)
    try:
        output_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        encoded_content = content_string.encode(encoding)
        cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS) # Use constants if defined globally
        with open(output_filepath, 'wb') as f_out:
             f_out.write(cctx.compress(encoded_content))
        log_statement(loglevel='debug', logstatement=f"Compressed string content to '{output_filepath}'", main_logger=func_logger)
        return True
    except FileNotFoundError:
         log_statement(loglevel='error', logstatement=f"Output path directory likely invalid for compression: {output_filepath}", main_logger=func_logger)
         return False
    except Exception as e:
         log_statement(loglevel='error', logstatement=f"Error compressing string to file '{output_filepath}': {e}", main_logger=func_logger, exc_info=True)
         # Clean up potentially incomplete output file
         if output_filepath.exists():
             try: output_filepath.unlink()
             except OSError: pass
         return False