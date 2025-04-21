# TLATOv4.1/src/m1.py
# Contains the logic for the Data Processing and Tokenization submenu

import os
import sys
import hashlib
import pandas as pd
from tqdm import tqdm
import time
import json
from pathlib import Path
import concurrent.futures
import random
import torch # For model loading and training type hints
from transformers import AutoTokenizer, PreTrainedTokenizerBase # For tokenization type hints
import inspect # For dynamic imports
import math # For determining worker counts

# --- Project Imports ---
# Assuming PYTHONPATH is set correctly or main.py handles sys.path
from src.utils.helpers import *
from src.utils.config import *
from src.data.constants import *
# Logger setup
try:
    from src.utils.logger import configure_logging, log_statement
    # Configure logger immediately upon import if needed by the module
    # Depending on logger.py, this might be done once in main.py instead.
    # If configure_logging sets up handlers globally, calling it here is fine.
    # If it returns a logger instance, adjust usage below. Assume it configures globally for now.
    configure_logging()
    log_statement(loglevel='info', logstatement="Logger configured via src.utils.logger.", main_logger=str(__name__))
except ImportError:
    # Fallback standard logging if custom logger fails
    import logging
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.WARNING)
    if not _logger.handlers: # Avoid adding duplicate handlers
        _ch = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _ch.setFormatter(_formatter)
        _logger.addHandler(_ch)
    # Define a compatible log_statement function using the standard logger
    def log_statement(loglevel, logstatement, main_logger, exc_info=False):
        level_map = {
            'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
            'error': logging.ERROR, 'critical': logging.CRITICAL
        }
        # Use the fallback logger specific to this module
        fallback_logger = logging.getLogger(main_logger) # Get logger by name
        # Ensure fallback logger has basic config if it wasn't the main __name__ logger
        if not fallback_logger.handlers:
             fallback_logger.setLevel(logging.WARNING)
             fallback_logger.addHandler(_ch) # Add the handler configured above
        fallback_logger.log(level_map.get(loglevel.lower(), logging.WARNING), logstatement, exc_info=exc_info)
    log_statement(loglevel='error', logstatement="Custom logger module not found. Using fallback standard logging.", main_logger=str(__name__))

# Configuration Constants
try:
    from src.utils.config import (
        load_config,
        BASE_DATA_DIR,
        RAW_DATA_DIR, # May not be used directly here, but available
        PROCESSED_DATA_DIR,
        TOKENIZED_DATA_DIR,
        SYNTHETIC_DATA_DIR, # May not be used directly here
        CHECKPOINT_DIR,
        LOG_DIR,
        DEFAULT_DEVICE
        # Constants related to data structure/columns (attempt import)
        # Assuming these might be defined in config or a constants file
        # If not, defaults will be used below
    )
    # Ensure base directories are Path objects
    BASE_DATA_DIR = Path(BASE_DATA_DIR)
    PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)
    TOKENIZED_DATA_DIR = Path(TOKENIZED_DATA_DIR)
    CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
    LOG_DIR = Path(LOG_DIR)
    log_statement(loglevel='info', logstatement="Configuration constants loaded.", main_logger=str(__name__))
except ImportError as e:
     log_statement(loglevel='error', logstatement=f"Failed to import constants from src.utils.config: {e}. Using defaults.", main_logger=str(__name__))
     # Define fallback defaults if config load fails
     BASE_DATA_DIR = Path(".")
     PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed_data"
     TOKENIZED_DATA_DIR = BASE_DATA_DIR / "tokenized_data"
     CHECKPOINT_DIR = BASE_DATA_DIR / "checkpoints"
     LOG_DIR = BASE_DATA_DIR / "logs"
     DEFAULT_DEVICE = "auto"

# Data/Repo Constants (Try importing, otherwise define locally)
try:
    # Dynamically import all constants and assign them as variables with their own names
    from src.data.constants import *
    # Use imported constants for column definitions
    REPO_COLUMNS = [COL_FILEPATH, COL_FILENAME, COL_SIZE, COL_MTIME, COL_CTIME, COL_HASH, COL_EXTENSION]
    # Define other repo columns, mixing constants and local names
    PROCESSED_REPO_COLUMNS = ['processed_filepath', 'processed_hash', COL_FILEPATH, COL_HASH, COL_MTIME] # Map to original cols
    TOKENIZED_REPO_COLUMNS = ['tokenized_filepath', 'processed_filepath', 'processed_hash', 'tokenizer_name'] # Use processed cols
    log_statement(loglevel='info', logstatement="Data constants (column names) loaded from src.data.constants.", main_logger=str(__name__))

except ImportError:
    log_statement(loglevel='warning', logstatement="src.data.constants not found or missing column names. Using local defaults.", main_logger=str(__name__))
    # Define local column names as fallback
    COL_FILEPATH = 'filepath' # Need to define these if import fails
    COL_FILENAME = 'filename'
    COL_SIZE = 'size'
    COL_MTIME = 'mtime'
    COL_CTIME = 'ctime'
    COL_HASH = 'file_hash'
    COL_EXTENSION = 'extension'
    REPO_COLUMNS = [COL_FILEPATH, COL_FILENAME, COL_SIZE, COL_MTIME, COL_CTIME, COL_HASH, COL_EXTENSION]
    PROCESSED_REPO_COLUMNS = ['processed_filepath', 'processed_hash', COL_FILEPATH, COL_HASH, COL_MTIME]
    TOKENIZED_REPO_COLUMNS = ['tokenized_filepath', 'processed_filepath', 'processed_hash', 'tokenizer_name']

# Other Core Project Imports
try:
    from src.utils.hashing import unhash_filepath as unhash_file
except ImportError: 
    def calculate_file_hash(p): 
        log_statement('warning','Dummy hash func used',__name__); return None
try:
    from src.utils.hashing import hash_filepath as calculate_file_hash
except ImportError:
    def set_compute_device(p): 
        log_statement('warning','Dummy GPU switch func used',__name__); return 'cpu'
try:
    from src.utils.gpu_switch import set_compute_device
except ImportError:
    def set_compute_device(p):
        log_statement('warning','Dummy GPU switch func used',__name__); return 'cpu'
try:    
    from src.utils.helpers import clear_screen # If needed
except ImportError:
    def clear_screen(): pass # Dummy function if not available
try:
    from src.data.readers import read_file
except ImportError:
    def read_file(p):
        log_statement('warning','Dummy read_file func used',__name__); return ""
try:
    from src.data.processing import DataProcessor as dp
except ImportError:
    class DataProcessor:
        def __init__(self, config=None): pass
        def preprocess_text(self, text): return text
except ImportError:
    class TokenizerWrapper:
        def __init__(self, model_name_or_path, config=None): 
            self.name_or_path=model_name_or_path
            def __call__(self, text): return text
try:
    from src.data.loaders import TokenDataset, create_dataloader # Assuming these exist
except ImportError:
    class TokenDataset:
        def __init__(self, token_files):
            self.files=token_files
            def __len__(self): return len(self.files)
    class TokenDataset:
        def __init__(self, token_files):
            self.files=token_files
            def __len__(self): return len(self.files)
try:
    from src.core.models import load_model_from_checkpoint
except ImportError:
    def load_model_from_checkpoint(p):
        log_statement('warning','Dummy model load func used',__name__); return None, None
try:
    from src.training.trainer import Trainer
except ImportError:
    class Trainer:
        def __init__(self, model, train_dataloader, config, device, hyperparameters): pass
        def train(self): time.sleep(2)
except Exception as e:
    log_statement(loglevel='error', logstatement=f"Failed to import core project modules: {e}. Functionality limited.", main_logger=str(__name__), exc_info=True)
    # Define dummies if needed, log_statement handles logger dummy
    if 'calculate_file_hash' not in globals():
        def calculate_file_hash(p): return None
    if 'set_compute_device' not in globals():
        def set_compute_device(p): return 'cpu'
    if 'read_file' not in globals():
        def read_file(p): return ""
    if 'DataProcessor' not in globals():
        class DataProcessor:
            def __init__(self, config=None): pass
            def preprocess_text(self, text): return text
    if 'TokenizerWrapper' not in globals():
        class TokenizerWrapper:
            def __init__(self, model_name_or_path, config=None): 
                self.name_or_path=model_name_or_path
                def __call__(self, text): return text
    if 'TokenDataset' not in globals():
        class TokenDataset:
            def __init__(self, token_files):
                self.files=token_files
                def __len__(self): return len(self.files)
    if 'create_dataloader' not in globals():
        def create_dataloader(ds, batch_size, shuffle=True, num_workers=0, persistent_workers=False): pass
    if 'load_model_from_checkpoint' not in globals(): 
        def load_model_from_checkpoint(p): return None, None
    if 'Trainer' not in globals():
        class Trainer:
            def __init__(self, model, train_dataloader, config, device, hyperparameters): pass
            def train(self): time.sleep(2)

# --- Derived Constants ---
DATA_REPO_DIR = BASE_DATA_DIR / "repositories" # Central place for repo metadata CSVs
INDEX_FILE = DATA_REPO_DIR / "repository_index.json" # Index file location
DEFAULT_MODEL_SAVE_DIR = CHECKPOINT_DIR # Use CHECKPOINT_DIR as default save location
DEFAULT_MAX_WORKERS = os.cpu_count() or 16

from src.utils.config import load_config

# --- Application State ---
app_state = {
    "file_path_str": None, "folder_path_str": None,
    "main_repo_path": None, "main_repo_df": None,
    "processed_repo_path": None, "processed_repo_df": None,
    "tokenized_repo_path": None, "tokenized_repo_df": None,
    "loaded_model": None, "loaded_model_path": None,
    "loaded_tokenizer": None, "config": load_config()
}

# --- Helper Functions ---
def _get_max_workers(config=None):
    """Gets the appropriate number of workers from config or defaults."""
    if config is None: config = app_state.get('config', {})
    proc_workers = config.get('data_processing', {}).get('max_workers')
    general_workers = config.get('max_workers')
    if isinstance(proc_workers, int) and proc_workers > 0: return proc_workers
    if isinstance(general_workers, int) and general_workers > 0: return general_workers
    return min(os.cpu_count() or 4, 32) # Cap default

def _ensure_dir_exists(dir_path: Path):
    """Creates a directory if it doesn't exist."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_statement(loglevel='error', logstatement=f"Could not create directory {dir_path}: {e}", main_logger=str(__name__), exc_info=True)
        raise

def _get_repo_hash(path_obj: Path):
    # Generates a consistent hash for a directory path.
    log_statement(loglevel='info', logstatement=f"Getting repository hash for {path_obj}:", main_logger=__name__)
    normalized_path_str = str(path_obj.resolve())
    log_statement(loglevel='info', logstatement=f"Generating hash for path {normalized_path_str}", main_logger=__name__)
    return hashlib.sha256(normalized_path_str.encode()).hexdigest()[:16]

def _get_repository_info(folder_path_obj: Path):
    """Generates hash and filename for the main data repository."""
    repo_hash = _get_repo_hash(folder_path_obj)
    # Store repo CSVs in DATA_REPO_DIR (derived from BASE_DATA_DIR)
    repo_filename = DATA_REPO_DIR / f"data_repository_{repo_hash}.csv.gz"
    log_statement(loglevel='debug', logstatement=f"Generated main repository info for '{folder_path_obj.resolve()}': hash={repo_hash}, filename={repo_filename}", main_logger=str(__name__))
    return repo_hash, repo_filename

def _load_repository_index():
    """Loads the hash -> original_path mapping from the index file."""
    log_statement(loglevel='info', logstatement=f"Ensuring directory exists for {DATA_REPO_DIR}", main_logger=__name__)
    _ensure_dir_exists(DATA_REPO_DIR) # Use central repo dir
    if not INDEX_FILE.exists():
        log_statement(loglevel='warning', logstatement=f"Repository index file '{INDEX_FILE}' not found. Starting fresh.", main_logger=str(__name__))
        return {}
    try:
        with open(INDEX_FILE, 'r') as f: index_data = json.load(f)
        log_statement(loglevel='info', logstatement=f"Loaded repository index from '{INDEX_FILE}' with {len(index_data)} entries.", main_logger=str(__name__))
        return {h: Path(p) for h, p in index_data.items()}
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to load repository index '{INDEX_FILE}': {e}", main_logger=str(__name__), exc_info=True)
        return {}

def _save_repository_index(index_data):
    """Saves the hash -> original_path mapping to the index file."""
    _ensure_dir_exists(DATA_REPO_DIR)
    try:
        serializable_data = {h: str(p.resolve()) for h, p in index_data.items()}
        with open(INDEX_FILE, 'w') as f: json.dump(serializable_data, f, indent=4)
        log_statement(loglevel='info', logstatement=f"Saved repository index to '{INDEX_FILE}' with {len(index_data)} entries.", main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to save repository index '{INDEX_FILE}': {e}", main_logger=str(__name__), exc_info=True)

def _find_sub_repositories(target_path_obj: Path, repo_index: dict):
    """Identifies existing repositories that cover subdirectories of the target path."""
    sub_repos = []
    target_path_res = target_path_obj.resolve()
    log_statement(loglevel='info', logstatement=f"Checking {len(repo_index)} existing repositories for sub-paths of '{target_path_res}'...", main_logger=str(__name__))
    for repo_hash, existing_path_obj in repo_index.items():
        try:
            existing_path_res = existing_path_obj.resolve()
            if target_path_res != existing_path_res and target_path_res in existing_path_res.parents:
                repo_filename = DATA_REPO_DIR / f"data_repository_{repo_hash}.csv.gz" # Use central repo dir
                if repo_filename.exists():
                    log_statement(loglevel='debug', logstatement=f"Found potential sub-repository: hash={repo_hash}, path='{existing_path_res}', file='{repo_filename}'", main_logger=str(__name__))
                    sub_repos.append((repo_hash, existing_path_res, repo_filename))
                else:
                     log_statement(loglevel='warning', logstatement=f"Index points to non-existent repo file '{repo_filename}'. Skipping.", main_logger=str(__name__))
        except Exception as e:
             log_statement(loglevel='error', logstatement=f"Error processing potential sub-repository hash {repo_hash} path '{existing_path_obj}': {e}", main_logger=str(__name__), exc_info=True)
    log_statement(loglevel='info', logstatement=f"Found {len(sub_repos)} potential sub-repositories.", main_logger=str(__name__))
    return sub_repos

def _get_file_metadata(filepath: Path):
    """Extracts metadata for a single file using constants for column names."""
    try:
        if not filepath.is_file(): return None
        stat = filepath.stat()
        file_hash = calculate_file_hash(str(filepath))
        if file_hash is None:
             log_statement(loglevel='warning', logstatement=f"Could not calculate hash for {filepath}", main_logger=str(__name__))

        # Use imported constants for keys if they exist
        metadata = {
            COL_FILEPATH: str(filepath.resolve()),
            COL_FILENAME: filepath.name,
            COL_SIZE: stat.st_size,
            COL_MTIME: stat.st_mtime,
            COL_CTIME: stat.st_ctime,
            COL_HASH: file_hash,
            COL_EXTENSION: filepath.suffix.lower()
        }
        return metadata
    except Exception as e:
        log_statement(loglevel='warning', logstatement=f"Could not get metadata for {filepath}: {e}", main_logger=str(__name__), exc_info=False)
        return None

def _validate_sub_repository(repo_file: Path, original_path: Path, num_check=10):
    """Validates a sub-repository using defined constants."""
    max_workers = _get_max_workers()
    log_statement(loglevel='info', logstatement=f"Validating sub-repository: {repo_file} (using up to {max_workers} workers)", main_logger=str(__name__))
    try:
        df = pd.read_csv(repo_file, compression='gzip')
        log_statement(loglevel='debug', logstatement=f"Loaded {repo_file} with {len(df)} entries.", main_logger=str(__name__))

        if df.empty: return False, None
        # Use imported constants for required columns
        required_cols = [COL_FILEPATH, COL_SIZE, COL_MTIME, COL_HASH]
        if not all(col in df.columns for col in required_cols):
             log_statement(loglevel='warning', logstatement=f"Sub-repository {repo_file} missing required columns ({required_cols}). Invalid.", main_logger=str(__name__))
             return False, None
        if not original_path.is_dir(): return False, None

        num_to_check = min(num_check, len(df))
        if num_to_check == 0: return True, df

        sample_indices = random.sample(range(len(df)), num_to_check)
        files_to_check = df.iloc[sample_indices]
        mismatches = 0

        log_statement(loglevel='debug', logstatement=f"Checking {num_to_check} sample files from {repo_file} using {max_workers} workers.", main_logger=str(__name__))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            path_to_stored_row = {Path(row[COL_FILEPATH]): row for _, row in files_to_check.iterrows()}
            future_to_path = {executor.submit(_get_file_metadata, path): path for path in path_to_stored_row.keys()}

            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(future_to_path), desc=f"Validating {repo_file.name}", leave=False, unit="file"):
                current_path = future_to_path[future]
                stored_row = path_to_stored_row[current_path]
                try:
                    current_metadata = future.result()
                    if current_metadata is None: mismatches += 1; continue
                    # Compare using constants
                    if current_metadata[COL_SIZE] != stored_row[COL_SIZE] or \
                       abs(current_metadata[COL_MTIME] - stored_row[COL_MTIME]) > 1 or \
                       current_metadata[COL_HASH] != stored_row[COL_HASH]:
                        mismatches += 1
                except Exception as exc:
                    log_statement(loglevel='error', logstatement=f'Validation check exception for {current_path}: {exc}', main_logger=str(__name__), exc_info=True)
                    mismatches += 1

        if mismatches == 0:
            log_statement(loglevel='info', logstatement=f"Validation successful for {repo_file}.", main_logger=str(__name__))
            return True, df
        else:
            log_statement(loglevel='warning', logstatement=f"Validation failed for {repo_file} ({mismatches}/{num_to_check} mismatches).", main_logger=str(__name__))
            return False, None
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error validating sub-repository {repo_file}: {e}", main_logger=str(__name__), exc_info=True)
        return False, None

def _generate_file_paths(start_path):
    """Generator to yield file paths using os.scandir."""
    try:
        for entry in os.scandir(start_path):
            try:
                if entry.is_dir(follow_symlinks=False): yield from _generate_file_paths(entry.path)
                elif entry.is_file(follow_symlinks=False): yield Path(entry.path)
            except OSError as e: log_statement(loglevel='warning', logstatement=f"Permission error accessing {entry.path}: {e}", main_logger=str(__name__))
    except OSError as e: log_statement(loglevel='warning', logstatement=f"Permission error scanning {start_path}: {e}", main_logger=str(__name__))

def _scan_directory(folder_path_obj: Path, existing_files_set: set):
    """Scans directory, gets metadata using constants and configured workers."""
    max_workers = _get_max_workers()
    new_files_metadata = []
    files_to_scan = []
    log_statement(loglevel='info', logstatement=f"Starting recursive scan of '{folder_path_obj}' (using up to {max_workers} workers). Excluding {len(existing_files_set)} known files.", main_logger=str(__name__))

    collected_count = 0
    log_statement(loglevel='debug', logstatement=f"Collecting file list using os.scandir generator...", main_logger=str(__name__))
    for filepath in _generate_file_paths(folder_path_obj):
        collected_count += 1
        resolved_filepath_str = str(filepath.resolve())
        if resolved_filepath_str not in existing_files_set:
            files_to_scan.append(filepath)
        if collected_count % 20000 == 0: # Log less frequently
             log_statement(loglevel='debug', logstatement=f"Collected {collected_count} paths, found {len(files_to_scan)} new candidates...", main_logger=str(__name__))

    log_statement(loglevel='info', logstatement=f"Found {len(files_to_scan)} potential new files to gather metadata for.", main_logger=str(__name__))
    if not files_to_scan: return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_get_file_metadata, filepath): filepath for filepath in files_to_scan}
        pbar = tqdm(total=len(files_to_scan), desc=f"Scanning {folder_path_obj.name}", unit="file")
        for future in concurrent.futures.as_completed(future_to_path):
            filepath = future_to_path[future]
            try:
                metadata = future.result()
                if metadata: new_files_metadata.append(metadata)
            except Exception as exc:
                 log_statement(loglevel='error', logstatement=f'Metadata extraction generated an exception for {filepath}: {exc}', main_logger=str(__name__), exc_info=True)
            finally: pbar.update(1)
        pbar.close()

    log_statement(loglevel='info', logstatement=f"Gathered metadata for {len(new_files_metadata)} new files.", main_logger=str(__name__))
    return new_files_metadata

def _process_file_wrapper(args):
    """Wrapper for parallel processing/tokenization task using constants."""
    original_filepath_str, output_dir, task_type, processor_or_tokenizer = args
    original_filepath = Path(original_filepath_str)
    try:
        content = read_file(original_filepath)
        if content is None: return None # Error logged in read_file usually

        if task_type == 'process':
            processed_text = processor_or_tokenizer.preprocess_text(content)
            output_filename = f"{original_filepath.stem}_processed{original_filepath.suffix}"
            output_filepath = output_dir / output_filename
            with open(output_filepath, 'w', encoding='utf-8') as f: f.write(processed_text)
            processed_hash = calculate_file_hash(str(output_filepath))
            original_hash = calculate_file_hash(str(original_filepath))
            stat = original_filepath.stat()
            # Return dict matching PROCESSED_REPO_COLUMNS structure (using constants where applicable)
            return {
                'processed_filepath': str(output_filepath.resolve()),
                'processed_hash': processed_hash,
                COL_FILEPATH: original_filepath_str, # Original filepath using constant
                COL_HASH: original_hash, # Original hash using constant
                COL_MTIME: stat.st_mtime # Original mtime using constant
            }
        elif task_type == 'tokenize':
            output_filename = f"{original_filepath.stem}_tokens.pt"
            output_filepath = output_dir / output_filename
            tokens = processor_or_tokenizer(content)
            torch.save(tokens, output_filepath)
            processed_hash = calculate_file_hash(str(original_filepath))
            # Return dict matching TOKENIZED_REPO_COLUMNS structure
            return {
                'tokenized_filepath': str(output_filepath.resolve()),
                'processed_filepath': original_filepath_str,
                'processed_hash': processed_hash,
                'tokenizer_name': processor_or_tokenizer.name_or_path
            }
        else: return None
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error in wrapper for file {original_filepath} task {task_type}: {e}", main_logger=str(__name__), exc_info=True)
        return None

# --- Main Menu Option Functions ---
def set_data_directory():
    """Handles Option 1: Set Data Directory using project constants and utilities."""
    global app_state
    log_statement(loglevel='info', logstatement="Starting 'Set Data Directory' process.", main_logger=str(__name__))
    folder_path_str = ""
    try:
        # Use BASE_DATA_DIR as potential default suggestion?
        folder_path_str = input(f"Enter the full path to the data directory (e.g., {BASE_DATA_DIR / 'my_raw_data'}): ")
        target_path = Path(folder_path_str)
        if not target_path.is_dir():
            log_statement(loglevel='error', logstatement=f"Invalid directory path provided: {folder_path_str}", main_logger=str(__name__), exc_info=True)
            return

        log_statement(loglevel=str("info"), logstatement=str(f"Processing directory: {target_path.resolve()}"), main_logger=str(__name__), exc_info=True)
        log_statement(loglevel='info', logstatement=f"Target data directory set to: {target_path.resolve()}", main_logger=str(__name__), exc_info=True)
        repo_hash, repo_filename = _get_repository_info(target_path) # Uses DATA_REPO_DIR now
        log_statement(loglevel='info', logstatement=f"Target repository hash={repo_hash}, file='{repo_filename}'", main_logger=str(__name__))
        _ensure_dir_exists(DATA_REPO_DIR) # Uses derived constant
        repo_index = _load_repository_index() # Uses DATA_REPO_DIR/INDEX_FILE now
        sub_repos_to_check = _find_sub_repositories(target_path, repo_index) # Uses DATA_REPO_DIR now

        valid_sub_repo_dfs = []
        processed_files_set = set()
        validation_futures = []
        max_validation_workers = _get_max_workers(app_state.get('config'))
        log_statement(loglevel=str("info"), logstatement=(f"Validating {len(sub_repos_to_check)} potential sub-repositories (using up to {max_validation_workers} workers)..."), main_logger=str(__name__), exc_info=True)
        if not sub_repos_to_check:
            log_statement(loglevel='info', logstatement="No sub-repositories found to validate.", main_logger=str(__name__))
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_validation_workers) as executor:
            for r_hash, r_path, r_filename in sub_repos_to_check:
                future = executor.submit(_validate_sub_repository, r_filename, r_path, num_check=10)
                validation_futures.append(future)
            pbar_validate = tqdm(total=len(validation_futures), desc="Validating sub-repos", unit="repo")
            for future in concurrent.futures.as_completed(validation_futures):
                try:
                    is_valid, df = future.result()
                    if is_valid and df is not None:
                        valid_sub_repo_dfs.append(df)
                        # Use constant for filepath column
                        processed_files_set.update(df[COL_FILEPATH].astype(str).tolist())
                        log_statement(loglevel='info', logstatement=f"Validated and included data from a sub-repository ({len(df)} files).", main_logger=str(__name__))
                except Exception as e:
                     log_statement(loglevel='error', logstatement=f"Error processing validation result: {e}", main_logger=str(__name__), exc_info=True)
                finally: pbar_validate.update(1)
            pbar_validate.close()
        log_statement(loglevel='info', logstatement=f"Validation complete. Merging data from {len(valid_sub_repo_dfs)} valid sub-repositories ({len(processed_files_set)} files.", main_logger=str(__name__))
        log_statement(loglevel='info', logstatement=f"Total files covered by sub-repositories: {len(processed_files_set)}", main_logger=str(__name__))

        merged_df = pd.DataFrame(columns=REPO_COLUMNS) # Use constant list
        if valid_sub_repo_dfs:
            # Use constant list for reindexing
            valid_sub_repo_dfs = [df.reindex(columns=REPO_COLUMNS) for df in valid_sub_repo_dfs if isinstance(df, pd.DataFrame)]
            if valid_sub_repo_dfs:
                merged_df = pd.concat(valid_sub_repo_dfs, ignore_index=True).drop_duplicates(subset=[COL_FILEPATH]) # Use constant
                log_statement(loglevel='info', logstatement=f"Merged DataFrame contains {len(merged_df)} entries.", main_logger=str(__name__))

        # Scan directory (uses helpers that use constants)
        new_files_metadata = _scan_directory(target_path, processed_files_set)
        new_files_df = pd.DataFrame(new_files_metadata, columns=REPO_COLUMNS) # Use constant list
        log_statement(loglevel='info', logstatement=f"Created DataFrame with {len(new_files_df)} newly scanned files.", main_logger=str(__name__))

        # Combine and clean
        final_df = pd.concat([merged_df, new_files_df], ignore_index=True)
        final_df = final_df.reindex(columns=REPO_COLUMNS).dropna(subset=[COL_FILEPATH, COL_HASH]).drop_duplicates(subset=[COL_FILEPATH], keep='last') # Use constants
        log_statement(loglevel='info', logstatement=f"Final combined repository contains {len(final_df)} unique file entries.", main_logger=str(__name__))
        log_statement(loglevel='info', logstatement=f"Saving final repository to {repo_filename}", main_logger=str(__name__))
        try:
            # Saving is sequential but generally fast for compressed CSV
            with tqdm(total=1, desc="Saving repository", unit="file") as pbar:
                 final_df.to_csv(repo_filename, compression='gzip', index=False)
                 pbar.update(1)
            log_statement(loglevel='info', logstatement="Repository saved successfully.", main_logger=__name__)
            log_statement(loglevel='info', logstatement=f"Successfully saved repository to {repo_filename}", main_logger=str(__name__))
            repo_index[repo_hash] = target_path # Store Path object
            _save_repository_index(repo_index)

            # Update App State
            app_state['main_repo_path'] = str(repo_filename.resolve()); app_state['main_repo_df'] = final_df
            app_state['processed_repo_path'] = None; app_state['processed_repo_df'] = None
            app_state['tokenized_repo_path'] = None; app_state['tokenized_repo_df'] = None
            log_statement(loglevel='info', logstatement=f"App state updated. Active repository: {repo_filename}", main_logger=str(__name__))

        except Exception as e:
             log_statement(loglevel='error', logstatement=f"Failed to save repository file {repo_filename}: {e}", main_logger=str(__name__), exc_info=True)

        log_statement(loglevel='info', logstatement=f"Repository generation/update complete for {target_path.resolve()}", main_logger=str(__name__))

    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error during Set Data Directory for path '{folder_path_str}': {e}", main_logger=str(__name__), exc_info=True)

def process_linguistic_data():
    """Handles Option 2: Process Linguistic Data using project constants."""
    global app_state
    log_statement(loglevel='info', logstatement="Starting linguistic data processing.", main_logger=str(__name__))

    if app_state.get('main_repo_df') is None:
        log_statement(loglevel='error', logstatement="Error: No main data repository loaded. Please run Option 1 first.", main_logger=str(__name__)); return

    if app_state.get('config') is None: app_state['config'] = load_config()
    config = app_state['config']
    main_repo_df = app_state['main_repo_df']
    main_repo_path = Path(app_state['main_repo_path'])
    main_repo_hash = _get_repo_hash(main_repo_path.parent)

    try:
        processing_config = config.get('data_processing', {})
        text_extensions = processing_config.get('text_extensions', ['.txt', '.md'])
        max_workers = _get_max_workers(config)
        # Use imported constant for output base directory
        output_dir = PROCESSED_DATA_DIR / main_repo_hash
        _ensure_dir_exists(output_dir)

        # Use constant COL_EXTENSION for filtering
        source_files_df = main_repo_df[main_repo_df[COL_EXTENSION].isin(text_extensions)].copy()
        if source_files_df.empty: log_statement(loglevel='warning', logstatement="No text files found to process.", main_logger=__name__); return

        log_statement(loglevel='info', logstatement=f"Found {len(source_files_df)} potential text files (using up to {max_workers} workers).", main_logger=str(__name__))
        dp = DataProcessor(repo_dir=REPO_DIR, filename=MAIN_REPO_FILENAME, max_workers=16)
        log_statement(loglevel='info', logstatement=f"DataProcessor initialized.", main_logger=str(__name__))

        # Use central DATA_REPO_DIR for processed repo CSV
        processed_repo_filename = DATA_REPO_DIR / f"processed_repository_{main_repo_hash}.csv.gz"
        existing_processed_df = pd.DataFrame(columns=PROCESSED_REPO_COLUMNS)
        files_to_process_args = []

        if processed_repo_filename.exists():
            try:
                existing_processed_df = pd.read_csv(processed_repo_filename, compression='gzip')
                log_statement(loglevel='info', logstatement=f"Loaded existing processed repository: {processed_repo_filename}", main_logger=str(__name__))
                # Use constants for merge columns
                comparison_df = pd.merge(
                    source_files_df[[COL_FILEPATH, COL_HASH, COL_MTIME]], # Current state from main repo
                    existing_processed_df[[COL_FILEPATH, COL_HASH, COL_MTIME]], # Stored original state in processed repo
                    how='left', on=COL_FILEPATH, suffixes=('_current', '_repo')
                )
                # Identify changes using constants
                needs_processing = comparison_df[
                    comparison_df[f"{COL_HASH}_repo"].isnull() | # New file (no match on filepath)
                    (comparison_df[f"{COL_HASH}_current"] != comparison_df[f"{COL_HASH}_repo"]) | # Hash changed
                    (comparison_df[f"{COL_MTIME}_current"].fillna(0).astype(int) != comparison_df[f"{COL_MTIME}_repo"].fillna(0).astype(int)) # Mtime changed
                ]
                # Use constant COL_FILEPATH for getting list
                files_to_process_args = [(fp, output_dir, 'process', dp) for fp in needs_processing[COL_FILEPATH].tolist()]
                log_statement(loglevel='info', logstatement=f"Identified {len(files_to_process_args)} files needing processing.", main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"Error loading/comparing processed repo {processed_repo_filename}: {e}. Reprocessing all.", main_logger=str(__name__), exc_info=True)
                existing_processed_df = pd.DataFrame(columns=PROCESSED_REPO_COLUMNS)
                files_to_process_args = [(fp, output_dir, 'process', dp) for fp in source_files_df[COL_FILEPATH].tolist()] # Use constant
        else:
            log_statement(loglevel='info', logstatement="No existing processed repository found. Processing all.", main_logger=str(__name__))
            files_to_process_args = [(fp, output_dir, 'process', dp) for fp in source_files_df[COL_FILEPATH].tolist()] # Use constant

        processed_results = []
        if files_to_process_args:
            log_statement(loglevel='info', logstatement=f"Processing {len(files_to_process_args)} files...", main_logger=str(__name__))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_process_file_wrapper, args): args[0] for args in files_to_process_args}
                pbar = tqdm(total=len(files_to_process_args), desc="Processing text", unit="file")
                for future in concurrent.futures.as_completed(future_to_path):
                    try:
                        result = future.result();
                        if result: processed_results.append(result)
                    except Exception as e: log_statement(loglevel='error', logstatement=f"Error processing result: {e}", main_logger=str(__name__), exc_info=True)
                    finally: pbar.update(1)
                pbar.close()
        else: log_statement(loglevel='warning', logstatement="No files require processing.", main_logger=__name__)

        # Update repository DataFrame
        new_processed_df = pd.DataFrame(processed_results, columns=PROCESSED_REPO_COLUMNS)
        # Use constant COL_FILEPATH for identifying reprocessed originals
        reprocessed_originals = set(new_processed_df[COL_FILEPATH]) if not new_processed_df.empty else set()
        source_filepaths = set(source_files_df[COL_FILEPATH])

        if not existing_processed_df.empty:
            final_processed_df = pd.concat([
                existing_processed_df[
                    ~existing_processed_df[COL_FILEPATH].isin(reprocessed_originals) &
                    existing_processed_df[COL_FILEPATH].isin(source_filepaths)
                 ],
                new_processed_df
            ], ignore_index=True).drop_duplicates(subset=['processed_filepath'], keep='last')
        else: final_processed_df = new_processed_df

        final_processed_df = final_processed_df.dropna(subset=['processed_filepath', COL_FILEPATH]) # Use constant
        log_statement(loglevel='info', logstatement=f"Processed repository contains {len(final_processed_df)} entries.", main_logger=str(__name__))

        # Save & Update State
        if not final_processed_df.empty:
            log_statement(loglevel='info', logstatement=f"Saving processed repository to {processed_repo_filename}", main_logger=str(__name__))
            try:
                _ensure_dir_exists(DATA_REPO_DIR) # Use central repo dir
                final_processed_df.to_csv(processed_repo_filename, compression='gzip', index=False)
                log_statement(loglevel='info', logstatement="Processed repository saved successfully.", main_logger=str(__name__))
                app_state['processed_repo_path'] = str(processed_repo_filename.resolve()); app_state['processed_repo_df'] = final_processed_df
                app_state['tokenized_repo_path'] = None; app_state['tokenized_repo_df'] = None
                log_statement(loglevel='info', logstatement=f"App state updated. Active processed repo: {processed_repo_filename}", main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"Failed to save processed repo {processed_repo_filename}: {e}", main_logger=str(__name__), exc_info=True)
        else:
            log_statement(loglevel='warning', logstatement="Final processed repository is empty.", main_logger=str(__name__))
            app_state['processed_repo_path'] = None; app_state['processed_repo_df'] = None

        log_statement(loglevel='info', logstatement="Linguistic data processing finished successfully.", main_logger=str(__name__))

    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error during linguistic data processing: {e}", main_logger=str(__name__), exc_info=True)

def tokenize_data():
    """Handles Option 3: Data Tokenization using project constants."""
    global app_state
    log_statement(loglevel='info', logstatement="Starting data tokenization.", main_logger=str(__name__))

    if app_state.get('processed_repo_df') is None:
        log_statement(loglevel='error', logstatement="Error: No processed data repository loaded. Please run Option 2 first.", main_logger=__name__); return

    if app_state.get('config') is None: app_state['config'] = load_config()
    config = app_state['config']; processed_repo_df = app_state['processed_repo_df']; processed_repo_path = Path(app_state['processed_repo_path'])
    processed_repo_hash = _get_repo_hash(processed_repo_path.parent)

    if processed_repo_df.empty: log_statement(loglevel=str("error"), logstatement="Error: Processed repository DataFrame is empty.", main_logger=str(__name__)); return
    try:
        tokenizer_config = config.get('tokenizer', {}); model_name = tokenizer_config.get('model_name', 'bert-base-uncased')
        max_workers = _get_max_workers(config)
        # Use imported constant for output base dir
        output_dir = TOKENIZED_DATA_DIR / processed_repo_hash
        _ensure_dir_exists(output_dir)

        # Initialize Tokenizer (same as before)
        try:
            if 'TokenizerWrapper' in globals() and tokenizer_config:
                tokenizer = TokenizerWrapper(model_name_or_path=model_name, config=tokenizer_config)
                log_statement(loglevel='info', logstatement=f"TokenizerWrapper initialized for '{model_name}'", main_logger=str(__name__))
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name); tokenizer.name_or_path = model_name
                log_statement(loglevel='info', logstatement=f"AutoTokenizer initialized for '{model_name}'", main_logger=str(__name__))
        except Exception as e:
            log_statement(loglevel='error', log_statement=f"Error initializing tokenizer '{model_name}': {e}", main_logger=__name__, exc_info=True); return

        # Use central DATA_REPO_DIR for tokenized repo CSV
        tokenized_repo_filename = DATA_REPO_DIR / f"tokenized_repository_{processed_repo_hash}.csv.gz"
        existing_tokenized_df = pd.DataFrame(columns=TOKENIZED_REPO_COLUMNS)
        files_to_tokenize_args = []

        required_cols = ['processed_filepath', 'processed_hash']
        if not all(col in processed_repo_df.columns for col in required_cols):
            log_statement(loglevel=str("error"), logstatement=str(f"Error: Processed repo DF missing required columns.  Required columns: {required_cols}"), main_logger=str(__name__), exc_info=True); return

        # Comparison Logic (same as before)
        if tokenized_repo_filename.exists():
            try:
                existing_tokenized_df = pd.read_csv(tokenized_repo_filename, compression='gzip')
                log_statement(loglevel='info', logstatement=f"Loaded existing tokenized repository: {tokenized_repo_filename}", main_logger=str(__name__))
                comparison_df = pd.merge(
                    processed_repo_df[['processed_filepath', 'processed_hash']], existing_tokenized_df[['processed_filepath', 'processed_hash', 'tokenizer_name']], how='left', on='processed_filepath', suffixes=('_current', '_repo')
                )
                needs_tokenizing = comparison_df[
                    comparison_df['processed_hash_repo'].isnull() |
                    (comparison_df['processed_hash_current'] != comparison_df['processed_hash_repo']) |
                    (comparison_df['tokenizer_name'].fillna('') != tokenizer.name_or_path)
                ]
                files_to_tokenize_args = [(fp, output_dir, 'tokenize', tokenizer) for fp in needs_tokenizing['processed_filepath'].tolist()]
                log_statement(loglevel='info', logstatement=f"Identified {len(files_to_tokenize_args)} files needing tokenization.", main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"Error loading/comparing tokenized repo {tokenized_repo_filename}: {e}. Retokenizing all.", main_logger=str(__name__), exc_info=True)
                existing_tokenized_df = pd.DataFrame(columns=TOKENIZED_REPO_COLUMNS)
                files_to_tokenize_args = [(fp, output_dir, 'tokenize', tokenizer) for fp in processed_repo_df['processed_filepath'].tolist()]
        else:
            log_statement(loglevel='info', logstatement="No existing tokenized repository found. Tokenizing all files.", main_logger=str(__name__))
            files_to_tokenize_args = [(fp, output_dir, 'tokenize', tokenizer) for fp in processed_repo_df['processed_filepath'].tolist()]

        # Parallel Tokenization (same as before)
        tokenized_results = []
        if files_to_tokenize_args:
            log_statement(loglevel='info', logstatement=f"Tokenizing {len(files_to_tokenize_args)} files (using up to {max_workers} workers)...", main_logger=__name__)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_process_file_wrapper, args): args[0] for args in files_to_tokenize_args}
                pbar = tqdm(total=len(files_to_tokenize_args), desc="Tokenizing files", unit="file")
                for future in concurrent.futures.as_completed(future_to_path):
                    try:
                        result = future.result()
                        if result: tokenized_results.append(result)
                    except Exception as e: log_statement(loglevel='error', logstatement=f"Error tokenizing result: {e}", main_logger=str(__name__), exc_info=True)
                    finally: pbar.update(1)
                pbar.close()
        else: log_statement(loglevel='warning', logstatement="No files require tokenization.", main_logger=__name__)

        # Update repository DataFrame (same as before)
        new_tokenized_df = pd.DataFrame(tokenized_results, columns=TOKENIZED_REPO_COLUMNS)
        retokenized_processed = set(new_tokenized_df['processed_filepath']) if not new_tokenized_df.empty else set()
        processed_filepaths = set(processed_repo_df['processed_filepath'])
        if not existing_tokenized_df.empty:
            final_tokenized_df = pd.concat([
                existing_tokenized_df[
                     ~existing_tokenized_df['processed_filepath'].isin(retokenized_processed) &
                     existing_tokenized_df['processed_filepath'].isin(processed_filepaths)
                ], new_tokenized_df
            ], ignore_index=True).drop_duplicates(subset=['tokenized_filepath'], keep='last')
        else: final_tokenized_df = new_tokenized_df
        final_tokenized_df = final_tokenized_df.dropna(subset=['tokenized_filepath', 'processed_filepath'])
        log_statement(loglevel='info', logstatement=f"Tokenized repository contains {len(final_tokenized_df)} entries.", main_logger=str(__name__))

        # Save & Update State
        if not final_tokenized_df.empty:
            log_statement(loglevel='info', logstatement=f"Saving tokenized repository to {tokenized_repo_filename}", main_logger=str(__name__))
            try:
                _ensure_dir_exists(DATA_REPO_DIR) # Use central repo dir
                final_tokenized_df.to_csv(tokenized_repo_filename, compression='gzip', index=False)
                log_statement(loglevel='info', logstatement="Tokenized repository saved successfully.", main_logger=str(__name__))
                app_state['tokenized_repo_path'] = str(tokenized_repo_filename.resolve())
                app_state['tokenized_repo_df'] = final_tokenized_df
                log_statement(loglevel='info', logstatement=f"App state updated. Active tokenized repo: {tokenized_repo_filename}", main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"Failed to save tokenized repo {tokenized_repo_filename}: {e}", main_logger=str(__name__), exc_info=True)
        else:
            log_statement(loglevel='warning', logstatement="Final tokenized repository is empty.", main_logger=str(__name__))
            app_state['tokenized_repo_path'] = None; app_state['tokenized_repo_df'] = None

        log_statement(loglevel='info', logstatement="Data tokenization finished successfully.", main_logger=str(__name__))

    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error during data tokenization: {e}", main_logger=str(__name__), exc_info=True)

def train_on_tokens():
    """Handles Option 4: Train On Tokenized Files using project constants."""
    global app_state
    log_statement(loglevel='info', logstatement="Starting model training setup.", main_logger=str(__name__))

    if app_state.get('tokenized_repo_df') is None: log_statement(loglevel='error', logstatement="Error: No tokenized data repository loaded.", main_logger=__name__, exc_info=True); return
    if app_state.get('config') is None: app_state['config'] = load_config()
    config = app_state['config']
    tokenized_repo_df = app_state['tokenized_repo_df']
    tokenized_repo_path = app_state['tokenized_repo_path']

    if tokenized_repo_df.empty: log_statement(loglevel=str("error"), logstatement=str("Error: Tokenized repository is empty."), main_logger=str(__name__)); return

    try:
        # Configs - use imported constants for paths
        train_config = config.get('training', {})
        model_config = config.get('model', {})
        data_loader_config = config.get('data_loader', {})
        default_model = model_config.get('name', 'bert-base-uncased')
        default_lr = train_config.get('learning_rate', 5e-5)
        default_epochs = train_config.get('num_epochs', 3)
        default_batch_size = train_config.get('batch_size', 16)
        # Use imported constants for dirs
        checkpoint_dir = Path(train_config.get('checkpoint_dir', CHECKPOINT_DIR))
        log_dir = Path(train_config.get('log_dir', LOG_DIR))
        num_labels = model_config.get('num_labels', 2)

        # Get Hyperparameters (same as before)
        print("--- Training Hyperparameters ---")
        model_name = input(f"Model name or path [{default_model}]: ") or default_model
        learning_rate = float(input(f"Learning Rate [{default_lr}]: ") or default_lr)
        epochs = int(input(f"Number of Epochs [{default_epochs}]: ") or default_epochs)
        batch_size = int(input(f"Batch Size [{default_batch_size}]: ") or default_batch_size)
        hyperparams = { 'model_name_or_path': model_name, 'learning_rate': learning_rate, 'num_epochs': epochs, 'batch_size': batch_size, 'num_labels': num_labels }
        log_statement(loglevel='info', logstatement=f"Training hyperparameters set: {hyperparams}", main_logger=str(__name__))

        # Device and Logging Config - use imported LOG_DIR
        preferred_device = train_config.get('device', DEFAULT_DEVICE) # Use imported default
        device = set_compute_device(preferred_device)
        log_statement(loglevel='info', logstatement=f"Compute device set to: {device}", main_logger=str(__name__))
        _ensure_dir_exists(log_dir) # Use imported LOG_DIR
        config_log_filename = log_dir / f"training_config_{time.strftime('%Y%m%d%H%M%S')}.log"
        log_statement(loglevel='info', logstatement=f"Logging training configuration to {config_log_filename} and terminal.", main_logger=__name__)
        config_details = f""" --- Training Configuration ---
        Timestamp: {time.asctime()}
        Tokenized Data Source: {tokenized_repo_path or 'N/A'} ({len(tokenized_repo_df)} files)
        Model: {model_name} Num Labels: {num_labels} Device: {device}
        Learning Rate: {learning_rate} Epochs: {epochs} Batch Size: {batch_size}
        Checkpoint Dir: {checkpoint_dir.resolve()} Log Dir: {log_dir.resolve()}
        --- End Configuration --- """
        print(config_details)
        try:
            with open(config_log_filename, 'w') as f: f.write(config_details)
            log_statement(loglevel='info', logstatement=f"Training config saved to {config_log_filename}", main_logger=str(__name__))
        except Exception as log_e: log_statement(loglevel='error', logstatement=f"Failed to write training config log: {log_e}", main_logger=str(__name__))

        # Prepare Data - Use imported config for dataloader workers
        log_statement(loglevel='info', logstatement="Preparing dataset and dataloader...", main_logger=str(__name__))
        token_filepaths = tokenized_repo_df['tokenized_filepath'].tolist()
        if 'TokenDataset' not in globals(): raise ImportError("TokenDataset not available.")
        dataset = TokenDataset(token_files=token_filepaths) # Adjust if labels needed
        num_workers_loader = data_loader_config.get('num_workers', 0)
        persistent_workers = data_loader_config.get('persistent_workers', False) if num_workers_loader > 0 else False
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_loader, persistent_workers=persistent_workers)
        log_statement(loglevel='info', logstatement=f"Dataset ({len(dataset)} items) and Dataloader created (num_workers={num_workers_loader}).", main_logger=str(__name__))

        # Initialize Model (same as before)
        log_statement(loglevel='info', logstatement=f"Initializing model: {model_name}", main_logger=str(__name__))
        try:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.to(device)
            log_statement(loglevel='info', logstatement="Model initialized successfully.", main_logger=str(__name__))
        except Exception as e: log_statement(loglevel='error', logstatement=f"Error initializing model: {e}", main_logger=__name__, exc_info=True); return

        # Initialize Trainer - Ensure checkpoint_dir from config is used
        log_statement(loglevel='info', logstatement="Initializing Trainer...", main_logger=str(__name__))
        try:
            if 'Trainer' not in globals(): raise ImportError("Trainer class not available.")
            # Pass checkpoint_dir explicitly if Trainer expects it
            trainer = Trainer(model=model, train_dataloader=dataloader, config=config, device=device, hyperparameters=hyperparams) # Assume Trainer gets checkpoint_dir from config
            log_statement(loglevel='info', logstatement="Trainer initialized.", main_logger=str(__name__))
        except Exception as e: log_statement(loglevel='error', logstatement=f"Error initializing Trainer: {e}", main_logger=__name__, exc_info=True); return

        # Start Training (same as before)
        log_statement(loglevel='info', logstatement="Starting training loop.", main_logger=str(__name__))
        try:
            trainer.train()
            log_statement(loglevel='info', logstatement="Training loop finished.", main_logger=str(__name__))
        except Exception as e: print(f"Error during training: {e}"); return

        log_statement(loglevel='info', logstatement="Model training process finished successfully.", main_logger=str(__name__))

    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Error during Train On Tokens function: {e}", main_logger=str(__name__), exc_info=True)

def list_and_select_model_path():
    """Handles Option 5A: Scan for models using CHECKPOINT_DIR."""
    print("\n--- Specify Model Folder Path ---")
    try:
        # Use imported CHECKPOINT_DIR as default
        model_folder_path_str = input(f"Enter model folder path (default: {CHECKPOINT_DIR}): ") or str(CHECKPOINT_DIR)
        model_folder_path = Path(model_folder_path_str)

        if not model_folder_path.is_dir(): print(f"Error: Not a valid directory."); return None

        print(f"Scanning '{model_folder_path}'...")
        log_statement(loglevel='info', logstatement=f"Scanning for models in: {model_folder_path}", main_logger=str(__name__))
        found_models = []
        with tqdm(desc="Scanning for models", unit="item") as pbar: # Changed unit
            for root, dirs, files in os.walk(model_folder_path):
                root_path = Path(root)
                # Check if current directory looks like a HF model save
                if 'config.json' in files and ('pytorch_model.bin' in files or 'model.safetensors' in files):
                    type_code = 'HF'; path_str = str(root_path.resolve())
                    if not any(m['path'] == path_str for m in found_models):
                        found_models.append({'path': path_str, 'type_code': type_code})
                        pbar.set_description(f"Found HF: {root_path.name}")
                        pbar.update(1)
                    dirs[:] = [] # Don't recurse further into a detected HF model dir
                    continue # Move to next item in os.walk

                # Check for individual model files in non-HF dirs
                for filename in files:
                    filepath = root_path / filename
                    is_pt_file = filename.lower().endswith(('.pth', '.pt'))
                    is_other = filename.lower().endswith(('.h5', '.onnx', '.pkl', '.joblib', '.bin')) # Exclude .bin if already handled by HF check?
                    if is_pt_file:
                        type_code = 'PT'; found_models.append({'path': str(filepath.resolve()), 'type_code': type_code}); pbar.update(1)
                    elif is_other and filename != 'pytorch_model.bin': # Avoid double counting
                        ext = filepath.suffix.lower()
                        type_code = {'.h5': 'TF', '.onnx': 'ON', '.pkl': 'SK', '.joblib': 'SK', '.bin':'BIN'}.get(ext, 'UN')
                        found_models.append({'path': str(filepath.resolve()), 'type_code': type_code}); pbar.update(1)
                # Update pbar even if no model file found in this dir's files list
                # pbar.update(1) # This would inflate count, update only when model found

        if not found_models: print("No models found."); return None

        log_statement(loglevel='info', logstatement=f"Found {len(found_models)} potential models.", main_logger=str(__name__))
        display_map = {}; found_models.sort(key=lambda x: x['path'])
        for i, model_info in enumerate(found_models):
            designator = f"{i+1}{model_info['type_code']}"; display_map[designator] = model_info['path']
            print(f"{designator} - {model_info['path']}")

        print("--- Model Scan Complete ---")
        return display_map
    except Exception as e: print(f"Error scanning models: {e}"); return None

def specify_model_for_loading(model_map):
    """Handles Option 5B: Specify File For Loading using project loader."""
    global app_state
    print("\n--- Specify Model For Loading ---")
    if not model_map: print("No models scanned."); return None

    while True:
        designator = input("Enter designator to load: ").strip().upper()
        if designator in model_map:
            selected_path_str = model_map[designator]
            selected_path = Path(selected_path_str)
            print(f"Loading model from: {selected_path}...")
            log_statement(loglevel='info', logstatement=f"Attempting to load model from: {selected_path}", main_logger=str(__name__))
            model, tokenizer = None, None
            try:
                with tqdm(total=1, desc=f"Loading {designator}", unit="model") as pbar:
                    # Prioritize using the project's model loader utility
                    if 'load_model_from_checkpoint' in globals():
                        model, tokenizer = load_model_from_checkpoint(selected_path_str)
                        if model is None:
                            raise ValueError(f"load_model_from_checkpoint failed for {selected_path_str}")
                        log_statement(loglevel='info', logstatement=f"Used project's load_model_from_checkpoint.", main_logger=str(__name__))
                    else:
                        # Fallback logic (less robust)
                        log_statement(loglevel='warning', logstatement="Project's load_model_from_checkpoint not found. Using basic loading.", main_logger=str(__name__))
                        if selected_path.is_dir() and (selected_path / 'config.json').exists():
                            from transformers import AutoModel, AutoTokenizer # Late import
                            model = AutoModel.from_pretrained(selected_path_str)
                            tokenizer = AutoTokenizer.from_pretrained(selected_path_str)
                        elif selected_path.is_file() and selected_path.suffix.lower() in ['.pt', '.pth']:
                            raise NotImplementedError("Fallback loading for .pt/.pth not implemented. Requires config/architecture info.")
                        else: raise ValueError(f"Unsupported model type/structure for fallback loading: {selected_path}")
                    pbar.update(1)

                if app_state.get('config') is None: app_state['config'] = load_config()
                device = set_compute_device(app_state['config'].get('training',{}).get('device', DEFAULT_DEVICE))
                if model: model.to(device)

                # Update State
                app_state['loaded_model'] = model
                app_state['loaded_model_path'] = selected_path_str
                app_state['loaded_tokenizer'] = tokenizer
                log_statement(loglevel='info', logstatement=f"Model/Tokenizer loaded from {selected_path} to {device}.", main_logger=str(__name__))
                return selected_path_str
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"Failed to load model {selected_path}: {e}", main_logger=str(__name__), exc_info=True)
                app_state['loaded_model'] = None; app_state['loaded_model_path'] = None; app_state['loaded_tokenizer'] = None
        else: print("Invalid designator.")

def execute_functions_on_loaded_model(loaded_model_path):
    """Handles Option 5C: Execute Functions on Loaded Model."""
    global app_state
    print("\n--- Execute Functions on Loaded Model ---")
    model = app_state.get('loaded_model')
    tokenizer = app_state.get('loaded_tokenizer')
    if not model: print("No model loaded."); return

    print(f"Selected Model Path: {loaded_model_path}")
    try: device = next(model.parameters()).device
    except: device = 'cpu' # Fallback
    print(f"Model on Device: {device}")

    while True:
        print("\n--- Model Operations Submenu ---")
        print("1. Inference")
        print("2. Evaluate (Not Implemented)")
        print("3. Continue Training (Not Implemented)")
        print("4. View Model Details")
        print("5. Return to Load Model Menu")
        choice = input("Enter choice: ")
        log_statement(loglevel='debug', logstatement=f"Model Ops choice: {choice}", main_logger=str(__name__))

        if choice == '1': # Inference
            print("\n--- Inference ---")
            if not tokenizer: print("Error: Tokenizer unavailable."); continue
            try:
                text = input("Enter text: ")
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                model.eval();
                with torch.no_grad(): outputs = model(**inputs)
                # Simple output processing (same as before)
                if hasattr(outputs, 'logits'):
                    preds = torch.argmax(outputs.logits, dim=-1); print(f"Prediction Index: {preds.cpu().item()}")
                else: print(f"Outputs:\n{outputs}")
                log_statement(loglevel='info', logstatement="Inference successful.", main_logger=str(__name__))
            except Exception as e: print(f"Inference error: {e}")
        elif choice == '2': print("\nEvaluate not implemented.")
        elif choice == '3': print("\nContinue Training not implemented.")
        elif choice == '4': # View Details
            print("\n--- Model Details ---"); print(f"Path: {loaded_model_path}\nDevice: {device}"); print("\nArchitecture:"); print(model)
            try: num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"\nTrainable Params: {num_params:,}")
            except: pass
        elif choice == '5': print("Returning..."); break
        else: print("Invalid choice.")

def load_model_submenu():
    """Handles Option 5: Load/Manage Saved Model(s)."""
    global app_state
    log_statement(loglevel='info', logstatement="Entered Load Model submenu.", main_logger=str(__name__))
    model_map = None
    while True:
        print("\n--- Load Model Submenu ---")
        print("A) Scan folder for models")
        print("B) Specify Model From Scan")
        status = f"(Loaded: {Path(app_state['loaded_model_path']).name})" if app_state.get('loaded_model_path') else "(No model loaded)"
        print(f"C) Execute Functions on Loaded Model {status}")
        print("D) Return to Data Processing Menu")
        choice = input("Enter choice (A/B/C/D): ").upper()
        log_statement(loglevel='debug', logstatement=f"Load Model choice: {choice}", main_logger=str(__name__))

        if choice == 'A': model_map = list_and_select_model_path()
        elif choice == 'B':
            if model_map: specify_model_for_loading(model_map)
            else: print("Run Option A first.")
        elif choice == 'C':
            if app_state.get('loaded_model'): execute_functions_on_loaded_model(app_state['loaded_model_path'])
            else: print("Load model via Option B first.")
        elif choice == 'D': print("Returning..."); break
        else: print("Invalid choice.")

# --- Main Submenu Function ---
def data_processing_submenu():
    """Displays the main Data Processing and Tokenization submenu."""
    global app_state
    log_statement(loglevel='info', logstatement="Entered Data Processing submenu.", main_logger=str(__name__))
    print(f"{app_state.get('config')}")
    if app_state.get('config') is None: # Load config once
        try:
            app_state['config'] = load_config()
            log_statement(loglevel='info', logstatement="Config loaded.", main_logger=str(__name__))
        except Exception as e: app_state['config'] = {} # Fallback

    while True:
        # Display status dynamically
        status_main = f"(Set: {Path(app_state['main_repo_path']).name})" if app_state.get('main_repo_path') else "(No repo set)"
        status_proc = f"(Set: {Path(app_state['processed_repo_path']).name})" if app_state.get('processed_repo_path') else "(Not run)"
        status_tok = f"(Set: {Path(app_state['tokenized_repo_path']).name})" if app_state.get('tokenized_repo_path') else "(Not run)"
        status_model = f"(Loaded: Path(app_state['loaded_model_path']).name)" if app_state.get('loaded_model_path') else "(None)"

        print("\n--- Data Processing & Tokenization Menu ---")
        print(f"1. Set Data Directory {status_main}")
        print(f"2. Process Linguistic Data {status_proc}")
        print(f"3. Data Tokenization {status_tok}")
        print(f"4. Train On Tokenized Files")
        print(f"5. Load/Manage Saved Model(s) {status_model}")
        print("6. Exit to Main Menu")
        print("-" * 40)
        choice = input("Enter choice (1-6): ")
        log_statement(loglevel='debug', logstatement=f"Data Processing choice: {choice}", main_logger=str(__name__))

        # Call functions
        if choice == '1': set_data_directory()
        elif choice == '2': process_linguistic_data()
        elif choice == '3': tokenize_data()
        elif choice == '4': train_on_tokens()
        elif choice == '5': load_model_submenu()
        elif choice == '6': print("Returning..."); break
        else: print("Invalid choice.")

        input("\nPress Enter to continue...")