# TLATOv4.1/src/m1.py
# Contains the logic for the Data Processing and Tokenization submenu
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import time
import os
import json
import sys
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Dict, Optional, Any, Tuple, List
from typing import TYPE_CHECKING
import json
from pathlib import Path
from threading import RLock, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import random
import torch # For model loading and training type hints
import inspect # For dynamic imports
import traceback

# --- Project Imports ---
# Assuming PYTHONPATH is set correctly or main.py handles sys.path
from src.analysis.labeler import SemanticLabeler, save_state, load_state
from src.utils.helpers import _generate_file_paths
from src.core.repo_handler import RepoHandler
from src.utils.helpers import *
from src.utils.config import *
from src.data.constants import *
from src.data.processing import DataProcessor, Tokenizer
from src.utils.logger import configure_logging, log_statement
from src.utils.config import *
from src.data.constants import *
from src.utils.hashing import generate_data_hash, hash_filepath as calculate_file_hash
from src.utils.gpu_switch import set_compute_device
from src.data.readers import *
from src.data.processing import DataProcessor, Tokenizer
from src.core.models import load_model_from_checkpoint, ZoneClassifier
from src.training.trainer import EnhancedTrainer, EnhancedDataLoader
from src.data.loaders import EnhancedDataLoader#, TokenDataset, create_dataloader
essential_imports_available = True

# --- Derived Constants ---
DATA_REPO_DIR = BASE_DATA_DIR / "repositories" # Central place for repo metadata CSVs
INDEX_FILE = DATA_REPO_DIR / "repository_index.json" # Index file location

if not INDEX_FILE.exists():
    log_statement('info', f"{LOG_INS}:INFO>>Creating new repository index file at {INDEX_FILE}.", Path(__file__).stem)
    DataProcessor._ensure_repo_exists(filepath=Path(INDEX_FILE), header=MAIN_REPO_HEADER)
    repository_index = {}
    with open(INDEX_FILE, 'w') as f:
        json.dump(repository_index, f, indent=4)
    log_statement('info', f"{LOG_INS}:INFO>>Initialized empty repository index file at {INDEX_FILE}.", Path(__file__).stem)
else:
    log_statement('info', f"{LOG_INS}:INFO>>Validating existing repository index file at {INDEX_FILE}.", Path(__file__).stem)
    try:
        with open(INDEX_FILE, 'r') as f:
            repository_index = json.load(f)
        if not isinstance(repository_index, dict):
            raise ValueError("Repository index file is not a valid dictionary. Reinitializing.")
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error reading repository index file: {e}. Reinitializing.", Path(__file__).stem, exc_info=True)
        repository_index = {}
        with open(INDEX_FILE, 'w') as f:
            json.dump(repository_index, f, indent=4)
        log_statement('info', f"{LOG_INS}:INFO>>Reinitialized repository index file at {INDEX_FILE}.", Path(__file__).stem)

DEFAULT_MODEL_SAVE_DIR = CHECKPOINT_DIR  # Use CHECKPOINT_DIR as default save location
DEFAULT_MAX_WORKERS = os.cpu_count() or 16
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

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

def _get_repo_hash(path_input: Union[str, Path]): # Accept str or Path
    """Generates a consistent hash for a directory path."""
    main_logger_name=__file__ # Define logger name locally
    try:
        # Ensure we have a Path object
        if isinstance(path_input, str):
            path_obj = Path(path_input)
        elif isinstance(path_input, Path):
            path_obj = path_input
        else:
            raise TypeError(f"Input must be a string or Path object, got {type(path_input)}")

        log_statement('debug', f"{LOG_INS}:DEBUG>>Getting repository hash for {path_obj}", main_logger=main_logger_name) # Use debug
        normalized_path_str = str(path_obj.resolve())
        log_statement('debug', f"{LOG_INS}:DEBUG>>Generating hash for resolved path: {normalized_path_str}", main_logger=main_logger_name) # Use debug
        return hashlib.sha256(normalized_path_str.encode()).hexdigest()[:16] # Keep consistent length

    except Exception as e:
        # Log the error and the input that caused it
        log_statement('error', f"{LOG_INS}:ERROR>>Error generating repo hash for input '{path_input}': {e}", main_logger=main_logger_name, exc_info=True)
        return None # Return None on error

def _get_repository_info(folder_path_obj: Path):
    """Generates hash and filename for the main data repository."""
    repo_hash = _get_repo_hash(folder_path_obj)
    # Store repo CSVs in DATA_REPO_DIR (derived from BASE_DATA_DIR)
    repo_filename = DATA_REPO_DIR / f"data_repository_{repo_hash}.csv.zst"
    log_statement('debug', f"{LOG_INS}:DEBUG>>Generated main repository info for '{folder_path_obj.resolve()}': hash={repo_hash}, filename={repo_filename}", Path(__file__).stem)
    return repo_hash, repo_filename

def _find_sub_repositories(target_path_obj: Path, repo_index: dict):
    """Identifies existing repositories that cover subdirectories of the target path."""
    sub_repos = []
    target_path_res = target_path_obj.resolve()
    log_statement('info', f"{LOG_INS}:INFO>>Checking {len(repo_index)} existing repositories for sub-paths of '{target_path_res}'...", Path(__file__).stem)
    for repo_hash, existing_path_obj in repo_index.items():
        try:
            existing_path_res = existing_path_obj.resolve()
            if target_path_res != existing_path_res and target_path_res in existing_path_res.parents:
                repo_filename = DATA_REPO_DIR / f"data_repository_{repo_hash}.csv.zst" # Use central repo dir
                if repo_filename.exists():
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Found potential sub-repository: hash={repo_hash}, path='{existing_path_res}', file='{repo_filename}'", Path(__file__).stem)
                    sub_repos.append((repo_hash, existing_path_res, repo_filename))
                else:
                     log_statement('warning', f"{LOG_INS}:WARNING>>Index points to non-existent repo file '{repo_filename}'. Skipping.", Path(__file__).stem)
        except Exception as e:
             log_statement('error', f"{LOG_INS}:ERROR>>Error processing potential sub-repository hash {repo_hash} path '{existing_path_obj}': {e}", Path(__file__).stem, exc_info=True)
    log_statement('info', f"{LOG_INS}:INFO>>Found {len(sub_repos)} potential sub-repositories.", Path(__file__).stem)
    return sub_repos

def _initialize_repository_index():
    """Ensures the repository index file is properly structured and populated."""
    log_statement('info', f"{LOG_INS}:INFO>>Initializing repository index structure.", Path(__file__).stem)
    try:
        with open(INDEX_FILE, 'r') as f: repository_index = json.load(f)
    except Exception as e:
        log_statement('warning', f"{LOG_INS}:WARNING>>Failed to load repository index: {e}. Reinitializing.", Path(__file__).stem, exc_info=False)
        repository_index = {} # Start fresh if load fails

    updated_index = {}
    if isinstance(repository_index, dict): # Process only if it's a dict
        for folder, files in repository_index.items():
            folder_path = Path(folder)
            if not folder_path.is_dir(): # Check if directory exists
                log_statement('warning', f"{LOG_INS}:WARNING>>Folder {folder} no longer exists. Skipping.", Path(__file__).stem)
                continue
            # Assume files is a list of file entries (dicts) - add more validation if needed
            if isinstance(files, list):
                 updated_files = []
                 for file_entry in files:
                     if isinstance(file_entry, dict) and 'filepath' in file_entry:
                         file_path = Path(file_entry.get('filepath', ''))
                         if not file_path.is_file(): # Check if file exists
                             file_entry['status'] = 'nonexistent'
                         else:
                             # Example updates, adjust as needed
                             file_entry['status'] = file_entry.get('status', 'unknown') # Keep existing status or mark unknown
                             try:
                                 stat = file_path.stat()
                                 file_entry['last_access_date'] = time.ctime(stat.st_atime)
                                 file_entry['last_modified_time'] = time.ctime(stat.st_mtime)
                                 file_entry['filesize'] = stat.st_size
                                 file_entry['extension'] = file_path.suffix.lower()
                                 # Recalculate hash only if missing or explicitly needed? Heavy operation.
                                 if 'data_hash' not in file_entry or not file_entry['data_hash']:
                                     file_entry['data_hash'] = calculate_file_hash(str(file_path)) or ""
                             except Exception as stat_e:
                                 log_statement('error', f"{LOG_INS}:ERROR>>Could not stat/hash file {file_path}: {stat_e}", Path(__file__).stem)
                                 file_entry['status'] = 'error'
                         updated_files.append(file_entry)
                     else:
                         log_statement('warning', f"{LOG_INS}:WARNING>>Skipping invalid file entry in index for folder {folder}: {file_entry}", Path(__file__).stem)
                 updated_index[str(folder_path.resolve())] = updated_files
            else:
                 log_statement('warning', f"{LOG_INS}:WARNING>>Invalid 'files' format for folder {folder} in index. Expected list, got {type(files)}.", Path(__file__).stem)
    else:
        log_statement('error', f"{LOG_INS}:ERROR>>Repository index file {INDEX_FILE} does not contain a valid dictionary.", Path(__file__).stem)
        # Decide whether to wipe the file or just return empty
        updated_index = {} # Reinitialize

    # Save the potentially modified index
    try:
        DATA_REPO_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists before saving
        with open(INDEX_FILE, 'w') as f:
            json.dump(updated_index, f, indent=4)
        log_statement('info', f"{LOG_INS}:INFO>>Repository index updated/validated and saved to {INDEX_FILE}.", Path(__file__).stem)
    except Exception as save_e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to save updated repository index {INDEX_FILE}: {save_e}", Path(__file__).stem, exc_info=True)

def print_welcome_message():
    log_statement('info', f">>>>>>>>>>>>>>>{LOG_INS}<<<<<<<<<<<<<<", __file__)
    log_statement('info', "=" * 44, __file__)
    log_statement('info', " >>>>>     Welcome, To Zombocom     <<<<< ", __file__)
    log_statement('info', " >>>>>     ANYTHING Is Possible     <<<<< ", __file__)
    log_statement('info', " >>>>>        At Zombocom           <<<<< ", __file__)
    log_statement('info', "", __file__)
    log_statement('info', "", __file__)
    log_statement('info', "*" * 44, __file__)

# Replace the existing _load_repository_index function (around line 211):
def _load_repository_index():
    """Loads the hierarchical repository index from the JSON file."""
    log_statement('info', f"{LOG_INS}:INFO>>Ensuring directory exists for {DATA_REPO_DIR}", Path(__file__).stem)
    DATA_REPO_DIR.mkdir(parents=True, exist_ok=True)

    # Use INDEX_FILE constant defined at top level
    if not INDEX_FILE.exists():
        _initialize_repository_index()
        log_statement('warning', f"{LOG_INS}:WARNING>>Repository index file '{INDEX_FILE}' not found. Starting fresh.", Path(__file__).stem)
        return {} # Return empty dict

    try:
        with open(INDEX_FILE, 'r') as f:
            index_data = json.load(f)

        # --- Validation and Conversion ---
        loaded_index = {}
        if isinstance(index_data, dict):
            for repo_hash, entry in index_data.items():
                # Check if entry is the old format (just path string)
                if isinstance(entry, str):
                     log_statement('warning', f"{LOG_INS}:WARNING>>Found old format entry for hash {repo_hash}. Converting.", Path(__file__).stem)
                     try:
                         path_obj = Path(entry)
                         loaded_index[repo_hash] = {
                             INDEX_KEY_PATH: path_obj,
                             INDEX_KEY_METADATA: {}, # Initialize empty metadata
                             INDEX_KEY_CHILDREN: []   # Initialize empty children
                         }
                     except Exception as path_e:
                         log_statement('error', f"{LOG_INS}:ERROR>>Error converting old path '{entry}' for hash {repo_hash}: {path_e}", Path(__file__).stem)
                # Check if entry is the new format (dict)
                elif isinstance(entry, dict) and INDEX_KEY_PATH in entry:
                    try:
                        path_obj = Path(entry[INDEX_KEY_PATH])
                        # Ensure metadata and children keys exist, default if not
                        metadata = entry.get(INDEX_KEY_METADATA, {})
                        children = entry.get(INDEX_KEY_CHILDREN, [])
                        if not isinstance(metadata, dict): metadata = {}
                        if not isinstance(children, list): children = []

                        loaded_index[repo_hash] = {
                            INDEX_KEY_PATH: path_obj,
                            INDEX_KEY_METADATA: metadata,
                            INDEX_KEY_CHILDREN: children
                        }
                    except Exception as path_e:
                         log_statement('error', f"{LOG_INS}:ERROR>>Error processing new format entry for hash {repo_hash}: {path_e}", Path(__file__).stem)
                else:
                     log_statement('warning', f"{LOG_INS}:WARNING>>Skipping invalid entry in index file for hash {repo_hash}: {entry}", Path(__file__).stem)

        else:
             log_statement('error', f"{LOG_INS}:ERROR>>Repository index file '{INDEX_FILE}' is not a valid JSON dictionary. Reinitializing.", Path(__file__).stem)
             return {} # Return empty on invalid format


        log_statement('info', f"{LOG_INS}:INFO>>Loaded repository index from '{INDEX_FILE}' with {len(loaded_index)} entries.", Path(__file__).stem)
        return loaded_index

    except json.JSONDecodeError as json_e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to decode JSON repository index '{INDEX_FILE}': {json_e}", Path(__file__).stem, exc_info=True)
        return {} # Return empty on decode error
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to load repository index '{INDEX_FILE}': {e}", Path(__file__).stem, exc_info=True)
        return {} # Return empty on other errors

# Replace the existing _save_repository_index function (around line 226):
def _save_repository_index(index_data: Dict[str, Dict]):
    """
    Saves the hierarchical repository index, calculates parent/child relationships,
    and updates metadata for specified entries.

    Args:
        index_data (Dict[str, Dict]): The repository index data structure to save.
                                      Structure: {hash: {"path": Path, "metadata": {}, "children": []}}
        update_metadata_for_hash (Optional[str]): If provided, recalculates metadata for this specific hash.
    """
    # Lock to prevent concurrent writes to the index file
    lock = RLock()
    with lock:
        log_statement('debug', f"{LOG_INS}:DEBUG>>Acquired lock for saving repository index.", Path(__file__).stem) # DEBUG

        # Ensure the base directory for repositories exists
        if not REPO_DIR.exists():
            REPO_DIR.mkdir(parents=True, exist_ok=True)
            log_statement('info', f"{LOG_INS}:INFO>>Created repositories directory at: {REPO_DIR}", Path(__file__).stem)

        if not isinstance(REPO_DIR, Path) or not isinstance(INDEX_FILE, str): # Assuming INDEX_FILE_NAME_CONST is "repository_index.json"
             log_statement('critical', f"REPO_DIR or INDEX_FILE_NAME_CONST not configured correctly as Path/str.", Path(__file__).stem)
             return # Critical configuration error

        index_file_path = REPO_DIR / INDEX_FILE # e.g., REPOSITORIES_DIR / "repository_index.json"

        if not REPO_DIR.exists():
            REPO_DIR.mkdir(parents=True, exist_ok=True)
            log_statement('info', f"{LOG_INS}:INFO>>Created repositories directory at: {REPO_DIR}", Path(__file__).stem)

        hash_to_path: Dict[str, Path] = {}
        valid_entries_for_json: Dict[str, Dict[str, Any]] = {}
        path_to_hash_temp: Dict[Path, str] = {}

        for repo_hash, entry_data in index_data.items():
            original_path_str = entry_data.get(INDEX_KEY_PATH) # Assuming INDEX_KEY_PATH is from constants
            if not original_path_str:
                log_statement('warning', f"Entry for hash {repo_hash} is missing '{INDEX_KEY_PATH}'. Skipping.", Path(__file__).stem)
                continue

            try:
                current_path_obj = Path(original_path_str)
                resolved_path = current_path_obj.resolve(strict=True)
            except FileNotFoundError:
                log_statement('warning', f"Path '{original_path_str}' for hash {repo_hash} does not exist. Skipping.", Path(__file__).stem)
                continue
            except Exception as e:
                log_statement('warning', f"Invalid or unresolvable path in index entry for hash {repo_hash}: '{original_path_str}'. Error: {e}", Path(__file__).stem)
                continue

            if resolved_path in path_to_hash_temp:
                existing_hash = path_to_hash_temp[resolved_path]
                if existing_hash != repo_hash:
                    log_statement('warning', (
                        f"Duplicate resolved path '{resolved_path}' found. "
                        f"Hash '{repo_hash}' (path: '{original_path_str}') conflicts with existing hash '{existing_hash}'. "
                        f"Skipping entry for hash '{repo_hash}' to avoid collision."
                    ), main_logger=Path(__file__).stem) # Corrected main_logger usage
                    continue
            
            valid_entries_for_json[repo_hash] = {
                **entry_data,
                INDEX_KEY_PATH: str(current_path_obj) # Store original, potentially relative, path as string
            }
            hash_to_path[repo_hash] = resolved_path
            path_to_hash_temp[resolved_path] = repo_hash

        index_data_to_save = valid_entries_for_json

        try:
            # Sort based on path depth
            sorted_hashes = sorted(hash_to_path.keys(), key=lambda h: len(hash_to_path[h].parts), reverse=True)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error during sorting of hashes for saving index: {e}", Path(__file__).stem, exc_info=True) # Corrected log level
            sorted_hashes = list(index_data_to_save.keys())

        final_data_to_save_ordered = {h: index_data_to_save[h] for h in sorted_hashes}

        try:
            # Corrected: Open the actual INDEX_FILE_PATH for writing
            with open(index_file_path, 'w') as f:
                json.dump(final_data_to_save_ordered, f, indent=4, sort_keys=True)
            log_statement('info', f"{LOG_INS}:INFO>>Repository index saved to '{index_file_path}' with {len(final_data_to_save_ordered)} entries.", Path(__file__).stem)
        except IOError as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to write repository index to '{index_file_path}': {e}", Path(__file__).stem, exc_info=True) # Corrected log level
        except Exception as e:
            log_statement('critical', f"An unexpected error occurred while saving the repository index to '{index_file_path}': {e}", Path(__file__).stem, exc_info=True)
        finally:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Save repository index attempt finished. Released local lock.", Path(__file__).stem)

# --- Helper function to update metadata for a specific repository ---
def _update_index_metadata(index_data: Dict[str, Dict], repo_hash_to_update: str):
    """
    Loads the corresponding .csv.zst, gets summary metadata, and updates the index_data.
    """
    if repo_hash_to_update not in index_data:
        log_statement('warning', f"{LOG_INS}:WARNING>>Cannot update metadata, hash '{repo_hash_to_update}' not found in index.", Path(__file__).stem)
        return

    entry = index_data[repo_hash_to_update]
    path_obj = entry[INDEX_KEY_PATH]
    repo_filename = DATA_REPO_DIR / f"data_repository_{repo_hash_to_update}.csv.zst"

    if not repo_filename.exists():
        log_statement('warning', f"{LOG_INS}:WARNING>>Cannot update metadata, repository file not found: {repo_filename}", Path(__file__).stem)
        entry[INDEX_KEY_METADATA] = {"error": "Repository file missing"}
        return

    log_statement('info', f"{LOG_INS}:INFO>>Updating index metadata for repository: {path_obj.name} (Hash: {repo_hash_to_update})", Path(__file__).stem)
    try:
        # Use RepoHandler to load and get summary
        repo = RepoHandler(metadata_compression='zst', repository_path=repo_filename)
        summary = repo.get_summary_metadata() # Call the new method
        if summary:
            entry[INDEX_KEY_METADATA] = summary
            log_statement('debug', f"{LOG_INS}:DEBUG>>Successfully updated metadata for hash {repo_hash_to_update}: {summary}", Path(__file__).stem)
        else:
             entry[INDEX_KEY_METADATA] = {"error": "Failed to calculate summary"}
             log_statement('warning', f"{LOG_INS}:WARNING>>get_summary_metadata returned empty for {repo_filename}", Path(__file__).stem)

    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to load repository or get summary for {repo_filename} to update index metadata: {e}", Path(__file__).stem, exc_info=True)
        entry[INDEX_KEY_METADATA] = {"error": f"Metadata update failed: {e}"}

# def _validate_sub_repository(repo_file: Path, original_path: Path, num_check=10):
#     """Validates a sub-repository using defined constants, with thread-safe counter."""
#     max_workers = _get_max_workers()
#     log_statement('info', logstatement=f"Validating sub-repository: {repo_file} (using up to {max_workers} workers)", Path(__file__).stem)
#     try:
#         repo = RepoHandler(metadata_compression='zst', repository_path=repo_file)
#         df = repo.df
#         if df is None:
#             log_statement('error', logstatement=f"RepoHandler failed to load {repo_file}. Invalid.", Path(__file__).stem)
#             return False, None
#         log_statement('debug', logstatement=f"Loaded {repo_file} via RepoHandler with {len(df)} entries.", Path(__file__).stem)

#         required_cols = [COL_FILEPATH, COL_SIZE, COL_MTIME, COL_HASH]
#         if not all(col in df.columns for col in required_cols):
#              log_statement('warning', logstatement=f"Sub-repository {repo_file} missing required columns ({required_cols}). Invalid.", Path(__file__).stem)
#              return False, None
#         if not original_path.is_dir(): return False, None # Original path must still exist

#         num_to_check = min(num_check, len(df))
#         if num_to_check == 0: return True, df

#         sample_indices = random.sample(range(len(df)), num_to_check)
#         files_to_check = df.iloc[sample_indices]

#         # --- Thread-safe mismatch counter ---
#         mismatches = 0
#         mismatch_lock = threading.Lock()
#         # ---

#         log_statement('debug', logstatement=f"Checking {num_to_check} sample files from {repo_file} using {max_workers} workers.", Path(__file__).stem)

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             path_to_stored_row = {Path(row[COL_FILEPATH]): row for _, row in files_to_check.iterrows()}
#             future_to_path = {executor.submit(_get_file_metadata, path): path for path in path_to_stored_row.keys()}

#             pbar_validate = tqdm(as_completed(future_to_path), total=len(future_to_path), desc=f"Validating {repo_file.name}", leave=False, unit="file")
#             for future in pbar_validate:
#                 current_path = future_to_path[future]
#                 stored_row = path_to_stored_row[current_path]
#                 increment_mismatch = False
#                 try:
#                     current_metadata = future.result()
#                     if current_metadata is None:
#                         increment_mismatch = True
#                     else:
#                         # Compare using constants
#                         size_mismatch = current_metadata[COL_SIZE] != stored_row[COL_SIZE]
#                         mtime_mismatch = abs(current_metadata[COL_MTIME] - stored_row[COL_MTIME]) > 1 # Allow 1s difference
#                         hash_mismatch = current_metadata[COL_HASH] != stored_row[COL_HASH]
#                         if size_mismatch or mtime_mismatch or hash_mismatch:
#                             increment_mismatch = True
#                             log_statement('debug',
#                                           f"{LOG_INS}Mismatch found for {current_path.name}: "
#                                                        f"Size({size_mismatch}), MTime({mtime_mismatch}), Hash({hash_mismatch})",
#                                           Path(__file__).stem)

#                 except Exception as exc:
#                     log_statement('error', logstatement=f'Validation check exception for {current_path}: {exc}', Path(__file__).stem, exc_info=False) # Keep log concise
#                     increment_mismatch = True

#                 if increment_mismatch:
#                     with mismatch_lock: # Lock the counter increment
#                         mismatches += 1

#         # Read the final mismatch count after the parallel block
#         final_mismatches = mismatches

#         if final_mismatches == 0:
#             log_statement('info', f"{LOG_INS}Validation successful for {repo_file}.", Path(__file__).stem)
#             return True, df
#         else:
#             log_statement('warning', f"{LOG_INS}Validation failed for {repo_file} ({final_mismatches}/{num_to_check} mismatches).", Path(__file__).stem)
#             return False, None

#     except ImportError:
#         log_statement('critical', f"{LOG_INS}RepoHandler class not available. Cannot load {repo_file}.", Path(__file__).stem)
#         return False, None
#     except Exception as e:
#         log_statement('error', f"{LOG_INS}Error instantiating/loading RepoHandler for {repo_file}: {e}", Path(__file__).stem, exc_info=True)
#         return False, None

def set_data_directory(): # Added app_state argument as it's likely needed
    """
    Handles Option 1: Set Data Directory.
    Integrates sub-repository discovery, validation, scanning, and
    detailed file reconciliation (new, modified, deleted).
    Updates the global app_state.
    """
    # global app_state # Keep global if app_state is truly global, otherwise remove if passed in
    global app_state

    log_statement('info', f"{LOG_INS}:INFO>>Starting 'Set Data Directory' process.", Path(__file__).stem)
    folder_path_str = ""
    target_path_resolved = None # Initialize for use in final log message

    try:
        # --- 1. Get and Validate Target Directory ---
        # Use BASE_DATA_DIR from config/constants as part of the example prompt
        example_path = BASE_DATA_DIR # Assuming BASE_DATA_DIR is Path object
        folder_path_str = input(f"Enter the full path to the data directory (e.g., {example_path}): ").strip()
        if not folder_path_str:
             log_statement('warning', f"{LOG_INS}:WARNING>>No path entered by user. Aborting.", Path(__file__).stem)
             print("Operation cancelled.")
             return

        target_path = Path(folder_path_str)
        if not target_path.exists():
             log_statement('error', f"{LOG_INS}:ERROR>>Directory path does not exist: {target_path}", Path(__file__).stem)
             log_statement('error', f"{LOG_INS}:ERROR>>Path does not exist: {target_path}")
             return
        if not target_path.is_dir():
            log_statement('error', f"{LOG_INS}:ERROR>>Provided path is not a directory: {target_path}", Path(__file__).stem)
            log_statement('error', f"{LOG_INS}:ERROR>>Invalid directory path: {target_path}")
            return

        target_path_resolved = target_path.resolve()
        log_statement('info', f"{LOG_INS}:INFO>>:INFO>>Processing directory: {target_path_resolved}", Path(__file__).stem)

        # --- Get Repo Info & Calculate Central Storage Path ---
        # Assuming _get_repository_info calculates hash and base filename from target_path_resolved
        repo_hash, repo_base_filename = _get_repository_info(target_path_resolved) # Requires helper
        repo_storage_path = DATA_REPO_DIR / repo_base_filename # Construct full path in central dir
        # Define the expected index path (e.g., in the same central directory)
        central_index_path = DATA_REPO_DIR / INDEX_FILE

        log_statement('info', f"{LOG_INS}:INFO>>Target data dir='{target_path_resolved}', Hash={repo_hash}, Repo Storage File='{repo_storage_path}'", Path(__file__).stem)
        DATA_REPO_DIR.mkdir(parents=True, exist_ok=True)

        # --- 2. Load and Update Hierarchical Index (Central Index) ---
        repo_index = _load_repository_index() # Load central index
        if repo_hash not in repo_index:
             # Add entry using the actual data directory path
             repo_index[repo_hash] = { INDEX_KEY_PATH: str(target_path_resolved), INDEX_KEY_METADATA: {}, INDEX_KEY_CHILDREN: [] }
        else:
             # Update existing entry path
             entry = repo_index[repo_hash]
             entry.setdefault(INDEX_KEY_METADATA, {})
             entry.setdefault(INDEX_KEY_CHILDREN, [])
             entry[INDEX_KEY_PATH] = str(target_path_resolved)
        _save_repository_index(repo_index) # Save central index immediately after potential update

        # --- 3. Find Potential Sub-repositories ---
        sub_repos_to_check = []
        log_statement('info', f"{LOG_INS}:INFO>>Checking {len(repo_index)} index entries for sub-paths of '{target_path_resolved}'...", Path(__file__).stem)
        for r_hash, entry in repo_index.items():
             if r_hash == repo_hash: continue # Skip self
             try:
                 existing_path_raw = entry.get(INDEX_KEY_PATH)
                 if not existing_path_raw: continue
                 existing_path_obj = Path(existing_path_raw)
                 existing_path_res = existing_path_obj.resolve() # Resolve path from index

                 # Check if target_path_resolved is an ancestor of existing_path_res
                 # Use is_relative_to for robust check (Python 3.9+)
                 if existing_path_res != target_path_resolved and existing_path_res.is_relative_to(target_path_resolved):
                 # Alt check: if target_path_resolved in existing_path_res.parents:
                     # Determine sub-repo filename based on its hash
                     sub_repo_filename = DATA_REPO_DIR / f"data_repository_{r_hash}.csv.zst"
                     if sub_repo_filename.exists():
                          log_statement('debug', f"{LOG_INS}:DEBUG>>Found potential sub-repository: Hash={r_hash}, Path='{existing_path_res}', File='{sub_repo_filename}'", Path(__file__).stem)
                          sub_repos_to_check.append((r_hash, existing_path_res, sub_repo_filename))
                     else:
                          log_statement('warning', f"{LOG_INS}:WARNING>>Index points to sub-repository path '{existing_path_res}' but its state file '{sub_repo_filename}' does not exist! Skipping.", Path(__file__).stem)
             except Exception as e:
                  log_statement('error', f"{LOG_INS}:ERROR>>Error processing potential sub-repository (Hash: {r_hash}, Path: '{entry.get(INDEX_KEY_PATH)}'): {e}", Path(__file__).stem, exc_info=False) # Keep log cleaner

        log_statement('info', f"{LOG_INS}:INFO>>Found {len(sub_repos_to_check)} potential existing sub-repositories.", Path(__file__).stem)

        # --- 4. Validate Sub-repositories in Parallel ---
        valid_sub_repo_dfs = []
        max_validation_workers = _get_max_workers(app_state.get('config')) # Requires helper
        log_statement('info', f"{LOG_INS}:INFO>>Validating {len(sub_repos_to_check)} potential sub-repositories (using up to {max_validation_workers} workers)...", Path(__file__).stem)

        if not sub_repos_to_check:
            log_statement('info', f"{LOG_INS}:INFO>>No sub-repositories found to validate.", Path(__file__).stem)
            merged_df_from_subs = pd.DataFrame(columns=MAIN_REPO_HEADER) # Use constant/schema
        else:
            # Ensure _validate_sub_repository is available
            validate_func = globals().get('_validate_sub_repository') # Check global scope
            if not callable(validate_func):
                 log_statement('critical', f"{LOG_INS}:CRITICAL>>Helper function '_validate_sub_repository' not found or not callable. Cannot validate sub-repos.", Path(__file__).stem)
                 return # Critical dependency missing

            validation_futures = []
            with ThreadPoolExecutor(max_workers=max_validation_workers, thread_name_prefix="SubRepo_Validator") as executor:
                for r_hash, r_path, r_filename in sub_repos_to_check:
                    # Pass required args to validation function
                    future = executor.submit(validate_func, r_filename, r_path, num_check=10) # Assume signature
                    validation_futures.append(future)

                pbar_validate = tqdm(as_completed(validation_futures), total=len(validation_futures), desc="Validating sub-repos", unit="repo", leave=False)
                temp_valid_dfs = []
                valid_count = 0
                for future in pbar_validate:
                    try:
                        # Assuming validate_func returns (bool, DataFrame or None)
                        is_valid, df_result = future.result()
                        if is_valid and isinstance(df_result, pd.DataFrame) and not df_result.empty:
                            temp_valid_dfs.append(df_result)
                            valid_count += 1
                            log_statement('debug', f"{LOG_INS}:DEBUG>>Collected valid data from sub-repository ({len(df_result)} files). Total valid: {valid_count}", Path(__file__).stem)
                        # else: Validation failed or empty, assuming logged within validate_func
                    except Exception as e:
                        log_statement('error', f"{LOG_INS}:ERROR>>Error processing validation result future: {e}", Path(__file__).stem, exc_info=True)
            valid_sub_repo_dfs = temp_valid_dfs

            # --- 5. Merge Valid Sub-repository Data ---
            if valid_sub_repo_dfs:
                log_statement('info', f"{LOG_INS}:INFO>>Concatenating data from {len(valid_sub_repo_dfs)} valid sub-repositories...", Path(__file__).stem)
                # Ensure consistent columns before concat using the authoritative header order
                safe_dfs_to_concat = []
                for df in valid_sub_repo_dfs:
                     try:
                          # Add missing columns from authoritative header, filled with NA/None
                          df_reindexed = df.reindex(columns=MAIN_REPO_HEADER)
                          safe_dfs_to_concat.append(df_reindexed)
                     except Exception as reindex_e:
                          log_statement('error', f"{LOG_INS}:ERROR>>Error reindexing sub-repo DataFrame before concat: {reindex_e}. Skipping this sub-repo.", Path(__file__).stem)

                if safe_dfs_to_concat:
                    try:
                        merged_df_from_subs = pd.concat(safe_dfs_to_concat, ignore_index=True)
                        # Deduplicate based on COL_FILEPATH (absolute paths assumed)
                        dedup_key = COL_FILEPATH
                        if dedup_key in merged_df_from_subs.columns:
                            initial_count = len(merged_df_from_subs)
                            # Ensure key is string for reliable drop_duplicates
                            merged_df_from_subs[dedup_key] = merged_df_from_subs[dedup_key].astype(str)
                            merged_df_from_subs.drop_duplicates(subset=[dedup_key], keep='first', inplace=True)
                            log_statement('info', f"{LOG_INS}:INFO>>Merged {len(merged_df_from_subs)} unique file entries from sub-repos (deduplicated by {dedup_key} from {initial_count}).", Path(__file__).stem)
                        else:
                            log_statement('error', f"{LOG_INS}:ERROR>>Cannot deduplicate merged sub-repos: Missing key column '{dedup_key}'.", Path(__file__).stem)
                            merged_df_from_subs = pd.DataFrame(columns=MAIN_REPO_HEADER) # Reset
                    except Exception as concat_e:
                        log_statement('error', f"{LOG_INS}:ERROR>>Error concatenating/deduplicating sub-repo DataFrames: {concat_e}", Path(__file__).stem, exc_info=True)
                        merged_df_from_subs = pd.DataFrame(columns=MAIN_REPO_HEADER) # Reset on error
                else: # No DFs survived reindexing
                     merged_df_from_subs = pd.DataFrame(columns=MAIN_REPO_HEADER)
            else:
                log_statement('info', f"{LOG_INS}:INFO>>No valid, non-empty DataFrames found from sub-repositories.", Path(__file__).stem)
                merged_df_from_subs = pd.DataFrame(columns=MAIN_REPO_HEADER)
        authoritative_cols = COL_SCHEMA # Need a way to get schema without instance? Static method? Or instance needed?
        if 'merged_df_from_subs' in locals() and isinstance(merged_df_from_subs, pd.DataFrame):
             merged_df_from_subs = merged_df_from_subs.reindex(columns=authoritative_cols)
        else:
             merged_df_from_subs = pd.DataFrame(columns=authoritative_cols)

        # --- 6. Initialize Target Repository Instance ---
        target_repo: Optional[RepoHandler] = None # Initialize
        try:
            log_statement('info', f"{LOG_INS}:INFO>>Initializing RepoHandler. Data Root='{target_path_resolved}', Storage Path='{repo_storage_path}'", Path(__file__).stem)
            # Pass the data directory and the calculated central storage path
            target_repo = RepoHandler(
                data_path=folder_path_str,
                repo_hash=generate_data_hash(Path(repo_storage_path)),
                repo_index_entry=None,
                metadata_compression='zst',
                repository_path=target_path_resolved
            )
            # Get existing data (or empty frame) from the instance
            existing_df = target_repo.df.copy() if target_repo.df is not None else pd.DataFrame(columns=target_repo.expected_columns_order)
            log_statement('info', f"{LOG_INS}:INFO>>RepoHandler instance initialized. Loaded/Created state file {target_repo.repo_file} ({len(existing_df)} entries).", Path(__file__).stem)
            repo_schema_dict = COL_SCHEMA

        except ImportError: # Should not happen if imports are correct
            log_statement('critical', f"{LOG_INS}:CRITICAL>> RepoHandler class not available.", Path(__file__).stem)
            return
        except Exception as load_e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error initializing/loading target repository for data root {target_path_resolved}: {load_e}", Path(__file__).stem, exc_info=True)
            return

        # --- 7. Scan Target Directory ---
        log_statement('info', f"{LOG_INS}:INFO>>Scanning target directory '{target_path_resolved.name}'...", Path(__file__).stem)
        scan_func = globals().get('_scan_directory') # Or repo_instance.scan_directory() if refactored
        if not callable(scan_func):
             log_statement('critical', f"{LOG_INS}:CRITICAL>>Helper function '_scan_directory' not found or not callable. Cannot scan directory.", Path(__file__).stem)
             return
        try:
            current_files_metadata = scan_func(target_path_resolved, existing_files_set=set()) # Assuming scan provides needed metadata
            current_files_df = pd.DataFrame(current_files_metadata) if current_files_metadata else pd.DataFrame()
            log_statement('info', f"{LOG_INS}:INFO>>Scan found {len(current_files_df)} files.", Path(__file__).stem)
        except Exception as scan_e:
            # ... (handle scan error) ...
             return

        # --- 8. Reconcile Data ---
        log_statement('info', f"{LOG_INS}:INFO>>Reconciling existing ({len(existing_df)}), sub-repository ({len(merged_df_from_subs)}), and scan ({len(current_files_df)}) data...", Path(__file__).stem)
        final_df_to_save = pd.DataFrame()
        authoritative_cols = COL_SCHEMA # Use order from instance

        # Define authoritative column order (crucial for consistency)
        if current_files_df.empty and existing_df.empty and merged_df_from_subs.empty:
            log_statement('warning', f"{LOG_INS}:WARNING>>No file data from any source. Resulting repository will be empty.", Path(__file__).stem)
            final_df_to_save = pd.DataFrame(columns=authoritative_cols)
        else:
            # --- Prepare Base (Existing + Unique Sub-Repos) ---
            base_dfs = []
            if not existing_df.empty: base_dfs.append(existing_df.reindex(columns=authoritative_cols))
            if not merged_df_from_subs.empty: base_dfs.append(merged_df_from_subs.reindex(columns=authoritative_cols))
            if base_dfs:
                 combined_base_df = pd.concat(base_dfs, ignore_index=True)
                 # Deduplicate base (prioritize existing over sub-repo if duplicate)
                 # Use absolute filepath string for reliable deduplication
                 dedup_key = COL_FILEPATH
                 if dedup_key in combined_base_df.columns:
                      combined_base_df[dedup_key] = combined_base_df[dedup_key].astype(str) # Ensure string type
                      combined_base_df.drop_duplicates(subset=[dedup_key], keep='first', inplace=True)
                      log_statement('debug', f"{LOG_INS}:DEBUG>>Combined base dataset (existing + unique subs) has {len(combined_base_df)} unique entries (by {dedup_key}).", Path(__file__).stem)
                 else:
                      log_statement('warning', f"{LOG_INS}:WARNING>>Cannot deduplicate combined base DF: Missing '{dedup_key}'.", Path(__file__).stem)
            else:
                 combined_base_df = pd.DataFrame(columns=authoritative_cols)

            # --- Prepare Current Scan DF ---
            current_files_df_reindexed = current_files_df.reindex(columns=authoritative_cols)

            # --- Reconciliation Merge ---
            merge_key = COL_FILEPATH # Use absolute filepath string
            try:
                if combined_base_df.empty and current_files_df_reindexed.empty: # Both sides empty
                     merged_output = pd.DataFrame() # Result is empty
                elif combined_base_df.empty: # Only new files
                     merged_output = current_files_df_reindexed.copy()
                     merged_output['merge_status'] = 'new'
                elif current_files_df_reindexed.empty: # Only old files (all deleted)
                     merged_output = combined_base_df.copy()
                     merged_output['merge_status'] = 'deleted'
                else: # Both have data, perform merge
                     # Ensure merge keys have compatible types (string)
                     combined_base_df[merge_key] = combined_base_df[merge_key].astype(str)
                     current_files_df_reindexed[merge_key] = current_files_df_reindexed[merge_key].astype(str)

                     # Outer merge to find matches, new, and deleted
                     merged_output = pd.merge(
                         combined_base_df,
                         current_files_df_reindexed,
                         on=merge_key, # Merge on filepath
                         how='outer',
                         suffixes=('_old', '_new'),
                         indicator='merge_status' # Add column indicating source ('left_only', 'right_only', 'both')
                     )
                     # Map indicator to meaningful status
                     merged_output['merge_status'] = merged_output['merge_status'].map({
                         'left_only': 'deleted', 'right_only': 'new', 'both': 'potential_match'
                     })
                log_statement('debug', f"{LOG_INS}:DEBUG>>Reconciliation merge completed ({len(merged_output)} rows). Status counts:\n{merged_output['merge_status'].value_counts() if 'merge_status' in merged_output else 'N/A'}", Path(__file__).stem)
            except Exception as merge_e:
                 log_statement('error', f"{LOG_INS}:ERROR>>Error during reconciliation merge: {merge_e}", Path(__file__).stem, exc_info=True)
                 return

            # --- Process Merged Rows ---
            output_rows = []
            for _, row in tqdm(merged_output.iterrows(), total=len(merged_output), desc="Reconciling files", leave=False, unit="file"):
                status = row.get('merge_status')
                output_row = {}

                if status == 'new':
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Reconcile NEW: {row.get(f'{COL_FILEPATH}')}", Path(__file__).stem)
                    for col_const in authoritative_cols:
                        # Get data from the '_new' columns (which are the original names since suffixes only added on merge='both')
                        output_row[col_const] = row.get(col_const)
                    output_row[COL_STATUS] = STATUS_NEW
                    output_rows.append(output_row)

                elif status == 'deleted':
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Reconcile DELETED (Ignoring): {row.get(f'{COL_FILEPATH}')}", Path(__file__).stem)
                    # Do not include deleted files in the final output

                elif status == 'potential_match':
                    # Compare mtime and size from _old (repo) and _new (scan) columns
                    # Convert timestamps (stored as objects/strings from merge) back to numeric/datetime for comparison
                    mtime_old_ts = pd.to_datetime(row.get(f"{COL_MTIME}_old"), errors='coerce', utc=True)
                    mtime_new_ts = pd.to_datetime(row.get(f"{COL_MTIME}_new"), errors='coerce', utc=True)
                    size_old = pd.to_numeric(row.get(f"{COL_SIZE}_old"), errors='coerce')
                    size_new = pd.to_numeric(row.get(f"{COL_SIZE}_new"), errors='coerce')

                    # Use a tolerance for timestamp comparison (e.g., 1 second)
                    mtime_diff_ok = abs((mtime_old_ts - mtime_new_ts).total_seconds()) < 1 if pd.notna(mtime_old_ts) and pd.notna(mtime_new_ts) else (pd.isna(mtime_old_ts) and pd.isna(mtime_new_ts))
                    size_match = size_old == size_new if pd.notna(size_old) and pd.notna(size_new) else (pd.isna(size_old) and pd.isna(size_new))

                    if mtime_diff_ok and size_match: # UNCHANGED
                        log_statement('debug', f"{LOG_INS}:DEBUG>>Reconcile UNCHANGED: {row.get(f'{COL_FILEPATH}')}", Path(__file__).stem)
                        # Keep the old data, as only mtime/size were compared
                        for col_const in authoritative_cols:
                            output_row[col_const] = row.get(f"{col_const}_old")
                        # Ensure status is carried over, default if missing
                        output_row[COL_STATUS] = row.get(f"{COL_STATUS}_old", STATUS_UNKNOWN)
                        output_rows.append(output_row)
                    else: # MODIFIED
                        log_statement('info', f"{LOG_INS}:INFO>>Reconcile MODIFIED: {row.get(f'{COL_FILEPATH}')} (MTIME Match: {mtime_diff_ok}, Size Match: {size_match})", Path(__file__).stem)
                        # Take the new scan data
                        for col_const in authoritative_cols:
                             output_row[col_const] = row.get(f"{col_const}_new")
                        output_row[COL_STATUS] = STATUS_NEW # Mark as new/modified, needs reprocessing
                        # Explicitly reset downstream fields? Or let processing handle it? Let's reset key ones.
                        output_row[COL_PROCESSED_PATH] = None # Example reset
                        output_row[COL_TOKENIZED_PATH] = None # Example reset
                        output_row[COL_SEMANTIC_LABEL] = None # Example reset
                        output_row[COL_LINGUISTIC_METADATA] = None # Example reset
                        output_rows.append(output_row)
                else: # Should not happen with outer merge if logic is correct
                     log_statement('warning', f"{LOG_INS}:WARNING>>Unexpected merge status '{status}' for row with key {row.get(COL_FILEPATH)}. Skipping.", Path(__file__).stem)


            # Create final DataFrame from the reconciled rows
            if output_rows:
                final_df_to_save = pd.DataFrame(output_rows)
                # Ensure final schema conformance (order and type application)
                final_df_to_save = final_df_to_save.reindex(columns=authoritative_cols) # Ensure order/presence
                # Apply final dtypes rigorously using the map from the target repo instance
                schema_dtype_map = COL_SCHEMA
                log_statement('debug', f"{LOG_INS}:DEBUG>>Applying final schema dtypes to reconciled DataFrame...", Path(__file__).stem)
                for col, dtype_obj in schema_dtype_map.items():
                     if col in final_df_to_save.columns:
                          try:
                              # Apply type conversions using the robust logic (similar to loading)
                                if isinstance(dtype_obj, pd.DatetimeTZDtype):
                                     final_df_to_save[col] = pd.to_datetime(final_df_to_save[col], errors='coerce', utc=True).astype(dtype_obj)
                                elif isinstance(dtype_obj, pd.Int64Dtype):
                                     final_df_to_save[col] = pd.to_numeric(final_df_to_save[col], errors='coerce').astype('Float64').astype(dtype_obj)
                                elif isinstance(dtype_obj, pd.BooleanDtype):
                                     bool_map = {'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                                                 'false': False, 'no': False, 'n': False, '0': False, 'f': False,
                                                 '': pd.NA, '<na>': pd.NA, 'none': pd.NA}
                                     lower_series = final_df_to_save[col].fillna('<NA>').astype(str).str.lower()
                                     final_df_to_save[col] = lower_series.map(bool_map).astype(dtype_obj)
                                elif isinstance(dtype_obj, pd.Float64Dtype):
                                     final_df_to_save[col] = pd.to_numeric(final_df_to_save[col], errors='coerce').astype(dtype_obj)
                                elif isinstance(dtype_obj, pd.StringDtype):
                                     final_df_to_save[col] = final_df_to_save[col].astype(dtype_obj)
                                else: # Other types
                                     final_df_to_save[col] = final_df_to_save[col].astype(dtype_obj)
                          except Exception as final_type_e:
                              log_statement('warning', f"{LOG_INS}:WARNING>>Error applying final dtype {dtype_obj} to column '{col}' after reconcile: {final_type_e}. Keeping current type.", Path(__file__).stem)
            else: # No rows survived reconciliation
                 final_df_to_save = pd.DataFrame(columns=authoritative_cols).astype(schema_dtype_map, errors='ignore')

        if 'final_df_to_save' not in locals() or not isinstance(final_df_to_save, pd.DataFrame): # Check if reconciliation produced DataFrame
             log_statement('error', f"{LOG_INS}:ERROR>>Reconciliation failed to produce final DataFrame.", Path(__file__).stem)
             return # Cannot proceed

        log_statement('info', f"{LOG_INS}:INFO>>Reconciliation complete. Final DataFrame has {len(final_df_to_save)} entries.", Path(__file__).stem)

        # --- 9. Save Final DataFrame and Update State ---
        try:
            if target_repo is None: # Should not happen if init succeeded
                log_statement('critical', f"{LOG_INS}:CRITICAL>> target_repo is None before final save.", Path(__file__).stem)
                return

            log_statement('info', f"{LOG_INS}:INFO>>Assigning final DataFrame ({len(final_df_to_save)} entries) to RepoHandler instance.", Path(__file__).stem)
            target_repo.df = final_df_to_save # Assign the final reconciled DataFrame

            log_statement('info', f"{LOG_INS}:INFO>>Requesting repository save via target_repo.save(save_type='repository')...", Path(__file__).stem)
            target_repo.save(save_type='repository') # Save the main repository data (DF and Index)

            # *** Robust Save Confirmation ***
            if target_repo.repo_file and target_repo.repo_file.exists():
                 log_statement('info', f"{LOG_INS}:INFO>>Repository save completed successfully for {target_path_resolved}. File: {target_repo.repo_file}", Path(__file__).stem)
            else:
                 # Log error but don't raise immediately, allow app state update attempt
                 log_statement('error', f"{LOG_INS}:ERROR>>Save reported success but repository file NOT FOUND at expected location: {target_repo.repo_file}", Path(__file__).stem)
                 # Maybe raise a custom exception or return a failure status?
                 # raise IOError(f"Save failed verification for {target_repo.repo_file}") # Optional: re-enable raise if needed

            # Update App State (use the repo instance and resolved path)
            app_state['repo'] = target_repo
            app_state['repo_path'] = target_path_resolved
            # app_state['main_repo_df'] = final_df_to_save.copy() # Probably redundant if app_state['repo'] holds the live object
            log_statement('info', f"{LOG_INS}:INFO>>App state updated. Active repository path: {target_path_resolved}", Path(__file__).stem)
            print(f"\nRepository for '{target_path_resolved.name}' set and updated successfully ({len(final_df_to_save)} files tracked).")

        except Exception as save_e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed during final repository save or app state update: {save_e}", Path(__file__).stem, exc_info=True)
            print(f"\nError saving repository changes for {target_path_resolved}. Check logs.")
            # Reset app state on save failure?
            app_state['repo'] = None
            app_state['repo_path'] = None


    except Exception as outer_e:
        print(f"\nAn unexpected error occurred: {outer_e}") # User feedback
        log_statement('error', f"{LOG_INS}:ERROR>>Error during 'Set Data Directory' process for input '{folder_path_str}': {outer_e}", Path(__file__).stem, exc_info=True)
        app_state['repo'] = None
        app_state['repo_path'] = None

    log_statement('info', f"{LOG_INS}:INFO>>'Set Data Directory' process finished.", Path(__file__).stem)


def has_file_changed(filepath: Path) -> tuple[bool, str, float, float]:
    """
    Checks if a file's metadata (mtime, size) or content hash has changed
    compared to stored information (requires access to the repository DataFrame).
    Logs errors WITHOUT exc_info=True in the handler to prevent recursion.

    Args:
        filepath (Path): The path to the file to check.

    Returns:
        tuple[bool, str, float, float]: (has_changed, change_flag, mtime, atime)
            has_changed: True if mtime, size, or hash differs from repo.
            change_flag: 'M' (Metadata), 'C' (Content), 'N' (None)
            mtime: Current modification time (float timestamp).
            atime: Current access time (float timestamp).
            Returns (False, 'E', 0.0, 0.0) on error accessing file/repo.
    """
    global app_state # Need access to the loaded main_repo_df
    main_logger_name = __file__
    change_flag = 'N'
    has_changed_flag = False
    current_mtime = 0.0
    current_atime = 0.0

    try:
        stat = filepath.stat()
        current_mtime = stat.st_mtime
        current_atime = stat.st_atime
        current_size = stat.st_size

        repo_df = app_state.get('main_repo_df')
        if repo_df is None or repo_df.empty:
            log_statement('debug', f"{LOG_INS}:DEBUG>>No repo DataFrame loaded, assuming file needs processing: {filepath.name}", main_logger=main_logger_name)
            return True, 'N', current_mtime, current_atime # Treat as new/changed if no repo baseline

        # Find the file entry in the DataFrame
        filepath_str = str(filepath.resolve())
        entry = repo_df[repo_df[COL_FILEPATH] == filepath_str]

        if entry.empty:
            # File not found in repo, it's new/changed
            log_statement('debug', f"{LOG_INS}:DEBUG>>File not found in repo, assuming changed: {filepath.name}", main_logger=main_logger_name)
            return True, 'N', current_mtime, current_atime

        # File found, compare metadata
        stored_row = entry.iloc[0]
        # Convert stored timestamps (which might be datetime objects after loading) back to float for comparison
        stored_mtime = stored_row[COL_MTIME]
        if isinstance(stored_mtime, pd.Timestamp): stored_mtime = stored_mtime.timestamp()
        stored_size = stored_row[COL_SIZE]
        stored_hash = stored_row[COL_HASH]

        # Check metadata change
        metadata_changes = abs(current_mtime - stored_mtime) > 1 or current_size != stored_mtime.timestamp()

        # Compare mtime (with tolerance) and size
        if not metadata_changes:
            return False, 
        else:
            change_flag = 'M' # Metadata changed
            has_changed_flag = True
            log_statement('debug', f"{LOG_INS}:DEBUG>>Metadata change detected for {filepath.name}", main_logger=main_logger_name)
            # Check content hash only if metadata changed
            current_hash = generate_data_hash(filepath)
            if not current_hash or current_hash != stored_hash:
                change_flag = 'C' # Content changed (implies metadata may also have)
                log_statement('debug', f"{LOG_INS}:DEBUG>>Content change detected for {filepath.name}", main_logger=main_logger_name)
        
        return has_changed_flag, change_flag, current_mtime, current_atime

    except FileNotFoundError:
        log_statement('warning', f"{LOG_INS}:WARNING>>File not found during change check: {filepath.name}", main_logger=main_logger_name, exc_info=False) # Log WITHOUT traceback
        return False, 'E', 0.0, 0.0 # Error state
    except Exception as e:
        # Log error WITHOUT exc_info=True to prevent recursion
        log_statement('error', f"{LOG_INS}:ERROR>>Error checking file changes for {filepath}: {e}", main_logger=main_logger_name, exc_info=False)
        return False, 'E', 0.0, 0.0 # Error state

def _get_file_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata for a single file using project constants defined
    in MAIN_REPO_HEADER. Handles errors gracefully.
    """
    metadata_logger_name = __file__
    metadata = {} # Initialize empty dict
    log_statement('debug', f"{LOG_INS}:DEBUG>>Getting metadata for: {filepath.name}", main_logger=metadata_logger_name)
    try:
        if not filepath.is_file():
            log_statement('warning', f"{LOG_INS}:WARNING>>Skipping non-file item: {filepath}", main_logger=metadata_logger_name)
            return None
        stat = filepath.stat()
        file_hash = generate_data_hash(filepath)
        file_hash_to_store = file_hash if file_hash else ""

        # --- Populate known metadata using constants ---
        metadata[COL_FILEPATH] = str(filepath.resolve())
        metadata[COL_FILENAME] = filepath.name
        metadata[COL_SIZE] = stat.st_size
        metadata[COL_MTIME] = stat.st_mtime # Float timestamp
        metadata[COL_CTIME] = stat.st_ctime # Float timestamp
        metadata[COL_HASH] = file_hash_to_store
        metadata[COL_EXTENSION] = filepath.suffix.lower().lstrip('.')
        metadata[COL_STATUS] = STATUS_DISCOVERED if file_hash else STATUS_ERROR
        metadata[COL_ERROR] = "" if file_hash else "Hash calculation failed"
        metadata[COL_PROCESSED_PATH] = ""
        metadata[COL_TOKENIZED_PATH] = ""
        metadata[COL_LAST_UPDATED] = time.time() # Float timestamp

        # --- Add defaults for ALL other columns in MAIN_REPO_HEADER ---
        for key in MAIN_REPO_HEADER:
            if key not in metadata:
                # Use sensible nulls/empty values based on likely type convention
                if 'ts' in key or 'time' in key: metadata[key] = np.nan # Float NaN
                elif 'size' in key or 'designation' in key: metadata[key] = pd.NA # Nullable Int NA
                else: metadata[key] = '' # Default empty string
            if COL_HASHED_PATH_ID in MAIN_REPO_HEADER: metadata[COL_HASHED_PATH_ID] = hash_filepath(metadata[COL_FILEPATH]) if 'hash_filepath' in globals() else ""
            if COL_COMPRESSED_FLAG in MAIN_REPO_HEADER: metadata[COL_COMPRESSED_FLAG] = 'N' # Default assumption
            if COL_IS_COPY_FLAG in MAIN_REPO_HEADER: metadata[COL_IS_COPY_FLAG] = 'N' # Default assumption

        # log_statement('debug', f"{LOG_INS}:DEBUG>>Successfully gathered metadata for {filepath.name}.", main_logger=metadata_logger_name)
        return metadata

    except PermissionError as pe:
         log_statement('error', f"{LOG_INS}:ERROR>>Permission error getting metadata for {filepath.name}: {pe}", main_logger=metadata_logger_name, exc_info=False) # REMOVED exc_info
         return None
    except OSError as ose:
        log_statement('error', f"{LOG_INS}:ERROR>>OS error getting metadata for {filepath.name}: {ose}", main_logger=metadata_logger_name, exc_info=False) # REMOVED exc_info
        return None
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Unexpected error getting metadata for {filepath.name}: {e}", main_logger=metadata_logger_name, exc_info=False) # REMOVED exc_info
        return None

def _validate_sub_repository(repo_file: Path, original_path: Path, num_check=10):
    """Validates a sub-repository using defined constants."""
    max_workers = _get_max_workers()
    log_statement('info', f"{LOG_INS}:INFO>>Validating sub-repository: {repo_file} (using up to {max_workers} workers)", Path(__file__).stem)
    try:
        # Instantiate RepoHandler for the sub-repo file (this loads the data)
        repo = RepoHandler(metadata_compression='zst', repository_path=repo_file)
        df = repo.df # Access the loaded DataFrame

        # Check if loading failed within RepoHandler
        if df is None:
            log_statement('error', f"{LOG_INS}:ERROR>>RepoHandler failed to load {repo_file}. Invalid.", Path(__file__).stem)
            return False, None
        log_statement('debug', f"{LOG_INS}:DEBUG>>Loaded {repo_file} via RepoHandler with {len(df)} entries.", Path(__file__).stem)

        required_cols = MAIN_REPO_HEADER
        if not all(col in df.columns for col in required_cols):
             log_statement('warning', f"{LOG_INS}:WARNING>>Sub-repository {repo_file} missing required columns ({required_cols}). Invalid.", Path(__file__).stem)
             return False, None
        if not original_path.is_dir(): return False, None

        num_to_check = min(num_check, len(df))
        if num_to_check == 0: return True, df

        sample_indices = random.sample(range(len(df)), num_to_check)
        files_to_check = df.iloc[sample_indices]
        mismatches = 0

        log_statement('debug', f"{LOG_INS}:DEBUG>>Checking {num_to_check} sample files from {repo_file} using {max_workers} workers.", Path(__file__).stem)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            path_to_stored_row = {Path(row[COL_FILEPATH]): row for _, row in files_to_check.iterrows()}
            future_to_path = {executor.submit(_get_file_metadata, path): path for path in path_to_stored_row.keys()}

            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc=f"Validating {repo_file.name}", leave=False, unit="file"):
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
                    log_statement('error', f'Validation check exception for {current_path}: {exc}', Path(__file__).stem, True)
                    mismatches += 1

        if mismatches == 0:
            log_statement('info', f"{LOG_INS}:INFO>>Validation successful for {repo_file}.", Path(__file__).stem)
            return True, df
        else:
            log_statement('warning', f"{LOG_INS}:WARNING>>Validation failed for {repo_file} ({mismatches}/{num_to_check} mismatches).", Path(__file__).stem)
            return False, None
    except ImportError: # Handle case where RepoHandler couldn't be imported
        log_statement('critical', f"{LOG_INS}:CRITICAL>>RepoHandler class not available. Cannot load {repo_file}.", Path(__file__).stem)
        return False, None
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error instantiating/loading RepoHandler for {repo_file}: {e}", Path(__file__).stem, exc_info=True)
        return False, None

def _scan_directory(folder_path_obj: Path, existing_files_set: set):
    """
    Scans directory recursively, gets metadata for new files in parallel,
    displays progress, and uses custom logging.

    Args:
        folder_path_obj (Path): The directory to scan.
        existing_files_set (set): A set of resolved file path strings already known
                                  (typically from loaded sub-repositories).

    Returns:
        list: A list of dictionaries, where each dictionary contains metadata
              for a newly discovered file (not present in existing_files_set).
    """
    max_workers = _get_max_workers(app_state.get('config'))
    new_files_metadata = []
    scan_logger_name = __file__ # Logger name for this context

    # Stage 1: Collect potential paths (Serial)
    potential_paths = []
    log_statement('debug', f"{LOG_INS}:DEBUG>>Stage 1: Collecting all potential file paths recursively...", main_logger=scan_logger_name)
    with tqdm(desc=f"Discovering items [{folder_path_obj.name}]", unit=" items", smoothing=0.1, leave=False) as pbar_discover:
        try:
            for filepath in _generate_file_paths(folder_path_obj):
                potential_paths.append(filepath)
                pbar_discover.update(1)
        except Exception as gen_e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error during path generation: {gen_e}", main_logger=scan_logger_name, exc_info=True)
    collected_count = len(potential_paths)
    log_statement('info', f"{LOG_INS}:INFO>>Stage 1 complete. Found {collected_count} total items.", main_logger=scan_logger_name)
    if not potential_paths: log_statement('info', f"{LOG_INS}:INFO>>No items found in the target directory to process further.", main_logger=scan_logger_name); return []

    log_statement(
        'info',
        logstatement=(
            f"Starting recursive scan: '{folder_path_obj.name}'. "
            f"Max Workers: {max_workers}. "
            f"Excluding: {len(existing_files_set)} known file paths."
        ),
        main_logger=scan_logger_name
    )

    def check_path(filepath: Path, known_paths: set) -> Optional[Path]:
        try:
            resolved_path_str = str(filepath.resolve())
            if resolved_path_str not in known_paths: return filepath
            return None
        except OSError as e:
            log_statement('warning', f"{LOG_INS}:WARNING>>OS Error resolving/checking path '{filepath.name}': {e}.  Skipping.", main_logger=scan_logger_name)
        except Exception as e:
            log_statement('warning', logstatement=f"{__file__}:{inspect.currentframe().f_linenu} - Error resolving/checking path '{filepath.name}': {e}", main_logger=scan_logger_name)
            return None

    # Stage 2: Parallel Path Filtering (Modified for safety)
    log_statement('info', logstatement=f"Stage 2: Filtering {collected_count} paths against {len(existing_files_set)} known paths (Parallel)...", main_logger=scan_logger_name)

    # --- Thread-safe counter and results collection ---
    skipped_known_count = 0
    files_to_scan_results = [] # Collect results here first
    skipped_lock = Lock() # Lock for the counter
    # ---

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='FilterPath') as executor:
        futures = {executor.submit(check_path, fpath, existing_files_set): fpath for fpath in potential_paths}
        pbar_filter = tqdm(as_completed(futures), total=len(potential_paths), desc=f"Filtering Paths [{folder_path_obj.name}]", unit="path", leave=False)

        for future in pbar_filter:
            original_path = futures[future]
            try:
                result_path = future.result()
                if result_path:
                    files_to_scan_results.append(result_path) # Append to temp list
                else:
                    with skipped_lock: # Lock the counter increment
                        skipped_known_count += 1
                if pbar_filter.n % 100 == 0 or pbar_filter.n == pbar_filter.total:
                     pbar_filter.set_postfix_str(f"New: {len(files_to_scan_results)}, Skip/Known: {skipped_known_count}", refresh=True)
            except Exception as e:
                log_statement('error', logstatement=f"Error processing path filter future for ~'{original_path.name}': {e}", main_logger=scan_logger_name, exc_info=False)
                with skipped_lock: # Also lock if incrementing due to error
                    skipped_known_count += 1

    # Now update the main list outside the parallel block
    files_to_scan = files_to_scan_results
    log_statement('info', logstatement=f"Stage 2 complete. Found {len(files_to_scan)} new file paths. Skipped/Known: {skipped_known_count}.", main_logger=scan_logger_name)
    if not files_to_scan: return []

    # --- Stage 3: Gather Metadata for new files (Parallel) ---
    log_statement('info', f"{LOG_INS}:INFO>>Stage 3: Gathering metadata for {len(files_to_scan)} files (Parallel)...", main_logger=scan_logger_name)
    # --- Thread-safe counter and results collection ---
    processed_count_meta = 0
    metadata_results = [] # Collect results here
    processed_lock = Lock() # Lock for counter
    start_time_meta = time.time()
    # ---
    mdata_err = None
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Metadata') as executor:
        future_to_path = { executor.submit(_get_file_metadata, filepath): filepath for filepath in files_to_scan }
        pbar_meta = tqdm(as_completed(future_to_path), total=len(files_to_scan), desc=f"Gather Metadata [{folder_path_obj.name[:20]}]", unit="file", leave=True, postfix={"files/s": "0.0", "CPU": "N/A", "Mem": "N/A"} )
        for future in pbar_meta:
            filepath = future_to_path[future]
            with processed_lock: # Lock counter increment
                processed_count_meta += 1
            current_count = processed_count_meta # Read locked value locally
            try:
                metadata = future.result()
                if metadata:
                    metadata_results.append(metadata) # Append to temp list
                else: metadata_errors += 1; mdata_err = metadata_errors
            except Exception as exc:
                log_statement('error', logstatement=f'{LOG_INS} - Metadata task generated an exception for {filepath.name}: {exc}', main_logger=scan_logger_name, exc_info=True)
                metadata_errors += 1
                mdata_err = metadata_errors

            # Update Metrics Postfix (less frequent)
            if current_count % 50 == 0 or current_count == len(files_to_scan):
                elapsed_time = time.time() - start_time_meta
                files_per_sec = current_count / elapsed_time if elapsed_time > 0 else 0
                metrics_postfix = {"files/s": f"{files_per_sec:.1f}"}
                if PSUTIL_AVAILABLE:
                    try:
                        cpu_usage = psutil.cpu_percent(interval=None)
                        mem_usage = psutil.virtual_memory().percent
                        metrics_postfix["CPU"] = f"{cpu_usage:.1f}%"
                        metrics_postfix["Mem"] = f"{mem_usage:.1f}%"
                    except Exception as psutil_e:
                         metrics_postfix["CPU"] = "ERR"; metrics_postfix["Mem"] = "ERR"
                         if current_count % 500 == 0: log_statement('warning', logstatement=f"psutil metrics failed: {psutil_e}", main_logger=scan_logger_name)
                pbar_meta.set_postfix(metrics_postfix, refresh=False)
            
    # Update main list outside parallel block
    new_files_metadata = metadata_results
    final_elapsed_time_meta = time.time() - start_time_meta
    final_rate_meta = processed_count_meta / final_elapsed_time_meta if final_elapsed_time_meta > 0 else 0
    log_statement('info', 
                logstatement=(f"Stage 3 finished. Processed {processed_count_meta} files for metadata in {final_elapsed_time_meta:.2f}s ({final_rate_meta:.2f} files/s).",
                f"({mdata_err} errors) in {final_elapsed_time_meta:.2f}s "
                f"({final_rate_meta:.2f} files/s)."
            ),
            main_logger=scan_logger_name
    )
    log_statement('info', f"{LOG_INS}:INFO>>Recursive scan complete. Returning metadata for {len(new_files_metadata)} new files.", main_logger=scan_logger_name)
    return new_files_metadata

# --- MODIFIED WRAPPER FUNCTION ---
def _process_file_wrapper(args):
    """
    Wrapper for parallel tokenization task using constants.
    Expects args: (processed_filepath_str, processed_file_hash, output_dir, tokenizer_instance)
    Returns a dictionary matching TOKENIZED_REPO_COLUMNS structure on success.
    """
    processed_filepath_str, processed_file_hash, output_dir, tokenizer = args
    processed_filepath = Path(processed_filepath_str)
    main_logger_name = __file__ # Define logger name
    log_statement('debug', f"{LOG_INS}:DEBUG>>Wrapper start for {processed_filepath.name}", main_logger=main_logger_name)

    try:
        # --- Load Processed Data ---
        # This requires a robust way to load based on the processed file format
        # Assuming a helper function _load_processed_data exists that handles
        # loading based on suffix (e.g., .json.zst, .parquet.zst) and returns
        # content suitable for the tokenizer (e.g., text string or structured data).
        # For simplicity, we'll simulate reading text content here.
        # Replace this with actual loading logic.
        try:
            # Example: Load compressed text from .json.zst (adjust based on actual save format)
            if processed_filepath.suffix == '.zst' and processed_filepath.suffixes[-2] == '.json':
                 import zstandard as zstd
                 import io
                 import json
                 dctx = zstd.ZstdDecompressor()
                 with open(processed_filepath, 'rb') as ifh:
                      with dctx.stream_reader(ifh) as reader:
                           content_bytes = reader.read()
                           json_data = json.loads(content_bytes.decode('utf-8'))
                           # Assuming the text is stored under a key like 'cleaned_text'
                           content = json_data.get('cleaned_text', '')
                           if not content:
                               # Try combining all string values if 'cleaned_text' is missing
                               content = " ".join([str(v) for v in json_data.values() if isinstance(v, str)])
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Loaded JSON content ({len(content)} chars) from {processed_filepath.name}", main_logger=main_logger_name)

            elif processed_filepath.suffix == '.zst' and processed_filepath.suffixes[-2] == '.parquet':
                 # Add logic to load parquet if needed
                 log_statement('warning', f"{LOG_INS}:WARNING>>Parquet loading logic not implemented in wrapper for {processed_filepath.name}", main_logger=main_logger_name)
                 content = None # Mark as failure if loader not implemented
            else:
                 # Add logic for other potential processed formats or raw text
                 log_statement('warning', f"{LOG_INS}:WARNING>>Unhandled processed file type in wrapper: {processed_filepath.name}", main_logger=main_logger_name)
                 content = None # Mark as failure

            if content is None:
                 raise ValueError(f"Failed to load or extract content from {processed_filepath.name}")

        except Exception as load_err:
            log_statement('error', f"{LOG_INS}:ERROR>>Error loading processed file {processed_filepath.name} in wrapper: {load_err}", main_logger=main_logger_name, exc_info=True)
            return None # Failed to load

        # --- Tokenize Content ---
        # Actual tokenization logic using the passed tokenizer instance
        # This might involve chunking for long content depending on the model
        # Example: Simple call (adjust based on tokenizer needs)
        try:
            # Assuming tokenizer expects text and returns something savable by torch.save
            # (e.g., dictionary of tensors like {'input_ids': ..., 'attention_mask': ...})
            tokens_output = tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512) # Example HF tokenizer call
        except Exception as tok_err:
            log_statement('error', f"{LOG_INS}:ERROR>>Tokenization failed for {processed_filepath.name}: {tok_err}", main_logger=main_logger_name, exc_info=True)
            return None

        # --- Save Tokenized Output ---
        # Use a consistent naming scheme, link to processed path hash maybe?
        # Using processed file stem + _tokens.pt.zst for now
        tokenized_filename = f"{processed_filepath.stem}_tokens.pt.zst"
        tokenized_filepath = output_dir / tokenized_filename
        tokenized_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Ensure tensors are on CPU before saving
            tokens_cpu = {k: v.cpu() for k, v in tokens_output.items()} if isinstance(tokens_output, dict) else tokens_output.cpu()

            # Compress and save using torch.save and zstd
            buffer = io.BytesIO()
            torch.save(tokens_cpu, buffer)
            buffer.seek(0)
            cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL) # Assumes COMPRESSION_LEVEL is defined
            with open(tokenized_filepath, 'wb') as f:
                with cctx.stream_writer(f) as compressor:
                    compressor.write(buffer.read())
            buffer.close()
            log_statement('debug', f"{LOG_INS}:DEBUG>>Saved tokenized output to {tokenized_filepath.name}", main_logger=main_logger_name)
        except Exception as save_err:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to save tokenized output {tokenized_filepath.name}: {save_err}", main_logger=main_logger_name, exc_info=True)
            if tokenized_filepath.exists(): # Cleanup partial file
                try: tokenized_filepath.unlink()
                except OSError: pass
            return None

        # --- Return structured result matching TOKENIZED_REPO_COLUMNS ---
        tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown_tokenizer')
        result_dict = {
            COL_TOKENIZED_PATH: str(tokenized_filepath.resolve()), # Absolute path to the *.pt.zst file
            COL_PROCESSED_PATH: processed_filepath_str, # Absolute path to the *.proc file used as input
            COL_HASH: processed_file_hash, # Hash of the processed file (*.proc) that was tokenized
            'tokenizer_name': tokenizer_name, # Name of the tokenizer used
            # Add other columns expected by TOKENIZED_REPO_COLUMNS, getting defaults if needed
            COL_LAST_UPDATED: time.time(), # Timestamp of this tokenization event
            # Fill others potentially needed for joins/info, e.g., original filepath if available
            # COL_FILEPATH: ??? # Needs to be passed into the wrapper or looked up
            # COL_STATUS: STATUS_TOKENIZED # Status is usually updated after collecting results
        }
        # Ensure all columns from TOKENIZED_REPO_COLUMNS are present
        final_result = {col: result_dict.get(col, pd.NA) for col in TOKENIZED_REPO_COLUMNS} # Use constant, default to NA

        log_statement('debug', f"{LOG_INS}:DEBUG>>Wrapper success for {processed_filepath.name}", main_logger=main_logger_name)
        return final_result

    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Unhandled error in wrapper for {processed_filepath_str}: {e}", Path(__file__).stem, exc_info=True)
        return None

def process_linguistic_data():
    """
    Performs linguistic processing on files marked as PROCESSED.

    Uses SemanticLabeler (if available) to generate labels for textual data.
    Updates the repository with results and new status.

    Args:
        app_state (dict): Dictionary holding the application state,
                          including 'repo' (RepoHandler instance) and 'config'.
    """
    global app_state
    log_statement('info', f"{LOG_INS}:INFO>>Starting linguistic data processing...", __file__)
    print("\n--- Starting Linguistic Data Processing ---")

    if 'repo' not in app_state or app_state['repo'] is None:
        log_statement('error', f"{LOG_INS}:ERROR>>Data repository not initialized. Cannot perform linguistic processing.", __file__)
        print("Error: Data repository is not initialized. Please set the data directory first.")
        return

    if 'config' not in app_state or app_state['config'] is None:
        log_statement('error', f"{LOG_INS}:ERROR>>Configuration not loaded. Cannot perform linguistic processing.", __file__)
        print("Error: Configuration not loaded. Aborting.")
        return

    repo = app_state['repo']
    config = app_state['config']
    data_processor = app_state.get('data_processor') # Get if already instantiated

    if data_processor is None:
        log_statement('debug', f"{LOG_INS}:DEBUG>>DataProcessor not found in app_state, attempting instantiation.", __file__)
        try:
            # Ensure config structure is correctly accessed
            dp_config = config.get('DataProcessingConfig', {})
            output_dir_str = dp_config.get('output_directory', './processed_data')
            if not output_dir_str:
                 raise ValueError("Output directory for DataProcessor is not configured.")
            output_dir = Path(output_dir_str)

            # Check if DataProcessor class is the dummy one
            if 'DataProcessor' not in globals() or DataProcessor.__name__ == 'DataProcessor': # Checks if it's the real class
                 data_processor = DataProcessor(repo, output_directory=output_dir)
                 app_state['data_processor'] = data_processor # Store it back
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Instantiated DataProcessor for linguistic processing.", __file__)
            else:
                 raise ImportError("DataProcessor class is not available (dummy class present).") # Raise specific error

        except (ImportError, NameError, ValueError, KeyError, Exception) as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to instantiate DataProcessor: {e}", __file__, exc_info=True)
            log_statement('error', f"{LOG_INS}:ERROR>>Could not initialize the Data Processor. Linguistic processing cannot continue. Details in logs.")
            return

    # --- Semantic Labeling (Example Linguistic Task) ---
    labeler = None
    try:
        # Check if SemanticLabeler class is the dummy one
        if 'SemanticLabeler' in globals() and SemanticLabeler.__name__ == 'SemanticLabeler': # Checks if it's the real class
            # Pass config if needed by the actual SemanticLabeler constructor
            labeler = SemanticLabeler(config=config)
            log_statement('info', f"{LOG_INS}:INFO>>SemanticLabeler initialized.", __file__)
        else:
            log_statement('warning', f"{LOG_INS}:WARNING>>SemanticLabeler class is not available (dummy class present). Skipping semantic labeling.", __file__)
            print("Warning: SemanticLabeler component not found. Skipping semantic labeling step.")

    except (NameError, Exception) as e: # Catch NameError just in case, plus general exceptions
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to initialize SemanticLabeler: {e}", __file__, exc_info=True)
        log_statement('error', f"{LOG_INS}:ERROR>>Could not initialize the Semantic Labeler. Skipping. Details in logs.")
        labeler = None # Ensure labeler is None if init failed

    # --- Processing Loop ---
    try:
        files_to_process_df = repo.get_files_by_status(STATUS_PROCESSED)
    except Exception as e:
         log_statement('error', f"{LOG_INS}:ERROR>>Failed to retrieve files from repository: {e}", __file__, exc_info=True)
         print("Error: Could not retrieve file list from repository. Aborting. Details in logs.")
         return

    if files_to_process_df.empty:
        log_statement('info', f"{LOG_INS}:INFO>>No files found with status '{STATUS_PROCESSED}'. Linguistic processing step skipped.", __file__)
        print("No files are ready for linguistic processing (Status 'PROCESSED').")
        print("Ensure data has been processed using option [2] first.")
        return

    log_statement('info', f"{LOG_INS}:INFO>>Found {len(files_to_process_df)} files to process linguistically.", __file__)
    print(f"Found {len(files_to_process_df)} processed files to analyze linguistically.")

    processed_count = 0
    failed_count = 0

    # Ensure necessary columns exist in the repository DataFrame
    # Use try-except for potentially missing repo.df
    try:
        if repo.df is None:
             raise AttributeError("Repository DataFrame (repo.df) is None.")
        if COL_SEMANTIC_LABEL not in repo.df.columns:
             repo.df[COL_SEMANTIC_LABEL] = None # Or appropriate dtype like object
             log_statement('debug', f"{LOG_INS}:DEBUG>>Added '{COL_SEMANTIC_LABEL}' column to repository.", __file__)
        if COL_LINGUISTIC_METADATA not in repo.df.columns:
             repo.df[COL_LINGUISTIC_METADATA] = None # Or appropriate dtype like object
             log_statement('debug', f"{LOG_INS}:DEBUG>>Added '{COL_LINGUISTIC_METADATA}' column to repository.", __file__)
    except AttributeError as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to access or modify repository DataFrame columns: {e}", __file__, exc_info=True)
        print("Error: Problem accessing repository data structure. Aborting. Details in logs.")
        return


    for index, row in files_to_process_df.iterrows():
        filepath_str = row[COL_FILEPATH] # Get path string from repo
        filepath = Path(filepath_str) # Convert to Path object
        file_hash = row[COL_HASH]
        log_statement('debug', f"{LOG_INS}:DEBUG>>Processing linguistically: {filepath} (Hash: {file_hash})", __file__)

        try:
            # Update status to indicate processing
            repo.update_entry(filepath_str, {COL_STATUS: STATUS_LINGUISTIC_PROCESSING}) # Pass path string

            # Check data type - only process textual data for this example
            data_type = row.get(COL_DTYPE) # Assumes DataProcessor adds this column
            if data_type != TYPE_TEXTUAL:
                log_statement('debug', f"{LOG_INS}:DEBUG>>Skipping non-textual file: {filepath} (Type: {data_type})", __file__)
                # Revert status - simpler to just not process than track another status
                repo.update_entry(filepath_str, {COL_STATUS: STATUS_PROCESSED})
                continue

            # Get the path to the *processed* data file generated by DataProcessor
            processed_path = repo.get_processed_path(filepath_str, app_state) # Pass path string and app_state
            if processed_path is None or not processed_path.exists():
                 log_statement('warning', f"{LOG_INS}:WARNING>>Processed file path not found or file does not exist for {filepath}. Skipping.", __file__)
                 print(f"Warning: Could not find processed data for {filepath}. Skipping.")
                 repo.update_entry(filepath_str, {COL_STATUS: STATUS_LINGUISTIC_FAILED, COL_ERROR_INFO: "Processed file missing"})
                 failed_count += 1
                 continue

            # --- Read Processed Content ---
            content = ""
            try:
                log_statement('debug', f"{LOG_INS}:DEBUG>>Attempting to read processed content from {processed_path}", __file__)
                reader_cls = get_reader_class(processed_path) # Use existing reader discovery
                if reader_cls:
                    reader = reader_cls(processed_path)
                    content = reader.read() # Assumes a simple read() method exists
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Successfully read processed content using {reader_cls.__name__} from {processed_path}", __file__)
                else:
                    # Fallback for unknown types or if get_reader_class fails
                    # Needs care for binary reads if compression involved but not handled by readers
                    try:
                         with open(processed_path, 'r', encoding='utf-8') as f: # Basic text fallback
                             content = f.read()
                         log_statement('debug', f"{LOG_INS}:DEBUG>>Read processed content using basic text file open from {processed_path}", __file__)
                    except UnicodeDecodeError:
                         log_statement('warning', f"{LOG_INS}:WARNING>>Basic text read failed (UnicodeDecodeError) for {processed_path}. Trying binary read (may not be useful).", __file__)
                         try:
                              with open(processed_path, 'rb') as f: # Basic binary fallback
                                  content = f.read().decode('utf-8', errors='ignore') # Attempt decode, ignore errors
                              log_statement('debug', f"{LOG_INS}:DEBUG>>Read processed content using basic binary file open (decoded) from {processed_path}", __file__)
                         except Exception as bin_read_err:
                              log_statement('error', f"{LOG_INS}:ERROR>>Basic binary read also failed for {processed_path}: {bin_read_err}", __file__, exc_info=True)
                              raise IOError(f"Failed to read file {processed_path} with text or binary fallback.") from bin_read_err
                    except Exception as text_read_err:
                         log_statement('error', f"{LOG_INS}:ERROR>>Basic text read failed for {processed_path}: {text_read_err}", __file__, exc_info=True)
                         raise IOError(f"Failed to read file {processed_path} with text fallback.") from text_read_err


            except Exception as read_err:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to read processed content from {processed_path}: {read_err}", __file__, exc_info=True)
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to read processed data for {filepath}. Skipping. Details in logs.")
                repo.update_entry(filepath_str, {COL_STATUS: STATUS_LINGUISTIC_FAILED, COL_ERROR_INFO: f"Failed to read processed file: {read_err}"})
                failed_count += 1
                continue

            # --- Apply Linguistic Processing ---
            semantic_label = None
            linguistic_metadata = {} # Placeholder for other potential metadata

            if labeler and content:
                try:
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Calling SemanticLabeler.generate_label for {filepath}", __file__)
                    semantic_label = labeler.generate_label(content)
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Generated semantic label '{semantic_label}' for {filepath}", __file__)
                    # Example: Add more processing here (POS tagging, NER, etc.)
                    # linguistic_metadata['pos_tags'] = perform_pos_tagging(content) # Placeholder
                    # linguistic_metadata['entities'] = perform_ner(content) # Placeholder

                except Exception as label_err:
                    log_statement('error', f"{LOG_INS}:ERROR>>Semantic labeling failed for {filepath}: {label_err}", __file__, exc_info=True)
                    print(f"Warning: Semantic labeling failed for {filepath}. Details in logs.")
                    # Decide if failure prevents marking as processed or just leaves label empty
                    # For now, allow processing to finish but label will be None

            # --- Update Repository ---
            update_data = {
                COL_STATUS: STATUS_LINGUISTIC_PROCESSED,
                COL_SEMANTIC_LABEL: semantic_label,
                 # Store complex data as JSON string, handle potential serialization errors
                COL_LINGUISTIC_METADATA: json.dumps(linguistic_metadata) if linguistic_metadata else None
            }
            repo.update_entry(filepath_str, update_data)
            processed_count += 1
            log_statement('info', f"{LOG_INS}:INFO>>Successfully processed linguistically: {filepath}", __file__)

        except Exception as e:
            failed_count += 1
            # Use logger formatting for critical errors
            log_statement('critical', f"{LOG_INS}:CRITICAL>>Unhandled error during linguistic processing for {filepath}: {e}", __file__, exc_info=True)
            log_statement('error', f"{LOG_INS}:ERROR>>An unexpected error occurred while processing {filepath}. Skipping. Details in logs.")
            # Attempt to mark as failed in repository
            try:
                repo.update_entry(filepath_str, {COL_STATUS: STATUS_LINGUISTIC_FAILED, COL_ERROR_INFO: str(e)})
            except Exception as repo_update_err:
                 log_statement('critical', f"{LOG_INS}:CRITICAL>>Failed to update repository status to FAILED for {filepath} after error: {repo_update_err}", __file__, exc_info=True)

    # --- Finalization ---
    log_statement('info', f"{LOG_INS}:INFO>>Linguistic processing finished. Processed: {processed_count}, Failed: {failed_count}.", __file__)
    print(f"\n--- Linguistic Processing Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")

    if processed_count > 0 or failed_count > 0:
        try:
            log_statement('info', f"{LOG_INS}:INFO>>Attempting to save repository after linguistic processing.", __file__)
            repo.save_repo()
            log_statement('info', f"{LOG_INS}:INFO>>Repository saved after linguistic processing.", __file__)
            print("Repository changes saved.")
        except Exception as e:
            log_statement('critical', f"{LOG_INS}:CRITICAL>>Failed to save repository after linguistic processing: {e}", __file__, exc_info=True)
            print(f"CRITICAL ERROR: Failed to save repository changes! Details in logs.")

# def process_linguistic_data():
#     """
#     Handles Option 2: Process Linguistic Data using DataProcessor.
#     Determines base directory, processes files via DataProcessor,
#     and updates app_state with the processed repository details.
#     """
#     global app_state
#     func_name = inspect.currentframe().f_code.co_name
#     log_statement('info', f"{LOG_INS}:INFO>>Starting linguistic data processing.", Path(__file__).stem)

#     # --- 1. Pre-checks ---
#     if app_state.get('main_repo_df') is None or app_state.get('main_repo_path') is None:
#         log_statement('error', f"{LOG_INS}:ERROR>>Error: No main data repository loaded/path set. Please run Option 1 first.", Path(__file__).stem)
#         print("Error: Main repository not set. Please run Option 1 first.")
#         return
#     if app_state.get('config') is None:
#         try:
#             app_state['config'] = load_config()
#             log_statement('info', f"{LOG_INS}:INFO>>Config loaded.", Path(__file__).stem)
#         except Exception as e:
#             app_state['config'] = {}
#             log_statement('error', f"{LOG_INS}:ERROR>>Failed to load config: {e}. Using empty config.", Path(__file__).stem, exc_info=True)
#             print("Error: Failed to load system configuration.")
#             return

#     config = app_state['config']
#     main_repo_path = Path(app_state['main_repo_path'])
#     target_repo_path_str = str(main_repo_path.resolve()) # Use resolved path string for DataProcessor init

#     # --- 2. Determine Base Directory Filter (Optional) ---
#     repo_index = _load_repository_index()
#     current_repo_hash = None
#     base_dir_filter_path = None
#     try:
#         filename_parts = main_repo_path.stem.split('data_repository_')
#         if len(filename_parts) > 1:
#              current_repo_hash = filename_parts[-1].split('.')[0]
#              if current_repo_hash in repo_index:
#                  base_dir_filter_path = repo_index[current_repo_hash].get(INDEX_KEY_PATH)
#                  if base_dir_filter_path: log_statement('info', f"{LOG_INS}:INFO>>Determined base directory for filtering: {base_dir_filter_path}", Path(__file__).stem)
#                  else: log_statement('warning', f"{LOG_INS}:WARNING>>Could not determine base directory from index for hash {current_repo_hash}.", Path(__file__).stem)
#              else: log_statement('warning', f"{LOG_INS}:WARNING>>Current repo hash '{current_repo_hash}' not found in index.", Path(__file__).stem)
#         else: log_statement('warning', f"{LOG_INS}:WARNING>>Could not parse repo hash from filename: {main_repo_path.name}", Path(__file__).stem)
#     except Exception as e:
#          log_statement('warning', f"{LOG_INS}:WARNING>>Could not reliably determine base directory for filtering from path {main_repo_path}: {e}. Proceeding without filter.", Path(__file__).stem)
#          base_dir_filter_path = None

#     dp = None # Initialize dp to None for finally block
#     processed_repo_path_determined = None # Track the path DataProcessor used
#     processed_repo_load_success = False   # Track if loading the processed repo worked

#     try:
#         # --- 3. Initialize DataProcessor ---
#         if 'DataProcessor' not in globals() or DataProcessor is None: raise ImportError("DataProcessor class not available.")
#         max_workers = _get_max_workers(config)
#         dp = DataProcessor(max_workers=max_workers, repo_path_override=target_repo_path_str)
#         log_statement('info', f"{LOG_INS}:INFO>>DataProcessor initialized for processing repository: {target_repo_path_str}", Path(__file__).stem)
#         # Store the path the DataProcessor intends to use for the processed repo
#         processed_repo_path_determined = dp.processed_repo_filepath

#         # --- 4. Execute Processing ---
#         log_statement('info', f"{LOG_INS}:INFO>>Calling DataProcessor.process_all (filter: {base_dir_filter_path})...", Path(__file__).stem)
#         statuses = ('discovered', 'error', STATUS_NEW) if 'STATUS_NEW' in globals() else ('discovered', 'error')
#         if not hasattr(dp, 'process_all'): raise AttributeError("DataProcessor instance lacks 'process_all' method.")
#         dp.process_all(base_dir_filter=base_dir_filter_path, statuses_to_process=statuses)
#         log_statement('info', f"{LOG_INS}:INFO>>DataProcessor.process_all finished.", Path(__file__).stem)

#         # --- 5. Reload Main Repository DF (reflects status updates from DataProcessor) ---
#         log_statement('info', f"{LOG_INS}:INFO>>Reloading main repository DataFrame into app_state after processing.", Path(__file__).stem)
#         repo_after_proc = RepoHandler(metadata_compression='zst', repository_path=main_repo_path)
#         app_state['main_repo_df'] = repo_after_proc.df
#         log_statement('info', f"{LOG_INS}:INFO>>Main repo DF reloaded ({len(app_state['main_repo_df']) if app_state['main_repo_df'] is not None else 0} entries).", Path(__file__).stem)

#         # --- 6. Update Processed Repository State in app_state ---
#         processed_repo_df_loaded = None
#         if processed_repo_path_determined and processed_repo_path_determined.exists():
#              log_statement('info', f"{LOG_INS}:INFO>>Processed repository file found at: {processed_repo_path_determined}. Attempting to load...", Path(__file__).stem)
#              try:
#                  if RepoHandler is None: raise ImportError("RepoHandler class not available.")
#                  processed_repo = RepoHandler(metadata_compression='zst', repository_path=processed_repo_path_determined)
#                  processed_repo_df_loaded = processed_repo.df
#                  if processed_repo_df_loaded is not None and not processed_repo_df_loaded.empty:
#                       app_state['processed_repo_path'] = str(processed_repo_path_determined.resolve())
#                       app_state['processed_repo_df'] = processed_repo_df_loaded
#                       processed_repo_load_success = True # Mark success
#                       log_statement('info', f"{LOG_INS}:INFO>>Successfully loaded processed DataFrame ({len(processed_repo_df_loaded)} entries) into app_state.", Path(__file__).stem)
#                  elif processed_repo_df_loaded is not None and processed_repo_df_loaded.empty:
#                       log_statement('warning', f"{LOG_INS}:WARNING>>Processed repository file is empty: {processed_repo_path_determined}. State 'processed_repo_df' set to empty DataFrame.", Path(__file__).stem)
#                       app_state['processed_repo_path'] = str(processed_repo_path_determined.resolve()) # Path exists, even if empty
#                       app_state['processed_repo_df'] = processed_repo_df_loaded # Store the empty DF
#                       processed_repo_load_success = True # Loading technically succeeded, even if empty
#                  else: # RepoHandler loaded None
#                      log_statement('warning', f"{LOG_INS}:WARNING>>RepoHandler loaded None for processed repo: {processed_repo_path_determined}. State 'processed_repo_df' set to None.", Path(__file__).stem)
#                      app_state['processed_repo_path'] = None # Clear path if load returned None
#                      app_state['processed_repo_df'] = None
#              except ImportError:
#                   log_statement('critical', f"{LOG_INS}:CRITICAL>>RepoHandler class not available. Cannot load processed repository.", Path(__file__).stem)
#                   app_state['processed_repo_path'] = None
#                   app_state['processed_repo_df'] = None
#              except Exception as load_err:
#                   log_statement('error', f"{LOG_INS}:ERROR>>Failed to load processed repository DataFrame from {processed_repo_path_determined}: {load_err}", Path(__file__).stem, exc_info=True)
#                   app_state['processed_repo_path'] = None
#                   app_state['processed_repo_df'] = None
#         else:
#              log_statement('warning', f"{LOG_INS}:WARNING>>Processed repository file path not determined or file does not exist ({processed_repo_path_determined}). Processed state not updated.", Path(__file__).stem)
#              app_state['processed_repo_path'] = None
#              app_state['processed_repo_df'] = None

#         # --- 7. Clear Subsequent State if Processed State Failed/Empty ---
#         if not processed_repo_load_success or (processed_repo_df_loaded is not None and processed_repo_df_loaded.empty):
#             log_statement('warning', f"{LOG_INS}:WARNING>>Processed repository state is not valid or empty. Clearing subsequent tokenization state.", Path(__file__).stem)
#             app_state['tokenized_repo_path'] = None
#             app_state['tokenized_repo_df'] = None

#         log_statement('info', f"{LOG_INS}:INFO>>Linguistic data processing function finished.", Path(__file__).stem)

#     except ImportError as imp_err:
#          log_statement('critical', f"{LOG_INS}:CRITICAL>>Missing import required for processing: {imp_err}", Path(__file__).stem, exc_info=True)
#          log_statement('error', f"{LOG_INS}:ERROR>>A required library is missing ({imp_err}). Processing cannot continue.")
#     except AttributeError as attr_err:
#          log_statement('error', f"{LOG_INS}:ERROR>>Missing required method/attribute (e.g., in DataProcessor): {attr_err}", Path(__file__).stem, exc_info=True)
#          log_statement('error', f"{LOG_INS}:ERROR>>A required component is missing ({attr_err}).")
#     except Exception as e:
#         log_statement('error', f"{LOG_INS}:ERROR>>Error during linguistic data processing: {e}", Path(__file__).stem, exc_info=True)
#         print(f"An unexpected error occurred during processing: {e}. Check logs.")
#     finally:
#         # --- 8. Cleanup ---
#         if dp is not None: # Check if dp was successfully assigned
#             # Check if dp has a 'shutdown' method before calling
#             if hasattr(dp, 'shutdown') and callable(dp.shutdown):
#                  try:
#                      log_statement('info', f"{LOG_INS}:INFO>>Shutting down DataProcessor resources...", Path(__file__).stem)
#                      dp.shutdown()
#                      log_statement('info', f"{LOG_INS}:INFO>>DataProcessor resources shut down.", Path(__file__).stem)
#                  except Exception as shutdown_e:
#                      log_statement('error', f"{LOG_INS}:ERROR>>Error during DataProcessor shutdown: {shutdown_e}", Path(__file__).stem, exc_info=True)
#             elif hasattr(dp, 'executor') and dp.executor is not None: # Fallback: try shutting down executor directly
#                 try:
#                     log_statement('info', f"{LOG_INS}:INFO>>Shutting down DataProcessor executor directly...", Path(__file__).stem)
#                     dp.executor.shutdown(wait=True)
#                     log_statement('info', f"{LOG_INS}:INFO>>DataProcessor executor shut down.", Path(__file__).stem)
#                 except Exception as shutdown_e:
#                      log_statement('error', f"{LOG_INS}:ERROR>>Error during DataProcessor executor shutdown: {shutdown_e}", Path(__file__).stem, exc_info=True)
                
def _load_tokenizer():
    m_workers = os.cpu_count()
    tokenizer = Tokenizer(max_workers=m_workers)
    return tokenizer

def tokenize_data():
    """
    Handles Option 3: Data Tokenization using project constants, with thread-safe collection.
    Loads processed data correctly using RepoHandler and compares against existing
    tokenized data to only process changed/new files. Includes diagnostic logging.
    """
    global app_state
    log_statement('info', f"{LOG_INS}:INFO>>Starting data tokenization.", Path(__file__).stem)

    # --- 1. Pre-checks & Load Processed Repo CORRECTLY ---
    processed_repo_path_str = app_state.get('processed_repo_path')
    if not processed_repo_path_str:
        log_statement('error', f"{LOG_INS}:ERROR>>Error: No processed data repository path set. Please run Option 2 first.", Path(__file__).stem)
        print("Error: Processed repository path not set. Please run Option 2 first.")
        return

    processed_repo_path = Path(processed_repo_path_str)
    processed_repo_df = None # Initialize

    log_statement('info', f"{LOG_INS}:INFO>>Loading processed repository from: {processed_repo_path}", Path(__file__).stem)
    try:
        # --- Load DataFrame using RepoHandler to ensure schema ---
        if not processed_repo_path.exists():
             raise FileNotFoundError(f"Processed repository file not found at {processed_repo_path}")

        # Use RepoHandler to load - this applies _load_repo logic
        proc_repo_loader = RepoHandler(metadata_compression='zst', repository_path=processed_repo_path)
        processed_repo_df = proc_repo_loader.df # Get the loaded, schema-compliant DataFrame

        if processed_repo_df is None:
             raise ValueError("RepoHandler failed to load the processed DataFrame.")
        if processed_repo_df.empty:
             log_statement("warning", f"{LOG_INS}:WARNING>>Processed repository DataFrame is empty. Nothing to tokenize.", Path(__file__).stem)
             print("Warning: Processed data repository is empty. Nothing to tokenize.")
             return # Exit if empty

        log_statement('info', f"{LOG_INS}:INFO>>Successfully loaded processed repository with {len(processed_repo_df)} entries.", Path(__file__).stem)

    except FileNotFoundError as fnf_err:
         log_statement('error', f"{LOG_INS}:ERROR>>{fnf_err}", main_logger=Path(__file__).stem)
         log_statement('error', f"{LOG_INS}:ERROR>>Cannot find the processed repository file expected at {processed_repo_path}.")
         print("Please ensure Option 2 (Process Linguistic Data) ran successfully and created the file.")
         return
    except Exception as load_err:
         log_statement('error', f"{LOG_INS}:ERROR>>Error loading processed repository from {processed_repo_path}: {load_err}", Path(__file__).stem, exc_info=True)
         print(f"Error loading processed data: {load_err}. Cannot proceed with tokenization.")
         return


    # --- Config Loading ---
    if app_state.get('config') is None:
        try:
            app_state['config'] = load_config() # Assuming load_config is available
            log_statement('info', f"{LOG_INS}:INFO>>Config loaded.", Path(__file__).stem)
        except Exception as e:
            app_state['config'] = {}
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to load config: {e}. Using empty config.", Path(__file__).stem, exc_info=True)
            print("Warning: Failed to load system configuration. Using defaults.")
    config = app_state['config']

    # --- 2. Extract Hash from Processed Repo Path ---
    try:
        filename_stem = processed_repo_path.stem
        if filename_stem.endswith('.csv'): # Remove .csv if present before splitting
             filename_stem = filename_stem[:-4]
        filename_parts = filename_stem.split('processed_repository_')
        if len(filename_parts) > 1:
             processed_repo_hash = filename_parts[-1]
             log_statement('debug', f"{LOG_INS}:DEBUG>>Extracted processed repo hash: {processed_repo_hash}", Path(__file__).stem)
        else: raise ValueError("Filename does not follow expected 'processed_repository_*_hash' pattern")
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Cannot extract hash from processed repo path '{processed_repo_path}'. Aborting tokenization. Error: {e}", Path(__file__).stem)
        return

    # --- Ensure Processed DataFrame Schema & ADD DIAGNOSTIC LOGGING ---
    required_processed_cols = MAIN_REPO_HEADER # Use constant
    log_statement('debug', f"{LOG_INS}:DEBUG>>Selecting required columns. Required list: {required_processed_cols}", Path(__file__).stem)

    # --- DIAGNOSTIC LOGGING ---
    actual_cols = processed_repo_df.columns.tolist()
    log_statement('info', f"{LOG_INS}:INFO>>Columns ACTUALLY PRESENT in loaded processed_repo_df: {actual_cols}", Path(__file__).stem) # INFO level for visibility
    missing_cols_check = [col for col in required_processed_cols if col not in actual_cols]
    if missing_cols_check:
        log_statement('error', f"{LOG_INS}:ERROR>>Columns confirmed missing before slicing attempt: {missing_cols_check}", Path(__file__).stem)
    else:
        log_statement('debug', f"{LOG_INS}:DEBUG>>All required columns appear present before slicing.", Path(__file__).stem)
    # --- END DIAGNOSTIC LOGGING ---

    # The potentially failing line: Wrap in try-except for clarity
    try:
        # Attempt to select the required columns
        processed_repo_df_subset = processed_repo_df[required_processed_cols].copy()
        log_statement('debug', f"{LOG_INS}:DEBUG>>Successfully selected required columns.", Path(__file__).stem)
    except KeyError as e:
        # This block will catch the error if columns are still missing
        log_statement('critical', f"{LOG_INS}:CRITICAL>> KeyError selecting columns: {e}. DF Columns were: {actual_cols}", Path(__file__).stem)
        print(f"Critical Error: Failed to select required columns from processed data. Columns present: {actual_cols}. Error: {e}")
        print("This indicates an issue with the processed repository file structure or the loading process.")
        return # Stop execution
    except Exception as e_slice:
         # Catch any other unexpected errors during slicing
         log_statement('critical', f"{LOG_INS}:CRITICAL>> Error during column selection: {e_slice}. DF Columns were: {actual_cols}", Path(__file__).stem)
         print(f"Critical Error: Failed to select required columns. Check logs. Error: {e_slice}")
         return # Stop execution


    # --- Ensure Correct Data Types for Logic ---
    # (Type conversion logic remains the same as previous version)
    try:
        path_cols = [COL_FILEPATH, COL_PROCESSED_PATH]
        hash_cols = [COL_HASH]
        for col in path_cols:
             if col in processed_repo_df_subset.columns:
                  processed_repo_df_subset[col] = processed_repo_df_subset[col].astype(str).fillna('')
        for col in hash_cols:
             if col in processed_repo_df_subset.columns:
                  processed_repo_df_subset[col] = processed_repo_df_subset[col].astype(str).fillna('')
        log_statement('debug', f"{LOG_INS}:DEBUG>>Ensured necessary string types for paths/hashes.", Path(__file__).stem)
    except Exception as type_err:
        log_statement('error', f"{LOG_INS}:ERROR>>Error converting types in processed DF subset: {type_err}", Path(__file__).stem)
        return


    # --- 3. Configuration and Tokenizer Initialization ---
    # ... (remains the same as previous version) ...
    try:
        tokenizer_config = config.get('tokenizer', {})
        model_name = tokenizer_config.get('model_name', 'bert-base-uncased')
        max_workers = _get_max_workers(config)
        output_dir = Path(f"{TOKENIZED_DATA_DIR}/{processed_repo_hash}") # Use constant and hash
        output_dir.mkdir(parents=True, exist_ok=True)
        log_statement('info', f"{LOG_INS}:INFO>>Token output directory set to: {output_dir}", Path(__file__).stem)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not hasattr(tokenizer, 'name_or_path'): tokenizer.name_or_path = model_name
        log_statement('info', f"{LOG_INS}:INFO>>AutoTokenizer initialized for '{model_name}'", Path(__file__).stem)
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error initializing tokenizer '{model_name}' or setting up output dir: {e}", Path(__file__).stem, exc_info=True)
        print(f"Error setting up tokenizer or output directory: {e}")
        return


    # --- 4. Load Existing Tokenized Repository & Determine Files to Process ---
    # ... (remains the same as previous version) ...
    tokenized_repo_filename = DATA_REPO_DIR / f"tokenized_repository_{processed_repo_hash}.csv.zst" # Use constant and hash
    existing_tokenized_df = pd.DataFrame()
    compare_cols_existing = [COL_PROCESSED_PATH, COL_HASH, 'tokenizer_name'] # Use constants
    if tokenized_repo_filename.exists():
        log_statement('info', f"{LOG_INS}:INFO>>Existing tokenized repository found at {tokenized_repo_filename}. Loading for comparison.", Path(__file__).stem)
        try:
            tokenized_repo = RepoHandler(metadata_compression='zst', repository_path=tokenized_repo_filename)
            loaded_df = tokenized_repo.df
            if loaded_df is not None and not loaded_df.empty:
                cols_to_select = [col for col in compare_cols_existing if col in loaded_df.columns]
                missing_compare_cols = [col for col in compare_cols_existing if col not in loaded_df.columns]
                if missing_compare_cols: log_statement('warning', f"{LOG_INS}:WARNING>>Loaded tokenized repo missing comparison columns: {missing_compare_cols}.", Path(__file__).stem)
                if cols_to_select:
                    existing_tokenized_df = loaded_df[cols_to_select].copy()
                    if COL_PROCESSED_PATH in existing_tokenized_df.columns: existing_tokenized_df[COL_PROCESSED_PATH] = existing_tokenized_df[COL_PROCESSED_PATH].astype(str).fillna('')
                    if COL_HASH in existing_tokenized_df.columns: existing_tokenized_df[COL_HASH] = existing_tokenized_df[COL_HASH].astype(str).fillna('')
                    if 'tokenizer_name' in existing_tokenized_df.columns: existing_tokenized_df['tokenizer_name'] = existing_tokenized_df['tokenizer_name'].astype(str).fillna('')
                    log_statement('info', f"{LOG_INS}:INFO>>Loaded and prepared existing tokenized repository ({len(existing_tokenized_df)} entries) for comparison.", Path(__file__).stem)
                else: log_statement('warning', f"{LOG_INS}:WARNING>>No usable comparison columns found in loaded tokenized repo. Will re-tokenize all.", Path(__file__).stem)
            else: log_statement('info', f"{LOG_INS}:INFO>>Existing tokenized repository was empty or failed load. Tokenizing all files.", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error loading/parsing tokenized repo {tokenized_repo_filename}: {e}. Retokenizing all.", Path(__file__).stem, exc_info=True)
            existing_tokenized_df = pd.DataFrame()
    else:
        log_statement('info', f"{LOG_INS}:INFO>>No existing tokenized repository found. Tokenizing all files.", Path(__file__).stem)

    # Determine files needing tokenization via merge
    files_to_tokenize_args = []
    processed_subset_for_compare = processed_repo_df_subset[[COL_PROCESSED_PATH, COL_HASH]].copy()
    # ... (Merge and comparison logic remains the same as previous version) ...
    if not existing_tokenized_df.empty and all(col in existing_tokenized_df.columns for col in compare_cols_existing):
        log_statement('debug', f"{LOG_INS}:DEBUG>>Comparing processed files against existing tokenized records.", Path(__file__).stem)
        try:
            comparison_df = pd.merge(processed_subset_for_compare, existing_tokenized_df, how='left', on=COL_PROCESSED_PATH, suffixes=('_current', '_repo'))
            needs_tokenizing_mask = ( comparison_df[COL_HASH + '_repo'].isnull() | (comparison_df[COL_HASH + '_current'] != comparison_df[COL_HASH + '_repo']) | (comparison_df['tokenizer_name'].fillna('') != tokenizer.name_or_path) )
            needs_tokenizing_df = comparison_df[needs_tokenizing_mask]
            files_to_tokenize_args = [ (row[COL_PROCESSED_PATH], row[COL_HASH + '_current'], output_dir, tokenizer) for _, row in needs_tokenizing_df.iterrows() if row[COL_PROCESSED_PATH] and Path(row[COL_PROCESSED_PATH]).is_file() ]
            num_invalid_paths = len(needs_tokenizing_df) - len(files_to_tokenize_args)
            if num_invalid_paths > 0: log_statement('warning', f"{LOG_INS}:WARNING>>Skipped {num_invalid_paths} entries needing tokenization due to invalid/missing processed paths.", Path(__file__).stem)
            log_statement('info', f"{LOG_INS}:INFO>>Comparison complete. Identified {len(files_to_tokenize_args)} files needing tokenization.", Path(__file__).stem)
        except Exception as merge_err:
             log_statement('error', f"{LOG_INS}:ERROR>>Error during comparison merge: {merge_err}. Retokenizing all valid processed files.", Path(__file__).stem, exc_info=True)
             existing_tokenized_df = pd.DataFrame()
             files_to_tokenize_args = [ (row[COL_PROCESSED_PATH], row[COL_HASH], output_dir, tokenizer) for _, row in processed_subset_for_compare.iterrows() if row[COL_PROCESSED_PATH] and Path(row[COL_PROCESSED_PATH]).is_file() ]
    else:
         log_statement('info', f"{LOG_INS}:INFO>>Tokenizing all valid processed files.", Path(__file__).stem)
         files_to_tokenize_args = [ (row[COL_PROCESSED_PATH], row[COL_HASH], output_dir, tokenizer) for _, row in processed_subset_for_compare.iterrows() if row[COL_PROCESSED_PATH] and Path(row[COL_PROCESSED_PATH]).is_file() ]
         num_invalid_paths = len(processed_subset_for_compare) - len(files_to_tokenize_args)
         if num_invalid_paths > 0: log_statement('warning', f"{LOG_INS}:WARNING>>Skipped {num_invalid_paths} processed entries due to invalid/missing paths.", Path(__file__).stem)


    # --- 5. Parallel Tokenization ---
    # ... (remains the same as previous version) ...
    tokenized_results_temp = []
    if files_to_tokenize_args:
        log_statement('info', f"{LOG_INS}:INFO>>Tokenizing {len(files_to_tokenize_args)} files (using up to {max_workers} workers)...", Path(__file__).stem)
        if not callable(globals().get('_process_file_wrapper')):
            log_statement('critical', f"{LOG_INS}:CRITICAL>>Helper function '_process_file_wrapper' is not defined. Tokenization aborted.", Path(__file__).stem)
            return
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Tokenizer') as executor:
            future_to_args = {executor.submit(_process_file_wrapper, args): args for args in files_to_tokenize_args}
            pbar = tqdm(as_completed(future_to_args), total=len(files_to_tokenize_args), desc="Tokenizing files", unit="file", leave=False)
            for future in pbar:
                args = future_to_args[future]; processed_path_arg = args[0]
                try:
                    result = future.result()
                    if result: tokenized_results_temp.append(result)
                    else: log_statement('warning', f"{LOG_INS}:WARNING>>Tokenization wrapper returned None for processed file: {Path(processed_path_arg).name}", Path(__file__).stem)
                except Exception as e: log_statement('error', f"{LOG_INS}:ERROR>>Error processing tokenization future for {Path(processed_path_arg).name}: {e}", Path(__file__).stem, exc_info=True)
        tokenized_results = tokenized_results_temp
        log_statement('info', f"{LOG_INS}:INFO>>Parallel tokenization finished. Collected {len(tokenized_results)} results.", Path(__file__).stem)
    else:
        tokenized_results = []; log_statement('info', f"{LOG_INS}:INFO>>No files require tokenization.", Path(__file__).stem)


    # --- 6. Update Repository DataFrame ---
    # ... (remains the same as previous version) ...
    log_statement('info', f"{LOG_INS}:INFO>>Updating tokenized repository DataFrame...", Path(__file__).stem)
    new_tokenized_df = pd.DataFrame(tokenized_results) if tokenized_results else pd.DataFrame()
    final_tokenized_schema_cols = TOKENIZED_REPO_COLUMNS if 'TOKENIZED_REPO_COLUMNS' in globals() else MAIN_REPO_HEADER
    if not new_tokenized_df.empty:
        for col in final_tokenized_schema_cols:
            if col not in new_tokenized_df.columns: new_tokenized_df[col] = pd.NA
        new_tokenized_df = new_tokenized_df.reindex(columns=final_tokenized_schema_cols)
    final_tokenized_df = pd.DataFrame(columns=final_tokenized_schema_cols)
    if not existing_tokenized_df.empty:
        newly_tokenized_processed_paths = set(new_tokenized_df[COL_PROCESSED_PATH]) if not new_tokenized_df.empty else set()
        valid_processed_paths = set(processed_repo_df_subset[COL_PROCESSED_PATH])
        cols_to_select_from_existing = [col for col in final_tokenized_schema_cols if col in tokenized_repo.df.columns] if 'tokenized_repo' in locals() and tokenized_repo.df is not None else []
        if cols_to_select_from_existing:
             existing_loaded_full = tokenized_repo.df[cols_to_select_from_existing]
             existing_to_keep = existing_loaded_full[ existing_loaded_full[COL_PROCESSED_PATH].astype(str).isin(valid_processed_paths) & ~existing_loaded_full[COL_PROCESSED_PATH].astype(str).isin(newly_tokenized_processed_paths) ].copy()
             log_statement('debug', f"{LOG_INS}:DEBUG>>Keeping {len(existing_to_keep)} existing tokenized entries.", Path(__file__).stem)
             if not new_tokenized_df.empty:
                 existing_to_keep = existing_to_keep.reindex(columns=final_tokenized_schema_cols); new_tokenized_df = new_tokenized_df.reindex(columns=final_tokenized_schema_cols)
                 final_tokenized_df = pd.concat([existing_to_keep, new_tokenized_df], ignore_index=True)
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Combined existing and new tokenized entries.", Path(__file__).stem)
             else: final_tokenized_df = existing_to_keep; log_statement('debug', f"{LOG_INS}:DEBUG>>Using only existing tokenized entries.", Path(__file__).stem)
        elif not new_tokenized_df.empty: final_tokenized_df = new_tokenized_df; log_statement('debug', f"{LOG_INS}:DEBUG>>Using only new tokenized results.", Path(__file__).stem)
    elif not new_tokenized_df.empty: final_tokenized_df = new_tokenized_df; log_statement('debug', f"{LOG_INS}:DEBUG>>Using only new tokenized results.", Path(__file__).stem)
    else: log_statement('info', f"{LOG_INS}:INFO>>No existing or new tokenized results.", Path(__file__).stem)
    if not final_tokenized_df.empty:
        final_tokenized_df = final_tokenized_df.reindex(columns=final_tokenized_schema_cols)
        if COL_TOKENIZED_PATH in final_tokenized_df.columns:
            final_tokenized_df = final_tokenized_df.drop_duplicates(subset=[COL_TOKENIZED_PATH], keep='last')
            final_tokenized_df = final_tokenized_df.dropna(subset=[COL_TOKENIZED_PATH])
            final_tokenized_df[COL_TOKENIZED_PATH] = final_tokenized_df[COL_TOKENIZED_PATH].astype(str)
            final_tokenized_df = final_tokenized_df[final_tokenized_df[COL_TOKENIZED_PATH] != '']
    log_statement('info', f"{LOG_INS}:INFO>>Final tokenized repository contains {len(final_tokenized_df)} entries.", Path(__file__).stem)


    # --- 7. Save & Update App State ---
    # ... (remains the same as previous version) ...
    if not final_tokenized_df.empty:
        log_statement('info', f"{LOG_INS}:INFO>>Saving tokenized repository using RepoHandler to {tokenized_repo_filename}", Path(__file__).stem)
        try:
            tok_repo_saver = RepoHandler(metadata_compression='zst', repository_path=tokenized_repo_filename)
            temp_df_to_save = final_tokenized_df.copy()
            schema = COL_SCHEMA
            for col in final_tokenized_schema_cols: # Use defined schema cols
                if col in temp_df_to_save.columns and col in schema:
                    target_dtype = schema[col]
                    try: # Apply type conversions robustly
                        if pd.api.types.is_datetime64_any_dtype(target_dtype) or 'datetime' in str(target_dtype): temp_df_to_save[col] = pd.to_datetime(temp_df_to_save[col], errors='coerce', utc=True)
                        elif target_dtype == 'Int64': temp_df_to_save[col] = pd.to_numeric(temp_df_to_save[col], errors='coerce').astype('Int64')
                        elif pd.api.types.is_string_dtype(target_dtype) or target_dtype == str: temp_df_to_save[col] = temp_df_to_save[col].astype(str).fillna('') # Save empty str for NA string
                        elif pd.api.types.is_float_dtype(target_dtype): temp_df_to_save[col] = pd.to_numeric(temp_df_to_save[col], errors='coerce').astype('Float64')
                        elif pd.api.types.is_bool_dtype(target_dtype): temp_df_to_save[col] = temp_df_to_save[col].astype('boolean')
                    except Exception as type_e: log_statement('warning', f"{LOG_INS}:WARNING>>Error applying type {target_dtype} to column {col} before save: {type_e}", Path(__file__).stem)
                elif col not in temp_df_to_save.columns: temp_df_to_save[col] = pd.NA
            tok_repo_saver.df = temp_df_to_save.reindex(columns=final_tokenized_schema_cols)
            tok_repo_saver.save()
            if not tokenized_repo_filename.exists(): raise IOError(f"Save failed: {tokenized_repo_filename}")
            log_statement('info', f"{LOG_INS}:INFO>>Tokenized repository saved successfully.", Path(__file__).stem)
            app_state['tokenized_repo_path'] = str(tokenized_repo_filename.resolve())
            app_state['tokenized_repo_df'] = final_tokenized_df
            log_statement('info', f"{LOG_INS}:INFO>>App state updated. Active tokenized repo: {tokenized_repo_filename}", Path(__file__).stem)
        except Exception as e:
             log_statement('error', f"{LOG_INS}:ERROR>>Failed to save tokenized repo {tokenized_repo_filename}: {e}", Path(__file__).stem, exc_info=True)
             print(f"Error saving tokenized data: {e}")
             app_state['tokenized_repo_path'] = None; app_state['tokenized_repo_df'] = None
    else:
        log_statement('warning', f"{LOG_INS}:WARNING>>Final tokenized repository is empty. Nothing to save. Clearing tokenized state.", Path(__file__).stem)
        app_state['tokenized_repo_path'] = None; app_state['tokenized_repo_df'] = None

    log_statement('info', f"{LOG_INS}:INFO>>Data tokenization finished.", Path(__file__).stem)

def train_on_tokens():
    """Handles Option 4: Train On Tokenized Files using project constants."""
    global app_state
    log_statement('info', f"{LOG_INS}:INFO>>Starting model training setup.", Path(__file__).stem)
    if app_state.get('tokenized_repo_df') is None or app_state.get('tokenized_repo_path') is None: log_statement('error', f"{LOG_INS}:ERROR>>Error: No tokenized data repository loaded/path set.", Path(__file__).stem, exc_info=True); return
    config = app_state['config']; tokenized_repo_df = app_state['tokenized_repo_df']; tokenized_repo_path = app_state['tokenized_repo_path']

    try:
        # Configs - use imported constants for paths
        train_config = config.get('training', {}); model_config = config.get('model', {}); data_loader_config = config.get('data_loader', {})
        default_model = model_config.get('name', 'bert-base-uncased'); default_lr = train_config.get('learning_rate', 5e-5)
        default_epochs = train_config.get('num_epochs', 3); default_batch_size = train_config.get('batch_size', 16)
        # Use imported constants for dirs
        checkpoint_dir = Path(train_config.get('checkpoint_dir', CHECKPOINT_DIR)); log_dir = Path(train_config.get('log_dir', LOG_DIR))
        num_labels = model_config.get('num_labels', 2)

        # Get Hyperparameters
        print(f"{LOG_INS} - --- Training Hyperparameters ---")
        model_name = input(f"Model name or path [{default_model}]: ") or default_model
        learning_rate = float(input(f"Learning Rate [{default_lr}]: ") or default_lr)
        epochs = int(input(f"Number of Epochs [{default_epochs}]: ") or default_epochs)
        batch_size = int(input(f"Batch Size [{default_batch_size}]: ") or default_batch_size)
        hyperparams = { 'model_name_or_path': model_name, 'learning_rate': learning_rate, 'num_epochs': epochs, 'batch_size': batch_size, 'num_labels': num_labels }
        log_statement('info', f"{LOG_INS}:INFO>>Training hyperparameters set: {hyperparams}", Path(__file__).stem)

        # Device and Logging Config - use imported LOG_DIR
        preferred_device = train_config.get('device', DEFAULT_DEVICE) # Use imported default
        device = set_compute_device(preferred_device)
        log_statement('info', f"{LOG_INS}:INFO>>Compute device set to: {device}", Path(__file__).stem)
        log_dir.mkdir(parents=True, exist_ok=True) # Use imported LOG_DIR
        config_log_filename = log_dir / f"training_config_{time.strftime('%Y%m%d%H%M%S')}.log"
        log_statement('info', f"{LOG_INS}:INFO>>Logging training configuration to {config_log_filename} and terminal.", Path(__file__).stem)
        config_details = f""" --- Training Configuration ---
        Timestamp: {time.asctime()}
        Tokenized Data Source: {tokenized_repo_path or 'N/A'} ({len(tokenized_repo_df)} files)
        Model: {model_name} Num Labels: {num_labels} Device: {device}
        Learning Rate: {learning_rate} Epochs: {epochs} Batch Size: {batch_size}
        Checkpoint Dir: {checkpoint_dir.resolve()} Log Dir: {log_dir.resolve()}
        --- End Configuration --- """
        print(f"{LOG_INS} - {config_details}")
        try:
            with open(config_log_filename, 'w') as f: f.write(config_details)
            log_statement('info', f"{LOG_INS}:INFO>>Training config saved to {config_log_filename}", Path(__file__).stem)
        except Exception as log_e: log_statement('error', f"{LOG_INS}:ERROR>>Failed to write training config log: {log_e}", Path(__file__).stem)

        # Prepare Data
        log_statement('info', f"{LOG_INS}:INFO>>Preparing dataset and dataloader...", Path(__file__).stem)
        token_filepaths = tokenized_repo_df['tokenized_filepath'].tolist()
        if 'TokenDataset' not in globals(): raise ImportError("TokenDataset class not available.")
        dataset = TokenDataset(token_files=token_filepaths) # Adjust if labels needed
        num_workers_loader = data_loader_config.get('num_workers', 0)
        persistent_workers = data_loader_config.get('persistent_workers', False) if num_workers_loader > 0 else False
        if 'create_dataloader' not in globals(): raise ImportError("create_dataloader function not available.")
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_loader, persistent_workers=persistent_workers)
        log_statement('info', f"{LOG_INS}:INFO>>Dataset ({len(dataset)} items) and Dataloader created (num_workers={num_workers_loader}).", Path(__file__).stem)

        # Initialize Model
        log_statement('info', f"{LOG_INS}:INFO>>Initializing model: {model_name}", Path(__file__).stem)
        try:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.to(device)
            log_statement('info', f"{LOG_INS}:INFO>>Model initialized successfully.", Path(__file__).stem)
        except Exception as e: log_statement('error', f"{LOG_INS}:ERROR>>Error initializing model: {e}", Path(__file__).stem, exc_info=True); return

        # Initialize Trainer - Ensure checkpoint_dir from config is used
        log_statement('info', f"{LOG_INS}:INFO>>Initializing Trainer...", Path(__file__).stem)
        try:
            if 'Trainer' not in globals(): raise ImportError("Trainer class not available.")
            # Pass checkpoint_dir explicitly if Trainer expects it
            trainer = Trainer(model=model, train_dataloader=dataloader, config=config, device=device, hyperparameters=hyperparams) # Assume Trainer gets checkpoint_dir from config
            log_statement('info', f"{LOG_INS}:INFO>>Trainer initialized.", Path(__file__).stem)
        except Exception as e: log_statement('error', f"{LOG_INS}:ERROR>>Error initializing Trainer: {e}", Path(__file__).stem, exc_info=True); return

        # Start Training
        log_statement('info', f"{LOG_INS}:INFO>>Starting training loop.", Path(__file__).stem)
        try:
            trainer.train()
            log_statement('info', f"{LOG_INS}:INFO>>Training loop finished.", Path(__file__).stem)
        except Exception as e: print(f"{LOG_INS} - Error during training: {e}"); return

        log_statement('info', f"{LOG_INS}:INFO>>Model training process finished successfully.", Path(__file__).stem)

    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error during Train On Tokens function: {e}", Path(__file__).stem, exc_info=True)

# --- Load Model Submenu Functions ---
def list_and_select_model_path():
    """Handles Option 5A: Scan for models using CHECKPOINT_DIR."""
    print(f"{LOG_INS} - \n--- Specify Model Folder Path ---")
    try:
        # Use imported CHECKPOINT_DIR as default
        model_folder_path_str = input(f"Enter model folder path (default: {CHECKPOINT_DIR}): ") or str(CHECKPOINT_DIR)
        model_folder_path = Path(model_folder_path_str)

        if not model_folder_path.is_dir(): print(f"{LOG_INS} - Error: Not a valid directory."); return None

        print(f"{LOG_INS} - Scanning '{model_folder_path}'...")
        log_statement('info', f"{LOG_INS}:INFO>>Scanning for models in: {model_folder_path}", Path(__file__).stem)
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
        if not found_models: print(f"{LOG_INS} - No models found."); return None
        log_statement('info', f"{LOG_INS}:INFO>>Found {len(found_models)} potential models.", Path(__file__).stem)
        display_map = {}; found_models.sort(key=lambda x: x['path'])
        for i, model_info in enumerate(found_models):
            designator = f"{i+1}{model_info['type_code']}"; display_map[designator] = model_info['path']
            print(f"{LOG_INS} - {designator} - {model_info['path']}")
        print(f"{LOG_INS} - --- Model Scan Complete ---")
        return display_map
    except Exception as e: print(f"{LOG_INS} - Error scanning models: {e}"); return None

def specify_model_for_loading(model_map):
    """Handles Option 5B: Specify File For Loading using project loader."""
    global app_state
    print(f"{LOG_INS} - \n--- Specify Model For Loading ---")
    if not model_map: print(f"{LOG_INS} - No models scanned."); return None
    while True:
        designator = input("Enter designator to load: ").strip().upper()
        if designator in model_map:
            selected_path_str = model_map[designator]; selected_path = Path(selected_path_str)
            print(f"{LOG_INS} - Loading model from: {selected_path}...")
            log_statement('info', f"{LOG_INS}:INFO>>Attempting to load model from: {selected_path}", Path(__file__).stem)
            model, tokenizer = None, None
            try:
                with tqdm(total=1, desc=f"Loading {designator}", unit="model") as pbar:
                    if 'load_model_from_checkpoint' in globals():
                        model, tokenizer = load_model_from_checkpoint(device=set_compute_device(), checkpoint_path=selected_path_str)
                        if model is None: raise ValueError(f"load_model_from_checkpoint failed for {selected_path_str}")
                        log_statement('info', f"{LOG_INS}:INFO>>Used project's load_model_from_checkpoint.", Path(__file__).stem)
                    else:
                        log_statement('warning', f"{LOG_INS}:WARNING>>Project's load_model_from_checkpoint not found. Using basic loading.", Path(__file__).stem)
                        if selected_path.is_dir() and (selected_path / 'config.json').exists():
                            model = AutoModel.from_pretrained(selected_path_str)
                            tokenizer = AutoTokenizer.from_pretrained(selected_path_str)
                        else: raise ValueError(f"Unsupported model type/structure for fallback loading: {selected_path}")
                    pbar.update(1)

                if app_state.get('config') is None: app_state['config'] = load_config()
                # Assume set_compute_device is available
                device = set_compute_device(app_state['config'].get('training',{}).get('device', str(DEFAULT_DEVICE)))
                if model: model.to(device)

                app_state['loaded_model'] = model; app_state['loaded_model_path'] = selected_path_str
                app_state['loaded_tokenizer'] = tokenizer
                log_statement('info', f"{LOG_INS}:INFO>>Model/Tokenizer loaded from {selected_path} to {device}.", Path(__file__).stem)
                return selected_path_str
            except Exception as e:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to load model {selected_path}: {e}", Path(__file__).stem, exc_info=True)
                app_state['loaded_model'] = None; app_state['loaded_model_path'] = None; app_state['loaded_tokenizer'] = None
                # Ask user if they want to try again or return? For now, just log and loop.
        else: print(f"{LOG_INS} - Invalid designator.")

def execute_functions_on_loaded_model(loaded_model_path):
    """Handles Option 5C: Execute Functions on Loaded Model."""
    global app_state
    print(f"{LOG_INS} - \n--- Execute Functions on Loaded Model ---")
    model = app_state.get('loaded_model'); tokenizer = app_state.get('loaded_tokenizer')
    if not model: print(f"{LOG_INS} - No model loaded."); return
    print(f"{LOG_INS} - Selected Model Path: {loaded_model_path}")
    try: device = next(model.parameters()).device
    except: device = 'cpu'
    print(f"{LOG_INS} - Model on Device: {device}")

    while True:
        print(f"{LOG_INS} - \n--- Model Operations Submenu ---"); print(f"{LOG_INS} - 1. Inference"); print(f"{LOG_INS} - 2. Evaluate (Not Implemented)"); print(f"{LOG_INS} - 3. Continue Training (Not Implemented)"); print(f"{LOG_INS} - 4. View Model Details"); print(f"{LOG_INS} - 5. Return to Load Model Menu")
        choice = input("Enter choice: ")
        log_statement('debug', f"{LOG_INS}:DEBUG>>Model Ops choice: {choice}", Path(__file__).stem)
        if choice == '1':
            print(f"{LOG_INS} - \n--- Inference ---")
            if not tokenizer: print(f"{LOG_INS} - Error: Tokenizer unavailable."); continue
            try:
                text = input("Enter text: ")
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                model.eval();
                with torch.no_grad(): outputs = model(**inputs)
                if hasattr(outputs, 'logits'): preds = torch.argmax(outputs.logits, dim=-1); print(f"{LOG_INS} - Prediction Index: {preds.cpu().item()}")
                else: print(f"{LOG_INS} - Outputs:\n{outputs}")
                log_statement('info', f"{LOG_INS}:INFO>>Inference successful.", Path(__file__).stem)
            except Exception as e: print(f"{LOG_INS} - Inference error: {e}")
        elif choice == '2': print(f"{LOG_INS} - \nEvaluate not implemented.")
        elif choice == '3': print(f"{LOG_INS} - \nContinue Training not implemented.")
        elif choice == '4':
            print(f"{LOG_INS} - \n--- Model Details ---"); print(f"{LOG_INS} - Path: {loaded_model_path}\nDevice: {device}"); print(f"{LOG_INS} - \nArchitecture:"); print(model)
            try: num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"{LOG_INS} - \nTrainable Params: {num_params:,}")
            except: pass
        elif choice == '5': print(f"{LOG_INS} - Returning..."); break
        else: print(f"{LOG_INS} - Invalid choice.")

def load_model_submenu():
    """Handles Option 5: Load/Manage Saved Model(s)."""
    global app_state
    log_statement('info', f"{LOG_INS}:INFO>>Entered Load Model submenu.", Path(__file__).stem)
    model_map = None
    while True:
        print(f"{LOG_INS} - \n--- Load Model Submenu ---")
        print(f"{LOG_INS} - A) Scan folder for models"); print(f"{LOG_INS} - B) Specify Model From Scan")
        status = f"(Loaded: {Path(app_state['loaded_model_path']).name})" if app_state.get('loaded_model_path') else "(No model loaded)"
        print(f"{LOG_INS} - C) Execute Functions on Loaded Model {status}"); print(f"{LOG_INS} - D) Return to Data Processing Menu")
        choice = input("Enter choice (A/B/C/D): ").upper()
        log_statement('debug', f"{LOG_INS}:DEBUG>>Load Model choice: {choice}", Path(__file__).stem)
        if choice == 'A': model_map = list_and_select_model_path()
        elif choice == 'B':
            if model_map: specify_model_for_loading(model_map)
            else: print(f"{LOG_INS} - Run Option A first.")
        elif choice == 'C':
            if app_state.get('loaded_model'): execute_functions_on_loaded_model(app_state['loaded_model_path'])
            else: print(f"{LOG_INS} - Load model via Option B first.")
        elif choice == 'D': print(f"{LOG_INS} - Returning..."); break
        else: print(f"{LOG_INS} - Invalid choice.")

    while True:
        status_main = f"(Set: {Path(app_state['main_repo_path']).name})" if app_state.get('main_repo_path') else "(No repo set)"
        status_proc = f"(Set: {Path(app_state['processed_repo_path']).name})" if app_state.get('processed_repo_path') else "(Not run)"
        status_tok = f"(Set: {Path(app_state['tokenized_repo_path']).name})" if app_state.get('tokenized_repo_path') else "(Not run)"
        status_model = f"(Loaded: {Path(app_state['loaded_model_path']).name})" if app_state.get('loaded_model_path') else "(None)"

        print(f"{LOG_INS}"
              "\n--- Data Processing & Tokenization Menu ---")
        print(f"1. Set Data Directory {status_main}"
              f"2. Process Linguistic Data {status_proc}"
              f"3. Data Tokenization {status_tok}"
              f"4. Train On Tokenized Files"
              f"5. Load/Manage Saved Model(s) {status_model}"
              f"6. Exit to Main Menu")
        print("-" * 40)
        choice = input("Enter choice (1-6): ")
        log_statement('debug', f"{LOG_INS}:DEBUG>>Data Processing choice: {choice}", Path(__file__).stem)

        if choice == '1': set_data_directory()
        elif choice == '2': process_linguistic_data()
        elif choice == '3': tokenize_data()
        elif choice == '4': train_on_tokens()
        elif choice == '5': load_model_submenu()
        elif choice == '6': print(f"{LOG_INS} - Returning..."); break
        else: print(f"{LOG_INS} - Invalid choice.")
        input("\nPress Enter to continue...") # Pause for user

# --- Main Submenu Function ---
def data_processing_submenu():
    """Displays the main Data Processing and Tokenization submenu."""
    global app_state
    if not essential_imports_available:
        log_statement("ERROR: Core modules not loaded. Cannot proceed with data processing.", "error")
        input("Press Enter to return...")
        return
    log_statement('info', f"{LOG_INS}:INFO>>Entered Data Processing submenu.", Path(__file__).stem)
    print(f"{LOG_INS}" 
          "----- CURRENT app_state['config'] CONTENTS"
          f"{app_state.get('config')}"
          " ----- END OUTPUT ----- "
          )
    if app_state.get('config') is None:
        try:
            app_state['config'] = load_config()
            log_statement('info',
                          f"{LOG_INS}"
                          ">>>> Config loaded. <<<<",
                          Path(__file__).stem)
        except Exception as e:
            app_state['config'] = {}
            log_statement('error',
                          f"{LOG_INS}"
                          f" >>>> Failed to load config: {e} <<<< ",
                          Path(__file__).stem)
    if app_state.get('config') is None: # Load config once
        try:
            app_state['config'] = load_config()
            log_statement('info',
                          f"{LOG_INS}"
                          " >>>> Config loaded. <<<< ",
                          Path(__file__).stem)
        except Exception as e:
            app_state['config'] = {} # Fallback
            log_statement('error',
                          f"{LOG_INS}"
                          " >>>> EMPTY CONFIG FILE BELOW LOADED <<<< "
                          f"{app_state['config']}",
                          Path(__file__).stem)

    while True:
        print_welcome_message()
        print("\n--- Data Processing & Model Training ---")
        repo_status = app_state.get('repository_path', 'Not Set')
        repo_instance = app_state.get('repository', None)
        repo_info = "Loaded" if repo_instance else "Not Loaded"
        print(f"Current Repository: {repo_status} ({repo_info})")
        print("--------------------------------------")
        print("1. Set/Scan Data Directory & Load Repository")
        print("2. Process Linguistic Data (Raw Files -> Processed)")
        print("3. Tokenize Processed Data (Processed -> Tokens)")
        print("4. Train Model on Tokenized Data")
        print("5. Load Saved Model / View Checkpoints")
        print("0. Back to Main Menu")
        print("--------------------------------------")

        choice = input("Enter your choice: ")

        try:
            if choice == '1':
                set_data_directory()
            elif choice == '2':
                process_linguistic_data()
            elif choice == '3':
                tokenize_data()
            elif choice == '4':
                train_on_tokens()
            elif choice == '5':
                load_model_submenu()
            elif choice == '0':
                log_statement("Returning to main menu.", "info")
                break
            else:
                print("Invalid choice. Please try again.")
                time.sleep(1)
        except Exception as e:
            log_statement('error', f"ERROR in data processing submenu option '{choice}': {e}", __file__)
            log_statement('error', f"Traceback: {traceback.format_exc()}", __file__)
            input("An unexpected error occurred. Press Enter to continue...")