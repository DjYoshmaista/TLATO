import csv
import hashlib
import collections
import io
import inspect
import json
import io
import zstandard as zstd
import collections
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import psutil
import threading
PSUTIL_AVAILABLE = True if psutil in sys.modules else False
from src.utils.helpers import _get_max_workers, _generate_file_paths, _get_file_metadata
from src.utils.logger import configure_logging, log_statement
from src.utils.hashing import *
configure_logging()

try:
    from src.utils.config import load_config
    load_config()
    try:
        import cudf
        import cupy
        GPU_AVAILABLE = True
    except ImportError:
        cudf = None
        GPU_AVAILABLE = False
except Exception as e:
    log_statement(loglevel='exception', logstatement=f"Error importing cudf and cupy {e}.", main_logger=__file__)
                  
# Assuming these constants and helper functions are defined elsewhere
try:
    from src.data.constants import *
    CONSTANTS_AVAILABLE = True
except ImportError:
    log_statement('exception', f"{LOG_INS}:ImportError>>Importing of constants from 'src.data.constants' in project folder failed!", __file__)
    CONSTANTS_AVAILABLE = False
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    log_statement('exception', f"{LOG_INS}:ImportError>>Import of zStandard as zstd failed!", __file__)
    ZSTD_AVAILBLE = False
from src.utils.helpers import *

COL_PATH_HASH = None


# --- Helper Function for Repo Filename ---
def get_repo_filename(repo_hash: str) -> Path:
    """Constructs the standard repository filename based on its hash."""
    return DATA_REPO_DIR / f"data_repository_{repo_hash}.csv.zst"

class DataRepository:        
    def __init__(self, directory_path: Optional[str | Path] = None, repo_path: Optional[Path] = None, target_dir: Optional[Path] = None, use_gpu: bool = False):
        """
        Initializes the DataRepository.

        Args:
            repo_path (Optional[Path]): Path to an existing repository file.
            target_dir (Optional[Path]): Path to the directory this repository manages.
                                         Required if creating a new repository.
            use_gpu (bool): Flag to attempt using cuDF if available.
        """
        if not CONSTANTS_AVAILABLE:
            raise ImportError("Essential constants could not be imported. Cannot initialize DataRepository.")
        if not ZSTD_AVAILABLE:
            log_statement(loglevel='warning', logstatement="zstandard library not found. Repository compression/decompression disabled.", main_logger=__file__)
            # Potentially raise error if zstd is mandatory
        print(f"{LOG_INS} >>>  DataRepository.__init__ start. directory_path={directory_path}, repo_path={repo_path}, target_dir={target_dir}, use_gpu={use_gpu}.")

        init_logger_name = __file__
        self.lock = threading.RLock()
        self.target_dir = target_dir.resolve() if target_dir else None
        self.repo_hash = hash_filepath(str(self.target_dir)) if self.target_dir else None
        # --- Define internal schema USING CONSTANTS ---
        self.columns_schema = COL_SCHEMA
        self.timestamp_columns = TIMESTAMP_COLUMNS # Use timestamp list from constants

        # Determine repository path
        if repo_path:
            self.repo_path = repo_path.resolve()
        elif self.repo_hash:
            self.repo_path = get_repo_filename(self.repo_hash) # Use helper
        else:
            # Cannot determine repo path if neither repo_path nor target_dir is given
             log_statement(loglevel='error', logstatement="Must provide either repo_path or target_dir to initialize DataRepository.", main_logger=__file__)
             raise ValueError("Must provide either repo_path or target_dir.")

        # Setup GPU usage
        self.use_gpu = use_gpu and GPU_AVAILABLE and cudf is not None
        log_statement(loglevel='info', logstatement=f"DataRepository initialized. GPU Mode: {self.use_gpu}. Repo Path: {self.repo_path}", main_logger=__file__)

        # Load or initialize repository index (consider doing this only when needed)
        # self.repository_index = self._load_repository_index() # Maybe lazy load?

        # Log warning if repo file didn't exist but was expected
        if repo_path and not repo_path.exists() and self.df.empty:
             log_statement(loglevel='warning', logstatement=f"Provided repo_path '{repo_path}' does not exist. Initialized empty repository.", main_logger=__file__)

        # --- Input Validation ---
        if not (directory_path is None) ^ (repo_path is None):
            print(f"{LOG_INS} >>>  DataRepository.__init__ ERROR - Invalid args.")
            raise ValueError("Initialize with either directory_path OR repo_path.")


        print(f"{LOG_INS} >>>  DataRepository Schema Columns: {list(self.columns_schema.keys())}")
        # Define timestamp columns based on the schema's datetime types
        print(f"{LOG_INS} >>>  DataRepository Timestamp Columns: {self.timestamp_columns}")

        # --- Initialize Path Attributes ---
        self.directory_path: Optional[Path] = None
        self.repo_dir: Optional[Path] = None
        self.repo_path: Optional[Path] = None
        self.sub_repo_index_path: Optional[Path] = None # <<< Define path attribute
        self.repo_index_dirty = False

        # --- Determine Paths based on Input ---
        try:
            if directory_path is not None:
                # Mode 1: Initialized with directory_path
                print(f"{LOG_INS} >>>  DataRepository Mode 1 Init (Directory: {directory_path})")
                dp_resolved = Path(directory_path).resolve()
                self.directory_path = dp_resolved
                self.repo_dir = self.directory_path / ".repository"
                self.repo_path = self.repo_dir / "file_index.csv.zst" # Default name
                if not self.repo_dir.exists():
                    self.repo_dir.mkdir(parents=True, exist_ok=True)
                # <<< FIX: Set sub_repo_index_path reliably >>>
                self.sub_repo_index_path = self.repo_dir / "sub_repo_index.json"
                print(f"{LOG_INS} >>>  DataRepository Mode 1 Paths: Dir={self.directory_path}, RepoDir={self.repo_dir}, RepoPath={self.repo_path}, SubIndexPath={self.sub_repo_index_path}")
            else: # repo_path is not None
                # Mode 2: Initialized with repo_path
                print(f"{LOG_INS} >>>  DataRepository Mode 2 Init (Repo Path: {repo_path})")
                rp_resolved = Path(repo_path).resolve()
                self.repo_path = rp_resolved
                self.repo_dir = self.repo_path.parent
                # <<< FIX: Set sub_repo_index_path reliably >>>
                self.sub_repo_index_path = self.repo_dir / "sub_repo_index.json"
                self.directory_path = None # Associated directory is unknown in this mode
                if not self.repo_path.exists():
                     print(f"{LOG_INS} >>>  DataRepository WARNING - Repo path does not exist: {self.repo_path}")
                     log_statement(loglevel="warning", logstatement=f"{LOG_INS} >>> Provided repo_path does not exist: {self.repo_path}. Load will initialize empty.", main_logger=init_logger_name)
                elif not self.repo_path.is_file():
                     print(f"{LOG_INS} >>>  DataRepository WARNING - Repo path is not a file: {self.repo_path}")
                     log_statement(loglevel="warning", logstatement=f"{LOG_INS} >>> Provided repo_path exists but is not a file: {self.repo_path}", main_logger=init_logger_name)
                print(f"{LOG_INS} >>>  DataRepository Mode 2 Paths: RepoPath={self.repo_path}, RepoDir={self.repo_dir}, SubIndexPath={self.sub_repo_index_path}, Dir=None")

        except TypeError as te:
             print(f"{LOG_INS} >>>  DataRepository CRITICAL - Invalid path type: {directory_path or repo_path}. Error: {te}")
             log_statement(loglevel='critical', logstatement=f"{LOG_INS} >>> Invalid path type provided: {directory_path or repo_path} ({type(directory_path or repo_path)}). Error: {te}", main_logger=init_logger_name)
             raise ValueError(f"Invalid path argument provided.") from te
        except Exception as e:
            print(f"{LOG_INS} >>>  DataRepository CRITICAL - Path setup failed: {e}")
            log_statement(loglevel='critical', logstatement=f"{LOG_INS} >>> Initialization path setup failed: {e}", main_logger=init_logger_name, exc_info=True)
            raise

        # --- Load Data ---
        print(f"{LOG_INS} >>>  DataRepository Calling _load_repo for: {self.repo_path}")
        self.df = self._load_repo() # Load data using self.repo_path
        print(f"{LOG_INS} >>>  DataRepository _load_repo returned DF length: {len(self.df) if self.df is not None else 'None'}")
        print(f"{LOG_INS} >>>  DataRepository.__init__ end. Repo: {self.repo_path}, Dir: {self.directory_path or 'N/A'}")
        log_statement(loglevel="info", logstatement=f"{LOG_INS} >>> DataRepository initialized. Repo file: {self.repo_path}. Associated Dir: {self.directory_path or 'N/A'}", main_logger=init_logger_name)

    def _scan_directory(self, folder_path_obj: Path, existing_files_set: set) -> List[Dict[str, Any]]:
        """
        Scans directory recursively, gets metadata using _get_file_metadata (which returns ISO timestamps),
        filters against known paths, and handles processing in parallel with progress bars.

        Args:
            folder_path_obj (Path): The directory path object to scan.
            existing_files_set (set): A set of resolved file path strings already known.

        Returns:
            List[Dict[str, Any]]: A list of metadata dictionaries for new/changed files found.
        """
        LOG_INS = f'{__name__}::_scan_directory::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        scan_logger_name = __name__

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot scan directory.", main_logger=scan_logger_name)
             return []

        # Attempt to get max_workers from config - needs robust fetching
        max_workers = _get_max_workers() # Simplified call, assumes helper handles config access
        new_files_metadata = []

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Starting recursive scan: '{folder_path_obj.name}'. Max Workers: {max_workers}. Excluding: {len(existing_files_set)} known paths.", main_logger=scan_logger_name)

        # --- Stage 1: Collect all potential file paths ---
        potential_paths = []
        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Stage 1: Discovering items in '{folder_path_obj}'...", main_logger=scan_logger_name)
        # Use tqdm for discovery progress
        with tqdm(desc=f"Discovering [{folder_path_obj.name}]", unit=" items", smoothing=0.1, leave=False) as pbar_discover:
            try:
                for filepath in _generate_file_paths(folder_path_obj): # Yields Path objects
                    potential_paths.append(filepath)
                    pbar_discover.update(1)
            except Exception as gen_e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error during path generation in '{folder_path_obj}': {gen_e}", main_logger=scan_logger_name, exc_info=True)

        collected_count = len(potential_paths)
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Stage 1 complete. Found {collected_count} total items in '{folder_path_obj}'.", main_logger=scan_logger_name)

        if not potential_paths:
            return []

        # --- Stage 2: Parallel Path Resolution and Filtering ---
        files_to_scan = [] # List for Path objects that need metadata processing
        skipped_known_count = 0
        skipped_unsupported_count = 0
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Stage 2: Filtering {collected_count} paths (Parallel)...", main_logger=scan_logger_name)

        # Helper function for parallel path checking
        def check_path_for_scan(filepath: Path, known_paths: set) -> Optional[Path]:
            """Resolves path, checks if known, checks if supported extension."""
            try:
                resolved_path_str = str(filepath.resolve())
                if resolved_path_str in known_paths:
                    return None # Skip known

                # Check extension using SUPPORTED_EXTENSIONS constant
                file_ext = filepath.suffix.lower().lstrip('.')
                if file_ext not in SUPPORTED_EXTENSIONS:
                    # Log infrequent warnings for unsupported skips
                    # if random.random() < 0.01: # Log ~1% of skipped unsupported files
                    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS} >>> Skipping unsupported type '.{file_ext}': {filepath.name}", main_logger=scan_logger_name)
                    return "UNSUPPORTED" # Special marker for unsupported type

                # Check if it's the repo file itself or temp file
                # Assume get_repo_filename requires hash - might need adjustment
                # Or check against self.repo_path directly if available
                # if self.repo_path and filepath.resolve() == self.repo_path: return None
                if filepath.name == INDEX_FILE.name: return None # Skip index file
                if filepath.name.startswith("data_repository_") and filepath.name.endswith(".csv.zst"): return None # Skip repo files by pattern
                if filepath.name.endswith('.tmp') or '.tmp_' in filepath.name: return None # Skip temp files


                return filepath # Path is new and supported

            except OSError as e: # Catch specific OS errors like file not found during resolve
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} >>> OS Error resolving/checking path '{filepath.name}': {e}. Skipping.", main_logger=scan_logger_name)
                 return None
            except Exception as e:
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} >>> Unexpected Error resolving/checking path '{filepath.name}': {e}. Skipping.", main_logger=scan_logger_name)
                 return None

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='FilterPath') as executor:
            futures = {executor.submit(check_path_for_scan, fpath, existing_files_set): fpath for fpath in potential_paths}

            pbar_filter = tqdm(
                as_completed(futures), total=len(potential_paths),
                desc=f"Filtering Paths [{folder_path_obj.name}]", unit="path", leave=False
            )
            for future in pbar_filter:
                original_path = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, Path):
                        files_to_scan.append(result)
                    elif result == "UNSUPPORTED":
                        skipped_unsupported_count += 1
                    else: # result is None (known path or error)
                        skipped_known_count += 1
                    # Update postfix less frequently
                    if pbar_filter.n % 100 == 0 or pbar_filter.n == pbar_filter.total:
                        pbar_filter.set_postfix_str(f"Scan:{len(files_to_scan)}, Skip(Known):{skipped_known_count}, Skip(Type):{skipped_unsupported_count}", refresh=True)
                except Exception as e:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} >>> Error processing path filter future for ~'{original_path.name}': {e}", main_logger=scan_logger_name)
                    skipped_known_count += 1 # Count as skipped

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Stage 2 complete. Found {len(files_to_scan)} new/supported files. Skipped(Known): {skipped_known_count}, Skipped(Unsupported): {skipped_unsupported_count}.", main_logger=scan_logger_name)

        if not files_to_scan:
            return []

        # --- Stage 3: Parallel Metadata Gathering ---
        # (This part remains largely the same as the input version, as _get_file_metadata handles the timestamp standardization)
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Stage 3: Gathering metadata for {len(files_to_scan)} files (Parallel)...", main_logger=scan_logger_name)
        processed_count_meta = 0
        start_time_meta = time.time()

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Metadata') as executor:
            future_to_path = {executor.submit(_get_file_metadata, filepath): filepath for filepath in files_to_scan}

            pbar_meta = tqdm(
                as_completed(future_to_path), total=len(files_to_scan),
                desc=f"Gather Metadata [{folder_path_obj.name[:20]}]", unit="file", leave=True,
                postfix={"files/s": "0.0", "CPU": "N/A", "Mem": "N/A"}
            )
            for future in pbar_meta:
                filepath = future_to_path[future]
                processed_count_meta += 1
                try:
                    metadata = future.result() # Metadata dict with ISO strings for timestamps
                    if metadata:
                        new_files_metadata.append(metadata)
                    # else: Errors logged within _get_file_metadata
                except Exception as exc:
                    log_statement(loglevel='error', logstatement=f'{LOG_INS} - Metadata task generated exception for {filepath.name}: {exc}', main_logger=scan_logger_name, exc_info=True)

                # --- Update Metrics Postfix (Remains the same) ---
                if processed_count_meta % 50 == 0 or processed_count_meta == len(files_to_scan):
                    elapsed_time = time.time() - start_time_meta
                    files_per_sec = processed_count_meta / elapsed_time if elapsed_time > 0 else 0
                    metrics_postfix = {"files/s": f"{files_per_sec:.1f}"}
                    if PSUTIL_AVAILABLE:
                        try:
                            metrics_postfix["CPU"] = f"{psutil.cpu_percent(interval=None):.1f}%"
                            metrics_postfix["Mem"] = f"{psutil.virtual_memory().percent:.1f}%"
                        except Exception: metrics_postfix["CPU"], metrics_postfix["Mem"] = "ERR", "ERR" # Ignore psutil errors for tqdm
                    pbar_meta.set_postfix(metrics_postfix, refresh=False)

        # Log final metrics
        final_elapsed_time_meta = time.time() - start_time_meta
        final_rate_meta = processed_count_meta / final_elapsed_time_meta if final_elapsed_time_meta > 0 else 0
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Stage 3 finished. Processed {processed_count_meta} files for metadata in {final_elapsed_time_meta:.2f}s ({final_rate_meta:.2f} files/s).", main_logger=scan_logger_name)
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Directory scan complete for '{folder_path_obj}'. Gathered metadata for {len(new_files_metadata)} new/supported files.", main_logger=scan_logger_name)

        return new_files_metadata

    def _get_repo_hash(self, path_obj: Path) -> str:
        """Generates a consistent hash for a directory path using the central hashing utility."""
        LOG_INS = f'{__name__}::_get_repo_hash::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        repo_hash_logger = __name__
        if not CONSTANTS_AVAILABLE:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants (and thus hashing utils) not available.", main_logger=repo_hash_logger)
            return "error_hash" # Return placeholder error hash

        try:
            normalized_path_str = str(path_obj.resolve())
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Generating hash for normalized path: {normalized_path_str}", main_logger=repo_hash_logger)
            # Use central hash_filepath function
            return hash_filepath(normalized_path_str)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error resolving path or generating hash for '{path_obj}': {e}", main_logger=repo_hash_logger, exc_info=True)
            return "error_hash" # Return placeholder error hash


    # --- MODIFIED _get_repository_info Method ---
    def _get_repository_info(self, folder_path_obj: Path) -> Tuple[str, Path]:
        """Generates hash and filename for the main data repository using standardized methods."""
        LOG_INS = f'{__name__}::_get_repository_info::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        repo_info_logger = __name__

        repo_hash = self._get_repo_hash(folder_path_obj) # Uses standardized hash function
        repo_filename = get_repo_filename(repo_hash)   # Uses standardized filename helper

        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Generated repository info for '{folder_path_obj.resolve()}': hash={repo_hash}, filename={repo_filename}", main_logger=repo_info_logger)
        return repo_hash, repo_filename

    def save(self, df_to_save: Optional[pd.DataFrame] = None, output_path: Optional[str | Path] = None): # Keep original signature for now
        """
        Saves the internal DataFrame using constants for columns/header and robust type handling.
        Includes fix for handling potential duplicate columns.
        """
        save_logger_name = __file__
        LOG_INS = f'{save_logger_name}::save::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}' # Use method name 'save'

        # --- Use internal state if args are None ---
        # Note: This overrides passed arguments, potentially confusing. Consider refactoring
        # to either always use internal state (_save_repo pattern) or always use args.
        # Sticking to the original method's behavior for now.
        output_path = self.repo_path if output_path is None else Path(output_path) # Use self.repo_path if output_path not provided
        df_to_save = self.df if df_to_save is None else df_to_save          # Use self.df if df_to_save not provided

        # --- Pre-checks ---
        if df_to_save is None:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - DataFrame is None. Cannot save.", main_logger=save_logger_name)
            return
        if output_path is None:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - Output path is None. Cannot save.", main_logger=save_logger_name)
            return
        if not CONSTANTS_AVAILABLE: # Assumes flag set in __init__
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot reliably save repository.", main_logger=save_logger_name)
            # Decide if we should attempt saving anyway or return
            return
        if not ZSTD_AVAILABLE: # Assumes flag set in __init__
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - zstandard library not found. Cannot save compressed repository.", main_logger=save_logger_name)
            return

        # --- Handle Empty DataFrame ---
        if df_to_save.empty:
            log_statement(loglevel="warning", logstatement=f"{LOG_INS} - Attempting to save an empty DataFrame to {output_path}.", main_logger=save_logger_name)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL) # Use constant
                with open(output_path, 'wb') as f_out:
                    with cctx.stream_writer(f_out) as writer:
                        # Write header using MAIN_REPO_HEADER constant
                        header_line = ",".join(MAIN_REPO_HEADER) + "\n"
                        writer.write(header_line.encode('utf-8'))
                log_statement(loglevel="info", logstatement=f"{LOG_INS} - Empty repository file with header saved to {output_path}.", main_logger=save_logger_name)
                return
            except Exception as empty_save_e:
                log_statement(loglevel="error", logstatement=f"{LOG_INS} - Failed to save empty repository file to {output_path}: {empty_save_e}", main_logger=save_logger_name, exc_info=True)
                return

        # --- Proceed with saving non-empty DataFrame ---
        # Acquire lock if modifying self.df, otherwise use local df_copy
        with self.lock: # Assume lock is needed if df_to_save could be self.df
            df_copy = df_to_save.copy() # Work on a copy

            temp_path = output_path.with_suffix(f'{output_path.suffix}.tmp_{int(time.time())}')
            log_statement(loglevel="info", logstatement=f"{LOG_INS} - Saving DataFrame ({len(df_copy)} entries) to: {output_path} via {temp_path}", main_logger=save_logger_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use the canonical header defined in constants
            expected_columns = MAIN_REPO_HEADER
            print(f"{LOG_INS} >>>  Save Repo - Expected Header for Save: {expected_columns}")

            # --- Ensure all expected columns exist ---
            current_cols = df_copy.columns.tolist()
            added_cols_save = []
            for col_const in expected_columns:
                if col_const not in current_cols:
                    added_cols_save.append(col_const)
                    target_dtype_str = self.columns_schema.get(col_const, 'string') # Use schema from self
                    print(f"{LOG_INS} >>>  Save Repo - Adding missing column '{col_const}' for save.")
                    # Add with appropriate null type based on schema string
                    if target_dtype_str == 'datetime64[ns, UTC]': df_copy[col_const] = pd.NaT
                    elif target_dtype_str == 'Int64': df_copy[col_const] = pd.NA
                    elif target_dtype_str == 'boolean': df_copy[col_const] = pd.NA
                    elif target_dtype_str == 'Float64': df_copy[col_const] = pd.NA
                    elif target_dtype_str == 'string': df_copy[col_const] = pd.NA
                    else: df_copy[col_const] = pd.NA # Default to NA

            if added_cols_save:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Columns added during save: {added_cols_save}", main_logger=save_logger_name)

            # --- Detect and Remove Duplicate Columns BEFORE Reindex ---
            cols_index = df_copy.columns
            if not cols_index.is_unique:
                duplicate_column_names = cols_index[cols_index.duplicated(keep=False)].unique().tolist()
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Duplicate columns found before reindex: {duplicate_column_names}. Keeping first instance.", main_logger=save_logger_name)
                # Use pandas Index method to get unique columns directly
                # This is generally more robust than boolean indexing for this specific task
                df_copy = df_copy.loc[:, ~cols_index.duplicated(keep='first')]
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Columns after duplicate removal attempt: {df_copy.columns.tolist()}", main_logger=save_logger_name)

            # --- Add Pre-Reindex Checks ---
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Performing pre-reindex checks...", main_logger=save_logger_name)
            # 1. Check uniqueness of the target header list
            if len(set(expected_columns)) != len(expected_columns):
                dup_headers = [item for item, count in collections.Counter(expected_columns).items() if count > 1]
                log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL ERROR - Duplicate columns detected in MAIN_REPO_HEADER constant: {dup_headers}! Cannot proceed.", main_logger=save_logger_name)
                raise ValueError(f"Duplicate columns detected in MAIN_REPO_HEADER constant: {dup_headers}")
            else:
                 log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Target header `expected_columns` is unique.", main_logger=save_logger_name)

            # 2. Check uniqueness of the DataFrame columns *after* duplicate removal attempt
            if not df_copy.columns.is_unique:
                final_duplicates = df_copy.columns[df_copy.columns.duplicated(keep=False)].unique().tolist()
                log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL ERROR - DataFrame columns are STILL NOT UNIQUE before reindex: {final_duplicates}! Duplicate removal failed.", main_logger=save_logger_name)
                # Optionally try one last ditch effort or raise
                # raise ValueError(f"Failed to remove duplicate columns before reindex: {final_duplicates}")
                # Last ditch: forcefully take unique columns
                df_copy = df_copy.loc[:, ~df_copy.columns.duplicated(keep='first')]
                if not df_copy.columns.is_unique: # Check again
                     raise ValueError(f"Failed to remove duplicate columns even with fallback: {final_duplicates}")
            else:
                 log_statement(loglevel='debug', logstatement=f"{LOG_INS} - DataFrame columns `df_copy` are unique before reindex.", main_logger=save_logger_name)

            # --- Reindex to ensure correct order and all columns from MAIN_REPO_HEADER ---
            try:
                # Check for duplicates in header constant itself before reindexing
                if len(set(expected_columns)) != len(expected_columns):
                    duplicates = [item for item, count in collections.Counter(expected_columns).items() if count > 1]
                    raise ValueError(f"Duplicate columns detected in MAIN_REPO_HEADER constant: {duplicates}")
                df_copy = df_copy.reindex(columns=expected_columns)
            except KeyError as ke:
                # This should ideally not happen if missing columns were added above
                log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL ERROR - Column mismatch during save reindex. Missing: {ke}. Available: {df_copy.columns.tolist()}", main_logger=save_logger_name)
                raise ValueError(f"DataFrame columns could not be aligned with MAIN_REPO_HEADER during save: {ke}") from ke
            except ValueError as ve: # Catch duplicate header error
                log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL ERROR - {ve}", main_logger=save_logger_name)
                raise ve

            # --- Prepare Data Types for CSV Saving ---
            print(f"{LOG_INS} >>> - Save Repo - Preparing data types for CSV saving...")
            for col_const in expected_columns: # Iterate through the final ordered columns
                if col_const not in df_copy.columns: continue # Safeguard check
                try:
                    # df_copy[col_const] should now be a Series
                    series = df_copy[col_const]
                    target_dtype_info = self.columns_schema.get(col_const)

                    if col_const in self.timestamp_columns: # Use constant list
                        # Convert internal datetime64[ns, UTC] to ISO string for saving
                        if pd.api.types.is_datetime64_any_dtype(series.dtype):
                            if series.dt.tz is None: series = series.dt.tz_localize('UTC')
                            elif str(series.dt.tz).upper() != 'UTC': series = series.dt.tz_convert('UTC')
                            df_copy[col_const] = series.apply(lambda dt: dt.isoformat(timespec='microseconds') if pd.notna(dt) else '')
                            # print(f"{LOG_INS} >>> - Save Repo - Converted '{col_const}' (Timestamp) to ISO string") # Optional Debug
                        else:
                            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Timestamp column '{col_const}' had unexpected dtype {series.dtype} during save. Converting to string.", main_logger=save_logger_name)
                            df_copy[col_const] = series.astype(str).fillna('')
                    elif target_dtype_info == 'Int64' or target_dtype_info == 'boolean' or target_dtype_info == 'Float64':
                        # Convert nullable types (Int, Bool, Float) to string, representing NA as empty string
                        df_copy[col_const] = series.apply(lambda x: '' if pd.isna(x) else str(x))
                        # print(f"{LOG_INS} >>> - Save Repo - Converted '{col_const}' (Nullable) to string") # Optional Debug
                    else: # Default to string for all OTHER types (including 'string' schema type)
                         df_copy[col_const] = series.fillna('').astype(str)
                         # print(f"{LOG_INS} >>> - Save Repo - Converting '{col_const}' (Other) to string") # Optional Debug

                except Exception as conv_e:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error preparing column '{col_const}' for CSV saving: {conv_e}. Saving as raw string.", main_logger=save_logger_name, exc_info=True)
                    # Fallback: ensure the column in df_copy is string
                    try:
                        df_copy[col_const] = df_copy[col_const].astype(str).fillna('')
                    except Exception as fallback_assign_e:
                         log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL ERROR assigning fallback string to column '{col_const}': {fallback_assign_e}", main_logger=save_logger_name)

            # --- Write Compressed CSV in Chunks ---
            chunk_size = 10000
            total_rows = len(df_copy)
            pbar = tqdm(total=total_rows, desc=f"Saving {output_path.name}", unit="row", leave=False)
            print(f"{LOG_INS} >>> Save Repo - Starting write loop (Chunks: {chunk_size})...")
            try:
                cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL) # Use constant
                with open(temp_path, 'wb') as f_out:
                    with cctx.stream_writer(f_out) as writer:
                        # Use expected_columns (MAIN_REPO_HEADER) for the header line
                        header_line = ",".join(expected_columns) + "\n"
                        writer.write(header_line.encode('utf-8'))
                        pbar.set_postfix_str("Writing header...")

                        # Iterate through chunks and write to CSV buffer
                        for i in range(0, total_rows, chunk_size):
                            chunk = df_copy.iloc[i:min(i + chunk_size, total_rows)]
                            csv_buffer = io.StringIO()
                            # Ensure writing uses the expected order and quoting
                            chunk.to_csv(csv_buffer, index=False, header=False, columns=expected_columns, quoting=csv.QUOTE_MINIMAL, escapechar='\\', encoding='utf-8')
                            writer.write(csv_buffer.getvalue().encode('utf-8'))
                            csv_buffer.close()
                            pbar.update(len(chunk))
                            pbar.set_postfix_str(f"Row {pbar.n}/{total_rows}")

                        pbar.set_postfix_str(f"Completed {total_rows} rows.")
                print(f"{LOG_INS} >>> Save Repo - Write to temp file {temp_path} complete.")

                # --- Atomic Replace ---
                print(f"{LOG_INS} >>> Save Repo - Moving temp file {temp_path} to {output_path}")
                shutil.move(str(temp_path), str(output_path)) # Move temp file to final destination
                log_statement(loglevel="info", logstatement=f"{LOG_INS} - Successfully saved DataFrame ({total_rows} entries) to {output_path}", main_logger=save_logger_name)
                print(f"{LOG_INS} >>> DataRepository.save END - Success.")

            except Exception as e:
                log_statement(loglevel="critical", logstatement=f"{LOG_INS} - CRITICAL ERROR during save process for {output_path}: {e}", main_logger=save_logger_name, exc_info=True)
                # Attempt to clean up temp file on error
                if temp_path.exists():
                    try: os.remove(temp_path)
                    except OSError: log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Failed to remove temporary save file {temp_path} after error.", main_logger=save_logger_name)
                raise # Re-raise exception after logging and cleanup attempt
            finally:
                pbar.close() # Ensure progress bar is closed
            
    def _save_repo(self):
        """
        Saves the current internal DataFrame (self.df) to its designated repository file (self.repo_path).
        Uses constants for columns/header, robust type handling, and Zstandard compression.
        Timestamps are saved as ISO 8601 UTC strings.
        """
        LOG_INS = f'{__name__}::_save_repo::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        save_logger_name = __name__

        # --- Pre-checks ---
        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot save repository.", main_logger=save_logger_name)
             return
        if not ZSTD_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - zstandard library not found. Cannot save repository '{self.repo_path}'.", main_logger=save_logger_name)
             return
        if self.df is None:
             log_statement(loglevel="error", logstatement=f"{LOG_INS} - Cannot save repository, internal DataFrame is None.", main_logger=save_logger_name)
             return
        if not self.repo_path:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - Cannot save repository, output path (self.repo_path) is not set.", main_logger=save_logger_name)
            return

        # --- Handle Empty DataFrame ---
        if self.df.empty:
             log_statement(loglevel="warning", logstatement=f"{LOG_INS} - Attempting to save an empty DataFrame to {self.repo_path}.", main_logger=save_logger_name)
             try:
                  self.repo_path.parent.mkdir(parents=True, exist_ok=True)
                  cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
                  with open(self.repo_path, 'wb') as f_out:
                      with cctx.stream_writer(f_out) as writer:
                           # Write only the header using MAIN_REPO_HEADER
                           header_line = ",".join(MAIN_REPO_HEADER) + "\n"
                           writer.write(header_line.encode('utf-8'))
                  log_statement(loglevel="info", logstatement=f"{LOG_INS} - Empty repository file with header saved to {self.repo_path}.", main_logger=save_logger_name)
                  return
             except Exception as empty_save_e:
                  log_statement(loglevel="error", logstatement=f"{LOG_INS} - Failed to save empty repository file to {self.repo_path}: {empty_save_e}", main_logger=save_logger_name, exc_info=True)
                  return

        # --- Prepare Non-Empty DataFrame for Saving ---
        temp_path = self.repo_path.with_suffix(f'{self.repo_path.suffix}.tmp_{int(time.time())}')
        log_statement(loglevel="info", logstatement=f"{LOG_INS} - Saving repository ({len(self.df)} entries) to: {self.repo_path} via {temp_path}", main_logger=save_logger_name)

        # Acquire lock for thread safety during save preparation and writing
        with self.lock:
            try:
                self.repo_path.parent.mkdir(parents=True, exist_ok=True)

                # Create a copy to modify for saving
                df_copy = self.df.copy()

                # Ensure DataFrame has all expected columns from MAIN_REPO_HEADER
                expected_columns = MAIN_REPO_HEADER
                current_cols = df_copy.columns.tolist()
                missing_cols = [col for col in expected_columns if col not in current_cols]
                if missing_cols:
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Columns missing from DataFrame during save: {missing_cols}. Adding with defaults.", main_logger=save_logger_name)
                    for col in missing_cols:
                         target_dtype = self.columns_schema.get(col, 'string') # Default to string
                         if 'datetime' in str(target_dtype): df_copy[col] = pd.NaT
                         elif target_dtype == 'Int64': df_copy[col] = pd.NA
                         elif target_dtype == 'boolean': df_copy[col] = pd.NA
                         elif 'Float' in str(target_dtype) or 'float' in str(target_dtype): df_copy[col] = np.nan
                         else: df_copy[col] = ''
                    # Reorder columns according to MAIN_REPO_HEADER
                    df_copy = df_copy[expected_columns]

                # --- Prepare Data Types for CSV Saving ---
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Preparing data types for CSV saving...", main_logger=save_logger_name)
                for col_const in expected_columns:
                     if col_const not in df_copy.columns: continue # Should not happen now
                     series = df_copy[col_const]

                     # Convert standardized Timestamps to ISO strings
                     if col_const in self.timestamp_columns:
                         if pd.api.types.is_datetime64_any_dtype(series.dtype):
                             # Ensure UTC before formatting (should be already, but safe check)
                             if series.dt.tz is None: series = series.dt.tz_localize('UTC')
                             elif str(series.dt.tz).upper() != 'UTC': series = series.dt.tz_convert('UTC')
                             # Format NaT as empty string, others as ISO 8601 UTC
                             df_copy[col_const] = series.apply(lambda dt: dt.isoformat(timespec='microseconds') if pd.notna(dt) else '')
                         else:
                             # If somehow not datetime, try converting to string robustly
                             df_copy[col_const] = series.astype(str).fillna('')
                     # Convert Nullable Int/Bool/Float to strings (empty string for NA)
                     elif isinstance(series.dtype, pd.Int64Dtype) or isinstance(series.dtype, pd.BooleanDtype) or isinstance(series.dtype, pd.Float64Dtype):
                         df_copy[col_const] = series.apply(lambda x: '' if pd.isna(x) else str(x))
                     # Ensure other types (string, object) have NAs as empty strings
                     else:
                         df_copy[col_const] = series.fillna('')
                         # Optional: Cast to string explicitly if not already string/object
                         # if not pd.api.types.is_string_dtype(df_copy[col_const].dtype) and not pd.api.types.is_object_dtype(df_copy[col_const].dtype):
                         #     df_copy[col_const] = df_copy[col_const].astype(str)


                # --- Write to Compressed CSV ---
                chunk_size = 10000 # Configurable chunk size
                total_rows = len(df_copy)
                pbar = tqdm(total=total_rows, desc=f"Saving {self.repo_path.name}", unit="row", leave=False) # leave=False recommended for internal saves

                cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
                with open(temp_path, 'wb') as f_out:
                    with cctx.stream_writer(f_out) as writer:
                        # Write header using MAIN_REPO_HEADER
                        header_line = ",".join(expected_columns) + "\n"
                        writer.write(header_line.encode('utf-8'))
                        pbar.set_postfix_str("Header Written")

                        # Write data in chunks
                        for i in range(0, total_rows, chunk_size):
                            chunk = df_copy.iloc[i:min(i + chunk_size, total_rows)]
                            csv_buffer = io.StringIO()
                            # Use QUOTE_MINIMAL and specify header=False for chunks
                            chunk.to_csv(csv_buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\', encoding='utf-8')
                            writer.write(csv_buffer.getvalue().encode('utf-8'))
                            csv_buffer.close()
                            pbar.update(len(chunk))
                            pbar.set_postfix_str(f"Row {pbar.n}/{total_rows}")

                pbar.close()
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Write to temp file {temp_path} complete.", main_logger=save_logger_name)

                # --- Atomic Replace ---
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Moving temp file {temp_path} to {self.repo_path}", main_logger=save_logger_name)
                shutil.move(str(temp_path), str(self.repo_path))
                log_statement(loglevel="info", logstatement=f"{LOG_INS} - Successfully saved repository ({total_rows} entries) to {self.repo_path}", main_logger=save_logger_name)

            except Exception as e:
               log_statement(loglevel="critical", logstatement=f"{LOG_INS} - CRITICAL ERROR saving repository to {self.repo_path}: {e}", main_logger=save_logger_name, exc_info=True)
               # Attempt to clean up temp file on error
               if temp_path.exists():
                   try: os.remove(temp_path)
                   except OSError: log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Failed to remove temporary save file {temp_path} after error.", main_logger=save_logger_name)
               raise # Re-raise the exception after logging

    def get_summary_metadata(self) -> Dict[str, Any]:
        """
        Calculates summary metadata from the loaded repository DataFrame using constants
        and standardized timestamp handling. Thread-safe.

        Returns:
            Dict[str, Any]: Summary statistics (file_count, total_size_bytes, min/max_mtime_utc).
                            Returns empty dict on error or if DataFrame is empty/None.
        """
        LOG_INS = f'{__name__}::get_summary_metadata::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        summary_logger = __name__

        if not CONSTANTS_AVAILABLE: return {}

        summary = {}
        # Acquire lock for safe access to self.df
        with self.lock:
            if self.df is None or self.df.empty:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Cannot get summary metadata, DataFrame is empty or None.", main_logger=summary_logger)
                return {}

            try:
                # --- File Count ---
                summary['file_count'] = len(self.df)

                # --- Total Size ---
                if COL_SIZE in self.df.columns:
                    # Use pandas nullable Int64 type for robust sum
                    size_series = pd.to_numeric(self.df[COL_SIZE], errors='coerce').astype('Int64').fillna(0)
                    summary['total_size_bytes'] = int(size_series.sum()) # Cast final sum to standard int
                else:
                    summary['total_size_bytes'] = 0
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Column '{COL_SIZE}' not found for size summary.", main_logger=summary_logger)

                # --- Date Range (Modification Time) ---
                if COL_MTIME in self.df.columns:
                    # Ensure column is parsed correctly (should be datetime64[ns, UTC] from _load_repo)
                    if pd.api.types.is_datetime64_any_dtype(self.df[COL_MTIME]):
                        mtime_series = self.df[COL_MTIME].dropna() # Drop NaT values before min/max
                        if not mtime_series.empty:
                             # Output as ISO 8601 UTC strings
                             summary['min_mtime_utc'] = mtime_series.min().isoformat(timespec='microseconds')
                             summary['max_mtime_utc'] = mtime_series.max().isoformat(timespec='microseconds')
                        else: summary['min_mtime_utc'], summary['max_mtime_utc'] = None, None
                    else:
                        # Attempt parsing if not already datetime (indicates potential load issue)
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Column '{COL_MTIME}' is not datetime type ({self.df[COL_MTIME].dtype}). Attempting parse for summary.", main_logger=summary_logger)
                        mtime_series = pd.to_datetime(self.df[COL_MTIME], errors='coerce', utc=True).dropna()
                        if not mtime_series.empty:
                             summary['min_mtime_utc'] = mtime_series.min().isoformat(timespec='microseconds')
                             summary['max_mtime_utc'] = mtime_series.max().isoformat(timespec='microseconds')
                        else: summary['min_mtime_utc'], summary['max_mtime_utc'] = None, None
                else:
                    summary['min_mtime_utc'] = None
                    summary['max_mtime_utc'] = None
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Column '{COL_MTIME}' not found for date range summary.", main_logger=summary_logger)

                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Generated summary metadata: {summary}", main_logger=summary_logger)
                return summary

            except Exception as e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error calculating summary metadata: {e}", main_logger=summary_logger, exc_info=True)
                return {} # Return empty dict on error

    def scan_and_update(self, base_dir: Path):
        """
        Scans a directory, compares findings against the repository,
        and updates the repository using standardized methods.

        Args:
            base_dir (Path): The directory path object to scan and process.
        """
        LOG_INS = f'{__name__}::scan_and_update::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        scan_update_logger = __name__

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot scan and update.", main_logger=scan_update_logger)
             return
        if self.df is None:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Repository not loaded (self.df is None). Cannot scan.", main_logger=scan_update_logger)
             return
        if not self.lock:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Repository lock not initialized. Cannot scan.", main_logger=scan_update_logger)
             return

        base_dir = base_dir.resolve()
        base_dir_str = str(base_dir)
        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Starting scan and update process for: {base_dir_str}", main_logger=scan_update_logger)

        processed_count, skipped_count, new_files_count, updated_files_count, error_count, hash_fail_count = 0, 0, 0, 0, 0, 0

        with self.lock: # Lock needed for reading existing files and applying updates
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Lock acquired for scan/update.", main_logger=scan_update_logger)

            # --- Get Existing Files from Repo for Comparison ---
            try:
                # Efficiently get existing file paths and key metadata for comparison
                # Ensure COL_FILEPATH exists
                if COL_FILEPATH not in self.df.columns:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Key column '{COL_FILEPATH}' missing from DataFrame. Cannot proceed.", main_logger=scan_update_logger)
                    return

                existing_files_data = {}
                cols_to_compare = [COL_FILEPATH, COL_MTIME, COL_SIZE, COL_HASH, COL_STATUS] # Use constants
                # Ensure columns exist before selecting
                valid_cols = [col for col in cols_to_compare if col in self.df.columns]
                if COL_FILEPATH not in valid_cols: # Re-check after filtering
                     log_statement(loglevel='error', logstatement=f"{LOG_INS} - Key column '{COL_FILEPATH}' missing from DataFrame. Cannot proceed.", main_logger=scan_update_logger)
                     return

                # Create lookup dictionary {absolute_path_string: {col: value}}
                # Handle potential NaNs or different types during comparison later
                existing_files_data = self.df[valid_cols].set_index(COL_FILEPATH).to_dict('index')
                existing_files_set = set(existing_files_data.keys())
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Created lookup for {len(existing_files_set)} existing files.", main_logger=scan_update_logger)

            except Exception as e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error preparing existing file data: {e}. Aborting scan.", main_logger=scan_update_logger, exc_info=True)
                return

            # --- Delegate Scanning to _scan_directory ---
            # Pass the set of known resolved paths
            new_files_metadata = self._scan_directory(base_dir, existing_files_set)
            total_scanned_files = len(new_files_metadata)
            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Scan completed. Found metadata for {total_scanned_files} new/supported files.", main_logger=scan_update_logger)

            # --- Process Scan Results and Prepare Updates ---
            files_to_update_dict: Dict[str, Dict[str, Any]] = {} # {abs_path_str: update_data_dict}

            if not new_files_metadata:
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - No new files found requiring repository update.", main_logger=scan_update_logger)
                # No need to save if no updates. Consider saving if only deletions occurred?
                return # Exit early

            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Comparing {total_scanned_files} scanned files with repository...", main_logger=scan_update_logger)
            pbar_compare = tqdm(new_files_metadata, desc="Comparing Scan Results", unit="file", leave=False)

            for current_metadata in pbar_compare:
                processed_count += 1
                try:
                    abs_file_path_str = current_metadata.get(COL_FILEPATH)
                    if not abs_file_path_str:
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Scanned metadata missing filepath. Skipping entry: {current_metadata}", main_logger=scan_update_logger)
                        error_count += 1
                        continue

                    pbar_compare.set_postfix_str(f"File: {current_metadata.get(COL_FILENAME, 'N/A')}")

                    # Get current values from metadata (timestamps are ISO strings)
                    current_mtime_iso = current_metadata.get(COL_MTIME)
                    current_size = current_metadata.get(COL_SIZE)
                    current_hash = current_metadata.get(COL_HASH)

                    if current_hash is None or current_hash == "":
                        # Hash failed during scan (_get_file_metadata logs error)
                        hash_fail_count += 1
                        # Prepare update with error status
                        update_data = current_metadata.copy() # Start with all scanned data
                        update_data[COL_STATUS] = STATUS_ERROR
                        update_data[COL_ERROR] = "Hash calculation failed during scan."
                        update_data['base_dir'] = base_dir_str # Add base directory context
                        files_to_update_dict[abs_file_path_str] = update_data
                        new_files_count += 1 # Count as new/updated despite error
                        continue # Move to next file

                    # File exists in the repository (was filtered out by _scan_directory),
                    # This loop should only contain *new* files.
                    # The logic needs adjustment: scan_and_update should check ALL files,
                    # not just new ones found by _scan_directory.

                    # --- REVISED LOGIC ---
                    # scan_and_update should compare *all* found files (not pre-filtered)
                    # OR _scan_directory needs modification to return status of known files (changed/unchanged)
                    # Let's assume scan_and_update needs to do the comparison itself for now,
                    # meaning it needs to call _get_file_metadata for existing files too.
                    # This duplicates work done by _scan_directory if called separately.
                    # A better pattern:
                    # 1. add_folder calls _scan_directory(..., existing_files_set=set()) -> gets metadata for ALL files.
                    # 2. add_folder then compares this list with self.df and calls update_entry.
                    # Sticking to the current structure where scan_and_update is separate:

                    # --- Reverting to the Original scan_and_update Logic (Standardized) ---
                    # This means scan_and_update *re-scans* the directory and does comparisons.

                    # (Re-introduce the file iteration and comparison logic from the original scan_and_update,
                    # but use standardized methods and data types)

                    # This method becomes very complex and overlaps heavily with `add_folder` + `_scan_directory`.
                    # Recommendation: Remove `scan_and_update` and integrate its logic into `add_folder`,
                    # which would call `_scan_directory` and then perform the comparison and updates.

                    # --- Assuming we keep scan_and_update for now (Simplified version based on input): ---
                    # This simplified version just adds the *new* files found by _scan_directory.
                    # It doesn't handle updates to existing files found during the scan.

                    # File is new based on _scan_directory filtering
                    update_data = current_metadata.copy() # Start with all scanned data
                    update_data['base_dir'] = base_dir_str # Add base directory context
                    # Status should be STATUS_DISCOVERED from _get_file_metadata
                    files_to_update_dict[abs_file_path_str] = update_data
                    new_files_count += 1

                except Exception as compare_e:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error comparing/processing metadata for file: {current_metadata.get(COL_FILENAME)}: {compare_e}", main_logger=scan_update_logger, exc_info=True)
                    error_count += 1

            pbar_compare.close()

             # --- Apply updates using update_entry ---
            if files_to_update_dict:
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - Applying {len(files_to_update_dict)} repository updates (New files)...", main_logger=scan_update_logger)
                update_errors = 0
                pbar_update = tqdm(files_to_update_dict.items(), desc="Applying Updates", unit="file", leave=False)
                for file_path_str, update_data in pbar_update:
                    try:
                        # update_entry expects Path object
                        self.update_entry(Path(file_path_str), **update_data)
                    except Exception as update_e:
                        log_statement(loglevel='error', logstatement=f"{LOG_INS} - Failed to update entry for {file_path_str}: {update_e}", main_logger=scan_update_logger, exc_info=True)
                        update_errors += 1
                pbar_update.close()

                if update_errors > 0:
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Encountered {update_errors} errors during repository update phase.", main_logger=scan_update_logger)

                # --- Save Repository After Updates ---
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - Saving repository after updates...", main_logger=scan_update_logger)
                self.save() # Call standardized save method
            else:
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - No new file updates identified for the repository.", main_logger=scan_update_logger)

        # --- Final Log ---
        log_statement(
            loglevel="info",
            logstatement=(
                f"{LOG_INS} - Scan & Update complete for {base_dir_str}. "
                f"Checked: {processed_count}, "
                # f"Skipped (Unchanged/Type): {skipped_count}, " # Skipped counts handled in _scan_directory
                f"New Files Added: {new_files_count}, "
                # f"Existing Files Updated: {updated_files_count}, " # Update logic needs rework for this
                f"Hash Fails: {hash_fail_count}, "
                f"Other Errors: {error_count}"
            ),
            main_logger=scan_update_logger
        )

    def update_entry(self, source_filepath: Path, **kwargs):
        """
        Adds or updates a repository entry using absolute paths, targeting the correct
        repository file (main or sub-repo). Thread-safe and schema-aware.

        Standardizes incoming data based on the internal DataFrame schema
        (defined by _define_columns_schema) using pandas nullable types.

        Args:
            source_filepath (Path): The path of the file (absolute path preferred).
            **kwargs: Metadata key-value pairs to update (using COL_* constants as keys).
                      Values will be automatically converted to match the schema's target types.
        """
        # Define local logger identifier using inspect
        frame = inspect.currentframe()
        LOG_INS = f"{__name__}::{self.__class__.__name__}::update_entry::{frame.f_lineno if frame else 'UnknownLine'}"
        update_logger_name = self.__class__.__name__ # Log under the class name

        # --- Pre-checks ---
        # Check for lock initialization (should be done in __init__)
        if not hasattr(self, 'lock') or not isinstance(self.lock, threading.RLock):
             # Use chosen logging format
             # Log as critical because proceeding without a lock breaks thread-safety guarantee
             log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL: DataRepository lock not initialized correctly. Cannot safely update entry.", main_logger=update_logger_name)
             # Depending on desired behavior, could raise an error instead of returning
             # raise RuntimeError("DataRepository lock not initialized.")
             return

        # Check if constants are available (assuming this is a necessary check for the environment)
        # Make sure CONSTANTS_AVAILABLE is defined appropriately (e.g., in constants.py or checked in __init__)
        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Required constants are not available. Cannot update entry.", main_logger=update_logger_name)
             return

        # Check for valid input path
        if not source_filepath:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: source_filepath cannot be None or empty.", main_logger=update_logger_name)
            return

        # Check if any update data was provided
        if not kwargs:
             log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: update_entry called with no keyword arguments (update data) for {source_filepath.name}. No action taken.", main_logger=update_logger_name)
             return

        # --- Resolve Path ---
        try:
            # Ensure path is absolute for consistent identification and comparison
            source_filepath_abs = source_filepath.resolve()
            source_filepath_abs_str = str(source_filepath_abs)
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Resolved path: {source_filepath_abs_str}", main_logger=update_logger_name)
        except Exception as res_e:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Could not resolve input path '{source_filepath}': {res_e}. Cannot update.", main_logger=update_logger_name, exc_info=True)
             return

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Update START for: {source_filepath_abs.name}. Updates requested: {list(kwargs.keys())}", main_logger=update_logger_name)

        # --- Prepare Input Data ---
        update_data_in = kwargs.copy() # Work on a copy of input kwargs

        # Automatically set/overwrite the last updated timestamp
        update_data_in[COL_LAST_UPDATED] = pd.Timestamp.utcnow()
        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Set internal {COL_LAST_UPDATED}={update_data_in[COL_LAST_UPDATED]}", main_logger=update_logger_name)

        # --- Acquire Lock and Perform Update ---
        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Acquiring lock for {source_filepath_abs.name} update...", main_logger=update_logger_name)
        update_successful = False # Flag to track overall success within the lock
        target_df = None # Initialize target DataFrame
        target_repo_file_path = None # Initialize target file path

        with self.lock:
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Lock acquired for {source_filepath_abs.name}.", main_logger=update_logger_name)

            try:
                # === Step 1: Determine Target Repository and Load/Access DataFrame ===
                target_repo_file_path = self._get_target_repo_path(source_filepath_abs) # Use absolute path
                is_main_repo = (target_repo_file_path.resolve() == self.repo_file.resolve()) # Compare resolved paths

                if is_main_repo:
                    # Target is the main repository DataFrame held in memory
                    if self.df is None:
                        # Attempt to load if not already loaded (e.g., if repo was initialized empty)
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Main repository DataFrame (self.df) is None. Attempting load from {self.repo_file}", main_logger=update_logger_name)
                        self._load_repo() # This method should handle initializing empty DF if file not found
                        if self.df is None: # Check again after load attempt
                            log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed to load or initialize main repository DataFrame. Cannot update entry.", main_logger=update_logger_name)
                            # Cannot proceed, exit lock cleanly
                            # Use 'return' here as we are inside the 'with' block
                            return

                    target_df = self.df # Use the DataFrame held in memory
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Target is main repository DataFrame (in memory).", main_logger=update_logger_name)
                else:
                    # Target is a sub-repository, load its DataFrame temporarily
                    log_statement(loglevel='info', logstatement=f"{LOG_INS} - Target is sub-repository file: {target_repo_file_path}", main_logger=update_logger_name)
                    target_df = self._load_repo_dataframe(target_repo_file_path)
                    if target_df is None:
                        # If loading failed or file doesn't exist, create an empty one in memory to proceed
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Failed to load sub-repository {target_repo_file_path}, initializing empty DataFrame for update.", main_logger=update_logger_name)
                        target_df = pd.DataFrame(columns=self._get_expected_columns())
                        # Apply schema dtypes to the new empty DataFrame
                        target_df = target_df.astype(self._get_schema_dtypes(), errors='ignore')
                        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Initialized new empty DataFrame for sub-repo {target_repo_file_path.name}.", main_logger=update_logger_name)


                # Check if target_df is valid after loading/initialization attempts
                if target_df is None: # Should ideally not happen with the logic above
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Target DataFrame could not be loaded or initialized for {target_repo_file_path}. Cannot proceed.", main_logger=update_logger_name)
                    return # Exit lock cleanly

                # === Step 2: Align Target DataFrame Columns with Authoritative Schema ===
                authoritative_schema_dict = self._define_columns_schema()
                authoritative_cols = authoritative_schema_dict.keys()
                added_cols_align = []
                current_target_cols = target_df.columns.tolist() # Get columns before alignment

                for col in authoritative_cols:
                    if col not in target_df.columns:
                        added_cols_align.append(col)
                        dtype_str = authoritative_schema_dict[col] # Get dtype string from schema
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Aligning Schema: Adding missing column to Target DF '{target_repo_file_path.name}': {col} (Type: {dtype_str})", main_logger=update_logger_name)
                        # Initialize with appropriate null type based on authoritative schema dtype string
                        # Directly assign Series with correct dtype and index matching target_df
                        target_dtype_obj = self._get_schema_dtypes().get(col) # Get the actual dtype object
                        if target_dtype_obj == pd.DatetimeTZDtype(tz='UTC'): series_data = pd.NaT
                        elif target_dtype_obj == pd.Int64Dtype(): series_data = pd.NA
                        elif target_dtype_obj == pd.BooleanDtype(): series_data = pd.NA
                        elif target_dtype_obj == pd.Float64Dtype(): series_data = pd.NA
                        elif target_dtype_obj == pd.StringDtype(): series_data = pd.NA
                        else: series_data = None # For object or basic types
                        target_df[col] = pd.Series(series_data, index=target_df.index, dtype=target_dtype_obj if target_dtype_obj else 'object')


                if added_cols_align:
                     log_statement(loglevel='info', logstatement=f"{LOG_INS} - Aligned target DataFrame columns for '{target_repo_file_path.name}'. Added: {added_cols_align}. Prev cols: {current_target_cols}, New cols: {target_df.columns.tolist()}", main_logger=update_logger_name)

                # Ensure column order matches authoritative schema for consistency
                target_df = target_df.reindex(columns=authoritative_cols)

                # === Step 3: Prepare Update Data Types based on Target DF Schema ===
                prepared_update_data = {}
                target_dtypes_map = target_df.dtypes.to_dict() # Get actual dtypes from aligned target DF
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Preparing update data against target dtypes: {target_dtypes_map}", main_logger=update_logger_name)

                for col, input_value in update_data_in.items():
                    if col not in target_dtypes_map:
                        # This case means input key is not in the authoritative schema
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Input column '{col}' is not in the authoritative schema. Skipping update for this column.", main_logger=update_logger_name)
                        continue

                    target_dtype_obj = target_dtypes_map[col] # Get the actual dtype object
                    prepared_value = None
                    # Use pandas isna for robust null checking, also treat empty strings as null for non-string types?
                    # Let's treat empty string as null only if target is NOT string type.
                    is_null = pd.isna(input_value) or (isinstance(input_value, str) and not input_value.strip() and not isinstance(target_dtype_obj, pd.StringDtype))


                    try:
                        # --- Handle Null Assignment ---
                        if is_null:
                            if isinstance(target_dtype_obj, pd.DatetimeTZDtype): prepared_value = pd.NaT
                            elif isinstance(target_dtype_obj, (pd.Int64Dtype, pd.BooleanDtype, pd.Float64Dtype, pd.StringDtype)): prepared_value = pd.NA
                            else: prepared_value = None # Fallback for basic dtypes (object, maybe float64)
                            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Prepared NULL value for Col='{col}' (Target Dtype: {target_dtype_obj}) as: {type(prepared_value)}", main_logger=update_logger_name)

                        # --- Handle Type Conversions for Non-Null Values ---
                        elif col in self.timestamp_columns:
                            if isinstance(input_value, (int, float)): # Handle Unix timestamps
                                prepared_value = pd.Timestamp.fromtimestamp(input_value, tz=timezone.utc)
                            else: # Assume string, datetime, or pd.Timestamp - parse robustly
                                prepared_value = pd.to_datetime(input_value, errors='coerce', utc=True) # Coerce invalid dates to NaT
                            if pd.isna(prepared_value): log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Could not parse timestamp for Col='{col}', Input='{input_value}'. Setting NaT.", main_logger=update_logger_name)

                        elif isinstance(target_dtype_obj, pd.Int64Dtype):
                            prepared_value = int(float(input_value)) # Convert via float first for flexibility (e.g., "1.0")

                        elif isinstance(target_dtype_obj, pd.BooleanDtype):
                            if isinstance(input_value, str): v_lower = input_value.strip().lower()
                            else: v_lower = str(input_value).lower()
                            if v_lower in ['true', 'yes', 'y', '1', 't']: prepared_value = True
                            elif v_lower in ['false', 'no', 'n', '0', 'f']: prepared_value = False
                            else: prepared_value = pd.NA # Invalid boolean input becomes NA

                        elif isinstance(target_dtype_obj, pd.Float64Dtype):
                            prepared_value = float(input_value)

                        elif isinstance(target_dtype_obj, pd.StringDtype):
                             prepared_value = str(input_value) # Convert any input to string
                             # Optional truncation (example for error column)
                             if col == COL_ERROR_INFO and len(prepared_value) > 1024:
                                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Truncating long string for column '{col}'. Original length: {len(prepared_value)}", main_logger=update_logger_name)
                                prepared_value = prepared_value[:1024] + '...'
                        else: # Fallback for object, basic float/int etc. (should be minimal if schema uses pandas types)
                            prepared_value = str(input_value) # Store as string by default

                        # --- Final Assignment ---
                        prepared_update_data[col] = prepared_value
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Prepared Col='{col}', Input='{input_value}', Prepared='{prepared_value}' (Target Type: {target_dtype_obj})", main_logger=update_logger_name)

                    except (ValueError, TypeError, OverflowError) as conv_e:
                        # Log error during conversion and skip updating this specific column
                        log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Type conversion failed for Col='{col}', Input='{input_value}', TargetType='{target_dtype_obj}': {conv_e}. Skipping update for this column.", main_logger=update_logger_name, exc_info=False) # Keep log concise
                        # Do not add col to prepared_update_data if conversion fails

                # === Step 4: Apply Updates to Target DataFrame ===
                if not prepared_update_data:
                     log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: No data could be prepared for update for {source_filepath_abs_str} after type conversion. No changes applied.", main_logger=update_logger_name)
                     # Mark as successful if no data needed updating? Or unsuccessful? Let's say neutral/successful.
                     update_successful = True
                     return # Exit lock cleanly

                # Find existing entry index using the absolute path string
                try:
                    mask = target_df[COL_FILEPATH] == source_filepath_abs_str
                    indices = target_df.index[mask].tolist()
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Found {len(indices)} existing indices for {source_filepath_abs.name} in target DF '{target_repo_file_path.name}'.", main_logger=update_logger_name)
                except KeyError:
                     # This should not happen if COL_FILEPATH is guaranteed by schema alignment
                     log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Cannot update. Key column '{COL_FILEPATH}' missing from target DataFrame '{target_repo_file_path.name}' despite alignment.", main_logger=update_logger_name)
                     return # Exit lock cleanly

                if indices: # --- Update Existing Row(s) ---
                    idx = indices[0] # Update first match
                    log_statement(loglevel='info', logstatement=f"{LOG_INS} - Updating existing row at index {idx} for {source_filepath_abs_str} in '{target_repo_file_path.name}' with {len(prepared_update_data)} values: {list(prepared_update_data.keys())}", main_logger=update_logger_name)
                    if len(indices) > 1:
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Multiple entries found for '{source_filepath_abs_str}' in '{target_repo_file_path.name}'. Updating only the first at index {idx}.", main_logger=update_logger_name)

                    row_update_errors = 0
                    for col, value in prepared_update_data.items():
                        try:
                            # Use .loc for reliable assignment by index and column label
                            target_df.loc[idx, col] = value
                        except Exception as assign_e:
                             # Catch potential errors during assignment (though less likely with prepared types)
                             log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed direct assign for Col='{col}' at Index={idx} in '{target_repo_file_path.name}': {assign_e}", main_logger=update_logger_name, exc_info=True)
                             row_update_errors += 1

                    if row_update_errors == 0:
                         update_successful = True # Mark successful if all columns assigned
                    else:
                         log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Update failed for {row_update_errors}/{len(prepared_update_data)} columns for existing row {idx} ({source_filepath_abs_str}).", main_logger=update_logger_name)
                         # Keep update_successful = False

                else: # --- Add New Row ---
                    log_statement(loglevel='info', logstatement=f"{LOG_INS} - Adding new row for {source_filepath_abs_str} to '{target_repo_file_path.name}'.", main_logger=update_logger_name)
                    # Start with default nulls based on authoritative schema dtypes
                    schema_dtypes_map = self._get_schema_dtypes()
                    new_data = {}
                    for col, dtype_obj in schema_dtypes_map.items():
                         if isinstance(dtype_obj, pd.DatetimeTZDtype): new_data[col] = pd.NaT
                         elif isinstance(dtype_obj, (pd.Int64Dtype, pd.BooleanDtype, pd.Float64Dtype, pd.StringDtype)): new_data[col] = pd.NA
                         else: new_data[col] = None

                    # Set the essential filepath (already validated as absolute string)
                    new_data[COL_FILEPATH] = source_filepath_abs_str

                    # Overwrite defaults with prepared update data
                    update_applied_count = 0
                    for col, value in prepared_update_data.items():
                        if col in new_data: # Should always be true due to schema alignment
                            new_data[col] = value
                            update_applied_count += 1
                        else: # Should not happen
                             log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Column '{col}' from prepared data not in new row schema. Skipping for new row.", main_logger=update_logger_name)

                    log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Applied {update_applied_count}/{len(prepared_update_data)} prepared values to new row data.", main_logger=update_logger_name)

                    # Fill missing mandatory fields if still null (e.g., if not provided in kwargs)
                    if pd.isna(new_data.get(COL_STATUS)):
                         new_data[COL_STATUS] = STATUS_DISCOVERED # Default status
                         log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Setting default status '{STATUS_DISCOVERED}' for new row.", main_logger=update_logger_name)
                    if pd.isna(new_data.get(COL_HASH)):
                         # Hash is critical, maybe try to calculate it if missing?
                         # Or just warn. For now, warn.
                         # hash_value = generate_data_hash(source_filepath_abs) # Requires file read -> performance impact
                         log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Adding new row for {source_filepath_abs_str} without required '{COL_HASH}'.", main_logger=update_logger_name)


                    # Create DataFrame for the new row, ensure columns match target_df for concat
                    try:
                        new_row_df = pd.DataFrame([new_data], columns=target_df.columns)
                        # Align dtypes just before concat using target_df's dtypes
                        new_row_df = new_row_df.astype(target_df.dtypes.to_dict(), errors='ignore')
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - New row DataFrame created and types aligned.", main_logger=update_logger_name)

                        # Concatenate new row using pandas concat
                        target_df = pd.concat([target_df, new_row_df], ignore_index=True)
                        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Concatenated new row successfully for {source_filepath_abs_str}.", main_logger=update_logger_name)
                        update_successful = True # Mark successful

                    except Exception as concat_err:
                         log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed to create or concatenate new row for {source_filepath_abs_str}: {concat_err}", main_logger=update_logger_name, exc_info=True)
                         update_successful = False # Mark failed


                # === Step 5: Post-Update - Save Sub-Repo or Update Main DF Reference ===
                if update_successful:
                    if is_main_repo:
                         self.df = target_df # Update the main DataFrame reference
                         log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Main DataFrame reference (self.df) updated in memory.", main_logger=update_logger_name)
                         # Main repo saving is handled by explicit save_repo() call outside this method typically

                    else: # Sub-repository was modified
                         log_statement(loglevel='info', logstatement=f"{LOG_INS} - Saving updated sub-repository file immediately: {target_repo_file_path}", main_logger=update_logger_name)
                         # Save the temporarily loaded/modified sub-repo DF
                         save_success = self._save_repo_dataframe(target_df, target_repo_file_path)
                         if save_success:
                             # Update the timestamp in the main index for this sub-repo
                             try:
                                 # Get relative path for index key
                                 subrepo_relative_dir = target_repo_file_path.parent.relative_to(self.root_dir)
                                 self._update_index_timestamp(subrepo_relative_dir)
                             except ValueError as rel_path_err:
                                  log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Could not determine relative path for sub-repo '{target_repo_file_path.parent}' to update index timestamp: {rel_path_err}", main_logger=update_logger_name)
                             except Exception as idx_time_err:
                                  log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed to update index timestamp for sub-repo '{target_repo_file_path.parent}': {idx_time_err}", main_logger=update_logger_name, exc_info=True)
                         else:
                             log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed to save updated sub-repository file {target_repo_file_path}. Changes might be lost.", main_logger=update_logger_name)
                             update_successful = False # Mark overall failure if sub-repo save fails

                else: # update_successful is False
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Update deemed unsuccessful for {source_filepath_abs_str}. No changes will be persisted for this file in this operation.", main_logger=update_logger_name)


            except Exception as e:
                # Catch exceptions happening within the lock after initial checks
                log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL: Unhandled exception during locked update operation for {source_filepath_abs_str}: {e}", main_logger=update_logger_name, exc_info=True)
                update_successful = False # Ensure it's marked as failed
            finally:
                # Log release regardless of success/failure inside the block
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Releasing lock for {source_filepath_abs.name}.", main_logger=update_logger_name)
                # Lock is released automatically by exiting the 'with' block

        # Final log message outside the lock
        log_statement(loglevel='info' if update_successful else 'error', logstatement=f"{LOG_INS} - Update END for: {source_filepath_abs.name}. Success: {update_successful}", main_logger=update_logger_name)

    def _update_index_timestamp(self, subrepo_relative_dir: Path):
        """Updates the last_updated timestamp for a sub-repository in the index using relative path."""
        log_statement('debug', f"{LOG_INS}::_update_index_timestamp - Attempting to update index timestamp for relative dir: {subrepo_relative_dir}", __file__)
        if not self.repo_index:
            log_statement('warning', f"{LOG_INS}::_update_index_timestamp - WARNING: Cannot update index timestamp for {subrepo_relative_dir}. Index not loaded.", __file__)
            return False # Indicate failure

        try:
            # Convert relative Path object to the string format used as key in the index (e.g., forward slashes)
            subrepo_root_key = str(subrepo_relative_dir).replace("\\", "/")
            current_timestamp = time.time() # Use Unix float timestamp

            if subrepo_root_key in self.repo_index:
                self.repo_index[subrepo_root_key]['last_updated'] = current_timestamp
                log_statement('debug', f"{LOG_INS}::_update_index_timestamp - Updated timestamp for sub-repo key '{subrepo_root_key}' in index to {current_timestamp}", __file__)
                # Mark index as dirty to ensure it's saved later
                self.repo_index_dirty = True # Assumes this flag exists and is checked by _save_repository_index
                return True # Indicate success
            else:
                log_statement('warning', f"{LOG_INS}::_update_index_timestamp - WARNING: Attempted to update timestamp for unknown sub-repo key '{subrepo_root_key}' (derived from {subrepo_relative_dir}) in index.", __file__)
                return False # Indicate failure: key not found

        except ValueError as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Could not get relative path for sub-repo timestamp update: {subrepo_root_path}. Error: {e}", __file__)
        except Exception as e:
            log_statement('error', f"{LOG_INS}::_update_index_timestamp - ERROR: Failed to update index timestamp for {subrepo_relative_dir}: {e}", __file__, exc_info=True)
            return False # Indicate failure

    def get_processed_path(self, source_filepath: Path) -> Optional[Path]:
        """
        Gets the absolute processed path for a given source path using constants.
        Thread-safe.
        """
        get_logger_name = __file__

        if not self.lock:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Lock not initialized.", main_logger=get_logger_name)
             return None
        if not CONSTANTS_AVAILABLE: return None
        if self.df is None: return None

        processed_path: Optional[Path] = None
        try:
            source_filepath_abs_str = str(source_filepath.resolve())
        except Exception as res_e:
             log_statement(loglevel="error", logstatement=f"{LOG_INS} - Could not resolve input path '{source_filepath}': {res_e}", main_logger=get_logger_name)
             return None

        with self.lock:
            try:
                # Check columns exist
                if COL_FILEPATH not in self.df.columns or COL_PROCESSED_PATH not in self.df.columns:
                     log_statement(loglevel='error', logstatement=f"{LOG_INS} - Required columns missing ({COL_FILEPATH}, {COL_PROCESSED_PATH}).", main_logger=get_logger_name)
                     return None

                entry = self.df[self.df[COL_FILEPATH] == source_filepath_abs_str]
                if not entry.empty:
                    processed_path_str = entry[COL_PROCESSED_PATH].iloc[0]
                    # Ensure it's a non-empty string before creating Path object
                    if processed_path_str and isinstance(processed_path_str, str):
                        try:
                            processed_path = Path(processed_path_str)
                        except Exception as path_e:
                            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Could not create Path object from stored processed path '{processed_path_str}': {path_e}", main_logger=get_logger_name)
                            processed_path = None # Reset on error
                    # else: path is empty string or not string, return None
            except Exception as e:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error getting processed path for '{source_filepath_abs_str}': {e}", main_logger=get_logger_name, exc_info=True)
                 processed_path = None

        return processed_path

    # Method _calculate_hash is REMOVED. Use generate_data_hash from hashing utils.

    # --- Other methods like add_folder, remove_entry etc. would go here ---
    # Ensure they are also standardized to use constants, ISO timestamps where applicable,
    # central hashing functions, and acquire the lock for DF modifications.
    # Example: add_folder would call _scan_directory, then loop through results,
    # compare with self.df, and call update_entry as needed, finally calling _save_repo.

    def _load_repo_dataframe(self, repo_file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads a repository DataFrame from a specified .repo_state.csv.zst file.

        Handles decompression, CSV parsing, legacy column renaming, schema alignment
        (adding missing columns), and robust type conversion based on the authoritative schema.
        Returns a DataFrame conforming to the schema or None on critical load failure.

        Args:
            repo_file_path (Path): The full path to the .repo_state.csv.zst file.

        Returns:
            Optional[pd.DataFrame]: Loaded and processed DataFrame, or None on failure.
                 Returns an empty, schema-aligned DataFrame if the file is not found or empty.
        """
        frame = inspect.currentframe()
        LOG_INS = f"{LOG_INS_MODULE}::_load_repo_dataframe::{frame.f_lineno if frame else 'UnknownLine'}"
        load_logger_name = LOG_INS_MODULE

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Attempting to load repository DataFrame from: {repo_file_path}", main_logger=load_logger_name)

        # --- Get Authoritative Schema Info ---
        try:
            target_schema = self._define_columns_schema()
            expected_header = self._get_expected_columns() # Defines order and full set
            schema_dtypes = self._get_schema_dtypes() # Defines target pandas dtypes
        except Exception as schema_e:
             log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL: Failed to define repository schema: {schema_e}. Cannot load repository.", main_logger=load_logger_name, exc_info=True)
             return None # Cannot proceed without schema

        # --- Create Empty DataFrame Structure (for returning on error/empty) ---
        try:
            empty_df = pd.DataFrame(columns=expected_header).astype(schema_dtypes, errors='ignore')
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Prepared empty DataFrame structure with target schema.", main_logger=load_logger_name)
        except Exception as empty_df_e:
            log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL: Failed to create empty DataFrame structure: {empty_df_e}. Load may fail.", main_logger=load_logger_name, exc_info=True)
            # Fallback to completely empty DF if schema application fails
            empty_df = pd.DataFrame()


        # --- Validate Repo Path & Check Existence/Size ---
        if not repo_file_path or not isinstance(repo_file_path, Path):
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Repository path is invalid or not set ({repo_file_path}). Returning empty DF.", main_logger=load_logger_name)
            return empty_df

        if not repo_file_path.exists():
            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Repository file '{repo_file_path}' not found. Returning empty DF.", main_logger=load_logger_name)
            return empty_df

        try:
            if repo_file_path.stat().st_size < 10: # Heuristic: Allow slightly more than 5 for header variations/newline
                log_statement(loglevel="warning", logstatement=f"{LOG_INS} - WARNING: Repo file '{repo_file_path}' appears empty or has only header (size < 10 bytes). Returning empty DF.", main_logger=load_logger_name)
                return empty_df
        except OSError as stat_e:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - ERROR: Error accessing repo file stats '{repo_file_path}': {stat_e}. Returning empty DF.", main_logger=load_logger_name)
            return empty_df

        log_statement(loglevel="info", logstatement=f"{LOG_INS} - Loading data repository: {repo_file_path}", main_logger=load_logger_name)

        # --- Decompress, Load, and Process File ---
        try:
            pdf = None # Loaded DataFrame placeholder

            # Check for zstandard library availability if needed (assuming ZSTD_AVAILABLE is set elsewhere)
            # if not ZSTD_AVAILABLE:
            #     log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Cannot load repository '{repo_file_path}' - zstandard library is required.", main_logger=load_logger_name)
            #     return empty_df

            # --- Decompress and Read CSV using pandas ---
            dctx = zstd.ZstdDecompressor()
            with open(repo_file_path, 'rb') as ifh:
                with dctx.stream_reader(ifh) as reader:
                    # Wrap binary stream with TextIOWrapper for pandas
                    with io.TextIOWrapper(reader, encoding='utf-8', errors='replace') as text_reader:
                        # Define values to treat as NA (include our standard <NA>)
                        na_values_list = ['<NA>', '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                                         '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null']
                        # Read data using pandas. Read all as string initially for robust handling.
                        # keep_default_na=False helps distinguish empty strings from NaN if needed later,
                        # but using na_values list is generally better. Use keep_default_na=True with na_values list.
                        pdf = pd.read_csv(
                            text_reader,
                            header=0,              # Use first row as header
                            dtype=str,             # Read all columns as string initially
                            encoding='utf-8',
                            keep_default_na=True, # Use pandas default NaN handling
                            na_values=na_values_list, # Specify strings to recognize as NaN
                            low_memory=False,      # Can prevent DtypeWarning but uses more memory
                            escapechar='\\'        # Define escape character if used during save
                        )

            if pdf is None or pdf.empty:
                log_statement(loglevel="info", logstatement=f"{LOG_INS} - Repository file '{repo_file_path}' loaded as empty or only header. Returning empty DF.", main_logger=load_logger_name)
                return empty_df

            log_statement(loglevel="info", logstatement=f"{LOG_INS} - CSV loaded {len(pdf)} rows from '{repo_file_path}'. Normalizing columns...", main_logger=load_logger_name)
            original_columns = pdf.columns.tolist() # Keep track of original cols for logging

            # --- Column Renaming (Map known legacy names -> current constants) ---
            # Reuse rename_map from user's version (ensure constants are imported/defined)
            rename_map = {
                'Filepath': COL_FILEPATH, 'Filename': COL_FILENAME, 'filepath': COL_FILEPATH,
                'Ext': COL_EXTENSION, 'extension': COL_EXTENSION, 'Filetype': COL_EXTENSION,
                'Size': COL_SIZE, 'size_bytes': COL_SIZE,
                'ModTime': COL_MTIME, 'mtime_ts': COL_MTIME, 'ModificationDate': COL_MTIME, 'last_modified_scan': COL_MTIME,
                'FileCreationTime': COL_CTIME, 'ctime_ts': COL_CTIME,
                'Hash': COL_HASH, 'content_hash': COL_HASH, 'DataHash': COL_HASH, # Map DataHash and content_hash -> COL_HASH
                'HashedPathID': COL_HASHED_PATH_ID, 'path_hash': COL_HASHED_PATH_ID,
                'Compressed': COL_COMPRESSED_FLAG, 'is_compressed': COL_COMPRESSED_FLAG,
                'Status': COL_STATUS, 'status': COL_STATUS,
                'ErrorMSG': COL_ERROR_INFO, 'error_message': COL_ERROR_INFO, 'COL_ERROR': COL_ERROR_INFO, # Map COL_ERROR -> COL_ERROR_INFO
                'Designation': COL_DESIGNATION, 'designation': COL_DESIGNATION,
                'TokenizedPath': COL_TOKENIZED_PATH, 'tokenized_path': COL_TOKENIZED_PATH,
                'ProcessedPath': COL_PROCESSED_PATH, 'processed_path': COL_PROCESSED_PATH,
                'BaseDirectory': BASE_DATA_DIR, 'base_dir': BASE_DATA_DIR,
                'IsCopy': COL_IS_COPY_FLAG, 'is_copy': COL_IS_COPY_FLAG,
                'last_updated_repo': COL_LAST_UPDATED, 'last_updated_ts': COL_LAST_UPDATED,
                # Add other potential legacy names here...
            }
            # Filter map: only apply if old name exists and old != new
            actual_rename_map = {old: new for old, new in rename_map.items() if old in pdf.columns and old != new}

            # Pre-Rename Check for Duplicate Targets (from user version)
            target_counts = collections.Counter(actual_rename_map.values())
            duplicate_targets = {target: count for target, count in target_counts.items() if count > 1}
            if duplicate_targets:
                problematic_renames = {old: new for old, new in actual_rename_map.items() if new in duplicate_targets}
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Rename conflict! Multiple columns map to same target: {problematic_renames}. Skipping these renames.", main_logger=load_logger_name)
                actual_rename_map = {old: new for old, new in actual_rename_map.items() if new not in duplicate_targets}

            if actual_rename_map:
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - Applying column renames: {actual_rename_map}", main_logger=load_logger_name)
                pdf.rename(columns=actual_rename_map, inplace=True, errors='raise') # Raise error if rename fails unexpectedly
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Columns after rename: {pdf.columns.tolist()}", main_logger=load_logger_name)

            # --- Create final_df based on authoritative expected_header (order matters) ---
            final_df = pd.DataFrame(index=pdf.index) # Preserve index from loaded data
            processed_cols_from_pdf = set()
            missing_cols_info = {}

            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Building final_df using Expected Header order: {expected_header}", main_logger=load_logger_name)
            for col_const in expected_header:
                if col_const in pdf.columns:
                    # Column exists in the (potentially renamed) pdf
                    if isinstance(pdf[col_const], pd.DataFrame): # Handle duplicate columns from source CSV
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Duplicate column '{col_const}' detected in source CSV '{repo_file_path}'. Using first instance.", main_logger=load_logger_name)
                        final_df[col_const] = pdf[col_const].iloc[:, 0]
                    else:
                        final_df[col_const] = pdf[col_const] # Copy Series data
                    processed_cols_from_pdf.add(col_const)
                else:
                    # Expected column is missing from loaded data, add it
                    missing_cols_info[col_const] = schema_dtypes.get(col_const, 'object') # Store target dtype
                    # Initialize with Nones/NaNs - type conversion will happen later
                    final_df[col_const] = pd.Series([None] * len(pdf), index=pdf.index)
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Adding missing expected column '{col_const}' (will apply type later).", main_logger=load_logger_name)

            if missing_cols_info:
                 added_str = ", ".join([f"'{c}'" for c in missing_cols_info.keys()])
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Columns added to conform to schema for '{repo_file_path}': {added_str}", main_logger=load_logger_name)

            # Log any columns present in CSV but *not* in expected header (ignored columns)
            ignored_cols = [col for col in pdf.columns if col not in processed_cols_from_pdf and col not in actual_rename_map.keys()]
            if ignored_cols:
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Ignored columns found in '{repo_file_path}' (not in expected header or rename map): {ignored_cols}", main_logger=load_logger_name)

            # --- Apply Robust Type Conversions using target_schema (schema_dtypes) ---
            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Applying target schema type conversions...", main_logger=load_logger_name)
            conversion_errors = {}
            for col_const, target_dtype_obj in schema_dtypes.items():
                if col_const not in final_df.columns:
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Cannot apply type for '{col_const}', column missing after final DF build.", main_logger=load_logger_name)
                    continue # Skip if column somehow still missing

                current_series = final_df[col_const]
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Converting '{col_const}' (current: {current_series.dtype}) to target: {target_dtype_obj}...", main_logger=load_logger_name)
                try:
                    if isinstance(target_dtype_obj, pd.DatetimeTZDtype):
                        # Parse ISO strings (or other formats) into UTC datetime
                        # errors='coerce' turns unparseable values into NaT
                        converted_dt = pd.to_datetime(current_series, errors='coerce', utc=True)
                        final_df[col_const] = converted_dt.astype(target_dtype_obj) # Ensure correct final type

                    elif isinstance(target_dtype_obj, pd.Int64Dtype):
                        # Convert to numeric (float first allows "1.0"), coerce errors, then nullable Int64
                        numeric_series = pd.to_numeric(current_series, errors='coerce')
                        final_df[col_const] = numeric_series.astype('Float64').astype(target_dtype_obj)

                    elif isinstance(target_dtype_obj, pd.StringDtype):
                        # Convert to pandas nullable string, preserving NA
                        final_df[col_const] = current_series.astype(target_dtype_obj)

                    elif isinstance(target_dtype_obj, pd.Float64Dtype):
                        # Convert to numeric, coerce errors, then nullable Float64
                        numeric_series = pd.to_numeric(current_series, errors='coerce')
                        final_df[col_const] = numeric_series.astype(target_dtype_obj)

                    elif isinstance(target_dtype_obj, pd.BooleanDtype):
                        # Map common string representations, handle existing bools/NAs
                        if pd.api.types.is_bool_dtype(current_series.dtype): # Already boolean-like
                             final_df[col_const] = current_series.astype(target_dtype_obj)
                        else: # Assume string or object, try mapping
                             bool_map = {'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                                         'false': False, 'no': False, 'n': False, '0': False, 'f': False,
                                         '': pd.NA, '<na>': pd.NA} # Include our saved NA rep
                             # Fill pandas NA/None before lowercasing, then map
                             mapped_bool = current_series.fillna(pd.NA).astype(str).str.lower().map(bool_map)
                             # If map fails (unrecognized string), result is NaN -> convert to pd.NA
                             final_df[col_const] = mapped_bool.fillna(pd.NA).astype(target_dtype_obj)
                    else:
                        # Attempt direct astype for other specific numpy dtypes (e.g., 'int32', 'float32')
                        if current_series.dtype != target_dtype_obj:
                            final_df[col_const] = current_series.astype(target_dtype_obj) # Let pandas handle conversion

                    log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Conversion result for '{col_const}': {final_df[col_const].dtype}", main_logger=load_logger_name)

                except Exception as conv_e:
                    conversion_errors[col_const] = str(conv_e)
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR converting column '{col_const}' to '{target_dtype_obj}' in '{repo_file_path}': {conv_e}. Keeping as object/string.", main_logger=load_logger_name, exc_info=False)
                    # Fallback: Keep as object or try converting to nullable string
                    try:
                         final_df[col_const] = final_df[col_const].astype(pd.StringDtype())
                    except Exception:
                         final_df[col_const] = final_df[col_const].astype('object')


            if conversion_errors:
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Type conversion errors occurred for columns: {list(conversion_errors.keys())} in '{repo_file_path}'. Data may not fully match schema.", main_logger=load_logger_name)


            # --- Final Check & Return ---
            # Ensure final column order one last time (should be correct from earlier build)
            final_df = final_df.reindex(columns=expected_header)
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Final column order ensured: {final_df.columns.tolist()}", main_logger=load_logger_name)
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Final Dtypes after all processing: {final_df.dtypes.to_dict()}", main_logger=load_logger_name)
            log_statement(loglevel="info", logstatement=f"{LOG_INS} - Repository loaded and processed ({len(final_df)} entries) from '{repo_file_path}'.", main_logger=load_logger_name)
            return final_df

        # --- Exception Handling for Overall Load Process ---
        except pd.errors.EmptyDataError:
            log_statement(loglevel="warning", logstatement=f"{LOG_INS} - WARNING: Repository file '{repo_file_path}' contains no data or only header (pandas EmptyDataError). Returning empty DF with schema.", main_logger=load_logger_name)
            return empty_df # Return structured empty DF
        except FileNotFoundError: # Should be caught earlier, but as safeguard
            log_statement('info', f"{LOG_INS} - Repository file {repo_file_path} not found during load attempt. Returning empty DF.", __file__)
            return empty_df
        except zstd.ZstdError as zstde:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Zstandard decompression error reading '{repo_file_path}': {zstde}. File might be corrupted or not Zstd. Returning empty DF.", main_logger=load_logger_name, exc_info=True)
             return empty_df
        except Exception as e:
            log_statement(loglevel="critical", logstatement=f"{LOG_INS} - CRITICAL: Repository load failed for '{repo_file_path}': {e}. Returning empty DF with schema.", main_logger=load_logger_name, exc_info=True)
            return empty_df

    def _save_repo_dataframe(self, df_to_save: pd.DataFrame, target_repo_file_path: Path) -> bool:
        """
        Saves a DataFrame to a specified .repo_state.csv.zst file.

        Handles pandas nullable types, ensures schema conformance, converts timestamps
        to ISO strings, and includes an optional backup mechanism on failure.

        Args:
            df_to_save (pd.DataFrame): The DataFrame to save.
            target_repo_file_path (Path): The full path for the output file.

        Returns:
            bool: True on success, False on failure.
        """
        # Use module-level ID for logging within helper methods
        frame = inspect.currentframe()
        LOG_INS = f"{LOG_INS_MODULE}::_save_repo_dataframe::{frame.f_lineno if frame else 'UnknownLine'}"
        save_logger_name = LOG_INS_MODULE # Log under the module name

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Attempting to save DataFrame ({'None' if df_to_save is None else f'{len(df_to_save)} rows'}) to: {target_repo_file_path}", main_logger=save_logger_name)

        if df_to_save is None:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Attempted to save a None DataFrame to {target_repo_file_path}. Aborting save.", main_logger=save_logger_name)
            return False

        try:
            # Ensure parent directory exists
            target_repo_file_path.parent.mkdir(parents=True, exist_ok=True)
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Ensured parent directory exists: {target_repo_file_path.parent}", main_logger=save_logger_name)

            # --- Prepare DataFrame for Saving ---
            # Make a copy to avoid modifying the original DataFrame passed to the function
            df_copy = df_to_save.copy()
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Created copy of DataFrame for saving.", main_logger=save_logger_name)

            # Use pandas standard textual representation for NA consistently
            na_rep_value = '<NA>'
            # Get authoritative column order and schema definition
            expected_cols = self._get_expected_columns() # Method defining the standard list/order
            schema_dtypes = self._get_schema_dtypes() # Method defining the target dtypes

            # Ensure correct column order and add missing columns if necessary before saving
            missing_cols = [col for col in expected_cols if col not in df_copy.columns]
            if missing_cols:
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Columns missing from DataFrame before save: {missing_cols}. Adding with null values.", main_logger=save_logger_name)
                 for col in missing_cols:
                      target_dtype = schema_dtypes.get(col)
                      if target_dtype == pd.DatetimeTZDtype(tz='UTC'): df_copy[col] = pd.NaT
                      elif target_dtype == pd.Int64Dtype(): df_copy[col] = pd.NA
                      elif target_dtype == pd.BooleanDtype(): df_copy[col] = pd.NA
                      elif target_dtype == pd.Float64Dtype(): df_copy[col] = pd.NA
                      elif target_dtype == pd.StringDtype(): df_copy[col] = pd.NA
                      else: df_copy[col] = None # Or np.nan? None is generally safer for object columns.
                      # Apply the type just in case
                      try:
                           if target_dtype: df_copy[col] = df_copy[col].astype(target_dtype)
                      except Exception: pass # Ignore errors applying type to newly added null column

            # Reindex to ensure exact column order matches expected header
            df_copy = df_copy.reindex(columns=expected_cols)
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - DataFrame columns reindexed to expected order: {expected_cols}", main_logger=save_logger_name)


            # --- Convert Types for Serialization ---
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Preparing data types for CSV serialization...", main_logger=save_logger_name)
            ts_cols = {col for col in self.timestamp_columns if col in df_copy.columns}
            for col in df_copy.columns:
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Processing column '{col}' for save (Dtype: {df_copy[col].dtype})", main_logger=save_logger_name)
                # Convert Timestamps to ISO 8601 UTC string format ('Z' indicates UTC)
                if col in ts_cols:
                     if pd.api.types.is_datetime64_any_dtype(df_copy[col].dtype):
                          # Using fillna(na_rep_value) *after* strftime might be problematic if NaT becomes string 'NaT' first
                          # Better: convert to object, apply strftime, then fill Nones/NaTs
                          df_copy[col] = df_copy[col].astype('object').apply(
                              lambda ts: ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if pd.notna(ts) else na_rep_value
                          )
                          log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Converted timestamp column '{col}' to ISO string.", main_logger=save_logger_name)
                     else: # Column listed but isn't datetime? Convert to string safely.
                          log_statement(loglevel='warning', logstatement=f"{LOG_INS} - WARNING: Column '{col}' listed in timestamp_columns but not datetime type. Converting to string.", main_logger=save_logger_name)
                          # Fill NA before converting to string to ensure NAs become na_rep_value
                          df_copy[col] = df_copy[col].fillna(na_rep_value).astype(str)
                else:
                    # For other columns, ensure NAs are represented by na_rep_value
                    # Convert boolean specifically to avoid 'True'/'False' strings if not desired
                    if isinstance(df_copy[col].dtype, pd.BooleanDtype):
                         df_copy[col] = df_copy[col].map({True: 'true', False: 'false', pd.NA: na_rep_value})
                         log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Converted boolean column '{col}' to string ('true'/'false'/'{na_rep_value}').", main_logger=save_logger_name)
                    # Convert all others to string, ensuring NAs become the chosen representation
                    # Note: This converts numbers to strings as well.
                    else:
                         df_copy[col] = df_copy[col].fillna(na_rep_value).astype(str)
                         log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Converted column '{col}' to string, filling NA with '{na_rep_value}'.", main_logger=save_logger_name)


            # --- Save using stream compression ---
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Writing {len(df_copy)} rows to {target_repo_file_path} using zstd compression.", main_logger=save_logger_name)
            # Use zstandard stream writer with pandas to_csv
            cctx = zstd.ZstdCompressor(level=3) # Use default compression level 3
            with open(target_repo_file_path, 'wb') as ofh:
                with cctx.stream_writer(ofh) as writer:
                    # Use TextIOWrapper to handle encoding for pandas
                    with io.TextIOWrapper(writer, encoding='utf-8', newline='') as text_writer:
                        df_copy.to_csv(
                            text_writer,
                            index=False,       # Don't write row index
                            header=True,       # Write header
                            encoding='utf-8',
                            quoting=csv.QUOTE_MINIMAL, # Quote fields only when necessary (contains delimiter, quote char, or newlines)
                            escapechar='\\',   # Use backslash for escaping quote chars within fields if needed
                            na_rep=na_rep_value # Ensure pandas uses our chosen NA representation
                        )

            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Successfully saved DataFrame to {target_repo_file_path}", main_logger=save_logger_name)
            return True # Indicate success

        except Exception as e:
            log_statement(loglevel='critical', logstatement=f"{LOG_INS} - CRITICAL: Failed to save repository file {target_repo_file_path}: {e}", main_logger=save_logger_name, exc_info=True)

            # --- Optional: Backup Logic ---
            try:
                backup_path = target_repo_file_path.with_suffix(f".save_error_{int(time.time())}.csv.zst")
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Attempting to save backup of failed save to {backup_path}", main_logger=save_logger_name)

                # Use the same saving logic for the backup
                cctx = zstd.ZstdCompressor(level=3)
                with open(backup_path, 'wb') as ofh_bak:
                    with cctx.stream_writer(ofh_bak) as writer_bak:
                        with io.TextIOWrapper(writer_bak, encoding='utf-8', newline='') as text_writer_bak:
                            # Use the already prepared df_copy if available, otherwise df_to_save
                            df_for_backup = df_copy if 'df_copy' in locals() else df_to_save
                            if df_for_backup is not None:
                                # Re-prepare just in case preparation failed earlier? Safer to just save prepared state.
                                df_for_backup.to_csv(
                                    text_writer_bak, index=False, header=True, encoding='utf-8',
                                    quoting=csv.QUOTE_MINIMAL, escapechar='\\', na_rep=na_rep_value
                                )
                                log_statement(loglevel='info', logstatement=f"{LOG_INS} - Backup save attempted to {backup_path}.", main_logger=save_logger_name)
                            else:
                                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Cannot save backup, DataFrame is None.", main_logger=save_logger_name)
            except Exception as backup_e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - ERROR: Failed to save backup to {backup_path}: {backup_e}", main_logger=save_logger_name, exc_info=True)
            # --- End Backup Logic ---

            return False # Indicate failure

    def _load_repo(self) -> pd.DataFrame:
        """
        Loads the repository CSV.ZST file, normalizes columns based on MAIN_REPO_HEADER,
        handles legacy column names, adds missing columns, and applies types from
        REPO_SCHEMA. Timestamps are parsed from ISO 8601 strings into datetime64[ns, UTC].
        Ensures the returned DataFrame strictly adheres to the MAIN_REPO_HEADER schema.
        """
        load_logger_name = __name__ # Use module name for logger
        LOG_INS = f'{load_logger_name}::_load_repo::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        print(f">>> DEBUG: {LOG_INS} - DataRepository._load_repo START for: {self.repo_path}")

        # Check if essential constants were imported successfully (assuming CONSTANTS_AVAILABLE flag is set in __init__)
        if not CONSTANTS_AVAILABLE:
            log_statement(loglevel='critical', logstatement=f"{LOG_INS} - Essential constants not available. Cannot load repository.", main_logger=load_logger_name)
            # Return an empty DataFrame that matches the expected structure as best as possible
            # This avoids downstream errors assuming a DataFrame exists, but functionality is nil.
            try:
                # Attempt to use MAIN_REPO_HEADER even if constants partially failed,
                # otherwise use a minimal default header.
                header = MAIN_REPO_HEADER if 'MAIN_REPO_HEADER' in globals() else ['filepath']
                schema = REPO_SCHEMA if 'REPO_SCHEMA' in globals() else COL_SCHEMA if 'COL_SCHEMA' in globals() else {'filepath': 'string'}
                empty_cols = {col: pd.Series(dtype=schema.get(col, 'string')) for col in header}
                return pd.DataFrame(empty_cols)
            except Exception:
                return pd.DataFrame() # Absolute fallback


        # Use constants for expected structure
        expected_header = MAIN_REPO_HEADER
        target_schema = self.columns_schema # REPO_SCHEMA from constants, set in __init__

        print(f">>> DEBUG: {LOG_INS} - Load Repo - Expected Header: {expected_header}")
        print(f">>> DEBUG: {LOG_INS} - Load Repo - Target Schema: {target_schema}")

        # Create empty DataFrame structure based on expected header and schema
        empty_df_cols = {}
        for col in expected_header:
            dtype_str = target_schema.get(col)
            if dtype_str:
                # Map schema string to actual dtype object for empty DF creation
                try:
                    if dtype_str == 'datetime64[ns, UTC]': dtype = 'datetime64[ns, UTC]'
                    elif dtype_str == 'Int64': dtype = pd.Int64Dtype()
                    elif dtype_str == 'boolean': dtype = pd.BooleanDtype()
                    elif dtype_str == 'Float64': dtype = pd.Float64Dtype()
                    elif dtype_str == 'string': dtype = pd.StringDtype()
                    else: dtype = dtype_str # Assume basic numpy dtype string like 'int32'
                    empty_df_cols[col] = pd.Series(dtype=dtype)
                except TypeError: # Handle potential issues resolving dtype string
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Could not resolve dtype '{dtype_str}' for empty DF column '{col}'. Using object.", main_logger=load_logger_name)
                    empty_df_cols[col] = pd.Series(dtype='object')
            else:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Column '{col}' in MAIN_REPO_HEADER but not in REPO_SCHEMA. Will be loaded as object/string if present.", main_logger=load_logger_name)
                empty_df_cols[col] = pd.Series(dtype='object') # Fallback dtype

        empty_df = pd.DataFrame(empty_df_cols)

        # --- Validate Repo Path ---
        if not self.repo_path or not isinstance(self.repo_path, Path):
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Repository path is invalid or not set. Returning empty DF.", main_logger=load_logger_name)
            return empty_df # Return correctly structured empty DF

        if not self.repo_path.exists():
            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Repository file '{self.repo_path}' not found. Returning empty DF.", main_logger=load_logger_name)
            return empty_df # Return correctly structured empty DF

        # --- Check File Size ---
        try:
            if self.repo_path.stat().st_size < 5: # Check for minimal content (header)
                log_statement(loglevel="warning", logstatement=f"{LOG_INS} - Repo file '{self.repo_path}' appears empty or incomplete. Returning empty DF.", main_logger=load_logger_name)
                return empty_df.astype(target_schema, errors='ignore') # Ensure empty df has schema
        except OSError as stat_e:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - Error accessing repo file stats '{self.repo_path}': {stat_e}. Returning empty DF.", main_logger=load_logger_name)
            return empty_df.astype(target_schema, errors='ignore')

        print(f">>> DEBUG: {LOG_INS} - Load Repo - Attempting to load: {self.repo_path}")
        log_statement(loglevel="info", logstatement=f"{LOG_INS} - Loading data repository: {self.repo_path}", main_logger=load_logger_name)

        # --- Load and Process File ---
        try:
            pdf = None # DataFrame placeholder
            if not ZSTD_AVAILABLE:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Cannot load repository '{self.repo_path}' - zstandard library is required.", main_logger=load_logger_name)
                return empty_df.astype(target_schema, errors='ignore')

            # --- Decompress and Read CSV ---
            dctx = zstd.ZstdDecompressor()
            with open(self.repo_path, 'rb') as ifh:
                try:
                    with dctx.stream_reader(ifh) as reader:
                        # Decode carefully, handle potential large file memory usage if needed
                        content = reader.read().decode('utf-8', errors='replace')
                        buffer = io.StringIO(content)
                        # Read header explicitly to handle potential issues before pandas reads
                        header_line = buffer.readline().strip()
                        actual_header = [h.strip() for h in header_line.split(',')]
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Actual Header Read: {actual_header}")
                        buffer.seek(0) # Reset buffer position for pandas
                        # Load data - keep_default_na=False important for empty strings vs NaN
                        # Use low_memory=False if DtypeWarning occurs, but monitor memory
                        pdf = pd.read_csv(buffer, keep_default_na=False, dtype=str, encoding='utf-8', header=0, low_memory=False)
                except UnicodeDecodeError as ude:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Unicode decode error reading '{self.repo_path}': {ude}. Check file encoding. Returning empty DF.", main_logger=load_logger_name)
                    return empty_df.astype(target_schema, errors='ignore')
                except zstd.ZstdError as zstde:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Zstandard decompression error reading '{self.repo_path}': {zstde}. File might be corrupted or not Zstd. Returning empty DF.", main_logger=load_logger_name)
                    return empty_df.astype(target_schema, errors='ignore')


            if pdf is None or pdf.empty:
                log_statement(loglevel="info", logstatement=f"{LOG_INS} - Repository file '{self.repo_path}' loaded as empty. Returning empty DF.", main_logger=load_logger_name)
                return empty_df.astype(target_schema, errors='ignore')

            print(f">>> DEBUG: {LOG_INS} - Load Repo - CSV loaded {len(pdf)} rows. Normalizing...")
            log_statement(loglevel="debug", logstatement=f"{LOG_INS} - CSV loaded {len(pdf)} rows. Normalizing columns...", main_logger=load_logger_name)

            # --- Column Renaming (Map known legacy names -> current constants) ---
            # Ensure all COL_* constants used here are imported from constants.py
            rename_map = {
                # Map potential legacy names found in older CSVs to current constants
                'Filepath': COL_FILEPATH, 'Filename': COL_FILENAME, 'filepath': COL_FILEPATH,
                'Ext': COL_EXTENSION, 'extension': COL_EXTENSION, 'Filetype': COL_EXTENSION,
                'Size': COL_SIZE, 'size_bytes': COL_SIZE,
                'ModTime': COL_MTIME, 'mtime_ts': COL_MTIME, 'ModificationDate': COL_MTIME, 'last_modified_scan': COL_MTIME,
                'FileCreationTime': COL_CTIME, 'ctime_ts': COL_CTIME,
                # --- FIX: Map ALL relevant hash columns consistently to COL_HASH ---
                'Hash': COL_HASH, 'content_hash': COL_HASH, 'DataHash': COL_HASH, # Map DataHash to COL_HASH
                # --- END FIX ---
                'HashedPathID': COL_HASHED_PATH_ID, 'path_hash': COL_HASHED_PATH_ID,
                'Compressed': COL_COMPRESSED_FLAG, 'is_compressed': COL_COMPRESSED_FLAG,
                'Status': COL_STATUS, 'status': COL_STATUS,
                'ErrorMSG': COL_ERROR, 'error_message': COL_ERROR,
                'Designation': COL_DESIGNATION, 'designation': COL_DESIGNATION,
                'TokenizedPath': COL_TOKENIZED_PATH, 'tokenized_path': COL_TOKENIZED_PATH,
                'ProcessedPath': COL_PROCESSED_PATH, 'processed_path': COL_PROCESSED_PATH,
                'BaseDirectory': BASE_DATA_DIR, 'base_dir': BASE_DATA_DIR, # Ensure BASE_DATA_DIR is the target constant
                'IsCopy': COL_IS_COPY_FLAG, 'is_copy': COL_IS_COPY_FLAG,
                'last_updated_repo': COL_LAST_UPDATED, 'last_updated_ts': COL_LAST_UPDATED,
                'AccessedDate': 'accessed_ts' # Define constant if needed, currently ignored
            }
            # Filter map to only include columns actually present in the loaded CSV
            # AND ensure the target name is different from the source name
            actual_rename_map = {old: new for old, new in rename_map.items() if old in pdf.columns and old != new}

            # --- IMPORTANT: Pre-Rename Check for Duplicate Targets ---
            # Check if multiple columns in pdf would be renamed to the *same* target constant
            target_counts = collections.Counter(actual_rename_map.values())
            duplicate_targets = {target: count for target, count in target_counts.items() if count > 1}
            if duplicate_targets:
                problematic_renames = {old: new for old, new in actual_rename_map.items() if new in duplicate_targets}
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Rename conflict! Multiple columns map to same target: {problematic_renames}. Cannot reliably rename.", main_logger=load_logger_name)
                # Decide policy: skip rename? raise error? For now, skip problematic renames.
                actual_rename_map = {old: new for old, new in actual_rename_map.items() if new not in duplicate_targets}
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Proceeding with non-conflicting renames only: {actual_rename_map}", main_logger=load_logger_name)
            # --- END PRE-RENAME CHECK ---


            if actual_rename_map:
                print(f">>> DEBUG: {LOG_INS} - Load Repo - Applying Renames: {actual_rename_map}")
                try:
                    pdf.rename(columns=actual_rename_map, inplace=True, errors='raise')
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - Columns after rename: {pdf.columns.tolist()}")
                except Exception as rename_err:
                    # More specific error handling possible (e.g., check for duplicate targets)
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - ERROR during rename: {rename_err}. Proceeding cautiously.")
                    log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error during column rename: {rename_err}. Check rename_map and CSV header: {actual_header}. Potential duplicate target columns?.", main_logger=load_logger_name)
                    # Decide recovery strategy: Stop? Drop conflicting legacy cols? Continue?
                    # Continuing assumes the target constant name might already exist correctly.

            # --- Create final_df based on MAIN_REPO_HEADER ---
            # Initialize with the correct index from the loaded data
            final_df = pd.DataFrame(index=pdf.index)
            added_cols_info = {}
            processed_cols_from_pdf = set() # Track which expected columns were found

            print(f">>> DEBUG: {LOG_INS} - Load Repo - Building final_df using Expected Header: {expected_header}")
            for col_const in expected_header:
                target_dtype_str = target_schema.get(col_const) # Get schema type string
                # Default to string/object if column is in header but not schema
                if target_dtype_str is None: target_dtype_str = 'string'

                if col_const in pdf.columns:
                    # Column exists in the (potentially renamed) pdf
                    pdf_col_data = pdf[col_const]
                    if isinstance(pdf_col_data, pd.DataFrame):
                        # Handle duplicate column names in source CSV - take first one
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - WARNING: Duplicate column '{col_const}' detected in source CSV. Using first instance.")
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Duplicate column '{col_const}' detected in source CSV '{self.repo_path}'. Using first instance.", main_logger=load_logger_name)
                        final_df[col_const] = pdf_col_data.iloc[:, 0]
                    else:
                        final_df[col_const] = pdf_col_data # Copy Series
                    processed_cols_from_pdf.add(col_const)
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - Copied existing column '{col_const}'")
                else:
                    # Column missing from pdf, add it with appropriate null type based on schema string
                    added_cols_info[col_const] = target_dtype_str
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - Adding missing column '{col_const}' with type {target_dtype_str}")
                    if target_dtype_str == 'datetime64[ns, UTC]':
                        final_df[col_const] = pd.Series(pd.NaT, index=pdf.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'Int64':
                        final_df[col_const] = pd.Series(pd.NA, index=pdf.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'boolean': # Pandas nullable boolean
                        final_df[col_const] = pd.Series(pd.NA, index=pdf.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'Float64': # Pandas nullable float
                        final_df[col_const] = pd.Series(pd.NA, index=pdf.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'string': # Pandas nullable string
                        final_df[col_const] = pd.Series(pd.NA, index=pdf.index, dtype=target_dtype_str)
                    else: # Fallback for other types (e.g., basic int32, float64, object)
                        try:
                            # Try creating with the specified type and numpy NaN default
                            final_df[col_const] = pd.Series([np.nan] * len(pdf), index=pdf.index, dtype=target_dtype_str)
                        except (TypeError, ValueError): # If dtype doesn't support NaN, use object
                            final_df[col_const] = pd.Series([None] * len(pdf), index=pdf.index, dtype='object')

            if added_cols_info:
                added_str = ", ".join([f"'{c}' ({t})" for c, t in added_cols_info.items()])
                print(f">>> DEBUG: {LOG_INS} - Load Repo - Added missing columns: {added_str}")
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Columns added/normalized for '{self.repo_path}': {added_str}", main_logger=load_logger_name)

            # --- Apply Type Conversions using target_schema ---
            print(f">>> DEBUG: {LOG_INS} - Load Repo - Applying type conversions to final_df...")
            for col_const in final_df.columns: # Iterate over columns actually in final_df
                target_dtype_str = target_schema.get(col_const) # Get the string representation from schema dict
                if target_dtype_str is None:
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - No target type in schema for '{col_const}'. Skipping conversion.")
                    continue # Skip conversion if column isn't in our defined schema

                current_series = final_df[col_const]
                print(f">>> DEBUG: {LOG_INS} - Load Repo - Converting '{col_const}' (current: {current_series.dtype}) to target: {target_dtype_str}...")
                try:
                    # Standardized Timestamp Conversion (using TIMESTAMP_COLUMNS constant from schema)
                    if col_const in self.timestamp_columns:
                        # Parse ISO strings (or other formats pandas recognizes) into UTC datetime
                        converted_dt = pd.to_datetime(current_series, errors='coerce', utc=True)
                        # Ensure the target column has the correct final dtype
                        final_df[col_const] = converted_dt.astype('datetime64[ns, UTC]')
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (Timestamp) -> {final_df[col_const].dtype}")

                    # Nullable Integer Conversion
                    elif target_dtype_str == 'Int64':
                        # Convert to numeric, coerce errors, use pandas Float64 then Int64 for null support
                        numeric_series = pd.to_numeric(current_series, errors='coerce')
                        final_df[col_const] = numeric_series.astype('Float64').astype('Int64')
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (Int64) -> {final_df[col_const].dtype}")

                    # String Conversion (use pandas StringDtype for nullable strings)
                    elif target_dtype_str == 'string': # Check against schema type 'string'
                        # Fill NA/NaN with pd.NA BEFORE converting to StringDtype
                        final_df[col_const] = current_series.fillna(pd.NA).astype('string')
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (String) -> {final_df[col_const].dtype}")

                    # Float Conversion (use pandas Float64 for nullable floats)
                    elif target_dtype_str == 'Float64':
                        numeric_series = pd.to_numeric(current_series, errors='coerce')
                        final_df[col_const] = numeric_series.astype('Float64')
                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (Float64) -> {final_df[col_const].dtype}")

                    # Boolean Conversion (use pandas BooleanDtype)
                    elif target_dtype_str == 'boolean':
                        # Map common string representations ('Y'/'N', 'True'/'False')
                        # Note: pd.NA should be preserved if already present
                        bool_map = {'true': True, 'yes': True, 'y': True, '1': True,
                                    'false': False, 'no': False, 'n': False, '0': False,
                                    '': pd.NA, None: pd.NA} # Map empty string/None to NA
                        # Apply map robustly, keep NA, convert others to bool, default non-matches to NA
                        if pd.api.types.is_string_dtype(current_series.dtype) or current_series.dtype == 'object':
                            lower_series = current_series.astype(str).str.lower().fillna('') # Fill NaN before lower
                            converted_bool = lower_series.map(bool_map)
                            # If original was already bool/NA, keep it, else default non-matches to NA
                            final_df[col_const] = converted_bool.astype('boolean')
                        elif pd.api.types.is_bool_dtype(current_series.dtype): # If already bool, just ensure correct type
                            final_df[col_const] = current_series.astype('boolean')
                        elif pd.api.types.is_numeric_dtype(current_series.dtype): # Handle numeric inputs
                            converted_bool = current_series.map({1: True, 0: False}).fillna(pd.NA)
                            final_df[col_const] = converted_bool.astype('boolean')
                        else: # Fallback for unexpected types
                            final_df[col_const] = pd.Series([pd.NA] * len(current_series), index=current_series.index, dtype='boolean')

                        print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (Boolean) -> {final_df[col_const].dtype}")

                    # Direct astype for other specific dtypes if needed (e.g., 'int32')
                    # Ensure this handles errors gracefully if target_dtype_str is invalid
                    else:
                        if current_series.dtype != target_dtype_str:
                            # Handle potential errors during direct astype
                            try:
                                final_df[col_const] = current_series.astype(target_dtype_str)
                                print(f">>> DEBUG: {LOG_INS} - Load Repo - Converted '{col_const}' (Direct to {target_dtype_str}) -> {final_df[col_const].dtype}")
                            except Exception as direct_astype_e:
                                print(f">>> DEBUG: {LOG_INS} - Load Repo - ERROR Direct Converting '{col_const}' to {target_dtype_str}: {direct_astype_e}. Falling back to object.")
                                log_statement(loglevel="error", logstatement=f"{LOG_INS} - Error direct converting column '{col_const}' to '{target_dtype_str}' in '{self.repo_path}': {direct_astype_e}. Falling back to object.", main_logger=load_logger_name, exc_info=False)
                                final_df[col_const] = final_df[col_const].astype('object') # Fallback to object

                except Exception as e:
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - ERROR Converting '{col_const}' to {target_dtype_str}: {e}. Falling back to string/object.")
                    log_statement(loglevel="error", logstatement=f"{LOG_INS} - Error converting column '{col_const}' to '{target_dtype_str}' in '{self.repo_path}': {e}. Falling back to string/object.", main_logger=load_logger_name, exc_info=False)
                    # Fallback: convert problematic column to pandas string type or object, filling NAs
                    try:
                        final_df[col_const] = final_df[col_const].astype('string') # Try nullable string first
                    except Exception:
                        final_df[col_const] = final_df[col_const].astype('object').fillna('') # Fallback to object

            # Ensure all expected columns from MAIN_REPO_HEADER are present *before* final reindex
            missing_cols = [col for col in expected_header if col not in final_df.columns]
            if missing_cols:
                missing_str = ", ".join([f"'{c}'" for c in missing_cols])
                print(f">>> DEBUG: {LOG_INS} - Load Repo - Adding completely missing columns from header: {missing_str}")
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Adding missing header columns for '{self.repo_path}': {missing_str}", main_logger=load_logger_name)
                for col in missing_cols:
                    # Add missing columns with appropriate null type based on REPO_SCHEMA
                    target_dtype_str = target_schema.get(col, 'string') # Default to string if not in schema
                    print(f">>> DEBUG: {LOG_INS} - Load Repo - Adding missing column '{col}' with target type {target_dtype_str}")
                    if target_dtype_str == 'datetime64[ns, UTC]':
                        final_df[col] = pd.Series(pd.NaT, index=final_df.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'Int64':
                        final_df[col] = pd.Series(pd.NA, index=final_df.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'boolean':
                        final_df[col] = pd.Series(pd.NA, index=final_df.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'Float64':
                        final_df[col] = pd.Series(pd.NA, index=final_df.index, dtype=target_dtype_str)
                    elif target_dtype_str == 'string':
                        final_df[col] = pd.Series(pd.NA, index=final_df.index, dtype=target_dtype_str)
                    else: # Fallback for other types (e.g., object, specific int/float)
                        try:
                            final_df[col] = pd.Series(pd.NA, index=final_df.index, dtype=target_dtype_str)
                        except (TypeError, ValueError): # Handle dtypes that don't support pd.NA well
                            final_df[col] = pd.Series([None] * len(final_df), index=final_df.index, dtype='object')

            # Ensure final column order matches MAIN_REPO_HEADER exactly
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Reindexing final DataFrame to expected header order.", main_logger=load_logger_name)
            try:
                if len(set(expected_header)) != len(expected_header):
                    duplicates = [item for item, count in collections.Counter(expected_header).items() if count > 1]
                    log_statement(loglevel='critical', logstatement=f"{LOG_INS} - Duplicate columns found in MAIN_REPO_HEADER constant: {duplicates}! Cannot reliably reindex.", main_logger=load_logger_name)
                    raise ValueError(f"Duplicate columns detected in MAIN_REPO_HEADER constant: {duplicates}")

                # Perform the reindex
                final_df = final_df.reindex(columns=expected_header) # This adds missing cols with NaN/NaT/NA automatically
                log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Re-applying schema types after final reindex...", main_logger=load_logger_name)
                for col, dtype_str in target_schema.items():
                    if col in final_df.columns:
                        try:
                            current_dtype = final_df[col].dtype
                            # Only apply if type doesn't match target schema string representation
                            # (This is a basic check, might need refinement for complex types)
                            # Comparison might be tricky (e.g., str vs np.dtype('O'))
                            # A safer bet is often to just re-apply the conversion logic used earlier
                            if dtype_str == 'datetime64[ns, UTC]' and current_dtype != 'datetime64[ns, UTC]':
                                final_df[col] = pd.to_datetime(final_df[col], errors='coerce', utc=True)
                            elif dtype_str == 'Int64' and not isinstance(current_dtype, pd.Int64Dtype):
                                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('Float64').astype('Int64')
                            elif dtype_str == 'boolean' and not isinstance(current_dtype, pd.BooleanDtype):
                                # Re-apply boolean conversion logic if needed
                                bool_map = {'true': True, 'yes': True, 'y': True, '1': True,'false': False, 'no': False, 'n': False, '0': False,'': pd.NA, None: pd.NA}
                                lower_series = final_df[col].astype(str).str.lower().fillna('')
                                converted_bool = lower_series.map(bool_map)
                                final_df[col] = converted_bool.astype('boolean')
                            elif dtype_str == 'Float64' and not isinstance(current_dtype, pd.Float64Dtype):
                                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').astype('Float64')
                            elif dtype_str == 'string' and not isinstance(current_dtype, pd.StringDtype):
                                final_df[col] = final_df[col].astype('string') # Use nullable string
                            # Add other necessary type checks/conversions here...

                        except Exception as type_apply_err:
                            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Error re-applying type '{dtype_str}' to column '{col}' after reindex: {type_apply_err}", main_logger=load_logger_name)

            except Exception as reindex_err:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error during final reindex for '{self.repo_path}': {reindex_err}. Returning potentially incomplete DataFrame.", main_logger=load_logger_name, exc_info=True)
                # Decide policy: raise error or return partially correct DF? Returning for now.
                # Note: This might still lead to KeyErrors downstream if columns are critically missing.
            except ValueError as ve:
                log_statement('error', f"{LOG_INS}:ValueError>>Value error: {ve}", __file__)
                raise ve

            print(f">>> DEBUG: {LOG_INS} - Load Repo - Final DF loaded. Length: {len(final_df)}, Columns: {final_df.columns.tolist()}")
            log_statement(loglevel="info", logstatement=f"{LOG_INS} - Repository loaded and processed ({len(final_df)} entries) from '{self.repo_path}'.", main_logger=load_logger_name)
            return final_df

        # --- Error Handling for Load Process ---
        except pd.errors.EmptyDataError:
            log_statement(loglevel="warning", logstatement=f"{LOG_INS} - Repository file '{self.repo_path}' contains no data or only header. Returning empty DF with schema.", main_logger=load_logger_name)
            return empty_df.astype(target_schema, errors='ignore') # Return structured empty DF
        except Exception as e:
            print(f">>> DEBUG: {LOG_INS} - Load Repo - CRITICAL ERROR loading {self.repo_path}: {type(e).__name__}: {e}")
            log_statement(loglevel="critical", logstatement=f"{LOG_INS} - Repository load failed for '{self.repo_path}': {e}. Returning empty DF with schema.", main_logger=load_logger_name, exc_info=True)
            return empty_df.astype(target_schema, errors='ignore') # Return structured empty DF

    def _load_repository_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the hierarchical repository index from the JSON file defined by
        constants.INDEX_FILE. Ensures DATA_REPO_DIR exists. Validates entries
        against expected keys (INDEX_KEY_* constants).
        """
        LOG_INS = f'{__name__}::_load_repository_index::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        load_logger_name = __name__

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot load index.", main_logger=load_logger_name)
             return {}

        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Ensuring directory exists: {DATA_REPO_DIR}", main_logger=load_logger_name)
        try:
            DATA_REPO_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as dir_e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Failed to create repository directory '{DATA_REPO_DIR}': {dir_e}. Cannot load index.", main_logger=load_logger_name)
            return {}

        if not INDEX_FILE.exists():
            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Repository index file '{INDEX_FILE}' not found. Initializing empty index.", main_logger=load_logger_name)
            return {}

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Loading repository index from: {INDEX_FILE}", main_logger=load_logger_name)
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                 index_data = json.load(f)

            loaded_index: Dict[str, Dict[str, Any]] = {}
            if not isinstance(index_data, dict):
                log_statement(loglevel='error', logstatement=f"{LOG_INS} - Repository index file '{INDEX_FILE}' is not a valid JSON dictionary. Reinitializing.", main_logger=load_logger_name)
                return {}

            # Process entries, validating structure using constants
            for repo_hash, entry in index_data.items():
                try:
                    if isinstance(entry, dict) and INDEX_KEY_PATH in entry:
                        # Validate required fields and types
                        path_str = entry[INDEX_KEY_PATH]
                        metadata = entry.get(INDEX_KEY_METADATA, {})
                        children = entry.get(INDEX_KEY_CHILDREN, [])

                        if not isinstance(path_str, str): raise TypeError("Path must be a string.")
                        if not isinstance(metadata, dict): metadata = {} # Allow empty/invalid metadata
                        if not isinstance(children, list): children = [] # Allow empty/invalid children list

                        path_obj = Path(path_str) # Convert string path to Path object
                        loaded_index[repo_hash] = {
                            INDEX_KEY_PATH: path_obj,
                            INDEX_KEY_METADATA: metadata,
                            INDEX_KEY_CHILDREN: children
                        }
                    elif isinstance(entry, str): # Handle legacy format (just path string)
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Found legacy format entry for hash {repo_hash}. Converting.", main_logger=load_logger_name)
                        path_obj = Path(entry)
                        loaded_index[repo_hash] = {
                            INDEX_KEY_PATH: path_obj,
                            INDEX_KEY_METADATA: {}, # Default empty metadata
                            INDEX_KEY_CHILDREN: []  # Default empty children
                        }
                    else:
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Skipping invalid entry in index file for hash {repo_hash}: Format incorrect.", main_logger=load_logger_name)
                except Exception as entry_e:
                     log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error processing index entry for hash {repo_hash}: {entry_e}", main_logger=load_logger_name)

            log_statement(loglevel='info', logstatement=f"{LOG_INS} - Loaded repository index from '{INDEX_FILE}' with {len(loaded_index)} valid entries.", main_logger=load_logger_name)
            return loaded_index

        except json.JSONDecodeError as json_e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Failed to decode JSON repository index '{INDEX_FILE}': {json_e}. Returning empty index.", main_logger=load_logger_name, exc_info=True)
            return {}
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Failed to load repository index '{INDEX_FILE}': {e}. Returning empty index.", main_logger=load_logger_name, exc_info=True)
            return {}

    def _save_repository_index(self, index_data: Dict[str, Dict[str, Any]]):
        """
        Saves the hierarchical repository index to the JSON file defined by
        constants.INDEX_FILE. Recalculates parent/child relationships.
        Ensures paths are stored as resolved strings. Uses INDEX_KEY_* constants.

        Args:
            index_data (Dict[str, Dict[str, Any]]): The repository index data structure.
                                        Expected structure: {hash: {INDEX_KEY_PATH: Path, ...}}
        """
        LOG_INS = f'{__name__}::_save_repository_index::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        save_logger_name = __name__

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot save index.", main_logger=save_logger_name)
             return
        if self.repo_index is None:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Repository index not loaded--nothing to save.", __file__)
            return
        if not self.repo_index_file:
            log_statement('error', f"{LOG_INS}:ERROR>>repo_index_file path is not set.  Cannot save index.  Current path set: {self.repo_index_file}", __file__)
            return
        if not self.repo_index_dirty:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Repository index not marked dirty.  Skipping save.", __file__)
            return
        
        # Ensure target directory exists
        try:
            DATA_REPO_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as dir_e:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Failed to create repository directory '{DATA_REPO_DIR}': {dir_e}. Cannot save index.", main_logger=save_logger_name)
             return

        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Preparing to save repository index with {len(index_data)} entries.", main_logger=save_logger_name)

        # --- Recalculate Parent/Child Relationships ---
        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Recalculating parent/child relationships.", main_logger=save_logger_name)
        try:
            # Create mappings: resolved path string -> hash and hash -> resolved Path object
            path_to_hash: Dict[str, str] = {}
            hash_to_path: Dict[str, Path] = {}
            valid_entries: Dict[str, Dict[str, Any]] = {} # Filter out invalid entries before processing

            for h, entry in index_data.items():
                if isinstance(entry, dict) and INDEX_KEY_PATH in entry and isinstance(entry[INDEX_KEY_PATH], Path):
                    resolved_path = entry[INDEX_KEY_PATH].resolve()
                    path_str = str(resolved_path)
                    path_to_hash[path_str] = h
                    hash_to_path[h] = resolved_path
                    valid_entries[h] = entry # Keep valid entry for saving
                    # Reset children list for recalculation
                    valid_entries[h][INDEX_KEY_CHILDREN] = []
                else:
                     log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Skipping invalid entry during relationship calculation for hash {h}.", main_logger=save_logger_name)

            # Determine children by comparing paths
            # Sort by path depth (descending) to process children before potential parents
            sorted_hashes = sorted(valid_entries.keys(), key=lambda h: len(hash_to_path[h].parts), reverse=True)

            for h_child in sorted_hashes:
                child_path = hash_to_path[h_child]
                child_parent_path = child_path.parent # Get parent path

                # Find potential parent hash using the resolved parent path string
                parent_path_str = str(child_parent_path)
                if parent_path_str in path_to_hash:
                    h_parent = path_to_hash[parent_path_str]
                    # Ensure parent is in our valid list and not the same hash
                    if h_parent in valid_entries and h_parent != h_child:
                         if h_child not in valid_entries[h_parent][INDEX_KEY_CHILDREN]:
                             valid_entries[h_parent][INDEX_KEY_CHILDREN].append(h_child)
                         # No need to break, relationship found

        except Exception as rel_e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error recalculating parent/child relationships: {rel_e}. Relationships might be incomplete.", main_logger=save_logger_name, exc_info=True)
            # Continue saving with potentially incorrect relationships or stop? Decide policy.
            # For now, continue using valid_entries which have children lists reset.

        # --- Serialize for JSON ---
        serializable_data = {}
        for repo_hash, entry in valid_entries.items(): # Use the filtered/processed valid entries
            try:
                 serializable_data[repo_hash] = {
                     INDEX_KEY_PATH: str(entry[INDEX_KEY_PATH].resolve()), # Store resolved path string
                     INDEX_KEY_METADATA: entry.get(INDEX_KEY_METADATA, {}),
                     INDEX_KEY_CHILDREN: sorted(entry.get(INDEX_KEY_CHILDREN, [])) # Sort children for consistency
                 }
            except Exception as serialize_e:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error serializing entry for hash {repo_hash}: {serialize_e}", main_logger=save_logger_name)


        # --- Save to JSON ---
        log_statement(loglevel='info', logstatement=f"Saving repository index to '{INDEX_FILE}' with {len(serializable_data)} entries to {self.repo_index_file}.", main_logger=save_logger_name)
        try:
            self.repo_index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4, sort_keys=True) # Use indent and sort keys
                self.repo_index_dirty = False
            log_statement(loglevel='info', logstatement=f"Successfully saved repository index to '{INDEX_FILE}'.", main_logger=save_logger_name)
        except TypeError as type_e:
            log_statement(loglevel='critical', logstatement=f"Failed to save repository index '{INDEX_FILE}' due to non-serializable data: {type_e}. Check metadata contents.", main_logger=save_logger_name, exc_info=True)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed to save repository index '{INDEX_FILE}': {e}", main_logger=save_logger_name, exc_info=True)

    # --- MODIFIED _find_sub_repositories Method ---
    def _find_sub_repositories(self, target_path_obj: Path, repo_index: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Path, Path]]:
        """
        Identifies existing repositories (from the index) that manage directories
        within the specified target_path_obj. Uses constants for file naming.

        Args:
            target_path_obj (Path): The path to check for sub-repositories within.
            repo_index (Dict[str, Dict[str, Any]]): The loaded repository index.

        Returns:
            List[Tuple[str, Path, Path]]: List of tuples containing
                                         (hash, path_obj, repo_file_path) for found sub-repositories.
        """
        LOG_INS = f'{__name__}::_find_sub_repositories::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        find_logger_name = __name__
        sub_repos = []

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot find sub-repositories.", main_logger=find_logger_name)
             return []

        try:
            target_path_res = target_path_obj.resolve()
        except Exception as res_e:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Could not resolve target path '{target_path_obj}': {res_e}. Cannot find sub-repositories.", main_logger=find_logger_name)
             return []

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Checking {len(repo_index)} index entries for sub-paths of '{target_path_res}'...", main_logger=find_logger_name)

        for repo_hash, index_entry in repo_index.items():
            try:
                # Check structure using constants
                if not isinstance(index_entry, dict) or INDEX_KEY_PATH not in index_entry:
                     log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Skipping invalid index entry for hash {repo_hash} during sub-repo check.", main_logger=find_logger_name)
                     continue

                existing_path_obj = index_entry[INDEX_KEY_PATH] # Should be a Path object from _load_repository_index
                if not isinstance(existing_path_obj, Path):
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Index entry for hash {repo_hash} has non-Path object for path '{existing_path_obj}'. Skipping.", main_logger=find_logger_name)
                    continue

                existing_path_res = existing_path_obj.resolve()

                # Check if existing_path_res is a subdirectory of target_path_res
                # Ensure they are not the same path.
                if target_path_res != existing_path_res and target_path_res in existing_path_res.parents:
                    # Construct repo filename using helper and constants
                    repo_filename = get_repo_filename(repo_hash)
                    if repo_filename.exists():
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Found potential sub-repository: hash={repo_hash}, path='{existing_path_res}', file='{repo_filename}'", main_logger=find_logger_name)
                        sub_repos.append((repo_hash, existing_path_res, repo_filename))
                    else:
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Index points to existing path '{existing_path_res}' but repo file '{repo_filename}' is missing. Skipping.", main_logger=find_logger_name)
            except OSError as path_e: # Catch errors during .resolve() or .parents
                 log_statement(loglevel='error', logstatement=f"{LOG_INS} - OS Error processing index entry hash {repo_hash} path '{index_entry.get(INDEX_KEY_PATH, 'N/A')}': {path_e}", main_logger=find_logger_name)
            except Exception as e:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS} - Unexpected error processing index entry hash {repo_hash}: {e}", main_logger=find_logger_name, exc_info=True)

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Found {len(sub_repos)} potential sub-repositories for '{target_path_res}'.", main_logger=find_logger_name)
        return sub_repos

    # --- MODIFIED _validate_sub_repository Method ---
    def _validate_sub_repository(self, repo_file: Path, original_path: Path, num_check: int = 10) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Validates a sub-repository by loading it via DataRepository, checking its schema
        against MAIN_REPO_HEADER, and comparing metadata (using constants and standardized
        timestamp parsing) for a sample of files against current filesystem state.

        Args:
            repo_file (Path): The path to the sub-repository .csv.zst file.
            original_path (Path): The original directory path associated with this sub-repo.
            num_check (int): Number of random files to check for validation.

        Returns:
            Tuple[bool, Optional[pd.DataFrame]]: (is_valid, loaded_dataframe if valid else None)
        """
        LOG_INS = f'{__name__}::_validate_sub_repository::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        validate_logger_name = __name__
        max_workers = _get_max_workers() # Use helper from utils

        if not CONSTANTS_AVAILABLE:
             log_statement(loglevel='error', logstatement=f"{LOG_INS} - Constants not available. Cannot validate repository.", main_logger=validate_logger_name)
             return False, None

        log_statement(loglevel='info', logstatement=f"{LOG_INS} - Validating sub-repository: '{repo_file}' (Path: '{original_path}', Workers: {max_workers})", main_logger=validate_logger_name)

        try:
            # Instantiate DataRepository for the sub-repo file
            # This implicitly uses the standardized _load_repo method
            # Pass the original path as the target_dir for context if needed,
            # but primarily use repo_file to load the specific data.
            # Ensure use_gpu matches the parent repo's setting or is False.
            repo = DataRepository(repo_path=repo_file, target_dir=original_path, use_gpu=self.use_gpu)
            df = repo.df # Access the loaded and normalized DataFrame

            # Check if loading failed or DF is empty
            if df is None or df.empty:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Sub-repository '{repo_file}' loaded empty or failed to load. Invalid.", main_logger=validate_logger_name)
                return False, None

            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Loaded '{repo_file}' with {len(df)} entries for validation.", main_logger=validate_logger_name)

            # --- Schema Validation ---
            required_cols = set(MAIN_REPO_HEADER)
            actual_cols = set(df.columns)
            if not required_cols.issubset(actual_cols):
                missing = required_cols - actual_cols
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Sub-repository '{repo_file}' missing required columns: {missing}. Invalid.", main_logger=validate_logger_name)
                return False, None
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Column schema check passed for '{repo_file}'.", main_logger=validate_logger_name)

            # --- Directory Existence Check ---
            # Check if the directory the sub-repo *claims* to manage still exists
            if not original_path.is_dir():
                 log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Original directory '{original_path}' for sub-repo '{repo_file}' does not exist. Invalid.", main_logger=validate_logger_name)
                 return False, None

            # --- Sample File Validation ---
            num_entries = len(df)
            if num_entries == 0:
                 log_statement(loglevel='info', logstatement=f"{LOG_INS} - Sub-repository '{repo_file}' is empty but schema is valid. Considered valid.", main_logger=validate_logger_name)
                 return True, df # Empty repo is valid if schema matches

            num_to_check = min(num_check, num_entries)
            if num_to_check <= 0:
                 log_statement(loglevel='info', logstatement=f"{LOG_INS} - No samples requested ({num_check=}). Skipping file validation for '{repo_file}'.", main_logger=validate_logger_name)
                 return True, df # Skip sample check if num_check is 0 or less

            # Select random samples using COL_FILEPATH as the reference
            sample_indices = random.sample(range(num_entries), num_to_check)
            # Use .iloc for positional indexing, ensure COL_FILEPATH exists (checked above)
            files_to_check_series = df.iloc[sample_indices]

            mismatches = 0
            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Checking {num_to_check} sample files from '{repo_file}' using {max_workers} workers.", main_logger=validate_logger_name)

            # Prepare data for parallel processing
            path_to_stored_row = {}
            paths_for_metadata = []
            for index, stored_row in files_to_check_series.iterrows():
                 try:
                     path_str = stored_row[COL_FILEPATH]
                     current_path = Path(path_str)
                     path_to_stored_row[current_path] = stored_row # Map Path object to row Series
                     paths_for_metadata.append(current_path)
                 except Exception as path_err:
                     log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Error creating Path object from stored path '{stored_row.get(COL_FILEPATH)}': {path_err}. Skipping sample.", main_logger=validate_logger_name)
                     mismatches += 1 # Count as mismatch if path is invalid

            # Run _get_file_metadata in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_get_file_metadata, path): path for path in paths_for_metadata}

                for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc=f"Validating {repo_file.name}", leave=False, unit="file"):
                    current_path = future_to_path[future]
                    stored_row = path_to_stored_row[current_path]
                    try:
                        current_metadata = future.result() # This now returns ISO strings for timestamps

                        if current_metadata is None:
                            # File might have been deleted or become inaccessible
                            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Validation: Could not get metadata for '{current_path}' (exists: {current_path.exists()}). Mismatch.", main_logger=validate_logger_name)
                            mismatches += 1
                            continue

                        # --- Comparison using Constants and Standardized Types ---
                        mismatch_details = []

                        # Size (Integer comparison)
                        try:
                            stored_size = int(stored_row[COL_SIZE])
                            current_size = int(current_metadata[COL_SIZE])
                            if current_size != stored_size:
                                mismatch_details.append(f"Size ({current_size} vs {stored_size})")
                        except (ValueError, TypeError): mismatch_details.append("Size (conversion error)")

                        # MTime (Timestamp comparison - parse ISO strings)
                        try:
                            # Parse ISO strings into datetime objects for comparison
                            current_mtime_dt = pd.to_datetime(current_metadata[COL_MTIME], errors='coerce', utc=True)
                            stored_mtime_dt = pd.to_datetime(stored_row[COL_MTIME], errors='coerce', utc=True)

                            if pd.isna(current_mtime_dt) or pd.isna(stored_mtime_dt):
                                 if not (pd.isna(current_mtime_dt) and pd.isna(stored_mtime_dt)):
                                     mismatch_details.append("MTime (one is NaT)")
                            # Allow small tolerance (e.g., 1 second)
                            elif abs((current_mtime_dt - stored_mtime_dt).total_seconds()) > 1:
                                 mismatch_details.append(f"MTime ({current_metadata[COL_MTIME]} vs {stored_row[COL_MTIME]})")
                        except Exception: mismatch_details.append("MTime (parsing/comparison error)")

                        # Hash (String comparison)
                        if current_metadata[COL_HASH] != stored_row[COL_HASH]:
                            mismatch_details.append(f"Hash ({current_metadata[COL_HASH][:8]}... vs {stored_row[COL_HASH][:8]}...)")

                        if mismatch_details:
                            log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Validation mismatch for '{current_path}': {'; '.join(mismatch_details)}", main_logger=validate_logger_name)
                            mismatches += 1

                    except Exception as exc:
                        log_statement(loglevel='error', logstatement=f'{LOG_INS} - Validation check exception for path {current_path}: {exc}', main_logger=validate_logger_name, exc_info=True)
                        mismatches += 1 # Count exceptions as mismatches

            # --- Final Verdict ---
            if mismatches == 0:
                log_statement(loglevel='info', logstatement=f"{LOG_INS} - Validation successful for '{repo_file}' ({num_to_check} files checked).", main_logger=validate_logger_name)
                return True, df # Return the loaded, normalized DataFrame
            else:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Validation failed for '{repo_file}' ({mismatches}/{num_to_check} mismatches in sample check). Invalid.", main_logger=validate_logger_name)
                return False, None

        except ImportError: # Handle case where DataRepository couldn't be imported (should not happen if validation runs from within the class)
            log_statement(loglevel='critical', logstatement=f"{LOG_INS} - DataRepository class seems unavailable during validation of '{repo_file}'.", main_logger=validate_logger_name)
            return False, None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error during validation of '{repo_file}': {e}", main_logger=validate_logger_name, exc_info=True)
            return False, None

    # --- get_status Method (Using Constants) ---
    def get_status(self, source_filepath: Path) -> Optional[str]:
        """Gets the current status of a file using COL_FILEPATH and COL_STATUS."""
        LOG_INS = f'{__name__}::get_status::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        get_logger_name = __name__

        if not CONSTANTS_AVAILABLE: return None
        if self.df is None:
            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - DataFrame not loaded. Cannot get status.", main_logger=get_logger_name)
            return None
        if not self.lock:
             log_statement(loglevel="error", logstatement=f"{LOG_INS} - Repository lock not initialized. Cannot get status.", main_logger=get_logger_name)
             return None # Or raise error?

        try:
            source_filepath_str = str(source_filepath.resolve())
        except Exception as res_e:
             log_statement(loglevel="error", logstatement=f"{LOG_INS} - Could not resolve input path '{source_filepath}': {res_e}", main_logger=get_logger_name)
             return None

        with self.lock:
             try:
                 # Check if required columns exist
                 if COL_FILEPATH not in self.df.columns or COL_STATUS not in self.df.columns:
                      log_statement(loglevel='error', logstatement=f"{LOG_INS} - Required columns ({COL_FILEPATH}, {COL_STATUS}) missing from DataFrame. Cannot get status.", main_logger=get_logger_name)
                      return None

                 # Use pandas for filtering
                 entry = self.df[self.df[COL_FILEPATH] == source_filepath_str]

                 if not entry.empty:
                     # Access status using the constant, handle potential multiple entries (shouldn't happen if filepath is unique key)
                     status = entry[COL_STATUS].iloc[0]
                     return status if pd.notna(status) else None # Return None if status is NA
                 else:
                     return None # File not found in repository
             except Exception as e:
                  log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error accessing DataFrame to get status for '{source_filepath_str}': {e}", main_logger=get_logger_name, exc_info=True)
                  return None


    # --- get_files_by_status Method (Using Constants) ---
    def get_files_by_status(self, status: Union[str, List[str]], base_dir: Optional[Path] = None) -> List[Path]:
        """
        Gets absolute source file paths (as Path objects) by status, optionally
        filtered by base_dir. Uses COL_STATUS and COL_FILEPATH constants.
        """
        LOG_INS = f'{__name__}::get_files_by_status::{inspect.currentframe().f_lineno if inspect else "UnknownLine"}'
        get_logger_name = __name__

        if not CONSTANTS_AVAILABLE: return []
        if self.df is None:
            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - DataFrame not loaded. Cannot get files by status.", main_logger=get_logger_name)
            return []
        if not self.lock:
            log_statement(loglevel="error", logstatement=f"{LOG_INS} - Repository lock not initialized. Cannot get files by status.", main_logger=get_logger_name)
            return []

        paths = []
        with self.lock:
            try:
                # Check if required columns exist
                if COL_FILEPATH not in self.df.columns or COL_STATUS not in self.df.columns:
                     log_statement(loglevel='error', logstatement=f"{LOG_INS} - Required columns ({COL_FILEPATH}, {COL_STATUS}) missing from DataFrame. Cannot get files.", main_logger=get_logger_name)
                     return []

                statuses = [status] if isinstance(status, str) else list(status)
                mask = self.df[COL_STATUS].isin(statuses)

                if base_dir:
                    try:
                        base_dir_str = str(base_dir.resolve())
                        # Ensure comparison is robust (paths stored should be resolved)
                        mask &= self.df[COL_FILEPATH].str.startswith(base_dir_str + os.sep) # Check startswith + separator
                    except Exception as res_e:
                         log_statement(loglevel="error", logstatement=f"{LOG_INS} - Could not resolve base_dir path '{base_dir}': {res_e}. Skipping base_dir filter.", main_logger=get_logger_name)

                # Get the list of filepaths using the constant
                filepath_list = self.df.loc[mask, COL_FILEPATH].tolist()

                # Convert strings to Path objects, handling potential errors
                for p_str in filepath_list:
                    if p_str and isinstance(p_str, str): # Ensure it's a non-empty string
                        try:
                            paths.append(Path(p_str))
                        except Exception as path_e:
                            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Could not convert stored path '{p_str}' to Path object: {path_e}", main_logger=get_logger_name)
                    # else: log warning about empty path?

            except Exception as e:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS} - Error accessing DataFrame to get files by status '{status}': {e}", main_logger=get_logger_name, exc_info=True)
                 return [] # Return empty list on error
        return paths

    def _get_schema_dtypes(self) -> Dict[str, Any]:
        """Returns a dictionary mapping column names to pandas dtype objects based on schema."""
        schema_dict = self._define_columns_schema()
        dtype_map = {}
        for col, dtype_str in schema_dict.items():
            try:
                # Convert string representations to actual dtype objects where needed
                 if dtype_str == 'datetime64[ns, UTC]': dtype_map[col] = pd.DatetimeTZDtype(tz='UTC')
                 elif dtype_str == 'Int64': dtype_map[col] = pd.Int64Dtype()
                 elif dtype_str == 'boolean': dtype_map[col] = pd.BooleanDtype()
                 elif dtype_str == 'Float64': dtype_map[col] = pd.Float64Dtype()
                 elif dtype_str == 'string': dtype_map[col] = pd.StringDtype()
                 else: dtype_map[col] = dtype_str # Use string directly for others like 'object'
            except Exception as e:
                 log_statement('error', f"{LOG_INS_MODULE}::_get_schema_dtypes - ERROR mapping dtype string '{dtype_str}' for column '{col}': {e}", __file__)
                 dtype_map[col] = 'object' # Fallback
        return dtype_map

    def _get_expected_columns(self) -> List[str]:
        """Returns a list of expected standard column names for any repository DataFrame."""
        # Ensure all relevant columns used throughout the process are listed here
        return [
            COL_FILEPATH, COL_HASH, COL_SIZE, COL_MODIFIED, COL_STATUS,
            COL_PROCESSED_FILENAME, COL_DATA_TYPE, COL_ERROR_INFO,
            # Add columns from linguistic processing if managed here
            COL_SEMANTIC_LABEL, COL_LINGUISTIC_METADATA,
            # Add columns from tokenization if managed here
            # COL_TOKEN_COUNT, COL_TOKENIZED_FILENAME,
            # Add other relevant columns from constants.py
            COL_STATUS_DETAIL,
        ]

    def _get_target_repo_path(self, filepath_str: str) -> Path:
        """
        Determines the correct repository file path (.repo_state.csv.zst) for a given file.

        Checks if the file path belongs to a registered sub-repository.
        If yes, returns the path to the sub-repository's state file.
        Otherwise, returns the path to the main repository's state file.

        Args:
            filepath_str (str): The path of the file to check (relative to main root_dir or absolute).

        Returns:
            Path: The Path object for the relevant .repo_state.csv.zst file.
        """
        target_repo_file = self.repo_file # Default to main repo file

        # Ensure filepath is relative to the root directory for comparison with index
        try:
            # Attempt to make the input path relative to the main root directory
            file_path_obj = Path(filepath_str)
            if not file_path_obj.is_absolute():
                 # Assume it's relative to root_dir if not absolute
                 # Or resolve it based on current working dir? Needs care.
                 # Safest might be to require paths passed to update_entry to be
                 # consistently relative to root_dir or absolute. Let's assume relative for now.
                 relative_filepath = file_path_obj
            else:
                 relative_filepath = file_path_obj.relative_to(self.root_dir)

            # Normalize path separator for comparison
            relative_filepath_str = str(relative_filepath).replace("\\", "/")
            log_statement('debug', f"{LOG_INS}:DEBUG>>Checking target repo for relative path: {relative_filepath_str}", __file__)

            if self.repo_index: # Check if index exists and is loaded
                best_match_subrepo_path = None
                longest_match_len = -1

                # Find the most specific sub-repository containing this file
                for subrepo_root_str, subrepo_info in self.repo_index.items():
                    # Normalize index path
                    norm_subrepo_root_str = subrepo_root_str.replace("\\", "/")
                    # Check if the file path starts with the sub-repository path
                    if relative_filepath_str.startswith(norm_subrepo_root_str + "/"):
                        if len(norm_subrepo_root_str) > longest_match_len:
                            longest_match_len = len(norm_subrepo_root_str)
                            # Construct the expected path to the sub-repo's state file
                            best_match_subrepo_path = self.root_dir / Path(subrepo_root_str) / REPO_FILENAME
                            log_statement('debug', f"{LOG_INS}:DEBUG>>File {relative_filepath_str} potentially belongs to sub-repo {norm_subrepo_root_str}. Candidate repo file: {best_match_subrepo_path}", __file__)

                if best_match_subrepo_path and best_match_subrepo_path.exists():
                    target_repo_file = best_match_subrepo_path
                    log_statement('debug', f"{LOG_INS}:DEBUG>>File {relative_filepath_str} confirmed to belong to sub-repo. Target repo file: {target_repo_file}", __file__)
                elif best_match_subrepo_path:
                     log_statement('warning', f"{LOG_INS}:WARNING>>File {relative_filepath_str} matched sub-repo {best_match_subrepo_path.parent} in index, but repo file {best_match_subrepo_path} does not exist! Defaulting to main repo.", __file__)
                else:
                     log_statement('debug', f"{LOG_INS}:DEBUG>>File {relative_filepath_str} does not belong to any registered sub-repo. Using main repo file: {target_repo_file}", __file__)

            else:
                log_statement('debug', f"{LOG_INS}:DEBUG>>Repository index not loaded. Using main repo file: {target_repo_file}", __file__)

        except ValueError as e:
             log_statement('warning', f"{LOG_INS}:WARNING>>Could not determine relative path for {filepath_str} against root {self.root_dir}. Using main repo file. Error: {e}", __file__)
        except Exception as e:
             log_statement('error', f"{LOG_INS}:ERROR>>Error determining target repository for {filepath_str}: {e}. Using main repo file.", __file__, exc_info=True)

        return target_repo_file

    def get_processed_path(self, source_filepath_str: str, app_state: dict) -> Path | None:
        """
        Constructs the expected path for the processed version of a source file.

        Args:
            source_filepath_str: The string path of the original source file (matching repo storage).
            app_state (dict): The application state dictionary to access config.


        Returns:
            The expected Path object for the processed file, or None if processing path
            cannot be determined.
        """
        log_statement('debug', f"{LOG_INS}:DEBUG>>Attempting to get processed path for: {source_filepath_str}", __file__)
        if self.df is None:
            log_statement('warning', f"{LOG_INS}:WARNING>>Repository DataFrame is not loaded.", __file__)
            return None

        # Find the entry in the DataFrame using the string path
        entry = self.df[self.df[COL_FILEPATH] == source_filepath_str]

        if entry.empty:
             # Maybe the stored path is absolute and input is relative or vice versa?
             # This part is tricky without knowing exactly how paths are stored.
             # Let's assume the string passed matches the storage format for now.
             log_statement('warning', f"{LOG_INS}:WARNING>>Source filepath '{source_filepath_str}' not found in repository DataFrame.", __file__)
             # Optionally, add more complex matching logic here if needed
             # e.g., try resolving Path(source_filepath_str).resolve() and comparing
             return None

        if len(entry) > 1:
             log_statement('warning', f"{LOG_INS}:WARNING>>Multiple entries found for '{source_filepath_str}'. Using the first one.", __file__)

        # Get necessary info from the DataFrame entry
        processed_filename = entry.iloc[0].get(COL_PROCESSED_FILENAME)

        if not processed_filename or not isinstance(processed_filename, str): # Check type
            log_statement('warning', f"{LOG_INS}:WARNING>>'{COL_PROCESSED_FILENAME}' not found, empty, or not a string for {source_filepath_str}. Cannot determine processed path.", __file__)
            return None

        # Construct the full path using the DataProcessor's output directory from config via app_state
        try:
             if 'config' not in app_state or not app_state['config']:
                  log_statement('error', f"{LOG_INS}:ERROR>>Configuration not available in app_state to determine output directory.", __file__)
                  return None

             dp_config = app_state['config'].get('DataProcessingConfig', {})
             output_dir_str = dp_config.get('output_directory') # Get path string from config

             if not output_dir_str:
                 log_statement('error', f"{LOG_INS}:ERROR>>'output_directory' not found in DataProcessingConfig.", __file__)
                 return None

             output_dir = Path(output_dir_str)
             full_processed_path = output_dir / processed_filename # Combine Path object and filename string

             log_statement('debug', f"{LOG_INS}:DEBUG>>Determined processed path for {source_filepath_str}: {full_processed_path}", __file__)
             # Return resolved path to ensure it's absolute and consistent
             return full_processed_path.resolve()

        except (KeyError, TypeError, ValueError, Exception) as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error constructing processed path for {source_filepath_str}: {e}", __file__, exc_info=True)
            return None
