import csv
import hashlib
import collections
import io
import inspect
import json
import os
import random
import shutil
import time
import re # Added import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt, timezone # Keep dt for potential direct use
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple # Removed Self, not standard before 3.11 easily

import numpy as np
import pandas as pd # Ensure pandas is consistently used for DataFrame ops
import torch # Assuming torch is for model saving, keep it
from tqdm import tqdm
import sys # Keep for PSUTIL_AVAILABLE check
import psutil
import zstandard as zstd
# import pickle # Pickle was imported but not visibly used; removing for now unless a use case is restored
# import zipfile # zipfile was imported but not visibly used; removing for now unless a use case is restored
import threading

# --- Assumed External Imports ---
# These imports point to other modules within the user's project structure.
# Their content is not defined here but is assumed to be available.
# If these modules or their contents are missing, NameErrors or ImportError will occur.

# For PSUTIL_AVAILABLE, a direct check is fine, but ensure psutil is actually used if True.
PSUTIL_AVAILABLE = 'psutil' in sys.modules

_helpers_module_present = False
try:
    from src.utils.helpers import _get_max_workers, _generate_file_paths, _get_file_metadata
    _helpers_module_present = True
except ImportError:
    # Placeholders if the module is missing, to allow partial linting/understanding
    def _get_max_workers() -> int: return min(8, os.cpu_count() or 1 + 4)
    def _generate_file_paths(folder_path_obj: Path) -> List[Path]: yield from folder_path_obj.rglob("*")
    def _get_file_metadata(abs_path: Path) -> Dict[str, Any]:
        return {"filepath": str(abs_path), "size": 0, "mtime": 0.0, "filename": abs_path.name, "extension": abs_path.suffix, "COL_HASH": None, "COL_STATUS": "Unknown"} # Added COL_HASH, COL_STATUS for scan_and_update
    _helpers_module_present = False # Explicitly mark as not fully available for logic later

_logger_module_present = False
try:
    from src.utils.logger import configure_logging, log_statement as actual_log_statement
    # configure_logging() # Call this once at application startup, not necessarily in a library file directly.
    _logger_module_present = True
except ImportError:
    # Basic fallback logger if the specified one is not found
    _fallback_logging_configured = False
    def _ensure_fallback_logging():
        global _fallback_logging_configured
        if not _fallback_logging_configured:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            _fallback_logging_configured = True

    LOG_LEVELS_FALLBACK = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL, "exception": logging.ERROR}
    def actual_log_statement(loglevel: str, logstatement: str, main_logger_name: Optional[str] = None, exc_info: bool = False):
        _ensure_fallback_logging()
        logger_to_use = logging.getLogger(main_logger_name if main_logger_name else __name__)
        level = LOG_LEVELS_FALLBACK.get(loglevel.lower(), logging.INFO)
        if loglevel.lower() == "exception" or exc_info:
            logger_to_use.log(level, logstatement, exc_info=True)
        else:
            logger_to_use.log(level, logstatement)
    _logger_module_present = False # Mark as not fully available

_hashing_module_present = False
try:
    from src.utils.hashing import hash_filepath # Assuming specific import rather than *
    _hashing_module_present = True
except ImportError:
    def hash_filepath(path_str: str) -> str: # Placeholder
        return hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:16]
    _hashing_module_present = False

_config_module_present = False
try:
    from src.utils.config import load_config, get_config_value
    # load_config() # Call this once at application startup
    _config_module_present = True
except ImportError:
    def load_config() -> Dict: return {} # Placeholder
    def get_config_value(section: str, key: str, default: Any = None) -> Any: return default # Placeholder
    _config_module_present = False

_constants_module_present = False
LOG_INS_MODULE_LEVEL = f"{__name__}::MODULE_LEVEL" # For module-level logging
try:
    from src.data.constants import * # This imports all constants
    CONSTANTS_AVAILABLE = True
    _constants_module_present = True
except ImportError:
    actual_log_statement('error', f"{LOG_INS_MODULE_LEVEL}:ERROR>>Importing of constants from 'src.data.constants' in project folder failed! Essential functionality will be impaired.", __file__, True)
    CONSTANTS_AVAILABLE = False
    # Define critical minimum constants if not available, to prevent immediate crashes
    COL_FILEPATH = 'filepath'
    COL_MTIME = 'mtime'
    COL_LAST_UPDATED = 'last_updated'
    COL_SIZE = 'size'
    COL_HASH = 'hash'
    COL_STATUS = 'status'
    COL_PROCESSED_PATH = 'processed_path'
    COL_ERROR_INFO = 'error_info' # Renamed from COL_ERROR to avoid conflict if COL_ERROR is a status
    COL_PROCESSED_FILENAME = 'processed_filename'
    # Add other essential COL_* constants used
    COL_FILENAME = 'filename'
    COL_FILETYPE = 'filetype' # Often same as extension
    COL_EXTENSION = 'extension'
    COL_CTIME = 'ctime'
    COL_HASHED_PATH_ID = 'hashed_path_id'
    COL_DATA_HASH = 'data_hash'
    COL_COMPRESSED_FLAG = 'compressed_flag'
    COL_TOKENIZED_PATH = 'tokenized_path'
    COL_IS_COPY_FLAG = 'is_copy_flag'
    COL_DESIGNATION = 'designation'
    COL_DATA_CLASSIFICATION = 'data_classification'
    COL_FINAL_CLASSIFICATION = 'final_classification'
    COL_SEMANTIC_LABEL = 'semantic_label'
    COL_LINGUISTIC_METADATA = 'linguistic_metadata'
    BASE_DATA_DIR = 'base_data_dir' # Placeholder if constant is missing


    COMPRESSION_LEVEL = 3
    STATE_DIR = Path("./.state")
    # Define a default name for the main repository data file.
    DEFAULT_REPO_DATA_FILENAME = "repository_data.csv.zst"
    INDEX_FILE = Path("./.repository_index.json") # Default path for the index
    MAIN_REPO_HEADER = [COL_FILEPATH, COL_FILENAME, COL_SIZE, COL_MTIME, COL_HASH, COL_STATUS] # Minimal
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.py', '.json'} # Minimal
    DEFAULT_NA_REP = "<NA>"
    CHECKPOINT_DIR = Path("./.checkpoints")
    LOG_DIR = Path("./.logs")
    STATUS_DISCOVERED = "discovered"
    STATUS_ERROR = "error"
    # Keys for repository index structure
    INDEX_KEY_PATH = "path"
    INDEX_KEY_METADATA = "metadata"
    INDEX_KEY_CHILDREN = "children"


_gpu_utils_present = False
try:
    import cudf
    import cupy # cupy might not be directly used by DataRepository but often comes with cudf
    GPU_AVAILABLE = True
    _gpu_utils_present = True
except ImportError:
    actual_log_statement('info', f"{LOG_INS_MODULE_LEVEL}:INFO>>cuDF and/or cuPy not found. GPU mode will be unavailable.", __file__)
    cudf = None # Ensure cudf is None if import fails
    cupy = None
    GPU_AVAILABLE = False

# Helper to generate LOG_INS prefix
def _get_log_ins(frame_info: Optional[inspect.FrameInfo], class_name: Optional[str] = None) -> str:
    if not frame_info:
        return f"{__name__}::{class_name or 'UnknownClass'}::UnknownFunction::UnknownLine"
    func_name = frame_info.function
    module_name = inspect.getmodule(frame_info.frame).__name__ if inspect.getmodule(frame_info.frame) else "__main__"
    line_no = frame_info.lineno
    if class_name:
        return f"{module_name}::{class_name}::{func_name}::{line_no}"
    return f"{module_name}::{func_name}::{line_no}"


class DataRepository:
    def __init__(self, directory_path: Optional[Union[str, Path]] = None,
                 repo_data_file_path: Optional[Union[str, Path]] = None, # Renamed from repo_path for clarity
                 target_dir_for_new_repo: Optional[Union[str, Path]] = None, # Clarified purpose
                 read_only: bool = False, use_gpu: bool = False):
        """
        Initializes the DataRepository.

        Args:
            directory_path (Optional[Union[str, Path]]): Path to an existing repository's root directory.
                                                        The data file is expected inside .repository subdir.
                                                        Used if repo_data_file_path is None.
            repo_data_file_path (Optional[Union[str, Path]]): Direct path to an existing repository data file (e.g., .csv.zst).
                                                              Takes precedence over directory_path.
            target_dir_for_new_repo (Optional[Union[str, Path]]): Path to the directory this repository will manage
                                                                  if creating a new one (i.e., if directory_path
                                                                  and repo_data_file_path do not lead to an existing repo).
            read_only (bool): If True, repository will be loaded in read-only mode (no saves).
            use_gpu (bool): Flag to attempt using cuDF if available.
        """
        self.LOG_INS_CLASS_PREFIX = f"{__name__}::{self.__class__.__name__}"
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)

        actual_log_statement('info', f"{LOG_INS}:INFO>>Initializing DataRepository. ReadOnly: {read_only}, UseGPU: {use_gpu}", __file__)

        if not CONSTANTS_AVAILABLE:
            actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Essential constants are not available. DataRepository may not function correctly.", __file__)
            # Decide if to raise error or proceed with caution. For now, proceed.

        self.lock = threading.RLock() # Changed from Lock to RLock for re-entrant capabilities
        self.read_only = read_only
        self.use_gpu = use_gpu and GPU_AVAILABLE and cudf is not None
        actual_log_statement('info', f"{LOG_INS}:INFO>>GPU Mode: {self.use_gpu}", __file__)

        self.root_dir: Optional[Path] = None
        self.repo_data_file: Optional[Path] = None # Path to the actual data file (e.g., csv.zst)
        self.repo_admin_dir: Optional[Path] = None # Path to the .repository directory
        self.repo_index_file: Optional[Path] = None # Path to the sub-repository index JSON

        if repo_data_file_path:
            self.repo_data_file = Path(repo_data_file_path).resolve()
            self.repo_admin_dir = self.repo_data_file.parent
            self.root_dir = self.repo_admin_dir.parent # Assumes .repository is one level down
            # If target_dir_for_new_repo is also given, it might be for context, or an inconsistency.
            # For now, repo_data_file_path defines the primary structure.
            actual_log_statement('info', f"{LOG_INS}:INFO>>Mode: Loading from direct repo data file: {self.repo_data_file}", __file__)
        elif directory_path:
            self.root_dir = Path(directory_path).resolve()
            self.repo_admin_dir = self.root_dir / ".repository" # Standard admin sub-directory
            # Determine repo_data_file name - this needs a consistent strategy.
            # Using a fixed name or a hash-based name. Let's use a fixed name for simplicity here.
            # The hash-based naming (`get_repo_filename`) can be complex with globals.
            # Assuming a constant for the default repo data file name if not using hashes for main repo.
            repo_data_filename_to_use = getattr(sys.modules.get('src.data.constants', object()), 'DEFAULT_REPO_DATA_FILENAME', 'repository_data.csv.zst')
            self.repo_data_file = self.repo_admin_dir / repo_data_filename_to_use
            actual_log_statement('info', f"{LOG_INS}:INFO>>Mode: Using directory_path. Root: {self.root_dir}, RepoFile: {self.repo_data_file}", __file__)
            if self.repo_data_file.exists():
                actual_log_statement('info', f"{LOG_INS}:INFO>>Repo data file exists at: {self.repo_data_file}", __file__)
            else:
                actual_log_statement('warning', f"{LOG_INS}:WARNING>>Repo data file does not exist at: {self.repo_data_file}. Will create new repo.", __file__)
                # If the file doesn't exist, create a new, empty, compressed repository file.
                if not self.read_only:
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Attempting to create new, empty repository data file: {self.repo_data_file}", __file__)
                    try:
                        # Ensure the parent directory for the repo data file exists.
                        # self.repo_admin_dir should be self.repo_data_file.parent
                        parent_dir = self.repo_data_file.parent
                        parent_dir.mkdir(parents=True, exist_ok=True)

                        # self.expected_columns_order should be initialized before this code runs.
                        if not hasattr(self, 'expected_columns_order') or not self.expected_columns_order:
                            actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Schema (expected_columns_order) not initialized. Cannot create new repo file header for {self.repo_data_file}.", __file__)
                            raise RuntimeError(f"Schema not initialized, cannot create {self.repo_data_file}")
                        
                        header_line = ",".join(self.expected_columns_order) + "\n"
                        
                        # Use self.compression_level if available, else a default from constants.
                        compression_lvl = getattr(self, 'compression_level', COMPRESSION_LEVEL)
                        cctx = zstd.ZstdCompressor(level=compression_lvl)
                        
                        # Write to a temporary file first to avoid corruption on error.
                        temp_repo_file_path = self.repo_data_file.with_suffix(f"{self.repo_data_file.suffix}.tmp_init_{int(time.time())}")

                        with open(temp_repo_file_path, 'wb') as f_out:
                            with cctx.stream_writer(f_out) as writer:
                                writer.write(header_line.encode('utf-8'))
                        
                        shutil.move(str(temp_repo_file_path), str(self.repo_data_file))
                        actual_log_statement('info', f"{LOG_INS}:INFO>>Successfully created new, empty repository data file: {self.repo_data_file}", __file__)

                    except Exception as e_create:
                        actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to create new repository data file {self.repo_data_file}: {e_create}", __file__, True)
                        # Depending on policy, you might want to raise an error here or allow initialization to proceed
                        # knowing the repo file couldn't be created.
                else:
                    actual_log_statement('warning', f"{LOG_INS}:WARNING>>Read-only mode. New repository file {self.repo_data_file} will not be created, though it was not found.", __file__)
        elif target_dir_for_new_repo:
            self.root_dir = Path(target_dir_for_new_repo).resolve()
            self.repo_admin_dir = self.root_dir / ".repository"
            repo_data_filename_to_use = getattr(sys.modules.get('src.data.constants', object()), 'DEFAULT_REPO_DATA_FILENAME', 'repository_data.csv.zst')
            self.repo_data_file = self.repo_admin_dir / repo_data_filename_to_use
            actual_log_statement('info', f"{LOG_INS}:INFO>>Mode: New repo for target_dir. Root: {self.root_dir}, RepoFile: {self.repo_data_file}", __file__)
        else:
            actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Must provide directory_path, repo_data_file_path, or target_dir_for_new_repo.", __file__)
            raise ValueError("Insufficient path information to initialize DataRepository.")

        # Ensure admin directory exists if we might write to it
        if not self.read_only and self.repo_admin_dir:
            try:
                self.repo_admin_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to create repo admin directory {self.repo_admin_dir}: {e}", __file__, True)
                # This might be critical depending on operations.

        # Sub-repository index file path (standardized location)
        index_file_name = getattr(sys.modules.get('src.data.constants', object()), 'INDEX_FILE_NAME', 'sub_repo_index.json')

        if self.repo_admin_dir:
             self.repo_index_file = self.repo_admin_dir / index_file_name
        else: # Should not happen if paths are set up correctly
             self.repo_index_file = self.root_dir / index_file_name if self.root_dir else Path('.') / index_file_name
             actual_log_statement('warning', f"{LOG_INS}:WARNING>>Repo admin directory not set, placing index file at: {self.repo_index_file}", __file__)


        self.repo_index: Optional[Dict[str, Any]] = None # Loaded structure for sub-repos
        self.repo_index_dirty = False

        # Define schema and timestamp columns
        try:
            self.columns_schema_dict = self._define_columns_schema()
            self.columns_schema_dtypes = self._get_schema_dtypes()
            self.expected_columns_order = self._get_expected_columns() # Uses the corrected version
            # Derive timestamp_columns from the schema dtypes
            self.timestamp_columns = {
                col for col, dtype_obj in self.columns_schema_dtypes.items()
                if isinstance(dtype_obj, pd.DatetimeTZDtype)
            }
            actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Schema defined. Dtypes map count: {len(self.columns_schema_dtypes)}. Timestamps: {self.timestamp_columns}", __file__)
        except Exception as schema_e:
             actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Failed to define repository schema during init: {schema_e}. Repository may be unusable.", __file__, True)
             raise RuntimeError("Failed to initialize repository schema") from schema_e

        # Load compression level
        self.compression_level = COMPRESSION_LEVEL # Default
        if _config_module_present:
            try:
                 cfg_comp_level = get_config_value('DataRepositoryConfig', 'compression_level')
                 if cfg_comp_level is not None: self.compression_level = int(cfg_comp_level)
            except (ValueError, TypeError) as e:
                 actual_log_statement('warning', f"{LOG_INS}:WARNING>>Invalid compression_level in config: {e}. Using default {self.compression_level}.", __file__)
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Compression level set to: {self.compression_level}", __file__)

        # State save directory
        self.state_save_dir = STATE_DIR
        try:
            self.state_save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as dir_err:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to create state directory {self.state_save_dir}: {dir_err}", __file__, True)
            self.state_save_dir = None # Indicate state saving might fail

        # Load main DataFrame and repository index
        self.df: Optional[pd.DataFrame] = None
        if not self.read_only:
            self._load_repository_index() # Load index first
        self._load_repo() # Load main DataFrame (handles empty/new if file doesn't exist)

        if self.df is None and not self.read_only: # Should be an empty DF if load fails, not None ideally
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>DataFrame is None after initial load attempt. Initializing empty.", __file__)
            self.df = pd.DataFrame(columns=self.expected_columns_order).astype(self.columns_schema_dtypes, errors='ignore')

        # Check if repo_data_file exists after setup; log if it was expected but not found.
        if repo_data_file_path and self.repo_data_file and not self.repo_data_file.exists() and (self.df is None or self.df.empty):
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Provided repo_data_file_path '{repo_data_file_path}' does not exist. Initialized empty repository.", __file__)

        actual_log_statement('info', f"{LOG_INS}:INFO>>DataRepository initialization complete. Repo File: {self.repo_data_file}, Root Dir: {self.root_dir}", __file__)

    def _define_columns_schema(self) -> Dict[str, str]:
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        schema = {
            COL_FILEPATH: 'string',
            COL_FILENAME: 'string',
            COL_FILETYPE: 'string',
            COL_EXTENSION: 'string',
            COL_SIZE: 'Int64',
            COL_MTIME: 'datetime64[ns, UTC]',
            COL_CTIME: 'datetime64[ns, UTC]',
            COL_HASH: 'string', # Primary content hash
            COL_HASHED_PATH_ID: 'string', # Hash of the filepath itself
            COL_DATA_HASH: 'string', # Could be redundant with COL_HASH or for different purpose
            COL_STATUS: 'string',
            COL_PROCESSED_PATH: 'string',
            COL_TOKENIZED_PATH: 'string',
            COL_LAST_UPDATED: 'datetime64[ns, UTC]',
            COL_COMPRESSED_FLAG: 'boolean',
            COL_IS_COPY_FLAG: 'boolean',
            COL_DESIGNATION: 'string',
            COL_ERROR_INFO: 'string', # Changed from COL_ERROR to avoid clash with a status
            COL_DATA_CLASSIFICATION: 'string',
            COL_FINAL_CLASSIFICATION: 'string',
            COL_SEMANTIC_LABEL: 'string',
            COL_LINGUISTIC_METADATA: 'string', # Store as JSON string
            BASE_DATA_DIR: 'string', # The base directory this file belonged to during scan
            # Add COL_PROCESSED_FILENAME if it's distinct from PROCESSED_PATH's name part
            COL_PROCESSED_FILENAME: 'string',
        }
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Schema defined with {len(schema)} columns.", __file__)
        return schema

    def _get_schema_dtypes(self) -> Dict[str, Any]:
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        schema_dict = self.columns_schema_dict # Use the stored dict
        dtype_map = {}
        for col, dtype_str in schema_dict.items():
            try:
                 if dtype_str == 'datetime64[ns, UTC]': dtype_map[col] = pd.DatetimeTZDtype(tz='UTC')
                 elif dtype_str == 'Int64': dtype_map[col] = pd.Int64Dtype()
                 elif dtype_str == 'boolean': dtype_map[col] = pd.BooleanDtype()
                 elif dtype_str == 'Float64': dtype_map[col] = pd.Float64Dtype() # Added for completeness
                 elif dtype_str == 'string': dtype_map[col] = pd.StringDtype()
                 else: dtype_map[col] = dtype_str # Basic numpy dtypes
            except Exception as e:
                 actual_log_statement('error', f"{LOG_INS}:ERROR>>Error mapping dtype string '{dtype_str}' for column '{col}': {e}. Defaulting to 'object'.", __file__, True)
                 dtype_map[col] = 'object'
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Mapped schema strings to {len(dtype_map)} dtype objects.", __file__)
        return dtype_map

    def _get_expected_columns(self) -> List[str]:
        """
        Returns a consistently ordered list of all columns defined in the schema.
        This should be the single source of truth for column order in DataFrames.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        # Return keys from the schema dictionary in a defined order (e.g., sorted or manually specified)
        # For consistency, let's use the order they are defined in _define_columns_schema
        defined_order = list(self.columns_schema_dict.keys())
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Expected columns order defined with {len(defined_order)} columns.", __file__)
        return defined_order

    def _load_repo(self):
        """Loads the main repository data file into self.df."""
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        if not self.repo_data_file:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>repo_data_file path is not set. Cannot load main repository.", __file__)
            self.df = pd.DataFrame(columns=self.expected_columns_order).astype(self.columns_schema_dtypes, errors='ignore')
            return

        actual_log_statement('info', f"{LOG_INS}:INFO>>Loading main repository data from: {self.repo_data_file}", __file__)
        loaded_df = self._load_repo_dataframe(self.repo_data_file)

        if loaded_df is None: # Critical load failure from helper
             actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>_load_repo_dataframe returned None for main repo {self.repo_data_file}. Initializing empty DataFrame.", __file__)
             self.df = pd.DataFrame(columns=self.expected_columns_order).astype(self.columns_schema_dtypes, errors='ignore')
        else:
             self.df = loaded_df
             if self.df.empty and self.repo_data_file.exists() and self.repo_data_file.stat().st_size > 0: # Check size to ensure it wasn't legitimately empty
                  actual_log_statement('warning', f"{LOG_INS}:WARNING>>Main repository file {self.repo_data_file} loaded as empty, but file exists and has size.", __file__)
             else:
                  actual_log_statement('info', f"{LOG_INS}:INFO>>Main repository DataFrame loaded/initialized into self.df ({len(self.df)} rows).", __file__)

    def _load_repo_dataframe(self, repo_file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads a repository DataFrame from a specified .csv.zst file.
        Returns a schema-aligned DataFrame, or an empty schema-aligned DataFrame
        if file not found/empty, or None on critical schema definition failure.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Attempting to load repository DataFrame from: {repo_file_path}", __file__)

        # Schema info should be available from self.
        if not self.columns_schema_dict or not self.expected_columns_order or not self.columns_schema_dtypes:
             actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Instance schema information not initialized. Cannot load DataFrame correctly.", __file__)
             return None # Critical failure

        # Prepare empty DataFrame structure for returning on error/empty
        empty_df = pd.DataFrame(columns=self.expected_columns_order).astype(self.columns_schema_dtypes, errors='ignore')

        if not repo_file_path.exists():
            actual_log_statement('info', f"{LOG_INS}:INFO>>Repository file '{repo_file_path}' not found. Returning empty schema-aligned DataFrame.", __file__)
            return empty_df
        try:
            if repo_file_path.stat().st_size < 10: # Heuristic for empty or header-only
                actual_log_statement("warning", f"{LOG_INS}:WARNING>>Repo file '{repo_file_path}' appears empty or has only header. Returning empty schema-aligned DataFrame.", __file__)
                return empty_df
        except OSError as stat_e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error accessing repo file stats '{repo_file_path}': {stat_e}. Returning empty schema-aligned DataFrame.", __file__)
            return empty_df

        try:
            pdf: Optional[pd.DataFrame] = None
            na_values_list = [DEFAULT_NA_REP, '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                             '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'nan', 'null', '<NONE>', '<NULL>'] # Added common ones

            dctx = zstd.ZstdDecompressor()
            with open(repo_file_path, 'rb') as ifh:
                with dctx.stream_reader(ifh) as reader:
                    # Read all as string first for robust parsing
                    pdf = pd.read_csv(
                        io.TextIOWrapper(reader, encoding='utf-8', errors='replace'),
                        header=0,
                        dtype=str,
                        encoding='utf-8',
                        keep_default_na=False, # Handle NAs manually after initial string load
                        na_values=na_values_list, # Recognize these strings as NA during initial load
                        low_memory=False,
                        escapechar='\\'
                    )

            if pdf is None or pdf.empty:
                actual_log_statement("info", f"{LOG_INS}:INFO>>Repository file '{repo_file_path}' loaded as empty or only header. Returning empty schema-aligned DataFrame.", __file__)
                return empty_df

            actual_log_statement("info", f"{LOG_INS}:INFO>>CSV loaded {len(pdf)} rows from '{repo_file_path}'. Normalizing columns...", __file__)

            # --- Column Renaming (simplified, assumes constants are mostly used now) ---
            # If complex legacy remapping is needed, it would go here.
            # For now, assume current column names mostly match constants.

            # --- Create final_df ensuring all expected columns are present ---
            final_df = pd.DataFrame(index=pdf.index)
            for col_const in self.expected_columns_order:
                if col_const in pdf.columns:
                    if isinstance(pdf[col_const], pd.DataFrame): # Handle duplicate source columns
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>Duplicate column '{col_const}' in source CSV '{repo_file_path}'. Using first instance.", __file__)
                        final_df[col_const] = pdf[col_const].iloc[:, 0]
                    else:
                        final_df[col_const] = pdf[col_const]
                else:
                    # Add missing expected column (will be typed later)
                    final_df[col_const] = pd.Series([None] * len(pdf), index=pdf.index, dtype='object') # Start as object

            # --- Apply Robust Type Conversions ---
            for col_const, target_dtype_obj in self.columns_schema_dtypes.items():
                if col_const not in final_df.columns: continue
                current_series = final_df[col_const]
                try:
                    # Replace various NA representations with np.nan for consistent conversion
                    current_series.replace(na_values_list, np.nan, inplace=True)

                    if isinstance(target_dtype_obj, pd.DatetimeTZDtype):
                        converted_dt = pd.to_datetime(current_series, errors='coerce', utc=True) # format='ISO8601' if strictly ISO
                        final_df[col_const] = converted_dt # Dtype applied by astype later if needed by DataFrame structure
                    elif isinstance(target_dtype_obj, pd.Int64Dtype):
                        final_df[col_const] = pd.to_numeric(current_series, errors='coerce') # astype to Int64 done by final DataFrame.astype
                    elif isinstance(target_dtype_obj, pd.StringDtype):
                        final_df[col_const] = current_series # astype to StringDtype done by final DataFrame.astype
                    elif isinstance(target_dtype_obj, pd.Float64Dtype):
                        final_df[col_const] = pd.to_numeric(current_series, errors='coerce') # astype to Float64Dtype by final .astype
                    elif isinstance(target_dtype_obj, pd.BooleanDtype):
                        bool_map = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}
                        # Create a series that's definitively string for .str.lower()
                        str_series = current_series.astype(str).str.lower()
                        final_df[col_const] = str_series.map(bool_map) # NaNs will result if not in map, handled by .astype('boolean')
                    else: # Basic dtypes
                        if current_series.dtype != target_dtype_obj: # Only convert if necessary
                            final_df[col_const] = current_series.astype(target_dtype_obj, errors='ignore')
                except Exception as conv_e:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Converting column '{col_const}' to '{target_dtype_obj}' in '{repo_file_path}': {conv_e}. Keeping as object.", __file__, True)
                    final_df[col_const] = final_df[col_const].astype('object') # Fallback

            # Ensure all columns in final_df conform to the schema dtypes
            final_df = final_df.astype(self.columns_schema_dtypes, errors='ignore')
            final_df = final_df.reindex(columns=self.expected_columns_order) # Final order check

            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository loaded and processed ({len(final_df)} entries) from '{repo_file_path}'.", __file__)
            return final_df

        except pd.errors.EmptyDataError:
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Repo file '{repo_file_path}' pandas EmptyDataError. Returning empty schema-aligned DataFrame.", __file__)
            return empty_df
        except FileNotFoundError: # Should be caught earlier, but as a safeguard
            actual_log_statement('info', f"{LOG_INS}:INFO>>Repo file {repo_file_path} FileNotFoundError. Returning empty schema-aligned DataFrame.", __file__)
            return empty_df
        except zstd.ZstdError as zstde:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>Zstandard error reading '{repo_file_path}': {zstde}. Returning empty.", __file__, True)
             return empty_df
        except Exception as e:
            actual_log_statement("critical", f"{LOG_INS}:CRITICAL>>Repository load failed for '{repo_file_path}': {e}. Returning empty.", __file__, True)
            return empty_df

    def save(self, save_type: str, **kwargs):
        """Gateway method for various save operations."""
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Save request. Type: '{save_type}', Args: {list(kwargs.keys())}", __file__)
        save_type_lower = save_type.lower()

        if self.read_only and save_type_lower != 'config_snapshot': # Allow saving config even if read_only
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Attempted to save '{save_type}' in read-only mode. Operation cancelled.", __file__)
            return

        # ... (rest of the save method logic from previous refactoring, adjusted for class members)
        # For brevity, not repeating the full save dispatcher here but it would call:
        if save_type_lower == 'repository':
            self.save_repo() # This now refers to the instance method for main repo save
        elif save_type_lower == 'progress':
            process_id = kwargs.get('process_id')
            current_state = kwargs.get('current_state')
            # output_dir optional, defaults to self.state_save_dir
            self.save_progress(process_id, current_state, kwargs.get('output_dir'))
        # ... other save types ...
        else:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Unsupported save_type: '{save_type}'", __file__)


    def save_repo(self):
        """Saves the main internal repository state (DataFrame and Index)."""
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Request to save main repository state.", __file__)

        if self.read_only:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Attempted to save main repository in read-only mode. Operation cancelled.", __file__)
            return

        if not self.repo_data_file:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>repo_data_file path is not set. Cannot save main repository.", __file__)
            return

        with self.lock:
            df_save_success = False
            if self.df is not None:
                df_save_success = self._save_repo_dataframe(self.df, self.repo_data_file)
            else: # self.df is None
                actual_log_statement('warning', f"{LOG_INS}:WARNING>>Main DataFrame (self.df) is None. Saving empty repository.", __file__)
                # Create an empty DataFrame with schema and save it
                empty_df = pd.DataFrame(columns=self.expected_columns_order).astype(self.columns_schema_dtypes, errors='ignore')
                df_save_success = self._save_repo_dataframe(empty_df, self.repo_data_file)

            # Save the sub-repository index if it's dirty
            index_save_success = self._save_repository_index() # No args, saves self.repo_index

            if df_save_success and index_save_success:
                actual_log_statement('info', f"{LOG_INS}:INFO>>Main repository state saved successfully.", __file__)
            else:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Main repository save failed. DF Save: {df_save_success}, Index Save: {index_save_success}", __file__)


    def _save_repo_dataframe(self, df_to_save: pd.DataFrame, target_repo_file_path: Path) -> bool:
        """Saves a DataFrame to a specified .csv.zst file."""
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Saving DataFrame ({len(df_to_save)} rows) to: {target_repo_file_path}", __file__)

        if self.read_only: # Double check, though save_repo should catch this
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Attempted _save_repo_dataframe in read-only mode.", __file__)
            return False

        temp_path = target_repo_file_path.with_suffix(f'{target_repo_file_path.suffix}.tmp_{int(time.time())}')
        try:
            target_repo_file_path.parent.mkdir(parents=True, exist_ok=True)
            df_copy = df_to_save.copy()

            # Ensure schema conformance (add missing, reorder)
            df_copy = df_copy.reindex(columns=self.expected_columns_order) # Adds missing as NaN
            # Apply schema types robustly, this ensures new NA columns get correct NA type
            for col, dtype_obj in self.columns_schema_dtypes.items():
                if col in df_copy.columns:
                    try:
                        if isinstance(dtype_obj, pd.StringDtype) and not isinstance(df_copy[col].dtype, pd.StringDtype):
                             df_copy[col] = df_copy[col].astype(pd.StringDtype()) # Convert to nullable string first
                        elif pd.api.types.is_datetime64_any_dtype(dtype_obj) and not pd.api.types.is_datetime64_any_dtype(df_copy[col].dtype):
                             df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', utc=True)

                        # For other pandas extension dtypes, ensure they are correctly cast
                        # Especially if they were filled with np.nan from reindex instead of pd.NA
                        if isinstance(dtype_obj, (pd.Int64Dtype, pd.BooleanDtype, pd.Float64Dtype)):
                             df_copy[col] = df_copy[col].astype(dtype_obj, errors='ignore') # errors='ignore' can be risky
                    except Exception as e_type_conv:
                         actual_log_statement('warning', f"{LOG_INS}:WARNING>>Error ensuring type for column '{col}' to '{dtype_obj}': {e_type_conv}. Data might not save correctly.", __file__)


            # Convert for CSV: Timestamps to ISO string, NAs to DEFAULT_NA_REP
            for col in df_copy.columns:
                series = df_copy[col]
                if col in self.timestamp_columns and pd.api.types.is_datetime64_any_dtype(series.dtype):
                    # Ensure it's UTC before formatting
                    if series.dt.tz is None: series = series.dt.tz_localize('UTC')
                    elif str(series.dt.tz).upper() != 'UTC': series = series.dt.tz_convert('UTC')
                    df_copy[col] = series.apply(lambda dt: dt.isoformat(timespec='microseconds') if pd.notna(dt) else DEFAULT_NA_REP)
                elif isinstance(series.dtype, pd.BooleanDtype): # Handle nullable boolean
                    df_copy[col] = series.apply(lambda x: 'true' if x is True else ('false' if x is False else DEFAULT_NA_REP))
                else: # General case, ensure NAs are consistently represented
                    df_copy[col] = series.fillna(DEFAULT_NA_REP).astype(str)


            cctx = zstd.ZstdCompressor(level=self.compression_level)
            with open(temp_path, 'wb') as f_out, \
                 tqdm(total=len(df_copy), desc=f"Saving {target_repo_file_path.name}", unit="row", leave=False) as pbar:
                with cctx.stream_writer(f_out) as writer:
                    # Write header
                    writer.write((",".join(self.expected_columns_order) + "\n").encode('utf-8'))
                    pbar.set_postfix_str("Header")
                    # Write data in chunks
                    chunk_size = 10000
                    for i in range(0, len(df_copy), chunk_size):
                        chunk = df_copy.iloc[i:min(i + chunk_size, len(df_copy))]
                        csv_buffer = io.StringIO()
                        chunk.to_csv(csv_buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\', encoding='utf-8', line_terminator='\n')
                        writer.write(csv_buffer.getvalue().encode('utf-8'))
                        csv_buffer.close()
                        pbar.update(len(chunk))
                        pbar.set_postfix_str(f"{pbar.n}/{len(df_copy)}")
            shutil.move(str(temp_path), str(target_repo_file_path))
            actual_log_statement("info", f"{LOG_INS}:INFO>>DataFrame saved to {target_repo_file_path}", __file__)
            return True
        except Exception as e:
            actual_log_statement("critical", f"{LOG_INS}:CRITICAL>>Save DataFrame failed for {target_repo_file_path}: {e}", __file__, True)
            if temp_path.exists():
                try: os.remove(temp_path)
                except OSError: pass
            return False

    def _load_repository_index(self):
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        if not self.repo_index_file:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Repository index file path not set. Cannot load index.", __file__)
            self.repo_index = {}
            return

        actual_log_statement('info', f"{LOG_INS}:INFO>>Loading repository index from: {self.repo_index_file}", __file__)
        if not self.repo_index_file.exists():
            actual_log_statement('info', f"{LOG_INS}:INFO>>Repository index file '{self.repo_index_file}' not found. Initializing empty index.", __file__)
            self.repo_index = {}
            return
        try:
            with open(self.repo_index_file, 'r', encoding='utf-8') as f:
                 raw_index_data = json.load(f)
            # Convert string paths back to Path objects
            self.repo_index = {}
            for key, entry_data in raw_index_data.items():
                if isinstance(entry_data, dict) and INDEX_KEY_PATH in entry_data:
                    try:
                        entry_data[INDEX_KEY_PATH] = Path(entry_data[INDEX_KEY_PATH])
                        self.repo_index[key] = entry_data
                    except TypeError:
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>Invalid path string in index for key '{key}'. Skipping.", __file__)
                else:
                    actual_log_statement('warning', f"{LOG_INS}:WARNING>>Invalid entry structure in index for key '{key}'. Skipping.", __file__)

            actual_log_statement('info', f"{LOG_INS}:INFO>>Loaded repository index with {len(self.repo_index)} entries.", __file__)
            self.repo_index_dirty = False
        except (json.JSONDecodeError, Exception) as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to load/parse repository index '{self.repo_index_file}': {e}. Initializing empty index.", __file__, True)
            self.repo_index = {}


    def _save_repository_index(self) -> bool:
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        if not self.repo_index_file:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Repository index file path not set. Cannot save index.", __file__)
            return False
        if not self.repo_index_dirty and self.repo_index_file.exists(): # Only save if dirty or if file doesn't exist yet (initial save)
            actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Repository index not dirty. Skipping save to {self.repo_index_file}.", __file__)
            return True # Considered success as no save was needed

        actual_log_statement('info', f"{LOG_INS}:INFO>>Saving repository index to: {self.repo_index_file}", __file__)
        if self.repo_index is None:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>self.repo_index is None. Cannot save index.", __file__)
            return False # Nothing to save

        # Convert Path objects to strings for JSON serialization
        serializable_index = {}
        for key, entry_data in self.repo_index.items():
            s_entry = entry_data.copy() # Shallow copy
            if isinstance(s_entry.get(INDEX_KEY_PATH), Path):
                s_entry[INDEX_KEY_PATH] = str(s_entry[INDEX_KEY_PATH].resolve())
            serializable_index[key] = s_entry

        try:
            self.repo_index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.repo_index_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_index, f, indent=4, sort_keys=True)
            self.repo_index_dirty = False
            actual_log_statement('info', f"{LOG_INS}:INFO>>Successfully saved repository index.", __file__)
            return True
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to save repository index: {e}", __file__, True)
            return False

    # Method to get a specific configuration for the repository data file name.
    # This replaces the problematic global REPO_FILENAME and get_repo_filename function.
    def _get_repo_data_filename(self, repo_root_path: Path) -> str:
        """
        Determines the name of the repository data file.
        Can be based on a hash of the repo_root_path or a fixed name.
        For this refactor, we'll use a fixed name for simplicity if constants aren't providing a pattern.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        # Option 1: Hash-based name (if src.utils.hashing.hash_filepath is robust)
        # if _hashing_module_present and CONSTANTS_AVAILABLE:
        #     repo_hash = hash_filepath(str(repo_root_path.resolve()))
        #     return f"data_repository_{repo_hash}.csv.zst"

        # Option 2: Fixed name (simpler, default if constants are missing this detail)
        fixed_name = getattr(sys.modules.get('src.data.constants', object()), 'DEFAULT_REPO_DATA_FILENAME', 'repository_data.csv.zst')
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Using data filename: {fixed_name} for repo at {repo_root_path}", __file__)
        return fixed_name


    def get_status(self, source_filepath: Union[str, Path]) -> Optional[str]:
        """Gets the current status of a file from the repository's DataFrame."""
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        file_path_obj = Path(source_filepath).resolve()
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Getting status for: {file_path_obj}", __file__)

        if self.df is None:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>DataFrame not loaded. Cannot get status.", __file__)
            return None

        with self.lock:
            try:
                if COL_FILEPATH not in self.df.columns or COL_STATUS not in self.df.columns:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Required columns missing for status check.", __file__)
                    return None
                entry = self.df[self.df[COL_FILEPATH] == str(file_path_obj)]
                if not entry.empty:
                    status = entry[COL_STATUS].iloc[0]
                    return status if pd.notna(status) else None
                return None # File not found in repository
            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Error getting status for '{file_path_obj}': {e}", __file__, True)
                return None

    # Keeping only one get_processed_path method (the one not relying on app_state)
    def get_processed_path(self, source_filepath: Union[str, Path]) -> Optional[Path]:
        """
        Gets the processed path for a given source file from the repository DataFrame.
        Relies on COL_PROCESSED_PATH being populated correctly.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        source_file_path_obj = Path(source_filepath).resolve()
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Getting processed path for: {source_file_path_obj}", __file__)

        if self.df is None:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>DataFrame not loaded. Cannot get processed path.", __file__)
            return None

        with self.lock:
            try:
                if COL_FILEPATH not in self.df.columns or COL_PROCESSED_PATH not in self.df.columns:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Required columns missing for processed path retrieval.", __file__)
                    return None
                entry = self.df[self.df[COL_FILEPATH] == str(source_file_path_obj)]
                if not entry.empty:
                    processed_path_str = entry[COL_PROCESSED_PATH].iloc[0]
                    if pd.notna(processed_path_str) and isinstance(processed_path_str, str) and processed_path_str.strip():
                        return Path(processed_path_str).resolve()
                    actual_log_statement('debug', f"{LOG_INS}:DEBUG>>No valid processed path found for {source_file_path_obj}.", __file__)
                return None
            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Error getting processed path for '{source_file_path_obj}': {e}", __file__, True)
                return None

    def save_progress(self, process_id: str, current_state: Dict[str, Any], output_dir: Optional[Path] = None) -> bool:
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        if not process_id or not isinstance(process_id, str):
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Process ID must be a non-empty string.", __file__)
            return False

        save_dir = output_dir or self.state_save_dir
        if not save_dir:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>State save directory is not configured. Cannot save progress for '{process_id}'.", __file__)
             return False

        safe_process_id = re.sub(r'[^\w\-.]', '_', process_id)
        state_filename = f"{safe_process_id}_state.json"
        state_file_path = save_dir / state_filename
        temp_path = state_file_path.with_suffix(f'{state_file_path.suffix}.tmp_{int(time.time())}')
        actual_log_statement('info', f"{LOG_INS}:INFO>>Saving progress for '{process_id}' to {state_file_path}", __file__)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            state_to_save = current_state.copy()
            state_to_save['_save_timestamp_utc'] = dt.now(timezone.utc).isoformat()
            state_to_save['_process_id'] = process_id

            with open(temp_path, 'w', encoding='utf-8') as f_out:
                 json.dump(state_to_save, f_out, indent=4, sort_keys=True, default=str)
            shutil.move(str(temp_path), str(state_file_path))
            actual_log_statement('info', f"{LOG_INS}:INFO>>Progress saved for '{process_id}'.", __file__)
            return True
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to save progress for '{process_id}': {e}", __file__, True)
            if temp_path.exists():
                try: os.remove(temp_path)
                except OSError: pass
            return False

    def load_progress(self, process_id: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        if not process_id or not isinstance(process_id, str):
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Process ID must be a non-empty string.", __file__)
            return None

        save_dir = output_dir or self.state_save_dir
        if not save_dir:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>State save directory not configured. Cannot load progress for '{process_id}'.", __file__)
             return None

        safe_process_id = re.sub(r'[^\w\-.]', '_', process_id)
        state_filename = f"{safe_process_id}_state.json"
        state_file_path = save_dir / state_filename
        actual_log_statement('info', f"{LOG_INS}:INFO>>Loading progress for '{process_id}' from {state_file_path}", __file__)

        if not state_file_path.exists():
            actual_log_statement('info', f"{LOG_INS}:INFO>>Progress file not found for '{process_id}'.", __file__)
            return None
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f_in:
                loaded_state = json.load(f_in)
            actual_log_statement('info', f"{LOG_INS}:INFO>>Progress loaded for '{process_id}'.", __file__)
            return loaded_state
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to load progress for '{process_id}': {e}", __file__, True)
            return None

    def update_entry(self, source_filepath: Path, **kwargs_meta_update) -> bool:
        """
        Adds or updates a single file's metadata entry in the internal DataFrame (self.df).
        This method modifies self.df in memory. Call self.save_repo() to persist changes.

        Args:
            source_filepath (Path): The absolute path of the file.
            **kwargs_meta_update: Metadata key-value pairs to update. Keys should match
                                  column names in the repository schema (COL_* constants).
                                  Values will be converted to match schema types.

        Returns:
            bool: True if the DataFrame was updated successfully, False otherwise.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Attempting to update/add entry for: {source_filepath}", __file__)

        if self.read_only:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Repository is in read-only mode. Cannot update entry for {source_filepath}.", __file__)
            return False

        if self.df is None:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>DataFrame (self.df) is not loaded. Cannot update entry for {source_filepath}.", __file__)
            return False

        try:
            source_filepath_abs = source_filepath.resolve()
            source_filepath_abs_str = str(source_filepath_abs)
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not resolve source_filepath '{source_filepath}': {e}", __file__, True)
            return False

        with self.lock:
            try:
                # Prepare data for update
                update_data: Dict[str, Any] = {}

                # Ensure COL_LAST_UPDATED is set
                update_data[COL_LAST_UPDATED] = dt.now(timezone.utc)

                for key, value in kwargs_meta_update.items():
                    if key not in self.columns_schema_dict:
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>Skipping unknown metadata key '{key}' for {source_filepath_abs_str}.", __file__)
                        continue

                    target_dtype_obj = self.columns_schema_dtypes.get(key)
                    prepared_value = value # Start with the original value

                    # Basic type coercion based on schema (can be expanded)
                    # This is a simplified conversion; a more robust one would handle errors better
                    try:
                        if pd.isna(value): # Handle explicit None or np.nan if passed
                            if isinstance(target_dtype_obj, pd.DatetimeTZDtype): prepared_value = pd.NaT
                            else: prepared_value = pd.NA # For pandas nullable dtypes (String, Int64, Boolean)
                        elif isinstance(target_dtype_obj, pd.DatetimeTZDtype):
                            prepared_value = pd.to_datetime(value, errors='coerce', utc=True)
                            if pd.isna(prepared_value):
                                actual_log_statement('warning', f"{LOG_INS}:WARNING>>Could not parse '{value}' as datetime for key '{key}'. Using NaT.", __file__)
                        elif isinstance(target_dtype_obj, pd.Int64Dtype):
                            prepared_value = int(float(value)) # pd.to_numeric(value, errors='coerce').astype('Int64')
                        elif isinstance(target_dtype_obj, pd.BooleanDtype):
                            if isinstance(value, str): v_lower = value.strip().lower()
                            else: v_lower = str(value).lower()
                            if v_lower in ['true', 'yes', '1', 't']: prepared_value = True
                            elif v_lower in ['false', 'no', '0', 'f']: prepared_value = False
                            else: prepared_value = pd.NA
                        elif isinstance(target_dtype_obj, pd.Float64Dtype): # Added Float64Dtype
                            prepared_value = float(value)
                        elif isinstance(target_dtype_obj, pd.StringDtype):
                            prepared_value = str(value) if not pd.isna(value) else pd.NA
                        # else: value remains as is, assumes it matches non-pandas extension dtypes if any

                    except (ValueError, TypeError) as e_conv:
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>Type conversion error for key '{key}', value '{value}' to {target_dtype_obj}: {e_conv}. Using raw value or pd.NA.", __file__)
                        prepared_value = pd.NA # Fallback for safety

                    update_data[key] = prepared_value

                # Find existing entry
                entry_indices = self.df.index[self.df[COL_FILEPATH] == source_filepath_abs_str].tolist()

                if entry_indices: # Update existing
                    idx = entry_indices[0]
                    if len(entry_indices) > 1:
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>Multiple entries found for '{source_filepath_abs_str}'. Updating first one at index {idx}.", __file__)
                    
                    for col, val in update_data.items():
                        if col in self.df.columns:
                            self.df.loc[idx, col] = val
                        else:
                             actual_log_statement('warning', f"{LOG_INS}:WARNING>>Column '{col}' not in DataFrame schema. Cannot set value during update for existing entry.", __file__)
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Updated existing entry for {source_filepath_abs_str} at index {idx}.", __file__)
                else: # Add new entry
                    new_row_data = {col: pd.NA for col in self.expected_columns_order} # Initialize with NAs
                    new_row_data[COL_FILEPATH] = source_filepath_abs_str
                    
                    # Populate with provided metadata, falling back to defaults if not provided
                    for col_const in self.expected_columns_order:
                        if col_const in update_data:
                            new_row_data[col_const] = update_data[col_const]
                        # Ensure essential fields from kwargs_meta_update are present, even if not explicitly in update_data after filtering
                        elif col_const in kwargs_meta_update:
                            # Re-prepare this specific value if it wasn't processed into update_data
                            # This is a bit redundant; ideally all kwargs_meta_update are processed into update_data
                            # For now, let's assume update_data contains all relevant processed values
                            pass


                    # Set default status if not provided
                    if pd.isna(new_row_data.get(COL_STATUS)):
                        new_row_data[COL_STATUS] = STATUS_DISCOVERED

                    # Ensure types before concat by creating a single-row DataFrame
                    new_row_df = pd.DataFrame([new_row_data], columns=self.expected_columns_order)
                    new_row_df = new_row_df.astype(self.columns_schema_dtypes, errors='ignore')

                    self.df = pd.concat([self.df, new_row_df], ignore_index=True)
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Added new entry for {source_filepath_abs_str}.", __file__)
                
                # Mark DataFrame as needing save implicitly (caller should handle save_repo)
                return True

            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to update/add entry for {source_filepath_abs_str}: {e}", __file__, True)
                return False
    
    def _scan_directory(self, folder_path_obj: Path, existing_files_set: Set[str]) -> List[Dict[str, Any]]:
        """
        Scans a directory recursively for files, gets their metadata using the
        external _get_file_metadata helper, and filters against known paths.
        This version is adapted from your earlier refactored code.

        Args:
            folder_path_obj (Path): The directory path object to scan.
            existing_files_set (Set[str]): A set of resolved file path strings already known,
                                           to potentially skip or handle differently.

        Returns:
            List[Dict[str, Any]]: A list of metadata dictionaries for files found.
                                  The definition of "new/changed" depends on how _get_file_metadata
                                  and the calling logic use existing_files_set.
                                  This implementation primarily focuses on discovering all files
                                  and getting their metadata.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Scanning directory '{folder_path_obj}' for file metadata.", __file__)

        if not _helpers_module_present: # Check if src.utils.helpers was available
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Helper module (src.utils.helpers) not available. Cannot perform scan.", __file__)
            return []

        max_workers = _get_max_workers() # From src.utils.helpers
        all_discovered_files_metadata = []

        # Stage 1: Collect all potential file paths using _generate_file_paths
        potential_paths_to_process = []
        try:
            for filepath_obj in _generate_file_paths(folder_path_obj): # From src.utils.helpers
                if filepath_obj.is_file(): # Ensure it's a file
                    # Basic exclusion: skip .repository admin dir and the repo data file itself
                    if self.repo_admin_dir and self.repo_admin_dir in filepath_obj.parents:
                        continue
                    if self.repo_data_file and filepath_obj == self.repo_data_file:
                        continue
                    if self.repo_index_file and filepath_obj == self.repo_index_file:
                        continue
                    # Add other exclusions if needed (e.g. .git)

                    potential_paths_to_process.append(filepath_obj)
        except Exception as gen_e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Error during path generation in '{folder_path_obj}': {gen_e}", __file__, True)
            return [] # Critical failure in path discovery

        actual_log_statement('info', f"{LOG_INS}:INFO>>Found {len(potential_paths_to_process)} potential file items in '{folder_path_obj}'. Getting metadata.", __file__)

        if not potential_paths_to_process:
            return []

        # Stage 2: Parallel Metadata Gathering using _get_file_metadata
        processed_count_meta = 0
        start_time_meta = time.time()

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='MetadataScan') as executor:
            future_to_path = {
                executor.submit(_get_file_metadata, path_obj): path_obj
                for path_obj in potential_paths_to_process
            }

            pbar_meta = tqdm(
                as_completed(future_to_path), total=len(potential_paths_to_process),
                desc=f"Scanning {folder_path_obj.name[:25]}", unit="file", leave=False
            )
            for future in pbar_meta:
                filepath = future_to_path[future]
                processed_count_meta += 1
                try:
                    metadata = future.result() # _get_file_metadata from src.utils.helpers
                    if metadata and isinstance(metadata, dict):
                        # Ensure essential keys like 'filepath' are present
                        if COL_FILEPATH not in metadata or not metadata[COL_FILEPATH]:
                            metadata[COL_FILEPATH] = str(filepath.resolve()) # Add if missing
                        all_discovered_files_metadata.append(metadata)
                    else:
                        actual_log_statement('warning', f"{LOG_INS}:WARNING>>_get_file_metadata returned invalid data for {filepath}.", __file__)
                except Exception as exc:
                    actual_log_statement('error', f'{LOG_INS}:ERROR>>Metadata task generated exception for {filepath.name}: {exc}', __file__, True)
                    # Optionally add a basic error entry
                    all_discovered_files_metadata.append({COL_FILEPATH: str(filepath.resolve()), "scan_error": str(exc)})


        final_elapsed_time_meta = time.time() - start_time_meta
        final_rate_meta = processed_count_meta / final_elapsed_time_meta if final_elapsed_time_meta > 0 else 0
        actual_log_statement('info', f"{LOG_INS}:INFO>>Scan metadata gathering finished. Processed {processed_count_meta} files in {final_elapsed_time_meta:.2f}s ({final_rate_meta:.2f} files/s).", __file__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Directory scan for '{folder_path_obj}' found metadata for {len(all_discovered_files_metadata)} files.", __file__)

        return all_discovered_files_metadata

    def scan_and_update(self, directory_to_scan: Optional[Path] = None, handle_deletions: bool = True) -> bool:
        """
        Scans a directory (defaults to self.root_dir), compares files found against
        the internal DataFrame (self.df), adds new files, updates metadata for
        modified files, optionally marks missing files, and then saves the repository data file.

        Args:
            directory_to_scan (Optional[Path]): The directory to scan. Defaults to self.root_dir.
            handle_deletions (bool): If True, files in the DataFrame but not found in the scan
                                     will have their status updated to STATUS_MISSING_FROM_SCAN.

        Returns:
            bool: True if the scan and update process completed (even if no changes were made),
                  False if a critical error occurred during the process.
        """
        LOG_INS = _get_log_ins(inspect.currentframe(), self.__class__.__name__)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Starting repository scan and update.", __file__)

        if self.read_only:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Repository is in read-only mode. Scan and update aborted.", __file__)
            return False

        scan_path = (directory_to_scan or self.root_dir)
        if not scan_path:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Scan path is not defined (directory_to_scan or self.root_dir). Aborting scan.", __file__)
            return False
        scan_path = scan_path.resolve()
        actual_log_statement('info', f"{LOG_INS}:INFO>>Target directory for scan: {scan_path}", __file__)


        with self.lock:
            if self.df is None:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>DataFrame (self.df) is not loaded. Cannot scan and update.", __file__)
                return False

            # Get current known file paths from the DataFrame
            # Ensure COL_FILEPATH exists and df is not None
            if COL_FILEPATH not in self.df.columns:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>'{COL_FILEPATH}' column missing from DataFrame. Cannot proceed with scan.", __file__)
                return False
            
            # Files currently tracked in the DataFrame, restricted to the scan_path if applicable
            # This logic assumes filepaths in df are absolute and resolved.
            df_files_in_scan_path = self.df[self.df[COL_FILEPATH].str.startswith(str(scan_path))]
            repo_filepaths_in_scan_path_set = set(df_files_in_scan_path[COL_FILEPATH].tolist())


            # Perform the directory scan to get metadata for all files currently on disk
            # The existing_files_set argument to _scan_directory is more about optimizing the scan
            # itself (e.g. if _get_file_metadata is very expensive and can be skipped for unchanged known files).
            # For this reconciliation logic, we need metadata for *all* files in scan_path.
            # So, pass an empty set or make _scan_directory always fetch if not git status based.
            # The _scan_directory I refactored previously primarily fetches all if not in existing_files_set.
            # Let's assume _scan_directory returns metadata for all relevant files in folder_path_obj.
            scanned_files_metadata = self._scan_directory(scan_path, existing_files_set=set()) # Pass empty set to get all

            if not scanned_files_metadata and not repo_filepaths_in_scan_path_set: # No files on disk, no files in repo for this path
                actual_log_statement('info', f"{LOG_INS}:INFO>>No files found in scan directory '{scan_path}' and no existing entries in repo for this path. No changes needed.", __file__)
                return True


            updated_count = 0
            added_count = 0
            processed_scanned_paths = set()

            for scanned_meta in scanned_files_metadata:
                if not isinstance(scanned_meta, dict) or COL_FILEPATH not in scanned_meta:
                    actual_log_statement('warning', f"{LOG_INS}:WARNING>>Skipping invalid metadata entry from scan: {scanned_meta}", __file__)
                    continue
                
                scanned_filepath_str = str(Path(scanned_meta[COL_FILEPATH]).resolve()) # Ensure resolved
                processed_scanned_paths.add(scanned_filepath_str)

                # Check against existing DataFrame entries
                existing_entry_series = self.df[self.df[COL_FILEPATH] == scanned_filepath_str].iloc[0] if scanned_filepath_str in repo_filepaths_in_scan_path_set else None

                if existing_entry_series is not None:
                    # File exists in DataFrame, check if modified
                    # A simple mtime/size check, or hash check if available and reliable from _get_file_metadata
                    is_modified = False
                    if COL_MTIME in scanned_meta and COL_MTIME in existing_entry_series and \
                       pd.to_datetime(scanned_meta[COL_MTIME], unit='s', utc=True) > pd.to_datetime(existing_entry_series[COL_MTIME], errors='coerce', utc=True): # Assuming _get_file_metadata mtime is float timestamp
                        is_modified = True
                    if COL_SIZE in scanned_meta and COL_SIZE in existing_entry_series and \
                       scanned_meta[COL_SIZE] != existing_entry_series[COL_SIZE]:
                        is_modified = True
                    # Add hash comparison if available and reliable
                    if COL_HASH in scanned_meta and pd.notna(scanned_meta[COL_HASH]) and \
                       COL_HASH in existing_entry_series and scanned_meta[COL_HASH] != existing_entry_series[COL_HASH]:
                        is_modified = True
                    
                    if is_modified:
                        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>File '{scanned_filepath_str}' detected as modified. Updating entry.", __file__)
                        # Prepare update data, ensuring status reflects modification
                        update_payload = scanned_meta.copy()
                        update_payload[COL_STATUS] = getattr(sys.modules.get('src.data.constants', object()), 'STATUS_MODIFIED', 'modified')
                        if self.update_entry(Path(scanned_filepath_str), **update_payload):
                            updated_count += 1
                else:
                    # File is new (not in DataFrame for this scan_path)
                    actual_log_statement('debug', f"{LOG_INS}:DEBUG>>File '{scanned_filepath_str}' detected as new. Adding entry.", __file__)
                    # Prepare data, ensuring status is discovered
                    add_payload = scanned_meta.copy()
                    add_payload[COL_STATUS] = STATUS_DISCOVERED # From constants
                    if self.update_entry(Path(scanned_filepath_str), **add_payload):
                        added_count += 1
            
            actual_log_statement('info', f"{LOG_INS}:INFO>>Scan processing: Added {added_count} new files, Updated {updated_count} modified files.", __file__)

            # Handle deletions if requested
            deleted_count = 0
            if handle_deletions:
                paths_in_df_but_not_scanned = repo_filepaths_in_scan_path_set - processed_scanned_paths
                if paths_in_df_but_not_scanned:
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Found {len(paths_in_df_but_not_scanned)} files in repo but not in current scan of '{scan_path}'. Marking as missing.", __file__)
                    for missing_filepath_str in paths_in_df_but_not_scanned:
                        # Mark status as missing/deleted
                        update_payload = {
                            COL_STATUS: getattr(sys.modules.get('src.data.constants', object()), 'STATUS_MISSING_FROM_SCAN', 'missing_from_scan'),
                            COL_LAST_UPDATED: dt.now(timezone.utc) # Update timestamp
                        }
                        if self.update_entry(Path(missing_filepath_str), **update_payload):
                            deleted_count +=1
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Marked {deleted_count} files as missing from scan.", __file__)

            if updated_count > 0 or added_count > 0 or deleted_count > 0 :
                actual_log_statement('info', f"{LOG_INS}:INFO>>Changes detected. Saving repository.", __file__)
                self.save_repo() # Persist changes to self.df
            else:
                actual_log_statement('info', f"{LOG_INS}:INFO>>No changes detected from scan. Repository not saved.", __file__)

            actual_log_statement('info', f"{LOG_INS}:INFO>>Scan and update process completed for {scan_path}.", __file__)
            return True

        # Lock is released automatically

    # ... Other methods like scan_and_update, update_entry, etc. would be refactored here,
    # ensuring they use the corrected class structure and helpers.
    # For scan_and_update, it would involve calling _scan_directory, then iterating
    # results and calling update_entry or similar logic to add/update self.df, then self.save_repo().

    # Ensure all COL_* constants are used correctly and are defined (even if as placeholders).
    # Ensure all helper methods (_define_columns_schema, etc.) are consistent with each other.

# Example main execution (for testing, typically removed in a library file)
if __name__ == '__main__':
    # This ensures that if actual_log_statement is the fallback, logging is configured.
    if not _logger_module_present:
        _ensure_fallback_logging()

    LOG_INS_MAIN = f"{__name__}::main::{inspect.currentframe().f_lineno if inspect.currentframe() else 'N/A'}"
    actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Starting DataRepository example usage.", __file__)

    # Create a temporary directory for the example repository
    example_repo_dir = Path("./example_repo_test_fixed")
    if example_repo_dir.exists():
        try:
            shutil.rmtree(example_repo_dir) # Clean up from previous runs
        except OSError as e:
            actual_log_statement("warning", f"{LOG_INS_MAIN}:WARNING>>Could not remove existing test repo dir {example_repo_dir}: {e}", __file__)


    try:
        # Initialize DataRepository with a path where the repo will be created/managed
        repo = DataRepository(target_dir_for_new_repo=example_repo_dir)
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>DataRepository initialized for {example_repo_dir}", __file__)

        # Create some dummy files
        (example_repo_dir / "file1.txt").write_text("Hello content 1")
        sub = example_repo_dir / "subdir"
        sub.mkdir(exist_ok=True)
        (sub / "file2.md").write_text("# Markdown test")

        # A more complete scan_and_update might look like this:
        # 1. Scan directory for all file metadata
        # scanned_files_metadata = repo._scan_directory(repo.root_dir, existing_files_set=set()) # Initial scan might need empty set
        # 2. Iterate scanned_files_metadata, compare with repo.df, and call repo.update_entry for new/modified
        # 3. Handle deletions (files in df but not in scan)
        # 4. repo.save_repo()
        # For now, let's assume a simplified update based on adding entries if they are new
        # and then saving the main repo.

        # Example: Add/Update entries (a full scan_and_update would be more complex)
        # This is a simplified way to populate for the example.
        if repo.df is not None: # df should be initialized
            # Example: Manually preparing data as if from a scan for file1.txt
            file1_path_obj = (example_repo_dir / "file1.txt").resolve()
            file1_meta_example = {
                COL_FILEPATH: str(file1_path_obj),
                COL_FILENAME: file1_path_obj.name,
                COL_SIZE: file1_path_obj.stat().st_size,
                COL_MTIME: dt.fromtimestamp(file1_path_obj.stat().st_mtime, tz=timezone.utc),
                COL_HASH: "dummyhash1", # Replace with actual hash if calculated
                COL_STATUS: STATUS_DISCOVERED,
                COL_LAST_UPDATED: dt.now(timezone.utc)
            }
            # Using update_entry
            # repo.update_entry(file1_path_obj, **file1_meta_example) # This would save sub-repo if different

            # Simplified update of the main df for example purposes
            # In real use, update_entry or a scan_and_update method should manage this
            new_row_df = pd.DataFrame([file1_meta_example])
            new_row_df = new_row_df.astype(repo.columns_schema_dtypes, errors='ignore')
            repo.df = pd.concat([repo.df, new_row_df], ignore_index=True)
            repo.save_repo() # Save changes to the main repository file
            actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Manually added/updated file1.txt metadata and saved repo.", __file__)


        df_loaded = repo._load_repo_dataframe(repo.repo_data_file) # Test loading
        if df_loaded is not None:
            actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Loaded repo dataframe has {len(df_loaded)} rows.", __file__)
            # print(df_loaded.to_string())


        status_f1 = repo.get_status(example_repo_dir / "file1.txt")
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Status of file1.txt: {status_f1}", __file__)


        # Save and load progress
        repo.save_progress("test_process", {"step": 1, "data": "abc"})
        progress = repo.load_progress("test_process")
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Loaded progress: {progress}", __file__)


    except Exception as e:
        actual_log_statement("critical", f"{LOG_INS_MAIN}:CRITICAL>>Error during example execution: {e}", __file__, True)
    finally:
        if example_repo_dir.exists():
            try:
                shutil.rmtree(example_repo_dir)
                actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Cleaned up example directory {example_repo_dir}", __file__)
            except OSError as e:
                actual_log_statement("warning", f"{LOG_INS_MAIN}:WARNING>>Failed to cleanup {example_repo_dir}: {e}", __file__)
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Example finished.", __file__)