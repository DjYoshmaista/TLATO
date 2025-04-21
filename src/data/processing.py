# src/data/processing.py
"""
Data Processing Module

Handles data preprocessing pipelines including:
- Managing a repository of dataset files and their processing status.
- Processing raw data (text cleaning, numerical scaling).
- Tokenizing/vectorizing processed data into tensors.
Utilizes GPU acceleration (cuDF, CuPy, cuML) if available and configured.
"""
import sys
import re
from threading import Lock
from typing import Union, List, Dict, Tuple, Optional, Generator, Any
import gc
from functools import partial
import torch
import pandas as pd
import zstandard as zstd
from pathlib import Path
import hashlib
import logging
import importlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv
import time
import datetime
from datetime import timezone
import fitz  # PyMuPDF
import shutil
import multiprocessing
import json
from datetime import datetime, timezone # Import timezone
import os
import time
import io

# Import project configuration and utilities
try:
    from src.utils.config import (
        DataProcessingConfig, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, DEFAULT_DEVICE,
        PROJECT_ROOT, BASE_DATA_DIR, COMPRESSION_LEVEL, COMPRESSION_ENABLED,
        DATA_REPO_FILE # Import central repo file path
    )
    from src.utils.logger import configure_logging, log_statement
    from src.utils.gpu_switch import check_gpu_support, get_compute_backend
    from src.utils.hashing import hash_filepath, unhash_filepath, generate_data_hash
    from src.utils.compression import stream_decompress_lines, stream_compress_lines, compress_file, decompress_file
    from src.data.constants import *
    from src.data.readers import get_reader_class, FileReader, PDFReader, CSVReader, TXTReader, JSONLReader, ExcelReader
    configure_logging()
    log_statement(loglevel=str("info"), logstatement=str("Logger and other utils imported successfully."), main_logger=str(__name__))
except ImportError:
    logging.ERROR("Failed relative import in processing.py, trying absolute from src...")
    try:
        from src.utils.config import (
            DataProcessingConfig, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, DEFAULT_DEVICE,
            PROJECT_ROOT, BASE_DATA_DIR, COMPRESSION_LEVEL, COMPRESSION_ENABLED,
            DATA_REPO_FILE
        )
        from src.utils.logger import configure_logging
        from src.utils.gpu_switch import check_gpu_support
        configure_logging()
        log_statement(loglevel=str("info"), logstatement=str("Logger and other utils imported successfully."), main_logger=str(__name__))
    except ImportError as e:
        logging.error(f"CRITICAL: Failed to import core config for processing: {e}")
        class DataProcessingConfig: MAX_WORKERS = 16; REPO_FILE = './data_repository.csv.zst'; SCAN_FILE_EXTENSIONS=['.txt']; TEXT_CLEANING_REGEX=r'[^\w\s]'
        BASE_DATA_DIR = Path('./data')
        PROCESSED_DATA_DIR = BASE_DATA_DIR / 'processed'
        TOKENIZED_DATA_DIR = BASE_DATA_DIR / 'tokenized'
        DEFAULT_DEVICE = 'cpu'
        PROJECT_ROOT = Path('.')
        COMPRESSION_ENABLED = True
        COMPRESSION_LEVEL = 22
        DATA_REPO_FILE = BASE_DATA_DIR / 'data_repository.csv.zst'
try:
    from src.utils.helpers import save_dataframe_to_parquet_zst, compress_string_to_file
except ImportError:
    log_statement(loglevel='error', logstatement='Failed to import helper functions. Attempting to reload...', main_logger=str(__name__))
    from src.utils.helpers import save_dataframe_to_parquet_zst, compress_string_to_file
    log_statement(loglevel='info', logstatement='Helper functions imported successfully.', main_logger=str(__name__))
except Exception as e:
    log_statement(loglevel='error', logstatement=f'Unexpected error importing helper functions: {e}', main_logger=str(__name__))
    raise

# --- Optional GPU Library Imports ---
# (Implementation remains the same as previous version)
GPU_AVAILABLE = False; cudf = None; cp = np; CumlScaler = None; UnsupportedCUDAError = None

try:
    # Determine backend (cudf or pandas)
    compute_backend = get_compute_backend()
    if compute_backend == 'cudf':
        try:
            import cudf as pd
            log_statement(loglevel='info', logstatement="cuDF backend selected.", main_logger=str(__name__))
            IS_CUDA_AVAILABLE = True
        except ImportError:
            log_statement(loglevel='warning', logstatement="cuDF requested but not available. Falling back to pandas.", main_logger=str(__name__))
            import pandas as pd
            IS_CUDA_AVAILABLE = False
    else:
        import pandas as pd
        log_statement(loglevel='info', logstatement="Pandas backend selected.", main_logger=str(__name__))
        IS_CUDA_AVAILABLE = False

except ImportError as initial_import_err:
    log_statement(loglevel=str("error"), logstatement=str(f"GPU library 'cudf' not found ({initial_import_err}). Using CPU fallback."), main_logger=str(__name__))
    GPU_AVAILABLE = False
except Exception as initial_cudf_err:
    if UnsupportedCUDAError and isinstance(initial_cudf_err, UnsupportedCUDAError):
         log_statement(loglevel=str("warning"), logstatement=str(f"cuDF found but GPU is incompatible ({initial_cudf_err}). Using CPU fallback."), main_logger=str(__name__))
    else:
         log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during initial cuDF import/setup: {initial_cudf_err}"), main_logger=str(__name__), exc_info=True)
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    log_statement(loglevel=str("warning"), logstatement=str("Using CPU fallback (pandas/numpy)."), main_logger=str(__name__))
    cp = np # Ensure cp is numpy alias if GPU failed
    # Define cudf dummy if necessary
    if cudf is None or not GPU_AVAILABLE:
        class cudf_dummy:
             @staticmethod
             def DataFrame(*args, **kwargs): return pd.DataFrame(*args, **kwargs)
             @staticmethod
             def Series(*args, **kwargs): return pd.Series(*args, **kwargs)
             @staticmethod
             def read_csv(*args, **kwargs): return pd.read_csv(*args, **kwargs)
             @staticmethod
             def concat(*args, **kwargs): return pd.concat(*args, **kwargs)
             @staticmethod
             def from_pandas(pdf): return pdf
        cudf = cudf_dummy
    # Define CumlScaler dummy if necessary
    if CumlScaler is None or not GPU_AVAILABLE:
        try:
            from sklearn.preprocessing import StandardScaler as SklearnScaler
            log_statement(loglevel=str("warning"), logstatement=str("Using sklearn.preprocessing.StandardScaler as CPU fallback for scaling."), main_logger=str(__name__))
            class SklearnScalerWrapper:
                def __init__(self, *args, **kwargs): self._scaler = SklearnScaler(*args, **kwargs)
                def fit_transform(self, data):
                    if isinstance(data, pd.DataFrame): return pd.DataFrame(self._scaler.fit_transform(data), columns=data.columns, index=data.index)
                    elif isinstance(data, np.ndarray):
                        original_shape = data.shape
                        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
                        scaled_data = self._scaler.fit_transform(data_2d)
                        return scaled_data.reshape(original_shape)
                    else: return data
            CumlScaler = SklearnScalerWrapper
        except ImportError:
            log_statement(loglevel=str("warning"), logstatement=str("Scikit-learn not found. Using basic dummy scaler (no operation)."), main_logger=str(__name__))
            class CumlScaler_dummy:
                 def __init__(self, *args, **kwargs): pass
                 def fit_transform(self, data): return data
            CumlScaler = CumlScaler_dummy

# --- NLTK Setup (Same as previous version) ---
NLTK_AVAILABLE = False
lemmatizer = None
stop_words = set()
try:
    import nltk
    # Define potential NLTK data path (adjust if needed relative to PROJECT_ROOT)
    # NLTK_DATA_DIR = PROJECT_ROOT / 'nltk_data'
    # nltk.data.path.append(str(NLTK_DATA_DIR))
    def download_nltk_data():
        resources = {'corpora/wordnet': 'wordnet', 'corpora/stopwords': 'stopwords'}
        for path_fragment, resource_id in resources.items():
            try:
                # Try finding the resource directly first
                nltk.data.find(path_fragment)
            except LookupError:
                 try:
                     # Try finding zipped version
                     nltk.data.find(f"{path_fragment}.zip")
                 except LookupError:
                    log_statement(loglevel=str("info"), logstatement=str(f"Downloading NLTK '{resource_id}' data..."), main_logger=str(__name__))
                    nltk.download(resource_id, quiet=True)
    download_nltk_data() # Consider calling this conditionally or manually

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    log_statement(loglevel=str("info"), logstatement=str("NLTK components loaded successfully."), main_logger=str(__name__))
except ImportError: log_statement(loglevel=str("warning"), logstatement=str("NLTK library not found. Text processing limited."), main_logger=str(__name__))
except LookupError as e: log_statement(loglevel=str("warning"), logstatement=str(f"NLTK data not found ({e}). Text processing limited."), main_logger=str(__name__))
except Exception as e: log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error loading NLTK: {e}"), main_logger=str(__name__), exc_info=True)
if not NLTK_AVAILABLE:
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word # Add pos arg for compatibility
    lemmatizer = DummyLemmatizer()
    stop_words = set() # No stop words if NLTK not available

# Import readers needed for actual content processing (Placeholder)
# from src.data.readers import FileReaderFactory

# Placeholder for tokenization logic
# from src.core.tokenizer import tokenize_content # Assuming exists

# --- Data Processor ---
class DataProcessor:
    """ Scans, processes raw data, saves compressed output. """
    def __init__(self, repo_dir: str = REPO_DIR, filename: str = MAIN_REPO_FILENAME, max_workers: int | None = None):
        # --- Path Setup ---
        self.repo_dir = Path(repo_dir)
        self.repo_filepath = self.repo_dir / filename # Main repo (tracks source files)
        # Define paths for separate repositories tracking processed/tokenized outputs
        self.processed_repo_filepath = self.repo_dir / PROCESSED_REPO_FILENAME
        self.tokenized_repo_filepath = self.repo_dir / TOKENIZED_REPO_FILENAME
        self.repo_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Define base directory for processed output files
        # Using the constant from config.py
        self.output_dir = PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Define base directory for tokenized output files
        self.tokenized_output_dir = TOKENIZED_DATA_DIR
        self.tokenized_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Repository Initialization ---
        # Initialize the main repository interface (assuming DataRepository handles loading/saving)
        # Pass the main repo file path to DataRepository
        self.repo = DataRepository(repo_path=self.repo_filepath) # Pass main repo path

        # Ensure repository files exist with headers (using internal DataRepository method or helper)
        # Note: Ensure DataRepository or a helper handles Zstd compression correctly for init
        self._ensure_repo_exists(self.repo_filepath, MAIN_REPO_HEADER)
        # Optional: Initialize processed/tokenized repo files if needed by workflow
        # These might track outputs with links back to the main Designation
        # Define PROCESSED_REPO_HEADER and TOKENIZED_REPO_HEADER in constants.py if used
        # self._ensure_repo_exists(self.processed_repo_filepath, PROCESSED_REPO_HEADER)
        # self._ensure_repo_exists(self.tokenized_repo_filepath, TOKENIZED_REPO_HEADER)

        # --- Instantiate a File Reader from src/data/readers.py class as a local class variable ---
        self.text_reader = FileReader(error_handling='replace')

        # --- State Management ---
        self._next_designation = self._get_next_designation()
        self._data_hashes = self._load_existing_hashes() # For copy detection
        self._file_hashes = {} # Potentially deprecated if _data_hashes covers content checks
        self.lock = Lock() # For thread-safe operations on shared state if needed

        # --- Execution Setup ---
        resolved_max_workers = max_workers if max_workers is not None else DataProcessingConfig.MAX_WORKERS
        self.max_workers = max(1, resolved_max_workers)
        log_statement(loglevel=str("info"), logstatement=str(f"Initializing DataProcessor with max_workers={self.max_workers}"), main_logger=str(__name__))
        # Using ProcessPoolExecutor for CPU-bound tasks like reading/processing
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

        # --- Processing Components ---
        self.scaler = self._initialize_scaler() # For numerical data
        # Compile regex for text cleaning (ensure TEXT_CLEANING_REGEX is in DataProcessingConfig)
        self.cleaning_regex = None
        try:
            regex_pattern = getattr(DataProcessingConfig, 'TEXT_CLEANING_REGEX', r'[^\w\s\-\.]') # Default
            if regex_pattern:
                self.cleaning_regex = re.compile(regex_pattern)
        except re.error as re_err:
            log_statement(loglevel='error', logstatement=f"Invalid cleaning regex pattern: {regex_pattern}. Error: {re_err}", main_logger=str(__name__))
        except AttributeError:
             log_statement(loglevel='warning', logstatement="TEXT_CLEANING_REGEX not found in DataProcessingConfig. Using default.", main_logger=str(__name__))
             self.cleaning_regex = re.compile(r'[^\w\s\-\.]') # Fallback default


        # --- Assign Helper Functions (assuming they are imported standalone functions) ---
        # Note: If these are methods of another class, instantiate that class here.
        # If they are intended to be methods of *this* class, define them directly below.
        # Assuming standalone functions imported from src.utils.helpers:
        try:
            from src.utils.helpers import save_dataframe_to_parquet_zst as helper_save_parquet
            from src.utils.helpers import compress_string_to_file as helper_compress_string
            self.save_dataframe_to_parquet_zst = helper_save_parquet
            self.compress_string_to_file = helper_compress_string
            log_statement(loglevel='info', logstatement='Helper functions assigned successfully.', main_logger=str(__name__))
        except ImportError:
            log_statement(loglevel='error', logstatement='Failed to import helper functions (save_dataframe_to_parquet_zst, compress_string_to_file) from src.utils.helpers. Processing methods requiring them will fail.', main_logger=str(__name__))
            # Define dummy fallbacks to prevent AttributeError, but log errors when called
            def dummy_save_parquet(*args, **kwargs):
                log_statement(loglevel='error', logstatement="Attempted to call missing helper 'save_dataframe_to_parquet_zst'", main_logger=str(__name__))
                return False
            def dummy_compress_string(*args, **kwargs):
                 log_statement(loglevel='error', logstatement="Attempted to call missing helper 'compress_string_to_file'", main_logger=str(__name__))
                 return False
            self.save_dataframe_to_parquet_zst = dummy_save_parquet
            self.compress_string_to_file = dummy_compress_string

        log_statement(loglevel=str("info"), logstatement=str(f"DataProcessor initialized. Output dirs: Processed='{self.output_dir}', Tokenized='{self.tokenized_output_dir}'"), main_logger=str(__name__))

    def _classify_data(self, df: pd.DataFrame, file_path_hint: Path) -> str:
        """
        Analyzes the content of a DataFrame to classify the data type.
        Implements logic based on rules #1, #2, #3.

        Args:
            df (pd.DataFrame): The DataFrame read from the file.
            file_path_hint (Path): Original file path, used for logging/context.

        Returns:
            str: A classification constant (e.g., TYPE_TEXTUAL, TYPE_NUMERICAL).
        """
        main_logger_name = str(__name__) # Logger name for this module
        log_statement(loglevel="debug", logstatement=f"Starting data classification for {file_path_hint.name}", main_logger=main_logger_name)

        if df is None or df.empty:
            log_statement(loglevel="warning", logstatement=f"Cannot classify empty DataFrame from {file_path_hint.name}", main_logger=main_logger_name)
            return TYPE_EMPTY

        num_rows, num_cols = df.shape
        total_cells = num_rows * num_cols
        if total_cells == 0:
             log_statement(loglevel="warning", logstatement=f"Cannot classify zero-cell DataFrame from {file_path_hint.name}", main_logger=main_logger_name)
             return TYPE_EMPTY

        numeric_cells = 0
        text_cells = 0
        potential_tensor_cells = 0
        potential_subword_cells = 0
        # Heuristic thresholds (adjust as needed)
        NUMERICAL_THRESHOLD = 0.7 # More than 70% of non-empty cells are numeric
        TOKEN_THRESHOLD = 0.9     # More than 90% of cells look like tokens/numerics
        AVG_SUBWORD_LEN_THRESHOLD = 10 # Average length for suspecting subwords


        # --- Analyze DataFrame Content ---
        all_data_numeric = True # Flag for Rule #2 check
        all_non_numeric_short = True # Flag for Rule #3 check
        total_analyzed_cells = 0
        avg_str_len_sum = 0
        str_cell_count = 0

        for col in df.columns:
            # Attempt numeric conversion (non-destructive)
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            num_numeric_in_col = numeric_series.notna().sum()
            num_non_numeric_in_col = df[col].notna().sum() - num_numeric_in_col

            numeric_cells += num_numeric_in_col
            text_cells += num_non_numeric_in_col # Count non-numeric as text initially
            total_analyzed_cells += df[col].notna().sum()

            # Check for numerical tokenized data
            # If a column has *any* non-numeric data after coercion, the whole DataFrame isn't purely numerical tokens
            if num_non_numeric_in_col > 0:
                all_data_numeric = False

            # Check for subword tokens - Analyze non-numeric cells
            for item in df[col][numeric_series.isna()]: # Iterate only over non-numeric items
                 if isinstance(item, str):
                     str_len = len(item)
                     avg_str_len_sum += str_len
                     str_cell_count += 1
                     # Basic check: are non-numeric strings short and lack spaces?
                     if str_len > AVG_SUBWORD_LEN_THRESHOLD or ' ' in item:
                         all_non_numeric_short = False
                 elif item is not None: # Handle non-string, non-numeric types if needed
                     all_non_numeric_short = False

        # --- Classification Logic ---
        classification = TYPE_UNKNOWN # Default

        # Rule #2: Check for purely numerical tokenized data
        if all_data_numeric and total_analyzed_cells > 0:
            # Further checks could involve inspecting structure (e.g., nested lists)
            # For now, assume if all convertible cells are numeric, it's tokenized.
            log_statement(loglevel="info", logstatement=f"Classified {file_path_hint.name} as {TYPE_TOKENIZED_NUMERICAL} (all convertible cells are numeric)", main_logger=main_logger_name)
            classification = TYPE_TOKENIZED_NUMERICAL

        # Rule #3: Check for subword tokenized data (heuristic)
        elif all_non_numeric_short and text_cells > 0 and str_cell_count > 0:
            # If all non-numeric cells are short strings, suspect subwords
            avg_len = avg_str_len_sum / str_cell_count if str_cell_count > 0 else 0
            if avg_len < AVG_SUBWORD_LEN_THRESHOLD:
                 # Consider if ratio of text_cells to total is high enough
                 if (text_cells / total_analyzed_cells if total_analyzed_cells else 0) > TOKEN_THRESHOLD:
                    log_statement(loglevel="info", logstatement=f"Classified {file_path_hint.name} as {TYPE_TOKENIZED_SUBWORD} (heuristic: high ratio of short non-numeric strings, avg len {avg_len:.2f})", main_logger=main_logger_name)
                    classification = TYPE_TOKENIZED_SUBWORD

        # Rule #1 (partially): Check for predominantly numerical tabular data
        if classification == TYPE_UNKNOWN and total_analyzed_cells > 0: # Check only if not already classified
             numeric_ratio = numeric_cells / total_analyzed_cells
             if numeric_ratio >= NUMERICAL_THRESHOLD:
                  log_statement(loglevel="info", logstatement=f"Classified {file_path_hint.name} as {TYPE_NUMERICAL} (numeric ratio {numeric_ratio:.2f} >= {NUMERICAL_THRESHOLD})", main_logger=main_logger_name)
                  classification = TYPE_NUMERICAL

        # Rule #1 (partially): Fallback to Textual
        if classification == TYPE_UNKNOWN and text_cells > 0: # If still unknown and has text cells
             # Assume it's textual if it passed the reader's content validation earlier
             log_statement(loglevel="info", logstatement=f"Classified {file_path_hint.name} as {TYPE_TEXTUAL} (fallback, contains text cells)", main_logger=main_logger_name)
             classification = TYPE_TEXTUAL

        if classification == TYPE_UNKNOWN:
             log_statement(loglevel="warning", logstatement=f"Could not reliably classify data type for {file_path_hint.name}. Defaulting to {TYPE_UNKNOWN}.", main_logger=main_logger_name)

        return classification        

    def _get_file_hash(self, filepath: Path) -> str:
        """Generates a SHA-256 hash for the file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    def _get_file_size(self, filepath: Path) -> int:    
        """Returns the size of the file in bytes."""
        return filepath.stat().st_size
    def _get_file_modification_time(self, filepath: Path) -> datetime:
        """Returns the last modification time of the file."""
        return datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
    def _get_file_creation_time(self, filepath: Path) -> datetime:
        """Returns the creation time of the file."""
        return datetime.fromtimestamp(filepath.stat().st_ctime, tz=timezone.utc)
    def _get_file_extension(self, filepath: Path) -> str:
        """Returns the file extension."""
        return filepath.suffix.lower().strip('.')
    def _get_file_name(self, filepath: Path) -> str:
        """Returns the file name without extension."""
        return filepath.stem
    def _get_file_path(self, filepath: Path) -> str:
        """Returns the file path."""
        return str(filepath.resolve())
    def _get_file_status(self, filepath: Path) -> str:
        """Returns the file status."""
        return self.repo.get_file_status(filepath)
    def _get_file_error_message(self, filepath: Path) -> str:
        """Returns the error message for the file."""
        return self.repo.get_file_error_message(filepath)
    def _get_file_processed_path(self, filepath: Path) -> str:
        """Returns the processed path for the file."""
        return self.repo.get_file_processed_path(filepath)
    def _get_file_tokenized_path(self, filepath: Path) -> str:
        """Returns the tokenized path for the file."""
        return self.repo.get_file_tokenized_path(filepath)
    def _get_file_tokenized(self, filepath: Path) -> str:
        """Returns the tokenized data for the file."""
        return self.repo.get_file_tokenized(filepath)
    def _get_file_processed(self, filepath: Path) -> str:
        """Returns the processed data for the file."""
        return self.repo.get_file_processed(filepath)
    def _get_file_base_dir(self, filepath: Path) -> str:
        """Returns the base directory for the file."""
        return self.repo.get_file_base_dir(filepath)
    def _get_file_hash(self, filepath: Path) -> str:
        """Returns the hash for the file."""
        return self.repo.get_file_hash(filepath)
    
    def scan_data_directory(self):
        """Initiates a recursive scan of the base data directory."""
        # Use the configured base directory path
        log_statement(loglevel=str("info"), logstatement=str(f"Starting repository scan using base directory: {BASE_DATA_DIR}"), main_logger=str(__name__))
        self.repo.scan_and_update(BASE_DATA_DIR)
        
    def __del__(self):
        """Ensures proper cleanup of the executor on object deletion."""
        if hasattr(self, 'executor') and self.executor:
            try:
                log_statement(loglevel=str("info"), logstatement=str("Shutting down DataProcessor executor..."), main_logger=str(__name__)) 
                self.executor.shutdown(wait=True)
                # Wait for tasks on shutdown
                log_statement(loglevel=str("info"), logstatement=str("DataProcessor executor shut down."), main_logger=str(__name__))
            except Exception as e: 
                log_statement(loglevel=str("error"), logstatement=str(f"Error shutting down executor: {e}"), main_logger=str(__name__))

    def _label_text_semantically(self, text_content: str, file_path_hint: Path, model_name: str = "mistral:latest") -> Optional[Dict]:
        """
        Uses Ollama/Mistral to label text sections hierarchically.

        Args:
            text_content (str): The text content to label.
            file_path_hint (Path): Original file path for context/logging.
            model_name (str): The Ollama model to use.

        Returns:
            Optional[Dict]: A dictionary representing the structured/labeled text
                           (e.g., {"Section Title": ["Para1", "Para2"]}), or None on failure.
        """
        main_logger_name = str(__name__)
        log_statement(loglevel="info", logstatement=f"Attempting semantic labeling for {file_path_hint.name} using Ollama model '{model_name}'...", main_logger=main_logger_name)

        # --- Pre-check for Ollama Library ---
        try:
            import ollama
        except ImportError:
            log_statement(loglevel="error", logstatement="Ollama library not installed. Cannot perform semantic labeling. Run 'pip install ollama'.", main_logger=main_logger_name)
            return None

        # --- Limit Input Size (Important!) ---
        # LLMs have context limits. Send manageable chunks or summarize first if needed.
        MAX_INPUT_CHARS = 8000 # Example limit, adjust based on model/memory
        if len(text_content) > MAX_INPUT_CHARS:
            log_statement(loglevel="warning", logstatement=f"Input text for {file_path_hint.name} exceeds {MAX_INPUT_CHARS} chars ({len(text_content)}). Truncating for labeling.", main_logger=main_logger_name)
            text_to_label = text_content[:MAX_INPUT_CHARS]
            # TODO: Consider more sophisticated chunking/summarization for very long texts
        else:
            text_to_label = text_content

        # --- Construct Prompt ---
        # This prompt needs careful engineering for reliable JSON output.
        prompt = f"""Analyze the following text and structure it hierarchically.
Identify distinct chapters or main sections based on content breaks or implicit headings.
Within each section, identify distinct paragraphs (separated by significant whitespace like double newlines).
Output the result ONLY as a valid JSON object. The JSON object should have keys representing the chapter/section titles (use "Section 1", "Section 2", etc. if no clear titles exist). The value for each key should be a JSON array of strings, where each string is a single paragraph from that section.
Do not include any explanations or introductory text outside the JSON object.

Text to analyze:
\"\"\"
{text_to_label}
\"\"\"

JSON Output:
"""

        # --- Call Ollama ---
        try:
            log_statement(loglevel="debug", logstatement=f"Sending request to Ollama for {file_path_hint.name}...", main_logger=main_logger_name)
            response = ollama.generate(model=model_name, prompt=prompt, format="json") # Request JSON format explicitly if supported

            # --- Process Response ---
            if not response or 'response' not in response:
                 log_statement(loglevel="error", logstatement=f"Ollama response invalid or empty for {file_path_hint.name}.", main_logger=main_logger_name)
                 return None

            raw_response_text = response.get('response', '').strip()
            log_statement(loglevel="debug", logstatement=f"Ollama raw response snippet for {file_path_hint.name}: {raw_response_text[:250]}...", main_logger=main_logger_name)

            # Attempt to parse the response string as JSON
            try:
                # Sometimes the model might still wrap the JSON in backticks or text
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                     # Assume the response is directly the JSON string if format='json' worked
                     json_str = raw_response_text

                structured_data = json.loads(json_str)

                # Basic validation: is it a dictionary? Are values lists of strings?
                if not isinstance(structured_data, dict):
                     raise ValueError("Parsed JSON is not a dictionary.")
                for key, value in structured_data.items():
                    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
                        raise ValueError(f"Invalid structure for section '{key}'. Value must be a list of strings.")

                log_statement(loglevel="info", logstatement=f"Successfully received and parsed semantic structure from Ollama for {file_path_hint.name}.", main_logger=main_logger_name)
                return structured_data

            except json.JSONDecodeError as jde:
                log_statement(loglevel="error", logstatement=f"Failed to parse Ollama response as JSON for {file_path_hint.name}: {jde}. Response: {raw_response_text[:500]}", main_logger=main_logger_name)
                return None
            except ValueError as ve:
                 log_statement(loglevel="error", logstatement=f"Ollama response JSON structure validation failed for {file_path_hint.name}: {ve}", main_logger=main_logger_name)
                 return None

        except Exception as e:
            # Catch connection errors, model not found errors, etc.
            log_statement(loglevel="error", logstatement=f"Error during Ollama API call for {file_path_hint.name}: {e}", main_logger=main_logger_name, exc_info=True)
            return None

    def process_all(self, base_dir_filter: Optional[Path] = None, statuses_to_process=('discovered', 'error')):
        """Processes files matching status, optionally filtered by base_dir."""
        files_to_process = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
        if not files_to_process:
            log_statement(loglevel=str("info"), logstatement=str(f"No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] found to process."), main_logger=str(__name__))
            return
        log_statement(loglevel=str("info"), logstatement=str(f"Starting processing for {len(files_to_process)} files [base_dir: {base_dir_filter}]."), main_logger=str(__name__))
        futures = [self.executor.submit(self._process_file, f_path) for f_path in files_to_process]
        with tqdm(total=len(futures), desc=f"Processing Files [{base_dir_filter.name if base_dir_filter else 'All'}]") as pbar:
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: log_statement(loglevel=str("error"), logstatement=str(f"Error retrieving result from future: {e}"), main_logger=str(__name__), exc_info=True)
                finally: pbar.update(1)
        self.repo.save()
        log_statement(loglevel=str("info"), logstatement=str("File processing complete."), main_logger=str(__name__))
        
    def _process_file(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single file based on metadata from the repository: reads content
        using an appropriate reader, classifies the data, applies processing based
        on classification, saves output, and updates repository metadata.

        Args:
            file_info (Dict[str, Any]): Dictionary containing file metadata from the repo
                                        (requires COL_FILEPATH, others like Designation).

        Returns:
            Optional[Dict[str, Any]]: An updated metadata dictionary reflecting the outcome.
                                      Includes processing status (e.g., STATUS_PROCESSED, STATUS_ERROR),
                                      error messages, classification, and output details (path, hash)
                                      if successful. Returns the dict even on failure.
                                      Returns None only on critical internal error before processing starts.
        """
        # --- 1. Pre-checks and Path Setup ---
        main_logger_name = str(__name__) # Logger name

        # Retrieve essential info from the input dictionary
        absolute_file_path_str = file_info.get(COL_FILEPATH, None)
        designation = file_info.get(COL_DESIGNATION, 'N/A') # Get designation for logging

        # Prepare updated_info dict to track changes and status during processing
        updated_info = file_info.copy()
        updated_info[COL_STATUS] = STATUS_PROCESSING # Mark as attempting processing
        updated_info[COL_ERROR] = '' # Clear any previous error message

        if not absolute_file_path_str:
            log_statement(loglevel='error', logstatement=f"Missing file path in file_info for Designation: {designation}", main_logger=main_logger_name)
            updated_info[COL_STATUS] = STATUS_ERROR
            updated_info[COL_ERROR] = "Missing file path in input metadata"
            self.repo.update_entry(Path(absolute_file_path_str) if absolute_file_path_str else f"Designation_{designation}", **updated_info) # Update repo with error
            return updated_info # Return info dict with error status

        try:
            # Convert string path to Path object for easier handling
            absolute_file_path = Path(absolute_file_path_str)
            file_ext = absolute_file_path.suffix.lower().strip('.') # Get clean extension

            # Construct relative path for mirrored output structure
            # Assumption: File paths in repo are absolute or consistently relative.
            # If paths are consistently relative to a known base (e.g., self.repo.base_dir), use that.
            # If paths are absolute, create a relative path suitable for the output structure.
            # Example using PROJECT_ROOT as base (adjust if paths stored differently):
            try:
                # Try making path relative to PROJECT_ROOT
                output_path_relative = absolute_file_path.relative_to(PROJECT_ROOT)
            except ValueError:
                # If not relative to project root (e.g., different drive), use full path parts
                log_statement(loglevel='warning', logstatement=f"File path {absolute_file_path} not relative to PROJECT_ROOT {PROJECT_ROOT}. Using full path parts for output structure.", main_logger=main_logger_name)
                # Remove drive/anchor for output structure
                output_path_relative = Path(*absolute_file_path.parts[1:]) # Skip drive/anchor

            # Construct base output path using the *processed* data directory (self.output_dir)
            output_path = self.output_dir / output_path_relative # Base path (suffix added by processing func)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

            log_statement(loglevel='debug', logstatement=f"Starting processing Designation {designation}: {absolute_file_path}", main_logger=main_logger_name)

        except Exception as path_err:
             log_statement(loglevel='critical', logstatement=f"CRITICAL error setting up paths for {absolute_file_path_str} (Designation: {designation}): {path_err}", main_logger=main_logger_name, exc_info=True)
             updated_info[COL_STATUS] = STATUS_ERROR
             updated_info[COL_ERROR] = f"Path setup error: {path_err}"
             # Attempt to update repo even with path error if possible
             self.repo.update_entry(Path(absolute_file_path_str) if absolute_file_path_str else f"Designation_{designation}", **updated_info)
             return updated_info


        # --- 2. Reader Discovery & Instantiation ---
        try:
            # Get the appropriate reader class based on the file extension
            ReaderClass = get_reader_class(file_ext) # Assumes readers.py provides this

            if ReaderClass is None:
                log_statement(loglevel='warning', logstatement=f"No reader found for file type '.{file_ext}'. Skipping {absolute_file_path.name} (Designation: {designation})", main_logger=main_logger_name)
                updated_info[COL_STATUS] = STATUS_ERROR
                updated_info[COL_ERROR] = f"Unsupported file type: .{file_ext}"
                self.repo.update_entry(absolute_file_path, **updated_info) # Update repo
                return updated_info # Return info with error status

            # Instantiate the reader
            reader = ReaderClass(absolute_file_path)
            log_statement(loglevel='debug', logstatement=f"Using reader {type(reader).__name__} for {absolute_file_path.name} (Designation: {designation})", main_logger=main_logger_name)

        except FileNotFoundError:
             # This might happen if file disappears between repo scan and processing start
             log_statement(loglevel='error', logstatement=f"File not found when creating reader for {absolute_file_path.name} (Designation: {designation})", main_logger=main_logger_name)
             updated_info[COL_STATUS] = STATUS_ERROR
             updated_info[COL_ERROR] = "File not found during processing start"
             self.repo.update_entry(absolute_file_path, **updated_info)
             return updated_info
        except Exception as reader_init_err:
             log_statement(loglevel='error', logstatement=f"Error initializing reader {ReaderClass.__name__ if 'ReaderClass' in locals() else 'Unknown'} for {absolute_file_path.name} (Designation: {designation}): {reader_init_err}", main_logger=main_logger_name, exc_info=True)
             updated_info[COL_STATUS] = STATUS_ERROR
             updated_info[COL_ERROR] = f"Reader init failed: {reader_init_err}"
             self.repo.update_entry(absolute_file_path, **updated_info)
             return updated_info

        # --- 3. Read Data ---
        read_data = None
        try:
            # Assume reader.read() returns data (e.g., DataFrame, text content)
            # or None/raises Exception if reading fails.
            read_data = reader.read()

            # Check if reader returned valid data (can be empty DataFrame, but not None)
            if read_data is None:
                # Reader should log specifics internally if possible
                log_statement(loglevel='warning', logstatement=f"Reader {type(reader).__name__} returned None for {absolute_file_path.name} (Designation: {designation}). Setting status to Error.", main_logger=main_logger_name)
                updated_info[COL_STATUS] = STATUS_ERROR
                updated_info[COL_ERROR] = "Reader returned None or content validation failed"
                self.repo.update_entry(absolute_file_path, **updated_info)
                return updated_info

        except Exception as read_err:
            log_statement(loglevel='error', logstatement=f"Reader {type(reader).__name__} failed for {absolute_file_path.name} (Designation: {designation}): {read_err}", main_logger=main_logger_name, exc_info=True)
            updated_info[COL_STATUS] = STATUS_ERROR
            updated_info[COL_ERROR] = f"Reader failed: {read_err}"
            self.repo.update_entry(absolute_file_path, **updated_info)
            return updated_info

        # --- 4. Classify Data Content ---
        data_classification = TYPE_UNKNOWN
        try:
            # Ensure _classify_data method exists
            if not hasattr(self, '_classify_data'):
                raise AttributeError("_classify_data method not found in DataProcessor.")

            # Classify based on the actual data read from the file
            # Pass read_data (which could be DataFrame, string, etc.)
            data_classification = self._classify_data(read_data, absolute_file_path)
            # Store classification result in the info dict
            updated_info['data_classification'] = data_classification # Store intermediate classification

            if data_classification == TYPE_EMPTY:
                log_statement(loglevel="warning", logstatement=f"Data classified as EMPTY for {absolute_file_path.name} (Designation: {designation}). Skipping processing.", main_logger=main_logger_name)
                updated_info[COL_STATUS] = STATUS_ERROR # Or a 'SKIPPED_EMPTY' status
                updated_info[COL_ERROR] = "Empty or unclassifiable content after read"
                self.repo.update_entry(absolute_file_path, **updated_info)
                return updated_info # Stop processing for this file

        except AttributeError as attr_err:
             log_statement(loglevel='error', logstatement=str(attr_err), main_logger=main_logger_name)
             updated_info[COL_STATUS] = STATUS_ERROR
             updated_info[COL_ERROR] = f"Internal error: {attr_err}"
             self.repo.update_entry(absolute_file_path, **updated_info)
             return updated_info
        except Exception as classify_err:
             log_statement(loglevel='error', logstatement=f"Error during data classification for {absolute_file_path.name} (Designation: {designation}): {classify_err}", main_logger=main_logger_name, exc_info=True)
             updated_info[COL_STATUS] = STATUS_ERROR
             updated_info[COL_ERROR] = f"Classification failed: {classify_err}"
             self.repo.update_entry(absolute_file_path, **updated_info)
             return updated_info # Stop processing if classification fails critically

        # --- 5. Process Data Based on Classification ---
        result = None           # Stores metadata dict from successful processing func
        processed_flag = False  # Track if a specific processing path was attempted

        try:
            # --- Check if the data type is TEXTUAL ---
            if data_classification == TYPE_TEXTUAL:
                log_statement(loglevel='debug', logstatement=f"Processing Designation {designation} ({absolute_file_path.name}) as TEXTUAL", main_logger=main_logger_name)
                if hasattr(self, '_process_textual_data'):
                    # Output structured JSON (semantic labeling if enabled)
                    # Output suffix determined within _process_textual_data (.json.zst)
                    result = self._process_textual_data(read_data, absolute_file_path, output_path)
                    processed_flag = True
                else: raise AttributeError("_process_textual_data method not found.")

            # --- Check if the data type is NUMERICAL ---
            elif data_classification == TYPE_NUMERICAL:
                log_statement(loglevel='debug', logstatement=f"Processing Designation {designation} ({absolute_file_path.name}) as NUMERICAL", main_logger=main_logger_name)
                if hasattr(self, '_process_numerical_data'):
                     # Pass read_data (expected to be DataFrame by _process_numerical_data)
                     # Output suffix determined within _process_numerical_data (.parquet.zst)
                    result = self._process_numerical_data(read_data, absolute_file_path, output_path)
                    processed_flag = True
                else: raise AttributeError("_process_numerical_data method not found.")

            # --- Check if data is already considered tokenized (subword) ---
            elif data_classification == TYPE_TOKENIZED_SUBWORD:
                log_statement(loglevel='info', logstatement=f"Data for Designation {designation} ({absolute_file_path.name}) classified as {TYPE_TOKENIZED_SUBWORD}. Applying passthrough/validation.", main_logger=main_logger_name)
                if hasattr(self, 'save_dataframe_to_parquet_zst'):
                    save_path = output_path.with_suffix('.parquet.zst') # Standardize output format
                    # Ensure the 'read_data' variable holds the data to be saved
                    if self.save_dataframe_to_parquet_zst(read_data, save_path): # Use read_data here
                        output_hash = generate_data_hash(str(save_path))
                        relative_output_path = save_path.relative_to(self.output_dir) # Use self.output_dir
                        result = {"processed_path": str(relative_output_path), COL_DATA_HASH: output_hash}
                        log_statement(loglevel='debug', logstatement=f"Saved pre-tokenized (subword) data to {save_path}", main_logger=main_logger_name)
                    else:
                        updated_info[COL_ERROR] = "Failed to save pre-tokenized (subword) data using helper"
                        log_statement(loglevel='error', logstatement=f"Helper save_dataframe_to_parquet_zst failed for {save_path}", main_logger=main_logger_name)
                        result = None # Indicate failure
                else: raise AttributeError("save_dataframe_to_parquet_zst helper method not found.")
                processed_flag = True # Mark classification type as handled

            # --- Check if data is already considered tokenized numerically ---
            elif data_classification == TYPE_TOKENIZED_NUMERICAL:
                log_statement(loglevel='info', logstatement=f"Data for Designation {designation} ({absolute_file_path.name}) classified as {TYPE_TOKENIZED_NUMERICAL}. Applying passthrough/validation.", main_logger=main_logger_name)
                if hasattr(self, 'save_dataframe_to_parquet_zst'):
                    save_path = output_path.with_suffix('.parquet.zst') # Standardize output format
                    # Ensure the 'read_data' variable holds the data to be saved
                    if self.save_dataframe_to_parquet_zst(read_data, save_path): # Use read_data here
                        output_hash = generate_data_hash(str(save_path))
                        relative_output_path = save_path.relative_to(self.output_dir) # Use self.output_dir
                        result = {"processed_path": str(relative_output_path), COL_DATA_HASH: output_hash}
                        log_statement(loglevel='debug', logstatement=f"Saved pre-tokenized (numerical) data to {save_path}", main_logger=main_logger_name)
                    else:
                        updated_info[COL_ERROR] = "Failed to save pre-tokenized (numerical) data using helper"
                        log_statement(loglevel='error', logstatement=f"Helper save_dataframe_to_parquet_zst failed for {save_path}", main_logger=main_logger_name)
                        result = None # Indicate failure
                else: raise AttributeError("save_dataframe_to_parquet_zst helper method not found.")
                processed_flag = True # Mark classification type as handled

            # --- Handle Already Processed/Tokenized Types (Metadata Update) ---
            # Example: Update status if classification identifies it as already done
            elif data_classification in [TYPE_PROCESSED_JSONL, TYPE_TOKENIZED_TEXT]: # Add other relevant types
                 log_statement(loglevel='info', logstatement=f"File Designation {designation} ({absolute_file_path.name}) classified as already processed/tokenized ({data_classification}). Updating status.", main_logger=main_logger_name)
                 # Determine appropriate final status based on classification
                 new_status = STATUS_PROCESSED if data_classification == TYPE_PROCESSED_JSONL else STATUS_TOKENIZED
                 if updated_info.get(COL_STATUS) != new_status:
                     updated_info[COL_STATUS] = new_status
                     log_statement(loglevel='debug', logstatement=f"Status updated to {new_status} based on classification.", main_logger=main_logger_name)
                 else:
                      log_statement(loglevel='debug', logstatement=f"Status already matches classification ({new_status}).", main_logger=main_logger_name)

                 # No actual processing needed here, but set flags to indicate handled
                 processed_flag = True
                 result = {} # Indicate success (status update is the action)

            # --- Fallback for UNKNOWN Classification ---
            elif data_classification == TYPE_UNKNOWN:
                log_statement(loglevel='warning', logstatement=f"Data classification UNKNOWN for Designation {designation} ({absolute_file_path.name}). Attempting text processing fallback.", main_logger=main_logger_name)
                if hasattr(self, '_process_textual_data'):
                    result = self._process_textual_data(read_data, absolute_file_path, output_path)
                    processed_flag = True # Consider it handled via fallback
                else: raise AttributeError("Fallback method _process_textual_data not found.")

            # --- Handle other classifications (e.g., IMAGE, AUDIO) ---
            # Add elif blocks here for other data types if processing logic exists
            # elif data_classification == TYPE_IMAGE:
            #    if hasattr(self, '_process_image_data'):
            #        result = self._process_image_data(read_data, ...)
            #        processed_flag = True
            #    else: raise AttributeError("_process_image_data method not found.")

            else:
                # If classification is known but no processing block exists
                log_statement(loglevel='error', logstatement=f"No processing logic defined for data classification '{data_classification}' for Designation {designation} ({absolute_file_path.name}).", main_logger=main_logger_name)
                updated_info[COL_ERROR] = f"No processing logic for classification '{data_classification}'"
                result = None # Indicate failure

        except AttributeError as attr_err:
            log_statement(loglevel='error', logstatement=f"Missing processing component for Designation {designation} ({absolute_file_path.name}): {attr_err}", main_logger=main_logger_name)
            updated_info[COL_ERROR] = f"Internal error: {attr_err}"
            result = None # Indicate failure
            # processed_flag might be True or False depending on where the error occurred
        except Exception as process_err:
            log_statement(loglevel='error', logstatement=f"Error during '{data_classification}' processing for Designation {designation} ({absolute_file_path.name}): {process_err}", main_logger=main_logger_name, exc_info=True)
            updated_info[COL_ERROR] = f"Processing error ({data_classification}): {process_err}"
            result = None # Indicate failure
            processed_flag = True # Mark as attempted if error happened within a block

        # --- 6. Check Result and Finalize Status ---
        if processed_flag and result is not None and isinstance(result, dict):
            # Processing function succeeded and returned metadata
            updated_info.update(result) # Merge results (e.g., processed_path, output_hash)
            updated_info[COL_STATUS] = STATUS_PROCESSED # Set final status to Processed
            updated_info[COL_ERROR] = '' # Clear error on success
            # Store the final classification along with the successful status
            updated_info['final_classification'] = data_classification
            log_statement(loglevel='info', logstatement=f"Successfully processed Designation {designation} ({absolute_file_path.name}) as {data_classification} -> {result.get('processed_path', 'N/A')}", main_logger=main_logger_name)
        else:
            # Processing failed or no specific path was taken/found
            if processed_flag: # An attempt was made but failed (result is None or not dict)
                log_statement(loglevel='error', logstatement=f"Processing failed for Designation {designation} ({absolute_file_path.name}, Classification: {data_classification}). See previous logs.", main_logger=main_logger_name)
                # Keep existing error message if already set by specific processing func
                if not updated_info.get(COL_ERROR):
                    updated_info[COL_ERROR] = f"Processing function ({data_classification}) failed or returned invalid result"
            else: # No processing path was attempted or matched
                log_statement(loglevel='error', logstatement=f"No processing attempted or suitable path found for Designation {designation} ({absolute_file_path.name}, Classification: {data_classification}).", main_logger=main_logger_name)
                if not updated_info.get(COL_ERROR):
                    updated_info[COL_ERROR] = f"No processing logic available or matched for classification '{data_classification}'"

            # Store classification even on error, set status to Error
            updated_info['final_classification'] = data_classification
            updated_info[COL_STATUS] = STATUS_ERROR


        # --- 7. Update Repository ---
        # Always update the repository with the final status and metadata
        try:
            # Use the absolute path of the source file as the key for updating
            self.repo.update_entry(absolute_file_path, **updated_info)
            log_statement(loglevel='debug', logstatement=f"Repository updated for Designation {designation} with status '{updated_info[COL_STATUS]}'.", main_logger=main_logger_name)
        except Exception as repo_update_err:
             log_statement(loglevel='critical', logstatement=f"CRITICAL: Failed to update repository for Designation {designation} ({absolute_file_path.name}) after processing attempt: {repo_update_err}", main_logger=main_logger_name, exc_info=True)
             # The processing might have succeeded, but the status is not saved.

        # Return the final metadata dictionary
        return updated_info

    def _process_textual_data(self, df: pd.DataFrame, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Processes textual data: performs cleaning, optional semantic labeling,
        saves structured output (JSON), and returns metadata.

        Args:
            df (pd.DataFrame): DataFrame containing the textual data.
            input_path (Path): Absolute path to the original input file (for context).
            output_path (Path): Absolute path for the structured JSON output file (.json.zst).


        Returns:
            Optional[Dict]: Dictionary with 'processed_path' (relative) and 'output_hash' on success, else None.
        """
        main_logger_name = str(__name__)
        log_statement(loglevel="debug", logstatement=f"Starting _process_textual_data for: {input_path.name}", main_logger=main_logger_name)
        try:
            # --- Consolidate Text from DataFrame ---
            # (Combine relevant text columns into a single string)
            text_content = ""
            # Example: combine all string columns, separated by newlines
            # Ensure only object/string dtypes are selected
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            if not string_cols.empty:
                 for col in string_cols:
                     # Ensure proper handling of potential NaN/None before astype(str) and str.cat
                     text_content += df[col].fillna('').astype(str).str.cat(sep='\n') + "\n"
                 text_content = text_content.strip()
            else:
                 # Handle case where DataFrame has no string columns
                 log_statement(loglevel="warning", logstatement=f"No string columns found in DataFrame for {input_path.name} to consolidate text.", main_logger=main_logger_name)
                 # Attempt to convert all columns to string as a fallback?
                 # Or return None? Let's try converting all.
                 for col in df.columns:
                      text_content += df[col].fillna('').astype(str).str.cat(sep='\n') + "\n"
                 text_content = text_content.strip()


            if not text_content:
                 log_statement(loglevel="warning", logstatement=f"No text content extracted from DataFrame for {input_path.name}", main_logger=main_logger_name)
                 return None

            # --- Basic Cleaning (Apply before labeling) ---
            cleaned_text = text_content.lower()
            if hasattr(self, 'cleaning_regex') and self.cleaning_regex: # Check if regex exists
                 # Ensure cleaning_regex is compiled, ideally in __init__
                 if isinstance(self.cleaning_regex, str): # Compile if it's still a string
                      try: self.cleaning_regex = re.compile(self.cleaning_regex)
                      except re.error as re_err: log_statement(loglevel='error', logstatement=f"Invalid cleaning regex: {re_err}", main_logger=main_logger_name); self.cleaning_regex = None # Disable if invalid

                 if hasattr(self.cleaning_regex, 'sub'): # Check if valid compiled regex
                      cleaned_text = self.cleaning_regex.sub('', cleaned_text)

                 # General whitespace normalization
                 cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            # Optional: NLTK processing could go here too

            # --- <<< Semantic Labeling (Instruction #12) >>> ---
            structured_data = None
            # Add a config flag to enable/disable this expensive step
            # Assume self.config exists and holds configuration
            semantic_labeling_enabled = False # Default to disabled
            if hasattr(self, 'config') and hasattr(self.config, 'ENABLE_SEMANTIC_LABELING'):
                semantic_labeling_enabled = self.config.ENABLE_SEMANTIC_LABELING

            if semantic_labeling_enabled:
                 # Ensure the labeling method exists
                 if hasattr(self, '_label_text_semantically'):
                     structured_data = self._label_text_semantically(cleaned_text, input_path)
                 else:
                     log_statement(loglevel="warning", logstatement=f"Semantic labeling enabled but _label_text_semantically method not found in DataProcessor.", main_logger=main_logger_name)


            # --- Prepare Output Data ---
            if structured_data:
                # Output is the structured dictionary from Ollama
                output_data_to_save = structured_data
            else:
                # Fallback: Output the cleaned text in a simple structure
                # (or just the raw cleaned text if preferred, adjust saving method)
                log_statement(loglevel="debug", logstatement=f"No structured data from labeling for {input_path.name}, saving cleaned text.", main_logger=main_logger_name)
                # Adhere to Rule #11 (tabular/JSON output)
                # Simple JSON structure containing the cleaned text
                output_data_to_save = {"cleaned_text": cleaned_text} # Or split into lines: cleaned_text.split('\n')

            # --- Save processed data (structured or fallback) as compressed JSON ---
            try:
                json_string = json.dumps(output_data_to_save, indent=2) # Pretty print JSON
            except TypeError as te:
                 log_statement(loglevel="error", logstatement=f"Failed to serialize processed data to JSON for {input_path.name}: {te}", main_logger=main_logger_name)
                 return None

            # Use the string compression helper
            if not hasattr(self, '_compress_string_to_file'):
                 log_statement(loglevel="error", logstatement="_compress_string_to_file helper method not found.", main_logger=main_logger_name)
                 raise NotImplementedError("_compress_string_to_file helper method not found.")

            # Ensure output path has the correct extension (.json.zst)
            output_path = output_path.with_suffix('.json.zst')
            save_success = self._compress_string_to_file(json_string, output_path)

            if not save_success:
                 return None # Error logged within helper

            # --- Calculate hash & Return metadata (as before) ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                 log_statement(loglevel='error', logstatement=f"Failed to generate hash for output file: {output_path}", main_logger=str(__name__))
                 return None

            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed in _process_textual_data for {input_path}: {e}", main_logger=str(__name__), exc_info=True)
            # Ensure output_path variable is defined before attempting unlink
            if 'output_path' in locals() and output_path.exists():
                try: output_path.unlink()
                except OSError: pass
            return None

    def _process_numerical(self, reader: FileReader, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Processes numerical files (e.g., from ExcelReader). Reads data, applies scaling,
        saves as compressed Parquet, and returns metadata.

        Args:
            reader (FileReader): An instance of a FileReader subclass (e.g., ExcelReader).
            input_path (Path): Absolute path to the input file.
            output_path (Path): Absolute path for the compressed Parquet output file (.parquet.zst).

        Returns:
            Optional[Dict]: Dictionary with 'output_path' (relative) and 'output_hash' on success, else None.
        """
        log_statement(loglevel='debug', logstatement=f"Starting _process_numerical for: {input_path.name}", main_logger=str(__name__))
        try:
            # Use the reader's read method - assumes it returns a DataFrame
            dataframe = reader.read() # reader initialized with input_path

            if dataframe is None or dataframe.empty:
                log_statement(loglevel='warning', logstatement=f"Reader returned empty or None DataFrame for {input_path.name}. Skipping.", main_logger=str(__name__))
                return None

            # --- Numerical Processing Logic ---
            # Convert to numeric types, handle errors
            numeric_df = dataframe.apply(pd.to_numeric, errors='coerce')
            # Drop columns that are entirely NaN after conversion attempt
            numeric_df = numeric_df.dropna(axis=1, how='all')

            if numeric_df.empty:
                log_statement(loglevel='warning', logstatement=f"No numeric data found after conversion in {input_path.name}. Skipping.", main_logger=str(__name__))
                return None

            # Scaling (using the scaler initialized in DataProcessor __init__)
            processed_df = numeric_df
            if hasattr(self, 'scaler') and self.scaler:
                log_statement(loglevel='debug', logstatement=f"Applying scaler to data from {input_path.name}", main_logger=str(__name__))
                try:
                    # Scaler expects numpy/cupy array. Handle DataFrame conversion.
                    # Use .values for numpy array. For cuDF, .to_cupy()
                    is_cudf = 'cudf' in sys.modules and isinstance(numeric_df, sys.modules['cudf'].DataFrame)
                    if is_cudf:
                        data_array = numeric_df.to_cupy()
                        scaled_array = self.scaler.fit_transform(data_array)
                        # Reconstruct DataFrame (assuming scaler preserves columns/index)
                        processed_df = sys.modules['cudf'].DataFrame(scaled_array, index=numeric_df.index, columns=numeric_df.columns)
                    else: # pandas
                        data_array = numeric_df.values
                        # Sklearn scaler needs 2D. Ensure input is suitable.
                        # The SklearnScalerWrapper should handle DataFrame input/output
                        processed_df = self.scaler.fit_transform(numeric_df) # Assuming wrapper handles DataFrame

                except Exception as scale_err:
                    log_statement(loglevel='error', logstatement=f"Scaling failed for {input_path.name}: {scale_err}. Proceeding with unscaled data.", main_logger=str(__name__), exc_info=True)
                    processed_df = numeric_df # Use original numeric data if scaling fails
            else:
                log_statement(loglevel='debug', logstatement=f"No scaler available or configured. Skipping scaling for {input_path.name}", main_logger=str(__name__))


            # --- Save processed DataFrame to compressed Parquet ---
            # Use a helper function or implement logic here
            if not hasattr(self, 'save_dataframe_to_parquet_zst'):
                log_statement(loglevel='error', logstatement="save_dataframe_to_parquet_zst helper method not found.", main_logger=str(__name__))
                raise NotImplementedError("save_dataframe_to_parquet_zst helper method not found.")

            save_success = self.save_dataframe_to_parquet_zst(processed_df, output_path)

            if not save_success:
                # Error logged within helper
                return None # Indicate failure

            # --- Calculate hash of the output file ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"Failed to generate hash for output file: {output_path}", main_logger=str(__name__))
                return None # Indicate failure

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed in _process_numerical for {input_path}: {e}", main_logger=str(__name__), exc_info=True)
            if output_path.exists():
                try: output_path.unlink()
                except OSError: pass
            return None

    def _process_pdf(self, reader: PDFReader, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Processes PDF files. Reads text using PDFReader, cleans it, saves the
        compressed text, and returns metadata.

        Args:
            reader (PDFReader): An instance of PDFReader.
            input_path (Path): Absolute path to the input PDF file.
            output_path (Path): Absolute path for the compressed text output file (.txt.zst).

        Returns:
            Optional[Dict]: Dictionary with 'output_path' (relative) and 'output_hash' on success, else None.
        """
        log_statement(loglevel='debug', logstatement=f"Starting _process_pdf for: {input_path.name}", main_logger=str(__name__))
        try:
            # PDFReader.read() returns a DataFrame with a 'text' column
            dataframe = reader.read()

            if dataframe is None or dataframe.empty or 'text' not in dataframe.columns or dataframe['text'].iloc[0] is None:
                log_statement(loglevel='warning', logstatement=f"PDFReader returned no text content for {input_path.name}. Skipping.", main_logger=str(__name__))
                return None

            # --- Text Processing Logic ---
            raw_text = dataframe['text'].iloc[0]
            # Apply cleaning similar to _process_text
            processed_text = raw_text.lower()
            if hasattr(self, 'cleaning_regex'):
                processed_text = self.cleaning_regex.sub('', processed_text) # Apply regex cleaning
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()

            # NLTK (optional)
            if NLTK_AVAILABLE and lemmatizer:
                def lemmatize_and_filter(text):
                    words = text.split()
                    lemmatized = [lemmatizer.lemmatize(w) for w in words if w]
                    filtered = [w for w in lemmatized if w not in stop_words]
                    return " ".join(filtered)
                processed_text = lemmatize_and_filter(processed_text)

            if not processed_text:
                log_statement(loglevel='warning', logstatement=f"No text content resulted after processing PDF {input_path.name}. Skipping save.", main_logger=str(__name__))
                return None

            # --- Save processed text to output_path (compressed) ---
            if not hasattr(self, 'compress_string_to_file'):
                log_statement(loglevel='error', logstatement="compress_string_to_file helper method not found.", main_logger=str(__name__))
                raise NotImplementedError("compress_string_to_file helper method not found.")

            save_success = self.compress_string_to_file(processed_text, output_path)

            if not save_success:
                return None # Error logged within helper

            # --- Calculate hash of the output file ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"Failed to generate hash for output file: {output_path}", main_logger=str(__name__))
                return None

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except ImportError:
            log_statement(loglevel='error', logstatement=f"PDF processing failed for {input_path}: Missing required library (e.g., pdfminer.six).", main_logger=str(__name__), exc_info=True)
            return None # Indicate failure due to missing dependency
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed in _process_pdf for {input_path}: {e}", main_logger=str(__name__), exc_info=True)
            if output_path.exists():
                try: output_path.unlink()
                except OSError: pass
            return None

    # Assuming a DocxReader exists that returns a DataFrame with a 'text' column.
    def _process_docx(self, reader: FileReader, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Processes DOCX files. Reads text using a hypothetical DocxReader, cleans it,
        saves the compressed text, and returns metadata.

        Args:
            reader (FileReader): An instance of a hypothetical DocxReader.
            input_path (Path): Absolute path to the input DOCX file.
            output_path (Path): Absolute path for the compressed text output file (.txt.zst).

        Returns:
            Optional[Dict]: Dictionary with 'output_path' (relative) and 'output_hash' on success, else None.
        """
        log_statement(loglevel='debug', logstatement=f"Starting _process_docx for: {input_path.name}", main_logger=str(__name__))
        try:
            # Assume reader.read() returns a DataFrame {'text': [content]}
            dataframe = reader.read()

            if dataframe is None or dataframe.empty or 'text' not in dataframe.columns or dataframe['text'].iloc[0] is None:
                log_statement(loglevel='warning', logstatement=f"DocxReader returned no text content for {input_path.name}. Skipping.", main_logger=str(__name__))
                return None

            # --- Text Processing Logic (similar to PDF) ---
            raw_text = dataframe['text'].iloc[0]
            processed_text = raw_text.lower()
            if hasattr(self, 'cleaning_regex'):
                processed_text = self.cleaning_regex.sub('', processed_text)
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            if NLTK_AVAILABLE and lemmatizer:
                def lemmatize_and_filter(text):
                    words = text.split()
                    lemmatized = [lemmatizer.lemmatize(w) for w in words if w]
                    filtered = [w for w in lemmatized if w not in stop_words]
                    return " ".join(filtered)
                processed_text = lemmatize_and_filter(processed_text)

            if not processed_text:
                log_statement(loglevel='warning', logstatement=f"No text content resulted after processing DOCX {input_path.name}. Skipping save.", main_logger=str(__name__))
                return None

            # --- Save processed text to output_path (compressed) ---
            if not hasattr(self, 'compress_string_to_file'):
                log_statement(loglevel='error', logstatement="compress_string_to_file helper method not found.", main_logger=str(__name__))
                raise NotImplementedError("compress_string_to_file helper method not found.")

            save_success = self.compress_string_to_file(processed_text, output_path)
            if not save_success: return None

            # --- Calculate hash ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"Failed to generate hash for output file: {output_path}", main_logger=str(__name__))
                return None

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except ImportError:
            log_statement(loglevel='error', logstatement=f"DOCX processing failed for {input_path}: Missing required library (e.g., python-docx).", main_logger=str(__name__), exc_info=True)
            return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed in _process_docx for {input_path}: {e}", main_logger=str(__name__), exc_info=True)
            if output_path.exists():
                try: output_path.unlink()
                except OSError: pass
            return None

    # This function essentially wraps _process_numerical for Excel files.
    def _process_excel(self, reader: ExcelReader, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Processes Excel files by treating them as numerical data sources. Passes
        handling to _process_numerical.

        Args:
            reader (ExcelReader): An instance of ExcelReader.
            input_path (Path): Absolute path to the input Excel file.
            output_path (Path): Absolute path for the compressed Parquet output file (.parquet.zst).

        Returns:
            Optional[Dict]: Dictionary with 'output_path' (relative) and 'output_hash' on success, else None.
        """
        log_statement(loglevel='debug', logstatement=f"Delegating Excel processing for {input_path.name} to _process_numerical.", main_logger=str(__name__))
        # Call the numerical processing function, passing the Excel reader and paths
        return self._process_numerical(reader, input_path, output_path)

    # Placeholder for Audio processing - Requires an AudioReader and audio processing logic
    def _process_audio(self, reader: FileReader, input_path: Path, output_path: Path) -> Optional[Dict]:
        """
        Placeholder for processing Audio files. Reads using an AudioReader, performs
        transcription/feature extraction, saves output (e.g., compressed WAV or features),
        and returns metadata.

        Args:
            reader (FileReader): An instance of a hypothetical AudioReader.
            input_path (Path): Absolute path to the input audio file.
            output_path (Path): Absolute path for the processed output file (e.g., .wav.zst).

        Returns:
            Optional[Dict]: Dictionary with 'output_path' (relative) and 'output_hash' on success, else None.
        """
        log_statement(loglevel='warning', logstatement=f"Audio processing (_process_audio) not implemented yet for: {input_path.name}", main_logger=str(__name__))
        # Example Steps (pseudo-code):
        # try:
        #     # 1. Read audio data using reader
        #     # audio_data, sample_rate = reader.read() # Reader needs to return data and rate
        #
        #     # 2. Perform processing (e.g., transcription, feature extraction)
        #     # processed_data = transcribe(audio_data, sample_rate) # Or extract_features(...)
        #     # Or maybe just standardize format/rate and save
        #
        #     # 3. Save processed data (e.g., save text, features, or standardized audio)
        #     # if isinstance(processed_data, str): # If transcription
        #     #     save_success = self.compress_string_to_file(processed_data, output_path.with_suffix('.txt.zst'))
        #     # elif isinstance(processed_data, np.ndarray): # If features
        #     #     # Save numpy array compressed (need helper)
        #     #     save_success = self._save_array_compressed(processed_data, output_path.with_suffix('.npy.zst'))
        #     # elif audio_data: # Save standardized audio
        #     #     # Need function to save audio (e.g., scipy.io.wavfile.write to buffer then compress)
        #     #     save_success = self._save_wav_compressed(audio_data, sample_rate, output_path) # Needs implementation
        #     # else:
        #     #     log_statement(...) return None
        #
        #     # if not save_success: return None
        #
        #     # 4. Calculate hash
        #     # output_hash = generate_data_hash(output_path)
        #     # if output_hash is None: return None
        #
        #     # 5. Return metadata
        #     # relative_output_path = output_path.relative_to(self.output_dir)
        #     # return {
        #     #     "processed_path": str(relative_output_path),
        #     #     COL_DATA_HASH: output_hash
        #     # }
        #
        # except ImportError:
        #      log_statement(..., f"Audio processing failed for {input_path}: Missing required library (e.g., librosa, soundfile).", ...)
        #      return None
        # except Exception as e:
        #     log_statement(..., f"Failed in _process_audio for {input_path}: {e}", ..., exc_info=True)
        #     if output_path.exists():
        #         try: output_path.unlink()
        #         except OSError: pass
        #     return None

        # Return None because it's not implemented
        
    def _process_odt(self, filepath: Path):
        """Processes ODT files using odfpy."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing ODT file: {filepath.name}"), main_logger=str(__name__))
        try:
            import odfpy
            from odf.opendocument import OpenDocumentText
            doc = OpenDocumentText(filepath)
            text = []
            for elem in doc.getElementsByType(odfpy.text.P):
                text.append(elem.firstChild.data)
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"ODT processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ODT processing failed: {e}")
            return None
        
    def _process_doc(self, filepath: Path):
        """Processes DOC files using python-docx."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing DOC file: {filepath.name}"), main_logger=str(__name__))
        try:
            import docx
            doc = docx.Document(filepath)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"DOC processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"DOC processing failed: {e}")
            return None
        
    def _process_html_xml(self, filepath: Path):
        """Processes HTML/XML files using BeautifulSoup."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing HTML/XML file: {filepath.name}"), main_logger=str(__name__))
        try:
            import bs4
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            soup = bs4.BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"HTML/XML processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"HTML/XML processing failed: {e}")
            return None
        
    def _process_rtf(self, filepath: Path):
        """Processes RTF files using striprtf."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing RTF file: {filepath.name}"), main_logger=str(__name__))
        try:
            import striprtf
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return striprtf.strip(content)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"RTF processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"RTF processing failed: {e}")
            return None
        
    def _process_epub(self, filepath: Path):
        """Processes EPUB files using EbookLib."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing EPUB file: {filepath.name}"), main_logger=str(__name__))
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(filepath)
            text = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                text.append(item.get_body_content_str())
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"EPUB processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"EPUB processing failed: {e}")
            return None
        
    def _process_zip(self, filepath: Path):
        """Processes ZIP files using zipfile."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Processing ZIP file: {filepath.name}"), main_logger=str(__name__))
        try:
            import zipfile
            with zipfile.ZipFile(filepath, 'r') as z:
                text = []
                for file in z.namelist():
                    with z.open(file) as f:
                        text.append(f.read().decode('utf-8'))
                return "\n".join(text)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"ZIP processing failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ZIP processing failed: {e}")
            return None

    def _read_content(self, filepath: Path):
        """Reads the content of a file, handling different encodings."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Reading content from {filepath.name}"), main_logger=str(__name__))
        try:
            if filepath.suffix.lower() == '.zst':
                dctx = zstd.ZstdDecompressor()
                with open(filepath, 'rb', encoding='utf-8') as ifh:
                    buffer = io.BytesIO()
                    # Read the decompressed stream in chunks
                    with dctx.stream_reader(ifh) as reader:
                        chunk_size = 1024 * 1024 # Read in 1MB chunks
                        while True:
                            chunk = reader.read(chunk_size)
                            if not chunk:
                                break
                            buffer.write(chunk)
                    decompressed_bytes = buffer.getvalue()
                    buffer.close() # Release memory held by the buffer
                return decompressed_bytes.decode('utf-8', errors='replace')
            else:
                return filepath.read_text(encoding='utf-8', errors='replace')
        except UnicodeDecodeError:
            # Try with ISO-8859-1 (Latin-1) encoding
            try:
                with open(filepath, 'r', encoding='ISO-8859-1') as f:
                    return f.read()
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to read file {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
                self.repo.update_entry(filepath, status='error', error_message=f"File read failed: {e}")
                return None
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to read content from {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            raise # Re-raise to be caught by _process_file
        except FileNotFoundError:
            log_statement(loglevel=str("error"), logstatement=str(f"File not found during read: {filepath}"), main_logger=str(__name__), exc_info=True)
            return None
        except IsADirectoryError:
            log_statement(loglevel=str("error"), logstatement=str(f"Expected file but found directory: {filepath}"), main_logger=str(__name__), exc_info=True)
            return None
        except zstd.ZstdError as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Zstandard decompression failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            return None
        except UnicodeDecodeError as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Unicode decoding failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            return None
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error reading {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            return None

    def _ensure_repo_exists(self, filepath: Path, header: List[str]):
        """Creates the repository file with a header if it doesn't exist."""
        if not filepath.exists():
            log_statement(loglevel=str("info"), logstatement=str(f"Repository file not found at '{filepath}'. Initializing..."), main_logger=str(__name__))
            try:
                # Write header to a new compressed file
                def header_gen():
                    yield ','.join(header) # CSV header line

                stream_compress_lines(str(filepath), header_gen())
                log_statement(loglevel=str("info"), logstatement=str(f"Initialized repository file: {filepath}"), main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel=str("critical"), logstatement=str(f"Failed to initialize repository file '{filepath}': {e}"), main_logger=str(__name__), exc_info=True)
                raise

    def _get_next_designation(self) -> int:
        """Finds the next available designation number based on the existing repo."""
        max_designation = 0
        if not self.repo_filepath.exists() or os.path.getsize(self.repo_filepath) == 0:
             return 1 # Start from 1 if repo is empty or doesn't exist

        try:
            # Read only the first column (Designation) efficiently
            for i, row in enumerate(self.read_repo_stream()):
                 # Skip header if reading raw lines, handled by csv.DictReader if used
                if i == 0 and row.get(COL_DESIGNATION) == COL_DESIGNATION: # Check header
                    continue
                try:
                    designation = int(row.get(COL_DESIGNATION, 0))
                    if designation > max_designation:
                        max_designation = designation
                except (ValueError, TypeError):
                     log_statement(loglevel=str("warning"), logstatement=str(f"Skipping row {i+1}: Invalid designation number in row: {row}"), main_logger=str(__name__))
                     continue # Skip rows with invalid numbers
            return max_designation + 1
        except FileNotFoundError:
            log_statement(loglevel=str("warning"), logstatement=str(f"Repository file '{self.repo_filepath}' not found while getting next designation. Starting from 1."), main_logger=str(__name__))
            return 1
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading repository to find next designation: {e}"), main_logger=str(__name__), exc_info=True)
            # Fallback or re-raise depending on desired robustness
            log_statement(loglevel=str("warning"), logstatement=str("Defaulting next designation to 1 due to error."), main_logger=str(__name__))
            return 1

    def _load_existing_hashes(self) -> Dict[str, int]:
        """Loads existing data hashes and their designations for copy checking."""
        hashes = {}
        if not self.repo_filepath.exists() or os.path.getsize(self.repo_filepath) == 0:
            return hashes
        try:
            for row in self.read_repo_stream():
                data_hash = row.get(COL_DATA_HASH)
                designation = row.get(COL_DESIGNATION)
                if data_hash and designation:
                    try:
                        # Store the first designation seen for a given hash
                        if data_hash not in hashes:
                             hashes[data_hash] = int(designation)
                    except ValueError:
                        log_statement(loglevel=str("warning"), logstatement=str(f"Invalid designation '{designation}' found for hash '{data_hash}'"), main_logger=str(__name__))
            return hashes
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error loading existing hashes from repository: {e}"), main_logger=str(__name__), exc_info=True)
            return {} # Return empty on error

    def read_repo_stream(self, filepath: Optional[Path] = None) -> Generator[Dict[str, str], None, None]:
        """
        Reads the repository CSV file line by line using streaming decompression.
        Yields each row as a dictionary. Handles potential errors during reading.
        """
        target_filepath = filepath or self.repo_filepath
        if not target_filepath.exists():
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to read non-existent repository: {target_filepath}"), main_logger=str(__name__))
            return # Yield nothing

        header = []
        try:
            line_generator = stream_decompress_lines(str(target_filepath))
            # Read header first
            try:
                header_line = next(line_generator)
                header = [h.strip() for h in header_line.split(',')]
            except StopIteration:
                 log_statement(loglevel=str("warning"), logstatement=str(f"Repository file is empty: {target_filepath}"), main_logger=str(__name__))
                 return # Empty file

            # Use csv.DictReader on the remaining lines
            # We need to simulate a file-like object for DictReader
            reader = csv.DictReader(line_generator, fieldnames=header, restval=None) # Use header as fieldnames
                                                                                       # restval handles rows with too few fields
            for row in reader:
                 if len(row) != len(header):
                     log_statement(loglevel=str("warning"), logstatement=str(f"Malformed row in {target_filepath} (expected {len(header)} fields, got {len(row)}): {row}"), main_logger=str(__name__))
                     # Optionally yield a partial dict or skip
                     # yield row # Yields what was parsed
                     continue # Skip malformed row
                 yield row

        except FileNotFoundError:
            log_statement(loglevel=str("error"), logstatement=str(f"Repository file not found during streaming read: {target_filepath}"), main_logger=str(__name__))
            # Or re-raise depending on desired behavior
        except zstd.ZstdError as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Zstd decompression error reading {target_filepath}: {e}"), main_logger=str(__name__), exc_info=True)
        except csv.Error as e:
            log_statement(loglevel=str("error"), logstatement=str(f"CSV parsing error reading {target_filepath}: {e}"), main_logger=str(__name__), exc_info=True)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error reading repository stream {target_filepath}: {e}"), main_logger=str(__name__), exc_info=True)

    def _append_repo_stream(self, filepath: Path, rows_generator: Generator[Dict[str, str], None, None], header: List[str]):
        """
        Appends rows from a generator to a zstandard compressed CSV.
        Reads existing content, appends new rows, and writes back compressed.
        NOTE: This is NOT memory-efficient for very large files as it reads all
              existing data. True streaming append to zstd requires more complex handling
              or potentially keeping files uncompressed during active addition.
              For now, we implement the read-all, append, write-all method.
        """
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        existing_data = list(self.read_repo_stream(filepath)) # Read all existing valid rows

        def combined_generator():
            # Yield header if the file was empty or didn't exist before
            if not existing_data:
                 yield ','.join(header)
            else:
                # Yield existing rows as comma-separated strings
                writer = io.StringIO()
                dict_writer = csv.DictWriter(writer, fieldnames=header, lineterminator='\n')
                if not any(row.get(header[0]) == header[0] for row in existing_data): # Check if header exists
                     dict_writer.writeheader() # Write header if missing
                dict_writer.writerows(existing_data)
                writer.seek(0)
                for line in writer:
                    yield line.strip() # Yield existing content

            # Yield new rows from the input generator
            writer = io.StringIO()
            dict_writer = csv.DictWriter(writer, fieldnames=header, lineterminator='\n')
            dict_writer.writerows(rows_generator) # Write new rows
            writer.seek(0)
            for line in writer:
                 yield line.strip() # Yield new content


        try:
            # Write combined data to temp file compressed
            stream_compress_lines(str(temp_filepath), (line for line in combined_generator() if line)) # Skip empty lines

            # Replace original file with temp file
            shutil.move(str(temp_filepath), str(filepath))
            log_statement(loglevel=str("info"), logstatement=str(f"Appended rows to repository: {filepath}"), main_logger=str(__name__))

        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to append rows to repository '{filepath}': {e}"), main_logger=str(__name__), exc_info=True)
            # Clean up temp file if it exists
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except OSError:
                    pass
            raise # Re-raise the exception

    # --- Methods for Data Manipulation Submenu ---
    def add_folder(self, folder_path: str):
        """
        Scans a folder recursively, identifies supported files, calculates metadata,
        checks for duplicates based on path and content hash, and adds new file paths
        to the main repository using streaming append. Marks content duplicates.
        Uses custom log_statement for logging.

        Args:
            folder_path (str): The path to the folder to scan.
        """
        try:
            abs_folder_path = Path(folder_path).resolve()
            if not abs_folder_path.is_dir():
                # Use custom log statement
                log_statement(loglevel='error', logstatement=str(f"Folder not found or is not a directory: {abs_folder_path}"), main_logger=str(__name__))
                return
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=str(f"Error resolving folder path '{folder_path}': {e}"), main_logger=str(__name__), exc_info=True)
            return

        # Use custom log statement
        log_statement(loglevel='info', logstatement=str(f"Scanning folder: {abs_folder_path}"), main_logger=str(__name__))
        new_files_to_add = []
        processed_count = 0
        added_count = 0
        skipped_count = 0
        error_count = 0

        # Track file paths already in the repo (read via stream) to avoid adding duplicate paths.
        # Content duplicates are handled separately via data hash.
        # This assumes read_repo_stream works correctly as defined previously.
        try:
            existing_paths = {row.get(COL_FILEPATH) for row in self.read_repo_stream() if row.get(COL_FILEPATH)}
            # Use custom log statement
            log_statement(loglevel='debug', logstatement=str(f"Loaded {len(existing_paths)} existing file paths from repository."), main_logger=str(__name__))
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=str(f"Failed to load existing paths from repository: {e}"), main_logger=str(__name__), exc_info=True)
            log_statement(loglevel='warning', logstatement=str("Proceeding without check for existing paths. Duplicates might be added."), main_logger=str(__name__))
            existing_paths = set() # Continue with an empty set

        # Use rglob for recursive iteration
        try:
            for item in abs_folder_path.rglob('*'):
                if item.is_file():
                    processed_count += 1
                    # Use resolved absolute path string for consistency
                    file_path_str = str(item.resolve())
                    file_ext = item.suffix.lower()

                    if file_path_str in existing_paths:
                        # Use custom log statement
                        log_statement(loglevel='debug', logstatement=str(f"Skipping already tracked file path: {file_path_str}"), main_logger=str(__name__))
                        skipped_count += 1
                        continue

                    # Check for supported file types
                    if file_ext not in ACCEPTED_FILE_TYPES:
                        # Use custom log statement
                        log_statement(loglevel='debug', logstatement=str(f"Skipping unsupported file type '{file_ext}': {file_path_str}"), main_logger=str(__name__))
                        skipped_count += 1
                        continue

                    # --- Handle Original Compression State & Archives ---
                    # TODO: Implement robust archive handling (extraction) if needed.
                    if file_ext in ['.zip', '.zst', '.zstd']:
                        # Use custom log statement
                        log_statement(loglevel='warning', logstatement=str(f"Archive file found: {file_path_str}. Treating as single compressed file. Extraction logic not implemented."), main_logger=str(__name__))
                        is_compressed_original = 'Y' # The archive file itself
                    else:
                        # Simple assumption: non-archive files are not compressed unless logic added here
                        # (e.g., using magic bytes) to detect compression.
                        is_compressed_original = 'N'

                    # --- Metadata and Hash Generation ---
                    try:
                        # Get file metadata
                        stat = item.stat()
                        mod_time_ts = stat.st_mtime
                        acc_time_ts = stat.st_atime
                        # Format timestamps as ISO UTC strings
                        mod_time_iso = datetime.datetime.fromtimestamp(mod_time_ts, tz=timezone.utc).isoformat()
                        acc_time_iso = datetime.datetime.fromtimestamp(acc_time_ts, tz=timezone.utc).isoformat()

                        # Generate hashes
                        # Ensure hash functions handle errors and return empty string or None on failure
                        hashed_path = hash_filepath(file_path_str)
                        data_hash = generate_data_hash(file_path_str) # Content hash

                        current_status = STATUS_LOADED # Default status
                        is_copy = 'N' # Default copy status

                        # Check hash generation results
                        if not hashed_path:
                            # Use custom log statement
                            log_statement(loglevel='error', logstatement=str(f"Failed to generate required filepath hash for: {file_path_str}. Marking as Error."), main_logger=str(__name__))
                            current_status = STATUS_ERROR
                            error_count += 1
                            # Skip adding? Or add with Error status? Let's add with Error status.
                            # data_hash = "" # Ensure data_hash is empty if path hash failed? Or proceed?

                        if not data_hash and current_status != STATUS_ERROR:
                            # Use custom log statement
                            log_statement(loglevel='error', logstatement=str(f"Failed to generate required data hash for: {file_path_str}. Marking as Error."), main_logger=str(__name__))
                            current_status = STATUS_ERROR
                            error_count += 1
                            # No valid data hash means we cannot check for copies effectively

                        # Check for content copy using data hash (only if hash was successful)
                        if data_hash and data_hash in self._data_hashes:
                            is_copy = 'Y'
                            original_designation = self._data_hashes[data_hash]
                            # Use custom log statement
                            log_statement(loglevel='info', logstatement=str(f"Detected content copy (Data Hash: {data_hash[:8]}...) for: {file_path_str} - Matches Designation: {original_designation}"), main_logger=str(__name__))
                        elif data_hash and current_status != STATUS_ERROR:
                            # If not a copy and hash is valid, add hash to our in-memory dict for this session
                            # Associate it with the *next* designation number this file will get
                            self._data_hashes[data_hash] = self._next_designation + len(new_files_to_add)

                        # Prepare row data dictionary using constants
                        row_data = {
                            COL_DESIGNATION: self._next_designation + len(new_files_to_add),
                            COL_FILETYPE: file_ext,
                            COL_FILEPATH: file_path_str, # Store absolute path
                            COL_HASHED_PATH_ID: hashed_path if hashed_path else "", # Ensure empty string if None/failed
                            COL_COMPRESSED_FLAG: is_compressed_original,
                            COL_MOD_DATE: mod_time_iso,
                            COL_ACC_DATE: acc_time_iso,
                            COL_DATA_HASH: data_hash if data_hash else "", # Ensure empty string if None/failed
                            COL_IS_COPY_FLAG: is_copy,
                            COL_STATUS: current_status
                        }
                        new_files_to_add.append(row_data)
                        # Only increment added_count if status isn't ERROR? Or count all attempts? Count all added rows.
                        added_count += 1
                        existing_paths.add(file_path_str) # Track path as added in this run
                        # Use custom log statement
                        log_statement(loglevel='debug', logstatement=str(f"Prepared file for repository addition (Status: {current_status}): {file_path_str}"), main_logger=str(__name__))

                    except OSError as e:
                        # Use custom log statement
                        log_statement(loglevel='error', logstatement=str(f"OS Error accessing metadata/hashes for {file_path_str}: {e}"), main_logger=str(__name__))
                        error_count += 1
                        # Optionally add an entry with STATUS_ERROR here? For now, we skip.
                    except Exception as e:
                        # Use custom log statement (including exc_info for unexpected errors)
                        log_statement(loglevel='error', logstatement=str(f"Unexpected error processing file {file_path_str}: {e}"), main_logger=str(__name__), exc_info=True)
                        error_count += 1
                        # Optionally add an entry with STATUS_ERROR here? For now, we skip.

                    # Log progress periodically
                    if processed_count % 500 == 0:
                        # Use custom log statement
                        log_statement(loglevel='info', logstatement=str(f"Scanned {processed_count} files so far..."), main_logger=str(__name__))

        except PermissionError as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=str(f"Permission error scanning folder '{abs_folder_path}': {e}. Check read permissions."), main_logger=str(__name__))
            error_count += 1 # Count this as an error for the summary
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=str(f"Unexpected error during folder scan of '{abs_folder_path}': {e}"), main_logger=str(__name__), exc_info=True)
            error_count += 1 # Count this as an error


        # --- Append all new files found in this run ---
        if new_files_to_add:
            # Use custom log statement
            log_statement(loglevel='info', logstatement=str(f"Scan found {len(new_files_to_add)} new file entries to add to the repository."), main_logger=str(__name__))
            try:
                # Define the generator function locally to pass to the append stream method
                def row_generator():
                    for row in new_files_to_add:
                        yield row
                # Use the previously defined stream append method
                self._append_repo_stream(self.repo_filepath, row_generator(), MAIN_REPO_HEADER)
                # Update the next designation number *after* successful append
                self._next_designation += len(new_files_to_add)
                # Use custom log statement
                log_statement(loglevel='info', logstatement=str(f"Successfully appended {len(new_files_to_add)} entries to {self.repo_filepath.name}"), main_logger=str(__name__))
            except Exception as e:
                # Use custom log statement
                log_statement(loglevel='error', logstatement=str(f"Failed to append batch of new files to repository: {e}"), main_logger=str(__name__), exc_info=True)
                # Note: _next_designation and _data_hashes might be inconsistent if append fails here.
                # Consider adding logic to handle this potential inconsistency if critical.
        else:
            # Use custom log statement
            log_statement(loglevel='info', logstatement=str("No new files found in the specified folder to add to the repository."), main_logger=str(__name__))

        # Use custom log statement for summary
        log_statement(loglevel='info', logstatement=str(f"Folder scan complete. Files Processed: {processed_count}, Entries Added: {added_count}, Paths Skipped (already tracked/unsupported): {skipped_count}, Errors during scan/processing: {error_count}"), main_logger=str(__name__))

    def remove_folder(self, folder_path: str):
        """
        Removes all entries from the main repository whose Filepath
        starts with the specified folder path.
        Rule: 7.1.B
        """
        abs_folder_path = str(Path(folder_path).resolve())
        log_statement(loglevel=str("warning"), logstatement=str(f"Removing entries for folder and subfolders: {abs_folder_path}"), main_logger=str(__name__))

        rows_to_keep = []
        removed_count = 0

        try:
            # Read the repo, keeping only rows that DON'T match the path prefix
            for row in self.read_repo_stream():
                 file_path = row.get(COL_FILEPATH)
                 if file_path and file_path.startswith(abs_folder_path):
                      log_statement(loglevel=str("debug"), logstatement=str(f"Marking for removal: {file_path} (Designation: {row.get(COL_DESIGNATION)})"), main_logger=str(__name__))
                      removed_count += 1
                 else:
                     # Keep header row and non-matching rows
                      if row.get(COL_DESIGNATION) == COL_DESIGNATION: # Ensure header is kept if present
                          rows_to_keep.append(row)
                      elif file_path: # Keep rows with valid file paths not matching prefix
                           rows_to_keep.append(row)
                      # Decide how to handle rows without file paths if they exist

            if removed_count > 0:
                # Overwrite the repo with only the rows to keep
                temp_filepath = self.repo_filepath.with_suffix(self.repo_filepath.suffix + '.tmp')

                def keep_generator():
                    # Write rows_to_keep back
                    writer = io.StringIO()
                    # Ensure header is present before writing rows
                    if not any(r.get(COL_DESIGNATION) == COL_DESIGNATION for r in rows_to_keep):
                         yield ','.join(MAIN_REPO_HEADER) # Add header if it got removed
                    else:
                        # Write header first if it's in rows_to_keep
                         header_row = next(r for r in rows_to_keep if r.get(COL_DESIGNATION) == COL_DESIGNATION)
                         yield ','.join(header_row.values())


                    dict_writer = csv.DictWriter(writer, fieldnames=MAIN_REPO_HEADER, lineterminator='\n')
                    # Write rows excluding the header row if already yielded
                    dict_writer.writerows([r for r in rows_to_keep if r.get(COL_DESIGNATION) != COL_DESIGNATION])

                    writer.seek(0)
                    for line in writer:
                         yield line.strip()


                stream_compress_lines(str(temp_filepath), (line for line in keep_generator() if line))
                shutil.move(str(temp_filepath), str(self.repo_filepath))

                log_statement(loglevel=str("info"), logstatement=str(f"Removed {removed_count} entries associated with folder: {abs_folder_path}"), main_logger=str(__name__))
                # Reload internal state as designations/hashes might have changed if we renumbered
                self._next_designation = self._get_next_designation()
                self._data_hashes = self._load_existing_hashes()
            else:
                log_statement(loglevel=str("info"), logstatement=str(f"No entries found for folder: {abs_folder_path}"), main_logger=str(__name__))

        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to remove folder entries from repository: {e}"), main_logger=str(__name__), exc_info=True)
            # State might be inconsistent, consider recovery or warning user

    def update_status(self, designation: int, new_status: str, target_repo_path: Optional[Path] = None):
        """
        Updates the status for a specific designation in the specified repository file.
        Uses the inefficient read-modify-write approach for zstd files.
        """
        repo_path = target_repo_path or self.repo_filepath
        if not repo_path.exists():
             log_statement(loglevel=str("error"), logstatement=str(f"Cannot update status. Repository file not found: {repo_path}"), main_logger=str(__name__))
             return False

        log_statement(loglevel=str("debug"), logstatement=str(f"Attempting to update status to '{new_status}' for designation {designation} in {repo_path.name}"), main_logger=str(__name__))
        updated = False
        rows_to_write = []

        try:
            for row in self.read_repo_stream(repo_path):
                current_designation_str = row.get(COL_DESIGNATION)
                try:
                    if current_designation_str and int(current_designation_str) == designation:
                        if row.get(COL_STATUS) != new_status:
                            log_statement(loglevel=str("info"), logstatement=str(f"Updating status for Designation {designation} from '{row.get(COL_STATUS)}' to '{new_status}' in {repo_path.name}"), main_logger=str(__name__))
                            row[COL_STATUS] = new_status
                            updated = True
                        else:
                             log_statement(loglevel=str("debug"), logstatement=str(f"Status for Designation {designation} is already '{new_status}' in {repo_path.name}"), main_logger=str(__name__))
                             # Keep row as is, no change needed
                    # Keep all rows (modified or not) to rewrite the file
                    rows_to_write.append(row)

                except (ValueError, TypeError):
                     log_statement(loglevel=str("warning"), logstatement=str(f"Skipping row due to invalid designation: {row}"), main_logger=str(__name__))
                     rows_to_write.append(row) # Keep malformed row? Or discard?

            if updated:
                # Overwrite the repo with the modified rows
                temp_filepath = repo_path.with_suffix(repo_path.suffix + '.tmp_status')

                def rewrite_generator():
                    # Write rows back
                    writer = io.StringIO()
                    dict_writer = csv.DictWriter(writer, fieldnames=MAIN_REPO_HEADER, lineterminator='\n') # Assuming main header for now
                    dict_writer.writeheader() # Write header
                    dict_writer.writerows([r for r in rows_to_write if r.get(COL_DESIGNATION) != COL_DESIGNATION]) # Write data rows
                    writer.seek(0)
                    for line in writer:
                         yield line.strip()

                stream_compress_lines(str(temp_filepath), (line for line in rewrite_generator() if line))
                shutil.move(str(temp_filepath), str(repo_path))
                log_statement(loglevel=str("debug"), logstatement=str(f"Successfully updated status for designation {designation} in {repo_path.name}"), main_logger=str(__name__))
                return True
            else:
                 log_statement(loglevel=str("debug"), logstatement=str(f"No status update needed or designation {designation} not found in {repo_path.name}"), main_logger=str(__name__))
                 return False # Return False if no update occurred

        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to update status for designation {designation} in {repo_path.name}: {e}"), main_logger=str(__name__), exc_info=True)
            return False

    def process_data_list(self):
        """
        Iterates through files marked as 'Loaded' (L) in the main repository,
        processes their content, saves output to a mirrored structure,
        and updates status to 'Processed' (P).
        Rule: 7.1.C
        """
        log_statement(loglevel=str("info"), logstatement=str("Starting data processing pipeline..."), main_logger=str(__name__))
        processed_count = 0
        error_count = 0
        output_base = Path(OUTPUT_DIR_BASE)
        output_base.mkdir(parents=True, exist_ok=True)

        # Get files to process (Status 'L')
        files_to_process = []
        try:
            for row in self.read_repo_stream():
                 if row.get(COL_STATUS) == STATUS_LOADED:
                     files_to_process.append(row)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to read repository to find files for processing: {e}"), main_logger=str(__name__), exc_info=True)
            return # Cannot proceed

        if not files_to_process:
            log_statement(loglevel=str("info"), logstatement=str("No files found with status 'Loaded' to process."), main_logger=str(__name__))
            return

        log_statement(loglevel=str("info"), logstatement=str(f"Found {len(files_to_process)} files to process."), main_logger=str(__name__))

        for row in files_to_process:
            designation = int(row.get(COL_DESIGNATION, -1))
            filepath_str = row.get(COL_FILEPATH)
            filetype = row.get(COL_FILETYPE)

            if designation == -1 or not filepath_str:
                log_statement(loglevel=str("warning"), logstatement=str(f"Skipping invalid row during processing: {row}"), main_logger=str(__name__))
                continue

            input_path = Path(filepath_str)
            if not input_path.exists():
                 log_statement(loglevel=str("error"), logstatement=str(f"File listed in repository not found: {filepath_str}. Setting status to Error."), main_logger=str(__name__))
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
                 continue

            # --- Create Mirrored Output Path ---
            relative_path = input_path.relative_to(input_path.anchor) # Get path relative to drive root
            output_subpath = output_base / relative_path.parent
            output_subpath.mkdir(parents=True, exist_ok=True)
            output_filename = input_path.stem + PROCESSED_EXT + ".zst" # Add .proc and .zst extension
            output_filepath = output_subpath / output_filename

            log_statement(loglevel=str("info"), logstatement=str(f"Processing Designation {designation}: {filepath_str} -> {output_filepath}"), main_logger=str(__name__))

            try:
                # --- Actual Processing Logic ---
                # This section needs to be customized based on file type and desired processing.
                # It should use streaming/generators where possible.
                # Example: Simple text file line processing (replace with your logic)

                # Placeholder: Get reader based on filetype (requires FileReaderFactory or similar)
                # reader = FileReaderFactory.get_reader(filepath_str, filetype)
                # if not reader:
                #    raise ValueError(f"No reader available for file type {filetype}")

                def processed_lines_generator():
                    # Example: Read lines, maybe clean them, yield processed lines
                    # This should interact with your actual data readers and processing functions
                    line_count = 0
                    # Replace this with actual reading logic, potentially decompressing first if needed
                    if filetype == '.txt': # Example
                         # Assuming text files are not separately compressed in this example
                         with open(filepath_str, 'r', encoding='utf-8', errors='ignore') as infile:
                              for line in infile:
                                   processed_line = line.strip().upper() # Simple example: uppercase
                                   if processed_line: # Skip empty lines after processing
                                       yield processed_line
                                       line_count += 1
                    # Add elif blocks for other filetypes (.csv, .json, .pdf etc.) using appropriate readers
                    # elif filetype == '.pdf':
                    #     import pypdf2 # Example dependency
                    #     reader = pypdf2.PdfReader(filepath_str)
                    #     for page in reader.pages:
                    #         text = page.extract_text()
                    #         if text:
                    #             # Process extracted text per page or line by line
                    #             for line in text.splitlines():
                    #                 processed_line = line.strip() # Example
                    #                 if processed_line:
                    #                     yield processed_line
                    #                     line_count += 1
                    else:
                         log_statement(loglevel=str("warning"), logstatement=str(f"Processing logic for file type {filetype} not implemented yet. Skipping content processing."), main_logger=str(__name__))
                         # yield "" # Yield nothing or handle differently
                         raise NotImplementedError(f"Processing for {filetype} not implemented.")

                    log_statement(loglevel=str("debug"), logstatement=str(f"Processed {line_count} lines/units from {filepath_str}"), main_logger=str(__name__))


                # Stream compressed output
                stream_compress_lines(str(output_filepath), processed_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_PROCESSED):
                    processed_count += 1
                    log_statement(loglevel=str("info"), logstatement=str(f"Successfully processed and updated status for Designation {designation}"), main_logger=str(__name__))
                    # TODO: Add entry to processed_repository.csv.zst (similar _append_repo_stream logic)
                else:
                    log_statement(loglevel=str("error"), logstatement=str(f"Processed Designation {designation} but FAILED to update status in repository."), main_logger=str(__name__))
                    error_count += 1
                    # Consider cleanup of the generated .proc.zst file?

            except NotImplementedError as e:
                 log_statement(loglevel=str("error"), logstatement=str(f"Processing failed for Designation {designation} ({filepath_str}): {e}"), main_logger=str(__name__))
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except FileNotFoundError:
                 log_statement(loglevel=str("error"), logstatement=str(f"Input file disappeared during processing: {filepath_str}. Setting status to Error."), main_logger=str(__name__))
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error processing Designation {designation} ({filepath_str}): {e}"), main_logger=str(__name__), exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if output_filepath.exists():
                    try: output_filepath.unlink()
                    except OSError: pass

        log_statement(loglevel=str("info"), logstatement=str(f"Data processing finished. Processed successfully: {processed_count}, Errors: {error_count}"), main_logger=str(__name__))

    def tokenize_processed_data(self):
        """
        Iterates through files marked as 'Processed' (P) in the main repository,
        finds the corresponding '.proc.zst' file, tokenizes its content,
        saves output to a mirrored structure with '.token.zst',
        and updates status to 'Tokenized' (T).
        Rule: 7.1.D
        """
        log_statement(loglevel=str("info"), logstatement=str("Starting tokenization pipeline..."), main_logger=str(__name__))
        tokenized_count = 0
        error_count = 0
        output_base = Path(OUTPUT_DIR_BASE) # Base for mirrored structure

        # Get files to tokenize (Status 'P')
        files_to_tokenize = []
        try:
            for row in self.read_repo_stream():
                 if row.get(COL_STATUS) == STATUS_PROCESSED:
                     files_to_tokenize.append(row)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to read repository to find files for tokenization: {e}"), main_logger=str(__name__), exc_info=True)
            return # Cannot proceed

        if not files_to_tokenize:
            log_statement(loglevel=str("info"), logstatement=str("No files found with status 'Processed' to tokenize."), main_logger=str(__name__))
            return

        log_statement(loglevel=str("info"), logstatement=str(f"Found {len(files_to_tokenize)} processed files to tokenize."), main_logger=str(__name__))

        # Placeholder: Initialize your tokenizer here
        # tokenizer = YourTokenizerClass() or load_tokenizer_function()
        # Example dummy tokenizer function
        def dummy_tokenize(text_line):
            return " ".join(text_line.lower().split()) # Simple split and join

        for row in files_to_tokenize:
            designation = int(row.get(COL_DESIGNATION, -1))
            original_filepath_str = row.get(COL_FILEPATH)

            if designation == -1 or not original_filepath_str:
                log_statement(loglevel=str("warning"), logstatement=str(f"Skipping invalid row during tokenization: {row}"), main_logger=str(__name__))
                continue

            original_path = Path(original_filepath_str)

            # --- Construct Processed File Path ---
            relative_path = original_path.relative_to(original_path.anchor)
            processed_subpath = output_base / relative_path.parent
            processed_filename = original_path.stem + PROCESSED_EXT + ".zst"
            processed_filepath = processed_subpath / processed_filename

            # --- Construct Tokenized Output Path ---
            tokenized_filename = original_path.stem + TOKENIZED_EXT + ".zst"
            tokenized_filepath = processed_subpath / tokenized_filename # Save in same mirrored dir

            if not processed_filepath.exists():
                log_statement(loglevel=str("error"), logstatement=str(f"Processed file not found for Designation {designation}: {processed_filepath}. Setting status to Error."), main_logger=str(__name__))
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                continue

            log_statement(loglevel=str("info"), logstatement=str(f"Tokenizing Designation {designation}: {processed_filepath} -> {tokenized_filepath}"), main_logger=str(__name__))

            try:
                # --- Tokenization Logic ---
                def tokenized_lines_generator():
                    line_count = 0
                    # Stream lines from the compressed processed file
                    for processed_line in stream_decompress_lines(str(processed_filepath)):
                         # Apply actual tokenization logic here
                         tokenized_output = dummy_tokenize(processed_line) # Replace with your tokenizer call
                         # Decide output format: space-separated tokens, one token per line, JSON, etc.
                         if tokenized_output: # Only yield if tokenization produced output
                             yield tokenized_output
                             line_count +=1
                    log_statement(loglevel=str("debug"), logstatement=str(f"Tokenized {line_count} lines/units from {processed_filepath}"), main_logger=str(__name__))

                # Stream compressed output for tokenized data
                stream_compress_lines(str(tokenized_filepath), tokenized_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_TOKENIZED):
                    tokenized_count += 1
                    log_statement(loglevel=str("info"), logstatement=str(f"Successfully tokenized and updated status for Designation {designation}"), main_logger=str(__name__))
                    # TODO: Add entry to tokenized_repository.csv.zst
                else:
                    log_statement(loglevel=str("error"), logstatement=str(f"Tokenized Designation {designation} but FAILED to update status in repository."), main_logger=str(__name__))
                    error_count += 1
                    # Consider cleanup?

            except FileNotFoundError: # Should be caught earlier, but just in case
                 log_statement(loglevel=str("error"), logstatement=str(f"Processed file disappeared during tokenization: {processed_filepath}. Setting status to Error."), main_logger=str(__name__))
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error tokenizing Designation {designation} ({processed_filepath}): {e}"), main_logger=str(__name__), exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if tokenized_filepath.exists():
                    try: tokenized_filepath.unlink()
                    except OSError: pass

        log_statement(loglevel=str("info"), logstatement=str(f"Tokenization finished. Tokenized successfully: {tokenized_count}, Errors: {error_count}"), main_logger=str(__name__))

    # Placeholder for other methods like creating the DataLoader file (Rule 7.1.E)
    def create_dataloader_file(self, output_filename: str = "dataloader_package.zst"):
        """
        Gathers information from the tokenized repository and potentially packages
        tokenized files into a single compressed file for DataLoader usage.
        Rule: 7.1.E
        """
        log_statement(loglevel=str("info"), logstatement=str("Starting DataLoader file creation..."), main_logger=str(__name__))
        tokenized_files_info = []
        try:
            # Read the main repo to find tokenized files and their original paths
            for row in self.read_repo_stream():
                if row.get(COL_STATUS) == STATUS_TOKENIZED:
                    tokenized_files_info.append(row)
            # Or, preferably, read from a dedicated tokenized_repository.csv.zst if created
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to read repository to find tokenized files: {e}"), main_logger=str(__name__), exc_info=True)
            return

        if not tokenized_files_info:
             log_statement(loglevel=str("warning"), logstatement=str("No tokenized files found to create DataLoader package."), main_logger=str(__name__))
             return

        output_filepath = self.repo_dir / output_filename
        output_base = Path(OUTPUT_DIR_BASE)

        # --- Strategy Choices ---
        # 1. Metadata only: Create a compressed JSON/CSV containing metadata (paths, designation, etc.)
        #    The DataLoader would then read individual .token.zst files based on this metadata.
        # 2. Package: Create a compressed archive (e.g., tar.zst) containing the metadata file
        #    AND all the corresponding .token.zst files. DataLoader unpacks/streams from this.

        # --- Example: Strategy 1 (Metadata file) ---
        metadata = []
        for row in tokenized_files_info:
             original_path_str = row.get(COL_FILEPATH)
             designation = row.get(COL_DESIGNATION)
             if not original_path_str or not designation: continue

             original_path = Path(original_path_str)
             relative_path = original_path.relative_to(original_path.anchor)
             tokenized_filename = original_path.stem + TOKENIZED_EXT + ".zst"
             # Assume tokenized file is in the mirrored output directory
             tokenized_file_relative_path = relative_path.parent / tokenized_filename
             # Store relative path within the OUTPUT_DIR_BASE
             tokenized_full_path = output_base / tokenized_file_relative_path

             if tokenized_full_path.exists():
                 metadata.append({
                     "designation": int(designation),
                     "original_filepath": original_path_str,
                     "tokenized_filepath": str(tokenized_full_path), # Store path relative to project root or abs?
                     "data_hash": row.get(COL_DATA_HASH),
                     # Add any other relevant metadata from the row
                 })
             else:
                 log_statement(loglevel=str("warning"), logstatement=str(f"Tokenized file not found for Designation {designation}: {tokenized_full_path}"), main_logger=str(__name__))

        if not metadata:
             log_statement(loglevel=str("warning"), logstatement=str("No valid tokenized file paths found for metadata."), main_logger=str(__name__))
             return

        try:
             import json
             # Compress JSON metadata using zstandard stream
             with open(output_filepath, 'wb') as fh:
                  cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
                  with cctx.stream_writer(fh) as compressor:
                       # Write JSON incrementally or all at once depending on size
                       json_str = json.dumps(metadata, indent=2)
                       compressor.write(json_str.encode('utf-8'))
             log_statement(loglevel=str("info"), logstatement=str(f"Created DataLoader metadata file: {output_filepath}"), main_logger=str(__name__))
             log_statement(loglevel=str("info"), logstatement=str(f"Contained metadata for {len(metadata)} tokenized files."), main_logger=str(__name__))

        except ImportError:
             log_statement(loglevel=str("error"), logstatement=str("json module not found. Cannot create JSON metadata file."), main_logger=str(__name__))
        except Exception as e:
             log_statement(loglevel=str("error"), logstatement=str(f"Failed to create DataLoader metadata file '{output_filepath}': {e}"), main_logger=str(__name__), exc_info=True)
             if output_filepath.exists():
                 try: output_filepath.unlink()
                 except OSError: pass

    def _compress_file(self, filepath: Path) -> Path:
        """Compresses the file using zstandard."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Compressing file: {filepath.name}"), main_logger=str(__name__))
        try:
            compressed_path = filepath.with_suffix('.zst')
            with open(filepath, 'rb') as f_in:
                with open(compressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdCompressor()
                    dctx.copy_stream(f_in, f_out)
            return compressed_path
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to compress file {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Compression failed: {e}")
            return None
        except FileNotFoundError:
            log_statement(loglevel=str("error"), logstatement=str(f"File not found during compression: {filepath}"), main_logger=str(__name__), exc_info=True)
            return None
    
    def _decompress_file(self, filepath: Path) -> Path:
        """Decompresses the file using zstandard."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Decompressing file: {filepath.name}"), main_logger=str(__name__))
        try:
            decompressed_path = filepath.with_suffix('')
            with open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
            return decompressed_path
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to decompress file {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Decompression failed: {e}")
            return None
        except FileNotFoundError:
            log_statement(loglevel=str("error"), logstatement=str(f"File not found during decompression: {filepath}"), main_logger=str(__name__), exc_info=True)
            return None

    def _save_processed(self, data, original_path: Path) -> Path | None:
        """Saves processed data to a file, compressing if necessary."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Saving processed data for {original_path.name}"), main_logger=str(__name__))
        try:
            # Generate a unique filename based on the source file
            original_path = original_path.resolve()
            if original_path.suffix == '.zst': original_path = original_path.with_suffix('')
            if original_path.suffix in ['.jsonl', '.json']: original_path = original_path.with_suffix('.json')
            if original_path.suffix == '.csv': original_path = original_path.with_suffix('.csv')
            if original_path.suffix == '.txt': original_path = original_path.with_suffix('.txt')
            if original_path.suffix == '.xlsx': original_path = original_path.with_suffix('.xlsx')
            if original_path.suffix == '.xls': original_path = original_path.with_suffix('.xls')
            if original_path.suffix == '.pdf': original_path = original_path.with_suffix('.pdf')
            if original_path.suffix == '.docx': original_path = original_path.with_suffix('.docx')
            if original_path.suffix == '.odt': original_path = original_path.with_suffix('.odt')
            if original_path.suffix == '.html': original_path = original_path.with_suffix('.html')
            if original_path.suffix == '.xml': original_path = original_path.with_suffix('.xml')
            if original_path.suffix == '.rtf': original_path = original_path.with_suffix('.rtf')
            if original_path.suffix == '.epub': original_path = original_path.with_suffix('.epub')
            if original_path.suffix == '.zip': original_path = original_path.with_suffix('.zip')
            if data is None: return None
            save_stem = original_path.stem; suffix = ""; is_numpy = False
            if isinstance(data, (pd.Series, pd.DataFrame)): suffix = ".csv"
            elif isinstance(data, (np.ndarray, cp.ndarray)): suffix = ".npy"; is_numpy = True
            else: log_statement(loglevel=str("error"), logstatement=str(f"Unsupported save type: {type(data)}"), main_logger=str(__name__)); return None
            if COMPRESSION_ENABLED: suffix += ".zst"
            save_path = PROCESSED_DATA_DIR / f"{save_stem}_processed{suffix}"
            if save_path.exists():
                log_statement(loglevel=str("warning"), logstatement=str(f"Processed file already exists: {save_path}. Overwriting."), main_logger=str(__name__))
                try: save_path.unlink()
                except OSError: log_statement(loglevel=str("error"), logstatement=str(f"Failed to remove existing processed file: {save_path}"), main_logger=str(__name__), exc_info=True)
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for saving (convert CuPy to NumPy if needed)
            data_to_save = data
            if cp is not np and isinstance(data, cp.ndarray):
                log_statement(loglevel=str("debug"), logstatement=str("Converting CuPy array to NumPy for saving."), main_logger=str(__name__))
                data_to_save = cp.asnumpy(data)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                # Ensure pandas data is ready (no specific conversion needed here unless GPU involved earlier)
                pass
            elif not isinstance(data, np.ndarray):
                log_statement(loglevel=str("error"), logstatement=str(f"Unexpected data type for saving: {type(data)}. Cannot save."), main_logger=str(__name__))
                return None

            # Save the processed data
            try:
                if COMPRESSION_ENABLED:
                    cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
                    with open(save_path, 'wb') as fd:
                        with cctx.stream_writer(fd) as compressor:
                            if is_numpy:
                                np.save(compressor, data_to_save, allow_pickle=False)
                            else: # pandas Series or DataFrame
                                csv_bytes = data_to_save.to_csv(index=False, header=isinstance(data_to_save, pd.DataFrame)).encode('utf-8')
                                compressor.write(csv_bytes)
                else: # Save uncompressed
                    if is_numpy:
                        np.save(save_path, data_to_save, allow_pickle=False)
                    else: # pandas Series or DataFrame
                        data_to_save.to_csv(save_path, index=False, header=isinstance(data_to_save, pd.DataFrame))

                log_statement(loglevel=str("info"), logstatement=str(f"Saved processed file: {save_path}"), main_logger=str(__name__))
                return save_path
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Save failed: {save_path}: {e}"), main_logger=str(__name__), exc_info=True)
                if save_path.exists(): 
                    try: 
                        os.remove(save_path)
                    except OSError: pass
                return None
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to save processed data for {original_path}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(original_path, status='error', error_message=f"Failed to save processed data: {e}")
            return None

    def _save_processed(self, processed_data, source_filepath: Path):
        """Saves processed data to a file, compressing if necessary."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Saving processed data for {source_filepath.name}"), main_logger=str(__name__))

# --- Data Repository ---
class DataRepository:
    # Manages a state repository tracking dataset files and their processing status.
    # Uses Zstandard compression for the repository file.
    def __init__(self, repo_path: str | Path = None):
        self.repo_path = Path(repo_path or DataProcessingConfig.REPO_FILE)
        self.repo_path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = { # Defined schema
            'source_filepath': 'str', 'base_dir': 'str', 'status': 'str',
            'processed_path': 'str', 'tokenized_path': 'str', 'file_size': 'int64',
            'file_hash': 'str', 'last_modified_scan': 'datetime64[ns, UTC]', # Store UTC
            'last_updated_repo': 'datetime64[ns, UTC]', 'error_message': 'str'
        }
        self.df = self._load_repo()
        self.lock = Lock()
        log_statement(loglevel=str("info"), logstatement=str(f"DataRepository initialized. Repo: {self.repo_path}"), main_logger=str(__name__))

    def _load_repo(self) -> pd.DataFrame:
        # (Load logic similar to previous, ensures UTC for datetimes)
        if self.repo_path.exists():
            log_statement(loglevel=str("info"), logstatement=str(f"Loading data repository: {self.repo_path}"), main_logger=str(__name__))
            try:
                dctx = zstd.ZstdDecompressor()
                with open(self.repo_path, 'rb') as ifh:
                    decompressed_data = dctx.decompress(ifh.read())
                pdf = pd.read_csv(io.BytesIO(decompressed_data), keep_default_na=False, dtype=str)
                for col in self.columns:
                    if col not in pdf.columns: pdf[col] = ''
                for col, dtype in self.columns.items():
                    try:
                        if 'datetime' in dtype:
                            # Convert columns after loading
                            df[COL_SIZE] = df[COL_SIZE].astype(np.int64)
                            # Convert mtime from float/int timestamp to datetime
                            df[COL_MTIME] = pd.to_datetime(df[COL_MTIME], unit='s')
                            # Attempt timezone localization, handling potential errors
                            if compute_backend == 'cudf':
                                # cuDF approach (if needed and different from pandas)
                                try:
                                    # Assuming cuDF requires similar handling or specific methods
                                    # Example: df[COL_MTIME] = df[COL_MTIME].dt.tz_localize('UTC') # Adjust if cuDF API differs
                                    # If cuDF handles naive timestamps differently or if errors occur, adapt this block
                                    log_statement(loglevel='debug', logstatement=f"Attempting timezone handling for cuDF mtime column in {repo_path}", main_logger=str(__name__))
                                    # If cuDF doesn't have tz_localize or it behaves differently, adjust logic here.
                                    # For now, assume pandas logic might apply or skip if problematic for cuDF.
                                    # df[COL_MTIME] = df[COL_MTIME].dt.tz_localize('UTC') # Example
                                except Exception as e: # Catch broader errors for cuDF if API is uncertain
                                    log_statement(loglevel='warning',
                                                logstatement=f"Could not handle timezone for {COL_MTIME} using cuDF in {repo_path}: {e}",
                                                main_logger=str(__name__))

                            else: # Pandas approach
                                if pd.__version__ >= '2.0.0':
                                    # Handle potential already-aware datetime columns (e.g., from older saves)
                                    if df[COL_MTIME].dt.tz is None:
                                        try:
                                            df[COL_MTIME] = df[COL_MTIME].dt.tz_localize('UTC')
                                        except TypeError as te: # Specifically catch errors like "Already tz-aware"
                                            log_statement(loglevel='warning',
                                                        logstatement=f"Could not localize timezone (possibly already aware) for {COL_MTIME} in {repo_path}: {te}",
                                                        main_logger=str(__name__))
                                        except Exception as e: # Catch other unexpected localization errors
                                            log_statement(loglevel='error',
                                                        logstatement=f"Unexpected error localizing timezone for {COL_MTIME} in {repo_path}: {e}",
                                                        main_logger=str(__name__))
                                    else:
                                        # If already timezone-aware, ensure it's UTC or convert
                                        try:
                                            df[COL_MTIME] = df[COL_MTIME].dt.tz_convert('UTC')
                                        except Exception as e:
                                            log_statement(loglevel='error',
                                                        logstatement=f"Error converting existing timezone to UTC for {COL_MTIME} in {repo_path}: {e}",
                                                        main_logger=str(__name__))

                                else:
                                    # Older pandas versions might handle this differently
                                    try:
                                        df[COL_MTIME] = df[COL_MTIME].dt.tz_localize('UTC')
                                    except TypeError as te:
                                        log_statement(loglevel='warning',
                                                        logstatement=f"Could not localize timezone (possibly already aware) for {COL_MTIME} in {repo_path} (Pandas < 2.0): {te}",
                                                        main_logger=str(__name__))
                                    except Exception as e:
                                        log_statement(loglevel='error',
                                                        logstatement=f"Unexpected error localizing timezone for {COL_MTIME} in {repo_path} (Pandas < 2.0): {e}",
                                                        main_logger=str(__name__))
                            # Ensure required columns exist, add if missing
                            for col in self.repo_columns:
                                if col not in df.columns:
                                    log_statement(loglevel='warning', logstatement=f"Column '{col}' missing in {repo_path}. Adding with default values.", main_logger=str(__name__))
                                    if col == COL_ERROR:
                                        df[col] = '' # Or None
                                    elif col == COL_STATUS:
                                        df[col] = STATUS_LOADED # Default status for newly added column
                                    else:
                                        df[col] = pd.NA # Use appropriate null representation

                            # Select and order columns
                            df = df[self.repo_columns]

                            log_statement(loglevel='info', logstatement=f"Loaded repository: {repo_path} with {len(df)} entries.", main_logger=str(__name__))
                            self.df = df
                        elif dtype == 'str': pdf[col] = pdf[col].fillna('').astype(str)
                        else: pdf[col] = pdf[col].astype(dtype) # Try direct conversion*
                    except Exception as e:
                        log_statement(loglevel=str("error"), logstatement=str(f"Error converting repo column '{col}' to '{dtype}': {e}. Keeping string."), main_logger=str(__name__), exc_info=False)
                        pdf[col] = pdf[col].astype(str).fillna('')
                log_statement(loglevel=str("info"), logstatement=str(f"Repository loaded ({len(pdf)} entries)."), main_logger=str(__name__))
                return pdf
            except Exception as e: log_statement(loglevel=str("error"), logstatement=str(f"Repo load failed: {e}. Initializing empty."), main_logger=str(__name__), exc_info=True)
        else: log_statement(loglevel=str("info"), logstatement=str("No repo found. Initializing empty."), main_logger=str(__name__), exc_info=False)
        # Return empty DataFrame matching the schema
        pdf = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in self.columns.items()})
        # Ensure datetime columns are explicitly UTC
        for col, dtype in self.columns.items():
            if 'datetime' in dtype:
                # Check if the column is timezone-naive
                if pdf[col].dt.tz is None:
                    pdf[col] = pdf[col].dt.tz_localize(timezone.utc)
                elif pdf[col].dt.tz != timezone.utc:
                    pdf[col] = pdf[col].dt.tz_convert(timezone.utc)
                else: # Already aware and already UTC, do nothing
                    pass
            return pdf

    def save(self):
        """Saves the repository DataFrame to a compressed CSV file atomically."""
        if not self.lock: return
        temp_path = self.repo_path.with_suffix(f'.tmp_{int(time.time())}')
        log_statement(loglevel=str("debug"), logstatement=str(f"Attempting to save repository to temporary file: {temp_path}"), main_logger=str(__name__))
        try:
            with self.lock: # Acquire lock
                pdf = self.df.copy() # Work on a copy within the lock
                # Ensure string columns don't contain NaN/NaT before saving
                for col, dtype in self.columns.items():
                    if dtype == 'str' and col in pdf.columns: pdf[col] = pdf[col].fillna('').astype(str)
                    # Handle NaT for datetime if needed (pandas handles it for CSV)
                    if 'datetime' in dtype and col in pdf.columns: pass # pdf[col] = pdf[col].fillna(pd.NaT) # Ensure NaT, not None

                    # Compress and write using configured level
                    repo_path = Path(self.repo_file_path)
                    repo_path.parent.mkdir(parents=True, exist_ok=True)
                    log_statement(loglevel='info', logstatement=f"Attempting to save repository to {repo_path}...", main_logger=str(__name__))

                    # ---> MODIFICATION START <---
                    try:
                        # Use stream compression for potentially large files
                        cctx = zstd.ZstdCompressor(level=self.compression_level)
                        with repo_path.open("wb") as f_out:
                            with cctx.stream_writer(f_out) as writer:
                                # Ensure mtime is converted back to timestamp float/int for saving if needed
                                # Or ensure the reader handles datetime objects correctly
                                # df_to_save = self.df.copy() # Avoid modifying self.df directly if needed
                                # if pd.api.types.is_datetime64_any_dtype(df_to_save[COL_MTIME]):
                                #    df_to_save[COL_MTIME] = df_to_save[COL_MTIME].astype(np.int64) // 10**9
                                # else: handle cases where it might already be int/float

                                # Write directly to the compressed stream
                                csv_buffer = io.StringIO()
                                self.df.to_csv(csv_buffer, index=False, encoding='utf-8')
                                csv_buffer.seek(0)
                                # Use stream_compress_lines or write buffer directly if appropriate
                                # writer.write(csv_buffer.getvalue().encode('utf-8')) # Simpler approach
                                # For line-by-line compression (if stream_compress_lines is designed for it):
                                for line in csv_buffer:
                                    writer.write(line.encode('utf-8'))


                        log_statement(loglevel='info', logstatement=f"Successfully saved repository ({len(self.df)} entries) to {repo_path}", main_logger=str(__name__))

                    except Exception as e:
                        log_statement(loglevel='critical', # Use critical as saving repo is vital
                                    logstatement=f"CRITICAL ERROR saving repository to {repo_path}: {e}",
                                    main_logger=str(__name__), exc_info=True)

            # Atomic replace (outside lock)
            os.replace(temp_path, self.repo_path)
            log_statement(loglevel=str("info"), logstatement=str(f"Repository saved successfully ({len(pdf)} entries) to: {self.repo_path}"), main_logger=str(__name__))
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Repository save failed: {e}"), main_logger=str(__name__), exc_info=True)
            if temp_path.exists():
                try: os.remove(temp_path)
                except OSError: pass
            log_statement(loglevel=str("error"), logstatement=str(f"Temporary file {temp_path} removed after save failure."), main_logger=str(__name__))
        finally:
            # Ensure lock is released even if save fails
            if self.lock.locked():
                self.lock.release()
                log_statement(loglevel=str("debug"), logstatement=str("Repository lock released after save attempt."), main_logger=str(__name__))

    def scan_and_update(self, base_dir: Path):
        """Recursively scans base_dir, updates repo with new/modified files."""
        # (Scan logic similar, ensures base_dir is stored, uses UTC timestamps)
        if not self.lock: return
        base_dir = base_dir.resolve() # Use absolute path
        log_statement(loglevel=str("info"), logstatement=str(f"Scanning for files in: {base_dir}"), main_logger=str(__name__))
        found, processed, skipped, new_files, updated_files = 0, 0, 0, 0, 0
        base_dir_str = str(base_dir)

        with self.lock: # Lock for reading existing files
            existing_files = {row['source_filepath']: {'hash': row['file_hash'], 'mtime': row['last_modified_scan']}
                              for idx, row in self.df.iterrows() if row['source_filepath']}

        files_to_update = {} # Store updates to apply after scan
    
        for root, _, files in os.walk(base_dir):
            current_dir = Path(root)
            for filename in files:
                 file_path = current_dir / filename
                 abs_file_path_str = str(file_path.resolve())
                 file_ext = file_path.suffix.lower()
                 if not file_ext or file_ext not in DataProcessingConfig.SCAN_FILE_EXTENSIONS:
                      if file_ext != '.zst' and not filename.endswith('.tmp'): skipped_count += 1 # Ignore repo/temp files
                      continue
                 found += 1
                 try:
                     current_stat = file_path.stat()
                     # Get mtime, make it timezone-aware (use UTC for consistency)
                     current_mtime = pd.Timestamp(datetime.fromtimestamp(current_stat.st_mtime, tz=timezone.utc))
                     current_size = current_stat.st_size
                     existing_info = existing_files.get(abs_file_path_str)
                     needs_processing = False

                     if existing_info: # Check if modified
                         # Compare UTC timestamps
                         if current_mtime > existing_info.get('mtime', pd.Timestamp(0, tz='UTC')):
                             current_hash = self._calculate_hash(file_path)
                             if current_hash != existing_info.get('hash'):
                                 log_statement(loglevel=str("info"), logstatement=str(f"Modified file detected: {file_path.name}"), main_logger=str(__name__))
                                 needs_processing = True; updated_files += 1
                             else: # Only mtime changed, update repo mtime but don't re-process
                                 files_to_update[abs_file_path_str] = {'last_modified_scan': current_mtime}
                         # else: file is unchanged
                     else: # New file
                         log_statement(loglevel=str("info"), logstatement=str(f"New file discovered: {file_path.name}"), main_logger=str(__name__))
                         needs_processing = True; new_files += 1; needs_update = True

                     if needs_processing:
                         current_hash = self._calculate_hash(file_path) # Ensure hash is calculated
                         files_to_update[abs_file_path_str] = {
                             'base_dir': base_dir_str, # Store the base dir for this scan
                             'status': 'discovered',
                             'file_size': current_size,
                             'file_hash': current_hash,
                             'last_modified_scan': current_mtime,
                             'processed_path': '', 'tokenized_path': '', 'error_message': ''
                         }
                         processed += 1

                     else:
                         # File is new
                         log_statement(loglevel=str("info"), logstatement=str(f"Discovered new file: {file_path.name}"), main_logger=str(__name__))
                         new_files_count += 1

                     # If new or modified, add/update entry with 'discovered' status
                     if needs_update:
                         current_hash = self._calculate_hash(file_path) # Calculate hash if needed
                         update_data = {
                             'status': 'discovered',
                             'file_size': current_size,
                             'file_hash': current_hash,
                             'last_modified_scan': current_mtime, # Store file's modification time
                             'processed_path': '', # Clear old paths on update
                             'tokenized_path': '',
                             'error_message': ''
                         }
                         self.update_entry(file_path, **update_data)
                         if existing_info: updated_files_count +=1 # Count as update if it existed before
                         processed_count += 1

                 except FileNotFoundError:
                      log_statement(loglevel=str("warning"), logstatement=str(f"File disappeared during scan: {file_path}"), main_logger=str(__name__))
                      # Optionally remove from repo if needed: self.remove_entry(file_path)
                      skipped_count += 1
                 except Exception as e:
                      log_statement(loglevel=str("error"), logstatement=str(f"Error scanning file {file_path}: {e}"), main_logger=str(__name__), exc_info=True)
                      # Add/update with error status? Or just skip? Let's skip for scan.
                      skipped_count += 1
        # Apply updates to the repository (thread-safe via update_entry)
        if files_to_update:
             log_statement(loglevel=str("info"), logstatement=str(f"Applying {len(files_to_update)} repository updates..."), main_logger=str(__name__))
             # Potential optimization: Batch updates if performance becomes an issue
             for file_path_str, update_data in files_to_update.items():
                 self.update_entry(Path(file_path_str), **update_data)
             self.save() # Save after applying all updates

        log_statement(loglevel=str("info"), logstatement=str(f"Scan complete. Found: {found}, Processed (New/Updated): {processed}, Skipped: {skipped}. (New: {new_files}, Updated: {updated_files})"), main_logger=str(__name__))
        # Save changes after scan completes
        self.save()
        log_statement(loglevel=str("info"), logstatement=str(f"Repository updated with {processed_count} new/modified files."), main_logger=str(__name__))

    def update_entry(self, source_filepath: Path, **kwargs):
        """Adds or updates an entry using absolute paths. Thread-safe."""
        if not self.lock: return

        source_filepath_abs_str = str(source_filepath.resolve())
        update_data = kwargs.copy()
        update_data['last_updated_repo'] = pd.Timestamp.now(tz=timezone.utc) # Use UTC

        # Ensure paths are absolute strings or empty
        for key in ['base_dir', 'processed_path', 'tokenized_path']:
             if key in update_data:
                 val = update_data[key]
                 update_data[key] = str(val.resolve()) if isinstance(val, Path) else (str(val) if val else '')

        if 'error_message' in update_data: update_data['error_message'] = str(update_data.get('error_message', ''))[:1024]

        # Ensure timestamp consistency (UTC)
        for key in ['last_modified_scan', 'last_updated_repo']:
             if key in update_data:
                 ts = pd.to_datetime(update_data[key], errors='coerce')
                 if pd.notna(ts):
                      if ts.tzinfo is None: update_data[key] = ts.tz_localize(timezone.utc)
                      elif ts.tzinfo != timezone.utc: update_data[key] = ts.tz_convert(timezone.utc)
                      else: update_data[key] = ts # Already correct
                 else: update_data[key] = pd.NaT # Store NaT if conversion fails
        # Ensure file size is an integer
        if 'file_size' in update_data:
             try: update_data['file_size'] = int(update_data['file_size'])
             except (ValueError, TypeError): update_data['file_size'] = 0
        # Ensure file hash is a string
        if 'file_hash' in update_data:
             update_data['file_hash'] = str(update_data['file_hash'])[:1024]
        # Ensure status is a string
        if 'status' in update_data:
             update_data['status'] = str(update_data['status'])[:1024]
        # Ensure error message is a string
        if 'error_message' in update_data:
             update_data['error_message'] = str(update_data['error_message'])[:1024]
        # Ensure processed_path and tokenized_path are strings
        if 'processed_path' in update_data:
             update_data['processed_path'] = str(update_data['processed_path'])[:1024]
        if 'tokenized_path' in update_data:
             update_data['tokenized_path'] = str(update_data['tokenized_path'])[:1024]
        # Ensure source_filepath is a string
        if 'source_filepath' in update_data:
             update_data['source_filepath'] = str(update_data['source_filepath'])[:1024]
        # Ensure base_dir is a string
        if 'base_dir' in update_data:
             update_data['base_dir'] = str(update_data['base_dir'])[:1024]
        # Ensure all columns are present in the DataFrame
        for col in self.columns:
             if col not in self.df.columns:
                 self.df[col] = pd.Series(dtype=self.columns[col])

        with self.lock:
            mask = self.df['source_filepath'] == source_filepath_abs_str
            indices = self.df.index[mask].tolist()

            if indices: # Entry exists, update it
                idx = indices[0] # Assume unique paths, take first if multiple somehow
                log_statement(loglevel=str("debug"), logstatement=str(f"Updating repo entry for: {source_filepath.name}"), main_logger=str(__name__))
                for col, value in update_data.items():
                    if col in self.df.columns:
                        try:
                            target_dtype = self.df[col].dtype
                            self.df.loc[idx, col] = value # Assign directly (assumes type conversion handled above)
                            if pd.isna(value):
                                # Assign appropriate NA based on dtype
                                value_to_assign = pd.NaT if 'datetime' in str(target_dtype) else (0 if 'int' in str(target_dtype) else '' if target_dtype == object else None)
                            elif 'datetime' in str(target_dtype) and not isinstance(value, pd.Timestamp):
                                value_to_assign = pd.Timestamp(value)
                            elif 'int' in str(target_dtype) and not isinstance(value, (int, np.integer)):
                                value_to_assign = int(float(value)) # Allow float conversion
                            elif target_dtype == object or target_dtype == 'string': # Handle string types
                                value_to_assign = str(value) if value is not None else ''
                            else:
                                value_to_assign = value # Assume compatible type

                            self.df.loc[idx, col] = value_to_assign
                        except Exception as e:
                            log_statement(loglevel=str("error"), logstatement=str(f"Failed to assign value '{value}' (type {type(value)}) to column '{col}' (dtype {target_dtype}) at index {idx}: {e}. Assigning as string."), main_logger=str(__name__))
                            self.df.loc[idx, col] = str(value) # Fallback assign as string
                    else:
                        log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to update non-existent repo column '{col}'"), main_logger=str(__name__))
            else: # New entry
                log_statement(loglevel=str("debug"), logstatement=str(f"Adding new repo entry for: {source_filepath.name}"), main_logger=str(__name__))
                # Prepare data for new row, ensuring all columns are present
                new_data = {col: None for col in self.columns} # Start with None for all columns
                # Fill basic info calculated during scan (should be in kwargs if called from scan)
                new_data.update({
                    'source_filepath': source_filepath_abs_str,
                    'status': 'discovered', # Default status
                    'file_size': kwargs.get('file_size', 0),
                    'file_hash': kwargs.get('file_hash', ''),
                    'last_modified_scan': kwargs.get('last_modified_scan', pd.NaT),
                    'error_message': ''
                })
                # Override with any other provided kwargs
                new_data.update(kwargs)

                # Create DataFrame for the new row and ensure types before concat
                new_row_pdf = pd.DataFrame([new_data])
                for col in self.df.columns:
                    target_dtype = self.df[col].dtype
                    try:
                        if pd.isna(new_row_pdf.loc[0, col]): # Check if value is NaN/NaT/None
                            if 'datetime' in str(target_dtype): new_row_pdf.loc[0, col] = pd.NaT
                            elif 'int' in str(target_dtype): new_row_pdf.loc[0, col] = 0
                            else: new_row_pdf.loc[0, col] = '' # Default NA for strings
                        elif 'datetime' in target_dtype: new_row_pdf[col] = pd.to_datetime(new_row_pdf[col], errors='coerce').dt.tz_localize(timezone.utc)
                        elif 'int' in target_dtype: new_row_pdf[col] = pd.to_numeric(new_row_pdf[col], errors='coerce').fillna(0).astype('int64')
                        elif target_dtype == 'str': new_row_pdf[col] = new_row_pdf[col].astype(str).fillna('')
                        # Add other specific type handling if necessary
                        elif target_dtype == object or target_dtype == 'string': new_row_pdf[col] = new_row_pdf[col].astype(str).fillna('')
                        elif self.df[col].dtype != new_row_pdf[col].dtype: # General type conversion if needed
                            new_row_pdf[col] = new_row_pdf[col].astype(target_dtype)
                    except Exception as e:
                        log_statement(loglevel=str("error"), logstatement=str(f"Error ensuring type for new row column '{col}' (target: {target_dtype}): {e}. Converting to string."), main_logger=str(__name__))
                        new_row_pdf[col] = new_row_pdf[col].astype(str).fillna('')
                        # Ensure target column is also string for concat
                        if self.df[col].dtype != object and self.df[col].dtype != 'string':
                            self.df[col] = self.df[col].astype(str).fillna('')

                # Concatenate new row
                self.df = pd.concat([self.df, new_row_pdf[self.df.columns]], ignore_index=True)
                log_statement(loglevel=str("info"), logstatement=str(f"New entry (' {self.df} ') added for: {source_filepath.name}"), main_logger=str(__name__))

    def get_status(self, source_filepath: Path) -> str | None:
        """Gets the current status of a file."""
        if not self.lock:
             log_statement(loglevel=str("error"), logstatement=str("Repository lock not initialized. Cannot get status."), main_logger=str(__name__))
             return None

        source_filepath_str = str(source_filepath.resolve())
        with self.lock:
             is_cudf = GPU_AVAILABLE and cudf is not None and hasattr(self.df, 'to_pandas')
             if is_cudf:
                 entry = self.df[self.df['source_filepath'] == source_filepath_str]
                 return entry['status'].iloc[0] if not entry.empty else None
             else: # pandas
                 entry = self.df[self.df['source_filepath'] == source_filepath_str]
                 return entry['status'].iloc[0] if not entry.empty else None

    def get_files_by_status(self, status: Union[str, List[str]], base_dir: Optional[Path] = None) -> List[Path]:
        """Gets absolute source file paths by status, optionally filtered by base_dir."""
        if not self.lock: return []
        with self.lock:
            statuses = [status] if isinstance(status, str) else status
            mask = self.df['status'].isin(statuses)
            if base_dir:
                 base_dir_str = str(base_dir.resolve())
                 # Filter where source_filepath starts with the base_dir path string
                 # Assumes base_dir is stored consistently (e.g., always resolved)
                 mask &= self.df['source_filepath'].str.startswith(base_dir_str)
            paths = [Path(p) for p in self.df.loc[mask, 'source_filepath'].tolist() if p]
            return paths

    def get_processed_path(self, source_filepath: Path) -> Path | None:
        """Gets the absolute processed path for a given absolute source path."""
        if not self.lock: return None
        source_filepath_abs_str = str(source_filepath.resolve())
        with self.lock:
            entry = self.df[self.df['source_filepath'] == source_filepath_abs_str]
            processed_path_str = entry['processed_path'].iloc[0] if not entry.empty else None
            return Path(processed_path_str) if processed_path_str else None

    def _calculate_hash(self, filepath: Path, chunk_size=8192) -> str:
        """Calculates BLAKE2b hash."""
        if not filepath.is_file(): return ''
        h = hashlib.blake2b()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(chunk_size): h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Hash calculation failed for {filepath}: {str(e)}"), main_logger=str(__name__), exc_info=False)
            return ''

# --- Tokenizer ---
class Tokenizer:
    """ Loads processed, tokenizes, saves compressed tensors. """
    def __init__(self, max_workers: int | None = None):
        self.repo = DataRepository()
        resolved_max_workers = max_workers if max_workers is not None else DataProcessingConfig.MAX_WORKERS
        self.max_workers = max(1, resolved_max_workers)
        log_statement(loglevel=str("info"), logstatement=str(f"Initializing Tokenizer with max_workers={self.max_workers}"), main_logger=str(__name__))
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.device = DEFAULT_DEVICE
        TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        if hasattr(self, 'executor') and self.executor:
            try: self.executor.shutdown(wait=True);
            except Exception: pass

    def tokenize_all(self, base_dir_filter: Optional[Path] = None, statuses_to_process=('processed',)):
        """Tokenizes files matching status, optionally filtered by base_dir."""
        files_to_tokenize_info = []
        with self.repo.lock:
             # Use get_files_by_status which already handles filtering
             source_paths = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
             # Get corresponding processed paths
             for src_path in source_paths:
                 proc_path = self.repo.get_processed_path(src_path)
                 if proc_path and proc_path.exists(): # Ensure processed file exists
                     files_to_tokenize_info.append((src_path, proc_path))
                 elif proc_path:
                     log_statement(loglevel=str("warning"), logstatement=str(f"Processed path found in repo but file missing: {proc_path}. Skipping tokenization for {src_path.name}."), main_logger=str(__name__))
                 else:
                     log_statement(loglevel=str("warning"), logstatement=str(f"No valid processed path found in repo for source: {src_path.name}. Skipping tokenization."), main_logger=str(__name__))


        if not files_to_tokenize_info:
            log_statement(loglevel=str("info"), logstatement=str(f"No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] with existing processed files found to tokenize."), main_logger=str(__name__))
            return

        log_statement(loglevel=str("info"), logstatement=str(f"Starting tokenization for {len(files_to_tokenize_info)} files [base_dir: {base_dir_filter}]."), main_logger=str(__name__))
        # (Executor logic remains the same)
        futures = [self.executor.submit(self._tokenize_file, src_path, proc_path)
                   for src_path, proc_path in files_to_tokenize_info]
        with tqdm(total=len(futures), desc=f"Tokenizing Files [{base_dir_filter.name if base_dir_filter else 'All'}]") as pbar:
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: log_statement(loglevel=str("error"), logstatement=str(f"Error retrieving result from tokenizer future: {e}"), main_logger=str(__name__), exc_info=True)
                finally: pbar.update(1)
        self.repo.save()
        log_statement(loglevel=str("info"), logstatement=str("File tokenization complete."), main_logger=str(__name__))


    def _tokenize_file(self, source_filepath: Path, processed_filepath: Path):
        """Loads processed, converts to tensor, saves compressed."""
        # (Implementation remains same - uses modified load/save)
        log_statement(loglevel=str("debug"), logstatement=str(f"Tokenizing file: {processed_filepath.name}..."), main_logger=str(__name__))
        save_path = None
        try:
            self.repo.update_entry(source_filepath, status='tokenizing', error_message='', tokenized_path='')
            data = self._load_processed(processed_filepath) # Handles decompression
            if data is None: raise ValueError("Load processed failed.")
            tokenized_tensor = self._vectorize(data) # Convert to tensor
            if tokenized_tensor is None: raise ValueError("Vectorization failed.")
            save_path = self._save_tokenized(tokenized_tensor, processed_filepath) # Handles compression
            if save_path is None: raise IOError("Save tokenized failed.")
            self.repo.update_entry(source_filepath, status='tokenized', tokenized_path=save_path, error_message='')
            log_statement(loglevel=str("info"), logstatement=str(f"Tokenized: {processed_filepath.name} -> {save_path.name}"), main_logger=str(__name__))
            return source_filepath
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Tokenization failed for {processed_filepath.name}: {e}"), main_logger=str(__name__), exc_info=True)
            self.repo.update_entry(source_filepath, status='error', error_message=f"{e}")
            if save_path and save_path.exists():
                try: save_path.unlink()
                except OSError: pass
            return None

    def _load_processed(self, filepath: Path):
        """Loads compressed or uncompressed processed data (CSV or NPY)."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Loading processed: {filepath.name}"), main_logger=str(__name__))
        if not filepath.exists():
            log_statement(loglevel=str("error"), logstatement=str(f"Not found: {filepath}"), main_logger=str(__name__))
            return None
        try:
            data = None
            is_compressed = COMPRESSION_ENABLED and filepath.suffix == '.zst'
            # Determine the actual suffix before potential decompression
            actual_suffix = filepath.suffixes[-2].lower() if is_compressed and len(filepath.suffixes) > 1 else filepath.suffix.lower()

            file_to_open = filepath
            buffer = None # Initialize buffer outside the 'if'

            if is_compressed:
                dctx = zstd.ZstdDecompressor()
                with open(filepath, 'rb') as ifh:
                    decompressed_content = dctx.decompress(ifh.read())
                buffer = io.BytesIO(decompressed_content)
                file_to_open = buffer # type: ignore

            # Use the determined actual_suffix
            if actual_suffix == '.npy':
                # Reset buffer position if reading from it
                if buffer: buffer.seek(0)
                data = np.load(file_to_open, allow_pickle=False)
            elif actual_suffix == '.csv':
                # Reset buffer position if reading from it
                if buffer: buffer.seek(0)
                # Pass the buffer directly if it exists, otherwise the filepath
                df = pd.read_csv(file_to_open, header=0, keep_default_na=False)
                data = df.iloc[:, 0] if df.shape[1] == 1 else df
            else:
                log_statement(loglevel=str("error"), logstatement=str(f"Cannot load processed: Unknown suffix '{actual_suffix}' for file {filepath.name}"), main_logger=str(__name__))
                return None

            # Close the buffer if it was used
            if buffer: buffer.close()
            return data
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Load failed for {filepath}: {e}"), main_logger=str(__name__), exc_info=True)
            # Ensure buffer is closed on error too
            if buffer and not buffer.closed: buffer.close()
            return None

    def _vectorize(self, data) -> torch.Tensor | None:
        """Converts loaded processed data into a tensor."""
        # (Implementation remains same - placeholder for text)
        log_statement(loglevel=str("debug"), logstatement=str(f"Vectorizing data type: {type(data)}"), main_logger=str(__name__))
        try:
            if isinstance(data, torch.Tensor):
                return data.to(dtype=torch.float32, device=self.device)
            elif isinstance(data, (np.ndarray, cp.ndarray)):
                # Convert CuPy array to NumPy before converting to Torch tensor if necessary
                if cp is not np and isinstance(data, cp.ndarray):
                    log_statement(loglevel=str("debug"), logstatement=str("Converting CuPy array to NumPy before Torch tensor conversion."), main_logger=str(__name__))
                    data = cp.asnumpy(data)
                # Ensure data is C-contiguous before creating tensor
                if not data.flags['C_CONTIGUOUS']:
                    log_statement(loglevel=str("debug"), logstatement=str("Data is not C-contiguous, making a copy."), main_logger=str(__name__))
                    data = np.ascontiguousarray(data)
                return torch.tensor(data, dtype=torch.float32, device=self.device)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                log_statement(loglevel=str("warning"), logstatement=str("Vectorizing pandas data - using DUMMY implementation."), main_logger=str(__name__))
                # Convert pandas data to NumPy first
                numpy_data = data.to_numpy(dtype=np.float32)
                # Ensure data is C-contiguous before creating tensor
                if not numpy_data.flags['C_CONTIGUOUS']:
                     log_statement(loglevel=str("debug"), logstatement=str("Pandas data (as NumPy) is not C-contiguous, making a copy."), main_logger=str(__name__))
                     numpy_data = np.ascontiguousarray(numpy_data)
                # Dummy implementation: create zeros based on rows, fixed columns
                num_rows = numpy_data.shape[0]
                # Ensure the dummy tensor matches the expected shape if possible
                # If 1D (Series), create (num_rows, 1). If 2D (DataFrame), use original cols or dummy.
                num_cols = numpy_data.shape[1] if numpy_data.ndim > 1 else 1
                dummy_cols = 10 # Keep the dummy dimension for now
                log_statement(loglevel=str("debug"), logstatement=str(f"Creating dummy tensor of shape ({num_rows}, {dummy_cols})"), main_logger=str(__name__))
                return torch.zeros((num_rows, dummy_cols), dtype=torch.float32, device=self.device) # Dummy
            else:
                log_statement(loglevel=str("error"), logstatement=str(f"Unsupported type for vectorization: {type(data)}"), main_logger=str(__name__))
                return None
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Vectorization failed: {e}"), main_logger=str(__name__), exc_info=True)
            return None

    def _save_tokenized(self, tensor: torch.Tensor, original_processed_path: Path) -> Path | None:
        """Saves tokenized tensor compressed."""
        # Determine the stem correctly, handling potential double suffixes like .csv.zst
        processed_stem = original_processed_path.name
        if COMPRESSION_ENABLED and original_processed_path.suffixes[-1] == '.zst':
             processed_stem = original_processed_path.name.removesuffix('.zst') # Remove .zst first

        # Remove the original data suffix (.npy or .csv) and '_processed'
        if processed_stem.endswith('_processed.npy'):
             save_stem = processed_stem.removesuffix('_processed.npy')
        elif processed_stem.endswith('_processed.csv'):
             save_stem = processed_stem.removesuffix('_processed.csv')
        else:
             # Fallback if the naming convention isn't strictly followed
             save_stem = original_processed_path.stem.replace('_processed', '')
             log_statement(loglevel=str("warning"), logstatement=str(f"Processed path '{original_processed_path.name}' did not strictly follow '_processed.npy/csv' pattern. Using stem '{save_stem}'."), main_logger=str(__name__))


        suffix = ".pt.zst" if COMPRESSION_ENABLED else ".pt"
        save_path = TOKENIZED_DATA_DIR / f"{save_stem}_tokenized{suffix}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_statement(loglevel=str("debug"), logstatement=str(f"Saving tokenized tensor to: {save_path.name}"), main_logger=str(__name__), exc_info=False)
        try:
            # Ensure tensor is on CPU before saving
            tensor_cpu = tensor.cpu()

            if COMPRESSION_ENABLED:
                buffer = io.BytesIO()
                torch.save(tensor_cpu, buffer)
                buffer.seek(0)
                cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
                with open(save_path, 'wb') as f:
                    with cctx.stream_writer(f) as compressor:
                        compressor.write(buffer.read())
                buffer.close() # Close the BytesIO buffer
            else:
                torch.save(tensor_cpu, save_path)

            log_statement(loglevel=str("info"), logstatement=str(f"Saved tokenized tensor: {save_path}"), main_logger=str(__name__), exc_info=False)
            return save_path
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Save tokenized failed for {save_path}: {e}"), main_logger=str(__name__), exc_info=True)
            if save_path.exists():
                try:
                    save_path.unlink()
                    log_statement(loglevel=str("debug"), logstatement=str(f"Removed partially saved file: {save_path}"), main_logger=str(__name__), exc_info=False)
                except OSError as unlink_e:
                     log_statement(loglevel=str("error"), logstatement=str(f"Failed to remove partially saved file {save_path}: {unlink_e}"), main_logger=str(__name__), exc_info=False)
            return None