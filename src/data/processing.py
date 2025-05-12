# src/data/processing.py
"""
Data Processing Module

Handles data preprocessing pipelines including:
- Managing a repository of dataset files and their processing status.
- Processing raw data (text cleaning, numerical scaling).
- Tokenizing/vectorizing processed data into tensors.
Utilizes GPU acceleration (cuDF, CuPy, cuML) if available and configured.
"""
import shutil
import inspect
import os
import sys
import csv
import codecs
import re
from threading import Lock
from typing import Dict, Optional, Any, List, Generator
import pandas as pd
import time
import pandas.api.types
from pathlib import Path
import hashlib
import logging
import numpy as np
from src.data.readers import RobustTextReader, FileReader
from src.data.readers import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import io
import csv
import datetime as dt
from datetime import timezone # Import timezone
import zstandard as zstd
from src.core.repo_handler import DataRepository
from src.data.constants import *
from src.utils.config import *
from src.utils.compression import *
from src.utils.hashing import *
# Import project configuration and utilities
try:
    from src.utils.config import *
    from src.utils.logger import configure_logging, log_statement
    from src.utils.gpu_switch import *
    configure_logging()
    log_statement(loglevel='info', logstatement=f"{LOG_INS}::Logger and other utils imported successfully.", main_logger=__file__)
except ImportError:
    logging.ERROR("Failed relative import in processing.py, trying absolute from src...")
    try:
        from src.utils.config import *
        from src.utils.logger import configure_logging, log_statement
        configure_logging()
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Logger and other utils imported successfully.", main_logger=__file__)
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

# --- Optional GPU Library Imports ---
# (Implementation remains the same as previous version)
GPU_AVAILABLE = False; cudf = None; cp = np; CumlScaler = None; UnsupportedCUDAError = None

try:
    # Determine backend (cudf or pandas)
    compute_backend = get_compute_backend()
    if compute_backend == 'cudf':
        try:
            import cudf as pd
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::cuDF backend selected.", main_logger=__file__)
            IS_CUDA_AVAILABLE = True
        except ImportError:
            log_statement(loglevel='warning', logstatement=f"{LOG_INS}::cuDF requested but not available. Falling back to pandas.", main_logger=__file__)
            import pandas as pd
            IS_CUDA_AVAILABLE = False
    else:
        import pandas as pd
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Pandas backend selected.", main_logger=__file__)
        IS_CUDA_AVAILABLE = False

except ImportError as initial_import_err:
    log_statement(loglevel='error', logstatement=f"{LOG_INS}::GPU library 'cudf' not found ({initial_import_err}). Using CPU fallback.", main_logger=__file__)
    GPU_AVAILABLE = False
except Exception as initial_cudf_err:
    if UnsupportedCUDAError and isinstance(initial_cudf_err, UnsupportedCUDAError):
         log_statement(loglevel="warning", logstatement=f"{LOG_INS}::cuDF found but GPU is incompatible ({initial_cudf_err}). Using CPU fallback.", main_logger=__file__)
    else:
         log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error during initial cuDF import/setup: {initial_cudf_err}", main_logger=__file__, exc_info=True)
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Using CPU fallback (pandas/numpy).", main_logger=__file__)
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
            log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Using sklearn.preprocessing.StandardScaler as CPU fallback for scaling.", main_logger=__file__)
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
            log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Scikit-learn not found. Using basic dummy scaler (no operation).", main_logger=__file__)
            class CumlScaler_dummy:
                 def __init__(self, *args, **kwargs): pass
                 def fit_transform(self, data): return data
            CumlScaler = CumlScaler_dummy
# try:
from src.utils.helpers import *
#     log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor Helper functions assigned.", main_logger=__file__)
# except ImportError:
#     log_statement(loglevel='error', logstatement=f"{LOG_INS}::DataProcessor ERROR importing helper functions.", main_logger=__file__)
#     def dummy_save_parquet(*args, **kwargs): return False
#     def dummy_compress_string(*args, **kwargs): return False

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
                    log_statement(loglevel='info', logstatement=f"{LOG_INS}::Downloading NLTK '{resource_id}' data...", main_logger=__file__)
                    nltk.download(resource_id, quiet=True)
    download_nltk_data() # Consider calling this conditionally or manually

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    log_statement(loglevel='info', logstatement=f"{LOG_INS}::NLTK components loaded successfully.", main_logger=__file__)
except ImportError: log_statement(loglevel="warning", logstatement=f"{LOG_INS}::NLTK library not found. Text processing limited.", main_logger=__file__)
except LookupError as e: log_statement(loglevel="warning", logstatement=f"{LOG_INS}::NLTK data not found ({e}). Text processing limited.", main_logger=__file__)
except Exception as e: log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error loading NLTK: {e}", main_logger=__file__, exc_info=True)
if not NLTK_AVAILABLE:
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word # Add pos arg for compatibility
    lemmatizer = DummyLemmatizer()
    stop_words = set() # No stop words if NLTK not available

# Import readers needed for actual content processing
from src.data.readers import FileReader

# Placeholder for tokenization logic
# from src.core.tokenizer import tokenize_content # Assuming exists
init_logger_name = __file__ # Logger name for init context
content_string = None
output_filepath = None


# --- Data Processor ---
class DataProcessor:
    """ Scans, processes raw data, saves compressed output. """
    def __init__(self, repo_path_override: Optional[str | Path] = None, repo_dir: str = REPO_DIR, filename: str = MAIN_REPO_FILENAME, max_workers: int | None = None):
        # --- Path Setup ---
        init_logger_name = __file__ # Define logger name
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::DataProcessor class initialized for {__name__}  --  repo_path_override: {repo_path_override}  --  repo_dir: {repo_dir}  --  filename: {filename} ", main_logger=init_logger_name)
        # --- Determine Repository Path ---
        if repo_path_override:
            try:
                self.repo_filepath = Path(repo_path_override).resolve()
                self.repo_dir = self.repo_filepath.parent
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor using overridden repo path: {self.repo_filepath}", main_logger=init_logger_name)
            except Exception as path_e:
                log_statement(loglevel='critical', logstatement=f"{LOG_INS}::Invalid repo_path_override provided: {repo_path_override} - Error: {path_e}", main_logger=init_logger_name)
                raise ValueError("Invalid repository override path provided.") from path_e
        else:
            try:
                # Use Path(REPO_DIR) and MAIN_REPO_FILENAME constants directly
                self.repo_dir = Path(REPO_DIR).resolve() # REPO_DIR from constants.py
                self.repo_filepath = self.repo_dir / MAIN_REPO_FILENAME # MAIN_REPO_FILENAME from constants.py
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor using default repo path: {self.repo_filepath}", main_logger=init_logger_name)
            except Exception as e:
                log_statement(loglevel='critical', logstatement=f"{LOG_INS}::Invalid default repo_dir/filename: {self.repo_filepath}"
                            f"{LOG_INS}::Error: {e}", main_logger=init_logger_name)
                raise ValueError("Invalid default repository directory/filename.") from e
        print(f"{LOG_INS} > DEBUG: repo_filepath == {self.repo_filepath}")
        # Ensure the determined repository directory exists (using self.repo_dir)
        try:
             self.repo_dir.mkdir(parents=True, exist_ok=True)
             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Ensured repository directory exists: {self.repo_dir}", main_logger=init_logger_name)
        except Exception as mkdir_e:
            print(f"{LOG_INS}::DataProcessor CRITICAL - Failed to create repo dir: {self.repo_dir}")
            raise

        # Define base directory for processed output files (using constants from config)
        self.output_dir = Path(PROCESSED_DATA_DIR) # Ensure Path object
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Define base directory for tokenized output files
        self.tokenized_output_dir = Path(TOKENIZED_DATA_DIR) # Ensure Path object
        self.tokenized_output_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = "utf-8"
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor Output Dirs: Processed='{self.output_dir}', Tokenized='{self.tokenized_output_dir}'", main_logger=__file__)

        # --- Repository Initialization ---
        try:
            self.repo = DataRepository(repo_path=self.repo_filepath)
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor's internal DataRepository initialized with: {self.repo_filepath}", main_logger=init_logger_name)
            df_len = len(self.repo.df) if self.repo.df is not None else 'None'
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor internal DataRepository load complete. DF length: {len(self.repo.df) if self.repo.df is not None else 'None'}", main_logger=__file__)
            if self.repo.df is not None and not self.repo.df.empty:
                if COL_STATUS in self.repo.df.columns:
                    log_statement(loglevel='info', logstatement=f"{LOG_INS} >>> MORE DEBUG: Status column values IN LOADED DF (Top 10): {self.repo.df[COL_STATUS].head(10).tolist()}"
                                                                f"{LOG_INS} >>> MORE DEBUG: Status value counts IN LOADED DF:\n{self.repo.df[COL_STATUS].value_counts().to_string()}", main_logger=__file__)
                else:
                    log_statement(loglevel='critical', logstatement=f"{LOG_INS} >>> MORE DEBUG: CRITICAL - '{COL_STATUS}' column MISSING in loaded DataFrame!", main_logger=__file__, exc_info=True)
            elif self.repo.df is not None:
                log_statement(loglevel='error', logstatement=f"{LOG_INS} >>> MORE DEBUG: Loaded DataFrame is empty.", main_logger=__file__)
        except ValueError as init_ve:
            log_statement(loglevel='exception', logstatement=f"{LOG_INS}::DataRepository initialization failed: {init_ve}", main_logger=init_logger_name)
            raise
        except Exception as init_e:
            log_statement(loglevel='exception', logstatement=f"{LOG_INS}::Unexpected error initializing DataRepository: {init_e}", main_logger=init_logger_name, exc_info=True)
            raise

        # --- State Management ---
        self.lock = Lock()

        # --- Execution Setup ---
        try: config_max_workers = DataProcessingConfig.MAX_WORKERS
        except AttributeError: config_max_workers = 16
        resolved_max_workers = max_workers if max_workers is not None else config_max_workers
        self.max_workers = max(1, resolved_max_workers)
        print(f"{LOG_INS}::DataProcessor max_workers set to: {self.max_workers}")
        log_statement(loglevel="info", logstatement=f"{LOG_INS}::Initializing DataProcessor internals with max_workers={self.max_workers}", main_logger=init_logger_name)
        # Use ThreadPoolExecutor from previous fix
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='DataProc_Thread')
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::DataProcessor executor initialized as ThreadPoolExecutor.", main_logger=init_logger_name)

        # --- Processing Components (Scaler, Regex) ---
        try:
            self.scaler = CumlScaler()
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor Scaler initialized ({type(self.scaler).__name__}).", main_logger=__file__)
        except Exception as scaler_init_e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::DataProcessor Scaler initialization FAILED.  Cause: {scaler_init_e}.", main_logger=__file__, exc_info=True)
            self.scaler = None
        self.cleaning_regex = None
        try:
            regex_pattern = getattr(DataProcessingConfig, 'TEXT_CLEANING_REGEX', r'[^\w\s\-\.]')
            if regex_pattern:
                 self.cleaning_regex = re.compile(regex_pattern)
                 print(f"{LOG_INS}::DataProcessor Cleaning regex compiled: {regex_pattern}")
            else:
                 print(f"{LOG_INS}::DataProcessor No cleaning regex defined.")
        except Exception as re_err:
            log_statement(loglevel='error',
            logstatement=f"{LOG_INS}::Invalid cleaning regex pattern: {regex_pattern}. Error: {re_err}"
            f"{__name__}>>> DEBUG: {__file__}:{inspect.currentframe().f_lineno} DataProcessor ERROR compiling regex: {re_err}",
            main_logger=__file__)

        # --- Assign Helper Functions ---
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor initialized. Output dirs: Processed='{self.output_dir}', Tokenized='{self.tokenized_output_dir}'", main_logger=__file__)

    def save_dataframe_to_parquet_zst(self, df: pd.DataFrame, output_path: Path):
        """Saves a pandas DataFrame to a Zstandard-compressed Parquet file."""
        func_logger = __file__ # Use module name for logger
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

    def _classify_data(self, filepath: Path, content: str | bytes) -> str:
        """
        Classifies the data type based on file extension and content analysis.

        Args:
            filepath (Path): The path to the file.
            content (str | bytes): The content of the file (or a sample).

        Returns:
            str: The classified data type string (e.g., TYPE_TEXTUAL, TYPE_CODE).
        """
        log_statement('debug', f"{LOG_INS}:DEBUG>>_classify_data called for {filepath}", __file__)
        extension = filepath.suffix.lower()
        classified_type = TYPE_UNKNOWN # Default

        # --- Stage 1: Extension-based Classification (Common & Relatively Unambiguous) ---
        log_statement('debug', f"{LOG_INS}:DEBUG>>Classifying based on extension: {extension}", __file__)
        if extension == '.pdf':
            classified_type = TYPE_PDF
        elif extension == '.html' or extension == '.htm':
            classified_type = TYPE_HTML
        elif extension == '.xml':
            classified_type = TYPE_XML
        elif extension == '.json':
            classified_type = TYPE_JSON
        elif extension == '.jsonl':
            classified_type = TYPE_JSONL
        elif extension == '.yaml' or extension == '.yml':
            classified_type = TYPE_YAML
        elif extension == '.md':
            classified_type = TYPE_MARKDOWN
        elif extension in ['.py', '.pyw', '.java', '.js', '.ts', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.php', '.rb', '.pl', '.sh', '.ps1', '.lua', '.swift', '.kt', '.kts']:
            classified_type = TYPE_CODE
        elif extension in ['.csv', '.tsv']: # Often requires content check too
            classified_type = TYPE_TABULAR
        elif extension == '.txt':
             classified_type = TYPE_TEXTUAL # Tentative, needs content check
        elif extension == '.png':
             classified_type = TYPE_IMAGE_PNG
        elif extension in ['.jpg', '.jpeg']:
             classified_type = TYPE_IMAGE_JPEG
        elif extension == '.gif':
             classified_type = TYPE_IMAGE_GIF
        elif extension == '.bmp':
             classified_type = TYPE_IMAGE_BMP
        elif extension == '.wav':
             classified_type = TYPE_AUDIO_WAV
        elif extension == '.mp3':
             classified_type = TYPE_AUDIO_MP3 # Tentative
        elif extension == '.zip':
             classified_type = TYPE_COMPRESSED_ZIP
        elif extension == '.gz':
             classified_type = TYPE_COMPRESSED_GZIP
        elif extension == '.bz2':
             classified_type = TYPE_COMPRESSED_BZ2
        elif extension == '.7z':
             classified_type = TYPE_COMPRESSED_7Z
        elif extension == '.rar':
             classified_type = TYPE_COMPRESSED_RAR
        elif extension == '.zst':
             classified_type = TYPE_COMPRESSED_ZSTD
        # Add more extensions here if needed

        log_statement('debug', f"{LOG_INS}:DEBUG>>Initial classification by extension: {classified_type}", __file__)

        # --- Stage 2: Content-based Classification (Magic Numbers & Structure/Syntax Analysis) ---
        if content is None or len(content) == 0:
            log_statement('warning', f"{LOG_INS}:WARNING>>Content is empty for {filepath}. Cannot perform content-based classification. Relying on extension: {classified_type}", __file__)
            return classified_type if classified_type != TYPE_UNKNOWN else TYPE_TEXTUAL # Default to TEXTUAL for empty files? Or keep UNKNOWN?

        log_statement('debug', f"{LOG_INS}:DEBUG>>Performing content-based classification for {filepath}. Content type: {type(content).__name__}, Length: {len(content)}", __file__)

        try:
            # --- Magic Number Checks (primarily for bytes content) ---
            # Ensure we check bytes if possible, even if decoded str is available
            content_bytes = None
            if isinstance(content, bytes):
                content_bytes = content
            elif isinstance(content, str):
                 try:
                    # Attempt to get bytes if needed, be careful with encoding choice
                    # Using utf-8 might fail for binary files, latin-1 is safer for byte preservation
                    content_bytes = content.encode('latin-1') # Reversible encoding for byte checking
                 except Exception as enc_err:
                     log_statement('warning', f"{LOG_INS}:WARNING>>Could not encode string content to bytes for magic number check: {enc_err}", __file__)

            if content_bytes:
                log_statement('debug', f"{LOG_INS}:DEBUG>>Checking magic numbers using bytes (first 16 bytes: {content_bytes[:16]!r})", __file__)
                # PDF
                if content_bytes.startswith(b'%PDF-'):
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: PDF", __file__)
                    return TYPE_PDF
                # Images
                elif content_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: PNG", __file__)
                    return TYPE_IMAGE_PNG
                elif content_bytes.startswith(b'\xff\xd8\xff'): # JPEG
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: JPEG", __file__)
                    return TYPE_IMAGE_JPEG
                elif content_bytes.startswith(b'GIF87a') or content_bytes.startswith(b'GIF89a'):
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: GIF", __file__)
                    return TYPE_IMAGE_GIF
                elif content_bytes.startswith(b'BM'): # BMP
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: BMP", __file__)
                    return TYPE_IMAGE_BMP
                # Audio
                elif content_bytes.startswith(b'RIFF') and content_bytes[8:12] == b'WAVE':
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: WAV", __file__)
                    return TYPE_AUDIO_WAV
                elif content_bytes.startswith(b'ID3'): # MP3 often starts with ID3 tag
                     log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: MP3 (ID3 tag)", __file__)
                     # Note: Lack of ID3 doesn't mean it's not MP3. More robust checks needed if critical.
                     return TYPE_AUDIO_MP3
                # Compressed
                elif content_bytes.startswith(b'PK\x03\x04') or content_bytes.startswith(b'PK\x05\x06') or content_bytes.startswith(b'PK\x07\x08'):
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: ZIP", __file__)
                    return TYPE_COMPRESSED_ZIP
                elif content_bytes.startswith(b'\x1f\x8b'): # Gzip
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: GZIP", __file__)
                    return TYPE_COMPRESSED_GZIP
                elif content_bytes.startswith(b'BZh'): # Bzip2
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: BZIP2", __file__)
                    return TYPE_COMPRESSED_BZ2
                elif content_bytes.startswith(b'7z\xBC\xAF\x27\x1C'): # 7zip
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: 7ZIP", __file__)
                    return TYPE_COMPRESSED_7Z
                elif content_bytes.startswith(b'Rar!\x1a\x07\x00') or content_bytes.startswith(b'Rar!\x1a\x07\x01\x00'): # RAR
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: RAR", __file__)
                    return TYPE_COMPRESSED_RAR
                elif content_bytes[0:4] in [b'\x28\xB5\x2F\xFD', b'\xFD\x2F\xB5\x28']: # Zstandard (Little/Big Endian)
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Magic number match: ZSTD", __file__)
                    return TYPE_COMPRESSED_ZSTD

            # --- Content Analysis (primarily for string content) ---
            content_str = None
            if isinstance(content, str):
                content_str = content
            elif isinstance(content, bytes):
                # Try decoding with common encodings if it wasn't identified by magic numbers
                try:
                    # Detect BOM first
                    bom = codecs.BOM_UTF8
                    if content.startswith(bom):
                         content_str = content[len(bom):].decode('utf-8', errors='strict')
                         log_statement('debug', f"{LOG_INS}:DEBUG>>Decoded content using UTF-8 (BOM detected)", __file__)
                    else:
                         # Try UTF-8 first (most common)
                         content_str = content.decode('utf-8', errors='strict')
                         log_statement('debug', f"{LOG_INS}:DEBUG>>Decoded content using UTF-8", __file__)
                except UnicodeDecodeError:
                    log_statement('debug', f"{LOG_INS}:DEBUG>>UTF-8 decoding failed, trying latin-1", __file__)
                    try:
                        # Fallback to latin-1 (preserves bytes)
                        content_str = content.decode('latin-1', errors='strict')
                        log_statement('debug', f"{LOG_INS}:DEBUG>>Decoded content using latin-1", __file__)
                    except Exception as dec_err:
                         log_statement('warning', f"{LOG_INS}:WARNING>>Could not decode byte content to string for analysis: {dec_err}", __file__)
                         # Stick with extension-based type or potentially classify as BINARY
                         return classified_type if classified_type != TYPE_UNKNOWN else TYPE_BINARY


            if content_str is not None:
                # Reduce analysis scope for performance on large files (e.g., first 1MB or 1000 lines)
                analysis_sample = content_str[:1024*1024] # Analyze first 1MB
                lines = analysis_sample.splitlines()
                num_lines = len(lines)
                sample_lines = lines[:1000] # Analyze first 1000 lines

                # JSON Check (override extension if content matches)
                if extension != '.jsonl' and (classified_type == TYPE_JSON or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN):
                     # Quick check for common JSON start/end characters
                     if analysis_sample.strip().startswith(('{', '[')) and analysis_sample.strip().endswith(('}', ']')):
                          try:
                               json.loads(analysis_sample) # Try parsing the sample
                               log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: JSON", __file__)
                               return TYPE_JSON
                          except json.JSONDecodeError:
                               log_statement('debug', f"{LOG_INS}:DEBUG>>Content looks like JSON but failed parsing (might be truncated or invalid).", __file__)
                               # Keep previous classification or mark as potentially textual?
                          except Exception as json_err:
                               log_statement('warning', f"{LOG_INS}:WARNING>>Error during JSON check: {json_err}", __file__, exc_info=True)

                # JSONL Check (override extension if content matches)
                if classified_type == TYPE_JSONL or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     is_jsonl = True
                     if not sample_lines: is_jsonl = False # Cannot be JSONL if empty
                     parsed_lines = 0
                     for i, line in enumerate(sample_lines):
                          line = line.strip()
                          if not line: continue # Allow empty lines
                          if not (line.startswith(('{', '[')) and line.endswith(('}', ']'))): # Quick check
                               is_jsonl = False
                               log_statement('debug', f"{LOG_INS}:DEBUG>>Line {i+1} doesn't look like JSON object/array, breaking JSONL check.", __file__)
                               break
                          try:
                               json.loads(line)
                               parsed_lines += 1
                          except json.JSONDecodeError:
                               log_statement('debug', f"{LOG_INS}:DEBUG>>JSONL check failed parsing line {i+1}", __file__)
                               is_jsonl = False
                               break
                          except Exception as jsonl_err:
                               log_statement('warning', f"{LOG_INS}:WARNING>>Error during JSONL check on line {i+1}: {jsonl_err}", __file__, exc_info=True)
                               is_jsonl = False
                               break
                     # Require at least one successfully parsed line for confidence
                     if is_jsonl and parsed_lines > 0:
                          log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: JSONL", __file__)
                          return TYPE_JSONL

                # XML Check
                if classified_type == TYPE_XML or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     # Simple regex checks for XML declaration and tags
                     if re.search(r'<\?xml\s+version=', analysis_sample, re.IGNORECASE) or \
                        re.search(r'<(\w+)\s*[^>]*>', analysis_sample): # Look for opening tags
                          # More robust: try parsing with ElementTree if needed
                          log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: XML", __file__)
                          return TYPE_XML

                # HTML Check
                if classified_type == TYPE_HTML or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     # Look for common HTML tags, doctype
                     if re.search(r'<!DOCTYPE html', analysis_sample, re.IGNORECASE) or \
                        re.search(r'<html[^>]*>', analysis_sample, re.IGNORECASE) or \
                        re.search(r'<head[^>]*>', analysis_sample, re.IGNORECASE) or \
                        re.search(r'<body[^>]*>', analysis_sample, re.IGNORECASE) or \
                        re.search(r'<div[^>]*>', analysis_sample, re.IGNORECASE):
                           log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: HTML", __file__)
                           return TYPE_HTML

                # Code Check (Refine based on Textual/Unknown)
                if classified_type == TYPE_CODE or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     # Heuristics: keywords, comments, syntax elements
                     code_keywords = {'def ', 'class ', 'import ', 'function ', 'public ', 'private ', 'static ', 'void ', 'int ', 'float ', 'const ', 'let '}
                     code_chars = {'{', '}', ';', '#', '//', '/*'}
                     keyword_hits = sum(1 for keyword in code_keywords if keyword in analysis_sample)
                     char_hits = sum(1 for char in code_chars if char in analysis_sample)
                     # Simple thresholding (adjust as needed)
                     if keyword_hits >= 2 or char_hits >= 3 or (keyword_hits >= 1 and char_hits >= 1):
                          # Stronger check: Look for typical line patterns (indentation, assignments)
                          assignments = re.search(r'\w+\s*=\s*', analysis_sample)
                          indentation = any(line.startswith((' ', '\t')) for line in sample_lines if line.strip())
                          if assignments or indentation or keyword_hits >=3:
                              log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: CODE (Keywords: {keyword_hits}, Chars: {char_hits}, Assign/Indent: {bool(assignments or indentation)})", __file__)
                              return TYPE_CODE

                # Markdown Check
                if classified_type == TYPE_MARKDOWN or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                    # Look for markdown syntax elements (more than a few occurrences)
                    md_patterns = [
                        r'^\s*#+\s+',  # Headers
                        r'^\s*[\*\-\+]\s+',  # List items
                        r'^\s*>',  # Blockquotes
                        r'```',  # Code fences
                        r'\*\*|__|\*|_', # Bold/Italics (can be noisy)
                        r'\[.+?\]\(.+?\)' # Links
                    ]
                    md_hits = sum(1 for pattern in md_patterns if re.search(pattern, analysis_sample, re.MULTILINE))
                    if md_hits >= 2: # Require at least 2 different MD features
                         log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: MARKDOWN (Hits: {md_hits})", __file__)
                         return TYPE_MARKDOWN

                # Tabular Check (CSV/TSV)
                if classified_type == TYPE_TABULAR or classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     # Use CSV Sniffer for more robust detection
                     try:
                         # Sniffing requires a sample string
                         sample_for_sniffing = "\n".join(sample_lines[:20]) # Use first 20 lines
                         if len(sample_for_sniffing) > 10240: # Limit sample size for sniffer
                              sample_for_sniffing = sample_for_sniffing[:10240]

                         if sample_for_sniffing:
                              dialect = csv.Sniffer().sniff(sample_for_sniffing, delimiters=',\t;|') # Common delimiters
                              # Check if a delimiter was actually found and seems consistent
                              if dialect.delimiter:
                                   # Further check: does the delimiter appear consistently across lines?
                                   delimiter_counts = [line.count(dialect.delimiter) for line in sample_lines[:10] if line.strip()]
                                   if len(delimiter_counts) > 1 and len(set(delimiter_counts)) <= len(delimiter_counts) / 2: # Heuristic: allow some variation
                                        log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis match: TABULAR (Delimiter: '{dialect.delimiter}')", __file__)
                                        return TYPE_TABULAR
                                   else:
                                        log_statement('debug', f"{LOG_INS}:DEBUG>>Sniffer found delimiter '{dialect.delimiter}' but usage seems inconsistent.", __file__)
                         else:
                              log_statement('debug', f"{LOG_INS}:DEBUG>>Sample too short or empty for CSV sniffing.", __file__)

                     except csv.Error as sniff_err:
                          log_statement('debug', f"{LOG_INS}:DEBUG>>CSV Sniffer could not detect dialect: {sniff_err}", __file__)
                     except Exception as csv_err:
                          log_statement('warning', f"{LOG_INS}:WARNING>>Error during CSV/Tabular check: {csv_err}", __file__, exc_info=True)


                # Final check for Textual if not matched above
                if classified_type == TYPE_TEXTUAL or classified_type == TYPE_UNKNOWN:
                     # If it survived all other checks, it's likely general text
                     log_statement('debug', f"{LOG_INS}:DEBUG>>Content analysis: Classified as general TEXTUAL", __file__)
                     return TYPE_TEXTUAL

                # If extension was misleading and content analysis didn't match, fall back?
                if classified_type != TYPE_UNKNOWN and classified_type not in [TYPE_TEXTUAL, TYPE_TABULAR, TYPE_CODE]:
                     log_statement('warning', f"{LOG_INS}:WARNING>>Content analysis did not confirm extension-based type '{classified_type}' for {filepath}. Reverting to TEXTUAL or UNKNOWN.", __file__)
                     # Decide: trust extension or mark as unknown/textual? Let's lean towards TEXTUAL if it's decodable.
                     return TYPE_TEXTUAL


            # --- Fallback: Binary or Unknown ---
            if isinstance(content, bytes) and not content_str: # If it's bytes and couldn't be decoded
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Content is bytes and could not be decoded or matched by magic number. Classifying as BINARY.", __file__)
                 return TYPE_BINARY
            elif isinstance(content_str, str):
                 # Optional: Check for high percentage of non-printable chars if needed
                 printable_ratio = sum(c.isprintable() or c.isspace() for c in analysis_sample) / len(analysis_sample)
                 if printable_ratio < 0.8: # Example threshold
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Low printable character ratio ({printable_ratio:.2f}). Classifying as BINARY.", __file__)
                    return TYPE_BINARY
                 pass # Already classified as Textual or other type above

            # If we reach here, no classification was successful
            log_statement('warning', f"{LOG_INS}:WARNING>>Classification failed for {filepath}. Returning TYPE_UNKNOWN.", __file__)
            return TYPE_UNKNOWN

        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error during data classification for {filepath}: {e}", __file__, exc_info=True)
            return TYPE_UNKNOWN # Return UNKNOWN on error

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
    def _get_file_modification_time(self, filepath: Path) -> dt:
        """Returns the last modification time of the file."""
        return dt.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
    def _get_file_creation_time(self, filepath: Path) -> dt:
        """Returns the creation time of the file."""
        return dt.fromtimestamp(filepath.stat().st_ctime, tz=timezone.utc)
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
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Starting repository scan using base directory: {BASE_DATA_DIR}", main_logger=__file__)
        self.repo.scan_and_update(BASE_DATA_DIR)
        
    def __del__(self):
        """Ensures proper cleanup of the executor on object deletion."""
        if hasattr(self, 'executor') and self.executor:
            try:
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::Shutting down DataProcessor executor...", main_logger=__file__) 
                self.executor.shutdown(wait=True)
                # Wait for tasks on shutdown
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::DataProcessor executor shut down.", main_logger=__file__)
            except Exception as e: 
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error shutting down executor: {e}", main_logger=__file__)

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
        main_logger_name = __file__
        log_statement(loglevel="info", logstatement=f"Attempting semantic labeling for {file_path_hint.name} using Ollama model '{model_name}'...", main_logger=main_logger_name)

        # --- Pre-check for Ollama Library ---
        try:
            import ollama
        except ImportError:
            log_statement(loglevel="error", logstatement=f"{LOG_INS}::Ollama library not installed. Cannot perform semantic labeling. Run 'pip install ollama'.", main_logger=main_logger_name)
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

    def _save_processed_repository(self, processed_df: pd.DataFrame):
        """
        Saves the DataFrame containing successfully processed file information
        to the dedicated processed repository file.

        Args:
            processed_df (pd.DataFrame): DataFrame containing entries with STATUS_PROCESSED.
        """
        save_logger_name = __file__
        output_path = self.processed_repo_filepath # Use the dedicated path attribute

        if processed_df is None:
            log_statement(loglevel="error", logstatement=f"Cannot save processed repository, provided DataFrame is None.", main_logger=save_logger_name)
            return False
        if output_path is None:
            log_statement(loglevel="error", logstatement=f"Cannot save processed repository, output path is not set.", main_logger=save_logger_name)
            return False

        # Use the header defined for processed repos
        expected_columns = PROCESSED_REPO_COLUMNS # Use the correct constant header
        df_copy = processed_df.copy()

        if df_copy.empty:
            log_statement(loglevel="info", logstatement=f"{LOG_INS}::Processed DataFrame is empty. Saving empty processed repo file to: {output_path}", main_logger=save_logger_name)
        else:
             log_statement(loglevel="info", logstatement=f"{LOG_INS}::Saving {len(df_copy)} processed entries to: {output_path}", main_logger=save_logger_name)

        # --- Ensure Schema Alignment (similar to DataRepository.save) ---
        current_cols = df_copy.columns.tolist()
        added_cols_save = []
        for col_const in expected_columns:
            if col_const not in current_cols:
                added_cols_save.append(col_const)
                target_dtype = self.columns_schema.get(col_const, str) # Use schema from DataProcessor if needed, or assume main repo schema for now
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Save Processed Repo - Adding missing column '{col_const}' with type {target_dtype}", main_logger=save_logger_name)
                # Add column with appropriate null type
                if 'datetime' in str(target_dtype): df_copy[col_const] = pd.NaT
                elif target_dtype == 'Int64': df_copy[col_const] = pd.NA
                else: df_copy[col_const] = ''
        if added_cols_save: log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Columns added during processed repo save: {added_cols_save}", main_logger=save_logger_name)

        try: # Ensure correct column order
            df_copy = df_copy[expected_columns]
        except KeyError as ke:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Save Processed Repo Error: Column mismatch after adding defaults. Missing: {ke}. Expected: {expected_columns}. Available: {df_copy.columns.tolist()}", main_logger=save_logger_name)
            return False # Cannot proceed if columns mismatch

        # --- Type Prep and Saving Logic (Simplified from DataRepository.save) ---
        temp_path = output_path.with_suffix(f'{output_path.suffix}.tmp_proc_{int(time.time())}')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
            with open(temp_path, 'wb') as f_out:
                with cctx.stream_writer(f_out) as writer:
                    # Convert DF to CSV in memory, handling types appropriately for CSV
                    # NOTE: Reusing the simple string conversion from DataRepository.save for now
                    df_stringified = df_copy.copy()
                    for col_const in expected_columns:
                        if col_const in df_stringified.columns:
                             series = df_stringified[col_const]
                             if col_const in self.timestamp_columns: # Check if it's a timestamp column
                                 # Convert float timestamp back to ISO string or keep as numeric string
                                 df_stringified[col_const] = series.apply(lambda x: '' if pd.isna(x) else f"{x:.6f}") # Save as float string
                             elif self.columns_schema.get(col_const) == 'Int64':
                                 df_stringified[col_const] = series.apply(lambda x: '' if pd.isna(x) else str(int(x))) # Explicit int string
                             else: # Default to string for others
                                 df_stringified[col_const] = series.fillna('').astype(str)

                    csv_buffer = io.StringIO()
                    df_stringified.to_csv(csv_buffer, index=False, header=True, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
                    csv_buffer.seek(0)
                    csv_data = csv_buffer.getvalue()
                    csv_buffer.close()
                    writer.write(csv_data.encode('utf-8'))

            shutil.move(str(temp_path), str(output_path))
            log_statement(loglevel="info", logstatement=f"{LOG_INS}::Successfully saved processed repository ({len(df_copy)} entries) to {output_path}", main_logger=save_logger_name)
            return True
        except Exception as e:
            log_statement(loglevel="error", logstatement=f"{LOG_INS}::CRITICAL ERROR saving processed repository DataFrame to {output_path}: {e}", main_logger=save_logger_name, exc_info=True)
            if temp_path.exists():
                try: os.remove(temp_path)
                except OSError: pass
            return False
        
    def process_all(self, base_dir_filter: Optional[Path] = None, statuses_to_process=('discovered', 'error', STATUS_NEW)):
        """Processes files matching status, optionally filtered by base_dir, with progress bar."""
        main_logger_name = str(__file__)
        print(f"{LOG_INS}::DataProcessor.process_all START - Filter: {base_dir_filter}, Statuses: {statuses_to_process}")
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Starting - Filter: {base_dir_filter}, Statuses: {statuses_to_process}", main_logger=main_logger_name)

        # --- Get files to process (existing logic) ---
        print(f"{LOG_INS}::PROCESS_ALL - Getting files by status...")
        files_to_process_paths = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
        print(f"{LOG_INS}::PROCESS_ALL - Found {len(files_to_process_paths)} files matching status.")
        if not files_to_process_paths:
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] found to process.", main_logger=main_logger_name)
            print(f"{LOG_INS}::DataProcessor.process_all END - No files found.")
            return

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Found {len(files_to_process_paths)} files matching status. Retrieving full info...", main_logger=main_logger_name)
        print(f"{LOG_INS}::PROCESS_ALL - Retrieving full file info...")
        file_info_list = []
        try:
            with self.repo.lock:
                files_to_process_str_set = {str(p.resolve()) for p in files_to_process_paths}
                if COL_FILEPATH not in self.repo.df.columns:
                    print(f"{LOG_INS}::PROCESS_ALL - CRITICAL ERROR - Column '{COL_FILEPATH}' not found.")
                    log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: Critical error - Column '{COL_FILEPATH}' not found in repository DataFrame.", main_logger=main_logger_name)
                    return

                matching_rows_df = self.repo.df[self.repo.df[COL_FILEPATH].isin(files_to_process_str_set)].copy()
                if not matching_rows_df.empty:
                    file_info_list = matching_rows_df.to_dict('records')
                    print(f"{LOG_INS}::PROCESS_ALL - Retrieved {len(file_info_list)} records.")
                    log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Retrieved {len(file_info_list)} full records for processing.", main_logger=main_logger_name)
                else:
                    print(f"{LOG_INS}::PROCESS_ALL - WARNING: Found paths but failed to retrieve rows.")
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_ALL: Found matching paths but failed to retrieve corresponding rows from repo DataFrame.", main_logger=main_logger_name)
        except Exception as e:
             print(f"{LOG_INS}::PROCESS_ALL - ERROR retrieving file info: {e}")
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: Failed to retrieve full file info for processing: {e}", main_logger=main_logger_name, exc_info=True)
             return
        if not file_info_list:
             print(f"{LOG_INS}::PROCESS_ALL - WARNING: Could not retrieve details for files.")
             log_statement(loglevel="warning", logstatement=f"{LOG_INS}::PROCESS_ALL: Could not retrieve details for files matching status {statuses_to_process}. Cannot process.", main_logger=main_logger_name)
             return

        # --- Submit processing tasks (existing logic) ---
        if not hasattr(self, 'executor') or self.executor is None:
             print(f"{LOG_INS}::PROCESS_ALL - ERROR: Executor not initialized.")
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: DataProcessor executor not initialized. Cannot process.", main_logger=__file__)
             return

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Submitting {len(file_info_list)} processing tasks to executor.", main_logger=main_logger_name)
        print(f"{LOG_INS}::PROCESS_ALL - Submitting {len(file_info_list)} tasks...")
        futures = [self.executor.submit(self._process_file, f_info) for f_info in file_info_list]

        # # --- Process results (existing logic) ---
        # pbar_desc = f"Processing [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]"
        # results_processed = 0
        # results_error = 0
        # successfully_processed_files_info = [] # Collect info for the processed repo

        # print(f"{LOG_INS}::PROCESS_ALL - Waiting for futures...")
        # for future in tqdm(as_completed(futures), total=len(futures), desc=pbar_desc, unit="file", leave=True):
        #     try:
        #         result_info = future.result() # result_info is the updated dict from _process_file
        #         file_path_hint = result_info.get(COL_FILEPATH, "Unknown Path") if isinstance(result_info, dict) else "Invalid Result"
        #         print(f"{LOG_INS}::PROCESS_ALL - Future completed for ~{Path(file_path_hint).name}. Result type: {type(result_info)}")
        #         if result_info and isinstance(result_info, dict):
        #             final_status = result_info.get(COL_STATUS, STATUS_ERROR)
        #             print(f"{LOG_INS}::PROCESS_ALL - Future Result Status: {final_status} for {Path(file_path_hint).name}")
        #             if final_status == STATUS_PROCESSED:
        #                 results_processed += 1
        #                 successfully_processed_files_info.append(result_info) # <<< Add successful info
        #                 log_statement(loglevel='debug', logstatement=f"PROCESS_ALL: Future completed successfully for: {Path(file_path_hint).name} (Status: {final_status})", main_logger=main_logger_name)
        #             else:
        #                 results_error += 1
        #                 error_msg = result_info.get(COL_ERROR, "Unknown error")
        #                 print(f"{LOG_INS}::PROCESS_ALL - Future completed with Error/Other Status '{final_status}' for {Path(file_path_hint).name}. Error: {error_msg}")
        #                 log_statement(loglevel='warning', logstatement=f"PROCESS_ALL: Future completed with status '{final_status}' for: {Path(file_path_hint).name}. Error: {error_msg}", main_logger=main_logger_name)
        #         else:
        #             results_error += 1
        #             print(f"{LOG_INS}::PROCESS_ALL - Future returned None or invalid result.")
        #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_ALL: Processing future returned None or invalid result (potential internal error in _process_file).", main_logger=main_logger_name)
        #     except Exception as e:
        #         results_error += 1
        #         # Log error from the future result itself more clearly
        #         print(f"{LOG_INS}::PROCESS_ALL - EXCEPTION retrieving future result: {e}")
        #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: Error retrieving result from processing future: {e}", main_logger=main_logger_name, exc_info=True)

        # # --- Save Main Repository (already handled by _process_file calling update_entry) ---
        # self.repo.save()
        log_statement(loglevel='info', logstatement=f"PROCESS_ALL: Submitting {len(file_info_list)} processing tasks to executor.", main_logger=main_logger_name)
        futures = [self.executor.submit(self._process_file, f_info) for f_info in file_info_list]

        # --- Process results ---
        pbar_desc = f"Processing [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]"
        results_processed_count = 0
        results_error_count = 0
        successfully_processed_files_info = [] # <<< Collect successful results here

        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_ALL: Waiting for futures...", main_logger=main_logger_name)
        for future in tqdm(as_completed(futures), total=len(futures), desc=pbar_desc, unit="file", leave=True):
            try:
                result_info = future.result() # result_info is the updated dict from _process_file
                file_path_hint = result_info.get(COL_FILEPATH, "Unknown Path") if isinstance(result_info, dict) else "Invalid Result"

                if result_info and isinstance(result_info, dict):
                    final_status = result_info.get(COL_STATUS, STATUS_ERROR)
                    if final_status == STATUS_PROCESSED:
                        results_processed_count += 1
                        successfully_processed_files_info.append(result_info) # <<< Append successful info
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_ALL: Future completed successfully for: {Path(file_path_hint).name} (Status: {final_status})", main_logger=main_logger_name)
                    else:
                        results_error_count += 1
                        error_msg = result_info.get(COL_ERROR, "Unknown error")
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_ALL: Future completed with status '{final_status}' for: {Path(file_path_hint).name}. Error: {error_msg}", main_logger=main_logger_name)
                else:
                    results_error_count += 1
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_ALL: Processing future returned None or invalid result (potential internal error in _process_file).", main_logger=main_logger_name)
            except Exception as e:
                results_error_count += 1
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: Error retrieving result from processing future: {e}", main_logger=main_logger_name, exc_info=True)

        # --- Save Main Repository (reflects status updates like 'processing' -> 'processed' or 'error') ---
        # The update_entry calls within _process_file handle marking success/failure status
        # Save the main repo once after all processing attempts are done
        log_statement(loglevel="info", logstatement=f"{LOG_INS}::PROCESS_ALL: Saving updated main repository after processing attempts...", main_logger=main_logger_name)
        self.repo.save()

        # # --- *** NEW: Save Dedicated Processed Repository *** ---
        # if successfully_processed_files_info:
        #      log_statement(loglevel='info', logstatement=f"PROCESS_ALL: Creating DataFrame for {len(successfully_processed_files_info)} successfully processed files.", main_logger=main_logger_name)
        #      processed_df = pd.DataFrame(successfully_processed_files_info)
        #      # Ensure the DataFrame uses the correct schema before saving
        #      processed_df = processed_df.reindex(columns=PROCESSED_REPO_COLUMNS) # Use correct columns
        #      log_statement(loglevel='info', logstatement=f"PROCESS_ALL: Attempting to save dedicated processed repository file: {self.processed_repo_filepath}", main_logger=main_logger_name)
        #      save_success = self._save_processed_repository(processed_df)
        #      if save_success:
        #          log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Successfully saved dedicated processed repository.", main_logger=main_logger_name)
        #      else:
        #          log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_ALL: Failed to save dedicated processed repository file.", main_logger=main_logger_name)
        # else:
        #      log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: No files were successfully processed in this run. Skipping processed repository save.", main_logger=main_logger_name)
        #      # Optionally, create an empty processed repo file if none exists
        #      if not self.processed_repo_filepath.exists():
        #           log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Creating empty processed repository file.", main_logger=main_logger_name)
        #           self._save_processed_repository(pd.DataFrame(columns=PROCESSED_REPO_COLUMNS))

        # # --- Final Log ---
        # print(f"{LOG_INS}::DataProcessor.process_all END - Success: {results_processed}, Errors: {results_error}")
        # log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_ALL: Stage complete. Successful: {results_processed}, Errors/Skipped: {results_error}.", main_logger=main_logger_name)
        # --- Save Main Repository (reflects status updates like 'processing' -> 'processed' or 'error') ---
        # The update_entry calls within _process_file handle marking success/failure status
        # Save the main repo once after all processing attempts are done

        log_statement(loglevel="info", logstatement=f"{LOG_INS}::Saving updated main repository after processing attempts...", main_logger=main_logger_name)
        self.repo.save()

        # --- REMOVE Internal Processed Repo Saving Logic ---
        # The logic previously here (_save_processed_repository call) is removed.
        # Saving the processed list is now handled in m1.py.

        # --- Final Log ---
        log_statement(loglevel="info", logstatement=f"{LOG_INS}::PROCESS_ALL: Stage complete. Successful: {results_processed_count}, Errors/Skipped: {results_error_count}.", main_logger=main_logger_name)

        # --- RETURN the list of successfully processed file info ---
        return successfully_processed_files_info # <<< RETURN RESULTS

    # def _process_file(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    #     """
    #     Refactored: Processes a single file based on metadata from the repository.
    #     Includes extensive debugging logs and corrected helper function calls.
    #     Returns updated metadata dict on success or failure, None on critical setup error.
    #     """
    #     # --- Setup & Input Logging ---
    #     main_logger_name = __file__ # Or define logger name appropriately
    #     LOG_INS = f"{__name__}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::{inspect.currentframe().f_lineno}" # Dynamic LOG_INS
    #     print(f"{LOG_INS}:DEBUG>>** Method START **")
    #     print(f"{LOG_INS}:DEBUG>>Input file_info: {file_info}")

    #     absolute_file_path_str = file_info.get(COL_FILEPATH)
    #     file_name_hint = Path(absolute_file_path_str).name if absolute_file_path_str else "Unknown"
    #     designation_hint = f" (Desig: {file_info.get(COL_DESIGNATION, 'N/A')})"
    #     # Use standard logging for general flow
    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE START for: {file_name_hint}{designation_hint}", main_logger=main_logger_name)

    #     updated_info = file_info.copy()
    #     updated_info[COL_STATUS] = STATUS_PROCESSING # Mark as attempting
    #     updated_info[COL_ERROR] = ''
    #     print(f"{LOG_INS}:DEBUG>>Initial updated_info (Status set to PROCESSING): {updated_info}")
    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE - Initial status set to '{STATUS_PROCESSING}' for {file_name_hint}", main_logger=main_logger_name)

    #     if not absolute_file_path_str:
    #         print(f"{LOG_INS}:DEBUG>>** ERROR: Missing file path in input file_info. **")
    #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_FILE: Missing file path for Designation: {file_info.get(COL_DESIGNATION, 'N/A')}", main_logger=main_logger_name)
    #         updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = "Missing file path"
    #         print(f"{LOG_INS}:DEBUG>>Attempting repo update for missing path...")
    #         # Construct a unique placeholder if path is truly missing
    #         placeholder_path = f"MISSING_PATH_{designation_hint.strip(' ()Desig:/NA')}_{int(time.time())}"
    #         try:
    #             # Update repo with error status using placeholder if necessary
    #             self.repo.update_entry(Path(absolute_file_path_str) if absolute_file_path_str else Path(placeholder_path), **updated_info)
    #             print(f"{LOG_INS}:DEBUG>>Repo update call successful for missing path.")
    #         except Exception as missing_path_update_err:
    #             print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo for missing path: {missing_path_update_err} **")
    #             log_statement(loglevel='critical', logstatement=f"{LOG_INS}::PROCESS_FILE: CRITICAL - Failed repository update for missing path entry ({placeholder_path}): {missing_path_update_err}", main_logger=main_logger_name, exc_info=True)
    #         print(f"{LOG_INS}:DEBUG>>Returning updated_info for missing path: {updated_info}")
    #         return updated_info

    #     # Define outside try block for use in final update
    #     absolute_file_path = Path(absolute_file_path_str)

    #     # --- Step 1: Path Setup ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 1: Path Setup START **")
    #     output_path_base = None
    #     # --- Step 1: Setup Output Paths ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 1: Path Setup START **")
    #     output_path_base: Optional[Path] = None # Initialize for clarity

    #     try:
    #         # Ensure we have an absolute, resolved path for the input file
    #         # The input absolute_file_path should already be resolved by the caller (e.g., repo_handler)
    #         # but resolving again is safe and ensures consistency.
    #         absolute_file_path = Path(absolute_file_path).resolve()
    #         print(f"{LOG_INS}:DEBUG>>Resolved absolute_file_path: {absolute_file_path}")

    #         # Get the parent directory of the input file
    #         input_parent_dir = absolute_file_path.parent
    #         print(f"{LOG_INS}:DEBUG>>Input parent directory: {input_parent_dir}")

    #         # Create the relative directory structure by removing the drive/anchor
    #         # Example: /mnt/aPrime/TestDir/Docs -> mnt/aPrime/TestDir/Docs
    #         # Example: C:\Users\Test\Doc -> Users\Test\Doc
    #         relative_dir_structure = Path(*input_parent_dir.parts[1:])
    #         print(f"{LOG_INS}:DEBUG>>Calculated relative directory structure: {relative_dir_structure}")

    #         # Combine the processor's base output directory with the relative structure
    #         # self.output_dir should be set during DataProcessor initialization
    #         # (e.g., Path(PROCESSED_DATA_DIR) or Path(TOKENIZED_DATA_DIR))
    #         target_output_dir = self.output_dir / relative_dir_structure
    #         print(f"{LOG_INS}:DEBUG>>Calculated target output directory: {target_output_dir}")

    #         # Construct the base path for the output file within the target directory
    #         # This includes the original filename stem, ready for adding suffixes later
    #         output_path_base = target_output_dir / absolute_file_path.name
    #         print(f"{LOG_INS}:DEBUG>>Calculated output_path_base (full path prefix): {output_path_base}")

    #         # Ensure the target directory exists, creating parent directories as needed
    #         target_output_dir.mkdir(parents=True, exist_ok=True)
    #         print(f"{LOG_INS}:DEBUG>>Ensured target output directory exists: {target_output_dir}")

    #         log_statement(loglevel='debug',
    #                       logstatement=f"{LOG_INS}::PROCESS_FILE Paths OK for {file_name_hint}. Output target dir: {target_output_dir}",
    #                       main_logger=main_logger_name)
    #         print(f"{LOG_INS}:DEBUG>>** Step 1: Path Setup END (Success) **")

    #     except Exception as path_err:
    #         # Handle any unexpected errors during path manipulation or directory creation
    #         print(f"{LOG_INS}:DEBUG>>** Step 1: Path Setup FAILED: {path_err} **")
    #         log_statement(loglevel='error',
    #                       logstatement=f"{LOG_INS}::PROCESS_FILE: CRITICAL error setting up paths for {file_name_hint}: {path_err}",
    #                       main_logger=main_logger_name,
    #                       exc_info=True)

    #         # Update repo entry with error status
    #         updated_info[COL_STATUS] = STATUS_ERROR
    #         updated_info[COL_ERROR] = f"Path setup error: {path_err}"
    #         print(f"{LOG_INS}:DEBUG>>Attempting repo update after path setup error...")
    #         try:
    #             # Use the original absolute_file_path as the key for updating the repo
    #             self.repo.update_entry(absolute_file_path, **updated_info)
    #             print(f"{LOG_INS}:DEBUG>>Repo update call successful after path error.")
    #         except Exception as update_err:
    #             print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after path setup error: {update_err} **")
    #             # Log this critical failure as well
    #             log_statement(loglevel='critical',
    #                           logstatement=f"{LOG_INS}::PROCESS_FILE: Failed to update repo status for {absolute_file_path} after path error: {update_err}",
    #                           main_logger=main_logger_name,
    #                           exc_info=True)

    #         print(f"{LOG_INS}:DEBUG>>Returning updated_info after path setup error: {updated_info}")
    #         return updated_info # Return error info dictionary

    #     # --- Step 2: Reader Instantiation ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 2: Reader Instantiation START **")
    #     reader = None
    #     ReaderClass = None
    #     try:
    #         file_suffix = absolute_file_path.suffix
    #         print(f"{LOG_INS}:DEBUG>>Getting reader class for suffix: '{file_suffix}'")
    #         ReaderClass = get_reader_class(file_suffix) # Pass suffix directly

    #         if ReaderClass:
    #             print(f"{LOG_INS}:DEBUG>>Found ReaderClass: {ReaderClass.__name__}")
    #             # Check if it's a text reader needing encoding args
    #             if issubclass(ReaderClass, RobustTextReader):
    #                 print(f"{LOG_INS}:DEBUG>>Instantiating RobustTextReader with encoding args...")
    #                 reader = ReaderClass(
    #                     filepath=absolute_file_path,
    #                     default_encoding='utf-8',
    #                     error_handling='replace',
    #                     detect_encoding=True # Explicitly enable detection attempt
    #                 )
    #                 print(f"{LOG_INS}:DEBUG>>RobustTextReader instantiated: {reader}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>Instantiating non-text reader...")
    #                 # Instantiate non-text readers normally (like PDF, Excel)
    #                 reader = ReaderClass(filepath=absolute_file_path)
    #                 print(f"{LOG_INS}:DEBUG>>Non-text reader instantiated: {reader}")
    #             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Instantiated reader: {ReaderClass.__name__} for {file_name_hint}", main_logger=main_logger_name)
    #             print(f"{LOG_INS}:DEBUG>>** Step 2: Reader Instantiation END (Success) **")

    #         else: # ReaderClass is None (Unsupported type)
    #             print(f"{LOG_INS}:DEBUG>>** Step 2: Reader Instantiation FAILED (Unsupported Type) **")
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_FILE: No reader for {file_name_hint}. Skipping.", main_logger=main_logger_name)
    #             updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = f"Unsupported file type: {file_suffix}"
    #             print(f"{LOG_INS}:DEBUG>>Attempting repo update after unsupported type...")
    #             try:
    #                 self.repo.update_entry(absolute_file_path, **updated_info)
    #                 print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #             except Exception as update_err:
    #                 print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after unsupported type: {update_err} **")
    #             print(f"{LOG_INS}:DEBUG>>Returning updated_info after unsupported type: {updated_info}")
    #             return updated_info

    #     except Exception as reader_init_err:
    #         print(f"{LOG_INS}:DEBUG>>** Step 2: Reader Instantiation FAILED (Exception): {reader_init_err} **")
    #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_FILE: Reader init failed for {file_name_hint}: {reader_init_err}", main_logger=main_logger_name, exc_info=True)
    #         updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = f"Reader init failed: {reader_init_err}"
    #         print(f"{LOG_INS}:DEBUG>>Attempting repo update after reader init error...")
    #         try:
    #             self.repo.update_entry(absolute_file_path, **updated_info)
    #             print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #         except Exception as update_err:
    #             print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after reader init error: {update_err} **")
    #         print(f"{LOG_INS}:DEBUG>>Returning updated_info after reader init error: {updated_info}")
    #         return updated_info

    #     # --- Step 3: Read Data ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 3: Read Data START **")
    #     read_data = None
    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Attempting read for {file_name_hint} using {ReaderClass.__name__}", main_logger=main_logger_name)
    #     try:
    #         print(f"{LOG_INS}:DEBUG>>Calling reader.read() for {file_name_hint}...")
    #         read_data = reader.read()
    #         print(f"{LOG_INS}:DEBUG>>reader.read() returned type: {type(read_data)}")

    #         if read_data is None:
    #             print(f"{LOG_INS}:DEBUG>>** Step 3: Read Data FAILED (Reader returned None) **")
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_FILE: Reader returned None for {file_name_hint}.", main_logger=main_logger_name)
    #             updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = "Reader returned None"
    #             print(f"{LOG_INS}:DEBUG>>Attempting repo update after read returned None...")
    #             try:
    #                 self.repo.update_entry(absolute_file_path, **updated_info)
    #                 print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #             except Exception as update_err:
    #                 print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after read returned None: {update_err} **")
    #             print(f"{LOG_INS}:DEBUG>>Returning updated_info after read returned None: {updated_info}")
    #             return updated_info
    #         elif isinstance(read_data, pd.DataFrame) and read_data.empty:
    #             log_statement(loglevel='info', logstatement=f"{LOG_INS}::PROCESS_FILE Read OK (Empty DataFrame) for {file_name_hint}.", main_logger=main_logger_name)
    #             print(f"{LOG_INS}:DEBUG>>Read data is Empty DataFrame. Shape: {read_data.shape}")
    #             # Proceed to classification, which should handle empty DF
    #             print(f"{LOG_INS}:DEBUG>>** Step 3: Read Data END (Success - Empty DataFrame) **")
    #         else:
    #             data_preview = f"type: {type(read_data)}"
    #             if isinstance(read_data, str): data_preview = str(read_data)[:100].replace('\n', '\\n') + "..."
    #             elif isinstance(read_data, pd.DataFrame): data_preview += f", shape: {read_data.shape}"
    #             print(f"{LOG_INS}:DEBUG>>Read data preview: {data_preview}")
    #             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Read OK for {file_name_hint}. Data {data_preview}", main_logger=main_logger_name)
    #             print(f"{LOG_INS}:DEBUG>>** Step 3: Read Data END (Success) **")
    #     except Exception as read_err:
    #         print(f"{LOG_INS}:DEBUG>>** Step 3: Read Data FAILED (Exception): {read_err} **")
    #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_FILE: Reader failed for {file_name_hint}: {read_err}", main_logger=main_logger_name, exc_info=True)
    #         updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = f"Reader failed: {read_err}"
    #         print(f"{LOG_INS}:DEBUG>>Attempting repo update after read error...")
    #         try:
    #             self.repo.update_entry(absolute_file_path, **updated_info)
    #             print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #         except Exception as update_err:
    #             print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after read error: {update_err} **")
    #         print(f"{LOG_INS}:DEBUG>>Returning updated_info after read error: {updated_info}")
    #         return updated_info

    #     # --- Step 4: Classify Data ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 4: Classify Data START **")
    #     data_classification = TYPE_UNKNOWN
    #     print(f"{LOG_INS}:DEBUG>>Initial data_classification: {data_classification}")
    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Attempting classification for {file_name_hint}", main_logger=main_logger_name)
    #     try:
    #         if not hasattr(self, '_classify_data'):
    #             print(f"{LOG_INS}:DEBUG>>** Step 4: Classify Data FAILED (AttributeError: _classify_data missing) **")
    #             raise AttributeError("_classify_data method missing from DataProcessor")

    #         print(f"{LOG_INS}:DEBUG>>Input data type for classification: {type(read_data)}")
    #         if isinstance(read_data, pd.DataFrame):
    #             print(f"{LOG_INS}:DEBUG>>Calling self._classify_data for DataFrame...")
    #             data_classification = self._classify_data(read_data, absolute_file_path)
    #             print(f"{LOG_INS}:DEBUG>>Classification result: {data_classification}")
    #         elif isinstance(read_data, str):
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Reader returned raw string for {file_name_hint}. Expected DataFrame. Classifying as TEXTUAL.", main_logger=main_logger_name)
    #             print(f"{LOG_INS}:DEBUG>>Reader returned string. Classifying as TEXTUAL. Converting to DataFrame.")
    #             data_classification = TYPE_TEXTUAL
    #             read_data = pd.DataFrame({'text': [read_data]}) # Convert string to DataFrame
    #             print(f"{LOG_INS}:DEBUG>>Converted read_data to DataFrame. Shape: {read_data.shape}")
    #         else:
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Unsupported data type from reader for classification: {type(read_data)}. Classifying as UNKNOWN.", main_logger=main_logger_name)
    #             print(f"{LOG_INS}:DEBUG>>Unsupported data type ({type(read_data)}). Classifying as UNKNOWN.")
    #             data_classification = TYPE_UNKNOWN

    #         updated_info[COL_DATA_CLASSIFICATION] = data_classification # Store classification result (Using Constant)
    #         print(f"{LOG_INS}:DEBUG>>Stored data classification '{data_classification}' in updated_info.")
    #         log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Classified {file_name_hint} as {data_classification}", main_logger=main_logger_name)

    #         # Handle EMPTY classification immediately
    #         if data_classification == TYPE_EMPTY or (isinstance(read_data, pd.DataFrame) and read_data.empty):
    #             print(f"{LOG_INS}:DEBUG>>Data classified as EMPTY or DataFrame is empty. Skipping further processing.")
    #             log_statement(loglevel="warning", logstatement=f"{LOG_INS}::PROCESS_FILE: Data classified as EMPTY for {file_name_hint}. Skipping processing.", main_logger=main_logger_name)
    #             updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = "Empty or unclassifiable content"
    #             print(f"{LOG_INS}:DEBUG>>Attempting repo update after EMPTY classification...")
    #             try:
    #                 self.repo.update_entry(absolute_file_path, **updated_info)
    #                 print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #             except Exception as update_err:
    #                 print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after EMPTY classification: {update_err} **")
    #             print(f"{LOG_INS}:DEBUG>>Returning updated_info after EMPTY classification: {updated_info}")
    #             return updated_info
    #         print(f"{LOG_INS}:DEBUG>>** Step 4: Classify Data END (Success) **")
    #     except Exception as classify_err:
    #         print(f"{LOG_INS}:DEBUG>>** Step 4: Classify Data FAILED (Exception): {classify_err} **")
    #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_FILE: Classification failed for {file_name_hint}: {classify_err}", main_logger=main_logger_name, exc_info=True)
    #         updated_info[COL_STATUS] = STATUS_ERROR; updated_info[COL_ERROR] = f"Classification failed: {classify_err}"
    #         # Store the attempted classification even if it failed
    #         updated_info[COL_DATA_CLASSIFICATION] = data_classification if data_classification != TYPE_UNKNOWN else "CLASSIFICATION_FAILED"
    #         print(f"{LOG_INS}:DEBUG>>Attempting repo update after classification error...")
    #         try:
    #             self.repo.update_entry(absolute_file_path, **updated_info)
    #             print(f"{LOG_INS}:DEBUG>>Repo update call successful.")
    #         except Exception as update_err:
    #             print(f"{LOG_INS}:DEBUG>>** CRITICAL ERROR: Failed to update repo after classification error: {update_err} **")
    #         print(f"{LOG_INS}:DEBUG>>Returning updated_info after classification error: {updated_info}")
    #         return updated_info

    #     # --- Step 5: Process Data Based on Classification ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 5: Process Data START (Classification: {data_classification}) **")
    #     result = None
    #     processed_flag = False
    #     actual_output_path = None # Define before try block
    #     specific_process_method_name = "None" # Track which method is called

    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Entering processing block for {file_name_hint} (Type: {data_classification})", main_logger=main_logger_name)
    #     try:
    #         if data_classification == TYPE_TEXTUAL:
    #             actual_output_path = output_path_base.with_suffix('.json.zst')
    #             specific_process_method_name = '_process_textual_data'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for {file_name_hint} -> {actual_output_path}")
    #             if hasattr(self, specific_process_method_name):
    #                 result = self._process_textual_data(read_data, absolute_file_path, actual_output_path)
    #                 processed_flag = True
    #                 print(f"{LOG_INS}:DEBUG>>{specific_process_method_name} returned: {type(result)}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>** ERROR: Method {specific_process_method_name} not found! **")
    #                 raise AttributeError(f"{specific_process_method_name} method not found.")

    #         elif data_classification == TYPE_NUMERICAL:
    #             actual_output_path = output_path_base.with_suffix('.parquet.zst')
    #             specific_process_method_name = '_process_numerical_data'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for {file_name_hint} -> {actual_output_path}")
    #             if hasattr(self, specific_process_method_name):
    #                 result = self._process_numerical_data(read_data, absolute_file_path, actual_output_path)
    #                 processed_flag = True
    #                 print(f"{LOG_INS}:DEBUG>>{specific_process_method_name} returned: {type(result)}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>** ERROR: Method {specific_process_method_name} not found! **")
    #                 raise AttributeError(f"{specific_process_method_name} method not found.")

    #         elif data_classification in [TYPE_PDF, TYPE_DOC, TYPE_DOCX]: # Assume these were read into text DataFrame
    #             actual_output_path = output_path_base.with_suffix('.json.zst')
    #             specific_process_method_name = '_process_textual_data (from doc)'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for {file_name_hint} -> {actual_output_path}")
    #             if hasattr(self, '_process_textual_data'):
    #                 result = self._process_textual_data(read_data, absolute_file_path, actual_output_path)
    #                 processed_flag = True
    #                 print(f"{LOG_INS}:DEBUG>>{specific_process_method_name} returned: {type(result)}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>** ERROR: Method _process_textual_data (fallback) not found! **")
    #                 raise AttributeError(f"Fallback _process_textual_data missing.")

    #         elif data_classification == TYPE_EXCEL: # Assume read into numerical DataFrame
    #             actual_output_path = output_path_base.with_suffix('.parquet.zst')
    #             specific_process_method_name = '_process_numerical_data (from excel)'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for {file_name_hint} -> {actual_output_path}")
    #             if hasattr(self, '_process_numerical_data'):
    #                 result = self._process_numerical_data(read_data, absolute_file_path, actual_output_path)
    #                 processed_flag = True
    #                 print(f"{LOG_INS}:DEBUG>>{specific_process_method_name} returned: {type(result)}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>** ERROR: Method _process_numerical_data not found! **")
    #                 raise AttributeError(f"_process_numerical_data method not found.")

    #         elif data_classification in [TYPE_TOKENIZED_SUBWORD, TYPE_TOKENIZED_NUMERICAL, TYPE_TOKENIZED_JSONL, TYPE_TOKENIZED_CSV]: # Add other token types
    #             actual_output_path = output_path_base.with_suffix('.parquet.zst')
    #             specific_process_method_name = 'save_dataframe_to_parquet_zst (passthrough)'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for pre-structured {file_name_hint} -> {actual_output_path}")
    #             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Passthrough save for pre-structured {file_name_hint} -> {actual_output_path}", main_logger=main_logger_name)
    #             # --- FIX: Assume save_dataframe_to_parquet_zst is imported directly ---
    #             try:
    #                 # Check if the function exists in the current scope (requires direct import)
    #                 if 'save_dataframe_to_parquet_zst' in globals() or 'save_dataframe_to_parquet_zst' in locals():
    #                     save_success = save_dataframe_to_parquet_zst(read_data, actual_output_path)
    #                     print(f"{LOG_INS}:DEBUG>>save_dataframe_to_parquet_zst returned: {save_success}")
    #                     if save_success:
    #                         print(f"{LOG_INS}:DEBUG>>Generating hash for saved file: {actual_output_path}")
    #                         # --- FIX: Assume generate_data_hash is imported directly ---
    #                         output_hash = generate_data_hash(actual_output_path)
    #                         print(f"{LOG_INS}:DEBUG>>Generated hash: {output_hash}")
    #                         if output_hash is None:
    #                             print(f"{LOG_INS}:DEBUG>>** WARNING: Hash generation failed for passthrough save. **")
    #                             updated_info[COL_ERROR] = f"Hash generation failed after saving pre-structured data"; result = None
    #                         else:
    #                             relative_output_path = actual_output_path.relative_to(self.output_dir)
    #                             result = {COL_PROCESSED_PATH: str(relative_output_path), COL_DATA_HASH: output_hash or ""}
    #                             print(f"{LOG_INS}:DEBUG>>Passthrough save result: {result}")
    #                     else:
    #                         updated_info[COL_ERROR] = f"Failed to save pre-structured data"; result = None
    #                 else:
    #                     print(f"{LOG_INS}:DEBUG>>** ERROR: Helper function save_dataframe_to_parquet_zst not found! **")
    #                     raise NameError(f"Helper function save_dataframe_to_parquet_zst not found.")
    #             except NameError as ne: # Catch if the function isn't imported
    #                 raise AttributeError(f"Helper function {ne} not found/imported.") from ne
    #             processed_flag = True

    #         elif data_classification == TYPE_UNKNOWN:
    #             actual_output_path = output_path_base.with_suffix('.json.zst')
    #             specific_process_method_name = '_process_textual_data (fallback)'
    #             print(f"{LOG_INS}:DEBUG>>Calling {specific_process_method_name} for {file_name_hint} -> {actual_output_path}")
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_FILE Unknown classification, fallback to text processing for {file_name_hint} -> {actual_output_path}", main_logger=main_logger_name)
    #             if hasattr(self, '_process_textual_data'):
    #                 result = self._process_textual_data(read_data, absolute_file_path, actual_output_path)
    #                 processed_flag = True
    #                 print(f"{LOG_INS}:DEBUG>>{specific_process_method_name} returned: {type(result)}")
    #             else:
    #                 print(f"{LOG_INS}:DEBUG>>** ERROR: Method _process_textual_data (fallback) not found! **")
    #                 raise AttributeError("Fallback _process_textual_data missing.")
    #         else:
    #             # Handle other specific types like IMAGE, AUDIO, BINARY etc. if needed
    #             # For now, treat as unprocessable
    #             print(f"{LOG_INS}:DEBUG>>No specific processing logic for classification '{data_classification}'. Skipping.")
    #             log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_FILE No specific processing logic for classification '{data_classification}' ({file_name_hint}). Skipping processing step.", main_logger=main_logger_name)
    #             updated_info[COL_ERROR] = f"No processing logic for '{data_classification}'"
    #             result = None # Indicate no successful processing occurred
    #             processed_flag = True # Mark as processed attempt failed

    #         print(f"{LOG_INS}:DEBUG>>** Step 5: Process Data END (Method: {specific_process_method_name}) **")
    #         print(f"{LOG_INS}:DEBUG>>Processing result type: {type(result)}, Processed flag: {processed_flag}")
    #         if isinstance(result, dict):
    #             print(f"{LOG_INS}:DEBUG>>Processing result dict: {result}")
    #         log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Method '{specific_process_method_name or 'N/A'}' finished for {file_name_hint}. Result: {'Success (dict)' if isinstance(result, dict) else 'Fail/None'}", main_logger=main_logger_name)

    #     except Exception as process_err:
    #         print(f"{LOG_INS}:DEBUG>>** Step 5: Process Data FAILED (Exception during {specific_process_method_name}): {process_err} **")
    #         log_statement(loglevel='error', logstatement=f"{LOG_INS}::PROCESS_FILE: Error during '{data_classification}' processing for {file_name_hint}: {process_err}", main_logger=main_logger_name, exc_info=True)
    #         updated_info[COL_ERROR] = f"Processing error ({data_classification}): {process_err}"
    #         result = None; processed_flag = True # Mark as processed even if exception occurred

    #     # --- Step 6: Check Result and Finalize Status ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 6: Finalize Status START **")
    #     # Check if processing was attempted, if result is a dict, and if processed_path is present and non-empty
    #     if processed_flag and isinstance(result, dict) and result.get(COL_PROCESSED_PATH):
    #         print(f"{LOG_INS}:DEBUG>>Processing SUCCEEDED. Updating status to PROCESSED.")
    #         log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Processing SUCCEEDED for {file_name_hint}. Setting status to PROCESSED.", main_logger=main_logger_name)
    #         updated_info.update(result) # Add processed_path and output_hash
    #         updated_info[COL_STATUS] = STATUS_PROCESSED
    #         updated_info[COL_ERROR] = '' # Clear previous error if reprocessing succeeded
    #         updated_info[COL_FINAL_CLASSIFICATION] = data_classification # Store final classification (Using Constant)
    #         print(f"{LOG_INS}:DEBUG>>updated_info after success: {updated_info}")
    #     else:
    #         print(f"{LOG_INS}:DEBUG>>Processing FAILED or no valid result returned. Setting status to ERROR.")
    #         log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PROCESS_FILE: Processing FAILED or no result/path for {file_name_hint} (Classification: {data_classification}). Processed flag: {processed_flag}", main_logger=main_logger_name)
    #         # Keep existing error if processing failed, otherwise set new one
    #         if not updated_info.get(COL_ERROR): # Only set error if not already set by exception
    #             updated_info[COL_ERROR] = f"Processing failed or no result/path for classification '{data_classification}'"
    #             print(f"{LOG_INS}:DEBUG>>Setting error message: {updated_info[COL_ERROR]}")
    #         else:
    #             print(f"{LOG_INS}:DEBUG>>Keeping existing error message: {updated_info.get(COL_ERROR)}")
    #         # Store final classification even on failure
    #         updated_info[COL_FINAL_CLASSIFICATION] = data_classification # Using Constant
    #         updated_info[COL_STATUS] = STATUS_ERROR
    #         print(f"{LOG_INS}:DEBUG>>updated_info after failure: {updated_info}")
    #     print(f"{LOG_INS}:DEBUG>>** Step 6: Finalize Status END **")

    #     # --- Step 7: Update Repository ---
    #     print(f"{LOG_INS}:DEBUG>>** Step 7: Update Repository START **")
    #     try:
    #         print(f"{LOG_INS}:DEBUG>>Calling update_entry for {file_name_hint}. Final Status: '{updated_info[COL_STATUS]}'.")
    #         print(f"{LOG_INS}:DEBUG>>Data being sent to update_entry: {updated_info}")
    #         log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE Calling update_entry for {file_name_hint}. Final Status: '{updated_info[COL_STATUS]}'. Update Keys: {list(updated_info.keys())}", main_logger=main_logger_name)
    #         # Use self.repo which should be the DataRepository instance for the MAIN repo
    #         self.repo.update_entry(absolute_file_path, **updated_info)
    #         print(f"{LOG_INS}:DEBUG>>update_entry call returned for {file_name_hint}.")
    #         log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE update_entry call returned for {file_name_hint}.", main_logger=main_logger_name)
    #         print(f"{LOG_INS}:DEBUG>>** Step 7: Update Repository END (Success) **")
    #     except Exception as repo_update_err:
    #         print(f"{LOG_INS}:DEBUG>>** Step 7: Update Repository FAILED (Exception): {repo_update_err} **")
    #         log_statement(loglevel='critical', logstatement=f"{LOG_INS}::PROCESS_FILE: CRITICAL - Failed repository update for {file_name_hint}: {repo_update_err}", main_logger=main_logger_name, exc_info=True)
    #         # Update failed, but we still need to return the info gathered so far
    #         updated_info[COL_ERROR] = f"Final Status: {updated_info.get(COL_STATUS, 'Unknown')}; Repo Update Err: {repo_update_err}"
    #         updated_info[COL_STATUS] = STATUS_ERROR # Mark as error if update failed
    #         print(f"{LOG_INS}:DEBUG>>updated_info after repo update failure: {updated_info}")

    #     # --- Return Final Result ---
    #     print(f"{LOG_INS}:DEBUG>>** Method END **")
    #     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::PROCESS_FILE END for: {file_name_hint}. Returning status: {updated_info.get(COL_STATUS)}", main_logger=main_logger_name)
    #     print(f"{LOG_INS}:DEBUG>>Final updated_info being returned: {updated_info}")
    #     return updated_info

    # Make sure filepath is Path object if type hints are used
    def _process_file(self, filepath: Path, use_gpu: bool):
        """Processes a single file: classify, clean/transform, save, update repo."""
        log_statement('debug', f"{LOG_INS}:DEBUG>>Starting processing for file: {filepath}", __file__)
        # Get file hash - Assuming it's needed for naming or already in repo
        # This might require reading the file or getting it from repo entry if available
        file_hash = "temp_hash" # Placeholder: Need reliable way to get hash if needed for filename
        # Better: Get hash from repository row if already scanned
        try:
             file_info = self.repository.get_file_info(str(filepath)) # Assumes method exists
             if file_info is not None and COL_HASH in file_info:
                 file_hash = file_info[COL_HASH]
             else:
                 log_statement('warning', f"{LOG_INS}:WARNING>>Could not retrieve hash for {filepath} from repo. Using placeholder.", __file__)
                 # Optionally calculate hash here if needed: from src.utils.hashing import generate_data_hash
        except Exception as e:
            log_statement('warning', f"{LOG_INS}:WARNING>>Error getting file hash from repo for {filepath}: {e}. Using placeholder.", __file__, exc_info=True)


        processed_content = None
        processed_path = None
        error_info = None
        final_status = None
        data_type = TYPE_UNKNOWN # Default
        processed_filename_str = None # Initialize

        # Use the string representation of the path for repository updates, matching storage format
        filepath_str = str(filepath)

        try:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Updating status to PROCESSING for {filepath_str}", __file__)
            self.repository.update_entry(filepath_str, {COL_STATUS: STATUS_PROCESSING})

            # --- Read Content ---
            # This assumes an internal method _read_file_content exists, otherwise use readers
            content = self._read_file_content(filepath)
            if content is None:
                raise ValueError("Failed to read file content.")
            log_statement('debug', f"{LOG_INS}:DEBUG>>Read content successfully for {filepath_str}", __file__)

            # --- Classify ---
            data_type = self._classify_data(filepath, content)
            log_statement('debug', f"{LOG_INS}:DEBUG>>Classified {filepath_str} as {data_type}", __file__)

            # --- Process based on type ---
            if data_type == TYPE_TEXTUAL:
                processed_content = self._process_textual_data(content)
                log_statement('debug', f"{LOG_INS}:DEBUG>>Processed textual data for {filepath_str}", __file__)
            elif data_type == TYPE_NUMERICAL:
                log_statement('warning', f"{LOG_INS}:WARNING>>Numerical processing not fully implemented for {filepath_str}. Saving raw.", __file__)
                processed_content = content # Placeholder
            # ... handle other types ...
            else:
                log_statement('warning', f"{LOG_INS}:WARNING>>Skipping processing for unsupported type {data_type}: {filepath_str}", __file__)
                final_status = STATUS_SKIPPED
                processed_content = None # Ensure we don't try to save if skipping

            # --- Save Processed Data (if content exists) ---
            if processed_content is not None:
                # Use reliable hash if available, otherwise stem might suffice if unique enough
                safe_hash_part = file_hash[:8] if file_hash and len(file_hash) >= 8 else filepath.stem[:8]
                processed_filename_str = f"{filepath.stem}_{safe_hash_part}_processed.txt.zst" # Example naming
                processed_path = self.output_directory / processed_filename_str
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                log_statement('debug', f"{LOG_INS}:DEBUG>>Attempting to save processed data to {processed_path}", __file__)

                try:
                    # Assuming saving compressed text
                    compress_string_to_file(str(processed_content), processed_path) # Ensure content is string
                    log_statement('info', f"{LOG_INS}:INFO>>Saved processed data to {processed_path}", __file__)
                    final_status = STATUS_PROCESSED
                except Exception as save_err:
                     log_statement('error', f"{LOG_INS}:ERROR>>Failed to save processed data for {filepath_str} to {processed_path}: {save_err}", __file__, exc_info=True)
                     final_status = STATUS_FAILED
                     error_info = f"Failed to save processed file: {save_err}"
                     processed_path = None # Ensure path isn't stored if saving failed
                     processed_filename_str = None # Clear filename too

            elif final_status != STATUS_SKIPPED:
                 # If no content was processed AND it wasn't explicitly skipped
                 log_statement('warning', f"{LOG_INS}:WARNING>>Processing resulted in empty content for {filepath_str}", __file__)
                 final_status = STATUS_FAILED
                 error_info = "Processing resulted in empty content."


        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error processing file {filepath_str}: {e}", __file__, exc_info=True)
            final_status = STATUS_FAILED
            error_info = str(e)
            data_type = data_type if data_type != TYPE_UNKNOWN else None # Keep classified type if available

        finally:
            # --- Update Repository Entry ---
            update_data = {
                COL_STATUS: final_status,
                COL_ERROR_INFO: error_info,
                COL_DATA_TYPE: data_type,
                COL_PROCESSED_FILENAME: processed_filename_str, # Use the determined filename string
            }
            # Remove keys with None values to avoid overwriting existing repo data unnecessarily
            update_data = {k: v for k, v in update_data.items() if v is not None}

            if not update_data:
                 log_statement('debug', f"{LOG_INS}:DEBUG>>No data to update in repository for {filepath_str}.", __file__)
            else:
                 try:
                     log_statement('debug', f"{LOG_INS}:DEBUG>>Updating repository for {filepath_str} with data: {update_data}", __file__)
                     self.repository.update_entry(filepath_str, update_data) # Use string path
                     log_statement('debug', f"{LOG_INS}:DEBUG>>Repository updated for {filepath_str} with status: {final_status}", __file__)
                 except Exception as repo_err:
                      # Use logger formatting for critical errors
                      log_statement('critical', f"{LOG_INS}:CRITICAL>>Failed to update repository for {filepath_str} after processing: {repo_err}", __file__, exc_info=True)

    # --- Placeholder/Assumed internal methods ---
    def _read_file_content(self, filepath: Path) -> str | bytes | None:
         # Implementation depends on how reading is handled (e.g., using readers.py)
         log_statement('debug', f"{LOG_INS}:DEBUG>>_read_file_content called for {filepath}", __file__)
         try:
              reader_cls = get_reader_class(filepath)
              if reader_cls:
                   reader = reader_cls(filepath)
                   return reader.read()
              else: # Basic fallback
                   with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
         except Exception as e:
              log_statement('error', f"{LOG_INS}:ERROR>>_read_file_content failed for {filepath}: {e}", __file__, exc_info=True)
              return None

    def _classify_data(self, filepath: Path, content: str | bytes) -> str:
         # Implementation for classifying data type
         log_statement('debug', f"{LOG_INS}:DEBUG>>_classify_data called for {filepath}", __file__)
         # Add logic here, e.g., check extension, content patterns
         if isinstance(content, str): # Basic check
              return TYPE_TEXTUAL
         return TYPE_UNKNOWN # Default


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
        main_logger_name = __file__
        log_statement(loglevel="debug", logstatement=f"{LOG_INS}::Starting _process_textual_data for: {input_path.name}", main_logger=main_logger_name)
        try:
            # --- Consolidate Text from DataFrame ---
            # (Combine relevant text columns into a single string)
            text_content = ""
            # Example: combine all string columns, separated by newlines
            # Ensure only object/string dtypes are selected
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            print(f"{LOG_INS}:DEBUG::string_cols value {string_cols}")
            if not string_cols.empty:
                print(f"{LOG_INS}::string_cols not empty!")
                for col in string_cols:
                    # Ensure proper handling of potential NaN/None before astype(str) and str.cat
                    text_content += df[col].fillna('').astype(str).str.cat(sep='\n') + "\n"
                    print(f"{LOG_INS}:DEBUG>>Column value is '{col}'"
                          f"{LOG_INS}:DEBUG>>text_content value is '{text_content}'")
                print(f"{LOG_INS}:DEBUG>>Stripping text content for '{text_content}'...")
                text_content = text_content.strip()
                print(f"{LOG_INS}:DEBUG>>Stripped content:"
                      f"{LOG_INS}:DEBUG>>{text_content}")
            else:
                # Handle case where DataFrame has no string columns
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::No string columns found in DataFrame for {input_path.name} to consolidate text.", main_logger=main_logger_name)
                # Attempt to convert all columns to string as a fallback?
                # Or return None? Let's try converting all.
                for col in df.columns:
                    text_content += df[col].fillna('').astype(str).str.cat(sep='\n') + "\n"
                    print(f"{LOG_INS}:DEBUG>>Column value is '{col}'"
                          f"{LOG_INS}:DEBUG>>text_content value is '{text_content}'")
                text_content = text_content.strip()
                print(f"{LOG_INS}:DEBUG>>Stripped content:"
                      f"{LOG_INS}:DEBUG>>{text_content}")

            if not text_content:
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::No text content extracted from DataFrame for {input_path.name}", main_logger=main_logger_name)
                return None

            # --- Basic Cleaning (Apply before labeling) ---
            cleaned_text = text_content.lower()
            print(f"{LOG_INS}:DEBUG>>Cleaned version of '{text_content}':"
                  f"{LOG_INS}:DEBUG>>{cleaned_text}"
                  f"{LOG_INS}:DEBUG>>Checking for 'cleaning_regex' attribute existence...")
            if hasattr(self, 'cleaning_regex') and self.cleaning_regex: # Check if regex exists
                # Ensure cleaning_regex is compiled, ideally in __init__
                print(f"{LOG_INS}:DEBUG>>Existence of cleaning_regex confirmed!  Checking to see if instance is a string to compile...")
                if isinstance(self.cleaning_regex, str): # Compile if it's still a string
                    print(f"{LOG_INS}:DEBUG>>cleaning_regex instance still a string!  Attempting to compile...")
                    try:
                        self.cleaning_regex = re.compile(self.cleaning_regex)
                        print(f"{LOG_INS}:DEBUG>>Successfully compiled string for cleaning_regex as '{self.cleaning_regex}' using the 're' python module!")
                    except re.error as re_err:
                        log_statement(loglevel='error',
                                     logstatement=f"{LOG_INS}::Invalid cleaning regex: {re_err}"
                                                  f"{LOG_INS}:DEBUG>>cleaning_regex setting to None...",
                                     main_logger=main_logger_name)
                        self.cleaning_regex = None # Disable if invalid

                if hasattr(self.cleaning_regex, 'sub'): # Check if valid compiled regex
                    cleaned_text = self.cleaning_regex.sub('', cleaned_text)

                # General whitespace normalization
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            # Optional: NLTK processing could go here too

            # --- <<< Semantic Labeling >>> ---
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
                    log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Semantic labeling enabled but _label_text_semantically method not found in DataProcessor.", main_logger=main_logger_name)


            # --- Prepare Output Data ---
            if structured_data:
                # Output is the structured dictionary from Ollama
                output_data_to_save = structured_data
            else:
                # Fallback: Output the cleaned text in a simple structure
                # (or just the raw cleaned text if preferred, adjust saving method)
                log_statement(loglevel="debug", logstatement=f"{LOG_INS}::No structured data from labeling for {input_path.name}, saving cleaned text.", main_logger=main_logger_name)
                # Adhere to Rule #11 (tabular/JSON output)
                # Simple JSON structure containing the cleaned text
                output_data_to_save = {"cleaned_text": cleaned_text} # Or split into lines: cleaned_text.split('\n')

            # --- Save processed data (structured or fallback) as compressed JSON ---
            try:
                json_string = json.dumps(output_data_to_save, indent=2) # Pretty print JSON
            except TypeError as te:
                log_statement(loglevel="error", logstatement=f"{LOG_INS}::Failed to serialize processed data to JSON for {input_path.name}: {te}", main_logger=main_logger_name)
                return None

            # Ensure output path has the correct extension (.json.zst)
            output_path = output_path.with_suffix('.json.zst')
            save_success = compress_string_to_file(json_string, output_path)

            if not save_success:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Save failed!", main_logger=__file__)
                return None # Error logged within helper

            # --- Calculate hash & Return metadata (as before) ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate hash for output file: {output_path}", main_logger=__file__)
                 return None

            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed in _process_textual_data for {input_path}: {e}", main_logger=__file__, exc_info=True)
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
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Starting _process_numerical for: {input_path.name}", main_logger=__file__)
        try:
            # Use the reader's read method - assumes it returns a DataFrame
            dataframe = reader.read() # reader initialized with input_path

            if dataframe is None or dataframe.empty:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Reader returned empty or None DataFrame for {input_path.name}. Skipping.", main_logger=__file__)
                return None

            # --- Numerical Processing Logic ---
            # Convert to numeric types, handle errors
            numeric_df = dataframe.apply(pd.to_numeric, errors='coerce')
            # Drop columns that are entirely NaN after conversion attempt
            numeric_df = numeric_df.dropna(axis=1, how='all')

            if numeric_df.empty:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::No numeric data found after conversion in {input_path.name}. Skipping.", main_logger=__file__)
                return None

            # Scaling (using the scaler initialized in DataProcessor __init__)
            processed_df = numeric_df
            if hasattr(self, 'scaler') and self.scaler:
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Applying scaler to data from {input_path.name}", main_logger=__file__)
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
                    log_statement(loglevel='error', logstatement=f"{LOG_INS}::Scaling failed for {input_path.name}: {scale_err}. Proceeding with unscaled data.", main_logger=__file__, exc_info=True)
                    processed_df = numeric_df # Use original numeric data if scaling fails
            else:
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::No scaler available or configured. Skipping scaling for {input_path.name}", main_logger=__file__)


            # --- Save processed DataFrame to compressed Parquet ---
            # Use a helper function or implement logic here
            if not hasattr(self, 'save_dataframe_to_parquet_zst'):
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::save_dataframe_to_parquet_zst helper method not found.", main_logger=__file__)
                raise NotImplementedError("save_dataframe_to_parquet_zst helper method not found.")

            save_success = self.save_dataframe_to_parquet_zst(processed_df, output_path)

            if not save_success:
                # Error logged within helper
                return None # Indicate failure

            # --- Calculate hash of the output file ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate hash for output file: {output_path}", main_logger=__file__)
                return None # Indicate failure

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed in _process_numerical for {input_path}: {e}", main_logger=__file__, exc_info=True)
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
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Starting _process_pdf for: {input_path.name}", main_logger=__file__)
        try:
            # PDFReader.read() returns a DataFrame with a 'text' column
            dataframe = reader.read()

            if dataframe is None or dataframe.empty or 'text' not in dataframe.columns or dataframe['text'].iloc[0] is None:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::PDFReader returned no text content for {input_path.name}. Skipping.", main_logger=__file__)
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
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::No text content resulted after processing PDF {input_path.name}. Skipping save.", main_logger=__file__)
                return None

            # --- Save processed text to output_path (compressed) ---
            if not hasattr(self, 'compress_string_to_file'):
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::compress_string_to_file helper method not found.", main_logger=__file__)
                raise NotImplementedError("compress_string_to_file helper method not found.")

            save_success = compress_string_to_file(processed_text, output_path)

            if not save_success:
                return None # Error logged within helper

            # --- Calculate hash of the output file ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate hash for output file: {output_path}", main_logger=__file__)
                return None

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except ImportError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::PDF processing failed for {input_path}: Missing required library (e.g., pdfminer.six).", main_logger=__file__, exc_info=True)
            return None # Indicate failure due to missing dependency
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed in _process_pdf for {input_path}: {e}", main_logger=__file__, exc_info=True)
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
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Starting _process_docx for: {input_path.name}", main_logger=__file__)
        try:
            # Assume reader.read() returns a DataFrame {'text': [content]}
            dataframe = reader.read()

            if dataframe is None or dataframe.empty or 'text' not in dataframe.columns or dataframe['text'].iloc[0] is None:
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::DocxReader returned no text content for {input_path.name}. Skipping.", main_logger=__file__)
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
                log_statement(loglevel='warning', logstatement=f"{LOG_INS}::No text content resulted after processing DOCX {input_path.name}. Skipping save.", main_logger=__file__)
                return None

            # --- Save processed text to output_path (compressed) ---
            if not hasattr(self, 'compress_string_to_file'):
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::compress_string_to_file helper method not found.", main_logger=__file__)
                raise NotImplementedError("compress_string_to_file helper method not found.")

            save_success = compress_string_to_file(processed_text, output_path)
            if not save_success: return None

            # --- Calculate hash ---
            output_hash = generate_data_hash(output_path)
            if output_hash is None:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate hash for output file: {output_path}", main_logger=__file__)
                return None

            # --- Return metadata ---
            relative_output_path = output_path.relative_to(self.output_dir)
            return {
                "processed_path": str(relative_output_path),
                COL_DATA_HASH: output_hash
            }

        except ImportError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::DOCX processing failed for {input_path}: Missing required library (e.g., python-docx).", main_logger=__file__, exc_info=True)
            return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed in _process_docx for {input_path}: {e}", main_logger=__file__, exc_info=True)
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
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Delegating Excel processing for {input_path.name} to _process_numerical.", main_logger=__file__)
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
        log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Audio processing (_process_audio) not implemented yet for: {input_path.name}", main_logger=__file__)
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
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing ODT file: {filepath.name}", main_logger=__file__)
        try:
            import odfpy
            from odf.opendocument import OpenDocumentText
            doc = OpenDocumentText(filepath)
            text = []
            for elem in doc.getElementsByType(odfpy.text.P):
                text.append(elem.firstChild.data)
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::ODT processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ODT processing failed: {e}")
            return None
        
    def _process_doc(self, filepath: Path):
        """Processes DOC files using python-docx."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing DOC file: {filepath.name}", main_logger=__file__)
        try:
            import docx
            doc = docx.Document(filepath)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::DOC processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"DOC processing failed: {e}")
            return None
        
    def _process_html_xml(self, filepath: Path):
        """Processes HTML/XML files using BeautifulSoup."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing HTML/XML file: {filepath.name}", main_logger=__file__)
        try:
            import bs4
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            soup = bs4.BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::HTML/XML processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"HTML/XML processing failed: {e}")
            return None
        
    def _process_rtf(self, filepath: Path):
        """Processes RTF files using striprtf."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing RTF file: {filepath.name}", main_logger=__file__)
        try:
            import striprtf
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return striprtf.strip(content)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::RTF processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"RTF processing failed: {e}")
            return None
        
    def _process_epub(self, filepath: Path):
        """Processes EPUB files using EbookLib."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing EPUB file: {filepath.name}", main_logger=__file__)
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(filepath)
            text = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                text.append(item.get_body_content_str())
            return "\n".join(text)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::EPUB processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"EPUB processing failed: {e}")
            return None
        
    def _process_zip(self, filepath: Path):
        """Processes ZIP files using zipfile."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processing ZIP file: {filepath.name}", main_logger=__file__)
        try:
            import zipfile
            with zipfile.ZipFile(filepath, 'r') as z:
                text = []
                for file in z.namelist():
                    with z.open(file) as f:
                        text.append(f.read().decode('utf-8'))
                return "\n".join(text)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::ZIP processing failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ZIP processing failed: {e}")
            return None

    def _read_content(self, filepath: Path):
        """Reads the content of a file, handling different encodings."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Reading content from {filepath.name}", main_logger=__file__)
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
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to read file {filepath}: {e}", main_logger=__file__, exc_info=True)
                self.repo.update_entry(filepath, status='error', error_message=f"File read failed: {e}")
                return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to read content from {filepath}: {e}", main_logger=__file__, exc_info=True)
            raise # Re-raise to be caught by _process_file
        except FileNotFoundError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::File not found during read: {filepath}", main_logger=__file__, exc_info=True)
            return None
        except IsADirectoryError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Expected file but found directory: {filepath}", main_logger=__file__, exc_info=True)
            return None
        except zstd.ZstdError as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Zstandard decompression failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            return None
        except UnicodeDecodeError as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unicode decoding failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error reading {filepath}: {e}", main_logger=__file__, exc_info=True)
            return None

    @staticmethod
    def _ensure_repo_exists(filepath: Path, header: List[str]):
        """Creates the repository file with a header if it doesn't exist."""
        if not filepath.exists():
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Repository file not found at '{filepath}'. Initializing...", main_logger=__file__)
            try:
                # Write header to a new compressed file
                def header_gen():
                    yield ','.join(header) # CSV header line

                stream_compress_lines(str(filepath), header_gen())
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::Initialized repository file: {filepath}", main_logger=__file__)
            except Exception as e:
                log_statement(loglevel='critical', logstatement=f"{LOG_INS}::Failed to initialize repository file '{filepath}': {e}", main_logger=__file__, exc_info=True)
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
                     log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Skipping row {i+1}: Invalid designation number in row: {row}", main_logger=__file__)
                     continue # Skip rows with invalid numbers
            return max_designation + 1
        except FileNotFoundError:
            log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Repository file '{self.repo_filepath}' not found while getting next designation. Starting from 1.", main_logger=__file__)
            return 1
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error reading repository to find next designation: {e}", main_logger=__file__, exc_info=True)
            # Fallback or re-raise depending on desired robustness
            log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Defaulting next designation to 1 due to error.", main_logger=__file__)
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
                        log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Invalid designation '{designation}' found for hash '{data_hash}'", main_logger=__file__)
            return hashes
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error loading existing hashes from repository: {e}", main_logger=__file__, exc_info=True)
            return {} # Return empty on error

    def read_repo_stream(self, filepath: Optional[Path] = None) -> Generator[Dict[str, str], None, None]:
        """
        Reads the repository CSV file line by line using streaming decompression.
        Yields each row as a dictionary. Handles potential errors during reading.
        """
        target_filepath = filepath or self.repo_filepath
        if not target_filepath.exists():
            log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Attempted to read non-existent repository: {target_filepath}", main_logger=__file__)
            return # Yield nothing

        header = []
        try:
            line_generator = stream_decompress_lines(str(target_filepath))
            # Read header first
            try:
                header_line = next(line_generator)
                header = [h.strip() for h in header_line.split(',')]
            except StopIteration:
                 log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Repository file is empty: {target_filepath}", main_logger=__file__)
                 return # Empty file

            # Use csv.DictReader on the remaining lines
            # We need to simulate a file-like object for DictReader
            reader = csv.DictReader(line_generator, fieldnames=header, restval=None) # Use header as fieldnames
                                                                                       # restval handles rows with too few fields
            for row in reader:
                 if len(row) != len(header):
                     log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Malformed row in {target_filepath} (expected {len(header)} fields, got {len(row)}): {row}", main_logger=__file__)
                     # Optionally yield a partial dict or skip
                     # yield row # Yields what was parsed
                     continue # Skip malformed row
                 yield row

        except FileNotFoundError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Repository file not found during streaming read: {target_filepath}", main_logger=__file__)
            # Or re-raise depending on desired behavior
        except zstd.ZstdError as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Zstd decompression error reading {target_filepath}: {e}", main_logger=__file__, exc_info=True)
        except csv.Error as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::CSV parsing error reading {target_filepath}: {e}", main_logger=__file__, exc_info=True)
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error reading repository stream {target_filepath}: {e}", main_logger=__file__, exc_info=True)

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
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Appended rows to repository: {filepath}", main_logger=__file__)

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to append rows to repository '{filepath}': {e}", main_logger=__file__, exc_info=True)
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
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Folder not found or is not a directory: {abs_folder_path}", main_logger=__file__)
                return
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error resolving folder path '{folder_path}': {e}", main_logger=__file__, exc_info=True)
            return

        # Use custom log statement
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Scanning folder: {abs_folder_path}", main_logger=__file__)
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
            log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Loaded {len(existing_paths)} existing file paths from repository.", main_logger=__file__)
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to load existing paths from repository: {e}", main_logger=__file__, exc_info=True)
            log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Proceeding without check for existing paths. Duplicates might be added.", main_logger=__file__)
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
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Skipping already tracked file path: {file_path_str}", main_logger=__file__)
                        skipped_count += 1
                        continue

                    # Check for supported file types
                    if file_ext not in ACCEPTED_FILE_TYPES:
                        # Use custom log statement
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Skipping unsupported file type '{file_ext}': {file_path_str}", main_logger=__file__)
                        skipped_count += 1
                        continue

                    # --- Handle Original Compression State & Archives ---
                    # TODO: Implement robust archive handling (extraction) if needed.
                    if file_ext in ['.zip', '.zst', '.zstd']:
                        # Use custom log statement
                        log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Archive file found: {file_path_str}. Treating as single compressed file. Extraction logic not implemented.", main_logger=__file__)
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
                        mod_time_iso = dt.fromtimestamp(mod_time_ts, tz=timezone.utc).isoformat()
                        acc_time_iso = dt.fromtimestamp(acc_time_ts, tz=timezone.utc).isoformat()

                        # Generate hashes
                        # Ensure hash functions handle errors and return empty string or None on failure
                        hashed_path = hash_filepath(file_path_str)
                        data_hash = generate_data_hash(file_path_str) # Content hash

                        current_status = STATUS_LOADED # Default status
                        is_copy = 'N' # Default copy status

                        # Check hash generation results
                        if not hashed_path:
                            # Use custom log statement
                            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate required filepath hash for: {file_path_str}. Marking as Error.", main_logger=__file__)
                            current_status = STATUS_ERROR
                            error_count += 1
                            # Skip adding? Or add with Error status? Let's add with Error status.
                            # data_hash = "" # Ensure data_hash is empty if path hash failed? Or proceed?

                        if not data_hash and current_status != STATUS_ERROR:
                            # Use custom log statement
                            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to generate required data hash for: {file_path_str}. Marking as Error.", main_logger=__file__)
                            current_status = STATUS_ERROR
                            error_count += 1
                            # No valid data hash means we cannot check for copies effectively

                        # Check for content copy using data hash (only if hash was successful)
                        if data_hash and data_hash in self._data_hashes:
                            is_copy = 'Y'
                            original_designation = self._data_hashes[data_hash]
                            # Use custom log statement
                            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Detected content copy (Data Hash: {data_hash[:8]}...) for: {file_path_str} - Matches Designation: {original_designation}", main_logger=__file__)
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
                        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Prepared file for repository addition (Status: {current_status}): {file_path_str}", main_logger=__file__)

                    except OSError as e:
                        # Use custom log statement
                        log_statement(loglevel='error', logstatement=f"{LOG_INS}::OS Error accessing metadata/hashes for {file_path_str}: {e}", main_logger=__file__)
                        error_count += 1
                        # Optionally add an entry with STATUS_ERROR here? For now, we skip.
                    except Exception as e:
                        # Use custom log statement (including exc_info for unexpected errors)
                        log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error processing file {file_path_str}: {e}", main_logger=__file__, exc_info=True)
                        error_count += 1
                        # Optionally add an entry with STATUS_ERROR here? For now, we skip.

                    # Log progress periodically
                    if processed_count % 500 == 0:
                        # Use custom log statement
                        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Scanned {processed_count} files so far...", main_logger=__file__)

        except PermissionError as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Permission error scanning folder '{abs_folder_path}': {e}. Check read permissions.", main_logger=__file__)
            error_count += 1 # Count this as an error for the summary
        except Exception as e:
            # Use custom log statement
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected error during folder scan of '{abs_folder_path}': {e}", main_logger=__file__, exc_info=True)
            error_count += 1 # Count this as an error


        # --- Append all new files found in this run ---
        if new_files_to_add:
            # Use custom log statement
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Scan found {len(new_files_to_add)} new file entries to add to the repository.", main_logger=__file__)
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
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::Successfully appended {len(new_files_to_add)} entries to {self.repo_filepath.name}", main_logger=__file__)
            except Exception as e:
                # Use custom log statement
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to append batch of new files to repository: {e}", main_logger=__file__, exc_info=True)
                # Note: _next_designation and _data_hashes might be inconsistent if append fails here.
                # Consider adding logic to handle this potential inconsistency if critical.
        else:
            # Use custom log statement
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::No new files found in the specified folder to add to the repository.", main_logger=__file__)

        # Use custom log statement for summary
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Folder scan complete. Files Processed: {processed_count}, Entries Added: {added_count}, Paths Skipped (already tracked/unsupported): {skipped_count}, Errors during scan/processing: {error_count}", main_logger=__file__)

    def remove_folder(self, folder_path: str):
        """
        Removes all entries from the main repository whose Filepath
        starts with the specified folder path.
        Rule: 7.1.B
        """
        abs_folder_path = str(Path(folder_path).resolve())
        log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Removing entries for folder and subfolders: {abs_folder_path}", main_logger=__file__)

        rows_to_keep = []
        removed_count = 0

        try:
            # Read the repo, keeping only rows that DON'T match the path prefix
            for row in self.read_repo_stream():
                 file_path = row.get(COL_FILEPATH)
                 if file_path and file_path.startswith(abs_folder_path):
                      log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Marking for removal: {file_path} (Designation: {row.get(COL_DESIGNATION)})", main_logger=__file__)
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

                log_statement(loglevel='info', logstatement=f"{LOG_INS}::Removed {removed_count} entries associated with folder: {abs_folder_path}", main_logger=__file__)
                # Reload internal state as designations/hashes might have changed if we renumbered
                self._next_designation = self._get_next_designation()
                self._data_hashes = self._load_existing_hashes()
            else:
                log_statement(loglevel='info', logstatement=f"{LOG_INS}::No entries found for folder: {abs_folder_path}", main_logger=__file__)

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to remove folder entries from repository: {e}", main_logger=__file__, exc_info=True)
            # State might be inconsistent, consider recovery or warning user

    def update_status(self, designation: int, new_status: str, target_repo_path: Optional[Path] = None):
        """
        Updates the status for a specific designation in the specified repository file.
        Uses the inefficient read-modify-write approach for zstd files.
        """
        repo_path = target_repo_path or self.repo_filepath
        if not repo_path.exists():
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::Cannot update status. Repository file not found: {repo_path}", main_logger=__file__)
             return False

        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Attempting to update status to '{new_status}' for designation {designation} in {repo_path.name}", main_logger=__file__)
        updated = False
        rows_to_write = []

        try:
            for row in self.read_repo_stream(repo_path):
                current_designation_str = row.get(COL_DESIGNATION)
                try:
                    if current_designation_str and int(current_designation_str) == designation:
                        if row.get(COL_STATUS) != new_status:
                            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Updating status for Designation {designation} from '{row.get(COL_STATUS)}' to '{new_status}' in {repo_path.name}", main_logger=__file__)
                            row[COL_STATUS] = new_status
                            updated = True
                        else:
                             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Status for Designation {designation} is already '{new_status}' in {repo_path.name}", main_logger=__file__)
                             # Keep row as is, no change needed
                    # Keep all rows (modified or not) to rewrite the file
                    rows_to_write.append(row)

                except (ValueError, TypeError):
                     log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Skipping row due to invalid designation: {row}", main_logger=__file__)
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
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Successfully updated status for designation {designation} in {repo_path.name}", main_logger=__file__)
                return True
            else:
                 log_statement(loglevel='debug', logstatement=f"{LOG_INS}::No status update needed or designation {designation} not found in {repo_path.name}", main_logger=__file__)
                 return False # Return False if no update occurred

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to update status for designation {designation} in {repo_path.name}: {e}", main_logger=__file__, exc_info=True)
            return False

    def process_data_list(self):
        """
        Iterates through files marked as 'Loaded' (L) in the main repository,
        processes their content, saves output to a mirrored structure,
        and updates status to 'Processed' (P).
        Rule: 7.1.C
        """
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Starting data processing pipeline...", main_logger=__file__)
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
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to read repository to find files for processing: {e}", main_logger=__file__, exc_info=True)
            return # Cannot proceed

        if not files_to_process:
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::No files found with status 'Loaded' to process.", main_logger=__file__)
            return

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Found {len(files_to_process)} files to process.", main_logger=__file__)

        for row in files_to_process:
            designation = int(row.get(COL_DESIGNATION, -1))
            filepath_str = row.get(COL_FILEPATH)
            filetype = row.get(COL_FILETYPE)

            if designation == -1 or not filepath_str:
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Skipping invalid row during processing: {row}", main_logger=__file__)
                continue

            input_path = Path(filepath_str)
            if not input_path.exists():
                 log_statement(loglevel='error', logstatement=f"{LOG_INS}::File listed in repository not found: {filepath_str}. Setting status to Error.", main_logger=__file__)
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
                 continue

            # --- Create Mirrored Output Path ---
            relative_path = input_path.relative_to(input_path.anchor) # Get path relative to drive root
            output_subpath = output_base / relative_path.parent
            output_subpath.mkdir(parents=True, exist_ok=True)
            output_filename = input_path.stem + PROCESSED_EXT + ".zst" # Add .proc and .zst extension
            output_filepath = output_subpath / output_filename

            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Processing Designation {designation}: {filepath_str} -> {output_filepath}", main_logger=__file__)

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
                         log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Processing logic for file type {filetype} not implemented yet. Skipping content processing.", main_logger=__file__)
                         # yield "" # Yield nothing or handle differently
                         raise NotImplementedError(f"Processing for {filetype} not implemented.")

                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Processed {line_count} lines/units from {filepath_str}", main_logger=__file__)


                # Stream compressed output
                stream_compress_lines(str(output_filepath), processed_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_PROCESSED):
                    processed_count += 1
                    log_statement(loglevel='info', logstatement=f"{LOG_INS}::Successfully processed and updated status for Designation {designation}", main_logger=__file__)
                    # TODO: Add entry to processed_repository.csv.zst (similar _append_repo_stream logic)
                else:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS}::Processed Designation {designation} but FAILED to update status in repository.", main_logger=__file__)
                    error_count += 1
                    # Consider cleanup of the generated .proc.zst file?

            except NotImplementedError as e:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS}::Processing failed for Designation {designation} ({filepath_str}): {e}", main_logger=__file__)
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except FileNotFoundError:
                 log_statement(loglevel='error', logstatement=f"{LOG_INS}::Input file disappeared during processing: {filepath_str}. Setting status to Error.", main_logger=__file__)
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error processing Designation {designation} ({filepath_str}): {e}", main_logger=__file__, exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if output_filepath.exists():
                    try: output_filepath.unlink()
                    except OSError: pass

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Data processing finished. Processed successfully: {processed_count}, Errors: {error_count}", main_logger=__file__)

    def tokenize_processed_data(self):
        """
        Iterates through files marked as 'Processed' (P) in the main repository,
        finds the corresponding '.proc.zst' file, tokenizes its content,
        saves output to a mirrored structure with '.token.zst',
        and updates status to 'Tokenized' (T).
        Rule: 7.1.D
        """
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Starting tokenization pipeline...", main_logger=__file__)
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
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to read repository to find files for tokenization: {e}", main_logger=__file__, exc_info=True)
            return # Cannot proceed

        if not files_to_tokenize:
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::No files found with status 'Processed' to tokenize.", main_logger=__file__)
            return

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Found {len(files_to_tokenize)} processed files to tokenize.", main_logger=__file__)

        # Placeholder: Initialize your tokenizer here
        # tokenizer = YourTokenizerClass() or load_tokenizer_function()
        # Example dummy tokenizer function
        def dummy_tokenize(text_line):
            return " ".join(text_line.lower().split()) # Simple split and join

        for row in files_to_tokenize:
            designation = int(row.get(COL_DESIGNATION, -1))
            original_filepath_str = row.get(COL_FILEPATH)

            if designation == -1 or not original_filepath_str:
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Skipping invalid row during tokenization: {row}", main_logger=__file__)
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
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Processed file not found for Designation {designation}: {processed_filepath}. Setting status to Error.", main_logger=__file__)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                continue

            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Tokenizing Designation {designation}: {processed_filepath} -> {tokenized_filepath}", main_logger=__file__)

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
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Tokenized {line_count} lines/units from {processed_filepath}", main_logger=__file__)

                # Stream compressed output for tokenized data
                stream_compress_lines(str(tokenized_filepath), tokenized_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_TOKENIZED):
                    tokenized_count += 1
                    log_statement(loglevel='info', logstatement=f"{LOG_INS}::Successfully tokenized and updated status for Designation {designation}", main_logger=__file__)
                    # TODO: Add entry to tokenized_repository.csv.zst
                else:
                    log_statement(loglevel='error', logstatement=f"{LOG_INS}::Tokenized Designation {designation} but FAILED to update status in repository.", main_logger=__file__)
                    error_count += 1
                    # Consider cleanup?

            except FileNotFoundError: # Should be caught earlier, but just in case
                 log_statement(loglevel='error', logstatement=f"{LOG_INS}::Processed file disappeared during tokenization: {processed_filepath}. Setting status to Error.", main_logger=__file__)
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error tokenizing Designation {designation} ({processed_filepath}): {e}", main_logger=__file__, exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if tokenized_filepath.exists():
                    try: tokenized_filepath.unlink()
                    except OSError: pass

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Tokenization finished. Tokenized successfully: {tokenized_count}, Errors: {error_count}", main_logger=__file__)

    # Placeholder for other methods like creating the DataLoader file (Rule 7.1.E)
    def create_dataloader_file(self, output_filename: str = "dataloader_package.zst"):
        """
        Gathers information from the tokenized repository and potentially packages
        tokenized files into a single compressed file for DataLoader usage.
        Rule: 7.1.E
        """
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Starting DataLoader file creation...", main_logger=__file__)
        tokenized_files_info = []
        try:
            # Read the main repo to find tokenized files and their original paths
            for row in self.read_repo_stream():
                if row.get(COL_STATUS) == STATUS_TOKENIZED:
                    tokenized_files_info.append(row)
            # Or, preferably, read from a dedicated tokenized_repository.csv.zst if created
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to read repository to find tokenized files: {e}", main_logger=__file__, exc_info=True)
            return

        if not tokenized_files_info:
             log_statement(loglevel="warning", logstatement=f"{LOG_INS}::No tokenized files found to create DataLoader package.", main_logger=__file__)
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
                 log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Tokenized file not found for Designation {designation}: {tokenized_full_path}", main_logger=__file__)

        if not metadata:
             log_statement(loglevel="warning", logstatement=f"{LOG_INS}::No valid tokenized file paths found for metadata.", main_logger=__file__)
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
             log_statement(loglevel='info', logstatement=f"{LOG_INS}::Created DataLoader metadata file: {output_filepath}", main_logger=__file__)
             log_statement(loglevel='info', logstatement=f"{LOG_INS}::Contained metadata for {len(metadata)} tokenized files.", main_logger=__file__)

        except ImportError:
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::json module not found. Cannot create JSON metadata file.", main_logger=__file__)
        except Exception as e:
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to create DataLoader metadata file '{output_filepath}': {e}", main_logger=__file__, exc_info=True)
             if output_filepath.exists():
                 try: output_filepath.unlink()
                 except OSError: pass

    def _compress_file(self, filepath: Path) -> Path:
        """Compresses the file using zstandard."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Compressing file: {filepath.name}", main_logger=__file__)
        try:
            compressed_path = filepath.with_suffix('.zst')
            with open(filepath, 'rb') as f_in:
                with open(compressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdCompressor()
                    dctx.copy_stream(f_in, f_out)
            return compressed_path
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to compress file {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Compression failed: {e}")
            return None
        except FileNotFoundError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::File not found during compression: {filepath}", main_logger=__file__, exc_info=True)
            return None
    
    def _decompress_file(self, filepath: Path) -> Path:
        """Decompresses the file using zstandard."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Decompressing file: {filepath.name}", main_logger=__file__)
        try:
            decompressed_path = filepath.with_suffix('')
            with open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
            return decompressed_path
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to decompress file {filepath}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Decompression failed: {e}")
            return None
        except FileNotFoundError:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::File not found during decompression: {filepath}", main_logger=__file__, exc_info=True)
            return None

    def _save_processed(self, data, original_path: Path) -> Path | None:
        """Saves processed data to a file, compressing if necessary."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Saving processed data for {original_path.name}", main_logger=__file__)
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
            else: log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unsupported save type: {type(data)}", main_logger=__file__); return None
            if COMPRESSION_ENABLED: suffix += ".zst"
            save_path = PROCESSED_DATA_DIR / f"{save_stem}_processed{suffix}"
            if save_path.exists():
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Processed file already exists: {save_path}. Overwriting.", main_logger=__file__)
                try: save_path.unlink()
                except OSError: log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to remove existing processed file: {save_path}", main_logger=__file__, exc_info=True)
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for saving (convert CuPy to NumPy if needed)
            data_to_save = data
            if cp is not np and isinstance(data, cp.ndarray):
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Converting CuPy array to NumPy for saving.", main_logger=__file__)
                data_to_save = cp.asnumpy(data)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                # Ensure pandas data is ready (no specific conversion needed here unless GPU involved earlier)
                pass
            elif not isinstance(data, np.ndarray):
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unexpected data type for saving: {type(data)}. Cannot save.", main_logger=__file__)
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

                log_statement(loglevel='info', logstatement=f"{LOG_INS}::Saved processed file: {save_path}", main_logger=__file__)
                return save_path
            except Exception as e:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Save failed: {save_path}: {e}", main_logger=__file__, exc_info=True)
                if save_path.exists(): 
                    try: 
                        os.remove(save_path)
                    except OSError: pass
                return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to save processed data for {original_path}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(original_path, status='error', error_message=f"Failed to save processed data: {e}")
            return None

    def _save_processed(self, processed_data, source_filepath: Path):
        """Saves processed data to a file, compressing if necessary."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Saving processed data for {source_filepath.name}", main_logger=__file__)

# --- Tokenizer ---
class Tokenizer:
    """ Loads processed, tokenizes, saves compressed tensors. """
    def __init__(self, max_workers: int | None = None):
        self.repo = DataRepository(repo_path=TOKENIZED_REPO_FILENAME)
        resolved_max_workers = max_workers if max_workers is not None else DataProcessingConfig.MAX_WORKERS
        self.max_workers = max(1, resolved_max_workers)
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Initializing Tokenizer with max_workers={self.max_workers}", main_logger=__file__)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.device = DEFAULT_DEVICE
        TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        if hasattr(self, 'executor') and self.executor:
            try: self.executor.shutdown(wait=True);
            except Exception: pass

    def tokenize_all(self, base_dir_filter: Optional[Path] = None, statuses_to_process=(STATUS_PROCESSED,)): # Use constant STATUS_PROCESSED
        """Tokenizes files matching status, optionally filtered by base_dir, with progress bar."""
        files_to_tokenize_info = []
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::tokenize_all called with filter: {base_dir_filter}, statuses: {statuses_to_process}", main_logger=__file__)

        with self.repo.lock:
             # Get source file paths matching status and optional base_dir filter
             source_paths = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)

             if not source_paths:
                 log_statement(loglevel='info', logstatement=f"{LOG_INS}::No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] found to tokenize.", main_logger=__file__)
                 return

             log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Found {len(source_paths)} source files with status {statuses_to_process}. Checking for processed paths...", main_logger=__file__)

             # Get corresponding processed paths efficiently
             source_paths_str_set = {str(p.resolve()) for p in source_paths}
             relevant_rows = self.repo.df[self.repo.df[COL_FILEPATH].isin(source_paths_str_set)].copy() # Use .copy()

             for index, row in relevant_rows.iterrows():
                 src_path_str = row[COL_FILEPATH]
                 proc_path_str_relative = row['processed_path'] # Path relative to PROCESSED_DATA_DIR

                 if proc_path_str_relative:
                      # Construct full absolute path to processed file
                      proc_path = PROCESSED_DATA_DIR / proc_path_str_relative
                      if proc_path.exists():
                           files_to_tokenize_info.append((Path(src_path_str), proc_path))
                      else:
                           log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Processed path found in repo ({proc_path_str_relative}) but file missing: {proc_path}. Skipping tokenization for {src_path_str}.", main_logger=__file__)
                           # Optionally set status back to error? self.repo.update_entry(Path(src_path_str), status=STATUS_ERROR, error_message="Processed file missing")
                 else:
                      log_statement(loglevel="warning", logstatement=f"{LOG_INS}::No valid processed path found in repo for source: {src_path_str}. Skipping tokenization.", main_logger=__file__)

        if not files_to_tokenize_info:
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::No existing processed files found for the files matching status {statuses_to_process}.", main_logger=__file__)
            return

        log_statement(loglevel='info', logstatement=f"{LOG_INS}::Starting tokenization for {len(files_to_tokenize_info)} files [base_dir: {base_dir_filter}].", main_logger=__file__)

        # Use the instance's ProcessPoolExecutor
        if not hasattr(self, 'executor') or self.executor is None:
             log_statement(loglevel='error', logstatement=f"{LOG_INS}::Tokenizer executor not initialized. Cannot tokenize.", main_logger=__file__)
             return

        futures = [self.executor.submit(self._tokenize_file, src_path, proc_path)
                   for src_path, proc_path in files_to_tokenize_info]

        # --- Add tqdm Progress Bar ---
        pbar_desc = f"Tokenizing [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]" # Shorten name
        results_tokenized = 0
        results_error = 0
        # Iterate over completed futures with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc=pbar_desc, unit="file", leave=True):
            try:
                result_path = future.result() # _tokenize_file returns source path on success, None on failure
                if result_path:
                    results_tokenized += 1
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Tokenization future completed successfully for: {result_path.name}", main_logger=__file__)
                else:
                    results_error += 1
                    log_statement(loglevel='warning', logstatement=f"{LOG_INS}::Tokenization future returned None (likely failed).", main_logger=__file__)
            except Exception as e:
                results_error += 1
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Error retrieving result from tokenizer future: {e}", main_logger=__file__, exc_info=True)
            # No pbar.update() needed here

        # Optional: Save repository after all tokenization if status updates need saving
        self.repo.save()
        log_statement(loglevel='info', logstatement=f"{LOG_INS}::File tokenization stage complete. Successful: {results_tokenized}, Errors: {results_error}.", main_logger=__file__)

    def _tokenize_file(self, source_filepath: Path, processed_filepath: Path):
        """Loads processed, converts to tensor, saves compressed."""
        # (Implementation remains same - uses modified load/save)
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Tokenizing file: {processed_filepath.name}...", main_logger=__file__)
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
            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Tokenized: {processed_filepath.name} -> {save_path.name}", main_logger=__file__)
            return source_filepath
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Tokenization failed for {processed_filepath.name}: {e}", main_logger=__file__, exc_info=True)
            self.repo.update_entry(source_filepath, status='error', error_message=f"{e}")
            if save_path and save_path.exists():
                try: save_path.unlink()
                except OSError: pass
            return None

    def _load_processed(self, filepath: Path):
        """Loads compressed or uncompressed processed data (CSV or NPY)."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Loading processed: {filepath.name}", main_logger=__file__)
        if not filepath.exists():
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Not found: {filepath}", main_logger=__file__)
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
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Cannot load processed: Unknown suffix '{actual_suffix}' for file {filepath.name}", main_logger=__file__)
                return None

            # Close the buffer if it was used
            if buffer: buffer.close()
            return data
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Load failed for {filepath}: {e}", main_logger=__file__, exc_info=True)
            # Ensure buffer is closed on error too
            if buffer and not buffer.closed: buffer.close()
            return None

    def _vectorize(self, data) -> torch.Tensor | None:
        """Converts loaded processed data into a tensor."""
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Vectorizing data type: {type(data)}", main_logger=__file__)
        try:
            if isinstance(data, torch.Tensor):
                return data.to(dtype=torch.float32, device=self.device)
            elif isinstance(data, (np.ndarray, cp.ndarray)):
                # Convert CuPy array to NumPy before converting to Torch tensor if necessary
                if cp is not np and isinstance(data, cp.ndarray):
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Converting CuPy array to NumPy before Torch tensor conversion.", main_logger=__file__)
                    data = cp.asnumpy(data)
                # Ensure data is C-contiguous before creating tensor
                if not data.flags['C_CONTIGUOUS']:
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Data is not C-contiguous, making a copy.", main_logger=__file__)
                    data = np.ascontiguousarray(data)
                return torch.tensor(data, dtype=torch.float32, device=self.device)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Vectorizing pandas data - using DUMMY implementation.", main_logger=__file__)
                # Convert pandas data to NumPy first
                numpy_data = data.to_numpy(dtype=np.float32)
                # Ensure data is C-contiguous before creating tensor
                if not numpy_data.flags['C_CONTIGUOUS']:
                     log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Pandas data (as NumPy) is not C-contiguous, making a copy.", main_logger=__file__)
                     numpy_data = np.ascontiguousarray(numpy_data)
                # Dummy implementation: create zeros based on rows, fixed columns
                num_rows = numpy_data.shape[0]
                # Ensure the dummy tensor matches the expected shape if possible
                # If 1D (Series), create (num_rows, 1). If 2D (DataFrame), use original cols or dummy.
                num_cols = numpy_data.shape[1] if numpy_data.ndim > 1 else 1
                dummy_cols = 10 # Keep the dummy dimension for now
                log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Creating dummy tensor of shape ({num_rows}, {dummy_cols})", main_logger=__file__)
                return torch.zeros((num_rows, dummy_cols), dtype=torch.float32, device=self.device) # Dummy
            else:
                log_statement(loglevel='error', logstatement=f"{LOG_INS}::Unsupported type for vectorization: {type(data)}", main_logger=__file__)
                return None
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Vectorization failed: {e}", main_logger=__file__, exc_info=True)
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
             log_statement(loglevel="warning", logstatement=f"{LOG_INS}::Processed path '{original_processed_path.name}' did not strictly follow '_processed.npy/csv' pattern. Using stem '{save_stem}'.", main_logger=__file__)


        suffix = ".pt.zst" if COMPRESSION_ENABLED else ".pt"
        save_path = TOKENIZED_DATA_DIR / f"{save_stem}_tokenized{suffix}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Saving tokenized tensor to: {save_path.name}", main_logger=__file__, exc_info=False)
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

            log_statement(loglevel='info', logstatement=f"{LOG_INS}::Saved tokenized tensor: {save_path}", main_logger=__file__, exc_info=False)
            return save_path
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"{LOG_INS}::Save tokenized failed for {save_path}: {e}", main_logger=__file__, exc_info=True)
            if save_path.exists():
                try:
                    save_path.unlink()
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS}::Removed partially saved file: {save_path}", main_logger=__file__, exc_info=False)
                except OSError as unlink_e:
                     log_statement(loglevel='error', logstatement=f"{LOG_INS}::Failed to remove partially saved file {save_path}: {unlink_e}", main_logger=__file__, exc_info=False)
            return None
