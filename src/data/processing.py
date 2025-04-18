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
from typing import Union, List, Dict, Tuple, Optional, Generator
from functools import partial
import torch
import pandas as pd
import zstandard as zstd
from ..utils.file_processor import FileProcessor
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
import fitz  # PyMuPDF
import shutil
import json
from datetime import datetime, timezone # Import timezone
import os
import time
import io

# Import project configuration and utilities
try:
    from ..utils.config import (
        DataProcessingConfig, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, DEFAULT_DEVICE,
        PROJECT_ROOT, BASE_DATA_DIR, COMPRESSION_LEVEL, COMPRESSION_ENABLED,
        DATA_REPO_FILE # Import central repo file path
    )
    from ..utils.logger import configure_logging
    from ..utils.gpu_switch import check_gpu_support
    from ..utils.hashing import hash_filepath, unhash_filepath, generate_data_hash
    from ..utils.compression import stream_decompress_lines, stream_compress_lines, compress_file, decompress_file
    from src.data.constants import (
    REPO_DIR, MAIN_REPO_FILENAME, PROCESSED_REPO_FILENAME, TOKENIZED_REPO_FILENAME,
    MAIN_REPO_HEADER, COL_DESIGNATION, COL_FILETYPE, COL_FILEPATH, COL_HASHED_PATH_ID,
    COL_COMPRESSED_FLAG, COL_MOD_DATE, COL_ACC_DATE, COL_DATA_HASH, COL_IS_COPY_FLAG,
    COL_STATUS, STATUS_LOADED, STATUS_PROCESSED, STATUS_TOKENIZED, STATUS_UNKNOWN, STATUS_ERROR,
    ACCEPTED_FILE_TYPES, OUTPUT_DIR_BASE, PROCESSED_EXT, TOKENIZED_EXT
)
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logger and other utils imported successfully.")
except ImportError:
    logging.ERROR("Failed relative import in processing.py, trying absolute from src...")
    try:
        from ..utils.config import (
            DataProcessingConfig, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, DEFAULT_DEVICE,
            PROJECT_ROOT, BASE_DATA_DIR, COMPRESSION_LEVEL, COMPRESSION_ENABLED,
            DATA_REPO_FILE
        )
        from ..utils.logger import configure_logging
        from ..utils.gpu_switch import check_gpu_support
        configure_logging()
        logger = logging.getLogger(__name__)
        logger.info("Logger and other utils imported successfully.")
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
    import cudf
    UnsupportedCUDAError = cudf.errors.UnsupportedCUDAError
    # Check compatibility before proceeding
    compatible_gpus = check_gpu_support(min_compute_capability=7.0) # Use config value?
    if compatible_gpus:
        cudf.validate_setup() # Can raise if setup is wrong even with compatible GPU
        try:
            import cupy as cp
            cuml_spec = importlib.util.find_spec("cuml")
            if cuml_spec:
                from cuml.preprocessing import StandardScaler as CumlScaler
            else:
                CumlScaler = None
                logger.warning("cuML library not found. Scaling skipped.")
            GPU_AVAILABLE = True
            logger.info(f"Compatible GPU detected ({len(compatible_gpus)} devices) and GPU libraries imported successfully.")
        except ImportError as other_import_error:
            logger.warning(f"cuDF compatible GPU found, but other libs (CuPy/cuML) failed: {other_import_error}. Falling back to CPU.")
            GPU_AVAILABLE = False
    else:
         logger.info("No compatible GPU (Compute Capability >= 7.0) found by check_gpu_support. Using CPU fallback.")
         GPU_AVAILABLE = False # Ensure it's False if no compatible GPU

except ImportError as initial_import_err:
    logger.error(f"GPU library 'cudf' not found ({initial_import_err}). Using CPU fallback.")
    GPU_AVAILABLE = False
except Exception as initial_cudf_err:
    if UnsupportedCUDAError and isinstance(initial_cudf_err, UnsupportedCUDAError):
         logger.warning(f"cuDF found but GPU is incompatible ({initial_cudf_err}). Using CPU fallback.")
    else:
         logger.error(f"Unexpected error during initial cuDF import/setup: {initial_cudf_err}", exc_info=True)
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    logger.warning("Using CPU fallback (pandas/numpy).")
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
            logger.warning("Using sklearn.preprocessing.StandardScaler as CPU fallback for scaling.")
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
            logger.warning("Scikit-learn not found. Using basic dummy scaler (no operation).")
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
                    logger.info(f"Downloading NLTK '{resource_id}' data...")
                    nltk.download(resource_id, quiet=True)
    # download_nltk_data() # Consider calling this conditionally or manually

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    logger.info("NLTK components loaded successfully.")
except ImportError: logger.warning("NLTK library not found. Text processing limited.")
except LookupError as e: logger.warning(f"NLTK data not found ({e}). Text processing limited.")
except Exception as e: logger.error(f"Unexpected error loading NLTK: {e}", exc_info=True)
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
        self.repo = DataRepository()
        self.repo_dir = Path(repo_dir)
        self.repo_filepath = self.repo_dir / filename
        self.processed_repo_filepath = self.repo_dir / PROCESSED_REPO_FILENAME
        self.tokenized_repo_filepath = self.repo_dir / TOKENIZED_REPO_FILENAME
        self.repo_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_repo_exists(self.repo_filepath, MAIN_REPO_HEADER)
        # Optionally ensure processed/tokenized repos exist - depends on workflow
        # self._ensure_repo_exists(self.processed_repo_filepath, PROCESSED_REPO_HEADER) # Define header if needed
        # self._ensure_repo_exists(self.tokenized_repo_filepath, TOKENIZED_REPO_HEADER) # Define header if needed

        self._next_designation = self._get_next_designation()
        self._data_hashes = self._load_existing_hashes() # For copy detection
        self._file_hashes = {}
        resolved_max_workers = max_workers if max_workers is not None else DataProcessingConfig.MAX_WORKERS
        self.max_workers = max(1, resolved_max_workers)
        logger.info(f"Initializing DataProcessor with max_workers={self.max_workers}")
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.scaler = self._initialize_scaler()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataProcessor initialized. Base directories: {PROCESSED_DATA_DIR}, {TOKENIZED_DATA_DIR}")
        # Initialize the scaler
        if self.scaler:
            logger.info("Scaler initialized successfully.")
        else:
            logger.warning("Scaler initialization failed. No scaling will be applied.")
        # Initialize the lock for thread-safe operations
        self.lock = Lock()
        # Initialize the regex for text cleaning
        self.text_cleaning_regex = getattr(DataProcessingConfig, 'TEXT_CLEANING_REGEX', r'[^\w\s\-\.]') # Default regex
        # Initialize the regex for text cleaning
        self.cleaning_regex = re.compile(self.text_cleaning_regex)
        
    def _initialize_scaler(self):
        """Initializes the appropriate scaler based on GPU availability."""
        if GPU_AVAILABLE and CumlScaler is not None and not isinstance(CumlScaler, type(CumlScaler_dummy)):
             try:
                 scaler = CumlScaler()
                 logger.info("Using cuML StandardScaler for numerical processing.")
                 return scaler
             except Exception as scaler_e:
                  logger.error(f"Failed to initialize CumlScaler: {scaler_e}. Scaling disabled.", exc_info=True)
                  return None
        # Fallback to Sklearn if GPU/cuML unavailable but Sklearn is
        elif 'SklearnScalerWrapper' in globals() and SklearnScalerWrapper is not None:
             try:
                 scaler = SklearnScalerWrapper()
                 logger.info("Using Sklearn StandardScaler fallback for numerical processing.")
                 return scaler
             except Exception as scaler_e:
                  logger.error(f"Failed to initialize SklearnScalerWrapper: {scaler_e}. Scaling disabled.", exc_info=True)
                  return None
        else:
            logger.info("No suitable scaler found (cuML or Sklearn). Numerical scaling will be skipped.")
            return None
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
        logger.info(f"Starting repository scan using base directory: {BASE_DATA_DIR}")
        self.repo.scan_and_update(BASE_DATA_DIR)
    def __del__(self):
        """Ensures proper cleanup of the executor on object deletion."""
        if hasattr(self, 'executor') and self.executor:
            try:
                logger.info("Shutting down DataProcessor executor...") 
                self.executor.shutdown(wait=True)
                # Wait for tasks on shutdown
                logger.info("DataProcessor executor shut down.")
            except Exception as e: 
                logger.error(f"Error shutting down executor: {e}")

    def process_all(self, base_dir_filter: Optional[Path] = None, statuses_to_process=('discovered', 'error')):
        """Processes files matching status, optionally filtered by base_dir."""
        files_to_process = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
        if not files_to_process:
            logger.info(f"No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] found to process.")
            return
        logger.info(f"Starting processing for {len(files_to_process)} files [base_dir: {base_dir_filter}].")
        futures = [self.executor.submit(self._process_file, f_path) for f_path in files_to_process]
        with tqdm(total=len(futures), desc=f"Processing Files [{base_dir_filter.name if base_dir_filter else 'All'}]") as pbar:
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: logger.error(f"Error retrieving result from future: {e}", exc_info=True)
                finally: pbar.update(1)
        self.repo.save()
        logger.info("File processing complete.")

    def process_file(self, filepath: Path):
        """Processes a single file based on its type and updates the repository."""
        logger.debug(f"Processing file: {filepath.name}")
        try:
            file_extension = self._get_file_extension(filepath)
            if file_extension in ['.txt', '.csv', '.json', '.jsonl']:
                processed_data = self._process_text(filepath)
            elif file_extension in ['.xlsx', '.xls']:
                processed_data = self._process_excel(filepath)
            elif file_extension == '.pdf':
                processed_data = self._process_pdf(filepath)
            elif file_extension == '.docx':
                processed_data = self._process_docx(filepath)
            elif file_extension == '.odt':
                processed_data = self._process_odt(filepath)
            elif file_extension == '.html' or file_extension == '.xml':
                processed_data = self._process_html_xml(filepath)
            elif file_extension == '.rtf':
                processed_data = self._process_rtf(filepath)
            elif file_extension == '.epub':
                processed_data = self._process_epub(filepath)
            elif file_extension == '.zip':
                processed_data = self._process_zip(filepath)
            else:
                logger.warning(f"Unsupported file type: {file_extension}. Skipping.")
                return
            if processed_data is not None:
                # Save the processed data to the repository
                self.repo.update_entry(filepath, status='processed', processed_data=processed_data)
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=str(e))

    def _process_text(self, filepath: Path):
        """Processes text data (CPU based)."""
        logger.debug(f"Processing text file: {filepath.name}")
        try:
            content = self._read_content(filepath)
            lines = content.splitlines()
            if not lines: return pd.Series(dtype=str)
            pdf = pd.Series(lines)
            processed = pdf.str.lower()
            processed = processed.str.replace(cleaning_regex, '', regex=True) # Keep word chars, space, hyphen, dot
            processed = processed.str.replace(r'\s+', ' ', regex=True).str.strip()
            if NLTK_AVAILABLE and lemmatizer:
                def lemmatize_and_filter(text):
                     words = text.split()
                     lemmatized = [lemmatizer.lemmatize(w) for w in words]
                     filtered = [w for w in lemmatized if w not in stop_words]
                     return " ".join(filtered)
                processed = processed.apply(lemmatize_and_filter)
            return processed.dropna()
        except Exception as e:
            logger.error(f"Text processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Text processing failed: {e}")
            return None

    def _process_numerical(self, filepath: Path):
        """Processes numerical data (GPU or CPU), applies scaling."""
        # (Implementation remains the same as previous version - handles fallback)
        logger.debug(f"Processing numerical file: {filepath.name}")
        num_array = None
        # --- Try JSON Lines first ---
        try:
            use_gpu_read = GPU_AVAILABLE and cudf is not None and not isinstance(cudf, type(cudf_dummy))
            if use_gpu_read:
                try:
                    df = cudf.read_json(filepath, lines=True, dtype=False)
                    num_array = df.to_cupy() if len(df.columns) > 1 else df.iloc[:, 0].to_cupy()
                except Exception: df = None # Fallback to pandas read
            if num_array is None: # If GPU read failed or not attempted
                 df = pd.read_json(filepath, lines=True, dtype=False)
                 num_array = df.to_numpy() if len(df.columns) > 1 else df.iloc[:, 0].to_numpy()
            if num_array.dtype.kind not in ['f', 'i']: num_array = num_array.astype(np.float32) # Ensure numeric
        except (ValueError, Exception): num_array = None # JSON Lines read failed

        # --- Fallback to full parse ---
        if num_array is None:
            try:
                json.loads(self._read_content(filepath))
                if isinstance(data, list): num_array = np.array(data, dtype=np.float32)
                elif isinstance(data, dict) and 'values' in data: num_array = np.array(data['values'], dtype=np.float32)
                elif isinstance(data, dict) and 'input' in data: num_array = np.array(data['input'], dtype=np.float32)
                else: raise ValueError("Unrecognized JSON structure")
            except Exception as e:
                logger.error(f"Numerical processing failed for {filepath}: {e}", exc_info=True)
                self.repo.update_entry(filepath, status='error', error_message=f"Numerical processing failed: {e}")
                return None

        # --- Process the array (common) ---
        try:
            if num_array is None or num_array.size == 0: return cp.array([]) if GPU_AVAILABLE else np.array([]) # Return empty
            if num_array.dtype.kind not in ['f', 'i']: num_array = num_array.astype(np.float32)
            if GPU_AVAILABLE and cp is not np and isinstance(num_array, np.ndarray): num_array = cp.asarray(num_array) # Move to GPU
            processed_array = num_array # Start with potentially GPU array
            if self.scaler: # Apply scaler if available
                original_ndim = num_array.ndim
                array_2d = num_array.reshape(-1, 1) if original_ndim == 1 else num_array
                try:
                    scaled_array_2d = self.scaler.fit_transform(array_2d)
                    processed_array = scaled_array_2d.reshape(-1) if original_ndim == 1 else scaled_array_2d
                except Exception as scale_err:
                    logger.error(f"Scaling failed for {filepath.name}: {scale_err}. Returning unscaled.", exc_info=True)
                    # processed_array remains the unscaled num_array
            return processed_array
        except Exception as final_err:
            logger.error(f"Final numerical processing failed for {filepath.name}: {final_err}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Final numerical processing failed: {final_err}")
            return None

    def _process_pdf(self, filepath: Path):
        """Processes PDF files using PyMuPDF (fitz)."""
        logger.debug(f"Processing PDF file: {filepath.name}")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"PDF processing failed: {e}")
            return None
        
    def _process_docx(self, filepath: Path):
        """Processes DOCX files using python-docx."""
        logger.debug(f"Processing DOCX file: {filepath.name}")
        try:
            import docx
            doc = docx.Document(filepath)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"DOCX processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"DOCX processing failed: {e}")
            return None
        
    def _process_odt(self, filepath: Path):
        """Processes ODT files using odfpy."""
        logger.debug(f"Processing ODT file: {filepath.name}")
        try:
            import odfpy
            from odf.opendocument import OpenDocumentText
            doc = OpenDocumentText(filepath)
            text = []
            for elem in doc.getElementsByType(odfpy.text.P):
                text.append(elem.firstChild.data)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"ODT processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ODT processing failed: {e}")
            return None
        
    def _process_excel(self, filepath: Path):
        """Processes Excel files using pandas."""
        logger.debug(f"Processing Excel file: {filepath.name}")
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            return df.to_numpy()
        except Exception as e:
            logger.error(f"Excel processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Excel processing failed: {e}")
            return None
        
    def _process_html_xml(self, filepath: Path):
        """Processes HTML/XML files using BeautifulSoup."""
        logger.debug(f"Processing HTML/XML file: {filepath.name}")
        try:
            import bs4
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            soup = bs4.BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.error(f"HTML/XML processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"HTML/XML processing failed: {e}")
            return None
        
    def _process_rtf(self, filepath: Path):
        """Processes RTF files using striprtf."""
        logger.debug(f"Processing RTF file: {filepath.name}")
        try:
            import striprtf
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return striprtf.strip(content)
        except Exception as e:
            logger.error(f"RTF processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"RTF processing failed: {e}")
            return None
        
    def _process_epub(self, filepath: Path):
        """Processes EPUB files using EbookLib."""
        logger.debug(f"Processing EPUB file: {filepath.name}")
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(filepath)
            text = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                text.append(item.get_body_content_str())
            return "\n".join(text)
        except Exception as e:
            logger.error(f"EPUB processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"EPUB processing failed: {e}")
            return None
        
    def _process_zip(self, filepath: Path):
        """Processes ZIP files using zipfile."""
        logger.debug(f"Processing ZIP file: {filepath.name}")
        try:
            import zipfile
            with zipfile.ZipFile(filepath, 'r') as z:
                text = []
                for file in z.namelist():
                    with z.open(file) as f:
                        text.append(f.read().decode('utf-8'))
                return "\n".join(text)
        except Exception as e:
            logger.error(f"ZIP processing failed for {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"ZIP processing failed: {e}")
            return None

    def _read_content(self, filepath: Path):
        """Reads the content of a file, handling different encodings."""
        logger.debug(f"Reading content from {filepath.name}")
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
                logger.error(f"Failed to read file {filepath}: {e}", exc_info=True)
                self.repo.update_entry(filepath, status='error', error_message=f"File read failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to read content from {filepath}: {e}", exc_info=True)
            raise # Re-raise to be caught by _process_file
        except FileNotFoundError:
            logger.error(f"File not found during read: {filepath}", exc_info=True)
            return None
        except IsADirectoryError:
            logger.error(f"Expected file but found directory: {filepath}", exc_info=True)
            return None
        except zstd.ZstdError as e:
            logger.error(f"Zstandard decompression failed for {filepath}: {e}", exc_info=True)
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decoding failed for {filepath}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}", exc_info=True)
            return None
        
    def _process_file(self, source_filepath: Path):
        """Processes a single file: read, process, save compressed."""
        # (Implementation remains the same - calls _save_processed which handles compression)
        logger.debug(f"Processing file: {source_filepath}")
        save_path = None
        try:
            self.repo.update_entry(source_filepath, status='processing', error_message='', processed_path='', tokenized_path='')
            ext = source_filepath.suffix.lower().strip('.')
            processed_data = None
            # Use configured text cleaning regex
            text_cleaning_regex = getattr(DataProcessingConfig, 'TEXT_CLEANING_REGEX', r'[^\w\s]')

            if ext in ['txt', 'csv', 'jsonl', 'md', 'py', 'log']:
                processed_data = self._process_text(source_filepath)
            elif ext == 'json': # Simplified numerical check
                 processed_data = self._process_numerical(source_filepath)
            elif ext == 'pdf':
                import fitz
                # Check if PyMuPDF is available
                PyMuPDF_AVAILABLE = 'fitz' in sys.modules
                if PyMuPDF_AVAILABLE:
                    processed_data = self._process_pdf(source_filepath)
                else:
                    logger.warning(f"Skipping PDF {source_filepath.name}: PyMuPDF (fitz) not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='PyMuPDF library not found')
                    return None
            elif ext == 'docx':
                import docx
                # Check if python-docx is available
                Docx_AVAILABLE = 'docx' in sys.modules
                if Docx_AVAILABLE:
                    processed_data = self._process_docx(source_filepath)
                else:
                    logger.warning(f"Skipping DOCX {source_filepath.name}: python-docx not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='python-docx library not found')
                    return None
            elif ext == 'odt':
                import odfpy
                # Check if odfpy is available
                Odfpy_AVAILABLE = 'odfpy' in sys.modules
                if Odfpy_AVAILABLE:
                    processed_data = self._process_odt(source_filepath)
                else:
                    logger.warning(f"Skipping ODT {source_filepath.name}: odfpy not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='odfpy library not found')
                    return None
            elif ext in ['xls', 'xlsx']:
                # Assumes pandas and optional engines (openpyxl, xlrd) are available
                processed_data = self._process_excel(source_filepath)
            elif ext in ['html', 'htm', 'xml']:
                import bs4
                # Check if BeautifulSoup4 is available
                BS4_AVAILABLE = 'bs4' in sys.modules
                if BS4_AVAILABLE:
                    processed_data = self._process_html_xml(source_filepath)
                else:
                    logger.warning(f"Skipping HTML/XML {source_filepath.name}: BeautifulSoup4 not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='BeautifulSoup4 library not found')
                    return None
            elif ext == 'rtf':
                import striprtf
                # Check if striprtf is available
                StripRTF_AVAILABLE = 'striprtf' in sys.modules
                if StripRTF_AVAILABLE:
                    processed_data = self._process_rtf(source_filepath)
                else:
                    logger.warning(f"Skipping RTF {source_filepath.name}: striprtf not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='striprtf library not found')
                    return None
            elif ext == 'epub':
                import ebooklib
                # Check if EbookLib is available
                EbookLib_AVAILABLE = 'ebooklib' in sys.modules
                if EbookLib_AVAILABLE:
                    processed_data = self._process_epub(source_filepath)
                else:
                    logger.warning(f"Skipping EPUB {source_filepath.name}: EbookLib not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='EbookLib library not found')
                    return None
            elif ext == 'ipynb':
                import nbformat
                import textract
                # Check if nbformat is available
                Nbformat_AVAILABLE = 'nbformat' in sys.modules
                if Nbformat_AVAILABLE:
                    processed_data = self._process_ipynb(source_filepath)
                else:
                    logger.warning(f"Skipping IPYNB {source_filepath.name}: nbformat not installed.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='nbformat library not found')
                    return None
            elif ext == 'doc':
                # Check if textract is available (requires external tools like antiword/catdoc)
                TEXTRACT_AVAILABLE = False
                try:
                    TEXTRACT_AVAILABLE = True
                except ImportError:
                    pass # Keep TEXTRACT_AVAILABLE False

                if TEXTRACT_AVAILABLE:
                    processed_data = self._process_doc(source_filepath)
                else:
                    logger.warning(f"Skipping DOC {source_filepath.name}: textract library not installed or functional.")
                    self.repo.update_entry(source_filepath, status='skipped', error_message='textract library not found')
                    return None
            else:
                logger.warning(f"No specific processing logic defined for file type .{ext}. Skipping {source_filepath.name}")
                self.repo.update_entry(source_filepath, status='skipped', error_message=f'No processor for .{ext}')
                return None

            if processed_data is None:
                logger.error(f"Processing sub-step returned None for {source_filepath.name}. Status likely updated to 'error'.")
                return None

            # Save using the modified save function (handles compression)
            save_path = self._save_processed(processed_data, source_filepath)
            if save_path is None: raise IOError("Failed to save processed data.")

            self.repo.update_entry(source_filepath, status='processed', processed_path=save_path, error_message='')
            logger.info(f"Successfully processed: {source_filepath.name} -> {save_path.name}")
            return source_filepath
        except Exception as e:
            logger.error(f"Processing failed for {source_filepath}: {str(e)}", exc_info=True)
            self.repo.update_entry(source_filepath, status='error', error_message=str(e))
            if save_path and save_path.exists():
                 try: save_path.unlink()
                 except OSError: pass
            return None


    def _ensure_repo_exists(self, filepath: Path, header: List[str]):
        """Creates the repository file with a header if it doesn't exist."""
        if not filepath.exists():
            logger.info(f"Repository file not found at '{filepath}'. Initializing...")
            try:
                # Write header to a new compressed file
                def header_gen():
                    yield ','.join(header) # CSV header line

                stream_compress_lines(str(filepath), header_gen())
                logger.info(f"Initialized repository file: {filepath}")
            except Exception as e:
                logger.critical(f"Failed to initialize repository file '{filepath}': {e}", exc_info=True)
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
                     logger.warning(f"Skipping row {i+1}: Invalid designation number in row: {row}")
                     continue # Skip rows with invalid numbers
            return max_designation + 1
        except FileNotFoundError:
            logger.warning(f"Repository file '{self.repo_filepath}' not found while getting next designation. Starting from 1.")
            return 1
        except Exception as e:
            logger.error(f"Error reading repository to find next designation: {e}", exc_info=True)
            # Fallback or re-raise depending on desired robustness
            logger.warning("Defaulting next designation to 1 due to error.")
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
                        logger.warning(f"Invalid designation '{designation}' found for hash '{data_hash}'")
            return hashes
        except Exception as e:
            logger.error(f"Error loading existing hashes from repository: {e}", exc_info=True)
            return {} # Return empty on error


    def read_repo_stream(self, filepath: Optional[Path] = None) -> Generator[Dict[str, str], None, None]:
        """
        Reads the repository CSV file line by line using streaming decompression.
        Yields each row as a dictionary. Handles potential errors during reading.
        """
        target_filepath = filepath or self.repo_filepath
        if not target_filepath.exists():
            logger.warning(f"Attempted to read non-existent repository: {target_filepath}")
            return # Yield nothing

        header = []
        try:
            line_generator = stream_decompress_lines(str(target_filepath))
            # Read header first
            try:
                header_line = next(line_generator)
                header = [h.strip() for h in header_line.split(',')]
            except StopIteration:
                 logger.warning(f"Repository file is empty: {target_filepath}")
                 return # Empty file

            # Use csv.DictReader on the remaining lines
            # We need to simulate a file-like object for DictReader
            reader = csv.DictReader(line_generator, fieldnames=header, restval=None) # Use header as fieldnames
                                                                                       # restval handles rows with too few fields
            for row in reader:
                 if len(row) != len(header):
                     logger.warning(f"Malformed row in {target_filepath} (expected {len(header)} fields, got {len(row)}): {row}")
                     # Optionally yield a partial dict or skip
                     # yield row # Yields what was parsed
                     continue # Skip malformed row
                 yield row

        except FileNotFoundError:
            logger.error(f"Repository file not found during streaming read: {target_filepath}")
            # Or re-raise depending on desired behavior
        except zstd.ZstdError as e:
            logger.error(f"Zstd decompression error reading {target_filepath}: {e}", exc_info=True)
        except csv.Error as e:
            logger.error(f"CSV parsing error reading {target_filepath}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error reading repository stream {target_filepath}: {e}", exc_info=True)


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
            logger.info(f"Appended rows to repository: {filepath}")

        except Exception as e:
            logger.error(f"Failed to append rows to repository '{filepath}': {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except OSError:
                    pass
            raise # Re-raise the exception

    # --- Methods for Rule 7.1 (Data Manipulation Submenu) ---

    def add_folder(self, folder_path: str):
        """
        Scans a folder recursively, identifies supported files, calculates metadata,
        and adds new files to the main repository using streaming append.
        Rule: 7.1.A (and initial loading)
        """
        abs_folder_path = Path(folder_path).resolve()
        if not abs_folder_path.is_dir():
            logger.error(f"Folder not found or is not a directory: {abs_folder_path}")
            return

        logger.info(f"Scanning folder: {abs_folder_path}")
        new_files_to_add = []
        processed_count = 0
        added_count = 0
        skipped_count = 0
        error_count = 0

        # Track file paths already in the repo to avoid adding duplicates from the same path
        # Note: This check is only for *filepath*. Duplicate *content* is checked via hash later.
        existing_paths = {row.get(COL_FILEPATH) for row in self.read_repo_stream() if row.get(COL_FILEPATH)}

        for item in abs_folder_path.rglob('*'): # Recursive glob
            if item.is_file():
                processed_count += 1
                file_path_str = str(item)
                file_ext = item.suffix.lower()

                if file_path_str in existing_paths:
                    logger.debug(f"Skipping already tracked file: {file_path_str}")
                    skipped_count += 1
                    continue

                # Basic check for supported extension
                # More robust check might involve magic numbers if needed
                if file_ext not in ACCEPTED_FILE_TYPES:
                    logger.debug(f"Skipping unsupported file type '{file_ext}': {file_path_str}")
                    skipped_count += 1
                    continue

                # Check for archive files that need extraction (Basic Example)
                # This needs proper implementation based on how you handle archives
                if file_ext in ['.zip', '.zst', '.zstd']:
                     logger.warning(f"Archive file found: {file_path_str}. Extraction logic not fully implemented.")
                     # TODO: Implement extraction and adding contained files recursively?
                     # For now, we can add the archive itself or skip. Let's add it.
                     is_compressed_original = 'Y' # The archive itself is compressed
                     # continue # Or skip archives if not handling them yet

                else:
                     # Check if the file *itself* is compressed using magic bytes (optional but good)
                     # For simplicity, we assume files are not compressed unless they are known archives
                     is_compressed_original = 'N'

                try:
                    # Get file metadata
                    stat = item.stat()
                    mod_time = datetime.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    acc_time = datetime.datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat()

                    # Generate hashes (can be time-consuming for large files)
                    hashed_path = hash_filepath(file_path_str)
                    data_hash = generate_data_hash(file_path_str) # Content hash

                    # Check for copy using data hash
                    is_copy = 'N'
                    if data_hash and data_hash in self._data_hashes:
                        is_copy = 'Y'
                        logger.info(f"Detected copy (Data Hash: {data_hash[:8]}...) for: {file_path_str} - Original Designation: {self._data_hashes[data_hash]}")
                    elif data_hash:
                         # Add new hash to our in-memory dict for this session
                         self._data_hashes[data_hash] = self._next_designation + len(new_files_to_add)


                    if not hashed_path or not data_hash:
                         logger.error(f"Failed to generate required hashes for: {file_path_str}. Skipping.")
                         error_count +=1
                         continue

                    # Prepare row data
                    row_data = {
                        COL_DESIGNATION: self._next_designation + len(new_files_to_add),
                        COL_FILETYPE: file_ext,
                        COL_FILEPATH: file_path_str,
                        COL_HASHED_PATH_ID: hashed_path,
                        COL_COMPRESSED_FLAG: is_compressed_original,
                        COL_MOD_DATE: mod_time,
                        COL_ACC_DATE: acc_time,
                        COL_DATA_HASH: data_hash,
                        COL_IS_COPY_FLAG: is_copy,
                        COL_STATUS: STATUS_LOADED # Initial status
                    }
                    new_files_to_add.append(row_data)
                    added_count += 1
                    existing_paths.add(file_path_str) # Track path as added in this run
                    logger.debug(f"Prepared to add file: {file_path_str}")

                except OSError as e:
                    logger.error(f"OS Error accessing file metadata for {file_path_str}: {e}")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing file {file_path_str}: {e}", exc_info=True)
                    error_count += 1

        # Append all new files found in this run
        if new_files_to_add:
            logger.info(f"Found {len(new_files_to_add)} new files to add to the repository.")
            try:
                def row_generator():
                    for row in new_files_to_add:
                        yield row
                self._append_repo_stream(self.repo_filepath, row_generator(), MAIN_REPO_HEADER)
                # Update the next designation number *after* successful append
                self._next_designation += len(new_files_to_add)
            except Exception as e:
                logger.error(f"Failed to append batch of new files to repository: {e}", exc_info=True)
                # Note: _next_designation and _data_hashes might be inconsistent if append fails partially
        else:
            logger.info("No new files found in the specified folder.")

        logger.info(f"Folder scan complete. Processed: {processed_count}, Added: {added_count}, Skipped: {skipped_count}, Errors: {error_count}")


    def remove_folder(self, folder_path: str):
        """
        Removes all entries from the main repository whose Filepath
        starts with the specified folder path.
        Rule: 7.1.B
        """
        abs_folder_path = str(Path(folder_path).resolve())
        logger.warning(f"Removing entries for folder and subfolders: {abs_folder_path}")

        rows_to_keep = []
        removed_count = 0

        try:
            # Read the repo, keeping only rows that DON'T match the path prefix
            for row in self.read_repo_stream():
                 file_path = row.get(COL_FILEPATH)
                 if file_path and file_path.startswith(abs_folder_path):
                      logger.debug(f"Marking for removal: {file_path} (Designation: {row.get(COL_DESIGNATION)})")
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

                logger.info(f"Removed {removed_count} entries associated with folder: {abs_folder_path}")
                # Reload internal state as designations/hashes might have changed if we renumbered
                self._next_designation = self._get_next_designation()
                self._data_hashes = self._load_existing_hashes()
            else:
                logger.info(f"No entries found for folder: {abs_folder_path}")

        except Exception as e:
            logger.error(f"Failed to remove folder entries from repository: {e}", exc_info=True)
            # State might be inconsistent, consider recovery or warning user


    def update_status(self, designation: int, new_status: str, target_repo_path: Optional[Path] = None):
        """
        Updates the status for a specific designation in the specified repository file.
        Uses the inefficient read-modify-write approach for zstd files.
        """
        repo_path = target_repo_path or self.repo_filepath
        if not repo_path.exists():
             logger.error(f"Cannot update status. Repository file not found: {repo_path}")
             return False

        logger.debug(f"Attempting to update status to '{new_status}' for designation {designation} in {repo_path.name}")
        updated = False
        rows_to_write = []

        try:
            for row in self.read_repo_stream(repo_path):
                current_designation_str = row.get(COL_DESIGNATION)
                try:
                    if current_designation_str and int(current_designation_str) == designation:
                        if row.get(COL_STATUS) != new_status:
                            logger.info(f"Updating status for Designation {designation} from '{row.get(COL_STATUS)}' to '{new_status}' in {repo_path.name}")
                            row[COL_STATUS] = new_status
                            updated = True
                        else:
                             logger.debug(f"Status for Designation {designation} is already '{new_status}' in {repo_path.name}")
                             # Keep row as is, no change needed
                    # Keep all rows (modified or not) to rewrite the file
                    rows_to_write.append(row)

                except (ValueError, TypeError):
                     logger.warning(f"Skipping row due to invalid designation: {row}")
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
                logger.debug(f"Successfully updated status for designation {designation} in {repo_path.name}")
                return True
            else:
                 logger.debug(f"No status update needed or designation {designation} not found in {repo_path.name}")
                 return False # Return False if no update occurred

        except Exception as e:
            logger.error(f"Failed to update status for designation {designation} in {repo_path.name}: {e}", exc_info=True)
            return False


    def process_data_list(self):
        """
        Iterates through files marked as 'Loaded' (L) in the main repository,
        processes their content, saves output to a mirrored structure,
        and updates status to 'Processed' (P).
        Rule: 7.1.C
        """
        logger.info("Starting data processing pipeline...")
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
            logger.error(f"Failed to read repository to find files for processing: {e}", exc_info=True)
            return # Cannot proceed

        if not files_to_process:
            logger.info("No files found with status 'Loaded' to process.")
            return

        logger.info(f"Found {len(files_to_process)} files to process.")

        for row in files_to_process:
            designation = int(row.get(COL_DESIGNATION, -1))
            filepath_str = row.get(COL_FILEPATH)
            filetype = row.get(COL_FILETYPE)

            if designation == -1 or not filepath_str:
                logger.warning(f"Skipping invalid row during processing: {row}")
                continue

            input_path = Path(filepath_str)
            if not input_path.exists():
                 logger.error(f"File listed in repository not found: {filepath_str}. Setting status to Error.")
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
                 continue

            # --- Create Mirrored Output Path ---
            relative_path = input_path.relative_to(input_path.anchor) # Get path relative to drive root
            output_subpath = output_base / relative_path.parent
            output_subpath.mkdir(parents=True, exist_ok=True)
            output_filename = input_path.stem + PROCESSED_EXT + ".zst" # Add .proc and .zst extension
            output_filepath = output_subpath / output_filename

            logger.info(f"Processing Designation {designation}: {filepath_str} -> {output_filepath}")

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
                         logger.warning(f"Processing logic for file type {filetype} not implemented yet. Skipping content processing.")
                         # yield "" # Yield nothing or handle differently
                         raise NotImplementedError(f"Processing for {filetype} not implemented.")

                    logger.debug(f"Processed {line_count} lines/units from {filepath_str}")


                # Stream compressed output
                stream_compress_lines(str(output_filepath), processed_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_PROCESSED):
                    processed_count += 1
                    logger.info(f"Successfully processed and updated status for Designation {designation}")
                    # TODO: Add entry to processed_repository.csv.zst (similar _append_repo_stream logic)
                else:
                    logger.error(f"Processed Designation {designation} but FAILED to update status in repository.")
                    error_count += 1
                    # Consider cleanup of the generated .proc.zst file?

            except NotImplementedError as e:
                 logger.error(f"Processing failed for Designation {designation} ({filepath_str}): {e}")
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except FileNotFoundError:
                 logger.error(f"Input file disappeared during processing: {filepath_str}. Setting status to Error.")
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                logger.error(f"Error processing Designation {designation} ({filepath_str}): {e}", exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if output_filepath.exists():
                    try: output_filepath.unlink()
                    except OSError: pass

        logger.info(f"Data processing finished. Processed successfully: {processed_count}, Errors: {error_count}")


    def tokenize_processed_data(self):
        """
        Iterates through files marked as 'Processed' (P) in the main repository,
        finds the corresponding '.proc.zst' file, tokenizes its content,
        saves output to a mirrored structure with '.token.zst',
        and updates status to 'Tokenized' (T).
        Rule: 7.1.D
        """
        logger.info("Starting tokenization pipeline...")
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
            logger.error(f"Failed to read repository to find files for tokenization: {e}", exc_info=True)
            return # Cannot proceed

        if not files_to_tokenize:
            logger.info("No files found with status 'Processed' to tokenize.")
            return

        logger.info(f"Found {len(files_to_tokenize)} processed files to tokenize.")

        # Placeholder: Initialize your tokenizer here
        # tokenizer = YourTokenizerClass() or load_tokenizer_function()
        # Example dummy tokenizer function
        def dummy_tokenize(text_line):
            return " ".join(text_line.lower().split()) # Simple split and join

        for row in files_to_tokenize:
            designation = int(row.get(COL_DESIGNATION, -1))
            original_filepath_str = row.get(COL_FILEPATH)

            if designation == -1 or not original_filepath_str:
                logger.warning(f"Skipping invalid row during tokenization: {row}")
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
                logger.error(f"Processed file not found for Designation {designation}: {processed_filepath}. Setting status to Error.")
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                continue

            logger.info(f"Tokenizing Designation {designation}: {processed_filepath} -> {tokenized_filepath}")

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
                    logger.debug(f"Tokenized {line_count} lines/units from {processed_filepath}")

                # Stream compressed output for tokenized data
                stream_compress_lines(str(tokenized_filepath), tokenized_lines_generator())

                # --- Update Status ---
                if self.update_status(designation, STATUS_TOKENIZED):
                    tokenized_count += 1
                    logger.info(f"Successfully tokenized and updated status for Designation {designation}")
                    # TODO: Add entry to tokenized_repository.csv.zst
                else:
                    logger.error(f"Tokenized Designation {designation} but FAILED to update status in repository.")
                    error_count += 1
                    # Consider cleanup?

            except FileNotFoundError: # Should be caught earlier, but just in case
                 logger.error(f"Processed file disappeared during tokenization: {processed_filepath}. Setting status to Error.")
                 self.update_status(designation, STATUS_ERROR)
                 error_count += 1
            except Exception as e:
                logger.error(f"Error tokenizing Designation {designation} ({processed_filepath}): {e}", exc_info=True)
                self.update_status(designation, STATUS_ERROR)
                error_count += 1
                # Clean up potentially corrupted output file
                if tokenized_filepath.exists():
                    try: tokenized_filepath.unlink()
                    except OSError: pass

        logger.info(f"Tokenization finished. Tokenized successfully: {tokenized_count}, Errors: {error_count}")


    # Placeholder for other methods like creating the DataLoader file (Rule 7.1.E)
    def create_dataloader_file(self, output_filename: str = "dataloader_package.zst"):
        """
        Gathers information from the tokenized repository and potentially packages
        tokenized files into a single compressed file for DataLoader usage.
        Rule: 7.1.E
        """
        logger.info("Starting DataLoader file creation...")
        tokenized_files_info = []
        try:
            # Read the main repo to find tokenized files and their original paths
            for row in self.read_repo_stream():
                if row.get(COL_STATUS) == STATUS_TOKENIZED:
                    tokenized_files_info.append(row)
            # Or, preferably, read from a dedicated tokenized_repository.csv.zst if created
        except Exception as e:
            logger.error(f"Failed to read repository to find tokenized files: {e}", exc_info=True)
            return

        if not tokenized_files_info:
             logger.warning("No tokenized files found to create DataLoader package.")
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
                 logger.warning(f"Tokenized file not found for Designation {designation}: {tokenized_full_path}")

        if not metadata:
             logger.warning("No valid tokenized file paths found for metadata.")
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
             logger.info(f"Created DataLoader metadata file: {output_filepath}")
             logger.info(f"Contained metadata for {len(metadata)} tokenized files.")

        except ImportError:
             logger.error("json module not found. Cannot create JSON metadata file.")
        except Exception as e:
             logger.error(f"Failed to create DataLoader metadata file '{output_filepath}': {e}", exc_info=True)
             if output_filepath.exists():
                 try: output_filepath.unlink()
                 except OSError: pass

    def _compress_file(self, filepath: Path) -> Path:
        """Compresses the file using zstandard."""
        logger.debug(f"Compressing file: {filepath.name}")
        try:
            compressed_path = filepath.with_suffix('.zst')
            with open(filepath, 'rb') as f_in:
                with open(compressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdCompressor()
                    dctx.copy_stream(f_in, f_out)
            return compressed_path
        except Exception as e:
            logger.error(f"Failed to compress file {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Compression failed: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found during compression: {filepath}", exc_info=True)
            return None
    
    def _decompress_file(self, filepath: Path) -> Path:
        """Decompresses the file using zstandard."""
        logger.debug(f"Decompressing file: {filepath.name}")
        try:
            decompressed_path = filepath.with_suffix('')
            with open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    dctx = zstd.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
            return decompressed_path
        except Exception as e:
            logger.error(f"Failed to decompress file {filepath}: {e}", exc_info=True)
            self.repo.update_entry(filepath, status='error', error_message=f"Decompression failed: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found during decompression: {filepath}", exc_info=True)
            return None
        
    


    def _process_text(self, filepath: Path, cleaning_regex: str):
        """Processes text data (CPU based)."""
        # (Uses cleaning_regex from config)
        logger.debug(f"Processing text file: {filepath.name}")
        try:
            content = self._read_content(filepath); lines = content.splitlines()
            if not lines: return pd.Series(dtype=str)
            pdf = pd.Series(lines); processed = pdf.str.lower()
            processed = processed.str.replace(cleaning_regex, '', regex=True)
            processed = processed.str.replace(r'\s+', ' ', regex=True).str.strip()
            if NLTK_AVAILABLE and lemmatizer:
                def lemmatize_and_filter(text):
                     words = text.split()
                     lemmatized = [lemmatizer.lemmatize(w) for w in words]
                     filtered = [w for w in lemmatized if w not in stop_words]
                     return " ".join(filtered)
                processed = processed.apply(lemmatize_and_filter)
            return processed.dropna()
        except Exception as e: logger.error(f"Text proc failed for {filepath}: {e}"); self.repo.update_entry(filepath, status='error', error_message=f"{e}"); return None

    def _process_numerical(self, filepath: Path):
        """Processes numerical data, applies scaling."""
        # (Implementation remains the same)
        logger.debug(f"Processing numerical file: {filepath.name}")
        num_array = None
        try: # JSONL
            use_gpu = GPU_AVAILABLE and cudf is not None and not isinstance(cudf, type(cudf_dummy))
            if use_gpu: df = cudf.read_json(filepath, lines=True, dtype=False); num_array = df.to_cupy() if len(df.columns) > 1 else df.iloc[:, 0].to_cupy()
            else: df = pd.read_json(filepath, lines=True, dtype=False); num_array = df.to_numpy() if len(df.columns) > 1 else df.iloc[:, 0].to_numpy()
            if num_array.dtype.kind not in ['f', 'i']: num_array = num_array.astype(np.float32)
        except: num_array = None # Fallback
        if num_array is None: # Full parse
            try:
                data = json.loads(self._read_content(filepath))
                if isinstance(data, list): num_array = np.array(data, dtype=np.float32)
                elif isinstance(data, dict) and 'values' in data: num_array = np.array(data['values'], dtype=np.float32)
                elif isinstance(data, dict) and 'input' in data: num_array = np.array(data['input'], dtype=np.float32)
                else: raise ValueError("JSON structure")
            except Exception as e: logger.error(f"Num proc failed {filepath}: {e}"); self.repo.update_entry(filepath, status='error', error_message=f"{e}"); return None
        try: # Process array
            if num_array is None or num_array.size == 0: return cp.array([]) if GPU_AVAILABLE else np.array([])
            if num_array.dtype.kind not in ['f', 'i']: num_array = num_array.astype(np.float32)
            if GPU_AVAILABLE and cp is not np and isinstance(num_array, np.ndarray): num_array = cp.asarray(num_array)
            processed_array = num_array
            if self.scaler: # Scale
                o_ndim=num_array.ndim; a2d=num_array.reshape(-1,1) if o_ndim==1 else num_array
                try: s2d=self.scaler.fit_transform(a2d); processed_array=s2d.reshape(-1) if o_ndim==1 else s2d
                except Exception as scale_e: logger.error(f"Scaling failed {filepath}: {scale_e}. Using unscaled.", exc_info=True)
            return processed_array
        except Exception as final_e: logger.error(f"Final num proc failed {filepath}: {final_e}"); self.repo.update_entry(filepath, status='error', error_message=f"{final_e}"); return None

    def _save_processed(self, data, original_path: Path) -> Path | None:
        """Saves processed data to a file, compressing if necessary."""
        logger.debug(f"Saving processed data for {original_path.name}")
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
            else: logger.error(f"Unsupported save type: {type(data)}"); return None
            if COMPRESSION_ENABLED: suffix += ".zst"
            save_path = PROCESSED_DATA_DIR / f"{save_stem}_processed{suffix}"
            if save_path.exists():
                logger.warning(f"Processed file already exists: {save_path}. Overwriting.")
                try: save_path.unlink()
                except OSError: logger.error(f"Failed to remove existing processed file: {save_path}", exc_info=True)
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for saving (convert CuPy to NumPy if needed)
            data_to_save = data
            if cp is not np and isinstance(data, cp.ndarray):
                logger.debug("Converting CuPy array to NumPy for saving.")
                data_to_save = cp.asnumpy(data)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                # Ensure pandas data is ready (no specific conversion needed here unless GPU involved earlier)
                pass
            elif not isinstance(data, np.ndarray):
                logger.error(f"Unexpected data type for saving: {type(data)}. Cannot save.")
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

                logger.info(f"Saved processed file: {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"Save failed: {save_path}: {e}", exc_info=True)
                if save_path.exists(): 
                    try: 
                        os.remove(save_path)
                    except OSError: pass
                return None
        except Exception as e:
            logger.error(f"Failed to save processed data for {original_path}: {e}", exc_info=True)
            self.repo.update_entry(original_path, status='error', error_message=f"Failed to save processed data: {e}")
            return None

    def _save_processed(self, processed_data, source_filepath: Path):
        """Saves processed data to a file, compressing if necessary."""
        logger.debug(f"Saving processed data for {source_filepath.name}")

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
        logger.info(f"DataRepository initialized. Repo: {self.repo_path}")

    def _load_repo(self) -> pd.DataFrame:
        # (Load logic similar to previous, ensures UTC for datetimes)
        if self.repo_path.exists():
            logger.info(f"Loading data repository: {self.repo_path}")
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
                            pdf[col] = pd.to_datetime(pdf[col], errors='coerce')
                            # Ensure timezone is UTC or convert if aware but different
                            if pdf[col].dt.tz is not None and pdf[col].dt.tz != timezone.utc:
                                 pdf[col] = pdf[col].dt.tz_convert(timezone.utc)
                            elif pdf[col].dt.tz is None: # Make naive timestamps UTC
                                 pdf[col] = pdf[col].dt.tz_localize(timezone.utc)
                        elif dtype == 'int64': pdf[col] = pd.to_numeric(pdf[col], errors='coerce').fillna(0).astype('int64')
                        elif dtype == 'str': pdf[col] = pdf[col].fillna('').astype(str)
                        else: pdf[col] = pdf[col].astype(dtype) # Try direct conversion
                    except Exception as e:
                        logger.error(f"Error converting repo column '{col}' to '{dtype}': {e}. Keeping string.", exc_info=False)
                        pdf[col] = pdf[col].astype(str).fillna('')
                logger.info(f"Repository loaded ({len(pdf)} entries).")
                return pdf
            except Exception as e: logger.error(f"Repo load failed: {e}. Initializing empty.", exc_info=True)
        else: logger.info("No repo found. Initializing empty.")
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
        logger.debug(f"Attempting to save repository to temporary file: {temp_path}")
        try:
            with self.lock: # Acquire lock
                 pdf = self.df.copy() # Work on a copy within the lock
                 # Ensure string columns don't contain NaN/NaT before saving
                 for col, dtype in self.columns.items():
                     if dtype == 'str' and col in pdf.columns: pdf[col] = pdf[col].fillna('').astype(str)
                     # Handle NaT for datetime if needed (pandas handles it for CSV)
                     if 'datetime' in dtype and col in pdf.columns: pass # pdf[col] = pdf[col].fillna(pd.NaT) # Ensure NaT, not None

                 # Compress and write using configured level
                 cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
                 with open(temp_path, 'wb') as ofh:
                     with cctx.stream_writer(ofh) as compressor:
                         pdf.to_csv(compressor, index=False, encoding='utf-8')

            # Atomic replace (outside lock)
            os.replace(temp_path, self.repo_path)
            logger.info(f"Repository saved successfully ({len(pdf)} entries) to: {self.repo_path}")
        except Exception as e:
            logger.error(f"Repository save failed: {e}", exc_info=True)
            if temp_path.exists():
                try: os.remove(temp_path)
                except OSError: pass
            logger.error(f"Temporary file {temp_path} removed after save failure.")
        finally:
            # Ensure lock is released even if save fails
            if self.lock.locked():
                self.lock.release()
                logger.debug("Repository lock released after save attempt.")

    def scan_and_update(self, base_dir: Path):
        """Recursively scans base_dir, updates repo with new/modified files."""
        # (Scan logic similar, ensures base_dir is stored, uses UTC timestamps)
        if not self.lock: return
        base_dir = base_dir.resolve() # Use absolute path
        logger.info(f"Scanning for files in: {base_dir}")
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
                                 logger.info(f"Modified file detected: {file_path.name}")
                                 needs_processing = True; updated_files += 1
                             else: # Only mtime changed, update repo mtime but don't re-process
                                 files_to_update[abs_file_path_str] = {'last_modified_scan': current_mtime}
                         # else: file is unchanged
                     else: # New file
                         logger.info(f"New file discovered: {file_path.name}")
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
                         logger.info(f"Discovered new file: {file_path.name}")
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
                      logger.warning(f"File disappeared during scan: {file_path}")
                      # Optionally remove from repo if needed: self.remove_entry(file_path)
                      skipped_count += 1
                 except Exception as e:
                      logger.error(f"Error scanning file {file_path}: {e}", exc_info=True)
                      # Add/update with error status? Or just skip? Let's skip for scan.
                      skipped_count += 1
        # Apply updates to the repository (thread-safe via update_entry)
        if files_to_update:
             logger.info(f"Applying {len(files_to_update)} repository updates...")
             # Potential optimization: Batch updates if performance becomes an issue
             for file_path_str, update_data in files_to_update.items():
                 self.update_entry(Path(file_path_str), **update_data)
             self.save() # Save after applying all updates

        logger.info(f"Scan complete. Found: {found}, Processed (New/Updated): {processed}, Skipped: {skipped}. (New: {new_files}, Updated: {updated_files})")
        # Save changes after scan completes
        self.save()
        logger.info(f"Repository updated with {processed_count} new/modified files.")

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
                logger.debug(f"Updating repo entry for: {source_filepath.name}")
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
                            logger.error(f"Failed to assign value '{value}' (type {type(value)}) to column '{col}' (dtype {target_dtype}) at index {idx}: {e}. Assigning as string.")
                            self.df.loc[idx, col] = str(value) # Fallback assign as string
                    else:
                        logger.warning(f"Attempted to update non-existent repo column '{col}'")
            else: # New entry
                logger.debug(f"Adding new repo entry for: {source_filepath.name}")
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
                        logger.error(f"Error ensuring type for new row column '{col}' (target: {target_dtype}): {e}. Converting to string.")
                        new_row_pdf[col] = new_row_pdf[col].astype(str).fillna('')
                        # Ensure target column is also string for concat
                        if self.df[col].dtype != object and self.df[col].dtype != 'string':
                            self.df[col] = self.df[col].astype(str).fillna('')

                # Concatenate new row
                self.df = pd.concat([self.df, new_row_pdf[self.df.columns]], ignore_index=True)
                logger.info(f"New entry (' {self.df} ') added for: {source_filepath.name}")

    def get_status(self, source_filepath: Path) -> str | None:
        """Gets the current status of a file."""
        if not self.lock:
             logger.error("Repository lock not initialized. Cannot get status.")
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
            logger.error(f"Hash calculation failed for {filepath}: {str(e)}", exc_info=False)
            return ''

# --- Tokenizer ---
class Tokenizer:
    """ Loads processed, tokenizes, saves compressed tensors. """
    def __init__(self, max_workers: int | None = None):
        self.repo = DataRepository()
        resolved_max_workers = max_workers if max_workers is not None else DataProcessingConfig.MAX_WORKERS
        self.max_workers = max(1, resolved_max_workers)
        logger.info(f"Initializing Tokenizer with max_workers={self.max_workers}")
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
                     logger.warning(f"Processed path found in repo but file missing: {proc_path}. Skipping tokenization for {src_path.name}.")
                 else:
                     logger.warning(f"No valid processed path found in repo for source: {src_path.name}. Skipping tokenization.")


        if not files_to_tokenize_info:
            logger.info(f"No files matching status {statuses_to_process} [in base_dir: {base_dir_filter}] with existing processed files found to tokenize.")
            return

        logger.info(f"Starting tokenization for {len(files_to_tokenize_info)} files [base_dir: {base_dir_filter}].")
        # (Executor logic remains the same)
        futures = [self.executor.submit(self._tokenize_file, src_path, proc_path)
                   for src_path, proc_path in files_to_tokenize_info]
        with tqdm(total=len(futures), desc=f"Tokenizing Files [{base_dir_filter.name if base_dir_filter else 'All'}]") as pbar:
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: logger.error(f"Error retrieving result from tokenizer future: {e}", exc_info=True)
                finally: pbar.update(1)
        self.repo.save()
        logger.info("File tokenization complete.")


    def _tokenize_file(self, source_filepath: Path, processed_filepath: Path):
        """Loads processed, converts to tensor, saves compressed."""
        # (Implementation remains same - uses modified load/save)
        logger.debug(f"Tokenizing file: {processed_filepath.name}...")
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
            logger.info(f"Tokenized: {processed_filepath.name} -> {save_path.name}")
            return source_filepath
        except Exception as e:
            logger.error(f"Tokenization failed for {processed_filepath.name}: {e}", exc_info=True)
            self.repo.update_entry(source_filepath, status='error', error_message=f"{e}")
            if save_path and save_path.exists():
                try: save_path.unlink()
                except OSError: pass
            return None

    def _load_processed(self, filepath: Path):
        """Loads compressed or uncompressed processed data (CSV or NPY)."""
        logger.debug(f"Loading processed: {filepath.name}")
        if not filepath.exists():
            logger.error(f"Not found: {filepath}")
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
                logger.error(f"Cannot load processed: Unknown suffix '{actual_suffix}' for file {filepath.name}")
                return None

            # Close the buffer if it was used
            if buffer: buffer.close()
            return data
        except Exception as e:
            logger.error(f"Load failed for {filepath}: {e}", exc_info=True)
            # Ensure buffer is closed on error too
            if buffer and not buffer.closed: buffer.close()
            return None

    def _vectorize(self, data) -> torch.Tensor | None:
        """Converts loaded processed data into a tensor."""
        # (Implementation remains same - placeholder for text)
        logger.debug(f"Vectorizing data type: {type(data)}")
        try:
            if isinstance(data, torch.Tensor):
                return data.to(dtype=torch.float32, device=self.device)
            elif isinstance(data, (np.ndarray, cp.ndarray)):
                # Convert CuPy array to NumPy before converting to Torch tensor if necessary
                if cp is not np and isinstance(data, cp.ndarray):
                    logger.debug("Converting CuPy array to NumPy before Torch tensor conversion.")
                    data = cp.asnumpy(data)
                # Ensure data is C-contiguous before creating tensor
                if not data.flags['C_CONTIGUOUS']:
                    logger.debug("Data is not C-contiguous, making a copy.")
                    data = np.ascontiguousarray(data)
                return torch.tensor(data, dtype=torch.float32, device=self.device)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                logger.warning("Vectorizing pandas data - using DUMMY implementation.")
                # Convert pandas data to NumPy first
                numpy_data = data.to_numpy(dtype=np.float32)
                # Ensure data is C-contiguous before creating tensor
                if not numpy_data.flags['C_CONTIGUOUS']:
                     logger.debug("Pandas data (as NumPy) is not C-contiguous, making a copy.")
                     numpy_data = np.ascontiguousarray(numpy_data)
                # Dummy implementation: create zeros based on rows, fixed columns
                num_rows = numpy_data.shape[0]
                # Ensure the dummy tensor matches the expected shape if possible
                # If 1D (Series), create (num_rows, 1). If 2D (DataFrame), use original cols or dummy.
                num_cols = numpy_data.shape[1] if numpy_data.ndim > 1 else 1
                dummy_cols = 10 # Keep the dummy dimension for now
                logger.debug(f"Creating dummy tensor of shape ({num_rows}, {dummy_cols})")
                return torch.zeros((num_rows, dummy_cols), dtype=torch.float32, device=self.device) # Dummy
            else:
                logger.error(f"Unsupported type for vectorization: {type(data)}")
                return None
        except Exception as e:
            logger.error(f"Vectorization failed: {e}", exc_info=True)
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
             logger.warning(f"Processed path '{original_processed_path.name}' did not strictly follow '_processed.npy/csv' pattern. Using stem '{save_stem}'.")


        suffix = ".pt.zst" if COMPRESSION_ENABLED else ".pt"
        save_path = TOKENIZED_DATA_DIR / f"{save_stem}_tokenized{suffix}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving tokenized tensor to: {save_path.name}")
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

            logger.info(f"Saved tokenized tensor: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Save tokenized failed for {save_path}: {e}", exc_info=True)
            if save_path.exists():
                try:
                    save_path.unlink()
                    logger.debug(f"Removed partially saved file: {save_path}")
                except OSError as unlink_e:
                     logger.error(f"Failed to remove partially saved file {save_path}: {unlink_e}")
            return None