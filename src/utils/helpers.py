# src/utils/config.py
"""
Central Configuration File

Consolidates settings, paths, hyperparameters, and other configuration
variables used throughout the application.
"""

import os
from pathlib import Path
import torch
import sys
import hashlib
import time
from datetime import datetime as dt, timezone
from typing import Optional, Dict, Any, Generator, Union
import inspect
from .logger import log_statement
from .hashing import generate_data_hash, hash_filepath
from src.data.constants import *

# --- Project Root ---
# Assumes this file is in project_root/src/utils
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PSUTIL_AVAILABLE = 'psutil' in sys.modules

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
    MODEL_NAME = 'qwen3:1.7b' # Or other suitable model
    TARGET_SAMPLES = 1_000 # Reduced for practicality, original was 1B
    BATCH_SIZE = 50 # Reduced for practicality
    DATA_FORMAT = 'jsonl'
    MAX_WORKERS = 16 # ThreadPoolExecutor workers

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
# (Keep as is, no parallel processing involved here)
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

import zstandard as zstd
from src.data.readers import *
from src.utils.config import *

# --- Helper Function for Saving DataFrame to Compressed Parquet (Needs Integration) ---
# This function is added based on the comment in the user prompt suggesting parquet output
# It should be placed within the DataProcessor class or imported if defined elsewhere.

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

# --- Helper Function: Convert timestamp to ISO UTC String ---
def _ts_to_iso_utc(timestamp: float) -> str:
    """Converts a POSIX timestamp (float) to an ISO 8601 UTC string."""
    try:
        # Create timezone-aware datetime object in UTC
        dt_utc = dt.fromtimestamp(timestamp, tz=timezone.utc)
        # Format as ISO 8601 with microseconds and 'Z' for UTC
        return dt_utc.isoformat(timespec='microseconds') # Appends 'Z' implicitly via offset
    except (ValueError, OSError, TypeError):
        # Handle potential errors with invalid timestamps
        return "" # Return empty string for invalid timestamps

def _get_repo_hash(path_obj: Path):
    # Generates a consistent hash for a directory path.
    log_statement(loglevel='info', logstatement=f"Getting repository hash for {path_obj}:", main_logger=__name__)
    normalized_path_str = str(path_obj.resolve())
    log_statement(loglevel='info', logstatement=f"Generating hash for path {normalized_path_str}", main_logger=__name__)
    return hashlib.sha256(normalized_path_str.encode()).hexdigest()[:16]

def _get_max_workers(config=None):
    """Gets the appropriate number of workers from config or defaults."""
    # (Implementation remains the same)
    if config is None:
        # Attempt to get from app_state if helpers are part of a larger class/context
        # This might need adjustment based on where _get_max_workers is actually used
        app_state_config = {}
        if 'app_state' in globals():
             app_state_config = app_state.get('config', {})
        config = app_state_config

    proc_workers = config.get('data_processing', {}).get('max_workers')
    general_workers = config.get('max_workers')

    if isinstance(proc_workers, int) and proc_workers > 0:
        return proc_workers
    if isinstance(general_workers, int) and general_workers > 0:
        return general_workers

    cpu_count = os.cpu_count();
    return min(cpu_count if cpu_count else 4, 32) # Sensible default with cap

def _generate_file_paths(start_path: Path) -> Generator[Path, None, None]:
    """Generator to yield file paths using os.scandir, handling errors."""
    # (Implementation remains the same)
    try:
        for entry in os.scandir(start_path):
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from _generate_file_paths(Path(entry.path)) # Ensure Path object
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path) # Ensure Path object
            except OSError as e:
                # Log permission errors or other OS issues accessing directory entries
                log_statement(loglevel='warning',
                              logstatement=f"{LOG_INS} - OS Error accessing entry {entry.path}: {e}. Skipping entry.",
                              main_logger=__name__)
            except Exception as e_inner:
                 log_statement(loglevel='error',
                              logstatement=f"{LOG_INS} - Unexpected error processing entry {entry.path}: {e_inner}",
                              main_logger=__name__, exc_info=True)
    except OSError as e_outer:
         log_statement(loglevel='error',
                      logstatement=f"{LOG_INS} - OS Error scanning directory {start_path}: {e_outer}. Check permissions.",
                      main_logger=__name__)
    except Exception as e_scan:
        log_statement(loglevel='error',
                      logstatement=f"{LOG_INS} - Unexpected error scanning directory {start_path}: {e_scan}",
                      main_logger=__name__, exc_info=True)

def get_file_type_from_extension(file_path: Path) -> str:
    """Determines a simple file type based on its extension."""
    # This is a basic implementation. A more robust one might use `python-magic` or a more extensive map.
    ext_to_type = {
        ".txt": "Text",
        ".md": "Markdown",
        ".py": "Python Script",
        ".json": "JSON Data",
        ".csv": "CSV Data",
        ".log": "Log File",
        ".xml": "XML Data",
        ".html": "HTML Document",
        ".css": "CSS Stylesheet",
        ".js": "JavaScript",
        ".jpg": "JPEG Image",
        ".jpeg": "JPEG Image",
        ".png": "PNG Image",
        ".gif": "GIF Image",
        ".pdf": "PDF Document",
        ".doc": "Word Document (Legacy)",
        ".docx": "Word Document",
        ".xls": "Excel Spreadsheet (Legacy)",
        ".xlsx": "Excel Spreadsheet",
        ".ppt": "PowerPoint Presentation (Legacy)",
        ".pptx": "PowerPoint Presentation",
        ".zip": "ZIP Archive",
        ".gz": "GZIP Archive",
        ".tar": "TAR Archive",
        ".exe": "Executable",
        ".dll": "Dynamic Link Library",
        ".iso": "Disk Image",
    }
    extension = file_path.suffix.lower()
    return ext_to_type.get(extension, "Unknown/Binary")

def _read_content_snippet(file_path: Path, is_likely_text: bool) -> str:
    """Reads a snippet of content from the file."""
    snippet = ""
    try:
        with open(file_path, 'rb') as f:
            if is_likely_text:
                lines_read = 0
                temp_snippet_lines = []
                for i, line_bytes in enumerate(f):
                    if i < CONTENT_SNIPPET_LINES:
                        try:
                            temp_snippet_lines.append(line_bytes.decode('utf-8', errors='replace').rstrip('\n\r'))
                        except UnicodeDecodeError:
                            # If decoding fails, switch to binary snippet for this line
                            temp_snippet_lines.append(f"[Binary data line, approx {len(line_bytes)} bytes]")
                            # Or revert to full binary snippet if many lines fail
                    else:
                        break
                snippet = "\n".join(temp_snippet_lines)
                if len(temp_snippet_lines) >= CONTENT_SNIPPET_LINES:
                    snippet += "\n[...more content...]"

            else: # Binary or unknown, read first N bytes
                binary_data = f.read(CONTENT_SNIPPET_BYTES)
                try:
                    # Try to decode as hex for display, or indicate unprintable
                    snippet = binary_data.hex()[:128] + "..." # Show a bit of hex
                    if len(binary_data) < CONTENT_SNIPPET_BYTES:
                         snippet += f" (EOF, {len(binary_data)} bytes total)"
                    else:
                         snippet += f" (first {CONTENT_SNIPPET_BYTES} bytes, hex representation)"
                except Exception:
                    snippet = f"[Binary data, {len(binary_data)} bytes, preview unavailable]"
        log_statement("debug", f"{LOG_INS}:DEBUG>>Read content snippet for {file_path}", __file__, False)
    except Exception as e:
        log_statement("warning", f"{LOG_INS}:WARNING>>Could not read content snippet for {file_path}: {e}", __file__, True)
        snippet = f"[Error reading snippet: {e}]"
    return snippet

def process_file(file_path_str: str) -> Dict[str, Any]:
    """
    Processes a single file to extract metadata and a content snippet.
    This function is intended to be run in parallel.

    Args:
        file_path_str (str): The absolute path to the file as a string.

    Returns:
        Dict[str, Any]: A dictionary containing processing results:
            - 'file_path': The original file path string.
            - 'file_name': Name of the file.
            - 'size_bytes': File size in bytes.
            - 'modification_time_iso': ISO 8601 formatted last modification time (UTC).
            - 'sha256_hash': SHA256 hash of the file content.
            - 'file_type': A simple type description based on extension.
            - 'content_snippet': A small snippet of the file's content.
            - 'error': An error message if processing failed, None otherwise.
    """
    log_statement("debug", f"{LOG_INS}:DEBUG>>Starting processing for file: {file_path_str}", __file__, False)

    file_path = Path(file_path_str)
    result: Dict[str, Any] = {
        "file_path": file_path_str,
        "file_name": file_path.name,
        "size_bytes": None,
        "modification_time_iso": None,
        "sha256_hash": None,
        "file_type": "Unknown",
        "content_snippet": "",
        "error": None,
    }

    try:
        if not file_path.exists():
            result["error"] = "File not found."
            log_statement("warning", f"{LOG_INS}:WARNING>>File not found: {file_path_str}", __file__, False)
            return result
        if not file_path.is_file():
            result["error"] = "Path is not a file."
            log_statement("warning", f"{LOG_INS}:WARNING>>Path is not a file: {file_path_str}", __file__, False)
            return result

        # Get basic file stats
        stat_info = file_path.stat()
        result["size_bytes"] = stat_info.st_size
        # Ensure mtime is in UTC and ISO format
        mtime_utc = Path(file_path).stat().st_mtime
        result["modification_time_iso"] = os.path.getmtime(file_path) #float(stat_info.st_mtime)


        # Calculate SHA256 hash
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192): # Read in chunks
                hasher.update(chunk)
        result["sha256_hash"] = hasher.hexdigest()
        log_statement("debug", f"{LOG_INS}:DEBUG>>Calculated SHA256 hash for {file_path.name}", __file__, False)

        # Determine file type and read snippet
        result["file_type"] = get_file_type_from_extension(file_path)
        is_likely_text = result["file_type"] in ["Text", "Markdown", "Python Script", "JSON Data", "CSV Data", "Log File", "XML Data", "HTML Document", "CSS Stylesheet", "JavaScript"]
        result["content_snippet"] = _read_content_snippet(file_path, is_likely_text)

        log_statement("info", f"{LOG_INS}:INFO>>Successfully processed file: {file_path.name}", __file__, False)

    except PermissionError:
        result["error"] = "Permission denied."
        log_statement("error", f"{LOG_INS}:ERROR>>Permission denied for file: {file_path_str}", __file__, True)
    except IOError as e:
        result["error"] = f"IOError: {e}"
        log_statement("error", f"{LOG_INS}:ERROR>>IOError for file {file_path_str}: {e}", __file__, True)
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        log_statement("critical", f"{LOG_INS}:CRITICAL>>Unexpected error processing file {file_path_str}: {e}", __file__, True)

    return result

def _ensure_pathResolve(object):
    if isinstance(object, str):
        object = Path(object).resolve()
    elif isinstance(object, Path):
        object = object.resolve()
    else:
        err_msg = f"Object '{object}' set as datatype '{type(object)}' instead of str or Path object."
        log_statement('error', f"{LOG_INS}:ERROR>>{err_msg}", Path(__file__).stem, True)
        raise TypeError(err_msg)
    return object

def init_progbar(total: int, desc: str = "") -> Generator:
    """
    Initializes a progress bar for tracking file processing.
    Args:
        total (int): Total number of items to process.
        desc (str): Description for the progress bar.
    Returns:
        Generator: A generator that yields the current progress.
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="file")
    except ImportError:
        log_statement("warning", f"{LOG_INS}:WARNING>>tqdm not available, using simple counter.", __file__, False)
        return range(total)  # Fallback to a simple range if tqdm is not available

def _get_file_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata for a single file, populating all columns defined in
    MAIN_REPO_HEADER with initial values or defaults.
    Timestamps (mtime, ctime, last_updated) are stored as ISO 8601 UTC strings.
    """
    metadata = {}

    try:
        if not filepath.is_file():
            log_statement(loglevel='warning', logstatement=f"{LOG_INS} - Skipping non-file item: {filepath}", main_logger=__file__)
            return None

        stat = filepath.stat()
        file_path_str = str(filepath.resolve())

        # Calculate content hash
        file_hash = generate_data_hash(filepath)
        file_hash_to_store = file_hash if file_hash else ""

        # Calculate path hash
        path_hash = hash_filepath(file_path_str)
        path_hash_to_store = path_hash if path_hash else ""

        # --- Populate core metadata fields ---
        metadata[COL_FILEPATH] = file_path_str
        metadata[COL_FILENAME] = filepath.name
        metadata[COL_SIZE] = stat.st_size # Keep as int
        metadata[COL_MTIME] = _ts_to_iso_utc(stat.st_mtime) # Store ISO string
        metadata[COL_CTIME] = _ts_to_iso_utc(stat.st_ctime) # Store ISO string
        metadata[COL_HASH] = file_hash_to_store # Content hash (string)
        metadata[COL_EXTENSION] = filepath.suffix.lower().lstrip('.') # Extension (string)
        metadata[COL_HASHED_PATH_ID] = path_hash_to_store # Path hash (string)

        # Set initial status based on hashing success
        metadata[COL_STATUS] = STATUS_DISCOVERED if file_hash and path_hash else STATUS_ERROR
        metadata[COL_ERROR] = ""
        if not file_hash: metadata[COL_ERROR] += "Content hash failed. "
        if not path_hash: metadata[COL_ERROR] += "Path hash failed."
        metadata[COL_ERROR] = metadata[COL_ERROR].strip()

        # --- Add defaults for ALL other columns in MAIN_REPO_HEADER ---
        # Get current time once as ISO string for last_updated default
        current_time_iso = _ts_to_iso_utc(time.time())

        for key in MAIN_REPO_HEADER:
            if key not in metadata: # If not already populated
                if key == COL_PROCESSED_PATH: metadata[key] = ""
                elif key == COL_TOKENIZED_PATH: metadata[key] = ""
                elif key == COL_LAST_UPDATED: metadata[key] = current_time_iso # Timestamp of this metadata gathering
                elif key == COL_COMPRESSED_FLAG: metadata[key] = 'N' # Assume not compressed initially
                elif key == COL_IS_COPY_FLAG: metadata[key] = 'N' # Assume not a copy initially
                elif key == COL_DESIGNATION: metadata[key] = pd.NA # Nullable Int - assigned later
                elif key == COL_DATA_CLASSIFICATION: metadata[key] = "" # Determined later
                elif key == COL_FINAL_CLASSIFICATION: metadata[key] = "" # Determined later
                elif key == 'base_dir': metadata[key] = "" # Added during scan/update call
                elif key == COL_ERROR and key not in metadata : metadata[key] = "" # Should be set above if hash failed
                elif key == COL_STATUS and key not in metadata : metadata[key] = STATUS_DISCOVERED # Should be set above
                # Handle potential legacy/alternate timestamp columns if present in header
                elif key == COL_MOD_DATE: metadata[key] = dt.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                elif key == COL_ACC_DATE: metadata[key] = dt.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat()
                elif key == COL_FILETYPE: metadata[key] = metadata[COL_EXTENSION] # Often redundant, use extension
                elif key == COL_DATA_HASH: metadata[key] = file_hash_to_store # Use content hash if this column exists
                # Default for any other unexpected columns
                else: metadata[key] = ""

        log_statement(loglevel='debug', logstatement=f"{LOG_INS} - Successfully gathered metadata for {filepath.name}.", main_logger=__file__)
        return metadata

    except PermissionError as pe:
        log_statement(loglevel='error', logstatement=f"{LOG_INS} - Permission error getting metadata for {filepath.name}: {pe}", main_logger=__file__)
        return None
    except OSError as ose:
        log_statement(loglevel='error', logstatement=f"{LOG_INS} - OS error getting metadata for {filepath.name}: {ose}", main_logger=__file__)
        return None
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"{LOG_INS} - Unexpected error getting metadata for {filepath.name}: {e}", main_logger=__file__)
        return None

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
    # Define ZSTD constants locally if not imported globally
    ZSTD_COMPRESSION_LEVEL = 22 # Or import from config/constants
    ZSTD_THREADS = 0 # Or import from config/constants
    try:
        output_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        encoded_content = content_string.encode(encoding)
        cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
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

# --- Dummy DataLoader (from original helpers) ---
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
            inputs = torch.randn(self.batch_size, self.input_dim, device=self.device)
            targets = torch.randn(self.batch_size, self.num_classes, device=self.device)
            yield inputs, targets

    def __len__(self):
        return self.num_batches

# --- Save and Load State Functions (from original helpers) ---
def save_state(model: torch.nn.Module, filename: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, epoch: Optional[int] = None, **kwargs):
    """ Saves model, optimizer, scheduler state, epoch, and extra metadata. """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        **kwargs # Add any extra metadata
    }
    filepath = CHECKPOINT_DIR / filename # Use global CHECKPOINT_DIR
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, filepath)
        log_statement(loglevel='info', logstatement=f"Checkpoint saved: {filepath}", main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to save checkpoint {filepath}: {e}", main_logger=str(__name__), exc_info=True)


def load_state(model: torch.nn.Module, filename: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, device: Optional[Union[str, torch.device]] = None, strict: bool = True) -> Optional[Dict[str, Any]]:
    """ Loads state from checkpoint, returning metadata. """
    filepath = CHECKPOINT_DIR / filename
    if not filepath.exists():
        log_statement(loglevel='error', logstatement=f"Checkpoint file not found: {filepath}", main_logger=str(__name__))
        return None
    try:
        # Determine map_location based on device argument or model's current device
        map_loc = device or next(model.parameters()).device
        checkpoint = torch.load(filepath, map_location=map_loc)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(map_loc) # Ensure model is on the correct device

        # Load optimizer state if provided and available
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to the correct device (important for Adam/AdamW)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_loc)

        # Load scheduler state if provided and available
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        log_statement(loglevel='info', logstatement=f"Checkpoint loaded successfully from {filepath} to device '{map_loc}'.", main_logger=str(__name__))

        # Return metadata (epoch and any other custom kwargs saved)
        metadata = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}
        return metadata

    except FileNotFoundError: # Should be caught by initial check, but redundant safety
         log_statement(loglevel='error', logstatement=f"Checkpoint file not found during load attempt: {filepath}", main_logger=str(__name__))
         return None
    except Exception as e:
        log_statement(loglevel='error', logstatement=f"Failed to load checkpoint from {filepath}: {e}", main_logger=str(__name__), exc_info=True)
        return None

# --- Dummy Input Function (from original helpers) ---
def dummy_input(batch_size: int = 4, seq_len: int = 10, features: int = 128, device = None):
    """Generates a dummy input tensor for testing."""
    if device is None: device = DEFAULT_DEVICE # Use global default device
    # Example shape, adjust if your models expect something different
    return torch.randn(batch_size, seq_len, features, device=device)


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
#         log_statement(loglevel=str("debug"), logstatement=str("Save and load cycle completed successfully."), main_logger=__file__)
#         # Clean up
#         if self.test_filepath.exists():
#             os.remove(self.test_filepath)
#         log_statement(loglevel=str("debug"), logstatement=str(f"Removed test checkpoint file: {self.test_filepath}"), main_logger=__file__)
#         log_statement(loglevel=str("debug"), logstatement=str("Save and load cycle completed successfully."), main_logger=__file__)
