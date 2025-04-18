# src/data/loaders.py
"""
Data Loading Module

Provides PyTorch DataLoader-like classes for efficiently loading processed
or synthetic datasets, handling Zstandard compressed files if needed,
filtering by base directory, and providing accurate batch count calculation.
"""

import torch
import numpy as np
import zstandard as zstd
import json
import logging
from pathlib import Path
from tqdm import tqdm
import io
from typing import List, Optional

# Import configuration and utility functions
try:
    from ..utils.config import DataLoaderConfig, DEFAULT_DEVICE, COMPRESSION_ENABLED, TOKENIZED_DATA_DIR
    from ..utils.logger import configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.debug("EnhancedDataLoader and SyntheticDataLoader initialized.")
except ImportError:
    try: # Fallback relative
        from ..utils.config import DataLoaderConfig, DEFAULT_DEVICE, COMPRESSION_ENABLED, TOKENIZED_DATA_DIR
        from ..utils.logger import configure_logging
        configure_logging()
        logger = logging.getLogger(__name__)
        logger.debug("EnhancedDataLoader and SyntheticDataLoader initialized.")
    except ImportError: # Fallback dummy
        logging.critical("Failed config import for DataLoader. Using defaults.")
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [loaders.py] - %(message)s')
        logger = logging.getLogger(__name__)
        logger.debug("Dummy config import for DataLoader.")
        class DataLoaderConfig: ENHANCED_LOADER_DATA_DIR = './tokenized'; ENHANCED_LOADER_BATCH_SIZE = 32; ENHANCED_LOADER_FILE_PATTERN = '*.pt.zst'; SYNTHETIC_LOADER_DATA_DIR = './synthetic'; SYNTHETIC_LOADER_BATCH_SIZE = 32; SYNTHETIC_LOADER_FILE_PATTERN = '*.jsonl'
        DEFAULT_DEVICE = 'cpu'; COMPRESSION_ENABLED = True; TOKENIZED_DATA_DIR = Path('./tokenized')

class EnhancedDataLoader:
    """
    Loads pre-processed and tokenized data batches, optimized for training.
    Handles compressed (.pt.zst) or uncompressed (.pt) files.
    Can filter files based on a base directory prefix (requires accurate naming or repo lookup).
    Provides accurate batch count via __len__ (cached).
    """
    def __init__(self,
                 data_dir: Optional[str | Path] = None,
                 batch_size: Optional[int] = None,
                 device: Optional[torch.device | str] = None,
                 file_pattern: Optional[str] = None, # Auto-set based on compression
                 shuffle: bool = True,
                 base_dir_filter: Optional[str | Path] = None): # Filter argument
        """
        Initializes the EnhancedDataLoader.

        Args:
            data_dir (str | Path, optional): Dir containing tokenized data. Defaults config.
            batch_size (int, optional): Samples per batch. Defaults config.
            device (torch.device | str, optional): Device ('cuda', 'cpu'). Defaults config.
            file_pattern (str, optional): DEPRECATED. Determined by config.
            shuffle (bool): Shuffle file order. Defaults to True.
            base_dir_filter (str | Path, optional): If provided, tries to load only files originating
                                                    from this base path. (Needs robust implementation).
        """
        self.data_dir = Path(data_dir or DataLoaderConfig.ENHANCED_LOADER_DATA_DIR)
        self.batch_size = batch_size or DataLoaderConfig.ENHANCED_LOADER_BATCH_SIZE
        self.device = device or DEFAULT_DEVICE
        # Determine file pattern based on config's compression flag
        self.file_pattern = DataLoaderConfig.ENHANCED_LOADER_FILE_PATTERN
        self.shuffle = shuffle
        self.base_dir_filter = Path(base_dir_filter).resolve() if base_dir_filter else None
        self._cached_len: Optional[int] = None # Cache for __len__

        self._validate_data_dir()
        self.files = self._get_files() # Apply filtering here

        logger.info(f"EnhancedDataLoader initialized: Dir='{self.data_dir}', BatchSize={self.batch_size}, Device='{self.device}', Pattern='{self.file_pattern}', Filter='{self.base_dir_filter}', Files={len(self.files)}")

    def _validate_data_dir(self):
        """Checks if the data directory exists."""
        if not self.data_dir.is_dir():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

    def _get_files(self) -> List[Path]:
        """Gets list of files matching pattern, applies base_dir filter if specified."""
        # (Filtering logic requires improvement - see previous notes)
        all_files = sorted(list(self.data_dir.glob(self.file_pattern)))
        if not all_files:
             logger.warning(f"No files found matching pattern '{self.file_pattern}' in '{self.data_dir}'.")
             return []

        if not self.base_dir_filter:
             logger.debug(f"Found {len(all_files)} files. No base_dir filter applied.")
             return all_files
        else:
             # TODO: Implement robust filtering based on DataRepository lookup.
             # Current placeholder returns all files.
             logger.warning(f"Base directory filtering ('{self.base_dir_filter}') requires DataRepository integration for accuracy. Currently returning all found files.")
             filtered_files = all_files
             logger.debug(f"Found {len(all_files)} files, filtered to {len(filtered_files)} (placeholder filter) based on base_dir: {self.base_dir_filter}")
             if not filtered_files:
                  logger.warning(f"No files matched pattern '{self.file_pattern}' AND base_dir filter '{self.base_dir_filter}' (placeholder filter).")
             return filtered_files

    def __iter__(self):
        """ Yields batches, handles compressed files. """
        # Reset cache at the start of iteration if needed (e.g., if files could change between epochs)
        # self.reset_cache() # Optional: uncomment if length needs re-calculating each epoch

        if self.shuffle:
            np.random.shuffle(self.files)

        file_iterator = tqdm(self.files, desc="Loading Batches", unit="file") if len(self.files) > 1 else self.files
        for fpath in file_iterator:
            try:
                # Determine if file is compressed based on pattern used to find it
                is_compressed = COMPRESSION_ENABLED and self.file_pattern.endswith('.zst')

                if is_compressed:
                    dctx = zstd.ZstdDecompressor()
                    with open(fpath, 'rb') as ifh:
                        # Potential optimization: read into buffer incrementally if files are huge
                        decompressed_data = dctx.decompress(ifh.read())
                    buffer = io.BytesIO(decompressed_data)
                    data = torch.load(buffer, map_location=self.device)
                    # logger.debug(f"Loaded compressed tensor from {fpath.name}") # Reduce log noise
                else:
                    data = torch.load(fpath, map_location=self.device)
                    # logger.debug(f"Loaded uncompressed tensor from {fpath.name}") # Reduce log noise

                # --- Batching Logic ---
                if isinstance(data, dict):
                    inputs, targets = data.get('inputs'), data.get('targets')
                    if inputs is None or targets is None:
                        logger.error(f"Dict loaded from {fpath.name} missing keys 'inputs' or 'targets'. Skipping.")
                        continue
                    current_batch_inputs, current_batch_targets = [], []
                    # Use zip for potentially variable length inputs/targets if needed
                    for i_tensor, t_tensor in zip(inputs, targets):
                         current_batch_inputs.append(i_tensor.to(self.device))
                         current_batch_targets.append(t_tensor.to(self.device))
                         if len(current_batch_inputs) >= self.batch_size:
                              yield torch.stack(current_batch_inputs), torch.stack(current_batch_targets)
                              current_batch_inputs, current_batch_targets = [], []
                    if current_batch_inputs: # Yield remainder
                         yield torch.stack(current_batch_inputs), torch.stack(current_batch_targets)
                elif isinstance(data, torch.Tensor):
                    tensor_data = data # Already on self.device
                    for i in range(0, len(tensor_data), self.batch_size):
                        yield tensor_data[i:min(i + self.batch_size, len(tensor_data))]
                else:
                     logger.error(f"Unsupported data type {type(data)} loaded from {fpath.name}. Skipping.")
                     continue

            except FileNotFoundError:
                 logger.error(f"File not found during iteration: {fpath.name}")
                 continue
            except Exception as e:
                logger.error(f"Error loading/processing file {fpath.name}: {str(e)}", exc_info=True)
                continue # Skip file on error

    def __len__(self):
        """
        Calculates the total number of batches across all relevant data files.
        Reads file headers/metadata (if possible) or loads data to determine sample count.
        Caches the result for efficiency.

        Returns:
            int: The total estimated number of batches.
        """
        if self._cached_len is not None:
            return self._cached_len

        logger.info("Calculating total batches for EnhancedDataLoader (might take time)...")
        total_samples = 0
        file_iterator = tqdm(self.files, desc="Calculating Length", unit="file", disable=len(self.files) <= 1)

        for fpath in file_iterator:
            try:
                is_compressed = COMPRESSION_ENABLED and self.file_pattern.endswith('.zst')
                data = None
                if is_compressed:
                    dctx = zstd.ZstdDecompressor()
                    with open(fpath, 'rb') as ifh:
                        # Try loading to CPU just for size check to conserve GPU memory
                        buffer = io.BytesIO(dctx.decompress(ifh.read()))
                        data = torch.load(buffer, map_location='cpu')
                else:
                    data = torch.load(fpath, map_location='cpu')

                # Determine sample count from loaded data
                if isinstance(data, dict):
                    inputs = data.get('inputs')
                    if inputs is not None and hasattr(inputs, '__len__'):
                        total_samples += len(inputs)
                    else: logger.warning(f"Cannot determine sample count from dict in {fpath.name}.")
                elif isinstance(data, torch.Tensor):
                    total_samples += len(data) # Assumes first dimension is samples
                else: logger.warning(f"Unsupported data type {type(data)} in {fpath.name} for length calc.")

            except FileNotFoundError: logger.error(f"File not found during len calc: {fpath.name}")
            except Exception as e: logger.error(f"Error reading {fpath.name} for len calc: {e}", exc_info=True)

        if total_samples == 0: logger.warning("Total sample count is zero. __len__ returns 0.")
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size if self.batch_size > 0 else 0
        self._cached_len = num_batches
        logger.info(f"Length calculation complete. Total samples: {total_samples}, Batches: {num_batches}")
        return self._cached_len

    def reset_cache(self):
        """Resets the cached length calculation."""
        self._cached_len = None
        logger.info("EnhancedDataLoader length cache reset.")

class SyntheticDataLoader:
    """
    Loads batches from synthetic datasets (JSONL format).
    Provides accurate batch count via __len__ (cached).
    """
    def __init__(self,
                 data_dir: Optional[str | Path] = None,
                 batch_size: Optional[int] = None,
                 device: Optional[torch.device | str] = None,
                 file_pattern: Optional[str] = None,
                 shuffle: bool = True):
        """ Initializes the SyntheticDataLoader. """
        self.data_dir = Path(data_dir or DataLoaderConfig.SYNTHETIC_LOADER_DATA_DIR)
        self.batch_size = batch_size or DataLoaderConfig.SYNTHETIC_LOADER_BATCH_SIZE
        self.device = device or DEFAULT_DEVICE
        self.file_pattern = file_pattern or DataLoaderConfig.SYNTHETIC_LOADER_FILE_PATTERN
        self.shuffle = shuffle
        self._cached_len: Optional[int] = None # Cache for __len__

        self._validate_data_dir()
        self.files = self._get_files()
        logger.info(f"SyntheticDataLoader initialized: Dir='{self.data_dir}', BS={self.batch_size}, Dev='{self.device}', Files={len(self.files)}")

    def _validate_data_dir(self):
        """ Checks if the data directory exists. """
        if not self.data_dir.is_dir():
            logger.error(f"Synthetic data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Synthetic data directory {self.data_dir} not found")

    def _get_files(self) -> List[Path]:
        """ Gets the list of files matching the pattern. """
        files = sorted(list(self.data_dir.glob(self.file_pattern)))
        if not files:
            logger.warning(f"No synthetic files found matching pattern '{self.file_pattern}' in directory '{self.data_dir}'.")
        return files

    def __iter__(self):
        """ Yields batches parsed from synthetic JSONL files. """
        # (Iterator logic remains the same as previous version)
        # self.reset_cache() # Optional: Reset cache at start of iteration
        if self.shuffle: np.random.shuffle(self.files)
        file_iterator=tqdm(self.files, desc="Loading Synth Batches", unit="file") if len(self.files)>1 else self.files
        batch_inputs, batch_targets = [], []
        for fpath in file_iterator:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            if not line.strip(): continue
                            sample = json.loads(line)
                            input_tensor = torch.tensor(sample['input'], dtype=torch.float32, device=self.device)
                            target_tensor = torch.tensor(sample['target'], dtype=torch.float32, device=self.device)
                            batch_inputs.append(input_tensor); batch_targets.append(target_tensor)
                            if len(batch_inputs) >= self.batch_size: yield self._format_batch(batch_inputs, batch_targets); batch_inputs, batch_targets = [], []
                        except (json.JSONDecodeError, KeyError) as e: logger.error(f"Error line {line_num+1} in {fpath.name}: {e}")
                        except Exception as e_inner: logger.error(f"Error proc line {line_num+1} in {fpath.name}: {e_inner}", exc_info=True)
            except FileNotFoundError: logger.error(f"File not found during iter: {fpath.name}")
            except Exception as e_outer: logger.error(f"Error reading {fpath.name}: {e_outer}", exc_info=True)
        if batch_inputs: yield self._format_batch(batch_inputs, batch_targets)

    def _format_batch(self, inputs: list, targets: list):
        """ Stacks lists of tensors into batch tensors. """
        # (Implementation remains the same as previous version)
        try: return torch.stack(inputs), torch.stack(targets)
        except Exception as e: logger.error(f"Failed batch stack: {e}"); return torch.tensor([]), torch.tensor([])

    def __len__(self):
        """ Calculates total batches by counting lines in JSONL files (cached). """
        if self._cached_len is not None:
            return self._cached_len

        logger.info("Calculating total batches for SyntheticDataLoader (counting lines)...")
        total_samples = 0
        file_iterator = tqdm(self.files, desc="Calculating Synth Length", unit="file", disable=len(self.files) <= 1)
        for fpath in file_iterator:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f if line.strip()) # Count non-empty lines
                total_samples += line_count
            except FileNotFoundError: logger.error(f"File not found during len calc: {fpath.name}")
            except Exception as e: logger.error(f"Error reading {fpath.name} for len calc: {e}", exc_info=True)

        if total_samples == 0: logger.warning("Total synthetic sample count is zero. __len__ returns 0.")
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size if self.batch_size > 0 else 0
        self._cached_len = num_batches
        logger.info(f"Length calculation complete. Total synthetic samples: {total_samples}, Batches: {num_batches}")
        return self._cached_len

    def reset_cache(self):
        """ Resets the cached length calculation. """
        self._cached_len = None
        logger.info("SyntheticDataLoader length cache reset.")

def hash_folder_path(folder_path: str) -> str:
    """Generates a hash for the given folder path."""
    return hashlib.md5(folder_path.encode()).hexdigest()

def set_data_directory():
    """Sets the data directory and generates a repository file."""
    folder_path = input("Enter the folder path containing your data: ").strip()
    if not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
        return

    folder_hash = hash_folder_path(folder_path)
    repository_file = f"data_repository_{folder_hash}.csv"

    print(f"Scanning folder: {folder_path}")
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    print(f"Found {len(all_files)} files. Generating repository...")
    with open(repository_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Path", "Size (Bytes)", "Last Modified", "Created", "Extension"])
        for file_path in tqdm(all_files, desc="Processing Files"):
            try:
                stats = os.stat(file_path)
                writer.writerow([
                    file_path,
                    stats.st_size,
                    stats.st_mtime,
                    stats.st_ctime,
                    os.path.splitext(file_path)[1]
                ])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Repository file created: {repository_file}")

def process_linguistic_data():
    """Processes linguistic data (lemmatization, stopword removal, etc.)."""
    print("Processing linguistic data...")
    # Placeholder for actual processing logic
    print("Linguistic data processing complete.")

def tokenize_data():
    """Tokenizes the processed data."""
    print("Tokenizing data...")
    # Placeholder for actual tokenization logic
    print("Data tokenization complete.")

def train_on_tokenized_files():
    """Trains on the tokenized data."""
    print("Training on tokenized files...")
    # Placeholder for actual training logic
    print("Training complete.")

def load_saved_model_files():
    """Loads saved model files."""
    print("Loading saved model files...")
    # Placeholder for actual model loading logic
    print("Model loading complete.")

def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\nMain Menu:")
        print("1) Set Data Directory")
        print("2) Process Linguistic Data")
        print("3) Data Tokenization")
        print("4) Train On Tokenized Files")
        print("5) Load Saved Model File(s)")
        print("6) Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            set_data_directory()
        elif choice == "2":
            process_linguistic_data()
        elif choice == "3":
            tokenize_data()
        elif choice == "4":
            train_on_tokenized_files()
        elif choice == "5":
            load_saved_model_files()
        elif choice == "6":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
