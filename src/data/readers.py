# src/data/readers.py
"""
File Reading Utilities

Provides classes and functions to read various file formats (CSV, JSON, TXT, PDF, etc.)
into pandas DataFrames. Includes a function to recursively scan a directory and
attempt to read supported files.
"""

import os
import pandas as pd
from pathlib import Path
import logging
import json # Needed for JSONReader if reading standard JSON
import zstandard # Although not used in readers, it was imported in original
from io import StringIO # Although not used in readers, it was imported in original

# Import optional dependencies and check availability
try:
    import cudf
    CUDF_AVAILABLE = True
    logging.debug("cuDF found. GPU DataFrame operations will be available in readers.")
except ImportError:
    CUDF_AVAILABLE = False
    logging.debug("cuDF not found. GPU DataFrame operations will not be available in readers.")
try:
    from ..utils.logger import configure_logging, log_statement
    configure_logging()
    logging.debug("Logger configured successfully.")
    logger = logging.getLogger(__name__)
except ImportError:
    logging.warning("Logger configuration failed. Using default logging settings.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [readers.py] - %(message)s')
    logger = logging.getLogger(__name__)
try:
    import jsonlines
    JSONLINES_AVAILABLE = True
    log_statement(loglevel=str("debug"), logstatement=str("jsonlines library found. JSONLReader will be available."), main_logger=str(__name__))
except ImportError:
    JSONLINES_AVAILABLE = False
    log_statement(loglevel=str("warning"), logstatement=str("jsonlines library not found. JSONLReader will not be available."), main_logger=str(__name__))

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
    log_statement(loglevel=str("debug"), logstatement=str("pdfminer.six (fitz) library found. PDFReader will be available."), main_logger=str(__name__))
except ImportError:
    PDFMINER_AVAILABLE = False
    log_statement(loglevel=str("warning"), logstatement=str("pdfminer.six library not found. PDFReader will not be available."), main_logger=str(__name__))

# Import configuration for paths if needed, or pass paths as arguments
from ..utils.config import DATA_DIR, RAW_DATA_DIR


class FileReader:
    """Base class for file readers."""
    def __init__(self, filepath: str | Path):
        """
        Initializes the FileReader.

        Args:
            filepath (str | Path): The full path to the file to be read.

        Raises:
            FileNotFoundError: If the filepath does not exist.
            TypeError: If filepath is not a string or Path object.
        """
        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        elif isinstance(filepath, Path):
            self.filepath = filepath
        else:
            raise TypeError(f"filepath must be a string or Path object, not {type(filepath)}")

        if not self.filepath.is_file():
            raise FileNotFoundError(f"File not found at path: {self.filepath}")

        self.filename = self.filepath.name
        log_statement(loglevel=str("debug"), logstatement=str(f"FileReader initialized for: {self.filepath}"), main_logger=str(__name__))

    def read(self) -> pd.DataFrame:
        """
        Reads the file content into a pandas DataFrame.
        Must be implemented by subclasses.

        Returns:
            pd.DataFrame: The content of the file as a DataFrame.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'read' method.")

# --- Concrete Reader Implementations ---

class CSVReader(FileReader):
    """Reads CSV files."""
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments passed directly to pd.read_csv.

        Returns:
            pd.DataFrame: DataFrame containing the CSV data.
        """
        log_statement(loglevel=str("info"), logstatement=str(f"Reading CSV file: {self.filepath}"), main_logger=str(__name__))
        try:
            # Example: Add encoding detection or specific defaults if needed
            # kwargs.setdefault('encoding', 'utf-8')
            return pd.read_csv(self.filepath, **kwargs)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading CSV file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise # Re-raise the exception

class JSONReader(FileReader):
    """Reads JSON files (standard JSON, not JSON Lines)."""
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Reads a JSON file into a pandas DataFrame.
        Assumes the JSON structure can be directly converted (e.g., list of objects).

        Args:
            **kwargs: Additional keyword arguments passed directly to pd.read_json.

        Returns:
            pd.DataFrame: DataFrame containing the JSON data.
        """
        log_statement(loglevel=str("info"), logstatement=str(f"Reading JSON file: {self.filepath}"), main_logger=str(__name__))
        try:
            # Common orientation is 'records' for list of JSON objects
            # kwargs.setdefault('orient', 'records')
            # kwargs.setdefault('lines', False) # Ensure it reads as standard JSON
            return pd.read_json(self.filepath, **kwargs)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading JSON file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise

class JSONLReader(FileReader):
    """Reads JSON Lines (JSONL) files."""
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Reads a JSON Lines file (one JSON object per line) into a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments (currently unused by jsonlines reader).

        Returns:
            pd.DataFrame: DataFrame containing the JSONL data.

        Raises:
            ImportError: If the jsonlines library is not installed.
            Exception: For errors during file reading or parsing.
        """
        if not JSONLINES_AVAILABLE:
            raise ImportError("The 'jsonlines' library is required to read JSONL files. Please install it.")

        log_statement(loglevel=str("info"), logstatement=str(f"Reading JSONL file: {self.filepath}"), main_logger=str(__name__))
        try:
            records = []
            with jsonlines.open(self.filepath, mode='r') as reader:
                for obj in reader:
                    records.append(obj)
            if not records:
                log_statement(loglevel=str("warning"), logstatement=str(f"JSONL file {self.filepath} is empty or contains no valid objects."), main_logger=str(__name__))
                return pd.DataFrame()
            return pd.DataFrame(records)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading JSONL file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise

class TXTReader(FileReader):
    """Reads plain text files into a single-row DataFrame."""
    def read(self, encoding='utf-8', **kwargs) -> pd.DataFrame:
        """
        Reads the entire content of a text file into a single 'text' column
        in a pandas DataFrame.

        Args:
            encoding (str): The file encoding to use. Defaults to 'utf-8'.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            pd.DataFrame: DataFrame with a single row and column 'text'.
        """
        log_statement(loglevel=str("info"), logstatement=str(f"Reading TXT file: {self.filepath}"), main_logger=str(__name__))
        try:
            with open(self.filepath, 'r', encoding=encoding) as f:
                content = f.read()
            return pd.DataFrame({'text': [content]})
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading TXT file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise

class ExcelReader(FileReader):
    """Reads Excel files (.xlsx, .xls)."""
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Reads an Excel file into a pandas DataFrame.
        By default, reads the first sheet. Use kwargs to specify sheet_name etc.

        Args:
            **kwargs: Additional keyword arguments passed directly to pd.read_excel.

        Returns:
            pd.DataFrame: DataFrame containing the Excel data.
        """
        log_statement(loglevel=str("info"), logstatement=str(f"Reading Excel file: {self.filepath}"), main_logger=str(__name__))
        try:
            # kwargs.setdefault('sheet_name', 0) # Read first sheet by default
            return pd.read_excel(self.filepath, **kwargs)
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading Excel file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise

class PDFReader(FileReader):
    """Reads text content from PDF files."""
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Extracts text content from a PDF file using pdfminer.six and returns it
        in a single 'text' column in a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments passed directly to extract_text
                      (e.g., password, page_numbers, maxpages).

        Returns:
            pd.DataFrame: DataFrame with a single row and column 'text'.

        Raises:
            ImportError: If the pdfminer.six library is not installed.
        """
        if not PDFMINER_AVAILABLE:
            raise ImportError("The 'pdfminer.six' library is required to read PDF files. Please install it.")

        log_statement(loglevel=str("info"), logstatement=str(f"Reading PDF file: {self.filepath}"), main_logger=str(__name__))
        try:
            text = extract_text(self.filepath, **kwargs)
            return pd.DataFrame({'text': [text]})
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading PDF file {self.filepath}: {e}", exc_info=True), main_logger=str(__name__))
            raise

# --- File Discovery Function ---
def get_reader_class(extension: str):
    """Maps file extension to the appropriate reader class."""
    extension = extension.lower().strip('.')
    reader_map = {
        'csv': CSVReader,
        'json': JSONReader,
        'jsonl': JSONLReader if JSONLINES_AVAILABLE else None,
        'txt': TXTReader,
        'xlsx': ExcelReader,
        'xls': ExcelReader,
        'pdf': PDFReader if PDFMINER_AVAILABLE else None
    }
    return reader_map.get(extension)

def open_files_recursively(folder_path: str | Path,
                           repo_file: str | Path = DATA_DIR / 'discovered_files.csv',
                           progress_file: str | Path = DATA_DIR / 'discovery_progress.csv'):
    """
    Recursively scans a folder, attempts to read supported files, and logs results.

    Creates two CSV files:
    - repo_file: Lists all files found, their type, and read status ('Read', 'Failed', 'Unsupported').
    - progress_file: Tracks processing progress (initially 0% for read files).

    Args:
        folder_path (str | Path): The path to the folder to scan.
        repo_file (str | Path, optional): Path to save the file repository CSV.
                                          Defaults to 'project_root/data/discovered_files.csv'.
        progress_file (str | Path, optional): Path to save the progress tracking CSV.
                                              Defaults to 'project_root/data/discovery_progress.csv'.
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    if not folder_path.is_dir():
        log_statement(loglevel=str("error"), logstatement=str(f"Folder not found: {folder_path}"), main_logger=str(__name__))
        return

    log_statement(loglevel=str("info"), logstatement=str(f"Starting recursive file discovery in: {folder_path}"), main_logger=str(__name__))

    repo_data = []
    progress_data = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = Path(root) / filename
            extension = filepath.suffix.lower().strip('.')
            file_info = {
                'filename': filename,
                'filepath': str(filepath.resolve()),
                'filetype': extension,
                'read_status': 'Unsupported', # Default status
                'error_message': ''
            }
            progress_info = None

            ReaderClass = get_reader_class(extension)
            if ReaderClass:
                try:
                    reader = ReaderClass(filepath)
                    # Attempt to read - we don't store the df here, just check if readable
                    df = reader.read()
                    log_statement(loglevel=str("info"), logstatement=str(f"Successfully opened '{filename}' using {ReaderClass.__name__}"), main_logger=str(__name__))
                    file_info['read_status'] = 'Read'
                    progress_info = {'filename': filename, 'processed_percentage': 0}

                except ImportError as ie:
                     log_statement(loglevel=str("warning"), logstatement=str(f"Skipping '{filename}': Required library not installed ({ie})"), main_logger=str(__name__))
                     file_info['read_status'] = 'Skipped (Missing Lib)'
                     file_info['error_message'] = str(ie)
                except FileNotFoundError:
                     # Should not happen if os.walk yields it, but handle defensively
                     log_statement(loglevel=str("error"), logstatement=str(f"File not found during read attempt (should exist): {filepath}"), main_logger=str(__name__))
                     file_info['read_status'] = 'Error (Not Found)'
                     file_info['error_message'] = 'File disappeared during scan'
                except Exception as e:
                    log_statement(loglevel=str("error"), logstatement=str(f"Failed to open '{filename}' using {ReaderClass.__name__}: {e}", exc_info=False), main_logger=str(__name__))
                    file_info['read_status'] = 'Failed'
                    file_info['error_message'] = str(e)
            else:
                 log_statement(loglevel=str("debug"), logstatement=str(f"Unsupported file type: {filename}"), main_logger=str(__name__))
                 file_info['read_status'] = 'Unsupported'


            repo_data.append(file_info)
            if progress_info:
                progress_data.append(progress_info)

    # Create DataFrames and save
    try:
        repo_df = pd.DataFrame(repo_data)
        repo_df.to_csv(repo_file, index=False)
        log_statement(loglevel=str("info"), logstatement=str(f"File discovery report saved to: {repo_file}"), main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Failed to save repository file {repo_file}: {e}", exc_info=True), main_logger=str(__name__))

    try:
        progress_df = pd.DataFrame(progress_data)
        progress_df.to_csv(progress_file, index=False)
        log_statement(loglevel=str("info"), logstatement=str(f"Initial progress file saved to: {progress_file}"), main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Failed to save progress file {progress_file}: {e}", exc_info=True), main_logger=str(__name__))


# Removed the original __main__ block. Use this function by importing it
# and calling it with the desired folder path.
# Example:
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     target_folder = RAW_DATA_DIR # Example: Scan the raw data directory from config
#     open_files_recursively(target_folder)

