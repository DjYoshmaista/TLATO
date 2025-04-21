# src/data/readers.py
"""
File Reading Utilities

Provides classes and functions to read various file formats (CSV, JSON, TXT, PDF, etc.)
into pandas DataFrames. Includes a function to recursively scan a directory and
attempt to read supported files.
"""
import csv
import os
import pandas as pd
from pathlib import Path
import logging
import json # Needed for JSONReader if reading standard JSON
import zstandard # Although not used in readers, it was imported in original
from io import StringIO # Although not used in readers, it was imported in original
from src.data.constants import *
import zstandard as zstd
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
# Use chardet for robust encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
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
from ..utils.config import BASE_DATA_DIR, RAW_DATA_DIR


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
        self.default_encoding = default_encoding
        self.error_handling = error_handling
        # Only try detection if chardet is available
        self.detect_encoding_flag = detect_encoding and HAS_CHARDET
        log_statement(loglevel='debug', logstatement=f"RobustTextReader initialized: default_encoding={default_encoding}, error_handling={error_handling}, detect_encoding={self.detect_encoding_flag}", main_logger=__name__)
        self.filename = self.filepath.name
        log_statement(loglevel=str("debug"), logstatement=str(f"FileReader initialized for: {self.filepath}"), main_logger=str(__name__))

    def _detect_encoding(self, file_path, sample_size=2048):
        """Detect file encoding using chardet if available."""
        if not self.detect_encoding_flag:
            return None
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            if not raw_data: # Handle empty files
                return self.default_encoding # Or None, depending on desired behavior

            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            # Log detection result
            log_statement(loglevel='debug', logstatement=f"Chardet result for {os.path.basename(file_path)}: ", main_logger=__name__)
            log_statement(loglevel='debug', logstatement=f"encoding='{encoding}', confidence={confidence:.2f}", main_logger=__name__)

            # Use detected encoding if confidence is reasonably high
            if encoding and confidence > 0.75:
                # Normalize common cases
                enc_lower = encoding.lower()
                if enc_lower == 'ascii':
                    return 'utf-8' # Treat ASCII as UTF-8 subset
                # Add other normalizations if needed (e.g., windows-1252 -> cp1252)
                return encoding
            else:
                log_statement(loglevel='debug', logstatement=f"Encoding detection confidence low or failed for {os.path.basename(file_path)}. ", main_logger=__name__)
                log_statement(loglevel='debug', logstatement=f"Falling back to default: {self.default_encoding}", main_logger=__name__)
                return None # Indicate fallback needed
        except FileNotFoundError:
            # Already handled by the main read method, but log here too if needed
            # logger.error(f"File not found during encoding detection: {file_path}")
            raise # Re-raise to be caught by caller
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Error detecting encoding for {file_path}: {e}", main_logger=__name__, exc_info=True)
            return None # Fallback to default

    def read(self, file_path):
        """Reads the file content using detected or default encoding."""
        log_statement(loglevel='debug', logstatement=f"Attempting to read: {file_path}", main_logger=__name__)
        read_encoding = self.default_encoding

        try:
            detected = self._detect_encoding(file_path)
            if detected:
                read_encoding = detected

            with open(file_path, 'r', encoding=read_encoding, errors=self.error_handling) as f:
                return f.read()

        except FileNotFoundError:
             log_statement(loglevel='warning', logstatement=f"File not found: {file_path}", main_logger=__name__, exc_info=True)
             return None # Return None for file not found
        except UnicodeDecodeError as ude:
            log_statement(loglevel='warning', logstatement=f"UnicodeDecodeError for {file_path} with encoding '{read_encoding}': {ude}. ", main_logger=__name__)
            log_statement(loglevel='warning', logstatement=f"Error handling is '{self.error_handling}'.", main_logger=__nmae__)
            # If errors='ignore' or 'replace', it should have returned content.
            # This might happen if errors='strict' was somehow used, or detection failed badly.
            # Optionally, try a fallback encoding like 'latin-1' if the first attempt fails.
            if read_encoding != 'latin-1':
                log_statement(loglevel='warning', logstatement=f"Attempting fallback read of {file_path} with 'latin-1'.", main_logger=__name__)
                try:
                    with open(file_path, 'r', encoding='latin-1', errors=self.error_handling) as f:
                        return f.read()
                except Exception as fallback_e:
                    log_statement(loglevel='error', logstatement=f"Fallback read failed for {file_path}: {fallback_e}", main_logger=__name__, exc_info=True)
            return None # Give up if decode error persists even with fallback/handling
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Error reading file {file_path}: {e}", main_logger=__name__, exc_info=True)
            return None # Return None on other errors
    
    def read_text_file(self, file_path: str) -> str | None:
        """Reads a plain text file using the robust reader."""
        return self.read(file_path)

    def read_generic_file(self, file_path: str) -> str | None:
        """Generic reader for unknown but likely text-based files (.log, .py, .chk)."""
        log_statement(loglevel='debug', logstatement=f"Using generic text reader for {file_path}", main_logger=__name__)
        return self.read(file_path)

# --- Concrete Reader Implementations ---

class CSVReader(FileReader):
    """Reads CSV files."""
    def read(self, file_path: str, text_column: str | None = None, **kwargs) -> list[str] | None:
        """
        Reads a CSV file, extracts text from a specific column or all columns.
        Handles potential encoding and dialect issues.

        Args:
            file_path (str): Path to the CSV file.
            text_column (str | None): The name of the column containing the primary text.
                                    If None, concatenates text from all columns for each row.

        Returns:
            list[str] | None: A list of text strings (one per row), or None on failure.
        """
        log_statement(loglevel='debug', log_statement=f"Attempting to read CSV: {file_path}", main_logger=__name__)
        try:
            # Use the robust reader to get content first, handling encoding
            file_content = FileReader.read(file_path)
            if file_content is None: # Handles FileNotFoundError and severe decode errors
                # Error already logged by text_reader
                return None
            if not file_content.strip():
                log_statement(loglevel='warning', logstatement=f"CSV file is empty: {file_path}", main_logger=__name__)
                return [] # Return empty list for empty file

            file_stream = StringIO(file_content)

            # Sniff dialect for better parsing
            try:
                # Read enough data to reliably sniff, but avoid reading huge files entirely
                sniffer = csv.Sniffer()
                # Increase sniff size for more reliability
                dialect = sniffer.sniff(file_stream.read(1024 * 10))
                log_statement(loglevel='debug', logstatement=f"Detected CSV dialect for {os.path.basename(file_path)}: ", main_logger=__name__)
                log_statement(loglevel='debug', logstatement=f"delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'", main_logger=__name__)
            except csv.Error as sniff_err:
                logger.warning(f"Could not detect CSV dialect for {os.path.basename(file_path)} ({sniff_err}). "
                            f"Using default (Excel comma-separated).")
                dialect = 'excel' # Default fallback dialect
            finally:
                file_stream.seek(0) # IMPORTANT: Reset stream position after sniffing/reading

            reader = csv.reader(file_stream, dialect=dialect)
            header = next(reader, None)

            if not header:
                logger.warning(f"CSV file has no header or is empty after header read: {file_path}")
                return [] # Return empty list

            log_statement(loglevel='debug', logstatement=f"CSV header for {os.path.basename(file_path)}: {header}", main_logger=__name__)
            header = [h.strip() for h in header] # Strip whitespace from header names

            data = []
            text_col_index = -1
            if text_column:
                try:
                    text_col_index = header.index(text_column)
                except ValueError:
                    log_statement(loglevel='warning', logstatement=f"Specified text column '{text_column}' not found in header ", main_logger=__name__)
                    log_statement(loglevel='warning', logstatement=f"of {file_path}. Header: {header}. Will concatenate all columns.", main_logger=__name__)
                    text_column = None # Fallback to concatenating all

            for i, row in enumerate(reader):
                # Handle rows with incorrect number of columns
                if len(row) != len(header):
                    log_statement(loglevel='warning', logstatement=f"Skipping malformed row {i+1} in {file_path}. ", main_logger=__name__)
                    log_statement(loglevel='warning', logstatement=f"Expected {len(header)} columns, got {len(row)}. Row: {row[:5]}...", main_logger=__name__) # Log first few items
                    continue

                row_text = ""
                try:
                    if text_column is not None and text_col_index != -1:
                        row_text = row[text_col_index].strip()
                    else:
                        # Concatenate all fields if no specific column or column not found
                        row_text = " ".join(field.strip() for field in row if field) # Join non-empty fields
                except IndexError:
                    log_statement(loglevel='warning', logstatement=f"Index error accessing data for row {i+1} in {file_path}. Row: {row}", main_logger=__name__)
                    continue # Skip row if unexpected index issue occurs
                except Exception as row_e:
                    log_statement(loglevel='error', logstatement=f"Error processing row {i+1} content in {file_path}: {row_e}", main_logger=__name__)
                    continue # Skip problematic row

                if row_text: # Only add if text was extracted
                    data.append(row_text)

            log_statement(lolevel='info', logstatement=f"Read {len(data)} rows with text content from CSV: {file_path}", main_logger=__name__)
            return data

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed to read or parse CSV file {file_path}: {e}", exc_info=True, main_logger=__name__)
            return None

class JSONReader(FileReader):
    """Reads JSON files (standard JSON, not JSON Lines)."""
    def read(self, file_path: str, text_key: str | None = None, **kwargs) -> list[str] | None:
        """
        Reads a JSON file (either a single JSON object or JSON Lines format).
        Extracts text based on a specified key or attempts common keys.

        Args:
            file_path (str): Path to the JSON file.
            text_key (str | None): The key containing the text. If None, tries common keys like 'text', 'content'.

        Returns:
            list[str] | None: A list of extracted text strings, or None on failure.
        """
        log_statement(loglevel='debug', logstatement=f"Attempting to read JSON: {file_path}", main_logger=__name__)
        content = FileReader.read(file_path) # Use robust reader for encoding
        if content is None:
            return None
        if not content.strip():
            log_statement(loglevel='warning', logstatement=f"JSON file is empty: {file_path}", main_logger=__name__)
            return []

        extracted_texts = []
        is_json_lines = False
        try:
            # Try parsing as a single JSON object/array first
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # If single JSON fails, try JSON Lines format
                log_statement(loglevel='debug', logstatement=f"Failed to parse {os.path.basename(file_path)} as single JSON, attempting JSON Lines.", main_logger=__name__)
                file_stream = StringIO(content)
                data = []
                for i, line in enumerate(file_stream):
                    line = line.strip()
                    if not line: continue
                    try:
                        data.append(json.loads(line))
                        is_json_lines = True
                    except json.JSONDecodeError as jsonl_err:
                        log_statement(loglevel='warning', logstatement=f"Skipping invalid JSON line {i+1} in {file_path}: {jsonl_err}. Line: {line[:100]}...", main_logger=__name__)
                        continue # Skip invalid lines in JSONL
                if not is_json_lines:
                    log_statement(loglevel='error', logstatement=f"File {file_path} is not valid single JSON or JSON Lines.", main_logger=__name__)
                    return None # Not valid JSON at all


            # Now process the loaded data (either single object/list or list from JSONL)
            if isinstance(data, list):
                items_to_process = data
            elif isinstance(data, dict):
                items_to_process = [data] # Treat single object as a list with one item
            else:
                log_statement(loglevel='warning', logstatement=f"Unexpected JSON structure in {file_path} (expected list or dict, got {type(data)}). Cannot extract text.", main_logger=__name__)
                return None

            possible_keys = [text_key] if text_key else ['text', 'content', 'body', 'message', 'value'] # Common text keys

            for item in items_to_process:
                if isinstance(item, dict):
                    found_text = None
                    for key in possible_keys:
                        if key in item and isinstance(item[key], str):
                            found_text = item[key].strip()
                            if found_text: break # Use first found non-empty text

                    if found_text:
                        extracted_texts.append(found_text)
                    elif text_key: # Only warn if specific key was requested but not found/valid
                        log_statement(loglevel='warning', logstatement=f"Key '{text_key}' not found or not a string in JSON item: {str(item)[:100]}... in {file_path}", main_logger=__name__)
                    # else: implicitly skip if no known text key found and none specified

                elif isinstance(item, str) and not is_json_lines: # Handle case where JSON is just a list of strings
                    item_text = item.strip()
                    if item_text: extracted_texts.append(item_text)

            log_statement(loglevel='info', logstatement=f"Read {len(extracted_texts)} text entries from JSON: {file_path}", main_logger=__name__)
            return extracted_texts

        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed to read or parse JSON file {file_path}: {e}", exc_info=True, main_logger=__name__)
            return None

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
    def read(self, file_path=None, encoding='utf-8', **kwargs) -> pd.DataFrame:
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
            content = FileReader.read(file_path)
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

class HTMLReader(FileReader):
    def read(self, file_path: str, **kwargs) -> str | None:
        """Reads an HTML file and extracts text content using BeautifulSoup if available."""
        if not HAS_BS4:
            log_statement(loglevel='warning', logstatement=f"Cannot process HTML file {file_path}. BeautifulSoup library not found. ", main_logger=__name__)
            log_statement(loglevel='warning', logstatement="Install with: pip install beautifulsoup4", main_logger=__name__)
            # Fallback: Read As Plaintext and return True as the second returned variable to indicate plaintext is true
            return FileReader.read_generic_file(file_path), True
            
        log_statement(loglevel='debug', logstatement=f"Attempting to read HTML: {file_path}", main_logger=__name__)
        # Use robust reader for initial content loading
        html_content = FileReader.read(file_path)
        if html_content is None: # Handles file not found / initial read errors
            return None
        if not html_content.strip():
            log_statement(loglevel='warning', logstatement=f"HTML file is empty: {file_path}", main_logger=__name__)
            return ""

        try:
            # Use 'html.parser' (built-in) or 'lxml' (faster, needs install)
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements to avoid extracting their content
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]): # Remove common non-content blocks
                element.decompose()

            # Get text, separated by spaces, and strip leading/trailing whitespace
            text = soup.get_text(separator=' ', strip=True)

            # Optional: Further clean the text (e.g., remove excessive whitespace)
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            log_statement(loglevel='info', logstatement=f"Extracted {len(text)} characters of text from HTML: {file_path}", main_logger=__name__)
            return text
        except Exception as e:
            log_statement(loglevel='error', logstatement=f"Failed to parse HTML file {file_path}: {e}", main_logger=__name__, exc_info=True)
            return None

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
        'pdf': PDFReader if PDFMINER_AVAILABLE else None,
        'log': TXTReader,
        'md': TXTReader
    }
    return reader_map.get(extension)

def open_files_recursively(folder_path: str | Path,
                           repo_file: str | Path = BASE_DATA_DIR / 'discovered_files.csv',
                           progress_file: str | Path = BASE_DATA_DIR / 'discovery_progress.csv'):
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

