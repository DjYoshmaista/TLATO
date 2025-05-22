import csv
import os
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Union

import json # Needed for JSONReader if reading standard JSON
import zstandard # Although not used in readers, it was imported in original
from io import StringIO, BytesIO # StringIO needed for CSV/JSON parsing from string
# Import constants using relative path if readers.py is in src/data
from src.data.constants import * # Use .. to go up one level from data to src
# Import logger using relative path
from src.utils.logger import log_statement
LOG_INS = f"{__file__}:{__name__}:"

# Import optional dependencies safely
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import jsonlines
    JSONLINES_AVAILABLE = True
    log_statement('debug', f"{LOG_INS}jsonlines library found. JSONLReader will be available.", Path(__file__).stem)
except ImportError:
    JSONLINES_AVAILABLE = False
    log_statement('warning', f"{LOG_INS}jsonlines library not found. JSONLReader will not be available.", Path(__file__).stem)

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
    log_statement('debug', f"{LOG_INS}pdfminer.six library found. PDFReader will be available.", Path(__file__).stem)
except ImportError:
    PDFMINER_AVAILABLE = False
    log_statement('warning', f"{LOG_INS}pdfminer.six library not found. PDFReader will not be available.", Path(__file__).stem)
from src.data.constants import *
from src.utils.config import *

# --- Base FileReader ---
class FileReader:
    """Base class for file readers. Handles basic path validation."""
    def __init__(self, filepath: Union[str, Path]):
        """
        Initializes the FileReader.

        Args:
            filepath (str | Path): The full path to the file to be read.

        Raises:
            FileNotFoundError: If the filepath does not exist or is not a file.
            TypeError: If filepath is not a string or Path object.
        """
        global LOG_INS
        self.l_ins = LOG_INS
        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        elif isinstance(filepath, Path):
            self.filepath = filepath
        else:
            raise TypeError(f"filepath must be a string or Path object, not {type(filepath)}")

        # Check if file exists *and* is a file (not a directory)
        if not self.filepath.is_file():
            # Log details if path exists but isn't a file
            exists = self.filepath.exists()
            is_dir = self.filepath.is_dir()
            log_statement('error', f"{self.l_ins}:ERROR>>File not found or not a file at path: {self.filepath} (Exists: {exists}, Is Dir: {is_dir})", Path(__file__).stem)
            raise FileNotFoundError(f"File not found or not a regular file at path: {self.filepath}")

        self.filename = self.filepath.name
        log_statement('debug', f"{self.l_ins}:DEBUG>>FileReader initialized for: {self.filepath}", Path(__file__).stem)

    def read(self, **kwargs):
        """
        Base read method. Subclasses should override this to implement
        specific file reading logic.
        """
        raise NotImplementedError("Subclasses must implement the read method.")

# --- Robust Text Reader ---
class RobustTextReader(FileReader):
    """
    FileReader subclass specifically for text files.
    Handles encoding detection (using chardet if available) and read errors.
    """
    def __init__(self, filepath: Union[str, Path], default_encoding='utf-8', error_handling='replace', detect_encoding=True):
        """
        Initializes the RobustTextReader.

        Args:
            filepath (str | Path): Path to the text file.
            default_encoding (str): Encoding to use if detection fails or is disabled.
            error_handling (str): How to handle decoding errors ('strict', 'replace', 'ignore').
            detect_encoding (bool): Whether to attempt encoding detection using chardet.
        """
        global LOG_INS
        self.l_ins = LOG_INS
        super().__init__(filepath) # Initialize base class (validates path)
        self.default_encoding = default_encoding
        self.error_handling = error_handling
        # Only enable detection if requested AND chardet library is available
        self.detect_encoding_flag = detect_encoding and HAS_CHARDET
        log_statement('debug', f"{self.l_ins}:DEBUG>>RobustTextReader initialized for {self.filename}: default_enc={default_encoding}, errors={error_handling}, detect={self.detect_encoding_flag}", Path(__file__).stem)

    def _detect_encoding(self, sample_size=4096) -> Optional[str]:
        """Detect file encoding using chardet if available."""
        if not self.detect_encoding_flag:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Encoding detection disabled or chardet not available.", Path(__file__).stem)
            return None # Indicate detection not performed or failed

        try:
            with open(self.filepath, 'rb') as f:
                raw_data = f.read(sample_size)
            if not raw_data: # Handle empty files
                log_statement('debug', f"{self.l_ins}:DEBUG>>File is empty, cannot detect encoding: {self.filename}", Path(__file__).stem)
                return self.default_encoding # Return default for empty file? Or None? Let's try default.

            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            log_statement('debug', f"{self.l_ins}:DEBUG>>Chardet result for {self.filename}: encoding='{encoding}', confidence={confidence:.2f}", Path(__file__).stem)

            # Use detected encoding if confidence is reasonably high
            if encoding and confidence > 0.75:
                # Normalize common cases
                enc_lower = encoding.lower()
                if enc_lower == 'ascii':
                    log_statement('debug', f"{self.l_ins}:DEBUG>>Detected ASCII, using UTF-8 as fallback.", Path(__file__).stem)
                    return 'utf-8' # Treat ASCII as UTF-8 subset for broader compatibility
                # Add other normalizations if needed (e.g., windows-1252 -> cp1252)
                return encoding
            else:
                log_statement('debug', f"{self.l_ins}:DEBUG>>Encoding detection confidence low or failed for {self.filename}. Falling back to default: {self.default_encoding}", Path(__file__).stem)
                return None # Indicate fallback needed
        except FileNotFoundError:
             # Should be caught by base class init, but handle defensively
             log_statement('error', f"{self.l_ins}:ERROR>>File not found during encoding detection: {self.filepath}", Path(__file__).stem)
             raise # Re-raise FileNotFoundError
        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Error detecting encoding for {self.filepath}: {e}", Path(__file__).stem, True)
            return None # Fallback to default on unexpected error

    def read(self) -> Optional[str]:
        """Reads the file content using detected or default encoding."""
        log_statement('debug', f"{self.l_ins}:DEBUG>>Attempting robust text read: {self.filepath}", Path(__file__).stem)
        read_encoding = self.default_encoding # Start with default

        try:
            detected_encoding = self._detect_encoding()
            if detected_encoding:
                read_encoding = detected_encoding

            log_statement('debug', f"{self.l_ins}:DEBUG>>Reading {self.filename} with encoding '{read_encoding}' (errors='{self.error_handling}')", Path(__file__).stem)
            with open(self.filepath, 'r', encoding=read_encoding, errors=self.error_handling) as f:
                content = f.read()
            log_statement('debug', f"{self.l_ins}:DEBUG>>Successfully read {len(content)} chars from {self.filename}", Path(__file__).stem)
            return content

        except FileNotFoundError:
             # Should be caught by base __init__, but handle again just in case.
             log_statement('error', f"{self.l_ins}:ERROR>>File not found during read attempt: {self.filepath}", Path(__file__).stem)
             return None
        except UnicodeDecodeError as ude:
            log_statement('warning', f"{self.l_ins}:WARNING>>UnicodeDecodeError for {self.filepath} with encoding '{read_encoding}': {ude}. Error handling: '{self.error_handling}'.", Path(__file__).stem)
            # If errors='strict' or detection failed badly, maybe try a fallback
            if read_encoding != 'latin-1' and self.error_handling == 'strict':
                 log_statement('warning', f"{self.l_ins}:WARNING>>Attempting fallback read of {self.filepath} with 'latin-1'.", Path(__file__).stem)
                 try:
                     with open(self.filepath, 'r', encoding='latin-1', errors=self.error_handling) as f:
                         return f.read()
                 except Exception as fallback_e:
                      log_statement('error', f"{self.l_ins}:ERROR>>Fallback read failed for {self.filepath}: {fallback_e}", Path(__file__).stem, True)
            return None # Give up if decode error persists
        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Unexpected error reading file {self.filepath}: {e}", Path(__file__).stem, True)
            return None # Return None on other errors

# --- Concrete Reader Implementations ---
class CSVReader(RobustTextReader):
    """Reads CSV files."""
    def read(self, text_column: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        """
        Reads a CSV file, handling encoding via RobustTextReader.
        Returns data as a pandas DataFrame.

        Args:
            text_column (str | None): DEPRECATED - This reader returns the full DataFrame.
                                      Downstream processing should handle column selection.
            **kwargs: Additional keyword arguments passed to pd.read_csv.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the CSV data, or None on failure.
        """
        log_statement('debug', f"{self.l_ins}:DEBUG>>Attempting to read CSV: {self.filepath}", Path(__file__).stem)
        try:
            # Use the robust reader to get content first, handling encoding
            file_content = super().read() # <<< Use parent's read method
            if file_content is None: return None # Error logged by parent
            if not file_content.strip():
                log_statement('warning', f"{self.l_ins}:WARNING>>CSV file is empty: {self.filepath}", Path(__file__).stem)
                return pd.DataFrame() # Return empty DataFrame

            file_stream = StringIO(file_content)

            # Sniff dialect for better parsing (optional but recommended)
            try:
                sniffer = csv.Sniffer()
                # Sample more data if needed for reliable sniffing
                sample = file_stream.read(min(len(file_content), 8192)) # Read up to 8k
                dialect = sniffer.sniff(sample)
                log_statement('debug', f"{self.l_ins}:DEBUG>>Detected CSV dialect for {self.filename}: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'", Path(__file__).stem)
                # Pass detected dialect parameters to pandas
                kwargs['delimiter'] = dialect.delimiter
                kwargs['quotechar'] = dialect.quotechar
                # Handle other dialect properties if necessary (e.g., quoting, doublequote)
                kwargs['quoting'] = dialect.quoting
                kwargs['doublequote'] = dialect.doublequote
                kwargs['escapechar'] = dialect.escapechar # Pass if detected
            except csv.Error as sniff_err:
                log_statement('warning', f"{self.l_ins}:WARNING>>Could not detect CSV dialect for {self.filename} ({sniff_err}). Using pandas defaults.", Path(__file__).stem)
                # Don't explicitly pass dialect='excel', let pandas handle defaults
            finally:
                file_stream.seek(0) # IMPORTANT: Reset stream after sniffing

            # Read using pandas, passing kwargs (which might include dialect info)
            # keep_default_na=False prevents pandas from interpreting 'NA', 'NULL' etc. as NaN
            df = pd.read_csv(file_stream, keep_default_na=False, low_memory=False, **kwargs)

            log_statement('info', f"{self.l_ins}:INFO>>Read {len(df)} rows from CSV: {self.filepath}", Path(__file__).stem)
            return df

        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Failed to read or parse CSV file {self.filepath}: {e}", Path(__file__).stem, True)
            return None

class JSONReader(RobustTextReader): # <<< Inherit from RobustTextReader
    """Reads standard JSON files (object or array) into a DataFrame."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        """
        Reads a JSON file (single object, array, or detected JSON Lines) using RobustTextReader.
        Normalizes the structure and returns a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments passed to json.loads or pd.json_normalize.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the JSON data, or None on failure.
        """
        log_statement('debug', f"{self.l_ins}:DEBUG>>Attempting to read JSON: {self.filepath}", Path(__file__).stem)
        content = super().read() # <<< Use parent's read method for encoding
        if content is None: return None
        if not content.strip():
            log_statement('warning', f"{self.l_ins}:WARNING>>JSON file is empty: {self.filepath}", Path(__file__).stem)
            return pd.DataFrame()

        data = None
        try:
            # Try parsing as a single JSON object/array first
            try:
                data = json.loads(content, **kwargs)
            except json.JSONDecodeError:
                # If single JSON fails, try JSON Lines format
                log_statement('debug', f"{self.l_ins}:DEBUG>>Failed to parse {self.filename} as single JSON, attempting JSON Lines.", Path(__file__).stem)
                file_stream = StringIO(content)
                records = []
                for i, line in enumerate(file_stream):
                    line = line.strip()
                    if not line: continue
                    try:
                        records.append(json.loads(line, **kwargs))
                    except json.JSONDecodeError as jsonl_err:
                        log_statement('warning', f"{self.l_ins}:WARNING>>Skipping invalid JSON line {i+1} in {self.filepath}: {jsonl_err}. Line: {line[:100]}...", Path(__file__).stem)
                        continue # Skip invalid lines in JSONL
                if not records:
                     log_statement('error', f"{self.l_ins}:ERROR>>File {self.filepath} is not valid single JSON or JSON Lines with valid objects.", Path(__file__).stem)
                     return None
                data = records # Use the list of records from JSONL

            # Now process the loaded data (list of dicts, single dict, or list of primitives)
            if isinstance(data, list):
                # Handle list of dicts (common case from JSONL or JSON array of objects)
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                # Handle list of primitives (e.g., [1, 2, 3] or ["a", "b"]) - create single column DF
                elif all(isinstance(item, (str, int, float, bool)) or item is None for item in data):
                     df = pd.DataFrame(data, columns=['value'])
                else: # Mixed list or list of lists - harder to normalize cleanly
                     log_statement('warning', f"{self.l_ins}:WARNING>>JSON file {self.filepath} contains a complex list structure. Attempting normalization, results may vary.", Path(__file__).stem)
                     try: df = pd.json_normalize(data) # Try pandas normalization
                     except Exception: df = pd.DataFrame({'complex_list': [data]}) # Fallback: store raw list
            elif isinstance(data, dict):
                # Convert single object into a DataFrame with one row
                df = pd.DataFrame([data])
            else:
                log_statement('warning', f"{self.l_ins}:WARNING>>Unexpected JSON structure in {self.filepath} (expected list or dict, got {type(data)}). Storing as single value.", Path(__file__).stem)
                df = pd.DataFrame({'value': [data]})

            log_statement('info', f"{self.l_ins}:INFO>>Read {len(df)} records from JSON: {self.filepath}", Path(__file__).stem)
            return df

        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Failed to read or parse JSON file {self.filepath}: {e}", Path(__file__).stem, True)
            return None
class JSONLReader(FileReader): # <<< Inherit from base FileReader (handles binary read)
    """Reads JSON Lines (JSONL) files efficiently."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        """
        Reads a JSON Lines file (one JSON object per line) into a pandas DataFrame.
        Uses the jsonlines library for robust line-by-line reading.

        Args:
            **kwargs: Additional keyword arguments (passed to json.loads via jsonlines).

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the JSONL data, or None on failure.

        Raises:
            ImportError: If the jsonlines library is not installed.
        """
        if not JSONLINES_AVAILABLE:
            log_statement('error', f"{self.l_ins}:ERROR>>The 'jsonlines' library is required to read JSONL files efficiently.", Path(__file__).stem)
            raise ImportError("The 'jsonlines' library is required. Please install it (`pip install jsonlines`).")

        log_statement('info', f"{self.l_ins}:INFO>>Reading JSONL file: {self.filepath}", Path(__file__).stem)
        records = []
        try:
            with jsonlines.open(self.filepath, mode='r', **kwargs) as reader:
                for obj in reader:
                    records.append(obj) # obj is already a dict/list/etc.
            if not records:
                log_statement('warning', f"{self.l_ins}:WARNING>>JSONL file {self.filepath} is empty or contains no valid objects.", Path(__file__).stem)
                return pd.DataFrame()
            return pd.DataFrame(records)
        except Exception as e: # Catch errors during reading/parsing lines
            log_statement('error', f"{self.l_ins}:ERROR>>Error reading JSONL file {self.filepath}: {e}", Path(__file__).stem, True)
            # Decide whether to return partial data or None
            if records:
                 log_statement('warning', f"{self.l_ins}:WARNING>>Returning partial data ({len(records)} records) from JSONL due to error.", Path(__file__).stem)
                 return pd.DataFrame(records) # Return what was read before error
            return None # Return None if error occurred before any records read

class TXTReader(RobustTextReader): # <<< Inherit from RobustTextReader
    """Reads plain text files into a single-row DataFrame."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        """
        Reads the entire content of a text file using RobustTextReader
        into a single 'text' column in a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Optional[pd.DataFrame]: DataFrame with a single row and column 'text', or None on failure.
        """
        log_statement('info', f"{self.l_ins}:INFO>>Reading TXT file: {self.filepath}", Path(__file__).stem)
        try:
            content = super().read() # <<< Use parent's robust read method
            if content is None:
                 # Error already logged by parent, or file not found
                 return None
            return pd.DataFrame({'text': [content]})
        except Exception as e:
            # Catch unexpected errors from DataFrame creation
            log_statement('error', f"{self.l_ins}:ERROR>>Error creating DataFrame from TXT content {self.filepath}: {e}", Path(__file__).stem, True)
            return None

class HTMLReader(RobustTextReader): # <<< Inherit from RobustTextReader
    """Reads HTML file and extracts text using BeautifulSoup."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        """Reads an HTML file, extracts text, returns as DataFrame."""
        if not HAS_BS4:
            log_statement('warning', f"{self.l_ins}:WARNING>>Cannot process HTML file {self.filepath}. BeautifulSoup library not found.", Path(__file__).stem)
            # Fallback: Read As Plaintext using parent's read
            log_statement('warning', f"{self.l_ins}:WARNING>>Reading HTML {self.filename} as plain text (BeautifulSoup not found).", Path(__file__).stem)
            content = super().read()
            return pd.DataFrame({'text': [content]}) if content is not None else None

        log_statement('debug', f"{self.l_ins}:DEBUG>>Attempting to read and parse HTML: {self.filepath}", Path(__file__).stem)
        html_content = super().read() # Use robust reader for initial content loading
        if html_content is None: return None
        if not html_content.strip():
            log_statement('warning', f"{self.l_ins}:WARNING>>HTML file is empty: {self.filepath}", Path(__file__).stem)
            return pd.DataFrame({'text': [""]}) # Return DF with empty string

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            # Get text, separated by spaces, strip whitespace
            text = soup.get_text(separator=' ', strip=True)
            # Further clean excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            log_statement('info', f"{self.l_ins}:INFO>>Extracted {len(text)} characters of text from HTML: {self.filepath}", Path(__file__).stem)
            return pd.DataFrame({'text': [text]})
        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Failed to parse HTML file {self.filepath}: {e}", Path(__file__).stem, True)
            # Fallback to raw content if parsing fails?
            log_statement('warning', f"{self.l_ins}:WARNING>>HTML parsing failed for {self.filename}. Returning raw content.", Path(__file__).stem)
            return pd.DataFrame({'text': [html_content]}) # Return raw on error

# --- Readers for Binary/Structured Formats (Inherit from base FileReader) ---

class ExcelReader(FileReader):
    """Reads Excel files (.xlsx, .xls)."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        log_statement('info', f"{self.l_ins}:INFO>>Reading Excel file: {self.filepath}", Path(__file__).stem)
        try:
            # kwargs.setdefault('sheet_name', 0) # Read first sheet by default if not specified
            return pd.read_excel(self.filepath, **kwargs)
        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Error reading Excel file {self.filepath}: {e}", Path(__file__).stem, True)
            return None

class PDFReader(FileReader):
    """Reads text content from PDF files."""
    def read(self, **kwargs) -> Optional[pd.DataFrame]: # Return DataFrame
        if not PDFMINER_AVAILABLE:
            log_statement('error', f"{self.l_ins}:ERROR>>The 'pdfminer.six' library is required to read PDF files.", Path(__file__).stem)
            raise ImportError("The 'pdfminer.six' library is required. Please install it (`pip install pdfminer.six`).")

        log_statement('info', f"{self.l_ins}:INFO>>Reading PDF file: {self.filepath}", Path(__file__).stem)
        try:
            text = extract_text(self.filepath, **kwargs)
            return pd.DataFrame({'text': [text]}) # Return as DataFrame
        except Exception as e:
            log_statement('error', f"{self.l_ins}:ERROR>>Error reading PDF file {self.filepath}: {e}", Path(__file__).stem, True)
            return None

# --- File Discovery Function ---
def get_reader_class(extension: str):
    """Maps file extension to the appropriate reader class."""
    extension = extension.lower().strip('.')
    # Map extensions to their corresponding reader classes
    reader_map = {
        'csv': CSVReader,
        'json': JSONReader, # Handles both standard JSON and JSON Lines detection
        'jsonl': JSONLReader if JSONLINES_AVAILABLE else JSONReader, # Use efficient reader if available, fallback to general JSON reader
        'txt': TXTReader,
        'md': TXTReader, # Treat markdown as text
        'log': TXTReader, # Treat log files as text
        'py': TXTReader, # Treat python files as text
        'xlsx': ExcelReader,
        'xls': ExcelReader,
        'pdf': PDFReader if PDFMINER_AVAILABLE else None, # Only if library available
        'html': HTMLReader if HAS_BS4 else TXTReader, # Use HTMLReader if bs4 exists, else TXTReader
        'htm': HTMLReader if HAS_BS4 else TXTReader,
        # Add other mappings here based on implemented readers
        # 'docx': DocxReader if DOCX_AVAILABLE else None,
        # 'odt': OdtReader if ODFPY_AVAILABLE else None,
        # 'rtf': RtfReader if STRIPRTF_AVAILABLE else None,
        # 'epub': EpubReader if EBOOKLIB_AVAILABLE else None,
        # 'zip': ZipReader if ZIPFILE_AVAILABLE else None, # For reading contents
        # Note: .zst on its own is ambiguous, needs other extension (e.g. csv.zst)
        # Handling for compressed types like csv.zst needs logic in the caller
        # to decompress first or use readers aware of compressed streams.
    }
    reader_class = reader_map.get(extension)
    if reader_class is None:
        log_statement('debug', f"{LOG_INS}:DEBUG>>No specific reader found for extension '.{extension}'.", Path(__file__).stem)
    return reader_class

# --- open_files_recursively Function (Simplified, Focus on Discovery) ---
# This function's purpose seems to overlap heavily with DataRepository.scan_and_update
# It might be better to rely on DataRepository for managing file status.
# Keeping a simplified version for potential standalone discovery needs.

def discover_files_recursively(folder_path: Union[str, Path],
                               output_csv: Optional[Union[str, Path]] = None):
    """
    Recursively scans a folder, identifies supported files based on extensions,
    and optionally saves a basic list to CSV.

    Args:
        folder_path (str | Path): The path to the folder to scan.
        output_csv (str | Path, optional): Path to save the discovered file list.
                                           If None, only logs findings.
    """
    if isinstance(folder_path, str): folder_path = Path(folder_path)
    if not folder_path.is_dir():
        log_statement('error', f"{LOG_INS}:ERROR>>Folder not found: {folder_path}", Path(__file__).stem); return

    log_statement('info', f"{LOG_INS}:INFO>>Starting recursive file discovery in: {folder_path}", Path(__file__).stem)
    discovered_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = Path(root) / filename
            extension = filepath.suffix.lower().strip('.')
            reader_class = get_reader_class(extension) # Check if we have a reader
            if reader_class: # If supported
                discovered_files.append({
                    'filepath': str(filepath.resolve()),
                    'filename': filename,
                    'extension': extension,
                    'reader_class': reader_class.__name__
                })
            else: # Unsupported
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Unsupported file type skipped: {filename}", Path(__file__).stem)

    log_statement('info', f"{LOG_INS}:INFO>>Discovery complete. Found {len(discovered_files)} supported files.", Path(__file__).stem)

    if output_csv:
        try:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(discovered_files)
            df.to_csv(output_csv, index=False)
            log_statement('info', f"{LOG_INS}:INFO>>Discovered file list saved to: {output_csv}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to save discovery list to {output_csv}: {e}", Path(__file__).stem, True)

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
        log_statement('error', f"{LOG_INS}:ERROR>>Folder not found: {folder_path}", Path(__file__).stem)
        return

    log_statement('info', f"{LOG_INS}:INFO>>Starting recursive file discovery in: {folder_path}", Path(__file__).stem)

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
                    log_statement('info', f"{LOG_INS}:INFO>>Successfully opened '{filename}' using {ReaderClass.__name__}", Path(__file__).stem)
                    file_info['read_status'] = 'Read'
                    progress_info = {'filename': filename, 'processed_percentage': 0}

                except ImportError as ie:
                     log_statement('warning', f"{LOG_INS}:WARNING>>Skipping '{filename}': Required library not installed ({ie})", Path(__file__).stem)
                     file_info['read_status'] = 'Skipped (Missing Lib)'
                     file_info['error_message'] = str(ie)
                except FileNotFoundError:
                     # Should not happen if os.walk yields it, but handle defensively
                     log_statement('error', f"{LOG_INS}:ERROR>>File not found during read attempt (should exist): {filepath}", Path(__file__).stem)
                     file_info['read_status'] = 'Error (Not Found)'
                     file_info['error_message'] = 'File disappeared during scan'
                except Exception as e:
                    log_statement('error', f"{LOG_INS}:ERROR>>Failed to open '{filename}' using {ReaderClass.__name__}: {e}", Path(__file__).stem)
                    file_info['read_status'] = 'Failed'
                    file_info['error_message'] = str(e)
            else:
                 log_statement('debug', f"{LOG_INS}:DEBUG>>Unsupported file type: {filename}", Path(__file__).stem)
                 file_info['read_status'] = 'Unsupported'


            repo_data.append(file_info)
            if progress_info:
                progress_data.append(progress_info)

    # Create DataFrames and save
    try:
        repo_df = pd.DataFrame(repo_data)
        repo_df.to_csv(repo_file, index=False)
        log_statement('info', f"{LOG_INS}:INFO>>File discovery report saved to: {repo_file}", Path(__file__).stem)
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to save repository file {repo_file}: {e}", Path(__file__).stem, True)

    try:
        progress_df = pd.DataFrame(progress_data)
        progress_df.to_csv(progress_file, index=False)
        log_statement('info', f"{LOG_INS}:INFO>>Initial progress file saved to: {progress_file}", Path(__file__).stem)
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to save progress file {progress_file}: {e}", Path(__file__).stem, True)

# Example usage (usually called from elsewhere)
# if __name__ == "__main__":
#     discover_files_recursively("./data/raw", "./data/discovery_report.csv")