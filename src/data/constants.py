# src/data/constants.py
import os
from pathlib import Path
import inspect
import dotenv

dotenv.load_dotenv()

# Version Number
VER_NO = '1.000a'

# Repository file constants
_project_folder_str = os.getenv("TLATO41DIR")
if not _project_folder_str:
    raise ValueError("Environment variable TLATO41DIR is not set.")
PROJECT_FOLDER = Path(_project_folder_str)
REPO_DIR = PROJECT_FOLDER / "src/data/repositories"
MAIN_REPO_FILENAME = "main_repository.csv.zst"
PROCESSED_REPO_FILENAME = "processed_repository.csv.zst"
TOKENIZED_REPO_FILENAME = "tokenized_repository.csv.zst"
DATALOADER_METADATA_FILENAME = "dataloader_metadata.json.zst" # Example for DataLoader info
DATA_REPO_DIR = REPO_DIR / "data_repository"
LOG_DIR = PROJECT_FOLDER / "logs"
STATE_DIR = PROJECT_FOLDER / "states"

# --- Repository Index Structure Keys ---
INDEX_FILE = "repositories/repository_index.json"
INDEX_KEY_PATH = "path"
INDEX_KEY_METADATA = "metadata"
INDEX_KEY_CHILDREN = "children" # List of child repository hashes
TOKENIZED_DATA_DIR = f"{os.getenv("TLATO41DIR")}/data/tokenized"

LOG_INS = f'{__name__}:{__file__}:{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:'

# --- Repository Index Metadata Keys ---
INDEX_META_FILE_COUNT = "file_count"
INDEX_META_TOTAL_SIZE = "total_size_bytes"
INDEX_META_MIN_MTIME = "min_mtime_utc"
INDEX_META_MAX_MTIME = "max_mtime_utc"
METADATA_FILENAME = "metadata.json.zst"

# --- Core File Metadata ---
COL_FILEPATH = 'filepath'         # Absolute path to the original file
COL_FILENAME = 'filename'       # Just the file name
COL_SIZE = 'size_bytes'         # File size in bytes
COL_MTIME = 'mtime_ts'          # Modification timestamp (float seconds since epoch)
COL_CTIME = 'ctime_ts'          # Creation timestamp (float seconds since epoch)
COL_HASH = 'content_hash'       # Hash of the file content (e.g., SHA256)
COL_EXTENSION = 'extension'     # File extension (lowercase, no dot)

# --- Processing Status & Info ---
COL_STATUS = 'status'           # e.g., discovered, loaded, processed, tokenized, error
COL_ERROR = 'error_message'     # Error message if status is 'error'
COL_PROCESSED_PATH = 'processed_path' # Relative path to processed output
COL_TOKENIZED_PATH = 'tokenized_path' # Relative path to tokenized output
COL_LAST_UPDATED = 'last_updated_ts' # Timestamp of last repo update for this row
COL_DTYPE = 'dtype'
COL_DATA_CLASSIFICATION = 'data_classification'
COL_FINAL_CLASSIFICATION = 'final_classification'

# --- Constants that might be used by these models or for validation ---
DEFAULT_APPLICATION_STATUS = "new" # Example default status

# Hashing Algorithms
HASH_MD5 = "md5"
HASH_SHA256 = "sha256"
HASH_SHA1 = "sha1" 
# Default list of hash algorithms to calculate for the 'custom_hashes' field in metadata
DEFAULT_CONTENT_HASH_ALGORITHMS = [HASH_MD5, HASH_SHA256] 
# General list of supported algorithms by the hashing utilities (broader than what's stored in metadata by default)
SUPPORTED_HASH_ALGORITHMS = [HASH_MD5, HASH_SHA1, HASH_SHA256, "sha512", "blake2b", "ed25591"] # Example
SUPPORTED_HASH_TYPES_FOR_CUSTOM = ["md5", "sha256", "sha1", "sha512", "blake2b", "ed25591"] # Align with what your hashing utility produces
HASH_BUFFER_SIZE = 65536  # 64k buffer
DEFAULT_HASH_ALGORITHM = HASH_SHA1

ROOT_DIR = os.getenv("TLATO41ROOTDIR")

# --- Optional/Additional Metadata (Add if used) ---
COL_DESIGNATION = 'designation'     # Unique integer ID (if needed)
COL_HASHED_PATH_ID = 'path_hash'    # Hash of the filepath string (if needed)
COL_COMPRESSED_FLAG = 'is_compressed' # Flag for original compression ('Y'/'N')
COL_IS_COPY_FLAG = 'is_copy'        # Flag for content duplicates ('Y'/'N').
COL_DATA_HASH = 'content_hash'      # Redundant? Ensure COL_HASH is used consistently for content. If COL_DATA_HASH is truly different, define it. Otherwise, remove one
COL_FILETYPE = "Filetype"
COL_MOD_DATE = "ModificationDate"
COL_ACC_DATE = "AccessedDate"
COL_SEMANTIC_LABEL = 'semantic_label' # Example column for storing results
COL_LINGUISTIC_METADATA = 'linguistic_metadata' # Example column for other metadata

# --- Compression Constants ---
COMPRESSION_ENABLED = True
COMPRESSION_LEVEL = 22
COMPRESSION_USED = "zstd"
COMPRESSION_USED_GZ = "gzip"

# Constants for process_file
CONTENT_SNIPPET_BYTES = 1024  # Read first 1KB for snippet
CONTENT_SNIPPET_LINES = 5     # Max number of lines for text snippet

# --- Status Constants ---
STATUS_LINGUISTIC_PROCESSING = 'LINGUISTIC_PROCESSING'
STATUS_LINGUISTIC_PROCESSED = 'LINGUISTIC_PROCESSED'
STATUS_LINGUISTIC_FAILED = 'LINGUISTIC_FAILED'

# Define the full header based on the columns
MAIN_REPO_HEADER = [
    COL_FILEPATH,
    COL_FILENAME,
    COL_FILETYPE,
    COL_SIZE,
    COL_MTIME,
    COL_CTIME,
    COL_EXTENSION,
    COL_DTYPE,
    COL_STATUS,
    COL_PROCESSED_PATH,
    COL_TOKENIZED_PATH,
    COL_LAST_UPDATED,
    COL_HASHED_PATH_ID,
    COL_DATA_HASH,
    COL_COMPRESSED_FLAG,
    COL_IS_COPY_FLAG,
    COL_DESIGNATION,
    COL_ERROR,
    COL_DATA_CLASSIFICATION,
    COL_FINAL_CLASSIFICATION,
    COL_SEMANTIC_LABEL,
    COL_LINGUISTIC_METADATA,
    'base_dir'       # Used by DataRepository methods
]
COL_SCHEMA = {
    COL_FILEPATH: str,
    COL_FILENAME: str,
    COL_FILETYPE: str,
    COL_SIZE: 'Int64',
    COL_MTIME: 'datetime64[ns, UTC]',
    COL_CTIME: 'datetime64[ns, UTC]',
    COL_HASH: str,
    COL_EXTENSION: str,
    COL_DTYPE: str,
    COL_STATUS: str,
    COL_PROCESSED_PATH: str,
    COL_TOKENIZED_PATH: str,
    COL_LAST_UPDATED: 'datetime64[ns, UTC]', # Use the correct constant
    COL_HASHED_PATH_ID: str,
    COL_DATA_HASH: str,
    COL_COMPRESSED_FLAG: str, 
    COL_IS_COPY_FLAG: str,
    COL_DESIGNATION: 'Int64',
    COL_ERROR: str,
    COL_DATA_CLASSIFICATION: str,
    COL_FINAL_CLASSIFICATION: str,
    COL_SEMANTIC_LABEL: str,
    COL_LINGUISTIC_METADATA: str,
    'base_dir': str,
}
TIMESTAMP_COLUMNS = [ COL_MTIME, COL_CTIME, COL_LAST_UPDATED ]
PROCESSED_REPO_COLUMNS = MAIN_REPO_HEADER
TOKENIZED_REPO_COLUMNS = MAIN_REPO_HEADER

# Status Codes
STATUS_UNKNOWN = 'U'      # Unknown status
STATUS_NEW = 'N'          # New file added to repository
STATUS_HASHING = 'H'      # File being hashed
STATUS_HASHED = 'Hd'      # File hashed, hash calculated
STATUS_COMPRESSING = 'C'  # File being compressed
STATUS_COMPRESSED = 'Cd'   # File compressed, hash calculated
STATUS_DECOMPRESSING = 'D' # File being decompressed
STATUS_DECOMPRESSED = 'Dd' # File decompressed, hash calculated
STATUS_READING = 'R'      # File being read
STATUS_READ = 'Rd'         # File read, hash calculated
STATUS_WRITING = 'W'      # File being written
STATUS_WRITTEN = 'Wr'      # File written, hash calculated
STATUS_LOADED = "loaded"       # File loaded, hash calculated
STATUS_PROCESSING = "processing"   # File sent for processing
STATUS_PROCESSED = "processed"    # File processing complete
STATUS_TOKENIZING = "tokenizing"   # File sent for tokenization
STATUS_TOKENIZED = "tokenized"    # File tokenization complete
STATUS_ERROR = "error"        # Error occurred during processing/hashing/tokenization
STATUS_DISCOVERED = "discovered"     # Or use 'discovered' if loading isn't a distinct step
STATUS_MISSING = "missing"

# --- Data Classification Types ---
# These constants represent the classified nature of the data content
TYPE_TEXTUAL = "TEXTUAL"           # Primarily natural language text
TYPE_NUMERICAL = "NUMERICAL"         # Primarily structured numerical data (tabular)
TYPE_TOKENIZED_NUMERICAL = "TOKENIZED_NUM" # Data resembling numerical tensors/vectors post-tokenization
TYPE_TOKENIZED_SUBWORD = "TOKENIZED_SUB" # Data resembling subword/text tokens post-tokenization
TYPE_TOKENIZED_JSONL = "TOKENIZED_JSONL" # Data resembling JSONL tokens post-tokenization" \
TYPE_TOKENIZED_CSV = "TOKENIZED_CSV" # Data resembling CSV tokens post-tokenization
TYPE_TOKENIZED_XML = "TOKENIZED_XML" # Data resembling XML tokens post-tokenization
TYPE_TOKENIZED_TEXT = "TOKENIZED_TEXT" # Data resembling text tokens post-tokenization
TYPE_UNKNOWN = "UNKNOWN"           # Could not reliably classify
TYPE_EMPTY = "EMPTY"             # File was read but contained no usable data
TYPE_BINARY = "BINARY"           # Binary data, not text or structured
TYPE_IMAGE = "IMAGE"           # Image data
TYPE_AUDIO = "AUDIO"           # Audio data
TYPE_VIDEO = "VIDEO"           # Video data
TYPE_ARCHIVE = "ARCHIVE"         # Compressed archive (zip, zst, etc.)
TYPE_DOCUMENT = "DOCUMENT"       # Document files (PDF, DOCX, etc.)
TYPE_CODE = "CODE"             # Source code files (Python, Java, etc.)
TYPE_TOKEN = "TOKEN"
TYPE_MIXED = "MIXED"
TYPE_PDF = "PDF"
TYPE_DOC = "DOC"
TYPE_DOCX = "DOCX"
TYPE_EXCEL = "EXCEL"
TYPE_JSON = "JSON"
TYPE_JSONL = "JSONL"
TYPE_XML = "XML"
TYPE_YAML = "YAML"
TYPE_IMAGE_PNG = 'IMAGE_PNG'
TYPE_IMAGE_JPEG = 'IMAGE_JPEG'
TYPE_IMAGE_GIF = 'IMAGE_GIF'
TYPE_IMAGE_BMP = 'IMAGE_BMP'
TYPE_AUDIO_WAV = 'AUDIO_WAV'
TYPE_AUDIO_MP3 = 'AUDIO_MP3' # Note: MP3 detection via magic number is less reliable due to ID3 tags
TYPE_COMPRESSED_ZIP = 'COMPRESSED_ZIP'
TYPE_COMPRESSED_GZIP = 'COMPRESSED_GZIP'
TYPE_COMPRESSED_BZ2 = 'COMPRESSED_BZ2'
TYPE_COMPRESSED_7Z = 'COMPRESSED_7Z'
TYPE_COMPRESSED_RAR = 'COMPRESSED_RAR'
TYPE_COMPRESSED_ZSTD = 'COMPRESSED_ZSTD'
TYPE_PROCESSED = "PROCESSED" # Data resembling processed data (e.g., tokenized)
TYPE_PROCESSED_CSV = "PROCESSED_CSV" # Data resembling processed CSV data
TYPE_PROCESSED_JSONL = "PROCESSED_JSONL" # Data resembling processed JSONL data
TYPE_PROCESSED_XML = "PROCESSED_XML" # Data resembling processed XML data
TYPE_PROCESSED_TEXT = "PROCESSED_TEXT" # Data resembling processed text data
TYPE_PROCESSED_NUMERICAL = "PROCESSED_NUM" # Data resembling processed numerical data
TYPE_PROCESSED_SUBWORD = "PROCESSED_SUB" # Data resembling processed subword/text tokens
TYPE_SYNTAX_ERROR = "SYNTAX_ERROR" # Syntax error in the file
TYPE_UNSUPPORTED = "UNSUPPORTED" # Unsupported file type or format
TYPE_TABULAR = 'TABULAR' # Example
TYPE_HTML = 'HTML'       # Example
TYPE_COMPRESSED = 'COMPRESSED' # Example
TYPE_MARKDOWN = 'MARKDOWN' # Example
# File extensions for processed/tokenized mirrored files
PROCESSED_EXT = ".proc"
TOKENIZED_EXT = ".token"

# Accepted file types for processing (expand as needed based on readers)
# This should align with your actual file reading capabilities (e.g., in readers.py)
ACCEPTED_FILE_TYPES = {
    '.csv', '.txt', '.json', '.jsonl',
    '.xls', '.xlsx', # Require openpyxl or similar
    '.pdf', # Require PyPDF2 or similar
    '.rtf', '.doc', '.docx', # Require specific libraries like python-docx, striprtf
    '.zst', '.zstd', '.zip', # Compressed archives need handling
    '.tar', '.gz', # Compressed archives need handling
    '.jpg', '.jpeg', '.png', # Image files
    '.mp3', '.wav', # Audio files
    '.mp4', '.avi', # Video files
    '.html', '.htm', # HTML files
    '.xml', # XML files
    '.yaml', '.yml', # YAML files
    '.pickle', # Pickle files
    '.parquet', # Parquet files
    '.hdf5', # HDF5 files
    '.feather', # Feather files
    '.msg', # Outlook message files
    '.eml', # Email files
    '.ppt', '.pptx', # PowerPoint files
    '.csv.zst', '.json.zst', # Compressed files
    '.txt.zst', '.xls.zst', # Compressed files
    '.xlsx.zst', # Compressed files
    '.zip.zst', # Compressed files
    '.tar.zst', # Compressed files
    '.gz.zst', # Compressed files
    '.md', # Markdown files
    '.log', # Log files
    '.tsv', # Tab-separated values
    '.tsv.zst', # Compressed TSV files
    '.parquet.zst', # Compressed Parquet files
    '.hdf5.zst', # Compressed HDF5 files
    '.feather.zst', # Compressed Feather files
    '.msg.zst', # Compressed Outlook message files
    '.eml.zst', # Compressed email files
    '.ppt.zst', # Compressed PowerPoint files
    '.pptx.zst', # Compressed PowerPoint files
    '.csv.gz', # Compressed CSV files
    '.json.gz', # Compressed JSON files
    '.txt.gz', # Compressed TXT files
    '.xls.gz' # Compressed XLS files
}
SUPPORTED_EXTENSIONS = ACCEPTED_FILE_TYPES
# Note: Handling archives (.zip, .zst containing other files) requires specific logic
# to extract and process contained files, adding them to the repository individually.

# Output directories
OUTPUT_DIR_BASE = "data/output" # Mirrored structure will be created under this
MODEL_DIR = "src/models"
PROGRESS_DIR = "data/progress"

# Schema for the repository file tracking DataFrame
REPO_FILE_SCHEMA = {
    'file_id': {'type': 'string', 'nullable': False, 'unique': True}, # Unique identifier for the file entry (e.g., hash of relative_path + version_tag)
    'file_path': {'type': 'string', 'nullable': False},           # Relative path within the repository
    'absolute_path': {'type': 'string', 'nullable': True},        # Absolute path on the filesystem (can be None if not applicable)
    'version': {'type': 'integer', 'nullable': False, 'default': 1}, # Version number of the file in the repo
    'file_hash': {'type': 'string', 'nullable': True},            # Hash of the file content (SHA256 recommended)
    'size': {'type': 'integer', 'nullable': True},                # File size in bytes
    'status': {'type': 'string', 'nullable': False,               # Status: 'added', 'modified', 'committed', 'untracked', 'deleted', 'cached'
               'allowed': ['added', 'modified', 'committed', 'untracked', 'deleted', 'cached', 'to_add', 'to_commit']},
    'timestamp': {'type': 'datetime64[ns]', 'nullable': False},   # Timestamp of last action or scan (use timezone-aware UTC)
    'metadata_json': {'type': 'string', 'nullable': True},        # Other metadata as a JSON string (e.g., tags, description)
    'zone_label': {'type': 'string', 'nullable': True}            # Semantic zone label if applicable
}

MAX_REPO_DF_CACHE_SIZE = 9999999  # Maximum number of entries in the RepoHandler's 'df' cache
