# src/data/constants.py

# Repository file constants
REPO_DIR = "data/repositories"
MAIN_REPO_FILENAME = "main_repository.csv.zst"
PROCESSED_REPO_FILENAME = "processed_repository.csv.zst"
TOKENIZED_REPO_FILENAME = "tokenized_repository.csv.zst"
DATALOADER_METADATA_FILENAME = "dataloader_metadata.json.zst" # Example for DataLoader info

# Column names for the main repository CSV

COL_DESIGNATION = "Designation"
COL_FILETYPE = "Filetype"
COL_FILEPATH = "Filepath"
COL_HASHED_PATH_ID = "HashedPathID"
COL_COMPRESSED_FLAG = "Compressed" # Y/N for original file compression
COL_MOD_DATE = "ModificationDate"
COL_ACC_DATE = "AccessedDate"
COL_DATA_HASH = "DataHash"
COL_IS_COPY_FLAG = "IsCopy" # Y/N
COL_STATUS = "Status"
COL_FILENAME = "Filename" 
COL_SIZE = "Size"
COL_MTIME = "ModTime" # Unix timestamp
COL_CTIME = "CreationTime" # Unix timestamp
COL_HASH = "Hash" # SHA-256 hash of the file content
COL_EXTENSION = "ExtType"
COL_ERROR = "ErrorMSG"

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
STATUS_LOADED = 'L'       # File loaded, hash calculated
STATUS_PROCESSING = 'P'   # File sent for processing
STATUS_PROCESSED = 'Pd'    # File processing complete
STATUS_TOKENIZING = 'T'   # File sent for tokenization
STATUS_TOKENIZED = 'Td'    # File tokenization complete
STATUS_ERROR = 'E'        # Error occurred during processing/hashing/tokenization

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
TYPE_PROCESSED = "PROCESSED" # Data resembling processed data (e.g., tokenized)
TYPE_PROCESSED_CSV = "PROCESSED_CSV" # Data resembling processed CSV data
TYPE_PROCESSED_JSONL = "PROCESSED_JSONL" # Data resembling processed JSONL data
TYPE_PROCESSED_XML = "PROCESSED_XML" # Data resembling processed XML data
TYPE_PROCESSED_TEXT = "PROCESSED_TEXT" # Data resembling processed text data
TYPE_PROCESSED_NUMERICAL = "PROCESSED_NUM" # Data resembling processed numerical data
TYPE_PROCESSED_SUBWORD = "PROCESSED_SUB" # Data resembling processed subword/text tokens
TYPE_TOKENIZED_SUBWORD = "TOKENIZED_SUBWORD" # Data resembling subword/text tokens post-tokenization
TYPE_TOKENIZED_NUMERICAL = "TOKENIZED_NUMERICAL" # Data resembling numerical tensors/vectors post-tokenization
TYPE_SYNTAX_ERROR = "SYNTAX_ERROR" # Syntax error in the file
TYPE_UNSUPPORTED = "UNSUPPORTED" # Unsupported file type or format

# Define the full header based on the columns
MAIN_REPO_HEADER = [
    COL_DESIGNATION,
    COL_FILETYPE,
    COL_FILEPATH,
    COL_HASHED_PATH_ID,
    COL_COMPRESSED_FLAG,
    COL_MOD_DATE,
    COL_ACC_DATE,
    COL_DATA_HASH,
    COL_IS_COPY_FLAG,
    COL_STATUS, 
    COL_SIZE,
    COL_MTIME,
    COL_CTIME,
    COL_HASH,
    COL_ERROR,
    'processed_path', # Used by DataRepository methods
    'tokenized_path', # Used by DataRepository methods
    'base_dir',       # Used by DataRepository methods
    'last_modified_scan', # Used by DataRepository methods
    'last_updated_repo'   # Used by DataRepository methods
]

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