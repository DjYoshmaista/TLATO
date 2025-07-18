# src/data/processing.py
"""
Enhanced Data Processing Module - Refactored for m1.py Integration

Handles data preprocessing pipelines including:
- Managing a repository of dataset files and their processing status
- Processing raw data (text cleaning, numerical scaling, semantic labeling)
- Tokenizing/vectorizing processed data into tensors
- Thread-safe operations with enhanced error handling
- GPU acceleration (cuDF, CuPy, cuML) with CPU fallbacks
- Integration with m1.py command pattern and dependency injection
"""

import shutil
import inspect
import os
import sys
import csv
import codecs
import re
import json
import io
import time
import datetime as dt
from datetime import timezone
from threading import Lock, RLock
from typing import Dict, Optional, Any, List, Generator, Union, Tuple, Callable
from pathlib import Path
import hashlib
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass, field
from enum import Enum
import threading
import zstandard as zstd
from src.utils.logger import log_statement, get_log_prefix
try:
    from src.context.container import DataProcessingContext, DataProcessingContainer
    log_statement('info', "DataProcessingContext and DataProcessingContainer imported successfully from src/context/container.py.", Path(__file__).stem)
except ImportError:
    log_statement('warning', "Failed to import DataProcessingContext and/or DataProcessingContainer from src/context/container.py. Using fallback context.", Path(__file__).stem)
    # Fallback import for m1.py context if not available
    class DataProcessingContext:
        """Dummy context for m1.py integration fallback"""
        def __init__(self):
            self.config = None
            self.container = None
            self.log_prefix = "m1_fallback"
            self.repo_handler = None
        def set_config(self, config):
            self.config = config
        def set_container(self, container):
            self.container = container
        def set_repo_handler(self, repo_handler):
            self.repo_handler = repo_handler
    class DataProcessingContainer:
        """Dummy container for m1.py integration fallback"""
        def __init__(self):
            self.context = DataProcessingContext()
            self.repo_handler = None
        def set_context(self, context):
            self.context = context
        def set_repo_handler(self, repo_handler):
            self.repo_handler = repo_handler
    log_statement('error', "DataProcessingContext import failed. Ensure src/context/container.py is correctly set up.", Path(__file__).stem)

# Import project configuration and utilities
try:
    from src.utils.config import *
    from src.utils.compression import *
    from src.utils.hashing import *
    from src.utils.helpers import *
    from src.data.constants import *
    from src.core.repo_handler import RepoHandler, OperationResult, OperationStatus, safe_operation
    from src.data.readers import RobustTextReader, FileReader, get_reader_class
    # ADDED: System resource optimization imports
    from src.utils.system_resources import get_system_resources, get_optimal_config
    # ADDED: Import the correct enhanced progress tracker factory
    from src.utils.progress_tracker import create_progress_tracker as create_enhanced_progress_tracker
    PROJECT_IMPORTS_AVAILABLE = True
    log_statement('info', f"'{__name__}':'{__file__}':INFO>>Project imports loaded successfully.", Path(__file__).stem)
except ImportError as e:
    PROJECT_IMPORTS_AVAILABLE = False
    logging.error(f"Failed relative import in processing.py: {e}")
    
    # Fallback definitions
    BASE_DATA_DIR = Path('./data')
    PROCESSED_DATA_DIR = BASE_DATA_DIR / 'processed'
    TOKENIZED_DATA_DIR = BASE_DATA_DIR / 'tokenized'
    DEFAULT_DEVICE = 'cpu'
    PROJECT_ROOT = Path('.')
    COMPRESSION_ENABLED = True
    COMPRESSION_LEVEL = 22
    DATA_REPO_FILE = BASE_DATA_DIR / 'data_repository.csv.zst'
    
    # Status constants
    STATUS_NEW = "new"
    STATUS_DISCOVERED = "discovered"
    STATUS_LOADED = "loaded"
    STATUS_PROCESSING = "processing"
    STATUS_PROCESSED = "processed"
    STATUS_TOKENIZING = "tokenizing"
    STATUS_TOKENIZED = "tokenized"
    STATUS_ERROR = "error"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED = "skipped"
    
    # Column constants
    COL_DESIGNATION = "designation"
    COL_FILEPATH = "filepath"
    COL_STATUS = "status"
    COL_ERROR = "error_message"
    COL_DATA_CLASSIFICATION = "data_classification"
    COL_FINAL_CLASSIFICATION = "final_classification"
    COL_PROCESSED_PATH = "processed_path"
    COL_PROCESSED_FILENAME = "processed_filename"
    COL_DATA_HASH = "data_hash"
    COL_HASH = "hash"
    
    # Type constants
    TYPE_TEXTUAL = "textual"
    TYPE_NUMERICAL = "numerical"
    TYPE_UNKNOWN = "unknown"
    TYPE_EMPTY = "empty"
    TYPE_BINARY = "binary"
    TYPE_PDF = "pdf"
    TYPE_DOC = "doc"
    TYPE_DOCX = "docx"
    TYPE_EXCEL = "excel"
    TYPE_HTML = "html"
    TYPE_XML = "xml"
    TYPE_JSON = "json"
    TYPE_JSONL = "jsonl"
    TYPE_YAML = "yaml"
    TYPE_MARKDOWN = "markdown"
    TYPE_CODE = "code"
    TYPE_TABULAR = "tabular"
    TYPE_CSV = "csv"
    TYPE_IMAGE_PNG = "image_png"
    TYPE_IMAGE_JPEG = "image_jpeg"
    TYPE_IMAGE_GIF = "image_gif"
    TYPE_IMAGE_BMP = "image_bmp"
    TYPE_AUDIO_WAV = "audio_wav"
    TYPE_AUDIO_MP3 = "audio_mp3"
    TYPE_COMPRESSED_ZIP = "compressed_zip"
    TYPE_COMPRESSED_GZIP = "compressed_gzip"
    TYPE_COMPRESSED_BZ2 = "compressed_bz2"
    TYPE_COMPRESSED_7Z = "compressed_7z"
    TYPE_COMPRESSED_RAR = "compressed_rar"
    TYPE_COMPRESSED_ZSTD = "compressed_zstd"
    TYPE_TOKENIZED_SUBWORD = "tokenized_subword"
    TYPE_TOKENIZED_NUMERICAL = "tokenized_numerical"
    TYPE_TOKENIZED_JSONL = "tokenized_jsonl"
    TYPE_TOKENIZED_CSV = "tokenized_csv"
    
    # Mock functions for fallback
    def log_statement(level, message, module=None, exc_info=False):
        print(f"[{level.upper()}] {message}")
    
    def generate_data_hash(filepath):
        return hashlib.sha256(str(filepath).encode()).hexdigest()[:16]
    
    def hash_filepath(filepath):
        return hashlib.sha256(str(filepath).encode()).hexdigest()[:12]
    
    def compress_string_to_file(content, filepath):
        try:
            with open(filepath, 'wb') as f:
                cctx = zstd.ZstdCompressor()
                compressed = cctx.compress(content.encode('utf-8'))
                f.write(compressed)
            return True
        except Exception:
            return False
    
    class OperationResult(dict):
        pass
    
    class OperationStatus:
        SUCCESS = "success"
        FAILURE = "failure"
    
    def safe_operation(name, func):
        try:
            return {"status": OperationStatus.SUCCESS, "result": func()}
        except Exception as e:
            return {"status": OperationStatus.FAILURE, "error": str(e)}
        
# GPU Library Imports with enhanced fallback handling
GPU_AVAILABLE = False
cudf = None
cp = np
CumlScaler = None
UnsupportedCUDAError = None

try:
    from src.utils.gpu_switch import get_compute_backend, IS_CUDA_AVAILABLE
    compute_backend = get_compute_backend()
    
    if compute_backend == 'cudf':
        try:
            import cudf
            import cupy as cp
            try:
                from cuml.preprocessing import StandardScaler as CumlScaler
            except ImportError:
                from cuml import StandardScaler as CumlScaler
            GPU_AVAILABLE = True
            log_statement('info', f"'{__name__}':INFO>>GPU backend (cuDF/CuPy/cuML) loaded successfully.", Path(__file__).stem)
        except ImportError:
            log_statement('warning', f"'{__name__}':WARNING>>cuDF requested but not available. Using CPU fallback.", Path(__file__).stem)
            import pandas as pd
            cudf = None
            cp = np
    else:
        import pandas as pd
        log_statement('info', f"'{__name__}':INFO>>CPU backend (pandas/numpy) selected.", Path(__file__).stem)
        
except ImportError:
    import pandas as pd
    log_statement('warning', f"'{__name__}':WARNING>>GPU utilities not available. Using CPU fallback.", Path(__file__).stem)

# Enhanced fallback implementations
if not GPU_AVAILABLE:
    cp = np  # Ensure cp is numpy alias
    
    # Enhanced cudf dummy
    class EnhancedCudfDummy:
        @staticmethod
        def DataFrame(*args, **kwargs):
            return pd.DataFrame(*args, **kwargs)
        
        @staticmethod
        def Series(*args, **kwargs):
            return pd.Series(*args, **kwargs)
        
        @staticmethod
        def read_csv(*args, **kwargs):
            return pd.read_csv(*args, **kwargs)
        
        @staticmethod
        def concat(*args, **kwargs):
            return pd.concat(*args, **kwargs)
        
        @staticmethod
        def from_pandas(pdf):
            return pdf
    
    if cudf is None:
        cudf = EnhancedCudfDummy
    
    # Enhanced scaler with better error handling
    if CumlScaler is None:
        try:
            from sklearn.preprocessing import StandardScaler as SklearnScaler
            
            class EnhancedSklearnScalerWrapper:
                def __init__(self, *args, **kwargs):
                    self._scaler = SklearnScaler(*args, **kwargs)
                    self._fitted = False
                
                def fit_transform(self, data):
                    if isinstance(data, pd.DataFrame):
                        result = pd.DataFrame(
                            self._scaler.fit_transform(data), 
                            columns=data.columns, 
                            index=data.index
                        )
                        self._fitted = True
                        return result
                    elif isinstance(data, np.ndarray):
                        original_shape = data.shape
                        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
                        scaled_data = self._scaler.fit_transform(data_2d)
                        self._fitted = True
                        return scaled_data.reshape(original_shape)
                    else:
                        return data
                
                def transform(self, data):
                    if not self._fitted:
                        raise ValueError("Scaler not fitted. Call fit_transform first.")
                    return self._scaler.transform(data)
            
            CumlScaler = EnhancedSklearnScalerWrapper
            log_statement('info', f"'{__name__}':INFO>>Using enhanced sklearn scaler wrapper.", Path(__file__).stem)
            
        except ImportError:
            class DummyScaler:
                def __init__(self, *args, **kwargs):
                    pass
                
                def fit_transform(self, data):
                    return data
                
                def transform(self, data):
                    return data
            
            CumlScaler = DummyScaler
            log_statement('warning', f"'{__name__}':WARNING>>No scaling library available. Using dummy scaler.", Path(__file__).stem)

# NLTK Setup with enhanced error handling
NLTK_AVAILABLE = False
lemmatizer = None
stop_words = set()

try:
    import nltk
    
    def download_nltk_data():
        """Enhanced NLTK data download with better error handling"""
        resources = {
            'corpora/wordnet': 'wordnet',
            'corpora/stopwords': 'stopwords',
            'tokenizers/punkt': 'punkt'
        }
        
        for path_fragment, resource_id in resources.items():
            try:
                nltk.data.find(path_fragment)
            except LookupError:
                try:
                    nltk.data.find(f"{path_fragment}.zip")
                except LookupError:
                    try:
                        log_statement('info', f"'{__name__}':INFO>>Downloading NLTK '{resource_id}' data...", Path(__file__).stem)
                        nltk.download(resource_id, quiet=True)
                    except Exception as e:
                        log_statement('warning', f"'{__name__}':WARNING>>Failed to download NLTK resource '{resource_id}': {e}", Path(__file__).stem)
    
    # Download NLTK data
    download_nltk_data()
    
    # Import NLTK components
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
    log_statement('info', f"'{__name__}':INFO>>NLTK components loaded successfully.", Path(__file__).stem)
    
except ImportError as e:
    log_statement('warning', f"'{__name__}':WARNING>>NLTK not available: {e}", Path(__file__).stem)
except Exception as e:
    log_statement('error', f"'{__name__}':ERROR>>NLTK setup failed: {e}", Path(__file__).stem, exc_info=True)

# Fallback NLTK components
if not NLTK_AVAILABLE:
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'):
            return word
    
    lemmatizer = DummyLemmatizer()
    stop_words = set()

# Enhanced logging setup
LOG_INS = f"'{__name__}':'{__file__}'"

# Progress tracking integration
try:
    from src.utils.progress_tracker import create_progress_tracker, EnhancedProgressTracker
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False
    
    class DummyProgressTracker:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total_items', 0)
            self.processed = 0
        
        def update(self, success=True, error_msg=None):
            self.processed += 1
        
        def finish(self):
            pass
        
        def get_statistics(self):
            return {
                'total_items': self.total,
                'processed_items': self.processed,
                'failed_items': 0,
                'process_rate': 0.0,
                'resources': []
            }
    
    def create_progress_tracker(*args, **kwargs):
        return DummyProgressTracker(*args, **kwargs)

# Configuration and State Management
@dataclass
class ProcessingConfig:
    """Enhanced configuration for data processing operations with system resource awareness"""
    max_workers: int = 16
    use_gpu: bool = True
    compression_enabled: bool = True
    compression_level: int = 22
    enable_semantic_labeling: bool = False
    enable_progress_tracking: bool = True
    batch_size: int = 1000
    timeout_seconds: int = 300
    retry_attempts: int = 3
    checkpoint_interval: int = 100
    
    # Text processing settings
    text_cleaning_regex: str = r'[^\w\s\-\.]'
    max_text_length: int = 10000000  # 10MB max text
    
    # File type settings
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.csv', '.json', '.jsonl', '.pdf', '.docx', '.html', '.xml', '.md'
    ])
    
    # Output settings
    processed_extension: str = '.processed'
    tokenized_extension: str = '.tokenized'
    
    # ADDED: System resource optimization settings
    use_system_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    dynamic_batch_sizing: bool = True
    resource_monitoring: bool = True
    adaptive_workers: bool = True
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if self.max_workers < 1:
            self.max_workers = 1
        if self.compression_level < 1 or self.compression_level > 22:
            self.compression_level = 22
        if self.batch_size < 1:
            self.batch_size = 1
        return True

class ProcessingMetrics:
    """Thread-safe metrics collection for processing operations"""
    
    def __init__(self):
        self._lock = RLock()
        self._metrics = {
            'files_processed': 0,
            'files_failed': 0,
            'bytes_processed': 0,
            'processing_time': 0.0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def start_processing(self):
        """Mark start of processing"""
        with self._lock:
            self._metrics['start_time'] = time.time()
    
    def end_processing(self):
        """Mark end of processing"""
        with self._lock:
            self._metrics['end_time'] = time.time()
            if self._metrics['start_time']:
                self._metrics['processing_time'] = self._metrics['end_time'] - self._metrics['start_time']
    
    def increment_processed(self, file_size: int = 0):
        """Increment processed file count and bytes"""
        with self._lock:
            self._metrics['files_processed'] += 1
            self._metrics['bytes_processed'] += file_size
    
    def increment_failed(self, error_msg: str = None):
        """Increment failed file count and add error"""
        with self._lock:
            self._metrics['files_failed'] += 1
            if error_msg:
                self._metrics['errors'].append({
                    'timestamp': time.time(),
                    'error': error_msg
                })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self._lock:
            return self._metrics.copy()
    
    def get_processing_rate(self) -> float:
        """Get files processed per second"""
        with self._lock:
            if self._metrics['processing_time'] > 0:
                return self._metrics['files_processed'] / self._metrics['processing_time']
            return 0.0

class DataProcessor:
    """
    Enhanced Data Processor with m1.py integration support
    
    Handles scanning, processing, and tokenizing data with:
    - Thread-safe operations
    - Enhanced error handling and recovery
    - GPU acceleration with CPU fallbacks
    - Progress tracking and metrics
    - Repository integration
    - Type-safe operations
    """
    
    def __init__(
        self,
        repo_path_override: Optional[Union[str, Path]] = None,
        repo_context: Optional[DataProcessingContext] = None,
        repo_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        max_workers: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[ProcessingConfig] = None,
        repo: Optional[RepoHandler] = None,
        context: Optional[Any] = None
    ):
        """
        Initialize DataProcessor with enhanced configuration and system resource optimization
        
        Args:
            repo_path_override: Override path for repository
            repo_dir: Repository directory
            filename: Repository filename
            max_workers: Maximum worker threads
            output_dir: Output directory for processed files
            config: Processing configuration
            repo: Existing RepoHandler instance
            context: DataProcessingContext from m1.py integration
        """
        # Initialize logging
        self.log_prefix = f"{LOG_INS}::{self.__class__.__name__}"
        
        # Validate path type
        if repo_path_override and not isinstance(repo_path_override, (str, Path)):
            raise TypeError(f"repo_path_override must be str or Path, got {type(repo_path_override)}")
        
        # Store context separately
        self.context = repo_context or DataProcessingContext()

        # Initialize configuration
        self.config = config or ProcessingConfig()
        self.config.validate()
        
        # Initialize metrics
        self.metrics = ProcessingMetrics()
        
        # Initialize locks for thread safety
        self.lock = RLock()
        self._processing_lock = Lock()
        
        # Store context for m1.py integration
        self.context = context
        
        # ADDED: Initialize system resource optimization
        self.system_resources = None
        self.resource_config = None
        if self.config.use_system_optimization:
            try:
                from src.utils.system_resources import get_system_resources, get_optimal_config
                self.system_resources = get_system_resources()
                self.resource_config = get_optimal_config('batch_processing')
                
                # Override configuration with system-optimized values if not explicitly set
                if max_workers is None and self.config.adaptive_workers:
                    self.config.max_workers = self.resource_config['worker_count']
                if self.config.memory_limit_gb is None:
                    self.config.memory_limit_gb = self.resource_config['memory_limit_gb']
                if self.config.dynamic_batch_sizing:
                    self.config.batch_size = self.resource_config['batch_size']
                    
                log_statement('info', f"{self.log_prefix}:INFO>>System optimization enabled: "
                            f"workers={self.config.max_workers}, batch_size={self.config.batch_size}, "
                            f"memory_limit={self.config.memory_limit_gb:.1f}GB", Path(__file__).stem)
            except ImportError:
                log_statement('warning', f"{self.log_prefix}:WARNING>>System optimization not available", 
                            Path(__file__).stem)
        
        log_statement('debug', f"{self.log_prefix}:DEBUG>>Initializing DataProcessor", Path(__file__).stem)
        
        try:
            # Setup repository
            self._setup_repository(repo_path_override, repo_dir, filename, repo)
            
            # Setup output directories
            self._setup_output_directories(output_dir)
            
            # Setup processing components
            self._setup_processing_components()
            
            # Setup executor with system-optimized worker count
            self._setup_executor(max_workers)
            
            log_statement('info', f"{self.log_prefix}:INFO>>DataProcessor initialized successfully"
                        f"{' with system optimization' if self.system_resources else ''}", Path(__file__).stem)
            
        except Exception as e:
            log_statement('critical', f"{self.log_prefix}:CRITICAL>>DataProcessor initialization failed: {e}", 
                        Path(__file__).stem, exc_info=True)
            raise

    def _setup_repository(
        self, 
        repo_path_override: Optional[Union[str, Path]], 
        repo_dir: Optional[Union[str, Path]], 
        filename: Optional[str],
        repo: Optional[RepoHandler]
    ):
        """Setup repository with enhanced error handling"""
        try:
            if repo is not None:
                # Use provided repository
                self.repo = repo
                self.repo_filepath = Path(repo.repository_path) if hasattr(repo, 'repository_path') else None
                log_statement('info', f"{self.log_prefix}:INFO>>Using provided RepoHandler instance", Path(__file__).stem)
                return
            
            # Determine repository path
            if not repo_dir and repo_path_override:
                try:
                    self.repo_filepath = Path(repo_path_override).resolve()
                    self.repo_dir = self.repo_filepath.parent
                    log_statement('info', f"{self.log_prefix}:INFO>>Using overridden repo path: {self.repo_filepath}", Path(__file__).stem)
                except Exception as path_e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>Invalid repo_path_override: {repo_path_override} - {path_e}", Path(__file__).stem)
                    raise ValueError("Invalid repository override path") from path_e
            else:
                # Use default paths
                try:
                    if PROJECT_IMPORTS_AVAILABLE and repo_dir:
                        self.repo_dir = Path(repo_dir).resolve()
                        self.repo_filepath = self.repo_dir / (filename or 'data_repository.csv.zst')
                    elif 'REPO_DIR' in globals():
                        self.repo_dir = Path(REPO_DIR).resolve()
                        self.repo_filepath = self.repo_dir / (filename or 'data_repository.csv.zst')
                    else:
                        self.repo_dir = Path('./repositories').resolve()
                        self.repo_filepath = self.repo_dir / 'data_repository.csv.zst'
                    
                    log_statement('info', f"{self.log_prefix}:INFO>>Using default repo path: {self.repo_filepath}", Path(__file__).stem)
                except Exception as e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>Failed to setup default repo path: {e}", Path(__file__).stem)
                    raise ValueError("Failed to setup repository path") from e
            
            # Ensure repository directory exists
            self.repo_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize repository handler
            if PROJECT_IMPORTS_AVAILABLE:
                try:
                    self.repo = RepoHandler(
                        metadata_compression='zst',
                        repository_path=self.repo_filepath
                    )
                    log_statement('info', f"{self.log_prefix}:INFO>>RepoHandler initialized", Path(__file__).stem)
                except Exception as repo_e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>RepoHandler initialization failed: {repo_e}", Path(__file__).stem, exc_info=True)
                    self.repo = None
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>RepoHandler not available, using basic repository", Path(__file__).stem)
                self.repo = None
                
        except Exception as e:
            log_statement('critical', f"{self.log_prefix}:CRITICAL>>Repository setup failed: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def _setup_output_directories(self, output_dir: Optional[Union[str, Path]]):
        """Setup output directories with enhanced error handling"""
        try:
            # Setup processed data directory
            if output_dir:
                if isinstance(output_dir, (str, Path)):
                    self.output_proc_dir = Path(output_dir).resolve()
                else:
                    raise ValueError(f"Invalid output_dir type: {type(output_dir)}")
            else:
                if PROJECT_IMPORTS_AVAILABLE and 'PROCESSED_DATA_DIR' in globals():
                    self.output_proc_dir = Path(PROCESSED_DATA_DIR).resolve()
                else:
                    self.output_proc_dir = Path('./data/processed').resolve()
            
            self.output_proc_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup tokenized data directory
            if PROJECT_IMPORTS_AVAILABLE and 'TOKENIZED_DATA_DIR' in globals():
                self.tokenized_output_dir = Path(TOKENIZED_DATA_DIR).resolve()
            else:
                self.tokenized_output_dir = Path('./data/tokenized').resolve()
            
            self.tokenized_output_dir.mkdir(parents=True, exist_ok=True)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Output directories setup: Processed='{self.output_proc_dir}', Tokenized='{self.tokenized_output_dir}'", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Output directory setup failed: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def _setup_processing_components(self):
        """Setup processing components with enhanced error handling"""
        try:
            # Setup scaler
            self.scaler = None
            if CumlScaler:
                try:
                    self.scaler = CumlScaler()
                    log_statement('info', f"{self.log_prefix}:INFO>>Scaler initialized: {type(self.scaler).__name__}", Path(__file__).stem)
                except Exception as scaler_e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Scaler initialization failed: {scaler_e}", Path(__file__).stem)
            
            # Setup text cleaning regex
            self.cleaning_regex = None
            try:
                if hasattr(self.config, 'text_cleaning_regex') and self.config.text_cleaning_regex:
                    self.cleaning_regex = re.compile(self.config.text_cleaning_regex)
                    log_statement('info', f"{self.log_prefix}:INFO>>Text cleaning regex compiled", Path(__file__).stem)
            except Exception as regex_e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to compile regex: {regex_e}", Path(__file__).stem)
            
            # Setup encoding
            self.encoding = "utf-8"
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Processing components setup failed: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def _setup_executor(self, max_workers: Optional[int]):
        """Setup thread pool executor with enhanced configuration"""
        try:
            resolved_max_workers = max_workers or self.config.max_workers
            self.max_workers = max(1, resolved_max_workers)
            
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='DataProcessor'
            )
            
            log_statement('info', f"{self.log_prefix}:INFO>>Executor initialized with {self.max_workers} workers", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Executor setup failed: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with proper resource management"""
        try:
            log_statement('info', f"{self.log_prefix}:INFO>>Starting cleanup", Path(__file__).stem)
            
            # Shutdown executor
            if hasattr(self, 'executor') and self.executor:
                try:
                    self.executor.shutdown(wait=True, timeout=30)
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Executor shutdown completed", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Executor shutdown error: {e}", Path(__file__).stem)
            
            # Save repository if available
            if hasattr(self, 'repo') and self.repo and hasattr(self.repo, 'save'):
                try:
                    self.repo.save()
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Repository saved", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Repository save error: {e}", Path(__file__).stem)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Cleanup completed", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Cleanup error: {e}", Path(__file__).stem, exc_info=True)
    
    def __del__(self):
        """Destructor with cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor
    
    # Enhanced file classification with comprehensive type detection
    def _classify_data(self, filepath: Path, content: Union[str, bytes, None] = None) -> str:
        """
        Enhanced data classification with comprehensive type detection
        
        Args:
            filepath: Path to the file
            content: Optional file content for analysis
            
        Returns:
            Classified data type string
        """
        log_statement('debug', f"{self.log_prefix}:DEBUG>>Classifying data for {filepath.name}", Path(__file__).stem)
        
        try:
            extension = filepath.suffix.lower()
            classified_type = TYPE_UNKNOWN
            
            # Stage 1: Extension-based classification
            extension_map = {
                '.pdf': TYPE_PDF,
                '.html': TYPE_HTML, '.htm': TYPE_HTML,
                '.xml': TYPE_XML,
                '.json': TYPE_JSON,
                '.jsonl': TYPE_JSONL,
                '.yaml': TYPE_YAML, '.yml': TYPE_YAML,
                '.md': TYPE_MARKDOWN,
                '.csv': TYPE_TABULAR, '.tsv': TYPE_TABULAR,
                '.txt': TYPE_TEXTUAL,
                '.docx': TYPE_DOCX, '.doc': TYPE_DOC,
                '.xlsx': TYPE_EXCEL, '.xls': TYPE_EXCEL,
                '.png': TYPE_IMAGE_PNG,
                '.jpg': TYPE_IMAGE_JPEG, '.jpeg': TYPE_IMAGE_JPEG,
                '.gif': TYPE_IMAGE_GIF,
                '.bmp': TYPE_IMAGE_BMP,
                '.wav': TYPE_AUDIO_WAV,
                '.mp3': TYPE_AUDIO_MP3,
                '.zip': TYPE_COMPRESSED_ZIP,
                '.gz': TYPE_COMPRESSED_GZIP,
                '.bz2': TYPE_COMPRESSED_BZ2,
                '.7z': TYPE_COMPRESSED_7Z,
                '.rar': TYPE_COMPRESSED_RAR,
                '.zst': TYPE_COMPRESSED_ZSTD
            }
            
            # Code file extensions
            code_extensions = {
                '.py', '.pyw', '.java', '.js', '.ts', '.c', '.cpp', '.h', '.hpp',
                '.cs', '.go', '.php', '.rb', '.pl', '.sh', '.ps1', '.lua',
                '.swift', '.kt', '.kts', '.rs', '.scala', '.r'
            }
            
            if extension in extension_map:
                classified_type = extension_map[extension]
            elif extension in code_extensions:
                classified_type = TYPE_CODE
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Extension-based classification: {classified_type}", Path(__file__).stem)
            
            # Stage 2: Content-based analysis (if content provided)
            if content is not None and len(content) > 0:
                classified_type = self._analyze_content(content, filepath, classified_type)
            
            # Stage 3: File size and existence checks
            if classified_type == TYPE_UNKNOWN:
                try:
                    if filepath.exists() and filepath.stat().st_size == 0:
                        classified_type = TYPE_EMPTY
                except Exception:
                    pass
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Final classification for {filepath.name}: {classified_type}", Path(__file__).stem)
            return classified_type
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Classification error for {filepath.name}: {e}", Path(__file__).stem, exc_info=True)
            return TYPE_UNKNOWN
    
    def _analyze_content(self, content: Union[str, bytes], filepath: Path, initial_type: str) -> str:
        """Enhanced content analysis for classification"""
        try:
            # Handle bytes content
            content_bytes = None
            content_str = None
            
            if isinstance(content, bytes):
                content_bytes = content
                # Try to decode for text analysis
                try:
                    content_str = content.decode('utf-8', errors='ignore')
                except Exception:
                    try:
                        content_str = content.decode('latin-1', errors='ignore')
                    except Exception:
                        pass
            else:
                content_str = str(content)
                content_bytes = content_str.encode('utf-8', errors='ignore')
            
            # Magic number detection for bytes
            if content_bytes and len(content_bytes) > 16:
                magic_type = self._detect_magic_numbers(content_bytes)
                if magic_type != TYPE_UNKNOWN:
                    return magic_type
            
            # Text content analysis
            if content_str and len(content_str.strip()) > 0:
                return self._analyze_text_content(content_str, filepath, initial_type)
            
            # Binary content
            if content_bytes and content_str is None:
                return TYPE_BINARY
                
            return initial_type
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Content analysis error: {e}", Path(__file__).stem)
            return initial_type
    
    def _detect_magic_numbers(self, content_bytes: bytes) -> str:
        """Detect file type by magic numbers"""
        magic_signatures = {
            b'%PDF-': TYPE_PDF,
            b'\x89PNG\r\n\x1a\n': TYPE_IMAGE_PNG,
            b'\xff\xd8\xff': TYPE_IMAGE_JPEG,
            b'GIF87a': TYPE_IMAGE_GIF,
            b'GIF89a': TYPE_IMAGE_GIF,
            b'BM': TYPE_IMAGE_BMP,
            b'RIFF': TYPE_AUDIO_WAV,  # Check for WAVE later
            b'ID3': TYPE_AUDIO_MP3,
            b'PK\x03\x04': TYPE_COMPRESSED_ZIP,
            b'PK\x05\x06': TYPE_COMPRESSED_ZIP,
            b'\x1f\x8b': TYPE_COMPRESSED_GZIP,
            b'BZh': TYPE_COMPRESSED_BZ2,
            b'7z\xBC\xAF\x27\x1C': TYPE_COMPRESSED_7Z,
            b'Rar!\x1a\x07\x00': TYPE_COMPRESSED_RAR,
            b'\x28\xB5\x2F\xFD': TYPE_COMPRESSED_ZSTD,
            b'\xFD\x2F\xB5\x28': TYPE_COMPRESSED_ZSTD
        }
        
        for signature, file_type in magic_signatures.items():
            if content_bytes.startswith(signature):
                # Special check for WAV files
                if file_type == TYPE_AUDIO_WAV and len(content_bytes) > 12:
                    if content_bytes[8:12] == b'WAVE':
                        return TYPE_AUDIO_WAV
                    else:
                        continue
                return file_type
        
        return TYPE_UNKNOWN
    
    def _analyze_text_content(self, content_str: str, filepath: Path, initial_type: str) -> str:
        """Analyze text content for classification"""
        try:
            # Limit analysis to first 10KB for performance
            analysis_sample = content_str[:10240]
            lines = analysis_sample.split('\n')[:100]  # First 100 lines
            
            # JSON detection
            stripped_content = analysis_sample.strip()
            if stripped_content.startswith(('{', '[')) and stripped_content.endswith(('}', ']')):
                try:
                    json.loads(analysis_sample)
                    return TYPE_JSON
                except json.JSONDecodeError:
                    pass
            
            # JSONL detection
            if lines and len(lines) > 1:
                jsonl_count = 0
                for line in lines[:10]:  # Check first 10 lines
                    line = line.strip()
                    if line and line.startswith('{') and line.endswith('}'):
                        try:
                            json.loads(line)
                            jsonl_count += 1
                        except json.JSONDecodeError:
                            break
                if jsonl_count >= 2:  # At least 2 valid JSON lines
                    return TYPE_JSONL
            
            # XML/HTML detection
            if re.search(r'<\?xml\s+version=', analysis_sample, re.IGNORECASE):
                return TYPE_XML
            elif re.search(r'<!DOCTYPE html|<html[^>]*>|<head[^>]*>|<body[^>]*>', analysis_sample, re.IGNORECASE):
                return TYPE_HTML
            elif re.search(r'<(\w+)[^>]*>.*?</\1>', analysis_sample, re.DOTALL):
                return TYPE_XML
            
            # CSV detection using sniffer
            try:
                sample_lines = '\n'.join(lines[:20])
                if len(sample_lines) > 100:
                    dialect = csv.Sniffer().sniff(sample_lines[:1024], delimiters=',\t;|')
                    if dialect.delimiter:
                        # Check consistency
                        delimiter_counts = [line.count(dialect.delimiter) for line in lines[:10] if line.strip()]
                        if len(set(delimiter_counts)) <= 2 and max(delimiter_counts, default=0) > 0:
                            return TYPE_TABULAR
            except (csv.Error, Exception):
                pass
            
            # Code detection
            code_patterns = [
                r'\bdef\s+\w+\s*\(',  # Python functions
                r'\bclass\s+\w+',  # Class definitions
                r'\bimport\s+\w+',  # Import statements
                r'\bfunction\s+\w+\s*\(',  # JavaScript functions
                r'\bpublic\s+class\s+\w+',  # Java classes
                r'#include\s*<',  # C/C++ includes
                r'\bif\s*\([^)]+\)\s*{',  # Conditional blocks
                r'\bfor\s*\([^)]+\)\s*{',  # For loops
            ]
            
            code_score = sum(1 for pattern in code_patterns if re.search(pattern, analysis_sample))
            if code_score >= 2:
                return TYPE_CODE
            
            # Markdown detection
            md_patterns = [
                r'^\s*#+\s+',  # Headers
                r'^\s*[\*\-\+]\s+',  # List items
                r'^\s*\d+\.\s+',  # Numbered lists
                r'^\s*>',  # Blockquotes
                r'```',  # Code blocks
                r'\[.+?\]\(.+?\)',  # Links
                r'\*\*.*?\*\*',  # Bold
                r'__.*?__',  # Bold alternative
            ]
            
            md_score = sum(1 for pattern in md_patterns if re.search(pattern, analysis_sample, re.MULTILINE))
            if md_score >= 2:
                return TYPE_MARKDOWN
            
            # Default to textual if decodable
            return TYPE_TEXTUAL
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Text content analysis error: {e}", Path(__file__).stem)
            return initial_type

    # Enhanced file processing methods
    def process_all(
        self, 
        base_dir_filter: Optional[Path] = None, 
        statuses_to_process: Tuple[str, ...] = (STATUS_NEW, STATUS_DISCOVERED, STATUS_ERROR),
        max_files: Optional[int] = None
    ) -> OperationResult:
        """
        Enhanced batch processing with comprehensive error handling, progress tracking, and system optimization
        
        Args:
            base_dir_filter: Optional directory filter
            statuses_to_process: File statuses to process
            max_files: Maximum number of files to process
            
        Returns:
            OperationResult with processing statistics
        """
        def _do_process_all():
            log_statement('info', f"{self.log_prefix}:INFO>>Starting batch processing - Filter: {base_dir_filter}, Statuses: {statuses_to_process}", Path(__file__).stem)
            
            # Initialize metrics
            self.metrics.start_processing()
            
            # ADDED: Get system-optimized configuration
            if self.config.use_system_optimization and self.system_resources:
                try:
                    from src.utils.system_resources import get_optimal_config
                    # Get file count estimate for optimization
                    files_to_process = self._get_files_to_process(base_dir_filter, statuses_to_process, max_files)
                    file_count = len(files_to_process)
                    
                    # Get optimized config
                    operation_config = get_optimal_config('batch_processing', file_count)
                    
                    # Update configuration dynamically
                    if self.config.adaptive_workers:
                        effective_workers = operation_config['worker_count']
                        log_statement('info', f"{self.log_prefix}:INFO>>Adjusting workers from {self.config.max_workers} to {effective_workers} based on workload", 
                                    Path(__file__).stem)
                    else:
                        effective_workers = self.config.max_workers
                        
                    if self.config.dynamic_batch_sizing:
                        effective_batch_size = operation_config['batch_size']
                        log_statement('info', f"{self.log_prefix}:INFO>>Adjusting batch size from {self.config.batch_size} to {effective_batch_size} based on workload", 
                                    Path(__file__).stem)
                    else:
                        effective_batch_size = self.config.batch_size
                        
                    memory_limit_gb = operation_config['memory_limit_gb']
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to optimize configuration: {e}", 
                                Path(__file__).stem)
                    files_to_process = self._get_files_to_process(base_dir_filter, statuses_to_process, max_files)
                    effective_workers = self.config.max_workers
                    effective_batch_size = self.config.batch_size
                    memory_limit_gb = self.config.memory_limit_gb or 4.0
            else:
                files_to_process = self._get_files_to_process(base_dir_filter, statuses_to_process, max_files)
                effective_workers = self.config.max_workers
                effective_batch_size = self.config.batch_size
                memory_limit_gb = self.config.memory_limit_gb or 4.0
            
            try:
                if not files_to_process:
                    log_statement('info', f"{self.log_prefix}:INFO>>No files found matching criteria", Path(__file__).stem)
                    return {
                        'processed_count': 0,
                        'failed_count': 0,
                        'total_files': 0,
                        'message': 'No files to process'
                    }
                
                # ADDED: Enhanced progress tracking with resource monitoring
                progress_tracker = None
                if self.config.enable_progress_tracking:
                    try:
                        from src.utils.progress_tracker import create_progress_tracker
                        progress_tracker = create_progress_tracker(
                            total_items=len(files_to_process),
                            description=f"Processing files [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]",
                            unit="files",
                            show_resources=self.config.resource_monitoring,
                            update_interval=0.5  # More responsive updates
                        )
                        log_statement('info', f"{self.log_prefix}:INFO>>Enhanced progress tracking enabled with resource monitoring", 
                                    Path(__file__).stem)
                    except ImportError:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Enhanced progress tracking not available", 
                                    Path(__file__).stem)
                        # Fallback to basic progress tracking
                        if PROGRESS_TRACKING_AVAILABLE:
                            progress_tracker = create_progress_tracker(
                                total_items=len(files_to_process),
                                description=f"Processing files [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]",
                                unit="files",
                                show_resources=False
                            )
                
                # Process files with enhanced error handling and memory management
                results = self._process_files_batch(files_to_process, progress_tracker, 
                                                effective_workers, effective_batch_size, memory_limit_gb)
                
                # Finalize metrics
                self.metrics.end_processing()
                
                # Save repository
                if self.repo and hasattr(self.repo, 'save'):
                    try:
                        self.repo.save()
                    except Exception as save_e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Repository save failed: {save_e}", Path(__file__).stem)
                
                # Compile final results
                final_metrics = self.metrics.get_metrics()
                result = {
                    'processed_count': final_metrics['files_processed'],
                    'failed_count': final_metrics['files_failed'],
                    'total_files': len(files_to_process),
                    'bytes_processed': final_metrics['bytes_processed'],
                    'processing_time': final_metrics['processing_time'],
                    'processing_rate': self.metrics.get_processing_rate(),
                    'successful_files': results.get('successful_files', []),
                    'failed_files': results.get('failed_files', []),
                    'errors': final_metrics['errors'][-10:],  # Last 10 errors
                    'system_optimized': self.config.use_system_optimization,
                    'effective_workers': effective_workers,
                    'effective_batch_size': effective_batch_size,
                    'memory_limit_gb': memory_limit_gb
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Batch processing completed: {result['processed_count']} processed, {result['failed_count']} failed"
                            f"{' (system-optimized)' if self.config.use_system_optimization else ''}", Path(__file__).stem)
                
                return result
                
            except Exception as e:
                self.metrics.end_processing()
                log_statement('error', f"{self.log_prefix}:ERROR>>Batch processing failed: {e}", Path(__file__).stem, exc_info=True)
                raise
            finally:
                if progress_tracker:
                    progress_tracker.finish()
        
        return safe_operation("process_all", _do_process_all)
    
    def _get_files_to_process(
        self, 
        base_dir_filter: Optional[Path], 
        statuses_to_process: Tuple[str, ...],
        max_files: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get files ready for processing with enhanced filtering"""
        try:
            if not self.repo:
                log_statement('warning', f"{self.log_prefix}:WARNING>>No repository available for file retrieval", Path(__file__).stem)
                return []
            
            # Get files by status
            if hasattr(self.repo, 'get_files_by_status'):
                file_paths = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
            else:
                # Fallback for basic repository
                file_paths = []
            
            if not file_paths:
                return []
            
            # Get detailed file information
            file_info_list = []
            try:
                with self.lock:
                    if hasattr(self.repo, 'df') and self.repo.df is not None:
                        files_str_set = {str(p.resolve()) for p in file_paths}
                        matching_rows = self.repo.df[self.repo.df[COL_FILEPATH].isin(files_str_set)]
                        if not matching_rows.empty:
                            file_info_list = matching_rows.to_dict('records')
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to get detailed file info: {e}", Path(__file__).stem)
                # Fallback: create basic file info
                file_info_list = [
                    {COL_FILEPATH: str(fp), COL_STATUS: STATUS_NEW}
                    for fp in file_paths
                ]
            
            # Apply max_files limit
            if max_files and len(file_info_list) > max_files:
                file_info_list = file_info_list[:max_files]
                log_statement('info', f"{self.log_prefix}:INFO>>Limited processing to {max_files} files", Path(__file__).stem)
            
            return file_info_list
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error getting files to process: {e}", Path(__file__).stem, exc_info=True)
            return []

    def _process_files_batch(
        self, 
        files_to_process: List[Dict[str, Any]], 
        progress_tracker: Optional[Any],
        effective_workers: int,
        effective_batch_size: int,
        memory_limit_gb: float
    ) -> Dict[str, Any]:
        """Process files in batch with enhanced error handling, recovery, and memory management"""
        successful_files = []
        failed_files = []
        
        # ADDED: Memory monitoring for adaptive processing
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)  # GB
        memory_threshold = initial_memory + (memory_limit_gb * 0.8)  # 80% of limit
        
        # Create thread pool with system-optimized worker count
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix='DataProc') as executor:
            # Process in batches for memory efficiency
            batches = [files_to_process[i:i + effective_batch_size] 
                    for i in range(0, len(files_to_process), effective_batch_size)]
            
            log_statement('info', f"{self.log_prefix}:INFO>>Processing {len(files_to_process)} files in {len(batches)} batches "
                        f"(batch_size={effective_batch_size}, workers={effective_workers})", Path(__file__).stem)
            
            for batch_idx, batch in enumerate(batches):
                # Check memory before processing batch
                current_memory = process.memory_info().rss / (1024**3)
                if current_memory > memory_threshold:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>High memory usage detected: {current_memory:.1f}GB / {memory_limit_gb:.1f}GB limit. "
                                "Forcing garbage collection.", Path(__file__).stem)
                    import gc
                    gc.collect()
                    time.sleep(0.1)  # Brief pause for memory cleanup
                    
                    # Re-check memory
                    current_memory = process.memory_info().rss / (1024**3)
                    if current_memory > memory_threshold:
                        # Reduce batch size dynamically
                        reduced_batch_size = max(1, effective_batch_size // 2)
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Reducing batch size from {effective_batch_size} to {reduced_batch_size} due to memory pressure", 
                                    Path(__file__).stem)
                        # Re-batch remaining files
                        remaining_files = sum(batches[batch_idx:], [])
                        batches = batches[:batch_idx] + [remaining_files[i:i + reduced_batch_size] 
                                                        for i in range(0, len(remaining_files), reduced_batch_size)]
                        effective_batch_size = reduced_batch_size
                
                # Submit batch processing tasks
                futures = []
                for file_info in batch:
                    try:
                        future = executor.submit(self._process_file_with_retry, file_info)
                        futures.append((future, file_info))
                    except Exception as submit_e:
                        log_statement('error', f"{self.log_prefix}:ERROR>>Failed to submit processing task: {submit_e}", Path(__file__).stem)
                        failed_files.append(file_info.get(COL_FILEPATH, 'Unknown'))
                        self.metrics.increment_failed(str(submit_e))
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=str(submit_e))
                
               # Process results with timeout and error handling
                for future, file_info in futures:
                    file_path = file_info.get(COL_FILEPATH, 'Unknown')
                    file_name = Path(file_path).name
                    
                    try:
                        # Wait for result with timeout
                        result_info = future.result(timeout=self.config.timeout_seconds)
                        
                        if result_info and isinstance(result_info, dict):
                            final_status = result_info.get(COL_STATUS, STATUS_ERROR)
                            
                            if final_status == STATUS_PROCESSED:
                                successful_files.append(file_path)
                                file_size = self._get_file_size_safe(Path(file_path))
                                self.metrics.increment_processed(file_size)
                                
                                if progress_tracker:
                                    progress_tracker.update(success=True)
                                
                                log_statement('debug', f"{self.log_prefix}:DEBUG>>Successfully processed: {file_name}", Path(__file__).stem)
                            else:
                                failed_files.append(file_path)
                                error_msg = result_info.get(COL_ERROR, "Unknown error")
                                self.metrics.increment_failed(f"{file_name}: {error_msg}")
                                
                                if progress_tracker:
                                    progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                                
                                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to process {file_name}: {error_msg}", Path(__file__).stem)
                        else:
                            failed_files.append(file_path)
                            error_msg = "Invalid processing result"
                            self.metrics.increment_failed(f"{file_name}: {error_msg}")
                            
                            if progress_tracker:
                                progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                        
                    except TimeoutError:
                        failed_files.append(file_path)
                        error_msg = f"Processing timeout ({self.config.timeout_seconds}s)"
                        self.metrics.increment_failed(f"{file_name}: {error_msg}")
                        
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>Processing timeout for {file_name}", Path(__file__).stem)
                        
                    except Exception as e:
                        failed_files.append(file_path)
                        error_msg = str(e)
                        self.metrics.increment_failed(f"{file_name}: {error_msg}")
                        
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>Exception processing {file_name}: {e}", Path(__file__).stem, exc_info=True)
                
                # Log batch completion and memory status
                current_memory = process.memory_info().rss / (1024**3)
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Batch {batch_idx + 1}/{len(batches)} completed. "
                            f"Memory: {current_memory:.1f}GB, Success: {len(successful_files)}, Failed: {len(failed_files)}", 
                            Path(__file__).stem)
        
        return {
            'successful_files': successful_files,
            'failed_files': failed_files
        }
    
    def _process_file_with_retry(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process file with retry logic and enhanced error recovery"""
        file_path = file_info.get(COL_FILEPATH, '')
        file_name = Path(file_path).name if file_path else 'Unknown'
        
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                if attempt > 0:
                    log_statement('info', f"{self.log_prefix}:INFO>>Retry attempt {attempt + 1} for {file_name}", Path(__file__).stem)
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10 seconds
                
                result = self._process_file(file_info)
                
                if result and result.get(COL_STATUS) == STATUS_PROCESSED:
                    return result
                else:
                    last_error = result.get(COL_ERROR, 'Unknown error') if result else 'No result returned'
                    
            except Exception as e:
                last_error = str(e)
                log_statement('warning', f"{self.log_prefix}:WARNING>>Attempt {attempt + 1} failed for {file_name}: {e}", Path(__file__).stem)
                
                if attempt == self.config.retry_attempts - 1:
                    log_statement('error', f"{self.log_prefix}:ERROR>>All retry attempts failed for {file_name}", Path(__file__).stem, exc_info=True)
        
        # All attempts failed
        return {
            COL_FILEPATH: file_path,
            COL_STATUS: STATUS_FAILED,
            COL_ERROR: f"All {self.config.retry_attempts} attempts failed. Last error: {last_error}"
        }
    
    def _process_file(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced single file processing with comprehensive error handling
        
        Args:
            file_info: Dictionary containing file metadata
            
        Returns:
            Updated file metadata dictionary or None on critical error
        """
        file_path_str = file_info.get(COL_FILEPATH)
        if not file_path_str:
            log_statement('error', f"{self.log_prefix}:ERROR>>Missing file path in file_info", Path(__file__).stem)
            return {
                COL_STATUS: STATUS_ERROR,
                COL_ERROR: "Missing file path"
            }
        
        file_path = Path(file_path_str)
        file_name = file_path.name
        
        # Initialize updated info
        updated_info = file_info.copy()
        updated_info[COL_STATUS] = STATUS_PROCESSING
        updated_info[COL_ERROR] = ''
        
        log_statement('debug', f"{self.log_prefix}:DEBUG>>Processing file: {file_name}", Path(__file__).stem)
        
        try:
            # Update repository status
            self._update_file_status_safe(file_path, STATUS_PROCESSING)
            
            # Validate file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Setup output paths
            output_info = self._setup_output_paths(file_path)
            
            # Read file content
            content = self._read_file_content_safe(file_path)
            if content is None:
                raise ValueError("Failed to read file content")
            
            # Classify data
            data_type = self._classify_data(file_path, content)
            updated_info[COL_DATA_CLASSIFICATION] = data_type
            
            if data_type == TYPE_EMPTY:
                updated_info[COL_STATUS] = STATUS_SKIPPED
                updated_info[COL_ERROR] = "Empty file"
                self._update_file_status_safe(file_path, STATUS_SKIPPED, "Empty file")
                return updated_info
            
            # Process based on data type
            processing_result = self._process_by_type(content, data_type, file_path, output_info)
            
            if processing_result and processing_result.get('success'):
                # Successfully processed
                updated_info[COL_STATUS] = STATUS_PROCESSED
                updated_info[COL_PROCESSED_PATH] = processing_result.get('output_path', '')
                updated_info[COL_PROCESSED_FILENAME] = processing_result.get('output_filename', '')
                updated_info[COL_DATA_HASH] = processing_result.get('data_hash', '')
                updated_info[COL_FINAL_CLASSIFICATION] = data_type
                updated_info[COL_ERROR] = ''
                
                # Update repository
                self._update_file_status_safe(file_path, STATUS_PROCESSED, "Processing completed successfully")
                
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Successfully processed {file_name}", Path(__file__).stem)
            else:
                # Processing failed
                error_msg = processing_result.get('error', 'Processing failed') if processing_result else 'No processing result'
                updated_info[COL_STATUS] = STATUS_FAILED
                updated_info[COL_ERROR] = error_msg
                updated_info[COL_FINAL_CLASSIFICATION] = data_type
                
                # Update repository
                self._update_file_status_safe(file_path, STATUS_FAILED, error_msg)
                
                log_statement('warning', f"{self.log_prefix}:WARNING>>Processing failed for {file_name}: {error_msg}", Path(__file__).stem)
            
            return updated_info
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            updated_info[COL_STATUS] = STATUS_ERROR
            updated_info[COL_ERROR] = error_msg
            
            # Update repository
            self._update_file_status_safe(file_path, STATUS_ERROR, error_msg)
            
            log_statement('error', f"{self.log_prefix}:ERROR>>Error processing {file_name}: {e}", Path(__file__).stem, exc_info=True)
            return updated_info

    def _setup_output_paths(self, input_path: Path) -> Dict[str, Path]:
        """Setup output paths with enhanced directory structure"""
        try:
            # Calculate relative path from input
            try:
                # Try to get relative path from a known base
                relative_path = input_path.relative_to(input_path.anchor)
            except ValueError:
                # Fallback: use the path as-is
                relative_path = Path(*input_path.parts[1:]) if len(input_path.parts) > 1 else input_path
            
            # Create output directory structure
            output_dir = self.output_proc_dir / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filenames
            base_name = input_path.stem
            output_base = output_dir / base_name
            
            return {
                'output_dir': output_dir,
                'output_base': output_base,
                'processed_path': output_base.with_suffix(f'{self.config.processed_extension}.zst'),
                'tokenized_path': output_base.with_suffix(f'{self.config.tokenized_extension}.zst')
            }
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Output path setup failed for {input_path}: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def _read_file_content_safe(self, file_path: Path) -> Optional[Union[str, bytes]]:
        """Safely read file content with enhanced encoding detection"""
        try:
            # Try to use project readers first
            if PROJECT_IMPORTS_AVAILABLE:
                try:
                    reader_class = get_reader_class(file_path.suffix)
                    if reader_class:
                        if issubclass(reader_class, RobustTextReader):
                            reader = reader_class(
                                filepath=file_path,
                                default_encoding='utf-8',
                                error_handling='replace',
                                detect_encoding=True
                            )
                        else:
                            reader = reader_class(filepath=file_path)
                        
                        content = reader.read()
                        if content is not None:
                            return content
                except Exception as reader_e:
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Reader failed for {file_path.name}: {reader_e}", Path(__file__).stem)
            
            # Fallback to basic file reading
            file_size = file_path.stat().st_size
            
            # Handle large files
            if file_size > self.config.max_text_length:
                log_statement('warning', f"{self.log_prefix}:WARNING>>File {file_path.name} is large ({file_size} bytes), reading first {self.config.max_text_length} bytes", Path(__file__).stem)
                with open(file_path, 'rb') as f:
                    content_bytes = f.read(self.config.max_text_length)
            else:
                content_bytes = file_path.read_bytes()
            
            # Try to decode as text
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    return content_bytes.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # Return as bytes if can't decode
            return content_bytes
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to read {file_path.name}: {e}", Path(__file__).stem, exc_info=True)
            return None
    
    def _process_by_type(
        self, 
        content: Union[str, bytes], 
        data_type: str, 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Process content based on classified data type"""
        try:
            if data_type == TYPE_TEXTUAL:
                return self._process_textual_content(content, input_path, output_info)
            elif data_type == TYPE_NUMERICAL:
                return self._process_numerical_content(content, input_path, output_info)
            elif data_type in [TYPE_PDF, TYPE_DOC, TYPE_DOCX, TYPE_HTML, TYPE_XML, TYPE_MARKDOWN]:
                return self._process_document_content(content, data_type, input_path, output_info)
            elif data_type in [TYPE_JSON, TYPE_JSONL]:
                return self._process_structured_content(content, data_type, input_path, output_info)
            elif data_type in [TYPE_CSV, TYPE_TABULAR]:
                return self._process_tabular_content(content, input_path, output_info)
            elif data_type == TYPE_CODE:
                return self._process_code_content(content, input_path, output_info)
            else:
                # Unsupported type, treat as textual
                log_statement('warning', f"{self.log_prefix}:WARNING>>Unsupported type {data_type}, treating as textual", Path(__file__).stem)
                return self._process_textual_content(content, input_path, output_info)
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Type-specific processing failed for {data_type}: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_textual_content(
        self, 
        content: Union[str, bytes], 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Enhanced textual content processing with semantic labeling support"""
        try:
            # Convert to string if bytes
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='replace')
            else:
                text_content = str(content)
            
            # Basic text cleaning
            cleaned_text = self._clean_text(text_content)
            
            # Semantic labeling (if enabled and available)
            semantic_data = None
            if self.config.enable_semantic_labeling and self.context:
                try:
                    semantic_labeler = self.context.container.get_semantic_labeler()
                    if semantic_labeler and hasattr(semantic_labeler, '_label_text_semantically'):
                        semantic_data = semantic_labeler._label_text_semantically(cleaned_text, input_path)
                except Exception as sem_e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Semantic labeling failed: {sem_e}", Path(__file__).stem)
            
            # Prepare output data
            if semantic_data:
                output_data = semantic_data
            else:
                output_data = {
                    'cleaned_text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text),
                    'processing_timestamp': dt.now(timezone.utc).isoformat()
                }
            
            # Save processed data
            output_path = output_info['processed_path']
            save_success = self._save_json_compressed(output_data, output_path)
            
            if save_success:
                # Generate hash
                data_hash = generate_data_hash(output_path)
                relative_path = output_path.relative_to(self.output_proc_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_path),
                    'output_filename': output_path.name,
                    'data_hash': data_hash,
                    'content_type': 'textual',
                    'semantic_labeled': semantic_data is not None
                }
            else:
                return {'success': False, 'error': 'Failed to save processed data'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Textual processing failed: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_numerical_content(
        self, 
        content: Union[str, bytes], 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process numerical/tabular content with scaling and analysis"""
        try:
            # Convert content to DataFrame
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='replace')
            else:
                content_str = str(content)
            
            # Try to parse as CSV
            try:
                df = pd.read_csv(io.StringIO(content_str))
            except Exception:
                # Fallback: treat as space/tab separated
                lines = content_str.strip().split('\n')
                data = []
                for line in lines:
                    row = re.split(r'\s+', line.strip())
                    if row:
                        data.append(row)
                if data:
                    df = pd.DataFrame(data[1:], columns=data[0] if len(data) > 1 else None)
                else:
                    raise ValueError("Could not parse numerical content")
            
            # Convert to numeric where possible
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
            numeric_df = numeric_df.dropna(axis=1, how='all')
            
            if numeric_df.empty:
                return {'success': False, 'error': 'No numeric data found'}
            
            # Apply scaling if scaler available
            processed_df = numeric_df
            if self.scaler:
                try:
                    processed_df = pd.DataFrame(
                        self.scaler.fit_transform(numeric_df),
                        columns=numeric_df.columns,
                        index=numeric_df.index
                    )
                except Exception as scale_e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Scaling failed: {scale_e}", Path(__file__).stem)
            
            # Save as compressed Parquet
            output_path = output_info['processed_path'].with_suffix('.parquet.zst')
            save_success = self._save_dataframe_compressed(processed_df, output_path)
            
            if save_success:
                data_hash = generate_data_hash(output_path)
                relative_path = output_path.relative_to(self.output_proc_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_path),
                    'output_filename': output_path.name,
                    'data_hash': data_hash,
                    'content_type': 'numerical',
                    'shape': processed_df.shape,
                    'columns': list(processed_df.columns)
                }
            else:
                return {'success': False, 'error': 'Failed to save processed data'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Numerical processing failed: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_document_content(
        self, 
        content: Union[str, bytes], 
        doc_type: str, 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process document content (PDF, DOC, HTML, etc.)"""
        try:
            # Extract text content based on document type
            if doc_type == TYPE_PDF:
                text_content = self._extract_pdf_text(content, input_path)
            elif doc_type in [TYPE_DOC, TYPE_DOCX]:
                text_content = self._extract_doc_text(content, input_path)
            elif doc_type in [TYPE_HTML, TYPE_XML]:
                text_content = self._extract_markup_text(content)
            elif doc_type == TYPE_MARKDOWN:
                text_content = self._extract_markdown_text(content)
            else:
                # Fallback to treating as plain text
                text_content = content.decode('utf-8', errors='replace') if isinstance(content, bytes) else str(content)
            
            if not text_content or not text_content.strip():
                return {'success': False, 'error': f'No text extracted from {doc_type} document'}
            
            # Process as textual content
            return self._process_textual_content(text_content, input_path, output_info)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Document processing failed for {doc_type}: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_structured_content(
        self, 
        content: Union[str, bytes], 
        struct_type: str, 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process structured content (JSON, JSONL)"""
        try:
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='replace')
            else:
                content_str = str(content)
            
            processed_data = None
            
            if struct_type == TYPE_JSON:
                # Parse and potentially restructure JSON
                try:
                    json_data = json.loads(content_str)
                    processed_data = {
                        'original_data': json_data,
                        'data_type': 'json',
                        'keys': list(json_data.keys()) if isinstance(json_data, dict) else None,
                        'length': len(json_data) if isinstance(json_data, (list, dict)) else None,
                        'processing_timestamp': dt.now(timezone.utc).isoformat()
                    }
                except json.JSONDecodeError as je:
                    return {'success': False, 'error': f'Invalid JSON: {je}'}
            
            elif struct_type == TYPE_JSONL:
                # Process JSONL line by line
                lines = content_str.strip().split('\n')
                processed_lines = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        try:
                            line_data = json.loads(line)
                            processed_lines.append(line_data)
                        except json.JSONDecodeError:
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Invalid JSON on line {i+1}", Path(__file__).stem)
                
                processed_data = {
                    'lines': processed_lines,
                    'data_type': 'jsonl',
                    'total_lines': len(lines),
                    'valid_lines': len(processed_lines),
                    'processing_timestamp': dt.now(timezone.utc).isoformat()
                }
            
            # Save processed data
            if processed_data:
                output_path = output_info['processed_path']
                save_success = self._save_json_compressed(processed_data, output_path)
                
                if save_success:
                    data_hash = generate_data_hash(output_path)
                    relative_path = output_path.relative_to(self.output_proc_dir)
                    
                    return {
                        'success': True,
                        'output_path': str(relative_path),
                        'output_filename': output_path.name,
                        'data_hash': data_hash,
                        'content_type': struct_type
                    }
                else:
                    return {'success': False, 'error': 'Failed to save processed data'}
            else:
                return {'success': False, 'error': 'No data processed'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Structured content processing failed: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_tabular_content(
        self, 
        content: Union[str, bytes], 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process tabular content (CSV, TSV)"""
        try:
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='replace')
            else:
                content_str = str(content)
            
            # Detect delimiter
            try:
                dialect = csv.Sniffer().sniff(content_str[:1024], delimiters=',\t;|')
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ',' if input_path.suffix.lower() == '.csv' else '\t'
            
            # Parse CSV
            try:
                df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
            except Exception as parse_e:
                return {'success': False, 'error': f'Failed to parse tabular data: {parse_e}'}
            
            if df.empty:
                return {'success': False, 'error': 'Empty tabular data'}
            
            # Analyze and process the data
            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'processing_timestamp': dt.now(timezone.utc).isoformat()
            }
            
            # Try numeric conversion and scaling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and self.scaler:
                try:
                    df_scaled = df.copy()
                    df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
                    analysis['scaled_columns'] = list(numeric_cols)
                    df = df_scaled
                except Exception as scale_e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Scaling failed for tabular data: {scale_e}", Path(__file__).stem)
            
            # Save as compressed Parquet with metadata
            output_path = output_info['processed_path'].with_suffix('.parquet.zst')
            save_success = self._save_dataframe_compressed(df, output_path)
            
            # Save analysis metadata
            metadata_path = output_info['processed_path'].with_suffix('.metadata.json.zst')
            metadata_success = self._save_json_compressed(analysis, metadata_path)
            
            if save_success:
                data_hash = generate_data_hash(output_path)
                relative_path = output_path.relative_to(self.output_proc_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_path),
                    'output_filename': output_path.name,
                    'data_hash': data_hash,
                    'content_type': 'tabular',
                    'analysis': analysis,
                    'metadata_saved': metadata_success
                }
            else:
                return {'success': False, 'error': 'Failed to save processed data'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Tabular processing failed: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _process_code_content(
        self, 
        content: Union[str, bytes], 
        input_path: Path, 
        output_info: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Process code content with syntax analysis"""
        try:
            if isinstance(content, bytes):
                code_content = content.decode('utf-8', errors='replace')
            else:
                code_content = str(content)
            
            # Basic code analysis
            lines = code_content.split('\n')
            analysis = {
                'total_lines': len(lines),
                'non_empty_lines': len([line for line in lines if line.strip()]),
                'comment_lines': len([line for line in lines if line.strip().startswith(('#', '//', '/*'))]),
                'file_extension': input_path.suffix,
                'estimated_language': self._detect_programming_language(code_content, input_path.suffix),
                'processing_timestamp': dt.now(timezone.utc).isoformat()
            }
            
            # Extract functions/classes if possible
            try:
                analysis['functions'] = self._extract_code_functions(code_content)
                analysis['classes'] = self._extract_code_classes(code_content)
            except Exception:
                pass
            
            # Prepare processed data
            processed_data = {
                'original_code': code_content,
                'analysis': analysis,
                'content_type': 'code'
            }
            
            # Save processed data
            output_path = output_info['processed_path']
            save_success = self._save_json_compressed(processed_data, output_path)
            
            if save_success:
                data_hash = generate_data_hash(output_path)
                relative_path = output_path.relative_to(self.output_proc_dir)
                
                return {
                    'success': True,
                    'output_path': str(relative_path),
                    'output_filename': output_path.name,
                    'data_hash': data_hash,
                    'content_type': 'code',
                    'language': analysis['estimated_language']
                }
            else:
                return {'success': False, 'error': 'Failed to save processed data'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Code processing failed: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}

    # Helper methods for content processing
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with configurable options"""
        try:
            # Basic cleaning
            cleaned = text.lower()
            
            # Apply regex cleaning if available
            if self.cleaning_regex:
                cleaned = self.cleaning_regex.sub(' ', cleaned)
            
            # Normalize whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Apply NLTK processing if available
            if NLTK_AVAILABLE and lemmatizer and stop_words:
                words = cleaned.split()
                words = [lemmatizer.lemmatize(word) for word in words if word and word not in stop_words]
                cleaned = ' '.join(words)
            
            return cleaned
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Text cleaning failed: {e}", Path(__file__).stem)
            return text
    
    def _extract_pdf_text(self, content: Union[str, bytes], file_path: Path) -> str:
        """Extract text from PDF content"""
        try:
            # This would typically use a PDF library like pdfminer or PyPDF2
            # For now, return a placeholder
            if isinstance(content, bytes):
                # Try to extract basic text if it's a text-based PDF
                try:
                    return content.decode('utf-8', errors='ignore')
                except Exception:
                    pass
            return f"PDF content extraction not implemented. File: {file_path.name}"
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>PDF text extraction failed: {e}", Path(__file__).stem)
            return ""
    
    def _extract_doc_text(self, content: Union[str, bytes], file_path: Path) -> str:
        """Extract text from DOC/DOCX content"""
        try:
            # This would typically use python-docx or similar
            # For now, return a placeholder
            return f"Document text extraction not implemented. File: {file_path.name}"
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Document text extraction failed: {e}", Path(__file__).stem)
            return ""
    
    def _extract_markup_text(self, content: Union[str, bytes]) -> str:
        """Extract text from HTML/XML content"""
        try:
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='replace')
            else:
                content_str = str(content)
            
            # Basic HTML/XML tag removal
            # This could be enhanced with BeautifulSoup
            text = re.sub(r'<[^>]+>', ' ', content_str)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Markup text extraction failed: {e}", Path(__file__).stem)
            return ""
    
    def _extract_markdown_text(self, content: Union[str, bytes]) -> str:
        """Extract text from Markdown content"""
        try:
            if isinstance(content, bytes):
                content_str = content.decode('utf-8', errors='replace')
            else:
                content_str = str(content)
            
            # Basic Markdown processing
            # Remove headers, code blocks, links, etc.
            text = re.sub(r'^#+\s+', '', content_str, flags=re.MULTILINE)  # Headers
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Markdown text extraction failed: {e}", Path(__file__).stem)
            return ""
    
    def _detect_programming_language(self, code: str, extension: str) -> str:
        """Simple programming language detection"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c_header',
            '.css': 'css',
            '.html': 'html',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.sh': 'shell'
        }
        
        return extension_map.get(extension.lower(), 'unknown')
    
    def _extract_code_functions(self, code: str) -> List[str]:
        """Extract function names from code"""
        functions = []
        patterns = [
            r'def\s+(\w+)\s*\(',  # Python
            r'function\s+(\w+)\s*\(',  # JavaScript
            r'(\w+)\s*\([^)]*\)\s*{',  # C/Java style
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            functions.extend(matches)
        
        return functions
    
    def _extract_code_classes(self, code: str) -> List[str]:
        """Extract class names from code"""
        classes = []
        patterns = [
            r'class\s+(\w+)',  # Python/Java/C++
            r'interface\s+(\w+)',  # Java/TypeScript
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            classes.extend(matches)
        
        return classes
    
    # Enhanced save methods
    def _save_json_compressed(self, data: Any, output_path: Path) -> bool:
        """Save data as compressed JSON"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            if self.config.compression_enabled:
                cctx = zstd.ZstdCompressor(level=self.config.compression_level)
                with open(output_path, 'wb') as f:
                    compressed = cctx.compress(json_str.encode('utf-8'))
                    f.write(compressed)
            else:
                output_path.write_text(json_str, encoding='utf-8')
            
            return True
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save JSON to {output_path}: {e}", Path(__file__).stem, exc_info=True)
            return False
    
    def _save_dataframe_compressed(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Save DataFrame as compressed Parquet"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert cuDF to pandas if needed
            if hasattr(df, 'to_pandas'):
                df_to_save = df.to_pandas()
            else:
                df_to_save = df
            
            if self.config.compression_enabled:
                df_to_save.to_parquet(output_path, compression='zstd', engine='pyarrow', index=False)
            else:
                df_to_save.to_parquet(output_path.with_suffix('.parquet'), engine='pyarrow', index=False)
            
            return True
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save DataFrame to {output_path}: {e}", Path(__file__).stem, exc_info=True)
            return False
    
    # Utility methods
    def _update_file_status_safe(self, file_path: Path, status: str, message: str = '') -> bool:
        """Safely update file status in repository"""
        try:
            if self.repo and hasattr(self.repo, 'update_file_status'):
                result = self.repo.update_file_status(file_path, status, change_description=message)
                return result.get('status') == OperationStatus.SUCCESS
            elif self.repo and hasattr(self.repo, 'update_entry'):
                self.repo.update_entry(file_path, status=status, error_message=message)
                return True
            return False
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update file status: {e}", Path(__file__).stem)
            return False
    
    def _get_file_size_safe(self, file_path: Path) -> int:
        """Safely get file size"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            return 0
    
    # Public interface methods for m1.py integration
    def scan_data_directory(self, directory: Optional[Path] = None) -> OperationResult:
        """Scan directory and update repository"""
        def _do_scan():
            scan_dir = directory or Path.cwd()
            if not scan_dir.exists():
                raise ValueError(f"Directory does not exist: {scan_dir}")
            
            log_statement('info', f"{self.log_prefix}:INFO>>Scanning directory: {scan_dir}", Path(__file__).stem)
            
            # Use repository scanning if available
            if self.repo and hasattr(self.repo, 'scan_and_update'):
                self.repo.scan_and_update(scan_dir)
                return {'directory': str(scan_dir), 'scanned': True}
            else:
                return {'directory': str(scan_dir), 'scanned': False, 'error': 'Repository scanning not available'}
        
        return safe_operation("scan_data_directory", _do_scan)
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return self.metrics.get_metrics()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of processing status"""
        try:
            if not self.repo:
                return {'error': 'No repository available'}
            
            if hasattr(self.repo, 'get_repository_statistics'):
                return self.repo.get_repository_statistics()
            else:
                return {'message': 'Status summary not available'}
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get status summary: {e}", Path(__file__).stem, exc_info=True)
            return {'error': str(e)}
    
    def process_linguistic_data(self, max_files: Optional[int] = None) -> OperationResult:
        """Process linguistic data - integration point for m1.py"""
        return self.process_all(
            statuses_to_process=(STATUS_NEW, STATUS_DISCOVERED),
            max_files=max_files
        )
    
    def get_files_by_status(self, status: str, limit: Optional[int] = None) -> List[str]:
        """Get files by processing status"""
        try:
            if self.repo and hasattr(self.repo, 'get_files_by_status'):
                files = self.repo.get_files_by_status([status])
                if limit:
                    files = files[:limit]
                return [str(f) for f in files]
            return []
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get files by status: {e}", Path(__file__).stem, exc_info=True)
            return []
    
    def save_state(self, container: DataProcessingContainer):
        container.update({
            'processed_files': self.processed_files,
            'repo_hash': self.repo_hash,
            'config': self.context.config  # Sync from context
        })

    def load_state(self, container: DataProcessingContainer):
        self.context.config = container.get('config', {})
        self.processed_files = container['processed_files']

# Enhanced Tokenizer with m1.py integration
class EnhancedTokenizer:
    """
    Enhanced Tokenizer with comprehensive error handling and m1.py integration
    
    Handles:
    - Loading processed data and converting to tensors
    - Multiple tokenization strategies
    - GPU acceleration with CPU fallbacks
    - Thread-safe operations
    - Progress tracking and metrics
    """
        
    def __init__(
        self,
        repo: Optional[RepoHandler] = None,
        max_workers: Optional[int] = None,
        config: Optional[ProcessingConfig] = None,
        context: Optional[Any] = None
    ):
        """
        Initialize Enhanced Tokenizer with system resource optimization
        
        Args:
            repo: Repository handler instance
            max_workers: Maximum worker threads
            config: Processing configuration
            context: DataProcessingContext from m1.py
        """
        self.log_prefix = f"{LOG_INS}::EnhancedTokenizer"
        self.config = config or ProcessingConfig()
        self.context = context
        self.repo = repo
        
        # Initialize metrics and locks
        self.metrics = ProcessingMetrics()
        self.lock = RLock()
        
        # Setup directories
        self._setup_directories()
        
        # ADDED: System resource optimization
        self.system_resources = None
        self.resource_config = None
        if self.config.use_system_optimization:
            try:
                from src.utils.system_resources import get_system_resources, get_optimal_config
                self.system_resources = get_system_resources()
                self.resource_config = get_optimal_config('mixed_workload')  # Tokenization is mixed CPU/IO
                
                # Override configuration with system-optimized values
                if max_workers is None and self.config.adaptive_workers:
                    effective_max_workers = self.resource_config['worker_count']
                else:
                    effective_max_workers = max_workers or self.config.max_workers
                    
                if self.config.memory_limit_gb is None:
                    self.config.memory_limit_gb = self.resource_config['memory_limit_gb']
                    
                log_statement('info', f"{self.log_prefix}:INFO>>System optimization enabled for tokenizer: "
                            f"workers={effective_max_workers}, memory_limit={self.config.memory_limit_gb:.1f}GB", 
                            Path(__file__).stem)
            except ImportError:
                log_statement('warning', f"{self.log_prefix}:WARNING>>System optimization not available for tokenizer", 
                            Path(__file__).stem)
                effective_max_workers = max_workers or self.config.max_workers
        else:
            effective_max_workers = max_workers or self.config.max_workers
        
        # Setup executor with system-optimized worker count
        self.max_workers = max(1, effective_max_workers)
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='Tokenizer'
        )
        
        # Setup device
        self.device = self._get_device()
        
        # Initialize tokenizer components
        self._tokenizer = None
        self._model = None
        
        log_statement('info', f"{self.log_prefix}:INFO>>EnhancedTokenizer initialized with {self.max_workers} workers on {self.device}"
                    f"{' (system-optimized)' if self.system_resources else ''}", Path(__file__).stem)
    
    def _setup_directories(self):
        """Setup tokenized data directories"""
        try:
            if PROJECT_IMPORTS_AVAILABLE and 'TOKENIZED_DATA_DIR' in globals():
                self.tokenized_output_dir = Path(TOKENIZED_DATA_DIR).resolve()
            else:
                self.tokenized_output_dir = Path('./data/tokenized').resolve()
            
            self.tokenized_output_dir.mkdir(parents=True, exist_ok=True)
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Tokenized output directory: {self.tokenized_output_dir}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to setup directories: {e}", Path(__file__).stem, exc_info=True)
            raise
    
    def _get_device(self) -> str:
        """Get appropriate device for tokenization"""
        try:
            # Try to import torch for device detection
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'
            except ImportError:
                return 'cpu'
        except Exception:
            return 'cpu'
    
    def setup_tokenizer(self, model_name: str = "bert-base-uncased") -> OperationResult:
        """Setup tokenizer with specified model"""
        def _do_setup():
            try:
                # Try to get tokenizer from context first
                if self.context and hasattr(self.context, 'container'):
                    try:
                        context_tokenizer = self.context.container.get_tokenizer(model_name)
                        if context_tokenizer:
                            self._tokenizer = context_tokenizer
                            log_statement('info', f"{self.log_prefix}:INFO>>Using tokenizer from context: {model_name}", Path(__file__).stem)
                            return {'model_name': model_name, 'source': 'context'}
                    except Exception as context_e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to get tokenizer from context: {context_e}", Path(__file__).stem)
                
                # Fallback to direct import
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    log_statement('info', f"{self.log_prefix}:INFO>>Loaded tokenizer directly: {model_name}", Path(__file__).stem)
                    return {'model_name': model_name, 'source': 'direct'}
                except ImportError:
                    raise RuntimeError("Transformers library not available")
                except Exception as load_e:
                    raise RuntimeError(f"Failed to load tokenizer {model_name}: {load_e}")
                    
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Tokenizer setup failed: {e}", Path(__file__).stem, exc_info=True)
                raise
        
        return safe_operation("setup_tokenizer", _do_setup)
    
    def tokenize_all(
        self, 
        base_dir_filter: Optional[Path] = None,
        statuses_to_process: Tuple[str, ...] = (STATUS_PROCESSED,),
        max_files: Optional[int] = None
    ) -> OperationResult:
        """
        Tokenize all processed files with enhanced error handling and system optimization
        
        Args:
            base_dir_filter: Optional directory filter
            statuses_to_process: File statuses to tokenize
            max_files: Maximum number of files to process
            
        Returns:
            OperationResult with tokenization statistics
        """
        def _do_tokenize_all():
            if not self._tokenizer:
                raise RuntimeError("Tokenizer not setup. Call setup_tokenizer first.")
            
            log_statement('info', f"{self.log_prefix}:INFO>>Starting tokenization - Filter: {base_dir_filter}, Statuses: {statuses_to_process}", Path(__file__).stem)
            
            # Initialize metrics
            self.metrics.start_processing()
            
            # ADDED: Dynamic configuration based on workload
            files_to_tokenize = self._get_files_to_tokenize(base_dir_filter, statuses_to_process, max_files)
            
            if self.config.use_system_optimization and self.system_resources:
                try:
                    from src.utils.system_resources import get_optimal_config
                    # Get optimized config for tokenization workload
                    operation_config = get_optimal_config('mixed_workload', len(files_to_tokenize))
                    
                    if self.config.adaptive_workers:
                        effective_workers = operation_config['worker_count']
                        log_statement('info', f"{self.log_prefix}:INFO>>Adjusting tokenizer workers from {self.max_workers} to {effective_workers}", 
                                    Path(__file__).stem)
                    else:
                        effective_workers = self.max_workers
                        
                    if self.config.dynamic_batch_sizing:
                        effective_batch_size = operation_config['batch_size']
                    else:
                        effective_batch_size = self.config.batch_size
                        
                    memory_limit_gb = operation_config['memory_limit_gb']
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to optimize tokenizer configuration: {e}", 
                                Path(__file__).stem)
                    effective_workers = self.max_workers
                    effective_batch_size = self.config.batch_size
                    memory_limit_gb = self.config.memory_limit_gb or 4.0
            else:
                effective_workers = self.max_workers
                effective_batch_size = self.config.batch_size
                memory_limit_gb = self.config.memory_limit_gb or 4.0
            
            try:
                if not files_to_tokenize:
                    log_statement('info', f"{self.log_prefix}:INFO>>No files found for tokenization", Path(__file__).stem)
                    return {
                        'tokenized_count': 0,
                        'failed_count': 0,
                        'total_files': 0,
                        'message': 'No files to tokenize'
                    }
                
                # ADDED: Enhanced progress tracking
                progress_tracker = None
                if self.config.enable_progress_tracking:
                    try:
                        from src.utils.progress_tracker import create_progress_tracker
                        progress_tracker = create_progress_tracker(
                            total_items=len(files_to_tokenize),
                            description=f"Tokenizing files [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]",
                            unit="files",
                            show_resources=self.config.resource_monitoring,
                            update_interval=0.5
                        )
                        log_statement('info', f"{self.log_prefix}:INFO>>Enhanced progress tracking enabled for tokenization", 
                                    Path(__file__).stem)
                    except ImportError:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Enhanced progress tracking not available", 
                                    Path(__file__).stem)
                        if PROGRESS_TRACKING_AVAILABLE:
                            progress_tracker = create_progress_tracker(
                                total_items=len(files_to_tokenize),
                                description=f"Tokenizing files [{base_dir_filter.name[:15] if base_dir_filter else 'All'}]",
                                unit="files",
                                show_resources=False
                            )
                
                # Tokenize files with system optimization
                results = self._tokenize_files_batch(files_to_tokenize, progress_tracker,
                                                effective_workers, effective_batch_size, memory_limit_gb)
                
                # Finalize metrics
                self.metrics.end_processing()
                
                # Save repository
                if self.repo and hasattr(self.repo, 'save'):
                    try:
                        self.repo.save()
                    except Exception as save_e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Repository save failed: {save_e}", Path(__file__).stem)
                
                # Compile results
                final_metrics = self.metrics.get_metrics()
                result = {
                    'tokenized_count': final_metrics['files_processed'],
                    'failed_count': final_metrics['files_failed'],
                    'total_files': len(files_to_tokenize),
                    'processing_time': final_metrics['processing_time'],
                    'processing_rate': self.metrics.get_processing_rate(),
                    'successful_files': results.get('successful_files', []),
                    'failed_files': results.get('failed_files', []),
                    'device': self.device,
                    'system_optimized': self.config.use_system_optimization,
                    'effective_workers': effective_workers,
                    'effective_batch_size': effective_batch_size,
                    'memory_limit_gb': memory_limit_gb
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Tokenization completed: {result['tokenized_count']} tokenized, {result['failed_count']} failed"
                            f"{' (system-optimized)' if self.config.use_system_optimization else ''}", Path(__file__).stem)
                
                return result
                
            except Exception as e:
                self.metrics.end_processing()
                log_statement('error', f"{self.log_prefix}:ERROR>>Tokenization batch failed: {e}", Path(__file__).stem, exc_info=True)
                raise
            finally:
                if progress_tracker:
                    progress_tracker.finish()
        
        return safe_operation("tokenize_all", _do_tokenize_all)
    
    def _get_files_to_tokenize(
        self, 
        base_dir_filter: Optional[Path],
        statuses_to_process: Tuple[str, ...],
        max_files: Optional[int]
    ) -> List[Tuple[Path, Path]]:
        """Get files ready for tokenization"""
        try:
            if not self.repo:
                log_statement('warning', f"{self.log_prefix}:WARNING>>No repository available", Path(__file__).stem)
                return []
            
            # Get source files by status
            if hasattr(self.repo, 'get_files_by_status'):
                source_paths = self.repo.get_files_by_status(list(statuses_to_process), base_dir=base_dir_filter)
            else:
                source_paths = []
            
            if not source_paths:
                return []
            
            # Find corresponding processed files
            files_to_tokenize = []
            
            with self.lock:
                if hasattr(self.repo, 'df') and self.repo.df is not None:
                    source_paths_str = {str(p.resolve()) for p in source_paths}
                    matching_rows = self.repo.df[self.repo.df[COL_FILEPATH].isin(source_paths_str)]
                    
                    for _, row in matching_rows.iterrows():
                        source_path = Path(row[COL_FILEPATH])
                        processed_path_rel = row.get(COL_PROCESSED_PATH, '')
                        
                        if processed_path_rel:
                            # Try to find the processed file
                            if hasattr(self, 'output_proc_dir'):
                                processed_path = self.output_proc_dir / processed_path_rel
                            elif PROJECT_IMPORTS_AVAILABLE and 'PROCESSED_DATA_DIR' in globals():
                                processed_path = Path(PROCESSED_DATA_DIR) / processed_path_rel
                            else:
                                processed_path = Path('./data/processed') / processed_path_rel
                            
                            if processed_path.exists():
                                files_to_tokenize.append((source_path, processed_path))
                            else:
                                log_statement('warning', f"{self.log_prefix}:WARNING>>Processed file not found: {processed_path}", Path(__file__).stem)
            
            # Apply max_files limit
            if max_files and len(files_to_tokenize) > max_files:
                files_to_tokenize = files_to_tokenize[:max_files]
                log_statement('info', f"{self.log_prefix}:INFO>>Limited tokenization to {max_files} files", Path(__file__).stem)
            
            return files_to_tokenize
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error getting files to tokenize: {e}", Path(__file__).stem, exc_info=True)
            return []
    
    def _tokenize_files_batch(
        self, 
        files_to_tokenize: List[Tuple[Path, Path]], 
        progress_tracker: Optional[Any],
        effective_workers: int,
        effective_batch_size: int,
        memory_limit_gb: float
    ) -> Dict[str, Any]:
        """Tokenize files in batch with error handling and memory management"""
        successful_files = []
        failed_files = []
        
        # ADDED: Memory monitoring
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)
        memory_threshold = initial_memory + (memory_limit_gb * 0.8)
        
        # Create optimized thread pool
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix='TokenizerProc') as batch_executor:
            # Process in memory-aware batches
            batches = [files_to_tokenize[i:i + effective_batch_size]
                    for i in range(0, len(files_to_tokenize), effective_batch_size)]
            
            log_statement('info', f"{self.log_prefix}:INFO>>Tokenizing {len(files_to_tokenize)} files in {len(batches)} batches "
                        f"(batch_size={effective_batch_size}, workers={effective_workers})", Path(__file__).stem)
            
            for batch_idx, batch in enumerate(batches):
                # Memory check before batch
                current_memory = process.memory_info().rss / (1024**3)
                if current_memory > memory_threshold:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>High memory usage before tokenization batch: {current_memory:.1f}GB", 
                                Path(__file__).stem)
                    import gc
                    gc.collect()
                    
                    # Clear any GPU cache if using GPU
                    if self.device != 'cpu':
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                    
                    time.sleep(0.1)
                
                # Submit tokenization tasks
                futures = []
                for source_path, processed_path in batch:
                    try:
                        future = batch_executor.submit(self._tokenize_file_with_retry, source_path, processed_path)
                        futures.append((future, source_path, processed_path))
                    except Exception as submit_e:
                        log_statement('error', f"{self.log_prefix}:ERROR>>Failed to submit tokenization task: {submit_e}", Path(__file__).stem)
                        failed_files.append(str(source_path))
                        self.metrics.increment_failed(str(submit_e))
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=str(submit_e))
                
                # Process results
                for future, source_path, processed_path in futures:
                    file_name = source_path.name
                    
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        
                        if result and result.get('success'):
                            successful_files.append(str(source_path))
                            file_size = self._get_file_size_safe(processed_path)
                            self.metrics.increment_processed(file_size)
                            
                            if progress_tracker:
                                progress_tracker.update(success=True)
                            
                            log_statement('debug', f"{self.log_prefix}:DEBUG>>Successfully tokenized: {file_name}", Path(__file__).stem)
                        else:
                            failed_files.append(str(source_path))
                            error_msg = result.get('error', 'Unknown error') if result else 'No result'
                            self.metrics.increment_failed(f"{file_name}: {error_msg}")
                            
                            if progress_tracker:
                                progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                            
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to tokenize {file_name}: {error_msg}", Path(__file__).stem)
                            
                    except TimeoutError:
                        failed_files.append(str(source_path))
                        error_msg = f"Tokenization timeout ({self.config.timeout_seconds}s)"
                        self.metrics.increment_failed(f"{file_name}: {error_msg}")
                        
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>Tokenization timeout for {file_name}", Path(__file__).stem)
                        
                    except Exception as e:
                        failed_files.append(str(source_path))
                        error_msg = str(e)
                        self.metrics.increment_failed(f"{file_name}: {error_msg}")
                        
                        if progress_tracker:
                            progress_tracker.update(success=False, error_msg=f"{file_name}: {error_msg}")
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>Exception tokenizing {file_name}: {e}", Path(__file__).stem, exc_info=True)
                
                # Log batch completion
                current_memory = process.memory_info().rss / (1024**3)
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Tokenization batch {batch_idx + 1}/{len(batches)} completed. "
                            f"Memory: {current_memory:.1f}GB, Device: {self.device}", Path(__file__).stem)
        
        return {
            'successful_files': successful_files,
            'failed_files': failed_files
        }
    
    def _tokenize_file_with_retry(self, source_path: Path, processed_path: Path) -> Dict[str, Any]:
        """Tokenize file with retry logic"""
        file_name = source_path.name
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                if attempt > 0:
                    log_statement('info', f"{self.log_prefix}:INFO>>Retry attempt {attempt + 1} for tokenizing {file_name}", Path(__file__).stem)
                    time.sleep(min(2 ** attempt, 10))
                
                result = self._tokenize_file(source_path, processed_path)
                
                if result and result.get('success'):
                    return result
                else:
                    last_error = result.get('error', 'Unknown error') if result else 'No result'
                    
            except Exception as e:
                last_error = str(e)
                log_statement('warning', f"{self.log_prefix}:WARNING>>Tokenization attempt {attempt + 1} failed for {file_name}: {e}", Path(__file__).stem)
                
                if attempt == self.config.retry_attempts - 1:
                    log_statement('error', f"{self.log_prefix}:ERROR>>All tokenization attempts failed for {file_name}", Path(__file__).stem, exc_info=True)
        
        return {
            'success': False,
            'error': f"All {self.config.retry_attempts} attempts failed. Last error: {last_error}"
        }
    
    def _tokenize_file(self, source_path: Path, processed_path: Path) -> Dict[str, Any]:
        """Tokenize a single processed file"""
        try:
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Tokenizing {source_path.name}", Path(__file__).stem)
            
            # Update repository status
            self._update_file_status_safe(source_path, STATUS_TOKENIZING)
            
            # Load processed data
            processed_data = self._load_processed_data(processed_path)
            if not processed_data:
                raise ValueError("Failed to load processed data")
            
            # Extract text content for tokenization
            text_content = self._extract_text_for_tokenization(processed_data)
            if not text_content:
                raise ValueError("No text content found for tokenization")
            
            # Tokenize content
            tokens = self._tokenize_content(text_content)
            if not tokens:
                raise ValueError("Tokenization failed")
            
            # Convert to tensors
            tensor_data = self._convert_to_tensors(tokens)
            
            # Save tokenized data
            output_path = self._generate_tokenized_output_path(source_path)
            save_success = self._save_tokenized_data(tensor_data, output_path)
            
            if not save_success:
                raise ValueError("Failed to save tokenized data")
            
            # Update repository
            self._update_file_status_safe(source_path, STATUS_TOKENIZED, f"Tokenized to {output_path.name}")
            
            # Generate hash
            data_hash = generate_data_hash(output_path)
            
            result = {
                'success': True,
                'source_path': str(source_path),
                'output_path': str(output_path),
                'data_hash': data_hash,
                'token_count': len(tokens.get('input_ids', [])) if isinstance(tokens, dict) else 0,
                'device': self.device
            }
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Successfully tokenized {source_path.name}", Path(__file__).stem)
            return result
            
        except Exception as e:
            # Update repository with error
            self._update_file_status_safe(source_path, STATUS_FAILED, f"Tokenization failed: {str(e)}")
            
            log_statement('error', f"{self.log_prefix}:ERROR>>Tokenization failed for {source_path.name}: {e}", Path(__file__).stem, exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _load_processed_data(self, processed_path: Path) -> Optional[Any]:
        """Load processed data from file"""
        try:
            if not processed_path.exists():
                return None
            
            # Handle different file formats
            if processed_path.suffix == '.zst':
                # Compressed file
                dctx = zstd.ZstdDecompressor()
                with open(processed_path, 'rb') as f:
                    decompressed = dctx.decompress(f.read())
                
                # Determine content type
                if processed_path.suffixes[-2:] == ['.json', '.zst']:
                    return json.loads(decompressed.decode('utf-8'))
                elif processed_path.suffixes[-2:] == ['.parquet', '.zst']:
                    # Handle compressed parquet
                    import io
                    return pd.read_parquet(io.BytesIO(decompressed))
                else:
                    return decompressed.decode('utf-8')
            
            elif processed_path.suffix == '.json':
                return json.loads(processed_path.read_text(encoding='utf-8'))
            elif processed_path.suffix == '.parquet':
                return pd.read_parquet(processed_path)
            else:
                return processed_path.read_text(encoding='utf-8')
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load processed data from {processed_path}: {e}", Path(__file__).stem, exc_info=True)
            return None
    
    def _extract_text_for_tokenization(self, processed_data: Any) -> Optional[str]:
        """Extract text content from processed data"""
        try:
            if isinstance(processed_data, str):
                return processed_data
            elif isinstance(processed_data, dict):
                # Try different keys for text content
                text_keys = ['cleaned_text', 'text', 'content', 'original_code']
                for key in text_keys:
                    if key in processed_data:
                        content = processed_data[key]
                        if isinstance(content, str):
                            return content
                
                # If structured data, try to extract text from various fields
                if 'original_data' in processed_data:
                    return str(processed_data['original_data'])
                elif 'lines' in processed_data:
                    lines = processed_data['lines']
                    if isinstance(lines, list):
                        return '\n'.join(str(line) for line in lines)
                
                # Fallback: convert entire dict to string
                return json.dumps(processed_data)
                
            elif isinstance(processed_data, pd.DataFrame):
                # Extract text from DataFrame
                text_cols = processed_data.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    return ' '.join(processed_data[text_cols[0]].astype(str).tolist())
                else:
                    return processed_data.to_string()
            else:
                return str(processed_data)
                
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to extract text for tokenization: {e}", Path(__file__).stem)
            return None
    
    def _tokenize_content(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Tokenize text content using the configured tokenizer"""
        try:
            if not self._tokenizer:
                raise ValueError("Tokenizer not configured")
            
            # Truncate if too long
            max_length = 512  # Standard BERT max length
            if len(text_content) > max_length * 4:  # Rough estimate for tokens
                text_content = text_content[:max_length * 4]
            
            # Tokenize
            tokens = self._tokenizer(
                text_content,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            return tokens
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Content tokenization failed: {e}", Path(__file__).stem, exc_info=True)
            return None
    
    def _convert_to_tensors(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tokenizer output to tensors"""
        try:
            # Move to appropriate device
            tensor_data = {}
            for key, value in tokens.items():
                if hasattr(value, 'to'):
                    tensor_data[key] = value.to(self.device)
                else:
                    tensor_data[key] = value
            
            return tensor_data
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Tensor conversion failed: {e}", Path(__file__).stem, exc_info=True)
            # Fallback: return original
            return tokens
    
    def _generate_tokenized_output_path(self, source_path: Path) -> Path:
        """Generate output path for tokenized data"""
        try:
            # Create relative path structure
            relative_path = source_path.relative_to(source_path.anchor)
            output_dir = self.tokenized_output_dir / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            output_filename = f"{source_path.stem}_tokenized.pt"
            if self.config.compression_enabled:
                output_filename += ".zst"
            
            return output_dir / output_filename
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to generate output path: {e}", Path(__file__).stem, exc_info=True)
            # Fallback
            return self.tokenized_output_dir / f"{source_path.stem}_tokenized.pt"
    
    def _save_tokenized_data(self, tensor_data: Dict[str, Any], output_path: Path) -> bool:
        """Save tokenized tensor data"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure tensors are on CPU for saving
            cpu_data = {}
            for key, value in tensor_data.items():
                if hasattr(value, 'cpu'):
                    cpu_data[key] = value.cpu()
                else:
                    cpu_data[key] = value
            
            if self.config.compression_enabled and output_path.suffix == '.zst':
                # Save compressed
                import io
                buffer = io.BytesIO()
                
                try:
                    import torch
                    torch.save(cpu_data, buffer)
                except ImportError:
                    # Fallback to pickle
                    import pickle
                    pickle.dump(cpu_data, buffer)
                
                buffer.seek(0)
                cctx = zstd.ZstdCompressor(level=self.config.compression_level)
                
                with open(output_path, 'wb') as f:
                    compressed = cctx.compress(buffer.read())
                    f.write(compressed)
                
                buffer.close()
            else:
                # Save uncompressed
                try:
                    import torch
                    torch.save(cpu_data, output_path)
                except ImportError:
                    import pickle
                    with open(output_path, 'wb') as f:
                        pickle.dump(cpu_data, f)
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Saved tokenized data to {output_path}", Path(__file__).stem)
            return True
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save tokenized data to {output_path}: {e}", Path(__file__).stem, exc_info=True)
            return False
    
    def _update_file_status_safe(self, file_path: Path, status: str, message: str = '') -> bool:
        """Safely update file status in repository"""
        try:
            if self.repo and hasattr(self.repo, 'update_file_status'):
                result = self.repo.update_file_status(file_path, status, change_description=message)
                return result.get('status') == OperationStatus.SUCCESS
            elif self.repo and hasattr(self.repo, 'update_entry'):
                self.repo.update_entry(file_path, status=status, error_message=message)
                return True
            return False
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update file status: {e}", Path(__file__).stem)
            return False
    
    def _get_file_size_safe(self, file_path: Path) -> int:
        """Safely get file size"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            return 0
    
    def get_tokenization_metrics(self) -> Dict[str, Any]:
        """Get tokenization metrics"""
        return self.metrics.get_metrics()
    
    def cleanup(self):
        """Clean up tokenizer resources"""
        try:
            log_statement('info', f"{self.log_prefix}:INFO>>Starting tokenizer cleanup", Path(__file__).stem)
            
            if hasattr(self, 'executor') and self.executor:
                try:
                    self.executor.shutdown(wait=True, timeout=30)
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Executor shutdown completed", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Executor shutdown error: {e}", Path(__file__).stem)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Tokenizer cleanup completed", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Tokenizer cleanup error: {e}", Path(__file__).stem, exc_info=True)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except Exception:
            pass


# Factory functions for m1.py integration
def create_data_processor(
    repo: Optional[RepoHandler] = None,
    config: Optional[ProcessingConfig] = None,
    context: Optional[Any] = None,
    **kwargs
) -> DataProcessor:
    """
    Factory function to create DataProcessor instance with system optimization
    
    Args:
        repo: Optional RepoHandler instance
        config: Optional ProcessingConfig
        context: Optional DataProcessingContext from m1.py
        **kwargs: Additional arguments for DataProcessor
        
    Returns:
        Configured DataProcessor instance
    """
    try:
        # Merge config if provided
        final_config = config or ProcessingConfig()
        
        # ADDED: Enable system optimization by default if available
        if 'use_system_optimization' not in kwargs:
            try:
                from src.utils.system_resources import get_system_resources
                final_config.use_system_optimization = True
                log_statement('info', f"{LOG_INS}:INFO>>System optimization enabled for DataProcessor", Path(__file__).stem)
            except ImportError:
                final_config.use_system_optimization = False
        
        # Extract relevant kwargs
        processor_kwargs = {
            'repo': repo,
            'config': final_config,
            'context': context
        }
        
        # Add other kwargs
        for key in ['max_workers', 'output_dir', 'repo_path_override']:
            if key in kwargs:
                processor_kwargs[key] = kwargs[key]
        
        processor = DataProcessor(**processor_kwargs)
        
        log_statement('info', f"{LOG_INS}:INFO>>DataProcessor created successfully"
                     f"{' with system optimization' if final_config.use_system_optimization else ''}", 
                     Path(__file__).stem)
        return processor
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to create DataProcessor: {e}", Path(__file__).stem, exc_info=True)
        raise

def create_tokenizer(
    repo: Optional[RepoHandler] = None,
    config: Optional[ProcessingConfig] = None,
    context: Optional[Any] = None,
    **kwargs
) -> EnhancedTokenizer:
    """
    Factory function to create EnhancedTokenizer instance with system optimization
    
    Args:
        repo: Optional RepoHandler instance
        config: Optional ProcessingConfig
        context: Optional DataProcessingContext from m1.py
        **kwargs: Additional arguments
        
    Returns:
        Configured EnhancedTokenizer instance
    """
    try:
        final_config = config or ProcessingConfig()
        
        # ADDED: Enable system optimization by default if available
        if 'use_system_optimization' not in kwargs:
            try:
                from src.utils.system_resources import get_system_resources
                final_config.use_system_optimization = True
                log_statement('info', f"{LOG_INS}:INFO>>System optimization enabled for EnhancedTokenizer", Path(__file__).stem)
            except ImportError:
                final_config.use_system_optimization = False
        
        tokenizer_kwargs = {
            'repo': repo,
            'config': final_config,
            'context': context
        }
        
        # Add other kwargs
        for key in ['max_workers']:
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]
        
        tokenizer = EnhancedTokenizer(**tokenizer_kwargs)
        
        log_statement('info', f"{LOG_INS}:INFO>>EnhancedTokenizer created successfully"
                     f"{' with system optimization' if final_config.use_system_optimization else ''}", 
                     Path(__file__).stem)
        return tokenizer
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to create EnhancedTokenizer: {e}", Path(__file__).stem, exc_info=True)
        raise

# Legacy compatibility functions
def get_data_processor(*args, **kwargs) -> DataProcessor:
    """Legacy function to get DataProcessor instance"""
    return create_data_processor(*args, **kwargs)

def get_tokenizer(*args, **kwargs) -> EnhancedTokenizer:
    """Legacy function to get Tokenizer instance"""
    return create_tokenizer(*args, **kwargs)

# Legacy Tokenizer class for backward compatibility
class Tokenizer(EnhancedTokenizer):
    """Legacy Tokenizer class - inherits from EnhancedTokenizer"""
    
    def __init__(self, *args, **kwargs):
        """Initialize legacy tokenizer"""
        super().__init__(*args, **kwargs)
        log_statement('info', f"{self.log_prefix}:INFO>>Legacy Tokenizer instance created", Path(__file__).stem)
    
    def tokenize_all(self, *args, **kwargs):
        """Legacy tokenize_all method"""
        return super().tokenize_all(*args, **kwargs)

# Module exports and initialization
__all__ = [
    'DataProcessor',
    'EnhancedTokenizer', 
    'Tokenizer',
    'ProcessingConfig',
    'ProcessingMetrics',
    'create_data_processor',
    'create_tokenizer',
    'get_data_processor', 
    'get_tokenizer',
    'LOG_INS'
]

# Initialize module-level components
try:
    # Log module initialization
    log_statement('info', f"{LOG_INS}:INFO>>Enhanced Data Processing Module loaded successfully", Path(__file__).stem)
    
    # Check GPU availability
    if GPU_AVAILABLE:
        log_statement('info', f"{LOG_INS}:INFO>>GPU acceleration available (cuDF/CuPy/cuML)", Path(__file__).stem)
    else:
        log_statement('info', f"{LOG_INS}:INFO>>Using CPU fallback (pandas/numpy/sklearn)", Path(__file__).stem)
    
    # Check NLTK availability
    if NLTK_AVAILABLE:
        log_statement('info', f"{LOG_INS}:INFO>>NLTK text processing available", Path(__file__).stem)
    else:
        log_statement('info', f"{LOG_INS}:INFO>>Basic text processing only (NLTK not available)", Path(__file__).stem)
    
    # Check progress tracking
    if PROGRESS_TRACKING_AVAILABLE:
        log_statement('info', f"{LOG_INS}:INFO>>Enhanced progress tracking available", Path(__file__).stem)
    else:
        log_statement('info', f"{LOG_INS}:INFO>>Basic progress tracking only", Path(__file__).stem)
    
    # ADDED: Check system optimization availability
    try:
        from src.utils.system_resources import get_system_resources
        system_resources = get_system_resources()
        log_statement('info', f"{LOG_INS}:INFO>>System resource optimization available - "
                     f"{system_resources.cpu_cores_physical} cores, {system_resources.ram_total_gb:.1f}GB RAM", 
                     Path(__file__).stem)
    except ImportError:
        log_statement('info', f"{LOG_INS}:INFO>>System resource optimization not available - using defaults", 
                     Path(__file__).stem)
    
    # Module ready
    log_statement('info', f"{LOG_INS}:INFO>>Module initialization completed - Ready for m1.py integration with enhanced capabilities", 
                 Path(__file__).stem)
    
except Exception as init_e:
    log_statement('error', f"{LOG_INS}:ERROR>>Module initialization error: {init_e}", Path(__file__).stem, exc_info=True)