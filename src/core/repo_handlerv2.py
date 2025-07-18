# src/core/repo_handler.py
"""
Repository Management System

A comprehensive system for managing Git repositories with metadata tracking,
file versioning, and integrated operations. This module provides classes for
handling repository operations, metadata management, and file tracking.

Refactored for better separation of concerns, consistent error handling,
and improved maintainability.
"""

import os
import sys
import json
import time
import hashlib
import inspect
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Set, Protocol, TypeVar
from threading import Lock, RLock
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Standard library imports with error handling
try:
    import shutil
    import warnings
    import traceback
    import io
    import csv
    import re
    import random
    import difflib
    STDLIB_AVAILABLE = True
except ImportError as e:
    STDLIB_AVAILABLE = False
    print(f"Critical error: Standard library components not available: {e}")
    sys.exit(1)

# Third-party imports with graceful fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple fallback for tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from filelock import FileLock, Timeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    FileLock = None
    Timeout = Exception

# Git-related imports with fallbacks
try:
    import git
    from git import Repo, Blob, PushInfo, GitCommandError
    from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None
    Repo = None
    GitCommandError = Exception
    BadName = Exception
    InvalidGitRepositoryError = Exception
    NoSuchPathError = Exception

# Compression imports with fallbacks
try:
    import gzip
    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False
    gzip = None

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None

# Pydantic imports with fallbacks
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception

# Project-specific imports with error handling
try:
    from src.utils.config import *
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Define fallback constants
    ROOT_DIR = Path.cwd()
    BASE_DATA_DIR = Path.cwd() / "data"
    DATA_REPO_DIR = Path.cwd() / "repositories"

try:
    from src.data.constants import *
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False
    # Define fallback constants
    METADATA_FILENAME = "metadata.json"
    GITIGNORE_FILENAME = ".gitignore"
    PROGRESS_DIR = "progress"
    STATUS_NEW = "new"
    STATUS_PROCESSED = "processed"
    STATUS_ARCHIVED = "archived"
    STATUS_DELETED = "deleted"
    STATUS_DISCOVERED = "discovered"
    DEFAULT_APPLICATION_STATUS = STATUS_NEW
    DEFAULT_HASH_ALGORITHM = "sha256"
    SUPPORTED_HASH_ALGORITHMS = ["md5", "sha256", "sha1"]
    MAX_WORKERS = 4
    DF_CACHE_MAXSIZE = 100

try:
    from src.utils.logger import log_statement
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def log_statement(level: str, message: str, module: str, exc_info: bool = False):
        """Fallback logging function"""
        logger = logging.getLogger(module)
        getattr(logger, level.lower(), logger.info)(message, exc_info=exc_info)

try:
    from src.utils.helpers import process_file, LRUCache, _ensure_pathResolve, ensure_dir_exists
    HELPERS_AVAILABLE = True
except ImportError:
    HELPERS_AVAILABLE = False
    
    def ensure_dir_exists(path: Path) -> None:
        """Fallback directory creation"""
        path.mkdir(parents=True, exist_ok=True)
    
    def _ensure_pathResolve(path: Union[str, Path]) -> Path:
        """Fallback path resolution"""
        return Path(path).resolve() if path else Path.cwd()
    
    class LRUCache:
        """Simple LRU cache fallback"""
        def __init__(self, maxsize: int = 100):
            self.maxsize = maxsize
            self.cache = {}
        
        def get(self, key: str):
            return self.cache.get(key)
        
        def put(self, key: str, value: Any):
            if len(self.cache) >= self.maxsize:
                # Remove oldest item (simple implementation)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value
        
        def clear(self):
            self.cache.clear()

try:
    from src.utils.hashing import generate_data_hash, HashInfo
    HASHING_AVAILABLE = True
except ImportError:
    HASHING_AVAILABLE = False
    
    def generate_data_hash(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
        """Fallback hash generation"""
        try:
            hasher = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    class HashInfo:
        """Fallback HashInfo class"""
        def __init__(self, hash_type: str, value: str):
            self.hash_type = hash_type
            self.value = value

try:
    from src.core.models import FileMetadataEntry, FileVersion, MetadataCollection
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    
    # Create fallback model classes
    class FileMetadataEntry:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def model_dump(self, **kwargs):
            return self.__dict__
    
    class FileVersion:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MetadataCollection:
        def __init__(self, root=None):
            self.root = root or {}
        
        def get_entry(self, key):
            return self.root.get(key)
        
        def add_or_update_entry(self, entry):
            self.root[getattr(entry, 'filepath_relative', str(entry))] = entry
        
        @classmethod
        def model_validate(cls, data):
            return cls(root=data)
        
        def to_dict(self):
            return self.root

# Module-level constants and configuration
LOG_INS = f"{Path(__file__).stem}:repo_handler"

# Error classes for better error handling
class RepoHandlerError(Exception):
    """Base exception for repository handler errors"""
    pass

class GitOperationError(RepoHandlerError):
    """Raised when Git operations fail"""
    pass

class MetadataError(RepoHandlerError):
    """Raised when metadata operations fail"""
    pass

class FileOperationError(RepoHandlerError):
    """Raised when file operations fail"""
    pass

class ConfigurationError(RepoHandlerError):
    """Raised when configuration is invalid"""
    pass

class DependencyError(RepoHandlerError):
    """Raised when required dependencies are missing"""
    pass

# Enums for better type safety
class OperationStatus(Enum):
    """Enumeration of operation statuses"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"

class FileStatus(Enum):
    """Enumeration of file statuses"""
    NEW = STATUS_NEW
    PROCESSED = STATUS_PROCESSED
    ARCHIVED = STATUS_ARCHIVED
    DELETED = STATUS_DELETED
    DISCOVERED = STATUS_DISCOVERED

# Type definitions for better type hints
T = TypeVar('T')
PathLike = Union[str, Path]
OperationResult = Dict[str, Any]

# Protocols for better interface definition
class GitOperationsProtocol(Protocol):
    """Protocol for Git operations"""
    
    def is_valid_repo(self) -> bool:
        """Check if repository is valid"""
        ...
    
    def commit_changes(self, files: List[PathLike], message: str) -> bool:
        """Commit changes to repository"""
        ...
    
    def get_status(self) -> Dict[str, List[str]]:
        """Get repository status"""
        ...

class MetadataHandlerProtocol(Protocol):
    """Protocol for metadata operations"""
    
    def read_metadata(self) -> Dict[str, Any]:
        """Read metadata from storage"""
        ...
    
    def write_metadata(self, data: Dict[str, Any], **kwargs) -> bool:
        """Write metadata to storage"""
        ...

# Configuration class for dependency injection
@dataclass
class RepoHandlerConfig:
    """Configuration for repository handler"""
    repo_path: Path
    metadata_filename: str = METADATA_FILENAME
    create_if_missing: bool = True
    use_git: bool = True
    use_compression: bool = True
    max_workers: int = MAX_WORKERS
    cache_size: int = DF_CACHE_MAXSIZE
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.repo_path = _ensure_pathResolve(self.repo_path)
        if not isinstance(self.metadata_filename, str):
            raise ConfigurationError("metadata_filename must be a string")
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")

# Dependency checker
class DependencyChecker:
    """Utility class to check and report on dependencies"""
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check availability of all dependencies"""
        return {
            'pandas': PANDAS_AVAILABLE,
            'git': GIT_AVAILABLE,
            'pydantic': PYDANTIC_AVAILABLE,
            'zstd': ZSTD_AVAILABLE,
            'gzip': GZIP_AVAILABLE,
            'tqdm': TQDM_AVAILABLE,
            'filelock': FILELOCK_AVAILABLE,
            'config': CONFIG_AVAILABLE,
            'constants': CONSTANTS_AVAILABLE,
            'logger': LOGGER_AVAILABLE,
            'helpers': HELPERS_AVAILABLE,
            'hashing': HASHING_AVAILABLE,
            'models': MODELS_AVAILABLE
        }
    
    @staticmethod
    def get_missing_dependencies() -> List[str]:
        """Get list of missing dependencies"""
        deps = DependencyChecker.check_dependencies()
        return [name for name, available in deps.items() if not available]
    
    @staticmethod
    def log_dependency_status():
        """Log status of all dependencies"""
        deps = DependencyChecker.check_dependencies()
        missing = DependencyChecker.get_missing_dependencies()
        
        log_statement('info', f"{LOG_INS}:INFO>>Dependency status: {len(deps) - len(missing)}/{len(deps)} available", 
                     Path(__file__).stem)
        
        if missing:
            log_statement('warning', f"{LOG_INS}:WARNING>>Missing dependencies: {', '.join(missing)}", 
                         Path(__file__).stem)
            log_statement('warning', f"{LOG_INS}:WARNING>>Some features may be unavailable or use fallbacks", 
                         Path(__file__).stem)

# Utility functions
def validate_path(path: PathLike, must_exist: bool = False, must_be_file: bool = False) -> Path:
    """Validate and resolve a path"""
    resolved_path = _ensure_pathResolve(path)
    
    if must_exist and not resolved_path.exists():
        raise FileOperationError(f"Path does not exist: {resolved_path}")
    
    if must_be_file and resolved_path.exists() and not resolved_path.is_file():
        raise FileOperationError(f"Path is not a file: {resolved_path}")
    
    return resolved_path

def safe_operation(operation_name: str, operation_func, *args, **kwargs) -> OperationResult:
    """Execute an operation safely with consistent error handling"""
    start_time = time.time()
    result = {
        'operation': operation_name,
        'status': OperationStatus.FAILURE.value,
        'duration': 0.0,
        'error': None,
        'result': None
    }
    
    try:
        log_statement('debug', f"{LOG_INS}:DEBUG>>Starting operation: {operation_name}", Path(__file__).stem)
        operation_result = operation_func(*args, **kwargs)
        result['status'] = OperationStatus.SUCCESS.value
        result['result'] = operation_result
        log_statement('debug', f"{LOG_INS}:DEBUG>>Operation completed: {operation_name}", Path(__file__).stem)
    except Exception as e:
        result['error'] = str(e)
        log_statement('error', f"{LOG_INS}:ERROR>>Operation failed: {operation_name} - {e}", 
                     Path(__file__).stem, exc_info=True)
    finally:
        result['duration'] = time.time() - start_time
    
    return result

def get_log_prefix(frame: Optional[object] = None) -> str:
    """Generate consistent log prefix for methods"""
    if frame is None:
        frame = inspect.currentframe().f_back
    
    if frame:
        filename = Path(frame.f_code.co_filename).stem
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        return f"{filename}:{function_name}:{line_number}"
    return LOG_INS

# Check dependencies on module load
DependencyChecker.log_dependency_status()

# Critical dependency check
if not GIT_AVAILABLE and not CONFIG_AVAILABLE:
    log_statement('critical', f"{LOG_INS}:CRITICAL>>Critical dependencies missing. Repository operations may fail.", 
                 Path(__file__).stem)

# Section 2: GitOpsHelper Class - Refactored for Git Operations Only

class GitOpsHelper:
    """
    Clean Git operations helper with single responsibility.
    
    This class handles ONLY Git operations - no DataFrame or metadata operations.
    All operations use consistent error handling and logging patterns.
    """
    
    def __init__(self, repo_path: Path, create_if_missing: bool = False, 
                 auto_init_gitignore: bool = True):
        """
        Initialize Git operations helper.
        
        Args:
            repo_path: Path to the Git repository
            create_if_missing: Whether to create repository if it doesn't exist
            auto_init_gitignore: Whether to automatically create .gitignore
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        
        if not GIT_AVAILABLE:
            raise DependencyError("GitPython is required for Git operations")
        
        self.repo_path = validate_path(repo_path)
        self.git_dir = self.repo_path / ".git"
        self.create_if_missing = create_if_missing
        self.auto_init_gitignore = auto_init_gitignore
        
        # Initialize repository state
        self.repo: Optional[Repo] = None
        self.is_new_repo = False
        
        # Initialize the repository
        self._initialize_repository()
        
        log_statement('info', f"{self.log_prefix}:INFO>>GitOpsHelper initialized for {self.repo_path}", 
                     Path(__file__).stem)
    
    def _initialize_repository(self) -> None:
        """Initialize or load Git repository with comprehensive error handling."""
        operation_result = safe_operation("initialize_repository", self._do_initialize_repository)
        
        if operation_result['status'] != OperationStatus.SUCCESS.value:
            error_msg = operation_result.get('error', 'Unknown error')
            if not self.create_if_missing:
                raise GitOperationError(f"Failed to initialize repository: {error_msg}")
            log_statement('warning', f"{self.log_prefix}:WARNING>>Repository initialization failed, but continuing: {error_msg}", 
                         Path(__file__).stem)
    
    def _do_initialize_repository(self) -> bool:
        """Internal method to perform repository initialization."""
        if self._load_existing_repository():
            return True
        
        if self.create_if_missing:
            return self._create_new_repository()
        
        raise GitOperationError(f"Git repository not found at {self.repo_path} and create_if_missing is False")
    
    def _load_existing_repository(self) -> bool:
        """Attempt to load an existing Git repository."""
        if not (self.git_dir.exists() and self.git_dir.is_dir()):
            return False
        
        try:
            self.repo = Repo(self.repo_path)
            log_statement('info', f"{self.log_prefix}:INFO>>Loaded existing Git repository", Path(__file__).stem)
            
            if self.auto_init_gitignore:
                self._ensure_gitignore()
            
            return True
        except InvalidGitRepositoryError:
            log_statement('error', f"{self.log_prefix}:ERROR>>Invalid Git repository at {self.git_dir}", 
                         Path(__file__).stem)
            return False
    
    def _create_new_repository(self) -> bool:
        """Create a new Git repository."""
        try:
            # Ensure directory exists
            self.repo_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize repository
            self.repo = Repo.init(str(self.repo_path))
            self.is_new_repo = True
            
            log_statement('info', f"{self.log_prefix}:INFO>>Created new Git repository at {self.repo_path}", 
                         Path(__file__).stem)
            
            if self.auto_init_gitignore:
                self._ensure_gitignore()
            
            return True
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to create Git repository: {e}", 
                         Path(__file__).stem, exc_info=True)
            self.repo = None
            self.is_new_repo = False
            raise GitOperationError(f"Failed to create Git repository: {e}")
    
    def _ensure_gitignore(self) -> None:
        """Ensure .gitignore file exists with basic configuration."""
        gitignore_path = self.repo_path / GITIGNORE_FILENAME
        
        if gitignore_path.exists():
            log_statement('debug', f"{self.log_prefix}:DEBUG>>.gitignore already exists", Path(__file__).stem)
            return
        
        try:
            with open(gitignore_path, "w", encoding='utf-8') as f:
                f.write("# Generated by GitOpsHelper\n")
                f.write(f"{METADATA_FILENAME}\n")
                f.write("*.tmp\n")
                f.write("*.log\n")
                f.write("__pycache__/\n")
                f.write(".DS_Store\n")
            
            log_statement('info', f"{self.log_prefix}:INFO>>Created .gitignore file", Path(__file__).stem)
        except IOError as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to create .gitignore: {e}", 
                         Path(__file__).stem, exc_info=True)
    
    def is_valid_repo(self) -> bool:
        """Check if the Git repository is valid and accessible."""
        if not self.repo:
            return False
        
        try:
            # Test repository access by checking if we can get the git directory
            return self.repo.git_dir is not None and Path(self.repo.git_dir).exists()
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, List[str]]:
        """
        Get comprehensive repository status.
        
        Returns:
            Dictionary with lists of files by status: modified, added, deleted, untracked
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{self.log_prefix}:WARNING>>Cannot get status: invalid repository", 
                         Path(__file__).stem)
            return {}
        
        def _do_get_status():
            status = {
                'modified': [],
                'added': [],
                'deleted': [],
                'untracked': [],
                'staged_modified': [],
                'staged_added': [],
                'staged_deleted': []
            }
            
            # Get working tree changes (unstaged)
            for item in self.repo.index.diff(None):
                file_path = item.a_path or item.b_path
                if item.change_type == 'M':
                    status['modified'].append(file_path)
                elif item.change_type == 'A':
                    status['added'].append(file_path)
                elif item.change_type == 'D':
                    status['deleted'].append(file_path)
            
            # Get staged changes
            try:
                for item in self.repo.index.diff("HEAD"):
                    file_path = item.a_path or item.b_path
                    if item.change_type == 'M':
                        status['staged_modified'].append(file_path)
                    elif item.change_type == 'A':
                        status['staged_added'].append(file_path)
                    elif item.change_type == 'D':
                        status['staged_deleted'].append(file_path)
            except BadName:
                # No HEAD exists (new repository)
                pass
            
            # Get untracked files
            status['untracked'] = list(self.repo.untracked_files)
            
            return status
        
        operation_result = safe_operation("get_status", _do_get_status)
        return operation_result.get('result', {})
    
    def add_files(self, file_paths: List[PathLike]) -> bool:
        """
        Add files to Git index.
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{self.log_prefix}:WARNING>>Cannot add files: invalid repository", 
                         Path(__file__).stem)
            return False
        
        if not file_paths:
            log_statement('warning', f"{self.log_prefix}:WARNING>>No files provided to add", 
                         Path(__file__).stem)
            return True
        
        def _do_add_files():
            # Convert paths to strings relative to repository root
            relative_paths = []
            for file_path in file_paths:
                try:
                    abs_path = validate_path(file_path, must_exist=True)
                    rel_path = abs_path.relative_to(self.repo_path)
                    relative_paths.append(str(rel_path))
                except (FileOperationError, ValueError) as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Skipping invalid path {file_path}: {e}", 
                                 Path(__file__).stem)
                    continue
            
            if not relative_paths:
                raise GitOperationError("No valid files to add")
            
            # Add files to index
            self.repo.index.add(relative_paths)
            log_statement('info', f"{self.log_prefix}:INFO>>Added {len(relative_paths)} files to index", 
                         Path(__file__).stem)
            return True
        
        operation_result = safe_operation("add_files", _do_add_files)
        return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def commit_changes(self, files: Optional[List[PathLike]] = None, message: str = "") -> bool:
        """
        Commit changes to repository.
        
        Args:
            files: Optional list of specific files to commit. If None, commits all staged changes.
            message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{self.log_prefix}:WARNING>>Cannot commit: invalid repository", 
                         Path(__file__).stem)
            return False
        
        if not message.strip():
            message = f"Automated commit at {datetime.now(timezone.utc).isoformat()}"
        
        def _do_commit():
            # Add specific files if provided
            if files:
                if not self.add_files(files):
                    raise GitOperationError("Failed to stage files for commit")
            
            # Check if there are changes to commit
            if not self._has_staged_changes():
                log_statement('info', f"{self.log_prefix}:INFO>>No changes to commit", Path(__file__).stem)
                return True
            
            # Create commit
            commit = self.repo.index.commit(message)
            log_statement('info', f"{self.log_prefix}:INFO>>Created commit {commit.hexsha[:8]} with message: {message}", 
                         Path(__file__).stem)
            return True
        
        operation_result = safe_operation("commit_changes", _do_commit)
        return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def _has_staged_changes(self) -> bool:
        """Check if there are staged changes ready to commit."""
        try:
            # Check if index differs from HEAD
            return len(list(self.repo.index.diff("HEAD"))) > 0
        except BadName:
            # No HEAD exists, so any staged files represent changes
            return len(self.repo.index.entries) > 0
    
    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        """
        Get the Git blob hash for a file's current content.
        
        Args:
            file_rel_path: Relative path to the file from repository root
            
        Returns:
            Blob hash string or None if failed
        """
        if not self.is_valid_repo():
            return None
        
        def _do_get_blob_hash():
            abs_path = self.repo_path / file_rel_path
            if not abs_path.is_file():
                raise FileOperationError(f"File not found: {abs_path}")
            
            # Calculate blob hash for current file content
            with open(abs_path, 'rb') as f:
                content = f.read()
                # Git blob format: "blob {size}\0{content}"
                blob_content = f"blob {len(content)}\0".encode() + content
                return hashlib.sha1(blob_content).hexdigest()
        
        operation_result = safe_operation("get_file_blob_hash", _do_get_blob_hash)
        return operation_result.get('result')
    
    def get_file_last_commit_hash(self, file_path: PathLike) -> Optional[str]:
        """
        Get the last commit hash that modified a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Commit hash string or None if not found
        """
        if not self.is_valid_repo():
            return None
        
        def _do_get_last_commit():
            try:
                # Ensure file_path is relative to repository root
                abs_path = validate_path(file_path)
                rel_path = abs_path.relative_to(self.repo_path)
                
                # Get commits that modified this file
                commits = list(self.repo.iter_commits(paths=str(rel_path), max_count=1))
                return commits[0].hexsha if commits else None
            except (ValueError, GitCommandError) as e:
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Could not get last commit for {file_path}: {e}", 
                             Path(__file__).stem)
                return None
        
        operation_result = safe_operation("get_file_last_commit", _do_get_last_commit)
        return operation_result.get('result')
    
    def execute_git_command(self, command: List[str], suppress_errors: bool = False, **kwargs) -> Optional[str]:
        """
        Execute a Git command safely.
        
        Args:
            command: Git command and arguments (e.g., ['status', '--porcelain'])
            suppress_errors: If True, returns None on error instead of raising
            **kwargs: Additional arguments for git command
            
        Returns:
            Command output or None if failed and suppress_errors=True
        """
        if not self.is_valid_repo():
            if suppress_errors:
                return None
            raise GitOperationError("Git repository not initialized")
        
        def _do_execute_command():
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Executing git {' '.join(command)}", 
                         Path(__file__).stem)
            
            # Use GitPython's git interface
            result = self.repo.git.execute(command, **kwargs)
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Git command completed successfully", 
                         Path(__file__).stem)
            return result
        
        operation_result = safe_operation("execute_git_command", _do_execute_command)
        
        if operation_result['status'] != OperationStatus.SUCCESS.value:
            if suppress_errors:
                return None
            error_msg = operation_result.get('error', 'Unknown error')
            raise GitOperationError(f"Git command failed: {error_msg}")
        
        return operation_result.get('result')
    
    def get_commit_history(self, max_count: Optional[int] = None, 
                          file_path: Optional[PathLike] = None) -> List[Dict[str, str]]:
        """
        Get commit history for repository or specific file.
        
        Args:
            max_count: Maximum number of commits to retrieve
            file_path: Optional path to get history for specific file
            
        Returns:
            List of commit information dictionaries
        """
        if not self.is_valid_repo():
            return []
        
        def _do_get_history():
            commits = []
            
            # Prepare arguments for iter_commits
            kwargs = {}
            if max_count:
                kwargs['max_count'] = max_count
            if file_path:
                abs_path = validate_path(file_path)
                rel_path = abs_path.relative_to(self.repo_path)
                kwargs['paths'] = str(rel_path)
            
            # Get commits
            for commit in self.repo.iter_commits(**kwargs):
                commits.append({
                    'hash': commit.hexsha,
                    'short_hash': commit.hexsha[:8],
                    'author': commit.author.name,
                    'author_email': commit.author.email,
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message.strip()
                })
            
            return commits
        
        operation_result = safe_operation("get_commit_history", _do_get_history)
        return operation_result.get('result', [])
    
    def clean_working_directory(self, force: bool = False, 
                               remove_untracked_dirs: bool = False) -> bool:
        """
        Clean the working directory of untracked files and directories.
        
        Args:
            force: Force removal of untracked files
            remove_untracked_dirs: Also remove untracked directories
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            return False
        
        def _do_clean():
            cmd = ['clean']
            if force:
                cmd.append('-f')
            if remove_untracked_dirs:
                cmd.append('-d')
            
            self.execute_git_command(cmd)
            log_statement('info', f"{self.log_prefix}:INFO>>Working directory cleaned", Path(__file__).stem)
            return True
        
        operation_result = safe_operation("clean_working_directory", _do_clean)
        return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def __repr__(self) -> str:
        """String representation of GitOpsHelper."""
        status = "valid" if self.is_valid_repo() else "invalid"
        return f"GitOpsHelper(repo_path={self.repo_path}, status={status})"

# Section 3: File Handlers - Metadata and Progress

class CompressionHandler:
    """Utility class for handling file compression operations."""
    
    @staticmethod
    def get_available_compression_types() -> List[str]:
        """Get list of available compression types."""
        available = []
        if GZIP_AVAILABLE:
            available.append('gzip')
        if ZSTD_AVAILABLE:
            available.append('zstd')
        return available
    
    @staticmethod
    def compress_content(content: Union[str, bytes], compression_type: str) -> bytes:
        """
        Compress content using specified compression type.
        
        Args:
            content: Content to compress (string or bytes)
            compression_type: Type of compression ('gzip' or 'zstd')
            
        Returns:
            Compressed content as bytes
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        if compression_type == 'gzip' and GZIP_AVAILABLE:
            return gzip.compress(content)
        elif compression_type == 'zstd' and ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(content)
        else:
            raise ValueError(f"Unsupported or unavailable compression type: {compression_type}")
    
    @staticmethod
    def decompress_content(compressed_content: bytes, compression_type: str) -> bytes:
        """
        Decompress content using specified compression type.
        
        Args:
            compressed_content: Compressed content as bytes
            compression_type: Type of compression ('gzip' or 'zstd')
            
        Returns:
            Decompressed content as bytes
        """
        if compression_type == 'gzip' and GZIP_AVAILABLE:
            return gzip.decompress(compressed_content)
        elif compression_type == 'zstd' and ZSTD_AVAILABLE:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(compressed_content)
        else:
            raise ValueError(f"Unsupported or unavailable compression type: {compression_type}")
    
    @staticmethod
    def detect_compression_type(file_path: Path) -> Optional[str]:
        """
        Detect compression type from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Compression type or None if not compressed
        """
        suffixes = file_path.suffixes
        if '.gz' in suffixes:
            return 'gzip'
        elif '.zst' in suffixes:
            return 'zstd'
        return None


class MetadataFileHandler:
    """
    Handles reading and writing of metadata files with compression support.
    
    This class is responsible ONLY for file I/O operations - no Git operations.
    Supports multiple compression formats with graceful fallbacks.
    """
    
    def __init__(self, metadata_path: Path, use_compression: Optional[str] = None):
        """
        Initialize metadata file handler.
        
        Args:
            metadata_path: Path to the metadata file
            use_compression: Compression type to use ('gzip', 'zstd', or None)
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.metadata_path = validate_path(metadata_path)
        self.use_compression = use_compression
        self._lock = RLock()
        
        # Validate compression type
        if self.use_compression:
            available_types = CompressionHandler.get_available_compression_types()
            if self.use_compression not in available_types:
                log_statement('warning', 
                             f"{self.log_prefix}:WARNING>>Compression '{self.use_compression}' not available. "
                             f"Available: {available_types}. Using uncompressed.", 
                             Path(__file__).stem)
                self.use_compression = None
        
        # Adjust file path for compression
        if self.use_compression:
            if not self.metadata_path.name.endswith(f'.{self.use_compression}'):
                if self.use_compression == 'gzip':
                    self.metadata_path = self.metadata_path.with_suffix(f'{self.metadata_path.suffix}.gz')
                elif self.use_compression == 'zstd':
                    self.metadata_path = self.metadata_path.with_suffix(f'{self.metadata_path.suffix}.zst')
        
        log_statement('info', f"{self.log_prefix}:INFO>>MetadataFileHandler initialized: {self.metadata_path}", 
                     Path(__file__).stem)
    
    def ensure_metadata_file_exists(self) -> bool:
        """
        Ensure metadata file exists, creating empty one if needed.
        
        Returns:
            True if file exists or was created successfully
        """
        with self._lock:
            if self.metadata_path.exists():
                return True
            
            def _create_empty_metadata():
                # Ensure parent directory exists
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create empty metadata structure
                empty_metadata = {}
                return self.write_metadata(empty_metadata)
            
            operation_result = safe_operation("ensure_metadata_file_exists", _create_empty_metadata)
            
            if operation_result['status'] == OperationStatus.SUCCESS.value:
                log_statement('info', f"{self.log_prefix}:INFO>>Created empty metadata file", 
                             Path(__file__).stem)
                return True
            else:
                log_statement('error', f"{self.log_prefix}:ERROR>>Failed to create metadata file: "
                             f"{operation_result.get('error')}", Path(__file__).stem)
                return False
    
    def read_metadata(self) -> Dict[str, Any]:
        """
        Read metadata from file with automatic compression detection.
        
        Returns:
            Dictionary containing metadata, empty dict if file doesn't exist or failed to read
        """
        with self._lock:
            if not self.metadata_path.exists():
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Metadata file not found: {self.metadata_path}", 
                             Path(__file__).stem)
                return {}
            
            def _do_read_metadata():
                # Try to detect compression if not explicitly set
                compression_type = self.use_compression or CompressionHandler.detect_compression_type(self.metadata_path)
                
                # Read file content
                with open(self.metadata_path, 'rb') as f:
                    content = f.read()
                
                # Decompress if needed
                if compression_type:
                    try:
                        content = CompressionHandler.decompress_content(content, compression_type)
                    except Exception as e:
                        log_statement('warning', 
                                     f"{self.log_prefix}:WARNING>>Decompression failed, trying as uncompressed: {e}", 
                                     Path(__file__).stem)
                        # Reset file pointer and read as uncompressed
                        with open(self.metadata_path, 'rb') as f:
                            content = f.read()
                
                # Parse JSON
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                if not content.strip():
                    return {}
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>JSON decode error in metadata file: {e}", 
                                 Path(__file__).stem)
                    return {}
            
            operation_result = safe_operation("read_metadata", _do_read_metadata)
            
            if operation_result['status'] == OperationStatus.SUCCESS.value:
                result = operation_result.get('result', {})
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Read metadata with {len(result)} entries", 
                             Path(__file__).stem)
                return result
            else:
                log_statement('error', f"{self.log_prefix}:ERROR>>Failed to read metadata: "
                             f"{operation_result.get('error')}", Path(__file__).stem)
                return {}
    
    def write_metadata(self, data: Dict[str, Any], backup_existing: bool = True) -> bool:
        """
        Write metadata to file with optional compression.
        
        Args:
            data: Metadata dictionary to write
            backup_existing: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            def _do_write_metadata():
                # Create backup if requested and file exists
                if backup_existing and self.metadata_path.exists():
                    self._create_backup()
                
                # Ensure parent directory exists
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Serialize to JSON
                json_content = json.dumps(data, indent=2, sort_keys=True, default=str)
                
                # Compress if needed
                if self.use_compression:
                    content = CompressionHandler.compress_content(json_content, self.use_compression)
                    mode = 'wb'
                else:
                    content = json_content
                    mode = 'w'
                
                # Write to temporary file first for atomic operation
                temp_path = self.metadata_path.with_suffix(f'{self.metadata_path.suffix}.tmp')
                
                try:
                    with open(temp_path, mode, encoding='utf-8' if mode == 'w' else None) as f:
                        f.write(content)
                    
                    # Atomic move
                    shutil.move(str(temp_path), str(self.metadata_path))
                    
                    log_statement('info', f"{self.log_prefix}:INFO>>Wrote metadata with {len(data)} entries", 
                                 Path(__file__).stem)
                    return True
                    
                except Exception as e:
                    # Clean up temp file if it exists
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    raise e
            
            operation_result = safe_operation("write_metadata", _do_write_metadata)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def _create_backup(self) -> Optional[Path]:
        """Create a backup of the existing metadata file."""
        if not self.metadata_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.metadata_path.with_suffix(f'{self.metadata_path.suffix}.backup_{timestamp}')
        
        try:
            shutil.copy2(self.metadata_path, backup_path)
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Created backup: {backup_path}", 
                         Path(__file__).stem)
            return backup_path
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to create backup: {e}", 
                         Path(__file__).stem)
            return None
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about the metadata file."""
        if not self.metadata_path.exists():
            return {'exists': False}
        
        try:
            stat = self.metadata_path.stat()
            return {
                'exists': True,
                'size_bytes': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                'compression': self.use_compression,
                'readable': os.access(self.metadata_path, os.R_OK),
                'writable': os.access(self.metadata_path, os.W_OK)
            }
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get file stats: {e}", 
                         Path(__file__).stem)
            return {'exists': True, 'error': str(e)}


class ProgressFileHandler:
    """
    Handles saving and loading of progress files for long-running operations.
    
    This class manages progress tracking without any Git integration.
    Progress files are stored as JSON with optional compression.
    """
    
    def __init__(self, progress_dir: Path, use_compression: bool = False):
        """
        Initialize progress file handler.
        
        Args:
            progress_dir: Directory to store progress files
            use_compression: Whether to use compression for progress files
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.progress_dir = validate_path(progress_dir)
        self.use_compression = use_compression and bool(CompressionHandler.get_available_compression_types())
        self._lock = RLock()
        
        # Ensure progress directory exists
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        log_statement('info', f"{self.log_prefix}:INFO>>ProgressFileHandler initialized: {self.progress_dir}", 
                     Path(__file__).stem)
    
    def save_progress(self, process_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        Save progress data for a specific process.
        
        Args:
            process_id: Unique identifier for the process
            progress_data: Progress data to save
            
        Returns:
            True if successful, False otherwise
        """
        if not process_id or not isinstance(process_id, str):
            log_statement('error', f"{self.log_prefix}:ERROR>>Invalid process_id: {process_id}", 
                         Path(__file__).stem)
            return False
        
        with self._lock:
            def _do_save_progress():
                # Prepare progress data with metadata
                complete_data = {
                    'process_id': process_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'progress': progress_data
                }
                
                # Determine file path
                extension = '.json.gz' if self.use_compression else '.json'
                progress_file = self.progress_dir / f"progress_{process_id}{extension}"
                
                # Serialize to JSON
                json_content = json.dumps(complete_data, indent=2, default=str)
                
                # Compress if needed
                if self.use_compression:
                    content = CompressionHandler.compress_content(json_content, 'gzip')
                    mode = 'wb'
                else:
                    content = json_content
                    mode = 'w'
                
                # Write file
                with open(progress_file, mode, encoding='utf-8' if mode == 'w' else None) as f:
                    f.write(content)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Saved progress for process: {process_id}", 
                             Path(__file__).stem)
                return True
            
            operation_result = safe_operation("save_progress", _do_save_progress)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def load_progress(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Load progress data for a specific process.
        
        Args:
            process_id: Unique identifier for the process
            
        Returns:
            Progress data dictionary or None if not found/failed
        """
        if not process_id or not isinstance(process_id, str):
            log_statement('error', f"{self.log_prefix}:ERROR>>Invalid process_id: {process_id}", 
                         Path(__file__).stem)
            return None
        
        with self._lock:
            def _do_load_progress():
                # Try both compressed and uncompressed files
                for extension in ['.json.gz', '.json']:
                    progress_file = self.progress_dir / f"progress_{process_id}{extension}"
                    
                    if not progress_file.exists():
                        continue
                    
                    # Read file content
                    with open(progress_file, 'rb') as f:
                        content = f.read()
                    
                    # Decompress if needed
                    if extension.endswith('.gz'):
                        try:
                            content = CompressionHandler.decompress_content(content, 'gzip')
                        except Exception as e:
                            log_statement('warning', 
                                         f"{self.log_prefix}:WARNING>>Failed to decompress progress file: {e}", 
                                         Path(__file__).stem)
                            continue
                    
                    # Parse JSON
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    
                    try:
                        data = json.loads(content)
                        log_statement('info', f"{self.log_prefix}:INFO>>Loaded progress for process: {process_id}", 
                                     Path(__file__).stem)
                        return data.get('progress', data)  # Return just progress data or full data if no 'progress' key
                    except json.JSONDecodeError as e:
                        log_statement('error', f"{self.log_prefix}:ERROR>>Failed to parse progress file: {e}", 
                                     Path(__file__).stem)
                        continue
                
                log_statement('debug', f"{self.log_prefix}:DEBUG>>No progress file found for process: {process_id}", 
                             Path(__file__).stem)
                return None
            
            operation_result = safe_operation("load_progress", _do_load_progress)
            return operation_result.get('result')
    
    def delete_progress(self, process_id: str) -> bool:
        """
        Delete progress file for a specific process.
        
        Args:
            process_id: Unique identifier for the process
            
        Returns:
            True if successful or file didn't exist, False on error
        """
        if not process_id or not isinstance(process_id, str):
            log_statement('error', f"{self.log_prefix}:ERROR>>Invalid process_id: {process_id}", 
                         Path(__file__).stem)
            return False
        
        with self._lock:
            def _do_delete_progress():
                deleted_any = False
                
                # Try to delete both compressed and uncompressed files
                for extension in ['.json.gz', '.json']:
                    progress_file = self.progress_dir / f"progress_{process_id}{extension}"
                    
                    if progress_file.exists():
                        progress_file.unlink()
                        deleted_any = True
                
                if deleted_any:
                    log_statement('info', f"{self.log_prefix}:INFO>>Deleted progress for process: {process_id}", 
                                 Path(__file__).stem)
                
                return True
            
            operation_result = safe_operation("delete_progress", _do_delete_progress)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def list_processes(self) -> List[str]:
        """
        List all process IDs that have progress files.
        
        Returns:
            List of process IDs
        """
        def _do_list_processes():
            process_ids = set()
            
            for file_path in self.progress_dir.glob("progress_*"):
                if file_path.is_file():
                    # Extract process ID from filename
                    filename = file_path.name
                    if filename.startswith("progress_"):
                        # Remove prefix and extensions
                        process_id = filename[9:]  # Remove "progress_"
                        process_id = process_id.replace('.json.gz', '').replace('.json', '')
                        process_ids.add(process_id)
            
            return sorted(list(process_ids))
        
        operation_result = safe_operation("list_processes", _do_list_processes)
        return operation_result.get('result', [])
    
    def cleanup_old_progress(self, max_age_days: int = 7) -> int:
        """
        Clean up old progress files.
        
        Args:
            max_age_days: Maximum age of progress files to keep
            
        Returns:
            Number of files deleted
        """
        def _do_cleanup():
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in self.progress_dir.glob("progress_*"):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        log_statement('warning', 
                                     f"{self.log_prefix}:WARNING>>Failed to delete old progress file {file_path}: {e}", 
                                     Path(__file__).stem)
            
            if deleted_count > 0:
                log_statement('info', f"{self.log_prefix}:INFO>>Cleaned up {deleted_count} old progress files", 
                             Path(__file__).stem)
            
            return deleted_count
        
        operation_result = safe_operation("cleanup_old_progress", _do_cleanup)
        return operation_result.get('result', 0)

# Section 4: GitignoreHandler and RepoAnalyzer Classes

class GitignoreHandler:
    """
    Handles .gitignore file operations with clean separation from Git operations.
    
    This class focuses solely on .gitignore file management - reading, writing,
    and pattern matching. Git operations are handled separately.
    """
    
    def __init__(self, repo_path: Path, gitignore_filename: str = GITIGNORE_FILENAME):
        """
        Initialize gitignore handler.
        
        Args:
            repo_path: Path to the repository root
            gitignore_filename: Name of the gitignore file (usually '.gitignore')
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.repo_path = validate_path(repo_path)
        self.gitignore_path = self.repo_path / gitignore_filename
        self._lock = RLock()
        
        log_statement('info', f"{self.log_prefix}:INFO>>GitignoreHandler initialized for {self.gitignore_path}", 
                     Path(__file__).stem)
    
    def ensure_gitignore_exists(self, default_patterns: Optional[List[str]] = None) -> bool:
        """
        Ensure .gitignore file exists with default patterns.
        
        Args:
            default_patterns: Optional list of default patterns to include
            
        Returns:
            True if file exists or was created successfully
        """
        with self._lock:
            if self.gitignore_path.exists():
                return True
            
            def _create_gitignore():
                patterns = default_patterns or [
                    "# Generated gitignore file",
                    "*.tmp",
                    "*.log",
                    "__pycache__/",
                    "*.pyc",
                    ".DS_Store",
                    "Thumbs.db"
                ]
                
                self.gitignore_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.gitignore_path, 'w', encoding='utf-8') as f:
                    for pattern in patterns:
                        f.write(f"{pattern}\n")
                
                log_statement('info', f"{self.log_prefix}:INFO>>Created .gitignore with {len(patterns)} patterns", 
                             Path(__file__).stem)
                return True
            
            operation_result = safe_operation("create_gitignore", _create_gitignore)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def read_patterns(self) -> List[str]:
        """
        Read all patterns from .gitignore file.
        
        Returns:
            List of gitignore patterns (excluding comments and empty lines)
        """
        if not self.gitignore_path.exists():
            log_statement('debug', f"{self.log_prefix}:DEBUG>>.gitignore file not found", Path(__file__).stem)
            return []
        
        def _do_read_patterns():
            patterns = []
            
            with open(self.gitignore_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    patterns.append(line)
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Read {len(patterns)} patterns from .gitignore", 
                         Path(__file__).stem)
            return patterns
        
        operation_result = safe_operation("read_patterns", _do_read_patterns)
        return operation_result.get('result', [])
    
    def add_patterns(self, patterns: List[str], avoid_duplicates: bool = True) -> bool:
        """
        Add patterns to .gitignore file.
        
        Args:
            patterns: List of patterns to add
            avoid_duplicates: Whether to check for existing patterns
            
        Returns:
            True if successful, False otherwise
        """
        if not patterns:
            return True
        
        with self._lock:
            def _do_add_patterns():
                # Ensure gitignore exists
                if not self.ensure_gitignore_exists():
                    raise FileOperationError("Failed to ensure .gitignore exists")
                
                existing_patterns = set()
                if avoid_duplicates:
                    existing_patterns = set(self.read_patterns())
                
                # Filter out duplicates
                new_patterns = []
                for pattern in patterns:
                    pattern = pattern.strip()
                    if pattern and (not avoid_duplicates or pattern not in existing_patterns):
                        new_patterns.append(pattern)
                
                if not new_patterns:
                    log_statement('info', f"{self.log_prefix}:INFO>>No new patterns to add", Path(__file__).stem)
                    return True
                
                # Append new patterns
                with open(self.gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n')  # Ensure newline before new patterns
                    for pattern in new_patterns:
                        f.write(f"{pattern}\n")
                
                log_statement('info', f"{self.log_prefix}:INFO>>Added {len(new_patterns)} patterns to .gitignore", 
                             Path(__file__).stem)
                return True
            
            operation_result = safe_operation("add_patterns", _do_add_patterns)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def remove_patterns(self, patterns_to_remove: List[str]) -> bool:
        """
        Remove patterns from .gitignore file.
        
        Args:
            patterns_to_remove: List of patterns to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not patterns_to_remove or not self.gitignore_path.exists():
            return True
        
        with self._lock:
            def _do_remove_patterns():
                patterns_set = set(patterns_to_remove)
                
                # Read all lines (including comments and empty lines)
                with open(self.gitignore_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Filter out patterns to remove
                new_lines = []
                removed_count = 0
                
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line in patterns_set:
                        removed_count += 1
                        continue
                    new_lines.append(line)
                
                # Write back the filtered content
                with open(self.gitignore_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Removed {removed_count} patterns from .gitignore", 
                             Path(__file__).stem)
                return True
            
            operation_result = safe_operation("remove_patterns", _do_remove_patterns)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def is_path_ignored_by_patterns(self, file_path: PathLike, 
                                   custom_patterns: Optional[List[str]] = None) -> bool:
        """
        Check if a path would be ignored by gitignore patterns.
        
        Args:
            file_path: Path to check
            custom_patterns: Optional custom patterns to use instead of file patterns
            
        Returns:
            True if path matches any ignore pattern
        """
        patterns = custom_patterns if custom_patterns is not None else self.read_patterns()
        
        if not patterns:
            return False
        
        # Convert to relative path for pattern matching
        try:
            abs_path = validate_path(file_path)
            if abs_path.is_absolute():
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                except ValueError:
                    # Path is outside repository
                    return False
            else:
                rel_path = abs_path
            
            path_str = str(rel_path).replace('\\', '/')  # Use forward slashes for patterns
            
            # Check each pattern
            for pattern in patterns:
                if self._matches_pattern(path_str, pattern):
                    return True
            
            return False
            
        except Exception as e:
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Error checking pattern match for {file_path}: {e}", 
                         Path(__file__).stem)
            return False
    
    def _matches_pattern(self, path_str: str, pattern: str) -> bool:
        """
        Check if a path matches a gitignore pattern.
        
        This is a simplified implementation. For full gitignore compatibility,
        consider using a dedicated library like pathspec.
        """
        import fnmatch
        
        # Handle directory patterns
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            # Check if any parent directory matches
            parts = path_str.split('/')
            for i in range(len(parts)):
                dir_path = '/'.join(parts[:i+1])
                if fnmatch.fnmatch(dir_path, pattern):
                    return True
            return False
        
        # Handle negation patterns
        if pattern.startswith('!'):
            # This is a negation pattern - would need more complex logic
            # For now, treat as non-matching
            return False
        
        # Handle patterns starting with /
        if pattern.startswith('/'):
            pattern = pattern[1:]
            return fnmatch.fnmatch(path_str, pattern)
        
        # Handle patterns with wildcards
        if fnmatch.fnmatch(path_str, pattern):
            return True
        
        # Check if pattern matches any part of the path
        parts = path_str.split('/')
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        
        return False
    
    def get_content(self) -> str:
        """
        Get the complete content of .gitignore file.
        
        Returns:
            Content of .gitignore file or empty string if not found
        """
        if not self.gitignore_path.exists():
            return ""
        
        def _do_get_content():
            with open(self.gitignore_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        operation_result = safe_operation("get_content", _do_get_content)
        return operation_result.get('result', "")
    
    def set_content(self, content: str, backup: bool = True) -> bool:
        """
        Set the complete content of .gitignore file.
        
        Args:
            content: New content for .gitignore
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            def _do_set_content():
                # Create backup if requested
                if backup and self.gitignore_path.exists():
                    backup_path = self.gitignore_path.with_suffix(f'.backup_{int(time.time())}')
                    shutil.copy2(self.gitignore_path, backup_path)
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Created backup: {backup_path}", 
                                 Path(__file__).stem)
                
                # Ensure parent directory exists
                self.gitignore_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content
                with open(self.gitignore_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Updated .gitignore content", Path(__file__).stem)
                return True
            
            operation_result = safe_operation("set_content", _do_set_content)
            return operation_result['status'] == OperationStatus.SUCCESS.value


class RepoAnalyzer:
    """
    Read-only repository analysis and scanning operations.
    
    This class provides methods for analyzing repository state, detecting
    discrepancies, and performing integrity checks without modifying anything.
    """
    
    def __init__(self, repo_path: Path, git_ops: Optional[GitOpsHelper] = None,
                 metadata_handler: Optional[MetadataFileHandler] = None):
        """
        Initialize repository analyzer.
        
        Args:
            repo_path: Path to the repository
            git_ops: Optional GitOpsHelper instance for Git operations
            metadata_handler: Optional metadata handler for reading metadata
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.repo_path = validate_path(repo_path)
        self.git_ops = git_ops
        self.metadata_handler = metadata_handler
        
        log_statement('info', f"{self.log_prefix}:INFO>>RepoAnalyzer initialized for {self.repo_path}", 
                     Path(__file__).stem)
    
    def scan_directory_files(self, include_ignored: bool = False, 
                           file_extensions: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Scan directory for all files and gather basic information.
        
        Args:
            include_ignored: Whether to include files that would be ignored by .gitignore
            file_extensions: Optional set of file extensions to include (e.g., {'.py', '.txt'})
            
        Returns:
            List of file information dictionaries
        """
        def _do_scan():
            files_info = []
            gitignore_handler = GitignoreHandler(self.repo_path) if not include_ignored else None
            
            for root, dirs, files in os.walk(self.repo_path):
                root_path = Path(root)
                
                # Skip .git directory
                if '.git' in root_path.parts:
                    continue
                
                # Skip hidden directories unless explicitly included
                dirs[:] = [d for d in dirs if not d.startswith('.') or d in ['.tlato']]
                
                for filename in files:
                    file_path = root_path / filename
                    
                    # Skip hidden files unless explicitly included
                    if filename.startswith('.') and filename not in ['.gitignore', '.gitattributes']:
                        continue
                    
                    # Filter by extension if specified
                    if file_extensions and file_path.suffix.lower() not in file_extensions:
                        continue
                    
                    # Check if ignored
                    if gitignore_handler and gitignore_handler.is_path_ignored_by_patterns(file_path):
                        continue
                    
                    # Gather file information
                    try:
                        file_info = self._get_file_basic_info(file_path)
                        if file_info:
                            files_info.append(file_info)
                    except Exception as e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to get info for {file_path}: {e}", 
                                     Path(__file__).stem)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Scanned {len(files_info)} files", Path(__file__).stem)
            return files_info
        
        operation_result = safe_operation("scan_directory_files", _do_scan)
        return operation_result.get('result', [])
    
    def _get_file_basic_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic information about a file."""
        try:
            stat = file_path.stat()
            rel_path = file_path.relative_to(self.repo_path)
            
            return {
                'absolute_path': str(file_path),
                'relative_path': str(rel_path).replace('\\', '/'),
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                'created_time': datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
                'is_file': file_path.is_file(),
                'is_symlink': file_path.is_symlink(),
                'parent_directory': str(rel_path.parent) if rel_path.parent != Path('.') else '.'
            }
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get file info for {file_path}: {e}", 
                         Path(__file__).stem)
            return {}
    
    def detect_repository_discrepancies(self) -> Dict[str, List[str]]:
        """
        Detect discrepancies between filesystem, Git, and metadata.
        
        Returns:
            Dictionary categorizing different types of discrepancies
        """
        def _do_detect_discrepancies():
            discrepancies = {
                'files_not_in_git': [],
                'files_not_in_metadata': [],
                'metadata_files_missing': [],
                'git_files_missing': [],
                'modified_files': [],
                'untracked_files': []
            }
            
            # Get filesystem files
            fs_files = set()
            for file_info in self.scan_directory_files():
                fs_files.add(file_info['relative_path'])
            
            # Get Git tracked files
            git_files = set()
            if self.git_ops and self.git_ops.is_valid_repo():
                try:
                    git_output = self.git_ops.execute_git_command(['ls-files'], suppress_errors=True)
                    if git_output:
                        git_files = set(git_output.strip().split('\n'))
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Could not get Git files: {e}", 
                                 Path(__file__).stem)
            
            # Get metadata files
            metadata_files = set()
            if self.metadata_handler:
                try:
                    metadata = self.metadata_handler.read_metadata()
                    metadata_files = set(metadata.keys())
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Could not read metadata: {e}", 
                                 Path(__file__).stem)
            
            # Detect discrepancies
            discrepancies['files_not_in_git'] = list(fs_files - git_files)
            discrepancies['files_not_in_metadata'] = list(fs_files - metadata_files)
            discrepancies['metadata_files_missing'] = list(metadata_files - fs_files)
            discrepancies['git_files_missing'] = list(git_files - fs_files)
            
            # Get Git status for more detailed analysis
            if self.git_ops and self.git_ops.is_valid_repo():
                status = self.git_ops.get_status()
                discrepancies['modified_files'] = status.get('modified', [])
                discrepancies['untracked_files'] = status.get('untracked', [])
            
            # Log summary
            total_issues = sum(len(files) for files in discrepancies.values())
            log_statement('info', f"{self.log_prefix}:INFO>>Found {total_issues} total discrepancies", 
                         Path(__file__).stem)
            
            return discrepancies
        
        operation_result = safe_operation("detect_discrepancies", _do_detect_discrepancies)
        return operation_result.get('result', {})
    
    def verify_file_integrity(self, file_path: PathLike, 
                            expected_hash: Optional[str] = None,
                            hash_algorithm: str = 'sha256') -> Dict[str, Any]:
        """
        Verify integrity of a specific file.
        
        Args:
            file_path: Path to the file to verify
            expected_hash: Expected hash value (if None, just calculates current hash)
            hash_algorithm: Hash algorithm to use
            
        Returns:
            Dictionary with integrity check results
        """
        def _do_verify_integrity():
            abs_path = validate_path(file_path, must_exist=True, must_be_file=True)
            rel_path = abs_path.relative_to(self.repo_path) if abs_path.is_relative_to(self.repo_path) else abs_path
            
            result = {
                'file_path': str(rel_path),
                'exists': True,
                'current_hash': None,
                'expected_hash': expected_hash,
                'hash_algorithm': hash_algorithm,
                'integrity_verified': False,
                'file_size': abs_path.stat().st_size,
                'last_modified': datetime.fromtimestamp(abs_path.stat().st_mtime, timezone.utc).isoformat()
            }
            
            # Calculate current hash
            if HASHING_AVAILABLE:
                current_hash = generate_data_hash(abs_path, hash_algorithm)
                result['current_hash'] = current_hash
                
                # Verify against expected hash if provided
                if expected_hash:
                    result['integrity_verified'] = (current_hash == expected_hash)
                else:
                    result['integrity_verified'] = True  # No expected hash to compare against
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Hashing not available for integrity check", 
                             Path(__file__).stem)
            
            return result
        
        operation_result = safe_operation("verify_file_integrity", _do_verify_integrity)
        return operation_result.get('result', {'exists': False, 'error': operation_result.get('error')})
    
    def get_repository_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive repository summary.
        
        Returns:
            Dictionary with repository statistics and status
        """
        def _do_get_summary():
            summary = {
                'repository_path': str(self.repo_path),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'git_available': self.git_ops is not None and self.git_ops.is_valid_repo(),
                'metadata_available': self.metadata_handler is not None,
                'file_counts': {},
                'git_status': {},
                'discrepancies': {}
            }
            
            # File system analysis
            try:
                files = self.scan_directory_files()
                summary['file_counts'] = {
                    'total_files': len(files),
                    'total_size_bytes': sum(f.get('size_bytes', 0) for f in files),
                    'extensions': {}
                }
                
                # Count by extension
                for file_info in files:
                    ext = file_info.get('extension', 'no_extension')
                    summary['file_counts']['extensions'][ext] = summary['file_counts']['extensions'].get(ext, 0) + 1
                
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Error in file system analysis: {e}", 
                             Path(__file__).stem)
            
            # Git analysis
            if summary['git_available']:
                try:
                    summary['git_status'] = self.git_ops.get_status()
                    history = self.git_ops.get_commit_history(max_count=10)
                    summary['recent_commits'] = len(history)
                    summary['latest_commit'] = history[0] if history else None
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Error in Git analysis: {e}", 
                                 Path(__file__).stem)
            
            # Discrepancy analysis
            try:
                summary['discrepancies'] = self.detect_repository_discrepancies()
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Error in discrepancy analysis: {e}", 
                             Path(__file__).stem)
            
            return summary
        
        operation_result = safe_operation("get_repository_summary", _do_get_summary)
        return operation_result.get('result', {'error': operation_result.get('error')})
    
    def find_duplicate_files(self, hash_algorithm: str = 'sha256') -> Dict[str, List[str]]:
        """
        Find duplicate files based on content hash.
        
        Args:
            hash_algorithm: Hash algorithm to use for comparison
            
        Returns:
            Dictionary mapping hash values to lists of file paths with that hash
        """
        def _do_find_duplicates():
            if not HASHING_AVAILABLE:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Hashing not available for duplicate detection", 
                             Path(__file__).stem)
                return {}
            
            hash_map = {}
            files = self.scan_directory_files()
            
            for file_info in tqdm(files, desc="Calculating hashes") if TQDM_AVAILABLE else files:
                try:
                    file_path = Path(file_info['absolute_path'])
                    file_hash = generate_data_hash(file_path, hash_algorithm)
                    
                    if file_hash:
                        if file_hash not in hash_map:
                            hash_map[file_hash] = []
                        hash_map[file_hash].append(file_info['relative_path'])
                        
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to hash {file_info['relative_path']}: {e}", 
                                 Path(__file__).stem)
            
            # Return only hashes with multiple files
            duplicates = {h: files for h, files in hash_map.items() if len(files) > 1}
            
            log_statement('info', f"{self.log_prefix}:INFO>>Found {len(duplicates)} groups of duplicate files", 
                         Path(__file__).stem)
            return duplicates
        
        operation_result = safe_operation("find_duplicate_files", _do_find_duplicates)
        return operation_result.get('result', {})

# Section 5: RepoModifier Class

class RepoModifier:
    """
    Handles all repository modification operations.
    
    This class is responsible for write operations: adding files to tracking,
    updating file status, removing files, and managing repository state.
    It coordinates between Git operations and metadata management.
    """
    
    def __init__(self, repo_path: Path, git_ops: Optional[GitOpsHelper] = None,
                 metadata_handler: Optional[MetadataFileHandler] = None,
                 progress_handler: Optional[ProgressFileHandler] = None):
        """
        Initialize repository modifier.
        
        Args:
            repo_path: Path to the repository
            git_ops: GitOpsHelper instance for Git operations
            metadata_handler: MetadataFileHandler for metadata operations
            progress_handler: ProgressFileHandler for progress tracking
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.repo_path = validate_path(repo_path)
        self.git_ops = git_ops
        self.metadata_handler = metadata_handler
        self.progress_handler = progress_handler
        self._lock = RLock()
        
        # Validate required dependencies
        if not MODELS_AVAILABLE:
            raise DependencyError("Pydantic models required for RepoModifier operations")
        
        log_statement('info', f"{self.log_prefix}:INFO>>RepoModifier initialized for {self.repo_path}", 
                     Path(__file__).stem)
    
    def add_file_to_tracking(self, file_path: PathLike, 
                           application_status: str = STATUS_NEW,
                           user_metadata: Optional[Dict[str, Any]] = None,
                           change_description: str = "Initial file registration",
                           auto_commit: bool = True,
                           compression_type: Optional[str] = None) -> OperationResult:
        """
        Add a file to repository tracking with metadata.
        
        Args:
            file_path: Path to the file to track
            application_status: Initial application status
            user_metadata: Optional user-defined metadata
            change_description: Description of this change
            auto_commit: Whether to automatically commit changes
            compression_type: Optional compression to apply ('gzip' or 'zstd')
            
        Returns:
            OperationResult with success/failure status and details
        """
        with self._lock:
            def _do_add_file():
                # Validate file path
                abs_path = validate_path(file_path, must_exist=True, must_be_file=True)
                
                # Get relative path for repository operations
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                    rel_path_str = str(rel_path).replace('\\', '/')
                except ValueError:
                    raise FileOperationError(f"File {abs_path} is not within repository {self.repo_path}")
                
                # Handle compression if requested
                actual_file_path = abs_path
                original_filename = None
                
                if compression_type:
                    actual_file_path, original_filename = self._handle_file_compression(
                        abs_path, compression_type
                    )
                    rel_path = actual_file_path.relative_to(self.repo_path)
                    rel_path_str = str(rel_path).replace('\\', '/')
                
                # Gather file metadata
                file_metadata = self._gather_file_metadata(actual_file_path)
                
                # Create or update metadata entry
                metadata_entry = self._create_metadata_entry(
                    rel_path_str, file_metadata, application_status, 
                    user_metadata or {}, change_description, original_filename, compression_type
                )
                
                # Update metadata collection
                success = self._update_metadata_collection(metadata_entry)
                if not success:
                    raise MetadataError("Failed to update metadata collection")
                
                # Add to Git and commit if requested
                commit_hash = None
                if auto_commit and self.git_ops and self.git_ops.is_valid_repo():
                    commit_success = self.git_ops.commit_changes(
                        files=[actual_file_path],
                        message=f"Track file: {rel_path_str} - {change_description}"
                    )
                    
                    if commit_success:
                        commit_hash = self.git_ops.get_file_last_commit_hash(actual_file_path)
                        # Update metadata with commit hash
                        self._update_version_with_commit_hash(rel_path_str, commit_hash)
                
                result = {
                    'file_path': rel_path_str,
                    'metadata_entry': metadata_entry.model_dump(exclude_none=True) if hasattr(metadata_entry, 'model_dump') else metadata_entry,
                    'commit_hash': commit_hash,
                    'compressed': compression_type is not None,
                    'original_filename': original_filename
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Successfully added file to tracking: {rel_path_str}", 
                             Path(__file__).stem)
                return result
            
            return safe_operation("add_file_to_tracking", _do_add_file)
    
    def _handle_file_compression(self, file_path: Path, compression_type: str) -> Tuple[Path, str]:
        """Handle file compression and return new path and original filename."""
        if compression_type not in CompressionHandler.get_available_compression_types():
            raise ValueError(f"Compression type '{compression_type}' not available")
        
        # Read original file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Compress content
        compressed_content = CompressionHandler.compress_content(content, compression_type)
        
        # Create compressed file path
        if compression_type == 'gzip':
            compressed_path = file_path.with_suffix(f'{file_path.suffix}.gz')
        elif compression_type == 'zstd':
            compressed_path = file_path.with_suffix(f'{file_path.suffix}.zst')
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
        
        # Write compressed file
        with open(compressed_path, 'wb') as f:
            f.write(compressed_content)
        
        log_statement('info', f"{self.log_prefix}:INFO>>Compressed {file_path.name} to {compressed_path.name}", 
                     Path(__file__).stem)
        
        return compressed_path, file_path.name
    
    def _gather_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Gather comprehensive metadata for a file."""
        try:
            stat = file_path.stat()
            
            metadata = {
                'filename': file_path.name,
                'extension': file_path.suffix.lower().lstrip('.') if file_path.suffix else '',
                'size_bytes': stat.st_size,
                'os_last_modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                'os_created_utc': datetime.fromtimestamp(stat.st_ctime, timezone.utc),
                'custom_hashes': {}
            }
            
            # Calculate hashes if hashing is available
            if HASHING_AVAILABLE:
                for algorithm in SUPPORTED_HASH_ALGORITHMS:
                    try:
                        hash_value = generate_data_hash(file_path, algorithm)
                        if hash_value:
                            metadata['custom_hashes'][algorithm] = hash_value
                    except Exception as e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to calculate {algorithm} hash: {e}", 
                                     Path(__file__).stem)
            
            return metadata
            
        except Exception as e:
            raise FileOperationError(f"Failed to gather metadata for {file_path}: {e}")
    
    def _create_metadata_entry(self, rel_path_str: str, file_metadata: Dict[str, Any],
                              application_status: str, user_metadata: Dict[str, Any],
                              change_description: str, original_filename: Optional[str] = None,
                              compression_type: Optional[str] = None) -> 'FileMetadataEntry':
        """Create a FileMetadataEntry from gathered metadata."""
        now_utc = datetime.now(timezone.utc)
        
        entry_data = {
            'filepath_relative': rel_path_str,
            'filename': file_metadata['filename'],
            'extension': file_metadata['extension'],
            'size_bytes': file_metadata['size_bytes'],
            'os_last_modified_utc': file_metadata['os_last_modified_utc'],
            'os_created_utc': file_metadata['os_created_utc'],
            'custom_hashes': file_metadata['custom_hashes'],
            'date_added_to_metadata_utc': now_utc,
            'last_metadata_update_utc': now_utc,
            'application_status': application_status,
            'user_metadata': user_metadata,
            'version_current': 1,
            'version_history_app': [],
            'original_filename_if_compressed': original_filename,
            'compression_type': compression_type
        }
        
        # Create initial version record
        initial_version = FileVersion(
            version_number=1,
            timestamp_utc=now_utc,
            change_description=change_description,
            size_bytes=file_metadata['size_bytes'],
            custom_hashes=[HashInfo(hash_type=ht, value=hv) for ht, hv in file_metadata['custom_hashes'].items()]
        )
        
        entry_data['version_history_app'] = [initial_version]
        
        return FileMetadataEntry(**entry_data)
    
    def _update_metadata_collection(self, metadata_entry: 'FileMetadataEntry') -> bool:
        """Update the metadata collection with new or updated entry."""
        if not self.metadata_handler:
            log_statement('warning', f"{self.log_prefix}:WARNING>>No metadata handler available", 
                         Path(__file__).stem)
            return False
        
        try:
            # Read current metadata
            current_metadata = self.metadata_handler.read_metadata()
            
            # Create or update collection
            if MODELS_AVAILABLE:
                collection = MetadataCollection.model_validate(current_metadata)
                collection.add_or_update_entry(metadata_entry)
                updated_data = collection.to_dict()
            else:
                # Fallback for when models aren't available
                updated_data = current_metadata.copy()
                updated_data[metadata_entry.filepath_relative] = metadata_entry.model_dump(exclude_none=True)
            
            # Write updated metadata
            return self.metadata_handler.write_metadata(updated_data)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to update metadata collection: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False
    
    def _update_version_with_commit_hash(self, rel_path_str: str, commit_hash: Optional[str]) -> None:
        """Update the latest version record with commit hash."""
        if not commit_hash or not self.metadata_handler:
            return
        
        try:
            current_metadata = self.metadata_handler.read_metadata()
            if rel_path_str in current_metadata:
                entry_data = current_metadata[rel_path_str]
                
                # Update latest version with commit hash
                if 'version_history_app' in entry_data and entry_data['version_history_app']:
                    entry_data['version_history_app'][-1]['git_commit_hash'] = commit_hash
                
                # Update git object hash
                entry_data['git_object_hash_current'] = self.git_ops.get_file_blob_hash(rel_path_str) if self.git_ops else None
                
                # Write back
                self.metadata_handler.write_metadata(current_metadata)
                
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update commit hash: {e}", 
                         Path(__file__).stem)
    
    def update_file_status(self, file_path: PathLike, new_status: str,
                          change_description: Optional[str] = None,
                          auto_commit: bool = True) -> OperationResult:
        """
        Update the application status of a tracked file.
        
        Args:
            file_path: Path to the file
            new_status: New application status
            change_description: Optional description of the change
            auto_commit: Whether to automatically commit the change
            
        Returns:
            OperationResult with success/failure status and details
        """
        with self._lock:
            def _do_update_status():
                # Resolve file path
                abs_path = validate_path(file_path)
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                    rel_path_str = str(rel_path).replace('\\', '/')
                except ValueError:
                    raise FileOperationError(f"File {abs_path} is not within repository")
                
                # Read current metadata
                if not self.metadata_handler:
                    raise MetadataError("No metadata handler available")
                
                current_metadata = self.metadata_handler.read_metadata()
                if rel_path_str not in current_metadata:
                    raise MetadataError(f"File {rel_path_str} not found in metadata")
                
                # Update status and create new version
                entry_data = current_metadata[rel_path_str]
                old_status = entry_data.get('application_status', 'unknown')
                
                if old_status == new_status:
                    log_statement('info', f"{self.log_prefix}:INFO>>Status already set to {new_status}", 
                                 Path(__file__).stem)
                    return {'changed': False, 'old_status': old_status, 'new_status': new_status}
                
                # Update entry
                now_utc = datetime.now(timezone.utc)
                entry_data['application_status'] = new_status
                entry_data['last_metadata_update_utc'] = now_utc.isoformat()
                entry_data['version_current'] = entry_data.get('version_current', 1) + 1
                
                # Create new version record
                description = change_description or f"Status changed from '{old_status}' to '{new_status}'"
                new_version = {
                    'version_number': entry_data['version_current'],
                    'timestamp_utc': now_utc.isoformat(),
                    'change_description': description,
                    'size_bytes': entry_data.get('size_bytes'),
                    'custom_hashes': entry_data.get('custom_hashes', {}),
                    'git_commit_hash': None
                }
                
                if 'version_history_app' not in entry_data:
                    entry_data['version_history_app'] = []
                entry_data['version_history_app'].append(new_version)
                
                # Write updated metadata
                success = self.metadata_handler.write_metadata(current_metadata)
                if not success:
                    raise MetadataError("Failed to write updated metadata")
                
                # Commit if requested
                commit_hash = None
                if auto_commit and self.git_ops and self.git_ops.is_valid_repo():
                    commit_success = self.git_ops.commit_changes(
                        message=f"Update status: {rel_path_str} -> {new_status}"
                    )
                    if commit_success:
                        commit_hash = self.git_ops.get_file_last_commit_hash(abs_path)
                        # Update version with commit hash
                        self._update_version_with_commit_hash(rel_path_str, commit_hash)
                
                result = {
                    'changed': True,
                    'old_status': old_status,
                    'new_status': new_status,
                    'version': entry_data['version_current'],
                    'commit_hash': commit_hash
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Updated status: {rel_path_str} -> {new_status}", 
                             Path(__file__).stem)
                return result
            
            return safe_operation("update_file_status", _do_update_status)
    
    def update_tracked_file(self, file_path: PathLike,
                          change_description: str = "File content updated",
                          auto_commit: bool = True) -> OperationResult:
        """
        Update a tracked file with new content/metadata.
        
        Args:
            file_path: Path to the file that was updated
            change_description: Description of the changes
            auto_commit: Whether to automatically commit changes
            
        Returns:
            OperationResult with success/failure status and details
        """
        with self._lock:
            def _do_update_file():
                # Validate file path
                abs_path = validate_path(file_path, must_exist=True, must_be_file=True)
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                    rel_path_str = str(rel_path).replace('\\', '/')
                except ValueError:
                    raise FileOperationError(f"File {abs_path} is not within repository")
                
                # Check if file is tracked
                if not self.metadata_handler:
                    raise MetadataError("No metadata handler available")
                
                current_metadata = self.metadata_handler.read_metadata()
                if rel_path_str not in current_metadata:
                    raise MetadataError(f"File {rel_path_str} not tracked. Use add_file_to_tracking first.")
                
                # Gather current file metadata
                new_file_metadata = self._gather_file_metadata(abs_path)
                entry_data = current_metadata[rel_path_str]
                
                # Check if file actually changed
                old_hashes = entry_data.get('custom_hashes', {})
                new_hashes = new_file_metadata['custom_hashes']
                
                content_changed = old_hashes != new_hashes
                size_changed = entry_data.get('size_bytes') != new_file_metadata['size_bytes']
                
                if not (content_changed or size_changed):
                    log_statement('info', f"{self.log_prefix}:INFO>>No significant changes detected for {rel_path_str}", 
                                 Path(__file__).stem)
                    return {'changed': False, 'file_path': rel_path_str}
                
                # Create previous version record
                old_version = {
                    'version_number': entry_data.get('version_current', 1),
                    'timestamp_utc': entry_data.get('last_metadata_update_utc'),
                    'change_description': f"Superseded by update: {change_description}",
                    'size_bytes': entry_data.get('size_bytes'),
                    'custom_hashes': old_hashes,
                    'git_commit_hash': entry_data.get('git_object_hash_current')
                }
                
                # Update entry with new metadata
                now_utc = datetime.now(timezone.utc)
                entry_data.update({
                    'size_bytes': new_file_metadata['size_bytes'],
                    'os_last_modified_utc': new_file_metadata['os_last_modified_utc'].isoformat(),
                    'custom_hashes': new_file_metadata['custom_hashes'],
                    'last_metadata_update_utc': now_utc.isoformat(),
                    'version_current': entry_data.get('version_current', 1) + 1
                })
                
                # Add version records
                if 'version_history_app' not in entry_data:
                    entry_data['version_history_app'] = []
                
                entry_data['version_history_app'].append(old_version)
                
                new_version = {
                    'version_number': entry_data['version_current'],
                    'timestamp_utc': now_utc.isoformat(),
                    'change_description': change_description,
                    'size_bytes': entry_data['size_bytes'],
                    'custom_hashes': entry_data['custom_hashes'],
                    'git_commit_hash': None
                }
                entry_data['version_history_app'].append(new_version)
                
                # Write updated metadata
                success = self.metadata_handler.write_metadata(current_metadata)
                if not success:
                    raise MetadataError("Failed to write updated metadata")
                
                # Commit changes
                commit_hash = None
                if auto_commit and self.git_ops and self.git_ops.is_valid_repo():
                    commit_success = self.git_ops.commit_changes(
                        files=[abs_path],
                        message=f"Update tracked file: {rel_path_str} - {change_description}"
                    )
                    if commit_success:
                        commit_hash = self.git_ops.get_file_last_commit_hash(abs_path)
                        self._update_version_with_commit_hash(rel_path_str, commit_hash)
                
                result = {
                    'changed': True,
                    'file_path': rel_path_str,
                    'version': entry_data['version_current'],
                    'content_changed': content_changed,
                    'size_changed': size_changed,
                    'commit_hash': commit_hash
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Updated tracked file: {rel_path_str}", 
                             Path(__file__).stem)
                return result
            
            return safe_operation("update_tracked_file", _do_update_file)
    
    def remove_file_from_tracking(self, file_path: PathLike,
                                 removal_action: str = "mark_deleted",
                                 change_description: Optional[str] = None,
                                 auto_commit: bool = True) -> OperationResult:
        """
        Remove a file from tracking or mark it as deleted.
        
        Args:
            file_path: Path to the file
            removal_action: Action to take ('mark_deleted', 'archive', 'delete_from_disk')
            change_description: Optional description of the removal
            auto_commit: Whether to automatically commit changes
            
        Returns:
            OperationResult with success/failure status and details
        """
        valid_actions = ['mark_deleted', 'archive', 'delete_from_disk']
        if removal_action not in valid_actions:
            removal_action = 'mark_deleted'
        
        with self._lock:
            def _do_remove_file():
                # Resolve file path
                abs_path = validate_path(file_path)
                try:
                    rel_path = abs_path.relative_to(self.repo_path)
                    rel_path_str = str(rel_path).replace('\\', '/')
                except ValueError:
                    raise FileOperationError(f"File {abs_path} is not within repository")
                
                # Check if file is tracked
                if not self.metadata_handler:
                    raise MetadataError("No metadata handler available")
                
                current_metadata = self.metadata_handler.read_metadata()
                if rel_path_str not in current_metadata:
                    raise MetadataError(f"File {rel_path_str} not tracked")
                
                entry_data = current_metadata[rel_path_str]
                old_status = entry_data.get('application_status', 'unknown')
                
                # Determine new status based on action
                if removal_action == 'archive':
                    new_status = STATUS_ARCHIVED
                else:
                    new_status = STATUS_DELETED
                
                # Update entry
                now_utc = datetime.now(timezone.utc)
                entry_data['application_status'] = new_status
                entry_data['last_metadata_update_utc'] = now_utc.isoformat()
                entry_data['version_current'] = entry_data.get('version_current', 1) + 1
                
                # Create version record
                description = change_description or f"File {removal_action}: {old_status} -> {new_status}"
                new_version = {
                    'version_number': entry_data['version_current'],
                    'timestamp_utc': now_utc.isoformat(),
                    'change_description': description,
                    'size_bytes': entry_data.get('size_bytes'),
                    'custom_hashes': entry_data.get('custom_hashes', {}),
                    'git_commit_hash': None
                }
                
                if 'version_history_app' not in entry_data:
                    entry_data['version_history_app'] = []
                entry_data['version_history_app'].append(new_version)
                
                # Handle file deletion from disk
                files_to_commit = []
                if removal_action == 'delete_from_disk' and abs_path.exists():
                    abs_path.unlink()
                    log_statement('info', f"{self.log_prefix}:INFO>>Deleted file from disk: {abs_path}", 
                                 Path(__file__).stem)
                else:
                    files_to_commit.append(abs_path)
                
                # Write updated metadata
                success = self.metadata_handler.write_metadata(current_metadata)
                if not success:
                    raise MetadataError("Failed to write updated metadata")
                
                # Commit changes
                commit_hash = None
                if auto_commit and self.git_ops and self.git_ops.is_valid_repo():
                    commit_message = f"Remove from tracking: {rel_path_str} ({removal_action})"
                    commit_success = self.git_ops.commit_changes(
                        files=files_to_commit,
                        message=commit_message
                    )
                    if commit_success:
                        commit_hash = self.git_ops.get_file_last_commit_hash(abs_path) if abs_path.exists() else "deleted"
                        self._update_version_with_commit_hash(rel_path_str, commit_hash)
                
                result = {
                    'file_path': rel_path_str,
                    'old_status': old_status,
                    'new_status': new_status,
                    'removal_action': removal_action,
                    'deleted_from_disk': removal_action == 'delete_from_disk',
                    'version': entry_data['version_current'],
                    'commit_hash': commit_hash
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Removed from tracking: {rel_path_str} ({removal_action})", 
                             Path(__file__).stem)
                return result
            
            return safe_operation("remove_file_from_tracking", _do_remove_file)
    
    def batch_add_files(self, file_paths: List[PathLike],
                       common_status: str = STATUS_NEW,
                       common_user_metadata: Optional[Dict[str, Any]] = None,
                       auto_commit: bool = True,
                       progress_callback: Optional[callable] = None) -> OperationResult:
        """
        Add multiple files to tracking in a batch operation.
        
        Args:
            file_paths: List of file paths to add
            common_status: Common application status for all files
            common_user_metadata: Common user metadata for all files
            auto_commit: Whether to automatically commit changes
            progress_callback: Optional callback for progress updates
            
        Returns:
            OperationResult with batch operation results
        """
        with self._lock:
            def _do_batch_add():
                results = {
                    'total_files': len(file_paths),
                    'successful': [],
                    'failed': [],
                    'errors': {}
                }
                
                # Save progress if handler available
                if self.progress_handler:
                    progress_id = f"batch_add_{int(time.time())}"
                    self.progress_handler.save_progress(progress_id, {
                        'total': len(file_paths),
                        'completed': 0,
                        'status': 'starting'
                    })
                
                for i, file_path in enumerate(file_paths):
                    try:
                        # Progress callback
                        if progress_callback:
                            progress_callback(i, len(file_paths), file_path)
                        
                        # Add individual file
                        result = self.add_file_to_tracking(
                            file_path=file_path,
                            application_status=common_status,
                            user_metadata=common_user_metadata,
                            auto_commit=False  # We'll commit all at once
                        )
                        
                        if result['status'] == OperationStatus.SUCCESS.value:
                            results['successful'].append(str(file_path))
                        else:
                            results['failed'].append(str(file_path))
                            results['errors'][str(file_path)] = result.get('error', 'Unknown error')
                        
                        # Update progress
                        if self.progress_handler:
                            self.progress_handler.save_progress(progress_id, {
                                'total': len(file_paths),
                                'completed': i + 1,
                                'status': 'processing',
                                'successful': len(results['successful']),
                                'failed': len(results['failed'])
                            })
                        
                    except Exception as e:
                        results['failed'].append(str(file_path))
                        results['errors'][str(file_path)] = str(e)
                        log_statement('error', f"{self.log_prefix}:ERROR>>Failed to add {file_path}: {e}", 
                                     Path(__file__).stem)
                
                # Commit all changes if requested
                commit_hash = None
                if auto_commit and results['successful'] and self.git_ops and self.git_ops.is_valid_repo():
                    commit_message = f"Batch add {len(results['successful'])} files"
                    commit_success = self.git_ops.commit_changes(message=commit_message)
                    if commit_success:
                        commit_hash = "batch_commit"
                
                # Final progress update
                if self.progress_handler:
                    self.progress_handler.save_progress(progress_id, {
                        'total': len(file_paths),
                        'completed': len(file_paths),
                        'status': 'completed',
                        'successful': len(results['successful']),
                        'failed': len(results['failed']),
                        'commit_hash': commit_hash
                    })
                
                results['commit_hash'] = commit_hash
                
                log_statement('info', 
                             f"{self.log_prefix}:INFO>>Batch add completed: {len(results['successful'])} successful, "
                             f"{len(results['failed'])} failed", Path(__file__).stem)
                return results
            
            return safe_operation("batch_add_files", _do_batch_add)
    
    def cleanup_repository(self, remove_orphaned_metadata: bool = False,
                          cleanup_old_versions: bool = False,
                          max_versions_per_file: int = 10) -> OperationResult:
        """
        Perform repository cleanup operations.
        
        Args:
            remove_orphaned_metadata: Remove metadata for files that no longer exist
            cleanup_old_versions: Remove old version records beyond max_versions_per_file
            max_versions_per_file: Maximum version records to keep per file
            
        Returns:
            OperationResult with cleanup statistics
        """
        with self._lock:
            def _do_cleanup():
                if not self.metadata_handler:
                    raise MetadataError("No metadata handler available")
                
                cleanup_stats = {
                    'orphaned_metadata_removed': 0,
                    'versions_cleaned': 0,
                    'files_processed': 0
                }
                
                current_metadata = self.metadata_handler.read_metadata()
                modified = False
                
                for rel_path_str, entry_data in list(current_metadata.items()):
                    cleanup_stats['files_processed'] += 1
                    file_path = self.repo_path / rel_path_str
                    
                    # Remove orphaned metadata
                    if remove_orphaned_metadata and not file_path.exists():
                        del current_metadata[rel_path_str]
                        cleanup_stats['orphaned_metadata_removed'] += 1
                        modified = True
                        continue
                    
                    # Cleanup old versions
                    if cleanup_old_versions and 'version_history_app' in entry_data:
                        versions = entry_data['version_history_app']
                        if len(versions) > max_versions_per_file:
                            # Keep the most recent versions
                            versions_to_keep = versions[-max_versions_per_file:]
                            entry_data['version_history_app'] = versions_to_keep
                            cleanup_stats['versions_cleaned'] += len(versions) - len(versions_to_keep)
                            modified = True
                
                # Write back if modified
                if modified:
                    success = self.metadata_handler.write_metadata(current_metadata)
                    if not success:
                        raise MetadataError("Failed to write cleaned metadata")
                
                log_statement('info', f"{self.log_prefix}:INFO>>Repository cleanup completed: {cleanup_stats}", 
                             Path(__file__).stem)
                return cleanup_stats
            
            return safe_operation("cleanup_repository", _do_cleanup)

# Section 6: RepoHandler Class - Part 1 (Initialization and Core)

class RepoHandler:
    """
    Main repository handler that coordinates all repository operations.
    
    This class serves as the primary interface for repository management,
    coordinating between Git operations, metadata management, file tracking,
    and analysis. It uses dependency injection for clean separation of concerns.
    """
    
    def __init__(self, config: Optional[RepoHandlerConfig] = None, **kwargs):
        """
        Initialize repository handler with clean dependency injection.
        
        Args:
            config: RepoHandlerConfig object with all settings
            **kwargs: Alternative way to pass config parameters
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        
        # Initialize configuration
        if config is None:
            # Create config from kwargs or defaults
            config_params = {
                'repo_path': kwargs.get('repo_path', Path.cwd()),
                'metadata_filename': kwargs.get('metadata_filename', METADATA_FILENAME),
                'create_if_missing': kwargs.get('create_if_missing', True),
                'use_git': kwargs.get('use_git', True),
                'use_compression': kwargs.get('use_compression', True),
                'max_workers': kwargs.get('max_workers', MAX_WORKERS),
                'cache_size': kwargs.get('cache_size', DF_CACHE_MAXSIZE)
            }
            config = RepoHandlerConfig(**config_params)
        
        self.config = config
        self._lock = RLock()
        self._initialized = False
        
        # Initialize core components
        self.git_ops: Optional[GitOpsHelper] = None
        self.metadata_handler: Optional[MetadataFileHandler] = None
        self.progress_handler: Optional[ProgressFileHandler] = None
        self.gitignore_handler: Optional[GitignoreHandler] = None
        self.analyzer: Optional[RepoAnalyzer] = None
        self.modifier: Optional[RepoModifier] = None
        
        # DataFrame management
        self.df: Optional[pd.DataFrame] = None
        self.df_cache: Optional[LRUCache] = None
        self.columns_schema = self._get_default_schema()
        self.expected_columns = self._get_expected_columns()
        
        # State tracking
        self.last_scan_time: Optional[datetime] = None
        self.is_dirty = False
        self.file_count = 0
        
        # Initialize the repository
        self._initialize_repository()
        
        log_statement('info', f"{self.log_prefix}:INFO>>RepoHandler initialized for {self.config.repo_path}", 
                     Path(__file__).stem)
    
    def _initialize_repository(self) -> None:
        """Initialize all repository components in proper order."""
        with self._lock:
            try:
                # Step 1: Validate and prepare paths
                self._prepare_repository_structure()
                
                # Step 2: Initialize Git operations (if enabled)
                if self.config.use_git:
                    self._initialize_git_operations()
                
                # Step 3: Initialize file handlers
                self._initialize_file_handlers()
                
                # Step 4: Initialize analysis and modification components
                self._initialize_operation_components()
                
                # Step 5: Initialize DataFrame support
                if PANDAS_AVAILABLE:
                    self._initialize_dataframe_support()
                
                # Step 6: Perform initial repository scan if empty
                self._perform_initial_setup()
                
                self._initialized = True
                log_statement('info', f"{self.log_prefix}:INFO>>Repository initialization completed successfully", 
                             Path(__file__).stem)
                
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Repository initialization failed: {e}", 
                             Path(__file__).stem, exc_info=True)
                self._initialized = False
                raise RepoHandlerError(f"Failed to initialize repository: {e}")
    
    def _prepare_repository_structure(self) -> None:
        """Prepare and validate repository directory structure."""
        # Ensure main repository directory exists
        self.config.repo_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories if needed
        metadata_dir = self.config.repo_path / '.tlato'
        metadata_dir.mkdir(exist_ok=True)
        
        progress_dir = metadata_dir / 'progress'
        progress_dir.mkdir(exist_ok=True)
        
        log_statement('debug', f"{self.log_prefix}:DEBUG>>Repository structure prepared", Path(__file__).stem)
    
    def _initialize_git_operations(self) -> None:
        """Initialize Git operations if available and requested."""
        if not GIT_AVAILABLE:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Git not available, skipping Git initialization", 
                         Path(__file__).stem)
            return
        
        try:
            self.git_ops = GitOpsHelper(
                repo_path=self.config.repo_path,
                create_if_missing=self.config.create_if_missing
            )
            
            if self.git_ops.is_valid_repo():
                log_statement('info', f"{self.log_prefix}:INFO>>Git operations initialized successfully", 
                             Path(__file__).stem)
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Git repository not valid, some features disabled", 
                             Path(__file__).stem)
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Git operations initialization failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            self.git_ops = None
    
    def _initialize_file_handlers(self) -> None:
        """Initialize file handlers for metadata and progress."""
        # Initialize metadata handler
        try:
            metadata_path = self.config.repo_path / '.tlato' / self.config.metadata_filename
            compression_type = 'zstd' if self.config.use_compression else None
            
            self.metadata_handler = MetadataFileHandler(
                metadata_path=metadata_path,
                use_compression=compression_type
            )
            
            # Ensure metadata file exists
            self.metadata_handler.ensure_metadata_file_exists()
            
            log_statement('info', f"{self.log_prefix}:INFO>>Metadata handler initialized", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Metadata handler initialization failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise RepoHandlerError(f"Failed to initialize metadata handler: {e}")
        
        # Initialize progress handler
        try:
            progress_dir = self.config.repo_path / '.tlato' / 'progress'
            self.progress_handler = ProgressFileHandler(
                progress_dir=progress_dir,
                use_compression=self.config.use_compression
            )
            
            log_statement('info', f"{self.log_prefix}:INFO>>Progress handler initialized", Path(__file__).stem)
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Progress handler initialization failed: {e}", 
                         Path(__file__).stem)
            self.progress_handler = None
    
    def _initialize_operation_components(self) -> None:
        """Initialize analysis and modification components."""
        # Initialize gitignore handler
        if self.git_ops:
            try:
                self.gitignore_handler = GitignoreHandler(repo_path=self.config.repo_path)
                self.gitignore_handler.ensure_gitignore_exists()
                log_statement('info', f"{self.log_prefix}:INFO>>Gitignore handler initialized", Path(__file__).stem)
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Gitignore handler initialization failed: {e}", 
                             Path(__file__).stem)
                self.gitignore_handler = None
        
        # Initialize analyzer
        try:
            self.analyzer = RepoAnalyzer(
                repo_path=self.config.repo_path,
                git_ops=self.git_ops,
                metadata_handler=self.metadata_handler
            )
            log_statement('info', f"{self.log_prefix}:INFO>>Repository analyzer initialized", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Analyzer initialization failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            self.analyzer = None
        
        # Initialize modifier
        try:
            self.modifier = RepoModifier(
                repo_path=self.config.repo_path,
                git_ops=self.git_ops,
                metadata_handler=self.metadata_handler,
                progress_handler=self.progress_handler
            )
            log_statement('info', f"{self.log_prefix}:INFO>>Repository modifier initialized", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Modifier initialization failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            self.modifier = None
    
    def _initialize_dataframe_support(self) -> None:
        """Initialize DataFrame support for repository data."""
        if not PANDAS_AVAILABLE:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Pandas not available, DataFrame support disabled", 
                         Path(__file__).stem)
            return
        
        try:
            # Initialize cache
            self.df_cache = LRUCache(maxsize=self.config.cache_size)
            
            # Load existing DataFrame or create empty one
            self.df = self._load_or_create_dataframe()
            
            log_statement('info', f"{self.log_prefix}:INFO>>DataFrame support initialized with {len(self.df)} entries", 
                         Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>DataFrame support initialization failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            self.df = None
            self.df_cache = None
    
    def _perform_initial_setup(self) -> None:
        """Perform initial setup tasks for new repositories."""
        # Check if this is a new repository (empty metadata)
        if self.metadata_handler:
            metadata = self.metadata_handler.read_metadata()
            if not metadata:
                log_statement('info', f"{self.log_prefix}:INFO>>New repository detected, performing initial setup", 
                             Path(__file__).stem)
                
                # Perform initial scan if analyzer is available
                if self.analyzer:
                    try:
                        files = self.analyzer.scan_directory_files(include_ignored=False)
                        if files:
                            log_statement('info', f"{self.log_prefix}:INFO>>Found {len(files)} files during initial scan", 
                                         Path(__file__).stem)
                            # Note: We don't automatically add files to tracking - user decision
                    except Exception as e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Initial scan failed: {e}", 
                                     Path(__file__).stem)
    
    def _get_default_schema(self) -> Dict[str, str]:
        """Get default DataFrame schema for file tracking."""
        return {
            'filepath': 'string',
            'filename': 'string',
            'extension': 'string',
            'size_bytes': 'Int64',
            'modified_time': 'datetime64[ns, UTC]',
            'created_time': 'datetime64[ns, UTC]',
            'content_hash': 'string',
            'status': 'string',
            'last_updated': 'datetime64[ns, UTC]',
            'metadata_json': 'string'
        }
    
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for DataFrame."""
        return list(self._get_default_schema().keys())
    
    def _load_or_create_dataframe(self) -> pd.DataFrame:
        """Load existing DataFrame from cache/storage or create new one."""
        if not PANDAS_AVAILABLE:
            return None
        
        # Try to load from cache first
        if self.df_cache:
            cached_df = self.df_cache.get('main_df')
            if cached_df is not None:
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Loaded DataFrame from cache", Path(__file__).stem)
                return cached_df.copy()
        
        # Create new empty DataFrame with proper schema
        df = pd.DataFrame(columns=self.expected_columns)
        
        # Apply schema types
        for col, dtype in self.columns_schema.items():
            if col in df.columns:
                try:
                    if 'datetime' in dtype:
                        df[col] = pd.to_datetime(df[col], utc=True)
                    elif dtype == 'Int64':
                        df[col] = df[col].astype('Int64')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to apply dtype {dtype} to {col}: {e}", 
                                 Path(__file__).stem)
        
        # Cache the new DataFrame
        if self.df_cache:
            self.df_cache.put('main_df', df.copy())
        
        return df
    
    def is_initialized(self) -> bool:
        """Check if repository is properly initialized."""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status."""
        status = {
            'initialized': self._initialized,
            'repo_path': str(self.config.repo_path),
            'git_available': self.git_ops is not None and self.git_ops.is_valid_repo(),
            'metadata_available': self.metadata_handler is not None,
            'dataframe_available': self.df is not None,
            'file_count': self.file_count,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'is_dirty': self.is_dirty,
            'components': {
                'git_ops': self.git_ops is not None,
                'metadata_handler': self.metadata_handler is not None,
                'progress_handler': self.progress_handler is not None,
                'gitignore_handler': self.gitignore_handler is not None,
                'analyzer': self.analyzer is not None,
                'modifier': self.modifier is not None
            }
        }
        
        # Add Git status if available
        if self.git_ops and self.git_ops.is_valid_repo():
            try:
                git_status = self.git_ops.get_status()
                status['git_status'] = git_status
            except Exception as e:
                status['git_status'] = {'error': str(e)}
        
        return status
    
    def refresh_dataframe_from_metadata(self) -> bool:
        """Refresh DataFrame content from current metadata."""
        if not PANDAS_AVAILABLE or not self.metadata_handler:
            return False
        
        def _do_refresh():
            # Read current metadata
            metadata = self.metadata_handler.read_metadata()
            
            if not metadata:
                # Create empty DataFrame
                self.df = self._load_or_create_dataframe()
                return True
            
            # Convert metadata to DataFrame rows
            rows = []
            for rel_path, entry_data in metadata.items():
                try:
                    # Extract relevant fields for DataFrame
                    row = {
                        'filepath': rel_path,
                        'filename': entry_data.get('filename', Path(rel_path).name),
                        'extension': entry_data.get('extension', ''),
                        'size_bytes': entry_data.get('size_bytes', 0),
                        'modified_time': entry_data.get('os_last_modified_utc'),
                        'created_time': entry_data.get('os_created_utc'),
                        'content_hash': self._extract_primary_hash(entry_data.get('custom_hashes', {})),
                        'status': entry_data.get('application_status', STATUS_NEW),
                        'last_updated': entry_data.get('last_metadata_update_utc'),
                        'metadata_json': json.dumps(entry_data, default=str)
                    }
                    rows.append(row)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to convert metadata entry {rel_path}: {e}", 
                                 Path(__file__).stem)
            
            # Create DataFrame from rows
            if rows:
                self.df = pd.DataFrame(rows)
                self.df = self.df.reindex(columns=self.expected_columns, fill_value=pd.NA)
                
                # Apply schema
                for col, dtype in self.columns_schema.items():
                    if col in self.df.columns:
                        try:
                            if 'datetime' in dtype:
                                self.df[col] = pd.to_datetime(self.df[col], utc=True, errors='coerce')
                            elif dtype == 'Int64':
                                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
                            else:
                                self.df[col] = self.df[col].astype(dtype, errors='ignore')
                        except Exception as e:
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Schema application failed for {col}: {e}", 
                                         Path(__file__).stem)
            else:
                self.df = self._load_or_create_dataframe()
            
            # Update cache and state
            if self.df_cache:
                self.df_cache.put('main_df', self.df.copy())
            
            self.file_count = len(self.df)
            self.last_scan_time = datetime.now(timezone.utc)
            
            log_statement('info', f"{self.log_prefix}:INFO>>DataFrame refreshed with {len(self.df)} entries", 
                         Path(__file__).stem)
            return True
        
        operation_result = safe_operation("refresh_dataframe_from_metadata", _do_refresh)
        return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def _extract_primary_hash(self, hashes_dict: Dict[str, str]) -> str:
        """Extract primary hash value from hashes dictionary."""
        if not hashes_dict:
            return ""
        
        # Prefer SHA256, then MD5, then any available
        for algorithm in ['sha256', 'md5', 'sha1']:
            if algorithm in hashes_dict:
                return hashes_dict[algorithm]
        
        # Return first available hash
        return next(iter(hashes_dict.values()), "")
    
    def scan_and_update_dataframe(self, include_ignored: bool = False) -> OperationResult:
        """Scan repository and update DataFrame with current file information."""
        def _do_scan_update():
            if not self.analyzer:
                raise RepoHandlerError("Repository analyzer not available")
            
            # Scan current files
            files = self.analyzer.scan_directory_files(include_ignored=include_ignored)
            
            if not files:
                log_statement('info', f"{self.log_prefix}:INFO>>No files found during scan", Path(__file__).stem)
                return {'files_found': 0, 'dataframe_updated': False}
            
            # Convert to DataFrame format
            df_rows = []
            for file_info in files:
                row = {
                    'filepath': file_info.get('relative_path', ''),
                    'filename': file_info.get('filename', ''),
                    'extension': file_info.get('extension', ''),
                    'size_bytes': file_info.get('size_bytes', 0),
                    'modified_time': file_info.get('modified_time'),
                    'created_time': file_info.get('created_time'),
                    'content_hash': '',  # Would need separate hash calculation
                    'status': STATUS_DISCOVERED,
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'metadata_json': json.dumps(file_info, default=str)
                }
                df_rows.append(row)
            
            # Update DataFrame
            if PANDAS_AVAILABLE and df_rows:
                scan_df = pd.DataFrame(df_rows)
                scan_df = scan_df.reindex(columns=self.expected_columns, fill_value=pd.NA)
                
                # Apply schema
                for col, dtype in self.columns_schema.items():
                    if col in scan_df.columns:
                        try:
                            if 'datetime' in dtype:
                                scan_df[col] = pd.to_datetime(scan_df[col], utc=True, errors='coerce')
                            elif dtype == 'Int64':
                                scan_df[col] = pd.to_numeric(scan_df[col], errors='coerce').astype('Int64')
                            else:
                                scan_df[col] = scan_df[col].astype(dtype, errors='ignore')
                        except Exception as e:
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Schema application failed for {col}: {e}", 
                                         Path(__file__).stem)
                
                # Replace current DataFrame
                self.df = scan_df
                
                # Update cache
                if self.df_cache:
                    self.df_cache.put('main_df', self.df.copy())
                
                self.file_count = len(self.df)
                self.last_scan_time = datetime.now(timezone.utc)
                self.is_dirty = True
                
                result = {
                    'files_found': len(files),
                    'dataframe_updated': True,
                    'total_entries': len(self.df)
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>DataFrame updated with {len(files)} files from scan", 
                             Path(__file__).stem)
                return result
            
            return {'files_found': len(files), 'dataframe_updated': False}
        
        return safe_operation("scan_and_update_dataframe", _do_scan_update)
    
    def get_dataframe(self, refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get the current DataFrame, optionally refreshing from metadata.
        
        Args:
            refresh: Whether to refresh DataFrame from current metadata
            
        Returns:
            DataFrame copy or None if not available
        """
        if not PANDAS_AVAILABLE:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Pandas not available", Path(__file__).stem)
            return None
        
        if refresh:
            self.refresh_dataframe_from_metadata()
        
        return self.df.copy() if self.df is not None else None
    
    def save_state(self, include_dataframe: bool = True) -> OperationResult:
        """
        Save current repository state to storage.
        
        Args:
            include_dataframe: Whether to save DataFrame state
            
        Returns:
            OperationResult with save status
        """
        def _do_save_state():
            saved_components = []
            
            # Save metadata (handled by individual operations through metadata_handler)
            if self.metadata_handler:
                # Metadata is automatically saved by modification operations
                saved_components.append('metadata')
            
            # Save DataFrame to cache
            if include_dataframe and self.df is not None and self.df_cache:
                self.df_cache.put('main_df', self.df.copy())
                saved_components.append('dataframe')
            
            # Update state flags
            self.is_dirty = False
            
            result = {
                'saved_components': saved_components,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            log_statement('info', f"{self.log_prefix}:INFO>>Repository state saved: {saved_components}", 
                         Path(__file__).stem)
            return result
        
        return safe_operation("save_state", _do_save_state)
    
    def __repr__(self) -> str:
        """String representation of RepoHandler."""
        status = "initialized" if self._initialized else "not initialized"
        file_count = len(self.df) if self.df is not None else "unknown"
        return f"RepoHandler(path={self.config.repo_path}, status={status}, files={file_count})"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            if self.is_dirty:
                self.save_state()
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Error during cleanup: {e}", Path(__file__).stem)
        
        # Clear caches
        if self.df_cache:
            self.df_cache.clear()

# Section 7: RepoHandler Class - Part 2 (Advanced Operations)

    # High-level file operations that delegate to components
    def add_file_to_tracking(self, file_path: PathLike, **kwargs) -> OperationResult:
        """
        Add a file to repository tracking with comprehensive metadata.
        
        This is a high-level method that coordinates between modifier and dataframe.
        
        Args:
            file_path: Path to file to track
            **kwargs: Additional parameters passed to modifier
            
        Returns:
            OperationResult with operation status and details
        """
        if not self._initialized:
            return safe_operation("add_file_to_tracking", 
                                lambda: self._raise_not_initialized())['result']
        
        if not self.modifier:
            return safe_operation("add_file_to_tracking", 
                                lambda: self._raise_component_unavailable("modifier"))['result']
        
        def _do_add_file():
            # Use modifier to add file
            result = self.modifier.add_file_to_tracking(file_path, **kwargs)
            
            if result['status'] == OperationStatus.SUCCESS.value:
                # Update DataFrame if available
                if self.df is not None:
                    self._add_file_to_dataframe(result['result'])
                
                self.is_dirty = True
                self.file_count += 1
            
            return result
        
        return safe_operation("add_file_to_tracking", _do_add_file)
    
    def update_file_status(self, file_path: PathLike, new_status: str, **kwargs) -> OperationResult:
        """Update file status with automatic DataFrame synchronization."""
        if not self._initialized or not self.modifier:
            return self._create_error_result("Repository not properly initialized")
        
        def _do_update_status():
            result = self.modifier.update_file_status(file_path, new_status, **kwargs)
            
            if result['status'] == OperationStatus.SUCCESS.value and self.df is not None:
                self._update_file_in_dataframe(file_path, {'status': new_status})
                self.is_dirty = True
            
            return result
        
        return safe_operation("update_file_status", _do_update_status)
    
    def remove_file_from_tracking(self, file_path: PathLike, **kwargs) -> OperationResult:
        """Remove file from tracking with DataFrame synchronization."""
        if not self._initialized or not self.modifier:
            return self._create_error_result("Repository not properly initialized")
        
        def _do_remove_file():
            result = self.modifier.remove_file_from_tracking(file_path, **kwargs)
            
            if result['status'] == OperationStatus.SUCCESS.value and self.df is not None:
                self._remove_file_from_dataframe(file_path)
                self.is_dirty = True
                self.file_count = max(0, self.file_count - 1)
            
            return result
        
        return safe_operation("remove_file_from_tracking", _do_remove_file)
    
    def batch_add_files(self, file_paths: List[PathLike], **kwargs) -> OperationResult:
        """Batch add files with progress tracking and DataFrame updates."""
        if not self._initialized or not self.modifier:
            return self._create_error_result("Repository not properly initialized")
        
        def _do_batch_add():
            # Set up progress tracking
            progress_id = f"batch_add_{int(time.time())}"
            
            def progress_callback(current: int, total: int, current_file: PathLike):
                if self.progress_handler:
                    self.progress_handler.save_progress(progress_id, {
                        'operation': 'batch_add_files',
                        'current': current,
                        'total': total,
                        'current_file': str(current_file),
                        'percent': (current / total) * 100 if total > 0 else 0
                    })
            
            # Execute batch add
            result = self.modifier.batch_add_files(
                file_paths, 
                progress_callback=progress_callback,
                **kwargs
            )
            
            if result['status'] == OperationStatus.SUCCESS.value:
                # Update DataFrame with successful additions
                batch_result = result['result']
                successful_files = batch_result.get('successful', [])
                
                if successful_files and self.df is not None:
                    # Refresh DataFrame from metadata to get new entries
                    self.refresh_dataframe_from_metadata()
                
                self.file_count += len(successful_files)
                self.is_dirty = True
                
                # Clean up progress
                if self.progress_handler:
                    self.progress_handler.delete_progress(progress_id)
            
            return result
        
        return safe_operation("batch_add_files", _do_batch_add)
    
    # Advanced analysis operations
    def get_repository_summary(self) -> Dict[str, Any]:
        """Get comprehensive repository summary combining all components."""
        if not self._initialized:
            return {'error': 'Repository not initialized'}
        
        def _do_get_summary():
            summary = {
                'repository_info': {
                    'path': str(self.config.repo_path),
                    'initialized': self._initialized,
                    'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
                    'file_count': self.file_count,
                    'is_dirty': self.is_dirty
                },
                'components': self.get_status()['components'],
                'git_summary': {},
                'metadata_summary': {},
                'file_analysis': {},
                'discrepancies': {}
            }
            
            # Git analysis
            if self.git_ops and self.git_ops.is_valid_repo():
                try:
                    git_status = self.git_ops.get_status()
                    commit_history = self.git_ops.get_commit_history(max_count=5)
                    
                    summary['git_summary'] = {
                        'is_valid': True,
                        'status': git_status,
                        'recent_commits': len(commit_history),
                        'latest_commit': commit_history[0] if commit_history else None
                    }
                except Exception as e:
                    summary['git_summary'] = {'error': str(e)}
            
            # Metadata analysis
            if self.metadata_handler:
                try:
                    metadata = self.metadata_handler.read_metadata()
                    file_stats = self.metadata_handler.get_file_stats()
                    
                    summary['metadata_summary'] = {
                        'entries_count': len(metadata),
                        'file_stats': file_stats
                    }
                    
                    # Analyze metadata entries
                    if metadata:
                        statuses = {}
                        total_size = 0
                        extensions = {}
                        
                        for entry_data in metadata.values():
                            status = entry_data.get('application_status', 'unknown')
                            statuses[status] = statuses.get(status, 0) + 1
                            
                            size = entry_data.get('size_bytes', 0)
                            if isinstance(size, (int, float)):
                                total_size += size
                            
                            ext = entry_data.get('extension', 'no_extension')
                            extensions[ext] = extensions.get(ext, 0) + 1
                        
                        summary['metadata_summary'].update({
                            'status_distribution': statuses,
                            'total_size_bytes': total_size,
                            'extension_distribution': extensions
                        })
                
                except Exception as e:
                    summary['metadata_summary'] = {'error': str(e)}
            
            # File system analysis
            if self.analyzer:
                try:
                    files = self.analyzer.scan_directory_files()
                    summary['file_analysis'] = {
                        'filesystem_file_count': len(files),
                        'total_size_bytes': sum(f.get('size_bytes', 0) for f in files)
                    }
                    
                    # Extension analysis
                    fs_extensions = {}
                    for file_info in files:
                        ext = file_info.get('extension', 'no_extension')
                        fs_extensions[ext] = fs_extensions.get(ext, 0) + 1
                    
                    summary['file_analysis']['filesystem_extensions'] = fs_extensions
                    
                except Exception as e:
                    summary['file_analysis'] = {'error': str(e)}
            
            # Discrepancy analysis
            if self.analyzer:
                try:
                    discrepancies = self.analyzer.detect_repository_discrepancies()
                    summary['discrepancies'] = discrepancies
                except Exception as e:
                    summary['discrepancies'] = {'error': str(e)}
            
            return summary
        
        operation_result = safe_operation("get_repository_summary", _do_get_summary)
        return operation_result.get('result', {'error': operation_result.get('error')})
    
    def scan_for_new_files(self, auto_add: bool = False, **add_kwargs) -> OperationResult:
        """
        Scan for new files and optionally add them to tracking.
        
        Args:
            auto_add: Whether to automatically add discovered files to tracking
            **add_kwargs: Additional arguments for file addition
            
        Returns:
            OperationResult with scan results
        """
        if not self._initialized or not self.analyzer:
            return self._create_error_result("Repository analyzer not available")
        
        def _do_scan_new_files():
            # Get current tracked files
            tracked_files = set()
            if self.metadata_handler:
                metadata = self.metadata_handler.read_metadata()
                tracked_files = set(metadata.keys())
            
            # Scan filesystem
            all_files = self.analyzer.scan_directory_files(include_ignored=False)
            
            # Find new files
            new_files = []
            for file_info in all_files:
                rel_path = file_info.get('relative_path', '')
                if rel_path not in tracked_files:
                    new_files.append(file_info)
            
            result = {
                'total_files_found': len(all_files),
                'tracked_files_count': len(tracked_files),
                'new_files_found': len(new_files),
                'new_files': [f['relative_path'] for f in new_files]
            }
            
            # Auto-add if requested
            if auto_add and new_files and self.modifier:
                file_paths = [self.config.repo_path / f['relative_path'] for f in new_files]
                batch_result = self.batch_add_files(file_paths, **add_kwargs)
                result['auto_add_result'] = batch_result
            
            log_statement('info', f"{self.log_prefix}:INFO>>Scan found {len(new_files)} new files", 
                         Path(__file__).stem)
            return result
        
        return safe_operation("scan_for_new_files", _do_scan_new_files)
    
    def verify_repository_integrity(self, check_file_hashes: bool = True) -> OperationResult:
        """
        Perform comprehensive repository integrity verification.
        
        Args:
            check_file_hashes: Whether to verify file content hashes
            
        Returns:
            OperationResult with integrity check results
        """
        if not self._initialized:
            return self._create_error_result("Repository not initialized")
        
        def _do_verify_integrity():
            integrity_report = {
                'overall_status': 'passed',
                'checks_performed': [],
                'issues_found': [],
                'file_checks': {},
                'summary': {}
            }
            
            # Check 1: Metadata file integrity
            if self.metadata_handler:
                try:
                    metadata = self.metadata_handler.read_metadata()
                    integrity_report['checks_performed'].append('metadata_file_readable')
                    
                    if not metadata:
                        integrity_report['issues_found'].append('metadata_file_empty')
                        integrity_report['overall_status'] = 'warning'
                    
                except Exception as e:
                    integrity_report['issues_found'].append(f'metadata_read_error: {e}')
                    integrity_report['overall_status'] = 'failed'
            
            # Check 2: Git repository integrity
            if self.git_ops and self.git_ops.is_valid_repo():
                try:
                    git_status = self.git_ops.get_status()
                    integrity_report['checks_performed'].append('git_status_check')
                    
                    # Check for uncommitted changes
                    if any(git_status.values()):
                        integrity_report['issues_found'].append('uncommitted_changes_detected')
                        if integrity_report['overall_status'] == 'passed':
                            integrity_report['overall_status'] = 'warning'
                
                except Exception as e:
                    integrity_report['issues_found'].append(f'git_check_error: {e}')
                    integrity_report['overall_status'] = 'failed'
            
            # Check 3: File existence and hash verification
            if check_file_hashes and self.metadata_handler and self.analyzer:
                try:
                    metadata = self.metadata_handler.read_metadata()
                    integrity_report['checks_performed'].append('file_hash_verification')
                    
                    hash_failures = 0
                    missing_files = 0
                    
                    for rel_path, entry_data in metadata.items():
                        file_path = self.config.repo_path / rel_path
                        
                        # Check file existence
                        if not file_path.exists():
                            missing_files += 1
                            integrity_report['file_checks'][rel_path] = 'missing'
                            continue
                        
                        # Verify hash if available
                        custom_hashes = entry_data.get('custom_hashes', {})
                        if custom_hashes and HASHING_AVAILABLE:
                            for algorithm, expected_hash in custom_hashes.items():
                                try:
                                    current_hash = generate_data_hash(file_path, algorithm)
                                    if current_hash != expected_hash:
                                        hash_failures += 1
                                        integrity_report['file_checks'][rel_path] = f'hash_mismatch_{algorithm}'
                                        break
                                    else:
                                        integrity_report['file_checks'][rel_path] = 'verified'
                                except Exception:
                                    integrity_report['file_checks'][rel_path] = f'hash_error_{algorithm}'
                        else:
                            integrity_report['file_checks'][rel_path] = 'no_hash_available'
                    
                    if missing_files > 0:
                        integrity_report['issues_found'].append(f'{missing_files}_files_missing')
                        integrity_report['overall_status'] = 'failed'
                    
                    if hash_failures > 0:
                        integrity_report['issues_found'].append(f'{hash_failures}_hash_mismatches')
                        if integrity_report['overall_status'] == 'passed':
                            integrity_report['overall_status'] = 'failed'
                
                except Exception as e:
                    integrity_report['issues_found'].append(f'file_verification_error: {e}')
                    integrity_report['overall_status'] = 'failed'
            
            # Generate summary
            integrity_report['summary'] = {
                'total_checks': len(integrity_report['checks_performed']),
                'total_issues': len(integrity_report['issues_found']),
                'files_checked': len(integrity_report['file_checks']),
                'verified_files': len([status for status in integrity_report['file_checks'].values() 
                                     if status == 'verified'])
            }
            
            log_statement('info', f"{self.log_prefix}:INFO>>Integrity check completed: {integrity_report['overall_status']}", 
                         Path(__file__).stem)
            return integrity_report
        
        return safe_operation("verify_repository_integrity", _do_verify_integrity)
    
    def cleanup_repository(self, **cleanup_kwargs) -> OperationResult:
        """Perform repository cleanup operations."""
        if not self._initialized or not self.modifier:
            return self._create_error_result("Repository modifier not available")
        
        def _do_cleanup():
            result = self.modifier.cleanup_repository(**cleanup_kwargs)
            
            if result['status'] == OperationStatus.SUCCESS.value:
                # Refresh DataFrame after cleanup
                if self.df is not None:
                    self.refresh_dataframe_from_metadata()
                
                # Update file count
                self.file_count = len(self.df) if self.df is not None else 0
                self.is_dirty = True
            
            return result
        
        return safe_operation("cleanup_repository", _do_cleanup)
    
    def find_duplicate_files(self) -> OperationResult:
        """Find duplicate files in the repository."""
        if not self._initialized or not self.analyzer:
            return self._create_error_result("Repository analyzer not available")
        
        def _do_find_duplicates():
            duplicates = self.analyzer.find_duplicate_files()
            
            result = {
                'duplicate_groups': len(duplicates),
                'total_duplicate_files': sum(len(files) for files in duplicates.values()),
                'duplicates': duplicates
            }
            
            return result
        
        return safe_operation("find_duplicate_files", _do_find_duplicates)
    
    # DataFrame helper methods
    def _add_file_to_dataframe(self, file_result: Dict[str, Any]) -> None:
        """Add a file entry to the DataFrame."""
        if self.df is None or not PANDAS_AVAILABLE:
            return
        
        try:
            file_path = file_result.get('file_path', '')
            metadata_entry = file_result.get('metadata_entry', {})
            
            new_row = {
                'filepath': file_path,
                'filename': metadata_entry.get('filename', ''),
                'extension': metadata_entry.get('extension', ''),
                'size_bytes': metadata_entry.get('size_bytes', 0),
                'modified_time': metadata_entry.get('os_last_modified_utc'),
                'created_time': metadata_entry.get('os_created_utc'),
                'content_hash': self._extract_primary_hash(metadata_entry.get('custom_hashes', {})),
                'status': metadata_entry.get('application_status', STATUS_NEW),
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'metadata_json': json.dumps(metadata_entry, default=str)
            }
            
            # Add to DataFrame
            new_df_row = pd.DataFrame([new_row])
            self.df = pd.concat([self.df, new_df_row], ignore_index=True)
            
            # Update cache
            if self.df_cache:
                self.df_cache.put('main_df', self.df.copy())
        
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to add file to DataFrame: {e}", 
                         Path(__file__).stem)
    
    def _update_file_in_dataframe(self, file_path: PathLike, updates: Dict[str, Any]) -> None:
        """Update a file entry in the DataFrame."""
        if self.df is None or not PANDAS_AVAILABLE:
            return
        
        try:
            # Convert to relative path
            abs_path = validate_path(file_path)
            rel_path = abs_path.relative_to(self.config.repo_path) if abs_path.is_absolute() else abs_path
            rel_path_str = str(rel_path).replace('\\', '/')
            
            # Find and update row
            mask = self.df['filepath'] == rel_path_str
            if mask.any():
                for column, value in updates.items():
                    if column in self.df.columns:
                        self.df.loc[mask, column] = value
                
                # Update last_updated timestamp
                self.df.loc[mask, 'last_updated'] = datetime.now(timezone.utc).isoformat()
                
                # Update cache
                if self.df_cache:
                    self.df_cache.put('main_df', self.df.copy())
        
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update file in DataFrame: {e}", 
                         Path(__file__).stem)
    
    def _remove_file_from_dataframe(self, file_path: PathLike) -> None:
        """Remove a file entry from the DataFrame."""
        if self.df is None or not PANDAS_AVAILABLE:
            return
        
        try:
            # Convert to relative path
            abs_path = validate_path(file_path)
            rel_path = abs_path.relative_to(self.config.repo_path) if abs_path.is_absolute() else abs_path
            rel_path_str = str(rel_path).replace('\\', '/')
            
            # Remove row
            self.df = self.df[self.df['filepath'] != rel_path_str]
            
            # Update cache
            if self.df_cache:
                self.df_cache.put('main_df', self.df.copy())
        
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to remove file from DataFrame: {e}", 
                         Path(__file__).stem)
    
    # Utility methods
    def _raise_not_initialized(self) -> None:
        """Raise exception for uninitialized repository."""
        raise RepoHandlerError("Repository handler not properly initialized")
    
    def _raise_component_unavailable(self, component_name: str) -> None:
        """Raise exception for unavailable component."""
        raise RepoHandlerError(f"Repository component '{component_name}' not available")
    
    def _create_error_result(self, error_message: str) -> OperationResult:
        """Create a standardized error result."""
        return {
            'status': OperationStatus.FAILURE.value,
            'error': error_message,
            'result': None,
            'duration': 0.0
        }
    
    # Public API convenience methods
    def add_file(self, file_path: PathLike, **kwargs) -> bool:
        """
        Convenience method to add a file and return simple success/failure.
        
        Returns:
            True if successful, False otherwise
        """
        result = self.add_file_to_tracking(file_path, **kwargs)
        return result['status'] == OperationStatus.SUCCESS.value
    
    def get_file_status(self, file_path: PathLike) -> Optional[str]:
        """Get the current status of a tracked file."""
        if not self.metadata_handler:
            return None
        
        try:
            abs_path = validate_path(file_path)
            rel_path = abs_path.relative_to(self.config.repo_path) if abs_path.is_absolute() else abs_path
            rel_path_str = str(rel_path).replace('\\', '/')
            
            metadata = self.metadata_handler.read_metadata()
            entry_data = metadata.get(rel_path_str, {})
            return entry_data.get('application_status')
        
        except Exception:
            return None
    
    def get_tracked_files(self, status_filter: Optional[str] = None) -> List[str]:
        """
        Get list of tracked files, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of relative file paths
        """
        if not self.metadata_handler:
            return []
        
        try:
            metadata = self.metadata_handler.read_metadata()
            files = []
            
            for rel_path, entry_data in metadata.items():
                if status_filter is None or entry_data.get('application_status') == status_filter:
                    files.append(rel_path)
            
            return sorted(files)
        
        except Exception:
            return []
    
    def get_repository_statistics(self) -> Dict[str, Any]:
        """Get basic repository statistics."""
        stats = {
            'total_tracked_files': 0,
            'total_size_bytes': 0,
            'status_counts': {},
            'extension_counts': {},
            'git_status': 'unknown',
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None
        }
        
        if self.metadata_handler:
            try:
                metadata = self.metadata_handler.read_metadata()
                stats['total_tracked_files'] = len(metadata)
                
                for entry_data in metadata.values():
                    # Count by status
                    status = entry_data.get('application_status', 'unknown')
                    stats['status_counts'][status] = stats['status_counts'].get(status, 0) + 1
                    
                    # Count by extension
                    ext = entry_data.get('extension', 'no_extension')
                    stats['extension_counts'][ext] = stats['extension_counts'].get(ext, 0) + 1
                    
                    # Sum sizes
                    size = entry_data.get('size_bytes', 0)
                    if isinstance(size, (int, float)):
                        stats['total_size_bytes'] += size
            
            except Exception:
                pass
        
        if self.git_ops and self.git_ops.is_valid_repo():
            stats['git_status'] = 'valid'
        elif self.git_ops:
            stats['git_status'] = 'invalid'
        else:
            stats['git_status'] = 'unavailable'
        
        return stats

# Section 8: RepoManager and Main Execution - Completing the Refactoring

# Add this section to the end of repo_handlerv2.py to complete the refactoring

# Section 8: RepoManager and Main Execution (500 lines)

class RepositoryIndex:
    """
    Manages an index of multiple repositories with their metadata.
    
    This class handles the global repository index that tracks multiple
    repository instances, their locations, and summary information.
    """
    
    def __init__(self, index_path: Path, use_compression: bool = False):
        """
        Initialize repository index manager.
        
        Args:
            index_path: Path to the index file
            use_compression: Whether to use compression for the index file
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.index_path = validate_path(index_path)
        self.use_compression = use_compression and bool(CompressionHandler.get_available_compression_types())
        self._lock = RLock()
        
        # Adjust path for compression
        if self.use_compression:
            if not self.index_path.name.endswith('.gz'):
                self.index_path = self.index_path.with_suffix(f'{self.index_path.suffix}.gz')
        
        # Ensure index file exists
        self._ensure_index_exists()
        
        log_statement('info', f"{self.log_prefix}:INFO>>RepositoryIndex initialized: {self.index_path}", 
                     Path(__file__).stem)
    
    def _ensure_index_exists(self) -> bool:
        """Ensure index file exists with proper structure."""
        if self.index_path.exists():
            return True
        
        with self._lock:
            def _create_index():
                # Create default index structure
                default_index = {
                    'version': '1.0',
                    'created_time': datetime.now(timezone.utc).isoformat(),
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                    'repositories': {}
                }
                
                return self.write_index(default_index)
            
            operation_result = safe_operation("create_index", _create_index)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def read_index(self) -> Dict[str, Any]:
        """Read the repository index from file."""
        with self._lock:
            if not self.index_path.exists():
                log_statement('warning', f"{self.log_prefix}:WARNING>>Index file not found: {self.index_path}", 
                             Path(__file__).stem)
                return self._get_default_index()
            
            def _do_read_index():
                # Read file content
                with open(self.index_path, 'rb') as f:
                    content = f.read()
                
                # Decompress if needed
                if self.use_compression:
                    try:
                        content = CompressionHandler.decompress_content(content, 'gzip')
                    except Exception as e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Decompression failed: {e}", 
                                     Path(__file__).stem)
                        # Try reading as uncompressed
                        with open(self.index_path, 'rb') as f:
                            content = f.read()
                
                # Parse JSON
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                if not content.strip():
                    return self._get_default_index()
                
                try:
                    index_data = json.loads(content)
                    # Validate index structure
                    if not isinstance(index_data, dict) or 'repositories' not in index_data:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Invalid index structure, using default", 
                                     Path(__file__).stem)
                        return self._get_default_index()
                    
                    return index_data
                except json.JSONDecodeError as e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>JSON decode error in index file: {e}", 
                                 Path(__file__).stem)
                    return self._get_default_index()
            
            operation_result = safe_operation("read_index", _do_read_index)
            return operation_result.get('result', self._get_default_index())
    
    def write_index(self, index_data: Dict[str, Any]) -> bool:
        """Write the repository index to file."""
        with self._lock:
            def _do_write_index():
                # Update metadata
                index_data['last_updated'] = datetime.now(timezone.utc).isoformat()
                index_data['version'] = index_data.get('version', '1.0')
                
                # Ensure parent directory exists
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Serialize to JSON
                json_content = json.dumps(index_data, indent=2, sort_keys=True, default=str)
                
                # Compress if needed
                if self.use_compression:
                    content = CompressionHandler.compress_content(json_content, 'gzip')
                    mode = 'wb'
                else:
                    content = json_content
                    mode = 'w'
                
                # Write to temporary file first
                temp_path = self.index_path.with_suffix(f'{self.index_path.suffix}.tmp')
                
                try:
                    with open(temp_path, mode, encoding='utf-8' if mode == 'w' else None) as f:
                        f.write(content)
                    
                    # Atomic move
                    shutil.move(str(temp_path), str(self.index_path))
                    
                    log_statement('info', f"{self.log_prefix}:INFO>>Index written with {len(index_data.get('repositories', {}))} repositories", 
                                 Path(__file__).stem)
                    return True
                    
                except Exception as e:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    raise e
            
            operation_result = safe_operation("write_index", _do_write_index)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def _get_default_index(self) -> Dict[str, Any]:
        """Get default index structure."""
        return {
            'version': '1.0',
            'created_time': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'repositories': {}
        }
    
    def add_repository(self, repo_id: str, repo_info: Dict[str, Any]) -> bool:
        """Add or update a repository in the index."""
        if not repo_id or not isinstance(repo_info, dict):
            log_statement('error', f"{self.log_prefix}:ERROR>>Invalid repository info", Path(__file__).stem)
            return False
        
        try:
            index_data = self.read_index()
            
            # Add repository info with metadata
            repo_entry = {
                'added_time': datetime.now(timezone.utc).isoformat(),
                'last_updated': datetime.now(timezone.utc).isoformat(),
                **repo_info
            }
            
            # Update existing entry
            if repo_id in index_data['repositories']:
                existing_entry = index_data['repositories'][repo_id]
                repo_entry['added_time'] = existing_entry.get('added_time', repo_entry['added_time'])
            
            index_data['repositories'][repo_id] = repo_entry
            
            success = self.write_index(index_data)
            if success:
                log_statement('info', f"{self.log_prefix}:INFO>>Repository added to index: {repo_id}", 
                             Path(__file__).stem)
            
            return success
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to add repository to index: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False
    
    def remove_repository(self, repo_id: str) -> bool:
        """Remove a repository from the index."""
        try:
            index_data = self.read_index()
            
            if repo_id in index_data['repositories']:
                del index_data['repositories'][repo_id]
                success = self.write_index(index_data)
                if success:
                    log_statement('info', f"{self.log_prefix}:INFO>>Repository removed from index: {repo_id}", 
                                 Path(__file__).stem)
                return success
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Repository not found in index: {repo_id}", 
                             Path(__file__).stem)
                return True  # Not an error if it wasn't there
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to remove repository from index: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False
    
    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Get repository information from the index."""
        try:
            index_data = self.read_index()
            return index_data['repositories'].get(repo_id)
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get repository from index: {e}", 
                         Path(__file__).stem)
            return None
    
    def list_repositories(self) -> List[str]:
        """Get list of all repository IDs."""
        try:
            index_data = self.read_index()
            return list(index_data['repositories'].keys())
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to list repositories: {e}", 
                         Path(__file__).stem)
            return []
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about the repository index."""
        try:
            index_data = self.read_index()
            repositories = index_data.get('repositories', {})
            
            stats = {
                'total_repositories': len(repositories),
                'index_version': index_data.get('version', 'unknown'),
                'created_time': index_data.get('created_time'),
                'last_updated': index_data.get('last_updated'),
                'file_size_bytes': self.index_path.stat().st_size if self.index_path.exists() else 0
            }
            
            return stats
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get index statistics: {e}", 
                         Path(__file__).stem)
            return {'error': str(e)}


class RepoManager:
    """
    Manages multiple repository instances with centralized index and operations.
    
    This class provides a high-level interface for managing multiple repositories,
    including creation, loading, and coordination between repositories.
    """
    
    def __init__(self, base_directory: Path, index_filename: str = "repository_index.json",
                 use_compression: bool = True, auto_create_directories: bool = True):
        """
        Initialize repository manager.
        
        Args:
            base_directory: Base directory for all repository management files
            index_filename: Name of the repository index file
            use_compression: Whether to use compression for index and metadata
            auto_create_directories: Whether to automatically create needed directories
        """
        self.log_prefix = get_log_prefix(inspect.currentframe())
        self.base_directory = validate_path(base_directory)
        self.use_compression = use_compression
        self.auto_create_directories = auto_create_directories
        
        # Set up paths
        self.index_path = self.base_directory / index_filename
        self.repositories_directory = self.base_directory / "repositories"
        
        # Create directories if needed
        if auto_create_directories:
            self.base_directory.mkdir(parents=True, exist_ok=True)
            self.repositories_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.repository_index = RepositoryIndex(self.index_path, use_compression)
        self.active_repositories: Dict[str, RepoHandler] = {}
        self._lock = RLock()
        
        log_statement('info', f"{self.log_prefix}:INFO>>RepoManager initialized: {self.base_directory}", 
                     Path(__file__).stem)
    
    def create_repository(self, repo_id: str, repo_path: PathLike, 
                         config_overrides: Optional[Dict[str, Any]] = None) -> OperationResult:
        """
        Create a new repository and add it to management.
        
        Args:
            repo_id: Unique identifier for the repository
            repo_path: Path where the repository should be created
            config_overrides: Optional configuration overrides
            
        Returns:
            OperationResult with creation status and repository instance
        """
        with self._lock:
            def _do_create_repository():
                if repo_id in self.active_repositories:
                    raise RepoHandlerError(f"Repository {repo_id} already active")
                
                # Check if repository already exists in index
                existing_repo = self.repository_index.get_repository(repo_id)
                if existing_repo:
                    raise RepoHandlerError(f"Repository {repo_id} already exists in index")
                
                # Prepare configuration
                repo_path_resolved = validate_path(repo_path)
                config_params = {
                    'repo_path': repo_path_resolved,
                    'create_if_missing': True,
                    'use_compression': self.use_compression,
                    'metadata_filename': 'metadata.json'
                }
                
                if config_overrides:
                    config_params.update(config_overrides)
                
                config = RepoHandlerConfig(**config_params)
                
                # Create repository instance
                repo_handler = RepoHandler(config=config)
                
                if not repo_handler.is_initialized():
                    raise RepoHandlerError("Failed to initialize repository")
                
                # Add to index
                repo_info = {
                    'repo_path': str(repo_path_resolved),
                    'repo_id': repo_id,
                    'config': config_params,
                    'status': 'active',
                    'created_by': 'RepoManager',
                    **repo_handler.get_repository_statistics()
                }
                
                success = self.repository_index.add_repository(repo_id, repo_info)
                if not success:
                    raise RepoHandlerError("Failed to add repository to index")
                
                # Add to active repositories
                self.active_repositories[repo_id] = repo_handler
                
                result = {
                    'repo_id': repo_id,
                    'repo_path': str(repo_path_resolved),
                    'repository': repo_handler,
                    'created': True
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Created repository: {repo_id} at {repo_path_resolved}", 
                             Path(__file__).stem)
                return result
            
            return safe_operation("create_repository", _do_create_repository)
    
    def load_repository(self, repo_id: str, 
                       config_overrides: Optional[Dict[str, Any]] = None) -> OperationResult:
        """
        Load an existing repository from the index.
        
        Args:
            repo_id: Repository identifier
            config_overrides: Optional configuration overrides
            
        Returns:
            OperationResult with loaded repository instance
        """
        with self._lock:
            def _do_load_repository():
                if repo_id in self.active_repositories:
                    return {
                        'repo_id': repo_id,
                        'repository': self.active_repositories[repo_id],
                        'loaded': False,
                        'already_active': True
                    }
                
                # Get repository info from index
                repo_info = self.repository_index.get_repository(repo_id)
                if not repo_info:
                    raise RepoHandlerError(f"Repository {repo_id} not found in index")
                
                # Prepare configuration
                repo_path = Path(repo_info['repo_path'])
                config_params = repo_info.get('config', {})
                config_params.update({
                    'repo_path': repo_path,
                    'create_if_missing': False  # Don't create when loading existing
                })
                
                if config_overrides:
                    config_params.update(config_overrides)
                
                config = RepoHandlerConfig(**config_params)
                
                # Load repository instance
                repo_handler = RepoHandler(config=config)
                
                if not repo_handler.is_initialized():
                    raise RepoHandlerError(f"Failed to initialize repository {repo_id}")
                
                # Add to active repositories
                self.active_repositories[repo_id] = repo_handler
                
                # Update index with current statistics
                current_stats = repo_handler.get_repository_statistics()
                repo_info.update(current_stats)
                repo_info['status'] = 'active'
                self.repository_index.add_repository(repo_id, repo_info)
                
                result = {
                    'repo_id': repo_id,
                    'repository': repo_handler,
                    'loaded': True,
                    'already_active': False
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Loaded repository: {repo_id}", Path(__file__).stem)
                return result
            
            return safe_operation("load_repository", _do_load_repository)
    
    def unload_repository(self, repo_id: str, save_state: bool = True) -> bool:
        """
        Unload a repository from active management.
        
        Args:
            repo_id: Repository identifier
            save_state: Whether to save repository state before unloading
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            def _do_unload_repository():
                if repo_id not in self.active_repositories:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Repository {repo_id} not active", 
                                 Path(__file__).stem)
                    return True
                
                repo_handler = self.active_repositories[repo_id]
                
                # Save state if requested
                if save_state:
                    try:
                        save_result = repo_handler.save_state()
                        if save_result['status'] != OperationStatus.SUCCESS.value:
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to save state for {repo_id}", 
                                         Path(__file__).stem)
                    except Exception as e:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Error saving state for {repo_id}: {e}", 
                                     Path(__file__).stem)
                
                # Update index status
                repo_info = self.repository_index.get_repository(repo_id)
                if repo_info:
                    repo_info['status'] = 'inactive'
                    repo_info.update(repo_handler.get_repository_statistics())
                    self.repository_index.add_repository(repo_id, repo_info)
                
                # Remove from active repositories
                del self.active_repositories[repo_id]
                
                log_statement('info', f"{self.log_prefix}:INFO>>Unloaded repository: {repo_id}", Path(__file__).stem)
                return True
            
            operation_result = safe_operation("unload_repository", _do_unload_repository)
            return operation_result['status'] == OperationStatus.SUCCESS.value
    
    def remove_repository(self, repo_id: str, delete_files: bool = False) -> OperationResult:
        """
        Remove a repository from management and optionally delete files.
        
        Args:
            repo_id: Repository identifier
            delete_files: Whether to delete repository files from disk
            
        Returns:
            OperationResult with removal status
        """
        with self._lock:
            def _do_remove_repository():
                # Unload if active
                if repo_id in self.active_repositories:
                    self.unload_repository(repo_id, save_state=False)
                
                # Get repository info for file deletion
                repo_info = self.repository_index.get_repository(repo_id)
                
                # Delete files if requested
                if delete_files and repo_info:
                    repo_path = Path(repo_info['repo_path'])
                    if repo_path.exists():
                        try:
                            shutil.rmtree(repo_path)
                            log_statement('info', f"{self.log_prefix}:INFO>>Deleted repository files: {repo_path}", 
                                         Path(__file__).stem)
                        except Exception as e:
                            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to delete repository files: {e}", 
                                         Path(__file__).stem)
                            raise RepoHandlerError(f"Failed to delete repository files: {e}")
                
                # Remove from index
                success = self.repository_index.remove_repository(repo_id)
                if not success:
                    raise RepoHandlerError("Failed to remove repository from index")
                
                result = {
                    'repo_id': repo_id,
                    'removed': True,
                    'files_deleted': delete_files
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Removed repository: {repo_id}", Path(__file__).stem)
                return result
            
            return safe_operation("remove_repository", _do_remove_repository)
    
    def get_repository(self, repo_id: str, auto_load: bool = True) -> Optional[RepoHandler]:
        """
        Get a repository instance, optionally loading it if not active.
        
        Args:
            repo_id: Repository identifier
            auto_load: Whether to automatically load the repository if not active
            
        Returns:
            RepoHandler instance or None if not found/failed to load
        """
        # Return if already active
        if repo_id in self.active_repositories:
            return self.active_repositories[repo_id]
        
        # Auto-load if requested
        if auto_load:
            result = self.load_repository(repo_id)
            if result['status'] == OperationStatus.SUCCESS.value:
                return result['result']['repository']
        
        return None
    
    def list_repositories(self, include_inactive: bool = True) -> List[Dict[str, Any]]:
        """
        List all repositories with their status and basic information.
        
        Args:
            include_inactive: Whether to include inactive repositories
            
        Returns:
            List of repository information dictionaries
        """
        try:
            all_repo_ids = self.repository_index.list_repositories()
            repositories = []
            
            for repo_id in all_repo_ids:
                repo_info = self.repository_index.get_repository(repo_id)
                if not repo_info:
                    continue
                
                is_active = repo_id in self.active_repositories
                
                if not include_inactive and not is_active:
                    continue
                
                repo_summary = {
                    'repo_id': repo_id,
                    'repo_path': repo_info.get('repo_path'),
                    'is_active': is_active,
                    'status': repo_info.get('status', 'unknown'),
                    'added_time': repo_info.get('added_time'),
                    'last_updated': repo_info.get('last_updated'),
                    'total_tracked_files': repo_info.get('total_tracked_files', 0),
                    'total_size_bytes': repo_info.get('total_size_bytes', 0)
                }
                
                # Add current statistics if active
                if is_active:
                    current_stats = self.active_repositories[repo_id].get_repository_statistics()
                    repo_summary.update(current_stats)
                
                repositories.append(repo_summary)
            
            return repositories
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to list repositories: {e}", 
                         Path(__file__).stem, exc_info=True)
            return []
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the repository manager."""
        try:
            index_stats = self.repository_index.get_index_statistics()
            repositories = self.list_repositories()
            
            active_repos = [r for r in repositories if r['is_active']]
            total_files = sum(r.get('total_tracked_files', 0) for r in repositories)
            total_size = sum(r.get('total_size_bytes', 0) for r in repositories)
            
            stats = {
                'manager_info': {
                    'base_directory': str(self.base_directory),
                    'repositories_directory': str(self.repositories_directory),
                    'use_compression': self.use_compression
                },
                'index_statistics': index_stats,
                'repository_counts': {
                    'total_repositories': len(repositories),
                    'active_repositories': len(active_repos),
                    'inactive_repositories': len(repositories) - len(active_repos)
                },
                'aggregate_statistics': {
                    'total_tracked_files': total_files,
                    'total_size_bytes': total_size,
                    'average_files_per_repo': total_files / len(repositories) if repositories else 0
                },
                'dependency_status': DependencyChecker.check_dependencies()
            }
            
            return stats
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get manager statistics: {e}", 
                         Path(__file__).stem, exc_info=True)
            return {'error': str(e)}
    
    def cleanup_inactive_repositories(self, max_inactive_days: int = 30) -> OperationResult:
        """
        Clean up repositories that have been inactive for too long.
        
        Args:
            max_inactive_days: Maximum days a repository can be inactive before cleanup
            
        Returns:
            OperationResult with cleanup statistics
        """
        def _do_cleanup():
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_inactive_days)
            repositories = self.list_repositories(include_inactive=True)
            
            cleanup_stats = {
                'repositories_checked': len(repositories),
                'repositories_cleaned': 0,
                'errors': []
            }
            
            for repo_info in repositories:
                if repo_info['is_active']:
                    continue  # Skip active repositories
                
                last_updated_str = repo_info.get('last_updated')
                if not last_updated_str:
                    continue
                
                try:
                    last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                    
                    if last_updated < cutoff_time:
                        repo_id = repo_info['repo_id']
                        
                        # Remove from index (but not delete files)
                        remove_result = self.remove_repository(repo_id, delete_files=False)
                        if remove_result['status'] == OperationStatus.SUCCESS.value:
                            cleanup_stats['repositories_cleaned'] += 1
                            log_statement('info', f"{self.log_prefix}:INFO>>Cleaned up inactive repository: {repo_id}", 
                                         Path(__file__).stem)
                        else:
                            cleanup_stats['errors'].append(f"Failed to clean {repo_id}: {remove_result.get('error')}")
                
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error processing {repo_info['repo_id']}: {e}")
            
            return cleanup_stats
        
        return safe_operation("cleanup_inactive_repositories", _do_cleanup)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Save state for all active repositories
        for repo_id in list(self.active_repositories.keys()):
            try:
                self.unload_repository(repo_id, save_state=True)
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Error during cleanup for {repo_id}: {e}", 
                             Path(__file__).stem)


# Main execution examples and testing code
def demonstrate_repository_operations():
    """
    Demonstrate proper usage patterns for the refactored repository system.
    
    This function shows best practices for:
    - Creating and managing repositories
    - Error handling patterns
    - Integration between components
    - Proper resource management
    """
    print("=" * 80)
    print("Repository Handler Refactored - Demonstration")
    print("=" * 80)
    
    # Check dependencies first
    print("\n1. Checking Dependencies:")
    deps = DependencyChecker.check_dependencies()
    missing = DependencyChecker.get_missing_dependencies()
    
    print(f"   Available dependencies: {sum(deps.values())}/{len(deps)}")
    if missing:
        print(f"   Missing dependencies: {', '.join(missing)}")
        print("   Some features may be limited or use fallbacks")
    else:
        print("   All dependencies available - full functionality enabled")
    
    # Set up test environment
    test_base = Path("./demo_repositories")
    if test_base.exists():
        shutil.rmtree(test_base, ignore_errors=True)
    
    try:
        print(f"\n2. Initializing Repository Manager:")
        print(f"   Base directory: {test_base}")
        
        # Create repository manager with proper error handling
        with RepoManager(base_directory=test_base, use_compression=True) as manager:
            print(f"   Manager initialized successfully")
            
            # Demonstrate repository creation
            print(f"\n3. Creating Test Repositories:")
            
            # Create first repository
            repo1_path = test_base / "project_alpha"
            result1 = manager.create_repository("project_alpha", repo1_path)
            
            if result1['status'] == OperationStatus.SUCCESS.value:
                print(f"   ✓ Created repository: project_alpha")
                repo1 = result1['result']['repository']
            else:
                print(f"   ✗ Failed to create project_alpha: {result1.get('error')}")
                return
            
            # Create second repository
            repo2_path = test_base / "project_beta"
            result2 = manager.create_repository("project_beta", repo2_path)
            
            if result2['status'] == OperationStatus.SUCCESS.value:
                print(f"   ✓ Created repository: project_beta")
                repo2 = result2['result']['repository']
            else:
                print(f"   ✗ Failed to create project_beta: {result2.get('error')}")
                return
            
            # Demonstrate file operations
            print(f"\n4. Adding Files to Repositories:")
            
            # Create test files
            test_files = []
            for i, repo_path in enumerate([repo1_path, repo2_path], 1):
                data_dir = repo_path / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                
                for j in range(3):
                    test_file = data_dir / f"test_file_{j}.txt"
                    with open(test_file, 'w') as f:
                        f.write(f"Test content for repository {i}, file {j}\n")
                        f.write(f"Created at: {datetime.now().isoformat()}\n")
                    test_files.append(test_file)
            
            # Add files to repositories
            for repo, repo_name in [(repo1, "project_alpha"), (repo2, "project_beta")]:
                repo_files = [f for f in test_files if repo_name.split('_')[1] in str(f)]
                
                for file_path in repo_files:
                    result = repo.add_file_to_tracking(
                        file_path=file_path,
                        application_status=STATUS_NEW,
                        user_metadata={"demo": True, "project": repo_name}
                    )
                    
                    if result['status'] == OperationStatus.SUCCESS.value:
                        print(f"   ✓ Added file: {file_path.name} to {repo_name}")
                    else:
                        print(f"   ✗ Failed to add {file_path.name}: {result.get('error')}")
            
            # Demonstrate repository analysis
            print(f"\n5. Repository Analysis:")
            
            for repo, repo_name in [(repo1, "project_alpha"), (repo2, "project_beta")]:
                summary = repo.get_repository_summary()
                
                print(f"   {repo_name}:")
                print(f"     Files tracked: {summary['repository_info']['file_count']}")
                print(f"     Git status: {summary['git_summary'].get('is_valid', 'unknown')}")
                print(f"     Metadata entries: {summary['metadata_summary'].get('entries_count', 0)}")
                
                # Demonstrate integrity verification
                integrity_result = repo.verify_repository_integrity()
                if integrity_result['status'] == OperationStatus.SUCCESS.value:
                    integrity_report = integrity_result['result']
                    print(f"     Integrity: {integrity_report['overall_status']}")
                    print(f"     Checks performed: {len(integrity_report['checks_performed'])}")
                
            # Demonstrate manager-level operations
            print(f"\n6. Manager Operations:")
            
            manager_stats = manager.get_manager_statistics()
            print(f"   Total repositories: {manager_stats['repository_counts']['total_repositories']}")
            print(f"   Active repositories: {manager_stats['repository_counts']['active_repositories']}")
            print(f"   Total tracked files: {manager_stats['aggregate_statistics']['total_tracked_files']}")
            
            # List all repositories
            print(f"\n7. Repository Listing:")
            repositories = manager.list_repositories()
            for repo_info in repositories:
                print(f"   {repo_info['repo_id']}:")
                print(f"     Path: {repo_info['repo_path']}")
                print(f"     Status: {repo_info['status']} ({'active' if repo_info['is_active'] else 'inactive'})")
                print(f"     Files: {repo_info['total_tracked_files']}")
            
            # Demonstrate error handling
            print(f"\n8. Error Handling Examples:")
            
            # Try to create repository with existing ID
            error_result = manager.create_repository("project_alpha", test_base / "duplicate")
            print(f"   Duplicate repository creation: {error_result['status']}")
            if error_result['status'] == OperationStatus.FAILURE.value:
                print(f"   Expected error: {error_result['error']}")
            
            # Try to add non-existent file
            fake_file = repo1_path / "non_existent.txt"
            error_result = repo1.add_file_to_tracking(fake_file)
            print(f"   Non-existent file addition: {error_result['status']}")
            if error_result['status'] == OperationStatus.FAILURE.value:
                print(f"   Expected error: {error_result['error']}")
            
            print(f"\n9. Advanced Operations:")
            
            # Demonstrate batch operations
            batch_files = []
            batch_dir = repo1_path / "batch_data"
            batch_dir.mkdir(exist_ok=True)
            
            for i in range(5):
                batch_file = batch_dir / f"batch_file_{i}.dat"
                with open(batch_file, 'w') as f:
                    f.write(f"Batch file {i} content\n")
                batch_files.append(batch_file)
            
            batch_result = repo1.batch_add_files(
                file_paths=batch_files,
                common_status=STATUS_NEW,
                common_user_metadata={"batch": True, "demo": True}
            )
            
            if batch_result['status'] == OperationStatus.SUCCESS.value:
                batch_stats = batch_result['result']
                print(f"   Batch operation: {len(batch_stats['successful'])} successful, {len(batch_stats['failed'])} failed")
            
            # Demonstrate duplicate detection
            duplicate_result = repo1.find_duplicate_files()
            if duplicate_result['status'] == OperationStatus.SUCCESS.value:
                duplicate_stats = duplicate_result['result']
                print(f"   Duplicate detection: {duplicate_stats['duplicate_groups']} groups found")
            
            print(f"\n10. Final Repository State:")
            
            for repo, repo_name in [(repo1, "project_alpha"), (repo2, "project_beta")]:
                tracked_files = repo.get_tracked_files()
                print(f"   {repo_name}: {len(tracked_files)} files tracked")
                
                # Show file statuses
                status_counts = {}
                for file_path in tracked_files:
                    status = repo.get_file_status(file_path)
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                for status, count in status_counts.items():
                    print(f"     {status}: {count} files")
    
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        log_statement('error', f"Demonstration error: {e}", "demo", exc_info=True)
    
    finally:
        # Cleanup
        print(f"\n11. Cleanup:")
        if test_base.exists():
            try:
                shutil.rmtree(test_base)
                print(f"   ✓ Cleaned up test directory: {test_base}")
            except Exception as e:
                print(f"   ⚠ Could not clean up {test_base}: {e}")
        
        print(f"\nDemonstration completed!")
        print("=" * 80)


def demonstrate_advanced_integration():
    """
    Demonstrate advanced integration patterns and best practices.
    """
    print("\n" + "=" * 80)
    print("Advanced Integration Examples")
    print("=" * 80)
    
    # This would include more complex examples:
    # - Multi-repository workflows
    # - Custom metadata schemas
    # - Integration with external systems
    # - Performance optimization examples
    # - Error recovery patterns
    
    print("Advanced integration examples would be implemented here...")
    print("These would demonstrate:")
    print("- Multi-repository synchronization")
    print("- Custom metadata validation")
    print("- External system integration")
    print("- Performance monitoring")
    print("- Error recovery strategies")


# Main execution block
if __name__ == "__main__":
    # Configure logging for demonstration
    if LOGGER_AVAILABLE:
        log_statement('info', "Starting repository handler demonstration", "main")
    else:
        print("Note: Advanced logging not available - using fallback logging")
    
    try:
        # Run main demonstration
        demonstrate_repository_operations()
        
        # Optionally run advanced examples
        if len(sys.argv) > 1 and sys.argv[1] == '--advanced':
            demonstrate_advanced_integration()
    
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user")
    
    except Exception as e:
        print(f"\n\nUnexpected error during demonstration: {e}")
        if LOGGER_AVAILABLE:
            log_statement('error', f"Main execution error: {e}", "main", exc_info=True)
    
    finally:
        print("\nRepository handler demonstration finished.")


# Export main classes for use as a module
__all__ = [
    # Core classes
    'RepoHandler',
    'RepoManager',
    'RepoHandlerConfig',
    
    # Component classes
    'GitOpsHelper',
    'MetadataFileHandler',
    'ProgressFileHandler',
    'GitignoreHandler',
    'RepoAnalyzer',
    'RepoModifier',
    'RepositoryIndex',
    
    # Utility classes
    'CompressionHandler',
    'DependencyChecker',
    
    # Exception classes
    'RepoHandlerError',
    'GitOperationError',
    'MetadataError',
    'FileOperationError',
    'ConfigurationError',
    'DependencyError',
    
    # Enums
    'OperationStatus',
    'FileStatus',
    
    # Utility functions
    'safe_operation',
    'validate_path',
    'get_log_prefix'
]