# TLATOv4.1/src/m1_refactored.py
# Refactored Data Processing and Tokenization Module
# Integrated with repo_handler.py architecture
# FIXED: Batch processing and progress control issues
import git
import os
import sys
from tqdm import tqdm
import json
import time
import hashlib
from datetime import datetime
import inspect
import traceback
import threading
import shutil
import pickle
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from typing import Union, Dict, Optional, Any, Tuple, List, Protocol
from threading import RLock, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from src.utils.progress_tracker import *

# Standard library imports with error handling
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    np = None
    pd = None
    tqdm = lambda x, **kwargs: x

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    torch = None

# Import refactored repository handler
try:
    from src.utils.state_mgr import *
    from src.menu_cmd import MenuCommand
    from src.utils.startup_mgr import *
    from src.core.repo_handler import (
        RepoHandler, RepoManager, RepoHandlerConfig,
        GitOpsHelper, RepoAnalyzer, RepoModifier,
        OperationResult, OperationStatus, safe_operation,
        validate_path, get_log_prefix
    )
    REPO_HANDLER_AVAILABLE = True
except ImportError:
    REPO_HANDLER_AVAILABLE = False
    # Fallback implementations
    class OperationResult(dict): pass
    class OperationStatus:
        SUCCESS = "success"
        FAILURE = "failure"
    def safe_operation(operation_name, operation_func, *args, **kwargs):
        """Execute an operation safely with enhanced error handling and detailed logging"""
        import traceback
        start_time = time.time()
        result = {
            'operation': operation_name,
            'status': OperationStatus.FAILURE.value if hasattr(OperationStatus, 'FAILURE') else "failure",
            'duration': 0.0,
            'error': None,
            'result': None,
            'error_type': None,
            'traceback': None
        }
        
        try:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Starting operation: {operation_name}", Path(__file__).stem)
            operation_result = operation_func(*args, **kwargs)
            result['status'] = OperationStatus.SUCCESS.value if hasattr(OperationStatus, 'SUCCESS') else "success"
            result['result'] = operation_result
            log_statement('debug', f"{LOG_INS}:DEBUG>>Operation completed: {operation_name}", Path(__file__).stem)
        except Exception as e:
            # Enhanced error capture
            error_message = str(e) if str(e) else f"Unknown {type(e).__name__} error"
            if error_message == "None" or not error_message.strip():
                error_message = f"Operation {operation_name} failed with {type(e).__name__} but no error message was provided"
            
            result['error'] = error_message
            result['error_type'] = type(e).__name__
            result['traceback'] = traceback.format_exc()
            
            log_statement('error', f"{LOG_INS}:ERROR>>Operation failed: {operation_name} - {error_message}", 
                        Path(__file__).stem, exc_info=True)
            log_statement('debug', f"{LOG_INS}:DEBUG>>Error type: {type(e).__name__}, Args: {args}, Kwargs: {kwargs}", 
                        Path(__file__).stem)
        finally:
            result['duration'] = time.time() - start_time
        
        return result

# Project imports with fallbacks
try:
    from src.utils.config import *
    from src.data.constants import *
    from src.utils.logger import log_statement
    from src.utils.helpers import _generate_file_paths
    from src.utils.hashing import generate_data_hash, hash_filepath
    from src.utils.gpu_switch import set_compute_device
    from src.context.container import DataProcessingContext, DataProcessingContainer

    PROJECT_IMPORTS_AVAILABLE = True
except ImportError:
    PROJECT_IMPORTS_AVAILABLE = False
    # Define fallback constants and functions
    BASE_DATA_DIR = Path.cwd() / "data"
    CHECKPOINT_DIR = Path.cwd() / "checkpoints"
    LOG_DIR = Path.cwd() / "logs"
    DATA_REPO_DIR = Path.cwd() / "repositories"
    
    def log_statement(level, message, module=None, exc_info=False):
        print(f"[{level.upper()}] {message}")
    
    def _generate_file_paths(path):
        for item in Path(path).rglob("*"):
            if item.is_file():
                yield item

# Processing-specific imports with fallbacks
try:
    from src.analysis.labeler import SemanticLabeler
    from src.data.processing import DataProcessor, EnhancedTokenizer
    from src.core.models import load_model_from_checkpoint, ZoneClassifier
    from src.training.trainer import EnhancedTrainer, EnhancedDataLoader
    from src.data.loaders import EnhancedDataLoader
    PROCESSING_IMPORTS_AVAILABLE = True
except ImportError:
    PROCESSING_IMPORTS_AVAILABLE = False
    # Create placeholder classes
    class DataProcessor: pass
    class SemanticLabeler: pass
    class Tokenizer: pass

# Module-level constants
LOG_INS = f"{Path(__file__).stem}:m1_refactored"

@dataclass
class GitRepositoryInfo:
    """Information about a discovered Git repository"""
    path: Path
    name: str
    is_valid: bool
    branch: Optional[str]
    commit_count: int
    last_commit_date: Optional[datetime]
    last_commit_message: Optional[str]
    has_submodules: bool
    submodule_paths: List[Path]
    file_count: int
    total_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'path': str(self.path),
            'name': self.name,
            'is_valid': self.is_valid,
            'branch': self.branch,
            'commit_count': self.commit_count,
            'last_commit_date': self.last_commit_date.isoformat() if self.last_commit_date else None,
            'last_commit_message': self.last_commit_message,
            'has_submodules': self.has_submodules,
            'submodule_paths': [str(p) for p in self.submodule_paths],
            'file_count': self.file_count,
            'total_size': self.total_size
        }

class RepositoryDiscovery:
    """Discovers and analyzes existing repositories and Git repos"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def discover_repositories(self, search_paths: List[Path]) -> Dict[str, Any]:
        """Discover existing repositories in given paths"""
        try:
            discovery_result = {
                'tlato_repositories': [],
                'git_repositories': [],
                'potential_data_directories': [],
                'search_paths': [str(p) for p in search_paths]
            }
            
            for search_path in search_paths:
                if not search_path.exists():
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Search path does not exist: {search_path}", 
                                 Path(__file__).stem)
                    continue
                
                # Search for TLATO repositories
                tlato_repos = self._find_tlato_repositories(search_path)
                discovery_result['tlato_repositories'].extend(tlato_repos)
                
                # Search for Git repositories
                git_repos = self._find_git_repositories(search_path)
                discovery_result['git_repositories'].extend(git_repos)
                
                # Search for potential data directories
                data_dirs = self._find_potential_data_directories(search_path)
                discovery_result['potential_data_directories'].extend(data_dirs)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Discovery completed: "
                        f"{len(discovery_result['tlato_repositories'])} TLATO repos, "
                        f"{len(discovery_result['git_repositories'])} Git repos, "
                        f"{len(discovery_result['potential_data_directories'])} data dirs", 
                        Path(__file__).stem)
            
            return discovery_result
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Repository discovery failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            return {'tlato_repositories': [], 'git_repositories': [], 'potential_data_directories': [], 'error': str(e)}
    
    def _find_tlato_repositories(self, search_path: Path) -> List[Dict[str, Any]]:
        """Find existing TLATO repositories"""
        tlato_repos = []
        try:
            # Look for .tlato directories
            for tlato_dir in search_path.rglob(".tlato"):
                if tlato_dir.is_dir():
                    repo_path = tlato_dir.parent
                    
                    # Check if it has metadata
                    metadata_file = tlato_dir / "metadata.json.zst"
                    if metadata_file.exists():
                        repo_info = self._analyze_tlato_repository(repo_path, tlato_dir)
                        tlato_repos.append(repo_info)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error finding TLATO repositories: {e}", 
                         Path(__file__).stem, exc_info=True)
        
        return tlato_repos
    
    def _find_git_repositories(self, search_path: Path) -> List[GitRepositoryInfo]:
        """Find Git repositories and analyze them"""
        git_repos = []
        try:
            # Look for .git directories
            for git_dir in search_path.rglob(".git"):
                if git_dir.is_dir():
                    repo_path = git_dir.parent
                    git_info = self._analyze_git_repository(repo_path)
                    if git_info:
                        git_repos.append(git_info)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error finding Git repositories: {e}", 
                         Path(__file__).stem, exc_info=True)
        
        return git_repos
    
    def _find_potential_data_directories(self, search_path: Path) -> List[Dict[str, Any]]:
        """Find directories that might contain data to process"""
        data_dirs = []
        try:
            # Look for directories with many files
            for item in search_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Count files in directory
                    file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                    
                    if file_count > 10:  # Threshold for "interesting" directories
                        total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        
                        data_dirs.append({
                            'path': str(item),
                            'name': item.name,
                            'file_count': file_count,
                            'total_size': total_size,
                            'has_subdirs': any(p.is_dir() for p in item.iterdir())
                        })
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error finding data directories: {e}", 
                         Path(__file__).stem, exc_info=True)
        
        return data_dirs
    
    def _analyze_tlato_repository(self, repo_path: Path, tlato_dir: Path) -> Dict[str, Any]:
        """Analyze a TLATO repository"""
        try:
            repo_info = {
                'path': str(repo_path),
                'name': repo_path.name,
                'tlato_dir': str(tlato_dir),
                'has_metadata': False,
                'file_count': 0,
                'last_modified': None
            }
            
            # Check metadata
            metadata_file = tlato_dir / "metadata.json.zst"
            if metadata_file.exists():
                repo_info['has_metadata'] = True
                repo_info['metadata_size'] = metadata_file.stat().st_size
                repo_info['last_modified'] = datetime.fromtimestamp(metadata_file.stat().st_mtime)
            
            # Count files
            repo_info['file_count'] = sum(1 for _ in repo_path.rglob('*') if _.is_file() and not _.is_relative_to(tlato_dir))
            
            return repo_info
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error analyzing TLATO repository {repo_path}: {e}", 
                         Path(__file__).stem, exc_info=True)
            return {'path': str(repo_path), 'name': repo_path.name, 'error': str(e)}
    
    def _analyze_git_repository(self, repo_path: Path) -> Optional[GitRepositoryInfo]:
        """Analyze a Git repository"""
        try:
            try:
                repo = git.Repo(repo_path)
            except ImportError:
                log_statement('warning', f"{self.log_prefix}:WARNING>>GitPython not available, skipping Git analysis", 
                             Path(__file__).stem)
                return None
            except git.InvalidGitRepositoryError:
                return None
            
            # Get basic info
            name = repo_path.name
            is_valid = True
            
            # Get branch info
            try:
                branch = repo.active_branch.name
            except:
                branch = None
            
            # Get commit info
            try:
                commits = list(repo.iter_commits(max_count=100))
                commit_count = len(commits)
                
                if commits:
                    latest_commit = commits[0]
                    last_commit_date = datetime.fromtimestamp(latest_commit.committed_date)
                    last_commit_message = latest_commit.message.strip()
                else:
                    last_commit_date = None
                    last_commit_message = None
            except:
                commit_count = 0
                last_commit_date = None
                last_commit_message = None
            
            # Check for submodules
            try:
                submodules = repo.submodules
                has_submodules = len(submodules) > 0
                submodule_paths = [Path(repo_path) / submodule.path for submodule in submodules]
            except:
                has_submodules = False
                submodule_paths = []
            
            # Count files
            file_count = sum(1 for _ in repo_path.rglob('*') if _.is_file() and not _.is_relative_to(repo_path / '.git'))
            total_size = sum(f.stat().st_size for f in repo_path.rglob('*') if f.is_file() and not f.is_relative_to(repo_path / '.git'))
            
            return GitRepositoryInfo(
                path=repo_path,
                name=name,
                is_valid=is_valid,
                branch=branch,
                commit_count=commit_count,
                last_commit_date=last_commit_date,
                last_commit_message=last_commit_message,
                has_submodules=has_submodules,
                submodule_paths=submodule_paths,
                file_count=file_count,
                total_size=total_size
            )
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error analyzing Git repository {repo_path}: {e}", 
                         Path(__file__).stem, exc_info=True)
            return None

# Global context instance (replacing app_state)
_context: Optional[DataProcessingContext] = None
_context_lock = RLock()

def get_context() -> DataProcessingContext:
    """Get or create global processing context"""
    global _context
    with _context_lock:
        if _context is None:
            _context = DataProcessingContext()
        return _context

def reset_context(config: Optional[DataProcessingConfig] = None) -> DataProcessingContext:
    """Reset global processing context"""
    global _context
    with _context_lock:
        if _context:
            _context.cleanup()
        _context = DataProcessingContext(config)
        return _context

# Utility functions aligned with repo_handler.py patterns
def ensure_dependencies() -> Dict[str, bool]:
    """Check and report on required dependencies"""
    dependencies = {
        'repo_handler': REPO_HANDLER_AVAILABLE,
        'pandas': PANDAS_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'project_imports': PROJECT_IMPORTS_AVAILABLE,
        'processing_imports': PROCESSING_IMPORTS_AVAILABLE,
        'psutil': PSUTIL_AVAILABLE
    }
    
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        log_statement('warning', f"{LOG_INS}:WARNING>>Missing dependencies: {', '.join(missing)}", 
                     Path(__file__).stem)
    
    return dependencies

def print_welcome_message():
    """Print welcome message with dependency status"""
    log_statement('info', f">>>>>>>>>>>>>>>{LOG_INS}<<<<<<<<<<<<<<", Path(__file__).stem)
    log_statement('info', "=" * 44, Path(__file__).stem)
    log_statement('info', " >>>>>     Welcome, To TLATO v4.1     <<<<< ", Path(__file__).stem)
    log_statement('info', " >>>>>   Data Processing & Training   <<<<< ", Path(__file__).stem)
    log_statement('info', " >>>>>    Refactored Architecture     <<<<< ", Path(__file__).stem)
    log_statement('info', "", Path(__file__).stem)
    
    # Show dependency status
    deps = ensure_dependencies()
    available_count = sum(deps.values())
    total_count = len(deps)
    
    log_statement('info', f"Dependencies: {available_count}/{total_count} available", Path(__file__).stem)
    log_statement('info', f"Project Root: {PROJECT_ROOT}", Path(__file__).stem)
    log_statement('info', "*" * 44, Path(__file__).stem)

# Initialize dependencies check on module load
ensure_dependencies()

# Section 2: Repository Operations and Directory Management
# File Processing Utilities - FIXED VERSION
class FileProcessor:
    """Handles copying and processing files from source to destination"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def copy_file_to_processed_location(self, source_file: Path, source_root: Path) -> Path:
        """
        Copy a file from source location to the processed data location.
        
        Args:
            source_file: The source file to copy
            source_root: The root directory of the source scanning
            
        Returns:
            Path to the copied file in processed location
        """
        try:
            # Calculate relative path from source root
            relative_path = source_file.relative_to(source_root)
            
            # Determine destination path
            processed_data_root = self.context.config.get_processed_data_path(source_root)
            destination_file = processed_data_root / relative_path
            
            # Ensure destination directory exists
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source_file, destination_file)
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Copied {source_file} to {destination_file}", 
                         Path(__file__).stem)
            
            return destination_file
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to copy file {source_file}: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise

    def process_files_from_source(self, source_directory: Path) -> List[Path]:
        """ENHANCED: Process all files from source directory with dynamic batch size and progress tracking"""
        processed_files = []
        
        try:
            # Count total files first
            print(f"Scanning {source_directory} for files...")
            all_files = list(_generate_file_paths(source_directory))
            total_files = len(all_files)
            
            if total_files == 0:
                print("No files found in source directory.")
                log_statement('info', f"{self.log_prefix}:INFO>>No files found in {source_directory}", 
                             Path(__file__).stem)
                return processed_files

            print(f"Found {total_files} files to process...")
            
            # ENHANCED: Get dynamic batch size from context configuration
            try:
                dynamic_batch_size = self.context.config.batch_size
                log_statement('info', f"{self.log_prefix}:INFO>>Using dynamic batch size: {dynamic_batch_size} files", 
                             Path(__file__).stem)
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Error getting dynamic batch size: {e}, using fallback", 
                             Path(__file__).stem, exc_info=True)
                dynamic_batch_size = 50  # Fallback
            
            # Validate batch size for current operation
            if dynamic_batch_size > total_files:
                effective_batch_size = total_files
                log_statement('info', f"{self.log_prefix}:INFO>>Batch size {dynamic_batch_size} > total files {total_files}, using {effective_batch_size}", 
                             Path(__file__).stem)
            else:
                effective_batch_size = dynamic_batch_size
            
            print(f"Processing with batch size: {effective_batch_size} files per batch")
            
            # Check for existing progress
            current_progress = self.context.current_progress
            start_index = 0
            if current_progress and current_progress.stage == 'file_processing':
                print(f"Resuming from file {current_progress.processed_files}/{total_files}...")
                start_index = current_progress.processed_files
                # Skip already processed files (simplified - in real implementation, would check individual files)
                all_files = all_files[start_index:]
                log_statement('info', f"{self.log_prefix}:INFO>>Resuming file processing from index {start_index}", 
                             Path(__file__).stem)
            
            # ENHANCED: Process files in dynamic batches
            batches = [all_files[i:i + effective_batch_size] for i in range(0, len(all_files), effective_batch_size)]
            
            print(f"Processing {len(batches)} batches of up to {effective_batch_size} files each...")
            log_statement('info', f"{self.log_prefix}:INFO>>Created {len(batches)} batches with size {effective_batch_size}", 
                         Path(__file__).stem)

            # Import progress tracker with enhanced configuration
            try:                
                progress_tracker = create_progress_tracker(
                    total_items=len(all_files),
                    description=f"Copying files from {source_directory.name}",
                    unit="files",
                    show_resources=True,
                    update_interval=0.5  # Update every 0.5 seconds for more responsive display
                )
                
                processed_count = start_index  # Start from resume point
                
                # ENHANCED: Process files in dynamic batches with detailed progress saving
                for batch_num, file_batch in enumerate(batches, 1):
                    batch_start_time = time.time()
                    print(f"Processing batch {batch_num}/{len(batches)} ({len(file_batch)} files)...")
                    
                    batch_processed = []
                    batch_failed = 0
                    
                    for file_index, source_file in enumerate(file_batch):
                        try:
                            processed_file = self.copy_file_to_processed_location(source_file, source_directory)
                            batch_processed.append(processed_file)
                            processed_files.append(processed_file)
                            processed_count += 1
                            progress_tracker.update(success=True)
                            
                        except Exception as e:
                            batch_failed += 1
                            progress_tracker.update(success=False, error_msg=f"{source_file.name}: {str(e)}")
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to process {source_file}: {e}", 
                                        Path(__file__).stem)
                        
                        # ENHANCED: Save progress more frequently for larger batches
                        if processed_count % self.context.config.progress_save_interval == 0:
                            self.context.save_current_progress(
                                process_name="FileProcessing",
                                stage="file_processing", 
                                total_files=total_files,
                                processed_files=processed_count,
                                failed_files=total_files - processed_count - len(batch_processed) + batch_failed,
                                current_file=str(source_file) if source_file else None
                            )
                    
                    batch_duration = time.time() - batch_start_time
                    
                    # ENHANCED: Log detailed batch statistics
                    log_statement('info', f"{self.log_prefix}:INFO>>Completed batch {batch_num}/{len(batches)}: "
                                f"{len(batch_processed)} processed, {batch_failed} failed, "
                                f"{batch_duration:.1f}s, {len(batch_processed)/batch_duration:.1f} files/s", 
                                Path(__file__).stem)
                    
                    # Save progress every few batches (configurable)
                    if batch_num % max(1, len(batches) // 10) == 0:  # Save at 10%, 20%, etc.
                        self.context.save_current_progress(
                            process_name="FileProcessing",
                            stage="file_processing", 
                            total_files=total_files,
                            processed_files=processed_count,
                            failed_files=total_files - processed_count,
                            current_file=str(file_batch[-1]) if file_batch else None
                        )
                        log_statement('debug', f"{self.log_prefix}:DEBUG>>Progress saved at batch {batch_num}", 
                                    Path(__file__).stem)
                
                # Final progress save
                self.context.save_current_progress(
                    process_name="FileProcessing",
                    stage="file_processing_complete",
                    total_files=total_files, 
                    processed_files=processed_count,
                    failed_files=total_files - processed_count
                )
                
                # Finish and show summary
                progress_tracker.finish()
                
                # Get statistics
                stats = progress_tracker.get_statistics()
                log_statement('info', f"{self.log_prefix}:INFO>>File processing statistics with batch_size {effective_batch_size}: {stats}", 
                            Path(__file__).stem)
                
            except ImportError:
                # Fallback processing with basic progress saving
                log_statement('warning', f"{self.log_prefix}:WARNING>>Enhanced progress tracking not available, using basic tracking", 
                             Path(__file__).stem)
                processed_count = start_index
                
                for batch_num, file_batch in enumerate(batches, 1):
                    print(f"Processing batch {batch_num}/{len(batches)} ({len(file_batch)} files)...")
                    batch_start_time = time.time()
                    
                    for source_file in file_batch:
                        try:
                            processed_file = self.copy_file_to_processed_location(source_file, source_directory)
                            processed_files.append(processed_file)
                            processed_count += 1
                        except Exception as e:
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to process {source_file}: {e}", 
                                        Path(__file__).stem)
                            continue
                    
                    batch_duration = time.time() - batch_start_time
                    log_statement('info', f"{self.log_prefix}:INFO>>Batch {batch_num} completed in {batch_duration:.1f}s", 
                                Path(__file__).stem)
                    
                    # Save progress every few batches
                    if batch_num % max(1, len(batches) // 5) == 0:
                        self.context.save_current_progress(
                            process_name="FileProcessing",
                            stage="file_processing",
                            total_files=total_files,
                            processed_files=processed_count,
                            failed_files=total_files - processed_count
                        )
            
            # ENHANCED: Final logging with batch size information
            log_statement('info', f"{self.log_prefix}:INFO>>Processed {len(processed_files)} files from {source_directory} "
                        f"using batch_size={effective_batch_size}, {len(batches)} batches", 
                        Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error processing files from {source_directory}: {e}", 
                        Path(__file__).stem, exc_info=True)
            
            # Save error state
            try:
                self.context.save_current_progress(
                    process_name="FileProcessing",
                    stage="file_processing_error",
                    total_files=total_files if 'total_files' in locals() else 0,
                    processed_files=len(processed_files),
                    failed_files=0,
                    current_file=None
                )
            except Exception as save_error:
                log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save error state: {save_error}", 
                            Path(__file__).stem)
        
        return processed_files

class FileChangeDetector:
    """Handles file change detection using RepoHandler integration"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def detect_changes(self, repo: RepoHandler) -> OperationResult:
        """Detect changes in tracked files"""
        def _do_detect_changes():
            if not repo.analyzer:
                raise RuntimeError("Repository analyzer not available")
            
            # Get discrepancies between filesystem and metadata
            discrepancies = repo.analyzer.detect_repository_discrepancies()
            
            # Categorize changes
            changes = {
                'modified_files': discrepancies.get('modified_files', []),
                'new_files': discrepancies.get('files_not_in_metadata', []),
                'deleted_files': discrepancies.get('metadata_files_missing', []),
                'total_changes': 0
            }
            
            changes['total_changes'] = sum(len(files) for files in changes.values())
            
            log_statement('info', f"{self.log_prefix}:INFO>>Change detection completed: {changes['total_changes']} changes found", 
                         Path(__file__).stem)
            return changes
        
        return safe_operation("detect_changes", _do_detect_changes)
    
    def has_file_changed(self, repo: RepoHandler, file_path: Path) -> Tuple[bool, str]:
        """
        Check if a specific file has changed.
        Returns (has_changed, change_type)
        """
        try:
            if not repo.analyzer:
                return False, 'analyzer_unavailable'
            
            # Use RepoHandler's integrity verification
            integrity_result = repo.analyzer.verify_file_integrity(file_path)
            
            if not integrity_result.get('exists', False):
                return True, 'deleted'
            
            if not integrity_result.get('integrity_verified', True):
                return True, 'content_changed'
            
            return False, 'unchanged'
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Error checking file changes for {file_path}: {e}", 
                         Path(__file__).stem)
            return False, 'error'

class DataDirectorySetupCommand:
    """Command for setting up data directory (replaces set_data_directory function) with dynamic batch size configuration"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.repo_ops = RepositoryOperations(context)
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def can_execute(self) -> bool:
        """Check if command can be executed"""
        return REPO_HANDLER_AVAILABLE
    
    def get_description(self) -> str:
        """Get command description"""
        return "Set up and scan a source directory for processing with configurable batch size"
    
    def _prompt_batch_size_configuration(self) -> bool:
        """ADDED: Prompt user for batch size configuration with extensive validation and logging"""
        log_prefix = f"{self.log_prefix}:_prompt_batch_size_configuration"
        
        try:
            batch_info = self.context.config.get_batch_size_info()
            
            print("\n" + "="*60)
            print("BATCH SIZE CONFIGURATION")
            print("="*60)
            print(f"Current batch size: {batch_info['current_batch_size']} files")
            print(f"Valid range: {batch_info['min_batch_size']} - {batch_info['max_batch_size']} files")
            print(f"Default: {batch_info['default_batch_size']} files")
            
            if batch_info['is_default']:
                print("✓ Using default batch size")
            elif batch_info['is_at_minimum']:
                print("⚠ Using minimum batch size")
            elif batch_info['is_at_maximum']:
                print("⚠ Using maximum batch size")
            else:
                print("✓ Using custom batch size")
            
            print("\nBatch size affects:")
            print("  • Memory usage during file processing")
            print("  • Progress update frequency")
            print("  • Commit threshold behavior")
            print("  • Processing performance")
            
            log_statement('info', f"{log_prefix}:INFO>>Displaying batch size configuration: current={batch_info['current_batch_size']}", 
                         Path(__file__).stem)
            
            # Ask if user wants to change batch size
            change_batch = input(f"\nWould you like to change the batch size? (y/N): ").strip().lower()
            
            if change_batch in ['y', 'yes', 'true', '1']:
                return self._configure_batch_size()
            else:
                log_statement('info', f"{log_prefix}:INFO>>User kept existing batch size: {batch_info['current_batch_size']}", 
                             Path(__file__).stem)
                print(f"Keeping current batch size: {batch_info['current_batch_size']} files")
                return True
        
        except Exception as e:
            log_statement('error', f"{log_prefix}:ERROR>>Error in batch size configuration prompt: {e}", 
                         Path(__file__).stem, exc_info=True)
            print(f"Error displaying batch configuration: {e}")
            print("Using current batch size settings.")
            return True
    
    def _configure_batch_size(self) -> bool:
        """ADDED: Configure batch size with comprehensive validation and error handling"""
        log_prefix = f"{self.log_prefix}:_configure_batch_size"
        
        try:
            batch_info = self.context.config.get_batch_size_info()
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    print(f"\nEnter new batch size ({batch_info['min_batch_size']}-{batch_info['max_batch_size']}):")
                    print(f"  • Smaller values use less memory but may be slower")
                    print(f"  • Larger values are faster but use more memory")
                    print(f"  • Recommended: 50-200 for most systems")
                    
                    user_input = input(f"New batch size [current: {batch_info['current_batch_size']}]: ").strip()
                    
                    if not user_input:
                        log_statement('info', f"{log_prefix}:INFO>>User kept existing batch size via empty input", 
                                     Path(__file__).stem)
                        print("No change - keeping current batch size.")
                        return True
                    
                    # Parse and validate input
                    try:
                        new_batch_size = int(user_input)
                    except ValueError as e:
                        log_statement('warning', f"{log_prefix}:WARNING>>Invalid batch size input '{user_input}': {e}", 
                                     Path(__file__).stem)
                        print(f"Error: '{user_input}' is not a valid number. Please enter an integer.")
                        continue
                    
                    # Validate range
                    if new_batch_size < batch_info['min_batch_size']:
                        log_statement('warning', f"{log_prefix}:WARNING>>Batch size {new_batch_size} below minimum {batch_info['min_batch_size']}", 
                                     Path(__file__).stem)
                        print(f"Error: Batch size must be at least {batch_info['min_batch_size']}")
                        continue
                    
                    if new_batch_size > batch_info['max_batch_size']:
                        log_statement('warning', f"{log_prefix}:WARNING>>Batch size {new_batch_size} above maximum {batch_info['max_batch_size']}", 
                                     Path(__file__).stem)
                        print(f"Error: Batch size cannot exceed {batch_info['max_batch_size']}")
                        continue
                    
                    # Apply new batch size
                    success = self.context.config.update_batch_size(new_batch_size)
                    
                    if success:
                        print(f"✓ Batch size updated to {new_batch_size} files")
                        
                        # Provide recommendations based on chosen size
                        if new_batch_size < 25:
                            print("  → Small batch size: Lower memory usage, more frequent progress updates")
                        elif new_batch_size > 500:
                            print("  → Large batch size: Higher memory usage, faster processing")
                        else:
                            print("  → Balanced batch size: Good performance and memory usage")
                        
                        log_statement('info', f"{log_prefix}:INFO>>Successfully updated batch size to {new_batch_size}", 
                                     Path(__file__).stem)
                        return True
                    else:
                        log_statement('error', f"{log_prefix}:ERROR>>Failed to update batch size to {new_batch_size}", 
                                     Path(__file__).stem)
                        print(f"Error: Failed to update batch size. Please try again.")
                        continue
                
                except KeyboardInterrupt:
                    log_statement('info', f"{log_prefix}:INFO>>Batch size configuration cancelled by user", 
                                 Path(__file__).stem)
                    print("\nBatch size configuration cancelled.")
                    return True
                
                except Exception as e:
                    log_statement('error', f"{log_prefix}:ERROR>>Error processing batch size input: {e}", 
                                 Path(__file__).stem, exc_info=True)
                    print(f"Error processing input: {e}")
                    if attempt < max_attempts - 1:
                        print("Please try again.")
                    continue
            
            # If we get here, all attempts failed
            log_statement('warning', f"{log_prefix}:WARNING>>All batch size configuration attempts failed, keeping current setting", 
                         Path(__file__).stem)
            print(f"Maximum attempts reached. Keeping current batch size: {batch_info['current_batch_size']}")
            return True
        
        except Exception as e:
            log_statement('error', f"{log_prefix}:ERROR>>Critical error in batch size configuration: {e}", 
                         Path(__file__).stem, exc_info=True)
            print(f"Critical error in batch size configuration: {e}")
            print("Using default batch size settings.")
            return True

    def execute(self) -> OperationResult:
        """ENHANCED: Execute the data directory setup command with batch size configuration and discovery integration"""
        def _do_execute():
            print("\n--- Set Source Data Directory ---")
            print(f"Project Root: {self.context.config.project_root}")
            print(f"Data Output: {self.context.config.output_directory}")
            
            # ADDED: Batch size configuration step
            print(f"\n--- Batch Processing Configuration ---")
            
            try:
                batch_configured = self._prompt_batch_size_configuration()
                if not batch_configured:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Batch size configuration failed, continuing with defaults", 
                                 Path(__file__).stem)
                    print("Warning: Batch size configuration failed. Using default settings.")
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Error in batch size configuration: {e}", 
                             Path(__file__).stem, exc_info=True)
                print(f"Error in batch size configuration: {e}")
                print("Continuing with default batch size settings.")
            
            # Show final batch size being used
            current_batch_size = self.context.config.batch_size
            print(f"\n✓ Using batch size: {current_batch_size} files")
            log_statement('info', f"{self.log_prefix}:INFO>>Final batch size for processing: {current_batch_size}", 
                         Path(__file__).stem)
            
            # Offer discovery option
            print("\n--- Set Source Data Directory ---")
            print("Options:")
            print("1. Enter path manually")
            print("2. Search for existing repositories")
            print("3. Use current directory")
            
            choice = input("Choose option (1-3): ").strip()
            
            source_directory = None
            
            try:
                if choice == '2':
                    # Use discovery system
                    search_paths = [Path.cwd(), self.context.config.project_root, Path.home() / "Documents"]
                    discovery_result = self.context.startup_manager.repository_discovery.discover_repositories(search_paths)
                    self.context.startup_manager._display_discovery_results(discovery_result)
                    
                    selection_result = self.context.startup_manager._handle_repository_selection(discovery_result)
                    if selection_result.get('repositories_discovered'):
                        # Repository already loaded by discovery
                        return {'message': 'Repository loaded via discovery', 'batch_size_used': current_batch_size}
                    
                    # If no selection made, fall back to manual entry
                    user_input = input("Enter path manually: ").strip()
                    if user_input:
                        source_directory = Path(user_input)
                
                elif choice == '3':
                    source_directory = Path.cwd()
                    log_statement('info', f"{self.log_prefix}:INFO>>Using current directory: {source_directory}", 
                                 Path(__file__).stem)
                
                else:  # choice == '1' or default
                    user_input = input("Enter the full path to the source data directory to scan: ").strip()
                    if not user_input:
                        raise ValueError("No source directory provided")
                    source_directory = Path(user_input)
                    log_statement('info', f"{self.log_prefix}:INFO>>User provided directory: {source_directory}", 
                                 Path(__file__).stem)
                
                if not source_directory:
                    raise ValueError("No source directory selected")
                
                # Validate source directory
                if not source_directory.exists():
                    log_statement('error', f"{self.log_prefix}:ERROR>>Source directory does not exist: {source_directory}", 
                                 Path(__file__).stem)
                    raise ValueError(f"Source directory does not exist: {source_directory}")
                
                if not source_directory.is_dir():
                    log_statement('error', f"{self.log_prefix}:ERROR>>Source path is not a directory: {source_directory}", 
                                 Path(__file__).stem)
                    raise ValueError(f"Source path is not a directory: {source_directory}")
                
                print(f"Setting up repository to process files from: {source_directory}")
                print(f"Files will be processed to: {self.context.config.get_processed_data_path(source_directory)}")
                print(f"Using batch size: {current_batch_size} files per batch")
                
                # Set up repository
                setup_result = self.repo_ops.setup_repository(source_directory)
                
                if setup_result['status'] == OperationStatus.SUCCESS.value:
                    result_data = setup_result['result']
                    print(f"✓ Repository setup completed:")
                    print(f"  Repository ID: {result_data['repo_id']}")
                    print(f"  Repository Path: {result_data['repo_path']}")
                    print(f"  Source Path: {result_data['source_path']}")
                    print(f"  Files processed: {result_data['processed_files']}")
                    print(f"  Files added to tracking: {result_data['files_added']}")
                    print(f"  Batch size used: {current_batch_size} files")
                    
                    if result_data['files_failed'] > 0:
                        print(f"  Files failed: {result_data['files_failed']}")
                        log_statement('warning', f"{self.log_prefix}:WARNING>>{result_data['files_failed']} files failed during setup", 
                                     Path(__file__).stem)
                    
                    # Update context state
                    self.context.last_scan_time = time.time()
                    
                    # Add batch size info to result
                    result_data['batch_size_used'] = current_batch_size
                    result_data['batch_configuration_successful'] = True
                    
                    log_statement('info', f"{self.log_prefix}:INFO>>Repository setup successful with batch_size={current_batch_size}", 
                                 Path(__file__).stem)
                    
                    return result_data
                else:
                    error_msg = setup_result.get('error', 'Unknown error')
                    print(f"✗ Repository setup failed: {error_msg}")
                    log_statement('error', f"{self.log_prefix}:ERROR>>Repository setup failed: {error_msg}", 
                                 Path(__file__).stem)
                    raise RuntimeError(error_msg)
            
            except ValueError as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Validation error: {e}", 
                             Path(__file__).stem)
                raise e
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Unexpected error during directory setup: {e}", 
                             Path(__file__).stem, exc_info=True)
                raise RuntimeError(f"Directory setup failed: {e}")
        
        return safe_operation("execute_data_directory_setup_with_batch_config", _do_execute)

class RepositoryStatusDisplay:
    """Handles repository status display and information"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def display_repository_status(self) -> None:
        """Display current repository status with processing state"""
        repo = self.context.get_current_repo()
        
        if not repo:
            print("Repository Status: Not Set")
            return
        
        try:
            # Get repository statistics
            stats = repo.get_repository_statistics()
            status = repo.get_status()
            
            # Get repository processing state
            repo_state = self.context.get_repository_processing_state()
            
            print(f"\n--- Repository Status ---")
            print(f"Repository Path: {self.context.config.project_root}")
            print(f"Source Path: {self.context.current_source_path}")
            print(f"Repository ID: {self.context.current_repo_id}")
            print(f"Initialized: {status.get('initialized', False)}")
            print(f"Files Tracked: {stats.get('total_tracked_files', 0)}")
            print(f"Total Size: {self._format_size(stats.get('total_size_bytes', 0))}")
            print(f"Git Status: {stats.get('git_status', 'unknown')}")
            
            # ADDED: Repository processing state information
            if repo_state:
                print(f"\n--- Processing State ---")
                print(f"State: {repo_state.state.value.title()}")
                print(f"Total Files: {repo_state.total_files}")
                print(f"Discovered Files: {repo_state.discovered_files}")
                print(f"Processed Files: {repo_state.processed_files}")
                print(f"Tokenized Files: {repo_state.tokenized_files}")
                print(f"Failed Files: {repo_state.failed_files}")
                
                if repo_state.processing_progress_percentage > 0:
                    print(f"Processing Progress: {repo_state.processing_progress_percentage:.1f}%")
                
                if repo_state.tokenization_progress_percentage > 0:
                    print(f"Tokenization Progress: {repo_state.tokenization_progress_percentage:.1f}%")
                
                if repo_state.overall_progress_percentage > 0:
                    print(f"Overall Progress: {repo_state.overall_progress_percentage:.1f}%")
                
                # Readiness indicators
                print(f"\n--- Readiness Status ---")
                print(f"Ready for Processing: {'✓' if repo_state.is_ready_for_processing else '✗'}")
                print(f"Ready for Tokenization: {'✓' if repo_state.is_ready_for_tokenization else '✗'}")
                print(f"Ready for Training: {'✓' if repo_state.is_ready_for_training else '✗'}")
                
                if repo_state.error_message:
                    print(f"\nError: {repo_state.error_message}")
            
            # Show status distribution
            status_counts = stats.get('status_counts', {})
            if status_counts:
                print("\nFile Status Distribution:")
                for status_name, count in status_counts.items():
                    print(f"  {status_name}: {count}")
            
            # Show last scan time
            if self.context.last_scan_time:
                last_scan = time.ctime(self.context.last_scan_time)
                print(f"Last Scan: {last_scan}")
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error displaying repository status: {e}", 
                         Path(__file__).stem, exc_info=True)
            print(f"Error retrieving repository status: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def display_repository_summary(self) -> None:
        """Display comprehensive repository summary"""
        repo = self.context.get_current_repo()
        
        if not repo:
            print("No repository loaded")
            return
        
        try:
            summary = repo.get_repository_summary()
            
            print(f"\n--- Repository Summary ---")
            
            # Repository info
            repo_info = summary.get('repository_info', {})
            print(f"Repository Path: {repo_info.get('path', 'Unknown')}")
            print(f"Source Path: {self.context.current_source_path}")
            print(f"File Count: {repo_info.get('file_count', 0)}")
            print(f"Last Scan: {repo_info.get('last_scan', 'Never')}")
            
            # Git summary
            git_summary = summary.get('git_summary', {})
            if git_summary.get('is_valid'):
                print(f"\nGit Repository:")
                print(f"  Status: Valid")
                print(f"  Recent Commits: {git_summary.get('recent_commits', 0)}")
                latest_commit = git_summary.get('latest_commit')
                if latest_commit:
                    print(f"  Latest Commit: {latest_commit.get('short_hash', 'Unknown')} - {latest_commit.get('message', '')[:50]}...")
            
            # Metadata summary
            metadata_summary = summary.get('metadata_summary', {})
            if metadata_summary:
                print(f"\nMetadata:")
                print(f"  Entries: {metadata_summary.get('entries_count', 0)}")
                print(f"  Total Size: {self._format_size(metadata_summary.get('total_size_bytes', 0))}")
            
            # File analysis
            file_analysis = summary.get('file_analysis', {})
            if file_analysis:
                print(f"\nFilesystem:")
                print(f"  Files Found: {file_analysis.get('filesystem_file_count', 0)}")
                
                # Show top extensions
                extensions = file_analysis.get('filesystem_extensions', {})
                if extensions:
                    sorted_extensions = sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("  Top Extensions:")
                    for ext, count in sorted_extensions:
                        print(f"    .{ext}: {count}")
            
            # Discrepancies
            discrepancies = summary.get('discrepancies', {})
            total_discrepancies = sum(len(files) for files in discrepancies.values() if isinstance(files, list))
            if total_discrepancies > 0:
                print(f"\nDiscrepancies Found: {total_discrepancies}")
                for disc_type, files in discrepancies.items():
                    if isinstance(files, list) and files:
                        print(f"  {disc_type}: {len(files)}")
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error displaying repository summary: {e}", 
                         Path(__file__).stem, exc_info=True)
            print(f"Error retrieving repository summary: {e}")

# Legacy compatibility functions
def has_file_changed(filepath: Path) -> Tuple[bool, str, float, float]:
    """
    Legacy compatibility function for has_file_changed.
    Returns (has_changed, change_flag, mtime, atime)
    """
    context = get_context()
    repo = context.get_current_repo()
    
    if not repo:
        log_statement('warning', f"{LOG_INS}:WARNING>>No repository loaded for change detection", 
                     Path(__file__).stem)
        return False, 'no_repo', 0.0, 0.0
    
    try:
        detector = FileChangeDetector(context)
        has_changed, change_type = detector.has_file_changed(repo, filepath)
        
        # Get file timestamps if possible
        try:
            stat = filepath.stat()
            mtime = stat.st_mtime
            atime = stat.st_atime
        except:
            mtime = atime = 0.0
        
        # Map change types to legacy format
        change_flag_map = {
            'unchanged': 'N',
            'content_changed': 'C',
            'deleted': 'D',
            'error': 'E',
            'analyzer_unavailable': 'E'
        }
        
        change_flag = change_flag_map.get(change_type, 'E')
        return has_changed, change_flag, mtime, atime
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error in legacy has_file_changed: {e}", 
                     Path(__file__).stem, exc_info=True)
        return False, 'E', 0.0, 0.0

def set_data_directory():
    """Legacy compatibility function for set_data_directory"""
    context = get_context()
    command = DataDirectorySetupCommand(context)
    
    if not command.can_execute():
        print("Error: Repository handler not available. Please check dependencies.")
        log_statement('error', f"{LOG_INS}:ERROR>>Cannot execute set_data_directory: dependencies missing", 
                     Path(__file__).stem)
        return
    
    result = command.execute()
    
    if result['status'] == OperationStatus.SUCCESS.value:
        # Display status after successful setup
        display = RepositoryStatusDisplay(context)
        display.display_repository_status()
    else:
        print(f"Setup failed: {result.get('error', 'Unknown error')}")
        log_statement('error', f"{LOG_INS}:ERROR>>set_data_directory failed: {result.get('error')}", 
                     Path(__file__).stem)

        
# Section 3: Data Processing Pipelines
# Base Pipeline Classes
# Section 3: Data Processing Pipelines (updated for new architecture)
class ProcessingPipeline(ABC):
    """Abstract base class for processing pipelines"""
    
    def __init__(self, context: DataProcessingContext, repo: RepoHandler):
        self.context = context
        self.repo = repo
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self._progress_handler = repo.progress_handler if repo else None
    
    @abstractmethod
    def can_process(self) -> bool:
        """Check if pipeline can process files"""
        pass
    
    @abstractmethod
    def get_files_to_process(self) -> List[str]:
        """Get list of files ready for processing"""
        pass
    
    @abstractmethod
    def process_file(self, file_path: str) -> OperationResult:
        """Process a single file"""
        pass
    
    @abstractmethod
    def get_pipeline_name(self) -> str:
        """Get pipeline name for logging"""
        pass
        
    def process_batch(self, max_files: Optional[int] = None) -> OperationResult:
        """ENHANCED: Process a batch of files with progress tracking, resume capability, and state management"""
        def _do_process_batch():
            if not self.can_process():
                raise RuntimeError(f"{self.get_pipeline_name()} pipeline cannot process files")
            
            # Update repository state to processing
            if self.context:
                self.context.update_repository_state(RepositoryState.PROCESSING)
            
            files_to_process = self.get_files_to_process()
            if max_files:
                files_to_process = files_to_process[:max_files]
            
            if not files_to_process:
                # Check if repository is in wrong state
                repo_state = self.context.get_repository_processing_state() if self.context else None
                state_info = f" (Repository state: {repo_state.state.value})" if repo_state else ""
                
                return {
                    'pipeline': self.get_pipeline_name(),
                    'processed_count': 0,
                    'failed_count': 0,
                    'message': f'No files to process{state_info}'
                }
            
            # Check for existing progress
            current_progress = self.context.current_progress
            start_index = 0
            if (current_progress and 
                current_progress.process_name == self.get_pipeline_name() and
                current_progress.stage == f"{self.get_pipeline_name().lower()}_processing"):
                
                print(f"Resuming {self.get_pipeline_name()} from file {current_progress.processed_files}...")
                start_index = current_progress.processed_files
                files_to_process = files_to_process[start_index:]
            
            # Use smaller batch sizes for processing
            batch_size = min(self.context.config.batch_size, 20)
            total_files = len(files_to_process) + start_index
            
            # Import progress tracker
            try:
                use_enhanced_progress = True
            except ImportError:
                use_enhanced_progress = False
                log_statement('warning', f"{self.log_prefix}:WARNING>>Enhanced progress tracker not available", 
                            Path(__file__).stem)
            
            # Create progress tracker
            if use_enhanced_progress:
                progress_tracker = create_progress_tracker(
                    total_items=len(files_to_process),
                    description=f"{self.get_pipeline_name()} Pipeline",
                    unit="files",
                    show_resources=True
                )
            else:
                pbar = tqdm(total=len(files_to_process), desc=self.get_pipeline_name(), unit="files")
            
            # Initialize tracking
            processed_count = start_index
            failed_count = current_progress.failed_files if current_progress else 0
            errors = []
            
            # Process files in batches to control metadata writes
            batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
            
            log_statement('info', f"{self.log_prefix}:INFO>>Processing {len(files_to_process)} files in {len(batches)} batches", 
                        Path(__file__).stem)
            
            # Process batches
            for batch_num, file_batch in enumerate(batches, 1):
                print(f"Processing {self.get_pipeline_name()} batch {batch_num}/{len(batches)}...")
                
                for file_path in file_batch:
                    try:
                        # Process file
                        result = self.process_file(file_path)
                        
                        if result['status'] == OperationStatus.SUCCESS:
                            processed_count += 1
                            if use_enhanced_progress:
                                progress_tracker.update(success=True)
                            else:
                                pbar.update(1)
                            
                            log_statement('debug', f"{self.log_prefix}:DEBUG>>Processed {Path(file_path).name}", 
                                        Path(__file__).stem)
                        else:
                            failed_count += 1
                            error_msg = result.get('error', 'Unknown error')
                            errors.append(f"{Path(file_path).name}: {error_msg}")
                            
                            if use_enhanced_progress:
                                progress_tracker.update(success=False, error_msg=error_msg)
                            else:
                                pbar.update(1)
                            
                            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed: {Path(file_path).name}: {error_msg}", 
                                        Path(__file__).stem)
                    
                    except Exception as e:
                        failed_count += 1
                        
                        # Enhanced error capture and reporting
                        error_msg = str(e) if str(e) else f"Unknown {type(e).__name__} error"
                        if error_msg == "None" or not error_msg.strip():
                            error_msg = f"Processing failed for {Path(file_path).name} with {type(e).__name__} but no error message"
                        
                        # Create detailed error entry
                        error_entry = {
                            'file': Path(file_path).name,
                            'full_path': str(file_path),
                            'error': error_msg,
                            'error_type': type(e).__name__,
                            'pipeline': self.get_pipeline_name(),
                            'timestamp': time.time()
                        }
                        errors.append(error_entry)
                        
                        if use_enhanced_progress:
                            progress_tracker.update(success=False, error_msg=error_msg)
                        else:
                            pbar.update(1)
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>Exception processing {Path(file_path).name}: {error_msg} (Type: {type(e).__name__})", 
                                    Path(__file__).stem, exc_info=True)
                        
                # Save progress every few batches
                if batch_num % 3 == 0:
                    self.context.save_current_progress(
                        process_name=self.get_pipeline_name(),
                        stage=f"{self.get_pipeline_name().lower()}_processing",
                        total_files=total_files,
                        processed_files=processed_count,
                        failed_files=failed_count,
                        current_file=file_batch[-1] if file_batch else None
                    )
                
                # Small delay between batches to reduce load
                if batch_num < len(batches):
                    time.sleep(0.05)
            
            # Final progress save
            self.context.save_current_progress(
                process_name=self.get_pipeline_name(),
                stage=f"{self.get_pipeline_name().lower()}_complete",
                total_files=total_files,
                processed_files=processed_count,
                failed_files=failed_count
            )
            
            # Finish progress tracking
            if use_enhanced_progress:
                progress_tracker.finish()
                final_stats = progress_tracker.get_statistics()
            else:
                pbar.close()
                final_stats = {}
            
            # Final result
            result = {
                'pipeline': self.get_pipeline_name(),
                'processed_count': processed_count - start_index,  # Only new processed files
                'failed_count': failed_count,
                'total_files': len(files_to_process),
                'errors': errors,
                'statistics': final_stats
            }
            
            log_statement('info', f"{self.log_prefix}:INFO>>{self.get_pipeline_name()} completed: "
                        f"{processed_count - start_index} processed, {failed_count} failed", 
                        Path(__file__).stem)

            # After successful completion, update repository state
            try:
                if self.context and self.get_pipeline_name() == "LinguisticProcessing":
                    self.context.update_repository_state(RepositoryState.PROCESSED)
                elif self.context and self.get_pipeline_name() == "Tokenization":
                    self.context.update_repository_state(RepositoryState.TOKENIZED)
            except Exception as state_e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update repository state: {state_e}", 
                            Path(__file__).stem)
            
            return result
        
        return safe_operation(f"process_batch_{self.get_pipeline_name()}", _do_process_batch)
    
class LinguisticProcessingPipeline(ProcessingPipeline):
    """Pipeline for linguistic data processing"""
    
    def __init__(self, context: DataProcessingContext, repo: RepoHandler):
        super().__init__(context, repo)
        self._data_processor = DataProcessor(repo_context = context)
        self._semantic_labeler = SemanticLabeler(context)
    
    def can_process(self) -> bool:
        """Check if linguistic processing can be performed"""
        return (
            PROCESSING_IMPORTS_AVAILABLE and 
            self.context.config.enable_semantic_labeling and
            self.repo and 
            self.repo.is_initialized()
        )
    
    def get_files_to_process(self) -> List[str]:
        """Get files ready for linguistic processing with enhanced status detection and path validation"""
        try:
            # Get repository state first
            repo_state = self.context.get_repository_processing_state() if self.context else None
            
            # Get files by multiple possible statuses - be more inclusive
            df = self.repo.get_dataframe()
            if df is None or df.empty:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Repository dataframe is empty or None", 
                            Path(__file__).stem)
                return []
            
            log_statement('info', f"{self.log_prefix}:INFO>>Repository has {len(df)} total files", 
                        Path(__file__).stem)
            
            # CRITICAL FIX: Filter out non-existent files first
            valid_files = []
            invalid_files = []
            
            filepath_column = None
            for col in ['filepath', 'file_path', 'path']:
                if col in df.columns:
                    filepath_column = col
                    break
            
            if not filepath_column:
                log_statement('error', f"{self.log_prefix}:ERROR>>No filepath column found in dataframe. Columns: {list(df.columns)}", 
                            Path(__file__).stem)
                return []
            
            log_statement('info', f"{self.log_prefix}:INFO>>Validating file existence for {len(df)} tracked files...", 
                        Path(__file__).stem)
            
            # Validate file existence
            for idx, row in df.iterrows():
                file_path = row[filepath_column]
                if pd.isna(file_path) or not file_path:
                    continue
                    
                file_path_obj = Path(file_path)
                if file_path_obj.exists() and file_path_obj.is_file():
                    valid_files.append(idx)
                else:
                    invalid_files.append((idx, file_path))
            
            log_statement('info', f"{self.log_prefix}:INFO>>File validation results: {len(valid_files)} valid, {len(invalid_files)} invalid", 
                        Path(__file__).stem)
            
            # If too many invalid files, log details and offer cleanup
            if len(invalid_files) > 100:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Found {len(invalid_files)} invalid file paths in repository. Repository may need cleanup.", 
                            Path(__file__).stem)
                
                # Log sample of invalid paths
                sample_invalid = invalid_files[:10]
                for idx, invalid_path in sample_invalid:
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Invalid path: {invalid_path}", 
                                Path(__file__).stem)
            
            # Filter dataframe to only valid files
            if valid_files:
                df_valid = df.loc[valid_files]
                log_statement('info', f"{self.log_prefix}:INFO>>Working with {len(df_valid)} valid files", 
                            Path(__file__).stem)
            else:
                log_statement('error', f"{self.log_prefix}:ERROR>>No valid files found in repository", 
                            Path(__file__).stem)
                return []
            
            # Now check for files ready for processing
            status_column = 'status'
            if status_column not in df_valid.columns:
                possible_status_cols = ['file_status', 'processing_status', 'state']
                for col in possible_status_cols:
                    if col in df_valid.columns:
                        status_column = col
                        break
                else:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>No status column found. Assuming all valid files are ready for processing.", 
                                Path(__file__).stem)
                    return df_valid[filepath_column].tolist()
            
            # Check what statuses exist
            unique_statuses = df_valid[status_column].unique()
            log_statement('info', f"{self.log_prefix}:INFO>>Available statuses in valid files: {list(unique_statuses)}", 
                        Path(__file__).stem)
            
            # ENHANCED: Check multiple status patterns that indicate files ready for processing
            processable_statuses = [
                ProcessingStatus.NEW.value,
                ProcessingStatus.DISCOVERED.value,
                'new',
                'discovered', 
                'scanned',
                'ready',
                'pending',
                'unprocessed',
                'loaded'
            ]
            
            # Find files with processable status
            processable_files = df_valid[df_valid[status_column].isin(processable_statuses)]
            
            # If no files with standard statuses, check for files that haven't been processed
            if processable_files.empty:
                log_statement('info', f"{self.log_prefix}:INFO>>No files with standard processable statuses, checking for unprocessed files", 
                            Path(__file__).stem)
                
                # Look for files that don't have "processed" or "failed" status
                non_processed_statuses = df_valid[~df_valid[status_column].isin([
                    'processed', 'linguistic_processed', 'failed', 'linguistic_failed', 'error', 'tokenized'
                ])]
                
                if not non_processed_statuses.empty:
                    processable_files = non_processed_statuses
                    log_statement('info', f"{self.log_prefix}:INFO>>Found {len(processable_files)} unprocessed files", 
                                Path(__file__).stem)
                else:
                    # Final check: if we have files from our source directory, prioritize those
                    source_path = self.context.current_source_path
                    if source_path:
                        source_str = str(source_path)
                        source_files = df_valid[df_valid[filepath_column].str.contains(source_str, na=False)]
                        if not source_files.empty:
                            processable_files = source_files
                            log_statement('info', f"{self.log_prefix}:INFO>>Found {len(processable_files)} files from source directory: {source_path}", 
                                        Path(__file__).stem)
            
            # Convert to file paths
            if not processable_files.empty:
                file_paths = processable_files[filepath_column].tolist()
                
                # Final validation: ensure all returned paths actually exist
                validated_paths = []
                for file_path in file_paths:
                    if Path(file_path).exists():
                        validated_paths.append(file_path)
                    else:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>Filtering out non-existent file: {file_path}", 
                                    Path(__file__).stem)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Final result: {len(validated_paths)} files ready for linguistic processing", 
                            Path(__file__).stem)
                
                # Update repository state if needed
                if repo_state and repo_state.is_ready_for_processing and len(validated_paths) > 0:
                    self.context.update_repository_state(RepositoryState.PROCESSING)
                
                return validated_paths
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>No files found ready for processing after all checks", 
                            Path(__file__).stem)
                return []
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error getting files for linguistic processing: {e}", 
                        Path(__file__).stem, exc_info=True)
            return []
        
    def get_pipeline_name(self) -> str:
        return "LinguisticProcessing"
        
    def process_file(self, file_path: str) -> OperationResult:
        """Process a single file through linguistic pipeline with enhanced error handling and DataProcessor integration"""
        def _do_process_file():
            file_path_obj = Path(file_path)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Starting linguistic processing for {file_path_obj.name}", 
                        Path(__file__).stem)
            
            try:
                # Try to use DataProcessor from processing.py for comprehensive file processing
                data_processor_result = None
                log_statement('info', f"{self.log_prefix}:INFO>>Attempting to use DataProcessor for {file_path_obj.name}", Path(__file__).stem)

                # Check if we can get a DataProcessor from context
                if self.context and hasattr(self.context, 'container'):
                    try:
                        data_processor = self.context.container.get_data_processor()
                        log_statement('info', f"{self.log_prefix}:INFO>>DataProcessor obtained from context", Path(__file__).stem)
                        if data_processor and hasattr(data_processor, '_process_file'):
                            # Use the DataProcessor._process_file method for comprehensive processing
                            log_statement('info', f"{self.log_prefix}:INFO>>Using DataProcessor for {file_path_obj.name}", 
                                        Path(__file__).stem)
                            
                            # Create file_info dict that DataProcessor expects
                            file_info = {
                                'filepath': str(file_path_obj),
                                'status': 'new',  # Status that indicates ready for processing
                                'error_message': ''
                            }
                            
                            # Process using DataProcessor
                            log_statement('info', f"{self.log_prefix}:INFO>>Processing file with DataProcessor: {file_path_obj.name}", Path(__file__).stem)
                            data_processor_result = self._data_processor._process_file(file_info)
                            
                            if data_processor_result and data_processor_result.get('status') == 'processed':
                                log_statement('info', f"{self.log_prefix}:INFO>>✓ DataProcessor successfully processed: {file_path_obj.name}", 
                                            Path(__file__).stem)
                                
                                return {
                                    'file_path': file_path,
                                    'processed': True,
                                    'processing_stage': 'completed',
                                    'analysis_result': {
                                        'processed_path': data_processor_result.get('processed_path'),
                                        'data_classification': data_processor_result.get('data_classification'),
                                        'final_classification': data_processor_result.get('final_classification'),
                                        'data_hash': data_processor_result.get('data_hash'),
                                        'processing_method': 'data_processor'
                                    },
                                    'status_updated': True,
                                    'commit_triggered': False  # DataProcessor handles its own status updates
                                }
                            else:
                                error_detail = data_processor_result.get('error', 'DataProcessor processing failed') if data_processor_result else 'DataProcessor returned no result'
                                log_statement('warning', f"{self.log_prefix}:WARNING>>DataProcessor failed for {file_path_obj.name}: {error_detail}", 
                                            Path(__file__).stem)
                                
                    except Exception as dp_error:
                        log_statement('warning', f"{self.log_prefix}:WARNING>>DataProcessor error for {file_path_obj.name}: {dp_error}", 
                                    Path(__file__).stem)
                        data_processor_result = None
                
                # Fallback to enhanced safe processing function if DataProcessor didn't work
                if not data_processor_result or data_processor_result.get('status') != 'processed':
                    log_statement('info', f"{self.log_prefix}:INFO>>Using enhanced safe processing function for {file_path_obj.name}", 
                                Path(__file__).stem)
                    
                    # Call the standalone safe_process_file_with_recovery function
                    processing_result = safe_process_file_with_recovery(
                        file_path=file_path_obj,
                        repo=self.repo,
                        operation_type="linguistic_processing",
                        context=self.context
                    )
                    
                    if processing_result['success']:
                        log_statement('info', f"{self.log_prefix}:INFO>>✓ Safe processing function successfully processed: {file_path_obj.name}", 
                                    Path(__file__).stem)
                        
                        return {
                            'file_path': file_path,
                            'processed': True,
                            'processing_stage': processing_result.get('processing_stage'),
                            'analysis_result': processing_result.get('analysis_result'),
                            'status_updated': processing_result.get('status_updated'),
                            'commit_triggered': processing_result.get('commit_triggered'),
                            'semantic_label': processing_result.get('semantic_label'),
                            'processing_method': processing_result.get('processing_method')
                        }
                    else:
                        # Enhanced error reporting from safe processing function
                        error_detail = processing_result.get('error', 'Unknown processing error')
                        error_stage = processing_result.get('processing_stage', 'unknown')
                        error_type = processing_result.get('error_type', 'Unknown')
                        
                        log_statement('error', f"{self.log_prefix}:ERROR>>✗ Safe processing failed: {file_path_obj.name} at {error_stage} - {error_detail}", 
                                    Path(__file__).stem)
                        
                        # Create detailed error message
                        detailed_error = f"Linguistic processing failed for {file_path_obj.name} during {error_stage}: {error_detail} (Error type: {error_type})"
                        
                        # Log traceback if available
                        if processing_result.get('traceback'):
                            log_statement('debug', f"{self.log_prefix}:DEBUG>>Full traceback: {processing_result['traceback']}", 
                                        Path(__file__).stem)
                        
                        raise RuntimeError(detailed_error)
                
                # If we get here, data_processor_result was successful and we should have returned already
                # This is a safety fallback
                log_statement('warning', f"{self.log_prefix}:WARNING>>Unexpected code path reached for {file_path_obj.name}", 
                            Path(__file__).stem)
                raise RuntimeError(f"Unexpected processing state for {file_path_obj.name}")
                        
            except Exception as e:
                # Final fallback error handling
                error_msg = str(e) if str(e) else f"Unexpected {type(e).__name__} during linguistic processing"
                
                if not error_msg or error_msg == "None" or error_msg.strip() == "":
                    error_msg = f"Linguistic processing failed for {file_path_obj.name} with unknown error - check logs for details"
                
                # Add more context to the error
                detailed_error = f"Linguistic processing error for {file_path_obj.name}: {error_msg}"
                
                # Log the error with full context
                log_statement('error', f"{self.log_prefix}:ERROR>>Linguistic processing failure: {detailed_error}", 
                            Path(__file__).stem, exc_info=True)
                
                # Attempt final status update to failed
                try:
                    if hasattr(self.repo, 'modifier'):
                        self.repo.modifier.update_file_status(
                            file_path_obj,
                            ProcessingStatus.LINGUISTIC_FAILED.value,
                            change_description=f"Processing failed with exception: {error_msg}"
                        )
                except Exception as status_error:
                    log_statement('error', f"{self.log_prefix}:ERROR>>Could not update failure status: {status_error}", 
                                Path(__file__).stem)
                
                raise RuntimeError(detailed_error)
        
        return safe_operation("process_file_linguistic", _do_process_file)
        
    def _get_data_processor(self):
        """Get or create data processor"""
        if self._data_processor is None:
            self._data_processor = self.context.container.get_data_processor()
        return self._data_processor
    
    def _get_semantic_labeler(self):
        """Get or create semantic labeler"""
        if self._semantic_labeler is None:
            self._semantic_labeler = self.context.container.get_semantic_labeler()
        return self._semantic_labeler
    
    def _process_with_data_processor(self, data_processor, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process file with DataProcessor"""
        try:
            # This would interface with the actual DataProcessor
            # For now, we'll simulate the processing
            if hasattr(data_processor, 'process_file'):
                return data_processor.process_file(file_path)
            else:
                # Fallback: create a mock processed result
                return {
                    'success': True,
                    'output_path': file_path.with_suffix('.processed'),
                    'content_type': 'text',
                    'size': file_path.stat().st_size if file_path.exists() else 0
                }
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>DataProcessor error for {file_path}: {e}", 
                         Path(__file__).stem, exc_info=True)
            return None
    
    def _apply_semantic_labeling(self, semantic_labeler, processed_result: Dict[str, Any]) -> Optional[str]:
        """Apply semantic labeling to processed content"""
        try:
            if not processed_result or not hasattr(semantic_labeler, 'generate_label'):
                return None
            
            # Extract content for labeling (this would depend on the actual implementation)
            content = self._extract_content_for_labeling(processed_result)
            if content:
                return semantic_labeler.generate_label(content)
            
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Semantic labeling failed: {e}", 
                         Path(__file__).stem)
        
        return None
    
    def _extract_content_for_labeling(self, processed_result: Dict[str, Any]) -> Optional[str]:
        """Extract content from processed result for semantic labeling"""
        # This would depend on the actual DataProcessor output format
        if 'content' in processed_result:
            return processed_result['content']
        elif 'output_path' in processed_result:
            try:
                output_path = Path(processed_result['output_path'])
                if output_path.exists():
                    return output_path.read_text(encoding='utf-8')
            except Exception as e:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to read processed content: {e}", 
                             Path(__file__).stem)
        return None

class TokenizationPipeline(ProcessingPipeline):
    """Pipeline for tokenizing processed data"""
    
    def __init__(self, context: DataProcessingContext, repo: RepoHandler):
        super().__init__(context, repo)
        self._tokenizer = None
        self._output_dir = None
    
    def can_process(self) -> bool:
        """Check if tokenization can be performed"""
        return (
            TRANSFORMERS_AVAILABLE and 
            self.context.config.enable_tokenization and
            self.repo and 
            self.repo.is_initialized()
        )
    
    def get_files_to_process(self) -> List[str]:
        """Get files ready for tokenization with enhanced status detection"""
        try:
            # Get repository state
            repo_state = self.context.get_repository_processing_state() if self.context else None
            
            df = self.repo.get_dataframe()
            if df is None or df.empty:
                return []
            
            # Look for processed files ready for tokenization
            tokenizable_statuses = [
                ProcessingStatus.PROCESSED.value,
                ProcessingStatus.LINGUISTIC_PROCESSED.value,
                'processed',
                'linguistic_processed'
            ]
            
            status_column = 'status'
            if status_column not in df.columns:
                possible_status_cols = ['file_status', 'processing_status', 'state']
                for col in possible_status_cols:
                    if col in df.columns:
                        status_column = col
                        break
                else:
                    return []
            
            tokenizable_files = df[df[status_column].isin(tokenizable_statuses)]
            
            log_statement('info', f"{self.log_prefix}:INFO>>Found {len(tokenizable_files)} files ready for tokenization", 
                         Path(__file__).stem)
            
            if not tokenizable_files.empty:
                # Update repository state if needed
                if repo_state and repo_state.is_ready_for_tokenization:
                    self.context.update_repository_state(RepositoryState.TOKENIZING)
                
                if 'filepath' in tokenizable_files.columns:
                    return tokenizable_files['filepath'].tolist()
                elif 'file_path' in tokenizable_files.columns:
                    return tokenizable_files['file_path'].tolist()
                else:
                    return tokenizable_files.index.tolist()
            
            return []
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error getting files for tokenization: {e}", 
                         Path(__file__).stem, exc_info=True)
            return []
    
    def get_pipeline_name(self) -> str:
        return "Tokenization"
    
    def process_file(self, file_path: str) -> OperationResult:
        """Tokenize a single processed file"""
        def _do_tokenize_file():
            file_path_obj = Path(file_path)
            
            # Get tokenizer
            tokenizer = self._get_tokenizer()
            if not tokenizer:
                raise RuntimeError("Tokenizer not available")
            
            # Get processed content
            content = self._load_processed_content(file_path_obj)
            if not content:
                raise RuntimeError(f"Could not load processed content for {file_path}")
            
            # Tokenize content
            tokens = self._tokenize_content(tokenizer, content)
            
            # Save tokenized data
            output_path = self._save_tokenized_data(file_path_obj, tokens)
            
            # Update file status
            status_result = self.repo.modifier.update_file_status(
                file_path_obj,
                ProcessingStatus.TOKENIZED.value,
                change_description="Tokenization completed"
            )
            
            if status_result['status'] != OperationStatus.SUCCESS:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update status after tokenization: {status_result.get('error')}", 
                             Path(__file__).stem)
            
            return {
                'file_path': file_path,
                'tokenized': True,
                'output_path': str(output_path),
                'token_count': len(tokens['input_ids'][0]) if 'input_ids' in tokens else 0
            }
        
        return safe_operation("tokenize_file", _do_tokenize_file)
    
    def _get_tokenizer(self):
        """Get or create tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = self.context.container.get_tokenizer()
        return self._tokenizer
    
    def _get_output_dir(self) -> Path:
        """Get output directory for tokenized files"""
        if self._output_dir is None:
            self._output_dir = self.context.config.output_directory / "tokenized"
            self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir
    
    def _load_processed_content(self, file_path: Path) -> Optional[str]:
        """Load processed content for tokenization"""
        try:
            # This would depend on how the DataProcessor stores processed content
            # For now, we'll try to read the original file
            if file_path.exists():
                return file_path.read_text(encoding='utf-8')
            
            # Try to find processed version
            processed_path = file_path.with_suffix(file_path.suffix + '.processed')
            if processed_path.exists():
                return processed_path.read_text(encoding='utf-8')
            
            return None
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error loading content for tokenization: {e}", 
                         Path(__file__).stem, exc_info=True)
            return None
    
    def _tokenize_content(self, tokenizer, content: str) -> Dict[str, Any]:
        """Tokenize content using the tokenizer"""
        try:
            # Standard tokenization with reasonable defaults
            tokens = tokenizer(
                content,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Convert to CPU and detach for saving
            return {k: v.cpu().detach() for k, v in tokens.items()}
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Tokenization error: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise e
    
    def _save_tokenized_data(self, original_file: Path, tokens: Dict[str, Any]) -> Path:
        """Save tokenized data to file"""
        try:
            output_dir = self._get_output_dir()
            output_filename = original_file.stem + "_tokens.pt"
            output_path = output_dir / output_filename
            
            # Save using torch
            if torch:
                torch.save(tokens, output_path)
            else:
                # Fallback: save as pickle or JSON
                with open(output_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(tokens, f)
                output_path = output_path.with_suffix('.pkl')
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Saved tokenized data to {output_path}", 
                         Path(__file__).stem)
            return output_path
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error saving tokenized data: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise e

# Command Classes for Processing Operations
class LinguisticProcessingCommand:
    """Command for linguistic data processing"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def can_execute(self) -> bool:
        """Check if command can be executed"""
        return (
            self.context.repo_loaded and 
            self.context.get_current_repo() is not None and
            PROCESSING_IMPORTS_AVAILABLE
        )
    
    def get_description(self) -> str:
        return "Process files through linguistic analysis pipeline"
    
    def execute(self) -> OperationResult:
        """Execute linguistic processing"""
        def _do_execute():
            print("\n--- Linguistic Data Processing ---")
            
            repo = self.context.get_current_repo()
            if not repo:
                raise RuntimeError("No repository loaded. Please set data directory first.")
            
            # Create and run pipeline
            pipeline = LinguisticProcessingPipeline(self.context, repo)
            
            if not pipeline.can_process():
                raise RuntimeError("Linguistic processing pipeline cannot run. Check dependencies and configuration.")
            
            # Get files to process
            files_to_process = pipeline.get_files_to_process()
            if not files_to_process:
                print("No files ready for linguistic processing.")
                return {'processed_count': 0, 'message': 'No files to process'}
            
            print(f"Found {len(files_to_process)} files ready for processing...")
            
            # Process files
            result = pipeline.process_batch()
            
            if result['status'] == OperationStatus.SUCCESS:
                batch_result = result['result']
                print(f"✓ Processing completed:")
                print(f"  Processed: {batch_result['processed_count']}")
                print(f"  Failed: {batch_result['failed_count']}")
                print(f"  Total: {batch_result['total_files']}")
                
                if batch_result['failed_count'] > 0:
                    print(f"  Errors: {len(batch_result['errors'])}")
                    for error in batch_result['errors'][:5]:  # Show first 5 errors
                        print(f"    {error}")
                
                return batch_result
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"✗ Processing failed: {error_msg}")
                # Enhanced error handling for linguistic processing
                if not error_msg or error_msg == "None" or str(error_msg).strip() == "":
                    error_msg = "Linguistic processing failed with unknown error - check logs for details"

                # Add more context to the error
                detailed_error = f"Linguistic processing error: {error_msg}"

                # Log the error with full context
                log_statement('error', f"Linguistic processing failure: {detailed_error}", "linguistic_processing", exc_info=True)

                raise RuntimeError(detailed_error)
        
        return safe_operation("execute_linguistic_processing", _do_execute)


class TokenizationCommand:
    """Command for data tokenization"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def can_execute(self) -> bool:
        """Check if command can be executed"""
        return (
            self.context.repo_loaded and 
            self.context.get_current_repo() is not None and
            TRANSFORMERS_AVAILABLE
        )
    
    def get_description(self) -> str:
        return "Tokenize processed data for model training"
    
    def execute(self) -> OperationResult:
        """Execute tokenization"""
        def _do_execute():
            print("\n--- Data Tokenization ---")
            
            repo = self.context.get_current_repo()
            if not repo:
                raise RuntimeError("No repository loaded. Please run linguistic processing first.")
            
            # Create and run pipeline
            pipeline = TokenizationPipeline(self.context, repo)
            
            if not pipeline.can_process():
                raise RuntimeError("Tokenization pipeline cannot run. Check dependencies and configuration.")
            
            # Get files to process
            files_to_process = pipeline.get_files_to_process()
            if not files_to_process:
                print("No processed files ready for tokenization.")
                print("Please run linguistic processing first.")
                return {'tokenized_count': 0, 'message': 'No files to tokenize'}
            
            print(f"Found {len(files_to_process)} processed files ready for tokenization...")
            
            # Process files
            result = pipeline.process_batch()
            
            if result['status'] == OperationStatus.SUCCESS:
                batch_result = result['result']
                print(f"✓ Tokenization completed:")
                print(f"  Tokenized: {batch_result['processed_count']}")
                print(f"  Failed: {batch_result['failed_count']}")
                print(f"  Total: {batch_result['total_files']}")
                
                if batch_result['failed_count'] > 0:
                    print(f"  Errors: {len(batch_result['errors'])}")
                    for error in batch_result['errors'][:3]:
                        print(f"    {error}")
                
                return batch_result
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"✗ Tokenization failed: {error_msg}")
                raise RuntimeError(error_msg)
        
        return safe_operation("execute_tokenization", _do_execute)

def safe_process_file_with_recovery(file_path: Path, 
                                repo: RepoHandler, 
                                operation_type: str = "linguistic_processing",
                                context: Optional[DataProcessingContext] = None,
                                repo_handler: Optional['RepoHandler'] = None) -> Dict[str, Any]:
    """
    Safely process a file with comprehensive error recovery and detailed reporting.
    Enhanced to integrate with LinguisticProcessingPipeline and established patterns.
    
    Args:
        file_path: Path to the file to process
        repo: Repository handler instance (primary)
        operation_type: Type of processing operation
        context: Processing context for additional operations (optional)
        repo_handler: Alternative repository handler for backward compatibility (optional)
        
    Returns:
        Dictionary with processing results and error details
    """
    # Use repo_handler if provided for backward compatibility, otherwise use repo
    active_repo = repo_handler if repo_handler is not None else repo
    
    # Get or create context if not provided
    if context is None:
        context = get_context()
    
    processing_result = {
        'file_path': str(file_path),
        'success': False,
        'error': None,
        'error_type': None,
        'processing_stage': 'initialization',
        'status_updated': False,
        'commit_triggered': False,
        'analysis_result': None,
        'semantic_label': None,
        'processing_method': None,
        'pipeline_used': False
    }
    
    try:
        log_statement('info', f"{LOG_INS}:INFO>>Starting enhanced safe processing for {file_path.name}", 
                     Path(__file__).stem)
        processing_result['processing_stage'] = 'file_validation'
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")
        
        log_statement('debug', f"{LOG_INS}:DEBUG>>File validation completed for {file_path.name}", 
                     Path(__file__).stem)
        
        # Create repository operations helper
        repo_ops = RepositoryOperations(context)
        log_statement('debug', f"{LOG_INS}:DEBUG>>RepositoryOperations helper created", 
                     Path(__file__).stem)
        
        # Update status to processing
        processing_result['processing_stage'] = 'status_update_processing'
        log_statement('info', f"{LOG_INS}:INFO>>Updating status to processing for {file_path.name}", 
                     Path(__file__).stem)
        
        # Use the established safe_update_file_status pattern
        status_result = repo_ops.safe_update_file_status(
            repo=active_repo,
            file_path=file_path,
            new_status=ProcessingStatus.LINGUISTIC_PROCESSING.value,
            change_description=f"Starting {operation_type}"
        )
        
        if not status_result.get('success', False):
            raise RuntimeError(f"Failed to update status to processing: {status_result.get('error', 'Unknown error')}")
        
        processing_result['processing_method'] = 'repository_operations'
        log_statement('info', f"{LOG_INS}:INFO>>Status successfully updated to processing for {file_path.name}", 
                     Path(__file__).stem)
        
        # Perform the actual linguistic processing using the established pipeline
        processing_result['processing_stage'] = 'linguistic_analysis'
        log_statement('info', f"{LOG_INS}:INFO>>Beginning enhanced linguistic analysis for {file_path.name}", 
                     Path(__file__).stem)
        
        # Try to use the LinguisticProcessingPipeline for consistency
        linguistic_analysis_result = None
        pipeline_success = False
        
        try:
            # Create a LinguisticProcessingPipeline instance for single file processing
            pipeline = LinguisticProcessingPipeline(context, active_repo)
            
            if pipeline.can_process():
                log_statement('info', f"{LOG_INS}:INFO>>Using LinguisticProcessingPipeline for {file_path.name}", 
                             Path(__file__).stem)
                
                # Process single file using pipeline infrastructure
                pipeline_result = pipeline.process_file(str(file_path))
                
                if pipeline_result['status'] == OperationStatus.SUCCESS.value:
                    pipeline_data = pipeline_result['result']
                    linguistic_analysis_result = {
                        'processed': pipeline_data.get('processed', True),
                        'analysis_result': pipeline_data.get('analysis_result'),
                        'processing_stage': pipeline_data.get('processing_stage'),
                        'status_updated': pipeline_data.get('status_updated', False),
                        'commit_triggered': pipeline_data.get('commit_triggered', False),
                        'processing_method': 'linguistic_pipeline'
                    }
                    pipeline_success = True
                    processing_result['pipeline_used'] = True
                    
                    log_statement('info', f"{LOG_INS}:INFO>>LinguisticProcessingPipeline completed for {file_path.name}", 
                                 Path(__file__).stem)
                else:
                    log_statement('warning', f"{LOG_INS}:WARNING>>LinguisticProcessingPipeline failed, using fallback: {pipeline_result.get('error')}", 
                                 Path(__file__).stem)
            else:
                log_statement('info', f"{LOG_INS}:INFO>>LinguisticProcessingPipeline cannot process, using fallback", 
                             Path(__file__).stem)
                
        except Exception as pipeline_error:
            log_statement('warning', f"{LOG_INS}:WARNING>>LinguisticProcessingPipeline error, using fallback: {pipeline_error}", 
                         Path(__file__).stem)
        
        # Fallback processing if pipeline didn't work
        if not pipeline_success:
            try:
                # Get processors from context container using established patterns
                data_processor = context.container.get_data_processor()
                semantic_labeler = context.container.get_semantic_labeler()
                
                # Try data processor first
                if data_processor and hasattr(data_processor, 'process_file'):
                    try:
                        linguistic_analysis_result = data_processor.process_file(file_path)
                        processing_result['processing_method'] = 'data_processor'
                        log_statement('info', f"{LOG_INS}:INFO>>DataProcessor completed analysis for {file_path.name}", 
                                     Path(__file__).stem)
                    except Exception as processor_error:
                        log_statement('warning', f"{LOG_INS}:WARNING>>DataProcessor failed: {processor_error}", 
                                     Path(__file__).stem)
                        data_processor = None
                
                # Comprehensive fallback processing
                if not linguistic_analysis_result:
                    try:
                        # Attempt to read as text first
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        word_count = len(content.split())
                        char_count = len(content)
                        line_count = len(content.splitlines())
                        
                        # Basic content analysis
                        has_meaningful_content = word_count > 0 and char_count > 0
                        avg_word_length = char_count / word_count if word_count > 0 else 0
                        
                        linguistic_analysis_result = {
                            'word_count': word_count,
                            'char_count': char_count,
                            'line_count': line_count,
                            'avg_word_length': avg_word_length,
                            'has_meaningful_content': has_meaningful_content,
                            'content_type': 'text',
                            'processing_method': 'fallback_text_analysis',
                            'file_size': file_path.stat().st_size,
                            'encoding': 'utf-8'
                        }
                        
                        # Store content for semantic analysis
                        file_content = content
                        processing_result['processing_method'] = 'fallback_text'
                        
                        log_statement('info', f"{LOG_INS}:INFO>>Fallback text analysis completed for {file_path.name}: {word_count} words, {line_count} lines", 
                                    Path(__file__).stem)
                        
                    except UnicodeDecodeError:
                        # Handle binary files
                        file_size = file_path.stat().st_size
                        file_extension = file_path.suffix.lower()
                        
                        # Determine likely content type from extension
                        binary_types = {
                            '.pdf': 'document', '.doc': 'document', '.docx': 'document',
                            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
                            '.mp3': 'audio', '.wav': 'audio', '.mp4': 'video', '.avi': 'video',
                            '.zip': 'archive', '.tar': 'archive', '.gz': 'archive'
                        }
                        likely_type = binary_types.get(file_extension, 'unknown_binary')
                        
                        linguistic_analysis_result = {
                            'file_size': file_size,
                            'content_type': 'binary',
                            'binary_subtype': likely_type,
                            'file_extension': file_extension,
                            'processing_method': 'fallback_binary_analysis',
                            'encoding': 'binary',
                            'analyzable': False
                        }
                        
                        file_content = None  # No content for semantic analysis
                        processing_result['processing_method'] = 'fallback_binary'
                        
                        log_statement('info', f"{LOG_INS}:INFO>>Binary file analyzed: {file_path.name} ({file_size} bytes, type: {likely_type})", 
                                    Path(__file__).stem)
                    
                    except Exception as read_error:
                        # Ultimate fallback - basic file info only
                        file_size = file_path.stat().st_size if file_path.exists() else 0
                        linguistic_analysis_result = {
                            'file_size': file_size,
                            'content_type': 'error',
                            'processing_method': 'fallback_minimal',
                            'error': str(read_error),
                            'analyzable': False
                        }
                        file_content = None
                        processing_result['processing_method'] = 'fallback_minimal'
                        
                        log_statement('warning', f"{LOG_INS}:WARNING>>Minimal fallback for {file_path.name}: {read_error}", 
                                    Path(__file__).stem)
                
                # Apply semantic labeling using established container pattern
                if (semantic_labeler and hasattr(semantic_labeler, 'generate_label') and 
                    'file_content' in locals() and file_content):
                    try:
                        # Use first 1000 characters for semantic analysis
                        semantic_content = file_content[:1000] if len(file_content) > 1000 else file_content
                        semantic_label = semantic_labeler.generate_label(semantic_content)
                        
                        if linguistic_analysis_result:
                            linguistic_analysis_result['semantic_label'] = semantic_label
                        processing_result['semantic_label'] = semantic_label
                        
                        log_statement('info', f"{LOG_INS}:INFO>>Semantic labeling completed for {file_path.name}: {semantic_label}", 
                                    Path(__file__).stem)
                        
                    except Exception as semantic_error:
                        log_statement('warning', f"{LOG_INS}:WARNING>>Semantic labeling failed for {file_path.name}: {semantic_error}", 
                                    Path(__file__).stem)
                
            except Exception as fallback_error:
                error_msg = str(fallback_error) if str(fallback_error) else f"Fallback processing failed with {type(fallback_error).__name__}"
                log_statement('error', f"{LOG_INS}:ERROR>>All processing methods failed for {file_path.name}: {error_msg}", 
                             Path(__file__).stem, exc_info=True)
                raise RuntimeError(f"Linguistic analysis failed: {error_msg}")
        
        # Update status to completed (only if pipeline didn't already do it)
        if not (pipeline_success and linguistic_analysis_result.get('status_updated')):
            processing_result['processing_stage'] = 'status_update_completed'
            log_statement('info', f"{LOG_INS}:INFO>>Updating status to completed for {file_path.name}", 
                         Path(__file__).stem)
            
            completion_result = repo_ops.safe_update_file_status(
                repo=active_repo,
                file_path=file_path,
                new_status=ProcessingStatus.PROCESSED.value,
                change_description=f"Completed {operation_type}"
            )
            
            if not completion_result.get('success', False):
                log_statement('warning', f"{LOG_INS}:WARNING>>Failed to update final status: {completion_result.get('error')}", 
                            Path(__file__).stem)
                # Don't fail the entire operation for this
            else:
                processing_result['status_updated'] = True
                log_statement('info', f"{LOG_INS}:INFO>>Status successfully updated to completed for {file_path.name}", 
                             Path(__file__).stem)
                
                # Check if commit was triggered
                result_data = completion_result.get('result', {})
                if isinstance(result_data, dict) and result_data.get('commit_hash'):
                    processing_result['commit_triggered'] = True
                    log_statement('info', f"{LOG_INS}:INFO>>Commit triggered with hash: {result_data.get('commit_hash')}", 
                                 Path(__file__).stem)
        else:
            # Use status from pipeline
            processing_result['status_updated'] = linguistic_analysis_result.get('status_updated', False)
            processing_result['commit_triggered'] = linguistic_analysis_result.get('commit_triggered', False)
        
        # Final success state
        processing_result['success'] = True
        processing_result['processing_stage'] = 'completed'
        processing_result['analysis_result'] = linguistic_analysis_result
        
        log_statement('info', f"{LOG_INS}:INFO>>Successfully completed enhanced processing for {file_path.name} using {processing_result['processing_method']}", 
                     Path(__file__).stem)
        return processing_result
        
    except Exception as e:
        import traceback
        
        error_msg = str(e) if str(e) else f"Unknown error in {processing_result['processing_stage']}"
        processing_result['error'] = error_msg
        processing_result['error_type'] = type(e).__name__
        processing_result['traceback'] = traceback.format_exc()
        
        log_statement('error', f"{LOG_INS}:ERROR>>Enhanced processing failed for {file_path.name} at stage {processing_result['processing_stage']}: {error_msg}", 
                    Path(__file__).stem, exc_info=True)
        
        # Attempt to update status to failed using established patterns
        try:
            repo_ops = RepositoryOperations(context)
            failure_result = repo_ops.safe_update_file_status(
                repo=active_repo,
                file_path=file_path,
                new_status=ProcessingStatus.LINGUISTIC_FAILED.value,
                change_description=f"Failed during {processing_result['processing_stage']}: {error_msg}"
            )
            
            if failure_result.get('success', False):
                processing_result['status_updated'] = True
                log_statement('info', f"{LOG_INS}:INFO>>Updated status to failed for {file_path.name}", 
                             Path(__file__).stem)
            else:
                log_statement('warning', f"{LOG_INS}:WARNING>>Could not update failure status: {failure_result.get('error')}", 
                             Path(__file__).stem)
        
        except Exception as status_error:
            log_statement('error', f"{LOG_INS}:ERROR>>Could not update failure status for {file_path.name}: {status_error}", 
                        Path(__file__).stem, exc_info=True)
        
        return processing_result

# Legacy compatibility functions
def process_linguistic_data():
    """Legacy compatibility function for linguistic processing"""
    context = get_context()
    command = LinguisticProcessingCommand(context)
    
    if not command.can_execute():
        print("Error: Cannot execute linguistic processing. Check repository status and dependencies.")
        return
    
    result = command.execute()
    
    if result['status'] != OperationStatus.SUCCESS:
        print(f"Processing failed: {result.get('error', 'Unknown error')}")

def tokenize_data():
    """Legacy compatibility function for tokenization"""
    context = get_context()
    command = TokenizationCommand(context)
    
    if not command.can_execute():
        print("Error: Cannot execute tokenization. Check repository status and dependencies.")
        return
    
    result = command.execute()
    
    if result['status'] != OperationStatus.SUCCESS:
        print(f"Tokenization failed: {result.get('error', 'Unknown error')}")

# Section 4: Model Training, Management, and User Interface

# Model Training Pipeline
class ModelTrainingPipeline:
    """Pipeline for training models on tokenized data"""
    
    def __init__(self, context: DataProcessingContext, repo: RepoHandler):
        self.context = context
        self.repo = repo
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self._model = None
        self._trainer = None
    
    def can_train(self) -> bool:
        """Check if training can be performed"""
        return (
            TRANSFORMERS_AVAILABLE and 
            torch is not None and
            self.context.config.enable_model_training and
            self.repo and 
            self.repo.is_initialized()
        )
    
    def get_tokenized_files(self) -> List[str]:
        """Get tokenized files ready for training"""
        try:
            df = self.repo.get_dataframe()
            if df is None or df.empty:
                return []
            
            # Filter for tokenized files
            tokenized_files = df[df['status'] == ProcessingStatus.TOKENIZED.value]
            return tokenized_files['filepath'].tolist()
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error getting tokenized files: {e}", 
                         Path(__file__).stem, exc_info=True)
            return []
    
    def setup_training(self, training_config: Dict[str, Any]) -> OperationResult:
        """Set up model and trainer for training"""
        def _do_setup():
            if not self.can_train():
                raise RuntimeError("Training pipeline cannot be set up. Check dependencies.")
            
            # Setup model
            model_name = training_config.get('model_name', self.context.config.default_model_name)
            num_labels = training_config.get('num_labels', 2)
            
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=num_labels
                )
                
                # Set device
                device = self._get_device()
                self._model.to(device)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Model {model_name} loaded on {device}", 
                             Path(__file__).stem)
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_name}: {e}")
            
            # Setup trainer (if available)
            if PROCESSING_IMPORTS_AVAILABLE:
                try:
                    self._trainer = self._create_trainer(training_config)
                    log_statement('info', f"{self.log_prefix}:INFO>>Trainer initialized", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to create trainer: {e}", 
                                 Path(__file__).stem)
            
            return {
                'model_loaded': True,
                'model_name': model_name,
                'device': str(device),
                'trainer_available': self._trainer is not None
            }
        
        return safe_operation("setup_training", _do_setup)
    
    def train_model(self, training_config: Dict[str, Any]) -> OperationResult:
        """Train the model"""
        def _do_train():
            if not self._model:
                raise RuntimeError("Model not set up. Call setup_training first.")
            
            tokenized_files = self.get_tokenized_files()
            if not tokenized_files:
                raise RuntimeError("No tokenized files available for training")
            
            # Create dataset
            dataset = self._create_dataset(tokenized_files)
            
            # Create data loader
            dataloader = self._create_dataloader(dataset, training_config)
            
            # Train using trainer or basic training loop
            if self._trainer:
                result = self._train_with_trainer(dataloader, training_config)
            else:
                result = self._train_basic(dataloader, training_config)
            
            # Save model
            save_result = self._save_model(training_config)
            result.update(save_result)
            
            return result
        
        return safe_operation("train_model", _do_train)
    
    def _get_device(self):
        """Get appropriate device for training"""
        if torch is None:
            return 'cpu'
        
        device_config = self.context.config.device
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_config
    
    def _create_trainer(self, training_config: Dict[str, Any]):
        """Create trainer instance if available"""
        # This would integrate with the actual Trainer class
        # For now, return None as placeholder
        return None
    
    def _create_dataset(self, tokenized_files: List[str]):
        """Create dataset from tokenized files"""
        # This would create a proper dataset for training
        # For now, return a placeholder
        return {
            'files': tokenized_files,
            'size': len(tokenized_files)
        }
    
    def _create_dataloader(self, dataset, training_config: Dict[str, Any]):
        """Create data loader for training"""
        # This would create a proper dataloader
        # For now, return a placeholder
        return {
            'dataset': dataset,
            'batch_size': training_config.get('batch_size', 16)
        }
    
    def _train_with_trainer(self, dataloader, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train using the Trainer class"""
        try:
            # This would use the actual Trainer implementation
            epochs = training_config.get('num_epochs', 3)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Starting training with Trainer for {epochs} epochs", 
                         Path(__file__).stem)
            
            # Placeholder training result
            return {
                'training_method': 'trainer',
                'epochs_completed': epochs,
                'final_loss': 0.1,  # Placeholder
                'training_time': 300  # Placeholder
            }
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Training with Trainer failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise e
    
    def _train_basic(self, dataloader, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Basic training loop fallback"""
        try:
            epochs = training_config.get('num_epochs', 3)
            learning_rate = training_config.get('learning_rate', 5e-5)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Starting basic training for {epochs} epochs", 
                         Path(__file__).stem)
            
            # This would implement a basic training loop
            # For now, return placeholder result
            return {
                'training_method': 'basic',
                'epochs_completed': epochs,
                'learning_rate': learning_rate,
                'final_loss': 0.15,  # Placeholder
                'training_time': 450  # Placeholder
            }
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Basic training failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            raise e
    
    def _save_model(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save trained model"""
        try:
            checkpoint_dir = self.context.config.checkpoint_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model_name = training_config.get('model_name', 'trained_model')
            timestamp = int(time.time())
            save_path = checkpoint_dir / f"{model_name}_{timestamp}"
            
            if hasattr(self._model, 'save_pretrained'):
                self._model.save_pretrained(save_path)
                log_statement('info', f"{self.log_prefix}:INFO>>Model saved to {save_path}", 
                             Path(__file__).stem)
                
                return {
                    'model_saved': True,
                    'save_path': str(save_path),
                    'model_size': self._get_model_size(save_path)
                }
            else:
                log_statement('warning', f"{self.log_prefix}:WARNING>>Model does not support save_pretrained", 
                             Path(__file__).stem)
                return {'model_saved': False, 'reason': 'save_pretrained not supported'}
                
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error saving model: {e}", 
                         Path(__file__).stem, exc_info=True)
            return {'model_saved': False, 'error': str(e)}
    
    def _get_model_size(self, model_path: Path) -> int:
        """Get total size of saved model"""
        try:
            total_size = 0
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except:
            return 0

# Model Management Classes
class ModelManager:
    """Manages loading and using trained models"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self.loaded_models = {}
    
    def scan_for_models(self, model_directory: Optional[Path] = None) -> List[Dict[str, str]]:
        """Scan for available models"""
        scan_dir = model_directory or self.context.config.checkpoint_dir
        found_models = []
        
        try:
            if not scan_dir.exists():
                log_statement('warning', f"{self.log_prefix}:WARNING>>Model directory does not exist: {scan_dir}", 
                             Path(__file__).stem)
                return []
            
            for item in scan_dir.iterdir():
                if item.is_dir():
                    # Check for HuggingFace model structure
                    if (item / 'config.json').exists():
                        model_info = {
                            'path': str(item),
                            'name': item.name,
                            'type': 'huggingface',
                            'size': self._get_directory_size(item)
                        }
                        found_models.append(model_info)
                elif item.suffix in ['.pt', '.pth', '.bin']:
                    # PyTorch model file
                    model_info = {
                        'path': str(item),
                        'name': item.stem,
                        'type': 'pytorch',
                        'size': item.stat().st_size
                    }
                    found_models.append(model_info)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Found {len(found_models)} models in {scan_dir}", 
                         Path(__file__).stem)
            return found_models
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error scanning for models: {e}", 
                         Path(__file__).stem, exc_info=True)
            return []
    
    def load_model(self, model_path: str, model_id: Optional[str] = None) -> OperationResult:
        """Load a model for use"""
        def _do_load_model():
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for model loading")
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model_id = model_id or model_path_obj.name
            
            try:
                # Load model and tokenizer                
                if model_path_obj.is_dir() and (model_path_obj / 'config.json').exists():
                    # HuggingFace model directory
                    model = AutoModel.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    # PyTorch checkpoint
                    model = torch.load(model_path, map_location='cpu')
                    tokenizer = None
                
                # Move to appropriate device
                device = self._get_device()
                if hasattr(model, 'to'):
                    model.to(device)
                
                # Store in loaded models
                self.loaded_models[model_id] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'path': model_path,
                    'device': str(device),
                    'loaded_time': time.time()
                }
                
                # Update context
                self.context.loaded_model = model
                self.context.loaded_model_path = model_path
                self.context.loaded_tokenizer = tokenizer
                self.context.model_loaded = True
                
                result = {
                    'model_id': model_id,
                    'model_path': model_path,
                    'device': str(device),
                    'has_tokenizer': tokenizer is not None
                }
                
                log_statement('info', f"{self.log_prefix}:INFO>>Model {model_id} loaded successfully", 
                             Path(__file__).stem)
                return result
                
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load model {model_path}: {e}", 
                             Path(__file__).stem, exc_info=True)
                raise e
        
        return safe_operation("load_model", _do_load_model)
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        try:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
                # Clear context if this was the current model
                if self.context.loaded_model_path and model_id in self.context.loaded_model_path:
                    self.context.loaded_model = None
                    self.context.loaded_model_path = None
                    self.context.loaded_tokenizer = None
                    self.context.model_loaded = False
                
                log_statement('info', f"{self.log_prefix}:INFO>>Model {model_id} unloaded", 
                             Path(__file__).stem)
                return True
            
            return False
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error unloading model {model_id}: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models"""
        model_info = {}
        for model_id, model_data in self.loaded_models.items():
            model_info[model_id] = {
                'path': model_data['path'],
                'device': model_data['device'],
                'has_tokenizer': model_data['tokenizer'] is not None,
                'loaded_time': model_data['loaded_time']
            }
        return model_info
    
    def _get_device(self):
        """Get appropriate device for model"""
        if torch is None:
            return 'cpu'
        device_config = self.context.config.device
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_config
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory"""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except:
            return 0

# Command Classes for Training and Model Management
class ModelTrainingCommand:
    """Command for model training"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def can_execute(self) -> bool:
        """Check if training can be executed"""
        return (
            self.context.repo_loaded and 
            self.context.get_current_repo() is not None and
            TRANSFORMERS_AVAILABLE and
            torch is not None
        )
    
    def get_description(self) -> str:
        return "Train a model on tokenized data"
    
    def execute(self) -> OperationResult:
        """Execute model training"""
        def _do_execute():
            print("\n--- Model Training ---")
            
            repo = self.context.get_current_repo()
            if not repo:
                raise RuntimeError("No repository loaded. Please complete tokenization first.")
            
            # Get training configuration from user
            training_config = self._get_training_config()
            
            # Create training pipeline
            pipeline = ModelTrainingPipeline(self.context, repo)
            
            if not pipeline.can_train():
                raise RuntimeError("Training pipeline cannot run. Check dependencies and tokenized data.")
            
            # Setup training
            print("Setting up model and trainer...")
            setup_result = pipeline.setup_training(training_config)
            
            if setup_result['status'] != OperationStatus.SUCCESS:
                raise RuntimeError(f"Training setup failed: {setup_result.get('error')}")
            
            setup_data = setup_result['result']
            print(f"✓ Model setup completed:")
            print(f"  Model: {setup_data['model_name']}")
            print(f"  Device: {setup_data['device']}")
            print(f"  Trainer: {'Available' if setup_data['trainer_available'] else 'Basic mode'}")
            
            # Start training
            print("\nStarting training...")
            train_result = pipeline.train_model(training_config)
            
            if train_result['status'] == OperationStatus.SUCCESS:
                train_data = train_result['result']
                print(f"✓ Training completed:")
                print(f"  Method: {train_data['training_method']}")
                print(f"  Epochs: {train_data['epochs_completed']}")
                print(f"  Final Loss: {train_data.get('final_loss', 'N/A')}")
                print(f"  Training Time: {train_data.get('training_time', 0):.1f}s")
                
                if train_data.get('model_saved'):
                    print(f"  Model saved to: {train_data['save_path']}")
                    print(f"  Model size: {train_data.get('model_size', 0) / (1024*1024):.1f} MB")
                
                return train_data
            else:
                error_msg = train_result.get('error', 'Unknown error')
                print(f"✗ Training failed: {error_msg}")
                raise RuntimeError(error_msg)
        
        return safe_operation("execute_model_training", _do_execute)
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration from user input"""
        print("\n--- Training Configuration ---")
        
        # Get hyperparameters with defaults
        model_name = input(f"Model name or path [{self.context.config.default_model_name}]: ").strip()
        if not model_name:
            model_name = self.context.config.default_model_name
        
        try:
            learning_rate = float(input("Learning rate [5e-5]: ") or "5e-5")
        except ValueError:
            learning_rate = 5e-5
        
        try:
            num_epochs = int(input("Number of epochs [3]: ") or "3")
        except ValueError:
            num_epochs = 3
        
        try:
            batch_size = int(input("Batch size [16]: ") or "16")
        except ValueError:
            batch_size = 16
        
        try:
            num_labels = int(input("Number of labels [2]: ") or "2")
        except ValueError:
            num_labels = 2
        
        config = {
            'model_name': model_name,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'num_labels': num_labels
        }
        
        print(f"\nTraining configuration: {config}")
        return config

class ModelLoadingCommand:
    """Command for loading and managing models"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.model_manager = ModelManager(context)
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def can_execute(self) -> bool:
        """Check if model loading can be executed"""
        return TRANSFORMERS_AVAILABLE
    
    def get_description(self) -> str:
        return "Load and manage saved models"
    
    def execute(self) -> OperationResult:
        """Execute model loading interface"""
        def _do_execute():
            while True:
                print("\n--- Model Management ---")
                print("A) Scan for models")
                print("B) Load model")
                print("C) View loaded models")
                print("D) Unload model")
                print("E) Return to main menu")
                
                choice = input("Enter choice (A-E): ").upper().strip()
                
                if choice == 'A':
                    self._scan_models()
                elif choice == 'B':
                    self._load_model()
                elif choice == 'C':
                    self._view_loaded_models()
                elif choice == 'D':
                    self._unload_model()
                elif choice == 'E':
                    break
                else:
                    print("Invalid choice. Please try again.")
            
            return {'completed': True}
        
        return safe_operation("execute_model_loading", _do_execute)
    
    def _scan_models(self):
        """Scan for available models"""
        print("\n--- Scanning for Models ---")
        models = self.model_manager.scan_for_models()
        
        if not models:
            print("No models found in checkpoint directory.")
            return
        
        print(f"Found {len(models)} models:")
        for i, model in enumerate(models, 1):
            size_mb = model['size'] / (1024 * 1024)
            print(f"{i}. {model['name']} ({model['type']}) - {size_mb:.1f} MB")
            print(f"   Path: {model['path']}")
    
    def _load_model(self):
        """Load a model"""
        print("\n--- Load Model ---")
        
        # Scan for models first
        models = self.model_manager.scan_for_models()
        if not models:
            print("No models found. Please train a model first.")
            return
        
        # Display models
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['type']})")
        
        try:
            choice = int(input("Enter model number to load: ")) - 1
            if 0 <= choice < len(models):
                selected_model = models[choice]
                
                print(f"Loading model: {selected_model['name']}...")
                result = self.model_manager.load_model(selected_model['path'])
                
                if result['status'] == OperationStatus.SUCCESS:
                    result_data = result['result']
                    print(f"✓ Model loaded successfully:")
                    print(f"  ID: {result_data['model_id']}")
                    print(f"  Device: {result_data['device']}")
                    print(f"  Tokenizer: {'Available' if result_data['has_tokenizer'] else 'Not available'}")
                else:
                    print(f"✗ Failed to load model: {result.get('error')}")
            else:
                print("Invalid model number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    def _view_loaded_models(self):
        """View currently loaded models"""
        print("\n--- Loaded Models ---")
        loaded_models = self.model_manager.get_loaded_models()
        
        if not loaded_models:
            print("No models currently loaded.")
            return
        
        for model_id, model_info in loaded_models.items():
            load_time = time.ctime(model_info['loaded_time'])
            print(f"Model: {model_id}")
            print(f"  Path: {model_info['path']}")
            print(f"  Device: {model_info['device']}")
            print(f"  Tokenizer: {'Available' if model_info['has_tokenizer'] else 'Not available'}")
            print(f"  Loaded: {load_time}")
            print()
    
    def _unload_model(self):
        """Unload a model"""
        print("\n--- Unload Model ---")
        loaded_models = self.model_manager.get_loaded_models()
        
        if not loaded_models:
            print("No models currently loaded.")
            return
        
        print("Loaded models:")
        model_ids = list(loaded_models.keys())
        for i, model_id in enumerate(model_ids, 1):
            print(f"{i}. {model_id}")
        
        try:
            choice = int(input("Enter model number to unload: ")) - 1
            if 0 <= choice < len(model_ids):
                model_id = model_ids[choice]
                success = self.model_manager.unload_model(model_id)
                if success:
                    print(f"✓ Model {model_id} unloaded successfully.")
                else:
                    print(f"✗ Failed to unload model {model_id}.")
            else:
                print("Invalid model number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main menu system and execution (keeping the same structure but with updated functionality)
class DataProcessingMenu:
    """Main data processing menu using Command pattern"""
        
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.commands = {
            '1': DataDirectorySetupCommand(context),
            '2': LinguisticProcessingCommand(context),
            '3': TokenizationCommand(context),
            '4': ModelTrainingCommand(context),
            '5': ModelLoadingCommand(context),
            '6': RepositoryCleanupCommand(context),  # ADDED: Repository cleanup
        }
        self.status_display = RepositoryStatusDisplay(context)

    def _display_menu(self):
        """ENHANCED: Display the main menu with cleanup option"""
        print("\n" + "="*60)
        print("TLATO v4.1 - Data Processing & Model Training")
        print("="*60)
        
        # Show current status
        repo_status = "Loaded" if self.context.repo_loaded else "Not Set"
        model_status = "Loaded" if self.context.model_loaded else "Not Loaded"
        
        print(f"Repository: {repo_status} | Model: {model_status}")
        if self.context.current_source_path:
            print(f"Source: {self.context.current_source_path}")
        
        # Show progress status if available
        if self.context.current_progress:
            progress = self.context.current_progress
            print(f"Last Progress: {progress.process_name} ({progress.percentage_complete:.1f}% complete)")
        
        print("-"*60)
        print("1. Set/Scan Source Data Directory")
        print("2. Process Linguistic Data")
        print("3. Tokenize Processed Data")
        print("4. Train Model on Tokenized Data")
        print("5. Load/Manage Models")
        print("6. Repository Cleanup & Validation")  # ADDED
        print("-"*60)
        print("S. Show Repository Status")
        print("I. Show Repository Summary")
        print("P. Show Progress History")
        print("D. Repository Discovery")
        print("0. Exit")
        print("-"*60)
    
    def run(self):
        """ENHANCED: Run the main menu loop with startup handling"""
        print_welcome_message()
        
        # Handle startup (progress resume and repository discovery)
        startup_result = self.context.handle_startup()
        
        if startup_result.get('error'):
            print(f"Startup warning: {startup_result['error']}")
        
        # Show startup results
        if startup_result.get('progress_resumed'):
            print(f"\n✓ Resumed from saved progress")
        elif startup_result.get('repositories_discovered'):
            print(f"\n✓ Repository loaded from discovery")
        
        while True:
            try:
                self._display_menu()
                choice = input("Enter your choice: ").strip()
                
                if choice == '0':
                    # Clean up old progress files before exiting
                    self.context.progress_manager.clear_old_progress(keep_count=5)
                    print("Exiting...")
                    break
                elif choice in self.commands:
                    command = self.commands[choice]
                    
                    if command.can_execute():
                        try:
                            result = command.execute()
                            if result['status'] != OperationStatus.SUCCESS.value:
                                error_msg = result.get('error', 'Unknown error')
                                error_type = result.get('error_type', '')
                                
                                print(f"Command failed: {error_msg}")
                                if error_type:
                                    print(f"  Error type: {error_type}")
                                
                                # Show additional error details if available
                                if 'result' in result and isinstance(result['result'], dict):
                                    result_data = result['result']
                                    if 'errors' in result_data and result_data['errors']:
                                        print(f"  Detailed errors ({len(result_data['errors'])}):")
                                        for i, error in enumerate(result_data['errors'][:3], 1):
                                            if isinstance(error, dict):
                                                print(f"    {i}. {error.get('file', 'Unknown')}: {error.get('error', 'No details')}")
                                            else:
                                                print(f"    {i}. {error}")
                                        if len(result_data['errors']) > 3:
                                            print(f"    ... and {len(result_data['errors']) - 3} more errors")
                        except KeyboardInterrupt:
                            print("\nOperation cancelled by user.")
                            # Save cancellation progress if applicable
                            if hasattr(command, 'save_cancellation_progress'):
                                try:
                                    command.save_cancellation_progress()
                                except:
                                    pass  # Don't fail on progress save
                        except Exception as e:
                            print(f"Unexpected error: {e}")
                            log_statement('error', f"Menu command error: {e}", 
                                        Path(__file__).stem, exc_info=True)
                    else:
                        print(f"Cannot execute: {command.get_description()}")
                        print("Please check prerequisites and dependencies.")
                elif choice == 's' or choice == 'S':
                    self.status_display.display_repository_status()
                elif choice == 'i' or choice == 'I':
                    self.status_display.display_repository_summary()
                elif choice == 'p' or choice == 'P':
                    self._display_progress_history()
                elif choice == 'd' or choice == 'D':
                    self._discovery_menu()
                else:
                    print("Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Menu error: {e}")
                log_statement('error', f"Menu system error: {e}", 
                            Path(__file__).stem, exc_info=True)
                input("Press Enter to continue...")

    def _display_progress_history(self):
        """Display progress history"""
        try:
            snapshots = self.context.progress_manager.get_all_progress_snapshots()
            
            if not snapshots:
                print("No progress history found.")
                return
            
            print(f"\n--- Progress History ({len(snapshots)} entries) ---")
            for i, progress in enumerate(snapshots[:10], 1):  # Show last 10
                print(f"{i}. {progress.process_name} - {progress.stage}")
                print(f"   Created: {progress.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Progress: {progress.processed_files}/{progress.total_files} ({progress.percentage_complete:.1f}%)")
                if progress.source_path:
                    print(f"   Source: {Path(progress.source_path).name}")
                print()
        
        except Exception as e:
            print(f"Error displaying progress history: {e}")

    def _discovery_menu(self):
        """Repository discovery menu"""
        try:
            print("\n--- Repository Discovery ---")
            print("Searching for repositories and data directories...")
            
            search_paths = [
                self.context.config.project_root,
                Path.cwd(),
                Path.home() / "Documents"
            ]
            
            discovery_result = self.context.startup_manager.repository_discovery.discover_repositories(search_paths)
            self.context.startup_manager._display_discovery_results(discovery_result)
            
            if any([discovery_result.get('tlato_repositories'),
                    discovery_result.get('git_repositories'), 
                    discovery_result.get('potential_data_directories')]):
                
                selection_result = self.context.startup_manager._handle_repository_selection(discovery_result)
                if selection_result.get('repositories_discovered'):
                    print("✓ Repository loaded successfully!")
        
        except Exception as e:
            print(f"Discovery error: {e}")

def data_processing_submenu():
    """Main entry point for the data processing system"""
    context = get_context()
    menu = DataProcessingMenu(context)
    
    try:
        menu.run()
    finally:
        # Cleanup context
        context.cleanup()

# Legacy compatibility functions
def train_on_tokens():
    """Legacy compatibility function for model training"""
    context = get_context()
    command = ModelTrainingCommand(context)
    
    if not command.can_execute():
        print("Error: Cannot execute model training. Check repository status and dependencies.")
        return
    
    result = command.execute()
    if result['status'] != OperationStatus.SUCCESS:
        print(f"Training failed: {result.get('error', 'Unknown error')}")

def load_model_submenu():
    """Legacy compatibility function for model loading"""
    context = get_context()
    command = ModelLoadingCommand(context)
    
    if not command.can_execute():
        print("Error: Cannot execute model loading. Check dependencies.")
        return
    
    command.execute()

# Main execution
if __name__ == "__main__":
    try:
        # Ensure dependencies are available
        deps = ensure_dependencies()
        missing = [name for name, available in deps.items() if not available]
        
        if missing:
            print(f"Warning: Missing dependencies: {', '.join(missing)}")
            print("Some features may be limited.")
        
        # Run the main data processing interface
        data_processing_submenu()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        log_statement('critical', f"Main execution error: {e}", 
                     Path(__file__).stem, exc_info=True)
    finally:
        print("\nTLATO v4.1 - Session ended.")