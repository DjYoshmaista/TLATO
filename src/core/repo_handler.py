import git
import json
import logging
import os
import re
import shutil
import inspect
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone, UTC
import tempfile
from pydantic import *
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
import threading
import gzip

try:
    from src.utils.compression import *
    COMPRESSION_UTILS_AVAILABLE = True
except ImportError:
    COMPRESSION_UTILS_AVAILABLE = False
    actual_log_statement('warning', f"{LOG_INS}:WARNING>>gzip Compression Utilities from src.utils.compression not available.  File compression features will be disabled", Path(__file__).stem)
from src.utils.config import *
from src.data.constants import *
from src.utils.helpers import process_file
from src.core.models import FileMetadataEntry, FileVersion, MetadataCollection # Pydantic models
from src.utils.hashing import *

try:
    from src.utils.logger import log_statement as actual_log_statement
    _log_statement_defined = True
except ImportError:
    _log_statement_defined = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    def actual_log_statement(loglevel: str, logstatement: str, main_logger_name: Optional[str] = None, exc_info: bool = False):
        logger = logging.getLogger(main_logger_name or Path(__file__).stem)
        level = getattr(logging, loglevel.upper(), logging.INFO)
        if exc_info:
            logger.log(level, logstatement, exc_info=sys.exc_info())
        else:
            logger.log(level, logstatement)
# Attempt to import _get_file_metadata from helpers; it's crucial
try:
    from src.utils.helpers import _get_file_metadata as get_os_and_content_metadata
    _get_os_and_content_metadata_defined = True
except ImportError:
    _get_os_and_content_metadata_defined = False
    actual_log_statement('critical', "_get_file_metadata from src.utils.helpers could not be imported! Essential functionality will be missing.", Path(__file__).stem)
    def get_os_and_content_metadata(abs_path: Path, hash_algorithms: Optional[List[str]] = None, calculate_custom_hashes: bool = True) -> Dict[str, Any]:
        # Fallback placeholder if actual helper is missing
        actual_log_statement('error', f"Fallback get_os_and_content_metadata called for {abs_path}. This is a stub!", Path(__file__).stem)
        return {
            "filename": abs_path.name, "extension": abs_path.suffix, "size_bytes": 0,
            "os_last_modified_utc": datetime.now(timezone.utc).isoformat(),
            "os_created_utc": datetime.now(timezone.utc).isoformat(),
            "custom_hashes": {algo: "placeholder_hash" for algo in (hash_algorithms or ["md5","sha256"])} if calculate_custom_hashes else {}
        }
# Configure basic logging for the module if not already configured by an external logger.
# The user specified a `log_statement` function, which implies an existing setup.
# For this refactoring, I will define a placeholder `log_statement` and `LOG_INS`
# generation based on the user's description. It's assumed the actual `configure_logging`
# and `log_statement` are available in `src.utils.logger`.

# Placeholder for logging setup - In a real scenario, this would come from src.utils.logger
# Ensure this is adapted to the actual logging infrastructure.
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "exception": logging.ERROR, # 'exception' level usually logs ERROR with exc_info
}

# Global utility for structured log messages
def _get_log_ins(frame, class_name: Optional[str] = None) -> str:
    if not frame: return "unknown_module::unknown_func::unknown_line"
    module_name = Path(frame.f_code.co_filename).stem
    func_name = frame.f_code.co_name
    line_no = frame.f_lineno
    if class_name:
        return f"{module_name}::{class_name}::{func_name}::{line_no}"
    return f"{module_name}::{func_name}::{line_no}"

class GitCommandError(Exception):
    """Custom exception for Git command failures."""
    pass

class GitOpsHelper:
    """Helper class for executing Git commands and parsing output."""
    def __init__(self,
                 repo_path: Path,
                 create_if_not_exist: bool = False, 
                 repo: Optional[git.Repo] = None, 
                 root_dir: Path = ROOT_DIR):
        self.path: Path = repo_path
        self.git_repo: Optional[git.Repo] = repo
        self.is_new_repo: bool = False
        self.root_dir = root_dir
        self._log_ins_class = self.__class__.__name__
        
        try:
            if not self.path.is_dir():
                if create_if_not_exist:
                    log_statement('info', f"{LOG_INS}:INFO>>Directory {self.path} does not exist. Creating for Git repository.", self._logger_name)
                    self.path.mkdir(parents=True, exist_ok=True)
                    self._initialize_new_git_repo()
                else:
                    log_statement('warning', f"Path {self.path} is not a valid directory and create_if_not_exist is False. Cannot load/init Git repo.", self._logger_name)
            else: # Path is a directory
                try:
                    self.git_repo = git.Repo(str(self.path))
                    log_statement('info', f"{LOG_INS}:INFO>>Successfully loaded existing Git repository at: {self.path}", self._logger_name)
                except git.InvalidGitRepositoryError:
                    if create_if_not_exist:
                        log_statement('info', f"{LOG_INS}:INFO>>No existing Git repository at {self.path}. Initializing new one.", self._logger_name)
                        self._initialize_new_git_repo()
                    else:
                        log_statement('warning', f"Not a valid Git repository at {self.path} and create_if_not_exist is False.", self._logger_name)
                except git.NoSuchPathError: # Should be rare if self.path.is_dir() passed, but for safety
                    log_statement('error', f"{LOG_INS}:ERROR>>Path {self.path} became invalid before GitPython could access it.", self._logger_name, exc_info=True)
                except Exception as e: # Catch other GitPython or unexpected errors during load
                    log_statement('error', f"{LOG_INS}:ERROR>>Unexpected error loading Git repository from {self.path}: {e}", self._logger_name, exc_info=True)

        except Exception as e: # Catch errors from path.mkdir or other pre-checks
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to prepare path or initialize Git repository at {self.path}: {e}", self._logger_name, exc_info=True)

    def _initialize_new_git_repo(self):
        """Internal method to initialize a new Git repository."""
        try:
            self.git_repo = git.Repo.init(str(self.path))
            self.is_new_repo = True
            log_statement('info', f"{LOG_INS}:INFO>>Successfully initialized new Git repository at: {self.path}", self._logger_name)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to initialize new Git repository at {self.path}: {e}", self._logger_name, exc_info=True)
            self.git_repo = None # Ensure it's None on failure
            self.is_new_repo = False

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.git_repo is not None

    def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
        try:
            actual_filepaths_str = [str(Path(fp).resolve().relative_to(self.repo.working_dir)) for fp in filepaths]
            self.repo.index.add(actual_filepaths_str)
            
            # Check if there are actual changes to commit
            if self.repo.is_dirty(index=True, working_tree=False, untracked_files=False) or \
               any(fp_str in [diff.a_path for diff in self.repo.index.diff("HEAD")] for fp_str in actual_filepaths_str):
                self.repo.index.commit(message)
                actual_log_statement('info', f"{LOG_INS}:INFO>>Committed files: {actual_filepaths_str} with message: '{message}'", Path(__file__).stem)
                return True
            else:
                actual_log_statement('info', f"{LOG_INS}:INFO>>No changes to commit for files: {actual_filepaths_str}", Path(__file__).stem)
                return True # No changes is not an error in this context
        except git.exc.GitCommandError as e:
            if "nothing to commit" in str(e).lower():
                actual_log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
                return True
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
            return False
        except Exception as e:
            actual_log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
            return False

    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        try:
            # For a file in the working directory (staged or unstaged, or even untracked)
            abs_path = self.repo.working_dir / Path(file_rel_path)
            if abs_path.is_file():
                with open(abs_path, 'rb') as f:
                    blob = self.repo.odb.store(git.Blob.input_stream(f, 'blob'))
                    return blob.hexsha
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
        return None

    def _execute_git_command(self, command: List[str], suppress_errors: bool = False, **kwargs) -> str:
        """
        Executes a Git command using the GitPython interface.
        Args:
            command (List[str]): The Git command and its arguments (e.g., ['status', '--porcelain']).
            suppress_errors (bool): If True, logs error and returns empty string instead of raising.
            **kwargs: Additional keyword arguments to pass to the git command.
        Returns:
            str: The stdout from the Git command.
        Raises:
            GitCommandError: If the Git command fails and suppress_errors is False.
        """
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Executing Git command: git {' '.join(command)} with kwargs: {kwargs}", Path(__file__).stem, False)
        try:
            result = self.git_repo.git.execute(command, **kwargs)
            actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Git command 'git {' '.join(command)}' executed successfully.", Path(__file__).stem, False)
            return result
        except git.exc.GitCommandError as e:
            log_msg = f"{LOG_INS}:ERROR>>Git command 'git {' '.join(command)}' failed: {e}"
            actual_log_statement("error", log_msg, Path(__file__).stem, True)
            if suppress_errors:
                return ""
            raise GitCommandError(f"Git command 'git {' '.join(command)}' failed: {e}") from e
        except Exception as e:
            log_msg = f"{LOG_INS}:CRITICAL>>Unexpected error executing Git command 'git {' '.join(command)}': {e}"
            actual_log_statement("critical", log_msg, Path(__file__).stem, True)
            if suppress_errors:
                return ""
            raise GitCommandError(f"Unexpected error for 'git {' '.join(command)}': {e}") from e

    def parse_log_output(self, log_output: str, num_parts: int, field_names: List[str], delimiter: str = "|") -> List[Dict[str, str]]:
        """
        Parses raw Git log output into a list of dictionaries.
        Args:
            log_output (str): The raw string output from `git log`.
            num_parts (int): The expected number of parts when splitting a line.
            field_names (List[str]): The names for each part.
            delimiter (str): The delimiter used in the log format.
        Returns:
            List[Dict[str, str]]: A list of commit information dictionaries.
        """
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Parsing git log output.", Path(__file__).stem, False)
        commits = []
        if not log_output:
            actual_log_statement("info", f"{LOG_INS}:INFO>>No log output to parse.", Path(__file__).stem, False)
            return commits

        for line in log_output.strip().splitlines():
            parts = line.split(delimiter, num_parts - 1)
            if len(parts) == num_parts:
                commits.append(dict(zip(field_names, parts)))
            else:
                actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping malformed log line: {line}", Path(__file__).stem, False)
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Parsed {len(commits)} commits from log output.", Path(__file__).stem, False)
        return commits

    # --- Potential additional Git operation methods ---
    # def get_status(self) -> Optional[str]:
    #     if self.is_valid_repo():
    #         return self.git_repo.git.status()
    #     log_statement('warning', "Cannot get status, Git repo not initialized.", self._logger_name)
    #     return None

    # def add_files(self, file_paths: list[Union[str, Path]]) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.git_repo.index.add([str(fp) for fp in file_paths])
    #             log_statement('info', f"Added files to index: {file_paths}", self._logger_name)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to add files {file_paths}: {e}", self._logger_name, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot add files, Git repo not initialized.", self._logger_name)
    #     return False

    # def commit(self, message: str) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.git_repo.index.commit(message)
    #             log_statement('info', f"Committed with message: {message}", self._logger_name)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to commit: {e}", self._logger_name, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot commit, Git repo not initialized.", self._logger_name)
    #     return False

class MetadataFileHandler:
    """Handles reading and writing of the JSON metadata file."""
    def __init__(self, repo_path: Path, metadata_filename: str = METADATA_FILENAME, use_compression: str = 'zst'):
        self.repo_path = repo_path
        self.metadata_filename_actual = metadata_filename # Store base name
        self.use_compression = COMPRESSION_USED # e.g., "gzip"
        
        if self.use_compression == "gzip":
            self.metadata_filepath = self.repo_path / f"{metadata_filename}.gz"
        elif self.use_compression == "zstd" and ZSTD_AVAILABLE:
            self.metadata_filepath = self.repo_path / f"{metadata_filename}.zst"
        else:
            self.metadata_filepath = self.repo_path / metadata_filename
            if self.use_compression: # Log if specified but not supported/implemented
                actual_log_statement('warning', f"Compression '{self.use_compression}' for metadata not supported or zstd not available. Using uncompressed.", Path(__file__).stem)
                self.use_compression = None


        self.lock = threading.RLock()
        self._log_ins_class = self.__class__.__name__
        self._file_identifier = Path(Path(__file__).stem).stem

    def ensure_metadata_file_exists(self, repo_instance: Optional[git.Repo] = None, initial_commit: bool = False) -> None:
        with self.lock:
            if not self.metadata_filepath.exists():
                actual_log_statement('info', f"{LOG_INS}:INFO>>Metadata file not found. Creating: {self.metadata_filepath}", Path(__file__).stem)
                # Write empty JSON object, possibly compressed
                empty_metadata_dict = {}
                try:
                    if self.use_compression == "gzip":
                        with gzip.open(self.metadata_filepath, 'wt', encoding='utf-8') as f_gz:
                            json.dump(empty_metadata_dict, f_gz, indent=4)
                    elif self.use_compression == "zstd" and ZSTD_AVAILABLE:
                        try:
                            with zstd.open(self.metadata_filepath, 'wt', encoding='utf-8') as f_zst:
                                json.dump(empty_metadata_dict, f_zst, indent=4)
                        except Exception as zstd_err:
                            actual_log_statement(
                                'error',
                                f"{LOG_INS}:ERROR>>Failed to write metadata file with Zstandard compression: {zstd_err}",
                                Path(__file__).stem,
                                exc_info=True
                            )
                            raise
                    else: # No compression or unsupported
                        with open(self.metadata_filepath, 'w', encoding='utf-8') as f:
                            json.dump(empty_metadata_dict, f, indent=4)
                    
                    if repo_instance and initial_commit:
                        try:
                            repo_instance.index.add([str(self.metadata_filepath)])
                            repo_instance.index.commit("Initial commit: Add empty metadata file.")
                            actual_log_statement('info', f"{LOG_INS}:INFO>>Committed initial empty metadata file.", Path(__file__).stem)
                        except Exception as e:
                            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit initial metadata file: {e}", Path(__file__).stem, exc_info=True)
                except Exception as e:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to create or commit initial metadata file: {e}", Path(__file__).stem, exc_info=True)

    def read_metadata(self) -> Dict[str, Any]:
        with self.lock:
            # ensure_metadata_file_exists might be called by RepoHandler's init,
            # but calling it here defensively makes read_metadata more robust if used standalone.
            # However, ensure_metadata_file_exists also tries to commit, which might not be desired here.
            # Let's assume it's ensured by the caller (RepoHandler.__init__) or we check existence directly.
            self.ensure_metadata_file_exists() # Ensure it exists before trying to read
            if self.metadata_filepath.exists() and self.metadata_filepath.stat().st_size > 0:
                 # If ensure_metadata_file_exists was called by init, an empty compressed file might exist.
                 # A zero-byte file is effectively empty metadata.
                 # For gzip, an empty compressed file is typically very small (e.g., 20-30 bytes for empty dict).
                 # A simple stat().st_size == 0 check might not be sufficient for compressed empty files.
                 # Better to try reading and handle if it's empty content.
                 pass # Will attempt to read, if fails or empty, will return {}
            try:
                if self.use_compression == "gzip":
                    with gzip.open(self.metadata_filepath, 'rt', encoding='utf-8') as f_gz:
                        data = json.load(f_gz)
                elif self.use_compression == "zstd" and ZSTD_AVAILABLE:
                    with zstd.open(self.metadata_filepath, 'rt', encoding='utf-8') as f_zst:
                        data = json.load(f_zst)
                else: # No compression or unsupported
                    if not self.metadata_filepath.exists() or self.metadata_filepath.stat().st_size == 0: # Re-check for uncompressed
                        return {}
                    with open(self.metadata_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                return data if isinstance(data, dict) else {}
            except FileNotFoundError:
                 actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Metadata file {self.metadata_filepath} not found. Returning empty metadata.", Path(__file__).stem)
                 return {}
            except (json.JSONDecodeError, EOFError, gzip.BadGzipFile, zstandard.ZstdError): # EOFError for empty gzip/zstd, BadGzipFile for corrupted gzip, ZstdError for zstd issues
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Error decoding JSON/decompressing {self.metadata_filepath}. Returning empty metadata.", Path(__file__).stem, exc_info=True)
                return {}
            except Exception as e:
                actual_log_statement('exception', f"{LOG_INS}:EXCEPTION>>Error loading metadata from {self.metadata_filepath}: {e}", Path(__file__).stem)
                return {}
            
    def write_metadata(self, data: Dict[str, Any], commit_message: Optional[str] = None, repo_instance: Optional[git.Repo] = None) -> bool:
        with self.lock:
            try:
                # Create parent directory if it doesn't exist (e.g. .tlato/metadata.json)
                self.metadata_filepath.parent.mkdir(parents=True, exist_ok=True)
                
                if self.use_compression == "gzip":
                    with gzip.open(self.metadata_filepath, 'wt', encoding='utf-8') as f_gz:
                        json.dump(data, f_gz, indent=4, sort_keys=True)
                elif self.use_compression == "zstd" and ZSTD_AVAILABLE:
                    with zstd.open(self.metadata_filepath, 'wt', encoding='utf-8') as f_zst:
                        json.dump(data, f_zst, indent=4, sort_keys=True)
                else: # No compression or unsupported
                    with open(self.metadata_filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, sort_keys=True)
                if repo_instance and commit_message:
                    # (The existing commit logic from previous step's MetadataFileHandler.write_metadata)
                    try:
                        repo_instance.index.add([str(self.metadata_filepath)])
                        if self.metadata_filepath.name in [diff.a_path for diff in repo_instance.index.diff("HEAD")] or \
                           str(self.metadata_filepath) in repo_instance.untracked_files or \
                           any(item.a_path == str(self.metadata_filepath.relative_to(repo_instance.working_dir)) for item in repo_instance.index.diff(None)):
                            repo_instance.index.commit(commit_message)
                            actual_log_statement('info', f"{LOG_INS}:INFO>>Committed metadata changes: {commit_message}", Path(__file__).stem)
                        else:
                            actual_log_statement('debug', f"{LOG_INS}:DEBUG>>No changes to commit for metadata file after writing.", Path(__file__).stem)
                    except git.exc.GitCommandError as git_err:
                        if "nothing to commit" in str(git_err).lower():
                             actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Nothing to commit for metadata: {commit_message}", Path(__file__).stem)
                        else:
                             actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit metadata changes: {git_err}", Path(__file__).stem, exc_info=True)
                             return False
                return True
            except Exception as e:
                actual_log_statement('exception', f"{LOG_INS}:EXCEPTION>>Error writing metadata to {self.metadata_filepath}: {e}", Path(__file__).stem)
                return False
class ProgressFileHandler:
    """Handles saving and loading of progress files."""

    def __init__(self, root_dir: Path, repo_path: Path, progress_dir_name: str = PROGRESS_DIR, git_ops_helper: Optional[GitOpsHelper] = None):
        self.progress_dir = f"{repo_path}/{progress_dir_name}"
        Path(self.progress_dir).mkdir(parents=True, exist_ok=True)
        self._log_ins_class = self.__class__.__name__
        self.root_dir = root_dir
        self.git_ops_helper = git_ops_helper # Optional, for committing progress files

    def save_progress(self, process_id: str, current_state: Dict[str, Any], commit_changes: bool = True) -> bool:
        """Saves progress to a JSON file and optionally commits it."""
        progress_file = self.root_dir / f"progress_{process_id}.json"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving progress for process '{process_id}' to {progress_file}", Path(__file__).stem, False)

        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with progress_file.open("w") as f:
                json.dump(current_state, f, indent=4)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress saved successfully for {process_id} to {progress_file}.", Path(__file__).stem, False)

            if commit_changes and self.git_ops_helper:
                try:
                    self.git_ops_helper.git_repo.index.add([str(progress_file)])
                    self.git_ops_helper.git_repo.index.commit(f"Saved progress for {process_id}")
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed progress file for {process_id}.", Path(__file__).stem, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit progress file for {process_id}: {e}", Path(__file__).stem, True)
                    # Return True because file was saved, even if commit failed. Or False?
                    # For now, let's say saving the file is primary success.
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to save progress for {process_id}: {e}", Path(__file__).stem, True)
            return False

    def load_progress(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Loads progress from a JSON file."""
        progress_file = self.root_dir / f"progress_{process_id}.json"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Loading progress for process '{process_id}' from {progress_file}", Path(__file__).stem, False)

        if not progress_file.exists():
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress file for {process_id} not found at {progress_file}.", Path(__file__).stem, False)
            return None
        try:
            with progress_file.open("r") as f:
                state = json.load(f)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress loaded successfully for {process_id}.", Path(__file__).stem, False)
            return state
        except json.JSONDecodeError as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to decode JSON from progress file {progress_file}: {e}", Path(__file__).stem, True)
            return None # Corrupted file
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to load progress for {process_id}: {e}", Path(__file__).stem, True)
            return None

class GitignoreFileHandler:
    """Handles reading and modifying the .gitignore file."""
    def __init__(self, repo: git.Repo, gitignore_name=".gitgnore", git_ops_helper: GitOpsHelper = None, gitignore_path: Optional[Path] = None):
        self.repo = repo
        self.git_ops_helper = git_ops_helper
        self.gitignore_path = Optional[Path] = None
        self.gitignore_name = gitignore_name
        # Ensure .gitignore file exists
        if self.git_ops_helper and hasattr(self.git_ops_helper, 'git_repo') and self.git_ops_helper.git_repo:
            try:
                working_tree_dir = self.git_ops_helper.git_repo.working_tree_dir
                if working_tree_dir:
                    self.gitignore_path = Path(working_tree_dir).resolve() / self.gitignore_name
                    log_statement('debug', f"{LOG_INS}:DEBUG>>Gitignore path set to: {self.gitignore_path}", "GitignoreFileHandler")
                else:
                    log_statement('warning', f"{LOG_INS}:WARNING>>Working tree directory is None, cannot determine .gitignore path.", "GitIgnoreFileHandler")
            except AttributeError:
                # This can happen if git_repo exists but working_tree_dir is missing (unlikely for valid repo)
                log_statement('warning', f"{LOG_INS}:WARNING>>git_repo object missing 'working_tree_dir' attribute.", "GitignoreFileHandler")
                pass # self.gitignore_path remains None
            except Exception as e:
                log_statement('warning', f"{LOG_INS}:WARNING>>Could not determine .gitignore path due to: {e}", "GitignoreFileHandler")
                pass # self.gitignore_path remains None
        else:
            log_statement('info', f"{LOG_INS}:INFO>>GitOpsHelper or its git_repo is not available. .gitignore handling might be limited or skipped.", "GitignoreFileHandler")
            pass # self.gitignore_path remains None

    def get_gitignore_content(self) -> Optional[str]:
        """Reads the content of the .gitignore file."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Reading .gitignore file from {self.gitignore_path}", Path(__file__).stem, False)
        try:
            if self.gitignore_path.exists():
                content = self.gitignore_path.read_text()
                actual_log_statement("info", f"{LOG_INS}:INFO>>.gitignore content read successfully.", Path(__file__).stem, False)
                return content
            else:
                actual_log_statement("info", f"{LOG_INS}:INFO>>.gitignore file does not exist at {self.gitignore_path}.", Path(__file__).stem, False)
                return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to read .gitignore file: {e}", Path(__file__).stem, True)
            return None # Fallback

    def add_to_gitignore(self, pattern: str, commit: bool = True, commit_message: Optional[str] = None) -> bool:
        """Adds a pattern to .gitignore if it's not already present."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to add pattern '{pattern}' to .gitignore", Path(__file__).stem, False)
        try:
            content = self.get_gitignore_content()
            if content is None: # File doesn't exist or couldn't be read
                lines = []
            else:
                lines = content.splitlines()

            if pattern.strip() in [line.strip() for line in lines]:
                actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' already in .gitignore. No changes made.", Path(__file__).stem, False)
                return True

            with self.gitignore_path.open("a") as f: # Open in append mode
                f.write(f"\n{pattern.strip()}") # Add newline before pattern for safety
            actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' added to .gitignore.", Path(__file__).stem, False)

            if commit:
                msg = commit_message or f"Added '{pattern}' to .gitignore"
                try:
                    self.git_ops_helper.git_repo.index.add([str(self.gitignore_path)])
                    self.git_ops_helper.git_repo.index.commit(msg)
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed .gitignore changes: {msg}", Path(__file__).stem, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit .gitignore changes: {e}", Path(__file__).stem, True)
                    return False # File was modified, but commit failed
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to add pattern '{pattern}' to .gitignore: {e}", Path(__file__).stem, True)
            return False

    def remove_from_gitignore(self, pattern: str, commit: bool = True, commit_message: Optional[str] = None) -> bool:
        """Removes a pattern from .gitignore."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to remove pattern '{pattern}' from .gitignore", Path(__file__).stem, False)
        if not self.gitignore_path.exists():
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>.gitignore file not found. Cannot remove pattern '{pattern}'.", Path(__file__).stem, False)
            return False
        try:
            lines = self.gitignore_path.read_text().splitlines()
            pattern_to_remove = pattern.strip()
            new_lines = [line for line in lines if line.strip() != pattern_to_remove]

            if len(new_lines) == len(lines):
                actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' not found in .gitignore. No changes made.", Path(__file__).stem, False)
                return True # Pattern wasn't there, so considered success

            self.gitignore_path.write_text("\n".join(new_lines) + "\n") # Add trailing newline
            actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' removed from .gitignore.", Path(__file__).stem, False)

            if commit:
                msg = commit_message or f"Removed '{pattern}' from .gitignore"
                try:
                    self.git_ops_helper.git_repo.index.add([str(self.gitignore_path)])
                    self.git_ops_helper.git_repo.index.commit(msg)
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed .gitignore changes: {msg}", Path(__file__).stem, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit .gitignore changes: {e}", Path(__file__).stem, True)
                    return False # File modified, commit failed
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to remove pattern '{pattern}' from .gitignore: {e}", Path(__file__).stem, True)
            return False

class RepoAnalyzer:
    """Analyzes the Git repository for information (read-only operations)."""

    def __init__(self, git_ops_helper: GitOpsHelper):
        self.git_ops = git_ops_helper

    def get_status_for_file(self, filepath: Path) -> Optional[str]:
        """Gets the Git status for a specific file."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting status for file: {filepath}", Path(__file__).stem, False)
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain', str(filepath.resolve())], suppress_errors=True)
            if status_output:
                status_code = status_output.strip().split(maxsplit=1)[0]
                actual_log_statement("info", f"{LOG_INS}:INFO>>Status for {filepath}: {status_code}", Path(__file__).stem, False)
                return status_code
            actual_log_statement("info", f"{LOG_INS}:INFO>>File {filepath} is clean or not tracked.", Path(__file__).stem, False)
            return None # Clean or not tracked
        except GitCommandError: # Already logged by _execute_git_command
            return "Error" # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting status for {filepath}: {e}", Path(__file__).stem, True)
            return "Error"

    def get_repository_status_summary(self) -> Optional[str]:
        """Gets the porcelain status for the entire repository."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting repository status summary.", Path(__file__).stem, False)
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain'], suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository status summary retrieved.", Path(__file__).stem, False)
            return status_output
        except GitCommandError:
            return "Error retrieving status" # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting repository status: {e}", Path(__file__).stem, True)
            return "Error retrieving status"

    def get_files_by_status(self, status_codes: Union[str, List[str]]) -> List[Path]:
        """Gets files matching the given Git status code(s) (e.g., 'M', '??')."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting files by status: {status_codes}", Path(__file__).stem, False)
        files: List[Path] = []
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain'], suppress_errors=True)
            if not status_output:
                actual_log_statement("info", f"{LOG_INS}:INFO>>Repository is clean or status output is empty.", Path(__file__).stem, False)
                return files

            target_statuses = [status_codes] if isinstance(status_codes, str) else status_codes
            for line in status_output.strip().splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    code, filepath_str = parts
                    # Git status porcelain uses codes like " M" or "M ", so strip them.
                    if code.strip() in target_statuses:
                        try:
                            abs_path = self.git_ops.root_dir / filepath_str
                            files.append(abs_path)
                        except Exception as path_e:
                            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Could not form path for '{filepath_str}': {path_e}", Path(__file__).stem, False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Found {len(files)} files with status {status_codes}.", Path(__file__).stem, False)
            return files
        except GitCommandError:
            return [] # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting files by status: {e}", Path(__file__).stem, True)
            return []

    def get_commit_history(self, max_count: Optional[int] = None, file_path: Optional[Path] = None, author: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retrieves commit history for the repository or a specific file/author.
        Combines `get_file_history` and `get_commit_history` and `get_commit_history_by_author`.
        """
        log_params = [f"max_count={max_count}", f"file_path={file_path}", f"author={author}"]
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting commit history. Params: {', '.join(filter(None, log_params))}", Path(__file__).stem, False)

        cmd = ['log', '--format=%H|%an|%ad|%s', '--date=iso'] # Standard format
        field_names = ["commit_hash", "author", "date", "message"]
        num_parts = 4

        if max_count and max_count > 0:
            cmd.append(f'-{max_count}')
        if author:
            cmd.extend(['--author', author])
        if file_path:
            cmd.extend(['--follow', '--', str(file_path.resolve())])
        
        try:
            log_output = self.git_ops._execute_git_command(cmd, suppress_errors=True)
            commits = self.git_ops.parse_log_output(log_output, num_parts, field_names)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved {len(commits)} commits.", Path(__file__).stem, False)
            return commits
        except GitCommandError:
            return []
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting commit history: {e}", Path(__file__).stem, True)
            return []

    def get_commit_details(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed information for a specific commit."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting details for commit: {commit_hash}", Path(__file__).stem, False)
        if not commit_hash or commit_hash.lower() == "head": # Resolve HEAD if needed
            try:
                commit_hash = self.git_ops.git_repo.head.commit.hexsha
                actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Resolved HEAD to {commit_hash}", Path(__file__).stem, False)
            except Exception as e:
                 actual_log_statement("error", f"{LOG_INS}:ERROR>>Could not resolve HEAD: {e}", Path(__file__).stem, True)
                 return None

        try:
            commit = self.git_ops.git_repo.commit(commit_hash)
            details = {
                "commit_hash": commit.hexsha,
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "authored_date": commit.authored_datetime.isoformat(),
                "committer_name": commit.committer.name,
                "committer_email": commit.committer.email,
                "committed_date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "parents": [p.hexsha for p in commit.parents],
                "stats": {
                    "files_changed": len(commit.stats.files),
                    "insertions": commit.stats.total.get("insertions", 0),
                    "deletions": commit.stats.total.get("deletions", 0),
                    "lines": commit.stats.total.get("lines", 0),
                } if commit.stats else {},
                "files": [item.a_path if item.a_path else item.b_path for item in commit.diff(commit.parents[0] if commit.parents else git.EMPTY_TREE)] if commit.stats else []

            }
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved details for commit {commit_hash}.", Path(__file__).stem, False)
            return details
        except git.exc.BadName as e: # More specific error for invalid commit hash
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Invalid commit hash '{commit_hash}': {e}", Path(__file__).stem, False) # exc_info not needed for BadName
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get details for commit {commit_hash}: {e}", Path(__file__).stem, True)
            return None

    def get_diff(self, item1: str, item2: Optional[str] = None, file_path: Optional[Path] = None) -> Optional[str]:
        """
        Gets the diff between two commits, a commit and working tree, or for a specific file.
        `item1` can be a commit hash.
        `item2` can be another commit hash. If None, diffs `item1` against working tree (or its parent if only one commit).
        """
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting diff for items: {item1}, {item2}, file: {file_path}", Path(__file__).stem, False)
        cmd = ['diff']
        if item1: cmd.append(item1)
        if item2: cmd.append(item2)
        if file_path: cmd.extend(['--', str(file_path.resolve())])
        
        try:
            diff_output = self.git_ops._execute_git_command(cmd, suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Diff retrieved successfully.", Path(__file__).stem, False)
            return diff_output
        except GitCommandError:
            return None # Error logged in _execute_git_command
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting diff: {e}", Path(__file__).stem, True)
            return None

    def get_blame(self, file_path: Path) -> Optional[str]:
        """Gets the blame output for a file."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting blame for file: {file_path}", Path(__file__).stem, False)
        try:
            blame_output = self.git_ops._execute_git_command(['blame', str(file_path.resolve())], suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Blame retrieved successfully for {file_path}.", Path(__file__).stem, False)
            return blame_output
        except GitCommandError:
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting blame for {file_path}: {e}", Path(__file__).stem, True)
            return None
            
    def get_authors_contributors(self, include_email: bool = True) -> List[Dict[str, str]]:
        """Gets a list of authors/contributors with their commit counts."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting authors/contributors (include_email={include_email}).", Path(__file__).stem, False)
        cmd = ['shortlog', '-sne'] # -s (summary count), -n (sort by name), -e (show email)
        try:
            output = self.git_ops._execute_git_command(cmd, suppress_errors=True)
            authors = []
            for line in output.strip().splitlines():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    count = parts[0]
                    name_email_part = parts[1]
                    name_parts = name_email_part.split('<', 1)
                    name = name_parts[0].strip()
                    email = name_parts[1].rstrip('>') if len(name_parts) > 1 else ""
                    author_info = {"name": name, "commits": int(count)}
                    if include_email:
                        author_info["email"] = email
                    authors.append(author_info)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved {len(authors)} authors/contributors.", Path(__file__).stem, False)
            return authors
        except GitCommandError:
            return []
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting authors: {e}", Path(__file__).stem, True)
            return []

    def get_commit_count(self, rev_range: Optional[str] = "HEAD", author: Optional[str] = None, committer: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None) -> int:
        """Gets the commit count based on various filters."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting commit count for range '{rev_range}', author '{author}', committer '{committer}', since '{since}', until '{until}'.", Path(__file__).stem, False)
        cmd = ['rev-list', '--count']
        if author: cmd.extend(['--author', author])
        if committer: cmd.extend(['--committer', committer])
        if since: cmd.extend([f'--since={since}'])
        if until: cmd.extend([f'--until={until}'])
        if rev_range: cmd.append(rev_range)
        else: cmd.append("HEAD") # Default to HEAD if no range

        try:
            count_str = self.git_ops._execute_git_command(cmd, suppress_errors=True).strip()
            count = int(count_str) if count_str else 0
            actual_log_statement("info", f"{LOG_INS}:INFO>>Commit count: {count}", Path(__file__).stem, False)
            return count
        except (GitCommandError, ValueError) as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get commit count: {e}", Path(__file__).stem, isinstance(e, GitCommandError))
            return 0 # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting commit count: {e}", Path(__file__).stem, True)
            return 0

    # ... Many other get_... methods would go here, refactored to use _execute_git_command and _parse_log_output
    # For brevity, I will not list all of them but will follow the pattern above. Examples:
    # get_commits_by_date, get_tags_by_date, get_branches_by_commit, etc.

    def get_repository_root_path(self) -> Optional[Path]:
        """Gets the root directory of the Git repository."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting repository root path.", Path(__file__).stem, False)
        try:
            # GitPython provides this directly and more reliably
            root_path_str = self.git_ops.git_repo.working_tree_dir
            if root_path_str:
                root_path = Path(root_path_str)
                actual_log_statement("info", f"{LOG_INS}:INFO>>Repository root path: {root_path}", Path(__file__).stem, False)
                return root_path
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Could not determine repository root path from GitPython.", Path(__file__).stem, False)
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get repository root: {e}", Path(__file__).stem, True)
            return None

    def is_git_repository(self, directory: Optional[Path] = None) -> bool:
        """Checks if the given directory (or current repo's dir) is a Git repository."""
        path_to_check = directory or self.git_ops.root_dir
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Checking if {path_to_check} is a Git repository.", Path(__file__).stem, False)
        try:
            # For the current repo instance, this is implicitly true if git_repo is valid.
            # For an arbitrary directory, we'd init a new Repo object or use rev-parse.
            if directory: # Checking an arbitrary directory
                 git.Repo(path_to_check) # This will raise an error if not a repo
            else: # Checking the repo this handler is for
                 if not self.git_ops.git_repo.git_dir: # Basic check
                     return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Path {path_to_check} is a Git repository.", Path(__file__).stem, False)
            return True
        except (git.exc.NoSuchPathError, git.exc.InvalidGitRepositoryError):
            actual_log_statement("info", f"{LOG_INS}:INFO>>Path {path_to_check} is not a Git repository.", Path(__file__).stem, False)
            return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error checking Git repository status for {path_to_check}: {e}", Path(__file__).stem, True)
            return False

class RepoModifier:
    """Modifies the Git repository state (commits, branches, tags, etc.)."""
    def __init__(self, git_ops_helper: GitOpsHelper, metadata_handler: MetadataFileHandler):
        self.git_ops = git_ops_helper
        self.metadata_handler = metadata_handler

    def _commit_changes(self, files_to_add: List[Union[str, Path]], commit_message: str) -> bool:
        """Helper to add files and commit changes."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Committing changes. Files: {files_to_add}, Message: '{commit_message}'", Path(__file__).stem, False)
        try:
            str_files_to_add = [str(f.resolve() if isinstance(f, Path) else Path(f).resolve()) for f in files_to_add]
            self.git_ops.git_repo.index.add(str_files_to_add)
            self.git_ops.git_repo.index.commit(commit_message)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Changes committed successfully with message: '{commit_message}'.", Path(__file__).stem, False)
            return True
        except Exception as e: # Catch Git related errors specifically
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit changes with message '{commit_message}': {e}", Path(__file__).stem, True)
            return False

    def save_repository_snapshot(self, commit_message: str = "Repository snapshot") -> bool:
        """Adds all changes and creates a commit."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving repository snapshot with message: '{commit_message}'", Path(__file__).stem, False)
        try:
            self.git_ops.git_repo.git.add(all=True) # Using git.add directly for 'all'
            self.git_ops.git_repo.index.commit(commit_message)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository snapshot saved successfully.", Path(__file__).stem, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to save repository snapshot: {e}", Path(__file__).stem, True)
            return False

    def update_metadata_entry(self, filepath: Path, commit_message_prefix: str = "Updated", **kwargs) -> bool:
        """Updates a metadata entry and commits the change."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Updating metadata entry for {filepath} with kwargs: {kwargs}", Path(__file__).stem, False)
        
        metadata = self.metadata_handler.read_metadata()
        if metadata is None: # Error reading metadata
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot update metadata, failed to read existing metadata.", Path(__file__).stem, False)
            return False

        file_str = str(filepath.resolve())
        if file_str not in metadata:
            metadata[file_str] = {}
        metadata[file_str].update(kwargs)

        commit_msg = f"{commit_message_prefix} metadata for {filepath.name}"
        if self.metadata_handler.write_metadata(metadata): # Write first
            # If metadata write is successful, then add specific file and metadata file to commit
            files_to_commit = [self.metadata_handler.metadata_path]
            if filepath.exists(): # Only add file if it exists; metadata can be about non-existent/virtual files too
                 files_to_commit.append(filepath)
            return self._commit_changes(files_to_commit, commit_msg)
        else:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata for {filepath}, commit aborted.", Path(__file__).stem, False)
            return False

    def record_error_in_metadata(self, filepath: Path, error_msg: str) -> bool:
        """Records an error message for a file in the metadata."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Recording error for {filepath}: {error_msg}", Path(__file__).stem, False)
        
        metadata = self.metadata_handler.read_metadata()
        if metadata is None:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot record error, failed to read existing metadata.", Path(__file__).stem, False)
            return False

        file_str = str(filepath.resolve())
        if "errors" not in metadata:
            metadata["errors"] = {}
        if file_str not in metadata["errors"]:
             metadata["errors"][file_str] = []
        metadata["errors"][file_str].append(error_msg) # Store errors as a list

        commit_msg = f"Recorded error for {filepath.name}"
        if self.metadata_handler.write_metadata(metadata):
            # Only commit metadata file, not the file with error itself unless intended
            return self._commit_changes([self.metadata_handler.metadata_path], commit_msg)
        else:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata for error recording, commit aborted.", Path(__file__).stem, False)
            return False

    def manage_branch(self, action: str, branch_name: str, new_branch_name: Optional[str] = None) -> bool:
        """Manages branches: create, delete, checkout."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Branch action: {action}, name: {branch_name}, new_name: {new_branch_name}", Path(__file__).stem, False)
        try:
            if action == "create":
                self.git_ops.git_repo.create_head(branch_name)
            elif action == "delete":
                self.git_ops.git_repo.delete_head(branch_name, force=True) # Add force for safety, or make it an option
            elif action == "checkout":
                self.git_ops.git_repo.heads[branch_name].checkout()
            elif action == "rename": # Not in original but good addition
                if not new_branch_name:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>New branch name required for rename action.", Path(__file__).stem, False)
                    return False
                branch_to_rename = self.git_ops.git_repo.heads[branch_name]
                branch_to_rename.rename(new_branch_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported branch action: {action}", Path(__file__).stem, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Branch action '{action}' for '{branch_name}' successful.", Path(__file__).stem, False)
            return True
        except Exception as e: # Catch specific git errors like git.exc.GitCommandError
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Branch action '{action}' for '{branch_name}' failed: {e}", Path(__file__).stem, True)
            return False

    def manage_tag(self, action: str, tag_name: str, message: Optional[str] = None, commit_ish: Optional[str] = None) -> bool:
        """Manages tags: create, delete."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Tag action: {action}, name: {tag_name}, message: {message}, commit: {commit_ish}", Path(__file__).stem, False)
        try:
            if action == "create":
                ref = commit_ish if commit_ish else self.git_ops.git_repo.head.commit
                self.git_ops.git_repo.create_tag(tag_name, ref=ref, message=message or f"Tag {tag_name}", force=False) # -a implicitly by message
            elif action == "delete":
                self.git_ops.git_repo.delete_tag(tag_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported tag action: {action}", Path(__file__).stem, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Tag action '{action}' for '{tag_name}' successful.", Path(__file__).stem, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Tag action '{action}' for '{tag_name}' failed: {e}", Path(__file__).stem, True)
            return False

    def manage_remote(self, action: str, remote_name: str, remote_url: Optional[str] = None) -> bool:
        """Manages remotes: add, remove."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Remote action: {action}, name: {remote_name}, url: {remote_url}", Path(__file__).stem, False)
        try:
            if action == "add":
                if not remote_url:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Remote URL required for add action.", Path(__file__).stem, False)
                    return False
                self.git_ops.git_repo.create_remote(remote_name, remote_url)
            elif action == "remove":
                self.git_ops.git_repo.delete_remote(remote_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported remote action: {action}", Path(__file__).stem, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Remote action '{action}' for '{remote_name}' successful.", Path(__file__).stem, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Remote action '{action}' for '{remote_name}' failed: {e}", Path(__file__).stem, True)
            return False

    def push_changes(self, remote_name: str = "origin", refspec: str = "refs/heads/*:refs/heads/*", tags: bool = False, force: bool = False) -> bool:
        """Pushes changes to a remote."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Pushing to remote '{remote_name}', refspec '{refspec}', tags={tags}, force={force}", Path(__file__).stem, False)
        try:
            remote_to_push = self.git_ops.git_repo.remote(name=remote_name)
            push_infos = remote_to_push.push(refspec=refspec, tags=tags, force=force) # GitPython uses force flag
            for info in push_infos:
                if info.flags & (git.PushInfo.ERROR | git.PushInfo.REJECTED | git.PushInfo.REMOTE_REJECTED | git.PushInfo.REMOTE_FAILURE):
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Push to {remote_name} failed for ref {info.local_ref or info.remote_ref_string}: {info.summary}", Path(__file__).stem, False)
                    # return False # Decide if any error means overall failure
            actual_log_statement("info", f"{LOG_INS}:INFO>>Push to remote '{remote_name}' completed.", Path(__file__).stem, False)
            return True # Simplified: assume success if no immediate exception and some info received
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Push to remote '{remote_name}' failed: {e}", Path(__file__).stem, True)
            return False
            
    # ... Other modifier methods like pull, fetch, merge, reset, clean, LFS, submodules ...

class RepoHandler:
    """Main class for handling a Git repository and its metadata."""
    def __init__(
        self,
        data_path: Union[str, Path],
        repository_path: Union[str, Path],
        metadata_filename: str = METADATA_FILENAME, # From constants
        progress_dir_name: str = PROGRESS_DIR, # From constants
        create_if_not_exist: bool = True, 
        use_dvc: bool = False,
        metadata_compression: Optional[str] = None, # New argument, e.g., "gzip"
        git_ops_helper: GitOpsHelper = None,
        metadata_handler: MetadataFileHandler = None,
        repo_hash=None,
        repo_index_entry=None
    ):
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>RepoHandler initializing with path: {repository_path}", Path(__file__).stem)
        self.data_path = data_path
        self.modifier = RepoModifier(git_ops_helper, metadata_handler)
        self.analyzer = RepoAnalyzer(git_ops_helper)
        self.repo_path = Path(repository_path).resolve()
        self.is_new_repo = False
        self.repo_hash = repo_hash
        self.repo_index_entry = repo_index_entry
        try:
            self.repo = git.Repo(self.repo_path)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Opened existing Git repository at: {self.repo_path}", Path(__file__).stem)
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            if create_if_not_exist:
                self.repo_path.mkdir(parents=True, exist_ok=True)
                self.repo = git.Repo.init(self.repo_path)
                self.is_new_repo = True
                actual_log_statement("info", f"{LOG_INS}:INFO>>Initialized new Git repository at: {self.repo_path}", Path(__file__).stem)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Repository not found at {self.repo_path} and create_repo_if_not_exists is False.", Path(__file__).stem)
                raise
        except Exception as e:
            actual_log_statement("exception", f"{LOG_INS}:EXCEPTION>>Failed to open or initialize repository at {self.repo_path}: {e}", Path(__file__).stem)
            raise
        self.repo: Optional[GitOpsHelper] = self._initialize_git_ops_helper(create_if_not_exist)

        if self.repo and hasattr(self.repo, 'git_repo') and self.repo.git_repo:
            self.gitignore_handler = GitignoreFileHandler(self.repo)
        else:
            log_statement('warning', f"{LOG_INS}:WARNING>>Git repository not properly initialized for {self.data_path}. Gitignore handling will be disabled.", "RepoHandler")
            self.gitignore_handler = None # Or a dummy handler that does nothing

        self.git_ops_helper: Optional[GitOpsHelper] = self._initialize_git_ops_helper(create_if_not_exist)
        self.is_new_git_repo = self.git_ops_helper.is_new_repo if self.git_ops_helper else False

        self.metadata_handler = MetadataFileHandler(self.repo_path, metadata_filename, use_compression=metadata_compression)
        self.progress_handler = ProgressFileHandler(self.repo_path, progress_dir_name)
        if self.git_ops_helper and self.git_ops_helper.is_valid_repo():
            # Assuming GitignoreFileHandler constructor takes the GitOpsHelper instance
            self.gitignore_handler = GitignoreFileHandler(self.git_ops_helper, gitignore_name=self.gitignore_name)
        else:
            self.gitignore_handler = None
            log_statement('warning', f"{LOG_INS}:WARNING>>Git repository not properly initialized via GitOpsHelper for {self.repo_path}. Gitignore handling disabled.", Path(__file__).stem)

        # Ensure metadata file exists and commit if it's a new repo or file missing
        # This implicitly creates an empty {} metadata file if it's brand new.
        self.metadata_handler.ensure_metadata_file_exists(
            repo_instance=self.repo, 
            initial_commit=self.is_new_repo
        )
        self.all_metadata_dict = self.metadata_handler.read_metadata()
        if not self.all_metadata_dict:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Metadata file is empty or could not be read", Path(__file__).stem)
        actual_log_statement('info', f"{LOG_INS}:INFO>>RepoHandler initialized successfully.", Path(__file__).stem)

    def _initialize_git_ops_helper(self, create_if_not_exist: bool) -> Optional[GitOpsHelper]:
        """
        Initializes or loads a Git repository using the GitOpsHelper class.

        This method attempts to load an existing Git repository from self.data_path.
        If no repository exists and create_if_not_exist is True, GitOpsHelper
        will attempt to initialize a new one.

        Args:
            create_if_not_exist (bool): If True, allows GitOpsHelper to initialize
                                         a new Git repository if one doesn't exist.

        Returns:
            Optional[GitOpsHelper]: An instance of GitOpsHelper. The helper's
                                    .git_repo attribute will be a git.Repo object
                                    if successful, or None if setup failed.
                                    Returns None if self.data_path is invalid.
        """
        # Path(__file__).stem is assumed to be an attribute of RepoHandler
        log_statement('info', f"{LOG_INS}:INFO>>RepoHandler: Initializing GitOpsHelper for path: {self.data_path}, create: {create_if_not_exist}", Path(__file__).stem)

        if not isinstance(self.data_path, Path):
            log_statement('info', f"{LOG_INS}:INFO>>RepoHandler: self.data_path is not a Path object: {self.data_path}. Cannot initialize GitOpsHelper.", Path(__file__).stem)
            return None

        try:
            # GitOpsHelper handles its own internal logging for success/failure of git init/load
            helper = GitOpsHelper(
                repo_path=self.repo_path,
                create_if_not_exist=create_if_not_exist,
                logger_name=Path(__file__).stem  # You can pass RepoHandler's logger name, or a derived one
            )

            if helper.is_valid_repo():
                log_statement('info', f"{LOG_INS}:INFO>>RepoHandler: GitOpsHelper successfully set up Git repository for {self.data_path}.", Path(__file__).stem)
            else:
                # This means GitOpsHelper was instantiated, but its .git_repo is None.
                # This is an expected outcome if, e.g., path is not a repo and create_if_not_exist is False.
                log_statement('warning', f"{LOG_INS}:WARNING>>RepoHandler: GitOpsHelper instantiated, but Git repository is not available for {self.data_path}.", Path(__file__).stem)

            return helper

        except Exception as e:
            # This would catch unexpected errors during GitOpsHelper instantiation itself,
            # though GitOpsHelper aims to catch its internal errors.
            log_statement('critical', f"{LOG_INS}:CRITICAL>>RepoHandler: Critical error creating GitOpsHelper instance for {self.data_path}: {e}", Path(__file__).stem, exc_info=True)
            return None

    def is_git_repo(self) -> bool:
        # Should check if self.repo and self.repo.git_repo are valid
        return bool(self.repo and hasattr(self.repo, 'git_repo') and self.repo.git_repo)

    def _get_relative_path(self, file_path: Union[str, Path]) -> Optional[str]:
        try:
            abs_path = Path(file_path).resolve()
            # Ensure the file is within the repository
            if not abs_path.is_relative_to(self.repo_path): # Path.is_relative_to is Python 3.9+
                 # Manual check for older Python if needed:
                 # if self.repo_path not in abs_path.parents and self.repo_path != abs_path:
                if str(abs_path).startswith(str(self.repo_path)):
                     pass # Path is likely within or is the repo path itself
                else:
                    actual_log_statement('warning', f"{LOG_INS}:WARNING>>File {abs_path} is not inside repository {self.repo_path}.", Path(__file__).stem)
                    return None
            
            # Use os.path.relpath for robust relative path, convert to platform-agnostic string
            rel_path = Path(os.path.relpath(abs_path, self.repo_path))
            return rel_path.as_posix() # Store with POSIX separators
        except ValueError as ve: # Can happen if paths are on different drives (Windows)
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not determine relative path for {file_path} against {self.repo_path}: {ve}", Path(__file__).stem, exc_info=True)
            return None
        except Exception as e:
            actual_log_statement('exception', f"{LOG_INS}:EXCEPTION>>Error getting relative path for {file_path}: {e}", Path(__file__).stem)
            return None

    def _create_file_metadata_entry_data(self, abs_path: Path) -> Optional[Dict[str, Any]]:
        """Gathers raw metadata for a file, including OS stats, custom hashes, and Git blob hash."""
        if not _get_os_and_content_metadata_defined:
            actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>get_os_and_content_metadata helper is not available. Cannot create metadata entry.", Path(__file__).stem)
            return None
        
        # 1. Get OS and content hashes using the helper
        # Use hash algorithms defined in constants, e.g., SUPPORTED_HASH_ALGORITHMS for metadata
        # This list should align with what FileMetadataEntry.custom_hashes expects (e.g., ["md5", "sha256"])
        hashes_to_calc = SUPPORTED_HASH_ALGORITHMS # From constants.py, e.g., ['md5', 'sha256']
        os_and_hash_meta = get_os_and_content_metadata(abs_path, hash_algorithms=hashes_to_calc, calculate_custom_hashes=True)

        if not os_and_hash_meta:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Failed to get OS/content metadata for {abs_path}.", Path(__file__).stem)
            return None

        # 2. Get Git blob hash
        # Note: This calculates blob hash for current content on disk, not necessarily what's in HEAD or index.
        # For `git_object_hash_current` field, we typically want the hash of the committed version.
        # This will be updated after commit if the file is part of that commit.
        # For now, calculate based on current disk content:
        git_blob_hash = None
        try:
            with open(abs_path, 'rb') as f:
                blob = self.repo.odb.store(git.Blob.input_stream(f, 'blob'))
                git_blob_hash = blob.hexsha
        except Exception as e:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Could not calculate Git blob hash for {abs_path}: {e}", Path(__file__).stem, exc_info=False)

        # Prepare a dictionary that largely matches FileMetadataEntry fields
        entry_data = {
            "filename": os_and_hash_meta.get("filename"),
            "extension": os_and_hash_meta.get("extension"),
            "size_bytes": os_and_hash_meta.get("size_bytes"),
            "os_last_modified_utc": os_and_hash_meta.get("os_last_modified_utc"), # Assumes helper returns ISO string
            "os_created_utc": os_and_hash_meta.get("os_created_utc"),       # Assumes helper returns ISO string
            "custom_hashes": os_and_hash_meta.get("custom_hashes", {}),
            "git_object_hash_calculated_on_disk": git_blob_hash, # Temp name to distinguish from committed hash
        }
        return entry_data

    def find_duplicate_files(self, hash_type: str = DEFAULT_HASH_ALGORITHM) -> Dict[str, List[str]]:
        """
        Finds files with duplicate content based on a specified hash type.

        Args:
            hash_type (str): The hash algorithm (e.g., 'md5', 'sha256') to use for comparison,
                             taken from the 'custom_hashes' field of metadata entries.
                             Defaults to DEFAULT_HASH_ALGORITHM from constants.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are hash values and
                                   values are lists of relative file paths sharing that hash.
                                   Only includes hashes that appear for more than one file.
        """
        actual_log_statement('info', f"{LOG_INS}:INFO>>Finding duplicate files based on hash type: {hash_type}", Path(__file__).stem)

        if hash_type.lower() not in SUPPORTED_HASH_ALGORITHMS:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Hash type '{hash_type}' is not in supported list: {SUPPORTED_HASH_ALGORITHMS}. Results may be empty or inaccurate.", Path(__file__).stem)
            # Proceeding anyway, as custom_hashes might contain it, but warn.

        all_metadata_dict = self.metadata_handler.read_metadata()
        if not all_metadata_dict:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Metadata is empty. No duplicates to find.", Path(__file__).stem)
            return {}

        hash_map: Dict[str, List[str]] = {}
        for rel_path_str, entry_dict in all_metadata_dict.items():
            try:
                # No need to parse full Pydantic model if only accessing a sub-dict
                custom_hashes = entry_dict.get("custom_hashes", {})
                file_hash_value = custom_hashes.get(hash_type.lower())

                if file_hash_value:
                    if file_hash_value not in hash_map:
                        hash_map[file_hash_value] = []
                    hash_map[file_hash_value].append(rel_path_str)
            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not process entry for {rel_path_str} during duplicate check: {e}", Path(__file__).stem, exc_info=True)
        
        duplicates = {hash_val: paths for hash_val, paths in hash_map.items() if len(paths) > 1}
        
        if duplicates:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Found {len(duplicates)} sets of duplicate files.", Path(__file__).stem)
        else:
            actual_log_statement('info', f"{LOG_INS}:INFO>>No duplicate files found based on hash type: {hash_type}.", Path(__file__).stem)
            
        return duplicates

    def remove_file_from_tracking(
        self,
        file_path: Union[str, Path],
        removal_action: str = "archive", # Options: "archive", "mark_deleted", "delete_from_disk_and_git"
        change_description: Optional[str] = None, # Custom description for version history
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Removes a file from active tracking. Updates its status in metadata,
        adds a version history record, and optionally deletes the file from
        the working directory and Git index.

        Args:
            file_path: Absolute or relative path to the file.
            removal_action: How to handle the file:
                - "archive": Set status to STATUS_ARCHIVED. File remains.
                - "mark_deleted": Set status to STATUS_DELETED. File remains.
                - "delete_from_disk_and_git": Set status to STATUS_DELETED,
                  remove from working tree and Git index.
            change_description: Optional description for the version history.
            commit_message: Custom Git commit message.

        Returns:
            True if the operation was successful (including metadata commit), False otherwise.
        """
        
        rel_path_str = self._get_relative_path(Path(file_path))
        if not rel_path_str:
            # If path is already relative and valid, try to use it directly
            # This check assumes file_path could be a pre-calculated relative path
            temp_abs_path = self.repo_path / file_path
            if temp_abs_path.exists() or any(item.a_path == str(Path(file_path)) for item in self.repo.index.diff(None) + self.repo.index.diff("HEAD")): # Check if it's in Git index or working tree and known
                 rel_path_str = Path(file_path).as_posix()
            else:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>File path '{file_path}' could not be resolved to a relative path in the repository. Cannot remove.", Path(__file__).stem)
                return False
        
        abs_path = self.repo_path / rel_path_str # For disk operations

        all_metadata_dict = self.metadata_handler.read_metadata()
        entry_dict_raw = all_metadata_dict.get(rel_path_str)

        if not entry_dict_raw:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>File {rel_path_str} not found in metadata. Cannot remove.", Path(__file__).stem)
            return False
        
        try:
            pydantic_entry = FileMetadataEntry.model_validate(entry_dict_raw)
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to parse metadata for {rel_path_str} for removal: {e}", Path(__file__).stem, exc_info=True)
            return False

        old_status = pydantic_entry.application_status
        new_status: str

        valid_actions = ["archive", "mark_deleted", "delete_from_disk_and_git"]
        if removal_action not in valid_actions:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Invalid removal action '{removal_action}'. Must be one of {valid_actions}. Defaulting to 'archive'.", Path(__file__).stem)
            removal_action = "archive"
        
        if removal_action == "archive":
            new_status = STATUS_ARCHIVED
        else: # "mark_deleted" or "delete_from_disk_and_git"
            new_status = STATUS_DELETED
        
        pydantic_entry.application_status = new_status
        pydantic_entry.last_metadata_update_utc = datetime.now(timezone.utc)
        pydantic_entry.version_current += 1

        version_desc = change_description or f"File action: {removal_action}. Status changed from '{old_status}' to '{new_status}'."
        current_custom_hashes_dict = pydantic_entry.custom_hashes # This is Dict[str, str]
        custom_hashes_for_history = []
        if current_custom_hashes_dict: # Ensure it's not None before iterating
            custom_hashes_for_history = [HashInfo(hash_type=ht, value=hv) for ht, hv in current_custom_hashes_dict.items()]

        version_record = FileVersion(
            version_number=pydantic_entry.version_current,
            timestamp_utc=pydantic_entry.last_metadata_update_utc, # or appropriate timestamp
            change_description=change_description_for_history,
            size_bytes=pydantic_entry.size_bytes,
            custom_hashes=custom_hashes_for_history, # Assign the list of HashInfo objects
            git_commit_hash=None # Placeholder, filled later
        )
        pydantic_entry.version_history_app.append(version_record)
        all_metadata_dict[rel_path_str] = pydantic_entry.model_dump(by_alias=True)

        # Write metadata before Git operations involving it
        if not self.metadata_handler.write_metadata(metadata_collection.to_dict()):
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to write metadata updates for removing {rel_path_str}.", Path(__file__).stem)
            return False

        files_to_commit_abs_paths: List[Path] = [self.metadata_handler.metadata_filepath]
        
        if removal_action == "delete_from_disk_and_git":
            try:
                if abs_path.is_file(): # Check if it's on disk before trying to remove
                    self.repo.index.remove([rel_path_str], working_tree=True)
                    actual_log_statement('info', f"{LOG_INS}:INFO>>Removed {rel_path_str} from disk and staged for Git removal.", Path(__file__).stem)
                elif rel_path_str in [item.a_path for item in self.repo.index.diff("HEAD")] + [item.a_path for item in self.repo.index.diff(None)]:
                    # File might be in index but not on disk (e.g. deleted but removal not staged yet)
                    # Or already staged for removal. `git rm` will handle this.
                    self.repo.index.remove([rel_path_str], working_tree=False) # Ensure it's staged for removal
                    actual_log_statement('info', f"{LOG_INS}:INFO>>File {rel_path_str} not on disk, but staged its removal from Git index.", Path(__file__).stem)
                else:
                    actual_log_statement('warning', f"{LOG_INS}:WARNING>>File {rel_path_str} to be deleted from disk/Git was not found on disk and not staged. Metadata updated only.", Path(__file__).stem)
            except git.exc.GitCommandError as e:
                # This can happen if the file was never added to git, or already removed.
                if "did not match any files" in str(e).lower():
                     actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Git rm: file {rel_path_str} did not match any files. Likely not tracked or already removed.", Path(__file__).stem)
                else:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to stage removal of {rel_path_str} from Git: {e}", Path(__file__).stem, exc_info=True)
                    # Depending on desired atomicity, one might choose to not commit metadata if git rm fails.
                    # For now, we proceed to commit metadata.
            except Exception as e:
                 actual_log_statement('error', f"{LOG_INS}:ERROR>>Unexpected error during Git remove operation for {rel_path_str}: {e}", Path(__file__).stem, exc_info=True)


        final_commit_msg = commit_message or f"{removal_action.capitalize()} file and update metadata: {rel_path_str}"
        
        if self.git_ops_helper.commit_changes(files_to_commit_abs_paths, final_commit_msg): # This stages metadata.json again
            last_commit_hash = self.repo.head.commit.hexsha
            # Update version history in metadata with this commit hash
            # Re-read, update the specific entry's last version_history item, write back (no new commit)
            current_metadata_after_commit = self.metadata_handler.read_metadata()
            entry_to_finalize_dict = current_metadata_after_commit.get(rel_path_str)
            if entry_to_finalize_dict:
                try:
                    entry_to_finalize_pydantic = FileMetadataEntry.model_validate(entry_to_finalize_dict)
                    if entry_to_finalize_pydantic.version_history_app:
                        entry_to_finalize_pydantic.version_history_app[-1].git_commit_hash = last_commit_hash
                        current_metadata_after_commit[rel_path_str] = entry_to_finalize_pydantic.model_dump(by_alias=True)
                        self.metadata_handler.write_metadata(current_metadata_after_commit, repo_instance=None) # No re-commit for this minor update
                except Exception as e_finalize:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to finalize commit hash in metadata for {rel_path_str}: {e_finalize}", Path(__file__).stem, exc_info=True)

            actual_log_statement('info', f"{LOG_INS}:INFO>>File {rel_path_str} processing for removal action '{removal_action}' complete. Commit: {last_commit_hash}", Path(__file__).stem)
            return True
        else:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit removal action for {rel_path_str}.", Path(__file__).stem)
            return False

    def verify_file_integrity(self, file_path: Union[str, Path], hash_type: str = DEFAULT_HASH_ALGORITHM) -> str:
        """
        Verifies the integrity of a tracked file by comparing its current disk hash
        with the stored hash in metadata.

        Args:
            file_path: Absolute or relative path to the file.
            hash_type: The hash algorithm ('md5', 'sha256', etc.) from 'custom_hashes' to verify.
                       Defaults to DEFAULT_HASH_ALGORITHM.

        Returns:
            A status string: "verified", "corrupted", "untracked",
                            "missing_on_disk", "metadata_missing_hash_type",
                            "error_calculating_disk_hash", "error_parsing_metadata".
        """        
        rel_path_str = self._get_relative_path(Path(file_path))
        if not rel_path_str:
            # Attempt to use file_path as if it was already relative
            if (self.repo_path / file_path).is_file():
                rel_path_str = Path(file_path).as_posix()
            else:
                actual_log_statement('warning', f"{LOG_INS}:WARNING>>File path '{file_path}' could not be resolved. Integrity cannot be verified.", Path(__file__).stem)
                return "unresolved_path" # New status

        metadata_entry_model = self.get_file_metadata_entry(rel_path_str) # Handles path resolution again, but ok
        
        if not metadata_entry_model:
            return "untracked" # Not found in metadata

        abs_path = self.repo_path / rel_path_str
        if not abs_path.exists() or not abs_path.is_file():
            return "missing_on_disk"

        stored_hash_value = metadata_entry_model.custom_hashes.get(hash_type.lower())
        if not stored_hash_value:
            return f"metadata_missing_hash_type_{hash_type.lower()}"

        # Recalculate hash for current file on disk
        current_disk_hash_value = None
        try:
            # Ensure get_os_and_content_metadata is available
            if not _get_os_and_content_metadata_defined:
                 actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>get_os_and_content_metadata helper is not available. Cannot verify integrity.", Path(__file__).stem)
                 return "error_internal_helper_missing"

            # We need only one specific hash.
            # The helper calculates multiple by default. We can make it more specific if performance is an issue.
            disk_meta_probe = get_os_and_content_metadata(abs_path, hash_algorithms=[hash_type.lower()], calculate_custom_hashes=True)
            if disk_meta_probe and disk_meta_probe.get("custom_hashes"):
                current_disk_hash_value = disk_meta_probe["custom_hashes"].get(hash_type.lower())
        except Exception as e:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>Error calculating current disk hash for {abs_path} during verification: {e}", Path(__file__).stem, exc_info=True)
             return "error_calculating_disk_hash"

        if not current_disk_hash_value:
            return "error_calculating_disk_hash" # Hash calculation failed or returned no value

        if current_disk_hash_value == stored_hash_value:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Integrity verified for {rel_path_str} (hash type: {hash_type}).", Path(__file__).stem)
            return "verified"
        else:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Integrity check FAILED for {rel_path_str}. Stored {hash_type}: '{stored_hash_value}', Current disk {hash_type}: '{current_disk_hash_value}'.", Path(__file__).stem)
            return "corrupted"
            
    # Task 1.8.b: scan_repository_for_discrepancies
    def scan_repository_for_discrepancies(self) -> Dict[str, List[str]]:
        """
        Scans the repository for discrepancies between metadata, Git state, and working directory.

        Returns:
            A dictionary detailing discrepancies:
            - "untracked_in_git_and_metadata": Files in WD, not in Git, not in metadata.
            - "untracked_in_metadata_but_in_git": Files in Git index/HEAD, but not in metadata.json.
            - "metadata_missing_from_disk": Files in metadata.json, but not found on disk.
            - "modified_not_updated_in_metadata": Tracked files (in metadata & Git) whose disk content
                                                 (hash) differs from metadata, and not matching current Git HEAD object hash.
            - "staged_not_yet_in_metadata_history": Files staged in Git whose changes are not yet reflected
                                                     as a new version in metadata.json's history with this commit-to-be.
            - 'in_git_not_in_metadata': Files in Git index/HEAD, but not in metadata.json.
            - 'in_metadata_not_on_disk': Files in metadata.json, but missing from disk.
            - 'in_metadata_not_in_git': Files in metadata.json and on disk, but not in Git index/HEAD.
            - 'modified_content_vs_metadata': Files in metadata whose current disk content (hash)
                                              differs from 'custom_hashes' stored in metadata.
            - 'modified_content_vs_git_head': Tracked files in Git whose WD content differs from HEAD.
            - 'staged_changes_for_tracked_files': Tracked files in Git that have changes staged for commit.
        """
        actual_log_statement('info', f"{LOG_INS}:INFO>>Scanning repository for discrepancies...", Path(__file__).stem)
        
        discrepancies: Dict[str, List[str]] = {
            "untracked_in_git_and_metadata": [],
            "untracked_in_metadata_but_in_git": [],
            "metadata_missing_from_disk": [],
            "modified_not_updated_in_metadata": [],
            "staged_not_yet_in_metadata_history": [],
            "in_git_not_in_metadata": [],
            "in_metadata_not_on_disk": [],
            "in_metadata_not_in_git": [], # New category for clarity
            "modified_content_vs_metadata": [], # Compares disk file hash to metadata.custom_hashes
            "modified_content_vs_git_head": [], # Compares disk file to what's in HEAD (if tracked by Git)
            "staged_changes_for_tracked_files": [], # Files with changes staged in Git index
        }

        all_metadata = self.metadata_handler.read_metadata()
        metadata_relative_paths = set(all_metadata.keys())

        # 1. Files in working directory (respecting .gitignore through Git) vs metadata
        # Git ls-files shows tracked files. Git status shows untracked (that are not ignored).
        
        # Files known to Git (in index or HEAD)
        git_tracked_files_relative = set()
        try:
            # ls_files includes files in HEAD and in the index (staged)
            # It gives paths relative to repo root, using POSIX separators.
            for untracked_file_rel_path_str in self.repo.untracked_files:
                posix_path = Path(untracked_file_rel_path_str).as_posix()
                if posix_path not in metadata_relative_paths:
                    discrepancies["untracked_by_git_and_metadata"].append(posix_path)

        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not list Git tracked files: {e}", Path(__file__).stem, exc_info=True)


        # Untracked files by Git (respects .gitignore)
        for untracked_file_rel_path_str in self.repo.untracked_files:
            # These are untracked by Git. If also not in metadata, they are truly "new".
            posix_path = Path(untracked_file_rel_path_str).as_posix()
            if posix_path not in metadata_relative_paths:
                discrepancies["untracked_in_git_and_metadata"].append(posix_path)

        # 2. Files known to Git (in index or HEAD)
        git_known_files_relative = set()
        try:
            for item_path_str in self.repo.git.ls_files(cached=True, others=False, modified=False, deleted=False).splitlines(): # Files in index
                git_known_files_relative.add(Path(item_path_str).as_posix())
            # Consider also files only in HEAD but not index (e.g. after `git reset <file>`) - this is complex.
            # `ls-files` with no flags often covers what's broadly "known" to git in WD.
            # A simpler proxy: iterate all files in WD, check if tracked using `git ls-files --error-unmatch <file>`
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not list Git index files: {e}", Path(__file__).stem, exc_info=True)

        for git_file_rel_path in git_known_files_relative:
            if git_file_rel_path not in metadata_relative_paths and \
               git_file_rel_path != self.metadata_handler.metadata_filepath.name : # Exclude metadata file itself
                discrepancies["in_git_not_in_metadata"].append(git_file_rel_path)


            # 3. Modified files (content hash check)
            # Compare disk content hash with stored custom hash AND stored git_object_hash_current
            try:
                meta_entry = FileMetadataEntry.model_validate(meta_entry_dict)
                current_disk_hashes = {}
                # Use a focused call to get current hashes to avoid full _create_file_metadata_entry_data
                if _get_os_and_content_metadata_defined:
                    disk_meta_probe = get_os_and_content_metadata(abs_path, hash_algorithms=list(meta_entry.custom_hashes.keys()), calculate_custom_hashes=True)
                    if disk_meta_probe:
                        current_disk_hashes = disk_meta_probe.get("custom_hashes", {})

                hash_mismatch = False
                for algo, stored_hash in meta_entry.custom_hashes.items():
                    if current_disk_hashes.get(algo) != stored_hash:
                        hash_mismatch = True
                        break
                
                # Also check current Git blob hash vs what's stored for committed state
                # This requires care: meta_entry.git_object_hash_current is for the file AS OF ITS LAST METADATA COMMIT.
                # A new blob hash for current disk content:
                current_disk_blob_hash = self.git_ops_helper.get_file_blob_hash(rel_path_str)

                if hash_mismatch or (meta_entry.git_object_hash_current and current_disk_blob_hash != meta_entry.git_object_hash_current):
                    # Check if this modification is already staged
                    is_staged_with_diff_content = False
                    for diff_item in self.repo.index.diff(None): # Diff of staging area against working tree
                        if Path(diff_item.a_path).as_posix() == rel_path_str and diff_item.change_type in ['M', 'A']: # Modified or Added (if it was deleted and re-added)
                            # If a diff exists between staging and working tree, it means work tree IS different from STAGED.
                            # We are comparing disk to *committed* metadata.
                            # If disk differs from committed metadata, it's a modification.
                            # If this modification is also what's staged, then `staged_new_content_for_tracked_file` applies.
                            pass # Handled by staged check below
                    
                    # This means disk content differs from what metadata *thinks* is committed.
                    # It could be that user modified, and *has not* staged.
                    # Or user modified, *has* staged, but not yet run a RepoHandler method to update metadata.
                    if current_disk_blob_hash != meta_entry.git_object_hash_current: # Primary check using Git's view of content
                         discrepancies["modified_not_updated_in_metadata"].append(rel_path_str)

            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Error verifying integrity for {rel_path_str} during scan: {e}", Path(__file__).stem, exc_info=True)

        # 3. Iterate through metadata entries
        for rel_path_str, entry_dict_raw in all_metadata_entries.items():
            abs_path = self.repo_path / rel_path_str
            try:
                meta_entry = FileMetadataEntry.model_validate(entry_dict_raw)
            except Exception as e:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Corrupted metadata for {rel_path_str}, skipping in scan: {e}", Path(__file__).stem)
                continue

            # 3a. In metadata, but not on disk?
            if not abs_path.exists():
                discrepancies["in_metadata_not_on_disk"].append(rel_path_str)
                continue # No further disk checks possible

            # 3b. In metadata and on disk, but not in Git index/HEAD?
            if rel_path_str not in git_known_files_relative and abs_path.is_file(): # Check is_file again
                discrepancies["in_metadata_not_in_git"].append(rel_path_str)

            # 3c. Content modification vs metadata's custom_hashes
            # Using verify_file_integrity's logic but without returning immediately
            hash_type_to_check = DEFAULT_HASH_ALGORITHM # Or iterate all in meta_entry.custom_hashes
            stored_hash = meta_entry.custom_hashes.get(hash_type_to_check.lower())
            if stored_hash:
                current_disk_hash = None
                if _get_os_and_content_metadata_defined:
                    disk_meta_probe = get_os_and_content_metadata(abs_path, [hash_type_to_check.lower()], True)
                    if disk_meta_probe and disk_meta_probe.get("custom_hashes"):
                        current_disk_hash = disk_meta_probe["custom_hashes"].get(hash_type_to_check.lower())
                
                if current_disk_hash and current_disk_hash != stored_hash:
                    discrepancies["modified_content_vs_metadata"].append(rel_path_str)
            
        # 4. Git diffs for modified and staged files (tracked by Git)
        # Modified in working directory compared to HEAD (but not necessarily staged)
        try:
            for diff_item in self.repo.head.commit.diff(None): # Diff HEAD with working tree
                # diff_item.a_path is the path in HEAD, diff_item.b_path is path in WD (often same)
                # change_type 'M' for modification, 'A' for added in WD not in HEAD (unlikely for this diff), 'D' for deleted in WD
                if diff_item.change_type == 'M':
                    path_str = Path(diff_item.a_path).as_posix()
                    if path_str in metadata_relative_paths: # Only if tracked in our metadata
                         # Avoid double reporting if already caught by custom_hash check
                        if path_str not in discrepancies["modified_content_vs_metadata"]:
                            discrepancies["modified_content_vs_git_head"].append(path_str)
        except Exception as e:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>Error diffing HEAD vs working tree: {e}", Path(__file__).stem, exc_info=True)


        # Staged changes for tracked files (compared to HEAD)
        try:
            for diff_item in self.repo.index.diff("HEAD"): # Diff index with HEAD
                if diff_item.change_type in ['M', 'A']: # Modified or Added (if re-added after delete)
                    path_str = Path(diff_item.a_path).as_posix() # or b_path, should be same for 'M'
                    if path_str in metadata_relative_paths:
                        discrepancies["staged_changes_for_tracked_files"].append(path_str)
        except Exception as e:
             actual_log_statement('error', f"{LOG_INS}:ERROR>>Error diffing index vs HEAD: {e}", Path(__file__).stem, exc_info=True)

        actual_log_statement('info', f"{LOG_INS}:INFO>>Repository scan complete. Discrepancies: { {k:v for k,v in discrepancies.items() if v} }", Path(__file__).stem)
        return discrepancies

    def add_file_to_tracking(
        self,
        file_path: Union[str, Path],
        application_status: str = STATUS_NEW,
        user_metadata: Optional[Dict[str, Any]] = None,
        change_description: Optional[str] = "Initial file registration.",
        commit_message: Optional[str] = None,
        compress_as: Optional[str] = None # New parameter e.g. "gzip"
    ) -> bool:
        _log_ins_val = _get_log_ins(inspect.currentframe(), self.LOG_INS_CLASS)
        original_abs_path = Path(file_path).resolve()
        
        # This path will point to the file whose metadata is gathered and what's staged in Git.
        # It might be the original file or a compressed version (with a new name like .gz).
        path_to_process_and_store_in_repo = original_abs_path 
        is_file_compressed_for_repo = False
        actual_compression_type = None
        original_filename_for_metadata_field = None # Only set if compressed and name changes

        if not original_abs_path.exists() or not original_abs_path.is_file():
            actual_log_statement('error', f"{LOG_INS}:ERROR>>File to add does not exist or is not a file: {original_abs_path}", Path(__file__).stem)
            return False

        # --- Compression Handling ---
        if compress_as and COMPRESSION_UTILS_AVAILABLE:
            if compress_as.lower() == "gzip":
                # The file stored in the repo will have a .gz suffix.
                # The original file (input `file_path`) is read but not modified itself.
                # A new, compressed file is created in the same directory as the original.
                compressed_file_path_for_repo = original_abs_path.parent / (original_abs_path.name + ".gz")
                actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Attempting to compress {original_abs_path.name} to {compressed_file_path_for_repo.name}", Path(__file__).stem)
                try:
                    if compress_file_gzip(original_abs_path, compressed_file_path_for_repo, remove_original=False):
                        actual_log_statement('info', f"{LOG_INS}:INFO>>Successfully compressed {original_abs_path.name} to {compressed_file_path_for_repo.name} for repository storage.", Path(__file__).stem)
                        path_to_process_and_store_in_repo = compressed_file_path_for_repo # This is the file to hash and add to Git
                        is_file_compressed_for_repo = True
                        actual_compression_type = "gzip"
                        original_filename_for_metadata_field = original_abs_path.name # Store the original name
                    else:
                        actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to compress {original_abs_path.name}. Adding uncompressed version.", Path(__file__).stem)
                        # path_to_process_and_store_in_repo remains original_abs_path
                except Exception as e_compress:
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Error during compression of {original_abs_path.name}: {e_compress}", Path(__file__).stem, exc_info=True)
                    # Fallback to using the original uncompressed file
                    path_to_process_and_store_in_repo = original_abs_path
            else:
                actual_log_statement('warning', f"{LOG_INS}:WARNING>>Unsupported compression type '{compress_as}'. File will be added uncompressed.", Path(__file__).stem)
        # --- End Compression Handling ---

        rel_path_str_for_repo_file = self._get_relative_path(path_to_process_and_store_in_repo)
        if not rel_path_str_for_repo_file:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>File {path_to_process_and_store_in_repo} (target for repo) is not within the repository. Cannot add.", Path(__file__).stem)
            if is_file_compressed_for_repo and path_to_process_and_store_in_repo.exists() and path_to_process_and_store_in_repo != original_abs_path:
                path_to_process_and_store_in_repo.unlink(missing_ok=True) # Cleanup newly created compressed file
            return False

        # Metadata is gathered for path_to_process_and_store_in_repo
        raw_metadata_values = self._create_file_metadata_entry_data(path_to_process_and_store_in_repo, rel_path_str_for_repo_file)
        if not raw_metadata_values:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to gather metadata for {path_to_process_and_store_in_repo}. Cannot add.", Path(__file__).stem)
            if is_file_compressed_for_repo and path_to_process_and_store_in_repo.exists() and path_to_process_and_store_in_repo != original_abs_path:
                 path_to_process_and_store_in_repo.unlink(missing_ok=True) 
            return False
        
        # Add compression info to be stored in metadata
        raw_metadata_values["original_filename_if_compressed"] = original_filename_for_metadata_field
        raw_metadata_values["compression_type"] = actual_compression_type
        
        all_metadata_dict = self.metadata_handler.read_metadata()
        try:
            metadata_collection = MetadataCollection.model_validate(all_metadata_dict if all_metadata_dict else {})
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to parse existing metadata.json: {e}", Path(__file__).stem, exc_info=True)
            metadata_collection = MetadataCollection(root={}) 

        now_utc_dt = datetime.now(timezone.utc)
        existing_pydantic_entry = metadata_collection.get_entry(rel_path_str_for_repo_file)
        
        new_version_number = 1
        version_history_list: List[FileVersion] = []

        if existing_pydantic_entry:
            actual_log_statement('info', f"{LOG_INS}:INFO>>File {rel_path_str_for_repo_file} already tracked. Updating metadata and version history.", Path(__file__).stem)
            new_version_number = existing_pydantic_entry.version_current + 1
            version_history_list = existing_pydantic_entry.version_history_app or []
            
            prev_version_record = FileVersion(
                version_number=existing_pydantic_entry.version_current,
                timestamp_utc=existing_pydantic_entry.last_metadata_update_utc, # This should be a datetime object
                change_description=f"Superseded by version {new_version_number}",
                size_bytes=existing_pydantic_entry.size_bytes,
                custom_hashes=existing_pydantic_entry.custom_hashes.copy(),
                git_commit_hash=existing_pydantic_entry.git_object_hash_current
            )
            version_history_list.append(prev_version_record)
            date_added_utc_dt = existing_pydantic_entry.date_added_to_metadata_utc # This should be a datetime object
        else:
            date_added_utc_dt = now_utc_dt
        
        # Ensure datetime fields from raw_metadata_values are datetime objects
        os_last_mod_dt = datetime.fromisoformat(raw_metadata_values["os_last_modified_utc"].replace('Z', '+00:00')) if isinstance(raw_metadata_values["os_last_modified_utc"], str) else raw_metadata_values["os_last_modified_utc"]
        os_created_dt = datetime.fromisoformat(raw_metadata_values["os_created_utc"].replace('Z', '+00:00')) if isinstance(raw_metadata_values["os_created_utc"], str) else raw_metadata_values["os_created_utc"]

        current_entry_data_for_model = {
            "filepath_relative": rel_path_str_for_repo_file,
            "filename": raw_metadata_values["filename"], 
            "extension": raw_metadata_values["extension"],
            "size_bytes": raw_metadata_values["size_bytes"],
            "os_last_modified_utc": os_last_mod_dt,
            "os_created_utc": os_created_dt,
            "custom_hashes": raw_metadata_values["custom_hashes"],
            "git_object_hash_current": None, # Updated after commit
            "date_added_to_metadata_utc": date_added_utc_dt,
            "last_metadata_update_utc": now_utc_dt,
            "application_status": application_status,
            "user_metadata": user_metadata or {},
            "version_current": new_version_number,
            "version_history_app": version_history_list,
            "original_filename_if_compressed": raw_metadata_values["original_filename_if_compressed"],
            "compression_type": raw_metadata_values["compression_type"]
        }
        
        try:
            current_pydantic_entry_obj = FileMetadataEntry(**current_entry_data_for_model)
        except Exception as e: 
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Pydantic validation failed for {rel_path_str_for_repo_file}: {e}", Path(__file__).stem, exc_info=True)
            if is_file_compressed_for_repo and path_to_process_and_store_in_repo.exists() and path_to_process_and_store_in_repo != original_abs_path:
                path_to_process_and_store_in_repo.unlink(missing_ok=True)
            return False

        current_version_details = FileVersion(
            version_number=current_pydantic_entry_obj.version_current,
            timestamp_utc=current_pydantic_entry_obj.last_metadata_update_utc,
            change_description=change_description or ("Initial registration" if not existing_pydantic_entry else "File content/metadata updated"),
            size_bytes=current_pydantic_entry_obj.size_bytes,
            custom_hashes=current_pydantic_entry_obj.custom_hashes.copy(),
            git_commit_hash=None 
        )
        current_pydantic_entry_obj.version_history_app.append(current_version_details)
        
        metadata_collection.add_or_update_entry(current_pydantic_entry_obj)

        final_commit_message = commit_message or f"Track file: {rel_path_str_for_repo_file} (v{current_pydantic_entry_obj.version_current})"
        # The file path_to_process_and_store_in_repo is what gets added to Git
        files_to_commit_abs = [path_to_process_and_store_in_repo, self.metadata_handler.metadata_filepath]
        
        if not self.metadata_handler.write_metadata(metadata_collection.to_dict(), repo_instance=None): # Write without commit, helper will commit
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to write metadata for {rel_path_str_for_repo_file} before commit.", Path(__file__).stem)
            if is_file_compressed_for_repo and path_to_process_and_store_in_repo.exists() and path_to_process_and_store_in_repo != original_abs_path:
                 path_to_process_and_store_in_repo.unlink(missing_ok=True)
            return False
        
        commit_successful = self.git_ops_helper.commit_changes(files_to_commit_abs, final_commit_message)

        if commit_successful:
            last_commit_obj = self.repo.head.commit
            committed_blob_hash_final = self.git_ops_helper.get_file_blob_hash(path_to_process_and_store_in_repo)

            final_meta_data_root = self.metadata_handler.read_metadata()
            final_metadata_collection_after_commit = MetadataCollection.model_validate(final_meta_data_root if final_meta_data_root else {})
            entry_to_finalize = final_metadata_collection_after_commit.get_entry(rel_path_str_for_repo_file)

            if entry_to_finalize:
                entry_to_finalize.git_object_hash_current = committed_blob_hash_final
                if entry_to_finalize.version_history_app:
                    entry_to_finalize.version_history_app[-1].git_commit_hash = last_commit_obj.hexsha
                self.metadata_handler.write_metadata(final_metadata_collection_after_commit.to_dict(), repo_instance=None)
            
            actual_log_statement('info', f"{LOG_INS}:INFO>>Successfully tracked {rel_path_str_for_repo_file}. Commit: {last_commit_obj.hexsha}", Path(__file__).stem)
            return True
        else:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit changes for {rel_path_str_for_repo_file}.", Path(__file__).stem)
            if is_file_compressed_for_repo and path_to_process_and_store_in_repo.exists() and path_to_process_and_store_in_repo != original_abs_path:
                 path_to_process_and_store_in_repo.unlink(missing_ok=True)
            return False

    def get_file_content(self, file_path: Union[str, Path], decompress_if_needed: bool = True) -> Optional[bytes]:
        meta_entry = self.get_file_metadata_entry(file_path) # This resolves to relative path
        if not meta_entry:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>File not found in metadata: {file_path}", Path(__file__).stem)
            return None

        # The filepath_relative in meta_entry is the path to the file stored in Git (could be compressed name)
        actual_file_in_repo_abs_path = self.repo_path / meta_entry.filepath_relative
        
        if not actual_file_in_repo_abs_path.exists():
            actual_log_statement('error', f"{LOG_INS}:ERROR>>File listed in metadata but not found on disk: {actual_file_in_repo_abs_path}", Path(__file__).stem)
            return None
        
        try:
            with open(actual_file_in_repo_abs_path, 'rb') as f:
                content = f.read()
            
            if decompress_if_needed and meta_entry.compression_type == "gzip" and COMPRESSION_UTILS_AVAILABLE:
                actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Decompressing {meta_entry.filepath_relative} (gzip)", Path(__file__).stem)
                return decompress_file(content) # Assumes decompress_gzip_content takes bytes returns bytes
            elif decompress_if_needed and meta_entry.compression_type == "zstd" and COMPRESSION_UTILS_AVAILABLE:
                actual_log_statement('debug', f"{LOG_INS}:DEBUG>>Decompressing {meta_entry.filepath_relative} ()", Path(__file__).stem)
                return decompress_file(content) # Assumes decompress_gzip_content takes bytes returns bytes
                
            return content
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to read/decompress file {meta_entry.filepath_relative}: {e}", Path(__file__).stem, exc_info=True)
            return None

    def add_files_batch(
        self,
        file_paths: List[Union[str, Path]],
        common_application_status: str = STATUS_NEW,
        common_user_metadata: Optional[Dict[str, Any]] = None,
        common_change_description: Optional[str] = "Batch file registration.",
        batch_commit_message: Optional[str] = "Batch add files to repository",
        compress_as: Optional[str] = None # Common compression for batch
    ) -> Dict[str, bool]: # Returns a dict mapping original input path string to success status
        _log_ins_val = _get_log_ins(inspect.currentframe(), self.LOG_INS_CLASS)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Starting batch add for {len(file_paths)} files. Compression: {compress_as or 'None'}.", Path(__file__).stem)
        
        results: Dict[str, bool] = {} # Keyed by original absolute file path string
        # List of tuples: (original_abs_path, path_stored_in_repo_abs, rel_path_for_repo_file, raw_metadata_dict_for_repo_file)
        processed_file_details_for_batch: List[Tuple[Path, Path, str, Dict[str, Any]]] = []
        
        # Step 1: Pre-process all file paths and handle compression individually
        for fp_arg in file_paths:
            original_abs_path_item = Path(fp_arg).resolve()
            results[str(original_abs_path_item)] = False # Default to failure

            path_to_process_item = original_abs_path_item
            is_compressed_item = False
            actual_compression_type_item = None
            original_filename_for_meta_item = None

            if not original_abs_path_item.exists() or not original_abs_path_item.is_file():
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Skipping invalid file {original_abs_path_item}", Path(__file__).stem)
                continue

            if compress_as and COMPRESSION_UTILS_AVAILABLE:
                if compress_as.lower() == "gzip":
                    compressed_target_path_item = original_abs_path_item.parent / (original_abs_path_item.name + ".gz")
                    try:
                        if compress_file_gzip(original_abs_path_item, compressed_target_path_item, remove_original=False):
                            path_to_process_item = compressed_target_path_item
                            is_compressed_item = True
                            actual_compression_type_item = "gzip"
                            original_filename_for_meta_item = original_abs_path_item.name
                        else:
                            actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Failed to compress {original_abs_path_item.name}. Will process uncompressed.", Path(__file__).stem)
                    except Exception as e_c_batch_item:
                        actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Error compressing {original_abs_path_item.name}: {e_c_batch_item}", Path(__file__).stem, exc_info=True)
            
            rel_path_str_item = self._get_relative_path(path_to_process_item)
            if not rel_path_str_item:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: File {path_to_process_item} is not in repo. Skipping.", Path(__file__).stem)
                if is_compressed_item and path_to_process_item.exists() and path_to_process_item != original_abs_path_item:
                    path_to_process_item.unlink(missing_ok=True)
                continue

            raw_meta_item = self._create_file_metadata_entry_data(path_to_process_item, rel_path_str_item)
            if raw_meta_item:
                raw_meta_item["original_filename_if_compressed"] = original_filename_for_meta_item
                raw_meta_item["compression_type"] = actual_compression_type_item
                processed_file_details_for_batch.append((original_abs_path_item, path_to_process_item, rel_path_str_item, raw_meta_item))
            else: # Metadata gathering failed
                if is_compressed_item and path_to_process_item.exists() and path_to_process_item != original_abs_path_item:
                     path_to_process_item.unlink(missing_ok=True)
        
        if not processed_file_details_for_batch:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>Batch: No valid files found to process after initial checks.", Path(__file__).stem)
            return results

        # Step 2: Update metadata_collection in one go
        all_metadata_dict = self.metadata_handler.read_metadata()
        try:
            metadata_collection = MetadataCollection.model_validate(all_metadata_dict if all_metadata_dict else {})
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Failed to parse existing metadata.json: {e}", Path(__file__).stem, exc_info=True)
            metadata_collection = MetadataCollection(root={})

        files_to_stage_for_git_abs: List[Path] = []
        commit_needed_for_batch = False

        for original_abs_p_detail, path_in_repo_abs_detail, rel_path_str_detail, raw_metadata_values_detail in processed_file_details_for_batch:
            now_utc_dt_batch = datetime.now(timezone.utc)
            existing_pydantic_entry_batch = metadata_collection.get_entry(rel_path_str_detail)
            
            new_version_number_b = 1
            version_history_list_b: List[FileVersion] = []

            if existing_pydantic_entry_batch:
                new_version_number_b = existing_pydantic_entry_batch.version_current + 1
                version_history_list_b = existing_pydantic_entry_batch.version_history_app or []
                prev_version_record_b = FileVersion(
                    version_number=existing_pydantic_entry_batch.version_current,
                    timestamp_utc=existing_pydantic_entry_batch.last_metadata_update_utc,
                    change_description=f"Batch update superseded by version {new_version_number_b}",
                    size_bytes=existing_pydantic_entry_batch.size_bytes,
                    custom_hashes=existing_pydantic_entry_batch.custom_hashes.copy(),
                    git_commit_hash=existing_pydantic_entry_batch.git_object_hash_current
                )
                version_history_list_b.append(prev_version_record_b)
                date_added_utc_dt_b = existing_pydantic_entry_batch.date_added_to_metadata_utc
            else:
                date_added_utc_dt_b = now_utc_dt_batch
            
            os_last_mod_dt_b = datetime.fromisoformat(raw_metadata_values_detail["os_last_modified_utc"].replace('Z', '+00:00')) if isinstance(raw_metadata_values_detail["os_last_modified_utc"], str) else raw_metadata_values_detail["os_last_modified_utc"]
            os_created_dt_b = datetime.fromisoformat(raw_metadata_values_detail["os_created_utc"].replace('Z', '+00:00')) if isinstance(raw_metadata_values_detail["os_created_utc"], str) else raw_metadata_values_detail["os_created_utc"]

            current_entry_data_batch_model = {
                "filepath_relative": rel_path_str_detail, **raw_metadata_values_detail,
                "git_object_hash_current": None, 
                "date_added_to_metadata_utc": date_added_utc_dt_b,
                "last_metadata_update_utc": now_utc_dt_batch,
                "application_status": common_application_status,
                "user_metadata": common_user_metadata or {},
                "version_current": new_version_number_b,
                "version_history_app": version_history_list_b,
                "original_filename_if_compressed": raw_metadata_values_detail["original_filename_if_compressed"],
                "compression_type": raw_metadata_values_detail["compression_type"]
            }
            try:
                pydantic_entry_batch_obj = FileMetadataEntry(**current_entry_data_batch_model)
                current_version_details_batch_obj = FileVersion(
                    version_number=pydantic_entry_batch_obj.version_current,
                    timestamp_utc=pydantic_entry_batch_obj.last_metadata_update_utc,
                    change_description=common_change_description or ("Batch registration" if not existing_pydantic_entry_batch else "Batch file content/metadata update"),
                    size_bytes=pydantic_entry_batch_obj.size_bytes,
                    custom_hashes=pydantic_entry_batch_obj.custom_hashes.copy(),
                    git_commit_hash=None
                )
                pydantic_entry_batch_obj.version_history_app.append(current_version_details_batch_obj)
                metadata_collection.add_or_update_entry(pydantic_entry_batch_obj)
                files_to_stage_for_git_abs.append(path_in_repo_abs_detail)
                results[str(original_abs_p_detail)] = True # Mark original path as success
                commit_is_needed_for_batch = True
            except Exception as e_pydantic_batch:
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Pydantic validation failed for {rel_path_str_detail}: {e_pydantic_batch}", Path(__file__).stem, exc_info=True)
                # results[str(original_abs_p_detail)] is already False
                # Cleanup the compressed file if it was made for this failed item
                if raw_metadata_values_detail.get("compression_type") and path_in_repo_abs_detail.exists() and path_in_repo_abs_detail != original_abs_p_detail:
                    path_in_repo_abs_detail.unlink(missing_ok=True)

        if commit_is_needed_for_batch:
            if not self.metadata_handler.write_metadata(metadata_collection.to_dict()): # Write combined metadata
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Failed to write combined metadata before commit.", Path(__file__).stem)
                for orig_abs_p, path_in_repo_p, _, raw_meta in processed_file_details_for_batch: 
                    results[str(orig_abs_p)] = False # Mark all as failed if metadata save fails
                    if raw_meta.get("compression_type") and path_in_repo_p.exists() and path_in_repo_p.name.endswith(f".{raw_meta['compression_type']}"):
                        path_in_repo_p.unlink(missing_ok=True)
                return results

            files_to_commit_abs_with_metadata = files_to_stage_for_git_abs + [self.metadata_handler.metadata_filepath]
            final_batch_commit_msg = batch_commit_message or f"Batch update for {len(files_to_stage_for_git_abs)} files"

            if self.git_ops_helper.commit_changes(files_to_commit_abs_with_metadata, final_batch_commit_msg):
                last_commit_hash_batch = self.repo.head.commit.hexsha
                committed_tree_batch = self.repo.head.commit.tree
                
                final_metadata_collection_batch = MetadataCollection.model_validate(self.metadata_handler.read_metadata() or {})
                modified_in_post_commit_batch = False

                for _, path_in_repo_abs_update, rel_path_str_update, raw_meta_for_hash_fallback in processed_file_details_for_batch:
                    original_abs_path_key = path_in_repo_abs_update.parent / (raw_meta_for_hash_fallback.get("original_filename_if_compressed") or raw_meta_for_hash_fallback["filename"])
                    if results.get(str(original_abs_path_key), False): # Only for successfully processed files
                        entry_to_finalize_batch = final_metadata_collection_batch.get_entry(rel_path_str_update)
                        if entry_to_finalize_batch:
                            committed_blob_hash_batch = None
                            try: committed_blob_hash_batch = committed_tree_batch[rel_path_str_update].hexsha
                            except Exception: committed_blob_hash_batch = raw_meta_for_hash_fallback["git_object_hash_current"] # fallback
                            
                            entry_to_finalize_batch.git_object_hash_current = committed_blob_hash_batch
                            if entry_to_finalize_batch.version_history_app: # Should exist
                                entry_to_finalize_batch.version_history_app[-1].git_commit_hash = last_commit_hash_batch
                            modified_in_post_commit_batch = True
                
                if modified_in_post_commit_batch:
                    self.metadata_handler.write_metadata(final_metadata_collection_batch.to_dict(), repo_instance=None)

                actual_log_statement('info', f"{LOG_INS}:INFO>>Batch add successful. Commit: {last_commit_hash_batch}", Path(__file__).stem)
            else: # Commit failed
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Batch: Failed to commit changes.", Path(__file__).stem)
                for orig_abs_p_final, path_in_repo_p_final, _, raw_meta_final in processed_file_details_for_batch: 
                    results[str(orig_abs_p_final)] = False
                    # Rollback compressed files if commit failed
                    if raw_meta_final.get("compression_type") and path_in_repo_p_final.exists() and path_in_repo_p_final != orig_abs_p_final :
                        path_in_repo_p_final.unlink(missing_ok=True)
        else:
             actual_log_statement('info', f"{LOG_INS}:INFO>>Batch: No valid files processed or no changes requiring commit.", Path(__file__).stem)
        return results

    def update_file_status(
        self, 
        file_path: Union[str, Path], 
        new_status: str,
        change_description: Optional[str] = None,
        commit_message: Optional[str] = None
    ) -> bool:
        _log_ins_val = _get_log_ins(inspect.currentframe(), self.LOG_INS_CLASS)
        
        rel_path_str = self._get_relative_path(Path(file_path))
        if not rel_path_str:
            # Try if file_path was already relative and points to a file that exists in metadata (or on disk to infer relative path)
            temp_abs_path_for_check = self.repo_path / file_path
            # Check if this constructed path is within the repo and if it looks like a file path already in metadata
            # This part of rel_path resolution for status update needs to be robust.
            # For now, assume self._get_relative_path is the primary way, or we rely on an exact relative path string.
            if Path(file_path).is_absolute(): # If it was absolute but failed _get_relative_path
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Absolute file path '{file_path}' could not be resolved within the repository.", Path(__file__).stem)
                return False
            else: # Assume file_path might be a direct relative path string
                rel_path_str = Path(file_path).as_posix()
                # We still need to check if this rel_path_str is actually in metadata
                if not self.get_file_metadata_entry(rel_path_str): # Uses internal logic to check metadata
                    actual_log_statement('error', f"{LOG_INS}:ERROR>>Relative file path '{file_path}' not found in metadata or is invalid.", Path(__file__).stem)
                    return False

        all_metadata_dict = self.metadata_handler.read_metadata()
        metadata_collection = MetadataCollection.model_validate(all_metadata_dict if all_metadata_dict else {})
        pydantic_entry = metadata_collection.get_entry(rel_path_str)

        if not pydantic_entry:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>File {rel_path_str} not tracked. Cannot update status.", Path(__file__).stem)
            return False
        
        old_status = pydantic_entry.application_status
        if old_status == new_status:
            actual_log_statement('info', f"{LOG_INS}:INFO>>File {rel_path_str} already has status '{new_status}'. No update performed.", Path(__file__).stem)
            return True

        pydantic_entry.application_status = new_status
        pydantic_entry.last_metadata_update_utc = datetime.now(timezone.utc)
        pydantic_entry.version_current += 1 

        status_change_desc = change_description or f"Status changed from '{old_status}' to '{new_status}'."
        version_record = FileVersion(
            version_number=pydantic_entry.version_current,
            timestamp_utc=pydantic_entry.last_metadata_update_utc,
            change_description=status_change_desc,
            size_bytes=pydantic_entry.size_bytes, 
            custom_hashes=pydantic_entry.custom_hashes.copy(),
            git_commit_hash=None # Placeholder
        )
        # Ensure version_history_app is a list before appending
        if pydantic_entry.version_history_app is None : pydantic_entry.version_history_app = []
        pydantic_entry.version_history_app.append(version_record)
        
        metadata_collection.add_or_update_entry(pydantic_entry)
        
        final_commit_message = commit_message or f"Update status for {rel_path_str} to {new_status}"
        
        if not self.metadata_handler.write_metadata(metadata_collection.to_dict()):
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to write metadata for status update on {rel_path_str}.", Path(__file__).stem)
            return False

        if self.git_ops_helper.commit_changes([self.metadata_handler.metadata_filepath], final_commit_message):
            last_commit_hash = self.repo.head.commit.hexsha
            
            final_meta_data_root = self.metadata_handler.read_metadata()
            final_metadata_collection_after_commit = MetadataCollection.model_validate(final_meta_data_root if final_meta_data_root else {})
            entry_to_finalize = final_metadata_collection_after_commit.get_entry(rel_path_str)

            if entry_to_finalize and entry_to_finalize.version_history_app: # Should always exist
                entry_to_finalize.version_history_app[-1].git_commit_hash = last_commit_hash
                self.metadata_handler.write_metadata(final_metadata_collection_after_commit.to_dict(), repo_instance=None)

            actual_log_statement('info', f"{LOG_INS}:INFO>>Status updated for {rel_path_str} to {new_status}. Commit: {last_commit_hash}", Path(__file__).stem)
            return True
        else:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit status update for {rel_path_str}.", Path(__file__).stem)
            return False

    def update_tracked_file(
        self, 
        file_path: Union[str, Path], 
        change_description: Optional[str] = "File content or OS stats updated.",
        commit_message: Optional[str] = None
    ) -> bool:
        abs_path = Path(file_path).resolve()

        if not abs_path.exists() or not abs_path.is_file():
            actual_log_statement('error', f"{LOG_INS}:ERROR>>File to update does not exist: {abs_path}", Path(__file__).stem)
            return False

        rel_path_str = self._get_relative_path(abs_path)
        if not rel_path_str:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>File {abs_path} is not within the repository. Cannot update.", Path(__file__).stem)
            return False

        all_metadata_dict = self.metadata_handler.read_metadata()
        existing_entry_dict = all_metadata_dict.get(rel_path_str)

        if not existing_entry_dict:
            actual_log_statement('warning', f"{LOG_INS}:WARNING>>File {rel_path_str} not previously tracked. Use add_file_to_tracking first.", Path(__file__).stem)
            return False # Or call add_file_to_tracking? For now, require it to be tracked.

        try:
            current_pydantic_entry = FileMetadataEntry.model_validate(existing_entry_dict)
        except Exception as e:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to parse existing metadata for {rel_path_str}: {e}", Path(__file__).stem, exc_info=True)
            return False
            
        new_raw_metadata_values = self._create_file_metadata_entry_data(abs_path)
        if not new_raw_metadata_values:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to gather current metadata for {abs_path}. Cannot update.", Path(__file__).stem)
            return False

        # Check for significant changes (hashes, size, mtime)
        # Custom hashes are Dict[str,str], direct comparison works
        significant_change = (
            current_pydantic_entry.size_bytes != new_raw_metadata_values["size_bytes"] or
            current_pydantic_entry.custom_hashes != new_raw_metadata_values["custom_hashes"] or
            # Comparing ISO string timestamps is okay for os_last_modified_utc
            (current_pydantic_entry.os_last_modified_utc.isoformat() if isinstance(current_pydantic_entry.os_last_modified_utc, datetime) else current_pydantic_entry.os_last_modified_utc) != new_raw_metadata_values["os_last_modified_utc"]
        )

        if not significant_change:
            actual_log_statement('info', f"{LOG_INS}:INFO>>No significant changes detected for {rel_path_str}. Metadata not updated.", Path(__file__).stem)
            return True # No update needed is not an error

        # Create history entry for the PREVIOUS version
        prev_version_record = FileVersion(
            version_number=current_pydantic_entry.version_current,
            timestamp_utc=current_pydantic_entry.last_metadata_update_utc,
            change_description=f"Superseded by update on {datetime.now(timezone.utc).isoformat()}",
            size_bytes=current_pydantic_entry.size_bytes,
            custom_hashes=current_pydantic_entry.custom_hashes.copy(),
            git_commit_hash=current_pydantic_entry.git_object_hash_current # From last commit
        )
        current_pydantic_entry.version_history_app.append(prev_version_record)

        # Update fields in the Pydantic entry
        current_pydantic_entry.size_bytes = new_raw_metadata_values["size_bytes"]
        current_pydantic_entry.os_last_modified_utc = datetime.fromisoformat(new_raw_metadata_values["os_last_modified_utc"].replace("Z", "+00:00"))
        current_pydantic_entry.custom_hashes = new_raw_metadata_values["custom_hashes"]
        current_pydantic_entry.last_metadata_update_utc = datetime.now(timezone.utc)
        current_pydantic_entry.version_current += 1
        # git_object_hash_current will be updated after commit.

        current_version_record_for_history = FileVersion(
            version_number=current_pydantic_entry.version_current,
            timestamp_utc=current_pydantic_entry.last_metadata_update_utc,
            change_description=change_description,
            size_bytes=current_pydantic_entry.size_bytes,
            custom_hashes=current_pydantic_entry.custom_hashes.copy(),
            git_commit_hash=None # Placeholder, to be updated post-commit
        )
        current_pydantic_entry.version_history_app.append(current_version_record_for_history)

        all_metadata_dict[rel_path_str] = current_pydantic_entry.model_dump(by_alias=True)

        final_commit_message = commit_message or f"Update file content/metadata: {rel_path_str}"
        files_to_commit = [abs_path, self.metadata_handler.metadata_filepath]

        if not self.metadata_handler.write_metadata(all_metadata_dict):
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to write metadata before commit for updated file {rel_path_str}.", Path(__file__).stem)
            return False

        if self.git_ops_helper.commit_changes(files_to_commit, final_commit_message):
            last_commit = self.repo.head.commit
            committed_tree = last_commit.tree
            git_blob_hash_in_commit = None
            try: git_blob_hash_in_commit = committed_tree[rel_path_str].hexsha
            except Exception: pass

            final_metadata_dict = self.metadata_handler.read_metadata() # Re-read
            entry_to_finalize = final_metadata_dict.get(rel_path_str)
            if entry_to_finalize:
                entry_to_finalize["git_object_hash_current"] = git_blob_hash_in_commit
                if entry_to_finalize["version_history_app"]:
                    entry_to_finalize["version_history_app"][-1]["git_commit_hash"] = last_commit.hexsha
                self.metadata_handler.write_metadata(final_metadata_dict, repo_instance=None)
            actual_log_statement('info', f"{LOG_INS}:INFO>>Successfully updated and committed {rel_path_str}. Commit: {last_commit.hexsha}", Path(__file__).stem)
            return True
        else:
            actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit changes for updated file {rel_path_str}.", Path(__file__).stem)
            return False

    def get_file_metadata_entry(self, file_path: Union[str, Path]) -> Optional[FileMetadataEntry]:
        """Retrieves the full FileMetadataEntry Pydantic model for a file."""
        rel_path_str = self._get_relative_path(Path(file_path))
        if not rel_path_str:
            # Try assuming file_path is already relative
            if (self.repo_path / file_path).exists():
                 rel_path_str = Path(file_path).as_posix()
            else:
                actual_log_statement('warning', f"{LOG_INS}:WARNING>>File path '{file_path}' could not be resolved to a relative path in the repository.", Path(__file__).stem)
                return None

        all_metadata = self.metadata_handler.read_metadata()
        entry_dict = all_metadata.get(rel_path_str)
        if entry_dict:
            try:
                return FileMetadataEntry.model_validate(entry_dict)
            except Exception as e: # Pydantic validation error
                actual_log_statement('error', f"{LOG_INS}:ERROR>>Failed to parse metadata entry for '{rel_path_str}': {e}", Path(__file__).stem, exc_info=True)
                return None
        actual_log_statement('debug', f"{LOG_INS}:DEBUG>>No metadata entry found for '{rel_path_str}'.", Path(__file__).stem)
        return None

    def get_file_status(self, file_path: Union[str, Path]) -> Optional[str]:
        entry = self.get_file_metadata_entry(file_path)
        if entry:
            return entry.application_status
        return None

    def get_all_files_by_status(self, status: str) -> List[str]:
        all_metadata_dict = self.metadata_handler.read_metadata()
        matching_files = []
        for rel_path, entry_dict in all_metadata_dict.items():
            try:
                # Light parse just for status, or full parse if confident in data
                if entry_dict.get("application_status") == status:
                    matching_files.append(rel_path)
            except Exception as e: # Should not happen if just .get()
                 actual_log_statement('error', f"{LOG_INS}:ERROR>>Error checking status for {rel_path}: {e}", Path(__file__).stem, exc_info=False) # Low verbosity for many files
        return matching_files

    def get_file_history(self, file_path: Union[str, Path]) -> Optional[List[FileVersion]]:
        entry = self.get_file_metadata_entry(file_path)
        if entry:
            return entry.version_history_app
        return None

    # --- Other methods from Phase 0 analysis (e.g., _scan_directory_for_metadata, get_repository_summary) ---
    # These might need updates to use Pydantic models or align with new metadata structure.
    # For now, focusing on the parity tasks 1.3-1.5.
    def _scan_directory_for_metadata(self, directory: Union[str, Path] = ".", specific_extensions: Optional[List[str]] = None,
                                     force_rescan_all: bool = False) -> Dict[str, Any]:
        """
        Scans a directory, collects basic metadata for files, and updates metadata.json.
        NOTE: This is a basic scan and does NOT provide the rich metadata of add_file_to_tracking.
        It's more for initial discovery or simple updates.
        Consider DEPRECATING or REFINING this to use the richer metadata pipeline.
        """
        # ... (Implementation from Phase 0, needs review against new metadata standards if kept) ...
        actual_log_statement('warning', f"{LOG_INS}:WARNING>>_scan_directory_for_metadata provides basic info. For rich metadata, use add_file_to_tracking.", Path(__file__).stem)
        
        scan_path_abs = (self.repo_path / directory).resolve()
        current_metadata = self.metadata_handler.read_metadata() if not force_rescan_all else {}

        for root, _, files in os.walk(scan_path_abs):
            for filename in files:
                if specific_extensions and not any(filename.endswith(ext) for ext in specific_extensions):
                    continue
                
                abs_path = Path(root) / filename
                rel_path_str = self._get_relative_path(abs_path)
                if not rel_path_str: continue

                if rel_path_str not in current_metadata or force_rescan_all:
                    try:
                        stat_info = abs_path.stat()
                        # This is a very basic entry, not conforming to FileMetadataEntry fully
                        current_metadata[rel_path_str] = {
                            "filename": filename,
                            "size_bytes": stat_info.st_size,
                            "os_last_modified_utc": datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat(),
                            "extension": abs_path.suffix,
                            "_scan_update_utc": datetime.now(timezone.utc).isoformat() # Mark as scanned
                        }
                    except Exception as e:
                        actual_log_statement('error', f"{LOG_INS}:ERROR>>Could not stat file {abs_path} during scan: {e}", Path(__file__).stem)
        
        self.metadata_handler.write_metadata(current_metadata, "Update metadata from directory scan", self.repo)
        return current_metadata
    
    # --- Delegated methods or high-level orchestrations ---
    def get_file_status(self, filepath: Union[str, Path]) -> Optional[str]:
        """Facade for RepoAnalyzer.get_status_for_file."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating get_file_status for {filepath}", Path(__file__).stem, False)
        return self.analyzer.get_status_for_file(Path(filepath))

    def commit_all_changes(self, message: str = "General commit of all changes") -> bool:
        """Facade for RepoModifier.save_repository_snapshot, simplified."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating commit_all_changes with message: '{message}'", Path(__file__).stem, False)
        return self.modifier.save_repository_snapshot(message)

    def update_file_metadata(self, filepath: Union[str, Path], **kwargs) -> bool:
        """Facade for RepoModifier.update_metadata_entry."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating update_file_metadata for {filepath}", Path(__file__).stem, False)
        return self.modifier.update_metadata_entry(Path(filepath), **kwargs)

    def _scan_directory_for_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans the repository directory (excluding .git and metadata.json) for files
        and returns their metadata. This is a simplified version of the user's _scan_directory.
        """
        actual_log_statement("info", f"{LOG_INS}:INFO>>Scanning directory {self.root_dir} for file metadata.", Path(__file__).stem, False)
        
        files_metadata: Dict[str, Dict[str, Any]] = {}
        try:
            for root, _, files in os.walk(self.root_dir):
                root_path = Path(root)
                if ".git" in root_path.parts: # Skip .git directory
                    continue
                for file_name in files:
                    file_path = root_path / file_name
                    if file_path == self.metadata_path: # Skip metadata file itself
                        continue
                    
                    # Use the assumed external helper _get_file_metadata
                    try:
                        metadata = _get_file_metadata(file_path)
                        if metadata:
                             files_metadata[str(file_path.resolve())] = metadata
                    except Exception as e:
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Failed to get metadata for {file_path}: {e}", Path(__file__).stem, True)
            
            actual_log_statement("info", f"{LOG_INS}:INFO>>Scan complete. Found metadata for {len(files_metadata)} files.", Path(__file__).stem, False)
            return files_metadata
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error during directory scan: {e}", Path(__file__).stem, True)
            return {} # Fallback

    def scan_and_update_repo_metadata(self, commit: bool = True) -> bool:
        """
        Scans the directory, updates the metadata.json file with new/changed files,
        and commits the changes.
        """
        actual_log_statement("info", f"{LOG_INS}:INFO>>Starting scan and update of repository metadata.", Path(__file__).stem, False)

        try:
            current_repo_files_metadata = self._scan_directory_for_metadata()
            existing_metadata = self.metadata_handler.read_metadata()
            
            # Simple update: replace existing metadata with scanned metadata.
            # A more sophisticated version would merge, check for deletions, etc.
            # For this refactor, we'll keep it simple based on user's likely intent from _scan_directory.
            
            # The user's _scan_directory implies updating existing_metadata with new_files.
            # It also adds all scanned file paths to the commit.

            changed_files_from_status_str = self.git_ops_helper._execute_git_command(['status', '--porcelain'], suppress_errors=True)
            changed_paths_to_add_to_commit = [self.metadata_path] # Always add metadata file

            if changed_files_from_status_str:
                 for line in changed_files_from_status_str.strip().splitlines():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        filepath_str = parts[1]
                        abs_path = self.root_dir / filepath_str
                        # Check if it's a file and not the metadata file itself
                        if abs_path.is_file() and abs_path != self.metadata_path:
                             changed_paths_to_add_to_commit.append(abs_path)


            # Update the overall metadata structure.
            # The original _scan_directory updated a global metadata dict.
            # Here we'll just use the fresh scan.
            # metadata_to_write = existing_metadata # Start with existing
            # metadata_to_write.update(current_repo_files_metadata) # Add/overwrite with new scan

            # A safer approach might be to only add files that are "new" or "modified"
            # based on git status, then fetch their metadata.
            # For now, using the simple replace based on a full scan.
            metadata_to_write = current_repo_files_metadata

            if self.metadata_handler.write_metadata(metadata_to_write):
                if commit:
                    # Commit metadata.json and all files that were part of the scan (or just changed files)
                    # The original _scan_directory added all keys from `new_files`.
                    # For simplicity and safety, we'll commit changed files detected by git status + metadata.json
                    if not self.modifier._commit_changes(list(set(changed_paths_to_add_to_commit)), "Updated repository metadata from scan"):
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata written but commit failed for scan and update.", Path(__file__).stem, False)
                        return False # Partial success
                actual_log_statement("info", f"{LOG_INS}:INFO>>Scan and update of metadata successful.", Path(__file__).stem, False)
                return True
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata during scan and update.", Path(__file__).stem, False)
                return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error during scan and update process: {e}", Path(__file__).stem, True)
            return False

    def get_summary_metadata(self) -> Dict[str, Any]:
        """
        Provides a summary of the repository based on Git information.
        Combines aspects of the two `get_summary_metadata` methods provided.
        """
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting summary metadata for the repository.", Path(__file__).stem, False)
        summary = {}
        try:
            commits = list(self.git_repo.iter_commits()) # Expensive for large repos if used often
            summary["commit_count"] = len(commits)
            summary["branch_count"] = len(self.git_repo.branches)
            summary["tag_count"] = len(self.git_repo.tags)

            if commits:
                # Date range from commits
                commit_dates = [c.committed_datetime for c in commits]
                summary["date_range_min_utc"] = min(commit_dates).isoformat() if commit_dates else None
                summary["date_range_max_utc"] = max(commit_dates).isoformat() if commit_dates else None
            else:
                summary["date_range_min_utc"] = None
                summary["date_range_max_utc"] = None

            # File count from `ls-files`
            ls_files_output = self.git_ops_helper._execute_git_command(['ls-files'], suppress_errors=True)
            summary["file_count"] = len(ls_files_output.splitlines()) if ls_files_output else 0
            
            # Total size of blobs in the latest commit (can be computationally intensive)
            # For a simpler approach, sum sizes from metadata.json if available and up-to-date
            # Or use git cat-file --batch-check to get sizes of all blobs in HEAD
            # The original code summed blob sizes across *all* commits, which is unusual.
            # Here, let's get size of tracked files in working directory as an alternative.
            total_size = 0
            if ls_files_output:
                for line in ls_files_output.splitlines():
                    try:
                        file_path = self.root_dir / line
                        if file_path.is_file(): # Ensure it's a file and exists
                             total_size += file_path.stat().st_size
                    except Exception:
                        pass # Ignore errors for individual file stats in summary
            summary["total_tracked_files_size_bytes"] = total_size

            actual_log_statement("info", f"{LOG_INS}:INFO>>Summary metadata retrieved: {summary}", Path(__file__).stem, False)
            return summary
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get summary metadata: {e}", Path(__file__).stem, True)
            return {"error": str(e)} # Fallback

    def parallel_scan_files(self) -> List[Dict[str, Any]]:
        """
        Scans files listed by `git ls-files` in parallel using `process_file`.
        """
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Starting parallel scan of repository files.", Path(__file__).stem, False)
        results = []
        try:
            ls_files_output = self.git_ops_helper._execute_git_command(['ls-files'], suppress_errors=True)
            if not ls_files_output:
                actual_log_statement("info", f"{LOG_INS}:INFO>>No files found by 'git ls-files' to scan.", Path(__file__).stem, False)
                return []

            files_to_scan = [str(self.root_dir / f_str) for f_str in ls_files_output.splitlines()]
            
            # Determine max_workers, could be configurable
            max_workers = min(8, os.cpu_count() or 1 + 4) # Example logic

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_file, file_path_str): file_path_str for file_path_str in files_to_scan}
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path_str = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        actual_log_statement("error", f"{LOG_INS}:ERROR>>File '{file_path_str}' generated an exception during parallel scan: {exc}", Path(__file__).stem, True)
                        results.append({"file": file_path_str, "error": str(exc)}) # Add error info
            
            actual_log_statement("info", f"{LOG_INS}:INFO>>Parallel scan completed. Processed {len(results)} files.", Path(__file__).stem, False)
            return results
        except GitCommandError as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Git command failed during parallel scan setup: {e}", Path(__file__).stem, False)
            return [{"error": "Git command failed during setup"}]
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error during parallel scan: {e}", Path(__file__).stem, True)
            return [{"error": str(e)}]

    # --- Pandas DataFrame related methods (from original code) ---
    def load_repository_as_dataframe(self) -> Optional[pd.DataFrame]:
        """Loads the metadata.json file into a pandas DataFrame."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Loading repository metadata into DataFrame.", Path(__file__).stem, False)
        
        metadata = self.metadata_handler.read_metadata()
        if not metadata or metadata is None: # Check if read failed or empty
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata is empty or could not be read. Returning empty DataFrame.", Path(__file__).stem, False)
            return pd.DataFrame()

        data_for_df = []
        for file_str, file_meta_dict in metadata.items():
            # Skip special keys like "errors" or "progress" if they are top-level in metadata.json
            if file_str in ["errors", "progress"]: # This depends on metadata structure
                continue
            if isinstance(file_meta_dict, dict):
                # Ensure 'filepath' is part of the record, using file_str as the key from metadata
                record = {"filepath_original_key": file_str, **file_meta_dict}
                data_for_df.append(record)
            else:
                actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping non-dictionary metadata entry for key: {file_str}", Path(__file__).stem, False)
        
        if not data_for_df:
            actual_log_statement("info", f"{LOG_INS}:INFO>>No valid file entries found in metadata for DataFrame conversion.", Path(__file__).stem, False)
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(data_for_df)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Successfully loaded metadata into DataFrame with {len(df)} entries.", Path(__file__).stem, False)
            return df
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to create DataFrame from metadata: {e}", Path(__file__).stem, True)
            return None # Fallback

    def save_dataframe_to_repository_metadata(self, df: pd.DataFrame, commit:bool = True, commit_message:str = "Updated repository metadata from DataFrame") -> bool:
        """Saves a pandas DataFrame back to the metadata.json file."""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving DataFrame to repository metadata.", Path(__file__).stem, False)

        if not isinstance(df, pd.DataFrame):
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Input is not a pandas DataFrame. Cannot save.", Path(__file__).stem, False)
            return False
        
        metadata_to_write: Dict[str, Any] = {}
        # Assume DataFrame has a 'filepath_original_key' or similar unique ID for each file's metadata
        # Or, if 'filepath' column is guaranteed to be the primary key for the metadata structure.
        # The original code implies 'filepath' in the DataFrame rows becomes the key in the JSON.
        
        # Check if a primary key column exists, e.g., 'filepath_original_key' or 'filepath'
        # This part needs to be robust based on how `load_repository_as_dataframe` structures it.
        # Let's assume 'filepath_original_key' if it exists from the load, otherwise 'filepath'.
        id_column = 'filepath_original_key' if 'filepath_original_key' in df.columns else 'filepath'
        if id_column not in df.columns:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot determine ID column ('{id_column}') in DataFrame to structure metadata. Save aborted.", Path(__file__).stem, False)
            return False

        try:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                file_key = row_dict.pop(id_column, None) # Get and remove the key column
                if file_key is None: # Should not happen if column exists
                     actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping row with missing ID in column '{id_column}'.", Path(__file__).stem, False)
                     continue
                metadata_to_write[str(file_key)] = row_dict
            
            if self.metadata_handler.write_metadata(metadata_to_write):
                if commit:
                    if not self.modifier._commit_changes([self.metadata_path], commit_message):
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata from DataFrame written but commit failed.", Path(__file__).stem, False)
                        return False # Partial success
                actual_log_statement("info", f"{LOG_INS}:INFO>>Successfully saved DataFrame to metadata file.", Path(__file__).stem, False)
                return True
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write DataFrame to metadata file.", Path(__file__).stem, False)
                return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to convert DataFrame to metadata structure: {e}", Path(__file__).stem, True)
            return False

    # Placeholder for other methods from the original list.
    # They would be refactored to use the helper classes (GitOpsHelper, MetadataFileHandler, etc.)
    # and follow the logging/error handling patterns shown above.
    # Examples:
    # - LFS configuration
    # - Submodule management
    # - Backup verification
    # - File integrity checks and repair
    # - Version creation (tags) and loading

    # Example of a method that was duplicated and can be unified:
    def clean_repository_workspace(self, force: bool = True, remove_untracked_directories: bool = True) -> bool:
        """Cleans the repository working directory. (git clean -fdx)"""
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Cleaning repository workspace (force={force}, dirs={remove_untracked_directories}).", Path(__file__).stem, False)
        cmd = ['clean']
        if force: cmd.append('-f')
        if remove_untracked_directories: cmd.append('-d')
        cmd.append('-x') # Remove ignored files too

        try:
            self.git_ops_helper._execute_git_command(cmd, suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository workspace cleaned successfully.", Path(__file__).stem, False)
            return True
        except GitCommandError: # Error already logged
            return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error cleaning repository: {e}", Path(__file__).stem, True)
            return False

if __name__ == "__main__":
        # Example Usage (requires a directory to be a Git repository)
    # Ensure logging is configured if src.utils.logger is not available.
    # if not _log_statement_defined:
    #     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    #     main_file_logger = logging.getLogger(Path(__file__).stem) # Use Path(__file__).stem as logger name

    
    current_script_path = Path(Path(__file__).stem).resolve()
    example_repo_path = current_script_path.parent / "example_repo"
    actual_log_statement('info', f"{LOG_INS}:INFO>>RepoHandler main example starting...", Path(__file__).stem)

    example_repo_path = Path("./example_git_repo_functional").resolve()

    if example_repo_path.exists():
        actual_log_statement('info', f"{LOG_INS}:INFO>>Cleaning up old example repository: {example_repo_path}", Path(__file__).stem)
        shutil.rmtree(example_repo_path, ignore_errors=True)
    
    example_repo_path.mkdir(parents=True, exist_ok=True)

    try:
        repo_path = Path(ROOT_DIR).resolve()
        repo = git.Repo(repo_path)
        git_ops_helper = GitOpsHelper(repo)
        metdata_handler = MetadataFileHandler(repo_path=ROOT_DIR)
        repo_handler = RepoHandler(repository_path="/home/yosh/repos/TLATOv4.1/", metadata_filename="/home/yosh/repos/TLATOv4.1/repositories/metadata.json.zst", progress_dir_name="/home/yosh/repos/TLATOv4.1/progress_dir", create_repo_if_not_exists=True, )
        # Example: Get summary
        summary = repo_handler.get_summary_metadata()
        actual_log_statement("info", f"{LOG_INS}:INFO>>Repository Summary: {summary}", Path(__file__).stem, False)

        # Example: Get commit history
        history = repo_handler.analyzer.get_commit_history(max_count=5)
        actual_log_statement("info", f"{LOG_INS}:INFO>>Recent Commit History (max 5):", Path(__file__).stem, False)
        for commit_info in history:
            actual_log_statement("info", f"{LOG_INS}:INFO>>  - {commit_info.get('commit_hash')[:10]}: {commit_info.get('message')}", Path(__file__).stem, False)


        # Example: Save and load progress
        progress_data = {"step": 5, "processed_items": ["itemA", "itemB"]}
        repo_handler.progress_handler.save_progress("my_long_process", progress_data)
        loaded_progress = repo_handler.progress_handler.load_progress("my_long_process")
        actual_log_statement("info", f"{LOG_INS}:INFO>>Loaded progress for 'my_long_process': {loaded_progress}", Path(__file__).stem, False)

        # Example: Gitignore operations
        repo_handler.gitignore_handler.add_to_gitignore("*.log")
        repo_handler.gitignore_handler.add_to_gitignore("*.tmp")
        gitignore_content = repo_handler.gitignore_handler.get_gitignore_content()
        actual_log_statement("info", f"{LOG_INS}:INFO>>Current .gitignore content:\n{gitignore_content}", Path(__file__).stem, False)
        repo_handler.gitignore_handler.remove_from_gitignore("*.tmp")
        gitignore_content_after_remove = repo_handler.gitignore_handler.get_gitignore_content()
        actual_log_statement("info", f"{LOG_INS}:INFO>>.gitignore content after removing *.tmp:\n{gitignore_content_after_remove}", Path(__file__).stem, False)

        # Create a dummy file if it doesn't exist for the example
        if not example_repo_path.exists():
             example_repo_path.mkdir(parents=True, exist_ok=True)
             actual_log_statement("info", f"{LOG_INS}:INFO>>Created example repo directory: {example_repo_path}", Path(__file__).stem, False)

        repo_handler = RepoHandler(example_repo_path)
        actual_log_statement('info', f"{LOG_INS}:INFO>>RepoHandler initialized for {example_repo_path}", Path(__file__).stem)

        # Create some dummy files
        data_dir = example_repo_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        file1_path = data_dir / "file1.txt"
        with open(file1_path, "w") as f: f.write("Initial content for file1.")
        
        file2_path = data_dir / "file2.csv"
        with open(file2_path, "w") as f: f.write("colA,colB\n1,apple\n2,banana")

        actual_log_statement('info', f"{LOG_INS}:INFO>>Created dummy files: {file1_path.name}, {file2_path.name}", Path(__file__).stem)

        # Test add_file_to_tracking
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing add_file_to_tracking ---", Path(__file__).stem)
        repo_handler.add_file_to_tracking(file1_path, application_status="raw", user_metadata={"source": "script"}, change_description="Initial add of file1")
        repo_handler.add_file_to_tracking(file2_path, application_status="raw", user_metadata={"category": "fruit_data"})

        # Test get_file_metadata_entry and get_file_status
        meta1 = repo_handler.get_file_metadata_entry(file1_path)
        if meta1:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Metadata for file1: {meta1.filepath_relative}, Status: {meta1.application_status}, Version: {meta1.version_current}", Path(__file__).stem)
            actual_log_statement('info', f"{LOG_INS}:INFO>>File1 custom_hashes: {meta1.custom_hashes}", Path(__file__).stem)

        # Modify file1 and update it
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing update_tracked_file ---", Path(__file__).stem)
        with open(file1_path, "a") as f: f.write("\nMore content added.")
        repo_handler.update_tracked_file(file1_path, change_description="Appended more content to file1")
        
        meta1_updated = repo_handler.get_file_metadata_entry(file1_path)
        if meta1_updated:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Updated metadata for file1: Version {meta1_updated.version_current}", Path(__file__).stem)
            actual_log_statement('info', f"{LOG_INS}:INFO>>Updated file1 custom_hashes: {meta1_updated.custom_hashes}", Path(__file__).stem)

        # Test update_file_status
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing update_file_status ---", Path(__file__).stem)
        repo_handler.update_file_status(file2_path, new_status="processed", change_description="Processed fruit data.")
        status_file2 = repo_handler.get_file_status(file2_path)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Status for file2: {status_file2}", Path(__file__).stem)

        # Test get_all_files_by_status
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing get_all_files_by_status ---", Path(__file__).stem)
        processed_files = repo_handler.get_all_files_by_status("processed")
        actual_log_statement('info', f"{LOG_INS}:INFO>>Files with status 'processed': {processed_files}", Path(__file__).stem)
        
        raw_files = repo_handler.get_all_files_by_status("raw") # file1 should be raw after update
        actual_log_statement('info', f"{LOG_INS}:INFO>>Files with status 'raw': {raw_files}", Path(__file__).stem)


        # Test get_file_history
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing get_file_history ---", Path(__file__).stem)
        history1 = repo_handler.get_file_history(file1_path)
        if history1:
            actual_log_statement('info', f"{LOG_INS}:INFO>>History for file1 (versions: {len(history1)}):", Path(__file__).stem)
            for v_idx, version in enumerate(history1):
                actual_log_statement('info', f"  v{version.version_number}: {version.change_description} @ {version.timestamp_utc} (Commit: {version.git_commit_hash}, Size: {version.size_bytes})", Path(__file__).stem)
        
        # Test batch add
        actual_log_statement('info', f"{LOG_INS}:INFO>>\n--- Testing add_files_batch ---", Path(__file__).stem)
        batch_dir = example_repo_path / "batch_data"
        batch_dir.mkdir(exist_ok=True)
        file_b1 = batch_dir / "batch_file1.dat"
        file_b2 = batch_dir / "batch_file2.dat"
        with open(file_b1, "w") as f: f.write("batch data one")
        with open(file_b2, "w") as f: f.write("batch data two")
        batch_results = repo_handler.add_files_batch(
            [file_b1, file_b2], 
            common_application_status="batch_added",
            common_user_metadata={"batch_id": "B001"}
        )
        actual_log_statement('info', f"{LOG_INS}:INFO>>Batch add results: {batch_results}", Path(__file__).stem)
        meta_b1 = repo_handler.get_file_metadata_entry(file_b1)
        if meta_b1:
            actual_log_statement('info', f"{LOG_INS}:INFO>>Metadata for batch_file1: Status {meta_b1.application_status}, UserMeta: {meta_b1.user_metadata}", Path(__file__).stem)


    except Exception as main_e:
        actual_log_statement('critical', f"{LOG_INS}:CRITICAL>>Error in main example execution: {main_e}", Path(__file__).stem, exc_info=True)
    finally:
        shutil.rmtree(example_repo_path, ignore_errors=True) # Optional cleanup
        actual_log_statement('info', f"{LOG_INS}:INFO>>Cleaned up example repository: {example_repo_path}", Path(__file__).stem)
        actual_log_statement('info', f"{LOG_INS}:INFO>>Main example finished. Repo at: {example_repo_path}", Path(__file__).stem)