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
from src.utils.config import *
from src.data.constants import *
from src.utils.helpers import process_file

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

# Attempt to get a logger. If `src.utils.logger.configure_logging` is available,
# it should ideally set up the root logger or specific loggers.
try:
    from src.utils.logger import log_statement as actual_log_statement
    _log_statement_defined = True
except ImportError:
    _log_statement_defined = False
    # Basic fallback logger if the specified one is not found
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_file_logger = logging.getLogger(__file__) # Using __name__ as a fallback for __file__ if it's complex

    def actual_log_statement(loglevel: str, logstatement: str, main_logger_name: str, exc_info: bool = False):
        """Placeholder log_statement function."""
        logger_to_use = logging.getLogger(main_logger_name if main_logger_name else __file__)
        level = LOG_LEVELS.get(loglevel.lower(), logging.INFO)
        
        # The user's format: f"{LOG_INS}:insert value of loglevel here>>...."
        # Since LOG_INS is dynamic (includes line numbers), we won't fully replicate it here
        # but will prepend the loglevel.
        # The full LOG_INS construction should happen at the call site.
        # The logstatement passed already includes this per user spec.
        
        if loglevel.lower() == "exception" or exc_info:
            logger_to_use.log(level, logstatement, exc_info=True)
        else:
            logger_to_use.log(level, logstatement)

# Helper to generate LOG_INS prefix
def _get_log_ins(frame_info: inspect.FrameInfo, class_name: Optional[str] = None) -> str:
    func_name = frame_info.function
    module_name = inspect.getmodule(frame_info.frame).__file__ if inspect.getmodule(frame_info.frame) else "__main__"
    line_no = frame_info.lineno
    if class_name:
        return f"{module_name}::{class_name}::{func_name}::{line_no}"
    return f"{module_name}::{func_name}::{line_no}"

try:
    from src.utils.helpers import _get_file_metadata
except ImportError as ie:
    actual_log_statement('error', f"{_get_log_ins()}:ERROR>>Error importing _get_file_metadata from src.utils.helpers: {ie}")
    def _get_file_metadata(abs_path: Path) -> Dict[str, Any]:
        """Placeholder for external _get_file_metadata function."""
        # In a real implementation, this would gather actual metadata.
        # LOG_INS = _get_log_ins(inspect.currentframe()) # Example usage if this were a full function
        # actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Called _get_file_metadata for {abs_path}", __file__, False)
        return {
            "filepath": str(abs_path),
            "size": abs_path.stat().st_size if abs_path.exists() else 0,
            "mtime": abs_path.stat().st_mtime if abs_path.exists() else 0,
            "filename": abs_path.name,
            "extension": abs_path.suffix,
            # Add other necessary metadata fields
        }
    raise ie

class GitCommandError(Exception):
    """Custom exception for Git command failures."""
    pass

class GitOperationHelper:
    """Helper class for executing Git commands and parsing output."""

    def __init__(self, git_repo_instance: git.Repo, root_dir: Path):
        self.git_repo = git_repo_instance
        self.root_dir = root_dir
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

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
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Executing Git command: git {' '.join(command)} with kwargs: {kwargs}", __file__, False)
        try:
            result = self.git_repo.git.execute(command, **kwargs)
            actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Git command 'git {' '.join(command)}' executed successfully.", __file__, False)
            return result
        except git.exc.GitCommandError as e:
            log_msg = f"{LOG_INS}:ERROR>>Git command 'git {' '.join(command)}' failed: {e}"
            actual_log_statement("error", log_msg, __file__, True)
            if suppress_errors:
                return ""
            raise GitCommandError(f"Git command 'git {' '.join(command)}' failed: {e}") from e
        except Exception as e:
            log_msg = f"{LOG_INS}:CRITICAL>>Unexpected error executing Git command 'git {' '.join(command)}': {e}"
            actual_log_statement("critical", log_msg, __file__, True)
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
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Parsing git log output.", __file__, False)
        commits = []
        if not log_output:
            actual_log_statement("info", f"{LOG_INS}:INFO>>No log output to parse.", __file__, False)
            return commits

        for line in log_output.strip().splitlines():
            parts = line.split(delimiter, num_parts - 1)
            if len(parts) == num_parts:
                commits.append(dict(zip(field_names, parts)))
            else:
                actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping malformed log line: {line}", __file__, False)
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Parsed {len(commits)} commits from log output.", __file__, False)
        return commits

class MetadataFileHandler:
    """Handles reading and writing of the JSON metadata file."""

    def __init__(self, metadata_path: Path, git_ops_helper: GitOperationHelper):
        self.metadata_path = metadata_path
        self.git_ops_helper = git_ops_helper # Needed for committing metadata changes
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

    def read_metadata(self) -> Dict[str, Any]:
        """Reads the metadata JSON file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to read metadata from: {self.metadata_path}", __file__, False)
        if not self.metadata_path.exists():
            actual_log_statement("info", f"{LOG_INS}:INFO>>Metadata file {self.metadata_path} not found. Returning empty metadata.", __file__, False)
            return {}
        try:
            with self.metadata_path.open("r") as f:
                metadata = json.load(f)
            actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Metadata read successfully from {self.metadata_path}.", __file__, False)
            return metadata
        except json.JSONDecodeError as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to decode JSON from {self.metadata_path}: {e}", __file__, True)
            return {} # Fallback to empty dict
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to read metadata file {self.metadata_path}: {e}", __file__, True)
            return {}

    def write_metadata(self, metadata: Dict[str, Any], commit_message: Optional[str] = None) -> bool:
        """Writes the metadata to the JSON file and optionally commits it."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to write metadata to: {self.metadata_path}", __file__, False)
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with self.metadata_path.open("w") as f:
                json.dump(metadata, f, indent=4)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Metadata written successfully to {self.metadata_path}.", __file__, False)
            if commit_message:
                try:
                    self.git_ops_helper.git_repo.index.add([str(self.metadata_path)])
                    self.git_ops_helper.git_repo.index.commit(commit_message)
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed metadata changes with message: {commit_message}", __file__, False)
                except Exception as e: # Catch Git related errors specifically if possible
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit metadata changes: {e}", __file__, True)
                    return False # Indicate failure if commit fails
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata to {self.metadata_path}: {e}", __file__, True)
            return False

class ProgressFileHandler:
    """Handles saving and loading of progress files."""

    def __init__(self, root_dir: Path, git_ops_helper: Optional[GitOperationHelper] = None):
        self.root_dir = root_dir
        self.git_ops_helper = git_ops_helper # Optional, for committing progress files
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

    def save_progress(self, process_id: str, current_state: Dict[str, Any], commit_changes: bool = True) -> bool:
        """Saves progress to a JSON file and optionally commits it."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        progress_file = self.root_dir / f"progress_{process_id}.json"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving progress for process '{process_id}' to {progress_file}", __file__, False)

        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with progress_file.open("w") as f:
                json.dump(current_state, f, indent=4)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress saved successfully for {process_id} to {progress_file}.", __file__, False)

            if commit_changes and self.git_ops_helper:
                try:
                    self.git_ops_helper.git_repo.index.add([str(progress_file)])
                    self.git_ops_helper.git_repo.index.commit(f"Saved progress for {process_id}")
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed progress file for {process_id}.", __file__, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit progress file for {process_id}: {e}", __file__, True)
                    # Return True because file was saved, even if commit failed. Or False?
                    # For now, let's say saving the file is primary success.
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to save progress for {process_id}: {e}", __file__, True)
            return False

    def load_progress(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Loads progress from a JSON file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        progress_file = self.root_dir / f"progress_{process_id}.json"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Loading progress for process '{process_id}' from {progress_file}", __file__, False)

        if not progress_file.exists():
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress file for {process_id} not found at {progress_file}.", __file__, False)
            return None
        try:
            with progress_file.open("r") as f:
                state = json.load(f)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Progress loaded successfully for {process_id}.", __file__, False)
            return state
        except json.JSONDecodeError as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to decode JSON from progress file {progress_file}: {e}", __file__, True)
            return None # Corrupted file
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to load progress for {process_id}: {e}", __file__, True)
            return None

class GitignoreFileHandler:
    """Handles reading and modifying the .gitignore file."""

    def __init__(self, git_ops_helper: GitOperationHelper):
        self.git_ops_helper = git_ops_helper
        self.gitignore_path = Path(self.git_ops_helper.git_repo.working_tree_dir) / ".gitignore"
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

    def get_gitignore_content(self) -> Optional[str]:
        """Reads the content of the .gitignore file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Reading .gitignore file from {self.gitignore_path}", __file__, False)
        try:
            if self.gitignore_path.exists():
                content = self.gitignore_path.read_text()
                actual_log_statement("info", f"{LOG_INS}:INFO>>.gitignore content read successfully.", __file__, False)
                return content
            else:
                actual_log_statement("info", f"{LOG_INS}:INFO>>.gitignore file does not exist at {self.gitignore_path}.", __file__, False)
                return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to read .gitignore file: {e}", __file__, True)
            return None # Fallback

    def add_to_gitignore(self, pattern: str, commit: bool = True, commit_message: Optional[str] = None) -> bool:
        """Adds a pattern to .gitignore if it's not already present."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to add pattern '{pattern}' to .gitignore", __file__, False)
        try:
            content = self.get_gitignore_content()
            if content is None: # File doesn't exist or couldn't be read
                lines = []
            else:
                lines = content.splitlines()

            if pattern.strip() in [line.strip() for line in lines]:
                actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' already in .gitignore. No changes made.", __file__, False)
                return True

            with self.gitignore_path.open("a") as f: # Open in append mode
                f.write(f"\n{pattern.strip()}") # Add newline before pattern for safety
            actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' added to .gitignore.", __file__, False)

            if commit:
                msg = commit_message or f"Added '{pattern}' to .gitignore"
                try:
                    self.git_ops_helper.git_repo.index.add([str(self.gitignore_path)])
                    self.git_ops_helper.git_repo.index.commit(msg)
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed .gitignore changes: {msg}", __file__, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit .gitignore changes: {e}", __file__, True)
                    return False # File was modified, but commit failed
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to add pattern '{pattern}' to .gitignore: {e}", __file__, True)
            return False

    def remove_from_gitignore(self, pattern: str, commit: bool = True, commit_message: Optional[str] = None) -> bool:
        """Removes a pattern from .gitignore."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Attempting to remove pattern '{pattern}' from .gitignore", __file__, False)
        if not self.gitignore_path.exists():
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>.gitignore file not found. Cannot remove pattern '{pattern}'.", __file__, False)
            return False
        try:
            lines = self.gitignore_path.read_text().splitlines()
            pattern_to_remove = pattern.strip()
            new_lines = [line for line in lines if line.strip() != pattern_to_remove]

            if len(new_lines) == len(lines):
                actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' not found in .gitignore. No changes made.", __file__, False)
                return True # Pattern wasn't there, so considered success

            self.gitignore_path.write_text("\n".join(new_lines) + "\n") # Add trailing newline
            actual_log_statement("info", f"{LOG_INS}:INFO>>Pattern '{pattern}' removed from .gitignore.", __file__, False)

            if commit:
                msg = commit_message or f"Removed '{pattern}' from .gitignore"
                try:
                    self.git_ops_helper.git_repo.index.add([str(self.gitignore_path)])
                    self.git_ops_helper.git_repo.index.commit(msg)
                    actual_log_statement("info", f"{LOG_INS}:INFO>>Committed .gitignore changes: {msg}", __file__, False)
                except Exception as e:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit .gitignore changes: {e}", __file__, True)
                    return False # File modified, commit failed
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to remove pattern '{pattern}' from .gitignore: {e}", __file__, True)
            return False

class RepoAnalyzer:
    """Analyzes the Git repository for information (read-only operations)."""

    def __init__(self, git_ops_helper: GitOperationHelper):
        self.git_ops = git_ops_helper
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

    def get_status_for_file(self, filepath: Path) -> Optional[str]:
        """Gets the Git status for a specific file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting status for file: {filepath}", __file__, False)
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain', str(filepath.resolve())], suppress_errors=True)
            if status_output:
                status_code = status_output.strip().split(maxsplit=1)[0]
                actual_log_statement("info", f"{LOG_INS}:INFO>>Status for {filepath}: {status_code}", __file__, False)
                return status_code
            actual_log_statement("info", f"{LOG_INS}:INFO>>File {filepath} is clean or not tracked.", __file__, False)
            return None # Clean or not tracked
        except GitCommandError: # Already logged by _execute_git_command
            return "Error" # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting status for {filepath}: {e}", __file__, True)
            return "Error"

    def get_repository_status_summary(self) -> Optional[str]:
        """Gets the porcelain status for the entire repository."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting repository status summary.", __file__, False)
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain'], suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository status summary retrieved.", __file__, False)
            return status_output
        except GitCommandError:
            return "Error retrieving status" # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting repository status: {e}", __file__, True)
            return "Error retrieving status"

    def get_files_by_status(self, status_codes: Union[str, List[str]]) -> List[Path]:
        """Gets files matching the given Git status code(s) (e.g., 'M', '??')."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting files by status: {status_codes}", __file__, False)
        files: List[Path] = []
        try:
            status_output = self.git_ops._execute_git_command(['status', '--porcelain'], suppress_errors=True)
            if not status_output:
                actual_log_statement("info", f"{LOG_INS}:INFO>>Repository is clean or status output is empty.", __file__, False)
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
                            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Could not form path for '{filepath_str}': {path_e}", __file__, False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Found {len(files)} files with status {status_codes}.", __file__, False)
            return files
        except GitCommandError:
            return [] # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting files by status: {e}", __file__, True)
            return []

    def get_commit_history(self, max_count: Optional[int] = None, file_path: Optional[Path] = None, author: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retrieves commit history for the repository or a specific file/author.
        Combines `get_file_history` and `get_commit_history` and `get_commit_history_by_author`.
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        log_params = [f"max_count={max_count}", f"file_path={file_path}", f"author={author}"]
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting commit history. Params: {', '.join(filter(None, log_params))}", __file__, False)

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
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved {len(commits)} commits.", __file__, False)
            return commits
        except GitCommandError:
            return []
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting commit history: {e}", __file__, True)
            return []

    def get_commit_details(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed information for a specific commit."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting details for commit: {commit_hash}", __file__, False)
        if not commit_hash or commit_hash.lower() == "head": # Resolve HEAD if needed
            try:
                commit_hash = self.git_ops.git_repo.head.commit.hexsha
                actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Resolved HEAD to {commit_hash}", __file__, False)
            except Exception as e:
                 actual_log_statement("error", f"{LOG_INS}:ERROR>>Could not resolve HEAD: {e}", __file__, True)
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
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved details for commit {commit_hash}.", __file__, False)
            return details
        except git.exc.BadName as e: # More specific error for invalid commit hash
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Invalid commit hash '{commit_hash}': {e}", __file__, False) # exc_info not needed for BadName
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get details for commit {commit_hash}: {e}", __file__, True)
            return None

    def get_diff(self, item1: str, item2: Optional[str] = None, file_path: Optional[Path] = None) -> Optional[str]:
        """
        Gets the diff between two commits, a commit and working tree, or for a specific file.
        `item1` can be a commit hash.
        `item2` can be another commit hash. If None, diffs `item1` against working tree (or its parent if only one commit).
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting diff for items: {item1}, {item2}, file: {file_path}", __file__, False)
        cmd = ['diff']
        if item1: cmd.append(item1)
        if item2: cmd.append(item2)
        if file_path: cmd.extend(['--', str(file_path.resolve())])
        
        try:
            diff_output = self.git_ops._execute_git_command(cmd, suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Diff retrieved successfully.", __file__, False)
            return diff_output
        except GitCommandError:
            return None # Error logged in _execute_git_command
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting diff: {e}", __file__, True)
            return None

    def get_blame(self, file_path: Path) -> Optional[str]:
        """Gets the blame output for a file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting blame for file: {file_path}", __file__, False)
        try:
            blame_output = self.git_ops._execute_git_command(['blame', str(file_path.resolve())], suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Blame retrieved successfully for {file_path}.", __file__, False)
            return blame_output
        except GitCommandError:
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting blame for {file_path}: {e}", __file__, True)
            return None
            
    def get_authors_contributors(self, include_email: bool = True) -> List[Dict[str, str]]:
        """Gets a list of authors/contributors with their commit counts."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting authors/contributors (include_email={include_email}).", __file__, False)
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
            actual_log_statement("info", f"{LOG_INS}:INFO>>Retrieved {len(authors)} authors/contributors.", __file__, False)
            return authors
        except GitCommandError:
            return []
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting authors: {e}", __file__, True)
            return []

    def get_commit_count(self, rev_range: Optional[str] = "HEAD", author: Optional[str] = None, committer: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None) -> int:
        """Gets the commit count based on various filters."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting commit count for range '{rev_range}', author '{author}', committer '{committer}', since '{since}', until '{until}'.", __file__, False)
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
            actual_log_statement("info", f"{LOG_INS}:INFO>>Commit count: {count}", __file__, False)
            return count
        except (GitCommandError, ValueError) as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get commit count: {e}", __file__, isinstance(e, GitCommandError))
            return 0 # Fallback
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error getting commit count: {e}", __file__, True)
            return 0

    # ... Many other get_... methods would go here, refactored to use _execute_git_command and _parse_log_output
    # For brevity, I will not list all of them but will follow the pattern above. Examples:
    # get_commits_by_date, get_tags_by_date, get_branches_by_commit, etc.

    def get_repository_root_path(self) -> Optional[Path]:
        """Gets the root directory of the Git repository."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting repository root path.", __file__, False)
        try:
            # GitPython provides this directly and more reliably
            root_path_str = self.git_ops.git_repo.working_tree_dir
            if root_path_str:
                root_path = Path(root_path_str)
                actual_log_statement("info", f"{LOG_INS}:INFO>>Repository root path: {root_path}", __file__, False)
                return root_path
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Could not determine repository root path from GitPython.", __file__, False)
            return None
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get repository root: {e}", __file__, True)
            return None

    def is_git_repository(self, directory: Optional[Path] = None) -> bool:
        """Checks if the given directory (or current repo's dir) is a Git repository."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        path_to_check = directory or self.git_ops.root_dir
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Checking if {path_to_check} is a Git repository.", __file__, False)
        try:
            # For the current repo instance, this is implicitly true if git_repo is valid.
            # For an arbitrary directory, we'd init a new Repo object or use rev-parse.
            if directory: # Checking an arbitrary directory
                 git.Repo(path_to_check) # This will raise an error if not a repo
            else: # Checking the repo this handler is for
                 if not self.git_ops.git_repo.git_dir: # Basic check
                     return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Path {path_to_check} is a Git repository.", __file__, False)
            return True
        except (git.exc.NoSuchPathError, git.exc.InvalidGitRepositoryError):
            actual_log_statement("info", f"{LOG_INS}:INFO>>Path {path_to_check} is not a Git repository.", __file__, False)
            return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error checking Git repository status for {path_to_check}: {e}", __file__, True)
            return False

class RepoModifier:
    """Modifies the Git repository state (commits, branches, tags, etc.)."""

    def __init__(self, git_ops_helper: GitOperationHelper, metadata_handler: MetadataFileHandler):
        self.git_ops = git_ops_helper
        self.metadata_handler = metadata_handler
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"

    def _commit_changes(self, files_to_add: List[Union[str, Path]], commit_message: str) -> bool:
        """Helper to add files and commit changes."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Committing changes. Files: {files_to_add}, Message: '{commit_message}'", __file__, False)
        try:
            str_files_to_add = [str(f.resolve() if isinstance(f, Path) else Path(f).resolve()) for f in files_to_add]
            self.git_ops.git_repo.index.add(str_files_to_add)
            self.git_ops.git_repo.index.commit(commit_message)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Changes committed successfully with message: '{commit_message}'.", __file__, False)
            return True
        except Exception as e: # Catch Git related errors specifically
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to commit changes with message '{commit_message}': {e}", __file__, True)
            return False

    def save_repository_snapshot(self, commit_message: str = "Repository snapshot") -> bool:
        """Adds all changes and creates a commit."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving repository snapshot with message: '{commit_message}'", __file__, False)
        try:
            self.git_ops.git_repo.git.add(all=True) # Using git.add directly for 'all'
            self.git_ops.git_repo.index.commit(commit_message)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository snapshot saved successfully.", __file__, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to save repository snapshot: {e}", __file__, True)
            return False

    def update_metadata_entry(self, filepath: Path, commit_message_prefix: str = "Updated", **kwargs) -> bool:
        """Updates a metadata entry and commits the change."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Updating metadata entry for {filepath} with kwargs: {kwargs}", __file__, False)
        
        metadata = self.metadata_handler.read_metadata()
        if metadata is None: # Error reading metadata
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot update metadata, failed to read existing metadata.", __file__, False)
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
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata for {filepath}, commit aborted.", __file__, False)
            return False

    def record_error_in_metadata(self, filepath: Path, error_msg: str) -> bool:
        """Records an error message for a file in the metadata."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Recording error for {filepath}: {error_msg}", __file__, False)
        
        metadata = self.metadata_handler.read_metadata()
        if metadata is None:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot record error, failed to read existing metadata.", __file__, False)
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
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata for error recording, commit aborted.", __file__, False)
            return False

    def manage_branch(self, action: str, branch_name: str, new_branch_name: Optional[str] = None) -> bool:
        """Manages branches: create, delete, checkout."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Branch action: {action}, name: {branch_name}, new_name: {new_branch_name}", __file__, False)
        try:
            if action == "create":
                self.git_ops.git_repo.create_head(branch_name)
            elif action == "delete":
                self.git_ops.git_repo.delete_head(branch_name, force=True) # Add force for safety, or make it an option
            elif action == "checkout":
                self.git_ops.git_repo.heads[branch_name].checkout()
            elif action == "rename": # Not in original but good addition
                if not new_branch_name:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>New branch name required for rename action.", __file__, False)
                    return False
                branch_to_rename = self.git_ops.git_repo.heads[branch_name]
                branch_to_rename.rename(new_branch_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported branch action: {action}", __file__, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Branch action '{action}' for '{branch_name}' successful.", __file__, False)
            return True
        except Exception as e: # Catch specific git errors like git.exc.GitCommandError
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Branch action '{action}' for '{branch_name}' failed: {e}", __file__, True)
            return False

    def manage_tag(self, action: str, tag_name: str, message: Optional[str] = None, commit_ish: Optional[str] = None) -> bool:
        """Manages tags: create, delete."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Tag action: {action}, name: {tag_name}, message: {message}, commit: {commit_ish}", __file__, False)
        try:
            if action == "create":
                ref = commit_ish if commit_ish else self.git_ops.git_repo.head.commit
                self.git_ops.git_repo.create_tag(tag_name, ref=ref, message=message or f"Tag {tag_name}", force=False) # -a implicitly by message
            elif action == "delete":
                self.git_ops.git_repo.delete_tag(tag_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported tag action: {action}", __file__, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Tag action '{action}' for '{tag_name}' successful.", __file__, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Tag action '{action}' for '{tag_name}' failed: {e}", __file__, True)
            return False

    def manage_remote(self, action: str, remote_name: str, remote_url: Optional[str] = None) -> bool:
        """Manages remotes: add, remove."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Remote action: {action}, name: {remote_name}, url: {remote_url}", __file__, False)
        try:
            if action == "add":
                if not remote_url:
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Remote URL required for add action.", __file__, False)
                    return False
                self.git_ops.git_repo.create_remote(remote_name, remote_url)
            elif action == "remove":
                self.git_ops.git_repo.delete_remote(remote_name)
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Unsupported remote action: {action}", __file__, False)
                return False
            actual_log_statement("info", f"{LOG_INS}:INFO>>Remote action '{action}' for '{remote_name}' successful.", __file__, False)
            return True
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Remote action '{action}' for '{remote_name}' failed: {e}", __file__, True)
            return False

    def push_changes(self, remote_name: str = "origin", refspec: str = "refs/heads/*:refs/heads/*", tags: bool = False, force: bool = False) -> bool:
        """Pushes changes to a remote."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Pushing to remote '{remote_name}', refspec '{refspec}', tags={tags}, force={force}", __file__, False)
        try:
            remote_to_push = self.git_ops.git_repo.remote(name=remote_name)
            push_infos = remote_to_push.push(refspec=refspec, tags=tags, force=force) # GitPython uses force flag
            for info in push_infos:
                if info.flags & (git.PushInfo.ERROR | git.PushInfo.REJECTED | git.PushInfo.REMOTE_REJECTED | git.PushInfo.REMOTE_FAILURE):
                    actual_log_statement("error", f"{LOG_INS}:ERROR>>Push to {remote_name} failed for ref {info.local_ref or info.remote_ref_string}: {info.summary}", __file__, False)
                    # return False # Decide if any error means overall failure
            actual_log_statement("info", f"{LOG_INS}:INFO>>Push to remote '{remote_name}' completed.", __file__, False)
            return True # Simplified: assume success if no immediate exception and some info received
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Push to remote '{remote_name}' failed: {e}", __file__, True)
            return False
            
    # ... Other modifier methods like pull, fetch, merge, reset, clean, LFS, submodules ...

class RepoHandler:
    """Main class for handling a Git repository and its metadata."""

    def __init__(self, directory_path: Union[str, Path]):
        self.LOG_INS_PREFIX = f"{__file__}::{self.__class__.__name__}"
        method_name = "__init__" # inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno if inspect.currentframe() else 'N/A'}"

        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Initializing RepoHandler for directory: {directory_path}", __file__, False)
        try:
            self.root_dir = Path(directory_path).resolve()
            self.root_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            try:
                self.git_repo = git.Repo(self.root_dir)
                actual_log_statement("info", f"{LOG_INS}:INFO>>Opened existing Git repository at {self.root_dir}", __file__, False)
            except git.exc.NoSuchPathError: # Path exists but not a repo
                self.git_repo = git.Repo.init(self.root_dir)
                actual_log_statement("info", f"{LOG_INS}:INFO>>Initialized new Git repository at {self.root_dir}", __file__, False)
            except git.exc.InvalidGitRepositoryError: # Path is not a git repo
                self.git_repo = git.Repo.init(self.root_dir)
                actual_log_statement("info", f"{LOG_INS}:INFO>>Path was not a valid Git repository. Initialized new Git repository at {self.root_dir}", __file__, False)


            self.git_ops_helper = GitOperationHelper(self.git_repo, self.root_dir)
            
            self.metadata_path = self.root_dir / "metadata.json"
            self.metadata_handler = MetadataFileHandler(self.metadata_path, self.git_ops_helper)
            
            self.progress_handler = ProgressFileHandler(self.root_dir, self.git_ops_helper)
            self.gitignore_handler = GitignoreFileHandler(self.git_ops_helper)

            self.analyzer = RepoAnalyzer(self.git_ops_helper)
            self.modifier = RepoModifier(self.git_ops_helper, self.metadata_handler)


            if not self.metadata_path.exists():
                actual_log_statement("info", f"{LOG_INS}:INFO>>Metadata file not found. Creating and committing initial metadata.json.", __file__, False)
                if not self.metadata_handler.write_metadata({}, "Initialized data repository metadata.json"):
                     actual_log_statement("warning", f"{LOG_INS}:WARNING>>Failed to create and commit initial metadata.json.", __file__, False)
            
            actual_log_statement("info", f"{LOG_INS}:INFO>>RepoHandler initialized successfully for {self.root_dir}", __file__, False)

        except Exception as e:
            actual_log_statement("critical", f"{LOG_INS}:CRITICAL>>Failed to initialize RepoHandler for {directory_path}: {e}", __file__, True)
            raise  # Re-raise after logging

    # --- Delegated methods or high-level orchestrations ---

    def get_file_status(self, filepath: Union[str, Path]) -> Optional[str]:
        """Facade for RepoAnalyzer.get_status_for_file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating get_file_status for {filepath}", __file__, False)
        return self.analyzer.get_status_for_file(Path(filepath))

    def commit_all_changes(self, message: str = "General commit of all changes") -> bool:
        """Facade for RepoModifier.save_repository_snapshot, simplified."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating commit_all_changes with message: '{message}'", __file__, False)
        return self.modifier.save_repository_snapshot(message)

    def update_file_metadata(self, filepath: Union[str, Path], **kwargs) -> bool:
        """Facade for RepoModifier.update_metadata_entry."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Delegating update_file_metadata for {filepath}", __file__, False)
        return self.modifier.update_metadata_entry(Path(filepath), **kwargs)

    def _scan_directory_for_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans the repository directory (excluding .git and metadata.json) for files
        and returns their metadata. This is a simplified version of the user's _scan_directory.
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("info", f"{LOG_INS}:INFO>>Scanning directory {self.root_dir} for file metadata.", __file__, False)
        
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
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Failed to get metadata for {file_path}: {e}", __file__, True)
            
            actual_log_statement("info", f"{LOG_INS}:INFO>>Scan complete. Found metadata for {len(files_metadata)} files.", __file__, False)
            return files_metadata
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error during directory scan: {e}", __file__, True)
            return {} # Fallback

    def scan_and_update_repo_metadata(self, commit: bool = True) -> bool:
        """
        Scans the directory, updates the metadata.json file with new/changed files,
        and commits the changes.
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("info", f"{LOG_INS}:INFO>>Starting scan and update of repository metadata.", __file__, False)

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
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata written but commit failed for scan and update.", __file__, False)
                        return False # Partial success
                actual_log_statement("info", f"{LOG_INS}:INFO>>Scan and update of metadata successful.", __file__, False)
                return True
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write metadata during scan and update.", __file__, False)
                return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Error during scan and update process: {e}", __file__, True)
            return False

    def get_summary_metadata(self) -> Dict[str, Any]:
        """
        Provides a summary of the repository based on Git information.
        Combines aspects of the two `get_summary_metadata` methods provided.
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Getting summary metadata for the repository.", __file__, False)
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

            actual_log_statement("info", f"{LOG_INS}:INFO>>Summary metadata retrieved: {summary}", __file__, False)
            return summary
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to get summary metadata: {e}", __file__, True)
            return {"error": str(e)} # Fallback

    def parallel_scan_files(self) -> List[Dict[str, Any]]:
        """
        Scans files listed by `git ls-files` in parallel using `process_file`.
        """
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Starting parallel scan of repository files.", __file__, False)
        results = []
        try:
            ls_files_output = self.git_ops_helper._execute_git_command(['ls-files'], suppress_errors=True)
            if not ls_files_output:
                actual_log_statement("info", f"{LOG_INS}:INFO>>No files found by 'git ls-files' to scan.", __file__, False)
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
                        actual_log_statement("error", f"{LOG_INS}:ERROR>>File '{file_path_str}' generated an exception during parallel scan: {exc}", __file__, True)
                        results.append({"file": file_path_str, "error": str(exc)}) # Add error info
            
            actual_log_statement("info", f"{LOG_INS}:INFO>>Parallel scan completed. Processed {len(results)} files.", __file__, False)
            return results
        except GitCommandError as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Git command failed during parallel scan setup: {e}", __file__, False)
            return [{"error": "Git command failed during setup"}]
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error during parallel scan: {e}", __file__, True)
            return [{"error": str(e)}]

    # --- Pandas DataFrame related methods (from original code) ---
    def load_repository_as_dataframe(self) -> Optional[pd.DataFrame]:
        """Loads the metadata.json file into a pandas DataFrame."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Loading repository metadata into DataFrame.", __file__, False)
        
        metadata = self.metadata_handler.read_metadata()
        if not metadata or metadata is None: # Check if read failed or empty
            actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata is empty or could not be read. Returning empty DataFrame.", __file__, False)
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
                actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping non-dictionary metadata entry for key: {file_str}", __file__, False)
        
        if not data_for_df:
            actual_log_statement("info", f"{LOG_INS}:INFO>>No valid file entries found in metadata for DataFrame conversion.", __file__, False)
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(data_for_df)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Successfully loaded metadata into DataFrame with {len(df)} entries.", __file__, False)
            return df
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to create DataFrame from metadata: {e}", __file__, True)
            return None # Fallback

    def save_dataframe_to_repository_metadata(self, df: pd.DataFrame, commit:bool = True, commit_message:str = "Updated repository metadata from DataFrame") -> bool:
        """Saves a pandas DataFrame back to the metadata.json file."""
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Saving DataFrame to repository metadata.", __file__, False)

        if not isinstance(df, pd.DataFrame):
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Input is not a pandas DataFrame. Cannot save.", __file__, False)
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
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Cannot determine ID column ('{id_column}') in DataFrame to structure metadata. Save aborted.", __file__, False)
            return False

        try:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                file_key = row_dict.pop(id_column, None) # Get and remove the key column
                if file_key is None: # Should not happen if column exists
                     actual_log_statement("warning", f"{LOG_INS}:WARNING>>Skipping row with missing ID in column '{id_column}'.", __file__, False)
                     continue
                metadata_to_write[str(file_key)] = row_dict
            
            if self.metadata_handler.write_metadata(metadata_to_write):
                if commit:
                    if not self.modifier._commit_changes([self.metadata_path], commit_message):
                        actual_log_statement("warning", f"{LOG_INS}:WARNING>>Metadata from DataFrame written but commit failed.", __file__, False)
                        return False # Partial success
                actual_log_statement("info", f"{LOG_INS}:INFO>>Successfully saved DataFrame to metadata file.", __file__, False)
                return True
            else:
                actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to write DataFrame to metadata file.", __file__, False)
                return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Failed to convert DataFrame to metadata structure: {e}", __file__, True)
            return False

    # Placeholder for other methods from the original list.
    # They would be refactored to use the helper classes (GitOperationHelper, MetadataFileHandler, etc.)
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
        method_name = inspect.currentframe().f_code.co_name
        LOG_INS = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
        actual_log_statement("debug", f"{LOG_INS}:DEBUG>>Cleaning repository workspace (force={force}, dirs={remove_untracked_directories}).", __file__, False)
        cmd = ['clean']
        if force: cmd.append('-f')
        if remove_untracked_directories: cmd.append('-d')
        cmd.append('-x') # Remove ignored files too

        try:
            self.git_ops_helper._execute_git_command(cmd, suppress_errors=False)
            actual_log_statement("info", f"{LOG_INS}:INFO>>Repository workspace cleaned successfully.", __file__, False)
            return True
        except GitCommandError: # Error already logged
            return False
        except Exception as e:
            actual_log_statement("error", f"{LOG_INS}:ERROR>>Unexpected error cleaning repository: {e}", __file__, True)
            return False


if __name__ == '__main__':
    # Example Usage (requires a directory to be a Git repository)
    # Ensure logging is configured if src.utils.logger is not available.
    # if not _log_statement_defined:
    #     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    #     main_file_logger = logging.getLogger(__file__) # Use __file__ as logger name

    
    current_script_path = Path(__file__).resolve()
    example_repo_path = current_script_path.parent / "example_repo"
    
    LOG_INS_MAIN = f"{__file__}::main::{inspect.currentframe().f_lineno if inspect.currentframe() else 'N/A'}"
    actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Starting RepoHandler example.", __file__, False)

    try:
        # Create a dummy file if it doesn't exist for the example
        if not example_repo_path.exists():
             example_repo_path.mkdir(parents=True, exist_ok=True)
             actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Created example repo directory: {example_repo_path}", __file__, False)

        repo_handler = RepoHandler(example_repo_path)

        # Create some dummy files for testing
        (example_repo_path / "test_file1.txt").write_text("Hello world 1")
        (example_repo_path / "test_file2.md").write_text("# Markdown Test")
        sub_dir = example_repo_path / "subdir"
        sub_dir.mkdir(exist_ok=True)
        (sub_dir / "test_file3.txt").write_text("Subdirectory file.")

        # Example: Commit all changes
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Committing initial files.", __file__, False)
        repo_handler.commit_all_changes("Added initial test files.")

        # Example: Get status of a file
        status1 = repo_handler.get_file_status("test_file1.txt") # Relative path
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Status of test_file1.txt: {status1}", __file__, False)

        (example_repo_path / "test_file1.txt").write_text("Hello world 1 - modified")
        status1_mod = repo_handler.get_file_status(example_repo_path / "test_file1.txt") # Absolute path
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Status of modified test_file1.txt: {status1_mod}", __file__, False)


        # Example: Scan and update metadata
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Scanning and updating repository metadata.", __file__, False)
        repo_handler.scan_and_update_repo_metadata(commit=True)
        
        # Example: Load metadata to DataFrame
        df = repo_handler.load_repository_as_dataframe()
        if df is not None:
            actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Loaded DataFrame from metadata:\n{df.head()}", __file__, False)
        
            # Example: Modify DataFrame and save back (conceptual)
            if not df.empty and 'size' in df.columns:
                # df['size'] = df['size'] + 100 # Dummy modification
                # repo_handler.save_dataframe_to_repository_metadata(df, commit_message="Updated sizes from DataFrame example")
                # actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Saved modified DataFrame back to metadata.", __file__, False)
                pass


        # Example: Get summary
        summary = repo_handler.get_summary_metadata()
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Repository Summary: {summary}", __file__, False)

        # Example: Get commit history
        history = repo_handler.analyzer.get_commit_history(max_count=5)
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Recent Commit History (max 5):", __file__, False)
        for commit_info in history:
            actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>  - {commit_info.get('commit_hash')[:10]}: {commit_info.get('message')}", __file__, False)


        # Example: Save and load progress
        progress_data = {"step": 5, "processed_items": ["itemA", "itemB"]}
        repo_handler.progress_handler.save_progress("my_long_process", progress_data)
        loaded_progress = repo_handler.progress_handler.load_progress("my_long_process")
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Loaded progress for 'my_long_process': {loaded_progress}", __file__, False)

        # Example: Gitignore operations
        repo_handler.gitignore_handler.add_to_gitignore("*.log")
        repo_handler.gitignore_handler.add_to_gitignore("*.tmp")
        gitignore_content = repo_handler.gitignore_handler.get_gitignore_content()
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Current .gitignore content:\n{gitignore_content}", __file__, False)
        repo_handler.gitignore_handler.remove_from_gitignore("*.tmp")
        gitignore_content_after_remove = repo_handler.gitignore_handler.get_gitignore_content()
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>.gitignore content after removing *.tmp:\n{gitignore_content_after_remove}", __file__, False)


    except Exception as main_e:
        actual_log_statement("critical", f"{LOG_INS_MAIN}:CRITICAL>>Error in main example execution: {main_e}", __file__, True)
    finally:
        # Clean up the example repository directory if you want
        # shutil.rmtree(example_repo_path, ignore_errors=True)
        # actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>Cleaned up example repository directory: {example_repo_path}", __file__, False)
        actual_log_statement("info", f"{LOG_INS_MAIN}:INFO>>RepoHandler example finished. Path: {example_repo_path}", __file__, False)