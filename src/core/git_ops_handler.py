"""
Git Operations Module
Handles all Git-related operations separately from repository management
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import git
from git import Repo, Blob
from git.exc import InvalidGitRepositoryError, NoSuchPathError, GitCommandError
import pandas as pd

from src.utils.logger import log_statement
from src.data.constants import *

LOG_INS = f"{__file__}:{__name__}:"

class GitOpsHelper:
    """Helper class for executing Git commands and parsing output."""
    
    def __init__(self,
                 repo_path: Path,
                 create_if_missing: bool = False, 
                 repo: Optional[Repo] = None, 
                 root_dir: Path = ROOT_DIR):
        self.repo_path = Path(repo_path).resolve()
        self.path = self.repo_path  # Maintain compatibility with both versions
        self.git_dir = self.path / ".git"
        self.root_dir = root_dir
        self.git_module = git
        self.create_if_missing = create_if_missing
        self.repo = None
        self.is_new_repo = False
        self.gitignore_handler = None  # Will be initialized after repo is set up
        self.l_ins = LOG_INS  # Instance-specific log prefix
        
        # Initialize or load the repository
        self._initialize_repository()
        
        # Set up gitignore if repo is valid
        if self.repo:
            self.gitignore_path = self.path / GITIGNORE_FILENAME
            self._ensure_gitignore()

    def _initialize_repository(self):
        """Initialize or load Git repository with comprehensive error handling."""
        if self.git_dir.exists() and self.git_dir.is_dir():
            log_statement('info', f"{LOG_INS}:INFO>>Git directory exists at {self.git_dir}.", Path(__file__).stem)
            try:
                self.repo = Repo(self.path)
                log_statement('info', f"{LOG_INS}:INFO>>Git repository loaded successfully.", Path(__file__).stem)
            except InvalidGitRepositoryError:
                log_statement('error', f"{LOG_INS}:ERROR>>Invalid Git repository at {self.git_dir}.", Path(__file__).stem)
                if self.create_if_missing:
                    self._init_repo()
                else:
                    self.repo = None
                    raise GitCommandError(f"Invalid Git repository at {self.git_dir}.")
        elif self.create_if_missing:
            log_statement('info', f"{LOG_INS}:INFO>>Git directory does not exist at {self.git_dir}. Creating new repository.", Path(__file__).stem)
            self.path.mkdir(parents=True, exist_ok=True)
            self._init_repo()
        else:
            log_statement('error', f"{LOG_INS}:ERROR>>Git directory does not exist at {self.git_dir} and create_if_missing is False.", Path(__file__).stem)
            self.repo = None
            raise GitCommandError(f"Git directory does not exist at {self.git_dir}.")

    def _init_repo(self):
        """Internal method to initialize a new Git repository."""
        try:
            self.repo = Repo.init(str(self.path))
            self.is_new_repo = True
            log_statement('info', f"{LOG_INS}:INFO>>Successfully initialized new Git repository at: {self.path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to initialize new Git repository at {self.path}: {e}", Path(__file__).stem, exc_info=True)
            self.repo = None
            self.is_new_repo = False
            raise

    def _ensure_gitignore(self):
        """Ensure .gitignore file exists and is properly configured."""
        if not self.gitignore_path.exists():
            try:
                with open(self.gitignore_path, "w") as f:
                    f.write("# Gitignore file created by TLATO GitOpsHelper\n")
                    f.write(f"{METADATA_FILENAME}\n")
                log_statement('info', f"{LOG_INS}:INFO>>Created .gitignore file at {self.gitignore_path}", Path(__file__).stem, False)
                # Initialize gitignore handler after creating the file
                if self.gitignore_handler is None:
                    self.gitignore_handler = GitignoreHandler(self.repo, git_ops=self)
                    self.gitignore_handler.add_to_gitignore([METADATA_FILENAME])
            except IOError as e:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to create .gitignore file at {self.gitignore_path}: {e}", Path(__file__).stem, True)
        else:
            log_statement('debug', f"{LOG_INS}:DEBUG>>.gitignore file already exists at {self.gitignore_path}", Path(__file__).stem, False)
            # Initialize gitignore handler for existing file

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.repo is not None


    # Line 450 - Class Method: save
    def save(self, save_path: Optional[Path] = None):
        """
        Save repository to file
        
        Args:
            save_path: Optional path to save to (defaults to self.repo_path)
        """
        save_path = save_path or self.repo_path
        
        if self.df is None:
            log_statement('warning', "No DataFrame to save", __file__)
            return
            
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary file
            temp_path = save_path.with_suffix(f'{save_path.suffix}.tmp_{int(time.time())}')
            
            # Save based on compression
            if save_path.suffix == '.zst':
                self._save_compressed_csv(self.df, temp_path)
            else:
                self.df.to_csv(temp_path, index=False)
            
            # Atomic move
            shutil.move(str(temp_path), str(save_path))
            log_statement('info', f"Repository saved to {save_path}", __file__)
            
            # Update cache
            cache_key = f"df_{save_path}"
            self._cache.put(cache_key, self.df.copy())
            
        except Exception as e:
            log_statement('error', f"Error saving repository: {e}", __file__)
            if temp_path.exists():
                temp_path.unlink()
            raise

    # Line 515 - Class Method: scan_and_update
    def scan_and_update(self, base_dir: Optional[Path] = None):
        """
        Scan directory and update repository
        
        Args:
            base_dir: Directory to scan (defaults to self.data_path)
        """
        base_dir = Path(base_dir) if base_dir else self.data_path
        
        if not base_dir or not base_dir.exists():
            log_statement('error', f"Invalid scan directory: {base_dir}", __file__)
            return
            
        log_statement('info', f"Starting scan of {base_dir}", __file__)
        
        # Get current files in repository
        existing_files = set()
        if self.df is not None and not self.df.empty:
            existing_files = set(self.df[COL_FILEPATH].tolist())
        
        # Scan directory
        new_files = []
        updated_files = []
        
        for filepath in self._scan_directory(base_dir):
            filepath_str = str(filepath)
            
            if filepath_str in existing_files:
                # Check if file has changed
                if self._has_file_changed(filepath):
                    updated_files.append(filepath)
            else:
                new_files.append(filepath)
        
        # Process new and updated files
        if new_files or updated_files:
            self._process_file_updates(new_files, updated_files)
            self.save()
        
        log_statement('info', f"Scan complete. New: {len(new_files)}, Updated: {len(updated_files)}", __file__)

    # Line 560 - Class Method: _scan_directory
    def _scan_directory(self, directory: Path) -> List[Path]:
        """Recursively scan directory for files"""
        files = []
        
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    # Skip hidden files and repository files
                    if item.name.startswith('.'):
                        continue
                    if 'data_repository_' in item.name:
                        continue
                    
                    # Check file extension
                    if item.suffix.lower() in ACCEPTED_FILE_TYPES:
                        files.append(item)
        except Exception as e:
            log_statement('error', f"Error scanning directory: {e}", __file__)
        
        return files

    # Line 585 - Class Method: _has_file_changed
    def _has_file_changed(self, filepath: Path) -> bool:
        """Check if file has changed since last scan"""
        try:
            filepath_str = str(filepath)
            
            # Get stored file info
            file_info = self.df[self.df[COL_FILEPATH] == filepath_str].iloc[0]
            
            # Check modification time
            current_mtime = filepath.stat().st_mtime
            stored_mtime = file_info[COL_MTIME]
            
            if isinstance(stored_mtime, pd.Timestamp):
                stored_mtime = stored_mtime.timestamp()
            
            return abs(current_mtime - stored_mtime) > 1.0
            
        except Exception as e:
            log_statement('debug', f"Error checking file change: {e}", __file__)
            return True

    # Line 610 - Class Method: _process_file_updates
    def _process_file_updates(self, new_files: List[Path], updated_files: List[Path]):
        """Process new and updated files"""
        all_files = new_files + updated_files
        
        if not all_files:
            return
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self._get_file_metadata, f): f for f in all_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                filepath = futures[future]
                try:
                    metadata = future.result()
                    if metadata:
                        self._update_file_entry(filepath, metadata)
                except Exception as e:
                    log_statement('error', f"Error processing {filepath}: {e}", __file__)

    # Line 635 - Class Method: _get_file_metadata
    def _get_file_metadata(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Get metadata for a file"""
        try:
            stat = filepath.stat()
            
            metadata = {
                COL_FILEPATH: str(filepath),
                COL_FILENAME: filepath.name,
                COL_SIZE: stat.st_size,
                COL_MTIME: stat.st_mtime,
                COL_CTIME: stat.st_ctime,
                COL_EXTENSION: filepath.suffix.lower().lstrip('.'),
                COL_HASH: generate_data_hash(filepath) or '',
                COL_STATUS: STATUS_DISCOVERED,
                COL_LAST_UPDATED: time.time()
            }
            
            # Add optional columns
            for col in self.expected_columns:
                if col not in metadata:
                    if col in self.timestamp_columns:
                        metadata[col] = pd.NaT
                    elif self.columns_schema.get(col) == 'Int64':
                        metadata[col] = pd.NA
                    else:
                        metadata[col] = ''
            
            return metadata
            
        except Exception as e:
            log_statement('error', f"Error getting metadata for {filepath}: {e}", __file__)
            return None

    # Line 670 - Class Method: _update_file_entry
    def _update_file_entry(self, filepath: Path, metadata: Dict[str, Any]):
        """Update or add file entry in repository"""
        filepath_str = str(filepath)
        
        with self.lock:
            if self.df is None:
                self._initialize_empty_dataframe()
            
            # Check if file exists in repository
            mask = self.df[COL_FILEPATH] == filepath_str
            
            if mask.any():
                # Update existing entry
                for col, value in metadata.items():
                    if col in self.df.columns:
                        self.df.loc[mask, col] = value
            else:
                # Add new entry
                new_row = pd.DataFrame([metadata])
                self.df = pd.concat([self.df, new_row], ignore_index=True)

    # Line 695 - Class Method: update_entry
    def update_entry(self, filepath: Union[str, Path], **kwargs):
        """
        Update a file entry in the repository
        
        Args:
            filepath: Path to the file
            **kwargs: Column values to update
        """
        filepath_str = str(filepath)
        
        with self.lock:
            if self.df is None:
                log_statement('warning', "No DataFrame to update", __file__)
                return
            
            mask = self.df[COL_FILEPATH] == filepath_str
            
            if not mask.any():
                log_statement('warning', f"File not found in repository: {filepath_str}", __file__)
                return
            
            # Update columns
            for col, value in kwargs.items():
                if col in self.df.columns:
                    self.df.loc[mask, col] = value
                else:
                    log_statement('warning', f"Column {col} not in repository schema", __file__)

    # Line 725 - Class Method: get_files_by_status
    def get_files_by_status(self, statuses: Union[str, List[str]], 
                            base_dir: Optional[Path] = None) -> List[Path]:
        """
        Get files with specific status(es)
        
        Args:
            statuses: Status or list of statuses to filter by
            base_dir: Optional base directory filter
            
        Returns:
            List of file paths matching the criteria
        """
        if self.df is None or self.df.empty:
            return []
        
        if isinstance(statuses, str):
            statuses = [statuses]
        
        # Filter by status
        mask = self.df[COL_STATUS].isin(statuses)
        
        # Apply base directory filter if provided
        if base_dir:
            base_dir_str = str(base_dir)
            mask &= self.df[COL_FILEPATH].str.startswith(base_dir_str)
        
        # Get file paths
        file_paths = self.df[mask][COL_FILEPATH].tolist()
        
        return [Path(p) for p in file_paths]

    # Line 760 - Class Method: get_summary_metadata
    def get_summary_metadata(self) -> Dict[str, Any]:
        """Get summary metadata for the repository"""
        if self.df is None or self.df.empty:
            return {
                INDEX_META_FILE_COUNT: 0,
                INDEX_META_TOTAL_SIZE: 0,
                INDEX_META_MIN_MTIME: None,
                INDEX_META_MAX_MTIME: None
            }
        
        try:
            return {
                INDEX_META_FILE_COUNT: len(self.df),
                INDEX_META_TOTAL_SIZE: self.df[COL_SIZE].sum() if COL_SIZE in self.df.columns else 0,
                INDEX_META_MIN_MTIME: self.df[COL_MTIME].min() if COL_MTIME in self.df.columns else None,
                INDEX_META_MAX_MTIME: self.df[COL_MTIME].max() if COL_MTIME in self.df.columns else None
            }
        except Exception as e:
            log_statement('error', f"Error getting summary metadata: {e}", __file__)
            return {
                INDEX_META_FILE_COUNT: len(self.df),
                INDEX_META_TOTAL_SIZE: 0,
                INDEX_META_MIN_MTIME: None,
                INDEX_META_MAX_MTIME: None
            }

    # Line 790 - Class Method: get_file_info
    def get_file_info(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get information for a specific file"""
        if self.df is None or self.df.empty:
            return None
        
        filepath_str = str(filepath)
        mask = self.df[COL_FILEPATH] == filepath_str
        
        if not mask.any():
            return None
        
        return self.df[mask].iloc[0].to_dict()

    # Line 805 - Class Method: get_file_hash
    def get_file_hash(self, filepath: Union[str, Path]) -> Optional[str]:
        """Get hash for a specific file"""
        file_info = self.get_file_info(filepath)
        return file_info.get(COL_HASH) if file_info else None

    # Line 815 - Class Method: get_processed_path
    def get_processed_path(self, filepath: Union[str, Path], app_state: Dict) -> Optional[Path]:
        """Get processed file path for a given source file"""
        file_info = self.get_file_info(filepath)
        if not file_info:
            return None
        
        processed_path_str = file_info.get(COL_PROCESSED_PATH)
        if not processed_path_str:
            return None
        
        # Construct full path
        processed_base = Path(app_state.get('config', {}).get('data_processing', {}).get('output_directory', PROCESSED_DATA_DIR))
        return processed_base / processed_path_str

    # Line 835 - Class Method: get_file_status
    def get_file_status(self, filepath: Union[str, Path]) -> Optional[str]:
        """Get status for a specific file"""
        file_info = self.get_file_info(filepath)
        return file_info.get(COL_STATUS) if file_info else None

    # Line 845 - Class Method: __del__
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # Clear cache
            if hasattr(self, '_cache'):
                self._cache.clear()
        except Exception:
            pass
            

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.repo is not None and hasattr(self.repo, 'git_dir') and self.repo.git_dir is not None

#     def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
#         try:
#             actual_filepaths_str = [str(Path(fp).resolve().relative_to(self.repo.working_dir)) for fp in filepaths]
#             self.repo.index.add(actual_filepaths_str)
            
#             # Check if there are actual changes to commit
#             if self.repo.is_dirty(index=True, working_tree=False, untracked_files=False) or \
#                any(fp_str in [diff.a_path for diff in self.repo.index.diff("HEAD")] for fp_str in actual_filepaths_str):
#                 self.repo.index.commit(message)
#                 log_statement('info', f"{LOG_INS}:INFO>>Committed files: {actual_filepaths_str} with message: '{message}'", Path(__file__).stem)
#                 return True
#             else:
#                 log_statement('info', f"{LOG_INS}:INFO>>No changes to commit for files: {actual_filepaths_str}", Path(__file__).stem)
#                 return True # No changes is not an error in this context
#         except GitCommandError as e:
#             if "nothing to commit" in str(e).lower():
#                 log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
#                 return True
#             log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
#             return False
#         except Exception as e:
#             log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
#             return False
    def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
        """
        Commit specific files with a message.
        
        Args:
            filepaths: List of file paths to commit
            message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{LOG_INS}:WARNING>>Cannot commit, Git repo not initialized.", Path(__file__).stem)
            return False
            
        try:
            actual_filepaths_str = [str(Path(fp).resolve().relative_to(self.repo.working_dir)) for fp in filepaths]
            self.repo.index.add(actual_filepaths_str)
            
            # Check if there are actual changes to commit
            if self.repo.is_dirty(index=True, working_tree=False, untracked_files=False) or \
               any(fp_str in [diff.a_path for diff in self.repo.index.diff("HEAD")] for fp_str in actual_filepaths_str):
                self.repo.index.commit(message)
                log_statement('info', f"{LOG_INS}:INFO>>Committed files: {actual_filepaths_str} with message: '{message}'", Path(__file__).stem)
                return True
            else:
                log_statement('info', f"{LOG_INS}:INFO>>No changes to commit for files: {actual_filepaths_str}", Path(__file__).stem)
                return True
        except GitCommandError as e:
            if "nothing to commit" in str(e).lower():
                log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
                return True
            log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
            return False
        except Exception as e:
            log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
            return False

    def add_files(self, file_paths: List[Union[str, Path]]) -> bool:
        """
        Add files to Git index.
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{LOG_INS}:WARNING>>Cannot add files, Git repo not initialized.", Path(__file__).stem)
            return False
            
        try:
            paths_str = [str(fp) for fp in file_paths]
            self.repo.index.add(paths_str)
            log_statement('info', f"{LOG_INS}:INFO>>Added files to index: {file_paths}", Path(__file__).stem)
            return True
        except GitCommandError as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to add files to Git index: {e}", Path(__file__).stem, exc_info=True)
            return False

    def commit(self, message: str) -> bool:
        """
        Create a commit with all staged changes.
        
        Args:
            message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{LOG_INS}:WARNING>>Cannot commit, Git repo not initialized.", Path(__file__).stem)
            return False
            
        try:
            self.repo.index.commit(message)
            log_statement('info', f"{LOG_INS}:INFO>>Committed with message: {message}", Path(__file__).stem)
            return True
        except GitCommandError as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to commit: {e}", Path(__file__).stem, exc_info=True)
            return False

    def get_status(self) -> Dict[str, List[str]]:
        """
        Get repository status.
        
        Returns:
            Dictionary with lists of modified, added, deleted, and untracked files
        """
        if not self.is_valid_repo():
            log_statement('warning', f"{LOG_INS}:WARNING>>Cannot get status, Git repo not initialized.", Path(__file__).stem)
            return {}
            
        status = {
            'modified': [],
            'added': [],
            'deleted': [],
            'untracked': []
        }
        
        try:
            # Get diff information
            for item in self.repo.index.diff(None):
                if item.change_type == 'M':
                    status['modified'].append(item.a_path)
                elif item.change_type == 'A':
                    status['added'].append(item.a_path)
                elif item.change_type == 'D':
                    status['deleted'].append(item.a_path)
                    
            # Get untracked files
            status['untracked'] = self.repo.untracked_files
            
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to get repository status: {e}", Path(__file__).stem, exc_info=True)
            
        return status

    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        """
        Get the Git blob hash for a file.
        
        Args:
            file_rel_path: Relative path to the file
            
        Returns:
            Blob hash string or None if failed
        """
        if not self.is_valid_repo():
            return None
            
        try:
            # For a file in the working directory (staged or unstaged, or even untracked)
            abs_path = Path(self.repo.working_dir) / Path(file_rel_path)
            if abs_path.is_file():
                with open(abs_path, 'rb') as f:
                    blob = self.repo.odb.store(git.objects.blob.Blob(self.repo.odb, f.read(), 'blob'))
                    return blob.hexsha
            log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
        return None

    def _execute_git_command(self, command: List[str], suppress_errors: bool = False, **kwargs) -> str:
        """
        Executes a Git command using the GitPython interface.
        
        Args:
            command: The Git command and its arguments (e.g., ['status', '--porcelain'])
            suppress_errors: If True, logs error and returns empty string instead of raising
            **kwargs: Additional keyword arguments to pass to the git command
            
        Returns:
            The stdout from the Git command
            
        Raises:
            GitCommandError: If the Git command fails and suppress_errors is False
        """
        if not self.is_valid_repo():
            if suppress_errors:
                return ""
            raise GitCommandError("Git repository not initialized")
            
        log_statement("debug", f"{LOG_INS}:DEBUG>>Executing Git command: git {' '.join(command)} with kwargs: {kwargs}", Path(__file__).stem, False)
        try:
            result = self.repo.git.execute(command, **kwargs)
            log_statement("debug", f"{LOG_INS}:DEBUG>>Git command 'git {' '.join(command)}' executed successfully.", Path(__file__).stem, False)
            return result
        except GitCommandError as e:
            log_msg = f"{LOG_INS}:ERROR>>Git command 'git {' '.join(command)}' failed: {e}"
            log_statement("error", log_msg, Path(__file__).stem, True)
            if suppress_errors:
                return ""
            raise GitCommandError(f"Git command 'git {' '.join(command)}' failed: {e}") from e
        except Exception as e:
            log_msg = f"{LOG_INS}:CRITICAL>>Unexpected error executing Git command 'git {' '.join(command)}': {e}"
            log_statement("critical", log_msg, Path(__file__).stem, True)
            if suppress_errors:
                return ""
            raise GitCommandError(f"Unexpected error for 'git {' '.join(command)}': {e}") from e

    def parse_log_output(self, log_output: str, num_parts: int, field_names: List[str], delimiter: str = "|") -> List[Dict[str, str]]:
        """
        Parses raw Git log output into a list of dictionaries.
        
        Args:
            log_output: The raw string output from `git log`
            num_parts: The expected number of parts when splitting a line
            field_names: The names for each part
            delimiter: The delimiter used in the log format
            
        Returns:
            A list of commit information dictionaries
        """
        log_statement("debug", f"{LOG_INS}:DEBUG>>Parsing git log output.", Path(__file__).stem, False)
        commits = []
        if not log_output:
            log_statement("info", f"{LOG_INS}:INFO>>No log output to parse.", Path(__file__).stem, False)
            return commits

        for line in log_output.strip().splitlines():
            parts = line.split(delimiter, num_parts - 1)
            if len(parts) == num_parts:
                commits.append(dict(zip(field_names, parts)))
            else:
                log_statement("warning", f"{LOG_INS}:WARNING>>Skipping malformed log line: {line}", Path(__file__).stem, False)
        log_statement("debug", f"{LOG_INS}:DEBUG>>Parsed {len(commits)} commits from log output.", Path(__file__).stem, False)
        return commits

    def get_file_last_commit_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get the last commit hash for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Commit hash string or None if not found
        """
        if not self.is_valid_repo():
            return None
            
        try:
            # Ensure file_path is relative to the repository root for the git command
            relative_file_path = Path(file_path).relative_to(self.path)
            commit = self.repo.git.log("-1", "--pretty=format:%H", "--", str(relative_file_path))
            return commit if commit else None
        except (GitCommandError, ValueError) as e:
            log_statement('debug', f"{LOG_INS}:DEBUG>>Could not get last commit hash for {file_path}: {e}", Path(__file__).stem, False)
            return None

    # Convenience property for backward compatibility
    @property
    def gifh(self):
        """Alias for gitignore_handler for backward compatibility."""
        return self.gitignore_handler

    def _init_git_operations(self):
        """Initialize Git operations for the data directory"""
        if not self.data_path or not self.data_path.exists():
            return
            
        try:
            self.git_ops = GitOpsHelper(self.data_path, create=True)
            if self.git_ops.repo:
                self.git_ops.ensure_gitignore()
                log_statement('info', f"Git operations initialized for {self.data_path}", __file__)
        except Exception as e:
            log_statement('warning', f"Could not initialize Git operations: {e}", __file__)
            self.git_ops = None

    def _init_repo(self):
        """Internal method to initialize a new Git repository."""
        try:
            self.repo = Repo.init(str(self.path))
            self.is_new_repo = True
            log_statement('info', f"{LOG_INS}:INFO>>Successfully initialized new Git repository at: {self.path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Failed to initialize new Git repository at {self.path}: {e}", Path(__file__).stem, exc_info=True)
            self.repo = None # Ensure it's None on failure
            self.is_new_repo = False

# class GitOpsHelper:
#     """Helper class for executing Git commands and parsing output."""
#     def __init__(self,
#                 repo_path: Path,
#                 create_if_missing: bool = False, 
#                 repo: Optional[Repo] = None, 
#                 root_dir: Path = ROOT_DIR):
#         self.path: Path = repo_path
#         self.git_dir: Path = self.path / ".git"
#         self.root_dir = root_dir
#         self.gitignore_handler = None  # Initialize as None, will be set later
#         self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path
#         self.git_module = git
#         self.create_if_missing = create_if_missing

#         if self.git_dir.exists() and self.git_dir.is_dir():
#             log_statement('info', f"{LOG_INS}:INFO>>Git directory exists at {self.git_dir}.", Path(__file__).stem)
#             try:
#                 self.repo = Repo(self.path)
#                 log_statement('info', f"{LOG_INS}:INFO>>Git repository loaded successfully.", Path(__file__).stem)
#             except InvalidGitRepositoryError:
#                 log_statement('error', f"{LOG_INS}:ERROR>>Invalid Git repository at {self.git_dir}.", Path(__file__).stem)
#                 if create_if_missing:
#                     self._init_repo()
#                 else:
#                     self.repo = None
#                     raise GitCommandError(f"Invalid Git repository at {self.git_dir}.")
#         elif create_if_missing:
#             log_statement('info', f"{LOG_INS}:INFO>>Git directory does not exist at {self.git_dir}. Creating new repository.", Path(__file__).stem)
#             self._init_repo()
#         else:
#             log_statement('error', f"{LOG_INS}:ERROR>>Git directory does not exist at {self.git_dir} and create_if_missing is False.", Path(__file__).stem)
#             self.repo = None
#             raise GitCommandError(f"Git directory does not exist at {self.git_dir}.")
        
#         if self.repo:
#             self.gitignore_path = self.path / GITIGNORE_FILENAME
#             self._ensure_gitignore()

#     # Line 435 - Class Method: _init_git_operations



#     def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
#         try:
#             # For a file in the working directory (staged or unstaged, or even untracked)
#             abs_path = self.repo.working_dir / Path(file_rel_path)
#             if abs_path.is_file():
#                 with open(abs_path, 'rb') as f:
#                     blob = self.repo.odb.store(Blob.input_stream(f, 'blob'))
#                     return blob.hexsha
#             log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
#         except Exception as e:
#             log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
#         return None

#     def _execute_git_command(self, command: List[str], suppress_errors: bool = False, **kwargs) -> str:
#         """
#         Executes a Git command using the GitPython interface.
#         Args:
#             command (List[str]): The Git command and its arguments (e.g., ['status', '--porcelain']).
#             suppress_errors (bool): If True, logs error and returns empty string instead of raising.
#             **kwargs: Additional keyword arguments to pass to the git command.
#         Returns:
#             str: The stdout from the Git command.
#         Raises:
#             GitCommandError: If the Git command fails and suppress_errors is False.
#         """
#         log_statement("debug", f"{LOG_INS}:DEBUG>>Executing Git command: git {' '.join(command)} with kwargs: {kwargs}", Path(__file__).stem, False)
#         try:
#             result = self.repo.git.execute(command, **kwargs)
#             log_statement("debug", f"{LOG_INS}:DEBUG>>Git command 'git {' '.join(command)}' executed successfully.", Path(__file__).stem, False)
#             return result
#         except GitCommandError as e:
#             log_msg = f"{LOG_INS}:ERROR>>Git command 'git {' '.join(command)}' failed: {e}"
#             log_statement("error", log_msg, Path(__file__).stem, True)
#             if suppress_errors:
#                 return ""
#             raise GitCommandError(f"Git command 'git {' '.join(command)}' failed: {e}") from e
#         except Exception as e:
#             log_msg = f"{LOG_INS}:CRITICAL>>Unexpected error executing Git command 'git {' '.join(command)}': {e}"
#             log_statement("critical", log_msg, Path(__file__).stem, True)
#             if suppress_errors:
#                 return ""
#             raise GitCommandError(f"Unexpected error for 'git {' '.join(command)}': {e}") from e

#     def parse_log_output(self, log_output: str, num_parts: int, field_names: List[str], delimiter: str = "|") -> List[Dict[str, str]]:
#         """
#         Parses raw Git log output into a list of dictionaries.
#         Args:
#             log_output (str): The raw string output from `git log`.
#             num_parts (int): The expected number of parts when splitting a line.
#             field_names (List[str]): The names for each part.
#             delimiter (str): The delimiter used in the log format.
#         Returns:
#             List[Dict[str, str]]: A list of commit information dictionaries.
#         """
#         log_statement("debug", f"{LOG_INS}:DEBUG>>Parsing git log output.", Path(__file__).stem, False)
#         commits = []
#         if not log_output:
#             log_statement("info", f"{LOG_INS}:INFO>>No log output to parse.", Path(__file__).stem, False)
#             return commits

#         for line in log_output.strip().splitlines():
#             parts = line.split(delimiter, num_parts - 1)
#             if len(parts) == num_parts:
#                 commits.append(dict(zip(field_names, parts)))
#             else:
#                 log_statement("warning", f"{LOG_INS}:WARNING>>Skipping malformed log line: {line}", Path(__file__).stem, False)
#         log_statement("debug", f"{LOG_INS}:DEBUG>>Parsed {len(commits)} commits from log output.", Path(__file__).stem, False)
#         return commits

#     def get_file_last_commit_hash(self, file_path: Union[str, Path]) -> Optional[str]:
#         if not self.repo:
#             return None
#         try:
#             # Ensure file_path is relative to the repository root for the git command
#             relative_file_path = Path(file_path).relative_to(self.path)
#             commit = self.repo.git.log("-1", "--pretty=format:%H", "--", str(relative_file_path))
#             return commit
#         except (GitCommandError, ValueError) as e: # ValueError if not relative
#             # Replace: logger.debug(f"GitOpsHelper: Could not get last commit hash for {file_path}: {e}")
#             log_statement('debug', f"{LOG_INS}:DEBUG>>Could not get last commit hash for {file_path}: {e}", Path(__file__).stem, False)
#             return None
        
#     # --- Potential additional Git operation methods ---
#     # def get_status(self) -> Optional[str]:
#     #     if self.is_valid_repo():
#     #         return self.repo.git.status()
#     #     log_statement('warning', "Cannot get status, Git repo not initialized.", Path(__file__).stem)
#     #     return None

#     # def add_files(self, file_paths: list[Union[str, Path]]) -> bool:
#     #     if self.is_valid_repo():
#     #         try:
#     #             self.repo.index.add([str(fp) for fp in file_paths])
#     #             log_statement('info', f"Added files to index: {file_paths}", Path(__file__).stem)
#     #             return True
#     #         except Exception as e:
#     #             log_statement('error', f"Failed to add files {file_paths}: {e}", Path(__file__).stem, exc_info=True)
#     #             return False
#     #     log_statement('warning', "Cannot add files, Git repo not initialized.", Path(__file__).stem)
#     #     return False

#     # def commit(self, message: str) -> bool:
#     #     if self.is_valid_repo():
#     #         try:
#     #             self.repo.index.commit(message)
#     #             log_statement('info', f"Committed with message: {message}", Path(__file__).stem)
#     #             return True
#     #         except Exception as e:
#     #             log_statement('error', f"Failed to commit: {e}", Path(__file__).stem, exc_info=True)
#     #             return False
#     #     log_statement('warning', "Cannot commit, Git repo not initialized.", Path(__file__).stem)
#     #     return False

