Can you combine these two versions of the GitOpsHelper class as well as its constituent methods, variables, and other elements?  Make sure to combine similar elements (elements that serve similar purpose, have similar definitions, or similar architectures, pipelines, or utilizations), remove redundant and/or unnecessary elements, but make sure to maintain the current abilities, functionalities, features, and/or properties as well as the same name(s) for each and every code element whenever and wherever possible.  Otherwise, make any necessary alterations, updates, or improvements to the code necessary to achieve the level, breadth, and scope of functionalities that are otherwise defined in the code separately, but combined into one cohesive and all-encompassing class instead of existing as they currently do separately:

### Code Snippet #1:

class GitOpsHelper:
    """Helper class for executing Git commands and parsing output."""
    def __init__(self,
                repo_path: Path,
                create_if_not_exist: bool = False, 
                repo: Optional[Repo] = None, 
                root_dir: Path = ROOT_DIR):
        self.path: Path = repo_path
        self.git_dir: Path = self.path / ".git"
        self.root_dir = root_dir
        self.gitignore_handler = None  # Initialize as None, will be set later
        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path
        self.git_module = git
        self.create_if_not_exist = create_if_not_exist

        if self.git_dir.exists() and self.git_dir.is_dir():
            log_statement('info', f"{LOG_INS}:INFO>>Git directory exists at {self.git_dir}.", Path(__file__).stem)
            try:
                self.repo = Repo(self.path)
                log_statement('info', f"{LOG_INS}:INFO>>Git repository loaded successfully.", Path(__file__).stem)
            except InvalidGitRepositoryError:
                log_statement('error', f"{LOG_INS}:ERROR>>Invalid Git repository at {self.git_dir}.", Path(__file__).stem)
                if create_if_not_exist:
                    self._init_repo()
                else:
                    self.repo = None
                    raise GitCommandError(f"Invalid Git repository at {self.git_dir}.")
        elif create_if_not_exist:
            log_statement('info', f"{LOG_INS}:INFO>>Git directory does not exist at {self.git_dir}. Creating new repository.", Path(__file__).stem)
            self._init_repo()
        else:
            log_statement('error', f"{LOG_INS}:ERROR>>Git directory does not exist at {self.git_dir} and create_if_not_exist is False.", Path(__file__).stem)
            self.repo = None
            raise GitCommandError(f"Git directory does not exist at {self.git_dir}.")
        
        if self.repo:
            self.gitignore_path = self.path / GITIGNORE_FILENAME
            self._ensure_gitignore()

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

    def _ensure_gitignore(self):
        if not self.gitignore_path.exists():
            try:
                with open(self.gitignore_path, "w") as f:
                    f.write("# Gitignore file created by TLATO GitOpsHelper\n")
                    f.write(f"{METADATA_FILENAME}\n") # Ensure METADATA_FILE_NAME_CONST is defined
                log_statement('info', f"{LOG_INS}:INFO>>Created .gitignore file at {self.gitignore_path}", Path(__file__).stem, False)
                # Initialize gitignore handler after creating the file
                if self.gitignore_handler is None:
                    self.gitignore_handler = GitignoreHandler(self.repo, git_ops=self)
                    self.gitignore_handler.add_to_gitignore([METADATA_FILENAME]) # Add default ignore
            except IOError as e:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to create .gitignore file at {self.gitignore_path}: {e}", Path(__file__).stem, True)
        else:
            log_statement('debug', f"{LOG_INS}:DEBUG>>.gitignore file already exists at {self.gitignore_path}", Path(__file__).stem, False)
            # Initialize gitignore handler for existing file
            if self.gitignore_handler is None:
                self.gitignore_handler = GitignoreHandler(self.repo, git_ops=self)

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.repo is not None

    def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
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
                return True # No changes is not an error in this context
        except GitCommandError as e:
            if "nothing to commit" in str(e).lower():
                log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
                return True
            log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
            return False
        except Exception as e:
            log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
            return False

    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        try:
            # For a file in the working directory (staged or unstaged, or even untracked)
            abs_path = self.repo.working_dir / Path(file_rel_path)
            if abs_path.is_file():
                with open(abs_path, 'rb') as f:
                    blob = self.repo.odb.store(Blob.input_stream(f, 'blob'))
                    return blob.hexsha
            log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
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
            log_output (str): The raw string output from `git log`.
            num_parts (int): The expected number of parts when splitting a line.
            field_names (List[str]): The names for each part.
            delimiter (str): The delimiter used in the log format.
        Returns:
            List[Dict[str, str]]: A list of commit information dictionaries.
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
        if not self.repo:
            return None
        try:
            # Ensure file_path is relative to the repository root for the git command
            relative_file_path = Path(file_path).relative_to(self.path)
            commit = self.repo.git.log("-1", "--pretty=format:%H", "--", str(relative_file_path))
            return commit
        except (GitCommandError, ValueError) as e: # ValueError if not relative
            # Replace: logger.debug(f"GitOpsHelper: Could not get last commit hash for {file_path}: {e}")
            log_statement('debug', f"{LOG_INS}:DEBUG>>Could not get last commit hash for {file_path}: {e}", Path(__file__).stem, False)
            return None
        
    # --- Potential additional Git operation methods ---
    # def get_status(self) -> Optional[str]:
    #     if self.is_valid_repo():
    #         return self.repo.git.status()
    #     log_statement('warning', "Cannot get status, Git repo not initialized.", Path(__file__).stem)
    #     return None

    # def add_files(self, file_paths: list[Union[str, Path]]) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.add([str(fp) for fp in file_paths])
    #             log_statement('info', f"Added files to index: {file_paths}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to add files {file_paths}: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot add files, Git repo not initialized.", Path(__file__).stem)
    #     return False

    # def commit(self, message: str) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.commit(message)
    #             log_statement('info', f"Committed with message: {message}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to commit: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot commit, Git repo not initialized.", Path(__file__).stem)
    #     return False


### Code Snippet #2:

"""
Git Operations Module
Handles all Git-related operations separately from repository management
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import git
from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError, GitCommandError

from src.utils.logger import log_statement
from src.data.constants import *

class GitOpsHelper:
    """Helper class for executing Git commands and parsing output."""
    def __init__(self,
                repo_path: Path,
                create_if_not_exist: bool = False, 
                repo: Optional[Repo] = None, 
                root_dir: Path = ROOT_DIR):
        self.path: Path = repo_path
        self.git_dir: Path = self.path / ".git"
        self.root_dir = root_dir
        self.gitignore_handler = None  # Initialize as None, will be set later
        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path
        self.git_module = git
        self.create_if_not_exist = create_if_not_exist

        if self.git_dir.exists() and self.git_dir.is_dir():
            log_statement('info', f"{LOG_INS}:INFO>>Git directory exists at {self.git_dir}.", Path(__file__).stem)
            try:
                self.repo = Repo(self.path)
                log_statement('info', f"{LOG_INS}:INFO>>Git repository loaded successfully.", Path(__file__).stem)
            except InvalidGitRepositoryError:
                log_statement('error', f"{LOG_INS}:ERROR>>Invalid Git repository at {self.git_dir}.", Path(__file__).stem)
                if create_if_not_exist:
                    self._init_repo()
                else:
                    self.repo = None
                    raise GitCommandError(f"Invalid Git repository at {self.git_dir}.")
        elif create_if_not_exist:
            log_statement('info', f"{LOG_INS}:INFO>>Git directory does not exist at {self.git_dir}. Creating new repository.", Path(__file__).stem)
            self._init_repo()
        else:
            log_statement('error', f"{LOG_INS}:ERROR>>Git directory does not exist at {self.git_dir} and create_if_not_exist is False.", Path(__file__).stem)
            self.repo = None
            raise GitCommandError(f"Git directory does not exist at {self.git_dir}.")
        
        if self.repo:
            self.gitignore_path = self.path / GITIGNORE_FILENAME
            self._ensure_gitignore()

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

    def _ensure_gitignore(self):
        if not self.gitignore_path.exists():
            try:
                with open(self.gitignore_path, "w") as f:
                    f.write("# Gitignore file created by TLATO GitOpsHelper\n")
                    f.write(f"{METADATA_FILENAME}\n") # Ensure METADATA_FILE_NAME_CONST is defined
                log_statement('info', f"{LOG_INS}:INFO>>Created .gitignore file at {self.gitignore_path}", Path(__file__).stem, False)
                # Initialize gitignore handler after creating the file
                if self.gitignore_handler is None:
                    self.gitignore_handler = GitignoreHandler(self.repo, git_ops=self)
                    self.gitignore_handler.add_to_gitignore([METADATA_FILENAME]) # Add default ignore
            except IOError as e:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to create .gitignore file at {self.gitignore_path}: {e}", Path(__file__).stem, True)
        else:
            log_statement('debug', f"{LOG_INS}:DEBUG>>.gitignore file already exists at {self.gitignore_path}", Path(__file__).stem, False)
            # Initialize gitignore handler for existing file
            if self.gitignore_handler is None:
                self.gitignore_handler = GitignoreHandler(self.repo, git_ops=self)

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.repo is not None

    def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
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
                return True # No changes is not an error in this context
        except GitCommandError as e:
            if "nothing to commit" in str(e).lower():
                log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
                return True
            log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
            return False
        except Exception as e:
            log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
            return False

    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        try:
            # For a file in the working directory (staged or unstaged, or even untracked)
            abs_path = self.repo.working_dir / Path(file_rel_path)
            if abs_path.is_file():
                with open(abs_path, 'rb') as f:
                    blob = self.repo.odb.store(Blob.input_stream(f, 'blob'))
                    return blob.hexsha
            log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
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
            log_output (str): The raw string output from `git log`.
            num_parts (int): The expected number of parts when splitting a line.
            field_names (List[str]): The names for each part.
            delimiter (str): The delimiter used in the log format.
        Returns:
            List[Dict[str, str]]: A list of commit information dictionaries.
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
        if not self.repo:
            return None
        try:
            # Ensure file_path is relative to the repository root for the git command
            relative_file_path = Path(file_path).relative_to(self.path)
            commit = self.repo.git.log("-1", "--pretty=format:%H", "--", str(relative_file_path))
            return commit
        except (GitCommandError, ValueError) as e: # ValueError if not relative
            # Replace: logger.debug(f"GitOpsHelper: Could not get last commit hash for {file_path}: {e}")
            log_statement('debug', f"{LOG_INS}:DEBUG>>Could not get last commit hash for {file_path}: {e}", Path(__file__).stem, False)
            return None
        
    # --- Potential additional Git operation methods ---
    # def get_status(self) -> Optional[str]:
    #     if self.is_valid_repo():
    #         return self.repo.git.status()
    #     log_statement('warning', "Cannot get status, Git repo not initialized.", Path(__file__).stem)
    #     return None

    # def add_files(self, file_paths: list[Union[str, Path]]) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.add([str(fp) for fp in file_paths])
    #             log_statement('info', f"Added files to index: {file_paths}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to add files {file_paths}: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot add files, Git repo not initialized.", Path(__file__).stem)
    #     return False

    # def commit(self, message: str) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.commit(message)
    #             log_statement('info', f"Committed with message: {message}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to commit: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot commit, Git repo not initialized.", Path(__file__).stem)
    #     return False

class GitOpsHelper:
    """Helper class for executing Git commands and parsing output."""
    def __init__(self,
                 repo_path: Path,
                 create_if_not_exist: bool = False, 
                 repo: Optional[Repo] = None, 
                 root_dir: Path = ROOT_DIR):
        self.repo_path = repo_path.resolve()
        self.repo = None
        self.is_new_repo = False
        
        self._initialize_repo(create_if_not_exist)
        global LOG_INS
        self.l_ins = LOG_INS
        self.path: Path = repo_path
        self.git_dir: Path = self.path / ".git"
        self.root_dir = root_dir
        self.git_module = git
        self.create_if_not_exist = create_if_not_exist
        if self.git_dir.exists() and self.git_dir.is_dir():
            log_statement('info', f"{LOG_INS}:INFO>>Git directory exists at {self.git_dir}.", Path(__file__).stem)
            try:
                self.repo = Repo(self.path)
                log_statement('info', f"{LOG_INS}:INFO>>Git repository loaded successfully.", Path(__file__).stem)
            except InvalidGitRepositoryError:
                log_statement('error', f"{LOG_INS}:ERROR>>Invalid Git repository at {self.git_dir}.", Path(__file__).stem)
                if create_if_not_exist:
                    self._init_repo()
                else:
                    self.repo = None
                    raise GitCommandError(f"Invalid Git repository at {self.git_dir}.")
        elif create_if_not_exist:
            log_statement('info', f"{LOG_INS}:INFO>>Git directory does not exist at {self.git_dir}. Creating new repository.", Path(__file__).stem)
            self._init_repo()
        else:
            log_statement('error', f"{LOG_INS}:ERROR>>Git directory does not exist at {self.git_dir} and create_if_not_exist is False.", Path(__file__).stem)
            self.repo = None
            raise GitCommandError(f"Git directory does not exist at {self.git_dir}.")
        
        self.gifh = GitignoreHandler(self.repo, git_ops=self)
        if self.repo:
            self.gitignore_path = self.path / GITIGNORE_FILENAME
            self._ensure_gitignore()

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

    def _ensure_gitignore(self):
        if not self.gifh.gitignore_path.exists():
            try:
                with open(self.gifh.gitignore_path, "w") as f:
                    f.write("# Gitignore file created by TLATO GitOpsHelper\n")
                    f.write(f"{METADATA_FILENAME}\n") # Ensure METADATA_FILE_NAME_CONST is defined
                log_statement('info', f"{LOG_INS}:INFO>>Created .gitignore file at {self.gifh.gitignore_path}", Path(__file__).stem, False)
                self.gifh.add_to_gitignore([METADATA_FILENAME]) # Add default ignore
            except IOError as e:
                log_statement('error', f"{LOG_INS}:ERROR>>Failed to create .gitignore file at {self.gifh.gitignore_path}: {e}", Path(__file__).stem, True)
        else:
            log_statement('debug', f"{LOG_INS}:DEBUG>>.gitignore file already exists at {self.gifh.gitignore_path}", Path(__file__).stem, False)

    def is_valid_repo(self) -> bool:
        """Checks if the GitPython Repo object is initialized and valid."""
        return self.repo is not None

    def commit_changes(self, filepaths: List[Union[str, Path]], message: str) -> bool:
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
                return True # No changes is not an error in this context
        except GitCommandError as e:
            if "nothing to commit" in str(e).lower():
                log_statement('info', f"{LOG_INS}:INFO>>Nothing to commit with message: '{message}'", Path(__file__).stem)
                return True
            log_statement('error', f"{LOG_INS}:ERROR>>Git commit failed: {e}", Path(__file__).stem, exc_info=True)
            return False
        except Exception as e:
            log_statement('exception', f"{LOG_INS}:EXCEPTION>>Unexpected error during commit: {e}", Path(__file__).stem)
            return False

    def get_file_blob_hash(self, file_rel_path: str) -> Optional[str]:
        try:
            # For a file in the working directory (staged or unstaged, or even untracked)
            abs_path = self.repo.working_dir / Path(file_rel_path)
            if abs_path.is_file():
                with open(abs_path, 'rb') as f:
                    blob = self.repo.odb.store(Blob.input_stream(f, 'blob'))
                    return blob.hexsha
            log_statement('warning', f"{LOG_INS}:WARNING>>File not found for blob hash calculation (or not a file): {abs_path}", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{LOG_INS}:ERROR>>Error getting Git blob hash for {file_rel_path}: {e}", Path(__file__).stem, exc_info=True)
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
            log_output (str): The raw string output from `git log`.
            num_parts (int): The expected number of parts when splitting a line.
            field_names (List[str]): The names for each part.
            delimiter (str): The delimiter used in the log format.
        Returns:
            List[Dict[str, str]]: A list of commit information dictionaries.
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
        if not self.repo:
            return None
        try:
            # Ensure file_path is relative to the repository root for the git command
            relative_file_path = Path(file_path).relative_to(self.path)
            commit = self.repo.git.log("-1", "--pretty=format:%H", "--", str(relative_file_path))
            return commit
        except (GitCommandError, ValueError) as e: # ValueError if not relative
            # Replace: logger.debug(f"GitOpsHelper: Could not get last commit hash for {file_path}: {e}")
            log_statement('debug', f"{LOG_INS}:DEBUG>>Could not get last commit hash for {file_path}: {e}", Path(__file__).stem, False)
            return None
        
    # --- Potential additional Git operation methods ---
    # def get_status(self) -> Optional[str]:
    #     if self.is_valid_repo():
    #         return self.repo.git.status()
    #     log_statement('warning', "Cannot get status, Git repo not initialized.", Path(__file__).stem)
    #     return None

    # def add_files(self, file_paths: list[Union[str, Path]]) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.add([str(fp) for fp in file_paths])
    #             log_statement('info', f"Added files to index: {file_paths}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to add files {file_paths}: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot add files, Git repo not initialized.", Path(__file__).stem)
    #     return False

    # def commit(self, message: str) -> bool:
    #     if self.is_valid_repo():
    #         try:
    #             self.repo.index.commit(message)
    #             log_statement('info', f"Committed with message: {message}", Path(__file__).stem)
    #             return True
    #         except Exception as e:
    #             log_statement('error', f"Failed to commit: {e}", Path(__file__).stem, exc_info=True)
    #             return False
    #     log_statement('warning', "Cannot commit, Git repo not initialized.", Path(__file__).stem)
    #     return False
    
    def _initialize_repo(self, create_if_not_exist: bool):
        """Initialize or load Git repository"""
        log_ins = f"{__name__}::{self.__class__.__name__}::_initialize_repo"
        
        try:
            self.repo = Repo(self.repo_path)
            log_statement('info', f"{log_ins}::Opened existing Git repository at: {self.repo_path}", Path(__file__).stem)
        except (InvalidGitRepositoryError, NoSuchPathError):
            if create_if_not_exist:
                self.repo_path.mkdir(parents=True, exist_ok=True)
                self.repo = Repo.init(self.repo_path)
                self.is_new_repo = True
                log_statement('info', f"{log_ins}::Initialized new Git repository at: {self.repo_path}", Path(__file__).stem)
            else:
                log_statement('error', f"{log_ins}::Repository not found at {self.repo_path} and create_if_not_exist is False", Path(__file__).stem)
                raise
                
    def is_valid_repo(self) -> bool:
        """Check if repository is valid"""
        return self.repo is not None and self.repo.git_dir is not None
        
    def add_files(self, file_paths: List[Union[str, Path]]) -> bool:
        """Add files to Git index"""
        if not self.is_valid_repo():
            return False
            
        try:
            paths_str = [str(p) for p in file_paths]
            self.repo.index.add(paths_str)
            return True
        except GitCommandError as e:
            log_statement('error', f"Failed to add files to Git index: {e}", Path(__file__).stem, True)
            return False
            
    def commit(self, message: str) -> bool:
        """Create a commit"""
        if not self.is_valid_repo():
            return False
            
        try:
            self.repo.index.commit(message)
            return True
        except GitCommandError as e:
            log_statement('error', f"Failed to commit: {e}", Path(__file__).stem, True)
            return False
            
    def get_status(self) -> Dict[str, List[str]]:
        """Get repository status"""
        if not self.is_valid_repo():
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
            log_statement('error', f"Failed to get repository status: {e}", Path(__file__).stem, True)
            
        return status
