# startup_mgr.py
import inspect
from pathlib import Path
from typing import Dict, Any, List
import hashlib
from src.utils.logger import log_statement, get_log_prefix
from src.utils.progress_tracker import ProgressManager, ProgressSnapshot
from src.utils.repository_discovery import RepositoryDiscovery
from src.core.repo_handler import RepoManager, REPO_HANDLER_AVAILABLE, RepositoryDiscovery
from src.utils.repo_operations import RepositoryOperations, OperationStatus
from src.context.container import DataProcessingContext, DataProcessingContainer
from src.core.repository_state import RepositoryState, RepositoryProcessingState, RepositoryStateManager


class StartupManager:
    """Handles application startup including progress resume and repository discovery"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.progress_manager = ProgressManager(context)
        self.repository_discovery = RepositoryDiscovery(context)
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
    
    def handle_startup(self) -> Dict[str, Any]:
        """Handle application startup sequence"""
        startup_result = {
            'progress_resumed': False,
            'repositories_discovered': False,
            'actions_taken': []
        }
        
        try:
            # Check for saved progress
            progress_result = self._handle_progress_resume()
            startup_result.update(progress_result)
            
            # If no progress resumed, check for existing repositories
            if not startup_result['progress_resumed']:
                discovery_result = self._handle_repository_discovery()
                startup_result.update(discovery_result)
            
            return startup_result
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Startup handling failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            startup_result['error'] = str(e)
            return startup_result
    
    def _handle_progress_resume(self) -> Dict[str, Any]:
        """Handle progress resume if available"""
        result = {'progress_resumed': False, 'actions_taken': []}
        
        try:
            latest_progress = self.progress_manager.get_latest_progress()
            if not latest_progress:
                result['actions_taken'].append("No saved progress found")
                return result
            
            # Display progress information to user
            print("\n" + "="*60)
            print("SAVED PROGRESS FOUND")
            print("="*60)
            print(f"Process: {latest_progress.process_name}")
            print(f"Stage: {latest_progress.stage}")
            print(f"Created: {latest_progress.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Progress: {latest_progress.processed_files}/{latest_progress.total_files} files ({latest_progress.percentage_complete:.1f}%)")
            if latest_progress.failed_files > 0:
                print(f"Failed: {latest_progress.failed_files} files")
            if latest_progress.current_file:
                print(f"Last file: {Path(latest_progress.current_file).name}")
            if latest_progress.source_path:
                print(f"Source: {latest_progress.source_path}")
            print("-"*60)
            
            # Ask user if they want to resume
            resume_choice = input("Would you like to resume from this saved progress? (y/N): ").strip().lower()
            
            if resume_choice in ['y', 'yes']:
                # Attempt to resume progress
                resume_result = self._resume_progress(latest_progress)
                result.update(resume_result)
            else:
                result['actions_taken'].append("User declined to resume progress")
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Progress resume handling failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _resume_progress(self, progress: ProgressSnapshot) -> Dict[str, Any]:
        """Resume from a progress snapshot"""
        result = {'progress_resumed': False, 'actions_taken': []}
        
        try:
            # Try to load the repository
            if progress.repository_id and progress.source_path:
                repo_manager = self.context.get_repo_manager()
                if repo_manager:
                    repo = repo_manager.get_repository(progress.repository_id, auto_load=True)
                    if repo and repo.is_initialized():
                        # Set up context
                        self.context.set_repository(
                            progress.repository_id, 
                            repo, 
                            Path(progress.source_path)
                        )
                        
                        result['progress_resumed'] = True
                        result['actions_taken'].append(f"Resumed {progress.process_name} at {progress.percentage_complete:.1f}%")
                        
                        print(f"✓ Successfully resumed {progress.process_name}")
                        print(f"  Repository loaded: {progress.repository_id}")
                        print(f"  Source path: {progress.source_path}")
                        print(f"  Ready to continue from {progress.processed_files}/{progress.total_files} files")
                        
                    else:
                        result['actions_taken'].append("Failed to load repository for progress resume")
                        print("✗ Failed to load repository for saved progress")
                else:
                    result['actions_taken'].append("Repository manager not available for progress resume")
                    print("✗ Repository manager not available")
            else:
                result['actions_taken'].append("Incomplete progress information for resume")
                print("✗ Saved progress missing required information")
        
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to resume progress: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _handle_repository_discovery(self) -> Dict[str, Any]:
        """Handle automatic repository discovery"""
        result = {'repositories_discovered': False, 'actions_taken': []}
        
        try:
            # Define search paths
            search_paths = [
                self.context.config.project_root,
                Path.cwd(),
                Path.home() / "Documents",
                Path.home() / "Desktop"
            ]
            
            # Filter to existing paths
            existing_paths = [p for p in search_paths if p.exists()]
            
            if not existing_paths:
                result['actions_taken'].append("No valid search paths found")
                return result
            
            print("\n" + "="*60)
            print("SEARCHING FOR EXISTING REPOSITORIES")
            print("="*60)
            print("Searching in:")
            for path in existing_paths[:3]:  # Show first 3 paths
                print(f"  {path}")
            print("...")
            
            # Perform discovery
            discovery_result = self.repository_discovery.discover_repositories(existing_paths)
            
            if discovery_result.get('error'):
                result['actions_taken'].append(f"Discovery failed: {discovery_result['error']}")
                return result
            
            # Display results
            self._display_discovery_results(discovery_result)
            
            # Handle user selection
            selection_result = self._handle_repository_selection(discovery_result)
            result.update(selection_result)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Repository discovery failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _display_discovery_results(self, discovery_result: Dict[str, Any]) -> None:
        """Display repository discovery results"""
        print("\n--- Discovery Results ---")
        
        # TLATO repositories
        tlato_repos = discovery_result.get('tlato_repositories', [])
        if tlato_repos:
            print(f"\nFound {len(tlato_repos)} TLATO repositories:")
            for i, repo in enumerate(tlato_repos, 1):
                last_mod = repo.get('last_modified')
                last_mod_str = last_mod.strftime('%Y-%m-%d %H:%M') if last_mod else 'Unknown'
                print(f"  {i}. {repo['name']} ({repo['file_count']} files, modified: {last_mod_str})")
                print(f"     Path: {repo['path']}")
        
        # Git repositories
        git_repos = discovery_result.get('git_repositories', [])
        if git_repos:
            print(f"\nFound {len(git_repos)} Git repositories:")
            for i, repo in enumerate(git_repos, 1):
                branch_info = f"branch: {repo.branch}" if repo.branch else "no branch"
                commit_info = f"{repo.commit_count} commits" if repo.commit_count > 0 else "no commits"
                print(f"  {i}. {repo.name} ({branch_info}, {commit_info})")
                print(f"     Path: {repo.path}")
                if repo.last_commit_date:
                    print(f"     Last commit: {repo.last_commit_date.strftime('%Y-%m-%d %H:%M')}")
                if repo.has_submodules:
                    print(f"     Has {len(repo.submodule_paths)} submodule(s)")
        
        # Data directories
        data_dirs = discovery_result.get('potential_data_directories', [])
        if data_dirs:
            print(f"\nFound {len(data_dirs)} potential data directories:")
            for i, dir_info in enumerate(data_dirs[:5], 1):  # Show first 5
                size_mb = dir_info['total_size'] / (1024 * 1024)
                print(f"  {i}. {dir_info['name']} ({dir_info['file_count']} files, {size_mb:.1f} MB)")
                print(f"     Path: {dir_info['path']}")
        
        if not any([tlato_repos, git_repos, data_dirs]):
            print("No repositories or data directories found.")
    
    def _handle_repository_selection(self, discovery_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user selection of discovered repositories"""
        result = {'repositories_discovered': False, 'actions_taken': []}
        
        try:
            git_repos = discovery_result.get('git_repositories', [])
            tlato_repos = discovery_result.get('tlato_repositories', [])
            
            if not git_repos and not tlato_repos:
                result['actions_taken'].append("No repositories found to load")
                return result
            
            print("\n--- Repository Loading Options ---")
            print("1. Load TLATO repository")
            print("2. Load Git repository")
            print("3. Skip and set up new repository")
            print("0. Continue without loading")
            
            choice = input("Enter your choice (0-3): ").strip()
            
            if choice == '1' and tlato_repos:
                # Handle TLATO repository loading
                result.update(self._load_tlato_repository(tlato_repos))
            elif choice == '2' and git_repos:
                # Handle Git repository loading
                result.update(self._load_git_repository(git_repos))
            elif choice == '3':
                result['actions_taken'].append("User chose to set up new repository")
            else:
                result['actions_taken'].append("User chose to continue without loading")
        
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Repository selection failed: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _load_tlato_repository(self, tlato_repos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load a selected TLATO repository"""
        result = {'repositories_discovered': False, 'actions_taken': []}
        
        try:
            if len(tlato_repos) == 1:
                selected_repo = tlato_repos[0]
            else:
                # Let user choose
                print("\nAvailable TLATO repositories:")
                for i, repo in enumerate(tlato_repos, 1):
                    print(f"{i}. {repo['name']} - {repo['path']}")
                
                try:
                    choice = int(input(f"Select repository (1-{len(tlato_repos)}): ")) - 1
                    if 0 <= choice < len(tlato_repos):
                        selected_repo = tlato_repos[choice]
                    else:
                        result['actions_taken'].append("Invalid repository selection")
                        return result
                except ValueError:
                    result['actions_taken'].append("Invalid repository selection")
                    return result
            
            # Try to load the repository
            repo_path = Path(selected_repo['path'])
            repo_id = hashlib.sha256(str(repo_path.resolve()).encode()).hexdigest()[:16]
            
            repo_manager = self.context.get_repo_manager()
            if repo_manager:
                repo = repo_manager.get_repository(repo_id, auto_load=True)
                if repo and repo.is_initialized():
                    self.context.set_repository(repo_id, repo, repo_path)
                    result['repositories_discovered'] = True
                    result['actions_taken'].append(f"Loaded TLATO repository: {selected_repo['name']}")
                    
                    print(f"✓ Successfully loaded TLATO repository: {selected_repo['name']}")
                    print(f"  Path: {repo_path}")
                    print(f"  Files tracked: {selected_repo.get('file_count', 'Unknown')}")
                else:
                    result['actions_taken'].append("Failed to load TLATO repository")
                    print("✗ Failed to load TLATO repository")
            else:
                result['actions_taken'].append("Repository manager not available")
                print("✗ Repository manager not available")
        
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load TLATO repository: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _load_git_repository(self, git_repos: List[GitRepositoryInfo]) -> Dict[str, Any]:
        """Load a selected Git repository"""
        result = {'repositories_discovered': False, 'actions_taken': []}
        
        try:
            if len(git_repos) == 1:
                selected_repo = git_repos[0]
            else:
                # Let user choose
                print("\nAvailable Git repositories:")
                for i, repo in enumerate(git_repos, 1):
                    print(f"{i}. {repo.name} - {repo.path}")
                    if repo.branch:
                        print(f"   Branch: {repo.branch}")
                    if repo.last_commit_date:
                        print(f"   Last commit: {repo.last_commit_date.strftime('%Y-%m-%d %H:%M')}")
                
                try:
                    choice = int(input(f"Select repository (1-{len(git_repos)}): ")) - 1
                    if 0 <= choice < len(git_repos):
                        selected_repo = git_repos[choice]
                    else:
                        result['actions_taken'].append("Invalid repository selection")
                        return result
                except ValueError:
                    result['actions_taken'].append("Invalid repository selection")
                    return result
            
            # Display Git repository details and confirm
            print(f"\n--- Git Repository Details ---")
            print(f"Name: {selected_repo.name}")
            print(f"Path: {selected_repo.path}")
            print(f"Branch: {selected_repo.branch or 'Unknown'}")
            print(f"Commits: {selected_repo.commit_count}")
            print(f"Files: {selected_repo.file_count}")
            print(f"Size: {selected_repo.total_size / (1024*1024):.1f} MB")
            if selected_repo.has_submodules:
                print(f"Submodules: {len(selected_repo.submodule_paths)}")
                for submodule_path in selected_repo.submodule_paths[:3]:  # Show first 3
                    print(f"  - {submodule_path}")
            
            confirm = input("\nLoad this Git repository for processing? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                # Set up repository for Git repo
                repo_ops = RepositoryOperations(self.context)
                setup_result = repo_ops.setup_repository(selected_repo.path)
                
                if setup_result['status'] == OperationStatus.SUCCESS.value:
                    result['repositories_discovered'] = True
                    result['actions_taken'].append(f"Loaded Git repository: {selected_repo.name}")
                    
                    print(f"✓ Successfully set up repository for Git repo: {selected_repo.name}")
                    
                    # Handle submodules if present
                    if selected_repo.has_submodules:
                        self._handle_submodules(selected_repo)
                else:
                    result['actions_taken'].append("Failed to set up repository for Git repo")
                    print("✗ Failed to set up repository for Git repository")
            else:
                result['actions_taken'].append("User declined to load Git repository")
        
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load Git repository: {e}", 
                         Path(__file__).stem, exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _handle_submodules(self, git_repo: GitRepositoryInfo) -> None:
        """Handle Git submodules"""
        try:
            if not git_repo.submodule_paths:
                return
            
            print(f"\nThis repository has {len(git_repo.submodule_paths)} submodule(s):")
            for submodule_path in git_repo.submodule_paths:
                print(f"  - {submodule_path}")
            
            include_submodules = input("Include submodules in processing? (y/N): ").strip().lower()
            
            if include_submodules in ['y', 'yes']:
                print("Submodules will be included in file processing.")
                # The existing file processing will automatically handle subdirectories
            else:
                print("Submodules will be skipped.")
                # Could implement submodule exclusion logic here if needed
        
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error handling submodules: {e}", 
                         Path(__file__).stem, exc_info=True)