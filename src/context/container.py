import inspect
from src.utils.logger import log_statement
from src.core.repo_handler import get_log_prefix, RepoManager, RepoHandler, REPO_HANDLER_AVAILABLE
from src.utils.config import DataProcessingConfig
from src.utils.file_processor import FileProcessor
from src.data.processing import DataProcessor, EnhancedTokenizer
from src.analysis.labeler import SemanticLabeler, TRANSFORMERS_AVAILABLE
from transformers import AutoTokenizer, AutoModel
from threading import RLock
from pathlib import Path
from typing import Optional, Any
from src.utils.progress_tracker import ProgressManager, ProgressSnapshot
from src.utils.startup import StartupManager
from src.utils.cleanup import RepositoryCleanupManager
from src.utils.state_mgr import RepositoryState, RepositoryProcessingState, RepositoryStateManager
import time
from datetime import datetime
# Ensure the LOG_INS is set for logging context
from src.utils.logger import get_log_prefix

PROJECT_IMPORTS_AVAILABLE = True
LOG_INS = f"{Path(__file__).stem}:{inspect.currentframe().f_lineno}"

# Application state management
class DataProcessingContext:
    """Centralized state management replacing global app_state"""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
            self.config = config or self._load_default_config()
            self.container = DataProcessingContainer(self.config, self)
            self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
            
            # State tracking
            self.repo_loaded = False
            self.current_repo_id: Optional[str] = None
            self.current_source_path: Optional[Path] = None
            self.file_count = 0
            self.last_scan_time: Optional[time.time] = None
            
            # Processing state
            self.processing_status = {}
            self.model_loaded = False
            self.loaded_model = None
            self.loaded_model_path: Optional[str] = None
            self.loaded_tokenizer = None
            
            # ADDED: Progress and discovery management
            self.progress_manager = ProgressManager(self)
            self.startup_manager = StartupManager(self)
            self.current_progress: Optional[ProgressSnapshot] = None
            
            # ADDED: Repository state management
            self.repository_state_manager = RepositoryStateManager(self)
            
            # ADDED: Make container aware of this context for better integration
            self.container.context = self
            self.cleanup_manager = RepositoryCleanupManager(self)
            
            log_statement('info', f"{self.log_prefix}:INFO>>DataProcessingContext initialized with enhanced integration, repository state management, and cleanup capabilities", Path(__file__).stem)

    def set_repository(self, repo_id: str, repo: RepoHandler, source_path: Path) -> None:
        """Set current repository and initialize its state"""
        self.current_repo_id = repo_id
        self.current_source_path = source_path
        self.container.set_current_repo(repo)
        self.repo_loaded = True
        
        # Update file count if possible
        try:
            df = repo.get_dataframe()
            self.file_count = len(df) if df is not None else 0
        except Exception:
            self.file_count = 0
        
        # ADDED: Initialize repository state
        repo_state = self.repository_state_manager.get_repository_state(repo_id)
        
        # Update state to scanned if we have files
        if self.file_count > 0 and repo_state.state == RepositoryState.CREATED:
            self.repository_state_manager.update_repository_state(repo_id, RepositoryState.SCANNED)
        
        log_statement('info', f"{self.log_prefix}:INFO>>Repository set: {repo_id} (source: {source_path}, {self.file_count} files, state: {repo_state.state.value})", 
                    Path(__file__).stem)
    
    def get_repository_processing_state(self) -> Optional[RepositoryProcessingState]:
        """Get current repository processing state"""
        if self.current_repo_id:
            return self.repository_state_manager.get_repository_state(self.current_repo_id)
        return None
    
    def update_repository_state(self, new_state: RepositoryState, error_message: Optional[str] = None) -> bool:
        """Update current repository state"""
        if self.current_repo_id:
            return self.repository_state_manager.update_repository_state(
                self.current_repo_id, new_state, error_message
            )
        return False
    
    def _load_default_config(self) -> DataProcessingConfig:
        """Load default configuration"""
        try:
            if PROJECT_IMPORTS_AVAILABLE:
                # Use project's load_config if available
                config_dict = load_config()
                return DataProcessingConfig(
                    project_root=PROJECT_ROOT,
                    output_directory=Path(config_dict.get('output_directory', PROJECT_ROOT / "data")),
                    max_workers=config_dict.get('max_workers', 4),
                    use_compression=config_dict.get('use_compression', True),
                    device=config_dict.get('device', 'auto')
                )
            else:
                return DataProcessingConfig()
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to load config, using defaults: {e}", 
                         Path(__file__).stem)
            return DataProcessingConfig()

    def save_current_progress(self, process_name: str, stage: str, 
                            total_files: int, processed_files: int, failed_files: int,
                            current_file: Optional[str] = None) -> bool:
        """Save current processing progress"""
        try:
            percentage = (processed_files / total_files * 100) if total_files > 0 else 0
            
            progress = ProgressSnapshot(
                process_name=process_name,
                creation_time=datetime.now(),
                total_files=total_files,
                processed_files=processed_files,
                failed_files=failed_files,
                percentage_complete=percentage,
                current_file=current_file,
                repository_id=self.current_repo_id,
                source_path=str(self.current_source_path) if self.current_source_path else None,
                stage=stage
            )
            
            self.current_progress = progress
            return self.progress_manager.save_progress(progress)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save progress: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False
    
    def handle_startup(self) -> Dict[str, Any]:
        """Handle application startup including progress resume and discovery"""
        return self.startup_manager.handle_startup()

    def get_repo_manager(self) -> Optional[RepoManager]:
        """Get repository manager"""
        return self.container.repo_manager
    
    def get_current_repo(self) -> Optional[RepoHandler]:
        """Get current repository handler"""
        return self.container.current_repo
    
    def get_dataframe(self, refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get current repository DataFrame"""
        repo = self.get_current_repo()
        if repo and PANDAS_AVAILABLE:
            return repo.get_dataframe(refresh=refresh)
        return None
    
    def get_repository_statistics(self) -> Dict[str, Any]:
        """Get current repository statistics"""
        repo = self.get_current_repo()
        if repo:
            return repo.get_repository_statistics()
        return {}
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.container.cleanup()
            log_statement('info', f"{self.log_prefix}:INFO>>Context cleanup completed", Path(__file__).stem)
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Error during context cleanup: {e}", 
                         Path(__file__).stem, exc_info=True)

# Dependency injection container
class DataProcessingContainer:
    """Dependency injection container for data processing components"""
    def __init__(self, config: DataProcessingConfig, context: DataProcessingContext):
        self.config = config
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self._repo_manager: Optional[RepoManager] = None
        self._current_repo: Optional[RepoHandler] = None
        self._data_processor: Optional[DataProcessor] = None
        self._semantic_labeler: Optional[SemanticLabeler] = None
        self._tokenizer: Optional[Any] = None
        self._lock = RLock()
        
        log_statement('info', f"{self.log_prefix}:INFO>>DataProcessingContainer initialized", Path(__file__).stem)
    
    @property
    def repo_manager(self) -> Optional[RepoManager]:
        """Get or create repository manager"""
        with self._lock:
            if self._repo_manager is None and REPO_HANDLER_AVAILABLE:
                try:
                    self._repo_manager = RepoManager(
                        base_directory=self.config.project_root / "repositories",
                        use_compression=self.config.use_compression
                    )
                    log_statement('info', f"{self.log_prefix}:INFO>>Repository manager created", Path(__file__).stem)
                except Exception as e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>Failed to create repository manager: {e}", 
                                 Path(__file__).stem, exc_info=True)
            return self._repo_manager
    
    @property
    def current_repo(self) -> Optional[RepoHandler]:
        """Get current repository handler"""
        return self._current_repo
    
    def set_current_repo(self, repo: RepoHandler) -> None:
        """Set current repository handler"""
        with self._lock:
            self._current_repo = repo
            log_statement('info', f"{self.log_prefix}:INFO>>Current repository set", Path(__file__).stem)

    def get_data_processor(self) -> Optional[DataProcessor]:
        """Get or create data processor"""
        with self._lock:
            if self._data_processor is None and PROCESSING_IMPORTS_AVAILABLE:
                try:
                    self._data_processor = DataProcessor(
                        repo=self._current_repo,
                        repo_context = self.context,
                        output_dir=self.config.output_directory
                    )
                    log_statement('info', f"{self.log_prefix}:INFO>>Data processor created", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to create data processor: {e}", 
                                 Path(__file__).stem)
            return self._data_processor
    
    def build(self):
        # Create the semantic labeler
        self.get_semantic_labeler()

        # Create the data processor with context injection
        self.data_processor = self.get_data_processor(
            repo=self.data_processor.repo,
            context=self.context
        )

        # Create the tokenizer with context injection
        self.tokenizer = self.get_tokenizer(
            repo=self.data_processor.repo,
            context=self.context
        )

        # Set container in context for access during processing
        self.context.container = self

    def _get_labeler_device(self) -> str:
        """Determine the appropriate device for semantic labeling"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        except ImportError:
            return 'cpu'

    def get_data_processor(self) -> DataProcessor:
        return self.data_processor

    def get_tokenizer(self) -> EnhancedTokenizer:
        return self.tokenizer

    def get_semantic_labeler(self) -> SemanticLabeler:
        return self.semantic_labeler

    def get_semantic_labeler(self) -> Optional[SemanticLabeler]:
        """Get or create semantic labeler"""
        with self._lock:
            if self._semantic_labeler is None and PROCESSING_IMPORTS_AVAILABLE and self.config.enable_semantic_labeling:
                try:
                    self._semantic_labeler = SemanticLabeler(device=self._get_labeler_device(), config=self.config.__dict__)
                    log_statement('info', f"{self.log_prefix}:INFO>>Semantic labeler created", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to create semantic labeler: {e}", 
                                 Path(__file__).stem)
            return self._semantic_labeler

    def get_tokenizer(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Get or create tokenizer"""
        with self._lock:
            if self._tokenizer is None and TRANSFORMERS_AVAILABLE and self.config.enable_tokenization:
                try:
                    model = model_name or self.config.default_model_name
                    self._tokenizer = AutoTokenizer.from_pretrained(model)
                    log_statement('info', f"{self.log_prefix}:INFO>>Tokenizer created for {model}", Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to create tokenizer: {e}", 
                                 Path(__file__).stem)
            return self._tokenizer
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        with self._lock:
            if self._repo_manager:
                try:
                    # Properly cleanup repository manager
                    self._repo_manager.__exit__(None, None, None)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Error during cleanup: {e}", 
                                 Path(__file__).stem)
            
            self._repo_manager = None
            self._current_repo = None
            self._data_processor = None
            self._semantic_labeler = None
            self._tokenizer = None
