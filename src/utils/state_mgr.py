class RepositoryState(Enum):
    """Repository-level processing states"""
    UNINITIALIZED = "uninitialized"
    CREATED = "created"
    SCANNED = "scanned"
    PROCESSING = "processing"
    PROCESSED = "processed"
    TOKENIZING = "tokenizing"
    TOKENIZED = "tokenized"
    TRAINING_READY = "training_ready"
    ERROR = "error"
# Status constants aligned with repo_handler.py
class ProcessingStatus(Enum):
    """Processing status enumeration aligned with repo_handler.py"""
    DISCOVERED = "discovered"
    NEW = "new"
    PROCESSED = "processed"
    TOKENIZED = "tokenized"
    LINGUISTIC_PROCESSED = "linguistic_processed"
    LINGUISTIC_PROCESSING = "linguistic_processing"
    LINGUISTIC_FAILED = "linguistic_failed"
    ERROR = "error"
    ARCHIVED = "archived"
    # ADDED: Repository-level statuses
    REPO_READY = "repo_ready"
    REPO_PROCESSING = "repo_processing"
    REPO_PROCESSED = "repo_processed"
    REPO_TOKENIZING = "repo_tokenizing"
    REPO_TOKENIZED = "repo_tokenized"
    REPO_TRAINING_READY = "repo_training_ready"

@dataclass
class RepositoryProcessingState:
    """Repository-level processing state tracking"""
    repository_id: str
    state: RepositoryState
    total_files: int = 0
    discovered_files: int = 0
    processed_files: int = 0
    tokenized_files: int = 0
    failed_files: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now())
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    tokenizing_start_time: Optional[datetime] = None
    tokenizing_end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def processing_progress_percentage(self) -> float:
        """Calculate processing progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def tokenization_progress_percentage(self) -> float:
        """Calculate tokenization progress percentage"""
        if self.processed_files == 0:
            return 0.0
        return (self.tokenized_files / self.processed_files) * 100
    
    @property
    def overall_progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_files == 0:
            return 0.0
        # Weight processing as 60% and tokenization as 40% of overall progress
        processing_weight = 0.6
        tokenization_weight = 0.4
        return (self.processing_progress_percentage * processing_weight + 
                self.tokenization_progress_percentage * tokenization_weight)
    
    @property
    def is_ready_for_processing(self) -> bool:
        """Check if repository is ready for linguistic processing"""
        return self.state in [RepositoryState.SCANNED, RepositoryState.CREATED] and self.discovered_files > 0
    
    @property
    def is_ready_for_tokenization(self) -> bool:
        """Check if repository is ready for tokenization"""
        return self.state == RepositoryState.PROCESSED and self.processed_files > 0
    
    @property
    def is_ready_for_training(self) -> bool:
        """Check if repository is ready for model training"""
        return self.state == RepositoryState.TOKENIZED and self.tokenized_files > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'repository_id': self.repository_id,
            'state': self.state.value,
            'total_files': self.total_files,
            'discovered_files': self.discovered_files,
            'processed_files': self.processed_files,
            'tokenized_files': self.tokenized_files,
            'failed_files': self.failed_files,
            'last_updated': self.last_updated.isoformat(),
            'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
            'processing_end_time': self.processing_end_time.isoformat() if self.processing_end_time else None,
            'tokenizing_start_time': self.tokenizing_start_time.isoformat() if self.tokenizing_start_time else None,
            'tokenizing_end_time': self.tokenizing_end_time.isoformat() if self.tokenizing_end_time else None,
            'error_message': self.error_message,
            'processing_progress_percentage': self.processing_progress_percentage,
            'tokenization_progress_percentage': self.tokenization_progress_percentage,
            'overall_progress_percentage': self.overall_progress_percentage
        }

class RepositoryStateManager:
    """Manages repository-level processing state"""
    
    def __init__(self, context: 'DataProcessingContext'):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self.state_dir = self.context.config.project_root / ".tlato" / "repository_states"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._current_state: Optional[RepositoryProcessingState] = None
        self._lock = RLock()
    
    def get_repository_state(self, repo_id: str) -> RepositoryProcessingState:
        """Get or create repository processing state"""
        with self._lock:
            if self._current_state and self._current_state.repository_id == repo_id:
                return self._current_state
            
            # Try to load existing state
            state_file = self.state_dir / f"{repo_id}_state.json"
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                    
                    state = RepositoryProcessingState(
                        repository_id=data['repository_id'],
                        state=RepositoryState(data['state']),
                        total_files=data.get('total_files', 0),
                        discovered_files=data.get('discovered_files', 0),
                        processed_files=data.get('processed_files', 0),
                        tokenized_files=data.get('tokenized_files', 0),
                        failed_files=data.get('failed_files', 0),
                        last_updated=datetime.fromisoformat(data['last_updated']),
                        processing_start_time=datetime.fromisoformat(data['processing_start_time']) if data.get('processing_start_time') else None,
                        processing_end_time=datetime.fromisoformat(data['processing_end_time']) if data.get('processing_end_time') else None,
                        tokenizing_start_time=datetime.fromisoformat(data['tokenizing_start_time']) if data.get('tokenizing_start_time') else None,
                        tokenizing_end_time=datetime.fromisoformat(data['tokenizing_end_time']) if data.get('tokenizing_end_time') else None,
                        error_message=data.get('error_message')
                    )
                    
                    self._current_state = state
                    log_statement('info', f"{self.log_prefix}:INFO>>Loaded repository state for {repo_id}: {state.state.value}", 
                                Path(__file__).stem)
                    return state
                    
                except Exception as e:
                    log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load repository state: {e}", 
                                Path(__file__).stem, exc_info=True)
            
            # Create new state
            state = RepositoryProcessingState(
                repository_id=repo_id,
                state=RepositoryState.CREATED
            )
            
            # Update with current repository statistics
            self._update_file_counts(state)
            
            self._current_state = state
            self.save_repository_state(state)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Created new repository state for {repo_id}", 
                        Path(__file__).stem)
            return state
    
    def _update_file_counts(self, state: RepositoryProcessingState):
        """Update file counts from current repository"""
        try:
            repo = self.context.get_current_repo()
            if repo and hasattr(repo, 'get_repository_statistics'):
                stats = repo.get_repository_statistics()
                state.total_files = stats.get('total_tracked_files', 0)
                
                # Get status counts
                status_counts = stats.get('status_counts', {})
                state.discovered_files = status_counts.get('discovered', 0) + status_counts.get('new', 0)
                state.processed_files = status_counts.get('processed', 0) + status_counts.get('linguistic_processed', 0)
                state.tokenized_files = status_counts.get('tokenized', 0)
                state.failed_files = status_counts.get('error', 0) + status_counts.get('linguistic_failed', 0)
                
                log_statement('debug', f"{self.log_prefix}:DEBUG>>Updated file counts: total={state.total_files}, "
                            f"discovered={state.discovered_files}, processed={state.processed_files}, "
                            f"tokenized={state.tokenized_files}, failed={state.failed_files}", 
                            Path(__file__).stem)
                
        except Exception as e:
            log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to update file counts: {e}", 
                        Path(__file__).stem)
    
    def update_repository_state(self, repo_id: str, new_state: RepositoryState, 
                              error_message: Optional[str] = None) -> bool:
        """Update repository state"""
        with self._lock:
            try:
                state = self.get_repository_state(repo_id)
                old_state = state.state
                state.state = new_state
                state.last_updated = datetime.now()
                
                if error_message:
                    state.error_message = error_message
                
                # Set timestamps for state transitions
                if new_state == RepositoryState.PROCESSING and old_state != RepositoryState.PROCESSING:
                    state.processing_start_time = datetime.now()
                elif new_state == RepositoryState.PROCESSED and old_state == RepositoryState.PROCESSING:
                    state.processing_end_time = datetime.now()
                elif new_state == RepositoryState.TOKENIZING and old_state != RepositoryState.TOKENIZING:
                    state.tokenizing_start_time = datetime.now()
                elif new_state == RepositoryState.TOKENIZED and old_state == RepositoryState.TOKENIZING:
                    state.tokenizing_end_time = datetime.now()
                
                # Update file counts
                self._update_file_counts(state)
                
                # Determine final state based on file counts
                if new_state == RepositoryState.PROCESSED and state.processed_files > 0:
                    state.state = RepositoryState.PROCESSED
                elif new_state == RepositoryState.TOKENIZED and state.tokenized_files > 0:
                    state.state = RepositoryState.TRAINING_READY
                
                self.save_repository_state(state)
                
                log_statement('info', f"{self.log_prefix}:INFO>>Repository state updated: {old_state.value} -> {state.state.value}", 
                            Path(__file__).stem)
                return True
                
            except Exception as e:
                log_statement('error', f"{self.log_prefix}:ERROR>>Failed to update repository state: {e}", 
                            Path(__file__).stem, exc_info=True)
                return False
    
    def save_repository_state(self, state: RepositoryProcessingState) -> bool:
        """Save repository state to file"""
        try:
            state_file = self.state_dir / f"{state.repository_id}_state.json"
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            log_statement('debug', f"{self.log_prefix}:DEBUG>>Repository state saved to {state_file}", 
                        Path(__file__).stem)
            return True
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save repository state: {e}", 
                        Path(__file__).stem, exc_info=True)
            return False
    
    def get_current_state(self) -> Optional[RepositoryProcessingState]:
        """Get current repository processing state"""
        return self._current_state