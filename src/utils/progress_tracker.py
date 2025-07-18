# src/utils/progress_tracker.py
"""Enhanced progress tracking with system resource monitoring - FIXED UPDATE FREQUENCY"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import inspect

try:
    from src.utils.logger import log_statement
except ImportError:
    def log_statement(level, message, module=None, exc_info=False):
        print(f"[{level.upper()}] {message}")

from src.context.container import *

class SystemResourceMonitor:
    """Monitors system resources during operations"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.resources = {
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'memory_percent': 0.0,
            'disk_read_mb': 0.0,
            'disk_write_mb': 0.0,
            'threads': 0
        }
        self.start_time = time.time()
        self.start_disk_io = None
        self.lock = threading.Lock()  # ADDED: Thread safety for resource updates
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        try:
            io_counters = self.process.io_counters()
            self.start_disk_io = {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes
            }
        except:
            self.start_disk_io = {'read_bytes': 0, 'write_bytes': 0}
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        log_statement('debug', f"Resource monitoring started for enhanced progress tracking", Path(__file__).stem)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            log_statement('debug', f"Resource monitoring stopped", Path(__file__).stem)
    
    def _monitor_loop(self):
        """Background monitoring loop - FIXED: More frequent updates"""
        while self.monitoring:
            try:
                with self.lock:  # ADDED: Thread safety
                    # CPU usage
                    self.resources['cpu_percent'] = self.process.cpu_percent(interval=0.1)
                    
                    # Memory usage
                    mem_info = self.process.memory_info()
                    self.resources['memory_mb'] = mem_info.rss / (1024 * 1024)
                    self.resources['memory_percent'] = self.process.memory_percent()
                    
                    # Disk I/O
                    try:
                        io_counters = self.process.io_counters()
                        if self.start_disk_io:
                            read_bytes = io_counters.read_bytes - self.start_disk_io['read_bytes']
                            write_bytes = io_counters.write_bytes - self.start_disk_io['write_bytes']
                            self.resources['disk_read_mb'] = read_bytes / (1024 * 1024)
                            self.resources['disk_write_mb'] = write_bytes / (1024 * 1024)
                    except:
                        pass
                    
                    # Thread count
                    self.resources['threads'] = self.process.num_threads()
                
            except Exception as e:
                log_statement('debug', f"Resource monitoring error: {e}", Path(__file__).stem)
            
            time.sleep(0.2)  # FIXED: Update every 200ms instead of 500ms for more responsive display
    
    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage - ADDED: Thread safety"""
        with self.lock:
            return self.resources.copy()
    
    def get_resource_string(self) -> str:
        """Get formatted resource string for display"""
        with self.lock:  # ADDED: Thread safety
            return (f"CPU: {self.resources['cpu_percent']:.1f}% | "
                    f"RAM: {self.resources['memory_mb']:.1f}MB ({self.resources['memory_percent']:.1f}%) | "
                    f"Disk R/W: {self.resources['disk_read_mb']:.1f}/{self.resources['disk_write_mb']:.1f}MB | "
                    f"Threads: {self.resources['threads']}")

# Progress Management System
@dataclass
class ProgressSnapshot:
    """Represents a saved progress point"""
    process_name: str
    creation_time: datetime
    total_files: int
    processed_files: int
    failed_files: int
    percentage_complete: float
    current_file: Optional[str]
    repository_id: Optional[str]
    source_path: Optional[str]
    stage: str  # e.g., 'file_processing', 'linguistic_processing', 'tokenization'
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'process_name': self.process_name,
            'creation_time': self.creation_time.isoformat(),
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'percentage_complete': self.percentage_complete,
            'current_file': self.current_file,
            'repository_id': self.repository_id,
            'source_path': self.source_path,
            'stage': self.stage,
            'additional_metadata': self.additional_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressSnapshot':
        """Create from dictionary"""
        return cls(
            process_name=data['process_name'],
            creation_time=datetime.fromisoformat(data['creation_time']),
            total_files=data['total_files'],
            processed_files=data['processed_files'],
            failed_files=data['failed_files'],
            percentage_complete=data['percentage_complete'],
            current_file=data.get('current_file'),
            repository_id=data.get('repository_id'),
            source_path=data.get('source_path'),
            stage=data['stage'],
            additional_metadata=data.get('additional_metadata', {})
        )

class ProgressManager:
    """Manages saving and loading of progress snapshots"""
    
    def __init__(self, context: DataProcessingContext):
        self.context = context
        self.log_prefix = get_log_prefix(inspect.currentframe()) if REPO_HANDLER_AVAILABLE else LOG_INS
        self.progress_dir = self.context.config.project_root / ".tlato" / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
    
    def save_progress(self, progress: ProgressSnapshot) -> bool:
        """Save a progress snapshot"""
        try:
            timestamp = progress.creation_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{progress.process_name}_{timestamp}.json"
            filepath = self.progress_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(progress.to_dict(), f, indent=2)
            
            log_statement('info', f"{self.log_prefix}:INFO>>Progress saved: {filename}", 
                         Path(__file__).stem)
            return True
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to save progress: {e}", 
                         Path(__file__).stem, exc_info=True)
            return False

    def get_latest_progress(self) -> Optional[ProgressSnapshot]:
        """Get the most recent progress snapshot"""
        try:
            progress_files = list(self.progress_dir.glob("*.json"))
            if not progress_files:
                return None
            
            # Sort by modification time, get most recent
            latest_file = max(progress_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            return ProgressSnapshot.from_dict(data)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to load latest progress: {e}", 
                         Path(__file__).stem, exc_info=True)
            return None
    
    def get_all_progress_snapshots(self) -> List[ProgressSnapshot]:
        """Get all available progress snapshots, sorted by creation time"""
        snapshots = []
        try:
            progress_files = list(self.progress_dir.glob("*.json"))
            
            for file_path in progress_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    snapshots.append(ProgressSnapshot.from_dict(data))
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to load progress file {file_path}: {e}", 
                                 Path(__file__).stem)
            
            # Sort by creation time, newest first
            snapshots.sort(key=lambda s: s.creation_time, reverse=True)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to get progress snapshots: {e}", 
                         Path(__file__).stem, exc_info=True)
        
        return snapshots
    
    def clear_old_progress(self, keep_count: int = 10) -> None:
        """Clear old progress files, keeping only the most recent ones"""
        try:
            snapshots = self.get_all_progress_snapshots()
            if len(snapshots) <= keep_count:
                return
            
            # Get files to delete
            progress_files = list(self.progress_dir.glob("*.json"))
            progress_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            files_to_delete = progress_files[keep_count:]
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    log_statement('debug', f"{self.log_prefix}:DEBUG>>Deleted old progress file: {file_path.name}", 
                                 Path(__file__).stem)
                except Exception as e:
                    log_statement('warning', f"{self.log_prefix}:WARNING>>Failed to delete progress file {file_path}: {e}", 
                                 Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"{self.log_prefix}:ERROR>>Failed to clear old progress files: {e}", 
                         Path(__file__).stem, exc_info=True)

class EnhancedProgressTracker:
    """Enhanced progress tracker with detailed metrics and resource monitoring - FIXED UPDATE FREQUENCY"""
    
    def __init__(self, 
                 total_items: int,
                 description: str = "Processing",
                 unit: str = "items",
                 show_resources: bool = True,
                 update_interval: float = 0.5):  # FIXED: Changed to 0.5 seconds for more responsive updates
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.errors = []
        self.description = description
        self.unit = unit
        self.show_resources = show_resources
        self.update_interval = update_interval
        
        self.start_time = time.time()
        self.last_update = 0
        self.last_resource_update = 0  # ADDED: Separate tracking for resource updates
        
        # System resource monitor
        self.resource_monitor = SystemResourceMonitor() if show_resources else None
        
        # FIXED: Create progress bar with better format and forced refresh
        bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        if show_resources:
            bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        
        self.pbar = tqdm(
            total=total_items,
            desc=description,
            unit=unit,
            bar_format=bar_format,
            leave=True,
            dynamic_ncols=True,
            miniters=1,  # FIXED: Update after every item
            mininterval=0.1,  # FIXED: Minimum time interval of 0.1 seconds between updates
            maxinterval=1.0   # FIXED: Maximum time interval of 1.0 seconds between updates
        )
        
        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        # FIXED: Force initial display update
        self._update_display(force=True)
        
        log_statement('debug', f"Enhanced progress tracker initialized: {total_items} {unit}, update_interval={update_interval}s", Path(__file__).stem)
    
    def update(self, success: bool = True, error_msg: Optional[str] = None):
        """Update progress with success/failure tracking - FIXED: More frequent display updates"""
        self.processed_items += 1
        
        if not success:
            self.failed_items += 1
            if error_msg:
                self.errors.append({
                    'item': self.processed_items,
                    'error': error_msg,
                    'timestamp': time.time()
                })
                log_statement('debug', f"Progress tracker recorded error for item {self.processed_items}: {error_msg[:100]}", Path(__file__).stem)
        
        # Update progress bar
        self.pbar.update(1)
        
        # FIXED: Update display more frequently and force resource updates every second
        current_time = time.time()
        force_update = False
        
        # Force resource update every 1 second regardless of other updates
        if current_time - self.last_resource_update >= 1.0:
            force_update = True
            self.last_resource_update = current_time
        
        # Update display if enough time has passed OR if forced
        if current_time - self.last_update >= self.update_interval or force_update:
            self._update_display(force=force_update)
            self.last_update = current_time
    
    def _update_display(self, force: bool = False):
        """Update progress display with detailed metrics - FIXED: Force updates when needed"""
        try:
            # Calculate rates
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                process_rate = self.processed_items / elapsed
                success_rate = (self.processed_items - self.failed_items) / self.processed_items * 100 if self.processed_items > 0 else 0
            else:
                process_rate = 0
                success_rate = 0
            
            # Build postfix with metrics
            postfix_parts = [
                f"Success: {self.processed_items - self.failed_items}",
                f"Failed: {self.failed_items}",
                f"Rate: {process_rate:.1f}/s",
                f"Success%: {success_rate:.1f}%"
            ]
            
            # Add resource info if available - FIXED: Always get fresh resource data
            if self.resource_monitor:
                try:
                    resources = self.resource_monitor.get_resource_string()
                    postfix_parts.append(resources)
                except Exception as e:
                    log_statement('debug', f"Error getting resource string: {e}", Path(__file__).stem)
                    postfix_parts.append("Resources: Error")
            
            # FIXED: Update progress bar postfix and force refresh
            postfix_string = " | ".join(postfix_parts)
            self.pbar.set_postfix_str(postfix_string)
            
            # FIXED: Force refresh if requested
            if force:
                self.pbar.refresh()
                
        except Exception as e:
            log_statement('error', f"Error updating progress display: {e}", Path(__file__).stem, exc_info=True)
    
    def finish(self):
        """Finish progress tracking and show summary"""
        try:
            # Final update
            self._update_display(force=True)
            
            # Stop resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
            
            # Close progress bar
            self.pbar.close()
            
            # Show summary
            elapsed = time.time() - self.start_time
            success_count = self.processed_items - self.failed_items
            
            print(f"\n{self.description} Summary:")
            print(f"  Total {self.unit}: {self.total_items}")
            print(f"  Processed: {self.processed_items}")
            print(f"  Successful: {success_count}")
            print(f"  Failed: {self.failed_items}")
            print(f"  Success Rate: {(success_count/self.processed_items*100) if self.processed_items > 0 else 0:.1f}%")
            print(f"  Time Elapsed: {elapsed:.1f}s")
            print(f"  Average Rate: {self.processed_items/elapsed if elapsed > 0 else 0:.1f} {self.unit}/s")
            
            if self.errors:
                print(f"\nFirst 5 errors:")
                for error in self.errors[:5]:
                    print(f"  Item {error['item']}: {error['error']}")
            
            log_statement('info', f"Progress tracking completed: {success_count}/{self.processed_items} successful in {elapsed:.1f}s", Path(__file__).stem)
            
        except Exception as e:
            log_statement('error', f"Error finishing progress tracking: {e}", Path(__file__).stem, exc_info=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        elapsed = time.time() - self.start_time
        return {
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'success_rate': (self.processed_items - self.failed_items) / self.processed_items if self.processed_items > 0 else 0,
            'elapsed_time': elapsed,
            'process_rate': self.processed_items / elapsed if elapsed > 0 else 0,
            'errors': self.errors.copy(),
            'resources': self.resource_monitor.get_resources() if self.resource_monitor else {}
        }

def create_progress_tracker(total_items: int, 
                          description: str = "Processing",
                          unit: str = "items",
                          show_resources: bool = True,
                          update_interval: float = 0.5) -> EnhancedProgressTracker:  # FIXED: Added update_interval parameter
    """Factory function to create progress tracker with configurable update frequency"""
    log_statement('debug', f"Creating progress tracker: {total_items} {unit}, update_interval={update_interval}s", Path(__file__).stem)
    return EnhancedProgressTracker(
        total_items=total_items,
        description=description,
        unit=unit,
        show_resources=show_resources,
        update_interval=update_interval
    )