# src/utils/hashing.py - System Resource Optimized Version
import hashlib
import os
import base64
import mmap
import asyncio
import psutil
from typing import Union, Optional, Dict, Any, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from threading import Lock, RLock
import time
import gc

# Cryptography imports with fallbacks
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None
    hashes = None
    PBKDF2HMAC = None

# Pydantic imports with fallbacks
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Project imports with fallbacks
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

# System resources import
try:
    from .system_resources import get_system_resources, get_optimal_config
    SYSTEM_RESOURCES_AVAILABLE = True
except ImportError:
    SYSTEM_RESOURCES_AVAILABLE = False
    log_statement('warning', "System resource optimization not available - using fallback configuration", "hashing")

try:
    from src.data.constants import *
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False

try:
    from src.utils.config import *
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Define fallback constants for encryption
    SALT = b"tlato_default_salt_change_in_production"
    PASSWORD = b"tlato_default_password_change_in_production"

LOG_INS = f"{Path(__file__).stem}:hashing"

# Enhanced performance configuration with system resource optimization
class HashingPerformanceConfig:
    """Configuration for optimizing hashing performance with system resource awareness"""
    
    def __init__(self, force_detection: bool = False):
        # Get system resources if available
        self.system_resources = None
        if SYSTEM_RESOURCES_AVAILABLE:
            try:
                self.system_resources = get_system_resources()
                self._configure_from_system_resources()
                log_statement('info', f"{LOG_INS}:INFO>>System resource optimization enabled for hashing", 
                             Path(__file__).stem)
            except Exception as e:
                log_statement('warning', f"{LOG_INS}:WARNING>>Failed to get system resources: {e}", 
                             Path(__file__).stem)
                self._configure_fallback()
        else:
            self._configure_fallback()
        
        # Additional performance parameters
        self.large_file_threshold = 100 * 1024 * 1024  # 100MB threshold for memory mapping
        self.use_memory_mapping = True
        self.use_process_pool = True  # Use processes for CPU-intensive work
        self.prefetch_files = True
        self.memory_pressure_threshold = 85.0  # Percentage
        self.gc_frequency = 100  # Force GC every N files
        
        log_statement('info', f"{LOG_INS}:INFO>>Enhanced performance config: CPU cores={self.cpu_cores}, "
                     f"file_workers={self.max_workers_files}, algo_workers={self.max_workers_algorithms}, "
                     f"memory_limit={self.memory_limit_gb:.1f}GB", Path(__file__).stem)
    
    def _configure_from_system_resources(self):
        """Configure from detected system resources"""
        sr = self.system_resources
        
        # CPU configuration with 75% utilization
        self.cpu_cores = sr.cpu_cores_physical
        self.logical_cores = sr.cpu_cores_logical
        self.max_workers_files = sr.optimal_worker_counts['file_hashing']
        self.max_workers_algorithms = min(16, sr.recommended_cpu_cores)
        
        # Memory configuration
        self.memory_limit_gb = sr.memory_limits['batch_operation_max']
        self.available_ram_gb = sr.ram_available_gb
        
        # Dynamic chunk size based on available RAM
        base_chunk_mb = 1  # 1MB base
        memory_multiplier = min(8, max(1, self.available_ram_gb / 8))
        self.chunk_size = int(base_chunk_mb * 1024 * 1024 * memory_multiplier)
        
        # Dynamic batch size based on system resources
        base_batch = 50
        cpu_factor = min(4, max(1, self.cpu_cores / 4))
        memory_factor = min(4, max(1, self.available_ram_gb / 16))
        self.batch_size = int(base_batch * cpu_factor * memory_factor)
        
        # I/O optimization
        self.use_memory_mapping = sr.io_optimization['use_memory_mapping']
        self.buffer_size = sr.io_optimization['buffer_size_kb'] * 1024
        
        log_statement('info', f"{LOG_INS}:INFO>>System-optimized hashing config applied: "
                     f"chunk_size={self.chunk_size//1024//1024}MB, batch_size={self.batch_size}, "
                     f"buffer_size={self.buffer_size//1024}KB", Path(__file__).stem)
    
    def _configure_fallback(self):
        """Fallback configuration when system resources not available"""
        self.cpu_cores = cpu_count()
        self.logical_cores = cpu_count()
        self.max_workers_files = min(self.cpu_cores, 16)
        self.max_workers_algorithms = min(self.cpu_cores, 16)
        self.chunk_size = 1024 * 1024  # 1MB chunks
        self.batch_size = 50
        self.memory_limit_gb = 4.0  # Conservative default
        self.available_ram_gb = 8.0  # Conservative estimate
        self.buffer_size = 64 * 1024  # 64KB buffer
        
        log_statement('info', f"{LOG_INS}:INFO>>Fallback hashing configuration applied", Path(__file__).stem)
    
    def update_for_operation(self, operation_type: str, file_count: int = 0, 
                           total_size_mb: float = 0, memory_limit_gb: Optional[float] = None):
        """Update configuration for specific operation"""
        if SYSTEM_RESOURCES_AVAILABLE and self.system_resources:
            try:
                optimal_config = get_optimal_config(operation_type, file_count, total_size_mb)
                
                # Update worker counts
                self.max_workers_files = optimal_config['worker_count']
                
                # Update memory limits
                if memory_limit_gb:
                    self.memory_limit_gb = memory_limit_gb
                else:
                    self.memory_limit_gb = optimal_config['memory_limit_gb']
                
                # Adjust batch size based on workload
                if file_count > 0:
                    if file_count < 50:
                        self.batch_size = min(self.batch_size, 20)
                    elif file_count > 1000:
                        self.batch_size = min(200, self.batch_size * 2)
                
                # Memory-based chunk size adjustment
                if total_size_mb > 0:
                    avg_file_size_mb = total_size_mb / file_count if file_count > 0 else 10
                    if avg_file_size_mb > 100:  # Large files
                        self.chunk_size = min(16 * 1024 * 1024, self.chunk_size * 4)  # Up to 16MB chunks
                    elif avg_file_size_mb < 1:  # Small files
                        self.chunk_size = max(256 * 1024, self.chunk_size // 2)  # Down to 256KB chunks
                
                log_statement('debug', f"{LOG_INS}:DEBUG>>Updated config for {operation_type}: "
                             f"workers={self.max_workers_files}, memory_limit={self.memory_limit_gb:.1f}GB, "
                             f"batch_size={self.batch_size}", Path(__file__).stem)
                
            except Exception as e:
                log_statement('warning', f"{LOG_INS}:WARNING>>Failed to update configuration: {e}", 
                             Path(__file__).stem)
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent > self.memory_pressure_threshold
        except ImportError:
            return False
    
    def get_adaptive_chunk_size(self, file_size: int) -> int:
        """Get adaptive chunk size based on file size and available memory"""
        if file_size < 1024 * 1024:  # < 1MB
            return min(self.chunk_size, 256 * 1024)  # Max 256KB for small files
        elif file_size < 100 * 1024 * 1024:  # < 100MB
            return self.chunk_size
        else:  # Large files
            # Increase chunk size for large files, but respect memory limits
            max_chunk = int(self.memory_limit_gb * 1024 * 1024 * 1024 / 32)  # 1/32 of memory limit
            return min(max_chunk, self.chunk_size * 4)

# Global performance configuration
PERF_CONFIG = HashingPerformanceConfig()

# Thread-safe cache for algorithm validation
_algorithm_cache = {}
_algorithm_cache_lock = Lock()

# Define safe, validated hash algorithms - these are guaranteed to work
SAFE_HASH_ALGORITHMS = ["md5", "sha1", "sha256", "sha512"]

# Algorithms that might be available depending on system
EXTENDED_HASH_ALGORITHMS = ["blake2b", "blake2s", "sha3_256", "sha3_512"]

def get_available_algorithms() -> List[str]:
    """
    Get list of hash algorithms that are actually available on this system.
    Cached for performance.
    """
    with _algorithm_cache_lock:
        if 'available_algorithms' in _algorithm_cache:
            return _algorithm_cache['available_algorithms']
    
    available = []
    test_data = b"test"
    
    # Test safe algorithms first
    for algorithm in SAFE_HASH_ALGORITHMS:
        try:
            hasher = hashlib.new(algorithm)
            hasher.update(test_data)
            hasher.hexdigest()
            available.append(algorithm)
        except (ValueError, TypeError) as e:
            log_statement('warning', f"{LOG_INS}:WARNING>>Algorithm {algorithm} not available: {e}", 
                         Path(__file__).stem)
    
    # Test extended algorithms
    for algorithm in EXTENDED_HASH_ALGORITHMS:
        try:
            hasher = hashlib.new(algorithm)
            hasher.update(test_data)
            hasher.hexdigest()
            available.append(algorithm)
        except (ValueError, TypeError, AttributeError):
            pass
    
    if not available:
        available = ["sha256"]
        log_statement('warning', f"{LOG_INS}:WARNING>>No algorithms validated, using emergency fallback", 
                     Path(__file__).stem)
    
    with _algorithm_cache_lock:
        _algorithm_cache['available_algorithms'] = available
    
    return available

SUPPORTED_HASH_ALGORITHMS = get_available_algorithms()
DEFAULT_HASH_ALGORITHM = SUPPORTED_HASH_ALGORITHMS[0] if SUPPORTED_HASH_ALGORITHMS else "sha256"
SUPPORTED_HASH_TYPES_FOR_CUSTOM = SUPPORTED_HASH_ALGORITHMS

def validate_algorithm(algorithm: str) -> bool:
    """Thread-safe algorithm validation with caching"""
    if not algorithm or not isinstance(algorithm, str):
        return False
    
    algorithm = algorithm.lower().strip()
    
    with _algorithm_cache_lock:
        if algorithm in _algorithm_cache:
            return _algorithm_cache[algorithm]
    
    # Check against our known good list
    if algorithm not in SUPPORTED_HASH_ALGORITHMS:
        with _algorithm_cache_lock:
            _algorithm_cache[algorithm] = False
        return False
    
    # Double-check by trying to create the hasher
    try:
        hasher = hashlib.new(algorithm)
        with _algorithm_cache_lock:
            _algorithm_cache[algorithm] = True
        return True
    except (ValueError, TypeError):
        with _algorithm_cache_lock:
            _algorithm_cache[algorithm] = False
        return False

# Enhanced hashing functions with system resource optimization
def _hash_small_file(file_path: Path, algorithm: str) -> Optional[str]:
    """Hash small files using regular file I/O with memory monitoring"""
    try:
        # Check memory pressure before processing
        if PERF_CONFIG.check_memory_pressure():
            gc.collect()
        
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error hashing small file {file_path}: {e}", 
                     Path(__file__).stem)
        return None

def _hash_large_file_chunked(file_path: Path, algorithm: str, chunk_size: Optional[int] = None) -> Optional[str]:
    """Hash large files using optimized chunked reading with adaptive chunk sizes"""
    try:
        if chunk_size is None:
            file_size = file_path.stat().st_size
            chunk_size = PERF_CONFIG.get_adaptive_chunk_size(file_size)
        
        hasher = hashlib.new(algorithm)
        bytes_processed = 0
        
        with open(file_path, 'rb', buffering=PERF_CONFIG.buffer_size) as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
                bytes_processed += len(chunk)
                
                # Periodic memory check for very large files
                if bytes_processed % (chunk_size * 10) == 0 and PERF_CONFIG.check_memory_pressure():
                    gc.collect()
                    
        return hasher.hexdigest()
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error hashing large file {file_path}: {e}", 
                     Path(__file__).stem)
        return None

def _hash_large_file_mmap(file_path: Path, algorithm: str, chunk_size: Optional[int] = None) -> Optional[str]:
    """Hash very large files using memory mapping with system resource awareness"""
    try:
        if chunk_size is None:
            file_size = file_path.stat().st_size
            chunk_size = PERF_CONFIG.get_adaptive_chunk_size(file_size)
        
        hasher = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Process in chunks even with mmap for better memory usage
                bytes_processed = 0
                for i in range(0, len(mm), chunk_size):
                    chunk = mm[i:i + chunk_size]
                    hasher.update(chunk)
                    bytes_processed += len(chunk)
                    
                    # Memory pressure check
                    if bytes_processed % (chunk_size * 5) == 0 and PERF_CONFIG.check_memory_pressure():
                        gc.collect()
                        
        return hasher.hexdigest()
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error hashing file with mmap {file_path}: {e}", 
                     Path(__file__).stem)
        return None

def _hash_file_single_algorithm(file_path: Path, algorithm: str) -> Tuple[str, Optional[str]]:
    """
    Hash a single file with a single algorithm, optimized for file size and system resources.
    Returns (algorithm, hash_value)
    """
    if not validate_algorithm(algorithm):
        return (algorithm, None)
    
    if not file_path.exists() or not file_path.is_file():
        return (algorithm, None)
    
    try:
        file_size = file_path.stat().st_size
        
        # Choose hashing strategy based on file size and system resources
        if file_size == 0:
            # Empty file
            hasher = hashlib.new(algorithm)
            return (algorithm, hasher.hexdigest())
        elif file_size < 1024 * 1024:  # < 1MB
            hash_value = _hash_small_file(file_path, algorithm)
        elif file_size < PERF_CONFIG.large_file_threshold:  # < threshold (default 100MB)
            hash_value = _hash_large_file_chunked(file_path, algorithm)
        else:  # >= threshold
            if PERF_CONFIG.use_memory_mapping and not PERF_CONFIG.check_memory_pressure():
                hash_value = _hash_large_file_mmap(file_path, algorithm)
                if hash_value is None:  # Fallback if mmap fails
                    hash_value = _hash_large_file_chunked(file_path, algorithm)
            else:
                hash_value = _hash_large_file_chunked(file_path, algorithm)
        
        return (algorithm, hash_value)
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error in single algorithm hash for {file_path}: {e}", 
                     Path(__file__).stem)
        return (algorithm, None)

def generate_data_hash(file_path: Union[str, Path], algorithm: str = None) -> Optional[str]:
    """
    Optimized single-algorithm hash generation with system resource awareness.
    """
    if algorithm is None:
        algorithm = DEFAULT_HASH_ALGORITHM
    
    if not validate_algorithm(algorithm):
        log_statement('error', f"{LOG_INS}:ERROR>>Unsupported or invalid hash algorithm: '{algorithm}'. "
                     f"Supported: {SUPPORTED_HASH_ALGORITHMS}", Path(__file__).stem)
        return None
    
    file_path = Path(file_path)
    algorithm, hash_value = _hash_file_single_algorithm(file_path, algorithm.lower().strip())
    
    if hash_value:
        log_statement('debug', f"{LOG_INS}:DEBUG>>Generated {algorithm} hash for {file_path.name}: {hash_value[:16]}...", 
                     Path(__file__).stem)
    
    return hash_value

def generate_multiple_hashes_parallel(file_path: Union[str, Path], 
                                    algorithms: Optional[List[str]] = None,
                                    max_workers: Optional[int] = None) -> Dict[str, str]:
    """
    Generate multiple hashes for a single file using parallel algorithm processing with system optimization.
    """
    if algorithms is None:
        algorithms = get_safe_algorithms()
    
    # Filter to only valid algorithms
    valid_algorithms = [alg for alg in algorithms if validate_algorithm(alg)]
    
    if not valid_algorithms:
        log_statement('warning', f"{LOG_INS}:WARNING>>No valid algorithms for {file_path}", 
                     Path(__file__).stem)
        return {}
    
    file_path = Path(file_path)
    
    # For single algorithm, don't use threading overhead
    if len(valid_algorithms) == 1:
        algorithm, hash_value = _hash_file_single_algorithm(file_path, valid_algorithms[0])
        return {algorithm: hash_value} if hash_value else {}
    
    # Use system-optimized worker count
    if max_workers is None:
        max_workers = min(len(valid_algorithms), PERF_CONFIG.max_workers_algorithms)
    
    # Memory pressure check
    if PERF_CONFIG.check_memory_pressure():
        max_workers = max(1, max_workers // 2)  # Reduce workers under memory pressure
        gc.collect()
    
    # Use thread pool for multiple algorithms on the same file
    hashes = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all algorithm jobs
        future_to_algorithm = {
            executor.submit(_hash_file_single_algorithm, file_path, alg): alg 
            for alg in valid_algorithms
        }
        
        # Collect results
        for future in as_completed(future_to_algorithm):
            algorithm, hash_value = future.result()
            if hash_value:
                hashes[algorithm] = hash_value
            else:
                log_statement('warning', f"{LOG_INS}:WARNING>>Failed to generate {algorithm} hash for {file_path}", 
                             Path(__file__).stem)
    
    return hashes

def generate_multiple_hashes(file_path: Union[str, Path], algorithms: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Backwards compatible wrapper for multiple hash generation.
    """
    return generate_multiple_hashes_parallel(file_path, algorithms)

def hash_multiple_files_parallel(file_paths: List[Union[str, Path]], 
                                algorithms: Optional[List[str]] = None,
                                progress_callback: Optional[Callable[[int, int], None]] = None,
                                max_workers: Optional[int] = None,
                                memory_limit_gb: Optional[float] = None,
                                operation_type: str = "file_hashing") -> Dict[str, Dict[str, str]]:
    """
    Hash multiple files in parallel using system resource optimization for maximum efficiency.
    
    Args:
        file_paths: List of file paths to hash
        algorithms: List of algorithms to use for each file
        progress_callback: Optional callback function(completed, total)
        max_workers: Override for worker count (uses system optimization if None)
        memory_limit_gb: Memory limit for operation (uses system optimization if None)
        operation_type: Type of operation for system optimization
        
    Returns:
        Dictionary mapping file paths to their hash dictionaries
    """
    if not file_paths:
        return {}
    
    if algorithms is None:
        algorithms = get_safe_algorithms()
    
    # Filter valid algorithms
    valid_algorithms = [alg for alg in algorithms if validate_algorithm(alg)]
    if not valid_algorithms:
        log_statement('error', f"{LOG_INS}:ERROR>>No valid algorithms provided", Path(__file__).stem)
        return {}
    
    # Calculate total size for optimization
    total_size_mb = 0
    try:
        for file_path in file_paths[:min(100, len(file_paths))]:  # Sample first 100 files
            path = Path(file_path)
            if path.exists():
                total_size_mb += path.stat().st_size / (1024 * 1024)
        
        # Estimate total size
        if len(file_paths) > 100:
            avg_size = total_size_mb / min(100, len(file_paths))
            total_size_mb = avg_size * len(file_paths)
    except Exception:
        total_size_mb = len(file_paths) * 10  # 10MB estimate per file
    
    # Update performance configuration for this operation
    PERF_CONFIG.update_for_operation(operation_type, len(file_paths), total_size_mb, memory_limit_gb)
    
    # Use system-optimized parameters
    effective_workers = max_workers or PERF_CONFIG.max_workers_files
    effective_memory_limit = memory_limit_gb or PERF_CONFIG.memory_limit_gb
    batch_size = PERF_CONFIG.batch_size
    
    results = {}
    total_files = len(file_paths)
    completed = 0
    files_since_gc = 0
    
    log_statement('info', f"{LOG_INS}:INFO>>Starting system-optimized parallel hashing of {total_files} files "
                 f"with {len(valid_algorithms)} algorithms (workers={effective_workers}, "
                 f"memory_limit={effective_memory_limit:.1f}GB, batch_size={batch_size})", 
                 Path(__file__).stem)
    
    # Monitor initial memory
    try:
        initial_memory = psutil.virtual_memory()
        log_statement('info', f"{LOG_INS}:INFO>>Initial memory usage: {initial_memory.percent:.1f}%", 
                     Path(__file__).stem)
    except ImportError:
        initial_memory = None
    
    # Process files in system-optimized batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = file_paths[batch_start:batch_end]
        
        # Memory pressure check before each batch
        if PERF_CONFIG.check_memory_pressure():
            log_statement('warning', f"{LOG_INS}:WARNING>>Memory pressure detected, reducing workers and forcing GC", 
                         Path(__file__).stem)
            effective_workers = max(1, effective_workers // 2)
            gc.collect()
        
        if PERF_CONFIG.use_process_pool and len(batch_files) > 1:
            # Use process pool for CPU-intensive work with memory monitoring
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                # Submit jobs for this batch
                future_to_file = {
                    executor.submit(generate_multiple_hashes_parallel, file_path, valid_algorithms): str(file_path)
                    for file_path in batch_files
                }
                
                # Collect results for this batch with memory monitoring
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_hashes = future.result()
                        results[file_path] = file_hashes
                        
                        # Update performance monitor
                        try:
                            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                            performance_monitor.record_file(file_size, len(file_hashes))
                        except Exception:
                            performance_monitor.record_file(0, len(file_hashes))
                            
                    except Exception as e:
                        log_statement('error', f"{LOG_INS}:ERROR>>Error hashing file {file_path}: {e}", 
                                     Path(__file__).stem)
                        results[file_path] = {}
                        performance_monitor.record_file(0, 0, error=True)
                    
                    completed += 1
                    files_since_gc += 1
                    
                    if progress_callback:
                        progress_callback(completed, total_files)
                    
                    # Periodic garbage collection and memory monitoring
                    if files_since_gc >= PERF_CONFIG.gc_frequency:
                        gc.collect()
                        files_since_gc = 0
                        
                        try:
                            current_memory = psutil.virtual_memory()
                            if current_memory.percent > 90:
                                log_statement('warning', f"{LOG_INS}:WARNING>>Very high memory usage: {current_memory.percent:.1f}%", 
                                             Path(__file__).stem)
                        except ImportError:
                            pass
        else:
            # Fallback to sequential processing for small batches or when process pool disabled
            for file_path in batch_files:
                try:
                    file_hashes = generate_multiple_hashes_parallel(file_path, valid_algorithms)
                    results[str(file_path)] = file_hashes
                    
                    # Update performance monitor
                    try:
                        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                        performance_monitor.record_file(file_size, len(file_hashes))
                    except Exception:
                        performance_monitor.record_file(0, len(file_hashes))
                        
                except Exception as e:
                    log_statement('error', f"{LOG_INS}:ERROR>>Error hashing file {file_path}: {e}", 
                                 Path(__file__).stem)
                    results[str(file_path)] = {}
                    performance_monitor.record_file(0, 0, error=True)
                
                completed += 1
                files_since_gc += 1
                
                if progress_callback:
                    progress_callback(completed, total_files)
                
                # Periodic cleanup
                if files_since_gc >= PERF_CONFIG.gc_frequency:
                    gc.collect()
                    files_since_gc = 0
    
    # Final statistics
    successful = len([r for r in results.values() if r])
    try:
        final_memory = psutil.virtual_memory()
        memory_change = final_memory.percent - initial_memory.percent if initial_memory else 0
        log_statement('info', f"{LOG_INS}:INFO>>System-optimized parallel hashing completed: "
                     f"{successful}/{total_files} files successful, "
                     f"memory change: {memory_change:+.1f}%", Path(__file__).stem)
    except ImportError:
        log_statement('info', f"{LOG_INS}:INFO>>System-optimized parallel hashing completed: "
                     f"{successful}/{total_files} files successful", Path(__file__).stem)
    
    return results

async def hash_multiple_files_async(file_paths: List[Union[str, Path]], 
                                   algorithms: Optional[List[str]] = None,
                                   progress_callback: Optional[Callable[[int, int], None]] = None,
                                   max_workers: Optional[int] = None,
                                   memory_limit_gb: Optional[float] = None) -> Dict[str, Dict[str, str]]:
    """
    Async version of parallel file hashing with system resource optimization.
    """
    loop = asyncio.get_event_loop()
    
    # Run the parallel hashing in a thread pool to avoid blocking the event loop
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor,
            hash_multiple_files_parallel,
            file_paths,
            algorithms,
            progress_callback,
            max_workers,
            memory_limit_gb,
            "async_file_hashing"
        )
    
    return result

# Enhanced performance monitoring with system resource awareness
class HashingPerformanceMonitor:
    """Monitor and report hashing performance metrics with system resource tracking"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.files_processed = 0
        self.total_bytes = 0
        self.total_hashes = 0
        self.errors = 0
        self.memory_samples = []
        self.cpu_samples = []
        self.gc_calls = 0
    
    def start(self):
        self.start_time = time.time()
        try:
            initial_memory = psutil.virtual_memory()
            initial_cpu = psutil.cpu_percent()
            self.memory_samples.append((0, initial_memory.percent))
            self.cpu_samples.append((0, initial_cpu))
        except ImportError:
            pass
    
    def record_file(self, file_size: int, hash_count: int, error: bool = False):
        self.files_processed += 1
        self.total_bytes += file_size
        self.total_hashes += hash_count
        if error:
            self.errors += 1
        
        # Sample system resources periodically
        if self.files_processed % 50 == 0:
            try:
                elapsed = time.time() - self.start_time if self.start_time else 0
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                self.memory_samples.append((elapsed, memory.percent))
                self.cpu_samples.append((elapsed, cpu))
            except ImportError:
                pass
    
    def record_gc(self):
        self.gc_calls += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if self.start_time is None:
            return {"error": "Monitoring not started"}
        
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return {"error": "No time elapsed"}
        
        stats = {
            "elapsed_seconds": elapsed,
            "files_processed": self.files_processed,
            "total_bytes": self.total_bytes,
            "total_hashes": self.total_hashes,
            "errors": self.errors,
            "gc_calls": self.gc_calls,
            "files_per_second": self.files_processed / elapsed,
            "bytes_per_second": self.total_bytes / elapsed,
            "hashes_per_second": self.total_hashes / elapsed,
            "average_file_size": self.total_bytes / self.files_processed if self.files_processed > 0 else 0,
            "success_rate": (self.files_processed - self.errors) / self.files_processed if self.files_processed > 0 else 0,
            "system_optimized": SYSTEM_RESOURCES_AVAILABLE
        }
        
        # Add system resource information if available
        if self.memory_samples:
            stats["memory_usage"] = {
                "initial_percent": self.memory_samples[0][1],
                "final_percent": self.memory_samples[-1][1],
                "peak_percent": max(sample[1] for sample in self.memory_samples),
                "samples": len(self.memory_samples)
            }
        
        if self.cpu_samples:
            stats["cpu_usage"] = {
                "average_percent": sum(sample[1] for sample in self.cpu_samples) / len(self.cpu_samples),
                "peak_percent": max(sample[1] for sample in self.cpu_samples),
                "samples": len(self.cpu_samples)
            }
        
        return stats

# Global performance monitor
performance_monitor = HashingPerformanceMonitor()

# System resource optimization functions
def optimize_for_system_resources(target_memory_percent: int = 75, target_cpu_percent: int = 75):
    """Optimize hashing performance for system resources"""
    try:
        if SYSTEM_RESOURCES_AVAILABLE:
            # Force refresh of system resources
            system_resources = get_system_resources()
            
            # Update global configuration
            global PERF_CONFIG
            PERF_CONFIG = HashingPerformanceConfig(force_detection=True)
            
            # Apply target resource utilization
            if target_cpu_percent != 75:
                scale_factor = target_cpu_percent / 75
                PERF_CONFIG.max_workers_files = max(1, int(PERF_CONFIG.max_workers_files * scale_factor))
                PERF_CONFIG.max_workers_algorithms = max(1, int(PERF_CONFIG.max_workers_algorithms * scale_factor))
            
            if target_memory_percent != 75:
                scale_factor = target_memory_percent / 75
                PERF_CONFIG.memory_limit_gb *= scale_factor
                PERF_CONFIG.chunk_size = max(64*1024, int(PERF_CONFIG.chunk_size * scale_factor))
            
            log_statement('info', f"{LOG_INS}:INFO>>Optimized for system resources: "
                         f"CPU target={target_cpu_percent}%, memory target={target_memory_percent}%", 
                         Path(__file__).stem)
        else:
            log_statement('warning', f"{LOG_INS}:WARNING>>System resource optimization not available", 
                         Path(__file__).stem)
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to optimize for system resources: {e}", 
                     Path(__file__).stem)

def get_optimized_config_for_files(file_count: int, total_size_mb: float = 0) -> Dict[str, Any]:
    """Get optimized configuration for a specific file processing job"""
    try:
        if SYSTEM_RESOURCES_AVAILABLE:
            return get_optimal_config("file_hashing", file_count, total_size_mb)
        else:
            # Fallback configuration
            return {
                'worker_count': min(16, max(1, file_count // 10)),
                'batch_size': min(100, max(10, file_count // 20)),
                'memory_limit_gb': 4.0,
                'use_async_io': False
            }
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to get optimized config: {e}", 
                     Path(__file__).stem)
        return {'worker_count': 4, 'batch_size': 50, 'memory_limit_gb': 4.0}

# Backwards compatibility and existing function updates
def filter_algorithms(algorithms: List[str]) -> List[str]:
    """Filter a list of algorithms to only include valid ones."""
    return [alg.lower().strip() for alg in algorithms if validate_algorithm(alg)]

def hash_filepath(filepath: Union[str, Path]) -> str:
    """Generate SHA256 hash for filepath string."""
    filepath_str = str(filepath)
    return hashlib.sha256(filepath_str.encode('utf-8')).hexdigest()

def verify_file_hash(file_path: Union[str, Path], expected_hash: str, algorithm: str = None) -> bool:
    """Verify file hash matches expected value."""
    if algorithm is None:
        algorithm = DEFAULT_HASH_ALGORITHM
    
    current_hash = generate_data_hash(file_path, algorithm)
    return current_hash is not None and current_hash.lower() == expected_hash.lower()

# Encryption functions (keeping existing functionality)
_CIPHER_SUITE = None
ENCRYPTION_KEY = None

def _initialize_encryption():
    """Initialize encryption if cryptography is available"""
    global _CIPHER_SUITE, ENCRYPTION_KEY
    
    if not CRYPTOGRAPHY_AVAILABLE:
        return False
    
    try:
        salt = SALT if CONFIG_AVAILABLE else b"tlato_default_salt_change_in_production"
        password = PASSWORD if CONFIG_AVAILABLE else b"tlato_default_password_change_in_production"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        ENCRYPTION_KEY = base64.urlsafe_b64encode(kdf.derive(password))
        _CIPHER_SUITE = Fernet(ENCRYPTION_KEY)
        
        return True
    except Exception:
        return False

_encryption_available = _initialize_encryption()

def unhash_filepath(hashed_path: str) -> str:
    """Decrypt encrypted filepath."""
    if not _encryption_available or not _CIPHER_SUITE:
        return ""
    
    try:
        decrypted_path = _CIPHER_SUITE.decrypt(hashed_path.encode('utf-8'))
        return decrypted_path.decode('utf-8')
    except Exception:
        return ""

def hash_filepath_encrypted(filepath: Union[str, Path]) -> str:
    """Encrypt filepath for secure storage."""
    filepath_str = str(filepath)
    
    if not _encryption_available or not _CIPHER_SUITE:
        return hash_filepath(filepath_str)
    
    try:
        encrypted_path = _CIPHER_SUITE.encrypt(filepath_str.encode('utf-8'))
        return encrypted_path.decode('utf-8')
    except Exception:
        return hash_filepath(filepath_str)

# HashInfo class (keeping existing functionality)
if PYDANTIC_AVAILABLE:
    class HashInfo(BaseModel):
        hash_type: str = Field(..., description="Type of hash algorithm used")
        value: str = Field(..., description="Hash value")

        @field_validator('hash_type')
        @classmethod
        def hash_type_supported(cls, v_hash_type: str):
            if not validate_algorithm(v_hash_type):
                raise ValueError(f"Unsupported hash type: '{v_hash_type}'. Supported: {SUPPORTED_HASH_TYPES_FOR_CUSTOM}")
            return v_hash_type.lower().strip()

        def model_dump(self, **kwargs) -> Dict[str, Any]:
            return {"hash_type": self.hash_type, "value": self.value}
else:
    @dataclass
    class HashInfo:
        hash_type: str
        value: str
        
        def __post_init__(self):
            if not validate_algorithm(self.hash_type):
                raise ValueError(f"Unsupported hash type: '{self.hash_type}'. Supported: {SUPPORTED_HASH_TYPES_FOR_CUSTOM}")
            self.hash_type = self.hash_type.lower().strip()

        def model_dump(self, **kwargs) -> Dict[str, Any]:
            return {"hash_type": self.hash_type, "value": self.value}

# Enhanced utility functions
def get_supported_algorithms() -> List[str]:
    return SUPPORTED_HASH_ALGORITHMS.copy()

def is_algorithm_supported(algorithm: str) -> bool:
    return validate_algorithm(algorithm)

def get_safe_algorithms() -> List[str]:
    return ["md5", "sha1", "sha256"]

def create_hash_info(file_path: Union[str, Path], algorithm: str = None) -> Optional[HashInfo]:
    if algorithm is None:
        algorithm = DEFAULT_HASH_ALGORITHM
    
    hash_value = generate_data_hash(file_path, algorithm)
    if hash_value is None:
        return None
    
    try:
        return HashInfo(hash_type=algorithm, value=hash_value)
    except Exception:
        return None

def get_file_hash_info(file_path: Union[str, Path], algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get comprehensive hash information with enhanced performance monitoring."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "File not found", "file_path": str(file_path)}
    
    if algorithms is None:
        algorithms = get_safe_algorithms()
    
    start_time = time.time()
    file_size = file_path.stat().st_size
    
    # Get system-optimized configuration
    config = get_optimized_config_for_files(1, file_size / (1024 * 1024))
    
    hashes = generate_multiple_hashes_parallel(file_path, algorithms, 
                                             max_workers=config.get('worker_count', 4))
    
    processing_time = time.time() - start_time
    
    result = {
        "file_path": str(file_path),
        "file_size": file_size,
        "algorithms_requested": algorithms,
        "algorithms_successful": list(hashes.keys()),
        "hashes": hashes,
        "primary_hash": hashes.get(algorithms[0]) if hashes and algorithms else None,
        "hash_count": len(hashes),
        "processing_time": processing_time,
        "system_optimized": SYSTEM_RESOURCES_AVAILABLE,
        "performance": {
            "bytes_per_second": file_size / processing_time if processing_time > 0 else 0,
            "hashes_per_second": len(hashes) / processing_time if processing_time > 0 else 0,
            "config_used": config
        }
    }
    
    return result

def check_hashing_capabilities() -> Dict[str, bool]:
    """Check hashing capabilities and performance configuration with system resource info."""
    capabilities = {
        "basic_hashing": True,
        "parallel_processing": True,
        "memory_mapping": PERF_CONFIG.use_memory_mapping,
        "process_pool": PERF_CONFIG.use_process_pool,
        "async_support": True,
        "encryption": _encryption_available,
        "pydantic_models": PYDANTIC_AVAILABLE,
        "performance_monitoring": True,
        "system_resource_optimization": SYSTEM_RESOURCES_AVAILABLE,
        "memory_pressure_detection": True,
        "adaptive_chunk_sizing": True,
        "algorithm_count": len(SUPPORTED_HASH_ALGORITHMS),
        "cpu_cores": PERF_CONFIG.cpu_cores,
        "max_file_workers": PERF_CONFIG.max_workers_files,
        "max_algo_workers": PERF_CONFIG.max_workers_algorithms,
        "memory_limit_gb": PERF_CONFIG.memory_limit_gb,
        "batch_size": PERF_CONFIG.batch_size
    }
    
    return capabilities

def optimize_performance_config(target_memory_mb: int = None, target_cpu_percent: int = None):
    """Dynamically optimize performance configuration based on system resources."""
    global PERF_CONFIG
    
    if target_cpu_percent:
        # Adjust worker counts based on target CPU usage
        cpu_scale = target_cpu_percent / 75  # 75% is our default target
        PERF_CONFIG.max_workers_files = max(1, int(PERF_CONFIG.max_workers_files * cpu_scale))
        PERF_CONFIG.max_workers_algorithms = max(1, int(PERF_CONFIG.max_workers_algorithms * cpu_scale))
    
    if target_memory_mb:
        # Adjust chunk size and batch size based on target memory usage
        target_memory_gb = target_memory_mb / 1024
        PERF_CONFIG.memory_limit_gb = target_memory_gb
        
        # Adjust chunk size - use more memory for larger chunks if available
        estimated_memory_per_worker = target_memory_gb / PERF_CONFIG.max_workers_files
        PERF_CONFIG.chunk_size = min(PERF_CONFIG.chunk_size, int(estimated_memory_per_worker * 1024 * 1024 / 4))
        PERF_CONFIG.batch_size = max(10, min(200, int(target_memory_gb * 10)))
    
    log_statement('info', f"{LOG_INS}:INFO>>Performance config optimized: workers={PERF_CONFIG.max_workers_files}, "
                 f"chunk_size={PERF_CONFIG.chunk_size//1024//1024}MB, batch_size={PERF_CONFIG.batch_size}, "
                 f"memory_limit={PERF_CONFIG.memory_limit_gb:.1f}GB", Path(__file__).stem)

# Log initialization with system resource information
if LOGGER_AVAILABLE:
    capabilities = check_hashing_capabilities()
    log_statement('info', f"{LOG_INS}:INFO>>System-optimized hashing module initialized", Path(__file__).stem)
    log_statement('info', f"{LOG_INS}:INFO>>Enhanced capabilities: {capabilities}", Path(__file__).stem)
    
    if SYSTEM_RESOURCES_AVAILABLE:
        log_statement('info', f"{LOG_INS}:INFO>>System resource optimization enabled", Path(__file__).stem)
    else:
        log_statement('warning', f"{LOG_INS}:WARNING>>System resource optimization not available - using fallback", Path(__file__).stem)

# Enhanced export list
__all__ = [
    'HashInfo',
    'generate_data_hash',
    'generate_multiple_hashes',
    'generate_multiple_hashes_parallel',
    'hash_multiple_files_parallel',
    'hash_multiple_files_async',
    'hash_filepath',
    'hash_filepath_encrypted',
    'unhash_filepath',
    'verify_file_hash',
    'create_hash_info',
    'get_supported_algorithms',
    'get_safe_algorithms',
    'is_algorithm_supported',
    'validate_algorithm',
    'filter_algorithms',
    'get_file_hash_info',
    'check_hashing_capabilities',
    'optimize_performance_config',
    'optimize_for_system_resources',
    'get_optimized_config_for_files',
    'performance_monitor',
    'HashingPerformanceMonitor',
    'HashingPerformanceConfig',
    'PERF_CONFIG'
]