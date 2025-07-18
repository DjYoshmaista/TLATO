# system_resources.py
# System Resource Detection and Optimization Utility
# Detects available system resources and calculates optimal usage parameters
# for high-performance file processing operations.

import psutil
import platform
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import multiprocessing
import threading

@dataclass
class SystemResources:
    """Container for system resource information and recommendations"""
    # Raw system capabilities
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_frequency_mhz: float
    l1_cache_kb: int
    l2_cache_kb: int
    l3_cache_mb: int
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_available_gb: float
    vram_total_gb: float
    vram_available_gb: float
    disk_space_gb: Dict[str, Dict[str, float]]  # {mount_point: {total, free, used}}
    
    # Recommended usage (75% of available resources)
    recommended_cpu_cores: int
    recommended_cpu_cores_io: int
    recommended_ram_gb: float
    recommended_swap_gb: float
    recommended_vram_gb: float
    recommended_disk_cache_gb: float
    recommended_batch_size: int
    recommended_parallel_threshold: int
    recommended_memory_chunk_mb: int
    
    # Performance optimization parameters
    optimal_worker_counts: Dict[str, int]
    memory_limits: Dict[str, float]
    io_optimization: Dict[str, Any]

class SystemResourceManager:
    """Manages system resource detection and optimization"""
    
    def __init__(self):
        self.resources: Optional[SystemResources] = None
        self._cache_valid = False
        self._last_check = 0
    
    def get_system_resources(self, force_refresh: bool = False) -> SystemResources:
        """Get comprehensive system resource information"""
        import time
        
        current_time = time.time()
        if not force_refresh and self._cache_valid and (current_time - self._last_check) < 60:
            return self.resources
        
        print("Detecting system resources...")
        
        # CPU Information
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        cpu_freq = psutil.cpu_freq()
        cpu_frequency_mhz = cpu_freq.current if cpu_freq else 2000.0
        
        # Cache Information (best effort - platform dependent)
        l1_cache_kb, l2_cache_kb, l3_cache_mb = self._get_cpu_cache_info()
        
        # Memory Information
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        
        swap = psutil.swap_memory()
        swap_total_gb = swap.total / (1024**3)
        swap_available_gb = (swap.total - swap.used) / (1024**3)
        
        # VRAM Information
        vram_total_gb, vram_available_gb = self._get_vram_info()
        
        # Disk Space Information
        disk_space_gb = self._get_disk_space_info()
        
        # Calculate optimal usage parameters (75% utilization)
        recommended_cpu_cores = max(1, int(cpu_cores_physical * 0.75))
        recommended_cpu_cores_io = max(2, int(cpu_cores_logical * 0.75))
        recommended_ram_gb = ram_available_gb * 0.75
        recommended_swap_gb = swap_available_gb * 0.75
        recommended_vram_gb = vram_available_gb * 0.75
        
        # Calculate optimal disk cache (up to 10GB or 10% of available space)
        max_disk_free = max((info['free'] for info in disk_space_gb.values()), default=0)
        recommended_disk_cache_gb = min(10.0, max_disk_free * 0.1)
        
        # Calculate dynamic batch sizes based on available memory
        base_batch_size = 50
        memory_factor = min(8, max(1, recommended_ram_gb / 8))  # Scale with RAM
        cpu_factor = min(4, max(1, recommended_cpu_cores / 4))   # Scale with CPU
        recommended_batch_size = int(base_batch_size * memory_factor * cpu_factor)
        
        # Parallel processing thresholds
        recommended_parallel_threshold = max(10, recommended_batch_size // 5)
        
        # Memory chunk size for large file operations (MB)
        recommended_memory_chunk_mb = min(256, max(16, int(recommended_ram_gb * 1024 / 32)))
        
        # Optimal worker counts for different task types
        optimal_worker_counts = {
            'cpu_intensive': recommended_cpu_cores,
            'io_intensive': recommended_cpu_cores_io,
            'mixed_workload': max(2, int((recommended_cpu_cores + recommended_cpu_cores_io) / 2)),
            'file_hashing': min(32, recommended_cpu_cores_io),
            'batch_processing': min(16, recommended_cpu_cores * 2),
            'concurrent_repos': min(8, max(2, recommended_cpu_cores // 2))
        }
        
        # Memory limits for different operations (GB)
        memory_limits = {
            'single_file_max': min(1.0, recommended_ram_gb * 0.1),
            'batch_operation_max': min(8.0, recommended_ram_gb * 0.5),
            'cache_max': min(4.0, recommended_ram_gb * 0.25),
            'buffer_size_mb': min(128, max(16, int(recommended_ram_gb * 32)))
        }
        
        # I/O optimization parameters
        io_optimization = {
            'use_memory_mapping': ram_total_gb > 8,
            'enable_disk_cache': recommended_disk_cache_gb > 1.0,
            'async_io_threshold': 100,  # files
            'parallel_io_threshold': recommended_parallel_threshold,
            'compression_level': 3 if cpu_cores_physical >= 4 else 1,
            'buffer_size_kb': min(1024, max(64, int(recommended_ram_gb * 16)))
        }
        
        self.resources = SystemResources(
            cpu_cores_physical=cpu_cores_physical,
            cpu_cores_logical=cpu_cores_logical,
            cpu_frequency_mhz=cpu_frequency_mhz,
            l1_cache_kb=l1_cache_kb,
            l2_cache_kb=l2_cache_kb,
            l3_cache_mb=l3_cache_mb,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            swap_total_gb=swap_total_gb,
            swap_available_gb=swap_available_gb,
            vram_total_gb=vram_total_gb,
            vram_available_gb=vram_available_gb,
            disk_space_gb=disk_space_gb,
            recommended_cpu_cores=recommended_cpu_cores,
            recommended_cpu_cores_io=recommended_cpu_cores_io,
            recommended_ram_gb=recommended_ram_gb,
            recommended_swap_gb=recommended_swap_gb,
            recommended_vram_gb=recommended_vram_gb,
            recommended_disk_cache_gb=recommended_disk_cache_gb,
            recommended_batch_size=recommended_batch_size,
            recommended_parallel_threshold=recommended_parallel_threshold,
            recommended_memory_chunk_mb=recommended_memory_chunk_mb,
            optimal_worker_counts=optimal_worker_counts,
            memory_limits=memory_limits,
            io_optimization=io_optimization
        )
        
        self._cache_valid = True
        self._last_check = current_time
        
        print(f"System Resources Detected:")
        print(f"  CPU: {cpu_cores_physical} physical cores, {cpu_cores_logical} logical cores")
        print(f"  RAM: {ram_total_gb:.1f}GB total, {ram_available_gb:.1f}GB available")
        print(f"  VRAM: {vram_total_gb:.1f}GB total, {vram_available_gb:.1f}GB available")
        print(f"  Recommended CPU cores: {recommended_cpu_cores}")
        print(f"  Recommended batch size: {recommended_batch_size}")
        print(f"  Recommended RAM usage: {recommended_ram_gb:.1f}GB")
        
        return self.resources
    
    def _get_cpu_cache_info(self) -> Tuple[int, int, int]:
        """Get CPU cache information (best effort, platform dependent)"""
        l1_cache_kb = 32  # Default fallback
        l2_cache_kb = 256  # Default fallback
        l3_cache_mb = 8    # Default fallback
        
        try:
            if platform.system() == "Linux":
                # Try to read from /proc/cpuinfo or lscpu
                try:
                    result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'L1d cache' in line:
                                cache_str = line.split(':')[1].strip()
                                if 'K' in cache_str:
                                    l1_cache_kb = int(cache_str.replace('K', '').strip())
                            elif 'L2 cache' in line:
                                cache_str = line.split(':')[1].strip()
                                if 'K' in cache_str:
                                    l2_cache_kb = int(cache_str.replace('K', '').strip())
                                elif 'M' in cache_str:
                                    l2_cache_kb = int(float(cache_str.replace('M', '').strip()) * 1024)
                            elif 'L3 cache' in line:
                                cache_str = line.split(':')[1].strip()
                                if 'M' in cache_str:
                                    l3_cache_mb = int(cache_str.replace('M', '').strip())
                except:
                    pass
            
            elif platform.system() == "Windows":
                # Try WMI or system info
                try:
                    import wmi
                    c = wmi.WMI()
                    for processor in c.Win32_Processor():
                        if hasattr(processor, 'L2CacheSize') and processor.L2CacheSize:
                            l2_cache_kb = processor.L2CacheSize
                        if hasattr(processor, 'L3CacheSize') and processor.L3CacheSize:
                            l3_cache_mb = processor.L3CacheSize // 1024
                except ImportError:
                    pass
            
            elif platform.system() == "Darwin":  # macOS
                try:
                    result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'hw.l1dcachesize' in line:
                                l1_cache_kb = int(line.split(':')[1].strip()) // 1024
                            elif 'hw.l2cachesize' in line:
                                l2_cache_kb = int(line.split(':')[1].strip()) // 1024
                            elif 'hw.l3cachesize' in line:
                                l3_cache_mb = int(line.split(':')[1].strip()) // (1024 * 1024)
                except:
                    pass
        
        except Exception:
            pass  # Use defaults
        
        return l1_cache_kb, l2_cache_kb, l3_cache_mb
    
    def _get_vram_info(self) -> Tuple[float, float]:
        """Get VRAM information from GPU(s)"""
        vram_total_gb = 0.0
        vram_available_gb = 0.0
        
        try:
            # Try NVIDIA GPU first
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            total_mb, free_mb = map(int, line.split(','))
                            vram_total_gb += total_mb / 1024
                            vram_available_gb += free_mb / 1024
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            # Try AMD GPU
            if vram_total_gb == 0:
                try:
                    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        # Parse AMD GPU memory info
                        # This is GPU-specific and may need adjustment
                        pass
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
            
            # Fallback: assume integrated graphics with shared system RAM
            if vram_total_gb == 0:
                memory = psutil.virtual_memory()
                # Estimate 1/8 of system RAM as shared graphics memory
                vram_total_gb = (memory.total / (1024**3)) / 8
                vram_available_gb = vram_total_gb * 0.8  # Assume 80% available
        
        except Exception:
            # Final fallback
            vram_total_gb = 2.0
            vram_available_gb = 1.5
        
        return vram_total_gb, vram_available_gb
    
    def _get_disk_space_info(self) -> Dict[str, Dict[str, float]]:
        """Get disk space information for all mounted volumes"""
        disk_info = {}
        
        try:
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    # Skip special file systems
                    if any(fs in partition.fstype.lower() for fs in ['proc', 'sys', 'dev', 'run', 'tmp']):
                        continue
                    
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    disk_info[partition.mountpoint] = {
                        'total': usage.total / (1024**3),
                        'free': usage.free / (1024**3),
                        'used': usage.used / (1024**3),
                        'device': partition.device,
                        'fstype': partition.fstype
                    }
                
                except (PermissionError, OSError):
                    continue
        
        except Exception:
            # Fallback: just get current directory info
            try:
                usage = psutil.disk_usage('.')
                disk_info['/'] = {
                    'total': usage.total / (1024**3),
                    'free': usage.free / (1024**3),
                    'used': usage.used / (1024**3),
                    'device': 'unknown',
                    'fstype': 'unknown'
                }
            except:
                disk_info['/'] = {
                    'total': 100.0, 'free': 50.0, 'used': 50.0,
                    'device': 'unknown', 'fstype': 'unknown'
                }
        
        return disk_info
    
    def get_optimal_config_for_operation(self, operation_type: str, 
                                       file_count: int = 0, 
                                       total_size_mb: float = 0) -> Dict[str, Any]:
        """Get optimal configuration for specific operation types"""
        if not self.resources:
            self.get_system_resources()
        
        config = {
            'worker_count': self.resources.optimal_worker_counts.get('mixed_workload', 4),
            'batch_size': self.resources.recommended_batch_size,
            'memory_limit_gb': self.resources.memory_limits['batch_operation_max'],
            'use_async_io': self.resources.io_optimization['use_memory_mapping'],
            'compression_level': self.resources.io_optimization['compression_level']
        }
        
        # Adjust based on operation type
        if operation_type == 'file_hashing':
            config['worker_count'] = self.resources.optimal_worker_counts['file_hashing']
            config['batch_size'] = min(config['batch_size'] * 2, 200)
        
        elif operation_type == 'batch_processing':
            config['worker_count'] = self.resources.optimal_worker_counts['batch_processing']
            
        elif operation_type == 'io_intensive':
            config['worker_count'] = self.resources.optimal_worker_counts['io_intensive']
            config['batch_size'] = max(20, config['batch_size'] // 2)
        
        elif operation_type == 'cpu_intensive':
            config['worker_count'] = self.resources.optimal_worker_counts['cpu_intensive']
        
        # Adjust based on workload size
        if file_count > 0:
            if file_count < 50:
                config['worker_count'] = min(config['worker_count'], 4)
                config['batch_size'] = min(config['batch_size'], 10)
            elif file_count > 10000:
                config['worker_count'] = self.resources.optimal_worker_counts['batch_processing']
                config['batch_size'] = min(1000, config['batch_size'] * 2)
        
        # Memory-based adjustments
        if total_size_mb > 0:
            estimated_memory_gb = total_size_mb / 1024
            if estimated_memory_gb > config['memory_limit_gb']:
                # Reduce batch size and worker count for large datasets
                scale_factor = config['memory_limit_gb'] / estimated_memory_gb
                config['batch_size'] = max(10, int(config['batch_size'] * scale_factor))
                config['worker_count'] = max(2, int(config['worker_count'] * scale_factor))
        
        return config

# Global instance
system_resource_manager = SystemResourceManager()

# Convenience functions
def get_system_resources() -> SystemResources:
    """Get system resources using global manager"""
    return system_resource_manager.get_system_resources()

def get_optimal_config(operation_type: str, file_count: int = 0, total_size_mb: float = 0) -> Dict[str, Any]:
    """Get optimal configuration for an operation"""
    return system_resource_manager.get_optimal_config_for_operation(operation_type, file_count, total_size_mb)

def get_optimal_worker_count(task_type: str = "mixed", file_count: int = 0) -> int:
    """Get optimal worker count for a specific task type"""
    resources = get_system_resources()
    
    base_count = resources.optimal_worker_counts.get(task_type, resources.recommended_cpu_cores)
    
    # Adjust based on file count
    if file_count > 0:
        if file_count < 20:
            return min(base_count, 2)
        elif file_count > 1000:
            return min(32, base_count * 2)
    
    return base_count