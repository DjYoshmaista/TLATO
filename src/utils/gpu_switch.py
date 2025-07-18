# src/utils/gpu_switch.py
"""
GPU/CPU Switching Utility

Provides functions to check GPU compute capability and assist in selecting
appropriate libraries (e.g., CUDA-accelerated vs. CPU-based).
"""

import importlib
import pandas as pd
import numpy as np
import logging
import torch
from typing import List, Optional, Union
from pathlib import Path
import inspect
import os
import sys
import dotenv
from src.utils.logger import configure_logging, log_statement
configure_logging()

dotenv.load_dotenv()

# Use standard logging
logger = logging.getLogger(__name__)

def get_pycuda_compute_capability(device_index=0):
    """
    Checks the compute capability of a CUDA device using PyCUDA.

    Args:
        device_index (int): The index of the CUDA device to check.

    Returns:
        str: Compute capability (e.g., "7.5") or None if PyCUDA/CUDA unavailable
             or device index is invalid.
    """
    try:
        cuda = importlib.import_module('pycuda.driver')
        cuda.init() # Initialize the CUDA driver
        if device_index >= cuda.Device.count():
            log_statement(loglevel=str("warning"), logstatement=str(f"Device index {device_index} out of range. Found {cuda.Device.count()} devices."), main_logger=str(__name__))
            return None
        device = cuda.Device(device_index)
        major = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR)
        minor = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR)
        capability = f"{major}.{minor}"
        log_statement(loglevel=str("info"), logstatement=str(f"Device {device_index} ({device.name()}): Compute Capability {capability}"), main_logger=str(__name__))
        return capability
    except ImportError:
        log_statement(loglevel=str("warning"), logstatement=str("PyCUDA not found. Cannot check GPU compute capability."), main_logger=str(__name__))
        return None
    except cuda.RuntimeError as e:
        log_statement(loglevel=str("error"), logstatement=str(f"PyCUDA runtime error during capability check: {e}"), main_logger=str(__name__))
        return None
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during compute capability check: {e}"), main_logger=str(__name__))
        return None

def set_compute_device(
    memory_required_mb: Optional[int] = None,
    backend_preference: Optional[List[str]] = None,
    device_id_preference: Optional[Union[int, str]] = None,
    force_cpu_env_var: str = 'FORCE_CPU',
    cuda_visible_env_var: str = 'CUDA_VISIBLE_DEVICES'
) -> str:
    """
    Determines and returns the appropriate compute device ('cuda:N' or 'cpu').

    Checks for GPU (CUDA via torch) availability, optional minimum memory requirement,
    honors FORCE_CPU and CUDA_VISIBLE_DEVICES environment variables, and allows
    specifying backend preference order and specific device ID preference.

    Args:
        memory_required_mb (Optional[int]): Minimum free GPU memory required in MB.
                                             If specified and no suitable GPU is found,
                                             falls back to CPU.
        backend_preference (Optional[List[str]]): Order of preferred backends.
                                                  Default: ['cuda', 'cpu'].
                                                  Example: ['cuda', 'mps', 'cpu'].
                                                  Currently only 'cuda' and 'cpu' via torch checked.
        device_id_preference (Optional[Union[int, str]]): Preferred GPU device ID (e.g., 0, 1)
                                                           or 'fastest' (not implemented, selects 0 if available),
                                                           or 'most_memory' (selects GPU with most free memory).
                                                           Overrides CUDA_VISIBLE_DEVICES if specified.
        force_cpu_env_var (str): Name of the environment variable to force CPU usage (e.g., 'FORCE_CPU=1').
        cuda_visible_env_var (str): Name of the CUDA environment variable to control device visibility.

    Returns:
        str: The selected device string (e.g., 'cuda:0', 'cpu').
    """
    if os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        visible_cuda_env = os.getenv('CUDA_VISIBLE_DEVICES')
    else:
        visible_cuda_env = None
    frame = inspect.currentframe()
    # Use caller's filename if possible for more context, fallback to module
    caller_frame = inspect.getouterframes(frame, 2)
    caller_context = Path(caller_frame[1].filename).name if len(caller_frame) > 1 else LOG_INS_MODULE
    LOG_INS_CUST = f"{caller_context} -> {LOG_INS}::set_compute_device::{frame.f_lineno if frame else 'UnknownLine'}"
    set_device_logger_name = LOG_INS_MODULE # Log within the utility's scope

    log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Determining compute device...", main_logger=__file__)
    log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Params - memory_required_mb={memory_required_mb}, backend_preference={backend_preference}, device_id_preference={device_id_preference}", main_logger=__file__)

    # --- Set Defaults ---
    if backend_preference is None:
        backend_preference = ['cuda', 'cpu'] # Default order

    selected_device = 'cpu' # Default to CPU

    # --- Check Environment Variable Overrides ---
    force_cpu = os.environ.get(force_cpu_env_var, '0')
    if force_cpu not in ('0', 'false', 'False', ''):
        log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Environment variable '{force_cpu_env_var}' set. Forcing CPU usage.", main_logger=__file__)
        return 'cpu'

    cuda_visible_devices = os.environ.get(cuda_visible_env_var)
    log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Env Var '{cuda_visible_env_var}' = {cuda_visible_devices}", main_logger=__file__)

    # --- Iterate through preferred backends ---
    for backend in backend_preference:
        backend = backend.lower()
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Checking preferred backend: '{backend}'", main_logger=__file__)

        if backend == 'cuda':
            try:
                if not torch.cuda.is_available():
                    log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>CUDA backend requested but torch reports CUDA not available.", main_logger=__file__)
                    continue # Try next preferred backend

                num_devices = torch.cuda.device_count()
                if num_devices == 0:
                    log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>CUDA backend available but no CUDA devices found by torch.", main_logger=__file__)
                    continue

                log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>CUDA available. Found {num_devices} device(s).", main_logger=__file__)

                # --- Determine candidate devices based on environment and preferences ---
                candidate_devices = []
                if device_id_preference is not None:
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Using device_id_preference: {device_id_preference}", main_logger=__file__)
                    if isinstance(device_id_preference, int):
                        if 0 <= device_id_preference < num_devices:
                             candidate_devices = [device_id_preference]
                        else:
                             log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Preference device_id {device_id_preference} is out of range (0-{num_devices-1}). Ignoring preference.", main_logger=__file__)
                             # Fall back to checking all devices respecting CUDA_VISIBLE_DEVICES
                             if cuda_visible_devices:
                                 candidate_devices = [int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip().isdigit() and 0 <= int(d.strip()) < num_devices]
                             else:
                                 candidate_devices = list(range(num_devices))
                    elif isinstance(device_id_preference, str):
                         pref_lower = device_id_preference.lower()
                         if pref_lower == 'most_memory':
                              # Check memory on all available devices
                              mem_info = []
                              all_possible_devices = list(range(num_devices))
                              # Filter by CUDA_VISIBLE_DEVICES if set, as torch respects it implicitly
                              if cuda_visible_devices:
                                   visible_ids = {int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip().isdigit()}
                                   all_possible_devices = [idx for idx in all_possible_devices if idx in visible_ids]

                              for i in all_possible_devices:
                                   try:
                                       free_mem, total_mem = torch.cuda.mem_get_info(i)
                                       mem_info.append({'id': i, 'free_mb': free_mem / (1024**2)})
                                       log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Device {i}: Free Memory = {free_mem / (1024**2):.2f} MB", main_logger=__file__)
                                   except Exception as mem_e:
                                       log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not get memory info for device {i}: {mem_e}", main_logger=__file__)
                              if mem_info:
                                   # Sort by free memory descending
                                   mem_info.sort(key=lambda x: x['free_mb'], reverse=True)
                                   # Use the one with most memory as the single candidate
                                   candidate_devices = [mem_info[0]['id']]
                                   log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Preference 'most_memory' selected device {candidate_devices[0]} with {mem_info[0]['free_mb']:.2f} MB free.", main_logger=__file__)
                              else:
                                   log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not get memory info for any device to satisfy 'most_memory' preference. Checking default devices.", main_logger=__file__)
                                   candidate_devices = all_possible_devices # Fallback check

                         elif pref_lower == 'fastest':
                              # Implementation to find the "fastest" device based on properties
                              log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Processing 'fastest' device preference.", main_logger=__file__)
                              device_properties = []
                              possible_gpus = list(range(num_devices))
                              # Respect CUDA_VISIBLE_DEVICES when checking properties
                              if visible_cuda_env:
                                  try:
                                       visible_ids = {int(d.strip()) for d in visible_cuda_env.split(',') if d.strip().isdigit()}
                                       possible_gpus = [idx for idx in possible_gpus if idx in visible_ids]
                                       log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>'fastest' check constrained by {cuda_visible_env_var} to devices: {possible_gpus}", main_logger=__file__)
                                  except Exception as vis_e:
                                       log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not parse {cuda_visible_env_var} ('{visible_cuda_env}'): {vis_e}. Checking all GPUs for 'fastest'.", main_logger=__file__)
                                       possible_gpus = list(range(num_devices)) # Fallback

                              for i in possible_gpus:
                                   try:
                                       props = torch.cuda.get_device_properties(i)
                                       device_properties.append({
                                           'id': i,
                                           'compute_capability': (props.major, props.minor),
                                           'total_memory_mb': props.total_memory / (1024**2),
                                           'clock_mhz': props.multi_processor_count * props.clock_rate / 1000, # Simple heuristic for compute power
                                           'name': props.name
                                       })
                                   except Exception as prop_e:
                                       log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not get properties for device {i}: {prop_e}", main_logger=__file__)

                              if device_properties:
                                   # Sort primarily by Compute Capability (desc), then by total memory (desc), then clock heuristic (desc)
                                   device_properties.sort(key=lambda x: (x['compute_capability'], x['total_memory_mb'], x['clock_mhz']), reverse=True)
                                   # Select the top device after sorting
                                   best_device_info = device_properties[0]
                                   candidate_devices = [best_device_info['id']]
                                   log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Preference 'fastest' selected candidate device {best_device_info['id']} ('{best_device_info['name']}', CC: {best_device_info['compute_capability'][0]}.{best_device_info['compute_capability'][1]}, Mem: {best_device_info['total_memory_mb']:.0f}MB)", main_logger=__file__)
                              else:
                                   log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not get properties for any visible/available device to satisfy 'fastest' preference.", main_logger=__file__)
                                   # Fall through to check default available devices (will use possible_gpus)
                                   candidate_devices = possible_gpus
                         else: # Try parsing as specific ID list (e.g., "0,2")
                              try:
                                   candidate_devices = [int(d.strip()) for d in device_id_preference.split(',') if d.strip().isdigit() and 0 <= int(d.strip()) < num_devices]
                                   if not candidate_devices: log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not parse any valid device IDs from preference '{device_id_preference}'. Ignoring.", main_logger=__file__)
                              except Exception:
                                   log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not parse device preference string '{device_id_preference}'. Ignoring.", main_logger=__file__)
                                   # Fall back logic (respect CUDA_VISIBLE_DEVICES)
                                   if cuda_visible_devices:
                                       candidate_devices = [int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip().isdigit() and 0 <= int(d.strip()) < num_devices]
                                   else:
                                       candidate_devices = list(range(num_devices))

                elif cuda_visible_devices:
                     # Use devices specified in CUDA_VISIBLE_DEVICES if preference not set
                     log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}", main_logger=__file__)
                     try:
                          candidate_devices = [int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip().isdigit()]
                          # Filter out invalid IDs relative to actual device count
                          candidate_devices = [d for d in candidate_devices if 0 <= d < num_devices]
                          if not candidate_devices:
                               log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>CUDA_VISIBLE_DEVICES ('{cuda_visible_devices}') specifies no valid devices (0-{num_devices-1}). Falling back to CPU.", main_logger=__file__)
                               return 'cpu' # Explicitly return CPU if env var makes no GPUs available
                     except Exception:
                          log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>Could not parse CUDA_VISIBLE_DEVICES ('{cuda_visible_devices}'). Checking all devices.", main_logger=__file__)
                          candidate_devices = list(range(num_devices)) # Fallback
                else:
                     # No preference, no env var, check all available devices
                     log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>No device preference or CUDA_VISIBLE_DEVICES set. Checking all {num_devices} devices.", main_logger=__file__)
                     candidate_devices = list(range(num_devices))

                log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Candidate device IDs to check: {candidate_devices}", main_logger=__file__)

                # --- Check candidates for suitability (memory) ---
                for device_id in candidate_devices:
                    log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Checking suitability of device cuda:{device_id}", main_logger=__file__)
                    try:
                        props = torch.cuda.get_device_properties(device_id)
                        log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Device cuda:{device_id}: {props.name}, Total Memory: {props.total_memory / (1024**2):.2f} MB", main_logger=__file__)

                        # Check memory requirement
                        if memory_required_mb is not None:
                             free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                             free_mem_mb = free_mem / (1024**2)
                             log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Device cuda:{device_id}: Free memory = {free_mem_mb:.2f} MB (Required: {memory_required_mb} MB)", main_logger=__file__)
                             if free_mem_mb < memory_required_mb:
                                 log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Device cuda:{device_id} skipped. Insufficient free memory (Free: {free_mem_mb:.2f} MB, Required: {memory_required_mb} MB).", main_logger=__file__)
                                 continue # Try next candidate device

                        # If memory check passes (or not required), select this device
                        selected_device = f'cuda:{device_id}'
                        log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Selected device: {selected_device}", main_logger=__file__)
                        # Break inner loop (device check) and outer loop (backend check)
                        return selected_device # Return immediately once a suitable CUDA device is found

                    except Exception as e:
                         log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Error checking properties/memory for device cuda:{device_id}: {e}", main_logger=__file__, exc_info=True)
                         continue # Try next candidate device

                # If loop finishes without finding a suitable CUDA device
                log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>No suitable CUDA device found matching criteria.", main_logger=__file__)
                # Continue to next preferred backend in the outer loop

            # Placeholder for other backends like MPS (Apple Silicon) or ROCm (AMD)
            # elif backend == 'mps':
            #     # Add checks for torch.backends.mps.is_available() etc.
            #     log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>MPS backend check not implemented.", main_logger=__file__)
            #     continue
            # elif backend == 'rocm':
            #     # Add checks for ROCm availability (might need different libraries)
            #     log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>ROCm backend check not implemented.", main_logger=__file__)
            #     continue
            except Exception as e:
                log_statement('error', f"{LOG_INS_CUST}:ERROR>>Error setting compute device: {e}", __file__, True)
        elif backend == 'cpu':
            # If CPU is explicitly preferred and reached, select it
            log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Selecting preferred backend: 'cpu'", main_logger=__file__)
            return 'cpu' # Return CPU if it's in preference list

        # --- Fallback ---
        # If loop finishes without returning (e.g., only CUDA preferred but none suitable)
        log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>No suitable preferred device found. Defaulting to 'cpu'.", main_logger=__file__)
        return 'cpu' # Default to CPU if no preferred/suitable device found

def check_gpu_support(min_capability_threshold=7.0):
    """
    Checks if a suitable GPU (meeting the minimum compute capability) is available.

    Args:
        min_capability_threshold (float): The minimum required compute capability.

    Returns:
        bool: True if a suitable GPU is found, False otherwise.
    """
    try:
        cuda = importlib.import_module('pycuda.driver')
        cuda.init()
        num_devices = cuda.Device.count()
        if num_devices == 0:
            log_statement(loglevel=str("info"), logstatement=str("No CUDA devices found. Using CPU."), main_logger=str(__name__))
            return False

        for i in range(num_devices):
            capability_str = get_pycuda_compute_capability(i)
            if capability_str:
                capability = float(capability_str)
                if capability >= min_capability_threshold:
                    log_statement(loglevel=str("info"), logstatement=str(f"Found suitable GPU (Device {i}) with capability {capability} >= {min_capability_threshold}."), main_logger=str(__name__))
                    return True
                else:
                     log_statement(loglevel=str("warning"), logstatement=str(f"Device {i} capability {capability} is below threshold {min_capability_threshold}."), main_logger=str(__name__))

        log_statement(loglevel=str("warning"), logstatement=str(f"No GPU found with compute capability >= {min_capability_threshold}. Using CPU."), main_logger=str(__name__))
        return False

    except ImportError:
        log_statement(loglevel=str("info"), logstatement=str("PyCUDA not found. Assuming CPU execution."), main_logger=str(__name__))
        return False
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Error during GPU support check: {e}. Defaulting to CPU."), main_logger=str(__name__))
        return False

def get_compute_backend(min_capability_threshold=7.0):
    """
    Determines the compute backend (GPU or CPU) and imports relevant libraries.

    Args:
        min_capability_threshold (float): Minimum compute capability to prefer GPU.

    Returns:
        tuple: (backend_name, pd_module, np_module)
               - backend_name (str): 'gpu' or 'cpu'
               - pd_module: The pandas-compatible module (cudf or pandas)
               - np_module: The numpy-compatible module (cupy or numpy)
    """
    use_gpu = check_gpu_support(min_capability_threshold)

    if use_gpu:
        try:
            # Try importing GPU libraries
            cudf = importlib.import_module('cudf')
            cupy = importlib.import_module('cupy')
            log_statement(loglevel=str("info"), logstatement=str("Successfully imported cuDF and CuPy. Using GPU backend."), main_logger=str(__name__))
            return 'gpu', cudf, cupy
        except ImportError as e:
            log_statement(loglevel=str("warning"), logstatement=str(f"GPU support detected, but failed to import cuDF/CuPy: {e}. Falling back to CPU."), main_logger=str(__name__))
            return 'cpu', pd, np
    else:
        log_statement(loglevel=str("info"), logstatement=str("Using CPU backend (pandas and numpy)."), main_logger=str(__name__))
        return 'cpu', pd, np

# Example Usage (can be removed or placed in a separate script/notebook)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     backend, pd_lib, np_lib = get_compute_backend(min_capability_threshold=7.0)
#     print(f"Selected backend: {backend}")
#
#     # Example data
#     my_data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
#     my_array = [10, 20, 30]
#
#     # Use the selected libraries
#     df = pd_lib.DataFrame(my_data)
#     arr = np_lib.array(my_array)
#
#     print("\nDataFrame type:", type(df))
#     print(df)
#
#     print("\nArray type:", type(arr))
#     print(arr)
#
#     if backend == 'gpu':
#         # Example GPU-specific operation
#         arr_squared = arr ** 2
#         print("\nCuPy array squared:")
#         print(arr_squared)
#     else:
#         # Example CPU-specific operation
#         arr_squared = arr ** 2
#         print("\nNumPy array squared:")
#         print(arr_squared)

