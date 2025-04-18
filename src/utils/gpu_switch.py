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
            logger.warning(f"Device index {device_index} out of range. Found {cuda.Device.count()} devices.")
            return None
        device = cuda.Device(device_index)
        major = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR)
        minor = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR)
        capability = f"{major}.{minor}"
        logger.info(f"Device {device_index} ({device.name()}): Compute Capability {capability}")
        return capability
    except ImportError:
        logger.warning("PyCUDA not found. Cannot check GPU compute capability.")
        return None
    except cuda.RuntimeError as e:
        logger.error(f"PyCUDA runtime error during capability check: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during compute capability check: {e}")
        return None

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
            logger.info("No CUDA devices found. Using CPU.")
            return False

        for i in range(num_devices):
            capability_str = get_pycuda_compute_capability(i)
            if capability_str:
                capability = float(capability_str)
                if capability >= min_capability_threshold:
                    logger.info(f"Found suitable GPU (Device {i}) with capability {capability} >= {min_capability_threshold}.")
                    return True
                else:
                     logger.warning(f"Device {i} capability {capability} is below threshold {min_capability_threshold}.")

        logger.warning(f"No GPU found with compute capability >= {min_capability_threshold}. Using CPU.")
        return False

    except ImportError:
        logger.info("PyCUDA not found. Assuming CPU execution.")
        return False
    except Exception as e:
        logger.error(f"Error during GPU support check: {e}. Defaulting to CPU.")
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
            logger.info("Successfully imported cuDF and CuPy. Using GPU backend.")
            return 'gpu', cudf, cupy
        except ImportError as e:
            logger.warning(f"GPU support detected, but failed to import cuDF/CuPy: {e}. Falling back to CPU.")
            return 'cpu', pd, np
    else:
        logger.info("Using CPU backend (pandas and numpy).")
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

