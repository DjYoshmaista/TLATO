# tests/test_utils.py
"""
Unit Tests for Utility Functions (src.utils.helpers, src.utils.gpu_switch)
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from pathlib import Path

# Import components to be tested
from src.utils.helpers import save_state, load_state, dummy_input
from src.utils.gpu_switch import get_compute_backend, check_gpu_support, get_pycuda_compute_capability
# Import config for paths and device
from src.utils.config import CHECKPOINT_DIR, DEFAULT_DEVICE, TestConfig
# Import a simple model for save/load tests
from src.core.models import ZoneClassifier # Example model

logger = logging.getLogger(__name__)

# Ensure checkpoint directory exists for testing save/load
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- Test Class for src.utils.helpers ---
class HelpersTests(unittest.TestCase):
    """Test suite for helper functions in src.utils.helpers."""

    def setUp(self):
        """Set up for helper tests."""
        # Create a simple model instance for save/load tests
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(DEFAULT_DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.test_filename = f"test_checkpoint_{self.__class__.__name__}.pt"
        self.test_filepath = CHECKPOINT_DIR / self.test_filename

    def tearDown(self):
        """Clean up test checkpoint files."""
        if self.test_filepath.exists():
            try:
                os.remove(self.test_filepath)
                log_statement(loglevel=str("debug"), logstatement=str(f"Removed test checkpoint file: {self.test_filepath}"), main_logger=str(__name__))
            except OSError as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error removing test checkpoint file {self.test_filepath}: {e}"), main_logger=str(__name__))

    def test_dummy_input_shape(self):
        """Test the shape of the generated dummy input."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing dummy_input shape."), main_logger=str(__name__))
        batch, seq, feat = 4, 10, 128
        input_tensor = dummy_input(batch_size=batch, seq_len=seq, features=feat, device=DEFAULT_DEVICE)
        expected_shape = (batch, seq, feat)
        self.assertEqual(input_tensor.shape, expected_shape)
        self.assertEqual(str(input_tensor.device), str(DEFAULT_DEVICE)) # Check device
        log_statement(loglevel=str("debug"), logstatement=str(f"dummy_input shape validated: {input_tensor.shape}"), main_logger=str(__name__))

    def test_save_load_cycle(self):
        """Test saving and loading model, optimizer, and scheduler state."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing save_state and load_state cycle."), main_logger=str(__name__))
        epoch_to_save = 5
        extra_meta = {"info": "test_metadata"}

        # Save state
        save_state(
            model=self.model,
            filename=self.test_filename,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch_to_save,
            **extra_meta
        )
        self.assertTrue(self.test_filepath.exists(), "Checkpoint file was not created.")

        # Create new instances to load into
        model_new = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(DEFAULT_DEVICE)
        optimizer_new = optim.Adam(model_new.parameters(), lr=0.1) # Different LR initially
        scheduler_new = optim.lr_scheduler.StepLR(optimizer_new, step_size=5, gamma=0.5) # Different params

        # Load state
        loaded_meta = load_state(
            model=model_new,
            filename=self.test_filename,
            optimizer=optimizer_new,
            scheduler=scheduler_new,
            device=DEFAULT_DEVICE
        )

        # --- Assertions ---
        # Check metadata
        self.assertIsNotNone(loaded_meta, "load_state returned None.")
        self.assertEqual(loaded_meta.get('epoch'), epoch_to_save, "Epoch mismatch after loading.")
        self.assertEqual(loaded_meta.get('info'), extra_meta['info'], "Metadata mismatch after loading.")

        # Check model state (compare parameters)
        self.assertEqual(str(self.model.state_dict()), str(model_new.state_dict()), "Model state mismatch after loading.")

        # Check optimizer state (compare state dicts, note LR might change due to scheduler step)
        # It's tricky to compare optimizers directly, comparing state dict structure is a start
        self.assertEqual(self.optimizer.state_dict().keys(), optimizer_new.state_dict().keys(), "Optimizer state keys mismatch.")
        # Check if LR was restored (might need scheduler step consideration)
        # self.assertEqual(optimizer_new.param_groups[0]['lr'], self.optimizer.param_groups[0]['lr'], "Optimizer LR mismatch.")

        # Check scheduler state
        self.assertEqual(self.scheduler.state_dict(), scheduler_new.state_dict(), "Scheduler state mismatch after loading.")

        log_statement(loglevel=str("debug"), logstatement=str("Save/load cycle test passed."), main_logger=str(__name__))

    def test_load_state_not_found(self):
        """Test load_state behavior when the file doesn't exist."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing load_state with non-existent file."), main_logger=str(__name__))
        loaded_meta = load_state(self.model, "non_existent_file.pt")
        self.assertIsNone(loaded_meta, "load_state should return None for non-existent file.")
        log_statement(loglevel=str("debug"), logstatement=str("Load non-existent file test passed."), main_logger=str(__name__))


# --- Test Class for src.utils.gpu_switch ---
# These tests might require specific hardware (NVIDIA GPU) and drivers/libraries (PyCUDA)
# Use unittest.skipIf to skip tests if dependencies are missing

# Check if pycuda can be imported
try:
    import pycuda.driver as cuda
    PYCUDA_AVAILABLE = True
    try:
        cuda.init() # Try initializing CUDA
        CUDA_INITIALIZED = True
        DEVICE_COUNT = cuda.Device.count()
    except Exception as e:
        log_statement(loglevel=str("warning"), logstatement=str(f"PyCUDA imported but failed to initialize CUDA: {e}. GPU tests may fail or be skipped."), main_logger=str(__name__))
        CUDA_INITIALIZED = False
        DEVICE_COUNT = 0
except ImportError:
    log_statement(loglevel=str("warning"), logstatement=str("PyCUDA not found. Skipping GPU switch tests."), main_logger=str(__name__))
    PYCUDA_AVAILABLE = False
    CUDA_INITIALIZED = False
    DEVICE_COUNT = 0


@unittest.skipUnless(PYCUDA_AVAILABLE and CUDA_INITIALIZED and DEVICE_COUNT > 0, "PyCUDA not available or CUDA not initialized or no devices found")
class GPUSwitchTests(unittest.TestCase):
    """Test suite for GPU switching functions in src.utils.gpu_switch."""

    def test_get_pycuda_compute_capability(self):
        """Test retrieving compute capability."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing get_pycuda_compute_capability."), main_logger=str(__name__))
        capability = get_pycuda_compute_capability(device_index=0)
        self.assertIsNotNone(capability, "Compute capability should not be None.")
        self.assertIsInstance(capability, str, "Compute capability should be a string.")
        self.assertRegex(capability, r"^\d+\.\d+$", "Compute capability format should be 'major.minor'.")
        log_statement(loglevel=str("info"), logstatement=str(f"Detected compute capability for device 0: {capability}"), main_logger=str(__name__))

    def test_check_gpu_support(self):
        """Test the GPU support check function."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing check_gpu_support."), main_logger=str(__name__))
        # Test with a threshold likely met by modern GPUs
        supported_high_thresh = check_gpu_support(min_capability_threshold=3.0)
        self.assertTrue(supported_high_thresh, "GPU should be supported with a low threshold.")
        # Test with a very high threshold (likely to fail)
        supported_low_thresh = check_gpu_support(min_capability_threshold=99.0)
        self.assertFalse(supported_low_thresh, "GPU should NOT be supported with an extremely high threshold.")
        log_statement(loglevel=str("debug"), logstatement=str("check_gpu_support tests passed."), main_logger=str(__name__))

    def test_get_compute_backend(self):
        """Test obtaining the compute backend and libraries."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing get_compute_backend."), main_logger=str(__name__))
        backend, pd_lib, np_lib = get_compute_backend(min_capability_threshold=3.0) # Use low threshold

        # Check if backend matches GPU availability and library imports
        # This depends on whether cudf/cupy are *also* installed
        try:
            import cudf
            import cupy
            LIBS_INSTALLED = True
        except ImportError:
            LIBS_INSTALLED = False

        if LIBS_INSTALLED:
            self.assertEqual(backend, 'gpu', "Backend should be 'gpu' if supported and libs installed.")
            self.assertTrue(hasattr(pd_lib, 'DataFrame') and 'cudf' in str(pd_lib), "pd_lib should be cuDF.")
            self.assertTrue(hasattr(np_lib, 'array') and 'cupy' in str(np_lib), "np_lib should be CuPy.")
        else:
            self.assertEqual(backend, 'cpu', "Backend should be 'cpu' if GPU libs not installed.")
            self.assertTrue(hasattr(pd_lib, 'DataFrame') and 'pandas' in str(pd_lib), "pd_lib should be pandas.")
            self.assertTrue(hasattr(np_lib, 'array') and 'numpy' in str(np_lib), "np_lib should be NumPy.")
        log_statement(loglevel=str("debug"), logstatement=str(f"get_compute_backend test passed with backend '{backend}'."), main_logger=str(__name__))


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_statement(loglevel=str("info"), logstatement=str("Running utility tests..."), main_logger=str(__name__))
    unittest.main()

