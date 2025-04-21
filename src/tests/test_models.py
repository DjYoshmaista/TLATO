# tests/test_models.py
"""
Unit Tests for Core Model Components (src.core.models)
"""

import unittest
import torch
import logging
import os

# Import components to be tested
from src.core.models import ZoneClassifier, GATLayer # Assuming these are defined
# Import utilities for testing (e.g., dummy input)
from src.utils.helpers import dummy_input
# Import config if needed for model parameters or device
from src.utils.config import DEFAULT_DEVICE, TestConfig

# Import PyTorch Geometric if GATLayer uses it
try:
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    # Define dummy classes if needed for tests when PyG is not installed
    class Data: pass
    class Batch: pass


logger = logging.getLogger(__name__)

# Ensure test artifact directory exists (if needed for saving test outputs)
# TEST_ARTIFACT_DIR = TestConfig.TEST_ARTIFACT_DIR
# os.makedirs(TEST_ARTIFACT_DIR, exist_ok=True)

@unittest.skipUnless(PYG_AVAILABLE, "PyTorch Geometric not found, skipping GATLayer tests")
class GATLayerTests(unittest.TestCase):
    """Test suite for GATLayer."""

    def setUp(self):
        """Set up for GATLayer tests."""
        self.input_dim = 128
        self.output_dim = 256
        self.num_nodes = 10
        self.num_edges = 20
        self.heads = 8 # Match default or configure
        # Create dummy graph data
        self.test_data = Data(
            x=torch.randn(self.num_nodes, self.input_dim),
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_edges))
        )
        # Ensure output_dim is divisible by heads for standard GATConv implementation
        self.gat_layer = GATLayer(self.input_dim, self.output_dim, heads=self.heads)
        self.gat_layer.to(DEFAULT_DEVICE)
        self.test_data = self.test_data.to(DEFAULT_DEVICE)


    def test_forward_pass_shape(self):
        """Test the output shape of the GATLayer forward pass."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing GATLayer forward pass output shape."), main_logger=str(__name__))
        # The forward pass in the placeholder returns the modified data object
        output_data = self.gat_layer(self.test_data)
        # Check the shape of the node features 'x' within the output data object
        expected_shape = (self.num_nodes, self.output_dim)
        self.assertEqual(output_data.x.shape, expected_shape)
        log_statement(loglevel=str("debug"), logstatement=str(f"GATLayer output shape validated: {output_data.x.shape}"), main_logger=str(__name__))

    def test_forward_pass_no_nan(self):
        """Test the GATLayer forward pass output for NaN values."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing GATLayer forward pass for NaN."), main_logger=str(__name__))
        output_data = self.gat_layer(self.test_data)
        self.assertFalse(torch.isnan(output_data.x).any(), "GATLayer output contains NaN values.")
        log_statement(loglevel=str("debug"), logstatement=str("GATLayer NaN check passed."), main_logger=str(__name__))

    # Add more tests: different graph structures, edge cases, batch processing


class ZoneClassifierTests(unittest.TestCase):
    """Test suite for the ZoneClassifier model."""

    def setUp(self):
        """Set up for ZoneClassifier tests."""
        self.input_dim = 128
        self.num_classes = 6
        self.batch_size = 4
        self.seq_len = 10 # Matches dummy_input default

        self.model = ZoneClassifier(input_features=self.input_dim, num_classes=self.num_classes, device=DEFAULT_DEVICE)
        self.model.eval() # Set to eval mode for testing unless testing training specifics

        # Generate dummy input based on expected model input type
        # If model uses PyG Data:
        if PYG_AVAILABLE and isinstance(getattr(self.model, 'gat_layer1', None), GATLayer):
             # Create a batch of graphs for testing
             data_list = []
             for _ in range(self.batch_size):
                  num_nodes = random.randint(5, 15)
                  num_edges = random.randint(num_nodes, num_nodes * 3)
                  data = Data(
                       x=torch.randn(num_nodes, self.input_dim),
                       edge_index=torch.randint(0, num_nodes, (2, num_edges))
                  )
                  data_list.append(data)
             # Import Batch class if needed: from torch_geometric.data import Batch
             self.dummy_input_data = Batch.from_data_list(data_list).to(DEFAULT_DEVICE)
             log_statement(loglevel=str("debug"), logstatement=str(f"Created dummy PyG Batch input for ZoneClassifier test."), main_logger=str(__name__))

        else:
             # Assume standard tensor input (adjust shape if needed)
             self.dummy_input_data = dummy_input(
                  batch_size=self.batch_size,
                  seq_len=self.seq_len, # Include if model expects sequence
                  features=self.input_dim,
                  device=DEFAULT_DEVICE
             )
             # If model expects (batch, features) instead of (batch, seq, features):
             # self.dummy_input_data = torch.randn(self.batch_size, self.input_dim, device=DEFAULT_DEVICE)
             log_statement(loglevel=str("debug"), logstatement=str(f"Created dummy Tensor input for ZoneClassifier test: shape={self.dummy_input_data.shape}"), main_logger=str(__name__))


    def test_forward_pass_shape(self):
        """Test the output shape of the ZoneClassifier forward pass."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing ZoneClassifier forward pass output shape."), main_logger=str(__name__))
        with torch.no_grad():
            output = self.model(self.dummy_input_data)

        # Expected shape depends on whether pooling is applied (for GNNs) or not
        # If GNN with pooling -> (batch_size, num_classes)
        # If MLP on sequence -> (batch_size, num_classes) potentially
        # If MLP operating per sequence item -> (batch_size, seq_len, num_classes)
        # Assuming output is (batch_size, num_classes) after pooling/flattening
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}")
        log_statement(loglevel=str("debug"), logstatement=str(f"ZoneClassifier output shape validated: {output.shape}"), main_logger=str(__name__))

    def test_forward_pass_no_nan(self):
        """Test the ZoneClassifier forward pass output for NaN values."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing ZoneClassifier forward pass for NaN."), main_logger=str(__name__))
        with torch.no_grad():
            output = self.model(self.dummy_input_data)
        self.assertFalse(torch.isnan(output).any(), "ZoneClassifier output contains NaN values.")
        log_statement(loglevel=str("debug"), logstatement=str("ZoneClassifier NaN check passed."), main_logger=str(__name__))

    # Add more tests: different input types, training mode vs eval mode, specific layer outputs


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_statement(loglevel=str("info"), logstatement=str("Running model tests..."), main_logger=str(__name__))
    unittest.main()

