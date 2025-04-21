# tests/test_analysis.py
"""
Unit Tests for Analysis Components (src.analysis.labeler)
"""

import unittest
import torch
import logging
import os

# Import components to be tested
from src.analysis.labeler import SemanticLabeler
# Import config if needed
from src.utils.config import LabelerConfig, DEFAULT_DEVICE, TestConfig

# Check if transformers is available (required for SemanticLabeler)
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("Transformers library not found. Cannot run SemanticLabeler tests.")


logger = logging.getLogger(__name__)

# Ensure test artifact directory exists (if needed)
# TEST_ARTIFACT_DIR = TestConfig.TEST_ARTIFACT_DIR
# os.makedirs(TEST_ARTIFACT_DIR, exist_ok=True)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not found, skipping SemanticLabeler tests")
class LabelerTests(unittest.TestCase):
    """Test suite for the SemanticLabeler class."""

    @classmethod
    def setUpClass(cls):
        """Set up class resources once (e.g., load model)."""
        # This might take time, so do it once per class if possible
        try:
            cls.labeler = SemanticLabeler(device=DEFAULT_DEVICE)
            # Determine embedding size from the loaded model
            cls.embedding_size = cls.labeler.model.config.hidden_size if cls.labeler.model else 768 # Default fallback
            log_statement(loglevel=str("info"), logstatement=str(f"SemanticLabeler initialized for tests. Embedding size: {cls.embedding_size}"), main_logger=str(__name__))
        except Exception as e:
             logger.critical(f"Failed to initialize SemanticLabeler for testing: {e}", exc_info=True)
             # Skip all tests in this class if setup fails
             raise unittest.SkipTest(f"SemanticLabeler initialization failed: {e}")


    def setUp(self):
        """Set up for each test method."""
        # Create a dummy embedding for tests
        self.test_embedding = torch.randn(self.embedding_size, device=DEFAULT_DEVICE)
        self.test_embeddings_list = [torch.randn(self.embedding_size, device=DEFAULT_DEVICE) for _ in range(3)]


    def test_label_generation_output_type(self):
        """Test if generate_label returns a string label."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing generate_label output type."), main_logger=str(__name__))
        label = self.labeler.generate_label(self.test_embedding)
        self.assertIsInstance(label, str, "generate_label should return a string.")
        self.assertNotIn("error", label.lower(), f"Label generation returned an error state: {label}")
        log_statement(loglevel=str("debug"), logstatement=str(f"generate_label returned: {label}"), main_logger=str(__name__))

    def test_label_generation_known_labels(self):
        """Test if generate_label returns one of the expected labels."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing generate_label output value."), main_logger=str(__name__))
        label = self.labeler.generate_label(self.test_embedding)
        # Expected labels based on the implementation in labeler.py
        expected_labels = ["high_level_cognition", "basic_processing", "unclassified"]
        self.assertIn(label, expected_labels, f"Generated label '{label}' not in expected labels {expected_labels}.")
        log_statement(loglevel=str("debug"), logstatement=str(f"generate_label returned expected value: {label}"), main_logger=str(__name__))

    def test_recursive_labeling_output_type(self):
        """Test if recursive_labeling returns a list of strings."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing recursive_labeling output type."), main_logger=str(__name__))
        labels = self.labeler.recursive_labeling(self.test_embeddings_list)
        self.assertIsInstance(labels, list, "recursive_labeling should return a list.")
        # Check if all elements in the list are strings (or handle nested lists if recursion is complex)
        self.assertTrue(all(isinstance(lbl, str) for lbl in labels), "All items in recursive_labeling output should be strings.")
        # Check for errors
        self.assertFalse(any("error" in str(lbl).lower() for lbl in labels), f"Recursive labeling returned error states: {labels}")
        log_statement(loglevel=str("debug"), logstatement=str(f"recursive_labeling returned: {labels}"), main_logger=str(__name__))

    def test_recursive_labeling_output_length(self):
        """Test if recursive_labeling returns a list of the correct length."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing recursive_labeling output length."), main_logger=str(__name__))
        labels = self.labeler.recursive_labeling(self.test_embeddings_list)
        self.assertEqual(len(labels), len(self.test_embeddings_list), "Output list length should match input list length.")
        log_statement(loglevel=str("debug"), logstatement=str("recursive_labeling output length validated."), main_logger=str(__name__))

    def test_recursive_labeling_max_depth(self):
        """Test the max recursion depth mechanism."""
        log_statement(loglevel=str("debug"), logstatement=str("Testing recursive_labeling max depth."), main_logger=str(__name__))
        # Temporarily set max depth to 0 for testing
        original_max_depth = self.labeler.config.MAX_RECURSION_DEPTH
        self.labeler.config.MAX_RECURSION_DEPTH = 0
        try:
            labels = self.labeler.recursive_labeling(self.test_embeddings_list, depth=0) # Start at depth 0
            self.assertTrue(all(lbl == "max_depth_exceeded" for lbl in labels), "Labels should indicate max depth reached.")
        finally:
            # Restore original max depth
            self.labeler.config.MAX_RECURSION_DEPTH = original_max_depth
        log_statement(loglevel=str("debug"), logstatement=str("Max depth test passed."), main_logger=str(__name__))

    # Add more tests:
    # - Test with different embedding shapes (e.g., batched) if supported
    # - Test specific similarity scores if reference embeddings are stable/mocked
    # - Test edge cases like empty input list


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_statement(loglevel=str("info"), logstatement=str("Running analysis (labeler) tests..."), main_logger=str(__name__))
    unittest.main()

