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
            logger.info(f"SemanticLabeler initialized for tests. Embedding size: {cls.embedding_size}")
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
        logger.debug("Testing generate_label output type.")
        label = self.labeler.generate_label(self.test_embedding)
        self.assertIsInstance(label, str, "generate_label should return a string.")
        self.assertNotIn("error", label.lower(), f"Label generation returned an error state: {label}")
        logger.debug(f"generate_label returned: {label}")

    def test_label_generation_known_labels(self):
        """Test if generate_label returns one of the expected labels."""
        logger.debug("Testing generate_label output value.")
        label = self.labeler.generate_label(self.test_embedding)
        # Expected labels based on the implementation in labeler.py
        expected_labels = ["high_level_cognition", "basic_processing", "unclassified"]
        self.assertIn(label, expected_labels, f"Generated label '{label}' not in expected labels {expected_labels}.")
        logger.debug(f"generate_label returned expected value: {label}")

    def test_recursive_labeling_output_type(self):
        """Test if recursive_labeling returns a list of strings."""
        logger.debug("Testing recursive_labeling output type.")
        labels = self.labeler.recursive_labeling(self.test_embeddings_list)
        self.assertIsInstance(labels, list, "recursive_labeling should return a list.")
        # Check if all elements in the list are strings (or handle nested lists if recursion is complex)
        self.assertTrue(all(isinstance(lbl, str) for lbl in labels), "All items in recursive_labeling output should be strings.")
        # Check for errors
        self.assertFalse(any("error" in str(lbl).lower() for lbl in labels), f"Recursive labeling returned error states: {labels}")
        logger.debug(f"recursive_labeling returned: {labels}")

    def test_recursive_labeling_output_length(self):
        """Test if recursive_labeling returns a list of the correct length."""
        logger.debug("Testing recursive_labeling output length.")
        labels = self.labeler.recursive_labeling(self.test_embeddings_list)
        self.assertEqual(len(labels), len(self.test_embeddings_list), "Output list length should match input list length.")
        logger.debug("recursive_labeling output length validated.")

    def test_recursive_labeling_max_depth(self):
        """Test the max recursion depth mechanism."""
        logger.debug("Testing recursive_labeling max depth.")
        # Temporarily set max depth to 0 for testing
        original_max_depth = self.labeler.config.MAX_RECURSION_DEPTH
        self.labeler.config.MAX_RECURSION_DEPTH = 0
        try:
            labels = self.labeler.recursive_labeling(self.test_embeddings_list, depth=0) # Start at depth 0
            self.assertTrue(all(lbl == "max_depth_exceeded" for lbl in labels), "Labels should indicate max depth reached.")
        finally:
            # Restore original max depth
            self.labeler.config.MAX_RECURSION_DEPTH = original_max_depth
        logger.debug("Max depth test passed.")

    # Add more tests:
    # - Test with different embedding shapes (e.g., batched) if supported
    # - Test specific similarity scores if reference embeddings are stable/mocked
    # - Test edge cases like empty input list


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running analysis (labeler) tests...")
    unittest.main()

