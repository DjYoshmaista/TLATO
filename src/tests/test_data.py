# tests/test_data.py
"""
Unit Tests for Data Modules (src.data.*)

Includes tests for readers, loaders, processing, and synthetic generation.
"""

import unittest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

# Import components to be tested
from src.data import readers, loaders, processing, synthetic
# Import config for paths, etc.
from src.utils.config import TestConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, SYNTHETIC_DATA_DIR, DEFAULT_DEVICE

logger = logging.getLogger(__name__)

# --- Test Setup ---
# Create dummy files in relevant directories for testing readers/loaders/processors
TEST_ARTIFACT_DIR = TestConfig.TEST_ARTIFACT_DIR
DUMMY_RAW_DIR = TEST_ARTIFACT_DIR / 'raw'
DUMMY_PROCESSED_DIR = TEST_ARTIFACT_DIR / 'processed'
DUMMY_TOKENIZED_DIR = TEST_ARTIFACT_DIR / 'tokenized'
DUMMY_SYNTHETIC_DIR = TEST_ARTIFACT_DIR / 'synthetic'

def setup_dummy_files():
    """Creates dummy files for testing."""
    logger.info(f"Setting up dummy data files in {TEST_ARTIFACT_DIR}")
    # Create directories
    for d in [DUMMY_RAW_DIR, DUMMY_PROCESSED_DIR, DUMMY_TOKENIZED_DIR, DUMMY_SYNTHETIC_DIR]:
        os.makedirs(d, exist_ok=True)

    # Create dummy raw CSV
    dummy_csv_path = DUMMY_RAW_DIR / 'dummy_data.csv'
    pd.DataFrame({'colA': [1, 2, 3], 'colB': ['x', 'y', 'z']}).to_csv(dummy_csv_path, index=False)

    # Create dummy raw TXT
    dummy_txt_path = DUMMY_RAW_DIR / 'dummy_text.txt'
    with open(dummy_txt_path, 'w') as f:
        f.write("This is line one.\nThis is line two.")

    # Create dummy processed NPY (e.g., from numerical processing)
    dummy_npy_path = DUMMY_PROCESSED_DIR / 'dummy_numeric_processed.npy.zst' # Assuming saved as compressed numpy
    dummy_numeric_data = np.array([[1.0], [2.5], [-0.5]], dtype=np.float32)
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=3)
        with open(dummy_npy_path, 'wb') as f:
            with cctx.stream_writer(f) as compressor:
                np.save(compressor, dummy_numeric_data)
    except Exception as e:
         logger.error(f"Failed to create dummy processed NPY file: {e}")


    # Create dummy tokenized PT
    dummy_pt_path = DUMMY_TOKENIZED_DIR / 'dummy_tokenized.pt'
    dummy_tensor_data = torch.randn(5, 128, device=DEFAULT_DEVICE) # Example tensor
    torch.save(dummy_tensor_data, dummy_pt_path)

    # Create dummy synthetic JSONL
    dummy_jsonl_path = DUMMY_SYNTHETIC_DIR / 'dummy_synthetic.jsonl'
    dummy_synth_samples = [
        {'input': list(np.random.rand(128) * 2 - 1), 'target': [0.1]*10},
        {'input': list(np.random.rand(128) * 2 - 1), 'target': [0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]}
    ]
    try:
        import json
        with open(dummy_jsonl_path, 'w') as f:
            for sample in dummy_synth_samples:
                f.write(json.dumps(sample) + '\n')
    except Exception as e:
         logger.error(f"Failed to create dummy synthetic JSONL file: {e}")


def tearDown_dummy_files():
    """Removes dummy files after testing."""
    logger.info(f"Tearing down dummy data files in {TEST_ARTIFACT_DIR}")
    import shutil
    if TEST_ARTIFACT_DIR.exists():
        try:
            shutil.rmtree(TEST_ARTIFACT_DIR)
        except OSError as e:
            logger.error(f"Error removing test artifact directory {TEST_ARTIFACT_DIR}: {e}")


# --- Test Classes ---

@unittest.skip("Data tests not yet implemented.")
class ReadersTests(unittest.TestCase):
    """Test suite for src.data.readers."""

    @classmethod
    def setUpClass(cls):
        setup_dummy_files()

    @classmethod
    def tearDownClass(cls):
        tearDown_dummy_files()

    def test_csv_reader(self):
        logger.debug("Testing CSVReader.")
        reader = readers.CSVReader(DUMMY_RAW_DIR / 'dummy_data.csv')
        df = reader.read()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['colA', 'colB'])
        # Add more assertions

    def test_txt_reader(self):
        logger.debug("Testing TXTReader.")
        reader = readers.TXTReader(DUMMY_RAW_DIR / 'dummy_text.txt')
        df = reader.read()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn('text', df.columns)
        self.assertIn("line one", df['text'].iloc[0])
        # Add more assertions

    def test_get_reader_class(self):
        logger.debug("Testing get_reader_class function.")
        self.assertIs(readers.get_reader_class('csv'), readers.CSVReader)
        self.assertIs(readers.get_reader_class('.TXT'), readers.TXTReader)
        self.assertIsNone(readers.get_reader_class('unknown'))
        # Add tests for optional readers (pdf, jsonl) based on availability

    def test_open_files_recursively(self):
        logger.debug("Testing open_files_recursively.")
        repo_path = TEST_ARTIFACT_DIR / "test_repo.csv"
        prog_path = TEST_ARTIFACT_DIR / "test_prog.csv"
        readers.open_files_recursively(DUMMY_RAW_DIR, repo_file=repo_path, progress_file=prog_path)
        self.assertTrue(repo_path.exists())
        self.assertTrue(prog_path.exists())
        repo_df = pd.read_csv(repo_path)
        # Check contents of repo_df based on dummy files
        self.assertIn("dummy_data.csv", repo_df['filename'].tolist())
        self.assertIn("Read", repo_df[repo_df['filename'] == 'dummy_data.csv']['read_status'].iloc[0])


@unittest.skip("Data tests not yet implemented.")
class LoadersTests(unittest.TestCase):
    """Test suite for src.data.loaders."""

    @classmethod
    def setUpClass(cls):
        setup_dummy_files()

    @classmethod
    def tearDownClass(cls):
        tearDown_dummy_files()

    def test_enhanced_data_loader(self):
        logger.debug("Testing EnhancedDataLoader.")
        # Test loading .pt files created in setup
        loader = loaders.EnhancedDataLoader(data_dir=DUMMY_TOKENIZED_DIR, batch_size=2, shuffle=False)
        batches = list(loader) # Consume iterator
        self.assertGreater(len(batches), 0, "Loader should yield at least one batch.")
        first_batch = batches[0]
        # Assuming loader yields tensors directly
        self.assertIsInstance(first_batch, torch.Tensor)
        self.assertEqual(first_batch.shape[0], 2) # Check batch size
        self.assertEqual(first_batch.shape[1], 128) # Check feature dim
        # Add more checks: device, number of batches, handling different file types

    def test_synthetic_data_loader(self):
        logger.debug("Testing SyntheticDataLoader.")
        loader = loaders.SyntheticDataLoader(data_dir=DUMMY_SYNTHETIC_DIR, batch_size=1, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 2, "Should yield 2 batches for 2 samples with batch_size=1.")
        input_batch, target_batch = batches[0]
        self.assertIsInstance(input_batch, torch.Tensor)
        self.assertIsInstance(target_batch, torch.Tensor)
        self.assertEqual(input_batch.shape, (1, 128))
        self.assertEqual(target_batch.shape, (1, 10))
        # Add more checks: device, content validation


@unittest.skip("Data tests not yet implemented.")
class ProcessingTests(unittest.TestCase):
    """Test suite for src.data.processing."""

    @classmethod
    def setUpClass(cls):
        setup_dummy_files()
        # Need a dummy repo file for DataRepository tests
        cls.repo_path = TEST_ARTIFACT_DIR / "test_processing_repo.csv.zst"
        # Ensure NLTK data is downloaded if text processing is tested
        if processing.NLTK_AVAILABLE:
             processing.download_nltk_data()


    @classmethod
    def tearDownClass(cls):
        tearDown_dummy_files()

    def setUp(self):
        # Create a fresh repo for each test to avoid interference
        if self.repo_path.exists():
             os.remove(self.repo_path)
        self.repo = processing.DataRepository(repo_path=self.repo_path)
        # Add dummy raw files to the repo for processing tests
        self.repo.update_entry(DUMMY_RAW_DIR / 'dummy_data.csv', status='discovered')
        self.repo.update_entry(DUMMY_RAW_DIR / 'dummy_text.txt', status='discovered')
        self.repo.save()


    def tearDown(self):
        # Clean up repo file
        if self.repo_path.exists():
             os.remove(self.repo_path)

    def test_data_repository_add_update(self):
        logger.debug("Testing DataRepository add/update.")
        test_path = DUMMY_RAW_DIR / 'dummy_data.csv'
        status = self.repo.get_status(test_path)
        self.assertEqual(status, 'discovered')

        proc_path = DUMMY_PROCESSED_DIR / 'dummy_data_processed.zst'
        self.repo.update_entry(test_path, status='processed', processed_path=proc_path)
        self.repo.save()

        # Reload repo to check persistence
        repo_reloaded = processing.DataRepository(repo_path=self.repo_path)
        status_reloaded = repo_reloaded.get_status(test_path)
        path_reloaded = repo_reloaded.get_processed_path(test_path)
        self.assertEqual(status_reloaded, 'processed')
        self.assertEqual(path_reloaded, proc_path.resolve())


    def test_data_processor_text(self):
        logger.debug("Testing DataProcessor text processing.")
        processor = processing.DataProcessor()
        # Process the dummy text file added in setUp
        processor.process_all(statuses_to_process=('discovered',)) # Only process newly discovered

        # Check repo status
        repo = processing.DataRepository(repo_path=self.repo_path) # Reload repo
        status = repo.get_status(DUMMY_RAW_DIR / 'dummy_text.txt')
        self.assertEqual(status, 'processed', "Text file status should be 'processed'.")
        processed_path = repo.get_processed_path(DUMMY_RAW_DIR / 'dummy_text.txt')
        self.assertIsNotNone(processed_path, "Processed path should exist.")
        self.assertTrue(processed_path.exists(), f"Processed file {processed_path} should exist.")
        # Add checks for content of processed file if needed

    # Add test_data_processor_numerical similarly

    def test_tokenizer(self):
        logger.debug("Testing Tokenizer.")
        # Setup: Ensure a file is marked as 'processed' with a valid processed_path
        processed_file_path = DUMMY_PROCESSED_DIR / 'dummy_numeric_processed.npy.zst'
        source_file_path = DUMMY_RAW_DIR / 'dummy_numeric.json' # Assume this was the source
        self.repo.update_entry(source_file_path, status='processed', processed_path=processed_file_path)
        self.repo.save()

        tokenizer = processing.Tokenizer()
        tokenizer.tokenize_all(statuses_to_process=('processed',))

        # Check repo status
        repo = processing.DataRepository(repo_path=self.repo_path) # Reload repo
        status = repo.get_status(source_file_path)
        self.assertEqual(status, 'tokenized', "File status should be 'tokenized'.")
        tokenized_path = repo.df[repo.df['source_filepath'] == str(source_file_path.resolve())]['tokenized_path'].iloc[0]
        self.assertTrue(tokenized_path, "Tokenized path should exist in repo.")
        self.assertTrue(Path(tokenized_path).exists(), f"Tokenized file {tokenized_path} should exist.")
        # Add checks for content of tokenized file if needed


@unittest.skip("Data tests not yet implemented.")
class SyntheticTests(unittest.TestCase):
    """Test suite for src.data.synthetic."""

    # These tests might require a running Ollama instance or mocking requests
    @unittest.skip("Skipping synthetic tests that require external API.")
    def test_synthetic_data_generation(self):
        # Mock the requests.post call
        # Instantiate SyntheticDataGenerator
        # Run generate_dataset (with small target_samples)
        # Check if output files are created in DUMMY_SYNTHETIC_DIR
        # Check format of generated files
        pass

    def test_validate_sample(self):
        logger.debug("Testing synthetic sample validation.")
        generator = synthetic.SyntheticDataGenerator() # Need instance for validation method
        valid_json_str = '{"input": [0.5]*128, "target": [0.1]*10}'
        invalid_json_str_keys = '{"inputs": [0.5]*128, "targets": [0.1]*10}'
        invalid_json_str_sum = '{"input": [0.5]*128, "target": [0.2]*10}'
        invalid_json_str_input_len = '{"input": [0.5]*10, "target": [0.1]*10}'

        self.assertIsNotNone(generator._validate_sample(valid_json_str))
        self.assertIsNone(generator._validate_sample(invalid_json_str_keys))
        self.assertIsNone(generator._validate_sample(invalid_json_str_sum))
        self.assertIsNone(generator._validate_sample(invalid_json_str_input_len))
        self.assertIsNone(generator._validate_sample("this is not json"))


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running data tests...")
    # Be cautious running tests that modify/delete files, ensure proper setup/teardown
    unittest.main()

