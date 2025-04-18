# tests/test_training.py
"""
Unit Tests for Training Module (src.training.trainer)
"""

import unittest
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import os
import logging

# Import components to be tested
from src.training.trainer import EnhancedTrainer, TrainingMetrics
# Import dependencies needed for testing trainer (model, loader, config)
from src.core.models import ZoneClassifier # Example model
from src.data.loaders import EnhancedDataLoader # Example loader
from src.utils.config import TrainingConfig, DEFAULT_DEVICE, CHECKPOINT_DIR, LOG_DIR, TestConfig
from src.utils.helpers import dummy_input # For creating dummy data

logger = logging.getLogger(__name__)

# --- Test Setup ---
# Ensure relevant directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEST_ARTIFACT_DIR = TestConfig.TEST_ARTIFACT_DIR
os.makedirs(TEST_ARTIFACT_DIR, exist_ok=True)

# Create a dummy dataloader for testing the trainer loop
class DummyDataLoader:
    """A simple iterable that yields dummy batches."""
    def __init__(self, batch_size=4, num_batches=10, input_dim=128, num_classes=6, device=DEFAULT_DEVICE):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device

    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate dummy input based on model expectation
            # Assuming ZoneClassifier takes (batch, features) or (batch, seq, features)
            # Using dummy_input which gives (batch, seq, features) - adapt if needed
            # inputs = dummy_input(batch_size=self.batch_size, features=self.input_dim, device=self.device)
            # Example for (batch, features):
            inputs = torch.randn(self.batch_size, self.input_dim, device=self.device)

            # Generate dummy targets (shape depends on loss function)
            # For MSELoss with output (batch, num_classes):
            targets = torch.randn(self.batch_size, self.num_classes, device=self.device)
            # For CrossEntropyLoss with output (batch, num_classes):
            # targets = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

            yield inputs, targets

    def __len__(self):
        return self.num_batches


# --- Test Classes ---

class TrainingMetricsTests(unittest.TestCase):
    """Test suite for TrainingMetrics."""

    def setUp(self):
        self.metrics = TrainingMetrics(save_dir=TEST_ARTIFACT_DIR)
        self.test_file = TEST_ARTIFACT_DIR / "test_metrics.csv"

    def tearDown(self):
        if self.test_file.exists():
            os.remove(self.test_file)
        self.metrics.clear()

    def test_record_and_get_dataframe(self):
        logger.debug("Testing TrainingMetrics record and get_dataframe.")
        self.metrics.record(epoch=0, batch=1, loss=0.5, lr=1e-4, pruned_count=10, duration=0.1, batch_size=4)
        self.metrics.record(epoch=0, batch=2, loss=0.4, lr=1e-4, pruned_count=10, duration=0.11, batch_size=4)
        df = self.metrics.get_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df['epoch'].iloc[0], 0)
        self.assertEqual(df['loss'].iloc[1], 0.4)
        self.assertListEqual(list(df.columns), self.metrics.columns)

    def test_save_and_clear(self):
        logger.debug("Testing TrainingMetrics save and clear.")
        self.metrics.record(epoch=0, batch=1, loss=0.5, lr=1e-4, pruned_count=10, duration=0.1, batch_size=4)
        self.metrics.save(filename=self.test_file.name) # Save to test dir
        self.assertTrue(self.test_file.exists())

        # Check content (optional)
        df_loaded = pd.read_csv(self.test_file)
        self.assertEqual(len(df_loaded), 1)

        self.metrics.clear()
        df_cleared = self.metrics.get_dataframe()
        self.assertTrue(df_cleared.empty)


@unittest.skip("Training tests not yet fully implemented or require significant setup.")
class EnhancedTrainerTests(unittest.TestCase):
    """Test suite for EnhancedTrainer."""

    def setUp(self):
        """Set up a dummy model, loader, and criterion for trainer tests."""
        self.input_dim = 128
        self.num_classes = 6
        self.batch_size = 4
        self.num_batches = 5 # Keep low for testing

        # Use a simple model for testing the trainer logic itself
        # self.model = ZoneClassifier(input_features=self.input_dim, num_classes=self.num_classes, device=DEFAULT_DEVICE)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, self.num_classes)
        ).to(DEFAULT_DEVICE)

        self.data_loader = DummyDataLoader(
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            device=DEFAULT_DEVICE
        )
        self.criterion = nn.MSELoss() # Match dummy target shape

        self.trainer = EnhancedTrainer(
            model=self.model,
            data_loader=self.data_loader,
            criterion=self.criterion,
            device=DEFAULT_DEVICE
        )
        # Override config for testing if needed
        self.trainer.config.MAX_EPOCHS = 2
        self.trainer.config.PRUNE_INTERVAL_EPOCHS = 0 # Disable pruning for basic test
        self.trainer.config.CHECKPOINT_INTERVAL_BATCH_PERCENT = 0 # Disable intra-epoch checkpoints

        self.checkpoint_files_to_clean = []


    def tearDown(self):
        """Clean up any generated checkpoint files."""
        for f in self.checkpoint_files_to_clean:
            filepath = CHECKPOINT_DIR / f
            if filepath.exists():
                try:
                    os.remove(filepath)
                    logger.debug(f"Removed test checkpoint: {filepath}")
                except OSError as e:
                    logger.error(f"Failed to remove test checkpoint {filepath}: {e}")

    def test_train_epoch(self):
        """Test running a single training epoch."""
        logger.debug("Testing EnhancedTrainer train_epoch.")
        initial_params = [p.clone() for p in self.model.parameters()]

        avg_loss = self.trainer.train_epoch()

        self.assertIsInstance(avg_loss, float)
        self.assertGreaterEqual(avg_loss, 0.0)
        # Check if model parameters changed
        params_changed = any(not torch.equal(p_init, p_final) for p_init, p_final in zip(initial_params, self.model.parameters()))
        self.assertTrue(params_changed, "Model parameters should change after one epoch.")
        # Check if metrics were recorded
        self.assertEqual(len(self.trainer.metrics.metrics_data), self.num_batches)
        logger.debug(f"train_epoch finished with avg_loss: {avg_loss:.4f}")

    def test_full_train_loop(self):
        """Test running the full training loop for a few epochs."""
        logger.debug("Testing EnhancedTrainer full train loop.")
        initial_params = [p.clone() for p in self.model.parameters()]
        max_epochs = self.trainer.config.MAX_EPOCHS # Use the (potentially overridden) config value

        self.trainer.train() # Run the loop

        self.assertEqual(self.trainer.current_epoch, max_epochs -1) # Epoch counter increments after loop finishes
        # Check if parameters changed significantly from start
        params_changed = any(not torch.equal(p_init, p_final) for p_init, p_final in zip(initial_params, self.model.parameters()))
        self.assertTrue(params_changed, "Model parameters should change after full training.")
        # Check total metrics recorded
        self.assertEqual(len(self.trainer.metrics.metrics_data), self.num_batches * max_epochs)
        # Check if checkpoints were saved (at least end-of-epoch ones)
        for i in range(max_epochs):
             chkpt_name = f"{type(self.model).__name__}_epoch_{i}_end.pt"
             self.assertTrue((CHECKPOINT_DIR / chkpt_name).exists(), f"Checkpoint {chkpt_name} not found.")
             self.checkpoint_files_to_clean.append(chkpt_name) # Mark for cleanup

        logger.debug("Full train loop test finished.")

    # Add tests for pruning, checkpoint loading/saving specifically


# Standard unittest execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running training tests...")
    unittest.main()
