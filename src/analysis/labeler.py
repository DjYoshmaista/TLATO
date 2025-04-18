# src/training/trainer.py
"""
Model Training Module

Contains the core logic for training neural network models, including
optimization, learning rate scheduling, optional pruning, checkpointing,
and metrics tracking.
"""

# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os
import logging
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from typing import List, Dict, Tuple
from src.utils.logger import configure_logger, log_statement

configure_logger()

# Import project configuration and utilities
try:
    from src.utils.config import TrainingConfig, LabelerConfig, DEFAULT_DEVICE, CHECKPOINT_DIR, LOG_DIR
    from src.utils.helpers import save_state, load_state
except ImportError:
    # Fallback to local configuration if src.utils.config is not available
    from utils.config import TrainingConfig, LabelerConfig, DEFAULT_DEVICE, CHECKPOINT_DIR, LOG_DIR
    from utils.helpers import save_state, load_state
from src.core.models import ZoneClassifier # Example model
from src.data.loaders import EnhancedDataLoader # Example loader
# Import transformers safely
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define dummy classes to avoid runtime errors if transformers is missing,
    # although the class initialization will fail.
    class AutoTokenizer: pass
    class AutoModel: pass
    logging.error("Transformers library not found. SemanticLabeler requires it to function.")

# Ensure the CHECKPOINT_DIR and LOG_DIR exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler()
    ]
)

# Ensure the logger is set up correctly
log_statement(loglevel=str("info"), logstatement=str("Logger initialized for training module."), main_log=str(__name__))

class SemanticLabeler:
    """
    Assigns semantic labels to input embeddings by comparing them to reference embeddings.

    Uses cosine similarity and a configurable threshold to determine the best label.
    Supports a basic recursive labeling mechanism primarily for depth checking.
    """

    def __init__(self, device: str | torch.device = None):
        """
        Initializes the SemanticLabeler.

        Loads the tokenizer and embedding model specified in LabelerConfig.
        Generates reference embeddings for predefined labels.

        Args:
            device (str | torch.device, optional): The device to run inference on.
                                                   Defaults to config.DEFAULT_DEVICE.

        Raises:
            ImportError: If the 'transformers' library is not installed.
            RuntimeError: If model/tokenizer loading fails.
        """
        if not TRANSFORMERS_AVAILABLE:
             # Log critical error and raise exception to prevent instantiation
             log_statement(loglevel=str("critical"), logstatement=str("Transformers library is not installed, which is essential for SemanticLabeler."), main_log=str(__name__))
             raise ImportError("Transformers library is required for SemanticLabeler but not found.")

        try:
             self.config = LabelerConfig()
        except NameError:
             log_statement(loglevel=str("error"), logstatement=str("LabelerConfig class not found. Using internal defaults."), main_log=str(__name__))
             # Use internal defaults if config import failed (less ideal)
             class InternalLabelerConfig:
                 SIMILARITY_THRESHOLD = 0.7
                 MAX_RECURSION_DEPTH = 5
                 TOKENIZER_MODEL = 'bert-base-uncased'
                 EMBEDDING_MODEL = 'bert-base-uncased'
             self.config = InternalLabelerConfig()

        # Determine device, defaulting to config.DEFAULT_DEVICE
        self.device = device or DEFAULT_DEVICE
        self.tokenizer = None
        self.model = None
        self.reference_embeddings: Dict[str, torch.Tensor] = {} # Stores label -> embedding tensor

        try:
            # Load Tokenizer specified in config
            log_statement(loglevel=str("info"), logstatement=str(f"Loading tokenizer: {self.config.TOKENIZER_MODEL}"), main_log=str(__name__))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.TOKENIZER_MODEL)

            # Load Model specified in config and move to device
            log_statement(loglevel=str("info"), logstatement=str(f"Loading embedding model: {self.config.EMBEDDING_MODEL} onto device: {self.device}"), main_log=str(__name__))
            self.model = AutoModel.from_pretrained(self.config.EMBEDDING_MODEL).to(self.device)
            self.model.eval() # Set model to evaluation mode is crucial
            log_statement(loglevel=str("info"), logstatement=str("SemanticLabeler model and tokenizer loaded successfully."), main_log=str(__name__))

            # Generate reference embeddings for known labels
            self._generate_reference_embeddings()

        except Exception as e:
            log_statement(loglevel=str("critical"), logstatement=str(f"Failed to initialize SemanticLabeler model/tokenizer from Hugging Face: {e}", exc_info=True), main_log=str(__name__))
            # Propagate error to prevent usage of partially initialized object
            raise RuntimeError(f"SemanticLabeler initialization failed: {e}")

    def _get_embedding(self, text: str) -> torch.Tensor | None:
        """
        Generates a mean-pooled embedding for a given text string.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor | None: The embedding tensor on the configured device, or None on failure.
        """
        # Ensure component initialization succeeded
        if not self.tokenizer or not self.model:
            log_statement(loglevel=str("error"), logstatement=str("Tokenizer or model not properly loaded. Cannot generate embedding."), main_log=str(__name__))
            return None
        try:
            # Handle empty input text
            if not text:
                 log_statement(loglevel=str("warning"), logstatement=str("Attempted to get embedding for empty text."), main_log=str(__name__))
                 return None

            # Tokenize text and prepare inputs for the model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move inputs to the model's device

            # Perform inference without calculating gradients
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden state tokens for sentence embedding
                # Attention mask is used to ignore padding tokens in mean calculation
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
                mean_pooled_embedding = sum_embeddings / sum_mask

            # Return the first (and only) embedding, keeping it on the specified device
            return mean_pooled_embedding[0].detach()

        except Exception as e:
            # Log any error during embedding generation
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to generate embedding for text '{text[:50]}...': {e}", exc_info=True), main_log=str(__name__))
            return None

    def _generate_reference_embeddings(self):
        """
        Generates and stores reference embeddings for the predefined semantic labels.
        This method is called during initialization.
        """
        log_statement(loglevel=str("info"), logstatement=str("Generating reference embeddings for predefined labels..."), main_log=str(__name__))
        # Define descriptive text prompts for each known label
        # These descriptions aim to capture the essence of the labels.
        reference_texts = {
            "high_level_cognition": "Concept representing abstract thought, planning, complex problem solving, reasoning, and executive functions.",
            "basic_processing": "Concept representing simple sensory input processing, basic motor control signals, routine automated tasks, or low-level feature detection.",
            # Add more reference texts here for additional predefined labels if needed
        }

        # Generate embedding for each reference text
        for label, text in reference_texts.items():
            embedding = self._get_embedding(text)
            if embedding is not None:
                # Store the generated embedding associated with its label
                self.reference_embeddings[label] = embedding # Already on self.device
                log_statement(loglevel=str("debug"), logstatement=str(f"Generated reference embedding for label: {label}"), main_log=str(__name__))
            else:
                # Log error if embedding generation fails for a label
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to generate reference embedding for label: {label}. This label might not be assigned correctly."), main_log=str(__name__))

        # Log a warning if no reference embeddings could be generated
        if not self.reference_embeddings:
             log_statement(loglevel=str("warning"), logstatement=str("No reference embeddings were successfully generated. Labeling will likely default to 'unclassified'."), main_log=str(__name__))

    def generate_label(self, embedding: torch.Tensor) -> str:
        """
        Generates a semantic label for a single input embedding tensor.

        Compares the input embedding to the generated reference embeddings using
        cosine similarity and applies the configured threshold.

        Args:
            embedding (torch.Tensor): The input embedding tensor. It will be moved
                                      to the SemanticLabeler's device if necessary.

        Returns:
            str: The determined semantic label ("high_level_cognition", "basic_processing",
                 "unclassified", or an "error:..." string).
        """
        # Check if model initialization was successful
        if self.model is None:
             log_statement(loglevel=str("error"), logstatement=str("Model not loaded. Cannot generate label."), main_log=str(__name__))
             return "error: model_not_loaded" # Return error string as observed in tests

        # Validate input type
        if not isinstance(embedding, torch.Tensor):
            log_statement(loglevel=str("error"), logstatement=str(f"Invalid input type for embedding: {type(embedding)}. Expected torch.Tensor."), main_log=str(__name__))
            return "error: invalid_input_type"

        # Check if reference embeddings are available
        if not self.reference_embeddings:
            log_statement(loglevel=str("warning"), logstatement=str("No reference embeddings available for comparison. Returning 'unclassified'."), main_log=str(__name__))
            return "unclassified" # Return "unclassified" as per test logic

        # Ensure input embedding is on the correct device and has the right shape
        try:
             # Get expected embedding size from the loaded model's config
             expected_size = self.model.config.hidden_size
             if embedding.shape != (expected_size,):
                  log_statement(loglevel=str("error"), logstatement=str(f"Input embedding has incorrect shape {embedding.shape}. Expected ({expected_size},)."), main_log=str(__name__))
                  return "error: incorrect_embedding_shape"

             embedding = embedding.to(self.device) # Move to the same device as reference embeddings
        except AttributeError:
             log_statement(loglevel=str("error"), logstatement=str("Could not determine expected embedding size from model config."), main_log=str(__name__))
             return "error: unknown_model_embedding_size"
        except Exception as e:
             log_statement(loglevel=str("error"), logstatement=str(f"Failed to move input embedding to device {self.device} or validate shape: {e}"), main_log=str(__name__))
             return f"error: device_transfer_or_shape_error"

        best_label = "unclassified" # Default label
        max_similarity = -1.0 # Initialize below valid cosine similarity range

        # Iterate through stored reference labels and their embeddings
        for label, ref_embedding in self.reference_embeddings.items():
            try:
                # Calculate cosine similarity between input and reference embedding
                # Both tensors must be on the same device and require unsqueezing for batch dimension
                similarity = F.cosine_similarity(embedding.unsqueeze(0), ref_embedding.unsqueeze(0), dim=1)
                similarity_score = similarity.item() # Extract float value
                log_statement(loglevel=str("debug"), logstatement=str(f"Similarity score for label '{label}': {similarity_score:.4f}"), main_log=str(__name__))

                # Check if similarity exceeds threshold and is the best match so far
                if similarity_score > self.config.SIMILARITY_THRESHOLD and similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_label = label

            except Exception as e:
                # Log error during similarity calculation but continue checking other labels
                log_statement(loglevel=str("error"), logstatement=str(f"Error calculating similarity for label '{label}': {e}", exc_info=True), main_log=str(__name__))
                # Don't return an error string here, let 'unclassified' be the fallback if all fail

        # Log the final decision and return the best label found (or default)
        log_statement(loglevel=str("debug"), logstatement=str(f"Final determined label: {best_label} (Max Similarity: {max_similarity:.4f}, Threshold: {self.config.SIMILARITY_THRESHOLD})"), main_log=str(__name__))
        return best_label

    def recursive_labeling(self, embeddings: List[torch.Tensor], depth: int = 0) -> List[str]:
        """
        Applies generate_label to a list of embeddings, checking recursion depth.

        Args:
            embeddings (List[torch.Tensor]): A list of input embedding tensors.
            depth (int, optional): The current recursion depth. Defaults to 0.

        Returns:
            List[str]: A list of generated labels. If max depth is reached,
                       returns a list of "max_depth_exceeded" strings.
                       Returns ["error: invalid_input_type"] if input is not a list.
        """
        # Check recursion depth against the configured limit
        if depth >= self.config.MAX_RECURSION_DEPTH:
            log_statement(loglevel=str("warning"), logstatement=str(f"Max recursion depth ({self.config.MAX_RECURSION_DEPTH}) reached at depth {depth}."), main_log=str(__name__))
            # Return the specific string expected by the test
            return ["max_depth_exceeded"] * len(embeddings)

        # Validate input type
        if not isinstance(embeddings, list):
            log_statement(loglevel=str("error"), logstatement=str(f"Invalid input type for embeddings: {type(embeddings)}. Expected List[torch.Tensor]."), main_log=str(__name__))
            return ["error: invalid_input_type"] # Return list with error string

        labels = []
        # Process each embedding in the input list
        for i, embedding in enumerate(embeddings):
            # Call generate_label for the individual embedding
            label = self.generate_label(embedding)
            labels.append(label)
            # Note: No actual recursive call logic here based on current tests/usage.
            # If label indicated a need for deeper analysis, recursive calls could be added here.

        log_statement(loglevel=str("debug"), logstatement=str(f"Recursive labeling at depth {depth} finished processing {len(labels)} embeddings."), main_log=str(__name__))
        return labels # Return the list of generated labels

    # --- Optional: Validation Loop ---
    # def validate(self):
    #     """Runs a validation loop on a separate dataset."""
    #     self.model.eval() # Set model to evaluation mode
    #     total_val_loss = 0.0
    #     # ... loop through validation data_loader ...
    #     with torch.no_grad():
    #         # ... forward pass, calculate loss ...
    #     avg_val_loss = total_val_loss / len(self.validation_loader)
    #     log_statement(loglevel=str("info"), logstatement=str("f"Epoch {self.current_epoch} Validation Loss: {avg_val_loss:.4f}")
    #     self.model.train() # Set back to training mode
    #     return avg_val_loss


# Removed the original __main__ block and show_menu function.
# Instantiate and run the trainer from a separate script or notebook.
# Example:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO) # Setup basic logging
#
#     # --- Configuration ---
#     DEVICE = DEFAULT_DEVICE
#     INPUT_DIM = 128 # Example
#     NUM_CLASSES = 6   # Example
#
#     # --- Instantiate Components ---
#     # Adjust model and loader instantiation as needed
#     model = ZoneClassifier(input_features=INPUT_DIM, num_classes=NUM_CLASSES, device=DEVICE)
#     # Ensure data loader provides data in the format expected by the model and criterion
#     data_loader = EnhancedDataLoader(device=DEVICE) # Use tokenized data loader
#     # Choose appropriate loss function
#     criterion = nn.MSELoss() # Example: if output is regression-like
#     # criterion = nn.CrossEntropyLoss() # Example: if output is classification logits
#
#     # --- Initialize Trainer ---
#     trainer = EnhancedTrainer(model=model, data_loader=data_loader, criterion=criterion, device=DEVICE)
#
#     # --- Optional: Load Checkpoint ---
#     # trainer.load_checkpoint("ZoneClassifier_epoch_X_end.pt")
#
#     # --- Start Training ---
#     trainer.train()

