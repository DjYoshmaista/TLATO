# src/core/models.py
"""
Neural Network Model Definitions

Contains PyTorch model classes used in the project, such as the main
ZoneClassifier or any Graph Neural Network layers (like GATLayer implied
by tests.txt).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Type, Union, Optional, List, Any
import inspect
from pathlib import Path
import pickle
import zipfile
from src.utils.logger import log_statement
from src.data.constants import *

# Import necessary components from PyTorch Geometric if used
try:
    # These imports are based on the usage seen in the original tests.txt
    from torch_geometric.nn import GATConv # Example: GATLayer might use GATConv
    from torch_geometric.data import Data, Batch # For handling graph data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning("PyTorch Geometric not found. Graph Neural Network components (e.g., GATLayer) will not be available.")
    # Define dummy classes if needed to avoid import errors elsewhere, or handle conditionally
    class GATConv: pass
    class Data: pass
    class Batch: pass

# --- Placeholder for GATLayer (implied by tests.txt) ---
class GATLayer(nn.Module):
    """
    Graph Attention Layer (Placeholder).

    This class needs to be implemented based on the requirements hinted at
    in tests.txt. It likely wraps one or more GATConv layers from PyTorch Geometric.
    """
    def __init__(self, input_dim, output_dim, heads=8, dropout=0.1, **kwargs):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("GATLayer requires PyTorch Geometric to be installed.")

        log_statement(loglevel=str("info"), logstatement=str(f"Initializing GATLayer: In={input_dim}, Out={output_dim}, Heads={heads}"), main_logger=str(__name__))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads

        # Example implementation using GATConv
        # Adjust parameters (e.g., concat, negative_slope) as needed
        self.gat_conv = GATConv(input_dim, output_dim // heads, heads=heads, dropout=dropout, **kwargs)
        # If output_dim is not divisible by heads, adjust logic or raise error

    def forward(self, data: Data or Batch):
        """
        Forward pass for the GAT layer.

        Args:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                  Graph data object containing node features (x) and edge indices (edge_index).

        Returns:
            torch_geometric.data.Data or torch_geometric.data.Batch:
                  The input data object with updated node features (x).
                  (Or potentially just the updated node features tensor).
        """
        if not PYG_AVAILABLE:
             log_statement(loglevel=str("error"), logstatement=str("Cannot perform GATLayer forward pass: PyTorch Geometric not available."), main_logger=str(__name__))
             # Return input features or raise error
             return data # Or return data.x, or raise RuntimeError

        x, edge_index = data.x, data.edge_index

        # Apply GAT convolution
        x = self.gat_conv(x, edge_index)

        # Update node features in the data object (common practice)
        # Alternatively, just return the tensor x
        data.x = F.elu(x) # Apply activation (ELU is common after GAT)

        log_statement(loglevel=str("debug"), logstatement=str(f"GATLayer output shape: {data.x.shape}"), main_logger=str(__name__))
        return data # Or return data.x


# --- Placeholder for ZoneClassifier (implied by training.txt & tests.txt) ---
class ZoneClassifier(nn.Module):
    """
    Main classification model (Placeholder).

    This model architecture needs to be defined based on the project's goals.
    It might integrate GAT layers, linear layers, etc. The input/output
    dimensions (128 -> 6 seen in training.txt main) should be configurable.
    """
    def __init__(self, input_features=128, num_classes=6, device=None, **kwargs):
        """
        Initializes the ZoneClassifier model.

        Args:
            input_features (int): Dimension of input features per node/element.
            num_classes (int): Number of output classes.
            device (torch.device or str, optional): Device to initialize on.
            **kwargs: Additional arguments for internal layers (e.g., hidden_dim).
        """
        super().__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_statement(loglevel=str("info"), logstatement=str(f"Initializing ZoneClassifier: In={input_features}, Classes={num_classes}, Device={self.device}"), main_logger=str(__name__))

        # --- Example Architecture ---
        # This is a *very* basic example. Replace with your actual model design.
        # It might involve GAT layers, LSTMs, Transformers, depending on input data structure.

        # Example: Using a GAT layer followed by linear layers
        hidden_dim = kwargs.get('hidden_dim', 256) # Example hidden dimension

        if PYG_AVAILABLE:
             # Assume input is graph data if GAT is used
             self.gat_layer1 = GATLayer(input_features, hidden_dim)
             # Add more layers if needed (e.g., another GAT or pooling)
             # self.gat_layer2 = GATLayer(hidden_dim, hidden_dim)

             # Global pooling layer (example: mean pooling)
             # Needs PyG pooling functions if used: from torch_geometric.nn import global_mean_pool
             # self.pooling = global_mean_pool

             # Final linear layers for classification based on pooled graph representation
             self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
             self.dropout = nn.Dropout(p=0.5)
             self.linear2 = nn.Linear(hidden_dim // 2, num_classes)
        else:
             # Fallback or alternative architecture if PyG is not used
             # Example: Simple MLP assuming input is a flat tensor per batch item
             # This might not match the dummy_input shape in helpers.py - adjust as needed!
             log_statement(loglevel=str("warning"), logstatement=str("ZoneClassifier falling back to MLP architecture due to missing PyTorch Geometric."), main_logger=str(__name__))
             # Assuming input needs flattening or different handling
             # Example: Flatten sequence data (batch, seq, feat) -> (batch, seq*feat)
             # self.flatten = nn.Flatten()
             # self.linear1 = nn.Linear(input_features * seq_len, hidden_dim) # Requires seq_len
             # Or maybe just operate on the features directly if input is (batch, features)
             self.linear1 = nn.Linear(input_features, hidden_dim)
             self.relu = nn.ReLU()
             self.dropout = nn.Dropout(p=0.5)
             self.linear2 = nn.Linear(hidden_dim, num_classes)


        # Move model to the specified device
        self.to(self.device)
        log_statement(loglevel=str("info"), logstatement=str(f"ZoneClassifier moved to device: {self.device}"), main_logger=str(__name__))


    def forward(self, input_data):
        """
        Forward pass of the ZoneClassifier.

        Args:
            input_data: The input data. Shape and type depend on the architecture.
                        Could be a tensor, or a PyG Data/Batch object.

        Returns:
            torch.Tensor: The output logits or probabilities for each class.
        """
        # --- Adapt forward pass based on the chosen architecture ---

        if PYG_AVAILABLE and isinstance(input_data, (Data, Batch)):
            # --- Graph-based forward pass ---
            log_statement(loglevel=str("debug"), logstatement=str(f"ZoneClassifier (GNN) input type: {type(input_data)}, x shape: {input_data.x.shape}"), main_logger=str(__name__))

            # Ensure data is on the correct device
            if input_data.x.device != self.device:
                 input_data = input_data.to(self.device)
                 log_statement(loglevel=str("warning"), logstatement=str(f"Input data moved to device {self.device} during forward pass."), main_logger=str(__name__))


            x = self.gat_layer1(input_data).x # Get updated node features
            # x = self.gat_layer2(data).x # If second GAT layer exists

            # Apply pooling (requires batch vector from PyG Batch object)
            # Example: x = self.pooling(x, input_data.batch)
            # If input is single graph (Data object), handle differently (e.g., mean over nodes)
            if hasattr(input_data, 'batch') and input_data.batch is not None:
                 # Import pooling if needed: from torch_geometric.nn import global_mean_pool
                 # x = global_mean_pool(x, input_data.batch)
                 # Placeholder: simple mean if pooling not imported/used
                 log_statement(loglevel=str("warning"), logstatement=str("Applying simple mean pooling as PyG pooling function is not explicitly used."), main_logger=str(__name__))
                 x = x.mean(dim=0, keepdim=True) # Simple mean pooling - likely needs proper PyG pooling
            else:
                 # Handle single graph Data object - e.g., mean pool all nodes
                 x = x.mean(dim=0, keepdim=True) # Pool across nodes

            log_statement(loglevel=str("debug"), logstatement=str(f"Shape after pooling: {x.shape}"), main_logger=str(__name__))

            # Apply final linear layers
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            log_statement(loglevel=str("debug"), logstatement=str(f"ZoneClassifier (GNN) output shape: {x.shape}"), main_logger=str(__name__))
            return x

        elif isinstance(input_data, torch.Tensor):
             # --- Tensor-based forward pass (e.g., MLP fallback) ---
             log_statement(loglevel=str("debug"), logstatement=str(f"ZoneClassifier (Tensor) input shape: {input_data.shape}"), main_logger=str(__name__))
             if input_data.device != self.device:
                  input_data = input_data.to(self.device)
                  log_statement(loglevel=str("warning"), logstatement=str(f"Input tensor moved to device {self.device} during forward pass."), main_logger=str(__name__))

             # Adapt based on expected tensor input and MLP structure
             # Example: if input is (batch, seq, feat) and MLP expects (batch, features)
             # x = self.flatten(input_data) # Requires defining self.flatten
             # x = self.relu(self.linear1(x))
             # Or if MLP operates directly on features (batch, features)
             x = self.relu(self.linear1(input_data))
             x = self.dropout(x)
             x = self.linear2(x)
             log_statement(loglevel=str("debug"), logstatement=str(f"ZoneClassifier (Tensor) output shape: {x.shape}"), main_logger=str(__name__))
             return x
        else:
            log_statement(loglevel=str("error"), logstatement=str(f"ZoneClassifier received unsupported input type: {type(input_data)}"), main_logger=str(__name__))
            raise TypeError(f"Unsupported input type for ZoneClassifier: {type(input_data)}")

def load_model_from_checkpoint(
    model_class: Type[nn.Module], # The class of the model to instantiate (e.g., ZoneClassifier)
    checkpoint_path: Path,
    device: str, # The target device ('cpu', 'cuda:0', etc.)
    strict_load: bool = True, # Whether state_dict keys must exactly match
    eval_mode: bool = True, # Set model.eval() after loading?
    *model_args: Any, # Positional arguments for model_class constructor
    **model_kwargs: Any # Keyword arguments for model_class constructor
) -> Optional[nn.Module]:
    """
    Loads model weights from a saved checkpoint file onto a specified device.

    Instantiates the model architecture using the provided class and arguments,
    loads the state dictionary from the checkpoint, moves the model to the
    target device, and optionally sets it to evaluation mode.

    Args:
        model_class (Type[nn.Module]): The Python class of the model architecture to load.
        checkpoint_path (Path): Path to the checkpoint file (.pt, .pth, etc.).
        device (str): The device to load the model onto (e.g., 'cuda:0', 'cpu').
                      Determined externally, e.g., using set_compute_device.
        strict_load (bool): If True (default), requires the keys in the checkpoint's
                            state_dict to exactly match the keys returned by the model's
                            state_dict(). If False, allows loading partial or mismatched keys.
        eval_mode (bool): If True (default), sets the model to evaluation mode
                          (model.eval()) after loading. Set to False if loading
                          for continued training.
        *model_args: Positional arguments needed to initialize model_class.
        **model_kwargs: Keyword arguments needed to initialize model_class.

    Returns:
        Optional[nn.Module]: The loaded model instance on the specified device,
                             or None if loading fails.
    """
    frame = inspect.currentframe()
    LOG_INS_CUST = f"{LOG_INS}::load_model_from_checkpoint::{frame.f_lineno if frame else 'UnknownLine'}"

    log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Attempting to load model checkpoint from: {checkpoint_path}", main_logger=__file__)
    log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Args - model_class={model_class.__name__}, device='{device}', strict={strict_load}, eval_mode={eval_mode}, model_args={model_args}, model_kwargs={model_kwargs}", main_logger=__file__)

    # --- Validate Inputs ---
    if not checkpoint_path.exists():
        log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Checkpoint file not found: {checkpoint_path}", main_logger=__file__)
        return None
    if not checkpoint_path.is_file():
        log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Checkpoint path is not a file: {checkpoint_path}", main_logger=__file__)
        return None
    if not device:
         log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Target device cannot be empty.", main_logger=__file__)
         return None

    try:
        # --- Load Checkpoint Dictionary ---
        # map_location=device ensures tensors are loaded onto the correct device directly
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Loading checkpoint dictionary using torch.load with map_location='{device}'...", main_logger=__file__)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Checkpoint dictionary loaded. Keys: {list(checkpoint.keys())}", main_logger=__file__)

        # --- Validate Checkpoint Content ---
        if not isinstance(checkpoint, dict):
             log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Checkpoint file did not contain a dictionary: {checkpoint_path}", main_logger=__file__)
             return None
        if 'model_state_dict' not in checkpoint:
            log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Checkpoint dictionary missing 'model_state_dict' key: {checkpoint_path}", main_logger=__file__)
            return None
        if 'epoch' in checkpoint: # Log epoch if available
             log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Checkpoint is from epoch: {checkpoint.get('epoch')}", main_logger=__file__)
        if 'extra_meta' in checkpoint: # Log extra metadata if available
             log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Checkpoint contains extra metadata: {checkpoint.get('extra_meta')}", main_logger=__file__)


        # --- Instantiate Model Architecture ---
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Instantiating model architecture: {model_class.__name__}(*{model_args}, **{model_kwargs})", main_logger=__file__)
        model = model_class(*model_args, **model_kwargs)
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Model instance created.", main_logger=__file__)


        # --- Load State Dictionary ---
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Loading state_dict into model instance (strict={strict_load})...", main_logger=__file__)
        try:
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=strict_load)
            # Log results of non-strict loading if applicable
            if not strict_load:
                 if load_result.missing_keys:
                      log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>State dict loaded with missing keys (strict=False): {load_result.missing_keys}", main_logger=__file__)
                 if load_result.unexpected_keys:
                      log_statement(loglevel='warning', logstatement=f"{LOG_INS_CUST}:WARNING>>State dict loaded with unexpected keys (strict=False): {load_result.unexpected_keys}", main_logger=__file__)
            log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Model state_dict loaded successfully.", main_logger=__file__)
        except RuntimeError as state_dict_error:
             # Common errors: size mismatches, key mismatches (if strict=True)
             log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>RuntimeError loading state_dict: {state_dict_error}. Ensure model architecture matches checkpoint.", main_logger=__file__, exc_info=True)
             return None
        except KeyError as key_error:
            # If 'model_state_dict' key itself was missing (should be caught earlier, but good practice)
            log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>KeyError loading state_dict (likely missing 'model_state_dict'): {key_error}", main_logger=__file__)
            return None


        # --- Move Model to Device (should be mostly redundant if map_location worked, but ensures consistency) ---
        log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Moving loaded model to device: '{device}'...", main_logger=__file__)
        model.to(device)


        # --- Set Evaluation Mode ---
        if eval_mode:
            log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Setting model to evaluation mode (model.eval()).", main_logger=__file__)
            model.eval()
        else:
            log_statement(loglevel='debug', logstatement=f"{LOG_INS_CUST}:DEBUG>>Leaving model in training mode (eval_mode=False).", main_logger=__file__)


        # --- Return Loaded Model ---
        log_statement(loglevel='info', logstatement=f"{LOG_INS_CUST}:INFO>>Model loaded successfully from {checkpoint_path} onto device '{device}'.", main_logger=__file__)
        # Optionally return other checkpoint info if needed, but standard practice is often just the model
        # return {
        #      'model': model,
        #      'epoch': checkpoint.get('epoch'),
        #      'optimizer_state_dict': checkpoint.get('optimizer_state_dict'), # May need deepcopy?
        #      'extra_meta': checkpoint.get('extra_meta')
        # }
        return model

    except FileNotFoundError:
        # Should be caught by initial check, but handle again just in case
        log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Checkpoint file not found during load: {checkpoint_path}", main_logger=__file__)
        return None
    except (pickle.UnpicklingError, EOFError, zipfile.BadZipFile) as load_err: # Catch common torch.load errors
         log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>Failed to load/unpickle checkpoint file {checkpoint_path}. File may be corrupted or incompatible: {load_err}", main_logger=__file__, exc_info=True)
         return None
    except RuntimeError as rte: # Catch potential device loading issues
         log_statement(loglevel='error', logstatement=f"{LOG_INS_CUST}:ERROR>>RuntimeError during model loading (potentially device related): {rte}", main_logger=__file__, exc_info=True)
         return None
    except Exception as e:
        # Catch unexpected errors during the process
        log_statement(loglevel='critical', logstatement=f"{LOG_INS_CUST}:CRITICAL>>CRITICAL: Unexpected error loading model from checkpoint {checkpoint_path}: {e}", main_logger=__file__, exc_info=True)
        return None
