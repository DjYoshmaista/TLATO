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
# src/core/models.py
from pydantic import BaseModel, RootModel, Field, HttpUrl, field_validator as validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone # Ensure timezone is imported
from pathlib import Path
from src.utils.logger import log_statement
from src.data.constants import *
from src.utils.config import *
from src.utils.hashing import HashInfo

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
LOG_INS = f"{__file__}:{__name__}:"
class FileVersion(BaseModel):
    """Represents a specific version of a file in its application-level history."""
    version_number: int = Field(VER_NO, description="Sequential application-level version number.")
    timestamp_utc: datetime = Field(description="Timestamp of this version record (UTC).")
    git_commit_hash: Optional[str] = Field(None, description="Git commit hash associated with the commit that established this version.")
    change_description: Optional[str] = Field(None, description="Brief description of changes in this version or reason for new version.")
    size_bytes: Optional[int] = Field(None, description="Size of the file at this version in bytes.")
    # Storing custom_hashes as a dictionary for this version, similar to FileMetadataEntry
    custom_hashes: List[HashInfo] = Field(default_factory=list, description="List of custom hashes for the file content at this version.") # CHANGED HERE

    """Represents a specific version of a file in its application-level history."""
    # Ensure timestamps are correctly handled
    @validator('timestamp_utc', mode='before', check_fields=True)
    def ensure_timestamp_utc(cls, v):
        if isinstance(v, str):
            try:
                dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00'))
                if dt_obj.tzinfo is None:
                    return dt_obj.replace(tzinfo=timezone.utc)
                return dt_obj.astimezone(timezone.utc)
            except ValueError:
                raise ValueError(f"Invalid datetime string format for timestamp_utc: {v}")
        elif isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

class FileMetadataEntry(BaseModel):
    """
    Pydantic model for a single file's metadata entry, aligning with REFINED-PROPOSALS.
    """
    filepath_relative: str = Field(..., description="Path relative to the repository root (POSIX format). Acts as the primary key.")
    filename: Optional[str] = Field(None, description="Filename with extension.")
    extension: Optional[str] = Field(None, description="File extension without the dot (e.g., 'txt', 'csv').")
    
    size_bytes: Optional[int] = Field(None, description="Current size of the file in bytes.")
    os_last_modified_utc: Optional[datetime] = Field(None, description="OS last modification timestamp (UTC).")
    os_created_utc: Optional[datetime] = Field(None, description="OS creation timestamp (UTC) - platform dependent (ctime).")
    
    git_object_hash_current: Optional[str] = Field(None, description="Git blob hash of the current file content as known to the last metadata commit.")
    
    custom_hashes: Dict[str, str] = Field(default_factory=dict, description="Dictionary of custom content hashes (e.g., {'md5': 'value', 'sha256': 'value'}).")

    date_added_to_metadata_utc: datetime = Field(description="Timestamp when this entry was first added to metadata (UTC).")
    last_metadata_update_utc: datetime = Field(description="Timestamp of the last update to this metadata entry (UTC).")
    
    application_status: str = Field(DEFAULT_APPLICATION_STATUS, description="Current status of the file in the application/pipeline.")
    
    user_metadata: Dict[str, Any] = Field(default_factory=dict, description="User-defined arbitrary metadata.")
    
    version_current: int = Field(1, description="Current active application-level version number for this file.")
    version_history_app: List[FileVersion] = Field(default_factory=list, description="Application-level version history of this file.")

    # Optional fields from repo_handlerORIG.py or for future use
    source_uri: Optional[Union[HttpUrl, str]] = Field(None, description="Original source URI of the file, if applicable.")
    # processing_attempts: int = Field(0, description="Number of times processing has been attempted on this file.") # Example, add if needed
    # last_processing_date_utc: Optional[datetime] = Field(None, description="Timestamp of the last processing attempt (UTC).") # Example
    tags: List[str] = Field(default_factory=list, description="List of tags associated with the file.")
    external_system_link: Optional[Dict[str, str]] = Field(default_factory=dict, description="Links to this file/asset in external systems.")

    original_filename_if_compressed: Optional[str] = Field(None, description="Original filename if the stored file in the repository is a compressed version (e.g., 'data.csv' if 'data.csv.gz' is stored).")
    compression_type: Optional[str] = Field(None, description="Type of compression used on the file stored in the repository (e.g., 'gzip').")

    @validator('filepath_relative', mode='before', check_fields=True)
    def normalize_filepath_relative(cls, v):
        if isinstance(v, str):
            return Path(v).as_posix() # Ensure POSIX-style separators
        return v

    @validator('extension', mode='before', check_fields=True)
    def set_extension_from_filename_or_filepath(cls, v, values, **kwargs):
        if v: # If extension is explicitly provided
            return v.lower().lstrip('.')
        
        filename = values.get('filename')
        if filename:
            ext = Path(filename).suffix
            if ext:
                return ext.lower().lstrip('.')
        
        # If no filename, try to derive from filepath_relative
        filepath_relative = values.get('filepath_relative')
        if filepath_relative:
            ext = Path(filepath_relative).suffix
            if ext:
                return ext.lower().lstrip('.')
        return None

    @validator('filename', mode='before', check_fields=True)
    def set_filename_from_filepath(cls, v, values, **kwargs):
        if v: # If filename is explicitly provided
            return v
        filepath_relative = values.get('filepath_relative')
        if filepath_relative:
            return Path(filepath_relative).name
        return None

    # Universal validator for all datetime fields to ensure they are UTC and correctly parsed/set
    @validator('os_last_modified_utc', 'os_created_utc', 'date_added_to_metadata_utc', 
               'last_metadata_update_utc', # 'last_processing_date_utc' (if added back)
               mode='before', check_fields=True)
    def ensure_datetime_utc(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                # Handle ISO format strings, including those ending with 'Z'
                dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                # Try parsing if it's just a float timestamp (less ideal but might occur)
                try:
                    dt_obj = datetime.fromtimestamp(float(v), timezone.utc)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid datetime string or timestamp format: {v}")
            
            if dt_obj.tzinfo is None:
                return dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        elif isinstance(v, (int, float)): # Treat as UNIX timestamp
            return datetime.fromtimestamp(v, timezone.utc)
        elif isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise TypeError(f"Unsupported type for datetime field: {type(v)}")
        
class Config:
    validate_assignment = True
    # Ensure Pydantic can handle Path objects if they are ever passed directly (though we convert to str for filepath_relative)
    arbitrary_types_allowed = True # If Path objects were stored directly in models

class MetadataCollection(RootModel[Dict[str, FileMetadataEntry]]):
    """
    Represents the entire metadata collection, typically the content of metadata.json.
    The keys are relative file paths (strings), and values are FileMetadataEntry objects.
    """
    # Convenience methods to interact with the collection
    # Need to access via self.root now
    def get_entry(self, filepath_relative: str) -> Optional[FileMetadataEntry]:
        # Ensure filepath_relative is normalized for lookup, as keys are stored normalized
        normalized_key = Path(filepath_relative).as_posix()
        return self.root.get(normalized_key)

    def add_or_update_entry(self, entry: FileMetadataEntry):
        # filepath_relative in entry should already be normalized by its validator
        self.root[entry.filepath_relative] = entry

    def remove_entry(self, filepath_relative: str) -> Optional[FileMetadataEntry]:
        # Ensure filepath_relative is normalized for lookup
        normalized_key = Path(filepath_relative).as_posix()
        return self.root.pop(normalized_key, None)

    # Allow iteration over the model as if it were the root dict
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        # Ensure item is normalized if it's a path string
        normalized_item = Path(item).as_posix() if isinstance(item, str) else item
        return self.root[normalized_item]

    def __len__(self):
        return len(self.root)

    # If you need to serialize back to a plain dict for metadata_handler.write_metadata
    def to_dict(self) -> Dict[str, Any]:
        # Pydantic's .dict() is deprecated for RootModel's root directly.
        # We need to serialize each FileMetadataEntry.
        return {key: value.model_dump(by_alias=True, exclude_none=True) for key, value in self.root.items()}

# In your repo_handler.py, when you call write_metadata, you'd now use:
# self.metadata_handler.write_metadata(metadata_collection.to_dict())
# instead of metadata_collection.dict(by_alias=True)['__root__']
#
# When parsing from read_metadata:
# metadata_collection = MetadataCollection.model_validate(all_metadata_dict if all_metadata_dict else {})
# (Pydantic v2 uses model_validate or model_validate_json instead of model_validate/parse_raw)


# --- Placeholder for GATLayer (implied by tests.txt) ---
class GATLayer(nn.Module):
    """
    Graph Attention Layer (Placeholder).

    This class needs to be implemented based on the requirements hinted at
    in tests.txt. It likely wraps one or more GATConv layers from PyTorch Geometric.
    """
    def __init__(self, input_dim, output_dim, heads=8, dropout=0.1, **kwargs):
        super().__init__()
        global LOG_INS
        LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
        self.l_ins = LOG_INS

        if not PYG_AVAILABLE:
            raise ImportError("GATLayer requires PyTorch Geometric to be installed.")

        log_statement('info', f"Initializing GATLayer: In={input_dim}, Out={output_dim}, Heads={heads}", Path(__file__).stem)
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
             log_statement('error', "Cannot perform GATLayer forward pass: PyTorch Geometric not available.", Path(__file__).stem)
             # Return input features or raise error
             return data # Or return data.x, or raise RuntimeError

        x, edge_index = data.x, data.edge_index

        # Apply GAT convolution
        x = self.gat_conv(x, edge_index)

        # Update node features in the data object (common practice)
        # Alternatively, just return the tensor x
        data.x = F.elu(x) # Apply activation (ELU is common after GAT)

        log_statement('debug', f"GATLayer output shape: {data.x.shape}", Path(__file__).stem)
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
        global LOG_INS
        LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
        self.l_ins = LOG_INS
        self.input_features = input_features
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_statement('info', f"Initializing ZoneClassifier: In={input_features}, Classes={num_classes}, Device={self.device}", Path(__file__).stem)

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
             log_statement('warning', "ZoneClassifier falling back to MLP architecture due to missing PyTorch Geometric.", Path(__file__).stem)
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
        log_statement('info', f"ZoneClassifier moved to device: {self.device}", Path(__file__).stem)


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
            log_statement('debug', f"ZoneClassifier (GNN) input type: {type(input_data)}, x shape: {input_data.x.shape}", Path(__file__).stem)

            # Ensure data is on the correct device
            if input_data.x.device != self.device:
                 input_data = input_data.to(self.device)
                 log_statement('warning', f"Input data moved to device {self.device} during forward pass.", Path(__file__).stem)


            x = self.gat_layer1(input_data).x # Get updated node features
            # x = self.gat_layer2(data).x # If second GAT layer exists

            # Apply pooling (requires batch vector from PyG Batch object)
            # Example: x = self.pooling(x, input_data.batch)
            # If input is single graph (Data object), handle differently (e.g., mean over nodes)
            if hasattr(input_data, 'batch') and input_data.batch is not None:
                 # Import pooling if needed: from torch_geometric.nn import global_mean_pool
                 # x = global_mean_pool(x, input_data.batch)
                 # Placeholder: simple mean if pooling not imported/used
                 log_statement('warning', "Applying simple mean pooling as PyG pooling function is not explicitly used.", Path(__file__).stem)
                 x = x.mean(dim=0, keepdim=True) # Simple mean pooling - likely needs proper PyG pooling
            else:
                 # Handle single graph Data object - e.g., mean pool all nodes
                 x = x.mean(dim=0, keepdim=True) # Pool across nodes

            log_statement('debug', f"Shape after pooling: {x.shape}", Path(__file__).stem)

            # Apply final linear layers
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            log_statement('debug', f"ZoneClassifier (GNN) output shape: {x.shape}", Path(__file__).stem)
            return x

        elif isinstance(input_data, torch.Tensor):
             # --- Tensor-based forward pass (e.g., MLP fallback) ---
             log_statement('debug', f"ZoneClassifier (Tensor) input shape: {input_data.shape}", Path(__file__).stem)
             if input_data.device != self.device:
                  input_data = input_data.to(self.device)
                  log_statement('warning', f"Input tensor moved to device {self.device} during forward pass.", Path(__file__).stem)

             # Adapt based on expected tensor input and MLP structure
             # Example: if input is (batch, seq, feat) and MLP expects (batch, features)
             # x = self.flatten(input_data) # Requires defining self.flatten
             # x = self.relu(self.linear1(x))
             # Or if MLP operates directly on features (batch, features)
             x = self.relu(self.linear1(input_data))
             x = self.dropout(x)
             x = self.linear2(x)
             log_statement('debug', f"ZoneClassifier (Tensor) output shape: {x.shape}", Path(__file__).stem)
             return x
        else:
            log_statement('error', f"ZoneClassifier received unsupported input type: {type(input_data)}", Path(__file__).stem)
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
    global LOG_INS
    LOG_INS_CUST = LOG_INS + f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"

    log_statement('info', f"{LOG_INS_CUST}:INFO>>Attempting to load model checkpoint from: {checkpoint_path}", __file__)
    log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Args - model_class={model_class.__name__}, device='{device}', strict={strict_load}, eval_mode={eval_mode}, model_args={model_args}, model_kwargs={model_kwargs}", __file__)

    # --- Validate Inputs ---
    if not checkpoint_path.exists():
        log_statement('error', f"{LOG_INS_CUST}:ERROR>>Checkpoint file not found: {checkpoint_path}", __file__)
        return None
    if not checkpoint_path.is_file():
        log_statement('error', f"{LOG_INS_CUST}:ERROR>>Checkpoint path is not a file: {checkpoint_path}", __file__)
        return None
    if not device:
         log_statement('error', f"{LOG_INS_CUST}:ERROR>>Target device cannot be empty.", __file__)
         return None

    try:
        # --- Load Checkpoint Dictionary ---
        # map_location=device ensures tensors are loaded onto the correct device directly
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Loading checkpoint dictionary using torch.load with map_location='{device}'...", __file__)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Checkpoint dictionary loaded. Keys: {list(checkpoint.keys())}", __file__)

        # --- Validate Checkpoint Content ---
        if not isinstance(checkpoint, dict):
             log_statement('error', f"{LOG_INS_CUST}:ERROR>>Checkpoint file did not contain a dictionary: {checkpoint_path}", __file__)
             return None
        if 'model_state_dict' not in checkpoint:
            log_statement('error', f"{LOG_INS_CUST}:ERROR>>Checkpoint dictionary missing 'model_state_dict' key: {checkpoint_path}", __file__)
            return None
        if 'epoch' in checkpoint: # Log epoch if available
             log_statement('info', f"{LOG_INS_CUST}:INFO>>Checkpoint is from epoch: {checkpoint.get('epoch')}", __file__)
        if 'extra_meta' in checkpoint: # Log extra metadata if available
             log_statement('info', f"{LOG_INS_CUST}:INFO>>Checkpoint contains extra metadata: {checkpoint.get('extra_meta')}", __file__)


        # --- Instantiate Model Architecture ---
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Instantiating model architecture: {model_class.__name__}(*{model_args}, **{model_kwargs})", __file__)
        model = model_class(*model_args, **model_kwargs)
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Model instance created.", __file__)


        # --- Load State Dictionary ---
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Loading state_dict into model instance (strict={strict_load})...", __file__)
        try:
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=strict_load)
            # Log results of non-strict loading if applicable
            if not strict_load:
                 if load_result.missing_keys:
                      log_statement('warning', f"{LOG_INS_CUST}:WARNING>>State dict loaded with missing keys (strict=False): {load_result.missing_keys}", __file__)
                 if load_result.unexpected_keys:
                      log_statement('warning', f"{LOG_INS_CUST}:WARNING>>State dict loaded with unexpected keys (strict=False): {load_result.unexpected_keys}", __file__)
            log_statement('info', f"{LOG_INS_CUST}:INFO>>Model state_dict loaded successfully.", __file__)
        except RuntimeError as state_dict_error:
             # Common errors: size mismatches, key mismatches (if strict=True)
             log_statement('error', f"{LOG_INS_CUST}:ERROR>>RuntimeError loading state_dict: {state_dict_error}. Ensure model architecture matches checkpoint.", __file__, True)
             return None
        except KeyError as key_error:
            # If 'model_state_dict' key itself was missing (should be caught earlier, but good practice)
            log_statement('error', f"{LOG_INS_CUST}:ERROR>>KeyError loading state_dict (likely missing 'model_state_dict'): {key_error}", __file__)
            return None


        # --- Move Model to Device (should be mostly redundant if map_location worked, but ensures consistency) ---
        log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Moving loaded model to device: '{device}'...", __file__)
        model.to(device)


        # --- Set Evaluation Mode ---
        if eval_mode:
            log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Setting model to evaluation mode (model.eval()).", __file__)
            model.eval()
        else:
            log_statement('debug', f"{LOG_INS_CUST}:DEBUG>>Leaving model in training mode (eval_mode=False).", __file__)


        # --- Return Loaded Model ---
        log_statement('info', f"{LOG_INS_CUST}:INFO>>Model loaded successfully from {checkpoint_path} onto device '{device}'.", __file__)
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
        log_statement('error', f"{LOG_INS_CUST}:ERROR>>Checkpoint file not found during load: {checkpoint_path}", __file__)
        return None
    except (pickle.UnpicklingError, EOFError, zipfile.BadZipFile) as load_err: # Catch common torch.load errors
         log_statement('error', f"{LOG_INS_CUST}:ERROR>>Failed to load/unpickle checkpoint file {checkpoint_path}. File may be corrupted or incompatible: {load_err}", __file__, True)
         return None
    except RuntimeError as rte: # Catch potential device loading issues
         log_statement('error', f"{LOG_INS_CUST}:ERROR>>RuntimeError during model loading (potentially device related): {rte}", __file__, True)
         return None
    except Exception as e:
        # Catch unexpected errors during the process
        log_statement('critical', f"{LOG_INS_CUST}:CRITICAL>>CRITICAL: Unexpected error loading model from checkpoint {checkpoint_path}: {e}", __file__, True)
        return None

if __name__ == "__main__":
    print("--- FileVersion Example ---")
    now_dt = datetime.now(timezone.utc)
    version_entry = FileVersion(
        version_number=1, 
        timestamp_utc=now_dt, # Pass datetime object
        change_description="Initial check-in.",
        size_bytes=100,
        custom_hashes={"md5": "hashval1"}
    )
    print(version_entry.model_dump_json(indent=2))

    print("\n--- FileMetadataEntry Example ---")
    file_meta_data_dict = {
        "filepath_relative": "data/raw/my_document.txt",
        # filename and extension will be derived
        "size_bytes": 1024,
        "os_last_modified_utc": datetime.now(timezone.utc).isoformat(), # Pass ISO string
        "custom_hashes": {"md5": "d41d8cd98f00b204e9800998ecf8427e", "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        "date_added_to_metadata_utc": now_dt.isoformat(), # Pass ISO string
        "last_metadata_update_utc": now_dt, # Pass datetime object
        "application_status": "new",
        "user_metadata": {"project": "alpha", "source_type": "upload"},
        "version_current": 2,
        "version_history_app": [version_entry.dict()], # Pass dict from FileVersion Pydantic model
        "tags": ["document", "important"]
    }
    file_entry = FileMetadataEntry(**file_meta_data_dict)
    print(f"Derived filename: {file_entry.filename}, Derived extension: {file_entry.extension}")
    print(file_entry.model_dump_json(indent=2, exclude_none=True))
    
    # Test derivation and default factory for timestamps
    file_meta_derive = FileMetadataEntry(filepath_relative="another/file.pdf", date_added_to_metadata_utc=datetime.now(timezone.utc), last_metadata_update_utc=datetime.now(timezone.utc))
    print("\nDerived filename/extension for another/file.pdf:")
    print(f"Filename: {file_meta_derive.filename}, Extension: {file_meta_derive.extension}")
    print(f"Date added: {file_meta_derive.date_added_to_metadata_utc}")


    print("\n--- MetadataCollection Example ---")
    # Create dicts from Pydantic models for the collection
    collection_data = {
        file_entry.filepath_relative: file_entry.dict(by_alias=True),
        file_meta_derive.filepath_relative: file_meta_derive.dict(by_alias=True)
    }
    # Initialize MetadataCollection with a dictionary of raw data that can be parsed into FileMetadataEntry
    metadata_collection = MetadataCollection.model_validate(collection_data)
    print(f"Collection has {len(metadata_collection.__root__)} entries.")
    retrieved_entry = metadata_collection.get_entry("data/raw/my_document.txt")
    if retrieved_entry:
        print(f"Retrieved status for my_document.txt: {retrieved_entry.application_status}")

    # Example of adding a new entry (assuming it's already a Pydantic model instance)
    new_file_data = FileMetadataEntry(
        filepath_relative="new/data/sample.json",
        size_bytes=2048,
        os_last_modified_utc=datetime.now(timezone.utc),
        custom_hashes={"md5": "someotherhash"},
        date_added_to_metadata_utc=datetime.now(timezone.utc),
        last_metadata_update_utc=datetime.now(timezone.utc)
    )
    metadata_collection.add_or_update_entry(new_file_data)
    print(f"Collection size after adding new entry: {len(metadata_collection.__root__)}")
    print(metadata_collection.model_dump_json(indent=2, exclude_none=True)) # Can be verbose