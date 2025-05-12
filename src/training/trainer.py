# src/training/trainer.py
"""
Model Training Module

Contains the core logic for training neural network models, including
optimization, learning rate scheduling, optional pruning, checkpointing,
metrics tracking, and configurable hyperparameters.
"""

from typing import Optional
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
import json # For logging config
from typing import Union, Dict, Any, Tuple

# Import project configuration and utilities
try:
    from src.data.constants import *
    from src.utils.config import TrainingConfig, DEFAULT_DEVICE, CHECKPOINT_DIR, LOG_DIR
    from src.utils.logger import configure_logging, log_statement
    from src.utils.helpers import save_state, load_state # Use helper functions
    # Import required model and data loader classes (adjust as needed)
    from src.core.models import ZoneClassifier # Example model
    from src.data.loaders import EnhancedDataLoader # Example loader
    configure_logging(log_file_path=LOG_DIR / 'training.log', error_log_file_path=LOG_DIR / 'errors.log', log_level=logging.INFO, console_logging=True, file_logging=True)
    log_statement(loglevel=str("info"), logstatement=str("Training module initialized with logging."), main_logger=str(__name__))
    log_statement(loglevel=str("info"), logstatement=str(f"Using device: {DEFAULT_DEVICE}"), main_logger=str(__name__))
    log_statement(loglevel=str("info"), logstatement=str(f"Checkpoint directory: {CHECKPOINT_DIR}"), main_logger=str(__name__))
    log_statement(loglevel=str("info"), logstatement=str(f"Log directory: {LOG_DIR}"), main_logger=str(__name__))
    log_statement(loglevel=str("info"), logstatement=str(f"Training configuration: {TrainingConfig.__dict__}"), main_logger=str(__name__))
except ImportError:
     # Basic fallback for core components needed by trainer
     logging.error("Failed to import core utils/models/loaders for Trainer.")
     class TrainingConfig: 
        MAX_EPOCHS=1
        INITIAL_LR=1e-3
        WEIGHT_DECAY=0
        PRUNE_INTERVAL_EPOCHS=0
        PRUNE_AMOUNT=0
        CHECKPOINT_DIR=Path('./checkpoints')
        CHECKPOINT_INTERVAL_BATCH_PERCENT=0
        METRICS_FILENAME_PREFIX="metrics"
        MODEL_INPUT_DIM=10
        MODEL_NUM_CLASSES=2
        LOSS_FUNCTION='MSELoss'
     DEFAULT_DEVICE='cpu'
     CHECKPOINT_DIR=Path('./checkpoints')
     LOG_DIR=Path('./logs')
     def save_state(*args, **kwargs): pass
     def load_state(*args, **kwargs): return None
     class ZoneClassifier(nn.Module): 
        def __init__(self, *args, **kwargs): 
            super().__init__()
            self.linear=nn.Linear(10,2)
        def forward(self, x):
            return self.linear(x) # Dummy model
     class EnhancedDataLoader:
        def __init__(self,*a,**k): pass
        def __iter__(self): yield torch.randn(4,10),torch.randn(4,2)
        def __len__(self): return 1 # Dummy loader

# --- Training Metrics Class ---
# class TrainingMetrics:
#     def __init__(self, save_dir: Optional[Path] = None):
#         self.metrics_data = []; self.columns = ['timestamp','epoch','batch','loss','lr','pruned_weights_count','batch_duration_sec','batch_size']
#         self.save_dir = Path(save_dir or LOG_DIR); self.save_dir.mkdir(parents=True, exist_ok=True)
#         log_statement(loglevel=str("info"), logstatement=str(f"TrainingMetrics OK. Save dir: {self.save_dir}"), main_logger=str(__name__))
#     def record(self, epoch, batch, loss, lr, pruned_count, duration, batch_size):
#         loss_item=loss.item() if isinstance(loss,torch.Tensor) else float(loss); lr_item=lr[0] if isinstance(lr,list) else float(lr)
#         self.metrics_data.append({'timestamp':datetime.now(),'epoch':int(epoch),'batch':int(batch),'loss':loss_item,'lr':lr_item,'pruned_weights_count':int(pruned_count),'batch_duration_sec':float(duration),'batch_size':int(batch_size)})
#     def get_dataframe(self) -> pd.DataFrame: return pd.DataFrame(self.metrics_data, columns=self.columns) if self.metrics_data else pd.DataFrame(columns=self.columns)
#     def save(self, filename: Optional[str] = None):
#         if not self.metrics_data: log_statement(loglevel=str("warning"), logstatement=str("No metrics to save."), main_logger=str(__name__)); return
#         df=self.get_dataframe(); df['timestamp']=pd.to_datetime(df['timestamp']); df['epoch']=df['epoch'].astype(int); df['batch']=df['batch'].astype(int); df['loss']=df['loss'].astype(float); df['lr']=df['lr'].astype(float); df['pruned_weights_count']=df['pruned_weights_count'].astype(int); df['batch_duration_sec']=df['batch_duration_sec'].astype(float); df['batch_size']=df['batch_size'].astype(int)
#         filename=filename or f"{TrainingConfig.METRICS_FILENAME_PREFIX}_{datetime.now():%Y%m%d%H%M%S}.csv"; filepath=self.save_dir/filename
#         try: df.to_csv(filepath, index=False, encoding='utf-8'); log_statement(loglevel=str("info"), logstatement=str(f"Saved metrics ({len(df)} rows) to {filepath}"), main_logger=str(__name__))
#         except Exception as e: log_statement(loglevel=str("error"), logstatement=str(f"Failed metrics save to {filepath}: {e}", exc_info=True)
#     def clear(self): self.metrics_data=[]; log_statement(loglevel=str("info"), logstatement=str("Cleared training metrics."), main_logger=str(__name__))



# --- Training Configuration ---

class TrainingMetrics:
    """
    Handles recording and saving of training performance metrics.
    """
    def __init__(self, save_dir: Path = None):
        """
        Initializes the TrainingMetrics handler.

        Args:
            save_dir (Path, optional): Directory to save metrics files. Defaults to LOG_DIR.
        """
        self.metrics_data = [] # Store metrics as a list of dicts for efficiency
        self.columns = [
            'timestamp', 'epoch', 'batch', 'loss', 'lr',
            'pruned_weights_count', 'batch_duration_sec', 'batch_size'
        ]
        self.save_dir = Path(save_dir or LOG_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log_statement(loglevel=str("info"), logstatement=str(f"TrainingMetrics initialized. Metrics will be saved to: {self.save_dir}"), main_logger=str(__name__))

    def record(self, epoch, batch, loss, lr, pruned_count, duration, batch_size):
        """
        Records metrics for a single training batch.

        Args:
            epoch (int): Current epoch number.
            batch (int): Current batch index within the epoch.
            loss (float): Loss value for the batch.
            lr (float): Current learning rate.
            pruned_count (int): Total number of pruned weights in the model.
            duration (float): Duration of the batch processing in seconds.
            batch_size (int): Number of samples in the batch.
        """
        # Ensure loss and lr are floats
        loss_item = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        lr_item = lr[0] if isinstance(lr, list) else float(lr) # Handle potential list from optimizer

        new_row = {
            'timestamp': datetime.now(),
            'epoch': int(epoch),
            'batch': int(batch),
            'loss': loss_item,
            'lr': lr_item,
            'pruned_weights_count': int(pruned_count),
            'batch_duration_sec': float(duration),
            'batch_size': int(batch_size)
        }
        self.metrics_data.append(new_row)

    def get_dataframe(self) -> pd.DataFrame:
        """Converts recorded metrics to a pandas DataFrame."""
        if not self.metrics_data:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.metrics_data, columns=self.columns)

    def save(self, filename: str = None):
        """
        Saves the recorded metrics to a CSV file in the configured directory.

        Args:
            filename (str, optional): The base name for the metrics file.
                                      Defaults to 'training_metrics_{timestamp}.csv'.
        """
        if not self.metrics_data:
            log_statement(loglevel=str("warning"), logstatement=str("No metrics data recorded, skipping save."), main_logger=str(__name__))
            return

        df = self.get_dataframe()
        # Ensure correct dtypes before saving
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['epoch'] = df['epoch'].astype(int)
        df['batch'] = df['batch'].astype(int)
        df['loss'] = df['loss'].astype(float)
        df['lr'] = df['lr'].astype(float)
        df['pruned_weights_count'] = df['pruned_weights_count'].astype(int)
        df['batch_duration_sec'] = df['batch_duration_sec'].astype(float)
        df['batch_size'] = df['batch_size'].astype(int)


        filename = filename or f"{TrainingConfig.METRICS_FILENAME_PREFIX}_{datetime.now():%Y%m%d%H%M%S}.csv"
        filepath = self.save_dir / filename
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            log_statement(loglevel=str("info"), logstatement=str(f"Saved training metrics ({len(df)} rows) to {filepath}"), main_logger=str(__name__))
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to save training metrics to {filepath}: {e}", exc_info=True), main_logger=str(__name__))

    def clear(self):
        """Clears the recorded metrics."""
        self.metrics_data = []
        log_statement(loglevel=str("info"), logstatement=str("Cleared recorded training metrics."), main_logger=str(__name__))

class EnhancedTrainer:
    """
    Provides a comprehensive and configurable training loop for PyTorch models,
    incorporating optimization, learning rate scheduling, optional weight pruning,
    robust checkpointing (intra-epoch and end-of-epoch), configuration overrides,
    detailed logging, and metrics tracking.

    Merges functionalities from two previous versions.
    """
    def __init__(self,
                 model: nn.Module,
                 data_loader, # Expects an iterable (e.g., DataLoader) -> (inputs, targets)
                 criterion, # Expects torch loss function instance (e.g., nn.MSELoss())
                 device: Optional[Union[str, torch.device]] = None,
                 training_config_override: Optional[Dict[str, Any]] = None): # Accept overrides
        """
        Initializes the EnhancedTrainer.

        Args:
            model (nn.Module): The neural network model to train.
            data_loader: Iterable data loader providing (inputs, targets) batches.
            criterion: Loss function instance (e.g., nn.MSELoss()).
            device (str | torch.device, optional): Device ('cuda', 'cpu').
                                                   Defaults to project's DEFAULT_DEVICE.
            training_config_override (dict, optional): Dictionary to override TrainingConfig
                                                       defaults (e.g., {'MAX_EPOCHS': 10}).
        """
        log_statement('info', f'{LOG_INS}::Initializing EnhancedTrainer...', __name__)

        # --- Configuration Setup ---
        log_statement('debug', f'{LOG_INS}::Creating runtime training configuration.', __name__)
        self.config = self._create_runtime_config(training_config_override)
        log_statement('debug', f'{LOG_INS}::Runtime configuration created.', __name__)

        # --- Device Setup ---
        self.device = device or DEFAULT_DEVICE
        log_statement('info', f'{LOG_INS}::Using device: {self.device}', __name__)

        # --- Model Setup ---
        self.model = model.to(self.device)
        log_statement('info', f'{LOG_INS}::Model {type(model).__name__} moved to device {self.device}.', __name__)

        # --- Data Loader ---
        self.data_loader = data_loader
        log_statement('debug', f'{LOG_INS}::Data loader assigned: {type(data_loader)}', __name__)

        # --- Criterion (Loss Function) ---
        self.criterion = criterion.to(self.device)
        log_statement('info', f'{LOG_INS}::Criterion {type(criterion).__name__} moved to device {self.device}.', __name__)

        # --- Metrics ---
        self.metrics = TrainingMetrics()
        log_statement('debug', f'{LOG_INS}::TrainingMetrics initialized.', __name__)

        # --- State Variables ---
        self.current_epoch = 0
        self.total_pruned_count = 0
        log_statement('debug', f'{LOG_INS}::Initial state set: current_epoch={self.current_epoch}, total_pruned_count={self.total_pruned_count}', __name__)

        # --- Optimizer and Scheduler ---
        log_statement('debug', f'{LOG_INS}::Initializing optimizer and scheduler...', __name__)
        self.optimizer, self.scheduler = self._init_optimizer_scheduler()
        if self.optimizer:
            log_statement('info', f'{LOG_INS}::Optimizer initialized: {type(self.optimizer).__name__}', __name__)
        else:
            log_statement('warning', f'{LOG_INS}::Optimizer initialization failed or returned None.', __name__)
        if self.scheduler:
            log_statement('info', f'{LOG_INS}::Scheduler initialized: {type(self.scheduler).__name__}', __name__)
        else:
            log_statement('info', f'{LOG_INS}::No scheduler will be used.', __name__)

        # --- Log Final Runtime Configuration ---
        self._log_training_config()

        log_statement('info', f'{LOG_INS}::EnhancedTrainer initialized successfully for {type(model).__name__} on {self.device}.', __name__)

    def _create_runtime_config(self, overrides: Optional[Dict[str, Any]]) -> TrainingConfig:
        """
        Creates a TrainingConfig instance, applying overrides from the dictionary.
        Logs the process and handles potential errors.
        """
        log_statement('debug', f'{LOG_INS}::Creating default TrainingConfig instance.', __name__)
        config_instance = TrainingConfig() # Get defaults

        if overrides:
            log_statement('info', f'{LOG_INS}::Applying training config overrides: {overrides}', __name__)
            for key, value in overrides.items():
                log_statement('debug', f'{LOG_INS}::Processing override: {key}={value}', __name__)
                if hasattr(config_instance, key):
                    try:
                        # Get the type of the default attribute to attempt conversion
                        default_value = getattr(config_instance, key)
                        required_type = type(default_value)
                        log_statement('debug', f'{LOG_INS}::Attribute {key} exists with type {required_type}. Attempting to set value {value} ({type(value)}).', __name__)
                        # Basic type coercion (can be expanded)
                        converted_value = value
                        if required_type != type(value):
                            try:
                                converted_value = required_type(value)
                                log_statement('debug', f'{LOG_INS}::Successfully coerced override value for {key} to type {required_type}.', __name__)
                            except Exception as conv_e:
                                log_statement('warning', f'{LOG_INS}::Could not convert override value {value} for {key} to required type {required_type}. Using original value. Error: {conv_e}', __name__)
                                # Keep original 'value' if conversion fails, maybe log error later if setattr fails

                        setattr(config_instance, key, converted_value)
                        log_statement('info', f'{LOG_INS}::Override applied: {key} = {getattr(config_instance, key)}', __name__)
                    except TypeError as e:
                        log_statement('error', f'{LOG_INS}::Type error applying config override {key}={value}. Required type might be {type(getattr(config_instance, key))}. Error: {e}', __name__, exc_info=True)
                    except Exception as e:
                        log_statement('error', f'{LOG_INS}::Unexpected error applying config override {key}={value}. Error: {e}', __name__, exc_info=True)
                else:
                    log_statement('warning', f'{LOG_INS}::Ignoring unknown config override key: {key}', __name__)
        else:
            log_statement('info', f'{LOG_INS}::No training configuration overrides provided.', __name__)

        log_statement('debug', f'{LOG_INS}::Runtime configuration instance created.', __name__)
        return config_instance

    def _log_training_config(self):
        """ Logs the final runtime training configuration to standard logs and a file. """
        log_statement('info', f'{LOG_INS}::Logging final runtime training configuration...', __name__)
        config_dict = {}
        try:
            # Extract config attributes
            log_statement('debug', f'{LOG_INS}::Extracting attributes from self.config.', __name__)
            for attr in dir(self.config):
                if not attr.startswith('_') and not callable(getattr(self.config, attr)):
                     value = getattr(self.config, attr)
                     # Handle nested config classes like LoggingConfig
                     if isinstance(value, type): # If it's a class itself (like LoggingConfig)
                          nested_dict = {}
                          for nested_attr in dir(value):
                               if not nested_attr.startswith('_') and not callable(getattr(value, nested_attr)):
                                    nested_dict[nested_attr] = getattr(value, nested_attr)
                          config_dict[attr] = nested_dict
                     else:
                          config_dict[attr] = value

            log_statement('debug', f'{LOG_INS}::Adding non-config parameters to log dict.', __name__)
            # Add relevant non-config params
            config_dict['DEVICE'] = str(self.device)
            config_dict['MODEL_CLASS'] = type(self.model).__name__
            config_dict['CRITERION_CLASS'] = type(self.criterion).__name__
            if self.optimizer:
                config_dict['OPTIMIZER_CLASS'] = type(self.optimizer).__name__
            if self.scheduler:
                 config_dict['SCHEDULER_CLASS'] = type(self.scheduler).__name__

            log_statement('debug', f'{LOG_INS}::Serializing configuration to JSON string.', __name__)
            # Use default=str for non-serializable types (like paths, classes)
            config_str = json.dumps(config_dict, indent=2, default=str)

            # Log to standard logger
            log_statement('warning', f"{LOG_INS}::--- Training Configuration Used ---", __name__)
            for line in config_str.splitlines():
                log_statement('info', line, __name__)
            log_statement('warning', f"{LOG_INS}::---------------------------------", __name__)

            # Save to dedicated config log file
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            config_log_path = LOG_DIR / f"training_config_{ts}.log"
            log_statement('debug', f'{LOG_INS}::Attempting to save config to file: {config_log_path}', __name__)
            with open(config_log_path, 'w') as f:
                f.write(config_str)
            log_statement('info', f"{LOG_INS}::Training configuration saved to: {config_log_path}", __name__)

        except Exception as e:
            log_statement('error', f"{LOG_INS}::Failed to log training configuration: {e}", __name__, exc_info=True)

    def _init_optimizer_scheduler(self) -> Tuple[Optional[optim.Optimizer], Optional[Any]]:
        """
        Configures the optimizer and learning rate scheduler based on runtime config.
        Handles potential errors during initialization.
        """
        log_statement('debug', f'{LOG_INS}::Initializing optimizer...', __name__)
        optimizer = None
        try:
            # Example: AdamW optimizer using config values
            lr = self.config.INITIAL_LR
            wd = self.config.WEIGHT_DECAY
            log_statement('debug', f'{LOG_INS}::Using AdamW with parameters: lr={lr}, weight_decay={wd}', __name__)
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
            log_statement('info', f"{LOG_INS}::Using AdamW optimizer: LR={lr}, WD={wd}", __name__)
        except AttributeError as e:
             log_statement('error', f"{LOG_INS}::Missing required attribute in TrainingConfig for optimizer: {e}", __name__, exc_info=True)
             return None, None # Cannot proceed without optimizer
        except Exception as e:
             log_statement('error', f"{LOG_INS}::Failed to initialize optimizer: {e}", __name__, exc_info=True)
             return None, None # Cannot proceed without optimizer

        log_statement('debug', f'{LOG_INS}::Initializing scheduler...', __name__)
        scheduler = None
        try:
            batches_per_epoch = 0
            log_statement('debug', f'{LOG_INS}::Attempting to get length of data_loader.', __name__)
            try:
                batches_per_epoch = len(self.data_loader)
                log_statement('debug', f'{LOG_INS}::DataLoader length: {batches_per_epoch}', __name__)
            except TypeError:
                log_statement('warning', f"{LOG_INS}::DataLoader does not support len(). Scheduler T_max will be estimated.", __name__)
                batches_per_epoch = 0 # Indicate len() failed

            if batches_per_epoch > 0:
                t_max = self.config.MAX_EPOCHS * batches_per_epoch
                eta_min = 0 # Common default for CosineAnnealingLR
                log_statement('info', f"{LOG_INS}::Using CosineAnnealingLR scheduler with T_max={t_max} (Epochs={self.config.MAX_EPOCHS}, BatchesPerEpoch={batches_per_epoch}), eta_min={eta_min}", __name__)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            else:
                # Fallback if len() fails or is zero
                t_max_estimate = self.config.MAX_EPOCHS * 1000 # Estimate batches/epoch
                eta_min = 0
                log_statement('warning', f"Using CosineAnnealingLR scheduler with estimated T_max={t_max_estimate}, eta_min={eta_min}", __name__)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_estimate, eta_min=eta_min)

        except AttributeError as e:
             log_statement('error', f"{LOG_INS}::Missing required attribute in TrainingConfig for scheduler: {e}", __name__, exc_info=True)
             scheduler = None # Proceed without scheduler
        except Exception as e:
            # Catch more general errors during scheduler init
            log_statement('error', f"{LOG_INS}::Failed to initialize scheduler: {e}. Proceeding without scheduler.", __name__, exc_info=True)
            scheduler = None

        log_statement('debug', f'{LOG_INS}::Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__ if scheduler else "None"}', __name__)
        return optimizer, scheduler

    def _apply_pruning(self):
        """
        Applies global unstructured magnitude pruning to the model's weights
        based on the runtime configuration. Makes pruning permanent after applying.
        """
        log_statement('info', f'{LOG_INS}::Checking pruning configuration...', __name__)
        prune_amount = getattr(self.config, 'PRUNE_AMOUNT', 0.0) # Default to 0 if not set

        if prune_amount <= 0:
            log_statement('info', f"{LOG_INS}::Pruning amount ({prune_amount}) is zero or negative. Skipping pruning.", __name__)
            return

        log_statement('info', f'{LOG_INS}::Attempting to apply global unstructured pruning: Amount={prune_amount:.2f}', __name__)
        parameters_to_prune: List[Tuple[nn.Module, str]] = []
        log_statement('debug', f'{LOG_INS}::Identifying prunable parameters (Linear/Conv2d weights)...', __name__)
        for module in self.model.modules():
            # Prune Linear and Conv2d layers by default
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                log_statement('debug', f'{LOG_INS}::Checking module: {module}', __name__)
                # Ensure the parameter 'weight' exists and is not None before adding
                if hasattr(module, 'weight') and module.weight is not None:
                    parameters_to_prune.append((module, 'weight'))
                    log_statement('debug', f'{LOG_INS}::Adding {module}.weight to prune list.', __name__)
                # else:
                #     log_statement('debug', f'{LOG_INS}::Skipping {module}: no weight attribute or weight is None.', __name__)

        if not parameters_to_prune:
            log_statement('warning', f"{LOG_INS}::No prunable parameters (Linear/Conv2d weights) found in the model.", __name__)
            return

        log_statement('info', f'{LOG_INS}::Found {len(parameters_to_prune)} parameter groups to prune.', __name__)

        try:
            log_statement('debug', f'{LOG_INS}::Applying prune.global_unstructured with method L1Unstructured, amount {prune_amount}.', __name__)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured, # Magnitude pruning (L1 norm)
                amount=prune_amount
            )
            log_statement('info', f"{LOG_INS}::Applied global unstructured pruning mask (Amount={prune_amount:.2f}).", __name__)

            # Make pruning permanent and calculate counts
            log_statement('info', f'{LOG_INS}::Making pruning permanent and calculating pruned counts...', __name__)
            current_total_pruned = 0
            total_params = 0
            for module, param_name in parameters_to_prune:
                log_statement('debug', f'{LOG_INS}::Processing {module}.{param_name} for permanent pruning.', __name__)
                param = getattr(module, param_name) # Get the parameter tensor
                if prune.is_pruned(module):
                    log_statement('debug', f'{LOG_INS}::Removing pruning hook for {module}.{param_name}.', __name__)
                    prune.remove(module, param_name) # Makes pruning permanent
                    log_statement('debug', f'{LOG_INS}::Pruning hook removed for {module}.{param_name}.', __name__)
                    # Recalculate zeros after removal
                    zeros = torch.sum(param == 0).item()
                    params_in_tensor = param.nelement()
                    current_total_pruned += zeros
                    total_params += params_in_tensor
                    log_statement('debug', f'{LOG_INS}::Post-removal: {zeros}/{params_in_tensor} zeros in {module}.{param_name}.', __name__)
                else:
                    # If not pruned by global_unstructured (shouldn't happen?), count existing zeros
                    zeros = torch.sum(param == 0).item()
                    params_in_tensor = param.nelement()
                    current_total_pruned += zeros
                    total_params += params_in_tensor
                    log_statement('debug', f'{LOG_INS}::Module {module}.{param_name} was not pruned by global_unstructured. Counted existing {zeros}/{params_in_tensor} zeros.', __name__)


            self.total_pruned_count = current_total_pruned # Update total count
            pruned_percentage = (self.total_pruned_count / total_params) * 100 if total_params > 0 else 0.0
            log_statement('info', f"{LOG_INS}::Pruning made permanent. Total pruned weights: {self.total_pruned_count} ({pruned_percentage:.2f}% of {total_params} total params in pruned layers).", __name__)

        except Exception as e:
            log_statement('error', f"{LOG_INS}::Pruning application or removal failed: {e}", __name__, exc_info=True)

    def train_epoch(self) -> float:
        """
        Runs the training process for a single epoch, iterating through the data_loader.
        Handles batch processing, loss calculation, backpropagation, optimizer steps,
        scheduler steps, metrics recording, progress bar updates, and intra-epoch checkpointing.

        Returns:
            float: The average loss for the epoch.
        """
        log_statement('info', f"{LOG_INS}::Starting Epoch {self.current_epoch + 1}/{self.config.MAX_EPOCHS}", __name__)
        self.model.train() # Set model to training mode
        total_loss = 0.0
        batch_count = 0
        epoch_start_time = datetime.now()

        log_statement('debug', f'{LOG_INS}::Determining total batches for progress bar...', __name__)
        try:
            total_batches = len(self.data_loader)
            has_len = True
            log_statement('debug', f'{LOG_INS}::Total batches in epoch: {total_batches}', __name__)
        except TypeError:
            total_batches = None # Dataloader might be infinite/iterable
            has_len = False
            log_statement('warning', f'{LOG_INS}::DataLoader has no length. Progress bar total will be unknown.', __name__)

        pbar_desc = f"Epoch {self.current_epoch + 1}/{self.config.MAX_EPOCHS}"
        log_statement('debug', f'{LOG_INS}::Initializing tqdm progress bar: desc="{pbar_desc}", total={total_batches}', __name__)
        pbar = tqdm(self.data_loader, desc=pbar_desc, total=total_batches, unit="batch", leave=True) # Keep after loop

        # --- Intra-epoch Checkpointing Setup ---
        last_checkpoint_batch = -1
        batches_between_checkpoints = -1
        checkpoint_interval_percent = getattr(self.config, 'CHECKPOINT_INTERVAL_BATCH_PERCENT', 0.0)
        if has_len and checkpoint_interval_percent > 0 and total_batches > 0:
             # Calculate how many batches between checkpoints
             # Ensure at least 1 batch between checkpoints if percent is very small
             batches_between_checkpoints = max(1, int(total_batches * checkpoint_interval_percent))
             log_statement('info', f'{LOG_INS}::Intra-epoch checkpointing enabled every {batches_between_checkpoints} batches.', __name__)
        else:
             log_statement('info', f'{LOG_INS}::Intra-epoch checkpointing disabled (interval <= 0 or dataloader has no length).', __name__)
        # --- End Checkpointing Setup ---

        log_statement('debug', f'{LOG_INS}::Starting batch iteration for epoch...', __name__)
        for batch_idx, batch_data in enumerate(pbar):
            batch_start_time = datetime.now()
            log_statement('debug', f'{LOG_INS}::Starting Batch {batch_idx + 1}{f"/{total_batches}" if has_len else ""}', __name__)

            try:
                # --- Data Handling ---
                log_statement('debug', f'{LOG_INS}::Unpacking batch data {batch_idx}...', __name__)
                inputs, targets = batch_data # Assuming structure
                log_statement('debug', f'{LOG_INS}::Moving batch data to device: {self.device}...', __name__)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                log_statement('debug', f'{LOG_INS}::Input shape: {inputs.shape}, Target shape: {targets.shape}', __name__)

                # --- Forward and Backward Pass ---
                log_statement('debug', f'{LOG_INS}::Zeroing optimizer gradients.', __name__)
                self.optimizer.zero_grad()

                log_statement('debug', f'{LOG_INS}::Performing model forward pass...', __name__)
                outputs = self.model(inputs)
                log_statement('debug', f'{LOG_INS}::Output shape: {outputs.shape}', __name__)

                log_statement('debug', f'{LOG_INS}::Calculating loss...', __name__)
                loss = self.criterion(outputs, targets)
                batch_loss = loss.item() # Get scalar value
                log_statement('debug', f'{LOG_INS}::Batch loss calculated: {batch_loss:.6f}', __name__)

                log_statement('debug', f'{LOG_INS}::Performing backward pass (calculating gradients)...', __name__)
                loss.backward()

                # Optional: Gradient Clipping (Uncomment if needed)
                # log_statement('debug', f'{LOG_INS}::Applying gradient clipping...', __name__)
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                log_statement('debug', f'{LOG_INS}::Performing optimizer step (updating weights)...', __name__)
                self.optimizer.step()

                # --- Scheduler Step ---
                if self.scheduler:
                    log_statement('debug', f'{LOG_INS}::Performing scheduler step.', __name__)
                    self.scheduler.step() # Step per batch typical for CosineAnnealingLR

                # --- Metrics and Logging ---
                batch_duration = (datetime.now() - batch_start_time).total_seconds()
                current_lr = self.optimizer.param_groups[0]['lr']
                log_statement('debug', f'{LOG_INS}::Batch {batch_idx + 1} completed. Loss: {batch_loss:.4f}, LR: {current_lr:.6f}, Duration: {batch_duration:.3f}s', __name__)

                total_loss += batch_loss
                batch_count += 1

                log_statement('debug', f'{LOG_INS}::Recording batch metrics.', __name__)
                self.metrics.record(
                    epoch=self.current_epoch + 1, # Use 1-based epoch for recording
                    batch=batch_idx + 1,       # Use 1-based batch for recording
                    loss=batch_loss,
                    lr=current_lr,
                    pruned_count=self.total_pruned_count, # Record current pruned count
                    duration=batch_duration,
                    batch_size=inputs.size(0) # Record actual batch size
                )

                # Update progress bar
                pbar.set_postfix({'Loss': f"{batch_loss:.4f}", 'LR': f"{current_lr:.6f}"})

                # --- Intra-epoch Checkpointing ---
                current_batch_num = batch_idx + 1
                if batches_between_checkpoints > 0 and current_batch_num % batches_between_checkpoints == 0:
                    # Avoid double checkpoint if interval is 1 batch, or at very start/end
                    if current_batch_num != last_checkpoint_batch and current_batch_num != total_batches:
                        reason = f"epoch_{self.current_epoch+1}_batch_{current_batch_num}"
                        log_statement('info', f'{LOG_INS}::Triggering intra-epoch checkpoint at batch {current_batch_num} (Reason: {reason}).', __name__)
                        self._save_checkpoint(reason=reason)
                        last_checkpoint_batch = current_batch_num

            except Exception as e:
                log_statement('error', f"{LOG_INS}::Critical error during training batch {batch_idx + 1} in epoch {self.current_epoch + 1}: {e}", __name__, exc_info=True)
                # Decide recovery strategy: continue, break epoch, or raise
                log_statement('warning', f'{LOG_INS}::Skipping problematic batch {batch_idx + 1} due to error.', __name__)
                continue # Example: Skip batch on error

        pbar.close()
        log_statement('debug', f'{LOG_INS}::Finished batch iteration for epoch.', __name__)
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        log_statement('info', f"{LOG_INS}::Epoch {self.current_epoch + 1} finished. Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s", __name__)

        return avg_loss

    def train(self):
        """
        Runs the full training loop over the configured number of epochs.
        Manages epoch iteration, calls train_epoch, handles pruning intervals,
        saves checkpoints, and saves metrics.
        """
        max_epochs = getattr(self.config, 'MAX_EPOCHS', 1) # Default to 1 epoch if not set
        log_statement('info', f"{LOG_INS}::Starting full training loop for {max_epochs} epochs...", __name__)
        start_time = datetime.now()

        initial_epoch = self.current_epoch # In case loaded from checkpoint
        log_statement('debug', f"{LOG_INS}::Starting from epoch index: {initial_epoch}", __name__)

        for epoch in range(initial_epoch, max_epochs):
            self.current_epoch = epoch # Update internal epoch index (0-based)
            log_statement('info', f"{LOG_INS}::===== Starting Epoch {epoch + 1}/{max_epochs} =====", __name__)

            # --- Run Single Epoch ---
            avg_loss = self.train_epoch()
            log_statement('info', f"{LOG_INS}::Epoch {epoch + 1} average loss: {avg_loss:.4f}", __name__)

            # --- Pruning Step (Scheduled End of Epoch) ---
            prune_interval = getattr(self.config, 'PRUNE_INTERVAL_EPOCHS', 0)
            if prune_interval > 0 and (epoch + 1) % prune_interval == 0:
                log_statement('info', f"{LOG_INS}::Applying pruning after epoch {epoch + 1} (Interval: {prune_interval} epochs).", __name__)
                self._apply_pruning()

            # --- End of Epoch Checkpoint ---
            reason = f"epoch_{epoch+1}_end"
            log_statement('info', f"{LOG_INS}::Saving end-of-epoch checkpoint (Reason: {reason}).", __name__)
            self._save_checkpoint(reason=reason)

            # --- Save Metrics (e.g., every epoch) ---
            log_statement('info', f"{LOG_INS}::Saving metrics after epoch {epoch + 1}.", __name__)
            self.metrics.save() # Saves with timestamp by default

            # --- Optional: Add validation loop call here ---
            # log_statement('info', f"{LOG_INS}::Running validation after epoch {epoch + 1}.", __name__)
            # self.validate()

        # --- Training Finished ---
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        actual_epochs_run = max_epochs - initial_epoch
        log_statement('info', f"{LOG_INS}::Training loop finished after {actual_epochs_run} epochs.", __name__)
        log_statement('info', f"{LOG_INS}::Total training duration: {total_duration:.2f} seconds.", __name__)

        # Save final metrics state with a specific name
        final_metrics_filename = f"{getattr(self.config, 'METRICS_FILENAME_PREFIX', 'training_metrics')}_final.csv"
        log_statement('info', f"{LOG_INS}::Saving final metrics to {final_metrics_filename}.", __name__)
        self.metrics.save(filename=final_metrics_filename)

    def _save_checkpoint(self, reason: str = "checkpoint"):
        """
        Saves the current training state (model, optimizer, scheduler, epoch, pruned count)
        using the configured helper function `save_state`.
        """
        filename = f"{type(self.model).__name__}_{reason}.pt"
        log_statement('info', f"{LOG_INS}::Saving checkpoint: {filename} (Reason: {reason})", __name__)
        log_statement('debug', f'{LOG_INS}::Current state to save: epoch={self.current_epoch}, pruned_count={self.total_pruned_count}', __name__)
        try:
            # Call the helper function, passing all relevant state
            save_state(
                model=self.model,
                filename=filename,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch, # Save the *completed* epoch index (0-based)
                total_pruned_count=self.total_pruned_count
                # Add any other custom data needed for restoring state if necessary
                # custom_data = {'example': 'value'}
            )
            log_statement('info', f'{LOG_INS}::Checkpoint save initiated for {filename}.', __name__)
        except Exception as e:
            # Error should ideally be logged within save_state, but log here too.
            log_statement('error', f"{LOG_INS}::Failed to initiate checkpoint save for '{filename}'. Error: {e}", __name__, exc_info=True)

    def load_checkpoint(self, filename: str, load_optimizer: bool = True, load_scheduler: bool = True, strict: bool = True) -> bool:
        """
        Loads training state from a checkpoint file using the helper function `load_state`.
        Restores model weights, and optionally optimizer/scheduler states, epoch, and pruned count.

        Args:
            filename (str): The name of the checkpoint file (expected in checkpoint dir).
            load_optimizer (bool): Whether to load the optimizer state. Defaults to True.
            load_scheduler (bool): Whether to load the scheduler state. Defaults to True.
            strict (bool): Whether to strictly enforce that the keys in state_dict
                           match the keys returned by this module's state_dict().
                           Defaults to True.

        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise.
        """
        log_statement('info', f"{LOG_INS}::Attempting to load checkpoint: {filename}", __name__)
        log_statement('debug', f'{LOG_INS}::Load flags: optimizer={load_optimizer}, scheduler={load_scheduler}, strict={strict}', __name__)
        try:
            # Call the helper function
            checkpoint_data = load_state(
                model=self.model,
                filename=filename,
                optimizer=self.optimizer if load_optimizer else None,
                scheduler=self.scheduler if load_scheduler else None,
                device=self.device,
                strict=strict
            )

            if checkpoint_data:
                log_statement('info', f"{LOG_INS}::Checkpoint file '{filename}' loaded by helper function.", __name__)
                # Restore metadata if available
                if 'epoch' in checkpoint_data and checkpoint_data['epoch'] is not None:
                    # Epoch saved is the completed epoch (0-based), so start next one
                    self.current_epoch = checkpoint_data['epoch'] + 1
                    log_statement('info', f"{LOG_INS}::Resuming training from epoch {self.current_epoch} (loaded completed epoch {checkpoint_data['epoch']}).", __name__)
                else:
                    log_statement('warning', f"'epoch' key not found or is None in checkpoint '{filename}'. Starting from epoch 0.", __name__)
                    self.current_epoch = 0 # Reset if not found

                if 'total_pruned_count' in checkpoint_data and checkpoint_data['total_pruned_count'] is not None:
                    self.total_pruned_count = checkpoint_data['total_pruned_count']
                    log_statement('info', f"{LOG_INS}::Restored pruned weight count: {self.total_pruned_count}", __name__)
                else:
                     log_statement('warning', f"'total_pruned_count' not found or is None in checkpoint '{filename}'. Resetting to 0.", __name__)
                     self.total_pruned_count = 0 # Reset if not found

                # Add logic here to restore metrics state if it was saved in the checkpoint
                # Example: if 'metrics_data' in checkpoint_data: self.metrics.load_state(checkpoint_data['metrics_data'])

                log_statement('info', f"{LOG_INS}::Checkpoint '{filename}' processed successfully.", __name__)
                return True
            else:
                # load_state returned None, likely file not found or other load error logged within helper
                log_statement('error', f"{LOG_INS}::Failed to load checkpoint '{filename}'. Check previous logs from load_state helper.", __name__)
                return False

        except Exception as e:
            log_statement('error', f"{LOG_INS}::Critical error during checkpoint loading process for {filename}: {e}", __name__, exc_info=True)
            return False

    # --- Optional: Validation Loop ---
    # def validate(self):
    #     """Runs a validation loop on a separate dataset."""
    #     self.model.eval() # Set model to evaluation mode
    #     total_val_loss = 0.0
    #     # ... loop through validation data_loader ...
    #     with torch.no_grad():
    #         # ... forward pass, calculate loss ...
    #     avg_val_loss = total_val_loss / len(self.validation_loader)
    #     log_statement(loglevel=str("info"), logstatement=str(f"Epoch {self.current_epoch} Validation Loss: {avg_val_loss:.4f}"), main_logger=str(__name__))
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

