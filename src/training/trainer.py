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

# Import project configuration and utilities
try:
    from ..utils.config import TrainingConfig, DEFAULT_DEVICE, CHECKPOINT_DIR, LOG_DIR
    from ..utils.logger import configure_logging
    from ..utils.helpers import save_state, load_state # Use helper functions
    # Import required model and data loader classes (adjust as needed)
    from ..core.models import ZoneClassifier # Example model
    from ..data.loaders import EnhancedDataLoader # Example loader
    configure_logging(log_file_path=LOG_DIR / 'training.log', error_log_file_path=LOG_DIR / 'errors.log', log_level=logging.INFO, console_logging=True, file_logging=True)
    logger = logging.getLogger(__name__)
    logger.info("Training module initialized with logging.")
    logger.info(f"Using device: {DEFAULT_DEVICE}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Training configuration: {TrainingConfig.__dict__}")
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

logger = logging.getLogger(__name__)

# --- Training Metrics Class ---
# class TrainingMetrics:
#     def __init__(self, save_dir: Optional[Path] = None):
#         self.metrics_data = []; self.columns = ['timestamp','epoch','batch','loss','lr','pruned_weights_count','batch_duration_sec','batch_size']
#         self.save_dir = Path(save_dir or LOG_DIR); self.save_dir.mkdir(parents=True, exist_ok=True)
#         logger.info(f"TrainingMetrics OK. Save dir: {self.save_dir}")
#     def record(self, epoch, batch, loss, lr, pruned_count, duration, batch_size):
#         loss_item=loss.item() if isinstance(loss,torch.Tensor) else float(loss); lr_item=lr[0] if isinstance(lr,list) else float(lr)
#         self.metrics_data.append({'timestamp':datetime.now(),'epoch':int(epoch),'batch':int(batch),'loss':loss_item,'lr':lr_item,'pruned_weights_count':int(pruned_count),'batch_duration_sec':float(duration),'batch_size':int(batch_size)})
#     def get_dataframe(self) -> pd.DataFrame: return pd.DataFrame(self.metrics_data, columns=self.columns) if self.metrics_data else pd.DataFrame(columns=self.columns)
#     def save(self, filename: Optional[str] = None):
#         if not self.metrics_data: logger.warning("No metrics to save."); return
#         df=self.get_dataframe(); df['timestamp']=pd.to_datetime(df['timestamp']); df['epoch']=df['epoch'].astype(int); df['batch']=df['batch'].astype(int); df['loss']=df['loss'].astype(float); df['lr']=df['lr'].astype(float); df['pruned_weights_count']=df['pruned_weights_count'].astype(int); df['batch_duration_sec']=df['batch_duration_sec'].astype(float); df['batch_size']=df['batch_size'].astype(int)
#         filename=filename or f"{TrainingConfig.METRICS_FILENAME_PREFIX}_{datetime.now():%Y%m%d%H%M%S}.csv"; filepath=self.save_dir/filename
#         try: df.to_csv(filepath, index=False, encoding='utf-8'); logger.info(f"Saved metrics ({len(df)} rows) to {filepath}")
#         except Exception as e: logger.error(f"Failed metrics save to {filepath}: {e}", exc_info=True)
#     def clear(self): self.metrics_data=[]; logger.info("Cleared training metrics.")

# --- Enhanced Trainer Class ---
class EnhancedTrainer:
    """
    Comprehensive training loop with configurable hyperparameters, pruning,
    checkpointing, and metrics.
    """
    def __init__(self,
                 model: nn.Module,
                 data_loader, # Expects EnhancedDataLoader instance or similar
                 criterion, # Expects torch loss function instance
                 device: Optional[str | torch.device] = None,
                 training_config_override: Optional[dict] = None): # Accept overrides
        """
        Initializes the EnhancedTrainer.

        Args:
            model (nn.Module): The neural network model to train.
            data_loader: Iterable data loader providing (inputs, targets) batches.
            criterion: Loss function instance (e.g., nn.MSELoss()).
            device (str | torch.device, optional): Device ('cuda', 'cpu'). Defaults to config.
            training_config_override (dict, optional): Dictionary to override TrainingConfig
                                                       defaults (e.g., {'MAX_EPOCHS': 10, 'INITIAL_LR': 1e-3}).
        """
        # Load base config and apply overrides
        self.config = self._create_runtime_config(training_config_override)

        self.device = device or DEFAULT_DEVICE
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.criterion = criterion.to(self.device)
        self.metrics = TrainingMetrics()

        self.current_epoch = 0
        self.total_pruned_count = 0

        self.optimizer, self.scheduler = self._init_optimizer_scheduler()

        # Log the final configuration being used for this training run
        self._log_training_config()

        logger.info(f"EnhancedTrainer initialized for {type(model).__name__} on {self.device}")

    def _create_runtime_config(self, overrides: Optional[dict]) -> TrainingConfig:
        """ Creates a TrainingConfig instance applying overrides. """
        config_instance = TrainingConfig() # Get defaults
        if overrides:
            logger.info(f"Applying training config overrides: {overrides}")
            for key, value in overrides.items():
                 if hasattr(config_instance, key):
                      # Optionally add type checking/conversion here
                      try:
                           setattr(config_instance, key, value)
                           logger.debug(f"Override applied: {key} = {value}")
                      except TypeError as e:
                           logger.error(f"Type error setting config override {key}={value}: {e}")
                 else:
                      logger.warning(f"Ignoring unknown config override key: {key}")
        return config_instance

    def _log_training_config(self):
        """ Logs the runtime training configuration. """
        config_dict = {attr: getattr(self.config, attr)
                       for attr in dir(self.config)
                       if not attr.startswith('_') and not callable(getattr(self.config, attr))}
        # Add relevant non-config params like device, model type, criterion type
        config_dict['DEVICE'] = str(self.device)
        config_dict['MODEL_CLASS'] = type(self.model).__name__
        config_dict['CRITERION_CLASS'] = type(self.criterion).__name__
        # Log as pretty JSON
        try:
             config_str = json.dumps(config_dict, indent=2, default=str) # Use default=str for non-serializable
             logger.info(f"--- Training Configuration Used ---")
             for line in config_str.splitlines(): # Log each line for better readability in file
                 logger.info(line)
             logger.info(f"---------------------------------")
             # Also save to a dedicated config log file
             config_log_path = LOG_DIR / f"training_config_{datetime.now():%Y%m%d%H%M%S}.log"
             with open(config_log_path, 'w') as f:
                  f.write(config_str)
             logger.info(f"Training config saved to: {config_log_path}")
        except Exception as e:
             logger.error(f"Failed to log training configuration: {e}")

    def _init_optimizer_scheduler(self):
        """Configures optimizer/scheduler based on runtime config."""
        # Example: AdamW optimizer using config values
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.INITIAL_LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        logger.info(f"Using AdamW optimizer: LR={self.config.INITIAL_LR}, WD={self.config.WEIGHT_DECAY}")

        # Example: Cosine Annealing scheduler using config values
        scheduler = None
        try:
            batches_per_epoch = len(self.data_loader)
            if batches_per_epoch > 0: # Ensure batches_per_epoch is valid
                t_max = self.config.MAX_EPOCHS * batches_per_epoch
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
                logger.info(f"Using CosineAnnealingLR scheduler: T_max={t_max}")
            else:
                 logger.warning("DataLoader length is 0. Cannot initialize CosineAnnealingLR scheduler.")
        except TypeError:
            t_max_estimate = self.config.MAX_EPOCHS * 1000 # Fallback T_max
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_estimate, eta_min=0)
            logger.warning(f"DataLoader has no length. Using estimated T_max={t_max_estimate} for scheduler.")
        except Exception as e:
             logger.error(f"Failed to initialize scheduler: {e}. Proceeding without scheduler.", exc_info=True)
             scheduler = None

        return optimizer, scheduler

    def _apply_pruning(self):
        """ Applies pruning based on runtime config. """
        # (Implementation remains the same - uses self.config)
        if not hasattr(self.config, 'PRUNE_AMOUNT') or self.config.PRUNE_AMOUNT <= 0: logger.info("Pruning skipped."); return
        params_to_prune = [(m, 'weight') for m in self.model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)) and hasattr(m, 'weight')]
        if not params_to_prune: logger.warning("No prunable layers found."); return
        try:
            prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=self.config.PRUNE_AMOUNT)
            logger.info(f"Applied pruning: Amount={self.config.PRUNE_AMOUNT:.2f}")
            pruned_count, total_params = 0, 0
            for module, name in params_to_prune:
                if prune.is_pruned(module): prune.remove(module, name) # Make permanent
                param = getattr(module, name); pruned_count += torch.sum(param == 0).item(); total_params += param.nelement()
            self.total_pruned_count = pruned_count; perc = (pruned_count/total_params)*100 if total_params else 0
            logger.info(f"Pruning made permanent. Total pruned: {pruned_count} ({perc:.2f}%)")
        except Exception as e: logger.error(f"Pruning failed: {e}", exc_info=True)

    def train_epoch(self):
        """ Runs training for one epoch. """
        # (Implementation mostly the same - uses self.config, logs progress)
        self.model.train(); total_loss = 0.0; batch_count = 0; epoch_start_time = datetime.now()
        try: total_batches = len(self.data_loader); has_len = True
        except TypeError: total_batches = None; has_len = False
        pbar_desc = f"Epoch {self.current_epoch}/{self.config.MAX_EPOCHS}"; pbar = tqdm(self.data_loader, desc=pbar_desc, total=total_batches, unit="batch", leave=True) # leave=True to keep after loop
        last_chkpt_batch=-1; batches_btwn_chkpts = int(total_batches * self.config.CHECKPOINT_INTERVAL_BATCH_PERCENT) if has_len and self.config.CHECKPOINT_INTERVAL_BATCH_PERCENT > 0 else -1

        for batch_idx, batch_data in enumerate(pbar):
            batch_start_time = datetime.now()
            try:
                inputs, targets = batch_data; inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad(); outputs = self.model(inputs); loss = self.criterion(outputs, targets)
                loss.backward(); self.optimizer.step()
                if self.scheduler: self.scheduler.step() # Step scheduler per batch
                batch_dur = (datetime.now() - batch_start_time).total_seconds(); cur_lr = self.optimizer.param_groups[0]['lr']; batch_loss = loss.item()
                total_loss += batch_loss; batch_count += 1
                self.metrics.record(epoch=self.current_epoch, batch=batch_idx, loss=batch_loss, lr=cur_lr, pruned_count=self.total_pruned_count, duration=batch_dur, batch_size=inputs.size(0))
                pbar.set_postfix({'Loss': f"{batch_loss:.4f}", 'LR': f"{cur_lr:.6f}"})
                # Intra-epoch checkpointing
                if has_len and batches_btwn_chkpts > 0 and (batch_idx + 1) % batches_btwn_chkpts == 0 and batch_idx != last_chkpt_batch:
                     self._save_checkpoint(reason=f"epoch_{self.current_epoch}_batch_{batch_idx+1}"); last_chkpt_batch=batch_idx
            except Exception as e: logger.error(f"Error batch {batch_idx} epoch {self.current_epoch}: {e}", exc_info=True); continue # Skip batch on error

        pbar.close(); avg_loss = total_loss / batch_count if batch_count > 0 else 0.0; epoch_dur = (datetime.now() - epoch_start_time).total_seconds()
        logger.info(f"Epoch {self.current_epoch} finished. Avg Loss: {avg_loss:.4f}, Duration: {epoch_dur:.2f}s")
        return avg_loss
    
    def train(self):
        """ Runs the full training loop. """
        # (Uses self.config for MAX_EPOCHS, PRUNE_INTERVAL_EPOCHS)
        logger.info(f"Starting training loop for {self.config.MAX_EPOCHS} epochs...")
        start_time = datetime.now()
        for epoch in range(self.current_epoch, self.config.MAX_EPOCHS):
            with tqdm(total=len(self.data_loader), desc=f"Epoch {epoch+1}/{self.config.MAX_EPOCHS}", unit="batch", leave=True) as pbar:
                pbar.set_postfix({'Loss': 0.0, 'LR': self.optimizer.param_groups[0]['lr']})
                logger.info(f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} started.")
                # Optionally load checkpoint here if needed
                # self.load_checkpoint("ZoneClassifier_epoch_X_end.pt")
                pbar.set_postfix({'Loss': 'N/A', 'LR': 'N/A'})
                pbar.refresh()
                self.model.train(); self.optimizer.zero_grad() # Reset optimizer state
                self.scheduler.step()
            self.current_epoch = epoch; avg_loss = self.train_epoch()
            if hasattr(self.config, 'PRUNE_INTERVAL_EPOCHS') and self.config.PRUNE_INTERVAL_EPOCHS > 0 and (epoch + 1) % self.config.PRUNE_INTERVAL_EPOCHS == 0:
                logger.info(f"Applying pruning after epoch {epoch}."); self._apply_pruning()
            self._save_checkpoint(reason=f"epoch_{epoch}_end") # End of epoch checkpoint
            self.metrics.save() # Save metrics each epoch (or less often if configured)
            # Add validation loop call here if implemented
        end_time = datetime.now(); total_duration = (end_time - start_time).total_seconds()
        logger.info(f"Training finished. Total duration: {total_duration:.2f}s")
        self.metrics.save(filename=f"{TrainingConfig.METRICS_FILENAME_PREFIX}_final.csv") # Save final metrics

    def _save_checkpoint(self, reason: str = "checkpoint"):
        """ Saves state using helpers.save_state. """
        # (Implementation remains the same)
        filename = f"{type(self.model).__name__}_{reason}.pt"
        logger.info(f"Saving checkpoint: {filename}")
        try: save_state(model=self.model, filename=filename, optimizer=self.optimizer, scheduler=self.scheduler, epoch=self.current_epoch, total_pruned_count=self.total_pruned_count)
        except Exception as e: logger.error(f"Checkpoint save failed for '{reason}'. Error logged in save_state.") # Error should be logged in save_state

    def load_checkpoint(self, filepath: str | Path, load_optimizer: bool = True, load_scheduler: bool = True, strict: bool = True):
         """ Loads state using helpers.load_state. """
         # (Implementation remains the same)
         logger.info(f"Attempting load checkpoint: {filepath}")
         try:
              checkpoint_data = load_state(model=self.model, filename=filepath, optimizer=self.optimizer if load_optimizer else None, scheduler=self.scheduler if load_scheduler else None, device=self.device, strict=strict)
              if checkpoint_data:
                   if 'epoch' in checkpoint_data: self.current_epoch = checkpoint_data['epoch'] + 1; logger.info(f"Resuming from epoch {self.current_epoch}")
                   if 'total_pruned_count' in checkpoint_data: self.total_pruned_count = checkpoint_data['total_pruned_count']; logger.info(f"Restored pruned count: {self.total_pruned_count}")
                   logger.info(f"Checkpoint '{filepath}' loaded.")
                   return True
              else: logger.error(f"Load failed for '{filepath}'. File not found or load_state is None."); return False
         except Exception as e: logger.error(f"Error loading checkpoint {filepath}: {e}", exc_info=True); return False

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
        logger.info(f"TrainingMetrics initialized. Metrics will be saved to: {self.save_dir}")

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
            logger.warning("No metrics data recorded, skipping save.")
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
            logger.info(f"Saved training metrics ({len(df)} rows) to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save training metrics to {filepath}: {e}", exc_info=True)

    def clear(self):
        """Clears the recorded metrics."""
        self.metrics_data = []
        logger.info("Cleared recorded training metrics.")

class EnhancedTrainer:
    """
    Provides a comprehensive training loop with optimization, scheduling,
    pruning, checkpointing, and metrics tracking.
    """
    def __init__(self, model: nn.Module, data_loader, criterion, device: str | torch.device = None):
        """
        Initializes the EnhancedTrainer.

        Args:
            model (nn.Module): The neural network model to train.
            data_loader: An iterable data loader providing batches of (inputs, targets).
            criterion: The loss function (e.g., nn.MSELoss(), nn.CrossEntropyLoss()).
            device (str | torch.device, optional): The device to run training on.
                                                   Defaults to config.DEFAULT_DEVICE.
        """
        self.config = TrainingConfig() # Load training config
        self.device = device or DEFAULT_DEVICE
        self.model = model.to(self.device)
        self.data_loader = data_loader # Assume loader yields (inputs, targets)
        self.criterion = criterion.to(self.device) # Move loss function to device
        self.metrics = TrainingMetrics() # Initialize metrics tracker

        self.current_epoch = 0
        self.total_pruned_count = 0 # Track total pruned weights

        # Ensure checkpoint directory exists (handled by helpers.save_state now)
        # CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        self.optimizer, self.scheduler = self._init_optimizer_scheduler()
        logger.info(f"EnhancedTrainer initialized for model {type(model).__name__} on device {self.device}")

    def _init_optimizer_scheduler(self):
        """Configures the optimizer and learning rate scheduler."""
        # Example: AdamW optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.INITIAL_LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        logger.info(f"Using AdamW optimizer with LR={self.config.INITIAL_LR}, WeightDecay={self.config.WEIGHT_DECAY}")

        # Example: Cosine Annealing scheduler
        # Calculate T_max based on estimated batches per epoch if loader has __len__
        try:
            batches_per_epoch = len(self.data_loader)
            t_max = self.config.MAX_EPOCHS * batches_per_epoch
            logger.info(f"Using CosineAnnealingLR scheduler with T_max={t_max} (MaxEpochs={self.config.MAX_EPOCHS}, BatchesPerEpoch={batches_per_epoch})")
        except TypeError:
            # If loader has no __len__, use a fixed large number or alternative scheduler
            t_max = self.config.MAX_EPOCHS * 1000 # Placeholder if len() fails
            logger.warning(f"DataLoader has no __len__. Using estimated T_max={t_max} for CosineAnnealingLR.")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=0 # Minimum learning rate
        )

        return optimizer, scheduler

    def _apply_pruning(self):
        """Applies global unstructured magnitude pruning to the model."""
        if self.config.PRUNE_AMOUNT <= 0:
             logger.info("Pruning amount is zero or negative. Skipping pruning step.")
             return

        parameters_to_prune = []
        for module in self.model.modules():
            # Prune Linear and Conv2d layers by default, add others if needed
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Ensure the parameter 'weight' exists before adding
                if hasattr(module, 'weight') and module.weight is not None:
                     parameters_to_prune.append((module, 'weight'))
                # Optionally prune bias as well:
                # if hasattr(module, 'bias') and module.bias is not None:
                #     parameters_to_prune.append((module, 'bias'))

        if not parameters_to_prune:
            logger.warning("No prunable parameters (Linear/Conv2d weights) found in the model.")
            return

        try:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured, # Magnitude pruning (L1 norm)
                amount=self.config.PRUNE_AMOUNT
            )
            logger.info(f"Applied global unstructured pruning with amount {self.config.PRUNE_AMOUNT:.2f}")

            # Make pruning permanent by removing re-parameterization hooks (optional, but good practice after pruning)
            # And calculate total pruned count
            current_total_pruned = 0
            total_params = 0
            for module, param_name in parameters_to_prune:
                 if prune.is_pruned(module):
                      # Remove the hook to make pruning permanent
                      prune.remove(module, param_name)
                      logger.debug(f"Made pruning permanent for {param_name} in {module}")
                      # Count remaining zeros after removal
                      param = getattr(module, param_name)
                      current_total_pruned += torch.sum(param == 0).item()
                      total_params += param.nelement()
                 else: # Should not happen if global_unstructured worked, but check
                      param = getattr(module, param_name)
                      current_total_pruned += torch.sum(param == 0).item() # Count existing zeros
                      total_params += param.nelement()


            self.total_pruned_count = current_total_pruned # Update total count
            pruned_percentage = (self.total_pruned_count / total_params) * 100 if total_params > 0 else 0
            logger.info(f"Total pruned weights after making permanent: {self.total_pruned_count} ({pruned_percentage:.2f}%)")

        except Exception as e:
            logger.error(f"Pruning failed: {e}", exc_info=True)

    def train_epoch(self):
        """Runs training for a single epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0.0
        batch_count = 0
        epoch_start_time = datetime.now()

        # Estimate total batches if possible
        try:
             total_batches = len(self.data_loader)
             has_len = True
        except TypeError:
             total_batches = None
             has_len = False

        # Use tqdm for progress bar
        pbar_desc = f"Epoch {self.current_epoch}/{self.config.MAX_EPOCHS}"
        pbar = tqdm(self.data_loader, desc=pbar_desc, total=total_batches, unit="batch", leave=False)

        last_checkpoint_batch = -1
        batches_between_checkpoints = int(total_batches * self.config.CHECKPOINT_INTERVAL_BATCH_PERCENT) if has_len and self.config.CHECKPOINT_INTERVAL_BATCH_PERCENT > 0 else -1

        for batch_idx, batch_data in enumerate(pbar):
            batch_start_time = datetime.now()

            try:
                # Assuming data_loader yields (inputs, targets)
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                # Optional: Gradient clipping
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Step the scheduler (typically per batch for CosineAnnealingLR)
                if self.scheduler:
                    self.scheduler.step()

                batch_duration = (datetime.now() - batch_start_time).total_seconds()
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_count += 1

                # Record metrics
                self.metrics.record(
                    epoch=self.current_epoch,
                    batch=batch_idx,
                    loss=batch_loss,
                    lr=current_lr,
                    pruned_count=self.total_pruned_count,
                    duration=batch_duration,
                    batch_size=inputs.size(0) # Get batch size from input tensor
                )

                # Update progress bar postfix
                pbar.set_postfix({'Loss': f"{batch_loss:.4f}", 'LR': f"{current_lr:.6f}"})

                # --- Checkpointing within epoch ---
                if has_len and batches_between_checkpoints > 0 and (batch_idx + 1) % batches_between_checkpoints == 0:
                     if batch_idx != last_checkpoint_batch: # Avoid saving twice if interval is 1 batch
                          reason = f"epoch_{self.current_epoch}_batch_{batch_idx+1}"
                          self._save_checkpoint(reason=reason)
                          last_checkpoint_batch = batch_idx

            except Exception as e:
                 logger.error(f"Error during training batch {batch_idx} in epoch {self.current_epoch}: {e}", exc_info=True)
                 # Decide whether to continue to next batch or stop epoch/training
                 # continue # Example: Skip problematic batch

        pbar.close()
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        logger.info(f"Epoch {self.current_epoch} finished. Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s")

        return avg_loss

    def train(self):
        """Runs the full training loop over multiple epochs."""
        logger.info(f"Starting training for {self.config.MAX_EPOCHS} epochs.")
        start_time = datetime.now()

        for epoch in range(self.current_epoch, self.config.MAX_EPOCHS):
            self.current_epoch = epoch
            avg_loss = self.train_epoch()

            # --- Optional Pruning Step (End of Epoch) ---
            if self.config.PRUNE_INTERVAL_EPOCHS > 0 and (epoch + 1) % self.config.PRUNE_INTERVAL_EPOCHS == 0:
                logger.info(f"Applying pruning after epoch {epoch}...")
                self._apply_pruning()

            # --- End of Epoch Checkpoint ---
            self._save_checkpoint(reason=f"epoch_{epoch}_end")

            # --- Save Metrics Periodically (e.g., every epoch) ---
            self.metrics.save() # Saves with timestamp

            # --- Optional: Add validation loop here ---
            # self.validate()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"Training finished after {self.config.MAX_EPOCHS} epochs. Total duration: {total_duration:.2f}s")
        # Save final metrics
        self.metrics.save(filename=f"{TrainingConfig.METRICS_FILENAME_PREFIX}_final.csv")

    def _save_checkpoint(self, reason: str = "checkpoint"):
        """Saves the current model, optimizer, and scheduler state."""
        filename = f"{type(self.model).__name__}_{reason}.pt"
        logger.info(f"Saving checkpoint: {filename}")
        try:
             # Use the helper function
             save_state(
                  model=self.model,
                  filename=filename,
                  optimizer=self.optimizer,
                  scheduler=self.scheduler,
                  epoch=self.current_epoch,
                  # Add any other relevant info
                  total_pruned_count=self.total_pruned_count,
                  # Note: Saving the entire metrics object might be large/inefficient.
                  # Consider saving the path to the metrics file or just recent metrics.
                  # metrics_data=self.metrics.metrics_data[-100:] # Example: last 100 rows
             )
        except Exception as e:
             # Error is logged within save_state
             logger.error(f"Checkpoint saving failed for reason '{reason}'.")

    def load_checkpoint(self, filename: str, load_optimizer: bool = True, load_scheduler: bool = True, strict: bool = True):
         """Loads state from a checkpoint file."""
         logger.info(f"Attempting to load checkpoint: {filename}")
         try:
              # Use the helper function
              checkpoint_data = load_state(
                   model=self.model,
                   filename=filename,
                   optimizer=self.optimizer if load_optimizer else None,
                   scheduler=self.scheduler if load_scheduler else None,
                   device=self.device,
                   strict=strict
              )

              if checkpoint_data:
                   # Restore epoch and other metadata if available
                   if 'epoch' in checkpoint_data and checkpoint_data['epoch'] is not None:
                        self.current_epoch = checkpoint_data['epoch'] + 1 # Start next epoch
                        logger.info(f"Resuming training from epoch {self.current_epoch}")
                   if 'total_pruned_count' in checkpoint_data:
                        self.total_pruned_count = checkpoint_data['total_pruned_count']
                        logger.info(f"Restored pruned weight count: {self.total_pruned_count}")
                   # Restore metrics if saved/needed (might need custom logic)

                   logger.info(f"Checkpoint '{filename}' loaded successfully.")
                   return True
              else:
                   logger.error(f"Failed to load checkpoint '{filename}'. File not found or load_state returned None.")
                   return False

         except Exception as e:
              logger.error(f"Error loading checkpoint {filename}: {e}", exc_info=True)
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
    #     logger.info(f"Epoch {self.current_epoch} Validation Loss: {avg_val_loss:.4f}")
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

