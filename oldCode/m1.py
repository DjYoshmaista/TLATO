# main.py
"""
Main Orchestrator Script

Provides a command-line interface to interact with different subsystems
of the neural processing project (data processing, training, analysis, etc.).
"""

import logging
import random
import sys
import torch # Keep for potential direct use or type hinting

# Import configuration and logger setup first
# Assuming the modular structure with 'src'
try:
    from src.utils.logger import setup_logger, LOG_FILE
    from src.utils import config # Import the whole config module for easy access
except ImportError as e:
     # Fallback if src structure isn't found (e.g., running directly from src?)
     print(f"[ERROR] Failed to import from src. Is the script run from the project root? Error: {e}")
     # Define fallbacks or exit
     LOG_FILE = Path("fallback_app.log") # type: ignore
     logging.basicConfig(level=logging.INFO)
     class config: # Dummy config
          class DataProcessingConfig: REPO_FILE = "fallback_repo.csv.zst"
          class SyntheticDataConfig: TARGET_SAMPLES = 0
          class TestConfig: pass
          DEFAULT_DEVICE = 'cpu'
          TOKENIZED_DATA_DIR = Path('./data/tokenized') # type: ignore

# Import specific functionalities or classes needed for the menu actions
# Use try-except blocks for robustness if some modules might be optional
try:
    from src.data.processing import DataProcessor, Tokenizer, DataRepository
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data processing modules not fully available: {e}")
    DATA_PROCESSING_AVAILABLE = False
    # Define dummy classes to avoid NameErrors later
    class DataProcessor: pass
    class Tokenizer: pass
    class DataRepository: pass

try:
    from src.data.synthetic import SyntheticDataGenerator
    SYNTHETIC_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Synthetic data module not available: {e}")
    SYNTHETIC_DATA_AVAILABLE = False
    class SyntheticDataGenerator: pass

try:
    from src.training.trainer import EnhancedTrainer
    # Need model and loader to instantiate trainer - import them too
    from src.core.models import ZoneClassifier # Example model
    from src.data.loaders import EnhancedDataLoader # Example loader
    import torch.nn as nn # For loss function example
    TRAINING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training modules not fully available: {e}")
    TRAINING_AVAILABLE = False
    # Define dummy classes correctly, each on its own line
    class EnhancedTrainer: pass
    class ZoneClassifier: pass
    class EnhancedDataLoader: pass
    # Define dummy nn structure correctly
    class _DummyNNModule: pass # Base dummy
    class _DummyMSELoss(_DummyNNModule): pass
    class nn: # Dummy nn module
        Module = _DummyNNModule
        MSELoss = _DummyMSELoss


try:
    from src.analysis.labeler import SemanticLabeler
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Analysis (labeler) module not available: {e}")
    ANALYSIS_AVAILABLE = False
    class SemanticLabeler: pass


# Setup root logger on startup (only if imports worked)
if 'setup_logger' in globals():
     setup_logger(console_level=logging.INFO)
logger = logging.getLogger(__name__) # Get logger for this main script


class SystemOrchestrator:
    """Handles the main CLI menu and orchestrates subsystem calls."""

    def __init__(self):
        log_statement(loglevel=str("info"), logstatement=str("Initializing System Orchestrator..."), main_logger=str(__name__))
        # Configuration is accessed via the imported config module, e.g., config.TrainingConfig

    def show_main_menu(self):
        """Displays the main menu and handles user input."""
        while True:
            print("\n--- Neural System Orchestrator ---")
            print("1. Data Processing & Tokenization")
            print("2. Synthetic Data Generation")
            print("3. Model Training")
            print("4. Semantic Labeling (Example)")
            # Removed Core Processing (integrated into Data), Model Ops (use specific actions),
            # Neuro Comms (undefined), Manifold Analysis (undefined)
            print("5. System Utilities")
            print("6. View System Logs")
            print("7. Exit System")
            print("---------------------------------")

            choice = input("Enter selection: ")

            try:
                if choice == '1':
                    self.run_data_pipeline()
                elif choice == '2':
                    self.run_synthetic_data_generation()
                elif choice == '3':
                    self.run_training()
                elif choice == '4':
                    self.run_labeling_example()
                elif choice == '5':
                    self.system_utilities()
                elif choice == '6':
                    self.view_system_logs()
                elif choice == '7':
                    log_statement(loglevel=str("info"), logstatement=str("System shutdown requested."), main_logger=str(__name__))
                    print("Exiting Neural System Orchestrator.")
                    sys.exit(0)
                else:
                    print("Invalid selection. Please try again.")
            except Exception as e:
                 # Use LOG_FILE variable which might be the fallback
                 log_file_path = LOG_FILE if 'LOG_FILE' in globals() else 'application.log'
                 logger.error(f"Error during menu action '{choice}': {e}", exc_info=True)
                 print(f"An error occurred. Check logs at {log_file_path}.")


    def run_data_pipeline(self):
        """Runs the data processing and tokenization pipeline."""
        if not DATA_PROCESSING_AVAILABLE:
            print("Data processing modules are not available.")
            log_statement(loglevel=str("warning"), logstatement=str("Attempted to run data pipeline, but modules are missing."), main_logger=str(__name__))
            return

        print("\n--- Data Processing Pipeline ---")
        try:
            processor = DataProcessor()
            print("Running data processing step (checking for 'discovered', 'error' files)...")
            log_statement(loglevel=str("info"), logstatement=str("Starting DataProcessor.process_all()"), main_logger=str(__name__))
            processor.process_all()
            log_statement(loglevel=str("info"), logstatement=str("DataProcessor.process_all() finished."), main_logger=str(__name__))
            print("Data processing step finished.")

            tokenizer = Tokenizer()
            print("Running tokenization step (checking for 'processed' files)...")
            log_statement(loglevel=str("info"), logstatement=str("Starting Tokenizer.tokenize_all()"), main_logger=str(__name__))
            tokenizer.tokenize_all()
            log_statement(loglevel=str("info"), logstatement=str("Tokenizer.tokenize_all() finished."), main_logger=str(__name__))
            print("Tokenization step finished.")
            print("------------------------------")

        except Exception as e:
             logger.error(f"Error during data pipeline: {e}", exc_info=True)
             print(f"Data pipeline failed. Check logs.")

    def run_synthetic_data_generation(self):
        """Runs the synthetic data generation process."""
        if not SYNTHETIC_DATA_AVAILABLE:
            print("Synthetic data module is not available.")
            log_statement(loglevel=str("warning"), logstatement=str("Attempted to run synthetic data generation, but module is missing."), main_logger=str(__name__))
            return

        print("\n--- Synthetic Data Generation ---")
        # Access config value safely
        target_samples = getattr(config.SyntheticDataConfig, 'TARGET_SAMPLES', 0)
        confirm = input(f"This will generate ~{target_samples} samples using the API. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Synthetic data generation cancelled.")
            return

        try:
            generator = SyntheticDataGenerator()
            print("Starting synthetic data generation...")
            log_statement(loglevel=str("info"), logstatement=str("Starting SyntheticDataGenerator.generate_dataset()"), main_logger=str(__name__))
            generator.generate_dataset()
            log_statement(loglevel=str("info"), logstatement=str("SyntheticDataGenerator.generate_dataset() finished."), main_logger=str(__name__))
            print("Synthetic data generation finished.")
            print("-------------------------------")
        except Exception as e:
            logger.error(f"Error during synthetic data generation: {e}", exc_info=True)
            print(f"Synthetic data generation failed. Check logs.")


    def run_training(self):
        """Initializes and runs the model training process."""
        if not TRAINING_AVAILABLE:
            print("Training modules are not available.")
            log_statement(loglevel=str("warning"), logstatement=str("Attempted to run training, but modules are missing."), main_logger=str(__name__))
            return

        print("\n--- Model Training ---")
        try:
            # --- Configuration (Example - Adapt as needed) ---
            DEVICE = config.DEFAULT_DEVICE
            # These should match the placeholder model/data
            INPUT_DIM = 128
            NUM_CLASSES = 6

            # --- Instantiate Components ---
            print(f"Initializing model ({ZoneClassifier.__name__}) on device {DEVICE}...")
            model = ZoneClassifier(input_features=INPUT_DIM, num_classes=NUM_CLASSES, device=DEVICE)

            print("Initializing data loader...")
            # Ensure data loader points to the correct tokenized data
            tokenized_dir = getattr(config, 'TOKENIZED_DATA_DIR', './data/tokenized')
            data_loader = EnhancedDataLoader(device=DEVICE, data_dir=tokenized_dir)

            # Choose appropriate loss function based on model output/task
            # criterion = nn.CrossEntropyLoss() # For classification logits
            criterion = nn.MSELoss() # Example from original trainer placeholder
            print(f"Using loss function: {type(criterion).__name__}")


            # --- Initialize Trainer ---
            trainer = EnhancedTrainer(model=model, data_loader=data_loader, criterion=criterion, device=DEVICE)

            # --- Optional: Load Checkpoint ---
            load_choice = input("Load checkpoint before training? (Enter filename or leave blank): ").strip()
            if load_choice:
                 if trainer.load_checkpoint(load_choice):
                      print(f"Checkpoint '{load_choice}' loaded successfully.")
                 else:
                      print(f"Failed to load checkpoint '{load_choice}'. Starting fresh training.")


            # --- Start Training ---
            print("Starting training loop...")
            log_statement(loglevel=str("info"), logstatement=str("Starting EnhancedTrainer.train()"), main_logger=str(__name__))
            trainer.train() # Runs the full training loop
            log_statement(loglevel=str("info"), logstatement=str("EnhancedTrainer.train() finished."), main_logger=str(__name__))
            print("Training finished.")
            print("--------------------")

        except Exception as e:
            logger.error(f"Error during training setup or execution: {e}", exc_info=True)
            print(f"Training failed. Check logs.")


    def run_labeling_example(self):
        """Runs a simple example of the semantic labeler."""
        if not ANALYSIS_AVAILABLE:
            print("Analysis (labeler) module is not available.")
            log_statement(loglevel=str("warning"), logstatement=str("Attempted to run labeling example, but module is missing."), main_logger=str(__name__))
            return

        print("\n--- Semantic Labeling Example ---")
        try:
            labeler = SemanticLabeler()
            # Example: Generate a dummy embedding (size should match model output, e.g., 768 for BERT base)
            # Adjust size based on the actual model used in LabelerConfig
            embedding_size = 768 # Example for bert-base-uncased
            dummy_embedding = torch.randn(embedding_size, device=config.DEFAULT_DEVICE)
            print(f"Generating label for a dummy embedding (size: {embedding_size})...")
            label = labeler.generate_label(dummy_embedding)
            print(f"Generated label: {label}")

            # Example for recursive (though recursion logic is basic in placeholder)
            # dummy_embeddings = [torch.randn(embedding_size, device=config.DEFAULT_DEVICE) for _ in range(3)]
            # print("\nRunning recursive labeling example on 3 dummy embeddings...")
            # labels = labeler.recursive_labeling(dummy_embeddings)
            # print(f"Generated labels: {labels}")
            print("-------------------------------")

        except Exception as e:
            logger.error(f"Error during labeling example: {e}", exc_info=True)
            print(f"Labeling example failed. Check logs.")


    def system_utilities(self):
        """Displays the system utilities submenu."""
        while True:
            print("\n--- System Utilities ---")
            print("1. View Configuration")
            print("2. Randomize Some Parameters (Example)")
            # Diagnostics removed - run tests separately using unittest/pytest
            print("3. Return to Main Menu")
            print("------------------------")

            choice = input("Select utility: ")

            if choice == '1':
                self.show_config()
            elif choice == '2':
                self.randomize_parameters_example()
            elif choice == '3':
                break
            else:
                print("Invalid selection.")

    def show_config(self):
        """Displays the current configuration loaded from src.utils.config."""
        print("\n--- Current Configuration ---")
        # Iterate through config classes defined in src.utils.config
        config_classes = {
            "Data Processing": getattr(config, 'DataProcessingConfig', None),
            "Data Loader": getattr(config, 'DataLoaderConfig', None),
            "Synthetic Data": getattr(config, 'SyntheticDataConfig', None),
            "Training": getattr(config, 'TrainingConfig', None),
            "Neural Zone": getattr(config, 'ZoneConfig', None),
            "Labeler": getattr(config, 'LabelerConfig', None),
            "Core/Attention": getattr(config, 'CoreConfig', None), # Example
            "Testing": getattr(config, 'TestConfig', None),
            "Paths & General": None # For top-level constants
        }
        for name, cfg_class in config_classes.items():
            print(f"\n# {name}:")
            if cfg_class:
                # Print attributes of the class instance
                try:
                    # Instantiate to get defaults if needed, handle potential errors
                    cfg_instance = cfg_class()
                    # Use dir() and getattr() for safer attribute access
                    for attr in dir(cfg_instance):
                         # Avoid internal attributes and methods
                         if not attr.startswith('_') and not callable(getattr(cfg_instance, attr)):
                              value = getattr(cfg_instance, attr)
                              print(f"  {attr}: {value}")
                except Exception as e:
                     print(f"  Could not display config for {name}: {e}")
            else:
                 # Print top-level constants from config module
                 for attr in dir(config):
                      if not attr.startswith('_') and attr.isupper(): # Convention for constants
                           value = getattr(config, attr)
                           # Avoid printing modules or classes again
                           if not isinstance(value, type) and not isinstance(value, type(config)):
                                print(f"  {attr}: {value}")
        print("---------------------------")


    def randomize_parameters_example(self):
        """
        Example of modifying parameters (Not recommended for production).
        This demonstrates interaction but doesn't permanently change config.py.
        """
        print("\n--- Parameter Randomization Example ---")
        print("NOTE: This only affects runtime values, not the config file.")
        # Example modifications (won't persist unless config module is designed to be mutable)
        try:
            # These changes won't actually modify the imported config values in other modules
            # unless the config module itself is designed for mutable state (generally not ideal).
            new_lr = 10**random.uniform(-5, -3)
            new_perplexity = random.randint(3, 10) # Example, not used currently
            print(f"Generated new LR (example): {new_lr:.6f}")
            print(f"Generated new Perplexity (example): {new_perplexity}")
            # To make this useful, subsequent calls (like starting training) would need
            # to explicitly use these randomized values instead of reading from config again.
            print("Apply these values manually when starting relevant processes if desired.")
        except Exception as e:
             logger.error(f"Error during parameter randomization example: {e}", exc_info=True)
             print("Failed to randomize parameters.")
        print("-------------------------------------")


    def view_system_logs(self):
        """Displays the last N lines of the main log file."""
        log_file_path = LOG_FILE if 'LOG_FILE' in globals() else 'application.log'
        print(f"\n--- System Logs (Last 100 lines of {log_file_path}) ---")
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                # Read last N lines efficiently (deque might be better for large files)
                lines = f.readlines()
                for line in lines[-100:]:
                    print(line.strip())
        except FileNotFoundError:
            print(f"Log file not found: {log_file_path}")
        except Exception as e:
             logger.error(f"Error reading log file {log_file_path}: {e}", exc_info=True)
             print("Could not read log file.")
        print("------------------------------------------")


if __name__ == "__main__":
    # Ensure Path is available if fallback config was used
    from pathlib import Path
    log_statement(loglevel=str("info"), logstatement=str("--- System Orchestrator Started ---"), main_logger=str(__name__))
    orchestrator = SystemOrchestrator()
    orchestrator.show_main_menu()
    log_statement(loglevel=str("info"), logstatement=str("--- System Orchestrator Exited ---"), main_logger=str(__name__))

