# main.py
"""
Main Orchestrator Script

Provides a command-line interface to interact with different subsystems
of the neural processing project (data processing, training, analysis, etc.).
Handles fallback to CPU libraries indirectly by relying on imported modules
that contain the necessary checks and alternative implementations.
"""

# --- Add project root to sys.path ---
# This MUST happen before importing any modules from 'src'
# Allows imports like `from src.utils...` when running `python main.py` from project root
import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ------------------------------------

import logging
import random
import os
import torch # Keep for potential direct use or type hinting

# Setup logger *after* potentially modifying sys.path but *before* importing other modules
# Assuming the modular structure with 'src'
try:
    # utils.logger should handle setting up file handlers etc.
    from src.utils.logger import setup_logger # Import LOG_FILE if defined there
    # Configure logging system wide (e.g., basicConfig or file handlers)
    # setup_logger() should be called *once* ideally
    setup_logger()
    logger = logging.getLogger(__name__) # Get logger for main script AFTER setup
    logger.info(f"Adding project root to sys.path for imports: {project_root}")
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"sys.path after modification: {sys.path}")
    # Import the whole config module for easy access to various config classes
    from src.utils import config
except ImportError as e:
    # Fallback if src structure isn't found or logger setup fails
    print(f"[CRITICAL] Failed to import from src or setup logger. Ensure 'src' directory exists and script is run from the project root. Error: {e}", file=sys.stderr)
    # Basic logging if setup failed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.critical("Failed to initialize core components from src. Limited functionality.")
    # Define fallbacks or exit if core components are essential
    LOG_FILE = Path("main_fallback.log") # type: ignore
    # Define dummy config class structure if actual config failed to load
    class DummyConfig:
        class DataProcessingConfig: REPO_FILE = "fallback_repo.csv.zst"; MAX_WORKERS = 4; PROCESSING_CHUNK_SIZE=8192
        class SyntheticDataConfig: TARGET_SAMPLES = 0; BATCH_SIZE=10; MAX_WORKERS=4; DATA_FORMAT='jsonl'; EMBED_MODEL_NAME='fallback_model'; OLLAMA_ENDPOINT='http://localhost:11434/api/generate'
        class TrainingConfig: pass # Add dummy attributes if needed by main.py directly
        class ZoneConfig: pass
        class LabelerConfig: EMBEDDING_SIZE = 768 # Example needed by labeling example
        class CoreConfig: pass
        class TestConfig: TEST_ARTIFACT_DIR = Path('./tests/artifacts')
        DEFAULT_DEVICE = 'cpu' # Sensible default fallback device
        TOKENIZED_DATA_DIR = Path('./data/tokenized')
        PROCESSED_DATA_DIR = Path('./data/processed')
        CHECKPOINT_DIR = Path('./checkpoints')
        LOG_DIR = Path('./logs')
        PROJECT_ROOT = project_root
        SYNTHETIC_DATA_DIR = Path('./data/synthetic')
    config = DummyConfig() # Assign dummy config to the 'config' variable

# --- Ensure required directories exist ---
# Use the imported (or fallback) config object now
try:
    LOG_DIR = config.LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = config.CHECKPOINT_DIR
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_ARTIFACT_DIR = getattr(config.TestConfig, 'TEST_ARTIFACT_DIR', Path('./tests/artifacts'))
    TEST_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure data directories exist (modules might also do this, but doesn't hurt)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception as dir_e:
    logger.error(f"Failed to create necessary directories defined in config: {dir_e}", exc_info=True)
    # Decide if execution should continue; maybe critical if logs can't be written

# --- Import Components ---
# Moved imports here, after path/logging/config setup

# Import project utilities first (might be used by other components)
try:
    from src.utils.helpers import save_state, load_state, dummy_input
    # gpu_switch contains the logic to check for GPUs and capability
    from src.utils.gpu_switch import get_compute_backend, check_gpu_support, get_pycuda_compute_capability
    UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core utilities not fully available: {e}")
    UTILS_AVAILABLE = False
    # Define dummy functions if essential for basic operation
    def dummy_input(*args, **kwargs): return None
    def check_gpu_support(): return []
    def get_compute_backend(): return "N/A"
    def get_pycuda_compute_capability(dev): return "N/A"


# Importing components based on availability
# These modules are expected to handle their own internal GPU/CPU fallbacks
try:
    from src.data.processing import DataProcessor, Tokenizer
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data processing modules not fully available: {e}")
    DATA_PROCESSING_AVAILABLE = False
    # Dummy classes allow the menu to run but operations will fail gracefully
    class DataProcessor:
        def __init__(self, *args, **kwargs): pass
        def process_all(self, *args, **kwargs): raise NotImplementedError(f"DataProcessor module not loaded due to import error: {e}")
    class Tokenizer:
        def __init__(self, *args, **kwargs): pass
        def tokenize_all(self, *args, **kwargs): raise NotImplementedError(f"Tokenizer module not loaded due to import error: {e}")


try:
    from src.data.synthetic import SyntheticDataGenerator
    SYNTHETIC_DATA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Synthetic data module not available: {e}")
    SYNTHETIC_DATA_AVAILABLE = False
    class SyntheticDataGenerator:
        def __init__(self, *args, **kwargs): pass
        def generate_dataset(self, *args, **kwargs): raise NotImplementedError(f"SyntheticDataGenerator module not loaded due to import error: {e}")


try:
    # Training components might also have GPU dependencies internally
    from src.training.trainer import EnhancedTrainer
    from src.core.models import ZoneClassifier # Example model, should be adaptable
    from src.data.loaders import EnhancedDataLoader # Example loader, check if device-aware
    import torch.nn as nn # For loss function example
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Training modules not fully available: {e}")
    TRAINING_AVAILABLE = False
    # Define dummy classes correctly, each on its own line
    class EnhancedTrainer:
        def __init__(self, *args, **kwargs): pass
        def train(self, *args, **kwargs): raise NotImplementedError(f"EnhancedTrainer module not loaded due to import error: {e}")
        def load_checkpoint(self, fn, *args, **kwargs): raise NotImplementedError(f"EnhancedTrainer module not loaded due to import error: {e}")
    class ZoneClassifier: # Dummy model
         def __init__(self, *args, **kwargs): pass
         def to(self, *args, **kwargs): return self # Basic compatibility
    class EnhancedDataLoader: # Dummy loader
         def __init__(self, *args, **kwargs): pass
    # Define dummy nn structure correctly if nn import itself failed (unlikely if torch installed)
    if 'nn' not in locals():
        class _DummyNNModule: pass
        class _DummyMSELoss(_DummyNNModule): pass
        class _DummyCrossEntropyLoss(_DummyNNModule): pass
        class nn: # Dummy nn module
            Module = _DummyNNModule
            MSELoss = _DummyMSELoss
            CrossEntropyLoss = _DummyCrossEntropyLoss


try:
    # Analysis components might use embeddings generated on GPU/CPU
    from src.analysis.labeler import SemanticLabeler
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analysis (labeler) module not available: {e}")
    ANALYSIS_AVAILABLE = False
    class SemanticLabeler:
        def __init__(self, *args, **kwargs): pass
        def generate_label(self, emb, *args, **kwargs): raise NotImplementedError(f"SemanticLabeler module not loaded due to import error: {e}")


# --- Main Orchestrator Class ---
class SystemOrchestrator:
    """Handles the main CLI menu and orchestrates subsystem calls."""

    def __init__(self):
        logger.info("Initializing System Orchestrator...")
        # Config accessed via imported module: config.ClassName.ATTRIBUTE
        # Determine the effective device based on config and actual availability
        self.effective_device = self._determine_effective_device()
        logger.info(f"Effective device set for orchestrator: {self.effective_device}")

    def _determine_effective_device(self):
        """Checks config preference and GPU availability to set the device."""
        # Get the preferred device from config
        preferred_device_config = getattr(config, 'DEFAULT_DEVICE', 'cpu') # Get value from config

        # --- MODIFICATION START ---
        # Check if the preferred device is CUDA, handling both string and torch.device object
        is_cuda_preferred = False
        if isinstance(preferred_device_config, str):
            # If it's a string, use startswith
            is_cuda_preferred = preferred_device_config.startswith('cuda')
        elif isinstance(preferred_device_config, torch.device):
            # If it's a torch.device object, check its type attribute
            is_cuda_preferred = preferred_device_config.type == 'cuda'
        # --- MODIFICATION END ---

        # Use the boolean flag `is_cuda_preferred` for the check now
        if is_cuda_preferred:
            # Check actual GPU availability using your existing logic
            if UTILS_AVAILABLE:
                try:
                    # Assuming check_gpu_support returns a list or similar boolean indicator
                    gpus_supported = check_gpu_support()
                    if gpus_supported:
                        # Check if PyTorch can actually use the CUDA device
                        if torch.cuda.is_available():
                            logger.info(f"GPU support verified. Config prefers CUDA. Using 'cuda'.")
                            # Return the standard string 'cuda' for consistency downstream
                            return 'cuda'
                        else:
                            logger.warning(f"Config prefers CUDA and compatible GPUs found, but torch.cuda.is_available() is False. Falling back to 'cpu'.")
                            return 'cpu'
                    else:
                        logger.warning(f"Config prefers CUDA, but no compatible GPUs found by check_gpu_support. Falling back to 'cpu'.")
                        return 'cpu'
                except Exception as e:
                    logger.error(f"Error during GPU check for device determination: {e}. Falling back to 'cpu'.")
                    return 'cpu'
            else:
                logger.warning("GPU utilities (gpu_switch) not available. Cannot verify GPU support. Falling back to 'cpu'.")
                return 'cpu'
        else:
            # Config prefers 'cpu' or another non-GPU device
            logger.info(f"Config prefers non-GPU device. Using 'cpu'.")
            # Return the standard string 'cpu'
            return 'cpu'

    def show_main_menu(self):
        """Displays the main menu and handles user input."""
        while True:
            print("\n--- Neural System Orchestrator ---")
            print(f"1. Data Processing & Tokenization {'(Available)' if DATA_PROCESSING_AVAILABLE else '(Unavailable)'}")
            print(f"2. Synthetic Data Generation {'(Available)' if SYNTHETIC_DATA_AVAILABLE else '(Unavailable)'}")
            print(f"3. Model Training {'(Available)' if TRAINING_AVAILABLE else '(Unavailable)'}")
            print(f"4. Semantic Labeling (Example) {'(Available)' if ANALYSIS_AVAILABLE else '(Unavailable)'}")
            print("5. System Utilities")
            print("6. View System Logs")
            print("7. Exit System")
            print("---------------------------------")
            choice = input("Enter selection: ")
            try:
                # Wrap the entire menu action dispatch in a try block
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
                    logger.info("System shutdown requested.")
                    print("Exiting Neural System Orchestrator.")
                    sys.exit(0) # Clean exit
                else:
                    print("Invalid selection. Please try again.")

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                logger.warning("User interrupted execution (Ctrl+C). Exiting.")
                print("\nOperation cancelled by user. Exiting.")
                sys.exit(1) # Indicate non-zero exit code for interruption
            except NotImplementedError as ni_err:
                 # Catch errors from dummy modules if functionality is unavailable
                 logger.error(f"Functionality not implemented: {ni_err}", exc_info=False)
                 print(f"\n!! Error: This feature is unavailable. Reason: {ni_err} !!")
            except Exception as e:
                # Log the exception with traceback to all configured handlers
                # (console, main log, error log)
                logger.exception(f"Unhandled error during menu action '{choice}': {e}")
                # Inform the user
                log_file_path = LOG_FILE if 'LOG_FILE' in globals() else Path('./logs/app.log') # Use actual log file path or default
                print(f"\n!! An unexpected error occurred. Details logged to console and '{log_file_path}'. !!")
                # Optionally wait for user input before showing menu again
                # input("Press Enter to continue...")


    def run_data_pipeline(self):
        """Runs the data processing and tokenization pipeline."""
        if not DATA_PROCESSING_AVAILABLE:
            print("Data processing modules are not available.")
            logger.warning("Attempted to run data pipeline, but modules are missing.")
            return
        print("\n--- Data Processing Pipeline ---")
        try:
            # DataProcessor and Tokenizer handle their own GPU/CPU logic internally
            processor = DataProcessor()
            print("Running data processing step (checking for 'discovered', 'error' files)...")
            logger.info("Starting DataProcessor.process_all()")
            processor.process_all()
            logger.info("DataProcessor.process_all() finished.")
            print("Data processing step finished.")

            tokenizer = Tokenizer()
            print("Running tokenization step (checking for 'processed' files)...")
            logger.info("Starting Tokenizer.tokenize_all()")
            tokenizer.tokenize_all() # Tokenizer will use its configured device if needed
            logger.info("Tokenizer.tokenize_all() finished.")
            print("Tokenization step finished.")
            print("------------------------------")
        except Exception as e:
            # Log with traceback
            logger.exception(f"Error during data pipeline: {e}")
            print(f"Data pipeline failed. Check logs for details.")


    def run_synthetic_data_generation(self):
        """Runs the synthetic data generation process."""
        if not SYNTHETIC_DATA_AVAILABLE:
            print("Synthetic data module is not available.")
            logger.warning("Attempted to run synthetic data generation, but module is missing.")
            return
        print("\n--- Synthetic Data Generation ---")
        try:
            # Read necessary config values safely
            target_samples = getattr(config.SyntheticDataConfig, 'TARGET_SAMPLES', 0)
            api_endpoint = getattr(config.SyntheticDataConfig, 'OLLAMA_ENDPOINT', 'Not Configured')

            confirm = input(f"This will generate ~{target_samples} samples using the API ({api_endpoint}). Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Synthetic data generation cancelled.")
                return

            generator = SyntheticDataGenerator()
            print("Starting synthetic data generation...")
            logger.info("Starting SyntheticDataGenerator.generate_dataset()")
            generator.generate_dataset()
            logger.info("SyntheticDataGenerator.generate_dataset() finished.")
            print("Synthetic data generation finished.")
            print("-------------------------------")
        except Exception as e:
            logger.exception(f"Error during synthetic data generation: {e}")
            print(f"Synthetic data generation failed. Check logs for details.")


    def run_training(self):
        """Initializes and runs the model training process."""
        if not TRAINING_AVAILABLE:
            print("Training modules are not available.")
            logger.warning("Attempted to run training, but modules are missing.")
            return
        print("\n--- Model Training ---")
        try:
            # Use the effective device determined during orchestrator init
            DEVICE = self.effective_device
            logger.info(f"Proceeding with training on device: {DEVICE}")

            # --- Get model parameters from config or define defaults ---
            # Example: replace these with actual config access or robust defaults
            INPUT_DIM = getattr(config, 'MODEL_INPUT_DIM', 128) # Example attribute
            NUM_CLASSES = getattr(config, 'MODEL_NUM_CLASSES', 10) # Example attribute

            print(f"Initializing model ({getattr(ZoneClassifier, '__name__', 'N/A')}) on device {DEVICE}...")
            # Pass necessary dimensions from config
            # The ZoneClassifier itself might need a 'device' argument if it creates tensors internally
            # Or rely on the trainer moving the model to the device
            model = ZoneClassifier(input_features=INPUT_DIM, num_classes=NUM_CLASSES) # Device handled by trainer?

            print("Initializing data loader...")
            tokenized_dir = config.TOKENIZED_DATA_DIR
            # Ensure data_loader uses config values and the effective device if needed
            data_loader = EnhancedDataLoader(device=DEVICE, data_dir=tokenized_dir)

            # Define loss function - ensure it matches model output and task
            # Using CrossEntropyLoss as an example for classification
            criterion = nn.CrossEntropyLoss() # Instantiate the loss function
            print(f"Using loss function: {type(criterion).__name__}")

            # Pass the effective device to the trainer
            trainer = EnhancedTrainer(model=model, data_loader=data_loader, criterion=criterion, device=DEVICE)

            load_choice = input("Load checkpoint before training? (Enter filename or leave blank): ").strip()
            if load_choice:
                # Use CHECKPOINT_DIR from config
                checkpoint_path = config.CHECKPOINT_DIR / load_choice
                if trainer.load_checkpoint(checkpoint_path): # Pass the full path
                    print(f"Checkpoint '{checkpoint_path}' loaded successfully.")
                else:
                    print(f"Failed to load checkpoint '{checkpoint_path}'. Starting fresh training.")

            print("Starting training loop...")
            logger.info(f"Starting EnhancedTrainer.train() on device {DEVICE}")
            trainer.train() # Trainer handles moving data/model to its device
            logger.info("EnhancedTrainer.train() finished.")
            print("Training finished.")
            print("--------------------")
        except Exception as e:
            logger.exception(f"Error during training setup or execution: {e}")
            print(f"Training failed. Check logs for details.")


    def run_labeling_example(self):
        """Runs a simple example of the semantic labeler."""
        if not ANALYSIS_AVAILABLE:
            print("Analysis (labeler) module is not available.")
            logger.warning("Attempted to run labeling example, but module is missing.")
            return
        print("\n--- Semantic Labeling Example ---")
        try:
            DEVICE = self.effective_device # Use effective device
            # Labeler might need device if it uses models internally
            labeler = SemanticLabeler(device=DEVICE)

            # Example embedding size, should match expected model output
            embedding_size = getattr(config.LabelerConfig, 'EMBEDDING_SIZE', 768)
            # Create dummy embedding on the correct device
            dummy_embedding = torch.randn(embedding_size, device=DEVICE)

            print(f"Generating label for a dummy embedding (size: {embedding_size}) on device {DEVICE}...")
            label = labeler.generate_label(dummy_embedding)
            print(f"Generated label: {label}")
            print("-------------------------------")
        except Exception as e:
            logger.exception(f"Error during labeling example: {e}")
            print(f"Labeling example failed. Check logs for details.")


    def system_utilities(self):
        """Displays the system utilities submenu."""
        while True:
            print("\n--- System Utilities ---")
            print("1. View Configuration")
            print(f"2. Check GPU Support (Effective Device: {self.effective_device})")
            print("3. Return to Main Menu")
            print("------------------------")
            choice = input("Select utility: ")
            if choice == '1':
                self.show_config()
            elif choice == '2':
                self.check_gpu()
            elif choice == '3':
                break
            else:
                print("Invalid selection.")


    def show_config(self):
        """Displays the current configuration loaded from src.utils.config."""
        print("\n--- Current Configuration ---")
        # Define config classes to inspect (add others as needed)
        # Use getattr for safety in case config module structure changes
        config_classes = {
            "Data Processing": getattr(config, 'DataProcessingConfig', None),
            "Data Loader": getattr(config, 'DataLoaderConfig', None),
            "Synthetic Data": getattr(config, 'SyntheticDataConfig', None),
            "Training": getattr(config, 'TrainingConfig', None),
            "Neural Zone": getattr(config, 'ZoneConfig', None),
            "Labeler": getattr(config, 'LabelerConfig', None),
            "Core/Attention": getattr(config, 'CoreConfig', None),
            "Testing": getattr(config, 'TestConfig', None),
            "Paths & General": config # For top-level constants
        }

        for name, cfg_class_or_module in config_classes.items():
            print(f"\n# {name}:")
            if cfg_class_or_module:
                source_obj = None
                try:
                    # If it's a class, access attributes directly from the class object
                    # No need to instantiate if we only read class variables/constants
                    if isinstance(cfg_class_or_module, type):
                         source_obj = cfg_class_or_module
                    else: # It's the main config module
                         source_obj = cfg_class_or_module

                    # Iterate through attributes of the class or module
                    for attr in dir(source_obj):
                        # Filter out private/protected/callable attributes/methods
                        if not attr.startswith('_') and not callable(getattr(source_obj, attr)):
                             # For the main config module, typically show uppercase constants
                             if source_obj is config and not attr.isupper():
                                 continue
                             value = getattr(source_obj, attr)
                             print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  Could not display config for {name}: {e}")
            else:
                 print("  (Config class not found or loaded)")
        print(f"\n# Orchestrator Effective Device:")
        print(f"  EFFECTIVE_DEVICE: {self.effective_device}")
        print("---------------------------")


    def check_gpu(self):
        """Displays detailed GPU information if available."""
        print("\n--- GPU Check ---")
        print(f"Orchestrator Effective Device: {self.effective_device}")
        if UTILS_AVAILABLE:
            try:
                supported_gpus = check_gpu_support() # Checks compatibility based on compute capability
                if supported_gpus:
                    print(f"Compatible GPU(s) detected by gpu_switch: {supported_gpus}")
                    compute_backend = get_compute_backend()
                    print(f"Compute Backend detected by gpu_switch: {compute_backend}")
                    # Try getting capability via PyCUDA if backend is CUDA
                    if compute_backend == 'cuda':
                        try:
                            cap = get_pycuda_compute_capability(supported_gpus[0]) # Check first detected GPU
                            print(f"PyCUDA Compute Capability (GPU 0): {cap if cap else 'N/A'}")
                        except Exception as pc_e:
                            print(f"Could not get PyCUDA capability: {pc_e}")
                    # Add a check for torch CUDA availability
                    if torch.cuda.is_available():
                        print(f"torch.cuda.is_available(): True")
                        print(f"PyTorch CUDA devices found: {torch.cuda.device_count()}")
                        for i in range(torch.cuda.device_count()):
                             print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
                             print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
                    else:
                         print(f"torch.cuda.is_available(): False")
                else:
                    print("No compatible GPUs found by gpu_switch (based on min compute capability check).")
                    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            except Exception as gpu_e:
                logger.error(f"Error during GPU check utility: {gpu_e}", exc_info=True)
                print(f"Error checking GPU details: {gpu_e}")
        else:
            print("GPU utilities (gpu_switch) are unavailable (import failed).")
        print("-----------------")


    def view_system_logs(self):
        """Displays the last N lines of the main and error log files."""
        # Determine log file paths safely using getattr on config
        log_dir_path = getattr(config, 'LOG_DIR', Path('./logs')) # Default if LOG_DIR missing
        app_log_name = getattr(config, 'LOG_FILE_APP', 'app.log') # Default name if not in config
        err_log_name = getattr(config, 'LOG_FILE_ERR', 'errors.log') # Default name

        # Construct full paths
        app_log_path = log_dir_path / app_log_name
        err_log_path = log_dir_path / err_log_name

        log_files = {
            "Application Log": app_log_path,
            "Error Log": err_log_path
        }
        lines_to_show = 100 # Number of lines to tail

        for name, log_file_path in log_files.items():
            print(f"\n--- {name} (Last {lines_to_show} lines of {log_file_path}) ---")
            if not log_file_path.exists():
                 print(f"Log file not found: {log_file_path}")
                 continue # Skip to next log file
            try:
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # More efficient tailing if files are large, but readlines is simpler for moderate logs
                    lines = f.readlines()
                for line in lines[-lines_to_show:]:
                    print(line.strip()) # Print stripped line
            except Exception as e:
                logger.exception(f"Error reading log file {log_file_path}: {e}")
                print(f"Could not read log file: {e}")
            print("-" * (len(name) + 18)) # Separator matching the header length


def main():
    """Entry point for the orchestrator script."""
    # Logging and path setup happens at the top level module import phase
    logger.info("--- System Orchestrator Main Function ---")
    try:
        orchestrator = SystemOrchestrator()
        orchestrator.show_main_menu()
    except Exception as main_err:
        # Catch any unexpected critical error during orchestrator init or main loop
        logger.critical(f"Critical error in main orchestrator execution: {main_err}", exc_info=True)
        print(f"\n!!! A critical error occurred: {main_err}. Check logs for details. Exiting. !!!")
        sys.exit(1) # Exit with error code
    finally:
        # This will run even if sys.exit() was called earlier
        logger.info("--- System Orchestrator Exiting ---")


if __name__ == "__main__":
    # This block executes when the script is run directly
    main()