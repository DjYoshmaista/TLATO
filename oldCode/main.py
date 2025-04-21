#!/home/yosh/miniconda3/conda/bin/env python3
# -*- coding: utf-8 -*-

"""
TLATO - Transformer Latent-Attention Auto Train Once
main.py

Entry point for the TLATO system orchestrator. Initializes modules,
handles user interaction, and coordinates different system components like
data processing, model training, and analysis.
"""

import os
import sys
import logging
import importlib
import traceback
from typing import Dict, Tuple, Optional, Any # Added Optional and Tuple
from pathlib import Path
from src.utils.logger import configure_logging, log_statement
configure_logging()

core_utils_available = False
data_processing_available = False
synthetic_data_available = False
training_available = False
analysis_available = False
logger = None
load_config = None
get_device = None
helpers = None
constants = None
data_readers = None
data_loaders = None
models = None
attention = None
zones = None
trainer = None
labeler = None

# --- Dynamic Path Addition ---
# Add the project root to sys.path to allow for absolute imports from 'src'
# This needs to happen *before* importing src.utils.logger
try:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve() # Assumes src/main.py
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"[INFO] Project Root added to sys.path: {PROJECT_ROOT}")
except Exception as path_e:
    print(f"[CRITICAL] Failed to set project root path: {path_e}", file=sys.stderr)
    sys.exit(1)

# Configure logging early
# Note: logger configuration might be better placed within src.utils.logger
# but done here to capture early path issues if any.
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
APP_LOG_FILE = os.path.join(LOG_DIR, 'app.log')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'errors.log')

log_statement(loglevel=str("info"), logstatement=str(f"Adding project root to sys.path for imports: {PROJECT_ROOT}"), main_logger=str(__name__))
sys.path.insert(0, PROJECT_ROOT)
log_statement(loglevel=str("info"), logstatement=str(f"Current working directory: {os.getcwd()}"), main_logger=str(__name__))
log_statement(loglevel=str("info"), logstatement=str(f"sys.path after modification: {sys.path}"), main_logger=str(__name__))

# --- Module Availability Check ---
# Attempt to import core components and set availability flags.
# Store error messages for unavailable modules.
# Using Tuple[bool, Optional[str]] to store status and error message.
module_availability: Dict[str, Tuple[bool, Optional[str]]] = {
    "core_utils": (False, None),
    "data_processing": (False, None),
    "synthetic_data": (False, None),
    "model_training": (False, None),
    "analysis": (False, None),
    "gpu_utils": (False, None),
}

def check_module_availability():
    """Checks availability of major system modules and logs issues."""
    global module_availability
    log_statement(loglevel=str("info"), logstatement=str("Checking module availability..."), main_logger=str(__name__))

    # Core Utilities (e.g., helpers, config - assuming essential)
    try:
        importlib.import_module('src.utils.helpers')
        importlib.import_module('src.utils.config')
        # Example: Check for a specific dependency if needed
        # importlib.import_module('pandas') # Example check
        module_availability["core_utils"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("Core utilities available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"Core utilities not fully available: {e}"
        module_availability["core_utils"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))
        # Decide if this is fatal? For now, assume non-fatal.

    # GPU Utilities (Optional)
    try:
        importlib.import_module('src.utils.gpu_switch')
        # Check for torch availability as gpu_switch likely depends on it
        importlib.import_module('torch')
        module_availability["gpu_utils"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("GPU utilities (gpu_switch) available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"GPU utilities (gpu_switch) not available: {e}. GPU support check might fail."
        module_availability["gpu_utils"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))

    # Data Processing
    try:
        importlib.import_module('src.data.processing')
        importlib.import_module('src.data.loaders')
        importlib.import_module('src.data.readers')
        # Add checks for specific heavy dependencies if needed
        importlib.import_module('pandas')
        importlib.import_module('numpy')
        importlib.import_module('transformers')
        module_availability["data_processing"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("Data processing modules available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"Data processing modules not fully available: {e}"
        module_availability["data_processing"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))

    # Synthetic Data Generation
    try:
        importlib.import_module('src.data.synthetic')
        # Add checks for dependencies like 'transformers', 'torch' if needed
        importlib.import_module('torch')
        importlib.import_module('transformers')
        module_availability["synthetic_data"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("Synthetic data generation modules available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"Synthetic data generation modules not fully available: {e}"
        module_availability["synthetic_data"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))


    # Model Training
    try:
        importlib.import_module('src.training.trainer')
        importlib.import_module('src.training.trainer.EnhancedTrainer')
        importlib.import_module('src.training.trainer.TrainingMetrics')
        importlib.import_module('src.training.trainer')
        importlib.import_module('src.core.models')
        # Add checks for heavy dependencies
        importlib.import_module('torch')
        importlib.import_module('transformers')
        importlib.import_module('pandas')
        importlib.import_module('sklearn')
        module_availability["model_training"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("Training modules available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"Training modules not fully available: {e}"
        module_availability["model_training"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))

    # Analysis (Labeler)
    try:
        importlib.import_module('src.analysis.labeler')
        # Add checks for dependencies
        importlib.import_module('torch')
        importlib.import_module('transformers')
        importlib.import_module('pandas') # Example dependency
        module_availability["analysis"] = (True, None)
        log_statement(loglevel=str("info"), logstatement=str("Analysis (labeler) module available."), main_logger=str(__name__))
    except ImportError as e:
        error_msg = f"Analysis (labeler) module not available: {e}"
        module_availability["analysis"] = (False, str(e)) # Store error message
        log_statement(loglevel=str("warning"), logstatement=str(error_msg), main_logger=str(__name__))

# Run the check immediately after definition
check_module_availability()

# --- Delayed Imports ---
# Import modules only after checking their availability (or handle failure)

# Configure full logging using the utility function if core utils are available
if module_availability["core_utils"][0]:
    try:
        configure_logging(APP_LOG_FILE, ERROR_LOG_FILE)
        log_statement(loglevel=str("info"), logstatement=str("Full logging configured using src.utils.logger."), main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Failed to configure full logging: {e}", exc_info=True), main_logger=str(__name__))
        # Continue with basic config
else:
    log_statement(loglevel=str("warning"), logstatement=str("Core utilities not available, using basic logging configuration."), main_logger=str(__name__))


# Import other components conditionally or handle ImportErrors gracefully
try:
    from src.utils.config import load_config
    CONFIG = load_config() # Load configuration if available
    log_statement(loglevel=str("info"), logstatement=str("Configuration loaded."), main_logger=str(__name__))
except ImportError:
    log_statement(loglevel=str("error"), logstatement=str("Could not import or load config. Using default settings or exiting might be necessary."), main_logger=str(__name__))
    CONFIG = {} # Provide a default empty config or handle appropriately

try:
    from src.utils.helpers import display_splash_screen, clear_screen
except ImportError:
    log_statement(loglevel=str("warning"), logstatement=str("Helper utilities (splash, clear) not available."), main_logger=str(__name__))
    # Define dummy functions if needed
    def display_splash_screen(): pass
    def clear_screen(): pass

# --- System Orchestrator Class ---

class SystemOrchestrator:
    """
    Manages the main workflow and user interactions for the TLATO system.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator, checks module availability, and sets up state.
        """
        log_statement(loglevel=str("info"), logstatement=str("Initializing System Orchestrator..."), main_logger=str(__name__))
        self.config = config
        self.module_status = module_availability # Use the globally checked status

        # Determine effective device (CPU/GPU)
        self.device = self._get_effective_device()
        log_statement(loglevel=str("info"), logstatement=str(f"Effective device set for orchestrator: {self.device}"), main_logger=str(__name__))

        # Placeholder for loaded data or models
        self.data = None
        self.model = None

    def _get_effective_device(self) -> str:
        """Determines whether to use CPU or GPU based on availability and config."""
        preferred_device = self.config.get('system', {}).get('device', 'cpu')
        gpu_available, gpu_error = self.module_status["gpu_utils"]

        if preferred_device == 'cuda' and gpu_available:
            try:
                from src.utils.gpu_switch import set_device
                return set_device() # Let gpu_switch handle detailed checks
            except Exception as e:
                log_statement(loglevel=str("warning"), logstatement=str(f"GPU utility available but failed during device setting: {e}. Falling back to 'cpu'."), main_logger=str(__name__))
                return 'cpu'
        elif preferred_device == 'cuda' and not gpu_available:
            log_statement(loglevel=str("warning"), logstatement=str(f"GPU (cuda) preferred but utilities not available (Reason: {gpu_error}). Falling back to 'cpu'."), main_logger=str(__name__))
            return 'cpu'
        else:
            # Handles 'cpu' preference or fallback
            return 'cpu'

    def display_menu(self):
        """Displays the main menu options to the user."""
        clear_screen()
        display_splash_screen() # Display splash/logo if available
        log_statement(loglevel=str("info"), logstatement=str("--- Neural System Orchestrator ---"), main_logger=str(__name__))

        # Display options based on module availability
        log_statement(loglevel=str("info"), logstatement=str(f"1. Data Processing & Tokenization {'(Available)' if self.module_status['data_processing'][0] else '(Unavailable)'}"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"2. Synthetic Data Generation {'(Available)' if self.module_status['synthetic_data'][0] else '(Unavailable)'}"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"3. Model Training {'(Available)' if self.module_status['model_training'][0] else '(Unavailable)'}"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"4. Semantic Labeling (Example) {'(Available)' if self.module_status['analysis'][0] else '(Unavailable)'}"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str("--- System Utilities ---"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str("5. View System Logs"), main_logger=str(__name__))
        # Add more utilities like config management, etc.
        log_statement(loglevel=str("info"), logstatement=str("6. Exit System"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str("-" * 30), main_logger=str(__name__))

    def run_data_pipeline(self):
        """Handles the data processing and tokenization workflow."""
        available, error_msg = self.module_status['data_processing']
        if available:
            try:
                log_statement(loglevel=str("info"), logstatement=str("Attempting to run data processing pipeline..."), main_logger=str(__name__))
                # Lazy import: Import only when needed
                from src.data.processing import run_processing_workflow
                # Assuming run_processing_workflow takes config and device
                self.data = run_processing_workflow(self.config, self.device)
                log_statement(loglevel=str("info"), logstatement=str("Data processing pipeline completed successfully."), main_logger=str(__name__))
                log_statement(loglevel=str("info"), logstatement=str("Data processing finished."), main_logger=str(__name__))
                # Potentially show summary or save location
            except ImportError as e:
                # This might happen if a deeper dependency was missed in initial check
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to import data processing components during execution: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"Error: Could not load data processing components. Reason: {e}"), main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error during data processing pipeline: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"An error occurred during data processing: {e}"), main_logger=str(__name__))
        else:
            # Module was unavailable from the start
            reason = f"Reason: {error_msg}" if error_msg else "Reason: Check logs for import errors."
            log_statement(loglevel=str("error"), logstatement=str(f"Data processing modules are not available. {reason}"), main_logger=str(__name__))
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to run data pipeline, but modules are missing. {reason}"), main_logger=str(__name__))

    def run_synthetic_data_generation(self):
        """Handles the synthetic data generation workflow."""
        available, error_msg = self.module_status['synthetic_data']
        if available:
            try:
                log_statement(loglevel=str("info"), logstatement=str("Attempting to run synthetic data generation..."), main_logger=str(__name__))
                from src.data.synthetic import generate_synthetic_data # Example function
                # Adjust parameters as needed by the actual function
                generate_synthetic_data(self.config, output_dir=os.path.join(PROJECT_ROOT, 'data', 'synthetic_output'))
                log_statement(loglevel=str("info"), logstatement=str("Synthetic data generation completed."), main_logger=str(__name__))
                log_statement(loglevel=str("info"), logstatement=str("Synthetic data generation finished."), main_logger=str(__name__))
            except ImportError as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to import synthetic data components during execution: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"Error: Could not load synthetic data components. Reason: {e}"), main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error during synthetic data generation: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"An error occurred during synthetic data generation: {e}"), main_logger=str(__name__))
        else:
            reason = f"Reason: {error_msg}" if error_msg else "Reason: Check logs for import errors."
            log_statement(loglevel=str("error"), logstatement=str(f"Synthetic data generation modules are not available. {reason}"), main_logger=str(__name__))
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to run synthetic data generation, but modules are missing. {reason}"), main_logger=str(__name__))

    def run_training_pipeline(self):
        """Handles the model training workflow."""
        available, error_msg = self.module_status['model_training']
        if available:
            try:
                log_statement(loglevel=str("info"), logstatement=str("Attempting to run model training pipeline..."), main_logger=str(__name__))
                from src.training.trainer import TrainingPipeline # Example class
                # Requires processed data
                if self.data is None:
                    log_statement(loglevel=str("error"), logstatement=str("Error: Processed data is not loaded. Run data processing first."), main_logger=str(__name__))
                    log_statement(loglevel=str("warning"), logstatement=str("Training pipeline requires processed data, which is not available."), main_logger=str(__name__))
                    return

                pipeline = TrainingPipeline(self.config, self.device)
                self.model = pipeline.train(self.data) # Assuming train takes data
                log_statement(loglevel=str("info"), logstatement=str("Model training pipeline completed."), main_logger=str(__name__))
                log_statement(loglevel=str("info"), logstatement=str("Model training finished."), main_logger=str(__name__))
                # Potentially save model info or path
            except ImportError as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to import training components during execution: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"Error: Could not load training components. Reason: {e}"), main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error during training pipeline: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"An error occurred during model training: {e}"), main_logger=str(__name__))
        else:
            reason = f"Reason: {error_msg}" if error_msg else "Reason: Check logs for import errors."
            log_statement(loglevel=str("error"), logstatement=str(f"Model training modules are not available. {reason}"), main_logger=str(__name__))
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to run training pipeline, but modules are missing. {reason}"), main_logger=str(__name__))

    def run_analysis_pipeline(self):
        """Handles the semantic labeling/analysis workflow."""
        available, error_msg = self.module_status['analysis']
        if available:
            try:
                log_statement(loglevel=str("info"), logstatement=str("Attempting to run analysis pipeline..."), main_logger=str(__name__))
                from src.analysis.labeler import run_labeling # Example function
                # Requires a trained model and potentially data
                if self.model is None:
                     log_statement(loglevel=str("error"), logstatement=str("Error: Model not trained or loaded. Run training first."), main_logger=str(__name__))
                     log_statement(loglevel=str("warning"), logstatement=str("Analysis pipeline requires a model, which is not available."), main_logger=str(__name__))
                     return
                # Example: Run labeling on new data or specific input
                # results = run_labeling(self.model, self.config, self.device, input_text="...")
                log_statement(loglevel=str("info"), logstatement=str("Running semantic labeling (example)..."), main_logger=str(__name__)) # Placeholder action
                # Add actual call to labeler function here
                log_statement(loglevel=str("info"), logstatement=str("Analysis pipeline (labeling example) executed."), main_logger=str(__name__))
                log_statement(loglevel=str("info"), logstatement=str("Semantic labeling finished (example)."), main_logger=str(__name__))
            except ImportError as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to import analysis components during execution: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"Error: Could not load analysis components. Reason: {e}"), main_logger=str(__name__))
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Error during analysis pipeline: {e}", exc_info=True), main_logger=str(__name__))
                log_statement(loglevel=str("error"), logstatement=str(f"An error occurred during analysis: {e}"), main_logger=str(__name__))
        else:
            reason = f"Reason: {error_msg}" if error_msg else "Reason: Check logs for import errors."
            log_statement(loglevel=str("error"), logstatement=str(f"Analysis (labeler) module is not available. {reason}"), main_logger=str(__name__))
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to run analysis pipeline, but module is missing. {reason}"), main_logger=str(__name__))

    def view_logs(self):
        """Displays the content of the log files."""
        log_statement(loglevel=str("info"), logstatement=str("User requested to view logs."), main_logger=str(__name__))
        print("\n--- Application Log (app.log) ---")
        try:
            with open(APP_LOG_FILE, 'r') as f:
                # Print last N lines? Or allow scrolling? For now, print tail.
                lines = f.readlines()
                for line in lines[-20:]: # Show last 20 lines
                    print(line.strip())
        except FileNotFoundError:
            print(f"App log file not found at {APP_LOG_FILE}")
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to view app log, but file not found: {APP_LOG_FILE}"), main_logger=str(__name__))
        except Exception as e:
            print(f"Error reading app log: {e}")
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading app log: {e}", exc_info=True), main_logger=str(__name__))

        print("\n--- Error Log (errors.log) ---")
        try:
            with open(ERROR_LOG_FILE, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print("No errors logged.")
                else:
                    for line in lines[-20:]: # Show last 20 lines
                        print(line.strip())
        except FileNotFoundError:
            print(f"Error log file not found at {ERROR_LOG_FILE}")
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to view error log, but file not found: {ERROR_LOG_FILE}"), main_logger=str(__name__))
        except Exception as e:
            print(f"Error reading error log: {e}")
            log_statement(loglevel=str("error"), logstatement=str(f"Error reading error log: {e}", exc_info=True), main_logger=str(__name__))

        input("\nPress Enter to return to the main menu...")

    def run(self):
        """Main loop for the orchestrator."""
        while True:
            self.display_menu()
            choice = input("Enter selection: ").strip()

            if choice == '1':
                self.run_data_pipeline()
            elif choice == '2':
                self.run_synthetic_data_generation()
            elif choice == '3':
                self.run_training_pipeline()
            elif choice == '4':
                self.run_analysis_pipeline()
            elif choice == '5':
                self.view_logs()
                continue # Skip wait prompt for logs, handled in view_logs
            elif choice == '6':
                log_statement(loglevel=str("info"), logstatement=str("User initiated system exit."), main_logger=str(__name__))
                log_statement(loglevel=str("info"), logstatement=str("Exiting System Orchestrator. Goodbye!"), main_logger=str(__name__))
                break
            else:
                log_statement(loglevel=str("warning"), logstatement=str("Invalid selection. Please try again."), main_logger=str(__name__))
                log_statement(loglevel=str("warning"), logstatement=str(f"Invalid user input received: {choice}"), main_logger=str(__name__))

            # Pause after action (except for exit/log view)
            if choice not in ['5', '6']:
                input("\nPress Enter to continue...")

# --- Main Execution ---
def main():
    """
    Main function to set up and run the System Orchestrator.
    """
    log_statement(loglevel=str("info"), logstatement=str("--- System Orchestrator Main Function ---"), main_logger=str(__name__))

    # Ensure core utilities are minimally available to proceed
    if not module_availability["core_utils"][0]:
         # Use basic print because full logger might rely on core utils
        log_statement(loglevel=str("critical"), logstatement=str(f"CRITICAL ERROR: Core utilities failed to load. Reason: {module_availability['core_utils'][1]}. Cannot start Orchestrator."), main_logger=str(__name__))
        # Log using basic config if possible
        log_statement(loglevel=str("critical"), logstatement=str(f"Core utilities failed to load: {module_availability['core_utils'][1]}. Aborting."), main_logger=str(__name__))
        sys.exit(1) # Exit if core components are missing

    # Load configuration (assuming config loading is part of core utils or handled above)
    global CONFIG
    if not CONFIG:
        log_statement(loglevel=str("warning"), logstatement=str("Configuration could not be loaded. Proceeding with defaults/limited functionality."), main_logger=str(__name__))
        # Define essential defaults if needed
        CONFIG.setdefault('system', {}).setdefault('device', 'cpu')

    try:
        orchestrator = SystemOrchestrator(config=CONFIG)
        orchestrator.run()
    except Exception as e:
        # Catch unexpected errors during orchestrator setup or run
        log_statement(loglevel=str("critical"), logstatement=str(f"An unhandled exception occurred in the orchestrator: {e}", exc_info=True), main_logger=str(__name__))
        log_statement(loglevel=str("error"), logstatement=str(f"\nA critical error occurred: {e}"), main_logger=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str("Please check the error logs (logs/errors.log) for details."), main_logger=str(__name__))
        # Optional: print traceback to console for immediate feedback
        traceback.print_exc()
    finally:
        log_statement(loglevel=str("info"), logstatement=str("System Orchestrator shutting down."), main_logger=str(__name__))
        logging.shutdown() # Ensure all log handlers are closed properly

if __name__ == "__main__":
    main()


"""
def main():
    from pathlib import Path
    # --- Add project root to sys.path ---
    # This allows imports like `from src.utils...` when running `python main.py` from project root
    project_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_root))
    # ------------------------------------

    # Import configuration and logger setup first
    # Assuming the modular structure with 'src'
    try:
        # Setup logger *before* importing other modules that might log during import
        from utils.logger import setup_logger
        setup_logger() # Configure logging system wide
        logger = logging.getLogger(__name__) # Get logger for main script AFTER setup

        from utils import config # Import the whole config module for easy access

    except ImportError as e:
         # Fallback if src structure isn't found
         print(f"[CRITICAL] Failed to import from src. Ensure 'src' directory exists and script is run from the project root. Error: {e}", file=sys.stderr)
         # Basic logging setup if main one failed
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
         logger = logging.getLogger(__name__)
         log_statement(loglevel=str("critical"), logstatement=str("Failed to initialize core components from src. Limited functionality."), main_logger=str(__name__))
         # Define fallbacks or exit if core components are essential
         LOG_FILE = Path("main_fallback.log") # type: ignore
         class config: # Dummy config
              class DataProcessingConfig: REPO_FILE = "fallback_repo.csv.zst"
              class SyntheticDataConfig: TARGET_SAMPLES = 0
              class TestConfig: pass
              DEFAULT_DEVICE = 'cpu'
              TOKENIZED_DATA_DIR = Path('./data/tokenized') # type: ignore

    # Now import other components using try-except
    try:
        from src.data.processing import DataProcessor, Tokenizer
        DATA_PROCESSING_AVAILABLE = True
    except ImportError as e:
        log_statement(loglevel=str("warning"), logstatement=str(f"Data processing modules not fully available: {e}"), main_logger=str(__name__))
        DATA_PROCESSING_AVAILABLE = False
        class DataProcessor:
            def process_all(self):
                raise NotImplementedError("Module not loaded")
        class Tokenizer:
            def tokenize_all(self):
                raise NotImplementedError("Module not loaded")

    try:
        from data.synthetic import SyntheticDataGenerator
        SYNTHETIC_DATA_AVAILABLE = True
    except ImportError as e:
        log_statement(loglevel=str("warning"), logstatement=str(f"Synthetic data module not available: {e}"), main_logger=str(__name__))
        SYNTHETIC_DATA_AVAILABLE = False
        class SyntheticDataGenerator:
            def generate_dataset(self):
                raise NotImplementedError("Module not loaded")

    try:
        from training.trainer import EnhancedTrainer
        from core.models import ZoneClassifier # Example model
        from data.loaders import EnhancedDataLoader # Example loader
        import torch.nn as nn # For loss function example
        TRAINING_AVAILABLE = True
    except ImportError as e:
        log_statement(loglevel=str("warning"), logstatement=str(f"Training modules not fully available: {e}"), main_logger=str(__name__))
        TRAINING_AVAILABLE = False
        # Define dummy classes correctly, each on its own line
        class EnhancedTrainer:
            def train(self):
                raise NotImplementedError("Module not loaded")
            def load_checkpoint(self, fn):
                raise NotImplementedError("Module not loaded")
        class ZoneClassifier: pass
        class EnhancedDataLoader: pass
        # Define dummy nn structure correctly
        class _DummyNNModule: pass # Base dummy
        class _DummyMSELoss(_DummyNNModule): pass
        class nn: # Dummy nn module
            Module = _DummyNNModule
            MSELoss = _DummyMSELoss
    try:
        from analysis.labeler import SemanticLabeler
        ANALYSIS_AVAILABLE = True
    except ImportError as e:
        log_statement(loglevel=str("warning"), logstatement=str(f"Analysis (labeler) module not available: {e}"), main_logger=str(__name__))
        ANALYSIS_AVAILABLE = False
        class SemanticLabeler:
            def generate_label(self, emb):
                raise NotImplementedError("Module not loaded")

    try:
        # Ensure Path is available if fallback config was used
        from pathlib import Path
        log_statement(loglevel=str("info"), logstatement=str("--- System Orchestrator Started ---"), main_logger=str(__name__))
        orchestrator = SystemOrchestrator()
        try:
            orchestrator.show_main_menu()
        except Exception as main_err:
            # Catch any unexpected error in the main loop itself
            logger.critical(f"Critical error in main orchestrator loop: {main_err}", exc_info=True)
            print(f"\n!!! A critical error occurred: {main_err}. Check logs for details. Exiting. !!!")
            sys.exit(1)
    finally:
        # This will run even if sys.exit() was called
        log_statement(loglevel=str("info"), logstatement=str("--- System Orchestrator Exiting ---"), main_logger=str(__name__))

if __name__ == "__main__":
    main()
"""