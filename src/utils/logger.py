# -*- coding: utf-8 -*-

"""
logger.py

Logging configuration utility for the TLATO project.
Sets up file and console logging handlers with specified formats.
"""

import logging
from logging.handlers import RotatingFileHandler
import logging.handlers
import os
import sys
from pathlib import Path

# --- Configuration ---
LOG_LEVEL = logging.INFO  # Default root log level (change as needed)
CONSOLE_LOG_LEVEL = logging.DEBUG # Level for console output
FILE_LOG_LEVEL = logging.DEBUG    # Level for file output

# Define default log directory relative to this file's location
# Assumes logger.py is in project_root/src/utils
try:
    # More robust path finding
    UTILS_DIR = Path(__file__).parent.resolve()
    SRC_DIR = UTILS_DIR.parent
    PROJECT_ROOT = SRC_DIR.parent
    DEFAULT_LOG_DIR = PROJECT_ROOT / 'logs'
except Exception:
    # Fallback if path resolution fails
    DEFAULT_LOG_DIR = Path('./logs')

DEFAULT_LOG_FILE_APP = DEFAULT_LOG_DIR / 'app.log'
DEFAULT_LOG_FILE_ERR = DEFAULT_LOG_DIR / 'errors.log'

# Log Rotation Settings
MAX_BYTES = 100 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 50

# Log Format
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Ensure log directory exists
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Define a basic formatter
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration Function ---

# Flag to track if configuration has been done
_logging_configured = False

def configure_logging(
    log_file_path: str | Path = DEFAULT_LOG_FILE_APP,
    error_log_file_path: str | Path = DEFAULT_LOG_FILE_ERR,
    log_level: int = LOG_LEVEL,
    console_log_level: int = CONSOLE_LOG_LEVEL,
    file_log_level: int = FILE_LOG_LEVEL,
    console_logging: bool = True,
    file_logging: bool = True
):
    """
    Configures the root logger for the application.

    Should ideally be called only ONCE at the application entry point.
    Includes checks to prevent adding duplicate handlers if called multiple times.

    Args:
        log_file_path (str | Path): Path to the main application log file.
        error_log_file_path (str | Path): Path to the error log file (warnings and above).
        log_level (int): The base level for the root logger.
        console_log_level (int): The level for console output.
        file_log_level (int): The level for the main log file output.
        console_logging (bool): Enable/disable console logging.
        file_logging (bool): Enable/disable file logging.
    """
    global _logging_configured
    if _logging_configured:
        logging.warning("Logger already configured. Skipping reconfiguration.")
        return # Avoid reconfiguring

    try:
        # Ensure paths are Path objects
        log_file_path = Path(log_file_path)
        error_log_file_path = Path(error_log_file_path)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        root_logger = logging.getLogger() # Get the root logger
        root_logger.setLevel(log_level) # Set the *root* logger level

        # --- Remove existing handlers (optional, but good practice before reconfiguring) ---
        # This prevents duplicate handlers if this function *were* accidentally called again
        # despite the _logging_configured flag.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handler to release file resources

        # --- Console Handler ---
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout) # Use stdout
            console_handler.setLevel(console_log_level) # Level for this specific handler
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # --- File Handlers ---
        if file_logging:
            # Ensure log directory exists
            log_dir = log_file_path.parent
            error_log_dir = error_log_file_path.parent
            try:
                if log_dir: log_dir.mkdir(parents=True, exist_ok=True)
                if error_log_dir: error_log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.error(f"Could not create log directory: {e}", exc_info=True)
                # Optionally disable file logging if directory creation fails
                file_logging = False

            if file_logging: # Re-check if directory creation succeeded
                # General File Handler (Rotating)
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                    encoding='utf-8' # Explicitly set encoding
                )
                file_handler.setLevel(file_log_level) # Level for file handler
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)

                # Error File Handler (Rotating) - Captures WARNING, ERROR, CRITICAL
                error_file_handler = logging.handlers.RotatingFileHandler(
                    error_log_file_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                    encoding='utf-8'
                )
                error_file_handler.setLevel(logging.WARNING) # Only log WARNING and above
                error_file_handler.setFormatter(formatter)
                root_logger.addHandler(error_file_handler)

        # Log confirmation message using the newly configured logger
        init_logger = logging.getLogger(__name__) # Logger for this module
        if file_logging:
            init_logger.info(f"Logging configured. Main log: {log_file_path}, Error log: {error_log_file_path}")
        elif console_logging:
            init_logger(loglevel=str("info"), logstatement=str(f"Logging configured (Console only)."), main_logger=str(__name__))
        else:
            print("WARNING: All logging handlers disabled.") # Use print if logger might not output

        _logging_configured = True # Mark as configured

    except Exception as e:
        # Fallback to basic config if setup fails
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - [logger.py] - Configuration Error: %(message)s')
        logging.error(f"Failed to configure custom logging: {e}", exc_info=True)
        # Do not set _logging_configured = True here
        # Depending on the severity, the main app might need to exit
        # raise # Re-raise the exception so the main app knows config failed?

# --- Define constants for use by other modules ---
LOG_DIR = DEFAULT_LOG_DIR
APP_LOG_FILE = DEFAULT_LOG_FILE_APP
ERROR_LOG_FILE = DEFAULT_LOG_FILE_ERR

def log_statement(loglevel, logstatement, main_logger, exc_info=None):
    """
    Logs a statement with the specified log level, logger, and optional exception info.

    Args:
        loglevel (str): The log level as a string (e.g., "info", "error").
        logstatement (str): The log message to be logged.
        main_logger (str): The name of the logger to use.
        exc_info (bool | Exception, optional): If True, includes exception traceback. 
                                               If an Exception object is provided, logs its traceback.
    """
    main_log = logging.getLogger(main_logger)
    loglevel_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    log_level = loglevel_mapping.get(loglevel.lower(), logging.INFO)

    # Log the message with optional exception info and correct stacklevel
    main_log.log(log_level, logstatement, exc_info=exc_info, stacklevel=2)

    # Additionally log to the "app.log" logger with correct stacklevel
    app_log = logging.getLogger("app.log")
    app_log.log(log_level, logstatement, exc_info=exc_info, stacklevel=2)

# --- Example Usage (if run directly) ---
if __name__ == '__main__':
    logger = logging.getLogger("TLATO_loggerLog")
    # Example of how to use the configure_logging function
    # In a real application, this would likely be called from the main entry point.

    # Create dummy log directory for testing
    test_log_dir = "test_logs"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)

    test_log_file = os.path.join(test_log_dir, "test_app.log")
    test_error_file = os.path.join(test_log_dir, "test_errors.log")

    log_statement(loglevel=str("info"), logstatement=str("Configuring logging for test..."), main_logger=str(__name__))
    configure_logging(log_file_path=test_log_file,
                      error_log_file_path=test_error_file,
                      log_level=logging.DEBUG) # Set root to DEBUG for testing

    # Get a logger instance for testing
    test_logger = logging.getLogger("TestLogger")

    log_statement(loglevel=str("info"), logstatement=str("Sending test log messages..."), main_logger=str(__name__))
    test_logger(loglevel=str("debug"), logstatement=str("This is a debug message."), main_logger=str(__name__))
    test_logger(loglevel=str("info"), logstatement=str("This is an info message."), main_logger=str(__name__))
    test_logger(loglevel=str("warning"), logstatement=str("This is a warning message."), main_logger=str(__name__))
    test_logger(loglevel=str("error"), logstatement=str("This is an error message."), main_logger=str(__name__))
    test_logger(loglevel=str("critical"), logstatement=str("This is a critical message."), main_logger=str(__name__))

    log_statement(loglevel=str("info"), logstatement=str(f"Check '{test_log_file}' and '{test_error_file}' for output."), main_logger=str(__name__))

    # Example of logging an exception
    try:
        result = 1 / 0
    except ZeroDivisionError:
        test_logger.error("An exception occurred!", exc_info=True) # exc_info=True adds traceback

    logging.shutdown() # Cleanly close handlers
    log_statement(loglevel=str("info"), logstatement=str("Logging test complete."), main_logger=str(__name__))

# --- Usage in other modules ---
# In other modules (e.g., src/data/processing.py), simply do:
# import logging
# logger = logging.getLogger(__name__)
#
# # Then use logger.info, logger.debug, logger.error, logger.exception etc.
# # Example error logging with traceback:
# try:
#   # ... some operation ...
#   result = 1 / 0
# except Exception as e:
#   logger.exception(f"An error occurred during operation: {e}") # Logs ERROR level + traceback
#   # or logger.error(f"An error occurred: {e}", exc_info=True)

