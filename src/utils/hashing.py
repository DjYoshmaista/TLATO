# src/utils/hashing.py
import hashlib
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, Optional
from pathlib import Path
from .logger import configure_logging, log_statement
configure_logging()

# --- Configuration ---
# WARNING: Keep this key secure and manage it properly (e.g., env variables, secrets manager)
# For demonstration, we derive a key from a password. In production, generate and store safely.
PASSWORD = b"your-secure-password-for-filepath-encryption" # CHANGE THIS!
SALT = os.urandom(16) # Store this salt alongside the encrypted data or derive consistently

# Constants
HASH_BUFFER_SIZE = 65536  # 64k buffer


# --- Key Derivation ---
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=SALT,
    iterations=480000, # OWASP recommendation as of 2023
)
# Derive a key and encode it for Fernet
ENCRYPTION_KEY = base64.urlsafe_b64encode(kdf.derive(PASSWORD))
_CIPHER_SUITE = Fernet(ENCRYPTION_KEY)



# --- Filepath Hashing (Encryption/Decryption) ---
# Using encryption for filepaths as hashing is one-way
def hash_filepath(filepath: str) -> str:
    """
    Encrypts a filepath to use as a 'hashed' identifier.
    Returns a URL-safe base64 encoded encrypted string.
    """
    try:
        encrypted_path = _CIPHER_SUITE.encrypt(filepath.encode('utf-8'))
        return encrypted_path.decode('utf-8')
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Error encrypting filepath '{filepath}': {e}"), main_logger=str(__name__))
        return "" # Return empty string or handle error as appropriate

def unhash_filepath(hashed_path: str) -> str:
    """
    Decrypts a 'hashed' (encrypted) filepath identifier.
    """
    try:
        decrypted_path = _CIPHER_SUITE.decrypt(hashed_path.encode('utf-8'))
        return decrypted_path.decode('utf-8')
    except Exception as e:
        # This can happen if the key is wrong, data is corrupt, or not valid base64/fernet token
        log_statement(loglevel=str("error"), logstatement=str(f"Error decrypting filepath hash '{hashed_path[:20]}...': {e}"), main_logger=str(__name__))
        return "" # Return empty string or handle error as appropriate

# --- Data Hashing (Content-based) ---
def generate_data_hash(file_path: Union[str, Path]) -> Optional[str]:
    """
    Generates a SHA256 hash for the contents of a file.

    Args:
        file_path (Union[str, Path]): The path to the file.

    Returns:
        Optional[str]: The hexadecimal SHA256 hash of the file content,
                       or None if an error occurs (e.g., file not found, permission error).
    """
    sha256_hash = hashlib.sha256()
    file_path = Path(file_path) # Ensure it's a Path object

    try:
        if not file_path.is_file():
            log_statement(loglevel='error',
                          logstatement=f"File not found or is not a regular file: {file_path}",
                          main_logger=str(__name__))
            return None

        with file_path.open("rb") as f:
            while True:
                data = f.read(HASH_BUFFER_SIZE)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()

    except PermissionError:
        log_statement(loglevel='error',
                      logstatement=f"Permission denied while trying to read file: {file_path}",
                      main_logger=str(__name__), exc_info=True)
        return None
    except OSError as e:
        log_statement(loglevel='error',
                      logstatement=f"OS error reading file {file_path}: {e}",
                      main_logger=str(__name__), exc_info=True)
        return None
    except Exception as e:
        # Catch any other unexpected errors during hashing
        log_statement(loglevel='error',
                      logstatement=f"Unexpected error hashing file {file_path}: {e}",
                      main_logger=str(__name__), exc_info=True)
        return None

# Keep the hash_filepath function as is (or review separately if needed)
def hash_filepath(filepath: Union[str, Path]) -> str:
    """Generates a SHA256 hash for a given filepath string."""
    filepath_str = str(filepath) # Ensure string representation
    return hashlib.sha256(filepath_str.encode('utf-8')).hexdigest()