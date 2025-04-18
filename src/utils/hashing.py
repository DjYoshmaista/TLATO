# src/utils/hashing.py
import hashlib
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Basic logger setup - adjust as needed or integrate with your main logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# WARNING: Keep this key secure and manage it properly (e.g., env variables, secrets manager)
# For demonstration, we derive a key from a password. In production, generate and store safely.
PASSWORD = b"your-secure-password-for-filepath-encryption" # CHANGE THIS!
SALT = os.urandom(16) # Store this salt alongside the encrypted data or derive consistently

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
        logger.error(f"Error encrypting filepath '{filepath}': {e}")
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
        logger.error(f"Error decrypting filepath hash '{hashed_path[:20]}...': {e}")
        return "" # Return empty string or handle error as appropriate

# --- Data Hashing (Content-based) ---
def generate_data_hash(filepath: str, buffer_size=65536) -> str:
    """
    Generates a SHA-256 hash representing the file's content,
    filename, size, and extension. Reads file in chunks for memory efficiency.
    Returns the hex digest of the hash.
    """
    sha256 = hashlib.sha256()
    try:
        # 1. Include file metadata
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)
        _, extension = os.path.splitext(filename)
        metadata_str = f"{filename}:{filesize}:{extension}"
        sha256.update(metadata_str.encode('utf-8'))

        # 2. Include file content
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found for data hashing: {filepath}")
        return ""
    except OSError as e:
        logger.error(f"OS error reading file for data hashing '{filepath}': {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error during data hashing for '{filepath}': {e}")
        return ""