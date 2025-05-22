# src/utils/compression.py
import pandas as pd
import zstandard as zstd
import gzip
import os
import io
import shutil
from pathlib import Path
from typing import Optional, Any, List, Union
from src.data.constants import *
from src.utils.logger import log_statement

# Zstandard compression level (1-22, default 3). Higher is slower but better compression.
# Use zstd.max_compress_level() for the absolute highest.
ZSTD_COMPRESSION_LEVEL = zstd.MAX_COMPRESSION_LEVEL
ZSTD_THREADS = 0 # 0 means auto-detect number of CPU cores for multi-threaded compression
LOG_INS = f"{__file__}:{__name__}:"

# --- Functions ---
def det_ext(filepath: str) -> str:
    """Determines the extension type from a filepath.
    
    Args:
        filepath: Path to the file
        
    Returns:
        String indicating the type of file based on extension
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext in ['.zst', '.zstd']:
        return 'zst'
    elif ext in ['.gz', '.gzip']:
        return 'gz'
    elif ext in ['.csv', '.json', '.tsv', '.xls', '.xlsx', '.parquet']:
        return 'df'
    elif ext in ['.txt', '.text']:
        return 'text'
    elif ext in ['.docx', '.doc']:
        return 'docx'
    elif ext == '.pdf':
        return 'pdf'
    else:
        return 'unknown'

def parse_filename(filepath: Union[str, Path]) -> str:
    """Parses the filename of the last file/folder in the path.
    
    Args:
        filepath: Path to parse
        
    Returns:
        The basename of the filepath
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"

    # Convert to string if it's a Path object
    if isinstance(filepath, Path):
        filepath = str(filepath)
    
    # Extract the last component of the path
    basename = os.path.basename(filepath)
    return basename

def separate_filename_ext(filename: str) -> tuple:
    """Separates a filename into base and extension parts.
    
    Args:
        filename: Filename to separate
        
    Returns:
        Tuple of (base_filename, extension)
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    base, ext = os.path.splitext(filename)
    return base, ext.lower()

def check_filename(input_filepath: str, output_filepath: str, filetype: Optional[str] = None, 
                   dtype: Optional[str] = None) -> List[str]:
    """Check and process filenames based on input filepath.
    
    Args:
        input_filepath: Path to the input file
        output_filepath: Path where output will be saved
        filetype: Type of file (optional)
        dtype: Data type (optional)
        
    Returns:
        List containing processed filename
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    if not isinstance(input_filepath, str):
        return [os.path.basename(str(input_filepath))]
    
    basename = os.path.basename(input_filepath)
    
    if "repo" in basename:
        return [basename]
    else:
        return [basename.split(".")[0]]

def set_filename(input_filename: Optional[str] = None, input_filepath: Any = None, 
                 output_filepath: Optional[str] = None, filetype: Optional[str] = None, 
                 dtype: Optional[str] = None) -> str:
    """Sets the filename based on the output directory and filetype.
    
    Args:
        input_filename: Base filename (optional)
        input_filepath: Path to input file or data object
        output_filepath: Directory for output
        filetype: Type of file (optional)
        dtype: Data type (optional)
        
    Returns:
        Complete path for the output file
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"

    # Determine extension based on input type
    if isinstance(input_filepath, str):
        ext = det_ext(input_filepath)
        base = os.path.basename(input_filepath)
        if 'repo' in base:
            base = base
            ext = 'repo'
        else:
            base, _ = os.path.splitext(base)
    elif isinstance(input_filepath, pd.DataFrame):
        ext = 'df'
        base = input_filename or "dataframe_output"
    elif isinstance(input_filepath, bytes):
        ext = 'bytes'
        base = input_filename or "bytes_output"
    elif isinstance(input_filepath, list):
        ext = 'list'
        base = input_filename or "list_output"
    elif isinstance(input_filepath, dict):
        ext = 'dict'
        base = input_filename or "dict_output"
    else:
        ext = 'unknown'
        base = input_filename or "unknown_output"

    # Ensure the output directory exists
    if output_filepath:
        os.makedirs(output_filepath, exist_ok=True)
    else:
        output_filepath = os.getcwd()
    
    # Set the filename based on filetype and dtype
    if filetype:
        filename = f"{base}.{filetype}"
    elif dtype:
        filename = f"{base}.{dtype}.{ext}"
    else:
        filename = f"{base}.{ext}"
    
    return os.path.join(output_filepath, filename)

def compress_files(file_list: List[Union[str, Path]], output_dir: str, compression: str = 'zst',
                   remove_original: bool = False, filetype: Optional[str] = None, 
                   dtype: Optional[str] = None) -> None:
    """
    Compress multiple files with consistent handling of batch operations.
    
    Args:
        file_list: List of files to compress
        output_dir: Directory to place compressed files
        compression: Compression format ('zst', 'zstd', 'gz', or 'gzip')
        remove_original: Whether to remove original files
        filetype: Type of file (optional)
        dtype: Data type (optional)
    
    -->Initialize action scope tracker
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"

    action_scope = {
        'action': None,
        'scope': None,
        'folder': None
    }
    
    processed_count = 0
    skipped_count = 0
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting compression of {len(file_list)} files to {output_dir}")
    
    # Process each file
    for i, filepath in enumerate(file_list):
        # Check if we should stop entirely based on previous decisions
        if action_scope.get('scope') == 'none_all':
            print(f"Stopping compression as requested. Processed {processed_count} files, skipped {skipped_count} files.")
            break
            
        current_folder = os.path.dirname(str(filepath))
        
        # Check if we should skip this folder based on previous decisions
        if action_scope.get('scope') == 'none_folder' and current_folder == action_scope.get('folder'):
            print(f"Skipping file in folder {current_folder}: {os.path.basename(str(filepath))}")
            skipped_count += 1
            continue

        # Progress indicator
        print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(str(filepath))}")
        
        try:
            # Compress the file and get updated action_scope
            success, action_scope = compress_file(
                input_filepath=filepath,
                output_filepath=output_dir,
                compression=compression,
                remove_original=remove_original,
                filetype=filetype,
                dtype=dtype,
                action_scope=action_scope
            )
            
            if success:
                processed_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            log_statement('error', f"Error processing file {filepath}: {e}", Path(__file__).stem)
            skipped_count += 1
    
    print(f"Compression complete. Successfully processed {processed_count} files, skipped {skipped_count} files.")

def overwrite_query(compression: str, remove_original: bool, filepath: str) -> tuple:
    """Ask user about overwriting existing files with option to apply to multiple files.
    
    Args:
        compression: Type of compression being used
        remove_original: Whether original will be removed
        filepath: Path of the current file being processed
        
    Returns:
        Tuple of (action, scope):
            action: User choice: 'O' (Overwrite), 'I' (Ignore), 'N' (New file), 'S' (Skip)
            scope: Scope of the action: 'single', 'folder', 'all', 'none_folder', 'none_all'
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    while True:
        print(f"Processing file: {os.path.basename(filepath)}")
        answer = input(f"Output file exists. Options:\n"
                      f"[O]verwrite: Replace existing file\n"
                      f"[I]gnore: Keep original and create compressed file\n"
                      f"[N]ew file: Create new file with different name\n"
                      f"[S]kip: Don't create compressed file\n"
                      f"Your choice [O/I/N/S]: ").upper()
        
        if answer in ['O', 'I', 'N', 'S']:
            # Ask about scope of this action
            scope = input(f"Apply this action to:\n"
                         f"[T]his file only\n"
                         f"[F]older: All files in the same folder\n"
                         f"[A]ll files in the entire operation\n"
                         f"[NF] No files in this folder (skip this folder)\n"
                         f"[NA] No more files (stop entire operation)\n"
                         f"Your choice [T/F/A/NF/NA]: ").upper()
            
            scope_map = {
                'T': 'single',    # Just this file
                'F': 'folder',    # All files in this folder
                'A': 'all',       # All files in the operation
                'NF': 'none_folder', # Skip all files in this folder
                'NA': 'none_all'  # Stop entire operation
            }
            
            if scope in scope_map:
                return answer, scope_map[scope]
            else:
                print("Invalid scope option. Please try again.")
        else:
            print("Invalid action option. Please try again.")

def compress_file(input_filepath: Any, output_filepath: str, compression: Optional[str] = 'zst', 
                  remove_original: Optional[bool] = False, filetype: Optional[str] = None, 
                  dtype: Optional[str] = None, 
                  action_scope: Optional[dict] = None) -> tuple:
    """Compresses a file using zstandard or gzip.
    
    Args:
        input_filepath: Path to input file or data object
        output_filepath: Path for output file
        compression: Compression format ('zst', 'zstd', 'gz', or 'gzip')
        remove_original: Whether to remove the original file
        filetype: Type of file (optional)
        dtype: Data type (optional)
        action_scope: Dictionary tracking user's batch operation decisions
        
    Returns:
        Tuple of (success, action_scope) where success is boolean and 
        action_scope is the updated action_scope dictionary
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    temp_csv = None
    
    # Initialize action_scope if not provided
    if action_scope is None:
        action_scope = {
            'action': None,
            'scope': None,
            'folder': None
        }
    
    try:
        # Handle DataFrame input by saving to temp file first
        if isinstance(input_filepath, pd.DataFrame):
            df_to_save = input_filepath
            temp_csv = "/tmp/TLATO.compression.input_filepath.csv"
            # Ensure the temp file is unique
            if os.path.exists(temp_csv):
                temp_csv = f"/tmp/TLATO.compression.input_filepath.{os.getpid()}.csv"
            # Save DataFrame to CSV
            df_to_save.to_csv(temp_csv, index=False)
            input_to_compress = temp_csv
        else:
            input_to_compress = input_filepath
        
        # Determine the full output path
        if os.path.isdir(output_filepath):
            full_output_path = set_filename(
                input_filepath=input_to_compress, 
                output_filepath=output_filepath, 
                filetype=filetype, 
                dtype=dtype
            )
            # Add compression extension if not present
            if not full_output_path.endswith(f".{compression}"):
                full_output_path = f"{full_output_path}.{compression}"
        else:
            full_output_path = output_filepath
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        # Perform compression based on format
        if compression in ['gz', 'gzip']:
            if isinstance(input_to_compress, str) and os.path.exists(input_to_compress):
                with open(input_to_compress, 'rb') as ifh:
                    with gzip.open(full_output_path, mode='wb', compresslevel=2) as ofh:
                        shutil.copyfileobj(ifh, ofh)
            else:
                # Handle non-file inputs
                buffer = io.BytesIO()
                with gzip.open(full_output_path, mode='wb', compresslevel=2) as fh:
                    fh.write(buffer.getvalue())
                    
        elif compression in ['zst', 'zstd']:
            cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
            if isinstance(input_to_compress, str) and os.path.exists(input_to_compress):
                with open(input_to_compress, 'rb') as ifh, open(full_output_path, 'wb') as ofh:
                    cctx.copy_stream(ifh, ofh)
            else:
                # Handle non-file inputs
                raise ValueError("Non-file inputs not implemented for zstd compression")
        else:
            raise ValueError(f"Invalid compression format: {compression}")
        
        # Handle original file removal if requested
        if remove_original and not isinstance(input_filepath, pd.DataFrame):
            if os.path.exists(input_filepath):
                current_folder = os.path.dirname(input_filepath)
                
                # Check if file already exists at destination
                if os.path.exists(full_output_path):
                    # Determine if we need to ask the user
                    action = None
                    
                    # Check if we already have a decision for this file/folder
                    if action_scope['action'] is not None:
                        if action_scope['scope'] == 'all':
                            action = action_scope['action']
                        elif action_scope['scope'] == 'folder' and current_folder == action_scope['folder']:
                            action = action_scope['action'] 
                        elif action_scope['scope'] == 'none_folder' and current_folder == action_scope['folder']:
                            return False, action_scope  # Skip this file
                        elif action_scope['scope'] == 'none_all':
                            return False, action_scope  # Skip all remaining files
                    
                    # If no previous decision applies, ask user
                    if action is None:
                        action, scope = overwrite_query(compression, remove_original, input_filepath)
                        
                        # Update action_scope based on user's choice
                        action_scope['action'] = action
                        action_scope['scope'] = scope
                        action_scope['folder'] = current_folder
                        
                        # Check if we should skip this file or stop entirely
                        if scope == 'none_folder' and current_folder == action_scope['folder']:
                            return False, action_scope  # Skip this file
                        elif scope == 'none_all':
                            return False, action_scope  # Skip all remaining files
                    
                    # Apply the action
                    if action == 'O':  # Overwrite
                        os.remove(input_filepath)
                    elif action == 'I':  # Ignore (keep both)
                        pass  # Do nothing, keep both files
                    elif action == 'N':  # New file
                        new_name = f"{full_output_path}.new"
                        if compression in ['zst', 'zstd']:
                            cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
                            with open(input_filepath, 'rb') as ifh, open(new_name, 'wb') as ofh:
                                cctx.copy_stream(ifh, ofh)
                        elif compression in ['gz', 'gzip']:
                            with open(input_filepath, 'rb') as ifh:
                                with gzip.open(new_name, mode='wb', compresslevel=2) as ofh:
                                    shutil.copyfileobj(ifh, ofh)
                    elif action == 'S':  # Skip
                        # Remove the compressed file since we're skipping
                        if os.path.exists(full_output_path):
                            os.remove(full_output_path)
                else:
                    # No conflict, can safely remove original
                    os.remove(input_filepath)
        
        return True, action_scope

    except FileNotFoundError:
        log_statement('error', f"{LOG_INS}:ERROR>>Input file not found for compression: {input_filepath}", 
                      Path(__file__).stem)
        raise  # Re-raise exception to be handled by caller
    
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error compressing file '{input_filepath}': {e}", 
                     Path(__file__).stem)
        # Clean up potentially incomplete output file
        if os.path.exists(full_output_path):
            try:
                os.remove(full_output_path)
            except OSError:
                pass
        raise  # Re-raise exception
    
    finally:
        # Clean up temporary files
        if temp_csv and os.path.exists(temp_csv):
            try:
                os.remove(temp_csv)
            except OSError:
                pass

def compress_file_gzip(source_path: Path, destination_path: Path, remove_original: bool = False, compresslevel: int = 9) -> bool:
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    try:
        with open(source_path, 'rb') as f_in:
            with gzip.open(destination_path, 'wb', compresslevel=compresslevel) as f_out:
                shutil.copyfileobj(f_in, f_out)
        if remove_original:
            source_path.unlink()
            log_statement("info", f"{LOG_INS}:INFO>>Successfully gzipped {source_path} to {destination_path}")
        return True
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to gzip {source_path}: {e}", exc_info=True)
        if destination_path.exists(): # Cleanup partial file
            destination_path.unlink(missing_ok=True)
        return False

def decompress_gzip_content(gzipped_content: bytes) -> bytes:
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    try:
        decompressed_bytes = gzip.decompress(gzipped_content)
        log_statement("info", f"{LOG_INS}:INFO>>Successfully decompressed gzip content in memory.")
        return decompressed_bytes
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to decompress gzip content: {e}", exc_info=True)
        raise # Re-raise or return None/empty bytes


def decompress_file(input_filepath: str, output_filepath: str, remove_original: Optional[bool] = False, compression: Optional[str] = None, decompresslevel: Optional[int] = 22, dec_to_df: Optional[bool] = False, dec_to_json: Optional[bool] = False):
    """Decompresses a zstandard file."""
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    try:
        if input_filepath.endswith('.zst') or input_filepath.endswith('.zstd'):
            dctx = zstd.ZstdDecompressor()
            with open(input_filepath, 'rb') as ifh, open(output_filepath, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
            # If the original file is not needed, remove it
            if os.path.exists(input_filepath) and remove_original:
                os.remove(input_filepath)
            log_statement('debug', f"{LOG_INS}:DEBUG>>Decompressed '{input_filepath}' to '{output_filepath}'", Path(__file__).stem)
        elif input_filepath.endswith('.gz') or input_filepath.endswith('.gzip'):
            with gzip.open(input_filepath, 'rb') as ifh, open(output_filepath, 'wb') as ofh:
                shutil.copyfileobj(ifh, ofh)
            # If the original file is not needed, remove it
            if os.path.exists(input_filepath) and remove_original:
                os.remove(input_filepath)
            # Log the successful decompression
            log_statement('debug', f"{LOG_INS}:DEBUG>>Decompressed '{input_filepath}' to '{output_filepath}'", Path(__file__).stem)
        else:
            raise ValueError(f"Unsupported compression format for file: {input_filepath}")
        if dec_to_df:
            n = output_filepath.split('.')[-1]
            if n == 'json':
                # Read the decompressed file into a DataFrame
                return pd.read_json(output_filepath, lines=True)
            elif n == 'parquet':
                # Read the decompressed file into a DataFrame
                return pd.read_parquet(output_filepath)
            elif n == 'txt':
                # Read the decompressed file into a DataFrame
                return pd.read_csv(output_filepath, sep='\t', header=None, quoting=3)
            if n in ['csv', 'tsv', 'txt', 'text', 'xls', 'xlsx']:
                # Read the decompressed file into a DataFrame
                return pd.read_csv(output_filepath, sep=',', header=None, quoting=3)
            else:
                raise ValueError(f"Unsupported file format for DataFrame conversion: {n}")
        return None
    except ValueError as ve:
        log_statement('error', f"{LOG_INS}:ERROR>>ValueError during decompression: {ve}", Path(__file__).stem)
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise
    except FileNotFoundError:
        log_statement('error', f"{LOG_INS}:ERROR>>Input file not found for decompression: {input_filepath}", Path(__file__).stem)
        raise
    except zstd.ZstdError as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Zstd decompression error for file '{input_filepath}': {e} - Might be corrupt or not a zstd file.", Path(__file__).stem)
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error decompressing file '{input_filepath}': {e}", Path(__file__).stem)
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise

def stream_decompress_lines(input_filepath: str, encoding='utf-8'):
    """
    Yields lines from a zstandard compressed text file using streaming decompression.
    Handles potential decompression errors during iteration.
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    try:
        with open(input_filepath, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            # Use iter_lines for text data, adjust buffer size if needed
            stream_reader = dctx.stream_reader(fh)
            text_io = io.TextIOWrapper(stream_reader, encoding=encoding)
            for line in text_io:
                yield line.rstrip('\n') # Remove trailing newline like standard file reading
    except FileNotFoundError:
        log_statement('error', f"{LOG_INS}:ERROR>>Input file not found for streaming decompression: {input_filepath}", Path(__file__).stem)
        # Decide if you want to raise an error or yield nothing
        # raise
        return
    except zstd.ZstdError as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Zstd decompression error during streaming '{input_filepath}': {e}", Path(__file__).stem)
        # Decide how to handle mid-stream errors, e.g., stop iteration
        # raise
        return
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Unexpected error during streaming decompression '{input_filepath}': {e}", Path(__file__).stem)
        # raise
        return

def stream_compress_lines(output_filepath: str, lines_generator, encoding='utf-8'):
    """
    Compresses lines from a generator into a zstandard file using streaming.
    """
    global LOG_INS
    LOG_INS += f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}:"
    try:
        with open(output_filepath, 'wb') as fh:
            cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
            compressor = cctx.stream_writer(fh)
            for line in lines_generator:
                # Ensure line ends with a newline and is encoded
                compressor.write(f"{line}\n".encode(encoding))
            compressor.flush(zstd.FLUSH_FRAME) # Ensure all data is written
        log_statement('debug', f"{LOG_INS}:DEBUG>>Stream compressed lines to '{output_filepath}'", Path(__file__).stem)
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error during streaming compression to '{output_filepath}': {e}", Path(__file__).stem)
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise