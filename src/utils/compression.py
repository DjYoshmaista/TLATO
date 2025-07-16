# src/utils/compression.py
import pandas as pd
import zstandard as zstd
import gzip
import os
import io
import shutil
from pathlib import Path
from typing import Optional, Any, List, Union, Dict
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

def _load_compressed_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Load DataFrame from zstd compressed CSV"""
    try:
        dctx = zstd.ZstdDecompressor()
        with open(filepath, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                df = pd.read_csv(text_stream)
        return df
    except Exception as e:
        log_statement('error', f"Error loading compressed CSV: {e}", __file__)
        return None

def compress_dataframe(df: pd.DataFrame, 
                      output_filepath: Union[str, Path],
                      compression: str = 'zstd',
                      compression_level: int = 3) -> bool:
    """
    Compress a pandas DataFrame to a file.
    
    Args:
        df: The DataFrame to compress
        output_filepath: Path where the compressed file will be saved
        compression: Type of compression ('zstd', 'gzip')
        compression_level: Compression level (1-22 for zstd, 1-9 for gzip)
    
    Returns:
        bool: True if successful, False otherwise
    """
    global LOG_INS
    try:
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')
        
        if compression == 'zstd':
            # Use zstandard compression
            cctx = zstd.ZstdCompressor(level=compression_level)
            compressed_data = cctx.compress(csv_data)
            
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
                
        elif compression == 'gzip':
            # Use gzip compression
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f:
                f.write(csv_data)
        else:
            # No compression, just save as CSV
            df.to_csv(output_path, index=False)
            
        log_statement('info', f"{LOG_INS}:INFO>>Successfully compressed DataFrame to {output_path} using {compression}", Path(__file__).stem)
        return True
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to compress DataFrame to {output_filepath}: {e}", Path(__file__).stem)
        return False

def decompress_dataframe(input_filepath: Union[str, Path],
                        compression: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Decompress a file and load it as a pandas DataFrame.
    
    Args:
        input_filepath: Path to the compressed file
        compression: Type of compression ('zstd', 'gzip'). If None, infers from extension.
    
    Returns:
        pd.DataFrame or None if failed
    """
    global LOG_INS
    try:
        input_path = Path(input_filepath)
        
        if not input_path.exists():
            log_statement('error', f"{LOG_INS}:ERROR>>File not found: {input_path}", Path(__file__).stem)
            return None
            
        # Infer compression type from extension if not specified
        if compression is None:
            if input_path.suffix == '.zst':
                compression = 'zstd'
            elif input_path.suffix == '.gz':
                compression = 'gzip'
            else:
                compression = 'none'
                
        if compression == 'zstd':
            # Decompress zstandard
            with open(input_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.decompress(f.read())
                
            # Convert bytes to DataFrame
            csv_buffer = io.StringIO(decompressed_data.decode('utf-8'))
            df = pd.read_csv(csv_buffer)
            
        elif compression == 'gzip':
            # Decompress gzip and read directly
            with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            # No compression, read as regular CSV
            df = pd.read_csv(input_path)

        log_statement('info', f"{LOG_INS}:INFO>>Successfully decompressed DataFrame from {input_path}", Path(__file__).stem)
        return df
        
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Failed to decompress DataFrame from {input_filepath}: {e}", Path(__file__).stem)
        return None

def _save_compressed_csv(df: pd.DataFrame, filepath: Path):
    """Save DataFrame as zstd compressed CSV"""
    # Prepare DataFrame for CSV serialization
    global LOG_INS
    df_to_save = df.copy()
    
    # Convert timestamps to strings
    for col in TIMESTAMP_COLUMNS:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else ''
            )
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    df_to_save.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    
    # Compress and write
    cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
    with open(filepath, 'wb') as f:
        f.write(cctx.compress(csv_data))

def decompress_file(input_filepath: Union[str, Path],
                   output_filepath: Optional[Union[str, Path]] = None,
                   remove_original: bool = False,
                   dec_to_df: bool = False) -> Optional[Union[pd.DataFrame, Dict, str, bool]]:
    """
    Generic decompression function.
    
    Args:
        input_filepath: Path to compressed file
        output_filepath: Where to save decompressed file (if not dec_to_df)
        remove_original: Whether to remove the compressed file after decompression
        dec_to_df: If True, returns DataFrame directly instead of saving to file
    
    Returns:
        DataFrame if dec_to_df=True, dict if JSON, True/False for file operations
    """
    try:
        input_path = Path(input_filepath)
        
        # Check if it's a DataFrame file (CSV compressed)
        if input_path.name.endswith('.csv.zst') or input_path.name.endswith('.csv.gz'):
            if dec_to_df:
                return decompress_dataframe(input_filepath)
            else:
                # Decompress to file
                df = decompress_dataframe(input_filepath)
                if df is not None and output_filepath:
                    df.to_csv(output_filepath, index=False)
                    if remove_original:
                        input_path.unlink()
                    return True
                return False
                
        # Add handlers for other file types as needed
        log_statement("warning", f"{LOG_INS}:WARNING>>Unsupported file type for decompression: {input_path}", Path(__file__).stem)
        return None
        
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to decompress file {input_filepath}: {e}", Path(__file__).stem)
        return None

def compress_json(json_str: str,
                 output_filepath: Union[str, Path],
                 compression: str = 'zstd') -> bool:
    """Compress JSON string to file."""
    try:
        output_path = Path(output_filepath)
        json_bytes = json_str.encode('utf-8')
        
        if compression == 'zstd':
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(json_bytes)
            with open(output_path, 'wb') as f:
                f.write(compressed)
        elif compression == 'gzip':
            with gzip.open(output_path, 'wb') as f:
                f.write(json_bytes)
        else:
            with open(output_path, 'w') as f:
                f.write(json_str)
        return True
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to compress JSON: {e}", Path(__file__).stem)
        return False

def compress_file_path(source_path: Union[str, Path],
                      output_filepath: Union[str, Path],
                      compression: str = 'zstd',
                      remove_original: bool = False) -> bool:
    """Compress a file from disk."""
    try:
        source = Path(source_path)
        if not source.exists():
            log_statement("error", f"{LOG_INS}:ERROR>>Source file not found: {source}", Path(__file__).stem)
            return False
            
        with open(source, 'rb') as f:
            data = f.read()
            
        if compression == 'zstd':
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(data)
            with open(output_filepath, 'wb') as f:
                f.write(compressed)
        elif compression == 'gzip':
            with gzip.open(output_filepath, 'wb') as f:
                f.write(data)
                
        if remove_original:
            source.unlink()
            
        return True
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to compress file {source_path}: {e}", Path(__file__).stem)
        return False

def compress_file_gzip(source_path: Path, destination_path: Path, remove_original: bool = False, compresslevel: int = 9) -> bool:
    global LOG_INS
    try:
        with open(source_path, 'rb') as f_in:
            with gzip.open(destination_path, 'wb', compresslevel=compresslevel) as f_out:
                shutil.copyfileobj(f_in, f_out)
        if remove_original:
            source_path.unlink()
            log_statement("info", f"{LOG_INS}:INFO>>Successfully gzipped {source_path} to {destination_path}", Path(__file__).stem)
        return True
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to gzip {source_path}: {e}", Path(__file__).stem, exc_info=True)
        if destination_path.exists(): # Cleanup partial file
            destination_path.unlink(missing_ok=True)
        return False

def decompress_gzip_content(gzipped_content: bytes) -> bytes:
    global LOG_INS
    try:
        decompressed_bytes = gzip.decompress(gzipped_content)
        log_statement("info", f"{LOG_INS}:INFO>>Successfully decompressed gzip content in memory.")
        return decompressed_bytes
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to decompress gzip content: {e}", Path(__file__).stem, exc_info=True)
        raise # Re-raise or return None/empty bytes


def decompress_file(input_filepath: str, output_filepath: str, remove_original: Optional[bool] = False, compression: Optional[str] = None, decompresslevel: Optional[int] = 22, dec_to_df: Optional[bool] = False, dec_to_json: Optional[bool] = False):
    """Decompresses a zstandard file."""
    global LOG_INS
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