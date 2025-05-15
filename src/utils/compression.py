# src/utils/compression.py
import pandas as pd
import zstandard as zstd
import gzip
import os
import io
import shutil
from pathlib import Path
from typing import Optional
from src.data.constants import *
# Basic logger setup - adjust as needed or integrate with your main logger
from src.utils.logger import log_statement

# Zstandard compression level (1-22, default 3). Higher is slower but better compression.
# Use zstd.max_compress_level() for the absolute highest.
ZSTD_COMPRESSION_LEVEL = zstd.MAX_COMPRESSION_LEVEL
ZSTD_THREADS = 0 # 0 means auto-detect number of CPU cores for multi-threaded compression


def overwrite_query(compression: str, remove_original: bool):
    """Prompts the user to overwrite or ignore a file."""
    if compression in ['gz', 'gzip']:
        choice = input("Output file exists, overwrite (O), ignore (I), or ask again (S)? [O/I/S]: ".upper())
        if choice == 'O':
            return 'O'
        elif choice == 'I':
            return 'I'
        elif choice == 'S':
            return 'S'
        else:
            return 'S'  # Default to asking again
    else:
        return None  # Handle other compression types as needed


def compress_file(input_filepath: str, output_filepath: str, compression: Optional[str], remove_original: Optional[bool]):
    """Compresses a file using zstandard."""
    try:
        if compression in ['gz', 'gzip']:
            with gzip.open(output_filepath, mode='wb', compresslevel=2) as fh:
                fh.write(buffer.getvalue())
            shutil.move(input_filepath, output_filepath)
        elif compression in ['.zst', 'zstd']:
            cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
            with open(input_filepath, 'rb') as ifh, open(output_filepath, 'rb') as ofh:
                cctx.copy_stream(ifh, ofh)
        # Additional functionality for GZIP
        if remove_original == True:
            # Check for existing files before moving/overwriting
            if os.path.exists(output_filepath) or (os.path.isdir(os.path.dirname(output_filepath)) and os.path.isfile(os.path.join(os.path.dirname(output_filepath), output_filepath))):
                print("Output file exists or destination folder contains the same filename.")
                # Overwrite, ignore, or ask user via overwrite_query function
                result = overwrite_query(compression, remove_original)
                if result == 'O':
                    shutil.move(input_filepath, output_filepath)
                    os.remove(input_filepath)
                elif result in ['I', 'N']:
                    # If the user chooses to ignore, create a new file and keep original
                    # Create new compressed file
                    cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
                    with open(input_filepath, 'rb') as ifh:
                        cctx.copy_ifh_to_string()
                    with io.StringIO(cctx._compressed) as buffer:
                        with gzip.open(buffer, mode='wb') as fh:
                            fh.write(buffer.getvalue())
                        shutil.move(input_filepath, output_filepath + '.gz')
                    # Also create the original file
                    if not os.path.exists(input_filepath):
                        os.makedirs(os.path.dirname(input_filepath), exist_ok=True)
                        shutil.copy2(input_filepath, input_filepath)
                else:
                    pass  # For now, assuming no handling for 'S' yet, as per initial plan.
            else:
                shutil.move(input_filepath, output_filepath)
        else:
            raise ValueError("Invalid compression format.")
        if remove_original and os.path.exists(f"{str(Path(input_filepath).resolve())}.{compression})"):
            os.remove(input_filepath)

    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error during compression: {e}", Path(__file__).stem)
        raise
    except FileNotFoundError:
        log_statement('error', f"{LOG_INS}:ERROR>>Input file not found for compression: {input_filepath}", Path(__file__).stem)
        raise # Re-raise exception to be handled by caller
    except Exception as e:
        log_statement('error', f"{LOG_INS}:ERROR>>Error compressing file '{input_filepath}': {e}", Path(__file__).stem)
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise # Re-raise exception

def compress_file_gzip(source_path: Path, destination_path: Path, remove_original: bool = False, compresslevel: int = 9) -> bool:
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
    try:
        decompressed_bytes = gzip.decompress(gzipped_content)
        log_statement("info", f"{LOG_INS}:INFO>>Successfully decompressed gzip content in memory.")
        return decompressed_bytes
    except Exception as e:
        log_statement("error", f"{LOG_INS}:ERROR>>Failed to decompress gzip content: {e}", exc_info=True)
        raise # Re-raise or return None/empty bytes


def decompress_file(input_filepath: str, output_filepath: str, remove_original: Optional[bool] = False, compression: Optional[str] = None, decompresslevel: Optional[int] = 22, dec_to_df: Optional[bool] = False, dec_to_json: Optional[bool] = False):
    """Decompresses a zstandard file."""
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