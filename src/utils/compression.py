# src/utils/compression.py
import zstandard as zstd
import os
import io

# Basic logger setup - adjust as needed or integrate with your main logger
from src.utils.logger import configure_logging, log_statement

# Zstandard compression level (1-22, default 3). Higher is slower but better compression.
# Use zstd.max_compress_level() for the absolute highest.
ZSTD_COMPRESSION_LEVEL = zstd.MAX_COMPRESSION_LEVEL
ZSTD_THREADS = 0 # 0 means auto-detect number of CPU cores for multi-threaded compression

def compress_file(input_filepath: str, output_filepath: str):
    """Compresses a file using zstandard."""
    try:
        cctx = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL, threads=ZSTD_THREADS)
        with open(input_filepath, 'rb') as ifh, open(output_filepath, 'wb') as ofh:
            cctx.copy_stream(ifh, ofh)
        log_statement(loglevel=str("debug"), logstatement=str(f"Compressed '{input_filepath}' to '{output_filepath}'"), main_logger=str(__name__))
    except FileNotFoundError:
        log_statement(loglevel=str("error"), logstatement=str(f"Input file not found for compression: {input_filepath}"), main_logger=str(__name__))
        raise # Re-raise exception to be handled by caller
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Error compressing file '{input_filepath}': {e}"), main_logger=str(__name__))
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise # Re-raise exception

def decompress_file(input_filepath: str, output_filepath: str):
    """Decompresses a zstandard file."""
    try:
        dctx = zstd.ZstdDecompressor()
        with open(input_filepath, 'rb') as ifh, open(output_filepath, 'wb') as ofh:
            dctx.copy_stream(ifh, ofh)
        log_statement(loglevel=str("debug"), logstatement=str(f"Decompressed '{input_filepath}' to '{output_filepath}'"), main_logger=str(__name__))
    except FileNotFoundError:
        log_statement(loglevel=str("error"), logstatement=str(f"Input file not found for decompression: {input_filepath}"), main_logger=str(__name__))
        raise
    except zstd.ZstdError as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Zstd decompression error for file '{input_filepath}': {e} - Might be corrupt or not a zstd file."), main_logger=str(__name__))
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Error decompressing file '{input_filepath}': {e}"), main_logger=str(__name__))
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
        log_statement(loglevel=str("error"), logstatement=str(f"Input file not found for streaming decompression: {input_filepath}"), main_logger=str(__name__))
        # Decide if you want to raise an error or yield nothing
        # raise
        return
    except zstd.ZstdError as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Zstd decompression error during streaming '{input_filepath}': {e}"), main_logger=str(__name__))
        # Decide how to handle mid-stream errors, e.g., stop iteration
        # raise
        return
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during streaming decompression '{input_filepath}': {e}"), main_logger=str(__name__))
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
        log_statement(loglevel=str("debug"), logstatement=str(f"Stream compressed lines to '{output_filepath}'"), main_logger=str(__name__))
    except Exception as e:
        log_statement(loglevel=str("error"), logstatement=str(f"Error during streaming compression to '{output_filepath}': {e}"), main_logger=str(__name__))
        # Clean up potentially incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except OSError:
                pass
        raise