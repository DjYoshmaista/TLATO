# --- Standalone Data Processing Functions ---
# (These could be part of DataRepository or separate)
from src.utils.logger import log_statement
from src.utils.compression import decompress_file, compress_file, stream_decompress_lines, stream_compress_lines
import os
import shutil

class FileProcessor:
    def __init(self, input_filepath: str, output_filepath: str):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.temp_decompressed = None
        self.temp_compressed = None
        self.temp_tokenized = None
        self.temp_tokenized_compressed = None
        self.temp_tokenized_decompressed = None

    # Placeholder for actual file processing logic called by DataRepository.process_data_list
    def process_file_content(self, input_filepath: str, output_filepath: str):
        """Processes content of a single file."""
        # TODO: Implement based on file type, calling specific readers/processors
        log_statement(loglevel=str("debug"), logstatement=str(f"Stub: Processing {input_filepath} -> {output_filepath}"), main_logger=str(__name__))
        # Example: Copying for now, replace with real processing
        try:
            # If input is zstd, need to decompress first
            if input_filepath.endswith(".zst"):
                temp_decompressed = input_filepath + ".tmpdec"
                decompress_file(input_filepath, temp_decompressed)
                input_to_process = temp_decompressed
            else:
                input_to_process = input_filepath

            # Simulate processing (e.g., read, modify, write)
            # Replace this with actual streaming processing if possible
            with open(input_to_process, 'rb') as infile, open(output_filepath + ".tmp", 'wb') as outfile:
                shutil.copyfileobj(infile, outfile) # Simplistic copy

            # Compress the result
            compress_file(output_filepath + ".tmp", output_filepath)
            os.remove(output_filepath + ".tmp")

            # Clean up decompressed temp file
            if input_filepath.endswith(".zst"):
                os.remove(temp_decompressed)

        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error in process_file_content for {input_filepath}: {e}"), main_logger=str(__name__))
            # Clean up partial files
            if os.path.exists(output_filepath): os.remove(output_filepath)
            if os.path.exists(output_filepath + ".tmp"): os.remove(output_filepath + ".tmp")
            if input_filepath.endswith(".zst") and os.path.exists(input_filepath + ".tmpdec"):
                os.remove(input_filepath + ".tmpdec")
            raise
        
    # Placeholder for actual tokenization logic called by DataRepository.tokenize_processed_data
    def tokenize_file_content(self, input_processed_zst_filepath: str, output_tokenized_zst_filepath: str):
        # Tokenizes content of a single processed file
        log_statement(loglevel=str("debug"), logstatement=str(f"Stub: Tokenizing {input_processed_zst_filepath} -> {output_tokenized_zst_filepath}"), main_logger=str(__name__))
        # Example: Read decompressed, tokenize, write compressed
        try:
            def tokenization_generator():
                for line in stream_decompress_lines(input_processed_zst_filepath):
                        # Replace with actual tokenizer call
                        tokenized_line = "TOK_" + line.upper().replace(" ", "_")
                        yield tokenized_line

            stream_compress_lines(output_tokenized_zst_filepath, tokenization_generator())

        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Error in tokenize_file_content for {input_processed_zst_filepath}: {e}"), main_logger=str(__name__))
            if os.path.exists(output_tokenized_zst_filepath): os.remove(output_tokenized_zst_filepath)
            raise


        # --- Simple Fix for Original Timezone Error ---
        # This logic is now incorporated into the DataRepository.add_folder method
        # when reading file metadata. Keeping this note for reference.
        # Original issue was in _load_repo:
        # if 'datetime' in dtype: pdf[col] = pdf[col].dt.tz_localize(timezone.utc)
        # Fix: Check if already timezone-aware before localizing.
        # if 'datetime' in dtype:
        #     if pdf[col].dt.tz is None:
        #         pdf[col] = pdf[col].dt.tz_localize(timezone.utc)
        #     elif pdf[col].dt.tz != timezone.utc: # Optional: convert if aware but not UTC
        #         pdf[col] = pdf[col].dt.tz_convert(timezone.utc)