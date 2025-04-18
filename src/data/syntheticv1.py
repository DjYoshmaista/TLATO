# src/data/synthetic.py
"""
Synthetic Data Generation Module

Handles large-scale synthetic data creation using external language models (like Ollama).
Generates data based on specified prompts and formats, saving it to disk.
"""
# src/data/synthetic.py

import requests
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import logging
import time
import torch
from datetime import datetime
import numpy as np
import sys

# Import project configuration and utilities using relative imports
try:
    from ..utils.config import SyntheticDataConfig, SYNTHETIC_DATA_DIR
    from ..utils.logger import setup_logger # Assuming logger setup is handled by main script
except ImportError:
    # Fallback for direct execution or testing? Less ideal.
    print("Failed relative import in synthetic.py, trying absolute from src...")
    try:
        from src.utils.config import SyntheticDataConfig, SYNTHETIC_DATA_DIR, DEFAULT_DEVICE
        from src.utils.logger import setup_logger
    except ImportError:
        print("Fallback imports also failed. Defining dummy config/logger.")
        # Define dummy config and logger setup if all imports fail
        logging.basicConfig(level=logging.INFO)
        class DummyConfig:
            class SyntheticDataConfig:
                TARGET_SAMPLES = 100
                BATCH_SIZE = 10
                MAX_WORKERS = 2
                DATA_FORMAT = 'jsonl'
                OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
                EMBED_MODEL_NAME = 'dummy_model'
                # Added for retry logic
                SAVE_RETRIES = 2
                RETRY_DELAY_SECONDS = 1
                # Added for adjusted output path
                ADJUSTED_SUBDIR = 'adjusted_samples'
            SYNTHETIC_DATA_DIR = Path('./data/synthetic')
            DEFAULT_DEVICE = 'cpu' # Sensible default
        config = DummyConfig
        SyntheticDataConfig = config.SyntheticDataConfig
        SYNTHETIC_DATA_DIR = config.SYNTHETIC_DATA_DIR
        DEFAULT_DEVICE = config.DEFAULT_DEVICE
        def setup_logger(): pass # Dummy setup

logger = logging.getLogger(__name__)

# Define expected lengths directly here or read from config if needed
EXPECTED_INPUT_LEN = 64
EXPECTED_TARGET_LEN = 10
PAD_VALUE = 0.0

class SyntheticDataGenerator:
    """
    Generates synthetic training data using an external API endpoint (e.g., Ollama).

    Uses a ThreadPoolExecutor for concurrent API requests to speed up generation.
    Validates the structure and constraints of the generated data before saving.
    Handles streaming JSON responses.
    """

    def __init__(self):
        """Initializes the SyntheticDataGenerator using settings from config."""
        self.config = SyntheticDataConfig() # Load config class
        self.output_dir = SYNTHETIC_DATA_DIR
        # Define path for adjusted samples
        self.adjusted_output_dir = self.output_dir / getattr(self.config, 'ADJUSTED_SUBDIR', 'adjusted_samples')
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure base output directory exists
        self.adjusted_output_dir.mkdir(parents=True, exist_ok=True) # Ensure adjusted output directory exists

        # Use ThreadPoolExecutor for I/O bound tasks (API requests/saving)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.generated_samples_count = 0 # Track *attempted* generations
        self.successfully_saved_count = 0 # Track successfully saved samples
        self.target_samples = self.config.TARGET_SAMPLES
        self.batch_size = self.config.BATCH_SIZE
        self.save_retries = getattr(self.config, 'SAVE_RETRIES', 2)
        self.retry_delay = getattr(self.config, 'RETRY_DELAY_SECONDS', 1)

        logger.info(f"SyntheticDataGenerator initialized: OutputDir='{self.output_dir}', AdjustedOutputDir='{self.adjusted_output_dir}', TargetSamples={self.target_samples}, BatchSize={self.batch_size}")
        logger.info(f"Using Ollama Endpoint: {self.config.OLLAMA_ENDPOINT} with model: {self.config.EMBED_MODEL_NAME}")
        logger.info(f"Save retries: {self.save_retries}, Retry delay: {self.retry_delay}s")

    def __del__(self):  
        """Cleans up the ThreadPoolExecutor on deletion."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down.")
        else:
            logger.warning("ThreadPoolExecutor was not initialized or already shut down.")
            
    def __enter__(self):
        """Enables context manager support for the generator."""
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        """Cleans up the ThreadPoolExecutor on exit."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down.")
        else:
            logger.warning("ThreadPoolExecutor was not initialized or already shut down.")


    def _get_device(self):
        """
        Returns the device to use for computation.
        Uses GPU if available, otherwise falls back to CPU.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU.")
        return device

    def _construct_prompt(self) -> str:
        """
        Constructs the prompt to send to the language model.
        Requests a shorter input list (64 floats).
        """
        # Simplified request: 64 input floats instead of 128
        INPUT_LIST_LENGTH = 64 # Define length here
        TARGET_LIST_LENGTH = 10 # Define length here

        return f"""You are a data generation API. Your SOLE function is to generate and output a single, valid JSON object. Generate data conforming to this exact schema and TYPE requirements: {{ "input": <MUST be JSON array (list) containing EXACTLY {EXPECTED_INPUT_LEN} floats between -1.0 and 1.0>, "target": <MUST be JSON array (list) containing EXACTLY {EXPECTED_TARGET_LEN} floats between 0.0 and 1.0, summing to 1.0> }} CRITICAL: Output MUST be ONLY the raw JSON object. The list lengths MUST be exactly {EXPECTED_INPUT_LEN} for 'input' and {EXPECTED_TARGET_LEN} for 'target'. Do NOT include ```json markers, explanations, or any text before or after the JSON object."""

    def _try_parse_json(self, json_string: str, context: str) -> dict | None:
        """Attempts to parse a JSON string, logging errors."""
        if not json_string or not isinstance(json_string, str):
             logger.warning(f"Invalid JSON string received for parsing ({context}): '{str(json_string)[:100]}...'")
             return None
        try:
            # Attempt to strip markdown fences if present
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            json_string = json_string.strip()
            # Find the first '{' and last '}'
            start = json_string.find('{')
            end = json_string.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_string = json_string[start:end+1]
            else:
                logger.warning(f"Could not find JSON object markers '{{' or '}}' in response ({context}): {json_string[:200]}...")
                return None

            return json.loads(json_string)
        except json.JSONDecodeError as e:
            log_content = json_string[:500] + ('...' if len(json_string) > 500 else '')
            logger.error(f"JSON decode error during validation ({context}): {e}. String was: '{log_content}'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing ({context}): {e}", exc_info=True)
            return None

    def _adjust_data(self, sample_dict: dict) -> dict | None:
        """
        Adjusts list lengths and normalizes target values in the sample dictionary.
        Handles potential TypeErrors during processing.

        Args:
            sample_dict (dict): The dictionary parsed from the JSON response.

        Returns:
            dict | None: The adjusted dictionary, or None if a critical error occurs
                         (e.g., missing keys, non-numeric data preventing normalization).
        """
        try:
            # --- Input List Adjustment ---
            input_list = self._ensure_list_from_data(sample_dict.get('input'), 'input')
            if input_list is None:
                logger.error("Adjustment failed: 'input' key missing or value is not a list/parsable string.")
                return None # Critical error

            current_input_len = len(input_list)
            if current_input_len != EXPECTED_INPUT_LEN:
                logger.warning(f"Correcting 'input' list length. Expected: {EXPECTED_INPUT_LEN}, Got: {current_input_len}.")
                if current_input_len > EXPECTED_INPUT_LEN:
                    input_list = input_list[:EXPECTED_INPUT_LEN]
                else:
                    input_list.extend([PAD_VALUE] * (EXPECTED_INPUT_LEN - current_input_len))
                sample_dict['input'] = input_list # Update the dict

            # --- Target List Adjustment ---
            target_list = self._ensure_list_from_data(sample_dict.get('target'), 'target')
            if target_list is None:
                logger.error("Adjustment failed: 'target' key missing or value is not a list/parsable string.")
                return None # Critical error

            current_target_len = len(target_list)
            if current_target_len != EXPECTED_TARGET_LEN:
                logger.warning(f"Correcting 'target' list length. Expected: {EXPECTED_TARGET_LEN}, Got: {current_target_len}.")
                if current_target_len > EXPECTED_TARGET_LEN:
                    target_list = target_list[:EXPECTED_TARGET_LEN]
                else:
                    target_list.extend([PAD_VALUE] * (EXPECTED_TARGET_LEN - current_target_len))
                sample_dict['target'] = target_list # Update the dict

            # --- Target Normalization ---
            # Ensure all items are numbers before summing/normalizing
            numeric_target_list = []
            for i, item in enumerate(target_list):
                 if isinstance(item, (int, float)):
                     numeric_target_list.append(float(item)) # Convert ints to float
                 else:
                     logger.warning(f"Non-numeric item found in target list at index {i}: '{item}'. Replacing with 0.0 for normalization.")
                     numeric_target_list.append(0.0)

            target_sum = sum(numeric_target_list)
            if abs(target_sum) < 1e-9: # Avoid division by zero if all items were non-numeric or zero
                logger.warning("Target list sum is zero after filtering non-numeric. Cannot normalize. Setting to uniform distribution.")
                # Assign uniform probability, respecting the length constraint
                normalized_target = [1.0 / EXPECTED_TARGET_LEN] * EXPECTED_TARGET_LEN
            elif abs(target_sum - 1.0) > 1e-5: # Normalize if sum is not close to 1.0
                logger.debug(f"Normalizing 'target' list. Original sum: {target_sum:.4f}")
                normalized_target = [x / target_sum for x in numeric_target_list]
                # Optional: Clip values to ensure they are still within [0, 1] after normalization (can happen with negative inputs)
                normalized_target = [max(0.0, min(1.0, x)) for x in normalized_target]
                # Re-normalize after clipping if needed (or accept slight deviation)
                final_sum = sum(normalized_target)
                if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-9:
                     normalized_target = [x / final_sum for x in normalized_target]
            else:
                normalized_target = numeric_target_list # Already normalized (or close enough)

            sample_dict['target'] = normalized_target # Update with normalized list
            logger.debug("Data adjustment (length correction, target normalization) successful.")
            return sample_dict

        except TypeError as te:
             logger.error(f"Type error during data adjustment: {te}. Sample: {str(sample_dict)[:500]}...", exc_info=True)
             return None
        except KeyError as ke:
             logger.error(f"Key error during data adjustment: Missing key {ke}. Sample: {str(sample_dict)[:500]}...", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"Unexpected error during data adjustment: {e}. Sample: {str(sample_dict)[:500]}...", exc_info=True)
             return None

    def _attempt_save_adjusted_sample(self, sample_dict: dict, batch_index: int, sample_index_in_batch: int) -> bool:
        """
        Attempts to save the adjusted sample dictionary to a JSON file with retries.

        Args:
            sample_dict (dict): The adjusted sample dictionary.
            batch_index (int): The index of the batch this sample belongs to.
            sample_index_in_batch (int): The index of this sample within the batch.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not sample_dict:
            logger.warning(f"Attempted to save an empty sample dict for batch {batch_index}, sample {sample_index_in_batch}. Skipping.")
            return False

        filename = self.adjusted_output_dir / f"adjusted_batch_{batch_index}_sample_{sample_index_in_batch}.json"
        attempts = 0
        max_attempts = self.save_retries + 1 # Total attempts = initial + retries

        while attempts < max_attempts:
            try:
                # Ensure the directory exists (might be created by another thread)
                self.adjusted_output_dir.mkdir(parents=True, exist_ok=True)

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(sample_dict, f, indent=2) # Save as formatted JSON

                logger.debug(f"Successfully saved adjusted sample to {filename.name}")
                return True # Save successful

            except (IOError, OSError) as io_err:
                logger.warning(f"Attempt {attempts + 1}/{max_attempts}: I/O error saving adjusted sample {filename.name}: {io_err}")
            except TypeError as type_err:
                # This might happen if adjustment resulted in non-serializable data
                logger.error(f"Attempt {attempts + 1}/{max_attempts}: Type error saving adjusted sample {filename.name} (data might not be JSON serializable): {type_err}", exc_info=True)
                # No point retrying a TypeError usually
                return False # Fail immediately on TypeError
            except Exception as e:
                logger.error(f"Attempt {attempts + 1}/{max_attempts}: Unexpected error saving adjusted sample {filename.name}: {e}", exc_info=True)

            # If we reach here, an error occurred (except TypeError)
            attempts += 1
            if attempts < max_attempts:
                logger.info(f"Retrying save in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            else:
                logger.error(f"Failed to save adjusted sample {filename.name} after {max_attempts} attempts.")

        return False # Failed after all retries

    def _generate_valid_sample_from_api(self, retries=3, timeout=900) -> dict | None:
        """
        Attempts to generate a single sample by calling the API and parsing the JSON.
        Handles retries for API calls and JSON parsing.

        Args:
            retries (int): Number of times to retry the API call on failure.
            timeout (int): Timeout in seconds for the API request.

        Returns:
            dict: A parsed data sample dictionary, or None if generation/parsing fails.
        """
        current_retry = 0
        prompt_text = self._construct_prompt() # Get prompt once
        payload = None # Define payload outside the loop to prevent UnboundLocalError

        if current_retry == 0: # Log prompt only once
             logger.debug(f"Prompt for API: {prompt_text[:200]}....")

        while current_retry <= retries:
            payload = { # Define payload inside loop to ensure it's fresh if retrying
                "model": self.config.EMBED_MODEL_NAME,
                "prompt": prompt_text,
                "format": "json", # Request JSON format directly if supported
                "stream": False, # Non-streaming for simpler handling
                 "options": {
                    "temperature": 0.0,
                    "num_predict": 8192, # Increased token limit
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.0
                }
            }
            logger.debug(f"Payload for API request (Attempt {current_retry + 1}/{retries + 1}): {json.dumps(payload)}")
            response = None
            try:
                response = requests.post(
                    self.config.OLLAMA_ENDPOINT,
                    json=payload,
                    timeout=timeout,
                    stream=False
                )
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

                # Attempt to parse the JSON response
                response_data = response.json()
                generated_content_str = response_data.get('response')

                # Try parsing the content string as JSON
                parsed_sample = self._try_parse_json(generated_content_str, "API response content")
                if parsed_sample is not None and isinstance(parsed_sample, dict):
                    logger.debug("Successfully generated and parsed sample from API.")
                    return parsed_sample
                else:
                    logger.warning(f"API response content was not valid JSON or not a dict. Response: '{str(generated_content_str)[:500]}...'. Retrying ({current_retry}/{retries})...")
                    # Fall through to retry

            except requests.exceptions.Timeout:
                logger.warning(f"API request timed out ({timeout}s). Retrying ({current_retry}/{retries})...")
            except requests.exceptions.RequestException as e:
                response_text = response.text if response else "NO RESPONSE"
                logger.error(f"API request failed: {e}. Response: {response_text[:500]}... Retrying ({current_retry}/{retries})...", exc_info=False)
            except json.JSONDecodeError as json_err:
                # Error decoding the *outer* API response, not the 'response' field content
                response_text = response.text if response else "NO RESPONSE"
                logger.error(f"Failed to decode overall API response as JSON: {json_err}. Response text: '{response_text[:500]}...'. Retrying ({current_retry}/{retries})...", exc_info=False)
            except Exception as e:
                logger.error(f"Unexpected error during API call or parsing: {e}. Retrying ({current_retry}/{retries})...", exc_info=True)

            # If loop continues, it means an error occurred
            current_retry += 1
            if current_retry <= retries:
                 time.sleep(1 * (current_retry + 1)) # Simple backoff

        logger.error(f"Failed to generate and parse valid sample from API after {retries + 1} attempts.")
        return None


    def _generate_and_save_batch(self, batch_size: int, batch_index: int):
        """
        Generates a batch of samples, adjusts them, and saves them individually.
        """
        logger.debug(f"Generating and saving batch {batch_index} ({batch_size} samples)...")
        samples_saved_in_batch = 0
        max_generation_attempts = batch_size * 5 # Allow more attempts than samples needed

        for i in range(batch_size): # Aim for 'batch_size' successful saves
             if self.successfully_saved_count >= self.target_samples:
                 logger.info("Target sample count reached during batch generation. Stopping early.")
                 break

             # Attempt to generate and parse a single sample from the API
             parsed_sample = self._generate_valid_sample_from_api()
             self.generated_samples_count += 1 # Count generation attempt

             if parsed_sample:
                 # Attempt to adjust the data (length correction, normalization)
                 adjusted_sample = self._adjust_data(parsed_sample)

                 if adjusted_sample:
                     # Attempt to save the adjusted sample with retries
                     save_successful = self._attempt_save_adjusted_sample(
                         adjusted_sample, batch_index, i
                     )
                     if save_successful:
                         self.successfully_saved_count += 1
                         samples_saved_in_batch += 1
                     else:
                         logger.error(f"Failed to save adjusted sample {i} for batch {batch_index} after retries.")
                 else:
                     logger.error(f"Failed to adjust sample {i} for batch {batch_index}.")
             else:
                 logger.error(f"Failed to generate/parse sample {i} for batch {batch_index} from API after retries.")

        logger.debug(f"Finished batch {batch_index}. Samples saved in this batch: {samples_saved_in_batch}")


    def generate_dataset(self):
        """
        Orchestrates the full synthetic dataset generation process.
        Submits batch generation tasks to the thread pool executor.
        Tracks progress using tqdm based on successfully saved samples.
        """
        if self.batch_size <= 0:
            logger.error("Batch size must be positive. Aborting generation.")
            return
        if self.target_samples <= 0:
            logger.info("Target samples is zero or negative. No data to generate.")
            return

        total_batches = (self.target_samples + self.batch_size - 1) // self.batch_size
        logger.info(f"Starting synthetic data generation for {self.target_samples} samples in {total_batches} batches.")

        futures = []
        # Use tqdm based on target samples, update manually
        with tqdm(total=self.target_samples, desc="Saving Samples", unit="sample") as pbar:
            # Submit all batch tasks
            for i in range(total_batches):
                samples_in_batch = min(self.batch_size, self.target_samples - (i * self.batch_size))
                if samples_in_batch <= 0: break
                future = self.executor.submit(self._generate_and_save_batch, batch_size=samples_in_batch, batch_index=i)
                futures.append(future)

            # Wait for futures and update progress bar based on actual saves
            # Need a way to track progress across threads - using instance variable `successfully_saved_count`
            last_reported_saves = 0
            for future in as_completed(futures):
                try:
                    future.result() # Check for exceptions from the batch task
                except Exception as e:
                    logger.error(f"Error processing a batch future: {e}", exc_info=True)

                # Update progress bar with the delta since last check
                current_saves = self.successfully_saved_count
                pbar.update(current_saves - last_reported_saves)
                last_reported_saves = current_saves

                # Check if target is met after each future completes
                if current_saves >= self.target_samples:
                    logger.info("Target sample count met or exceeded. Shutting down remaining tasks.")
                    # Cancel remaining futures if possible (though they might already be running)
                    # for f in futures:
                    #     if not f.done():
                    #         f.cancel() # Note: cancellation might not be effective if task already running
                    break # Exit the as_completed loop


        self.executor.shutdown(wait=True) # Ensure all threads finish cleanly
        logger.info(f"Synthetic data generation finished. Target: {self.target_samples}, Actual Successfully Saved: {self.successfully_saved_count}")
        logger.info(f"Total samples attempted: {self.generated_samples_count}, Total successfully saved: {self.successfully_saved_count}")
        logger.info(f"Generated samples saved in: {self.output_dir.name} and {self.adjusted_output_dir.name}")