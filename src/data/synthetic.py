# src/data/synthetic.py
"""
Synthetic Data Generation Module

Handles large-scale synthetic data creation using external language models (like Ollama).
Generates data based on specified prompts and formats, saving it to disk.
"""
import requests
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import torch
from datetime import datetime

# Import project configuration and utilities using relative imports
try:
    # Assumes execution from project root or that src is in PYTHONPATH
    from ..utils.config import SyntheticDataConfig, SYNTHETIC_DATA_DIR, DEFAULT_DEVICE
    from ..utils.logger import configure_logging, log_statement
    configure_logging() # Setup logger for this module
    logger = logging.getLogger(__name__)
    log_statement(loglevel=str("info"), logstatement=str("Logger for Synthetic Data Generation module initialized."), main_log=str(__name__))
    log_statement(loglevel=str("info"), logstatement=str("Synthetic data generation module initialized."), main_log=str(__name__))
except ImportError:
    # Fallback for relative import failure (e.g., direct execution)
    try:
        # Attempt import assuming script is run from within src/data/
        from ..utils.config import SyntheticDataConfig, SYNTHETIC_DATA_DIR, DEFAULT_DEVICE
        from ..utils.logger import configure_logging, log_statement
        configure_logging()
    except ImportError:
        # Final fallback: Define dummy config if all imports fail
        log_statement(loglevel=str("error"), logstatement=str("Failed all imports for config/logger. Defining dummy config/logger."), main_log=str(__name__))
        class DummyConfig:
            class SyntheticDataConfig:
                TARGET_SAMPLES = 100
                BATCH_SIZE = 10
                MAX_WORKERS = 16 # Adjusted default
                DATA_FORMAT = 'jsonl'
                OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
                GENERATION_MODEL_NAME = 'dolphin-mistral:latest' # Renamed from EMBED_MODEL_NAME for clarity
                # Added for correction logic
                CORRECTION_MODEL_NAME = 'deepseek-r1:14b' # Model used for fixing errors
                ENABLE_API_CORRECTION = True # Flag to enable/disable API-based correction
                JSON_CORRECTION_RETRIES = 1
                LIST_CORRECTION_RETRIES = 1
                API_TIMEOUT_SECONDS = 4200 # Timeout for API calls
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
        # Define a dummy logger setup function if needed
        if 'setup_logger' not in locals():
             def setup_logger(): pass

# Setup logger using the (potentially dummy) setup function
configure_logging()
logger = logging.getLogger(__name__)

# Define expected structure constants
EXPECTED_INPUT_LEN = 64
EXPECTED_TARGET_LEN = 10
INPUT_RANGE = (-1.0, 1.0)
TARGET_RANGE = (0.0, 1.0)
PAD_VALUE = 0.0

class SyntheticDataGenerator:
    """
    Generates synthetic training data using an external API endpoint (e.g., Ollama).
    Includes optional self-correction steps using the API for JSON format, list length,
    and data type errors before final adjustment and saving.
    """
    def __init__(self):
        """Initializes the SyntheticDataGenerator using settings from config."""
        self.config = SyntheticDataConfig() # Load config class
        self.output_dir = SYNTHETIC_DATA_DIR
        self.adjusted_output_dir = self.output_dir / getattr(self.config, 'ADJUSTED_SUBDIR', 'adjusted_samples')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adjusted_output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = getattr(self.config, 'MAX_WORKERS', 4)
        if self.config.DEVICE:
            self.device = getattr(self.config, 'DEVICE', DEFAULT_DEVICE) # Use default device if not set
        else:
            self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.generated_samples_count = 0
        self.successfully_saved_count = 0
        self.target_samples = self.config.TARGET_SAMPLES
        self.batch_size = self.config.BATCH_SIZE
        self.save_retries = getattr(self.config, 'SAVE_RETRIES', 2)
        self.retry_delay = getattr(self.config, 'RETRY_DELAY_SECONDS', 1)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers if hasattr(self.config, 'MAX_WORKERS') else 16)
        self.data_format = getattr(self.config, 'DATA_FORMAT', 'jsonl') # Default to jsonl if not set   

        # Correction settings
        self.enable_api_correction = getattr(self.config, 'ENABLE_API_CORRECTION', True)
        self.json_correction_retries = getattr(self.config, 'JSON_CORRECTION_RETRIES', 1)
        self.list_correction_retries = getattr(self.config, 'LIST_CORRECTION_RETRIES', 1)
        self.generation_model = self.config.GENERATION_MODEL_NAME
        self.correction_model = getattr(self.config, 'CORRECTION_MODEL_NAME', self.generation_model)
        self.api_timeout = getattr(self.config, 'API_TIMEOUT_SECONDS', 4200)
        self.ollama_endpoint = self.config.OLLAMA_ENDPOINT

        log_statement(loglevel=str("info"), logstatement=str(f"SyntheticDataGenerator initialized: OutputDir='{self.output_dir}', AdjustedOutputDir='{self.adjusted_output_dir}', TargetSamples={self.target_samples}, BatchSize={self.batch_size}"), main_log=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"Using Ollama Endpoint: {self.ollama_endpoint}"), main_log=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"Generation Model: {self.generation_model}, Correction Model: {self.correction_model}"), main_log=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"API Correction Enabled: {self.enable_api_correction}"), main_log=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"Save retries: {self.save_retries}, API Timeout: {self.api_timeout}s"), main_log=str(__name__))

    def _construct_prompt(self) -> str:
        """ Constructs the initial prompt for data generation. """
        # Ensures the prompt requests the structure defined by constants
        return f"""You are a data generation API. Your SOLE function is to generate and output a single, valid JSON object. Generate data conforming to this exact schema and TYPE requirements: {{ "input": <MUST be JSON array (list) containing EXACTLY {EXPECTED_INPUT_LEN} floats between {INPUT_RANGE[0]} and {INPUT_RANGE[1]}>, "target": <MUST be JSON array (list) containing EXACTLY {EXPECTED_TARGET_LEN} floats between {TARGET_RANGE[0]} and {TARGET_RANGE[1]}, ideally summing to 1.0> }} CRITICAL: Output MUST be ONLY the raw JSON object. Do NOT include ```json markers, explanations, or any text before or after the JSON object."""

    def _call_ollama_api(self, prompt: str, context: str, model_name: str, retries: int = 1) -> str | None:
        """
        Calls the Ollama API with a given prompt and handles basic errors/retries.

        Args:
            prompt (str): The prompt to send to the model.
            context (str): Description of the call for logging (e.g., "initial generation", "json correction").
            model_name (str): The name of the Ollama model to use.
            retries (int): Number of times to retry on failure.

        Returns:
            str | None: The raw response text from the API, or None if all attempts fail.
        """
        # Defines payload for Ollama API call
        payload = {
            "model": model_name,
            "prompt": prompt,
            "format": "json", # Request JSON format directly
            "stream": False,
             "options": {
                 "temperature": 0.0 if "generation" in context else 0.5, # Adjust temperature based on context
                 "num_predict": -1, # Let model decide based on context, or set high like 8192
                 "top_p": 1.0,
                 "top_k": 0 # Consider 0 or a higher value like 40
            }
        }
        log_statement(loglevel=str("debug"), logstatement=str(f"API Call [{context}] - Model: {model_name}, Prompt: {prompt[:150]}..."), main_log=str(__name__))

        # Retry loop for API call
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.ollama_endpoint,
                    json=payload,
                    timeout=self.api_timeout,
                    stream=False
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                # Extract text, handling potential JSON structure in response body
                try:
                    response_data = response.json()
                    # Check common keys for Ollama response content
                    if 'response' in response_data:
                         return response_data['response']
                    elif 'message' in response_data and 'content' in response_data['message']:
                         return response_data['message']['content']
                    else:
                         # Fallback to raw text if structure is unexpected
                         log_statement(loglevel=str("warning"), logstatement=str(f"Unexpected JSON structure in API response [{context}]: {response_data}. Using raw text."), main_log=str(__name__))
                         return response.text
                except json.JSONDecodeError:
                    # If response is not JSON, return raw text
                    return response.text

            except requests.exceptions.Timeout:
                log_statement(loglevel=str("warning"), logstatement=str(f"API request [{context}] timed out ({self.api_timeout}s). Attempt {attempt + 1}/{retries + 1}"), main_log=str(__name__))
            except requests.exceptions.RequestException as e:
                # Log HTTP errors or connection errors
                response_text = getattr(e.response, 'text', 'No response body')[:64000] # Limit to 6400 chars
                log_statement(loglevel=str("warning"), logstatement=str(f"API request [{context}] failed: {e}. Attempt {attempt + 1}/{retries + 1}. Response: {response_text}..."), main_log=str(__name__))
            except Exception as e:
                # Catch any other unexpected errors during the request
                log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during API call [{context}]: {e}. Attempt {attempt + 1}/{retries + 1}", exc_info=True), main_log=str(__name__))

            # Wait before retrying
            if attempt < retries:
                time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff

        # Log failure after all retries
        log_statement(loglevel=str("error"), logstatement=str(f"API call [{context}] failed after {retries + 1} attempts."), main_log=str(__name__))
        return None

    def _try_parse_json(self, json_string: str, context: str) -> dict | list | None:
        """
        Attempts to parse a JSON string, performing cleanup and logging errors.

        Args:
            json_string (str): The string potentially containing JSON.
            context (str): Description for logging purposes.

        Returns:
            dict | list | None: The parsed Python object (dict or list) or None if parsing fails.
        """
        # Check for valid input string
        if not json_string or not isinstance(json_string, str):
             log_statement(loglevel=str("warning"), logstatement=str(f"Invalid input for JSON parsing ({context}): Type {type(json_string)}. Content: '{str(json_string)[:100]}...'"), main_log=str(__name__))
             return None
        try:
            # --- Cleanup ---
            processed_string = json_string.strip()
            # Remove common markdown fences
            if processed_string.startswith("```json"): processed_string = processed_string[7:]
            if processed_string.endswith("```"): processed_string = processed_string[:-3]
            processed_string = processed_string.strip()

            # --- Find JSON boundaries ---
            # Attempt to find the outermost '{...}' or '[...]' structure
            start_obj, start_arr = processed_string.find('{'), processed_string.find('[')
            end_obj, end_arr = processed_string.rfind('}'), processed_string.rfind(']')

            start, end = -1, -1
            # Prioritize object if both found and object starts first or no array found
            if start_obj != -1 and end_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                start, end = start_obj, end_obj
            # Else, prioritize array if found
            elif start_arr != -1 and end_arr != -1:
                 start, end = start_arr, end_arr

            # Extract content within boundaries if found
            if start != -1 and end != -1 and end > start:
                processed_string = processed_string[start:end+1]
            elif processed_string.startswith(('{', '[')) and processed_string.endswith(('}', ']')):
                 # Assume it might be okay if boundaries weren't found but starts/ends correctly
                 pass
            else:
                # If no clear boundaries found and doesn't start/end like JSON, likely invalid
                log_statement(loglevel=str("warning"), logstatement=str(f"Could not find valid JSON object/array markers in response ({context}): {processed_string[:200]}..."), main_log=str(__name__))
                # Try parsing anyway, maybe json.loads is more lenient
                # return None # Stricter approach: return None here

            # --- Parsing ---
            return json.loads(processed_string)

        except json.JSONDecodeError as e:
            # Log JSON specific decoding errors
            log_content = processed_string[:500] + ('...' if len(processed_string) > 500 else '')
            log_statement(loglevel=str("warning"), logstatement=str(f"JSON decode error during parsing ({context}): {e}. String was: '{log_content}'"), main_log=str(__name__))
            return None
        except Exception as e:
            # Catch any other unexpected errors during cleanup or parsing
            log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during JSON parsing ({context}): {e}", exc_info=True), main_log=str(__name__))
            return None

    def _correct_json_formatting_via_api(self, bad_json_string: str) -> dict | list | None:
        """
        Attempts to correct malformed JSON using the correction LLM.

        Args:
            bad_json_string (str): The string suspected to be malformed JSON.

        Returns:
            dict | list | None: The parsed, potentially corrected object, or None if correction fails.
        """
        # Check if API correction is enabled in config
        if not self.enable_api_correction:
            log_statement(loglevel=str("debug"), logstatement=str("API correction disabled, skipping JSON format correction."), main_log=str(__name__))
            return None

        log_statement(loglevel=str("warning"), logstatement=str("Attempting JSON correction via API..."), main_log=str(__name__))
        # Construct prompt asking the LLM to fix the JSON
        prompt = f"The following text is supposed to be a single valid JSON object or array but is malformed. Please correct it and output ONLY the valid JSON object or array, nothing else:\n\n```\n{bad_json_string}\n```"

        # Call the LLM API for correction
        corrected_text = self._call_ollama_api(prompt, "json correction", self.correction_model, self.json_correction_retries)

        if corrected_text:
            # Try parsing the LLM's corrected output
            parsed_result = self._try_parse_json(corrected_text, "json correction result")
            if parsed_result is not None: # Check if parsing succeeded (could be dict or list)
                log_statement(loglevel=str("info"), logstatement=str("Successfully corrected JSON format via API."), main_log=str(__name__))
                return parsed_result
            else:
                log_statement(loglevel=str("error"), logstatement=str(f"API correction returned text that still couldn't be parsed as JSON: {corrected_text[:500]}..."), main_log=str(__name__))
        else:
            # Log if the API call itself failed
            log_statement(loglevel=str("error"), logstatement=str("API call for JSON correction failed."), main_log=str(__name__))
        return None # Return None if correction failed or API call failed

    def _correct_list_via_api(self, current_list: list | None, list_name: str, expected_len: int, value_range: tuple[float, float]) -> list | None:
        """
        Attempts to correct list length or type errors using the correction LLM.

        Args:
            current_list (list | None): The list to potentially correct.
            list_name (str): Name of the list ('input' or 'target') for logging/prompts.
            expected_len (int): The target length of the list.
            value_range (tuple[float, float]): The min/max allowed values for items.

        Returns:
            list | None: The corrected list, the original list if no correction needed/possible,
                         or None if correction fails critically.
        """
         # Fallback if API correction is disabled or if the list is None initially
        if not self.enable_api_correction:
            log_statement(loglevel=str("debug"), logstatement=str(f"API correction disabled. Applying local correction for '{list_name}'."), main_log=str(__name__))
            return self._apply_local_list_correction(current_list, expected_len, value_range, list_name)

        # Ensure we have a list to work with, even if empty
        if current_list is None: current_list = []

        # --- Identify Issues ---
        current_len = len(current_list)
        # Find indices of items that are not int or float
        bad_types_indices = [i for i, item in enumerate(current_list) if not isinstance(item, (int, float))]
        is_length_wrong = current_len != expected_len
        has_bad_types = bool(bad_types_indices)

        # --- Construct Correction Prompt ---
        prompt = ""
        context = f"list correction for '{list_name}'"
        correction_needed = False

        if is_length_wrong and not has_bad_types:
             correction_needed = True
             if current_len < expected_len:
                 needed = expected_len - current_len
                 prompt = f"This JSON list named '{list_name}' must contain exactly {expected_len} float items between {value_range[0]} and {value_range[1]}. It currently has only {current_len}. Generate {needed} additional float items within the specified range and append them. Current list: {json.dumps(current_list)}. Output ONLY the single, complete, corrected JSON list with exactly {expected_len} float items."
                 context += " (length too short)"
             else: # current_len > expected_len
                 prompt = f"This JSON list named '{list_name}' must contain exactly {expected_len} float items between {value_range[0]} and {value_range[1]}. It currently has {current_len}. Truncate it to exactly {expected_len} items. Current list: {json.dumps(current_list)}. Output ONLY the single, complete, corrected JSON list with exactly {expected_len} float items."
                 context += " (length too long)"
        elif has_bad_types:
             correction_needed = True
             # Handle type errors, potentially combined with length adjustment
             prompt = f"This JSON list named '{list_name}' must contain exactly {expected_len} float items between {value_range[0]} and {value_range[1]}. The current list (length {current_len}) has non-float items at indices {bad_types_indices}. Replace ONLY the non-float items with valid floats within the specified range. AFTER replacement, ensure the list has exactly {expected_len} float items by truncating excess items or appending new floats in the range if too short. Current list: {json.dumps(current_list)}. Output ONLY the single, complete, corrected JSON list containing exactly {expected_len} float items."
             context += " (type errors)"
        else:
             # List is already correct length and type
             log_statement(loglevel=str("debug"), logstatement=str(f"List '{list_name}' requires no API correction."), main_log=str(__name__))
             # Convert ints to float if present
             return [float(item) for item in current_list]

        # --- Call API for Correction ---
        if correction_needed:
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempting {context} via API..."), main_log=str(__name__))
            corrected_text = self._call_ollama_api(prompt, context, self.correction_model, self.list_correction_retries)

            if corrected_text:
                # Parse the corrected list from the API response
                parsed_list = self._try_parse_json(corrected_text, f"{context} result")
                if isinstance(parsed_list, list):
                    # --- Final Validation of Corrected List ---
                    if len(parsed_list) == expected_len and all(isinstance(item, (int, float)) for item in parsed_list):
                        log_statement(loglevel=str("info"), logstatement=str(f"Successfully corrected list '{list_name}' via API."), main_log=str(__name__))
                        # Ensure all are floats
                        return [float(item) for item in parsed_list]
                    else:
                        # Log error if API correction resulted in unexpected format
                        log_statement(loglevel=str("error"), logstatement=str(f"API correction for '{list_name}' resulted in a list with incorrect length ({len(parsed_list)} has, {expected_len} needed) or non-float types. Applying local correction as fallback."), main_log=str(__name__))
                        # Fall through to local correction
                else:
                     # Log error if API response wasn't a list
                     log_statement(loglevel=str("error"), logstatement=str(f"API correction for '{list_name}' did not return a valid JSON list: {corrected_text[:500]}... Applying local correction."), main_log=str(__name__))
                     # Fall through to local correction
            else:
                # Log error if API call failed
                log_statement(loglevel=str("error"), logstatement=str(f"API call for list correction ('{list_name}') failed. Applying local correction."), main_log=str(__name__))
                # Fall through to local correction
        else:
             # Should have returned earlier if no correction was needed
             pass

        # --- Fallback to Local Correction ---
        # This block is reached if API correction is disabled, fails, or returns invalid data
        return self._apply_local_list_correction(current_list, expected_len, value_range, list_name)

    def _apply_local_list_correction(self, current_list: list | None, expected_len: int, value_range: tuple[float, float], list_name: str) -> list:
        """Applies simple local corrections: type casting, padding, truncating."""
        log_statement(loglevel=str("debug"), logstatement=str(f"Applying local correction to list '{list_name}'."), main_log=str(__name__))
        if current_list is None: current_list = []

        corrected_list = []
        # First pass: Convert valid types to float, replace others with PAD_VALUE
        for item in current_list:
            if isinstance(item, (int, float)):
                # Clamp value to the expected range
                clamped_value = max(value_range[0], min(value_range[1], float(item)))
                corrected_list.append(clamped_value)
            else:
                # Replace non-numeric with PAD_VALUE (could also use 0.0 or mean)
                corrected_list.append(PAD_VALUE)

        # Second pass: Adjust length
        current_len = len(corrected_list)
        if current_len > expected_len:
            log_statement(loglevel=str("warning"), logstatement=str(f"Locally correcting '{list_name}': Truncating list from {current_len} to {expected_len}."), main_log=str(__name__))
            corrected_list = corrected_list[:expected_len]
        elif current_len < expected_len:
            padding_needed = expected_len - current_len
            log_statement(loglevel=str("warning"), logstatement=str(f"Locally correcting '{list_name}': Padding list with {padding_needed} value(s) ({PAD_VALUE})."), main_log=str(__name__))
            corrected_list.extend([PAD_VALUE] * padding_needed)

        return corrected_list

    def _generate_and_correct_sample(self) -> dict | None:
        """
        Generates a sample, attempts JSON correction, list correction, and normalization.

        Returns:
            dict | None: The fully processed sample dictionary or None if any critical step fails.
        """
        # === Step 1: Initial Generation ===
        initial_prompt = self._construct_prompt()
        # Use generation model, allow retries
        raw_response_text = self._call_ollama_api(initial_prompt, "initial generation", self.generation_model, retries=1)

        if not raw_response_text:
            log_statement(loglevel=str("error"), logstatement=str("Failed to get any response from generation API after retries."), main_log=str(__name__))
            return None

        # === Step 2: Parse Initial Response ===
        parsed_sample = self._try_parse_json(raw_response_text, "initial generation response")

        # === Step 3: Attempt JSON Correction (if needed) ===
        if not isinstance(parsed_sample, dict): # Check if it's a dictionary specifically
            log_statement(loglevel=str("warning"), logstatement=str("Initial generation response was not parsed as a valid dictionary."), main_log=str(__name__))
            # Attempt correction using the correction model
            parsed_sample = self._correct_json_formatting_via_api(raw_response_text)
            if not isinstance(parsed_sample, dict):
                log_statement(loglevel=str("error"), logstatement=str("Failed to parse or correct response into a dictionary structure."), main_log=str(__name__))
                return None # Cannot proceed without a dictionary

        # === Step 4: Check Core Keys ===
        if 'input' not in parsed_sample or 'target' not in parsed_sample:
             log_statement(loglevel=str("error"), logstatement=str(f"Essential key 'input' or 'target' missing after parsing/correction. Keys present: {list(parsed_sample.keys())}"), main_log=str(__name__))
             return None

        # === Step 5: Check and Correct 'input' List ===
        # Ensure 'input' is a list (or attempt basic parse if string)
        input_list = self._ensure_list_from_data(parsed_sample.get('input'), 'input', allow_correction=True)
        # Correct length and type issues (via API or locally)
        corrected_input_list = self._correct_list_via_api(input_list, 'input', EXPECTED_INPUT_LEN, INPUT_RANGE)
        if corrected_input_list is None: # Check if correction failed badly (should return list or fallback)
             log_statement(loglevel=str("error"), logstatement=str("Failed to correct 'input' list structure even with fallback."), main_log=str(__name__))
             # Depending on desired strictness, could return None here or proceed with potentially bad data
             # Let's proceed, normalization/saving might handle None gracefully or fail there.
             parsed_sample['input'] = [] # Assign empty list to avoid downstream TypeErrors
        else:
            parsed_sample['input'] = corrected_input_list

        # === Step 6: Check and Correct 'target' List ===
        target_list = self._ensure_list_from_data(parsed_sample.get('target'), 'target', allow_correction=True)
        corrected_target_list = self._correct_list_via_api(target_list, 'target', EXPECTED_TARGET_LEN, TARGET_RANGE)
        if corrected_target_list is None:
             log_statement(loglevel=str("error"), logstatement=str("Failed to correct 'target' list structure even with fallback."), main_log=str(__name__))
             parsed_sample['target'] = []
        else:
             parsed_sample['target'] = corrected_target_list

        # === Step 7: Final Adjustment (Target Normalization) ===
        # Assumes list correction resulted in a list of the correct length containing numbers
        adjusted_sample = self._normalize_target(parsed_sample)
        if adjusted_sample is None:
            log_statement(loglevel=str("error"), logstatement=str("Failed during final target normalization step."), main_log=str(__name__))
            return None # Normalization failure is critical

        log_statement(loglevel=str("debug"), logstatement=str("Successfully generated, corrected, and adjusted sample structure."), main_log=str(__name__))
        return adjusted_sample

    def _ensure_list_from_data(self, data: any, key_name: str, allow_correction: bool = False) -> list | None:
        """
        Checks if data is a list, handles stringified lists minimally, or returns None.

        Args:
            data: The value associated with key_name.
            key_name: The name of the key ('input' or 'target').
            allow_correction: If True, returns None on failure for correction logic to handle.
                              If False, logs warning and returns empty list or None.

        Returns:
            list | None: The list if valid, or None/empty list if invalid.
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, str):
             # Only attempt basic parse if it looks like a list string
             if data.strip().startswith('[') and data.strip().endswith(']'):
                 log_statement(loglevel=str("warning"), logstatement=str(f"Value for '{key_name}' is a string. Attempting basic parse."), main_log=str(__name__))
                 parsed = self._try_parse_json(data, f"secondary parse of '{key_name}'")
                 if isinstance(parsed, list):
                     return parsed
                 else:
                      log_statement(loglevel=str("warning"), logstatement=str(f"Secondary parse of string for '{key_name}' failed."), main_log=str(__name__))
                      return None if allow_correction else [] # Return None for correction, else empty list
             else:
                 log_statement(loglevel=str("warning"), logstatement=str(f"Value for '{key_name}' is a string but doesn't look like a list: {data[:50]}..."), main_log=str(__name__))
                 return None if allow_correction else []
        # If not list or parsable string
        log_statement(loglevel=str("warning"), logstatement=str(f"Value for '{key_name}' is not a list or suitable string (type={type(data)})."), main_log=str(__name__))
        return None if allow_correction else []

    def _normalize_target(self, sample_dict: dict) -> dict | None:
        """
        Normalizes the 'target' list sum to 1.0. Assumes list exists and correction fixed types/length.

        Args:
            sample_dict (dict): The dictionary containing the 'target' list.

        Returns:
            dict | None: The dictionary with the normalized 'target' list, or None on critical failure.
        """
        try:
            target_list = sample_dict.get('target')

            # Check if target is a list and has expected length (should be true after correction)
            if not isinstance(target_list, list):
                 log_statement(loglevel=str("error"), logstatement=str(f"Cannot normalize target: value is not a list (type: {type(target_list)}). Correction likely failed."), main_log=str(__name__))
                 return None
            if len(target_list) != EXPECTED_TARGET_LEN:
                 log_statement(loglevel=str("error"), logstatement=str(f"Cannot normalize target: length is {len(target_list)}, expected {EXPECTED_TARGET_LEN}. Correction likely failed."), main_log=str(__name__))
                 # Option: Apply local correction again here? Or fail. Let's fail for now.
                 return None

            # Ensure all items are numeric, convert non-numeric gracefully (should be rare after correction)
            numeric_target_list = []
            for i, item in enumerate(target_list):
                 if isinstance(item, (int, float)):
                     # Clamp value to target range during conversion
                     numeric_target_list.append(max(TARGET_RANGE[0], min(TARGET_RANGE[1], float(item))))
                 else:
                     log_statement(loglevel=str("warning"), logstatement=str(f"Non-numeric item '{item}' found at index {i} in target list during normalization (post-correction). Replacing with {PAD_VALUE}."), main_log=str(__name__))
                     numeric_target_list.append(PAD_VALUE)

            # --- Normalization Logic ---
            target_sum = sum(numeric_target_list)
            if abs(target_sum) < 1e-9: # Handle sum close to zero
                log_statement(loglevel=str("warning"), logstatement=str("Target list sum is zero after clamping/replacement. Setting uniform distribution."), main_log=str(__name__))
                # Assign uniform probability
                normalized_target = [1.0 / EXPECTED_TARGET_LEN] * EXPECTED_TARGET_LEN
            elif abs(target_sum - 1.0) > 1e-5: # Normalize if sum is not approx 1.0
                log_statement(loglevel=str("debug"), logstatement=str(f"Normalizing 'target' list. Original sum after clamping: {target_sum:.4f}"), main_log=str(__name__))
                # Normalize
                normalized_target = [x / target_sum for x in numeric_target_list]
                # Re-clamp after normalization (optional, but good practice)
                normalized_target = [max(TARGET_RANGE[0], min(TARGET_RANGE[1], x)) for x in normalized_target]
                # Final re-normalization pass if clamping changed the sum significantly
                final_sum = sum(normalized_target)
                if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-9:
                     log_statement(loglevel=str("debug"), logstatement=str(f"Re-normalizing target list after clamping. Sum was: {final_sum:.4f}"), main_log=str(__name__))
                     normalized_target = [x / final_sum for x in normalized_target]
            else:
                # Sum is already close to 1.0
                normalized_target = numeric_target_list

            # Update the dictionary and return
            sample_dict['target'] = normalized_target
            log_statement(loglevel=str("debug"), logstatement=str("Target normalization successful."), main_log=str(__name__))
            return sample_dict

        except Exception as e:
             # Catch any unexpected errors during normalization
             log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during target normalization: {e}", exc_info=True), main_log=str(__name__))
             return None

    def _attempt_save_adjusted_sample(self, sample_dict: dict, batch_index: int, sample_index_in_batch: int) -> bool:
        """
        Attempts to save the fully corrected/adjusted sample dictionary to a JSON file with retries.

        Args:
            sample_dict (dict): The final, processed sample dictionary.
            batch_index (int): The index of the current batch.
            sample_index_in_batch (int): The index of the sample within the batch.

        Returns:
            bool: True if saving was successful, False otherwise.1
        """
        
        # Check for valid input dictionary
        if not sample_dict or not isinstance(sample_dict, dict):
            log_statement(loglevel=str("warning"), logstatement=str(f"Attempted to save an invalid sample dict (empty or wrong type) for batch {batch_index}, sample {sample_index_in_batch}. Skipping."), main_log=str(__name__))
            return False

        # Construct filename within the adjusted output directory
        filename = self.adjusted_output_dir / f"adjusted_batch_{batch_index}_sample_{sample_index_in_batch}.json"
        attempts = 0
        max_attempts = self.save_retries + 1 # Total attempts = initial + retries

        # Retry loop for saving
        while attempts < max_attempts:
            try:
                # Ensure the output directory exists (important in multi-threaded env)
                self.adjusted_output_dir.mkdir(parents=True, exist_ok=True)

                # Write the dictionary to a JSON file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(sample_dict, f, indent=2) # Use indent for readability

                log_statement(loglevel=str("debug"), logstatement=str(f"Successfully saved adjusted sample to {filename.name}"), main_log=str(__name__))
                return True # Save successful

            # --- Exception Handling for Saving ---
            except (IOError, OSError) as io_err:
                # Handle file system errors (permissions, disk full, etc.)
                log_statement(loglevel=str("warning"), logstatement=str(f"Attempt {attempts + 1}/{max_attempts}: I/O error saving adjusted sample {filename.name}: {io_err}"), main_log=str(__name__))
            except TypeError as type_err:
                # Handle errors if data in dict is not JSON serializable (e.g., sets, custom objects)
                # This shouldn't happen if correction/normalization worked, but good to catch.
                log_statement(loglevel=str("error"), logstatement=str(f"Attempt {attempts + 1}/{max_attempts}: Type error saving adjusted sample {filename.name} (data not JSON serializable?): {type_err}", exc_info=False), main_log=str(__name__))
                # No point retrying a TypeError usually, indicates bad data state
                return False # Fail immediately
            except Exception as e:
                # Catch any other unexpected errors during file writing
                log_statement(loglevel=str("error"), logstatement=str(f"Attempt {attempts + 1}/{max_attempts}: Unexpected error saving adjusted sample {filename.name}: {e}", exc_info=True), main_log=str(__name__))

            # --- Retry Logic ---
            attempts += 1
            if attempts < max_attempts:
                log_statement(loglevel=str("info"), logstatement=str(f"Retrying save for {filename.name} in {self.retry_delay} seconds..."), main_log=str(__name__))
                time.sleep(self.retry_delay)
            else:
                # Log failure after all retries
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to save adjusted sample {filename.name} after {max_attempts} attempts."), main_log=str(__name__))

        return False # Return False if all attempts failed

    def _generate_and_save_batch(self, batch_size: int, batch_index: int):
        """
        Generates a batch of samples, attempts full correction cycle for each,
        and saves successful ones individually.

        Args:
            batch_size (int): Number of samples to attempt in this batch.
            batch_index (int): Index of the current batch (for naming files).
        """
        log_statement(loglevel=str("debug"), logstatement=str(f"Generating and saving batch {batch_index} (target size {batch_size})..."), main_log=str(__name__))
        samples_saved_in_batch = 0

        # Loop to generate the required number of samples for the batch
        for i in range(batch_size):
             # Check if overall target has been met to allow early exit
             if self.successfully_saved_count >= self.target_samples:
                 log_statement(loglevel=str("info"), logstatement=str(f"Target sample count ({self.target_samples}) reached during batch {batch_index}. Stopping batch early."), main_log=str(__name__))
                 break

             # Increment count for each generation *attempt*
             self.generated_samples_count += 1
             log_statement(loglevel=str("info"), logstatement=str(f"Attempting generation/correction for sample {self.generated_samples_count}/{self.target_samples} (Batch {batch_index}, Item {i})..."), main_log=str(__name__))

             # --- Core Generation and Correction Cycle ---
             # This method handles initial call, parsing, JSON correction, list correction, normalization
             corrected_sample = self._generate_and_correct_sample()

             if corrected_sample:
                 # If generation and correction were successful, attempt to save
                 save_successful = self._attempt_save_adjusted_sample(
                     corrected_sample, batch_index, i
                 )
                 if save_successful:
                     # Increment success count only if save succeeds
                     self.successfully_saved_count += 1
                     samples_saved_in_batch += 1
                 else:
                     # Log if saving failed after retries
                     log_statement(loglevel=str("error"), logstatement=str(f"Failed to save processed sample {i} for batch {batch_index} after retries."), main_log=str(__name__))
             else:
                  # Log if the generation/correction cycle failed for this sample
                  log_statement(loglevel=str("error"), logstatement=str(f"Failed to generate/correct sample {i} for batch {batch_index}."), main_log=str(__name__))

        # Log completion of the batch
        log_statement(loglevel=str("debug"), logstatement=str(f"Finished batch {batch_index}. Samples successfully saved in this batch: {samples_saved_in_batch}/{batch_size}"), main_log=str(__name__))

    def generate_dataset(self):
        """
        Orchestrates the full synthetic dataset generation process using a thread pool.
        Manages batch submission and progress tracking.
        """
        # Validate configuration before starting
        if self.batch_size <= 0:
            log_statement(loglevel=str("error"), logstatement=str("Batch size must be positive. Aborting generation."), main_log=str(__name__))
            return
        if self.target_samples <= 0:
            log_statement(loglevel=str("info"), logstatement=str("Target samples is zero or negative. No data generation needed."), main_log=str(__name__))
            return

        # Calculate total number of batches needed
        total_batches = (self.target_samples + self.batch_size - 1) // self.batch_size
        log_statement(loglevel=str("info"), logstatement=str(f"Starting synthetic data generation for {self.target_samples} target samples in {total_batches} batches."), main_log=str(__name__))

        futures = []
        # Use tqdm for progress bar, based on target samples saved
        with tqdm(total=self.target_samples, desc="Saving Samples", unit="sample") as pbar:
            # Submit tasks to the thread pool for each batch
            for i in range(total_batches):
                 # Determine actual number of samples needed for this batch (can be less than batch_size for the last batch)
                 # This logic seems slightly off - should submit tasks aiming for target, not fixed batch count?
                 # Let's rethink: submit tasks until target is likely met.
                 # Simpler: Submit based on total_batches, let the internal check stop early.

                 # Calculate samples 'intended' for this batch index (for logging/naming mainly)
                 # The actual number generated might vary if target met early.
                 samples_in_this_batch_run = min(self.batch_size, self.target_samples - (i * self.batch_size))
                 if samples_in_this_batch_run <= 0: continue # Should not happen with ceil division, but safe check

                 # Submit the batch processing task to the executor
                 future = self.executor.submit(self._generate_and_save_batch, batch_size=self.batch_size, batch_index=i)
                 futures.append(future)

            # --- Monitor Progress and Update Bar ---
            last_reported_saves = 0
            # Process completed futures as they finish
            for future in as_completed(futures):
                try:
                    # Call result() to raise any exceptions that occurred within the thread
                    future.result()
                except Exception as e:
                    # Log errors from worker threads
                    log_statement(loglevel=str("error"), logstatement=str(f"Error processing a batch generation future: {e}", exc_info=True), main_log=str(__name__))

                # Update progress bar based on the shared counter
                current_saves = self.successfully_saved_count
                # Ensure progress bar only increases
                if current_saves > last_reported_saves:
                    pbar.update(current_saves - last_reported_saves)
                    last_reported_saves = current_saves

                # Check if target has been met after processing this future
                # (The check inside _generate_and_save_batch provides faster stopping)
                if current_saves >= self.target_samples:
                    log_statement(loglevel=str("info"), logstatement=str("Target sample count met or exceeded based on completed futures."), main_log=str(__name__))
                    # Optional: Cancel remaining futures if precise count is critical
                    # Note: Cancellation is best-effort and might not stop running tasks.
                    for f in futures:
                        if not f.done(): f.cancel()
                    break # Exit as_completed loop once target is met

        # Shutdown executor cleanly, waiting for running tasks
        self.executor.shutdown(wait=True)
        log_statement(loglevel=str("info"), logstatement=str(f"Synthetic data generation finished. Target: {self.target_samples}, Actual Successfully Saved: {self.successfully_saved_count}"), main_log=str(__name__))
        log_statement(loglevel=str("info"), logstatement=str(f"Total generation attempts (including failures/corrections): {self.generated_samples_count}"), main_log=str(__name__))

# Example usage block (usually commented out in production code)
# if __name__ == "__main__":
#     # Ensure logger is configured if running directly
#     # setup_logger() should ideally be called by the main entry point of the application
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     log_statement(loglevel=str("info"), logstatement=str("Running SyntheticDataGenerator directly for testing...")
#     generator = SyntheticDataGenerator()
#
#     # Override config for a small test run
#     generator.target_samples = 1 # Generate only 5 samples
#     generator.batch_size = 1    # Use small batches
#     generator.enable_api_correction = True # Ensure correction is enabled for test
#     log_statement(loglevel=str("info"), logstatement=str(f"Test run configured: Target={generator.target_samples}, BatchSize={generator.batch_size}, Correction={generator.enable_api_correction}")
#
#     # Start the generation process
#     generator.generate_dataset()
#
#     log_statement(loglevel=str("info"), logstatement=str("Direct execution test finished.")

    def _generate_valid_sample(self, retries=3, timeout=90000):
        """
        Attempts to generate a single valid sample by calling the API (non-streaming).
        Handles retries.

        Args:
            retries (int): Number of times to retry the API call on failure.
            timeout (int): Timeout in seconds for the API request.

        Returns:
            dict: A validated data sample dictionary, or None if generation fails.
        """
        current_retry = 0
        prompt_text = self._construct_prompt() # Get prompt once

        if current_retry == 0:
             log_statement(loglevel=str("info"), logstatement=str(f"Prompt for API: {prompt_text[:200]}...."), main_log=str(__name__))

        while current_retry <= retries:
            payload = {
                "model": self.config.EMBED_MODEL_NAME,
                "prompt": prompt_text,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 8192, # <--- INCREASED TOKEN LIMIT SIGNIFICANTLY
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.0
                }
            }
            if current_retry == 0:
                log_statement(loglevel=str("debug"), logstatement=str(f"Payload for API request (Attempt {current_retry+1}): {json.dumps(payload)}"), main_log=str(__name__))

            response = None
            try:
                response = requests.post(
                    self.config.OLLAMA_ENDPOINT,
                    json=payload,
                    timeout=timeout,
                    stream=False
                )
                response.raise_for_status()
                response_data = response.json()
                generated_content_str = response_data.get('response')

                if not generated_content_str:
                    log_statement(loglevel=str("warning"), logstatement=str(f"API response key 'response' is empty or missing. Response dict: {response_data}. Retrying ({current_retry}/{retries})..."), main_log=str(__name__))
                    current_retry += 1
                    time.sleep(1 * (current_retry + 1))
                    continue

                # --- Optional: Log the raw response for debugging ---
                log_statement(loglevel=str("debug"), logstatement=str(f"Raw response content string received:\n{generated_content_str}"), main_log=str(__name__))
                # --- End Optional ---

                sample = self._validate_sample(generated_content_str)
                if sample:
                    log_statement(loglevel=str("debug"), logstatement=str("Successfully generated and validated sample."), main_log=str(__name__))
                    return sample
                else:
                    # Validation failed - _validate_sample logs details including JSONDecodeError
                    log_statement(loglevel=str("warning"), logstatement=str(f"API response content failed validation (see previous log). Retrying ({current_retry}/{retries})..."), main_log=str(__name__))
                    current_retry += 1
                    time.sleep(1 * (current_retry + 1))

            except requests.exceptions.Timeout:
                log_statement(loglevel=str("warning"), logstatement=str(f"API request timed out ({timeout}s). Retrying ({current_retry}/{retries})..."), main_log=str(__name__))
                current_retry += 1
                time.sleep(2 * (current_retry + 1))
            except requests.exceptions.RequestException as e:
                response_text = response.text if response else "NO RESPONSE"
                log_statement(loglevel=str("error"), logstatement=str(f"API request failed: {e}. Response: {response_text[:500]}... Retrying ({current_retry}/{retries})...", exc_info=False), main_log=str(__name__))
                current_retry += 1
                time.sleep(2 * (current_retry + 1))
            except json.JSONDecodeError as json_err:
                response_text = response.text if response else "NO RESPONSE"
                log_statement(loglevel=str("error"), logstatement=str(f"Failed to decode overall API response as JSON: {json_err}. Response text: '{response_text[:500]}...'. Retrying ({current_retry}/{retries})...", exc_info=False), main_log=str(__name__))
                current_retry += 1
                time.sleep(1 * (current_retry + 1))
            except Exception as e:
                log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during sample generation: {e}. Retrying ({current_retry}/{retries})...", exc_info=True), main_log=str(__name__))
                current_retry += 1
                time.sleep(1 * (current_retry + 1))

        log_statement(loglevel=str("error"), logstatement=str(f"Failed to generate valid sample after {retries} retries."), main_log=str(__name__))
        return None

    def _validate_sample(self, response_content: str) -> dict | None:
        """
        Validates the structure and content of the generated JSON string.
        Handles cases where list data might be incorrectly nested within strings.

        Args:
            response_content (str): The string response content from the language model,
                                     expected to be a JSON object string.

        Returns:
            dict: The parsed and validated sample dictionary, or None if invalid.
        """
        try:
            # Attempt to parse the main JSON string
            sample = json.loads(response_content)

            # --- Start of structural/content validation logic ---
            if 'input' not in sample or 'target' not in sample:
                log_statement(loglevel=str("warning"), logstatement=str("Validation failed: Missing 'input' or 'target' key."), main_log=str(__name__))
                return None

            input_data = sample['input']
            target_data = sample['target']

            # --- Check if input_data is a string and try parsing it ---
            if isinstance(input_data, str):
                log_statement(loglevel=str("debug"), logstatement=str("Detected 'input' value is a string. Attempting secondary parse..."), main_log=str(__name__))
                try:
                    input_data = json.loads(input_data) # Try parsing the string as JSON (expecting a list)
                    # Overwrite sample['input'] with the parsed list for consistency if needed later,
                    # but for validation, using the local input_data variable is sufficient.
                    # sample['input'] = input_data
                except json.JSONDecodeError as e:
                    log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'input' was a string but failed secondary parse: {e}. String was: '{input_data[:100]}...'"), main_log=str(__name__))
                    return None

            # --- Check if target_data is a string and try parsing it ---
            if isinstance(target_data, str):
                log_statement(loglevel=str("debug"), logstatement=str("Detected 'target' value is a string. Attempting secondary parse..."), main_log=str(__name__))
                try:
                    target_data = json.loads(target_data) # Try parsing the string as JSON (expecting a list)
                    # sample['target'] = target_data # Optional: update original dict
                except json.JSONDecodeError as e:
                    log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'target' was a string but failed secondary parse: {e}. String was: '{target_data[:100]}...'"), main_log=str(__name__))
                    return None

            # Validate input structure and constraints
            if not isinstance(input_data, list):
                 # If it's still not a list after potential secondary parse, then it's truly invalid
                 log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'input' is not a list (type={type(input_data)})."), main_log=str(__name__))
                 return None
            if len(input_data) != 128:
                log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'input' list length is not 128 (len={len(input_data)})."), main_log=str(__name__))
                return None
            # Check input item constraints
            if not all(isinstance(x, (int, float)) and -1.0 <= x <= 1.0 for x in input_data):
                first_invalid_input = next((x for x in input_data if not (isinstance(x, (int, float)) and -1.0 <= x <= 1.0)), "N/A")
                log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'input' values not all floats/ints between -1.0 and 1.0. First invalid: {first_invalid_input}"), main_log=str(__name__))
                return None

            # Validate target structure and constraints
            if not isinstance(target_data, list):
                log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'target' is not a list (type={type(target_data)})."), main_log=str(__name__))
                return None
            if len(target_data) != 10:
                log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'target' list length is not 10 (len={len(target_data)})."), main_log=str(__name__))
                return None
            # Check target item constraints
            if not all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in target_data):
                 first_invalid_target = next((x for x in target_data if not (isinstance(x, (int, float)) and 0.0 <= x <= 1.0)), "N/A")
                 log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'target' values not all floats/ints between 0.0 and 1.0. First invalid: {first_invalid_target}"), main_log=str(__name__))
                 return None

            # Check sum with tolerance
            target_sum = sum(target_data)
            if not abs(target_sum - 1.0) < 1e-5:
                log_statement(loglevel=str("warning"), logstatement=str(f"Validation failed: 'target' values do not sum to 1.0 (sum={target_sum})."), main_log=str(__name__))
                return None
            # --- End of structural/content validation logic ---

            log_statement(loglevel=str("debug"), logstatement=str("Validation successful."), main_log=str(__name__))
            # Return the original sample dictionary, even if input/target were parsed from strings
            # The downstream code using the 'sample' dict might expect the lists directly.
            # If you updated sample['input']/sample['target'] above, this is fine.
            # If not, you might want to return a *new* dict with the correctly parsed lists:
            # return {'input': input_data, 'target': target_data}
            # Let's return the original 'sample' assuming it might be used elsewhere,
            # and the validation passed using the (potentially re-parsed) local variables.
            # Consider if you need to update the original 'sample' dict in-place if necessary.
            return sample

        except json.JSONDecodeError as e:
            log_content = response_content[:500] + ('...' if len(response_content) > 500 else '')
            log_statement(loglevel=str("error"), logstatement=str(f"JSON decode error during validation: {e}. String was: '{log_content}'"), main_log=str(__name__))
            return None
        except TypeError as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Type error during validation (likely incorrect data structure): {e}"), main_log=str(__name__))
            return None
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error during validation: {e}", exc_info=True), main_log=str(__name__))
            return None
        finally:
            time.sleep(0.1)

    def _save_batch(self, batch_data: list, batch_index: int):
        """Saves a batch of generated samples to a JSONL file."""
        if not batch_data:
            return

        # Use a timestamp and batch index for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"batch_{batch_index}_{timestamp}.{self.config.DATA_FORMAT}"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for sample in batch_data:
                    f.write(json.dumps(sample) + '\n')
            log_statement(loglevel=str("info"), logstatement=str(f"Successfully saved batch {batch_index} ({len(batch_data)} samples) to {filename.name}"), main_log=str(__name__))
            self.generated_samples_count += len(batch_data) # Increment successful count
        except IOError as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Failed to write batch file {filename.name}: {e}", exc_info=True), main_log=str(__name__))
        except Exception as e:
            log_statement(loglevel=str("error"), logstatement=str(f"Unexpected error saving batch {filename.name}: {e}", exc_info=True), main_log=str(__name__))