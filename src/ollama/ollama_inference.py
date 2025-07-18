"""
Ollama LLM Inference Module

Provides functionality for communicating with Ollama to generate
AI-powered commit messages and file analysis.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

# Fallback logging if main logger not available
try:
    from src.utils.logger import log_statement
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def log_statement(level: str, message: str, module: str, exc_info: bool = False):
        logger = logging.getLogger(module)
        getattr(logger, level.lower(), logger.info)(message, exc_info=exc_info)

LOG_PREFIX = "ollama_inference"

class OllamaInferenceError(Exception):
    """Raised when Ollama inference operations fail"""
    pass


class OllamaClient:
    """
    Client for communicating with Ollama API to generate commit messages
    and perform file analysis using specified LLMs.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 default_model: str = "qwen3:14b",
                 default_temperature: float = 0.65,
                 timeout: int = 30):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL
            default_model: Default LLM model to use
            default_temperature: Default temperature for inference
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.timeout = timeout
        
        log_statement('info', f"{LOG_PREFIX}:INFO>>OllamaClient initialized with model {default_model}", 
                     "ollama_inference")
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                log_statement('info', f"{LOG_PREFIX}:INFO>>Ollama connection verified", "ollama_inference")
                return True
            else:
                log_statement('warning', f"{LOG_PREFIX}:WARNING>>Ollama responded with status {response.status_code}", 
                             "ollama_inference")
                return False
        except Exception as e:
            log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to connect to Ollama: {e}", 
                         "ollama_inference")
            return False
    
    def list_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            log_statement('info', f"{LOG_PREFIX}:INFO>>Found {len(models)} available models", "ollama_inference")
            return models
            
        except Exception as e:
            log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to list models: {e}", "ollama_inference")
            return []
    
    def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure specified model is available, attempt to pull if not.
        
        Args:
            model_name: Name of the model to ensure
            
        Returns:
            True if model is available, False otherwise
        """
        available_models = self.list_models()
        
        if model_name in available_models:
            return True
        
        log_statement('info', f"{LOG_PREFIX}:INFO>>Model {model_name} not found, attempting to pull", 
                     "ollama_inference")
        
        try:
            # Attempt to pull the model
            pull_response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model pull
            )
            
            if pull_response.status_code == 200:
                log_statement('info', f"{LOG_PREFIX}:INFO>>Successfully pulled model {model_name}", 
                             "ollama_inference")
                return True
            else:
                log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to pull model {model_name}: {pull_response.status_code}", 
                             "ollama_inference")
                return False
                
        except Exception as e:
            log_statement('error', f"{LOG_PREFIX}:ERROR>>Error pulling model {model_name}: {e}", 
                         "ollama_inference")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         model: Optional[str] = None,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: Input prompt for the model
            model: Model name (uses default if not specified)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if model is None:
            model = self.default_model
        
        if temperature is None:
            temperature = self.default_temperature
        
        # Ensure model is available
        if not self.ensure_model_available(model):
            raise OllamaInferenceError(f"Model {model} is not available and could not be pulled")
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        try:
            log_statement('debug', f"{LOG_PREFIX}:DEBUG>>Generating response with model {model}", 
                         "ollama_inference")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get('response', '').strip()
            
            log_statement('debug', f"{LOG_PREFIX}:DEBUG>>Generated {len(generated_text)} characters", 
                         "ollama_inference")
            
            return generated_text
            
        except Exception as e:
            log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to generate response: {e}", 
                         "ollama_inference", exc_info=True)
            raise OllamaInferenceError(f"Failed to generate response: {e}")


class CommitMessageGenerator:
    """
    Generates intelligent commit messages using Ollama LLM based on file changes.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize commit message generator.
        
        Args:
            ollama_client: OllamaClient instance
        """
        self.ollama_client = ollama_client
        
        log_statement('info', f"{LOG_PREFIX}:INFO>>CommitMessageGenerator initialized", "ollama_inference")
    
    def generate_commit_message(self, 
                               files_changed: List[str],
                               operation_type: str = "update",
                               file_summaries: Optional[Dict[str, str]] = None,
                               batch_number: Optional[int] = None) -> str:
        """
        Generate an intelligent commit message based on changed files.
        
        Args:
            files_changed: List of files that were changed
            operation_type: Type of operation (update, add, remove, etc.)
            file_summaries: Optional summaries of file contents/changes
            batch_number: Optional batch number for this commit
            
        Returns:
            Generated commit message
        """
        try:
            # Create context for the LLM
            context = self._build_commit_context(files_changed, operation_type, file_summaries, batch_number)
            
            # Generate commit message prompt
            prompt = self._build_commit_prompt(context)
            
            # Generate commit message
            commit_message = self.ollama_client.generate_response(
                prompt=prompt,
                max_tokens=200  # Keep commit messages concise
            )
            
            # Clean and validate the commit message
            cleaned_message = self._clean_commit_message(commit_message)
            
            log_statement('info', f"{LOG_PREFIX}:INFO>>Generated commit message: {cleaned_message[:50]}...", 
                         "ollama_inference")
            
            return cleaned_message
            
        except Exception as e:
            log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to generate commit message: {e}", 
                         "ollama_inference")
            # Fallback to simple commit message
            return self._generate_fallback_commit_message(files_changed, operation_type, batch_number)
    
    def _build_commit_context(self, 
                             files_changed: List[str], 
                             operation_type: str,
                             file_summaries: Optional[Dict[str, str]],
                             batch_number: Optional[int]) -> Dict[str, Any]:
        """Build context information for commit message generation."""
        context = {
            "operation_type": operation_type,
            "files_count": len(files_changed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files": []
        }
        
        if batch_number:
            context["batch_number"] = batch_number
        
        # Analyze file types and patterns
        file_extensions = {}
        for file_path in files_changed[:20]:  # Limit to first 20 files for analysis
            path = Path(file_path)
            ext = path.suffix.lower()
            if ext:
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
            
            file_info = {
                "name": path.name,
                "extension": ext,
                "path": str(path.parent) if path.parent != Path('.') else "root"
            }
            
            if file_summaries and file_path in file_summaries:
                file_info["summary"] = file_summaries[file_path]
            
            context["files"].append(file_info)
        
        context["file_extensions"] = file_extensions
        
        return context
    
    def _build_commit_prompt(self, context: Dict[str, Any]) -> str:
        """Build the prompt for commit message generation."""
        files_count = context["files_count"]
        operation_type = context["operation_type"]
        batch_number = context.get("batch_number")
        
        prompt = f"""You are an expert software developer creating Git commit messages. 

CONTEXT:
- Operation: {operation_type}
- Files changed: {files_count}
{"- Batch number: " + str(batch_number) if batch_number else ""}
- File types: {', '.join(f'{ext}({count})' for ext, count in context.get('file_extensions', {}).items())}

FILES (first 20):
"""
        
        for file_info in context["files"][:20]:
            prompt += f"- {file_info['name']} ({file_info['extension'] or 'no ext'}) in {file_info['path']}\n"
            if 'summary' in file_info:
                prompt += f"  Summary: {file_info['summary'][:100]}...\n"
        
        if files_count > 20:
            prompt += f"... and {files_count - 20} more files\n"
        
        prompt += f"""
REQUIREMENTS:
1. Create a concise, professional Git commit message (50-72 characters for subject)
2. Use conventional commit format if appropriate (feat:, fix:, docs:, etc.)
3. Focus on the primary purpose of these changes
4. Be specific but concise
5. Use present tense ("Add" not "Added")

EXAMPLES:
- "feat: process linguistic data for 150 files in batch 3"
- "update: advance linguistic processing pipeline status"
- "feat: complete document analysis for research corpus"

Generate ONLY the commit message (no explanations or additional text):"""
        
        return prompt
    
    def _clean_commit_message(self, raw_message: str) -> str:
        """Clean and validate the generated commit message."""
        # Remove any markdown, quotes, or extra formatting
        cleaned = raw_message.strip().strip('"').strip("'").strip('`')
        
        # Remove any explanatory text after the commit message
        lines = cleaned.split('\n')
        commit_line = lines[0].strip()
        
        # Ensure reasonable length
        if len(commit_line) > 72:
            # Try to truncate at word boundary
            words = commit_line.split()
            truncated = ""
            for word in words:
                if len(truncated + " " + word) <= 69:  # Leave room for "..."
                    truncated += (" " + word) if truncated else word
                else:
                    break
            commit_line = truncated + "..."
        
        # Ensure minimum length
        if len(commit_line) < 10:
            commit_line = "Update: automated batch processing"
        
        return commit_line
    
    def _generate_fallback_commit_message(self, 
                                        files_changed: List[str], 
                                        operation_type: str,
                                        batch_number: Optional[int]) -> str:
        """Generate a simple fallback commit message if LLM fails."""
        files_count = len(files_changed)
        
        if batch_number:
            return f"{operation_type}: process {files_count} files in batch {batch_number}"
        else:
            return f"{operation_type}: process {files_count} files"


# Global client instance
_ollama_client = None
_commit_generator = None


def get_ollama_client(base_url: str = "http://localhost:11434",
                     model: str = "qwen3:14b",
                     temperature: float = 0.65) -> OllamaClient:
    """Get or create global Ollama client instance."""
    global _ollama_client
    
    if _ollama_client is None:
        _ollama_client = OllamaClient(
            base_url=base_url,
            default_model=model,
            default_temperature=temperature
        )
    
    return _ollama_client


def get_commit_generator() -> CommitMessageGenerator:
    """Get or create global commit message generator."""
    global _commit_generator
    
    if _commit_generator is None:
        client = get_ollama_client()
        _commit_generator = CommitMessageGenerator(client)
    
    return _commit_generator


def generate_commit_message_for_batch(files_changed: List[str],
                                     operation_type: str = "update",
                                     batch_number: Optional[int] = None,
                                     model: str = "gemma2:12b",
                                     temperature: float = 0.65) -> str:
    """
    Convenience function to generate commit message for a batch of files.
    
    Args:
        files_changed: List of files that were changed
        operation_type: Type of operation
        batch_number: Batch number
        model: LLM model to use
        temperature: Temperature for generation
        
    Returns:
        Generated commit message
    """
    try:
        # Get client with specified parameters
        client = get_ollama_client(model=model)
        client.default_model = model
        client.default_temperature = temperature
        
        generator = CommitMessageGenerator(client)
        
        return generator.generate_commit_message(
            files_changed=files_changed,
            operation_type=operation_type,
            batch_number=batch_number
        )
        
    except Exception as e:
        log_statement('error', f"{LOG_PREFIX}:ERROR>>Failed to generate commit message: {e}", 
                     "ollama_inference")
        
        # Fallback commit message
        files_count = len(files_changed)
        if batch_number:
            return f"{operation_type}: process {files_count} files in batch {batch_number}"
        else:
            return f"{operation_type}: automated processing of {files_count} files"