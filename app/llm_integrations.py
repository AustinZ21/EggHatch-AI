"""
LLM Integration module for EggHatch AI.

This module provides client wrappers for interacting with LLMs,
specifically Gemma 3 12B via Ollama API.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union, Generator
from pydantic import BaseModel, Field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OllamaResponse(BaseModel):
    """Schema for Ollama API response."""
    response: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model used for generation")
    created_at: str = Field(..., description="Timestamp of response")
    done: bool = Field(..., description="Whether the response is complete")
    error: Optional[str] = Field(None, description="Error message if any")

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, 
                base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), 
                model: str = os.getenv("OLLAMA_MODEL", "gemma3:12b"),
                max_retries: int = int(os.getenv("MAX_RETRIES", "3")),
                retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
            model: Model to use for generation
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.base_url = base_url
        self.model = model
        self.generate_endpoint = f"{base_url}/api/generate"
        self.chat_endpoint = f"{base_url}/api/chat"
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        # Create session with retry strategy
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Ollama API with retry logic.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Returns:
            Response from the Ollama API
            
        Raises:
            requests.exceptions.RequestException: If the request fails after retries
        """
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return {"error": str(e)}
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = float(os.getenv("TEMPERATURE", "0.7")), 
                max_tokens: int = int(os.getenv("MAX_TOKENS", "4096")),
                stream: bool = True) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response from the Ollama API or generator for streaming responses
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_response(self.generate_endpoint, payload)
        else:
            response = self._make_request(self.generate_endpoint, payload)
            return OllamaResponse(**response).dict()
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = float(os.getenv("TEMPERATURE", "0.7")),
                       max_tokens: int = int(os.getenv("MAX_TOKENS", "4096")),
                       stream: bool = True) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a chat completion using the Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response from the Ollama API or generator for streaming responses
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            return self._stream_response(self.chat_endpoint, payload)
        else:
            response = self._make_request(self.chat_endpoint, payload)
            return OllamaResponse(**response).dict()
    
    def _stream_response(self, endpoint: str, payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream responses from the Ollama API.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Yields:
            Response chunks from the Ollama API
        """
        try:
            with self.session.post(endpoint, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield OllamaResponse(**chunk).dict()
                        except json.JSONDecodeError as e:
                            print(f"Error decoding stream chunk: {e}")
                            continue
        except requests.exceptions.RequestException as e:
            print(f"Error streaming from Ollama API: {e}")
            yield {"error": str(e)}

# Example usage
if __name__ == "__main__":
    client = OllamaClient()
    
    # Non-streaming example
    response = client.generate("What is EggHatch AI?", stream=False)
    print("Non-streaming response:", response)
    
    # Streaming example
    print("\nStreaming response:")
    for chunk in client.generate("What is EggHatch AI?", stream=True):
        print(chunk.get("response", ""), end="", flush=True)
