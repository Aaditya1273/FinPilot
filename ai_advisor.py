import os
import time
import threading
import logging
from typing import Optional, Dict, Any, List, Generator, Union, Callable
from functools import lru_cache, wraps
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    api_key: str = os.getenv("API_KEYS")
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 1024

class PerformanceMonitor:
    def __init__(self):
        self._stats = {"queries": 0, "total_time": 0.0, "errors": 0, "cache_hits": 0}
        self._lock = threading.RLock()
        
    def record_query(self, duration: float, cached: bool = False):
        with self._lock:
            self._stats["queries"] += 1
            self._stats["total_time"] += duration
            if cached:
                self._stats["cache_hits"] += 1
                
    def record_error(self):
        with self._lock:
            self._stats["errors"] += 1
            
    @property
    def metrics(self) -> Dict[str, float]:
        with self._lock:
            total = max(self._stats["queries"], 1)
            return {
                "queries": self._stats["queries"],
                "avg_response_time": self._stats["total_time"] / total,
                "error_rate": self._stats["errors"] / total,
                "cache_hit_rate": self._stats["cache_hits"] / total
            }

def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
            self._monitor.record_query(time.perf_counter() - start)
            return result
        except Exception as e:
            self._monitor.record_error()
            raise e
    return wrapper

class GeminiEngine:
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GeminiEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._config = ModelConfig()
        self._monitor = PerformanceMonitor()
        self._cache_lock = threading.RLock()
        self._model = None
        self._error = None
        self._initialized = True
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            genai.configure(api_key=self._config.api_key)
            self._model = genai.GenerativeModel(self._config.model_name)
            self._error = None
            logger.info(f"Initialized Gemini model: {self._config.model_name}")
        except Exception as e:
            self._error = str(e)
            self._model = None
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
    
    @lru_cache(maxsize=256)
    def _build_prompt(self, prompt: str, system: str = "") -> Dict[str, Any]:
        generation_config = {
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "max_output_tokens": self._config.max_output_tokens,
        }
        
        # Combine system prompt and user prompt if system prompt is provided
        if system:
            content = f"{system}\n\n{prompt}"
        else:
            content = prompt
            
        return {
            "contents": [{"role": "user", "parts": [content]}],
            "generation_config": generation_config
        }
    
    @performance_monitor
    def query(self, prompt: str, system: str = "", **kwargs) -> str:
        if not self._model:
            raise RuntimeError(f"Model unavailable: {self._error}")
        
        # Override default config with any provided kwargs
        generation_config = {
            "temperature": kwargs.get("temperature", self._config.temperature),
            "top_p": kwargs.get("top_p", self._config.top_p),
            "top_k": kwargs.get("top_k", self._config.top_k),
            "max_output_tokens": kwargs.get("max_tokens", self._config.max_output_tokens),
        }
        
        try:
            # Prepare the prompt
            if system:
                content = f"{system}\n\n{prompt}"
            else:
                content = prompt
                
            # Generate response
            response = self._model.generate_content(
                content,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise
    
    def stream(self, prompt: str, system: str = "", **kwargs) -> Generator[str, None, None]:
        if not self._model:
            raise RuntimeError(f"Model unavailable: {self._error}")
        
        # Override default config with any provided kwargs
        generation_config = {
            "temperature": kwargs.get("temperature", self._config.temperature),
            "top_p": kwargs.get("top_p", self._config.top_p),
            "top_k": kwargs.get("top_k", self._config.top_k),
            "max_output_tokens": kwargs.get("max_tokens", self._config.max_output_tokens),
        }
        
        try:
            # Prepare the prompt
            if system:
                content = f"{system}\n\n{prompt}"
            else:
                content = prompt
                
            # Generate streaming response
            response = self._model.generate_content(
                content,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise
    
    def batch_query(self, prompts: List[str], max_concurrent: int = 4, **kwargs) -> List[str]:
        results = [None] * len(prompts)
        
        with threading.ThreadPoolExecutor(max_workers=min(max_concurrent, len(prompts))) as executor:
            future_to_index = {
                executor.submit(self.query, prompt, **kwargs): i 
                for i, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = f"Error: {str(e)}"
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._model is not None,
            "error": self._error,
            "model_name": self._config.model_name,
            "cache_info": self._build_prompt.cache_info()._asdict(),
            "metrics": self._monitor.metrics
        }
    
    def optimize_memory(self):
        with self._cache_lock:
            self._build_prompt.cache_clear()
    
    def reload(self):
        self._initialize_model()
        self.optimize_memory()

# Create a singleton instance
engine = GeminiEngine()

# Public API functions to maintain compatibility with the existing code
def initialize_llm():
    """Initialize the LLM engine"""
    engine.reload()
    return engine.health_check()["model_loaded"]

def get_advice(prompt: str, system_prompt: str = "") -> str:
    """Get financial advice from the LLM"""
    try:
        return engine.query(prompt, system=system_prompt)
    except Exception as e:
        logger.error(f"Error getting advice: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# Additional functions that match the original API
query = engine.query
stream = engine.stream
batch_query = engine.batch_query
health = engine.health_check
optimize = engine.optimize_memory