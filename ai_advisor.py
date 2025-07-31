import threading
import time
import asyncio
import weakref
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Generator, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from llama_cpp import Llama
import gc
import psutil
import os

@dataclass
class ModelConfig:
    model_path: str = "./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    n_ctx: int = 8192
    n_batch: int = 1024
    n_gpu_layers: int = -1
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    use_mlock: bool = True
    use_mmap: bool = True
    rope_freq_base: float = 10000.0
    rope_scaling_type: int = 1
    flash_attn: bool = True
    verbose: bool = False

@dataclass
class QueryParams:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    min_p: float = 0.05
    typical_p: float = 1.0
    tfs_z: float = 1.0
    mirostat: int = 0

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
                "cache_hit_rate": self._stats["cache_hits"] / total,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
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

class AdvancedLLMEngine:
    _instances = weakref.WeakValueDictionary()
    _global_lock = threading.RLock()
    _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm_async")
    
    def __new__(cls, config_id: str = "default"):
        with cls._global_lock:
            if config_id not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[config_id] = instance
                instance._initialized = False
            return cls._instances[config_id]
    
    def __init__(self, config_id: str = "default"):
        if self._initialized:
            return
            
        self._config_id = config_id
        self._config = ModelConfig()
        self._params = QueryParams()
        self._model = None
        self._error = None
        self._monitor = PerformanceMonitor()
        self._cache_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._initialized = True
        self._load_model()
    
    def _load_model(self):
        try:
            with self._model_lock:
                if self._model:
                    del self._model
                    gc.collect()
                
                model_config = {
                    "model_path": self._config.model_path,
                    "n_ctx": self._config.n_ctx,
                    "n_batch": self._config.n_batch,
                    "n_gpu_layers": self._config.n_gpu_layers,
                    "n_threads": self._config.n_threads,
                    "use_mlock": self._config.use_mlock,
                    "use_mmap": self._config.use_mmap,
                    "rope_freq_base": self._config.rope_freq_base,
                    "rope_scaling_type": self._config.rope_scaling_type,
                    "flash_attn": self._config.flash_attn,
                    "verbose": self._config.verbose,
                    "embedding": False,
                    "logits_all": False
                }
                
                self._model = Llama(**model_config)
                self._error = None
        except Exception as e:
            self._error = str(e)
            self._model = None
    
    @lru_cache(maxsize=256)
    def _build_prompt(self, prompt: str, system: str = "", format_type: str = "llama3") -> str:
        if format_type == "llama3":
            sys_part = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>" if system else ""
            return f"<|begin_of_text|>{sys_part}<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif format_type == "chatml":
            sys_part = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
            return f"{sys_part}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    @performance_monitor
    def query(self, prompt: str, system: str = "", **kwargs) -> str:
        if not self._model:
            raise RuntimeError(f"Model unavailable: {self._error}")
        
        params = {**self._params.__dict__, **kwargs}
        input_text = self._build_prompt(prompt, system)
        
        generation_params = {
            "prompt": input_text,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repeat_penalty": params["repeat_penalty"],
            "min_p": params["min_p"],
            "typical_p": params["typical_p"],
            "tfs_z": params["tfs_z"],
            "mirostat": params["mirostat"],
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
            "echo": False
        }
        
        with self._model_lock:
            response = self._model(**generation_params)
            return response["choices"][0]["text"].strip()
    
    def stream(self, prompt: str, system: str = "", **kwargs) -> Generator[str, None, None]:
        if not self._model:
            raise RuntimeError(f"Model unavailable: {self._error}")
        
        params = {**self._params.__dict__, **kwargs}
        input_text = self._build_prompt(prompt, system)
        
        generation_params = {
            "prompt": input_text,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repeat_penalty": params["repeat_penalty"],
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
            "echo": False,
            "stream": True
        }
        
        with self._model_lock:
            for chunk in self._model(**generation_params):
                if chunk["choices"][0]["text"]:
                    yield chunk["choices"][0]["text"]
    
    def batch_query(self, prompts: List[str], max_concurrent: int = 4, **kwargs) -> List[str]:
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(prompts))) as executor:
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
    
    async def async_query(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.query, prompt, **kwargs)
    
    @contextmanager
    def configure(self, **config_updates):
        original_config = {k: getattr(self._config, k) for k in config_updates.keys()}
        original_params = {k: getattr(self._params, k) for k in config_updates.keys() if hasattr(self._params, k)}
        
        try:
            for key, value in config_updates.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                elif hasattr(self._params, key):
                    setattr(self._params, key, value)
            
            if any(hasattr(self._config, k) for k in config_updates.keys()):
                self._load_model()
            
            yield self
        finally:
            for key, value in original_config.items():
                setattr(self._config, key, value)
            for key, value in original_params.items():
                setattr(self._params, key, value)
    
    def optimize_memory(self):
        with self._cache_lock:
            self._build_prompt.cache_clear()
            gc.collect()
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._model is not None,
            "error": self._error,
            "config_id": self._config_id,
            "cache_info": self._build_prompt.cache_info()._asdict(),
            "metrics": self._monitor.metrics
        }
    
    def reload(self):
        self._load_model()
        self.optimize_memory()
    
    @classmethod
    def shutdown_all(cls):
        cls._executor.shutdown(wait=True)
        for instance in cls._instances.values():
            if instance._model:
                del instance._model
        cls._instances.clear()
        gc.collect()

llm = AdvancedLLMEngine()
query = llm.query
stream = llm.stream
batch_query = llm.batch_query
async_query = llm.async_query
health = llm.health_check
optimize = llm.optimize_memory