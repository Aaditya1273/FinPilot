import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import threading
import signal
import atexit
from collections import defaultdict, deque
import pickle
import zlib
from io import StringIO
import csv

from flask import Flask, request, jsonify, g, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest, InternalServerError, Unauthorized, TooManyRequests
from werkzeug.middleware.proxy_fix import ProxyFix
import redis
from redis.sentinel import Sentinel
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from marshmallow import Schema, fields, ValidationError, validate, post_load
import jwt
from cryptography.fernet import Fernet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

from financial_engine import calculate_metrics
from ai_advisor import initialize_llm, get_advice

class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class CacheStrategy(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

@dataclass
class PerformanceMetrics:
    request_count: int = 0
    total_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    active_connections: int = 0

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                raise e

class SmartCache:
    def __init__(self, strategy: CacheStrategy = CacheStrategy.HYBRID):
        self.strategy = strategy
        self.memory_cache = {}
        self.cache_stats = defaultdict(int)
        self.expiry_times = {}
        self.access_pattern = defaultdict(deque)
        self._lock = threading.RLock()
        
    def _generate_key(self, data: Dict) -> str:
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _is_expired(self, key: str) -> bool:
        return key in self.expiry_times and time.time() > self.expiry_times[key]
    
    def _update_access_pattern(self, key: str):
        now = time.time()
        self.access_pattern[key].append(now)
        if len(self.access_pattern[key]) > 100:
            self.access_pattern[key].popleft()
    
    def _predict_cache_value(self, key: str) -> float:
        if key not in self.access_pattern:
            return 0.5
        
        accesses = list(self.access_pattern[key])
        if len(accesses) < 2:
            return 0.5
        
        recent_access_frequency = len([t for t in accesses if time.time() - t < 300]) / 300
        total_accesses = len(accesses)
        
        return min(1.0, (recent_access_frequency * 0.7) + (total_accesses / 1000 * 0.3))
    
    def get(self, data: Dict) -> Optional[Any]:
        key = self._generate_key(data)
        
        with self._lock:
            if key in self.memory_cache and not self._is_expired(key):
                self._update_access_pattern(key)
                self.cache_stats['hits'] += 1
                compressed_data = self.memory_cache[key]
                return pickle.loads(zlib.decompress(compressed_data))
            
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, data: Dict, value: Any, ttl: int = 300):
        key = self._generate_key(data)
        cache_value = self._predict_cache_value(key)
        
        if cache_value > 0.3 or self.strategy == CacheStrategy.MEMORY:
            with self._lock:
                compressed_data = zlib.compress(pickle.dumps(value), level=6)
                self.memory_cache[key] = compressed_data
                self.expiry_times[key] = time.time() + ttl
                self._update_access_pattern(key)
    
    def invalidate(self, pattern: str = None):
        with self._lock:
            if pattern:
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    self.expiry_times.pop(key, None)
            else:
                self.memory_cache.clear()
                self.expiry_times.clear()

class AdvancedRateLimiter:
    def __init__(self):
        self.windows = defaultdict(lambda: defaultdict(deque))
        self.user_patterns = defaultdict(lambda: {'requests': 0, 'errors': 0, 'first_seen': time.time()})
        self._lock = threading.RLock()
    
    def is_allowed(self, client_id: str, endpoint: str, limit: int, window: int) -> Tuple[bool, Dict]:
        current_time = time.time()
        
        with self._lock:
            window_requests = self.windows[client_id][endpoint]
            
            while window_requests and current_time - window_requests[0] > window:
                window_requests.popleft()
            
            user_info = self.user_patterns[client_id]
            user_info['requests'] += 1
            
            is_suspicious = self._detect_suspicious_behavior(client_id, current_time)
            adaptive_limit = self._calculate_adaptive_limit(client_id, limit, is_suspicious)
            
            if len(window_requests) >= adaptive_limit:
                user_info['errors'] += 1
                return False, {
                    'allowed': False,
                    'limit': adaptive_limit,
                    'remaining': 0,
                    'reset_time': int(current_time + window),
                    'suspicious': is_suspicious
                }
            
            window_requests.append(current_time)
            return True, {
                'allowed': True,
                'limit': adaptive_limit,
                'remaining': adaptive_limit - len(window_requests),
                'reset_time': int(current_time + window),
                'suspicious': is_suspicious
            }
    
    def _detect_suspicious_behavior(self, client_id: str, current_time: float) -> bool:
        user_info = self.user_patterns[client_id]
        
        if user_info['requests'] > 1000:
            error_rate = user_info['errors'] / user_info['requests']
            if error_rate > 0.5:
                return True
        
        time_active = current_time - user_info['first_seen']
        if time_active > 0:
            request_rate = user_info['requests'] / time_active
            if request_rate > 10:
                return True
        
        return False
    
    def _calculate_adaptive_limit(self, client_id: str, base_limit: int, is_suspicious: bool) -> int:
        if is_suspicious:
            return max(1, base_limit // 4)
        
        user_info = self.user_patterns[client_id]
        if user_info['requests'] > 100 and user_info['errors'] / user_info['requests'] < 0.1:
            return int(base_limit * 1.5)
        
        return base_limit

class DatabaseManager:
    def __init__(self):
        self.pool = None
        self.connection_string = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/fundpilot')
        self._initialize_pool()
    
    def _initialize_pool(self):
        try:
            self.pool = ThreadedConnectionPool(
                minconn=5,
                maxconn=20,
                dsn=self.connection_string
            )
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            self.pool = None
    
    def execute_query(self, query: str, params: Tuple = None) -> Optional[List]:
        if not self.pool:
            return None
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
                conn.commit()
                return []
        except Exception as e:
            logging.error(f"Database query failed: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self.pool.putconn(conn)

class MLPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['users', 'churn', 'ltv', 'cac', 'monthly_expenses']
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            self.model = joblib.load('models/financial_predictor.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
        except Exception:
            logging.warning("ML model not found, using default predictions")
            self.model = None
    
    def predict_metrics(self, data: Dict) -> Dict:
        if not self.model:
            return self._fallback_prediction(data)
        
        try:
            features = np.array([[data[name] for name in self.feature_names]])
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)[0]
            
            return {
                'predicted_revenue': float(prediction * data.get('ltv', 1000)),
                'risk_score': self._calculate_risk_score(data),
                'growth_potential': self._calculate_growth_potential(data),
                'recommendations': self._generate_recommendations(data)
            }
        except Exception as e:
            logging.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data: Dict) -> Dict:
        return {
            'predicted_revenue': data.get('users', 0) * data.get('ltv', 0) * (1 - data.get('churn', 0.1)),
            'risk_score': min(1.0, data.get('churn', 0.1) * 2),
            'growth_potential': max(0.0, 1.0 - data.get('churn', 0.1)),
            'recommendations': ['Optimize user acquisition', 'Reduce churn rate', 'Increase customer lifetime value']
        }
    
    def _calculate_risk_score(self, data: Dict) -> float:
        churn_weight = 0.4
        cac_ltv_ratio = data.get('cac', 0) / max(data.get('ltv', 1), 1)
        expense_ratio = data.get('monthly_expenses', 0) / max(data.get('users', 1) * data.get('ltv', 1), 1)
        
        risk = (data.get('churn', 0) * churn_weight + 
                min(1.0, cac_ltv_ratio) * 0.3 + 
                min(1.0, expense_ratio) * 0.3)
        return min(1.0, risk)
    
    def _calculate_growth_potential(self, data: Dict) -> float:
        ltv_cac_ratio = data.get('ltv', 0) / max(data.get('cac', 1), 1)
        low_churn_bonus = max(0, 0.3 - data.get('churn', 0.3))
        efficiency_score = min(1.0, ltv_cac_ratio / 3.0)
        
        return min(1.0, efficiency_score + low_churn_bonus)
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        recommendations = []
        
        if data.get('churn', 0) > 0.2:
            recommendations.append("High churn detected - implement retention strategies")
        
        if data.get('cac', 0) / max(data.get('ltv', 1), 1) > 0.5:
            recommendations.append("CAC to LTV ratio is concerning - optimize acquisition costs")
        
        if data.get('monthly_expenses', 0) / max(data.get('users', 1), 1) > 100:
            recommendations.append("High operational costs per user - review expense structure")
        
        return recommendations or ["Metrics look healthy - continue current strategy"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fundpilot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

CORS(app, 
     origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key'],
     expose_headers=['X-RateLimit-Remaining', 'X-Response-Time'])

try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        max_connections=20
    )
    redis_client.ping()
    CACHE_ENABLED = True
except:
    redis_client = None
    CACHE_ENABLED = False
    logger.warning("Redis unavailable, using memory cache")

executor = ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 8)), thread_name_prefix="FundPilot")
smart_cache = SmartCache(CacheStrategy.HYBRID if CACHE_ENABLED else CacheStrategy.MEMORY)
rate_limiter = AdvancedRateLimiter()
db_manager = DatabaseManager()
ml_predictor = MLPredictor()
circuit_breaker = CircuitBreaker()
performance_metrics = PerformanceMetrics()

class AdvancedCalculationSchema(Schema):
    users = fields.Integer(required=True, validate=validate.Range(min=1, max=10000000))
    churn = fields.Float(required=True, validate=validate.Range(min=0.001, max=0.999))
    ltv = fields.Float(required=True, validate=validate.Range(min=1, max=1000000))
    monthly_expenses = fields.Float(required=True, validate=validate.Range(min=0, max=10000000))
    initial_capital = fields.Float(required=True, validate=validate.Range(min=0, max=100000000))
    cac = fields.Float(required=True, validate=validate.Range(min=0.01, max=10000))
    market_segment = fields.String(missing='B2B', validate=validate.OneOf(['B2B', 'B2C', 'B2B2C']))
    industry = fields.String(missing='Technology')
    prediction_horizon = fields.Integer(missing=12, validate=validate.Range(min=1, max=60))
    include_ml_predictions = fields.Boolean(missing=True)
    
    @post_load
    def validate_business_logic(self, data, **kwargs):
        if data['cac'] > data['ltv']:
            logger.warning(f"CAC ({data['cac']}) exceeds LTV ({data['ltv']})")
        return data

class EnhancedAdviceSchema(Schema):
    prompt = fields.String(required=True, validate=validate.Length(min=10, max=2000))
    context = fields.Dict(missing={})
    priority = fields.Enum(Priority, missing=Priority.NORMAL)
    user_id = fields.String(missing=lambda: str(uuid.uuid4()))
    session_id = fields.String(missing=lambda: str(uuid.uuid4()))
    language = fields.String(missing='en', validate=validate.OneOf(['en', 'es', 'fr', 'de']))
    max_response_length = fields.Integer(missing=500, validate=validate.Range(min=100, max=2000))

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        valid_keys = os.getenv('API_KEYS', 'dev-key-123,admin-key-456').split(',')
        if api_key not in valid_keys:
            return jsonify({'error': 'Invalid API key'}), 401
        
        g.api_key = api_key
        return f(*args, **kwargs)
    return decorated_function

@app.route('/v2/export', methods=['POST'])
@require_api_key
def export_data():
    """Export financial model data as CSV or JSON."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    export_format = data.get('format', 'json').lower()
    model_data = data.get('model_data')

    if not model_data or 'projections' not in model_data or 'summary_metrics' not in model_data:
        return jsonify({"error": "Incomplete or invalid model data provided"}), 400

    if export_format == 'csv':
        try:
            projections = model_data['projections']
            summary = model_data['summary_metrics']
            
            si = StringIO()
            cw = csv.writer(si)
            
            cw.writerow(['Metric', 'Value'])
            for key, value in summary.items():
                cw.writerow([key.replace('_', ' ').title(), value])
            cw.writerow([])
            
            cw.writerow(['Month'] + list(projections.keys()))
            
            num_months = len(next(iter(projections.values())))
            for i in range(num_months):
                row = [f'Month {i+1}'] + [projections[key][i] for key in projections.keys()]
                cw.writerow(row)
            
            output = si.getvalue()
            return Response(
                output,
                mimetype='text/csv',
                headers={"Content-disposition": "attachment; filename=financial_model.csv"}
            )
        except Exception as e:
            logging.error(f"CSV export failed: {e}")
            return jsonify({"error": "Failed to generate CSV file."}), 500

    elif export_format == 'json':
        return jsonify(model_data)
    
    else:
        return jsonify({"error": "Unsupported format. Please choose 'json' or 'csv'."}), 400

def log_request_to_db(endpoint: str, data: Dict, response_time: float, status_code: int):
    if db_manager.pool:
        try:
            query = """
                INSERT INTO request_logs (endpoint, request_data, response_time, status_code, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """
            db_manager.execute_query(query, (
                endpoint, json.dumps(data), response_time, status_code, datetime.utcnow()
            ))
        except Exception as e:
            logger.error(f"Failed to log request: {e}")

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({
        'error': 'Validation failed',
        'details': e.messages,
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 400

@app.errorhandler(429)
def handle_rate_limit(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests, please slow down',
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 429

def validate_and_extract(schema, data):
    try:
        return schema.load(data)
    except ValidationError as e:
        raise BadRequest(f"Validation error: {e.messages}")

async def async_calculate_enhanced_metrics(data):
    try:
        basic_results = await asyncio.get_event_loop().run_in_executor(
            executor, calculate_metrics, data
        )
        
        if data.get('include_ml_predictions', True):
            ml_results = await asyncio.get_event_loop().run_in_executor(
                executor, ml_predictor.predict_metrics, data
            )
            basic_results.update(ml_results)
        
        return basic_results
    except Exception as e:
        logger.error(f"Enhanced calculation failed: {e}")
        raise

@app.route('/v2/calculate', methods=['POST'])
@require_api_key
@advanced_rate_limit('calculate', limit=30, window=60)
def handle_advanced_calculation():
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    request_data = request.get_json()
    data = validate_and_extract(calculation_schema, request_data)
    
    cached_result = smart_cache.get(data)
    if cached_result:
        performance_metrics.cache_hits += 1
        log_request_to_db('/v2/calculate', data, time.time() - g.start_time, 200)
        return jsonify({
            'success': True,
            'data': cached_result,
            'cached': True,
            'timestamp': int(time.time()),
            'request_id': g.request_id,
            'processing_time': f"{time.time() - g.start_time:.3f}s"
        }), 200
    
    performance_metrics.cache_misses += 1
    
    try:
        def calculate_with_circuit_breaker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_calculate_enhanced_metrics(data))
            finally:
                loop.close()
        
        results = circuit_breaker.call(calculate_with_circuit_breaker)
        smart_cache.set(data, results, ttl=300)
        
        log_request_to_db('/v2/calculate', data, time.time() - g.start_time, 200)
        
        return jsonify({
            'success': True,
            'data': results,
            'cached': False,
            'timestamp': int(time.time()),
            'request_id': g.request_id,
            'processing_time': f"{time.time() - g.start_time:.3f}s",
            'ml_enhanced': data.get('include_ml_predictions', True)
        }), 200
    
    except Exception as e:
        logger.error(f"Calculation error [{g.request_id}]: {str(e)}")
        log_request_to_db('/v2/calculate', data, time.time() - g.start_time, 500)
        raise InternalServerError(f"Calculation failed: {str(e)}")

@app.route('/v2/get-advice', methods=['POST'])
@require_api_key
@advanced_rate_limit('advice', limit=10, window=60)
def handle_enhanced_advice():
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    request_data = request.get_json()
    data = validate_and_extract(advice_schema, request_data)
    
    if not initialize_ai_models():
        raise InternalServerError("AI models not available")
    
    try:
        advice = get_advice(data['prompt'])
        
        log_request_to_db('/v2/get-advice', data, time.time() - g.start_time, 200)
        
        return jsonify({
            'success': True,
            'advice': advice,
            'context': data.get('context', {}),
            'priority': data['priority'].name,
            'language': data['language'],
            'user_id': data['user_id'],
            'session_id': data['session_id'],
            'timestamp': int(time.time()),
            'request_id': g.request_id,
            'processing_time': f"{time.time() - g.start_time:.3f}s"
        }), 200
    
    except Exception as e:
        logger.error(f"Advice generation error [{g.request_id}]: {str(e)}")
        log_request_to_db('/v2/get-advice', data, time.time() - g.start_time, 500)
        raise InternalServerError(f"Advice generation failed: {str(e)}")

@app.route('/health', methods=['GET'])
def comprehensive_health_check():
    health_status = {
        'status': 'healthy',
        'timestamp': int(time.time()),
        'version': '2.0.0',
        'uptime': int(time.time()),
        'request_id': getattr(g, 'request_id', 'health-check'),
        'services': {
            'redis': CACHE_ENABLED,
            'database': db_manager.pool is not None,
            'ai_models': initialize_ai_models(),
            'ml_predictor': ml_predictor.model is not None,
            'circuit_breaker': circuit_breaker.state
        },
        'performance': {
            'total_requests': performance_metrics.request_count,
            'error_rate': performance_metrics.error_count / max(performance_metrics.request_count, 1),
            'avg_response_time': performance_metrics.total_response_time / max(performance_metrics.request_count, 1),
            'cache_hit_rate': performance_metrics.cache_hits / max(performance_metrics.cache_hits + performance_metrics.cache_misses, 1),
            'active_connections': performance_metrics.active_connections
        }
    }
    
    status_code = 200
    if not all(health_status['services'].values()):
        health_status['status'] = 'degraded'
        status_code = 206
    
    return jsonify(health_status), status_code

@app.route('/metrics', methods=['GET'])
@require_api_key
def get_comprehensive_metrics():
    metrics = {
        'timestamp': int(time.time()),
        'performance': performance_metrics.__dict__,
        'cache_stats': dict(smart_cache.cache_stats),
        'memory_usage': len(smart_cache.memory_cache),
        'rate_limiting': {
            'active_windows': len(rate_limiter.windows),
            'tracked_users': len(rate_limiter.user_patterns)
        }
    }
    
    if CACHE_ENABLED and redis_client:
        try:
            redis_info = redis_client.info()
            metrics['redis'] = {
                'memory_usage': redis_info.get('used_memory_human', 'N/A'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'commands_processed': redis_info.get('total_commands_processed', 0)
            }
        except Exception as e:
            metrics['redis'] = {'error': str(e)}
    
    return jsonify(metrics), 200

@app.route('/admin/cache/clear', methods=['POST'])
@require_api_key
def clear_cache():
    pattern = request.json.get('pattern') if request.is_json else None
    smart_cache.invalidate(pattern)
    
    if CACHE_ENABLED and redis_client:
        try:
            if pattern:
                keys = redis_client.keys(f"*{pattern}*")
                if keys:
                    redis_client.delete(*keys)
            else:
                redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis cache clear failed: {e}")
    
    return jsonify({
        'success': True,
        'message': f"Cache cleared{f' for pattern: {pattern}' if pattern else ''}",
        'timestamp': int(time.time())
    }), 200

def graceful_shutdown(signum, frame):
    logger.info("Initiating graceful shutdown...")
    executor.shutdown(wait=True)
    if db_manager.pool:
        db_manager.pool.closeall()
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

@app.route('/admin/analytics', methods=['GET'])
@require_api_key
def get_analytics_dashboard():
    """Advanced analytics endpoint providing deep insights into API usage patterns"""
    time_range = request.args.get('range', '24h')
    
    # Calculate time boundaries
    current_time = time.time()
    if time_range == '1h':
        start_time = current_time - 3600
    elif time_range == '24h':
        start_time = current_time - 86400
    elif time_range == '7d':
        start_time = current_time - 604800
    elif time_range == '30d':
        start_time = current_time - 2592000
    else:
        start_time = current_time - 86400  # Default to 24h
    
    analytics_data = {
        'time_range': time_range,
        'start_time': int(start_time),
        'end_time': int(current_time),
        'overview': {
            'total_requests': performance_metrics.request_count,
            'error_rate': performance_metrics.error_count / max(performance_metrics.request_count, 1),
            'avg_response_time': performance_metrics.total_response_time / max(performance_metrics.request_count, 1),
            'cache_hit_rate': performance_metrics.cache_hits / max(performance_metrics.cache_hits + performance_metrics.cache_misses, 1),
            'active_connections': performance_metrics.active_connections
        },
        'rate_limiting': {
            'blocked_requests': sum(user['errors'] for user in rate_limiter.user_patterns.values()),
            'suspicious_users': len([uid for uid, data in rate_limiter.user_patterns.items() 
                                   if rate_limiter._detect_suspicious_behavior(uid, current_time)]),
            'top_clients': dict(list(sorted(
                [(uid, data['requests']) for uid, data in rate_limiter.user_patterns.items()],
                key=lambda x: x[1], reverse=True
            ))[:10])
        },
        'cache_performance': {
            'memory_cache_size': len(smart_cache.memory_cache),
            'cache_stats': dict(smart_cache.cache_stats),
            'most_accessed_patterns': dict(list(sorted(
                [(key, len(accesses)) for key, accesses in smart_cache.access_pattern.items()],
                key=lambda x: x[1], reverse=True
            ))[:10])
        },
        'circuit_breaker': {
            'state': circuit_breaker.state,
            'failure_count': circuit_breaker.failure_count,
            'last_failure': circuit_breaker.last_failure_time
        }
    }
    
    # Add database analytics if available
    if db_manager.pool:
        try:
            query_stats = db_manager.execute_query("""
                SELECT endpoint, COUNT(*) as request_count, AVG(response_time) as avg_time
                FROM request_logs 
                WHERE timestamp >= %s 
                GROUP BY endpoint 
                ORDER BY request_count DESC
            """, (datetime.fromtimestamp(start_time),))
            
            if query_stats:
                analytics_data['database'] = {
                    'endpoint_stats': [
                        {'endpoint': row[0], 'requests': row[1], 'avg_time': float(row[2])}
                        for row in query_stats
                    ]
                }
        except Exception as e:
            analytics_data['database'] = {'error': str(e)}
    
    return jsonify(analytics_data), 200

@app.route('/admin/users', methods=['GET'])
@require_api_key
def get_user_analytics():
    """Get detailed user behavior analytics"""
    current_time = time.time()
    
    user_analytics = []
    for user_id, data in rate_limiter.user_patterns.items():
        user_info = {
            'user_id': user_id,
            'total_requests': data['requests'],
            'error_count': data['errors'],
            'error_rate': data['errors'] / max(data['requests'], 1),
            'first_seen': datetime.fromtimestamp(data['first_seen']).isoformat(),
            'time_active': current_time - data['first_seen'],
            'request_rate': data['requests'] / max(current_time - data['first_seen'], 1),
            'is_suspicious': rate_limiter._detect_suspicious_behavior(user_id, current_time),
            'risk_score': min(1.0, (data['errors'] / max(data['requests'], 1)) * 2)
        }
        user_analytics.append(user_info)
    
    # Sort by risk score descending
    user_analytics.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return jsonify({
        'users': user_analytics,
        'summary': {
            'total_users': len(user_analytics),
            'suspicious_users': len([u for u in user_analytics if u['is_suspicious']]),
            'high_risk_users': len([u for u in user_analytics if u['risk_score'] > 0.5]),
            'avg_requests_per_user': sum(u['total_requests'] for u in user_analytics) / max(len(user_analytics), 1)
        },
        'timestamp': int(current_time)
    }), 200

@app.route('/admin/optimize', methods=['POST'])
@require_api_key
def optimize_system():
    """Perform system optimization tasks"""
    optimization_tasks = request.json.get('tasks', []) if request.is_json else []
    results = {}
    
    if 'cache_cleanup' in optimization_tasks or not optimization_tasks:
        # Remove expired cache entries
        current_time = time.time()
        expired_keys = [key for key, expiry in smart_cache.expiry_times.items() 
                       if expiry < current_time]
        
        for key in expired_keys:
            smart_cache.memory_cache.pop(key, None)
            smart_cache.expiry_times.pop(key, None)
        
        results['cache_cleanup'] = {
            'expired_keys_removed': len(expired_keys),
            'cache_size_after': len(smart_cache.memory_cache)
        }
    
    if 'rate_limit_cleanup' in optimization_tasks or not optimization_tasks:
        # Clean old rate limiting data
        current_time = time.time()
        old_users = [uid for uid, data in rate_limiter.user_patterns.items() 
                    if current_time - data['first_seen'] > 86400 and data['requests'] < 10]
        
        for uid in old_users:
            rate_limiter.user_patterns.pop(uid, None)
            rate_limiter.windows.pop(uid, None)
        
        results['rate_limit_cleanup'] = {
            'old_users_removed': len(old_users),
            'active_users_remaining': len(rate_limiter.user_patterns)
        }
    
    if 'circuit_breaker_reset' in optimization_tasks:
        # Reset circuit breaker if it's been open for too long
        if (circuit_breaker.state == 'OPEN' and 
            circuit_breaker.last_failure_time and 
            time.time() - circuit_breaker.last_failure_time > 300):
            circuit_breaker.state = 'CLOSED'
            circuit_breaker.failure_count = 0
            results['circuit_breaker_reset'] = {'status': 'reset_successful'}
        else:
            results['circuit_breaker_reset'] = {'status': 'no_reset_needed'}
    
    return jsonify({
        'success': True,
        'optimization_results': results,
        'timestamp': int(time.time())
    }), 200

@app.route('/v2/batch-calculate', methods=['POST'])
@require_api_key
@advanced_rate_limit('batch_calculate', limit=5, window=60)
def handle_batch_calculation():
    """Handle multiple calculations in a single request"""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    request_data = request.get_json()
    calculations = request_data.get('calculations', [])
    
    if not calculations or len(calculations) > 10:
        raise BadRequest("Must provide 1-10 calculations")
    
    results = []
    start_time = time.time()
    
    # Process calculations in parallel
    def process_single_calculation(calc_data):
        try:
            validated_data = calculation_schema.load(calc_data)
            
            # Check cache first
            cached_result = smart_cache.get(validated_data)
            if cached_result:
                return {'success': True, 'data': cached_result, 'cached': True}
            
            # Calculate if not cached
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(async_calculate_enhanced_metrics(validated_data))
                smart_cache.set(validated_data, result, ttl=300)
                return {'success': True, 'data': result, 'cached': False}
            finally:
                loop.close()
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(len(calculations), 5)) as batch_executor:
        future_to_calc = {batch_executor.submit(process_single_calculation, calc): i 
                         for i, calc in enumerate(calculations)}
        
        for future in as_completed(future_to_calc):
            calc_index = future_to_calc[future]
            try:
                result = future.result()
                results.append({'index': calc_index, **result})
            except Exception as e:
                results.append({'index': calc_index, 'success': False, 'error': str(e)})
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    processing_time = time.time() - start_time
    successful_calcs = len([r for r in results if r['success']])
    
    log_request_to_db('/v2/batch-calculate', request_data, processing_time, 200)
    
    return jsonify({
        'success': True,
        'results': results,
        'summary': {
            'total_calculations': len(calculations),
            'successful': successful_calcs,
            'failed': len(calculations) - successful_calcs,
            'processing_time': f"{processing_time:.3f}s"
        },
        'timestamp': int(time.time()),
        'request_id': g.request_id
    }), 200

@app.route('/v2/export', methods=['GET'])
@require_api_key
@advanced_rate_limit('export', limit=3, window=300)
def export_data():
    """Export analytics data in various formats"""
    export_format = request.args.get('format', 'json').lower()
    data_type = request.args.get('type', 'analytics').lower()
    time_range = request.args.get('range', '24h')
    
    if export_format not in ['json', 'csv']:
        raise BadRequest("Format must be 'json' or 'csv'")
    
    if data_type not in ['analytics', 'users', 'performance']:
        raise BadRequest("Type must be 'analytics', 'users', or 'performance'")
    
    # Generate export data based on type
    if data_type == 'analytics':
        export_data = {
            'performance_metrics': performance_metrics.__dict__,
            'cache_stats': dict(smart_cache.cache_stats),
            'rate_limiting_stats': {
                'total_users': len(rate_limiter.user_patterns),
                'suspicious_users': len([uid for uid, data in rate_limiter.user_patterns.items() 
                                       if rate_limiter._detect_suspicious_behavior(uid, time.time())])
            }
        }
    elif data_type == 'users':
        current_time = time.time()
        export_data = [
            {
                'user_id': uid,
                'requests': data['requests'],
                'errors': data['errors'],
                'first_seen': datetime.fromtimestamp(data['first_seen']).isoformat(),
                'is_suspicious': rate_limiter._detect_suspicious_behavior(uid, current_time)
            }
            for uid, data in rate_limiter.user_patterns.items()
        ]
    else:  # performance
        export_data = {
            'timestamp': int(time.time()),
            'metrics': performance_metrics.__dict__,
            'circuit_breaker_state': circuit_breaker.state,
            'active_connections': performance_metrics.active_connections
        }
    
    if export_format == 'json':
        response = jsonify(export_data)
        response.headers['Content-Disposition'] = f'attachment; filename=fundpilot_{data_type}_{int(time.time())}.json'
        return response
    
    else:  # CSV format
        if data_type == 'users' and isinstance(export_data, list):
            import io
            import csv
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            
            csv_data = output.getvalue()
            output.close()
            
            response = Response(csv_data, mimetype='text/csv')
            response.headers['Content-Disposition'] = f'attachment; filename=fundpilot_{data_type}_{int(time.time())}.csv'
            return response
        else:
            raise BadRequest("CSV format only available for 'users' data type")

# Add a startup task to initialize ML models
@app.before_first_request
def startup_tasks():
    """Perform startup initialization tasks"""
    logger.info("FundPilot API starting up...")
    
    # Initialize AI models
    initialize_ai_models()
    
    # Warm up cache with common calculations
    sample_data = {
        'users': 1000,
        'churn': 0.1,
        'ltv': 500,
        'monthly_expenses': 10000,
        'initial_capital': 100000,
        'cac': 50
    }
    
    try:
        validated_sample = calculation_schema.load(sample_data)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_calculate_enhanced_metrics(validated_sample))
            smart_cache.set(validated_sample, result, ttl=3600)
            logger.info("Cache warmed up with sample calculation")
        finally:
            loop.close()
    except Exception as e:
        logger.warning(f"Cache warmup failed: {e}")
    
    logger.info("FundPilot API startup complete")

# Register cleanup function
atexit.register(lambda: executor.shutdown(wait=True))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting FundPilot API on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Redis enabled: {CACHE_ENABLED}")
    logger.info(f"Database enabled: {db_manager.pool is not None}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
    # Calculate time bounda