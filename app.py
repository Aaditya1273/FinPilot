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
from dotenv import load_dotenv

from flask import Flask, request, jsonify, g, Response, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError, Unauthorized, TooManyRequests
from werkzeug.middleware.proxy_fix import ProxyFix

# Optional imports with fallbacks for Vercel compatibility
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: flask_limiter not available. Rate limiting disabled.")
    LIMITER_AVAILABLE = False

try:
    import redis
    from redis.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: redis not available. Caching disabled.")
    REDIS_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: psycopg2 not available. Database disabled.")
    POSTGRES_AVAILABLE = False

try:
    from marshmallow import Schema, fields, ValidationError, validate, post_load
    MARSHMALLOW_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: marshmallow not available. Advanced validation disabled.")
    MARSHMALLOW_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: PyJWT not available. JWT authentication disabled.")
    JWT_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: cryptography not available. Encryption disabled.")
    CRYPTO_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: numpy/pandas not available. Advanced analytics disabled.")
    NUMPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scikit-learn not available. ML features disabled.")
    SKLEARN_AVAILABLE = False

# Import your custom modules with error handling
try:
    from financial_engine import calculate_metrics, calculate_mrr_arr, calculate_burn_rate_runway, calculate_ltv_cac
except ImportError:
    print("⚠️  Warning: financial_engine module not found. Using fallback functions.")
    def calculate_metrics(data):
        """Comprehensive financial metrics calculation"""
        # Extract input values
        users = data.get('users', 0)
        ltv = data.get('ltv', 0)  # ARPU
        cac = data.get('cac', 0)
        churn = data.get('churn', 0.05)
        monthly_expenses = data.get('monthly_expenses', 0)
        initial_capital = data.get('initial_capital', 0)
        monthly_growth = data.get('monthly_growth', 0) / 100  # Convert percentage to decimal
        cogs_percent = data.get('cogs', 20) / 100  # Convert percentage to decimal
        
        # Calculate core metrics
        mrr = users * ltv  # Monthly Recurring Revenue
        arr = mrr * 12  # Annual Recurring Revenue
        
        # Calculate LTV (actual lifetime value, not just ARPU)
        actual_ltv = ltv / max(churn, 0.001)  # LTV = ARPU / Churn Rate
        
        # LTV/CAC ratio
        ltv_cac_ratio = actual_ltv / max(cac, 1)
        
        # Burn rate and runway
        burn_rate = monthly_expenses
        runway = initial_capital / max(burn_rate, 1) if burn_rate > 0 else float('inf')
        
        # Break even users
        break_even_users = monthly_expenses / max(ltv * (1 - cogs_percent), 1)
        
        # Churn loss
        churn_loss = mrr * churn
        
        # Growth efficiency and potential
        growth_efficiency = (monthly_growth * 100) / max(cac / ltv, 1) if cac > 0 else 0
        growth_potential = min(100, (ltv_cac_ratio / 3) * 50 + (1 - churn) * 50)
        
        # Payback period (in months)
        payback_period = cac / max(ltv * (1 - cogs_percent), 1)
        
        # Profitability
        gross_profit_margin = 1 - cogs_percent
        profitability = (mrr * gross_profit_margin - monthly_expenses) / max(mrr, 1)
        
        # Risk score calculation
        risk_score = min(100, (churn * 40) + (min(1, cac/actual_ltv) * 30) + (min(1, burn_rate/max(initial_capital, 1)) * 30))
        
        # Predicted revenue (12-month projection)
        predicted_revenue = mrr * 12 * (1 + monthly_growth) ** 12
        
        # Generate 12-month projections
        months = [f"Month {i+1}" for i in range(12)]
        mrr_projection = []
        capital_projection = []
        current_mrr = mrr
        current_capital = initial_capital
        
        for month in range(12):
            # Apply growth and churn
            current_mrr = current_mrr * (1 + monthly_growth - churn)
            current_capital = max(0, current_capital - burn_rate)
            
            mrr_projection.append(round(current_mrr, 2))
            capital_projection.append(round(current_capital, 2))
        
        return {
            # Core metrics
            "mrr": round(mrr, 2),
            "arr": round(arr, 2),
            "ltv": round(actual_ltv, 2),
            "cac": round(cac, 2),
            "ltv_cac_ratio": round(ltv_cac_ratio, 2),
            "burn_rate": round(burn_rate, 2),
            "runway": round(runway, 2) if runway != float('inf') else "Infinite",
            "break_even_users": round(break_even_users, 2),
            "churn_loss": round(churn_loss, 2),
            "growth_efficiency": round(growth_efficiency, 2),
            "growth_potential": round(growth_potential, 2),
            "payback_period": round(payback_period, 2),
            "profitability": round(profitability, 4),
            "risk_score": round(risk_score, 2),
            "predicted_revenue": round(predicted_revenue, 2),
            
            # Projections for charts
            "projections": {
                "months": months,
                "mrr_projection": mrr_projection,
                "capital_projection": capital_projection
            }
        }
    
    def calculate_mrr_arr(data):
        return {"mrr": data.get('mrr', 0), "arr": data.get('mrr', 0) * 12}
    
    def calculate_burn_rate_runway(data):
        return {"burn_rate": data.get('burn_rate', 0), "runway": data.get('runway', 0)}
    
    def calculate_ltv_cac(data):
        return {"ltv": data.get('ltv', 0), "cac": data.get('cac', 0), "ratio": data.get('ltv', 0) / max(data.get('cac', 1), 1)}

try:
    from ai_advisor import initialize_llm, get_advice
except ImportError:
    print("⚠️  Warning: ai_advisor module not found. Using fallback functions.")
    def initialize_llm():
        return True
    
    def get_advice(prompt):
        return f"AI advice for: {prompt[:50]}... (Fallback response - AI module not available)"

def initialize_ai_models():
    """Initialize AI models for advice generation"""
    try:
        return initialize_llm()
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        return False

load_dotenv()  # Load environment variables from .env file

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
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except Exception as e:
            # Fallback key generation
            return str(hash(str(data)))[:16]
    
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
        try:
            key = self._generate_key(data)
            
            with self._lock:
                if key in self.memory_cache and not self._is_expired(key):
                    self._update_access_pattern(key)
                    self.cache_stats['hits'] += 1
                    compressed_data = self.memory_cache[key]
                    return pickle.loads(zlib.decompress(compressed_data))
                
                self.cache_stats['misses'] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, data: Dict, value: Any, ttl: int = 300):
        try:
            key = self._generate_key(data)
            cache_value = self._predict_cache_value(key)
            
            if cache_value > 0.3 or self.strategy == CacheStrategy.MEMORY:
                with self._lock:
                    compressed_data = zlib.compress(pickle.dumps(value), level=6)
                    self.memory_cache[key] = compressed_data
                    self.expiry_times[key] = time.time() + ttl
                    self._update_access_pattern(key)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
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
            features = np.array([[data.get(name, 0) for name in self.feature_names]])
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
        users = data.get('users', 0)
        ltv = data.get('ltv', 0)
        churn = data.get('churn', 0.1)
        
        return {
            'predicted_revenue': users * ltv * (1 - churn),
            'risk_score': min(1.0, churn * 2),
            'growth_potential': max(0.0, 1.0 - churn),
            'recommendations': self._generate_recommendations(data)
        }
    
    def _calculate_risk_score(self, data: Dict) -> float:
        churn_weight = 0.4
        cac = data.get('cac', 0)
        ltv = data.get('ltv', 1)
        cac_ltv_ratio = cac / max(ltv, 1)
        
        users = data.get('users', 1)
        monthly_expenses = data.get('monthly_expenses', 0)
        expense_ratio = monthly_expenses / max(users * ltv, 1)
        
        risk = (data.get('churn', 0) * churn_weight + 
                min(1.0, cac_ltv_ratio) * 0.3 + 
                min(1.0, expense_ratio) * 0.3)
        return min(1.0, risk)
    
    def _calculate_growth_potential(self, data: Dict) -> float:
        ltv = data.get('ltv', 0)
        cac = data.get('cac', 1)
        ltv_cac_ratio = ltv / max(cac, 1)
        
        churn = data.get('churn', 0.3)
        low_churn_bonus = max(0, 0.3 - churn)
        efficiency_score = min(1.0, ltv_cac_ratio / 3.0)
        
        return min(1.0, efficiency_score + low_churn_bonus)
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        recommendations = set()
        ltv = data.get('ltv', 0)
        cac = data.get('cac', 1)
        churn = data.get('churn', 0)
        monthly_expenses = data.get('monthly_expenses', 0)
        initial_capital = data.get('initial_capital', 0)

        ltv_cac_ratio = ltv / max(cac, 1)
        if ltv_cac_ratio < 3.0:
            recommendations.add("Improve LTV/CAC ratio. Aim for a ratio of at least 3:1 by increasing customer lifetime value or reducing acquisition costs.")
        if ltv_cac_ratio < 1.0:
            recommendations.add("Your LTV/CAC ratio is below 1, meaning you're losing money on each new customer. Immediately focus on reducing CAC or increasing LTV.")

        if churn > 0.05: # 5% monthly churn
            recommendations.add("High churn rate detected. Focus on customer retention strategies to reduce churn and improve long-term stability.")

        if 'summary_metrics' in data and 'runway' in data['summary_metrics']:
            runway = data['summary_metrics']['runway']
            if runway < 12:
                recommendations.add("Your financial runway is less than 12 months. Prioritize extending your runway by increasing revenue or reducing expenses.")
            if runway < 6:
                recommendations.add("Critical Alert: Runway is less than 6 months. Take immediate action to secure funding or drastically cut costs to avoid running out of capital.")

        if cac > ltv:
            recommendations.add("Customer Acquisition Cost is higher than Lifetime Value. Re-evaluate your marketing channels and acquisition strategy.")

        if not recommendations:
            recommendations.add("Your key metrics look healthy. Focus on scaling your current strategies and exploring new growth opportunities.")

        return list(recommendations)
        
        churn = data.get('churn', 0)
        if churn > 0.2:
            recommendations.append("High churn detected - implement retention strategies")
        
        cac = data.get('cac', 0)
        ltv = data.get('ltv', 1)
        if cac / max(ltv, 1) > 0.5:
            recommendations.append("CAC to LTV ratio is concerning - optimize acquisition costs")
        
        monthly_expenses = data.get('monthly_expenses', 0)
        users = data.get('users', 1)
        if monthly_expenses / max(users, 1) > 100:
            recommendations.append("High operational costs per user - review expense structure")
        
        return recommendations or ["Metrics look healthy - continue current strategy"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fundpilot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Setup CORS
CORS(app, 
     origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key'],
     expose_headers=['X-RateLimit-Remaining', 'X-Response-Time'])

# Initialize Redis with error handling
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
    logger.info("✅ Redis connected successfully")
except Exception as e:
    redis_client = None
    CACHE_ENABLED = False
    logger.warning(f"⚠️  Redis unavailable: {e}")

# Initialize components
executor = ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 8)), thread_name_prefix="FundPilot")
smart_cache = SmartCache(CacheStrategy.HYBRID if CACHE_ENABLED else CacheStrategy.MEMORY)
rate_limiter = AdvancedRateLimiter()
db_manager = DatabaseManager()
ml_predictor = MLPredictor()
circuit_breaker = CircuitBreaker()
performance_metrics = PerformanceMetrics()

# API Key Management
VALID_API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "test-key-123").split(',') if key.strip()]

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not VALID_API_KEYS or api_key in VALID_API_KEYS:
            return f(*args, **kwargs)
        return jsonify({'error': 'Invalid or missing API key'}), 403
    return decorated_function

# Request middleware
@app.before_request
def before_request():
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())[:8]
    performance_metrics.request_count += 1
    performance_metrics.active_connections += 1

@app.after_request
def after_request(response):
    try:
        response_time = time.time() - g.start_time
        performance_metrics.total_response_time += response_time
        performance_metrics.active_connections = max(0, performance_metrics.active_connections - 1)
        
        if response.status_code >= 400:
            performance_metrics.error_count += 1
        
        response.headers['X-Response-Time'] = f"{response_time:.3f}s"
        response.headers['X-Request-ID'] = g.request_id
    except Exception as e:
        logger.error(f"After request error: {e}")
    
    return response

# Schema validation
class AdvancedCalculationSchema(Schema):
    users = fields.Integer(required=True, validate=validate.Range(min=1, max=10000000))
    churn = fields.Float(required=True, validate=validate.Range(min=0.001, max=0.999))
    ltv = fields.Float(required=True, validate=validate.Range(min=1, max=1000000))
    monthly_expenses = fields.Float(required=True, validate=validate.Range(min=0, max=10000000))
    initial_capital = fields.Float(required=True, validate=validate.Range(min=0, max=100000000))
    cac = fields.Float(required=True, validate=validate.Range(min=0.01, max=10000))
    market_segment = fields.String(load_default='B2B', validate=validate.OneOf(['B2B', 'B2C', 'B2B2C']))
    industry = fields.String(load_default='Technology')
    prediction_horizon = fields.Integer(load_default=12, validate=validate.Range(min=1, max=60))
    include_ml_predictions = fields.Boolean(load_default=True)
    
    @post_load
    def validate_business_logic(self, data, **kwargs):
        if data['cac'] > data['ltv']:
            logger.warning(f"CAC ({data['cac']}) exceeds LTV ({data['ltv']})")
        return data

class EnhancedAdviceSchema(Schema):
    prompt = fields.String(required=True, validate=validate.Length(min=10, max=2000))
    context = fields.Dict(load_default={})
    priority = fields.Enum(Priority, load_default=Priority.NORMAL)
    user_id = fields.String(load_default=lambda: str(uuid.uuid4()))
    session_id = fields.String(load_default=lambda: str(uuid.uuid4()))
    language = fields.String(load_default='en', validate=validate.OneOf(['en', 'es', 'fr', 'de']))
    max_response_length = fields.Integer(load_default=500, validate=validate.Range(min=100, max=2000))

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    logger.error(f"Validation error [{g.request_id}]: {e.messages}")
    return jsonify({
        'success': False,
        'error': 'Validation failed',
        'details': e.messages,
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 400

@app.errorhandler(429)
def handle_rate_limit(e):
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded',
        'message': 'Too many requests, please slow down',
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 429

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error [{g.request_id}]: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 500

@app.errorhandler(Exception)
def handle_general_exception(e):
    logger.error(f"Unhandled exception [{g.request_id}]: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Unexpected error',
        'message': str(e),
        'request_id': getattr(g, 'request_id', 'unknown')
    }), 500

def validate_and_extract(schema, data):
    try:
        return schema.load(data)
    except ValidationError as e:
        raise e

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

# 🔥 FIXED MAIN CALCULATION ROUTE
@app.route('/v2/calculate', methods=['POST'])
def handle_advanced_calculation():
    """Fixed calculation endpoint with comprehensive error handling"""
    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                'success': False, 
                'error': 'Invalid Content-Type', 
                'message': 'Request must be application/json'
            }), 400

        # Get and validate JSON data
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False, 
                'error': 'Bad Request', 
                'message': 'No JSON data received'
            }), 400

        logger.info(f"📦 Received calculation request [{g.request_id}]: {request_data}")

        # Validate data against schema
        try:
            validated_data = validate_and_extract(AdvancedCalculationSchema(), request_data)
        except ValidationError as e:
            logger.error(f"Validation failed [{g.request_id}]: {e.messages}")
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': e.messages,
                'request_id': g.request_id
            }), 400

        # Check cache first
        cache_key = validated_data
        cached_result = smart_cache.get(cache_key)
        if cached_result:
            logger.info(f"✅ Cache hit [{g.request_id}]")
            performance_metrics.cache_hits += 1
            return jsonify({
                'success': True,
                'data': {'summary_metrics': cached_result},
                'cached': True,
                'request_id': g.request_id,
                'processing_time': f"{time.time() - g.start_time:.3f}s"
            }), 200

        # Perform calculation
        logger.info(f"🔄 Calculating metrics [{g.request_id}]")
        
        try:
            # Use circuit breaker for external calculations
            results = circuit_breaker.call(calculate_metrics, validated_data)
            
            # Add ML predictions if requested
            if validated_data.get('include_ml_predictions', True):
                try:
                    ml_results = ml_predictor.predict_metrics(validated_data)
                    results.update(ml_results)
                except Exception as ml_error:
                    logger.warning(f"ML prediction failed [{g.request_id}]: {ml_error}")
                    # Continue without ML predictions
            
        except Exception as calc_error:
            logger.error(f"Calculation failed [{g.request_id}]: {calc_error}")
            return jsonify({
                'success': False,
                'error': 'Calculation failed',
                'message': str(calc_error),
                'request_id': g.request_id
            }), 500

        # Cache the results
        smart_cache.set(cache_key, results, ttl=300)
        performance_metrics.cache_misses += 1

        # Log to database if available
        log_request_to_db('/v2/calculate', validated_data, time.time() - g.start_time, 200)

        logger.info(f"✅ Calculation completed [{g.request_id}]")
        
        response_data = {
            'success': True,
            'data': {'summary_metrics': results},
            'cached': False,
            'request_id': g.request_id,
            'processing_time': f"{time.time() - g.start_time:.3f}s"
        }
        
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"❌ Unhandled error in calculation [{g.request_id}]: {str(e)}", exc_info=True)
        performance_metrics.error_count += 1
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred during calculation',
            'request_id': getattr(g, 'request_id', 'unknown'),
            'debug': str(e) if app.debug else None
        }), 500

# Legacy API compatibility
@app.route('/api/calculate', methods=['POST'])
def api_calculate():
    """Legacy calculation endpoint with error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid input'}), 400

        logger.info(f"📦 Legacy API request [{g.request_id}]: {data}")

        results = {}

        # Extract data for calculations
        mrr_data = data.get('mrr_data')
        burn_data = data.get('burn_data')
        ltv_cac_data = data.get('ltv_cac_data')

        try:
            if mrr_data:
                results['mrr_arr'] = calculate_mrr_arr(mrr_data)
            
            if burn_data:
                results['burn_runway'] = calculate_burn_rate_runway(burn_data)

            if ltv_cac_data:
                results['ltv_cac'] = calculate_ltv_cac(ltv_cac_data)

        except Exception as calc_error:
            logger.error(f"Legacy calculation failed [{g.request_id}]: {calc_error}")
            return jsonify({
                'success': False,
                'error': 'Calculation failed',
                'message': str(calc_error)
            }), 500

        return jsonify({
            'success': True,
            'data': results,
            'request_id': g.request_id
        }), 200

    except Exception as e:
        logger.error(f"❌ Legacy API error [{g.request_id}]: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/v2/get-advice', methods=['POST'])
@require_api_key
def handle_enhanced_advice():
    """Enhanced advice endpoint with proper error handling"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Invalid Content-Type',
                'message': 'Request must be application/json'
            }), 400
        
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'Bad Request',
                'message': 'No JSON data received'
            }), 400

        logger.info(f"📦 Advice request [{g.request_id}]: {request_data}")
        
        # Validate data
        try:
            data = validate_and_extract(EnhancedAdviceSchema(), request_data)
        except ValidationError as e:
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': e.messages,
                'request_id': g.request_id
            }), 400
        
        # Check if AI models are available
        if not initialize_ai_models():
            return jsonify({
                'success': False,
                'error': 'AI models not available',
                'message': 'AI advisory service is currently unavailable'
            }), 503
        
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
        
        except Exception as advice_error:
            logger.error(f"Advice generation error [{g.request_id}]: {str(advice_error)}")
            log_request_to_db('/v2/get-advice', data, time.time() - g.start_time, 500)
            return jsonify({
                'success': False,
                'error': 'Advice generation failed',
                'message': str(advice_error),
                'request_id': g.request_id
            }), 500

    except Exception as e:
        logger.error(f"❌ Advice endpoint error [{g.request_id}]: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e),
            'request_id': g.request_id
        }), 500

@app.route('/v2/export', methods=['POST'])
@require_api_key
def export_data():
    """Export financial model data as CSV or JSON with error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request body',
                'message': 'No JSON data received'
            }), 400

        export_format = data.get('format', 'json').lower()
        model_data = data.get('model_data')

        if not model_data or 'projections' not in model_data or 'summary_metrics' not in model_data:
            return jsonify({
                'success': False,
                'error': 'Incomplete or invalid model data provided',
                'message': 'model_data must contain projections and summary_metrics'
            }), 400

        if export_format == 'csv':
            try:
                projections = model_data['projections']
                summary = model_data['summary_metrics']
                
                si = StringIO()
                cw = csv.writer(si)
                
                # Write summary metrics
                cw.writerow(['Metric', 'Value'])
                for key, value in summary.items():
                    cw.writerow([key.replace('_', ' ').title(), value])
                cw.writerow([])
                
                # Write projections
                if projections:
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
            except Exception as csv_error:
                logger.error(f"CSV export failed [{g.request_id}]: {csv_error}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate CSV file',
                    'message': str(csv_error)
                }), 500

        elif export_format == 'json':
            return jsonify({
                'success': True,
                'data': model_data,
                'format': 'json',
                'request_id': g.request_id
            }), 200
        
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported format',
                'message': "Please choose 'json' or 'csv'"
            }), 400

    except Exception as e:
        logger.error(f"❌ Export error [{g.request_id}]: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e),
            'request_id': g.request_id
        }), 500

@app.route('/health', methods=['GET'])
def comprehensive_health_check():
    """Comprehensive health check endpoint"""
    try:
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

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': int(time.time())
        }), 500

@app.route('/metrics', methods=['GET'])
@require_api_key
def get_comprehensive_metrics():
    """Get comprehensive system metrics"""
    try:
        metrics = {
            'timestamp': int(time.time()),
            'performance': performance_metrics.__dict__,
            'cache_stats': dict(smart_cache.cache_stats),
            'memory_usage': len(smart_cache.memory_cache),
            'rate_limiting': {
                'active_windows': len(rate_limiter.windows),
                'tracked_users': len(rate_limiter.user_patterns)
            },
            'request_id': g.request_id
        }
        
        if CACHE_ENABLED and redis_client:
            try:
                redis_info = redis_client.info()
                metrics['redis'] = {
                    'memory_usage': redis_info.get('used_memory_human', 'N/A'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'commands_processed': redis_info.get('total_commands_processed', 0)
                }
            except Exception as redis_error:
                metrics['redis'] = {'error': str(redis_error)}
        
        return jsonify(metrics), 200

    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve metrics',
            'message': str(e),
            'timestamp': int(time.time())
        }), 500

@app.route('/admin/cache/clear', methods=['POST'])
@require_api_key
def clear_cache():
    """Clear system cache"""
    try:
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
            except Exception as redis_error:
                logger.error(f"Redis cache clear failed: {redis_error}")
        
        return jsonify({
            'success': True,
            'message': f"Cache cleared{f' for pattern: {pattern}' if pattern else ''}",
            'timestamp': int(time.time()),
            'request_id': g.request_id
        }), 200

    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear cache',
            'message': str(e)
        }), 500

# Static file serving routes
@app.route('/')
def serve_landing():
    """Serve the landing page as the home page"""
    try:
        return send_from_directory('.', 'landing.html')
    except Exception as e:
        logger.error(f"Failed to serve landing.html: {e}")
        return jsonify({
            'error': 'Failed to serve landing page',
            'message': str(e)
        }), 500

@app.route('/calculator')
@app.route('/app')
def serve_calculator():
    """Serve the calculator/calculative page"""
    try:
        return send_from_directory('calculative', 'index.html')
    except Exception as e:
        logger.error(f"Failed to serve calculator index.html: {e}")
        return jsonify({
            'error': 'Failed to serve calculator page',
            'message': str(e)
        }), 500

@app.route('/about')
def serve_about():
    """Serve the about page"""
    try:
        return send_from_directory('.', 'about.html')
    except Exception as e:
        logger.error(f"Failed to serve about.html: {e}")
        return jsonify({
            'error': 'Failed to serve about page',
            'message': str(e)
        }), 500

@app.route('/assets/<path:path>')
def serve_static_assets(path):
    """Serve static assets from the assets directory"""
    try:
        return send_from_directory('assets', path)
    except Exception as e:
        logger.error(f"Failed to serve asset {path}: {e}")
        return jsonify({
            'error': 'Asset not found',
            'path': path
        }), 404

@app.route('/calculative/assets/<path:path>')
def serve_calculative_assets(path):
    """Serve static assets from the calculative/assets directory"""
    try:
        return send_from_directory('calculative/assets', path)
    except Exception as e:
        logger.error(f"Failed to serve calculative asset {path}: {e}")
        return jsonify({
            'error': 'Asset not found',
            'path': path
        }), 404

@app.route('/svg/<path:path>')
def serve_svg_assets(path):
    """Serve SVG assets"""
    try:
        return send_from_directory('svg', path)
    except Exception as e:
        logger.error(f"Failed to serve SVG {path}: {e}")
        return jsonify({
            'error': 'SVG not found',
            'path': path
        }), 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files like CSS and JS"""
    try:
        # First try to serve from calculative directory for calculator-related files
        if filename.endswith(('.css', '.js', '.html')):
            return send_from_directory('calculative', filename)
        # Otherwise serve from root directory
        return send_from_directory('.', filename)
    except Exception as e:
        logger.error(f"Failed to serve static file {filename}: {e}")
        return jsonify({
            'error': 'File not found',
            'filename': filename
        }), 404

# 🔥 SIMPLE TEST ROUTE FOR DEBUGGING
@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Simple test endpoint for debugging"""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'message': 'Test endpoint working',
                'method': 'GET',
                'timestamp': int(time.time()),
                'request_id': g.request_id
            }), 200
        
        elif request.method == 'POST':
            data = request.get_json() if request.is_json else {}
            return jsonify({
                'success': True,
                'message': 'Test POST endpoint working',
                'received_data': data,
                'method': 'POST',
                'timestamp': int(time.time()),
                'request_id': g.request_id
            }), 200

    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Test endpoint failed',
            'message': str(e)
        }), 500

# Graceful shutdown handling
def graceful_shutdown(signum, frame):
    """Handle graceful shutdown"""
    logger.info("🔄 Initiating graceful shutdown...")
    try:
        executor.shutdown(wait=True)
        if db_manager.pool:
            db_manager.pool.closeall()
        logger.info("✅ Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Register cleanup function
atexit.register(lambda: executor.shutdown(wait=True))

# Development-specific routes and handlers
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')  # Changed for Vercel compatibility
    
    logger.info("=" * 60)
    logger.info("🚀 Starting FinPilot API Server")
    logger.info("=" * 60)
    logger.info(f"📍 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🐛 Debug mode: {debug}")
    logger.info(f"💾 Redis enabled: {CACHE_ENABLED}")
    logger.info(f"🗄️  Database enabled: {db_manager.pool is not None}")
    logger.info(f"🤖 AI models available: {initialize_ai_models()}")
    logger.info(f"🔑 API keys configured: {len(VALID_API_KEYS)}")
    logger.info("=" * 60)
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

# Add this line at the very end for Vercel compatibility
application = app

def validate_calculations():
    """Test function to validate financial calculations accuracy"""
    print("🧮 Validating Financial Calculations...")
    
    # Test case 1: Basic SaaS metrics
    test_data = {
        'users': 1000,
        'ltv': 50,  # ARPU
        'cac': 100,
        'churn': 0.05,  # 5% monthly churn
        'monthly_expenses': 15000,
        'initial_capital': 100000,
        'monthly_growth': 10,  # 10% growth
        'cogs': 20  # 20% COGS
    }
    
    results = calculate_metrics(test_data)
    
    # Expected calculations
    expected_mrr = 1000 * 50  # 50,000
    expected_arr = expected_mrr * 12  # 600,000
    expected_actual_ltv = 50 / 0.05  # 1,000
    expected_ltv_cac_ratio = 1000 / 100  # 10.0
    expected_runway = 100000 / 15000  # 6.67 months
    expected_break_even = 15000 / (50 * 0.8)  # 375 users
    expected_churn_loss = 50000 * 0.05  # 2,500
    expected_payback = 100 / (50 * 0.8)  # 2.5 months
    
    # Validation checks
    validations = [
        ("MRR", results['mrr'], expected_mrr),
        ("ARR", results['arr'], expected_arr),
        ("Actual LTV", results['ltv'], expected_actual_ltv),
        ("LTV/CAC Ratio", results['ltv_cac_ratio'], expected_ltv_cac_ratio),
        ("Runway", results['runway'], expected_runway),
        ("Break Even Users", results['break_even_users'], expected_break_even),
        ("Churn Loss", results['churn_loss'], expected_churn_loss),
        ("Payback Period", results['payback_period'], expected_payback),
    ]
    
    all_passed = True
    for metric, actual, expected in validations:
        if abs(actual - expected) < 0.01:  # Allow small rounding differences
            print(f"✅ {metric}: {actual} (Expected: {expected})")
        else:
            print(f"❌ {metric}: {actual} (Expected: {expected}) - MISMATCH!")
            all_passed = False
    
    # Additional validation checks
    print(f"\n📊 Additional Metrics:")
    print(f"Growth Efficiency: {results['growth_efficiency']}")
    print(f"Growth Potential: {results['growth_potential']}")
    print(f"Risk Score: {results['risk_score']}")
    print(f"Profitability: {results['profitability']}")
    print(f"Predicted Revenue: {results['predicted_revenue']}")
    
    if all_passed:
        print(f"\n🎉 All core calculations PASSED! System is calculating correctly.")
    else:
        print(f"\n⚠️  Some calculations failed validation. Review formulas.")
    
    return all_passed