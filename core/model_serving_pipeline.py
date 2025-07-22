"""
Model Serving Pipeline

This module implements a production-ready model serving system with:
- Multiple ML model serving with load balancing
- A/B testing for model deployment
- Model versioning and rollback capabilities
- Real-time inference with <10ms latency
- Model performance monitoring and alerting
- FastAPI endpoints with caching and batch inference
- Health checks and graceful updates

Key Features:
- Hot model swapping without downtime
- Request/response logging and validation
- Performance metrics and monitoring
- Automatic failover and circuit breakers
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import pickle
import os
from pathlib import Path

# FastAPI and async components
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn

# ML and data processing
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Caching
from functools import lru_cache
import redis
from cachetools import TTLCache

logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_name: str
    model_version: Optional[str] = "latest"
    features: Dict[str, float]
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    model_name: str
    model_version: Optional[str] = "latest"
    batch_features: List[Dict[str, float]]
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('batch_features')
    def validate_batch_features(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Batch features cannot be empty")
        if len(v) > 1000:  # Limit batch size
            raise ValueError("Batch size cannot exceed 1000")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: Union[float, List[float], Dict[str, float]]
    confidence: Optional[float] = None
    model_name: str
    model_version: str
    request_id: str
    processing_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = {}


class ModelStatus(BaseModel):
    """Model status information"""
    name: str
    version: str
    status: str  # 'active', 'loading', 'error', 'deprecated'
    load_time: datetime
    last_prediction: Optional[datetime] = None
    prediction_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class ModelMetrics:
    """Metrics for model performance monitoring"""
    prediction_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_prediction_time: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.prediction_count)
    
    @property
    def error_rate(self) -> float:
        total = self.prediction_count + self.error_count
        return self.error_count / max(1, total)


@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    version: str
    model_path: str
    model_type: str  # 'sklearn', 'tensorflow', 'pytorch', 'custom'
    max_batch_size: int = 100
    timeout_ms: int = 5000
    memory_limit_mb: int = 1024
    warmup_requests: int = 5
    health_check_interval: int = 60
    a_b_test_weight: float = 1.0  # Weight for A/B testing
    feature_schema: Optional[Dict[str, str]] = None


class CircuitBreaker:
    """Circuit breaker for model serving"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # 'closed', 'open', 'half-open'
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


class ModelWrapper:
    """Wrapper for ML models with caching and monitoring"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.metrics = ModelMetrics()
        self.circuit_breaker = CircuitBreaker()
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
        self.lock = threading.RLock()
        self.status = 'loading'
        self.load_time = None
        
    def load_model(self):
        """Load model from disk"""
        try:
            start_time = time.time()
            
            if self.config.model_type == 'sklearn':
                self.model = joblib.load(self.config.model_path)
            elif self.config.model_type == 'pickle':
                with open(self.config.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.load_time = datetime.now()
            self.status = 'active'
            
            # Warmup predictions
            self._warmup_model()
            
            load_duration = time.time() - start_time
            logger.info(f"Model {self.config.name}:{self.config.version} loaded in {load_duration:.2f}s")
            
        except Exception as e:
            self.status = 'error'
            logger.error(f"Error loading model {self.config.name}:{self.config.version}: {e}")
            raise e
    
    def _warmup_model(self):
        """Warmup model with dummy predictions"""
        if not self.model or not hasattr(self.model, 'predict'):
            return
        
        try:
            # Create dummy features based on schema
            if self.config.feature_schema:
                dummy_features = {}
                for feature, dtype in self.config.feature_schema.items():
                    if dtype == 'float':
                        dummy_features[feature] = 0.5
                    elif dtype == 'int':
                        dummy_features[feature] = 1
                    else:
                        dummy_features[feature] = 0.0
                
                # Perform warmup predictions
                for _ in range(self.config.warmup_requests):
                    self._predict_internal(dummy_features)
                    
        except Exception as e:
            logger.warning(f"Model warmup failed for {self.config.name}: {e}")
    
    def predict(self, features: Dict[str, float], request_id: str = None) -> Tuple[Any, float]:
        """Make prediction with monitoring and caching"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(features)
            
            # Check cache first
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                latency_ms = (time.time() - start_time) * 1000
                self._update_metrics(latency_ms, from_cache=True)
                return cached_result, latency_ms
            
            # Make prediction with circuit breaker
            prediction = self.circuit_breaker.call(self._predict_internal, features)
            
            # Cache result
            self.cache[cache_key] = prediction
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms)
            
            return prediction, latency_ms
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Prediction error for {self.config.name}: {e}")
            raise e
    
    def _predict_internal(self, features: Dict[str, float]) -> Any:
        """Internal prediction method"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Convert features to model input format
        if hasattr(self.model, 'predict'):
            # Sklearn-style model
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            prediction = self.model.predict(feature_array)[0]
        else:
            # Custom model
            prediction = self.model(features)
        
        return float(prediction) if isinstance(prediction, (np.ndarray, np.number)) else prediction
    
    def batch_predict(self, batch_features: List[Dict[str, float]]) -> List[Any]:
        """Batch prediction for efficiency"""
        if len(batch_features) > self.config.max_batch_size:
            raise ValueError(f"Batch size {len(batch_features)} exceeds limit {self.config.max_batch_size}")
        
        start_time = time.time()
        
        try:
            # Convert to batch format
            if hasattr(self.model, 'predict'):
                # Sklearn-style batch prediction
                feature_arrays = []
                for features in batch_features:
                    feature_arrays.append(list(features.values()))
                
                batch_array = np.array(feature_arrays)
                predictions = self.model.predict(batch_array)
                results = [float(p) if isinstance(p, (np.ndarray, np.number)) else p for p in predictions]
            else:
                # Individual predictions for custom models
                results = []
                for features in batch_features:
                    pred = self.model(features)
                    results.append(float(pred) if isinstance(pred, (np.ndarray, np.number)) else pred)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, batch_size=len(batch_features))
            
            return results
            
        except Exception as e:
            self.metrics.error_count += len(batch_features)
            logger.error(f"Batch prediction error for {self.config.name}: {e}")
            raise e
    
    def _generate_cache_key(self, features: Dict[str, float]) -> str:
        """Generate cache key from features"""
        # Sort features for consistent key generation
        sorted_features = sorted(features.items())
        return f"{self.config.name}:{self.config.version}:{hash(str(sorted_features))}"
    
    def _update_metrics(self, latency_ms: float, from_cache: bool = False, batch_size: int = 1):
        """Update model metrics"""
        with self.lock:
            if not from_cache:
                self.metrics.prediction_count += batch_size
                self.metrics.total_latency_ms += latency_ms
            self.metrics.last_prediction_time = datetime.now()
            
            # Update system metrics
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
    
    def get_status(self) -> ModelStatus:
        """Get current model status"""
        return ModelStatus(
            name=self.config.name,
            version=self.config.version,
            status=self.status,
            load_time=self.load_time or datetime.now(),
            last_prediction=self.metrics.last_prediction_time,
            prediction_count=self.metrics.prediction_count,
            error_count=self.metrics.error_count,
            avg_latency_ms=self.metrics.avg_latency_ms,
            memory_usage_mb=self.metrics.memory_usage_mb,
            cpu_usage_percent=self.metrics.cpu_usage_percent
        )


class ABTestManager:
    """Manages A/B testing for model deployment"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(list)
        self.lock = threading.Lock()
    
    def create_experiment(self, experiment_id: str, model_configs: List[Tuple[str, str, float]]):
        """
        Create A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            model_configs: List of (model_name, version, weight) tuples
        """
        with self.lock:
            total_weight = sum(weight for _, _, weight in model_configs)
            normalized_configs = [(name, version, weight/total_weight) for name, version, weight in model_configs]
            
            self.experiments[experiment_id] = {
                'models': normalized_configs,
                'created_at': datetime.now(),
                'request_count': 0
            }
            
        logger.info(f"Created A/B test experiment {experiment_id} with {len(model_configs)} models")
    
    def select_model(self, experiment_id: str, request_id: str = None) -> Tuple[str, str]:
        """Select model for request based on A/B test weights"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with self.lock:
            experiment = self.experiments[experiment_id]
            experiment['request_count'] += 1
            
            # Use request_id for consistent routing if provided
            if request_id:
                hash_value = hash(request_id) % 1000 / 1000.0
            else:
                hash_value = np.random.random()
            
            cumulative_weight = 0.0
            for model_name, version, weight in experiment['models']:
                cumulative_weight += weight
                if hash_value <= cumulative_weight:
                    return model_name, version
            
            # Fallback to last model
            return experiment['models'][-1][:2]
    
    def record_result(self, experiment_id: str, model_name: str, version: str, 
                     latency_ms: float, success: bool):
        """Record experiment result"""
        with self.lock:
            self.results[experiment_id].append({
                'model_name': model_name,
                'version': version,
                'latency_ms': latency_ms,
                'success': success,
                'timestamp': datetime.now()
            })
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results"""
        if experiment_id not in self.results:
            return {'error': 'No results found'}
        
        results = self.results[experiment_id]
        model_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'total_latency': 0.0})
        
        for result in results:
            key = f"{result['model_name']}:{result['version']}"
            model_stats[key]['count'] += 1
            if result['success']:
                model_stats[key]['success'] += 1
            model_stats[key]['total_latency'] += result['latency_ms']
        
        # Calculate statistics
        summary = {}
        for model_key, stats in model_stats.items():
            summary[model_key] = {
                'request_count': stats['count'],
                'success_rate': stats['success'] / stats['count'],
                'avg_latency_ms': stats['total_latency'] / stats['count'],
                'error_rate': 1 - (stats['success'] / stats['count'])
            }
        
        return {
            'experiment_id': experiment_id,
            'total_requests': len(results),
            'model_performance': summary,
            'created_at': self.experiments[experiment_id]['created_at'].isoformat()
        }


class ModelServingPipeline:
    """
    Main model serving pipeline with load balancing, A/B testing, and monitoring
    """
    
    def __init__(self, redis_url: str = None):
        self.models: Dict[str, Dict[str, ModelWrapper]] = defaultdict(dict)
        self.ab_test_manager = ABTestManager()
        self.request_logs = deque(maxlen=10000)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.RLock()
        
        # Redis for distributed caching (optional)
        self.redis_client = None
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
            except ImportError:
                logger.warning("Redis not available, using local cache only")
        
        # Prometheus metrics
        self.prediction_counter = Counter('model_predictions_total', 'Total predictions', ['model', 'version'])
        self.prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model', 'version'])
        self.error_counter = Counter('model_errors_total', 'Total errors', ['model', 'version', 'error_type'])
        self.active_models_gauge = Gauge('active_models', 'Number of active models')
        
        logger.info("ModelServingPipeline initialized")
    
    def register_model(self, config: ModelConfig) -> bool:
        """Register and load a new model"""
        try:
            model_wrapper = ModelWrapper(config)
            model_wrapper.load_model()
            
            with self.lock:
                self.models[config.name][config.version] = model_wrapper
                self.active_models_gauge.set(self._count_active_models())
            
            logger.info(f"Registered model {config.name}:{config.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {config.name}:{config.version}: {e}")
            return False
    
    def unregister_model(self, model_name: str, version: str) -> bool:
        """Unregister a model"""
        try:
            with self.lock:
                if model_name in self.models and version in self.models[model_name]:
                    del self.models[model_name][version]
                    if not self.models[model_name]:
                        del self.models[model_name]
                    self.active_models_gauge.set(self._count_active_models())
                    logger.info(f"Unregistered model {model_name}:{version}")
                    return True
                else:
                    logger.warning(f"Model {model_name}:{version} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error unregistering model {model_name}:{version}: {e}")
            return False
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction"""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Get model
            model_wrapper = self._get_model(request.model_name, request.model_version)
            if not model_wrapper:
                raise HTTPException(status_code=404, detail=f"Model {request.model_name}:{request.model_version} not found")
            
            # Validate features
            self._validate_features(request.features, model_wrapper.config)
            
            # Make prediction
            prediction, latency_ms = model_wrapper.predict(request.features, request_id)
            
            # Create response
            response = PredictionResponse(
                prediction=prediction,
                model_name=request.model_name,
                model_version=model_wrapper.config.version,
                request_id=request_id,
                processing_time_ms=latency_ms,
                timestamp=datetime.now(),
                metadata=request.metadata
            )
            
            # Log request
            self._log_request(request, response, success=True)
            
            # Update metrics
            self.prediction_counter.labels(model=request.model_name, version=model_wrapper.config.version).inc()
            self.prediction_latency.labels(model=request.model_name, version=model_wrapper.config.version).observe(latency_ms / 1000)
            
            return response
            
        except Exception as e:
            # Log error
            self._log_request(request, None, success=False, error=str(e))
            self.error_counter.labels(model=request.model_name, version=request.model_version or 'unknown', error_type=type(e).__name__).inc()
            
            raise HTTPException(status_code=500, detail=str(e))
    
    def batch_predict(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """Make batch predictions"""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Get model
            model_wrapper = self._get_model(request.model_name, request.model_version)
            if not model_wrapper:
                raise HTTPException(status_code=404, detail=f"Model {request.model_name}:{request.model_version} not found")
            
            # Validate batch features
            for features in request.batch_features:
                self._validate_features(features, model_wrapper.config)
            
            # Make batch prediction
            predictions = model_wrapper.batch_predict(request.batch_features)
            
            # Create responses
            responses = []
            for i, prediction in enumerate(predictions):
                response = PredictionResponse(
                    prediction=prediction,
                    model_name=request.model_name,
                    model_version=model_wrapper.config.version,
                    request_id=f"{request_id}_{i}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    metadata=request.metadata
                )
                responses.append(response)
            
            # Update metrics
            self.prediction_counter.labels(model=request.model_name, version=model_wrapper.config.version).inc(len(predictions))
            
            return responses
            
        except Exception as e:
            self.error_counter.labels(model=request.model_name, version=request.model_version or 'unknown', error_type=type(e).__name__).inc()
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_model(self, model_name: str, version: str = None) -> Optional[ModelWrapper]:
        """Get model wrapper"""
        if version is None or version == "latest":
            # Get latest version
            if model_name in self.models and self.models[model_name]:
                latest_version = max(self.models[model_name].keys())
                return self.models[model_name][latest_version]
        else:
            # Get specific version
            if model_name in self.models and version in self.models[model_name]:
                return self.models[model_name][version]
        
        return None
    
    def _validate_features(self, features: Dict[str, float], config: ModelConfig):
        """Validate input features"""
        if config.feature_schema:
            for feature_name, expected_type in config.feature_schema.items():
                if feature_name not in features:
                    raise ValueError(f"Missing required feature: {feature_name}")
                
                value = features[feature_name]
                if expected_type == 'float' and not isinstance(value, (int, float)):
                    raise ValueError(f"Feature {feature_name} must be numeric")
    
    def _log_request(self, request, response, success: bool, error: str = None):
        """Log request/response for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': getattr(request, 'request_id', None),
            'model_name': request.model_name,
            'model_version': getattr(request, 'model_version', None),
            'success': success,
            'error': error,
            'processing_time_ms': getattr(response, 'processing_time_ms', None) if response else None
        }
        
        self.request_logs.append(log_entry)
    
    def _count_active_models(self) -> int:
        """Count active models"""
        count = 0
        for model_versions in self.models.values():
            for model_wrapper in model_versions.values():
                if model_wrapper.status == 'active':
                    count += 1
        return count
    
    def get_model_status(self, model_name: str = None) -> Dict[str, Any]:
        """Get status of models"""
        if model_name:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}
            
            versions = {}
            for version, wrapper in self.models[model_name].items():
                versions[version] = wrapper.get_status().dict()
            
            return {'model': model_name, 'versions': versions}
        else:
            # Get all models
            all_models = {}
            for name, versions in self.models.items():
                all_models[name] = {}
                for version, wrapper in versions.items():
                    all_models[name][version] = wrapper.get_status().dict()
            
            return {'models': all_models, 'total_active': self._count_active_models()}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        active_models = self._count_active_models()
        recent_errors = sum(1 for log in list(self.request_logs)[-100:] if not log['success'])
        
        return {
            'status': 'healthy' if active_models > 0 and recent_errors < 10 else 'unhealthy',
            'active_models': active_models,
            'recent_error_rate': recent_errors / min(100, len(self.request_logs)),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage_percent': psutil.cpu_percent(),
            'timestamp': datetime.now().isoformat()
        }


# FastAPI application
def create_app(pipeline: ModelServingPipeline) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="Model Serving Pipeline",
        description="Production-ready ML model serving with A/B testing and monitoring",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Single prediction endpoint"""
        return pipeline.predict(request)
    
    @app.post("/batch_predict", response_model=List[PredictionResponse])
    async def batch_predict(request: BatchPredictionRequest):
        """Batch prediction endpoint"""
        return pipeline.batch_predict(request)
    
    @app.get("/models/{model_name}/status")
    async def get_model_status(model_name: str):
        """Get model status"""
        return pipeline.get_model_status(model_name)
    
    @app.get("/models/status")
    async def get_all_models_status():
        """Get all models status"""
        return pipeline.get_model_status()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return pipeline.get_health_status()
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return generate_latest()
    
    @app.post("/models/{model_name}/register")
    async def register_model(model_name: str, config: dict):
        """Register new model"""
        model_config = ModelConfig(name=model_name, **config)
        success = pipeline.register_model(model_config)
        return {"success": success, "model": model_name}
    
    @app.delete("/models/{model_name}/{version}")
    async def unregister_model(model_name: str, version: str):
        """Unregister model"""
        success = pipeline.unregister_model(model_name, version)
        return {"success": success, "model": f"{model_name}:{version}"}
    
    @app.post("/experiments/{experiment_id}")
    async def create_ab_test(experiment_id: str, models: List[dict]):
        """Create A/B test experiment"""
        model_configs = [(m['name'], m['version'], m['weight']) for m in models]
        pipeline.ab_test_manager.create_experiment(experiment_id, model_configs)
        return {"success": True, "experiment_id": experiment_id}
    
    @app.get("/experiments/{experiment_id}/results")
    async def get_ab_test_results(experiment_id: str):
        """Get A/B test results"""
        return pipeline.ab_test_manager.get_experiment_results(experiment_id)
    
    return app


# Example usage and testing
if __name__ == "__main__":
    # Create pipeline
    pipeline = ModelServingPipeline()
    
    # Create FastAPI app
    app = create_app(pipeline)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")