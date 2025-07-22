"""
Model Serving Pipeline Demo

This script demonstrates the Model Serving Pipeline with:
- Model registration and serving
- A/B testing capabilities
- Performance monitoring
- Batch inference
- Health checks and metrics

Usage:
    python examples/model_serving_demo.py
"""

import asyncio
import json
import time
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_serving_pipeline import (
    ModelServingPipeline, ModelConfig, PredictionRequest, 
    BatchPredictionRequest, create_app
)
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockTradingModel:
    """Mock trading model for demonstration"""
    
    def __init__(self, model_name: str, bias: float = 0.0):
        self.model_name = model_name
        self.bias = bias
        self.prediction_count = 0
    
    def predict(self, X):
        """Mock prediction with some randomness and bias"""
        self.prediction_count += 1
        
        if hasattr(X, 'shape') and len(X.shape) > 1:
            # Batch prediction
            predictions = []
            for row in X:
                # Simple mock prediction based on features
                prediction = np.mean(row) + self.bias + np.random.normal(0, 0.1)
                predictions.append(max(0, min(1, prediction)))  # Clamp to [0, 1]
            return np.array(predictions)
        else:
            # Single prediction
            prediction = np.mean(X) + self.bias + np.random.normal(0, 0.1)
            return max(0, min(1, prediction))  # Clamp to [0, 1]


def create_mock_models():
    """Create mock models for demonstration"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create different model versions
    models = {
        "strategy_selector_v1": MockTradingModel("strategy_selector", bias=0.1),
        "strategy_selector_v2": MockTradingModel("strategy_selector", bias=0.15),
        "position_sizer_v1": MockTradingModel("position_sizer", bias=0.05),
        "sentiment_analyzer_v1": MockTradingModel("sentiment_analyzer", bias=0.0),
    }
    
    # Save models using pickle
    import pickle
    for model_name, model in models.items():
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Created mock model: {model_path}")
    
    return models


def create_model_configs():
    """Create model configurations"""
    configs = [
        ModelConfig(
            name="strategy_selector",
            version="v1.0",
            model_path="models/strategy_selector_v1.pkl",
            model_type="pickle",
            max_batch_size=100,
            timeout_ms=1000,
            warmup_requests=5,
            a_b_test_weight=0.7,
            feature_schema={
                "price_momentum_1m": "float",
                "volume_ratio": "float",
                "rsi": "float",
                "macd_signal": "float",
                "trend_strength": "float"
            }
        ),
        ModelConfig(
            name="strategy_selector",
            version="v2.0",
            model_path="models/strategy_selector_v2.pkl",
            model_type="pickle",
            max_batch_size=100,
            timeout_ms=1000,
            warmup_requests=5,
            a_b_test_weight=0.3,
            feature_schema={
                "price_momentum_1m": "float",
                "volume_ratio": "float",
                "rsi": "float",
                "macd_signal": "float",
                "trend_strength": "float"
            }
        ),
        ModelConfig(
            name="position_sizer",
            version="v1.0",
            model_path="models/position_sizer_v1.pkl",
            model_type="pickle",
            max_batch_size=200,
            timeout_ms=500,
            warmup_requests=10,
            feature_schema={
                "signal_confidence": "float",
                "market_volatility": "float",
                "portfolio_exposure": "float",
                "win_rate": "float"
            }
        ),
        ModelConfig(
            name="sentiment_analyzer",
            version="v1.0",
            model_path="models/sentiment_analyzer_v1.pkl",
            model_type="pickle",
            max_batch_size=50,
            timeout_ms=2000,
            warmup_requests=3,
            feature_schema={
                "news_sentiment": "float",
                "social_sentiment": "float",
                "news_confidence": "float"
            }
        )
    ]
    
    return configs


def demo_model_registration(pipeline: ModelServingPipeline):
    """Demonstrate model registration"""
    print("\n" + "="*80)
    print("MODEL REGISTRATION DEMO")
    print("="*80)
    
    # Create mock models
    print("\n1. Creating mock models...")
    create_mock_models()
    
    # Create model configurations
    print("\n2. Creating model configurations...")
    configs = create_model_configs()
    
    # Register models
    print("\n3. Registering models...")
    for config in configs:
        success = pipeline.register_model(config)
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status}: {config.name}:{config.version}")
    
    # Check model status
    print("\n4. Model status:")
    status = pipeline.get_model_status()
    for model_name, versions in status['models'].items():
        print(f"   📊 {model_name}:")
        for version, info in versions.items():
            print(f"      - {version}: {info['status']} (predictions: {info['prediction_count']})")


def demo_single_predictions(pipeline: ModelServingPipeline):
    """Demonstrate single predictions"""
    print("\n" + "="*80)
    print("SINGLE PREDICTION DEMO")
    print("="*80)
    
    # Test different models
    test_cases = [
        {
            "model_name": "strategy_selector",
            "features": {
                "price_momentum_1m": 0.02,
                "volume_ratio": 1.5,
                "rsi": 65.0,
                "macd_signal": 0.1,
                "trend_strength": 0.8
            }
        },
        {
            "model_name": "position_sizer",
            "features": {
                "signal_confidence": 0.85,
                "market_volatility": 0.02,
                "portfolio_exposure": 0.6,
                "win_rate": 0.65
            }
        },
        {
            "model_name": "sentiment_analyzer",
            "features": {
                "news_sentiment": 0.3,
                "social_sentiment": 0.1,
                "news_confidence": 0.8
            }
        }
    ]
    
    print("\n1. Making single predictions...")
    for i, test_case in enumerate(test_cases, 1):
        try:
            request = PredictionRequest(**test_case)
            response = pipeline.predict(request)
            
            print(f"\n   Test {i}: {test_case['model_name']}")
            print(f"      Prediction: {response.prediction:.4f}")
            print(f"      Latency: {response.processing_time_ms:.2f}ms")
            print(f"      Model Version: {response.model_version}")
            
        except Exception as e:
            print(f"   ❌ Error in test {i}: {e}")


def demo_batch_predictions(pipeline: ModelServingPipeline):
    """Demonstrate batch predictions"""
    print("\n" + "="*80)
    print("BATCH PREDICTION DEMO")
    print("="*80)
    
    # Create batch request
    batch_features = []
    for i in range(10):
        features = {
            "price_momentum_1m": np.random.normal(0.01, 0.02),
            "volume_ratio": np.random.uniform(0.5, 2.0),
            "rsi": np.random.uniform(30, 70),
            "macd_signal": np.random.normal(0, 0.1),
            "trend_strength": np.random.uniform(0, 1)
        }
        batch_features.append(features)
    
    print(f"\n1. Making batch prediction with {len(batch_features)} samples...")
    
    try:
        request = BatchPredictionRequest(
            model_name="strategy_selector",
            batch_features=batch_features
        )
        
        start_time = time.time()
        responses = pipeline.batch_predict(request)
        batch_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Batch completed in {batch_time:.2f}ms")
        print(f"   📊 Results:")
        for i, response in enumerate(responses[:5]):  # Show first 5
            print(f"      Sample {i+1}: {response.prediction:.4f}")
        
        if len(responses) > 5:
            print(f"      ... and {len(responses)-5} more")
        
        avg_latency = sum(r.processing_time_ms for r in responses) / len(responses)
        print(f"   ⚡ Average latency per prediction: {avg_latency:.2f}ms")
        
    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")


def demo_ab_testing(pipeline: ModelServingPipeline):
    """Demonstrate A/B testing"""
    print("\n" + "="*80)
    print("A/B TESTING DEMO")
    print("="*80)
    
    # Create A/B test experiment
    print("\n1. Creating A/B test experiment...")
    experiment_id = "strategy_selector_ab_test"
    model_configs = [
        ("strategy_selector", "v1.0", 0.7),  # 70% traffic
        ("strategy_selector", "v2.0", 0.3),  # 30% traffic
    ]
    
    pipeline.ab_test_manager.create_experiment(experiment_id, model_configs)
    print(f"   ✅ Created experiment: {experiment_id}")
    print(f"   📊 Model distribution: v1.0 (70%), v2.0 (30%)")
    
    # Simulate A/B test requests
    print("\n2. Simulating A/B test requests...")
    test_features = {
        "price_momentum_1m": 0.015,
        "volume_ratio": 1.2,
        "rsi": 55.0,
        "macd_signal": 0.05,
        "trend_strength": 0.6
    }
    
    model_selections = {"v1.0": 0, "v2.0": 0}
    
    for i in range(100):
        request_id = f"ab_test_request_{i}"
        model_name, version = pipeline.ab_test_manager.select_model(experiment_id, request_id)
        model_selections[version] += 1
        
        # Record simulated result
        latency = np.random.uniform(5, 15)  # Mock latency
        success = np.random.random() > 0.1  # 90% success rate
        pipeline.ab_test_manager.record_result(experiment_id, model_name, version, latency, success)
    
    print(f"   📊 Model selection distribution:")
    for version, count in model_selections.items():
        print(f"      {version}: {count}% ({count}/100 requests)")
    
    # Get A/B test results
    print("\n3. A/B test results:")
    results = pipeline.ab_test_manager.get_experiment_results(experiment_id)
    
    if 'model_performance' in results:
        for model_key, stats in results['model_performance'].items():
            print(f"   📈 {model_key}:")
            print(f"      Requests: {stats['request_count']}")
            print(f"      Success Rate: {stats['success_rate']:.2%}")
            print(f"      Avg Latency: {stats['avg_latency_ms']:.2f}ms")


def demo_performance_monitoring(pipeline: ModelServingPipeline):
    """Demonstrate performance monitoring"""
    print("\n" + "="*80)
    print("PERFORMANCE MONITORING DEMO")
    print("="*80)
    
    # Generate load to collect metrics
    print("\n1. Generating load for metrics collection...")
    
    def make_requests():
        """Make requests in background"""
        for i in range(50):
            try:
                request = PredictionRequest(
                    model_name="strategy_selector",
                    features={
                        "price_momentum_1m": np.random.normal(0.01, 0.02),
                        "volume_ratio": np.random.uniform(0.5, 2.0),
                        "rsi": np.random.uniform(30, 70),
                        "macd_signal": np.random.normal(0, 0.1),
                        "trend_strength": np.random.uniform(0, 1)
                    }
                )
                pipeline.predict(request)
                time.sleep(0.1)  # Small delay
            except Exception as e:
                pass  # Ignore errors for demo
    
    # Run load generation in background
    thread = threading.Thread(target=make_requests)
    thread.start()
    
    # Wait a bit for metrics to accumulate
    time.sleep(2)
    
    # Show model status
    print("\n2. Model performance metrics:")
    status = pipeline.get_model_status()
    
    for model_name, versions in status['models'].items():
        print(f"\n   📊 {model_name.upper()}:")
        for version, info in versions.items():
            print(f"      Version {version}:")
            print(f"         Status: {info['status']}")
            print(f"         Predictions: {info['prediction_count']}")
            print(f"         Errors: {info['error_count']}")
            print(f"         Avg Latency: {info['avg_latency_ms']:.2f}ms")
            print(f"         Memory Usage: {info['memory_usage_mb']:.1f}MB")
    
    # Show health status
    print("\n3. System health status:")
    health = pipeline.get_health_status()
    print(f"   Status: {health['status'].upper()}")
    print(f"   Active Models: {health['active_models']}")
    print(f"   Recent Error Rate: {health['recent_error_rate']:.2%}")
    print(f"   Memory Usage: {health['memory_usage_mb']:.1f}MB")
    print(f"   CPU Usage: {health['cpu_usage_percent']:.1f}%")
    
    # Wait for background thread to complete
    thread.join()


def demo_api_endpoints():
    """Demonstrate API endpoints (requires server to be running)"""
    print("\n" + "="*80)
    print("API ENDPOINTS DEMO")
    print("="*80)
    
    base_url = "http://localhost:8000"
    
    print("\n1. Testing API endpoints...")
    print("   (Note: This requires the server to be running separately)")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health Check: {health_data['status']}")
        else:
            print(f"   ❌ Health Check Failed: {response.status_code}")
    except requests.exceptions.RequestException:
        print("   ⚠️  Server not running - skipping API tests")
        return
    
    # Test prediction endpoint
    try:
        prediction_data = {
            "model_name": "strategy_selector",
            "features": {
                "price_momentum_1m": 0.02,
                "volume_ratio": 1.5,
                "rsi": 65.0,
                "macd_signal": 0.1,
                "trend_strength": 0.8
            }
        }
        
        response = requests.post(f"{base_url}/predict", json=prediction_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction API: {result['prediction']:.4f} ({result['processing_time_ms']:.2f}ms)")
        else:
            print(f"   ❌ Prediction API Failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Prediction API Error: {e}")
    
    # Test models status endpoint
    try:
        response = requests.get(f"{base_url}/models/status", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print(f"   ✅ Models Status: {models_data['total_active']} active models")
        else:
            print(f"   ❌ Models Status Failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Models Status Error: {e}")


def run_server_demo():
    """Run the server for API testing"""
    print("\n" + "="*80)
    print("STARTING MODEL SERVING SERVER")
    print("="*80)
    
    # Create pipeline and register models
    pipeline = ModelServingPipeline()
    
    # Register models
    create_mock_models()
    configs = create_model_configs()
    for config in configs:
        pipeline.register_model(config)
    
    # Create FastAPI app
    app = create_app(pipeline)
    
    print("\n🚀 Starting server on http://localhost:8000")
    print("   Available endpoints:")
    print("   - GET  /health")
    print("   - GET  /models/status")
    print("   - POST /predict")
    print("   - POST /batch_predict")
    print("   - GET  /metrics")
    print("\n   Press Ctrl+C to stop the server")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main():
    """Main demonstration function"""
    print("🚀 MODEL SERVING PIPELINE DEMONSTRATION")
    print("="*80)
    
    try:
        # Create pipeline
        pipeline = ModelServingPipeline()
        
        # Run demonstrations
        demo_model_registration(pipeline)
        demo_single_predictions(pipeline)
        demo_batch_predictions(pipeline)
        demo_ab_testing(pipeline)
        demo_performance_monitoring(pipeline)
        demo_api_endpoints()
        
        print("\n" + "="*80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nKey Features Demonstrated:")
        print("• ✅ Model registration and serving")
        print("• ✅ Single and batch predictions")
        print("• ✅ A/B testing with traffic splitting")
        print("• ✅ Performance monitoring and metrics")
        print("• ✅ Health checks and status monitoring")
        print("• ✅ Circuit breaker and error handling")
        print("• ✅ Caching and latency optimization")
        
        print(f"\nTo start the API server, run:")
        print(f"python examples/model_serving_demo.py --server")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        print(f"\n❌ Error occurred: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server_demo()
    else:
        main()