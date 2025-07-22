#!/usr/bin/env python3
"""
Model Serving Pipeline Deployment Script

This script handles the deployment of the Model Serving Pipeline with:
- Environment setup and validation
- Model loading and registration
- Health checks and monitoring setup
- Graceful startup and shutdown
- Configuration management

Usage:
    python scripts/deploy_model_serving.py [options]
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import threading
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.model_serving_pipeline import ModelServingPipeline, ModelConfig, create_app
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelServingDeployment:
    """Handles deployment of the Model Serving Pipeline"""
    
    def __init__(self, config_path: str = "config/model_serving_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.pipeline = None
        self.server_process = None
        self.shutdown_event = threading.Event()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise e
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", False, f"Requires Python 3.8+, found {python_version.major}.{python_version.minor}"))
        
        # Check required directories
        required_dirs = ["models", "logs", "data"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                checks.append((f"Directory {dir_name}", True, "exists"))
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                checks.append((f"Directory {dir_name}", True, "created"))
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb > 1.0:  # At least 1GB free
            checks.append(("Disk space", True, f"{free_gb:.1f}GB free"))
        else:
            checks.append(("Disk space", False, f"Only {free_gb:.1f}GB free"))
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb > 2.0:  # At least 2GB available
            checks.append(("Memory", True, f"{available_gb:.1f}GB available"))
        else:
            checks.append(("Memory", False, f"Only {available_gb:.1f}GB available"))
        
        # Check Redis (optional)
        if self.config and self.config.get('cache', {}).get('type') == 'redis':
            try:
                import redis
                redis_url = self.config['cache']['redis_url']
                r = redis.from_url(redis_url)
                r.ping()
                checks.append(("Redis connection", True, "connected"))
            except Exception as e:
                checks.append(("Redis connection", False, str(e)))
        
        # Print validation results
        print("\n" + "="*60)
        print("ENVIRONMENT VALIDATION")
        print("="*60)
        
        all_passed = True
        for check_name, passed, details in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        print("="*60)
        
        if all_passed:
            logger.info("Environment validation passed")
        else:
            logger.error("Environment validation failed")
        
        return all_passed
    
    def setup_models(self) -> bool:
        """Setup and register models"""
        logger.info("Setting up models...")
        
        if not self.config or 'models' not in self.config:
            logger.error("No models configuration found")
            return False
        
        self.pipeline = ModelServingPipeline()
        
        success_count = 0
        total_count = 0
        
        for model_name, model_config in self.config['models'].items():
            total_count += 1
            try:
                # Create ModelConfig object
                config = ModelConfig(
                    name=model_config['name'],
                    version=model_config['version'],
                    model_path=model_config['model_path'],
                    model_type=model_config['model_type'],
                    max_batch_size=model_config.get('max_batch_size', 100),
                    timeout_ms=model_config.get('timeout_ms', 5000),
                    memory_limit_mb=model_config.get('memory_limit_mb', 1024),
                    warmup_requests=model_config.get('warmup_requests', 5),
                    a_b_test_weight=model_config.get('a_b_test_weight', 1.0),
                    feature_schema=model_config.get('feature_schema', {})
                )
                
                # Check if model file exists
                if not os.path.exists(config.model_path):
                    logger.warning(f"Model file not found: {config.model_path}")
                    continue
                
                # Register model
                if self.pipeline.register_model(config):
                    success_count += 1
                    logger.info(f"Registered model: {config.name}:{config.version}")
                else:
                    logger.error(f"Failed to register model: {config.name}:{config.version}")
                    
            except Exception as e:
                logger.error(f"Error setting up model {model_name}: {e}")
        
        logger.info(f"Model setup completed: {success_count}/{total_count} models registered")
        return success_count > 0
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Start the FastAPI server"""
        logger.info(f"Starting server on {host}:{port} with {workers} workers...")
        
        if not self.pipeline:
            logger.error("Pipeline not initialized")
            return False
        
        try:
            # Create FastAPI app
            app = create_app(self.pipeline)
            
            # Configure server
            server_config = self.config.get('server', {})
            host = server_config.get('host', host)
            port = server_config.get('port', port)
            workers = server_config.get('workers', workers)
            log_level = server_config.get('log_level', 'info')
            
            logger.info(f"Server configuration: {host}:{port}, workers={workers}, log_level={log_level}")
            
            # Start server
            uvicorn.run(
                app,
                host=host,
                port=port,
                workers=workers,
                log_level=log_level,
                access_log=server_config.get('access_log', True)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def health_check(self, host: str = "localhost", port: int = 8000, timeout: int = 30) -> bool:
        """Perform health check on running server"""
        import requests
        
        url = f"http://{host}:{port}/health"
        
        logger.info(f"Performing health check: {url}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        logger.info("Health check passed")
                        return True
                    else:
                        logger.warning(f"Server unhealthy: {health_data}")
                        
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed: {e}")
                
            time.sleep(2)
        
        logger.error("Health check timeout")
        return False
    
    def setup_monitoring(self):
        """Setup monitoring and metrics collection"""
        logger.info("Setting up monitoring...")
        
        monitoring_config = self.config.get('monitoring', {})
        
        if monitoring_config.get('enable_prometheus', False):
            prometheus_port = monitoring_config.get('prometheus_port', 9090)
            logger.info(f"Prometheus metrics available on port {prometheus_port}")
        
        # Setup log rotation
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'logs/model_serving.log')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.info(f"Logging to: {log_file}")
    
    def graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Received shutdown signal, shutting down gracefully...")
        self.shutdown_event.set()
        
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        
        logger.info("Shutdown completed")
        sys.exit(0)
    
    def deploy(self, validate_env: bool = True, setup_monitoring: bool = True):
        """Full deployment process"""
        print("\n🚀 STARTING MODEL SERVING PIPELINE DEPLOYMENT")
        print("="*80)
        
        try:
            # Load configuration
            self.load_config()
            
            # Validate environment
            if validate_env and not self.validate_environment():
                logger.error("Environment validation failed, aborting deployment")
                return False
            
            # Setup models
            if not self.setup_models():
                logger.error("Model setup failed, aborting deployment")
                return False
            
            # Setup monitoring
            if setup_monitoring:
                self.setup_monitoring()
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.graceful_shutdown)
            signal.signal(signal.SIGTERM, self.graceful_shutdown)
            
            # Start server
            logger.info("Deployment completed successfully")
            print("\n✅ DEPLOYMENT SUCCESSFUL")
            print("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            print(f"\n❌ DEPLOYMENT FAILED: {e}")
            return False


def create_sample_models():
    """Create sample models for testing"""
    logger.info("Creating sample models for testing...")
    
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create sample models
    models = {
        "strategy_selector_v1": RandomForestRegressor(n_estimators=10, random_state=42),
        "position_sizer_v1": LinearRegression(),
        "sentiment_analyzer_v1": RandomForestRegressor(n_estimators=5, random_state=42)
    }
    
    # Generate sample training data and train models
    for model_name, model in models.items():
        # Generate sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Train model
        model.fit(X, y)
        
        # Save model
        model_path = models_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Created sample model: {model_path}")
    
    logger.info("Sample models created successfully")


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Model Serving Pipeline")
    parser.add_argument("--config", default="config/model_serving_config.yaml", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--no-validation", action="store_true", help="Skip environment validation")
    parser.add_argument("--create-samples", action="store_true", help="Create sample models")
    parser.add_argument("--health-check-only", action="store_true", help="Only perform health check")
    
    args = parser.parse_args()
    
    # Create sample models if requested
    if args.create_samples:
        create_sample_models()
        return
    
    # Initialize deployment
    deployment = ModelServingDeployment(args.config)
    
    # Health check only
    if args.health_check_only:
        success = deployment.health_check(args.host, args.port)
        sys.exit(0 if success else 1)
    
    # Full deployment
    try:
        # Deploy
        if deployment.deploy(validate_env=not args.no_validation):
            # Start server
            deployment.start_server(args.host, args.port, args.workers)
        else:
            logger.error("Deployment failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()