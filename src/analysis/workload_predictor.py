import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import tensorflow as tf
from enum import Enum
import joblib
import json
import asyncio

class PredictionModel(Enum):
    """Supported prediction models."""
    LINEAR = "linear"
    PROPHET = "prophet"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"

@dataclass
class PredictionResult:
    """Container for prediction results."""
    timestamp: datetime
    value: float
    confidence: float
    model: PredictionModel
    features_used: List[str]

class WorkloadPredictor:
    """
    Predicts future workload based on historical metrics.
    Supports multiple prediction models and ensemble methods.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the workload predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.models: Dict[PredictionModel, object] = {}
        self.model_weights: Dict[PredictionModel, float] = {}
        
        # Training settings
        self.min_training_points = config.get('training', {}).get('min_points', 100)
        self.training_window = config.get('training', {}).get('window', 7 * 24 * 3600)  # 7 days
        self.prediction_horizon = config.get('prediction', {}).get('horizon', 3600)  # 1 hour
        
        # Feature settings
        self.feature_columns = config.get('features', [
            'cpu_usage',
            'memory_usage',
            'request_count',
            'response_time'
        ])
        
        # Initialize models
        self._initialize_models()
        
        # Metrics history
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Prediction history for accuracy tracking
        self.prediction_history: List[Tuple[PredictionResult, float]] = []
        
        # Start background tasks
        if config.get('auto_train', {}).get('enabled', True):
            asyncio.create_task(self._auto_training_loop())

    def _initialize_models(self):
        """Initialize prediction models based on configuration."""
        enabled_models = self.config.get('models', {})
        
        if enabled_models.get('linear', True):
            self.models[PredictionModel.LINEAR] = LinearRegression()
            self.model_weights[PredictionModel.LINEAR] = 1.0
            
        if enabled_models.get('prophet', True):
            self.models[PredictionModel.PROPHET] = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            self.model_weights[PredictionModel.PROPHET] = 1.0
            
        if enabled_models.get('lstm', True):
            self.models[PredictionModel.LSTM] = self._build_lstm_model()
            self.model_weights[PredictionModel.LSTM] = 1.0
            
        if enabled_models.get('random_forest', True):
            self.models[PredictionModel.RANDOM_FOREST] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model_weights[PredictionModel.RANDOM_FOREST] = 1.0

    def _build_lstm_model(self) -> tf.keras.Model:
        """Build and compile LSTM model."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(24, len(self.feature_columns))),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

    async def predict_workload(
        self,
        horizon: Optional[int] = None,
        model_type: Optional[PredictionModel] = None
    ) -> List[PredictionResult]:
        """
        Predict future workload.
        
        Args:
            horizon: Prediction horizon in seconds
            model_type: Specific model to use for prediction
            
        Returns:
            List of prediction results
        """
        horizon = horizon or self.prediction_horizon
        
        try:
            # Prepare features
            features_df = self._prepare_features()
            
            if features_df.empty:
                return []
            
            predictions = []
            
            if model_type:
                # Use specific model
                if model_type not in self.models:
                    raise ValueError(f"Model not available: {model_type}")
                predictions.extend(
                    await self._predict_with_model(model_type, features_df, horizon)
                )
            else:
                # Use ensemble prediction
                all_predictions = []
                for model_type in self.models:
                    model_predictions = await self._predict_with_model(
                        model_type,
                        features_df,
                        horizon
                    )
                    all_predictions.append(model_predictions)
                
                # Combine predictions using weighted average
                predictions = self._combine_predictions(all_predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting workload: {str(e)}")
            return []

    async def _predict_with_model(
        self,
        model_type: PredictionModel,
        features_df: pd.DataFrame,
        horizon: int
    ) -> List[PredictionResult]:
        """Make predictions using a specific model."""
        try:
            model = self.models[model_type]
            
            if model_type == PredictionModel.PROPHET:
                return await self._predict_prophet(model, features_df, horizon)
            elif model_type == PredictionModel.LSTM:
                return await self._predict_lstm(model, features_df, horizon)
            else:
                return await self._predict_sklearn(model, model_type, features_df, horizon)
                
        except Exception as e:
            self.logger.error(f"Error predicting with {model_type}: {str(e)}")
            return []

    async def _predict_prophet(
        self,
        model: Prophet,
        features_df: pd.DataFrame,
        horizon: int
    ) -> List[PredictionResult]:
        """Make predictions using Prophet model."""
        # Prepare Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': features_df.index,
            'y': features_df['cpu_usage']  # Primary metric for Prophet
        })
        
        # Fit model if needed
        model.fit(prophet_df)
        
        # Make future DataFrame
        future_dates = pd.date_range(
            start=prophet_df['ds'].max(),
            periods=int(horizon / 60) + 1,  # Convert seconds to minutes
            freq='1min'
        )
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Predict
        forecast = model.predict(future_df)
        
        return [
            PredictionResult(
                timestamp=row['ds'].to_pydatetime(),
                value=float(row['yhat']),
                confidence=float(row['yhat_upper'] - row['yhat_lower']) / 4,  # Approximate 95% CI
                model=PredictionModel.PROPHET,
                features_used=['cpu_usage']
            )
            for _, row in forecast.iterrows()
        ]

    async def _predict_lstm(
        self,
        model: tf.keras.Model,
        features_df: pd.DataFrame,
        horizon: int
    ) -> List[PredictionResult]:
        """Make predictions using LSTM model."""
        # Prepare sequences
        sequence_length = 24
        sequences = self._prepare_sequences(features_df, sequence_length)
        
        if len(sequences) == 0:
            return []
        
        # Predict
        predictions = model.predict(sequences[-1:])
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=features_df.index[-1],
            periods=len(predictions) + 1,
            freq='1min'
        )[1:]
        
        return [
            PredictionResult(
                timestamp=ts.to_pydatetime(),
                value=float(val),
                confidence=0.8,  # Fixed confidence for LSTM
                model=PredictionModel.LSTM,
                features_used=self.feature_columns
            )
            for ts, val in zip(timestamps, predictions.flatten())
        ]

    async def _predict_sklearn(
        self,
        model: Union[LinearRegression, RandomForestRegressor],
        model_type: PredictionModel,
        features_df: pd.DataFrame,
        horizon: int
    ) -> List[PredictionResult]:
        """Make predictions using scikit-learn models."""
        # Prepare features
        X = features_df.values
        y = features_df['cpu_usage'].values
        
        # Fit model if needed
        model.fit(X, y)
        
        # Prepare future features
        future_features = self._prepare_future_features(features_df, horizon)
        
        # Predict
        predictions = model.predict(future_features)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=features_df.index[-1],
            periods=len(predictions) + 1,
            freq='1min'
        )[1:]
        
        # Calculate confidence
        if isinstance(model, RandomForestRegressor):
            confidences = [
                tree.predict(future_features)
                for tree in model.estimators_
            ]
            confidence_intervals = np.std(confidences, axis=0)
        else:
            confidence_intervals = np.ones(len(predictions)) * 0.7  # Fixed for linear
        
        return [
            PredictionResult(
                timestamp=ts.to_pydatetime(),
                value=float(val),
                confidence=float(conf),
                model=model_type,
                features_used=self.feature_columns
            )
            for ts, val, conf in zip(timestamps, predictions, confidence_intervals)
        ]

    def _combine_predictions(
        self,
        predictions_list: List[List[PredictionResult]]
    ) -> List[PredictionResult]:
        """Combine predictions from multiple models using weighted average."""
        if not predictions_list:
            return []
            
        # Group predictions by timestamp
        predictions_by_timestamp = {}
        for predictions in predictions_list:
            for pred in predictions:
                if pred.timestamp not in predictions_by_timestamp:
                    predictions_by_timestamp[pred.timestamp] = []
                predictions_by_timestamp[pred.timestamp].append(pred)
        
        # Combine predictions for each timestamp
        combined_predictions = []
        for timestamp, preds in predictions_by_timestamp.items():
            weights = [
                self.model_weights[pred.model] * pred.confidence
                for pred in preds
            ]
            weight_sum = sum(weights)
            
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
                
                weighted_value = sum(
                    pred.value * weight
                    for pred, weight in zip(preds, weights)
                )
                
                average_confidence = np.mean([pred.confidence for pred in preds])
                
                combined_predictions.append(PredictionResult(
                    timestamp=timestamp,
                    value=weighted_value,
                    confidence=average_confidence,
                    model=PredictionModel.ENSEMBLE,
                    features_used=self.feature_columns
                ))
        
        return sorted(combined_predictions, key=lambda x: x.timestamp)

    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features DataFrame from metrics history."""
        # Combine all metrics into DataFrame
        data = []
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue
            
            df = pd.DataFrame(
                history,
                columns=['timestamp', metric_name]
            ).set_index('timestamp')
            
            data.append(df)
        
        if not data:
            return pd.DataFrame()
            
        # Merge all metrics
        features_df = pd.concat(data, axis=1)
        
        # Resample to regular intervals
        features_df = features_df.resample('1min').mean()
        
        # Fill missing values
        features_df = features_df.interpolate(method='time')
        
        return features_df

    def _prepare_sequences(
        self,
        features_df: pd.DataFrame,
        sequence_length: int
    ) -> np.ndarray:
        """Prepare sequences for LSTM model."""
        values = features_df.values
        sequences = []
        
        for i in range(len(values) - sequence_length):
            sequences.append(values[i:(i + sequence_length)])
            
        return np.array(sequences)

    def _prepare_future_features(
        self,
        features_df: pd.DataFrame,
        horizon: int
    ) -> np.ndarray:
        """Prepare feature matrix for future predictions."""
        # Simple forward projection of features
        last_values = features_df.iloc[-1].values
        num_steps = int(horizon / 60)  # Convert seconds to minutes
        
        return np.tile(last_values, (num_steps, 1))

    async def _auto_training_loop(self):
        """Background task for automatic model training."""
        while True:
            try:
                await self._train_models()
                await self._update_model_weights()
                
                # Sleep until next training
                training_interval = self.config.get('auto_train', {}).get('interval', 3600)
                await asyncio.sleep(training_interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-training loop: {str(e)}")
                await asyncio.sleep(60)  # Sleep shortly before retrying

    async def _train_models(self):
        """Train all models with recent data."""
        features_df = self._prepare_features()
        
        if len(features_df) < self.min_training_points:
            return
            
        for model_type, model in self.models.items():
            try:
                if model_type == PredictionModel.PROPHET:
                    prophet_df = pd.DataFrame({
                        'ds': features_df.index,
                        'y': features_df['cpu_usage']
                    })
                    model.fit(prophet_df)
                    
                elif model_type == PredictionModel.LSTM:
                    sequences = self._prepare_sequences(features_df, 24)
                    if len(sequences) > 0:
                        X = sequences[:, :-1, :]
                        y = sequences[:, -1, 0]  # Predict CPU usage
                        model.fit(X, y, epochs=10, verbose=0)
                        
                else:
                    X = features_df.values
                    y = features_df['cpu_usage'].values
                    model.fit(X, y)
                    
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {str(e)}")

    async def _update_model_weights(self):
        """Update model weights based on prediction accuracy."""
        if not self.prediction_history:
            return
            
        # Calculate accuracy for each model
        model_errors = {model_type: [] for model_type in self.models}
        
        for prediction, actual in self.prediction_history:
            error = abs(prediction.value - actual)
            model_errors[prediction.model].append(error)
        
        # Calculate average error for each model
        avg_errors = {
            model_type: np.mean(errors) if errors else float('inf')
            for model_type, errors in model_errors.items()
        }
        
        # Update weights inversely proportional to error
        total_inverse_error = sum(1/error for error in avg_errors.values() if error > 0)
        
        if total_inverse_error > 0:
            self.model_weights = {
                model_type: (1/error) / total_inverse_error
                for model_type, error in avg_errors.items()
            }

    def add_metric_value(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """Add a new metric value to history."""
        timestamp = timestamp or datetime.utcnow()
        
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
            
        self.metrics_history[metric_name].append((timestamp, value))
        
        # Remove old entries
        cutoff_time = timestamp - timedelta(seconds=self.training_window)
        self.metrics_history[metric_name] = [
            (ts, val) for ts, val in self.metrics_history[metric_name]
            if ts > cutoff_time
        ]

    def record_prediction_accuracy(
        self,
        prediction: PredictionResult,
        actual_value: float
    ):
        """Record prediction accuracy for model weight updates."""
        self.prediction_history.append((prediction, actual_value))
        
        # Keep only recent history
        max_history = self.config.get('accuracy_history_size', 1000)
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]

    def save_models(self, directory: str):
        """Save trained models to disk."""
        for model_type, model in self.models.items():
            try:
                if model_type == PredictionModel.PROPHET:
                    with open(f"{directory}/prophet_model.json", 'w') as f:
                        model.serialize_specify().to_json(f)
                elif model_type == PredictionModel.LSTM:
                    model.save(f"{directory}/lstm_model")
                else:
                    joblib.dump(model, f"{directory}/{model_type.value}_model.joblib")
                    
            except Exception as e:
                self.logger.error(f"Error saving {model_type} model: {str(e)}")

    def load_models(self, directory: str):
        """Load trained models from disk."""
        for model_type in self.models:
            try:
                if model_type == PredictionModel.PROPHET:
                    with open(f"{directory}/prophet_model.json", 'r') as f:
                        self.models[model_type] = Prophet.from_json(f.read())
                elif model_type == PredictionModel.LSTM:
                    self.models[model_type] = tf.keras.models.load_model(
                        f"{directory}/lstm_model"
                    )
                else:
                    self.models[model_type] = joblib.load(
                        f"{directory}/{model_type.value}_model.joblib"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error loading {model_type} model: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'models': {
            'linear': True,
            'prophet': True,
            'lstm': True,
            'random_forest': True
        },
        'training': {
            'min_points': 100,
            'window': 7 * 24 * 3600  # 7 days
        },
        'prediction': {
            'horizon': 3600  # 1 hour
        },
        'auto_train': {
            'enabled': True,
            'interval': 3600  # 1 hour
        },
        'features': [
            'cpu_usage',
            'memory_usage',
            'request_count',
            'response_time'
        ]
    }
    
    # Initialize predictor
    predictor = WorkloadPredictor(config)
    
    # Example prediction
    async def main():
        # Add some sample data
        for i in range(1000):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            predictor.add_metric_value('cpu_usage', 50 + np.sin(i/100) * 20, timestamp)
            predictor.add_metric_value('memory_usage', 60 + np.cos(i/100) * 15, timestamp)
        
        # Make predictions
        predictions = await predictor.predict_workload(
            horizon=1800,  # 30 minutes
            model_type=PredictionModel.ENSEMBLE
        )
        
        # Print predictions
        for pred in predictions:
            print(f"Time: {pred.timestamp}, Value: {pred.value:.2f}, "
                  f"Confidence: {pred.confidence:.2f}, Model: {pred.model.value}")
    
    # Run example
    asyncio.run(main())