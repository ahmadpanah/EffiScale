import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import logging
from dataclasses import dataclass

@dataclass
class Pattern:
    """Data class for storing identified patterns."""
    type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    metrics: List[str]
    description: str
    severity: float
    suggested_action: Optional[str] = None

class PatternAnalyzer:
    """
    Analyzes metrics patterns to identify trends, anomalies, and seasonal patterns.
    Provides insights for scaling decisions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pattern analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis settings
        self.window_size = config.get('analysis', {}).get('window_size', 3600)  # 1 hour
        self.min_pattern_confidence = config.get('analysis', {}).get('min_confidence', 0.8)
        self.anomaly_threshold = config.get('analysis', {}).get('anomaly_threshold', 2.0)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)  # For dimensionality reduction
        
        # Pattern cache
        self._pattern_cache: Dict[str, List[Pattern]] = {}

    async def analyze_metrics(
        self,
        metrics: List[Dict],
        metric_types: Optional[List[str]] = None
    ) -> List[Pattern]:
        """
        Analyze metrics data to identify patterns.
        
        Args:
            metrics: List of metric data points
            metric_types: Optional list of metric types to analyze
            
        Returns:
            List of identified patterns
        """
        try:
            if not metrics:
                return []
                
            # Convert metrics to DataFrame
            df = self._prepare_dataframe(metrics, metric_types)
            
            # Identify patterns
            patterns = []
            patterns.extend(await self._analyze_trends(df))
            patterns.extend(await self._analyze_seasonality(df))
            patterns.extend(await self._analyze_anomalies(df))
            patterns.extend(await self._analyze_correlations(df))
            
            # Cache patterns
            self._update_pattern_cache(patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing metrics: {str(e)}")
            return []

    def _prepare_dataframe(
        self,
        metrics: List[Dict],
        metric_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare metrics data for analysis.
        
        Args:
            metrics: Raw metrics data
            metric_types: Optional metric types filter
            
        Returns:
            Prepared DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Filter metric types
        if metric_types:
            df = df[df['name'].isin(metric_types)]
        
        # Pivot metrics to columns
        df = df.pivot_table(
            index='timestamp',
            columns='name',
            values='value',
            aggfunc='mean'
        )
        
        # Resample to regular intervals
        df = df.resample('1min').mean()
        
        # Interpolate missing values
        df = df.interpolate(method='time')
        
        return df

    async def _analyze_trends(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Analyze metric trends.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            List of identified trend patterns
        """
        patterns = []
        
        for column in df.columns:
            try:
                # Calculate rolling statistics
                rolling_mean = df[column].rolling(window=60).mean()
                rolling_std = df[column].rolling(window=60).std()
                
                # Perform trend analysis
                trend_coefficient = np.polyfit(
                    range(len(df[column])),
                    df[column].fillna(method='ffill'),
                    deg=1
                )[0]
                
                # Check for significant trends
                if abs(trend_coefficient) > self.config.get('analysis', {}).get('trend_threshold', 0.1):
                    trend_type = 'increasing' if trend_coefficient > 0 else 'decreasing'
                    severity = abs(trend_coefficient)
                    
                    patterns.append(Pattern(
                        type='trend',
                        confidence=min(abs(trend_coefficient) * 2, 1.0),
                        start_time=df.index[0].to_pydatetime(),
                        end_time=df.index[-1].to_pydatetime(),
                        metrics=[column],
                        description=f'{trend_type.capitalize()} trend detected in {column}',
                        severity=severity,
                        suggested_action=self._get_trend_suggestion(column, trend_type, severity)
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error analyzing trend for {column}: {str(e)}")
                
        return patterns

    async def _analyze_seasonality(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Analyze seasonal patterns in metrics.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            List of identified seasonal patterns
        """
        patterns = []
        
        for column in df.columns:
            try:
                if len(df) < 2:  # Need at least 2 periods for seasonal analysis
                    continue
                    
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    df[column],
                    period=60,  # 1-hour seasonality
                    model='additive'
                )
                
                # Calculate seasonality strength
                seasonality_strength = np.std(decomposition.seasonal) / np.std(df[column])
                
                if seasonality_strength > self.config.get('analysis', {}).get('seasonality_threshold', 0.2):
                    patterns.append(Pattern(
                        type='seasonality',
                        confidence=min(seasonality_strength * 2, 1.0),
                        start_time=df.index[0].to_pydatetime(),
                        end_time=df.index[-1].to_pydatetime(),
                        metrics=[column],
                        description=f'Seasonal pattern detected in {column}',
                        severity=seasonality_strength,
                        suggested_action=self._get_seasonality_suggestion(column, seasonality_strength)
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error analyzing seasonality for {column}: {str(e)}")
                
        return patterns

    async def _analyze_anomalies(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Detect anomalies in metrics.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            List of identified anomalies
        """
        patterns = []
        
        for column in df.columns:
            try:
                # Calculate rolling statistics
                rolling_mean = df[column].rolling(window=60).mean()
                rolling_std = df[column].rolling(window=60).std()
                
                # Detect anomalies using Z-score
                z_scores = (df[column] - rolling_mean) / rolling_std
                anomalies = np.abs(z_scores) > self.anomaly_threshold
                
                if anomalies.any():
                    # Group consecutive anomalies
                    anomaly_groups = self._group_consecutive_anomalies(anomalies)
                    
                    for start_idx, end_idx in anomaly_groups:
                        severity = float(np.max(np.abs(z_scores[start_idx:end_idx])))
                        
                        patterns.append(Pattern(
                            type='anomaly',
                            confidence=min(severity / self.anomaly_threshold, 1.0),
                            start_time=df.index[start_idx].to_pydatetime(),
                            end_time=df.index[end_idx].to_pydatetime(),
                            metrics=[column],
                            description=f'Anomaly detected in {column}',
                            severity=severity,
                            suggested_action=self._get_anomaly_suggestion(column, severity)
                        ))
                    
            except Exception as e:
                self.logger.error(f"Error analyzing anomalies for {column}: {str(e)}")
                
        return patterns

    async def _analyze_correlations(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Analyze correlations between metrics.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            List of identified correlation patterns
        """
        patterns = []
        
        try:
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Find highly correlated pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    correlation = corr_matrix.iloc[i, j]
                    
                    if abs(correlation) > self.config.get('analysis', {}).get('correlation_threshold', 0.8):
                        metric1 = corr_matrix.columns[i]
                        metric2 = corr_matrix.columns[j]
                        
                        patterns.append(Pattern(
                            type='correlation',
                            confidence=abs(correlation),
                            start_time=df.index[0].to_pydatetime(),
                            end_time=df.index[-1].to_pydatetime(),
                            metrics=[metric1, metric2],
                            description=f'Strong correlation between {metric1} and {metric2}',
                            severity=abs(correlation),
                            suggested_action=self._get_correlation_suggestion(metric1, metric2, correlation)
                        ))
                    
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")
            
        return patterns

    def _group_consecutive_anomalies(
        self,
        anomalies: pd.Series
    ) -> List[Tuple[int, int]]:
        """
        Group consecutive anomalies into ranges.
        
        Args:
            anomalies: Boolean series indicating anomalies
            
        Returns:
            List of (start_index, end_index) tuples
        """
        groups = []
        start_idx = None
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and start_idx is None:
                start_idx = i
            elif not is_anomaly and start_idx is not None:
                groups.append((start_idx, i))
                start_idx = None
                
        if start_idx is not None:
            groups.append((start_idx, len(anomalies)))
            
        return groups

    def _get_trend_suggestion(
        self,
        metric: str,
        trend_type: str,
        severity: float
    ) -> str:
        """Generate suggestion based on trend pattern."""
        if 'cpu' in metric.lower():
            return f"Consider {'scaling up' if trend_type == 'increasing' else 'scaling down'} CPU resources"
        elif 'memory' in metric.lower():
            return f"Consider {'increasing' if trend_type == 'increasing' else 'decreasing'} memory allocation"
        return f"Monitor {metric} for continued {trend_type} trend"

    def _get_seasonality_suggestion(
        self,
        metric: str,
        strength: float
    ) -> str:
        """Generate suggestion based on seasonal pattern."""
        return f"Consider implementing predictive scaling for {metric} based on seasonal pattern"

    def _get_anomaly_suggestion(
        self,
        metric: str,
        severity: float
    ) -> str:
        """Generate suggestion based on anomaly pattern."""
        if severity > 3.0:
            return f"Urgent: Investigate abnormal behavior in {metric}"
        return f"Monitor {metric} for continued anomalies"

    def _get_correlation_suggestion(
        self,
        metric1: str,
        metric2: str,
        correlation: float
    ) -> str:
        """Generate suggestion based on correlation pattern."""
        return f"Consider joint scaling strategy for {metric1} and {metric2}"

    def _update_pattern_cache(self, patterns: List[Pattern]):
        """Update pattern cache with new patterns."""
        current_time = datetime.utcnow()
        
        # Add new patterns
        for pattern in patterns:
            pattern_key = f"{pattern.type}_{','.join(pattern.metrics)}"
            
            if pattern_key not in self._pattern_cache:
                self._pattern_cache[pattern_key] = []
                
            self._pattern_cache[pattern_key].append(pattern)
        
        # Remove old patterns
        for patterns in self._pattern_cache.values():
            patterns[:] = [
                p for p in patterns
                if (current_time - p.end_time).total_seconds() < self.window_size
            ]

    def get_recent_patterns(
        self,
        pattern_type: Optional[str] = None,
        metric: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Pattern]:
        """
        Get recent patterns from cache.
        
        Args:
            pattern_type: Optional pattern type filter
            metric: Optional metric name filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching patterns
        """
        patterns = []
        
        for cached_patterns in self._pattern_cache.values():
            for pattern in cached_patterns:
                if pattern.confidence < min_confidence:
                    continue
                    
                if pattern_type and pattern.type != pattern_type:
                    continue
                    
                if metric and metric not in pattern.metrics:
                    continue
                    
                patterns.append(pattern)
                
        return patterns

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'analysis': {
            'window_size': 3600,
            'min_confidence': 0.8,
            'anomaly_threshold': 2.0,
            'trend_threshold': 0.1,
            'seasonality_threshold': 0.2,
            'correlation_threshold': 0.8
        }
    }
    
    # Initialize analyzer
    analyzer = PatternAnalyzer(config)
    
    # Example metrics data
    metrics_data = [
        {
            'timestamp': datetime.utcnow() - timedelta(minutes=i),
            'name': 'cpu_usage',
            'value': 50 + i * 0.5 + np.random.normal(0, 2)
        }
        for i in range(120)
    ]
    
    # Analyze patterns
    async def main():
        patterns = await analyzer.analyze_metrics(metrics_data)
        for pattern in patterns:
            print(f"Pattern: {pattern.type}")
            print(f"Description: {pattern.description}")
            print(f"Confidence: {pattern.confidence:.2f}")
            print(f"Suggestion: {pattern.suggested_action}")
            print("---")
    
    # Run analysis
    import asyncio
    asyncio.run(main())