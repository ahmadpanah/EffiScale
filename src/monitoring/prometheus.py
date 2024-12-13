from typing import Dict, List, Optional, Union
from datetime import datetime
import aiohttp
import logging
from urllib.parse import urljoin
import json

class PrometheusClient:
    """
    Client for querying Prometheus metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Prometheus client.
        
        Args:
            config: Configuration dictionary
        """
        self.prometheus_url = config.get('prometheus_url', 'http://localhost:9090')
        self.query_timeout = config.get('query_timeout', 10)
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.QUERY_RANGE_ENDPOINT = '/api/v1/query_range'
        self.QUERY_INSTANT_ENDPOINT = '/api/v1/query'

    async def _make_request(
        self,
        endpoint: str,
        params: Dict
    ) -> Optional[Dict]:
        """
        Make HTTP request to Prometheus API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data or None if request failed
        """
        url = urljoin(self.prometheus_url, endpoint)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=self.query_timeout
                ) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Prometheus API request failed: {response.status} - {await response.text()}"
                        )
                        return None
                    
                    data = await response.json()
                    
                    if data['status'] != 'success':
                        self.logger.error(
                            f"Prometheus query failed: {data.get('error', 'Unknown error')}"
                        )
                        return None
                    
                    return data
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP request to Prometheus failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Prometheus response: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error querying Prometheus: {str(e)}")
            return None

    async def query_range(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        step: str = '1m',
        labels: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Query Prometheus for metric values over a time range.
        
        Args:
            metric_name: Name of the metric to query
            start_time: Start time
            end_time: End time
            step: Query resolution step
            labels: Optional metric labels to filter
            
        Returns:
            List of metric values with timestamps
        """
        # Build Prometheus query
        query = metric_name
        if labels:
            label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
            query = f'{metric_name}{{{label_str}}}'

        # Prepare request parameters
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': step
        }

        # Make request to Prometheus
        response_data = await self._make_request(
            self.QUERY_RANGE_ENDPOINT,
            params
        )
        
        if not response_data:
            return []

        try:
            # Process response data
            result = []
            for data in response_data['data']['result']:
                # Extract metric labels
                metric_labels = data.get('metric', {})
                
                # Process values
                for timestamp, value in data['values']:
                    result.append({
                        'timestamp': datetime.fromtimestamp(timestamp),
                        'value': float(value),
                        'labels': metric_labels
                    })
            
            return result
            
        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to process Prometheus response: {str(e)}")
            return []

    async def query_instant(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """
        Query Prometheus for instant metric value.
        
        Args:
            metric_name: Name of the metric to query
            labels: Optional metric labels to filter
            
        Returns:
            Current metric value or None if not found
        """
        # Build Prometheus query
        query = metric_name
        if labels:
            label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
            query = f'{metric_name}{{{label_str}}}'

        # Prepare request parameters
        params = {
            'query': query,
            'time': datetime.utcnow().timestamp()
        }

        # Make request to Prometheus
        response_data = await self._make_request(
            self.QUERY_INSTANT_ENDPOINT,
            params
        )
        
        if not response_data:
            return None

        try:
            # Process response data
            results = response_data['data']['result']
            
            if not results:
                return None
                
            # Get the first result's value
            return float(results[0]['value'][1])
            
        except (KeyError, ValueError, IndexError) as e:
            self.logger.error(f"Failed to process Prometheus instant query response: {str(e)}")
            return None

    async def get_metric_labels(self, metric_name: str) -> List[str]:
        """
        Get all available labels for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of label names
        """
        params = {
            'query': f'labels({metric_name})'
        }
        
        response_data = await self._make_request(
            self.QUERY_INSTANT_ENDPOINT,
            params
        )
        
        if not response_data:
            return []
            
        try:
            return response_data['data']
        except KeyError:
            return []

    async def get_metric_label_values(
        self,
        metric_name: str,
        label: str
    ) -> List[str]:
        """
        Get all values for a specific label of a metric.
        
        Args:
            metric_name: Name of the metric
            label: Label name
            
        Returns:
            List of label values
        """
        params = {
            'query': f'label_values({metric_name}, {label})'
        }
        
        response_data = await self._make_request(
            self.QUERY_INSTANT_ENDPOINT,
            params
        )
        
        if not response_data:
            return []
            
        try:
            return response_data['data']
        except KeyError:
            return []