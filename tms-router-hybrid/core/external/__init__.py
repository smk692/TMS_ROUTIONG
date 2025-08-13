"""
TMS Router Hybrid - 외부 API 클라이언트
"""

from .weather_client import WeatherClient, OpenWeatherMapClient, WeatherData
from .traffic_client import TrafficClient, HereTrafficClient, TrafficData, TrafficIncident
from .routing_client import RoutingClient, KakaoRoutingClient, RouteInfo, MatrixResult
from .cache_manager import CacheManager, get_cache_manager, cached, cache_weather, cache_traffic, cache_routing

__all__ = [
    # Weather
    'WeatherClient', 'OpenWeatherMapClient', 'WeatherData',
    
    # Traffic  
    'TrafficClient', 'HereTrafficClient', 'TrafficData', 'TrafficIncident',
    
    # Routing
    'RoutingClient', 'KakaoRoutingClient', 'RouteInfo', 'MatrixResult',
    
    # Caching
    'CacheManager', 'get_cache_manager', 'cached', 
    'cache_weather', 'cache_traffic', 'cache_routing'
]