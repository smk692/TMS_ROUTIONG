"""
TMS 배차 시스템 설정 관리
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


@dataclass
class AlgorithmSettings:
    """알고리즘 설정"""
    small_order_threshold: int = 30        # 소규모 주문 임계값
    medium_order_threshold: int = 100      # 중규모 주문 임계값 
    large_order_threshold: int = 300       # 대규모 주문 임계값
    
    # 시간 제한 (초)
    nearest_neighbor_time_limit: int = 30
    genetic_algorithm_time_limit: int = 300
    simulated_annealing_time_limit: int = 600
    
    # 품질 임계값
    quality_threshold: float = 0.8
    early_stopping_enabled: bool = True


@dataclass  
class ExternalAPISettings:
    """외부 API 설정"""
    # OpenWeatherMap
    openweather_api_key: Optional[str] = None
    openweather_timeout: int = 10
    
    # HERE Maps
    here_api_key: Optional[str] = None  
    here_timeout: int = 15
    
    # 카카오맵
    kakao_rest_api_key: Optional[str] = None
    kakao_timeout: int = 10
    
    # API 호출 제한
    max_concurrent_requests: int = 10
    requests_per_second: int = 5
    
    def __post_init__(self):
        """환경변수에서 API 키 로드"""
        self.openweather_api_key = self.openweather_api_key or os.getenv('OPENWEATHER_API_KEY')
        self.here_api_key = self.here_api_key or os.getenv('HERE_API_KEY') 
        self.kakao_rest_api_key = self.kakao_rest_api_key or os.getenv('KAKAO_REST_API_KEY')


@dataclass  
class DatabaseSettings:
    """데이터베이스 설정"""
    host: str = "localhost"
    port: int = 3306
    database: str = "tms_db"
    user: str = "tms_user"
    password: str = "tms_password"
    
    # 연결 풀 설정
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1시간
    
    # SQLAlchemy 설정
    echo_sql: bool = False
    
    def __post_init__(self):
        """환경변수에서 데이터베이스 설정 로드"""
        self.host = os.getenv('MYSQL_HOST', self.host)
        self.port = int(os.getenv('MYSQL_PORT', str(self.port)))
        self.database = os.getenv('MYSQL_DATABASE', self.database)
        self.user = os.getenv('MYSQL_USER', self.user)
        self.password = os.getenv('MYSQL_PASSWORD', self.password)
        self.echo_sql = os.getenv('MYSQL_ECHO_SQL', 'false').lower() == 'true'
    
    @property
    def database_url(self) -> str:
        """데이터베이스 연결 URL 생성"""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset=utf8mb4"


@dataclass
class CacheSettings:
    """캐시 설정"""
    cache_dir: str = "./cache"
    memory_size_mb: int = 100
    
    # TTL 설정 (분)
    weather_cache_ttl: int = 30
    traffic_cache_ttl: int = 15  
    routing_cache_ttl: int = 60
    
    # 자동 정리
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 6


@dataclass
class VehicleSettings:
    """차량 설정"""
    default_max_capacity: int = 40
    default_safe_capacity: int = 35
    
    # 경험도별 계수
    experience_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'BEGINNER': 0.70,    # 신입 - 70%
        'JUNIOR': 0.85,      # 초급 - 85%
        'INTERMEDIATE': 1.00, # 중급 - 100%
        'SENIOR': 1.15,      # 고급 - 115%
        'EXPERT': 1.30       # 전문가 - 130%
    })


@dataclass
class WeatherSettings:
    """날씨 설정"""
    # 날씨별 영향 계수 (배송량 조정)
    weather_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'clear': 1.1,        # 맑음 +10%
        'clouds': 1.0,       # 구름 변화없음
        'rain': 0.8,         # 비 -20%
        'heavy_rain': 0.6,   # 폭우 -40%
        'snow': 0.5,         # 눈 -50%
        'storm': 0.3         # 폭풍 -70%
    })
    
    # 심각도 임계값
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'good': 1.5,         # 양호 (1.0-1.5)
        'fair': 2.5,         # 보통 (1.5-2.5)
        'poor': 3.5,         # 나쁨 (2.5-3.5)  
        'bad': 4.5,          # 매우 나쁨 (3.5-4.5)
        'severe': 5.0        # 극한 (4.5-5.0)
    })


@dataclass
class TrafficSettings:
    """교통 설정"""
    # 정체도별 영향 계수
    congestion_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'free_flow': 1.1,    # 원활 +10% (0.0-0.2)
        'normal': 1.0,       # 보통 변화없음 (0.2-0.6)
        'congested': 0.8,    # 정체 -20% (0.6-0.8)
        'heavy': 0.6         # 심각한정체 -40% (0.8-1.0)
    })
    
    # 정체 수준 임계값
    congestion_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'free_flow': 0.2,
        'normal': 0.6,
        'congested': 0.8,
        'heavy': 1.0
    })


class TMSSettings(BaseSettings):
    """TMS 전체 설정 (Pydantic 기반)"""
    
    # 기본 설정
    app_name: str = Field(default="TMS Router Hybrid")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    
    # 로깅 설정
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    
    # 하위 설정들
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    algorithm: AlgorithmSettings = Field(default_factory=AlgorithmSettings)
    external_api: ExternalAPISettings = Field(default_factory=ExternalAPISettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    vehicle: VehicleSettings = Field(default_factory=VehicleSettings)
    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    traffic: TrafficSettings = Field(default_factory=TrafficSettings)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        "extra": "allow"
    }
        
        # 환경변수 예시:
        # TMS_DEBUG=true
        # TMS_EXTERNAL_API__OPENWEATHER_API_KEY=your_key
        # TMS_CACHE__MEMORY_SIZE_MB=200
    
    @field_validator('log_level')
    def validate_log_level(cls, v):
        """로그 레벨 검증"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'로그 레벨은 다음 중 하나여야 함: {valid_levels}')
        return v.upper()
    
    @property
    def database_url(self) -> str:
        """데이터베이스 연결 URL"""
        return self.database.database_url
    
    def setup_logging(self):
        """로깅 설정"""
        log_config = {
            'level': getattr(logging, self.log_level),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        
        if self.log_file:
            log_config['filename'] = self.log_file
            log_config['filemode'] = 'a'
        
        logging.basicConfig(**log_config)
        
        # 디버그 모드에서는 외부 라이브러리 로그도 표시
        if self.debug:
            logging.getLogger('aiohttp').setLevel(logging.DEBUG)
            logging.getLogger('diskcache').setLevel(logging.DEBUG)
    
    def get_api_keys_status(self) -> Dict[str, bool]:
        """API 키 설정 상태 확인"""
        return {
            'openweather': bool(self.external_api.openweather_api_key and 
                              self.external_api.openweather_api_key != 'demo_key'),
            'here': bool(self.external_api.here_api_key and 
                        self.external_api.here_api_key != 'demo_key'),
            'kakao': bool(self.external_api.kakao_rest_api_key and 
                         self.external_api.kakao_rest_api_key != 'demo_key')
        }
    
    def save_to_file(self, file_path: str):
        """설정을 파일로 저장"""
        settings_dict = self.model_dump()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'TMSSettings':
        """파일에서 설정 로드"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없음: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            settings_dict = json.load(f)
        
        return cls(**settings_dict)


# 전역 설정 인스턴스
_global_settings = None

def get_settings() -> TMSSettings:
    """전역 설정 인스턴스 반환"""
    global _global_settings
    if _global_settings is None:
        _global_settings = TMSSettings()
        _global_settings.setup_logging()
    return _global_settings

def load_settings(config_file: Optional[str] = None) -> TMSSettings:
    """설정 파일에서 로드하거나 기본 설정 사용"""
    global _global_settings
    
    if config_file and Path(config_file).exists():
        _global_settings = TMSSettings.load_from_file(config_file)
    else:
        _global_settings = TMSSettings()
    
    _global_settings.setup_logging()
    return _global_settings

def reload_settings():
    """설정 재로드"""
    global _global_settings
    _global_settings = None
    return get_settings()