"""
외부 조건 분석 서비스
"""
from typing import Dict, List
from datetime import datetime
import logging

from ..models import Region


class ConditionAnalyzer:
    """날씨, 교통 등 외부 조건 분석"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def analyze_weather_conditions(self, regions: List[Region]) -> Dict[str, Dict]:
        """권역별 날씨 조건 분석"""
        weather_data = {}
        
        # 실제 날씨 API 사용
        from ..external import WeatherClient, get_cache_manager
        
        weather_client = WeatherClient(
            api_key=self.config.get('weather_api_key')
        )
        cache_manager = get_cache_manager()
        
        for region in regions:
            try:
                # 캐시된 날씨 데이터 확인
                cache_key = cache_manager.get_cache_key(
                    'weather',
                    region_id=region.id,
                    coordinates=region.center_coordinates
                )
                
                cached_weather = cache_manager.get('weather', cache_key)
                
                if cached_weather:
                    weather_info = cached_weather
                    self.logger.debug(f"캐시된 날씨 데이터 사용: {region.id}")
                else:
                    # API 호출하여 실시간 날씨 데이터 수집
                    weather_api_data = weather_client.get_weather_data(region.center_coordinates)
                    
                    if weather_api_data:
                        weather_info = self._convert_weather_api_to_dict(weather_api_data)
                        # 캐시에 저장 (30분)
                        cache_manager.set('weather', cache_key, weather_info)
                        self.logger.debug(f"실시간 날씨 데이터 수집: {region.id}")
                    else:
                        # API 실패 시 샘플 데이터 사용
                        weather_info = self._get_sample_weather_data(region)
                        self.logger.warning(f"날씨 API 실패, 샘플 데이터 사용: {region.id}")
                
                weather_data[region.id] = weather_info
                
                # 권역에 실시간 날씨 정보 업데이트
                region.update_real_time_conditions(
                    weather_severity=weather_info['severity_score']
                )
                
            except Exception as e:
                self.logger.error(f"날씨 데이터 수집 오류 ({region.id}): {str(e)}")
                # 오류 시 기본값 사용
                weather_info = self._get_sample_weather_data(region)
                weather_data[region.id] = weather_info
                region.update_real_time_conditions(weather_severity=1.0)
        
        self.logger.info(f"날씨 조건 분석 완료: {len(regions)}개 권역")
        return weather_data
    
    def analyze_traffic_conditions(self, regions: List[Region]) -> Dict[str, Dict]:
        """권역별 교통 조건 분석"""
        traffic_data = {}
        
        # 실제 교통 API 사용
        from ..external import TrafficClient, get_cache_manager
        
        traffic_client = TrafficClient(
            api_key=self.config.get('traffic_api_key')
        )
        cache_manager = get_cache_manager()
        
        for region in regions:
            try:
                # 캐시된 교통 데이터 확인
                cache_key = cache_manager.get_cache_key(
                    'traffic', 
                    region_id=region.id,
                    coordinates=region.center_coordinates
                )
                
                cached_traffic = cache_manager.get('traffic', cache_key)
                
                if cached_traffic:
                    traffic_info = cached_traffic
                    self.logger.debug(f"캐시된 교통 데이터 사용: {region.id}")
                else:
                    # API 호출하여 실시간 교통 데이터 수집
                    traffic_api_data = traffic_client.get_traffic_data(region)
                    
                    if traffic_api_data:
                        traffic_info = self._convert_traffic_api_to_dict(traffic_api_data)
                        # 캐시에 저장 (15분)
                        cache_manager.set('traffic', cache_key, traffic_info)
                        self.logger.debug(f"실시간 교통 데이터 수집: {region.id}")
                    else:
                        # API 실패 시 샘플 데이터 사용
                        traffic_info = self._get_sample_traffic_data(region)
                        self.logger.warning(f"교통 API 실패, 샘플 데이터 사용: {region.id}")
                
                traffic_data[region.id] = traffic_info
                
                # 권역에 실시간 교통 정보 업데이트
                region.update_real_time_conditions(
                    traffic_congestion=traffic_info['congestion_level']
                )
                
            except Exception as e:
                self.logger.error(f"교통 데이터 수집 오류 ({region.id}): {str(e)}")
                # 오류 시 기본값 사용
                traffic_info = self._get_sample_traffic_data(region)
                traffic_data[region.id] = traffic_info
                region.update_real_time_conditions(traffic_congestion=0.4)
        
        self.logger.info(f"교통 조건 분석 완료: {len(regions)}개 권역")
        return traffic_data
    
    def calculate_adjustment_factors(self, regions: List[Region]) -> Dict[str, float]:
        """권역별 조정 계수 계산"""
        adjustment_factors = {}
        
        for region in regions:
            total_factor = region.get_total_adjustment_factor()
            adjustment_factors[region.id] = total_factor
            
            self.logger.debug(f"권역 {region.name} 조정 계수: {total_factor:.2f}")
        
        return adjustment_factors
    
    def check_delivery_feasibility(self, regions: List[Region]) -> Dict[str, bool]:
        """권역별 배송 실행 가능성 확인"""
        feasibility = {}
        
        for region in regions:
            is_feasible = region.is_delivery_feasible()
            feasibility[region.id] = is_feasible
            
            if not is_feasible:
                self.logger.warning(f"권역 {region.name}은 배송 불가 상태")
        
        return feasibility
    
    def get_emergency_conditions(self, regions: List[Region]) -> List[str]:
        """비상 상황 조건 확인"""
        emergency_regions = []
        
        for region in regions:
            # 극한 날씨 조건
            if region.weather_severity and region.weather_severity >= 4.0:
                emergency_regions.append(f"{region.name}: 극한 날씨 (심각도 {region.weather_severity})")
            
            # 심각한 교통 정체
            if region.traffic_congestion and region.traffic_congestion >= 0.9:
                emergency_regions.append(f"{region.name}: 심각한 교통 정체 ({region.traffic_congestion*100:.0f}%)")
        
        if emergency_regions:
            self.logger.warning(f"비상 상황 감지: {len(emergency_regions)}개 권역")
        
        return emergency_regions
    
    def _convert_weather_api_to_dict(self, weather_data) -> Dict:
        """WeatherData 객체를 딕셔너리로 변환"""
        return {
            'condition': weather_data.weather_main.lower(),
            'temperature': weather_data.temperature,
            'humidity': weather_data.humidity,
            'wind_speed': weather_data.wind_speed,
            'precipitation': weather_data.precipitation,
            'severity_score': weather_data.severity_score,
            'description': weather_data.weather_description,
            'visibility': weather_data.visibility,
            'timestamp': weather_data.timestamp.isoformat()
        }
    
    def _convert_traffic_api_to_dict(self, traffic_data) -> Dict:
        """TrafficData 객체를 딕셔너리로 변환"""
        incidents_info = []
        for incident in traffic_data.incidents[:3]:  # 최대 3개만
            incidents_info.append({
                'type': incident.incident_type,
                'description': incident.description,
                'severity': incident.severity,
                'road': incident.affected_road
            })
        
        return {
            'congestion_level': traffic_data.congestion_level,
            'average_speed': traffic_data.average_speed,
            'jam_factor': traffic_data.jam_factor,
            'confidence': traffic_data.confidence,
            'incidents': incidents_info,
            'timestamp': traffic_data.timestamp.isoformat()
        }
    
    def _get_sample_weather_data(self, region: Region) -> Dict:
        """샘플 날씨 데이터 생성"""
        # 시간대별 다른 날씨 조건 시뮬레이션
        current_hour = datetime.now().hour
        
        if 6 <= current_hour <= 18:  # 낮 시간
            return {
                'condition': 'clear',
                'temperature': 22,
                'humidity': 60,
                'wind_speed': 5,
                'severity_score': 1.5,
                'description': '맑음'
            }
        else:  # 밤 시간
            return {
                'condition': 'partly_cloudy',
                'temperature': 18,
                'humidity': 70,
                'wind_speed': 8,
                'severity_score': 2.0,
                'description': '구름 조금'
            }
    
    def _get_sample_traffic_data(self, region: Region) -> Dict:
        """샘플 교통 데이터 생성"""
        # 시간대별 교통 패턴 시뮬레이션
        current_hour = datetime.now().hour
        
        if 7 <= current_hour <= 9 or 18 <= current_hour <= 20:  # 출퇴근 시간
            return {
                'congestion_level': 0.7,
                'average_speed': 25,
                'incidents_count': 2,
                'estimated_delay': 15,
                'description': '정체'
            }
        elif 10 <= current_hour <= 17:  # 일반 시간
            return {
                'congestion_level': 0.4,
                'average_speed': 40,
                'incidents_count': 0,
                'estimated_delay': 5,
                'description': '보통'
            }
        else:  # 심야 시간
            return {
                'congestion_level': 0.1,
                'average_speed': 55,
                'incidents_count': 0,
                'estimated_delay': 0,
                'description': '원활'
            }