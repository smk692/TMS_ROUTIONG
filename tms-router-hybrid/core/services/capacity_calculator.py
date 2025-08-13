"""
용량 계산 서비스
"""
from typing import List, Dict, Tuple
import logging

from ..models import Vehicle, Region


class CapacityCalculator:
    """차량 용량 동적 계산"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_vehicle_capacities(self, vehicles: List[Vehicle], 
                                   regions: List[Region],
                                   weather_conditions: Dict[str, Dict] = None,
                                   traffic_conditions: Dict[str, Dict] = None) -> Dict[str, int]:
        """차량별 조정된 용량 계산"""
        vehicle_capacities = {}
        
        for vehicle in vehicles:
            if not vehicle.is_auto_dispatch_eligible():
                vehicle_capacities[vehicle.id] = 0
                continue
            
            # 해당 차량의 권역 정보 찾기
            vehicle_region = self._find_vehicle_region(vehicle, regions)
            if not vehicle_region:
                self.logger.warning(f"차량 {vehicle.id}의 권역 {vehicle.region_id}를 찾을 수 없음")
                vehicle_capacities[vehicle.id] = vehicle.safe_capacity
                continue
            
            # 조정 계수들 계산
            weather_factor = self._calculate_weather_factor(vehicle_region, weather_conditions)
            traffic_factor = self._calculate_traffic_factor(vehicle_region, traffic_conditions)
            
            # 최종 용량 계산
            adjusted_capacity = vehicle.calculate_adjusted_capacity(
                weather_factor=weather_factor,
                traffic_factor=traffic_factor
            )
            
            vehicle_capacities[vehicle.id] = adjusted_capacity
            
            self.logger.debug(
                f"차량 {vehicle.driver_name}: "
                f"기본 {vehicle.safe_capacity} → 조정 {adjusted_capacity}개 "
                f"(경험도: {vehicle.get_experience_multiplier():.2f}, "
                f"날씨: {weather_factor:.2f}, 교통: {traffic_factor:.2f})"
            )
        
        total_capacity = sum(vehicle_capacities.values())
        self.logger.info(f"총 차량 용량: {total_capacity}개 (차량 {len(vehicles)}대)")
        
        return vehicle_capacities
    
    def calculate_region_load_distribution(self, vehicles: List[Vehicle], 
                                         regions: List[Region]) -> Dict[str, Dict]:
        """권역별 부하 분산 계산"""
        region_distribution = {}
        
        for region in regions:
            # 해당 권역의 차량들
            region_vehicles = [v for v in vehicles if v.region_id == region.id and v.is_auto_dispatch_eligible()]
            
            if not region_vehicles:
                region_distribution[region.id] = {
                    'vehicle_count': 0,
                    'total_capacity': 0,
                    'average_capacity': 0,
                    'difficulty_multiplier': region.get_difficulty_multiplier()
                }
                continue
            
            # 권역 내 차량 통계
            total_capacity = sum(v.safe_capacity for v in region_vehicles)
            avg_capacity = total_capacity / len(region_vehicles) if region_vehicles else 0
            
            region_distribution[region.id] = {
                'vehicle_count': len(region_vehicles),
                'total_capacity': total_capacity,
                'average_capacity': avg_capacity,
                'difficulty_multiplier': region.get_difficulty_multiplier(),
                'vehicles': [v.id for v in region_vehicles]
            }
            
            self.logger.debug(f"권역 {region.name}: {len(region_vehicles)}대, 총 용량 {total_capacity}개")
        
        return region_distribution
    
    def get_capacity_summary(self, vehicle_capacities: Dict[str, int], 
                           vehicles: List[Vehicle]) -> Dict:
        """용량 계산 요약 정보"""
        active_vehicles = [v for v in vehicles if v.is_auto_dispatch_eligible()]
        excluded_vehicles = [v for v in vehicles if not v.is_auto_dispatch_eligible()]
        
        total_original_capacity = sum(v.safe_capacity for v in active_vehicles)
        total_adjusted_capacity = sum(vehicle_capacities.values())
        
        capacity_utilization = 0.0
        if total_original_capacity > 0:
            capacity_utilization = total_adjusted_capacity / total_original_capacity
        
        summary = {
            'total_vehicles': len(vehicles),
            'active_vehicles': len(active_vehicles),
            'excluded_vehicles': len(excluded_vehicles),
            'original_total_capacity': total_original_capacity,
            'adjusted_total_capacity': total_adjusted_capacity,
            'capacity_adjustment_ratio': capacity_utilization,
            'average_adjusted_capacity': total_adjusted_capacity / len(active_vehicles) if active_vehicles else 0
        }
        
        self.logger.info(
            f"용량 요약: {len(active_vehicles)}/{len(vehicles)}대 활성, "
            f"총 용량 {total_original_capacity} → {total_adjusted_capacity}개 "
            f"(조정비율 {capacity_utilization:.1%})"
        )
        
        return summary
    
    def _find_vehicle_region(self, vehicle: Vehicle, regions: List[Region]) -> Region:
        """차량의 권역 정보 찾기"""
        for region in regions:
            if region.id == vehicle.region_id:
                return region
        return None
    
    def _calculate_weather_factor(self, region: Region, 
                                weather_conditions: Dict[str, Dict] = None) -> float:
        """날씨 조정 계수 계산"""
        if not weather_conditions or region.id not in weather_conditions:
            return 1.0  # 기본값
        
        weather_data = weather_conditions[region.id]
        severity_score = weather_data.get('severity_score', 1.0)
        
        # 날씨 심각도에 따른 계수
        if severity_score >= 4.0:  # 폭풍
            return 0.3
        elif severity_score >= 3.0:  # 폭우/폭설
            return 0.6
        elif severity_score >= 2.0:  # 비/눈
            return 0.8
        else:  # 맑음
            return 1.1
    
    def _calculate_traffic_factor(self, region: Region,
                                traffic_conditions: Dict[str, Dict] = None) -> float:
        """교통 조정 계수 계산"""
        if not traffic_conditions or region.id not in traffic_conditions:
            return 1.0  # 기본값
        
        traffic_data = traffic_conditions[region.id]
        congestion_level = traffic_data.get('congestion_level', 0.5)
        
        # 교통 정체도에 따른 계수
        if congestion_level >= 0.8:  # 심각한 정체
            return 0.6
        elif congestion_level >= 0.6:  # 정체
            return 0.8
        elif congestion_level <= 0.2:  # 원활
            return 1.1
        else:  # 보통
            return 1.0