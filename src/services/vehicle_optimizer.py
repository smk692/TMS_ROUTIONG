from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime
from ..models.delivery_point import DeliveryPoint
from ..models.vehicle import Vehicle
from ..utils.geo_utils import calculate_distance, get_center_point

class VehicleOptimizer:
    """차량 할당 및 최적화를 담당하는 클래스"""
    
    def __init__(self):
        self.vehicle_specs = self._load_vehicle_specifications()
    
    def assign_vehicles(
        self,
        areas: List[List[DeliveryPoint]],
        vehicles: List[Vehicle],
        assignment_strategy: str = 'balanced'
    ) -> List[Tuple[List[DeliveryPoint], Vehicle]]:
        """
        각 구역에 최적의 차량 할당

        Args:
            areas: 분할된 배송 구역 리스트
            vehicles: 사용 가능한 차량 리스트
            assignment_strategy: 할당 전략 ('balanced', 'capacity', 'distance')

        Returns:
            (배송구역, 차량) 튜플의 리스트
        """
        if assignment_strategy == 'balanced':
            return self._assign_balanced(areas, vehicles)
        elif assignment_strategy == 'capacity':
            return self._assign_by_capacity(areas, vehicles)
        elif assignment_strategy == 'distance':
            return self._assign_by_distance(areas, vehicles)
        else:
            raise ValueError(f"Unknown assignment strategy: {assignment_strategy}")

    def _load_vehicle_specifications(self) -> Dict:
        """차량 유형별 세부 사양 로드"""
        # 실제로는 DB나 설정 파일에서 로드
        return {
            'TRUCK_1TON': {
                'max_volume': 5.0,  # m³
                'max_weight': 1000.0,  # kg
                'fuel_efficiency': 10.0,  # km/L
                'average_speed': 60.0,  # km/h
                'cost_per_km': 500.0  # 원/km
            },
            'VAN_1TON': {
                'max_volume': 3.0,
                'max_weight': 1000.0,
                'fuel_efficiency': 12.0,
                'average_speed': 65.0,
                'cost_per_km': 450.0
            },
            'TRUCK_2.5TON': {
                'max_volume': 16.0,
                'max_weight': 2500.0,
                'fuel_efficiency': 8.0,
                'average_speed': 55.0,
                'cost_per_km': 700.0
            }
        }

    def _assign_balanced(
        self,
        areas: List[List[DeliveryPoint]],
        vehicles: List[Vehicle]
    ) -> List[Tuple[List[DeliveryPoint], Vehicle]]:
        """작업량과 차량 특성을 균형있게 고려한 할당"""
        assignments = []
        
        # 각 구역의 작업량 계산
        area_metrics = []
        for area in areas:
            total_volume = sum(point.volume for point in area)
            total_weight = sum(point.weight for point in area)
            total_priority = sum(point.get_priority_weight() for point in area)
            center = get_center_point(area)
            
            area_metrics.append({
                'area': area,
                'volume': total_volume,
                'weight': total_weight,
                'priority': total_priority,
                'center': center
            })

        # 각 차량의 특성 평가
        vehicle_metrics = []
        for vehicle in vehicles:
            specs = self.vehicle_specs[vehicle.type]
            vehicle_metrics.append({
                'vehicle': vehicle,
                'capacity_score': (
                    vehicle.capacity.volume / specs['max_volume'] +
                    vehicle.capacity.weight / specs['max_weight']
                ) / 2,
                'efficiency_score': specs['fuel_efficiency'] / 10.0,  # 정규화
                'speed_score': specs['average_speed'] / 60.0  # 정규화
            })

        # 구역과 차량 매칭
        used_areas = set()
        used_vehicles = set()
        
        # 우선 순위가 높은 구역부터 처리
        area_metrics.sort(key=lambda x: x['priority'], reverse=True)
        
        for area_metric in area_metrics:
            if len(used_vehicles) == len(vehicles):
                break
                
            # 해당 구역에 가장 적합한 차량 찾기
            best_match = None
            best_score = float('-inf')
            
            for vehicle_metric in vehicle_metrics:
                if vehicle_metric['vehicle'] in used_vehicles:
                    continue
                
                # 차량 적합도 점수 계산
                capacity_fit = min(
                    area_metric['volume'] / vehicle_metric['vehicle'].capacity.volume,
                    area_metric['weight'] / vehicle_metric['vehicle'].capacity.weight
                )
                
                if capacity_fit > 1.0:  # 용량 초과
                    continue
                
                # 종합 점수 계산
                score = (
                    capacity_fit * 0.4 +
                    vehicle_metric['efficiency_score'] * 0.3 +
                    vehicle_metric['speed_score'] * 0.3
                )
                
                if score > best_score:
                    best_score = score
                    best_match = vehicle_metric['vehicle']
            
            if best_match:
                assignments.append((area_metric['area'], best_match))
                used_vehicles.add(best_match)
                used_areas.add(tuple(area_metric['area']))

        return assignments

    def _assign_by_capacity(
        self,
        areas: List[List[DeliveryPoint]],
        vehicles: List[Vehicle]
    ) -> List[Tuple[List[DeliveryPoint], Vehicle]]:
        """적재 용량을 우선적으로 고려한 할당"""
        assignments = []
        
        # 구역별 필요 용량 계산
        area_requirements = []
        for area in areas:
            total_volume = sum(point.volume for point in area)
            total_weight = sum(point.weight for point in area)
            
            area_requirements.append({
                'area': area,
                'volume': total_volume,
                'weight': total_weight
            })
        
        # 용량이 큰 순서대로 정렬
        area_requirements.sort(
            key=lambda x: (x['volume'], x['weight']),
            reverse=True
        )
        
        # 차량 용량별 정렬
        available_vehicles = sorted(
            vehicles,
            key=lambda v: (v.capacity.volume, v.capacity.weight),
            reverse=True
        )
        
        # 매칭
        used_vehicles = set()
        for req in area_requirements:
            for vehicle in available_vehicles:
                if vehicle in used_vehicles:
                    continue
                    
                if (vehicle.capacity.volume >= req['volume'] and
                    vehicle.capacity.weight >= req['weight']):
                    assignments.append((req['area'], vehicle))
                    used_vehicles.add(vehicle)
                    break
        
        return assignments

    def _assign_by_distance(
        self,
        areas: List[List[DeliveryPoint]],
        vehicles: List[Vehicle]
    ) -> List[Tuple[List[DeliveryPoint], Vehicle]]:
        """이동 거리를 우선적으로 고려한 할당"""
        assignments = []
        
        # 각 구역의 중심점 계산
        area_centers = []
        for area in areas:
            center = get_center_point(area)
            area_centers.append({
                'area': area,
                'center': center
            })
        
        # 각 차량의 현재 위치 기준 거리 계산
        for vehicle in vehicles:
            if not vehicle.current_location:
                continue
                
            for area_info in area_centers:
                distance = calculate_distance(
                    vehicle.current_location,
                    area_info['center']
                )
                area_info[f'distance_to_{vehicle.id}'] = distance
        
        # 거리 기반 매칭
        used_vehicles = set()
        used_areas = set()
        
        while area_centers and len(used_vehicles) < len(vehicles):
            min_distance = float('inf')
            best_match = None
            
            for area_info in area_centers:
                if tuple(area_info['area']) in used_areas:
                    continue
                    
                for vehicle in vehicles:
                    if vehicle in used_vehicles:
                        continue
                        
                    distance = area_info.get(f'distance_to_{vehicle.id}')
                    if distance and distance < min_distance:
                        min_distance = distance
                        best_match = (area_info['area'], vehicle)
            
            if best_match:
                assignments.append(best_match)
                used_vehicles.add(best_match[1])
                used_areas.add(tuple(best_match[0]))
            else:
                break
        
        return assignments

    def calculate_vehicle_costs(
        self,
        assignments: List[Tuple[List[DeliveryPoint], Vehicle]]
    ) -> Dict[str, float]:
        """차량별 예상 운영 비용 계산"""
        costs = {}
        
        for area, vehicle in assignments:
            # 이동 거리 계산
            total_distance = 0
            points = [vehicle.current_location] if vehicle.current_location else []
            points.extend((p.latitude, p.longitude) for p in area)
            
            for i in range(len(points) - 1):
                distance = calculate_distance(points[i], points[i + 1])
                total_distance += distance
            
            # 비용 계산
            specs = self.vehicle_specs[vehicle.type]
            fuel_cost = (total_distance / specs['fuel_efficiency']) * 1500  # 리터당 1500원 가정
            maintenance_cost = total_distance * specs['cost_per_km']
            
            costs[vehicle.id] = {
                'fuel_cost': fuel_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': fuel_cost + maintenance_cost,
                'distance': total_distance
            }
        
        return costs