#!/usr/bin/env python3
"""
고급 TMS 시스템 기반 배송 경로 추출 서비스

src 디렉토리의 엔터프라이즈급 TMS 시스템을 활용한 최적화:
- 모듈화된 아키텍처
- 고급 클러스터링 알고리즘
- 성능 모니터링
- 제약조건 처리
- 실시간 최적화
"""

import os
import sys
import json
import math
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# src 모듈들 import
from src.db.maria_client import fetch_orders
from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle, VehicleCapacity
from src.model.route import Route, RoutePoint
from src.model.time_window import TimeWindow
from src.algorithm.cluster import cluster_points
from src.algorithm.anti_overlap_clustering import create_anti_overlap_clusters
from src.algorithm.tsp import solve_tsp
from src.core.logger import setup_logger
from src.core.distance_matrix import DistanceMatrix
from src.services.constraint_handler import ConstraintHandler
from src.services.depot_selector import DepotSelectorService
from src.services.polyline_service import PolylineService
from src.services.vehicle_id_service import get_next_vehicle_id, reset_vehicle_counter
from src.utils.distance_calculator import assign_points_to_nearest_centers, calculate_distance
from src.visualization.polyline_visualizer import create_polyline_visualization

# 로거 설정
logger = setup_logger('route_extractor')

class RouteExtractorService:
    """고급 TMS 시스템 기반 배송 경로 추출 서비스"""
    
    def __init__(self, config=None):
        self.distance_matrix = DistanceMatrix()
        self.constraint_handler = ConstraintHandler()
        
        # config 설정
        self.config = config
        
        # 물류센터 선택 서비스 초기화
        self.depot_selector = DepotSelectorService(config) if config else None
        
        # 기본 설정 (config에서 가져오기)
        if config:
            # 다중 TC 지원 확인
            depots = config.get('logistics.depots', [])
            if depots:
                # 기본 물류센터 설정 (나중에 자동 선택으로 업데이트됨)
                default_depot = self.depot_selector.get_default_depot()
                self.depot_info = {
                    "id": default_depot['id'],
                    "latitude": default_depot['latitude'],
                    "longitude": default_depot['longitude'],
                    "address1": default_depot['name'],
                    "address2": default_depot['address']
                }
            else:
                # 기존 단일 depot 설정 (하위 호환성)
                self.depot_info = {
                    "id": "depot_legacy",
                    "latitude": config.get('logistics.depot.latitude', 37.263573),
                    "longitude": config.get('logistics.depot.longitude', 127.028601),
                    "address1": config.get('logistics.depot.name', '수원센터'),
                    "address2": config.get('logistics.depot.address', '')
                }
        else:
            # 기본값 사용
            self.depot_info = {
                "id": "depot_suwon",
                "latitude": 37.263573,
                "longitude": 127.028601,
                "address1": "수원센터",
                "address2": ""
            }
        
        # 차량 설정
        self.vehicles = self._create_default_vehicles()
        
        logger.info("고급 TMS 경로 추출 서비스가 초기화되었습니다.")
        if self.depot_selector:
            logger.info(f"다중 TC 지원 활성화 - 현재 기본 센터: {self.depot_info['address1']}")
    
    def _create_default_vehicles(self) -> List[Vehicle]:
        """기본 차량 설정 생성 (config 기반) - 전역 고유 ID 사용"""
        vehicles = []
        
        # config에서 차량 설정 가져오기
        if self.config:
            vehicle_count = self.config.get('vehicles.count', 15)
            volume_capacity = self.config.get('vehicles.capacity.volume', 5.0)
            weight_capacity = self.config.get('vehicles.capacity.weight', 1000.0)
            cost_per_km = self.config.get('vehicles.cost_per_km', 500.0)
            start_hour = self.config.get('vehicles.operating_hours.start_hour', 6)
            start_minute = self.config.get('vehicles.operating_hours.start_minute', 0)
            end_hour = self.config.get('vehicles.operating_hours.end_hour', 14)
            end_minute = self.config.get('vehicles.operating_hours.end_minute', 0)
        else:
            # 기본값 사용
            vehicle_count = 15
            volume_capacity = 5.0
            weight_capacity = 1000.0
            cost_per_km = 500.0
            start_hour = 6
            start_minute = 0
            end_hour = 14
            end_minute = 0
        
        # 차량 생성 (전역 고유 ID 사용)
        for i in range(vehicle_count):
            global_vehicle_id = get_next_vehicle_id()
            vehicle = Vehicle(
                id=f"TRUCK_1TON_{global_vehicle_id:03d}",  # 전역 고유 ID 사용
                name=f"{global_vehicle_id}호차",  # 전역 번호 사용
                type="TRUCK_1TON",
                capacity=VehicleCapacity(volume=volume_capacity, weight=weight_capacity),
                features=["STANDARD"],
                cost_per_km=cost_per_km,
                start_time=datetime.now().replace(hour=start_hour, minute=start_minute, second=0),
                end_time=datetime.now().replace(hour=end_hour, minute=end_minute, second=0)
            )
            vehicles.append(vehicle)
        
        logger.info(f"기본 차량 {len(vehicles)}대가 설정되었습니다. (ID: {get_next_vehicle_id()-len(vehicles)+1}~{get_next_vehicle_id()})")
        return vehicles
    
    def load_delivery_data(self) -> Optional[List[DeliveryPoint]]:
        """데이터베이스에서 배송 데이터를 로드하고 DeliveryPoint 객체로 변환"""
        try:
            logger.info("데이터베이스에서 배송 데이터를 로드합니다...")
            
            # 데이터베이스에서 주문 데이터 가져오기
            df = fetch_orders()
            
            print(df)
            if df.empty:
                logger.error("데이터베이스에서 배송 데이터를 찾을 수 없습니다.")
                return None
            
            logger.info(f"데이터베이스에서 {len(df)}개의 배송 데이터를 로드했습니다.")
            
            # 데이터 전처리
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
            
            logger.info(f"유효한 좌표를 가진 배송지: {len(df)}개")
            
            # config에서 배송 설정 가져오기
            if self.config:
                service_time = self.config.get('logistics.delivery.service_time', 5)
                default_volume = self.config.get('logistics.delivery.default_volume', 0.1)
                default_weight = self.config.get('logistics.delivery.default_weight', 5.0)
                default_priority = self.config.get('logistics.delivery.default_priority', 3)
                start_hour = self.config.get('vehicles.operating_hours.start_hour', 6)
                start_minute = self.config.get('vehicles.operating_hours.start_minute', 0)
                end_hour = self.config.get('vehicles.operating_hours.end_hour', 14)
                end_minute = self.config.get('vehicles.operating_hours.end_minute', 0)
            else:
                # 기본값 사용
                service_time = 5
                default_volume = 0.1
                default_weight = 5.0
                default_priority = 3
                start_hour = 6
                start_minute = 0
                end_hour = 14
                end_minute = 0
            
            # DeliveryPoint 객체 생성
            delivery_points = []
            for idx, (_, row) in enumerate(df.iterrows()):
                # 시간 윈도우 설정 (config 기반)
                start_time = datetime.now().replace(hour=start_hour, minute=start_minute, second=0)
                end_time = datetime.now().replace(hour=end_hour, minute=end_minute, second=0)
                time_window = (start_time, end_time)  # 튜플로 설정
                
                # ID가 문자열인 경우 해시값으로 변환
                invoice_id = row['id']
                if isinstance(invoice_id, str):
                    # 문자열 ID를 해시값으로 변환하여 고유한 숫자 ID 생성
                    point_id = abs(hash(invoice_id)) % (10**8)  # 8자리 숫자로 제한
                else:
                    point_id = int(invoice_id)
                
                point = DeliveryPoint(
                    id=point_id,
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    address1=row.get('address1', ''),
                    address2=row.get('address2', ''),
                    time_window=time_window,
                    service_time=service_time,
                    volume=default_volume,
                    weight=default_weight,
                    special_requirements=[],
                    priority=default_priority
                )
                delivery_points.append(point)
            
            logger.info(f"DeliveryPoint 객체 {len(delivery_points)}개가 생성되었습니다.")
            
            # 다중 TC 지원 시 최적 물류센터 자동 선택
            if self.depot_selector and self.depot_selector.auto_select:
                self._select_optimal_depot(delivery_points)
            
            return delivery_points
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return None
    
    def _select_optimal_depot(self, delivery_points: List[DeliveryPoint]):
        """배송지 분석을 통한 최적 물류센터 자동 선택"""
        try:
            logger.info("배송지 분석을 통한 최적 물류센터를 선택합니다...")
            
            # 배송지 데이터를 DataFrame으로 변환
            import pandas as pd
            delivery_data = []
            for point in delivery_points:
                delivery_data.append({
                    'id': point.id,
                    'lat': point.latitude,
                    'lng': point.longitude,
                    'address': f"{point.address1} {point.address2}".strip()
                })
            
            df = pd.DataFrame(delivery_data)
            
            # 최적 물류센터 분석
            analysis = self.depot_selector.analyze_delivery_coverage(df)
            selected_depot = analysis['selected_depot']
            
            # 물류센터 정보 업데이트
            self.depot_info = {
                "id": selected_depot['id'],
                "latitude": selected_depot['latitude'],
                "longitude": selected_depot['longitude'],
                "address1": selected_depot['name'],
                "address2": selected_depot['address']
            }
            
            logger.info(f"✅ 최적 물류센터 선택: {selected_depot['name']}")
            logger.info(f"📊 커버리지 분석: {analysis['coverage_summary']}")
            logger.info(f"📍 평균 거리: {analysis['average_distance_km']:.2f}km")
            
        except Exception as e:
            logger.warning(f"최적 물류센터 선택 중 오류 발생, 기본 센터 사용: {str(e)}")
    
    def get_current_depot_info(self) -> Dict:
        """현재 선택된 물류센터 정보 반환"""
        return self.depot_info.copy()
    
    def print_depot_status(self):
        """물류센터 현황 출력"""
        if self.depot_selector:
            self.depot_selector.print_depot_status()
            print(f"\n🎯 현재 선택된 센터: {self.depot_info['address1']}")
        else:
            print(f"\n🏢 현재 물류센터: {self.depot_info['address1']}")
            print(f"📍 위치: {self.depot_info['address2']}")
            print(f"🗺️  좌표: {self.depot_info['latitude']:.6f}, {self.depot_info['longitude']:.6f}")
    
    def filter_points_by_distance(self, points: List[DeliveryPoint], max_distance: float = None) -> List[DeliveryPoint]:
        """물류센터로부터 일정 거리 내의 배송지점만 필터링"""
        from src.model.distance import distance
        
        # config에서 최대 거리 가져오기
        if max_distance is None:
            if self.config:
                max_distance = self.config.get('logistics.delivery.max_distance', 15.0)
            else:
                max_distance = 15.0
        
        filtered_points = []
        depot_coord = (self.depot_info['latitude'], self.depot_info['longitude'])
        
        for point in points:
            point_coord = (point.latitude, point.longitude)
            dist = distance(depot_coord[0], depot_coord[1], point_coord[0], point_coord[1])
            
            if dist <= max_distance:
                filtered_points.append(point)
        
        logger.info(f"물류센터 {max_distance}km 반경 내 배송지: {len(filtered_points)}개")
        return filtered_points
    
    def _create_additional_vehicles(self, count: int) -> List[Vehicle]:
        """동적으로 추가 차량 생성"""
        from src.model.vehicle import VehicleCapacity
        
        additional_vehicles = []
        
        # 기존 차량 설정 가져오기
        if self.config:
            capacity_volume = self.config.get('vehicles.capacity.volume', 5.0)
            capacity_weight = self.config.get('vehicles.capacity.weight', 1000.0)
            start_hour = self.config.get('vehicles.operating_hours.start_hour', 6)
            start_minute = self.config.get('vehicles.operating_hours.start_minute', 0)
            end_hour = self.config.get('vehicles.operating_hours.end_hour', 14)
            end_minute = self.config.get('vehicles.operating_hours.end_minute', 0)
            cost_per_km = self.config.get('vehicles.cost_per_km', 500.0)
        else:
            capacity_volume = 5.0
            capacity_weight = 1000.0
            start_hour = 6
            start_minute = 0
            end_hour = 14
            end_minute = 0
            cost_per_km = 500.0
        
        # 시작 차량 ID
        start_id = len(self.vehicles) + 1
        
        for i in range(count):
            vehicle_id = start_id + i
            
            # 시간 윈도우 설정
            start_time = datetime.now().replace(hour=start_hour, minute=start_minute, second=0)
            end_time = datetime.now().replace(hour=end_hour, minute=end_minute, second=0)
            
            # Vehicle 객체 생성 (VehicleCapacity 사용)
            vehicle = Vehicle(
                id=f"TRUCK_1TON_{vehicle_id:03d}",
                name=f"동적차량{vehicle_id}",
                type="TRUCK_1TON",
                capacity=VehicleCapacity(volume=capacity_volume, weight=capacity_weight),
                features=["STANDARD"],
                cost_per_km=cost_per_km,
                start_time=start_time,
                end_time=end_time
            )
            
            additional_vehicles.append(vehicle)
        
        logger.info(f"🚚 추가 차량 생성 완료: {count}대 (ID: {start_id}~{start_id + count - 1})")
        return additional_vehicles
    
    def clustering_optimization(self, delivery_points: List[DeliveryPoint]) -> Optional[List[Route]]:
        """중복 방지 클러스터링 기반 최적화"""
        try:
            logger.info("🎯 중복 방지 클러스터링 최적화를 실행합니다...")
            
            # config에서 차량당 배송지 수 가져오기
            if self.config:
                points_per_vehicle = self.config.get('logistics.delivery.points_per_vehicle', 15)
                average_speed = self.config.get('vehicles.average_speed', 30.0)
            else:
                points_per_vehicle = 15
                average_speed = 30.0
            
            # 기존 차량 사용 (동적 추가 제거)
            required_vehicles = min(len(self.vehicles), math.ceil(len(delivery_points) / points_per_vehicle))
            selected_vehicles = self.vehicles[:required_vehicles]
            
            logger.info(f"✅ 차량 할당: {len(selected_vehicles)}대 (배송지 {len(delivery_points)}개)")
            
            # 🎯 중복 방지 클러스터링 실행
            logger.info("🚫 중복 방지 클러스터링 실행 중...")
            clusters = create_anti_overlap_clusters(
                points=delivery_points,
                vehicles=selected_vehicles
            )
            
            if not clusters:
                logger.warning("중복 방지 클러스터링 실패 - 기본 클러스터링으로 대체")
            clusters = cluster_points(
                points=delivery_points,
                vehicles=selected_vehicles,
                strategy='enhanced_kmeans'
            )
            
            if not clusters:
                logger.error("모든 클러스터링 방법이 실패했습니다.")
                return None
            
            logger.info(f"✅ 클러스터링 완료: {len(clusters)}개 클러스터 (중복 제거됨)")
            
            # 💡 배송지 누락 검증 로직 추가
            total_clustered_points = sum(len(cluster) for cluster in clusters)
            original_points_count = len(delivery_points)
            
            logger.info(f"📊 배송지 분배 현황:")
            logger.info(f"   - 원본 배송지: {original_points_count}개")
            logger.info(f"   - 클러스터링 후: {total_clustered_points}개")
            
            if total_clustered_points != original_points_count:
                missing_count = original_points_count - total_clustered_points
                logger.warning(f"⚠️ 배송지 누락 감지: {missing_count}개 누락!")
                
                # 누락된 배송지 찾기
                clustered_ids = set()
                for cluster in clusters:
                    for point in cluster:
                        clustered_ids.add(point.id)
                
                missing_points = [p for p in delivery_points if p.id not in clustered_ids]
                logger.warning(f"   누락된 배송지 ID: {[p.id for p in missing_points[:5]]}")
                
                # 누락된 배송지를 가장 가까운 클러스터에 추가
                for missing_point in missing_points:
                    best_cluster_idx = 0
                    min_distance = float('inf')
                    
                    for i, cluster in enumerate(clusters):
                        if not cluster:
                            continue
                        # 클러스터 중심과의 거리 계산
                        center_lat = sum(p.latitude for p in cluster) / len(cluster)
                        center_lng = sum(p.longitude for p in cluster) / len(cluster)
                        
                        dist = calculate_distance(missing_point.latitude, missing_point.longitude,
                                       center_lat, center_lng)
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_cluster_idx = i
                    
                    clusters[best_cluster_idx].append(missing_point)
                    logger.info(f"   ✅ 배송지 {missing_point.id} → 클러스터 {best_cluster_idx+1}에 추가")
                
                # 재검증
                final_total = sum(len(cluster) for cluster in clusters)
                logger.info(f"🔄 재검증: {final_total}개 배송지 (누락 해결: {final_total == original_points_count})")
            
            # 각 클러스터별 TSP 최적화
            routes = []
            total_distance = 0
            total_time = 0
            
            for i, (cluster, vehicle) in enumerate(zip(clusters, selected_vehicles)):
                if not cluster:
                    continue
                
                logger.info(f"차량 {i+1}: {len(cluster)}개 배송지 TSP 최적화 중...")
                
                # TSP 최적화
                optimized_sequence, cluster_distance = solve_tsp(cluster)
                
                if optimized_sequence:
                    # OSRM API를 사용한 실제 시간 계산
                    route_time, route_points = self._calculate_route_with_osrm(
                        optimized_sequence, vehicle, cluster_distance
                    )
                    
                    # Depot 객체 생성
                    depot = type('Depot', (), {
                        'id': self.depot_info['id'],
                        'name': self.depot_info.get('name', self.depot_info.get('address', 'Unknown')),
                        'latitude': self.depot_info['latitude'],
                        'longitude': self.depot_info['longitude']
                    })()
                    
                    route = Route(
                        id=f"R{vehicle.id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        vehicle=vehicle,
                        depot_id=self.depot_info['id'],  # depot_id 설정
                        points=route_points,
                        total_distance=cluster_distance,
                        total_time=route_time,
                        total_load={
                            'volume': sum(p.point.volume for p in route_points),
                            'weight': sum(p.point.weight for p in route_points)
                        },
                        start_time=vehicle.start_time,
                        end_time=route_points[-1].departure_time if route_points else vehicle.start_time,
                        status='PLANNED',
                        depot_name=self.depot_info.get('name', self.depot_info.get('address', 'Unknown'))
                    )
                    
                    # depot 속성 동적 추가
                    route.depot = depot
                    
                    routes.append(route)
                    total_distance += cluster_distance
                    total_time += route_time
            
            logger.info(f"🎯 중복 방지 클러스터링 최적화 완료: {len(routes)}개 경로 생성")
            logger.info(f"총 거리: {total_distance:.1f}km, 총 시간: {total_time:.0f}분")
            
            # 제약조건 검증 및 조정 (config 기반) - 성능 개선
            logger.info("제약조건 검증 및 최적화를 실행합니다...")
            
            # 초고속 모드에서는 제약조건 검증 완전히 건너뛰기
            ultra_fast_mode = self.config and self.config.get('algorithms.tsp.max_iterations', 100) <= 30
            if ultra_fast_mode:
                logger.info("⚡ 초고속 모드: 모든 제약조건 검증 건너뛰기")
                validated_routes = routes
            # 빠른 모드에서는 기본 검증만 수행
            elif self.config and self.config.get('algorithms.tsp.max_iterations', 150) <= 60:
                logger.info("🚀 빠른 모드: 기본 제약조건만 검증합니다...")
                # 기본적인 용량 검증만 수행
                for route in routes:
                    if route.is_capacity_exceeded():
                        logger.warning(f"차량 {route.vehicle.name} 용량 초과 감지")
                validated_routes = routes
            else:
                # 전체 제약조건 검증
                if self.config:
                    constraints = {
                        'max_working_hours': self.config.get('constraints.max_working_hours', 8),
                        'max_points_per_vehicle': self.config.get('constraints.max_points_per_vehicle', 50),
                        'min_points_per_vehicle': self.config.get('constraints.min_points_per_vehicle', 10),
                        'allow_overtime': self.config.get('constraints.allow_overtime', False),
                        'consider_traffic': self.config.get('constraints.consider_traffic', True),
                        'target_efficiency': self.config.get('constraints.target_efficiency', 0.1)
                    }
                else:
                    # 기본값 사용
                    constraints = {
                        'max_working_hours': 8,
                        'max_points_per_vehicle': 50,
                        'min_points_per_vehicle': 10,
                        'allow_overtime': False,
                        'consider_traffic': True,
                        'target_efficiency': 0.1
                    }
                
                validated_routes = self.constraint_handler.validate_and_adjust(routes, constraints)
            
            if validated_routes:
                logger.info(f"✅ 제약조건 검증 완료: {len(validated_routes)}개 경로 최종 확정")
                return validated_routes
            else:
                logger.warning("제약조건 검증에서 문제가 발생했지만 기본 경로를 반환합니다.")
                return routes
            
        except Exception as e:
            logger.error(f"중복 방지 클러스터링 최적화 중 오류 발생: {str(e)}")
            return None
    
    def _calculate_route_with_osrm(self, optimized_sequence: List[DeliveryPoint], vehicle: Vehicle, cluster_distance: float) -> Tuple[int, List[RoutePoint]]:
        """OSRM API를 사용한 실제 경로 시간 계산"""
        try:
            # 좌표 리스트 생성 (depot → 배송지들 → depot)
            coordinates = []
            
            # 시작 depot
            coordinates.append({
                'id': self.depot_info['id'],
                'lat': self.depot_info['latitude'],
                'lng': self.depot_info['longitude'],
                'type': 'depot'
            })
            
            # 배송지들
            for point in optimized_sequence:
                coordinates.append({
                    'id': point.id,
                    'lat': point.latitude,
                    'lng': point.longitude,
                    'type': 'delivery'
                })
            
            # 종료 depot
            coordinates.append({
                'id': f"{self.depot_info['id']}_end",
                'lat': self.depot_info['latitude'],
                'lng': self.depot_info['longitude'],
                'type': 'depot'
            })
            
            # 폴리라인 서비스로 OSRM API 호출
            polyline_service = PolylineService(self.config)
            osrm_result = polyline_service.get_vehicle_route_from_osrm(coordinates)
            
            if osrm_result.get('success') and 'segment_durations' in osrm_result:
                # OSRM에서 받은 실제 시간 데이터 사용
                segment_durations = osrm_result['segment_durations']  # 초 단위
                total_osrm_time = osrm_result.get('total_duration', 0)  # 초 단위
                
                logger.info(f"  ✅ OSRM 실제 시간: {total_osrm_time/60:.1f}분")
                
                # RoutePoint 객체들 생성 (실제 시간 기반)
                route_points = []
                current_time = vehicle.start_time
                cumulative_distance = 0.0
                
                for j, point in enumerate(optimized_sequence):
                    arrival_time = current_time
                    departure_time = arrival_time + timedelta(minutes=point.service_time)
                    
                    route_point = RoutePoint(
                        point=point,
                        arrival_time=arrival_time,
                        departure_time=departure_time,
                        cumulative_distance=cumulative_distance,
                        cumulative_load={
                            'volume': point.volume,
                            'weight': point.weight
                        }
                    )
                    route_points.append(route_point)
                    current_time = departure_time
                    
                    # 다음 지점으로의 실제 이동 시간 추가 (OSRM 데이터)
                    if j < len(segment_durations) - 1:  # 마지막 구간(depot 복귀) 제외
                        travel_time_seconds = segment_durations[j]
                        travel_time_minutes = travel_time_seconds / 60
                        current_time += timedelta(minutes=travel_time_minutes)
                        
                        # 거리 정보도 있으면 사용
                        if 'segment_distances' in osrm_result and j < len(osrm_result['segment_distances']):
                            cumulative_distance += osrm_result['segment_distances'][j] / 1000  # km 변환
                
                # 총 시간 계산 (서비스 시간 + 이동 시간)
                service_time_total = sum(point.service_time for point in optimized_sequence)
                travel_time_total = total_osrm_time / 60  # 분 단위 변환
                total_time = int(service_time_total + travel_time_total)
                
                logger.info(f"  📊 시간 분석: 서비스 {service_time_total}분 + 이동 {travel_time_total:.1f}분 = 총 {total_time}분")
                
                return total_time, route_points
            
            else:
                # OSRM 실패 시 기존 방식으로 fallback
                logger.warning("  ⚠️ OSRM 시간 계산 실패, 기존 방식 사용")
                return self._calculate_route_fallback(optimized_sequence, vehicle, cluster_distance)
                
        except Exception as e:
            logger.warning(f"  ⚠️ OSRM 시간 계산 오류: {str(e)}, 기존 방식 사용")
            return self._calculate_route_fallback(optimized_sequence, vehicle, cluster_distance)
    
    def _calculate_route_fallback(self, optimized_sequence: List[DeliveryPoint], vehicle: Vehicle, cluster_distance: float) -> Tuple[int, List[RoutePoint]]:
        """기존 방식의 시간 계산 (fallback)"""
        if self.config:
            average_speed = self.config.get('vehicles.average_speed', 30.0)
        else:
            average_speed = 30.0
        
        route_points = []
        current_time = vehicle.start_time
        cumulative_distance = 0.0
        
        for j, point in enumerate(optimized_sequence):
            arrival_time = current_time
            departure_time = arrival_time + timedelta(minutes=point.service_time)
            
            route_point = RoutePoint(
                point=point,
                arrival_time=arrival_time,
                departure_time=departure_time,
                cumulative_distance=cumulative_distance,
                cumulative_load={
                    'volume': point.volume,
                    'weight': point.weight
                }
            )
            route_points.append(route_point)
            current_time = departure_time
            
            # 다음 지점으로의 이동 시간 추가 (기존 방식)
            if j < len(optimized_sequence) - 1:
                next_point = optimized_sequence[j + 1]
                travel_distance = self.distance_matrix.get_distance(point, next_point)
                travel_time = (travel_distance / average_speed) * 60  # 분 단위
                current_time += timedelta(minutes=travel_time)
                cumulative_distance += travel_distance
        
        route_time = int((current_time - vehicle.start_time).total_seconds() / 60)
        return route_time, route_points
    
    def convert_routes_to_json(self, routes: List[Route]) -> Dict[str, Any]:
        """Route 객체들을 JSON 형태로 변환"""
        try:
            logger.info("경로 데이터를 JSON 형태로 변환합니다...")
            
            # 통계 계산
            total_distance = sum(route.total_distance for route in routes)
            total_time = sum(route.total_time for route in routes)
            total_points = sum(len(route.points) for route in routes)
            
            # JSON 데이터 구조 생성
            result_data = {
                "depot": {
                    "id": self.depot_info['id'],
                    "label": self.depot_info['address1'],
                    "lat": self.depot_info['latitude'],
                    "lng": self.depot_info['longitude']
                },
                "routes": [],
                "stats": {
                    'depot_id': self.depot_info['id'],
                    'total_points': total_points,
                    'total_vehicles': len(routes),
                    'total_distance': total_distance,
                    'total_time': total_time,
                    'avg_distance_per_vehicle': total_distance / len(routes) if routes else 0,
                    'avg_time_per_vehicle': total_time / len(routes) if routes else 0,
                    'time_efficiency': total_time / (len(routes) * 480) if routes else 0  # 8시간 기준
                },
                "generated_at": datetime.now().isoformat(),
                "route_type": "advanced_tms",
                "optimization_method": "enhanced_clustering_tsp"
            }
            
            # 각 경로를 JSON 형태로 변환
            for route in routes:
                route_coordinates = []
                
                # 물류센터 시작점 추가
                depot_coord = {
                    "id": self.depot_info['id'],
                    "label": self.depot_info['address1'],
                    "lat": self.depot_info['latitude'],
                    "lng": self.depot_info['longitude'],
                    "type": "depot",
                    "sequence": 0
                }
                route_coordinates.append(depot_coord)
                
                # 배송지점들 추가
                for idx, route_point in enumerate(route.points):
                    point = route_point.point
                    coord = {
                        "id": point.id,
                        "label": point.address1 or f"배송지 {point.id}",
                        "lat": point.latitude,
                        "lng": point.longitude,
                        "type": "delivery",
                        "sequence": idx + 1,
                        "arrival_time": route_point.arrival_time.isoformat(),
                        "departure_time": route_point.departure_time.isoformat(),
                        "service_time": point.service_time,
                        "priority": point.priority
                    }
                    route_coordinates.append(coord)
                
                # 물류센터 종료점 추가
                depot_end_coord = {
                    "id": f"{self.depot_info['id']}_end",
                    "label": f"{self.depot_info['address1']} (복귀)",
                    "lat": self.depot_info['latitude'],
                    "lng": self.depot_info['longitude'],
                    "type": "depot",
                    "sequence": len(route.points) + 1
                }
                route_coordinates.append(depot_end_coord)
                
                result_data["routes"].append({
                    "vehicle_id": int(route.vehicle.id.split('_')[-1]) if '_' in route.vehicle.id else 0,
                    "vehicle_name": route.vehicle.name,
                    "vehicle_type": route.vehicle.type,
                    "depot_id": self.depot_info['id'],
                    "depot_name": self.depot_info['address1'],
                    "coordinates": route_coordinates,
                    "distance": route.total_distance,
                    "time": route.total_time,
                    "delivery_count": len(route.points),
                    "capacity_usage": {
                        "volume": route.total_load['volume'] / route.vehicle.capacity.volume,
                        "weight": route.total_load['weight'] / route.vehicle.capacity.weight
                    },
                    "status": route.status
                })
            
            logger.info("JSON 변환이 완료되었습니다.")
            return result_data
            
        except Exception as e:
            logger.error(f"JSON 변환 중 오류 발생: {str(e)}")
            return {}
    
    def save_to_file(self, data: Dict[str, Any], output_dir: str = "data") -> Optional[str]:
        """JSON 데이터를 파일로 저장"""
        try:
            # 프로젝트 루트 기준으로 경로 설정
            output_file = project_root / output_dir / "extracted_coordinates.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과가 '{output_file}'에 저장되었습니다.")
            
            # 파일 크기 정보
            file_size = output_file.stat().st_size / 1024
            logger.info(f"파일 크기: {file_size:.1f}KB")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"파일 저장 중 오류 발생: {str(e)}")
            return None
    
    def extract_routes(self, output_dir: str = "data") -> Optional[Dict[str, Any]]:
        """전체 경로 추출 프로세스 실행 - 다중 TC 지원"""
        start_time = time.time()
        
        try:
            logger.info("🚛 고급 TMS 시스템 기반 배송 경로 최적화 시작")
            logger.info("=" * 60)
            
            # 1. 데이터 로드
            delivery_points = self.load_delivery_data()
            if not delivery_points:
                return None
            
            # 2. 다중 TC 지원 여부 확인 (force_multi_center 플래그 추가 확인)
            force_multi_center = self.config.get('logistics.force_multi_center', False) if self.config else False
            
            if (self.depot_selector and len(self.depot_selector.depots) > 1) or force_multi_center:
                logger.info("🏢 다중 TC 시스템으로 모든 배송지를 처리합니다...")
                if force_multi_center:
                    logger.info("🔧 강제 다중 센터 모드 활성화됨")
                return self._process_multi_depot_routes(delivery_points, output_dir, start_time)
            else:
                logger.info("🏢 단일 TC 시스템으로 처리합니다...")
                return self._process_single_depot_routes(delivery_points, output_dir, start_time)
            
        except Exception as e:
            logger.error(f"최적화 프로세스 중 오류 발생: {str(e)}")
            return None
    
    def _process_multi_depot_routes(self, delivery_points: List[DeliveryPoint], output_dir: str, start_time: float) -> Optional[Dict[str, Any]]:
        """다중 TC 시스템으로 모든 배송지 처리 - 병렬 처리 최적화"""
        try:
            # 중복 처리 방지를 위한 플래그
            if hasattr(self, '_multi_depot_processing'):
                logger.warning("⚠️ 다중 TC 처리가 이미 진행 중입니다. 중복 실행을 방지합니다.")
                return None
            
            self._multi_depot_processing = True
            
            # 배송지를 각 TC별로 할당
            tc_assignments = self._assign_deliveries_to_nearest_tc(delivery_points)
            
            all_routes = []
            all_stats = []
            total_processed_points = 0
            
            # 성능 최적화: 병렬 처리로 TC별 최적화 실행
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # TC별 작업 준비
            tc_tasks = []
            for tc_id, tc_data in tc_assignments.items():
                if tc_data['delivery_points']:
                    tc_tasks.append((tc_id, tc_data))
            
            logger.info(f"🚀 {len(tc_tasks)}개 TC에서 병렬 처리 시작...")
            
            # 병렬 처리 실행 (최대 4개 스레드로 제한)
            max_workers = min(4, len(tc_tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 TC별 작업 제출
                future_to_tc = {}
                for tc_id, tc_data in tc_tasks:
                    future = executor.submit(self._process_single_tc, tc_id, tc_data, output_dir)
                    future_to_tc[future] = tc_id
                
                # 결과 수집
                for future in as_completed(future_to_tc):
                    tc_id = future_to_tc[future]
                    try:
                        tc_result = future.result(timeout=300)  # 5분 타임아웃
                        if tc_result:
                            all_routes.extend(tc_result['routes'])
                            all_stats.append(tc_result['stats'])
                            total_processed_points += tc_result['stats']['processed_points']
                            logger.info(f"✅ {tc_id} 처리 완료: {tc_result['stats']['processed_points']}개 배송지")
                        else:
                            logger.warning(f"⚠️ {tc_id} 처리 결과가 없습니다.")
                    except Exception as e:
                        logger.error(f"❌ {tc_id} 처리 중 오류: {str(e)}")
                
            # 중복 처리 방지 플래그 해제
            delattr(self, '_multi_depot_processing')
            
            if not all_routes:
                logger.error("❌ 처리된 경로가 없습니다.")
                return None
            
            # 전체 통계 계산
            total_distance = sum(stat['total_distance'] for stat in all_stats)
            total_time = sum(stat['total_time'] for stat in all_stats)
            total_vehicles = sum(stat['vehicle_count'] for stat in all_stats)
            
            # 결과 저장 - Route 객체를 딕셔너리로 변환
            # 다중 센터 정보를 depots 배열로 저장
            depots_info = []
            for depot in self.depot_selector.depots:
                depots_info.append({
                    'id': depot['id'],
                    'label': depot['name'],
                    'lat': depot['latitude'],
                    'lng': depot['longitude'],
                    'address': depot.get('address', depot['name'])
                })
            
            result = {
                'routes': [route.to_dict() if hasattr(route, 'to_dict') else route for route in all_routes],
                'depot': self.depot_selector.depots[0],  # 첫 번째 depot을 대표로 사용 (하위 호환성)
                'depots': depots_info,  # 다중 센터 정보 추가
                'stats': {
                    'total_distance': total_distance,
                    'total_time': total_time,
                    'vehicle_count': total_vehicles,
                    'processed_points': total_processed_points,
                    'tc_count': len(tc_tasks),
                    'tc_stats': all_stats
                }
            }
            
            # JSON 파일로 저장
            output_file = os.path.join(output_dir, 'extracted_coordinates.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            file_size = os.path.getsize(output_file) / 1024  # KB
            logger.info(f"결과가 '{output_file}'에 저장되었습니다.")
            logger.info(f"파일 크기: {file_size:.1f}KB")
            
            # 🗺️ 폴리라인 시각화 생성
            try:
                logger.info("🗺️ 폴리라인 시각화 생성 중...")
                visualization_file = os.path.join(output_dir, 'route_visualization_final.html')
                create_polyline_visualization(all_routes, visualization_file)
                logger.info(f"✅ 폴리라인 시각화 완료: {visualization_file}")
            except Exception as e:
                logger.warning(f"⚠️ 폴리라인 시각화 생성 실패: {e}")
            
            # 결과 요약 출력
            processing_time = time.time() - start_time
            logger.info("\n=== 다중 TC TMS 최적화 결과 ===")
            logger.info(f"처리 시간: {processing_time:.1f}초")
            logger.info(f"처리된 TC: {len(tc_tasks)}개")
            logger.info(f"총 배송지: {total_processed_points}개")
            logger.info(f"투입 차량: {total_vehicles}대")
            logger.info(f"총 거리: {total_distance:.1f}km")
            logger.info(f"총 시간: {total_time:.0f}분 ({total_time/60:.1f}시간)")
            
            if total_vehicles > 0:
                efficiency = (total_time / 60) / (total_vehicles * 8) * 100  # 8시간 기준
                avg_distance = total_distance / total_vehicles
                avg_time = total_time / total_vehicles
                
                logger.info(f"시간 효율성: {efficiency:.1f}%")
                logger.info(f"차량당 평균 거리: {avg_distance:.1f}km")
                logger.info(f"차량당 평균 시간: {avg_time:.0f}분")
            
            # TC별 상세 결과
            logger.info("\n📊 TC별 상세 결과:")
            for stat in all_stats:
                logger.info(f"   {stat['tc_name']}: {stat['processed_points']}개 배송지, {stat['vehicle_count']}대 차량, {stat['total_distance']:.1f}km")
                
            return result
            
        except Exception as e:
            # 중복 처리 방지 플래그 해제
            if hasattr(self, '_multi_depot_processing'):
                delattr(self, '_multi_depot_processing')
            logger.error(f"❌ 다중 TC 처리 중 오류 발생: {str(e)}")
            return None
    
    def _process_single_tc(self, tc_id: str, tc_data: Dict, output_dir: str) -> Optional[Dict]:
        """단일 TC 처리 (병렬 처리용)"""
        try:
            depot = tc_data['depot']
            delivery_points = tc_data['delivery_points']
            
            if not delivery_points:
                logger.warning(f"⚠️ {depot['name']}: 배송지가 없습니다.")
                return None
            
            logger.info(f"🏢 {depot['name']} 처리 시작: {len(delivery_points)}개 배송지")
            
            # 임시로 depot 정보 설정
            original_depot = self.depot_info
            self.depot_info = depot
            
            # 해당 TC의 배송지만으로 클러스터링 최적화 실행
            routes = self.clustering_optimization(delivery_points)
            
            # 원래 depot 정보 복원
            self.depot_info = original_depot
            
            if not routes:
                logger.warning(f"⚠️ {depot['name']}: 경로 최적화 실패")
                return None
            
            # 통계 계산
            total_distance = sum(route.total_distance for route in routes)
            total_time = sum(route.total_time for route in routes)
            total_points = len(delivery_points)
            vehicle_count = len(routes)
            
            logger.info(f"✅ {depot['name']} 완료: {vehicle_count}대 차량, {total_distance:.1f}km, {total_time:.0f}분")
            
            return {
                'routes': routes,
                'stats': {
                    'tc_id': tc_id,
                    'tc_name': depot['name'],
                    'processed_points': total_points,
                    'vehicle_count': vehicle_count,
                    'total_distance': total_distance,
                    'total_time': total_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ {tc_id} 처리 중 오류: {str(e)}")
            return None
    
    def _assign_deliveries_to_nearest_tc(self, delivery_points: List[DeliveryPoint]) -> Dict:
        """모든 배송지를 가장 가까운 TC로 할당 - 정확한 거리 기반 배정 (개선됨)"""
        from src.utils.distance_calculator import assign_points_to_nearest_centers, calculate_distance
        
        tc_assignments = {}
        
        # 각 TC 초기화
        for depot in self.depot_selector.depots:
            tc_assignments[depot['id']] = {
                'depot': depot,
                'delivery_points': []
            }
        
        if not delivery_points:
            return tc_assignments
        
        logger.info(f"📊 {len(delivery_points)}개 배송지를 {len(self.depot_selector.depots)}개 TC로 정확한 거리 기반 할당 중...")
        
        # 개별 배송지별로 가장 가까운 TC 찾기 (정확한 방법)
        assignment_stats = []
        
        for point in delivery_points:
            min_distance = float('inf')
            best_tc_id = None
            best_tc_depot = None
            
            # 모든 TC와의 거리 계산
            for depot in self.depot_selector.depots:
                distance = calculate_distance(
                    point.latitude, point.longitude,
                    depot['latitude'], depot['longitude']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_tc_id = depot['id']
                    best_tc_depot = depot
            
            # 가장 가까운 TC에 할당
            if best_tc_id and best_tc_depot:
                tc_assignments[best_tc_id]['delivery_points'].append(point)
                assignment_stats.append({
                    'point_id': point.id,
                    'tc_id': best_tc_id,
                    'tc_name': best_tc_depot['name'],
                    'distance': min_distance
                })
        
        # 할당 결과 상세 로깅 및 검증
        logger.info("📊 TC별 배송지 할당 결과 (정확한 거리 기반):")
        total_assigned = 0
        avg_distances = []
        
        for tc_id, tc_data in tc_assignments.items():
            count = len(tc_data['delivery_points'])
            total_assigned += count
            percentage = (count / len(delivery_points)) * 100 if delivery_points else 0
            
            # 해당 TC에 할당된 배송지들의 거리 통계 계산
            if count > 0:
                tc_depot = tc_data['depot']
                distances = [
                    calculate_distance(
                        point.latitude, point.longitude,
                        tc_depot['latitude'], tc_depot['longitude']
                    )
                    for point in tc_data['delivery_points']
                ]
                avg_distance = sum(distances) / len(distances)
                max_distance = max(distances)
                min_distance = min(distances)
                avg_distances.append(avg_distance)
                
                logger.info(f"   {tc_data['depot']['name']}: {count}개 ({percentage:.1f}%) - "
                          f"평균거리: {avg_distance:.1f}km, 최대: {max_distance:.1f}km, 최소: {min_distance:.1f}km")
            else:
                logger.info(f"   {tc_data['depot']['name']}: {count}개 ({percentage:.1f}%)")
        
        # 전체 할당 품질 검증 및 최적화 확인
        if avg_distances:
            overall_avg = sum(avg_distances) / len(avg_distances)
            logger.info(f"📈 할당 품질: 전체 평균 거리 {overall_avg:.1f}km")
            
            # 🔍 할당 최적성 검증: 각 배송지가 정말 가장 가까운 TC에 할당되었는지 확인
            misassigned_count = 0
            for stat in assignment_stats:
                # 다른 TC들과의 거리도 계산해서 정말 최단거리인지 확인
                point_id = stat['point_id']
                assigned_distance = stat['distance']
                assigned_tc_id = stat['tc_id']
                
                # 해당 배송지 찾기
                point = next((p for p in delivery_points if p.id == point_id), None)
                if not point:
                    continue
                
                # 다른 모든 TC와의 거리 확인
                for depot in self.depot_selector.depots:
                    if depot['id'] != assigned_tc_id:
                        other_distance = calculate_distance(
                            point.latitude, point.longitude,
                            depot['latitude'], depot['longitude']
                        )
                        if other_distance < assigned_distance:
                            logger.warning(f"⚠️ 배송지 {point_id}: {depot['name']}이 더 가까움 ({other_distance:.1f}km < {assigned_distance:.1f}km)")
                            misassigned_count += 1
                            break
            
            if misassigned_count == 0:
                logger.info(f"✅ 할당 최적성 검증 완료: 모든 배송지가 최적 TC에 할당됨")
            else:
                logger.warning(f"⚠️ 할당 오류 발견: {misassigned_count}개 배송지가 최적이 아닌 TC에 할당됨")
            
            # 비정상적으로 먼 할당이 있는지 확인
            long_assignments = [stat for stat in assignment_stats if stat['distance'] > 50.0]  # 50km 이상
            if long_assignments:
                logger.warning(f"⚠️ 장거리 할당 발견: {len(long_assignments)}개 (50km 이상)")
                for stat in long_assignments[:5]:  # 상위 5개만 로깅
                    logger.warning(f"   배송지 {stat['point_id']} → {stat['tc_name']}: {stat['distance']:.1f}km")
        
        # 할당 검증: 모든 배송지가 할당되었는지 확인
        if total_assigned != len(delivery_points):
            logger.error(f"❌ 할당 오류: 총 {len(delivery_points)}개 중 {total_assigned}개만 할당됨")
        else:
            logger.info(f"✅ 모든 배송지 할당 완료: {total_assigned}개")
        
        # ✅ 가장 가까운 TC 배정을 유지 (용량 최적화 제거)
        logger.info("🔒 가장 가까운 TC 배정 원칙 준수: 용량 재할당 건너뜀")
        
        return tc_assignments
    
    def _process_single_depot_routes(self, delivery_points: List[DeliveryPoint], output_dir: str, start_time: float) -> Optional[Dict[str, Any]]:
        """단일 TC 시스템으로 처리 (기존 로직)"""
        try:
            # 거리 필터링
            filtered_points = self.filter_points_by_distance(delivery_points, max_distance=15.0)
            if not filtered_points:
                logger.error("필터링된 배송지점이 없습니다.")
                return None
            
            # 클러스터링 기반 경로 최적화
            routes = self.clustering_optimization(filtered_points)
            
            if not routes:
                logger.error("경로 최적화에 실패했습니다.")
                return None
            
            # JSON 변환
            result_data = self.convert_routes_to_json(routes)
            if not result_data:
                return None
            
            # 파일 저장
            output_file = self.save_to_file(result_data, output_dir)
            if not output_file:
                return None
            
            # 🗺️ 폴리라인 시각화 생성
            try:
                logger.info("🗺️ 폴리라인 시각화 생성 중...")
                visualization_file = os.path.join(output_dir, 'route_visualization_final.html')
                create_polyline_visualization(routes, visualization_file)
                logger.info(f"✅ 폴리라인 시각화 완료: {visualization_file}")
            except Exception as e:
                logger.warning(f"⚠️ 폴리라인 시각화 생성 실패: {e}")
            
            # 최종 결과 출력
            elapsed_time = time.time() - start_time
            stats = result_data['stats']
            
            logger.info(f"\n=== 고급 TMS 최적화 결과 ===")
            logger.info(f"처리 시간: {elapsed_time:.1f}초")
            logger.info(f"투입 차량: {stats['total_vehicles']}대")
            logger.info(f"배송지점: {stats['total_points']}개")
            logger.info(f"총 거리: {stats['total_distance']:.1f}km")
            logger.info(f"총 시간: {stats['total_time']:.0f}분 ({stats['total_time']/60:.1f}시간)")
            logger.info(f"시간 효율성: {stats['time_efficiency']:.1%}")
            logger.info(f"차량당 평균 거리: {stats['avg_distance_per_vehicle']:.1f}km")
            logger.info(f"차량당 평균 시간: {stats['avg_time_per_vehicle']:.0f}분")
            
            return result_data
            
        except Exception as e:
            logger.error(f"단일 TC 처리 중 오류 발생: {str(e)}")
            return None

def main():
    """메인 실행 함수"""
    extractor = RouteExtractorService()
    result = extractor.extract_routes()
    
    if result:
        print("✅ 고급 TMS 시스템 기반 최적화가 성공적으로 완료되었습니다!")
        return result
    else:
        print("❌ 최적화에 실패했습니다.")
        return None

if __name__ == "__main__":
    main() 