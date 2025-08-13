"""
배차 오케스트레이터 - 전체 배차 프로세스 관리
"""
from typing import List, Dict, Optional
from datetime import datetime
import time
import logging

from ..models import (Order, Vehicle, Region, DispatchResult, VehicleAssignment, 
                   DispatchStatus, DispatchMetrics)
from .data_collector import DataCollector
from .condition_analyzer import ConditionAnalyzer
from .capacity_calculator import CapacityCalculator
from ..database.transaction_manager import get_transaction_manager


class DispatchOrchestrator:
    """배차 프로세스 전체 관리"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 서비스 초기화
        self.data_collector = DataCollector(config)
        self.condition_analyzer = ConditionAnalyzer(config)
        self.capacity_calculator = CapacityCalculator(config)
        self.transaction_manager = get_transaction_manager()
    
    def execute_dispatch(self, center_id: str = None, driver_name: str = None) -> DispatchResult:
        """배차 실행 메인 프로세스 - 트랜잭션 지원"""
        if not center_id:
            raise ValueError("센터 ID는 필수입니다")
            
        batch_id = self._generate_batch_id()
        start_time = time.time()
        
        self.logger.info(f"배차 시작: {batch_id} (센터: {center_id}, 기사: {driver_name})")
        
        # 트랜잭션 컨텍스트에서 배차 실행
        try:
            with self.transaction_manager.dispatch_transaction(batch_id, center_id) as tx_context:
                # 1단계: 데이터 수집
                dispatch_data = self._collect_data(center_id, driver_name)
                
                # 2단계: 외부 조건 분석
                conditions = self._analyze_conditions(dispatch_data['regions'])
                
                # 3단계: 용량 계산
                capacities = self._calculate_capacities(
                    dispatch_data['vehicles'], 
                    dispatch_data['regions'],
                    conditions['weather'],
                    conditions['traffic']
                )
                
                # 4단계: 최적화 실행
                assignments = self._execute_optimization(
                    dispatch_data['orders'],
                    dispatch_data['vehicles'],
                    capacities,
                    conditions
                )
                
                # 5단계: 원자적 배정 처리
                if assignments:
                    success = tx_context.assign_orders_to_vehicle(assignments)
                    if not success:
                        raise ValueError("주문 배정 처리 실패")
                
                # 6단계: 배차 완료 처리
                execution_time = time.time() - start_time
                algorithm_used = "OR-Tools VRP"  # OR-Tools VRP 알고리즘
                
                success = tx_context.complete_dispatch(
                    algorithm_used=algorithm_used,
                    execution_time=execution_time,
                    weather_conditions=conditions.get('weather'),
                    traffic_conditions=conditions.get('traffic')
                )
                
                if not success:
                    raise ValueError("배차 완료 처리 실패")
                
                # 7단계: 결과 생성
                result = self._create_dispatch_result(
                    batch_id=batch_id,
                    assignments=assignments,
                    excluded_vehicles=dispatch_data['excluded_vehicles'],
                    conditions=conditions,
                    execution_time=execution_time,
                    transaction_stats=tx_context.get_batch_statistics()
                )
                
                self.logger.info(f"배차 완료: {result.get_summary_text()}")
                return result
                
        except Exception as e:
            self.logger.error(f"배차 실행 오류: {str(e)}")
            return DispatchResult(
                batch_id=batch_id,
                timestamp=datetime.now(),
                status=DispatchStatus.FAILED,
                error_message=str(e)
            )
    
    def _collect_data(self, center_id: str = None, driver_name: str = None) -> Dict:
        """1단계: 데이터 수집"""
        self.logger.info("데이터 수집 시작")
        
        # 기본 데이터 수집
        orders = self.data_collector.get_pending_orders(center_id)
        vehicles = self.data_collector.get_available_vehicles(center_id)
        regions = self.data_collector.get_regions(center_id)
        excluded_vehicles = self.data_collector.get_excluded_vehicles(center_id)
        
        # 특정 기사 지정 시 필터링
        if driver_name:
            vehicles = [v for v in vehicles if v.driver_name == driver_name]
            if not vehicles:
                raise ValueError(f"지정된 기사 '{driver_name}'를 찾을 수 없습니다")
        
        # 데이터 유효성 검증
        if not orders:
            raise ValueError("배차할 주문이 없습니다")
        if not vehicles:
            raise ValueError("사용 가능한 차량이 없습니다")
        if not regions:
            raise ValueError("권역 정보가 없습니다")
        
        # 데이터 일관성 검증
        if not self.data_collector.validate_data_consistency(orders, vehicles, regions):
            raise ValueError("데이터 일관성 검증 실패")
        
        self.logger.info(f"데이터 수집 완료: 주문 {len(orders)}개, 차량 {len(vehicles)}대, 권역 {len(regions)}개")
        
        return {
            'orders': orders,
            'vehicles': vehicles,
            'regions': regions,
            'excluded_vehicles': excluded_vehicles
        }
    
    def _analyze_conditions(self, regions: List[Region]) -> Dict:
        """2단계: 외부 조건 분석"""
        self.logger.info("외부 조건 분석 시작")
        
        # 날씨 조건 분석
        weather_conditions = self.condition_analyzer.analyze_weather_conditions(regions)
        
        # 교통 조건 분석
        traffic_conditions = self.condition_analyzer.analyze_traffic_conditions(regions)
        
        # 배송 실행 가능성 확인
        feasibility = self.condition_analyzer.check_delivery_feasibility(regions)
        
        # 비상 상황 확인
        emergency_conditions = self.condition_analyzer.get_emergency_conditions(regions)
        
        self.logger.info("외부 조건 분석 완료")
        if emergency_conditions:
            self.logger.warning(f"비상 상황: {', '.join(emergency_conditions)}")
        
        return {
            'weather': weather_conditions,
            'traffic': traffic_conditions,
            'feasibility': feasibility,
            'emergency': emergency_conditions
        }
    
    def _calculate_capacities(self, vehicles: List[Vehicle], regions: List[Region],
                            weather_conditions: Dict, traffic_conditions: Dict) -> Dict:
        """3단계: 용량 계산"""
        self.logger.info("차량 용량 계산 시작")
        
        # 차량별 조정된 용량 계산
        vehicle_capacities = self.capacity_calculator.calculate_vehicle_capacities(
            vehicles, regions, weather_conditions, traffic_conditions
        )
        
        # 권역별 부하 분산 계산
        region_distribution = self.capacity_calculator.calculate_region_load_distribution(
            vehicles, regions
        )
        
        # 용량 요약 정보
        capacity_summary = self.capacity_calculator.get_capacity_summary(
            vehicle_capacities, vehicles
        )
        
        self.logger.info("차량 용량 계산 완료")
        
        return {
            'vehicle_capacities': vehicle_capacities,
            'region_distribution': region_distribution,
            'summary': capacity_summary
        }
    
    def _execute_optimization(self, orders: List[Order], vehicles: List[Vehicle],
                            capacities: Dict, conditions: Dict) -> List[VehicleAssignment]:
        """4단계: OR-Tools VRP 최적화 실행"""
        self.logger.info("OR-Tools VRP 최적화 알고리즘 실행 시작")
        
        try:
            # 알고리즘 팩토리에서 최적 알고리즘 선택
            from ..algorithms import get_algorithm_factory
            
            # 외부 API 키들 전달
            api_conditions = {
                'openroute_api_key': self.config.get('weather_api_key', 'demo_key'),  # 임시로 weather_api_key 사용
                'here_api_key': self.config.get('traffic_api_key', 'demo_key'),      # 임시로 traffic_api_key 사용
                'kakao_api_key': 'demo_key',
                'emergency': conditions.get('emergency', False),
                'time_limit_seconds': 120,  # 2분 제한
                'verbose': False,
                **conditions
            }
            
            factory = get_algorithm_factory()
            
            # 지역(regions) 정보가 없으면 빈 리스트로 처리
            regions = []
            
            # 최적 알고리즘 선택 및 실행
            algorithm = factory.create_optimal_algorithm(orders, vehicles, regions, api_conditions)
            
            self.logger.info(f"선택된 알고리즘: {algorithm.algorithm_name}")
            
            # 최적화 실행
            result = algorithm.optimize(orders, vehicles, regions, api_conditions)
            
            if result.assignments:
                assigned_orders_count = result.convergence_info.get('assigned_orders', 0) if result.convergence_info else 0
                unassigned_orders_count = len(result.unassigned_orders)
                self.logger.info(f"OR-Tools VRP 최적화 완료: "
                               f"{len(result.assignments)}대 차량 배정, "
                               f"{assigned_orders_count}개 주문 배정, "
                               f"{unassigned_orders_count}개 미배정, "
                               f"품질점수: {result.quality_score:.3f}, "
                               f"실행시간: {result.execution_time_seconds:.1f}초")
                return result.assignments
            else:
                self.logger.warning("OR-Tools VRP 최적화 결과가 없음, 폴백 알고리즘으로 처리")
                return self._execute_fallback_optimization(orders, vehicles, capacities)
                
        except Exception as e:
            self.logger.error(f"OR-Tools VRP 최적화 오류: {str(e)}")
            self.logger.info("폴백 알고리즘으로 전환")
            return self._execute_fallback_optimization(orders, vehicles, capacities)
    
    def _execute_fallback_optimization(self, orders: List[Order], vehicles: List[Vehicle],
                                     capacities: Dict) -> List[VehicleAssignment]:
        """폴백: 기존 간단한 최적화 알고리즘"""
        self.logger.info("폴백 최적화 알고리즘 실행")
        
        assignments = []
        vehicle_capacities = capacities['vehicle_capacities']
        unassigned_orders = orders.copy()
        
        for vehicle in vehicles:
            if not vehicle.is_auto_dispatch_eligible():
                continue
            
            vehicle_capacity = vehicle_capacities.get(vehicle.id, 0)
            if vehicle_capacity <= 0:
                continue
            
            # 해당 차량 권역의 주문들 필터링
            region_orders = [o for o in unassigned_orders if o.region_id == vehicle.region_id]
            
            # 용량만큼 주문 배정 (간단한 구현)
            assigned_orders = region_orders[:vehicle_capacity]
            
            if assigned_orders:
                # 개선된 거리 및 시간 추정
                estimated_distance = len(assigned_orders) * 2.0  # 주문당 평균 2km로 더 현실적으로
                travel_time = int(estimated_distance / 25 * 60)  # 25km/h 기준 이동시간
                delivery_time = len(assigned_orders) * 8  # 주문당 8분 배송시간  
                setup_time = 5  # 차량 준비시간
                estimated_time = travel_time + delivery_time + setup_time
                
                assignment = VehicleAssignment(
                    vehicle_id=vehicle.id,
                    driver_name=vehicle.driver_name,
                    vehicle_type=vehicle.vehicle_type.value,
                    region_name=f"권역_{vehicle.region_id}",
                    assigned_orders=[o.id for o in assigned_orders],
                    estimated_distance_km=estimated_distance,
                    estimated_time_minutes=estimated_time,
                    capacity_utilization=len(assigned_orders) / vehicle_capacity
                )
                
                assignments.append(assignment)
                
                # 배정된 주문 제거
                for order in assigned_orders:
                    unassigned_orders.remove(order)
        
        self.logger.info(f"폴백 최적화 완료: {len(assignments)}대 차량에 배정")
        
        return assignments
    
    def _create_dispatch_result(self, batch_id: str, assignments: List[VehicleAssignment],
                               excluded_vehicles: List[Vehicle], conditions: Dict,
                               execution_time: float, transaction_stats: Dict = None) -> DispatchResult:
        """5단계: 결과 생성"""
        
        # 트랜잭션 통계가 있으면 사용, 없으면 기본 계산
        if transaction_stats:
            total_assigned = transaction_stats.get('assigned_orders', 0)
            status_str = transaction_stats.get('status', 'unknown')
            try:
                status = DispatchStatus(status_str) if status_str != 'unknown' else DispatchStatus.FAILED
            except ValueError:
                status = DispatchStatus.FAILED
        else:
            # 배정되지 않은 주문 계산
            total_assigned = sum(len(a.assigned_orders) for a in assignments)
            
            # 상태 결정
            if assignments and total_assigned > 0:
                status = DispatchStatus.SUCCESS
            elif assignments:
                status = DispatchStatus.PARTIAL_SUCCESS
            else:
                status = DispatchStatus.FAILED
        
        result = DispatchResult(
            batch_id=batch_id,
            timestamp=datetime.now(),
            status=status,
            vehicle_assignments=assignments,
            excluded_vehicles=[v.id for v in excluded_vehicles],
            external_conditions=conditions
        )
        
        # 메트릭스 업데이트
        if result.metrics:
            result.metrics.execution_time_seconds = execution_time
            result.metrics.algorithm_used = "OR-Tools VRP"  # OR-Tools VRP 알고리즘
        
        # 경고 메시지 추가
        if excluded_vehicles:
            result.add_warning(f"수동 배차 필요: {len(excluded_vehicles)}대 차량")
        
        if conditions.get('emergency'):
            result.add_warning(f"비상 상황: {', '.join(conditions['emergency'])}")
        
        return result
    
    def _generate_batch_id(self) -> str:
        """배치 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"DISPATCH_{timestamp}"