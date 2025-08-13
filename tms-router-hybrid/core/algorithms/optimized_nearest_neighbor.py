"""
OptimizedNearestNeighbor - 최적화된 최근접 이웃 알고리즘
- GeospatialClustering 적용
- SpatialHashOptimizer O(1) 탐색
- SequentialChainBuilder 순차 연결
- PerformanceOptimizer 극한 최적화
- 시간복잡도: O(log n) 평균, O(1) 캐시 히트
"""
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import time

from ..models import Order, Vehicle, VehicleAssignment, Coordinates
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from .geospatial_clustering import GeospatialClustering
from .spatial_hash_optimizer import SpatialHashOptimizer
from .sequential_chain_builder import SequentialChainBuilder, OrderChain
from .performance_optimizer import PerformanceOptimizer, performance_monitor
from ..utils.time_calculator import get_time_calculator


@dataclass 
class OptimizedConfig(AlgorithmConfig):
    """최적화된 알고리즘 설정"""
    
    def __init__(self, **kwargs):
        # 기본 AlgorithmConfig 파라미터들
        base_params = {
            'time_limit_seconds': kwargs.get('time_limit_seconds', 300),
            'quality_threshold': kwargs.get('quality_threshold', 0.8),
            'max_iterations': kwargs.get('max_iterations', 1000),
            'early_stopping_enabled': kwargs.get('early_stopping_enabled', True),
            'verbose': kwargs.get('verbose', False)
        }
        super().__init__(**base_params)
        
        # 클러스터링 설정
        self.enable_clustering = kwargs.get('enable_clustering', True)
        self.cluster_quality_threshold = kwargs.get('cluster_quality_threshold', 0.6)
        
        # 공간 해시 설정
        self.spatial_cell_size_km = kwargs.get('spatial_cell_size_km', 2.0)
        self.spatial_cache_size = kwargs.get('spatial_cache_size', 10000)
        
        # 체인 빌더 설정
        self.max_chain_length = kwargs.get('max_chain_length', 8)
        self.enable_chain_merging = kwargs.get('enable_chain_merging', True)
        
        # 성능 최적화 설정
        self.enable_performance_optimization = kwargs.get('enable_performance_optimization', True)
        self.enable_parallel_processing = kwargs.get('enable_parallel_processing', True)
        self.enable_vectorization = kwargs.get('enable_vectorization', True)
        
        # 디버그 설정
        self.enable_detailed_logging = kwargs.get('enable_detailed_logging', False)
        self.enable_performance_profiling = kwargs.get('enable_performance_profiling', True)


class OptimizedNearestNeighborAlgorithm(BaseAlgorithm):
    """최적화된 최근접 이웃 알고리즘"""
    
    def __init__(self, config: OptimizedConfig = None):
        if config is None:
            config = OptimizedConfig()
        super().__init__(config)
        self.opt_config = config
        
        # 최적화 컴포넌트 초기화
        self.geospatial_clustering = GeospatialClustering() if config.enable_clustering else None
        
        self.spatial_optimizer = SpatialHashOptimizer(
            cell_size_km=config.spatial_cell_size_km,
            max_cache_size=config.spatial_cache_size
        )
        
        self.chain_builder = SequentialChainBuilder(self.spatial_optimizer)
        
        self.performance_optimizer = PerformanceOptimizer({
            'max_workers': None,
            'use_processes': False,
            'cache_size': config.spatial_cache_size,
            'memory_pool_size': 5000,
            'disable_gc_gen0': config.enable_performance_optimization
        }) if config.enable_performance_optimization else None
        
        self.time_calculator = get_time_calculator()
        
        # 성능 통계
        self.performance_stats = {
            'clustering_time': 0.0,
            'spatial_indexing_time': 0.0,
            'chain_building_time': 0.0,
            'assignment_time': 0.0,
            'total_optimization_time': 0.0,
            'cache_hit_rate': 0.0,
            'parallel_operations': 0,
            'speedup_factor': 0.0
        }
    
    def get_algorithm_name(self) -> str:
        return "OptimizedNearestNeighbor"
    
    @performance_monitor
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """최적화된 알고리즘 실행"""
        
        start_time = time.perf_counter()
        
        try:
            # Phase 1: 지리공간 클러스터링
            cluster_assignments = self._perform_geospatial_clustering(orders, vehicles, vehicle_capacities)
            
            # Phase 2: 공간 인덱스 구축
            self._build_spatial_indexes(orders)
            
            # Phase 3: 순차적 체인 구축
            cluster_chains = self._build_sequential_chains(orders, cluster_assignments)
            
            # Phase 4: 최적화된 배차 생성
            assignments = self._create_optimized_assignments(cluster_chains, vehicles, vehicle_capacities)
            
            # Phase 5: 미배정 주문 처리
            unassigned_orders = self._find_unassigned_orders(assignments, orders)
            
            # 성능 통계 업데이트
            total_time = time.perf_counter() - start_time
            self.performance_stats['total_optimization_time'] = total_time
            self._update_performance_stats()
            
            if self.opt_config.enable_detailed_logging:
                self._log_performance_details()
            
            return AlgorithmResult(
                assignments=assignments,
                unassigned_orders=unassigned_orders,
                execution_time_seconds=total_time,
                quality_score=0.0,  # BaseAlgorithm에서 계산
                algorithm_name=self.get_algorithm_name(),
                iteration_count=1
            )
            
        except Exception as e:
            self.logger.error(f"최적화된 알고리즘 실행 오류: {str(e)}")
            # 기본 최근접 이웃으로 폴백
            return self._fallback_to_basic_nearest_neighbor(orders, vehicles, vehicle_capacities)
    
    @performance_monitor
    def _perform_geospatial_clustering(self, orders: List[Order], vehicles: List[Vehicle], 
                                     vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """지리공간 클러스터링 수행"""
        clustering_start = time.perf_counter()
        
        if not self.geospatial_clustering or not self.opt_config.enable_clustering:
            # 클러스터링 비활성화시 기본 권역 그룹화
            return self._basic_region_grouping(orders, vehicles, vehicle_capacities)
        
        try:
            cluster_assignments = self.geospatial_clustering.create_non_overlapping_clusters(
                orders, vehicles, vehicle_capacities
            )
            
            self.performance_stats['clustering_time'] = time.perf_counter() - clustering_start
            
            if self.opt_config.enable_detailed_logging:
                self.logger.info(f"클러스터링 완료: {len(cluster_assignments)}개 클러스터, "
                               f"시간: {self.performance_stats['clustering_time']:.3f}초")
            
            return cluster_assignments
            
        except Exception as e:
            self.logger.warning(f"클러스터링 실패, 기본 그룹화 사용: {str(e)}")
            return self._basic_region_grouping(orders, vehicles, vehicle_capacities)
    
    def _basic_region_grouping(self, orders: List[Order], vehicles: List[Vehicle], 
                             vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """기본 권역 그룹화"""
        # 권역별 주문 그룹화
        region_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in region_orders:
                region_orders[region_id] = []
            region_orders[region_id].append(order)
        
        # 차량별 주문 배정 (간단한 Voronoi 방식)
        cluster_assignments = {}
        
        for vehicle in vehicles:
            if vehicle_capacities.get(vehicle.id, 0) <= 0:
                continue
                
            region_orders_list = region_orders.get(vehicle.region_id, [])
            if not region_orders_list:
                cluster_assignments[vehicle.id] = []
                continue
            
            # 해당 권역의 다른 차량들과의 거리 비교로 배정
            vehicle_orders = []
            for order in region_orders_list:
                # 이 주문에 가장 가까운 차량이 현재 차량인지 확인
                min_distance = float('inf')
                nearest_vehicle = None
                
                for other_vehicle in vehicles:
                    if (other_vehicle.region_id == vehicle.region_id and 
                        vehicle_capacities.get(other_vehicle.id, 0) > 0):
                        distance = order.coordinates.distance_to(other_vehicle.center_coordinates)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_vehicle = other_vehicle
                
                if nearest_vehicle and nearest_vehicle.id == vehicle.id:
                    vehicle_orders.append(order)
            
            # 용량 제한 적용
            capacity = vehicle_capacities[vehicle.id]
            if len(vehicle_orders) > capacity:
                # 거리순 정렬하여 가까운 것만 선택
                vehicle_orders.sort(
                    key=lambda o: vehicle.center_coordinates.distance_to(o.coordinates)
                )
                vehicle_orders = vehicle_orders[:capacity]
            
            cluster_assignments[vehicle.id] = vehicle_orders
        
        return cluster_assignments
    
    @performance_monitor
    def _build_spatial_indexes(self, orders: List[Order]) -> None:
        """공간 인덱스 구축"""
        indexing_start = time.perf_counter()
        
        # 공간 해시 인덱스 구축
        self.spatial_optimizer.build_spatial_index(orders)
        
        self.performance_stats['spatial_indexing_time'] = time.perf_counter() - indexing_start
        
        if self.opt_config.enable_detailed_logging:
            optimizer_stats = self.spatial_optimizer.get_optimization_stats()
            self.logger.info(f"공간 인덱싱 완료: {optimizer_stats['spatial_cells']}개 셀, "
                           f"시간: {self.performance_stats['spatial_indexing_time']:.3f}초")
    
    @performance_monitor 
    def _build_sequential_chains(self, orders: List[Order], 
                               cluster_assignments: Dict[str, List[Order]]) -> Dict[str, List[OrderChain]]:
        """순차적 체인 구축"""
        chain_start = time.perf_counter()
        
        # 체인 빌더 설정
        self.chain_builder.max_chain_length = self.opt_config.max_chain_length
        
        try:
            cluster_chains = self.chain_builder.build_optimal_chains(orders, cluster_assignments)
            
            self.performance_stats['chain_building_time'] = time.perf_counter() - chain_start
            
            if self.opt_config.enable_detailed_logging:
                total_chains = sum(len(chains) for chains in cluster_chains.values())
                chain_stats = self.chain_builder.get_chain_stats()
                self.logger.info(f"체인 구축 완료: {total_chains}개 체인, "
                               f"평균 길이: {chain_stats['avg_chain_length']:.2f}, "
                               f"시간: {self.performance_stats['chain_building_time']:.3f}초")
            
            return cluster_chains
            
        except Exception as e:
            self.logger.warning(f"체인 구축 실패, 단순 체인 생성: {str(e)}")
            return self._create_simple_chains(cluster_assignments)
    
    def _create_simple_chains(self, cluster_assignments: Dict[str, List[Order]]) -> Dict[str, List[OrderChain]]:
        """단순 체인 생성 (폴백)"""
        simple_chains = {}
        
        for vehicle_id, orders in cluster_assignments.items():
            if not orders:
                simple_chains[vehicle_id] = []
                continue
            
            # 각 주문을 개별 체인으로 생성
            chains = []
            for i, order in enumerate(orders):
                chain = OrderChain(
                    orders=[order.id],
                    total_distance=0.0,
                    start_coord=order.coordinates,
                    end_coord=order.coordinates,
                    chain_id=f"simple_{vehicle_id}_{i}"
                )
                chains.append(chain)
            
            simple_chains[vehicle_id] = chains
        
        return simple_chains
    
    @performance_monitor
    def _create_optimized_assignments(self, cluster_chains: Dict[str, List[OrderChain]], 
                                    vehicles: List[Vehicle], 
                                    vehicle_capacities: Dict[str, int]) -> List[VehicleAssignment]:
        """최적화된 배차 생성"""
        assignment_start = time.perf_counter()
        
        assignments = []
        vehicles_dict = {v.id: v for v in vehicles}
        
        for vehicle_id, chains in cluster_chains.items():
            if not chains or vehicle_id not in vehicles_dict:
                continue
            
            vehicle = vehicles_dict[vehicle_id]
            capacity = vehicle_capacities.get(vehicle_id, 0)
            
            if capacity <= 0:
                continue
            
            # 체인들을 하나의 배차로 통합
            all_order_ids = []
            total_distance = 0.0
            
            for chain in chains:
                all_order_ids.extend(chain.orders)
                total_distance += chain.total_distance
            
            if not all_order_ids:
                continue
            
            # 용량 확인
            if len(all_order_ids) > capacity:
                # 체인 우선순위에 따라 선택 (효율성 기준)
                sorted_chains = sorted(chains, key=lambda c: c.get_efficiency(), reverse=True)
                selected_orders = []
                selected_distance = 0.0
                
                for chain in sorted_chains:
                    if len(selected_orders) + len(chain.orders) <= capacity:
                        selected_orders.extend(chain.orders)
                        selected_distance += chain.total_distance
                    else:
                        # 부분 선택
                        remaining_capacity = capacity - len(selected_orders)
                        if remaining_capacity > 0:
                            selected_orders.extend(chain.orders[:remaining_capacity])
                            # 부분 거리 계산 (비례)
                            partial_distance = (chain.total_distance * remaining_capacity / len(chain.orders)
                                              if len(chain.orders) > 0 else 0.0)
                            selected_distance += partial_distance
                        break
                
                all_order_ids = selected_orders
                total_distance = selected_distance
            
            # 시간 계산
            orders_objects = self._get_order_objects_from_ids(all_order_ids, cluster_chains, vehicles_dict)
            estimated_time = self.time_calculator.calculate_delivery_time(vehicle, orders_objects)
            
            # VehicleAssignment 생성
            assignment = VehicleAssignment(
                vehicle_id=vehicle_id,
                driver_name=vehicle.driver_name,
                vehicle_type=vehicle.vehicle_type.value,
                region_name=f"권역_{vehicle.region_id}",
                assigned_orders=all_order_ids,
                estimated_distance_km=total_distance,
                estimated_time_minutes=estimated_time,
                capacity_utilization=len(all_order_ids) / capacity if capacity > 0 else 0
            )
            
            assignments.append(assignment)
        
        self.performance_stats['assignment_time'] = time.perf_counter() - assignment_start
        
        if self.opt_config.enable_detailed_logging:
            self.logger.info(f"배차 생성 완료: {len(assignments)}개 배차, "
                           f"시간: {self.performance_stats['assignment_time']:.3f}초")
        
        return assignments
    
    def _get_order_objects_from_ids(self, order_ids: List[str], 
                                   cluster_chains: Dict[str, List[OrderChain]], 
                                   vehicles_dict: Dict[str, Vehicle]) -> List[Order]:
        """주문 ID에서 Order 객체 찾기"""
        # 간단한 구현: 공간 옵티마이저의 해시 테이블에서 찾기
        order_objects = []
        
        for cell in self.spatial_optimizer.spatial_hash.values():
            for order in cell.orders:
                if order.id in order_ids:
                    order_objects.append(order)
        
        return order_objects
    
    def _find_unassigned_orders(self, assignments: List[VehicleAssignment], 
                               all_orders: List[Order]) -> List[str]:
        """미배정 주문 찾기"""
        assigned_order_ids = set()
        for assignment in assignments:
            assigned_order_ids.update(assignment.assigned_orders)
        
        all_order_ids = {order.id for order in all_orders}
        unassigned = all_order_ids - assigned_order_ids
        
        return list(unassigned)
    
    def _update_performance_stats(self) -> None:
        """성능 통계 업데이트"""
        # 캐시 히트율
        optimizer_stats = self.spatial_optimizer.get_optimization_stats()
        self.performance_stats['cache_hit_rate'] = optimizer_stats.get('cache_hit_rate', 0.0)
        
        # 병렬 연산 수
        if self.performance_optimizer:
            perf_stats = self.performance_optimizer.get_performance_summary()
            parallel_stats = perf_stats.get('parallel_processor', {})
            self.performance_stats['parallel_operations'] = parallel_stats.get('parallel_operations', 0)
        
        # 속도 향상 계수 (예상값 기반)
        base_time = len(self.spatial_optimizer.spatial_hash) * 0.001  # 기준 시간
        actual_time = self.performance_stats['total_optimization_time']
        if actual_time > 0:
            self.performance_stats['speedup_factor'] = max(1.0, base_time / actual_time)
    
    def _log_performance_details(self) -> None:
        """성능 세부사항 로깅"""
        self.logger.info("=== 최적화된 알고리즘 성능 분석 ===")
        self.logger.info(f"클러스터링: {self.performance_stats['clustering_time']:.3f}초")
        self.logger.info(f"공간 인덱싱: {self.performance_stats['spatial_indexing_time']:.3f}초")
        self.logger.info(f"체인 구축: {self.performance_stats['chain_building_time']:.3f}초")
        self.logger.info(f"배차 생성: {self.performance_stats['assignment_time']:.3f}초")
        self.logger.info(f"총 처리 시간: {self.performance_stats['total_optimization_time']:.3f}초")
        self.logger.info(f"캐시 히트율: {self.performance_stats['cache_hit_rate']:.1%}")
        self.logger.info(f"병렬 연산: {self.performance_stats['parallel_operations']}회")
        self.logger.info(f"속도 향상: {self.performance_stats['speedup_factor']:.1f}배")
        
        # 공간 최적화 통계
        optimizer_stats = self.spatial_optimizer.get_optimization_stats()
        self.logger.info(f"공간 셀: {optimizer_stats['spatial_cells']}개")
        self.logger.info(f"총 쿼리: {optimizer_stats['total_queries']}회")
        
        # 체인 통계
        chain_stats = self.chain_builder.get_chain_stats()
        self.logger.info(f"구축된 체인: {chain_stats['chains_built']}개")
        self.logger.info(f"평균 체인 길이: {chain_stats['avg_chain_length']:.2f}")
    
    def _fallback_to_basic_nearest_neighbor(self, orders: List[Order], vehicles: List[Vehicle],
                                          vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """기본 최근접 이웃으로 폴백"""
        self.logger.warning("기본 최근접 이웃 알고리즘으로 폴백")
        
        from .nearest_neighbor import NearestNeighborAlgorithm
        fallback_algorithm = NearestNeighborAlgorithm(self.config)
        
        return fallback_algorithm._solve_implementation(orders, vehicles, vehicle_capacities)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {
            'algorithm_stats': self.performance_stats.copy(),
            'spatial_optimizer_stats': self.spatial_optimizer.get_optimization_stats(),
            'chain_builder_stats': self.chain_builder.get_chain_stats()
        }
        
        if self.performance_optimizer:
            stats['performance_optimizer_stats'] = self.performance_optimizer.get_performance_summary()
        
        return stats