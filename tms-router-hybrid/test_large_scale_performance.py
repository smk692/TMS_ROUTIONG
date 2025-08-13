#!/usr/bin/env python3
"""
TMS Router Hybrid - 대용량 데이터 성능 테스트
OR-Tools VRP 알고리즘으로 3,000개 주문 처리 성능 검증
"""

import sys
import os

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import asyncio
import time
import logging
from typing import List, Dict, Any
import json
from dataclasses import asdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

from core.database.db_manager import DatabaseManager
from core.models import Order, Vehicle, Coordinates, Priority, OrderStatus, VehicleType, VehicleStatus
from core.algorithms.ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig
from core.orchestration.dispatch_orchestrator import DispatchOrchestrator
from core.orchestration.dispatch_config import DispatchConfig


class PerformanceTestResult:
    """성능 테스트 결과"""
    
    def __init__(self):
        self.test_name: str = ""
        self.center_id: str = ""
        self.total_orders: int = 0
        self.total_vehicles: int = 0
        self.assigned_orders: int = 0
        self.used_vehicles: int = 0
        self.execution_time_seconds: float = 0.0
        self.quality_score: float = 0.0
        self.total_distance_km: float = 0.0
        self.total_time_minutes: int = 0
        self.average_capacity_utilization: float = 0.0
        self.vrp_objective_value: int = 0
        self.algorithm_used: str = ""
        self.success: bool = False
        self.error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self):
        """결과 요약 출력"""
        print(f"\n=== {self.test_name} 테스트 결과 ===")
        print(f"센터: {self.center_id}")
        print(f"처리 결과: {'성공' if self.success else '실패'}")
        if not self.success:
            print(f"오류: {self.error_message}")
            return
        
        print(f"전체 주문: {self.total_orders}개")
        print(f"배정 주문: {self.assigned_orders}개 ({self.assigned_orders/self.total_orders*100:.1f}%)")
        print(f"전체 차량: {self.total_vehicles}대")
        print(f"사용 차량: {self.used_vehicles}대 ({self.used_vehicles/self.total_vehicles*100:.1f}%)")
        print(f"실행 시간: {self.execution_time_seconds:.2f}초")
        print(f"품질 점수: {self.quality_score:.3f}")
        print(f"총 거리: {self.total_distance_km:.1f}km")
        print(f"총 시간: {self.total_time_minutes}분")
        print(f"평균 용량 활용도: {self.average_capacity_utilization:.3f}")
        print(f"VRP 목적함수 값: {self.vrp_objective_value}")
        print(f"알고리즘: {self.algorithm_used}")


class LargeScalePerformanceTest:
    """대용량 데이터 성능 테스트"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        
        # OR-Tools VRP 알고리즘 설정 - 대용량 최적화
        self.vrp_config = ORToolsVRPConfig(
            max_solve_time_seconds=300,  # 5분 제한
            use_clustering=True,
            min_cluster_size=15,
            max_cluster_size=50,
            epsilon=0.008,  # 약 800m
            max_work_hours=8,
            max_distance_km=150,
            unassigned_penalty=150000,  # 미배정 페널티 증가
            distance_weight=1.2,
            vehicle_fixed_cost=8000,
            distance_api={
                'api_priority': ['haversine'],  # 테스트용으로 Haversine만 사용
                'distance_cache_ttl': 24 * 3600,
                'max_locations_per_request': 100,
                'request_delay': 0.05
            }
        )
        
        self.ortools_algorithm = ORToolsVRPAlgorithm(self.vrp_config)
        
        # 디스패치 오케스트레이터 설정
        self.dispatch_config = DispatchConfig(
            algorithm_preference=['ortools_vrp'],
            max_execution_time_seconds=400,
            enable_parallel_processing=True,
            cache_enabled=True
        )
        
        self.orchestrator = DispatchOrchestrator(
            db_manager=self.db_manager,
            config=self.dispatch_config
        )
        
    async def load_test_data(self, center_id: str = None) -> tuple[List[Order], List[Vehicle]]:
        """테스트 데이터 로드"""
        
        try:
            # 주문 데이터 로드
            if center_id:
                # 특정 센터의 테스트 주문만 로드
                query = """
                    SELECT id, center_id, region_id, address, latitude, longitude, 
                           priority, status, created_at
                    FROM orders 
                    WHERE id LIKE 'TEST_ORD_%' 
                      AND center_id = %s 
                      AND status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                """
                params = (center_id,)
            else:
                # 모든 테스트 주문 로드
                query = """
                    SELECT id, center_id, region_id, address, latitude, longitude, 
                           priority, status, created_at
                    FROM orders 
                    WHERE id LIKE 'TEST_ORD_%' 
                      AND status = 'pending'
                    ORDER BY center_id, priority DESC, created_at ASC
                """
                params = ()
            
            order_rows = await self.db_manager.fetch_all(query, params)
            
            orders = []
            for row in order_rows:
                order = Order(
                    id=row['id'],
                    center_id=row['center_id'],
                    region_id=row['region_id'],
                    coordinates=Coordinates(
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude'])
                    ),
                    address=row['address'],
                    priority=Priority(row['priority']),
                    status=OrderStatus(row['status']),
                    created_at=row['created_at']
                )
                orders.append(order)
            
            # 차량 데이터 로드
            if center_id:
                # 특정 센터의 테스트 차량만 로드
                vehicle_query = """
                    SELECT v.id, v.center_id, v.region_id, v.driver_name, v.vehicle_type,
                           v.experience_months, v.max_capacity, v.safe_capacity, v.status,
                           r.center_latitude, r.center_longitude
                    FROM vehicles v
                    JOIN regions r ON v.region_id = r.id
                    WHERE v.id LIKE 'TEST_VEH_%' 
                      AND v.center_id = %s
                      AND v.status = 'ACTIVE'
                    ORDER BY v.experience_months DESC
                """
                vehicle_params = (center_id,)
            else:
                # 모든 테스트 차량 로드
                vehicle_query = """
                    SELECT v.id, v.center_id, v.region_id, v.driver_name, v.vehicle_type,
                           v.experience_months, v.max_capacity, v.safe_capacity, v.status,
                           r.center_latitude, r.center_longitude
                    FROM vehicles v
                    JOIN regions r ON v.region_id = r.id
                    WHERE v.id LIKE 'TEST_VEH_%'
                      AND v.status = 'ACTIVE'
                    ORDER BY v.center_id, v.experience_months DESC
                """
                vehicle_params = ()
            
            vehicle_rows = await self.db_manager.fetch_all(vehicle_query, vehicle_params)
            
            vehicles = []
            for row in vehicle_rows:
                vehicle = Vehicle(
                    id=row['id'],
                    center_id=row['center_id'],
                    region_id=row['region_id'],
                    driver_name=row['driver_name'],
                    vehicle_type=VehicleType(row['vehicle_type']),
                    experience_months=row['experience_months'],
                    max_capacity=row['max_capacity'],
                    safe_capacity=row['safe_capacity'],
                    status=VehicleStatus(row['status']),
                    center_coordinates=Coordinates(
                        latitude=float(row['center_latitude']),
                        longitude=float(row['center_longitude'])
                    )
                )
                vehicles.append(vehicle)
            
            self.logger.info(f"테스트 데이터 로드 완료: 주문 {len(orders)}개, 차량 {len(vehicles)}대")
            if center_id:
                self.logger.info(f"센터: {center_id}")
            
            return orders, vehicles
            
        except Exception as e:
            self.logger.error(f"테스트 데이터 로드 오류: {str(e)}")
            raise

    async def test_single_center_performance(self, center_id: str) -> PerformanceTestResult:
        """단일 센터 성능 테스트 (500개 주문)"""
        
        result = PerformanceTestResult()
        result.test_name = f"단일 센터 성능 테스트"
        result.center_id = center_id
        
        try:
            self.logger.info(f"\n=== {center_id} 센터 성능 테스트 시작 ===")
            
            # 테스트 데이터 로드
            orders, vehicles = await self.load_test_data(center_id)
            
            result.total_orders = len(orders)
            result.total_vehicles = len(vehicles)
            
            if not orders:
                result.error_message = "테스트할 주문이 없습니다"
                return result
            
            if not vehicles:
                result.error_message = "사용 가능한 차량이 없습니다"
                return result
            
            # OR-Tools VRP 알고리즘 직접 실행
            start_time = time.time()
            
            algorithm_result = await self.ortools_algorithm.optimize_async(
                orders=orders,
                vehicles=vehicles,
                regions=[],
                conditions=None
            )
            
            end_time = time.time()
            
            # 결과 기록
            result.execution_time_seconds = end_time - start_time
            result.assigned_orders = len([order_id for assignment in algorithm_result.assignments 
                                         for order_id in assignment.assigned_orders])
            result.used_vehicles = len(algorithm_result.assignments)
            result.quality_score = algorithm_result.quality_score
            result.algorithm_used = algorithm_result.algorithm_name
            
            # 수렴 정보에서 상세 데이터 추출
            convergence_info = algorithm_result.convergence_info
            result.total_distance_km = convergence_info.get('total_distance', 0.0)
            result.total_time_minutes = convergence_info.get('total_time', 0)
            result.average_capacity_utilization = convergence_info.get('average_capacity_utilization', 0.0)
            result.vrp_objective_value = convergence_info.get('vrp_objective_value', 0)
            
            result.success = True
            
            self.logger.info(f"{center_id} 센터 테스트 완료: "
                           f"{result.assigned_orders}/{result.total_orders}개 배정 "
                           f"({result.execution_time_seconds:.2f}초)")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"{center_id} 센터 테스트 오류: {str(e)}")
        
        return result

    async def test_all_centers_performance(self) -> List[PerformanceTestResult]:
        """전체 센터 성능 테스트"""
        
        centers = ['GANGNAM', 'SONGPA', 'MAPO', 'YONGSAN', 'JONGRO', 'SEOCHO']
        results = []
        
        self.logger.info(f"\n=== 전체 센터 성능 테스트 시작 ({len(centers)}개 센터) ===")
        
        total_start_time = time.time()
        
        for center_id in centers:
            result = await self.test_single_center_performance(center_id)
            results.append(result)
            
            # 센터별 결과 즉시 출력
            result.print_summary()
        
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        
        # 전체 결과 요약
        self.print_overall_summary(results, total_execution_time)
        
        return results

    def print_overall_summary(self, results: List[PerformanceTestResult], total_time: float):
        """전체 결과 요약 출력"""
        
        print(f"\n" + "="*70)
        print(f"전체 성능 테스트 결과 요약")
        print(f"="*70)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            total_orders = sum(r.total_orders for r in successful_results)
            total_assigned = sum(r.assigned_orders for r in successful_results)
            total_vehicles = sum(r.total_vehicles for r in successful_results)
            total_used_vehicles = sum(r.used_vehicles for r in successful_results)
            total_distance = sum(r.total_distance_km for r in successful_results)
            total_minutes = sum(r.total_time_minutes for r in successful_results)
            avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
            avg_capacity_util = sum(r.average_capacity_utilization for r in successful_results) / len(successful_results)
            total_vrp_objective = sum(r.vrp_objective_value for r in successful_results)
            
            print(f"성공한 센터: {len(successful_results)}/{len(results)}개")
            print(f"전체 처리 시간: {total_time:.2f}초")
            print(f"전체 주문: {total_orders}개")
            print(f"전체 배정: {total_assigned}개 ({total_assigned/total_orders*100:.1f}%)")
            print(f"전체 차량: {total_vehicles}대")
            print(f"사용된 차량: {total_used_vehicles}대 ({total_used_vehicles/total_vehicles*100:.1f}%)")
            print(f"총 이동 거리: {total_distance:.1f}km")
            print(f"총 소요 시간: {total_minutes}분 ({total_minutes/60:.1f}시간)")
            print(f"평균 품질 점수: {avg_quality:.3f}")
            print(f"평균 용량 활용도: {avg_capacity_util:.3f}")
            print(f"총 VRP 목적함수 값: {total_vrp_objective}")
            
            print(f"\n센터별 성능:")
            for result in successful_results:
                assignment_rate = result.assigned_orders / result.total_orders * 100
                vehicle_usage_rate = result.used_vehicles / result.total_vehicles * 100
                print(f"  {result.center_id}: "
                      f"{result.assigned_orders}/{result.total_orders}개 배정({assignment_rate:.1f}%), "
                      f"{result.used_vehicles}/{result.total_vehicles}대 사용({vehicle_usage_rate:.1f}%), "
                      f"{result.execution_time_seconds:.1f}초, "
                      f"품질 {result.quality_score:.3f}")
        
        if failed_results:
            print(f"\n실패한 센터:")
            for result in failed_results:
                print(f"  {result.center_id}: {result.error_message}")
        
        print(f"\n결론:")
        if len(successful_results) == len(results):
            print(f"✅ 모든 센터에서 성공적으로 처리 완료!")
            print(f"✅ OR-Tools VRP 알고리즘이 3,000개 주문을 {total_time:.1f}초에 처리")
            print(f"✅ 평균 배정률 {total_assigned/total_orders*100:.1f}%, 품질 점수 {avg_quality:.3f}")
        else:
            print(f"⚠️  일부 센터에서 실패 발생: {len(failed_results)}개 센터")
        
        print(f"="*70)

    async def save_results_to_file(self, results: List[PerformanceTestResult], filename: str = "performance_test_results.json"):
        """결과를 JSON 파일로 저장"""
        
        try:
            results_data = {
                'test_timestamp': time.time(),
                'test_name': '대용량 OR-Tools VRP 성능 테스트',
                'total_centers': len(results),
                'successful_centers': len([r for r in results if r.success]),
                'results': [result.to_dict() for result in results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"테스트 결과를 {filename}에 저장했습니다")
            
        except Exception as e:
            self.logger.error(f"결과 저장 오류: {str(e)}")


async def main():
    """메인 실행 함수"""
    
    tester = LargeScalePerformanceTest()
    
    try:
        print("=== TMS Router Hybrid - OR-Tools VRP 대용량 성능 테스트 ===\n")
        
        # 1. 강남 센터 단일 테스트 (500개 주문)
        print("1단계: 강남 센터 단일 성능 테스트 시작...")
        gangnam_result = await tester.test_single_center_performance('GANGNAM')
        gangnam_result.print_summary()
        
        if gangnam_result.success:
            print(f"\n✅ 강남 센터 테스트 성공! 다음 단계로 진행합니다.")
            
            # 2. 전체 센터 성능 테스트 (3,000개 주문)
            print(f"\n2단계: 전체 센터 성능 테스트 시작...")
            all_results = await tester.test_all_centers_performance()
            
            # 3. 결과 저장
            await tester.save_results_to_file(all_results)
            
        else:
            print(f"\n❌ 강남 센터 테스트 실패: {gangnam_result.error_message}")
            print(f"전체 테스트를 중단합니다.")
        
    except Exception as e:
        print(f"테스트 실행 오류: {str(e)}")
    
    finally:
        # 데이터베이스 연결 정리
        await tester.db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())