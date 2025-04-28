from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from ..models.delivery_point import DeliveryPoint
from ..models.vehicle import Vehicle
from ..utils.time_utils import calculate_travel_time
from ..utils.geo_utils import calculate_distance

class TimeOptimizer:
    """배송 시간을 최적화하는 클래스"""
    
    def __init__(self):
        self.traffic_patterns = self._load_traffic_patterns()
        self.service_time_patterns = self._load_service_time_patterns()
        
    def optimize(
        self,
        points: List[DeliveryPoint],
        vehicle: Vehicle,
        start_time: datetime,
        end_time: datetime,
        **kwargs
    ) -> List[DeliveryPoint]:
        """
        배송 순서를 시간 효율적으로 최적화

        Args:
            points: 배송지점 목록
            vehicle: 배송 차량
            start_time: 시작 시간
            end_time: 종료 시간
            **kwargs: 추가 최적화 파라미터

        Returns:
            최적화된 순서의 배송지점 목록
        """
        if not points:
            return []

        # 시간 제약 조건 설정
        time_windows = self._get_time_windows(points, start_time, end_time)
        
        # 시간 행렬 계산
        time_matrix = self._calculate_time_matrix(
            points,
            vehicle,
            start_time,
            kwargs.get('consider_traffic', True)
        )
        
        # 초기 해 생성
        initial_solution = self._create_initial_solution(
            points,
            time_matrix,
            time_windows
        )
        
        # 지역 탐색을 통한 최적화
        optimized_sequence = self._local_search(
            initial_solution,
            time_matrix,
            time_windows,
            max_iterations=kwargs.get('max_iterations', 1000)
        )
        
        return optimized_sequence

    def _load_traffic_patterns(self) -> Dict[str, float]:
        """시간대별 교통 패턴 로드"""
        # 실제로는 DB나 파일에서 로드
        return {
            '0000': 0.8,  # 자정
            '0600': 1.2,  # 출근 시간
            '1000': 1.0,  # 오전
            '1200': 1.1,  # 점심
            '1700': 1.3,  # 퇴근 시간
            '2000': 0.9,  # 저녁
            '2200': 0.8   # 심야
        }

    def _load_service_time_patterns(self) -> Dict[str, float]:
        """시간대별 서비스 시간 패턴 로드"""
        return {
            '0000': 1.0,  # 기본
            '0600': 1.2,  # 출근 시간
            '1000': 1.0,  # 오전
            '1200': 1.1,  # 점심
            '1700': 1.2,  # 퇴근 시간
            '2000': 1.0,  # 저녁
            '2200': 1.0   # 심야
        }

    def _get_time_windows(
        self,
        points: List[DeliveryPoint],
        global_start: datetime,
        global_end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """각 배송지점의 유효 배송 시간대 계산"""
        time_windows = []
        for point in points:
            if point.time_window and all(point.time_window):
                # 배송지점 자체 시간대와 전역 시간대의 교집합
                start = max(point.time_window[0], global_start)
                end = min(point.time_window[1], global_end)
            else:
                # 전역 시간대 사용
                start = global_start
                end = global_end
            
            time_windows.append((start, end))
        
        return time_windows

    def _calculate_time_matrix(
        self,
        points: List[DeliveryPoint],
        vehicle: Vehicle,
        start_time: datetime,
        consider_traffic: bool = True
    ) -> np.ndarray:
        """지점 간 이동 시간 행렬 계산"""
        size = len(points)
        time_matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                if i != j:
                    distance = calculate_distance(
                        (points[i].latitude, points[i].longitude),
                        (points[j].latitude, points[j].longitude)
                    )
                    
                    # 기본 이동 시간 계산
                    base_time = calculate_travel_time(
                        distance,
                        vehicle.type
                    )
                    
                    if consider_traffic:
                        # 시간대별 교통 가중치 적용
                        traffic_factor = self._get_traffic_factor(start_time)
                        time_matrix[i][j] = base_time * traffic_factor
                    else:
                        time_matrix[i][j] = base_time
        
        return time_matrix

    def _get_traffic_factor(self, time: datetime) -> float:
        """해당 시간대의 교통 가중치 반환"""
        time_str = time.strftime('%H%M')
        
        # 가장 가까운 시간대의 가중치 반환
        closest_time = min(
            self.traffic_patterns.keys(),
            key=lambda x: abs(int(x) - int(time_str))
        )
        
        return self.traffic_patterns[closest_time]

    def _create_initial_solution(
        self,
        points: List[DeliveryPoint],
        time_matrix: np.ndarray,
        time_windows: List[Tuple[datetime, datetime]]
    ) -> List[DeliveryPoint]:
        """초기 해 생성 (Nearest Neighbor 알고리즘 사용)"""
        if not points:
            return []
            
        unvisited = list(range(len(points)))
        current = 0  # 첫 번째 포인트에서 시작
        solution = [current]
        unvisited.remove(current)
        
        while unvisited:
            # 현재 위치에서 가장 가까운 다음 포인트 선택
            next_point = min(
                unvisited,
                key=lambda x: time_matrix[current][x]
            )
            
            solution.append(next_point)
            unvisited.remove(next_point)
            current = next_point
        
        return [points[i] for i in solution]

    def _local_search(
        self,
        initial_solution: List[DeliveryPoint],
        time_matrix: np.ndarray,
        time_windows: List[Tuple[datetime, datetime]],
        max_iterations: int = 1000
    ) -> List[DeliveryPoint]:
        """지역 탐색을 통한 해 개선"""
        current_solution = initial_solution.copy()
        current_cost = self._calculate_solution_cost(
            current_solution,
            time_matrix,
            time_windows
        )
        
        iteration = 0
        while iteration < max_iterations:
            # 2-opt 이웃해 생성
            improved = False
            for i in range(1, len(current_solution) - 2):
                for j in range(i + 1, len(current_solution)):
                    new_solution = self._two_opt_swap(
                        current_solution,
                        i,
                        j
                    )
                    
                    new_cost = self._calculate_solution_cost(
                        new_solution,
                        time_matrix,
                        time_windows
                    )
                    
                    if new_cost < current_cost:
                        current_solution = new_solution
                        current_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
                
            iteration += 1
        
        return current_solution

    def _two_opt_swap(
        self,
        solution: List[DeliveryPoint],
        i: int,
        j: int
    ) -> List[DeliveryPoint]:
        """2-opt 교환을 통한 새로운 해 생성"""
        new_solution = solution[:i]
        new_solution.extend(reversed(solution[i:j + 1]))
        new_solution.extend(solution[j + 1:])
        return new_solution

    def _calculate_solution_cost(
        self,
        solution: List[DeliveryPoint],
        time_matrix: np.ndarray,
        time_windows: List[Tuple[datetime, datetime]]
    ) -> float:
        """해의 총 비용 계산"""
        total_time = 0
        penalty = 0
        
        for i in range(len(solution) - 1):
            current_idx = solution[i].id
            next_idx = solution[i + 1].id
            
            # 이동 시간
            travel_time = time_matrix[current_idx][next_idx]
            total_time += travel_time
            
            # 시간 제약 위반 패널티
            arrival_time = time_windows[current_idx][0] + timedelta(minutes=total_time)
            if arrival_time > time_windows[next_idx][1]:
                penalty += (arrival_time - time_windows[next_idx][1]).total_seconds() / 60
        
        return total_time + penalty * 1000  # 패널티에 큰 가중치 부여