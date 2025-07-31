from typing import List, Tuple, Optional
from src.model.delivery_point import DeliveryPoint
from src.model.distance import distance
from src.core.logger import setup_logger
from src.core.distance_matrix import DistanceMatrix
import numpy as np
from datetime import datetime
import random
from concurrent.futures import ProcessPoolExecutor
import math

logger = setup_logger('tsp')
distance_matrix = DistanceMatrix()

class TSPSolver:
    """최적화된 Lin-Kernighan 휴리스틱 기반 TSP Solver"""
    
    def __init__(self, points: List[DeliveryPoint]):
        self.points = points
        self.distances = distance_matrix.compute_matrix(points)
        self.size = len(points)
    
    def solve(self) -> Tuple[List[DeliveryPoint], float]:
        """TSP 문제 해결 - 병렬 처리 적용"""
        try:
            if not self.points:
                logger.error("입력된 배송지점이 없습니다.")
                return [], 0.0

            if len(self.points) == 1:
                logger.info("배송지점이 1개뿐입니다.")
                return self.points, 0.0

            logger.info(f"TSP 해결 시작: {len(self.points)}개 지점")
            
            # 병렬 처리를 위한 시작점 선택
            start_points = self._select_diverse_starts(min(4, self.size))
            
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                for start in start_points:
                    futures.append(
                        executor.submit(self._solve_from_start, start)
                    )
                
                # 가장 좋은 결과 선택
                best_route = None
                best_distance = float('inf')
                for future in futures:
                    route, distance = future.result()
                    if distance < best_distance:
                        best_route = route
                        best_distance = distance
                        logger.debug(f"새로운 최적 경로 발견: {best_distance:.2f}km")
            
            logger.info(f"TSP 해결 완료: 총 거리 = {best_distance:.2f}km")
            return [self.points[i] for i in best_route], best_distance
            
        except Exception as e:
            logger.error(f"TSP 해결 중 오류 발생: {str(e)}")
            raise

    def _solve_from_start(self, start: int) -> Tuple[List[int], float]:
        """단일 시작점에서의 해 탐색"""
        current_route = self._nearest_neighbor(start)
        current_distance = self._calculate_total_distance(current_route)
        
        improved_route, improved_distance = self._lin_kernighan(
            current_route,
            current_distance
        )
        
        return improved_route, improved_distance

    def _select_diverse_starts(self, num_starts: int) -> List[int]:
        """다양한 시작점 선택 - 우선순위와 거리 고려"""
        if num_starts >= self.size:
            return list(range(self.size))
            
        starts = [0]  # depot는 항상 포함
        while len(starts) < num_starts:
            max_score = -1
            best_point = None
            
            for i in range(self.size):
                if i in starts:
                    continue
                    
                # 거리와 우선순위를 모두 고려한 점수 계산
                min_dist = min(self.distances[i][j] for j in starts)
                priority_weight = self.points[i].get_priority_weight()
                score = min_dist * priority_weight
                
                if score > max_score:
                    max_score = score
                    best_point = i
                    
            starts.append(best_point)
            
        return starts

    def _nearest_neighbor(self, start: int = 0) -> List[int]:
        """최적화된 Nearest Neighbor 알고리즘"""
        unvisited = set(range(self.size))
        unvisited.remove(start)
        route = [start]
        current = start

        # NumPy 배열 사용으로 속도 향상
        distances = np.array(self.distances)
        priorities = np.array([p.get_priority_weight() for p in self.points])
        
        while unvisited:
            # 벡터화된 연산으로 다음 지점 선택
            next_distances = distances[current][list(unvisited)]
            next_priorities = priorities[list(unvisited)]
            scores = next_distances / next_priorities
            
            next_idx = np.argmin(scores)
            next_point = list(unvisited)[next_idx]
            
            route.append(next_point)
            unvisited.remove(next_point)
            current = next_point

        return route

    def _calculate_total_distance(self, route: List[int]) -> float:
        """경로의 총 거리 계산"""
        return sum(self.distances[route[i]][route[i + 1]]
                  for i in range(len(route) - 1)) + self.distances[route[-1]][route[0]]

    def _lin_kernighan(
        self,
        initial_route: List[int],
        initial_distance: float,
        max_iterations: int = 150,  # 반복 횟수 증가
        max_no_improve: int = 30,   # 개선 없을 때 허용 횟수 증가
        temperature: float = 100.0  # Simulated Annealing 파라미터 추가
    ) -> Tuple[List[int], float]:
        """개선된 Lin-Kernighan 알고리즘 - Simulated Annealing 결합"""
        best_route = initial_route.copy()
        best_distance = initial_distance
        current_route = initial_route.copy()
        current_distance = initial_distance
        no_improve_count = 0
        
        # 작은 경로는 간단한 최적화만 수행
        if len(best_route) <= 3:
            logger.info(f"작은 경로({len(best_route)}개)는 기본 최적화로 처리")
            return best_route, best_distance
        
        # 이웃 탐색 범위를 제한
        max_segment_size = min(20, len(best_route) // 4)
        if max_segment_size < 3:
            max_segment_size = len(best_route) - 1
        
        for iteration in range(max_iterations):
            if no_improve_count >= max_no_improve:
                break
            
            # 현재 온도 계산 (점진적으로 감소)
            temp = temperature / (1 + iteration)
            improved = False
            
            # 안전한 범위에서 구간 선택
            if len(best_route) < 4:
                break  # 너무 작은 경로는 더 이상 최적화하지 않음
                
            max_start = len(best_route) - max_segment_size - 1
            if max_start < 1:
                max_start = 1
                
            # 무작위로 구간 선택하여 최적화
            i = random.randint(1, max_start)
            segment_size = min(max_segment_size, len(best_route) - i - 1)
            if segment_size < 2:
                segment_size = 2
            
            # 3-opt 이동 시도
            for j in range(i + 2, min(i + segment_size - 2, len(best_route) - 1)):
                if improved:
                    break
                for k in range(j + 2, min(i + segment_size, len(best_route))):
                    # 새로운 경로 생성
                    new_route = (
                        current_route[:i] +
                        current_route[i:j][::-1] +
                        current_route[j:k][::-1] +
                        current_route[k:]
                    )
                    new_distance = self._calculate_total_distance(new_route)
                    
                    # Simulated Annealing 기반 해 수용
                    delta = new_distance - current_distance
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        current_route = new_route
                        current_distance = new_distance
                        improved = True
                        
                        # 전역 최적해 업데이트
                        if current_distance < best_distance:
                            best_route = current_route.copy()
                            best_distance = current_distance
                            no_improve_count = 0
                            logger.debug(
                                f"반복 {iteration}: 개선된 거리 = {best_distance:.2f}km"
                            )
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count % 3 == 0:  # 더 자주 교란 적용
                    current_route = self._perturb_route(current_route)
                    current_distance = self._calculate_total_distance(current_route)
                    
                    # 교란 후에도 전역 최적해 검사
                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance
                        no_improve_count = 0
            
            # 주기적으로 지역 최적해에서 재시작
            if iteration % 20 == 0:
                current_route = best_route.copy()
                current_distance = best_distance
        
        return best_route, best_distance

    def _perturb_route(self, route: List[int]) -> List[int]:
        """개선된 경로 교란 방법"""
        perturbed = route.copy()
        
        # Double-Bridge 이동 (4-opt)
        if len(perturbed) >= 8:
            pos = sorted(random.sample(range(1, len(perturbed) - 1), 4))
            p1, p2, p3, p4 = pos
            perturbed = (
                perturbed[:p1] +
                perturbed[p3:p4] +
                perturbed[p2:p3] +
                perturbed[p1:p2] +
                perturbed[p4:]
            )
        
        return perturbed

def solve_tsp(points: List[DeliveryPoint]) -> Tuple[List[DeliveryPoint], float]:
    """TSP 해결을 위한 편의 함수 - 소규모 클러스터 안전 처리"""
    try:
        if not points:
            logger.warning("⚠️ 빈 배송지 리스트가 입력되었습니다.")
            return [], 0.0
        
        if len(points) == 1:
            logger.info("📍 배송지가 1개뿐입니다. TSP 최적화 건너뜀.")
            return points, 0.0
        
        if len(points) == 2:
            logger.info("📍 배송지가 2개입니다. 간단한 경로 계산.")
            dist = distance(points[0].latitude, points[0].longitude, 
                          points[1].latitude, points[1].longitude)
            return points, dist
            
        # 3개 이상일 때만 복잡한 TSP 알고리즘 적용
        logger.info(f"🔍 TSP 최적화 시작: {len(points)}개 배송지")
        solver = TSPSolver(points)
        optimized_points, total_distance = solver.solve()
        
        logger.info(f"✅ TSP 최적화 완료: {total_distance:.2f}km")
        return optimized_points, total_distance
        
    except Exception as e:
        logger.error(f"❌ TSP 해결 중 오류 발생: {str(e)}")
        logger.warning("⚠️ 기본 순서로 대체 처리합니다.")
        
        # 오류 발생 시 기본 순서 반환 (완전 실패 방지)
        if points:
            total_dist = 0.0
            for i in range(len(points) - 1):
                total_dist += distance(points[i].latitude, points[i].longitude,
                                     points[i + 1].latitude, points[i + 1].longitude)
            return points, total_dist
        else:
            return [], 0.0