"""
SequentialChainBuilder - 순차적 최적 연결 알고리즘
- 주문에서 가장 가까운 주문끼리 순차 연결
- 동적 계획법(Dynamic Programming) 활용
- Union-Find 구조로 연결 관리
- 병렬 체인 구축
"""
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import Order, Vehicle, Coordinates
from .spatial_hash_optimizer import SpatialHashOptimizer


@dataclass
class OrderChain:
    """주문 체인"""
    orders: List[str] = field(default_factory=list)
    total_distance: float = 0.0
    start_coord: Optional[Coordinates] = None
    end_coord: Optional[Coordinates] = None
    chain_id: str = ""
    
    def add_order(self, order_id: str, order_coord: Coordinates, distance_to_add: float):
        """체인에 주문 추가"""
        self.orders.append(order_id)
        self.total_distance += distance_to_add
        
        if self.start_coord is None:
            self.start_coord = order_coord
        self.end_coord = order_coord
    
    def get_efficiency(self) -> float:
        """체인 효율성 계산 (주문수/총거리)"""
        return len(self.orders) / self.total_distance if self.total_distance > 0 else 0.0


@dataclass
class ChainConnection:
    """체인 간 연결 정보"""
    from_chain_id: str
    to_chain_id: str
    connection_distance: float
    connection_point: str  # 연결점이 되는 주문 ID
    

class UnionFind:
    """Union-Find 자료구조로 체인 연결 관리"""
    
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.size: Dict[str, int] = {}
    
    def make_set(self, x: str) -> None:
        """새로운 집합 생성"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
    
    def find(self, x: str) -> str:
        """루트 찾기 (경로 압축 적용)"""
        if x not in self.parent:
            self.make_set(x)
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """두 집합 합치기"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # 이미 같은 집합
        
        # 랭크 기반 합치기
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        return True
    
    def get_component_size(self, x: str) -> int:
        """컴포넌트 크기 반환"""
        root = self.find(x)
        return self.size[root]
    
    def get_components(self) -> Dict[str, List[str]]:
        """모든 컴포넌트 반환"""
        components = defaultdict(list)
        for node in self.parent.keys():
            root = self.find(node)
            components[root].append(node)
        return dict(components)


class DynamicProgrammingOptimizer:
    """동적 계획법 기반 체인 최적화"""
    
    def __init__(self):
        self.memo: Dict[Tuple, float] = {}
        self.path_memo: Dict[Tuple, List[str]] = {}
    
    def find_optimal_chain(self, orders: List[Order], start_order: Order, 
                          spatial_optimizer: SpatialHashOptimizer, max_chain_length: int = 10) -> OrderChain:
        """동적 계획법으로 최적 체인 찾기"""
        if not orders or max_chain_length <= 1:
            return OrderChain(orders=[start_order.id], chain_id=f"chain_{start_order.id}")
        
        # 상태: (현재_주문_인덱스, 방문한_주문들의_비트마스크)
        n = min(len(orders), max_chain_length)  # 계산 복잡도 제한
        order_indices = {order.id: i for i, order in enumerate(orders[:n])}
        
        if start_order.id not in order_indices:
            return OrderChain(orders=[start_order.id], chain_id=f"chain_{start_order.id}")
        
        start_idx = order_indices[start_order.id]
        
        # DP로 최적 경로 찾기
        best_distance, best_path = self._dp_solve(orders[:n], start_idx, 1 << start_idx, spatial_optimizer)
        
        # OrderChain 생성
        chain = OrderChain(chain_id=f"chain_{start_order.id}")
        current_coord = start_order.coordinates
        
        for order_id in best_path:
            order = next(o for o in orders if o.id == order_id)
            distance = spatial_optimizer._get_cached_distance(current_coord, order.coordinates)
            chain.add_order(order_id, order.coordinates, distance)
            current_coord = order.coordinates
        
        return chain
    
    def _dp_solve(self, orders: List[Order], current: int, visited: int, 
                 spatial_optimizer: SpatialHashOptimizer) -> Tuple[float, List[str]]:
        """동적 계획법 재귀 해결"""
        n = len(orders)
        
        # 메모이제이션 확인
        state = (current, visited)
        if state in self.memo:
            return self.memo[state], self.path_memo[state]
        
        # 모든 주문을 방문했거나 더 이상 갈 곳이 없으면 종료
        if visited == (1 << n) - 1:
            self.memo[state] = 0.0
            self.path_memo[state] = [orders[current].id]
            return 0.0, [orders[current].id]
        
        min_cost = float('inf')
        best_path = [orders[current].id]
        
        # 다음 방문할 주문 선택
        for next_order in range(n):
            if visited & (1 << next_order) == 0:  # 아직 방문하지 않음
                distance = spatial_optimizer._get_cached_distance(
                    orders[current].coordinates, orders[next_order].coordinates
                )
                
                next_visited = visited | (1 << next_order)
                next_cost, next_path = self._dp_solve(orders, next_order, next_visited, spatial_optimizer)
                
                total_cost = distance + next_cost
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = [orders[current].id] + next_path
        
        # 메모이제이션 저장
        self.memo[state] = min_cost
        self.path_memo[state] = best_path
        
        return min_cost, best_path
    
    def clear_memo(self) -> None:
        """메모이제이션 캐시 초기화"""
        self.memo.clear()
        self.path_memo.clear()


class SequentialChainBuilder:
    """순차적 체인 구축 알고리즘"""
    
    def __init__(self, spatial_optimizer: SpatialHashOptimizer = None):
        self.spatial_optimizer = spatial_optimizer or SpatialHashOptimizer()
        self.dp_optimizer = DynamicProgrammingOptimizer()
        self.union_find = UnionFind()
        self.logger = logging.getLogger(__name__)
        
        # 성능 설정
        self.max_chain_length = 8  # DP 복잡도 제한
        self.parallel_threshold = 50  # 병렬 처리 임계값
        self.max_workers = 4  # 최대 워커 수
        
        # 통계
        self.stats = {
            'chains_built': 0,
            'avg_chain_length': 0.0,
            'total_distance_saved': 0.0,
            'parallel_operations': 0
        }
    
    def build_optimal_chains(self, orders: List[Order], cluster_assignments: Dict[str, List[Order]]) -> Dict[str, List[OrderChain]]:
        """클러스터별로 최적 체인 구축"""
        self.logger.info(f"순차 체인 구축 시작: {len(orders)}개 주문, {len(cluster_assignments)}개 클러스터")
        
        # 공간 인덱스 구축
        self.spatial_optimizer.build_spatial_index(orders)
        
        # 클러스터별로 병렬 처리
        cluster_chains = {}
        
        if len(cluster_assignments) >= self.parallel_threshold:
            # 병렬 처리
            cluster_chains = self._build_chains_parallel(cluster_assignments)
            self.stats['parallel_operations'] += 1
        else:
            # 순차 처리
            for vehicle_id, cluster_orders in cluster_assignments.items():
                cluster_chains[vehicle_id] = self._build_chains_for_cluster(cluster_orders, vehicle_id)
        
        # 통계 업데이트
        self._update_stats(cluster_chains)
        
        self.logger.info(f"체인 구축 완료: 평균 체인 길이 {self.stats['avg_chain_length']:.2f}")
        return cluster_chains
    
    def _build_chains_parallel(self, cluster_assignments: Dict[str, List[Order]]) -> Dict[str, List[OrderChain]]:
        """클러스터별 병렬 체인 구축"""
        cluster_chains = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 클러스터별로 Future 제출
            future_to_cluster = {
                executor.submit(self._build_chains_for_cluster, orders, vehicle_id): vehicle_id
                for vehicle_id, orders in cluster_assignments.items()
            }
            
            # 결과 수집
            for future in as_completed(future_to_cluster):
                vehicle_id = future_to_cluster[future]
                try:
                    chains = future.result()
                    cluster_chains[vehicle_id] = chains
                except Exception as e:
                    self.logger.error(f"클러스터 {vehicle_id} 체인 구축 실패: {str(e)}")
                    cluster_chains[vehicle_id] = []
        
        return cluster_chains
    
    def _build_chains_for_cluster(self, orders: List[Order], vehicle_id: str) -> List[OrderChain]:
        """단일 클러스터의 체인 구축"""
        if not orders:
            return []
        
        if len(orders) == 1:
            return [OrderChain(orders=[orders[0].id], chain_id=f"chain_{vehicle_id}_0")]
        
        # Union-Find 초기화
        for order in orders:
            self.union_find.make_set(order.id)
        
        # 그리디 + DP 하이브리드 방식으로 체인 구축
        visited_orders = set()
        chains = []
        chain_counter = 0
        
        for order in orders:
            if order.id in visited_orders:
                continue
            
            # 현재 주문을 시작점으로 하는 최적 체인 구축
            chain = self._build_single_chain(order, orders, visited_orders)
            if chain.orders:
                chain.chain_id = f"chain_{vehicle_id}_{chain_counter}"
                chains.append(chain)
                chain_counter += 1
                
                # 방문 표시
                visited_orders.update(chain.orders)
        
        # 체인 후처리 (연결 가능한 체인들 병합)
        merged_chains = self._merge_compatible_chains(chains, orders)
        
        return merged_chains
    
    def _build_single_chain(self, start_order: Order, all_orders: List[Order], 
                           visited_orders: Set[str]) -> OrderChain:
        """단일 체인 구축 (그리디 + DP)"""
        available_orders = [o for o in all_orders if o.id not in visited_orders]
        
        if len(available_orders) <= self.max_chain_length:
            # DP 사용 (소규모)
            return self.dp_optimizer.find_optimal_chain(
                available_orders, start_order, self.spatial_optimizer, self.max_chain_length
            )
        else:
            # 그리디 사용 (대규모)
            return self._build_greedy_chain(start_order, available_orders)
    
    def _build_greedy_chain(self, start_order: Order, available_orders: List[Order]) -> OrderChain:
        """그리디 방식으로 체인 구축"""
        chain = OrderChain(chain_id=f"greedy_{start_order.id}")
        current_order = start_order
        remaining_orders = [o for o in available_orders if o.id != start_order.id]
        
        # 첫 번째 주문 추가
        chain.add_order(start_order.id, start_order.coordinates, 0.0)
        
        # 그리디하게 가장 가까운 주문들 연결
        while remaining_orders and len(chain.orders) < self.max_chain_length:
            nearest_order = self.spatial_optimizer.find_nearest_order_fast(
                current_order.coordinates,
                exclude_orders={o.id for o in remaining_orders if o.id not in {ro.id for ro in remaining_orders}}
            )
            
            if nearest_order and nearest_order.id in {o.id for o in remaining_orders}:
                distance = self.spatial_optimizer._get_cached_distance(
                    current_order.coordinates, nearest_order.coordinates
                )
                chain.add_order(nearest_order.id, nearest_order.coordinates, distance)
                current_order = nearest_order
                remaining_orders = [o for o in remaining_orders if o.id != nearest_order.id]
            else:
                break
        
        return chain
    
    def _merge_compatible_chains(self, chains: List[OrderChain], all_orders: List[Order]) -> List[OrderChain]:
        """호환 가능한 체인들 병합"""
        if len(chains) <= 1:
            return chains
        
        orders_dict = {o.id: o for o in all_orders}
        merged_chains = []
        used_chains = set()
        
        # 체인 간 연결 비용 계산
        connection_costs = []
        
        for i, chain1 in enumerate(chains):
            if i in used_chains:
                continue
                
            for j, chain2 in enumerate(chains[i+1:], i+1):
                if j in used_chains:
                    continue
                
                # 체인1의 끝과 체인2의 시작 거리
                end_order1 = orders_dict[chain1.orders[-1]]
                start_order2 = orders_dict[chain2.orders[0]]
                distance = self.spatial_optimizer._get_cached_distance(
                    end_order1.coordinates, start_order2.coordinates
                )
                
                connection_costs.append((distance, i, j, 'forward'))
                
                # 체인2의 끝과 체인1의 시작 거리 (역방향)
                end_order2 = orders_dict[chain2.orders[-1]]
                start_order1 = orders_dict[chain1.orders[0]]
                distance_reverse = self.spatial_optimizer._get_cached_distance(
                    end_order2.coordinates, start_order1.coordinates
                )
                
                connection_costs.append((distance_reverse, i, j, 'reverse'))
        
        # 비용 순으로 정렬하여 가장 가까운 체인들부터 병합
        connection_costs.sort()
        
        for distance, i, j, direction in connection_costs:
            if i in used_chains or j in used_chains:
                continue
            
            # 병합 조건 확인 (총 길이 제한)
            total_orders = len(chains[i].orders) + len(chains[j].orders)
            if total_orders > self.max_chain_length * 1.5:  # 약간의 여유
                continue
            
            # 체인 병합
            if direction == 'forward':
                merged_chain = self._merge_chains_forward(chains[i], chains[j], distance)
            else:
                merged_chain = self._merge_chains_forward(chains[j], chains[i], distance)
            
            merged_chains.append(merged_chain)
            used_chains.add(i)
            used_chains.add(j)
            break  # 하나씩만 병합
        
        # 병합되지 않은 체인들 추가
        for i, chain in enumerate(chains):
            if i not in used_chains:
                merged_chains.append(chain)
        
        return merged_chains
    
    def _merge_chains_forward(self, chain1: OrderChain, chain2: OrderChain, connection_distance: float) -> OrderChain:
        """두 체인을 순방향으로 병합"""
        merged_chain = OrderChain(
            chain_id=f"merged_{chain1.chain_id}_{chain2.chain_id}",
            orders=chain1.orders + chain2.orders,
            total_distance=chain1.total_distance + connection_distance + chain2.total_distance,
            start_coord=chain1.start_coord,
            end_coord=chain2.end_coord
        )
        
        return merged_chain
    
    def build_sequential_connections(self, chains: List[OrderChain], orders_dict: Dict[str, Order]) -> List[ChainConnection]:
        """체인들 간의 순차적 연결 구축"""
        if len(chains) <= 1:
            return []
        
        connections = []
        
        # 모든 체인 쌍에 대해 최적 연결점 찾기
        for i, chain1 in enumerate(chains):
            for j, chain2 in enumerate(chains[i+1:], i+1):
                connection = self._find_best_connection(chain1, chain2, orders_dict)
                if connection:
                    connections.append(connection)
        
        # 최소 신장 트리 방식으로 최적 연결들만 선택
        optimal_connections = self._select_optimal_connections(connections, chains)
        
        return optimal_connections
    
    def _find_best_connection(self, chain1: OrderChain, chain2: OrderChain, 
                            orders_dict: Dict[str, Order]) -> Optional[ChainConnection]:
        """두 체인 간 최적 연결점 찾기"""
        min_distance = float('inf')
        best_connection = None
        
        # 체인1의 각 주문과 체인2의 각 주문 간 거리 확인
        for order_id1 in chain1.orders:
            for order_id2 in chain2.orders:
                if order_id1 in orders_dict and order_id2 in orders_dict:
                    distance = self.spatial_optimizer._get_cached_distance(
                        orders_dict[order_id1].coordinates,
                        orders_dict[order_id2].coordinates
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_connection = ChainConnection(
                            from_chain_id=chain1.chain_id,
                            to_chain_id=chain2.chain_id,
                            connection_distance=distance,
                            connection_point=f"{order_id1}->{order_id2}"
                        )
        
        return best_connection
    
    def _select_optimal_connections(self, connections: List[ChainConnection], 
                                  chains: List[OrderChain]) -> List[ChainConnection]:
        """최적 연결들 선택 (최소 신장 트리)"""
        if not connections:
            return []
        
        # 체인 ID를 인덱스로 매핑
        chain_ids = {chain.chain_id for chain in chains}
        
        # 거리순으로 정렬
        connections.sort(key=lambda c: c.connection_distance)
        
        # Union-Find로 사이클 방지하며 연결 선택
        uf = UnionFind()
        for chain_id in chain_ids:
            uf.make_set(chain_id)
        
        optimal_connections = []
        
        for connection in connections:
            if uf.union(connection.from_chain_id, connection.to_chain_id):
                optimal_connections.append(connection)
                
                # 모든 체인이 연결되었으면 종료
                if len(optimal_connections) >= len(chain_ids) - 1:
                    break
        
        return optimal_connections
    
    def _update_stats(self, cluster_chains: Dict[str, List[OrderChain]]) -> None:
        """통계 업데이트"""
        total_chains = 0
        total_chain_length = 0
        
        for chains in cluster_chains.values():
            total_chains += len(chains)
            total_chain_length += sum(len(chain.orders) for chain in chains)
        
        if total_chains > 0:
            self.stats['chains_built'] = total_chains
            self.stats['avg_chain_length'] = total_chain_length / total_chains
        
        # DP 메모이제이션 정리
        self.dp_optimizer.clear_memo()
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """체인 구축 통계 반환"""
        optimizer_stats = self.spatial_optimizer.get_optimization_stats()
        
        return {
            **self.stats,
            'spatial_optimizer': optimizer_stats,
            'max_chain_length': self.max_chain_length,
            'parallel_threshold': self.parallel_threshold
        }