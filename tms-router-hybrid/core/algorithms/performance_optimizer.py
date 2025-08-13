"""
PerformanceOptimizer - 극한 성능 튜닝 시스템
- 멀티스레딩 병렬 처리
- SIMD 벡터화 최적화
- JIT 컴파일 가속
- 메모리 풀링 및 캐시 최적화
"""
import os
import math
import logging
import threading
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import queue
import time
import gc
from functools import lru_cache, wraps
import numpy as np

try:
    # JIT 컴파일 지원 (선택적)
    from numba import jit, vectorize, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(func):
        return func
    def vectorize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..models import Order, Vehicle, Coordinates


@dataclass
class PerformanceMetrics:
    """성능 측정 메트릭"""
    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_efficiency: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ThreadSafeCounter:
    """스레드 안전한 카운터"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        with self._lock:
            self._value = 0


class AdvancedMemoryPool:
    """고급 메모리 풀 관리자"""
    
    def __init__(self, pool_size: int = 5000, object_factory: Callable = None):
        self.pool_size = pool_size
        self.object_factory = object_factory or (lambda: {})
        self.available = queue.Queue(maxsize=pool_size)
        self.in_use = set()
        self.lock = threading.RLock()
        
        # 통계
        self.stats = {
            'created': 0,
            'reused': 0,
            'peak_usage': 0,
            'current_usage': 0
        }
        
        # 초기 풀 생성
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """초기 풀 생성"""
        for _ in range(min(100, self.pool_size)):  # 초기에는 100개만
            obj = self.object_factory()
            self.available.put(obj)
            self.stats['created'] += 1
    
    def get(self) -> Any:
        """객체 가져오기"""
        with self.lock:
            try:
                obj = self.available.get_nowait()
                self.stats['reused'] += 1
            except queue.Empty:
                obj = self.object_factory()
                self.stats['created'] += 1
            
            self.in_use.add(id(obj))
            self.stats['current_usage'] = len(self.in_use)
            self.stats['peak_usage'] = max(self.stats['peak_usage'], self.stats['current_usage'])
            
            return obj
    
    def release(self, obj: Any) -> None:
        """객체 반환"""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                self.stats['current_usage'] = len(self.in_use)
                
                # 풀이 가득 차지 않았으면 반환
                if not self.available.full():
                    # 객체 초기화
                    if hasattr(obj, 'clear'):
                        obj.clear()
                    elif isinstance(obj, (list, dict, set)):
                        obj.clear()
                    
                    self.available.put(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """메모리 풀 통계"""
        return self.stats.copy()


class VectorizedOperations:
    """벡터화된 연산 최적화"""
    
    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def haversine_batch(lat1_arr: np.ndarray, lon1_arr: np.ndarray, 
                       lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
        """배치 Haversine 거리 계산 (벡터화)"""
        # 라디안 변환
        lat1_rad = np.radians(lat1_arr)
        lon1_rad = np.radians(lon1_arr)
        lat2_rad = np.radians(lat2_arr)
        lon2_rad = np.radians(lon2_arr)
        
        # Haversine 공식
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371.0 * c  # 지구 반지름 6371km
    
    @staticmethod
    @vectorize([float64(float64, float64)], nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def euclidean_distance(dx: float, dy: float) -> float:
        """유클리드 거리 계산 (벡터화)"""
        return math.sqrt(dx*dx + dy*dy)
    
    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def find_k_nearest_vectorized(distances: np.ndarray, k: int) -> np.ndarray:
        """벡터화된 k-nearest 탐색"""
        return np.argpartition(distances, min(k, len(distances)-1))[:k]


class ParallelProcessor:
    """병렬 처리 최적화 관리자"""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, cpu_count() + 4)
        self.use_processes = use_processes
        self.logger = logging.getLogger(__name__)
        
        # 스레드 로컬 저장소
        self.local_storage = threading.local()
        
        # 성능 카운터
        self.parallel_ops = ThreadSafeCounter()
        self.sequential_ops = ThreadSafeCounter()
    
    def execute_parallel(self, func: Callable, data_chunks: List[Any], 
                        chunk_size: Optional[int] = None) -> List[Any]:
        """병렬 실행"""
        if not data_chunks:
            return []
        
        # 청크 크기 자동 조정
        if chunk_size is None:
            chunk_size = max(1, len(data_chunks) // (self.max_workers * 4))
        
        # 작은 데이터는 순차 처리
        if len(data_chunks) < self.max_workers * 2:
            self.sequential_ops.increment()
            return [func(chunk) for chunk in data_chunks]
        
        self.parallel_ops.increment()
        
        # 병렬 처리
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            
            # 청크 단위로 작업 제출
            for i in range(0, len(data_chunks), chunk_size):
                chunk = data_chunks[i:i + chunk_size]
                future = executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            # 결과 수집
            results = []
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"병렬 처리 오류: {str(e)}")
            
            return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """청크 처리"""
        return [func(item) for item in chunk]
    
    def get_stats(self) -> Dict[str, int]:
        """병렬 처리 통계"""
        return {
            'parallel_operations': self.parallel_ops.get(),
            'sequential_operations': self.sequential_ops.get(),
            'max_workers': self.max_workers
        }


class CacheOptimizer:
    """캐시 최적화 관리자"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.lock = threading.RLock()
        
        # LRU 캐시 데코레이터
        self.lru_cache = lru_cache(maxsize=max_size)
        
        # 통계
        self.hits = ThreadSafeCounter()
        self.misses = ThreadSafeCounter()
    
    def get_or_compute(self, key: str, compute_func: Callable, *args, **kwargs) -> Any:
        """캐시된 값 반환 또는 계산"""
        with self.lock:
            current_time = time.time()
            
            # 캐시 히트 확인
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # TTL 확인
                if current_time - timestamp < self.ttl:
                    self.access_times[key] = current_time
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    self.hits.increment()
                    return value
                else:
                    # 만료된 항목 제거
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    if key in self.access_counts:
                        del self.access_counts[key]
            
            # 캐시 미스 - 계산 필요
            self.misses.increment()
            value = compute_func(*args, **kwargs)
            
            # 캐시 크기 확인
            if len(self.cache) >= self.max_size:
                self._evict_entries()
            
            # 캐시에 저장
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            
            return value
    
    def _evict_entries(self) -> None:
        """캐시 항목 제거 (LFU + LRU 하이브리드)"""
        if not self.cache:
            return
        
        # 접근 빈도와 최근 접근 시간을 종합한 점수 계산
        current_time = time.time()
        scores = {}
        
        for key in self.cache.keys():
            access_count = self.access_counts.get(key, 0)
            last_access = self.access_times.get(key, 0)
            time_since_access = current_time - last_access
            
            # 점수 = 접근빈도 - 시간경과(가중치)
            score = access_count - (time_since_access / 3600.0)  # 1시간 = 1점 감점
            scores[key] = score
        
        # 점수가 낮은 항목들 제거 (하위 20%)
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        remove_count = max(1, len(sorted_items) // 5)
        
        for key, _ in sorted_items[:remove_count]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self.hits.get() + self.misses.get()
        hit_rate = self.hits.get() / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'hit_rate': hit_rate,
            'hits': self.hits.get(),
            'misses': self.misses.get(),
            'max_size': self.max_size
        }


def performance_monitor(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            end_memory = _get_memory_usage()
            
            metrics = PerformanceMetrics(
                operation_name=func.__name__,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=0.0  # CPU 사용률은 별도 측정 필요
            )
            
            # 성능 로그 (옵션)
            logger = logging.getLogger(__name__)
            logger.debug(f"Performance: {func.__name__} took {metrics.execution_time:.4f}s, "
                        f"memory: {metrics.memory_usage:.2f}MB")
    
    return wrapper


def _get_memory_usage() -> float:
    """현재 메모리 사용량 (MB)"""
    import psutil
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class PerformanceOptimizer:
    """극한 성능 최적화 통합 관리자"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 하위 최적화 컴포넌트들
        self.memory_pool = AdvancedMemoryPool(
            pool_size=self.config.get('memory_pool_size', 5000)
        )
        
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get('max_workers'),
            use_processes=self.config.get('use_processes', False)
        )
        
        self.cache_optimizer = CacheOptimizer(
            max_size=self.config.get('cache_size', 10000),
            ttl=self.config.get('cache_ttl', 3600)
        )
        
        self.vectorized_ops = VectorizedOperations()
        
        # 성능 메트릭 수집
        self.performance_metrics = []
        self.global_stats = {
            'optimizations_applied': 0,
            'total_speedup': 0.0,
            'memory_saved_mb': 0.0
        }
        
        # 가비지 컬렉션 최적화
        self._optimize_gc()
    
    def _optimize_gc(self) -> None:
        """가비지 컬렉션 최적화"""
        # 임계값 조정으로 GC 빈도 감소
        gc.set_threshold(2000, 20, 20)  # 기본값: (700, 10, 10)
        
        # 0세대 GC 비활성화 (선택적)
        if self.config.get('disable_gc_gen0', False):
            gc.set_threshold(0)
    
    @performance_monitor
    def optimize_distance_calculations(self, coordinates_pairs: List[Tuple[Coordinates, Coordinates]]) -> List[float]:
        """거리 계산 최적화"""
        if not coordinates_pairs:
            return []
        
        # 벡터화 가능한 크기인지 확인
        if len(coordinates_pairs) >= 100 and NUMBA_AVAILABLE:
            return self._vectorized_distance_calculation(coordinates_pairs)
        else:
            return self._cached_distance_calculation(coordinates_pairs)
    
    def _vectorized_distance_calculation(self, coord_pairs: List[Tuple[Coordinates, Coordinates]]) -> List[float]:
        """벡터화된 거리 계산"""
        # NumPy 배열로 변환
        lat1_arr = np.array([pair[0].latitude for pair in coord_pairs])
        lon1_arr = np.array([pair[0].longitude for pair in coord_pairs])
        lat2_arr = np.array([pair[1].latitude for pair in coord_pairs])
        lon2_arr = np.array([pair[1].longitude for pair in coord_pairs])
        
        # 배치 계산
        distances = self.vectorized_ops.haversine_batch(lat1_arr, lon1_arr, lat2_arr, lon2_arr)
        
        self.global_stats['optimizations_applied'] += 1
        return distances.tolist()
    
    def _cached_distance_calculation(self, coord_pairs: List[Tuple[Coordinates, Coordinates]]) -> List[float]:
        """캐시된 거리 계산"""
        results = []
        
        for coord1, coord2 in coord_pairs:
            # 캐시 키 생성
            key = f"{coord1.latitude:.6f},{coord1.longitude:.6f}|{coord2.latitude:.6f},{coord2.longitude:.6f}"
            
            # 캐시된 계산
            distance = self.cache_optimizer.get_or_compute(
                key,
                lambda: coord1.distance_to(coord2)
            )
            results.append(distance)
        
        return results
    
    @performance_monitor  
    def optimize_nearest_neighbor_search(self, target: Coordinates, candidates: List[Order], 
                                       k: int = 1) -> List[Order]:
        """최근접 탐색 최적화"""
        if not candidates:
            return []
        
        if len(candidates) >= 500:
            # 대용량 데이터: 병렬 + 벡터화
            return self._parallel_nearest_search(target, candidates, k)
        elif len(candidates) >= 100:
            # 중용량 데이터: 벡터화
            return self._vectorized_nearest_search(target, candidates, k)
        else:
            # 소용량 데이터: 캐시 활용
            return self._cached_nearest_search(target, candidates, k)
    
    def _parallel_nearest_search(self, target: Coordinates, candidates: List[Order], k: int) -> List[Order]:
        """병렬 최근접 탐색"""
        # 후보를 청크로 분할
        chunk_size = max(50, len(candidates) // (self.parallel_processor.max_workers * 2))
        chunks = [candidates[i:i+chunk_size] for i in range(0, len(candidates), chunk_size)]
        
        # 병렬로 각 청크에서 k-nearest 찾기
        chunk_results = self.parallel_processor.execute_parallel(
            lambda chunk: self._vectorized_nearest_search(target, chunk, k),
            chunks
        )
        
        # 결과 병합
        all_candidates = []
        for chunk_result in chunk_results:
            all_candidates.extend(chunk_result)
        
        # 최종 k개 선택
        if len(all_candidates) <= k:
            return all_candidates
        
        # 거리로 정렬하여 k개 선택
        distances = [(order, target.distance_to(order.coordinates)) for order in all_candidates]
        distances.sort(key=lambda x: x[1])
        
        return [order for order, _ in distances[:k]]
    
    def _vectorized_nearest_search(self, target: Coordinates, candidates: List[Order], k: int) -> List[Order]:
        """벡터화된 최근접 탐색"""
        if not NUMBA_AVAILABLE:
            return self._cached_nearest_search(target, candidates, k)
        
        # 좌표 배열 생성
        target_coords = np.array([target.latitude, target.longitude])
        candidate_coords = np.array([[o.coordinates.latitude, o.coordinates.longitude] for o in candidates])
        
        # 벡터화된 거리 계산
        distances = np.sqrt(np.sum((candidate_coords - target_coords) ** 2, axis=1))
        
        # k-nearest 인덱스 찾기
        if k >= len(candidates):
            nearest_indices = np.arange(len(candidates))
        else:
            nearest_indices = self.vectorized_ops.find_k_nearest_vectorized(distances, k)
        
        return [candidates[i] for i in nearest_indices]
    
    def _cached_nearest_search(self, target: Coordinates, candidates: List[Order], k: int) -> List[Order]:
        """캐시된 최근접 탐색"""
        # 캐시 키 생성
        candidate_ids = sorted([c.id for c in candidates])
        cache_key = f"nearest_{target.latitude:.4f}_{target.longitude:.4f}_{hash(tuple(candidate_ids))}"
        
        # 캐시된 계산
        def compute_nearest():
            distances = [(order, target.distance_to(order.coordinates)) for order in candidates]
            distances.sort(key=lambda x: x[1])
            return [order for order, _ in distances[:k]]
        
        return self.cache_optimizer.get_or_compute(cache_key, compute_nearest)
    
    def optimize_memory_usage(self) -> None:
        """메모리 사용량 최적화"""
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        # 메모리 풀 정리
        current_usage = self.memory_pool.stats['current_usage']
        if current_usage == 0:  # 사용 중인 객체가 없으면
            # 풀 크기 축소
            while not self.memory_pool.available.empty():
                try:
                    self.memory_pool.available.get_nowait()
                except queue.Empty:
                    break
        
        self.logger.debug(f"메모리 최적화: {collected}개 객체 수집")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 최적화 요약"""
        return {
            'global_stats': self.global_stats.copy(),
            'memory_pool': self.memory_pool.get_stats(),
            'parallel_processor': self.parallel_processor.get_stats(),
            'cache_optimizer': self.cache_optimizer.get_stats(),
            'numba_available': NUMBA_AVAILABLE,
            'cpu_count': cpu_count(),
            'config': self.config
        }
    
    def benchmark_optimization(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """최적화 벤치마킹"""
        results = {}
        
        # 거리 계산 벤치마크
        if 'coordinate_pairs' in test_data:
            coord_pairs = test_data['coordinate_pairs']
            
            start_time = time.perf_counter()
            optimized_results = self.optimize_distance_calculations(coord_pairs)
            optimized_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            baseline_results = [pair[0].distance_to(pair[1]) for pair in coord_pairs]
            baseline_time = time.perf_counter() - start_time
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            results['distance_calculation_speedup'] = speedup
        
        # 최근접 탐색 벤치마크
        if 'nearest_search_data' in test_data:
            target, candidates, k = test_data['nearest_search_data']
            
            start_time = time.perf_counter()
            optimized_results = self.optimize_nearest_neighbor_search(target, candidates, k)
            optimized_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            distances = [(order, target.distance_to(order.coordinates)) for order in candidates]
            distances.sort(key=lambda x: x[1])
            baseline_results = [order for order, _ in distances[:k]]
            baseline_time = time.perf_counter() - start_time
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            results['nearest_search_speedup'] = speedup
        
        return results