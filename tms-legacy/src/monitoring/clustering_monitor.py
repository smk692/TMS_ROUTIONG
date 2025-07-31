import logging
import json
import time
import tracemalloc
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from functools import wraps

from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle

logger = logging.getLogger(__name__)

@dataclass
class ClusteringMetrics:
    """클러스터링 메트릭스를 저장하는 클래스"""
    strategy_name: str
    start_time: datetime
    end_time: datetime
    num_points: int
    num_vehicles: int
    num_clusters: int
    execution_time: float
    memory_usage: float
    cluster_sizes: List[int]
    average_priority: float
    capacity_usage: List[Dict[str, float]]
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "num_points": self.num_points,
            "num_vehicles": self.num_vehicles,
            "num_clusters": self.num_clusters,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cluster_sizes": self.cluster_sizes,
            "average_priority": self.average_priority,
            "capacity_usage": self.capacity_usage,
            "success": self.success,
            "error_message": self.error_message
        }

class ClusteringMonitor:
    """클러스터링 모니터링 클래스"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics_history = []
        return cls._instance

    def monitor(self, strategy_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(points: List[DeliveryPoint], vehicles: List[Vehicle], *args, **kwargs):
                # 메모리 추적 시작
                tracemalloc.start()
                start_memory = tracemalloc.get_traced_memory()[0]
                
                start_time = datetime.now()
                success = True
                error_message = ""
                clusters = None
                
                try:
                    clusters = func(points, vehicles, *args, **kwargs)
                    
                    if clusters is None:
                        success = False
                        error_message = "Clustering returned None"
                        
                except Exception as e:
                    success = False
                    error_message = str(e)
                    logger.error(f"Clustering error: {error_message}")
                    logger.error(traceback.format_exc())
                    
                finally:
                    end_time = datetime.now()
                    current_memory = tracemalloc.get_traced_memory()[0]
                    memory_usage = (current_memory - start_memory) / 1024 / 1024  # MB
                    tracemalloc.stop()
                    
                    metrics = ClusteringMetrics(
                        strategy_name=strategy_name,
                        start_time=start_time,
                        end_time=end_time,
                        num_points=len(points),
                        num_vehicles=len(vehicles),
                        num_clusters=len(clusters) if clusters else 0,
                        execution_time=(end_time - start_time).total_seconds(),
                        memory_usage=memory_usage,
                        cluster_sizes=[len(c) for c in clusters] if clusters else [],
                        average_priority=self._calculate_average_priority(clusters) if clusters else 0.0,
                        capacity_usage=self._calculate_capacity_usage(clusters, vehicles) if clusters else [],
                        success=success,
                        error_message=error_message
                    )
                    
                    self.metrics_history.append(metrics)
                    self._log_metrics(metrics)
                    
                return clusters
            return wrapper
        return decorator

    def _calculate_average_priority(self, clusters: Optional[List[List[DeliveryPoint]]]) -> float:
        if not clusters:
            return 0.0
        total_priority = sum(sum(p.priority for p in cluster) for cluster in clusters)
        total_points = sum(len(cluster) for cluster in clusters)
        return total_priority / total_points if total_points > 0 else 0.0

    def _calculate_capacity_usage(self, 
                                clusters: Optional[List[List[DeliveryPoint]]], 
                                vehicles: List[Vehicle]) -> List[Dict[str, float]]:
        if not clusters:
            return []
        
        usage = []
        for cluster, vehicle in zip(clusters, vehicles):
            total_volume = sum(p.volume for p in cluster)
            total_weight = sum(p.weight for p in cluster)
            
            usage.append({
                "volume_ratio": total_volume / vehicle.capacity.volume,
                "weight_ratio": total_weight / vehicle.capacity.weight
            })
        return usage

    def _log_metrics(self, metrics: ClusteringMetrics):
        logger.info(f"""
        클러스터링 실행 결과:
        - 전략: {metrics.strategy_name}
        - 실행 시간: {metrics.execution_time:.2f}초
        - 메모리 사용: {metrics.memory_usage:.2f}MB
        - 포인트 수: {metrics.num_points}
        - 클러스터 수: {metrics.num_clusters}
        - 성공 여부: {metrics.success}
        - 클러스터 크기: {metrics.cluster_sizes}
        - 평균 우선순위: {metrics.average_priority:.2f}
        """)
        
        if not metrics.success:
            logger.error(f"클러스터링 실패: {metrics.error_message}")

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
            
        return {
            "total_runs": len(self.metrics_history),
            "success_rate": sum(1 for m in self.metrics_history if m.success) / len(self.metrics_history),
            "average_execution_time": sum(m.execution_time for m in self.metrics_history) / len(self.metrics_history),
            "average_memory_usage": sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history),
            "strategy_distribution": self._get_strategy_distribution()
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        distribution = {}
        for metrics in self.metrics_history:
            distribution[metrics.strategy_name] = distribution.get(metrics.strategy_name, 0) + 1
        return distribution

    def save_metrics(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)

    def load_metrics(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.metrics_history = [ClusteringMetrics(**m) for m in data]

    def clear_metrics(self):
        """메트릭스 히스토리 초기화"""
        self.metrics_history = []
