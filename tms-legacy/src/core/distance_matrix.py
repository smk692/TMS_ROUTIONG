# src/core/distance_matrix.py
from typing import Dict, List, Tuple
from functools import lru_cache
import numpy as np
from src.model.delivery_point import DeliveryPoint
from src.model.distance import distance

class DistanceMatrix:
    """최적화된 거리 행렬 캐싱 클래스"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance
    
    def __init__(self):
        # 싱글톤 패턴이므로 __new__에서 초기화
        pass
    
    def get_distance(self, point1: DeliveryPoint, point2: DeliveryPoint) -> float:
        """두 지점 간의 거리를 계산하고 캐싱"""
        # 좌표 기반으로 캐시 키 생성
        key = (
            (point1.latitude, point1.longitude),
            (point2.latitude, point2.longitude)
        )
        
        if key not in self._cache:
            self._cache[key] = distance(
                point1.latitude, point1.longitude,
                point2.latitude, point2.longitude
            )
        return self._cache[key]
    
    def compute_matrix(self, points: List[DeliveryPoint]) -> np.ndarray:
        """전체 거리 행렬 미리 계산 - NumPy 배열 사용"""
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.get_distance(points[i], points[j])
                distances[i,j] = distances[j,i] = dist
        
        return distances
    
    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()