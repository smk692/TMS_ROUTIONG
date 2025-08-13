"""
지리적 데이터 전처리 모듈
"""

import numpy as np
import logging
from typing import List, Tuple
from geopy.distance import geodesic

from ...models import Order, Coordinates


class GeographicPreprocessor:
    """지리적 데이터 전처리"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def preprocess_coordinates(self, orders: List[Order]) -> np.ndarray:
        """좌표 전처리 및 정규화"""
        
        # 1. 기본 좌표 추출
        coords = [(order.coordinates.latitude, order.coordinates.longitude) 
                 for order in orders]
        
        # 2. 이상치 제거 (표준편차 3배 초과)
        coords_filtered = self._remove_outliers(coords)
        
        # 3. 가중치 적용 (우선순위 높은 주문에 더 큰 영향)
        weighted_coords = self._apply_priority_weights(coords_filtered, orders)
        
        return np.array(weighted_coords)
    
    def _remove_outliers(self, coordinates: List[Tuple[float, float]]) -> List[Tuple]:
        """지리적 이상치 제거"""
        if len(coordinates) < 3:
            return coordinates
            
        coords_array = np.array(coordinates)
        
        # 위도, 경도 각각에 대해 Z-score 계산
        lat_mean, lat_std = coords_array[:, 0].mean(), coords_array[:, 0].std()
        lng_mean, lng_std = coords_array[:, 1].mean(), coords_array[:, 1].std()
        
        if lat_std == 0 or lng_std == 0:
            return coordinates
            
        lat_zscore = np.abs((coords_array[:, 0] - lat_mean) / lat_std)
        lng_zscore = np.abs((coords_array[:, 1] - lng_mean) / lng_std)
        
        # Z-score 3 이하인 좌표만 유지
        valid_mask = (lat_zscore <= 3) & (lng_zscore <= 3)
        return coords_array[valid_mask].tolist()
    
    def _apply_priority_weights(self, coordinates: List[Tuple], orders: List[Order]) -> List[Tuple]:
        """우선순위 가중치 적용"""
        
        # 현재는 단순히 좌표 반환 (향후 우선순위 기반 가중치 구현 가능)
        return coordinates