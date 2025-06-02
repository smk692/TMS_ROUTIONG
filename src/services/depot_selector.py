"""
물류센터 자동 선택 서비스
배송지와 가장 가까운 물류센터를 자동으로 선택합니다.
"""

import math
from typing import List, Dict, Tuple, Optional
import pandas as pd
from src.utils.distance_calculator import calculate_distance_geodesic, find_nearest_point, assign_points_to_nearest_centers, calculate_distances_batch


class DepotSelectorService:
    """물류센터 자동 선택 서비스"""
    
    def __init__(self, config):
        self.config = config
        self.depots = config.get('logistics.depots', [])
        self.default_depot_id = config.get('logistics.default_depot_id', 'icheon_center')
        self.auto_select = config.get('logistics.auto_select_nearest', True)
    
    def find_nearest_depot(self, delivery_lat: float, delivery_lon: float) -> Dict:
        """배송지와 가장 가까운 물류센터 찾기 - 최적화된 버전"""
        if not self.depots:
            raise ValueError("물류센터 정보가 설정되지 않았습니다.")
        
        # 최적화된 거리 계산 사용
        depot_coords = [(depot['latitude'], depot['longitude']) for depot in self.depots]
        nearest_idx, min_distance = find_nearest_point(delivery_lat, delivery_lon, depot_coords)
            
        if nearest_idx >= 0:
            nearest_depot = self.depots[nearest_idx].copy()
            nearest_depot['distance_km'] = min_distance
        return nearest_depot
        
        # fallback: 첫 번째 depot 반환
        return self.depots[0]
    
    def get_depot_by_id(self, depot_id: str) -> Optional[Dict]:
        """ID로 물류센터 찾기"""
        for depot in self.depots:
            if depot['id'] == depot_id:
                return depot
        return None
    
    def get_default_depot(self) -> Dict:
        """기본 물류센터 반환"""
        depot = self.get_depot_by_id(self.default_depot_id)
        if depot:
            return depot
        
        # 기본 센터가 없으면 첫 번째 센터 반환
        if self.depots:
            return self.depots[0]
        
        raise ValueError("사용 가능한 물류센터가 없습니다.")
    
    def analyze_delivery_coverage(self, delivery_points: pd.DataFrame) -> Dict:
        """배송지별 최적 물류센터 분석 - 최적화된 버전"""
        if not self.auto_select:
            default_depot = self.get_default_depot()
            return {
                "selected_depot": default_depot,
                "coverage_analysis": f"자동 선택 비활성화 - {default_depot['name']} 사용"
            }
        
        depot_assignments = {}
        depot_counts = {}
        total_distance = 0
        
        # 배치 처리를 위한 좌표 추출
        delivery_coords = [(row['lat'], row['lng']) for _, row in delivery_points.iterrows()]
        depot_coords = [(depot['latitude'], depot['longitude']) for depot in self.depots]
        
        # 벡터화된 할당
        assignments = assign_points_to_nearest_centers(delivery_coords, depot_coords)
        distances = calculate_distances_batch(delivery_coords, depot_coords)
        
        for i, (_, point) in enumerate(delivery_points.iterrows()):
            depot_idx = assignments[i]
            depot = self.depots[depot_idx]
            depot_id = depot['id']
            distance_km = distances[i][depot_idx]
            
            if depot_id not in depot_assignments:
                depot_assignments[depot_id] = []
                depot_counts[depot_id] = 0
            
            depot_assignments[depot_id].append({
                'delivery_id': point.get('id', ''),
                'address': point.get('address', ''),
                'distance_km': distance_km
            })
            depot_counts[depot_id] += 1
            total_distance += distance_km
        
        # 가장 많은 배송지를 담당하는 센터 선택
        primary_depot_id = max(depot_counts, key=depot_counts.get)
        primary_depot = self.get_depot_by_id(primary_depot_id)
        
        analysis = {
            "selected_depot": primary_depot,
            "depot_assignments": depot_assignments,
            "depot_counts": depot_counts,
            "total_delivery_points": len(delivery_points),
            "average_distance_km": total_distance / len(delivery_points),
            "coverage_summary": self._generate_coverage_summary(depot_counts)
        }
        
        return analysis
    
    def _generate_coverage_summary(self, depot_counts: Dict) -> str:
        """커버리지 요약 생성"""
        total = sum(depot_counts.values())
        summary_parts = []
        
        for depot_id, count in sorted(depot_counts.items(), key=lambda x: x[1], reverse=True):
            depot = self.get_depot_by_id(depot_id)
            percentage = (count / total) * 100
            summary_parts.append(f"{depot['name']}: {count}개 ({percentage:.1f}%)")
        
        return " | ".join(summary_parts)
    
    def get_all_depots_info(self) -> List[Dict]:
        """모든 물류센터 정보 반환"""
        return self.depots.copy()
    
    def print_depot_status(self):
        """물류센터 현황 출력"""
        print("\n🏢 물류센터 현황:")
        print("=" * 60)
        
        for i, depot in enumerate(self.depots, 1):
            print(f"{i}. {depot['name']} ({depot['id']})")
            print(f"   📍 위치: {depot['address']}")
            print(f"   🗺️  좌표: {depot['latitude']:.6f}, {depot['longitude']:.6f}")
            print(f"   🌍 권역: {depot['region']}")
            print()
        
        print(f"✅ 총 {len(self.depots)}개 물류센터 등록")
        print(f"🔧 자동 선택: {'활성화' if self.auto_select else '비활성화'}")
        print(f"🏠 기본 센터: {self.get_default_depot()['name']}") 