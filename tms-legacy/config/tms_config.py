#!/usr/bin/env python3
"""
TMS 배송 경로 최적화 시스템 설정 파일

모든 변경 가능한 변수들을 중앙에서 관리합니다.
이 파일을 수정하면 코드 변경 없이 시스템 동작을 조정할 수 있습니다.
"""

from datetime import datetime
from typing import Dict, Any, List

class TMSConfig:
    """TMS 시스템 설정 클래스"""
    
    def __init__(self):
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """기본 설정 로드"""
        return {
            # 🚗 차량 관련 설정
            "vehicles": {
                "count": 24,                    # 차량 수 (8 → 24로 증가)
                "capacity": {
                    "volume": 5.0,              # 부피 용량 (m³)
                    "weight": 1000.0            # 무게 용량 (kg)
                },
                "operating_hours": {
                    "start_hour": 6,            # 시작시간 (시)
                    "start_minute": 0,          # 시작시간 (분)
                    "end_hour": 14,             # 종료시간 (시)
                    "end_minute": 0             # 종료시간 (분)
                },
                "cost_per_km": 500.0,           # km당 운영비용 (원)
                "average_speed": 30.0           # 평균 속도 (km/h)
            },
            
            # 📍 물류센터 & 배송 설정
            "logistics": {
                "depots": [
                    {
                        "id": "incheon_center",
                        "name": "인천센터",
                        "address": "인천광역시 서구 청라국제도시",
                        "latitude": 37.5394,
                        "longitude": 126.6648,
                        "capacity": 1000,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "icheon_center", 
                        "name": "이천센터",
                        "address": "경기도 이천시 마장면",
                        "latitude": 37.263573,
                        "longitude": 127.028601,
                        "capacity": 1200,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "hwaseong_center",
                        "name": "화성센터", 
                        "address": "경기도 화성시 향남읍",
                        "latitude": 37.1967,
                        "longitude": 126.8169,
                        "capacity": 800,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "hanam_center",
                        "name": "하남센터",
                        "address": "경기도 하남시 미사강변도시",
                        "latitude": 37.5394,
                        "longitude": 127.2067,
                        "capacity": 900,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "gwangju_center",
                        "name": "광주센터",
                        "address": "경기도 광주시 오포읍",
                        "latitude": 37.4292,
                        "longitude": 127.2558,
                        "capacity": 700,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "ilsan_center",
                        "name": "일산센터",
                        "address": "경기도 고양시 일산서구",
                        "latitude": 37.6756,
                        "longitude": 126.7764,
                        "capacity": 600,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "namyangju_center",
                        "name": "남양주센터",
                        "address": "경기도 남양주시 화도읍",
                        "latitude": 37.6414,
                        "longitude": 127.3108,
                        "capacity": 500,
                        "operating_hours": {"start": 6, "end": 18}
                    },
                    {
                        "id": "gunpo_center",
                        "name": "군포센터",
                        "address": "경기도 군포시 산본동",
                        "latitude": 37.3617,
                        "longitude": 126.9352,
                        "capacity": 400,
                        "operating_hours": {"start": 6, "end": 18}
                    }
                ],
                "delivery": {
                    "max_distance": 15.0,       # 배송 반경 (km)
                    "points_per_vehicle": 15,   # 차량당 배송지 수 (50 → 15로 감소)
                    "service_time": 5,          # 배송지당 서비스 시간 (분)
                    "default_volume": 0.1,      # 기본 부피 (m³)
                    "default_weight": 5.0,      # 기본 무게 (kg)
                    "default_priority": 3       # 기본 우선순위 (1-5)
                }
            },
            
            # ⚖️ 제약조건 설정
            "constraints": {
                "max_working_hours": 8,         # 최대 근무시간 (시간)
                "max_points_per_vehicle": 20,   # 차량당 최대 배송지 (25 → 20으로 감소)
                "min_points_per_vehicle": 5,    # 차량당 최소 배송지 (10 → 5로 감소)
                "allow_overtime": False,        # 초과근무 허용 여부
                "consider_traffic": True,       # 교통상황 고려 여부
                "target_efficiency": 0.1        # 목표 효율성 (0.0-1.0)
            },
            
            # 🧠 알고리즘 파라미터
            "algorithms": {
                "clustering": {
                    "strategy": "enhanced_kmeans",  # 클러스터링 전략
                    "max_iterations": 50            # 클러스터링 최대 반복 (100→50)
                },
                "tsp": {
                    "max_iterations": 100,          # TSP 최대 반복 횟수 (150→100)
                    "max_no_improve": 20,           # 개선 없을 때 허용 횟수 (30→20)
                    "temperature": 80.0,            # Simulated Annealing 온도 (100→80)
                    "parallel_workers": 6           # 병렬 처리 워커 수 (4→6)
                }
            },
            
            # 🎨 시각화 설정
            "visualization": {
                "colors": [
                    "#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFA500",
                    "#800080", "#008080", "#FFD700", "#FF1493", "#32CD32",
                    "#FF4500", "#4169E1", "#DC143C", "#00CED1", "#FF6347"
                ],
                "map_center": {
                    "zoom_start": 11,               # 초기 줌 레벨
                    "tiles": "OpenStreetMap"        # 지도 타일
                },
                "marker_offset": 0.0003             # 동일 좌표 마커 오프셋
            },
            
            # 🔧 시스템 설정
            "system": {
                "api": {
                    "osrm_url": "http://router.project-osrm.org",
                    "timeout": 8,                   # API 타임아웃 (30→8초)
                    "max_workers": 6                # API 병렬 처리 워커 수 (3→6)
                },
                "performance": {
                    "cache_enabled": True,          # 캐시 사용 여부
                    "memory_limit": 1024,           # 메모리 제한 (MB)
                    "log_level": "INFO"             # 로그 레벨
                }
            }
        }
    
    def get(self, key_path: str, default=None):
        """점 표기법으로 설정값 가져오기 (예: 'vehicles.count')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """점 표기법으로 설정값 변경하기"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def update_from_args(self, args):
        """명령행 인수로부터 설정 업데이트"""
        if hasattr(args, 'vehicles') and args.vehicles:
            self.set('vehicles.count', args.vehicles)
        
        if hasattr(args, 'capacity_volume') and args.capacity_volume:
            self.set('vehicles.capacity.volume', args.capacity_volume)
        
        if hasattr(args, 'capacity_weight') and args.capacity_weight:
            self.set('vehicles.capacity.weight', args.capacity_weight)
        
        if hasattr(args, 'max_distance') and args.max_distance:
            self.set('logistics.delivery.max_distance', args.max_distance)
        
        if hasattr(args, 'points_per_vehicle') and args.points_per_vehicle:
            self.set('logistics.delivery.points_per_vehicle', args.points_per_vehicle)
        
        if hasattr(args, 'max_working_hours') and args.max_working_hours:
            self.set('constraints.max_working_hours', args.max_working_hours)
        
        if hasattr(args, 'tsp_iterations') and args.tsp_iterations:
            self.set('algorithms.tsp.max_iterations', args.tsp_iterations)
    
    def validate(self) -> List[str]:
        """설정값 검증"""
        errors = []
        
        # 차량 수 검증
        if self.get('vehicles.count') <= 0:
            errors.append("차량 수는 1 이상이어야 합니다.")
        
        # 용량 검증
        if self.get('vehicles.capacity.volume') <= 0:
            errors.append("차량 부피 용량은 0보다 커야 합니다.")
        
        if self.get('vehicles.capacity.weight') <= 0:
            errors.append("차량 무게 용량은 0보다 커야 합니다.")
        
        # 시간 검증
        start_hour = self.get('vehicles.operating_hours.start_hour')
        end_hour = self.get('vehicles.operating_hours.end_hour')
        
        if not (0 <= start_hour <= 23):
            errors.append("시작시간은 0-23 사이여야 합니다.")
        
        if not (0 <= end_hour <= 23):
            errors.append("종료시간은 0-23 사이여야 합니다.")
        
        if start_hour >= end_hour:
            errors.append("시작시간은 종료시간보다 빨라야 합니다.")
        
        # 배송 반경 검증
        if self.get('logistics.delivery.max_distance') <= 0:
            errors.append("배송 반경은 0보다 커야 합니다.")
        
        # 효율성 목표 검증
        efficiency = self.get('constraints.target_efficiency')
        if not (0.0 <= efficiency <= 1.0):
            errors.append("목표 효율성은 0.0-1.0 사이여야 합니다.")
        
        return errors
    
    def print_summary(self):
        """현재 설정 요약 출력"""
        print("🔧 현재 TMS 시스템 설정:")
        print(f"   🚗 차량: {self.get('vehicles.count')}대")
        print(f"   📦 용량: {self.get('vehicles.capacity.volume')}m³, {self.get('vehicles.capacity.weight')}kg")
        print(f"   ⏰ 운영: {self.get('vehicles.operating_hours.start_hour')}:00 - {self.get('vehicles.operating_hours.end_hour')}:00")
        print(f"   📍 반경: {self.get('logistics.delivery.max_distance')}km")
        print(f"   👥 배송지/차량: {self.get('logistics.delivery.points_per_vehicle')}개")
        print(f"   ⚖️ 최대 근무: {self.get('constraints.max_working_hours')}시간")
        print(f"   🧠 TSP 반복: {self.get('algorithms.tsp.max_iterations')}회")

# 전역 설정 인스턴스
tms_config = TMSConfig()

# 편의 함수들
def get_config(key_path: str, default=None):
    """설정값 가져오기"""
    return tms_config.get(key_path, default)

def set_config(key_path: str, value):
    """설정값 변경하기"""
    tms_config.set(key_path, value)

def validate_config():
    """설정 검증"""
    return tms_config.validate()

def print_config_summary():
    """설정 요약 출력"""
    tms_config.print_summary()

# 프리셋 설정들
PRESETS = {
    "ultra_fast": {
        "description": "초고속 처리 (최소 품질)",
        "overrides": {
            "algorithms.tsp.max_iterations": 30,        # 100→30 (70% 감소)
            "algorithms.tsp.max_no_improve": 8,         # 20→8 (60% 감소)
            "algorithms.clustering.max_iterations": 15, # 50→15 (70% 감소)
            "constraints.max_points_per_vehicle": 25,   # 40→25 (더 작은 클러스터)
            "constraints.max_working_hours": 5,         # 10→5 (짧은 근무시간)
            "system.api.timeout": 8,                    # 3→8초 (안정적인 타임아웃)
            "system.api.max_workers": 8,                # 12→8 (안정적인 병렬 처리)
            "logistics.delivery.max_distance": 8.0,     # 12→8km (작은 반경)
            "vehicles.count": 8                         # 15→8 (적은 차량)
        }
    },
    "fast": {
        "description": "빠른 처리 (낮은 품질)",
        "overrides": {
            "algorithms.tsp.max_iterations": 60,        # 100→60
            "algorithms.tsp.max_no_improve": 15,        # 20→15
            "algorithms.clustering.max_iterations": 30, # 50→30
            "constraints.max_points_per_vehicle": 30,   # 40→30
            "constraints.max_working_hours": 6,
            "system.api.timeout": 5,                    # 8→5초
            "system.api.max_workers": 8                 # 6→8
        }
    },
    "quality": {
        "description": "높은 품질 (느린 처리)",
        "overrides": {
            "algorithms.tsp.max_iterations": 200,
            "algorithms.tsp.temperature": 150.0,
            "constraints.target_efficiency": 0.05
        }
    },
    "large_scale": {
        "description": "대규모 처리",
        "overrides": {
            "vehicles.count": 25,
            "logistics.delivery.max_distance": 25.0,
            "constraints.max_points_per_vehicle": 80
        }
    },
    "test": {
        "description": "테스트 모드",
        "overrides": {
            "vehicles.count": 3,
            "logistics.delivery.max_distance": 10.0,
            "constraints.max_points_per_vehicle": 20
        }
    }
}

def apply_preset(preset_name: str):
    """프리셋 적용"""
    if preset_name not in PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {preset_name}")
    
    preset = PRESETS[preset_name]
    print(f"🎯 프리셋 적용: {preset_name} - {preset['description']}")
    
    for key_path, value in preset['overrides'].items():
        tms_config.set(key_path, value)
    
    print("✅ 프리셋 적용 완료")

def list_presets():
    """사용 가능한 프리셋 목록"""
    print("📋 사용 가능한 프리셋:")
    for name, preset in PRESETS.items():
        print(f"   {name}: {preset['description']}") 