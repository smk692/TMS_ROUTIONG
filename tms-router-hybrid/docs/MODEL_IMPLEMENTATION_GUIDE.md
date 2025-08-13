# 🚀 TMS Models 구조 개선 구현 가이드

## 📖 개요

기존 평면적인 models 구조를 계층적이고 논리적인 구조로 개선하기 위한 단계별 구현 가이드입니다.

## 🗂️ 목표 구조

```
core/models/
├── __init__.py                    # 깔끔한 통합 인터페이스
├── base/                          # 기본 클래스와 공통 요소
│   ├── __init__.py
│   ├── base_entity.py            # 기본 엔티티 추상 클래스
│   ├── value_objects.py          # 공통 값 객체 (ID, 시간 등)
│   └── enums.py                  # 모든 열거형 통합
├── domain/                        # 핵심 비즈니스 도메인
│   ├── __init__.py
│   ├── order.py                  # 주문 엔티티 (개선)
│   ├── vehicle.py                # 차량 엔티티 (개선)
│   └── region.py                 # 권역 엔티티 (개선)
├── results/                       # 처리 결과 및 집계 모델
│   ├── __init__.py
│   ├── dispatch_result.py        # 배차 실행 결과
│   └── map_display_result.py     # UI 표시용 모델
└── coordinates/                   # 지리 정보 특화
    ├── __init__.py
    └── coordinates.py            # 좌표 + 지리 계산
```

## 🔧 구현 단계

### Phase 1: 기본 구조 생성 ⭐

#### Step 1.1: 디렉토리 생성
```bash
mkdir -p core/models/{base,domain,results,coordinates}
touch core/models/{base,domain,results,coordinates}/__init__.py
```

#### Step 1.2: Base 모듈 구현

**`core/models/base/base_entity.py`**
```python
"""기본 엔티티 추상 클래스"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

class BaseEntity(ABC):
    """모든 도메인 엔티티의 기본 클래스
    
    Features:
    - 공통 식별자 관리
    - 생성/수정 시간 자동 관리
    - 유효성 검증 인터페이스
    - 딕셔너리 변환 지원
    """
    
    def __init__(self, id: str):
        if not id or not id.strip():
            raise ValueError("Entity ID는 비어있을 수 없습니다")
        
        self.id = id.strip()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @abstractmethod
    def validate(self) -> bool:
        """엔티티 유효성 검증
        
        Returns:
            bool: 유효한 경우 True
        """
        pass
    
    def touch(self):
        """수정 시간 업데이트"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """기본 딕셔너리 변환"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __eq__(self, other) -> bool:
        """동등성 비교 (ID 기준)"""
        return isinstance(other, self.__class__) and self.id == other.id
    
    def __hash__(self) -> int:
        """해시 값 (ID 기준)"""
        return hash(f"{self.__class__.__name__}:{self.id}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"
```

**`core/models/base/value_objects.py`**
```python
"""공통 값 객체들"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(frozen=True)
class EntityId:
    """엔티티 ID 값 객체"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("EntityId는 비어있을 수 없습니다")
    
    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class Timestamp:
    """시간 값 객체"""
    value: datetime
    
    @classmethod
    def now(cls) -> 'Timestamp':
        return cls(datetime.now())
    
    @classmethod
    def from_string(cls, date_string: str) -> 'Timestamp':
        """ISO 형식 문자열에서 생성"""
        return cls(datetime.fromisoformat(date_string))
    
    def is_before(self, other: 'Timestamp') -> bool:
        return self.value < other.value
    
    def is_after(self, other: 'Timestamp') -> bool:
        return self.value > other.value
    
    def to_string(self) -> str:
        """ISO 형식 문자열로 변환"""
        return self.value.isoformat()
    
    def __str__(self) -> str:
        return self.to_string()

@dataclass(frozen=True)
class Money:
    """금액 값 객체"""
    amount: float
    currency: str = "KRW"
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("금액은 음수일 수 없습니다")
        if not self.currency:
            raise ValueError("통화는 비어있을 수 없습니다")
    
    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("다른 통화끼리 계산할 수 없습니다")
        return Money(self.amount + other.amount, self.currency)
    
    def __str__(self) -> str:
        return f"{self.amount:,.0f} {self.currency}"
```

**`core/models/base/enums.py`**
```python
"""프로젝트 전역 열거형"""
from enum import Enum

# =============================================================================
# 주문 관련 열거형
# =============================================================================
class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"  
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    
    def is_assignable(self) -> bool:
        """배정 가능한 상태인지 확인"""
        return self == OrderStatus.PENDING

class Priority(Enum):
    """우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    
    def get_weight(self) -> float:
        """우선순위 가중치"""
        weights = {
            Priority.LOW: 0.8,
            Priority.NORMAL: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 2.0
        }
        return weights[self]

# =============================================================================
# 차량 관련 열거형
# =============================================================================
class VehicleType(Enum):
    """차량 유형"""
    TOP_CAR = "TOP_CAR"
    CARGO = "CARGO"
    OTHER = "OTHER"

class VehicleStatus(Enum):
    """차량 상태"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    IN_DELIVERY = "IN_DELIVERY"
    
    def is_available(self) -> bool:
        """사용 가능한 상태인지 확인"""
        return self == VehicleStatus.ACTIVE

class ExperienceLevel(Enum):
    """기사 경험 수준"""
    BEGINNER = 1      # 신입 - 70% 용량
    JUNIOR = 2        # 초급 - 85% 용량  
    INTERMEDIATE = 3  # 중급 - 100% 용량
    SENIOR = 4        # 고급 - 115% 용량
    EXPERT = 5        # 전문가 - 130% 용량
    
    def get_capacity_multiplier(self) -> float:
        """용량 계수 반환"""
        multipliers = {
            ExperienceLevel.BEGINNER: 0.70,
            ExperienceLevel.JUNIOR: 0.85,
            ExperienceLevel.INTERMEDIATE: 1.00,
            ExperienceLevel.SENIOR: 1.15,
            ExperienceLevel.EXPERT: 1.30
        }
        return multipliers[self]

# =============================================================================
# 권역 관련 열거형
# =============================================================================
class RegionDifficulty(Enum):
    """권역 난이도"""
    EASY = 1      # 쉬움 - 도심, 접근성 좋음
    NORMAL = 2    # 보통 - 일반 주거지역  
    HARD = 3      # 어려움 - 언덕, 골목길
    VERY_HARD = 4 # 매우 어려움 - 산간, 접근 제한
    
    def get_time_multiplier(self) -> float:
        """시간 계수 반환"""
        multipliers = {
            RegionDifficulty.EASY: 0.9,
            RegionDifficulty.NORMAL: 1.0,
            RegionDifficulty.HARD: 1.2,
            RegionDifficulty.VERY_HARD: 1.5
        }
        return multipliers[self]

# =============================================================================
# 배차 관련 열거형
# =============================================================================
class DispatchStatus(Enum):
    """배차 상태"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    def is_successful(self) -> bool:
        """성공적인 상태인지 확인"""
        return self in [DispatchStatus.SUCCESS, DispatchStatus.PARTIAL_SUCCESS]
```

### Phase 2: Coordinates 모듈 개선 🌍

**`core/models/coordinates/coordinates.py`**
```python
"""지리 정보 및 좌표 계산 모듈"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass(frozen=True)
class Coordinates:
    """좌표 값 객체 - 불변
    
    Features:
    - 좌표 유효성 검증
    - Haversine 거리 계산
    - 다양한 포맷 지원
    """
    latitude: float
    longitude: float
    
    def __post_init__(self):
        """좌표 유효성 검증"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"잘못된 위도: {self.latitude} (범위: -90~90)")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"잘못된 경도: {self.longitude} (범위: -180~180)")
    
    def distance_to(self, other: 'Coordinates') -> float:
        """Haversine 거리 계산 (km)"""
        R = 6371  # 지구 반지름 (km)
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def to_tuple(self) -> Tuple[float, float]:
        """튜플 형태로 반환"""
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> dict:
        """딕셔너리 형태로 반환"""
        return {"latitude": self.latitude, "longitude": self.longitude}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Coordinates':
        """딕셔너리에서 생성"""
        return cls(data['latitude'], data['longitude'])
    
    def is_within_radius(self, center: 'Coordinates', radius_km: float) -> bool:
        """특정 반경 내에 있는지 확인"""
        return self.distance_to(center) <= radius_km
    
    def __str__(self) -> str:
        return f"({self.latitude:.6f}, {self.longitude:.6f})"

class GeoCalculator:
    """지리 계산 유틸리티 클래스"""
    
    @staticmethod
    def find_centroid(coordinates: List[Coordinates]) -> Coordinates:
        """좌표들의 중심점 계산"""
        if not coordinates:
            raise ValueError("좌표 리스트가 비어있습니다")
        
        lat_sum = sum(c.latitude for c in coordinates)
        lon_sum = sum(c.longitude for c in coordinates)
        count = len(coordinates)
        
        return Coordinates(lat_sum / count, lon_sum / count)
    
    @staticmethod
    def calculate_bounding_box(coordinates: List[Coordinates]) -> Tuple[Coordinates, Coordinates]:
        """경계 박스 계산 (남서쪽, 북동쪽 모서리)"""
        if not coordinates:
            raise ValueError("좌표 리스트가 비어있습니다")
        
        min_lat = min(c.latitude for c in coordinates)
        max_lat = max(c.latitude for c in coordinates)
        min_lon = min(c.longitude for c in coordinates)
        max_lon = max(c.longitude for c in coordinates)
        
        return (
            Coordinates(min_lat, min_lon),  # 남서쪽
            Coordinates(max_lat, max_lon)   # 북동쪽
        )
    
    @staticmethod
    def calculate_total_distance(coordinates: List[Coordinates]) -> float:
        """순차적 좌표들의 총 거리 계산"""
        if len(coordinates) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(coordinates) - 1):
            total += coordinates[i].distance_to(coordinates[i + 1])
        
        return total
    
    @staticmethod
    def find_nearest_coordinate(target: Coordinates, 
                              candidates: List[Coordinates]) -> Optional[Coordinates]:
        """가장 가까운 좌표 찾기"""
        if not candidates:
            return None
        
        return min(candidates, key=lambda c: target.distance_to(c))
    
    @staticmethod
    def group_by_proximity(coordinates: List[Coordinates], 
                          max_distance_km: float) -> List[List[Coordinates]]:
        """근접도에 따른 좌표 그룹핑"""
        if not coordinates:
            return []
        
        groups = []
        remaining = coordinates.copy()
        
        while remaining:
            current_group = [remaining.pop(0)]
            i = 0
            
            while i < len(remaining):
                # 현재 그룹의 어느 점과라도 충분히 가까우면 그룹에 추가
                if any(coord.distance_to(remaining[i]) <= max_distance_km 
                       for coord in current_group):
                    current_group.append(remaining.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
```

### Phase 3: Domain 모델 개선 🏢

#### Step 3.1: Order 도메인 개선

**`core/models/domain/order.py`** (기존 구조 유지 + BaseEntity 상속)
```python
"""주문 도메인 모델"""
from datetime import datetime
from typing import Optional, Dict, Any

from ..base import BaseEntity, OrderStatus, Priority
from ..coordinates import Coordinates

class Order(BaseEntity):
    """주문 엔티티
    
    Features:
    - BaseEntity 상속으로 표준화
    - 풍부한 비즈니스 로직
    - 유효성 검증 강화
    """
    
    def __init__(self, id: str, center_id: str, region_id: str, 
                 coordinates: Coordinates, address: str, 
                 priority: Priority = Priority.NORMAL):
        super().__init__(id)
        self.center_id = center_id
        self.region_id = region_id
        self.coordinates = coordinates
        self.address = address
        self.priority = priority
        self.status = OrderStatus.PENDING
        self.assigned_vehicle_id: Optional[str] = None
        self.estimated_delivery_time: Optional[int] = None
    
    def validate(self) -> bool:
        """주문 유효성 검증"""
        return (
            bool(self.id and self.center_id and self.region_id 
                 and self.coordinates and self.address) and
            isinstance(self.priority, Priority) and
            isinstance(self.status, OrderStatus)
        )
    
    def assign_to_vehicle(self, vehicle_id: str, estimated_time: int = None):
        """차량에 할당"""
        if not self.is_assignable():
            raise ValueError(f"주문 {self.id}는 배정 불가능한 상태입니다: {self.status}")
        
        self.assigned_vehicle_id = vehicle_id
        self.status = OrderStatus.ASSIGNED
        if estimated_time:
            self.estimated_delivery_time = estimated_time
        self.touch()
    
    def is_assignable(self) -> bool:
        """배정 가능한 상태인지 확인"""
        return self.status.is_assignable()
    
    def get_priority_weight(self) -> float:
        """우선순위 가중치 반환"""
        return self.priority.get_weight()
    
    def start_delivery(self):
        """배송 시작"""
        if self.status != OrderStatus.ASSIGNED:
            raise ValueError("배정된 주문만 배송을 시작할 수 있습니다")
        self.status = OrderStatus.IN_PROGRESS
        self.touch()
    
    def complete_delivery(self):
        """배송 완료"""
        if self.status != OrderStatus.IN_PROGRESS:
            raise ValueError("진행 중인 주문만 완료할 수 있습니다")
        self.status = OrderStatus.COMPLETED
        self.touch()
    
    def cancel(self):
        """주문 취소"""
        if self.status == OrderStatus.COMPLETED:
            raise ValueError("완료된 주문은 취소할 수 없습니다")
        self.status = OrderStatus.CANCELLED
        self.assigned_vehicle_id = None
        self.touch()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        base_dict = super().to_dict()
        base_dict.update({
            'center_id': self.center_id,
            'region_id': self.region_id,
            'coordinates': self.coordinates.to_dict(),
            'address': self.address,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'estimated_delivery_time': self.estimated_delivery_time
        })
        return base_dict
    
    def __str__(self) -> str:
        return f"Order({self.id}, {self.address}, {self.priority.value})"
```

### Phase 4: 마이그레이션 스크립트 💫

**`migrate_models.py`** (실제 마이그레이션용)
```python
"""Models 구조 개선 마이그레이션 스크립트"""
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """새로운 디렉토리 구조 생성"""
    base_path = Path("core/models")
    
    # 디렉토리 생성
    subdirs = ["base", "domain", "results", "coordinates"]
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)
        (base_path / subdir / "__init__.py").touch()
    
    print("✅ 디렉토리 구조 생성 완료")

def backup_existing_files():
    """기존 파일들 백업"""
    backup_dir = Path("backup/models_old")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("core/models")
    existing_files = [
        "order.py", "vehicle.py", "region.py", 
        "dispatch_result.py", "map_display_result.py", "coordinates.py"
    ]
    
    for file in existing_files:
        src = models_dir / file
        if src.exists():
            shutil.copy2(src, backup_dir / file)
    
    print("✅ 기존 파일 백업 완료")

def move_files_to_new_structure():
    """파일들을 새로운 구조로 이동"""
    models_dir = Path("core/models")
    
    # 이동 맵핑
    move_mapping = {
        "order.py": "domain/order.py",
        "vehicle.py": "domain/vehicle.py", 
        "region.py": "domain/region.py",
        "dispatch_result.py": "results/dispatch_result.py",
        "map_display_result.py": "results/map_display_result.py",
        "coordinates.py": "coordinates/coordinates.py"
    }
    
    for src_file, dst_path in move_mapping.items():
        src = models_dir / src_file
        dst = models_dir / dst_path
        
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"이동: {src_file} → {dst_path}")
    
    print("✅ 파일 이동 완료")

def update_init_files():
    """__init__.py 파일들 업데이트"""
    models_dir = Path("core/models")
    
    # 각 서브 모듈의 __init__.py 작성
    init_contents = {
        "base/__init__.py": '''"""기본 추상화 계층"""
from .base_entity import BaseEntity
from .value_objects import EntityId, Timestamp, Money
from .enums import *

__all__ = [
    'BaseEntity', 'EntityId', 'Timestamp', 'Money',
    'OrderStatus', 'Priority', 'VehicleType', 'VehicleStatus', 
    'ExperienceLevel', 'RegionDifficulty', 'DispatchStatus'
]''',
        
        "domain/__init__.py": '''"""핵심 도메인 계층"""
from .order import Order
from .vehicle import Vehicle
from .region import Region

__all__ = ['Order', 'Vehicle', 'Region']''',
        
        "results/__init__.py": '''"""결과 모델 계층"""
from .dispatch_result import DispatchResult, VehicleAssignment, DispatchMetrics
from .map_display_result import MapDisplayResult, VehicleAssignmentResult

__all__ = [
    'DispatchResult', 'VehicleAssignment', 'DispatchMetrics',
    'MapDisplayResult', 'VehicleAssignmentResult'
]''',
        
        "coordinates/__init__.py": '''"""지리 정보 계층"""
from .coordinates import Coordinates, GeoCalculator

__all__ = ['Coordinates', 'GeoCalculator']'''
    }
    
    for file_path, content in init_contents.items():
        full_path = models_dir / file_path
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("✅ __init__.py 파일 업데이트 완료")

if __name__ == "__main__":
    print("🚀 Models 구조 개선 마이그레이션 시작")
    
    try:
        create_directory_structure()
        backup_existing_files()
        move_files_to_new_structure()
        update_init_files()
        
        print("\n🎉 마이그레이션 완료!")
        print("📁 새로운 구조:")
        print("  - base/: 기본 추상화 계층")
        print("  - domain/: 핵심 비즈니스 도메인")
        print("  - results/: 처리 결과 모델")
        print("  - coordinates/: 지리 정보")
        
    except Exception as e:
        print(f"❌ 마이그레이션 실패: {e}")
        print("백업 파일을 확인하여 복원하세요.")
```

## 📈 기대 효과

### 1. **구조적 명확성**
```python
# Before: 모든 것이 한 곳에
from core.models import Order, DispatchResult, Coordinates

# After: 논리적 그룹핑
from core.models.domain import Order
from core.models.results import DispatchResult
from core.models.coordinates import Coordinates

# 하위 호환성도 유지
from core.models import Order, DispatchResult, Coordinates  # 여전히 작동
```

### 2. **확장성 향상**
- 새로운 도메인 추가 시 명확한 위치
- 기존 구조에 영향 없이 기능 확장
- 테스트 케이스 분리 용이

### 3. **유지보수성 개선**
- 관련 기능들의 논리적 그룹핑
- 의존성 방향 명확화
- 코드 검색 및 네비게이션 향상

### 4. **개발 생산성**
- IDE 자동완성 개선
- 타입 힌팅 강화
- 문서화 자동 생성 지원

## ⚠️ 주의사항

### 1. **점진적 마이그레이션**
- 기존 코드 호환성 100% 유지
- 단계별 검증 후 진행
- 롤백 계획 수립

### 2. **테스트 coverage**
- 마이그레이션 전후 동일한 동작 보장
- 새로운 기능에 대한 테스트 추가
- 성능 저하 없음 확인

### 3. **문서화**
- 새로운 구조에 대한 명확한 문서
- 마이그레이션 가이드 제공
- 개발자 온보딩 자료 업데이트

## 🎯 다음 단계

1. **Phase 1 구현**: base 모듈 먼저 구현
2. **단위 테스트**: 각 계층별 독립적 테스트
3. **통합 테스트**: 전체 시스템 동작 확인
4. **문서화**: 새로운 구조 가이드 작성
5. **점진적 적용**: 새로운 기능부터 적용

이 가이드를 따라 구현하면 TMS Router의 모델 계층이 더욱 견고하고 확장 가능한 구조로 발전할 것입니다! 🚀