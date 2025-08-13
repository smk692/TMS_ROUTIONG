# TMS 배차 시스템 - 개선된 프로젝트 구조 및 개발 가이드

## 1. 개선된 프로젝트 디렉토리 구조

### 1.1 UseCase 기반 클린 아키텍처 구성

```
tms_dispatch_system/
├── main.py                          # 메인 실행 스크립트 (간소화)
├── app/                            # 애플리케이션 레이어
│   ├── __init__.py
│   ├── dispatch_app.py             # 배차 애플리케이션 컨트롤러
│   └── config_loader.py            # 설정 로더
├── usecases/                       # UseCase 레이어 (비즈니스 로직)
│   ├── __init__.py
│   ├── execute_dispatch.py         # 배차 실행 UseCase
│   ├── collect_data.py             # 데이터 수집 UseCase
│   ├── adjust_orders.py            # 주문량 조정 UseCase
│   ├── calculate_routes.py         # 경로 계산 UseCase
│   ├── optimize_dispatch.py        # 최적화 실행 UseCase
│   └── validate_results.py         # 결과 검증 UseCase
├── strategies/                     # 전략 패턴 구현
│   ├── __init__.py
│   ├── algorithm_strategies/       # 알고리즘 전략들
│   │   ├── __init__.py
│   │   ├── algorithm_strategy.py   # 알고리즘 전략 인터페이스
│   │   ├── simple_strategy.py      # 단순 알고리즘 전략
│   │   ├── advanced_strategy.py    # 고급 알고리즘 전략
│   │   └── emergency_strategy.py   # 비상 알고리즘 전략
│   ├── api_strategies/            # API 전략들
│   │   ├── __init__.py
│   │   ├── api_strategy.py        # API 전략 인터페이스
│   │   ├── primary_api_strategy.py
│   │   ├── backup_api_strategy.py
│   │   └── offline_strategy.py
│   └── adjustment_strategies/      # 조정 전략들
│       ├── __init__.py
│       ├── adjustment_strategy.py
│       ├── weather_strategy.py
│       ├── traffic_strategy.py
│       └── combined_strategy.py
├── models/                        # 도메인 모델
│   ├── __init__.py
│   ├── domain/                    # 도메인 엔티티
│   │   ├── __init__.py
│   │   ├── order.py              # 주문 엔티티
│   │   ├── vehicle.py            # 차량 엔티티
│   │   ├── region.py             # 권역 엔티티
│   │   └── dispatch_result.py    # 배차 결과 엔티티
│   └── value_objects/            # 값 객체들
│       ├── __init__.py
│       ├── coordinates.py
│       ├── time_window.py
│       └── capacity.py
├── services/                     # 도메인 서비스
│   ├── __init__.py
│   ├── data_service.py          # 데이터 서비스
│   ├── condition_service.py     # 외부 조건 서비스
│   ├── route_service.py         # 경로 서비스
│   └── optimization_service.py  # 최적화 서비스
├── repositories/                # 데이터 접근 레이어
│   ├── __init__.py
│   ├── interfaces/             # 레포지토리 인터페이스
│   │   ├── __init__.py
│   │   ├── order_repository.py
│   │   ├── vehicle_repository.py
│   │   └── result_repository.py
│   └── implementations/        # 구현체
│       ├── __init__.py
│       ├── tms_order_repository.py
│       ├── tms_vehicle_repository.py
│       └── file_result_repository.py
├── algorithms/                  # 알고리즘 구현
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── algorithm_interface.py
│   │   └── algorithm_factory.py
│   ├── basic/
│   │   ├── __init__.py
│   │   ├── nearest_neighbor.py
│   │   └── capacity_first.py
│   ├── advanced/
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py
│   │   ├── simulated_annealing.py
│   │   └── large_neighborhood.py
│   └── hybrid/
│       ├── __init__.py
│       └── hybrid_algorithm.py
├── external/                   # 외부 시스템 연동
│   ├── __init__.py
│   ├── apis/                  # 외부 API
│   │   ├── __init__.py
│   │   ├── weather/
│   │   │   ├── __init__.py
│   │   │   ├── kma_api.py
│   │   │   └── openweather_api.py
│   │   ├── traffic/
│   │   │   ├── __init__.py
│   │   │   ├── molit_api.py
│   │   │   └── here_traffic_api.py
│   │   └── routing/
│   │       ├── __init__.py
│   │       ├── openrouteservice_api.py
│   │       ├── here_routing_api.py
│   │       ├── kakao_api.py
│   │       └── mapbox_api.py
│   └── databases/            # 데이터베이스 연동
│       ├── __init__.py
│       ├── tms_database.py
│       └── cache_database.py
├── infrastructure/           # 인프라스트럭처
│   ├── __init__.py
│   ├── cache/               # 캐시 시스템
│   │   ├── __init__.py
│   │   ├── cache_manager.py
│   │   ├── memory_cache.py
│   │   ├── file_cache.py
│   │   └── redis_cache.py
│   ├── logging/            # 로깅 시스템
│   │   ├── __init__.py
│   │   ├── logger_config.py
│   │   └── performance_logger.py
│   └── monitoring/         # 모니터링
│       ├── __init__.py
│       ├── metrics_collector.py
│       └── health_checker.py
├── config/                 # 설정 관리
│   ├── __init__.py
│   ├── settings/          # 설정 파일들
│   │   ├── __init__.py
│   │   ├── base_settings.py
│   │   ├── algorithm_settings.py
│   │   ├── api_settings.py
│   │   └── cache_settings.py
│   └── rules/            # 비즈니스 룰 설정
│       ├── __init__.py
│       ├── selection_rules.yaml
│       ├── adjustment_rules.yaml
│       └── validation_rules.yaml
├── utils/                # 공통 유틸리티
│   ├── __init__.py
│   ├── calculators/     # 계산 유틸리티
│   │   ├── __init__.py
│   │   ├── distance_calculator.py
│   │   ├── time_calculator.py
│   │   └── complexity_calculator.py
│   ├── validators/      # 검증 유틸리티
│   │   ├── __init__.py
│   │   ├── data_validator.py
│   │   └── result_validator.py
│   └── converters/      # 변환 유틸리티
│       ├── __init__.py
│       ├── coordinate_converter.py
│       └── format_converter.py
├── tests/              # 테스트 코드
│   ├── __init__.py
│   ├── unit/          # 유닛 테스트
│   │   ├── test_usecases/
│   │   ├── test_strategies/
│   │   ├── test_algorithms/
│   │   └── test_services/
│   ├── integration/   # 통합 테스트
│   │   ├── test_api_integration/
│   │   └── test_end_to_end/
│   └── fixtures/      # 테스트 데이터
│       ├── sample_orders.json
│       ├── sample_vehicles.json
│       └── sample_regions.json
├── docs/              # 문서
│   ├── rules/
│   │   ├── algorithm_selection_rules.md
│   │   ├── dynamic_adjustment_rules.md
│   │   ├── route_optimization_rules.md
│   │   └── process_flow_integration.md
│   ├── architecture/
│   │   ├── clean_architecture.md
│   │   └── strategy_patterns.md
│   └── deployment/
│       ├── installation.md
│       └── configuration.md
├── scripts/           # 실행 스크립트
│   ├── deploy.sh
│   ├── test.sh
│   └── benchmark.sh
├── cache/            # 캐시 디렉토리
├── logs/             # 로그 디렉토리
├── .env.example      # 환경 변수 예제
├── requirements.txt  # 의존성 패키지
└── README.md        # 프로젝트 설명
```

## 2. UseCase 기반 비즈니스 로직 분리

### 2.1 간소화된 메인 스크립트

```python
# main.py
"""
TMS 배차 시스템 메인 실행 스크립트
- UseCase 패턴으로 비즈니스 로직 분리
- 전략 패턴으로 알고리즘 선택
- 깔끔한 의존성 주입
"""
import sys
from pathlib import Path
from app.dispatch_app import DispatchApplication
from app.config_loader import ConfigLoader

def main():
    """메인 실행 함수 - 매우 간단하게 유지"""
    try:
        # 설정 로드
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # 애플리케이션 생성
        dispatch_app = DispatchApplication(config)
        
        # 배차 실행
        result = dispatch_app.execute_dispatch()
        
        # 결과 출력
        print(f"배차 완료: {result.batch_id}")
        print(f"처리된 주문: {result.total_orders}개")
        print(f"총 거리: {result.total_distance:.1f}km")
        
        return 0
        
    except Exception as e:
        print(f"배차 실행 실패: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

### 2.2 애플리케이션 컨트롤러

```python
# app/dispatch_app.py
"""
배차 애플리케이션 컨트롤러
- UseCase들을 조합하여 전체 플로우 관리
- 의존성 주입을 통한 느슨한 결합
"""
from usecases.execute_dispatch import ExecuteDispatchUseCase
from usecases.collect_data import CollectDataUseCase
from usecases.adjust_orders import AdjustOrdersUseCase
from usecases.calculate_routes import CalculateRoutesUseCase
from usecases.optimize_dispatch import OptimizeDispatchUseCase
from usecases.validate_results import ValidateResultsUseCase

class DispatchApplication:
    """배차 애플리케이션 메인 컨트롤러"""
    
    def __init__(self, config):
        self.config = config
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        """의존성 주입 설정"""
        # UseCase들 초기화
        self.collect_data_usecase = CollectDataUseCase(self.config)
        self.adjust_orders_usecase = AdjustOrdersUseCase(self.config)
        self.calculate_routes_usecase = CalculateRoutesUseCase(self.config)
        self.optimize_dispatch_usecase = OptimizeDispatchUseCase(self.config)
        self.validate_results_usecase = ValidateResultsUseCase(self.config)
        
        # 메인 UseCase 초기화
        self.execute_dispatch_usecase = ExecuteDispatchUseCase(
            collect_data_usecase=self.collect_data_usecase,
            adjust_orders_usecase=self.adjust_orders_usecase,
            calculate_routes_usecase=self.calculate_routes_usecase,
            optimize_dispatch_usecase=self.optimize_dispatch_usecase,
            validate_results_usecase=self.validate_results_usecase
        )
    
    def execute_dispatch(self):
        """배차 실행 - UseCase에 위임"""
        return self.execute_dispatch_usecase.execute()
```

## 3. UseCase 구현 패턴

### 3.1 메인 배차 실행 UseCase

```python
# usecases/execute_dispatch.py
"""
배차 실행 UseCase
- 전체 배차 프로세스의 오케스트레이션
- 각 단계별 UseCase 조합
- 오류 처리 및 복구 로직
"""
from datetime import datetime
import time
from models.domain.dispatch_result import DispatchResult

class ExecuteDispatchUseCase:
    """배차 실행 UseCase"""
    
    def __init__(self, collect_data_usecase, adjust_orders_usecase, 
                 calculate_routes_usecase, optimize_dispatch_usecase, 
                 validate_results_usecase):
        self.collect_data = collect_data_usecase
        self.adjust_orders = adjust_orders_usecase
        self.calculate_routes = calculate_routes_usecase
        self.optimize_dispatch = optimize_dispatch_usecase
        self.validate_results = validate_results_usecase
    
    def execute(self) -> DispatchResult:
        """배차 실행 메인 플로우"""
        
        batch_id = self._generate_batch_id()
        start_time = time.time()
        
        try:
            # 1. 데이터 수집 (60초)
            dispatch_data = self.collect_data.execute()
            
            # 2. 주문량 동적 조정 (60초)
            adjusted_data = self.adjust_orders.execute(dispatch_data)
            
            # 3. 경로 계산 (240초)
            route_data = self.calculate_routes.execute(adjusted_data)
            
            # 4. 최적화 실행 (120초)
            optimization_results = self.optimize_dispatch.execute(route_data)
            
            # 5. 결과 검증 (30초)
            final_result = self.validate_results.execute(optimization_results)
            
            execution_time = time.time() - start_time
            
            return DispatchResult(
                batch_id=batch_id,
                results=final_result,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            return DispatchResult(
                batch_id=batch_id,
                error=str(e),
                execution_time=time.time() - start_time,
                success=False
            )
    
    def _generate_batch_id(self) -> str:
        """배치 ID 생성"""
        return f"dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### 3.2 데이터 수집 UseCase

```python
# usecases/collect_data.py
"""
데이터 수집 UseCase
- TMS 데이터베이스에서 배차 데이터 수집
- 데이터 검증 및 정제
- 외부 조건 데이터 수집
"""
from services.data_service import DataService
from services.condition_service import ConditionService
from models.domain.dispatch_data import DispatchData

class CollectDataUseCase:
    """데이터 수집 UseCase"""
    
    def __init__(self, config):
        self.data_service = DataService(config['data'])
        self.condition_service = ConditionService(config['conditions'])
    
    def execute(self) -> DispatchData:
        """데이터 수집 실행"""
        
        # 1. TMS 기본 데이터 수집
        orders = self.data_service.get_pending_orders()
        vehicles = self.data_service.get_available_vehicles()
        regions = self.data_service.get_regions()
        centers = self.data_service.get_centers()
        
        # 2. 데이터 검증
        self._validate_basic_data(orders, vehicles, regions, centers)
        
        # 3. 외부 조건 데이터 수집
        weather_conditions = self.condition_service.get_weather_conditions(regions)
        traffic_conditions = self.condition_service.get_traffic_conditions(regions)
        
        return DispatchData(
            orders=orders,
            vehicles=vehicles,
            regions=regions,
            centers=centers,
            weather_conditions=weather_conditions,
            traffic_conditions=traffic_conditions
        )
    
    def _validate_basic_data(self, orders, vehicles, regions, centers):
        """기본 데이터 검증"""
        if not orders:
            raise ValueError("배차할 주문이 없습니다")
        if not vehicles:
            raise ValueError("사용 가능한 차량이 없습니다")
        if not regions or not centers:
            raise ValueError("권역 또는 센터 정보가 없습니다")
```

### 3.3 주문량 조정 UseCase

```python
# usecases/adjust_orders.py
"""
주문량 조정 UseCase
- 날씨/교통 조건 기반 동적 조정
- 기사 경험도 반영
- 권역별 배차 밀도 최적화
"""
from strategies.adjustment_strategies.adjustment_strategy import AdjustmentStrategyFactory
from models.domain.adjusted_dispatch_data import AdjustedDispatchData

class AdjustOrdersUseCase:
    """주문량 조정 UseCase"""
    
    def __init__(self, config):
        self.strategy_factory = AdjustmentStrategyFactory(config['adjustment'])
    
    def execute(self, dispatch_data: DispatchData) -> AdjustedDispatchData:
        """주문량 조정 실행"""
        
        adjusted_assignments = []
        
        # 권역별로 처리
        regional_orders = self._group_orders_by_region(dispatch_data.orders)
        
        for region_id, region_orders in regional_orders.items():
            # 해당 권역의 차량들
            region_vehicles = [v for v in dispatch_data.vehicles 
                             if v.region_id == region_id]
            
            if not region_vehicles:
                continue
            
            # 조정 전략 선택
            adjustment_strategy = self.strategy_factory.create_strategy(
                region_id, 
                dispatch_data.weather_conditions.get(region_id),
                dispatch_data.traffic_conditions.get(region_id)
            )
            
            # 권역별 주문량 조정 실행
            region_assignment = adjustment_strategy.adjust(
                region_orders, 
                region_vehicles, 
                dispatch_data.regions[region_id]
            )
            
            adjusted_assignments.append(region_assignment)
        
        return AdjustedDispatchData(
            original_data=dispatch_data,
            adjusted_assignments=adjusted_assignments
        )
    
    def _group_orders_by_region(self, orders):
        """주문을 권역별로 그룹화"""
        regional_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in regional_orders:
                regional_orders[region_id] = []
            regional_orders[region_id].append(order)
        return regional_orders
```

## 4. 전략 패턴 구현

### 4.1 알고리즘 선택 전략

```python
# strategies/algorithm_strategies/algorithm_strategy.py
"""
알고리즘 선택 전략 인터페이스 및 구현
- 상황별 최적 알고리즘 선택
- 전략 패턴으로 유연한 확장
"""
from abc import ABC, abstractmethod
from algorithms.base.algorithm_factory import AlgorithmFactory

class AlgorithmStrategy(ABC):
    """알고리즘 선택 전략 인터페이스"""
    
    @abstractmethod
    def select_algorithm(self, problem_features, conditions):
        """알고리즘 선택"""
        pass

class SimpleAlgorithmStrategy(AlgorithmStrategy):
    """단순 알고리즘 전략 - 빠른 처리 우선"""
    
    def __init__(self, config):
        self.algorithm_factory = AlgorithmFactory(config)
    
    def select_algorithm(self, problem_features, conditions):
        # 주문량 기준 단순 선택
        order_count = problem_features['order_count']
        
        if order_count <= 30:
            return self.algorithm_factory.create('nearest_neighbor')
        elif order_count <= 100:
            return self.algorithm_factory.create('capacity_first')
        else:
            return self.algorithm_factory.create('genetic_algorithm', 
                                                {'generations': 50})  # 빠른 설정

class AdvancedAlgorithmStrategy(AlgorithmStrategy):
    """고급 알고리즘 전략 - 품질 우선"""
    
    def __init__(self, config):
        self.algorithm_factory = AlgorithmFactory(config)
        self.complexity_calculator = ComplexityCalculator()
    
    def select_algorithm(self, problem_features, conditions):
        # 복잡도 기반 정교한 선택
        complexity_score = self.complexity_calculator.calculate(
            problem_features, conditions
        )
        
        if complexity_score <= 1.5:
            return self.algorithm_factory.create('nearest_neighbor')
        elif complexity_score <= 2.5:
            return self.algorithm_factory.create('capacity_first')
        elif complexity_score <= 3.0:
            return self.algorithm_factory.create('genetic_algorithm')
        elif complexity_score <= 3.5:
            return self.algorithm_factory.create('simulated_annealing')
        else:
            return self.algorithm_factory.create('large_neighborhood_search')

class EmergencyAlgorithmStrategy(AlgorithmStrategy):
    """비상 알고리즘 전략 - 안정성 우선"""
    
    def __init__(self, config):
        self.algorithm_factory = AlgorithmFactory(config)
    
    def select_algorithm(self, problem_features, conditions):
        # 무조건 안전한 알고리즘 선택
        return self.algorithm_factory.create('nearest_neighbor')

class AlgorithmStrategyFactory:
    """알고리즘 전략 팩토리"""
    
    def __init__(self, config):
        self.config = config
    
    def create_strategy(self, conditions) -> AlgorithmStrategy:
        """조건에 따른 전략 선택"""
        
        weather_severity = conditions.get('weather', {}).get('severity_score', 1.0)
        time_limit = conditions.get('time_limit', 600)  # 10분
        
        # 비상 상황 확인
        if weather_severity >= 4.0 or time_limit < 120:
            return EmergencyAlgorithmStrategy(self.config)
        
        # 시간 여유에 따른 전략 선택
        if time_limit >= 300:  # 5분 이상
            return AdvancedAlgorithmStrategy(self.config)
        else:
            return SimpleAlgorithmStrategy(self.config)
```

### 4.2 API 선택 전략

```python
# strategies/api_strategies/api_strategy.py
"""
API 선택 전략
- 상황별 최적 API 선택
- 장애 대응 및 백업 전략
"""
from abc import ABC, abstractmethod

class APIStrategy(ABC):
    """API 선택 전략 인터페이스"""
    
    @abstractmethod
    def select_api(self, request_count=1):
        """API 선택"""
        pass

class PrimaryAPIStrategy(APIStrategy):
    """주력 API 전략 - 품질 우선"""
    
    def __init__(self, api_config):
        self.api_priority = ['kakao_maps', 'openrouteservice', 'here_maps', 'mapbox']
        self.api_limits = api_config['limits']
        self.current_usage = api_config.get('current_usage', {})
    
    def select_api(self, request_count=1):
        for api_name in self.api_priority:
            if self._is_available(api_name, request_count):
                return api_name
        
        # 모든 API 한도 초과 시 추정 모드
        return 'estimation'
    
    def _is_available(self, api_name, request_count):
        limit = self.api_limits.get(api_name, {}).get('daily', float('inf'))
        usage = self.current_usage.get(api_name, 0)
        return usage + request_count <= limit

class BackupAPIStrategy(APIStrategy):
    """백업 API 전략 - 안정성 우선"""
    
    def __init__(self, api_config):
        # 한도가 큰 순서로 우선순위
        self.api_priority = ['here_maps', 'kakao_maps', 'mapbox', 'openrouteservice']
        self.api_limits = api_config['limits']
        self.current_usage = api_config.get('current_usage', {})
    
    def select_api(self, request_count=1):
        # 백업 전략은 더 보수적
        for api_name in self.api_priority:
            if self._is_safely_available(api_name, request_count):
                return api_name
        
        return 'estimation'
    
    def _is_safely_available(self, api_name, request_count):
        limit = self.api_limits.get(api_name, {}).get('daily', float('inf'))
        usage = self.current_usage.get(api_name, 0)
        # 80% 사용 시 백업으로 간주
        return usage + request_count <= limit * 0.8

class OfflineStrategy(APIStrategy):
    """오프라인 전략 - API 없이 처리"""
    
    def select_api(self, request_count=1):
        return 'estimation'
```

## 5. 서비스 레이어 구현

### 5.1 최적화 서비스

```python
# services/optimization_service.py
"""
최적화 서비스
- 알고리즘 실행 관리
- 성능 모니터링
- 결과 품질 관리
"""
from strategies.algorithm_strategies.algorithm_strategy import AlgorithmStrategyFactory
from utils.calculators.complexity_calculator import ComplexityCalculator

class OptimizationService:
    """최적화 서비스"""
    
    def __init__(self, config):
        self.config = config
        self.algorithm_strategy_factory = AlgorithmStrategyFactory(config['algorithms'])
        self.complexity_calculator = ComplexityCalculator()
    
    def optimize_region(self, region_data, conditions, time_limit=120):
        """권역별 최적화 실행"""
        
        # 1. 문제 특성 분석
        problem_features = self._analyze_problem(region_data)
        
        # 2. 알고리즘 전략 선택
        algorithm_strategy = self.algorithm_strategy_factory.create_strategy({
            **conditions,
            'time_limit': time_limit
        })
        
        # 3. 알고리즘 선택 및 실행
        algorithm = algorithm_strategy.select_algorithm(problem_features, conditions)
        
        # 4. 최적화 실행
        result = algorithm.solve(
            region_data['orders'],
            region_data['vehicles'],
            region_data['distance_matrix']
        )
        
        # 5. 결과 메타데이터 추가
        result.algorithm_name = algorithm.get_algorithm_name()
        result.execution_time = algorithm.execution_time
        result.quality_score = algorithm.solution_quality
        
        return result
    
    def _analyze_problem(self, region_data):
        """문제 특성 분석"""
        return {
            'order_count': len(region_data['orders']),
            'vehicle_count': len(region_data['vehicles']),
            'geographical_spread': self._calculate_geographical_spread(region_data['orders']),
            'capacity_constraints': self._analyze_capacity_constraints(
                region_data['orders'], region_data['vehicles']
            )
        }
    
    def _calculate_geographical_spread(self, orders):
        """지리적 분산도 계산"""
        if len(orders) < 2:
            return 1.0
        
        # 주문 위치들의 분산 계산
        latitudes = [order.latitude for order in orders]
        longitudes = [order.longitude for order in orders]
        
        lat_range = max(latitudes) - min(latitudes)
        lon_range = max(longitudes) - min(longitudes)
        
        # 정규화된 분산도 반환 (1-5 범위)
        spread = (lat_range + lon_range) * 100  # 대략적인 스케일링
        return min(5.0, max(1.0, spread))
```

## 6. 의존성 주입 및 설정 관리

### 6.1 설정 로더

```python
# app/config_loader.py
"""
설정 로더
- 환경별 설정 로드
- 의존성 주입 설정
- 런타임 설정 검증
"""
import os
import yaml
import importlib.util
from pathlib import Path
from config.settings.base_settings import BaseSettings

class ConfigLoader:
    """설정 로더"""
    
    def __init__(self, env=None):
        self.env = env or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(__file__).parent.parent / 'config'
    
    def load_config(self):
        """전체 설정 로드"""
        
        # 기본 설정 로드
        base_settings = BaseSettings()
        config = base_settings.to_dict()
        
        # 환경별 설정 오버라이드
        env_config = self._load_environment_config()
        config.update(env_config)
        
        # 룰 설정 로드
        rules_config = self._load_rules_config()
        config['rules'] = rules_config
        
        # 설정 검증
        self._validate_config(config)
        
        return config
    
    def _load_environment_config(self):
        """환경별 설정 로드"""
        env_file = self.config_dir / 'settings' / f'{self.env}_settings.py'
        if env_file.exists():
            # 동적 import로 환경별 설정 로드
            spec = importlib.util.spec_from_file_location("env_settings", env_file)
            env_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(env_module)
            return env_module.settings
        return {}
    
    def _load_rules_config(self):
        """비즈니스 룰 설정 로드"""
        rules_config = {}
        rules_dir = self.config_dir / 'rules'
        
        for rule_file in rules_dir.glob('*.yaml'):
            rule_name = rule_file.stem
            with open(rule_file, 'r', encoding='utf-8') as f:
                rules_config[rule_name] = yaml.safe_load(f)
        
        return rules_config
    
    def _validate_config(self, config):
        """설정 검증"""
        required_sections = ['data', 'algorithms', 'apis', 'cache']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"필수 설정 섹션이 없습니다: {section}")
```

## 7. 사용 예시 및 확장성

### 7.1 새로운 알고리즘 추가

```python
# algorithms/advanced/new_algorithm.py
"""새로운 알고리즘 추가 예시"""

from algorithms.base.algorithm_interface import AlgorithmInterface

class NewAdvancedAlgorithm(AlgorithmInterface):
    """새로운 고급 알고리즘"""
    
    def solve(self, orders, vehicles, distance_matrix=None):
        # 새로운 알고리즘 로직 구현
        pass
    
    def get_algorithm_name(self):
        return "new_advanced_algorithm"

# algorithms/base/algorithm_factory.py에 등록
ALGORITHM_REGISTRY = {
    # 기존 알고리즘들...
    'new_advanced_algorithm': NewAdvancedAlgorithm,
}
```

### 7.2 새로운 조정 전략 추가

```python
# strategies/adjustment_strategies/new_strategy.py
"""새로운 조정 전략 추가 예시"""

from strategies.adjustment_strategies.adjustment_strategy import AdjustmentStrategy

class NewAdjustmentStrategy(AdjustmentStrategy):
    """새로운 조정 전략"""
    
    def adjust(self, orders, vehicles, region):
        # 새로운 조정 로직 구현
        pass

# strategies/adjustment_strategies/adjustment_strategy.py에 등록
STRATEGY_REGISTRY = {
    # 기존 전략들...
    'new_strategy': NewAdjustmentStrategy,
}
```

### 7.3 새로운 UseCase 추가

```python
# usecases/new_feature.py
"""새로운 기능 UseCase 추가 예시"""

class NewFeatureUseCase:
    """새로운 기능 UseCase"""
    
    def __init__(self, config):
        self.config = config
        # 필요한 서비스들 주입
    
    def execute(self, input_data):
        """새로운 기능 실행"""
        # 비즈니스 로직 구현
        pass

# app/dispatch_app.py에서 사용
class DispatchApplication:
    def _setup_dependencies(self):
        # 기존 UseCase들...
        self.new_feature_usecase = NewFeatureUseCase(self.config)
```

## 8. 테스트 전략

### 8.1 UseCase 테스트

```python
# tests/unit/test_usecases/test_execute_dispatch.py
"""UseCase 단위 테스트"""

import unittest
from unittest.mock import Mock
from usecases.execute_dispatch import ExecuteDispatchUseCase

class TestExecuteDispatchUseCase(unittest.TestCase):
    """배차 실행 UseCase 테스트"""
    
    def setUp(self):
        # Mock UseCase들 생성
        self.mock_collect_data = Mock()
        self.mock_adjust_orders = Mock()
        self.mock_calculate_routes = Mock()
        self.mock_optimize_dispatch = Mock()
        self.mock_validate_results = Mock()
        
        # UseCase 생성
        self.usecase = ExecuteDispatchUseCase(
            self.mock_collect_data,
            self.mock_adjust_orders,
            self.mock_calculate_routes,
            self.mock_optimize_dispatch,
            self.mock_validate_results
        )
    
    def test_execute_success(self):
        """정상적인 배차 실행 테스트"""
        # Given
        self.mock_collect_data.execute.return_value = Mock()
        self.mock_adjust_orders.execute.return_value = Mock()
        
        # When
        result = self.usecase.execute()
        
        # Then
        self.assertTrue(result.success)
        self.assertIsNotNone(result.batch_id)
```

### 8.2 전략 패턴 테스트

```python
# tests/unit/test_strategies/test_algorithm_strategy.py
"""전략 패턴 테스트"""

import unittest
from strategies.algorithm_strategies.algorithm_strategy import SimpleAlgorithmStrategy

class TestAlgorithmStrategy(unittest.TestCase):
    """알고리즘 전략 테스트"""
    
    def setUp(self):
        self.strategy = SimpleAlgorithmStrategy({})
    
    def test_select_algorithm_for_small_orders(self):
        """소규모 주문에 대한 알고리즘 선택 테스트"""
        # Given
        problem_features = {'order_count': 20}
        conditions = {}
        
        # When
        algorithm = self.strategy.select_algorithm(problem_features, conditions)
        
        # Then
        self.assertEqual(algorithm.get_algorithm_name(), 'nearest_neighbor')
```

## 9. 배포 및 운영

### 9.1 Docker 컨테이너화

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### 9.2 환경별 설정

```python
# config/settings/production_settings.py
"""프로덕션 환경 설정"""

settings = {
    'data': {
        'database_url': 'postgresql://prod_user:password@prod_db:5432/tms',
        'connection_pool_size': 20
    },
    'cache': {
        'redis_url': 'redis://prod_redis:6379/0',
        'ttl': 3600
    },
    'apis': {
        'rate_limit_per_minute': 100,
        'timeout': 30
    }
}
```

이 구조로 **main.py는 극도로 간단**해지고, **UseCase별로 독립적인 비즈니스 로직**을 가지며, **전략 패턴으로 유연한 확장**이 가능해집니다! 🚀✨