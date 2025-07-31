"""
도메인 엔티티 단위 테스트

Vehicle, DeliveryOrder, Route 등 핵심 도메인 엔티티의 
비즈니스 로직을 테스트합니다.
"""
import pytest
from datetime import datetime, timedelta
from src.domain.entities.vehicle import Vehicle, VehicleStatus
from src.domain.entities.delivery_order import DeliveryOrder, OrderStatus
from src.domain.entities.route import Route, RouteStatus
from src.domain.value_objects.coordinate import Coordinate
from src.domain.value_objects.time_window import TimeWindow


class TestVehicle:
    """Vehicle 엔티티 테스트"""
    
    def test_vehicle_creation(self):
        """차량 생성 테스트"""
        location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=location,
            driver_id="D001"
        )
        
        assert vehicle.vehicle_id == "V001"
        assert vehicle.capacity_tons == 5.0
        assert vehicle.current_location == location
        assert vehicle.driver_id == "D001"
        assert vehicle.status == VehicleStatus.AVAILABLE
        assert vehicle.special_capabilities == []
    
    def test_vehicle_capability_management(self):
        """차량 특수 능력 관리 테스트"""
        location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=location
        )
        
        # 능력 추가
        vehicle.add_capability("refrigerated")
        vehicle.add_capability("oversized_cargo")
        
        assert "refrigerated" in vehicle.special_capabilities
        assert "oversized_cargo" in vehicle.special_capabilities
        assert len(vehicle.special_capabilities) == 2
        
        # 중복 추가 방지
        vehicle.add_capability("refrigerated")
        assert len(vehicle.special_capabilities) == 2
        
        # 능력 제거
        vehicle.remove_capability("refrigerated")
        assert "refrigerated" not in vehicle.special_capabilities
        assert len(vehicle.special_capabilities) == 1
    
    def test_vehicle_availability_check(self):
        """차량 가용성 확인 테스트"""
        location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=location
        )
        
        # 기본적으로 가용
        assert vehicle.is_available()
        
        # 상태 변경 후 불가용
        vehicle.status = VehicleStatus.IN_TRANSIT
        assert not vehicle.is_available()
        
        vehicle.status = VehicleStatus.MAINTENANCE
        assert not vehicle.is_available()
    
    def test_vehicle_cost_calculation(self):
        """차량 비용 계산 테스트"""
        location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=location,
            hourly_cost=20000.0,
            fuel_efficiency_kmpl=10.0,
            fuel_cost_per_liter=1500.0
        )
        
        # 시간 기반 비용
        time_cost = vehicle.calculate_time_cost(3.5)  # 3.5시간
        assert time_cost == 70000.0  # 20000 * 3.5
        
        # 거리 기반 연료비
        fuel_cost = vehicle.calculate_fuel_cost(100.0)  # 100km
        expected_fuel_cost = (100 / 10) * 1500  # 10L * 1500원
        assert fuel_cost == expected_fuel_cost


class TestDeliveryOrder:
    """DeliveryOrder 엔티티 테스트"""
    
    def test_order_creation(self):
        """주문 생성 테스트"""
        pickup = Coordinate(latitude=37.5547, longitude=126.9706)
        delivery = Coordinate(latitude=37.5172, longitude=127.0473)
        time_window = TimeWindow(
            start_time=datetime(2024, 1, 1, 9, 0),
            end_time=datetime(2024, 1, 1, 17, 0)
        )
        
        order = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=2.5,
            volume_cbm=3.0,
            priority="HIGH",
            time_window=time_window,
            customer_id="C001"
        )
        
        assert order.order_id == "O001"
        assert order.pickup_location == pickup
        assert order.delivery_location == delivery
        assert order.weight_tons == 2.5
        assert order.volume_cbm == 3.0
        assert order.priority == "HIGH"
        assert order.time_window == time_window
        assert order.customer_id == "C001"
        assert order.status == OrderStatus.PENDING
    
    def test_order_distance_calculation(self):
        """주문 거리 계산 테스트"""
        pickup = Coordinate(latitude=37.5665, longitude=126.9780)  # 서울시청
        delivery = Coordinate(latitude=37.5172, longitude=127.0473)  # 강남구
        
        order = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=1.0
        )
        
        distance = order.calculate_straight_distance()
        assert isinstance(distance, float)
        assert distance > 0
        # 서울시청-강남구 직선거리는 대략 7-10km
        assert 5 < distance < 15
    
    def test_order_time_window_validation(self):
        """주문 시간창 검증 테스트"""
        pickup = Coordinate(latitude=37.5547, longitude=126.9706)
        delivery = Coordinate(latitude=37.5172, longitude=127.0473)
        
        # 유효한 시간창
        valid_time_window = TimeWindow(
            start_time=datetime(2024, 1, 1, 9, 0),
            end_time=datetime(2024, 1, 1, 17, 0)
        )
        
        order = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=1.0,
            time_window=valid_time_window
        )
        
        # 시간창 내 시간 확인
        delivery_time = datetime(2024, 1, 1, 14, 0)
        assert order.is_within_time_window(delivery_time)
        
        # 시간창 밖 시간 확인
        late_delivery_time = datetime(2024, 1, 1, 20, 0)
        assert not order.is_within_time_window(late_delivery_time)


class TestRoute:
    """Route 엔티티 테스트"""
    
    def test_route_creation(self):
        """경로 생성 테스트"""
        vehicle_location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=vehicle_location
        )
        
        route = Route(
            route_id="R001",
            vehicle=vehicle
        )
        
        assert route.route_id == "R001"
        assert route.vehicle == vehicle
        assert route.status == RouteStatus.PLANNED
        assert len(route.orders) == 0
        assert len(route.segments) == 0
    
    def test_route_order_management(self):
        """경로 주문 관리 테스트"""
        vehicle_location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=vehicle_location
        )
        
        route = Route(
            route_id="R001",
            vehicle=vehicle
        )
        
        # 주문 추가
        pickup = Coordinate(latitude=37.5547, longitude=126.9706)
        delivery = Coordinate(latitude=37.5172, longitude=127.0473)
        order = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=2.0
        )
        
        route.add_order(order)
        assert len(route.orders) == 1
        assert order in route.orders
        
        # 주문 제거
        route.remove_order(order)
        assert len(route.orders) == 0
        assert order not in route.orders
    
    def test_route_capacity_validation(self):
        """경로 용량 검증 테스트"""
        vehicle_location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=vehicle_location
        )
        
        route = Route(
            route_id="R001",
            vehicle=vehicle
        )
        
        # 용량 초과 주문 추가 시도
        pickup = Coordinate(latitude=37.5547, longitude=126.9706)
        delivery = Coordinate(latitude=37.5172, longitude=127.0473)
        heavy_order = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=6.0  # 차량 용량(5.0t) 초과
        )
        
        # 용량 초과 확인
        assert not route.can_accommodate_order(heavy_order)
        
        # 적정 용량 주문
        normal_order = DeliveryOrder(
            order_id="O002",
            pickup_location=pickup,
            delivery_location=delivery,
            weight_tons=3.0
        )
        
        assert route.can_accommodate_order(normal_order)
    
    def test_route_total_calculations(self):
        """경로 총계 계산 테스트"""
        vehicle_location = Coordinate(latitude=37.5665, longitude=126.9780)
        vehicle = Vehicle(
            vehicle_id="V001",
            capacity_tons=5.0,
            current_location=vehicle_location
        )
        
        route = Route(
            route_id="R001",
            vehicle=vehicle
        )
        
        # 여러 주문 추가
        pickup1 = Coordinate(latitude=37.5547, longitude=126.9706)
        delivery1 = Coordinate(latitude=37.5172, longitude=127.0473)
        order1 = DeliveryOrder(
            order_id="O001",
            pickup_location=pickup1,
            delivery_location=delivery1,
            weight_tons=2.0
        )
        
        pickup2 = Coordinate(latitude=37.5735, longitude=126.9788)
        delivery2 = Coordinate(latitude=37.6022, longitude=127.0163)
        order2 = DeliveryOrder(
            order_id="O002",
            pickup_location=pickup2,
            delivery_location=delivery2,
            weight_tons=1.5
        )
        
        route.add_order(order1)
        route.add_order(order2)
        
        # 총 중량 계산
        total_weight = route.calculate_total_weight()
        assert total_weight == 3.5  # 2.0 + 1.5
        
        # 총 주문 수
        assert route.get_order_count() == 2


class TestCoordinate:
    """Coordinate 값 객체 테스트"""
    
    def test_coordinate_creation(self):
        """좌표 생성 테스트"""
        coord = Coordinate(latitude=37.5665, longitude=126.9780)
        assert coord.latitude == 37.5665
        assert coord.longitude == 126.9780
    
    def test_coordinate_validation(self):
        """좌표 검증 테스트"""
        # 유효한 좌표
        valid_coord = Coordinate(latitude=37.5665, longitude=126.9780)
        assert valid_coord.is_valid()
        
        # 위도 범위 초과
        with pytest.raises(ValueError):
            Coordinate(latitude=91.0, longitude=126.9780)
        
        # 경도 범위 초과
        with pytest.raises(ValueError):
            Coordinate(latitude=37.5665, longitude=181.0)
    
    def test_distance_calculation(self):
        """거리 계산 테스트"""
        coord1 = Coordinate(latitude=37.5665, longitude=126.9780)  # 서울시청
        coord2 = Coordinate(latitude=37.5172, longitude=127.0473)  # 강남구
        
        distance = coord1.distance_to(coord2)
        assert isinstance(distance, float)
        assert distance > 0
        # 서울시청-강남구 직선거리는 대략 7-10km
        assert 5 < distance < 15


class TestTimeWindow:
    """TimeWindow 값 객체 테스트"""
    
    def test_time_window_creation(self):
        """시간창 생성 테스트"""
        start_time = datetime(2024, 1, 1, 9, 0)
        end_time = datetime(2024, 1, 1, 17, 0)
        time_window = TimeWindow(start_time=start_time, end_time=end_time)
        
        assert time_window.start_time == start_time
        assert time_window.end_time == end_time
    
    def test_time_window_validation(self):
        """시간창 검증 테스트"""
        start_time = datetime(2024, 1, 1, 9, 0)
        end_time = datetime(2024, 1, 1, 17, 0)
        
        # 유효한 시간창
        valid_window = TimeWindow(start_time=start_time, end_time=end_time)
        assert valid_window.is_valid()
        
        # 시작 시간이 종료 시간보다 늦은 경우
        with pytest.raises(ValueError):
            TimeWindow(start_time=end_time, end_time=start_time)
    
    def test_time_within_window(self):
        """시간창 내 시간 확인 테스트"""
        start_time = datetime(2024, 1, 1, 9, 0)
        end_time = datetime(2024, 1, 1, 17, 0)
        time_window = TimeWindow(start_time=start_time, end_time=end_time)
        
        # 시간창 내 시간
        within_time = datetime(2024, 1, 1, 14, 0)
        assert time_window.contains(within_time)
        
        # 시간창 밖 시간
        outside_time = datetime(2024, 1, 1, 20, 0)
        assert not time_window.contains(outside_time)
        
        # 경계 시간
        assert time_window.contains(start_time)
        assert time_window.contains(end_time)
    
    def test_time_window_duration(self):
        """시간창 지속시간 계산 테스트"""
        start_time = datetime(2024, 1, 1, 9, 0)
        end_time = datetime(2024, 1, 1, 17, 0)
        time_window = TimeWindow(start_time=start_time, end_time=end_time)
        
        duration = time_window.duration_hours()
        assert duration == 8.0  # 9시부터 17시까지 8시간 