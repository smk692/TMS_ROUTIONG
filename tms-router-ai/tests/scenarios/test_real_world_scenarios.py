"""
실제 TMS 시나리오 테스트

실제 물류 업계에서 발생할 수 있는 복잡한 배차 시나리오를 테스트합니다.
"""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tests import (
    generate_test_vehicle, generate_test_order, generate_test_constraints,
    generate_seoul_locations, validate_optimization_response
)


class TestRealWorldScenarios:
    """실제 물류 시나리오 테스트"""
    
    def test_seoul_delivery_rush_hour(self):
        """서울 출퇴근 시간 배송 시나리오"""
        # 출퇴근 시간대 제약
        rush_hour_constraints = generate_test_constraints(
            working_hours={"start": "06:00", "end": "22:00"},
            avoid_tolls=True,  # 출퇴근 시간 톨게이트 회피
            optimize_for="time"  # 시간 우선 최적화
        )
        
        # 서울 주요 지역에 위치한 차량들
        seoul_locations = generate_seoul_locations(5)
        vehicles = []
        for i, location in enumerate(seoul_locations[:3]):
            vehicle = generate_test_vehicle(
                vehicle_id=f"SEOUL_V{i+1:03d}",
                current_location=location,
                capacity_tons=3.0,
                hourly_cost=30000.0  # 출퇴근 시간 높은 비용
            )
            vehicles.append(vehicle)
        
        # 다양한 우선순위의 주문들
        orders = []
        locations = generate_seoul_locations(15)
        priorities = ["URGENT", "HIGH", "MEDIUM", "LOW", "MEDIUM"]
        
        for i in range(10):
            pickup_loc = locations[i]
            delivery_loc = locations[i + 5]
            
            order = generate_test_order(
                order_id=f"RUSH_O{i+1:03d}",
                pickup_location=pickup_loc,
                delivery_location=delivery_loc,
                weight_tons=0.5 + (i % 3) * 0.5,  # 0.5~2.0t
                priority=priorities[i % len(priorities)]
            )
            
            # 긴급 주문은 시간창 설정
            if order["priority"] == "URGENT":
                order["time_window"] = {
                    "start": "08:00",
                    "end": "10:00"
                }
            
            orders.append(order)
        
        scenario = {
            "scenario_name": "서울_출퇴근시간_배송",
            "description": "출퇴근 시간대 서울 시내 긴급 배송 최적화",
            "vehicles": vehicles,
            "orders": orders,
            "constraints": rush_hour_constraints
        }
        
        # 시나리오 검증
        assert len(scenario["vehicles"]) == 3
        assert len(scenario["orders"]) == 10
        assert scenario["constraints"]["avoid_tolls"] is True
        
        # 긴급 주문이 있는지 확인
        urgent_orders = [o for o in orders if o.get("priority") == "URGENT"]
        assert len(urgent_orders) > 0
    
    def test_cold_chain_logistics(self):
        """콜드체인 물류 시나리오"""
        # 냉장/냉동 운송 차량
        cold_vehicles = [
            generate_test_vehicle(
                vehicle_id="COLD_001",
                capacity_tons=5.0,
                special_capabilities=["refrigerated", "temperature_controlled"],
                fuel_efficiency_kmpl=8.0,  # 냉장차량은 연비가 낮음
                hourly_cost=50000.0  # 특수 차량 고비용
            ),
            generate_test_vehicle(
                vehicle_id="COLD_002", 
                capacity_tons=3.0,
                special_capabilities=["frozen", "temperature_controlled"],
                fuel_efficiency_kmpl=7.0,
                hourly_cost=55000.0
            )
        ]
        
        # 온도 민감 상품 주문들
        cold_orders = []
        locations = generate_seoul_locations(8)
        
        for i in range(6):
            order = generate_test_order(
                order_id=f"COLD_O{i+1:03d}",
                pickup_location=locations[i],
                delivery_location=locations[6+i%2],
                weight_tons=1.0 + i * 0.3,
                special_requirements=["temperature_controlled"]
            )
            
            # 교대로 냉장/냉동 요구사항 설정
            if i % 2 == 0:
                order["special_requirements"].append("refrigerated")
                order["temperature_range"] = {"min": 2, "max": 8}  # 냉장
            else:
                order["special_requirements"].append("frozen")
                order["temperature_range"] = {"min": -25, "max": -18}  # 냉동
            
            # 짧은 배송 시간창 (상품 특성상)
            order["time_window"] = {
                "start": "09:00",
                "end": "15:00"
            }
            
            cold_orders.append(order)
        
        # 콜드체인 제약조건
        cold_constraints = generate_test_constraints(
            max_duration_hours=6,  # 온도 유지 한계
            optimize_for="time",   # 신선도 우선
            special_rules={
                "temperature_monitoring": True,
                "cross_contamination_prevention": True
            }
        )
        
        scenario = {
            "scenario_name": "콜드체인_물류",
            "description": "냉장/냉동 상품의 온도 유지 배송 최적화",
            "vehicles": cold_vehicles,
            "orders": cold_orders,
            "constraints": cold_constraints
        }
        
        # 특수 요구사항 검증
        assert all(
            "temperature_controlled" in v["special_capabilities"] 
            for v in cold_vehicles
        )
        assert all(
            "temperature_controlled" in o["special_requirements"]
            for o in cold_orders
        )
    
    def test_multi_day_planning(self):
        """다중일 계획 시나리오"""
        # 3일간의 배송 계획
        planning_days = 3
        base_date = datetime(2024, 1, 1)
        
        vehicles = [
            generate_test_vehicle(f"MULTI_V{i+1:03d}", capacity_tons=4.0)
            for i in range(4)
        ]
        
        # 각 날짜별 주문 생성
        all_orders = []
        locations = generate_seoul_locations(20)
        
        for day in range(planning_days):
            current_date = base_date + timedelta(days=day)
            daily_orders = []
            
            # 하루 10개 주문
            for i in range(10):
                order = generate_test_order(
                    order_id=f"MULTI_D{day+1}_O{i+1:03d}",
                    pickup_location=locations[i],
                    delivery_location=locations[i+10],
                    weight_tons=1.0 + (i % 4) * 0.5
                )
                
                # 날짜별 시간창 설정
                order["scheduled_date"] = current_date.strftime("%Y-%m-%d")
                order["time_window"] = {
                    "start": "09:00",
                    "end": "17:00"
                }
                
                # 3일차에는 높은 우선순위 주문 추가
                if day == 2:
                    order["priority"] = "HIGH"
                
                daily_orders.append(order)
            
            all_orders.extend(daily_orders)
        
        # 다중일 제약조건
        multi_day_constraints = generate_test_constraints(
            max_distance_km=200,  # 하루 최대 운행거리
            optimize_for="cost",   # 비용 최적화
            multi_day_rules={
                "vehicle_rest_hours": 12,  # 차량 휴식시간
                "driver_working_hours_per_day": 8,
                "weekend_operations": False
            }
        )
        
        scenario = {
            "scenario_name": "다중일_배송계획",
            "description": f"{planning_days}일간의 통합 배송 계획 최적화",
            "planning_horizon_days": planning_days,
            "vehicles": vehicles,
            "orders": all_orders,
            "constraints": multi_day_constraints
        }
        
        # 다중일 시나리오 검증
        assert len(scenario["orders"]) == planning_days * 10
        assert scenario["planning_horizon_days"] == 3
        
        # 날짜별 주문 분포 확인
        date_counts = {}
        for order in all_orders:
            date = order["scheduled_date"]
            date_counts[date] = date_counts.get(date, 0) + 1
        
        assert len(date_counts) == planning_days
        assert all(count == 10 for count in date_counts.values())
    
    def test_peak_season_logistics(self):
        """성수기 물류 시나리오 (추석, 크리스마스 등)"""
        # 성수기 특성: 주문량 급증, 차량 부족, 높은 비용
        
        # 일반 차량 + 임시 렌탈 차량
        regular_vehicles = [
            generate_test_vehicle(
                f"REG_V{i+1:03d}",
                capacity_tons=5.0,
                hourly_cost=25000.0
            ) for i in range(3)
        ]
        
        rental_vehicles = [
            generate_test_vehicle(
                f"RENTAL_V{i+1:03d}",
                capacity_tons=3.0,
                hourly_cost=40000.0,  # 렌탈 차량 높은 비용
                special_capabilities=["temporary_rental"]
            ) for i in range(2)
        ]
        
        all_vehicles = regular_vehicles + rental_vehicles
        
        # 대량 주문 (성수기 특성)
        peak_orders = []
        locations = generate_seoul_locations(30)
        
        for i in range(25):  # 평소보다 많은 주문
            order = generate_test_order(
                order_id=f"PEAK_O{i+1:03d}",
                pickup_location=locations[i],
                delivery_location=locations[25+i%5],
                weight_tons=0.5 + (i % 5) * 0.4,  # 다양한 중량
                priority="HIGH" if i < 10 else "MEDIUM"  # 초기 주문 높은 우선순위
            )
            
            # 짧은 배송 시간창 (고객 기대치 높음)
            if i < 5:  # 초긴급 주문
                order["time_window"] = {"start": "09:00", "end": "12:00"}
            elif i < 15:  # 당일 배송
                order["time_window"] = {"start": "09:00", "end": "18:00"}
            else:  # 일반 주문
                order["time_window"] = {"start": "09:00", "end": "21:00"}
            
            peak_orders.append(order)
        
        # 성수기 제약조건
        peak_constraints = generate_test_constraints(
            max_duration_hours=12,  # 연장 운행
            optimize_for="time",    # 시간 우선 (고객 만족도)
            peak_season_rules={
                "overtime_allowed": True,
                "weekend_delivery": True,
                "express_delivery_premium": 1.5,
                "rental_vehicle_priority": "last_resort"
            }
        )
        
        scenario = {
            "scenario_name": "성수기_물류_대응",
            "description": "추석/크리스마스 등 성수기 대량 주문 처리",
            "vehicles": all_vehicles,
            "orders": peak_orders,
            "constraints": peak_constraints,
            "special_conditions": {
                "season": "peak",
                "expected_volume_increase": "250%",
                "customer_expectation": "high"
            }
        }
        
        # 성수기 특성 검증
        assert len(scenario["orders"]) == 25  # 대량 주문
        assert len(scenario["vehicles"]) == 5   # 추가 차량 투입
        
        # 렌탈 차량 포함 확인
        rental_count = len([v for v in all_vehicles 
                           if "temporary_rental" in v.get("special_capabilities", [])])
        assert rental_count == 2
        
        # 긴급 주문 비율 확인
        urgent_orders = [o for o in peak_orders if o.get("priority") == "HIGH"]
        assert len(urgent_orders) >= 10  # 40% 이상이 높은 우선순위
    
    def test_emergency_delivery_scenario(self):
        """응급 배송 시나리오"""
        # 의료용품, 긴급 부품 등의 응급 배송
        
        # 응급 배송 전용 차량
        emergency_vehicles = [
            generate_test_vehicle(
                "EMERGENCY_001",
                capacity_tons=2.0,
                special_capabilities=["emergency", "priority_access", "medical_certified"],
                hourly_cost=80000.0,  # 높은 비용
                fuel_efficiency_kmpl=10.0
            )
        ]
        
        # 일반 차량 (보조)
        regular_vehicles = [
            generate_test_vehicle(f"REG_E{i+1:03d}", capacity_tons=3.0)
            for i in range(2)
        ]
        
        all_vehicles = emergency_vehicles + regular_vehicles
        
        # 응급 주문들
        emergency_orders = []
        locations = generate_seoul_locations(8)
        
        # 최고 우선순위 응급 주문
        critical_order = generate_test_order(
            order_id="EMERGENCY_001",
            pickup_location=locations[0],  # 병원
            delivery_location=locations[4],  # 응급실
            weight_tons=0.1,  # 의료용품
            priority="CRITICAL",
            special_requirements=["emergency", "medical", "temperature_controlled"]
        )
        critical_order["time_window"] = {"start": "즉시", "end": "30분이내"}
        critical_order["max_delivery_time_minutes"] = 30
        emergency_orders.append(critical_order)
        
        # 긴급 부품 배송
        for i in range(3):
            order = generate_test_order(
                order_id=f"URGENT_PART_{i+1:03d}",
                pickup_location=locations[i+1],
                delivery_location=locations[i+5],
                weight_tons=0.5 + i * 0.3,
                priority="URGENT",
                special_requirements=["fragile", "time_critical"]
            )
            order["time_window"] = {"start": "즉시", "end": "2시간이내"}
            order["max_delivery_time_minutes"] = 120
            emergency_orders.append(order)
        
        # 응급 배송 제약조건
        emergency_constraints = generate_test_constraints(
            optimize_for="time",  # 시간이 최우선
            emergency_rules={
                "traffic_priority": True,
                "route_preemption": True,
                "cost_no_limit": True,
                "real_time_tracking": True
            }
        )
        emergency_constraints["max_duration_hours"] = 24  # 24시간 대기체제
        
        scenario = {
            "scenario_name": "응급_배송_대응",
            "description": "의료용품, 긴급부품 등의 응급 배송 최적화",
            "vehicles": all_vehicles,
            "orders": emergency_orders,
            "constraints": emergency_constraints,
            "sla_requirements": {
                "critical_delivery_time": "30분",
                "urgent_delivery_time": "2시간",
                "success_rate_target": "99.9%"
            }
        }
        
        # 응급 시나리오 검증
        critical_orders = [o for o in emergency_orders if o.get("priority") == "CRITICAL"]
        assert len(critical_orders) == 1
        assert critical_orders[0]["max_delivery_time_minutes"] == 30
        
        # 응급 차량 확인
        emergency_capable = [v for v in all_vehicles 
                           if "emergency" in v.get("special_capabilities", [])]
        assert len(emergency_capable) == 1
    
    @pytest.mark.integration
    def test_integrated_complex_scenario(self):
        """통합 복합 시나리오"""
        # 여러 특수 상황이 동시에 발생하는 복합 시나리오
        
        # 다양한 유형의 차량들
        mixed_vehicles = [
            # 일반 차량
            generate_test_vehicle("MIXED_001", capacity_tons=5.0),
            # 냉장 차량
            generate_test_vehicle(
                "COLD_001", 
                capacity_tons=3.0,
                special_capabilities=["refrigerated"]
            ),
            # 응급 차량
            generate_test_vehicle(
                "EMERGENCY_001",
                capacity_tons=2.0,
                special_capabilities=["emergency", "medical_certified"]
            ),
            # 대형 차량
            generate_test_vehicle(
                "LARGE_001",
                capacity_tons=10.0,
                special_capabilities=["oversized_cargo"]
            )
        ]
        
        # 복합 주문들
        complex_orders = []
        locations = generate_seoul_locations(15)
        
        # 1. 응급 의료용품
        complex_orders.append(generate_test_order(
            "COMPLEX_EMERGENCY",
            pickup_location=locations[0],
            delivery_location=locations[10],
            weight_tons=0.1,
            priority="CRITICAL",
            special_requirements=["emergency", "medical"]
        ))
        
        # 2. 냉장 식품
        complex_orders.append(generate_test_order(
            "COMPLEX_COLD",
            pickup_location=locations[1],
            delivery_location=locations[11],
            weight_tons=2.0,
            priority="HIGH",
            special_requirements=["refrigerated", "temperature_controlled"]
        ))
        
        # 3. 대형 화물
        complex_orders.append(generate_test_order(
            "COMPLEX_LARGE",
            pickup_location=locations[2],
            delivery_location=locations[12],
            weight_tons=8.0,
            priority="MEDIUM",
            special_requirements=["oversized_cargo", "special_handling"]
        ))
        
        # 4-8. 일반 주문들
        for i in range(3, 8):
            complex_orders.append(generate_test_order(
                f"COMPLEX_NORMAL_{i}",
                pickup_location=locations[i],
                delivery_location=locations[i+7],
                weight_tons=1.0 + i * 0.3,
                priority="MEDIUM"
            ))
        
        # 복합 제약조건
        complex_constraints = generate_test_constraints(
            optimize_for="balanced",  # 균형잡힌 최적화
            complex_rules={
                "priority_ordering": True,
                "capability_matching": True,
                "cost_efficiency": True,
                "time_windows_strict": True
            }
        )
        
        scenario = {
            "scenario_name": "통합_복합_시나리오",
            "description": "다양한 특수 상황이 동시 발생하는 복합 배송 최적화",
            "vehicles": mixed_vehicles,
            "orders": complex_orders,
            "constraints": complex_constraints,
            "complexity_factors": [
                "emergency_delivery",
                "cold_chain",
                "oversized_cargo",
                "mixed_priorities",
                "capability_constraints"
            ]
        }
        
        # 복합 시나리오 검증
        assert len(scenario["vehicles"]) == 4
        assert len(scenario["orders"]) == 8
        assert len(scenario["complexity_factors"]) == 5
        
        # 특수 요구사항 매칭 확인
        emergency_orders = [o for o in complex_orders 
                          if "emergency" in o.get("special_requirements", [])]
        emergency_vehicles = [v for v in mixed_vehicles 
                            if "emergency" in v.get("special_capabilities", [])]
        
        assert len(emergency_orders) == 1
        assert len(emergency_vehicles) == 1
        
        print(f"복합 시나리오 구성:")
        print(f"  - 차량 유형: {len(set([tuple(v.get('special_capabilities', [])) for v in mixed_vehicles]))}개")
        print(f"  - 주문 유형: {len(set([tuple(o.get('special_requirements', [])) for o in complex_orders]))}개")
        print(f"  - 복잡도 요소: {len(scenario['complexity_factors'])}개") 