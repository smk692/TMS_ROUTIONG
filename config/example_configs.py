#!/usr/bin/env python3
"""
TMS 시스템 Config 예시 모음

다양한 비즈니스 시나리오에 맞는 설정 예시들을 제공합니다.
"""

from config.tms_config import tms_config, apply_preset

def apply_small_business_config():
    """소규모 사업체용 설정"""
    print("🏪 소규모 사업체 설정 적용")
    
    # 소규모 설정
    tms_config.set('vehicles.count', 5)
    tms_config.set('vehicles.capacity.volume', 3.0)
    tms_config.set('vehicles.capacity.weight', 500.0)
    tms_config.set('logistics.delivery.max_distance', 8.0)
    tms_config.set('logistics.delivery.points_per_vehicle', 30)
    tms_config.set('constraints.max_working_hours', 6)
    tms_config.set('algorithms.tsp.max_iterations', 100)
    
    print("✅ 소규모 사업체 설정 완료")
    print("   - 차량 5대, 용량 3m³/500kg")
    print("   - 배송 반경 8km, 차량당 30개 배송지")
    print("   - 근무시간 6시간, TSP 100회 반복")

def apply_enterprise_config():
    """대기업용 설정"""
    print("🏢 대기업 설정 적용")
    
    # 대기업 설정
    tms_config.set('vehicles.count', 50)
    tms_config.set('vehicles.capacity.volume', 10.0)
    tms_config.set('vehicles.capacity.weight', 2000.0)
    tms_config.set('logistics.delivery.max_distance', 30.0)
    tms_config.set('logistics.delivery.points_per_vehicle', 100)
    tms_config.set('constraints.max_working_hours', 10)
    tms_config.set('algorithms.tsp.max_iterations', 200)
    tms_config.set('system.api.max_workers', 10)
    
    print("✅ 대기업 설정 완료")
    print("   - 차량 50대, 용량 10m³/2000kg")
    print("   - 배송 반경 30km, 차량당 100개 배송지")
    print("   - 근무시간 10시간, TSP 200회 반복")
    print("   - API 워커 10개 (고성능 처리)")

def apply_same_day_delivery_config():
    """당일 배송 서비스용 설정"""
    print("⚡ 당일 배송 서비스 설정 적용")
    
    # 당일 배송 설정
    tms_config.set('vehicles.count', 20)
    tms_config.set('vehicles.operating_hours.start_hour', 8)
    tms_config.set('vehicles.operating_hours.end_hour', 22)  # 14시간 운영
    tms_config.set('vehicles.average_speed', 40.0)  # 빠른 배송
    tms_config.set('logistics.delivery.max_distance', 20.0)
    tms_config.set('logistics.delivery.service_time', 3)  # 빠른 서비스
    tms_config.set('logistics.delivery.points_per_vehicle', 40)
    tms_config.set('constraints.max_working_hours', 12)
    tms_config.set('constraints.target_efficiency', 0.05)  # 높은 효율성
    
    print("✅ 당일 배송 설정 완료")
    print("   - 차량 20대, 8시-22시 운영 (14시간)")
    print("   - 평균 속도 40km/h, 서비스 시간 3분")
    print("   - 배송 반경 20km, 높은 효율성 목표")

def apply_rural_delivery_config():
    """농촌 지역 배송용 설정"""
    print("🌾 농촌 지역 배송 설정 적용")
    
    # 농촌 배송 설정
    tms_config.set('vehicles.count', 8)
    tms_config.set('vehicles.capacity.volume', 15.0)  # 대용량
    tms_config.set('vehicles.capacity.weight', 3000.0)
    tms_config.set('vehicles.average_speed', 25.0)  # 느린 속도
    tms_config.set('logistics.delivery.max_distance', 50.0)  # 넓은 반경
    tms_config.set('logistics.delivery.service_time', 10)  # 긴 서비스 시간
    tms_config.set('logistics.delivery.points_per_vehicle', 20)  # 적은 배송지
    tms_config.set('constraints.max_working_hours', 12)
    tms_config.set('constraints.allow_overtime', True)
    
    print("✅ 농촌 지역 배송 설정 완료")
    print("   - 차량 8대, 대용량 15m³/3000kg")
    print("   - 평균 속도 25km/h, 배송 반경 50km")
    print("   - 차량당 20개 배송지, 초과근무 허용")

def apply_eco_friendly_config():
    """친환경 배송용 설정"""
    print("🌱 친환경 배송 설정 적용")
    
    # 친환경 설정
    tms_config.set('vehicles.count', 12)
    tms_config.set('vehicles.capacity.volume', 4.0)  # 소형 차량
    tms_config.set('vehicles.capacity.weight', 800.0)
    tms_config.set('vehicles.cost_per_km', 200.0)  # 저비용 (전기차)
    tms_config.set('vehicles.average_speed', 25.0)  # 에코 드라이빙
    tms_config.set('logistics.delivery.max_distance', 12.0)
    tms_config.set('logistics.delivery.points_per_vehicle', 35)
    tms_config.set('constraints.target_efficiency', 0.15)  # 높은 효율성
    tms_config.set('algorithms.tsp.max_iterations', 250)  # 최적화 강화
    
    print("✅ 친환경 배송 설정 완료")
    print("   - 소형 차량 12대, 4m³/800kg")
    print("   - km당 200원 (전기차), 에코 드라이빙")
    print("   - 높은 효율성, 최적화 강화")

def apply_peak_season_config():
    """성수기용 설정"""
    print("🎄 성수기 설정 적용")
    
    # 성수기 설정
    tms_config.set('vehicles.count', 30)
    tms_config.set('vehicles.operating_hours.start_hour', 5)  # 일찍 시작
    tms_config.set('vehicles.operating_hours.end_hour', 23)  # 늦게 끝
    tms_config.set('logistics.delivery.max_distance', 25.0)
    tms_config.set('logistics.delivery.points_per_vehicle', 80)  # 많은 배송지
    tms_config.set('constraints.max_working_hours', 15)  # 긴 근무
    tms_config.set('constraints.allow_overtime', True)
    tms_config.set('system.api.max_workers', 8)  # 고성능 처리
    
    print("✅ 성수기 설정 완료")
    print("   - 차량 30대, 5시-23시 운영 (18시간)")
    print("   - 차량당 80개 배송지, 15시간 근무")
    print("   - 초과근무 허용, 고성능 API 처리")

def show_all_examples():
    """모든 예시 설정 보기"""
    print("📋 TMS Config 예시 모음")
    print("=" * 50)
    print("1. 소규모 사업체용")
    print("2. 대기업용")
    print("3. 당일 배송 서비스용")
    print("4. 농촌 지역 배송용")
    print("5. 친환경 배송용")
    print("6. 성수기용")
    print()
    print("사용법:")
    print("  from config.example_configs import apply_small_business_config")
    print("  apply_small_business_config()")
    print("  python run_all.py")

if __name__ == "__main__":
    show_all_examples() 