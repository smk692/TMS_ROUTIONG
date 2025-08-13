#!/usr/bin/env python3
"""
대용량 데이터 OR-Tools VRP 최적화
500개 주문 처리 성능 개선
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def optimize_anyang_center():
    """안양 센터 500개 주문 최적화 방안"""
    
    print("=== 안양 센터 대용량 처리 최적화 방안 ===\n")
    
    print("🚨 현재 상황:")
    print("- 주문: 500개")
    print("- 차량: 2대")
    print("- 결과: 300초 타임아웃")
    print("- 주문/차량 비율: 250:1 (과부하)")
    
    print("\n💡 즉시 적용 가능한 해결책:")
    print("1. 차량 수 증가")
    print("2. VRP 솔버 최적화 설정")
    print("3. 클러스터링 강화")
    print("4. 시간 제한 조정")
    
    return [
        "add_vehicles_anyang",
        "optimize_vrp_settings", 
        "enhance_clustering",
        "adjust_time_limits"
    ]

def add_vehicles_anyang():
    """안양 센터 차량 추가"""
    
    print("\n=== 1. 차량 추가 방안 ===")
    print("현재: 2대 → 권장: 8-10대")
    print("이유: 500개 주문을 2대로는 물리적으로 불가능")
    
    sql_script = """
-- 안양 센터 차량 추가 (2대 → 10대)
INSERT INTO vehicles (id, center_id, region_id, vehicle_type, driver_name, max_capacity, safe_capacity, status, experience_months) VALUES
('VEH_ANYANG_003', 'CENTER_ANYANG', 'REGION_ANYANG_01', 'TOP_CAR', '안양기사003', 15, 12, 'ACTIVE', 24),
('VEH_ANYANG_004', 'CENTER_ANYANG', 'REGION_ANYANG_01', 'CARGO', '안양기사004', 25, 20, 'ACTIVE', 18),
('VEH_ANYANG_005', 'CENTER_ANYANG', 'REGION_ANYANG_01', 'TOP_CAR', '안양기사005', 15, 12, 'ACTIVE', 36),
('VEH_ANYANG_006', 'CENTER_ANYANG', 'REGION_ANYANG_02', 'TOP_CAR', '안양기사006', 15, 12, 'ACTIVE', 12),
('VEH_ANYANG_007', 'CENTER_ANYANG', 'REGION_ANYANG_02', 'CARGO', '안양기사007', 40, 32, 'ACTIVE', 48),
('VEH_ANYANG_008', 'CENTER_ANYANG', 'REGION_ANYANG_02', 'TOP_CAR', '안양기사008', 15, 12, 'ACTIVE', 30),
('VEH_ANYANG_009', 'CENTER_ANYANG', 'REGION_ANYANG_03', 'OTHER', '안양기사009', 8, 6, 'ACTIVE', 6),
('VEH_ANYANG_010', 'CENTER_ANYANG', 'REGION_ANYANG_03', 'CARGO', '안양기사010', 25, 20, 'ACTIVE', 42);
"""
    
    print("추가할 SQL:")
    print(sql_script)
    return sql_script

def optimize_vrp_settings():
    """VRP 솔버 설정 최적화"""
    
    print("\n=== 2. VRP 솔버 최적화 설정 ===")
    
    optimized_config = """
# 대용량 데이터용 OR-Tools VRP 설정
vrp_config = ORToolsVRPConfig(
    max_solve_time_seconds=180,     # 3분으로 단축
    use_clustering=True,            # 클러스터링 필수
    min_cluster_size=25,            # 클러스터 크기 증가
    max_cluster_size=80,            # 더 큰 클러스터 허용
    epsilon=0.01,                   # 1km 반경으로 확대
    
    # 거리/시간 제약 완화
    max_work_hours=10,              # 10시간으로 확대
    max_distance_km=200,            # 200km으로 확대
    
    # 목적함수 조정
    unassigned_penalty=50000,       # 페널티 감소
    distance_weight=0.8,            # 거리 가중치 감소
    vehicle_fixed_cost=3000,        # 차량 비용 감소
    
    # 거리 계산 최적화
    distance_api={
        'api_priority': ['haversine'],  # Haversine만 사용
        'distance_cache_ttl': 24 * 3600,
        'max_locations_per_request': 200,
        'request_delay': 0.02
    }
)
"""
    
    print("최적화된 설정:")
    print(optimized_config)
    return optimized_config

def enhance_clustering():
    """클러스터링 강화 방안"""
    
    print("\n=== 3. 클러스터링 강화 ===")
    print("목표: 500개 주문을 지리적으로 10개 클러스터로 분할")
    print("방법:")
    print("- K-means 사전 클러스터링 추가")
    print("- HDBSCAN epsilon 파라미터 조정")
    print("- 권역별 사전 분할")
    
    strategy = """
# 개선된 클러스터링 전략
1. 권역별 분할: 500개 → 3개 권역 (167개씩)
2. 권역 내 클러스터링: 167개 → 3-4개 클러스터 (50-60개씩)
3. 총 9-12개 클러스터로 분산
4. 차량당 평균 50-60개 주문 할당
"""
    
    print(strategy)
    return strategy

def adjust_time_limits():
    """시간 제한 조정"""
    
    print("\n=== 4. 시간 제한 조정 ===")
    print("현재: 300초 타임아웃")
    print("권장:")
    print("- VRP 솔빙: 180초 (3분)")
    print("- 전체 프로세스: 600초 (10분)")
    print("- 거리계산: Haversine 전용으로 고속화")

def create_optimized_test():
    """최적화된 테스트 생성"""
    
    print("\n=== 최적화 테스트 실행 방법 ===")
    
    steps = """
1. 차량 추가:
   docker exec -i tms_mysql mysql -utms_user -ptms_password tms_db < add_anyang_vehicles.sql

2. 일부 주문만 테스트 (100개):
   UPDATE orders SET status = 'completed' WHERE center_id = 'CENTER_ANYANG';
   UPDATE orders SET status = 'pending' WHERE center_id = 'CENTER_ANYANG' ORDER BY created_at LIMIT 100;

3. 최적화된 설정으로 테스트:
   timeout 600 python main.py dispatch --center-id CENTER_ANYANG --algorithm auto

4. 성공 시 점진적 확대:
   100개 → 200개 → 300개 → 500개
"""
    
    print(steps)
    return steps

def main():
    """메인 실행"""
    
    solutions = optimize_anyang_center()
    
    # 1. 차량 추가
    sql_script = add_vehicles_anyang()
    
    with open('add_anyang_vehicles.sql', 'w', encoding='utf-8') as f:
        f.write(sql_script)
    
    # 2. VRP 설정 최적화
    config = optimize_vrp_settings()
    
    # 3. 클러스터링 강화
    clustering = enhance_clustering()
    
    # 4. 시간 제한 조정
    adjust_time_limits()
    
    # 5. 테스트 방법
    create_optimized_test()
    
    print("\n=== 다음 단계 ===")
    print("1. add_anyang_vehicles.sql 파일이 생성되었습니다")
    print("2. 다음 명령어로 차량을 추가하세요:")
    print("   docker exec -i tms_mysql mysql -utms_user -ptms_password tms_db < add_anyang_vehicles.sql")
    print("3. 100개 주문으로 점진적 테스트를 시작하세요")
    
    print("\n💡 핵심 전략:")
    print("- 차량 부족이 근본 원인 → 2대에서 10대로 증가")
    print("- 점진적 테스트 → 100개씩 단계적 증가")
    print("- 설정 최적화 → 시간제한 및 클러스터링 개선")

if __name__ == "__main__":
    main()