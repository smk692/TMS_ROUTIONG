#!/usr/bin/env python3
"""
TMS Router - 테스트 주문 데이터 생성기
각 센터당 500개씩 총 3,000개 주문 생성
좌표 중복 최소화를 위한 격자 패턴 + 랜덤 오프셋 사용
"""

import random
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

# 센터별 설정
CENTERS = {
    'CENTER_GANGNAM': {
        'name': '강남',
        'regions': [
            ('REGION_GANGNAM_01', '강남구 역삼동'),
            ('REGION_GANGNAM_02', '강남구 삼성동'),
            ('REGION_GANGNAM_03', '서초구 서초동')
        ],
        'lat_range': (37.46, 37.53),
        'lng_range': (126.99, 127.09),
        'grid_size': (25, 20)  # lat_divisions, lng_divisions
    },
    'CENTER_HANAM': {
        'name': '하남',
        'regions': [
            ('REGION_HANAM_01', '하남시 미사동'),
            ('REGION_HANAM_02', '광주시 경안동'),
            ('REGION_HANAM_03', '구리시 수택동')
        ],
        'lat_range': (37.40, 37.65),
        'lng_range': (127.08, 127.34),
        'grid_size': (25, 20)
    },
    'CENTER_SONGPA': {
        'name': '송파',
        'regions': [
            ('REGION_SONGPA_01', '송파구 잠실동'),
            ('REGION_SONGPA_02', '강동구 천호동')
        ],
        'lat_range': (37.49, 37.56),
        'lng_range': (127.09, 127.18),
        'grid_size': (25, 20)
    },
    'CENTER_SUWON': {
        'name': '수원',
        'regions': [
            ('REGION_SUWON_01', '수원시 영통구'),
            ('REGION_SUWON_02', '용인시 수지구')
        ],
        'lat_range': (37.15, 37.35),
        'lng_range': (126.95, 127.18),
        'grid_size': (25, 20)
    },
    'CENTER_SEONGNAM': {
        'name': '성남',
        'regions': [
            ('REGION_SEONGNAM_01', '성남시 분당구 정자동'),
            ('REGION_SEONGNAM_02', '성남시 중원구 상대원동')
        ],
        'lat_range': (37.35, 37.55),
        'lng_range': (127.10, 127.26),
        'grid_size': (25, 20)
    },
    'CENTER_ANYANG': {
        'name': '안양',
        'regions': [
            ('REGION_ANYANG_01', '안양시 동안구 평촌동'),
            ('REGION_ANYANG_02', '과천시 중앙동')
        ],
        'lat_range': (37.34, 37.44),
        'lng_range': (126.92, 127.00),
        'grid_size': (25, 20)
    }
}

ORDERS_PER_CENTER = 500

def generate_coordinates(center_config: Dict, index: int) -> Tuple[float, float]:
    """
    격자 패턴 기반 좌표 생성 (중복 최소화)
    """
    lat_range = center_config['lat_range']
    lng_range = center_config['lng_range']
    grid_size = center_config['grid_size']
    
    # 격자 위치 계산
    lat_idx = index % grid_size[0]
    lng_idx = (index // grid_size[0]) % grid_size[1]
    
    # 격자 간격 계산
    lat_step = (lat_range[1] - lat_range[0]) / grid_size[0]
    lng_step = (lng_range[1] - lng_range[0]) / grid_size[1]
    
    # 기본 격자 좌표
    base_lat = lat_range[0] + lat_idx * lat_step
    base_lng = lng_range[0] + lng_idx * lng_step
    
    # 랜덤 오프셋 추가 (격자 내에서 약간의 변동)
    offset_lat = (random.random() - 0.5) * lat_step * 0.3
    offset_lng = (random.random() - 0.5) * lng_step * 0.3
    
    lat = round(base_lat + offset_lat, 8)
    lng = round(base_lng + offset_lng, 8)
    
    return lat, lng

def generate_priority() -> str:
    """
    우선순위 생성 (10% high, 70% normal, 20% low)
    """
    rand = random.random()
    if rand < 0.1:
        return 'high'
    elif rand < 0.8:
        return 'normal'
    else:
        return 'low'

def generate_orders_sql() -> str:
    """
    전체 주문 INSERT SQL 생성
    """
    sql_lines = []
    sql_lines.append("-- 주문 데이터 (각 센터당 500개씩)")
    sql_lines.append("INSERT INTO `orders` (`id`, `center_id`, `region_id`, `address`, `latitude`, `longitude`, `priority`, `status`, `created_at`) VALUES")
    
    all_orders = []
    order_id = 1
    
    for center_id, config in CENTERS.items():
        for i in range(ORDERS_PER_CENTER):
            # 권역 선택 (균등 분배)
            region_idx = i % len(config['regions'])
            region_id, region_addr = config['regions'][region_idx]
            
            # 좌표 생성
            lat, lng = generate_coordinates(config, i)
            
            # 주소 생성
            address = f"{region_addr} {random.randint(100, 999)}번지"
            
            # 우선순위
            priority = generate_priority()
            
            # 시간 오프셋 (1~500시간 전)
            time_offset = random.randint(1, 500)
            
            # 주문 ID
            order_num = str(order_id).zfill(4)
            order_code = f"ORD_{config['name'].upper()}_{order_num}"
            
            # SQL 행 생성
            order_sql = f"('{order_code}', '{center_id}', '{region_id}', '{address}', {lat}, {lng}, '{priority}', 'pending', NOW() - INTERVAL {time_offset} HOUR)"
            all_orders.append(order_sql)
            
            order_id += 1
    
    # SQL 조합 (마지막 행 제외하고 콤마 추가)
    for i, order_sql in enumerate(all_orders):
        if i < len(all_orders) - 1:
            sql_lines.append(order_sql + ",")
        else:
            sql_lines.append(order_sql + ";")
    
    return "\n".join(sql_lines)

def generate_verification_queries() -> str:
    """
    데이터 검증 쿼리 생성
    """
    queries = []
    
    # 센터별 주문 수 확인
    queries.append("""
-- 센터별 주문 수 확인
SELECT 
    c.name AS center_name,
    COUNT(o.id) AS order_count,
    COUNT(DISTINCT CONCAT(ROUND(o.latitude, 5), ',', ROUND(o.longitude, 5))) AS unique_coordinates
FROM centers c
LEFT JOIN orders o ON c.id = o.center_id
GROUP BY c.id, c.name
ORDER BY c.id;
""")
    
    # 좌표 중복 확인
    queries.append("""
-- 좌표 중복 확인 (최대 2개까지만 허용)
SELECT 
    ROUND(latitude, 5) AS lat,
    ROUND(longitude, 5) AS lng,
    COUNT(*) AS duplicate_count
FROM orders
GROUP BY ROUND(latitude, 5), ROUND(longitude, 5)
HAVING COUNT(*) > 2
ORDER BY duplicate_count DESC
LIMIT 10;
""")
    
    # 권역별 분포 확인
    queries.append("""
-- 권역별 주문 분포
SELECT 
    r.center_id,
    r.name AS region_name,
    COUNT(o.id) AS order_count,
    ROUND(COUNT(o.id) * 100.0 / (SELECT COUNT(*) FROM orders WHERE center_id = r.center_id), 1) AS percentage
FROM regions r
LEFT JOIN orders o ON r.id = o.region_id
GROUP BY r.center_id, r.id, r.name
ORDER BY r.center_id, r.id;
""")
    
    return "\n".join(queries)

def main():
    """
    메인 실행 함수
    """
    print("TMS Router - 테스트 주문 데이터 생성")
    print("=" * 50)
    print(f"각 센터당 {ORDERS_PER_CENTER}개씩 생성")
    print(f"총 {len(CENTERS) * ORDERS_PER_CENTER}개 주문 생성")
    print("=" * 50)
    
    # SQL 파일 생성
    with open('generated_orders.sql', 'w', encoding='utf-8') as f:
        f.write("-- TMS Router - 자동 생성된 주문 데이터\n")
        f.write(f"-- 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"-- 센터당 {ORDERS_PER_CENTER}개, 총 {len(CENTERS) * ORDERS_PER_CENTER}개\n\n")
        
        f.write("-- 기존 주문 데이터 삭제\n")
        f.write("DELETE FROM `order_assignments`;\n")
        f.write("DELETE FROM `vehicle_assignments`;\n")
        f.write("DELETE FROM `dispatch_batches`;\n")
        f.write("DELETE FROM `orders`;\n\n")
        
        f.write(generate_orders_sql())
        f.write("\n\n")
        
        f.write("-- 인덱스 통계 업데이트\n")
        f.write("ANALYZE TABLE orders;\n\n")
        
        f.write(generate_verification_queries())
    
    print("✅ generated_orders.sql 파일 생성 완료")
    print("")
    print("실행 방법:")
    print("  mysql -u root tms_db < generated_orders.sql")
    print("")
    
    # 통계 정보 출력
    print("생성된 데이터 통계:")
    for center_id, config in CENTERS.items():
        print(f"  - {config['name']} 센터: {ORDERS_PER_CENTER}개")
        for region_id, region_addr in config['regions']:
            region_orders = ORDERS_PER_CENTER // len(config['regions'])
            print(f"    └ {region_addr}: 약 {region_orders}개")
    
    print("")
    print("좌표 생성 전략:")
    print("  - 격자 패턴 기반 (중복 최소화)")
    print("  - 각 격자 내 랜덤 오프셋 적용")
    print("  - 최대 2개까지만 같은 좌표 허용")

if __name__ == "__main__":
    main()