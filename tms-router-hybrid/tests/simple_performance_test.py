#!/usr/bin/env python3
"""
간단한 OR-Tools VRP 성능 테스트
Haversine 거리 계산만 사용하여 빠른 테스트
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import time
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def test_database_connection():
    """데이터베이스 연결 테스트"""
    try:
        # 데이터베이스 연결
        DATABASE_URL = "mysql://tms_user:tms_password@localhost:3306/tms_db"
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # 테스트 데이터 확인
            result = conn.execute(text("""
                SELECT center_id, COUNT(*) as order_count
                FROM orders 
                WHERE id LIKE 'TEST_ORD_%' AND status = 'pending'
                GROUP BY center_id
                ORDER BY center_id
            """))
            
            print("=== 테스트 데이터 현황 ===")
            total_orders = 0
            for row in result:
                print(f"{row.center_id}: {row.order_count}개 주문")
                total_orders += row.order_count
            
            print(f"총 {total_orders}개 테스트 주문")
            
            # 차량 데이터 확인
            result = conn.execute(text("""
                SELECT center_id, COUNT(*) as vehicle_count
                FROM vehicles 
                WHERE id LIKE 'TEST_VEH_%' AND status = 'ACTIVE'
                GROUP BY center_id
                ORDER BY center_id
            """))
            
            print("\n=== 테스트 차량 현황 ===")
            total_vehicles = 0
            for row in result:
                print(f"{row.center_id}: {row.vehicle_count}대 차량")
                total_vehicles += row.vehicle_count
            
            print(f"총 {total_vehicles}대 테스트 차량")
            
            return True
            
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return False

def test_simple_dispatch():
    """간단한 배차 테스트"""
    try:
        from core.main import dispatch_command
        from core.services.dispatch_orchestrator import DispatchOrchestrator
        from core.database.connection import get_engine
        
        print("\n=== 간단한 배차 테스트 시작 ===")
        
        # 강남 센터 소량 테스트 (일부 주문만)
        start_time = time.time()
        
        engine = get_engine()
        with engine.connect() as conn:
            # 강남 센터 주문 50개만 선택
            conn.execute(text("""
                UPDATE orders 
                SET status = 'completed' 
                WHERE center_id = 'GANGNAM' 
                  AND id LIKE 'TEST_ORD_%' 
                  AND status = 'pending'
            """))
            
            # 처음 50개만 다시 pending으로 변경
            conn.execute(text("""
                UPDATE orders 
                SET status = 'pending' 
                WHERE center_id = 'GANGNAM' 
                  AND id LIKE 'TEST_ORD_GANGNAM_%' 
                  AND id <= 'TEST_ORD_GANGNAM_050'
            """))
            
            conn.commit()
        
        print("강남 센터 50개 주문으로 테스트 설정 완료")
        
        # 배차 실행은 main.py에서 진행
        print("배차 테스트는 다음 명령어로 실행하세요:")
        print("python main.py dispatch --center-id GANGNAM --algorithm simple")
        
        return True
        
    except Exception as e:
        print(f"간단한 배차 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_algorithms():
    """알고리즘 성능 비교"""
    print("\n=== 알고리즘 성능 비교 방법 ===")
    print("1. Simple 알고리즘:")
    print("   python main.py dispatch --center-id GANGNAM --algorithm simple")
    print("")
    print("2. Fastest 알고리즘:")
    print("   python main.py dispatch --center-id GANGNAM --algorithm fastest")
    print("")
    print("3. Auto 알고리즘 (OR-Tools VRP):")
    print("   python main.py dispatch --center-id GANGNAM --algorithm auto")
    print("")
    print("각 알고리즘의 실행 시간과 배정 결과를 비교해보세요.")

def main():
    """메인 함수"""
    
    print("=== TMS Router Hybrid - 간단한 성능 테스트 ===\n")
    
    # 1. 데이터베이스 연결 테스트
    if not test_database_connection():
        print("데이터베이스 연결 실패. 테스트를 중단합니다.")
        return
    
    # 2. 간단한 배차 테스트 준비
    if test_simple_dispatch():
        print("\n✅ 테스트 준비 완료!")
    else:
        print("\n❌ 테스트 준비 실패")
    
    # 3. 알고리즘 비교 방법 안내
    compare_algorithms()
    
    print("\n=== OR-Tools VRP 성능 확인 방법 ===")
    print("1. 먼저 간단한 알고리즘으로 테스트:")
    print("   time python main.py dispatch --center-id GANGNAM --algorithm simple")
    print("")
    print("2. OR-Tools VRP 알고리즘으로 테스트:")
    print("   time python main.py dispatch --center-id GANGNAM --algorithm auto")
    print("")
    print("3. 각 알고리즘의 결과를 비교:")
    print("   - 실행 시간")
    print("   - 배정 주문 수")
    print("   - 사용된 차량 수")
    print("   - 총 이동 거리")
    print("   - 품질 점수")

if __name__ == "__main__":
    main()