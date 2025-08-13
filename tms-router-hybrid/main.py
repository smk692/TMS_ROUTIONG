#!/usr/bin/env python3
"""
TMS Router Hybrid - 배차 최적화 시스템
실행 방법: 
  python main.py center CENTER_GANGNAM
  python main.py rider RIDER_001
"""

import sys
import os
import json
from datetime import datetime
from typing import Optional

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.services.dispatch_orchestrator import DispatchOrchestrator
from core.config import get_settings


def print_usage():
    """사용법 출력"""
    print("TMS Router Hybrid - 배차 최적화 시스템")
    print("사용법:")
    print("  python main.py center <CENTER_ID>    # 센터별 전체 배차")
    print("  python main.py rider <VEHICLE_ID>    # 차량 주문 재배차")
    print()
    print("예시:")
    print("  python main.py center CENTER_GANGNAM")
    print("  python main.py rider VEH_SEONGNAM_001")


def execute_center_dispatch(center_id: str):
    """센터별 전체 배차 실행"""
    print(f"🏢 센터 '{center_id}' 전체 배차를 시작합니다...")
    
    try:
        # 설정 로드
        settings = get_settings()
        
        # 배차 오케스트레이터 초기화
        config = {
            'database_url': settings.database_url,
            'weather_api_key': settings.external_api.openweather_api_key,
            'traffic_api_key': settings.external_api.here_api_key
        }
        
        orchestrator = DispatchOrchestrator(config)
        
        # 배차 실행
        print("📊 데이터 수집 중...")
        result = orchestrator.execute_dispatch(center_id=center_id)
        
        # 결과 출력
        print_dispatch_result(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 배차 실행 중 오류 발생: {error_msg}")
        
        # 차량 부족 상황은 예상 가능한 상황이므로 정상 처리
        if "자동 배차 가능한 차량이 없습니다" in error_msg or "센터에 등록된 차량이 없거나" in error_msg:
            print("\n" + "="*60)
            print("📋 배차 결과")
            print("="*60)
            print(f"상태: vehicle_shortage")
            print(f"실행 시간: 0.0초")
            print(f"❌ 오류: {error_msg}")
            print("="*60)
            print(f"✅ 배차 완료 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            return True  # 예상 가능한 상황이므로 성공으로 처리
        
        return False
    
    return True


def execute_rider_dispatch(vehicle_id: str):
    """차량 주문 재배차 실행"""
    print(f"🔄 차량 '{vehicle_id}' 주문 재배차를 시작합니다...")
    
    try:
        settings = get_settings()
        
        config = {
            'database_url': settings.database_url,
            'weather_api_key': settings.external_api.openweather_api_key,
            'traffic_api_key': settings.external_api.here_api_key
        }
        
        orchestrator = DispatchOrchestrator(config)
        
        print("📊 차량 배정 주문 수집 중...")
        result = orchestrator.execute_vehicle_redispatch(vehicle_id=vehicle_id)
        
        print_dispatch_result(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 차량 재배차 실행 중 오류 발생: {error_msg}")
        
        # 차량 부족 상황은 예상 가능한 상황이므로 정상 처리
        if "자동 배차 가능한 차량이 없습니다" in error_msg or "센터에 등록된 차량이 없거나" in error_msg:
            print("\n" + "="*60)
            print("📋 배차 결과")
            print("="*60)
            print(f"상태: vehicle_shortage")
            print(f"실행 시간: 0.0초")
            print(f"❌ 오류: {error_msg}")
            print("="*60)
            print(f"✅ 배차 완료 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            return True  # 예상 가능한 상황이므로 성공으로 처리
        
        return False
    
    return True


def print_dispatch_result(result):
    """배차 결과 출력"""
    print("\n" + "="*60)
    print("📋 배차 결과")
    print("="*60)
    
    # 기본 정보
    status = result.status.value if hasattr(result.status, 'value') else result.status
    
    # 차량 부족 상황을 특별히 처리
    if hasattr(result, 'error_message') and result.error_message:
        if ("자동 배차 가능한 차량이 없습니다" in result.error_message or 
            "센터에 등록된 차량이 없거나" in result.error_message):
            status = "vehicle_shortage"
    
    print(f"상태: {status}")
    print(f"실행 시간: {result.execution_time_seconds:.1f}초")
    
    if result.metrics:
        print(f"사용 알고리즘: {result.metrics.algorithm_used}")
        print(f"품질 점수: {result.metrics.quality_score:.3f}")
        print(f"총 차량: {result.metrics.total_vehicles}대")
        print(f"사용 차량: {result.metrics.used_vehicles}대")
        print(f"총 주문: {result.metrics.total_orders}건")
        print(f"배정 주문: {result.metrics.assigned_orders}건")
        print(f"미배정 주문: {result.metrics.unassigned_orders}건")
    
    # 배차 세부 내용
    if hasattr(result, 'vehicle_assignments') and result.vehicle_assignments:
        print("\n📝 배차 세부 내용:")
        print("-" * 80)
        print(f"{'차량ID':<12} {'주문수':<8} {'용량활용도':<12} {'예상거리':<12} {'예상시간':<12}")
        print("-" * 80)
        
        for assignment in result.vehicle_assignments:
            print(f"{assignment.vehicle_id:<12} "
                  f"{len(assignment.assigned_orders):<8} "
                  f"{assignment.capacity_utilization:.1%:<12} "
                  f"{assignment.estimated_distance_km:.1f}km{'':<8} "
                  f"{assignment.estimated_time_minutes:.0f}분{'':<8}")
    
    # 경고 및 오류
    if hasattr(result, 'warnings') and result.warnings:
        print("\n⚠️  경고:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if hasattr(result, 'error_message') and result.error_message:
        # 차량 부족 상황은 경고로 표시
        if ("자동 배차 가능한 차량이 없습니다" in result.error_message or 
            "센터에 등록된 차량이 없거나" in result.error_message):
            print(f"\n⚠️  차량 부족: {result.error_message}")
            print("💡 해결 방법:")
            print("   - 차량을 추가 등록하세요")
            print("   - 기존 차량의 상태를 'ACTIVE'로 변경하세요")
            print("   - 차량의 'auto_dispatch'를 활성화하세요")
            print("   - 차량 유형이 'TOP_CAR' 또는 'CARGO'인지 확인하세요")
        else:
            print(f"\n❌ 오류: {result.error_message}")
    
    print("="*60)
    completion_status = "차량 부족으로 배차 불가" if status == "vehicle_shortage" else "배차 완료"
    print(f"✅ {completion_status} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


def main():
    """메인 실행 함수"""
    # 인자 검증
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    target_id = sys.argv[2]
    
    # 모드별 실행
    if mode == "center":
        success = execute_center_dispatch(target_id)
    elif mode == "rider":
        success = execute_rider_dispatch(target_id)
    else:
        print(f"❌ 알 수 없는 모드: {mode}")
        print_usage()
        sys.exit(1)
    
    # 실행 결과에 따른 종료 코드
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()