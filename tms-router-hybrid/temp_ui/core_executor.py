#!/usr/bin/env python3
"""
Core Dispatch Executor - Temp UI에서 독립된 core CLI를 호출하는 서브프로세스 실행기
"""
import subprocess
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple


class CoreDispatchExecutor:
    """Core 모듈을 subprocess로 실행하는 클래스"""
    
    def __init__(self, core_path: Optional[Path] = None):
        """
        Args:
            core_path: core 모듈 경로 (기본값: 프로젝트 루트)
        """
        self.logger = logging.getLogger(__name__)
        
        # Core 모듈 경로 설정
        if core_path is None:
            # temp_ui에서 한 단계 위 디렉토리가 프로젝트 루트
            self.core_path = Path(__file__).parent.parent
        else:
            self.core_path = core_path
            
        # Python 실행 경로
        self.python_executable = sys.executable
        
        self.logger.info(f"Core 경로: {self.core_path}")
        self.logger.info(f"Python 실행 경로: {self.python_executable}")
    
    def execute_dispatch(
        self,
        center_id: str,
        algorithm: str = 'auto',
        region_id: Optional[str] = None,
        dry_run: bool = False,
        timeout: int = 600
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        배차를 실행하고 결과를 JSON으로 반환
        
        Args:
            center_id: 물류센터 ID
            algorithm: 사용할 알고리즘
            region_id: 권역 ID (선택사항)
            dry_run: 시뮬레이션 모드
            timeout: 타임아웃 (초)
            
        Returns:
            (성공여부, 결과_딕셔너리)
        """
        try:
            # 새로운 main.py 방식으로 명령어 구성
            cmd = [
                self.python_executable,
                'main.py',
                'center', center_id
            ]
            
            # 주의: 새로운 main.py는 단순한 형태이므로 대부분의 옵션은 무시됩니다
            # algorithm, region_id, dry_run 등은 현재 지원되지 않음
            
            self.logger.info(f"실행할 명령어: {' '.join(cmd)}")
            
            # subprocess 실행 (PYTHONPATH 설정)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.core_path)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.core_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                env=env
            )
            
            if result.returncode == 0:
                # 성공 - 텍스트 출력 파싱하여 JSON 형태로 변환
                try:
                    dispatch_result = self._parse_text_output_to_json(result.stdout, center_id)
                    self.logger.info(f"배차 성공: {dispatch_result.get('batch_id', 'Unknown')}")
                    return True, dispatch_result
                except Exception as e:
                    self.logger.error(f"출력 파싱 오류: {e}")
                    self.logger.error(f"출력 내용: {result.stdout}")
                    return False, {
                        'status': 'error',
                        'error_message': f'출력 파싱 오류: {e}',
                        'raw_output': result.stdout
                    }
            else:
                # 실패 - 텍스트 오류 메시지 처리
                error_output = result.stderr or result.stdout
                self.logger.error(f"배차 실패: {error_output}")
                
                # 텍스트 출력에서 오류 메시지 추출
                error_message = self._extract_error_message(error_output)
                
                return False, {
                    'status': 'failed',
                    'error_message': error_message,
                    'return_code': result.returncode,
                    'center_id': center_id,
                    'algorithm': 'unknown'
                }
                    
        except subprocess.TimeoutExpired:
            self.logger.error(f"배차 실행 타임아웃 ({timeout}초)")
            return False, {
                'status': 'error',
                'error_message': f'배차 실행이 {timeout}초 안에 완료되지 않았습니다. 대용량 데이터의 경우 더 많은 시간이 필요할 수 있습니다.'
            }
        except Exception as e:
            self.logger.error(f"subprocess 실행 오류: {e}")
            return False, {
                'status': 'error',
                'error_message': f'시스템 오류: {str(e)}'
            }
    
    def get_system_status(self) -> Tuple[bool, Dict[str, Any]]:
        """시스템 상태 확인"""
        try:
            cmd = [
                self.python_executable,
                '-m', 'core.main',
                'status'
            ]
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.core_path)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.core_path),
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                env=env
            )
            
            if result.returncode == 0:
                return True, {
                    'status': 'ok',
                    'message': '시스템 상태 정상',
                    'details': result.stdout
                }
            else:
                return False, {
                    'status': 'error',
                    'message': '시스템 상태 확인 실패',
                    'details': result.stderr
                }
                
        except Exception as e:
            return False, {
                'status': 'error',
                'message': f'상태 확인 오류: {str(e)}'
            }
    
    def clear_cache(self, cache_type: str = 'all') -> Tuple[bool, str]:
        """캐시 삭제"""
        try:
            cmd = [
                self.python_executable,
                '-m', 'core.main',
                'clear-cache',
                '--cache-type', cache_type
            ]
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.core_path)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.core_path),
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                env=env
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, f'캐시 삭제 오류: {str(e)}'
    
    def _parse_text_output_to_json(self, text_output: str, center_id: str) -> Dict[str, Any]:
        """텍스트 출력을 JSON 형태로 파싱"""
        import re
        from datetime import datetime
        
        result = {
            'batch_id': f'DISPATCH_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'status': 'completed',
            'execution_time_seconds': 0.0,
            'center_id': center_id,
            'metrics': {
                'total_orders': 0,
                'assigned_orders': 0,
                'unassigned_orders': 0,
                'total_vehicles': 0,
                'used_vehicles': 0,
                'unused_vehicles': 0,
                'average_capacity_utilization': 0.0,
                'total_estimated_distance': 0.0,
                'total_estimated_time': 0.0,
                'algorithm_used': 'unknown',
                'quality_score': 0.0
            },
            'vehicle_assignments': [],
            'unassigned_orders': [],
            'warnings': [],
            'error_message': None
        }
        
        try:
            # 상태 추출
            status_match = re.search(r'상태:\s*([^\n]+)', text_output)
            if status_match:
                status = status_match.group(1).strip()
                result['status'] = status
                
                # vehicle_shortage 상태를 성공으로 처리하되 경고 메시지 추가
                if status == 'vehicle_shortage':
                    result['status'] = 'completed'
                    result['warnings'] = ['차량이 부족하여 배차를 수행할 수 없습니다']
            
            # 실행 시간 추출
            if '실행 시간:' in text_output:
                time_match = re.search(r'실행 시간:\s*([\d.]+)초', text_output)
                if time_match:
                    result['execution_time_seconds'] = float(time_match.group(1))
            
            # 알고리즘 추출
            if '사용 알고리즘:' in text_output:
                algo_match = re.search(r'사용 알고리즘:\s*([^\n]+)', text_output)
                if algo_match:
                    result['metrics']['algorithm_used'] = algo_match.group(1).strip()
            
            # 품질 점수 추출
            if '품질 점수:' in text_output:
                quality_match = re.search(r'품질 점수:\s*([\d.]+)', text_output)
                if quality_match:
                    result['metrics']['quality_score'] = float(quality_match.group(1))
            
            # 차량 정보 추출 (실제 출력 형식에 맞게 수정)
            vehicle_match = re.search(r'총 차량:\s*(\d+)대', text_output)
            if vehicle_match:
                result['metrics']['total_vehicles'] = int(vehicle_match.group(1))
            
            used_vehicle_match = re.search(r'사용 차량:\s*(\d+)대', text_output)
            if used_vehicle_match:
                result['metrics']['used_vehicles'] = int(used_vehicle_match.group(1))
            
            # 주문 정보 추출 (실제 출력 형식에 맞게 수정)
            total_order_match = re.search(r'총 주문:\s*(\d+)건', text_output)
            if total_order_match:
                result['metrics']['total_orders'] = int(total_order_match.group(1))
            
            assigned_order_match = re.search(r'배정 주문:\s*(\d+)건', text_output)
            if assigned_order_match:
                result['metrics']['assigned_orders'] = int(assigned_order_match.group(1))
            
            unassigned_order_match = re.search(r'미배정 주문:\s*(\d+)건', text_output)
            if unassigned_order_match:
                result['metrics']['unassigned_orders'] = int(unassigned_order_match.group(1))
            
            # 경고 및 오류 메시지 추출
            warning_patterns = [
                r'⚠️\s*차량 부족:\s*([^\n]+)',
            ]
            
            error_patterns = [
                r'❌ 오류:\s*([^\n]+)',
                r'❌ 배차 실행 중 오류 발생:\s*([^\n]+)',
                r'ERROR:\s*([^\n]+)',
            ]
            
            # 경고 메시지 먼저 체크 (차량 부족)
            for pattern in warning_patterns:
                warning_match = re.search(pattern, text_output)
                if warning_match:
                    result['warnings'].append(warning_match.group(1).strip())
                    result['status'] = 'completed'  # 차량 부족은 성공으로 처리
                    break
            
            # 실제 오류 메시지 체크
            if result['status'] != 'completed':  # 경고가 없는 경우만
                for pattern in error_patterns:
                    error_match = re.search(pattern, text_output)
                    if error_match:
                        result['error_message'] = error_match.group(1).strip()
                        result['status'] = 'failed'
                        break
            
            # 차량별 배정 정보 추출 (배차 세부 내용 테이블)
            if '배차 세부 내용:' in text_output:
                # 테이블 형태의 데이터 파싱
                lines = text_output.split('\n')
                in_table = False
                for line in lines:
                    if '차량ID' in line and '주문수' in line:
                        in_table = True
                        continue
                    elif in_table and line.strip().startswith('─'):
                        continue
                    elif in_table and line.strip():
                        # 테이블 행 파싱
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                vehicle_assignment = {
                                    'vehicle_id': parts[0],
                                    'driver_name': 'Unknown',
                                    'vehicle_type': 'Unknown',
                                    'region_name': 'Unknown',
                                    'assigned_orders': [f'ORDER_{i}' for i in range(int(parts[1]))],
                                    'estimated_distance_km': float(parts[3].replace('km', '')),
                                    'estimated_time_minutes': float(parts[4].replace('분', '')) if len(parts) > 4 else 0,
                                    'capacity_utilization': float(parts[2].replace('%', '')) / 100 if '%' in parts[2] else 0
                                }
                                result['vehicle_assignments'].append(vehicle_assignment)
                            except (ValueError, IndexError):
                                continue
                    elif in_table and not line.strip():
                        break
                        
        except Exception as e:
            self.logger.warning(f"텍스트 파싱 중 일부 오류: {e}")
        
        return result
    
    def _extract_error_message(self, error_output: str) -> str:
        """오류 출력에서 주요 오류 메시지 추출"""
        import re
        
        if not error_output:
            return "알 수 없는 오류가 발생했습니다."
        
        # 다양한 오류 패턴 매칭
        error_patterns = [
            r'❌ 배차 실행 중 오류 발생:\s*([^\n]+)',
            r'❌ 오류:\s*([^\n]+)',
            r'ERROR:\s*([^\n]+)',
            r'Error:\s*([^\n]+)',
            r'Exception:\s*([^\n]+)',
            r'Traceback.*?:\s*([^\n]+)',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, error_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 특정 오류 키워드 검색
        if 'No module named' in error_output:
            return "모듈을 찾을 수 없습니다. 시스템 설정을 확인해주세요."
        elif 'permission denied' in error_output.lower():
            return "권한이 없습니다. 시스템 권한을 확인해주세요."
        elif 'connection' in error_output.lower() and 'refused' in error_output.lower():
            return "데이터베이스 연결이 거부되었습니다."
        elif 'timeout' in error_output.lower():
            return "작업 시간이 초과되었습니다."
        
        # 첫 번째 의미있는 라인 반환
        lines = error_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Traceback') and not line.startswith('File'):
                return line[:200]  # 최대 200자
        
        return "시스템 오류가 발생했습니다."


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    executor = CoreDispatchExecutor()
    
    print("=== 시스템 상태 확인 ===")
    success, status = executor.get_system_status()
    print(f"성공: {success}")
    print(f"상태: {status}")
    
    print("\n=== 배차 테스트 (dry-run) ===")
    success, result = executor.execute_dispatch(
        center_id='CENTER_GANGNAM',
        dry_run=True
    )
    print(f"성공: {success}")
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")