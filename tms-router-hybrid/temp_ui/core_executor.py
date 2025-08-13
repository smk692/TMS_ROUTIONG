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
            # 명령어 구성
            cmd = [
                self.python_executable,
                '-m', 'core.main',
                'dispatch',
                '--center-id', center_id,
                '--algorithm', algorithm,
                '--output-format', 'json'
            ]
            
            if region_id:
                cmd.extend(['--region-id', region_id])
                
            if dry_run:
                cmd.append('--dry-run')
            
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
                # 성공 - JSON 파싱
                try:
                    dispatch_result = json.loads(result.stdout)
                    self.logger.info(f"배차 성공: {dispatch_result.get('batch_id', 'Unknown')}")
                    return True, dispatch_result
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON 파싱 오류: {e}")
                    self.logger.error(f"출력 내용: {result.stdout}")
                    return False, {
                        'status': 'error',
                        'error_message': f'JSON 파싱 오류: {e}',
                        'raw_output': result.stdout
                    }
            else:
                # 실패 - 오류 메시지 처리
                try:
                    # stderr에서 JSON 오류 메시지 파싱 시도
                    error_result = json.loads(result.stderr)
                    self.logger.error(f"배차 실패: {error_result.get('message', 'Unknown error')}")
                    return False, error_result
                except json.JSONDecodeError:
                    # 일반 텍스트 오류 메시지
                    self.logger.error(f"배차 실패: {result.stderr}")
                    return False, {
                        'status': 'error',
                        'error_message': result.stderr or result.stdout,
                        'return_code': result.returncode
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