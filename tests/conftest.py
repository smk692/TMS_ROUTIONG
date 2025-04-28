import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# pytest 설정
def pytest_configure(config):
    """pytest 설정 추가"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )