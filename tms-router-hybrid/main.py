#!/usr/bin/env python3
"""
TMS Router Hybrid - Main Entry Point
실행 방법: python main.py dispatch --center-id CENTER_GANGNAM
"""

import sys
import os

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# core.main 모듈 import 및 실행
if __name__ == '__main__':
    from core.main import cli
    cli()