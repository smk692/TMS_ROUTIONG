# src/core/logger.py
import logging
from datetime import datetime
import os

def setup_logger(name: str) -> logging.Logger:
    """알고리즘별 로거 설정"""
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 스킵
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 로그 디렉토리 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 핸들러
    log_file = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger