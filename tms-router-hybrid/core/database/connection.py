"""
데이터베이스 연결 관리
"""
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from ..config.settings import get_settings


class DatabaseManager:
    """데이터베이스 연결 및 세션 관리"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
    def _create_engine(self) -> Engine:
        """데이터베이스 엔진 생성"""
        db_config = self.settings.database
        
        # MySQL 연결 URL 구성
        connection_url = (
            f"mysql+pymysql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
            f"?charset=utf8mb4&use_unicode=1&collation=utf8mb4_unicode_ci"
        )
        
        # 엔진 설정
        engine_kwargs = {
            'poolclass': QueuePool,
            'pool_size': db_config.pool_size,
            'max_overflow': db_config.max_overflow,
            'pool_timeout': db_config.pool_timeout,
            'pool_recycle': db_config.pool_recycle,
            'pool_pre_ping': True,  # 연결 유효성 검증
            'echo': db_config.echo_sql,  # SQL 로깅
            'future': True,  # SQLAlchemy 2.0 스타일
        }
        
        try:
            engine = create_engine(connection_url, **engine_kwargs)
            self.logger.info(f"데이터베이스 엔진 생성 완료: {db_config.host}:{db_config.port}")
            return engine
            
        except Exception as e:
            self.logger.error(f"데이터베이스 엔진 생성 실패: {str(e)}")
            raise
    
    @property
    def engine(self) -> Engine:
        """엔진 반환 (지연 생성)"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """세션 팩토리 반환"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                future=True
            )
        return self._session_factory
    
    def get_session(self) -> Session:
        """새 세션 반환"""
        return self.session_factory()
    
    @contextmanager
    def session_scope(self, auto_commit: bool = True):
        """세션 컨텍스트 매니저"""
        session = self.get_session()
        try:
            yield session
            if auto_commit:
                session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"데이터베이스 세션 오류: {str(e)}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            with self.session_scope() as session:
                result = session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 테스트 실패: {str(e)}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """연결 정보 반환"""
        db_config = self.settings.database
        return {
            'host': db_config.host,
            'port': db_config.port,
            'database': db_config.database,
            'user': db_config.user,
            'pool_size': db_config.pool_size,
            'max_overflow': db_config.max_overflow,
            'echo_sql': db_config.echo_sql,
        }
    
    def close(self):
        """연결 종료"""
        if self._engine:
            self._engine.dispose()
            self.logger.info("데이터베이스 연결 종료")


# 전역 데이터베이스 매니저 인스턴스
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """데이터베이스 매니저 싱글톤 반환"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session() -> Session:
    """새 데이터베이스 세션 반환"""
    return get_database_manager().get_session()


@contextmanager
def db_session(auto_commit: bool = True):
    """데이터베이스 세션 컨텍스트 매니저"""
    with get_database_manager().session_scope(auto_commit=auto_commit) as session:
        yield session


def test_database_connection() -> bool:
    """데이터베이스 연결 테스트"""
    return get_database_manager().test_connection()