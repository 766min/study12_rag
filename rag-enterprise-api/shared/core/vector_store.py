from typing import Protocol, Any
import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from shared.core.config import settings
from shared.core.logger import get_logger

logger = get_logger(__name__)

def get_llm() -> BaseChatModel: # 질문에 답변을 생성할 LLM (채팅 모델)
    """"설정된 프로바이더에 맞는 채팅 모델 인스턴스를 추상 타입으로 반환합니다."""
    return init_chat_model(
        model=settings.LLM_MODEL,
        model_provider=settings.LLM_PROVIDER,
        api_key=settings.API_KEY,
        reasoning_effort=settings.REASONING_EFFORT,
    )

def get_embeddings() -> Embeddings: # 텍스트를 벡터(숫자 배열)로 바꿔주는 임베딩 모델
    """임베딩 모델 인스턴스를 공통 Embeddings 인터페이스로 반환합니다."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.API_KEY,
    )

def get_vector_store(collection_name: str) -> VectorStore: # 벡터를 저장/검색하는 Postgres 기반 임베딩 모델
    """지정된 컬렉션 이름의 벡터 스토어를 공통 VectorStore 인터페이스로 반환합니다."""
    return PGVector(
        Embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=settings.DATABASE_URL,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )

class DBConnection(Protocol): # DB연결이라면 최소한 이런 메서드들은 가지고 있어야한다는 설계도 정의 Protocol은 구체적으로 어떤 클래스인지는 몰라도, 이 모양만 갖추면 DB연결로 인정해주겠다는 타입 힌트 도구다.
    def cursor(self) -> Any: ...
    def commit(self) -> None: ... 
    def rollback(self) -> None: ...
    def close(slef) -> None: ...

def get_rw_db_connection() -> DBConnection: # 벡터 DB와는 별개로, SQL을 직접 날릴 수 있는 원시 DB 연결
    """충분 해시 조회 및 BM25 코퍼스 로드를 위한 DB 커넥션을 반환합니다."""
    raw_dsn = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(raw_dsn)