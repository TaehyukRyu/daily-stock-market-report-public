"""
ChromaDB 컬렉션 관리 모듈 — L2 Knowledge & Memory Layer

역할:
  - 6개 컬렉션 초기화 및 접근
  - 문서 임베딩 → ChromaDB 저장
  - 벡터 유사도 검색 (단독 사용 또는 EnsembleRetriever의 한 축)

임베딩 모델: OpenAI text-embedding-3-small (1536차원)
저장 경로: data/chroma_db/ (영속 저장)
패키지: langchain-chroma 1.1.0 + chromadb 1.5.x
"""

import os
from pathlib import Path

# 텔레메트리 비활성화 — 콘솔 노이즈 제거 (실행보다 먼저 설정해야 함)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── 경로 설정 ────────────────────────────────────────────────
CHROMA_PERSIST_DIR = Path("data/chroma_db")

# src/rag/chroma_store.py 컬렉션 정의 주석 수정
COLLECTIONS = {
    "news_articles":         "뉴스/공시 (매 실행)",
    "analyst_reports":       "증권사 리포트 (주 단위)",
    "market_reports":        "시장 현황 보고서 (주 단위)",  # ← 월→주 수정
    "strategy_outcomes":     "전략 결과 피드백 (매 실행 후)",
    "economic_indicators":   "경제지표 해석 (주 단위)",
    "earnings_data":         "실적 데이터 (분기 단위)",
}


EMBEDDING_MODEL = "text-embedding-3-small"


class _BudgetedEmbeddings(OpenAIEmbeddings):
    """토큰 수를 LLMBudget에 기록하는 OpenAIEmbeddings.

    임베딩 API 응답은 LangChain 경로에서 usage를 노출하지 않으므로
    tiktoken(cl100k_base)으로 입력 토큰을 직접 센다. 출력 토큰은 0.
    집계 실패가 검색을 막으면 안 되므로 예외는 전부 삼킨다.
    """

    def _record(self, texts: list[str]) -> None:
        try:
            import tiktoken
            from src.utils.llm_budget import get_budget
            enc = tiktoken.get_encoding("cl100k_base")
            n = sum(len(enc.encode(t)) for t in texts)
            get_budget().record(EMBEDDING_MODEL, n, 0)
        except Exception:
            pass

    def embed_documents(self, texts: list[str], *args, **kwargs) -> list[list[float]]:
        self._record(texts)
        return super().embed_documents(texts, *args, **kwargs)

    def embed_query(self, text: str, *args, **kwargs) -> list[float]:
        self._record([text])
        return super().embed_query(text, *args, **kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI text-embedding-3-small 임베딩 모델 반환 (토큰 사용량 집계 포함)."""
    return _BudgetedEmbeddings(model=EMBEDDING_MODEL)


def get_collection(collection_name: str) -> Chroma:
    """ChromaDB 컬렉션을 LangChain Chroma 객체로 반환.

    Args:
        collection_name: COLLECTIONS에 정의된 6개 컬렉션 중 하나

    Returns:
        langchain_chroma.Chroma 객체 (저장/검색/retriever 변환 가능)

    Raises:
        ValueError: 허용되지 않은 컬렉션 이름
    """
    if collection_name not in COLLECTIONS:
        raise ValueError(
            f"알 수 없는 컬렉션: '{collection_name}'. "
            f"허용된 컬렉션: {list(COLLECTIONS.keys())}"
        )

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def add_documents(collection_name: str, documents: list[Document]) -> int:
    """문서 리스트를 컬렉션에 임베딩 후 저장.

    Returns:
        저장된 문서 수
    """
    if not documents:
        return 0

    db = get_collection(collection_name)
    db.add_documents(documents)
    return len(documents)


def search(
    collection_name: str,
    query: str,
    k: int = 10,
    filter: dict | None = None,
) -> list[Document]:
    """벡터 유사도 검색 (Dense Retrieval).

    Note:
        단독 사용보다 hybrid_retriever.py의 EnsembleRetriever 사용 권장.
    """
    db = get_collection(collection_name)

    if filter:
        return db.similarity_search(query, k=k, filter=filter)
    return db.similarity_search(query, k=k)


def get_collection_count(collection_name: str) -> int:
    """컬렉션에 저장된 문서 수 반환."""
    db = get_collection(collection_name)
    try:
        # langchain_chroma 1.x + chromadb 1.x
        return db._collection.count()
    except AttributeError:
        # 내부 API가 다를 경우 fallback
        return len(db.get()["ids"])