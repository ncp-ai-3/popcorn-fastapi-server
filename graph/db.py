"""DB 연결·RAG 임계값."""

import logging
import os

import psycopg2

logger = logging.getLogger(__name__)


def _db_password() -> str:
    raw = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD") or ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw


def _db_connect_kwargs():
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = _db_password()
    port = os.getenv("DB_PORT")
    missing = [
        k
        for k, v in [
            ("DB_HOST", host),
            ("DB_NAME", database),
            ("DB_USER", user),
            ("DB_PASS or DB_PASSWORD", password or None),
            ("DB_PORT", port),
        ]
        if not v
    ]
    if missing:
        logger.error("DB 설정 누락: %s", ", ".join(missing))
    return host, database, user, password, port


def get_db_connection():
    host, database, user, password, port = _db_connect_kwargs()
    return psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port,
    )


def rag_cosine_distance_max() -> float:
    raw = os.getenv("RAG_COSINE_DISTANCE_MAX", "0.4").strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("RAG_COSINE_DISTANCE_MAX=%r invalid; using 0.4", raw)
        return 0.4
