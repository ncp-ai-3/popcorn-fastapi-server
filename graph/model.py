"""로컬 Llama(GGUF) 단일 인스턴스."""

import logging
import os

from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()
logger = logging.getLogger(__name__)


def _llm_model_path() -> str:
    return os.getenv(
        "LLM_MODEL_PATH",
        "./qwen2-1_5b-instruct-q4_k_m.gguf",
    ).strip()


def _llm_n_gpu_layers() -> int:
    raw = os.getenv("LLM_N_GPU_LAYERS", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("LLM_N_GPU_LAYERS=%r invalid; using 0", raw)
        return 0


def _llm_n_ctx() -> int:
    raw = os.getenv("LLM_N_CTX", "2048").strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return 2048


_model_path = _llm_model_path()
_n_gpu = _llm_n_gpu_layers()
_n_ctx = _llm_n_ctx()
logger.info(
    "Loading LLM model_path=%s n_gpu_layers=%d n_ctx=%d",
    _model_path,
    _n_gpu,
    _n_ctx,
)
llm = Llama(
    model_path=_model_path,
    n_gpu_layers=_n_gpu,
    n_ctx=_n_ctx,
    verbose=False,
)


def get_llm() -> Llama:
    return llm
