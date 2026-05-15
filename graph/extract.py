"""LLM 기반 키워드·의도·조건(JSON) 추출."""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from config.prompt_loader import build_chatml, load_prompts, render_template
from graph.recommend_llm import complete_extract_prompt

logger = logging.getLogger(__name__)

_STOP = ["<|im_end|>"]


def _llm_complete(prompt: str, *, max_tokens: int) -> str:
    return complete_extract_prompt(
        prompt,
        max_tokens=max_tokens,
        stop_sequences=_STOP,
    )


def _log_intent_extract_prompt_stats(prompt: str, query: str) -> None:
    """의도 추출 LLM 호출 직전: system 블록 / few-shot / 마지막 질문 블록 글자 수."""
    sys_open = "<|im_start|>system\n"
    sys_close = "<|im_end|>"
    s = prompt.find(sys_open)
    e = prompt.find(sys_close, s + 1) if s >= 0 else -1
    if s < 0 or e < 0:
        logger.info("[LLM:intent_extract] prompt_chars total=%d (구간분석_생략)", len(prompt))
        return
    system_body = prompt[s : e + len(sys_close)]
    tail_marker = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    if prompt.endswith(tail_marker):
        mid = prompt[e + len(sys_close) : -len(tail_marker)]
        logger.info(
            "[LLM:intent_extract] prompt_chars system=%d fewshot_예시=%d 마지막질문_블록=%d total=%d",
            len(system_body),
            len(mid),
            len(tail_marker),
            len(prompt),
        )
    else:
        mid = prompt[e + len(sys_close) :]
        logger.info(
            "[LLM:intent_extract] prompt_chars system=%d 나머지=%d total=%d",
            len(system_body),
            len(mid),
            len(prompt),
        )


def _log_embed_compose_prompt_stats(prompt: str) -> None:
    """임베딩 문장 compose LLM: system 쪽 / user 이후."""
    u = "<|im_start|>user\n"
    i = prompt.find(u)
    if i < 0:
        logger.info("[LLM:embed_compose] prompt_chars total=%d", len(prompt))
        return
    sys_part = prompt[: i + len(u)]
    rest = prompt[i + len(u) :]
    logger.info(
        "[LLM:embed_compose] prompt_chars system_및_user_태그까지=%d user본문_및_assistant=%d total=%d",
        len(sys_part),
        len(rest),
        len(prompt),
    )


def _log_keyword_extract_prompt_stats(prompt: str, query: str) -> None:
    """검색 키워드 추출 LLM."""
    tail = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    if prompt.endswith(tail):
        head = prompt[: -len(tail)]
        logger.info(
            "[LLM:keyword_extract] prompt_chars 예시_prefix=%d 마지막질문_블록=%d total=%d",
            len(head),
            len(tail),
            len(prompt),
        )
    else:
        logger.info("[LLM:keyword_extract] prompt_chars total=%d", len(prompt))


def _preview_for_log(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [{len(text) - max_chars} chars truncated]"


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def parse_llm_json_object(text: str) -> dict[str, Any] | None:
    s = _strip_json_fences(text)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패: %s", _preview_for_log(text, 500))
        return None


# DB/서비스와 동일한 category 문자열. extract_intent JSON의 category·아래 힌트의 우측 값과 일치시킴.
DB_POPUP_CATEGORIES: tuple[str, ...] = (
    "연예/크리에이터",
    "캐릭터/IP",
    "전시",
    "인테리어/리빙",
    "소품/굿즈",
    "푸드/음료",
    "뷰티/헬스",
    "패션",
    "디지털/게임/e스포츠",
    "콘텐츠/문화",
    "키친/가전",
    "여행/레저/스포츠",
    "패밀리/라이프",
)

# 질문에 나온 단서(좌) → 위 DB category (우). augment·strip_ungrounded 에서만 사용 (DB 조회 SQL 아님).
_DB_CATEGORY_QUERY_HINTS: tuple[tuple[str, str], ...] = (
    ("e스포츠", "디지털/게임/e스포츠"),
    ("이스포츠", "디지털/게임/e스포츠"),
    ("게임", "디지털/게임/e스포츠"),
    ("디지털", "디지털/게임/e스포츠"),
    ("크리에이터", "연예/크리에이터"),
    ("유튜브", "연예/크리에이터"),
    ("유튜버", "연예/크리에이터"),
    ("연예", "연예/크리에이터"),
    ("아이돌", "연예/크리에이터"),
    ("애니메이션", "캐릭터/IP"),
    ("애니", "캐릭터/IP"),
    ("캐릭터", "캐릭터/IP"),
    ("원신", "캐릭터/IP"),
    ("IP", "캐릭터/IP"),
    ("전시", "전시"),
    ("미술", "전시"),
    ("갤러리", "전시"),
    ("인테리어", "인테리어/리빙"),
    ("리빙", "인테리어/리빙"),
    ("가구", "인테리어/리빙"),
    ("문구", "소품/굿즈"),
    ("소품", "소품/굿즈"),
    ("굿즈", "소품/굿즈"),
    ("맛집", "푸드/음료"),
    ("푸드", "푸드/음료"),
    ("음료", "푸드/음료"),
    ("디저트", "푸드/음료"),
    ("카페", "푸드/음료"),
    ("음식", "푸드/음료"),
    ("메이크업", "뷰티/헬스"),
    ("스킨케어", "뷰티/헬스"),
    ("화장품", "뷰티/헬스"),
    ("헬스", "뷰티/헬스"),
    ("뷰티", "뷰티/헬스"),
    ("의류", "패션"),
    ("옷", "패션"),
    ("패션", "패션"),
    ("OTT", "콘텐츠/문화"),
    ("드라마", "콘텐츠/문화"),
    ("영화", "콘텐츠/문화"),
    ("웹툰", "콘텐츠/문화"),
    ("문화", "콘텐츠/문화"),
    ("주방", "키친/가전"),
    ("가전", "키친/가전"),
    ("키친", "키친/가전"),
    ("캠핑", "여행/레저/스포츠"),
    ("골프", "여행/레저/스포츠"),
    ("여행", "여행/레저/스포츠"),
    ("레저", "여행/레저/스포츠"),
    ("스포츠", "여행/레저/스포츠"),
    ("육아", "패밀리/라이프"),
    ("키즈", "패밀리/라이프"),
    ("패밀리", "패밀리/라이프"),
)

# LLM이 예전 라벨로 줄 때 canonical 로 정규화 (그 다음 strip이 근거 검증).
_CATEGORY_VALUE_ALIASES: dict[str, str] = {
    "뷰티": "뷰티/헬스",
    "음식": "푸드/음료",
    "애니메이션": "캐릭터/IP",
    "캐릭터": "캐릭터/IP",
    "굿즈": "소품/굿즈",
}


def _query_mentions_popup(q: str) -> bool:
    q_lower = q.lower()
    return (
        "팝업" in q
        or "팝업스토어" in q
        or "팝업 스토어" in q
        or "pop-up" in q_lower
        or "popup" in q_lower
    )


# 질문에 팝업 표현이 없고, 아래 주제만 있으면 out_of_domain (DB category 와 무관).
_OUT_OF_DOMAIN_TOPIC_NEEDLES: tuple[str, ...] = (
    "자동차",
    "차량",
    "오토모빌",
    "주식",
    "코인",
    "비트코인",
    "암호화폐",
    "부동산",
    "아파트 분양",
    "날씨",
    "기상",
    "코딩",
    "프로그래밍",
    "파이썬",
    "자바",
    "주식시장",
    "정치",
    "대통령",
    "환율",
)


def coerce_intent_for_popup_queries(query: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """질문에 팝업 관련 표현이 있는데 LLM이 out_of_domain/chitchat으로 준 경우 → action 보정."""
    raw = (extracted.get("intent") or "").strip().lower()
    if raw == "action":
        return extracted
    q = (query or "").strip()
    if not q:
        return extracted
    if raw in ("out_of_domain", "chitchat") and _query_mentions_popup(q):
        logger.info(
            "[intent_coerce] was=%s -> action (popup-related wording) q=%r",
            raw,
            q[:200],
        )
        return {**extracted, "intent": "action"}
    return extracted


def coerce_off_topic_to_out_of_domain(
    query: str, extracted: dict[str, Any]
) -> dict[str, Any]:
    """팝업 DB 주제가 아닌 질문(자동차·주식 등) → out_of_domain. '팝업'이 있으면 적용 안 함."""
    q = (query or "").strip()
    if not q or _query_mentions_popup(q):
        return extracted
    if (extracted.get("intent") or "").strip().lower() == "out_of_domain":
        return extracted
    for needle in _OUT_OF_DOMAIN_TOPIC_NEEDLES:
        if needle in q:
            logger.info(
                "[intent_coerce] -> out_of_domain (off-topic needle=%r) q=%r",
                needle,
                q[:200],
            )
            return {
                "intent": "out_of_domain",
                "category": None,
                "start_date": None,
                "end_date": None,
                "location": None,
            }
    cat = extracted.get("category")
    if cat is not None and str(cat).strip() and str(cat).strip() not in DB_POPUP_CATEGORIES:
        logger.info(
            "[intent_coerce] -> out_of_domain (category not in DB) category=%r q=%r",
            cat,
            q[:200],
        )
        return {
            "intent": "out_of_domain",
            "category": None,
            "start_date": None,
            "end_date": None,
            "location": None,
        }
    return extracted


def augment_action_extracted(query: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """LLM이 category를 비운 action 질문에 대해, 질문 문구만으로 주제 힌트 보강."""
    if (extracted.get("intent") or "").strip().lower() != "action":
        return extracted
    if (
        extracted.get("category")
        or extracted.get("location")
        or extracted.get("start_date")
        or extracted.get("end_date")
    ):
        return extracted
    q = query or ""
    for needle, cat in _DB_CATEGORY_QUERY_HINTS:
        if needle in q:
            return {**extracted, "category": cat}
    return extracted


def normalize_extracted_category(extracted: dict[str, Any]) -> dict[str, Any]:
    """LLM이 옛 라벨(뷰티, 음식 등)로 준 category를 DB 도메인 문자열로 맞춤."""
    if (extracted.get("intent") or "").strip().lower() != "action":
        return extracted
    cat = extracted.get("category")
    if cat is None or not str(cat).strip():
        return extracted
    cat_s = str(cat).strip()
    canon = _CATEGORY_VALUE_ALIASES.get(cat_s)
    if canon:
        return {**extracted, "category": canon}
    return extracted


def strip_ungrounded_category(query: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """category 의미 매핑은 LLM에 위임. 코드는 DB 멤버십만 검사한다.

    LLM이 동의어를 정확히 13개 DB 카테고리로 매핑했는지 신뢰하고, 목록 밖 라벨만 환각으로
    간주해 null로 떨어뜨린다. 단어 근거 휴리스틱은 제거됨 (예: "먹거리" 같은 동의어가 사전에
    없어도 통과시키기 위해).
    """
    if (extracted.get("intent") or "").strip().lower() != "action":
        return extracted
    cat = extracted.get("category")
    if cat is None or not str(cat).strip():
        return extracted
    cat_s = str(cat).strip()
    if cat_s in DB_POPUP_CATEGORIES:
        return extracted
    logger.info(
        "[intent_sanitize] db에 없는 category=%r -> null q_preview=%r",
        cat_s,
        (query or "")[:200],
    )
    return {**extracted, "category": None}


def _few_shot_turns(cfg: dict[str, Any], *, vars: dict[str, str] | None = None) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for shot in cfg.get("few_shots") or []:
        turns.append(
            {
                "role": "user",
                "content": render_template(shot["user"], **(vars or {})),
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": render_template(
                    (shot.get("assistant") or "").strip(), **(vars or {})
                ),
            }
        )
    return turns


def extract_search_keywords(query: str) -> str:
    cfg = load_prompts()["keyword_extract"]
    prompt = build_chatml(
        system=cfg["system"].strip(),
        turns=_few_shot_turns(cfg),
        final_user=query,
    )
    logger.debug(
        "[LLM:keyword_extract] to_llm: original_query=%r prompt_len=%d prompt=\n%s",
        query,
        len(prompt),
        _preview_for_log(prompt, max_chars=6000),
    )
    _log_keyword_extract_prompt_stats(prompt, query)
    t_llm = time.perf_counter()
    extracted = _llm_complete(prompt, max_tokens=64)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    logger.debug("[LLM:keyword_extract] from_llm: extracted_for_embedding=%r", extracted)
    logger.debug(
        "[keyword_extract:done] llm_elapsed_ms=%.1f original_len=%d extracted_len=%d "
        "original=%r -> extracted=%r",
        llm_ms,
        len(query),
        len(extracted),
        query,
        extracted,
    )
    return extracted


def _slot_nonempty_for_embed(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def _merge_embed_lines(a: str, b: str, max_len: int = 256) -> str:
    """검색용 두 줄을 어절로 합치며 중복 토큰 제거(순서: a 먼저)."""
    if not (a or "").strip():
        return ((b or "").strip())[:max_len]
    if not (b or "").strip():
        return ((a or "").strip())[:max_len]

    def toks(x: str) -> list[str]:
        return [t for t in re.split(r"\s+", x.strip()) if t]

    seen: set[str] = set()
    out: list[str] = []
    for t in toks(a) + toks(b):
        if t not in seen:
            seen.add(t)
            out.append(t)
    s = " ".join(out).strip()
    return s if len(s) <= max_len else s[:max_len]


def _embedding_output_suspicious(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 256:
        return True
    # 짧은 줄에서만: 모델이 "없습니다" 한 마디만 내는 경우 등 (긴 검색문 안의 부정은 허용)
    if len(t) <= 48:
        for n in ("없습니다", "없어요", "없다"):
            if n in t:
                return True
    needles = (
        "불가능",
        "도와드릴 수 없",
        "도와 드릴 수 없",
        "제공할 수 없",
        "답변할 수 없",
        "죄송",
        "sorry",
        "cannot",
        "찾을 수 없",
        "팝업은 없",
        "팝업이 없",
    )
    tl = t.lower()
    for n in needles:
        if n in t or n in tl:
            return True
    return False


def _fallback_embed_line_from_conditions(query: str, sc: dict[str, Any]) -> str:
    bits: list[str] = []
    for k in ("category", "location", "start_date", "end_date"):
        v = sc.get(k)
        if _slot_nonempty_for_embed(v):
            bits.append(str(v).strip())
    s = " ".join(bits).strip()
    if s and "팝업" not in s and "popup" not in s.lower():
        s += " 팝업"
    slot_part = s.strip()
    q = (query or "").strip()

    # category 가 비면 JSON만으로는 주제(예: 화장품)가 빠지므로 원문 키워드 추출과 합침
    if not _slot_nonempty_for_embed(sc.get("category")) and q:
        kw = extract_search_keywords(q).strip()
        if kw:
            merged = _merge_embed_lines(slot_part, kw) if slot_part else kw
            if merged:
                return merged[:256]

    return slot_part if slot_part else q


def embedding_text_without_llm(
    query: str, search_conditions: dict[str, Any] | None
) -> str:
    """임베딩 API용 한 줄. LLM 없이 search_conditions 슬롯만 조합하고, 비면 원문 질문."""
    sc = dict(search_conditions) if search_conditions else {}
    bits: list[str] = []
    for k in ("category", "location", "start_date", "end_date"):
        v = sc.get(k)
        if _slot_nonempty_for_embed(v):
            bits.append(str(v).strip())
    s = " ".join(bits).strip()
    if s and "팝업" not in s and "popup" not in s.lower():
        s += " 팝업"
    return s if s else (query or "").strip()


def compose_embedding_text_from_search_conditions(
    query: str,
    search_conditions: dict[str, Any] | None,
) -> str:
    """의도·조건 JSON(search_conditions)을 넣어 임베딩 API용 한 줄 문장을 LLM으로 재구성."""
    sc = dict(search_conditions) if search_conditions else {}
    has_slot = (
        _slot_nonempty_for_embed(sc.get("category"))
        or _slot_nonempty_for_embed(sc.get("location"))
        or _slot_nonempty_for_embed(sc.get("start_date"))
        or _slot_nonempty_for_embed(sc.get("end_date"))
    )
    if not has_slot:
        logger.info(
            "[embed_compose] 조건 슬롯 비어 있음 → 기존 extract_search_keywords 경로"
        )
        return extract_search_keywords(query)

    key_subset = {
        "intent": sc.get("intent"),
        "category": sc.get("category"),
        "start_date": sc.get("start_date"),
        "end_date": sc.get("end_date"),
        "location": sc.get("location"),
    }
    slim = json.dumps(key_subset, ensure_ascii=False)
    ecfg = load_prompts()["embed_compose"]
    user_block = render_template(
        ecfg["user_template"].strip(),
        slim=slim,
        query=query,
    )
    prompt = build_chatml(
        system=ecfg["system"].strip(),
        final_user=user_block,
    )
    logger.debug(
        "[LLM:embed_compose] to_llm: query_preview=%r slim=%s",
        (query or "")[:200],
        slim,
    )
    _log_embed_compose_prompt_stats(prompt)
    t_llm = time.perf_counter()
    line = _llm_complete(prompt, max_tokens=96)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    line = line.split("\n")[0].strip()
    if line.startswith("AI:"):
        line = line[3:].lstrip()
    logger.debug(
        "[LLM:embed_compose] from_llm: %r llm_elapsed_ms=%.1f",
        line,
        llm_ms,
    )
    if _embedding_output_suspicious(line):
        fb = _fallback_embed_line_from_conditions(query, sc)
        logger.warning(
            "[embed_compose] LLM 출력 의심 line_preview=%r -> fallback=%r",
            line[:200],
            fb,
        )
        return fb
    return line


def extract_intent_and_conditions(query: str) -> dict[str, Any]:
    """의도·검색 조건 JSON 추출. 실패 시 chitchat + null 필드."""
    from config.chat_llm import get_chat_llm_config

    from datetime import timedelta

    from graph.popup_date_filter import format_reference_context, reference_now_for_extract

    intent_max = get_chat_llm_config().extract_intent_max_tokens
    now = reference_now_for_extract()
    ref_ctx = format_reference_context(now)
    today_d = now.date()
    days_to_sun = (6 - today_d.weekday()) % 7
    this_week_sun = today_d + timedelta(days=days_to_sun or 7)
    next_week_sun = this_week_sun + timedelta(days=7)

    icfg = load_prompts()["intent_extract"]
    system = render_template(
        icfg["system"].strip(),
        reference_context=ref_ctx,
        categories=", ".join(DB_POPUP_CATEGORIES),
    )
    shot_vars = {"next_week_sun": next_week_sun.isoformat()}
    prompt = build_chatml(
        system=system,
        turns=_few_shot_turns(icfg, vars=shot_vars),
        final_user=query,
    )
    _log_intent_extract_prompt_stats(prompt, query)
    result_text = _llm_complete(prompt, max_tokens=intent_max)
    parsed = parse_llm_json_object(result_text)
    if parsed is None:
        fb = {
            "intent": "chitchat",
            "category": None,
            "start_date": None,
            "end_date": None,
            "location": None,
        }
        logger.info(
            "[intent_extract] parse_failed query_preview=%r raw_head=%r fallback=%s",
            (query or "")[:200],
            result_text[:400],
            json.dumps(fb, ensure_ascii=False),
        )
        return fb
    logger.info(
        "[intent_extract] query_preview=%r json=%s",
        (query or "")[:200],
        json.dumps(parsed, ensure_ascii=False),
    )
    return parsed
