"""LLM 기반 키워드·의도·조건(JSON) 추출."""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from graph.model import get_llm

logger = logging.getLogger(__name__)

_STOP = ["<|im_end|>"]


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


def coerce_intent_for_popup_queries(query: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """질문에 팝업 관련 표현이 있는데 LLM이 out_of_domain/chitchat으로 준 경우 → action 보정."""
    raw = (extracted.get("intent") or "").strip().lower()
    if raw == "action":
        return extracted
    q = (query or "").strip()
    if not q:
        return extracted
    q_lower = q.lower()
    popup_kw = (
        "팝업" in q
        or "팝업스토어" in q
        or "팝업 스토어" in q
        or "pop-up" in q_lower
        or "popup" in q_lower
    )
    if raw in ("out_of_domain", "chitchat") and popup_kw:
        logger.info(
            "[intent_coerce] was=%s -> action (popup-related wording) q=%r",
            raw,
            q[:200],
        )
        return {**extracted, "intent": "action"}
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
    """질문에 근거 없는 category면 제거 (지역만 묻는데 뷰티 등으로 채운 환각 방지)."""
    if (extracted.get("intent") or "").strip().lower() != "action":
        return extracted
    cat = extracted.get("category")
    if cat is None or not str(cat).strip():
        return extracted
    cat_s = str(cat).strip()
    q = query or ""
    if cat_s in q:
        return extracted
    for needle, mapped in _DB_CATEGORY_QUERY_HINTS:
        if mapped == cat_s and needle in q:
            return extracted
    if cat_s not in DB_POPUP_CATEGORIES:
        logger.info(
            "[intent_sanitize] db에 없는 category=%r -> null q_preview=%r",
            cat_s,
            q[:200],
        )
        return {**extracted, "category": None}
    logger.info(
        "[intent_sanitize] ungrounded category=%r -> null q_preview=%r",
        cat_s,
        q[:200],
    )
    return {**extracted, "category": None}


def extract_search_keywords(query: str) -> str:
    llm = get_llm()
    prompt = f"""<|im_start|>system
당신은 검색 키워드 추출기입니다. 
사용자의 질문에서 불필요한 서술어, 조사, 인사말은 모두 제거하고 띄어쓰기로만 구분된 핵심 명사(지역, 장소, 날짜, 이벤트 종류 등)만 출력하세요.
<|im_end|>
<|im_start|>user
이번 주말에 부산에서 열리는 뷰티 팝업스토어 알려줘<|im_end|>
<|im_start|>assistant
이번 주말 부산 뷰티 팝업스토어<|im_end|>
<|im_start|>user
오늘 성수동에서 하는 캐릭터 팝업스토어 추천해줄래?<|im_end|>
<|im_start|>assistant
오늘 성수동 캐릭터 팝업스토어<|im_end|>
<|im_start|>user
내일 여의도 더현대에서 하는 음식 팝업 있어?<|im_end|>
<|im_start|>assistant
내일 여의도 더현대 음식 팝업<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    logger.debug(
        "[LLM:keyword_extract] to_llm: original_query=%r prompt_len=%d prompt=\n%s",
        query,
        len(prompt),
        _preview_for_log(prompt, max_chars=6000),
    )
    t_llm = time.perf_counter()
    output = llm(prompt, max_tokens=64, stop=_STOP, echo=False)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    extracted = output["choices"][0]["text"].strip()
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
    llm = get_llm()
    prompt = f"""<|im_start|>system
당신은 벡터 검색용 짧은 한국어 문장을 만드는 변환기입니다.
[추출 JSON]에 있는 값만 근거로, 지역·주제(category)·기간(날짜)이 있으면 모두 드러나게 **한 줄** 검색 문장만 출력하세요.
JSON에 없는 행사명·매장명·브랜드는 넣지 마세요. 거절·사과·설명·질문·**있음/없음 판단** 금지. 출력은 그 한 줄뿐입니다.
**"없습니다", "있습니다" 같은 답변형 문장은 절대 쓰지 마세요.** 오직 검색에 쓸 명사구(예: 홍대 패션 팝업)만 쓰세요.
원문 질문은 표현 보조일 뿐이며, JSON 필드와 모순되면 JSON을 따르세요.
<|im_end|>
<|im_start|>user
[추출 JSON]
{slim}

[원문 질문]
{query}
<|im_end|>
<|im_start|>assistant
"""
    logger.debug(
        "[LLM:embed_compose] to_llm: query_preview=%r slim=%s",
        (query or "")[:200],
        slim,
    )
    t_llm = time.perf_counter()
    output = llm(prompt, max_tokens=96, stop=_STOP, echo=False)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    line = (output["choices"][0]["text"] or "").strip()
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
    llm = get_llm()
    today = datetime.now()
    current_date = today.strftime("%Y-%m-%d")

    prompt = f"""<|im_start|>system
당신은 팝업스토어 안내 챗봇의 요청 분석기입니다. 오늘 날짜는 {current_date}입니다.
사용자의 질문을 분석하여 반드시 아래 JSON 형식으로만 응답하세요.

[분류 규칙 - intent]
1. action: 팝업스토어, 놀거리, 장소 추천 및 검색 요청 (조건이 없어도 팝업을 원하면 action)
2. chitchat: 인사, 감사, 감정 표현 등 가벼운 대화 (**팝업·추천·지역 검색이 주 내용이면 chitchat 금지, action**)
3. out_of_domain: 팝업스토어와 **완전히 무관**한 질문만 (주식, 날씨, 코딩, 일반 상식 등). **'팝업' '팝업스토어' 'pop-up' 등이 질문에 있으면 절대 out_of_domain이 아닙니다 — 항상 action입니다.**

[추출 규칙]
- 해당하지 않는 조건은 null로 비워두세요.
- 날짜(시간)는 YYYY-MM-DD 형식의 범위로 추론하세요.
- 지역·날짜 없이 **주제만** 묻는 경우에는 **category**에 아래 목록 중 질문과 맞는 **하나만** 넣으세요.
- **category**: 사용자가 주제/장르를 **직접 말했을 때만** 채우세요. "○○에 팝업 있어?"처럼 지역·시기만 묻고 주제를 말하지 않으면 **반드시 null**입니다. 주제를 추측하지 마세요.
- **category** 허용 값(정확히 동일한 문자열만, 아니면 null): {", ".join(DB_POPUP_CATEGORIES)}

<|im_end|>
<|im_start|>user
이번 주말에 성수동에서 하는 뷰티 팝업 찾아줘<|im_end|>
<|im_start|>assistant
{{
  "intent": "action",
  "category": "뷰티/헬스",
  "start_date": "2026-05-16",
  "end_date": "2026-05-17",
  "location": "성수동"
}}<|im_end|>
<|im_start|>user
안녕! 넌 이름이 뭐야?<|im_end|>
<|im_start|>assistant
{{
  "intent": "chitchat",
  "category": null,
  "start_date": null,
  "end_date": null,
  "location": null
}}<|im_end|>
<|im_start|>user
파이썬으로 크롤링 어떻게 해?<|im_end|>
<|im_start|>assistant
{{
  "intent": "out_of_domain",
  "category": null,
  "start_date": null,
  "end_date": null,
  "location": null
}}<|im_end|>
<|im_start|>user
요즘 갈만한 곳 추천해줘<|im_end|>
<|im_start|>assistant
{{
  "intent": "action",
  "category": null,
  "start_date": null,
  "end_date": null,
  "location": null
}}<|im_end|>
<|im_start|>user
애니메이션 팝업은 혹시 있어?<|im_end|>
<|im_start|>assistant
{{
  "intent": "action",
  "category": "캐릭터/IP",
  "start_date": null,
  "end_date": null,
  "location": null
}}<|im_end|>
<|im_start|>user
지금 홍대에서 열리는 팝업 있어?<|im_end|>
<|im_start|>assistant
{{
  "intent": "action",
  "category": null,
  "start_date": null,
  "end_date": null,
  "location": "홍대"
}}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    output = llm(prompt, max_tokens=200, stop=_STOP, echo=False)
    result_text = output["choices"][0]["text"].strip()
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
