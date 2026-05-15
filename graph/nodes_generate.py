"""추천 LLM 답변 생성 (route별 프롬프트)."""

import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from config.chat_llm import get_chat_llm_config
from config.generate_prompts import (
    LeanRecommendPromptParts,
    RecommendPromptParts,
    assemble_lean_recommend_parts,
    assemble_recommend_parts,
)
from config.response_suffixes import missing_slot_suffix, non_action_popup_invite
from config.prompt_loader import chatml_tokens
from graph.recommend_llm import complete_recommend_prompt, recommend_llm_backend
from graph.state import (
    RECOMMEND_HISTORY_TURNS,
    ROUTE_ACTION,
    AgentState,
)

logger = logging.getLogger(__name__)
_STOP_VERTEX = ["<|im_end|>", "</answer>"]
_STOP_LOCAL = ["<|im_end|>"]

_ANSWER_XML_OPEN = "<answer>"
_ANSWER_XML_CLOSE = "</answer>"
_POPUP_SECTION_LOG_PREVIEW_CHARS = 600
_LEAN_POPUP_DESC_CHARS_DEFAULT = 70


def _desc_for_prompt(desc: Any, max_chars: int) -> str:
    t = str(desc or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _log_recommend_prompt_segments(segments: dict[str, str], route: str) -> None:
    """generate LLM 호출 직전: 프롬프트 구간별 글자 수(유니코드 문자 기준)."""
    parts = " ".join(f"{name}={len(s)}" for name, s in segments.items())
    total = sum(len(s) for s in segments.values())
    logger.info(
        "[LLM:recommend] route=%s prompt_char_breakdown total=%d | %s",
        route,
        total,
        parts,
    )


# Gemini 등이 시스템 프롬프트·ChatML을 본문에 베끼는 경우 잘라냄
_ANSWER_LEAK_CUT_MARKERS: tuple[str, ...] = (
    "\n[직전",
    "\n[마무리",
    "\n[답변",
    "\n[조건",
    "\n[이번 턴",
    "\n[팝업",
    "\n[out_of_domain",
    "\n[추천 ",
    "\n[참고 ",
    "\n[말투",
    "\n[역할",
    "\n[출력",
    "\n<|im_start|>",
    "\n<|im_end|>",
    "\nUser:",
    "\n유저:",
    "\nAI:",
    "\nAssistant:",
)


def _extract_answer_xml(text: str) -> tuple[str, bool]:
    """LLM 출력에서 <answer>...</answer> 본문만 추출. found=False면 원문 반환."""
    raw = (text or "").strip()
    if not raw:
        return "", False
    m = re.search(
        r"<answer\s*>(.*?)</answer>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(), True
    m_open = re.search(
        r"<answer\s*>(.*)\Z",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_open:
        return m_open.group(1).strip(), True
    m_close_only = re.search(
        r"^(.*?)</answer>\s*\Z",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_close_only:
        return m_close_only.group(1).strip(), True
    return raw, False


def _sanitize_recommend_answer(answer: str) -> str:
    """프롬프트·역할 라벨 유출 제거."""
    text = (answer or "").strip()
    if not text:
        return text
    for marker in _ANSWER_LEAK_CUT_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()
    for prefix in (
        "[직전",
        "[마무리",
        "[답변",
        "[조건",
        "[이번",
        "[팝업",
        "<|im_start|>",
        "<|im_end|>",
    ):
        if text.startswith(prefix):
            return ""
    return text.strip()


def _ensure_non_action_popup_invite(answer: str, route: str) -> str:
    """action이 아닐 때 답변 말미에 팝업 유도 고정 멘트를 한 번 붙임(이미 있으면 생략)."""
    if route == ROUTE_ACTION:
        return answer
    invite = non_action_popup_invite()
    if not invite:
        return answer
    text = (answer or "").strip()
    if not text:
        return invite
    if invite in text:
        return text
    return f"{text.rstrip()} {invite}".strip()


def _slots_status_lines(state: AgentState) -> str:
    """condition_vector + search_conditions를 LLM용 한국어 몇 줄로 요약."""
    raw_vec = state.get("condition_vector")
    if not raw_vec or len(raw_vec) != 3:
        vec = (0, 0, 0)
    else:
        vec = (int(raw_vec[0]), int(raw_vec[1]), int(raw_vec[2]))
    raw_sc = state.get("search_conditions")
    sc: dict[str, Any] = dict(raw_sc) if isinstance(raw_sc, dict) else {}

    def slot_line(bit: int, label: str, detail: str) -> str:
        if bit:
            return f"- {label}: **채워짐** → {detail}"
        return f"- {label}: **비어 있음** (이 항목은 사용자에게 물어볼 수 있음)"

    cat = sc.get("category")
    cat_d = repr(cat) if cat is not None and str(cat).strip() else "(값 없음)"
    sd, ed = sc.get("start_date"), sc.get("end_date")
    time_d = f"start_date={sd!r}, end_date={ed!r}"
    loc = sc.get("location")
    loc_d = repr(loc) if loc is not None and str(loc).strip() else "(값 없음)"

    return "\n".join(
        [
            f"[조건 벡터] (주제, 시기, 지역) 순서의 이진값 = {vec} — 각 1이면 해당 슬롯이 채워진 것으로 간주합니다.",
            slot_line(vec[0], "주제(category)", cat_d),
            slot_line(vec[1], "시기(start_date / end_date)", time_d),
            slot_line(vec[2], "지역(location)", loc_d),
        ]
    )


def _allowed_titles(popups: list[dict[str, Any]]) -> list[str]:
    return [
        str(p["name"]).strip()
        for p in popups
        if (p.get("name") or "").strip()
    ]


def _build_popup_section(
    route: str,
    popups: list[dict[str, Any]],
    chat_cfg: Any,
) -> tuple[str, list[str]]:
    """Vertex 경로 전용: 기존 [팝업 목록] 블록 (긴 설명 + [추천]/[참고] 라벨)."""
    if route != ROUTE_ACTION:
        return "", []
    desc_limit = chat_cfg.popup_desc_chars
    context_lines: list[str] = []
    for i, p in enumerate(popups):
        d = _desc_for_prompt(p.get("desc"), desc_limit)
        if p.get("is_below_threshold"):
            context_lines.append(f"[추천 {i + 1}] {p['name']} - {d}")
        else:
            context_lines.append(f"[참고 {i + 1}] {p['name']} - {d}")
    if not context_lines:
        context_lines.append("(해당하는 팝업스토어 없음)")
    return f"\n[팝업 목록]:\n" + "\n".join(context_lines) + "\n", _allowed_titles(popups)


def _build_lean_popup_block(
    popups: list[dict[str, Any]],
    *,
    desc_chars: int | None = None,
) -> tuple[str, list[str]]:
    """로컬 SLM용 짧은 팝업 블록 — `1. 이름 - 70자 설명` 한 줄씩."""
    cap_env = os.getenv("LLM_POPUP_DESC_CHARS_LOCAL")
    if desc_chars is None:
        try:
            desc_chars = int(cap_env) if cap_env else _LEAN_POPUP_DESC_CHARS_DEFAULT
        except ValueError:
            desc_chars = _LEAN_POPUP_DESC_CHARS_DEFAULT
    desc_chars = max(20, int(desc_chars))
    if not popups:
        return "[팝업]\n(없음)", []
    lines: list[str] = ["[팝업]"]
    for i, p in enumerate(popups, start=1):
        name = str(p.get("name") or "").strip()
        d = _desc_for_prompt(p.get("desc"), desc_chars)
        lines.append(f"{i}. {name} - {d}" if d else f"{i}. {name}")
    return "\n".join(lines), _allowed_titles(popups)


def _log_recommend_popup_section(
    route: str,
    popups: list[dict[str, Any]],
    popup_section: str,
    allowed_titles: list[str],
) -> None:
    """SLM/Vertex 추천 프롬프트에 실제로 들어가는 [팝업 목록] 확인용."""
    if route != ROUTE_ACTION:
        return
    line_bits: list[str] = []
    for p in popups:
        tier = "추천" if p.get("is_below_threshold") else "참고"
        cos = p.get("cosine_distance")
        cos_bit = f" cos={float(cos):.4f}" if cos is not None else ""
        line_bits.append(f"id={p.get('id')} [{tier}] {p.get('name')!r}{cos_bit}")
    preview = popup_section.replace("\n", " ").strip()
    if len(preview) > _POPUP_SECTION_LOG_PREVIEW_CHARS:
        preview = preview[: _POPUP_SECTION_LOG_PREVIEW_CHARS - 1] + "…"
    logger.info(
        "[LLM:recommend] popup_section count=%d chars=%d allowed_titles=%d | %s",
        len(popups),
        len(popup_section),
        len(allowed_titles),
        " | ".join(line_bits) if line_bits else "(empty)",
    )
    logger.info("[LLM:recommend] popup_section_preview=%r", preview)
    logger.debug("[LLM:recommend] popup_section_full:\n%s", popup_section.rstrip())


def _chatml_prompt_from_parts(
    parts: RecommendPromptParts,
    *,
    user_body: str,
) -> tuple[str, dict[str, str]]:
    t = chatml_tokens()
    seg_intro = f"{t['im_start']}system\n{parts.intro}\n\n"
    seg_output_guard = f"{parts.output_guard}\n"
    seg_rules_nl = f"{parts.rules}\n"
    seg_allowed_extra = f"{parts.allowed_names}{parts.route_context}\n\n"
    seg_popup = parts.popup_section
    system_body = (
        seg_intro
        + seg_output_guard
        + seg_rules_nl
        + seg_allowed_extra
        + seg_popup
    )
    seg_sys_end = f"{t['im_end']}\n"
    few_shot_parts: list[str] = []
    for turn in parts.few_shot_turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        few_shot_parts.append(f"{t['im_start']}{role}\n{content}{t['im_end']}\n")
    seg_few_shots = "".join(few_shot_parts)
    seg_user = f"{t['im_start']}user\n{user_body}{t['im_end']}\n"
    seg_asst = f"{t['im_start']}assistant\n{_ANSWER_XML_OPEN}\n"
    prompt = system_body + seg_sys_end + seg_few_shots + seg_user + seg_asst
    segments = {
        "1_system": system_body + seg_sys_end,
        "2_few_shots": seg_few_shots,
        "3_user": seg_user,
        "4_assistant_prefill": seg_asst,
    }
    return prompt, segments


def _chatml_prompt_from_parts_lean(
    parts: LeanRecommendPromptParts,
    *,
    user_body: str,
) -> tuple[str, dict[str, str]]:
    """로컬 SLM용 ChatML — XML prefill 없음, system 한 덩어리."""
    t = chatml_tokens()
    seg_system = f"{t['im_start']}system\n{parts.system_text}{t['im_end']}\n"
    few_shot_parts: list[str] = []
    for turn in parts.few_shot_turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        few_shot_parts.append(f"{t['im_start']}{role}\n{content}{t['im_end']}\n")
    seg_few_shots = "".join(few_shot_parts)
    seg_user = f"{t['im_start']}user\n{user_body}{t['im_end']}\n"
    seg_asst = f"{t['im_start']}assistant\n"
    prompt = seg_system + seg_few_shots + seg_user + seg_asst
    segments = {
        "1_system": seg_system,
        "2_few_shots": seg_few_shots,
        "3_user": seg_user,
        "4_assistant_prefill": seg_asst,
    }
    return prompt, segments


def _filter_hallucinated_names(
    answer: str,
    allowed_titles: list[str],
    route: str,
    popups: list[dict[str, Any]] | None = None,
) -> str:
    """action인데 답변에 허용 행사명이 하나도 없으면 자연어 폴백으로 대체."""
    if route != ROUTE_ACTION:
        return answer
    if not allowed_titles:
        return answer
    text = (answer or "").strip()
    if not text:
        # 빈 답변일 때도 첫 팝업으로 안내
        first = allowed_titles[0]
        return f"{first}이(가) 마침 열려 있어요. 한 번 들러보세요."
    for t in allowed_titles:
        if t and t in text:
            return text
    first = allowed_titles[0]
    fallback = f"{first}이(가) 마침 열려 있어요. 한 번 들러보세요."
    logger.warning(
        "[LLM:recommend] fallback used reason=allowed_titles_missing "
        "raw_chars=%d first_title=%r",
        len(text),
        first,
    )
    return fallback


def generate_recommendation(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state["user_query"]
    route = state.get("route", ROUTE_ACTION)
    popups = state.get("retrieved_popups") or []
    history = state.get("history") or []
    chat_cfg = get_chat_llm_config()
    logger.info(
        "[Node:generate] enter route=%s recommend_backend=%s popup_count=%d history_lines=%d",
        route,
        chat_cfg.recommend_backend,
        len(popups),
        len(history),
    )

    is_local = chat_cfg.recommend_backend == "local"
    sc = state.get("search_conditions") or {}
    category = sc.get("category") if isinstance(sc, dict) else None
    category_label = (
        str(category).strip() if category is not None and str(category).strip() else "관심"
    )

    _recent_line_count = RECOMMEND_HISTORY_TURNS * 2
    history_context = "\n".join(history[-_recent_line_count:]).strip()

    if is_local:
        if route == ROUTE_ACTION:
            popup_block, allowed_titles = _build_lean_popup_block(popups)
        else:
            popup_block, allowed_titles = "", []
        _log_recommend_popup_section(route, popups, popup_block, allowed_titles)

        lean_parts = assemble_lean_recommend_parts(route, category=category_label)
        if route == ROUTE_ACTION:
            user_body = f"{popup_block}\n질문: {query}"
        else:
            user_body = query
        if history_context:
            user_body = f"이전 대화:\n{history_context}\n\n{user_body}"

        prompt, segments = _chatml_prompt_from_parts_lean(
            lean_parts, user_body=user_body
        )
        stop_sequences = _STOP_LOCAL
    else:
        popup_section, allowed_titles = _build_popup_section(route, popups, chat_cfg)
        _log_recommend_popup_section(route, popups, popup_section, allowed_titles)

        parts = assemble_recommend_parts(
            route,
            current_time=datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
            answer_open=_ANSWER_XML_OPEN,
            answer_close=_ANSWER_XML_CLOSE,
            slots_block=_slots_status_lines(state),
            allowed_titles=allowed_titles,
            popup_section=popup_section,
            category=category_label,
        )
        user_body = query
        if history_context:
            user_body = f"이전 대화:\n{history_context}\n\n현재 질문:\n{query}"

        prompt, segments = _chatml_prompt_from_parts(parts, user_body=user_body)
        stop_sequences = _STOP_VERTEX

    _log_recommend_prompt_segments(segments, route)

    answer_raw = complete_recommend_prompt(
        prompt,
        max_tokens=chat_cfg.generate_max_tokens,
        stop_sequences=stop_sequences,
    )
    if not (answer_raw or "").strip():
        logger.error(
            "[LLM:recommend] empty model output backend=%s",
            recommend_llm_backend(),
        )
        answer = "잠시 답변을 만들지 못했습니다. 같은 질문으로 다시 시도해 주세요."
    else:
        answer = answer_raw.strip()
    if answer.startswith("AI:"):
        answer = answer[3:].lstrip()
    if is_local:
        # lean 경로는 XML prefill을 쓰지 않지만, 모델이 자발적으로 감쌀 수 있음
        parsed, found_xml = _extract_answer_xml(answer)
        if found_xml and parsed != answer:
            logger.info(
                "[LLM:recommend] lean: stripped self-emitted <answer> xml inner_chars=%d",
                len(parsed),
            )
            answer = parsed
    else:
        parsed, found_xml = _extract_answer_xml(answer)
        if found_xml:
            if parsed != answer:
                logger.info(
                    "[LLM:recommend] extracted <answer> xml inner_chars=%d",
                    len(parsed),
                )
            answer = parsed
        else:
            logger.warning(
                "[LLM:recommend] <answer> XML not found; using raw after sanitize route=%s",
                route,
            )
    cleaned = _sanitize_recommend_answer(answer)
    if cleaned != answer:
        logger.warning(
            "[LLM:recommend] sanitized leaked prompt markers raw_chars=%d -> %d",
            len(answer),
            len(cleaned),
        )
        answer = cleaned

    if is_local:
        answer = _filter_hallucinated_names(answer, allowed_titles, route, popups)

    answer = _ensure_non_action_popup_invite(answer, route)

    if route == ROUTE_ACTION:
        suf = missing_slot_suffix(state.get("condition_vector"))
        if suf and suf not in (answer or ""):
            answer = f"{(answer or '').rstrip()} {suf}".strip()

    new_history = history + [f"User: {query}", f"AI: {answer}"]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    ans_preview = (answer[:80] + "…") if len(answer) > 80 else answer
    logger.info(
        "[Node:generate] exit route=%s backend=%s elapsed_ms=%.1f answer_chars=%d preview=%r",
        route,
        chat_cfg.recommend_backend,
        elapsed_ms,
        len(answer),
        ans_preview.replace("\n", " "),
    )
    return {
        "final_answer": answer,
        "history": new_history,
    }
