"""추천 LLM 답변 생성 (route별 프롬프트)."""

import logging
import os
import time
from datetime import datetime
from typing import Any

from graph.model import get_llm
from graph.state import (
    RECOMMEND_HISTORY_TURNS,
    ROUTE_ACTION,
    ROUTE_CHITCHAT,
    ROUTE_CLARIFICATION_ALL,
    ROUTE_CLARIFICATION_LOC,
    ROUTE_OUT_OF_DOMAIN,
    AgentState,
)

logger = logging.getLogger(__name__)
_STOP = ["<|im_end|>"]


def _generate_max_tokens() -> int:
    """추천 답변 생성 시 신규 토큰 상한. 한국어는 512에서도 문장 중간에 끊길 수 있어 기본 1024."""
    raw = os.getenv("LLM_GENERATE_MAX_TOKENS", "1024").strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return 1024

_NON_ACTION_POPUP_INVITE = "다른 찾으시는 팝업 있을까요?"


def _ensure_non_action_popup_invite(answer: str, route: str) -> str:
    """action이 아닐 때 답변 말미에 팝업 유도 고정 멘트를 한 번 붙임(이미 있으면 생략)."""
    if route == ROUTE_ACTION:
        return answer
    text = (answer or "").strip()
    if not text:
        return _NON_ACTION_POPUP_INVITE
    if _NON_ACTION_POPUP_INVITE in text:
        return text
    return f"{text.rstrip()} {_NON_ACTION_POPUP_INVITE}".strip()


# (category, time, location) 비트. action 답변 말미에 고정 문장으로 append.
_MISSING_SLOTS_FIXED_SUFFIX: dict[tuple[int, int, int], str] = {
    (1, 0, 0): "어느 지역에서 찾으실지, 또는 방문하고 싶은 시기(주말·평일 등)를 알려주시면 더 잘 맞춰드릴게요.",
    (0, 1, 0): "관심 주제(뷰티·캐릭터·굿즈 등)나 희망 지역을 알려주시면 검색을 더 좁혀 드릴게요.",
    (0, 0, 1): "어떤 종류의 팝업을 찾는지, 또는 방문 일정을 알려주시면 더 맞춰드릴게요.",
    (1, 1, 0): "희망 지역을 알려주시면 더 정확히 안내드릴게요.",
    (1, 0, 1): "방문 시기나 기간을 알려주시면 더 좋아요.",
    (0, 1, 1): "관심 있는 주제나 스타일을 알려주시면 더 맞춰드릴게요.",
}


def _missing_slots_fixed_suffix(vec: tuple[int, int, int] | None) -> str:
    """조건 벡터 기준: 비어 있는 슬롯 안내를 고정 문장으로( action 답변 말미 append )."""
    if not vec or len(vec) != 3:
        return ""
    key = (int(vec[0]), int(vec[1]), int(vec[2]))
    return _MISSING_SLOTS_FIXED_SUFFIX.get(key, "")


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


def _route_extra_rules(route: str, state: AgentState) -> str:
    if route == ROUTE_ACTION:
        slots = _slots_status_lines(state)
        return "\n".join(
            [
                slots,
                "",
                "[action 턴 참고] 위 [조건 벡터]에서 **채워짐**인 슬롯은 사용자가 이미 준 정보입니다. 그 내용을 다시 묻지 마세요.",
                "**비어 있음**인 슬롯에 대한 질문 문장은 시스템이 답변 말미에 고정 형식으로 붙입니다. "
                "본문에서는 그 부분을 쓰지 말고 팝업 소개만 하세요.",
                "팝업 소개에는 **[팝업 목록]에 적힌 제목 문자열만** 쓰세요.",
            ]
        )
    if route == ROUTE_CLARIFICATION_ALL:
        slots = _slots_status_lines(state)
        return (
            f"{slots}\n\n"
            "[이번 턴 역할] 사용자 조건이 아직 모호합니다. "
            "팝업 목록은 제공되지 않으니, **구체 행사명을 나열하거나 지어내지 마세요.** "
            "위 [조건 벡터]에서 **비어 있음**인 슬롯을 우선 짧게 물어보고, "
            "**채워짐**인 값은 이미 알고 있는 정보로만 언급하세요.\n"
            "[마무리 필수] 팝업을 찾으시면 위 항목을 알려주시면 검색에 도움이 된다고 한 문장으로 덧붙이세요."
        )
    if route == ROUTE_CLARIFICATION_LOC:
        cat = (state.get("search_conditions") or {}).get("category") or "관심"
        return (
            f"[이번 턴 역할] 사용자가 '{cat}' 쪽을 찾는 것 같습니다. "
            "팝업 목록은 없으므로 행사명을 말하지 말고, **어느 지역**에서 찾으실지 짧게 물어보세요.\n"
            "[마무리 필수] 방문 시기나 다른 관심사를 알려주시면 더 좋다고 한 문장 덧붙이세요."
        )
    if route == ROUTE_CHITCHAT:
        return (
            "[이번 턴 역할] 가벼운 대화에 먼저 친절히 응답하세요. "
            "팝업 목록은 없으니 **특정 팝업 이름이나 장소를 지어내지 마세요.**\n"
            "[마무리 필수] 팝업 추천이 필요하시면 지역이나 관심사를 알려달라고 짧게 물어보세요."
        )
    if route == ROUTE_OUT_OF_DOMAIN:
        return (
            "[out_of_domain 답변 형식]\n"
            "**맨 앞 두 가지 뜻**을 자연스러운 한국어로 이어서 말하세요(문장 표현은 다듬어도 되고, 아래 예시와 같은 뉘앙스여야 합니다).\n"
            "1) 지금 물으신 주제로는 저희가 안내하는 **팝업 범위에 해당하는 건 없다**는 뜻 — 예: 「그런 주제의 팝업은 없어요.」\n"
            "2) **다른 관심사**(주제·지역·시기 등)를 물어보는 한 문장 — 예: 「다른 관심사 있으세요?」\n"
            "팝업 목록은 제공되지 않습니다. **구체 팝업·행사명·링크는 쓰지 말고**, 필요하면 팝업을 찾으실 때 지역·주제를 알려달라고 한 문장만 덧붙이면 됩니다."
        )
    return ""


def generate_recommendation(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state["user_query"]
    route = state.get("route", ROUTE_ACTION)
    popups = state.get("retrieved_popups") or []
    history = state.get("history") or []
    logger.debug(
        "[Node:enter] generate route=%s popup_count=%d history_lines=%d",
        route,
        len(popups),
        len(history),
    )

    allowed_names_block = ""
    llm = get_llm()
    if route == ROUTE_ACTION:
        context_lines: list[str] = []
        for i, p in enumerate(popups):
            if p.get("is_below_threshold"):
                context_lines.append(f"[추천 {i + 1}] {p['name']} - {p['desc']}")
            else:
                context_lines.append(f"[참고 {i + 1}] {p['name']} - {p['desc']}")
        if not context_lines:
            context_lines.append("(해당하는 팝업스토어 없음)")
        context = "\n".join(context_lines)
        allowed_titles = [
            str(p["name"]).strip()
            for p in popups
            if (p.get("name") or "").strip()
        ]
        if allowed_titles:
            allowed_names_block = (
                "[이번 턴 인용 허용 행사명 — **아래 줄에 적힌 문자열만** 행사 이름으로 쓸 수 있습니다. "
                "철자·띄어쓰기·기호를 바꾸거나 줄임말·별칭을 쓰지 마세요. **여기 없는 이름은 모두 금지(환각)**입니다.]\n"
                + "\n".join(f"- {t}" for t in allowed_titles)
                + "\n\n"
            )
        rules_block = """[답변 작성 규칙]
1. 정보 제한: 반드시 아래 [팝업 목록]·위 [인용 허용 행사명]에 있는 제목과 내용만 사용하세요. 목록에 없는 행사명·약칭(예: PW 팝업)·가상 브랜드는 **한 글자도 쓰지 마세요.** (목록이 비어 있으면 솔직히 안내하세요.)
2. 대화형 화법: 기계적인 번호 매기기(1, 2, 3)나 기호([추천], [참고])는 답변에 쓰지 마세요. 옆 사람에게 추천하듯 자연스럽게 말하세요.
3. 추천 순서: [추천] 태그가 붙은 팝업을 먼저 자세히 소개하고, [참고] 태그가 붙은 곳은 "시간이 남으면 가보기 좋은 곳" 정도로 가볍게 뒤에 덧붙이세요.
4. **'○○' '△△' 등 자리 표시·예시 이름은 절대 쓰지 마세요.** [말투 참고]에 나온 기호도 답에 넣지 마세요. 오직 [인용 허용 행사명]과 동일한 제목만 부르세요.
5. 부족한 조건(주제·시기·지역)을 묻는 맺음말은 **본문에 넣지 마세요.** 시스템이 말미에 붙입니다.
"""
        example_block = """[말투 참고 — 아래 목록의 제목만 실제 이름으로 사용]
한두 개 팝업을 골라, 목록에 적힌 제목을 그대로 읽어 주듯 소개하고 설명은 목록 본문을 한두 문장만 요약하세요. 가상의 스팟 이름이나 기호는 쓰지 마세요.
"""
        popup_section = f"\n[팝업 목록]:\n{context}\n"
    else:
        rules_block = """[답변 작성 규칙 — 팝업 목록 없음]
이번 턴에는 검색된 팝업 목록이 **제공되지 않습니다.** 사용자 질문·[직전 대화]·아래 [역할] 지시만으로 답하세요.
구체적인 팝업 이름·행사 일정·장소·링크는 **사실 근거 없이 쓰지 마세요.**
"""
        if route == ROUTE_OUT_OF_DOMAIN:
            example_block = """[흐름 예시 — 팝업 이름·행사는 언급하지 않음]
Assistant: 그런 주제의 팝업은 없어요. 다른 관심사 있으세요? 팝업을 찾으시면 지역이나 관심 주제를 알려 주세요.
"""
        elif route == ROUTE_CLARIFICATION_ALL:
            example_block = """[흐름 예시 — 구체 행사명 없이 되묻기]
User: 요즘 갈만한 팝업 있어?
Assistant: 어느 쪽 지역이 편하세요? 그리고 뷰티·굿즈처럼 마음에 드는 주제가 있으면 같이 알려주시면 맞춰 볼게요.
"""
        else:
            example_block = """[말투 예시 — 목록 없음, 팝업 이름은 언급하지 않음]
User: 오늘 날씨 어때?
Assistant: 저는 날씨 쪽은 잘 모르겠어요. 대신 팝업 찾으실 계획이면 지역이나 가고 싶은 주제 알려주시면 그에 맞춰 볼게요.
"""
        popup_section = ""

    _recent_line_count = RECOMMEND_HISTORY_TURNS * 2
    history_context = "\n".join(history[-_recent_line_count:])
    current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    extra = _route_extra_rules(route, state)

    prompt = f"""<|im_start|>system
당신은 팝업스토어를 소개하는 친절하고 자연스러운 대화형 AI 가이드입니다.
현재 시간은 {current_time}입니다. 일정은 이 시간을 기준으로 판단하세요.

{rules_block}
{allowed_names_block}{extra}

{example_block}{popup_section}[직전 {RECOMMEND_HISTORY_TURNS}턴 대화만 참고]:
{history_context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    logger.debug(
        "[LLM:recommend] route=%r prompt_len=%d",
        route,
        len(prompt),
    )
    output = llm(prompt, max_tokens=_generate_max_tokens(), stop=_STOP, echo=False)
    choices = output.get("choices") if isinstance(output, dict) else None
    if not choices:
        logger.error("[LLM:recommend] empty choices output_keys=%s", output.keys() if isinstance(output, dict) else type(output))
        answer = "잠시 답변을 만들지 못했습니다. 같은 질문으로 다시 시도해 주세요."
    else:
        answer = (choices[0].get("text") or "").strip()
    if answer.startswith("AI:"):
        answer = answer[3:].lstrip()

    answer = _ensure_non_action_popup_invite(answer, route)

    if route == ROUTE_ACTION:
        suf = _missing_slots_fixed_suffix(state.get("condition_vector"))
        if suf and suf not in (answer or ""):
            answer = f"{(answer or '').rstrip()} {suf}".strip()

    new_history = history + [f"User: {query}", f"AI: {answer}"]

    logger.debug(
        "[Node:exit] generate elapsed_ms=%.1f answer_chars=%d",
        (time.perf_counter() - t0) * 1000,
        len(answer),
    )
    return {
        "final_answer": answer,
        "history": new_history,
    }
