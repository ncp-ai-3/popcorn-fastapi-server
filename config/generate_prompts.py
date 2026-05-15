"""추천(generate) route별 프롬프트 조각 로드·조립."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.prompt_loader import load_prompts, render_template

# graph.state.ROUTE_* 와 동일한 문자열 (config → graph import 회피)
_ROUTE_ACTION = "action"
_ROUTE_CLARIFICATION_ALL = "clarification_all"
_ROUTE_CLARIFICATION_LOC = "clarification_loc"
_ROUTE_CHITCHAT = "chitchat"
_ROUTE_OUT_OF_DOMAIN = "out_of_domain"

_ROUTE_PROMPT_KEYS: dict[str, str] = {
    _ROUTE_ACTION: "generate_action",
    _ROUTE_CLARIFICATION_ALL: "generate_route_clarification_all",
    _ROUTE_CLARIFICATION_LOC: "generate_route_clarification_loc",
    _ROUTE_CHITCHAT: "generate_route_chitchat",
    _ROUTE_OUT_OF_DOMAIN: "generate_route_out_of_domain",
}

_LEAN_ROUTE_PROMPT_KEYS: dict[str, str] = {
    _ROUTE_ACTION: "lean_action",
    _ROUTE_CLARIFICATION_ALL: "lean_clarification_all",
    _ROUTE_CLARIFICATION_LOC: "lean_clarification_loc",
    _ROUTE_CHITCHAT: "lean_chitchat",
    _ROUTE_OUT_OF_DOMAIN: "lean_out_of_domain",
}

_ROUTES_WITH_SLOT_BLOCK = frozenset({_ROUTE_ACTION, _ROUTE_CLARIFICATION_ALL})


def _strip(text: Any) -> str:
    return (text or "").strip() if text is not None else ""


def _cfg(key: str) -> dict[str, Any]:
    block = load_prompts().get(key)
    if not isinstance(block, dict):
        raise KeyError(f"prompt block missing: {key}")
    return block


def _few_shot_turns(cfg: dict[str, Any], **vars: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for shot in cfg.get("few_shots") or []:
        turns.append(
            {
                "role": "user",
                "content": render_template(_strip(shot.get("user")), **vars),
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": render_template(_strip(shot.get("assistant")), **vars),
            }
        )
    return turns


@dataclass(frozen=True)
class RecommendPromptParts:
    """이번 턴에만 LLM system에 넣을 조각."""

    intro: str
    output_guard: str
    rules: str
    allowed_names: str
    route_context: str
    popup_section: str
    few_shot_turns: tuple[dict[str, str], ...]


def build_allowed_names_block(titles: list[str]) -> str:
    if not titles:
        return ""
    header = _strip(_cfg("generate_base").get("allowed_names_header"))
    return header + "\n" + "\n".join(f"- {t}" for t in titles) + "\n\n"


def build_route_context(
    route: str,
    *,
    slots_block: str,
    category: str | None = None,
) -> str:
    route_key = _ROUTE_PROMPT_KEYS.get(route)
    if not route_key:
        return slots_block.strip()
    route_cfg = _cfg(route_key)
    extra = _strip(route_cfg.get("route_extra"))
    if category is not None:
        extra = render_template(extra, category=category)
    if route in _ROUTES_WITH_SLOT_BLOCK and slots_block.strip():
        return f"{slots_block.strip()}\n\n{extra}".strip() if extra else slots_block.strip()
    return extra


def assemble_recommend_parts(
    route: str,
    *,
    current_time: str,
    answer_open: str,
    answer_close: str,
    slots_block: str,
    allowed_titles: list[str],
    popup_section: str,
    category: str | None = None,
) -> RecommendPromptParts:
    """route에 맞는 YAML 블록만 읽어 이번 턴 프롬프트 조각을 만든다."""
    base = _cfg("generate_base")
    ax = {"answer_open": answer_open, "answer_close": answer_close}

    intro = render_template(_strip(base.get("intro")), current_time=current_time)
    output_guard = render_template(_strip(base.get("output_guard")), **ax)

    shot_vars = dict(ax)
    if category is not None:
        shot_vars["category"] = category

    if route == _ROUTE_ACTION:
        route_cfg = _cfg("generate_action")
        rules = _strip(route_cfg.get("rules"))
        route_context = build_route_context(
            route, slots_block=slots_block, category=category
        )
        allowed_names = build_allowed_names_block(allowed_titles)
    else:
        common = _cfg("generate_no_popups_common")
        route_cfg = _cfg(_ROUTE_PROMPT_KEYS[route])
        rules = _strip(common.get("rules"))
        rules_extra = _strip(route_cfg.get("rules_extra"))
        if rules_extra:
            rules = f"{rules}\n{rules_extra}"
        route_context = build_route_context(
            route, slots_block=slots_block, category=category
        )
        allowed_names = ""

    few_shots = tuple(_few_shot_turns(route_cfg, **shot_vars))

    return RecommendPromptParts(
        intro=intro,
        output_guard=output_guard,
        rules=rules,
        allowed_names=allowed_names,
        route_context=route_context,
        popup_section=popup_section,
        few_shot_turns=few_shots,
    )


@dataclass(frozen=True)
class LeanRecommendPromptParts:
    """로컬 SLM 전용 lean 프롬프트 조각 — system 한 덩어리 + few_shot."""

    system_text: str
    few_shot_turns: tuple[dict[str, str], ...]


def assemble_lean_recommend_parts(
    route: str,
    *,
    category: str | None = None,
) -> LeanRecommendPromptParts:
    """`lean.yaml`만 사용해 로컬 SLM용 최소 프롬프트 조각을 만든다."""
    key = _LEAN_ROUTE_PROMPT_KEYS.get(route) or _LEAN_ROUTE_PROMPT_KEYS[_ROUTE_ACTION]
    cfg = _cfg(key)

    tmpl_vars: dict[str, str] = {}
    if category is not None:
        tmpl_vars["category"] = category

    system_text = render_template(_strip(cfg.get("system")), **tmpl_vars)
    few_shots = tuple(_few_shot_turns(cfg, **tmpl_vars))
    return LeanRecommendPromptParts(
        system_text=system_text,
        few_shot_turns=few_shots,
    )
