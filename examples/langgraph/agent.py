#!/usr/bin/env python3
"""LangGraph example agent for Agent Gauntlet.

This agent uses LangGraph's ReAct pattern with MCP tools to autonomously
discover and solve the challenge. Install dependencies first:

    pip install langgraph langchain-openai langchain-mcp-adapters mcp

Usage:
    cd examples/langgraph
    python agent.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from arena_clients import (
    ArenaAPIError,
    ArenaConnectionError,
    HttpArenaClient,
    McpArenaClient,
    McpArenaError,
    ensure_connected,
    get_api_base,
    get_arena_api_key,
    get_llm_api_key,
    get_mcp_url,
    get_proxy_host,
)
from base_strategy import ChallengeContext
from model_selector import fetch_available_models, select_model
from my_strategy import MyStrategy


DEFAULT_AGENT_ID = "langgraph-agent"
DEFAULT_AGENT_NAME = "LangGraph Agent"
DEFAULT_TEXT_SYSTEM_PROMPT = (
    "You are a logic puzzle solver. "
    "Return the final ordering on the FIRST line using exactly: "
    "ANSWER: Name1, Name2, Name3, Name4, Name5. "
    "Do not output <think> tags. "
    "If you add reasoning, keep it to at most 2 short lines after the ANSWER line. "
    "Never output 'unknown'."
)
DEFAULT_TEXT_STRATEGY_NOTES = ""
DEFAULT_IMAGE_STRATEGY_NOTES = ""
DEFAULT_TEXT_TEMPERATURE = 0.0
DEFAULT_TEXT_MAX_TOKENS = 1024
DEFAULT_PREFERRED_MODEL = ""
BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)
STRATEGY = MyStrategy()


def _coerce_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value if value is not None else default))
    except (TypeError, ValueError):
        return default


def _build_proxy_headers(agent_id: str, usage_scope: str | None = None) -> dict[str, str]:
    headers = {"X-Agent-ID": agent_id}
    scope_value = str(usage_scope or "").strip()
    if scope_value:
        headers["X-Round-ID"] = scope_value
    return headers


def _fetch_proxy_usage(
    llm_host: str,
    api_key: str,
    agent_id: str,
    scope_id: str | None = None,
) -> dict[str, int] | None:
    """Best-effort fetch of token usage for this agent, scoped when available."""
    if not llm_host or not api_key or not agent_id:
        return None
    encoded_agent_id = quote(agent_id, safe="")

    def _read_usage(url: str) -> dict[str, int] | None:
        request = Request(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return {
            "prompt_tokens": _coerce_nonnegative_int(usage.get("prompt_tokens")),
            "completion_tokens": _coerce_nonnegative_int(usage.get("completion_tokens")),
            "total_tokens": _coerce_nonnegative_int(usage.get("total_tokens")),
        }

    if scope_id:
        encoded_scope_id = quote(scope_id, safe="")
        scoped_url = f"{llm_host.rstrip('/')}/usage/{encoded_scope_id}/{encoded_agent_id}"
        try:
            return _read_usage(scoped_url)
        except HTTPError as exc:
            if exc.code not in {400, 404}:
                raise

    aggregate_url = f"{llm_host.rstrip('/')}/usage/{encoded_agent_id}"
    return _read_usage(aggregate_url)


def _build_context(
    *,
    challenge_type: str,
    description: str = "",
    rules: str = "",
    clues: list[str] | None = None,
    max_time_s: int = 0,
    available_models: list[str] | None = None,
    time_remaining_s: float = 0.0,
    image_url: str | None = None,
) -> ChallengeContext:
    return ChallengeContext(
        challenge_type=challenge_type or "text",
        difficulty="unknown",
        challenge_text=description,
        description=description,
        rules=rules,
        clues=list(clues or []),
        time_remaining_s=float(time_remaining_s or 0.0),
        max_time_s=int(max_time_s or 0),
        available_models=list(available_models or []),
        tools_used=[],
        tokens_used=0,
        required_tools=[],
        image_url=image_url,
    )


def _resolve_preferred_model(available_models: list[str]) -> str | None:
    preferred = (
        os.getenv("PREFERRED_MODEL")
        or str(getattr(STRATEGY, "preferred_model", "")).strip()
        or DEFAULT_PREFERRED_MODEL
    ).strip()
    if not preferred:
        return None
    if available_models and preferred not in available_models:
        print(
            f"   Preferred model '{preferred}' not in proxy model list; "
            "falling back to autonomous selection.",
        )
        return None
    return preferred


async def _wait_for_start_gate(http_client: HttpArenaClient, agent_id: str) -> None:
    """Wait for organizer start and respect competition eligibility gate."""
    await asyncio.to_thread(http_client.update_status, agent_id, "ready")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        "✅ Connected to Agent Gauntlet",
    )
    print("   Connected to Agent Gauntlet")

    last_phase: str | None = None
    last_countdown: object | None = None
    waiting_for_next_round = False

    while True:
        try:
            competition = await asyncio.to_thread(http_client.get_competition)
        except Exception:
            print("   Competition endpoint unavailable; proceeding without start gate.")
            return

        phase = str(competition.get("phase") or "").lower()
        countdown_value = competition.get("countdown_value")
        eligible_agent_ids = competition.get("eligible_agent_ids")

        eligible_for_current_round = True
        if isinstance(eligible_agent_ids, list):
            eligible_set = {
                str(value)
                for value in eligible_agent_ids
                if isinstance(value, str) and value.strip()
            }
            eligible_for_current_round = bool(eligible_set and agent_id in eligible_set)

        if phase == "running":
            if not eligible_for_current_round:
                if not waiting_for_next_round:
                    print("   Battle already running. Waiting for next organizer start...")
                    await asyncio.to_thread(
                        http_client.broadcast_thought,
                        agent_id,
                        "⏸️ Battle already running. Waiting for next round.",
                    )
                    waiting_for_next_round = True
                await asyncio.sleep(1.0)
                continue
            print("🏁 GO - challenge unlocked")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "🏁 GO - challenge unlocked",
            )
            return

        waiting_for_next_round = False

        if phase == "countdown":
            if countdown_value != last_countdown:
                print(f"⏳ Countdown: {countdown_value}")
                await asyncio.to_thread(
                    http_client.broadcast_thought,
                    agent_id,
                    f"⏳ Countdown: {countdown_value}",
                )
                last_countdown = countdown_value
        elif phase != last_phase:
            print("   Waiting for organizer start...")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "⏸️ Waiting for organizer start",
            )
            last_phase = phase

        await asyncio.sleep(1.0)


def _check_dependencies():
    """Verify all required packages are installed."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_mcp_adapters.client import MultiServerMCPClient
        return True
    except ImportError as exc:
        print("Missing LangGraph dependencies. Install with:")
        print("  pip install langgraph langchain-openai langchain-mcp-adapters mcp")
        print(f"Error: {exc}")
        return False


def extract_answer(raw_response: str) -> str:
    """Extract the strict ANSWER line from the LLM response."""
    # Strip closed and unclosed <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    for line in cleaned.splitlines():
        match = re.search(r"ANSWER:\s*(.+)", line, flags=re.IGNORECASE)
        if not match:
            continue
        answer = match.group(1).strip().strip("`\"' .")
        lowered = answer.lower()
        if (
            answer
            and "<final answer>" not in lowered
            and "return nothing except" not in lowered
            and "follow challenge rules exactly" not in lowered
        ):
            return answer
    return ""


def _extract_ordered_answer_from_rules(rules: str) -> str:
    if not isinstance(rules, str) or not rules.strip():
        return ""
    for quoted in re.findall(r"'([^']+)'", rules):
        candidate = quoted.strip()
        parts = [part.strip() for part in candidate.split(",") if part.strip()]
        if len(parts) >= 2 and all(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9 .&/_-]{0,40}", part) for part in parts
        ):
            return ", ".join(parts)
    return ""


def _is_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "503",
            "service unavailable",
            "serviceunavailable",
            "midstreamfallbackerror",
            "apiconnectionerror",
            "timeout",
            "temporarily unavailable",
        )
    )


def _extract_image_uri_from_tool_result(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    image_uri = result.get("image_uri")
    if isinstance(image_uri, str):
        return image_uri.strip()
    return ""


def _message_field(message: object, field: str) -> object:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _message_payload_to_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            text = _message_payload_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(payload, dict):
        for key in ("text", "content"):
            value = payload.get(key)
            text = _message_payload_to_text(value)
            if text:
                return text
    return ""


def _message_kind(message: object) -> str:
    for field in ("type", "role"):
        value = _message_field(message, field)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    class_name = message.__class__.__name__.lower()
    if "aimessage" in class_name:
        return "ai"
    if "humanmessage" in class_name:
        return "human"
    if "systemmessage" in class_name:
        return "system"
    if "toolmessage" in class_name:
        return "tool"
    return ""


def _extract_latest_message_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if _message_kind(message) not in {"ai", "assistant"}:
            continue
        for field in ("content", "artifact", "additional_kwargs"):
            text = _message_payload_to_text(_message_field(message, field))
            if text:
                return text
    return ""


def _extract_image_output_from_payload(payload: object, *, _depth: int = 0) -> tuple[str, str]:
    if _depth > 6:
        return "", ""
    if isinstance(payload, dict):
        maybe_uri = payload.get("image_uri")
        maybe_model = payload.get("model")
        if isinstance(maybe_uri, str) and maybe_uri.strip():
            model_name = maybe_model.strip() if isinstance(maybe_model, str) else ""
            return maybe_uri.strip(), model_name
        for value in payload.values():
            uri, model_name = _extract_image_output_from_payload(value, _depth=_depth + 1)
            if uri:
                if not model_name and isinstance(maybe_model, str):
                    model_name = maybe_model.strip()
                return uri, model_name
        return "", ""
    if isinstance(payload, list):
        for value in payload:
            uri, model_name = _extract_image_output_from_payload(value, _depth=_depth + 1)
            if uri:
                return uri, model_name
        return "", ""
    if isinstance(payload, str):
        text = payload.strip()
        lowered = text.lower()
        if lowered.startswith("data:image/") and "base64," in lowered:
            return text, ""
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                return "", ""
            return _extract_image_output_from_payload(parsed, _depth=_depth + 1)
    return "", ""


def _extract_react_image_output(messages: object) -> tuple[str, str, bool]:
    if not isinstance(messages, list):
        return "", "", False
    image_tool_names = {"image_edit", "image_generate", "image_analyze"}
    used_image_tool = False
    for message in messages:
        name = _message_field(message, "name")
        if isinstance(name, str) and name in image_tool_names:
            used_image_tool = True
        tool_calls = _message_field(message, "tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if isinstance(call_name, str) and call_name in image_tool_names:
                    used_image_tool = True
    for message in reversed(messages):
        for field in ("content", "artifact", "additional_kwargs"):
            payload = _message_field(message, field)
            if payload is None:
                continue
            image_uri, model_name = _extract_image_output_from_payload(payload)
            if image_uri:
                return image_uri, model_name, used_image_tool
    return "", "", used_image_tool


def _derive_react_timeout_s(max_time_s: int, modality: str) -> float:
    """Leave budget for submission/fallback while bounding ReAct runtime."""
    override_s = _coerce_float(os.getenv("REACT_TIMEOUT_S"), 0.0)
    if override_s > 0.0:
        return override_s
    max_time_s = int(max_time_s or 0)
    if max_time_s > 0:
        reserve_s = max(8.0, min(30.0, float(max_time_s) * 0.25))
        return max(20.0, float(max_time_s) - reserve_s)
    return 80.0 if modality == "image" else 90.0


async def main() -> int:
    if not _check_dependencies():
        return 1

    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_mcp_adapters.client import MultiServerMCPClient

    agent_id = (
        os.getenv("AGENT_ID")
        or str(getattr(STRATEGY, "agent_id", "")).strip()
        or DEFAULT_AGENT_ID
    )
    agent_name = (
        os.getenv("AGENT_NAME")
        or str(getattr(STRATEGY, "agent_name", "")).strip()
        or DEFAULT_AGENT_NAME
    )
    os.environ["AGENT_ID"] = str(agent_id).strip() or DEFAULT_AGENT_ID
    os.environ["AGENT_NAME"] = str(agent_name).strip() or DEFAULT_AGENT_NAME

    ensure_connected()

    api_base = get_api_base()
    mcp_url = get_mcp_url()
    llm_host = get_proxy_host()
    api_key = get_arena_api_key()
    llm_api_key = get_llm_api_key()
    default_ctx = _build_context(challenge_type="text")
    default_llm_params = STRATEGY.get_llm_params(default_ctx)
    text_strategy_notes = str(getattr(STRATEGY, "text_strategy_notes", "") or "").strip()
    image_strategy_notes = str(getattr(STRATEGY, "image_strategy_notes", "") or "").strip()
    text_temperature = _coerce_float(
        os.getenv("TEXT_TEMPERATURE"),
        _coerce_float(
            str(default_llm_params.get("temperature", DEFAULT_TEXT_TEMPERATURE)),
            DEFAULT_TEXT_TEMPERATURE,
        ),
    )
    text_max_tokens = _coerce_positive_int(
        os.getenv("TEXT_MAX_TOKENS"),
        _coerce_positive_int(
            str(default_llm_params.get("max_tokens", DEFAULT_TEXT_MAX_TOKENS)),
            DEFAULT_TEXT_MAX_TOKENS,
        ),
    )

    print(f"🤖 {agent_name} starting...")
    print(f"   Agent ID: {agent_id}")
    print(f"   API: {api_base}")
    print(f"   MCP: {mcp_url}")
    print(f"   LLM: {llm_host}")
    print()

    # Gauntlet REST client
    http_client = HttpArenaClient(api_base=api_base, api_key=api_key, timeout=90.0)

    print("📝 Registering with Agent Gauntlet...")
    session = http_client.register(agent_id, agent_name)
    print(f"   Session: {session.session_id}")
    print(f"   Status: {session.status}")
    print()
    await _wait_for_start_gate(http_client, agent_id)
    usage_scope = http_client.fetch_usage_scope() or (os.getenv("ARENA_USAGE_SCOPE") or "").strip() or None
    if usage_scope:
        os.environ["ARENA_USAGE_SCOPE"] = usage_scope
    print()

    # Fetch challenge first so model selection is challenge-aware.
    print("🎯 Getting challenge for model selection...")
    async with McpArenaClient(mcp_url) as arena_mcp:
        discovered_tool_names = await arena_mcp.list_tools()
        modality = McpArenaClient.detect_modality(discovered_tool_names)
        while True:
            try:
                if modality == "image":
                    challenge = await arena_mcp.get_image_challenge(agent_id)
                else:
                    challenge = await arena_mcp.get_challenge(agent_id)
                break
            except McpArenaError as exc:
                message = str(exc).lower()
                if "locked" in message or "waiting for organizer start" in message:
                    print("   Waiting for organizer start...")
                    await asyncio.sleep(1.0)
                    continue
                raise

    challenge_rules_for_selection = (
        challenge.rules
        if modality == "text"
        else "\n".join(
            part
            for part in (challenge.prompt, challenge.reference_notes)
            if isinstance(part, str) and part.strip()
        )
    )
    available_models = fetch_available_models(llm_host, llm_api_key)
    selection_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description,
        rules=challenge_rules_for_selection,
        max_time_s=challenge.max_time_s,
        available_models=available_models,
        image_url=getattr(challenge, "input_image_uri", None),
    )
    ranked_models = STRATEGY.rank_models(selection_ctx, available_models)
    preferred_model = _resolve_preferred_model(ranked_models)
    if preferred_model:
        model_name = preferred_model
    else:
        model_name = STRATEGY.pick_model("solve", ranked_models, selection_ctx)
        if model_name not in available_models:
            model_name = select_model(
                challenge_type=challenge.challenge_type,
                challenge_description=challenge.description,
                challenge_rules=challenge_rules_for_selection,
                max_time_s=challenge.max_time_s,
                available_models=available_models,
                proxy_host=llm_host,
                api_key=llm_api_key,
            )
    text_system_prompt = (
        STRATEGY.build_system_prompt(selection_ctx).strip() or DEFAULT_TEXT_SYSTEM_PROMPT
    )
    llm_params = STRATEGY.get_llm_params(selection_ctx)
    text_temperature = _coerce_float(
        os.getenv("TEXT_TEMPERATURE"),
        _coerce_float(str(llm_params.get("temperature", text_temperature)), text_temperature),
    )
    text_max_tokens = _coerce_positive_int(
        os.getenv("TEXT_MAX_TOKENS"),
        _coerce_positive_int(str(llm_params.get("max_tokens", text_max_tokens)), text_max_tokens),
    )
    print(f"   Selected model: {model_name}")
    try:
        http_client.broadcast_thought(agent_id, f"🤖 Selected model: {model_name}")
    except Exception:
        pass
    print()

    # Connect to MCP server and get tools
    print("🔧 Connecting to MCP server...")
    mcp_sse_url = f"{mcp_url}/sse"
    if api_key:
        mcp_sse_url = f"{mcp_sse_url}?api_key={quote(api_key, safe='')}"
    mcp_client = MultiServerMCPClient(
        {
            "arena": {
                "url": mcp_sse_url,
                "transport": "sse",
            }
        }
    )
    tools = await mcp_client.get_tools()
    tool_names = [t.name for t in tools]
    print(f"   Tools discovered: {tool_names}")
    print()

    # Solve the puzzle
    print("🧠 Solving with ReAct agent...")
    http_client.broadcast_thought(agent_id, "Starting ReAct agent with MCP tools...")

    start_ms = time.time() * 1000

    if modality == "image":
        image_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description or "",
            rules=challenge_rules_for_selection,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
            image_url=getattr(challenge, "input_image_uri", None),
        )
        preferred_image_tool = STRATEGY.plan_image_tool(
            image_ctx,
            [name for name in ("image_edit", "image_generate", "image_analyze") if name in tool_names],
        )
        strategy_image_prompt = STRATEGY.build_image_prompt(image_ctx)
        image_strategy_block = ""
        if image_strategy_notes:
            image_strategy_block = (
                "Competitor strategy notes:\n"
                f"{image_strategy_notes}\n\n"
            )
        prompt = (
            f"{image_strategy_block}"
            f"You are competing in a timed image challenge. "
            f"Use only MCP tools to complete the task.\n\n"
            f"Strategy preferred tool: {preferred_image_tool or 'auto'}\n"
            f"Strategy image prompt hint: {strategy_image_prompt}\n\n"
            f"Required sequence:\n"
            f"1) Call arena.image.get_challenge with agent_id='{agent_id}'.\n"
            f"2) Decide which capability tools to use from: image_edit, image_generate, image_analyze.\n"
            f"3) Produce an edited/generated image candidate.\n"
            f"4) Keep image requests at standard resolution only; do not ask for HD, 4K, or upscaling.\n"
            f"5) Return one short summary sentence.\n\n"
            f"Do NOT call arena.image.submit_edit directly in this ReAct phase. "
            f"The runtime will perform a guaranteed final submit step after ReAct completes.\n"
        )
    else:
        text_strategy_block = "Competitor strategy guidance:\n"
        text_strategy_block += f"{text_system_prompt}\n\n"
        if text_strategy_notes:
            text_strategy_block += (
                "Additional strategy notes:\n"
                f"{text_strategy_notes}\n\n"
            )
        clue_preview = "\n".join(
            f"- {clue}" for clue in (challenge.clues or []) if isinstance(clue, str) and clue.strip()
        )
        if not clue_preview:
            clue_preview = "- (No clues provided.)"
        challenge_type = (challenge.challenge_type or "").lower()
        rules_lower = (challenge.rules or "").lower()
        required_tool_hint = ""
        if challenge_type in {"web-search", "market-research"} or "firecrawl_search" in rules_lower:
            required_tool_hint = (
                "- You MUST call firecrawl_search at least once before your final answer.\n"
            )
        prompt = (
            f"{text_strategy_block}"
            f"You are competing in a timed challenge.\n\n"
            f"Challenge snapshot:\n"
            f"- Type: {challenge.challenge_type}\n"
            f"- Description: {challenge.description}\n"
            f"- Rules: {challenge.rules}\n"
            f"- Clues:\n{clue_preview}\n\n"
            f"Execution requirements:\n"
            f"{required_tool_hint}"
            f"- Use available tools as needed before finalizing.\n"
            f"- Keep reasoning EXTREMELY short (3 sentences max)\n"
            f"- After solving, output the answer on its own line as: ANSWER: <your answer>\n"
            f"- If Rules require strict one-line output, follow it exactly.\n"
            f"- The ANSWER line is MANDATORY. Without it, you score zero.\n"
        )

    default_recursion_limit = 8 if modality == "image" else 12
    react_recursion_limit = _coerce_positive_int(
        os.getenv("REACT_RECURSION_LIMIT"),
        default_recursion_limit,
    )
    if modality == "image":
        react_recursion_limit = _coerce_positive_int(
            os.getenv("IMAGE_REACT_RECURSION_LIMIT"),
            react_recursion_limit,
        )
    else:
        react_recursion_limit = _coerce_positive_int(
            os.getenv("TEXT_REACT_RECURSION_LIMIT"),
            react_recursion_limit,
        )
    react_timeout_s = _derive_react_timeout_s(int(challenge.max_time_s or 0), modality)
    print(
        f"   ReAct limits: recursion={react_recursion_limit}, timeout={react_timeout_s:.1f}s"
    )

    candidate_models: list[str] = []
    for candidate in [model_name, *ranked_models, *available_models]:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in candidate_models:
            candidate_models.append(normalized)
    max_attempts = min(
        len(candidate_models),
        _coerce_positive_int(os.getenv("SOLVE_MAX_ATTEMPTS"), 3),
    )
    candidate_models = candidate_models[:max_attempts]

    live_prompt_tokens = 0
    live_completion_tokens = 0
    live_total_tokens = 0
    active_model_name = model_name
    reporter_stop = asyncio.Event()

    async def _refresh_live_usage() -> None:
        nonlocal live_prompt_tokens, live_completion_tokens, live_total_tokens
        try:
            usage = await asyncio.to_thread(
                _fetch_proxy_usage,
                llm_host,
                llm_api_key,
                agent_id,
                usage_scope,
            )
        except Exception:
            usage = None
        if not isinstance(usage, dict):
            return
        live_prompt_tokens = max(0, _coerce_nonnegative_int(usage.get("prompt_tokens")))
        live_completion_tokens = max(
            0, _coerce_nonnegative_int(usage.get("completion_tokens"))
        )
        live_total_tokens = max(0, _coerce_nonnegative_int(usage.get("total_tokens")))

    def _build_live_metrics(elapsed_ms: int) -> dict[str, object]:
        return {
            "model_name": active_model_name or model_name,
            "total_tokens": str(live_total_tokens),
            "prompt_tokens": str(live_prompt_tokens),
            "completion_tokens": str(live_completion_tokens),
            "total_time_ms": int(max(0, elapsed_ms)),
        }

    async def _push_live_status(status: str = "running") -> None:
        await _refresh_live_usage()
        elapsed_ms = int(time.time() * 1000 - start_ms)
        try:
            await asyncio.to_thread(
                http_client.update_status,
                agent_id,
                status,
                _build_live_metrics(elapsed_ms),
            )
        except Exception:
            return

    async def _runtime_metrics_reporter() -> None:
        while not reporter_stop.is_set():
            await _push_live_status("running")
            try:
                await asyncio.wait_for(reporter_stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    await _push_live_status("running")
    metrics_task = asyncio.create_task(_runtime_metrics_reporter())

    result = None
    solve_error: Exception | None = None
    for attempt_idx, candidate_model in enumerate(candidate_models, start=1):
        active_model_name = candidate_model
        if candidate_model != model_name:
            print(
                f"   Retrying with fallback model: {candidate_model} "
                f"(attempt {attempt_idx}/{len(candidate_models)})"
            )
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"🔁 Retrying with fallback model: {candidate_model}",
            )
        llm = ChatOpenAI(
            model=candidate_model,
            base_url=llm_host,
            api_key=llm_api_key,
            temperature=text_temperature,
            max_tokens=text_max_tokens,
            default_headers=_build_proxy_headers(agent_id, usage_scope),
        )
        agent = create_react_agent(llm, tools)
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"recursion_limit": react_recursion_limit},
                ),
                timeout=react_timeout_s,
            )
            model_name = candidate_model
            solve_error = None
            break
        except asyncio.TimeoutError as exc:
            solve_error = exc
            print(
                f"   solve attempt {attempt_idx} timed out after {react_timeout_s:.1f}s"
            )
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                (
                    f"⏱️ ReAct solve attempt timed out after {react_timeout_s:.0f}s "
                    f"(attempt {attempt_idx}/{len(candidate_models)})."
                ),
            )
            if attempt_idx >= len(candidate_models):
                break
            await asyncio.sleep(float(attempt_idx))
        except Exception as exc:
            solve_error = exc
            print(f"   solve attempt {attempt_idx} failed: {exc}")
            retryable = _is_retryable_llm_error(exc)
            if attempt_idx >= len(candidate_models) or not retryable:
                break
            await asyncio.sleep(float(attempt_idx))

    reporter_stop.set()
    try:
        await asyncio.wait_for(metrics_task, timeout=2.0)
    except Exception:
        if not metrics_task.done():
            metrics_task.cancel()
    await _push_live_status("running")

    total_time_ms = int(time.time() * 1000 - start_ms)

    raw_content = ""
    react_image_uri = ""
    react_image_model_name = ""
    react_used_image_tool = False
    if result is not None:
        raw_content = _extract_latest_message_text(result.get("messages"))
        if modality == "image":
            react_image_uri, react_image_model_name, react_used_image_tool = _extract_react_image_output(
                result.get("messages")
            )
    elif solve_error:
        await asyncio.to_thread(
            http_client.broadcast_thought,
            agent_id,
            "⚠️ LLM unavailable after retries. Falling back to rules-safe answer path.",
        )
    if modality == "image":
        if react_image_uri:
            print("   ReAct produced an image candidate; reusing it for final submit.")
        elif react_used_image_tool:
            print("   ReAct used image tools but no image URI found; using guaranteed submit path.")
    print(f"   Raw output: {raw_content[:120]}...")
    print(f"   Total time: {total_time_ms}ms")
    print()

    if modality == "image":
        print("📤 Finalizing image submission...")
        submission_confirmed = False
        try:
            session_state = http_client.get_session(agent_id)
            submission_confirmed = str(session_state.get("status") or "").lower() == "submitted"
        except Exception:
            submission_confirmed = False

        if not submission_confirmed and react_image_uri:
            print("   Submitting ReAct image candidate.")
            try:
                async with McpArenaClient(mcp_url) as submit_mcp:
                    effective_image_model_name = react_image_model_name or model_name
                    submit_result = await submit_mcp.submit_image(
                        agent_id=agent_id,
                        image_uri=react_image_uri,
                        client_metrics={
                            "model_name": effective_image_model_name,
                            "planner_model_name": model_name,
                            "total_tokens": str(live_total_tokens),
                            "prompt_tokens": str(live_prompt_tokens),
                            "completion_tokens": str(live_completion_tokens),
                            "ttft_ms": 0,
                            "total_time_ms": total_time_ms,
                        },
                        rationale="Submitting image produced during ReAct phase.",
                    )
                    submission_confirmed = bool(submit_result)
                    if isinstance(submit_result, dict):
                        submit_log = dict(submit_result)
                        submit_log.pop("edited_image", None)
                        submit_log.pop("image_uri", None)
                        print(f"   ReAct image submit result: {submit_log}")
                    else:
                        print(f"   ReAct image submit result: {submit_result}")
            except Exception as exc:
                print(f"   ReAct image submission failed: {exc}")

        if not submission_confirmed:
            print("   Running guaranteed submit path.")
            http_client.broadcast_thought(
                agent_id,
                "🛠️ Finalizing image submission in guaranteed submit path.",
            )
            try:
                async with McpArenaClient(mcp_url) as fallback_mcp:
                    image_challenge = await fallback_mcp.get_image_challenge(agent_id)
                    fallback_ctx = _build_context(
                        challenge_type=image_challenge.challenge_type,
                        description=image_challenge.description or "",
                        rules="\n".join(
                            part
                            for part in (
                                image_challenge.prompt,
                                image_challenge.reference_notes,
                            )
                            if isinstance(part, str) and part.strip()
                        ),
                        max_time_s=image_challenge.max_time_s,
                        available_models=ranked_models,
                        image_url=image_challenge.input_image_uri or None,
                    )
                    fallback_prompt = (
                        STRATEGY.build_image_prompt(fallback_ctx).strip()
                        or image_challenge.prompt
                        or image_challenge.description
                        or "Create an edited image that satisfies the challenge."
                    ).strip()
                    planned_tool = STRATEGY.plan_image_tool(
                        fallback_ctx,
                        [
                            name
                            for name in ("image_edit", "image_generate", "image_analyze")
                            if name in tool_names
                        ],
                    )
                    image_uri = ""
                    image_tool_model_name = ""
                    if (
                        planned_tool == "image_edit"
                        and image_challenge.input_image_uri
                        and "image_edit" in tool_names
                    ):
                        edit_result = await fallback_mcp.call_tool(
                            "image_edit",
                            {
                                "image_uri": image_challenge.input_image_uri,
                                "prompt": fallback_prompt,
                                "agent_id": agent_id,
                            },
                        )
                        image_uri = _extract_image_uri_from_tool_result(edit_result)
                        if isinstance(edit_result, dict):
                            maybe_model_name = edit_result.get("model")
                            if isinstance(maybe_model_name, str):
                                image_tool_model_name = maybe_model_name.strip()
                    if not image_uri and planned_tool == "image_generate" and "image_generate" in tool_names:
                        generate_result = await fallback_mcp.call_tool(
                            "image_generate",
                            {"prompt": fallback_prompt, "agent_id": agent_id},
                        )
                        image_uri = _extract_image_uri_from_tool_result(generate_result)
                        if isinstance(generate_result, dict):
                            maybe_model_name = generate_result.get("model")
                            if isinstance(maybe_model_name, str):
                                image_tool_model_name = maybe_model_name.strip()
                    if not image_uri and image_challenge.input_image_uri and "image_edit" in tool_names:
                        edit_result = await fallback_mcp.call_tool(
                            "image_edit",
                            {
                                "image_uri": image_challenge.input_image_uri,
                                "prompt": fallback_prompt,
                                "agent_id": agent_id,
                            },
                        )
                        image_uri = _extract_image_uri_from_tool_result(edit_result)
                        if isinstance(edit_result, dict):
                            maybe_model_name = edit_result.get("model")
                            if isinstance(maybe_model_name, str):
                                image_tool_model_name = maybe_model_name.strip()
                    if not image_uri and "image_generate" in tool_names:
                        generate_result = await fallback_mcp.call_tool(
                            "image_generate",
                            {"prompt": fallback_prompt, "agent_id": agent_id},
                        )
                        image_uri = _extract_image_uri_from_tool_result(generate_result)
                        if isinstance(generate_result, dict):
                            maybe_model_name = generate_result.get("model")
                            if isinstance(maybe_model_name, str):
                                image_tool_model_name = maybe_model_name.strip()
                    if not image_uri and image_challenge.input_image_uri:
                        image_uri = image_challenge.input_image_uri
                    if not image_uri:
                        image_uri = BLANK_PNG_DATA_URI
                    effective_image_model_name = image_tool_model_name or model_name

                    submit_result = await fallback_mcp.submit_image(
                        agent_id=agent_id,
                        image_uri=image_uri,
                        client_metrics={
                            "model_name": effective_image_model_name,
                            "planner_model_name": model_name,
                            "total_tokens": str(live_total_tokens),
                            "prompt_tokens": str(live_prompt_tokens),
                            "completion_tokens": str(live_completion_tokens),
                            "ttft_ms": 0,
                            "total_time_ms": total_time_ms,
                        },
                        rationale="Guaranteed submit path after ReAct run.",
                    )
                    submission_confirmed = bool(submit_result)
                    if isinstance(submit_result, dict):
                        submit_log = dict(submit_result)
                        submit_log.pop("edited_image", None)
                        submit_log.pop("image_uri", None)
                        print(f"   Guaranteed submit result: {submit_log}")
                    else:
                        print(f"   Guaranteed submit result: {submit_result}")
            except Exception as exc:
                print(f"   Guaranteed image submission failed: {exc}")

        if submission_confirmed:
            http_client.broadcast_thought(agent_id, "✅ Image submission completed.")
        else:
            http_client.broadcast_thought(
                agent_id,
                "⚠️ Image flow ended without confirmed submission.",
            )
            try:
                http_client.update_status(agent_id, "failed")
            except Exception:
                pass
        print("\n✅ Agent completed!")
        return 0

    # Extract the answer from the latest non-empty text returned by LangGraph.
    answer = extract_answer(raw_content)

    if not answer and raw_content:
        # First retry: re-prompt with strict instructions if initial extraction fails
        print("   Extraction failed; retrying with strict formatter...")
        strict_system_msg = "You are a strict answer extractor. Extract the final answer from the provided reasoning and return it in exactly one line: ANSWER: <final answer>. No preamble, no tags, no explanation, no thinking blocks, no additional lines. If the reasoning is incomplete, make your best guess for the ordering of events and return it in the ANSWER format. DO NOT USE <think> TAGS. OUTPUT ONLY THE ANSWER LINE."
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=llm_host,
                api_key=llm_api_key,
                default_headers=_build_proxy_headers(agent_id, usage_scope),
            )
            strict_response = client.chat.completions.create(
                model="nemotron-nano-9b",
                messages=[
                    {"role": "system", "content": strict_system_msg},
                    {"role": "user", "content": f"Reasoning:\n\n{raw_content}\n\nExtract the final answer now."},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw_retry = strict_response.choices[0].message.content or ""
            print(f"   Retry output: {raw_retry.strip()}")
            answer = extract_answer(raw_retry)
        except Exception as e:
            print(f"   Retry failed: {e}")
            pass

    if not answer and solve_error:
        answer = _extract_ordered_answer_from_rules(challenge.rules or "")
        if answer:
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"✅ Using fallback answer from challenge rules: {answer}",
            )
        else:
            answer = "unknown"
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "⚠️ No fallback answer pattern found; submitting 'unknown'.",
            )
    try:
        async with McpArenaClient(mcp_url) as timing_mcp:
            time_info = await timing_mcp.time_remaining(agent_id)
        remaining_s = float(time_info.get("time_remaining_s", 0.0))
    except Exception:
        remaining_s = 0.0
    submit_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description or "",
        rules=challenge.rules or "",
        max_time_s=challenge.max_time_s,
        available_models=ranked_models,
        time_remaining_s=remaining_s,
    )
    if STRATEGY.should_submit_early(answer, submit_ctx):
        http_client.broadcast_thought(agent_id, "⚡ Strategy submitting early.")
    revised_answer = STRATEGY.on_time_warning(remaining_s, answer, submit_ctx)
    if isinstance(revised_answer, str) and revised_answer.strip():
        answer = revised_answer.strip()

    # Submit
    if not answer:
        print("⚠️ No valid answer found; skipping submission.")
        return 1

    print(f"📤 Submitting answer: {answer}")
    try:
        http_client.broadcast_thought(agent_id, f"Answer: {answer}")
    except Exception as exc:
        print(f"   broadcast skipped ({exc})")

    result = None
    for attempt in range(1, 4):
        try:
            result = http_client.submit(
                agent_id=agent_id,
                answer=answer,
                client_metrics={
                    "model_name": model_name,
                    "total_tokens": str(live_total_tokens),
                    "prompt_tokens": str(live_prompt_tokens),
                    "completion_tokens": str(live_completion_tokens),
                    "total_time_ms": total_time_ms,
                },
            )
            break
        except (TimeoutError, ArenaConnectionError) as exc:
            print(f"   submit attempt {attempt} failed ({exc}); retrying...")
            if attempt < 3:
                await asyncio.sleep(float(attempt))
            else:
                print("   all submit attempts failed")
        except ArenaAPIError as exc:
            print(f"   submit API error: {exc}")
            break

    if result:
        print(f"   Accepted: {result.accepted}")
        print(f"   Status: {result.status}")
        if result.score:
            print(f"   Score: {result.score.get('final_score', 0)}")
            print(f"   Quality: {result.score.get('quality_score', 0)}")
            print(f"   Speed: {result.score.get('speed_score', 0)}")
    else:
        print("   ⚠️ Submit failed after retries")

    print("\n✅ Agent completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
