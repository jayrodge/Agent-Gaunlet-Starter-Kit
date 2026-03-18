#!/usr/bin/env python3
"""CrewAI example agent for Agent Gauntlet.

This example uses CrewAI native tools backed by the starter-kit MCP client
instead of CrewAI's direct MCP transport layer. That keeps the example aligned
with current CrewAI function-calling rules while preserving runtime tool
discovery and both text and image challenge flows.

Install dependencies first:

    pip install 'crewai[tools]' mcp

Usage:
    cd examples/crewai
    python agent.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from arena_clients import (
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
from model_selector import fetch_available_models
from my_strategy import MyStrategy
from arena_tools import (
    ArenaToolState,
    ToolSpec,
    build_crewai_tools,
    classify_image_tool,
    discover_tool_specs,
    unsupported_required_fields,
)

BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)
DEFAULT_AGENT_ID = "crewai-agent"
DEFAULT_AGENT_NAME = "CrewAI Agent"
STRATEGY = MyStrategy()


def _check_dependencies() -> bool:
    try:
        from crewai import Agent, Crew, LLM, Task
        from crewai.tools import BaseTool

        _ = (Agent, Crew, LLM, Task, BaseTool)
        return True
    except ImportError as exc:
        print("Missing CrewAI dependencies. Install with:")
        print("  pip install 'crewai[tools]' mcp")
        print(f"Error: {exc}")
        return False


def extract_answer(raw_response: str) -> str:
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_response, flags=re.DOTALL).strip()
    for line in cleaned.splitlines():
        match = re.match(
            r"^\s*(?:answer|final answer)\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            answer = match.group(1).strip().strip("`\"'")
            if answer:
                return answer
    return cleaned.splitlines()[0].strip().strip("`\"'") if cleaned else ""


def extract_image_uri(raw_response: str) -> str:
    patterns = (
        r"IMAGE_URI:\s*(\S+)",
        r'"image_uri"\s*:\s*"([^"]+)"',
        r"'image_uri'\s*:\s*'([^']+)'",
        r"(data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip().rstrip(".,)")
        if candidate.lower() in {"stored_by_runtime", "stored_by_tool", "runtime"}:
            return ""
        return candidate
    return ""


def extract_image_plan(raw_response: str) -> tuple[str, str, str]:
    tool_name = ""
    instruction_text = ""
    summary = ""
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("TOOL:"):
            tool_name = stripped.split(":", 1)[1].strip().lower()
        elif stripped.upper().startswith("INSTRUCTION:") or stripped.upper().startswith("PROMPT:"):
            instruction_text = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("SUMMARY:"):
            summary = stripped.split(":", 1)[1].strip()
    return tool_name, instruction_text, summary


def _extract_image_uri_from_tool_result(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    for field_name in ("image_uri", "output_image_uri", "edited_image", "data_uri"):
        image_uri = result.get(field_name)
        if isinstance(image_uri, str) and image_uri.strip():
            return image_uri.strip()
    return ""


def _build_context(
    *,
    challenge_type: str,
    description: str,
    rules: str,
    clues: list[str] | None = None,
    max_time_s: int,
    available_models: list[str],
    time_remaining_s: float = 0.0,
    image_url: str | None = None,
) -> ChallengeContext:
    return ChallengeContext(
        challenge_type=challenge_type,
        difficulty="unknown",
        challenge_text=description,
        description=description,
        rules=rules,
        clues=list(clues or []),
        time_remaining_s=time_remaining_s,
        max_time_s=max_time_s,
        available_models=available_models,
        tools_used=[],
        tokens_used=0,
        required_tools=[],
        image_url=image_url,
    )


def _require_selected_model(
    *,
    ranked_models: list[str],
    available_models: list[str],
    ctx: ChallengeContext,
) -> str:
    model_name = str(STRATEGY.pick_model("solve", ranked_models, ctx) or "").strip()
    if not model_name:
        roster_preview = ", ".join(available_models[:8]) or "(proxy roster unavailable)"
        raise RuntimeError(
            "No model selected. Update MyStrategy.pick_model() to return an exact proxy model alias. "
            f"Available models: {roster_preview}"
        )
    if available_models and model_name not in available_models:
        raise RuntimeError(
            f"Selected model '{model_name}' is not in the proxy roster: {', '.join(available_models)}"
        )
    return model_name


def _challenge_rules_text(challenge: object, modality: str) -> str:
    if modality == "text":
        rules = str(getattr(challenge, "rules", "") or "").strip()
        return rules or str(getattr(challenge, "description", "") or "").strip()
    return "\n".join(
        part
        for part in (
            getattr(challenge, "prompt", ""),
            getattr(challenge, "reference_notes", ""),
            getattr(challenge, "description", ""),
        )
        if isinstance(part, str) and part.strip()
    )


def _normalize_tool_key(value: str) -> str:
    return str(value or "").strip().lower()


def _is_runtime_control_tool(tool_name: str) -> bool:
    lower = _normalize_tool_key(tool_name)
    if not lower.startswith("arena."):
        return False
    return (
        "get_challenge" in lower
        or "broadcast_thought" in lower
        or ".submit" in lower
    )


def _order_image_tool_specs(
    image_tool_specs: list[ToolSpec],
    *,
    has_input_image: bool,
) -> list[ToolSpec]:
    kind_priority = (
        {"edit": 0, "generate": 1, "analyze": 2, "other": 3, "none": 4}
        if has_input_image
        else {"generate": 0, "edit": 1, "analyze": 2, "other": 3, "none": 4}
    )
    return sorted(
        image_tool_specs,
        key=lambda spec: (
            kind_priority.get(classify_image_tool(spec), 9),
            spec.original_name.lower(),
        ),
    )


def _build_image_tool_selection_map(
    image_tool_specs: list[ToolSpec],
) -> tuple[list[str], dict[str, ToolSpec]]:
    selection_choices: list[str] = []
    selection_map: dict[str, ToolSpec] = {}

    def register(key: str, spec: ToolSpec) -> None:
        normalized = _normalize_tool_key(key)
        if not normalized or normalized in selection_map:
            return
        selection_map[normalized] = spec
        selection_choices.append(key)

    first_by_kind: dict[str, ToolSpec] = {}
    for spec in image_tool_specs:
        kind = classify_image_tool(spec)
        if kind in {"edit", "generate", "analyze"} and kind not in first_by_kind:
            first_by_kind[kind] = spec

    for kind, alias in (
        ("edit", "image_edit"),
        ("generate", "image_generate"),
        ("analyze", "image_analyze"),
    ):
        spec = first_by_kind.get(kind)
        if spec:
            register(alias, spec)

    for spec in image_tool_specs:
        register(spec.original_name, spec)
        register(spec.sanitized_name, spec)

    return selection_choices, selection_map


def _choose_image_tool_spec(
    image_ctx: ChallengeContext,
    image_tool_specs: list[ToolSpec],
    planned_tool: str,
) -> ToolSpec | None:
    if not image_tool_specs:
        return None

    strategy_choices, selection_map = _build_image_tool_selection_map(image_tool_specs)
    planned_spec = selection_map.get(_normalize_tool_key(planned_tool))
    if planned_spec:
        return planned_spec

    strategy_choice = STRATEGY.plan_image_tool(image_ctx, strategy_choices)
    strategy_spec = selection_map.get(_normalize_tool_key(strategy_choice))
    if strategy_spec:
        return strategy_spec

    return image_tool_specs[0]


def _describe_image_tool(spec: ToolSpec) -> str:
    kind = classify_image_tool(spec)
    kind_label = {
        "edit": "edits an existing image",
        "generate": "generates an image from text",
        "analyze": "analyzes an input image",
        "other": "handles image-related data",
    }.get(kind, "image-related")
    parts = [kind_label]
    if spec.runtime_hints.image_input_field:
        parts.append(f"image input via `{spec.runtime_hints.image_input_field}`")
    if spec.instruction_field:
        parts.append(f"instruction via `{spec.instruction_field}`")
    return "; ".join(parts)

async def _wait_for_start_gate(http_client: HttpArenaClient, agent_id: str) -> None:
    await asyncio.to_thread(http_client.update_status, agent_id, "ready")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        "Connected to Agent Gauntlet",
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
            if eligible_set:
                eligible_for_current_round = agent_id in eligible_set

        if phase == "running":
            if not eligible_for_current_round:
                if not waiting_for_next_round:
                    print("   Battle already running. Waiting for next organizer start...")
                    await asyncio.to_thread(
                        http_client.broadcast_thought,
                        agent_id,
                        "Battle already running. Waiting for next round.",
                    )
                    waiting_for_next_round = True
                await asyncio.sleep(1.0)
                continue
            print("   GO - challenge unlocked")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "GO - challenge unlocked",
            )
            return

        waiting_for_next_round = False
        if phase == "countdown":
            if countdown_value != last_countdown:
                print(f"   Countdown: {countdown_value}")
                await asyncio.to_thread(
                    http_client.broadcast_thought,
                    agent_id,
                    f"Countdown: {countdown_value}",
                )
                last_countdown = countdown_value
        elif phase != last_phase:
            print("   Waiting for organizer start...")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "Waiting for organizer start",
            )
            last_phase = phase

        await asyncio.sleep(1.0)


async def _fetch_challenge(arena_mcp: McpArenaClient, modality: str, agent_id: str):
    while True:
        try:
            if modality == "image":
                return await arena_mcp.get_image_challenge(agent_id)
            return await arena_mcp.get_challenge(agent_id)
        except McpArenaError as exc:
            message = str(exc).lower()
            if "locked" in message or "waiting for organizer" in message:
                print("   Waiting for organizer start...")
                await asyncio.sleep(1.0)
                continue
            raise


def _build_text_task_description(
    challenge,
    text_ctx: ChallengeContext,
    available_capabilities: list[str],
) -> str:
    clue_preview = "\n".join(
        f"- {clue}"
        for clue in (challenge.clues or [])
        if isinstance(clue, str) and clue.strip()
    )
    if not clue_preview:
        clue_preview = "- (No clues provided.)"

    challenge_type = str(challenge.challenge_type or "").lower()
    rules_lower = str(challenge.rules or "").lower()
    required_tool_hint = ""
    if challenge_type in {"web-search", "market-research"} or "firecrawl_search" in rules_lower:
        required_tool_hint += "- Call the search tool at least once before your final answer.\n"
    if challenge_type == "youtube-transcript" or "transcript" in rules_lower:
        required_tool_hint += "- Use the transcript tool when relevant to gather the answer.\n"

    strategy_notes = str(getattr(STRATEGY, "text_strategy_notes", "") or "").strip()
    strategy_block = f"Additional strategy notes:\n{strategy_notes}\n\n" if strategy_notes else ""
    available_capabilities_text = ", ".join(available_capabilities) or "no additional tools"

    return (
        f"{strategy_block}"
        f"{STRATEGY.build_solver_prompt(text_ctx)}\n\n"
        f"Available runtime capabilities: {available_capabilities_text}\n\n"
        f"Execution requirements:\n"
        f"{required_tool_hint}"
        f"- Use the available tools when they improve answer quality.\n"
        f"- Make sure every clue is satisfied before you finalize the ordering.\n"
        f"- Keep reasoning extremely short (3 sentences max).\n"
        f"- Output the final answer on its own line exactly as: ANSWER: <your answer>\n"
        f"- If the rules demand a stricter one-line output, follow them exactly after the ANSWER label.\n"
        f"- The ANSWER line is mandatory.\n"
    )


def _build_image_task_description(
    challenge,
    image_ctx: ChallengeContext,
    planning_tool_names: list[str],
    image_tool_specs: list[ToolSpec],
) -> str:
    image_strategy_notes = str(getattr(STRATEGY, "image_strategy_notes", "") or "").strip()
    strategy_block = f"Image strategy notes:\n{image_strategy_notes}\n\n" if image_strategy_notes else ""
    strategy_choices, selection_map = _build_image_tool_selection_map(image_tool_specs)
    preferred_choice = STRATEGY.plan_image_tool(image_ctx, strategy_choices)
    preferred_tool = selection_map.get(_normalize_tool_key(preferred_choice))
    strategy_prompt = STRATEGY.build_image_prompt(image_ctx).strip() or (
        getattr(challenge, "prompt", "") or challenge.description or ""
    ).strip()
    available_capabilities = ", ".join(planning_tool_names) or "no additional planning tools"
    action_tool_lines = "\n".join(
        f"- {spec.original_name}: {_describe_image_tool(spec)}"
        for spec in image_tool_specs
    )
    if not action_tool_lines:
        action_tool_lines = "- No executable image-producing tool was inferred from the discovered schemas."
    preferred_tool_name = preferred_tool.original_name if preferred_tool else (preferred_choice or "auto")
    return (
        f"{strategy_block}"
        f"Image planning hint:\n{strategy_prompt}\n\n"
        f"Challenge snapshot:\n"
        f"- Type: {challenge.challenge_type}\n"
        f"- Description: {challenge.description}\n"
        f"- Prompt: {getattr(challenge, 'prompt', '')}\n"
        f"- Reference notes: {getattr(challenge, 'reference_notes', '')}\n"
        f"- Available planning tools: {available_capabilities}\n"
        f"- Final image-capable tools:\n{action_tool_lines}\n\n"
        f"Execution requirements:\n"
        f"- Preferred final image tool: {preferred_tool_name}\n"
        f"- Your job is to PLAN the best image action, not to submit it directly.\n"
        f"- The runtime will execute the final image tool call and submit the result after you respond.\n"
        f"- If extra planning tools are available and helpful, you may use them before deciding.\n"
        f"- Return exactly these lines:\n"
        f"TOOL: <exact tool name from the discovered final image-capable list>\n"
        f"INSTRUCTION: <text to place in that tool's main instruction field>\n"
        f"SUMMARY: <one short sentence>\n"
    )


def _extract_usage_metrics(result: object) -> dict[str, int]:
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    return {
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


async def _resolve_image_candidate(
    *,
    initial_image_uri: str,
    tool_state: ArenaToolState,
    challenge,
    image_tool_specs: list[ToolSpec],
    ranked_models: list[str],
    mcp_url: str,
    api_key: str | None,
    agent_id: str,
    planned_tool: str = "",
    planned_instruction: str = "",
) -> tuple[str, str]:
    image_uri = initial_image_uri.strip()
    if not image_uri and tool_state.latest_image_uri:
        image_uri = tool_state.latest_image_uri
    selected_tool = tool_state.last_image_tool or "framework-output"
    if image_uri:
        return image_uri, selected_tool

    ordered_image_specs = _order_image_tool_specs(
        image_tool_specs,
        has_input_image=bool(getattr(challenge, "input_image_uri", "")),
    )
    image_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description,
        rules=_challenge_rules_text(challenge, "image"),
        max_time_s=challenge.max_time_s,
        available_models=ranked_models,
        image_url=getattr(challenge, "input_image_uri", None),
    )
    preferred_spec = _choose_image_tool_spec(image_ctx, ordered_image_specs, planned_tool)
    prompt_text = planned_instruction.strip() or STRATEGY.build_image_prompt(image_ctx).strip() or (
        getattr(challenge, "prompt", "") or challenge.description or ""
    ).strip()

    candidate_specs: list[ToolSpec] = []
    if preferred_spec:
        candidate_specs.append(preferred_spec)
    for spec in ordered_image_specs:
        if preferred_spec and spec.original_name == preferred_spec.original_name:
            continue
        candidate_specs.append(spec)

    tool_result: dict[str, object] = {}
    for spec in candidate_specs:
        payload: dict[str, object] = {}
        if spec.runtime_hints.accepts_agent_id:
            payload["agent_id"] = agent_id
        if spec.runtime_hints.image_input_field:
            challenge_image_uri = str(getattr(challenge, "input_image_uri", "") or "").strip()
            fallback_image_uri = tool_state.current_image_uri()
            image_input_uri = challenge_image_uri or fallback_image_uri
            if image_input_uri:
                payload[spec.runtime_hints.image_input_field] = image_input_uri
        if spec.instruction_field and prompt_text:
            payload[spec.instruction_field] = prompt_text

        try:
            async with McpArenaClient(mcp_url, api_key) as submit_mcp:
                tool_result = await submit_mcp.call_tool(spec.original_name, payload)
        except Exception as exc:
            print(f"   Image tool '{spec.original_name}' failed: {exc}")
            continue

        tool_state.record_result(spec.original_name, tool_result)
        image_uri = tool_state.latest_image_uri or _extract_image_uri_from_tool_result(tool_result)
        selected_tool = spec.original_name
        if image_uri:
            return image_uri, selected_tool

    if not image_uri and getattr(challenge, "input_image_uri", ""):
        image_uri = str(getattr(challenge, "input_image_uri")).strip()
    if not image_uri:
        image_uri = BLANK_PNG_DATA_URI
    return image_uri, selected_tool


async def main() -> int:
    if not _check_dependencies():
        return 1

    from crewai import Agent as CrewAgent, Crew, LLM, Task

    ensure_connected()

    api_base = get_api_base()
    mcp_url = get_mcp_url()
    llm_host = get_proxy_host()
    api_key = get_arena_api_key()
    llm_api_key = get_llm_api_key()

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

    print(f"  {agent_name} starting...")
    print(f"   Agent ID: {agent_id}")
    print(f"   API: {api_base}")
    print(f"   MCP: {mcp_url}")
    print(f"   LLM: {llm_host}")
    print()

    http_client = HttpArenaClient(api_base=api_base, api_key=api_key, timeout=90.0)

    print("Registering with Agent Gauntlet...")
    session = await asyncio.to_thread(http_client.register, agent_id, agent_name)
    print(f"   Session: {session.session_id}")
    print(f"   Status: {session.status}")
    print()

    await _wait_for_start_gate(http_client, agent_id)
    print()

    print("Discovering runtime tools...")
    async with McpArenaClient(mcp_url, api_key) as arena_mcp:
        discovered_tools = await arena_mcp.list_tools()
        tool_defs = await arena_mcp.list_tool_defs()
        modality = McpArenaClient.detect_modality(discovered_tools)
        challenge = await _fetch_challenge(arena_mcp, modality, agent_id)

    all_tool_specs = discover_tool_specs(tool_defs)
    control_tool_names = {
        spec.original_name
        for spec in all_tool_specs
        if _is_runtime_control_tool(spec.original_name)
    }
    executable_image_tool_specs = [
        spec
        for spec in all_tool_specs
        if classify_image_tool(spec) in {"edit", "generate"}
        and not unsupported_required_fields(spec)
    ]
    if not executable_image_tool_specs:
        executable_image_tool_specs = [
            spec
            for spec in all_tool_specs
            if spec.image_related and not unsupported_required_fields(spec)
        ]

    challenge_rules_text = _challenge_rules_text(challenge, modality)
    print(f"   Modality: {modality}")
    print(f"   Challenge type: {challenge.challenge_type}")
    print(f"   Puzzle: {challenge.puzzle_id}")
    print(f"   Time limit: {challenge.max_time_s}s")
    print()

    available_models = fetch_available_models(llm_host, llm_api_key)
    selection_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description,
        rules=challenge_rules_text,
        clues=list(getattr(challenge, "clues", []) or []),
        max_time_s=challenge.max_time_s,
        available_models=available_models,
        time_remaining_s=float(getattr(challenge, "time_remaining_s", 0.0) or 0.0),
        image_url=getattr(challenge, "input_image_uri", None),
    )
    ranked_models = STRATEGY.rank_models(selection_ctx, available_models)
    model_name = _require_selected_model(
        ranked_models=ranked_models,
        available_models=available_models,
        ctx=selection_ctx,
    )
    print(f"   Selected model: {model_name}")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        f"Selected model: {model_name}",
    )

    llm_params = STRATEGY.get_llm_params(selection_ctx)
    llm_temperature = float(llm_params.get("temperature", 0.0) or 0.0)
    llm_max_tokens = int(llm_params.get("max_tokens", 1024) or 1024)

    if modality == "text":
        excluded_tools = set(control_tool_names)
        excluded_tools.update(
            spec.original_name
            for spec in all_tool_specs
            if spec.image_related
        )
    else:
        excluded_tools = set(control_tool_names)
        excluded_tools.update(spec.original_name for spec in executable_image_tool_specs)
    crew_tool_specs = [
        spec
        for spec in all_tool_specs
        if spec.original_name not in excluded_tools
    ]
    crew_tools, tool_state = build_crewai_tools(
        tool_defs,
        agent_id=agent_id,
        mcp_url=mcp_url,
        api_key=api_key,
        challenge_image_uri=str(getattr(challenge, "input_image_uri", "") or ""),
        exclude_tools=excluded_tools,
    )
    tool_listing = ", ".join(
        f"{tool.name}->{tool_state.tool_name_map.get(tool.name, '?')}"
        for tool in crew_tools
    )
    print(f"   CrewAI tools: {tool_listing or '(none)'}")
    print()

    if modality == "text":
        task_description = _build_text_task_description(
            challenge,
            selection_ctx,
            [spec.original_name for spec in crew_tool_specs],
        )
        expected_output = "Return the final answer as: ANSWER: <your answer>"
    else:
        task_description = _build_image_task_description(
            challenge,
            selection_ctx,
            [spec.original_name for spec in crew_tool_specs],
            executable_image_tool_specs,
        )
        expected_output = "Return TOOL, INSTRUCTION, and SUMMARY lines."

    print("Solving with CrewAI native tools...")
    await asyncio.to_thread(http_client.update_status, agent_id, "thinking")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        "Starting CrewAI agent with native arena tools...",
    )

    start_ms = time.time() * 1000
    result = None
    active_model_name = model_name
    last_error: Exception | None = None
    print(f"   Starting with model: {model_name}")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        f"Starting model: {model_name}",
    )

    llm = LLM(
        model=model_name,
        api_key=llm_api_key,
        base_url=llm_host,
        api_base=llm_host,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    solver_agent = CrewAgent(
        role="Agent Gauntlet Challenge Solver",
        goal="Solve the current Agent Gauntlet challenge accurately and quickly.",
        backstory=(
            "You are a competitive AI agent solving timed Agent Gauntlet challenges. "
            "Use tools selectively, keep reasoning concise, and respect the required answer format."
        ),
        llm=llm,
        function_calling_llm=llm,
        tools=crew_tools,
        verbose=True,
        allow_delegation=False,
        max_iter=6 if modality == "image" else 8,
        max_execution_time=max(30, int(challenge.max_time_s or 0)),
    )
    solve_task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=solver_agent,
    )
    crew = Crew(agents=[solver_agent], tasks=[solve_task], verbose=True)
    try:
        result = await crew.kickoff_async()
    except Exception as exc:
        last_error = exc
        print(f"   Model '{model_name}' failed: {exc}")

    if result is None:
        final_error = last_error or RuntimeError("CrewAI solve failed for the selected model.")
        await asyncio.to_thread(http_client.update_status, agent_id, "failed")
        await asyncio.to_thread(
            http_client.broadcast_thought,
            agent_id,
            f"CrewAI solve failed: {final_error}",
        )
        raise final_error

    total_time_ms = int(time.time() * 1000 - start_ms)
    raw_content = getattr(result, "raw", str(result))
    usage_metrics = _extract_usage_metrics(result)
    print(f"   Solved with model: {active_model_name}")
    print(f"   Raw output: {raw_content[:200]}...")
    print(f"   Total time: {total_time_ms}ms")
    print(f"   Tokens: {usage_metrics['total_tokens']}")
    print()

    if modality == "text":
        answer = extract_answer(raw_content)
        print(f"Submitting answer: {answer}")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, f"Answer: {answer}")
        submit_result = await asyncio.to_thread(
            http_client.submit,
            agent_id,
            answer,
            {
                "model_name": active_model_name,
                "total_tokens": usage_metrics["total_tokens"],
                "prompt_tokens": usage_metrics["prompt_tokens"],
                "completion_tokens": usage_metrics["completion_tokens"],
                "total_time_ms": total_time_ms,
            },
            "text",
        )
        print(f"   Accepted: {submit_result.accepted}")
        print(f"   Status: {submit_result.status}")
        if submit_result.score:
            print(f"   Score: {submit_result.score.get('final_score', 0)}")
        await asyncio.to_thread(http_client.update_status, agent_id, "submitted")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, "Text challenge submitted.")
    else:
        image_uri = extract_image_uri(raw_content)
        planned_tool, planned_instruction, planned_summary = extract_image_plan(raw_content)
        image_uri, selected_tool = await _resolve_image_candidate(
            initial_image_uri=image_uri,
            tool_state=tool_state,
            challenge=challenge,
            image_tool_specs=executable_image_tool_specs,
            ranked_models=ranked_models,
            mcp_url=mcp_url,
            api_key=api_key,
            agent_id=agent_id,
            planned_tool=planned_tool,
            planned_instruction=planned_instruction,
        )
        async with McpArenaClient(mcp_url, api_key) as submit_mcp:
            submit_result = await submit_mcp.submit_image(
                agent_id=agent_id,
                image_uri=image_uri,
                client_metrics={
                    "model_name": active_model_name,
                    "planner_tool": selected_tool,
                    "total_tokens": usage_metrics["total_tokens"],
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "total_time_ms": total_time_ms,
                },
                rationale="CrewAI native tool run",
            )
        submit_log = dict(submit_result) if isinstance(submit_result, dict) else {"result": submit_result}
        submit_log.pop("edited_image", None)
        submit_log.pop("image_uri", None)
        print(f"   Image submission: {submit_log}")
        await asyncio.to_thread(http_client.update_status, agent_id, "submitted")
        if planned_summary:
            await asyncio.to_thread(http_client.broadcast_thought, agent_id, f"Image plan summary: {planned_summary}")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, "Image challenge completed and submitted.")

    print("\nAgent completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
