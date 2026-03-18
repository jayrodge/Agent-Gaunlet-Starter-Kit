#!/usr/bin/env python3
"""Simple Agent Gauntlet competitor agent.

This is a minimal example showing how to connect to Agent Gauntlet from any Python code.
It uses:
- HTTP client for Gauntlet coordination (REST API)
- MCP client for challenge tools
- OpenAI SDK for LLM calls via the proxy

Usage:
    cd examples/python_simple
    python agent.py

This example loads `.env` from the repository root automatically.
"""

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

# Use OpenAI SDK for LLM calls through the proxy.
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)


def extract_answer(raw_response: str) -> str:
    """Extract the answer line from the LLM response."""
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_response, flags=re.DOTALL).strip()

    # Accept either "ANSWER:" or "Final answer:" on its own line.
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

    return ""


def _build_context(
    *,
    challenge_type: str,
    description: str,
    rules: str,
    clues: list[str] | None = None,
    max_time_s: int = 0,
    available_models: list[str] | None = None,
    time_remaining_s: float = 0.0,
    tokens_used: int = 0,
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
        available_models=list(available_models or []),
        tools_used=[],
        tokens_used=tokens_used,
        required_tools=[],
        image_url=image_url,
    )


def _require_selected_model(
    strategy: MyStrategy,
    *,
    stage: str,
    ranked_models: list[str],
    available_models: list[str],
    ctx: ChallengeContext,
) -> str:
    model_name = str(strategy.pick_model(stage, ranked_models, ctx) or "").strip()
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


def _coerce_nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value if value is not None else default))
    except (TypeError, ValueError):
        return default


def _resolve_usage_scope() -> str | None:
    scope_value = str(os.getenv("ARENA_USAGE_SCOPE") or "").strip()
    return scope_value or None


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
    """Best-effort fetch of proxy usage for one agent, scoped when possible."""
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


def _build_live_metrics(model_name: str, usage: dict[str, int] | None, elapsed_ms: int) -> dict[str, object]:
    usage = usage or {}
    prompt_tokens = _coerce_nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _coerce_nonnegative_int(usage.get("total_tokens"))
    return {
        "model_name": str(model_name or "").strip(),
        "total_tokens": str(total_tokens),
        "prompt_tokens": str(prompt_tokens),
        "completion_tokens": str(completion_tokens),
        "total_time_ms": int(max(0, elapsed_ms)),
    }


async def solve_challenge(
    challenge,
    clues: list[str],
    llm_client,
    model_name: str,
    strategy: MyStrategy,
    ctx: ChallengeContext,
    *,
    http_client=None,
    agent_id: str = "",
    broadcast_thought: bool = True,
):
    """Solve the challenge using the selected LLM model.
    
    When http_client and agent_id are provided, streams reasoning to the arena in real time.
    
    Returns:
        tuple: (extracted_answer, raw_response, model_name, usage_dict, ttft_ms, total_time_ms)
    """
    
    _ = challenge
    _ = clues
    prompt = strategy.build_solver_prompt(ctx)
    system_msg = strategy.build_system_prompt(ctx)
    llm_params = strategy.get_llm_params(ctx)
    max_tokens = int(llm_params.get("max_tokens", 1024))
    temperature = float(llm_params.get("temperature", 0.0))
    
    # Track timing
    start_ms = time.time() * 1000
    ttft_ms = 0
    raw_content_parts = []
    pending_broadcast = ""
    _BROADCAST_CHUNK = 80  # chars before sending a thought

    def _flush_broadcast(force: bool = False) -> None:
        nonlocal pending_broadcast
        if not http_client or not agent_id or not broadcast_thought:
            return
        if force or len(pending_broadcast) >= _BROADCAST_CHUNK or "\n" in pending_broadcast:
            chunk = pending_broadcast.strip()
            pending_broadcast = ""
            if chunk:
                try:
                    http_client.broadcast_thought(agent_id, chunk[:300])
                except Exception:
                    pass

    stream_enabled = os.getenv("LLM_STREAM", "1").lower() not in {"0", "false", "no"}
    usage = None
    if stream_enabled and http_client and agent_id:
        # Stream and broadcast in real time
        try:
            stream = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            header_sent = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
                    text = reasoning or content
                    if text:
                        if not ttft_ms:
                            ttft_ms = int(time.time() * 1000 - start_ms)
                        raw_content_parts.append(text)
                        if broadcast_thought:
                            if not header_sent:
                                http_client.broadcast_thought(agent_id, "💭 LLM Reasoning:")
                                header_sent = True
                            pending_broadcast += text
                            _flush_broadcast()
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
            _flush_broadcast(force=True)
        except Exception as e:
            stream_enabled = False
            raw_content_parts = []
            if "stream" in str(e).lower() or "Stream" in str(e):
                pass  # Fall through to non-streaming
            else:
                raise

    if not stream_enabled or not raw_content_parts:
        # Non-streaming fallback
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_content = response.choices[0].message.content or ""
        usage = response.usage
        ttft_ms = ttft_ms or getattr(response, "ttft_ms", 0) or int((time.time() * 1000 - start_ms) * 0.1)
    else:
        raw_content = "".join(raw_content_parts)

    total_time_ms = int(time.time() * 1000 - start_ms)
    answer = extract_answer(raw_content)
    if not answer:
        strict_system_msg = (
            "Return only one line in exact format: ANSWER: <final answer>. "
            "No preamble, no tags, no explanation, no additional lines."
        )
        try:
            strict_response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": strict_system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=min(temperature, 0.0),
                stream=False,
            )
            raw_content = strict_response.choices[0].message.content or ""
            usage = strict_response.usage
            total_time_ms = int(time.time() * 1000 - start_ms)
            answer = extract_answer(raw_content)
        except Exception:
            pass

    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    if not usage_dict["total_tokens"] and raw_content:
        usage_dict["total_tokens"] = len(raw_content.split()) * 2  # rough estimate
    ttft_ms = ttft_ms or int(total_time_ms * 0.1)

    return answer, raw_content, model_name, usage_dict, ttft_ms, total_time_ms


async def main():
    """Main agent loop."""
    strategy = MyStrategy()
    agent_id = (
        os.getenv("AGENT_ID")
        or str(getattr(strategy, "agent_id", "")).strip()
        or "simple-agent"
    ).strip()
    agent_name = (
        os.getenv("AGENT_NAME")
        or str(getattr(strategy, "agent_name", "")).strip()
        or "Team Nova"
    ).strip()
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENT_NAME"] = agent_name
    print(f"🤖 {agent_name} starting...")
    print(f"   Agent ID: {agent_id}")

    ensure_connected()

    # Initialize clients
    api_base = get_api_base()
    mcp_url = get_mcp_url()
    llm_host = get_proxy_host()
    api_key = get_arena_api_key()
    llm_api_key = get_llm_api_key()

    print(f"   API: {api_base}")
    print(f"   MCP: {mcp_url}")
    print(f"   LLM: {llm_host}")
    print()

    # HTTP client for Gauntlet coordination
    http_client = HttpArenaClient(api_base=api_base, api_key=api_key)

    if not HAS_OPENAI:
        raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")
    
    # Step 1: Register with Agent Gauntlet
    print("📝 Registering with Agent Gauntlet...")
    session = http_client.register(agent_id, agent_name)
    print(f"   Session: {session.session_id}")
    print(f"   Status: {session.status}")
    
    # Step 2: Get challenge from MCP
    print("\n🎯 Getting challenge...")
    async with McpArenaClient(mcp_url) as mcp_client:
        tools = await mcp_client.list_tools()
        modality = McpArenaClient.detect_modality(tools)
        usage_scope = http_client.fetch_usage_scope() or _resolve_usage_scope()
        if usage_scope:
            os.environ["ARENA_USAGE_SCOPE"] = usage_scope

        live_prompt_tokens = 0
        live_completion_tokens = 0
        live_total_tokens = 0
        active_model_name = ""
        metrics_started_ms = 0.0
        reporter_stop = asyncio.Event()
        metrics_task: asyncio.Task | None = None

        def _elapsed_metrics_ms() -> int:
            if metrics_started_ms <= 0:
                return 0
            return int(max(0.0, time.time() * 1000 - metrics_started_ms))

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
            live_prompt_tokens = _coerce_nonnegative_int(usage.get("prompt_tokens"))
            live_completion_tokens = _coerce_nonnegative_int(usage.get("completion_tokens"))
            live_total_tokens = _coerce_nonnegative_int(usage.get("total_tokens"))

        async def _push_live_status(status: str = "running") -> None:
            await _refresh_live_usage()
            try:
                await asyncio.to_thread(
                    http_client.update_status,
                    agent_id,
                    status,
                    _build_live_metrics(
                        active_model_name,
                        {
                            "prompt_tokens": live_prompt_tokens,
                            "completion_tokens": live_completion_tokens,
                            "total_tokens": live_total_tokens,
                        },
                        _elapsed_metrics_ms(),
                    ),
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

        async def _start_metrics_reporter(model_name: str) -> None:
            nonlocal active_model_name, metrics_started_ms, metrics_task
            active_model_name = str(model_name or "").strip()
            metrics_started_ms = time.time() * 1000
            reporter_stop.clear()
            await _push_live_status("running")
            metrics_task = asyncio.create_task(_runtime_metrics_reporter())

        async def _stop_metrics_reporter() -> None:
            reporter_stop.set()
            if metrics_task is not None:
                try:
                    await asyncio.wait_for(metrics_task, timeout=2.0)
                except Exception:
                    if not metrics_task.done():
                        metrics_task.cancel()
            await _push_live_status("running")

        if modality == "image":
            challenge = None
            while True:
                try:
                    challenge = await mcp_client.get_image_challenge(agent_id)
                    break
                except McpArenaError as e:
                    message = str(e).lower()
                    if "locked" in message or "waiting for organizer" in message:
                        print("   Lobby open; waiting for organizer to start battle...")
                        await asyncio.sleep(3.0)
                    else:
                        raise
            print(f"   Type: {challenge.challenge_type}")
            print(f"   Puzzle: {challenge.puzzle_id}")
            print(f"   Time limit: {challenge.max_time_s}s")
            usage_scope = http_client.fetch_usage_scope() or _resolve_usage_scope()
            if usage_scope:
                os.environ["ARENA_USAGE_SCOPE"] = usage_scope

            available_models = fetch_available_models(llm_host, llm_api_key)
            image_rules = "\n".join(
                part
                for part in (challenge.prompt, challenge.reference_notes)
                if isinstance(part, str) and part.strip()
            )
            image_prompt = (challenge.prompt or challenge.description or "").strip()
            image_ctx = _build_context(
                challenge_type=challenge.challenge_type,
                description=challenge.description,
                rules=image_rules,
                max_time_s=challenge.max_time_s,
                available_models=available_models,
                image_url=challenge.input_image_uri or None,
            )
            ranked_models = strategy.rank_models(image_ctx, available_models)
            model_name = _require_selected_model(
                strategy,
                stage="solve",
                ranked_models=ranked_models,
                available_models=available_models,
                ctx=image_ctx,
            )
            print(f"   Selected model: {model_name}")
            http_client.broadcast_thought(agent_id, f"Selected planner model: {model_name}")
            await _start_metrics_reporter(model_name)

            start_ms = time.time() * 1000
            prompt_text = strategy.build_image_prompt(image_ctx).strip() or image_prompt
            image_tools = [
                tool_name
                for tool_name in ("image_edit", "image_generate", "image_analyze")
                if tool_name in tools
            ]
            selected_tool = strategy.plan_image_tool(image_ctx, image_tools)
            tool_result: dict = {}
            if (
                selected_tool == "image_edit"
                and challenge.input_image_uri
                and "image_edit" in image_tools
            ):
                print("   Using image_edit tool")
                tool_result = await mcp_client.call_tool(
                    "image_edit",
                    {
                        "image_uri": challenge.input_image_uri,
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                    },
                )
            elif selected_tool == "image_generate" and "image_generate" in image_tools:
                print("   Using image_generate tool")
                tool_result = await mcp_client.call_tool(
                    "image_generate",
                    {"prompt": prompt_text, "agent_id": agent_id},
                )
            elif (
                selected_tool == "image_analyze"
                and challenge.input_image_uri
                and "image_analyze" in image_tools
            ):
                print("   Using image_analyze tool (no image output tool available)")
                tool_result = await mcp_client.call_tool(
                    "image_analyze",
                    {
                        "image_uri": challenge.input_image_uri,
                        "question": prompt_text,
                        "agent_id": agent_id,
                    },
                )
                analysis_text = str(tool_result.get("text") or "").strip()
                if analysis_text:
                    http_client.broadcast_thought(
                        agent_id,
                        f"Image analysis: {analysis_text[:180]}",
                    )
            elif challenge.input_image_uri and "image_edit" in image_tools:
                selected_tool = "image_edit"
                tool_result = await mcp_client.call_tool(
                    "image_edit",
                    {
                        "image_uri": challenge.input_image_uri,
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                    },
                )
            elif "image_generate" in image_tools:
                selected_tool = "image_generate"
                tool_result = await mcp_client.call_tool(
                    "image_generate",
                    {"prompt": prompt_text, "agent_id": agent_id},
                )

            if tool_result.get("error"):
                print(f"   Tool warning: {tool_result.get('error')}")

            output_image_uri = str(tool_result.get("image_uri") or "").strip()
            if not output_image_uri and challenge.input_image_uri:
                output_image_uri = challenge.input_image_uri
            if not output_image_uri:
                output_image_uri = BLANK_PNG_DATA_URI

            total_time_ms = int(time.time() * 1000 - start_ms)
            await _stop_metrics_reporter()
            image_submit_result = await mcp_client.submit_image(
                agent_id=agent_id,
                image_uri=output_image_uri,
                client_metrics={
                    "model_name": model_name,
                    "planner_tool": selected_tool,
                    "total_tokens": str(live_total_tokens),
                    "prompt_tokens": str(live_prompt_tokens),
                    "completion_tokens": str(live_completion_tokens),
                    "ttft_ms": 0,
                    "total_time_ms": total_time_ms,
                },
                rationale="simple_agent dynamic image flow",
            )
            print(f"   Image submission: {image_submit_result}")
            print("\n✅ Agent completed!")
            return

        # Wait for organizer to start (challenge is locked while lobby is open)
        challenge = None
        while True:
            try:
                challenge = await mcp_client.get_challenge(agent_id)
                break
            except McpArenaError as e:
                message = str(e).lower()
                if "locked" in message or "waiting for organizer" in message:
                    print("   Lobby open; waiting for organizer to start battle...")
                    await asyncio.sleep(3.0)
                else:
                    raise
        print(f"   Type: {challenge.challenge_type}")
        print(f"   Puzzle: {challenge.puzzle_id}")
        print(f"   Time limit: {challenge.max_time_s}s")
        usage_scope = http_client.fetch_usage_scope() or _resolve_usage_scope()
        if usage_scope:
            os.environ["ARENA_USAGE_SCOPE"] = usage_scope
        llm_client = OpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
            default_headers=_build_proxy_headers(agent_id, usage_scope),
        )
        print("   LLM: Enabled")
        print()

        available_models = fetch_available_models(llm_host, llm_api_key)
        model_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            max_time_s=challenge.max_time_s,
            available_models=available_models,
        )
        ranked_models = strategy.rank_models(model_ctx, available_models)
        model_name = _require_selected_model(
            strategy,
            stage="solve",
            ranked_models=ranked_models,
            available_models=available_models,
            ctx=model_ctx,
        )
        print(f"   Selected model: {model_name}")
        await _start_metrics_reporter(model_name)
        
        # Broadcast thought
        http_client.broadcast_thought(agent_id, "Reading challenge and clues...")
        
        # Get all clues
        print("\n📖 Reading clues...")
        clues = []
        clue_ids = await mcp_client.list_clues(agent_id)
        for i, clue_id in enumerate(clue_ids):
            clue = await mcp_client.get_clue(clue_id, agent_id)
            clues.append(clue.text)
            print(f"   Clue {i}: {clue.text[:50]}...")
        if not clues:
            clues = [str(clue_text) for clue_text in challenge.clues]
        
        # Broadcast thought
        http_client.broadcast_thought(agent_id, f"Analyzing {len(clues)} clues...")
        
        # Save draft (backup)
        http_client.save_draft(agent_id, "Working on solution...")
        
        # Step 3: Solve
        print("\n🧠 Solving...")
        http_client.broadcast_thought(agent_id, "Calling LLM...")
        solve_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            clues=clues,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
        )
        
        answer, raw_response, model_name, usage, ttft_ms, total_time_ms = await solve_challenge(
            challenge,
            clues,
            llm_client,
            model_name,
            strategy,
            solve_ctx,
            http_client=http_client,
            agent_id=agent_id,
            broadcast_thought=True,
        )
        print(f"   Raw LLM output: {raw_response[:120]}...")
        print(f"   Extracted answer: {answer}")
        print(f"   Model: {model_name} | Tokens: {usage.get('total_tokens', 0)} | Time: {total_time_ms}ms")
        
        if not answer:
            print("   ⚠️  No valid ANSWER format found; using last line as fallback.")
            lines = raw_response.strip().splitlines()
            last_line = lines[-1].strip().strip("`\"'") if lines else ""
            answer = last_line if last_line and len(last_line) < 400 else "unknown"

        # Save draft with extracted answer as backup
        http_client.save_draft(agent_id, answer, "LLM solution")
        http_client.broadcast_thought(agent_id, f"Answer: {answer}")
        
        # Check time remaining
        time_info = await mcp_client.time_remaining(agent_id)
        remaining_s = float(time_info.get("time_remaining_s", 0.0))
        submit_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            clues=clues,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
            time_remaining_s=remaining_s,
            tokens_used=usage.get("total_tokens", 0),
        )
        if strategy.should_submit_early(answer, submit_ctx):
            http_client.broadcast_thought(agent_id, "⚡ Strategy submitting early.")
        revised_answer = strategy.on_time_warning(remaining_s, answer, submit_ctx)
        if isinstance(revised_answer, str) and revised_answer.strip():
            answer = revised_answer.strip()
        print(f"   Time remaining: {time_info['time_remaining_s']:.1f}s")
    
    # Step 4: Submit the extracted answer with real metrics
    print("\n📤 Submitting answer...")
    http_client.broadcast_thought(agent_id, "Submitting final answer!")
    await _stop_metrics_reporter()
    
    result = http_client.submit(
        agent_id=agent_id,
        answer=answer,
        client_metrics={
            "model_name": model_name,
            "total_tokens": str(live_total_tokens),
            "prompt_tokens": str(live_prompt_tokens),
            "completion_tokens": str(live_completion_tokens),
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms,
        }
    )
    
    print(f"   Accepted: {result.accepted}")
    print(f"   Status: {result.status}")
    if result.score:
        print(f"   Score: {result.score.get('final_score', 0)}")
        print(f"   Quality: {result.score.get('quality_score', 0)}")
        print(f"   Speed: {result.score.get('speed_score', 0)}")
    
    print("\n✅ Agent completed!")


if __name__ == "__main__":
    asyncio.run(main())
