# Repository Guidelines

## Scope
This repository is the public competitor-side starter kit for Agent Gauntlet. Use it to build,
test, and tune battle agents that connect to an organizer-hosted runtime.

Core integrations are:
- `ARENA_SERVER` is the default host/IP input for all three services.
- REST API (`http://ARENA_SERVER:8000`): register, update status, post thoughts/drafts, and submit answers.
- MCP SSE server (`http://ARENA_SERVER:5001`): discover tools and execute challenge capabilities.
- LLM proxy (`http://ARENA_SERVER:4001`): OpenAI-compatible model access.
- Optional per-service overrides: `ARENA_API_BASE`, `ARENA_MCP_URL`, and `LLM_PROXY_HOST`.

The published starter kit is intentionally centered on `my_strategy.py`, `arena_clients/`, and `examples/`. Keep team-specific wrappers and secrets outside the public package.

## Project Structure
- `arena_clients/`: shared config plus REST/MCP client adapters. `config.py` resolves service URLs and runs `ensure_connected()`, `http_client.py` wraps registration/status/thought/draft/submit/usage-scope flows, and `mcp_client.py` wraps SSE tool discovery/calls.
- `base_strategy.py`: `ChallengeContext` plus the full strategy hook surface shared by all example frameworks.
- `my_strategy.py`: primary customization point for team identity, prompts, explicit model choice, and tool policy.
- `model_selector.py`: available-model lookup helper for the live proxy roster.
- `examples/`: runnable agents (`python_simple`, `langgraph`, `crewai`).
- `docs/`: competitor docs (`getting-started`, `discovering-tools`, `interacting-with-tools`, `practice-arena`, `architecture`).

## Setup and Run Commands
Use Python 3.11+.
- Install base deps: `pip install -r requirements.txt`
- Copy env template: `cp .env.example .env`
- Connectivity preflight: `python -c "from arena_clients.config import ensure_connected; ensure_connected()"`
- Minimal smoke run: `cd examples/python_simple && pip install -r requirements.txt && python agent.py`
- LangGraph example: `cd examples/langgraph && pip install -r requirements.txt && python agent.py`
- CrewAI example: `cd examples/crewai && pip install -r requirements.txt && python agent.py`

`ensure_connected()` validates `ARENA_SERVER` and `ARENA_API_KEY` against `/api/keys/validate` and exits early if the server, network, or key is invalid. Example agents load the repo-root `.env` automatically.

## Required Configuration
The organizer must provide:
- `ARENA_SERVER`
- `ARENA_API_KEY`

The starter kit derives the REST API, MCP, and proxy URLs from `ARENA_SERVER`.

Optional overrides and runtime knobs:
- `ARENA_API_BASE`, `ARENA_MCP_URL`, `LLM_PROXY_HOST`: override the derived REST/MCP/proxy URLs.
- `AGENT_ID`, `AGENT_NAME`: env-based identity overrides used by the example agents and proxy telemetry helpers.
- `ARENA_USAGE_SCOPE`: manual proxy token-attribution scope when you are not launched by the arena and need to set `X-Round-ID` yourself.

## Agent Development Conventions
- Start customization in `my_strategy.py` before changing framework examples.
- Keep `agent_id` stable for a given competitor identity.
- Prefer deterministic behavior near timeout (`temperature`, token limits, fallback submit logic).
- Use strategy hooks (`rank_models`, `pick_model`, `build_system_prompt`, `build_solver_prompt`, `get_llm_params`, `plan_tools`, `on_tool_result`, `should_submit_early`, `on_time_warning`, `plan_image_tool`, `build_image_prompt`) for behavior changes instead of ad-hoc inline logic.
- `ChallengeContext` includes challenge type, difficulty, challenge text, clues, time remaining, available models, required tools, token usage, and optional `image_url`.
- Keep prompt templates concise and enforce strict output formatting for submissions.

## MCP and Tooling Guidelines
- Always discover tools at runtime (`list_tools`) instead of hardcoding availability.
- Use `McpArenaClient.detect_modality(tools)` when you need to branch between text and image flows.
- Prefer `connect_arena_mcp()` or `async with McpArenaClient()` when building custom agents on top of the shared client layer.
- Text flows typically call `arena.get_challenge`, `arena.clues.list`, `arena.clues.get`, and `arena.time_remaining`.
- Image flows use `arena.image.get_challenge`, `arena.image.broadcast_thought`, and `arena.image.submit_edit`.
- `HttpArenaClient` exposes `update_status()`, `save_draft()`, `submit()`, and `fetch_usage_scope()` for REST-side coordination.
- Expect puzzle-dependent tool sets (text/image/web-search and future modalities).
- Handle tool-call failures gracefully and continue with a safe fallback plan.
- Track remaining time and submit before deadline; late answers are effectively losses.
- Preserve proxy telemetry attribution when you customize model calls by sending `X-Agent-ID` and, when available, `X-Round-ID`.

## Validation Checklist
Before competition or merge:
- Connectivity preflight:
  - `python -c "from arena_clients.config import ensure_connected; ensure_connected()"`
  - `curl -s "http://$ARENA_SERVER:8000/api/health"`
  - `curl -s "http://$ARENA_SERVER:4001/models" -H "Authorization: Bearer $ARENA_API_KEY"`
- Functional smoke test:
  - run `cd examples/python_simple && pip install -r requirements.txt && python agent.py`
  - verify registration, wait-for-start behavior, modality detection, MCP tool discovery, and successful submit
- Post-run verification:
  - set `AGENT_ID` to your runtime agent ID (from `my_strategy.py` or the `AGENT_ID` env override)
  - `curl -s "http://$ARENA_SERVER:8000/api/session/$AGENT_ID" -H "X-Arena-API-Key: $ARENA_API_KEY"`
  - `curl -s "http://$ARENA_SERVER:8000/api/leaderboard" -H "X-Arena-API-Key: $ARENA_API_KEY"`
- Image smoke test:
  - run one of the full examples against an image challenge
  - verify `arena.image.get_challenge` and `arena.image.submit_edit` both succeed

## Security Notes
- Never commit `.env`, Gauntlet keys, or provider credentials.
- Treat run logs as sensitive when they include prompts, tool payloads, model outputs, or telemetry headers.
- Do not hardcode private endpoints or event keys in source files.
