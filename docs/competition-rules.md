# Agent Gauntlet Competition Rules

This document captures the high-level event guidance shared with competitors for the live
Agent Gauntlet event at GTC. It is intentionally focused on expectations, not hidden challenge
details.

## Participation Requirements

- Bring a working agent before you arrive. There is no coding time on stage, only execution.
- Participation is tied to the live event flow and on-site logistics described on the
  [Live Agent Gauntlet event page](https://luma.com/gtc-live-agent-gauntlet).
- Expect your progress to be visible on-screen during the event.

## Tournament Format

- Teams compete in groups of 4 during each active round.
- Top-performing teams from each group advance to the next round.
- The event runs as multiple elimination rounds until a final head-to-head.
- If teams tie in an earlier round, they can both advance.
- If the final round ends in a tie, organizers can run a fresh challenge to break it.

## Challenge Modalities

This release is focused on the currently active paths in the repos:

- **Text challenges**: logic, reasoning, retrieval, structured output, and live research tasks
- **Image challenges**: image understanding, editing, and generation workflows

Each challenge is time-boxed, so reliability and time management matter.

## Judging Criteria

Judging is intentionally shared at a high level rather than as a public formula.

- Answer quality is weighted most heavily.
- Speed, tool usage, model usage, and token efficiency also contribute.
- There is a quality threshold before efficiency bonuses meaningfully help.
- Token usage is tracked server-side rather than trusted from self-reported numbers.

The best strategy is usually a balanced one: accurate enough to score well, fast enough to finish,
and efficient enough to avoid wasting budget.

## The Black Box MCP Model

The MCP server should be treated like a black box.

- Tool availability can change by challenge.
- Your agent is expected to discover tools at runtime rather than assume a fixed tool list.
- Unknown functions, hidden capabilities, and challenge-specific workflows are part of the event.
- Hardcoding tool names or fixed assumptions is risky.

## What Teams Control

Teams are expected to differentiate themselves through engineering choices inside their own agents:

- prompt design
- explicit model selection strategy
- tool orchestration
- timeout and early-submit behavior
- image workflow decisions

## What Stays Fixed

Some parts of the event are fixed for fairness:

- the API contract
- the MCP server transport
- the active model roster exposed by the proxy
- organizer-controlled timing and round progression
- organizer-controlled battle keys

## Practice Guidance

- Use the practice environment before event day.
- Use the organizer-provided starter kit setup and credentials for your runs.

## Community

- Channel: `#gtc-live-agent-gauntlet`
