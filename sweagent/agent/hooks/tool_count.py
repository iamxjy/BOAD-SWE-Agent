from __future__ import annotations

import json
from pathlib import Path

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory


class ToolCountAgentHook(AbstractAgentHook):
    """Counts tool invocations by first token of the action and stores it in info.

    Counts are stored under info["tool_call_counts"] as a dict[str, int].
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._instance_dir: Path | None = None
        self._counts_path: Path | None = None
        self._agent = None

    def on_init(self, *, agent) -> None:  # type: ignore[override]
        self._agent = agent

    def on_setup_done(self) -> None:  # type: ignore[override]
        try:
            traj_path = getattr(self._agent, "traj_path", None)
            if traj_path is not None:
                self._instance_dir = Path(traj_path).parent
                self._counts_path = self._instance_dir / "tool_call_counts.json"
        except Exception:
            self._instance_dir = None
            self._counts_path = None

    def _bump(self, action: str) -> None:
        action = (action or "").strip()
        if not action:
            return
        token = action.split()[0]
        if not token:
            return
        self._counts[token] = int(self._counts.get(token, 0)) + 1

    def on_actions_generated(self, *, step: StepOutput) -> None:  # type: ignore[override]
        # Do not count here to avoid double counting; we count on action start
        return

    def on_action_started(self, *, step: StepOutput) -> None:  # type: ignore[override]
        # Count as soon as an action starts to be resilient to step failures
        self._bump(step.action)
        self._persist()

    def on_step_done(self, *, step: StepOutput, info: AgentInfo) -> None:  # type: ignore[override]
        # Nothing else to do; counts already bumped on action start
        self._persist()
        self._persist()

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo) -> None:  # type: ignore[override]
        # Store final aggregated counts into the shared info dict\
        info["tool_call_counts"] = dict(self._counts)
        self._persist()

    def _persist(self) -> None:
        if self._counts_path is None:
            return
        try:
            self._counts_path.write_text(json.dumps(self._counts, indent=2))
        except Exception:
            # Best-effort; do not crash the agent due to logging issues
            pass


