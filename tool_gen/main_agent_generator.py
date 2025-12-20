from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Dict, Any

import openai
from sweagent.tools.commands import Command, Argument
from sweagent.tools.utils import generate_command_docs

from tools import Tool


class MainAgentGenerator:
    """Generate a complete agent config by merging bundles and injecting a plan.

    - Prepends chosen subagent bundles to agent.tools.bundles
    - Generates a plain-text numbered plan from the provided prompt and
      injects it into agent.templates.instance_template
    - Preserves all other fields in the base agent config
    """

    def __init__(
        self,
        *,
        client: openai.Client,
        prompt_path: Path,
        model_name: str,
        temperature: float = 0.0,
        output_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.prompt_path = prompt_path
        self.model_name = model_name
        if model_name.startswith("openai/"):
            self._openai_model = model_name.replace("openai/", "", 1)
        elif model_name.startswith("together_ai/"):
            self._openai_model = model_name.replace("together_ai/", "", 1)
        else:
            self._openai_model = model_name
        self.temperature = temperature
        self._log_path = (output_dir / "llm_calls.log") if output_dir is not None else None

    def _log(self, text: str) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    def _add_subagent_bundles(self, cfg: Dict[str, Any], tools: List[Tool]) -> None:
        """Add subagent bundles to the agent config."""
        agent = cfg.setdefault("agent", {})
        tools_cfg = agent.setdefault("tools", {})
        bundles = list(tools_cfg.get("bundles", []))
        
        subagent_bundles = [tool.bundle_entry() for tool in tools]
        tools_cfg["bundles"] = subagent_bundles + bundles

    def _convert_tools_to_commands(self, tools: List[Tool]) -> List[Command]:
        """Convert Tool objects to Command objects for documentation."""
        commands = []
        for tool in tools:
            arguments = []
            for arg_dict in tool.arguments:
                arguments.append(Argument(
                    name=arg_dict.get("name", "arg"),
                    type=arg_dict.get("type", "string"),
                    required=arg_dict.get("required", False),
                    description=arg_dict.get("description", "")
                ))
            
            command = Command(
                name=tool.name,
                subagent=True,
                docstring=tool.docstring,
                signature=tool.signature,
                arguments=arguments
            )
            commands.append(command)
        return commands

    def _generate_plan(self, tools: List[Tool]) -> str:
        """Generate the main agent plan using LLM."""
        commands = self._convert_tools_to_commands(tools)
        overview = generate_command_docs(commands, [])
        
        prompt_template = self.prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("{{subagents_overview}}", overview)
        
        self._log("🤖 LLM Call - generate_main_agent_plan")
        self._log(f"📝 Prompt:\n{prompt}")
        resp = self.client.chat.completions.create(
            model=self._openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        plan_text = resp.choices[0].message.content or ""
        self._log(f"🤖 Response:\n{plan_text}")
        return plan_text.strip()

    def _inject_plan(self, cfg: Dict[str, Any], plan_text: str) -> None:
        """Inject the generated plan into the agent config by replacing {{plan}} placeholder."""
        agent = cfg.setdefault("agent", {})
        templates = agent.setdefault("templates", {})
        instance_template = templates.get("instance_template", "")
        templates["instance_template"] = instance_template.replace("{{plan}}", plan_text)

    def _get_fallback_plan(self) -> str:
        """Get a fallback plan if LLM generation fails."""
        return """1. Analyze the <pr_description> and outline the subtasks needed.
2. Use subagents to read and extract relevant code.
3. Use subagents to run commands (e.g., executing a reproduction script to confirm the error).
4. Use subagents to edit the source code to fix the issue.
   - Use `str_replace_editor` only for trivial, single-line fixes.
   - For anything larger, rely on subagents to handle the edits.
5. Use subagents to rerun the reproduction script and verify the fix.
6. Reflect on edge cases and ensure your fix handles them.
7. After you have solved the issue, use the submit tool to submit the changes to the repository."""

    def _generate_agent_config_internal(self, base_agent_config: Dict[str, Any], tools: List[Tool]) -> Dict[str, Any]:
        """Internal method that does the actual work of generating agent config."""
        cfg = copy.deepcopy(base_agent_config)
        
        # Add subagent bundles
        self._add_subagent_bundles(cfg, tools)
        
        # Generate and inject the plan
        try:
            plan_text = self._generate_plan(tools)
            self._inject_plan(cfg, plan_text)
        except Exception as e:
            self._log(f"⚠️ Plan generation failed, using fallback: {e}")
            fallback_plan = self._get_fallback_plan()
            self._inject_plan(cfg, fallback_plan)
        
        return cfg

    def generate_agent_config(self, base_agent_config: Dict[str, Any], tools: List[Tool]) -> Dict[str, Any] | None:
        try:
            return self._generate_agent_config_internal(base_agent_config, tools)
        except Exception as e:
            self._log(f"⚠️ Failed to generate agent config: {e}")
            return None



