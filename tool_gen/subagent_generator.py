"""
Subagent generation module for SWE-agent tool evolution.

This module provides functionality to generate experiment configurations
and corresponding subagent configurations using LLM prompts.
"""

import openai
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tools import Tool, ToolArchive
from utils import extract_yaml_from_response

class SubagentGenerator:
    """Design and generate subagent configurations."""
    
    def __init__(self, client: openai.Client, prompt_dir: Path, prompt_filename: str, template_dir: Path, output_dir: Path, model_name: str = "deepseek-v3", temperature: float = 0.0):
        """Initialize the subagent generator."""
        self.client = client
        self.prompt_dir = prompt_dir
        self.prompt_filename = prompt_filename
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.temperature = temperature
        self.rng = random.Random(42)
        # For OpenAI client calls, we need to strip the provider prefix

        if model_name.startswith("openai/"):
            self.openai_model_name = model_name.replace("openai/", "", 1)
        elif model_name.startswith("together_ai/"):
            self.openai_model_name = model_name.replace("together_ai/", "", 1)
        else:
            self.openai_model_name = model_name
        self.llm_calls_log_path = output_dir / "llm_calls.log"
        # Clear the LLM log at the start of the run
        with open(self.llm_calls_log_path, "w", encoding="utf-8"):
            pass

    def log_llm_call(self, message: str, model_name: str = None) -> None:
        with open(self.llm_calls_log_path, "a", encoding="utf-8") as f:
            if model_name:
                f.write(f"Model: {model_name}\n")
            f.write(message)
            if not message.endswith('\n'):
                f.write('\n')

    def log_llm_iteration_separator(self, iteration_name: str) -> None:
        sep = f"\n{'='*30} LLM ITERATION: {iteration_name} {'='*30}\n"
        with open(self.llm_calls_log_path, "a", encoding="utf-8") as f:
            f.write(sep)

    def _read_yaml_file(self, file_path: Path) -> dict:
        """Read YAML file safely, return empty dict if file doesn't exist or can't be read."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}
    
    def format_feedback(self, tool_archive: ToolArchive) -> str:
        """Format feedback for inclusion in prompts."""
        feedback_text = ""
        if not tool_archive.tools:
            return feedback_text

        # get all tool name, config:
        tool_info = [(tool.name, tool.docstring) for tool in tool_archive.tools]
        tool_info_str = "\n".join([f"{name}: {docstring}" for name, docstring in tool_info])

        k = min(2, len(tool_archive.tools))
        feedback_text += f"Complete configs for sampled {k} tools:\n"
        # sample tools proportional to helpful rate:
        weights = [tool.helpful_count / tool.n if tool.n > 0 else 0.0 for tool in tool_archive.tools]
        # If all weights are zero, fall back to uniform sampling
        if all(weight == 0.0 for weight in weights):
            weights = [1.0 for _ in tool_archive.tools]
        # weighted sampling without replacement
        samples = []
        pool = list(tool_archive.tools)
        w = list(weights)
        for _ in range(min(k, len(pool))):
            idx = self.rng.choices(range(len(pool)), weights=w, k=1)[0]
            samples.append(pool.pop(idx))
            w.pop(idx)
        for tool in samples:
            feedback_text += f"Tool: {tool.name}\n"
            feedback_text += f"Signature: {tool.signature}\n"
            feedback_text += f"Docstring: {tool.docstring}\n"
            feedback_text += f"System template: {tool.system_template}\n"
            feedback_text += f"Instance template: {tool.instance_template}\n"
            if tool.n > 0:
                feedback_text += f"RESULTS: {tool.helpful_count / tool.n}% of tasks where subagent was helpful\n"
            else:
                feedback_text += f"RESULTS: N/A (no runs yet)\n"
            if tool.subagent_invoked_count > 0 and tool.average_token_count > 0:
                feedback_text += f"TOKEN USAGE: Average {tool.average_token_count:.0f} tokens per use\n"
            feedback_text += "\n"

        feedback_text += f"\nCRITICAL: Please create a subagent DIFFERENT from all previous tools:\n{tool_info_str}\n\n"
        
        return feedback_text

    def _generate_subagent_tool(self, tool_archive: ToolArchive) -> Dict[str, Any]:
        """Generate new subagent using LLM."""
        prompt = (self.prompt_dir / "generate_subagent_tool_prompt.txt").read_text()
        prompt += self.format_feedback(tool_archive)

        # Log the prompt
        self.log_llm_call(f"🤖 LLM Call - generate_subagent_tool")
        self.log_llm_call(f"📝 Prompt:\n{prompt}")
        
        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""

        # Log the response
        self.log_llm_call(f"🤖 Response:\n{content}", self.openai_model_name)
        
        tool_yaml = extract_yaml_from_response(content)
        return tool_yaml
    
    def generate_subagent_parts(self, tool_config: Dict, base_prompt_path: Path, tool_archive: ToolArchive) -> Tuple[str, str]:
        """Generate system and instance templates for a specific tool."""
        base_prompt = base_prompt_path.read_text()
        full_prompt = base_prompt + self.format_feedback(tool_archive)
        full_prompt += f"Tool configuration to generate templates for:\n"
        full_prompt += f"\n{yaml.dump(tool_config)}\n\n"
        
        # Log the prompt
        self.log_llm_call(f"🤖 LLM Call - generate_subagent_parts (tool: {list(tool_config.keys())[0]})")
        self.log_llm_call(f"📝 Prompt:\n{full_prompt}")
        
        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        self.log_llm_call(f"🤖 Response:\n{content}")

        templates = extract_yaml_from_response(content)
        return templates['system_template'], templates['instance_template']

    def _deduplicate_tool_name(self, name: str, tool_archive: ToolArchive) -> str:
        existing_tool_names = [tool.name for tool in tool_archive.tools]
        idx = 1
        base = name
        while name in existing_tool_names:
            name = f"{base}_{idx:02d}"
            idx += 1
        return name
    
    def generate_new_tool(self, tool_archive: ToolArchive, exp_num: int) -> Tool:
        tool_config = self._generate_subagent_tool(tool_archive)
        tool_name = list(tool_config.keys())[0]
        system_template, instance_template = self.generate_subagent_parts(tool_config, self.prompt_dir / self.prompt_filename, tool_archive)

        unique_tool_name = self._deduplicate_tool_name(tool_name, tool_archive)
        
        tool = Tool(
            name=unique_tool_name,
            signature=tool_config[tool_name]['signature'],
            docstring=tool_config[tool_name]['docstring'],
            arguments=tool_config[tool_name]['arguments'],
            bundle_dir=tool_archive.output_dir/(unique_tool_name),
            subagent=True,
            system_template=system_template,
            instance_template=instance_template,
            exp_num=exp_num
        )
        tool.write_to_file()
        return tool