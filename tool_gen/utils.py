"""Utility functions for working with LLM responses."""

from typing import Dict, Any, List
import yaml
from pathlib import Path

# LM UTILS:
def extract_yaml_from_response(content: str, expected_top_key: str = None) -> Dict[str, Any]:
    """Extract and parse YAML from LLM response."""
    content = content.strip()
    
    # Extract YAML from code block
    if "```yaml" in content:
        start = content.find("```yaml") + 7
        end = content.find("```", start)
        if end == -1:
            raise ValueError("Malformed YAML code block")
        yaml_content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end == -1:
            raise ValueError("Malformed code block")
        yaml_content = content[start:end].strip()
    else:
        yaml_content = content
    
    # Parse YAML
    parsed = yaml.safe_load(yaml_content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed content is not a dict: {parsed}")
    
    # Validate expected key
    if expected_top_key and expected_top_key not in parsed:
        raise ValueError(f"Response missing '{expected_top_key}' key")
    
    return parsed


# FILE UTILS:
def get_traj_paths(iteration_dir: Path, subagent_name: str = None) -> List[Path]:
    """
    Get all trajectory paths in the iteration directory.
    Outputs list of paths with main agent first, then subagents by call number
    """
    # Get all .traj files in the iteration directory (subagent and main agent traj)
    all_traj_paths = list(iteration_dir.rglob("*.traj"))

    # If a specific subagent is requested, return only that subagent's calls sorted by call number
    if subagent_name:
        def is_target_subagent(traj_path: Path) -> bool:
            parent_name = traj_path.parent.name
            if traj_path.parent == iteration_dir:
                return False
            return parent_name.startswith(f"subagent_{subagent_name}_")

        def call_number(traj_path: Path) -> int:
            # Directory name format: subagent_{name}_{call_number}; take the last suffix
            try:
                suffix = traj_path.parent.name.rsplit("_", 1)[-1]
                return int(suffix)
            except Exception:
                return 10_000

        filtered = [p for p in all_traj_paths if is_target_subagent(p)]
        return sorted(filtered, key=call_number)

    # Otherwise, include main agent first, then any subagents by call number
    def sort_key(traj_path: Path) -> int:
        if traj_path.parent == iteration_dir:
            return 0
        dir_name = traj_path.parent.name
        if dir_name.startswith("subagent_"):
            try:
                return int(dir_name.rsplit("_", 1)[-1])
            except Exception:
                return 1000
        return 1000

    return sorted(all_traj_paths, key=sort_key)