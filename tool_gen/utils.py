"""Utility functions for working with LLM responses."""


from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json, yaml
from pydantic import BaseModel

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


class TrajectoryStep(BaseModel):
    """A single step in the agent trajectory."""
    action: str
    observation: str
    response: str
    state: Dict[str, str]
    thought: str
    execution_time: float
    query: List[Dict[str, Any]]
    extra_info: Dict[str, Any]


class HistoryItem(BaseModel):
    """A single item in the conversation history."""
    role: str
    content: Union[str, List[Dict[str, Any]]]
    message_type: str
    agent: Optional[str] = None
    is_demo: Optional[bool] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    tool_calls: Optional[List[Dict[str, str]]] = None
    tool_call_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    cache_control: Optional[Dict[str, Any]] = None


class ModelStats(BaseModel):
    """Model usage statistics."""
    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0


class AgentInfo(BaseModel):
    """Information about the agent run."""
    # Version information
    swe_agent_hash: Optional[str] = None
    swe_agent_version: Optional[str] = None
    swe_rex_version: Optional[str] = None
    swe_rex_hash: Optional[str] = None
    
    # Execution information
    exit_status: Optional[str] = None
    submission: Optional[str] = None
    
    # File editing information with different context lengths
    edited_files30: Optional[str] = None
    edited_files50: Optional[str] = None
    edited_files70: Optional[str] = None
    
    # Model usage statistics
    model_stats: Optional[ModelStats] = None
    
    # Optional fields that may be present
    review: Optional[Dict[str, Any]] = None
    summarizer: Optional[Dict[str, Any]] = None
    tool_call_counts: Optional[Dict[str, int]] = None


class Trajectory(BaseModel):
    """
    A complete SWE-agent trajectory containing all information from a .traj file.
    
    The structure matches the JSON format saved by SWE-agent:
    {
        "trajectory": [...],  # List of execution steps
        "history": [...],     # Conversation history
        "info": {...},        # Agent execution metadata
        "replay_config": str, # Configuration for replay (JSON string)
        "environment": str    # Environment/instance identifier
    }
    """
    
    # Core trajectory data
    trajectory: List[TrajectoryStep]
    history: Optional[List[HistoryItem]] = None
    info: Optional[AgentInfo] = None
    
    # Additional metadata
    replay_config: Optional[str] = None  # JSON string of replay configuration
    environment: Optional[str] = None    # Environment/instance identifier

    @classmethod
    def from_filepath(cls, filepath: Path) -> "Trajectory":
        """Load a trajectory from a .traj file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Convert trajectory steps
        trajectory_steps = []
        for step in data.get("trajectory", []):
            trajectory_steps.append(TrajectoryStep(**step))
        
        # Convert history items
        history_items = None
        if "history" in data and data["history"]:
            history_items = []
            for item in data["history"]:
                history_items.append(HistoryItem(**item))
        
        # Convert info
        agent_info = None
        if "info" in data and data["info"]:
            info_dict = data["info"].copy()
            # Handle model_stats separately
            if "model_stats" in info_dict and info_dict["model_stats"]:
                info_dict["model_stats"] = ModelStats(**info_dict["model_stats"])
            agent_info = AgentInfo(**info_dict)
        
        return cls(
            trajectory=trajectory_steps,
            history=history_items,
            info=agent_info,
            replay_config=data.get("replay_config"),
            environment=data.get("environment")
        )
    
    def get_n_steps(self) -> int:
        """Get the number of steps in the trajectory."""
        return len(self.trajectory)
    
    def get_total_tokens(self) -> int:
        """Get total tokens used (sent + received)."""
        if self.info and self.info.model_stats:
            return self.info.model_stats.tokens_sent + self.info.model_stats.tokens_received
        return 0
    
    def get_final_submission(self) -> Optional[str]:
        """Get the final submission from the agent."""
        if self.info:
            return self.info.submission
        return None
    
    def get_exit_status(self) -> Optional[str]:
        """Get the exit status of the agent run."""
        if self.info:
            return self.info.exit_status
        return None
    
    def was_submitted(self) -> bool:
        """Check if the agent run was successful (has a submission)."""
        submission = self.get_final_submission()
        return submission is not None and submission.strip() != ""

    def get_prompt(self) -> str:
        """Get the prompt used for the agent run."""
        return self.history[0].content