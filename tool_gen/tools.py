from dataclasses import dataclass, field
from typing import List, Dict, Any
import random
import numpy as np
from pathlib import Path
import yaml
import json

@dataclass
class Tool:
    name: str
    signature: str
    docstring: str
    arguments: List[Dict[str, Any]]
    bundle_dir: Path
    subagent: bool = True

    system_template : str = ""
    instance_template : str = ""
    code_dict: Dict[str, str] = field(default_factory=dict) # keys: yaml, code, install_script
    
    # -- bandit stats --
    n: int = 0
    successes: int = 0
    helpful_count: int = 0  # count of times this tool actually helped to make progress in the problem

    exp_num: int = 0
    total_token_count: int = 0
    subagent_invoked_count: int = 0
    average_token_count: float = 0.0

    def mean_reward(self) -> float:
        if self.n == 0:
            return 0.0
        return self.helpful_count / self.n
        # return self.successes / self.n
    
    def ucb_score(self, step: int) -> float:
        if step == 0:
            return 1.0
        exploration_bonus = np.sqrt(2 * np.log(step) / (self.n))
        return self.mean_reward() + exploration_bonus
        # return self.mean_reward() + 2 * exploration_bonus
    
    def helpful_rate(self) -> float:
        """Get the rate of times this tool actually helped solve the problem."""
        if self.n == 0:
            return 0.0
        return self.helpful_count / self.n
    
    def get_average_token_count(self) -> float:
        """Get the average token count of the tool."""
        if self.subagent_invoked_count == 0:
            return 0.0
        return self.average_token_count

    # convenience for agent.yaml
    def bundle_entry(self) -> Dict[str, Any]:
        return {"path": str(self.bundle_dir)}
    
    # write the entire bundle to a file at self.bundle_dir
    def write_to_file(self):
        '''
        Constructs the bundle directory:
            bundle_dir/
                bin/<name>
                install.sh
                config.yaml
                templates.yaml
        '''

        # create bin directory
        bin_dir = self.bundle_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_file = bin_dir / self.name
        bin_file.touch()

        # create install.sh
        install_file = self.bundle_dir / "install.sh"
        install_file.touch()

        # write config.yaml
        cfg = {
            "tools": {
                self.name: {
                    "arguments": self.arguments,
                    "docstring": self.docstring,
                    "signature": self.signature,
                    "subagent": self.subagent
                }
            }
        }
        (self.bundle_dir / "config.yaml").write_text(yaml.dump(cfg, indent=2, default_flow_style=False, allow_unicode=True))

        # write subagent.yaml
        templates = {
            "system_template": self.system_template,
            "instance_template": self.instance_template
        }
        (self.bundle_dir / "templates.yaml").write_text(yaml.dump(templates, indent=2, default_flow_style=False, allow_unicode=True))
    
    def update_average_token_count(self, token_count: int):
        self.total_token_count += token_count
        self.subagent_invoked_count += 1
        # Don't increment n here - it's already incremented per instance in the evolution engine
        # Calculate average based on total tokens and number of invocations
        if self.subagent_invoked_count > 0:
            self.average_token_count = self.total_token_count / self.subagent_invoked_count
    
    def __str__(self) -> str:
        return f"Tool(name={self.name}, signature={self.signature}, docstring={self.docstring}, arguments={self.arguments}, subagent={self.subagent}, system_template={self.system_template}, instance_template={self.instance_template})"
    
class ToolArchive:
    def __init__(self, output_dir: Path = Path("tool_archive")):
        self.output_dir = output_dir
        self.tools : List[Tool] = []
        self.step = 0

        if self.output_dir.exists():
            self._load()
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
   
    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    # sample k tools based on UCB scores, only include tools that have been used at least once
    def sample(self, k: int, rng: random.Random = None) -> List[Tool]:
        usable = [t for t in self.tools if t.n > 0]
        count = min(k, len(usable))
        if count == 0: 
            return []
        # sort by ucb score
        usable.sort(key=lambda t: t.ucb_score(self.step), reverse=True)
        return usable[:count]

    def save(self, output_dir: Path = None, filename: str = "archive.json"):
        data = {"step": self.step, "tools": []}
        for t in self.tools:
            d = t.__dict__.copy()
            # make bundle_dir JSON-safe
            d["bundle_dir"] = str(t.bundle_dir)
            d["ucb_score"] = t.ucb_score(self.step)
            data["tools"].append(d)

        save_dir = output_dir if output_dir is not None else self.output_dir
        p = save_dir / filename
        p.write_text(json.dumps(data, indent=2))

    
    def _load(self):
        p = self.output_dir / "archive.json"
        if not p.exists():
            return

        loaded = json.loads(p.read_text())
        
        # Handle new format with step and tools fields
        if isinstance(loaded, dict) and "tools" in loaded:
            self.step = loaded.get("step", 0)
            tools_data = loaded.get("tools", [])
        # Handle old format (list of tools)
        elif isinstance(loaded, list):
            self.step = 0
            tools_data = loaded
        else:
            return
            
        self.tools = []
        for d in tools_data:
            # restore bundle_dir as a Path
            d["bundle_dir"] = Path(d["bundle_dir"])
            # drop derived fields not in Tool constructor
            d.pop("ucb_score", None)
            self.tools.append(Tool(**d))
    
    def __str__(self) -> str:
        return f"ToolArchive(tools={self.tools})"
    
    def __len__(self) -> int:
        return len(self.tools)
    