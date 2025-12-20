# Suppress Pydantic serialization warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from pathlib import Path
from typing import Dict, Any, List
import random
import openai
import yaml
from pydantic import BaseModel, Field

from tools import Tool
from instance_sampling import sample_single_instance
from trajectory_processor import format_trajectories
from sweagent.run.batch_instances import AbstractInstanceSource
from experiment import Experiment
from utils import extract_yaml_from_response, get_traj_paths


class WarmupConfig(BaseModel):
    """Configuration for the warmup engine."""
    warmup_iterations: int = Field(default=3, description="Number of warmup iterations per tool")
    output_dir: Path = Field(description="Base output directory for warmup artifacts")
    prompt_dir: Path = Field(description="Directory containing prompt files")
    template_dir: Path = Field(description="Directory containing agent templates")
    model_name: str = Field(description="LLM model name for refinement")
    num_workers: int = Field(default=1, description="Number of workers for warmup runs")


class WarmupEngine:
    """Engine for warming up newly generated subagents."""
    
    def __init__(self, client: openai.Client, config: WarmupConfig):
        self.client = client
        self.config = config
        self.rng = random.Random()
        
        # Load config fields into self for convenience
        self.warmup_iterations = config.warmup_iterations
        self.output_dir = config.output_dir
        self.prompt_dir = config.prompt_dir
        self.template_dir = config.template_dir
        self.model_name = config.model_name
        self.num_workers = config.num_workers
        
    def run(self, tool: Tool, instances_config: AbstractInstanceSource, base_agent_template: Path, base_subagent_template: Path) -> Tool:
        """Run warmup iterations for a tool and return the refined version."""
        tool_dir = tool.bundle_dir
        warmup_dir = self.output_dir / "warmup" / tool.name
        warmup_dir.mkdir(parents=True, exist_ok=True)
        
        # Run warmup iterations
        for iteration in range(1, self.warmup_iterations + 1):
            iteration_dir = warmup_dir / f"iteration_{iteration:03d}"
            iteration_dir.mkdir(exist_ok=True)
            
            # Select random instance
            sample_instance = sample_single_instance(instances_config, self.rng)

            # Use Experiment to setup agent/subagent and run batch for this single instance
            exp = Experiment(
                evolution_output_dir=iteration_dir,
                exp_num=1,
                chosen_tools=[tool],
                instances=sample_instance,
                template_dir=self.template_dir,
            )
            exp.run_swe_agent()
            traj_paths = get_traj_paths(iteration_dir)
            
            # Extract meta inputs for LLM analysis
            trajectories_summary = format_trajectories(traj_paths)
            updates = self._refine_tool(tool, trajectories_summary)
            self._apply_updates(tool, updates, tool_dir, iteration_dir)
        
        return tool
        
    def _refine_tool(self, tool: Tool, trajectories_summary: str) -> Dict[str, Any]:
        """Use meta LLM to refine the tool. Returns dict with updates."""
        # Load the prompt template
        prompt_path = self.prompt_dir / "evolve_subagent_from_warmup.txt"
        prompt_template = prompt_path.read_text()
        
        # Format the prompt with meta inputs
        prompt = prompt_template + "\n\n===SUBAGENT TO IMPROVE===\n\n" + str(tool)
        prompt += "=== TRAJECTORY SUMMARIES START ===\n\n" + trajectories_summary
        
        try:
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )
            
            # Get the full response content
            full_response = response.choices[0].message.content
            
            # Parse the response using utility function
            updates = extract_yaml_from_response(full_response, expected_top_key='updates')
            
            # Add the full response to the updates dict
            if updates:
                updates['full_response'] = full_response
            
            return updates
        except Exception as e:
            print(f"Error refining tool: {e}\nNo edits made.")
            return {}
        
    def _apply_updates(self, tool: Tool, updates: Dict[str, Any], tool_dir: Path, log_dir: Path) -> None:
        """Apply updates to tool bundle and in-memory object."""
        if not updates or 'updates' not in updates:
            return
        
        tool_updates = updates['updates']
        
        # Create a simple log of the before state
        before_state = {
            'before': {
                'tool_name': tool.name,
                'docstring': tool.docstring,
                'instance_template': tool.instance_template
            }
        }
        
        # Update in-memory tool object
        if 'docstring' in tool_updates:
            before_state['before']['docstring'] = tool.docstring
            tool.docstring = tool_updates['docstring']
            
        if 'context_description' in tool_updates:
            # Find current context description
            current_desc = ""
            for arg in tool.arguments:
                if arg.get('name') == 'context':
                    current_desc = arg.get('description', '')
                    break
            
            before_state['before']['context_description'] = current_desc
            
            # Update the context argument description
            for arg in tool.arguments:
                if arg.get('name') == 'context':
                    arg['description'] = tool_updates['context_description']
                    break
            
        if 'instance_template' in tool_updates:
            before_state['before']['instance_template'] = tool.instance_template
            tool.instance_template = tool_updates['instance_template']
        
        # Use the Tool's write_to_file method to update the bundle
        tool.write_to_file()
        
        # Log the before state
        with open(log_dir / "subagent_before.yaml", 'w') as f:
            yaml.dump(before_state, f, default_flow_style=False, width=float("inf"))
        
        # Log the updates (original format)
        with open(log_dir / "updates.yaml", 'w') as f:
            yaml.dump(updates, f)
        