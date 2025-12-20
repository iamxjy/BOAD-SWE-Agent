# Suppress Pydantic serialization warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import yaml
from pydantic import BaseModel, Field, ConfigDict

import shutil
from datetime import datetime
from subagent_generator import SubagentGenerator
from main_agent_generator import MainAgentGenerator
from sweagent.run.batch_instances import AbstractInstanceSource, CustomSingleInstanceSource, CustomBatchInstanceSource
from instance_sampling import sample_single_instance, sample_batch_instances
from sweagent.utils.config import load_environment_variables
from sweagent.tool_generation_chaos.core.tool_generator import ToolGenerator
from experiment_result import ExperimentResult
from experiment import Experiment
from tools import Tool, ToolArchive
from warmup import WarmupEngine, WarmupConfig
from bilevel.trajectory import Trajectory
from trajectory_processor import format_trajectories
from utils import get_traj_paths, extract_yaml_from_response

class EvolutionEngineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Configuration for the tool evolution engine."""
    # LLM configuration
    model_name: str = Field(default="deepseek-v3", description="LLM model for tool generation")
    api_base: str = Field(default="http://localhost:1004/v1", description="API base URL")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    use_rits: bool = Field(default=False, description="Whether to use RITS API format")
    
    # Experiment configuration
    max_iterations: int = Field(default=20, description="Maximum number of iterations")
    subagent_tool_count: int = Field(default=3, description="Number of subagent tools to sample per agent")
    code_tool_count: int = Field(default=3, description="Number of code tools to sample per agent")
    new_tool_theta: float = Field(default = 1.0, description="CRP theta parameter for new tool creation probability")
    
    # Warmup configuration
    warmup_iterations: int = Field(default=2, description="Number of warmup iterations per new subagent")

    # SWE-bench instances configuration (optional)
    instances_config: AbstractInstanceSource = Field(description="Instances configuration")
    batch_size: int = Field(default=10, description="Batch size for instance sampling")
    
    # Paths
    prompt_dir: Path = Field(default=Path("tool_gen/prompts"), description="Directory containing prompt files")
    prompt_filename: str = Field(default="generate_subagent_parts_prompt.txt", description="Prompt filename")
    template_dir: Path = Field(default=Path("tool_gen/templates"), description="Directory containing template files")
    swe_agent_config: Path = Field(default=Path("config/default.yaml"), description="Base SWE-agent config")
    output_dir: Path = Field(default=Path("tool_gen/generated"), description="Output directory")
    
    resume: bool = Field(default=False, description="Whether to resume from checkpoint")


class EvolutionEngine:
    """Main engine for iterative tool evolution."""
    
    def __init__(self, config: EvolutionEngineConfig):
        # Load environment variables first
        load_environment_variables()
        
        self.model_name = config.model_name
        if self.model_name.startswith("openai/"):
            self.openai_model_name = self.model_name.replace("openai/", "", 1)
        elif self.model_name.startswith("together_ai/"):
            self.openai_model_name = self.model_name.replace("together_ai/", "", 1)
        else:
            self.openai_model_name = self.model_name
        self.api_base = config.api_base
        self.use_rits = config.use_rits
        self.resume = config.resume

        # Get API key from environment variable (can be None for local models)
        if config.use_rits:
            self.api_key = os.getenv("RITS_API_KEY")
        elif config.model_name.startswith("claude"):
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize client based on API type
        if config.use_rits:
            self.client = openai.Client(
                base_url=config.api_base,
                api_key=self.api_key,
                default_headers={"RITS_API_KEY": self.api_key}
            )
        elif config.model_name.startswith("together_ai"):
            from together import Together
            self.client = Together()
        else:
            self.client = openai.Client(
                base_url=config.api_base,
                api_key=self.api_key
            )
        self.temperature = config.temperature
        self.max_iterations = config.max_iterations
        self.output_dir = config.output_dir
        self.prompt_dir = config.prompt_dir
        self.prompt_filename = config.prompt_filename
        self.template_dir = config.template_dir
        self.instances_config = config.instances_config
        self.subagent_tool_count = config.subagent_tool_count
        self.code_tool_count = config.code_tool_count
        self.new_tool_theta = config.new_tool_theta  # CRP theta parameter for new tool creation probability
        self.warmup_iterations = config.warmup_iterations
        self.batch_size = config.batch_size

        # Clear the output directory at the start of the run (delete and recreate the directory)
        if not self.resume and self.output_dir.exists() and self.output_dir.is_dir():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the tool archive
        self.subagent_archive = ToolArchive(output_dir=self.output_dir / "subagent_tool_archive")
        self.code_tool_archive = ToolArchive(output_dir=self.output_dir / "code_tool_archive")

        # Initialize the subagent generator
        self.subagent_generator = SubagentGenerator(
            client=self.client,
            prompt_dir=self.prompt_dir,
            prompt_filename=self.prompt_filename,
            template_dir=self.template_dir,
            output_dir=self.output_dir,
            model_name=self.model_name,
            temperature=self.temperature
        )
        self.experiments = []

        # initialize the code tool generator
        self.code_tool_generator = ToolGenerator(
            client=self.client,
            prompt_path=self.prompt_dir / "generate_tool_code_prompt.txt",
            model=self.model_name,
            base_dir=self.output_dir / "code_tool_archive"
        )

        # initialize the warmup engine
        warmup_config = WarmupConfig(
            warmup_iterations=config.warmup_iterations,
            output_dir=self.output_dir,
            prompt_dir=self.prompt_dir,
            template_dir=self.template_dir,
            model_name=self.model_name,
            num_workers=1
        )
        self.warmup_engine = WarmupEngine(
            client=self.client,
            config=warmup_config,
        )

        # initialize the main agent generator
        self.main_agent_generator = MainAgentGenerator(
            client=self.client,
            prompt_path=self.prompt_dir / "generate_main_instance_template.txt",
            model_name=self.model_name,
            temperature=self.temperature,
            output_dir=self.output_dir,
        )

        # Seed the random number generator for reproducibility
        seed = int(time.time()) if not self.resume else 42
        self.rng = random.Random(seed)
        print(f"🔧 Random seed: {seed}")

    # for resume checkpointing: get the next experiment number to run
    # and clean up any incomplete experiments
    def _get_start_experiment_number(self) -> int:
        """Find the next experiment number to run, handling checkpointing."""
        if not self.resume:
            return 1
            
        # Find last completed experiment
        last_completed = 0
        experiments_dir = self.output_dir / "experiments"
        if experiments_dir.exists():
            for exp_dir in sorted(experiments_dir.glob("exp_*")):
                if (exp_dir / "experiment_result.json").exists():
                    exp_num = int(exp_dir.name.split("_")[1])
                    last_completed = max(last_completed, exp_num)

        # Clean up any incomplete experiments (delete them)
        incomplete_tools = set()
        if experiments_dir.exists():
            for exp_dir in sorted(experiments_dir.glob("exp_*")):
                exp_num = int(exp_dir.name.split("_")[1])
                if exp_num > last_completed:
                    print(f"Cleaning up incomplete experiment {exp_dir.name}")
                    # Collect tool names from incomplete experiment
                    meta_path = exp_dir / "experiment.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            for tool in meta.get("chosen_tools", []):
                                if isinstance(tool, dict) and "name" in tool:
                                    incomplete_tools.add(tool["name"])
                        except Exception:
                            pass
                    shutil.rmtree(exp_dir)
                    # Also clean up logs for this experiment
                    exp_name = exp_dir.name
                    traj_dir = self.output_dir / "logs" / "trajectories" / exp_name
                    if traj_dir.exists():
                        shutil.rmtree(traj_dir)
                    eval_dir = self.output_dir / "logs" / "run_evaluation"
                    if eval_dir.exists():
                        # Remove any evaluation runs for this experiment
                        for eval_run in eval_dir.glob(f"{exp_name}_*"):
                            if eval_run.is_dir():
                                shutil.rmtree(eval_run)

        # Clean up unused tools from archives (n=0)
        if incomplete_tools:
            print(f"Cleaning up unused tools: {incomplete_tools}")
            # Remove from subagent archive
            self.subagent_archive.tools = [
                t for t in self.subagent_archive.tools 
                if t.name not in incomplete_tools or t.n > 0
            ]
            # Remove from code tool archive  
            self.code_tool_archive.tools = [
                t for t in self.code_tool_archive.tools
                if t.name not in incomplete_tools or t.n > 0
            ]
            # Save cleaned archives
            self.subagent_archive.save()
            self.code_tool_archive.save()

        return last_completed + 1

    def choose_subagent_tools(self, exp_num: int) -> List[Tool]:
        """Sample <subagent_tool_count> tools from the archive using UCB sampling."""
        chosen: list[Tool] = []
        p_new = self.new_tool_theta / (self.new_tool_theta + len(self.subagent_archive))

        # Create subagents sequentially
        for _ in range(self.subagent_tool_count):
            rand_val = self.rng.random()
            if rand_val < p_new:
                print(f"Creating a new subagent... p_new: {rand_val} < {p_new}")
                try:
                    # import ipdb; ipdb.set_trace()
                    new_tool = self.subagent_generator.generate_new_tool(tool_archive=self.subagent_archive, exp_num=exp_num)
                    chosen.append(new_tool)
                    print(f"   Created new subagent: {new_tool.name}")
                    self.subagent_archive.add_tool(new_tool)
                except Exception as e:
                    print(f"   Failed to create tool: {e}")
                    # This slot will be filled from archive instead

        # Warmup newly created subagents in parallel
        if chosen and self.warmup_iterations > 0:
            print(f"🔥 Warming up {len(chosen)} subagents in parallel...")
            with ThreadPoolExecutor(max_workers=min(len(chosen), 16)) as executor:
                # Submit warmup tasks
                future_to_tool = {
                    executor.submit(self.warmup_engine.run, tool, self.instances_config, self.template_dir / "agent.yaml", self.template_dir / "subagent.yaml"): tool
                    for tool in chosen
                }
                for future in as_completed(future_to_tool):
                    tool = future_to_tool[future]
                    try:
                        warmed_tool = future.result()
                        print(f"   ✅ Warmup completed for: {warmed_tool.name}")
                    except Exception as e:
                        print(f"   Failed to warmup {tool.name}: {e}")
                        # Tool remains in archive but warmup failed

        # Sample remaining tools from archive
        remaining_count = self.subagent_tool_count - len(chosen)
        if remaining_count > 0:
            print(f"Sampling {remaining_count} tools from the archive")
            archive_tools = self.subagent_archive.sample(remaining_count, self.rng)
            chosen += archive_tools
        
        print(f"Final subagent tool selection: {[t.name for t in chosen]}")
        return chosen

    def _get_rand_trajectory(self) -> Optional[Path]:
        """Get a random trajectory JSON from logs/trajectories across experiments."""
        traj_root = self.output_dir / "logs" / "trajectories"
        exp_dirs = sorted(traj_root.glob("exp_*"))
        if not exp_dirs:
            return None
        latest = exp_dirs[-1]
        trajs = sorted(latest.rglob("*.traj"))
        return self.rng.choice(trajs) if trajs else None

    def _generate_new_code_tool(self, exp_num: int) -> Tool:
        """Generate a new code tool."""
        traj_path = self._get_rand_trajectory()
        if traj_path is None:
            return None
        new_tool_dict = self.code_tool_generator.generate_tool_from_trajectory(traj_path)
        if not new_tool_dict or not new_tool_dict.get("success"):
            return None
        tool_name = new_tool_dict.get("tool_name") or "diy_tool"
        bundle_dir = self.code_tool_archive.output_dir / "tools" / Path(new_tool_dict.get("tool_path", f"diy_{tool_name}"))
        code_dict = new_tool_dict.get("parsed_code") or {}
        cfg = yaml.safe_load((bundle_dir / "config.yaml").read_text()) or {}
        info = next(iter((cfg.get("tools") or {}).values()), {})
        return Tool(
            name=tool_name,
            signature=info.get("signature", ""),
            docstring=info.get("docstring", ""),
            arguments=info.get("arguments", []),
            bundle_dir=bundle_dir,
            code_dict=code_dict,
            subagent=False,
            exp_num=exp_num
        )
    
    def choose_code_tools(self, exp_num: int) -> List[Tool]:
        """Sample <code_tool_count> tools from the archive using UCB sampling."""
        chosen: list[Tool] = []
        p_new = self.new_tool_theta / (self.new_tool_theta + len(self.code_tool_archive))

        # If it's the first experiment (no trajectories) and no prior code tools, skip code tools entirely
        if self._get_rand_trajectory() is None and len(self.code_tool_archive) == 0:
            print("No trajectories and no prior code tools; skipping code tools for this experiment.")
            return []

        for _ in range(self.code_tool_count):
            rand_val = self.rng.random()
            print(f"rand_val: {rand_val}, p_new: {p_new}")
            if rand_val < p_new:
                print(f"Creating a new tool... p_new: {p_new}")
                # Create a new tool with retries
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        new_tool = self._generate_new_code_tool(exp_num)
                        if not new_tool:
                            continue
                        self.code_tool_archive.add_tool(new_tool)
                        chosen.append(new_tool)
                        break
                    except Exception as e:
                        print(f"Failed to generate new tool (attempt {attempt+1}/{max_attempts}): {e}")
                        if attempt == max_attempts - 1:
                            print("❌ Giving up on generating a new tool for this slot; will sample from archive instead.")
                            raise Exception("Failed to generate a new tool.")
                        else:
                            print(f"Retrying... (attempt {attempt+1}/{max_attempts})")

        print(f"Sampling {self.code_tool_count - len(chosen)} tools from the archive")
        chosen += self.code_tool_archive.sample(self.code_tool_count - len(chosen), self.rng)
        return chosen
        
    def choose_tools(self, exp_num: int) -> List[Tool]:
        """Choose subagent and code tools for the agent."""
        subagent_tools = self.choose_subagent_tools(exp_num)
        code_tools = self.choose_code_tools(exp_num)
        return subagent_tools + code_tools

    def get_rand_instance(self) -> AbstractInstanceSource:
        """Use SGD style sampling to get instance"""
        return sample_single_instance(self.instances_config, self.rng)
        # return self.instances_config # for all instances at each experiment

    def get_batch_instance(self) -> AbstractInstanceSource:
        """Get random mini batch sample of instances"""
        return sample_batch_instances(self.instances_config, self.rng, self.batch_size)
    
    def check_if_helpful(self, formatted_trajectories: str, tool: Tool) -> bool:
        """Check if the tool is helpful in the trajectory."""
        # load prompt from prompt_dir/check_if_subagent_helpful.txt
        prompt = (self.prompt_dir / "check_if_subagent_helpful.txt").read_text()
        prompt = prompt.replace("{{TRAJECTORIES}}", formatted_trajectories)
        prompt = prompt.replace("{{TOOL_NAME}}", tool.name)
        self.subagent_generator.log_llm_call(f"🤖 LLM Call - check_if_helpful")
        self.subagent_generator.log_llm_call(f"📝 Prompt:\n{prompt}", self.model_name)

        response = self.client.chat.completions.create(
            model=self.openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        self.subagent_generator.log_llm_call(f"🤖 Response:\n{response.choices[0].message.content}", self.model_name)
        content = extract_yaml_from_response(response.choices[0].message.content)
        return content.get("helpful", False)

    def update_helpful_counts(self, experiment_dir: Path, tools: List[Tool]) -> None:
        print(f"Updating helpful counts for {experiment_dir} with tools: {[tool.name for tool in tools]}")
        """Update helpful counts for each tool based on the trajectories."""
        # Trajectories are in: output_dir/logs/trajectories/exp_{001}/inst_name/
        trajectory_dir = Path(self.output_dir) / "logs/trajectories" / experiment_dir.name
        
        for instance_dir in trajectory_dir.iterdir():
            if instance_dir.is_dir():
                for tool in tools:
                    # Include exactly one main agent trajectory PLUS this tool's subagent trajectories (if any)
                    main_traj = sorted(instance_dir.glob("*.traj"))[:1]
                    subagent_trajs = get_traj_paths(instance_dir, tool.name)

                    # If the tool was never invoked or main trajectory missing, skip evaluation
                    if not main_traj or not subagent_trajs:
                        continue

                    instance_traj_paths = main_traj + subagent_trajs
                    try:
                        if self.check_if_helpful(format_trajectories(instance_traj_paths), tool):
                            tool.helpful_count += 1
                            print(f"  {tool.name} helped in {instance_dir.name}")
                    except Exception as e:
                        self.subagent_generator.log_llm_call(f"❌ LLM Call - check_if_helpful FAILED {e}")

    def update_token_counts(self, experiment_dir: Path, tools: List[Tool]) -> None:
        """Update token counts for each tool based on the trajectories."""
        print(f"Updating token counts for {experiment_dir} with tools: {[tool.name for tool in tools]}")
        # Trajectories are in: output_dir/logs/trajectories/exp_{001}/inst_name/
        trajectory_dir = Path(self.output_dir) / "logs/trajectories" / experiment_dir.name
        
        for instance_dir in trajectory_dir.iterdir():
            if instance_dir.is_dir():
                for tool in tools:
                    instance_traj_paths = get_traj_paths(instance_dir, tool.name)
                    # Process each trajectory file for this tool
                    for traj_path in instance_traj_paths:
                        if traj_path.exists():
                            traj = Trajectory.from_filepath(traj_path)
                            total_tokens = traj.get_total_tokens()
                            if total_tokens > 0:
                                tool.update_average_token_count(total_tokens)
                                print(f"  {tool.name} used {total_tokens} tokens in {instance_dir.name}")
      
    def run(self) -> None:
        """Run the complete evolution process."""
        # Find last completed experiment
        start_experiment_number = self._get_start_experiment_number()

        # Start from next experiment
        for exp_num in range(start_experiment_number, self.max_iterations + 1):
            print(f"Choosing tools for experiment {exp_num}:")
            chosen_tools = self.choose_tools(exp_num)
            
            # Build the full main agent config up-front (controlled experiment input)
            base_agent_config = yaml.safe_load((self.template_dir / "agent.yaml").read_text())
            designed_agent_config = self.main_agent_generator.generate_agent_config(base_agent_config, chosen_tools)

            experiment = Experiment(
                evolution_output_dir=self.output_dir,
                exp_num=exp_num,
                chosen_tools=chosen_tools,
                instances=self.get_batch_instance(),
                template_dir=self.template_dir,
                designed_agent_config=designed_agent_config,
            )
            
            # Save archive state snapshot to experiment directory
            self.subagent_archive.save(experiment.experiment_dir, "subagent_archive_snapshot.json")
            self.code_tool_archive.save(experiment.experiment_dir, "code_tool_archive_snapshot.json")
            print(f"Running experiment {exp_num} with chosen tools")
            result = experiment.run()

            print(f"Saving results and updating counts for {experiment.experiment_dir}")

            # Save result and update archives
            result_path = experiment.experiment_dir / "experiment_result.json"
            result_path.write_text(json.dumps(result.to_dict(), indent=2))

            # --- update bandit stats ---
            for tool in chosen_tools:
                tool.n += len(experiment.instances.get_instance_configs())
                tool.successes += result.resolved

            # --- analyze trajectories for helpful counts ---
            self.update_helpful_counts(experiment.experiment_dir, chosen_tools)

            # --- update token counts ---
            self.update_token_counts(experiment.experiment_dir, chosen_tools)

            # Increment step counter for UCB calculations
            self.subagent_archive.step += 1
            self.code_tool_archive.step += 1

            self.subagent_archive.save()
            self.code_tool_archive.save()
        
        # to-do: save overall results