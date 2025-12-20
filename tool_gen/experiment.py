from pathlib import Path
from typing import List, Optional, Dict, Any
import copy
import json
import os
import subprocess
import time
import yaml
from sweagent.utils.serialization import merge_nested_dicts
from sweagent.run.batch_instances import AbstractInstanceSource
from sweagent.run.run_batch import RunBatchConfig
from sweagent.run.run_batch import run_from_config
from sweagent.agent.agents import DefaultAgentConfig
from tools import Tool
from experiment_result import ExperimentResult
 

class Experiment:
    """Represents a single experiment with tools and configuration."""
    def __init__(
        self,
        evolution_output_dir: Path,
        exp_num: int,
        chosen_tools: List[Tool],
        instances: AbstractInstanceSource,
        template_dir: Path,
        designed_agent_config: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_dir = evolution_output_dir / "experiments" / f"exp_{exp_num:03d}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.instances = instances
        self.chosen_tools = chosen_tools
        self.evolution_output_dir = evolution_output_dir
        self.template_dir = template_dir
        self.agent_config_path = self.experiment_dir / "agent.yaml"
        self.subagent_config_path = self.experiment_dir / "subagent.yaml"
        self.trajectory_dir = evolution_output_dir / "logs/trajectories" / self.experiment_dir.name
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_meta_path = self.experiment_dir / "experiment.json"
        self.designed_agent_config = designed_agent_config
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Generate and write out subagent configs and metadata."""
        if self.designed_agent_config is not None:
            final_agent_cfg = self.designed_agent_config
        else:
            merged_cfg = yaml.safe_load((self.template_dir / "agent.yaml").read_text())
            existing_bundles = merged_cfg["agent"]["tools"].get("bundles", [])
            subagent_bundles = [tool.bundle_entry() for tool in self.chosen_tools]
            merged_cfg["agent"]["tools"]["bundles"] = subagent_bundles + list(existing_bundles)
            final_agent_cfg = merged_cfg

        (self.experiment_dir / "agent.yaml").write_text(yaml.dump(final_agent_cfg, indent=2, sort_keys=False))
        
        subagent_template = yaml.safe_load((self.template_dir / "subagent.yaml").read_text())["agent"]["subagents"][0]
        subagents = []
        for tool in self.chosen_tools:
            if not tool.subagent:
                continue
            subagent_config = copy.deepcopy(subagent_template)
            subagent_config["name"] = tool.name
            subagent_config["templates"]["system_template"] = tool.system_template
            subagent_config["templates"]["instance_template"] = tool.instance_template
            subagents.append(subagent_config)
        (self.experiment_dir / "subagent.yaml").write_text(yaml.dump({"agent": {"subagents": subagents}}, indent=2, sort_keys=False))
    

    def _extract_cost(self) -> float:
        """Extract total cost from run_batch_exit_statuses.yaml."""
        cost_file = self.trajectory_dir / "run_batch_exit_statuses.yaml"
        if cost_file.exists():
            try:
                with open(cost_file) as f:
                    data = yaml.safe_load(f)
                return data.get("total_cost", 0.0)
            except Exception as e:
                print(f"⚠️ Failed to extract cost from {cost_file}: {e}")
        return 0.0

    def run_swe_agent(self) -> None:
        # Load and merge agent configs
        try:
            with open(self.agent_config_path) as f:
                config_data = yaml.safe_load(f)
            with open(self.subagent_config_path) as f:
                subagent_data = yaml.safe_load(f)
            config_data = merge_nested_dicts(config_data, subagent_data)
            agent_config = DefaultAgentConfig.model_validate(config_data["agent"])
            
            config = RunBatchConfig(
                instances=self.instances,
                agent=agent_config,
                output_dir=self.trajectory_dir,
                num_workers=16,
                redo_existing=True,
            )
            # Run the batch
            run_from_config(config)
        except Exception as e:
            raise Exception(f"Failed to run SWE-agent for {self.experiment_dir.name}: {e}")
    
    def evaluate_patches(self) -> ExperimentResult:
        """Evaluate the patches using SWE-Bench evaluation harness."""
        self.run_id = f"{self.experiment_dir.name}_{int(time.time()*1000)}"
        print(f"Evaluating {self.experiment_dir.name}")

        # Convert predictions to a list of dicts
        predictions_path = self.trajectory_dir / "preds.json"
        if not predictions_path.exists():
            print(f"No predictions found for {self.experiment_dir.name} at {predictions_path}")
            return ExperimentResult(
                experiment_dir=self.experiment_dir.name,
                config_path=self.experiment_dir,
                total_cost=self._extract_cost())
        with open(predictions_path) as f:
            predictions = json.load(f)
        predictions_list = list(predictions.values())
        predictions_list_path = self.trajectory_dir / "preds_list.json"
        with open(predictions_list_path, "w") as f:
            json.dump(predictions_list, f, indent=2)

        # Change to tool_gen/generated directory so logs go to the right place
        original_cwd = Path.cwd()
        try:
            # Use the evolution output directory directly
            os.chdir(self.evolution_output_dir)
            
            # Update predictions_path to be relative to tool_gen/generated directory
            relative_predictions_path = str(Path("logs/trajectories") / self.experiment_dir.name / "preds_list.json")
            
            print(f"Running evaluation with predictions_path: {relative_predictions_path}")
            result = subprocess.run([
                "python", "-m", "swebench.harness.run_evaluation",
                "--predictions_path", relative_predictions_path,
                "--cache_level", "instance",
                # "--dataset", "SWE-Gym/SWE-Gym",
                "--dataset", "princeton-nlp/SWE-bench_Verified",
                "--split", "test",
                "--run_id", self.run_id,
                "--max_workers", "16",
            ], capture_output=True, text=True)

        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            return ExperimentResult(
                experiment_dir=self.experiment_dir.name,
                p2p_success=0,
                p2p_failure=1,
                f2p_success=0,
                f2p_failure=1,
                resolved=0,
                unresolved=0,
                config_path=self.experiment_dir,
                total_cost=self._extract_cost()
            )
        finally:
            os.chdir(original_cwd)
        
        print("Ran evaluation... aggregating results")

        # Always try to aggregate results, even if the evaluation command failed
        run_eval_dir = self.evolution_output_dir / "logs/run_evaluation" / self.run_id
        run_eval_dir.mkdir(parents=True, exist_ok=True)
        
        p2p_success = 0
        p2p_failure = 0
        f2p_success = 0
        f2p_failure = 0
        resolved = 0
        unresolved = 0

        # Check if any report files exist
        found_reports = False
        
        # Count empty patches from predictions
        empty_patch_count = 0
        for prediction in predictions_list:
            if prediction.get("model_patch", None) in ["", None]:
                empty_patch_count += 1
                unresolved += 1  # Empty patches count as unresolved
                p2p_failure += 1
                f2p_failure += 1
        
        for subdir in run_eval_dir.iterdir():
            report_path = subdir / "report.json"
            if report_path.exists():
                found_reports = True
                try:
                    with open(report_path) as f:
                        report_data = json.load(f)
                    
                    # Iterate through all instance results in the report
                    for instance_id, instance_data in report_data.items():
                        tests_status = instance_data.get("tests_status", {})

                        # Aggregate resolved results
                        is_resolved = instance_data.get("resolved", False)
                        resolved += int(is_resolved)
                        unresolved += int(not is_resolved)
                        
                        # Aggregate FAIL_TO_PASS results
                        fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
                        all_f2p_success = (len(fail_to_pass.get("failure", [])) == 0 and len(fail_to_pass.get("success", [])) > 0)
                        f2p_success += int(all_f2p_success)
                        f2p_failure += int(not all_f2p_success)

                        # Aggregate PASS_TO_PASS results
                        pass_to_pass = tests_status.get("PASS_TO_PASS", {})
                        all_p2p_success = (len(pass_to_pass.get("failure", []))  == 0 and len(pass_to_pass.get("success", [])) > 0)
                        p2p_success += int(all_p2p_success)
                        p2p_failure += int(not all_p2p_success)
                except Exception as e:
                    print(f"⚠️ Failed to parse report {report_path}: {e}")
                    f2p_failure += 1
                    p2p_failure += 1
                    unresolved += 1

        if not found_reports:
            print(f"⚠️ No evaluation reports found for {self.run_id}")
            if result.returncode != 0:
                print(f"Evaluation command failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            return ExperimentResult(
                experiment_dir=self.experiment_dir.name,
                unresolved=1,
                config_path=self.experiment_dir,
                total_cost=self._extract_cost()
            )

        p2p_rate = p2p_success / (p2p_success + p2p_failure) if (p2p_success + p2p_failure) > 0 else 0.0
        f2p_rate = f2p_success / (f2p_success + f2p_failure) if (f2p_success + f2p_failure) > 0 else 0.0
        resolved_rate = resolved / (resolved + unresolved) if (resolved + unresolved) > 0 else 0.0

        print(f"✅ Eval pass to pass test result: {p2p_success}/{p2p_success + p2p_failure} tests ({p2p_rate:.1%})")
        print(f"✅ Eval fail to pass test result: {f2p_success}/{f2p_success + f2p_failure} tests ({f2p_rate:.1%})")
        print(f"✅ Eval resolved test result: {resolved}/{resolved + unresolved} tests ({resolved_rate:.1%})")

        return ExperimentResult(
            experiment_dir=self.experiment_dir.name,
            p2p_success=p2p_success,
            p2p_failure=p2p_failure,
            f2p_success=f2p_success,
            f2p_failure=f2p_failure,
            resolved=resolved,
            unresolved=unresolved,
            config_path=self.experiment_dir,
            total_cost=self._extract_cost()
        )
    
    def _save_metadata(self, result: ExperimentResult) -> None:
        """Save experiment metadata after successful completion."""
        chosen_tools_meta = [
            {
                "name": t.name,
                "docstring": t.docstring,
                "arguments": t.arguments,
                "subagent": t.subagent,
                "n": t.n,
                "successes": t.successes,
                "bundle_dir": str(t.bundle_dir),
            }
            for t in self.chosen_tools
        ]
        instance_ids = [
            getattr(inst.problem_statement, "id", None) 
            for inst in self.instances.get_instance_configs()
        ]
        instance_ids = [i for i in instance_ids if isinstance(i, str)]
        
        self.experiment_meta_path.write_text(json.dumps({
            "chosen_tools": chosen_tools_meta,
            "instance_ids": instance_ids,
        }, indent=2))

    def run(self) -> ExperimentResult:
        """Run the experiment using run_batch and return results."""
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.run_swe_agent()
            result = self.evaluate_patches()
            # Only save metadata after successful completion
            self._save_metadata(result)
            return result
        except Exception as e:
            print(f"Failed to run experiment {self.experiment_dir.name}: {e}")
            return ExperimentResult(
                experiment_dir=self.experiment_dir.name,
                p2p_success=0,
                p2p_failure=0,
                f2p_success=0,
                f2p_failure=0,
                resolved=0,
                unresolved=0,
                config_path=self.experiment_dir,
                total_cost=self._extract_cost()
            )
