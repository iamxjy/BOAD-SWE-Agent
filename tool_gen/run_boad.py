import argparse
import yaml
from pathlib import Path
from tool_evolution_engine import EvolutionEngineConfig, EvolutionEngine
from sweagent.run.batch_instances import InstancesFromFile

def load_config(config_name: str) -> EvolutionEngineConfig:
    """Load configuration from YAML file and create EvolutionEngineConfig."""
    config_path = Path(f"tool_gen/config/{config_name}.yaml")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Convert string paths to Path objects
    for key in ['prompt_dir', 'template_dir', 'output_dir']:
        if key in config_data:
            config_data[key] = Path(config_data[key])
    
    # Handle instances_config
    if 'instances_config' in config_data:
        instances_data = config_data['instances_config']
        if instances_data['type'] == 'InstancesFromFile':
            config_data['instances_config'] = InstancesFromFile(
                path=Path(instances_data['path']),
                slice=instances_data['slice'],
                shuffle=instances_data['shuffle']
            )
    
    return EvolutionEngineConfig(**config_data)

def main():
    parser = argparse.ArgumentParser(description="Run tool evolution with different model configurations")
    parser.add_argument("--config", type=str, required=True,)
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    print(f"Running tool evolution with {args.config} configuration...")
    engine = EvolutionEngine(config)
    engine.run()

if __name__ == "__main__":
    main()