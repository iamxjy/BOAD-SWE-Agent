# generate the main plan for the agent

import yaml
from pathlib import Path
import sys
import openai
import argparse
import dotenv
import os
dotenv.load_dotenv()

# Add the parent directory to the path to import main_agent_generator
sys.path.append(str(Path(__file__).parent.parent.parent))
from main_agent_generator import MainAgentGenerator

def generate_main_plan(test_dir: Path) -> None:
    client = openai.Client(
        base_url="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Load the agent config
    with open(test_dir / "agent.yaml", "r") as f:
        agent_cfg = yaml.safe_load(f)
    
    # Load the subagent config to get tools
    with open(test_dir / "subagent.yaml", "r") as f:
        subagent_cfg = yaml.safe_load(f)
    
    # Extract tools from subagent config - subagents are defined under agent.subagents
    tools = []
    subagents = subagent_cfg.get("agent", {}).get("subagents", [])
    print(f"Found {len(subagents)} subagents in subagent config")
    
    for subagent in subagents:
        subagent_name = subagent.get("name", "")
        if not subagent_name:
            continue
            
        from tools import Tool
        # Create bundle directory path for this tool
        bundle_dir = test_dir / "tools" / subagent_name
        
        # Extract docstring from the instance template
        instance_template = subagent.get("templates", {}).get("instance_template", "")
        docstring = f"Subagent: {subagent_name}"
        if "You are a helpful" in instance_template:
            # Extract the description from the instance template
            lines = instance_template.split('\n')
            for line in lines:
                if "You are a helpful" in line and "assistant" in line:
                    docstring = line.strip()
                    break
        
        tool = Tool(
            name=subagent_name,
            docstring=docstring,
            signature=f"{subagent_name}(...)",  # Generic signature
            arguments=[],  # Subagents don't have explicit arguments in this format
            bundle_dir=bundle_dir
        )
        tools.append(tool)
        print(f"Created tool: {subagent_name} - {docstring[:50]}...")
    
    # Create MainAgentGenerator with required parameters
    prompt_path = Path("./tool_gen/prompts/generate_main_instance_template.txt")
    main_agent_generator = MainAgentGenerator(
        client=client,
        prompt_path=prompt_path,
        model_name= "claude-sonnet-4-20250514",
        temperature=0.0,
        output_dir=test_dir
    )
    
    # Generate the plan using the MainAgentGenerator
    plan_output_file = test_dir / "generated_plan.txt"
    
    try:
        # Convert tools to commands for documentation
        commands = main_agent_generator._convert_tools_to_commands(tools)
        plan_text = main_agent_generator._generate_plan(tools)
        
        # Write the plan to a file
        with open(plan_output_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("GENERATED PLAN:\n")
            f.write("="*80 + "\n")
            f.write(plan_text + "\n")
            f.write("="*80 + "\n")
            f.write("\nCopy the plan above and paste it into the {{plan}} placeholder in agent.yaml\n")
        
        print(f"Plan generated and saved to: {plan_output_file}")
        
    except Exception as e:
        print(f"Failed to generate plan: {e}")
        print("Using fallback plan:")
        fallback_plan = main_agent_generator._get_fallback_plan()
        
        # Write the fallback plan to a file
        with open(plan_output_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("FALLBACK PLAN:\n")
            f.write("="*80 + "\n")
            f.write(fallback_plan + "\n")
            f.write("="*80 + "\n")
        
        print(f"Fallback plan saved to: {plan_output_file}")

    return plan_text
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()
    generate_main_plan(Path("./tool_gen/evaluation/eval_runs") / args.folder)