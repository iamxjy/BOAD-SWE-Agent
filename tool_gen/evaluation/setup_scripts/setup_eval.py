from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy
import argparse
import yaml

def load_agent_template(template_dir: str) -> dict[str, Any]:
    base = Path(template_dir)
    return yaml.safe_load((base / "agent.yaml").read_text())

def load_subagent_template(template_dir: str) -> dict[str, Any]:
    base = Path(template_dir)
    return yaml.safe_load((base / "subagent.yaml").read_text())


def build_subagent_entry(
    name: str,
    system_template: str,
    instance_template: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    entry = deepcopy(base)
    entry["name"] = name
    entry["type"] = "subagent"
    entry.setdefault("templates", {})["system_template"] = system_template
    entry["templates"]["instance_template"] = instance_template
    return entry


def generate_configs(test_dir: Path, template_dir: str) -> None:
    tools_dir = test_dir / "tools"

    agent_cfg = load_agent_template(template_dir)["agent"]
    base_subagent = load_subagent_template(template_dir)["agent"]["subagents"][0]

    subagents: list[dict[str, Any]] = []
    bundle_paths: list[str] = []
    for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
        tfile = tool_dir / "templates.yaml"
        if not tfile.exists():
            continue
        tdata = yaml.safe_load(tfile.read_text())
        subagents.append(
            build_subagent_entry(
                name=tool_dir.name,
                system_template=tdata.get("system_template", ""),
                instance_template=tdata.get("instance_template", ""),
                base=base_subagent,
            )
        )
        bundle_paths.append(str(tool_dir.resolve()))

    bundles = list(agent_cfg.get("tools", {}).get("bundles", []))
    existing = {b.get("path") for b in bundles if isinstance(b, dict)}
    for rel in bundle_paths:
        if rel not in existing:
            bundles.append({"path": rel})
    agent_cfg.setdefault("tools", {})["bundles"] = bundles

    (test_dir / "agent.yaml").write_text(yaml.dump({"agent": agent_cfg}, indent=2, sort_keys=False))
    (test_dir / "subagent.yaml").write_text(yaml.dump({"agent": {"subagents": subagents}}, indent=2, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--template_dir", type=str, default="tool_gen/templates_seed_oss_36b")
    args = parser.parse_args()
    generate_configs(Path("tool_gen/evaluation/eval_runs") / args.folder, args.template_dir)


if __name__ == "__main__":
    main()
