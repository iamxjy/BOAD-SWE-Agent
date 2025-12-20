# analyze the results of the evaluation

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
import argparse

def load_report(report_path: Path) -> Dict[str, Any]:
    """Load a single report.json file."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {report_path}: {e}")
        return {}

def analyze_reports(eval_dir: Path) -> Dict[str, Any]:
    """Analyze all reports in the evaluation directory."""
    results = {
        'total_instances': 0,
        'resolved_count': 0,
        'success_rate': 0.0,
        'by_project': defaultdict(lambda: {'total': 0, 'resolved': 0, 'success_rate': 0.0}),
        'patch_stats': {
            'patch_exists': 0,
            'patch_successfully_applied': 0,
            'patch_success_rate': 0.0
        },
        'resolved_instances': [],
        'unresolved_instances': []
    }
    
    # Find all report.json files
    report_files = list(eval_dir.rglob("report.json"))
    print(f"Found {len(report_files)} report files")
    
    for report_path in report_files:
        report_data = load_report(report_path)
        if not report_data:
            continue
            
        # Extract instance name from path
        instance_name = report_path.parent.name
        project = instance_name.split('__')[0] if '__' in instance_name else 'unknown'
        
        # Process each instance in the report
        for instance_id, instance_data in report_data.items():
            results['total_instances'] += 1
            results['by_project'][project]['total'] += 1
            
            # Track resolution status
            if instance_data.get('resolved', False):
                results['resolved_count'] += 1
                results['by_project'][project]['resolved'] += 1
                results['resolved_instances'].append(f"{instance_id} ({project})")
            else:
                results['unresolved_instances'].append(f"{instance_id} ({project})")
            
            # Track patch statistics
            if instance_data.get('patch_exists', False):
                results['patch_stats']['patch_exists'] += 1
            if instance_data.get('patch_successfully_applied', False):
                results['patch_stats']['patch_successfully_applied'] += 1
    
    # Calculate success rates
    if results['total_instances'] > 0:
        results['success_rate'] = results['resolved_count'] / results['total_instances']
        results['patch_stats']['patch_success_rate'] = results['patch_stats']['patch_successfully_applied'] / results['patch_stats']['patch_exists'] if results['patch_stats']['patch_exists'] > 0 else 0.0
    
    # Calculate project-level success rates
    for project, stats in results['by_project'].items():
        if stats['total'] > 0:
            stats['success_rate'] = stats['resolved'] / stats['total']
    
    return results

def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of the analysis."""
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Instances: {results['total_instances']}")
    print(f"  Resolved Instances: {results['resolved_count']}")
    print(f"  Overall Success Rate: {results['success_rate']:.2%}")
    
    print(f"\nPatch Statistics:")
    print(f"  Patches Generated: {results['patch_stats']['patch_exists']}")
    print(f"  Patches Successfully Applied: {results['patch_stats']['patch_successfully_applied']}")
    print(f"  Patch Success Rate: {results['patch_stats']['patch_success_rate']:.2%}")
    
    print(f"\nSuccess Rates by Project:")
    # Sort projects by success rate (descending)
    sorted_projects = sorted(
        results['by_project'].items(), 
        key=lambda x: x[1]['success_rate'], 
        reverse=True
    )
    
    for project, stats in sorted_projects:
        print(f"  {project}: {stats['resolved']}/{stats['total']} ({stats['success_rate']:.2%})")
    
    print(f"\nInstance Lists:")
    print(f"  Resolved Instances ({len(results['resolved_instances'])}):")
    for instance in results['resolved_instances']:
        print(f"    {instance}")
    
    print(f"\n  Unresolved Instances ({len(results['unresolved_instances'])}):")
    for instance in results['unresolved_instances']:
        print(f"    {instance}")

def save_detailed_results(results: Dict[str, Any], output_path: Path):
    """Save detailed results to a JSON file."""
    # Convert defaultdict to regular dict for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, defaultdict):
            serializable_results[key] = dict(value)
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Evaluation subfolder under logs/run_evaluation")
    args = parser.parse_args()
    eval_dir = Path("logs/run_evaluation") / args.folder
    
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        print("Please run this script from the SWE-agent root directory")
        exit(1)
    
    print(f"Analyzing evaluation results from: {eval_dir}")
    results = analyze_reports(eval_dir)
    
    # Print summary
    print_summary(results)
    
    # Save detailed results
    output_path = eval_dir / "analysis_summary.json"
    save_detailed_results(results, output_path)
    