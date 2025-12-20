#!/usr/bin/env python3
"""
Script to analyze token usage statistics by resolved/unresolved status.

Usage:
    python analyze_token_stats.py <folder_name>
    
Example:
    python analyze_token_stats.py test19_2_sub_seed
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


def load_token_data(folder_name: str) -> Dict:
    """Load token counts data from the specified folder."""
    token_file = Path(f"tool_gen/evaluation/eval_runs/{folder_name}/token_counts.json")
    if not token_file.exists():
        raise FileNotFoundError(f"Token counts file not found: {token_file}")
    
    with open(token_file, 'r') as f:
        return json.load(f)


def load_resolved_instances(folder_name: str) -> List[str]:
    """Load resolved instances from analysis summary."""
    analysis_file = Path(f"logs/run_evaluation/{folder_name}/analysis_summary.json")
    if not analysis_file.exists():
        raise FileNotFoundError(f"Analysis summary file not found: {analysis_file}")
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    # Extract instance names from resolved_instances list
    # Format: "django__django-16899 (django)" -> "django__django-16899"
    resolved_instances = []
    for instance in data.get('resolved_instances', []):
        # Remove the project name in parentheses
        instance_name = instance.split(' (')[0]
        resolved_instances.append(instance_name)
    
    return resolved_instances


def categorize_instances(token_data: Dict, resolved_instances: List[str]) -> Tuple[Dict, Dict]:
    """Categorize instances into resolved and unresolved."""
    resolved_data = {}
    unresolved_data = {}
    
    for instance_name, stats in token_data.items():
        if instance_name in resolved_instances:
            resolved_data[instance_name] = stats
        else:
            unresolved_data[instance_name] = stats
    
    return resolved_data, unresolved_data


def calculate_summary_stats(data: Dict, category_name: str) -> Dict:
    """Calculate summary statistics for a category of instances."""
    if not data:
        return {
            'category': category_name,
            'count': 0,
            'total_tokens_sent': 0,
            'total_tokens_received': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'avg_tokens_sent': 0,
            'avg_tokens_received': 0,
            'avg_total_tokens': 0,
            'avg_api_calls': 0,
            'median_tokens_sent': 0,
            'median_tokens_received': 0,
            'median_total_tokens': 0,
            'median_api_calls': 0
        }
    
    tokens_sent = [stats['tokens_sent'] for stats in data.values()]
    tokens_received = [stats['tokens_received'] for stats in data.values()]
    total_tokens = [stats['total_tokens'] for stats in data.values()]
    api_calls = [stats['api_calls'] for stats in data.values()]
    
    return {
        'category': category_name,
        'count': len(data),
        'total_tokens_sent': sum(tokens_sent),
        'total_tokens_received': sum(tokens_received),
        'total_tokens': sum(total_tokens),
        'total_api_calls': sum(api_calls),
        'avg_tokens_sent': statistics.mean(tokens_sent),
        'avg_tokens_received': statistics.mean(tokens_received),
        'avg_total_tokens': statistics.mean(total_tokens),
        'avg_api_calls': statistics.mean(api_calls),
        'median_tokens_sent': statistics.median(tokens_sent),
        'median_tokens_received': statistics.median(tokens_received),
        'median_total_tokens': statistics.median(total_tokens),
        'median_api_calls': statistics.median(api_calls)
    }


def print_summary_stats(stats: Dict):
    """Print formatted summary statistics."""
    print(f"\n=== {stats['category'].upper()} STATISTICS ===")
    print(f"Count: {stats['count']}")
    print(f"Total tokens sent: {stats['total_tokens_sent']:,}")
    print(f"Total tokens received: {stats['total_tokens_received']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Total API calls: {stats['total_api_calls']:,}")
    print(f"Average tokens sent: {stats['avg_tokens_sent']:,.1f}")
    print(f"Average tokens received: {stats['avg_tokens_received']:,.1f}")
    print(f"Average total tokens: {stats['avg_total_tokens']:,.1f}")
    print(f"Average API calls: {stats['avg_api_calls']:,.1f}")
    print(f"Median tokens sent: {stats['median_tokens_sent']:,.1f}")
    print(f"Median tokens received: {stats['median_tokens_received']:,.1f}")
    print(f"Median total tokens: {stats['median_total_tokens']:,.1f}")
    print(f"Median API calls: {stats['median_api_calls']:,.1f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_token_stats.py <folder_name>")
        print("Example: python analyze_token_stats.py test19_2_sub_seed")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    
    try:
        # Load data
        print(f"Loading data for folder: {folder_name}")
        token_data = load_token_data(folder_name)
        resolved_instances = load_resolved_instances(folder_name)
        
        print(f"Total instances in token data: {len(token_data)}")
        print(f"Resolved instances: {len(resolved_instances)}")
        
        # Categorize instances
        resolved_data, unresolved_data = categorize_instances(token_data, resolved_instances)
        
        print(f"Resolved instances found in token data: {len(resolved_data)}")
        print(f"Unresolved instances: {len(unresolved_data)}")
        
        # Calculate statistics
        resolved_stats = calculate_summary_stats(resolved_data, "Resolved")
        unresolved_stats = calculate_summary_stats(unresolved_data, "Unresolved")
        all_stats = calculate_summary_stats(token_data, "All")
        
        # Print results
        print_summary_stats(resolved_stats)
        print_summary_stats(unresolved_stats)
        print_summary_stats(all_stats)
        
        # Save results to JSON
        results = {
            'folder_name': folder_name,
            'resolved_stats': resolved_stats,
            'unresolved_stats': unresolved_stats,
            'all_stats': all_stats,
            'resolved_instances': list(resolved_data.keys()),
            'unresolved_instances': list(unresolved_data.keys())
        }
        
        output_file = Path(f"tool_gen/scripts_analyze_eval/{folder_name}_token_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
