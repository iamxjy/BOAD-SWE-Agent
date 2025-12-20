#!/usr/bin/env python3
"""
Trajectory processing utilities for SWE-agent analysis.
Extracts and formats conversation data from trajectory files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from tools import Tool

MAX_WORDS = 180
MAX_TURNS = 60

def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + f" [truncated - {len(words)} words total]"


def process_trajectory(trajectory_file: Path, max_turns: int = MAX_TURNS) -> Dict[str, Any]:
    """Process trajectory file and extract conversation data."""
    with open(trajectory_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversation = []
    history = data.get('history', [])
    if len(history) > max_turns:
        history = history[:max_turns]
    
    for message in history:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'assistant':
            if "<function=" in content:
                start = content.find("<function=")
                if start != -1:
                    end = content.find("</function>", start)
                    if end != -1:
                        content = content[start:end + len("</function>")]
                    else:
                        content = content[start:]
            conversation.append({
                'role': role,
                'content': content,
                'message_type': message.get('message_type', ''),
                'agent': message.get('agent', 'main')
            })
        elif role == 'user':
            truncated_content = truncate_text(content, MAX_WORDS)
            conversation.append({
                'role': role,
                'content': truncated_content,
                'message_type': message.get('message_type', ''),
                'agent': message.get('agent', 'main'),
                'original_length': len(content.split())
            })
        elif role == 'system':
            conversation.append({
                'role': role,
                'content': content,
                'message_type': message.get('message_type', ''),
                'agent': message.get('agent', 'main')
            })
    
    info = data.get('info', {})
    submission = info.get('submission', '')
    original_history_length = len(data.get('history', []))
    truncated = original_history_length > max_turns
    
    return {
        'conversation': conversation,
        'submission': submission,
        'total_messages': len(conversation),
        'original_total_messages': original_history_length,
        'truncated': truncated,
        'max_turns_limit': max_turns
    }


def format_conversation(processed_data: Dict[str, Any]) -> str:
    """Format the processed conversation as a string."""
    lines = []
    
    if processed_data.get('truncated', False):
        original_total = processed_data.get('original_total_messages', 0)
        max_turns = processed_data.get('max_turns_limit', MAX_TURNS)
        lines.append(f"[TRAJECTORY TRUNCATED: Showing first {max_turns} of {original_total} total messages]")
        lines.append("=" * 80)
    
    for i, message in enumerate(processed_data['conversation'], 1):
        role = message['role'].lower()
        content = message['content']
        
        if role == 'assistant':
            lines.append(f"AGENT: {content}")
        elif role == 'user':
            lines.append(f"{content}")
            if message.get('original_length', 0) > MAX_WORDS:
                lines.append(f"[Note: Original message was {message['original_length']} words, truncated to {MAX_WORDS}]")
        elif role == 'system':
            lines.append(f"SYSTEM: {content}")
        
        lines.append("")
    
    if processed_data.get('submission'):
        lines.append("SUBMISSION:")
        lines.append(processed_data['submission'])
    
    return "\n".join(lines) 

def format_trajectory(trajectory_file: Path) -> str:
    """Format the trajectory file as a string."""
    return format_conversation(process_trajectory(trajectory_file))

# format trajectories from the same instance with the first trajectory being the main agent's trajectory
# takes a list of trajectory file paths
def format_trajectories(trajectory_files: List[Path]) -> str:
    """Format multiple trajectory files as a single string."""
    formatted_parts = []
    for i, traj_file in enumerate(trajectory_files):
        if i==0:
            formatted_parts.append(f"=== MAIN AGENT TRAJECTORY: {traj_file.name} ===")
        else:
            formatted_parts.append(f"=== SUBAGENT TRAJECTORY: {traj_file.name} ===")
        formatted_parts.append(format_trajectory(traj_file))
    
    return "\n".join(formatted_parts)
    return "\n".join([format_conversation(process_trajectory(traj_file)) for traj_file in trajectory_files])