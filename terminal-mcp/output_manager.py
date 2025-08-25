import os
import uuid
import tiktoken
from datetime import datetime
from typing import Tuple


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def truncate_output(
    output: str, 
    max_tokens: int = 10000,
    head_tokens: int = 5000,
    tail_tokens: int = 5000
) -> Tuple[str, str, bool]:
    """
    Truncate output if it exceeds max_tokens.
    
    Args:
        output: The output text to potentially truncate
        max_tokens: Maximum tokens allowed before truncating
        head_tokens: Tokens to keep from the beginning
        tail_tokens: Tokens to keep from the end
        
    Returns:
        Tuple of (truncated_output, full_output_path_or_empty_string, was_truncated)
    """
    token_count = count_tokens(output)
    
    if token_count <= max_tokens:
        return output, "", False
    
    # Create tool_output directory if it doesn't exist
    output_dir = "./tool_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
    full_output_path = os.path.join(output_dir, filename)
    
    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            f.write(output)
    except Exception as e:
        # If we can't save the file, return the original output
        return output, "", False
    
    # Truncate the output
    lines = output.split('\n')
    
    # Get head portion
    head_lines = []
    head_token_count = 0
    for line in lines:
        line_tokens = count_tokens(line + '\n')
        if head_token_count + line_tokens > head_tokens:
            break
        head_lines.append(line)
        head_token_count += line_tokens
    
    # Get tail portion  
    tail_lines = []
    tail_token_count = 0
    for line in reversed(lines):
        line_tokens = count_tokens(line + '\n')
        if tail_token_count + line_tokens > tail_tokens:
            break
        tail_lines.insert(0, line)
        tail_token_count += line_tokens
    
    # Combine head + middle indicator + tail
    head_text = '\n'.join(head_lines)
    tail_text = '\n'.join(tail_lines)
    
    # Ensure we don't duplicate content if head and tail overlap
    head_line_count = len(head_lines)
    total_lines = len(lines)
    tail_start = total_lines - len(tail_lines)
    
    if head_line_count >= tail_start:
        # Overlap detected, just use head portion with different truncation message
        truncated = head_text + f"\n\n[... Output truncated after {head_line_count} lines due to length. Full output saved to {full_output_path} ...]"
    else:
        # No overlap, show head + tail with gap indicator
        middle_lines_count = tail_start - head_line_count
        truncated = (
            head_text + 
            f"\n\n[... {middle_lines_count} lines omitted due to length. Full output saved to {full_output_path} ...]\n\n" + 
            tail_text
        )
    
    return truncated, full_output_path, True