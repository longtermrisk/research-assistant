import os
from typing import List, Optional
from dataclasses import dataclass
import subprocess
import json
from pathlib import Path
from collections import defaultdict
import re

@dataclass
class CodeDiagnostic:
    """Represents a diagnostic message from static code analysis"""
    line: int
    column: Optional[int]
    severity: str  # 'error', 'warning', 'info'
    message: str
    source: str  # e.g., 'pylint', 'eslint', 'typescript'
    code: Optional[str] = None  # e.g., 'TS2305' or 'E0401'

def analyze_python(file_path: str) -> List[CodeDiagnostic]:
    """Analyze Python code using pylint"""
    diagnostics = []
    
    try:
        # Run pylint with JSON output format
        result = subprocess.run(
            ['pylint', '--output-format=json', file_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout:
            results = json.loads(result.stdout)
            for result in results:
                # Convert pylint symbols to codes (e.g., 'missing-module' -> 'E0401')
                message_id = result.get('message-id', '')
                symbol = result.get('symbol', '')
                code = result.get('type', '').upper()[0] + str(result.get('message-id', ''))
                
                diagnostics.append(CodeDiagnostic(
                    line=result['line'],
                    column=result.get('column'),
                    severity=result['type'],
                    message=f"{result['message']} ({symbol})",
                    source='pylint',
                    code=code
                ))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='error',
            message='pylint not found. Install with: pip install pylint',
            source='pylint'
        )]
    
    return diagnostics

def find_project_root(file_path: str) -> Optional[Path]:
    """Find the root of the project by looking for package.json or tsconfig.json"""
    current_dir = Path(file_path).parent
    while current_dir != current_dir.parent:
        if (current_dir / 'package.json').exists() or (current_dir / 'tsconfig.json').exists():
            return current_dir
        current_dir = current_dir.parent
    return None

def analyze_javascript(file_path: str) -> List[CodeDiagnostic]:
    """Analyze JavaScript code using eslint"""
    diagnostics = []
    
    # Find project root (reusing existing function)
    project_root = find_project_root(file_path)
    if not project_root:
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='error',
            message='Could not find project root (no package.json found)',
            source='eslint'
        )]
    
    try:
        # Run eslint with JSON output format
        result = subprocess.run(
            ['eslint', '--format=json', file_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout:
            results = json.loads(result.stdout)
            for file_result in results:
                for message in file_result.get('messages', []):
                    severity_map = {
                        2: 'error',
                        1: 'warning',
                        0: 'info'
                    }
                    
                    diagnostics.append(CodeDiagnostic(
                        line=message.get('line', 1),
                        column=message.get('column'),
                        severity=severity_map.get(message.get('severity', 2), 'error'),
                        message=message.get('message', ''),
                        source='eslint',
                        code=message.get('ruleId')
                    ))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='error',
            message='eslint not found. Install with: npm install -g eslint',
            source='eslint'
        )]
    
    return diagnostics

def analyze_typescript(file_path: str) -> List[CodeDiagnostic]:
    """Analyze TypeScript code using tsc"""
    diagnostics = []
    
    # Find project root
    project_root = find_project_root(file_path)
    if not project_root:
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='error',
            message='Could not find project root (no package.json or tsconfig.json found)',
            source='typescript'
        )]
    
    # Get relative path from project root
    rel_path = Path(file_path).relative_to(project_root)
    
    # Run tsc from project root
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        result = subprocess.run(
            ['tsc', str(rel_path), '--noEmit', '--jsx', 'react'],
            capture_output=True,
            text=True
        )
        
        # Parse errors
        target_file = Path(file_path).name
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if not line.strip() or target_file not in line:
                continue
                
            try:
                # Parse location
                loc_start = line.find('(')
                loc_end = line.find(')')
                if loc_start != -1 and loc_end != -1:
                    location = line[loc_start + 1:loc_end]
                    line_num, col = map(int, location.split(','))
                    
                    # Get the error message (everything after the location)
                    error_part = line[loc_end + 2:].strip()
                    
                    # Extract error code if present (e.g., TS2305)
                    code_match = re.search(r'error TS(\d+)', error_part)
                    error_code = f"TS{code_match.group(1)}" if code_match else None
                    
                    # Get the actual error message (everything after the error code)
                    if code_match:
                        error_msg = error_part[code_match.end():].strip(': ')
                    else:
                        error_msg = error_part
                    
                    # Filter out common setup errors
                    if "Cannot use JSX unless the '--jsx' flag is provided" in error_msg:
                        continue
                        
                    diagnostics.append(CodeDiagnostic(
                        line=line_num,
                        column=col,
                        severity='error',
                        message=error_msg,
                        source='typescript',
                        code=error_code
                    ))
            except (ValueError, IndexError):
                continue
                
    finally:
        os.chdir(original_cwd)
    
    return diagnostics

def analyze_code(file_path: str) -> List[CodeDiagnostic]:
    """
    Analyze code and return diagnostics based on the file type.
    Supports Python, TypeScript, and JavaScript files.
    """
    if not os.path.exists(file_path):
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='error',
            message=f'File not found: {file_path}',
            source='analyzer'
        )]
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.py':
        return analyze_python(file_path)
    elif file_extension in ['.ts', '.tsx']:
        return analyze_typescript(file_path)
    elif file_extension in ['.js', '.jsx']:
        return analyze_javascript(file_path)
    else:
        return [CodeDiagnostic(
            line=1,
            column=1,
            severity='info',
            message=f'No analyzer available for files with extension: {file_extension}',
            source='analyzer'
        )]

def format_diagnostics(diagnostics: List[CodeDiagnostic]) -> List[str]:
    """Format diagnostics by grouping similar messages together"""
    # Group diagnostics by message and code
    message_groups = defaultdict(list)
    for diag in diagnostics:
        # Create a key that combines message and code
        key = (diag.message, diag.code)
        message_groups[key].append(diag.line)
    
    # Format each group
    formatted = []
    for (message, code), lines in message_groups.items():
        lines.sort()
        line_str = ", ".join(str(line) for line in lines)
        
        # Format code based on the source
        if code and code.startswith('TS'):
            code_str = f".ts({code})"
        elif code:  # pylint codes
            code_str = f".py({code})"
        else:
            code_str = ""
            
        formatted.append(f"Line {line_str}: {message}{code_str}")
    
    return formatted


def analyze_and_format(file_path: str) -> str:
    """Analyze code and format the diagnostics"""
    diagnostics = analyze_code(file_path)
    formatted_diagnostics = format_diagnostics(diagnostics)
    output = '\n'.join(formatted_diagnostics)
    if not output:
        output = "No issues found!"
    return output


if __name__ == "__main__":
    # Example usage
    import sys
    
    file_path = sys.argv[1]
    
    print(f"Analyzing {file_path}...")
    diagnostics = analyze_code(file_path)
    formatted_diagnostics = format_diagnostics(diagnostics)
    if formatted_diagnostics:
        for msg in formatted_diagnostics:
            print(msg)
    else:
        print("No issues found!")