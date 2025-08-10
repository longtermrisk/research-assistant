import re
from bs4 import BeautifulSoup


def simplify_html(
        html, drop_elements=['svg', 'style'], shorten_elements=['script', 'li', 'ul', 'ol'],
        attributes=['style', 'class', 'source', 'href']):
    """Shorten the HTML:
    - for listed elements, keep at most 10 children and shorten text to 100 characters
    - for listed attributes, keep at most 10 characters
    - replaces dropped elements with minified placeholders (e.g. <svg>...</svg>)
    - replaces repeated elements with a count (e.g. <style>...</style> (x19))
    Uses BeautifulSoup to handle potentially invalid HTML.
    """
    # Parse with 'html.parser' which is more forgiving than 'lxml'
    soup = BeautifulSoup(html, 'html.parser')
    
    # Processalements
    for tag_name in drop_elements:
        elements = soup.find_all(tag_name)
        
        # Group consecutive elements
        groups = []
        current_group = []
        
        for i, element in enumerate(elements):
            if not current_group:
                current_group.append(element)
            else:
                # Check if current element is next sibling of last element in group
                last_elem = current_group[-1]
                if element.previous_sibling == last_elem:
                    current_group.append(element) 
                else:
                    groups.append(current_group)
                    current_group = [element]
                    
        if current_group:
            groups.append(current_group)
            
        # Process each group
        for group in groups:
            if len(group) > 1:
                # Keep first occurrence with count for consecutive elements
                first = group[0]
                placeholder = soup.new_tag(tag_name)
                placeholder.string = "..."
                first.replace_with(placeholder)
                # Remove the rest in group
                for element in group[1:]:
                    element.decompose()
                # Add count after first element
                count_text = soup.new_string(f" (x{len(group)})")
                placeholder.insert_after(count_text)
            else:
                # Single element - just replace with placeholder
                element = group[0]
                placeholder = soup.new_tag(tag_name)
                placeholder.string = "..."
                element.replace_with(placeholder)

    for tag_name in shorten_elements:
        for element in soup.find_all(tag_name):
            # Keep only first 10 children
            children = element.findChildren(recursive=False)
            for child in children[10:]:
                child.decompose()
            
            # Truncate text
            if element.string:
                element.string = element.string[:100]
    
    # Process attributes
    for element in soup.find_all():
        for attr in list(element.attrs):
            if attr in attributes:
                element[attr] = element[attr][:10] if isinstance(element[attr], str) else element[attr]
    
    return soup.prettify()


def clean_ansi(text):
    # Step 1: Remove ANSI escape sequences
    ansi_escape = re.compile(r'(?:\x1B[@-Z\\-_]|\x1B\[.*?[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Step 2: Split into lines and process terminal control characters
    lines = text.splitlines()
    screen = []
    current_line = ""
    
    for line in lines:
        # Handle carriage return (simulate line overwrites)
        if '\r' in line:
            parts = line.split('\r')
            # Process each part
            for part in parts:
                # Handle backspaces in this part
                while '\x08' in part:
                    part = re.sub(r'.\x08', '', part, 1)
                
                # Overwrite current line from the start
                current_line = part
        else:
            # Handle backspaces
            while '\x08' in line:
                line = re.sub(r'.\x08', '', line, 1)
            current_line = line
        
        # Only add non-empty lines that aren't just progress indicators
        if current_line.strip() and not is_progress_line(current_line):
            screen.append(current_line)
    
    # Remove duplicate consecutive lines
    unique_lines = []
    prev_line = None
    for line in screen:
        if line != prev_line:
            unique_lines.append(line)
            prev_line = line
    
    # If we have 0 lines because all where progress lines, return the last one
    if not unique_lines:
        return current_line
    
    # Join lines and clean up any remaining control characters
    cleaned_output = '\n'.join(unique_lines) + '\n'
    return cleaned_output


def is_progress_line(line):
    """
    Check if a line is likely a progress indicator that should be skipped
    """
    # Common patterns in progress lines
    progress_patterns = [
        r'⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏',  # Spinner characters
        r'\[#*\s*\]',  # Progress bars
        r'idealTree:.*',  # npm specific progress
        r'reify:.*',  # npm specific progress
        r'\([^)]*\)\s*�[⠀-⣿]',  # Progress with spinner
    ]
    
    return any(re.search(pattern, line) for pattern in progress_patterns)

