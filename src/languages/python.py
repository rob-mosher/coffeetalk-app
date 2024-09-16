import re


def extract_code_snippets(content):
    """
    Extract functions and classes from Python code.
    """
    pattern = r'(def\s+\w+\s*\(.*?\)\s*:\s*[^#]*?(?:\n\s+.*)*?)\n(?=\n|def|class|$)|' \
              r'(class\s+\w+\s*(\(.*?\))?\s*:\s*[^#]*?(?:\n\s+.*)*?)\n(?=\n|def|class|$)'
    matches = re.findall(pattern, content, re.MULTILINE)

    # Flatten the list of tuples and filter out empty strings
    snippets = [item for sublist in matches for item in sublist if item]
    return snippets
