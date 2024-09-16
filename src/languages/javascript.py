import re


def extract_code_snippets(content):
    """
    Extract functions and classes from JavaScript code.
    """
    pattern = r'(export\s+)?(async\s+)?(function|class)\s+\w+\s*\(.*?\)\s*{[^}]*}'
    matches = re.findall(pattern, content, re.DOTALL)

    snippets = []
    for match in matches:
        snippet = ''.join(match)
        snippets.append(snippet)
    return snippets
