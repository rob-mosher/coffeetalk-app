import re


def extract_code_snippets(content):
    """
    Extract functions, classes, interfaces, and enums from TypeScript code.
    """
    pattern = r'(export\s+)?(async\s+)?(function|class|interface|enum)\s+\w+\s*(<.*?>)?\s*(\(.*?\))?\s*{[^}]*}'
    matches = re.findall(pattern, content, re.DOTALL)

    snippets = []
    for match in matches:
        snippet = ''.join(match)
        snippets.append(snippet)
    return snippets
