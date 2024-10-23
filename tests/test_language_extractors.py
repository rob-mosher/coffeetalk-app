import pytest
from src.languages.python import extract_code_snippets

def test_python_extractor():
    sample_code = '''
def hello_world():
    print("Hello")

class TestClass:
    def method(self):
        pass
'''
    snippets = extract_code_snippets(sample_code)
    assert len(snippets) == 2
    assert 'def hello_world' in snippets[0]
    assert 'class TestClass' in snippets[1]
