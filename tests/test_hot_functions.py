import pytest
import tempfile
import os
from hot_functions import (
    extract_defined_functions_ast,
    load_hot_functions,
    find_hot_candidates,
    validate_hot_functions,
    get_hot_functions
)

def test_extract_defined_functions_ast():
    source = '''
def foo(): pass
def bar(): pass
class Baz:
    def qux(self): pass
'''
    result = extract_defined_functions_ast(source)
    assert result == {'foo', 'bar', 'qux'}

def test_load_hot_functions(tmp_path):
    file_path = tmp_path / "hot.txt"
    file_path.write_text("foo\nbar\n\nbaz\n")
    result = load_hot_functions(str(file_path))
    assert result == {'foo', 'bar', 'baz'}

def test_find_hot_candidates_detects_loops():
    source = '''
def short(): return 1
def long(): 
    for i in range(10): 
        print(i)
def verbose(): 
    x = 0
    y = 1
    z = 2
    a = 3
    b = 4
    c = 5
    d = 6
    e = 7
    f = 8
    g = 9
    h = 10
    return x + y
'''
    result = find_hot_candidates(source)
    assert 'long' in result
    assert 'verbose' in result
    assert 'short' not in result

def test_validate_hot_functions_filters_invalid():
    hot = {'foo', 'bar', 'baz'}
    defined = {'foo', 'baz'}
    result = validate_hot_functions(hot, defined)
    assert result == {'foo', 'baz'}

def test_get_hot_functions_combines_manual_and_auto(tmp_path):
    source = '''
def fast(): pass
def slow(): 
    for i in range(10): print(i)
'''
    file_path = tmp_path / "hot.txt"
    file_path.write_text("fast\nmissing\n")
    result = get_hot_functions(source, str(file_path), use_hybrid=True)
    assert 'fast' in result
    assert 'slow' in result
    assert 'missing' not in result

def test_extract_defined_functions_ast_with_syntax_error():
    source = '''
def good(): pass
def broken(  # missing closing parenthesis
    print("oops")
'''
    result = extract_defined_functions_ast(source)
    assert result == {'good'}  # should skip broken function silently

def test_empty_source():
    source = ''
    assert extract_defined_functions_ast(source) == set()
    assert find_hot_candidates(source) == set()

def test_decorated_functions():
    source = '''
@log
def foo(): pass

@profile
def bar(): pass
'''
    result = extract_defined_functions_ast(source)
    assert result == {'foo', 'bar'}

def test_empty_hot_function_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    result = load_hot_functions(str(file_path))
    assert result == set()

def test_hot_function_file_with_comments(tmp_path):
    file_path = tmp_path / "hot.txt"
    file_path.write_text("""
# This is a comment
foo

# Another comment
bar

""")
    result = load_hot_functions(str(file_path))
    assert result == {'foo', 'bar'}

def test_logging_for_extraction(caplog):
    source = '''
def foo(): pass
def bar(): pass
'''
    with caplog.at_level("INFO"):
        result = extract_defined_functions_ast(source)
    assert "Extracted function names" in caplog.text or "Extracted" in caplog.text
    assert result == {'foo', 'bar'}

def test_load_hot_functions_with_invalid_encoding(tmp_path):
    file_path = tmp_path / "bad_encoding.txt"
    file_path.write_bytes(b"\xff\xfe\xfd")  # invalid UTF-8

    try:
        result = load_hot_functions(str(file_path))
    except UnicodeDecodeError:
        result = set()

    assert result == set()

def test_hybrid_mode_conflict(tmp_path):
    source = '''
def fast(): pass
def slow(): 
    for i in range(10): print(i)
def noisy(): 
    for i in range(100): print("noise")
'''

    file_path = tmp_path / "hot.txt"
    file_path.write_text("fast\nnoisy\nmissing\n")

    result = get_hot_functions(source, str(file_path), use_hybrid=True)

    # 'missing' should be excluded, 'slow' should be auto-detected
    assert 'fast' in result
    assert 'noisy' in result
    assert 'slow' in result
    assert 'missing' not in result
