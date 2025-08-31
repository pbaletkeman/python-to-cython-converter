import ast
import logging

def extract_defined_functions_ast(source):
    tree = ast.parse(source)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

def load_hot_functions(file_path):
    hot_funcs = set()
    try:
        with open(file_path, "r") as f:
            for line in f:
                func = line.strip()
                if func:
                    hot_funcs.add(func)
    except Exception as e:
        logging.error(f"Error reading hot functions file: {e}")
    return hot_funcs

def find_hot_candidates(source):
    tree = ast.parse(source)
    hot_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body_len = len(node.body)
            has_loop = any(isinstance(stmt, (ast.For, ast.While)) for stmt in node.body)
            if body_len > 10 or has_loop:
                hot_funcs.add(node.name)
    return hot_funcs

def validate_hot_functions(hot_funcs, defined_funcs, verbose=False):
    valid = hot_funcs & defined_funcs
    invalid = hot_funcs - defined_funcs
    for func in sorted(invalid):
        logging.warning(f"Function '{func}' not found in source.")
    if verbose:
        logging.info(f"Valid hot functions: {sorted(valid)}")
    return valid

def get_hot_functions(source, hot_file, use_hybrid=True, verbose=False):
    defined = extract_defined_functions_ast(source)
    manual_hot = load_hot_functions(hot_file) if hot_file else set()
    auto_hot = find_hot_candidates(source)
    combined = manual_hot | auto_hot if use_hybrid else manual_hot or auto_hot
    return validate_hot_functions(combined, defined, verbose)
