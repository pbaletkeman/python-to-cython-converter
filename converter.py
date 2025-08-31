import ast
import re
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Python file to Cython-compatible .pyx")
    parser.add_argument("input", help="Path to the input Python file")
    parser.add_argument("output", help="Path to the output .pyx file")
    parser.add_argument("--hot-functions-file", help="Path to file listing hot functions (one per line)")
    parser.add_argument("--target-class", help="Class to convert to cdef class")
    parser.add_argument("--config", default="converter_config.json", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--infer-types", action="store_true")
    parser.add_argument("--convert-vars", action="store_true")
    parser.add_argument("--optimize-loops", action="store_true")
    parser.add_argument("--convert-numpy", action="store_true")
    parser.add_argument("--clean-decorators", action="store_true")
    parser.add_argument("--refine-exceptions", action="store_true")
    parser.add_argument("--inline-functions", action="store_true")
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--add-cython-imports", action="store_true")
    parser.add_argument("--add-profiling", action="store_true")
    return parser.parse_args()

# --- Config Loading ---
def load_config(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load config file: {e}")
        return {}

# --- Hot Functions ---
def load_hot_functions(file_path: str) -> set:
    hot_funcs = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                func = line.strip()
                if func:
                    hot_funcs.add(func)
        if not hot_funcs:
            logging.warning("Hot functions file is empty or contains no valid entries.")
    except Exception as e:
        logging.error(f"Error reading hot functions file: {e}")
    return hot_funcs

def extract_defined_functions_ast(source: str) -> set:
    try:
        tree = ast.parse(source)
        return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    except SyntaxError as e:
        logging.error(f"Syntax error while parsing source: {e}")
        return set()

def validate_hot_functions(hot_funcs: set, defined_funcs: set, verbose=False) -> set:
    valid = hot_funcs & defined_funcs
    invalid = hot_funcs - defined_funcs
    for func in sorted(invalid):
        logging.warning(f"Function '{func}' not found in source code. Skipping annotation.")
    if verbose:
        logging.info(f"Valid hot functions: {sorted(valid)}")
    return valid

def find_hot_candidates(source: str) -> set:
    logging.info("No hot functions file provided. Using static heuristics to detect hot functions...")
    tree = ast.parse(source)
    hot_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body_len = len(node.body)
            has_loop = any(isinstance(stmt, (ast.For, ast.While)) for stmt in node.body)
            uses_numpy = any(
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and
                isinstance(stmt.value.func, ast.Attribute) and
                getattr(stmt.value.func.value, "id", "") == "np"
                for stmt in node.body if isinstance(stmt, ast.Expr)
            )
            if body_len > 10 or has_loop or uses_numpy:
                hot_funcs.add(node.name)
    if not hot_funcs:
        logging.warning("No hot functions detected using heuristics.")
    else:
        logging.info(f"Detected hot functions: {sorted(hot_funcs)}")
    return hot_funcs

# --- Transformations ---
def move_nested_classes(source: str, target_class: str) -> str:
    if not target_class:
        return source
    class_pattern = rf'^\s*class\s+{re.escape(target_class)}\s*[:(]'
    lines = source.splitlines()
    new_lines, nested_classes = [], []
    inside_target_class, indent_level = False, None
    for line in lines:
        if re.match(class_pattern, line):
            inside_target_class = True
            indent_level = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue
        if inside_target_class:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                inside_target_class = False
        if inside_target_class and re.match(r'^\s*class\s+\w+\s*[:(]', line):
            nested_classes.append(line.lstrip())
        else:
            new_lines.append(line)
    return "\n".join(new_lines + nested_classes)

def add_hot_function_annotations(source: str, hot_functions: set, target_class: str) -> str:
    lines = source.splitlines()
    new_lines = []
    inside_target_class, indent_level = False, None
    for line in lines:
        if target_class and re.match(rf'^\s*class\s+{re.escape(target_class)}\s*[:(]', line):
            inside_target_class = True
            indent_level = len(line) - len(line.lstrip())
            line = line.replace("class", "cdef class", 1)
        if inside_target_class:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                inside_target_class = False
        func_match = re.match(r'^(\s*)(def|async def)\s+(\w+)\s*\(', line)
        if func_match:
            indent, _, func_name = func_match.groups()
            if func_name in hot_functions:
                line = f"{indent}cpdef {line.strip()[4:]}"
        new_lines.append(line)
    return "\n".join(new_lines)

def ensure_groupentry_dataclass(source: str) -> str:
    if "GroupEntry" in source and "@dataclass" not in source:
        source = "from dataclasses import dataclass\n\n" + source
        source = re.sub(r"(class\s+GroupEntry\s*[:(])", r"@dataclass\n\1", source)
    return source

def apply_type_inference(source: str) -> str:
    logging.info("Applying type inference...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cpdef \1(\2) -> object:', source)

def convert_local_variables(source: str) -> str:
    logging.info("Converting local variables to cdef...")
    return re.sub(r'(\n\s*)(\w+)\s*=\s*(\d+)', r'\1cdef int \2 = \3', source)

def optimize_loops(source: str) -> str:
    logging.info("Optimizing loops...")
    return re.sub(r'for (\w+) in range\(', r'cdef int \1\nfor \1 in range(', source)

def convert_numpy_arrays(source: str) -> str:
    logging.info("Converting NumPy arrays to memoryviews...")
    return re.sub(r'np\.zeros\((.*?)\)', r'<double[:,:]> np.zeros(\1)', source)

def clean_decorators(source: str) -> str:
    logging.info("Cleaning unsupported decorators...")
    return re.sub(r'^\s*@staticmethod\s*\n', '', source, flags=re.MULTILINE)

def refine_exceptions(source: str) -> str:
    logging.info("Refining exception blocks...")
    return re.sub(r'except\s*:', 'except Exception:', source)

def inline_functions(source: str) -> str:
    logging.info("Inlining short functions...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cdef inline \1(\2):', source)

def apply_parallelization(source: str) -> str:
    logging.info("Applying parallelization with prange...")
    source = re.sub(r'for (\w+) in range\(', r'for \1 in prange(', source)
    if 'from cython.parallel import prange' not in source:
        source = 'from cython.parallel import prange\n' + source
    return source

def add_cython_imports(source: str) -> str:
    logging.info("Adding Cython-specific imports...")
    directives = (
        'from cython import boundscheck, wraparound\n'
        '@boundscheck(False)\n@wraparound(False)\n'
    )
    return directives + source

def add_profiling_hooks(source: str, hot_functions: set) -> str:
    logging.info("Adding profiling hooks to hot functions...")
    for func in hot_functions:
        pattern = rf'(def|cpdef)\s+{func}\s*\((.*?)\):'
        replacement = rf'\1 {func}(\2):\n    import time\n    start = time.time()'
        source