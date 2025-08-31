import re
import ast
import logging

# --- Cython Transformations ---

def move_nested_classes(source, target_class):
    if not target_class:
        return source
    class_pattern = rf'^\s*class\s+{target_class}\s*[:(]'
    lines = source.splitlines()
    new_lines, nested_classes = [], []
    inside, indent = False, None
    for line in lines:
        if re.match(class_pattern, line):
            inside, indent = True, len(line) - len(line.lstrip())
        if inside and re.match(r'^\s*class\s+\w+\s*[:(]', line):
            nested_classes.append(line.lstrip())
        else:
            new_lines.append(line)
    return "\n".join(new_lines + nested_classes)

def add_hot_function_annotations(source, hot_functions, target_class):
    lines = source.splitlines()
    new_lines = []
    inside, indent = False, None
    for line in lines:
        if target_class and re.match(rf'^\s*class\s+{target_class}\s*[:(]', line):
            inside, indent = True, len(line) - len(line.lstrip())
            line = line.replace("class", "cdef class", 1)
        if inside and len(line) - len(line.lstrip()) <= indent and line.strip():
            inside = False
        match = re.match(r'^(\s*)(def|async def)\s+(\w+)\s*\(', line)
        if match and match.group(3) in hot_functions:
            line = f"{match.group(1)}cpdef {line.strip()[4:]}"
        new_lines.append(line)
    return "\n".join(new_lines)

def ensure_groupentry_dataclass(source):
    if "GroupEntry" in source and "@dataclass" not in source:
        source = "from dataclasses import dataclass\n\n" + source
        source = re.sub(r"(class\s+GroupEntry\s*[:(])", r"@dataclass\n\1", source)
    return source

def apply_type_inference(source):
    logging.info("Applying type inference...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cpdef \1(\2) -> object:', source)

def convert_local_variables(source):
    logging.info("Converting local variables to cdef...")
    return re.sub(r'(\n\s*)(\w+)\s*=\s*(\d+)', r'\1cdef int \2 = \3', source)

def optimize_loops(source):
    logging.info("Optimizing loops...")
    return re.sub(r'for (\w+) in range\(', r'cdef int \1\nfor \1 in range(', source)

def convert_numpy_arrays(source):
    logging.info("Converting NumPy arrays to memoryviews...")
    return re.sub(r'np\.zeros\((.*?)\)', r'<double[:,:]> np.zeros(\1)', source)

def clean_decorators(source):
    logging.info("Cleaning unsupported decorators...")
    return re.sub(r'^\s*@staticmethod\s*\n', '', source, flags=re.MULTILINE)

def refine_exceptions(source):
    logging.info("Refining exception blocks...")
    return re.sub(r'except\s*:', 'except Exception:', source)

def inline_functions(source):
    logging.info("Inlining short functions...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cdef inline \1(\2):', source)

def apply_parallelization(source):
    logging.info("Applying parallelization with prange...")
    source = re.sub(r'for (\w+) in range\(', r'for \1 in prange(', source)
    if 'from cython.parallel import prange' not in source:
        source = 'from cython.parallel import prange\n' + source
    return source

def add_cython_imports(source):
    logging.info("Adding Cython-specific imports...")
    directives = (
        'from cython import boundscheck, wraparound\n'
        '@boundscheck(False)\n@wraparound(False)\n'
    )
    return directives + source

def add_profiling_hooks(source, hot_functions):
    logging.info("Adding profiling hooks to hot functions...")
    for func in hot_functions:
        pattern = rf'(def|cpdef)\s+{func}\s*\((.*?)\):'
        replacement = rf'\1 {func}(\2):\n    import time\n    start = time.time()'
        source = re.sub(pattern, replacement, source)
        source = re.sub(r'(return .*)', r'\1\n    print("Elapsed:", time.time() - start)', source)
    return source

# --- Static Analysis Tools ---

def score_cyclomatic_complexity(source):
    tree = ast.parse(source)
    scores = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            score = 1 + sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp)) for n in ast.walk(node))
            scores[node.name] = score
    print("\n🧠 Cyclomatic Complexity Scores:")
    for name, score in scores.items():
        print(f" - {name}: {score}")
    return source

def generate_call_graph(source):
    tree = ast.parse(source)
    graph = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and hasattr(n.func, 'id')]
            graph[node.name] = calls
    print("\n📈 Call Graph:")
    for func, calls in graph.items():
        print(f" - {func} calls: {calls}")
    return source

def detect_dead_code(source):
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and hasattr(n.func, 'id')}
    dead = defined - called
    print("\n🧹 Dead Code Detected:")
    for func in dead:
        print(f" - {func}")
    return source

def check_decorator_compatibility(source):
    print("🧪 Checking decorators for Cython compatibility...")
    incompatible = re.findall(r'^\s*@\w+', source, re.MULTILINE)
    for deco in incompatible:
        print(f" - Found decorator: {deco}")
    return source

# --- Benchmark Harness ---

def generate_benchmark_harness(hot_functions):
    print("\n🧪 Generating benchmark harness...")
    for func in hot_functions:
        print(f"def benchmark_{func}():\n    import time\n    start = time.time()\n    {func}()\n    print('Time:', time.time() - start)\n")

# --- HTML Report Generator ---

def generate_html_report(applied_steps, output_path, report_name=None):
    report_path = report_name if report_name else output_path.replace(".py", "_report.html")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transformation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #333;
        }}
        ul {{
            list-style-type: square;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
            color: #555;
        }}
        .footer {{
            margin-top: 40px;
            font-size: 0.9em;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>🧾 Transformation Report</h1>
    <p><strong>Output File:</strong> {output_path}</p>
    <ul>
"""
    for step in applied_steps:
        html += f"        <li>{step}</li>\n"

    html += """    </ul>
    <div class="footer">
        Generated by Copilot Transformation Tool
    </div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 HTML report saved to: {report_path}")
