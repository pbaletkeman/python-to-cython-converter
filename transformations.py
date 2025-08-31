import re
import ast
import logging

# --- Cython Transformations ---


def smart_format_v2(source: str) -> str:
    lines = source.splitlines()
    formatted = []
    INDENT = "    "
    indent_level = 0
    prev_type = None

    def classify(line):
        stripped = line.strip()
        if stripped == "":
            return "empty"
        if re.match(r'^(import|from)\s+', stripped):
            return "import"
        if re.match(r'^class\s+\w+', stripped):
            return "class"
        if re.match(r'^(cpdef|def)\s+\w+', stripped):
            return "function"
        if stripped.startswith("#"):
            return "comment"
        if re.match(r'^(for|while|if|elif|else|try|except|with)\b', stripped):
            return "control"
        if stripped.startswith("return"):
            return "return"
        return "code"

    for i, line in enumerate(lines):
        raw = line.rstrip().replace("\t", INDENT)
        stripped = raw.strip()
        line_type = classify(line)

        # Insert blank line before class/function/control if previous wasn't empty
        if line_type in {"class", "function"} and prev_type not in {"empty", "import", None}:
            if formatted and formatted[-1].strip() != "":
                formatted.append("")

        # Normalize indentation
        if line_type in {"class", "function"}:
            indent_level = 0
            formatted.append(stripped)
        elif line_type == "control":
            formatted.append(INDENT * indent_level + stripped)
            indent_level += 1
        elif line_type == "return":
            indent_level = max(indent_level - 1, 0)
            formatted.append(INDENT * indent_level + stripped)
        elif line_type == "comment":
            formatted.append(INDENT * indent_level + stripped)
        elif line_type == "code":
            formatted.append(INDENT * indent_level + stripped)
        elif line_type == "import":
            formatted.append(stripped)
        else:
            formatted.append("")

        prev_type = line_type

    # Final cleanup: remove trailing blanks
    while formatted and formatted[-1].strip() == "":
        formatted.pop()

    return "\n".join(formatted)
def smart_format(source: str) -> str:
    lines = source.splitlines()
    formatted = []
    INDENT = "    "  # 4 spaces
    prev_type = None

    def classify(line):
        stripped = line.strip()
        if stripped == "":
            return "empty"
        if re.match(r'^(import|from)\s+', stripped):
            return "import"
        if re.match(r'^class\s+\w+', stripped):
            return "class"
        if re.match(r'^(cpdef|def)\s+\w+', stripped):
            return "function"
        if stripped.startswith("#"):
            return "comment"
        return "code"

    for i, line in enumerate(lines):
        line_type = classify(line)

        # Insert blank line before class/function if previous wasn't empty or import
        if line_type in {"class", "function"} and prev_type not in {"empty", "import", None}:
            formatted.append("")

        # Normalize indentation
        raw = line.rstrip().replace("\t", INDENT)
        indent_level = len(raw) - len(raw.lstrip())
        stripped = raw.strip()

        # Re-indent based on context
        if line_type in {"class", "function"}:
            formatted.append(stripped)
        elif line_type == "comment":
            formatted.append(INDENT + stripped)
        elif line_type == "code":
            formatted.append(INDENT * (indent_level // 4) + stripped)
        else:
            formatted.append(stripped)

        prev_type = line_type

    # Final cleanup: remove excess trailing blank lines
    while formatted and formatted[-1] == "":
        formatted.pop()

    return "\n".join(formatted)
def bulletproof_format(source: str) -> str:
    lines = source.splitlines()
    formatted = []
    indent_stack = []
    current_indent = 0
    INDENT = "    "  # 4 spaces

    def get_indent_level(line):
        return len(line) - len(line.lstrip())

    for i, line in enumerate(lines):
        raw = line.rstrip().replace("\t", INDENT)
        stripped = raw.strip()

        # Skip empty lines
        if stripped == "":
            formatted.append("")
            continue

        # Top-level imports stay flush
        if re.match(r'^(import|from)\s+', stripped):
            formatted.append(stripped)
            continue

        # Class or function headers
        if re.match(r'^(class|cpdef)\s+\w+', stripped):
            current_indent = get_indent_level(raw)
            formatted.append(INDENT * (current_indent // 4) + stripped)
            continue

        # Decorators
        if stripped.startswith("@"):
            formatted.append(INDENT * (current_indent // 4) + stripped)
            continue

        # Comments before loops or logic
        if stripped.startswith("#"):
            formatted.append(INDENT * ((current_indent + 4) // 4) + stripped)
            continue

        # Loop headers
        if re.match(r'^for\s+\w+\s+in\s+prange\(', stripped):
            formatted.append(INDENT * ((current_indent + 4) // 4) + stripped)
            indent_stack.append(current_indent + 8)
            continue

        # Loop body or nested logic
        if indent_stack and get_indent_level(raw) >= indent_stack[-1]:
            formatted.append(INDENT * (indent_stack[-1] // 4) + stripped)
            continue

        # Return statements or logic
        if stripped.startswith("return") or stripped.startswith("cdef") or stripped.startswith("self."):
            formatted.append(INDENT * ((current_indent + 4) // 4) + stripped)
            continue

        # Default fallback
        formatted.append(INDENT * ((current_indent + 4) // 4) + stripped)

    return "\n".join(formatted)
def fix_indentation(source: str) -> str:
    lines = source.splitlines()
    fixed_lines = []
    indent_stack = []
    current_indent = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if stripped == "":
            fixed_lines.append("")
            continue

        # Normalize tabs to spaces
        line = line.replace("\t", "    ")

        # Detect class or function headers
        if re.match(r'^\s*(class|cpdef)\s+', line):
            current_indent = " " * (len(line) - len(line.lstrip()))
            fixed_lines.append(line.rstrip())
            continue

        # Detect for loops
        if re.match(r'^\s*for\s+\w+\s+in\s+prange\(', line):
            loop_indent = current_indent + "    "
            fixed_lines.append(current_indent + line.strip())
            # Check for comments above loop
            if i > 0 and lines[i - 1].strip().startswith("#"):
                comment = lines[i - 1].strip()
                fixed_lines[-2] = current_indent + comment
            continue

        # Detect loop body lines
        if indent_stack and line.startswith(indent_stack[-1] + "    "):
            fixed_lines.append(line.rstrip())
            continue

        # Default: align to current indent
        fixed_lines.append(current_indent + line.strip())

    return "\n".join(fixed_lines)

def add_cython_imports(source: str) -> str:
    lines = source.splitlines()
    new_lines = []
    import_inserted = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Insert import after existing imports
        if not import_inserted and re.match(r'^import|^from', stripped):
            new_lines.append(line)
            # Check if next line is not an import
            if i + 1 < len(lines) and not re.match(r'^import|^from', lines[i + 1].strip()):
                new_lines.append("from cython import boundscheck, wraparound")
                import_inserted = True
            continue

        # Insert decorators above function definitions
        if re.match(r'^\s*cpdef\s+\w+\s*\(', line):
            new_lines.append("    @boundscheck(False)")
            new_lines.append("    @wraparound(False)")
            new_lines.append(line)
            continue

        new_lines.append(line)

    # If no imports were found, insert at top
    if not import_inserted:
        new_lines.insert(0, "from cython import boundscheck, wraparound")

    return "\n".join(new_lines)

def move_nested_classes(source: str, hot_functions=None, target_class=None):
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

def add_hot_function_annotations(source: str, hot_functions=None, target_class=None):
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
        if match and hot_functions and match.group(3) in hot_functions:
            line = f"{match.group(1)}cpdef {line.strip()[4:]}"
        new_lines.append(line)
    return "\n".join(new_lines)

def ensure_groupentry_dataclass(source: str, hot_functions=None, target_class=None):
    if "GroupEntry" in source and "@dataclass" not in source:
        source = "from dataclasses import dataclass\n\n" + source
        source = re.sub(r"(class\s+GroupEntry\s*[:(])", r"@dataclass\n\1", source)
    return source

def apply_type_inference(source: str, hot_functions=None, target_class=None):
    logging.info("Applying type inference...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cpdef \1(\2) -> object:', source)

def convert_local_variables(source: str, hot_functions=None, target_class=None):
    logging.info("Converting local variables to cdef...")
    return re.sub(r'(\n\s*)(\w+)\s*=\s*(\d+)', r'\1cdef int \2 = \3', source)

def optimize_loops(source: str, hot_functions=None, target_class=None):
    logging.info("Optimizing loops...")
    return re.sub(r'for (\w+) in range\(', r'cdef int \1\nfor \1 in range(', source)

def convert_numpy_arrays(source: str, hot_functions=None, target_class=None):
    logging.info("Converting NumPy arrays to memoryviews...")
    return re.sub(r'np\.zeros\((.*?)\)', r'<double[:,:]> np.zeros(\1)', source)

def clean_decorators(source: str, hot_functions=None, target_class=None):
    logging.info("Converting static and class methods to cpdef with comments...")
    lines = source.splitlines()
    new_lines = []
    decorator_buffer = []
    skip_next_def = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@"):
            if "@staticmethod" in stripped:
                skip_next_def = "static"
                decorator_buffer.append((line, skip_next_def))
                continue
            elif "@classmethod" in stripped:
                skip_next_def = "class"
                decorator_buffer.append((line, skip_next_def))
                continue
            else:
                decorator_buffer.append((line, None))
                continue

        if skip_next_def and re.match(r'^\s*def\s+\w+\s*\(', line):
            indent = ' ' * (len(line) - len(line.lstrip()))
            comment = f"{indent}# Originally a {'static' if skip_next_def == 'static' else 'class'} method"
            def_line = re.sub(
                r'^\s*def\s+(\w+)\s*\((.*?)\):',
                lambda m: f"{indent}cpdef {m.group(1)}({remove_self_or_cls(m.group(2), skip_next_def)}):",
                line
            )
            for deco, kind in decorator_buffer:
                if kind is None:
                    new_lines.append(deco)
            new_lines.append(comment)
            new_lines.append(def_line)
            decorator_buffer = []
            skip_next_def = None
        else:
            if decorator_buffer:
                for deco, _ in decorator_buffer:
                    new_lines.append(deco)
                decorator_buffer = []
                skip_next_def = None
            new_lines.append(line)

    return "\n".join(new_lines)

def remove_self_or_cls(params, mode):
    parts = [p.strip() for p in params.split(',')]
    if mode == "static":
        parts = [p for p in parts if p != "self"]
    elif mode == "class":
        parts = [p for p in parts if p != "cls"]
    return ', '.join(parts)

def refine_exceptions(source: str, hot_functions=None, target_class=None):
    logging.info("Refining exception blocks...")
    return re.sub(r'except\s*:', 'except Exception:', source)

def inline_functions(source: str, hot_functions=None, target_class=None):
    logging.info("Inlining short functions...")
    return re.sub(r'def (\w+)\((.*?)\):', r'cdef inline \1(\2):', source)

def apply_parallelization(source: str) -> str:
    lines = source.splitlines()
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        match = re.match(r'^(\s*)for\s+(\w+)\s+in\s+range\((.+)\):', line)
        if match:
            indent, var, range_args = match.groups()

            # Check if 'cdef int var' already exists
            already_declared = any(
                re.match(rf'^\s*cdef\s+int\s+{var}\b', l.strip()) for l in lines[max(0, i-10):i]
            )

            if not already_declared:
                new_lines.append(f"{indent}cdef int {var}  # Declare loop variable for Cython")

            # Add comments for nogil and filtering
            new_lines.append(f"{indent}# Consider using 'with nogil:' here for thread safety and performance")
            new_lines.append(f"{indent}# Consider skipping parallelization if range size is small (e.g., < 100)")
            new_lines.append(f"{indent}for {var} in prange({range_args}):")

            # Copy loop body
            i += 1
            while i < len(lines) and lines[i].startswith(indent + "    "):
                new_lines.append(lines[i])
                i += 1
            continue

        new_lines.append(line)
        i += 1

    return "\n".join(new_lines)


def add_cython_imports(source: str) -> str:
    lines = source.splitlines()
    new_lines = []
    import_inserted = False
    decorator_inserted = set()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Insert import once, ideally after existing imports
        if not import_inserted and re.match(r'^(import|from)\s+', stripped):
            new_lines.append(line)
            # Insert after the last import
            if i + 1 < len(lines) and not re.match(r'^(import|from)\s+', lines[i + 1].strip()):
                new_lines.append("from cython import boundscheck, wraparound")
                import_inserted = True
            continue

        # Insert decorators only above cpdef functions, once per function
        if re.match(r'^\s*cpdef\s+\w+', line):
            func_name = re.findall(r'cpdef\s+(\w+)', line)[0]
            if func_name not in decorator_inserted:
                new_lines.append("    @boundscheck(False)")
                new_lines.append("    @wraparound(False)")
                decorator_inserted.add(func_name)
            new_lines.append(line)
            continue

        new_lines.append(line)

    # If no imports were found, insert at top
    if not import_inserted:
        new_lines.insert(0, "from cython import boundscheck, wraparound")

    return "\n".join(new_lines)


def add_profiling_hooks(source: str, hot_functions=None, target_class=None):
    logging.info("Adding profiling hooks to hot functions...")
    for func in hot_functions or []:
        pattern = rf'(def|cpdef)\s+{func}\s*\((.*?)\):'
        replacement = rf'\1 {func}(\2):\n    import time\n    start = time.time()'
        source = re.sub(pattern, replacement, source)
        source = re.sub(r'(return .*)', r'\1\n    print("Elapsed:", time.time() - start)', source)
    return source

# --- Static Analysis Tools ---

def score_cyclomatic_complexity(source: str, hot_functions=None, target_class=None):
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

def generate_call_graph(source: str, hot_functions=None, target_class=None):
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

def detect_dead_code(source: str, hot_functions=None, target_class=None):
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and hasattr(n.func, 'id')}
    dead = defined - called
    print("\n🧹 Dead Code Detected:")
    for func in dead:
        print(f" - {func}")
    return source

def check_decorator_compatibility(source: str, hot_functions=None, target_class=None):
    tree = ast.parse(source)
    incompatible = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Name) and deco.id in {"staticmethod", "classmethod"}:
                    incompatible.append(node.name)
    print("\n⚠️ Incompatible Decorators Detected:")
    for name in incompatible:
        print(f" - {name}")
    return source

def summarize_transformations(source: str, hot_functions=None, target_class=None):
    print("\n📋 Summary of Transformations Applied:")
    print(f" - Target class: {target_class if target_class else 'None'}")
    print(f" - Hot functions: {', '.join(hot_functions) if hot_functions else 'None'}")
    return source

def validate_syntax(source: str, hot_functions=None, target_class=None):
    try:
        ast.parse(source)
        print("\n✅ Syntax check passed.")
    except SyntaxError as e:
        print(f"\n❌ Syntax error: {e}")
    return source

def extract_hot_functions(source: str, hot_functions=None, target_class=None):
    if not hot_functions:
        return source
    tree = ast.parse(source)
    extracted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot_functions:
            extracted.append(ast.unparse(node))
    return "\n\n".join(extracted)

def wrap_in_class(source: str, hot_functions=None, target_class=None):
    if not target_class:
        return source
    indented = "\n".join("    " + line if line.strip() else line for line in source.splitlines())
    return f"class {target_class}:\n{indented}"

def generate_benchmark_harness(hot_functions):
    print("\n🧪 Generating benchmark harness...")
    for func in hot_functions:
        print(f"def benchmark_{func}():\n    import time\n    start = time.time()\n    {func}()\n    print('Time:', time.time() - start)\n")

# --- HTML Report Generator ---

def generate_html_report(applied_steps, output_path, report_name=None):
    report_path = report_name if report_name else output_path.replace(".pyx", "_report.html")
    
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
