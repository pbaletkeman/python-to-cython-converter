import ast
import astor
import re

class CythonInjector(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Convert def to cpdef manually
        node.name = node.name
        node.decorator_list = []
        node.returns = ast.Name(id='object', ctx=ast.Load())
        return node

    def visit_ClassDef(self, node):
        # Inject cpdef __init__ if missing
        method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        if '__init__' not in method_names:
            init_func = ast.FunctionDef(
                name='__init__',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='self'), ast.arg(arg='size')],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='data', ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='zeros', ctx=ast.Load()),
                            args=[ast.Tuple(elts=[ast.Name(id='size', ctx=ast.Load()), ast.Name(id='size', ctx=ast.Load())], ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                ],
                decorator_list=[],
                returns=ast.Name(id='object', ctx=ast.Load())
            )
            node.body.insert(0, init_func)
        return node

def format_and_inject(source: str) -> str:
    tree = ast.parse(source)
    tree = CythonInjector().visit(tree)
    ast.fix_missing_locations(tree)
    return astor.to_source(tree)

import ast

class CpdefTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = node.name
        node.returns = ast.Name(id='object', ctx=ast.Load())
        node.decorator_list = []
        node.type_comment = None
        return node

def inject_cython_lines(lines):
    formatted = []
    indent_stack = [0]
    for line in lines:
        stripped = line.strip()

        # Handle cdef lines
        if stripped.startswith("cdef "):
            indent = "    " * indent_stack[-1]
            formatted.append(indent + stripped)
            continue

        # Handle prange loops
        if "prange(" in stripped:
            indent = "    " * indent_stack[-1]
            formatted.append(indent + stripped)
            indent_stack.append(indent_stack[-1] + 1)
            continue

        # Handle with nogil
        if "with nogil:" in stripped:
            indent = "    " * indent_stack[-1]
            formatted.append(indent + stripped)
            indent_stack.append(indent_stack[-1] + 1)
            continue

        # Handle return or end of block
        if stripped.startswith("return") or stripped == "":
            if len(indent_stack) > 1:
                indent_stack.pop()
            indent = "    " * indent_stack[-1]
            formatted.append(indent + stripped)
            continue

        # Default line
        indent = "    " * indent_stack[-1]
        formatted.append(indent + stripped)

    return formatted

def format_comments_and_docstrings(lines):
    formatted = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            indent = "    " if i > 0 and lines[i - 1].strip().startswith("def") else ""
            formatted.append(indent + stripped)
            continue

        # Inline comments
        if stripped.startswith("#"):
            indent = "    "
            formatted.append(indent + stripped)
            continue

        formatted.append(line)
    return formatted

def cython_formatter(source: str) -> str:
    tree = ast.parse(source)
    transformer = CpdefTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Step 1: AST-based formatting
    python_code = ast.unparse(tree)

    # Step 2: Line-level Cython injection
    lines = python_code.splitlines()
    cythonized = inject_cython_lines(lines)

    # Step 3: Comment/docstring formatting
    final = format_comments_and_docstrings(cythonized)

    return "\n".join(final)


def format_memoryviews(lines):
    formatted = []
    for line in lines:
        if "<" in line and ">" in line:
            # Normalize spacing around cast
            line = re.sub(r"<\s*", "<", line)
            line = re.sub(r"\s*>", ">", line)
            line = re.sub(r">\s*", "> ", line)
        formatted.append(line)
    return formatted

def attach_decorators(lines):
    formatted = []
    pending_decorators = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@"):
            pending_decorators.append(stripped)
            continue
        if stripped.startswith("cpdef") or stripped.startswith("def"):
            formatted.extend(pending_decorators)
            pending_decorators = []
        formatted.append(line)
    return formatted

def infer_cdef_types(lines):
    inferred = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r"(\w+)\s*=\s*(.+)", stripped)
        if match:
            var, value = match.groups()
            if value.isdigit():
                inferred.append(f"cdef int {var} = {value}")
            elif re.match(r"\d+\.\d+", value):
                inferred.append(f"cdef double {var} = {value}")
            elif value in {"True", "False"}:
                inferred.append(f"cdef bint {var} = {value}")
            else:
                inferred.append(line)
        else:
            inferred.append(line)
    return inferred

import re

def reindent_cython_blocks(lines):
    INDENT = "    "
    formatted = []
    indent_stack = [0]

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if stripped == "":
            formatted.append("")
            continue

        # Detect block starters
        block_start = re.match(r'^(class|cpdef|def|for|while|if|elif|else|try|except|with)\b', stripped)
        if block_start:
            current_indent = indent_stack[-1]
            formatted.append(INDENT * current_indent + stripped)
            indent_stack.append(current_indent + 1)
            continue

        # Dedent after return or end of block
        if stripped.startswith("return") or stripped.startswith("pass"):
            if len(indent_stack) > 1:
                indent_stack.pop()
            current_indent = indent_stack[-1]
            formatted.append(INDENT * current_indent + stripped)
            continue

        # Decorators and comments stay at current level
        if stripped.startswith("@") or stripped.startswith("#"):
            current_indent = indent_stack[-1]
            formatted.append(INDENT * current_indent + stripped)
            continue

        # Default: apply current indentation
        current_indent = indent_stack[-1]
        formatted.append(INDENT * current_indent + stripped)

    return formatted


def cython_transform_pipeline(source: str) -> str:
    lines = source.splitlines()

    # Step 1: Decorator handling
    lines = attach_decorators(lines)

    # Step 2: Memoryview formatting
    lines = format_memoryviews(lines)

    # Step 3: Type inference
    lines = infer_cdef_types(lines)

    # Step 4: Final cleanup
    return "\n".join(lines)

def format_cython_code(source: str) -> str:
    lines = source.splitlines()

    # Step 1: Attach decorators properly
    lines = attach_decorators(lines)

    # Step 2: Format memoryviews
    lines = format_memoryviews(lines)

    # Step 3: Infer and format cdef types
    lines = infer_cdef_types(lines)

    # Step 4: Re-indent based on block structure
    lines = reindent_cython_blocks(lines)

    # Step 5: Format comments and docstrings
    lines = format_comments_and_docstrings(lines)

    return "\n".join(lines)