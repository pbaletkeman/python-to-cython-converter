import ast
import asttokens
import textwrap
import re

def infer_return_type(func_node):
    """Infer return type from return statements using simple heuristics."""
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return):
            value = stmt.value
            if isinstance(value, ast.Constant):
                if isinstance(value.value, int):
                    return "int"
                elif isinstance(value.value, str):
                    return "str"
                elif isinstance(value.value, float):
                    return "float"
            elif isinstance(value, ast.List):
                return "list"
            elif isinstance(value, ast.Dict):
                return "dict"
            elif isinstance(value, ast.Call):
                return "object"  # Could be anything
    return "object"


import ast

def rewrite_function_to_cpdef(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    rewritten = []
    buffer = []
    inside_function = False
    indent = ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return_type = infer_return_type(node)
            header_line = lines[node.lineno - 1]
            indent_match = re.match(r'^(\s*)', header_line)
            indent = indent_match.group(1) if indent_match else ""
            args = ", ".join([arg.arg for arg in node.args.args])
            new_header = f"{indent}cpdef {return_type} {node.name}({args}):"

            # Flush decorators + header
            rewritten.extend(buffer)
            buffer.clear()
            rewritten.append(new_header)
            inside_function = True
            continue

        # Handle decorators
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            buffer.append(lines[node.lineno - 1])
            continue

    # Fallback: return original if no match
    if not rewritten:
        return source

    # Append remaining lines
    for i in range(node.lineno, len(lines)):
        rewritten.append(lines[i])

    return "\n".join(rewritten)
