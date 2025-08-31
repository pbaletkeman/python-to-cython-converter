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


def rewrite_function_to_cpdef(source: str) -> str:
    lines = source.splitlines()
    rewritten = []
    buffer = []
    inside_function = False
    indent = ""

    for line in lines:
        stripped = line.strip()

        # Buffer decorators
        if stripped.startswith("@") and not inside_function:
            buffer.append(line)
            continue

        # Detect function header
        def_match = re.match(r'^(\s*)(def|cpdef)\s+(\w+)\s*\(.*\)\s*->\s*\w+\s*:', line)
        if def_match:
            indent, keyword, name = def_match.groups()

            # Avoid double cpdef
            if keyword == "cpdef":
                header = line
            else:
                header = line.replace("def", "cpdef", 1)

            # Flush decorators + header
            rewritten.extend(buffer)
            buffer.clear()
            rewritten.append(header)
            inside_function = True
            continue

        # Detect end of function (naively: dedent or blank line after body)
        if inside_function and (stripped == "" or not line.startswith(indent + "    ")):
            inside_function = False
            buffer.clear()

        rewritten.append(line)

    return "\n".join(rewritten)


