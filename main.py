import argparse
import logging
import subprocess
from transformations import (
    move_nested_classes,
    add_hot_function_annotations,
    ensure_groupentry_dataclass,
    apply_type_inference,
    convert_local_variables,
    optimize_loops,
    convert_numpy_arrays,
    clean_decorators,
    refine_exceptions,
    inline_functions,
    apply_parallelization,
    add_cython_imports,
    add_profiling_hooks,
    score_cyclomatic_complexity,
    generate_call_graph,
    detect_dead_code,
    check_decorator_compatibility,
    generate_benchmark_harness,
    generate_html_report
)

def load_source(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_source(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def format_and_lint(path):
    print("\n🧼 Formatting and linting...")
    subprocess.run(["black", path])
    subprocess.run(["flake8", path])

def main():
    parser = argparse.ArgumentParser(description="Apply Cython transformations to Python source code.")
    parser.add_argument("input", help="Path to input Python file")
    parser.add_argument("output", help="Path to output transformed file")
    parser.add_argument("--class", dest="target_class", help="Target class to transform")
    parser.add_argument("--hot", nargs="*", default=[], help="List of hot functions to optimize")
    parser.add_argument("--analyze", action="store_true", help="Run advanced analysis tools")
    parser.add_argument("--benchmark", action="store_true", help="Generate benchmark harness")
    parser.add_argument("--report-name", help="Custom name for the HTML transformation report")
    parser.add_argument("--format", action="store_true", help="Format and lint the output file using Black and Flake8")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    source = load_source(args.input)
    applied = []

    # --- Apply Transformations ---
    source = move_nested_classes(source, args.target_class); applied.append("Moved nested classes")
    source = add_hot_function_annotations(source, args.hot, args.target_class); applied.append("Annotated hot functions")
    source = ensure_groupentry_dataclass(source); applied.append("Ensured GroupEntry dataclass")
    source = apply_type_inference(source); applied.append("Applied type inference")
    source = convert_local_variables(source); applied.append("Converted local variables")
    source = optimize_loops(source); applied.append("Optimized loops")
    source = convert_numpy_arrays(source); applied.append("Converted NumPy arrays to memoryviews")
    source = clean_decorators(source); applied.append("Removed unsupported decorators")
    source = refine_exceptions(source); applied.append("Refined exception blocks")
    source = inline_functions(source); applied.append("Inlined short functions")
    source = apply_parallelization(source); applied.append("Applied parallelization with prange")
    source = add_cython_imports(source); applied.append("Added Cython-specific imports")
    source = add_profiling_hooks(source, args.hot); applied.append("Inserted profiling hooks")

    # --- Optional Analysis ---
    if args.analyze:
        score_cyclomatic_complexity(source)
        generate_call_graph(source)
        detect_dead_code(source)
        check_decorator_compatibility(source)

    # --- Optional Benchmark Harness ---
    if args.benchmark:
        generate_benchmark_harness(args.hot)

    # --- Save Transformed Code ---
    save_source(args.output, source)
    print(f"\n✅ Transformation complete. Output saved to: {args.output}")

    # --- Generate HTML Report ---
    generate_html_report(applied, args.output, args.report_name)

    # --- Optional Formatting ---
    if args.format:
        format_and_lint(args.output)

if __name__ == "__main__":
    main()
