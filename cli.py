import argparse
import json
import os
import sys

from config_validator import validate_config

def load_config(path):
    """
    Load configuration from a JSON file.
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse config file '{path}': {e}")
            sys.exit(1)
    return {}

def merge_args_with_config(args, config):
    """
    Merge CLI arguments with config file values.
    CLI arguments take precedence.
    """
    return {
        "input": args.input or config.get("input"),
        "output": args.output or config.get("output"),
        "hot_functions_file": args.hot_functions_file or config.get("hot_functions_file"),
        "target_class": args.target_class or config.get("target_class"),
        "dry_run": args.dry_run or config.get("dry_run", False),
        "verbose": args.verbose or config.get("verbose", False),
        "infer_types": args.infer_types or config.get("infer_types", False),
        "convert_vars": args.convert_vars or config.get("convert_vars", False),
        "optimize_loops": args.optimize_loops or config.get("optimize_loops", False),
        "convert_numpy": args.convert_numpy or config.get("convert_numpy", False),
        "clean_decorators": args.clean_decorators or config.get("clean_decorators", False),
        "refine_exceptions": args.refine_exceptions or config.get("refine_exceptions", False),
        "inline_functions": args.inline_functions or config.get("inline_functions", False),
        "parallelize": args.parallelize or config.get("parallelize", False),
        "add_cython_imports": args.add_cython_imports or config.get("add_cython_imports", False),
        "add_profiling": args.add_profiling or config.get("add_profiling", False),
        "score_complexity": args.score_complexity or config.get("score_complexity", False),
        "generate_call_graph": args.generate_call_graph or config.get("generate_call_graph", False),
        "detect_dead_code": args.detect_dead_code or config.get("detect_dead_code", False),
        "auto_memoryview": args.auto_memoryview or config.get("auto_memoryview", False),
        "type_signatures": args.type_signatures or config.get("type_signatures", False),
        "check_decorators": args.check_decorators or config.get("check_decorators", False),
        "auto_profiling": args.auto_profiling or config.get("auto_profiling", False),
        "generate_benchmark": args.generate_benchmark or config.get("generate_benchmark", False)
    }

def validate_settings(settings):
    """
    Validate required fields in the merged settings.
    """
    missing = []
    if not settings["input"]:
        missing.append("input")
    if not settings["output"]:
        missing.append("output")
    if missing:
        print(f"Error: Missing required fields: {', '.join(missing)}")
        sys.exit(1)

def print_usage_examples():
    """
    Print usage examples for the CLI.
    """
    print("\nExamples:")
    print("  python cli.py input.py output.py --target-class MyClass --infer-types --convert-vars")
    print("  python cli.py input.py output.py --config converter_config.json --generate-benchmark")
    print("  python cli.py input.py output.py --dry-run --verbose\n")

def get_cli_parser():
    """
    Define and return the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Python-to-Cython Transformer CLI",
        epilog="Use --config to load settings from a JSON file."
    )

    # Positional arguments
    parser.add_argument("input", help="Path to input Python file")
    parser.add_argument("output", help="Path to output file")

    # Optional arguments
    parser.add_argument("--hot-functions-file", help="Path to file listing hot functions")
    parser.add_argument("--target-class", help="Target class name to convert to cdef")
    parser.add_argument("--config", default="converter_config.json", help="Path to config file")

    # Transformation flags
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--infer-types", action="store_true", help="Infer types from usage")
    parser.add_argument("--convert-vars", action="store_true", help="Convert local variables to cdef")
    parser.add_argument("--optimize-loops", action="store_true", help="Optimize loop structures")
    parser.add_argument("--convert-numpy", action="store_true", help="Optimize NumPy array usage")
    parser.add_argument("--clean-decorators", action="store_true", help="Remove unsupported decorators")
    parser.add_argument("--refine-exceptions", action="store_true", help="Improve exception handling")
    parser.add_argument("--inline-functions", action="store_true", help="Inline short functions")
    parser.add_argument("--parallelize", action="store_true", help="Add parallelization with prange")
    parser.add_argument("--add-cython-imports", action="store_true", help="Insert Cython imports and directives")
    parser.add_argument("--add-profiling", action="store_true", help="Insert profiling hooks into hot functions")
    parser.add_argument("--score-complexity", action="store_true", help="Score cyclomatic complexity")
    parser.add_argument("--generate-call-graph", action="store_true", help="Generate function call graph")
    parser.add_argument("--detect-dead-code", action="store_true", help="Detect unreachable or unused code")
    parser.add_argument("--auto-memoryview", action="store_true", help="Convert arrays to memoryviews")
    parser.add_argument("--type-signatures", action="store_true", help="Add type signatures to functions")
    parser.add_argument("--check-decorators", action="store_true", help="Check decorator compatibility")
    parser.add_argument("--auto-profiling", action="store_true", help="Automatically detect and profile hot functions")
    parser.add_argument("--generate-benchmark", action="store_true", help="Generate benchmark harnesses")

    return parser

def main():
    parser = get_cli_parser()
    args = parser.parse_args()
    config = validate_config(args.config)
    # config = load_config(args.config)
    settings = merge_args_with_config(args, config)
    validate_settings(settings)

    print("\n\ud83d\udcc4 Final Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    print_usage_examples()

    # Placeholder for actual transformation logic
    # transform_code(**settings)

if __name__ == "__main__":
    main()
