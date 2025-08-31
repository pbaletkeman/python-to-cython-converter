import argparse
import json
import os
import sys

from config_validator import validate_config
from converter import convert

def load_config(path):
    """
    Load configuration from a JSON file.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse config file '{path}': {e}")
            sys.exit(1)
    return {}


def resolve_flag(cli_value, config_value):
    return config_value if cli_value is None else cli_value

def resolve(cli_value, config_value):
    return config_value if cli_value is None else cli_value

def merge_args_with_config(args, config):
    """
    Merge CLI arguments with config file values.
    CLI arguments take precedence only if explicitly set.
    """
    transformations = config.get("transformations", {})

    return {
        "input": resolve(args.input, config.get("input")),
        "output": resolve(args.output, config.get("output")),
        "hot_functions": config.get("hot_functions", []),  # list directly from config
        "hot_functions_file": resolve(args.hot_functions_file, config.get("hot_functions_file")),
        "target_class": resolve(args.target_class, config.get("target_class")),
        "dry_run": resolve(args.dry_run, config.get("dry_run", False)),
        "verbose": resolve(args.verbose, config.get("verbose", False)),

        # Flattened transformation flags
        "infer_types": resolve(args.infer_types, transformations.get("apply_type_inference", config.get("infer_types", False))),
        "convert_vars": resolve(args.convert_vars, transformations.get("convert_local_variables", config.get("convert_vars", False))),
        "optimize_loops": resolve(args.optimize_loops, transformations.get("optimize_loops", config.get("optimize_loops", False))),
        "convert_numpy": resolve(args.convert_numpy, transformations.get("convert_numpy_arrays", config.get("convert_numpy", False))),
        "clean_decorators": resolve(args.clean_decorators, transformations.get("clean_decorators", config.get("clean_decorators", False))),
        "refine_exceptions": resolve(args.refine_exceptions, transformations.get("refine_exceptions", config.get("refine_exceptions", False))),
        "inline_functions": resolve(args.inline_functions, transformations.get("inline_functions", config.get("inline_functions", False))),
        "parallelize": resolve(args.parallelize, transformations.get("apply_parallelization", config.get("parallelize", False))),
        "add_cython_imports": resolve(args.add_cython_imports, transformations.get("add_cython_imports", config.get("add_cython_imports", False))),
        "add_profiling": resolve(args.add_profiling, transformations.get("add_profiling_hooks", config.get("add_profiling", False))),

        # Additional flags
        "score_complexity": resolve(args.score_complexity, transformations.get("score_complexity", False)),
        "generate_call_graph": resolve(args.generate_call_graph, transformations.get("generate_call_graph", False)),
        "detect_dead_code": resolve(args.detect_dead_code, transformations.get("detect_dead_code", False)),
        "auto_memoryview": resolve(args.auto_memoryview, transformations.get("auto_memoryview", False)),
        "type_signatures": resolve(args.type_signatures, transformations.get("type_signatures", False)),
        "check_decorators": resolve(args.check_decorators, transformations.get("check_decorators", False)),
        "auto_profiling": resolve(args.auto_profiling, transformations.get("auto_profiling", False)),
        "generate_benchmark": resolve(args.generate_benchmark, transformations.get("generate_benchmark", False))
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
    print("  python cli.py --input input.py --output output.py --target-class MyClass --infer-types --convert-vars")
    print("  python cli.py --input input.py --output output.py --config converter_config.json --generate-benchmark")
    print("  python cli.py --input input.py --output output.py --dry-run --verbose\n")

def get_cli_parser():
    """
    Define and return the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Python-to-Cython Transformer CLI",
        epilog="Use --config to load settings from a JSON file."
    )

    # Optional arguments
    parser.add_argument("--input", help="Path to input Python file")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--hot-functions-file", help="Path to file listing hot functions")
    parser.add_argument("--target-class", help="Target class name to convert to cdef")
    parser.add_argument("--config", default="converter_config.json", help="Path to config file")

    # Transformation flags
    parser.add_argument("--add-cython-imports", action="store_true", default=None, help="Insert Cython imports and directives")
    parser.add_argument("--auto-profiling", action="store_true", default=None, help="Automatically detect and profile hot functions")
    parser.add_argument("--add-profiling", action="store_true", default=None, help="Insert profiling hooks into hot functions")
    parser.add_argument("--auto-memoryview", action="store_true", default=None, help="Convert arrays to memoryviews")
    parser.add_argument("--convert-vars", action="store_true", default=None, help="Convert local variables to cdef")
    parser.add_argument("--convert-numpy", action="store_true", default=None, help="Optimize NumPy array usage")
    parser.add_argument("--clean-decorators", action="store_true", default=None, help="Remove unsupported decorators")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Preview changes without writing output")
    parser.add_argument("--detect-dead-code", action="store_true", default=None, help="Detect unreachable or unused code")
    parser.add_argument("--generate-benchmark", action="store_true", default=None, help="Generate benchmark harnesses")
    parser.add_argument("--generate-call-graph", action="store_true", default=None, help="Generate function call graph")
    parser.add_argument("--infer-types", action="store_true", default=None, help="Infer types from usage")
    parser.add_argument("--inline-functions", action="store_true", default=None, help="Inline short functions")
    parser.add_argument("--optimize-loops", action="store_true", default=None, help="Optimize loop structures")
    parser.add_argument("--parallelize", action="store_true", default=None, help="Add parallelization with prange")
    parser.add_argument("--type-signatures", action="store_true", default=None, help="Add type signatures to functions")
    parser.add_argument("--check-decorators", action="store_true", default=None, help="Check decorator compatibility")
    parser.add_argument("--score-complexity", action="store_true", default=None, help="Score cyclomatic complexity")
    parser.add_argument("--refine-exceptions", action="store_true", default=None, help="Improve exception handling")
    parser.add_argument("--verbose", action="store_true", default=None, help="Enable verbose logging")

    return parser

def main():
    parser = get_cli_parser()
    args = parser.parse_args()
    config = validate_config(args.config)
    # config = load_config(args.config)
    settings = merge_args_with_config(args, config)
    
    print("Loaded transformations:")
    for k, v in config["transformations"].items():
        print(f"  {k}: {v}")

    
    validate_settings(settings)

    print("\nFinal Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    print_usage_examples()

    source_code = open(settings["input"], "r", encoding="utf-8").read()

    transformed, applied_steps = convert(
        source_code,
        hot_functions=settings.get("hot_functions", []),
        class_name=settings.get("target_class"),
        steps=settings.get("transformations")
    )
    with open(settings["output"], "w", encoding="utf-8") as f:
        f.write(transformed)
    print("\n✅ Transformation complete.")

if __name__ == "__main__":
    main()
