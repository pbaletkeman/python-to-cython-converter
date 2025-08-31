import json
import os
import sys

REQUIRED_FIELDS = ["input", "output", "transformations"]
VALID_TRANSFORMATIONS = {
    "add_cython_imports",
    "add_hot_function_annotations",
    "add_profiling",
    "add_profiling_hooks",
    "apply_parallelization",
    "apply_type_inference",
    "auto_memoryview",
    "auto_profiling",
    "check_decorators",
    "clean_decorators",
    "convert_local_variables",
    "convert_numpy",
    "convert_numpy_arrays",
    "convert_vars",
    "detect_dead_code",
    "ensure_groupentry_dataclass",
    "generate_benchmark",
    "generate_call_graph",
    "infer_types",
    "inline_functions",
    "move_nested_classes",
    "optimize_loops",
    "parallelize",
    "refine_exceptions",
    "score_complexity",
    "type_signatures"
}

def validate_config(config_path):
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        sys.exit(1)

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            print(f"❌ Missing required field: '{field}'")
            sys.exit(1)

    # Validate types
    if not isinstance(config["input"], str):
        print("❌ 'input' must be a string")
        sys.exit(1)
    if not isinstance(config["output"], str):
        print("❌ 'output' must be a string")
        sys.exit(1)
    if not isinstance(config["transformations"], dict):
        print("❌ 'transformations' must be a dictionary")
        sys.exit(1)

    # Validate transformation flags
    for key in config["transformations"]:
        if key not in VALID_TRANSFORMATIONS:
            print(f"❌ Unknown transformation: '{key}'")
            sys.exit(1)
        if not isinstance(config["transformations"][key], bool):
            print(f"❌ Transformation '{key}' must be a boolean")
            sys.exit(1)

    # Optional fields
    if "hot_functions" in config and not isinstance(config["hot_functions"], list):
        print("❌ 'hot_functions' must be a list")
        sys.exit(1)
    if "target_class" in config and not isinstance(config["target_class"], str):
        print("❌ 'target_class' must be a string")
        sys.exit(1)
    if "dry_run" in config and not isinstance(config["dry_run"], bool):
        print("❌ 'dry_run' must be a boolean")
        sys.exit(1)
    if "verbose" in config and not isinstance(config["verbose"], bool):
        print("❌ 'verbose' must be a boolean")
        sys.exit(1)

    print("✅ Config validation passed.")
    return config
