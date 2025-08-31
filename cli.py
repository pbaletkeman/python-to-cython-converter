import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Python to Cython-compatible .pyx")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--hot-functions-file")
    parser.add_argument("--target-class")
    parser.add_argument("--config", default="converter_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
    parser.add_argument("--score-complexity", action="store_true")
    parser.add_argument("--generate-call-graph", action="store_true")
    parser.add_argument("--detect-dead-code", action="store_true")
    parser.add_argument("--auto-memoryview", action="store_true")
    parser.add_argument("--type-signatures", action="store_true")
    parser.add_argument("--check-decorators", action="store_true")
    parser.add_argument("--auto-profiling", action="store_true")
    parser.add_argument("--generate-benchmark", action="store_true")

    return parser.parse_args()

def load_config(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}
