# 🔧 Python Cython Transformer Tool

This tool automates the transformation of Python source code into Cython-optimized code. It supports performance enhancements, static analysis, profiling hooks, and generates a clean HTML report of all applied transformations.

---

## 🚀 Features

- Move nested classes outside their parent
- Annotate hot functions with `cpdef`
- Convert `GroupEntry` to a dataclass
- Infer basic type hints and convert local variables to `cdef`
- Optimize loops and NumPy arrays
- Remove unsupported decorators
- Refine exception handling
- Inline short functions
- Add parallelization with `prange`
- Insert profiling hooks
- Run static analysis (complexity, call graph, dead code)
- Generate benchmark harnesses
- Format and lint output with Black + Flake8
- Create a styled HTML transformation report

---

## 📦 Installation

```bash
pip install black flake8 asttokens astor
```

---

## 📁 File Structure

```
project/
├── main.py
├── transformations.py
├── config.json   # Optional config file
├── config_validator.py # verify config.json file is valid
├── input.py      # Your source file
├── output.py     # Transformed output
└── transformation_report.html  # Optional HTML report
```

---

## 🧪 Usage

### 🔹 Basic Transformation

```bash
python cli.py --input input.py --output output.pyx
```

### 🔹 With Class Targeting and Hot Functions

```bash
python cli.py --input input.py --output output.pyx --class MyClass --hot process_data compute_score
```

### 🔹 Enable Static Analysis

```bash
python cli.py --input input.py --input output.pyx --analyze
```

### 🔹 Generate Benchmark Harness

```bash
python cli.py --input input.py --output output.pyx --benchmark
```

### 🔹 Format and Lint Output

```bash
python cli.py --input input.py --output output.pyx --format
```

### 🔹 Custom HTML Report Name

```bash
python cli.py --input input.py --output output.pyx --report-name my_report.html
```

### 🔹 Full Example

```bash
python cli.py --input input.py --output output.pyx
  --class MyClass
  --hot process_data compute_score
  --analyze
  --benchmark
  --format
  --report-name my_report.html
```

---

## ⚙️ Optional: config.json

You can define default settings in a `config.json` file:

```json
{
  "input": "samples\\input1.py",
  "output": "samples\\input1.pyx",
  "target_class": "FastProcessor",
  "hot_functions": [
    "cached_compute",
    "nested_loop",
    "global_function",
    "short_inline"
  ],
  "dry_run": false,
  "verbose": true,
  "transformations": {
    "add_cython_imports": true,
    "add_hot_function_annotations": true,
    "add_profiling": true,
    "add_profiling_hooks": true,
    "apply_parallelization": true,
    "apply_type_inference": true,
    "auto_memoryview": true,
    "auto_profiling": true,
    "check_decorators": true,
    "clean_decorators": true,
    "convert_local_variables": true,
    "convert_numpy": true,
    "convert_numpy_arrays": true,
    "convert_vars": true,
    "detect_dead_code": true,
    "ensure_groupentry_dataclass": true,
    "generate_benchmark": true,
    "generate_call_graph": true,
    "infer_types": true,
    "inline_functions": true,
    "move_nested_classes": true,
    "optimize_loops": true,
    "parallelize": true,
    "refine_exceptions": true,
    "score_complexity": true,
    "type_signatures": true
  }
}
```

---

## 🧾 Switch Reference

| Switch          | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `--input`       | Path to the input Python file                                          |
| `--output`      | Path to save the transformed output                                    |
| `--class`       | Target class name to convert to `cdef class`                           |
| `--hot`         | List of hot functions to annotate with `cpdef` and add profiling hooks |
| `--analyze`     | Run static analysis: complexity, call graph, dead code                 |
| `--benchmark`   | Generate benchmark harnesses for hot functions                         |
| `--format`      | Format and lint the output file using Black and Flake8                 |
| `--report-name` | Custom filename for the HTML transformation report                     |

---

## 🧠 Author

Pete Letkeman  
Enhanced by Microsoft Copilot ✨

## 📜 License

This project is licensed under a Custom Non-Commercial License. Commercial use requires written authorization. See the `LICENSE` file for details.
