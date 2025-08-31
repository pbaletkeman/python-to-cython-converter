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
pip install black flake8
```

---

## 📁 File Structure

```
project/
├── main.py
├── transformations.py
├── config.json   # Optional config file
├── input.py      # Your source file
├── output.py     # Transformed output
└── transformation_report.html  # Optional HTML report
```

---

## 🧪 Usage

### 🔹 Basic Transformation

```bash
python main.py input.py output.py
```

### 🔹 With Class Targeting and Hot Functions

```bash
python main.py input.py output.py --class MyClass --hot process_data compute_score
```

### 🔹 Enable Static Analysis

```bash
python main.py input.py output.py --analyze
```

### 🔹 Generate Benchmark Harness

```bash
python main.py input.py output.py --benchmark
```

### 🔹 Format and Lint Output

```bash
python main.py input.py output.py --format
```

### 🔹 Custom HTML Report Name

```bash
python main.py input.py output.py --report-name my_report.html
```

### 🔹 Full Example

```bash
python main.py input.py output.py
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
  "input_file": "input.py",
  "output_file": "output.py",
  "target_class": "MyClass",
  "hot_functions": ["process_data", "compute_score"],
  "enable_analysis": true,
  "enable_benchmark": true,
  "enable_formatting": true,
  "report_name": "transformation_report.html"
}
```

---

## 🧾 Switch Reference

| Switch          | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `input.py`      | Path to the input Python file                                          |
| `output.py`     | Path to save the transformed output                                    |
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
