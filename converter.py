import logging
import time
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
    add_profiling_hooks
)

def timed_step(step_name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start
    logging.info(f"{step_name} took {duration:.3f}s")
    return result

def convert(source: str, hot_functions=None, class_name=None, steps=None):
    """
    Applies a series of code transformations to convert Python source code
    into a Cython-compatible version optimized for performance.

    Parameters:
        source (str): The original Python source code.
        hot_functions (list): Names of performance-critical functions to annotate.
        class_name (str): Optional class name to target for specific transformations.
        steps (dict): Optional dict to enable/disable specific transformations.

    Returns:
        Tuple[str, List[str]]: Transformed source code and list of applied transformations.
    """
    logging.info("Starting conversion pipeline...")
    applied = []
    hot_functions = hot_functions or []

    default_steps = {
        "move_nested_classes": True,
        "add_hot_function_annotations": True,
        "ensure_groupentry_dataclass": True,
        "apply_type_inference": True,
        "convert_local_variables": True,
        "optimize_loops": True,
        "convert_numpy_arrays": True,
        "clean_decorators": True,
        "refine_exceptions": True,
        "inline_functions": True,
        "apply_parallelization": True,
        "add_cython_imports": True,
        "add_profiling_hooks": True,
    }

    steps = steps or default_steps

    if steps["move_nested_classes"]:
        source = timed_step("Moved nested classes", move_nested_classes, source)
        applied.append("Moved nested classes")

    if steps["add_hot_function_annotations"]:
        source = timed_step("Annotated hot functions", add_hot_function_annotations, source, hot_functions)
        applied.append("Annotated hot functions")

    if steps["ensure_groupentry_dataclass"]:
        source = timed_step("Ensured GroupEntry is a dataclass", ensure_groupentry_dataclass, source)
        applied.append("Ensured GroupEntry is a dataclass")

    if steps["apply_type_inference"]:
        source = timed_step("Applied type inference", apply_type_inference, source)
        applied.append("Applied type inference")

    if steps["convert_local_variables"]:
        source = timed_step("Converted local variables", convert_local_variables, source)
        applied.append("Converted local variables to cdef")

    if steps["optimize_loops"]:
        source = timed_step("Optimized loops", optimize_loops, source)
        applied.append("Optimized loops")

    if steps["convert_numpy_arrays"]:
        source = timed_step("Converted NumPy arrays", convert_numpy_arrays, source)
        applied.append("Converted NumPy arrays to memoryviews")

    if steps["clean_decorators"]:
        source = timed_step("Cleaned decorators", clean_decorators, source)
        applied.append("Cleaned decorators and preserved static/class methods")

    if steps["refine_exceptions"]:
        source = timed_step("Refined exceptions", refine_exceptions, source)
        applied.append("Refined exception handling")

    if steps["inline_functions"]:
        source = timed_step("Inlined functions", inline_functions, source)
        applied.append("Inlined short functions")

    if steps["apply_parallelization"]:
        source = timed_step("Applied parallelization", apply_parallelization, source)
        applied.append("Applied parallelization")

    if steps["add_cython_imports"]:
        source = timed_step("Added Cython imports", add_cython_imports, source)
        applied.append("Added Cython imports")

    if steps["add_profiling_hooks"]:
        source = timed_step("Inserted profiling hooks", add_profiling_hooks, source, hot_functions)
        applied.append("Inserted profiling hooks")

    logging.info("Conversion complete.")
    return source, applied
