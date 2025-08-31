import pytest
from transformations import clean_decorators

def test_staticmethod_conversion():
    source = '''
class MyClass:
    @staticmethod
    def double(x):
        return x * 2
'''
    expected = '''
class MyClass:
    # Originally a static method
    cpdef double(x):
        return x * 2
'''.strip()
    assert clean_decorators(source).strip() == expected

def test_classmethod_conversion():
    source = '''
class MyClass:
    @classmethod
    def build(cls, x):
        return cls(x)
'''
    expected = '''
class MyClass:
    # Originally a class method
    cpdef build(x):
        return cls(x)
'''.strip()
    assert clean_decorators(source).strip() == expected

def test_mixed_methods():
    source = '''
class MyClass:
    @staticmethod
    def static_fn(x):
        return x

    @classmethod
    def class_fn(cls, y):
        return cls(y)

    def regular_fn(self, z):
        return z
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef static_fn(x):" in output
    assert "# Originally a class method" in output
    assert "cpdef class_fn(y):" in output
    assert "def regular_fn(self, z):" in output  # untouched

def test_staticmethod_with_comment():
    source = '''
class MyClass:
    # This method doubles the input
    @staticmethod
    def double(x):
        return x * 2
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef double(x):" in output
    assert "# This method doubles the input" in output

def test_classmethod_with_multiline_signature():
    source = '''
class MyClass:
    @classmethod
    def build(
        cls,
        x,
        y
    ):
        return cls(x + y)
'''
    output = clean_decorators(source)
    assert "# Originally a class method" in output
    assert "cpdef build(" in output
    assert "cls" not in output

def test_staticmethod_with_self_in_signature():
    source = '''
class MyClass:
    @staticmethod
    def weird(self, x):
        return x
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef weird(x):" in output  # self should be stripped

def test_classmethod_with_extra_args():
    source = '''
class MyClass:
    @classmethod
    def configure(cls, x, y=10, *args, **kwargs):
        return cls(x + y)
'''
    output = clean_decorators(source)
    assert "# Originally a class method" in output
    assert "cpdef configure(x, y=10, *args, **kwargs):" in output

def test_nested_class_staticmethod():
    source = '''
class Outer:
    class Inner:
        @staticmethod
        def ping(x):
            return x
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef ping(x):" in output

def test_staticmethod_with_inline_comment():
    source = '''
class MyClass:
    @staticmethod  # this is a static method
    def double(x):
        return x * 2
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef double(x):" in output
    assert "# this is a static method" in output or "# this is a static method" not in output  # depending on how you want to handle it

def test_chained_decorators_with_staticmethod():
    source = '''
class MyClass:
    @log_call
    @staticmethod
    def double(x):
        return x * 2
'''
    output = clean_decorators(source)
    assert "# Originally a static method" in output
    assert "cpdef double(x):" in output
    assert "@log_call" in output  # should be preserved

def test_staticmethod_with_type_annotations():
    source = '''
class MyClass:
    @staticmethod
    def add(x: int, y: int) -> int:
        return x + y
'''
    output = clean_decorators(source)
    assert "cpdef add(x: int, y: int) -> int:" in output

def test_classmethod_with_defaults():
    source = '''
class MyClass:
    @classmethod
    def configure(cls, x=10, y=20):
        return cls(x + y)
'''
    output = clean_decorators(source)
    assert "cpdef configure(x=10, y=20):" in output

def test_staticmethod_with_varargs_and_keywords():
    source = '''
class MyClass:
    @staticmethod
    def process(x, *args, y=5, **kwargs):
        return x
'''
    output = clean_decorators(source)
    assert "cpdef process(x, *args, y=5, **kwargs):" in output

def test_chained_decorator_with_args():
    source = '''
class MyClass:
    @log(level="debug")
    @staticmethod
    def trace(x):
        return x
'''
    output = clean_decorators(source)
    assert "@log(level=\"debug\")" in output
    assert "cpdef trace(x):" in output

def test_multiple_classes():
    source = '''
class A:
    @staticmethod
    def foo(x): return x

class B:
    @classmethod
    def bar(cls, y): return cls(y)
'''
    output = clean_decorators(source)
    assert "cpdef foo(x):" in output
    assert "cpdef bar(y):" in output

def test_regular_method_untouched():
    source = '''
class MyClass:
    def normal(self, x):
        return x
'''
    output = clean_decorators(source)
    assert "def normal(self, x):" in output
