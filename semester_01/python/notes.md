<!-- markdownlint-disable MD025, MD001, MD024 -->

# Notes

## Interesting Packages

- `click`: command-line interfaces (CLIs): arguments, options, and input validation

- `ipywidgets`: interactive widgets in Jupyter: dynamic visualizations and user interactions like sliders, buttons, and dropdowns

## `int`

Python's `int` have **arbitrary precision**, meaning they can grow as large as the available memory allow.

Python does not have integer overflow.

## `.sort()` & `sorted()`

Create new array:

```py
l: List[Tuple[str, int]]
l = sorted(l, key=lambda x: x[1], reverse=True)
```

Sort within array:

```py
l.sort(key=lambda x: x[1], reverse=True)
```

## `copy`

`copy()` - creates a **shallow copy** of an object: copies the object itself but does not create copies of nested objects, only reference them.

`deepcopy()` - recursive `copy()`.

```py
import copy

# Example with nested list
original = [[1, 2, 3], [4, 5, 6]]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

# Modify the nested list
original[0][0] = 100

print(shallow)  # [[100, 2, 3], [4, 5, 6]] (shallow copy still references original nested lists)
print(deep)     # [[1, 2, 3], [4, 5, 6]] (deep copy remains unchanged)
```

## `std::swap` in Python

Since each variable is a reference:

```py
a, b = b, a
```

This works because the right-hand side (`b, a`) is evaluated as a `(b, a)` tuple first, and then unpacked into `a` and `b`.

# Lecture 1 - Introduction

## Language

**Python** is the language specification, it is **crossplatform** language, meaning it has many implementations.  
**CPython** is the most commonly used implementation of that specification.

Other implementations:

- PyPy: faster alternative to CPython using Just-In-Time (JIT) compilation. Written in python.
- Jython: on Java
- IronPython: for .NET
- MicroPython: lightweight implementation of Python for microcontrollers

---

Python is **multiparadigm**: OOP prevails, a bit of functional, declarative, procedural, $\dots$  
Python is **interpretable**.  
Python is **dynamic**.

## help(), dir()

```py
help(function_name)
dir(function_name)      # all methods
```

**Magic methods** - special methods in Python that begin and end with \_\_.  
They enable Python's built-in behaviors and syntax to work with user-defined objects.

# Lecture 2 - None

# Lecture 3.1 - Types

## Types

```py
a = 1
id(a)       # reference (where it stored in memory)
type(a)
```

```py
a == b      # by value
a is b      # by `id()`
```

### `id`

`id` - unique integer identifier for an object during its lifetime.

```py
a is b
# the same
id(a) == id(b)
```

In CPython, `id(obj)` often corresponds to an object's memory address, but this is implementation-dependent.

```py
import ctypes

x = 42
print(id(x))  # Some integer
print(ctypes.cast(id(x), ctypes.py_object).value)  # Retrieves object
```

### Immutable

These types cannot be modified in place by reference, they create a new object.

`id()` changes.

```py
a = 5                           # integer
a = a + 1

b = 3.14                        # float
b = b * 2

s = "hello"                     # string
s = s + " world"

t = (1, 2, 3)                   # tuple
# Note: If the tuple contains mutable objects, they can be *indirectly* modified.

fs = frozenset([1, 2, 3])       # frozenset

b = b"immutable"                # bytes
```

### Mutable

These types can be modified in place by reference.

`id()` doesn't change.

```py
my_list = [1, 2, 3]             # list
my_list[0] = 4

my_dict = {'key': 'value'}      # dictionary
my_dict['key'] = 'new_value'

my_set = {1, 2, 3}              # set
my_set.add(4)

my_bytearray = bytearray(b"hi") # bytearray
my_bytearray[0] = ord('H')
```

#### `__hash__`

All mutable types are **unhashable**, because they can be changed.

-`tuple` with mutables is unhashable, because tuple references it's elements.

-`frozenset` with mutables can't be created, since they have no `__hash__`.

# Lecture 3.2 - Memory model

## Object

In Python everything is represented as an **object**.

Each object consists of:

1. reference count
2. type identifier (PyTypeObject)
3. Data (the actual value of the object)

- **Heap memory**: every object stored here.

- **Call stack memory**: Stores **frames** (execution contexts of function calls):
  1. Code being executed: `f_code`
  2. Scopes with references pointing to objects in the heap:
     - `f_locals`
     - `f_back` - nonlocal
     - `f_globals`

Variables in Python are **references** (pointers) (`id()`) to objects in memory, not the objects themselves.  
Each object has a **reference count**, which tracks the number of references to it.  
When the count drops to zero, the memory is **deallocated**.

Conceptualization: **Variable $\rightarrow$ Reference & Reference Count $\rightarrow$ Object**

```py
import copy
x = [[1]]
y = x                   # references are the same
y = copy.deepcopy(x)    # deep copy, independent of x

```

## Memory Optimization

### Interning

Python preallocates and caches integers in the range of **-5 to 256**.

```py
a = 256
b = 256

a is b          # True (same cached integer)
```

### Lazy Copying

Use of **shallow copying** and **copy-on-write** to avoid unnecessary duplication of data.

## `list`

### Structure of `list`

`struct.unpack(format, buffer)` - unpack binary data into Python objects

| **Parameter**                    | **Size**     | **Type**          |
| -------------------------------- | ------------ | ----------------- |
| Reference counter                | 8 bytes      | L - unsigned long |
| Type pointer                     | 8 bytes      | L - unsigned long |
| List length                      | 8 bytes      | L - unsigned long |
| Data pointer                     | 8 bytes      | L - unsigned long |
| Pointers to elements             | 8 bytes each | L - unsigned long |
| List length including allocation | 8 bytes      | L - unsigned long |

```py
a = [1, 3, "ab"]

data_1 = struct.unpack("5L", ctypes.string_at(id(a), 40))
data_2 = struct.unpack(f"{len(a)}L", ctypes.string_at(data_1[3], len(a) * 8))
```

```txt
(1 - Reference counter, 139946864953824 - Type pointer of list, 3 - List length, 139945580358192 - Data pointer, 4 - preallocated length),

Chunk of data: (139946864964448, 139946864964552, 139946828382896)
```

### +=

```py
a = [1]
b = a
...
a, b, id(a) == id(b)
```

- if

  ```py
  b = b + [2]
  ```

  then redefinition:

  ```txt
  ([1], [1, 2], False)
  ```

- if

  ```py
  b += [2]
  ```

  then modification:

  ```txt
  ([1, 2], [1, 2], True)
  ```

### Slicing

`[:n]` creates a new object.
`[n:]` only references slice of list.

### References

The list only stores a references to it's elements, that's why:

```py
[1, 2, []].__sizeof__() == [1, 2, [1, 1, 1, 1]].__sizeof__()
```

`__size_of__()` - the memory size of the container itself.

`sys.getsizeof()` = `__size_of__()` + garbage collection overhead

## `set`

### Structure of `set`

| **Parameter**                                | **Size**     | **Type**                   |
| -------------------------------------------- | ------------ | -------------------------- |
| Reference counter                            | 8 bytes      | L - unsigned long          |
| Type pointer                                 | 8 bytes      | L - unsigned long          |
| Length of the set                            | 8 bytes      | L - unsigned long          |
| Number of filled slots in the hash table     | 8 bytes      | L - unsigned long          |
| Number of active slots in the hash table - 1 | 8 bytes      | L - unsigned long          |
| Pointer to the hash table                    | 8 bytes      | L - unsigned long          |
| Hash                                         | 8 bytes      | l - signed long            |
| Auxiliary info                               | 8 bytes      | L - unsigned long          |
| Embedded small table                         | 16 bytes × 8 | L - unsigned + signed long |
| Weakref                                      | 8 bytes      | L - unsigned long          |

### `frozenset`

Immutable version of set.

## (Un)Packing

Does in $O(1)$, because of only referencing:

```py
a, b = b, a
```

---

```py
x = [...]

a, *b, c = x    # a=x[0], b=x[0:-1], c=x[-1]
```

---

```py
d = {"a": 1, "b": 2, "c": 3}

def _f(a, b, c):
    print(a, b, c)

_f(**d)
```

## Data Structures

From `collections`:

```py
from collections import deque, defaultdict, Counter, OrderedDict
import heapq
```

# Lecture 4.1 - Functions

## `args`

All of the arguments are taken by reference (they are not copied).

### function as arg

```py
from typing import Callable

# Callable[[Arg1Type, Arg2Type, ...], ReturnType] | DefaultType = DefaultValue
def func(func: Callable[[int, str], bool] | None = None) -> list[str]:
    ...
```

### `*`

All arguments after `*` are obligated to be called with specification:

```py
def func(a, b, *, c, d) -> None:
    ...

func(1, 2, c=3, d=5)
```

```py
a_list = [1]
kv = {"b"=2, "c"=3, "d"=5}

func(*a, **kv)
```

### Mutable Default

```py
def md(l: list[str] = []) -> list[str]:
    l.append("Hi!")
    return l

md()    # ["Hi!"]
md()    # ["Hi!", "Hi!"]

def md(l: list[str] = None) -> list[str]:
    if l is None:
        l = []
    l.append("Hi!")
    return l

md()    # ["Hi!"]
md()    # ["Hi!"]
```

$\rightarrow$ use `None`.

### \*args

```py
def func(*args: float) -> None:     # or *any_name
    ...

func(1, 2, 3)
```

## str

In python every char is Unicode (not ASCII, ...).

Unicode: UTF-8, UTF-16, UTF-32.

UTF-8 is ASCII compatible.

# Lecture 4.2 - I/O

```py
import sys

sys.stdout.write("Hi")
sys.stderr.write("Hi")

print("Hi", end=' ', flush=True)

fd = open("smth.txt")
f.readline()            # reads until '\n'
f.close()               # implicit flush
```

# Lecture 5.1 - How Python Works

## 1. Source Code

`.py` file using human-readable syntax

## 2. Parsing & AST

Whole source code $\rightarrow$ **tokenized** + **parsed** to construct AST.

### AST

**Abstract Syntax Tree (AST)** in Python - representation of abstract syntactic structure of Python source code.

AST - intermediate representation that is closer to semantics of code but abstracted away from syntax details.  
AST is constructed for the _whole_ code at once, not line by line.

Compilers use ASTs for optimization before generating bytecode.

## 3. Bytecode Compilation

AST $\rightarrow$ bytecode.

**Bytecode** - a low-level, _platform-independent_ representation of code.

Bytecode is stored as `.pyc` files in the `__pycache__` directory for reuse.  
Bytecode instructions are executed one at a time, making it feel like Python is executing _line by line_.

## 4. Python Virtual Machine

**Python Virtual Machine (PVM)** - is a execution stack-based interpreter of bytecode.

**Runtime** starts after the bytecode is generated, during the execution by PVM.

# Lecture 5.2 - Namespaces

## Declaration

Variable cannot be defined without initializing it.  
Because Python is a dynamically typed language, and the type of a variable is inferred from the value assigned to it.

## Namespaces

**LEGB rule**: Local, Enclosing, Global, Built-in.

1. Built-In

   ```py
   print(dir(__builtins__))
   ```

2. Global

   ```py
   print(globals())
   ```

3. Enclosing

   ```py
   def outer_function():
       x = 1
       def inner_function():
           #nonlocal x  # Declare x from the enclosing scope as nonlocal
           print(x)  # This x is from the enclosing namespace
   ```

4. Local

   ```py
   print(locals())
   ```

# Lecture 5.3 - Decorators

## Decorator

```py
import warnings
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")  # Parameter specification
R = TypeVar("R")    # Return type

def deprecated(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        warnings.warn(
            f"Function '{func.__name__}' is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)
    return wrapper

@deprecated
def func(a: int, b: str) -> str:
    """This is an old function that will be deprecated."""
    return f"Received {a} and {b}"

func(42, "hello")
print(func.__doc__)
```

`wraps` preserves function's metadata.

`ParamSpec` preserves the parameter types of the original function.

`Typevar` defines a generic type.

## Class Decorator

```py
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.call_times = []  # Tracks the timestamps of recent calls

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            # Remove timestamps older than the allowed period
            self.call_times = [t for t in self.call_times if t > current_time - self.period]

            if len(self.call_times) >= self.max_calls:
                time_to_wait = self.period - (current_time - self.call_times[0])
                raise Exception(f"Rate limit exceeded. Try again in {time_to_wait:.2f} seconds.")

            self.call_times.append(current_time)
            return func(*args, **kwargs)
        return wrapper

# Example usage
@RateLimiter(max_calls=3, period=5)  # Allow 3 calls every 5 seconds
def greet(name):
    print(f"Hello, {name}!")

try:
    greet("Alice")
    greet("Bob")
    greet("Charlie")
    greet("David")                  # This should raise an exception
except Exception as e:
    print(e)                        # Rate limit exceeded. Try again in 4.98 seconds.
```

# Lecture 6 - Classes

type $\rightarrow$ class $\rightarrow$ instance

In python class of any instance is it's type:

```py
(1).__class__ == type(1)
```

class of class itself is `type`:

```py
class Smth:
    ...

print(Smth.__class__)   # type
```

## OOP in Python

**\+** **Encapsulation** - protecting the data by restricting direct access to it and allows controlled access through methods.

**\+** **Abstraction** - hiding the complex implementation details and exposing only the necessary functionalities.

**\+** **Inheritance** - creating new classes that inherit attributes and methods from existing classes.

**\+/\-** **Polymorphism** - ability to use the same interface or method name for different types or classes:

1. **\+** has dynamic polymorphism (`@abstractmethod`)
2. **\-** has no static polymorphism (overloading) (which is obvious due to duck typing of Python).  
   However, Python has _some_ overloading using `functools.singledispatch` based on the type of the 1st argument.

## Syntax

Static method `__new__` which is typically not touched creates `self` $\rightarrow$ `__init__()` initializes it.

```py
# class A:                          # Pure new class
class A(B):                         # Inheritence
    """Class docstring"""

    X = 10                          # Class variable; it's common to initialize `const` this way

    def __init__(self, x: int, y: int, z: int):     # Instance variable initialization (instance-specific state)
        self.x = x                  # public:       may be accessed outside class
        self._y = y                 # protected:    shouldn't be accessed outside class;                can be accessed by heir
        self.__z = z                # private:      can be accessed outside class only by `a._A__z`;    can't be accessed by heir

    # The same logic for methods:
    def public():
        ...

    def _protected():
        ...

    def __private():
        ...

    def foo(self) -> None:          # Instance method: operates on instance data
        self._x += 1

    @staticmethod
    def bar() -> str:               # Static method: does not receive the instance (self)
        return "bar"

    @classmethod
    def baz(cls: type) -> str:      # Class method: operates on class-level data
        return cls.X

    @property
    def y(self) -> int:             # Property method: getter for the instance variable
        return self._x              # use through: `a.y`

    @y.setter
    def y(self, x: int) -> None:    # Setter method for the property
        self._x = x                 # use through: `a.y = ...`

    def __hash__(self):             # Magick method
        ...


obj.__dict__                        # stores all attributes of the object `obj`.
cls.__dict__                        # stores all methods of the class `cls`.

a = A()
a.foo()                             # bound method  (`a` is `self`)
A.foo(a)                            # function      (takes `self`)
```

## `@dataclass`

```py
from dataclasses import dataclass

@dataclass
class Case:
    name: str = "default"
    given: int = 1
    expected: int
```

is equivalent for

```py
class Case:
    def __init__(self, name: str, given: int, expected: int):
        self.name = name
        self.given = given
        self.expected = expected

    def __eq__(self, other):
        return (self.name, self.given, self.expected) == (other.name, other.given, other.expected)

    def __repr__(self):
        return f"Case(name='{self.name}', given={self.given}, expected={self.expected})"
```

### options

```py
@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class A:
    ...
```

Hash is unsafe, because object can be changed $\rightarrow$ hash will change.  
But you can froze the object.  
BUT it will be frozen like tuple, so if attribute is mutable itself, then it can be changed either way.

## `@abstractmethod`

```py
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass  # No implementation

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
```

## `Enum`

Members of an `Enum` are singletons, while `Enum` itself is not a singleton (it can have multiple members).

`Enum` has no `__init__` and members of it are accessed like static.

```py
from enum import Enum

class A(Enum):
    X = "lol"
    Y = "kek"

ref1 = A.X
ref2 = A.X

ref1 == ref2    # True
ref1 is ref2    # True
```

### `unique`

```py
from enum import Enum, unique

@unqiue
class A(Enum):
    X = "lol"
    Y = "lol"   # error
    Z = "kek"
```

## Inheritance

### `super`

`super` enabling access to inherited methods and constructors.

```py
class Parent:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print("Hello from Parent!")

class Child(Parent):
    def __init__(self, name, gender):
        super().__init__(name)
        self.gender = gender

    def greet(self):
        super().greet()
        print("Hello from Child!")
```

### MRO

**Method Resolution Order (MRO)** - MRO determines the order in which Python searches for methods in class hierarchies:

```py
class A:
    def show(self): print("A")

class B(A):
    def show(self): print("B")

class C(A):
    def show(self): print("C")

class D(B, C):
    pass

print(D.mro())
# [<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
```

### `isinstance`

```py
a = A()
isinstance(a, A)            # True

isinstance(object, type)    # True
isinstance(type, object)    # True
```

### `issubclass`

```py
class A:
    ...

class B(A):
    ...

issubclass(A, B)            # True

issubclass(object, type)    # False
issubclass(type, object)    # True
```

# Lecture 7 - Exceptions

## Exception Structure

```txt
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── ArithmeticError
    │   └── ...
    ├── AssertionError
    ├── AttributeError
    ├── BufferError
    ├── EOFError
    ├── ImportError
    │   └── ...
    ├── LookupError
    │   └── ...
    ├── MemoryError
    ├── NameError
    │   └── ...
    ├── OSError
    │   └── ...
    ├── RuntimeError
    │   └── ...
    ├── SyntaxError
    │   └── ...
    ├── TypeError
    ├── ValueError
    │   └── ...
    └── Warning
        └── ...
```

```py
exc.__cause__
exc.__context__
exc.__traceback__
```

## Custom exceptions

```py
class NetworkError(Exception):
    pass

class TimeoutError(NetworkError):
    pass

raise TimeoutError("This is a custom exception!")
```

### `__str__` and `__init__`

You can override `__str__` or `__repr__` to customize how the exception is displayed.

```py
class CustomError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"[Error {self.code}]: {self.message}"

try:
    raise CustomError(404, "Resource not found")
except CustomError as e:
    print(e)    # Output: [Error 404]: Resource not found
```

## `ExceptionGroup`

```py
from exceptions import ExceptionGroup
```

```py
try:
    raise ExceptionGroup(
        "Batch processing errors",
        [ValueError("Invalid value"), KeyError("Missing key"), ValueError("Another invalid value")]
    )
except* ValueError as ve:               # handles all ValueError exceptions from the group
    print(f"Caught ValueErrors: {ve}")
except* KeyError as ke:                 # handles all KeyError exceptions from the group
    print(f"Caught KeyErrors: {ke}")
```

## `try`-`except`-`else`-`finally`

```py
try:
    file = open("example.txt", "r")
    content = file.read()
except FileNotFoundError:
    ...
except PermissionError:
    ...
except:
    ...
else:                                           # no exceptions occurred
    print("File read successfully. Content:")
    print(content)
finally:                                        # ensures the file is closed, even if another exception occurred
    file.close()

# file.close()                                  # won't execute if another exception occurs
```

## `raise`

```py
raise ValueError("Invalid input provided")
```

### Propagate error

Inside `except` block, you can re-raise the same exception to propagate it further.

```py
try:
    raise ValueError
except ValueError as e:
    raise                           # Re-raises the same ValueError
```

### `raise` ... `from`

```py
try:
    1 / 0
except ZeroDivisionError as e:
    raise ValueError("Invalid operation") from e
```

## context manager

A context manager allows you to allocate and release resources using `with`.

```py
with acquire_resource() as r:
    use_resource(r)
```

is syntax sugar of:

```py
manager = acquire_resource()
resource = manager.__enter__()
try:
    use_resource(resource)
finally:
    exc_type, exc_value, traceback = sys.exc_info()
    suppress = manager.__exit__(exc_type, exc_value, traceback)
    if exc_value is not None and not suppress:
        raise exc_value
```

### Syntax of context manager

- `__enter__`: before the code block begins; often used to set up a resource.
- `__exit__`: regardless of whether an exception occurred; often used for cleanup.

```py
from typing import Optional, Type, Any

class CustomContextManager:
    def __enter__(self) -> "CustomContextManager":
        print("Entering the context")
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[Any]) -> bool:
        print("Exiting the context")
        # Handle exceptions if needed
        if exc_type:
            print(f"Exception: {exc_value}")
        return True  # Suppress the exception
```

### `contextlib`

#### `contextmanager`

`@contextmanager` decorator allows you to write a context manager using a **generator** function.

```py
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Resource setup")

    try:
        yield "Resource"  # code inside `with` block
    finally:
        print("Resource cleanup")

with managed_resource() as res:
    ...
```

#### `ExitStack`

```py
from contextlib import ExitStack

with ExitStack() as stack:
    file1 = stack.enter_context(open("file1.txt", "r"))
    file2 = stack.enter_context(open("file2.txt", "r"))
    # Use both files
```

# Lecture 8 - Importing Modules \& Libraries

## Importing Modules

1. Search for the module in `sys.path`

2. Compilation to bytecode and storing it in a `.pyc` file in the `__pycache__` directory

3. Executing the bytecode to initialize the module, creating objects for its definitions

4. Caching module in the `sys.modules` dictionary: key is the module's name, and the value is the module object itself

Already imported module is imported (again) directly from `sys.modules`.

```py
import sys

sys.modules["sys"] is sys
```

### `sys.path`

`sys.path` is a list of directories that the Python interpreter searches for modules when importing

Python populates `sys.path` with default values at startup, which include:

- current working directory
- directories of env `PYTHONPATH`
- standard library directories
- site-packages directories for installed third-party packages from `pip`

The interpreter will import the first one it finds in `sys.path`.  
Reordering `sys.path` can help sometimes.

```py
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "custom_modules"))
```

### Private Variables of Module

All variable that start with `_` are not imported.

## `__main__`

```py
def main():
    local = ...

if __name__ == "__main__":
    variable_will_be_global = ...   # don't do like this, only main()
    main()
```

## `__init__.py`

Best practices:

- Keep `__init__.py` files as simple as possible.
- Explicit imports

```py
from .module1 import function1
from .module2 import function2

__all__ = ["function1", "function2"]
```

# Lecture 9 - Iterators & Generators

## Iterator

**Iterator**: implements `__iter__` (returning itself) and `__next__`.  
**Iterable**: implements `__iter__` returning iterator.

```py
l = [1, 2]
it = iter(l)
next(it)
```

### Iterator Initialization

When you can, it's best to implement also: `__len__`, `__getitem__`, `__contains__`.

#### 1. Custom Iterator

Write class with `__iter__` and `__next__`.

```py
class CounterIterator:
    def __init__(self, start=0, end=10):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        self.current += 1
        return self.current

it = CounterIterator(1, 10)
```

#### 2. Custom Callable

Write class with `__call__` || function $\rightarrow$ create iterator from instance of it.

```py
class Counter:
    def __init__(self, start=0):
        self.current = start

    def __call__(self):
        self.current += 1
        return self.current

counter = Counter()
it = iter(counter, 10)
```

### Functionality

Built-in:

- `enumerate`
- `zip`
- `map`
- `filter`

`itertools`:

- `chain`
- `tee`
- `groupby`

## Generators

**Generators** are iterators that generate values lazily using `yield`.

Does not execute in a single run (like a normal function):

- function runs until it reaches yield, returns the yielded value, and pauses execution
- next call resumes execution from where it left off

```py
from collections.abc import Iterator

def my_generator() -> Iterator[int]:
    yield 1
    yield 2

gen = my_generator()
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(next(gen))  # Raises StopIteration

gen = my_generator()
for i in gen:
    print(i)
```

Generators help do stuff without additional memory.

```py
#Generator expressions
max((i for i in range(10_000_000_000) if x % 11))   # 20 minutes and O(1) by memory
max([i for i in range(10_000_000_000) if x % 11])   # 20 minutes and O(n) by memory
```

### `yield from`

```py
def sub_generator():
    yield 1
    yield 2
    yield 3

def main_generator():
    yield "Start"
    yield from sub_generator()  # Delegates yielding to another generator
    yield "End"
```

### Generator State Management (send, close, throw)

<!-- https://chatgpt.com/c/679bc147-3394-8010-9972-ae958a909980 -->

### Generator as Coroutine

```py
def process(text: str, sleep_time: int) -> Generator[None, None, None]:
    next_time = monotonic()
    while True:
        now = monotonic()

        if now >= next_time:
            print(text)
            next_time = now + sleep_time

        yield

# [*process("task!", 1)]

def sheduler(*tasks: Generator) -> None:
    queue = deque(tasks)

    try:
        while queue:
            task = queue.pop()

            try:
                next(task)
            except StopIteration:
                continue
            else:
                queue.appendleft(task)

    except KeyboardInterrupt:
        return

sheduler(process("ping", 3.1), process("pong", 1))
```

# Lecture 10 - Typing

- Strong vs Weak typing:

  - **Strongly**: only explicit type conversions (Java, C#, Python, Pascal, TypeScript).
  - **Weakly**: allows implicit conversions (JavaScript, PHP, Perl, C).

- Static vs Dynamic typing:
  - **Statically**: type checking occurs at compile-time (C, C++, Java, Rust, Go).
  - **Dynamically**: type checking occurs at runtime (Python, JavaScript, Ruby, PHP).

Python - **dynamically & strongly** typed language. From python3.5 static typing was introduced.

```py
x = "10"
y = 5
print(x + y)  # TypeError in runtime
```

## Duck Typing

There is **nominal subtyping**: inheritance of classes.

**Duck Typing** is a **structural subtyping**:  
_"If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck."_

Python doesn't check a variable's type explicitly but rather whether it supports a given operation.

```py
def add(a, b):
    return a + b

# they both implement .__add__
add(10, 5)
add("Hello, ", "World!")
```

## `mypy`

`mypy` enables static (compile-time) type checking but does not enforce types at runtime.

```py
pip install mypy
mypy script.py
```

```py
import typing
lst = [1, 2, 3]
typing.reveal_type(lst)
```

## `typeguard`

Stricter runtime checking:

```py
from typeguard import typechecked

@typechecked
def add(a: int, b: int) -> int:
    return a + b

print(add("5", "10"))  # Raises TypeError at runtime
```

## Specials

```python
from typing import Any, Union, Optional

def func_any(x: Any) -> Any:
    return x

def func_union(x: Union[int, str]) -> str:
    return str(x)

def func_optional(x: Optional[int]) -> int:
    return x or 0
```

## Protocol Types

Utilizes duck typing: if it implements all the required methods, it's considered compatible.  
Does not verify method signatures, only that the attribute exists.

Again, no runtime errors, only for hints and static analysis via `mypy`.

```py
from typing import Protocol

class Flyer(Protocol):
    def fly(self) -> None:
        ...

class Bird:
    def fly(self) -> None:
        ...

class Car:
    def drive(self) -> None:
        ...

def takeoff(entity: Flyer) -> None:
    entity.fly()

bird = Bird()
takeoff(bird)  # ✅ Works because Bird has `fly()`

car = Car()
takeoff(car)  # ❌ Type error: Car does not implement `fly()`
```

## Generic Types

**Generic** _class_, _function_, or _protocol_ allows you to specify type parameters (instead of specific types).

`collections.abc` has some preimplemented protocols many of which are generics:

| **Protocol[T, V]**      | **Is Generic?** |
| ----------------------- | --------------- |
| `Iterable[T]`           | ✅ Yes          |
| `Iterator[T]`           | ✅ Yes          |
| `Collection[T]`         | ✅ Yes          |
| `Sequence[T]`           | ✅ Yes          |
| `MutableSequence[T]`    | ✅ Yes          |
| `Mapping[K, V]`         | ✅ Yes          |
| `MutableMapping[K, V]`  | ✅ Yes          |
| `Set[T]`                | ✅ Yes          |
| `MutableSet[T]`         | ✅ Yes          |
| `MappingView[T]`        | ✅ Yes          |
| `ItemsView[K, V]`       | ✅ Yes          |
| `KeysView[K]`           | ✅ Yes          |
| `ValuesView[V]`         | ✅ Yes          |
| `Callable[[T1, T2], R]` | ✅ Yes          |
| `Container[T]`          | ✅ Yes          |
| `Sized`                 | ❌ No           |
| `ByteString`            | ❌ No           |
| `Hashable`              | ❌ No           |

There are others.

### Creating Custom Generics and Protocol Types

```py
from typing import TypeVar, Generic, Protocol

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, item: T): self.item = item

class Reader(Protocol):
    def read(self) -> str: ...
```

## `typing.overload`

It doesn't affect runtime at all.  
All functions are get overwritten by the last.

Use @overload when:

- need precise static typing using `mypy`.
- want better autocompletion in IDEs.
- building public APIs or libraries.
- combine it with `singledispatch` for runtime overloading.

```py
from typing import overload

@overload
def repeat(value: int) -> int: ...
@overload
def repeat(value: str) -> str: ...

def repeat(value):
    if isinstance(value, int):
        return value * 2
    elif isinstance(value, str):
        return value + value

num = repeat(10)      # Type checker knows `num` is `int`
text = repeat("hi")   # Type checker knows `text` is `str`
```

# Lecture 11 - MapReduce

MapReduce - model for processing and generating large datasets that can be parallelized across a distributed cluster of computers.

The input data is divided into chunks, and each chunk is processed in parallel by _map & shuffle_ $\rightarrow$ grouped among chunks and processed by _reduce_.

MapReduce consists of:

1. **Map**: For each input key-value pair $ (k_i, v_i) $, transform it into an intermediate arbitrary ($0, 1, 2, \dots$) number of pairs $ (k'\_i, v'\_i) $:

   $$
   M(k_i, v_i) \rightarrow (k'_i, v'_i)
   $$

2. **Shuffle (Partitioning)**: Group all intermediate pairs by key:

   $$
   \{ k_1: [v_{11}, v_{12}, \dots], k_2: [v_{21}, v_{22}, \dots], \dots \}
   $$

3. **Reduce**: For each key $ k $, aggregate the values:

   $$
   R(k, \{v_1, v_2, \dots, v_m\}) \rightarrow (k, \text{aggregate}(v_1, v_2, \dots, v_m))
   $$

## Time Complexity

- **Map**: $ O(n) $, where $n$ is the number of input key-value pairs.
- **Shuffle**: $ O(n \log n) $, where $n$ is the number of key-value pairs being shuffled and sorted.
- **Reduce**: $ O(m) $, where $m$ is the number of distinct keys.

# Lecture 12 - Serialization

**Serialization** - converting an object into a format that can be easily stored or transmitted.

Serialized text:

- JSON
- YAML
- XML
- ...

Serialized byte stream:

- pickle
- parquet
- protobuff
- ...

## JSON

You can write custom encoder & decoder.

All keys in JSON are casted to `str`.

```py
with open("file.json", "r") as fin:
    .loads(fin)    # read line
    .load(fin)     # read whole

with open("file.json", "w") as fout:
    .dumps(data, fout)
    .dump(data, fout)
```

## YAML

YAML - superset of JSON from YAML1.2.

Supports tags and datetime.

```py
.safe_load()        # read as 1 YAML
.safe_load_all()    # break into several YAMLs
.safe_dump()
.safe_dump_all()
```

## `pickle`

`pickle` is Python's built-in module for serializing.

```py
import pickle
pickle.HIGHEST_PROTOCOL                 # version

with open("file.json", "rb") as f:      # b - binary
    .load()

with open("file.json", "wb") as f:      # b - binary
    .dump()
```

lambda can't be serealized.

Pickle serializes **object state**, but not the code or class definition.  
So it requires the class definition to be present in the scope where `load` or `loads` is called.

## Apache Parquet

**Apache Parquet** - **columnar storage** format optimized for doing analytical **queries** without need to load all the data.

Parquet supports nested data structures with complex types: list, map, struct

```py
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ParquetExample").getOrCreate()
df = spark.read.parquet("example.snappy.parquet")
df.filter(df.age > 30).show()
```

### Structure

1. Row Groups:

   - Logical horizontal partitioning of data.
   - Each row group contains column chunks.

2. Column Chunks:

   - Data for a particular column within a row group.
   - Stored contiguously to enable efficient columnar reads.

3. Pages:

   - Column chunks are further divided into pages.
   - Two types of pages:
     - Data Pages: Contain the actual column data.
     - Dictionary Pages: Contain encoding information.

4. File Footer:
   - Contains metadata about the file, schema, and statistics.
   - Stored at the end for efficient sequential reads.

### Data Encoding

- **Run-Length Encoding** (RLE):  
   Example: AAAAAABBBCC → A6B3C2

- Dictionary Encoding:  
   Stores unique values once and references them using indexes.

- **Delta Encoding**:  
   Stores differences between consecutive values.
  Useful for sorted columns (e.g., timestamps).

# Lecture 13.1 - Tests

The Three Laws of TDD (Test Driven Development) (by Robert C. Martin, aka Uncle Bob):

1. Write a **failing** test first
2. Write only enough test code to fail
3. Write only enough production code to pass

```py
import pytest

@pytest.mark.parametrize("name", ["Mike", "John", "Greg"])
def parametrized_test(input):
    assert ...

@pytest.mark.parametrize(
    "name",
    ["Mike", " Mike", "Mike ", " Mike ", "  Mike", " Mike  "],
    ids=["no spaces", "left space", "right space", "two-side space", "double space", "two-sided double space"]
)
def parametrized_test_with_ids(input):
    assert not ...
```

# Lecture 13.2 - Logging

## Logger Tree

When you create a logger, it is added to the **logger tree**.

The logger's path in the tree is determined by its name.  
If a logger is named `a.b.c`, it will become a child logger of `b`, which in turn is a child logger of `a`.  
The `a` logger is a child of the **root logger**, which always exists by default.

If you create a logger and do not configure it in any way, it will inherit the settings of its parent.

BEST PRACTICE:

```python
logger = logging.getLogger(__name__)
```

The module hierarchy will match the logger hierarchy, making it intuitively clear where a message originated.
This also makes it easy to configure logging in the `__init__.py` file.

## Logging flow

![alt text](notes_images/logging.png)

# Lecture 14 - Protobuf & Flatbuf

Protocol Buffers (Protobuf) & Flat Buffers (Flatbuf) - language-neutral, platform-neutral serialization libraries developed by Google.

Key Features of both:

- Requires defining data structures in a `.proto`/`.fbs` file.

- Cross-language support.

- Backward & forward version compatibility.

## Protobuf

**Protobuf** - general-purpose binary serialization format designed for efficient data transmission and storage.

Protocol buffers provide a serialization format for packets of typed, structured data that are up to a _few megabytes_ in size.  
The format is suitable for both ephemeral network traffic and long-term data storage.

✅ Much smaller than JSON and faster, but requires parsing.

✅ Can be streamed efficiently over networks.

❌ No direct random access to serialized data (all fields must be parsed).

> Used for: gRPC, microservices, or APIs.

Example of `.proto` file:

```proto
edition = "2023";

package tutorial;

option features.field_presence = EXPLICIT;

message Person {
  string name = 1;
  int32 id = 2;
  string email = 3;

  enum PhoneType {
    PHONE_TYPE_UNSPECIFIED = 0;
    PHONE_TYPE_MOBILE = 1;
    PHONE_TYPE_HOME = 2;
    PHONE_TYPE_WORK = 3;
  }

  message PhoneNumber {
    string number = 1;
    PhoneType type = 2 [default = PHONE_TYPE_HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

## Flatbuf

**Zero-copy access** - does not require deserialization before accessing fields.

**Flatbuf** - high-performance serialization library optimized for zero-copy access.

✅ Very fast random access: Data can be read in-place directly from memory.

✅ Zero-copy

❌ More complex schema compared to Protobuf

> Used for: real-time applications (e.g., gaming, IoT, VR).

---

SKIPPED MOST OF THE LECTURE

# Lecture 15 - DB

NEED TO WATCH

# Lecture 16 - API

## HTTP Methods

> Use POST instead of GET, because POST guarantees that HTTP's body won't be lost, while GET does not.

- GET = Read  
- POST = Create  
- PUT = Save  
- PATCH = Update  
- DELETE = Delete  

- GET — safe, you can retry it without worries  
- GET, PUT, DELETE — _idempotent_. If something breaks, you can safely retry  
- POST, PATCH — unsafe, they might change something. Think before retrying  

## REST API

**REST API** - web service interface that conforms to the architectural constraints of REST:

1. Client–Server:
    - Separation of concerns between the client (user interface) and the server (data storage and logic).

2. Stateless:

3. Cacheable:
    - Responses must explicitly indicate whether they are cacheable or not to improve performance and scalability.

4. Uniform Interface (core principle of REST):
    - Resource identification in requests (e.g., /users/1)
    - Resource manipulation via representations (typically JSON or XML)
    - Self-descriptive messages (the request includes all necessary info)
    - Hypermedia as the engine of application state (HATEOAS)

There are many different libraries and frameworks for building **REST APIs**:

1. AIOHTTP
2. Django, Django Rest
3. Flask, Flask RESTX (formerly Flask-RESTplus), etc.
4. Falcon
5. Pyramid
6. FastAPI $\leftarrow$ _use it_

# Lecture 17 - Optimization

## GIL

**Global Interpreter Lock (GIL)** - a mutex (lock) that protects access to Python objects, preventing multiple native threads from executing Python bytecode simultaneously.

1. Only one thread can execute Python bytecode at a time.
2. The GIL switches between threads periodically (after a few milliseconds).
3. If one thread is waiting (I/O operations), another thread gets control.

Python uses a GIL switch interval to allow context switching between threads, but it does not allow _true parallel execution_ of Python code.

Some libraries like NumPy, TensorFlow, and Numba bypass the GIL by using **C extensions** with native threading.

### `threads`

- Python threads great with **I/O-bound** tasks

```py
import threading

threads = [threading.Thread(target=my_func, args=(url,)) for args in list_of_args]

for t in threads: t.start()
for t in threads: t.join()
```

### `multiprocessing`

- Python threads do not help with **CPU-bound** tasks.

Since _each Python process_ has _its own_ GIL, multiple processes can run in _true parallelism_.

Use `multiprocessing` instead of `threads`:

```py
from multiprocessing import Pool

with Pool(4) as p:  # Uses 4 CPU cores
    p.map(my_func, args)
```

```py
from multiprocessing import Process

p1 = Process(target=my_func)
p2 = Process(target=my_func)

p1.start()
p2.start()
p1.join()
p2.join()
```

## Profiling

**Profiling** - process of measuring the execution time and resource usage of a program.

### cProfile

```bash
python -m cProfile script.py
```

or directly in script:

```py
import cProfile

def test_function():
    sum([i**2 for i in range(10000)])

cProfile.run('test_function()')
```

### pstats

**pstats** allows sorting and filtering of cProfile results for better readability.

```py
import cProfile
import pstats

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

cProfile.run('fib(20)', 'fib_profile.prof')

stats = pstats.Stats('fib_profile.prof')
stats.strip_dirs().sort_stats('tottime').print_stats(5)
```

### SnakeViz

**SnakeViz** - tool that provides a graphical visualization of profiling data.

CONTINUE FROM 0:42:00
