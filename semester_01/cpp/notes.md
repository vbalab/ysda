<!-- markdownlint-disable MD024 MD025 MD049 -->

# Set up

- <https://gitlab.manytask.org/cpp0/public-2024-fall/-/tree/main/multiplication>
- <https://gitlab.manytask.org/cpp0/public-2024-fall/-/blob/main/docs/setup.md>
- <https://docs.google.com/document/d/1K0t05Bmqb3he3gW4ORQXfkVfFouS4FRT/edit>

2025:

- [Setup](https://gitlab.manytask.org/cpp/public-2025-spring/-/blob/main/docs/SETUP.md)
  <https://gitlab.manytask.org/cpp/public-2025-spring/-/tree/main/range-sum>

## Docker

```Dockerfile
FROM ubuntu:24.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y gnupg
RUN apt update && apt -y upgrade

RUN apt install -y \
    build-essential \
    cmake \
    ccache \
    ninja-build \
    g++ \
    clang-19 \
    clang-format-19 \
    clang-tidy-19 \
    libclang-19-dev

RUN apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git

RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    libpoco-dev \
    libjsoncpp-dev \
    libboost-dev \
    libre2-dev \
    libpng-dev \
    libjpeg-dev \
    libfftw3-dev \
    libedit-dev \
    libtbb-dev \
    protobuf-compiler \
    libprotoc-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    m4 \
    firejail

RUN apt clean && rm -rf /var/lib/apt/lists/*
```

```bash
docker build -t checker .

docker run -it --rm   -v ~/all/study/shad/semester_02/cpp/vbalab:/vbalab   -w /vbalab   checker   bash
```

Inside the container:

```bash
rm -rf build
mkdir build
cmake ../
make test_...
./test_...
```

## How to set up linter (on Fedora Linux)

```sh
sudo dnf install clang clang-tools-extra
```

Then at run_linter.sh remove "-16" everywhere (from clang-format-16, clang-tidy-16)

# Tasks

## Pull

In vbalab do:

```sh
git pull upstream main
git pull --rebase upstream main
```

OR [*I don't know why, but origin works*]

```sh
git pull --rebase origin main
```

to update tasks.

## Debug

In lower panel of VScode select test_NAME_OF_TASK and press bug.

## Tests & Benchmarks

### Setup

To create build for benchmarks do:

```sh
mkdir build
cd build
cmake ..
```

```sh
mkdir build-asan
cd build-asan
cmake -DCMAKE_BUILD_TYPE=Asan ..
```

```sh
mkdir build-tsan
cd build-tsan
cmake -DCMAKE_BUILD_TYPE=Tsan ..
```

For benchmarks:

```sh
mkdir build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

### Run Tests

If no particular test_NAME_OF_TASK (being in "build-..." directory):

```sh
cmake .
# OR
cmake -DCMAKE_BUILD_TYPE=Asan .
# OR
cmake -DCMAKE_BUILD_TYPE=Tsan .
```

to run tests:

```sh
make test_NAME_OF_TASK
./test_NAME_OF_TASK
./test_NAME_OF_TASK "Name of the test among all tests"
```

### Run Benchmarks

If no particular bench_NAME_OF_TASK:

```sh
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .
```

To run:

```sh
make bench_NAME_OF_TASK
./bench_NAME_OF_TASK
```

## Linter

to run linter (being in "build" directory):

```sh
../run_linter.sh NAME_OF_TASK/
```

## Push

to add to gitlab:

```sh
git add ../NAME_OF_TASK/NAME_OF_TASK.h
git commit -m "solve NAME_OF_TASK, try 1"
git push origin main
```

# Notes

**C-style string** is `const char*`

---

`size_t` is a platform-dependent (32 or 64), unsigned type used for representing sizes and counts.

---

For small primitives using direct copying like `size_t` is better than `const size_t&` which goes with:

- Additional Indirection
- Potential Cache Misses
- Aliasing

---

`std::any` - a type-safe way to store and retrieve values of any type, using type erasure.

---

In `std::cout` use explicit **flushing** through `std::flush` or `std::endl` when you want to ensure the output is immediately displayed, such as in debugging or interactive applications.

`std::cerr` is **unbuffered**, it doesn’t need explicit flushing.

---

-`std::map` is an ordered associative container that uses **Red-Black Tree** for storage.

-`std::unordered_map` is a hash-based container that requires a **hash function** to determine key uniqueness.

---

`int` overflow - **UB**: could result in anything from crashing the program to returning a wrapped-around value, depending on C++ version and computer architecture.

`uint` overflow - well-defined and causes the value to wrap around.

---

By definition of signed integer: `-a == (~a) + 1`

# Lecture 1

Nothing interesting.

# Lecture 2 - RVO

## Return Value Optimization (C++17)

**RVO**: Optimizes away the copy for temporary objects returned by value.

Say you have

```cpp
struct A {
    A() {
        std::cout << "Constructor\n";
    }
    A(const A&) {
        std::cout << "Copy Constructor\n";
    }
    ~A() {
        std::cout << "Destructor\n";
    }
};

A CreateA() {
    return A();
}

int main() {
    A obj = CreateA();  // No copy or move constructor will be called due to RVO
}
```

Output without RVO (if RVO were not applied):

```txt
Constructor
Copy Constructor
Destructor          // of original
Destructor          // of copy
```

With RVO:

```txt
Constructor
Destructor
```

### Named RVO

RVO did for:

```cpp
template <typename T>
T f() {
    return T(); // **unnamed** temporary object
}
```

**NRVO** allows for:

```cpp
template <typename T>
T f() {
    T x();
    return x;   // **named** object
}
```

## .hpp & .cpp

- declaration $\rightarrow$ `.hpp` (headers):  
  Includes necessary `#include` directives and header guards (`#pragma once` or `#ifndef` ... `#define`) and does only declaration.

- definitions $\rightarrow$ `.cpp`:  
  Includes the corresponding header file and does all implementation of declared in `.h`.

**Linker** is a tool that combines one or more object files generated by a compiler into a single executable or library.

If `.hpp` gets changed $\rightarrow$ everything in project gets recompiled.  
If `.cpp` gets changed $\rightarrow$ recompiled only 1 file & linker does the job.

# Lecture 3 - Structs & Classes

## don't do function creation

**TQ**:

```cpp
T x(1);    // T(int)
T x{1};    // T(int)

T x;    // T()
T x();  // function! - be aware
T x{};  // T()
```

> $\to$ Always create using `{}`;

## `class` vs `struct`

| **Property**                       | **struct**                   | **class**                                   |
| ---------------------------------- | ---------------------------- | ------------------------------------------- |
| Default member & base-class access | public                       | private                                     |
| Default inheritance access         | public                       | private                                     |
| Typical convention                 | Plain-data aggregates or POD | Encapsulated types with invariants/behavior |

## `const` & `mutable`

```cpp
struct MyStruct {
    int value;
    mutable size_t count;  // to bypass `const` of method

    int getValueConst() const { // `const`: Cannot modify member variables
        count++;                // ok
        value++;                // CE
        return value;
    }

    int getValue() const {
        return value;
    }
};
```

## Memory Initializer List

```cpp
struct S {
    int v1_;
    const int v2_;

    S(int val1, int val2)
        // Memory Initializer List:
        // `const` can be initialized only here
        : v1_(val1)
        , v2_(val2) {
        // at this stage `S` is already initialized and only changes attributes
        v1_ = v1 + v2;
    }
}
```

## `static`

```cpp
struct S {
    constexpr static size_t kZero = 0; // "a member with an **in-class initializer** must be `const`"
    static size_t init_count;       // attribute of class `S`, not instance of `S`

    S() {
        ++init_count;
    }
}

size_t S::init_count = 0;
S s1{};                    // init_count = 1
S s2{};                    // init_count = 2
```

## `delete`

The **delete** keyword can be used to explicitly disable certain functionsv (like copying, moving, or assignment).

```cpp
class S {
    S() = default;

    S(const S&) = delete;               // Disable copy constructor
    S(S&&) = delete;                    // Disable move constructor

    S& operator=(const S&) = delete;    // Disable copy assignment
    S& operator=(S&&) = delete;         // Disable move assignment
};
```

# Lecture 4.1 - Overloading & Templates

Check out Lecture 23, 24 of Meshcherin Notes.

# Lecture 4.2 - Functors & Lambdas

## Functors

**Functor** in C++ - instance of a `class`/`struct` that has `operator()` overloaded.

```cpp
struct Functor {
    int factor;

    Functor(int f) : factor(f) {}

    int operator()(int x) const {
        return x * factor;
    }
};
```

Functions are _not functors_.

## Lambda

**Lambda** - _unnamed_ function object (i.e., a closure).

After compilation lambda function becomes **unnamed functor**.

Syntax: `[ capture ] ( parameters ) -> ReturnType { body }`.

### Capture

**Capture** - variables from the surrounding scope to capture as _members_ of compiled unnamed functor.

- `[ ]`: No capture (only parameters can be used).
- `[=]`: Capture all variables by value.
- `[&]`: Capture all variables by reference.
- `[x]`: Capture x by value.
- `[&x]`: Capture x by reference.

### `mutable`

By default, a lambda’s `operator()()` is `const`.

```cpp
int i = 0;

auto count = [i]() { 
    ++i;    // CE
};

auto count = [i]() mutable {
    ++i;
};

auto count = [&i]() {
    ++i;    // ok, because member is `int& i`
};
```

### why `auto`?

C++ doesn't have a lambda type because lambdas are implemented as **anonymous (unnamed) functors (class)**, so their types are often complex and compiler-generated, making them difficult to explicitly specify.

## Higher-Order Functions

**Higher-order functions** are typically implemented using:

- function pointer (see Meshcherin Notes, Lecture 8)
- `std::function`
- lambda
- functor

# Lecture 4.3 - Iterators

## Iterators

```cpp
std::next(begin)
++begin;    // the same
```

1. **Input**

   - _Read-only_ & _single pass_
   - Supports `operator*` and `operator++`

   - _Use case_: reading from input streams.

2. **Forward** = Input + allows multiple passes

   - `std::unordered_map` & `std::unordered_set`

3. **Bidirectional** = Forward + backward (`operator--`)

   - support `==`, `!=` comparison
   - `std::map` & `std::set` & `std::list`

4. **Random Access** = Bidirectional + const time for `it + n` or `it[n]`

   - support `<`, `>`, `<=`, `>=` comparison
   - `std::deque`.

5. **Contiguous** = Random + elements are stored sequentially in memory - **Acts like a raw pointer**

   - `std::vector`, `std::array`, raw pointer

6. **Output**
   - _Write-only_ & _single pass_
   - Use case: writing to output streams.

## `ranges`

**Ranges** (C++20) provide a more modern, flexible way to work with sequences of data, offering a powerful alternative to the traditional iterator-based approach.

The main difference between ranges & iterators is that types of range for the start and end may be different.

C++20 provides range-based versions of algorithms (like `std::ranges::sort`, `std::ranges::find`), which operate directly on ranges.

### `views`

`ranges::views` - powerful way to create **lazy** **non-owning** adaptors over ranges.

```cpp
std::vector<int> vec;

for (int x : std::views::filter(vec, [](int x) { return x % 2 == 0; })) {
    std::cout << x << " ";
}
```

### Chaining

```cpp
std::vector<int> vec;

for (auto x: vec | std::views::filter([](int x) { return x % 2 == 0; })
                 | std::views::transform([](int x) { return x * x; })
                 | std::views::take(3)) {
    std::cout << x << " ";
}
```

# Lecture 5 - Move Semantics

## lvalue & rvalue

Object **has identity** if both:

1. It occupies a distinct region of storage in memory
2. Persists beyond a single expression

Object **can be moved** if both:

1. It has a move constructor or move assignment operator defined (either explicitly or implicitly)

2. The object is used in a context that allows moving — i.e., it’s an rvalue or xvalue (e.g., `std::move(x)`).

> If no move constructor is defined, the compiler may _fall back_ to a **copy**.

### Value Categories

1. **glvalue** = lvalue || xvalue:

   - **lvalue** — named object in memory (has identity, persistent storage)

   ```cpp
   int x = 5;
   int& ref = x;
   x;          // lvalue
   ref;        // lvalue
   ```

   - xvalue

2. **rvalue** = prvalue || xvalue:

   - **prvalue** — pure temporary value, no identity

   ```cpp
   42;                 // prvalue
   std::string("hi");  // prvalue
   x + y;              // prvalue if x, y are ints
   ```

   - xvalue

3. **xvalue** = glvalue (has identity) && rvalue (can be moved) — movable, _expiring object_:

   ```cpp
   std::move(x);  // xvalue
   a[0];          // xvalue if a is std::vector<T>
   ```

### `++`

- `++a` $\leftarrow$ lvalue;

- `a++` $\leftarrow$ rvalue (creates a copy, does increment to `a`, returns copy);

### `std::move`

- See "can be moved" definition.

```cpp
template<typename T>
constexpr typename std::remove_reference<T>::type&& move(T&& t) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(t);
}
// The cast to `&&` takes Move Constructor OR Move Assignment Operator depending on context
```

Objects that can't be moved (and no copy will be taken) (CE):

1. deleted move constructors

2. holding resources that cannot be transferred (e.g., certain OS handles)

3. `const` — since you can’t modify (move from) them

#### Funny `std::move`

```cpp
struct Thing {
    int val;

    Thing(Thing&& other) noexcept {
        val = 100;    // funny, and std::move will be funny
        // value = std::move(other.value); nope))
        other.val = 10;
    }
};

Thing t1(0);
Thing t2 = std::move(t1); // invokes move constructor
// t1.val == 10
// t2.val == 100
// LOL
```

## lvalue to rvalue conversion

```cpp
T x = y;
// x.operator=(static_cast<T>(y));  // what really happens
```

- x — **LHS** — modifiable (non-const) lvalue
- y — **RHS** — lvalue-to-rvalue conversion applies

**lvalue-to-rvalue conversion**: If an lvalue expression is used in RHS, it is converted to a _prvalue_.

```cpp
int a = 10;
int b = a;      // lvalue-to-rvalue conversion
int c = a + b   // lvalue-to-rvalue conversion of both `a` and `b`

void g(int);
g(a);           // lvalue-to-rvalue conversion

int& ref = a;   // no conversion
```

> Don't confuse with `std::move` that casts it to _xvalue_ && moves when you do `x = std::move(y)`

## Rule of 3, 5, 0

If a class defines one of the following, it should define all 3,5:

- Destructor
- Copy Constructor
- Copy Assignment Operator
- Move Constructor
- Move Assignment Operator

**Rule of 0**: developers should avoid manually defining any of the special member functions whenever possible, leveraging automatic resource management techniques provided by C++ standard library utilities such as `std::unique_ptr`, `std::shared_ptr`, and other **RAII** (Resource Acquisition Is Initialization) patterns.

```cpp
class String {
public:
    String(const char* str = "") : data_(new char[std::strlen(str) + 1]) {
        std::strcpy(data_, str);
    }

    // Destructor
    ~String() {
        delete[] data_;
    }

    // Copy Constructor
    String(const String& other) : data_(new char[std::strlen(other.data_) + 1]) {
        std::strcpy(data_, other.data_);  // or std::memcpy()
    }

    // Copy Assignment Operator
    // since it results in `Class&` as return type -> `(x = y) = z` is the same as `(x = z)`
    String& operator=(const String& other) {
        if (this == &other) {
            return *this;
        }

        delete[] data_;  // Clean up existing data

        data_ = new char[std::strlen(other.data_) + 1];
        std::strcpy(data_, other.data_);

        // OR:
        // String temp(other);
        // std::swap(data_, temp.data_);    // OR EVEN: std::swap(*this, temp);
        return *this;
    }

    // Move Constructor
    String(String&& other) noexcept : size(other.size), data_(other.data_) {  // no `const`, since it's already rvalue
        // so when other is deleted attribes of 'this' (which now share memory) won't get affected
        other.data_ = nullptr;
        // done to prevent **Double Deletion**: After moving data_ from other to this, both this and other would point to the same memory. When other is destroyed, its destructor would attempt to delete[] data_
    }

    // Move Assignment Operator
    String& operator=(String&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        delete[] data_;

        size = other.size;
        data_ = other.data_;

        other.data_ = nullptr;

        // OR:
        // std::swap(size, other.size);
        // std::swap(data_, other.data_);

        return *this;
    }

private:
    char* data_;
};


int main() {
    String s1("Hello");

    String s2 = s1;  // copy constructor
    String s3(std::move(s1));  // move constructor

    String s4;
    s4 = s2;  // copy assignment
    s4 = std::move(s3);  // move assignment

    return 0;
}
```

By declaring a function as `noexcept`, you inform the compiler (and anyone reading the code) that the function will not throw an exception under any circumstances.

## Construction Delegation

**Delegating constructor** $\to$ constructor.

```cpp
struct Header {
    uint8_t version;
    uint8_t command;
    uint16_t port;
    uint32_t ip;

    Header(uint16_t port, uint32_t ip) : version(4), command(1), port(port), ip(ip) {
    }

    Header(const cactus::SocketAddress& endpoint) : Header(endpoint.GetPort(), endpoint.GetIp()) {  // constructor delegation
    }
};
```

## Reference Collapsing

**Reference collapsing** dictates how references behave when combined, especially when working with template parameters.

Reference Type $\rightarrow$ Result:

- & + & = &
- & + && (&& + &) = &
- && + && = &&

```cpp
template <typename T>
void foo(T&& arg) {
    bar(std::forward<T>(arg));
}

void bar(int& x) {};

void bar(int&& x) {};

int x = 10;
foo(x);             // T becomes int&
foo(std::move(x))   // T becomes int
foo(5);             // T becomes int
```

`std::remove_reference` strips references (& or &&) and gives just the type T.

### `std::forward`

```cpp
#include <type_traits>

template <typename T>
T&& forward(typename std::remove_reference<T>::type& param) {
    return static_cast<T&&>(param);
}
```

This allows for **perfect forwarding** with `std::forward` that preserves the value category (lvalue or rvalue) of an argument.

## Type Deduction

```cpp
int a = 10;
decltype(a) b = 20;  // int
auto c = 30;         // int
```

Type deduction determines the type of a variable or template parameter automatically, based on the context.

# Lecture 6 - Dynamic Memory Allocation

## Dangling Reference (UB)

After freeing variable all pointers to it are trash.

```cpp
int& f() {
    int x = 0;
    return x;
}

int* f() {
    int x = 0;
    return &x;
}
```

## Types of C++ Memory

### static/global memory

- allocated at start of program and deallocated at the end
- not limited by scope

Syntax: `static` keyword or global scope

### stack memory

- static memory allocation at compile time
- automatically freed and lifetime is limited to scope
- mapped continously in memory

### heap memory

- dynamic memory allocation at runtime
- requires manual management
- **fragmentation** and indirect access

Syntax: `new T` or `new T(args)`, `delete p`; `new T[n]` and `delete[] p`.  
_Each_ `new` should have it's `delete`.

Modern OS reclaim memory when the program ends, including any dynamically allocated memory on the heap.

new (C++) $\rightarrow$ malloc (C) $\rightarrow$ mmap (asm)

## Array overhead

**Array overhead** is `size_t` stored (in memory) before array as metadata for size of array.  
That's how `delete []` knows how much to deallocate.  
!!!NO!!! it's not like that: see "akos10.2 Pointer Meta Info"

The metadata is managed internally by the memory allocator, and it is not exposed to the user, so I can't access size of array.

If many small arrays are allocated, the overhead can accumulate and become a problem. Can be solved using **memory pools** or single contiguous allocation managed manually.

```cpp
struct T {
    int i;
    ~T() {};    // !!
};

T* create_T(const size_t& n) {
    return new T[n];
}

void destroy_T(T* p) {
    delete[] p;
}
```

## Placement `new`

**Placement** `new` - construct object on _pre-allocated_ memory buffer.

```cpp
alignas(T) char buffer[sizeof(T)];

T* obj = new(buffer) T(42);             // constructs T at buffer's address

obj->~T();                              // you **must** destruct manually
// delete obj;                          // no allocation memory -> do NOT `delete`
```

> Use `alignas` for ensuring proper memory alignment with object's type.

## Causes of memory leaks

```cpp
{
    int* ptr = new int{5};  // 1. not freeing
}

int* ptr1 = new int{5};
ptr1 = new int{6};          // 2. losing references
```

There are some others like: circular references, ...

## Fragmentation

**Fragmentation** occurs due to the way dynamic memory allocation is handled by the standard heap allocator.

### External fragmentation

When free memory is scattered in small blocks across the heap, making it difficult to allocate large contiguous memory blocks, even if the total free memory is sufficient.

```txt
[Allocated] [Free] [Allocated] [Free] [Allocated]
```

### Internal fragmentation

When memory allocated to a program is larger than what the program actually needs, resulting in wasted space within allocated blocks.

For example - because of **alignment requirement** extra unused space is allocated.

## Allocation

There are a lot of different allocators.  
Here presented basic ones.

### Linear allocation (dynamic)

Memory is allocated sequentially in a contiguous block.

`+` fast and simple, minimizes external fragmentation.  
`-` no deallocation; memory must be reset in bulk.

### Stack allocation (static)

(!= memory on stack)

When a function is called, memory for local variables is allocated on the stack. When the function returns, the stack frame is automatically deallocated.

Since stack $\rightarrow$ uses **LIFO**.

**Padding** (in stack allocation) - extra space added between variables or within structures to satisfy alignment requirements.

```cpp
struct Example {
    char a;     // 1 byte
    int b;      // 4 bytes (requires 4-byte alignment)
    char c;     // 1 byte
};
// Padding:
// 3 bytes after 'a' to align 'b'
// 3 bytes after 'c' to align the structure size.
```

Total size of `struct` should be a multiple of the largest alignment requirement of its members.  
It needs for ensuring that when array of Example structs is created, each struct starts at an address aligned to the highest alignment requirement.

### Pool Allocation (dynamic)

Memory is allocated from a pre-allocated pool of fixed-size blocks, rather than allocating memory directly from the system heap.

### std::allocator

`std::allocator` is a default allocator in cpp unless a custom allocator is specified.

In performance-critical applications, custom allocators for **performance tuning** can be used to improve allocation performance or control memory layout.

```cpp
template <typename T>
struct allocator {
    T* allocate(size_t);
    void deallocate(T*, size_t);
    void constructor(T*, args);
    void destroy(T*);
}
```

# Lecture 7 - Smart Pointers

## RAII

**Resource Acquisition Is Initialization (RAII)** - C++ idiom where resource management is tied to the lifecycle of an object.

Many say that using exceptions should be forbidden due to situations like:

```cpp
void f() {
    int* p = new int(0);
    throw;
    delete p;       // doesn't get to it.
}

struct S {
    int* p;

    S(int x) : p(new int(x)) {
        throw;
        // `S` isn't initialized fully, so destructor won't be used
    }

    ~S() {
        delete p;   // doesn't get to it.
    }
};
```

> Use **smart pointers** for ensuring **RAII**, because in their destructors they `delete` raw pointers.

## std smart pointers

## `std::unique_ptr`

- Ownership Model: Owns a resource exclusively. No other unique_ptr can manage the same resource.
- Move-Only: unique_ptr cannot be copied, only moved, to transfer ownership.
- Memory Management: Calls delete on the managed object when it goes out of scope.

```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(10);
```

### with arrays

```cpp
std::unique_ptr<int[]> arr(new int[10]);

std::shared_ptr<int> arr(new int[10], std::default_delete<int[]>());
```

### deleter

You have to set delete for arrays:

```cpp
std::unique_ptr<int[]> arr(new int[5]{1, 2, 3, 4, 5}, std::default_delete<int[]>());
```

You can create custom deleter:

```cpp
void customArrayDeleter(int* p) {
    std::cout << "Custom deleter called for array" << '\n';
    delete[] p;
}

std::unique_ptr<int[], decltype(&customArrayDeleter)> arr(new int[5]{1, 2, 3, 4, 5}, customArrayDeleter);
```

### class initialization

Since `std::unique_ptr` is non-copyable, you can't pass it by value, but you can pass it by rvalue reference or by `std::move`ing it into the constructor.

```cpp
class S {
public:
    S(std::unique_ptr<int> p) : ptr(std::move(p)) {}
private:
    std::unique_ptr<int> ptr;
};
```

## `std::shared_ptr`

- Ownership Model: Allows multiple smart pointers to share ownership of a single resource.
- Thread-Safe: The reference count is managed in a thread-safe manner to allow use in multithreaded programs.

Uses a **control block** of:

- Reference count
- Weak count

When the reference count drops to zero, the resource is deallocated, but the control block persists until the weak count reaches zero, ensuring weak_ptr instances can still check for the resource's validity.

```cpp
std::shared_ptr<int> shared1 = std::make_shared<int>(20);
std::shared_ptr<int> shared2 = shared1;
```

### deleter

```cpp
void customArrayDeleter(int* p) {
    std::cout << "Custom deleter called for array\n";
    delete[] p;
}

std::shared_ptr<int> arr(new int[5]{1, 2, 3, 4, 5}, customArrayDeleter);
```

### `std::enable_shared_from_this`

Problem:

```cpp
class S {
public:
    std::shared_ptr<S> getShared() {
        return std::shared_ptr<S>(this);  // Incorrect!
    }
    ~S() {
        std::cout << "S destroyed\n";
    }
};

int main() {
    std::shared_ptr<S> ptr1 = std::make_shared<S>();
    std::shared_ptr<S> ptr2 = ptr1->getShared();  // Separate reference count!

    // `ptr1` and `ptr2` will try to delete `S` separately, leading to undefined behavior
    return 0;
}
```

The way:

```cpp
class S : public std::enable_shared_from_this<S> {
public:
    std::shared_ptr<S> getShared() {
        return shared_from_this();  // Uses the existing control block
    }
};
```

Avoid calling `shared_from_this()` in constructor, because it will cause UB since the control block is not yet fully set up.

## `std::weak_ptr`

- Non-Owning Reference: Does not increase the reference count.
- Deletion: Gets deleted when last owning pointer was deleted.
- Observation: Can observe and access the resource managed by shared_ptr without owning it.

Uses `lock()` to create a _temporary_ shared_ptr that increments the reference count if the resource is still available.

# Lecture 8 - OOP

Definition $\rightarrow$ Declaration $\rightarrow$ Initialization

## name lookup

### unqulified (~~::~~) name lookup

Global & Static $\rightarrow$ Namespace $\rightarrow$ { Block } $\rightarrow$ Class/Struct

```cpp
std::string_view x = "global";

namespace N {
    auto x = "namespace";

    void f() {
        auto x = "function";
        {
            std::string x = "block";
        }
    }
}
```

### qulified (::) name lookup

```cpp
const static int x = 5;

namespace N {
    struct A {
        struct B {
            const static int x = 10;
        };
    };
}

int main() {
    const int x = 3;

    std::cout << N::A::B::x;    // 10
    std::cout << ::x;           // 5
}
```

## Anonymous Namespace

- namespace without a name:

```cpp
namespace {
    int x = 1;
}
```

- static:

```cpp
static const int x = 1;
```

Variables from them can be accessed only _within_ the file they are declared.

## Inheritance

```cpp
class Animal {
public:
    void makeSound() const {
        std::cout << "Some generic animal sound" << '\n';
    }
};

class Dog : public Animal {
public:
    void makeSound() const {
        std::cout << "Woof!" << '\n';
    }
};
```

### Multiple Inheritance

```cpp
class Clickable { ... };
class Rectangle { ... };

class Button : public Clickable, public Rectangle { ... };
```

### `final`

`final` = "No further inheritance allowed"

```cpp
class ExecutionContext final : private ITrampoline { ... }
```

## Polymorphism (Static)

- Defining multiple functions with the same name but different parameters
- **Operator Overloading**

## Polymorphism (Dynamic)

### `virtual`

`final` $\to$ `override` $\to$ `virtual`.

- `final` = `final` + `override` + `virtual`
- `override` = `override` + `virtual`

`override` & `final` are needed _solely_ for the purpose of CE.  
In derived, method with the same **signature** will be overrided as is without `override`.

`= 0` makes a virtual function **pure virtual** — any subclass must implement this function.  
With `= 0` virtual class can't be initialized.

### Polymorthic & ABC

- **Abstract Base Class** - class that has pure virtual method.

- **Polymorthic Class** - class with virtual, but no pure virtual.

```cpp
// `I` - interface
class IAnimal {
public:
    virtual void makeSound() const {
        std::cout << "Some generic animal sound" << '\n';
    }

    virtual void Action() = 0;  // derived classes HAVE to implement this; `IAnimal` can't be initialized
};

class Dog : public IAnimal {
public:
    void makeSound() const override {
        std::cout << "Woof!" << '\n';
    }
    void Action() override {
        Bite();
    };
};
```

# Lecture 9 - Exceptions

Don't use exceptions like if-else, because they are very costly and not scalable - they mutex::lock thread while gathering cache info if exception occur.

## Different ways of handling

```cpp
std::abort();   // to kill execution
```

```cpp
std::expected<T, E>;    // like std::variant<...>
```

The best practice:

```cpp
void F() {
    try {
        throw std::runtime_error("FOO");
    } catch (const std::logic_error&) {
        // This block is skipped because std::runtime_error is not a logic_error
    } catch (const std::runtime_error&) {
        // Land here <-- std::runtime_error matches exactly
    } catch (const std::exception&) {
        // This block is skipped because the runtime_error is already caught
    }
}
```

## Custom exception

```cpp
class MyException : public std::exception {
private:
    const char* message;
public:
    MyException(const char* msg) : message(msg) {}

    const char* what() const noexcept override {
        return message;
    }
};
```

## Derived exception

```cpp
class FileException : public std::exception {
    const char* what() const noexcept override {
        return "File exception occurred";
    }
};
```

# Lecture 10 - Object Memory Layout

## Standard Layout Type

**POD (Plain Old Data)** $\rightarrow$ // C++11 // $\rightarrow$ **Standard Layout Type**

Requirements for Standard Layout Types:

- no virtual functions or inherit from a virtual base class.

- all non-static data members must have the same access control (e.g., all public, all private).

- no non-static members of base class:

- single inheritance only

- if a class has non-static data members, they must align properly with respect to the base class (no unexpected gaps in memory layout).

- members must not introduce unexpected overlaps in memory.

Standard Layout Types ensure compatibility between C++ and C.

## Alignment (Padding)

```cpp
struct MyStruct {
    char a;    // 1 byte
    int b;     // 4 bytes
    short c;   // 2 bytes
};

std::cout << sizeof(MyStruct) << std::endl; // Output may be 12 (with padding)
```

To change padding:

```cpp
#pragma pack(push, n) // Save current alignment and set new alignment to 'n'
#pragma pack(n)       // Set new alignment to 'n'
#pragma pack(pop)     // Restore the previous alignment
```

```cpp
#pragma pack(push, 1) // Set 1-byte alignment
struct MyStruct {
    char a;    // 1 byte
    int b;     // 4 bytes
    short c;   // 2 bytes
};
#pragma pack(pop) // Restore default alignment

std::cout << sizeof(MyStruct) << std::endl; // Output: 7 (no padding)
```

```cpp
struct MyStruct {
    char a;
    int b;
};

sizeof(MyStruct);       // Output: 8
alignof(int);           // Output: 4
offsetof(MyStruct, b);  // Output: 4
```

### `alignas`

C++11 and later:

```cpp
struct alignas(1) MyStruct {
    char a;
    int b;
    short c;
};
```

Using `alignas(1)` (or `#pragma pack(1)`) to minimize memory usage comes with slower memory access: CPU has to perform extra work to read or write **misaligned** data.

### `std::aligned_storage` & Placement `new`

`std::aligned_storage` provides raw, untyped memory with a specified size and alignment.

```cpp
template<class T>
class FixedContainer {
public:
    std::aligned_storage_t<sizeof(T), alignof(T)> storage;

    void construct(const T& value) {
        new (&storage) T(value);    // placement new
    }
};
```

## Referencing members

```cpp
int A::* ptr = &A::x;

a.*ptr;
a_ptr->*ptr;
```

## Inlining

Inlining (literally) of function content into code [when the compiler optimizes it] could:

1. improve speed of execution: since no need for addressing function & creating separate scope
2. worsen speed of execution: the larger executable file - the harder for OS to cache it to layers.

Different levels of optimization:

```bash
clang -O0 my_program.c
clang -O2 my_program.c
```

### `inline`

Originally, `inline` suggested to the compiler to replace a function call with the actual code of the function, removing the need for doing stack call of function.

However, modern compilers automatically optimize code and decide whether to inline a function based on heuristics.  
This makes its role as a performance hint less relevant today.

For functions defined inside a class definition, the compiler treats them as `inline` by default.

Modern use of `inline` is to allow function definitions in header files without violating the **One Definition Rule (ODR)**.

## `#define`

`#define` can be used to:

```cpp
#define PI 3.14159                  // symbolic constants
#define SQUARE(x) ((x) * (x))       // macros that look like functions but involve direct text substitution (no type safety)

#ifndef MY_HEADER_H                 // include guards
#define MY_HEADER_H
// Header content
#endif // MY_HEADER_H
```

## 4 stages of compilation

```txt
         source.cpp
             │
   ┌─────────┴───────────┐
   │ Preprocessing       │
   │ (clang -E)          │
   │ - Expand macros     │
   │ - Include headers   │
   │ - Remove comments   │
   └─────────┬───────────┘
             │
         source.i
             │
   ┌─────────┴───────────┐
   │ Compilation         │
   │ (clang -S)          │ <- `clang -emit-llvm -S` - LLVM Intermediate Representation - step between source code and machine code
   │ - Syntax analysis   │
   │ - Optimization      │
   │ - Convert to ASM    │
   └─────────┬───────────┘
             │
         source.s
             │
   ┌─────────┴───────────┐
   │ Assembly            │
   │ (clang -c)          │
   │ - Convert ASM to    │
   │   machine code      │
   └─────────┬───────────┘
             │
         source.o
             │
   ┌─────────┴───────────┐
   │ Linking             │
   │ (clang)             │
   │ - Resolve symbols   │
   │ - Link libraries    │
   │ - Produce binary    │
   └─────────┬───────────┘
             │
         executable
```

### Object files

When compiling C(++), the compiler produces .o files.  
These object files contain machine code (binary instructions) generated from your source code, but they are not standalone executables.  
Object files are intermediate files that are later linked together to create an executable or library.

```bash
clang -c my_program.cpp -o my_program.o

objdump -d my_program.o
```

```bash
clang -c a.cpp b.cpp c.cpp              # producing solo .o files
clang a.o b.o c.o                       # linking - final stage of compiling
./a.out
```

# Lecture 11.1 - gdb

## Basics

Let's say our program is:

```cpp
int main() {
    std::vector<int> vec_name = {1, 2, 3, 4, 5};
    return 0;
}
```

```bash
clang++ -g -O0 example_gdb.cpp      # to compile with debag & zero optimization level

gdb ./a.out
```

```gdb
info functions
info functions main

info variables
info locals                     # variables in a current function

break main()                    # set a breakpoint for function
info breakpoints

run                             # runs program until: crashes || end || breakpoint
start                           # runs program until the beginning of `main` function (or equivalent)

info frame                      # name of the current function you're in

# next - goes to following line
# step - goes into current line if there is inner function, else does next
# next, step (n, s) - for actual code (c++, rust, ...)
next
step

whatis vector_name              # will print the type
ptype vector_name               # will print the type in detail

print vector_name
print vector_name.size()
print\d vector_name.size()      # decimal
print vector_name.empty()
print &vector_name              # for example, gives address = 0x7fffffffdae0
```

### No debugging symbols

```gdb
# nexti, stepi (ni, si) - for asm;
nexti
stepi

context

info registers                  # `info locals`: No symbol table info available.

disassemble                     # assembly code for the current function
x/45i $pc                       # shows the next 45 instructions starting from the current instruction pointer ($pc)

print/d $rax
```

## Examine memory

Examine memory: x/FMT ADDRESS
Format letters are:

- o(octal)
- x(hex)
- d(decimal)
- u(unsigned decimal)
- t(binary)
- f(float)
- a(address)
- i(instruction)
- c(char)
- s(string)
- z(hex, zero padded on the left)

Size letters are: b(byte), h(halfword), w(word), g(giant, 8 bytes).

```gdb
x/3xw 0x7fffffffdae0            # 3 memory addresses from 0x7fffffffdae0 in xw format
# OUTPUT:
0x7fffffffdae0: 0x004172b0      0x00000000      0x004172c4
# 0x004172b0 - Pointer to the data buffer

x/4dw 0x004172b0
# OUTPUT - exact values of vector:
0x4172b0:       1      2       3       4

set *(int*)0x004172b0 = 10
set *((int*)0x004172b0 + 1) = 20

x/4dw 0x004172b0
# OUTPUT:
0x4172b0:       10     20      3       4

set *(int*)(0x7fffffffdae0) = 1
p vec_name
# OUTPUT:
$3 = std::vector of length 1072305, capacity 1072305 = {Cannot access memory at address 0x0}
# which now leads to Segmentation Fault
```

## Conditional breakpoints

```cpp
void print_numbers(int limit) {
    int j;
    for (int i = 0; i < limit; ++i) {
        if (i % 2) {
            j = i;
        }

        std::cout << "Number: " << i << '\n';
    }
}

int main() {
    print_numbers(10);
    return 0;
}
```

```gdb
break print_numbers
condition 1 i == 5              # condition to the breakpoint

run
print i
# OUTPUT:
$1 = 5
```

### No debugging symbols

```gdb
break *0x0000555555400662
condition 1 $esi == 90000
```

### Watch

```gdb
start
next

watch j
continue
continue
continue
```

## Core dumps

**Core dump** - snapshot of the program's state (program's memory [heap, stack, global variables], processor registers) of a running program at a specific point in time when the program crashes.

```bash
gdb /path/to/executable /path/to/coredump
```

When analyzing a core dump with GDB, you usually want to inspect the state of the program at the time of the crash, without needing to restart it.

Common causes: segmentation faults, illegal instructions, or invalid memory access.

```cpp
int div(int a, int b) {
    return a / b;
}

int main() {
    div(1, 0);
    return 0;
}
```

```bash
clang++ -g -O0 example_gdb.cpp -o core_dump_example.out
./core_dump_example.out                                     # Floating point exception (core dumped)

ls /var/lib/systemd/coredump/
mv /var/lib/systemd/coredump/

sudo mv /var/lib/systemd/coredump/core.core_dump_examp.1000.429bb74afb9e4a21ba0abbfc96caa5b2.235604.1733034544000000.zst core_dump

gdb core_dump_example.out core_dump
```

```gdb
backtrace

info locals
info args

list                        # code around the crash
```

# Lecture 11.2 - OSI model

## OSI Model - Layers and Protocols

**Open Systems Interconnection (OSI)** model - reference model from the **International Organization for Standardization (ISO)** that partitions the flow of data in a communication system into 7 abstraction layers to describe networked communication:

Each layer talks _only_ to the one above and below it — this makes systems **modular**.

### 1. Physical Layer

- Hardware transmission of raw bits over a physical medium.

- Concerned with the physical connection between devices (e.g., cables, switches, and wireless signals).

---

### 2. Data Link Layer

- Error-free transmission of data from one node to another over the physical layer.

- Responsible for framing, addressing, and error detection.

Examples: Ethernet (802.3), Wi-Fi (802.11), Bluetooth.

---

### 3. Network Layer

- **Routing** of data packets between devices across multiple networks.

- Ensures data reaches the correct destination using logical addressing (IP addresses).

Examples: Internet Protocol (IPv4, IPv6), ICMP (for ping).

---

### 4. Transport Layer

- Provides end-to-end communication and ensures reliable or efficient data transfer.

### TCP

**TCP** (Transmission Control Protocol) - _connection-oriented_ protocol designed to provide _reliable_ (data integrity) communication in _correct order_ with between devices.

- Resends lost or corrupted packets (**error correction**).

- Higher overhead due to connection setup, acknowledgments, and error correction.

Use cases:

- HTTP/HTTPS (web browsing)
- FTP (file transfers)
- SMTP/IMAP/POP3 (emails)

### UDP

**UDP** (User Datagram Protocol) - _connectionless_ protocol designed for _low-latency_ and time-sensitive (may be not reliable & ordered) communication.

- No handshake or connection establishment is required.

Use cases:

- DNS lookups
- Video streaming
- Online gaming

### URI & URL

**URI** (Uniform Resource Identifier) - string used to identify a resource on the internet.

URIs often come in the form of **URLs** (Uniform Resource Locators), which specify the location of the resource and the protocol used to access it (e.g., HTTP or HTTPS).

---

### 5. Session Layer

- Manages and controls the dialog between two devices.

- Ensures that sessions remain open and data is synchronized.

Examples: APIs for maintaining sessions (e.g., **sockets**).

---

### 6. Presentation Layer

- Translates data between the application and network formats.

Key Concepts:

- Data Encoding/Decoding: Formats such as ASCII, JPEG, MP3.
- Encryption/Decryption: Securing data during transmission.

Examples: SSL/TLS for secure data presentation, data compression.

---

### 7. Application Layer

- Closest to the end-user and directly interacts with software applications.

---

### Real-World Example: Sending an Email

1. Application Layer: You compose an email in an email client.
2. Presentation Layer: Data is formatted and encrypted.
3. Session Layer: A session is established with the mail server.
4. Transport Layer: TCP ensures email data is transmitted reliably.
5. Network Layer: Email is routed across multiple networks to the destination server.
6. Data Link: Ensures error-free transmission over local network.
7. Physical Layer: Converts data to electrical signals for transmission.

## API

**Application Programming Interface (API)** - set of rules and protocols that allow for contract between a client (like an application or a service) and a server (another application or service).

Types of APIs:

- Web APIs
- Library APIs (in a programming language)
- Operating System APIs (e.g., Windows API, POSIX API)

**Endpoint** - Specific URL where the API can be accessed.

Common HTTP methods:

- GET: Retrieve data.
- POST: Submit data.
- PUT: Update data.
- DELETE: Remove data.

## `nc` NetCat (for TCP & UDP)

Basic syntax:

```bash
nc [options] [hostname] [port]
```

1. Establishing TCP | UDP connections (Connect to a specific host and port)

   ```bash
   nc example.com 80
   ```

   \*"Establishing a UDP connection" is a linguistic simplification or abstraction, because UDP is _connectionless_.

2. Simple web request (Send a raw HTTP request)

   ```bash
   nc example.com 80
   GET / HTTP/1.1
   Host: example.com
   ```

   Press `Enter` twice to complete the request.

3. Open port scanning

   ```bash
   nc -zv example.com 20-80
   ```

4. File Transfer

   ```bash
   nc -l -p 1234 < file.txt        # Server side (send a file)
   nc server_ip 1234 > file.txt    # Client side (receive a file)
   ```

5. Chat Between Two Machines

   ```bash
   nc -l -p 1234                   # Server side (set up a chat server)
   nc server_ip 1234               # Client side (connect to the server)
   ```

6. Create a TCP/UDP Server

   ```bash
   nc -l -p 1234                   # TCP
   nc -l -u -p 1234                # UDP

   # from another terminal to send http request
   curl -i "http://localhost:1234/p/a/t/h?text=cats&query=dogs"
   ```

### POCO C++ library

POCO (Portable Components) is ideal for building efficient, lightweight C++ applications that require: robust networking, multithreading, and data processing capabilities.

Alternatives:

- Boost: Offers a broader range of functionality but might feel heavier.
- Qt: Provides GUI features along with similar functionalities but targets more graphical applications.

# Lecture 12 - Undefined behavior

Behaviors of C++ Standard:

1. implementation-defined
2. unspecified
3. undefined

## Implementation-defined

**Implementation-defined behavior**: defined (documented) by the compiler but may vary across compilers.

```cpp
int main() {
    int x = -5;
    unsigned int y = 3;
    std::cout << x % y << std::endl; // the standard allows compilers to decide how modulo works with negative numbers
}
```

## Unspecified behavior

**Unspecified behavior** occurs when the C++ Standard allows multiple valid outcomes for a given construct, but it does not mandate which one will happen in a specific scenario.  
The program remains well-formed, but the exact result is not guaranteed.

```cpp
int x = 1, y = 2;
int z = foo(x++, y++); // The order of evaluation of `x++` and `y++` is unspecified.
```

## Undefined behavior

**Undefined Behavior** (UB) is unpredictable and not specified by the C++ standard.  
It occurs when the program violates the rules of the language, such as accessing out-of-bounds memory, dereferencing null pointers, or dividing by zero.

UB means the C++ standard imposes no requirements on what happens when UB is encountered: results can vary, including crashes, incorrect outputs, or seemingly correct behavior.

Sometimes a compiler can throw UB, but sometimes it doesn't and code somehow runs even though it has UB.

> Compiler always optimize based on the assumption that **UB does not exist** in the code.

### Example

```cpp
int square(int n) {
    for (int i = 0; ; i += 3) {
        if (i == n * n) {
            return i;
        }
    }
}

int main() {
    std::cout << square(3);     // seemingly correct behavior: 9, even though it did no looping
    std::cout << square(10);    // incorrect output: 100 ! even though with +=3 10 can't be reached
    return 0;
}
```

Disassemble of it with `-O1`:

```asm
square(int):
        imul    edi, edi
        imul    rax, rdi, 1431655766
        shr     rax, 32
        lea     eax, [rax + 2*rax]
        ret
```

Compiler knows that infinite loop is UB, so there is only 1 option in trying to avoid it: return n \* n without doing loop.  
But it's still UB, that's why in case of UB behavior is unpredictable.

## Passing by value

This code in asm does the loop fully with accessing `vec.size` each time.  
This happens, because compiler gives possibility to `vec.size` be changed while looping.

```cpp
struct Vector {
    int* arr;
    int size;
};

void foo(Vector& vec) {
    for (size_t i = 0; i < vec.size; ++i) {
        vec.arr[i] = 0;
    }
}
```

Following code in asm just uses `memset` for whole array once, because size isn't changed and it's clear, thus it is very optimized:

```cpp
struct Vector {
    int* arr;
    int size;
};

void foo(Vector& vec) {
    int size = vec.size;

    for (size_t i = 0; i < size; ++i) {
        vec.arr[i] = 0;
    }
}
```

# Lecture 13 - Design patterns

Types of Patterns:

- Creational: abstract the process of instance creation
- Structural: organizing classes and objects to form larger structures
- Behavioral: delegate responsibilities and manage communication between objects

## Creational: Factory

Hide the logic of object creation, and clients use the factory interface to get the object instead of instantiating it directly.

Benefits:

- **Loose coupling**: components or modules of a system are minimally dependent on each other.
- **Single Responsibility Principle**: delegates the creation logic to the factory.
- **Open/Closed Principle**: makes the code open to extension but closed to modification.

### Structure

1. Product:  
   The interface or abstract base class for objects the factory will create.

   ```cpp
   class Button {
   public:
       virtual void render() = 0;
       virtual ~Button() = default;
   };
   ```

2. ConcreteProduct:  
   The specific implementation of the Product.

   ```cpp
   class WindowsButton : public Button { ... };

   class MacOSButton : public Button { ... };
   ```

3. Creator:  
   Declares the factory method which returns a Product object.

   ```cpp
   class Dialog {
   public:
       virtual Button* createButton() = 0;
       virtual ~Dialog() = default;
   };
   ```

4. ConcreteCreator:  
   Overrides the factory method to create specific ConcreteProduct objects.

   ```cpp
   class WindowsDialog : public Dialog {
   public:
       Button* createButton() override {
           return new WindowsButton();
       }
   };

   class MacOSDialog : public Dialog {
   public:
       Button* createButton() override {
           return new MacOSButton();
       }
   };
   ```

Client:

```cpp
void renderUI(Dialog* dialog) {
    Button* button = dialog->createButton();
    button->render(); // Use the button without knowing its concrete type
    delete button;
}

int main() {
    Dialog* dialog = new WindowsDialog();
    renderUI(dialog);
    delete dialog;

    dialog = new MacOSDialog();
    renderUI(dialog);
    delete dialog;

    return 0;
}
```

## Creational: Abstract Factory

Create families of related or dependent objects without specifying their concrete classes.

Adding new families of products requires only the addition of new factory and product classes, leaving existing code unchanged.

Additional benefits:

- **Encapsulation**: Hides the creation logic for a family of related objects.
- Consistency: Ensures all created objects are part of the same family

### Difference from Factory Method

Factory method focuses on creating a _single_ product through a method that subclasses can override to decide which concrete class to instantiate.

### Structure

1. Abstract Product:  
   Defines the interface for product objects.

   ```cpp
   class Button {
   public:
       virtual void render() = 0;
       virtual ~Button() = default;
   };

   class Checkbox {
   public:
       virtual void check() = 0;
       virtual ~Checkbox() = default;
   };
   ```

2. Concrete Product:  
   Implements the abstract product interface for a specific variant.

   ```cpp
   class WindowsButton : public Button { ... };
   class WindowsCheckbox : public Checkbox { ... };

   class MacOSButton : public Button { ... };
   class MacOSCheckbox : public Checkbox { ... };
   ```

3. Abstract Factory:  
   Declares creation methods for each product family.

   ```cpp
   class GUIFactory {
   public:
       virtual Button* createButton() = 0;
       virtual Checkbox* createCheckbox() = 0;
       virtual ~GUIFactory() = default;
   };
   ```

4. Concrete Factory:  
   Implements creation methods for specific product families.

   ```cpp
   class WindowsFactory : public GUIFactory {
   public:
       Button* createButton() override {
           return new WindowsButton();
       }
       Checkbox* createCheckbox() override {
           return new WindowsCheckbox();
       }
   };

   class MacOSFactory : public GUIFactory {
   public:
       Button* createButton() override {
           return new MacOSButton();
       }
       Checkbox* createCheckbox() override {
           return new MacOSCheckbox();
       }
   };
   ```

Client:

```cpp
class Application {
private:
    GUIFactory* factory;
    Button* button;
    Checkbox* checkbox;

public:
    Application(GUIFactory* f) : factory(f) {
        button = factory->createButton();
        checkbox = factory->createCheckbox();
    }

    void renderUI() {
        button->render();
        checkbox->check();
    }

    ~Application() {
        delete button;
        delete checkbox;
    }
};

int main() {
    // Example: Using WindowsFactory
    GUIFactory* factory = new WindowsFactory();
    Application* app = new Application(factory);
    app->renderUI();

    delete app;
    delete factory;

    return 0;
}
```

## Creational: Singleton

Ensures that"

1. only one instance of a class exists
2. provides a global access to the instance

Used in managing shared resources (configuration, settings, logging, database connections).

```cpp
struct Singleton {
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

private:
    Singleton();
    ~Singleton() = default;

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    Singleton(Singleton&&) = delete;
    Singleton& operator=(Singleton&&) = delete;
};
```

### LeakySingleton

LeakySingleton is never explicitly destroyed, so it "leaks" the instance intentionally, ensuring it outlives all other static objects.  
The operating system reclaims the memory when the program exits.

```cpp
class LeakySingleton {
public:
    static LeakySingleton& getInstance() {
        static LeakySingleton* instance = new LeakySingleton(); // Dynamically allocate
        return *instance;
    }

    void doSomething() const {
        std::cout << "LeakySingleton is working!" << std::endl;
    }

private:
    LeakySingleton();
    ~LeakySingleton() { std::cout << "LeakySingleton Destructor will never be used" << std::endl; }

    LeakySingleton(const LeakySingleton&) = delete;
    LeakySingleton& operator=(const LeakySingleton&) = delete;
    LeakySingleton(LeakySingleton&&) = delete;
    LeakySingleton& operator=(LeakySingleton&&) = delete;
};
```

## Structural: Adapter

Allows incompatible interfaces to work together by providing a bridge (adapter) between them.

Adapter: Converts Adaptee's interface to Target's interface.

```cpp
// Existing class with incompatible interface
struct Adaptee {
    void specificRequest();
};

// Target interface expected by the client
struct Target {
    virtual void request() const = 0;
    virtual ~Target() = default;
};

struct Adapter : public Target {
    Adapter(Adaptee* a) : adaptee(a) {}

    void request() const override {
        std::cout << "Adapter: Translating request...\n";
        adaptee->specificRequest(); // Call the Adaptee's method
    }
private:
    Adaptee* adaptee;
};
```

## Creational: Decorator

Dynamically add behavior to objects at runtime without modifying their structure.

```cpp
class Coffee {...};

class ConcreteCoffee : public Coffee {...};

// Base Decorator
class CoffeeDecorator : public Coffee {
protected:
    std::shared_ptr<Coffee> coffee;

public:
    CoffeeDecorator(std::shared_ptr<Coffee> c) : coffee(std::move(c)) {}
    virtual ~CoffeeDecorator() = default;
};

// Concrete Decorators
class MilkDecorator : public CoffeeDecorator {
public:
    MilkDecorator(std::shared_ptr<Coffee> c) : CoffeeDecorator(std::move(c)) {}

    double cost() const override {
        return coffee->cost() + 1.5; // Add cost for milk
    }
};

class SugarDecorator : public CoffeeDecorator {
public:
    SugarDecorator(std::shared_ptr<Coffee> c) : CoffeeDecorator(std::move(c)) {}

    double cost() const override {
        return coffee->cost() + 0.5; // Add cost for sugar
    }
};

int main() {
    // Base coffee
    std::shared_ptr<Coffee> myCoffee = std::make_shared<SimpleCoffee>();
    myCoffeeWithSugar = std::make_shared<SugarDecorator>(myCoffee);
    myCoffeeWithSugarAndMilk = std::make_shared<MilkDecorator>(myCoffeeWithSugar);

    return 0;
}
```

## Behavioral: some

- Strategy: Encapsulates interchangeable algorithms.
- Observer: Notifies dependent objects of changes.
- Command: Encapsulates a request as an object.
- State: Allows an object to change behavior when its state changes.
- Mediator: Centralizes communication between objects.
