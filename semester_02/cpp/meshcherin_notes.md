<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD049 -->

# Notes

**TQ** = Tricky question;

[Playlist](https://www.youtube.com/playlist?list=PLbtaNY5hOUWnKP8uHAqfqSlQHTR4GMX3p)

# Lecture 3 - Declarations and Definitions, Scopes, Namespaces

**Keyword** - reserved word that has a predefined meaning in the language and cannot be used for naming identifier.

## Lookups

| Lookup Priority | Lookup Type                         | Example                | Comment                      |
| :-------------: | :---------------------------------- | :--------------------- | :--------------------------- |
|       1️⃣        | **Qualified lookup**                | `X::x = 5;` or `obj.x` | Compiler knows exactly where |
|       2️⃣        | **Unqualified lookup**              | `x = 5;`               | Compiler hunts by rules      |
|       3️⃣        | **Argument-dependent lookup (ADL)** | `add(v1, v2);`         | Special case for functions   |

```cpp
namespace X {
    int32_t x = 0;
}

int main() {
    using namespace X;  // only affects `unqualified lookup` (names are visible but not declared).
    using X::x;         // declares X::x into the current scope
    int32_t x = 1;      // okay with `namespace X`; _conflict_ with `X::x`
}
```

### Unintentional Overloading

```cpp
using namespace std;

void distance();        // but std already has distance -> this is an overload
```

### "reference to 'x' is ambiguous"

```cpp
namespace X {
    int32_t x = 0;
}

namespace Y {
    int32_t x = 1;
}

int main() {
    using namespace X;
    using namespace Y;

    std::cerr << x << '\n'; // error: reference to 'x' is ambiguous
}
```

### Point of Declaration

**Point of declaration**: the variable is declated immidiately after `=` sign.

**TQ**:

```cpp
int32_t x = 0;

int main() {
    int32_t x =         x;  // warning: variable 'x' is uninitialized when used within its own initialization -> UB
                // . here `x` is already declared, but not initialized and we initialize it with itself
}
```

# Lecture 4 - Statements & Operators

## `operator`

Left-Associative Operators:

- Overloadable globally: `()` `[]` `->` `->*` `*` `/` `%` `+` `-` `<<` `>>` `<` `<=` `>` `>=` `==` `!=` `&` `^` `|` `&&` `||` `,`

- Overloadable only within class: `::` `.` `.*`

Right-Associative Operators:

- Overloadable globally: `++` `--` `+` `-` `!` `~` `*` `&` `=` `+=` `-=` `*=` `/=` `%=` `>>=` `<<=` `&=` `^=` `|=` `new` `delete`

### Priority

[C++ Operator Precedence](https://en.cppreference.com/w/cpp/language/operator_precedence)

**TQ**:

1. `++a++` $\equiv$ `++(a++)` $\leftarrow$ is not allowed (`...++` takes lvalue, but after first `..++` we get rvalue)

2. `a+++b` $\equiv$ `(a++) + b` $\leftarrow$ okay!

3. `a++++` $\leftarrow$ is not allowed (`...++` takes lvalue, but after first `..++` we get rvalue)

## Random

```cpp
for (declaration | expression; bool expression; expression) {}
```

```cpp
int32_t x = 0;      // here `=` is copy constructor operator
x = 1;              // here `=` is assignment operator
// they are completely different `=`
```

# Lecture 5 - CE, RE, UB

## Floating Point Exception

**TQ**:

```cpp
1 / 0;      // Floating Point Exception
1f / 0f;    // `inf` <- OKAY
```

# Lecture 6 - Pointers

## C-style Casting

```cpp
(type)expression
```

A C-style cast in C++ tries to apply these casts in order until one works:

1. `const_cast`

2. `static_cast`

    ```cpp
    int i = (int)3.14f;
    ```

3. `static_cast` + `const_cast`

4. `reinterpret_cast`

    ```cpp
    float f = 3.14f;
    int* p = (int*)&f;  // dangerous bit reinterpretation
    ```

5. `reinterpret_cast` + `const_cast`

> Avoid C-style casting, and use C++-style casts explicitely.

### `void*`

`void*` — generic pointer type — it can point to any type of data, but knows nothing about the type.

`void*` cannot be dereferenced.

## C libraries

**TQ**: why we use `int32_t` just like that if we include it with `<cstdint>`?

`int32_t` is from `<cstdint>`, so from C standard library, but it doesn't need anything like `cstd::`, because C had _no namespaces_, so everything from C standard library is _global_.

# Lecture 7 - Stack, Static, Heap Memory

---

# Lecture 8 - Arrays & Function Pointers

## Arrays

### Array indexing

**TQ**:

```cpp
int arr[3]{1, 2, 3};

*arr  == arr[0];                                // **array-to-pointer conversion**

arr[2] == *(arr + 2) == *(2 + arr) = 2[arr];    // !!!

int* p = a + 1;
p[-1];                                          // totally okay, same as *((a + 1) - 1)
```

### Array VS Pointer

```cpp
void f(int* arr);
// the same as
template <size_t N>
void f(int arr[N]);
```

```cpp
int* a[10];     // array of pointers
int (*a)[10];   // pointer to array
```

### `char*`

```cpp
const char* s = "a";        // takes **2 bytes** because C strings always have an automatic '\0' terminator.

const char* str = "ab\0cd";
std::cout << str << '\n';   // "ab" - dereferencing step by step until '\0'
```

## Function Pointer

Suppose you have:

```cpp
template <typename T>
bool Cmp(const T& l, const T& r);
```

Three _C_ ways of using it:

1. **Function reference** - address of where function stored in **.text/code section of ELF** binary file.

    ```cpp
    std::sort(v.begin(), v.end(), &Cmp);
    ```

2. **Function pointer** to _instantiated_ version.

    ```cpp
    bool (*cmp)(const T&, const T&) = &Cmp; // or = Cmp
    std::sort(v.begin(), v.end(), cmp);
    ```

3. Implicit **function-to-pointer conversion** to `bool (*)(const T&, const T&)`.

    ```cpp
    std::sort(v.begin(), v.end(), Cmp);
    ```

# Lecture 9 - Complex Definitions

**TQ**: is this allowed? - Yes

```cpp
int& f(int& x) {
    return ++x;
}

f(x) = 10;
```

# Lecture 10 - Constants
