<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD049 -->

# Notes

[Playlist](https://www.youtube.com/playlist?list=PLbtaNY5hOUWnKP8uHAqfqSlQHTR4GMX3p)

Use [cppinsights.io](https://cppinsights.io/).

**TQ** = Tricky question;

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
    // so it's the same as:
    // int32_t x;
    // but UB of using uninitialized memory :)
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

## Reference

Once reference like `int& r = x` is called, r _cannot_ be changed what to reference $\to$ references has to be initialized.

### References: Object VS Variable VS Function VS Method

```cpp
sizeof(int*);   // 8

int a;
sizeof(&a);     // 8

void f() {}
sizeof(&f);     // 8

struct A {
  void f() {}
};
sizeof(&A::f);  // 16 !!!
```

**Pointer-to-member-function** may include both code pointer & **this-adjustment offset** to support multiple inheritance and virtual functions.

## `void*`

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

### VLA

**VLA (Variable-Length Array)** - array (stored in stack memory) whose size is determined at runtime.

```cpp
int n;
std::cin >> n;
int arr[n];     // GCC extension, not standard C++
```

C++ standard forbids VLAs.  
However, some compilers (like GCC) **violate** standard allowing it as an extension.

Why Are VLAs Not in C++? $\to$ allocating large arrays on the stack can lead to **stack overflow**.

> Compilers generally strive to adhere strictly to the C++ standard.  
> However, in practice, compilers may **violate** the C++ standard.

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

`const`-**qualified** object type restricts access to only those member _methods and operators_ that are also `const`-qualified.

## `const`

```cpp
const int x;    // CE: uninitialized `const`
```

## `T* const` && `const T*`

**TQ**: difference between `const int*` and `int* const`

```cpp
int x;

int* const cpx = &x;

const int* px = &x;
// same as
int const* px = &x;
```

```cpp
int x;

const int* pxc = &x;    // allowed implicitly: `int*` -> `const int*`; `x` is still `int*`

*pxc = 1;               // CE
++x;                    // okay


int* px = pxc;          // CE, no implicit `remove_const` conversion
```

## `T& const`

`T& const` is not allowed $\to$ no constant references.

```cpp
int x = 0;
int y = 0;

int& ref = x;
ref = y;

int& const cref = x;    // CE
```

## Lifetime Expansion

**Lifetime expansion** of temporary object:

```cpp
const int& ref = 1;     // 1. lifetime expansion via const reference
int&& rref = 1;         // 2. lifetime expansion via rvalue reference

// the same with functions:
void f(const T& val);
void f(T&& val);
f(1);

// ---

int& ref = 1;           // CE
void f(T& val);
f(1);                   // CE
```

## Typical Arguments

```cpp
void f(const T& val);   // binds to both _lvalues and rvalues_ (lifetime expansion)
void f(T& val);         // only binds to _non-const lvalues_
void f(T val);          // also: must have for small types

void f(const T val);    // don't do this shit :)
```

# Lecture 11 - Typecasts, Assembly stages, Sanitizers

## Typecasts

### 1. `static_cast<T>(expr)`

- Compile-time type conversion
- safe-ish
- _checks types_

- UB if used incorrectly

```cpp
int i = 42;
float f = static_cast<float>(i);      // OK

Base* b = new Derived();
Derived* d = static_cast<Derived*>(b); // OK if you're sure
```

### 2. `reinterpret_cast<T>(expr)`

- Bit-level reinterpretation of memory.
- Ignores types completely

- May violate strict aliasing ($\to$ UB)

```cpp
int* ip = new int(42);
char* cp = reinterpret_cast<char*>(ip); // View int memory as bytes
```

> Avoid using it. Causes UB when byte structure is different.

### 3. `dynamic_cast<T>(expr)`

- Safe RTTI-based cast for polymorphic types.

- Only works if the base class has at least one `virtual` function
- Returns `nullptr` on failure (for pointers)
- Throws `std::bad_cast` (for references)

Use it for:

- Downcasting polymorphic pointers
- Checking cast validity at runtime

```cpp
Base* b = new Derived();
Derived* d = dynamic_cast<Derived*>(b); // Works if actually Derived*
```

Counter-example:

```cpp
struct A {};                // No virtual functions
struct B : A {};
A* a = new B;
B* b = dynamic_cast<B*>(a); // Compile error!
```

### 4. `const_cast<T>(expr)`

- Adds or removes `const` / `volatile` qualifiers.

- Modifying a `const` object $\to$ UB

```cpp
const int x = 10;
int* px = const_cast<int*>(&x);     // ⚠️ Legal, but modifying is UB

void func(char* p);
const char* s = "hello";
func(const_cast<char*>(s));         // OK if func doesn’t write to s
```

> Avoid using it. Causes UB when modifying

### 5-ish. C-style Casting

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

> If it would work you won't know which cast it even was.  
> Avoid C-style casting, and use C++-style casts explicitely.

## Casting to: Object VS Pointer

### Object

Cast to object $\to$ essentially _creating a new object_ from the result of the cast.

Casting an object to its base will **slice** off derived attributes.

### Pointer

Cast to pointer $\to$ just changes the interpretation of the memory address without copying the object.

Often used for:  
**Upcasting** for polymorphic behavior, **downcasting** with `dynamic_cast`.

## Standard Library

We know that at the preprocessing stage each header gets included in the file like text.  
We also know that C++ compiling is pretty slow.  
**TQ**: why headers from standard library don't take compilation time?

The _definitions_ of standard library are **precompiled** and stored in static libraries (`.a` or `.lib` files) or dynamic/shared libraries (`.so`, `.dll`, or `.dylib` files), rather than being compiled each time to raw object files (`.o` or `.obj`).

At the preprocessing stage `<iostream>` will expand into the file only _declarations_ to which _definitions_ are already precompiled. So, the only linker will take time for compiling `<iostream>`.  
That's why it is `<iostream>` and not `"iostream"` to tell the compiler that these includes are from standard library.

> # PART 2: OOP

# Lecture 12 - Classes

## Private Overloaing

**TQ**:

```cpp
class A {
	void f(int a) {}    // witthout it `0` will be implicitely converted to `float` and `f(float)` will be used
public:
  	void f(float a) {}
};

A cl;
cl.f(0);    // CE: 'f' is a private member of 'A'
cl.f(3.14); // CE: `3.14` is `double` $\to$ no explicit function for `double` $\to$ `f(int)` & `f(float)` are both available $\to$ tries `f(int)` $\to$ private
cl.f(3.14f) // okay
```

This is quite contrintuitive.  
It finds most suitable overload (even if it's private) and tries to call it.

# Lecture 13 - Constructors

## `{}` initializations

```cpp
class A;

// Initialization Types:

// **Initializer List**: if it has `A(std::initializer_list<T>)`
// **Direct List**: if it has any constuctor (except `std::initializer_list`)
// **Aggregate**: if it has no constuctor
A obj{1, 2};

// **Copy List**: if it has any constuctor (except `std::initializer_list`)
// **Copy List with Initializer List**: if it has `A(std::initializer_list<T>)`
// **Aggregate**: if it has no constuctor
A obj = {1, 2};

// **Default Constructor**: if it has `A()`
// **Value**: attributes default-initialized
A obj4{};
```

Constructor with `std::initializer_list` has the highest priority and it will always try to do implicit conversion to it first.

$\to$ it's sometimes better to use `A obj(...)` instead of `A obj{...}` if you don't want do Initializer List Initialization.

## Explicitly Declared Implicitly Defined Constructor (`default`)

Compiler automatically generates a default implementation for it.

```cpp
class A {
    ...
public: 
    A(...) = default;

    A(const A&) = default;

    A(A&&) = default;

    ~A() = default;
};
```

In cases where other constructors exist, explicitly declaring an implicit constructor can help the compiler resolve **ambiguities**.

# Lecture 14 - Move Semantics

## `delete`

**TQ**: you have template smth. How to forbid some types to use?

1. SFINAE

2. using `concept`s and `require`

3. `delete`

    ```cpp
    template <typename T>
    void f(T&& val);

    void f(int val) = delete;
    ```

---

## Copy assignment

**TQ**: In Copy Assignment Operator why do we return `ClassName&` and not:

- `void`                  // because we want to support `(a = b) = c`;

- `ClassName`             // doing extra copy that is unused

- `const ClassName&`      // because we want to support `(a = b) = c`;

# Lecture 15 - `const`, `mutable`, `static`, `explicit`

## Type Cast Operator

`operator target_type() const;`

```cpp
class Number {
public:
    Number(double v) : value(v) {}

    operator double() const {
        return value;
    }

    operator int() const {
        return static_cast<value>;
    }

    explicit operator bool() const {
        return value <= 0;
    }

private:
    double value;
};
```

`explicit` **forbids** implicit conversions $\to$ only via `static_cast`.

> It's good practice to make conversion to `bool` explicit (for `if`).

# Lecture 16 - Operator Overloading

> Overload binary operators out of the class.

## `+` & `+=`

> Express overload of `+` through overload of `+=` to avoid extra copying.

```cpp
class Number {
public:
    Number() : value(0) {}

    Number(double v) : value(v) {}

    Number& operator+=(const Number& other) {
        value = value + other.value;
        return *this;
    }

private:
    double value;
};

Number operator+(const Complex& left, const Complex& right) {
    Complex result = left;
    result += right;
    return result;
}

Number x(1.0);
x + 1.0;    // ok
1.0 + x;    // ok
```

### rvalue assigning

**TQ**:

```cpp
Number a, b, c;

a + b = c;  // Ok, because `a + b` returns `Number`, but strange as fuck

// to forbid that do:
Complex& Complex::operator=(const Complex& other) & {... }
// now we can assign only to lvalue
```

## `operator<=>`

```cpp
#include <compare>

std::weak_ordering Number::operator<=>(const Number other) const = default;
```

# Lecture 17 - Enums, Inheritence

## Enums

Use `enum class`, because:

- No implicit conversion to/from `int`

- No global name pollution of enumerator's attributes

## `public`, `protected`, `private`

### Access Modifiers

```cpp
class A {
public:     // to all
    ...

protected:  // to this & inheritor only (but not to inheritor of inheritor)
    ...

private:    // to this & friends
    ...
}
```

Access modifiers don't affect **scope resolution** (visibility) at all, only **accessibility**.

When accessing smth: 1. In scope? $\to$ 2. Accessable?

### Inheritance Semantics

```cpp
class Base {
public:
    int a;

protected:
    int b;

private:
    int c;
};

class PublicDerived : public Base {
    void f() {
        a = 1; // ✅ OK (public -> public)
        b = 2; // ✅ OK (protected -> protected)
        c = 3; // ❌ was private
    }
};

class ProtectedDerived : protected Base {
    void f() {
        a = 1; // ✅ OK (public -> protected)
        b = 2; // ✅ OK (protected -> protected)
        c = 3; // ❌ was private
    }
};

class PrivateDerived : private Base {
    void f() {
        a = 1; // ✅ OK (public -> private)
        b = 2; // ✅ OK (protected -> private)
        c = 3; // ❌ was private
    }
};

int main() {
    PublicDerived d1;
    d1.a = 10; // ✅ OK
    d1.b = 20; // ❌ protected
    d1.c = 30; // ❌ private

    ProtectedDerived d2;
    d2.a = 10; // ❌ now protected

    PrivateDerived d3;
    d3.a = 10; // ❌ now private
}
```

> Note that attributes of Derived are all **visible**, but some of them are not **accessible**.

Example:

```cpp
class FancyVector : private std::vector<int> {
public:
    void PushBackTwo(int x, int y) {
        push_back(x);
        push_back(y);
    }
};
```

## `friend`

`friend` is not inheritable.

### best

```cpp
// class B;    // no need

class A {
    friend class B;
}

class B {
}
```

### "строгая мама не разрешает общаться с доброй бабушкой"

**TQ**:

```cpp
struct Granny {
    friend struct Son;

private:
    int x;
};

struct Mom : protected Granny { // or `private`
};

struct Son : Mom {
};

int main() {
    Son s;
    s.x;    // CE
}
```

# Lecture 18 - Inheritence

## EBO

**EBO** - Empty Base Optimization

```cpp
class Base {};          // sizeof(Base) == 1

class Derived: Base {}; // sizeof(Derived) == 1


class Derived: Base {   // sizeof(Derived) == sizeof(Derived::x) == 8
    int64_t x;

    void Method();      // methods weight nothing
};
```

## Contructors

```cpp
struct Base {
    int x;
    // NO: Base() : x(0) {};
    // NO: Base(int x) = default
    Base(int x) : x(x) {}
}

struct Derived: Base {
    double y;

    Derived(double y) : Base(0), y(y) {}    // otherwise CE
}
```

## Implicit Inheritence Cast

```cpp
class Derived: public Base {...}

void c(Base base);  // slicing

void r(Base& base);

void p(Base* base);

Derived d;
c(d);       // OK: slicing
r(d);       // OK: implicit cast to Base
p(d);       // OK
```

**Slicing**: `Base` will be **sliced** out of `Derived` using `Base(const Base& b);` copy constructor (which can be default generated or manually implemented).

# Lecture 19 - Multiple & Virtual Inheritence

## Multiple Inheritence

```cpp
class A {
    void f();
    void g(int);
}

class B {
    void f();
    void g(double);
};

class C: public A, public B {};

C c;        // in memory: A B C
c.f();      // CE
c.g(1);     // OK
c.g(1.0)    // OK
```

Methods of base classes get overloaded with each other in derived class.

## Diamond Problem

```txt
  A
 / \
B   C
 \ /
  D
```

```cpp
class A {
public:
    void f();
};

class B : public A {};
class C : public A {};

class D : public B, public C;

D d;        // in memory: |A| |B| |A| |C| |D|

d.f();      // CE
d.A;        // CE, it should be: `d.B::A;` or `d.C::A;`
A& a = d;   // CE
```

### Solution

With **`virtual` inheritance**, D contains only one instance of A, shared by B and C.

```cpp
class A {
public:
    void f();
};

class B : virtual public A {};
class C : virtual public A {};

class D : public B, public C {};

D d;        // in memory: |ptr of B on A| |B| |ptr of C on A| |C| |D| |A|

d.f();      // OK
d.A;        // OK
A& a = d;   // OK
```

## Virtual Table

**Virtual table** - lookup table of function pointers - static, per-class array of function pointers pointing to the most derived overrides of the virtual functions.

A class gets a **vtable** if:

- has at least one `virtual` function.

- not marked `final`

# Lecture 20 - 
