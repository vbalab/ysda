<!-- markdownlint-disable MD025, MD001, MD024 -->

# Notes

### Stack Overflow

Simpliest way to get Stack Overlow is:

```cpp
int foo(i) {
    return foo(i + 1)
}

foo(0)
```

# Lecture 1 - Metaprogramming

**Metaprogramming** - writing code that can generate/manipulate other code as data at _compile time_, rather than at runtime.

## Static Typing

In general, C++'s type system is primarily statically typed.

**Static Typing**: The majority of type checks, such as ensuring that variables are of the expected type, occur during compilation.

**Dynamic Typing (Runtime)**: In cases involving polymorphism (like using virtual functions), C++ can perform type checks at runtime. This is typically done using dynamic_cast, which is used to safely downcast from a base class pointer or reference to a derived class pointer or reference.

## Macroses (preprocessor stage)

**Macros** - preprocessor directives that provide a way to define reusable code snippets.  
They are not evaluated during runtime; instead, they are expanded by the **preprocessor** before the actual compilation process begins.

Macros don't have any concept of types, they simply replace the _text_.

### Examples

- `#include` basically includes all program code into the file.
- `#define PI 3.14` defining constants
- `#ifdef, #endif` conditional compilation
- `#define SQUARE(x) ((x)*(x))` function-like macros

### Macros as metaprogramming

Macros is metaprogramming if they involve:

- conditional logic
- code generation

that happens before the actual program is compiled.

```cpp
#ifdef __unix__
#include <unistd.h>
#elif defined _WIN32
#include <windows.h>
#else
#error "Unsupported platform"
#endif
```

## `template`s

### Turing Completeness

**Turing complete** programming language is one that can perform any computation that a **Turing machine**: conditionals (if-else), recursion, and the ability to store and manipulate data.

SQL is not turing complete, because it has no loops (recursion).

---

`template`s - Turing complete because they allow for metaprogramming — performing computations at _compile time_.

```cpp
template <int N>
struct Factorial {
    constexpr static int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    constexpr static int value = 1;
};

Factorial<5>::value;
```

```cpp
template <int N>
constexpr int Factorial() {
    return N * Factorial<N - 1>();
}

template <>
constexpr int Factorial<0>() {
    return 1;
}

Factorial<5>();
```

### Overload resolution

**Overload resolution**: full specialization $\rightarrow$ partial specialization $\rightarrow$ general template.  
Within classes there is no order.

```cpp
// 1st attempt: fail
// Full specialization (most specific match)
template <>
void foo<int>(int value) { ... }

// 2nd attempt: fail
// Partial specialization (less specific match)
template <typename T>
void foo(T* value) { ... }

// 3rd attempt: success
// General template (most general match)
template <typename T>
void foo(T value) { ... }

foo("a");
```

### SFINAE

**SFINAE** - Substitution Failure Is Not An Error: tries to substitute into appropriate template, if results in type compilation error, continues searching for other possible matches.

```cpp
template <class It, class Category>
struct IsIteratorCategory {
    constexpr static bool kValue =
        std::is_base_of<Category, typename std::iterator_traits<It>::iterator_category>::value;
};

template <class It>
using IsRandomAccessIterator = IsIteratorCategory<It, std::random_access_iterator_tag>;

template <class It>
using IsBidirectionalIterator = IsIteratorCategory<It, std::bidirectional_iterator_tag>;

template <class It>
using IsInputIterator = IsIteratorCategory<It, std::input_iterator_tag>;

template <class It>
constexpr typename std::enable_if_t<IsRandomAccessIterator<It>::kValue> 
Advance(It& iterator, ptrdiff_t n) { ... }

template <class It>
constexpr typename std::enable_if_t<!IsRandomAccessIterator<It>::kValue &&
                                    IsBidirectionalIterator<It>::kValue>
Advance(It& iterator, ptrdiff_t n) { ... }

template <class It>
constexpr typename std::enable_if_t<!IsBidirectionalIterator<It>::kValue && IsInputIterator<It>::kValue>
Advance(It& iterator, ptrdiff_t n) { ... }
```

### `concept` & `requires` (C++20)

`concept` refers to a compile-time constraint that specifies the `require`ments for template arguments.

Before C++20 it was done through **SFINAE**.

```cpp
#include <concepts>

template <class P, class T>
concept Predicate = requires(P p, T t) {
    { p(t) } -> std::same_as<bool>;
    // std::convertible_to<bool> is not possible, since it should be explicit `bool`
};

// ---

template <class T>
concept Indexable = (requires(T t, std::size_t index) {
    { t[index] };   // regardless of its return type
} && !std::is_void_v<decltype(std::declval<T>()[std::declval<std::size_t>()])>) || (requires(T t) {
    { t.begin() } -> std::random_access_iterator;
} && requires(T t) {
    { t.end() } -> std::random_access_iterator;
});

// ---

template <class T>
struct IsSerializableToJson : std::false_type {};

template <>
struct IsSerializableToJson<bool> : std::true_type {};

template <class T>
    requires std::is_arithmetic<T>::value
struct IsSerializableToJson<T> : std::true_type {};

template <class T>
concept IsStringLike =
    std::convertible_to<T, std::string> || std::convertible_to<T, std::string_view>;

template <class T>
    requires IsStringLike<T>
struct IsSerializableToJson<T> : std::true_type {};

template <class T>
struct IsSerializableToJson<std::optional<T>> : IsSerializableToJson<T> {};

template <class... Args>
struct IsSerializableToJson<std::variant<Args...>>
    : std::conjunction<IsSerializableToJson<Args>...> {};

// int[N]
template <class T, std::size_t N>
struct IsSerializableToJson<T[N]> : IsSerializableToJson<T> {};

template <typename T>
concept IsPair = std::is_same_v<T, std::pair<typename T::first_type, typename T::second_type>> &&
                 IsStringLike<typename T::first_type>;

// std::vector<int>
template <class T>
    requires std::ranges::range<T> && (!IsStringLike<T>) && (!IsPair<typename T::value_type>)
struct IsSerializableToJson<T> : IsSerializableToJson<typename T::value_type> {};

// std::map<std::string, int>
template <class T>
    requires std::ranges::range<T> && IsPair<typename T::value_type>
struct IsSerializableToJson<T> : IsSerializableToJson<typename T::value_type::second_type> {};

template <class T>
concept SerializableToJson = IsSerializableToJson<T>::value;
```

## `using`

1. Namespace Aliases:

    ```cpp
    using namespace std;
    ```

    Or specific members:

    ```cpp
    using std::cout;
    using std::endl;
    ```

2. Type Aliases (better than `typedef`)

    ```cpp
    using IntVec = std::vector<int>
    ```

3. Template Aliases:

    ```cpp
    template<typename T>
    struct Some {
        using type = T
        using ptr = T*;
    }
    ```

4. **Name hiding** - by default, when a derived class declares a function with the same name as a function in the base class, all overloads of that function from the base class are hidden.

    `using Parent::Func` unhides all `Parent` functions `Func` from derived class.

    ```cpp
    class Parent {
    public:
        void display();
        void display(int x);
    };

    class Derived : public Parent {
    public:
        using Parent::display;  // Brings Parent::display() and Parent::display(int) into Derived's scope

        void display(double d) { ... }
    };
    ```

5. Resolving ambiguity:

    ```cpp
    class C : public A, public B {
    public:
        using A::foo; // Resolves ambiguity if `foo` was in A and B
    };
    ```

## `constexpr` (C++11)

`constexpr` - keyword guarantees compile-time evaluation _if possible_.  
It will still work correctly at runtime if the computation can't be done during compilation.

```cpp
constexpr int square(int x) {
    return x * x;
}

square(5);      // compile time
std::cin >> y;
square(y);      // runtime
```

> Best practice: use when there’s a clear need for compile-time computation, rather than applying it everywhere for the sake of optimization.

- `constexpr` $\rightarrow$ `const`
- `constexpr` functions $\rightarrow$ `inline`

### `constexpr` with dynamic memory

C++20 allows dynamic memory allocation inside `constexpr` _functions_.

```cpp
constexpr int* create_array(int size) {
    int* arr = new int[size];
    arr[0] = 10;
    return arr;
}

// arr is statistically allocated and known at compile time; arr points to the dynamically allocated memory
constexpr int* arr = create_array(10);  
```

But `constexpr` variable declaration _can't_ be used with non-`constexpr` dynamic allocation (like with pure `new`, `std::vector`, ...).

## `consteval`

`consteval` function (C++20) must be evaluated at compile time.

```cpp
consteval int square(int x) {
    return x * x;
}

square(5);      // compile time
std::cin >> y;
square(y);      // ERROR
```

## `decltype`

`decltype` - deduce the type of a variable or expression at compile time.

```cpp
const int x = 20;
decltype(x) y = x;  // `const int`
```

```cpp
int foo() {...};
decltype(foo) bar = foo;  // `int()`
```

```cpp
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}
```

## Return Type

### `auto` Return Type Deduction

C++14 introduced **return type deduction** for functions with `auto`.  
This allows you to omit the explicit return type and let the compiler deduce it based on the function body _if it can_.

```cpp
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;  // Return type is deduced based on a + b
}
```

But when compiler cannot deduce a _single_ return type, we need **explicit return type** OR **trailing return type**.

### Trailing Return Type Deduction

Using `auto` with a trailing return type is useful when the return type is not easily known or depends on templates or complex expressions.

`std::declval` - type deduction without evaluating.

```cpp
template <typename T, typename U>
// std::optional<decltype(std::declval<T>() + std::declval<U>())> add(T a, U b) {
auto add(T a, U b) -> std::optional<decltype(std::declval<T>() + std::declval<U>())> {
    return a + b;
}
```

Basically, everything that you can do with `auto` + `->` can be done without it just specifying return type, so look for it.

#### lambda

```cpp
auto lambda = []() -> std::optional<int> {...};
```

can be done without trailing return type:

```cpp
std::function<std::optional<int>()> lambda = []() {...};
```

## Type Traits

```cpp
#include <type_traits>
```

**Type traits** - functionality to:

1. query
2. modify
3. enable conditional compilation (SFINAE)

of types at compile-time.

### Query

```cpp
std::is_integral<T>::value;
std::is_class<T>::value;
std::is_pointer<T>::value;
std::is_same<T, U>::value;
```

`value` member - `constexpr` boolean.  
`std::...::value` == `std::..._v`

### Modify

```cpp
using U = std::remove_const<T>::type;
using U = std::remove_reference<T>::type;
using U = std::add_pointer<T>::type;
using U = std::decay<T>::type;              // remove_const + remove_reference + remove_cv
```

### Conditional Compilation

Basically, SFINAE is used here.

```cpp
std::enable_if_t<std::is_integral<T>::value, void>;
std::is_convertible<int, double>::value;
using U = std::conditional_t<std::is_integral<T>::value, int, double>;  // T is `int` -> U is `int`, otherwise `double`
```

# Lecture 2 - Variadic `template`s (Metaprogramming)

```cpp
template <typename... Args>                 // **variadic template parameter**
void print(Args... args) {                  // **parameter pack**   (C++11 feature)
    (std::cout << ... << args) << '\n';     // **fold expression**  (C++17 feature)
}
```

The types of args are deduced automatically.

## Expanding Parameter Pack

```cpp
template <typename... Args>
void foo(Args&&... args) {
    bar(std::forward<Args>(args)...);
    bar(std::forward<decltype(args)>(args)...); // same
}
```

## Recursive Unpacking

Recursion is often used to process each element in the pack:

```cpp
// base case overload
template <typename T>
void print(T& t) {
    std::cout << t << '\n';
}

template <typename T, typename... Args>
void print(T& t, Args&... args) {
    std::cout << t << " ";
    print(args...);
}
```

## Fold Expressions

```cpp
(pack op ...) -> (p0 op (p1 op (p2 op p3)));
(... op pack) -> (((p0 op p1) op p2) op p3);
(pack op ... op init) -> (p0 op (p1 op (p2 op (p3 op init))))
(init op ... op pack) -> ((((init op p0) op p1) op p2) op p3);
```

Everything that can be done via fold expression, can be done using recursion.

## `initializer_list`

```cpp
#include <initializer_list>

template <typename T>
void print(std::initializer_list<T> args) {
    for (const auto& arg : args) {
        std::cout << arg << " ";
    }
}
```

# Lecture 3 - Concurrency

## `std::thread`

- accepts any callable object

- starts executing immediately upon creation; if thread can't be created $\rightarrow$ `system_error`

- args are copied || moved; for referencing use `std::ref`, `std::cref`

- always make sure to either `join` or `detach` threads, otherwise `std::terminate` because `std::thread` is not RAII

```cpp
#include <thread>

template <typename T, typename U>
void foo(T& t, U u);

int main() {
    std::thread t(foo, std::ref(some_obj), "2");
    t.join();   // ensure main waits for thread 't' to finish before proceeding
    return 0;
}
```

Unfinished thread of finished program $\rightarrow$ UB.

C++11 introduced the concept of **memory orderings**.

### `std::jthread` (C++20)

`std::jthread`:

- is superset of `std::thread`
- automatically `join`s the thread when it goes out of scope
- supports cooperative cancellation via `request_stop` and `stop_token`

### IDs

```cpp
int main() {
    std::jthread t1( []{
        std::cout << std::this_thread::get_id() << '\n';    // "1234"
    })
    std::jthread t2;
    std::cout << t1.get_id() << '\n';                       // "1234"
    std::cout << t2.get_id() << '\n';                       // thread::id of a non-executing thread

    return 0;
}
```

## `std::shared_mutex`

Shared lock for reading.  
Exclusive (1) lock for writing.

```cpp
void read_data() {
    std::shared_lock<std::shared_mutex> lock(mtx);
}

void write_data(int value) {
    std::unique_lock<std::shared_mutex> lock(mtx);
}
```

## `std::conditional_variable`

`std::condition_variable` works with `std::mutex`:

- `wait`: A thread can wait on a condition variable to be notified.
- `notify_one`: Notify one waiting thread.
- `notify_all`: Notify all waiting threads.

## `std::future` & `std::promise`

They provide a _one-time_ communication channel between two threads.

- `std::promise` sets value/excpetion
- `std::future` retrieves it

```cpp
#include <future>

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::jthread t([&prom]() {
        prom.set_value(1);
        // prom.set_exception(...)
    });

    std::cout << fut.get() << '\n'

    t.join()
    return 0;
}
```

### `std::async`

`std::async` is designed to launch a function:

- `std::launch::async`: **asynchronously** - in a separate thread;  
    starts immediately
- `std::launch::deferred`: **synchronously** - in the calling thread;  
    executed lazily: only runs when `get` or `wait` is called

```cpp
std::future<int> fut = std::async(
    std::launch policy,     // std::launch::async or `std::launch::deferred`
    func, 
    args...
);
```

## False Sharing

(Read from semester01/akos/notes.md)

Be aware of this, since writing to the same container (e.g., `std::vector`) might cause this even though none of the indecies of different threads overlap.
