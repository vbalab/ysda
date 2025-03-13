<!-- markdownlint-disable MD025, MD001, MD024 -->

# Lecture 1 - Philosophy

Zero-Cost abstraction  
Safety guarantee: while there is no "unsafe" in code, it's safe. But in standard library there's lots of unsafe (even push to a vector).  
But even with unsafe rust has "Safe abstructions" which prevent any UB  

> \$ cargo init dir_name  
> \$ cargo run

println! is a macros (not a function)  
variable in **let** are **const** by default so you should supply **mut**  
If you want program to stop when encountering error [called "panic"] use **.unwrap()** or **.expect("this error due to ...")**  

# Lecture 2 - Lifetimes, ownership, borrow checker

## C++: Use After Move

```cpp
std::vector<int> v = {1, 2, 3};
int s = sum(std::move(v));
// `v` is in "valid but unspecified state"
v.clear();
// now `v` is in valid and specified state
```

## Ownership

In rust each value has a single owner.

```rust
fn main() {
    let s = String::from("Hello, Rust!"); // `s` owns the String
    let t = s; // Ownership is moved to `t`
    // println!("{}", s); // Error! `s` no longer owns the String
}
```

```rust
fn drop<I>(_: I) {}
```

This is destructor, because it moves ownership into `drop` which scope immidiately ends and destructs given value.

## Borrowing

Instead of transferring ownership, Rust allows you to borrow a value.

Forms of borrowing:

- Immutable (`&T`)
- Mutable (`&mut T`)

We can have N `&T` borrows OR only 1 `&mut T`.  
This prevents data races.

### Copy

- Invocation: Implicit
- Performance: Cheap (bitwise copy)
- Scope: For simple, stack-only data

```rust
let x = 42;  // `i32` implements `Copy`
let y = x;   // `x` is copied implicitly
```

### Clone

- Invocation: Explicit (.clone())
- Performance: Potentially expensive (deep copy)
- Scope: For complex, heap-allocated data

`Clone` trait is used for explicitly creating deep copies of *heap*-allocated data.

This operation is more expensive than a simple memory copy.

```rust
#[derive(Clone)]
struct Foo {...}

let v = vec![1, 2, 3];
let v2 = v.clone();
```

## Lifetime

**Lifetime** - scope for which a reference is valid.

`'a` specifies that the references `x` and `y` must live at least as long as the returned reference.

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {...}

fn main() {
    let string1 = String::from("long string");
    let string2 = "short";
    let result = longest(&string1, string2);  // `result` has the same lifetime as the shorter borrow
}
```

Without specifying lifetimes, they're assumed to be different:

```rust
fn get_crd(name: u8, p: &Point) -> &i32 {...}
// the same as
fn get_crd<'a>(name: u8, p: &'a Point) -> &'a i32 {...}    
```

## Borrow checker

**Borrow checker** enforces the rules of ownership, borrowing, and lifetimes. It ensures:

- References are valid and safe.
- Lifetimes are compatible.

## Reborrowing

### 1. Immutable reborrowing

```rust
let mut value = 42;
let ref = &mut value;
let imm_ref = &*ref;
```

### 2. Mutable reborrowing

```rust
let mut value = 42;
let ref = &mut value;
let ref_new = &mut *ref;    // 
```

The mutable reference `&mut *` temporarily "reborrows" ownership from `ref` to `ref_new` while ensuring Rust's borrowing rules are not violated.

While working with `ref_new`, `ref` is not valid. When `ref` started to being used $\rightarrow$ `ref_new` is invalidated.

### Implicit reborrowing

```rust
fn bar(user: &mut User) {
    foo(user);
}
// the same as:
fn bar(user: &mut User) {
    foo(&mut *user);
}
```

# Lecture 3 - Slices

## Slice

Slice is represented as a tuple:

- pointer to the first element
- length of the slice

`&array[1..4]` is represented as (ptr, 3) where ptr points to 1st element.

```rust
let array = [1, 2, 3, 4, 5];
let slice = &array[1..4];       // Borrow elements 1, 2, 3
let slice = &array[..];        // [1, 2, 3, 4, 5]

let slice = &mut array[1..4];   // this or multy immutables
```

```rust
fn print_slice(slice: &[i32]) {
    for &item in slice {...}
}
```

### Split slices

```rust
let array = [1, 2, 3, 4, 5];

let s1 = &mut array[..4];
let s2 = &mut array[4..];               // error of double borrowing

let (s1, s2) = array.split_at_mut(4);   // proper way
```
