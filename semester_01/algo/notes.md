<!-- markdownlint-disable MD001 -->

# Practice

## Formatting

Be sure .clang-format & .clang-tidy yaml files are placed in wd or in parents of wd.

```bash
clang-format -i TASK_NAME.cpp
clang-tidy TASK_NAME.cpp -- -std=c++20
```

[If this doesn't work, try to do `$ clang-...` from directory with clang-_.json]

## Compiling

```bash
clang++ -Wall -Wextra -pedantic -std=c++20 TASK_NAME.cpp
```

To check execution time:

```bash
time ./a.out
```

## Debuging

[  
in `.vscode/tasks.json` in line "args" add

```txt
                "--std",
                "c++20",
```

]

## Testing

[How to run tests](https://chatgpt.com/c/66f90f7f-4494-8002-aab9-6542c8e9a328?conversationId=66f90f7f-4494-8002-aab9-6542c8e9a328&model=gpt-4o)

Be sure that you have:

```bash
sudo dnf install gtest gtest-devel
```

Comment out main() function in TASK_NAME.cpp, then

```bash
g++ -std=c++17 -lgtest -lgtest_main -pthread TASK_NAME_test.cpp -o test_runner
./test_runner
```

# Notes

$\texttt{capacity}$ of `std::vector` doesn't get decreased when $\texttt{.resize(), .clear()}$, it could be only increased (that's why allocated memory doesn't decrease).

## Backtracking

```py
def find_solutions(parameters):
    if valid_solution(parameters):
        store_solution(parameters)
        return

    for choice in all_choices(parameters):
        if valid_choice(choice, parameters):
            apply_choice(choice, parameters)
            find_solutions(parameters)
            backtrack(choice, parameters)
    return
```

# Seminar 1 - Backpack

## Binary Decomposition (as Efficient Resource Allocation)

If we're given some money say number $N$ and we need to give some payment that consists of this money, we don't need to give money coin by coin, but rather we can split our money into butches of $2^k$ and some reminder of $N - 2^{k_max}$ that get's splitted too recursively.  
This way, we can get to payment with $N \rightarrow \log(N)$, since with $2^k$ we can represent any number $<N$.

# Seminar 2 - DIDN'T WATCHED

# Seminar 3 - Rectangles

## Scan line

Used to solve computational geometry problems, especially those that involve intervals, rectangles, or overlapping objects.

```cpp
struct Event {
    int time;
    int type;                                       // 1 for start, -1 for end
};

for (auto event : events) {                         // events are sorted by time
    currentOverlap += event.type;                   // Add 1 for start, subtract 1 for end
    maxOverlap = max(maxOverlap, currentOverlap);   // Update maximum overlap
}
```

# Seminar 4 - DIDN'T WATCHED

# Seminar 5 - Squares

Write **bruteforce** solution fastly to test small testcases.

## Lazy delete

Hold a binary mask (0, 1, 1, 0, 0) of where index suggests that this object should be deleted when observed.  
Should be used when there is no value of deleting an element before observing it.

# Lecture 1 - RAM Machine

$ time(N) = max(time(i))$, where $i ∈ Input_N$

**Arithmetic Logic Unit** (ALU) is a component of CPU. It is responsible for performing all the arithmetic and logical operations that occur during program execution.

## RAM Machine

Bit Volume of RAM = $M * W$ (number of cells \* bits in one **Machine Word**)

$W$ is assumed not to be constant in RAM machine model. So if somethink depends on $W$, then it's not a constant.  
Multipling 2 numbers of $W$ length takes constant time. If we're talking about any 2 numbers (int64, int128, ...) it would take non-constant time in general.

1. **Registers** (RF) = $R_i$  
   Registers are located in CPUs: In actual processors, there are a limited number of registers that are used for high-speed data manipulation, since accessing registers is quicker than accessing main memory.  
   They are used for temporary storage of data, particularly for storing operands during computations and keeping track of intermediate results.  
   They store the same blocks of $W$ length.

2. Memory

3. Instructions:  
   $R_i$ = value  
   $R_i$ = $R_j$  
   3.1. LOAD(i, j): $R_i$ $:=$ RAM$$$R_j$$$, where $R_i$ has address, $R_j$ has value. [Copy a value from memory to a register.]  
   3.2. STORE(i, j): RAM$$$R_i$$$ $:=$ $R_j$, where $R_i$ has address, $R_j$ has value. [Copy a value from a register to memory.]  
   3.3. ADD, SUBTRACT, MULTIPLY, DIVIDE  
   3.4. JUMP: Transfer control to a different instruction.  
   3.5. BREQ: if $R_i == R_j$ then do JUMP
   3.5. HALT: Stop execution.

4. **Program Counter** (PC)

Condition for RAM machine: $2^W \geq M$, otherwise we won't reach all addresses of M [OR you should come up with different addressing of memory cells].  
Typically now W=64 bits, which is more than enough (terrabite is ~2^40).

In real life between CPU and RAM there is a number of layers of some memory and cached memory.

# Lecture 2 - Complexity, Sorting

## Complexity

Some complexity measure:  
$Complexity_{N} = max_{input∈Inputs_{N}}(avg_{seed}(time(input, seed)))$

## Amortized complexity

Example: vector.push() - $O(1)$, unless size reaches capacity and need to relocate vector - $O(n)$.  
How to count for that in estimation of $\sum_{i}^{n}cost(a_i) \leqslant ?$?

### 1. Accounting (Banker’s) Method

$\sum_{i}^{n}cost(a_i) \leqslant n * C$ where C is some function of n, typically constant.

Method assigns amortized cost to each operation such that we store surplus cost (credit) during cheaper operations to pay for more expensive ones later. The idea is to "overcharge" inexpensive operations and save the surplus for later use.

### 2. Potential Method

The potential method uses a potential function $ \Phi $ to track the "stored energy" or potential future work in the data structure. The difference in potential before and after an operation is used to define the amortized cost.

Let $ cost'(a*i) = cost(a_i) + (\Phi_i - \Phi*{i - 1})$, then $\sum_{i}^{n}cost(a_i) = \sum_{i}^{n}(cost'(a_i) + \Phi_{i - 1} - \Phi_{i}) = \sum_{i}^{n}cost'(a_i) + \Phi_{0} - \Phi_{n}$

Where:

- $ \Phi_i $ is the potential after the $ i $-th operation;
- $ \Phi\_{i-1} $ is the potential before the operation.

## Sorting

### QuickSort

Uses **Divide & Conquer**. Recursive.

Selects a pivot element and **partitioning** the array into two sub-arrays: left & right from pivot. There a many schemes of partition.

1. Hoare partitioning

   - $ i, j $ that start at the ends of the array and move towards each other until they meet.

   ```python
   function partition(arr, low, high)
       pivot = arr[low]
       i = low - 1
       j = high + 1
       while true
           repeat
               i = i + 1
           until arr[i] >= pivot
           repeat
               j = j - 1
           until arr[j] <= pivot
           if i >= j
               return j
           swap(arr[i], arr[j])
   ```

2. Lomuto partitioning

   - $i$ at the beginning partitions into elements less than the pivot and elements greater than the pivot.

   ```python
   function partition(arr, low, high)
       pivot = arr[high]
       i = low - 1
       for j = low to high - 1
           if arr[j] <= pivot
               i = i + 1
               swap(arr[i], arr[j])
       swap(arr[i + 1], arr[high])
       return i + 1
   ```

### Dutch Flag Sort

Sorts an array containing $n$ distinct keys (e.g., 0, 1, 2, ...) based on some criteria (many pivots). It partitions the array into $n$ sections.

### IntroSort

If QuickSort (on average $n \log n$ with very low constant, worst N^2) seems like N^2 use HeapSort (always $n \log n$, but high constant).

# Lecture 3 - Sorting

## Tail recursion elimintaion

**Tail recursion** is a special kind of recursion where the recursive call is the last operation in the function.

Example of tail recursion:

```python
def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    return factorial_tail(n - 1, acc * n)
```

**Tail recursion elimination (or optimization)** is a technique where the compiler or interpreter transforms a tail-recursive function into an iterative loop.

```python
def factorial_iterative(n):
    acc = 1
    while n > 0:
        acc *= n
        n -= 1
    return acc
```

Benefits: Reduces memory usage and avoids stack overflow for deep recursions.

If it was:

```txt
   a
  / \
 b   c
/ \ / \
d e f g
```

now it's

```txt
    a,c,g
   /     \
  b,e     f
 /
d
```

Often, such optimization is done by compiler.

## Selects

Algorithms that are used to find the $k$-th smallest element in an unsorted array. They are in $O(n)$.

### QuickSelect

Take QuickSort, but do nothing on a part which doesn't contain the number.

### Medians of median (Medians of five)

Recursively:

1. Divide the list into groups of five
2. Sort each group and find the median of each group
3. Create a list of medians

## Master Theorem

$$
T(n)=aT(\frac{n}{b}) + O(n^d)
$$

- $T(n)$ is the time complexity.
- $a$ is the number of subproblems in the recursion.
- $\frac{n}{b}$ is the size of each subproblem (the problem size is reduced by a factor of $b$).
- $O(n^d)$ represents the cost of dividing the problem and combining the results.

There are three main cases based on the relationship between $a$, $b^d$, and $n^d$.

### Case 1: $a > b^d$ (Recursive work dominates)

The time complexity is:

$$
T(n) = O\left(n^{\log_b a}\right)
$$

### Case 2: $a = b^d$ (Balanced work)

The time complexity is:

$$
T(n) = O\left(n^d \log n\right)
$$

### Case 3: $a < b^d$ (Non-recursive work dominates)

The time complexity is:

$$
T(n) = O\left(n^d\right)
$$

## MergeSort & Exponential BS

Recursively:

1. Divide: Split the array into two halves.

2. Conquer: Recursively sort each half.

3. Combine: Merge the two sorted halves (with sizes $n_1$ and $n_2$) back together into one sorted array. Here use **exponential BS** instead of linear way, because $O(n_1 log(\frac{n_1 + n_2}{n_1}))$ where $n_1 \le n_2$ is better than $O(n_1 + n_2)$ when $n_1$ is approximately $n_2$. And since our arrays are halves $ \rightarrow $ $n_1$ is $n_2$ with accuracy by 1.

### Exponential BS

Check indices at exponentially increasing intervals, i.e., $1, 2, 4, 8, \dots, 2^k$.

## MergeSort vs QuickSort

MergeSort is $O(n \log n)$ in worst case, but it's not better than QuickSort (that is $O(n \log n)$ only on avg. and $O(n^2)$ in worst), because it's not in-place sorting, so MergeSort $O(n)$ by memory, when QuickSort is $O(\log n)$.

# Lecture 4 - Hashing

direct addressing

## Hash function

Hash function:

1. $ h: K \rightarrow \{0, ..., m-1\} $ (i.e. keys $ \rightarrow $ buckets)
2. $ h: K \rightarrow \texttt{int}$ , $\texttt{int} \mod m \rightarrow \{0, ..., m-1\} $

Hash functions in programming languages are very simple in order to be fast (unlike hash functions in crypto). In C++ hash of $\texttt{int}$ is the same $\texttt{int}$.

### Collisions

If $|K| > m \rightarrow \exists k',k'' \in K $ s.t. $ h(k') == h(k'')$ - **collision**, could happen even if $|K| < m $.

$ \frac{n}{m} $ - load factor.

### Universal family

$H$ - **universal family** if $ \forall h \in H, \forall k' \neq k'' \in K: P_h(h(k') = h(k'')) = \frac{1}{m}$.

Example of a universal family:
$$ h\_{a,b}(x) = ((a \cdot x + b) \mod p) \mod m $$

- $p$: prime number $\ge$ any possible key.
- a & b: randomly chosen integers such that $ 1 \leq a < p $, $ 0 \leq b < p $.

## Chaining

Let $l$ - length of chain in bucket, $l_{max} \ge [\frac{n}{m}]$.  
Then: $\texttt{put, get, remove}$ are $ O(l) $ on average ($ O(n) $ in worst case, when hashes are the same).

Let $ h \in H \subset \{0, ..., m - 1\}^k $, where $H$ - universal family.
Then $ E_h(l) = \frac{n}{m} + 1$, which is constant if $ n = q m $.

## Perfect Hashing

Is such $ h: K \rightarrow \{0, ..., m-1\} $ that $\forall a' \neq a'' \in A \subset K: h(a') \neq h(a'') $.

The idea is to find such $h$ for $A$ so there would be no collisions.

$P(h$ has no collisions on $A) = P( collisions =0) = 1 - P( collisions \ge 1) \ge 1 - \frac{E(collisions)}{1} = 1 - \frac{n(n-1)}{2m} \ge \frac{1}{2}$, if $m \ge n^2$.

We set $m \ge n^2$ trying different $h$ until there are no collisions on $A$, the probability is high enough.  
But this approach requires $m \ge n^2$, so it's bad.

### FKS Hashing (used)

A two-level hashing scheme for **static sets** to achieve **O(1)** lookups.

Structure:

1. First-Level Hashing

   - Maps $ n $ keys into $ m $ buckets using a universal hash function:
     $$
     h_1(x) = (a_1 \cdot x + b_1) \mod p \mod m
     $$
   - $ a_1 $ and $ b_1 $ are random, $ p $ is prime.

2. Second-Level Hashing
   - Uses a perfect hash function to resolve collisions in each bucket:
     $$
     h_2(x) = (a_2 \cdot x + b_2) \mod p \mod m_i^2
     $$

### CMH hashing (not used)

Structure:

1. **Hash Functions**: Uses $ d $ independent hash functions $ h_1, h_2, \ldots, h_d $ mapping elements to a range of $ [0, w-1] $:

   $$
   h_i(x) \rightarrow [0, w-1] \quad \text{for } i = 1, 2, \ldots, d
   $$

   where $ w $ is the width of each row, and $ d $ is the number of rows.

2. **Count Array**: Maintains a 2D array $ C $ of size $ d \times w $ initialized to zero. Each hash function maps an element to an index in a different row, and the corresponding counter is incremented.

Use Case:
CMH is suitable for **approximate frequency counting** in applications like network traffic analysis, natural language processing, and data stream analytics.

# Lecture 5 - Hashing

## Bloom Filter

**Bloom filter** - test set membership data structure with a possibility of false positives (FP) but no false negatives (FN).

Structure:

1. bit array $ B $ of size $ m $, initially set to all zeros.

2. $ k $ independent hash functions $ h_1, h_2, \ldots, h_k $

3. $\texttt{insert}$:

   $$
   \forall i = 1 \text{ to } k: \quad B[h_i(x)] = 1
   $$

4. $\texttt{contains}$:
   $$
   \text{If all } B[h_i(x)] = 1 \text{ for } i = 1, 2, \ldots, k, \text{ then } x \text{ is possibly in the set, else it is definitely not.}
   $$

After inserting $ n $ elements:

$$
P(FP) \approx \left( 1 - e^{-kn/m} \right)^k
$$

$k$ that minimizes FP:

$$
k_{optimal} = \frac{m}{n} \ln 2
$$

Complexity:

- **Space**: $ O(m) $
- **Insert Time**: $ O(k) $
- **Query Time**: $ O(k) $

Bloom filters are used for **efficient membership testing** in scenarios like database queries, cache filtering, and network security (e.g., checking for malicious URLs).

## KV-storage

A KV-storage algorithm manages key-value pairs by:

- Data Structure: Uses hash tables (in-memory) or B-trees (disk storage) for efficient lookups and storage.
- Concurrency: Uses locks or lock-free methods for safe, concurrent operations.
- Scaling: Implements **sharding** and replication for scalability and reliability.
- Compaction: Periodically reclaims storage space and optimizes performance.

## Count Min Sketch

It's used for processing large streams when it's impractical to store or count each item exactly.

CMS is a probabilistic data structure for estimating the frequency of elements in a data stream, with limited memory.

Key points:

- Data Structure: Uses a 2D array with multiple hash functions mapping elements to rows, where counts are updated.
- Estimation: Returns the minimum count across rows for each element, reducing overestimation errors.
- Efficiency: Provides approximate frequency counts with sublinear memory and constant time complexity.

Applications: Ideal for large-scale data streams, like tracking popular items in network traffic or web analytics.

$$
O\left(\frac{1}{\epsilon} \log \frac{1}{\delta}\right)
$$

$$
P\left[\hat{\text{count}}(x) - \text{count}(x) \geq \epsilon n\right] \leq \delta
$$

# Lecture 6

## Misra-Gries Algorithm for Finding Frequent Items

The **Misra-Gries** algorithm is an efficient algorithm used to find frequent items in a data stream. It approximates the frequency of items with a specified error bound and can be used in scenarios where the exact frequency is not required but an approximate count suffices.

#### Notations and Definitions

- Let $ C $ be a data structure that keeps track of potential frequent items and their approximate counts.
- Each entry $ C[x] $ corresponds to an item $ x $ with an associated counter.
- When a new item arrives, $ C[x] $ is incremented if $ x $ is already tracked; otherwise, an existing entry is replaced or all counters are decremented.
- **Parameters**:
  - $ \varepsilon $: Error tolerance (relative approximation error).
  - $ m $: Number of counters (inversely proportional to $ \varepsilon $), approximately $ m \approx \frac{1}{\varepsilon} $.

#### Algorithm

1. For each incoming item $ x $:
   - If $ x $ is in $ C $, increment $ C[x] $ by 1.
   - If $ x $ is not in $ C $ and $ C $ has fewer than $ m $ items, add $ x $ to $ C $ with a counter of 1.
   - If $ x $ is not in $ C $ and $ C $ is full, decrement all counters in $ C $ by 1.

#### Key Properties and Inequalities

- $ \forall x: $ $ \text{count}(x) \geq \hat{\text{count}}(x) $
- Approximation Bound:

  $$
  \forall x \quad (\text{count}(x) - \hat{\text{count}}(x)) \cdot (m + 1) \leq \sum_y (\text{count}(y) - \hat{\text{count}}(y))
  $$

- Error Bound:

  $$
  \forall x \quad \text{count}(x) - \hat{\text{count}}(x) \leq \frac{n}{m + 1} \leq n \varepsilon
  $$

  where $ n $ is the total number of items processed.

- Time Complexity: $ O\left(\frac{1}{\varepsilon}\right) $
- Space Complexity: $ O\left(\frac{1}{\varepsilon}\right) $

## Inversed Bloom Filter

# Lecture 7 - Trees

Different approaches to balancing and efficiency in tree-based data structures:

## 1. Strict guarantees - Worst Case (AVL, RB, ...)

- Data structures like **AVL Trees** and **Red-Black Trees** guarantee a worst-case logarithmic height for all operations.
- These structures maintain strict balancing rules to ensure that operations like search, insertion, and deletion have a time complexity of $O(\log n)$ in the worst case.
- They are deterministic and provide consistency, making them suitable for applications where the worst-case performance is critical.

## 2. Randomized (Treap = Cartesian (Декартово) Tree)

**Treap** is BST that: BST & heap . It uses pairs $(k_i, p_i)$, where $p_i$ - randomized priority from 0 to 1 to maintain balance.

BST property to $k_i$ and max-heap property to $p_i$ (the priority of the node must be greater than or equal to the priorities of its children).

There is a unique Treap for each vector of pairs.

The balance of a treap is probabilistic, meaning it doesn't strictly enforce balancing rules but achieves $O(\log n)$ expected time complexity for operations through randomization.

## 3. Amortized (Splay, Scapegoat)

### Scapegoat Trees

- These maintain balance by occasionally rebuilding parts of the tree after certain operations.
- Ensure $O(\log n)$ amortized complexity while avoiding the overhead of strict rebalancing after every operation.

### Splay Trees

- Self-adjusting binary search trees that do not explicitly store balance information.
- Frequently accessed elements are moved closer to the root - they are moved by delta_depth=2 upper, providing $O(\log n)$ amortized time for operations.

Lookup = OrdinaryLookup + Splay

- zig
- zig-zig
- zig-zag

# Lecture 8 - RMQ (Read Minimal Query) & LCA (Least Common Ancestor)

## Euler Tour

```cpp
class EulerTour {
public:
    unordered_map<int, pair<int, int>> nodeTimes;
    vector<int> eulerTourSequence;
    int timeCounter = 0;

    void dfs(TreeNode* node) {
        nodeTimes[node->value].first = timeCounter++;
        eulerTourSequence.push_back(node->value);

        for (TreeNode* child : node->children) {
            dfs(child);
        }
        nodeTimes[node->value].second = timeCounter++;
        eulerTourSequence.push_back(node->value);
    }
};
```

```txt

    1
  / | \
 2  3  4
      /|\
     5 6 7

### Euler Tour Representation

| id    | 1 | 2 | 1 | 3 | 1 | 4 | 5 | 4 | 6 | 4 | 7 | 4 | 1 |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| depth | 0 | 1 | 0 | 1 | 0 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 0 |

```

## Read Minimal Query & Least Common Ancestor

- **Read Minimal Query (RMQ)**:
  For an array $ A[1 \dots n] $, the task is to answer:  
$ \text{RMQ}(i, j) = \min(A[i], A[i+1], \dots, A[j]) $

- **Least Common Ancestor (LCA)**: $ \texttt{LCA(x, y)} $ of tree.

LCA solution is RMQ on Euler Tour representation depth vector with changed order so that $ |a*i - a*{i+1}| = 1 $.

RMQ solution is LCA on Cartesian Tree (Treap) with priorities as values of $a_k$ and keys=0,...,n.

## Farach-Colton-Bender Algorithm for Sparse Precompute in RMQ

Precompute $ M(i, k) = \min\_{i \le j \le i + 2^k }\{a_i\} $, where $ M(i, k + 1) = \min\{ M(i, k), M(i + 2^k, k) \} $.

It takes $ O(n \log(n)) $ to precompute sparsly.

### Sparse Table Preprocessing

- **Sparse Table Definition**: $$ ST[i][0] = A[i] $$, $$ ST[i][j] = \min(ST[i][j-1], ST[i + 2^{j-1}][j-1]) $$

- **Precompute** for all $ 0 \leq j \leq \lfloor \log n \rfloor $: $$ ST[i][j] \text{ for } i = 1 \dots n, j = 0 \dots \lfloor \log n \rfloor $$

- **Query Formula**: $$ \text{RMQ}(L, R) = \min(ST[L][\lfloor \log (R-L+1) \rfloor], ST[R - 2^{\lfloor \log (R-L+1) \rfloor} + 1][\lfloor \log (R-L+1) \rfloor]) $$

---

### Block Division and Intra-block RMQ

- **Block Index**: $$ \text{block}(k) = \left\lfloor \frac{k}{\log n} \right\rfloor $$

- **Precompute Intra-block RMQ**: $$ P[i][j] = \text{argmin}(A[i], A[i+1], \dots, A[j]) $$

- Use **bit masks** to encode intra-block RMQs for constant-time lookup.

---

### Preprocessing Steps

1. **Sparse Table Construction**:

   - Precompute $ ST[i][j] $ for all ranges in $ O(n \log n) $.

2. **Intra-block RMQ**:

   - Precompute all RMQs inside blocks using $ O(2^{\log n}) $ bit masks.

3. **Query Reduction**:
   - Reduce global RMQ to intra-block RMQs and inter-block RMQs, both resolved in $ O(1) $.

---

### Final Complexity

- **Preprocessing Time**: $ O(n \log n) $
- **Query Time**: $ O(1) $
- **Space Complexity**: $ O(n \log n) $

# Lecture 9 - Graphs: BFS, DFS

$G(V, E)$, it's size is $|V| + |E|$.

Adjacency matrix $O(V^2)$ by memory is bad for sparce case.  
Using matricies as structure for graph is useless, since we're not interested in relation between any 2 arbitrary verticies.  

std::vector\<Vericies\>(std::vector\<VerticiesFromEachVertex\>) is good  
std::vector\<std::pair<v1, v2>\> is good

## BFS

```pseudo
Q := {s}
visited := ∅

while Q ≠ ∅:
    u := deque(Q)
    for (u, v) ∈ E:
        if ¬visited(v):
            enqueue(Q, v)
            visited.insert(v)
```

time: $O(V+E) \sim O(E)$, because it's considered that $V < E$, otherwise graph is strange.

## DFS

col: V $\rightarrow$ {W, G, B} (white, gray, black)

```pseudo
col(u) := G

for uv ∈ E:
    if col(v) = W:
        DFS(v)

col(u) := B
```

time: $O(V+E) \sim O(E)$

**Lemma** (of white path):  
If col(s) = W, then `DFS(s)` visits {v | s → v along white path}.

**Topological order** (for oriented graph):  
$ \pi: V → \{1, \dots, n\}$ - bijection  
$ \forall (u, v) \in E: \pi(u) < \pi(v) $. Could be >.

### Cycle detection (for oriented)

`DFS(u)`:

```pseudo
col(u) := G

for uv ∈ E:
    if col(v) = G:
        return "cycle"
    if col(v) = W:
        DFS(v)

col(u) := B
```

But this way we could lose some cycles, because the order of verticies from verticies can be different.  
We need to create vertex `s` and build edges from it to all other verticies and do `DFS(s)`.

### Topological order (for oriented)

`DFS(u)`:

```pseudo
col(u) := G

for uv ∈ E:
    if col(v) = W:
        DFS(v)

col(u) := B

$\pi$(u) := time++;
```

## Connectivity

**Strong connectivity**:  
$ u \sim v \iff u $ is reachable from $ v $ \& $ v $ is reachable from $ u $.

**Equivalence classes OR strongly connected component (SCC)** group vertices based on strong connectivity, where $\forall u, v$ belong to the same class if $u \sim v$.

**Condensation** of a directed graph is a new graph where each SCC of the original graph is represented as a single vertex.  
Edges between these vertices exist if there is an edge between the corresponding SCCs in the original graph. The condensation graph is always a **Directed Acyclic Graph (DAG)**.

Algorithm to find SCCs:

1. $DFS(s) \to \pi(v_1), \pi(v_2), \dots, \pi(v_n)$  
2. $\pi(v_1) > \pi(v_2) > \dots > \pi(v_n)$  
3.

```pseudo
reset $col$  
   for $i = 1 \dots n$:  
   if $col(v_i) = W$:  
   $DFS'(v_i)$  
```

where `DFS'(v_i)` goes in the other direction (in oriented graph).

# Lecture 10 - Graphs: Articulation points, DSU, Spanning Tree, Kruscal algorithm

## Edges types

DFS:

```pseudo
enter[u] := time++

col(u) := G

for uv ∈ E:
    if col(v) = W:
        DFS(v)

col(u) := B

exit[u] := time++
```

For understanding of who descendant ($u$) of who ($v$).

### Directed graph

1. tree edge ($col[u] = W$)
2. backward edge ($col[u] = G$)
3. forward edge ($col[u] = B$, $v$ - descendant of $u$)
4. cross edge ($col[u] = B$, $v$ - not descendant of $u$)

#### Laminar

$\forall A, B:$ only one is true:  
$ A \cap B = \emptyset \cup $  
$ A \subseteq B \cup $  
$ B \subseteq A $

### Undirected graph

1. tree (previous 1.-2.)
2. forward=backword (previous 2.-3.)

    ```txt
           (u)
            .
          ^ | \
         /  |  \
       2 |  |1 | 3
         \  |  /
          \ v v
            .
           (v)
    ```

There are no cross edges.

$G$ - acyclic $\iff$ no backward egdes.

## Articulation points

$v \in V$ - **articulation point** (a.p.), if $\#SCC(G-v) < \#SCC(G)$.

Find all a.p.'s in $O(V+E)$ using DFS:

```pseudo
enter[u] := time++
lambda[u] = enter[u]
col(u) := G

children := 0
is_articulation := false

for uv \in E:
    if col(v) = G:  # Back edge
        lambda[u] = min(lambda[u], enter[v])

    if col(v) = W:  # Tree edge
        DFS(v)
        children += 1
        lambda[u] = min(lambda[u], lambda[v])

        # Check articulation condition
        if lambda[v] >= enter[u] and parent[u] != null:
            is_articulation := true

if parent[u] = null and children > 1:
    is_articulation := true

if is_articulation:
    mark u as articulation point

col(u) := B
exit[u] := time++
```

`lambda[u]`: Lowest reachable timestamp from the DFS subtree rooted at `u`.

---

$e \in E$ between x & y - **bridge**, if $x \in \text{SSC}_1$ & $y \in \text{SSC}_2$.

Graph connectivity:

1. Static (solved via DSU)
2. Dynamic

    - Incremental
    - Decremental
    - Full

## Disjoint Set Union

**Disjoint Set Union (DSU) (Union Find)** is used to solve problems related to connectivity in graphs, such as determining whether two nodes belong to the same connected component.

`find(x)` just follows parent pointers until it finds a root.

`union(x, y)` - merges the sets (SCC) containing elements x and y: attaches one set’s root to the other set’s root.  
If x and y are already in the same set, no change is made.

### Pure DSU

```py
class DSU_Pure:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
    
    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            self.parent[rootY] = rootX
```

Worst-case time complexity for find and union can be $O(n)$ in cases where the tree becomes like a long chain.

### DSU + union by rank

```py
class DSU_WithRank:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n  # rank array
    
    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            # Attach the smaller rank tree under the larger rank tree
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                # If ranks are the same, attach one to the other and increase rank
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
```

`find` and `union` can be $O(\log n)$.

### DSU + union by rank + path compression

```py
class DSU_Full:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
```

The amortized time complexity for find and union operations is $O(1)$

## Spanning Tree

**Spanning Tree** is a subset of a connected $G(V, E)$ that includes all the vertices of the graph and is a tree.  
It is $G(V, E)$ without unnesessary part of $E$.

If we want to find Minimal Spanning Tree $\rightarrow$ use DFS.

### Kruscal algorithm

We have connected $G(V, E)$ has weighted edges with $C: E \rightarrow R_+$.  
We interested in finding Minimal Spanning Tree s.t. $\sum c(e) \rightarrow min$.

```py
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def addEdge(self, u, v, weight):
        self.edges.append((weight, u, v))

    def find(self, parent, i):
        """
        DSU + union by rank
        """

    def union(self, parent, rank, x, y):
        """
        DSU + union by rank
        """

    def kruskalMST(self):
        # Sort from min to max weight
        self.edges.sort()

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        mst = []

        for weight, u, v in self.edges:
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                mst.append((u, v, weight))
                self.union(parent, rank, x, y)

        return mst
```

# Lecture 11 - Weighted Graphs: Shortest paths

**Length** $l: E \rightarrow \real$ - weight of an edge.

Length of path $P: l(P)=\sum_{e \in P} l(e) \rightarrow min$

Tasks:

- point-to-point (s, r)
- single source (s)
- all pairs

In Bellman-Ford & Dijkstra when doing choice we work with paths $ g(v) =\sum l$ between start point $s$ and $v$.  
In other words $g(v)$ - shortest known length from the start node to $ v $.

Set $ g(s) = 0 $ for the start node $ s $, and $ g(v) = \infty $ for all other vertices.

## Bellman-Ford

Bellman-Ford is for Single Source & Point-to-Point.

$G$ is **conservative** if $\forall \text{cycle} Г: l(Г) \ge 0$

Bellman-Ford can work with negative lengths.  
But it's $O(VE)$.

```py
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))

    def bellman_ford(self, src):
        # Initialize distances from the source to all vertices as infinite
        dist = [float('inf')] * self.V
        dist[src] = 0  # Distance to the source is 0

        # Relax all edges (V-1) times
        for _ in range(self.V - 1):
            for u, v, weight in self.edges:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

        # Check for conservativeness: negative-weight cycles
        for u, v, weight in self.edges:
            if dist[u] + weight < dist[v]:
                print("Graph contains a negative-weight cycle")
                return None

        return dist
```

## Dijkstra

Dijkstra is for Single Source & Point-to-Point.

Dijkstra for Single Source is also named **Uniform Cost Search (UCS)**.

Only for **non-negative** lenghts.

### Pure Dijkstra

```py
import heapq

def dijkstra_with_states(graph, start):
    dist = [float('inf')] * len(graph)
    dist[start] = 0

    # White: not visited, Gray: in queue, Black: finalized
    state = ['W'] * len(graph)
    state[start] = 'G'

    # Min-heap
    heap = [(0, start)]  # (distance, node)

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if state[current_node] == 'B':
            continue

        state[current_node] = 'B'

        for neighbor, weight in graph[current_node]:
            if state[neighbor] != 'B':
                distance = current_distance + weight

                if distance < dist[neighbor]:
                    state[neighbor] = 'G'
                    dist[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))

    return dist
```

Time complexity: $O((V + E) \log(V))$.

### Bidirectional Dijkstra

Only for Point-to-Point.

```py
import heapq

def bidirectional_dijkstra_with_states(graph, reverse_graph, source, target):
    n = len(graph)
    inf = float('inf')

    dist_forward = [inf] * n
    dist_backward = [inf] * n
    dist_forward[source] = 0
    dist_backward[target] = 0

    # White: not visited, Gray: in queue, Black: finalized
    state_forward = ['W'] * n
    state_backward = ['W'] * n
    state_forward[source] = 'G'
    state_backward[target] = 'G'

    pq_forward = [(0, source)]  # (distance, node)
    pq_backward = [(0, target)]

    shortest_path = inf

    while pq_forward or pq_backward:
        if pq_forward:
            d_f, u = heapq.heappop(pq_forward)
            if state_forward[u] == 'B':
                continue

            state_forward[u] = 'B'

            for v, weight in graph[u]:
                if state_forward[v] != 'B':
                    new_distance = dist_forward[u] + weight
                    if new_distance < dist_forward[v]:
                        dist_forward[v] = new_distance
                        heapq.heappush(pq_forward, (new_distance, v))
                        state_forward[v] = 'G'

            if u in state_backward and state_backward[u] == 'B':
                shortest_path = min(shortest_path, dist_forward[u] + dist_backward[u])

        if pq_backward:
            d_b, u = heapq.heappop(pq_backward)
            if state_backward[u] == 'B':
                continue

            state_backward[u] = 'B'

            for v, weight in reverse_graph[u]:
                if state_backward[v] != 'B':
                    new_distance = dist_backward[u] + weight
                    if new_distance < dist_backward[v]:
                        dist_backward[v] = new_distance
                        heapq.heappush(pq_backward, (new_distance, v))
                        state_backward[v] = 'G'

            if u in state_forward and state_forward[u] == 'B':
                shortest_path = min(shortest_path, dist_forward[u] + dist_backward[u])

        # Stop if the shortest path has been found
        if shortest_path < inf:
            break

    return shortest_path if shortest_path < inf else -1
```

### Handling Negative Weights Using Potentials

Reweighting edges can be applied to Dijkstra to handle graphs with **negative weights** but no negative cycles.

By applying **Johnson's reweighting technique**:  
$G$ with possibly negative weights $ w(u, v) $ $\rightarrow$ $G'$ with non-negative weights $ w'(u, v) $:

$$ w'(u, v) = w(u, v) + \pi(u) - \pi(v) $$

where $ \pi(v) $ - **potential function** is computed using Bellman-Ford from an artificial source.

Since $ w'(u, v) \geq 0 $, we can now run Dijkstra on the modified graph $ G' $, obtaining correct shortest paths.

After computing shortest distances $ d'(s, v) $ in $ G' $, we obtain the original distances $ d(s, v) $ using:

$$
 d(s, v) = d'(s, v) - \pi(s) + \pi(v)
$$

This ensures that the shortest path distances in the reweighted graph match those in the original graph.

## A*

A* minimizes visited vertices by prioritizing nodes close to the goal, reducing unnecessary explorations.

It extends Dijkstra's algorithm by introducing a **heuristic** function $ h(v) $, which estimates the cost to reach the goal from vertex $ v $.

The main idea behind A* is to prioritize nodes using the function:

$$ f(v) = g(v) + h(v) $$

$ h(v) $ is an **admissible** heuristic (never overestimates the true cost).

With a well-chosen heuristic, A* significantly outperforms Dijkstra's algorithm in many practical applications.

### Choice of $h(v)$

Grid-based pathfinding:

- Euclidean distance - if diagonal moves allowed
- Manhattan distance - if only vertical/horizontal moves allowed

Road networks:

- straight-line distance or travel time based on speed limits

# Lecture 12 - Persistence

- **Full persistence**: allows both access and modification to all versions.

- **Partial persistence**: allows access to all versions, modifications only to the current version.

## Full Persistence: Stack

```cpp
template <typename T>
class PersistentStack {
private:
    struct Node {
        T value;
        std::shared_ptr<Node> next;

        Node(T val, std::shared_ptr<Node> next = nullptr)
            : value(val), next(next) {}
    };

    std::shared_ptr<Node> top;

public:
    PersistentStack() : top(nullptr) {}

    PersistentStack<T> push(T value) const {
        // Create a new node and point it to the current top
        return PersistentStack<T>(std::make_shared<Node>(value, top));
    }

    PersistentStack<T> pop() const {
        if (top == nullptr) {
            throw std::runtime_error("Stack is empty!");
        }
        return PersistentStack<T>(top->next);
    }
}
```

## Patrial Persistence

Pointer Machine = RAM Machine, but instead of random access to memory, it uses pointers to access any memory cell directly by its address.
