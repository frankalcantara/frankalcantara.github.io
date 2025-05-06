---
author: Frank
beforetoc: |-
    [Anterior](2024-09-24-16-The-Dynamic-Programming-Classic-Problems.md)
    [Próximo](2024-09-24-18-6.-Computational-Geometry.md)
categories:
    - Matemática
    - Linguagens Formais
    - Programação
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: false
image: assets/images/prog_dynamic..webp
keywords:
    - Dynamic Programming
lastmod: 2025-05-06T11:04:18.071Z
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Competitive Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: competitive-programming-techniques-insights
tags:
    - Matemática
    - Linguagens Formais
title: 5. Graphs and Graph Theory
toc: true
---

# 5. Graphs and Graph Theory


**This is a work in progress, we will get there sooner or later.**

**Range Minimum Queries (RMQ)**

Data structure to find the minimum in any range in $O(1)$ after $O(n \log n)$ preprocessing.

[text](https://contest.cs.cmu.edu/295/s20/tutorials/lca.mark)

- Algorithm: Sparse Table for RMQ

- **Depth-First Search (DFS) and Breadth-First Search (BFS)**: Basic graph traversal algorithms, often used to explore nodes or determine connectivity in graphs.

- **Minimum Spanning Tree**: Problems where the goal is to find a subset of the edges that connects all vertices in a weighted graph while minimizing the total edge weight (Kruskal’s and Prim’s algorithms).

- **Shortest Path Algorithms**: Algorithms such as Dijkstra’s, Bellman-Ford, and Floyd-Warshall are used to find the shortest path between nodes in a graph.

- **Maximum Flow**: Problems involving optimizing flow through a network, such as Ford-Fulkerson and Edmonds-Karp algorithms.

- **Strongly Connected Components**: Identifying maximal strongly connected subgraphs in directed graphs using algorithms like Kosaraju or Tarjan’s.


## 11.7. The Fenwick Tree

The Fenwick Tree, also know as Binary Indexed Tree (BIT), is an efficient data structure designed to handle dynamic cumulative frequency tables. It was introduced by Peter M. Fenwick in 1994 in his paper _"A new data structure for cumulative frequency tables."_

The Fenwick tree allows two main operations in $O(\log n)$ time:

1. Compute the sum of elements in a range (range query)
2. Update the value of an individual element (point update)

These characteristics make the Fenwick tree ideal for applications involving frequent updates and queries, such as competitive programming problems and real-time data analysis. Consider the following problem: given an array $A$ of size $n$, efficiently perform the following operations:

1. Update the value of an element at a specific position
2. Compute the sum of elements in a range $[l, r]$

A naive approach to solve this problem would be:

```cpp
void update(int i, int val) {
    A[i] = val;
}

int rangeSum(int l, int r) {
int sum = 0;
for (int i = l; i <= r; i++) {
sum += A[i];
}
return sum;
}
```

_An illustration showing a naive approach to range sum computation, where each element of the array is accessed individually, leading to $O(n)$ complexity._

This solution has $O(1)$ complexity for updates and $O(n)$ for sum queries. To improve query efficiency, we could use a prefix sum array:

```cpp
vector<int> prefixSum;

void buildPrefixSum() {
prefixSum.resize(A.size() + 1, 0);
for (int i = 0; i < A.size(); i++) {
prefixSum[i + 1] = prefixSum[i] + A[i];
}
}

int rangeSum(int l, int r) {
return prefixSum[r + 1] - prefixSum[l];
}
```

**[Image placeholder]**
_Visualize the prefix sum technique, where the prefix sums are precomputed and used to speed up range sum queries._

Now, sum queries have $O(1)$ complexity, but updates still require $O(n)$ to rebuild the prefix sum array.

The Binary Indexed Tree offers a balance between these two approaches, allowing both updates and queries in $O(\log n)$.

### 11.7.1 Fundamental Concept

The Binary Indexed Tree (BIT) is built on the idea that each index $i$ in the tree stores a cumulative sum of elements from the original array. **The range of elements summed at each index $i$ is determined by the position of the least significant set bit (LSB) in the binary representation of $i$**.

> Note: In this explanation and the following examples, we use 0-based indexing. This means the first element of the array is at index 0, which is a common convention in programming.

The LSB (_Least Significante bit_) can be found using a bitwise operation:

$$\text{LSB}(i) = i \& (-i)$$

This operation isolates the last set bit in the binary representation of $i$, which helps define the size of the segment for which the cumulative sum is stored. The segment starts at index $i - \text{LSB}(i) + 1$ and ends at $i$.

When you perform the bitwise $AND$ operation between $i$ and $-i$, what happens is:

- $i$ in its binary form contains some bits set to 1.
- $-i$ is the complement of $i$ plus 1, which means it inverts all the bits of $i$ up to the last bit set to 1, and this last bit set to 1 remains.

This operation effectively isolates the last bit set to 1 in $i$. In other words, all bits to the right of the last set bit are zeroed, while the least significant bit that was set remains. For example, let's take $i = 11 \ (1011_2)$:

- $i = 1011_2$
- $-i = 0101_2$

Now, applying $AND$ bit by bit:

$$1011_2 \& 0101_2 = 0001_2$$

Therefore, $\text{LSB}(11) = 1$. This means that index 11 in the Fenwick tree only covers the value stored at position 11. Now let's take $i = 12 \ (1100_2)$:

- $i = 1100_2$
- $-i = 0100_2$

Now, applying $AND$ bit by bit:

$$1100_2 \& 0100_2 = 0100_2$$

Therefore, $\text{LSB}(12) = 4$. This means that index 12 in the Fenwick tree represents the sum of elements from index 9 to index 12.

**Example**:

Let's consider an array $A = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3, 1]$. The corresponding Fenwick tree will store cumulative sums for segments determined by the $\text{LSB}(i)$:

| Index $i$ | Binary $i$ | LSB(i) | Cumulative Sum Represented         | Value Stored in Fenwick tree[i] |
| --------- | ---------- | ------ | ---------------------------------- | ------------------------------- |
| 0         | $0000_2$   | 1      | $A[0]$                             | 3                               |
| 1         | $0001_2$   | 1      | $A[1]$                             | 2                               |
| 2         | $0010_2$   | 2      | $A[0] + A[1] + A[2]$               | 4                               |
| 3         | $0011_2$   | 1      | $A[2]$                             | -1                              |
| 4         | $0100_2$   | 4      | $A[0] + A[1] + A[2] + A[3] + A[4]$ | 15                              |
| 5         | $0101_2$   | 1      | $A[5]$                             | 4                               |
| 6         | $0110_2$   | 2      | $A[4] + A[5] + A[6]$               | 6                               |
| 7         | $0111_2$   | 1      | $A[6]$                             | -3                              |
| 8         | $1000_2$   | 8      | $A[0] + \dots + A[7]$              | 19                              |
| 9         | $1001_2$   | 1      | $A[8]$                             | 7                               |
| 10        | $1010_2$   | 2      | $A[8] + A[9]$                      | 9                               |
| 11        | $1011_2$   | 1      | $A[10]$                            | 3                               |
| 12        | $1100_2$   | 4      | $A[8] + A[9] + A[10] + A[11]$      | 13                              |

The value stored in each position of the Fenwick tree is the incremental contribution that helps compose the cumulative sum. For example, at position 2, the value stored is $4$, which is the sum of $A[0] + A[1] + A[2]$. At position 4, the value stored is $15$, which is the sum of $A[0] + A[1] + A[2] + A[3] + A[4]$.

![]({{ site.baseurl }}/assets/images/bit1.webp){: class="lazyimg"}
_Gráfico 1.1 - Example Fenwick tree diagram._{: class="legend"}

### 11.7.2. Querying the Fenwick tree

When querying the sum of elements from the start of the array to index $i$, the Fenwick tree allows us to sum over non-overlapping segments by traversing the tree upwards:

Here's the pseudocode for the sum operation:

```python
def sum(i):
    total = 0
    while i >= 0:
        total += BIT[i]
        i -= LSB(i)
    return total
```

For example, to compute the sum of elements from index $0$ to $5$, we perform the following steps:

- Start at index 5. The LSB of 5 is 1, so add $A[5]$.
- Move to index 4, since $5 - \text{LSB}(5) = 4$. The LSB of 4 is 4, so add $A[0] + A[1] + A[2] + A[3] + A[4]$.

Thus, the sum of elements from index $0$ to $5$ is:

$$ \text{sum}(0, 5) = \text{BIT}[5] + \text{BIT}[4] = A[5] + (A[0] + A[1] + A[2] + A[3] + A[4]) $$

### 11.7.3. Updating the Fenwick tree

When updating the value of an element in the original array, the Fenwick tree allows us to update all the relevant cumulative sums efficiently. Here's the pseudocode for the update operation:

```python
def update(i, delta):
    while i < len(BIT):
        BIT[i] += delta
        i += LSB(i)
```

For example, if we update $A[4]$, the Fenwick tree must update the sums stored at indices that cover $A[4]$'s range.

- Start at index 4. Add the change to $\text{BIT}[4]$.
- Move to index 8 and update $\text{BIT}[8]$.

In each case, the number of operations required is proportional to the number of set bits in the index, which guarantees that both update and query operations run in $O(\log n)$.

### 11.7.4. Basic Operations

#### Update

To update an element at position $i$, we traverse the tree as follows:

```cpp
void update(int i, int delta) {
    for (; i < n; i += i & (-i)) {
        BIT[i] += delta;
    }
}
```

**[Image placeholder]**
_Illustrate the update process, showing how the Fenwick tree array is updated step by step using the least significant bit._

#### 4.2 Prefix Sum Query

To compute the sum of elements from 0 to $i$:

```cpp
int query(int i) {
    int sum = 0;
    for (; i >= 0; i -= i & (-i)) {
        sum += BIT[i];
    }
    return sum;
}
```

**[Image placeholder]**
_Visualize the prefix sum query operation, showing how the Fenwick tree is traversed from $i$ down to 0 using the least significant bit._

#### 4.3 Range Query

To compute the sum of elements in the range $[l, r]$:

```cpp
int rangeQuery(int l, int r) {
    return query(r) - query(l - 1);
}
```

### 5. Fenwick tree Construction

The Fenwick tree can be constructed in $O(n)$ time using the following technique:

```cpp
vector<int> constructBIT(const vector<int>& arr) {
    int n = arr.size();
    vector<int> BIT(n, 0);
    for (int i = 0; i < n; i++) {
        int idx = i;
        BIT[idx] += arr[i];
        int parent = idx + (idx & (-idx));
        if (parent < n) {
            BIT[parent] += BIT[idx];
        }
    }
    return BIT;
}
```

**[Image placeholder]**
_An illustration that explains how the Fenwick tree is constructed from an array, showing the incremental process of building the tree._

### Complexity Analysis

- Construction: $O(n)$
- Update: $O(\log n)$
- Query: $O(\log n)$
- Space: $O(n)$

### Variations and Extensions

#### Range Update and Point Query

It is possible to modify the Fenwick tree to support range updates and point queries:

```cpp
void rangeUpdate(int l, int r, int val) {
    update(l, val);
    update(r + 1, -val);
}

int pointQuery(int i) {
    return query(i);
}
```

#### Range Update and Range Query

To support both range updates and range queries, we need two Fenwick trees:

```cpp
void rangeUpdate(int l, int r, int val) {
    update(BIT1, l, val);
    update(BIT1, r + 1, -val);
    update(BIT2, l, val * (l - 1));
    update(BIT2, r + 1, -val * r);
}

int prefixSum(int i) {
    return query(BIT1, i) * i - query(BIT2, i);
}

int rangeQuery(int l, int r) {
    return prefixSum(r) - prefixSum(l - 1);
}
```

#### 2D Fenwick tree

The Fenwick tree can be extended to two dimensions:

```cpp
void update2D(int x, int y, int delta) {
    for (int i = x; i < n; i += i & (-i))
        for (int j = y; j < m; j += j & (-j))
            BIT[i][j] += delta;
}

int query2D(int x, int y) {
    int sum = 0;
    for (int i = x; i >= 0; i -= i & (-i))
        for (int j = y; j >= 0; j -= j & (-j))
            sum += BIT[i][j];
    return sum;
}
```

**[Image placeholder]**
_A diagram illustrating how a 2D Fenwick tree operates, showing how updates and queries are performed in two dimensions._

### Applications

1. Efficient computation of prefix sums in mutable arrays
2. Counting inversions in an array
3. Solving the "k-th smallest element" problem
4. Implementation of arithmetic coding algorithm

### Comparison with Other Structures

| Structure    | Update      | Query       | Space  |
| ------------ | ----------- | ----------- | ------ |
| Array        | $O(1)$      | $O(n)$      | $O(n)$ |
| Prefix Sum   | $O(n)$      | $O(1)$      | $O(n)$ |
| Segment Tree | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Fenwick tree | $O(\log n)$ | $O(\log n)$ | $O(n)$ |

The Fenwick tree offers a good balance between update and query efficiency, with a simpler implementation than a Segment Tree.

### Problem Example: "Humidity Levels in a Greenhouse" (Problem 1)

The same problem we saw earlier in the section: **Algorithm: Difference Array - Efficient Range Updates**. Below is the implementation in C++20:

I chose to write this code using as much modern C++ as possible. This means you will face two challenges. The first is understanding the Fenwick tree algorithm, and the second is understanding the C++ syntax. To help make this easier, I will explain the code block by block, highlighting each C++ feature and why I chose to write it this way.

**Code 4**:

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <syncstream>

using namespace std;
namespace fs = filesystem;

namespace config {
    enum class InputMethod { Hardcoded, Stdin, File };

    // Altere esta linha para mudar o método de entrada
    inline constexpr InputMethod input_method = InputMethod::Hardcoded;
}

using vi = vector<int>;
using vvi = vector<vector<int>>;
using vll = vector<long long>;

class BIT {
    vi tree;
    int n;

public:
    Fenwick tree(int size) : tree(size + 1), n(size) {}

    void update(int i, int delta) {
        for (++i; i <= n; i += i & -i) tree[i] += delta;
    }

    long long query(int i) const {
        long long sum = 0;
        for (++i; i > 0; i -= i & -i) sum += tree[i];
        return sum;
    }
};

vll adjustHumidity(vi& humidity, const vvi& adjustments) {
    int n = humidity.size();
    BIT bit(n);
    vll result;
    result.reserve(adjustments.size());

    auto updateBit = [&](int i, int old_val, int new_val) {
        if (!(old_val & 1)) bit.update(i, -old_val);
        if (!(new_val & 1)) bit.update(i, new_val);
        };

    for (int i = 0; i < n; ++i) {
        if (!(humidity[i] & 1)) bit.update(i, humidity[i]);
    }

    for (const auto& adj : adjustments) {
        int i = adj[1], old_val = humidity[i], new_val = old_val + adj[0];
        updateBit(i, old_val, new_val);
        humidity[i] = new_val;
        result.push_back(bit.query(n - 1));
    }

    return result;
}

void printResult(osyncstream& out, const vi& humidity, const vvi& adjustments, const vll& result) {
    out << "**Input**: humidity = [" << humidity[0];
    for (int i = 1; i < humidity.size(); ++i) out << ", " << humidity[i];
    out << "], adjustments = [";
    for (const auto& adj : adjustments)
        out << "[" << adj[0] << "," << adj[1] << "]" << (&adj != &adjustments.back() ? ", " : "");
    out << "]\n**Output**: ";
    for (auto res : result) out << res << " ";
    out << "\n\n";
}

pair<vi, vvi> readInput(istream& in) {
    vi humidity;
    vvi adjustments;
    int n, m;
    in >> n;
    humidity.resize(n);
    for (int& h : humidity) in >> h;
    in >> m;
    adjustments.resize(m, vi(2));
    for (auto& adj : adjustments) in >> adj[0] >> adj[1];
    return { humidity, adjustments };
}

void processInput(istream& in, osyncstream& out) {
    int t;
    in >> t;
    for (int i = 1; i <= t; ++i) {
        out << "Example " << i << ":\n";
        auto [humidity, adjustments] = readInput(in);
        auto result = adjustHumidity(humidity, adjustments);
        printResult(out, humidity, adjustments, result);
    }
}

int main() {
    osyncstream syncout(cout);
{% raw %}
    if constexpr (config::input_method == config::InputMethod::Hardcoded) {
        vector<pair<vi, vvi>> tests = {{{45, 52, 33, 64}, {{5,0}, {-20,1}, {-14,0}, {18,3}}},{{40}, {{12,0}}},{{30, 41, 55, 68, 72}, {{10,0}, {-15,2}, {22,1}, {-8,4}, {5,3}}}};
{% endraw %}
        for (int i = 0; i < tests.size(); ++i) {
            syncout << "Example " << i + 1 << ":\n";
            auto& [humidity, adjustments] = tests[i];
            auto result = adjustHumidity(humidity, adjustments);
            printResult(syncout, humidity, adjustments, result);
        }
    }
    else if constexpr (config::input_method == config::InputMethod::Stdin) {
        processInput(cin, syncout);
    }
    else if constexpr (config::input_method == config::InputMethod::File) {
        fs::path inputPath = "input.txt";
        if (fs::exists(inputPath)) {
            ifstream inputFile(inputPath);
            processInput(inputFile, syncout);
        }
        else {
            syncout << "Input file not found: " << inputPath << endl;
        }
    }
    else {
        syncout << "Invalid input method defined" << endl;
    }

    return 0;
}
```

The first thing you should notice is that I chose to include all three possible input methods in the same code. Obviously, in a competition, you wouldn't do that. You would include only the method that interests you. Additionally, I opted to use modern C++20 capabilities instead of using the old preprocessor directives (`#defines`). However, before diving into the analysis of Code 4, let's look at an example of what the `main` function would look like if we were using preprocessor directives.

```cpp
#include ...

// Define input methods
#define INPUT_HARDCODED 1
#define INPUT_STDIN 2
#define INPUT_FILE 3

// Select input method here
#define INPUT_METHOD INPUT_STDIN

// lot of code goes here

int main() {
    // Creates a synchronized output stream (osyncstream) to ensure thread-safe output to cout.
    osyncstream syncout(cout);

    // Check if the input method is defined as INPUT_HARDCODED using preprocessor directives.
#if INPUT_METHOD == INPUT_HARDCODED
    // Define a vector of pairs where each pair contains:
    // 1. A vector of humidity levels.
    // 2. A 2D vector representing adjustments (value, index) to be applied to the humidity levels.
{% raw %}
    vector<pair<vi, vvi>> tests = {
        {{45, 52, 33, 64}, {{5,0}, {-20,1}, {-14,0}, {18,3}}},
        {{40}, {{12,0}}},
        {{30, 41, 55, 68, 72}, {{10,0}, {-15,2}, {22,1}, {-8,4}, {5,3}}}
    };
{% endraw %}
    // Iterate over each hardcoded test case.
    for (int i = 0; i < tests.size(); ++i) {
        // Print the example number using synchronized output to avoid race conditions in a multithreaded context.
        syncout << "Example " << i + 1 << ":\n";

        // Extract the humidity vector and adjustments vector using structured bindings (C++17 feature).
        auto& [humidity, adjustments] = tests[i];

        // Call the adjustHumidity function to apply the adjustments and get the results.
        auto result = adjustHumidity(humidity, adjustments);

        // Print the humidity, adjustments, and the results using the printResult function.
        printResult(syncout, humidity, adjustments, result);
    }

    // If the input method is INPUT_STDIN, process input from standard input.
#elif INPUT_METHOD == INPUT_STDIN
    // Call processInput to read input from standard input and produce output.
    processInput(cin, syncout);

    // If the input method is INPUT_FILE, read input from a file.
#elif INPUT_METHOD == INPUT_FILE
    // Define the file path where the input data is expected.
    fs::path inputPath = "input.txt";

    // Check if the file exists at the specified path.
    if (fs::exists(inputPath)) {
        // If the file exists, open it as an input file stream.
        ifstream inputFile(inputPath);

        // Call processInput to read data from the input file and produce output.
        processInput(inputFile, syncout);
    } else {
        // If the file does not exist, print an error message indicating that the input file was not found.
        syncout << "Input file not found: " << inputPath << endl;
    }

    // If none of the above input methods are defined, print an error message for an invalid input method.
#else
    syncout << "Invalid INPUT_METHOD defined" << endl;
#endif

    // Return 0 to indicate successful program termination.
    return 0;
}
```

The code fragment uses **preprocessor directives** to switch between different input methods for reading data, based on a pre-defined configuration. This is done using `#define` statements at the top of the code and `#if`, `#elif`, and `#else` directives in the `main` function.

**Input Method Definitions**:

```cpp
#define INPUT_HARDCODED 1
#define INPUT_STDIN 2
#define INPUT_FILE 3
```

These `#define` statements assign integer values to three possible input methods:

- `INPUT_HARDCODED`: The input data is hardcoded directly into the program.
- `INPUT_STDIN`: The input data is read from standard input (`stdin`), such as from the console.
- `INPUT_FILE`: The input data is read from a file, typically stored on disk.

**Input Method Selection**:

```cpp
#define INPUT_METHOD INPUT_STDIN
```

This line selects the input method by defining `INPUT_METHOD`. In this case, it is set to `INPUT_STDIN`, meaning that the program will expect to read input from the console. Changing this to `INPUT_HARDCODED` or `INPUT_FILE` would switch the input source.

**Conditional Compilation (`#if`, `#elif`, `#else`)**:

The conditional compilation directives (`#if`, `#elif`, `#else`) are used to include or exclude specific blocks of code based on the value of `INPUT_METHOD`.

```cpp
#if INPUT_METHOD == INPUT_HARDCODED
    // Code for hardcoded input goes here
#elif INPUT_METHOD == INPUT_STDIN
    // Code for reading from standard input goes here
#elif INPUT_METHOD == INPUT_FILE
    // Code for reading from a file goes here
#else
    // Code for handling invalid input method goes here
#endif
```

- **`#if INPUT_METHOD == INPUT_HARDCODED`**: If the input method is hardcoded, a predefined set of test cases (humidity levels and adjustments) will be used.
- **`#elif INPUT_METHOD == INPUT_STDIN`**: If the input method is set to standard input, the program will read from the console.
- **`#elif INPUT_METHOD == INPUT_FILE`**: If the input method is set to file input, the program will attempt to read from a file (`input.txt`).
- **`#else`**: If an invalid `INPUT_METHOD` is defined, an error message is printed.

These preprocessor directives enable the program to easily switch between input methods without having to manually modify the logic inside `main`, providing flexibility depending on how the input is expected during execution. But, since we are using C++20, this might not be the best solution. It may be the fastest for competitions, but there is a fundamental reason why I'm making things a bit more complex here. Beyond just learning how to write code for competitions, we are also learning C++20. Let's start by:

The code starts by importing the `std` namespace globally with **`using namespace std;`, which allows using standard C++ objects (like `cout`, `vector`, etc.) without having to prefix them with `std::`**.
s

```cpp
using namespace std;  // Use the standard namespace to avoid typing "std::" before standard types.
```

The line **`namespace fs = filesystem;`** creates an alias for the `filesystem` namespace, allowing the code to reference `filesystem` functions more concisely, using `fs::` instead of `std::filesystem::`.

```cpp
namespace fs = filesystem;  // Alias for the filesystem namespace.
```

Inside the **`config` namespace**, there is an **enum class** `InputMethod` that defines three possible input methods: `Hardcoded`, `Stdin`, and `File`. This helps manage how input will be provided to the program.

```cpp
namespace config {
    enum class InputMethod { Hardcoded, Stdin, File };  // Enum to define input methods
```

> The **`namespace config`** is used to encapsulate related constants and configuration settings into a specific scope. In this case, it organizes the input methods and settings used in the program. By placing these within a namespace, we avoid cluttering the global namespace, ensuring that these settings are logically grouped together. This encapsulation makes it easier to maintain the code, preventing potential naming conflicts and allowing future expansion of the configuration without affecting other parts of the program.
>
> The **`namespace config`** does not come from the standard C++ library; it is created specifically within this code to group configurations like the `InputMethod`. The use of namespaces in C++ allows developers to organize code and avoid naming conflicts but is independent of the C++ Standard Library or language itself.
>
> The **`enum class InputMethod`** provides a strongly typed, scoped enumeration. Unlike traditional enums, an `enum class` does not implicitly convert its values to integers, which helps prevent accidental misuse of values. The scoped nature of `enum class` also means that its values are contained within the enumeration itself, avoiding naming conflicts with other parts of the program. For instance, instead of directly using `Hardcoded`, you use `InputMethod::Hardcoded`, making the code more readable and avoiding ambiguity.
>
> Here's an example of using an **enum class** in a small program. This example demonstrates how to select an input method based on the defined `InputMethod`:
>
> ```cpp
> #include <iostream>
>
> enum class InputMethod { Hardcoded, Stdin, File };
>
> void selectInputMethod(InputMethod method) {
>     switch (method) {
>         case InputMethod::Hardcoded:
>             std::cout << "Using hardcoded input.\n";
>             break;
>         case InputMethod::Stdin:
>             std::cout << "Reading input from stdin.\n";
>             break;
>         case InputMethod::File:
>             std::cout << "Reading input from a file.\n";
>             break;
>     }
> }
>
> int main() {
>     InputMethod method = InputMethod::File;
>     selectInputMethod(method);  // **Output**: Reading input from a file.
>     return 0;
> }
> ```
>
> In this example, the `enum class InputMethod` allows for a clear, type-safe way to represent the input method, making the code easier to manage and less error-prone.

The **`inline constexpr`** constant `input_method` specifies which input method will be used by default. In this case, it is set to `InputMethod::Hardcoded`, meaning the input will be predefined inside the code. The `inline constexpr` allows the value to be defined at compile time, making it a more efficient configuration option.

```cpp
    inline constexpr InputMethod input_method = InputMethod::Hardcoded;  // Default input method is hardcoded.
}
```

> The **`inline`** keyword in C++ specifies that a function, variable, or constant is defined inline, meaning the compiler should attempt to replace function calls with the actual code of the function. This can improve performance by avoiding the overhead of a function call. However, the main use of `inline` in modern C++ is to avoid the "multiple definition" problem when defining variables or functions in header files that are included in multiple translation units.
>
> ```cpp
> inline int square(int x) {
>     return x * x;  // This function is defined inline, so calls to square(3) may be replaced with 3 * 3 directly.
> }
> ```
>
> When `inline` is used with **variables or constants**, it allows those variables or constants to be defined in a header file without violating the One Definition Rule (ODR). Each translation unit that includes the header will have its own copy of the inline variable, but the linker will ensure that only one copy is used in the final binary.
>
> ```cpp
> inline constexpr int max_value = 100;  // This constant can be included in multiple translation units without causing redefinition errors.
> ```
>
> The **`constexpr`** keyword specifies that a function or variable can be evaluated at compile-time. It guarantees that, if possible, the function will be computed by the compiler, not at runtime. This is especially useful in optimization, as it allows constants to be determined and used during the compilation process rather than execution.
>
> **`constexpr` with Variables**:
> When you use `constexpr` with variables, the compiler knows that the variable's value is constant and should be computed at compile time.
>
> ```cpp
> constexpr int max_items = 42;  // The value of max_items is known at compile-time and cannot change.
> ```
>
> You can use `constexpr` variables to define array sizes or template parameters because their values are known during compilation.
>
> ```cpp
> constexpr int size = 10;
> int array[size];  // Valid, because size is a constant expression.
> ```
>
> **`constexpr` with Functions**:
> A **`constexpr` function** is a function whose return value can be computed at compile time if the inputs are constant expressions. The function must have a single return statement and all operations within it must be valid at compile time.
>
> ```cpp
> constexpr int factorial(int n) {
>     return n <= 1 ? 1 : n * factorial(n - 1);  // Recursive function that computes the factorial at compile time.
> }
> ```
>
> If `factorial(5)` is called with a constant value, the compiler will compute the result at compile time and replace the function call with `120` in the generated binary.
> ?
> **Combining `inline` and `constexpr`**:
> A function can be both **`inline`** and **`constexpr`**, which means the function can be evaluated at compile time and its calls may be inlined if appropriate.
>
> ```cpp
> inline constexpr int power(int base, int exp) {
>     return (exp == 0) ? 1 : base * power(base, exp - 1);
> }
> ```
>
> In this case, the `power` function will be inlined when called at runtime and computed at compile time if the arguments are constant. For example, `power(2, 3)` would be replaced by `8` at compile time.
>
> **Practical Use of `constexpr`**:
> `constexpr` can be used in a wide variety of contexts, such as constructing constant data, optimizing algorithms, and defining efficient compile-time logic. Here are a few examples:
>
> 1. **Compile-time array size**:
>
> ```cpp
> constexpr int size = 5;
> int array[size];  // The size is computed at compile time.
> ```
>
> 2. **Compile-time strings**:
>
> ```cpp
> constexpr const char* greet() { return "Hello, World!"; }
> constexpr const char* message = greet();  // The message is computed at compile time.
> ```
>
> 3. **Compile-time mathematical operations**:
>
> ```cpp
> constexpr int area(int length, int width) {
>     return length * width;
> }
> constexpr int room_area = area(10, 12);  // Computed at compile time.
> ```
>
> **Using `constexpr` in Competitive Programming**:
> In competitive programming, **`constexpr`** can be both an advantage and a disadvantage, depending on how it is used.
>
> - **Advantage**: `constexpr` can optimize code by computing results at compile time rather than runtime, which can save valuable processing time. For example, if you know certain values or calculations are constant throughout the competition, you can use `constexpr` to precompute them, thereby avoiding recalculations during execution.
>
> - **Disadvantage**: However, in many competitive programming problems, the input is dynamic and provided at runtime, meaning that `constexpr` cannot be used for computations that depend on this input. Since the focus in competitive programming is on runtime efficiency, the use of `constexpr` is limited to cases where you can precompute values before the competition or during compilation.
>
> Overall, `constexpr` is valuable when solving problems with static data or fixed input sizes, but in typical ICPC-style competitions, its usage may be less frequent because most problems require dynamic input processing.
>
> In summary, **`inline`** helps with reducing overhead by allowing the compiler to replace function calls with the actual function code, and it prevents multiple definitions of variables in multiple translation units. **`constexpr`** enables computations to be performed at compile time, which can significantly optimize performance by avoiding runtime calculations, although its applicability in competitive programming may be limited.

AINDA TEM MUITO QUE EXPLICAR AQUI.

### Inventory Restocking

You manage a warehouse where products are stored and moved frequently. The warehouse tracks its inventory by recording the stock count at different times during the day in an array $inventory$. Occasionally, inventory managers report the amount by which a product's stock needs to be adjusted, represented by an integer array $adjustments$, where each adjustment is a pair $[adjustment, index]$. Your task is to apply these adjustments and after each, calculate the total count of products with even stock numbers.

**Input Format:**

- The first line contains an integer $n$, representing the size of the $inventory$ array.
- The second line contains $n$ integers representing the initial values in the $inventory$ array.
- The third line contains an integer $q$, the number of stock adjustments.
- The following $q$ lines each contain a pair $adjustment$ and $index$, where $adjustment$ is the amount to be added or subtracted, and $index$ is the position in the $inventory$ array to adjust.

**Constraints:**

- $1 \leq n, q \leq 10^5$
- $-10^4 \leq inventory[i], adjustment \leq 10^4$

**Example **Input**:**

```text
6
10 3 5 6 8 2
4
[3, 1]
[-4, 0]
[2, 3]
[-3, 4]
```

**Example **Output**:**

```text
26
16
20
16
```

**Explanation:**

Initially, the array is $[10, 3, 5, 6, 8, 2]$, and the sum of even values is $10 + 6 + 8 + 2 = 26$.

- After adding $3$ to $inventory[1]$, the array becomes $[10, 6, 5, 6, 8, 2]$, and the sum of even values is $10 + 6 + 6 + 8 + 2 = 32$.
- After subtracting $4$ from $inventory[0]$, the array becomes $[6, 6, 5, 6, 8, 2]$, and the sum of even values is $6 + 6 + 6 + 8 + 2 = 28$.

**Input Method:**

The input will be provided via **hardcoded values** inside the code for testing purposes.

#### Naïve Solution

- Initially, the even numbers in $inventory$ are $10$, $6$, $8$, $2$. The sum of these values is $26$.
- After the first adjustment $[3, 1]$, the inventory becomes $[10, 6, 5, 6, 8, 2]$. The even numbers are now $10$, $6$, $8$, $2$. The sum remains $26$.
- After the second adjustment $[-4, 0]$, the inventory becomes $[6, 6, 5, 6, 8, 2]$. The even numbers are $6$, $6$, $8$, $2$. The sum is $16$.
- After the third adjustment $[2, 3]$, the inventory becomes $[6, 6, 5, 8, 8, 2]$. The even numbers are $6$, $6$, $8$, $8$, $2$. The sum is $20$.
- After the fourth adjustment $[-3, 4]$, the inventory becomes $[6, 6, 5, 8, 5, 2]$. The even numbers are $6$, $6$, $8$, $2$. The sum is $16$.

**Pseudo Code Solution using python**:

Here is a Python solution that solves the problem as simply and directly as requested:

```python
# Read the size of the inventory array
n = int(input())

# Read the inventory array
inventory = list(map(int, input().split()))

# Read the number of adjustments
q = int(input())

# Initialize the sum of even numbers
even_sum = sum([x for x in inventory if x % 2 == 0])

# Process the adjustments
for _ in range(q):
    adjustment, index = map(int, input().strip('[]').split(','))

    # Check if the current value at the index is even before the adjustment
    if inventory[index] % 2 == 0:
        even_sum -= inventory[index]  # Remove from the sum if it was even

    # Apply the adjustment
    inventory[index] += adjustment

    # Check if the new value at the index is even after the adjustment
    if inventory[index] % 2 == 0:
        even_sum += inventory[index]  # Add to the sum if it is now even

    # Print the updated sum of even numbers
    print(even_sum)
```

In this pseudocode, we observe the following steps:

1. **Input Reading**: First, we read the value of $n$ (inventory size) and the integer array $inventory$. Then, we read the number of adjustments $q$ and each adjustment.
2. **Initial Calculation**: We calculate the initial sum of even numbers in the $inventory$ array.
3. **Processing Adjustments**: For each adjustment, we check if the value at the affected index is even before the adjustment. If it is, we remove that value from the sum of even numbers. We then apply the adjustment, and if the new value at the index is even, we add it to the sum of even numbers.
4. **Output**: After each adjustment, we print the updated sum of even numbers.

For this solution the time complexity is $O(n + q)$, where $n$ is the size of the $inventory$ array and $q$ is the number of adjustments. The code processes each adjustment in constant time since the sum is maintained incrementally.

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Hardcoded input values
    int n = 6, q = 4;
    vector<int> inventory = { 10, 3, 5, 6, 8, 2 };
    vector<pair<int, int>> adjustments = { {3, 1}, {-4, 0}, {2, 3}, {-3, 4} };

    // Initial sum of even numbers
    int even_sum = 0;
    for (int i = 0; i < n; ++i)
        if (!(inventory[i] & 1)) even_sum += inventory[i]; // Check if even

    // Process adjustments
    for (int i = 0; i < q; ++i) {
        int adj = adjustments[i].first, idx = adjustments[i].second;
        if (!(inventory[idx] & 1)) even_sum -= inventory[idx]; // Subtract if even
        inventory[idx] += adj;  // Apply adjustment
        if (!(inventory[idx] & 1)) even_sum += inventory[idx]; // Add if now even
        cout << even_sum << '\n';
    }
}
```

> In C++, the expression `inventory[i] & 1` is a bitwise operation that checks whether the value at index $i$ in the `inventory` array is **even or odd**.

**Detailed Explanation**:

- **`&` Operator**: This is the **bitwise AND** operator. It performs an AND operation on the binary representation of two numbers, comparing corresponding bits.
- **`1` in Binary**: The number $1$ in binary is represented as $0000...0001$ (depending on the size of the integer). Since only the least significant bit is set to $1$, this operation focuses specifically on the least significant bit of `inventory[i]`.
- **Bitwise AND (`&`)**: The bitwise AND operator returns $1$ if both corresponding bits of the operands are $1$. For other cases, it returns $0$.

In the naïve code this operation checks whether the **least significant bit** (LSB) of the value at `inventory[i]` is $1$ or $0$, which directly indicates if the number is odd or even:

- If the result of `inventory[i] & 1` is $1$, the number is **odd**.
- If the result of `inventory[i] & 1` is $0$, the number is **even**.

This bitwise approach is **faster** than using the modulo operation (`inventory[i] % 2 == 0`) for determining even/odd status, as it avoids division and is optimized at the hardware level. Consider the following values for `inventory[i]`:

- For `inventory[i] = 6` (binary: $110$), the operation:
  - $6 \& 1 = 0$
  - Since the result is $0$, 6 is **even**.
- For `inventory[i] = 5` (binary: $101$), the operation:

  - $5 \& 1 = 1$
  - Since the result is $1$, 5 is **odd**.

> ### Bitwise Operations in C++
>
> Bitwise operations in C++ manipulate individual bits of integers. These operations are low-level but powerful, allowing programmers to perform tasks like toggling, setting, or clearing specific bits. They are commonly used in scenarios where performance is critical, such as embedded systems, cryptography, and competitive programming.
>
> C++ provides several bitwise operators that work directly on the binary representation of numbers. These operators include:
>
> - **AND (`&`)**
> - **OR (`|`)**
> - **XOR (`^`)**
> - **NOT (`~`)**
> - **Left Shift (`<<`)**
> - **Right Shift (`>>`)**
>
> **Bitwise AND (`&`)**
>
> The **bitwise AND** operator compares each bit of its operands and returns $1$ if both bits are $1$. Otherwise, it returns $0$.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a & b;  // result = 2 (Binary: 010)
> ```
>
> In this code we have: the binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the AND operation:
>
> - $1 \& 0 = 0$
> - $1 \& 1 = 1$
> - $0 \& 1 = 0$
>
> Therefore, the result is $010$ in binary, which is $2$ in decimal.
>
> **Bitwise OR (`|`)**
>
> The **bitwise OR** operator compares each bit of its operands and returns $1$ if either of the bits is $1$.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a | b;  // result = 7 (Binary: 111)
> ```
>
> The binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the OR operation:
>
> - $1 \| 0 = 1$
> - $1 \| 1 = 1$
> - $0 \| 1 = 1$
>
> Therefore, the result is $111$ in binary, which is $7$ in decimal.
>
> **Bitwise XOR (`^`)**
>
> The **bitwise XOR** (exclusive OR) operator compares each bit of its operands and returns $1$ if the bits are different, and $0$ if they are the same.
>
> ```cpp
> int a = 6;    // Binary: 110
> int b = 3;    // Binary: 011
> int result = a ^ b;  // result = 5 (Binary: 101)
> ```
>
> The binary representation of 6 is $110$ and the binary representation of 3 is $011$. When performing the XOR operation:
>
> - $1 \oplus 0 = 1$
> - $1 \oplus 1 = 0$
> - $0 \oplus 1 = 1$
> - Therefore, the result is $101$ in binary, which is $5$ in decimal.
>
> **Bitwise NOT (`~`)**
>
> The **bitwise NOT** operator inverts all the bits of its operand. It converts $1$s to $0$s and $0$s to $1$s.
>
> ```cpp
> int a = 6;    // Binary: 00000000 00000000 00000000 00000110 (32-bit system)
> int result = ~a;  // result = -7 (Binary: 11111111 11111111 11111111 11111001)
> ```
>
> The binary representation of 6 is $0000...0110$ (with 32 bits). The NOT operation flips each bit: $~110$ becomes $111...1001$.The result is the two's complement representation of $-7$.
>
> **Left Shift (`<<`)**
>
> The **left shift** operator shifts the bits of its first operand to the left by the number of positions specified by the second operand. This effectively multiplies the number by powers of 2.
>
> ```cpp
> int a = 3;    // Binary: 00000000 00000000 00000000 00000011
> int result = a << 1;  // result = 6 (Binary: 00000000 00000000 00000000 00000110)
> ```
>
> The binary representation of 3 is $011$. Shifting it left by $1$ position results in $110$, which is $6$ in decimal. Shifting by $n$ positions is equivalent to multiplying by $2^n$.
>
> **Right Shift (`>>`)**
>
> The **right shift** operator shifts the bits of its first operand to the right by the number of positions specified by the second operand. This effectively divides the number by powers of 2 (for positive integers).
>
> ```cpp
> int a = 6;    // Binary: 00000000 00000000 00000000 00000110
> int result = a >> 1;  // result = 3 (Binary: 00000000 00000000 00000000 00000011)
> ```
>
> The binary representation of 6 is $110$. Shifting it right by $1$ position results in $011$, which is $3$ in decimal.Shifting by $n$ positions is equivalent to dividing by $2^n$ (for non-negative integers).
>
> ### Summary Table of Bitwise Operations
>
> | Operation   | Symbol | Effect                                                                    |
> | ----------- | ------ | ------------------------------------------------------------------------- | -------------------------------------------------- |
> | Bitwise AND | `&`    | Compares bits, returns $1$ if both are $1$.                               |
> | Bitwise OR  | `      | `                                                                         | Compares bits, returns $1$ if at least one is $1$. |
> | Bitwise XOR | `^`    | Compares bits, returns $1$ if bits are different.                         |
> | Bitwise NOT | `~`    | Inverts each bit (turns $0$s into $1$s and $1$s into $0$s).               |
> | Left Shift  | `<<`   | Shifts bits to the left, multiplying by powers of 2.                      |
> | Right Shift | `>>`   | Shifts bits to the right, dividing by powers of 2 (for positive numbers). |
>
> **Applications of Bitwise Operations**:
>
> 1. **Efficiency**: Bitwise operations are faster than arithmetic operations, making them useful in performance-critical code.
> 2. **Bit Manipulation**: They are commonly used for tasks such as toggling, setting, and clearing bits in low-level programming, such as working with hardware or network protocols.
> 3. **Masking and Flagging**: Bitwise operators are often used to manipulate flags in bitmasks, where individual bits represent different conditions or options.

### 4 - Sales Target Analysis

You are tasked with analyzing sales data to determine how many subarrays of daily sales sum to a multiple of a target value $T$ . The sales data is recorded in an array sales , and you need to calculate how many contiguous subarrays of sales have a sum divisible by $T$ .

**Input Format**:

The first line contains two integers $n$ (the size of the sales array) and $T$ (the target value).
The second line contains $n$ integers, representing the daily sales data.
Constraints:

$$
1 \leq n \leq 10^5 \\
1 \leq T \leq 10^4 \\
-10^4 \leq \text{sales}[i] \leq 10^4
$$

Output Format:

Output a single integer representing the number of subarrays whose sum is divisible by $T$ .

Example **Input**: 6 5 4 5 0 -2 -3 1

Example **Output**: 7

Explanation:

There are $7$ subarrays whose sum is divisible by $T=5$ :

```text
[4,5,0,−2,−3,1]
[5]
[5,0]
[5,0,−2,−3]
[0]
[0,−2,−3]
[−2,−3]
```

**Input Method**:

The input is provided via command-line arguments.

#### Naïve Code

The algorithm works as follows:

1. We define a function `count_divisible_subarrays` that takes the `sales` array and target value `T` as inputs.

2. We use two nested loops to generate all possible subarrays:

   - The outer loop (`start`) determines the starting index of each subarray.
   - The inner loop (`end`) determines the ending index of each subarray.

3. For each subarray, we calculate the sum (`subarray_sum`) and check if it's divisible by `T`.

4. If a subarray sum is divisible by `T`, we increment our `count`.

5. After checking all subarrays, we return the total `count`.

6. Outside the function, we read the input from command-line arguments:

   - `n` and `T` are the first two arguments.
   - The `sales` array is constructed from the remaining arguments.

7. We call our function with the input data and print the result.

Here's a simple algorithm to solve the Sales Target Analysis problem, described in English with Python as pseudo-code:

1. Define a function to count divisible subarrays:

   ```python
   def count_divisible_subarrays(sales, T):
   count = 0
   for start in range(len(sales)):
       subarray_sum = 0
       for end in range(start, len(sales)):
           subarray_sum += sales[end]
           if subarray_sum % T == 0:
               count += 1
   return count
   ```

2. Read input from command-line arguments:

   ```python
   n, T = map(int, sys.argv[1:3])
   sales = list(map(int, sys.argv[3:]))
   ```

3. Call the function and print the result:

   ```python
   result = count_divisible_subarrays(sales, T)
   print(result)
   ```

The time complexity of this algorithm is $O(n^2)$, where $n$ is the number of elements in the sales array. This is because we're checking all possible subarrays, which requires two nested loops.

While this solution is straightforward and works for small inputs, it may not be efficient for large datasets (up to $10^5$ elements as per the problem constraints). An optimized solution using prefix sums and modular arithmetic could solve this problem in $O(n)$ time, but that's beyond the scope of a beginner's approach.

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    // Read n and T from command-line arguments
    int n = stoi(argv[1]);
    int T = stoi(argv[2]);

    // Read the sales data from command-line arguments
    vector<int> sales(n);
    for (int i = 0; i < n; ++i) {
        sales[i] = stoi(argv[i + 3]);
    }

    // Function to count subarrays with sum divisible by T
    int count = 0;

    // Iterate over all possible subarrays
    for (int start = 0; start < n; ++start) {
        int subarray_sum = 0;
        for (int end = start; end < n; ++end) {
            subarray_sum += sales[end];
            if (subarray_sum % T == 0) {
                count++;
            }
        }
    }

    // Output the result
    cout << count << endl;

    return 0;
}
```

> Both `atoi` and `stoi` are functions used in C++ to convert strings to integers, but they differ in terms of capabilities, safety, and how they handle errors.
>
> **`atoi` (ASCII to Integer)**
>
> The `atoi` function is a legacy C-style function that converts a C-string (i.e., a character array) to an integer. It is part of the `<cstdlib>` header.
>
> ```cpp
> int atoi(const char* str);
> ```
>
> - **Parameter**: The function takes a single argument, `str`, which is a pointer to a null-terminated C-string (an array of characters).
> - **Return**: The function returns the integer representation of the string, or `0` if the conversion fails.
>
> ```cpp
> #include <iostream>
> #include <cstdlib>
>
> int main() {
>     const char* str = "12345";
>     int num = atoi(str);
>     std::cout << num << std::endl;  // **Output**: 12345
>     return 0;
> }
> ```
>
> - `atoi` reads the string and converts valid characters (i.e., digits) into an integer.
> - If the string contains any non-numeric characters, `atoi` stops reading at the first invalid character and returns the number formed so far.
> - If the string does not contain a valid integer at all, it returns `0`.
>
> **Limitations**:
>
> - **Error Handling**: `atoi` does not provide any error checking or exception handling. If the input is invalid (e.g., an empty string or a string with non-numeric characters), it simply returns `0`, making it hard to detect if an error occurred.
> - **Overflow/Underflow**: There is no way to detect if the resulting integer overflows or underflows the limits of the `int` type.
>
> **`stoi` (String to Integer)**:
>
> The `stoi` function is a C++-specific function that is more robust and safer than `atoi`. It is part of the `<string>` header and can convert a `std::string` or C-string to an integer. Unlike `atoi`, `stoi` provides error handling and works with the `std::string` class.
>
> ```cpp
> int stoi(const std::string& str, size_t* idx = 0, int base = 10);
> ```
>
> - **Parameters**:
>   - `str`: A reference to a `std::string` object or a C-string to be converted.
>   - `idx` (optional): A pointer to a `size_t` variable where the function stores the index of the first invalid character. If no invalid character is found, this is ignored.
>   - `base` (optional): The base of the number system (default is base 10). It can handle different bases like hexadecimal (base 16), octal (base 8), etc.
> - **Return**: The function returns the integer representation of the string.
> - **Exception**: Throws `std::invalid_argument` if no conversion can be performed, or `std::out_of_range` if the resulting integer is out of range for the `int` type.
>
> ```cpp
> #include <iostream>
> #include <string>
>
> int main() {
>     std::string str = "6789";
>     int num = stoi(str);
>     std::cout << num << std::endl;  // **Output**: 6789
>
>     // Handling conversion with an invalid character
>     try {
>         std::string invalid_str = "123abc";
>         int invalid_num = stoi(invalid_str);  // Throws an exception
>     } catch (const std::invalid_argument& e) {
>         std::cout << "Invalid argument: " << e.what() << std::endl;
>     }
>
>     return 0;
> }
> ```
>
> - `stoi` starts from the beginning of the string and converts as many valid numeric characters as it can into an integer.
> - If the string contains any non-numeric characters, the function will throw an exception (`std::invalid_argument`).
> - If the resulting integer exceeds the bounds of the `int` type, `std::out_of_range` is thrown.
>
> **Advantages**:
>
> - **Error Handling**: Unlike `atoi`, `stoi` provides proper error handling through exceptions, making it more robust.
> - **Base Conversion**: `stoi` can convert numbers in different bases (e.g., hexadecimal or octal) by specifying the `base` parameter.
> - **Range Checking**: It handles integer overflow/underflow and throws `std::out_of_range` if the value is too large or too small for the `int` type.
>
> **Example with Base Conversion**:
>
> ```cpp
> std::string hex_str = "1A";  // Hexadecimal string
> int num = stoi(hex_str, nullptr, 16);  // Converts from base 16 (hex) to base 10
> std::cout << num << std::endl;  // **Output**: 26
> ```
>
> **Summary of Differences**:
>
> | Feature            | `atoi`                        | `stoi`                                                |
> | ------------------ | ----------------------------- | ----------------------------------------------------- |
> | Input              | C-string (`const char*`)      | `std::string` or C-string                             |
> | Error Handling     | Returns `0` for invalid input | Throws `std::invalid_argument` or `std::out_of_range` |
> | Base Conversion    | Only base 10                  | Supports multiple bases                               |
> | Exception Safety   | No                            | Yes (uses C++ exceptions)                             |
> | Overflow/Underflow | No handling                   | Detects and throws `std::out_of_range`                |
        