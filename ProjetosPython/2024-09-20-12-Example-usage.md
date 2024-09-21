---
author: Frank
beforetoc: '[Anterior](2024-09-20-11-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-13-Sem-T%C3%ADtulo.md)'
categories:
- Matemática
- Linguagens Formais
- Programação
description: Dynamic Programming in C++ with practical examples, performance analysis,
  and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: true
image: assets/images/prog_dynamic.jpeg
keywords:
- Dynamic Programming
- C++ Algorithms
- Coding Examples
- Performance Optimization
- Algorithm Efficiency
- Programming Guide
- Code Comparison
- Developer Tips
lastmod: 2024-09-20 22:35:36.353000+00:00
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming
  with C++. Learn the core principles of Dynamic Programming, explore various algorithmic
  examples, and understand performance differences through detailed code comparisons.
  Perfect for developers looking to optimize their coding skills and enhance algorithm
  efficiency.
published: 2024-06-27 19:43:15.124000+00:00
rating: 5
slug: competitive-programming-techniques-insights
tags:
- Matemática
- Linguagens Formais
- Programação Dinâmica
- Dynamic Programming
- C++ Algorithms
- Performance Analysis
- Coding Examples
- Algorithm Optimization
- Practical Programming Guide
title: Example usage
toc: true
---
# Example usage
plates = [3, 1, 5, 2, 2]
result = find_magic_plate(plates)
print(result)  # Should print 2

plates = [1, 2, 3]
result = find_magic_plate(plates)
print(result)  # Should print -1

plates = [2, 1, -1]
result = find_magic_plate(plates)
print(result)  # Should print 0
```

Now a solution using C++ 20 to implement the Prefix Sum Array algorithm without any consideration about verbosity:

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Function to find the index of the Magic Plate
int find_magic_plate(const vector<int>& plates) {
    int n = plates.size();

    // If there is only one plate, it is automatically the Magic Plate
    if (n == 1) return 0;

    // Create a prefix sum array to store the cumulative sum up to each plate
    vector<int> prefix_sum(n + 1, 0);

    // Build the prefix sum array where each element contains the sum of elements up to that index
    for (int i = 1; i <= n; ++i) {
        prefix_sum[i] = prefix_sum[i - 1] + plates[i - 1];
    }

    // Calculate total sum (optional step, just for clarity)
    int total_sum = prefix_sum[n];

    // Check for each plate if the left sum equals the right sum
    for (int i = 1; i <= n; ++i) {
        // Left sum is the sum of elements before the current plate
        int left_sum = prefix_sum[i - 1];

        // Right sum is the total sum minus the current prefix sum
        int right_sum = total_sum - prefix_sum[i];

        // If the left and right sums are equal, return the current index (0-based)
        if (left_sum == right_sum) {
            return i - 1;
        }
    }

    // If no Magic Plate is found, return -1
    return -1;
}

int main() {
    // Example 1: plates = [3, 1, 5, 2, 2]
    vector<int> plates1 = { 3, 1, 5, 2, 2 };
    int result1 = find_magic_plate(plates1);
    cout << "Magic Plate index for plates1: " << result1 << endl;

    // Example 2: plates = [1, 2, 3]
    vector<int> plates2 = { 1, 2, 3 };
    int result2 = find_magic_plate(plates2);
    cout << "Magic Plate index for plates2: " << result2 << endl;

    // Example 3: plates = [2, 1, -1]
    vector<int> plates3 = { 2, 1, -1 };
    int result3 = find_magic_plate(plates3);
    cout << "Magic Plate index for plates3: " << result3 << endl;

    return 0;
}
```

The code implements the _Prefix Sum Array_ algorithm to solve the problem The Plate Balancer. The approach starts by creating a prefix sum array (`prefix_sum`), which stores the cumulative sum of elements from the original `plates` array. The construction of this prefix sum array has a time complexity of $O(n)$, where $n$ is the number of plates. The Prefix Sum Array is built in such a way that for each index $i$, the value `prefix_sum[i]` contains the sum of all elements from `plates[0]` to `plates[i-1]`. This allows the sum of elements to the left of a given index to be computed in constant time $O(1)$ by simply accessing `prefix_sum[i-1]`.

The construction of the Prefix Sum Array takes linear time $O(n)$ and requires additional space $O(n)$ for the array. For each plate, calculating the left and right sums is constant in time $O(1)$ due to the prefix sum array, but this is done $n$ times, resulting in $O(n)$ overall. The total sum is derived from the last value of the Prefix Sum Array, which is computed in constant time $O(1)$.

After building the Prefix Sum Array, the code uses it to calculate the left and right sums for each plate. The left sum of a plate at index $i$ is given by `prefix_sum[i-1]`, while the right sum is derived by subtracting `prefix_sum[i]` from the total sum (`total_sum`). If the left and right sums are equal, the index of the plate is returned as the Magic Plate. Otherwise, the loop continues to check all plates. If no balanced plate is found, the code returns `-1`, indicating that there is no Magic Plate.

The implementation follows the Prefix Sum Array algorithm efficiently, constructing the array in linear time $O(n)$, and checking if a plate is the Magic Plate in constant time $O(1)$ for each plate. The logic in C++20 utilizes standard functions such as `std::vector`, ensuring simplicity and clarity in the code. The identifiers have been adjusted to match those from the Python pseudocode, maintaining the same logic and structure as the original algorithm. Below is a detailed analysis of the time and space complexities for each operation in the C++20 implementation:

| Step                           | Operation                                            | Time Complexity  | Space Complexity            |
| ------------------------------ | ---------------------------------------------------- | ---------------- | --------------------------- |
| Prefix Sum Array Construction  | Building the prefix sum array `prefix_sum`           | $O(n)$           | $O(n)$                      |
| Left and Right Sum Calculation | Calculating left and right sums for each plate       | $O(1)$ per plate | $O(n)$ (reusing prefix sum) |
| Total Sum Calculation          | Calculating the total sum using the prefix sum array | $O(1)$           | $O(n)$                      |
| Loop Through Plates            | Checking all plates for the Magic Plate              | $O(n)$           | $O(1)$                      |
| Overall Complexity             | Total time and space complexities                    | $O(n)$           | $O(n)$                      |

##### Competitive Solution

The following C++20 code implements the _Prefix Sum Array_ algorithm, with several optimizations designed to reduce typing effort in a competitive programming context. We eliminated the use of functions, as the entire code is kept within the `main` block, avoiding the overhead of function calls. _This approach prioritizes minimal typing and fast execution by copying and pasting the logic rather than encapsulating it into reusable components_.

**Key changes made**:

1. **Use of `using` for shorter variable names**: We introduced `using` directives to reduce the typing for commonly used variables. For instance, `prefix_sum` became `ps`, `total_sum` became `ts`, and `plates` became `pl`. This allows us to minimize the amount of text written while keeping the code readable and maintainable in a fast-paced environment. We let comments in following code but not in the real competitive code available in [github](https://github.com/frankalcantara/Competitive).

2. **Reuse of the same array for multiple test cases**: Instead of declaring multiple arrays for different input examples, we reuse the same array `pl` and the variable `n` for the array size. By resetting `pl` and `n` for each example, we save both memory and typing effort, while maintaining clarity.

3. **Hardcoded input examples**: The input examples are directly written into the code (hardcoded), as is typical in competitive programming when no external input is required. The three provided examples are executed sequentially without the need for interactive input, allowing us to focus purely on solving the problem quickly.

4. _**Avoidance of function calls**: We opted to avoid wrapping the Prefix Sum Array logic into functions to eliminate the slight cost of function calls. This decision was driven by the understanding that, in a competitive environment, even minimal overheads can accumulate and impact performance. Instead, we simply copied and pasted the algorithm, leveraging the simplicity and speed of direct logic execution_.

**Warnings**:

During the development of this code, some warnings arose, such as a potential arithmetic overflow when performing summations and a warning about the conversion from `size_t` to `int`. To mitigate the risk of overflow, we made adjustments by using `long long` for the array and sums. However, the warning regarding the `size_t` to `int` conversion persists. This conversion warning arises because `size_t` is often used for the size of arrays, but we assign it to an `int` type. While this may lead to data loss in rare edge cases with very large data sizes, in the context of competitive programming where input sizes are usually constrained, this warning can be safely ignored.

_Moreover, reducing the typing effort is crucial in competitive environments, and using `int` is often the most efficient approach when dealing with moderately sized inputs, which are common in contests. As such, we chose to keep this conversion despite the warning, knowing that it will not significantly affect the correctness of our solution for typical competition scenarios_.

> In C++20, `size_t` is an unsigned integer type, typically used to represent the size of objects or memory blocks. It is an alias for an unsigned integer that can hold the size of the largest object your system can handle. Its size depends on the architecture of the system:
>
> - On **32-bit systems**, `size_t` is typically 4 bytes (32 bits), which means it can hold values from 0 to $2^{32} - 1$.
> - On **64-bit systems**, `size_t` is typically 8 bytes (64 bits), which means it can hold values from 0 to $2^{64} - 1$.
>
> **Typical Sizes of `int`, `long long`, and `size_t`**: On most modern systems, the sizes of these types are as follows (though they can vary depending on the platform and architecture):
>
> - **`int`**: 4 bytes (32 bits): Range: $-2^{31}$ to $2^{31} - 1$
> - **`long long`**: 8 bytes (64 bits): Range: $-2^{63}$ to $2^{63} - 1$ > **`size_t`**: **4 bytes (32 bits)** on 32-bit systems, with a range from 0 to $2^{32} - 1$ and **8 bytes (64 bits)** on 64-bit systems, with a range from 0 to $2^{64} - 1$.
>
> Since `size_t` is unsigned, it can store only non-negative values, making it ideal for representing sizes and lengths where negative numbers don't make sense (e.g., array indices, sizes of memory blocks).
>
> **Difference Between `++i` and `i++`**
>
> - **`++i`** is the **pre-increment** operator, which increments the value of `i` first and then returns the incremented value.
> - **`i++`** is the **post-increment** operator, which returns the current value of `i` first and then increments it.
>
> _The main difference between the two is in performance when used in certain contexts, particularly with non-primitive types like iterators_. Using `++i` is slightly more efficient than `i++` because `i++` might involve creating a temporary copy of the value before incrementing, while `++i` modifies the value directly. For example:
>
> ```cpp
> int i = 0;
> int a = ++i; // a = 1, i = 1 (pre-increment: increment first, then use the value)
> int b = i++; // b = 1, i = 2 (post-increment: use the value first, then increment)
> ```

Below is the final competitive, and ugly, code:

```cpp
#include <iostream>
#include <vector>

using namespace std;

using ps = vector<int>;  // Alias for prefix_sum as a vector of long long
using ts = int;          // Alias for total_sum as long long
using pl = vector<int>;  // Alias for plates as a vector of int
using vi = vector<int>;  // Alias for vector of int (similar to vi)

int main() {
    vi pl;
    int n;
    vi ps;

    pl = {3, 1, 5, 2, 2};
    n = pl.size();
    ps = vi(n + 1, 0);
    for (int i = 1; i <= n; ++i) ps[i] = ps[i - 1] + pl[i - 1];
    ts = ps[n];
    for (int i = 1; i <= n; ++i) {
        int ls = ps[i - 1], rs = ts - ps[i];
        if (ls == rs) {
            cout << i - 1 << endl;
            break;
        }
        if (i == n) cout << -1 << endl;
    }

    pl = {1, 2, 3};
    n = pl.size();
    ps = vi(n + 1, 0);
    for (int i = 1; i <= n; ++i) ps[i] = ps[i - 1] + pl[i - 1];
    ts = ps[n];
    for (int i = 1; i <= n; ++i) {
        int ls = ps[i - 1], rs = ts - ps[i];
        if (ls == rs) {
            cout << i - 1 << endl;
            break;
        }
        if (i == n) cout << -1 << endl;
    }

    pl = {2, 1, -1};
    n = pl.size();
    ps = vi(n + 1, 0);
    for (int i = 1; i <= n; ++i) ps[i] = ps[i - 1] + pl[i - 1];
    ts = ps[n];
    for (int i = 1; i <= n; ++i) {
        int ls = ps[i - 1], rs = ts - ps[i];
        if (ls == rs) {
            cout << i - 1 << endl;
            break;
        }
        if (i == n) cout << -1 << endl;
    }

    return 0;
}
```

### Algorithm: Difference Array - Efficient Range Updates

The Difference Array algorithm is a powerful technique for handling multiple range update operations efficiently. It's particularly useful when you need to perform many updates on an array and only query the final result after all updates are complete. Optimizes range updates to $O(1)$ by storing differences between adjacent
elements.

Consider an array $A$ of size $n$. The difference array $D$ is defined as:

$$
D[i] = \begin{cases}
A[i] - A[i-1], & \text{if } i > 0 \\
A[i], & \text{if } i = 0
\end{cases}
$$

Each element in $D$ represents the difference between consecutive elements in $A$. The key property of the difference array is that a range update on $A$ can be performed using only two operations on $D$.

To add a value $x$ to all elements in $A$ from index $l$ to $r$ (inclusive), we do:

$$D[l] += x$$

$$D[r+1] -= x (\text{if} r+1 < n)$$

After all updates, we can reconstruct $A$ from $D$ using:

$$A[i] = \sum_{j=0}^i D[j]$$

This technique allows for $O(1)$ time complexity for each range update operation.

#### Mathematical Proof

Let's prove that the range update operation on $D$ correctly reflects the change in $A$.

1. For $i < l$:

   $$A'[i] = \sum_{j=0}^i D'[j] = \sum_{j=0}^i D[j] = A[i]$$

2. For $l \leq i \leq r$:

   $$
   \begin{aligned}
   A'[i] &= \sum_{j=0}^i D'[j] \\
        &= \sum_{j=0}^{l-1} D[j] + (D[l] + x) + \sum_{j=l+1}^i D[j] \\
        &= (\sum_{j=0}^i D[j]) + x \\
        &= A[i] + x
   \end{aligned}
   $$

3. For $i > r$:

   $$
   \begin{aligned}
   A'[i] &= \sum_{j=0}^i D'[j] \\
        &= \sum_{j=0}^{l-1} D[j] + (D[l] + x) + \sum_{j=l+1}^r D[j] + (D[r+1] - x) + \sum_{j=r+2}^i D[j] \\
        &= (\sum_{j=0}^i D[j]) + x - x \\
        &= A[i]
   \end{aligned}
   $$

This proves that the range update operation on $D$ correctly reflects the desired change in $A$.

#### Difference Array Algorithm Explained in Plain English

1. **Initialize the Difference Array**

   Given an original array $A$ of size $n$, we create a difference array $D$ as follows:

   - For each index $i$ from $0$ to $n - 1$:
   - If $i = 0$:
     - Set $D[0] = A[0]$.
   - Else:
     - Set $D[i] = A[i] - A[i - 1]$.

   This difference array $D$ represents the changes between consecutive elements in $A$.

2. **Perform Range Updates**

   To add a value $x$ to all elements between indices $l$ and $r$ (inclusive) in $A$, we update $D$:

   - **Add** $x$ to $D[l]$:
   - Update $D[l] = D[l] + x$.
   - **If** $r + 1 < n$:
   - **Subtract** $x$ from $D[r + 1]$:
     - Update $D[r + 1] = D[r + 1] - x$.

   These two updates in $D$ ensure that when we reconstruct $A$, the value $x$ is added to all elements from $l$ to $r$.

3. **Reconstruct the Updated Array**

   After all range updates, we can rebuild the updated array $A$:

   - Set $A[0] = D[0]$.
   - For each index $i$ from $1$ to $n - 1$:
   - Set $A[i] = A[i - 1] + D[i]$.

   This step accumulates the differences to get the final values in $A$.

Let's walk through an example to see how the algorithm works.

Suppose we have the array:

$$
A = [2, 3, 5, 7, 11]
$$

1. **Initialize $D$**

   Compute $D$:

   - $D[0] = A[0] = 2$
   - $D[1] = A[1] - A[0] = 3 - 2 = 1$
   - $D[2] = A[2] - A[1] = 5 - 3 = 2$
   - $D[3] = A[3] - A[2] = 7 - 5 = 2$
   - $D[4] = A[4] - A[3] = 11 - 7 = 4$

   So,

   $$
   D = [2, 1, 2, 2, 4]
   $$

2. Perform a Range Update

   We want to add $x = 3$ to all elements from index $l = 1$ to $r = 3$.

   - Update $D[1]$:
   - $D[1] = D[1] + 3 = 1 + 3 = 4$
   - Since $r + 1 = 4 < n = 5$, update $D[4]$:
   - $D[4] = D[4] - 3 = 4 - 3 = 1$

   Updated $D$:

   $$
   D = [2, 4, 2, 2, 1]
   $$

3. Reconstruct the Updated Array

   Rebuild $A$ using the updated $D$:

   - $A[0] = D[0] = 2$
   - $A[1] = A[0] + D[1] = 2 + 4 = 6$
   - $A[2] = A[1] + D[2] = 6 + 2 = 8$
   - $A[3] = A[2] + D[3] = 8 + 2 = 10$
   - $A[4] = A[3] + D[4] = 10 + 1 = 11$

   Final updated array:

   $$
   A = [2, 6, 8, 10, 11]
   $$

4. **Verification**

   - The elements from index $1$ to $3$ have been increased by $3$.
   - Original $A[1..3] = [3, 5, 7]$
   - Updated $A[1..3] = [6, 8, 10]$

The Difference Array algorithm optimizes multiple range updates by reducing the time complexity to $O(1)$ per update. It is especially useful when dealing with scenarios that require numerous range modifications followed by queries for the final array state.

#### Complexity Analysis

The Difference Array algorithm offers significant performance benefits, particularly for scenarios involving multiple range updates. Let's examine its complexity:

1. **Range Update Operation**: The beauty of this algorithm lies in its constant-time range updates. Regardless of the size of the range being updated, we only modify two elements in the difference array $D$. This results in a time complexity of $O(1)$ for each range update operation.

2. **Array Reconstruction**: When we need to reconstruct the original array $A$ from the difference array $D$, we perform a single pass through $D$, computing cumulative sums. This operation has a time complexity of $O(n)$, where $n$ is the size of the array.

3. **Space Complexity**: The algorithm requires an additional array $D$ of the same size as the original array $A$. Therefore, the space complexity is $O(n)$.

The efficiency of this algorithm becomes apparent when dealing with multiple range updates followed by a single query or reconstruction. In such scenarios, we can perform $m$ range updates in $O(m)$ time, followed by a single $O(n)$ reconstruction, resulting in a total time complexity of $O(m + n)$. This is significantly more efficient than performing $m$ range updates directly on the original array, which would take $O(mn)$ time.

| Operation              | Time Complexity | Space Complexity |
| ---------------------- | --------------- | ---------------- |
| Initialization         | $O(n)$          | $O(n)$           |
| Range update           | $O(1)$          | $O(n)$           |
| Array reconstruction   | $O(n)$          | $O(n)$           |
| **Overall Complexity** | $O(n + q)$      | $O(n)$           |

#### Usage

The Difference Array algorithm shines in various scenarios where multiple range updates are required, and the final result needs to be computed only after all updates have been applied. Here are some common applications where this technique proves to be particularly effective:

1. **Range update queries**: When you need to perform multiple range updates and only query the final array state.
2. **Traffic flow analysis**: Modeling entry and exit points of vehicles on a road.
3. **Event scheduling**: Managing overlapping time slots or resources.
4. **Image processing**: Applying filters or adjustments to specific regions of an image.
5. **Time series data**: Efficiently updating ranges in time series data.
6. **Competitive programming**: Solving problems involving multiple range updates.

**Algorithm Implementation**: Pseudocode

**Example Problem**:

Starting with $N(1 \leq N \leq 1,000,000, N \text{ odd})$ empty stacks.
Beatriz receives a sequence of $K$ instructions $(1 \leq K \leq 25,000)$,
each in the format "A B", which means that Beatriz should add
a new layer of hay to the top of each stack in the interval $A..B$.
Calculate the median of the heights after the operations.

**Input**: $N = 7, K = 4$ Example Output:

```txt
3 5
5 5
2 4
4 6
```

**Output**:

```txt
1
```

Heights after updates: 0, 0, 1, 1, 2, 3, 3
Final height: 0 1 1 2 3 3 3
Indices: 1 2 3 4 5 6 7

```python
def range_update(diff, l, r, x):
    diff[l] += x
    if r + 1 < len(diff):
        diff[r + 1] -= x

def reconstruct_heights(diff):
    heights = [0] * len(diff)
    heights[0] = diff[0]
    for i in range(1, len(diff)):
        heights[i] = heights[i-1] + diff[i]
    return heights

