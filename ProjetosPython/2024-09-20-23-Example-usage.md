---
author: Frank
beforetoc: '[Anterior](2024-09-20-22-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-24-Sem-T%C3%ADtulo.md)'
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
arr = [1, 2, 3, 4, 5]  # Example array
queries = [(0, 2, 0), (1, 3, 1), (2, 4, 2)]  # Example queries (L, R, index)
result = mo_algorithm(arr, queries)  # Process the queries
print(result)  # Output the results
```

#### C++20 Code Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

// Structure to store each query
struct Query {
    int L, R, idx; // L and R are the bounds of the subarray, idx is the original index of the query
};

// Comparison function used to sort queries in Mo's Algorithm
bool compare(Query a, Query b) {
    // Define the block size as the square root of the number of elements
    int block_a = a.L / sqrt_n;
    int block_b = b.L / sqrt_n;

    // If the two blocks are different, sort by block
    if (block_a != block_b)
        return block_a < block_b;

    // If the blocks are the same, sort by the value of R
    return a.R < b.R;
}

// Function to process the queries using Mo's Algorithm
void moAlgorithm(vector<int>& arr, vector<Query>& queries) {
    int n = arr.size();            // Size of the input array
    int q = queries.size();        // Number of queries
    sqrt_n = sqrt(n);              // Square root of n, used for block size

    vector<int> result(q);         // Array to store the results of the queries
    vector<int> freq(1000001, 0);  // Frequency array to count occurrences of elements

    // Sort the queries using the compare function
    sort(queries.begin(), queries.end(), compare);

    int currL = 0, currR = 0;      // Initialize current left and right pointers
    int currSum = 0;               // Variable to store the current sum (or any other property)

    // Iterate over all queries
    for (int i = 0; i < q; i++) {
        int L = queries[i].L;      // Left bound of the current query
        int R = queries[i].R;      // Right bound of the current query

        // Move the current left pointer to L
        while (currL < L) {
            currSum -= arr[currL];  // Remove the element from the sum
            freq[arr[currL]]--;     // Decrease the frequency of the element
            currL++;                // Increment the current left pointer
        }
        while (currL > L) {
            currL--;                // Decrement the current left pointer
            currSum += arr[currL];  // Add the element to the sum
            freq[arr[currL]]++;     // Increase the frequency of the element
        }

        // Move the current right pointer to R
        while (currR <= R) {
            currSum += arr[currR];  // Add the element to the sum
            freq[arr[currR]]++;     // Increase the frequency of the element
            currR++;                // Increment the current right pointer
        }
        while (currR > R + 1) {
            currR--;                // Decrement the current right pointer
            currSum -= arr[currR];  // Remove the element from the sum
            freq[arr[currR]]--;     // Decrease the frequency of the element
        }

        // Store the result of the current query in the result array
        result[queries[i].idx] = currSum;
    }

    // Output the results of all queries
    for (int i = 0; i < q; i++) {
        cout << result[i] << endl;
    }
}
```

The code begins by reading the array and the queries. Next, the queries are sorted using the block decomposition technique. As we process each query, the current segment is adjusted to match the query’s range, and the current sum is updated. Finally, the answers are stored and output in the order of the original queries.

**Example**:

Let's look at a concrete example to better understand how Mo's Algorithm works in practice.

Given an array of $n$ integers, answer $q$ queries, each asking for the sum of a subarray from index $L_i$ to $R_i$.

**Sample Input**:

```txt
n = 5
arr = [1, 2, 3, 4, 5]
q = 3
queries = [(0, 2), (1, 3), (2, 4)]
```

**Expected Output**:

```txt
6   # Sum of arr[0...2] = 1 + 2 + 3
9   # Sum of arr[1...3] = 2 + 3 + 4
12  # Sum of arr[2...4] = 3 + 4 + 5
```

**Step-by-Step**:

1. **Sorting Queries**:
   With $\sqrt{5} \approx 2$, we divide the array into blocks of size 2.
   The sorted queries become: [(0, 2), (1, 3), (2, 4)]

2. **Processing**:

   - For (0, 2): We sum $1 + 2 + 3 = 6$
   - For (1, 3): We remove 1, add 4. New sum: $6 - 1 + 4 = 9$
   - For (2, 4): We remove 2, add 5. New sum: $9 - 2 + 5 = 12$

3. **Result**: [6, 9, 12]

This example shows how Mo's Algorithm minimizes work between adjacent queries, leveraging previous calculations.

Mo's Algorithm is highly effective for range query problems, making it ideal when multiple queries need to be answered over array intervals. Its efficiency has made it a popular tool in competitive programming, where speed is essential. Beyond that, it can also be adapted for data analysis, offering a way to efficiently handle subsets of large datasets.

However, there are some limitations to the algorithm. It is not suitable for handling online queries, where answers are required immediately as queries arrive. Additionally, since all queries must be stored, this can become a challenge for extremely large datasets. Finally, implementing Mo's Algorithm can be more complex than simpler, more straightforward methods, which might not be ideal in all cases.

#### Problem: "Humidity Levels in a Greenhouse" (Problem 1)

We've already solved this type of problem in _Section 5: Range Query Problems_ of the attached document. In that section, we explored different algorithms and analyzed their time and space complexities when applied to various range query scenarios.

Below is a summary of the time complexity for each solution, showing how Mo's Algorithm compares to other approaches.

| Solution                    | Time Complexity     | Space Complexity |
| --------------------------- | ------------------- | ---------------- |
| Naive Solution              | $O(n \times m)$     | $O(1)$           |
| Slightly Less Naive         | $O(n + m)$          | $O(1)$           |
| Parallel with `std::reduce` | $O(n + m)$          | $O(n)$           |
| Fenwick Tree (BIT)          | $O((n + m) \log n)$ | $O(n)$           |

Where:

$n$ = \text{number of sensors in the greenhouse}

$m$ = \text{number of adjustments}

These solutions have been discussed in depth, along with their respective advantages and limitations. For the current problem, all we need to do is implement Mo's Algorithm in C++, which provides a substantial performance improvement for large input sizes.

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Function to handle Mo's Algorithm for the humidity adjustments
vector<int> mo_algorithm(vector<int>& humidity, vector<pair<int, int>>& adjustments) {
    int n = humidity.size();  // Number of sensors
    int q = adjustments.size();  // Number of adjustments

    vector<int> result(q);  // To store the result for each adjustment
    int even_sum = 0;  // To keep track of the sum of even numbers

    // Calculate initial even sum
    for (int i = 0; i < n; i++) {
        if (humidity[i] % 2 == 0) {
            even_sum += humidity[i];
        }
    }

    // Process each adjustment
    for (int i = 0; i < q; i++) {
        int adj_value = adjustments[i].first;  // Value to add
        int sensor_index = adjustments[i].second;  // Sensor index

        // If the current value is even, remove it from the even sum
        if (humidity[sensor_index] % 2 == 0) {
            even_sum -= humidity[sensor_index];
        }

        // Apply the adjustment to the sensor
        humidity[sensor_index] += adj_value;

        // If the new value is even, add it to the even sum
        if (humidity[sensor_index] % 2 == 0) {
            even_sum += humidity[sensor_index];
        }

        // Store the result for this adjustment
        result[i] = even_sum;
    }

    return result;
}

void print_example(const vector<int>& humidity, const vector<pair<int, int>>& adjustments, const vector<int>& result, int example_num) {
    // Print the formatted example output
    cout << "Example " << example_num << ":" << endl;
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        cout << humidity[i];
        if (i < humidity.size() - 1) cout << ", ";
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        cout << "[" << adjustments[i].first << "," << adjustments[i].second << "]";
        if (i < adjustments.size() - 1) cout << ",";
    }
    cout << "]" << endl;
    cout << "**Output**: ";
    for (size_t i = 0; i < result.size(); ++i) {
        cout << result[i];
        if (i < result.size() - 1) cout << " ";
    }
    cout << endl << endl;
}

int main() {
    // Example 1
    vector<int> humidity1 = { 45, 52, 33, 64 };
    vector<pair<int, int>> adjustments1 = { {5, 0}, {-20, 1}, {-14, 0}, {18, 3} };
    vector<int> result1 = mo_algorithm(humidity1, adjustments1);  // Process the adjustments
    print_example(humidity1, adjustments1, result1, 1);

    // Example 2
    vector<int> humidity2 = { 40 };
    vector<pair<int, int>> adjustments2 = { {12, 0} };
    vector<int> result2 = mo_algorithm(humidity2, adjustments2);  // Process the adjustments
    print_example(humidity2, adjustments2, result2, 2);

    // Example 3
    vector<int> humidity3 = { 30, 41, 55, 68, 72 };
    vector<pair<int, int>> adjustments3 = { {10, 0}, {-15, 2}, {22, 1}, {-8, 4}, {5, 3} };
    vector<int> result3 = mo_algorithm(humidity3, adjustments3);  // Process the adjustments
    print_example(humidity3, adjustments3, result3, 3);

    return 0;
}
```

Now that we have implemented Mo's Algorithm in C++, we can compare its complexity with the previous solutions to the same problem. From a complexity point of view, the **Slightly Less Naive** solution has the lowest complexity, as shown in the table bellow.

| Solution                    | Time Complexity       | Space Complexity |
| --------------------------- | --------------------- | ---------------- |
| Naive Solution              | $O(n \times m)$       | $O(1)$           |
| Slightly Less Naive         | $O(n + m)$            | $O(1)$           |
| Parallel with `std::reduce` | $O(n + m)$            | $O(n)$           |
| Fenwick Tree (BIT)          | $O((n + m) \log n)$   | $O(n)$           |
| Mo's Algorithm              | $O((n + m) \sqrt{n})$ | $O(n)$           |

Where:

$n$ = \text{number of sensors in the greenhouse}

$m$ = \text{number of adjustments}

**Analysis for Small and Large Inputs**:

For **small inputs** (e.g., small values of $n$ and $m$):

- The **Slightly Less Naive** solution, with a time complexity of $O(n + m)$, will likely perform best due to its simplicity and minimal overhead. This solution efficiently handles small problems because the number of operations remains proportional to the sum of $n$ and $m$, without the logarithmic or square root factors present in more advanced algorithms.

- On the other hand, **Mo's Algorithm** and the **Fenwick Tree (BIT)** may introduce additional computational overhead due to the $\log n$ and $\sqrt{n}$ terms, which might not justify their use when $n$ and $m$ are small.

For **large inputs** (e.g., very large values of $n$ and $m$):

- **Mo's Algorithm**, with its complexity of $O((n + m)\sqrt{n})$, becomes more advantageous as $n$ grows, especially in cases where $\sqrt{n}$ is much smaller than $\log n$. This is particularly useful for large datasets where balancing query and update efficiency is crucial.

- The **Fenwick Tree (BIT)** remains efficient for large inputs as well, with a complexity of $O((n + m) \log n)$. However, depending on the relative sizes of $n$ and $m$, the logarithmic factor might make it slightly less competitive than **Mo's Algorithm** for extremely large inputs, particularly when $n$ grows significantly.

- The **Slightly Less Naive** solution, while efficient for small inputs, may struggle with scalability as it does not benefit from logarithmic or square root optimizations, leading to potential performance bottlenecks for very large input sizes.

## Auxiliary Data Structures

Specific data structures used to optimize operations on arrays.

### Deque (for Sliding Window Minimum/Maximum)

Double-ended queue that maintains relevant elements of the current window.

### Sparse Table (for RMQ)

Structure that stores pre-computed results for power-of-2 intervals.

### Segment Tree

Tree-based data structure for range queries and updates in $O(\log n)$.

## Complexity Optimization Techniques

Methods to reduce the computational complexity of common operations.

### Reduction from $O(n^2)$ to $O(n)$

Use of prefix sums to optimize range sum calculations.

- Problem Example: "Sales Target Analysis" - Uses prefix sum technique to optimize subarray calculations

### Update in $O(1)$

Difference arrays for constant-time range updates.

- Problem Example: "Inventory Restocking" - Makes point adjustments to the inventory

### Query in $O(1)$ after preprocessing

RMQ and static array queries with instant responses after pre-calculation.

- Problem Example: "The Plate Balancer" - After calculating cumulative sums, can find the "Magic Plate" in O(n)

### Processing in $O((n + q) \sqrt{n})$

Mo's Algorithm to optimize multiple range queries.

## Subarray Algorithms

Specific techniques for problems involving subarrays.

### Kadane's Algorithm

Finds the contiguous subarray with the largest sum in $O(n)$. Useful for sum maximization problems.

- Algorithm: Kadane's Algorithm

### Two Pointers

Technique for problems involving pairs of elements or subarrays that satisfy certain conditions.

- Algorithm: Two Pointers Method

## Hashing Techniques

Methods that use hashing to optimize certain operations on arrays.

### Prefix Hash

Uses hashing to quickly compare substrings or subarrays.

- Algorithm: Rolling Hash

### Rolling Hash

Technique to efficiently calculate hashes of substrings or subarrays when sliding a window.

- Algorithm: Rabin-Karp Algorithm

## Partitioning Algorithms

Techniques for dividing or reorganizing arrays.

### Partition Algorithm (QuickSelect)

Used to find the kth smallest element in average linear time.

- Algorithm: QuickSelect

### Dutch National Flag

Algorithm to partition an array into three parts, useful in sorting problems with few unique values.

- Algorithm: Dutch National Flag Algorithm

## The Fenwick Tree

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

**[Image placeholder]**
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

### Fundamental Concept

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

![]({{ site.baseurl }}/assets/images/bit1.jpg){: class="lazyimg"}
_Gráfico 1.1 - Example Fenwick tree diagram._{: class="legend"}

#### Querying the Fenwick tree

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

#### Updating the Fenwick tree

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

### Basic Operations

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
