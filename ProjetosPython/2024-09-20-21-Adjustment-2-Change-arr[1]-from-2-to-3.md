---
author: Frank
beforetoc: '[Anterior](2024-09-20-20-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-22-Sem-T%C3%ADtulo.md)'
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
title: 'Adjustment 2: Change arr[1] from 2 to 3'
toc: true
---
# Adjustment 2: Change arr[1] from 2 to 3
adjust(arr, 1, 3, even_sum)  # even_sum = 10 (6 + 4)
```

#### Complexity Analysis

The algorithm's overall time complexity can be expressed as $O(n + m)$, where $n$ is the initial array size and $m$ is the number of adjustments. This represents a significant improvement over the naive approach of recalculating the sum after each adjustment, which would result in a time complexity of $O(n \times m)$.

In scenarios involving large arrays with frequent updates, the Incremental Sum Algorithm offers substantial performance benefits. It proves particularly useful in real-time data processing, financial calculations, and various computational problems where maintaining a running sum is crucial. By avoiding redundant calculations, it not only improves execution speed but also reduces computational resource usage, making it an invaluable tool for efficient array manipulation and sum maintenance in a wide range of applications.

#### Incremental Sum Mathematical Definitions

Let:

- $n$ be the size of the array $A$,
- $Q$ be the number of queries (adjustments),
- $A[i]$ be the value at index $i$ in the array,
- $adjustments[k] = [val_k, index_k]$ be the adjustment in the $k$-th query, where $val_k$ is the adjustment value and $index_k$ is the index to be adjusted.

Our goal is to calculate the sum of the even numbers in $A$ after each adjustment incrementally, without recalculating the entire sum from scratch after each query.

**Step 1: Initial Calculation of the Sum of Even Numbers**:

First, define $S$ as the initial sum of even numbers in the array $A$. This sum can be expressed as:

$$S = \sum_{i=0}^{n-1} \text{if } (A[i] \% 2 == 0) \text{ then } A[i]$$

The conditional function indicates that only even values are summed.

**Step 2: Incremental Update**:

When we receive a query $adjustments[k] = [val_k, index_k]$, we adjust the value at index $index_k$ by adding $val_k$ to the current value of $A[index_k]$. The new value is:

$$\text{new\_value} = A[index_k] + val_k$$

We update the sum $S$ efficiently as follows:

1. If the original value $A[index_k]$ was **even**, we subtract it from $S$:

   $$
   S = S - A[index_k]
   $$

2. After applying the adjustment, if the new value $\text{new\_value}$ is **even**, we add it to $S$:

   $$
   S = S + \text{new\_value}
   $$

**Formal Analysis of Updates**:

For each adjustment, we have the following operations:

- **Remove the old value (if even):**
  If $A[index_k]$ is even before the adjustment:

  $$ S = S - A[index_k] $$

- **Add the new value (if even):**
  If $\text{new\_value}$ is even after the adjustment:

  $$ S = S + \text{new_value} $$

These two operations ensure that the sum $S$ is correctly maintained after each adjustment.

**Demonstration for a Generic Example**:

Let us demonstrate the update for a generic example. Suppose we have the initial array:

$$A = [a_0, a_1, a_2, \dots, a_{n-1}]$$

The initial sum of even numbers will be:

$$S = \sum_{i=0}^{n-1} \text{if } a_i \% 2 == 0 \text{ then } a_i$$

Now, let $adjustments[k] = [val_k, index_k]$ be an adjustment:

- The previous value of $A[index_k]$ is $a_{index_k}$.
- The new value will be:

  $$\text{new\_value} = a_{index_k} + val_k$$

The sum $S$ will be updated as follows:

- If $a_{index_k} \% 2 == 0$ (i.e., the old value was even), then:

  $$S = S - a_{index_k}$$

- If $\text{new\_value} \% 2 == 0$ (i.e., the new value is even), then:

  $$S = S + \text{new\_value}$$

**Mathematical Justification**:

With each adjustment, we ensure that:

1. If the old value was even, it is removed from the sum $S$.
2. If the new value is even, it is added to the sum $S$.

These operations guarantee that the sum of all even numbers is correctly maintained without the need to recalculate the entire sum after each adjustment.

### Incremental Sum Algorithm Explained in Plain English

The **Incremental Sum Algorithm** efficiently maintains the sum of specific elements in an array (such as even numbers) when the array undergoes frequent changes. Instead of recalculating the entire sum after each modification, it updates the sum incrementally, which saves time and computational resources.

1. **Initial Sum Calculation**

   - **Step 1**: Calculate the initial sum of the elements of interest in the array.
     - For example, sum all even numbers in the array.
     - Iterate through the array once.
     - Add each element to the sum if it meets the condition (e.g., if it's even).

2. Processing Adjustments

   When an element in the array is adjusted (modified), the algorithm updates the sum as follows:

   1. **Subtract the Old Value (if it affects the sum)**:

      - Check if the old value at the adjusted index meets the condition (e.g., is even).
      - If it does, subtract this old value from the sum.

   2. **Update the Array Element**:

      - Modify the array element with the new value.

   3. **Add the New Value (if it affects the sum)**:

      - Check if the new value meets the condition.
      - If it does, add the new value to the sum.

   These steps ensure that the sum remains accurate without needing to recalculate it from scratch.

**Example**:

Consider the array:

```txt
A = [1, 2, 3, 4, 5]
```

Initial Sum of Even Numbers: $Sum = 2 + 4 = 6$

Adjustment 1: Change `A[0]` from $1$ to $6$

1.  **Old Value**: `A[0] = 1` (odd)

    - Since it's odd, it doesn't affect the sum.

2.  **Update Element**:

    - `A[0] = 1 + 5 = 6`

3.  **New Value**: `A[0] = 6` (even)
    - Add the new value to the sum: Sum = 6 + 6 = **12**

Adjustment 2: Change `A[1]` from $2$ to $3$

1.  **Old Value**: `A[1] = 2` (even)

    - Subtract the old value from the sum: Sum = 12 - 2 = **10**

2.  **Update Element**:

    - `A[1] = 2 + 1 = 3`

3.  **New Value**: `A[1] = 3` (odd)
    - Since it's odd, the sum remains unchanged.

Adjustment 3: Change `A[2]` from $3$ to $2$

1.  **Old Value**: `A[2] = 3` (odd)

    - Doesn't affect the sum.

2.  **Update Element**:

    - `A[2] = 3 - 1 = 2`

3.  **New Value**: `A[2] = 2` (even)
    - Add the new value to the sum: Sum = 10 + 2 = **12**

### Problem Example: "Humidity Levels in a Greenhouse" (Problem 1)

The same problem we saw earlier in the section: **Algorithm: Difference Array - Efficient Range Updates**. Below is the implementation in C++20:

```cpp
#include <vector>
#include <iostream>
using namespace std;
using vi = vector<long long>;

// Function to compute the sum of even numbers after each adjustment
vi sumEvenAfterAdjustments(vi& humidity, const vector<vi>& adjustments) {
    long long sumEven = 0;
    vi result;

    // Calculate the initial sum of even numbers in the humidity array
    for (auto level : humidity) {
        if (level % 2 == 0) {
            sumEven += level;
        }
    }

    // Process each adjustment
    for (const auto& adjustment : adjustments) {
        long long val = adjustment[0];  // The adjustment value to add
        int index = adjustment[1];      // The index of the sensor to adjust
        long long oldValue = humidity[index];  // Store the old humidity value
        long long newValue = oldValue + val;   // Compute the new humidity value

        // Apply the adjustment to the humidity array
        humidity[index] = newValue;

        // --- Incremental sum update algorithm starts here ---
        // Update sumEven based on the old and new values

        // If the old value was even, subtract it from sumEven
        if (oldValue % 2 == 0) {
            sumEven -= oldValue;  // Remove the old even value from the sum
        }
        // If the new value is even, add it to sumEven
        if (newValue % 2 == 0) {
            sumEven += newValue;  // Add the new even value to the sum
        }
        // --- Incremental sum update algorithm ends here ---

        // Store the current sum after the adjustment
        result.push_back(sumEven);
    }
    return result;
}

int main() {
    // Example 1
    vi humidity1 = { 45, 52, 33, 64 };
    vector<vi> adjustments1 = { {5, 0}, {-20, 1}, {-14, 0}, {18, 3} };
    vi result1 = sumEvenAfterAdjustments(humidity1, adjustments1);
    cout << "Example 1: ";
    for (const auto& sum : result1) cout << sum << " ";
    cout << endl;

    // Example 2
    vi humidity2 = { 40 };
    vector<vi> adjustments2 = { {12, 0} };
    vi result2 = sumEvenAfterAdjustments(humidity2, adjustments2);
    cout << "Example 2: ";
    for (const auto& sum : result2) cout << sum << " ";
    cout << endl;

    // Example 3
    vi humidity3 = { 30, 41, 55, 68, 72 };
    vector<vi> adjustments3 = { {10, 0}, {-15, 2}, {22, 1}, {-8, 4}, {5, 3} };
    vi result3 = sumEvenAfterAdjustments(humidity3, adjustments3);
    cout << "Example 3: ";
    for (const auto& sum : result3) cout << sum << " ";
    cout << endl;

    return 0;
}
```

### Static Array Queries

Techniques for arrays that don't change between queries, allowing efficient pre-calculations.

- Algorithm: Sparse Table

- Problem Example: "Inventory Restocking" - Performs queries after each inventory adjustment

### Range Minimum Queries (RMQ)

Data structure to find the minimum in any range in $O(1)$ after $O(n \log n)$ preprocessing.

- Algorithm: Sparse Table for RMQ

### Fenwick Tree

Data structure for prefix sums and efficient updates, with operations in $O(\log n)$.

- Algorithm: Binary Indexed Tree (BIT)

## Sliding Window Algorithms

Techniques for efficiently processing contiguous subarrays of fixed size.

### Sliding Window Minimum

Finds the minimum in a fixed-size window that slides through the array in $O(n)$ using a deque.

- Algorithm: Monotonic Deque

### Sliding Window Maximum

Similar to the minimum, but for finding the maximum in each window.

- Algorithm: Monotonic Deque

- Problem Example: "Weather Monitoring System" - Uses a sliding window of size k to find the subarray with the highest average

## Multiple Query Processing

Methods for handling multiple queries efficiently.

### Mo's Algorithm

Imagine you're organizing a library with thousands of books. You need to answer questions about specific sections of the shelves, and each question feels like wandering through endless rows, searching for the right answers. Enter Mo's Algorithm. It’s like having a librarian who doesn’t waste time. This librarian knows exactly how to group your questions, answering them swiftly, without scouring the entire library each time.

Mo's Algorithm was developed by the Bangladeshi programmer [Mostofa Saad Ibrahim](https://sites.google.com/site/mostafasibrahim/). It’s a technique that allows efficient answers to range queries on arrays. The trick? It works best with offline queries, those you can reorder. Over time, it has become a crucial part of the competitive programmer’s toolkit, speeding up what once was slow.

$$ Mo's \, Algorithm \, = \, \text{Efficient \, Librarian} $$

With Mo's Algorithm, each question becomes easier, and each answer quicker, which makes it invaluable for competitive programming.

Imagine you have an array of $n$ elements and $q$ queries. Each query asks for some property, maybe the sum or the frequency of elements, over a subarray $[L_i, R_i]$. The simple way is to handle each query on its own. You would go through the array again and again, and before you know it, you’re dealing with a time complexity of $O(n \times q)$. For large arrays and many queries, that’s just too slow.

This is where Mo's Algorithm steps in. It answers all your queries in $O(n \sqrt{n})$ time, assuming add and remove operations take $O(1)$. For big datasets, that’s the difference between drowning in work and getting it done on time.

Mo’s Algorithm works by processing queries in a way that reduces how often elements are added or removed from the current segment. It achieves this in two steps:

$$ 1. \, \text{Reorder \, queries \, for \, efficiency.} $$

$$ 2. \, \text{Add \, and \, remove \, elements \, smartly.} $$

With Mo's Algorithm, even large sets of queries can be handled efficiently:

1. **Sorting the queries**: The array is divided into blocks of size $\sqrt{n}$. Queries are then sorted, first by the block of $L_i$, and within the same block, by $R_i$.

2. **Processing the queries**: As we move from one query to the next, we adjust the boundaries of the current segment, adding or removing elements as necessary.

This method keeps the operations minimal and ensures a much faster solution.

#### Why Choose $\sqrt{n}$ as the Block Size?

The choice of $\sqrt{n}$ as the block size is crucial for the algorithm's efficiency. Here's why:

- The number of blocks becomes $\sqrt{n}$.
- The number of times we change the left boundary is $O(\sqrt{n})$.
- The total number of add/remove operations is $O(n \sqrt{n})$.

This choice balances the work done when moving between blocks and within blocks, optimizing overall performance.

#### Complexity Analysis

**Time Complexity Analysis**:

The total time complexity of Mo's Algorithm is:

$$O\left( q \times \frac{n}{\sqrt{n}} + n \sqrt{n} \right) = O(n \sqrt{n})$$

To understand this, sorting the queries takes $O(q \log q)$ time. Given that $q$ is generally $O(n)$, this remains efficient. The adjustment of segment boundaries between queries takes $O(n \sqrt{n})$ time, which contributes to the overall complexity. When compared to the naive approach with a time complexity of $O(n \times q)$, Mo's Algorithm provides a marked improvement, particularly when dealing with larger datasets.

**Space Complexity Analysis**:

The space complexity of Mo's Algorithm is:

$$O(n + q)$$

To understand the space complexity, we need $O(n)$ space to store the array elements and $O(q)$ space to store the queries. As a result, the overall space usage is linear, ensuring that the algorithm remains efficient even for large datasets.

#### Implementation

Let's see how to implement Mo's Algorithm in Python and C++20. These implementations assume we're calculating the sum over intervals, but the concept can be adapted for other types of queries.

#### Python Pseudocode

```python
import math

