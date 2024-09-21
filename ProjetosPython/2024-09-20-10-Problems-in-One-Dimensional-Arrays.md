---
author: Frank
beforetoc: '[Anterior](2024-09-20-9-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-11-Sem-T%C3%ADtulo.md)'
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
title: Problems in One-Dimensional Arrays
toc: true
---
# Problems in One-Dimensional Arrays

One-dimensional arrays are fundamental data structures in computer science and are the basis for many algorithmic problems. This classification organizes common problem types, algorithms, and techniques used to solve challenges involving 1D arrays. From basic operations to advanced optimization strategies, this comprehensive guide covers a wide range of approaches, helping developers and algorithm enthusiasts to identify and apply the most efficient solutions to array-based problems.

## Preprocessing and Efficient Query Techniques

Methods that prepare the array to respond to queries quickly, typically trading preprocessing time for faster queries. This approach involves investing time upfront to organize or transform the array data in a way that allows for rapid responses to subsequent queries. For example, in a scenario where frequent sum calculations of array intervals are needed, a preprocessing step might involve creating a prefix sum array. This initial step takes $O(n)$ time but enables constant-time $O(1)$ sum queries afterward, as opposed to $O(n)$ time per query without preprocessing. This trade-off is beneficial when the number of queries is large, as the initial time investment is offset by the significant speed improvement in query operations. Such techniques are common in algorithmic problem-solving, where strategic data preparation can dramatically enhance overall performance, especially in scenarios with repetitive operations on the same dataset.

### Algorithm: Sums and Prefixes

Calculation of cumulative sums for fast range queries. Reduces complexity from $O(n^2)$ to $O(n)$ in construction and $O(1)$ per query.

#### Algorithm: Prefix Sum Array

The Prefix Sum Array is a preprocessing technique used to efficiently calculate the sum of elements in a given range of an array. It works by creating a new array where each element is the sum of all previous elements in the# Prefix Sum Array Algorithm

The **Prefix Sum Array** is a technique used to calculate the sum of elements in a given range of an array efficiently. It involves creating a new array where each element is the cumulative sum of all previous elements in the original array, including the current one.

Given an array $A$ of $n$ elements, the prefix sum array $P$ is defined as:

$$
P[i] = \sum_{k=0}^{i} A[k], \quad \text{for } 0 \leq i < n
$$

This means that each element $P[i]$ represents the sum of all elements from $A[0]$ to $A[i]$.

1. **Construction**

   1. **Initialize** $P[0] = A[0]$
   2. **For** $i$ from $1$ to $n - 1$:
      - $P[i] = P[i - 1] + A[i]$

2. **Usage**

To find the sum of elements from index $i$ to $j$ (inclusive) in the original array $A$:

- **If** $i = 0$:
  - $\text{Sum}(0, j) = P[j]$
- **If** $i > 0$:
  - $\text{Sum}(i, j) = P[j] - P[i - 1]$

This allows for constant time $O(1)$ range sum queries after the initial $O(n)$ preprocessing.

We will prove that the range sum $\text{Sum}(i, j) = P[j] - P[i - 1]$ correctly computes the sum of elements from index $i$ to $j$ in array $A$.

**Case 1**: When $i = 0$.

- $\text{Sum}(0, j) = P[j]$
- Since $P[j] = \sum_{k=0}^{j} A[k]$, it directly gives the sum from $A[0]$ to $A[j]$.

**Case 2**: When $i > 0$.

- $P[j] = \sum_{k=0}^{j} A[k]$
- $P[i - 1] = \sum_{k=0}^{i - 1} A[k]$
- Therefore:
  $$
  \begin{aligned}
  \text{Sum}(i, j) &= P[j] - P[i - 1] \\
                   &= \left( \sum_{k=0}^{j} A[k] \right) - \left( \sum_{k=0}^{i - 1} A[k] \right) \\
                   &= \sum_{k=i}^{j} A[k]
  \end{aligned}
  $$
- This shows that $\text{Sum}(i, j)$ correctly computes the sum of elements from $A[i]$ to $A[j]$.

#### Algorithm Prefix Sum in Plain English

The **Prefix Sum Array** is an algorithm that helps quickly calculate the sum of any subarray (a range of elements) within an original array. After an initial preprocessing step, you can find the sum of elements between any two indices in constant time.

1. **Construct the Prefix Sum Array**

   Given an original array $A$ of size $n$, we create a prefix sum array $P$:

   - **Initialize**:
   - Set $P[0] = A[0]$.
   - **Iterate**:
   - For each index $i$ from $1$ to $n - 1$:
     - Calculate $P[i] = P[i - 1] + A[i]$.
   - **Purpose**:
   - Each element $P[i]$ represents the total sum of all elements from $A[0]$ up to $A[i]$.

2. **Perform Range Sum Queries**

   To find the sum of elements from index $i$ to $j$ (inclusive):

   - **If** $i = 0$:
   - The sum is simply $P[j]$.
   - **If** $i > 0$:
   - The sum is $P[j] - P[i - 1]$.
   - **Reasoning**:
   - $P[j]$ includes the sum from $A[0]$ to $A[j]$.
   - Subtracting $P[i - 1]$, which is the sum from $A[0]$ to $A[i - 1]$, leaves us with the sum from $A[i]$ to $A[j]$.

### Understanding

1. **Construction Phase**:

   - **What Happens**:
     - We iterate through the original array once.
     - At each step, we add the current element to the cumulative sum.
   - **Result**:
     - We get an array where each position holds the total sum up to that index.

2. **Query Phase**

   - **Efficient Summation**:
     - Instead of adding up elements each time we need a sum, we use the precomputed sums.
   - **Quick Calculation**:
     - By using the formula, we reduce the time complexity of range sum queries to $O(1)$.

#### Example

Suppose we have the array:

$$A = [3, 1, 4, 1, 5, 9, 2, 6]$$

**Step 1**: Construct the Prefix Sum Array $P$

Compute $P$:

- $P[0] = A[0] = 3$
- $P[1] = P[0] + A[1] = 3 + 1 = 4$
- $P[2] = P[1] + A[2] = 4 + 4 = 8$
- $P[3] = P[2] + A[3] = 8 + 1 = 9$
- $P[4] = P[3] + A[4] = 9 + 5 = 14$
- $P[5] = P[4] + A[5] = 14 + 9 = 23$
- $P[6] = P[5] + A[6] = 23 + 2 = 25$
- $P[7] = P[6] + A[7] = 25 + 6 = 31$

Resulting prefix sum array:

$$P = [3, 4, 8, 9, 14, 23, 25, 31]$$

**Step 2**: Perform Range Sum Queries

Example Query\*: Find the sum of elements from index $2$ to $5$ in $A$.

- **Compute**: Since $i = 2 > 0$, use $\text{Sum}(2, 5) = P[5] - P[1]$
- **Calculate**: $\text{Sum}(2, 5) = 23 - 4 = 19$
- **Verification**:
  - Sum of $A[2]$ to $A[5]$:
    - $A[2] + A[3] + A[4] + A[5] = 4 + 1 + 5 + 9 = 19$

#### Complexity Analysis

The Prefix Sum Array algorithm's complexity can be analyzed by considering its two main operations: constructing the prefix sum array and performing range sum queries.

In the construction phase, we initialize the prefix sum array $P$ by setting $P[0] = A[0]$, which requires constant time $O(1)$. Then, for each index $i$ from $1$ to $n - 1$, we compute $P[i] = P[i - 1] + A[i]$. This loop runs for $n - 1$ iterations, and each iteration involves a single addition operation, which is a constant-time operation $O(1)$. Therefore, the total time complexity for constructing the prefix sum array is:

$$O(1) + (n - 1) \times O(1) = O(n)$$

Thus, the construction of the prefix sum array has a linear time complexity of $O(n)$. Regarding space complexity, we require an additional array $P$ of size $n$ to store the prefix sums, resulting in an extra space complexity of $O(n)$.

For performing a range sum query to calculate the sum of elements from index $i$ to $j$ in the original array $A$, we utilize the prefix sum array $P$. If $i = 0$, the sum is simply $P[j]$, which is retrieved in constant time $O(1)$. If $i > 0$, the sum is calculated as $P[j] - P[i - 1]$, involving two array accesses and one subtraction, all of which are constant-time operations. Therefore, each range sum query is executed in $O(1)$ time.

The space complexity for executing queries is $O(1)$, as no additional space is required beyond the already constructed prefix sum array.

In conclusion, the Prefix Sum Array algorithm has a time complexity of $O(n)$ for the preprocessing step of constructing the prefix sum array and $O(1)$ time per range sum query. The overall space complexity is $O(n)$ due to the storage of the prefix sum array. This efficiency makes the algorithm particularly useful when dealing with multiple range sum queries on a static array, as it significantly reduces the time complexity per query from $O(n)$ to $O(1)$ after the initial preprocessing.

##### The Plate Balancer (Problem 2)

In a famous restaurant, Chef André is known for his incredible skill in balancing plates. He has a long table with several plates, each containing a different amount of food. André wants to find the "Magic Plate" - the plate where, when he places his finger underneath it, the weight of the food on the left and right balances perfectly.

Given a list of $plates$, where each number represents the weight of the food on each plate, your task is to help André find the index of the Magic Plate. The Magic Plate is the one where the sum of the weights of all plates to its left is equal to the sum of the weights of all plates to its right.

If André places his finger under the leftmost plate, consider the weight on the left as $0$. The same applies if he chooses the rightmost plate.

Return the leftmost Magic Plate index. If no such plate exists, return $-1$.

**Example 1:**

**Input**: $plates = [3,1,5,2,2]$
**Output**: $2$
Explanation:
The Magic Plate is at index $2$.
Weight on the left = $plates[0] + plates[1] = 3 + 1 = 4$
Weight on the right = $plates[3] + plates[4] = 2 + 2 = 4$

**Example 2:**

**Input**: $plates = [1,2,3]$
**Output**: $-1$
Explanation:
There is no plate that can be the Magic Plate.

**Example 3:**

**Input**: $plates = [2,1,-1]$
**Output**: $0$
Explanation:
The Magic Plate is the first plate.
Weight on the left = $0$ (no plates to the left of the first plate)
Weight on the right = $plates[1] + plates[2] = 1 + (-1) = 0$

**Constraints:**

$$1 \leq plates.length \leq 10^4$$
$$-1000 \leq plates[i] \leq 1000$$

Note: André is very skilled, so don't worry about the real-world physics of balancing plates. Focus only on the mathematical calculations!

###### Naïve Solution

This solution is considered naïve because it doesn't take advantage of any precomputation or optimization techniques such as the Prefix Sum Array. Instead, it recalculates the sum of elements to the left and right of each plate using two separate loops for every plate. This leads to a time complexity of $O(n^2)$, as for each plate, the entire array is traversed twice — once for the left sum and once for the right sum.

_A developer who writes this kind of code typically has a basic understanding of problem-solving but might not be familiar with more advanced algorithms or computational complexity analysis_. They often rely on straightforward, brute-force approaches, focusing on getting a working solution without considering performance for large datasets. While this approach works for small inputs, it quickly becomes inefficient for larger ones due to its quadratic complexity.

The following is a Python pseudocode version of the naïve C++ solution, using the same variables and logic:

```python
def find_magic_plate_naive(plates):
    n = len(plates)

    # Check every plate to see if it's the Magic Plate
    for i in range(n):
        left_sum = 0
        right_sum = 0

        # Calculate sum of elements to the left of plate i
        for j in range(i):
            left_sum += plates[j]

        # Calculate sum of elements to the right of plate i
        for j in range(i + 1, n):
            right_sum += plates[j]

        # If left and right sums are equal, return the current index
        if left_sum == right_sum:
            return i

    # If no Magic Plate found, return -1
    return -1

