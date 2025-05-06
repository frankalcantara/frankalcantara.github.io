---
author: Frank
beforetoc: |-
    [Anterior](2024-09-24-10-10.-Loops-the-Heart-of-All-Competitive-Programming.md)
    [Próximo](2024-09-24-12-Search-and-Sorting-Algorithms.md)
categories:
    - Matemática
    - Linguagens Formais
    - Programação
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: false
image: assets/images/prog_dynamic.webp
keywords:
    - Dynamic Programming
    - C++ Algorithms
lastmod: 2025-05-06T11:04:18.034Z
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Competitive Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: competitive-programming-techniques-insights
tags:
    - Practical Programming Guide
title: 11. Problems in One-Dimensional Arrays
toc: true
---

# One-Dimensional Arrays

One-dimensional arrays are fundamental data structures in computer science and are the basis for many algorithmic problems. This classification organizes common problem types, algorithms, and techniques used to solve challenges involving 1D arrays. From basic operations to advanced optimization strategies, this comprehensive guide covers a wide range of approaches, helping developers and algorithm enthusiasts to identify and apply the most efficient solutions to array-based problems.

## 11.1. Preprocessing and Efficient Query Techniques - Arrays

Methods that prepare the array to respond to queries quickly, typically trading preprocessing time for faster queries. This approach involves investing time upfront to organize or transform the array data in a way that allows for rapid responses to subsequent queries. For example, in a scenario where frequent sum calculations of array intervals are needed, a preprocessing step might involve creating a prefix sum array. This initial step takes $O(n)$ time but enables constant-time $O(1)$ sum queries afterward, as opposed to $O(n)$ time per query without preprocessing. This trade-off is beneficial when the number of queries is large, as the initial time investment is offset by the significant speed improvement in query operations. Such techniques are common in algorithmic problem-solving, where strategic data preparation can dramatically enhance overall performance, especially in scenarios with repetitive operations on the same dataset.

### 11.1.1 Algorithm: Prefix Sum Array

Calculation of cumulative sums for fast range queries. Reduces complexity from $O(n^2)$ to $O(n)$ in construction and $O(1)$ per query.

The Prefix Sum Array is a preprocessing technique used to efficiently calculate the sum of elements in a given range of an array. It works by creating a new array where each element is the sum of all previous elements in the Prefix Sum Array Algorithm

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

#### 11.1.1.1. Algorithm Prefix Sum in Plain English

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

1. **Perform Range Sum Queries**

   To find the sum of elements from index $i$ to $j$ (inclusive):

   - **If** $i = 0$:
   - The sum is simply $P[j]$.
   - **If** $i > 0$:
   - The sum is $P[j] - P[i - 1]$.
   - **Reasoning**:
   - $P[j]$ includes the sum from $A[0]$ to $A[j]$.
   - Subtracting $P[i - 1]$, which is the sum from $A[0]$ to $A[i - 1]$, leaves us with the sum from $A[i]$ to $A[j]$.

Let's understand this algorithm set by step:

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

**Example - Prefix Sum Array**:

Suppose we have the array:

$$A = [3, 1, 4, 1, 5, 9, 2, 6]$$

**Step 1**: Construct the Prefix Sum Array $P$

![]({{ site.baseurl }}/assets/images/PrefixSum1.webp)
_Figura 11.1.1.1.A - Step in the Prefix Sum Algorithm: The image shows the calculation of the fifth element of the prefix sum array $P$ from the original array $A$._{: class="legend"}

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

Example Query: Find the sum of elements from index $2$ to $5$ in $A$.

![]({{ site.baseurl }}/assets/images/PrefixSum3.webp)
_Figura 11.1.1.1.A - The image illustrates how the prefix sum array $P$ is constructed from the original array $A$. Example shows the sum calculation from $A[2]$ to $A[5]$ using the prefix sums._{: class="legend"}

- **Compute**: Since $i = 2 > 0$, use $\text{Sum}(2, 5) = P[5] - P[1]$
- **Calculate**: $\text{Sum}(2, 5) = 23 - 4 = 19$
- **Verification**:
  - Sum of $A[2]$ to $A[5]$:
    - $A[2] + A[3] + A[4] + A[5] = 4 + 1 + 5 + 9 = 19$

![]({{ site.baseurl }}/assets/images/PrefixSum4.webp)
_Figura 11.1.1.1.A - The image demonstrates how to use the prefix sum array $P$ to calculate the sum of elements from $A[2]$ to $A[5]$. The calculation is done using the formula $P[5] - P[1]$, resulting in $23 - 4 = 19$._{: class="legend"}

#### 11.1.1.2. Complexity Analysis

The Prefix Sum Array algorithm's complexity can be analyzed by considering its two main operations: constructing the prefix sum array and performing range sum queries.

In the construction phase, we initialize the prefix sum array $P$ by setting $P[0] = A[0]$, which requires constant time $O(1)$. Then, for each index $i$ from $1$ to $n - 1$, we compute $P[i] = P[i - 1] + A[i]$. This loop runs for $n - 1$ iterations, and each iteration involves a single addition operation, which is a constant-time operation $O(1)$. Therefore, the total time complexity for constructing the prefix sum array is:

$$O(1) + (n - 1) \times O(1) = O(n)$$

Thus, the construction of the prefix sum array has a linear time complexity of $O(n)$. Regarding space complexity, we require an additional array $P$ of size $n$ to store the prefix sums, resulting in an extra space complexity of $O(n)$.

For performing a range sum query to calculate the sum of elements from index $i$ to $j$ in the original array $A$, we utilize the prefix sum array $P$. If $i = 0$, the sum is simply $P[j]$, which is retrieved in constant time $O(1)$. If $i > 0$, the sum is calculated as $P[j] - P[i - 1]$, involving two array accesses and one subtraction, all of which are constant-time operations. Therefore, each range sum query is executed in $O(1)$ time.

The space complexity for executing queries is $O(1)$, as no additional space is required beyond the already constructed prefix sum array.

In conclusion, the Prefix Sum Array algorithm has a time complexity of $O(n)$ for the preprocessing step of constructing the prefix sum array and $O(1)$ time per range sum query. The overall space complexity is $O(n)$ due to the storage of the prefix sum array. This efficiency makes the algorithm particularly useful when dealing with multiple range sum queries on a static array, as it significantly reduces the time complexity per query from $O(n)$ to $O(1)$ after the initial preprocessing.

#### 11.1.1.3. Typical Problem: The Plate Balancer (Problem 2)

In a famous restaurant, Chef André is known for his incredible skill in balancing plates. He has a long table with several plates, each containing a different amount of food. André wants to find the "Magic Plate" - the plate where, when he places his finger underneath it, the weight of the food on the left and right balances perfectly.

Given a list of $plates$, where each number represents the weight of the food on each plate, your task is to help André find the index of the Magic Plate. The Magic Plate is the one where the sum of the weights of all plates to its left is equal to the sum of the weights of all plates to its right.

If André places his finger under the leftmost plate, consider the weight on the left as $0$. The same applies if he chooses the rightmost plate.

Return the leftmost Magic Plate index. If no such plate exists, return $-1$.

**Example 1:**

**Input**: $plates = [3,1,5,2,2]$

**Output**: $2$

**Explanation**:

The Magic Plate is at index $2$.

- Weight on the left = $plates[0] + plates[1] = 3 + 1 = 4$
- Weight on the right = $plates[3] + plates[4] = 2 + 2 = 4$

**Example 2:**

**Input**: $plates = [1,2,3]$

**Output**: $-1$

**Explanation**:

There is no plate that can be the Magic Plate.

**Example 3:**

**Input**: $plates = [2,1,-1]$

**Output**: $0$

**Explanation**:

The Magic Plate is the first plate.

- Weight on the left = $0$ (no plates to the left of the first plate)
- Weight on the right = $plates[1] + plates[2] = 1 + (-1) = 0$

**Constraints:**

$$1 \leq plates.length \leq 10^4$$
$$-1000 \leq plates[i] \leq 1000$$

Note: André is very skilled, so don't worry about the real-world physics of balancing plates. Focus only on the mathematical calculations!

##### 11.1.1.3.A Naïve Solution

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

# Example usage
plates = [3, 1, 5, 2, 2]
result = find_magic_plate_naive(plates)
print(result)  # Should print 2
```

_The following C++20 code implements a naïve solution to the problem of finding the Magic Plate_. It uses a brute-force approach by iterating through each plate and calculating the sum of all plates to its left and right using two separate loops. While this method successfully solves the problem for small input sizes, it lacks efficiency, resulting in a time complexity of $O(n^2)$. This approach is typical of developers who prioritize a working solution over performance optimization, as it recalculates sums repeatedly without leveraging more advanced techniques such as the Prefix Sum Array.

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Function to find the index of the Magic Plate without optimization
int find_magic_plate_naive(const vector<int>& plates) {
    int n = plates.size();

    // Check every plate to see if it's the Magic Plate
    for (int i = 0; i < n; ++i) {
        int left_sum = 0;
        int right_sum = 0;

        // Calculate sum of elements to the left of plate i
        for (int j = 0; j < i; ++j) {
            left_sum += plates[j];
        }

        // Calculate sum of elements to the right of plate i
        for (int j = i + 1; j < n; ++j) {
            right_sum += plates[j];
        }

        // If left and right sums are equal, return the current index
        if (left_sum == right_sum) {
            return i;
        }
    }

    // If no Magic Plate found, return -1
    return -1;
}

int main() {
    // Example 1: plates = [3, 1, 5, 2, 2]
    vector<int> plates1 = { 3, 1, 5, 2, 2 };
    int result1 = find_magic_plate_naive(plates1);
    cout << "Magic Plate index for plates1: " << result1 << endl;

    // Example 2: plates = [1, 2, 3]
    vector<int> plates2 = { 1, 2, 3 };
    int result2 = find_magic_plate_naive(plates2);
    cout << "Magic Plate index for plates2: " << result2 << endl;

    // Example 3: plates = [2, 1, -1]
    vector<int> plates3 = { 2, 1, -1 };
    int result3 = find_magic_plate_naive(plates3);
    cout << "Magic Plate index for plates3: " << result3 << endl;

    return 0;
}
```

The C++20 code implements a solution to the Magic Plate problem by iterating over each plate and calculating the sum of the plates to its left and right. For each plate, two separate loops are used: one for calculating the left sum and another for calculating the right sum. The outer loop runs through all the plates, starting from the first plate to the last, and for each plate, the two sums are calculated to determine if it is the Magic Plate.

The left sum is calculated by iterating from the first plate up to, but not including, the current plate. As the code checks plates further down the list, the left sum loop becomes longer, meaning that plates near the end of the list require more iterations. Similarly, the right sum is calculated by looping through the plates to the right of the current plate. This right sum loop becomes longer for plates near the beginning of the list. The code compares these two sums, and if they are equal, the current plate index is returned as the solution. If no such plate is found, the function returns `-1`.

In terms of complexity, the time required to calculate the left and right sums for each plate depends on the position of the plate in the list. For the $i^{th}$ plate, the left sum takes approximately $O(i)$ iterations, while the right sum takes $O(n-i-1)$ iterations, where $n$ is the total number of plates. Since these calculations are done for every plate, the overall time complexity of the algorithm is $O(n^2)$. The space complexity is $O(1)$ because no additional arrays or data structures are created; the sums are calculated using simple scalar variables.

The following table summarizes the time and space complexities of each step in the algorithm:

| Step                          | Operation                                              | Time Complexity | Space Complexity |
| ----------------------------- | ------------------------------------------------------ | --------------- | ---------------- |
| Left Sum Calculation          | Calculating sum of elements to the left of each plate  | $O(i)$          | $O(1)$           |
| Right Sum Calculation         | Calculating sum of elements to the right of each plate | $O(n-i-1)$      | $O(1)$           |
| Outer Loop (Plates Iteration) | Looping through each plate                             | $O(n)$          | $O(1)$           |
| Overall Complexity            | Total time and space complexities                      | $O(n^2)$        | $O(1)$           |

This approach, while correct, leads to a quadratic time complexity of $O(n^2)$ because it recalculates the sums from scratch for every plate. The space complexity remains constant at $O(1)$, as no extra space is required beyond the scalar variables for sum calculation. Nevertheless, there are better solutions.

##### 11.1.1.3.B Prefix Sum Array Solution

Let's start solving the problem "The Plate Balancer" using the Prefix Sum Array algorithm, using Python to create a pseudocode:

```python
def find_magic_plate(plates):
    n = length(plates)

    # Create prefix sum array
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i-1] + plates[i-1]

    # Calculate total sum
    total_sum = prefix_sum[n]

    # Find magic plate
    for i in range(1, n + 1):
        left_sum = prefix_sum[i-1]
        right_sum = total_sum - prefix_sum[i]

        if left_sum == right_sum:
            return i - 1  # Return 0-based index

    # If no magic plate found
    return -1

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

##### 11.1.1.3.C Competitive Solution

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

### 11.1.2 Histogram

Histograms are sequences of bars with distinct heights anchored to a common baseline. These structures are vital in algorithmic problem-solving, especially in computational geometry and data analysis. Many problems with histograms involve calculating specific metrics about these bars, like finding the largest rectangle that can be formed from consecutive bars.

![]({{ site.baseurl }}/assets/images/histo1.png)
_Image 11.1.3.A - Histogram example_{: class="legend"}

Efficient preprocessing is key to solving histogram problems. A simple approach calculates the area for every possible combination of bars, leading to a time complexity of $O(n^2)$. Advanced methods reduce this to $O(n)$ by using better data structures.

A common technique for solving histogram problems is to manage the sequence of bar heights well. One effective approach is to use a stack to track bar indices. This method makes it easier to decide when to process each bar to maximize its contribution to the area. The stack keeps the order and helps identify which bars to evaluate at each step.

To find the largest rectangle in a histogram, iterate through the bars while maintaining a stack. Push each bar onto the stack if it forms an ascending sequence. When a lower bar appears, pop bars from the stack and calculate areas based on their heights until the current bar can be pushed. This ensures you find the largest possible area for each bar efficiently.

Histogram problems also include challenges like computing the maximum area under a set of heights or finding consecutive bars that meet certain conditions. In each case, careful tracking of bar heights is crucial, which makes preprocessing valuable.

By using stacks and managing each bar's height well, you can reduce time complexity. The goal is to avoid redundant calculations and keep track of potential areas without repeating work. The histogram is a simple yet effective tool for tackling problems with sequences of heights, using efficient algorithms and preprocessing.

#### 11.1.3.1. Typical Problem: Radar Coverage on a Highway

A highway has **N** radar posts positioned along its length. Each radar post has a **detection range** that extends to a certain height from the ground, represented by a sequence `R_i`, where $1 \leq R_i \leq 10^6$ and $1 \leq N \leq 10^6$. The goal is to install the minimum number of **antennas** to cover the detection ranges of all radar posts along the highway.

- Each **antenna** can be installed at a specific height, and it will cover all radar posts whose detection range is equal to or less than the height of the antenna.
- The **antenna moves from left to right**, and when it covers a radar post, its height decreases by one unit (as soon it hits a radar).
- An antenna can only cover a radar post if its current height is equal to the radar's detection range, not greater than or equal to it.
- The task is to determine the **minimum number of antennas** required to cover all radar posts.

- **Input**: The first value is the number of radar posts, followed by their detection ranges in sequence.

- **Output**: The output is the minimum number of antennas required.

Example 1

Input: 6
Detection ranges: 3 1 4 5 1 2

Output: 4

Explanation: The radar posts have varying detection ranges. The first antenna is set at height 3 to cover the first radar. A new antenna is needed for the second radar with a range of 1. The third radar, with a range of 4, requires a new antenna. This antenna also covers the fourth radar (height 5, which decreases). Finally, two more antennas are needed for the last two radars (1 and 2). This results in four antennas being used.

Example 2

Input: 7
Detection ranges: 6 2 5 4 5 1 6

Output: 5

Explanation: The sequence begins with a radar at height 6, requiring an antenna at 6. The second radar, at height 2, needs its own antenna. For the next sequence (5, 4, 5), a new antenna at 5 is used, and it can be decreased to cover the next heights. The sixth radar, at height 1, needs another antenna. The last radar at height 6 also requires a separate antenna, resulting in a total of five antennas.

Example 3

Input: 5
Detection ranges: 4 5 2 1 4

Output: 3

Explanation: The radar posts start with a detection range of 4, requiring an antenna at height 4. The next radar at height 5 requires a new antenna at 5. The third radar, with a detection range of 2, does not match any current antenna, so a new one is needed at 2. The fourth radar, at height 1, can be covered by the existing antenna at 2, reducing its height. The final radar, at height 4, is already covered by the antenna initially set at 5 that has decreased to height 4. Therefore, three antennas are sufficient.

The solution to this problem uses a strategy like histogram problems. We use a tracking mechanism to manage antennas and keep installations minimal. Each antenna covers one radar post and keeps moving, with its height reduced.

An efficient solution uses a stack or a height count array, like in the largest rectangle in a histogram. This ensures we use the antennas well, covering all radar posts with the fewest installations. The goal is to track heights efficiently, like managing bar heights in a histogram, to cut redundancy and cover effectively.

##### Solution Naive

The following code shows a solution from a newbie. 

```cpp

#include <cstdio>
#include <vector>
using namespace std;

int main() {
    int n = 0;          // Variable to store the number of radar posts
    char c;             // Character for reading input

    // Read the number of radar posts using getchar()
    while ((c = getchar()) != '\n' && c != EOF && c != ' ') {
        if (c >= '0' && c <= '9') {
            n = n * 10 + (c - '0');
        }
    }

    vector<int> r;          // Vector to store the detection ranges
    int current_value = 0;  // Variable to build the current detection range value
    bool reading_value = false; // Flag to indicate if we're currently reading a number

    // Read the detection ranges using getchar()
    while (true) {
        c = getchar();
        if (c == EOF || c == '\n') {
            if (reading_value) {
                r.push_back(current_value);
                current_value = 0;
                reading_value = false;
            }
            break;
        }
        else if (c >= '0' && c <= '9') {
            current_value = current_value * 10 + (c - '0');
            reading_value = true;
        }
        else if (c == ' ' || c == ',') {
            if (reading_value) {
                r.push_back(current_value);
                current_value = 0;
                reading_value = false;
            }
        }
    }

    int n_radar = r.size();
    int min_antennas = 0;
    vector<bool> covered(n_radar, false); // Vector to keep track of covered radar posts

    // Loop through each position to place antennas
    for (int i = 0; i < n_radar; ++i) {
        if (!covered[i]) {
            int antenna_height = r[i]; // Install antenna at the radar's detection range
            min_antennas++;            // Increment the number of antennas

            // Antenna moves from position i to the right
            int height = antenna_height;
            for (int j = i; j < n_radar && height > 0; ++j, --height) {
                if (height == r[j]) {
                    covered[j] = true; // Mark radar post as covered
                }
            }
        }
    }

    // Output the minimum number of antennas required
    printf("%d\n", min_antennas);
    return 0;
}
```

This code has a interesting complexity analysis.  

###### Naïve Code Complexity Analysis

The algorithm reads input. It does this in two parts. First, it reads the number of radar posts. This takes $O(K)$ time. $K$ is the number of digits in the count. It is small compared to $N$. Then, it reads the detection ranges. This takes $O(N)$ time. $N$ is the number of radar posts.

The total time for input reading is $O(N)$. This is straightforward.

The main logic has two loops. They work together.

```cpp
for (int i = 0; i < n_radar; ++i)
```

This loop runs $N$ times. It goes through each radar post.

```cpp
for (int j = i; j < n_radar && height > 0; ++j, --height)
```

This loop is more complex. It can run many times or few times. In the worst case, it runs $O(N)$ times for each outer loop. In the best case, it runs $O(1)$ times for each outer loop.

We can look at the total number of times the inner loop runs. Call this number $T$.

$$T = \sum_{k=1}^{A} h_k$$

$A$ is the number of antennas placed. $h_k$ is the height of each antenna.

There is a limit to $T$. It cannot be more than $N$ times $R_{max}$. $R_{max}$ is the largest detection range.

The worst case is $O(N^2)$. This happens when we need many tall antennas.

The best case is $O(N)$. This happens when we need few short antennas.

The average case depends on the input. Without knowing more, we assume the worst case.

We can say $O(N \times R_{max})$ is a tighter bound. But $R_{max}$ could be as big as $N$. So $O(N^2)$ is still correct.

The space used is $O(N)$. This comes from the vectors that store radar ranges and coverage information.

**Conclusion**:

The time complexity is $O(N^2)$ in the worst case. It could be better, depending on the input.
The space complexity is $O(N)$. This is simple and constant.

#### O(n) Solution

```cpp

#include <cstdio>
#include <unordered_map>
using namespace std;

int main() {
    int n = 0;          // Variable to store the number of radar posts
    char c;             // Character for reading input

    // Read the number of radar posts using getchar()
    while ((c = getchar()) != '\n' && c != EOF && c != ' ') {
        if (c >= '0' && c <= '9') {
            // Convert character digit to integer and build the number
            n = n * 10 + (c - '0');
        }
    }

    int r_i;                        // Variable to store each radar's detection range
    unordered_map<int, int> counts; // Map to keep track of antennas at each current height
    int min_antennas = 0;           // Counter for the minimum number of antennas needed
    int current_value = 0;          // Variable to build the current detection range value
    bool reading_value = false;     // Flag to indicate if we're currently reading a number

    // Read and process each radar detection range
    while (true) {
        c = getchar();  // Read the next character from input

        // Check for end of input or end of line
        if (c == EOF || c == '\n') {
            if (reading_value) {
                // Finish reading the current number (detection range)
                r_i = current_value;
                current_value = 0;
                reading_value = false;

                // Process the radar post
                // Check if there is an antenna of height r_i + 1 available
                if (counts[r_i + 1] > 0) {
                    // An antenna of height r_i + 1 exists and can be used
                    // Decrease its count since we're using it
                    counts[r_i + 1]--;
                }
                else {
                    // No antenna can be used to cover this radar
                    // Install a new antenna at height r_i
                    min_antennas++;
                }
                // After covering this radar, the antenna is at height r_i
                // Increase the count of antennas at height r_i
                counts[r_i]++;
            }
            break;  // Exit the loop as we've reached the end of input
        }
        else if (c >= '0' && c <= '9') {
            // If the character is a digit, build the current number
            current_value = current_value * 10 + (c - '0');
            reading_value = true;   // We are currently reading a number
        }
        else if (c == ' ' || c == ',') {
            // If we encounter a space or comma, it signifies the end of a number
            if (reading_value) {
                // Finish reading the current number (detection range)
                r_i = current_value;
                current_value = 0;
                reading_value = false;

                // Process the radar post
                // Check if there is an antenna of height r_i + 1 available
                if (counts[r_i + 1] > 0) {
                    // Use an existing antenna of height r_i + 1
                    counts[r_i + 1]--;
                }
                else {
                    // No existing antenna can cover this radar
                    // Install a new antenna at height r_i
                    min_antennas++;
                }
                // Update the count of antennas at height r_i
                counts[r_i]++;
            }
            // Ignore the space or comma and continue reading
        }
        // Other characters (if any) are ignored
    }

    // Output the minimum number of antennas required
    printf("%d\n", min_antennas);
    return 0;
}
```

###### O(n) Complexity Analisys

The algorithm reads input character by character. It does this once. For each character, it does simple operations. It builds the number of radar posts $n$ and each detection range $r_i$.

The total time for input reading is $O(N)$. $N$ is the number of radar posts.

For each radar post, the algorithm does these things:

1. It checks if an antenna of height $r_i + 1$ exists in the `counts` map. This takes $O(1)$ average time.
2. It changes counts in the `unordered_map`. This also takes $O(1)$ average time.
3. It updates the `min_antennas` counter. This takes $O(1)$ time.

The algorithm processes each radar post once.

The total time is based on going through all radar posts once. Each post involves constant-time operations.

The overall time complexity is $O(N)$. $N$ is the number of radar posts.

The algorithm uses some simple variables. These use constant space.

It also uses an `unordered_map` called `counts`. This map stores antenna counts at different heights.

In the worst case, each radar post has a unique detection range. This could lead to $N$ unique heights in the map.

Each entry in the `unordered_map` uses space for a key and a value. Both are integers.

The total space for the map is $O(N)$ in the worst case.

The space used grows linearly with the number of radar posts. This is because of the `counts` map.

The overall space complexity is $O(N)$. $N$ is the number of radar posts.

**Conclusion**

Time Complexity: $O(N)$
Space Complexity: $O(N)$

The algorithm processes each radar post once. It uses constant-time operations for each post.

The `unordered_map` might store up to $N$ different antenna heights in the worst case. The algorithm scales well for large inputs. It can handle up to $N = 10^6$ or more posts.

Using `unordered_map` is efficient for this problem. If we know the range of detection ranges, we could use a fixed-size array instead. This might be faster, but it would use more space if the range is large.

For example, if $1 \leq R_i \leq 10^6$, we could use a vector of size $10^6$. This would remove the overhead of hashing. But it would use more space, especially if the actual range of detection ranges is small.

**Note**: The `unordered_map` gives average-case constant-time complexity for insertions and lookups. Hash collisions could make this worse. But with a good hash function, the average-case stays $O(1)$.

### 11.1.2. Algorithm: Difference Array - Efficient Range Updates

The Difference Array algorithm is a powerful technique for handling multiple range update operations efficiently. It's particularly useful when you need to perform many updates on an array and only query the final result after all updates are complete. Optimizes range updates to $O(1)$ by storing differences between adjacent elements.

The Difference Array algorithm shines in various scenarios where multiple range updates are required, and the final result needs to be computed only after all updates have been applied. Here are some common applications where this technique proves to be particularly effective:

1. **Range update queries**: When you need to perform multiple range updates and only query the final array state.
2. **Traffic flow analysis**: Modeling entry and exit points of vehicles on a road.
3. **Event scheduling**: Managing overlapping time slots or resources.
4. **Image processing**: Applying filters or adjustments to specific regions of an image.
5. **Time series data**: Efficiently updating ranges in time series data.
6. **Competitive programming**: Solving problems involving multiple range updates.

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

#### 11.1.2.1 Difference Array Algorithm Explained in Plain English

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

##### 11.1.2.1.A Example Problem

Starting with $N(1 \leq N \leq 1,000,000, N \text{ odd})$ empty stacks.
Beatriz receives a sequence of $K$ instructions $(1 \leq K \leq 25,000)$, each in the format "A B", which means that Beatriz should add a new layer of hay to the top of each stack in the interval $A..B$. Calculate the median of the heights after the operations.

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

# Hardcoded inputs
N = 7  # Number of stacks
K = 4  # Number of instructions
instructions = [
    (3, 5),
    (5, 5),
    (2, 4),
    (4, 6)
]

# Initialize difference array
diff = [0] * (N + 1)

# Apply all instructions
for A, B in instructions:
    range_update(diff, A - 1, B - 1, 1)  # -1 for 0-based indexing

# Reconstruct final heights
final_heights = reconstruct_heights(diff)

# Print final heights for verification
print("Final heights:", final_heights[:-1])  # Exclude the last element as it's not part of the original array

# Calculate the median
sorted_heights = sorted(final_heights[:-1])
if N % 2 == 1:
    median = sorted_heights[N // 2]
else:
    median = (sorted_heights[(N - 1) // 2] + sorted_heights[N // 2]) // 2

print("Median height:", median)*Output**: [2, 5, 3, 0]
```

**Algorithm Implementation**: C++20

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// Function to perform range update on the difference array
void range_update(vector<int>& diff, int l, int r, int x) {
    diff[l] += x;
    if (r + 1 < static_cast<int>(diff.size())) {
        diff[r + 1] -= x;
    }
}

// Function to reconstruct the final heights from the difference array
vector<int> reconstruct_heights(const vector<int>& diff) {
    vector<int> heights(diff.size());
    partial_sum(diff.begin(), diff.end(), heights.begin());
    return heights;
}

int main() {
    // Hardcoded inputs based on the example in the image
    int N = 7;  // Number of stacks
    int K = 4;  // Number of instructions
    vector<pair<int, int>> instructions = {
        {3, 5},
        {5, 5},
        {2, 4},
        {4, 6}
    };

    vector<int> diff(N + 1, 0);  // Difference array initialized with 0s

    // Apply all instructions
    for (const auto& [A, B] : instructions) {
        range_update(diff, A - 1, B - 1, 1);  // -1 for 0-based indexing
    }

    vector<int> final_heights = reconstruct_heights(diff);

    // Print final heights for verification
    cout << "Final heights: ";
    for (int height : final_heights) {
        cout << height << " ";
    }
    cout << endl;

    // Sort the heights to find the median
    sort(final_heights.begin(), final_heights.end());

    // Calculate the median
    int median;
    if (N % 2 == 1) {
        median = final_heights[N / 2];
    } else {
        median = (final_heights[(N - 1) / 2] + final_heights[N / 2]) / 2;
    }

    cout << "Median height: " << median << endl;

    return 0;
}
```

#### 11.1.2.2 Complexity Analysis

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

#### 11.1.2.3. Typical Problem: Humidity Levels in a Greenhouse (Problem 1)

This problem is the same as the one described in section 11.1.1.3. To solve it, we need to efficiently compute the sum of even humidity readings after each adjustment without recalculating the entire sum each time. We start by calculating the initial sum $S$ of all even numbers in the $humidity$ array. For each adjustment $[\text{adjustment}, \, \text{sensor\_index}]$, we first retrieve the original value $v = humidity[\text{sensor\_index}]$. If $v$ is even, we subtract it from $S$ because its value will change and it may no longer be even. We then update the humidity reading to $v_{\text{new}} = v + \text{adjustment}$. If $v_{\text{new}}$ is even, we add it to $S$. This way, after each adjustment, $S$ accurately reflects the sum of even humidity readings. By updating $S$ incrementally, we avoid the need to sum over the entire array after each adjustment, thus optimizing the computation.

##### 11.1.2.3.A. Naïve Solution

**Algorithm**:

1. Initialize an empty list called `results` to store the sums of even values after each adjustment.

2. For each adjustment $[adjustment, sensorIndex]$ in the `adjustments` list:

   a. Update the corresponding sensor's value in the `humidity` array:
   $humidity[sensorIndex] = humidity[sensorIndex] + adjustment$

   b. Calculate the sum of even values in the `humidity` array:

   - Initialize a variable `even_sum` to 0
   - For each value $h$ in `humidity`:
     - If $h$ is even (i.e., $h \bmod 2 = 0$), add $h$ to `even_sum`

   c. Append `even_sum` to the `results` list

3. Return the `results` list

**Implementation - Pseudo code**:

```python
def calculate_even_sum_after_adjustments(humidity, adjustments):
    # Inicializa uma lista para armazenar os resultados
    results = []

    # Para cada ajuste na lista de ajustes
    for adjustment, sensor_index in adjustments:
        # Atualiza o valor do sensor correspondente no array de umidade
        humidity[sensor_index] += adjustment

        # Calcula a soma dos valores pares na lista de umidade
        even_sum = 0
        for h in humidity:
            if h % 2 == 0:  # Verifica se o valor é par
                even_sum += h

        # Adiciona a soma atual dos valores pares na lista de resultados
        results.append(even_sum)

    # Retorna a lista de resultados
    return results

# Exemplo de uso:
humidity = [45, 52, 33, 64]
adjustments = [[5, 0], [-20, 1], [-14, 0], [18, 3]]
result = calculate_even_sum_after_adjustments(humidity, adjustments)
print(result)  # Saída: [166, 146, 132, 150]
```

**Implementation - C++ 20**:

```cpp
#include <iostream>  // Includes the library for input and output operations.
#include <vector>    // Includes the library to use vectors.
#include <numeric>   // Includes the library that provides the accumulate function.

using namespace std;

// Function that adjusts the humidity levels and calculates the sum of even values after each adjustment.
vector<long long> adjustHumidity(vector<int>& humidity, const vector<vector<int>>& adjustments) {
    // Creates a vector to store the results, reserving enough space to avoid unnecessary reallocations.
    vector<long long> result;
    result.reserve(adjustments.size());

    // Iterates over each adjustment provided.
    for (const auto& adjustment : adjustments) {
        int value = adjustment[0];  // Extracts the adjustment value.
        int index = adjustment[1];  // Extracts the sensor index to be adjusted.

        // Updates the value in humidity[index] with the adjustment.
        humidity[index] += value;

        // Calculates the sum of even values in the humidity array after the update.
        long long sum = accumulate(humidity.begin(), humidity.end(), 0LL,
            [](long long acc, int val) {
                return acc + (val % 2 == 0 ? val : 0);  // Adds to the sum if the value is even.
            });

        // Adds the current sum of even values to the result vector.
        result.push_back(sum);
    }
    // Returns the vector containing the sum of even values after each adjustment.
    return result;
}

// Helper function to print the results in a formatted way.
void printResult(const vector<int>& humidity, const vector<vector<int>>& adjustments, const vector<long long>& result) {
    // Prints the initial humidity and the adjustments.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");  // Prints each humidity value, separating them with commas.
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        // Prints each adjustment in the form [value, index].
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Prints the result after each adjustment.
    cout << "**Output**: ";
    for (long long res : result) {
        cout << res << " ";  // Prints each result, separating them by spaces.
    }
    cout << "\n\n";
}

int main() {
    // Example 1
    vector<int> humidity1 = { 45, 52, 33, 64 };  // Initial humidity vector.
    vector<vector<int>> adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustment vector.
    cout << "Example 1:\n";
    auto result1 = adjustHumidity(humidity1, adjustments1);  // Calculates the results.
    printResult(humidity1, adjustments1, result1);  // Prints the results.

    // Example 2
    vector<int> humidity2 = { 40 };  // Initial humidity vector.
    vector<vector<int>> adjustments2 = { {12,0} };  // Adjustment vector.
    cout << "Example 2:\n";
    auto result2 = adjustHumidity(humidity2, adjustments2);  // Calculates the results.
    printResult(humidity2, adjustments2, result2);  // Prints the results.

    return 0;  // Indicates that the program terminated successfully.
}
```

The only noteworthy fragment in previous C++ implementation is the lambda function used to calculate the sum in:

```cpp
// Calculates the sum of even values in the humidity array after the update.
        long long sum = accumulate(humidity.begin(), humidity.end(), 0LL,
            [](long long acc, int val) {
                return acc + (val % 2 == 0 ? val : 0);  // Adds to the sum if the value is even.
            });
```

This line calculates the sum of even values in the `humidity` array after the update. The `accumulate` function is used to iterate over the `humidity` array and sum only the even values. The first two parameters, `humidity.begin()` and `humidity.end()`, define the range of elements in the array to be processed. The third parameter, `0LL`, initializes the accumulator with a value of $0$, where `LL` specifies that it is a `long long` integer.

The fourth parameter is a lambda function that takes two arguments: `acc`, which is the accumulated sum so far, and `val`, the current value being processed from the array. Inside the lambda function, the expression `val % 2 == 0 ? val : 0` checks whether the current value `val` is even (i.e., divisible by 2). If `val` is even, it is added to the accumulator `acc`; otherwise, 0 is added, which does not affect the sum.

Thus, the final result of the `accumulate` function is the sum of only the even values in the array, which is then stored in the variable `sum`. Well, something needs a little bit of attention.

> The `<numeric>` library in C++ provides a collection of functions primarily focused on numerical operations. These functions are designed to simplify common tasks such as accumulating sums, performing inner products, calculating partial sums, and more. One of the most commonly used functions in this library is `accumulate`, which is used to compute the sum (or other types of accumulation) of a range of elements in a container.
>
> The general syntax for the `accumulate` function is:
>
> ```cpp
> T accumulate(InputIterator first, InputIterator last, T init);
> T accumulate(InputIterator first, InputIterator last, T init, BinaryOperation op);
> ```
>
> - **InputIterator first, last**: These define the range of elements to be accumulated. The `first` points to the beginning of the range, and `last` points to one past the end of the range.
> - **T init**: This is the initial value of the accumulator, where the result of the accumulation will start.
> - **BinaryOperation op** _(optional)_: This is an optional custom function (usually a lambda or function object) that specifies how two elements are combined during the accumulation. If not provided, the function defaults to using the addition operator (`+`).
>
> **Example 1**: Simple Accumulation (Summing Elements): In its simplest form, `accumulate` can be used to sum all elements in a range.
>
> ```cpp
> #include <numeric>
> #include <vector>
> #include <iostream>
>
> int main() {
>     std::vector<int> vec = {1, 2, 3, 4, 5};
>     int sum = std::accumulate(vec.begin(), vec.end(), 0);  // Sum of all elements
>     std::cout << sum;  // Outputs: 15
>     return 0;
> }
> ```
>
> In this example, `accumulate` is used with the addition operator (default behavior) to sum the elements in the vector.
>
> **Example 2**: Custom Accumulation Using a Lambda Function: A custom operation can be applied during accumulation by providing a binary operation. For instance, to multiply all elements instead of summing them:
>
> ```cpp
> int product = std::accumulate(vec.begin(), vec.end(), 1, [](int acc, int x) {
>     return acc * x;
> });
> std::cout << product;  // Outputs: 120
> ```
>
> Here, instead of summing, the lambda function multiplies the elements.
>
> **Key Features of `accumulate`**:
>
> - **Default behavior**: When no custom operation is provided, `accumulate` simply adds the elements of the range, starting with the initial value.
> - **Custom operations**: By passing a custom binary operation, `accumulate` can perform more complex operations like multiplication, finding the maximum, or applying transformations.
> - **Initial value**: The initial value is critical for defining the result type and the starting point of the accumulation. For instance, starting the accumulation with `0` results in a sum, while starting with `1` can be useful for calculating products.
>
> **Example 3**: Accumulating with Different Types
> `accumulate` can also work with different types by adjusting the initial value and operation. For example, accumulating floating-point values from integers:
>
> ```cpp
> double avg = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
> std::cout << avg;  // Outputs: 3.0
> ```
>
> In this case, starting the accumulation with a double (`0.0`) ensures that the result is a floating-point number.
>
> **Limitations and Considerations**:
>
> - **No built-in parallelism**: The standard `accumulate` function does not support parallel execution, meaning it processes elements sequentially. For parallel processing, alternative solutions like algorithms from the `<execution>` library introduced in C++17 are required.
> - **Performance**: The time complexity of `accumulate` is $O(n)$, as it iterates over each element exactly once, applying the operation specified.
>
> **Example 4**: Custom Accumulation to Filter Elements
> You can use `accumulate` in combination with a lambda to perform conditional accumulation. For example, to sum only even numbers:
>
> ```cpp
> int even_sum = std::accumulate(vec.begin(), vec.end(), 0, [](int acc, int x) {
>     return (x % 2 == 0) ? acc + x : acc;
> });
> std::cout << even_sum;  // Outputs: 6 (2 + 4)
> ```
>
> In this example, only the even numbers are added to the sum by applying a condition within the lambda function.

Finally, we need to clarify lambda functions in C++ 20.

> **Lambda functions** in C++, available since C++ 11, are anonymous functions, meaning they do not have a name like regular functions. These are used when a function is needed only temporarily, typically for short operations, such as inline calculations or callback functions. Lambda functions are defined in place where they are used and can capture variables from their surrounding scope. Lambdas in C++ have been available since C++11, but in C++20, their capabilities were further expanded, making them more powerful and flexible.
>
> The general syntax for a lambda function in C++ is as follows:
>
> ```cpp
> [capture](parameters) -> return_type {
>     // function body
> };
> ```
>
> - **Capture**: Specifies which variables from the surrounding scope can be used inside the lambda. Variables can be captured by value `[=]` or by reference `[&]`. You can also specify individual variables, such as `[x]` or `[&x]`, to capture them by value or reference, respectively.
> - **Parameters**: The input parameters for the lambda function, similar to function arguments.
> - **Return Type**: Optional in most cases, as C++ can infer the return type automatically. However, if the return type is ambiguous or complex, it can be specified explicitly using `-> return_type`.
> - **Body**: The actual code to be executed when the lambda is called.
>
> C++20 brought some new features to lambda functions. One of the most important improvements is the ability to use lambdas in **immediate functions** (with `consteval`), and lambdas can now be default-constructed without capturing any variables. Additionally, lambdas in C++20 can use **template parameters**, allowing them to be more flexible and generic.
>
> **Example 1**: Basic Lambda Function: A simple example of a lambda function that sums two numbers:
>
> ```cpp
> auto sum = [](int a, int b) -> int {
>     return a + b;
> };
> std::cout << sum(5, 3);  // Outputs: 8
> ```
>
> **Example 2**: Lambda with Capture: In this example, a variable from the surrounding scope is captured by value:
>
> ```cpp
> int x = 10;
> auto multiply = [x](int a) {
>     return x * a;
> };
> std::cout << multiply(5);  // Outputs: 50
> ```
>
> Here, the lambda captures `x` by value and uses it in its body.
>
> **Example 3**: Lambda with Capture by Reference: In this case, the variable `y` is captured by reference, allowing the lambda to modify it:
>
> ```cpp
> int y = 20;
> auto increment = [&y]() {
>     y++;
> };
> increment();
> std::cout << y;  // Outputs: 21
> ```
>
> **Example 4**: Generic Lambda Function with C++20: With C++20, lambdas can now use template parameters, making them more generic:
>
> ```cpp
> auto generic_lambda = []<typename T>(T a, T b) {
>     return a + b;
> };
> std::cout << generic_lambda(5, 3);      // Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
> ```
>
> This lambda can add both integers and floating-point numbers by utilizing template parameters.
>
> **Key Improvements in C++20**:
>
> - **Default-constructed lambdas**: In C++20, lambdas that do not capture any variables can now be _default-constructed_. This means they can be created and assigned to a variable without being immediately invoked or fully defined. This allows storing and passing lambdas for later use when default behavior is required.
>
>   ```cpp
>   auto default_lambda = [] {};  // Define a lambda with no capture or parameters
>   default_lambda();             // Call the lambda; valid as of C++20
>   ```
>
>   This feature enables the initialization of lambdas for deferred execution.
>
> - **Immediate lambdas**: C++20 introduces **consteval**, which ensures that functions marked with this keyword are evaluated at compile-time. When used with lambdas, this feature guarantees that the lambda's execution happens during compilation, and the result is already known by the time the program runs. A lambda used within a `consteval` function enforces compile-time evaluation.
>
>   **In programming competitions, `consteval` lambdas are unlikely to be useful because contests focus on runtime performance, and compile-time evaluation does not offer any competitive advantage. Problems in contests rarely benefit from compile-time execution, as the goal is typically to optimize runtime efficiency.**
>
>   **Consteval** ensures that the function cannot be executed at runtime. If a function marked `consteval` is invoked in a context that does not allow compile-time evaluation, it results in a compile-time error.
>
>   Example:
>
>   ```cpp
>   consteval auto square(int x) {
>       return [] (int y) { return y * y; }(x);
>   }
>   int value = square(5);  // Computed at compile-time
>   ```
>
>   In this example, the lambda inside the `square` function is evaluated at compile-time, producing the result before the program starts execution.
>
>   **Since programming contests focus on runtime behavior and dynamic inputs, features like `consteval` are not typically useful. Compile-time operations are not usually required in contests, where inputs are provided after the program has already started executing.**
>
> - **Template lambdas**: C++20 allows lambdas to accept **template parameters**, enabling generic behavior. This feature lets lambdas handle different data types without the need for function overloads or separate template functions. The template parameter is declared directly in the lambda's definition, allowing the same lambda to adapt to various types.
>
>   Example:
>
>   ```cpp
>   auto generic_lambda = []<typename T>(T a, T b) {
>       return a + b;
>   };
>   std::cout << generic_lambda(5, 3);      // Outputs: 8
>   std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
>   ```
>
>   In this case, the lambda can process both integer and floating-point numbers, dynamically adapting to the types of its arguments.

**Data Type Analysis in the `adjustHumidity` Function:**

The choice of `long long` for the return type of the `adjustHumidity` function and for storing intermediate sums is made to ensure safety and prevent overflow in extreme cases:

- **Array size**: The problem specifies that there can be up to $10^4$ elements in the humidity array.
- **Maximum element value**: Each element in the array can have a value of up to $10^4$.
- **Worst-case scenario**: If all elements in the array are even and have the maximum value, the sum would be $10^4 \times 10^4 = 10^8$.
- **`int` limit**: In most implementations, an `int` has 32 bits, with a maximum value of $2^{31} - 1 ≈ 2.15 \times 10^9$.
- **Safety margin**: Although $10^8$ fits within an `int`, it is best practice to leave a safety margin, especially considering there may be multiple adjustments that could further increase the values.
- **`long long` guarantee**: A `long long` is guaranteed to be at least 64 bits, providing a much larger range (up to $2^{63} - 1$ for `signed long long`), which is more than sufficient for this problem.

By using `long long`, we ensure that no overflow occurs, even in extreme or unexpected cases. However, this could potentially lead to higher memory usage, which may exceed the limits in some competitive programming environments, depending on memory constraints.

**Time and Space Complexity Analysis**:

The current implementation recalculates the sum of even numbers in the `humidity` array after each adjustment using the `std::accumulate` function. This results in a time complexity of $O(n \times m)$, where $n$ is the size of the `humidity` array and $m$ is the number of adjustments in the `adjustments` list.

- **Accumulation per adjustment**: For each adjustment, the `std::accumulate` function iterates over all `n` elements in the `humidity` array. This operation takes $O(n)$ time.
- **Total complexity**: Since there are $m$ adjustments, the overall time complexity becomes $O(n \times m)$. This approach is inefficient for large values of $n$ and $m$ (e.g., if both $n$ and $m$ approach $10^4$), leading to performance issues in cases where the number of elements or adjustments is large.

The space complexity is primarily influenced by the size of the input arrays: The `humidity` array contains $n$ elements, each of which is an `int`, so the space required for this array is $O(n)$; The `adjustments` array contains $m$ adjustments, where each adjustment is a pair of integers. Therefore, the space required for this array is $O(m)$. Finally, the `result` vector stores $m$ results, each of type `long long`, so the space required for this vector is $O(m)$. _In total, the space complexity is $O(n + m)$_.

The usage of `long long` ensures that the results and intermediate sums are safe from overflow, but it may slightly increase memory usage compared to using `int`. The overall space requirements are manageable within typical constraints in competitive programming environments, where both $n$ and $m$ are capped at $10^4$.

##### 11.1.2.3.B. Algorithm for a Slightly Less Naive Code

Let's try a slightly less naive solution starting from we saw earlier: Initialize the variable `even_sum` with the value $0$ and create an empty list `results` to store the sums of even values after each adjustment.

Initially, calculate the sum of the even values in the `humidity` array. For each value $h$ in `humidity`, if $h$ is even (i.e., $h \bmod 2 = 0$), add $h$ to `even_sum`.

For each adjustment $[adjustment\_value, sensor\_index]$ in the `adjustments` list, check if the current value in `humidity[sensor\_index]` is even. If it is, subtract it from `even_sum`. Then, update the sensor's value by adding `adjustment\_value` to the existing value:

$$
humidity[sensor\_index] = humidity[sensor\_index] + adjustment\_value
$$

Check if the new value in `humidity[sensor\_index]` is even. If it is, add it to `even_sum`. Add the current value of `even_sum` to the `results` list. Finally, return the `results` list.

**Implementation - C++ 20**:

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Function that adjusts humidity levels and calculates the sum of even values after each adjustment.
vector<long long> adjustHumidity(vector<int>& humidity, const vector<vector<int>>& adjustments) {
    // Initialize the sum of even numbers to zero.
    long long sum = 0;

    // Calculate the initial sum of even values in the humidity array.
    for (int h : humidity) {
        if (h % 2 == 0) {  // Check if the value is even.
            sum += h;  // Add to the sum if it's even.
        }
    }

    // Create a vector to store the results, reserving enough space to avoid unnecessary reallocations.
    vector<long long> result;
    result.reserve(adjustments.size());

    // Iterate through each adjustment provided.
    for (const auto& adjustment : adjustments) {
        int value = adjustment[0];  // Extract the adjustment value.
        int index = adjustment[1];  // Extract the index of the sensor to be adjusted.

        // Check if the current value in humidity[index] is even.
        if (humidity[index] % 2 == 0) {
            sum -= humidity[index];  // If it's even, subtract it from the sum of even numbers.
        }

        // Update the value in humidity[index] with the adjustment.
        humidity[index] += value;

        // Check if the new value in humidity[index] is even after the update.
        if (humidity[index] % 2 == 0) {
            sum += humidity[index];  // If it's even, add it to the sum of even numbers.
        }

        // Add the current sum of even values to the result vector.
        result.push_back(sum);
    }

    // Return the vector containing the sum of even values after each adjustment.
    return result;
}

// Helper function to print the results in a formatted way.
void printResult(const vector<int>& humidity, const vector<vector<int>>& adjustments, const vector<long long>& result) {
    // Print the initial humidity values and the adjustments.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Print the result after each adjustment.
    cout << "**Output**: ";
    for (long long res : result) {
        cout << res << " ";
    }
    cout << "\n\n";
}

int main() {
    // Example 1
    vector<int> humidity1 = { 45, 52, 33, 64 };  // Initial humidity array.
    vector<vector<int>> adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustment array.
    cout << "Example 1:\n";
    auto result1 = adjustHumidity(humidity1, adjustments1);  // Compute the results.
    printResult(humidity1, adjustments1, result1);  // Print the results.

    // Example 2
    vector<int> humidity2 = { 40 };  // Initial humidity array.
    vector<vector<int>> adjustments2 = { {12,0} };  // Adjustment array.
    cout << "Example 2:\n";
    auto result2 = adjustHumidity(humidity2, adjustments2);  // Compute the results.
    printResult(humidity2, adjustments2, result2);  // Print the results.

    return 0;  // Indicate that the program completed successfully.
}
```

This code adjusts the humidity levels in an array and computes the sum of even numbers after each adjustment. It begins by initializing the sum of even numbers from the `humidity` array, adding each even element to a running total. This sum is stored in the variable `sum`, which is later updated based on adjustments made to the `humidity` array.

For each adjustment in the `adjustments` list, the code checks if the value at the target sensor index (i.e., `humidity[index]`) is even. If it is, that value is subtracted from the running total. After updating the sensor's value, the code checks again if the new value is even and adds it to the total if true. This ensures that only even numbers are considered in the running total, which is then stored in a results vector after each adjustment.

Finally, the results vector is returned, which contains the sum of even numbers in the `humidity` array after each adjustment. The `printResult` function is used to display the initial humidity values, the adjustments applied, and the resulting sums in a formatted manner.

> The `auto` keyword in C++ is used to automatically deduce the type of a variable at compile-time. This feature has been available since C++11, but with C++20, its functionality has been improved, allowing for greater flexibility in template functions, lambdas, and other contexts where type inference can simplify code. The `auto` keyword is particularly useful when dealing with complex types, such as iterators, lambdas, or template instantiations, as it reduces the need for explicitly specifying types.
>
> When declaring a variable with `auto`, the type is inferred from the initializer. This eliminates the need to explicitly specify the type, which can be especially useful when working with types that are long or difficult to express.
>
> ```cpp
> auto x = 10;         // x is automatically deduced as an int
> auto y = 3.14;       // y is deduced as a double
> auto str = "Hello";  // str is deduced as a const char*
> ```
>
> In each case, the type of the variable is inferred based on the assigned value. This helps make code more concise and easier to maintain.
>
> **`auto` and Functions**:
>
> In C++20, the `auto` keyword can be used in function return types and parameters. The compiler deduces the return type or parameter type, allowing for greater flexibility in function definitions, especially with lambdas and template functions.
>
> **Example**:
>
> ```cpp
> auto add(auto a, auto b) {
>    return a + b;
> }
>
> int main() {
>    std::cout << add(5, 3);       // Outputs: 8
>    std::cout << add(2.5, 1.5);   // Outputs: 4.0
> }
> ```
>
> In this example, the `add` function can handle both integer and floating-point numbers because the types are deduced automatically. This simplifies function declarations, especially in template-like contexts.
>
> **`auto` with Lambdas and Template Functions**:
>
> C++20 allows for more complex use cases of `auto` within lambdas and template functions. For instance, lambda expressions can use `auto` to deduce parameter types without explicitly specifying them. Additionally, the `auto` keyword can be combined with template parameters to create generic, flexible code.
>
> **Example**:
>
> ```cpp
> auto lambda = [](auto a, auto b) {
>     return a + b;
> };
>
> std::cout << lambda(5, 3);        // Outputs: 8
> std::cout << lambda(2.5, 1.5);    // Outputs: 4.0
> ```
>
> Here, the lambda function uses `auto` to deduce the types of its parameters, making it applicable to both integers and floating-point numbers.

##### A Parallel Competitive Code

Using parallel code in this problem offers a advantage by allowing the calculation of the sum of even humidity values to be distributed across multiple processing threads. This can improve performance, especially for large humidity arrays, as the `reduce` function could leverage parallel execution policies to sum even values concurrently, reducing overall runtime. However, in the current implementation, the sequential execution policy (`exec_seq`) is used to maintain order. Additionally, the Code 3 already employs techniques to reduce verbosity, such as type aliases (`vi`, `vvi`, `vll`) and the use of `auto` for type deduction, making the code cleaner and easier to maintain without sacrificing readability.

In ICPC programming competitions, extremely large input arrays are not typically common, as problems are designed to be solvable within strict time limits, often with manageable input sizes. However, in other competitive programming environments, such as online coding platforms or specific algorithm challenges, larger datasets may appear, requiring more optimized solutions. These scenarios may involve parallel processing techniques or more efficient algorithms to handle the increased computational load. While this problem's input size is moderate, the techniques used here, like reducing verbosity with type aliases and utilizing `reduce`, ensure that the code can scale if needed.

Code 3 is already optimized to minimize function overhead, which can be an important factor in competitive programming. For instance, the entire algorithm is placed inside the `main` function, reducing the need for additional function calls and thus improving performance in time-sensitive environments.

**Code 3**:

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <execution>  // Necessary for execution policies in reduce

using namespace std;

// Aliases to reduce typing of long types
using vi = vector<int>;           // Alias for vector<int>
using vvi = vector<vector<int>>;  // Alias for vector of vectors of int
using vll = vector<long long>;    // Alias for vector<long long>
using exec_seq = execution::sequenced_policy; // Alias for execution::seq (sequential execution)

// Helper function to print the results in a formatted way.
void printResult(const vi& humidity, const vvi& adjustments, const vll& result) {
    // Prints the initial humidity array and the adjustments array.
    cout << "**Input**: humidity = [";
    for (size_t i = 0; i < humidity.size(); ++i) {
        // Print each humidity value, separating them with commas.
        cout << humidity[i] << (i < humidity.size() - 1 ? ", " : "");
    }
    cout << "], adjustments = [";
    for (size_t i = 0; i < adjustments.size(); ++i) {
        // Print each adjustment as [value, index], separating them with commas.
        cout << "[" << adjustments[i][0] << "," << adjustments[i][1] << "]" << (i < adjustments.size() - 1 ? ", " : "");
    }
    cout << "]\n";

    // Prints the results after each adjustment.
    cout << "**Output**: ";
    for (auto res : result) {  // Using `auto` to automatically deduce the type (long long)
        cout << res << " ";    // Print each result followed by a space.
    }
    cout << "\n\n";
}

int main() {
    // Example 1: Initialize the humidity vector and the adjustments to be made.
    vi humidity1 = { 45, 52, 33, 64 };  // Initial humidity levels for each sensor.
    vvi adjustments1 = { {5,0}, {-20,1}, {-14,0}, {18,3} };  // Adjustments in format {adjustment value, sensor index}.

    // Create a vector to store the results, reserving space to avoid reallocation during execution.
    vll result1;
    result1.reserve(adjustments1.size());

    // Process each adjustment for the humidity array.
    for (const auto& adjustment : adjustments1) {
        int value = adjustment[0];  // Get the adjustment value.
        int index = adjustment[1];  // Get the index of the sensor to be adjusted.

        // Apply the adjustment to the corresponding humidity value.
        humidity1[index] += value;

        // Calculate the sum of even values in the humidity array using the `reduce` function.
        auto sum = reduce(
            exec_seq{},              // Use sequential execution policy to maintain order.
            humidity1.begin(),       // Start iterator of the humidity vector.
            humidity1.end(),         // End iterator of the humidity vector.
            0LL,                     // Initial sum is 0 (as long long to avoid overflow).
            [](auto acc, auto val) { // Lambda function to accumulate even numbers.
                return acc + (val % 2 == 0 ? val : 0);  // Add to the sum only if the value is even.
            }
        );

        // Store the current sum of even values after the adjustment in the result vector.
        result1.push_back(sum);
    }

    // Print the results for the first example.
    cout << "Example 1:\n";
    printResult(humidity1, adjustments1, result1);

    // Example 2: Initialize the second humidity vector and the adjustments.
    vi humidity2 = { 40 };  // Initial humidity levels for the second example.
    vvi adjustments2 = { {12,0} };  // Adjustments for the second example.

    // Create a vector to store the results.
    vll result2;
    result2.reserve(adjustments2.size());

    // Process each adjustment for the second humidity array.
    for (const auto& adjustment : adjustments2) {
        int value = adjustment[0];  // Get the adjustment value.
        int index = adjustment[1];  // Get the index of the sensor to be adjusted.

        // Apply the adjustment to the corresponding humidity value.
        humidity2[index] += value;

        // Calculate the sum of even values in the humidity array using `reduce`.
        auto sum = reduce(
            exec_seq{},              // Use sequential execution policy to maintain order.
            humidity2.begin(),       // Start iterator of the humidity vector.
            humidity2.end(),         // End iterator of the humidity vector.
            0LL,                     // Initial sum is 0 (as long long to avoid overflow).
            [](auto acc, auto val) { // Lambda function to accumulate even numbers.
                return acc + (val % 2 == 0 ? val : 0);  // Add to the sum only if the value is even.
            }
        );

        // Store the current sum of even values after the adjustment in the result vector.
        result2.push_back(sum);
    }

    // Print the results for the second example.
    cout << "Example 2:\n";
    printResult(humidity2, adjustments2, result2);

    return 0;  // Indicate that the program finished successfully.
}
```

The core of the algorithm in Code 3 focuses on adjusting humidity levels based on a series of adjustments and then calculating the sum of even humidity values after each adjustment. The main part responsible for solving the problem involves iterating over each adjustment and performing two key operations: updating the humidity values and calculating the sum of even numbers in the updated array. This is done by:

1. **Adjusting the Humidity**: For each adjustment (which consists of an adjustment value and an index), the corresponding humidity value is updated by adding the adjustment value. This modifies the sensor reading at the specified index in the `humidity` vector.

   Example:

   ```cpp
   humidity[index] += value;
   ```

   This line updates the humidity value at the sensor located at `index` by adding the provided `value`.

2. **Calculating the Sum of Even Values**: After each adjustment, the algorithm calculates the sum of the even values in the `humidity` array. This is done using the `reduce` function with a lambda function that filters and sums only the even numbers. The key here is that the algorithm iterates over the entire `humidity` array and sums the values that are divisible by 2.

   Example:

   ```cpp
   auto sum = reduce(
       exec_seq{},              // Sequential execution
       humidity.begin(),        // Start of the humidity array
       humidity.end(),          // End of the humidity array
       0LL,                     // Initial sum set to 0 (long long)
       [](auto acc, auto val) { // Lambda to sum even values
           return acc + (val % 2 == 0 ? val : 0);
       }
   );
   ```

   This code calculates the sum of all even values in the `humidity` array after each adjustment, ensuring that only even numbers contribute to the total sum.

3. **Storing and Printing Results**: After calculating the sum of even values for each adjustment, the result is stored in a `result` vector, which is later printed to display the output. The `printResult` function is used to format and output the humidity values, adjustments, and the resulting sum of even values after each adjustment.

In this context, the parallel version of `reduce` is particularly useful when dealing with large datasets, where summing or reducing values sequentially can be time-consuming. The key advantage of using `reduce` with a parallel execution policy is its ability to distribute the workload across multiple cores, significantly reducing the overall execution time.

When `reduce` is used with the `execution::par` policy, it breaks the range of elements into smaller chunks and processes them in parallel. This means that instead of iterating through the array in a single thread (as done with `execution::seq`), the work is split among multiple threads, each of which processes a part of the array concurrently.

**Parallel Execution Example**:

In the following example, the `reduce` function is used to sum an array of humidity values, utilizing the `execution::par` policy:

```cpp
auto parallel_sum = std::reduce(std::execution::par, humidity.begin(), humidity.end(), 0LL,
                                [](auto acc, auto val) {
                                    return acc + (val % 2 == 0 ? val : 0);  // Sum only even values
                                });
```

**How the parallel execution works**:

1. **Data Splitting**: The `humidity` array is divided into smaller chunks, and each chunk is processed by a separate thread.
2. **Concurrent Processing**: Each thread sums the even values in its respective chunk. The `execution::par` policy ensures that this happens in parallel, taking advantage of multiple CPU cores.
3. **Final Reduction**: Once all threads complete their tasks, the partial results are combined into a final sum, which includes only the even values from the original array.

By distributing the workload across multiple threads, the program can achieve significant performance improvements when the `humidity` array is large. This approach is particularly useful in competitive programming contexts where optimizing time complexity for large inputs can be crucial to solving problems within strict time limits.

> The `reduce` function, introduced in C++17, is part of the `<numeric>` library and provides a way to aggregate values in a range by applying a binary operation, similar to `accumulate`. However, unlike `accumulate`, `reduce` can take advantage of parallel execution policies, making it more efficient for large data sets when concurrency is allowed. In C++20, `reduce` gained even more flexibility, making it a preferred choice for operations that benefit from parallelism.
>
> **Basic Syntax of `reduce`**:
>
> The general syntax for `reduce` is as follows:
>
> ```cpp
> T reduce(ExecutionPolicy policy, InputIterator first, InputIterator last, T init);
> T reduce(ExecutionPolicy policy, InputIterator first, InputIterator last, T init, BinaryOperation binary_op);
> ```
>
> - **ExecutionPolicy**: This specifies the execution policy, which can be `execution::seq` (sequential execution), `execution::par` (parallel execution), or `execution::par_unseq` (parallel and vectorized execution).
> - **InputIterator first, last**: These define the range of elements to be reduced.
> - **T init**: The initial value for the reduction (e.g., 0 for summing values).
> - **BinaryOperation binary_op** (optional): A custom operation to apply instead of the default addition.
>
> **Example 1**: Basic Reduce with Sequential Execution: This example demonstrates a basic sum reduction with sequential execution:
>
> ```cpp
> #include <iostream>
> #include <vector>
> #include <numeric>
> #include <execution>  // Required for execution policies
>
> int main() {
> std::vector<int> vec = {1, 2, 3, 4, 5};
> auto sum = std::reduce(std::execution::seq, vec.begin(), vec.end(), 0);
> std::cout << "Sum: " << sum; // Outputs: 15
> return 0;
> }
> ```
>
> Here, the `reduce` function uses the `execution::seq` policy to ensure that the >reduction happens in a sequential order, summing the values from `vec`.
>
> **Example 2**: Custom Binary Operation: You can also provide a custom binary operation using a lambda function. In this case, the reduction will multiply the elements instead of summing them:
>
> ```cpp
> auto product = std::reduce(std::execution::seq, vec.begin(), vec.end(), 1,
>                            [](int a, int b) { return a * b; });
> std::cout << "Product: " << product;  // Outputs: 120
> ```
>
> In this example, `reduce` applies the custom binary operation (multiplication) to aggregate the values in `vec`.
>
> **Parallelism in `reduce`**:
>
> The major advantage of `reduce` over `accumulate` is its ability to handle parallel execution. Using the `execution::par` policy allows `reduce` to split the workload across multiple threads, significantly improving performance on large datasets:
>
> ```cpp
> auto parallel_sum = std::reduce(std::execution::par, vec.begin(), vec.end(), 0);
> ```
>
> This enables `reduce` to sum the elements in `vec` concurrently, improving efficiency on large arrays, especially in multi-core environments.

### 11.1.3. Algorithm: Incremental Sum

The **Incremental Sum Algorithm** offers an efficient method for maintaining a running sum of specific elements (such as even numbers) in an array while applying adjustments. This approach eliminates the need to recalculate the entire sum after each modification, instead updating the sum incrementally by subtracting old values and adding new ones as necessary.

The algorithm begins with an initial calculation of the sum of even numbers in the array. This step has a time complexity of $O(n)$, where $n$ represents the array size. For example, in Python, this initial calculation could be implemented as:

```python
def initial_sum(arr):
    return sum(x for x in arr if x % 2 == 0)
```

Following the initial calculation, the algorithm processes each adjustment to the array. For each adjustment, it performs three key operations: If the old value at the adjusted index was even, it subtracts this value from the sum. It then updates the array element with the new value. Finally, if the new value is even, it adds this value to the sum. This process maintains the sum's accuracy with a constant time complexity of $O(1)$ per adjustment. In C++, this adjustment process could be implemented as follows:

```cpp
void adjust(vector<int>& arr, int index, int new_value, int& even_sum) {
    if (arr[index] % 2 == 0) even_sum -= arr[index];
    arr[index] = new_value;
    if (new_value % 2 == 0) even_sum += new_value;
}
```

The algorithm's efficiency stems from its ability to process adjustments in constant time, regardless of the array's size. This approach is particularly beneficial when dealing with numerous adjustments, as it eliminates the need for repeated full array traversals.

To illustrate the algorithm's operation, consider the following example:

```python
arr = [1, 2, 3, 4, 5]
even_sum = initial_sum(arr)  # even_sum = 6 (2 + 4)

# Adjustment 1: Change arr[0] from 1 to 6
adjust(arr, 0, 6, even_sum)  # even_sum = 12 (6 + 2 + 4)

# Adjustment 2: Change arr[1] from 2 to 3
adjust(arr, 1, 3, even_sum)  # even_sum = 10 (6 + 4)
```

Let's try to look at it from another perspective:

- Let $n$ be the size of the array $A$.
- Let $Q$ be the number of queries (adjustments).
- Let $A[i]$ be the value at index $i$ in the array.
- Let $adjustments[k] = [val_k, index_k]$ represent the adjustment in the $k$-th query, where $val_k$ is the adjustment value and $index_k$ is the index to be adjusted.

Our goal is to calculate the sum of the even numbers in $A$ incrementally after each adjustment, without recalculating the entire sum from scratch after each query.

**Step 1: Initial Calculation of the Sum of Even Numbers**:

First, define $S$ as the initial sum of even numbers in the array $A$. This sum can be expressed as:

$$S = \sum_{i=0}^{n-1} \text{if } (A[i] \% 2 == 0) \text{ then } A[i]$$

The conditional function indicates that only even values are summed.

**Step 2: Incremental Update**:

When we receive a query $adjustments[k] = [val_k, index_k]$, we adjust the value at index $index_k$ by adding $val_k$ to the current value of $A[index_k]$. The new value is:

$$\text{new\_value} = A[index_k] + val_k$$

We update the sum $S$ efficiently as follows:

1. If the original value $A[index_k]$ was **even**, we subtract it from $S$:

   $$S = S - A[index_k]$$

2. After applying the adjustment, if the new value $\text{new\_value}$ is **even**, we add it to $S$:

   $$S = S + \text{new\_value}$$

**Formal Analysis of Updates**:

For each adjustment, we have the following operations:

- **Remove the old value (if even):**
  If $A[index_k]$ is even before the adjustment:

  $$S = S - A[index_k]$$

- **Add the new value (if even):**
  If $\text{new\_value}$ is even after the adjustment:

  $$S = S + \text{new_value}$$

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

#### 11.1.3.1. Incremental Sum Algorithm Explained in Plain English

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

1. **Old Value**: `A[0] = 1` (odd)

   - Since it's odd, it doesn't affect the sum.

2. **Update Element**:

   - `A[0] = 1 + 5 = 6`

3. **New Value**: `A[0] = 6` (even)
   - Add the new value to the sum: Sum = 6 + 6 = **12**

Adjustment 2: Change `A[1]` from $2$ to $3$

1. **Old Value**: `A[1] = 2` (even)

   - Subtract the old value from the sum: Sum = 12 - 2 = **10**

2. **Update Element**:

   - `A[1] = 2 + 1 = 3`

3. **New Value**: `A[1] = 3` (odd)
   - Since it's odd, the sum remains unchanged.

Adjustment 3: Change `A[2]` from $3$ to $2$

1. **Old Value**: `A[2] = 3` (odd)

   - Doesn't affect the sum.

2. **Update Element**:

   - `A[2] = 3 - 1 = 2`

3. **New Value**: `A[2] = 2` (even)
   - Add the new value to the sum: Sum = 10 + 2 = **12**

#### 11.1.3.2 Complexity Analysis

The algorithm's overall time complexity can be expressed as $O(n + m)$, where $n$ is the initial array size and $m$ is the number of adjustments. This represents a significant improvement over the naive approach of recalculating the sum after each adjustment, which would result in a time complexity of $O(n \times m)$.

In scenarios involving large arrays with frequent updates, the Incremental Sum Algorithm offers substantial performance benefits. It proves particularly useful in real-time data processing, financial calculations, and various computational problems where maintaining a running sum is crucial. By avoiding redundant calculations, it not only improves execution speed but also reduces computational resource usage, making it an invaluable tool for efficient array manipulation and sum maintenance in a wide range of applications.

#### 11.1.4.3. Typical Problem: "Humidity Levels in a Greenhouse" (Problem 1)

The same problem we saw earlier in the section: 11.1.1.3. Below is the implementation of Difference Array Algorithm in C++20:

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

### 11.1.5. Static Array Queries

**This is a work in progress, we will get there sooner or later.**

Techniques for arrays that don't change between queries, allowing efficient pre-calculations.

- Algorithm: Sparse Table

- Problem Example: "Inventory Restocking" - Performs queries after each inventory adjustment

### 11.1.7. Fenwick Tree

**This is a work in progress, we will get there sooner or later.**

Data structure for prefix sums and efficient updates, with operations in $O(\log n)$.

- Algorithm: Binary Indexed Tree (BIT)

##### Finally, the code using Fenwick tree

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
>  constexpr int size = 5;
>  int array[size];  // The size is computed at compile time.
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

## 11.2. Sliding Window Algorithms

Sliding window algorithms handle data sequences by moving a window over them. The window captures a subset of elements, and we perform computations on this subset.

### Sliding Window Fixed Sum

This problem involves finding a fixed-size window (of size $k$) where the sum of elements equals a target value. We compute the sum of elements within a fixed-size window as it moves along the sequence. For an array $A$ and window size $k$, the sum at position $i$ is:

$$
\text{Sum}_i = \sum_{j=i}^{i+k-1} A_j
$$

To calculate sums efficiently, we update the sum by subtracting the element leaving the window and adding the new element entering.

```python
window_sum = sum(A[:k])
result = [window_sum]

for i in range(k, len(A)):
    window_sum += A[i] - A[i - k]
    result.append(window_sum)
```

In C++20 we could have:

```cpp
#include <vector>

std::vector<int> sliding_window_fixed_sum(const std::vector<int>& A, int k) {
    std::vector<int> result;
    int window_sum = 0;

    for (int i = 0; i < k; ++i)
        window_sum += A[i];
    result.push_back(window_sum);

    for (size_t i = k; i < A.size(); ++i) {
        window_sum += A[i] - A[i - k];
        result.push_back(window_sum);
    }
    return result;
}
```

#### Typical Problem

Given a 1-indexed array of integers `numbers` that is already sorted in non-decreasing order, find a contiguous subarray of length $k$ such that the sum of its elements is equal to a specific target number. 

Let the subarray be `numbers[index1, index1+1, ..., index1+k-1]` where $1 \leq \text{index1} \leq \text{numbers.length} - k + 1$.

Return the starting index of the subarray, `index1`, added by one. If no such subarray exists, return $-1$.

The solution must use only constant extra space.

**Input Format**:

- The first line contains a single integer $n$ ($2 \leq n \leq 3 \times 10^4$), the length of the array.
- The second line contains $n$ space-separated integers $numbers_i$ ($-1000 \leq numbers_i \leq 1000$), representing the array elements.
- The third line contains two space-separated integers $k$ and $target$ ($1 \leq k \leq n$, $-10^6 \leq target \leq 10^6$), where $k$ is the window size and $target$ is the sum to find.

**Example 1**:
Input:
5
1 2 3 4 5
3 9

Output: 3
Explanation: The subarray [2, 3, 4] sums to 9. Therefore, index1 = 2. We return 2 + 1 = 3.

**Example 2**:
Input:
4
2 4 6 8
2 10

Output: 3
Explanation: The subarray [4, 6] sums to 10. Therefore index1 = 2. We return 2 + 1 = 3.

**Example 3**:
Input:
3
1 2 3
3 10

Output: -1
Explanation: There is no subarray of length 3 that sums to 10.

**Example 4**:
Input:
10
1 3 2 5 4 6 8 7 9 10
4 20

Output: 5
Explanation: The subarray [4, 6, 8, 7] sums to 20. Therefore, index1 = 4. We return 4 + 1 = 5.

**Constraints**:

- $2 \leq \text{numbers.length} \leq 3 \times 10^4$
- $1 \leq k \leq \text{numbers.length}$
- $-1000 \leq \text{numbers}[i] \leq 1000$
- `numbers` is sorted in non-decreasing order.
- $-10^6 \leq \text{target} \leq 10^6$

### 11.2.1. Sliding Window Minimum

We find the minimum value within each window. Using a deque, we maintain candidates for the minimum value.

```python
from collections import deque

def sliding_window_minimum(A, k):
    q = deque()
    result = []

    for i in range(len(A)):
        while q and q[-1] > A[i]:
            q.pop()
        q.append(A[i])
        if i >= k and q[0] == A[i - k]:
            q.popleft()
        if i >= k - 1:
            result.append(q[0])
    return result
```

Or

```cpp
#include <vector>
#include <deque>

std::vector<int> sliding_window_minimum(const std::vector<int>& A, int k) {
    std::deque<int> q;
    std::vector<int> result;

    for (size_t i = 0; i < A.size(); ++i) {
        while (!q.empty() && q.back() > A[i])
            q.pop_back();
        q.push_back(A[i]);
        if (i >= k && q.front() == A[i - k])
            q.pop_front();
        if (i >= k - 1)
            result.push_back(q.front());
    }
    return result;
}
```

>The `std::deque` class has been part of C++ since the standardization in 1998. It provides a double-ended queue, allowing fast insertions and deletions at both the front and back of the sequence. You can access any element by its index in constant time.
>
>Consider adding an element to the end:
>
> ```cpp
> #include <deque>
> 
> std::deque<int> dq;
> dq.push_back(10); // The deque now contains: 10
> ```
>
>You can also add an element to the front:
>
> ```cpp
> dq.push_front(20); // The deque now contains: 20, 10
> ```
>
> Removing elements works the same way. Remove from the back:
>
> ```cpp
> dq.pop_back(); // The deque now contains: 20
> ```
>
> Or remove from the front:
> 
> ```cpp
> dq.pop_front(); // The deque is now empty
> ```
>
> Access elements by index:
>
> ```cpp
> dq.push_back(30);
> dq.push_back(40); // The deque now contains: 30, 40
> int value = dq[1]; // value is 40
> ```
>
> To find out how many elements are in the deque:
>
> ```cpp
> size_t size = dq.size(); // size is 2
> ```
>
> Here is how you might use `std::deque` in a sliding window maximum problem:
>
> ```cpp
> #include <iostream>
> #include <deque>
> #include <vector>
>
> std::vector<int> sliding_window_maximum(const std::vector<int>& nums, int k) {
>     std::deque<int> dq;
>     std::vector<int> result;
>
>     for (size_t i = 0; i < nums.size(); ++i) {
>         if (!dq.empty() && dq.front() == i - k)
>             dq.pop_front();
>
>         while (!dq.empty() && nums[dq.back()] < nums[i])
>             dq.pop_back();
>
>         dq.push_back(i);
>
>         if (i >= k - 1)
>            result.push_back(nums[dq.front()]);
>    }
>    return result;
> }
>
> int main() {
>     std::vector<int> nums = {1, 3, 2, 5, 4, 6};
>     int k = 3;
>     std::vector<int> max_values = sliding_window_maximum(nums, k);
>
>     for (int val : max_values)
>         std::cout << val << " "; // Outputs: 3 5 5 6
>
>     return 0;
>}
> ```
>
>In this example, we keep track of indices in the deque. We remove indices that are out of the current window or whose values are less than the current number. The front of the deque always holds the index of the maximum value in the current window.
>
>The `std::deque` provides constant-time access to elements using the subscript operator. **Inserting or deleting elements at either end takes constant time as well. However, inserting or deleting in the middle takes linear time**. In general we have:
>
> - Insertion/deletion at ends: $O(1)$
> - Insertion/deletion in middle: $O(n)$
> - Random access: $O(1)$
>
>Check if the deque is empty:
>
> ```cpp
> if (dq.empty()) {
>     // The deque is empty
> }
> ```
>
> Remove all elements:
>
> ```cpp
> dq.clear(); // The deque is now empty
> ```
>
> Iterate over the deque:
>
> ```cpp
> for (auto it = dq.begin(); it != dq.end(); ++it) {
>    std::cout << *it << " ";
>}
> ```
>
> If you need to insert an element at a specific position:
>
> ```cpp
> dq.insert(dq.begin() + 1, 25); // Inserts 25 at index 1
> ```
>
>Or erase an element:
>
> ```cpp
> dq.erase(dq.begin()); // Removes the first element
> ```
>
>The `std::deque` is useful when you need a dynamic array with fast insertions and deletions at both ends. It is implemented as a sequence of contiguous memory blocks. This allows it to grow in both directions without reallocating the entire container.
>
>**Comparison with Other Structures**
>
>The `std::deque` offers a unique balance between `std::vector` and `std::list`, combining advantages of both.
>
>Unlike `std::vector`, `std::deque` allows efficient insertions at the beginning in $O(1)$ time, which is particularly useful for implementing queues or for algorithms that frequently add elements to the front of the container. Similar to `std::vector` but unlike `std::list`, `std::deque` offers random access to elements in $O(1)$ time. This means you can quickly access any element by its index, making it suitable for scenarios where both fast insertion/deletion at ends and quick random access are required. `std::deque` doesn't require contiguous memory like `std::vector`, which means it can handle larger datasets without the need for expensive reallocation operations when it grows. However, `std::deque` may have slightly higher memory overhead compared to `std::vector` due to its more complex internal structure. 
>
>Unlike `std::list`, which excels at insertions and deletions anywhere in the container, `std::deque` is optimized for operations at both ends, making it an excellent choice for double-ended queues or sliding window algorithms. This balance of features makes `std::deque` a versatile container, often providing a good compromise when neither `std::vector` nor `std::list` is ideal for a particular use case.
>
>`std::deque` uses a structure of memory blocks:
>
> ```cpp
> [Block1][Block2][Block3]...[BlockN]
> ```
>
>This structure allows efficient growth in both directions without frequent reallocations. However, `std::deque` may use more memory than `std::vector` due to its internal structure:
>
> $$ \text{Memory} \approx n \cdot \text{sizeof}(T) + \text>{overhead}$$
>
>Where $n$ is the number of elements and $T$ is the element type.
>
>**Finally, `std::deque` iterators are random access, but can be invalidated: In a std::deque, insertions or deletions at any position can invalidate iterators. Specifically, insertions or deletions at the ends invalidate all iterators but do not affect references or pointers to existing elements. Insertions or deletions in the middle of the deque invalidate all iterators, references, and pointers to elements.**.
>
>
>Beyond those methods already mentioned, `std::deque` offers:
>
>- `at(n)`: Access with bounds checking.
>- `front()` and `back()`: Access to first and last elements.
>- `emplace_front()` and `emplace_back()`: In-place construction.
>
> `emplace_front()` and `emplace_back()` are methods for efficient element Construction.
>
>`emplace_front()` and `emplace_back()` build new elements directly in the deque. They don't create temporary objects. This saves time and memory. You pass the constructor arguments directly to these methods. The deque then makes the new element in place. 
>
>For types with complex constructors, this is faster than `push_front()` or `push_back()`. Those methods would create a temporary object first. With `emplace_front()`, the new element goes at the start of the deque. With `emplace_back()`, it goes at the end. Use these when you need to add elements often and want the best performance. They work well with objects that are expensive to copy or move.
>
> The std::deque is compatible with most STL algorithms::
>
> ```cpp
> #include <algorithm>
> #include <deque>
> 
> std::deque<int> dq = {3, 1, 4, 1, 5, 9};
> std::sort(dq.begin(), dq.end());
> ```
>
>Like most STL containers, `std::deque` is not thread-safe by default. For concurrent access, consider using mutex:
>
> ```cpp
> #include <mutex>
> #include <deque>
> 
> std::mutex mtx;
> std::deque<int> shared_deque;
> 
> // In a thread:
> {
>     std::lock_guard<std::mutex> lock(mtx);
>     shared_deque.push_back(42);
> }
> ```
>
>C++20 introduces `std::erase` and `std::erase_if` for `std::deque`:
>
> ```cpp
> #include <deque>
> #include <algorithm>
> 
> std::deque<int> dq = {1, 2, 3, 2, 4, 2};
> std::erase(dq, 2);  // Removes all 2s
> ```
>
> This addition simplifies element removal based on value or predicate.
>
>C++20's `std::erase` and `std::erase_if` for `std::deque` are useful but come with costs. The time complexity is linear, $O(n)$, where $n$ is the number of elements in the deque. These functions must check each element and may move elements to fill gaps. For `std::erase(dq, value)`, it checks every element, always taking $O(n)$ time. It moves elements to close gaps, which can take up to $O(n)$ time. In the worst case, it removes all elements, taking $O(n)$ time. `std::erase_if(dq, pred)` behaves similarly. It checks every element with the predicate, always taking $O(n)$ time. Moving elements to close gaps can take up to $O(n)$ time. The worst case is the same as `std::erase`. The space complexity for both is $O(1)$. These functions don't use extra storage that grows with input size. Here's an example:
>
> ```cpp
> std::deque<int> dq = {1, 2, 3, 2, 4, 2};
> std::erase(dq, 2);  // This operation is O(n)
> ```
>
>These functions are simple to use. But for large deques or frequent use, consider the performance impact. In some cases, manual element removal might be faster. It depends on your specific needs.
>

### 11.2.2. Sliding Window Maximum

This method tracks the maximum value in each window. It uses a similar approach to the minimum.

```python
from collections import deque

def sliding_window_maximum(A, k):
    q = deque()
    result = []

    for i in range(len(A)):
        while q and q[-1] < A[i]:
            q.pop()
        q.append(A[i])
        if i >= k and q[0] == A[i - k]:
            q.popleft()
        if i >= k - 1:
            result.append(q[0])
    return result
```

Or, in C++20:

```cpp
#include <vector>
#include <deque>

std::vector<int> sliding_window_maximum(const std::vector<int>& A, int k) {
    std::deque<int> q;
    std::vector<int> result;

    for (size_t i = 0; i < A.size(); ++i) {
        while (!q.empty() && q.back() < A[i])
            q.pop_back();
        q.push_back(A[i]);
        if (i >= k && q.front() == A[i - k])
            q.pop_front();
        if (i >= k - 1)
            result.push_back(q.front());
    }
    return result;
}
```

## 11.3. Multiple Query Processing

**This is a work in progress, we will get there sooner or later.**

Methods for handling multiple queries efficiently.

### 11.3.1 Mo's Algorithm

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

#### 11.3.1.1 Why Choose $\sqrt{n}$ as the Block Size?

The choice of $\sqrt{n}$ as the block size is crucial for the algorithm's efficiency. Here's why:

- The number of blocks becomes $\sqrt{n}$.
- The number of times we change the left boundary is $O(\sqrt{n})$.
- The total number of add/remove operations is $O(n \sqrt{n})$.

This choice balances the work done when moving between blocks and within blocks, optimizing overall performance.

#### 11.3.1.2. Complexity Analysis

**Time Complexity Analysis**:

The total time complexity of Mo's Algorithm is:

$$O\left( q \times \frac{n}{\sqrt{n}} + n \sqrt{n} \right) = O(n \sqrt{n})$$

To understand this, sorting the queries takes $O(q \log q)$ time. Given that $q$ is generally $O(n)$, this remains efficient. The adjustment of segment boundaries between queries takes $O(n \sqrt{n})$ time, which contributes to the overall complexity. When compared to the naive approach with a time complexity of $O(n \times q)$, Mo's Algorithm provides a marked improvement, particularly when dealing with larger datasets.

**Space Complexity Analysis**:

The space complexity of Mo's Algorithm is:

$$O(n + q)$$

To understand the space complexity, we need $O(n)$ space to store the array elements and $O(q)$ space to store the queries. As a result, the overall space usage is linear, ensuring that the algorithm remains efficient even for large datasets.

#### 11.3.1.3 Implementation

Let's see how to implement Mo's Algorithm in Python and C++20. These implementations assume we're calculating the sum over intervals, but the concept can be adapted for other types of queries.

**Python Pseudocode**:

```python
import math

# Function to process the queries using Mo's Algorithm
def mo_algorithm(arr, queries):
    n = len(arr)  # Length of the input array
    q = len(queries)  # Number of queries
    sqrt_n = int(math.sqrt(n))  # Square root of n, used for block size

    # Result array to store the answers to the queries
    result = [0] * q

    # Frequency array to keep track of the frequency of elements in the current range
    freq = [0] * (max(arr) + 1)

    # Sorting the queries using Mo's Algorithm
    queries.sort(key=lambda x: (x[0] // sqrt_n, x[1]))  # Sort by block and then by R value

    currL, currR = 0, 0  # Initialize current left and right pointers
    curr_sum = 0  # Variable to keep track of the current sum (or any other property)

    # Process each query
    for i in range(q):
        L, R, idx = queries[i]  # Extract the left, right bounds and the original index of the query

        # Move current left pointer to L
        while currL < L:
            curr_sum -= arr[currL]  # Remove element from current sum
            freq[arr[currL]] -= 1   # Decrease frequency of the element
            currL += 1              # Move left pointer to the right

        while currL > L:
            currL -= 1              # Move left pointer to the left
            curr_sum += arr[currL]  # Add element to current sum
            freq[arr[currL]] += 1   # Increase frequency of the element

        # Move current right pointer to R
        while currR <= R:
            curr_sum += arr[currR]  # Add element to current sum
            freq[arr[currR]] += 1   # Increase frequency of the element
            currR += 1              # Move right pointer to the right

        while currR > R + 1:
            currR -= 1              # Move right pointer to the left
            curr_sum -= arr[currR]  # Remove element from current sum
            freq[arr[currR]] -= 1   # Decrease frequency of the element

        # Store the result for the current query
        result[idx] = curr_sum

    # Return the final results for all queries
    return result

# Example usage
arr = [1, 2, 3, 4, 5]  # Example array
queries = [(0, 2, 0), (1, 3, 1), (2, 4, 2)]  # Example queries (L, R, index)
result = mo_algorithm(arr, queries)  # Process the queries
print(result)  # Output the results
```

**Implementation: C++20**:

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

#### 11.3.1.4. Typical Problem: Humidity Levels in a Greenhouse (Problem 1)

We've already solved this type of problem earlier in _Section: 11.1.1.3_ of this document. In that section, we explored different algorithms and analyzed their time and space complexities when applied to various range query scenarios.

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

## 11.4. Auxiliary Data Structures

**This is a work in progress, we will get there sooner or later.**

Specific data structures used to optimize operations on arrays.

### 11.4.1 Deque (for Sliding Window Minimum/Maximum)

**This is a work in progress, we will get there sooner or later.**

Double-ended queue that maintains relevant elements of the current window.

### 11.4.2 Sparse Table (for RMQ)

**This is a work in progress, we will get there sooner or later.**

Structure that stores pre-computed results for power-of-2 intervals.

### 11.4.3 Segment Tree

**This is a work in progress, we will get there sooner or later.**

Tree-based data structure for range queries and updates in $O(\log n)$.

## 11.5. Complexity Optimization Techniques

**This is a work in progress, we will get there sooner or later.**

Methods to reduce the computational complexity of common operations.

### 11.5.1. Reduction from $O(n^2)$ to $O(n)$

**This is a work in progress, we will get there sooner or later.**

Use of prefix sums to optimize range sum calculations.

- Problem Example: "Sales Target Analysis" - Uses prefix sum technique to optimize subarray calculations

### 11.5.2. Update in $O(1)$

**This is a work in progress, we will get there sooner or later.**

Difference arrays for constant-time range updates.

- Problem Example: "Inventory Restocking" - Makes point adjustments to the inventory

### 11.5.3. Query in $O(1)$ after preprocessing

**This is a work in progress, we will get there sooner or later.**

RMQ and static array queries with instant responses after pre-calculation.

- Problem Example: "The Plate Balancer" - After calculating cumulative sums, can find the "Magic Plate" in O(n)

### 11.5.4. Processing in $O((n + q) \sqrt{n})$

**This is a work in progress, we will get there sooner or later.**

Mo's Algorithm to optimize multiple range queries.

## 11.6. Subarray Algorithms

**This is a work in progress, we will get there sooner or later.**

Specific techniques for problems involving subarrays.

### 11.6.1 Kadane's Algorithm

**This is a work in progress, we will get there sooner or later.**

Finds the contiguous subarray with the largest sum in $O(n)$. Useful for sum maximization problems.

- Algorithm: Kadane's Algorithm

### 11.6.2 Two Pointers

**This is a work in progress, we will get there sooner or later.**

Technique for problems involving pairs of elements or subarrays that satisfy certain conditions.

- Algorithm: Two Pointers Method

## 11.7. Hashing Techniques

**This is a work in progress, we will get there sooner or later.**

Methods that use hashing to optimize certain operations on arrays.

### 11.6.1. Prefix Hash

**This is a work in progress, we will get there sooner or later.**

Uses hashing to quickly compare substrings or subarrays.

- Algorithm: Rolling Hash

### 11.6.2. Rolling Hash

**This is a work in progress, we will get there sooner or later.**

Technique to efficiently calculate hashes of substrings or subarrays when sliding a window.

- Algorithm: Rabin-Karp Algorithm

## 11.8. Partitioning Algorithms

**This is a work in progress, we will get there sooner or later.**

Techniques for dividing or reorganizing arrays.

### 11.6.1. Partition Algorithm (QuickSelect)

**This is a work in progress, we will get there sooner or later.**

Used to find the kth smallest element in average linear 
time.

- Algorithm: QuickSelect

### 11.6.2. Dutch National Flag

**This is a work in progress, we will get there sooner or later.**

Algorithm to partition an array into three parts, useful in sorting problems with few unique values.

- Algorithm: Dutch National Flag Algorithm

