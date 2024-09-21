---
author: Frank
beforetoc: '[Anterior](2024-09-20-10-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-12-Sem-T%C3%ADtulo.md)'
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

###### Prefix Sum Array Solution

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

