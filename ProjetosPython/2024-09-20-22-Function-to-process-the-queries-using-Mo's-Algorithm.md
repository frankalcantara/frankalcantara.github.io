---
author: Frank
beforetoc: '[Anterior](2024-09-20-21-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-23-Sem-T%C3%ADtulo.md)'
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
title: Function to process the queries using Mo's Algorithm
toc: true
---
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

