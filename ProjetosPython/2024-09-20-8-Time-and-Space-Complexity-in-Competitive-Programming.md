---
author: Frank
beforetoc: '[Anterior](2024-09-20-7-Sem-T%C3%ADtulo.md)

  [Próximo](2024-09-20-9-Sem-T%C3%ADtulo.md)'
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
title: Time and Space Complexity in Competitive Programming
toc: true
---
# Time and Space Complexity in Competitive Programming

In this section, we will delve deeper into understanding both time and space complexities, providing a more comprehensive look into how these affect the efficiency of algorithms, particularly in competitive programming environments. This includes examining loops, recursive algorithms, and how various complexity classes dictate algorithm performance. We'll also consider the impact of space complexity and memory usage, which is crucial when dealing with large datasets.

## Loops, Time and Space Complexity

**One of the most common reasons for slow algorithms is the presence of multiple loops iterating over input data**. The more nested loops an algorithm contains, the slower it becomes. If there are $k$ nested loops, the time complexity becomes $O(n^k)$.

For instance, the time complexity of the following code is $O(n)$:

```cpp
for (int i = 1; i <= n; i++) {
    // code
}
```

And the time complexity of the following code is $O(n^2)$ due to the nested loops:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        // code
    }
}
```

While the focus is often on time complexity, it's equally important to consider space complexity, especially when handling large inputs. A loop like the one below has a time complexity of $O(n)$ but also incurs a space complexity of $O(n)$ if an array is created to store values:

```cpp
std::vector<int> arr(n);
for (int i = 1; i <= n; i++) {
    arr.push_back(i);
}
```

**In competitive programming, excessive memory use can cause the program to exceed memory limits**. Therefore, always account for the space complexity of your solution, particularly when using arrays, matrices, or data structures that grow with input size.

### Order of Growth

Time complexity doesn't tell us the exact number of times the code within a loop executes but rather gives the order of growth. In the following examples, the code inside the loop executes $3n$, $n+5$, and $\lfloor n/2 \rfloor$ times, but the time complexity of each code is still $O(n)$:

```cpp
for (int i = 1; i <= 3*n; i++) {
    // code
}
```

```cpp
for (int i = 1; i <= n+5; i++) {
    // code
}
```

```cpp
for (int i = 1; i <= n; i += 2) {
    // code
}
```

Another example where time complexity is $O(n^2)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = i+1; j <= n; j++) {
        // code
    }
}
```

### Algorithm Phases and Time Complexity

When an algorithm consists of consecutive phases, the total time complexity is the largest time complexity of any single phase. This is because the slowest phase typically becomes the bottleneck of the code.

For instance, the following code has three phases with time complexities of $O(n)$, $O(n^2)$, and $O(n)$, respectively. Thus, the total time complexity is $O(n^2)$:

```cpp
for (int i = 1; i <= n; i++) {
    // phase 1 code
}
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
        // phase 2 code
    }
}
for (int i = 1; i <= n; i++) {
    // phase 3 code
}
```

#### Space and Time Complexity of Multiple Phases

When analyzing algorithms that consist of multiple phases, consider that each phase may also introduce additional memory usage. In the example above, if phase 2 allocates a matrix of size $n \times n$, the space complexity would increase to $O(n^2)$, matching the time complexity.

Sometimes, time complexity depends on multiple factors. In this case, the formula for time complexity includes multiple variables. For example, the time complexity of the following code is $O(nm)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
        // code
    }
}
```

If the above algorithm also uses a data structure such as a matrix of size $n \times m$, the space complexity would also be $O(nm)$, increasing memory usage significantly, particularly for large input sizes.

## Recursive Algorithms

The time complexity of a recursive function depends on the number of times the function is called and the time complexity of a single call. The total time complexity is the product of these values.

For example, consider the following function:

```cpp
void f(int n) {
    if (n == 1) return;
    f(n-1);
}
```

The call `f(n)` makes $n$ recursive calls, and the time complexity of each call is $O(1)$. Thus, the total time complexity is $O(n)$.

### Exponential Recursion

Consider the following function, which makes two recursive calls for every **Input**:

```cpp
void g(int n) {
    if (n == 1) return;
    g(n-1);
    g(n-1);
}
```

Here, each function call generates two other calls, except when $n = 1$. The table below shows the function calls for a single initial call to $g(n)$:

| Function Call | Number of Calls |
| ------------- | --------------- |
| $g(n)$        | 1               |
| $g(n-1)$      | 2               |
| $g(n-2)$      | 4               |
| ...           | ...             |
| $g(1)$        | $2^{n-1}$       |

Thus, the time complexity is:

$$1 + 2 + 4 + \cdots + 2^{n-1} = 2^n - 1 = O(2^n)$$

Recursive functions also have space complexity considerations. Each recursive call adds to the call stack, and in the case of deep recursion (like in the exponential example above), this can lead to $O(n)$ space complexity. Be cautious with recursive algorithms, as exceeding the maximum stack size can cause a program to fail due to stack overflow.

### Common Complexity Classes

Here is a list of common time complexities of algorithms:

- $O(1)$: A constant-time algorithm doesn't depend on the input size. A typical example is a direct formula calculation.
- $O(\log n)$: A logarithmic algorithm often halves the input size at each step, such as binary search.

- $O(\sqrt{n})$: Slower than $O(\log n)$ but faster than $O(n)$, this complexity might appear in algorithms that involve square root reductions in input size.

- $O(n)$: A linear-time algorithm processes the input a constant number of times.

- $O(n \log n)$: Common in efficient sorting algorithms (e.g., mergesort, heapsort), or algorithms using data structures with $O(\log n)$ operations.

- $O(n^2)$: Quadratic complexity, often seen with nested loops processing all pairs of input elements.

- $O(n^3)$: Cubic complexity arises with three nested loops, such as algorithms processing all triples of input elements.

- $O(2^n)$: This complexity usually indicates exponential growth, common in recursive algorithms that explore all subsets.

- $O(n!)$: Common in algorithms that generate all permutations of the input.

### Estimating Efficiency

When calculating an algorithm's time complexity, you can estimate whether it will be efficient enough for the given problem before implementation. A modern computer can perform hundreds of millions of operations per second.

For example, assume that the input size is $n = 10^5$. If the time complexity is $O(n^2)$, the algorithm would perform roughly $(10^5)^2 = 10^{10}$ operations, which would take several seconds, likely exceeding the time limits of most competitive programming environments.

On the other hand, given the input size, we can estimate the required time complexity of an algorithm. The following table provides useful estimates, assuming a time limit of one second:

| Input Size    | Required Time Complexity |
| ------------- | ------------------------ |
| $n \leq 10$   | $O(n!)$                  |
| $n \leq 20$   | $O(2^n)$                 |
| $n \leq 500$  | $O(n^3)$                 |
| $n \leq 5000$ | $O(n^2)$                 |
| $n \leq 10^6$ | $O(n \log n)$ or $O(n)$  |
| $n$ is large  | $O(1)$ or $O(\log n)$    |

For example, if the input size is $n = 10^5$, it is likely that the algorithm must have a time complexity of $O(n)$ or $O(n \log n)$. This insight can help guide the design of the algorithm and eliminate approaches that would result in worse time complexity.

While time complexity is a good estimate of efficiency, it hides constant factors. For example, an $O(n)$ algorithm might perform $n/2$ or $5n$ operations, and these constants can significantly affect the actual running time.

Since loops have a significant impact on code performance, we can dive deeper into the possible loop options available.

