---
author: Frank
beforetoc: |-
  [Anterior](2024-09-24-1-Se%C3%A7%C3%A3o-1.md)
  [Próximo](2024-09-24-3-2.-C%2B%2B-Competitive-Programming-Hacks.md)
categories:
  - Matemática
  - Linguagens Formais
  - Programação
description: Introduction of a journey to explore C++ in competitive programming. Learn optimization, algorithms, and data structures. Improve coding skills for challenges.
draft: null
featured: true
image: assets/images/compte_introd.jpg
keywords: Competitive Programming, C++ Algorithms, Dynamic Programming, Performance Analysis, Coding Examples, Algorithm Optimization, Practical Programming Guide
lastmod: 2024-09-28T03:20:05.536Z
layout: post
preview: Why we will study C++ 20 and a bunch of algorithms for competitive programming, Here are the reasons for use C++ 20 and write this huge document.
published: false
rating: 5
slug: competitive-prog-tech-insights-introduction
tags:
  - Algorithm Optimization
  - Competitive Programming Guide
  - C++ 20
  - Cpp 20
title: Competitive Programmint. in C++ Technologies and Insights - Introduction
toc: false
---

# 1. Introduction

C++ remains one of the most popular languages in competitive programming due to its performance, flexibility, and rich standard library. However, knowledge of efficient algorithms is as important—if not more so—than the programming language itself. Mastering efficient algorithms and optimized techniques is crucial for success in programming contests, where solving complex problems under strict time and memory constraints is the norm. This guide explores advanced algorithmic strategies, optimization techniques, and data structures that help tackle a wide range of computational challenges effectively. By optimizing input/output operations, leveraging modern C++ features, and utilizing efficient algorithms, we'll see how to improve problem-solving skills in competitive programming.

For instance, one common optimization in competitive programming is to speed up input/output operations. By default, C++ performs synchronized I/O with C's standard I/O libraries, which can be slower. A simple trick to improve I/O speed is disabling this synchronization:

```cpp
std::ios_base::sync_with_stdio(false);
cin.tie(nullptr);
```

This small change can make a significant difference when dealing with large input datasets. Throughout this guide, we will explore similar techniques, focusing on the efficient use of data structures like arrays and vectors, as well as modern C++20 features that streamline your code. You will learn how to apply optimizations that minimize overhead and effectively leverage STL containers and algorithms to boost both runtime performance and code readability. From handling large-scale data processing to optimizing solutions under tight time constraints, these strategies will prepare you to excel in programming contests.

Besides C++20, we will study the most efficient algorithms for each of the common problems found in competitions held by organizations like [USACO](https://usaco.org/) and [ICPC](https://www.icpc.org/).

In this goal, processing large arrays or sequences quickly and efficiently is essential in competitive programming. Techniques in array manipulation are tailored to handle millions of operations with minimal overhead, maximizing performance. In graph algorithms, the focus shifts to implementing complex traversals and finding shortest paths, addressing problems from route optimization to network connectivity. These methods require precise control over nodes and edges, whether working with directed or undirected graphs.

String processing handles tasks like matching, parsing, and manipulating sequences of characters. It requires carefully crafted algorithms to search, replace, and transform strings, often using specialized data structures like suffix trees to maintain performance. Data structures go beyond basic types, introducing advanced forms such as segment trees and Fenwick trees. These are essential for managing and querying data ranges quickly, especially in scenarios where direct computation is too slow.

Computational geometry tackles geometric problems with high precision, calculating intersections, areas, and volumes. The focus is on solving spatial problems using algorithms that respect boundaries of precision and efficiency, delivering results where accuracy is crucial.

The journey through competitive programming in C++ starts with the basics. It's not a journey we chose lightly. [Ana Flávia Martins dos Santos](https://github.com/aflavinhams), [Isabella Vanderlinde Berkembrock](https://github.com/isabella1709), and [Michele Cristina Otta](https://github.com/micheleotta) inspired it. They succeeded without training. We wanted to build on that. Our goal is not only to present C++20 and a key set of algorithms for competitive programming, but also to inspire them, and everyone who reads this work or follows my courses, to become better professionals. We hope they master a wide range of computational tools that are rarely covered in traditional courses.

We’ll start with a set of small tricks and tips for typing and writing code to cut down the time spent drafting solutions. In many contests, it’s not enough for your code to be correct and fast; the time you spend presenting your solutions to the challenges matters too. So, typing less and typing fast is key.

For each problem type, we’ll study the possible algorithms, show the best one based on complexity, and give you a Python pseudocode followed by a full C++20 solution. We won’t fine-tune the Python code, but it runs and works as a solid base for competitive programming practice in Python. When needed, we’ll break down C++20 methods, functions, operators, and classes, often highlighting them for focus. There are clear, solid reasons why we picked C++20.

C++ is a powerful tool for competitive programming. It excels in areas like array manipulation, graph algorithms, string processing, advanced data structures, and computational geometry. Its speed and rich standard library make it ideal for creating efficient solutions in competitive scenarios.

We cover the essentials of looping, from simple `for` and `while` loops to modern C++20 techniques like range-based `for` loops with views and parallel execution policies. The guide also explores key optimizations: reducing typing overhead, leveraging the Standard Template Library (STL) effectively, and using memory-saving tools like `std::span`.

Mastering these C++20 techniques prepares you for a wide range of competitive programming challenges. Whether handling large datasets, solving complex problems, or optimizing for speed, these strategies will help you write fast, efficient code. This knowledge will sharpen your skills, improve your performance in competitions, and deepen your understanding of both C++ and algorithmic thinking—skills that go beyond the competition.

C++ shows its strength when solving complex problems. In array manipulation, it supports fast algorithms like binary search with $O(\log n)$ time complexity, crucial for quick queries in large datasets. For graph algorithms, C++ can implement structures like adjacency lists with a space complexity of $O(V + E)$, where $V$ is vertices and $E$ is edges, making it ideal for sparse graphs.

In string processing, C++ handles pattern searching efficiently, using algorithms like KMP (Knuth-Morris-Pratt), which runs in $O(n + m)$, where $n$ is the text length and $m$ is the pattern length. Advanced data structures, such as segment trees, allow for query and update operations in $O(\log n)$, essential for range queries and frequent updates.

C++ also handles computational geometry well. Algorithms like Graham's scan find the convex hull with $O(n \log n)$ complexity, demonstrating C++'s efficiency in handling geometric problems.

We are going to cover a lot of ground. From basic techniques to advanced algorithms. But remember, it all started with Ana, Isabella, and Michele. Their success without training showed us what's possible. Now, armed with these tools and knowledge, you're ready to take on any challenge in competitive programming. The code is clean. The algorithms are efficient. The path is clear. Go forth and compete.

## 1.1 Time and Space Complexity

In this section, we’ll gona try to understand time and space complexities, looking at how they affect algorithm efficiency, especially in competitive programming without all mathematics. We’ll break down loops, recursive algorithms, and how different complexity classes shape performance. We’ll also look at space complexity and memory use, which matter when handling large datasets.

**One major cause of slow algorithms is having multiple nested loops that run over the input data.** The more nested loops there are, the slower the algorithm gets. With $k$ nested loops, the time complexity rises to $O(n^k)$.

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

While time complexity gets most of the attention, space complexity is just as crucial, especially with large inputs. A loop like the one below has a time complexity of $O(n)$, but if it creates an array to store values, it also has a space complexity of $O(n)$:

```cpp
std::vector<int> arr(n);
for (int i = 1; i <= n; i++) {
    arr.push_back(i);
}
```

**In competitive programming, excessive memory use can cause your program to exceed memory limits, though this isn’t very common in competitions.** Always keep an eye on space complexity, especially when using arrays, matrices, or other data structures that grow with input size. Manage memory wisely to avoid crashes and penalties.

## 1.1.2. Order of Growth

Time complexity doesn’t show the exact number of times the code inside a loop runs; it shows how the runtime grows. In these examples, the loop runs $3n$, $n+5$, and $\lfloor n/2 \rfloor$ times, but all still have a time complexity of $O(n)$:

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

This is true because time complexity looks at growth, not exact counts. Big-O ignores constants and small terms since they don’t matter much as input size ($n$) gets large.

Here’s why each example still counts as $O(n)$:

1. **$3n$ executions**: The loop runs $3n$ times, but the constant 3 doesn’t change the growth. It’s still linear, so it’s $O(n)$.

2. **$n + 5$ executions**: The $+5$ is just a fixed number. It’s small next to $n$ when things get big. The main growth is still $n$, so it’s $O(n)$.

3. **$\lfloor n/2 \rfloor$ executions**: Cutting $n$ in half or any fraction doesn’t change the overall growth rate. It’s still linear, so it’s $O(n)$.

Big-O isn’t disconnected from real execution speed. Constants like the 3 in $3n$ do affect how fast the code runs, but they aren’t the focus of Big-O. In real terms, an algorithm with $3n$ operations will run slower than one with $n$, but both grow at the same rate—linearly. That’s why they both fall under $O(n)$.

Big-O doesn’t ignore these factors because they don’t matter; it simplifies them. It’s all about the growth rate, not the exact count, because that’s what matters most when inputs get large.

Another example where time complexity is $O(n^2)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = i+1; j <= n; j++) {
        // code
    }
}
```

When an algorithm has consecutive phases, the overall time complexity is driven by the slowest phase. The phase with the highest complexity usually becomes the bottleneck.

For example, the code below has three phases with time complexities of $O(n)$, $O(n^2)$, and $O(n)$. The slowest phase dominates, making the total time complexity $O(n^2)$:

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

When looking at algorithms with multiple phases, remember that each phase can also add to memory use. In the example above, if phase 2 creates an $n \times n$ matrix, the space complexity jumps to $O(n^2)$, matching the time complexity.

Sometimes, time complexity depends on more than one factor. This means the formula includes multiple variables. For instance, the time complexity of the code below is $O(nm)$:

```cpp
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
        // code
    }
}
```

If the algorithm above uses a data structure like an $n \times m$ matrix, the space complexity also becomes $O(n\times m)$. This increases memory usage, especially with large inputs.

The time complexity of a recursive function depends on how often it’s called and the complexity of each call. Multiply these together to get the total time complexity.

For example, look at this function:

```cpp
void f(int n) {
    if (n == 1) return;
    f(n-1);
}
```

The call `f(n)` makes $n$ recursive calls, each with a time complexity of $O(1)$. So, the total time complexity is $O(n)$.

We also need to watch out for functions with exponential growth, like the one below, which makes two recursive calls for every input:

```cpp
void g(int n) {
    if (n == 1) return;
    g(n-1);
    g(n-1);
}
```

Here, each call to the function creates two more calls, except when $n = 1$. The table below shows the calls made from a single initial call to $g(n)$:

| Function Call | Number of Calls |
| ------------- | --------------- |
| $g(n)$        | 1               |
| $g(n-1)$      | 2               |
| $g(n-2)$      | 4               |
| ...           | ...             |
| $g(1)$        | $2^{n-1}$       |

So, the total time complexity is:

$$1 + 2 + 4 + \cdots + 2^{n-1} = 2^n - 1 = O(2^n)$$

Recursive functions also bring space complexity issues. Each call adds to the call stack, and with deep recursion, like this exponential example, the space complexity can be $O(n)$. Be careful: too many recursive calls can lead to a stack overflow and cause the program to crash.

### 1.1.3. Common Complexity Classes

Here is a table of common time complexities of algorithms:

| Complexity    | Description                                                                                    | Examples                               |
| ------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------- |
| $O(1)$        | Constant time; doesn't depend on input size.                                                   | Direct calculations.                   |
| $O(\log n)$   | Logarithmic time; halves the input size each step.                                             | Binary search.                         |
| $O(\sqrt{n})$ | Slower than $O(\log n)$ but faster than $O(n)$. Appears when input reduces by its square root. | Some specific mathematical algorithms. |
| $O(n)$        | Linear time; processes input once or a fixed number of times.                                  | Simple loops.                          |
| $O(n \log n)$ | Common in efficient sorting and algorithms using $O(\log n)$ operations.                       | Mergesort, heapsort.                   |
| $O(n^2)$      | Quadratic time; nested loops processing pairs of input elements.                               | Matrix operations, bubble sort.        |
| $O(n^3)$      | Cubic time; three nested loops, processing triples.                                            | Some dynamic programming algorithms.   |
| $O(2^n)$      | Exponential growth; often in recursive algorithms exploring all subsets.                       | Subset generation, recursive searches. |
| $O(n!)$       | Factorial time; algorithms generating all permutations of the input.                           | Permutation generation.                |

### 1.1.4. Estimating Efficiency

When figuring out an algorithm’s time complexity, you can estimate if it’ll be fast enough before you even start coding. A modern computer can handle hundreds of millions of operations per second.

For instance, if your input size is $n = 10^5$ and your time complexity is $O(n^2)$, the algorithm would need about $(10^5)^2 = 10^{10}$ operations. That’s several seconds, too slow for most competitive programming limits.

You can also judge what time complexity you need based on the input size. Here’s a quick guide, assuming a one-second time limit, very common for C++ competitions:

| Input Size    | Required Time Complexity |
| ------------- | ------------------------ |
| $n \leq 10$   | $O(n!)$                  |
| $n \leq 20$   | $O(2^n)$                 |
| $n \leq 500$  | $O(n^3)$                 |
| $n \leq 5000$ | $O(n^2)$                 |
| $n \leq 10^6$ | $O(n \log n)$ or $O(n)$  |
| $n$ is large  | $O(1)$ or $O(\log n)$    |

So, if your input size is $n = 10^5$, you’ll probably need an algorithm with $O(n)$ or $O(n \log n)$ time complexity. This helps you steer clear of approaches that are too slow.

**Remember, time complexity is an estimate; it hides constant factors. An $O(n)$ algorithm could do $n/2$ or $5n$ operations, and these constants can change the actual speed.**
