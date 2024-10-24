---
author: Frank
beforetoc: Advanced optimization strategies for competitive programming in C++20 focus on using macros, lambdas, and templates to improve efficiency and minimize errors. Techniques like dynamic data manipulation and compile-time evaluation streamline coding for high-performance contests. By applying these methods, programmers can write cleaner, faster code, making complex algorithms easier to implement during competition.
categories:
    - Matemática
    - Linguagens Formais
    - Programação
description: Optimization in C++20, to boost efficiency, reduce errors, and handle dynamic data, helping create faster, cleaner code for programming contests.
draft: null
featured: true
image: assets/images/compte_introd.webp
keywords: Competitive Programming, C++ Algorithms, Dynamic Programming, Performance Analysis, Coding Examples, Algorithm Optimization, Practical Programming Guide
lastmod: 2024-10-14T02:15:25.422Z
layout: post
preview: Advanced optimization strategies for competitive programming in C++20 focus on using macros, lambdas, and templates to improve efficiency and minimize errors. Techniques like dynamic data manipulation and compile-time evaluation streamline coding for high-performance contests. By applying these methods, programmers can write cleaner, faster code, making complex algorithms easier to implement during competition.
published: true
rating: 5
slug: competitive-programming-insights-introduction
tags:
    - C++20
    - Competitive programming
    - Code optimization
    - Macros
    - Lambdas
    - Templates
    - Compile-time evaluation
    - Dynamic data structures
    - Efficient algorithms
    - High-performance coding
    - Sorting algorithms
    - Generic programming
    - Vector manipulation
    - Runtime operations
    - Competitive coding techniques
    - Code efficiency
    - Debugging
    - Template lambdas
    - Data collections
    - Programming contests
title: Competitive Programming in C++ Insights - Introduction
toc: true
date: 2024-10-14T01:41:22.930Z
---

# 1. Introduction

C++ remains one of the most popular languages in competitive programming due to its performance, flexibility, and rich standard library. However, knowledge of efficient algorithms is as important,if not more so, than the programming language itself. Mastering efficient algorithms and optimized techniques is crucial for success in programming contests, where solving complex problems under strict time and memory constraints is the norm. in that same vein, understanding the nuances of algorithmic complexity can dramatically affect both performance and outcomes.

This work    explores advanced algorithmic strategies, optimization techniques, and data structures that help tackle a wide range of computational challenges effectively. By optimizing input/output operations, leveraging modern C++ features, and utilizing efficient algorithms, we'll see how to improve problem-solving skills in competitive programming. We'll be pushing the boundaries of what can be achieved with minimal resources and tight constraints.

The scene of this particular whodunnit may seem unlikely, but input/output operations often become the hidden bottleneck in performance-critical applications. By default, C++ performs synchronized I/O with C's standard I/O libraries, which can be slower. A simple trick to improve I/O speed is disabling this synchronization:

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

_Table 1.1.2.A - Counting calls to understand exponential growth._{: class="legend"}

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

_Table 1.1.3.A - Common time complexities of algorithms._{: class="legend"}

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

_Table 1.1.4.A - Theoretical relationship between time complexity and input size._{: class="legend"}

So, if your input size is $n = 10^5$, you’ll probably need an algorithm with $O(n)$ or $O(n \log n)$ time complexity. This helps you steer clear of approaches that are too slow.

**Remember, time complexity is an estimate; it hides constant factors. An $O(n)$ algorithm could do $n/2$ or $5n$ operations, and these constants can change the actual speed.**

## 1.2. Typing Better, Faster, Less

This section gives you practical steps to improve your speed and performance in competitive programming with C++ 20. C++ is fast and powerful, but using it well takes skill and focus. We will cover how to type faster, write cleaner code, and manage complexity. The goal is to help you code quicker, make fewer mistakes, and keep your solutions running fast.

Typing matters. The faster you type, the more time you save. Accuracy also counts, mistakes slow you down. Next, we cut down on code size without losing what’s important. Using tools like the Standard Template Library (STL), you can write less code and keep it clean. This is about direct, simple code that does the job right.

In this section, we won’t confine ourselves to general tricks. We will explore certain nuances and techniques specific to C++20, particularly those related to typing efficiency. Undoubtedly, in other chapters and sections, we will return to these same topics, but from different angles.

### 1.2.1. Typing Tips

If you don’t type quickly, **you should invest at least two hours per week** on the website: [https://www.speedcoder.net](https://www.speedcoder.net). Once you have completed the basic course, select the C++ lessons and practice regularly. Time is crucial in competitive programming, and slow typing can be disastrous.

To expand on this, efficient typing isn’t just about speed; it’s about reducing errors and maintaining a steady flow of code. When you're in a competitive programming, every second matters. Correcting frequent typos or having to look at your keyboard will significantly slow down your progress. Touch typing—knowing the layout of the keyboard and typing without looking—becomes a vital skill.

In a typical competitive programming contest, you have to solve several, typically 12 or 15, problems within a set time, about five hours. Typing fast lets you spend more time solving problems instead of struggling to get the code in. But speed means nothing without accuracy. Accurate and fast typing ensures that once you know the solution, you can code it quickly and correctly.

Typing slow or making frequent errors costs you valuable time. You waste time fixing mistakes, lose focus on solving the problem, and increase the chance of not finishing on time. Aim for a typing speed of at least 60 words per minute with high accuracy. Websites like [https://www.speedcoder.net](https://www.speedcoder.net) let you practice typing code syntax, which helps more than regular typing lessons. Learning specific shortcuts in C++ or Python boosts your speed in real coding situations.

$$\text{Time spent fixing errors } + \text{Time lost from slow typing }$$

$$ = \text{Lower overall performance in competitive programming}$$

To improve your typing in competitive programming, start by using IDE shortcuts. Learn the keyboard shortcuts for your preferred Integrated Development Environment (IDE). Shortcuts help you navigate and edit code faster, cutting down the time spent moving between the keyboard and mouse. In [ICPC](https://icpc.global/) contests, the IDE is usually [Eclipse](https://www.eclipse.org/downloads/packages/release/helios/sr2/eclipse-ide-cc-developers) or [VsCode](https://code.visualstudio.com/), so knowing its shortcuts can boost your efficiency. Always check which IDE will be used in each competition since this may vary. And use it daily while training.

while typing, focus on frequent patterns in your code. Practice typing common elements like loops, if-else conditions, and function declarations. Embedding these patterns into your muscle memory saves time during contests. The faster you can type these basic structures, the quicker you can move on to solving the actual problem.

To succeed in a C++ programming competition, your first challenge is to type the following code in under two minutes. If you can't, don't give up. Just keep practicing. To be the best in the world at anything, no matter what it is, the only secret is to train and train some more.

```cpp
#include <iostream>
#include <vector>
#include <span>
#include <string>
#include <algorithm>
#include <random>

// Type aliases
using VI = std::vector<int>;
using IS = std::span<int>;
using STR = std::string;

// Function to double each element in the vector
void processVector(VI& vec) {
    std::cout << "Processing vector...\n";
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= 2;
    }
}

// Function to display elements of a span
void displaySpan(IS sp) {
    std::cout << "Displaying span: ";
    for (const auto& elem : sp) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    VI numbers;
    STR input;

    // Input loop: collect numbers from user
    std::cout << "Enter integers (type 'done' when finished):\n";
    while (true) {
        std::cin >> input;
        if (input == "done") break;
        numbers.push_back(std::stoi(input));
    }

    // Process and display the vector
    processVector(numbers);
    std::cout << "Processed vector:\n";
    int index = 0;
    while (index < numbers.size()) {
        std::cout << numbers[index] << " ";
        ++index;
    }
    std::cout << "\n";

    // Shuffle the vector using a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    std::cout << "Shuffled vector:\n";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << "\n";

    // Display a span of the first 5 elements (if available)
    if (numbers.size() >= 5) {
        IS numberSpan(numbers.data(), 5);
        displaySpan(numberSpan);
    }

    // Calculate sum of elements at even indices
    int sum = 0;
    for (int i = 0; i < numbers.size(); i += 2) {
        sum += numbers[i];
    }
    std::cout << "Sum of elements at even indices: " << sum << "\n";

    return 0;
}
```

_Code 1.2.1.A - Code for self-assessment of your typing speed._{: class="legend"}

Don't give up before trying. If you can't type fast enough, don't stop here. Keep going. With each new algorithm, copy and practice again until it feels natural to type between 60 and 80 words per minute with an accuracy above $95%$.

### 1.2.2. Typing Less in Competitive Programming

In competitive programming, time is a critical resource and C++ is a language where you have to type a lot. Therefore, optimizing typing speed and avoiding repetitive code can make a significant difference. Below, we will discuss strategies to minimize typing when working with `std::vector` during competitive programmings, where access to the internet or pre-prepared code snippets may be restricted.

#### 1.2.2.1. Abbreviations

We can use `#define` to create short aliases for common vector types. This is particularly useful when you need to declare multiple vectors throughout the code.

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

With these definitions, declaring vectors becomes much faster:

```cpp
VI numbers;  // std::vector<int> numbers;
VVI matrix;  // std::vector<std::vector<int>> matrix;
VS words;    // std::vector<std::string> words;
```

In C++, you can use `#define` to create macros and short aliases. Macros can define constants or functions at the preprocessor level. But in C++20, most macros have better alternatives. A typical use is reduce the definition and common operations with vectors. Some examples follow:

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

With these definitions, declaring vectors becomes much faster:

```cpp
VI numbers;  // std::vector<int> numbers;
VVI matrix;  // std::vector<std::vector<int>> matrix;
VS words;    // std::vector<std::string> words;
```

_Macros can cause problems_. They ignore scopes and can lead to unexpected behavior. For constants, use `constexpr`, `const` or `` instead of macros.

```cpp
// Old way using macros
#define PI 3.14159

// Modern way using constexpr
constexpr double PI = 3.14159;
```

> The `constexpr` keyword tells the compiler to evaluate a function or variable at compile time. If possible, the compiler computes it before runtime. This helps in optimization, as constants are determined during compilation, not execution.
>
> When you use `constexpr` with variables, the compiler knows the value at compile time. The value is constant and cannot change.
>
> ```cpp
> constexpr int max_items = 42;  // Value known at compile time, cannot change.
> ```
>
> You can use `constexpr` variables to define array sizes or template parameters when their values are known before or during compilation.
>
> ```cpp
> constexpr int size = 10;
> int array[size];  // Valid, size is a constant expression.
> ```
>
> We can use `constexpr` with Functions. A `constexpr` function can be evaluated at compile time if its inputs are constant expressions. All operations inside must be valid at compile time.
>
> ```cpp
> constexpr int factorial(int n) {
>     return n <= 1 ? 1 : n * factorial(n - 1);  // Recursive function computed at compile time.
> }
> ```
>
> If you call `factorial(6)` with a constant value, the compiler computes it at compile time and replaces the call with `720`.
>
> `constexpr` is useful in many contexts. It helps construct constant data, optimize algorithms, and define compile-time logic. Here are some examples:
>
> 1. Compile-time array size:
>
>    ```cpp
>    constexpr int size = 5;
>    int array[size];  // Size computed at compile time.
>    ```
>
> 2. Compile-time strings:
>
>    ```cpp
>    constexpr const char* greet() { return "Hello, World!"; }
>    constexpr const char* message = greet();  // Message computed at compile time.
>    ```
>
> 3. Compile-time mathematical operations:
>
>    ```cpp
>    constexpr int area(int length, int width) {
>        return length * width;
>    }
>    constexpr int room_area = area(10, 12);  // Computed at compile time.
>    ```
>
> In competitive programming, `constexpr` can be an advantage or a disadvantage. Advantage, `constexpr` can optimize code by computing results at compile time. This saves processing time during execution. If certain values are constant, you can precompute them with `constexpr`. Disadvantage: Many problems have dynamic input provided at runtime. `constexpr` cannot compute values that depend on runtime input. Since runtime efficiency is crucial, `constexpr` use is limited when inputs are dynamic.
>
> Overall, `constexpr` is valuable when dealing with static data or fixed input sizes. But in typical ICPC-style competitions, you use it less often because most problems require processing dynamic input.
>
> _For functions, macros can be unsafe_. They don't respect types or scopes. Modern C++ provides templates and `constexpr` functions.
>
> ```cpp
> // Macro function
> #define SQUARE(x) ((x) * (x))
>
> // Template function
> template<typename T>
> constexpr T square(T x) {
>     return x * x;
> }
> ```
>
> _Macros are processed before compilation. This makes debugging hard_. The compiler doesn't see macros the same way it sees code. With modern C++, you have better tools that the compiler understands. C++20 offers features like `constexpr` functions, inline variables, and templates. These replace most uses of macros. They provide type safety and respect scopes. They make code easier to read and maintain. You can define a `constexpr` function to compute the square of a number:
>
> ```cpp
> constexpr int square(int n) {
>     return n * n;
> }
> ```
>
> If you call `square(5)` in a context requiring a constant expression, the compiler evaluates it at compile time.
>
> In summary, avoid macros when you can. Use modern C++ features instead. They make your code safer and clearer.

**A better way to reduce typing time is by using `typedef` or `using` to create abbreviations for frequently used vector types.** In many cases, the use of `#define` can be replaced with more modern and safe C++ constructs like `using`, `typedef`, or `constexpr`. **The old `#define` does not respect scoping rules and does not offer type checking, which can lead to unintended behavior**. Using `typedef` or `using` provides better type safety and integrates smoothly with the C++ type system, making the code more predictable and easier to debug.

For example:

```cpp
#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>
#define VS std::vector<std::string>
```

Can be replaced with `using` or `typedef` to create type aliases:

```cpp
using VI = std::vector<int>;
using VVI = std::vector<std::vector<int>>;
using VS = std::vector<std::string>;

// Or using typedef (more common in C++98/C++03)
typedef std::vector<int> VI;
typedef std::vector<std::vector<int>> VVI;
typedef std::vector<std::string> VS;
```

**`using` and `typedef` are preferred because they respect C++ scoping rules and offer better support for debugging, making the code more secure and readable**. nevertheless, there are moments when we need a constant function.

If you have macros that perform calculations, you can replace them with `constexpr` functions:

```cpp
  #define SQUARE(x) ((x) * (x))
```

Can be replaced with:

```cpp
 constexpr int square(int x) {
     return x * x;
}
```

**`constexpr` functions provide type safety, avoid unexpected side effects, and allow the compiler to evaluate the expression at compile-time, resulting in more efficient and safer code**.

In competitive programming, you might think using `#define` is the fastest way to type less and code faster. But `typedef` or `using` are usually more efficient. They avoid issues with macros and integrate better with the compiler. **While reducing variable names or abbreviating functions might save time in a contest, remember that in professional code, clarity and maintainability are more important than typing speed**. So avoid using shortened names and unsafe constructs like `#define` in production code, libraries, or larger projects.

> In C++, you can create aliases for types. This makes your code cleaner. You use `typedef` or `using` to do this.
>
> `typedef` lets you give a new name to an existing type.
>
> ```cpp
> using ull = unsigned long long;
> ```
>
> Now, `ull` is an alias for `unsigned long long`. You can use it like this:
>
> ```cpp
> ull bigNum = 123456789012345ULL;
> ```
>
> In C++, numbers need type-specific suffixes like `ULL`. When you write `ull bigNumber = 123456789012345ULL;`, the `ULL` tells the compiler the number is an `unsigned long long`. Without it, the compiler might assume a smaller type like `int` or `long`, which can't handle large values. This leads to errors and bugs. The suffix forces the right type, avoiding overflow and keeping your numbers safe. It’s a simple step but crucial. The right suffix means the right size, no surprises.
>
> **In C++, suffixes are also used with floating-point numbers to specify their exact type**. The suffix `f` designates a `float`, while no suffix indicates a `double`, and `l` or `L` indicates a `long double`. By default, the compiler assumes `double` if no suffix is provided. Using these suffixes is important when you need specific control over the type, such as saving memory with `float` or gaining extra precision with `long double`. The suffix ensures that the number is treated correctly according to your needs.Exact type, exact behavior.

#### 1.2.2.2. Predefining Common Operations

If you know that certain operations, such as sorting or summing elements, are frequent in a competitive programming, consider defining these operations at the beginning of the code. _The only real reason to use a macro in competitive programming is to predefine functions_. For example:

```cpp
// A function to sort vectors
#define SORT_VECTOR(vec) std::sort(vec.begin(), vec.end())
```

This function can be further simplified in its use:

```cpp
#include <vector>
#include <algorithm>

#define SORT_VECTOR(vec) std::sort(vec.begin(), vec.end())
#define ALL(vec) vec.begin(), vec.end()
using VI = std::vector<int>; // Alias with using needs a semicolon

// Usage:
VI vec = {5, 3, 8, 1};
SORT_VECTOR(vec); // Sorts the vector using the SORT_VECTOR macro
// Alternatively, you can sort using ALL macro:
std::sort(ALL(vec)); Another way to use the macro to sort
```

> In C++, `#include` brings code from libraries into your program. It lets you use functions and classes defined elsewhere. The line `#include <vector>` includes the vector library.
>
> Vectors are dynamic arrays. They can change size at runtime. You can add or remove elements as needed. We will know more about vectors and the vector library soon. In early code fragments we saw some examples of vector initialization.
>
> The line `#include <algorithm>` includes the algorithm library. It provides functions to work with data structures. You can sort, search, and manipulate collections.
>
> We can merge `<vector>` and `<algorithm>` for efficient data processing. We’ve seen this in previous code examples where we used competitive programming techniques. Without competitive programming tricks and hacks, the libraries can be combined like this:
>
> ```cpp
> #include <vector>
> #include <algorithm>
> #include <iostream>
>
> int main() {
>     std::vector<int> numbers = {4, 2, 5, 1, 3};
>     std::sort(numbers.begin(), numbers.end());
>
>     for (int num : numbers) {
>         std::cout << num << " ";
>     }
>     return 0;
> }
> ```
>
> This program, a simple example of professional code, sorts the numbers and prints:
>
> ```txt
> 1 2 3 4 5
> ```

Another common operation in competitions is the addition of elements within the same vector. For this, C++20 offers `std::accumulate`.

The difference between `sum` and `accumulate` is in their usage and flexibility. In C++, `std::accumulate` is part of the `<numeric>` library and calculates the sum (or other operations) over a range of elements starting from an initial value. Unlike other languages, C++ does not have a built-in `sum` function, but `std::accumulate` serves that purpose.

In the provided code:

```cpp
#define ALL(x) x.begin(), x.end()

\\...

int sum_vector(const VI& vec) {
    return std::accumulate(ALL(vec), 0);
}
```

The code `std::accumulate(ALL(vec), 0);` will be replaced in compilation time for std::accumulate(vec.begin(), vec.end(), 0)` witch takes three arguments:

- The first two define the range to sum (`vec.begin()` to `vec.end()`).
- The third argument is the initial value, which is $0$, added to the summation result.

If `std::accumulate` is used without a custom function, it defaults to addition, behaving like a simple summation. To calculate the sum of a vector's elements:

> `std::accumulate` uses functions to operate on elements. These functions are straightforward. They take two values and return one. Let's see how they work.
>
> We can begin for Lambda functions, they are unnamed functions, useful for quick operations.
>
> ```cpp
> [](int a, int b) { return a + b; }
> ```
>
> This lambda adds two numbers. The `[]` is the capture list. It's empty here. `(int a, int b)` are the parameters. `{ return a + b; }` is the function body. For example we could write:
>
> ```cpp
> int sum = std::accumulate(ALL(vec), 0, [](int a, int b) { return a + b; });
> ```
>
> This lambda sums all elements in `v`. Nevertheless, the C++ provides standard functions for common operations in `std::accumulate`.
>
> ```cpp
> int sum = std::accumulate(All(vec), 0, std::plus<>());
> ```
>
> `std::plus<>` adds two numbers. It’s the default for `std::accumulate`.
>
> ```cpp
> int product = std::accumulate(ALL(vec), 1, std::multiplies<>());
> ```
>
> `std::multiplies<>` multiplies two numbers.
>
> Using lambda functions, or not, you can create your own functions.
>
> ```cpp
> int max_element = std::accumulate(ALL(vec), v[0],
>     [](int a, int b) { return std::max(a, b); });
> ```
>
> This lambda function finds the largest element in `v`. In this case, `std::accumulate` applies the function repeatedly. It starts with an initial value. For each element:
>
> 1. It takes the previous result.
> 2. It takes the current element.
> 3. It applies the function to both values.
> 4. The result is used in the next iteration.
>
> This process continues until the sequence ends. Let’s see how `std::accumulate` sums the squares of numbers:
>
> ```cpp
> std::vector<int> vec = {1, 2, 3, 4};
> int sum_of_squares = std::accumulate(ALL(vec), 0,
>     [](int acc, int x) { return acc + x * x; });
> ```
>
> The process goes like this:
>
> 1. Start: $acc = 0$
> 2. For $1$: $acc = 0 + 1 * 1 = 1$
> 3. For $2$: $acc = 1 + 2 * 2 = 5$
> 4. For $3$: $acc = 5 + 3 * 3 = 14$
> 5. For $4$: $acc = 14 + 4 * 4 = 30$
>
> The final result is $30$.

### 1.2.2.3. Using Lambda Functions

Let's can back to Lambdas.

Starting with C++11, C++ introduced lambda functions. Lambdas are anonymous functions that can be defined exactly where they are needed. If your code needs a simple function used only once, you should consider using lambdas. Let’s start with a simple example, written without competitive programming tricks.

```cpp
#include <iostream>  // Includes the input/output stream library for console operations
#include <vector>    // Includes the vector library for using dynamic arrays
#include <algorithm> // Includes the algorithm library for functions like sort

// Traditional function to sort in descending order
bool compare(int a, int b) {
    return a > b; // Returns true if 'a' is greater than 'b', ensuring descending order
}

int main() {
    std::vector<int> numbers = {1, 3, 2, 5, 4}; // Initializes a vector of integers

    // Uses the compare function to sort the vector in descending order
    std::sort(numbers.begin(), numbers.end(), compare);

    // Prints the sorted vector
    for (int num : numbers) {
        std::cout << num << " "; // Prints each number followed by a space
    }
    std::cout << "\n"; // Prints a newline at the end

    return 0; // Returns 0, indicating successful execution
}
```

This code sorts a vector of integers in descending order using a traditional comparison function. It begins by including the necessary libraries: `<iostream>` for input and output, `<vector>` for dynamic arrays, and `<algorithm>` for sorting operations. The `compare` function is defined to take two integers, returning `true` if the first integer is greater than the second, setting the sorting order to descending.

In the `main` function, a vector named `numbers` is initialized with the integers `{1, 3, 2, 5, 4}`. The `std::sort` function is called on this vector, using the `compare` function to sort the elements from highest to lowest. After sorting, a `for` loop iterates through the vector, printing each number followed by a space. The program ends with a newline to cleanly finish the output. This code is a simple and direct example of using a custom function to sort data in C++. Now, let's see the same code using lambda functions and other competitive programming tricks.

```cpp
#include <iostream>  // Includes the input/output stream library for console operations
#include <vector>    // Includes the vector library for using dynamic arrays
#include <algorithm> // Includes the algorithm library for functions like sort

#define ALL(vec) vec.begin(), vec.end() // Macro to simplify passing the entire range of a vector
using VI = std::vector<int>; // Alias for vector<int> to simplify code and improve readability

int main() {
    VI num = {1, 3, 2, 5, 4}; // Initializes a vector of integers using the alias VI

    // Sorts the vector in descending order using a lambda function
    std::sort(ALL(num), [](int a, int b) { return a > b; });

    // Prints the sorted vector
    for (int n : num) {
        std::cout << n << " "; // Prints each number followed by a space
    }
    std::cout << "\n"; // Prints a newline at the end

    return 0; // Returns 0, indicating successful execution
}
```

To see the typing time gain, just compare the normal definition of the `compare` function followed by its usage with the use of the lambda function.

This code sorts a vector of integers in descending order using a lambda function, a modern and concise way to define operations directly in the place where they are needed. It starts by including the standard libraries for input/output, dynamic arrays, and algorithms. The macro `ALL(vec)` is defined to simplify the use of `vec.begin(), vec.end()`, making the code cleaner and shorter.

An alias `VI` is used for `std::vector<int>`, reducing the verbosity when declaring vectors. Inside the `main` function, a vector named `num` is initialized with the integers `{1, 3, 2, 5, 4}`. The `std::sort` function is called to sort the vector, using a lambda function `[](int a, int b) { return a > b; }` that sorts the elements in descending order.

The lambda is defined and used inline, removing the need to declare a separate function like `compare`. After sorting, a `for` loop prints each number followed by a space, ending with a newline. This approach saves time and keeps the code concise, highlighting the effectiveness of lambda functions in simplifying tasks that would otherwise require traditional function definitions.

> Lambda functions in C++, introduced in C++11, are anonymous and defined where they are needed. They shine in short, temporary tasks like inline calculations or _callbacks_. Unlike regular functions, lambdas can capture variables from their surrounding scope. With C++20, lambdas became even more powerful and flexible, extending their capabilities beyond simple operations.
>
> The general syntax for a lambda function in C++ is as follows:
>
> ```cpp
> [capture](parameters) -> return_type { // function body};
> ```
>
> Where:
>
> - `Capture`: Specifies which variables from the surrounding scope can be used inside the lambda. Variables can be captured by value `[=]` or by reference `[&]`. You can also specify individual variables, such as `[x]` or `[&x]`, to capture them by value or reference, respectively.
>   `Parameters`: The input parameters for the lambda function, similar to function arguments.
>   `Return Type`: Optional in most cases, as C++ can infer the return type automatically. However, if the return type is ambiguous or complex, it can be specified explicitly using `-> return_type`.
> - `Body`: The actual code to be executed when the lambda is called.
>
> C++20 brought new powers to lambdas. Now, they work in **immediate functions** with `consteval`, running faster at compile-time. Lambdas can be default-constructed without capturing anything. They can also use **template parameters**, making them more flexible and generic. Let’s see some examples.
>
> Example 1: Basic Lambda Function: A simple example of a lambda function that sums two numbers:
>
> ```cpp
> auto sum = [](int a, int b) -> int {return a + b;};
> std::cout << sum(5, 3);  // Outputs: 8
> ```
>
> Example 2: Lambda with Capture: In this example, a variable from the surrounding scope is captured by value:
>
> ```cpp
>  int x = 10; // Initializes an integer variable x with the value 10
>
>  // Defines a lambda function that captures x by value (creates a copy of x)
>  auto multiply = [x](int a) {return x * a;}; // Multiplies the captured value of x by the argument a
>
>  std::cout << multiply(5);  // Calls the lambda with 5; Outputs: 50
> ```
>
> Here, the lambda captures `x` by value and uses it in its body. This means `x` is copied when the lambda is created. The lambda holds its own version of `x`, separate from the original. Changes to `x` outside the lambda won’t affect the copy inside. It’s like taking a snapshot of `x` at that moment. The lambda works with this snapshot, keeping the original safe and unchanged. But this copy comes at a cost—extra time and memory are needed. For simple types like integers, it’s minor, but for larger objects, the overhead can add up.
>
> Example 3: Lambda with Capture by Reference: In this case, the variable `y` is captured by reference, allowing the lambda to modify it:
>
> ```cpp
> int y = 20;  // Initializes an integer variable y with the value 20
>
> // Defines a lambda function that captures y by reference (no copy is made)
> auto increment = [&y]() {
>     y++;  // Increments y directly
> };
>
> increment();  // Calls the lambda, which increments y
> std::cout << y;  // Outputs: 21
>
> ```
>
> In this fragment, there’s no extra memory or time cost. The lambda captures `y` by reference, meaning it uses the original variable directly. No copy is made, so there’s no overhead. When `increment()` is called, it changes `y` right where it lives. The lambda works with the real `y`, not a snapshot, so any change happens instantly and without extra resources. This approach keeps the code fast and efficient, avoiding the pitfalls of capturing by value. The result is immediate and uses only what’s needed. **In competitive or high-performance programming, we capture by reference. It's faster and uses less memory**.
>
> Example 4: Generic Lambda Function with C++20: With C++20, lambdas can now use template parameters, making them more generic:
>
> ```cpp
> // Defines a generic lambda function using a template parameter <typename T>
> // The lambda takes two parameters of the same type T and returns their sum
> auto generic_lambda = []<typename T>(T a, T b) { return a + b; };
>
> std::cout << generic_lambda(5, 3);      // Calls the lambda with integers, Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Calls the lambda with doubles, Outputs: 4.0
> ```
>
> This code defines a generic lambda using a template parameter, a feature from C++20. The lambda accepts two inputs of the same type `T` and returns their sum. It’s flexible—first, it adds integers, then it adds doubles. The power of this lambda is in its simplicity and versatility. It’s short, clear, and works with any type as long as the operation makes sense. C++20 lets you keep your code clean and adaptable, making lambdas more powerful than ever. And it doesn’t stop there.
>
> Default-constructed lambdas: In C++20, lambdas that don’t capture variables can be default-constructed. You can create them, assign them, and save them for later without calling them right away. This makes it easy to store and pass lambdas when you need a default behavior.
>
> ```cpp
> #include <iostream>
> #include <vector>
> #include <algorithm>
>
> #define ALL(vec) vec.begin(), vec.end() // Macro to simplify passing the entire range of a vector
> using VI = std::vector<int>; // Alias for vector<int> to simplify code and improve readability
>
> // Define a default-constructed lambda that prints a message
> auto print_message = []() {std::cout << "Default behavior: Printing message." << "\n";};
>
> int main() {
>     // Store the default-constructed lambda and call it later
>     print_message();
>
>     // Define a vector and use the lambda as a fallback action
>     VI num = {1, 2, 3, 4, 5};
>
>     // If vector is not empty, do something; else, call the default lambda
>     if (!num.empty()) {
>         std::for_each(ALL(num), [](int n) {std::cout << n * 2 << " ";});  // Prints double of each number
>     } else {
>         print_message(); // Calls the default lambda if no numbers to process
>     }
>
> return 0;
> }
> ```
>
> **This feature lets you set up lambdas for later use (deferred execution)**. In the last code, the lambda `print_message` is default-constructed. It captures nothing and waits until it’s needed. The main function shows this in action. If the vector has numbers, it doubles them. If not, it calls the default lambda and prints a message. C++20 makes lambdas simple and ready for action, whenever you need them.
>
> Immediate lambdas: C++20 brings in `consteval`, a keyword that forces functions to run at compile-time. With lambdas, this means the code is executed during compilation, and the result is set before the program starts. When a lambda is used in a `consteval` function, it must run at compile-time, making your code faster and results predictable.
>
> **In programming competitions, `consteval` lambdas are rarely useful**. Contests focus on runtime performance, not compile-time tricks. Compile-time evaluation doesn’t give you an edge when speed at runtime is what counts. Most problems don’t benefit from execution before the program runs; the goal is to be fast during execution.
>
> `Consteval` ensures the function runs only at compile-time. If you try to use a `consteval` function where it can’t run at compile-time, you’ll get a compile-time error. It’s strict: no runtime allowed.
>
> ```cpp
> consteval auto square(int x) {
>     return [] (int y) { return y * y; }(x);
> }
> int value = square(5);  // Computed at compile-time
> ```
>
> In this example, the lambda inside the `square` function is evaluated at compile-time, producing the result before the program starts execution. **Programming contests focus on runtime behavior and dynamic inputs, making `consteval` mostly useless**. In contests, you deal with inputs after the program starts running, so compile-time operations don’t help. The challenge is to be fast when the program is live, not before it runs.
>
> Finally, we have template lambdas. C++20 lets lambdas take template parameters, making them generic. They can handle different data types without needing overloads or separate template functions. The template parameter is declared right in the lambda’s definition, allowing one lambda to adapt to any type.
>
> Example:
>
> ```cpp
> auto generic_lambda = []<typename T>(T a, T b) {
>     return a + b;
> };
> std::cout << generic_lambda(5, 3);      // Outputs: 8
> std::cout << generic_lambda(2.5, 1.5);  // Outputs: 4.0
> ```
>
> **Template lambdas are a powerful tool in competitive programming**. They let you write one lambda that works with different data types, saving you time and code. Instead of writing multiple functions for integers, doubles, or custom types, you use a single template lambda. It adapts on the fly, making your code clean and versatile. In contests, where every second counts, this can speed up coding and reduce bugs. You get generic, reusable code without the hassle of writing overloads or separate templates.
>
> _Lambdas are great for quick, one-time tasks. But too many, especially complex ones, can make code harder to read and maintain. In competitive programming, speed often trumps clarity, so this might not seem like a big deal. Still, keeping code readable helps, especially when debugging tough algorithms. Use lambdas wisely._
