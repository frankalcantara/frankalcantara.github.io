---
author: Frank
beforetoc: |-
    [Anterior](2024-09-24-13-Data-Structures.md)
    [Próximo](2024-09-24-15-Dynamic-Programming.md)
categories:
    - Matemática
    - Linguagens Formais
    - Programação
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: false
image: assets/images/prog_dynamic..webp
keywords:
    - Developer Tips
lastmod: 2025-05-06T11:04:18.059Z
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Competitive Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: competitive-programming-techniques-insights
tags:
    - Practical Programming Guide
title: Dynamic Programming (DP)
toc: true
---

# Dynamic Programming (DP)


**This is a work in progress, we will get there sooner or later.**


## Knapsack Problem

Select items to maximize a total value without exceeding a capacity. Variations include 0/1 Knapsack, fractional knapsack, and bounded knapsack.

## Longest Increasing Subsequence

Find the longest subsequence of a sequence where the elements are in increasing order. The time complexity can be reduced to $O(n \log n)$ using binary search in combination with dynamic programming.

## Grid Pathfinding

DP-based grid traversal problems, such as finding the minimum or maximum cost path from one corner of a grid to another, often appear.

# Dynamic Programming

Dynamic Programming is a different way of thinking when it comes to solving problems. Programming itself is already a different way of thinking, so, to be honest, I can say that Dynamic Programming is a different way within a different way of thinking. And, if you haven't noticed yet, there is a concept of recursion trying to emerge in this definition.

The general idea is that you, dear reader, should be able to break a large and difficult problem into small and easy pieces. This involves storing and reusing information within the algorithm as needed.

It is very likely that you, kind reader, have been introduced to Dynamic Programming techniques while studying algorithms without realizing it. So, it is also very likely that you will encounter, in this text, algorithms you have seen before without knowing they were Dynamic Programming.

My intention is to break down the Dynamic Programming process into clear steps, focusing on the solution algorithm, so that you can understand and implement these steps on your own whenever you face a problem in technical interviews, production environments, or programming competitive programmings. Without any hesitation, I will try to present performance tips and tricks in C++. However, this should not be considered a limitation; we will prioritize understanding the algorithms before diving into the code, and you will be able to implement the code in your preferred programming language.

I will be using functions for all the algorithms I study primarily because it will make it easier to measure and compare the execution time of each one, even though I am aware of the additional computational overhead associated with function calls. After studying the problems in C++ and identifying the solution with the lowest complexity, eventually, we will also explore the best solution in C. Additionally, whenever possible, we will examine the most popular solution for the problem in question that I can find online.

Some say that Dynamic Programming is a technique to make recursive code more efficient. If we look at Dynamic Programming, we will see an optimization technique that is based on recursion but adds storage of intermediate results to avoid redundant calculations. _Memoization and tabulation are the two most common Dynamic Programming techniques_, each with its own approach to storing and reusing the results of subproblems:

- **Memoization (Top-Down)**: _This technique is recursive in nature_. It involves storing the results of expensive function calls and returning the cached result when the same inputs occur again. This approach can be seen as an optimization of the top-down recursive process.
- **Tabulation (Bottom-Up**): _Tabulation takes an iterative approach, solving smaller subproblems first and storing their solutions in a table (often an array or matrix)_. It then uses these stored values to calculate the solutions to larger subproblems, gradually building up to the final solution. The iterative nature of tabulation typically involves using loops to fill the table in a systematic manner.

Throughout our exploration of Dynamic Programming concepts, we've been using Python as a form of pseudocode. Its versatility and simplicity have served us well, especially considering that many of my students are already familiar with it. Python's readability has made it an excellent choice for introducing and illustrating algorithmic concepts. However, as we progress into more advanced territory, it's time to acknowledge that Python, despite its strengths, isn't the most suitable language for high-performance applications or programming competitive programmings.

With this in mind, we're going to transition to using **C++ 20** as our primary language moving forward. C++ offers superior performance, which is crucial when dealing with computationally intensive tasks often found in competitive programming scenarios. It also provides more direct control over memory management, a feature that can be essential when optimizing algorithms for speed and efficiency. Additionally, we'll occasionally use data structures compatible with **C 17** within our **C++ 20** environment, ensuring a balance between modern features and broader compatibility.

For our development environment, we'll be using Visual Studio Community Edition. This robust IDE will allow us to write, compile, and evaluate our C++ code effectively. It offers powerful debugging tools and performance profiling features, which will become increasingly valuable as we delve into optimizing our algorithms.

Despite this shift, we won't be discarding the work we've done so far. To maintain consistency and provide a bridge between our previous discussions and this new approach, I'll be converting the functions we originally wrote in Python to C++.

As we make this transition, we'll gradually introduce C++ specific optimizations and techniques, broadening your understanding of Dynamic Programming implementation across different language paradigms. I hope this approach will equip you with both a solid conceptual foundation and the practical skills needed for high-performance coding.

**Example 4: Fibonacci in C++ using `std::vectors`**:

Let's begin with a straightforward, naive implementation in **C++20**.

```Cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <functional>

// Recursive function to calculate Fibonacci
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

// Recursive function with memoization to calculate Fibonacci
int fibonacci_memo(int n, std::unordered_map<int, int>& memo) {
    if (memo.find(n) != memo.end()) {
        return memo[n];
    }
    if (n <= 1) {
        return n;
    }
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo);
    return memo[n];
}

// Iterative function with tabulation to calculate Fibonacci
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Function to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}

int main() {

    const int iterations = 1000;
    std::vector<int> test_cases = { 10, 20, 30 };

    for (int n : test_cases) {
        std::cout << "Calculating Fibonacci(" << n << ")\n";

        // Calculation and average time using the simple recursive function
        long long avg_time_recursive = average_time(fibonacci, iterations, n);
        std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";

        // Calculation and average time using the memoization function
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        long long avg_time_memo = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Average time for memoized Fibonacci: " << avg_time_memo << " ns\n";

        // Calculation and average time using the tabulation function
        long long avg_time_tabulation = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Average time for tabulated Fibonacci: " << avg_time_tabulation << " ns\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
```

_Code 1 - Running `std::vector` and tail recursion_{: class="legend"}

The Code 1 demonstrates not only our Fibonacci functions but also two functions for calculating execution time (`long long measure_time(Func func, Args&&... args)` and `long long measure_time(Func func, Args&&... args)`). From this point forward, I will be using this code model to maintain consistent computational cost when calculating the average execution time of the functions we create. This approach will ensure that our performance measurements are standardized across different implementations, allowing for more accurate comparisons as we explore various Dynamic Programming techniques.

Now, the attentive reader will agree with me: we must to break this code down.

## The Recursive Function

Let's start with `fibonacci(int n)`, the simple and pure tail recursive function.

```Cpp
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
```

_Code Fragment 1A - C++ Tail Recursion Function_{: class="legend"}

This is a similar C++ recursive function to the one we used to explain recursion in Python. Perhaps the most relevant aspect of `fibonacci(int n)` is its argument: `int n`. Using the `int` type limits our Fibonacci number to $46$. Especially because the `int` type on my system, a 64-bit computer running Windows 11, is limited, by default, to storing a maximum value of $2^31 - 1 = 2,147,483,647$, and the $46$th Fibonacci number is $1,836,311,903$. The 47th Fibonacci number will be bigger will be bigger than `int` capacity.

The next function is the C++ memoization version:

### The Dynamic Programming Function Using Memoization

```Cpp
// Recursive function with memoization to calculate Fibonacci
int fibonacci_memo(int n, std::unordered_map<int, int>& memo) {
    if (memo.find(n) != memo.end()) {
        return memo[n];
    }
    if (n <= 1) {
        return n;
    }
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo);
    return memo[n];
}
```

_Code Fragment 2A - C++ Memoization Function_{: class="legend"}

Let's highlight the `std::unordered_map<int, int>& memo` in function arguments. The argument `std::unordered_map<int, int>& memo` in C++ is used to pass a reference to an unordered map (hash table) that maps integers to integers. Breaking it down we will have:

> `std::unordered_map` is a template class provided by the C++ Standard Library (STL) that implements a hash table. The template parameters `<int, int>` specify that the keys and values stored in the unordered map are both integers.

The ampersand (`&`) indicates that the argument is a reference. This means that the function will receive a reference to the original unordered map, rather than a copy of it. _Passing by reference is efficient because it avoids copying the entire map, which could be expensive in terms of time and memory, especially for large maps_. Pay attention: Thanks to the use of `&`, all changes made to the map inside the function will affect the original map outside the function. Finally `memo` is the identifier of the parameter which type is `std::unordered_map<int, int>`.In the context of memoization (hence the name `memo` we used earlier), this unordered map is used to store the results of previously computed values to avoid redundant calculations.

> One `unordered_map` in C++ is quite similar to Python's dict in terms of functionality. Both provide an associative container that allows for efficient key-value pair storage and lookup. The `std::unordered_map` is a template class and a C++ only construct implemented as a hash table. _Unordered maps store key-value pairs and provide average constant-time complexity for insertion, deletion, and lookup operations, thanks to their underlying hash table structure_. They grow dynamically as needed, managing their own memory, which is freed upon destruction. Unordered maps can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements.
>
> Unlike arrays, unordered maps do not decay to pointers, and you cannot get a pointer to their internal data. Instead, unordered maps maintain an internal hash table, which is allocated dynamically by the allocator specified in the template parameter, usually obtaining memory from the freestore (heap) independently of the object's actual allocation. This makes unordered maps efficient for fast access and manipulation of key-value pairs, though they do not maintain any particular order of the elements.
>
> Unordered maps do not require a default constructor for stored objects and are well integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, unordered maps rehash their elements, which involves reassigning the elements to new buckets based on their hash values. This rehashing process can involve copying or moving (in C++11 and later) the elements to new locations in memory.
>
> Rehashing, is the process used in `std::unordered_map` to maintain efficient performance by redistributing elements across a larger array when the load factor (the number of elements divided by the number of buckets) becomes too high. The rehashing process involves determining the new size, allocating a new array of buckets to hold the redistributed elements, rehashing elements by applying a hash function to each key to compute a new bucket index and inserting the elements into this new index, and finally, updating the internal state by updating internal pointers, references, and variables, and deallocating the old bucket array. Rehashing in `std::unordered_map` is crucial for maintaining efficient performance by managing the load factor and ensuring that hash collisions remain manageable.

Overall, `std::unordered_map` is a versatile and efficient container for associative data storage, offering quick access and modification capabilities while seamlessly integrating with the C++ Standard Library and, for our purposes, very similar to Python's dictionary

The `fibonacci_memo(int n, std::unordered_map<int, int>& memo)` function works just like the Python function we explained before with the same complexity, $O(n)$, for space and time. That said we can continue to `fibonacci_tabulation(int n)`.

Memoization offers significant advantages over the simple recursive approach when implementing the Fibonacci sequence. The primary benefit is improved time complexity, reducing it from exponential $O(2^n)$ to linear $O(n)$. This optimization is achieved by storing previously computed results in a hash table (memo), eliminating redundant calculations that plague the naive recursive method. This efficiency becomes increasingly apparent for larger $n$ values, where the simple recursive method's performance degrades exponentially, while the memoized version maintains linear time growth. Consequently, memoization allows for the computation of much larger Fibonacci numbers in practical time frames.

### The Dynamic Programming Function Using Tabulation

The `fibonacci_tabulation(int n)`, which uses a `std::vector`, was designed to be as similar as possible to the tabulation function we studied in Python.

```CPP
// Iterative function with tabulation to calculate Fibonacci
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code 1C - C++ Tabulation Function_{: class="legend"}

> The `std::vector` is a template class and a C++ only construct implemented as a dynamic array. _Vectors grow and shrink dynamically, automatically managing their memory, which is freed upon destruction. They can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements_.
>
> Unlike arrays, vectors do not decay to pointers, but you can explicitly get a pointer to their data using `&vec[0]`. Vectors maintain their size (number of elements currently stored) and capacity (number of elements that can be stored in the currently allocated block) along with the internal dynamic array. This internal array is allocated dynamically by the allocator specified in the template parameter, _usually obtaining memory from the freestore (heap) independently of the object's actual allocation_. Although this can make vectors less efficient than regular arrays for small, short-lived, local arrays, vectors do not require a default constructor for stored objects and are better integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, vectors copy (or move, in C++11) their objects.

Besides the `std::vector` template type, the time and space complexity are the same, $O(n)$, we found in Python version. What left us with the generic part of Code 1: Evaluation.

### Performance Evaluation and Support Functions

All the effort we have made so far will be useless if we are not able to measure the execution times of these functions. In addition to complexity, we need to observe the execution time. This time will depend on the computational cost of the structures used, the efficiency of the compiler, and the machine on which the code will be executed. I chose to find the average execution time for calculating the tenth, twentieth, and thirtieth Fibonacci numbers. To find the average, we will calculate each of them 1000 times. For that, I created two support functions:

```Cpp
// Function 1: to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function 2: to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}
```

_Code Fragment 7 - Support Functions for Time Execution Measurement_{: class="legend"}

Let's initiate with function `long long average_time(Func func, int iterations, Args&&... args)`. This function is a template function designed to measure the execution time of a given function `func` with arbitrary arguments `Args&&... args`. It returns the time taken to execute the function in nanoseconds. Let's break down each part of this function to understand how it works in detail.

```cpp
template <typename Func, typename... Args>
```

The keyword `template` in `measure_time` declaration indicates that `measure_time` is a template function, which means it can operate with generic types.

> A template is a C++ language feature that allows functions and classes to operate with generic types, enabling code reuse and type safety, allowing the creation of functions and classes that can work with any data type without being rewritten for each specific type. This is achieved by defining a blueprint that specifies how the function or class should operate with type parameters that are provided when the template is instantiated. The advantage of templates is their ability to provide high levels of abstraction while maintaining performance, as template code is resolved at compile time, resulting in optimized and type-safe code. This leads to more flexible and reusable code structures, reducing redundancy and the potential for errors, and allowing developers to write more generic and maintainable code.

The first argument, `typename Func` specifies that the first template parameter, `Func`, can be any callable type, such as functions, function pointers, lambdas, or functors. When `typename Func` is specified in a template definition, it indicates that the template will accept a callable entity as a parameter. The use of `typename` in this context ensures that the template parameter is interpreted as a type, enabling the compiler to correctly process the callable type during instantiation. I am using `Func` to call the function whose execution time will be measured.

The last argument, `typename... Args`: This is a variadic template parameter, allowing the function to accept any number of additional arguments of any types.

> The `typename... Args` declaration is used in C++ templates to define a variadic template parameter, which allows a template to accept an arbitrary number of arguments of any types. When `typename... Args` is specified, it indicates that the template can handle a variable number of parameters, making it highly flexible and adaptable. This is particularly useful for functions and classes that need to operate on a diverse set of inputs without knowing their types or number in advance.

The line `auto start = std::chrono::high_resolution_clock::now();` is a crucial component in precise time measurement in C++. It utilizes the C++ Standard Library's `<chrono>` library, which provides a set of time utilities.

> The `std::chrono::high_resolution_clock` is a clock type that represents the clock with the smallest tick period available on the system. The `now()` function is a static member function of this clock class that returns the current time point. By calling `now()`, we capture the exact moment in time when this line is executed.
>
> The auto keyword is used here for type inference, allowing the compiler to automatically deduce the type of the start variable. In this case, start will be of type `std::chrono::time_point<std::chrono::high_resolution_clock>, which represents a point in time as measured by the high-resolution clock. This time point can later be compared with another time point (typically captured after the execution of the code we want to measure) to calculate the duration of the executed code.

In our case, we will compare the `start` time with the `end` time acquired by `auto end = std::chrono::high_resolution_clock::now();`. Between this two lines is `func(std::forward<Args>(args)...);`.

The line `func(std::forward<Args>(args)...);` is a key component of the `measure_time` function. In this context, it serves to execute the function `func` that we aim to measure, while passing along all arguments provided to `measure_time`. This line appears between the two time measurements (`start` and `end`), allowing us to capture the execution time of func with its given arguments. The use of `std::forward` and parameter pack expansion allows `measure_time` to work with functions of any signature, making it a flexible timing utility.

In the context of template functions like `measure_time`, `func` typically represents a callable object. This can be a regular function, a function pointer, a lambda expression, or a function object (functor). The exact type of `func` is deduced by the compiler based on the argument passed to `measure_time`.

> `std::forward` is a utility function template defined in the `<utility>` header of the C++20 Standard Library. Its primary use is in implementing perfect forwarding. `std::forward` preserves the value category (`lvalue` or `rvalue`) of a template function argument when passing it to another function. This allows the called function to receive the argument with the same value category as it was originally passed.
>
> Perfect forwarding allows a function template to pass its arguments to another function while retaining the `lvalue`/`rvalue` nature of the arguments. This is achieved by declaring function parameters as forwarding references (`T&&`) and using `std::forward` when passing these parameters to other functions.

For example:

```cpp
template<class T>
void wrapper(T&& arg) {
    foo(std::forward<T>(arg));
}
```

In this example, wrapper can accept both `lvalues` and `rvalues`, and will pass them to foo preserving their original value category. The combination of forwarding references and `std::forward` enables the creation of highly generic code that can work efficiently with a wide variety of argument types and value categories. This is particularly useful in library design and when creating wrapper functions or function templates that need to preserve the exact characteristics of their arguments when forwarding them to other functions.

However, all this code will not work if we don't take the necessary precautions in the function signature.

```Cpp
long long measure_time(Func func, Args&&... args) {
```

In the context of a template function, `Args&&... args` is often used to perfectly forward these arguments to another function, preserving their value categories (`lvalues` or `rvalues`). The use of `typename...` ensures that each parameter in the pack is treated as a type, enabling the compiler to correctly process each argument during template instantiation.

> An `lvalue` (locator value) represents an object that occupies a specific location in memory (i.e., has an identifiable address). `lvalues` are typically variables or dereferenced pointers. They can appear on the left-hand side of an assignment expression, hence the name `lvalue`.
>
> An `rvalue` (read value) represents a temporary value or an object that does not have a persistent memory location. rvalues are typically literals, temporary objects, or the result of expressions that do not persist. They can appear on the right-hand side of an assignment expression.
>
> C++11 introduced `rvalue` references to boost performance by enabling move semantics. An rvalue reference is declared using `&&`, allowing functions to distinguish between copying and moving resources. This is particularly useful for optimizing the performance of classes that manage resources such as dynamic memory or file handles.

The return type of the function, `long long`, represents the duration of the function execution in nanoseconds. I choose a `long long` integer because I have no idea how long our Dynamic Programming functions will take to compute, and I wanted to ensure a default function that can be used for all problems we will work on. The maximum value that can be stored in a `long long` type in C++ is defined by the limits of the type, which are specified in the `<climits>` header. For a signed `long long` type, the maximum value is $2^{63} - 1 = 9,223,372,036,854,775,807$.

The function `measure_time` arguments are:

- `Func func`: The callable entity whose execution time we want to measure.
- `Args&&... args: A parameter pack representing the arguments to be forwarded to the callable entity. The use of && indicates that these arguments are perfect forwarded, preserving their value category (`lvalue`or`rvalue`).

As we saw before, the the body of function `measure_time` starts with:

```Cpp
auto start = std::chrono::high_resolution_clock::now();
```

Where `auto start` declares a variable `start` to store the starting time point and
`std::chrono::high_resolution_clock::now()` retrieves the current time using a high-resolution clock, which provides the most accurate and precise measurement of time available on the system. The `std::chrono::high_resolution_clock::now()` returns a `time_point` object representing the current point in time.

> In C++, a `time_point` object is a part of the `<chrono>` library, which provides facilities for measuring and representing time. A `time_point` object represents a specific point in time relative to a clock. It is templated on a clock type and a duration, allowing for precise and high-resolution time measurements. The Clock is represented by a `clock type`, and can be system_clock, steady_clock, or high_resolution_clock. The clock type determines the epoch (the starting point for time measurement) and the tick period (the duration between ticks).

Following we have the function call:

```Cpp
func(std::forward<Args>(args)...);
```

`func`: Calls the function or callable entity passed as the `func` parameter while `std::forward<Args>(args)...` forwards the arguments to the function call. This ensures that the arguments are passed to the called function, `func` with the same value category (`lvalue` or `rvalue`) that they were passed to `measure_time`.

We measure the time and store it in `start`, then we call the function. Now we need to measure the time again.

```Cpp
auto end = std::chrono::high_resolution_clock::now();
```

In this linha `auto end` declares a variable `end` to store the ending time point while `std::chrono::high_resolution_clock::now()` retrieves the current time again after the function `func` has completed execution. Finally we can calculate the time spent to call the function `func`.

```Cpp
std::chrono::duration<long long, std::nano> duration = end - start;
```

Both the `start` and `end` variables store a `time_point` object. `std::chrono::duration<long long, std::nano>` represents a duration in nanoseconds. `end - start` calculates the difference between the ending and starting time points, which gives the duration of the function execution.

> In C++, the `<chrono>` library provides a set of types and functions for dealing with time and durations in a precise and efficient manner. One of the key components of this library is the std::chrono::duration class template, which represents a time duration with a specific period.

The `std::chrono::duration<long long, std::nano>` declaration can be break down as:

- `std::chrono`: This specifies that the `duration` class template is part of the `std::chrono` namespace, which contains types and functions for time utilities.
- `duration<long long, std::nano>`: The `long long` is the representation type (`Rep`) of the `std::chrono::duration` class template, which is the type used to store the number of ticks (e.g., `int`, `long`, `double`). It indicates that the number of ticks will be stored as a `long long` integer, providing a large range to accommodate very fine-grained durations.
- `std::nano` is the period type (`Period`) of the `std::chrono::duration` class template. The period type represents the tick period (e.g., seconds, milliseconds, nanoseconds). The default is `ratio<1>`, which means the duration is in seconds. `std::ratio` is a template that represents a compile-time rational number. The `std::nano` is a `typedef` for `std::ratio<1, 1000000000>`, which means each tick represents one nanosecond.

The last line is:

```cpp
return duration.count();
```

Where `duration.count()` returns the count of the duration in nanoseconds as a `long long` value, which is the total time taken by `func` to execute.

Whew! That was long and exhausting. I'll try to be more concise in the future. I had to provide some details because most of my students are familiar with Python but have limited knowledge of C++.

The next support function is Function 2, `long long average_time(Func func, int iterations, Args&&... args)`:

```Cpp
// Function 2: to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}
```

_Code Fragment 8 - Average Time Function_{: class="legend"}

The `average_time` function template was designed to measure and calculate the average execution time of a given callable entity, such as a function, lambda, or functor, over a specified number of iterations. The template parameters `typename Func` and `typename... Args` allow the function to accept any callable type and a variadic list of arguments that can be forwarded to the callable. The function takes three parameters: the callable entity `func`, the number of iterations `iterations`, and the arguments `args` to be forwarded to the callable. Inside the function, a variable, `total_time`, is initialized to zero to accumulate the total execution time. A loop runs for the specified number of iterations, and during each iteration, the `measure_time` function is called to measure the execution time of `func` with the forwarded arguments, which is then added to `total_time`.

After the loop completes, `total_time` contains the sum of the execution times for all iterations. The function then calculates the average execution time by dividing `total_time` by the number of iterations and returns this value. This approach ensures that the average time provides a more reliable measure of the callable's performance by accounting for variations in execution time across multiple runs. The use of `std::forward<Args>(args)...` in the call to `measure_time` ensures that the arguments are forwarded with their original value categories, maintaining their efficiency and correctness. I like to think that `average_time()` provides a robust method for benchmarking the performance of callable entities in a generic and flexible manner.

I said I would be succinct! Despite all the setbacks, we have reached the
`int main()`:

```Cpp
int main() {

    const int iterations = 1000;
    std::vector<int> test_cases = { 10, 20, 30 }; //fibonacci numbers

    for (int n : test_cases) {
        std::cout << "Calculating Fibonacci(" << n << ")\n";

        // Calculation and average time using the simple recursive function
        long long avg_time_recursive = average_time(fibonacci, iterations, n);
        std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";

        // Calculation and average time using the memoization function
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        long long avg_time_memo = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Average time for memoized Fibonacci: " << avg_time_memo << " ns\n";

        // Calculation and average time using the tabulation function
        long long avg_time_tabulation = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Average time for tabulated Fibonacci: " << avg_time_tabulation << " ns\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
```

_Code Fragment 9 - C++ `std::vector` main() function_{: class="legend"}

The `main()` function measures and compares the average execution time of different implementations of the Fibonacci function. Here's a detailed explanation of each part:

The program starts by defining the number of iterations (`const int iterations = 1000;`) and a vector of test cases (`std::vector<int> test_cases = { 10, 20, 30 };`). It then iterates over each test case, calculating the Fibonacci number using different methods and measuring their average execution times.

For the memoized Fibonacci implementation, the program first creates an unordered map `memo` to store previously computed Fibonacci values. It then defines a lambda function `fibonacci_memo_wrapper` that captures `memo` by reference and calls the `fibonacci_memo` function. The `average_time` function is used to measure the average execution time of this memoized implementation over 1000 iterations for each test case.

The other functions follow a similar pattern to measure and print their execution times. For instance, in the case of the recursive Fibonacci function, the line `long long avg_time_recursive = average_time(fibonacci, iterations, n);` calls the `average_time` function to measure the average execution time of the simple recursive Fibonacci function over 1000 iterations for the current test case $n$. The result, stored in `avg_time_recursive`, represents the average time in nanoseconds. The subsequent line, `std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";`, outputs this average execution time to the console, providing insight into the performance of the recursive method.

The results are printed to the console, showing the performance gain achieved through memoization compared to the recursive and tabulation methods.

**Running Example 4 - `std::vector`**

Example 4, the simple and intuitive code for testing purposes, finds three specific Fibonacci numbers — the 10th, 20th, and 30th — using three different functions, 1,000 times each. This code uses an `int`, `std::vector`, and `std::unordered_map` for storing the values of the Fibonacci sequence and, when executed, presents the following results.

```shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 660 ns
Average time for memoized Fibonacci: 607 ns
Average time for tabulated Fibonacci: 910 ns
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 75712 ns
Average time for memoized Fibonacci: 444 ns
Average time for tabulated Fibonacci: 1300 ns
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 8603451 ns
Average time for memoized Fibonacci: 414 ns
Average time for tabulated Fibonacci: 1189 ns
-----------------------------------
```

_Output 1 - running Example 4 - std::vector_{: class="legend"}

the careful reader should note that the execution times vary non-linearly and, in all cases, for this problem, the Dynamic Programming version using tabulation was faster. There is much discussion about the performance of the Vector class compared to the Array class. To test the performance differences between `std::vector` and `std::array`, we will retry using `std::array`

**Example 5: using `std::array`**:

First and foremost, `std::array` is a container from the C++ Standard Library with some similarities to, and some differences from, `std::vector`, namely:

> The `std::array` is a template class introduced in C++11, which provides a fixed-size array that is more integrated with the STL than traditional C-style arrays. Unlike `std::vector`, `std::array` does not manage its own memory dynamically; its size is fixed at compile-time, which makes it more efficient for cases where the array size is known in advance and does not change. `std::array` objects can be passed to and returned from functions, and they support copy and assignment operations. They provide the same `begin()`/`end()` methods as vectors, allowing for easy iteration and integration with other STL algorithms. One significant advantage of `std::array` over traditional arrays is that it encapsulates the array size within the type itself, eliminating the need for passing size information separately. Additionally, `std::array` provides member functions such as `size()`, which returns the number of elements in the array, enhancing safety and usability. However, since `std::array` has a fixed size, it does not offer the dynamic resizing capabilities of `std::vector`, making it less flexible in scenarios where the array size might need to change.

When considering performance differences between `std::vector` and `std::array`, it's essential to understand their underlying characteristics and use cases. _`std::array` is a fixed-size array, with its size determined at compile-time, making it highly efficient for situations where the array size is known and constant_. The lack of dynamic memory allocation means that `std::array` avoids the overhead associated with heap allocations, resulting in faster access and manipulation times. This fixed-size nature allows the compiler to optimize memory layout and access patterns, often resulting in better cache utilization and reduced latency compared to dynamically allocated structures.

In contrast, _`std::vector` provides a dynamic array that can grow or shrink in size at runtime, offering greater flexibility but at a cost. The dynamic nature of `std::vector` necessitates managing memory allocations and deallocations, which introduces overhead_. When a `std::vector` needs to resize, it typically allocates a new block of memory and copies existing elements to this new block, an operation that can be costly, especially for large vectors. Despite this, `std::vector` employs strategies such as capacity doubling to minimize the frequency of reallocations, balancing flexibility and performance.

_For small, fixed-size arrays, `std::array` usually outperforms `std::vector` due to its minimal overhead and compile-time size determination_. It is particularly advantageous in performance-critical applications where predictable and low-latency access is required. On the other hand, `std::vector` shines in scenarios where the array size is not known in advance or can change, offering a more convenient and safer alternative to manually managing dynamic arrays.

In summary, `std::array` generally offers superior performance for fixed-size arrays due to its lack of dynamic memory management and the resultant compiler optimizations. However, `std::vector` provides essential flexibility and ease of use for dynamically sized arrays, albeit with some performance trade-offs. The choice between `std::array` and `std::vector` should be guided by the specific requirements of the application, weighing the need for fixed-size efficiency against the benefits of dynamic resizing.

| Feature           | `std::vector`                                    | `std::array`                                   |
| ----------------- | ------------------------------------------------ | ---------------------------------------------- |
| Size              | Dynamic (can change at runtime)                  | Fixed (determined at compile time)             |
| Memory Management | Dynamic allocation on the heap                   | Typically on the stack, no dynamic allocation  |
| Performance       | Can have overhead due to resizing                | Generally more efficient for fixed-size data   |
| Use Cases         | When the number of elements is unknown or varies | When the number of elements is known and fixed |
| Flexibility       | High (can add/remove elements easily)            | Low (size cannot be changed)                   |
| STL Integration   | Yes (works with algorithms and iterators)        | Yes (similar interface to vector)              |

_Tabela 1 - std::vector and std::array comparison_{: class="legend"}

So, we can test this performance advantages, running a code using `std::array`. Since I am lazy, I took the same code used in Example 4 and only replaced the container in the `fibonacci_tabulation` function. You can see it below:

```Cpp
// Iterative function with tabulation to calculate Fibonacci using arrays
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    std::array<int, 41> dp = {};  // array to support up to Fibonacci(40)
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code Fragment 10 - C++, `std::array`, Tabulation Function_{: class="legend"}

This is basically the same code that we discussed in the previous section, only replacing the `std::vector` class with the `std::array` class. Therefore, we do not need to analyze the code line by line and can consider the flowcharts and complexity analysis already performed.

**Example 5: using `std::array`**

Running the Example 5 will produces the following result:

```shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 807 ns
Average time for memoized Fibonacci: 426 ns
Average time for tabulated Fibonacci: 159 ns
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 88721 ns
Average time for memoized Fibonacci: 434 ns
Average time for tabulated Fibonacci: 371 ns
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 10059626 ns
Average time for memoized Fibonacci: 414 ns
Average time for tabulated Fibonacci: 439 ns

```

_Output 2 - Example 5 running std::vector_{: class="legend"}

We have reached an interesting point. Just interesting!

We achieved a performance gain using memoization and tabulation, as evidenced by the different complexities among the recursive $O(n^2)$, memoization $O(n)$, and tabulation $O(n)$. Additionally, we observed a slight improvement in execution time by choosing `std::array` instead of `std::vector`. However, we still have some options to explore. Options never end!

#### Code 3: C-style Array

We are using a C++ container of integers to store the already calculated Fibonacci numbers as the basis for the two Dynamic Programming processes we are studying so far, memoization and tabulation, one `std::unordered_map` and one `std::vector` or `std::array`. However, there is an even simpler container in C++: the array. The C-Style array.

For compatibility, C++ allows the use of code written in C, including data structures, libraries, and functions. So, why not test these data structures? For this, I wrote new code, keeping the functions using `std::array` and `std::unordered_map` and creating two new dynamic functions using C-style arrays. The code is basically the same except for the following fragment:

```Cpp
const int MAXN = 100;
bool found[MAXN] = { false };
int memo[MAXN] = { 0 };

// New function with memoization using arrays
int cArray_fibonacci_memo(int n) {
    if (found[n]) return memo[n];
    if (n == 0) return 0;
    if (n == 1) return 1;

    found[n] = true;
    return memo[n] = cArray_fibonacci_memo(n - 1) + cArray_fibonacci_memo(n - 2);
}

// New function with tabulation using arrays
int cArray_fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    int dp[MAXN] = { 0 };  // array to support up to MAXN
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

_Code Fragment 11 - C++, C-Style Array, Memoization and Tabulation Functions_{: class="legend"}

As I said, this code segment introduces two new functions for calculating Fibonacci numbers using C-style arrays, with a particular focus on the function for memoization. Instead of using an `std::unordered_map` to store the results of previously computed Fibonacci numbers, the memoization function `cArray_fibonacci_memo` uses two arrays: `found` and `memo`. The `found` array is a boolean array that tracks whether the Fibonacci number for a specific index has already been calculated, while the `memo` array stores the calculated Fibonacci values[^1]. The function checks if the result for the given $n$ is already computed by inspecting the `found` array. If it is, the function returns the value from the `memo` array. If not, it recursively computes the Fibonacci number, stores the result in the `memo` array, and marks the `found` array as true for that index. To be completely honest, this idea of using two arrays comes from [this site](<[URL](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html)>).

The `cArray_fibonacci_tabulation` function, on the other hand, implements the tabulation method using a single C-Style array `dp` to store the Fibonacci numbers up to the $n$th value. The function initializes the base cases for the Fibonacci Sequence, with `dp[0]` set to $0$ and `dp[1]` set to $1$. It then iterates from $2$ to $n$, filling in the `dp` array by summing the two preceding values. This iterative approach avoids the overhead of recursive calls, making it more efficient for larger values of $n$.

Again succinct! I think I'm learning. These structures have the same space and time complexities that we have observed since Example 4. In other words, all that remains is to run this code and evaluate the execution times.

#### Running Code 3: using C-Style Array

Will give us the following answer:

```Shell
Calculating Fibonacci(10)
Average time for recursive Fibonacci: 718 ns
Fibonacci(10) = 55
Average time for memoized Fibonacci: 439 ns
Fibonacci(10) = 55
Average time for tabulated Fibonacci: 67 ns
Fibonacci(10) = 55
Average time for new memoized Fibonacci: 29 ns
Fibonacci(10) = 55
Average time for new tabulated Fibonacci: 72 ns
Fibonacci(10) = 55
-----------------------------------
Calculating Fibonacci(20)
Average time for recursive Fibonacci: 71414 ns
Fibonacci(20) = 6765
Average time for memoized Fibonacci: 449 ns
Fibonacci(20) = 6765
Average time for tabulated Fibonacci: 83 ns
Fibonacci(20) = 6765
Average time for new memoized Fibonacci: 28 ns
Fibonacci(20) = 6765
Average time for new tabulated Fibonacci: 87 ns
Fibonacci(20) = 6765
-----------------------------------
Calculating Fibonacci(30)
Average time for recursive Fibonacci: 8765969 ns
Fibonacci(30) = 832040
Average time for memoized Fibonacci: 521 ns
Fibonacci(30) = 832040
Average time for tabulated Fibonacci: 102 ns
Fibonacci(30) = 832040
Average time for new memoized Fibonacci: 29 ns
Fibonacci(30) = 832040
Average time for new tabulated Fibonacci: 115 ns
Fibonacci(30) = 832040
-----------------------------------
```

_Output 3: running C-Style array_{: class="legend"}

And there it is, we have found a code fast enough for calculating the nth Fibonacci number in an execution time suitable to my ambitions. The only problem is that we used C-Style arrays in a C++ solution. In other words, we gave up all C++ data structures to make the program as fast as possible. We traded a diverse and efficient language for a simple and straightforward one. This choice will be up to the kind reader. You will have to decide if you know enough C to solve any problem or if you need to use predefined data structures to solve your problems. _Unless there is someone in the competitive programming using C. In that case, it's C and that's it_.

Before we start solving problems with Dynamic Programming, let's summarize the execution time reports in a table for easy visualization and to pique the curiosity of the kind reader.

## Execution Time Comparison Table

| Container          | Number | Recursive (ns) | Memoized (ns) | Tabulated (ns) |
| ------------------ | ------ | -------------- | ------------- | -------------- |
| **Vectors**        | 10     | 660            | 607           | 910            |
|                    | 20     | 75,712         | 444           | 1,300          |
|                    | 30     | 8,603,451      | 414           | 1,189          |
| **Arrays**         | 10     | 807            | 426           | 159            |
|                    | 20     | 88,721         | 434           | 371            |
|                    | 30     | 10,059,626     | 414           | 439            |
| **C-Style Arrays** | 10     | 718            | 29            | 72             |
|                    | 20     | 71,414         | 28            | 87             |
|                    | 30     | 8,765,969      | 29            | 115            |

_Tabela 2 - Code Execution Time Comparison_{: class="legend"}

With sufficient practice, Dynamic Programming concepts will become intuitive. I know, the text is dense and complicated. I purposefully mixed concepts of Dynamic Programming, complexity analysis, C++, and performance. If the kind reader is feeling hopeless, stand up, have a soda, walk a bit, and start again. Like everything worthwhile in life, Dynamic Programming requires patience, effort, and time. If, on the other hand, you feel confident, let's move on to our first problem.

## Your First Dynamic Programming Problem

Dynamic Programming concepts became popular in the early 21st century thanks to job interviews for large companies. Until then, only high-performance and competitive programmers were concerned with these techniques. Today, among others, we have [LeetCode](https://leetcode.com/) with hundreds, perhaps thousands of problems to solve. I strongly recommend trying to solve some of them. Here, I will only solve problems whose solutions are already available on other sites. You might even come across some from LeetCode problem, but that will be by accident. The only utility of LeetCode, for me, for you, and for them, is that the problems are not easy to find or solve. Let's start with a problem that is now a classic on the internet and, according to legend, was part of a Google interview.

### The "Two-Sum" problem

**Statement**: In a technical interview, you've been given an array of numbers, and you need to find a pair of numbers that sum up to a given target value. The numbers can be positive, negative, or both. Can you design an algorithm that works in $O(n)$ time complexity or better?

For example, given the array: `[8, 10, 2, 9, 7, 5]` and the target sum: 11

Your function should return a pair of numbers that add up to the target sum. Your answer must be a function in form: `Values(sequence, targetSum)`, In this case, your function should return (9, 2).

### Brute-Force for Two-Sum's problem

The most obvious solution, usually the first that comes to mind, involves checking all pairs in the array to see if any pair meets the desired target value. This solution is not efficient for large arrays; it has a time complexity of $O(n^2)$ where $n$ is the number of elements in the array. The flow of the Brute-Force function can be seen in Flowchart 4.

![]({{ site.baseurl }}/assets/images/flow4.webp)
_Flowchart 4 - Brute-Force solution for Two-Sum problem_{: class="legend"}

Flowchart 4 enables the creation of a function to solve the two-sum problem in C++20, as can be seen in Code 4 below:

```cpp
#include <iostream>
#include <vector>
#include <utility>

// Function to find a pair of numbers that add up to the target sum Brute-Force
std::pair<int, int> Values(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();

    // Iterate over all possible pairs
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Check if the current pair sums to the target
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]); // Pair found
            }
        }
    }

    // No pair found
    return std::make_pair(-1, -1);
}

int main() {
    // Example usage
    std::vector<int> sequence = { 8, 10, 2, 9, 7, 5 };
    int targetSum = 11;

    // Call the function and print the result
    std::pair<int, int> result = Values(sequence, targetSum);
    if (result.first != -1) {
        std::cout << "Pair found: (" << result.first << ", " << result.second << ")" << std::endl;
    }
    else {
        std::cout << "No pair found." << std::endl;
    }

    return 0;
}
```

_Code 3: Full code of a two-sum using `std::vector`_{: class="legend"}

The Values function is quite simple, but the use of `std::vector` and `std::pair` in Code 4 deserves a closer look. While `std::array` might offer a slight performance edge, the dynamic nature of `std::vector` makes it a better fit for interview and competitive programming scenarios where the size of the input data isn't known in advance. This flexibility is crucial when dealing with data read from external sources like the terminal or text files.

> `std::pair` is a standard template class in C++ used to store a pair of values, which can be of different types. It has been available since C++98 and is defined in the `<utility>` header. This class is particularly useful for returning two values from a function or for storing two related values together. It has two public member variables, `first` and `second`, which store the values. A `std::pair` can be initialized using constructors or the helper function `std::make_pair`.
> The kind reader can create a `std::pair` directly using its constructor (`std::pair<int, double> p1(42, 3.14);`) or by using `std::make_pair` (`auto p2 = std::make_pair(42, 3.14);`). It is straightforward to access the members `first` and `second` directly (`std::cout << "First: " << p1.first << ", Second: " << p1.second << std::endl;`).
> Since C++11, we also have `std::tuple` and `std::tie`, which extend the functionality of `std::pair` by allowing the grouping of more than two values. `std::tuple` can store any number of values of different types, making it more versatile than `std::pair`. The `std::tie` function can be used to unpack a `std::tuple` into individual variables. While `std::pair` is simpler and more efficient for just two values, `std::tuple` provides greater flexibility for functions needing to return multiple values. For example, a `std::tuple` can be created using `std::make_tuple` (`auto t = std::make_tuple(1, 2.5, "example");`), and its elements can be accessed using `std::get<index>(t)`.

The kind reader may have noticed the use of `(-1, -1)` as sentinel values to indicate that the function did not find any pair. There is a better way to do this. Use the `std::optional` class as we can see in Code 5:

```cpp
#include <iostream>
#include <vector>
#include <optional>
#include <utility>

// Function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> Values(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();

    // Iterate over all possible pairs
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Check if the current pair sums to the target
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]); // Pair found
            }
        }
    }

    // No pair found
    return std::nullopt;
}

int main() {
    // Example usage
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5};
    int targetSum = 11;

    // Call the function and print the result
    if (auto result = Values(sequence, targetSum)) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")" << std::endl;
    } else {
        std::cout << "No pair found." << std::endl;
    }

    return 0;
}
```

_Code 4: Full code of a two-sum using `std::vector` and `std::optional`_{: class="legend"}

> `std::optional` is a feature introduced in C++17 that provides a way to represent optional (or nullable) values. It is a template class that can contain a value or be empty, effectively indicating the presence or absence of a value without resorting to pointers or sentinel values. This makes `std::optional` particularly useful for functions that may not always return a meaningful result. By using `std::optional`, developers can avoid common pitfalls associated with null pointers and special sentinel values, thereby writing safer and more expressive code. `std::optional` is similar to the `Maybe` type in Haskell, which also represents an optional value that can either be `Just` a value or `Nothing`. An equivalent in Python is the use of `None` to represent the absence of a value, often combined with the `Optional` type hint from the `typing` module to indicate that a function can return either a value of a specified type or `None`.

Here is an example of how you can use `Optional` from the `typing` module in Python to represent a function that may or may not return a value:

```python
from typing import Optional

def find_min_max(numbers: list[int]) -> Optional[tuple[int, int]]:
    if not numbers:
        return None

    min_val = min(numbers)
    max_val = max(numbers)

    return min_val, max_val
```

_Code Fragment 12 - Optional implemented in Python_{: class="legend"}

Relying solely on brute-force solutions won't impress interviewers or win coding competitive programmings. It's crucial to strive for solutions with lower time complexity whenever possible. While some problems might not have more efficient alternatives, most interview and competitive programming questions are designed to filter out candidates who only know brute-force approaches.

#### Recursive Approach: Divide and Conquer

The recursive solution leverages a two-pointer approach to efficiently explore the array within a dynamically shrinking window defined by the `start` and `end` indices. It operates by progressively dividing the search space into smaller subproblems, each represented by a narrower window, until a base case is reached or the target sum is found. Here's the refined description, flowchart and code:

#### Base Cases

1. **Empty **Input**:** If the array is empty (or if the `start` index is greater than or equal to the `end` index), there are no pairs to consider. In this case, we return `std::nullopt` to indicate that no valid pair was found.

2. **Target Sum Found:** If the sum of the elements at the current `start` and `end` indices equals the `target` value, we've found a matching pair. We return this pair as `std::optional<std::pair<int, int>>` to signal success and provide the result.

#### Recursive Step

1. **Explore Leftward:** We make a recursive call to the function, incrementing the `start` index by one. This effectively shifts our focus to explore pairs that include the next element to the right of the current `start` position.

2. **Explore Rightward (If Necessary):** If the recursive call in step 1 doesn't yield a solution, we make another recursive call, this time decrementing the `end` index by one. This shifts our focus to explore pairs that include the next element to the left of the current `end` position.

This leads us to the illustration of the algorithm in Flowchart 4 and its implementation in C++ Code 5:

![]({{ site.baseurl }}/assets/images/twoSumRecur.webp)
_Flowchart 4 - Two-Sum problem recursive solution_{: class="legend"}

```Cpp
#include <vector>
#include <optional>
#include <iostream>

// Recursive function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    // Base case: If start index is greater than or equal to end index, no pair is found
    if (start >= end) {
        return std::nullopt; // Return no value (null optional)
    }
    // Base case: If the sum of elements at start and end indices equals the target, pair is found
    if (arr[start] + arr[end] == target) {
        return std::make_optional(std::make_pair(arr[start], arr[end])); // Return the pair
    }
    // Recursive call: Move the start index forward to check the next element
    auto result = findPairRecursively(arr, target, start + 1, end);
    if (result) {
        return result;   // If a pair is found in the recursive call, return it
    }
    // Recursive call: Move the end index backward to check the previous element
    return findPairRecursively(arr, target, start, end - 1);
}

// Function to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> Values(const std::vector<int>& sequence, int targetSum) {
    // Call the recursive function with initial indices (0 and size-1)
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

int main() {
    // Example usage
    std::vector<int> sequence = { 8, 10, 2, 9, 7, 5 }; // Input array
    int targetSum = 11; // Target sum

    // Call the function to find the pair
    auto result = Values(sequence, targetSum);

    // Print the result
    if (result) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")\n";
    } else {
        std::cout << "No pair found.\n";
    }
    return 0;
}
```

_Code 5: Full code of a two-sum using a recursive function_{: class="legend"}

#### Solution Analysis

The recursion systematically explores all possible pairs in the array by moving the start and end indices in a controlled manner. With each recursive call, the problem is reduced until one of the base cases is reached.

The `std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end)` recursive function explores all possible pairs in the array by moving the `start` and `end` indices. Let's analyze its time complexity:

1. **Base Case**: The base case occurs when `start` is greater than or equal to `end`. In the worst case, this happens after exploring all possible pairs.

2. **Recursive Calls**: For each pair `(start, end)`, there are two recursive calls:
   - One that increments the `start` index.
   - Another that decrements the `end` index.

Given an array of size `n`, the total number of pairs to explore is approximately `n^2 / 2` (combinatorial pairs). Since each recursive call reduces the problem size by one element, the number of recursive calls can be modeled as a binary tree with a height of `n`, leading to a total of `2^n` calls in the worst case.

Therefore, the recursive function exhibits a time complexity of $O(2^n)$. This exponential complexity arises because each unique pair of indices in the array triggers recursive calls, leading to a rapid increase in computation time as the array size grows. This makes the recursive approach impractical for handling large arrays.

The space complexity, however, is determined by the maximum depth of the recursion stack:

1. **Recursion Stack Depth**: In the worst-case scenario, the recursive function will delve to a depth of $n$. This happens when each recursive call processes a single element and spawns further recursive calls until it reaches the base case (when a single element remains).

2. **Auxiliary Space**: Besides the space occupied by the recursion stack, no other substantial amount of memory is utilized.

Consequently, the recursive function's space complexity is $O(n)$, where $n$ denotes the array size. This linear space complexity results from the recursion stack, which stores information about each function call.

In summary, we have the following characteristics:

- **Time Complexity**: $O(2^n)$;
- **Space Complexity**: $O(n)$.

While the recursive approach proves to be highly inefficient in terms of time complexity, rendering it unsuitable for large inputs, it's crucial to compare its performance with the previously discussed brute-force solutions to gain a comprehensive understanding of its strengths and weaknesses.

The Brute-Force solution to the two-sum problem involves checking all possible pairs in the array to see if any pair meets the desired target value. This approach has a time complexity of $O(n^2)$ because it uses nested loops to iterate over all pairs. The space complexity is $O(1)$ as it does not require additional storage beyond the input array and a few variables.

On the other hand, the recursive solution systematically explores all possible pairs by moving the `start` and `end` indices. Although it achieves the same goal, its time complexity is much worse, at $O(2^n)$. This exponential complexity arises because each recursive call generates two more calls, leading to an exponential growth in the number of calls. The space complexity of the recursive solution is $O(n)$, as it requires a recursion stack that can grow up to the depth of the array size.

In summary, while both approaches solve the problem, the Brute-Force solution is significantly more efficient in terms of time complexity ($O(n^2)$ vs. $O(2^n)$), and it also has a lower space complexity ($O(1)$ vs. $O(n)$). However, we are not interested in either of these solutions. The Brute-Force solution is naive and offers no advantage, and the recursive solution is impractical. Thus, we are left with the Dynamic Programming solutions.

#### Dynamic Programming: memoization

> Regardless of the efficiency of the recursive code, the first law of Dynamic Programming says: always start with recursion. Thus, the recursive function will be useful for defining the structure of the code using memoization and tabulation.

Memoization is a technique that involves storing the results of expensive function calls and reusing the cached result when the same inputs occur again. By storing intermediate results, we can avoid redundant calculations, thus optimizing the solution.

In the context of the two-sum problem, memoization can help reduce the number of redundant checks by storing the pairs that have already been evaluated. We'll use a `std::unordered_map` to store the pairs of indices we've already checked and their sums. This will help us quickly determine if we've already computed the sum for a particular pair of indices.

We'll modify the recursive function to check the memoization map before performing any further calculations. If the pair has already been computed, we'll use the stored result instead of recalculating. After calculating the sum of a pair, we'll store the result in the memoization map before returning it. This ensures that future calls with the same pair of indices can be resolved quickly. By using memoization, we aim to reduce the number of redundant calculations, thus improving the efficiency compared to a purely recursive approach.

#### Memoized Recursive Solution in C++20

The only significant modification in Code 5 is the conversion of the recursive function to Dynamic Programming with memoization. Code 6 presents this updated function.

```cpp
// Recursive function with memoization to find a pair of numbers that add up to the target sum
std::optional<std::pair<int, int>> findPairRecursivelyMemo(
    const std::vector<int>& arr,
    int target,
    int start,
    int end,
    std::unordered_map<std::string, std::optional<std::pair<int, int>>>& memo
) {
    // Base case: If start index is greater than or equal to end index, no pair is found
    if (start >= end) {
        return std::nullopt; // Return no value (null optional)
    }

    // Create a unique key for memoization
    std::string key = createKey(start, end);

    // Check if the result is already in the memoization map
    if (memo.find(key) != memo.end()) {
        return memo[key]; // Return the memoized result
    }

    // Base case: If the sum of elements at start and end indices equals the target, pair is found
    if (arr[start] + arr[end] == target) {
        auto result = std::make_optional(std::make_pair(arr[start], arr[end]));
        memo[key] = result; // Store the result in the memoization map
        return result; // Return the pair
    }

    // Recursive call: Move the start index forward to check the next element
    auto result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result) {
        memo[key] = result; // Store the result in the memoization map
        return result; // If a pair is found in the recursive call, return it
    }

    // Recursive call: Move the end index backward to check the previous element
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[key] = result; // Store the result in the memoization map
    return result; // Return the result
}
```

_Code Fragment 13 - Two-sum using a Memoized function_{: class="legend"}

#### Complexity Analysis of the Memoized Solution

In the memoized solution, we store the results of the subproblems in a map to avoid redundant calculations. We can analyze the time complexity step-by-step:

1. **Base Case Check**:

   - If the base case is met (when `start >= end`), the function returns immediately. This takes constant time, $O(1)$.

2. **Memoization Check**:

   - Before performing any calculations, the function checks if the result for the current pair of indices (`start`, `end`) is already stored in the memoization map. Accessing the map has an average time complexity of $O(1)$.

3. **Recursive Calls**:
   - The function makes two recursive calls for each pair of indices: one that increments the `start` index and another that decrements the `end` index.
   - In the worst case, without memoization, this would lead to $2^n$ recursive calls due to the binary nature of the recursive calls.

However, with memoization, each unique pair of indices is computed only once and stored. Given that there are $n(n-1)/2$ unique pairs of indices in an array of size $n$, the memoized function will compute the sum for each pair only once. Thus, the total number of unique computations is limited to the number of pairs, which is $O(n^2)$. Therefore, the time complexity of the memoized solution is **O(n^2)**.

The space complexity of the memoized solution is influenced by two primary factors:

1. **Recursion Stack Depth**: In the most extreme scenario, the recursion could reach a depth of $n$. This means $n$ function calls would be active simultaneously, each taking up space on the call stack. This contributes $O(n)$ to the space complexity.

2. **Memoization Map Size**: This map is used to store the results of computations to avoid redundant calculations. The maximum number of unique entries in this map is determined by the number of distinct pairs of indices we can form from a set of $n$ elements. This number is given by the combination formula n^2, which simplifies to $n(n-1)/2$. As each entry in the map takes up constant space, the overall space used by the memoization map is $O(n^2)$.

Combining these two factors, the overall space complexity of the memoized solution is dominated by the larger of the two, which is $O(n^2)$. This leads us to the following conclusions:

- **Time Complexity**: $O(n^2)$ - The time it takes to process the input grows quadratically with the input size due to the nested loops and memoization overhead.
- **Space Complexity**: $O(n^2)$ - The amount of memory required for storing memoized results also increases quadratically with the input size because the memoization table grows proportionally to n^2.

By storing the results of subproblems, the memoized solution reduces redundant calculations, achieving a time complexity of $O(n^2)$. The memoization map and recursion stack together contribute to a space complexity of $O(n^2)$. Although it has the same time complexity as the Brute-Force solution, memoization significantly improves efficiency by avoiding redundant calculations, making it more practical for larger arrays.

The Brute-Force solution involves nested loops to check all possible pairs, leading to a time complexity of $O(n^2)$. This solution does not use any additional space apart from a few variables, so its space complexity is $O(1)$. While straightforward, the Brute-Force approach is not efficient for large arrays due to its quadratic time complexity.

The naive recursive solution, on the other hand, explores all possible pairs without any optimization, resulting in an exponential time complexity of $O(2^n)$. The recursion stack can grow up to a depth of $n$, leading to a space complexity of $O(n)$. This approach is highly inefficient for large inputs because it redundantly checks the same pairs multiple times.

At this point we can create a summary table.

| Solution Type | Time Complexity | Space Complexity |
| ------------- | --------------- | ---------------- |
| Brute-Force   | $O(n^2)$        | $O(1)$           |
| Recursive     | $O(2^n)$        | $O(n)$           |
| Memoized      | $O(n^2)$        | $O(n^2)$         |

_Tabela 3 - Brute-Force, Recursive and Memoized Solutions Complexity Comparison_{: class="legend"}

The situation may seem grim, with the brute-force approach holding the lead as our best solution so far. But don't lose hope just yet! We have a secret weapon up our sleeves: Dynamic Programming with tabulation.

#### Dynamic Programming: tabulation

Think of it like this: we've been wandering through a maze, trying every path to find the treasure (our solution). The brute-force approach means we're checking every single path, even ones we've already explored. It's exhausting and time-consuming.

But Dynamic Programming with tabulation is like leaving breadcrumbs along the way. As we explore the maze, we mark the paths we've already taken. This way, we avoid wasting time revisiting those paths and focus on new possibilities. It's a smarter way to navigate the maze and find the treasure faster.

In the context of our problem, tabulation means creating a table to store solutions to smaller subproblems. As we solve larger problems, we can refer to this table to avoid redundant calculations. It's a clever way to optimize our solution and potentially find the treasure much faster.

So, even though the brute-force approach may seem like the only option right now, don't give up! Attention! Spoiler Alert! With Dynamic Programming and tabulation, we can explore the maze more efficiently and hopefully find the treasure we've been seeking.

#### C++ code for Two-Sum problem using tabulation

The code is:

```Cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>

// Function to find a pair of numbers that add up to the target sum using tabulation
std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<int, int> table; // Hash table to store elements and their indices

    for (int i = 0; i < sequence.size(); ++i) {
        int complement = targetSum - sequence[i];

        // Check if the complement exists in the hash table
        if (table.find(complement) != table.end()) {
            return std::make_optional(std::make_pair(sequence[i], complement)); // Pair found
        }

        // Store the current element in the hash table
        table[sequence[i]] = i;
    }

    // No pair found
    return std::nullopt;
}

int main() {
    // Example usage
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5}; // Input array
    int targetSum = 11; // Target sum

    // Call the function to find the pair
    auto result = ValuesTabulation(sequence, targetSum);

    // Print the result
    if (result) {
        std::cout << "Pair found: (" << result->first << ", " << result->second << ")\n";
    } else {
        std::cout << "No pair found.\n";
    }

    return 0;
}
```

_Code 9: Full code of a two-sum using a tabulated function_{: class="legend"}

The `std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum)` function uses a hash table (`std::unordered_map`) to store elements of the array and their indices. For each element in the array, it calculates the complement, which is the difference between the target sum and the current element. It then checks if the complement exists in the hash table. If the complement is found, a pair that sums to the target has been identified and the function returns this pair. If the complement does not exist, the function stores the current element and its index in the hash table and proceeds to the next element.

#### Complexity Analysis of the Tabulation Function

The `std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum)` function uses a hash table to efficiently find a pair of numbers that add up to the target sum. The function iterates through each element of the array once, making its time complexity $O(n)$. For each element, it calculates the complement (the difference between the target sum and the current element) and checks if this complement exists in the hash table. _Accessing and inserting elements into the hash table both have an average time complexity of $O(1)$, contributing to the overall linear time complexity of the function_.

The space complexity of the function is also $O(n)$, as it uses a hash table to store the elements of the array and their indices. The size of the hash table grows linearly with the number of elements in the array, which is why the space complexity is linear.

Comparing this with the other solutions, the Brute-Force solution has a time complexity of $O(n^2)$ because it involves nested loops to check all possible pairs, and its space complexity is $O(1)$ since it uses only a few additional variables. The recursive solution without optimization has an exponential time complexity of $O(2^n)$ due to redundant calculations in exploring all pairs, with a space complexity of $O(n)$ due to the recursion stack. The memoized solution improves upon the naive recursion by storing results of subproblems, achieving a time complexity of $O(n^2)$ and a space complexity of $O(n^2)$ due to the memoization map and recursion stack.

_In comparison, the tabulation function is significantly more efficient in terms of both time and space complexity. It leverages the hash table to avoid redundant calculations and provides a linear time solution with linear space usage, making it the most efficient among the four approaches._ Wha we can see in the following table.

| Solution Type | Time Complexity | Space Complexity |
| ------------- | --------------- | ---------------- |
| Brute-Force   | $O(n^2)$        | $O(1)$           |
| Recursive     | $O(2^n)$        | $O(n)$           |
| Memoized      | $O(n^2)$        | $O(n^2)$         |
| Tabulation    | $O(n)$          | $O(n)$           |

_Tabela 4 - Brute-Force, Recursive, Memoized and Tabulated Solutions Complexity Comparison_{: class="legend"}

And so, it seems, we have a champion: Dynamic Programming with tabulation! Anyone armed with this technique has a significant advantage when tackling this problem, especially in job interviews where optimization and clever problem-solving are highly valued.

However, let's be realistic: in the fast-paced world of programming competitive programmings, where every millisecond counts, tabulation might not always be the winner. It can require more memory and setup time compared to other approaches, potentially slowing you down in a race against the clock.

So, while tabulation shines in showcasing your understanding of optimization and problem-solving, it's important to be strategic in a competitive programming setting. Sometimes, a simpler, faster solution might be the key to victory, even if it's less elegant.

The bottom line? Mastering Dynamic Programming and tabulation is a valuable asset, but knowing when and where to use it is the mark of a true programming champion. Now, all that's left is to analyze the execution times.

#### Execution Time Analysis

I started by testing with the same code we used to test the Fibonacci functions. However, in my initial analysis, I noticed some inconsistencies in the execution times. To address this, I refined our measurement methodology by eliminating lambda functions and directly measuring execution time within the main loop. This removed potential overhead introduced by the lambdas, leading to more reliable results. So, I wrote a new, simpler, and more direct code to test the functions:

```Cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <utility>
#include <chrono>

// Function to measure execution time
template <typename Func, typename... Args>
long long measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return duration.count();
}

// Function to calculate average execution time
template <typename Func, typename... Args>
long long average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    for (int i = 0; i < iterations; ++i) {
        total_time += measure_time(func, std::forward<Args>(args)...);
    }
    return total_time / iterations;
}

// Brute-Force Solution
std::pair<int, int> ValuesBruteForce(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sequence[i] + sequence[j] == targetSum) {
                return std::make_pair(sequence[i], sequence[j]);
            }
        }
    }
    return std::make_pair(-1, -1);
}

// Naive Recursive Solution
std::optional<std::pair<int, int>> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    if (start >= end) {
        return std::nullopt;
    }
    if (arr[start] + arr[end] == target) {
        return std::make_optional(std::make_pair(arr[start], arr[end]));
    }
    auto result = findPairRecursively(arr, target, start + 1, end);
    if (result) {
        return result;
    }
    return findPairRecursively(arr, target, start, end - 1);
}

std::optional<std::pair<int, int>> ValuesRecursive(const std::vector<int>& sequence, int targetSum) {
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

// Memoized Recursive Solution
std::string createKey(int start, int end) {
    return std::to_string(start) + "," + std::to_string(end);
}

std::optional<std::pair<int, int>> findPairRecursivelyMemo(
    const std::vector<int>& arr, int target, int start, int end,
    std::unordered_map<std::string, std::optional<std::pair<int, int>>>& memo) {
    if (start >= end) {
        return std::nullopt;
    }
    std::string key = createKey(start, end);
    if (memo.find(key) != memo.end()) {
        return memo[key];
    }
    if (arr[start] + arr[end] == target) {
        auto result = std::make_optional(std::make_pair(arr[start], arr[end]));
        memo[key] = result;
        return result;
    }
    auto result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result) {
        memo[key] = result;
        return result;
    }
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[key] = result;
    return result;
}

std::optional<std::pair<int, int>> ValuesMemoized(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<std::string, std::optional<std::pair<int, int>>> memo;
    return findPairRecursivelyMemo(sequence, targetSum, 0, sequence.size() - 1, memo);
}

// Tabulation Solution
std::optional<std::pair<int, int>> ValuesTabulation(const std::vector<int>& sequence, int targetSum) {
    std::unordered_map<int, int> table;
    for (int i = 0; i < sequence.size(); ++i) {
        int complement = targetSum - sequence[i];
        if (table.find(complement) != table.end()) {
            return std::make_optional(std::make_pair(sequence[i], complement));
        }
        table[sequence[i]] = i;
    }
    return std::nullopt;
}

int main() {
    std::vector<int> sequence = {8, 10, 2, 9, 7, 5};; // 40 numbers
    int targetSum = 11;
    int iterations = 1000;

    std::cout << "-----------------------------------\n";
    std::cout << "Calculating Two-Sum (" << targetSum << ")\n";

    // Measure average execution time for Brute-Force Solution
    auto bruteForceTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesBruteForce(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Brute-Force: " << bruteForceTime << " ns\n";

    // Measure average execution time for Naive Recursive Solution
    auto recursiveTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesRecursive(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Recursive: " << recursiveTime << " ns\n";

    // Measure average execution time for Memoized Recursive Solution
    auto memoizedTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesMemoized(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Memoized: " << memoizedTime << " ns\n";

    // Measure average execution time for Tabulation Solution
    auto tabulationTime = average_time([](const std::vector<int>& seq, int target) {
        ValuesTabulation(seq, target);
        }, iterations, sequence, targetSum);
    std::cout << "Average time for Tabulation: " << tabulationTime << " ns\n";

    std::cout << "-----------------------------------\n";

    return 0;
}
```

_Code 10: Code for execution time test of all functions we create to Two-Sum problem._{: class="legend"}

I simply replicated the functions from the previous code snippets, without any optimization, precisely because our current objective is to solely examine the execution times. Running the new code, we have the following **Output**:

```Shell
-----------------------------------
Calculating Two-Sum (18)
Average time for Brute-Force: 217 ns
Average time for Recursive: 415 ns
Average time for Memoized: 41758 ns
Average time for Tabulation: 15144 ns
-----------------------------------
```

_Output 4: Execution time of Two-Sum solutions._{: class="legend"}

As we've seen, when dealing with a small amount of input data, the brute-force approach surprisingly outshines even more complex algorithms. This might seem counterintuitive, but it's all about the hidden costs of memory management.

When we use sophisticated data structures like `std::string` and `std::unordered_map`, we pay a price in terms of computational overhead. Allocating and deallocating memory on the heap for these structures takes time and resources. This overhead becomes especially noticeable when dealing with small datasets, where the time spent managing memory can easily overshadow the actual computation involved. On the other hand, the brute-force method often relies on simple data types and avoids dynamic memory allocation, resulting in a faster and more efficient solution for smaller inputs.

#### The Dynamic Memory Bottleneck

There are some well-known bottlenecks that can explain why a code with lower complexity runs much slower in a particular environment.

**Hash Table Overhead**: Both memoized and tabulation solutions rely on `std::unordered_map`, which inherently involves dynamic memory allocation. Operations like insertions and lookups, while powerful, come with a cost due to memory management overhead. This is typically slower than accessing elements in a simple array.

**Recursion's Toll**: The naive recursive and memoized solutions utilize deep recursion, leading to a considerable overhead from managing the recursion stack. Each recursive call adds a new frame to the stack, requiring additional memory operations that accumulate over time.

**Memoization's Complexity**: While memoization optimizes by storing intermediate results, it also introduces complexity through the use of `std::unordered_map`. Each new pair calculated requires storage in the hash table, involving dynamic allocations and hash computations, adding to the overall time complexity.

**Cache Friendliness**: Dynamic memory allocations often lead to suboptimal cache utilization. In contrast, the Brute-Force and tabulation solutions likely benefit from better cache locality due to their predominant use of array accesses. Accessing contiguous memory locations (as in arrays) generally results in faster execution due to improved cache hit rates.

**Function Call Overhead**: The overhead from frequent function calls, including those made through lambda functions, can accumulate, particularly in performance-critical code.

By understanding and mitigating these bottlenecks, we can better optimize our code and achieve the expected performance improvements.

_In essence, the dynamic nature of memory operations associated with hash tables and recursion significantly affects execution times_. These operations are generally slower than accessing static memory structures like arrays. The deep recursion in the memoized and naive recursive approaches exacerbates this issue, as the growing recursion stack necessitates increased memory management.

The memoized solution, while clever, bears the brunt of both issues – extensive recursion and frequent hash table operations. This combination leads to higher execution times compared to the Brute-Force and tabulation approaches, which primarily rely on array accesses and enjoy the benefits of better cache performance and reduced memory management overhead.

In conclusion, the observed differences in execution times can be attributed to the distinct memory access patterns and associated overheads inherent in each approach. Understanding these nuances is crucial for making informed decisions when optimizing code for performance.

### We will always have C

As we delve into Dynamic Programming with C++, our focus is on techniques that shine in interviews and coding competitive programmings. Since competitive coding often favors slick C-style code, we'll zero in on a tabulation solution for this problem. Tabulation, as we know, is usually the most efficient approach. To show you what I mean, check out the `int* ValuesTabulationCStyle(const int* sequence, int length, int targetSum)` function in Code Fragment 12.

```Cpp
int* ValuesTabulationCStyle(const int* sequence, int length, int targetSum) {
    const int MAX_VAL = 1000; // Assuming the values in the sequence are less than 1000
    static int result[2] = { -1, -1 }; // Static array to return the result
    int table[MAX_VAL];
    memset(table, -1, sizeof(table));

    for (int i = 0; i < length; ++i) {
        int complement = targetSum - sequence[i];
        if (complement >= 0 && table[complement] != -1) {
            result[0] = sequence[i];
            result[1] = complement;
            return result;
        }
        table[sequence[i]] = i;
    }
    return result;
}
```

_Code Fragment 14 - Two-sum with C-Style code, using a Memoized Function_{: class="legend"}

The C-style function is as straightforward as it gets, and as far as I can tell, it's equivalent to the C++ tabulation function. Perhaps the only thing worth noting is the use of the memset function.

> The `memset` function in C and C++ is your go-to tool for filling a block of memory with a specific value. You'll usually find it defined in the `<cstring>` header. It takes a pointer to the memory block you want to fill (`ptr`), the value you want to set (converted to an `unsigned char`), and the number of bytes you want to set.
> In our code, `memset(table, -1, sizeof(table))`; does exactly that—it fills the entire table array with the value $-1$, which is a handy way to mark those spots as empty or unused.
> Why use `memset` instead of `malloc` or `alloc`? Well, `memset` is all about initializing memory that's already been allocated. `malloc` and `alloc` are for allocating new memory. If you need to do both, C++ has you covered with the `new` operator, which can allocate and initialize in one step.
> So, `memset` is the memory magician for resetting or initializing arrays and structures, while `malloc`, `alloc`, and `calloc` handle the memory allocation part of the job.

The use of `menset` bring us to analise the function complexity.

#### Two-Sum C-Style Tabulation Function Complexity

The function `ValuesTabulationCStyle` uses `memset` to initialize the `table` array. The complexity of the function can be divided into two parts:

1. **Initialization with `memset`:**

   ```cpp
   memset(table, -1, sizeof(table));
   ```

   The memset function initializes each byte of the array. Since the size of the array is constant (`MAX_VAL`), the complexity of this operation is $O(1)$ in terms of asymptotic complexity, although it has a constant cost depending on the array size.

   ```cpp
   for (int i = 0; i < length; ++i) {
       int complement = targetSum - sequence[i];
       if (complement >= 0 && table[complement] != -1) {
           result[0] = sequence[i];
           result[1] = complement;
           return result;
       }
       table[sequence[i]] = i;
   }
   ```

2. The Main Loop

   The main loop iterates over each element of the input array sequence. Therefore, the complexity of this part is $O(n)$, where n is the length of the sequence array.

Combining all the pieces, the function's overall complexity is still $O(n)$. Even though initializing the array with `memset` takes $O(1)$ time, it doesn't change the fact that we have to loop through each element in the input array, which takes $O(n)$ time.

Looking back at our original C++ function using `std::unordered_map`, we also had `O(1)` complexity for the map initialization and `O(n)` for the main loop. So, while using `memset` for our C-style array feels different, it doesn't change the big picture – both approaches end up with linear complexity.

The key takeaway here is that while using `memset` might feel like a win for initialization time, it doesn't change the overall complexity when you consider the whole function. This leaves us with the task of running the code and analyzing the execution times, as shown in Output 5.

```Shell
-----------------------------------
Calculating Two-Sum (11)
Average time for Brute-Force: 318 ns
Average time for Recursive: 626 ns
Average time for Memoized: 39078 ns
Average time for Tabulation: 5882 ns
Average time for Tabulation C-Style: 189 ns
-----------------------------------
```

_Output 5: Execution time of Two-Sum solutions, including C-Style Arrays._{: class="legend"}

Analyzing Output 5, it's clear that the C-style solution is, for all intents and purposes, twice as fast as the C++ tabulation solution. However, there are a few caveats: the C++ code was written to showcase the language's flexibility, not to optimize for performance. On the other hand, the C-style function was designed for simplicity. Often, simplicity equates to speed, and this is something to maximize when creating a function with linear complexity. _Now, we need to compare the C solution with C++ code that prioritizes high performance over flexibility in writing_.

### High Performance C++

Code Fragment 15 was rewritten by stripping away all the complex data structures we were previously using. The `main()` function remains largely unchanged, so it's been omitted here. I've also removed the functions used for measuring and printing execution times.

```Cpp
// Brute-Force Solution
std::array<int, 2> ValuesBruteForce(const std::vector<int>& sequence, int targetSum) {
    int n = sequence.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sequence[i] + sequence[j] == targetSum) {
                return { sequence[i], sequence[j] };
            }
        }
    }
    return { -1, -1 };
}

// Naive Recursive Solution
std::array<int, 2> findPairRecursively(const std::vector<int>& arr, int target, int start, int end) {
    if (start >= end) {
        return { -1, -1 };
    }
    if (arr[start] + arr[end] == target) {
        return { arr[start], arr[end] };
    }
    std::array<int, 2> result = findPairRecursively(arr, target, start + 1, end);
    if (result[0] != -1) {
        return result;
    }
    return findPairRecursively(arr, target, start, end - 1);
}

std::array<int, 2> ValuesRecursive(const std::vector<int>& sequence, int targetSum) {
    return findPairRecursively(sequence, targetSum, 0, sequence.size() - 1);
}

// Memoized Recursive Solution
std::array<int, 2> findPairRecursivelyMemo(
    const std::vector<int>& arr, int target, int start, int end,
    std::array<std::array<int, 2>, 1000>& memo) {
    if (start >= end) {
        return { -1, -1 };
    }
    if (memo[start][end][0] != -1) {
        return memo[start][end];
    }
    if (arr[start] + arr[end] == target) {
        memo[start][end] = { arr[start], arr[end] };
        return { arr[start], arr[end] };
    }
    std::array<int, 2> result = findPairRecursivelyMemo(arr, target, start + 1, end, memo);
    if (result[0] != -1) {
        memo[start][end] = result;
        return result;
    }
    result = findPairRecursivelyMemo(arr, target, start, end - 1, memo);
    memo[start][end] = result;
    return result;
}

std::array<int, 2> ValuesMemoized(const std::vector<int>& sequence, int targetSum) {
    std::array<std::array<int, 2>, 1000> memo;
    for (auto& row : memo) {
        row = { -1, -1 };
    }
    return findPairRecursivelyMemo(sequence, targetSum, 0, sequence.size() - 1, memo);
}

// Tabulation Solution using C-style arrays
std::array<int, 2> ValuesTabulationCStyle(const int* sequence, int length, int targetSum) {
    const int MAX_VAL = 1000; // Assuming the values in the sequence are less than 1000
    std::array<int, 2> result = { -1, -1 }; // Static array to return the result
    int table[MAX_VAL];
    memset(table, -1, sizeof(table));

    for (int i = 0; i < length; ++i) {
        int complement = targetSum - sequence[i];
        if (complement >= 0 && table[complement] != -1) {
            result[0] = sequence[i];
            result[1] = complement;
            return result;
        }
        table[sequence[i]] = i;
    }
    return result;
}
```

_Code Fragment 15 - All Two-sum functions including a pure `std::array` tabulated function_{: class="legend"}

Running this modified code, we get the following **Output**:

```Shell
-----------------------------------
Calculating Two-Sum (11)
Average time for Brute-Force: 157 ns
Average time for Recursive: 652 ns
Average time for Memoized: 39514 ns
Average time for Tabulation: 5884 ns
Average time for Tabulation C-Style: 149 ns
-----------------------------------
```

_Output 6: Execution time of All Two-Sum solutions including C-style Tabulation._{: class="legend"}

Let's break down the results for calculating the Two-Sum problem, keeping in mind that the fastest solutions have a linear time complexity, O(n):

- **Brute-Force**: Blazing fast at 157 ns on average. This is our baseline, but remember, Brute-Force doesn't always scale well for larger problems.
- **Recursive**: A bit slower at 652 ns. Recursion can be elegant, but it can also lead to overhead.
- **Memoized**: This one's the outlier at 39514 ns. Memoization can be a powerful optimization, but it looks like the overhead is outweighing the benefits in this case.
- **Tabulation**: A respectable 5884 ns. Tabulation is a solid Dynamic Programming technique, and it shows here.
- **Tabulation C-Style**: A close winner at 149 ns! This stripped-down, C-style implementation of tabulation is just a hair behind Brute-Force in terms of speed.

The C++ and C versions of our tabulation function are practically neck and neck in terms of speed for a few key reasons:

1. **Shared Simplicity**: Both versions use plain old arrays (`std::array` in C++, C-style arrays in C) to store data. This means memory access is super efficient in both cases.

2. **Speedy Setup**: We use `memset` to initialize the array in both versions. This is a highly optimized operation, so the initialization step is lightning fast in both languages.

3. **Compiler Magic**: Modern C++ compilers are incredibly smart. They can often optimize the code using `std::array` to be just as fast, or even faster, than hand-written C code.

4. **No Frills**: Both functions have a straightforward design without any fancy branching or extra layers. The operations within the loops are simple and take the same amount of time every time, minimizing any overhead.

5. **Compiler Boost**: _Compilers like GCC and Clang have a whole bag of tricks to make code run faster, like loop unrolling and smart prediction_. These tricks work equally well for both C and C++ code, especially when we're using basic data structures and simple algorithms.

Thanks to all this, the C++ version of our function, using `std::array`, runs just as fast as its C counterpart.

And this is how C++ code should be for competitive programmings. However, not for interviews. In interviews, unless high performance is specifically requested, what they're looking for is your mastery of the language and the most efficient algorithms. So, an O(n) solution using the appropriate data structures will give you a better chance of success.

### Exercises: Variations of the Two Sum

There are few interesting variations of Two-Sum problem:

1. The array can contain both positive and negative integers.
2. Each input would have exactly one solution, and you may not use the same element twice.
3. Each input can have multiple solutions, and the same element cannot be used twice in a pair.
4. The function should return all pairs that sum to the target value.

Try to solve these variations. Take as much time as you need; I will wait.

# The Dynamic Programming Classic Problems

From now on, we will explore 10 classic Dynamic Programming problems. For each one, we will delve into Brute-Force techniques, recursion, memoization, tabulation, and finally the most popular solution for each, even if it is not among the techniques we have chosen. The problems we will address are listed in the table below[^1].

| Name                                           | Description/Example                                                                                                                                                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Counting all possible paths in a matrix        | Given $N$ and $M$, count all possible distinct paths from $(1,1)$ to $(N, M)$, where each step is either from $(i,j)$ to $(i+1,j)$ or $(i,j+1)$.                                                                               |
| Subset Sum                                     | Given $N$ integers and $T$, determine whether there exists a subset of the given set whose elements sum up to $T$.                                                                                                             |
| Longest Increasing Subsequence                 | You are given an array containing $N$ integers. Your task is to determine the LCS in the array, i.e., LCS where every element is larger than the previous one.                                                                 |
| Rod Cutting                                    | Given a rod of length $n$ units, Given an integer array cuts where cuts[i] denotes a position you should perform a cut at. The cost of one cut is the length of the rod to be cut. What is the minimum total cost of the cuts. |
| Longest Common Subsequence                     | You are given strings $s$ and $t$. Find the length of the longest string that is a subsequence of both $s$ and $t$.                                                                                                            |
| Longest Palindromic Subsequence                | Finding the Longest Palindromic Subsequence (LPS) of a given string.                                                                                                                                                           |
| Edit Distance                                  | The edit distance between two strings is the minimum number of operations required to transform one string into the other. Operations are ["Add", "Remove", "Replace"].                                                        |
| Coin Change Problem                            | Given an array of coin denominations and a target amount, find the minimum number of coins needed to make up that amount.                                                                                                      |
| 0-1 knapsack                                   | Given $W$, $N$, and $N$ items with weights $w_i$ and values $v_i$, what is the maximum $\sum_{i=1}^{k} v_i$ for each subset of items of size $k$ ($1 \le k \le N$) while ensuring $\sum_{i=1}^{k} w_i \le W$?                  |
| Longest Path in a Directed Acyclic Graph (DAG) | Finding the longest path in Directed Acyclic Graph (DAG).                                                                                                                                                                      |
| Traveling Salesman Problem (TSP)               | Given a list of cities and the distances between each pair of cities, find the shortest possible route that visits each city exactly once and returns to the origin city.                                                      |
| Matrix Chain Multiplication                    | Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications. |

_Tabela 5 - The Dynamic Programming we'll study and solve._{: class="legend"}

Stop for a moment, perhaps have a soda or a good wine. Rest a bit, then gather your courage, arm yourself with patience, and continue. Practice makes perfect.

This will continue!!!

## Problem 1 Statement: Counting All Possible Paths in a Matrix

Given two integers $m$ and $n$, representing the dimensions of a matrix, count all possible distinct paths from the top-left corner $(0,0)$ to the bottom-right corner $(m-1,n-1)$. Each step can either be to the right or down.

**Input**:

- Two integers $m$ and $n$ where $1 \leq m, n \leq 100$.

#\*_Output_:

- An integer representing the number of distinct paths from $(0,0)$ to $(m-1,n-1)$.

**Example**:

**Input**:
3 3

**Output**:
6

**Constraints**:

- You can only move to the right or down in each step.

**Analysis**:

Let's delve deeper into the "unique paths" problem. Picture a matrix as a grid, where you start at the top-left corner $(0, 0)$ and your goal is to reach the bottom-right corner $(m-1, n-1)$. The twist? You're only allowed to move down or right. The challenge is to figure out how many distinct paths you can take to get from start to finish.

This problem so intriguing because it blends the elegance of combinatorics (the study of counting) with the power of Dynamic Programming (a clever problem-solving technique). The solution involves combinations, a fundamental concept in combinatorics. Moreover, it's a classic example of how Dynamic Programming can streamline a seemingly complex problem.

The applications of this type of problem extend beyond theoretical interest. They can be found in practical scenarios like robot navigation in a grid environment, calculating probabilities in games with grid-like movements, and analyzing maze-like structures. Understanding this problem provides valuable insights into a range of fields, from mathematics to robotics and beyond.

Now, let's bring in some combinatorics! To journey from the starting point $(0, 0)$ to the destination $(m-1, n-1)$, you'll need a total of $(m - 1) + (n - 1)$ moves. That's a mix of downward and rightward steps.

The exciting part is figuring out how many ways we can arrange those moves. Imagine choosing $(m - 1)$ moves to go down and $(n - 1)$ moves to go right, out of a total of $(m + n - 2)$ moves. This can be calculated using the following formula:

$$C(m + n - 2, m - 1) = \frac{(m + n - 2)!}{(m - 1)! * (n - 1)!}$$

For our 3x3 matrix example $(m = 3, n = 3)$, the calculation is:

$$C(3 + 3 - 2, 3 - 1) = \frac{4!}{(2! * 2!)} = 6$$

This tells us there are $6$ distinct paths to reach the bottom-right corner. Let's also visualize this using Dynamic Programming:

**Filling the Matrix with Dynamic Programming:**

1. **Initialize:** Start with a $dp$ matrix where $dp[0][0] = 1$ (one way to be at the start).

   dp = \begin{bmatrix}
   1 & 0 & 0 \
   0 & 0 & 0 \
   0 & 0 & 0
   \end{bmatrix}

2. **Fill First Row and Column:** There's only one way to reach each cell in the first row and column (either from the left or above).

   dp = \begin{bmatrix}
   1 & 1 & 1 \
   1 & 0 & 0 \
   1 & 0 & 0
   \end{bmatrix}

3. **Fill Remaining Cells:** For the rest, the number of paths to a cell is the sum of paths to the cell above and the cell to the left: $dp[i][j] = dp[i-1][j] + dp[i][j-1]$

   dp = \begin{bmatrix}
   1 & 1 & 1 \
   1 & 2 & 3 \
   1 & 3 & 6
   \end{bmatrix}

The bottom-right corner, $dp[2][2]$, holds our answer: $6$ unique paths.

Bear with me, dear reader, as I temporarily diverge from our exploration of Dynamic Programming. Before delving deeper, it's essential to examine how we might solve this problem using a Brute-Force approach.

### Using Brute-Force

To tackle the unique paths problem with a Brute-Force approach, we can use an iterative solution and a stack in C++20. The stack will keep track of our current position in the matrix and the number of paths that led us there. Here's a breakdown of how it works:

1. We'll define a structure called `Position` to store the current coordinates $(i, j)$ and the count of paths leading to that position.

2. We'll start by pushing the initial position $(0, 0)$ onto the stack, along with a path count of $1$ (since there's one way to reach the starting point).

3. While the stack isn't empty:

   - Pop the top position from the stack.
   - If we've reached the bottom-right corner $(m-1, n-1)$, increment the total path count.
   - If moving right is possible (within the matrix bounds), push the new position (with the same path count) onto the stack.
   - If moving down is possible, do the same.

4. When the stack is empty, the total path count will be the answer we seek – the total number of unique paths from $(0, 0)$ to $(m-1, n-1)$.

Code Fragment 16 demonstrates how to implement this algorithm in C++.

```cpp
#include <iostream>
#include <stack>
#include <chrono>

//.....
// Structure to represent a position in the matrix
struct Position {
    int i, j;
    int pathCount;
};

// Function to count paths using Brute-Force
int countPaths(int m, int n) {
    std::stack<Position> stk;
    stk.push({ 0, 0, 1 });
    int totalPaths = 0;

    while (!stk.empty()) {
        Position pos = stk.top();
        stk.pop();

        int i = pos.i, j = pos.j, pathCount = pos.pathCount;

        // If we reach the bottom-right corner, add to total paths
        if (i == m - 1 && j == n - 1) {
            totalPaths += pathCount;
            continue;
        }

        // Move right if within bounds
        if (j + 1 < n) {
            stk.push({ i, j + 1, pathCount });
        }

        // Move down if within bounds
        if (i + 1 < m) {
            stk.push({ i + 1, j, pathCount });
        }
    }

    return totalPaths;
}
```

_Code Fragment 16 - Count all paths function using Brute-Force._{: class="legend"}

Let's start looking at `std::stack` data structure:

> In C++20, the `std::stack` is a part of the Standard Template Library (STL) and is used to implement a stack data structure, which follows the Last-In-First-Out (LIFO) principle. A `std::stack` is a container adapter that provides a stack interface, designed to operate in a LIFO context. Elements are added to and removed from the top of the stack. The syntax for creating a stack is `std::stack<T> stack_name;` where `T` is the type of elements contained in the stack.
> To create a stack, you can use constructors such as `std::stack<int> s1;` for the default constructor or `std::stack<int> s2(some_container);` to construct with a container. To add an element to the top of the stack, use the `push` function: `stack_name.push(value);`. To remove the element at the top of the stack, use the `pop` function: `stack_name.pop();`. To return a reference to the top element of the stack, use `T& top_element = stack_name.top();`. To check whether the stack is empty, use `bool is_empty = stack_name.empty();`. Finally, to return the number of elements in the stack, use `std::size_t stack_size = stack_name.size();`.
> The `std::stack` class is a simple and effective way to manage a stack of elements, ensuring efficient access and modification of the last inserted element.

In the context of our code for counting paths in a matrix using a stack, the line `stk.push({ 0, 0, 1 });` is used to initialize the stack with the starting position of the matrix traversal. The `stk` is a `std::stack` of `Position` structures. The `Position` structure is defined as follows:

```cpp
struct Position {
    int i, j;
    int pathCount;
};
```

The `Position` structure has three members: `i` and `j`, which represent the current coordinates in the matrix, and `pathCount`, which represents the number of paths that lead to this position.

The line `stk.push({ 0, 0, 1 });` serves a specific purpose in our algorithm:

1. It creates a `Position` object with the initializer list `{ 0, 0, 1 }`, setting `i = 0`, `j = 0`, and `pathCount = 1`.
2. It then pushes this `Position` object onto the stack using the `push` function of `std::stack`.

In simpler terms, this line of code initializes the stack with the starting position of the matrix traversal, which is the top-left corner of the matrix at coordinates $(0, 0)$. It also sets the initial path count to $1$, indicating that there is one path starting from this position. From here, the algorithm will explore all possible paths from the top-left corner.

_The Brute-Force solution for counting paths in a matrix involves exploring all possible paths from the top-left corner to the bottom-right corner. Each path consists of a series of moves either to the right or down. The algorithm uses a stack to simulate the recursive exploration of these paths_.

To analyze the time complexity, consider that the function must explore each possible combination of moves in the matrix. For a matrix of size $m \times n$, there are a total of $m + n - 2$ moves required to reach the bottom-right corner from the top-left corner. Out of these moves, $m - 1$ must be down moves and $n - 1$ must be right moves. The total number of distinct paths is given by the binomial coefficient $C(m+n-2, m-1)$, which represents the number of ways to choose $m-1$ moves out of $m+n-2$.

The time complexity of the Brute-Force approach is exponential in nature because it explores every possible path. Specifically, the time complexity is $O(2^{m+n})$ since each step involves a choice between moving right or moving down, leading to $2^{m+n}$ possible combinations of moves in the worst case. This exponential growth makes the Brute-Force approach infeasible for large matrices, as the number of paths grows very quickly with increasing $m$ and $n$.

The space complexity is also significant because the algorithm uses a stack to store the state of each position it explores. In the worst case, the depth of the stack can be as large as the total number of moves, which is $m + n - 2$. Thus, the space complexity is $O(m+n)$, primarily due to the stack's storage requirements.

In summary, the Brute-Force solution has an exponential time complexity $O(2^{m+n})$ and a linear space complexity $O(m+n)$, making it suitable only for small matrices where the number of possible paths is manageable.

My gut feeling tells me this complexity is very, very bad. We'll definitely find better complexities as we explore Dynamic Programming solutions. Either way, we need to measure the runtime. Running the function `int countPaths(int m, int n)` within the same structure we created earlier to measure execution time, we will have:

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 10865 ns
-----------------------------------
```

_Output 7: Execution time of Counting all paths problem using Brute-Force._{: class="legend"}

Finally, I won't be presenting the code done with pure recursion. As we've seen, recursion is very elegant and can score points in interviews. However, the memoization solution will include recursion, so if you use memoization and recursion in the same solution, you'll ace the interview.

### Using Memoization

Code Fragment 17 shows the functions I created to apply memoization. There are two functions: the `int countPathsMemoizationWrapper(int m, int n)` function used to initialize the dp data structure and call the recursive function `int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp)`. I used `std::vector` already anticipating that we won't know the size of the matrix beforehand.

```Cpp
// Function to count paths using Dynamic Programming with memoization
int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp) {
    if (m == 1 || n == 1) return 1;  // Base case
    if (dp[m - 1][n - 1] != -1) return dp[m - 1][n - 1];  // Return memoized result
    dp[m - 1][n - 1] = countPathsMemoization(m - 1, n, dp) + countPathsMemoization(m, n - 1, dp);  // Memoize result
    return dp[m - 1][n - 1];
}

int countPathsMemoizationWrapper(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, -1));
    return countPathsMemoization(m, n, dp);
}

```

_Code Fragment 17 - Count all paths function using Memoization._{: class="legend"}

The function `int countPathsMemoization(int m, int n, std::vector<std::vector<int>>& dp)` counts all possible paths in an $m \times n$ matrix using Dynamic Programming with memoization. The `dp` matrix serves as a cache, storing and reusing intermediate results to avoid redundant calculations.

Upon invocation, the function first checks if the current position is in the first row (`m == 1`) or first column (`n == 1`). If so, it returns $1$, as there is only one way to reach such cells. Next, it checks the `dp[m-1][n-1]` cell. If the value is not $-1$ (the initial value indicating "not yet calculated"), it signifies that the result for this position has already been memoized and is immediately returned.

Otherwise, the function recursively calculates the number of paths from the cell above (`m-1`, `n`) and the cell to the left (`m`, `n-1`). The sum of these values is then stored in `dp[m-1][n-1]` before being returned.

Utilizing `std::vector<std::vector<int>>` for `dp` ensures efficient storage of intermediate results, significantly reducing the number of recursive calls required compared to a purely recursive approach. This memoized version substantially improves the function's performance, especially for larger matrices.

The `countPathsMemoization` function exemplifies memoization rather than tabulation due to its top-down recursive approach, where results are stored on-the-fly. Memoization involves recursive function calls that cache subproblem results upon their first encounter, ensuring each subproblem is solved only once and reused when needed.

Conversely, tabulation employs a bottom-up approach, utilizing an iterative method to systematically fill a table (or array) with subproblem solutions. This typically begins with the smallest subproblems, iteratively combining results to solve larger ones until the final solution is obtained.

In `countPathsMemoization`, the function checks if the result for a specific cell is already computed and stored in the `dp` matrix. If not, it recursively computes the result by combining the results of smaller subproblems and then stores it in the `dp` matrix. This process continues until all necessary subproblems are solved, a characteristic of memoization.

The `dp` matrix is utilized to store intermediate results, preventing redundant calculations. Each cell within `dp` corresponds to a subproblem, representing the number of paths to that cell from the origin. Recursive calls compute the number of paths to a given cell by summing the paths from the cell above and the cell to the left.

_The function boasts a time complexity of $O(m \times n)$_. This is because each cell in the `dp` matrix is computed only once, with each computation requiring constant time. Thus, the total operations are proportional to the number of cells in the matrix, namely $m \times n$.

Similarly, _the space complexity is $O(m \times n)$ due to the `dp` matrix_, which stores the number of paths for each cell. The matrix necessitates $m \times n$ space to accommodate these intermediate results.

In essence, the Dynamic Programming approach with memoization transforms the exponential time complexity of the naive Brute-Force solution into a linear complexity with respect to the matrix size. Consequently, this solution proves far more efficient and viable for larger matrices.

Finally, we need to run this code and compare it to the Brute-Force version.

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 12494 ns
Average time for Memoization: 4685 ns
-----------------------------------
```

_Output 9: Comparison between Brute-Force, Memoization and Tabulation functions._{: class="legend"}

I ran it dozens of times and, most of the time, the memoized function was twice as fast as the Brute-Force function, and sometimes it was three times faster. Now we need to look at the Dynamic Programming solution using tabulation.

### Using Tabulation

Like I did before, the Code Fragment 18 shows the function I created to apply tabulation. The function `int countPathsTabulation(int m, int n)` uses Dynamic Programming with tabulation to count all possible paths in an $m \times n$ matrix.

```Cpp
// Function to count paths using Dynamic Programming with tabulation
int countPathsTabulation(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0 || j == 0) {
                dp[i][j] = 1;
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    return dp[m - 1][n - 1];
}
```

_Code Fragment 18 - Count all paths function using Tabulation._{: class="legend"}

The function `int countPathsTabulation(int m, int n)` counts all possible paths in an $m \times n$ matrix using Dynamic Programming with tabulation. The `dp` matrix is used to store the number of paths to each cell in a bottom-up manner, ensuring that each subproblem is solved iteratively. Each subproblem represents the number of distinct paths to a cell $(i, j)$ from the top-left corner $(0, 0)$.

The function initializes a `dp` matrix of size $m \times n$ with all values set to $0$. It then iterates over each cell in the matrix. If a cell is in the first row or first column, it sets the value to $1$ because there is only one way to reach these cells (either all the way to the right or all the way down). For other cells, it calculates the number of paths by summing the values from the cell above (`dp[i - 1][j]`) and the cell to the left (`dp[i][j - 1]`). This sum is then stored in `dp[i][j]`.

Using `std::array<std::array<int, MAX_SIZE>, MAX_SIZE>` for the `dp` matrix ensures efficient storage and retrieval of intermediate results. This tabulated approach systematically fills the `dp` matrix from the smallest subproblems up to the final solution, avoiding the overhead of recursive function calls and providing a clear and straightforward iterative solution.

_The time complexity of this function is $O(m \times n)$_. Each cell in the `dp` matrix is computed once, and each computation takes constant time, making the total number of operations proportional to the number of cells in the matrix. Similarly, _the space complexity is $O(m \times n)$_ due to the `dp` matrix, which requires $m \times n$ space to store the number of paths for each cell.

This Dynamic Programming approach with tabulation significantly improves performance compared to the naive Brute-Force solution. By transforming the problem into a series of iterative steps, it provides an efficient and scalable solution for counting paths in larger matrices.

Here's something interesting, dear reader. While using `std::vector` initially yielded similar execution times to the memoization function, switching to `std::array` resulted in a dramatic improvement. Not only did it significantly reduce memory usage, but it also made the function up to 4 times faster!

After running additional tests, the tabulation function averaged $3$ times faster execution than the original. However, there's a caveat: using `std::array` necessitates knowing the input size beforehand, which might not always be practical in real-world scenarios. Finally we can take a look in this function execution time:

```Shell
-----------------------------------
Calculating Paths in a 3x3 matrix
Average time for Brute-Force: 10453 ns
Average time for Memoization: 5348 ns
Average time for Tabulation: 1838 ns
-----------------------------------
```

The key takeaway is this: both memoization and tabulation solutions share the same time complexity. Therefore, in an interview setting, the choice between them boils down to personal preference. But if performance is paramount, tabulation (especially with `std::array` if the input size is known) is the way to go. Of course, now it's up to the diligent reader to test all the functions we've developed to solve problem "Counting All Possible Paths in a Matrix" with `std::array`. Performance has its quirks, and since there are many factors outside the programmer's control involved in execution, we always need to test in the an environment just like the production environment.

## Problem 2 Statement: Subset Sum

Given $N$ integers and $T$, determine whether there exists a subset of the given set whose elements sum up to $T$.

**Input**:

- An integer $N$ representing the number of integers.
- An integer $T$ representing the target sum.
- A list of $N$ integers.

**Output**:

- A boolean value indicating whether such a subset exists.

**Example**:

**Input**:

```txt
5 10
2 3 7 8 10
```

**Output**:

```txt
true
```

**Constraints**:

- $1 \leq N \leq 100$
- $1 \leq T \leq 1000$
- Each integer in the list is positive and does not exceed $100$.

**Analysis**:

The "Subset Sum" problem has already been tackled in the chapter: "Your First Dynamic Programming Problem." Therefore, our diligent reader should review the conditions presented here and see if the solution we presented for the "Two-Sum" problem applies in this case. If not, it'll be up to the reader to adapt the previous code accordingly. I'll kindly wait before we go on.

## Problem 3 Statement: Longest Increasing Subsequence

You are given an array containing $N$ integers. Your task is to determine the Longest Increasing Subsequence (LIS) in the array, where every element is larger than the previous one.

**Input**:

- An integer $N$ representing the number of integers.
- A list of $N$ integers.

**Output**:

- An integer representing the length of the Longest Increasing Subsequence.

**Example**:

**Input**:

```txt
6
5 2 8 6 3 6 9 7
```

**Output**:

```txt
4
```

**Constraints**:

- $1 \leq N \leq 1000$
- Each integer in the list can be positive or negative.

**Analysis**:

The "Longest Increasing Subsequence" (LIS) problem is a classic problem in Dynamic Programming, often appearing in interviews and programming competitive programmings. The goal is to find the length of the longest subsequence in a given array such that all elements of the subsequence are in strictly increasing order. _There are three main approaches to solving this problem: Brute-Force, memoization, and tabulation. Coincidentally, these are the three solutions we are studying_. So, let's go.

### Brute-Force

In the Brute-Force approach, we systematically generate all possible subsequences of the array and examine each one to determine if it's strictly increasing. The length of the longest increasing subsequence is then tracked and ultimately returned. Here's the algorithm for solving the LIS problem using Brute-Force:

1. Iteratively generate all possible subsequences of the array. (We'll reserve recursion for the memoization approach.)
2. For each subsequence, verify if it is strictly increasing.
3. Keep track of the maximum length encountered among the increasing subsequences.

Code Fragment 19 presents the function I developed using a brute-force approach.

```Cpp
// Iterative Brute-Force LIS function
int longestIncreasingSubsequenceBruteForce(const std::vector<int>& arr) {
    int n = arr.size();
    int maxLen = 0;

    // Generate all possible subsequences using bitmasking
    for (int mask = 1; mask < (1 << n); ++mask) {
        std::vector<int> subsequence;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                subsequence.push_back(arr[i]);
            }
        }

        // Check if the subsequence is strictly increasing
        bool isIncreasing = true;
        for (int i = 1; i < subsequence.size(); ++i) {
            if (subsequence[i] <= subsequence[i - 1]) {
                isIncreasing = false;
                break;
            }
        }

        if (isIncreasing) {
            maxLen = std::max(maxLen, (int)subsequence.size());
        }
    }

    return maxLen;
}
```

_Code Fragment 19 - Interactive function to solve the LIS problem._{: class="legend"}

In the function `int longestIncreasingSubsequenceBruteForce(const std::vector<int>& arr)`, bitmasking is used to generate all possible subsequences of the array. Bitmasking involves using a binary number, where each bit represents whether a particular element in the array is included in the subsequence. For an array of size $n$, there are $2^n$ possible subsequences, corresponding to all binary numbers from $1$ to $(1 << n) - 1$. In each iteration, the bitmask is checked, and if the i-th bit is set (i.e., `mask & (1 << i)` is true), the i-th element of the array is included in the current subsequence. This process ensures that every possible combination of elements is considered, allowing the function to generate all potential subsequences for further evaluation.

For every generated subsequence, the function meticulously examines its elements to ensure they are in strictly increasing order. This involves comparing each element with its predecessor, discarding any subsequence where an element is not greater than the one before it.

Throughout the process, the function keeps track of the maximum length among the valid increasing subsequences encountered. If a subsequence's length surpasses the current maximum, the function updates this value accordingly.

While ingenious, this brute-force method has a notable drawback: _an exponential time complexity of $O(2^n _ n)$*. This arises from the $2^n$ possible subsequences and the $n$ operations needed to verify the increasing order of each. Consequently, this approach becomes impractical for large arrays due to its high computational cost.

Notice that, once again, I started by using `std::vector` since we usually don't know the size of the input dataset beforehand. Now, all that remains is to run this code and observe its execution time. Of course, my astute reader should remember that using `std::array` would likely require knowing the maximum input size in advance, but it would likely yield a faster runtime.

```Cpp
-----------------------------------
Calculating LIS in the array
Average time for LIS: 683023 ns
-----------------------------------
```

_Output 10: Execution time for LIS solution using Brute-Force._{: class="legend"}

At this point, our dear reader should have a mantra in mind: 'Don't use Brute-Force... Don't use Brute-Force.' With that said, let's delve into Dynamic Programming algorithms, keeping Brute-Force as a reference point for comparison.

### Memoization

Memoization is a handy optimization technique that remembers the results of expensive function calls. If the same inputs pop up again, we can simply reuse the stored results, saving precious time. Let's see how this applies to our LIS problem.

We can use memoization to avoid redundant calculations by storing the length of the LIS ending at each index. Here's the game plan:

1. Define a recursive function `LIS(int i, const std::vector<int>& arr, std::vector<int>& dp)` that returns the length of the LIS ending at index `i`.
2. Create a memoization array called `dp`, where `dp[i]` will store the length of the LIS ending at index `i`.
3. For each element in the array, compute `LIS(i)` by checking all previous elements `j` where `arr[j]` is less than `arr[i]`. Update `dp[i]` accordingly.
4. Finally, the maximum value in the `dp` array will be the length of the longest increasing subsequence.

Here's the implementation of the function: (Code Snippet 21)

```Cpp
// Recursive function to find the length of LIS ending at index i with memoization
int LIS(int i, const std::vector<int>& arr, std::vector<int>& dp) {
    if (dp[i] != -1) return dp[i];

    int maxLength = 1; // Minimum LIS ending at index i is 1
    for (int j = 0; j < i; ++j) {
        if (arr[j] < arr[i]) {
            maxLength = std::max(maxLength, LIS(j, arr, dp) + 1);
        }
    }
    dp[i] = maxLength;
    return dp[i];
}

// Function to find the length of the Longest Increasing Subsequence using memoization
int longestIncreasingSubsequenceMemoization(const std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;

    std::vector<int> dp(n, -1);
    int maxLength = 1;
    for (int i = 0; i < n; ++i) {
        maxLength = std::max(maxLength, LIS(i, arr, dp));
    }

    return maxLength;
}
```

_Code Fragment 20 - Function to solve the LIS problem using recursion and Memoization._{: class="legend"}

Let's break down how these two functions work together to solve the Longest Increasing Subsequence (LIS) problem using memoization.

The `LIS(int i, const std::vector<int>& arr, std::vector<int>& dp)` recursive function calculates the length of the LIS that ends at index $i$ within the input array `arr`. The `dp` vector acts as a memoization table, storing results to avoid redundant calculations. If `dp[i]` is not $-1$, it means the LIS ending at index `i` has already been calculated and stored. In this case, the function directly returns the stored value. If `dp[i]` is $-1$, the LIS ending at index `i` has not been computed yet. The function iterates through all previous elements (`j` from 0 to `i-1`) and checks if `arr[j]` is less than `arr[i]`. If so, it recursively calls itself to find the LIS ending at index `j`. The maximum length among these recursive calls (plus 1 to include the current element `arr[i]`) is then stored in `dp[i]` and returned as the result.

The `longestIncreasingSubsequenceMemoization(const std::vector<int>& arr)` function serves as a wrapper for the `LIS` function and calculates the overall LIS of the entire array. It initializes the `dp` array with -1 for all indices, indicating that no LIS values have been computed yet. It iterates through the array and calls the `LIS` function for each index `i`. It keeps track of the maximum length encountered among the results returned by `LIS(i)` for all indices. Finally, it returns this maximum length as the length of the overall LIS of the array.

Comparing the complexity of the memoization solution with the Brute-Force solution highlights significant differences in efficiency. The Brute-Force solution generates all possible subsequences using bitmasking, which results in a time complexity of $O(2^n \cdot n)$ due to the exponential number of subsequences and the linear time required to check each one. _In contrast, the memoization solution improves upon this by storing the results of previously computed LIS lengths, reducing redundant calculations. This reduces the time complexity to $O(n^2)$, as each element is compared with all previous elements, and each comparison is done once_. The space complexity also improves from potentially needing to store numerous subsequences in the Brute-Force approach to a linear $O(n)$ space for the memoization array in the Dynamic Programming solution. Thus, the memoization approach provides a more scalable and practical solution for larger arrays compared to the Brute-Force method. What can be seen in output 22:

```Shell
-----------------------------------
Calculating LIS in the array
Average time for LIS (Brute-Force): 690399 ns
Average time for LIS (Memoization): 3018 ns
-----------------------------------
```

_Output 11: Execution time for LIS solution using Memoization._{: class="legend"}

The provided output clearly demonstrates the significant performance advantage of memoization over the Brute-Force approach for calculating the Longest Increasing Subsequence (LIS). The Brute-Force method, with its average execution time of $690,399$ nanoseconds (ns), suffers from exponential time complexity, leading to a sharp decline in performance as the input size increases.

In contrast, the memoization approach boasts an average execution time of a mere $3,018$ ns. This dramatic improvement is a direct result of eliminating redundant calculations through the storage and reuse of intermediate results. In this particular scenario, memoization is approximately $228$ times faster than Brute-Force, highlighting the immense power of Dynamic Programming techniques in optimizing algorithms that involve overlapping subproblems.

Now, let's turn our attention to the last Dynamic Programming technique we are studying: tabulation.

### Tabulation

Tabulation, a bottom-up Dynamic Programming technique, iteratively computes and stores results in a table. For the LIS problem, we create a table `dp` where `dp[i]` represents the length of the LIS ending at index `i`.

Here's a breakdown of the steps involved:

1. A table `dp` is initialized with all values set to 1, representing the minimum LIS (the element itself) ending at each index.
2. Two nested loops are used to populate the `dp` table:
   - The outer loop iterates through the array from the second element (`i` from 1 to N-1).
   - The inner loop iterates through all preceding elements (`j` from 0 to i-1).
   - For each pair (i, j), if `arr[j]` is less than `arr[i]`, it signifies that `arr[i]` can extend the LIS ending at `arr[j]`. In this case, `dp[i]` is updated to `dp[i] = max(dp[i], dp[j] + 1)`.
3. After constructing the `dp` table, the maximum value within it is determined, representing the length of the longest increasing subsequence in the array.

This brings us to Code Fragment 21, which demonstrates the implementation of this tabulation approach:

```Cpp
// Function to find the length of the Longest Increasing Subsequence using tabulation
int longestIncreasingSubsequenceTabulation(const std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;

    std::vector<int> dp(n, 1);
    int maxLength = 1;

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (arr[i] > arr[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
        maxLength = std::max(maxLength, dp[i]);
    }

    return maxLength;
}
```

_Code Fragment 21 - Function to solve the LIS problem using recursion and Memoization._{: class="legend"}

The `int longestIncreasingSubsequenceTabulation(const std::vector<int>& arr)` function efficiently determines the length of the Longest Increasing Subsequence (LIS) in a given array using a tabulation approach.

Initially, a vector `dp` of size `n` (the array's length) is created, with all elements initialized to $1$. This signifies that the minimum LIS ending at each index is $1$ (the element itself). Additionally, a variable `maxLength` is initialized to $1$ to track the overall maximum LIS length encountered.

The function then employs nested loops to construct the `dp` table systematically. The outer loop iterates through the array starting from the second element (`i` from $1$ to `n-1`). The inner loop examines all previous elements (`j` from $0$ to `i-1`).

For each pair of elements (`arr[i]`, `arr[j]`), the function checks if `arr[i]` is greater than `arr[j]`. If so, it means `arr[i]` can extend the LIS ending at `arr[j]`. In this case, `dp[i]` is updated to the maximum of its current value and `dp[j] + 1` (representing the length of the LIS ending at `j` plus the current element `arr[i]`).

After each iteration of the outer loop, `maxLength` is updated to the maximum of its current value and `dp[i]`. This ensures that `maxLength` always reflects the length of the longest LIS found so far.

Finally, the function returns `maxLength`, which now holds the length of the overall LIS in the entire array.

In terms of complexity, the tabulation solution is on par with the memoization approach, both boasting a time complexity of $O(n^2)$. This is a substantial improvement over the Brute-Force method's exponential $O(2^n \times n)$ time complexity. The quadratic time complexity arises from the nested loops in both tabulation and memoization, where each element is compared with all preceding elements to determine potential LIS extensions.

However, _the tabulation approach has a slight edge in space complexity. While memoization requires $O(n)$ space to store the memoization table (``dp` array), tabulation only necessitates $O(n)$ space for the dp table as well._ This is because memoization might incur additional space overhead due to the recursive call stack, especially in cases with deep recursion.

The Output 12 demonstrates this performance difference. Both memoization and tabulation significantly outperform the Brute-Force approach, which takes an exorbitant amount of time compared to the other two. While the difference between memoization and tabulation is less pronounced in this specific example, tabulation can sometimes be slightly faster due to the overhead associated with recursive function calls in memoization.

```Shell
-----------------------------------
Calculating LIS in the array
Average time for LIS (Brute-Force): 694237 ns
Average time for LIS (Memoization): 2484 ns
Average time for LIS (Tabulation): 2168 ns
-----------------------------------
```

_Output 12: Execution time for LIS solution using Tabulation._{: class="legend"}

Ultimately, the choice between memoization and tabulation often comes down to personal preference and specific implementation details. Both offer substantial improvements over Brute-Force and are viable options for solving the LIS problem efficiently.
