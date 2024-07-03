---
layout: post
title: Exploring Dynamic Programming in C++ - Techniques and Performance Insights
author: Frank
categories:
    - Matemática
    - Linguagens Formais
    - Programação
tags:
    - Matemática
    - Linguagens Formais
    - Programação Dinâmica
    - Dynamic Programming
    - Dynamic Programming
    - C++ Algorithms
    - Performance Analysis
    - Coding Examples
    - Algorithm Optimization
    - Practical Programming Guide
image: assets/images/prog_dinamic.jpeg
description: Dynamic programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
slug: dynamic-programming
keywords:
    - Dynamic Programming
    - C++ Algorithms
    - Coding Examples
    - Performance Optimization
    - Algorithm Efficiency
    - Programming Guide
    - Code Comparison
    - Developer Tips
rating: 5
published: 2024-06-27T19:43:15.124Z
draft: null
preview: In this comprehensive guide, we delve into the world of dynamic programming with C++. Learn the core principles of dynamic programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and enhance algorithm efficiency.
---

Dynamic programming is a different way of thinking when it comes to solving problems. Programming itself is already a different way of thinking, so, to be honest, I can say that dynamic programming is a different way within a different way of thinking. And, if you haven't noticed yet, there is a concept of recursion trying to emerge in this definition.

The general idea is that you, dear reader, should be able to break a large and difficult problem into small and easy pieces. To do this, we will store information and reuse it whenever necessary in our algorithm.

It is very likely that you, kind reader, have been introduced to dynamic programming techniques while studying algorithms without realizing it. So, it is also very likely that you will encounter, in this text, algorithms you have seen before without knowing they were dynamic programming.

My intention is to break down the dynamic programming process into clear steps, focusing on the solution algorithm, so that you can understand and implement these steps on your own whenever you face a problem in technical interviews, production environments, or programming competitions. Without any hesitation, I will try to present performance tips and tricks in C++. However, this should not be considered a limitation; we will look at the algorithms before the code, and you will be able to implement the code in your preferred programming language.

## There was a hint of recursion sneaking in.

Some say that dynamic programming is a technique to make recursive code more efficient. There is a relationship that needs to be explored: *All dynamic programming algorithms are recursive, but not all recursive algorithms are dynamic programming*.

Recursion is a powerful problem-solving technique. Recursive code can be mathematically proven correct relatively easily. And that alone is reason enough to use recursion in all your code. So, let's begin with that.

The proof of the correctness of a recursive algorithm generally involves only two steps: Proving that the base case of the recursion is correct and proving that the recursive step is correct. In the domain of mathematical induction proof, we can refer to these components as the *base case* and the *inductive step*, respectively. In this case:

- To prove the **base case**, we check the simplest # case of the recursion, usually the base case or cases, to ensure it is correct. These are the cases that do not depend on recursive calls.

- To prove the **inductive step**, we verify that if the recursive function is correct for all smaller cases or subproblems, then it is also correct for the general case. In other words, we assume that the function is correct for smaller inputs, or for a smaller set of inputs, the induction hypothesis, and based on this, we prove, or not, that the recursive function is correct.

Beyond the ease of mathematical proof, recursive code stands out for being clear and intuitive, especially for problems with repetitive structures such as tree traversal, maze solving, and calculating mathematical series.

Many problems are naturally defined in a recursive manner. For example, the mathematical definition of the Fibonacci sequence or the structure of binary trees are inherently recursive. In these cases, the recursive solution will be simpler, more straightforward, and likely more efficient.

Often, the recursive solution is more concise and requires fewer lines of code compared to the iterative solution. Fewer lines, fewer errors, easier to read and understand. Sounds good.

Finally, recursion is an almost ideal approach for applying divide-and-conquer techniques. Since Julius Caesar, we know it is easier to divide and conquer. In this case, a problem is divided into subproblems, solved individually, and then combined to form the final solution. Classic academic examples of these techniques include sorting algorithms like quicksort and mergesort.

The sweet reader might have raised her eyebrows. This is where recursion and dynamic programming touch, not subtly and delicately, like a lover's caress on the beloved's face. But with the decisiveness and impact of Mike Tyson's glove on the opponent's chin. The division of the main problem into subproblems is the fundamental essence of both recursion and dynamic programming.

Dynamic programming and recursion are related; *both involve solving problems by breaking a problem into smaller problems. However, while recursion solves the smaller problems without considering the computational cost of repeated calls, dynamic programming optimizes these solutions by storing and reusing previously obtained results*. The most typical example of recursion is determining the nth order value of the Fibonacci sequence can be seen in Flowchart 1.

![]({{ site.baseurl }}/assets/images/recursive-memo.jpg)
*Flowchart 1 - Recursive Fibonacci nth algorithm*

The Flowchart 1 represents a function for calculating the nth number of the Fibonacci Sequence, for all $n \geq 0$ as the desired number.

In Flowchart 1 we will have:

- **Base Case**: The base case is the condition that terminates the recursion. For the Fibonacci sequence, the base cases are for $n = 0$ and $n = 1$:
  - When $n = 0$, the function returns 0.
  - When $n = 1$, the function returns 1.

- **Recursive Step**: The recursive step is the part of the function that calls itself to solve smaller subproblems. In the Fibonacci sequence, each number is the sum of the two preceding ones:

$$F(n) = F(n - 1) + F(n - 2)$$

When the function receives a value $n$:

- **Base Case**: It checks if $n$ is 0 or 1. If so, it returns $n$.
- **Recursive Step**: If $n$ is greater than 1, the function calls itself twice: Once with $n - 1$ and once with $n - 2$. The sum of these two results is returned.

This leads us to Example 1.

### Example 1: Let's calculate `fibonacci(5)`:

1. `fibonacci(5)` calls `fibonacci(4)` and `fibonacci}(3)`
2. `fibonacci(4)` calls `fibonacci(3)` and `fibonacci}(2)`
3. `fibonacci(3)` calls `fibonacci(2)` and `fibonacci}(1)`
4. `fibonacci(2)` calls `fibonacci(1)` and `fibonacci(0)`
5. `fibonacci(1)` returns 1
6. `fibonacci(0)` returns 0
7. `fibonacci(2)` returns $1 + 0 = 1$
8. `fibonacci(3)` returns $1 + 1 = 2$
9. `fibonacci(2)` returns 1 (recalculated)
10. `fibonacci(4)` returns $2 + 1 = 3$
11. `fibonacci(3)` returns 2 (recalculated)
12. `fibonacci(5)` returns $3 + 2 = 5$

Thus, `fibonacci(5)` returns $5$. The function breaks down recursively until it reaches the base cases, then combines the results to produce the final value.

And we can write it in Python, using Python as a kind of pseudocode, as:

```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

In Example 1, the `fibonacci` function calls itself to calculate the preceding terms of the Fibonacci Sequence. Note that for each desired value, we have to go through all the others. This is an example of correct and straightforward recursion and, in this specific case, very efficient. We will look at this efficiency issue more carefully later.

To determine the number of times the `fibonacci` function is called to calculate the 5th number in the Fibonacci sequence, we can analyze the recursive call tree. Let's count all the function calls, including the duplicate calls.

Using the previous example and counting the calls:

1. `fibonacci(1)``fibonacci(5)`
2. `fibonacci(4)` + `fibonacci(3)`
3. `fibonacci(3)` + `fibonacci(2)` + `fibonacci(2)` + `fibonacci(1)`
4. `fibonacci(2)` + `fibonacci(1)` + `fibonacci(1)` + `fibonacci(0)` + `fibonacci(1)` + `fibonacci(0)`
5. `fibonacci(1)` + `fibonacci(0)`

Counting all the calls, we have:

- `fibonacci(5)`: 1 call
- `fibonacci(4)`: 1 call
- `fibonacci(3)`: 2 calls
- `fibonacci(2)`: 3 calls
- `fibonacci(1)`: 5 calls
- `fibonacci(0)`: 3 calls

Total: 15 calls. Therefore, the `fibonacci(n)` function is called 15 times to calculate `fibonacci(5)`.

To calculate the number of times the function will be called for any value $n$, we can use the following formula based on the analysis of the recursion tree:

$$ T(n) = T(n-1) + T(n-2) + 1 $$

Where $T (n) $ is the total number of calls to calculate `fibonacci(n)`.

To illustrate the formula $T(n) = T(n-1) + T(n-2) + 1$ with $n = 10$, we can calculate the number of recursive calls $T(10)$. Let's start with the base values $T(0)$ and $T(1)$, and then calculate the subsequent values up to $T(10)$.

Assuming that $T(0) = 1$ and $T(1) = 1$:

$$
\begin{aligned}
T(2) &= T(1) + T(0) + 1 = 1 + 1 + 1 = 3 \\
T(3) &= T(2) + T(1) + 1 = 3 + 1 + 1 = 5 \\
T(4) &= T(3) + T(2) + 1 = 5 + 3 + 1 = 9 \\
T(5) &= T(4) + T(3) + 1 = 9 + 5 + 1 = 15 \\
T(6) &= T(5) + T(4) + 1 = 15 + 9 + 1 = 25 \\
T(7) &= T(6) + T(5) + 1 = 25 + 15 + 1 = 41 \\
T(8) &= T(7) + T(6) + 1 = 41 + 25 + 1 = 67 \\
T(9) &= T(8) + T(7) + 1 = 67 + 41 + 1 = 109 \\
T(10) &= T(9) + T(8) + 1 = 109 + 67 + 1 = 177 \\
\end{aligned}
$$

Therefore, $T(10) = 177$.

Each value of $T(n)$ represents the number of recursive calls to compute $\text{fibonacci}(n)$ using the formula $T(n) = T(n-1) + T(n-2) + 1$. If everything is going well, at this point the esteemed reader must be thinking about creating a recursive function to count how many times the `fibonacci(n)` function will be called. But let's try to avoid this recursion of recursions.

This formula can be used to build a recursion tree and sum the total number of recursive calls. However, for large values of $n$, this can become inefficient. A more efficient approach is to use dynamic programming to calculate and store the number of recursive calls, avoiding duplicate calls.

## Returning to Dynamic Programming

If we look at dynamic programming, we will see an optimization technique that is based on recursion but adds storage of intermediate results to avoid redundant calculations. There are two main approaches to implementing dynamic programming:

- **Memoization (Top-Down)**: Stores the results of recursive calls in a data structure (such as a dictionary or a list, etc.) for reuse. The name memoization is a horrible borrowing from the English word *memoization*.

- **Tabulation (Bottom-Up)**: Solves the problem iteratively, filling a table (usually a list or matrix) with the results of the subproblems.

In this case, we can see two examples in Python. First, an example of Dynamic Programming with memoization:

### Example 2: Memoization

Let's continue with the same problem, finding the nth number in the Fibonacci Sequence. This time, using dynamic programming with memoization. This problem can be described with Flowchart 2.

![]({{ site.baseurl }}/assets/images/recursive-memo.jpg)
*Flowchart 2 - Recursive Fibonacci nth algorithm with memoization*

```python
# Criação do dicionário memo
memo = {}
def fibonacci_memo(n, memo):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

Let's analyze the provided code in detail.

#### Function Definition and Initialization

```python
memo = {}
def fibonacci_memo(n, memo):
```

In this code, a dictionary named `memo` is used to store the results of previous Fibonacci calculations, preventing redundant calculations and improving efficiency (this is memoization!). The `fibonacci_memo` function is then defined to calculate the n-th Fibonacci number using the stored values in this dictionary, which brings us to the consideration of the base case within recursion.

#### Base Case

```python
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
```

if $n$ in `memo`: Checks if the value of $n$ has already been calculated and stored in the `memo` dictionary. If so, it returns the stored value, avoiding recalculation.

if $n <= 1$: Handles the base cases of the Fibonacci sequence:

- `fibonacci(0)` = 0
- `fibonacci(1)` = 1

#### Recursive Step and Memoization

```python
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

The line `fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)` makes recursive calls to calculate the $(n-1)$th and $(n-2)$th Fibonacci numbers. This is the fundamental recursive relationship in the Fibonacci Sequence: each number is the sum of the two preceding ones.

The `memo` dictionary is the key to memoization. Before making the recursive calls, the function checks if the results for $n-1$ and $n-2$ are already stored in `memo`. If so, those stored values are used directly, avoiding redundant calculations. If not, the recursive calls are made, and the results are stored in `memo` for future reference.

The calculated result (`fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)`) is assigned to `memo[n]`, storing it for potential reuse later.

Finally, return `memo[n]` returns the calculated (and now memoized) value for the n-th Fibonacci number.

From the perspective of dynamic programming, the function `fibonacci_memo` divides the larger problem (calculating Fibonacci of $n$) into smaller subproblems (calculating Fibonacci of $n-1$ and $n-2$), uses a data structure, the `memo` dictionary, to store the results of the subproblems. This avoids redundant calculations of the same values, and before calculating the Fibonacci value for a given $n$, the function checks if the result is already stored in the `memo` dictionary. If it is, it reuses that result, saving computation time. Finally, the function ensures that each subproblem is solved only once, resulting in more efficiency compared to the simple recursive approach.

The last statement of the previous paragraph requires reflection. I am considering performance in this statement only in terms of computation time. Performance can also be considered in relation to memory usage, energy consumption, and any other factor that is interesting or important for a given problem. Keep this in mind whenever I state that performance has improved in this text. Well, who is thinking about a example?

#### How many recursive calls does a memoized Fibonacci function make?

To figure this out, let's see how memoization changes the usual recursion tree:

Base Case: If the Fibonacci number for n is already stored in our memoization cache, or if $n$ is $0$ or $1$, the function returns directly without any further calls.

Memoization Check:  If $n$ isn't in the cache, the function makes two recursive calls: `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.

The Memoization Effect: The very first time we call `fibonacci_memo` with a new value of $n$, it will keep making recursive calls until it hits the base cases.  The key is that once a Fibonacci number is calculated, it gets stored in the cache. Any later calls with the same $$ simply return this stored value, preventing further recursion.

Calculating the Number of Calls:

Initial Call: We start the whole process with a single call to `fibonacci_memo(n, memo)`.

Recursive Expansion: For every new $n$ value we encounter, the function branches out into calls for `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.

Memoization Storage: Each calculated value is stored, so any future calls with the same $n$ don't create new branches.

Counting Unique Calls: Because of memoization, we only need to calculate each Fibonacci number once. This means the total number of recursive calls is roughly equal to the number of unique Fibonacci numbers up to $n$.

In conclusion: While a naive Fibonacci implementation would have an exponential number of calls, memoization brings this down significantly. We still have roughly $2n$ calls to calculate Fibonacci numbers from $0$ to $n$, but the key is that each unique number is only calculated once, making the process efficient.

To calculate the number of times the function will be called for any value $n$, we can use the following formula based on the analysis of the memoized recursion tree:

$$ T(n) = T(n-1) + T(n-2) + 1 $$

Where $T(n)$ is the total number of calls to calculate `fibonacci_memo(n)`.

To illustrate the formula $T(n) = T(n-1) + T(n-2) + 1$ with $n = 10$, we can calculate the number of recursive calls $T(10)$. Let's start with the base values $T(0)$ and $T(1)$, and then calculate the subsequent values up to $T(10)$.

Assuming that $T(0) = 1$ and $T(1) = 1$:

$$
\begin{aligned}
T(2) &= T(1) + T(0) + 1 = 1 + 1 + 1 = 3 \\
T(3) &= T(2) + T(1) + 1 = 3 + 1 + 1 = 5 \\
T(4) &= T(3) + T(2) + 1 = 5 + 3 + 1 = 9 \\
T(5) &= T(4) + T(3) + 1 = 9 + 5 + 1 = 15 \\
T(6) &= T(5) + T(4) + 1 = 15 + 9 + 1 = 25 \\
T(7) &= T(6) + T(5) + 1 = 25 + 15 + 1 = 41 \\
T(8) &= T(7) + T(6) + 1 = 41 + 25 + 1 = 67 \\
T(9) &= T(8) + T(7) + 1 = 67 + 41 + 1 = 109 \\
T(10) &= T(9) + T(8) + 1 = 109 + 67 + 1 = 177 \\
\end{aligned}
$$

Therefore, $T(10) = 177$.

Each value of $T(n)$ represents the number of recursive calls to compute `fibonacci_memo(n)` using the formula $T(n) = T(n-1) + T(n-2) + 1$. And we have only 10 recursive calls.

### Example 3: Fibonacci with Tabulation

Finally, we can have an example of Dynamic Programming with Tabulation. Again using Python as a kind of pseudocode:

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

In this example, the `fibonacci_tabulation()` function uses a list `dp` to store the results of all subproblems, building the solution from the bottom up. Just to highlight, in Example 2 data is being stored.

This is true! But look closely. The `fibonacci_tabulation(2)` function is an example of tabulation, not memoization, due to the specific characteristics of how the subproblems are solved and stored.

Tabulation is a bottom-up approach to dynamic programming where you solve all subproblems first and store their solutions in a data structure, usually a table, array, list, or tree. The solution to the larger problem is then built from these smaller solutions by traversing the data structure from the bottom up. This implies an iterative resolution process. The subproblems are solved iteratively, starting from the smallest until the larger problem is reached. In this case, recursion is irrelevant.

## There is more between heaven and earth, Mr. Shakespeare

Memoization and Tabulation are the most common techniques, but they are not the only dynamic programming techniques:

- **Dynamic Programming with State Compression**: The goal is to reduce the space needed to store the results of the subproblems by keeping only the states relevant to calculating the final solution.
- **Dynamic Programming with Sliding Window**: Maintains only the results of the most recent subproblems in a fixed-size window, useful when the solution depends only on a limited number of previous subproblems.
- **Dynamic Programming with Decision Tree**: Represents the subproblems and their relationships in a decision tree, allowing a clear visualization of the problem structure and the decisions to be made.

Let's see how far we go in this text. At the moment of writing, I still have no idea.

## Now I realize: C++, where is C++?

Python, which I used as pseudocode, is a versatile and simple language. However, it is still not the most suitable language for high-performance use or programming competitions. Therefore, we will move to C++ 20 and, eventually, use data structures compatible with C 17, even in the C++ 20 environment. Speaking of the environment, from this point on, I will be using Visual Studio Community edition to run and evaluate all the code. To maintain consistency in our text so far, I will convert the same functions we wrote in Python to C++ and assess the results.

### Code 1: `std::vectors`

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

    // Set locale to Portuguese of Brazil with UTF-8 support
    std::setlocale(LC_ALL, "pt_BR.UTF-8");
    std::wcout.imbue(std::locale("pt_BR.UTF-8"));

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

This simple and intuitive code generates a Fibonacci number, stores it in an integer (`int`), and then, for testing purposes, finds 3 specific Fibonacci numbers—the 10th, 20th, and 30th—1000 times each. This code uses `std::vectors` and `std::unordered_map` for storing the values of the Fibonacci sequence and, when executed, presents the following result.

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

The kind reader should note that the times vary in a non-linear fashion and that, in all cases, for this problem, the dynamic programming version using tabulation was faster. Much is said about the performance of the Vector class compared to the Array class.

`std::vector` is a template class and a C++-only construct implemented as a dynamic array. Vectors grow and shrink dynamically, automatically managing their memory, which is freed upon destruction. They can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements. Unlike arrays, vectors do not decay to pointers, but you can explicitly get a pointer to their data using `&vec[0]`. Vectors maintain their size (number of elements currently stored) and capacity (number of elements that can be stored in the currently allocated block) along with the internal dynamic array. This internal array is allocated dynamically by the allocator specified in the template parameter, usually obtaining memory from the freestore (heap) independently of the object's actual allocation. Although this can make vectors less efficient than regular arrays for small, short-lived, local arrays, vectors do not require a default constructor for stored objects and are better integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, vectors copy (or move, in C++11) their objects.

`std::array` is a template class introduced in C++11, which provides a fixed-size array that is more integrated with the STL than traditional C-style arrays. Unlike `std::vector`, `std::array` does not manage its own memory dynamically; its size is fixed at compile-time, which makes it more efficient for cases where the array size is known in advance and does not change. `std::array` objects can be passed to and returned from functions, and they support copy and assignment operations. They provide the same `begin()`/`end()` methods as vectors, allowing for easy iteration and integration with other STL algorithms. One significant advantage of `std::array` over traditional arrays is that it encapsulates the array size within the type itself, eliminating the need for passing size information separately. Additionally, `std::array` provides member functions such as `size()`, which returns the number of elements in the array, enhancing safety and usability. However, since `std::array` has a fixed size, it does not offer the dynamic resizing capabilities of `std::vector`, making it less flexible in scenarios where the array size might need to change.

When considering performance differences between `std::vector` and `std::array`, it's essential to understand their underlying characteristics and use cases. `std::array` is a fixed-size array, with its size determined at compile-time, making it highly efficient for situations where the array size is known and constant. The lack of dynamic memory allocation means that `std::array` avoids the overhead associated with heap allocations, resulting in faster access and manipulation times. This fixed-size nature allows the compiler to optimize memory layout and access patterns, often resulting in better cache utilization and reduced latency compared to dynamically allocated structures.

In contrast, `std::vector` provides a dynamic array that can grow or shrink in size at runtime, offering greater flexibility but at a cost. The dynamic nature of `std::vector` necessitates managing memory allocations and deallocations, which introduces overhead. When a `std::vector` needs to resize, it typically allocates a new block of memory and copies existing elements to this new block, an operation that can be costly, especially for large vectors. Despite this, `std::vector` employs strategies such as capacity doubling to minimize the frequency of reallocations, balancing flexibility and performance.

For small, fixed-size arrays, `std::array` usually outperforms `std::vector` due to its minimal overhead and compile-time size determination. It is particularly advantageous in performance-critical applications where predictable and low-latency access is required. On the other hand, `std::vector` shines in scenarios where the array size is not known in advance or can change, offering a more convenient and safer alternative to manually managing dynamic arrays.

In summary, `std::array` generally offers superior performance for fixed-size arrays due to its lack of dynamic memory management and the resultant compiler optimizations. However, `std::vector` provides essential flexibility and ease of use for dynamically sized arrays, albeit with some performance trade-offs. The choice between `std::array` and `std::vector` should be guided by the specific requirements of the application, weighing the need for fixed-size efficiency against the benefits of dynamic resizing.

| Feature        | `std::vector`                 | `std::array`                    |
| -------------- | ------------------------------ | ------------------------------- |
| Size           | Dynamic (can change at runtime) | Fixed (determined at compile time) |
| Memory Management | Dynamic allocation on the heap     | Typically on the stack, no dynamic allocation |
| Performance     | Can have overhead due to resizing | Generally more efficient for fixed-size data |
| Use Cases       | When the number of elements is unknown or varies | When the number of elements is known and fixed |
| Flexibility     | High (can add/remove elements easily) | Low (size cannot be changed)       |
| STL Integration | Yes (works with algorithms and iterators) | Yes (similar interface to vector) |

So, we can test this performance advantages.

### Code 2: `std::array`

```Cpp
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <array>

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

Which, when executed, produces the following output:

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

We have reached an interesting point. Just interesting. We achieved a performance gain using memoization and tabulation. However, we still have some options.

I will continue later.
