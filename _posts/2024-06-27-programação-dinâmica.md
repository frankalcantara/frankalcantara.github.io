---
layout: post
title: Dynamic Programming in C++ - Techniques and Insights
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
image: assets/images/prog_dynamic.jpeg
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
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
featured: true
toc: true
preview: "In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Dynamic Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and enhance algorithm efficiency."

beforetoc: "In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Dynamic Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and enhance algorithm efficiency."
---

Dynamic Programming is a different way of thinking when it comes to solving problems. Programming itself is already a different way of thinking, so, to be honest, I can say that Dynamic Programming is a different way within a different way of thinking. And, if you haven't noticed yet, there is a concept of recursion trying to emerge in this definition.

The general idea is that you, dear reader, should be able to break a large and difficult problem into small and easy pieces. This involves storing and reusing information within the algorithm as needed.

It is very likely that you, kind reader, have been introduced to Dynamic Programming techniques while studying algorithms without realizing it. So, it is also very likely that you will encounter, in this text, algorithms you have seen before without knowing they were Dynamic Programming.

My intention is to break down the Dynamic Programming process into clear steps, focusing on the solution algorithm, so that you can understand and implement these steps on your own whenever you face a problem in technical interviews, production environments, or programming competitions. Without any hesitation, I will try to present performance tips and tricks in C++. However, this should not be considered a limitation; we will prioritize understanding the algorithms before diving into the code, and you will be able to implement the code in your preferred programming language.

## There was a hint of recursion sneaking in

Some say that Dynamic Programming is a technique to make recursive code more efficient. There is a connection worth examining: *All Dynamic Programming algorithms are recursive, but not all recursive algorithms are Dynamic Programming*.

Recursion is a powerful problem-solving technique. Recursive code can be mathematically proven correct relatively easily. This clarity alone makes a compelling case for using recursion. So, let's begin with that.

The proof of the correctness of a recursive algorithm generally involves only two steps: Proving that the base case of the recursion is correct and proving that the recursive step is correct. In the domain of mathematical induction proof, we can refer to these components as the *base case* and the *inductive step*, respectively. In this case:

- To prove the **base case**, we check the simplest # case of the recursion, usually the base case or cases, to ensure it is correct. These are the cases that do not depend on recursive calls.

- To prove the **inductive step**, we verify that if the recursive function is correct for all smaller cases or subproblems, then it is also correct for the general case. In other words, we assume that the function is correct for smaller inputs, or for a smaller set of inputs, the induction hypothesis, and based on this, we prove, or not, that the recursive function is correct.

Beyond the ease of mathematical proof, which, by the way, is your problem, recursive code stands out for being clear and intuitive, especially for problems with repetitive structures such as tree traversal, maze solving, and calculating mathematical series.

Many problems are naturally defined in a recursive manner. For example, the mathematical definition of the Fibonacci Sequence or the structure of binary trees are inherently recursive. In these cases, the recursive solution will be simpler, more straightforward, and likely more efficient.

Often, the recursive solution is more concise and requires fewer lines of code compared to the iterative solution. Fewer lines, fewer errors, easier to read and understand. Sounds good.

Finally, recursion is an almost ideal approach for applying divide-and-conquer techniques. Since Julius Caesar, we know it is easier to divide and conquer. In this case, a problem is divided into subproblems, solved individually, and then combined to form the final solution. Classic academic examples of these techniques include sorting algorithms like quicksort and mergesort.

The sweet reader might have raised her eyebrows. This is where recursion and Dynamic Programming touch, not subtly and delicately, like a lover's caress on the beloved's face. But with the decisiveness and impact of Mike Tyson's glove on the opponent's chin. The division of the main problem into subproblems is the fundamental essence of both recursion and Dynamic Programming.

Dynamic Programming and recursion are related; *both involve solving problems by breaking a problem into smaller problems. However, while recursion solves the smaller problems without considering the computational cost of repeated calls, Dynamic Programming optimizes these solutions by storing and reusing previously obtained results*.A classic illustration of recursion is the calculation of the nth term in the Fibonacci Sequence, as depicted in Flowchart 1.

![]({{ site.baseurl }}/assets/images/recursive.jpg)
*Flowchart 1 - Recursive Fibonacci nth algorithm*{: class="legend"}

The Flowchart 1 represents a function for calculating the nth number of the Fibonacci Sequence, for all $n \geq 0$ as the desired number. In Flowchart 1 we have:

- **Base Case**: *The base case is the condition that terminates the recursion*. For the Fibonacci Sequence, the base cases are for $n = 0$ and $n = 1$:
  - When $n = 0$, the function returns $0$.
  - When $n = 1$, the function returns $1$.

- **Recursive Step**: *The recursive step is the part of the function that calls itself to solve smaller subproblems*. In the Fibonacci Sequence, each number is the sum of the two preceding ones: $F(n) = F(n - 1) + F(n - 2)$ which leads to a representation of the base cases as: $F(0) = 0$ and $F(1) = 1$

When the function receives a value $n$:

- **Base Case**: It checks if $n$ is 0 or 1. If so, it returns $n$.
- **Recursive Step**: If $n$ is greater than 1, the function calls itself twice: Once with $n - 1$ and once with $n - 2$. The sum of these two results is returned.

To illustrate, let's calculate the tenth Fibonacci number.

### Example 1: Recursion

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

*Code Fragment 1 - Recursive Fibonacci Function for the nth Term*{: class="legend"}

In Example 1, the `fibonacci` function calls itself to calculate the preceding terms of the Fibonacci Sequence. Note that for each desired value, we have to go through all the others. This is an example of correct and straightforward recursion and, in this specific case, very efficient. We will look at this efficiency issue more carefully later.

#### Calculate the Number of Recursive Calls

To quantify the number of times the `fibonacci` function is called to calculate the 5th number in the Fibonacci Sequence, we can analyze the recursive call tree. "Let's enumerate all function calls, accounting for duplicates.

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

To understand the computational cost of the recursive Fibonacci function, let's analyze the number of function calls made for a given input $n$. We can express this using the following recurrence relation:

$$T(n) = T(n-1) + T(n-2) + 1$$

where:

- $T(n)$ represents the total number of calls to the `fibonacci(n)` function.
- $T(n-1)$ represents the number of calls made when calculating fibonacci(n-1).
- $T(n-2)$ represents the number of calls made when calculating fibonacci(n-2).
- The $+ 1$ term accounts for the current call to the `fibonacci(n)` function itself.

The base cases for this recurrence relation are:

- $T(0) = 1$
- $T(1) = 1$

These base cases indicate that calculating `fibonacci(0)` or `fibonacci(1)` requires a single function call.

To illustrate the formula $T(n) = T(n-1) + T(n-2) + 1$ with $n = 10$, we can calculate the number of recursive calls $T(10)$. Let's start with the base values $T(0)$ and $T(1)$, and then calculate the subsequent values up to $T(10)$. Therefore we will have:

$$
\begin{aligned}
T(0) &= 1 \\
T(1) &= 1 \\
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

Each value of $T(n)$ represents the number of recursive calls to compute `fibonacci(n)` using the formula $T(n) = T(n-1) + T(n-2) + 1$. If everything is going well, at this point the esteemed reader must be thinking about creating a recursive function to count how many times the `fibonacci(n)` function will be called. But let's try to avoid this recursion of recursions in recursions to keep our fragile sanity.

The formula $T(n) = T(n-1) + T(n-2) + 1$ can be used to build a recursion tree, as we can see in Figure 1, and sum the total number of recursive calls. However, for large values of $n$, this can become inefficient. A more efficient approach is to use Dynamic Programming to calculate and store the number of recursive calls, avoiding duplicate calls.

![]({{ site.baseurl }}/assets/images/recursion_tree.jpg)
*Figure 1 - Recursive Tree for Fibonacci 5*{: class="legend"}

#### Space and Time Efficiency

The `fibonacci(n)` function uses a straightforward recursive approach to calculate Fibonacci numbers. Let's analyze its time and space complexity..

To understand the time complexity, consider the frequency of function calls. Each call to `fibonacci(n)` results in two more calls: `fibonacci(n-1)` and `fibonacci(n-2)`. This branching continues until we reach the base cases.

Imagine this process as a tree, as we saw earlier:

- The root is `fibonacci(n)`.
- The next level has two calls: `fibonacci(n-1)` and `fibonacci(n-2)`.
- The level after that has four calls, and so on.

Each level of the tree results in a doubling of calls. If we keep doubling for each level until we reach the base case, we end up with about $2^n$ calls. This exponential growth results in a time complexity of O(2^n), meaning a exponencial growth. This is quite inefficient because the number of calls increases very quickly as $n$ gets larger.

The space complexity depends on how deep the recursion goes. Every time the function calls itself, it adds a new frame to the call stack.

- The maximum depth of recursion is $n$ levels (from `fibonacci(n)` down to `fibonacci(0)` or `fibonacci(1)`).

Therefore, the space complexity is $O(n)$, because the stack can grow linearly with $n$.

In short, *the recursive `fibonacci` function is simple but inefficient for large $n$ due to its exponential time complexity*. This conclusion justifies the need for Dynamic Programming.

## Returning to Dynamic Programming

If we look at Dynamic Programming, we will see an optimization technique that is based on recursion but adds storage of intermediate results to avoid redundant calculations. *Memoization and tabulation are the two most common Dynamic Programming techniques*, each with its own approach to storing and reusing the results of subproblems:

- **Memoization (Top-Down)**: *This technique is recursive in nature*. It involves storing the results of expensive function calls and returning the cached result when the same inputs occur again. This approach can be seen as an optimization of the top-down recursive process.
- **Tabulation (Bottom-Up**): *Tabulation takes an iterative approach, solving smaller subproblems first and storing their solutions in a table (often an array or matrix)*. It then uses these stored values to calculate the solutions to larger subproblems, gradually building up to the final solution. The iterative nature of tabulation typically involves using loops to fill the table in a systematic manner.

At this point, we can take a look at two examples using Python as pseudocode, since most of my students feel comfortable with Python. First, an example of Dynamic Programming with memoization:

### Example 2: Memoization

Let's revisit the Fibonacci sequence problem: finding the nth number in the Fibonacci sequence. This time, we'll use Dynamic Programming with memoization. Flowchart 2 illustrates this approach.

![]({{ site.baseurl }}/assets/images/recursive-memo.jpg)
*Flowchart 2 - Recursive Fibonacci nth algorithm with memoization*{: class="legend"}

From Flowchart 2, we can derive the following Python code:

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

*Code Fragment 2 - Memoization Function for the nth Term*{: class="legend"}

The `fibonacci_memo()` function is then defined to calculate the nth Fibonacci number using the stored values in a dictionary. Let's analyze the `fibonacci_memo()` code in detail.

#### Function Definition and Initialization

The `fibonacci_memo()` function begins by:

```python
memo = {}
def fibonacci_memo(n, memo):
```

In this code fragment, there is a dictionary named memo declared as `memo = {}`. It will be used to *store the results of previous Fibonacci calculations, preventing redundant calculations and improving efficiency* (this is memoization!), which brings us to the consideration of the base case within recursion.

#### Base Case

```python
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
```

The line `if n in memo` checks if the value of $𝑛$ has already been calculated and stored in the `memo` dictionary. If so, it returns the stored value, avoiding recalculation. On the other hand, `if n <= 1` handles the base cases of the Fibonacci sequence:

- `fibonacci(0) = 0`
- `fibonacci(1) = 1`

#### Recursive Step and Memoization

```python
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

The expression `fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)` initiates recursive calls to determine the $(n-1)$th and $(n-2)$th Fibonacci numbers. This is the fundamental recursive relationship in the Fibonacci Sequence: each number is the sum of the two preceding ones.

The `memo` dictionary is the key to memoization. Before making the recursive calls, the function checks if the results for $n-1$ and $n-2$ are already stored in `memo`. If so, those stored values are used directly, avoiding redundant calculations. If not, the recursive calls are made, and the results are stored in `memo` for future reference.

The calculated result (`fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)`) is assigned to `memo[n]`, storing it for potential reuse later.

Finally, return `memo[n]` returns the calculated (and now memoized) value for the nth Fibonacci number.

From the perspective of Dynamic Programming, the `fibonacci_memo` function employs a divide-and-conquer strategy, breaking down the calculation of the nth Fibonacci number into smaller subproblems (calculating the ($n-1$)th and ($n-2$)th numbers). It leverages a dictionary, memo, to store and retrieve the results of these subproblems. This approach eliminates redundant computations, enhancing efficiency, and before calculating the Fibonacci value for a given $n$, the function checks if the result is already stored in the `memo` dictionary. If it is, it reuses that result, saving computation time. Finally, the function ensures that each subproblem is solved only once, resulting in more efficiency compared to the simple recursive approach.

The last statement of the previous paragraph requires reflection. I am considering performance in this statement only in terms of computation time. Performance can also be considered in relation to memory usage, energy consumption, and any other factor that is interesting or important for a given problem. Keep this in mind whenever I state that performance has improved in this text.

Performance can be evaluated through complexity analysis. When analyzing the complexity of an algorithm, we often refer to its time complexity and space complexity. *Time complexity refers to the amount of time an algorithm takes to run as a function of the size of its input. Space complexity refers to the amount of memory an algorithm uses as a function of the size of its input*. Both are crucial aspects of performance.

For example, the naive recursive Fibonacci algorithm has a time complexity of $O(2^n)$ because it makes an exponential number of calls. With memoization, the time complexity is reduced to $O(n)$ since each Fibonacci number up to $n$ is computed only once. The space complexity also becomes $O(n)$ due to the storage of computed values in the `memo` dictionary.

Now, you might wonder: How many recursive calls does a memoized Fibonacci function actually make?

To figure this out, let's see how memoization changes the usual recursion tree:

- **Base Case**: If the Fibonacci number for $n$ is already stored in our memoization cache, or if $n$ is $0$ or $1$, the function returns directly without any further calls.
- **Memoization Check**:  If $n$ isn't in the cache, the function makes two recursive calls: `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.
- **The Memoization Effect**: The very first time we call `fibonacci_memo(n, memo)` with a new value of $n$, it will keep making recursive calls until it hits the base cases. Crucially, each Fibonacci number, once computed, is stored in the cache, it gets stored in the cache. Subsequent calls with the same value of n retrieve the stored result, circumventing further recursive calls.

#### Calculating the Number of Recursive Calls

To understand the efficiency of our memoized Fibonacci function, we need to calculate the number of recursive calls made during its execution. Memoization significantly reduces the number of redundant calls, resulting in a more efficient algorithm. Below, we break down the process:

- **Initial Call**: We start the whole process with a single call to `fibonacci_memo(n, memo)`.
- **Recursive Expansion**: For every new $n$ value we encounter, the function branches out into calls for `fibonacci_memo(n-1, memo)` and `fibonacci_memo(n-2, memo)`.
- **Memoization Storage**: Each calculated value is stored, hence any future calls with the same $n$ don't create new branches.
- **Counting Unique Calls**: Because of memoization, we only need to calculate each Fibonacci number once. This means the total number of recursive calls is roughly equal to the number of unique Fibonacci numbers up to $n$.

In conclusion, while a naive Fibonacci implementation would have an exponential number of calls, memoization brings this down significantly. We still have roughly $2n$ calls to calculate Fibonacci numbers from $0$ to $n$, this time a linear growth, but the key is that each unique number is only calculated once, making the process efficient.

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

#### Time and Space Complexity

We must proceed to the complexity analysis, focusing on the Big O notation, of the fibonacci_memo function, which uses memoization to calculate Fibonacci numbers. Let's analyze its time and space complexity. The key to understanding the time complexity is that each unique value of $n$ is calculated only once and then stored in `memo`.

So, as there are $n$ unique values (from $0$ to $n$), for each value of $n$, the function executes a fixed amount of operations (checking, adding, and retrieving values from `memo`). Therefore, the total time complexity of the function is $O(n)$, since each Fibonacci number up to $n$ is computed and stored once, and only once.

The space complexity is determined by the additional storage used by the memoization dictionary (`memo`). The dictionary `memo` can contain up to $n$ entries (one for each Fibonacci number up to $n$). Consequently, the space complexity is also $O(n)$ due to the storage needs of the `memo` dictionary.

We are now ready to study Dynamic Programming with Tabulation.

### Example 3: Fibonacci with Tabulation

Now, let's explore an example of Dynamic Programming using the tabulation technique:

![]({{ site.baseurl }}/assets/images/interactive-fibbo.jpg)
*Flowchart 3 - Interactive Fibonacci nth algorithm*{: class="legend"}

Here is the function `fibonacci_tabulation` defined to calculate the nth Fibonacci number using tabulation, utilizing Python in a manner similar to pseudocode:

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

*Code Fragment 3 - Tabulation Function for the nth Term*{: class="legend"}

Unlike the previous recursive function, this function uses an iterative approach known as tabulation, a bottom-up Dynamic Programming technique. In the Example 3, the `fibonacci_tabulation()` function uses a list, `dp`, to store the results of all subproblems, building the solution from the bottom up. It is important to note that data is being stored in Example 3.

Indeed! But look closely. The `fibonacci_tabulation()` function is an example of tabulation, not memoization, due to the distinct manner in which subproblems are solved and their solutions stored.

*Tabulation is a bottom-up approach to Dynamic Programming where you solve all subproblems first and store their solutions in a data structure, usually a table, array, list, or tree*. The solution to the larger problem is then built from these smaller solutions by traversing the data structure from the bottom up. *This implies an iterative resolution process*. The subproblems are solved iteratively, starting from the smallest until the larger problem is reached. In this case, recursion is irrelevant.

#### Function Definition and Initialization

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
```

*Code Fragment 3A - Tabulation Function Initialization*{: class="legend"}

- `if n <= 1: return n`: This handles the base cases of the Fibonacci Sequence. If $n$ is $0$ or $1$, it directly returns n because:

  - `fibonacci_tabulation(0)` $= 0$
  - `fibonacci_tabulation(1)` $= 1$

- `dp = [0] * (n + 1)`: This initializes a list dp with `n+1` zeros. This list will store the Fibonacci numbers up to $n$.

- `dp[1] = 1`: This declaration sets the second element of `dp` to $1$, since `fibonacci_tabulation(1)` $= 1$.

#### Iteration and Calculation

```python
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

*Code Fragment 3A - Tabulation Function Iteration*{: class="legend"}

-`for i in range(2, n + 1)`: This loop starts from $2$ and iterates up to $n$.
-`dp[i] = dp[i-1] + dp[i-2]`: This calculates the ith Fibonacci number by summing the previous two Fibonacci numbers (i.e., `fibonacci_tabulation(i-1)` and `fibonacci_tabulation(i-2)`) and stores it in the `dp` list at index $i$.
-`return dp[n]`: After the loop completes, the function returns the nth Fibonacci number stored in `dp[n]`.

#### Flow Explanation

Let's try with the tenth Fibonacci number. When `fibonacci_tabulation(10)` is called, it checks if $10 <= 1$. It is not, so it proceeds.

Initializes the `dp` list with zeros: `dp = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.
Iterates from $2$ to $10$:

- `For i = 2: dp[2] = dp[1] + dp[0]` $= 1 + 0 = 1$
- `For i = 3: dp[3] = dp[2] + dp[1]` $= 1 + 1 = 2$
- `For i = 4: dp[4] = dp[3] + dp[2]` $= 2 + 1 = 3$
- This continues until $i = 10$.

After the loop, `dp` is `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]`.
Returns `dp[10]`, which is $55$. And this is the moment when we stop to celebrate.

#### Time and Space Complexity

The time complexity analysis of the function `fibonacci_tabulation()` begins with the initial check to see if `n` is less than or equal to $1$, which is a constant time operation, $O(1)$. If `n` is greater than $1$, the function initializes a list `dp` with `n + 1` elements, which takes $O(n)$ time. After this, the function sets `dp[1]` to $1$, another constant time operation, $O(1)$.However, the main iterative computation part is missing from this snippet.

Assuming a complete implementation that iterates from $2$ to `n` to fill in the `dp` array, the total time complexity would be $O(n)$, as each Fibonacci number up to `n` is computed and stored exactly once.

The space complexity is dominated by the list `dp` with `n + 1` elements, which requires $O(n)$ space. Therefore, with the complete implementation in mind, the function achieves an efficient computation with both time and space complexities of $O(n)$. While the complexity analysis covers the basic aspects of the function's efficiency, there are additional considerations and potential optimizations that could further enhance its performance, inviting deeper exploration.

### There is more between heaven and earth, Mr. Shakespeare

Memoization and tabulation are the most common techniques in dynamic programming; however, they are not the only techniques.

- **Dynamic Programming with State Compression**: The goal is to reduce the space needed to store the results of the subproblems by keeping only the states relevant to calculating the final solution.
- **Dynamic Programming with Sliding Window**: Maintains only the results of the most recent subproblems in a fixed-size window, useful when the solution depends only on a limited number of previous subproblems.
- **Dynamic Programming with Decision Tree**: Represents the subproblems and their relationships in a decision tree, allowing a clear visualization of the problem structure and the decisions to be made.

Let's see how far we get in this text. As I write this, I still have no idea.

## Now I realize: C++, where is C++?

Python, which I used as pseudocode, is a versatile and simple language. BBesides that, most of my students are accustomed to Python. However, it is still not the most suitable language for high-performance use or programming competitions. Therefore, we will move to C++ 20 and, eventually, use data structures compatible with C 17, in the C++ 20 environment. Speaking of the environment, from this point on, I will be using Visual Studio Community edition to run and evaluate all the code. Nevertheless, we cannot throw away all the work we have done so far. To maintain consistency in our text, I will convert the same functions we wrote in Python to C++ and assess the results.

### Example 4: Fibonacci in C++ using `std::vectors`

Let's begin with a straightforward, intuitive implementation in C++20, following the flow and data structures of the Python functions provided earlier.

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

*Example 4 - running std::vector and tail recursion*{: class="legend"}

Now, the attentive reader will agree with me: we must to break this code down.

#### The Recursive Function

Let's start with `fibonacci(int n)`, the simple and pure recursive function.

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

*Code Fragment 4 - C++ Dynamic Programing, Tail Recursion Function*{: class="legend"}

This is a similar C++ recursive function to the one we used to explain recursion in Python. Perhaps the most relevant aspect of `fibonacci(int n)` is its argument: `int n`. Using the `int` type limits our Fibonacci number to $46$. Especially because the `int` type on my system, a 64-bit computer running Windows 11, is limited, by default, to storing a maximum value of $2^31 - 1 = 2,147,483,647$, and the $46$th Fibonacci number is $1,836,311,903$. The next one will be bigger than `int` capacity. Since Python uses floating-point numbers (like doubles) by default, we can calculate up to the $78$th Fibonacci number, which is $8,944,394,323,791,464$. We could achieve the same result in C++ by changing from `int` to another data type. However, that is not our goal here.

The next function is the C++ memoization version:

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

*Code Fragment 5 - C++ Memoization Function*{: class="legend"}

Let's highlight the `std::unordered_map<int, int>& memo` in function arguments. The argument `std::unordered_map<int, int>& memo` in C++ is used to pass a reference to an unordered map (hash table) that maps integers to integers. Breaking it down we will have:

The `std::unordered_map<int, int>` specifies the type of the argument. `std::unordered_map` is a template class provided by the C++ Standard Library that implements a hash table. The template parameters `<int, int>` specify that the keys and values stored in the unordered map are both integers.

The ampersand (`&`) indicates that the argument is a reference. This means that the function will receive a reference to the original unordered map, rather than a copy of it. *Passing by reference is efficient because it avoids copying the entire map, which could be expensive in terms of time and memory, especially for large maps*. Pay attention: Thanks to the use of `&`, all changes made to the map inside the function will affect the original map outside the function. Finally `memo` is the identifier of the parameter which type is `std::unordered_map<int, int>`.In the context of memoization (hence the name `memo` we used earlier), this unordered map is used to store the results of previously computed values to avoid redundant calculations.

One `unordered_map` in C++ is quite similar to Python's dict in terms of functionality. Both provide an associative container that allows for efficient key-value pair storage and lookup. The `std::unordered_map` is a template class and a C++ only construct implemented as a hash table. *Unordered maps store key-value pairs and provide average constant-time complexity for insertion, deletion, and lookup operations, thanks to their underlying hash table structure*. They grow dynamically as needed, managing their own memory, which is freed upon destruction. Unordered maps can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements.

Unlike arrays, unordered maps do not decay to pointers, and you cannot get a pointer to their internal data. Instead, unordered maps maintain an internal hash table, which is allocated dynamically by the allocator specified in the template parameter, usually obtaining memory from the freestore (heap) independently of the object's actual allocation. This makes unordered maps efficient for fast access and manipulation of key-value pairs, though they do not maintain any particular order of the elements.

Unordered maps do not require a default constructor for stored objects and are well integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, unordered maps rehash their elements, which involves reassigning the elements to new buckets based on their hash values. This rehashing process can involve copying or moving (in C++11 and later) the elements to new locations in memory.

>Rehashing, is the process used in `std::unordered_map` to maintain efficient performance by redistributing elements across a larger array when the load factor (the number of elements divided by the number of buckets) becomes too high. The rehashing process involves determining the new size, allocating a new array of buckets to hold the redistributed elements, rehashing elements by applying a hash function to each key to compute a new bucket index and inserting the elements into this new index, and finally, updating the internal state by updating internal pointers, references, and variables, and deallocating the old bucket array. Rehashing in `std::unordered_map` is crucial for maintaining efficient performance by managing the load factor and ensuring that hash collisions remain manageable.

Overall, `std::unordered_map` is a versatile and efficient container for associative data storage, offering quick access and modification capabilities while seamlessly integrating with the C++ Standard Library and, for our purposes, very similar to Python's dictionary

The `fibonacci_memo(int n, std::unordered_map<int, int>& memo)` function works just like the Python function we explained before with the same complexity, $O(n)$, for space and time. That said we can continue to `fibonacci_tabulation(int n)`.

#### The Dynamic Programming Function Using Memoization

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

*Code Fragment 6 - C++ Tabulation Function*{: class="legend"}

The `std::vector` is a template class and a C++-only construct implemented as a dynamic array. *Vectors grow and shrink dynamically, automatically managing their memory, which is freed upon destruction. They can be passed to or returned from functions by value and can be copied or assigned, performing a deep copy of all stored elements*.

Unlike arrays, vectors do not decay to pointers, but you can explicitly get a pointer to their data using `&vec[0]`. Vectors maintain their size (number of elements currently stored) and capacity (number of elements that can be stored in the currently allocated block) along with the internal dynamic array. This internal array is allocated dynamically by the allocator specified in the template parameter, *usually obtaining memory from the freestore (heap) independently of the object's actual allocation*. Although this can make vectors less efficient than regular arrays for small, short-lived, local arrays, vectors do not require a default constructor for stored objects and are better integrated with the rest of the STL, providing `begin()`/`end()` methods and the usual STL typedefs. When reallocating, vectors copy (or move, in C++11) their objects.

Besides the `std::vector` template type, the time and space complexity are the same, $O(n)$, we found in Python version. What left us with the generic part of Example 4. Evaluation.

#### Performance Evaluation and Support Functions

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

*Code Fragment 7 - Support Functions for Time Execution Measurement*{: class="legend"}

Let's initiate with Function 1, `long long average_time(Func func, int iterations, Args&&... args)`. This function is a template function designed to measure the execution time of a given function `func` with arbitrary arguments `Args&&... args`. It returns the time taken to execute the function in nanoseconds. Let's break down each part of this function to understand how it works in detail.

```cpp
template <typename Func, typename... Args>
```

The keyword `template` in `measure_time` declaration indicates that `measure_time` is a template function, which means it can operate with generic types.

>A template is a C++ language feature that allows functions and classes to operate with generic types, enabling code reuse and type safety, allowing the creation of functions and classes that can work with any data type without being rewritten for each specific type. This is achieved by defining a blueprint that specifies how the function or class should operate with type parameters that are provided when the template is instantiated. The advantage of templates is their ability to provide high levels of abstraction while maintaining performance, as template code is resolved at compile time, resulting in optimized and type-safe code. This leads to more flexible and reusable code structures, reducing redundancy and the potential for errors, and allowing developers to write more generic and maintainable code.

The first argument, `typename Func` specifies that the first template parameter, `Func`, can be any callable type, such as functions, function pointers, lambdas, or functors. When `typename Func` is specified in a template definition, it indicates that the template will accept a callable entity as a parameter. The use of `typename` in this context ensures that the template parameter is interpreted as a type, enabling the compiler to correctly process the callable type during instantiation. I am using `Func` to call the function whose execution time will be measured.

The last argument, `typename... Args`: This is a variadic template parameter, allowing the function to accept any number of additional arguments of any types.

>The `typename... Args` declaration is used in C++ templates to define a variadic template parameter, which allows a template to accept an arbitrary number of arguments of any types. When `typename... Args` is specified, it indicates that the template can handle a variable number of parameters, making it highly flexible and adaptable. This is particularly useful for functions and classes that need to operate on a diverse set of inputs without knowing their types or number in advance.

```Cpp
long long measure_time(Func func, Args&&... args) {
```

In the context of a template function, `Args&&... args` is often used to perfectly forward these arguments to another function, preserving their value categories (`lvalues` or `rvalues`). The use of `typename...` ensures that each parameter in the pack is treated as a type, enabling the compiler to correctly process each argument during template instantiation.

>An `lvalue` (locator value) represents an object that occupies a specific location in memory (i.e., has an identifiable address). `lvalues` are typically variables or dereferenced pointers. They can appear on the left-hand side of an assignment expression, hence the name `lvalue`.
>An `rvalue` (read value) represents a temporary value or an object that does not have a persistent memory location. rvalues are typically literals, temporary objects, or the result of expressions that do not persist. They can appear on the right-hand side of an assignment expression.
>C++11 introduced `rvalue` references to enhance performance by enabling move semantics. An rvalue reference is declared using `&&`, allowing functions to distinguish between copying and moving resources. This is particularly useful for optimizing the performance of classes that manage resources such as dynamic memory or file handles.

The return type of the function, `long long`, represents the duration of the function execution in nanoseconds. I choose a `long long` integer because I have no idea how long our Dynamic Programming functions will take to compute, and I wanted to ensure a default function that can be used for all problems we will work on. The maximum value that can be stored in a `long long` type in C++ is defined by the limits of the type, which are specified in the `<climits>` header. For a signed `long long` type, the maximum value is $2^{63} - 1 = 9,223,372,036,854,775,807$.

The function `measure_time` arguments are:

- `Func func`: The callable entity whose execution time we want to measure.
- `Args&&... args: A parameter pack representing the arguments to be forwarded to the callable entity. The use of && indicates that these arguments are perfect forwarded, preserving their value category (`lvalue` or `rvalue`).

The the body of function `measure_time` starts with:

```Cpp
auto start = std::chrono::high_resolution_clock::now();
```

Where `auto start` declares a variable `start` to store the starting time point and
`std::chrono::high_resolution_clock::now()` retrieves the current time using a high-resolution clock, which provides the most accurate and precise measurement of time available on the system. The `std::chrono::high_resolution_clock::now()` returns a `time_point` object representing the current point in time.

>In C++, a `time_point` object is a part of the `<chrono>` library, which provides facilities for measuring and representing time. A `time_point` object represents a specific point in time relative to a clock. It is templated on a clock type and a duration, allowing for precise and high-resolution time measurements. The Clock is represented by a  `clock type`, and can be system_clock, steady_clock, or high_resolution_clock. The clock type determines the epoch (the starting point for time measurement) and the tick period (the duration between ticks).

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

>In C++, the `<chrono>` library provides a set of types and functions for dealing with time and durations in a precise and efficient manner. One of the key components of this library is the std::chrono::duration class template, which represents a time duration with a specific period.

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

*Code Fragment 8 - Average Time Function*{: class="legend"}

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

*Code Fragment 9 - C++ `std::vector` main() function*{: class="legend"}

The `main()` function measures and compares the average execution time of different implementations of the Fibonacci function. Here's a detailed explanation of each part:

The program starts by defining the number of iterations (`const int iterations = 1000;`) and a vector of test cases (`std::vector<int> test_cases = { 10, 20, 30 };`). It then iterates over each test case, calculating the Fibonacci number using different methods and measuring their average execution times.

For the memoized Fibonacci implementation, the program first creates an unordered map `memo` to store previously computed Fibonacci values. It then defines a lambda function `fibonacci_memo_wrapper` that captures `memo` by reference and calls the `fibonacci_memo` function. The `average_time` function is used to measure the average execution time of this memoized implementation over 1000 iterations for each test case.

The other functions follow a similar pattern to measure and print their execution times. For instance, in the case of the recursive Fibonacci function, the line `long long avg_time_recursive = average_time(fibonacci, iterations, n);` calls the `average_time` function to measure the average execution time of the simple recursive Fibonacci function over 1000 iterations for the current test case $n$. The result, stored in `avg_time_recursive`, represents the average time in nanoseconds. The subsequent line, `std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";`, outputs this average execution time to the console, providing insight into the performance of the recursive method.

The results are printed to the console, showing the performance gain achieved through memoization compared to the recursive and tabulation methods.

#### Running Example 4 - `std::vector`

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

*Output 1 - running Example 4 - std::vector*{: class="legend"}

the careful reader should note that the execution times vary non-linearly and, in all cases, for this problem, the Dynamic Programming version using tabulation was faster. There is much discussion about the performance of the Vector class compared to the Array class. To test the performance differences between `std::vector` and `std::array`, we will retry using `std::array`

### Example 5: using `std::array`

First and foremost, `std::array` is a container from the C++ Standard Library with some similarities to, and some differences from, `std::vector`, namely:

>The `std::array` is a template class introduced in C++11, which provides a fixed-size array that is more integrated with the STL than traditional C-style arrays. Unlike `std::vector`, `std::array` does not manage its own memory dynamically; its size is fixed at compile-time, which makes it more efficient for cases where the array size is known in advance and does not change. `std::array` objects can be passed to and returned from functions, and they support copy and assignment operations. They provide the same `begin()`/`end()` methods as vectors, allowing for easy iteration and integration with other STL algorithms. One significant advantage of `std::array` over traditional arrays is that it encapsulates the array size within the type itself, eliminating the need for passing size information separately. Additionally, `std::array` provides member functions such as `size()`, which returns the number of elements in the array, enhancing safety and usability. However, since `std::array` has a fixed size, it does not offer the dynamic resizing capabilities of `std::vector`, making it less flexible in scenarios where the array size might need to change.

When considering performance differences between `std::vector` and `std::array`, it's essential to understand their underlying characteristics and use cases. *`std::array` is a fixed-size array, with its size determined at compile-time, making it highly efficient for situations where the array size is known and constant*. The lack of dynamic memory allocation means that `std::array` avoids the overhead associated with heap allocations, resulting in faster access and manipulation times. This fixed-size nature allows the compiler to optimize memory layout and access patterns, often resulting in better cache utilization and reduced latency compared to dynamically allocated structures.

In contrast, *`std::vector` provides a dynamic array that can grow or shrink in size at runtime, offering greater flexibility but at a cost. The dynamic nature of `std::vector` necessitates managing memory allocations and deallocations, which introduces overhead*. When a `std::vector` needs to resize, it typically allocates a new block of memory and copies existing elements to this new block, an operation that can be costly, especially for large vectors. Despite this, `std::vector` employs strategies such as capacity doubling to minimize the frequency of reallocations, balancing flexibility and performance.

*For small, fixed-size arrays, `std::array` usually outperforms `std::vector` due to its minimal overhead and compile-time size determination*. It is particularly advantageous in performance-critical applications where predictable and low-latency access is required. On the other hand, `std::vector` shines in scenarios where the array size is not known in advance or can change, offering a more convenient and safer alternative to manually managing dynamic arrays.

In summary, `std::array` generally offers superior performance for fixed-size arrays due to its lack of dynamic memory management and the resultant compiler optimizations. However, `std::vector` provides essential flexibility and ease of use for dynamically sized arrays, albeit with some performance trade-offs. The choice between `std::array` and `std::vector` should be guided by the specific requirements of the application, weighing the need for fixed-size efficiency against the benefits of dynamic resizing.

| Feature        | `std::vector`                 | `std::array`                    |
| -------------- | ------------------------------ | ------------------------------- |
| Size           | Dynamic (can change at runtime) | Fixed (determined at compile time) |
| Memory Management | Dynamic allocation on the heap     | Typically on the stack, no dynamic allocation |
| Performance     | Can have overhead due to resizing | Generally more efficient for fixed-size data |
| Use Cases       | When the number of elements is unknown or varies | When the number of elements is known and fixed |
| Flexibility     | High (can add/remove elements easily) | Low (size cannot be changed)       |
| STL Integration | Yes (works with algorithms and iterators) | Yes (similar interface to vector) |

*Tabela 1 - std::vector and std::array comparison*{: class="legend"}

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

*Code Fragment 10 - C++, `std::array`, Tabulation Function*{: class="legend"}

This is basically the same code that we discussed in the previous section, only replacing the `std::vector` class with the `std::array` class. Therefore, we do not need to analyze the code line by line and can consider the flowcharts and complexity analysis already performed.

#### Running Example 5: using `std::array`

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

*Output 2 - Example 5 running std::vector*{: class="legend"}

We have reached an interesting point. Just interesting!

We achieved a performance gain using memoization and tabulation, as evidenced by the different complexities among the recursive $O(n^2)$, memoization $O(n)$, and tabulation $O(n)$. Additionally, we observed a slight improvement in execution time by choosing `std::array` instead of `std::vector`. However, we still have some options to explore. Options never end!

### Code 3: C-style Array

We are using a C++ container of integers to store the already calculated Fibonacci numbers as the basis for the two Dynamic Programming processes we are studying so far, memoization and tabulation, one `std::unordered_map` and one `std::vector` or `std::array`. However, there is an even simpler container in C++: the array. The C-Style array.

For compatibility, C++ allows the use of code written in C, including data structures, libraries, and functions. So, why not test these data structures? For this, I wrote new code, keeping the functions using `std::array` and `std::unordered_map` and creating two new dynamic functions using C-style arrays. We will call it The CODE 3.

```Cpp
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <array>
#include <utility>

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

// Iterative function with tabulation to calculate Fibonacci using C-style arrays
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    int dp[41] = { 0 };  // array to support up to Fibonacci(41) biggest in int type
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// structs for C style functions
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

// Function to measure execution time and return the result
template <typename Func, typename... Args>
std::pair<long long, int> measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    int result = func(std::forward<Args>(args)...);  // Get the function result
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return { duration.count(), result };
}

// Function to calculate average execution time and return the last calculated result
template <typename Func, typename... Args>
std::pair<long long, int> average_time(Func func, int iterations, Args&&... args) {
    long long total_time = 0;
    int last_result = 0;
    for (int i = 0; i < iterations; ++i) {
        auto [time, result] = measure_time(func, std::forward<Args>(args)...);
        total_time += time;
        last_result = result;
    }
    return { total_time / iterations, last_result };
}

int main() {
    
    const int iterations = 1000;
    int test_cases[] = { 10, 20, 30};  // C-style array for test cases

    for (int n : test_cases) {
        std::cout << "Calculating Fibonacci(" << n << ")\n";

        // Calculation and average time using the simple recursive function
        auto [avg_time_recursive, result_recursive] = average_time(fibonacci, iterations, n);
        std::cout << "Average time for recursive Fibonacci: " << avg_time_recursive << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_recursive << "\n";

        // Calculation and average time using the memoization function
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        auto [avg_time_memo, result_memo] = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Average time for memoized Fibonacci: " << avg_time_memo << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_memo << "\n";

        // Calculation and average time using the tabulation function
        auto [avg_time_tabulation, result_tabulation] = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Average time for tabulated Fibonacci: " << avg_time_tabulation << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_tabulation << "\n";

        // Calculation and average time using the new memoization function with arrays
        auto [avg_time_novofIbb, result_cArray_fibonacci_memo] = average_time(cArray_fibonacci_memo, iterations, n);
        std::cout << "Average time for new memoized Fibonacci: " << avg_time_novofIbb << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_cArray_fibonacci_memo << "\n";

        // Calculation and average time using the new tabulation function with arrays
        auto [avg_time_novo_tabulation, result_cArray_tabulation] = average_time(cArray_fibonacci_tabulation, iterations, n);
        std::cout << "Average time for new tabulated Fibonacci: " << avg_time_novo_tabulation << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_cArray_tabulation << "\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
```

*Code 3: full code using C-Style array*{: class="legend"}

This code is basically the same except for the following fragment:

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

*Code Fragment 11 - C++, C-Style Array, Memoization and Tabulation Functions*{: class="legend"}

As I said, this code segment introduces two new functions for calculating Fibonacci numbers using C-style arrays, with a particular focus on the function for memoization. Instead of using an `std::unordered_map` to store the results of previously computed Fibonacci numbers, the memoization function `cArray_fibonacci_memo` uses two arrays: `found` and `memo`. The `found` array is a boolean array that tracks whether the Fibonacci number for a specific index has already been calculated, while the `memo` array stores the calculated Fibonacci values. The function checks if the result for the given $n$ is already computed by inspecting the `found` array. If it is, the function returns the value from the `memo` array. If not, it recursively computes the Fibonacci number, stores the result in the `memo` array, and marks the `found` array as true for that index. To be completely honest, this idea of using two arrays comes from [this site]([URL](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html)).

The `cArray_fibonacci_tabulation` function, on the other hand, implements the tabulation method using a single C-Style array `dp` to store the Fibonacci numbers up to the $n$th value. The function initializes the base cases for the Fibonacci Sequence, with `dp[0]` set to $0$ and `dp[1]` set to $1$. It then iterates from $2$ to $n$, filling in the `dp` array by summing the two preceding values. This iterative approach avoids the overhead of recursive calls, making it more efficient for larger values of $n$.

Again succinct! I think I'm learning. These structures have the same space and time complexities that we have observed since Example 4. In other words, all that remains is to run this code and evaluate the execution times.

#### Running Code 3: using C-Style Array

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

*Output 3: running C-Style array*{: class="legend"}

And there it is, we have found a code fast enough for calculating the nth Fibonacci number in an execution time suitable to my ambitions. The only problem is that we used C-Style arrays in a C++ solution. In other words, we gave up all C++ data structures to make the program as fast as possible. We traded a diverse and efficient language for a simple and straightforward one. This choice will be up to the kind reader. You will have to decide if you know enough C to solve any problem or if you need to use predefined data structures to solve your problems. *Unless there is someone in the competition using C. In that case, it's C and that's it*.

Before we start solving problems with dynamic programming, let's summarize the execution time reports in a table for easy visualization and to pique the curiosity of the kind reader.

## Execution Time Comparison Table

| Container  | Number | Recursive (ns) | Memoized (ns) | Tabulated (ns) |
|-----------------|------------------|--------------------------|-------------------------|--------------------------|
| **Vectors**     | 10               | 660                      | 607                     | 910                      |
|                 | 20               | 75,712                   | 444                     | 1,300                    |
|                 | 30               | 8,603,451                | 414                     | 1,189                    |
| **Arrays**      | 10               | 807                      | 426                     | 159                      |
|                 | 20               | 88,721                   | 434                     | 371                      |
|                 | 30               | 10,059,626               | 414                     | 439                      |
| **C-Style Arrays** | 10            | 718                      | 29                      | 72                       |
|                 | 20               | 71,414                   | 28                      | 87                       |
|                 | 30               | 8,765,969                | 29                      | 115                      |

*Tabela 2 - Code Execution Time Comparison*{: class="legend"}

With sufficient practice, Dynamic Programming concepts will become intuitive. I know, the text is dense and complicated. I purposefully mixed concepts of Dynamic Programming, complexity analysis, C++, and performance. If the kind reader is feeling hopeless, stand up, have a soda, walk a bit, and start again. Like everything worthwhile in life, Dynamic Programming requires patience, effort, and time. If, on the other hand, you feel confident, let's move on to our first problem.

## Your First Dynamic Programming Problem

Dynamic programming concepts became popular in the early 21st century thanks to job interviews for large companies. Until then, only high-performance and competitive programmers were concerned with these techniques. Today, among others, we have [LeetCode](https://leetcode.com/) with hundreds, perhaps thousands of problems to solve. I strongly recommend trying to solve some of them. Here, I will only solve problems whose solutions are already available on other sites. You might even come across some from LeetCode problem, but that will be by accident. The only utility of LeetCode, for me, for you, and for them, is that the problems are not easy to find or solve. Let's start with a problem that is now a classic on the internet and, according to legend, was part of a Google interview.

### Problem 1: The "Two Sum" problem

**Statement**: In a technical interview, you've been given an array of numbers, and you need to find a pair of numbers that sum up to a given target value. The numbers can be positive, negative, or both. Can you design an algorithm that works in $O(n)$ time complexity or better?

For example, given the array: `[8, 10, 2, 9, 7, 5]` and the target sum: 11

Your function should return a pair of numbers that add up to the target sum. Your answer must be a function in form: `Values(sequence, targetSum)`, In this case, your function should return (9, 2).

This will continue!!!