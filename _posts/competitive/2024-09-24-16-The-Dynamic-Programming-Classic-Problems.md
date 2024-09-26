---
author: Frank
beforetoc: |-
  [Anterior](2024-09-24-15-Dynamic-Programming.md)
  [Próximo](2024-09-24-17-5.-Graphs-and-Graph-Theory.md)
categories:
  - Matemática
  - Linguagens Formais
  - Programação
description: Dynamic Programming in C++ with practical examples, performance analysis, and detailed explanations to optimize your coding skills and algorithm efficiency.
draft: null
featured: false
image: assets/images/prog_dynamic.jpeg
keywords:
  - Developer Tips
lastmod: 2024-09-25T23:33:05.735Z
layout: post
preview: In this comprehensive guide, we delve into the world of Dynamic Programming with C++. Learn the core principles of Competitive Programming, explore various algorithmic examples, and understand performance differences through detailed code comparisons. Perfect for developers looking to optimize their coding skills and boost algorithm efficiency.
published: false
rating: 5
slug: competitive-programming-techniques-insights
tags:
  - Matemática
  - Linguagens Formais
title: The Dynamic Programming Classic Problems
toc: true
---

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
