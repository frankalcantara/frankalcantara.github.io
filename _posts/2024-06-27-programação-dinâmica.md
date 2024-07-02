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
image: assets/images/deriva.jpeg
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

Some say that dynamic programming is a technique to make recursive code more efficient. There is a relationship that needs to be explored: *all dynamic programming algorithms are recursive, but not all recursive algorithms are dynamic programming*.

Recursion is a powerful problem-solving technique. Recursive code can be mathematically proven correct relatively easily. And that alone is reason enough to use recursion in all your code. So, let's begin with that.

The proof of the correctness of a recursive algorithm generally involves only two steps: proving that the base case of the recursion is correct and proving that the recursive step is correct. In the domain of mathematical induction proof, we can refer to these components as the *base case* and the *inductive step*, respectively. In this case:

- To prove the **base case**, we check the simplest # case of the recursion, usually the base case or cases, to ensure it is correct. These are the cases that do not depend on recursive calls.

- To prove the **inductive step**, we verify that if the recursive function is correct for all smaller cases or subproblems, then it is also correct for the general case. In other words, we assume that the function is correct for smaller inputs, or for a smaller set of inputs, the induction hypothesis, and based on this, we prove, or not, that the recursive function is correct.

Beyond the ease of mathematical proof, recursive code stands out for being clear and intuitive, especially for problems with repetitive structures such as tree traversal, maze solving, and calculating mathematical series.

Many problems are naturally defined in a recursive manner. For example, the mathematical definition of the Fibonacci sequence or the structure of binary trees are inherently recursive. In these cases, the recursive solution will be simpler, more straightforward, and likely more efficient.

Often, the recursive solution is more concise and requires fewer lines of code compared to the iterative solution. Fewer lines, fewer errors, easier to read and understand. Sounds good.

Finally, recursion is an almost ideal approach for applying divide-and-conquer techniques. Since Julius Caesar, we know it is easier to divide and conquer. In this case, a problem is divided into subproblems, solved individually, and then combined to form the final solution. Classic academic examples of these techniques include sorting algorithms like quicksort and mergesort.

The sweet reader might have raised her eyebrows. This is where recursion and dynamic programming touch, not subtly and delicately, like a lover's caress on the beloved's face. But with the decisiveness and impact of Mike Tyson's glove on the opponent's chin. The division of the main problem into subproblems is the fundamental essence of both recursion and dynamic programming.

Dynamic programming and recursion are related; *both involve solving problems by breaking a problem into smaller problems. However, while recursion solves the smaller problems without considering the computational cost of repeated calls, dynamic programming optimizes these solutions by storing and reusing previously obtained results*. The most typical example of recursion is determining the nth order value of the Fibonacci sequence can be seen in Flowchart 1.

![]({{ site.baseurl }}/assets/images/fibbo_recursivo_flow.png)
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
- **Recursive Step**: If $n$ is greater than 1, the function calls itself twice: once with $n - 1$ and once with $n - 2$. The sum of these two results is returned.

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

- **Memoization (Top-Down)**: stores the results of recursive calls in a data structure (such as a dictionary or a list, etc.) for reuse. The name memoization is a horrible borrowing from the English word *memoization*.

- **Tabulation (Bottom-Up)**: solves the problem iteratively, filling a table (usually a list or matrix) with the results of the subproblems.

In this case, we can see two examples in Python. First, an example of Dynamic Programming with memoization:

### Example 2: memoization

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

In this example, `fibonacci_memo` stores the results of previous calls in the `memo` dictionary, avoiding repeated calculations. From the perspective of dynamic programming, this function divides the larger problem (calculating Fibonacci of $n$) into smaller subproblems (calculating Fibonacci of $n-1$ and $n-2$), uses a data structure, the `memo` dictionary, to store the results of the subproblems. This avoids redundant calculations of the same values, and before calculating the Fibonacci value for a given $n$, the function checks if the result is already stored in the `memo` dictionary. If it is, it reuses that result, saving computation time. Finally, the function ensures that each subproblem is solved only once, resulting in more efficiency compared to the simple recursive approach.

The last statement of the previous paragraph requires reflection. I am considering performance in this statement only in terms of computation time. Performance can also be considered in relation to memory usage, energy consumption, and any other factor that is interesting or important for a given problem. Keep this in mind whenever I state that performance has improved in this text. Well, who is thinking about a example?

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

Python is a relatively simple and very popular programming language. However, it is not yet the most suitable programming language when we talk about performance. So, yes. I started with Python, almost as if I were using pseudocode, just to highlight the concepts. From this point on, we will use C++. I will run all the code presented here on a Windows 11 machine, using Visual Studio Community Edition, configured for C++ 20. Just to maintain some coherence, let's revisit the functions we have already seen.


```Cpp
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <windows.h>  // Necessário para definir a página de código do console

// Função recursiva para calcular o Fibonacci
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

// Função recursiva com memoização para calcular o Fibonacci
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

// Função iterativa com tabulação para calcular o Fibonacci usando arrays de estilo C
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    int dp[41] = { 0 };  // array para suportar até Fibonacci(40)
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Função para medir o tempo de execução e retornar o resultado
template <typename Func, typename... Args>
std::pair<long long, int> measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    int result = func(std::forward<Args>(args)...);  // Obtenha o resultado da função
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return { duration.count(), result };
}

// Função para calcular o tempo médio de execução e retornar o último resultado calculado
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
    // Define a página de código do console para UTF-8
    SetConsoleOutputCP(CP_UTF8);

    const int iterations = 1000;
    int test_cases[] = { 10, 20, 30};  // array de estilo C para os casos de teste

    for (int n : test_cases) {
        std::cout << "Calculando Fibonacci(" << n << ")\n";

        // Cálculo e tempo médio usando a função recursiva simples
        auto [avg_time_recursive, result_recursive] = average_time(fibonacci, iterations, n);
        std::cout << "Tempo médio Fibonacci recursivo: " << avg_time_recursive << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_recursive << "\n";

        // Cálculo e tempo médio usando a função com memoização
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        auto [avg_time_memo, result_memo] = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Tempo médio Fibonacci com memoização: " << avg_time_memo << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_memo << "\n";

        // Cálculo e tempo médio usando a função com tabulação
        auto [avg_time_tabulation, result_tabulation] = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Tempo médio Fibonacci com tabulação: " << avg_time_tabulation << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_tabulation << "\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}

```

Este código, inocente e instintivo, gera um número de Fibonacci, armazena este número em um tipo inteiro (`int`) depois, para testes, encontra $3$ números de Fibonacci, o décimo, o vigésimo, e o trigésimo, $1000$ vezes seguidas para cada um, calcula o tempo médio para gerar cada um destes números usando as três funções que vimos em Python convertidas para seu equivalente em C++. Com um único cuidado. Eu usei para armazenar a estrutura de dados `Array` no estilo do C em busca de um pouto de velocidade. Ao rodar este código temos a seguinte saída:

```shell
Calculando Fibonacci(10)
Tempo médio Fibonacci recursivo: 1058 ns
Fibonacci(10) = 55
Tempo médio Fibonacci com memoização: 720 ns
Fibonacci(10) = 55
Tempo médio Fibonacci com tabulação: 67 ns
Fibonacci(10) = 55
-----------------------------------
Calculando Fibonacci(20)
Tempo médio Fibonacci recursivo: 86602 ns
Fibonacci(20) = 6765
Tempo médio Fibonacci com memoização: 728 ns
Fibonacci(20) = 6765
Tempo médio Fibonacci com tabulação: 187 ns
Fibonacci(20) = 6765
-----------------------------------
Calculando Fibonacci(30)
Tempo médio Fibonacci recursivo: 9265282 ns
Fibonacci(30) = 832040
Tempo médio Fibonacci com memoização: 541 ns
Fibonacci(30) = 832040
Tempo médio Fibonacci com tabulação: 116 ns
Fibonacci(30) = 832040
-----------------------------------
```

A amável leitora deve observar que os tempos variam de forma não linear e que, em todos os casos, para este problema a versão da programação dinâmica usando tabulação foi mais rápida. Mas, na verdade, dá para fazer ainda mais rápido, se tirarmos o `std::unordered_map` que usamos na função de memoização. Como no código a seguir:

```Cpp
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <array>
#include <utility>
#include <windows.h>  // Necessário para definir a página de código do console

// Função recursiva para calcular o Fibonacci
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

// Função recursiva com memoização para calcular o Fibonacci
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

// Função iterativa com tabulação para calcular o Fibonacci usando arrays de estilo C
int fibonacci_tabulation(int n) {
    if (n <= 1) {
        return n;
    }
    int dp[41] = { 0 };  // array para suportar até Fibonacci(40)
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Nova função com memoização utilizando arrays
const int MAXN = 46; //o maior número de Fibonacci que cabe em um int é o 47.
bool found[MAXN] = { false };
int memo[MAXN] = { 0 };

int novoFIbb(int n) {
    if (found[n]) return memo[n];
    if (n == 0) return 0;
    if (n == 1) return 1;

    found[n] = true;
    return memo[n] = novofibb(n - 1) + novofibb(n - 2);
}

// Função para medir o tempo de execução e retornar o resultado
template <typename Func, typename... Args>
std::pair<long long, int> measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    int result = func(std::forward<Args>(args)...);  // Obtenha o resultado da função
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::nano> duration = end - start;
    return { duration.count(), result };
}

// Função para calcular o tempo médio de execução e retornar o último resultado calculado
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
    // Define a página de código do console para UTF-8
    SetConsoleOutputCP(CP_UTF8);

    const int iterations = 1000;
    int test_cases[] = { 10, 20, 30};  // array de estilo C para os casos de teste

    for (int n : test_cases) {
        std::cout << "Calculando Fibonacci(" << n << ")\n";

        // Cálculo e tempo médio usando a função recursiva simples
        auto [avg_time_recursive, result_recursive] = average_time(fibonacci, iterations, n);
        std::cout << "Tempo médio Fibonacci recursivo: " << avg_time_recursive << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_recursive << "\n";

        // Cálculo e tempo médio usando a função com memoização
        std::unordered_map<int, int> memo;
        auto fibonacci_memo_wrapper = [&memo](int n) { return fibonacci_memo(n, memo); };
        auto [avg_time_memo, result_memo] = average_time(fibonacci_memo_wrapper, iterations, n);
        std::cout << "Tempo médio Fibonacci com memoização: " << avg_time_memo << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_memo << "\n";

        // Cálculo e tempo médio usando a função com tabulação
        auto [avg_time_tabulation, result_tabulation] = average_time(fibonacci_tabulation, iterations, n);
        std::cout << "Tempo médio Fibonacci com tabulação: " << avg_time_tabulation << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_tabulation << "\n";

        // Cálculo e tempo médio usando a nova função com memoização e arrays
        auto [avg_time_novofIbb, result_novofIbb] = average_time(novoFIbb, iterations, n);
        std::cout << "Tempo médio Fibonacci com nova memoização: " << avg_time_novofIbb << " ns\n";
        std::cout << "Fibonacci(" << n << ") = " << result_novofIbb << "\n";

        std::cout << "-----------------------------------\n";
    }

    return 0;
}
``` 

Que ao ser executado gera a seguinte resposta:

```shell
Calculando Fibonacci(10)
Tempo médio Fibonacci recursivo: 822 ns
Fibonacci(10) = 55
Tempo médio Fibonacci com memoização: 512 ns
Fibonacci(10) = 55
Tempo médio Fibonacci com tabulação: 82 ns
Fibonacci(10) = 55
Tempo médio Fibonacci com nova memoização: 50 ns
Fibonacci(10) = 55
-----------------------------------
Calculando Fibonacci(20)
Tempo médio Fibonacci recursivo: 96510 ns
Fibonacci(20) = 6765
Tempo médio Fibonacci com memoização: 457 ns
Fibonacci(20) = 6765
Tempo médio Fibonacci com tabulação: 93 ns
Fibonacci(20) = 6765
Tempo médio Fibonacci com nova memoização: 38 ns
Fibonacci(20) = 6765
-----------------------------------
Calculando Fibonacci(30)
Tempo médio Fibonacci recursivo: 9236120 ns
Fibonacci(30) = 832040
Tempo médio Fibonacci com memoização: 510 ns
Fibonacci(30) = 832040
Tempo médio Fibonacci com tabulação: 142 ns
Fibonacci(30) = 832040
Tempo médio Fibonacci com nova memoização: 43 ns
Fibonacci(30) = 832040
-----------------------------------

```

Agora chegamos no Bom lugar! O cálculo dos números de Fibonacci, com memoização é, no pior caso, mais ou menos 215.000 vezes mais rápido que o versão recursiva que usamos tão frequentemente nos cursos de Ciência da Computação. Você pode encontrar o código original desta função. `novoFibb` no site {*Introduction to Dynamic Programming*](<https://cp-algorithms.com/dynamic_programming/intro-to-dp.html>). Creio que poderia melhorar um pouco mais a função usando tabulação mas, acho que a leitora já entendeu a ideia.
