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

    - To prove the **base case**, we check the simplest case of the recursion, usually the base case or cases, to ensure it is correct. These are the cases that do not depend on recursive calls.

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

### Example 1: Let's calculate $\text{fibonacci}(5)$:

1. $\text{fibonacci}(5)$ calls $\text{fibonacci}(4)$ and $\text{fibonacci}(3)$
2. $\text{fibonacci}(4)$ calls $\text{fibonacci}(3)$ and $\text{fibonacci}(2)$
3. $\text{fibonacci}(3)$ calls $\text{fibonacci}(2)$ and $\text{fibonacci}(1)$
4. $\text{fibonacci}(2)$ calls $\text{fibonacci}(1)$ and $\text{fibonacci}(0)$
5. $\text{fibonacci}(1)$ returns 1
6. $\text{fibonacci}(0)$ returns 0
7. $\text{fibonacci}(2)$ returns $1 + 0 = 1$
8. $\text{fibonacci}(3)$ returns $1 + 1 = 2$
9. $\text{fibonacci}(2)$ returns 1 (recalculated)
10. $\text{fibonacci}(4)$ returns $2 + 1 = 3$
11. $\text{fibonacci}(3)$ returns 2 (recalculated)
12. $\text{fibonacci}(5)$ returns $3 + 2 = 5$

Thus, $\text{fibonacci}(5)$ returns 5. The function breaks down recursively until it reaches the base cases, then combines the results to produce the final value.

And we can write it in Python as:

```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

Neste exemplo, a função `fibonacci` chama a si mesma para calcular os termos anteriores da Sequência de Fibonacci. Observe que para cada valor desejado, temos que passar por todos os outros. Este é um exemplo de recursão correto e inocente e, neste caso específico, muito eficiente. Veremos esta coisa da eficiência com mais cuidado logo a frente.

## Voltando a programação dinâmica

Se olharmos a programação dinâmica veremos uma técnica de otimização que se baseia na recursividade, mas adiciona armazenamento de resultados intermediários para evitar cálculos redundantes. Existem duas abordagens principais para implementar a programação dinâmica:

***memoização* (Top-Down)**: armazena os resultados das chamadas recursivas em uma estrutura de dados (como um dicionário, ou uma lista, etc.) para reutilização. O nome memoização é um estrangeirismo horrível da palavra *memoization* do inglês.

**Tabulação (Bottom-Up)**: resolve o problema de forma iterativa, preenchendo uma tabela (geralmente uma lista ou matriz) com os resultados dos subproblemas.

Neste caso, podemos ver dois exemplos em Python. Primeiro, um exemplo de Programação Dinâmica com *memoização*:

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

Neste exemplo, `fibonacci_memo` armazena os resultados das chamadas anteriores no dicionário `memo`, evitando cálculos repetidos. Do ponto de vista da programação dinâmica, esta função divide o problema maior (calcular Fibonacci de $n$) em subproblemas menores (calcular Fibonacci de $n-1$ e $n-2$), usa uma estrutura de dados, dicionário `memo`, para armazenar os resultados dos subproblemas. Isso evita o cálculo redundante dos mesmos valores e antes de calcular o valor de Fibonacci para um dado $n$, a função verifica se o resultado já está armazenado no dicionário `memo`. Se estiver, ela reutiliza esse resultado, economizando tempo de computação. Finalmente a função garante que cada subproblema é resolvido uma única vez, resultando em mais eficiência quando comparamos com a abordagem recursiva simples.

A última afirmação do parágrafo anterior requer reflexão. Eu estou considerando performance, nesta afirmação, apenas no que diz respeito a tempo de computação. Performance pode ser considerada também em relação ao uso de memória, ao consumo de energia e a qualquer outro fator que seja interessante, ou importante, para um determinado problema. Lembre-se disso, sempre que, neste texto eu afirmar que a performance melhorou.

Finalmente podemos ter um exemplo de Programação Dinâmica com Tabulação:

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

Neste exemplo, a função  `fibonacci_tabulation` usa uma lista dp para armazenar os resultados de todos os subproblemas, construindo a solução de baixo para cima.

## Mas, no último exemplo dados estão sendo armazenados

Isso é verdade! Mas olhe bem. A função `fibonacci_tabulation` é um exemplo de tabulação, e não de *memoização*, devido às características específicas de como os subproblemas são resolvidos e armazenados.

A tabulação é uma abordagem *bottom-up* de programação dinâmica onde você resolve todos os subproblemas  primeiro e armazena suas soluções em uma estrutura de dados, geralmente uma tabela, *array*, lista ou árvore. A solução do problema maior é então construída a partir dessas soluções menores varrendo a estrutura de dados de baixo para cima. Isto implica em um processo de resolução iterativo. Os subproblemas são resolvidos iterativamente, começando dos menores até alcançar o problema maior. E, neste caso, a recursão é irrelevante.

## Há mais entre o céu e a terra

Memoização e Tabulação, são as técnicas mais comuns, mas não são as únicas técnicas de programação dinâmica:

- **Programação Dinâmica com Compressão de Estado**: o objetivo é reduzir o espaço necessário para armazenar os resultados dos subproblemas, mantendo apenas os estados relevantes para o cálculo da solução final.
- **Programação Dinâmica com Janela Deslizante**: mantém apenas os resultados dos subproblemas mais recentes em uma janela de tamanho fixo, útil quando a solução depende apenas de um número limitado de subproblemas anteriores.
- **Programação Dinâmica com Árvore de Decisão**: Representa os subproblemas e suas relações em uma árvore de decisão, permitindo uma visualização clara da estrutura do problema e das decisões a serem tomadas.

Vamos ver até onde vamos chegar neste texto. Neste momento em que escrevo, ainda não tenho ideia.

## Agora que me dei conta; Python

O Python, é uma linguagem de programação relativamente simples e muito popular. Contudo, não é, ainda, a linguagem de programação mais adequada quando estamos falando de performance. Então, sim. Eu comecei com Python, quase como se estivesse usando pseudocódigo, apenas para destacar os conceitos. Deste ponto em diante vamos de C++. Eu vou rodar todos os códigos que apresentar aqui em uma máquina Windows 11, usando o Visual Studio Community Edition, configurado para o C++ 20. Só para manter um tanto de coerência, vamos voltar as funções que já vimos.

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
