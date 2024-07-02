---
layout: post
title: Programação Dinâmica
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
image: assets/images/deriva.jpeg
description: Introdução as técnicas de programação dinâmica com os exemplos mais comuns.
slug: dynamic-programming
keywords:
    - programming
    - programação
    - Dynamic
    - Dinâmica
    - Algorithms
    - Algoritmos
rating: 5
published: 2024-06-27T19:43:15.124Z
draft: null
preview: O que é programação dinâmica com exemplos de algoritmos desenvolvidos em C++
---

A programação dinâmica é uma forma diferente de pensar na hora de resolver problemas. Programar, por si só, já é uma forma diferente de pensar, então, tentando ser honesto, posso dizer que a programação dinâmica é uma forma diferente dentro de uma forma diferente de pensar. E, se não percebeu ainda, há um conceito de recursão tentando emergir nesta definição.

A ideia geral é que você, paciente leitora, seja capaz de dividir um problema grande e difícil em pedaços pequenos e fáceis. Para tanto, armazenaremos informações e reutilizaremos estas informações sempre que necessário no nosso algoritmo.

É muito provável que a amável leitora ao estudar algoritmos tenha sido apresentada a técnicas de programação dinâmica sem se dar conta. Então, também é muito provável que encontre algoritmos que já viu sem saber que se tratava de programação dinâmica.

Minha pretensão será decompor o processo de programação dinâmica em etapas claras, focando no algoritmo de solução, de modo que você possa entender e implementar essas etapas por conta e risco sempre enfrentar um problema em entrevistas técnicas, ambientes de produção ou competições de programação.

## Havia um quê de recursão se intrometendo

Há quem diga que *a programação dinâmica é uma técnica para tornar o código recursivo mais eficiente*. Há uma relação que precisa ser explorada: todos os algoritmos de programação dinâmica são recursivos mas, nem todos os algoritmos recursivos são programação dinâmica.

A recursividade é uma poderosa técnica de resolução de problemas. Um código recursivo pode ser provado matematicamente correto de forma relativamente simples. E, só isso já seria motivo suficiente para usar recursão em todos os seus códigos.

A prova da correção de um algoritmo recursivo geralmente envolve apenas duas etapas: a prova de que a base da recursão está correta e a prova de que o passo recursivo está correto. No domínio da prova matemática por indução podemos nos referir a estes componentes como a **base da indução** e o **passo da indução**, respectivamente. Neste caso:

  Para provar a **base da indução** verificamos o caso mais simples da recursão, geralmente o caso base ou os casos base, está correto. Esses são os casos que não dependem de chamadas recursivas.

  Para provar o **passo da indução** verificamos se a função recursiva está correta para todos os casos menores ou subproblemas, então ela também está correta para o caso geral. Ou seja, assume-se que a função está correta para entradas menores, ou para um conjunto menor de entradas, as hipótese de indução, e, com base nisso, provamos, ou não, que a função recursiva está correta.

Além da facilidade de prova matemática, o código recursivo se destaca por ser claro e intuitivo, principalmente para os problemas que possuem estruturas repetitivas como a travessia de árvores, a resolução de labirintos e o cálculo de séries matemáticas.

Muitos problemas são naturalmente definidos de forma recursiva. Por exemplo, a definição matemática da sequência de Fibonacci ou a estrutura de árvores binárias são inerentemente recursivas a partir da sua própria. Nestes casos, a solução recursiva será mais simples, direta e, provavelmente, mais eficiente.

Frequentemente, a solução recursiva é mais concisa e requer menos linhas de código quando comparada com a solução iterativa. Menos linhas, menos erros, mais facilidade de leitura e entendimento. Parece bom.

Por fim, a recursividade é a abordagem quase ideal para a aplicação das técnicas de solução baseadas em divisão e conquista. Desde Júlio Cesar que sabemos que é mais fácil dividir para conquistar. Neste caso, um problema é dividido em subproblemas, resolvidos individualmente e, em seguida, combinados para formar a solução final. Exemplos clássicos, e acadêmicos, destas técnicas incluem algoritmos de ordenação como *quicksort* e *mergesort*.

A doce leitora deve ter erguido as sobrancelhas. É aqui que a recursão e a programação dinâmica se tocam, não de forma sutil e delicada, como a carícia de um amante no rosto da amada. Mas, com a decisão e impacto da luva de Mike Tyson no queixo do adversário. A divisão do problema principal em subproblemas é a essência fundamental da recursão e da programação dinâmica.

A programação dinâmica e a recursividade estão relacionadas, ambas envolvem a resolução de problemas com a divisão de um problema em problemas menores. No entanto, enquanto a recursividade resolve os problemas menores sem considerar o custo computacional das chamadas repetidas, a programação dinâmica otimiza essas soluções via armazenamento e reutilização de resultados já obtidos. O exemplo mais típico de recursividade é a determinação do valor de ordem $n$ da Sequência de Fibonacci.

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
