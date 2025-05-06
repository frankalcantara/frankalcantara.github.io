---
layout: post
title: Multiplicação de Matrizes
author: Frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - C++
    - Python
    - Matemática
    - inteligência artificial
image: assets/images/multimatriz1.webp
featured: false
rating: 5
description: Análise dos algoritmos de multiplicação de matrizes.
date: 2025-02-09T22:55:34.524Z
preview: Uma introdução a matemática que suporta a criação de transformers para processamento de linguagem natural com exemplos de código em C++20.
keywords: |-
    transformers
    matemática
    processamento de linguagem natural
    C++
    aprendizado de máquina
    vetores
    produto escalar
    álgebra linear
    embeddings
    atenção
    deep learning
    inteligência artificial
toc: true
published: true
lastmod: 2025-05-06T11:04:17.985Z
---

A multiplicação de matrizes pode, sem dúvida, ser um dos tópicos mais importantes dos modelos de linguagem, e aprendizagem de máquina, disponíveis no mercado atualmente. Neste artigo, vamos explorar alguns algoritmos para multiplicação de matrizes, suas aplicações e como ele se relaciona com o funcionamento de modelos de aprendizado profundo, como os Transformers, que estamos estudando ([aqui](https://frankalcantara.com/voce-pensa-como-fala/),[aqui](https://frankalcantara.com/transformers-um/) e [aqui](https://frankalcantara.com/transformers-dois/)).

Apesar de toda a sua importância, a multiplicação de matrizes é um tópico que pode ser facilmente esquecido, ou mesmo negligenciado, em cursos de aprendizado de máquina. Isso ocorre porque a maioria dos cursos se concentra em algoritmos de aprendizado profundo e redes neurais, sem entrar em detalhes sobre os fundamentos matemáticos subjacentes. No entanto, entender como as matrizes são multiplicadas e como isso se relaciona com o funcionamento dos modelos parece que será relevante novamente nos anos que virão. Anos em que o custo de treinamento de modelos de linguagem tende a ser cada vez mais maior. Indicando um cenário onde a eficiência computacional voltará a ser um fator decisivo para o sucesso comercial.

Apesar de toda sua importância, em quase 90 anos de ciência da computação, progredimos muito pouco no estudo da complexidade computacional desta operação. A Tabela 1, a seguir resume os principais algoritmos conhecidos para multiplicação de matrizes, em relação ao expoente da sua complexidade assintótica.

| Expoente ($\omega$)         | Algoritmo/Pesquisador(es)   | Ano  |
| :------------------------: | :-------------------------- | :--: |
| $3$                        | Naive                       | -    |
| $\approx 2.808$            | Strassen                    | 1969 |
| $\approx 2.796$            | Pan                         | 1978 |
| $\approx 2.78$             | Bini et al.                 | 1979 |
| $\approx 2.522$            | Schönhage                   | 1981 |
| $\approx 2.496$            | Coppersmith & Winograd      | 1982 |
| $\approx 2.479$            | Strassen                    | 1986 |
| $\approx 2.375477$         | Coppersmith & Winograd      | 1987 |
| $\approx 2.374$            | Stothers                    | 2010 |
| $\approx 2.3728642$        | Williams                    | 2011 |
| $\approx 2.3728639$        | Le Gall                     | 2014 |
| $\approx 2.3728639$        | Alman, Duan, Wu, Zhou       | 2020-2022 |

_Tabela 1: Algoritmos de multiplicação de matrizes e seus expoentes._{: class="legend"}

**Notas:**

* O expoente $\omega \approx 2.375477$ (Coppersmith & Winograd, 1987) refere-se ao trabalho cujo artigo detalhado foi publicado no *Journal of Symbolic Computation* em 1990. O ano de 1987 geralmente se refere à publicação inicial nos anais da [conferência STOC](https://acm-stoc.org/).

* O valor para Stothers (2010) é um limite superior ($\omega < 2.374$), frequentemente arredondado ou aproximado na literatura; o limite exato alcançado foi ligeiramente menor.

* um conjunto de trabalhos recentes que refinaram o expoente da multiplicação de matrizes, mantendo-o em aproximadamente $2.37$, mas com melhorias incrementais nas constantes e análises teóricas.

Veremos alguns destes algoritmos, começando pelo algoritmo clássico, que é o mais simples e intuitivo. Em seguida, abordaremos o algoritmo de Strassen, que é um dos mais conhecidos e utilizados na prática. Por fim, discutiremos o algoritmo de Coppersmith-Winograd, que é um dos mais avançados e complexos.

## 1. Algoritmo Clássico (Ingênuo) de Multiplicação de Matrizes

### Introdução e Aplicações

A origem do algoritmo clássico, também chamado de ingênuo, para a multiplicação de matrizes deriva diretamente da definição matemática formal desta operação. Se temos uma matriz $A$ de dimensões $m \times p$ e uma matriz $B$ de dimensões $p \times n$, o produto $C = A \times B$ será uma matriz de dimensões $m \times n$.

A atenta leitora irá perceber que o funcionamento deste algoritmo baseia-se no cálculo de cada elemento $C_{ij}$ da matriz resultante $C$. Assim, o elemento na $i$-ésima linha e $j$-ésima coluna ($C_{ij}$) será obtido calculando-se o produto escalar (_dot product_) entre a $i$-ésima linha da matriz $A$ e a $j$-ésima coluna da matriz $B$. Matematicamente, isso será expresso por:

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}
$$

De tal forma que:

* $C_{ij}$ é o elemento na linha $i$ e coluna $j$ da matriz resultante $C$;
* $A_{ik}$ é o elemento na linha $i$ e coluna $k$ da matriz $A$;
* $B_{kj}$ é o elemento na linha $k$ e coluna $j$ da matriz $B$;
* A soma é realizada sobre o índice $k$, que varia de 1 até $p$ (o número de colunas de $A$ e o número de linhas de $B$).

Este processo pode ser visto na Figura 1, a seguir:

![exemplo de multiplicação de matrizes](/assets/images/multmatriz2.webp)
_Figura 1: Exemplo de multiplicação de matrizes._{: class="legend"}

Para calcular todos os elementos da matriz $C$, o algoritmo percorre todas as linhas $i$ de $A$ (de 1 a $m$), todas as colunas $j$ de $B$ (de 1 a $n$), e para cada par $(i, j)$, realiza a soma dos produtos dos elementos correspondentes, percorrendo o índice $k$ (de 1 a $p$). Resultando em uma implementação com três laços (_loops_) aninhados.

```shell
Algoritmo Multiplicacao_Classica(A, B)
Entrada: Matriz A (m x p), Matriz B (p x n)
Saída: Matriz C (m x n)

1.  Verificar se o número de colunas de A é igual ao número de linhas de B. Se não, retornar erro.
2.  Inicializar a matriz resultado C com dimensões m x n, preenchida com zeros.
3.  Para i de 0 até m-1:
4.    Para j de 0 até n-1:
5.      Inicializar soma = 0
6.      Para k de 0 até p-1:
7.        soma = soma + A[i][k] * B[k][j]
8.      Fim Para (k)
9.      C[i][j] = soma
10.   Fim Para (j)
11. Fim Para (i)
12. Retornar C
```

### Exemplos Numéricos - Algoritmo Clássico

**Exemplo 1**: Matrizes Quadradas $2\times 2$

Sejam as matrizes $A$ e $B$:

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}
$$

O produto $C = A \times B$ será uma matriz $2 \times 2$:

$$
C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$

Calculando cada elemento, teremos:

* $C_{11} = \sum_{k=1}^{2} A_{1k} B_{k1} = A_{11}B_{11} + A_{12}B_{21} = (1)(5) + (2)(7) = 5 + 14 = 19$
* $C_{12} = \sum_{k=1}^{2} A_{1k} B_{k2} = A_{11}B_{12} + A_{12}B_{22} = (1)(6) + (2)(8) = 6 + 16 = 22$
* $C_{21} = \sum_{k=1}^{2} A_{2k} B_{k1} = A_{21}B_{11} + A_{22}B_{21} = (3)(5) + (4)(7) = 15 + 28 = 43$
* $C_{22} = \sum_{k=1}^{2} A_{2k} B_{k2} = A_{21}B_{12} + A_{22}B_{22} = (3)(6) + (4)(8) = 18 + 32 = 50$

Portanto, a matriz resultante será:

$$
C = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}
$$

**Exemplo 2**: Matrizes Retangulares ($3\times 2$ e $2\times 3$)

Sejam as matrizes $A$ (dimensão $3 \times 2$) e $B$ (dimensão $2 \times 3$):

$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \quad B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}
$$

O produto $C = A \times B$ será uma matriz $3 \times 3$:

$$
C = \begin{pmatrix} C_{11} & C_{12} & C_{13} \\ C_{21} & C_{22} & C_{23} \\ C_{31} & C_{32} & C_{33} \end{pmatrix}
$$

Calculando cada elemento, teremos:

* $C_{11} = (1)(7) + (2)(10) = 7 + 20 = 27$
* $C_{12} = (1)(8) + (2)(11) = 8 + 22 = 30$
* $C_{13} = (1)(9) + (2)(12) = 9 + 24 = 33$
* $C_{21} = (3)(7) + (4)(10) = 21 + 40 = 61$
* $C_{22} = (3)(8) + (4)(11) = 24 + 44 = 68$
* $C_{23} = (3)(9) + (4)(12) = 27 + 48 = 75$
* $C_{31} = (5)(7) + (6)(10) = 35 + 60 = 95$
* $C_{32} = (5)(8) + (6)(11) = 40 + 66 = 106$
* $C_{33} = (5)(9) + (6)(12) = 45 + 72 = 117$

Portanto, a matriz resultante será:

$$
C = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}
$$

### Implementação em Python (Exemplo)

```python
def multiplicar_matrizes_classico(matriz_a, matriz_b):
   """
   Multiplica duas matrizes usando o algoritmo clássico (ingênuo).

   Args:
       matriz_a: Uma lista de listas representando a matriz A (m x p).
       matriz_b: Uma lista de listas representando a matriz B (p x n).

   Returns:
       Uma lista de listas representando a matriz resultante C (m x n),
       ou None se as matrizes não forem compatíveis para multiplicação.
   """
   # Obter dimensões
   m = len(matriz_a)
   if m == 0:
       return None # Matriz A vazia
   p_a = len(matriz_a[0])
   
   # Verificar se todas as linhas de A têm o mesmo tamanho
   if any(len(linha) != p_a for linha in matriz_a):
       print("Erro: Matriz A não é regular.")
       return None
   
   if p_a == 0:
       return None # Linhas da Matriz A vazias

   p_b = len(matriz_b)
   if p_b == 0:
       return None # Matriz B vazia
   n = len(matriz_b[0])
   
   # Verificar se todas as linhas de B têm o mesmo tamanho
   if any(len(linha) != n for linha in matriz_b):
       print("Erro: Matriz B não é regular.")
       return None
       
   if n == 0:
       return None # Linhas da Matriz B vazias

   # Verificar compatibilidade de dimensões
   if p_a != p_b:
       print(f"Erro: Número de colunas de A ({p_a}) não é igual ao número de linhas de B ({p_b}).")
       return None

   # Inicializar matriz resultado C com zeros
   matriz_c = [[0 for _ in range(n)] for _ in range(m)]

   # Executar a multiplicação
   for i in range(m):        # Itera sobre as linhas de A (e C)
       for j in range(n):    # Itera sobre as colunas de B (e C)
           soma_produto = 0
           for k in range(p_a):  # Itera sobre as colunas de A / linhas de B
               soma_produto += matriz_a[i][k] * matriz_b[k][j]
           matriz_c[i][j] = soma_produto

   return matriz_c

# Exemplo de uso com os dados do Exemplo 1:
A1 = [[1, 2], [3, 4]]
B1 = [[5, 6], [7, 8]]
C1 = multiplicar_matrizes_classico(A1, B1)
print("Resultado Exemplo 1:")
if C1:
   for linha in C1:
       print(linha)
# Esperado:
# [19, 22]
# [43, 50]

print("\n" + "="*20 + "\n")

# Exemplo de uso com os dados do Exemplo 2:
A2 = [[1, 2], [3, 4], [5, 6]]
B2 = [[7, 8, 9], [10, 11, 12]]
C2 = multiplicar_matrizes_classico(A2, B2)
print("Resultado Exemplo 2:")
if C2:
   for linha in C2:
       print(linha)
# Esperado:
# [27, 30, 33]
# [61, 68, 75]
# [95, 106, 117]

print("\n" + "="*20 + "\n")

# Exemplo de matrizes incompatíveis:
A3 = [[1, 2], [3, 4]]
B3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] # B é 3x3, A é 2x2 -> incompatível
print("Resultado Exemplo 3 (Incompatível):")
C3 = multiplicar_matrizes_classico(A3, B3)
if C3 is None:
   print("As matrizes são incompatíveis para multiplicação, como esperado.")
```

### Implementação em C++ 20 (Exemplo)

```cpp
#include <iostream>
#include <vector>
#include <optional>
#include <format>

/**
 * @typedef Matrix
 * @brief Define uma matriz como um vetor bidimensional de doubles.
 */
using Matrix = std::vector<std::vector<double>>;

/**
 * @brief Verifica se uma matriz é regular (todas as linhas têm o mesmo tamanho).
 * @param mat A matriz a ser verificada.
 * @return true se a matriz for regular, false caso contrário.
 */
bool is_regular(const Matrix& mat) {
    if (mat.empty()) return true;
    
    const size_t cols = mat[0].size();
    return std::all_of(mat.begin(), mat.end(), 
                      [cols](const auto& row) { return row.size() == cols; });
}

/**
 * @brief Obtém as dimensões de uma matriz.
 * @param mat A matriz cuja dimensão será obtida.
 * @return Um par contendo (linhas, colunas). Retorna (0,0) se a matriz for vazia.
 */
std::pair<size_t, size_t> get_dimensions(const Matrix& mat) {
    if (mat.empty()) return {0, 0};
    if (mat[0].empty()) return {mat.size(), 0};
    return {mat.size(), mat[0].size()};
}

/**
 * @brief Imprime uma matriz formatada no console.
 * @param mat A matriz a ser impressa.
 * @param name Nome opcional da matriz para exibição.
 */
void print_matrix(const Matrix& mat, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << " =\n";
    }
    
    for (const auto& row : mat) {
        std::cout << "[ ";
        for (const auto& elem : row) {
            std::cout << std::format("{:8.2f} ", elem);
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Multiplica duas matrizes usando o algoritmo clássico (ingênuo).
 * @param A Primeira matriz de dimensões m x p.
 * @param B Segunda matriz de dimensões p x n.
 * @return Uma std::optional contendo a matriz resultante C de dimensões m x n, 
 *         ou std::nullopt se as matrizes não puderem ser multiplicadas.
 */
std::optional<Matrix> multiply_matrices(const Matrix& A, const Matrix& B) {
    // Verificar se as matrizes são regulares
    if (!is_regular(A) || !is_regular(B)) {
        std::cerr << "Erro: As matrizes devem ser regulares (todas as linhas devem ter o mesmo tamanho).\n";
        return std::nullopt;
    }
    
    // Obter dimensões
    auto [m, p_a] = get_dimensions(A);
    auto [p_b, n] = get_dimensions(B);
    
    // Verificar se as matrizes são vazias
    if (m == 0 || p_a == 0 || p_b == 0 || n == 0) {
        std::cerr << "Erro: As matrizes não podem ser vazias.\n";
        return std::nullopt;
    }
    
    // Verificar compatibilidade de dimensões
    if (p_a != p_b) {
        std::cerr << std::format("Erro: Número de colunas de A ({}) não é igual ao número de linhas de B ({}).\n", p_a, p_b);
        return std::nullopt;
    }
    
    // Inicializar matriz resultado C com zeros
    Matrix C(m, std::vector<double>(n, 0.0));
    
    // Executar a multiplicação
    for (size_t i = 0; i < m; ++i) {         // Itera sobre as linhas de A (e C)
        for (size_t j = 0; j < n; ++j) {     // Itera sobre as colunas de B (e C)
            double sum = 0.0;
            for (size_t k = 0; k < p_a; ++k) {  // Itera sobre as colunas de A / linhas de B
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

int main() {
    // Exemplo 1: Matrizes 2x2
    Matrix A1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix B1 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    std::cout << "=== Exemplo 1: Matrizes 2x2 ===\n";
    print_matrix(A1, "A");
    print_matrix(B1, "B");
    
    if (auto C1 = multiply_matrices(A1, B1)) {
        print_matrix(*C1, "Resultado C = A x B");
    } else {
        std::cout << "Não foi possível multiplicar as matrizes A e B.\n";
    }
    
    // Exemplo 2: Matrizes retangulares (3x2 e 2x3)
    Matrix A2 = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    
    Matrix B2 = {
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0}
    };
    
    std::cout << "\n=== Exemplo 2: Matrizes retangulares (3x2 e 2x3) ===\n";
    print_matrix(A2, "A");
    print_matrix(B2, "B");
    
    if (auto C2 = multiply_matrices(A2, B2)) {
        print_matrix(*C2, "Resultado C = A x B");
    } else {
        std::cout << "Não foi possível multiplicar as matrizes A e B.\n";
    }
    
    // Exemplo 3: Matrizes incompatíveis
    Matrix A3 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix B3 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    std::cout << "\n=== Exemplo 3: Matrizes incompatíveis ===\n";
    print_matrix(A3, "A");
    print_matrix(B3, "B");
    
    if (auto C3 = multiply_matrices(A3, B3)) {
        print_matrix(*C3, "Resultado C = A x B");
    } else {
        std::cout << "Não foi possível multiplicar as matrizes A e B (incompatíveis).\n";
    }
    
    return 0;
}
```

### Complexidade Computacional

A complexidade computacional deste algoritmo é determinada pelo número de multiplicações e adições realizadas. Assim, teremos:

* Para cada um dos $m \times n$ elementos de $C$, realizamos $p$ multiplicações e $p-1$ adições;
* O número total de multiplicações é $m \times n \times p$;
* Se as matrizes forem quadradas de dimensão $n \times n$ ($m=p=n$), a complexidade é $O(n^3)$

Esta complexidade cúbica para matrizes quadradas motivou a busca por algoritmos mais eficientes, como o algoritmo de Strassen (complexidade $O(n^{2.807})$) e outros algoritmos assintoticamente mais rápidos, que se tornam vantajosos para matrizes de grandes dimensões.

## Algoritmo de Strassen (1969)

O algoritmo de Strassen foi publicado por Volker Strassen em 1969, representando um marco histórico na teoria de complexidade computacional. Foi o primeiro algoritmo a demonstrar que a multiplicação de matrizes $n \times n$ poderia ser realizada com complexidade assintótica inferior a $O(n^3)$, que era considerada ótima até então.

O algoritmo baseia-se na técnica de **dividir para conquistar**. A ideia central é tratar as matrizes $n \times n$ a serem multiplicadas, $A$ e $B$, como se fossem matrizes $2 \times 2$ compostas por blocos (submatrizes) de tamanho $(n/2) \times (n/2)$. Assumindo, por simplicidade inicial, que $n$ é uma potência de 2:

$$
A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}, \quad B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}, \quad C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$

A atenta leitora deve lembrar que o algoritmo clássico para multiplicação destas matrizes exigiria $8$ multiplicações de submatrizes $(n/2) \times (n/2)$ e $4$ adições. _Strassen descobriu uma forma de calcular o resultado utilizando apenas $7$ multiplicações de submatrizes, com o custo de realizar mais operações de adição e subtração_.

### Processo de Particionamento de Matriz

As $7$ multiplicações intermediárias (produtos $P_1$ a $P_7$) serão definidas como:

* $P_1 = (A_{11} + A_{22})(B_{11} + B_{22})$
* $P_2 = (A_{21} + A_{22})B_{11}$
* $P_3 = A_{11}(B_{12} - B_{22})$
* $P_4 = A_{22}(B_{21} - B_{11})$
* $P_5 = (A_{11} + A_{12})B_{22}$
* $P_6 = (A_{21} - A_{11})(B_{11} + B_{12})$
* $P_7 = (A_{12} - A_{22})(B_{21} + B_{22})$

E os blocos da matriz resultante $C$ são calculados combinando esses produtos:

* $C_{11} = P_1 + P_4 - P_5 + P_7$
* $C_{12} = P_3 + P_5$
* $C_{21} = P_2 + P_4$
* $C_{22} = P_1 - P_2 + P_3 + P_6$

O algoritmo de Strassen aplica **recursivamente** este mesmo método para calcular cada um desses 7 produtos, até atingir um caso base, como matrizes $1 \times 1$, onde a multiplicação é trivial, ou um tamanho mínimo onde o algoritmo clássico seja mais eficiente devido à sobrecarga das operações adicionais.

### Exemplos Numéricos - Algoritmo de Strassen

**Exemplo 1**: Conceitual para $2\times 2$

Vamos aplicar as fórmulas de Strassen para $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ e $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$.
Neste caso, as "submatrizes" $A_{ij}$ e $B_{ij}$ são apenas números ($n=2$, $n/2=1$).

* $A_{11}=1, A_{12}=2, A_{21}=3, A_{22}=4$
* $B_{11}=5, B_{12}=6, B_{21}=7, B_{22}=8$

Calculando os produtos $P_i$ (agora são multiplicações de números):

* $P_1 = (1+4)(5+8) = 5 \times 13 = 65$
* $P_2 = (3+4)(5) = 7 \times 5 = 35$
* $P_3 = (1)(6-8) = 1 \times (-2) = -2$
* $P_4 = (4)(7-5) = 4 \times 2 = 8$
* $P_5 = (1+2)(8) = 3 \times 8 = 24$
* $P_6 = (3-1)(5+6) = 2 \times 11 = 22$
* $P_7 = (2-4)(7+8) = (-2) \times 15 = -30$

Calculando os elementos $C_{ij}$:

* $C_{11} = P_1 + P_4 - P_5 + P_7 = 65 + 8 - 24 + (-30) = 19$
* $C_{12} = P_3 + P_5 = -2 + 24 = 22$
* $C_{21} = P_2 + P_4 = 35 + 8 = 43$
* $C_{22} = P_1 - P_2 + P_3 + P_6 = 65 - 35 + (-2) + 22 = 50$

A matriz resultante é $C = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$, que coincide com o resultado do algoritmo clássico.

**Exemplo 2**: passo a passo do algoritmo de Strassen para multiplicação de matrizes $4\times 4$.

Vamos considerar as seguintes matrizes 4×4:

$$A = \begin{pmatrix}
2 & 3 & 4 & 1 \\
1 & 0 & 2 & 3 \\
5 & 2 & 1 & 4 \\
3 & 4 & 2 & 0
\end{pmatrix}, \quad
B = \begin{pmatrix}
1 & 2 & 3 & 4 \\
2 & 1 & 4 & 0 \\
3 & 4 & 1 & 2 \\
4 & 3 & 2 & 1
\end{pmatrix}$$

**Passo 1**: dividir cada matriz em submatrizes $2\times 2$. Primeiro, dividimos cada matriz em 4 submatrizes:

$$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}, \quad
B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}$$

De forma que:

$$A_{11} = \begin{pmatrix} 2 & 3 \\ 1 & 0 \end{pmatrix}, \quad
A_{12} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$$

$$A_{21} = \begin{pmatrix} 5 & 2 \\ 3 & 4 \end{pmatrix}, \quad
A_{22} = \begin{pmatrix} 1 & 4 \\ 2 & 0 \end{pmatrix}$$

$$B_{11} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}, \quad
B_{12} = \begin{pmatrix} 3 & 4 \\ 4 & 0 \end{pmatrix}$$

$$B_{21} = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix}, \quad
B_{22} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$$

**Passo 2**: calcular os termos intermediários para os produtos Strassen. Calculamos as $10$ somas/diferenças de submatrizes:

$S_1 = A_{11} + A_{22} = \begin{pmatrix} 2 & 3 \\ 1 & 0 \end{pmatrix} + \begin{pmatrix} 1 & 4 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 7 \\ 3 & 0 \end{pmatrix}$

$S_2 = B_{11} + B_{22} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} + \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 4 \\ 4 & 2 \end{pmatrix}$

$S_3 = A_{21} + A_{22} = \begin{pmatrix} 5 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 1 & 4 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 6 & 6 \\ 5 & 4 \end{pmatrix}$

$S_4 = B_{12} - B_{22} = \begin{pmatrix} 3 & 4 \\ 4 & 0 \end{pmatrix} - \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 2 & -1 \end{pmatrix}$

$S_5 = B_{21} - B_{11} = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix} - \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}$

$S_6 = A_{11} + A_{12} = \begin{pmatrix} 2 & 3 \\ 1 & 0 \end{pmatrix} + \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 6 & 4 \\ 3 & 3 \end{pmatrix}$

$S_7 = A_{21} - A_{11} = \begin{pmatrix} 5 & 2 \\ 3 & 4 \end{pmatrix} - \begin{pmatrix} 2 & 3 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 3 & -1 \\ 2 & 4 \end{pmatrix}$

$S_8 = B_{11} + B_{12} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} + \begin{pmatrix} 3 & 4 \\ 4 & 0 \end{pmatrix} = \begin{pmatrix} 4 & 6 \\ 6 & 1 \end{pmatrix}$

$S_9 = A_{12} - A_{22} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix} - \begin{pmatrix} 1 & 4 \\ 2 & 0 \end{pmatrix} = \begin{pmatrix} 3 & -3 \\ 0 & 3 \end{pmatrix}$

$S_{10} = B_{21} + B_{22} = \begin{pmatrix} 3 & 4 \\ 4 & 3 \end{pmatrix} + \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 4 & 6 \\ 6 & 4 \end{pmatrix}$

**Passo 3**: calcular os $7$ produtos de Strassen. Agora, calculamos os 7 produtos de Strassen. Aqui a atenta leitora deve considerar que em um algoritmo recursivo completo seriam calculados usando Strassen novamente:

$P_1 = S_1 \times S_2 = \begin{pmatrix} 3 & 7 \\ 3 & 0 \end{pmatrix} \times \begin{pmatrix} 2 & 4 \\ 4 & 2 \end{pmatrix}$

Calculando essa multiplicação de matriz 2×2 diretamente:
$P_1 = \begin{pmatrix} 3 \times 2 + 7 \times 4 & 3 \times 4 + 7 \times 2 \\ 3 \times 2 + 0 \times 4 & 3 \times 4 + 0 \times 2 \end{pmatrix} = \begin{pmatrix} 34 & 26 \\ 6 & 12 \end{pmatrix}$

Similarmente, para os outros produtos:

$P_2 = S_3 \times B_{11} = \begin{pmatrix} 6 & 6 \\ 5 & 4 \end{pmatrix} \times \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 18 & 18 \\ 13 & 14 \end{pmatrix}$

$P_3 = A_{11} \times S_4 = \begin{pmatrix} 2 & 3 \\ 1 & 0 \end{pmatrix} \times \begin{pmatrix} 2 & 2 \\ 2 & -1 \end{pmatrix} = \begin{pmatrix} 10 & 1 \\ 2 & 2 \end{pmatrix}$

$P_4 = A_{22} \times S_5 = \begin{pmatrix} 1 & 4 \\ 2 & 0 \end{pmatrix} \times \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 10 & 10 \\ 4 & 4 \end{pmatrix}$

$P_5 = S_6 \times B_{22} = \begin{pmatrix} 6 & 4 \\ 3 & 3 \end{pmatrix} \times \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 14 & 16 \\ 9 & 9 \end{pmatrix}$

$P_6 = S_7 \times S_8 = \begin{pmatrix} 3 & -1 \\ 2 & 4 \end{pmatrix} \times \begin{pmatrix} 4 & 6 \\ 6 & 1 \end{pmatrix} = \begin{pmatrix} 6 & 17 \\ 32 & 16 \end{pmatrix}$

$P_7 = S_9 \times S_{10} = \begin{pmatrix} 3 & -3 \\ 0 & 3 \end{pmatrix} \times \begin{pmatrix} 4 & 6 \\ 6 & 4 \end{pmatrix} = \begin{pmatrix} -6 & 6 \\ 18 & 12 \end{pmatrix}$

**Passo 4**: calcular os quadrantes da matriz resultante. Neste ponto, utilizamos os produtos $P_1$ a $P_7$ para calcular os quatro quadrantes da matriz resultante $C$:

$C_{11} = P_1 + P_4 - P_5 + P_7 = \begin{pmatrix} 34 & 26 \\ 6 & 12 \end{pmatrix} + \begin{pmatrix} 10 & 10 \\ 4 & 4 \end{pmatrix} - \begin{pmatrix} 14 & 16 \\ 9 & 9 \end{pmatrix} + \begin{pmatrix} -6 & 6 \\ 18 & 12 \end{pmatrix} = \begin{pmatrix} 24 & 26 \\ 19 & 19 \end{pmatrix}$

$C_{12} = P_3 + P_5 = \begin{pmatrix} 10 & 1 \\ 2 & 2 \end{pmatrix} + \begin{pmatrix} 14 & 16 \\ 9 & 9 \end{pmatrix} = \begin{pmatrix} 24 & 17 \\ 11 & 11 \end{pmatrix}$

$C_{21} = P_2 + P_4 = \begin{pmatrix} 18 & 18 \\ 13 & 14 \end{pmatrix} + \begin{pmatrix} 10 & 10 \\ 4 & 4 \end{pmatrix} = \begin{pmatrix} 28 & 28 \\ 17 & 18 \end{pmatrix}$

$C_{22} = P_1 - P_2 + P_3 + P_6 = \begin{pmatrix} 34 & 26 \\ 6 & 12 \end{pmatrix} - \begin{pmatrix} 18 & 18 \\ 13 & 14 \end{pmatrix} + \begin{pmatrix} 10 & 1 \\ 2 & 2 \end{pmatrix} + \begin{pmatrix} 6 & 17 \\ 32 & 16 \end{pmatrix} = \begin{pmatrix} 32 & 26 \\ 27 & 16 \end{pmatrix}$

**Passo 5**: recombinar para obter a matriz resultante completa. Finalmente, combinamos os quatro quadrantes para formar a matriz resultado:

$$C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix} = \begin{pmatrix}
24 & 26 & 24 & 17 \\
19 & 19 & 11 & 11 \\
28 & 28 & 32 & 26 \\
17 & 18 & 27 & 16
\end{pmatrix}$$

**Passo 6**: verificação. Para verificar, podemos calcular o resultado usando o algoritmo clássico:
$$C = A \times B$$

Onde $C_{ij} = \sum_{k=1}^{4} A_{ik} \times B_{kj}$

Calculando alguns elementos para verificação:
$C_{11} = 2 \times 1 + 3 \times 2 + 4 \times 3 + 1 \times 4 = 2 + 6 + 12 + 4 = 24$
$C_{12} = 2 \times 2 + 3 \times 1 + 4 \times 4 + 1 \times 3 = 4 + 3 + 16 + 3 = 26$

E assim por diante, confirmando que o resultado do algoritmo de Strassen está correto. Este exemplo ilustra como o algoritmo de Strassen funciona para matrizes $4\times 4$, dividindo-as em submatrizes $2\times 2$. Para matrizes maiores, o mesmo processo se aplicaria recursivamente. Por exemplo, para matrizes $8\times 8$, dividiríamos em submatrizes $4\times 4$, e cada multiplicação de submatrizes 4×4 seguiria o processo mostrado neste exemplo. Para este tamanho de matriz, o algoritmo de Strassen não será, necessariamente, mais eficiente que o algoritmo clássico devido ao custo das operações adicionais. _A vantagem de Strassen torna-se aparente apenas para matrizes muito grandes_.

### Implementação em Python (Exemplo)

```python
import numpy as np
import time
from typing import List, Tuple, Optional, Union
import math

# Tipo Matrix: lista de listas de float
Matrix = List[List[float]]

def print_matrix(mat: Matrix, name: str = "") -> None:
   """Imprime uma matriz formatada no console."""
   if name:
       print(f"{name} =")

   for row in mat:
       print("[", end=" ")
       for elem in row:
           print(f"{elem:8.2f}", end=" ")
       print("]")
   print()

def get_dimensions(mat: Matrix) -> Tuple[int, int]:
   """Obtém as dimensões de uma matriz."""
   if not mat:
       return 0, 0
   if not mat[0]:
       return len(mat), 0
   return len(mat), len(mat[0])

def is_regular(mat: Matrix) -> bool:
   """Verifica se uma matriz é regular (todas as linhas têm o mesmo tamanho)."""
   if not mat:
       return True

   cols = len(mat[0])
   return all(len(row) == cols for row in mat)

def add_matrices(A: Matrix, B: Matrix) -> Optional[Matrix]:
   """Adiciona duas matrizes A e B."""
   rows_a, cols_a = get_dimensions(A)
   rows_b, cols_b = get_dimensions(B)

   if rows_a != rows_b or cols_a != cols_b:
       return None

   C = [[0.0 for _ in range(cols_a)] for _ in range(rows_a)]

   for i in range(rows_a):
       for j in range(cols_a):
           C[i][j] = A[i][j] + B[i][j]

   return C

def subtract_matrices(A: Matrix, B: Matrix) -> Optional[Matrix]:
   """Subtrai a matriz B da matriz A."""
   rows_a, cols_a = get_dimensions(A)
   rows_b, cols_b = get_dimensions(B)

   if rows_a != rows_b or cols_a != cols_b:
       return None

   C = [[0.0 for _ in range(cols_a)] for _ in range(rows_a)]

   for i in range(rows_a):
       for j in range(cols_a):
           C[i][j] = A[i][j] - B[i][j]

   return C

def split_matrix(mat: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
   """Divide uma matriz em quatro submatrizes (quadrantes)."""
   rows, cols = get_dimensions(mat)
   mid_row = rows // 2
   mid_col = cols // 2

   A11 = [[mat[i][j] for j in range(mid_col)] for i in range(mid_row)]
   A12 = [[mat[i][j] for j in range(mid_col, cols)] for i in range(mid_row)]
   A21 = [[mat[i][j] for j in range(mid_col)] for i in range(mid_row, rows)]
   A22 = [[mat[i][j] for j in range(mid_col, cols)] for i in range(mid_row, rows)]

   return A11, A12, A21, A22

def combine_matrices(A11: Matrix, A12: Matrix, A21: Matrix, A22: Matrix) -> Matrix:
   """Combina quatro submatrizes em uma matriz maior."""
   rows11, cols11 = get_dimensions(A11)
   rows12, cols12 = get_dimensions(A12)
   rows21, cols21 = get_dimensions(A21)
   rows22, cols22 = get_dimensions(A22)

   rows = rows11 + rows21
   cols = cols11 + cols12

   result = [[0.0 for _ in range(cols)] for _ in range(rows)]

   # Copiar A11 (superior esquerdo)
   for i in range(rows11):
       for j in range(cols11):
           result[i][j] = A11[i][j]

   # Copiar A12 (superior direito)
   for i in range(rows12):
       for j in range(cols12):
           result[i][j + cols11] = A12[i][j]

   # Copiar A21 (inferior esquerdo)
   for i in range(rows21):
       for j in range(cols21):
           result[i + rows11][j] = A21[i][j]

   # Copiar A22 (inferior direito)
   for i in range(rows22):
       for j in range(cols22):
           result[i + rows11][j + cols11] = A22[i][j]

   return result

def multiply_matrices_classic(A: Matrix, B: Matrix) -> Optional[Matrix]:
   """Multiplicação de matrizes usando o algoritmo clássico."""
   rows_a, cols_a = get_dimensions(A)
   rows_b, cols_b = get_dimensions(B)

   if cols_a != rows_b:
       print(f"Dimensões incompatíveis para multiplicação: {cols_a} != {rows_b}")
       return None

   C = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

   for i in range(rows_a):
       for j in range(cols_b):
           for k in range(cols_a):
               C[i][j] += A[i][k] * B[k][j]

   return C

def is_power_of_two(n: int) -> bool:
   """Verifica se um número é potência de 2."""
   return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n: int) -> int:
   """Retorna a próxima potência de 2 maior ou igual a n."""
   if n == 0:
       return 1
   if is_power_of_two(n):
       return n

   return 2 ** math.ceil(math.log2(n))

def pad_matrix(mat: Matrix, new_size: int) -> Matrix:
   """Redimensiona uma matriz para uma dimensão potência de 2."""
   rows, cols = get_dimensions(mat)

   padded = [[0.0 for _ in range(new_size)] for _ in range(new_size)]

   # Copiar os valores originais
   for i in range(rows):
       for j in range(cols):
           padded[i][j] = mat[i][j]

   return padded

def unpad_matrix(padded: Matrix, original_rows: int, original_cols: int) -> Matrix:
   """Remove o padding de uma matriz."""
   result = [[padded[i][j] for j in range(original_cols)] for i in range(original_rows)]
   return result

def strassen(A: Matrix, B: Matrix, threshold: int = 64) -> Optional[Matrix]:
   """
   Multiplicação de matrizes usando o algoritmo de Strassen.

   Args:
       A: Primeira matriz
       B: Segunda matriz
       threshold: Tamanho mínimo da matriz para usar Strassen (caso base)

   Returns:
       Matriz resultante ou None se as dimensões forem incompatíveis
   """
   rows_a, cols_a = get_dimensions(A)
   rows_b, cols_b = get_dimensions(B)

   # Verificar compatibilidade para multiplicação
   if cols_a != rows_b:
       print(f"Dimensões incompatíveis para multiplicação: {cols_a} != {rows_b}")
       return None

   # Determinar o tamanho da matriz resultante
   result_rows = rows_a
   result_cols = cols_b

   # Verificar se as matrizes precisam de padding
   needs_padding = (not is_power_of_two(rows_a) or
                   not is_power_of_two(cols_a) or
                   not is_power_of_two(rows_b) or
                   not is_power_of_two(cols_b) or
                   rows_a != cols_a or rows_b != cols_b)

   padded_A = A
   padded_B = B

   if needs_padding:
       # Encontrar o tamanho para o padding (próxima potência de 2)
       max_dim = max(rows_a, cols_a, rows_b, cols_b)
       padded_size = next_power_of_two(max_dim)

       # Aplicar padding
       padded_A = pad_matrix(A, padded_size)
       padded_B = pad_matrix(B, padded_size)

   # Caso base: usar o algoritmo clássico se a matriz for pequena
   padded_size_a, _ = get_dimensions(padded_A)
   if padded_size_a <= threshold:
       result = multiply_matrices_classic(padded_A, padded_B)

       if result is None or not needs_padding:
           return result

       # Remover o padding do resultado
       return unpad_matrix(result, result_rows, result_cols)

   # Dividir cada matriz em quatro submatrizes
   A11, A12, A21, A22 = split_matrix(padded_A)
   B11, B12, B21, B22 = split_matrix(padded_B)

   # Calcular os sete produtos de Strassen
   # P1 = (A11 + A22) * (B11 + B22)
   S1 = add_matrices(A11, A22)
   S2 = add_matrices(B11, B22)
   P1 = strassen(S1, S2, threshold)

   # P2 = (A21 + A22) * B11
   S3 = add_matrices(A21, A22)
   P2 = strassen(S3, B11, threshold)

   # P3 = A11 * (B12 - B22)
   S4 = subtract_matrices(B12, B22)
   P3 = strassen(A11, S4, threshold)

   # P4 = A22 * (B21 - B11)
   S5 = subtract_matrices(B21, B11)
   P4 = strassen(A22, S5, threshold)

   # P5 = (A11 + A12) * B22
   S6 = add_matrices(A11, A12)
   P5 = strassen(S6, B22, threshold)

   # P6 = (A21 - A11) * (B11 + B12)
   S7 = subtract_matrices(A21, A11)
   S8 = add_matrices(B11, B12)
   P6 = strassen(S7, S8, threshold)

   # P7 = (A12 - A22) * (B21 + B22)
   S9 = subtract_matrices(A12, A22)
   S10 = add_matrices(B21, B22)
   P7 = strassen(S9, S10, threshold)

   # Calcular os blocos da matriz resultante
   # C11 = P1 + P4 - P5 + P7
   C11_temp1 = add_matrices(P1, P4)
   C11_temp2 = subtract_matrices(C11_temp1, P5)
   C11 = add_matrices(C11_temp2, P7)

   # C12 = P3 + P5
   C12 = add_matrices(P3, P5)

   # C21 = P2 + P4
   C21 = add_matrices(P2, P4)

   # C22 = P1 - P2 + P3 + P6
   C22_temp1 = subtract_matrices(P1, P2)
   C22_temp2 = add_matrices(C22_temp1, P3)
   C22 = add_matrices(C22_temp2, P6)

   # Combinar os blocos para formar a matriz resultante
   result = combine_matrices(C11, C12, C21, C22)

   # Remover o padding, se necessário
   if needs_padding:
       return unpad_matrix(result, result_rows, result_cols)

   return result

# Exemplo 1: Matrizes 2x2
A1 = [
   [1.0, 2.0],
   [3.0, 4.0]
]

B1 = [
   [5.0, 6.0],
   [7.0, 8.0]
]

print("=== Exemplo 1: Matrizes 2x2 ===")
print_matrix(A1, "A")
print_matrix(B1, "B")

# Multiplicação usando o algoritmo clássico
start_classic = time.time()
C1_classic = multiply_matrices_classic(A1, B1)
end_classic = time.time()
duration_classic = (end_classic - start_classic) * 1000  # ms

if C1_classic:
   print_matrix(C1_classic, "Resultado (Clássico)")
   print(f"Tempo (Clássico): {duration_classic:.6f} ms\n")

# Multiplicação usando o algoritmo de Strassen
start_strassen = time.time()
C1_strassen = strassen(A1, B1, 1)  # Threshold = 1 para forçar o uso de Strassen
end_strassen = time.time()
duration_strassen = (end_strassen - start_strassen) * 1000  # ms

if C1_strassen:
   print_matrix(C1_strassen, "Resultado (Strassen)")
   print(f"Tempo (Strassen): {duration_strassen:.6f} ms\n")

# Exemplo 2: Matrizes 4x4
A2 = [
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [9.0, 10.0, 11.0, 12.0],
   [13.0, 14.0, 15.0, 16.0]
]

B2 = [
   [17.0, 18.0, 19.0, 20.0],
   [21.0, 22.0, 23.0, 24.0],
   [25.0, 26.0, 27.0, 28.0],
   [29.0, 30.0, 31.0, 32.0]
]

print("=== Exemplo 2: Matrizes 4x4 ===")
print_matrix(A2, "A")
print_matrix(B2, "B")

# Multiplicação usando o algoritmo clássico
start_classic = time.time()
C2_classic = multiply_matrices_classic(A2, B2)
end_classic = time.time()
duration_classic = (end_classic - start_classic) * 1000  # ms

if C2_classic:
   print_matrix(C2_classic, "Resultado (Clássico)")
   print(f"Tempo (Clássico): {duration_classic:.6f} ms\n")

# Multiplicação usando o algoritmo de Strassen
start_strassen = time.time()
C2_strassen = strassen(A2, B2, 2)  # Threshold = 2 para forçar o uso de Strassen
end_strassen = time.time()
duration_strassen = (end_strassen - start_strassen) * 1000  # ms

if C2_strassen:
   print_matrix(C2_strassen, "Resultado (Strassen)")
   print(f"Tempo (Strassen): {duration_strassen:.6f} ms\n")

# Exemplo 3: Matriz não-quadrada
A3 = [
   [1.0, 2.0, 3.0],
   [4.0, 5.0, 6.0],
   [7.0, 8.0, 9.0]
]

B3 = [
   [10.0, 11.0],
   [12.0, 13.0],
   [14.0, 15.0]
]

print("=== Exemplo 3: Matrizes não-quadradas (3x3 e 3x2) ===")
print_matrix(A3, "A")
print_matrix(B3, "B")

# Multiplicação usando o algoritmo clássico
start_classic = time.time()
C3_classic = multiply_matrices_classic(A3, B3)
end_classic = time.time()
duration_classic = (end_classic - start_classic) * 1000  # ms

if C3_classic:
   print_matrix(C3_classic, "Resultado (Clássico)")
   print(f"Tempo (Clássico): {duration_classic:.6f} ms\n")

# Multiplicação usando o algoritmo de Strassen com padding
start_strassen = time.time()
C3_strassen = strassen(A3, B3, 2)
end_strassen = time.time()
duration_strassen = (end_strassen - start_strassen) * 1000  # ms

if C3_strassen:
   print_matrix(C3_strassen, "Resultado (Strassen)")
   print(f"Tempo (Strassen): {duration_strassen:.6f} ms\n")

# Exemplo 4: Teste de desempenho com matrizes grandes
# Gere matrizes aleatórias de tamanho 2^n x 2^n
n = 7  # 2^7 = 128
size = 2 ** n

print(f"=== Exemplo 4: Teste de desempenho com matrizes {size}x{size} ===")

# Criar matrizes grandes com valores aleatórios
np.random.seed(42)  # Para reprodutibilidade
A4_np = np.random.random((size, size))
B4_np = np.random.random((size, size))

# Converter para listas Python
A4 = A4_np.tolist()
B4 = B4_np.tolist()

print(f"Matriz A: {size}x{size} (valores aleatórios)")
print(f"Matriz B: {size}x{size} (valores aleatórios)\n")

# Multiplicação usando o algoritmo clássico
print("Calculando com o algoritmo clássico...")
start_classic = time.time()
C4_classic = multiply_matrices_classic(A4, B4)
end_classic = time.time()
duration_classic = (end_classic - start_classic) * 1000  # ms

if C4_classic:
   print(f"Resultado (Clássico): {size}x{size} matriz calculada")
   print(f"Tempo (Clássico): {duration_classic:.6f} ms\n")

# Multiplicação usando o algoritmo de Strassen
print("Calculando com o algoritmo de Strassen...")
start_strassen = time.time()
C4_strassen = strassen(A4, B4, 32)  # Threshold = 32 para melhor desempenho
end_strassen = time.time()
duration_strassen = (end_strassen - start_strassen) * 1000  # ms

if C4_strassen:
   print(f"Resultado (Strassen): {size}x{size} matriz calculada")
   print(f"Tempo (Strassen): {duration_strassen:.6f} ms\n")

# Verificar se os resultados são iguais (com uma pequena tolerância)
if C4_classic and C4_strassen:
   results_match = True
   tolerance = 1e-10

   for i in range(size):
       for j in range(size):
           if abs(C4_classic[i][j] - C4_strassen[i][j]) > tolerance:
               results_match = False
               print(f"Resultados diferem na posição [{i}][{j}]: "
                     f"{C4_classic[i][j]} vs {C4_strassen[i][j]}")
               break
       if not results_match:
           break

   if results_match:
       print("Os resultados dos dois algoritmos são idênticos.")

   # Calcular speedup
   speedup = duration_classic / duration_strassen
   print(f"Speedup do Strassen vs. Clássico: {speedup:.2f}x")
```

### Implementação em C++ (Exemplo)

```cpp
```cpp
#include <iostream>
#include <vector>
#include <optional>
#include <format>

/**
 * @typedef Matrix
 * @brief Define uma matriz como um vetor bidimensional de doubles.
 */
using Matrix = std::vector<std::vector<double>>;

/**
 * @brief Verifica se uma matriz é regular (todas as linhas têm o mesmo tamanho).
 * @param mat A matriz a ser verificada.
 * @return true se a matriz for regular, false caso contrário.
 */
bool is_regular(const Matrix& mat) {
    if (mat.empty()) return true;
    
    const size_t cols = mat[0].size();
    return std::all_of(mat.begin(), mat.end(), 
                      [cols](const auto& row) { return row.size() == cols; });
}

/**
 * @brief Obtém as dimensões de uma matriz.
 * @param mat A matriz cuja dimensão será obtida.
 * @return Um par contendo (linhas, colunas). Retorna (0,0) se a matriz for vazia.
 */
std::pair<size_t, size_t> get_dimensions(const Matrix& mat) {
    if (mat.empty()) return {0, 0};
    if (mat[0].empty()) return {mat.size(), 0};
    return {mat.size(), mat[0].size()};
}

/**
 * @brief Imprime uma matriz formatada no console.
 * @param mat A matriz a ser impressa.
 * @param name Nome opcional da matriz para exibição.
 */
void print_matrix(const Matrix& mat, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << " =\n";
    }
    
    for (const auto& row : mat) {
        std::cout << "[ ";
        for (const auto& elem : row) {
            std::cout << std::format("{:8.2f} ", elem);
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Aloca uma matriz com as dimensões especificadas, inicializada com zeros.
 * @param rows Número de linhas da matriz.
 * @param cols Número de colunas da matriz.
 * @return A matriz alocada.
 */
Matrix allocate_matrix(size_t rows, size_t cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

/**
 * @brief Divide uma matriz em quatro submatrizes (quadrantes).
 * @param src A matriz de origem.
 * @param row_start Índice inicial da linha.
 * @param col_start Índice inicial da coluna.
 * @param size Tamanho do quadrante (assumindo quadrantes quadrados).
 * @return A submatriz extraída.
 */
Matrix get_submatrix(const Matrix& src, size_t row_start, size_t col_start, size_t size) {
    Matrix result = allocate_matrix(size, size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result[i][j] = src[row_start + i][col_start + j];
        }
    }
    return result;
}

/**
 * @brief Combina quatro submatrizes em uma matriz maior.
 * @param dest A matriz de destino.
 * @param sub A submatriz a ser inserida.
 * @param row_start Índice inicial da linha na matriz de destino.
 * @param col_start Índice inicial da coluna na matriz de destino.
 */
void set_submatrix(Matrix& dest, const Matrix& sub, size_t row_start, size_t col_start) {
    for (size_t i = 0; i < sub.size(); ++i) {
        for (size_t j = 0; j < sub[0].size(); ++j) {
            dest[row_start + i][col_start + j] = sub[i][j];
        }
    }
}

/**
 * @brief Soma duas matrizes elemento a elemento.
 * @param A Primeira matriz.
 * @param B Segunda matriz.
 * @param subtract Se true, subtrai B de A; se false, soma A e B.
 * @return A matriz resultante da soma ou subtração.
 */
Matrix add_matrices(const Matrix& A, const Matrix& B, bool subtract = false) {
    auto [rows, cols] = get_dimensions(A);
    Matrix result = allocate_matrix(rows, cols);
    double sign = subtract ? -1.0 : 1.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + sign * B[i][j];
        }
    }
    return result;
}

/**
 * @brief Multiplica duas matrizes usando o algoritmo de Strassen.
 * @param A Primeira matriz (n x n, n sendo uma potência de 2).
 * @param B Segunda matriz (n x n, n sendo uma potência de 2).
 * @param min_size Tamanho mínimo da matriz para usar multiplicação clássica.
 * @return Uma std::optional contendo a matriz resultante, ou std::nullopt se as matrizes não forem válidas.
 * @note O algoritmo divide recursivamente as matrizes em quadrantes, computando sete produtos para reduzir a complexidade de O(n^3) para aproximadamente O(n^2.807).
 */
std::optional<Matrix> strassen(const Matrix& A, const Matrix& B, size_t min_size = 64) {
    // Verificações iniciais
    if (!is_regular(A) || !is_regular(B)) {
        std::cerr << "Erro: As matrizes devem ser regulares.\n";
        return std::nullopt;
    }
    
    auto [n, m] = get_dimensions(A);
    auto [p, q] = get_dimensions(B);
    
    if (n == 0 || m == 0 || p == 0 || q == 0) {
        std::cerr << "Erro: As matrizes não podem ser vazias.\n";
        return std::nullopt;
    }
    
    if (n != m || p != q || m != p) {
        std::cerr << "Erro: As matrizes devem ser quadradas e compatíveis.\n";
        return std::nullopt;
    }
    
    // Caso base: usar multiplicação clássica para matrizes pequenas
    if (n <= min_size) {
        return multiply_matrices(A, B); // Assumindo que multiply_matrices está definida
    }
    
    // Dividir as matrizes em quadrantes
    size_t half = n / 2;
    
    Matrix A11 = get_submatrix(A, 0, 0, half);
    Matrix A12 = get_submatrix(A, 0, half, half);
    Matrix A21 = get_submatrix(A, half, 0, half);
    Matrix A22 = get_submatrix(A, half, half, half);
    
    Matrix B11 = get_submatrix(B, 0, 0, half);
    Matrix B12 = get_submatrix(B, 0, half, half);
    Matrix B21 = get_submatrix(B, half, 0, half);
    Matrix B22 = get_submatrix(B, half, half, half);
    
    // Calcular os sete produtos de Strassen
    auto P1 = strassen(add_matrices(A11, A22), add_matrices(B11, B22), min_size);
    auto P2 = strassen(add_matrices(A21, A22), B11, min_size);
    auto P3 = strassen(A11, add_matrices(B12, B22, true), min_size);
    auto P4 = strassen(A22, add_matrices(B21, B11, true), min_size);
    auto P5 = strassen(add_matrices(A11, A12), B22, min_size);
    auto P6 = strassen(add_matrices(A21, A11, true), add_matrices(B11, B12), min_size);
    auto P7 = strassen(add_matrices(A12, A22, true), add_matrices(B21, B22), min_size);
    
    if (!P1 || !P2 || !P3 || !P4 || !P5 || !P6 || !P7) {
        std::cerr << "Erro: Falha ao calcular produtos de Strassen.\n";
        return std::nullopt;
    }
    
    // Calcular os quadrantes da matriz resultante
    Matrix C11 = add_matrices(add_matrices(*P1, *P4), add_matrices(*P7, *P5, true));
    Matrix C12 = add_matrices(*P3, *P5);
    Matrix C21 = add_matrices(*P2, *P4);
    Matrix C22 = add_matrices(add_matrices(*P1, *P3, true), add_matrices(*P2, *P6));
    
    // Combinar os quadrantes em uma única matriz
    Matrix C = allocate_matrix(n, n);
    set_submatrix(C, C11, 0, 0);
    set_submatrix(C, C12, 0, half);
    set_submatrix(C, C21, half, 0);
    set_submatrix(C, C22, half, half);
    
    return C;
}

int main() {
    // Exemplo: Matrizes 4x4
    Matrix A = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };
    
    Matrix B = {
        {17.0, 18.0, 19.0, 20.0},
        {21.0, 22.0, 23.0, 24.0},
        {25.0, 26.0, 27.0, 28.0},
        {29.0, 30.0, 31.0, 32.0}
    };
    
    std::cout << "=== Exemplo: Multiplicação de matrizes 4x4 usando Strassen ===\n";
    print_matrix(A, "A");
    print_matrix(B, "B");
    
    if (auto C = strassen(A, B)) {
        print_matrix(*C, "Resultado C = A x B (Strassen)");
    } else {
        std::cout << "Não foi possível multiplicar as matrizes A e B.\n";
    }
    
    return 0;
}
```
```

### Análise de Complexidade

A relação de recorrência para a complexidade $T(n)$ é:

$$T(n) = 7 T(n/2) + O(n^2)$$

Na qual:

* $7 T(n/2)$ representa as $7$ chamadas recursivas em subproblemas de tamanho $n/2$
* $O(n^2)$ representa o custo das $18$ adições e subtrações de matrizes $(n/2) \times (n/2)$

Pelo Teorema Mestre, esta recorrência resolve para:

$$T(n) = O(n^{\log_2 7}) \approx O(n^{2.8074})$$

Esta complexidade é assintoticamente mais eficiente que o algoritmo clássico $O(n^3)$, especialmente para matrizes de grande dimensão.

>**O Teorema Mestre**
>
>O Teorema Mestre é uma ferramenta matemática fundamental na análise de algoritmos recursivos que seguem o paradigma de divisão e conquista. Este teorema fornece um método direto para resolver recorrências da forma:
>
>$$T(n) = a \cdot T\left(\frac{n}{b}\right) + f(n)$$
>
>no qual, temos:
>
>* $T(n)$ é a função de complexidade para um problema de tamanho $n$
>* $a$ é o número de subproblemas recursivos
>* $b$ é o fator de redução do tamanho do problema em cada chamada recursiva
>* $f(n)$ é o custo do trabalho realizado fora das chamadas recursivas (tipicamente para dividir o problema e combinar as soluções)
>
>O teorema fornece três casos distintos, dependendo da relação entre a taxa de crescimento de $f(n)$ e $n^{\log_b a}$:
>
>**Caso 1**: $f(n) = O(n^{\log_b a - \epsilon})$ para algum $\epsilon > 0$
>Quando o custo do trabalho fora da recursão cresce mais lentamente que o custo das chamadas recursivas:
>
>$$T(n) = \Theta(n^{\log_b a})$$
>
>**Caso 2**: $f(n) = \Theta(n^{\log_b a} \cdot \log^k n)$ para algum $k \geq 0$
>Quando o custo do trabalho fora da recursão é comparável ao custo das chamadas recursivas:
>
>$$T(n) = \Theta(n^{\log_b a} \cdot \log^{k+1} n)$$
>
>**Caso 3**: $f(n) = \Omega(n^{\log_b a + \epsilon})$ para algum $\epsilon > 0$, e $a \cdot f(n/b) \leq c \cdot f(n)$ para algum $c < 1$ e $n$ suficientemente grande
>Quando o custo do trabalho fora da recursão domina o custo das chamadas recursivas:
>
>$$T(n) = \Theta(f(n))$$
>
>No caso do algoritmo de Strassen, a recorrência é:
>
>$$T(n) = 7 \cdot T\left(\frac{n}{2}\right) + O(n^2)$$
>
>Neste caso, teremos:
>
>* $a = 7$ (sete chamadas recursivas)
>* $b = 2$ (cada subproblema tem metade do tamanho)
>* $f(n) = O(n^2)$ (custo das operações de adição e subtração de matrizes)
>
>Calculando $n^{\log_b a} = n^{\log_2 7} \approx n^{2.807}$
>
>Como $f(n) = O(n^2)$ e $n^2 = O(n^{\log_2 7 - \epsilon})$ para $\epsilon \approx 0.807$, estamos no Caso 1 do Teorema Mestre.
>
>Portanto, a complexidade do algoritmo de Strassen é:
>
>$$T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$$
>

### Considerações Práticas e Otimizações

Embora o algoritmo de Strassen seja teoricamente mais eficiente que o algoritmo clássico, na prática ele apresenta algumas limitações:

1. **Tamanho Mínimo Eficiente**: Para matrizes pequenas, o algoritmo clássico é geralmente mais rápido devido à menor constante oculta na notação Big-O. É comum implementar um threshold abaixo do qual o algoritmo clássico é usado.

2. **Estabilidade Numérica**: O algoritmo de Strassen pode apresentar menor estabilidade numérica devido ao maior número de operações aritméticas, que podem propagar erros de arredondamento.

3. **Implementação e Manutenção**: O código é significativamente mais complexo que o algoritmo clássico, o que dificulta manutenção e otimização.

A implementação do algoritmo de Strassen pode ser otimizada para matrizes cujas dimensões não são potências de $2$. Algumas abordagens incluem:

* **Padding**: preencher a matriz com zeros até a próxima potência de 2. Simples, mas pode adicionar overhead significativo.

* **Subdivisão Desigual**: dividir a matriz em blocos de tamanhos diferentes, acomodando dimensões ímpares. Mais eficiente, mas complexo de implementar.

* **Abordagem Híbrida**: usar Strassen para as partes principais da matriz e algoritmo clássico para as bordas, quando as dimensões não são divisíveis por 2.

Finalmente, o algoritmo de Strassen é particularmente adequado para implementações paralelas. Que pode ser entendido se a atenta leitora considerar que os $7$ produtos, $P_1$ a $P_7$, podem ser calculados independentemente e em paralelo. Além disso, é preciso considerar também que as adições e subtrações de matrizes também são facilmente paralelizáveis.

A atenta leitora não deve esquecer que em sistemas multi-core ou distribuídos, esta característica pode fornecer aceleração adicional significativa e, eventualmente, redução dos custos computacionais.

## Algoritmo de Coppersmith-Winograd (1987) e Seus Sucessores

O algoritmo de Coppersmith-Winograd, desenvolvido por [Don Coppersmith](https://en.wikipedia.org/wiki/Don_Coppersmith) e [Shmuel Winograd](https://en.wikipedia.org/wiki/Shmuel_Winograd) e publicado em 1987, representou um marco teórico importante na busca pelo expoente ótimo $\omega$ da multiplicação de matrizes. Este expoente diz respeito a complexidade do algoritmo.

O algoritmo de Coppersmith-Winograd como resultado parcial de uma linha de pesquisa iniciada após o algoritmo de Strassen (1969), em que matemáticos e cientistas da computação buscavam reduzir o limite superior teórico do expoente de complexidade dos algoritmos para a multiplicação de matrizes.

A importância histórica deste algoritmo está no fato de ter reduzido o expoente de $O(n^{\log_2 7}) \approx O(n^{2.8074})$ de Strassen para $O(n^{2.376})$, aproximando-se do limite teórico inferior de $\omega = 2$, que muitos pesquisadores acreditam ser o expoente ótimo. O mundo está cheio de gente otimista.

>São os sonhos que, todos os dias, levam homens aos céus. (?? ouvi isso, não sei onde)

O algoritmo de Coppersmith-Winograd difere fundamentalmente da abordagem relativamente intuitiva de Strassen. Em vez de simplesmente dividir a matriz em blocos $2 \times 2$ com uma estratégia fixa de recombinação, este algoritmo baseia-se em construções algébricas complexas relacionadas a Teoria da Complexidade Algébrica como rank tensorial e multiplicação bilinear.

> **Operações Bilineares na Multiplicação de Matrizes**: _as operações bilineares são funções matemáticas que são lineares em cada uma de suas variáveis quando a outra é mantida constante_. No contexto da multiplicação de matrizes e da álgebra tensorial, isso tem implicações importantes.
>
> Uma operação bilinear $f: U \times V \rightarrow W$ entre espaços vetoriais possui as seguintes propriedades:
>
> 1. Para um $u \in U$ fixo, a função $f(u, \cdot): V \rightarrow W$ é linear.
> 2. Para um $v \in V$ fixo, a função $f(\cdot, v): U \rightarrow W$ é linear.
>
> No caso específico da multiplicação de matrizes, a operação que mapeia o par $(A, B)$ para o produto $C = AB$ é bilinear porque:
>
> - Se fixarmos $A$ e dobrarmos $B$, o resultado $AB$ também dobra
> - Se fixarmos $B$ e dobrarmos $A$, o resultado $AB$ também dobra
> - A operação distribui sobre a adição: $A(B + C) = AB + AC$ e $(A + B)C = AC + BC$
>
>> Essa bilinearidade é a base para muitos algoritmos de multiplicação de matrizes, incluindo o algoritmo de Strassen e o de Coppersmith-Winograd. A ideia é decompor a multiplicação em operações mais simples que podem ser computadas separadamente e depois combinadas.
>
> Matematicamente, podemos expressar isso como:
>
> $$f(A, B) = \sum_{r=1}^R L_r(A) \cdot M_r(B)$$
>
> Neste caso, teremos: $L_r$ e $M_r$ são formas lineares e $R$ é o rank da decomposição.

> **Ranks Tensoriais e Multiplicação Bilinear**: a multiplicação de matrizes pode ser vista como uma operação bilinear tensorial. Para matrizes $A \in \mathbb{R}^{m \times n}$ e $B \in \mathbb{R}^{n \times p}$, o produto $C = A \times B$ representa uma forma bilinear.
>
> **Rank Tensorial:** O rank tensorial $R$ de uma operação bilinear é o número mínimo de multiplicações escalares necessárias para calcular a operação. Formalmente, para a multiplicação de matrizes, buscamos a menor decomposição:
>
> $$C_{ij} = \sum_{k=1}^n A_{ik}B_{kj} = \sum_{r=1}^R \left(\sum_{i} \alpha_{ir}A_{ik}\right)\left(\sum_{j} \beta_{jr}B_{kj}\right)$$
>
> O algoritmo de Coppersmith-Winograd explora estruturas tensoriais específicas para alcançar limites assintóticos inferiores para $R$, reduzindo a complexidade da multiplicação de matrizes $n \times n$ para $O(n^{\omega})$, onde $\omega < 2.376$.
>
> **Multiplicação Bilinear:** Em termos algorítmicos, podemos representar essa abordagem como uma decomposição da forma:
>
> $$\langle A, B \rangle = \sum_{r=1}^R \langle u_r, A \rangle \cdot \langle v_r, B \rangle$$
>
> onde $u_r$ e $v_r$ são tensores de ordem apropriada.
>
> O algoritmo de Strassen, que a atenta leitora viu antes, demonstra que o rank tensorial da multiplicação de matrizes $2 \times 2$ é $7$, não $8$ como na abordagem ingênua, estabelecendo assim $\omega < \log_2 7 \approx 2.807$.

A ideia central do algoritmo de Coppersmith-Winograd envolve construir métodos para multiplicar matrizes de um tamanho base $k \times k$ usando um número $m$ de multiplicações escalares significativamente menor que $k^3$. Quando tal método é aplicado recursivamente, obtém-se uma complexidade de $O(n^{\log_k m})$.

O algoritmo de Coppersmith-Winograd conseguiu, através de construções baseadas em propriedades de polinômios e corpos finitos, demonstrar a existência de um método que leva à complexidade $O(n^{2.376})$ [^1][^2].

[^1]: COPPERSMITH, D.; WINOGRAD, S. **Matrix multiplication via arithmetic progressions**. *In*: ANNUAL ACM SYMPOSIUM ON THEORY OF COMPUTING, 19., 1987, New York. **Proceedings**. New York: ACM, 1987. p. 1-6.

[^2]:COPPERSMITH, D.; WINOGRAD, S.. **Matrix multiplication via arithmetic progressions**. **Journal of Symbolic Computation**, v. 9, n. 3, p. 251-280, 1990.

### Representação Conceitual

Em vez de um exemplo numérico detalhado, que seria impraticável devido à complexidade intrínseca destes algoritmos, podemos pensar conceitualmente:

Enquanto o algoritmo de Strassen segue a recorrência:

$$T(n) = 7 \cdot T(n/2) + O(n^2)$$

Um algoritmo como Coppersmith-Winograd pode ser conceitualizado, de forma simplificada, como:

$$T(n) = m \cdot T(n/k) + O(n^2)$$

Neste caso, teremos:

- $k$ é um tamanho de bloco base (potencialmente grande);
- $m$ é o número de multiplicações (menor que $k^3$) necessárias para esse bloco;
- $\omega = \log_k m$ resulta no expoente desejado ($2.376$ obtido em [^2]).

A construção específica de como realizar a multiplicação do bloco $k \times k$ com $m$ operações é a parte complexa e teoricamente sofisticada destes algoritmos.

## Sucessores e Melhorias Incrementais

Após o trabalho de Coppersmith-Winograd, vários pesquisadores conseguiram melhorias incrementais no expoente $\omega$:

| Ano | Pesquisadores | Expoente $\omega$ |
|-----|--------------|-----------------|
| 1969 | Strassen | 2.8074 |
| 1987 | Coppersmith-Winograd | 2.376 |
| 2010 | Stothers | 2.374 |
| 2012 | Williams | 2.3729 |
| 2014 | Le Gall | 2.3728 |
| 2020-2022 | Alman, Duan, Wu, Zhou | ~2.37 |

Estas melhorias envolvem refinamentos do método original, frequentemente baseados no chamado "método laser" e suas variações, aplicado a potências do tensor de multiplicação de matrizes.

## Limitações Práticas

Apesar do avanço teórico impressionante, os algoritmos de Coppersmith-Winograd e seus sucessores têm limitações críticas para aplicações práticas:

1. **Constantes Ocultas Enormes**: Embora a complexidade assintótica seja melhor, as constantes ocultas na notação O são enormes, tornando estes algoritmos mais lentos que métodos mais simples para todos os tamanhos de matriz encontrados na prática.

2. **Relevância Teórica vs. Prática**: Estima-se que estes algoritmos só superariam métodos mais simples para matrizes de dimensões astronomicamente grandes, muito além da capacidade de armazenamento e processamento de qualquer computador atual ou previsível.

3. **Complexidade de Implementação**: A implementação destes algoritmos envolveria estruturas algébricas avançadas que transcendem o escopo da programação convencional.

4. **Estabilidade Numérica**: Questões de estabilidade numérica, já presentes em Strassen, são potencialmente mais severas nesses algoritmos mais complexos.

## Referências

ALMAN, J.; WILLIAMS, V. V. **Further limitations of the known approaches for matrix multiplication**. In: *Proceedings of the 9th Innovations in Theoretical Computer Science Conference (ITCS'18)*, p. 25:1–25:15, 2018.  
**Disponível em:** <https://people.csail.mit.edu/virgi/matmultlimits.pdf>.

ALMAN, J.; WILLIAMS, V. V.** Limits on all known (and some unknown) approaches to matrix multiplication**. In: *Proceedings of the 59th Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, p. 580–591, 2018.  
**Disponível em:** <https://cims.nyu.edu/~regev/toc/articles/v017a001/v017a001.pdf>.

BINI, D. A. et al. **O(n²·⁷⁷⁹⁹) complexity for n×n approximate matrix multiplication**. *Information Processing Letters*, v. 8, n. 5, p. 234–235, 1979.  
**Disponível em:** <https://www.cs.toronto.edu/~yuvalf/Limitations.pdf>.

D'ALBERTO, P.; NICOLAU, A. **Adaptive Strassen's matrix multiplication**. In: *Proceedings of the 21st Annual International Conference on Supercomputing*, p. 284–292, 2007.  
**Disponível em:** <https://ics.uci.edu/~paolo/Reference/dalberto-nicolau.winograd.TOMS.pdf>.

WILLIAMS, V. V. Lecture 23: **Border Rank and Fast Matrix Multiplication**. *6.890: Matrix Multiplication and Graph Algorithms*, Massachusetts Institute of Technology, 2021.  
**Disponível em:** <https://people.csail.mit.edu/virgi/6.890/lecture23.pdf>.

### Para Pesquisa Posterior

A lista de todos os trabalhos usados como referência para os cinco trabalhos que usei para este artigo. Estou colocando aqui para facilitar a pesquisa posterior. Para quando eu precisar, ou quiser, me aprofundar mais.

1. AHO, A. V.; HOPCROFT, J. E.; ULLMAN, J. *The design and analysis of computer algorithms*. Addison-Wesley Longman Publishing Co., Boston, MA, 1974.
2. ALMAN, J.; WILLIAMS, V. V. Further limitations of the known approaches for matrix multiplication. In: *46th Annual Symposium on Foundations of Computer Science (FOCS 2005)*, 2005.
3. ALMAN, J.; WILLIAMS, V. V. Further limitations of the known approaches for matrix multiplication. In: *Proc. of ITCS*, p. 25:1-25:15, 2018.
4. ALMAN, J.; WILLIAMS, V. V. Limits on all known (and some unknown) approaches to matrix multiplication. In: *Proc. 59th FOCS*, p. 580-591, 2018.
5. ALMAN, J.; WILLIAMS, V. V. A refined laser method and faster matrix multiplication. In: *Proc. 32nd Ann. ACM-SIAM Symp. on Discrete Algorithms (SODA'21)*, p. 522-539, 2021.
6. ALON, N.; SHPILKA, A.; UMANS, C. On sunflowers and matrix multiplication. *Comput. Complexity*, v. 22, n. 2, p. 219-243, 2013.
7. AMBAINIS, A.; FILMUS, Y.; LE GALL, F. Fast matrix multiplication: limitations of the Coppersmith-Winograd method. In: *Proc. 47th STOC*, p. 585-593, 2015.
8. ANDERSON, E. et al. *LAPACK User' Guide, Release 2.0*. 2 ed. SIAM, 1995.
9. BEHREND, F. A. On sets of integers which contain no three terms in arithmetic progression. *Proc. Nat. Acad. Sci.*, p. 331-332, 1946.
10. BILARDI, G.; D'ALBERTO, P.; NICOLAU, A. Fractal matrix multiplication: a case study on portability of cache performance. In: *Workshop on Algorithm Engineering 2001*. Aarhus, Denmark, 2001.
11. BILMES, J. et al. Optimizing matrix multiply using PHiPAC: a portable, high-performance, Ansi C coding methodology. In: *Proceedings of the annual International Conference on Supercomputing*, 1997.
12. BINI, D. A.; CAPOVANI, M.; ROMANI, F.; LOTTI, G. O(n2.7799) complexity for n×n approximate matrix multiplication. *Information Processing Letters*, v. 8, p. 234-235, 1979.
13. BLACKFORD, L. S. et al. An updated set of basic linear algebra subprograms (BLAS). *ACM Transactions on Mathematical Software*, v. 28, n. 2, p. 135-151, 2002.
14. BLASIAK, J. et al. On cap sets and the group-theoretic approach to matrix multiplication. *Discrete Analysis*, v. 2017, n. 3, p. 1-27, 2017.
15. BLÄSER, M. *Fast Matrix Multiplication*. Number 5 in Graduate Surveys. Theory of Computing Library, 2013.
16. BRENT, R. P. Algorithms for matrix multiplication. Tech. Rep. TR-CS-70-157, Stanford University, Mar. 1970.
17. BÜRGISSER, P.; CLAUSEN, M.; SHOKROLLAHI, M. A. *Algebraic Complexity Theory*. Springer, 1997.
18. CHATTERJEE, S. et al. Recursive array layouts and fast matrix multiplication. *IEEE Transactions on Parallel Distributed Systems*, v. 13, n. 11, p. 1105-1123, 2002.
19. COHN, H.; KLEINBERG, R.; SZEGEDY, B.; UMANS, C. Group-theoretic algorithms for matrix multiplication. In: *46th Annual Symposium on Foundations of Computer Science (FOCS 2005)*, 2005.
20. COPPERSMITH, D.; WINOGRAD, S. Matrix multiplication via arithmetic progressions. *Journal of Symbolic Computation*, v. 9, p. 251-280, March 1990.
21. D'ALBERTO, P.; NICOLAU, A. Adaptive Strassen and ATLAS's DGEMM: A fast square-matrix multiply for modern high-performance systems. In: *The 8th International Conference on High Performance Computing in Asia Pacific Region (HPC asia)*. Beijing, p. 45-52, 2005.
22. DAVIE, A. M.; STOTHERS, A. J. Improved bound for complexity of matrix multiplication. *Proceedings of the Royal Society of Edinburgh*, v. 143A, p. 351-370, 2013.
23. DEMMEL, J. et al. Self-Adapting linear algebra algorithms and software. *Proceedings of the IEEE, special issue on "Program Generation, Optimization, and Adaptation"*, v. 93, n. 2, 2005.
24. DONGARRA, J. J. et al. A set of level 3 Basic Linear Algebra Subprograms. *ACM Transactions on Mathematical Software*, v. 16, p. 1-17, 1990.
25. DOUGLAS, C. et al. GEMMW: A portable level 3 BLAS Winograd variant of Strassen's matrix-matrix multiply algorithm. *J. Comp. Phys.*, v. 110, p. 1-10, 1994.
26. FRENS, J.; WISE, D. Auto-Blocking matrix-multiplication or tracking BLAS3 performance from source code. *Proc. 1997 ACM Symp. on Principles and Practice of Parallel Programming*, v. 32, n. 7 (Jul.), p. 206-216, 1997.
27. GOTO, K.; VAN DE GEIJN, R. Anatomy of high-performance matrix multiplication. *ACM Transactions on Mathematical Software*, v. 34, n. 3, p. 1-25, 2008.
28. GUNNELS, J. et al. FLAME: Formal Linear Algebra Methods Environment. *ACM Transactions on Mathematical Software*, v. 27, n. 4 (Dec.), p. 422-455, 2001.
29. HÅSTAD, J. Tensor rank is NP-complete. *J. Algorithms*, v. 11, n. 4, p. 644-654, December 1990.
30. HIGHAM, N. *Accuracy and Stability of Numerical Algorithms*. Second Edition. SIAM, 2002.
31. KAGSTROM, B. et al. GEMM-based level 3 BLAS: high-performance model implementations and performance evaluation benchmark. *ACM Transactions on Mathematical Software*, v. 24, n. 3 (Sept), p. 268-302, 1998.
32. KAPORIN, I. A practical algorithm for faster matrix multiplication. *Numerical Linear Algebra with Applications*, v. 6, n. 8, p. 687-700, 1999.
33. LAWSON, C. L. et al. Basic Linear Algebra Subprograms for FORTRAN usage. *ACM Transactions on Mathematical Software*, v. 5, p. 308-323, 1979.
34. LE GALL, F. Powers of tensors and fast matrix multiplication. In: *39th International Symposium on Symbolic and Algebraic Computation (ISSAC 2014)*, 2014. arXiv:1401.771.
35. PAN, V. How can we speed up matrix multiplication? *SIAM Review*, v. 26, n. 3, p. 393-415, 1984.
36. SCHÖNHAGE, A. Partial and total matrix multiplication. *SIAM J. Comput.*, v. 10, n. 3, p. 434-455, 1981.
37. STRASSEN, V. Gaussian elimination is not optimal. *Numerische Mathematik*, v. 13, n. 4, p. 354-356, 1969.
38. STRASSEN, V. The asymptotic spectrum of tensors and the exponent of matrix multiplication. In: *FOCS*, p. 49-54, 1986.
39. WHALEY, R. C.; PETITET, A. Minimizing development and maintenance costs in supporting persistently optimized BLAS. *Software: Practice and Experience*, v. 35, n. 2 (Feb.), p. 101-121, 2005.
40. WILLIAMS, V. V. Multiplying matrices faster than Coppersmith-Winograd. In: *Proc. 44th STOC*, p. 887-898, 2012.
