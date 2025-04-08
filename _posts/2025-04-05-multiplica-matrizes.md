---
layout: post
title: Estudando Multiplicação de Matrizes
author: frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - C++
    - Python
    - Matemática
    - inteligência artificial
image: assets/images/transformers1.webp
featured: false
rating: 5
description: Análise dos algoritmos de multiplicação de matrizes.
date: 2025-02-09T22:55:34.524Z
preview: Uma introdução a matemática que suporta a criação de transformers para processamento de linguagem natural com exemplos de código em C++20.
keywords:
    - transformers
    - matemática
    - processamento de linguagem natural
    - C++
    - aprendizado de máquina
    - vetores
    - produto escalar
    - álgebra linear
    - embeddings
    - atenção
    - deep learning
    - inteligência artificial
toc: true
published: false
beforetoc: ""
lastmod: 2025-04-08T14:00:45.541Z
---

A multiplicação de matrizes pode, sem dúvida, ser um dos tópicos mais importantes dos modelos de linguagem, e aprendizagem de máquina, disponíveis no mercado atualmente. Neste artigo, vamos explorar alguns algoritmos para multiplicação de matrizes, suas aplicações e como ele se relaciona com o funcionamento de modelos de aprendizado profundo, como os Transformers, que estamos estudando ([aqui](https://frankalcantara.com/voce-pensa-como-fala/),[aqui](https://frankalcantara.com/transformers-um/) e [aqui](https://frankalcantara.com/transformers-dois/)). 

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

### Complexidade Computacional

A complexidade computacional deste algoritmo é determinada pelo número de multiplicações e adições realizadas:
- Para cada um dos $m \times n$ elementos de $C$, realizamos $p$ multiplicações e $p-1$ adições
- O número total de multiplicações é $m \times n \times p$
- Se as matrizes forem quadradas de dimensão $n \times n$ ($m=p=n$), a complexidade é $O(n^3)$

Esta complexidade cúbica para matrizes quadradas motivou a busca por algoritmos mais eficientes, como o algoritmo de Strassen (complexidade $O(n^{2.807})$) e outros algoritmos assintoticamente mais rápidos, que se tornam vantajosos para matrizes de grandes dimensões.

### Ilustrações

As seguintes ilustrações ajudam a visualizar o processo de multiplicação:

1. **Diagrama de Cálculo de Elemento:** A figura abaixo mostra como o elemento $C_{11}$ é calculado a partir da primeira linha de $A$ e primeira coluna de $B$.

2. **Visualização dos Loops Aninhados:** Esta ilustração representa os três loops aninhados do algoritmo e a operação realizada na parte mais interna.

### Exemplos Numéricos

**Exemplo 1: Matrizes Quadradas 2x2**

Sejam as matrizes $A$ e $B$:
$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}
$$
O produto $C = A \times B$ será uma matriz $2 \times 2$:
$$
C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$
Calculando cada elemento:
* $C_{11} = \sum_{k=1}^{2} A_{1k} B_{k1} = A_{11}B_{11} + A_{12}B_{21} = (1)(5) + (2)(7) = 5 + 14 = 19$
* $C_{12} = \sum_{k=1}^{2} A_{1k} B_{k2} = A_{11}B_{12} + A_{12}B_{22} = (1)(6) + (2)(8) = 6 + 16 = 22$
* $C_{21} = \sum_{k=1}^{2} A_{2k} B_{k1} = A_{21}B_{11} + A_{22}B_{21} = (3)(5) + (4)(7) = 15 + 28 = 43$
* $C_{22} = \sum_{k=1}^{2} A_{2k} B_{k2} = A_{21}B_{12} + A_{22}B_{22} = (3)(6) + (4)(8) = 18 + 32 = 50$

Portanto, a matriz resultante é:
$$
C = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}
$$

**Exemplo 2: Matrizes Retangulares (3x2 e 2x3)**

Sejam as matrizes $A$ (dimensão $3 \times 2$) e $B$ (dimensão $2 \times 3$):
$$
A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \quad B = \begin{pmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{pmatrix}
$$
O produto $C = A \times B$ será uma matriz $3 \times 3$:
$$
C = \begin{pmatrix} C_{11} & C_{12} & C_{13} \\ C_{21} & C_{22} & C_{23} \\ C_{31} & C_{32} & C_{33} \end{pmatrix}
$$
Calculando cada elemento:
* $C_{11} = (1)(7) + (2)(10) = 7 + 20 = 27$
* $C_{12} = (1)(8) + (2)(11) = 8 + 22 = 30$
* $C_{13} = (1)(9) + (2)(12) = 9 + 24 = 33$
* $C_{21} = (3)(7) + (4)(10) = 21 + 40 = 61$
* $C_{22} = (3)(8) + (4)(11) = 24 + 44 = 68$
* $C_{23} = (3)(9) + (4)(12) = 27 + 48 = 75$
* $C_{31} = (5)(7) + (6)(10) = 35 + 60 = 95$
* $C_{32} = (5)(8) + (6)(11) = 40 + 66 = 106$
* $C_{33} = (5)(9) + (6)(12) = 45 + 72 = 117$

Portanto, a matriz resultante é:
$$
C = \begin{pmatrix} 27 & 30 & 33 \\ 61 & 68 & 75 \\ 95 & 106 & 117 \end{pmatrix}
$$

### Implementação em Python

Abaixo está uma implementação em Python do algoritmo clássico de multiplicação de matrizes:

<code>
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
</code>