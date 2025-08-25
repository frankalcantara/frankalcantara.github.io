---
layout: post
title: Tabelas de Derivação (Análise) LL(1)
author: Frank
categories:
    - Matemática
    - Linguagens Formais
tags:
    - Matemática
    - Linguagens Formais
    - Compiladores
image: assets/images/deriva.webp
description: Como criar tabelas de análise para parsers ll(1) usando os conjuntos First e Follow.
slug: tabela-derivacao
keywords:
    - parsers
    - first
    - follow
    - derivation
    - derivação
rating: 5
published: 2024-06-21T19:41:42.444Z
draft: 2024-06-21T19:41:40.386Z
preview: Definição, processo de funcionamento e outras informações necessárias ao entendimento do funcionamento de parsers LL(1)
lastmod: 2025-08-25T00:18:39.010Z
---

**ESTE ARQUIVO FOI CORRIGIDO, E SUBSTITUÍDO PELO LIVRO [LINGUAGENS FORMAIS](https://frankalcantara.com/lf/index.html)**

A Tabela de Derivação $LL(1)$, também chamada de tabela da análise LL(1), é uma ferramenta fundamental do funcionamento do *parser* $LL(1)$. Esta tabela será utilizada no processo de *parser* para verificar se um determinado *string* está de acordo com a gramática da linguagem. Esta tabela será usada pelo algoritmo de *parser* para determinar qual regra de produção deverá ser aplicada, considerando o símbolo de entrada corrente e o não-terminal no topo de uma pilha. Este par de símbolos, (terminal / não-terminal) será usado como índice da tabela e determinará qual regra de produção deverá ser utilizada, ou se existe uma inconsistência entre a *string* de entrada e a gramática. Já vimos [o que é um parser](https://frankalcantara.com/parsers-ll(1)/), e como criar [os conjuntos $FIRST$ e $FOLLOW$](https://frankalcantara.com/first-follow/). Agora vamos usar este conhecimento para criar a Tabela de Derivação.

Para construir a Tabela de Derivação $LL(1)$, acrescentaremos um $\$$ no final da *string* de entrada para indicar o seu término e, além disso, seguiremos três regras:

- Para cada terminal $a$ em $FIRST(\alpha)$, adicione a regra $A \to \alpha$ à célula $[A, a]$ da tabela.
- Se $\varepsilon$ está em $FIRST(\alpha)$, adicione a regra $A \to \alpha$ à tabela em $[A, b]$ para cada $b$ em $FOLLOW(A)$.
- Se $ \\$ $ está em $FOLLOW(A)$, adicione também $A \to \alpha$ à célula $[A, \\$]$.

Podemos retornar ao um [exemplo 1 de criação do conjunto $FIRST$](https://frankalcantara.com/first-follow/) e, a partir deste exemplo, criar a Tabela de Derivação correspondente a gramática dada naquele exemplo.

**Exemplo 1**: Considere a gramática definida pelo seguinte conjunto de regras de produção

$$
\begin{array}{cc}
1. & S \rightarrow aB \mid bA \\
2. & A \rightarrow c \mid d \\
3. & B \rightarrow e \mid f \\
\end{array}
$$

A partir deste conjunto de regras de produção podemos definir o seguinte conjunto $FIRST$:

$$
\begin{array}{ccl}
FIRST(S) & = & \{a, b\} \\
FIRST(A) & = & \{c, d\} \\
FIRST(B) & = & \{e, f\} \\
\end{array}
$$

E o conjunto $FOLLOW$ dado por:

$$
\begin{array}{ccl}
FOLLOW(S) & = & \{\$\} \\
FOLLOW(A) & = & \{\$, c, d\} \\
FOLLOW(B) & = & \{\$\} \\
\end{array}
$$

Com estes conjuntos podemos criar uma Tabela de Derivação se seguirmos as regras dadas acima teremos:

- Para $S \to aB$: Como $a$ está em $FIRST(aB)$, adicionamos $S \to aB$ em $[S, a]$.
- Para $S \to bA$: Como $b$ está em $FIRST(bA)$, adicionamos $S \to bA$ em $[S, b]$.
- Para $A \to c$: Como $c$ está em $FIRST(c)$, adicionamos $A \to c$ em $[A, c]$.
- Para $A \to d$: Como $d$ está em $FIRST(d)$, adicionamos $A \to d$ em $[A, d]$.
- Para $B \to e$: Como $e$ está em $FIRST(e)$, adicionamos $B \to e$ em $[B, e]$.
- Para $B \to f$: Como $f$ está em $FIRST(f)$, adicionamos $B \to f$ em $[B, f]$.

O que permite criar a seguinte Tabela de Derivação:

| não-terminal | a           | b           | c        | d        | e        | f        | $   |
|--------------|-------------|-------------|----------|----------|----------|----------|-----|
| S            | $S \to aB$  | $S \to bA$  |          |          |          |          |     |
| A            |             |             | $A \to c$| $A \to d$|          |          |     |
| B            |             |             |          |          | $B \to e$| $B \to f$|     |

Este exemplo é perfeito, para cada par terminal / não-terminal existe apenas uma regra de produção. Infelizmente, quando estamos construindo linguagens livres de contexto, este não é o cenário mais comum.

