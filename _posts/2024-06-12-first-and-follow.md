---
layout: post
title: First & Follow
author: Frank
categories:
    - Matemática
    - Linguagens Formais
tags:
    - Matemática
    - Linguagens Formais
    - Compiladores
image: assets/images/first.webp
description: Como construir os conjuntos First  e Follow que irão permitir a criação da tabela de análise para o algoritmo de parser.
slug: first-follow
keywords:
    - parsers
    - first
    - follow
    - derivation
    - derivação
rating: 5
published: 2024-06-12T13:16:53.564Z
draft: 2024-06-12T13:00:39.229Z
preview: Como construir os conjuntos First  e Follow que irão permitir a criação da tabela de análise para o algoritmo de parser.
lastmod: 2025-08-25T00:18:28.801Z
---

**ESTE ARQUIVO FOI CORRIGIDO, E SUBSTITUÍDO PELO LIVRO [LINGUAGENS FORMAIS](https://frankalcantara.com/lf/index.html)**

Não dá nem para começar a pensar em criar um *parser* $LL(1)$ se não entender os conjuntos $FIRST$ e $FOLLOW$. [Também não dá para entender estes conjuntos se não souber o que é um *parser* $LL(1)$](https://frankalcantara.com/parsers-ll(1)/) Imagine que você está aprendendo um novo idioma. Para formar frases corretas, você precisará entender quais palavras podem vir antes ou depois de outras. Ou corre o risco de falar como o Yoda. Se quiser evitar ser confundido com um velho alienígena, precisa aprender, no mínimo, a ordem das palavras, muito antes de entender a classe gramatical destas mesmas palavras. Como uma criança aprendendo a falar.

Eu forcei um pouco a barra na metáfora, mas na análise sintática de linguagens livres de contexto, os conjuntos $FIRST$ e $FOLLOW$ desempenham um papel importante que quase valida minha metáfora. Estes conjuntos ajudam a decifrar a gramática da linguagem de forma determinística determinando as regras de produção que serão  aplicadas aos símbolos da *string* de entrada para garantir que ele faça parte da linguagem.

O conjunto $FIRST$ de um símbolo não-terminal será composto dos símbolos terminais que podem aparecer como **primeiro símbolo** de qualquer sequência de símbolos que seja derivada desse não-terminal. Em outras palavras, o conjunto $FIRST$ indica quais terminais podem iniciar uma declaração válida (frase) dentro da estrutura gramática definida por um não-terminal. Por exemplo, considere uma gramática para definir expressões aritméticas. O não-terminal *EXPR* pode derivar diversas sequências de símbolos, como *2 + 3*, *(4 * 5)*, *x - y*. O conjunto $FIRST$ do não-terminal *EXPR* seria, neste caso específico, ${número, '+', '-', '('}$, pois esses são os símbolos que podem iniciar qualquer expressão aritmética válida nesta gramática até onde podemos saber com as informações passadas neste parágrafo. Uma gramática para todas as expressões aritméticas possíveis teria um conjunto $FIRST$ maior.

O conjunto $FOLLOW$, por sua vez, determina o conjunto de símbolos terminais que podem aparecer **imediatamente após** um não-terminal em alguma derivação da gramática. Ou colocando de outra forma, o conjunto $FOLLOW$ indica quais terminais podem seguir (*follow*) um não-terminal em uma declaração válida da linguagem.

Diferentemente do $FIRST$, que se concentra no início de uma derivação, o $FOLLOW$ analisa a situação em que um não-terminal aparece, considerando as produções diretas do não-terminal e também as produções de outros não terminais que, por ventura, contenham o não-terminal em análise. Por exemplo, considere uma gramática que define declarações de variáveis. O não-terminal *DECLARACAO_VAR* pode ser seguido por diferentes símbolos, dependendo do contexto. Em uma linguagem como o $C$, uma declaração de variável pode terminar com um ponto e vírgula, ser seguida por um operador de atribuição e uma expressão, ou até mesmo ser parte de uma estrutura maior. Neste cenário, o conjunto $FOLLOW$ do não-terminal *DECLARACAO_VAR* incluiria, portanto, o ponto e vírgula ';', o sinal de igual '=', e todos os outros símbolos que podem iniciar uma expressão ou um comando que a linguagem permita ocorrer na mesma linha da declaração da variável.

Os conjuntos $FIRST$ e $FOLLOW$ serão utilizados para construir a Tabela de Derivação $LL(1)$. A forma tecnicamente mais correta seria dizer que estes conjuntos formam a Tabela De Análise $LL(1)$. Entretanto, pobre de mim, prefiro chamar de Tabela de Derivação.

As Tabelas de Derivação são tabelas que guiam o processo de análise sintática descendente preditiva no *parser* $LL(1)$ deterministicamente. Cada célula dessas tabelas corresponde a relação que existe em um par não-terminal, terminal. De forma que o valor da célula apontada por este par indica qual regra de produção deve ser aplicada quando o analisador encontrar este par específico durante a análise preditiva $LL(1)$.

