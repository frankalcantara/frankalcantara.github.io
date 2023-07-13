---
title: Introdução a Programação Lógica
layout: post
author: Frank
description: Uma aventura pelo universo matemático que fundamenta a programação lógica.
date: 2023-07-13T02:50:56.534Z
preview: ""
image: assets/images/prolog1.jpeg
tags:
  - Lógica
  - Programação Lógica
  - Prolog
categories:
  - disciplina
  - Lógica
  - Material de Aula
  - Matemática
rating: 5
slug: introducao-programacao-logica
keywords:
  - lógica
  - Programação
  - programação lógica
draft: true
---

A Programação Lógica é como ensinar um detetive computadorizado a resolver os mais intricados mistérios, permitindo que se preocupe apenas com o o quê e deixando o como a cargo da máquina. Nessa jornada, encontraremos linguagens matemáticas universais, inferências e deduções dignas de um Sherlock Holmes. Base de alguns modelos computacionais que estão mudando o mundo.

> "Logic programming is the future of artificial intelligence." - [Marvin Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky)

# Introdução

Imagine, por um momento, que estamos explorando o universo dos computadores, mas em vez de sermos os comandantes, ditando cada passo do caminho, nós damos as diretrizes gerais e deixamos que o computador deduza o caminho. Pode parecer estranho, a princípio, mas isso é exatamente o que a Programação Lógica faz.

Em vez de termos que mapear cada detalhe, como as linguagens que usam a Programação Imperativa, a Programação Lógica permite que declaremos o que queremos, e então o computador faz o trabalho pesado de descobrir a melhor rota para chegar até lá. Esta é a diferença entre computação e dedução. 

Na computação partimos de uma determinada expressão e seguimos um conjunto fixo de instruções até encontrar o resultado. Na dedução, partimos de uma conjectura e, de acordo com um conjunto específico de regras tentamos construir uma prova para esta conjectura. Esta prova, não é simples, o [Grande Teorema de Fermat tomou 357 anos para ser provado](https://en.wikipedia.org/wiki/Wiles%27s_proof_of_Fermat%27s_Last_Theorem).


Em nossa jornada, encontraremos algumas regras do universo da lógica, como a lógica de primeira ordem. Um tipo de linguagem matemática usada para expressar conceitos lógicos suficientemente rica para expressar uma quantidade enorme de problemas que precisamos resolver. Alguns que ainda nem conhecemos. Ainda assim,  suficientemente gerenciável para que os computadores possam lidar com ela.

Da mesma forma, vamos enfrentar a inferência e a dedução, duas ferramentas poderosas para extrair conhecimento de nossas declarações lógicas. Quase como um detetive que tira conclusões a partir de pistas: você tem algumas verdades e precisa descobrir outras verdades que são consequências diretas das primeiras verdades.

Vamos falar da Cláusula de Horn, um conceito um pouco mais estranho. Uma regra que torna todos os problemas expressos em lógica mais fácies de resolver. É como uma receita de bolo que, se corretamente seguida, torna o processo de cozinhar muito mais simples.

No final do dia, tudo que queremos é que nossos computadores sejam capazes de resolver problemas complexos com menos intervenção da nossa parte. Queremos que eles pensem, ou pelo menos, que simulem o pensamento. E a Programação Lógica é uma maneira de fazer isso.

A Programação Lógica, como quase tudo na computação é ridiculamente nova. Aparece em meados dos anos 1970 como uma evolução dos esforços das pesquisas sobre a prova computacional de teoremas matemáticos e inteligência artificial. Deste esforço surgiu a esperança de que poderíamos usar a lógica como um linguagem de programação. Em inglês, "programming logic" ou Prolog. Este artigo faz parte de uma série, e nesta série, vamos abordar Prolog. 

# 1. Lógica de Primeira Ordem

A lógica de primeira ordem, também chamada de lógica de predicados, é um dos fundamentos essenciais da ciência da computação e da programação. Essa lógica nos permite quantificar sobre objetos - isto é, fazer declarações que se aplicam a todos os membros de um conjunto ou a um membro particular desse conjunto. Por outro lado, ela nos limita de quantificar diretamente sobre predicados ou funções.

"A lógica de primeira ordem é como olhar para as estrelas em uma noite clara. Nós podemos ver e quantificar as estrelas individuais (objetos), mas não podemos quantificar diretamente sobre as constelações (predicados ou funções).

Essa restrição não é um defeito, mas sim um equilíbrio cuidadoso entre poder expressivo e simplicidade computacional. Dá-nos uma maneira de formular uma grande variedade de problemas que queremos resolver, sem tornar o processo de resolução desses problemas excessivamente complexo.

A lógica de primeira ordem é o nosso ponto de partida, é a nossa base. É uma maneira poderosa e útil de olhar para o universo, não tão complicada que não podemos começar a compreendê-la, mas suficientemente complexa para nos permitir descobrir coisas incríveis.

A Lógica de primeira ordem consiste de uma linguagem, consequentemente de um alfabeto, $\Sigma$ de um conjunto de axiomas e de um conjunto de regras de inferência. Esta linguagem consiste de todas as fórmulas bem formadas da teoria da lógica proposicional e predicativa. O conjunto de axiomas é um subconjunto deste conjunto acrescido de um conjunto de regras de inferência. O alfabeto $\Sigma$ pode ser dividido em conjuntos de símbolos agrupados por classes:

**variáveis, constantes e símbolos de pontuação**: vamos usar os símbolos do alfabeto latino em minúsculas e os símbolos de pontuação que usamos no português do Brasil. Especificamente os símbolos $($ e $)$ para definir a prioridade de operações. 

Vamos os símbolos $u$, $v$, $w$, $x$, $y$ e $z$ para indicar variáveis. Usaremos $a$, $b$, $c$, $d$ e $e$ para constantes.

**Funções**: usaremos os símbolos $\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$ e $\mathbf{i}$ para indicar todas as funções.

**Predicados**: usaremos os símbolos $\mathbf{p}$, $\mathbf{q}$, $\mathbf{r}$ e $\mathbf{s}$ para indicar predicados. 

**Operadores**: usaremos os símbolos tradicionais da lógica proposicional: $\neg$ (negação), $\wedge$ (disjunção, *and*), $\vee$ (conjunção, *or*), $\rightarrow$ (implicação) e $\leftrightarrow$ (equivalência).

**Quantificadores**: vamos nos manter na tradição e usar $\exists$ (quantificador existencial) e $\forall$ (quantificador universal).

**Fórmulas Bem Formatadas**: usaremos letras do alfabeto latino, maiúsculas para representar as fórmulas bem formatadas: $F$, $G$, $I$, $J$, $K$. 

Em qualquer linguagem matemática, a precedência das operações é como um rota, obrigatória para chegar ao objetivo real. Aqui, vamos usar $($ e $)$ de forma aninhada para garantir a ordem correta das operações. Contudo, vamos definir uma ordem de precedência para garantir a legibilidade das nossas fórmulas bem formatadas: 

$$\neg, \forall, \exists, \wedge, \vee, \rightarrow, \leftrightarrow$$

Com a maior ordem de precedência dada a $\neg$ e a menor $\leftrightarrow$. 

O uso dos parenteses e da ordem de precedência com parcimônia, permite que possamos escrever $(\forall x(\exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))))$ ou $\forall x \exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))$ que são a mesma fórmula bem formatada. Escolha a opção que seja mais fácil de ler e entender. Eu, vou tentar usar o mínimo da parenteses possível.

Fórmulas bem formatadas são conjuntos de termos e operações seguindo a ordem de precedência definida anteriormente e as regras de cada operação. 

Termos são definidos segundo as seguintes regras:

1. uma variável $x$ é um termo em sí;
2. uma constante $a$ é um termo em si;
3. se $\mathbf{f}$ é uma função de termos $(t_1, ... t_2)$ então $\mathbf{f}(t_1, ... t_2)$ é um termo. 

Agora que definimos termos, podemos definir uma fórmula bem formatada como sendo; 

1. se $\mathbf{p}$ é um predicado de termos $(t_1, ... t_2)$ então $\mathbf{p}(t_1, ... t_2)$ é um fórmula bem formatada, um átomo. 
2. se $F$ e $G$ são fórmulas bem formatadas então: $\neg F$, $F\wedge G$, $F \vee G$, $F \rightarrow G$ e $F \leftrightarrow G$ são fórmulas bem formatadas. 
3. se $F$ é uma fórmula bem formatada e $x$ uma variável então $\exists x F$ e $\forall x F$ são fórmulas bem formatadas. 

Finalmente podemos dizer que a linguagem da lógica de primeira ordem é o conjunto de todas as fórmulas bem formatadas construídas a partir da combinação dos símbolos do alfabet $\Sigma$. 

