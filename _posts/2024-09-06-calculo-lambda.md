---
layout: post
title: Cálculo Lambda para Novatos
author: Frank
categories:
   - Matemática
   - Linguagens Formais
   - Lógica Matemática
tags:
   - Matemática
   - Linguagens Formais
image: assets/images/calculolambda.webp
description: Introdução ao cálculo lambda.
slug: calculo-lambda-para-novatos
keywords:
   - cálculo lambda
   - linguagens formais
   - lógica matemática
   - computação funcional
   - abstração lambda
   - aplicação de funções
   - currying
   - combinadores de ponto fixo
   - alonzo church
   - teoria da computação
   - funções matemáticas
   - recursão
   - teoria de funções
   - expressão lambda
rating: 5
published: 2024-09-08T21:19:20.392Z
draft: 2024-09-08T21:19:20.392Z
featured: True
toc: True
preview: Começamos com os fundamentos teóricos e seguimos para as aplicações práticas em linguagens de programação funcionais. Explicamos abstração, aplicação e recursão. Mostramos exemplos de _currying_e combinadores de ponto fixo. O cálculo lambda é a base da computação funcional.
beforetoc: Começamos com os fundamentos teóricos e seguimos para as aplicações práticas em linguagens de programação funcionais. Explicamos abstração, aplicação e recursão. Mostramos exemplos de _currying_e combinadores de ponto fixo. O cálculo lambda é a base da computação funcional.
lastmod: 2024-10-21T21:28:02.879Z
date: 2024-09-08T21:19:30.955Z
---

# 1. Introdução, História e Motivações e Limites

O cálculo lambda é uma teoria formal para expressar computação por meio da visão de funções como fórmulas. Um sistema para manipular funções como sentenças, desenvolvido por [Alonzo Church](https://en.wikipedia.org/wiki/Alonzo_Church) sob uma visão extensionista das funções na década de 1930. Nesta teoria usamos funções para representar todos os dados e operações. Em cálculo lambda, tudo é uma função e uma função simples é parecida com:

$$\lambda x.\;x + 1$$

Esta função adiciona $1$ ao seu argumento. O $\lambda$ indica que estamos definindo uma função.

Na teoria da computação definida por Church com o cálculo lambda existem três componentes básicos: as variáveis: $x\,$, $y\,$, $z$; as abstrações $\lambda x.\;E\,$, onde $E$ é uma expressão lambda e a aplicação $(E\;M)\,$, onde $E$ e $M$ são expressões lambda. Com estes três componentes e o cálculo lambda é possível expressar qualquer função computacionalmente possível.

A década de 1930 encerrou a busca pela consistência da matemática iniciada nas última décadas do século XIX. Neste momento histórico os matemáticos buscavam entender os limites da computação. Questionavam: Quais problemas podem ser resolvidos por algoritmos? Existem problemas não computáveis?

Estas questões surgiram como consequência dos trabalhos no campo da lógica e da lógica combinatória que despontaram no final do século XIX e começo do século XX. Em um momento crítico, Church ofereceu respostas, definindo que as funções computáveis são aquelas que podem ser expressas em cálculo lambda. Um exemplo simples de função computável seria:

$$\text{add} = \lambda m. \lambda n.\;m + n$$

Esta função soma dois números. **Todas as funções lambda são, por definição unárias e anônimas**. Assim, a função acima está sacrificando o rigor matemático para facilitar o entendimento. Esta é uma liberdade que é abusada descaradamente, neste livro, sempre com a esperança que estando mais próximo do que aprendemos nos ciclos básicos de estudo, é mais simples criar o nível de entendimento necessário.

O trabalho de Church estabeleceu limites claros para computação, ajudando a revelar o que é e o que não é computável. Sobre esta formalização foi construída a Ciência da Computação. Seu objetivo era entender e formalizar a noção de _computabilidade_. Church buscava um modelo matemático preciso para computabilidade. Nesta busca ele criou uma forma de representar funções e operações matemáticas de forma abstrata, usando como base a lógica combinatória desenvolvida anos antes [^cita4].

Na mesma época, [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) propôs a [máquina de Turing](https://en.wikipedia.org/wiki/Turing_machine), uma abordagem diferente para tratar a computabilidade. Apesar das diferenças, essas duas abordagens provaram ser equivalentes e, juntas, estabeleceram os alicerces da teoria da computação moderna. O objetivo de Church era capturar o conceito de _cálculo efetivo_[^cita5]. Em 1936, no artigo _On
computable numbers, with an application to the Entscheidungsproblem_[^cita6], Turing criou a Ciência da Computação e iniciou a computação artificial determinando o futuro da civilização[^cita9].

O artigo _On
computable numbers, with an application to the Entscheidungsproblem_ foi submetido para publicação em 28 de maio de 1936. Sendo esta a data de nascimento da Ciência da Computação.

Seu trabalho foi uma das primeiras tentativas de formalizar matematicamente o ato de computar. Mais tarde, a equivalência entre o cálculo lambda e a máquina de Turing consolidou a ideia de que ambos podiam representar qualquer função computável, levando à formulação da [Tese de Church-Turing](https://en.wikipedia.org/wiki/Church%E2%80%93Turing_thesis). Afirmando que qualquer função computável pode ser resolvida pela máquina de touring e, equivalentemente, pelo cálculo lambda, fornecendo uma definição matemática precisa do que é, ou não é, computável.

A partir do meio da década de 1930, vários matemáticos e lógicos, como [Church](https://en.wikipedia.org/wiki/Alonzo_Church), [Turing](https://en.wikipedia.org/wiki/Alan_Turing), [Gödel](https://en.wikipedia.org/wiki/Kurt_G%C3%B6del) e [Post](https://en.wikipedia.org/wiki/Emil_Leon_Post), desenvolveram modelos diferentes para formalizar a computabilidade. Cada um desses modelos abordou o problema de uma perspectiva exclusiva. Como pode ser visto na Tabela 1.

| Abordagem                               | Características Principais                                      | Contribuições / Diferenças                                           |
|-----------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------|
| Cálculo Lambda<br> (Church, $1936$)     | • Sistema formal baseado em funções<br>• Usa abstração ($\lambda$) e aplicação<br>• Funções como objetos de primeira classe | • Base para linguagens funcionais<br>• Ênfase em composição de funções<br>• Influenciou teoria dos tipos |
| Máquina de Turing <br>(Turing, $1936$)  | • Modelo abstrato de máquina<br>• Fita infinita, cabeçote de leitura/escrita<br>• Estados finitos e transições | • Modelo intuitivo de computação<br>• Base para análise de complexidade<br>• Inspirou arquitetura de computadores |
| Funções Recursivas<br> (Gödel, $1934$)  | • Baseado em teoria dos números<br>• Usa recursão e minimização<br>• Definição indutiva de funções | • Formalização rigorosa<br>• Conexão com lógica matemática<br>• Base para teoria da recursão |
| Cálculo Sentencial<br> (Post, $1943$)   | • Manipulação de strings<br>• Regras de produção<br>• Transformação de símbolos | • Simplicidade conceitual<br>• Base para gramáticas formais<br>• Influenciou linguagens de programação |

_Tabela 1.A. Relação entre as contribuições de Church, Gödel e Post_{: Legenda}

Church propôs o cálculo lambda para descrever funções de forma simbólica, usando a _abstração lambda_. Esse modelo representa funções como estruturas de primeira classe formalizando a computabilidade de funções e variáveis.

Em 1936, [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) propôs a máquina de Turing. Essa máquina, conceitual, é formada por uma fita infinita que pode ser lida e manipulada por uma cabeça de leitura/escrita, seguindo um conjunto de regras e se movendo entre estados fixos.

A visão de Turing apresentava uma abordagem mecânica da computação, complementando a perspectiva simbólica de Church e sendo complementada por esta. Church havia provado que algumas funções não são computáveis. O _Problema da Parada_ é um exemplo famoso:

$$\text{parada} = \lambda f. \lambda x. \text{(f(x) para?)}$$

Church mostrou que esta função não pode ser expressa no cálculo lambda e, consequentemente, não pode ser computada. A atenta leitora deve saber que Church e Turing, não trabalharam sozinhos.

[Kurt Gödel](https://en.wikipedia.org/wiki/Kurt_G%C3%B6del) contribuiu com a ideia de funções recursivas, uma abordagem algébrica que define a computação por meio de funções primitivas e suas combinações. Ele explorou a computabilidade a partir de uma perspectiva aritmética, usando funções que podem ser definidas recursivamente. Essa visão trouxe uma base numérica e algébrica para o conceito de computabilidade.

Em paralelo, [Emil Post](https://en.wikipedia.org/wiki/Emil_Leon_Post) desenvolveu os sistemas de reescrita, ou Cálculo Sentencial, baseados em regras de substituição de strings. O trabalho de Post foi importante para a teoria das linguagens formais e complementou as outras abordagens, fornecendo uma visão baseada em regras de substituição.

Apesar das diferenças estruturais entre o cálculo lambda, as máquinas de Turing, as funções recursivas e o Cálculo Sentencial de Post, todos esses modelos têm o mesmo poder computacional. Uma função que não for computável em um destes modelos, não o será em todos os outros. Neste ponto estava definida a base para a construção da Ciência da Computação.

## 1.1. A Inovação de Church: Abstração Funcional

O trabalho de Alonzo Church é estruturado sobre a ideia de _abstração funcional_. Esta abstração permite tratar funções como estruturas de primeira classe. Neste cenário, as funções podem ser passadas como argumentos, retornadas como resultados e usadas em expressões compostas.

No cálculo lambda, uma função é escrita como $\lambda x.\;E\,$. Aqui, $\lambda$ indica que é uma função, $x$ é a variável ligada, onde a função é aplicada, e $E$ é o corpo da função. Por exemplo, a função que soma $1$ a um número é escrita como $\lambda x.\;x + 1\,$. Isso possibilita a manipulação direta de funções, sem a necessidade de linguagens ou estruturas rígidas. A Figura 1.1.A apresenta o conceito de funções de primeira classe.

![Diagrama mostrando uma função cujo corpo é composto por outra função lambda e um valor. No diagrama vemos a função principal recebendo a função do corpo, e um valor. Finalmente mostra a função composta e o resultado da sua aplicação](/assets/images/funcPrima.webp)
_Figura 1.1.A: Diagrama de Abstração e Aplicação usando funções no corpo da função. A função principal é a função de ordem superior, ou de primeira classe_{: legenda}

>No cálculo lambda, uma função de ordem superior é uma função que aceita outra função como argumento ou retorna uma função como resultado. Isso significa que uma função de ordem superior trata funções como valores, podendo aplicá-las, retorná-las ou combiná-las com outras funções.
>
>Seja $f$ uma função no cálculo lambda. Dizemos que $f$ é uma função de ordem superior se:
>
>1. $f$ aceita uma função como argumento.
>2. $f$ retorna uma função como resultado.
>
>No cálculo lambda puro, as funções são anônimas. No entanto, em contextos de programação funcional, é comum nomear funções de ordem superior para facilitar seu uso e identificação em operações complexas. Vamos tomar esta licença poética, importada da programação funcional, de forma livre e temerária em todo este texto. Sempre que agradar ao pobre autor.
>
>Considere a mesma função de adição de ordem superior, agora nomeada como `adicionar`:
>
>$$\text{adicionar} = \lambda x.\; \lambda y.\; x + y$$
>
>Essa função nomeada pode ser usada como argumento para outras funções de ordem superior, como `mapear`:
>
>$$\text{mapear} \; (\text{adicionar} \; 2) \; [1, 2, 3]$$
>
>Neste caso, a aplicação resulta em:
>
>$$[3, 4, 5]$$
>

A abstração funcional induziu a criação do conceito de _funções anônimas_ em linguagens de programação, em especial, e em linguagens formais de forma geral. Linguagens de programação, como Haskell, Lisp, Python e JavaScript, adotam essas funções como parte das ferramentas disponíveis em sua sintaxe. Tais funções são conhecidas como _lambda functions_ ou _arrow functions_.

Na matemática, a abstração funcional possibilitou a criação de operações de combinação, um conceito da lógica combinatória. Estas operações de combinação são representadas na aplicação de combinadores que, por sua vez, definem como combinar funções. No cálculo lambda, e nas linguagens funcionais, os combinadores, como o _combinador Y_, facilitam a prova de conceitos matemáticos ou, permitem acrescentar funcionalidades ao cálculo lambda. O combinador $Y\,$, por exemplo, permite o uso de recursão em funções. O combinador $Y\,$, permitiu provar a equivalência entre o Cálculo lambda, a máquina de touring e a recursão de Gödel. Solidificando a noção de computabilidade.

Na notação matemática clássica, as funções são representadas usando símbolos de variáveis e operadores. Por exemplo, uma função quadrática pode ser escrita como:

$$f(x) \, = x^2 + 2x + 1$$

Essa notação é direta e representa um relação matemática entre dois conjuntos. Descrevendo o resultado da aplicação da relação a um dos elementos de un conjunto, encontrando o elemento relacionado no outro. No exemplo acima, se aplicarmos $f$ em $2$ teremos $9$ como resultado da aplicação. A definição da função $f$ não apresenta o processo de computação necessário. Nós sabemos como calcular o resultado porque conhecemos a sintaxe da aritmética e a semântica da álgebra.

O cálculo lambda descreve um processo de aplicação e transformação de variáveis. Enquanto a Máquina de Turing descreve a computação de forma mecânica, o cálculo lambda foca na transformação de expressões. Para começarmos a entender o poder do cálculo lambda, podemos trazer a função $F$ um pouco mais perto dos conceitos de Church.

Vamos começar definindo uma expressão $M$ contendo uma variável $x\,$, na forma:

$$M(x) = x^2 + 2x + 1$$

A medida que $x$ varia no domínio dos números naturais podemos obter a função representada na notação matemática padrão por $x \mapsto x^2 + x + 1$ este relação define o conjunto de valores que $M$ pode apresentar em relação aos valores de $x\,$. Porém, se fornecermos um valor de entrada específico, por exemplo, $2\,$, para $x\,$, valor da função será $2^2 + 4 + 1 = 9\,$.

Avaliando funções desta forma, Church introduziu a notação

$$λx: (x^2 + x + 1)$$

Para representar a expressão $M\,$. Nesta representação temos uma abstração. Justamente porque a expressão estática $M(x)\,$, para $x$ fixo, torna-se uma função _abstrata_ representada por $λx:M\,$.

Linguagens de programação modernas, como Python ou JavaScript, têm suas próprias formas de representar funções. Por exemplo, em Python, uma função pode ser representada assim:

```haskell
-- Define a função f, que toma um argumento x and devolve x^2 + 2*x + 1
f :: Int -> Int
f x = x^2 + 2*x + 1
```

As linguagens funcionais representam funções em um formato baseado na sintaxe do cálculo lambda. Em linguagens funcionais, funções são tratadas como elementos e a aplicação de funções é a operação que define a computação. Neste ambiente as funções têm tal importância, e destaque, que dizemos que no cálculo lambda, funções são cidadãos de primeira classe. Uma metáfora triste. Porém, consistente.

**No cálculo lambda, usamos _abstração_ e _aplicação_ para criar e aplicar funções.** Na criação de uma função que soma dois números, escrita como:

$$\lambda x. \lambda y.\;(x + y)$$

A notação $\lambda$ indica que estamos criando uma função anônima. Essa abstração explícita é menos comum na notação matemática clássica na qual, geralmente definimos funções nomeadas.

A atenta leitora deve notar que a abstração e a aplicação são operações distintas do cálculo lambda, como pode ser visto na Figura 1.1.B.

![Diagrama mostrando abstração, a aplicação da função a um valor e, finalmente o resultado da aplicação da função](/assets/images/abstAplica.webp)
_Figura 1.1.B: Diagrama da relação entre abstração e aplicação no cálculo lambda_{: legenda}

A abstração, representada por $\lambda x.\;E\,$, define uma função onde $x$ é o parâmetro e $E$ é o corpo da função. Por exemplo, $\lambda x.\;x + 5$ define uma função que soma $5$ ao argumento fornecido. Outro exemplo é $\lambda f. \lambda x.\;f\;(f\;x)\,$, que descreve uma função que aplica o argumento $f$ duas vezes ao segundo argumento $x\,$.

A abstração cria uma função sem necessariamente avaliá-la. A variável $x$ em $\lambda x.\;E$ está ligada à função e não é avaliada até que um argumento seja aplicado. **A abstração é puramente declarativa**, descreve o comportamento da função sem produzir um valor imediato.

**A aplicação**, expressa por $M\;N\,$, **é o processo equivalente a avaliar uma função algébrica em um argumento**. Aqui, $M$ representa a função e $N$ o argumento que é passado para essa função. Ou, como dizemos em cálculo lambda, **o argumento que será aplicado a função***. Considere a expressão:

$$(\lambda x.\;x + 5)\;3$$

Neste caso, temos a aplicação da função $\lambda x.\;x + 5$ ao argumento $3\,$, resultando em $8\,$. Outro exemplo:

$$(\lambda f. \lambda x.\;f\;(f\;x))\;(\lambda y.\;y * 2)\;3$$

Neste caso, temos uma função de composição dupla é aplicada à função que multiplica valores por dois e, em seguida, ao número $3\,$, resultando em $12\,$.

Em resumo, **a abstração define uma função ao associar um parâmetro a um corpo de expressão; enquanto a aplicação avalia essa função ao fornecer um argumento**. Ambas operações são independentes, mas mas interagem para permitir a avaliação de expressões no cálculo lambda.

O elo entre abstração e aplicação é uma forma de avaliação chamada redução-$beta\,$. Dada uma abstração $λ\,$, $λx:M$ e algum outro termo $N\,$, pensado como um argumento, temos a regra de avaliação, chamada redução-$beta$ dada por:

$$(λx:M)\ N \longrightarrow_{\beta} M[x := N];$$

onde $M[N/x]$ indica o resultado de substituir $N$ em todas as ocorrências de $x$ em $M\,$. Por exemplo, se $M = λx: (x^2 + x + 1)$ e $N = 2y + 1\,$, teremos:

$$(λx: (x^2 + x + 1))(2y + 1) \longrightarrow_{\beta} (2y + 1)^2 + 2y + 1 + 1.$$

Esta uma operação puramente formal, inserindo $N$ onde quer que $x$ ocorra em $M\,$.

Ainda há uma coisa que a amável leitora deve ter em mente antes de continuarmos.  No cálculo lambda, os números naturais, as operações aritméticas $+$ e $\times\,$, assim como a exponenciação que usamos em $M$ precisam ser representados como termos $λ\,$. Só assim, a avaliação das expressões lambda irão computar corretamente.

## 1.2. O Cálculo Lambda e a Lógica

O cálculo lambda possui uma relação direta com a lógica matemática, especialmente através do **isomorfismo de Curry-Howard**. Esse isomorfismo cria uma correspondência entre provas matemáticas e programas computacionais. Em termos simples, uma prova de um teorema é um programa que constrói um valor a partir de uma entrada, e provar teoremas equivale a computar funções.

Essa correspondência deu origem ao paradigma das _provas como programas_.

>O paradigma de _provas como programas_ é uma correspondência entre demonstrações matemáticas e programas de computador, conhecida como **correspondência de Curry-Howard**. Segundo esse paradigma, cada prova em lógica formal corresponde a um programa e cada tipo ao qual uma prova pertence corresponde ao tipo de dado que um programa manipula. Essa ideia estabelece uma ponte entre a lógica e a teoria da computação, permitindo a formalização de demonstrações como estruturas computáveis e o desenvolvimento de sistemas de prova automáticos e seguros.

O cálculo lambda define computações e serve como uma linguagem para representar e verificar a correção de algoritmos. Esse conceito se expandiu na pesquisa moderna e fundamenta assistentes de prova e linguagens de programação com sistemas de tipos avançados, como o **Sistema F** e o **Cálculo de Construções**.

>O **Sistema F**, conhecido como cálculo lambda polimórfico de segunda ordem, é uma extensão do cálculo lambda que permite quantificação universal sobre tipos. Desenvolvido por [Jean-Yves Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard) e [John Reynolds](https://en.wikipedia.org/wiki/John_C._Reynolds) de forma independente.

O **Sistema F** é utilizado na teoria da tipagem em linguagens de programação, permitindo expressar abstrações mais poderosas, como tipos genéricos e polimorfismo paramétrico. Servindo como base para a formalização de alguns sistemas de tipos usados em linguagens funcionais modernas.

>O **Cálculo de Construções** é um sistema formal que combina elementos do cálculo lambda e da teoria dos tipos para fornecer uma base para a lógica construtiva. Ele foi desenvolvido por [Thierry Coquand](https://en.wikipedia.org/wiki/Thierry_Coquand) e é uma extensão do **Sistema F**, com a capacidade de definir tipos dependentes e níveis mais complexos de abstração. O cálculo de construções é a base da linguagem **Coq**, um assistente de prova utilizado para formalizar demonstrações matemáticas e desenvolver software verificado.

A atenta leitora deve ter percebido que o cálculo lambda não é um conceito teórico abstrato; ele possui implicações práticas, especialmente na programação funcional. Linguagens como Lisp, Haskell, OCaml e F# incorporam princípios do cálculo lambda. Exemplos incluem:

1. **Funções como cidadãos de primeira classe**: No cálculo lambda, funções são valores. Podem ser passadas como argumentos, retornadas como resultados e manipuladas livremente. Isso é um princípio central da programação funcional, notadamente em Haskell.

2. **Funções de ordem superior**: O cálculo lambda permite a criação de funções que operam sobre outras funções. Isso se traduz em conceitos aplicados em funções como `map`, `filter` e `reduce` em linguagens funcionais.

3. **currying**: A técnica de transformar uma função com múltiplos argumentos em uma sequência de funções de um único argumento é natural no cálculo lambda e no Haskell e em outras linguagens funcionais.

4. **Avaliação preguiçosa (_lazy_)**: Embora não faça parte do cálculo lambda puro, a semântica de redução do cálculo lambda, notadamente a estratégia de redução normal inspirou o conceito de avaliação preguiçosa em linguagens como Haskell.

5. **Recursão**: Definir funções recursivas é essencial em programação funcional. No cálculo lambda, isso é feito com combinadores de ponto fixo.

## 1.3. Representação de Valores e Computações

Uma das características principais do cálculo lambda é representar valores, dados e computações complexas, usando exclusivamente funções. Até números e _booleanos_ são representados de forma funcional. Um exemplo indispensável é a representação dos números naturais, chamada **Numerais de Church**:

$$\begin{align*}
0 &= \lambda s.\;\lambda z.\;z \\
1 &= \lambda s.\;\lambda z.\;s\;z \\
2 &= \lambda s.\;\lambda z. s\;(s\;z) \\
3 &= \lambda s.\;\lambda z.\;s\;(s (s\;z))
\end{align*}$$

Voltaremos a esta notação mais tarde. O importante é que essa codificação permite que operações aritméticas sejam definidas inteiramente em termos de funções. Por exemplo, a função sucessor, usada para provar a criação de conjuntos de números contáveis, como os naturais e os inteiros, pode ser expressa como:

$$\text{succ} = \lambda n.\;\lambda s.\;\lambda z.\;s\;(n\;s\;z)$$

Assim, operações como adição e multiplicação podem ser construídas usando termos lambda.

Um dos resultados mais profundos da formalização da computabilidade, utilizando o cálculo lambda e as máquinas de Turing, foi a identificação de problemas _indecidíveis_. Problemas para os quais não podemos decidir se o algoritmo que os resolve irá parar em algum ponto, ou não.

O exemplo mais emblemático é o Problema da Parada, formulado por Alan Turing em 1936. O Problema da Parada questiona se é possível construir um algoritmo que, dado qualquer programa e uma entrada, determine se o programa eventualmente terminará ou continuará a executar indefinidamente. Em termos formais, essa questão pode ser expressa como:

$$
\exists f : \text{Programa} \times \text{Entrada} \rightarrow \{\text{Para}, \text{NãoPara}\}?
$$

Turing demonstrou, por meio de um argumento de diagonalização, que tal função $f$ não pode existir. Esse resultado mostra que não é possível determinar, de forma algorítmica, o comportamento de todos os programas para todas as possíveis entradas.

Outro problema indecidível, elucidado pelas descobertas em computabilidade, é o _décimo problema de Hilbert_. Esse problema questiona se existe um algoritmo que, dado um polinômio com coeficientes inteiros, possa determinar se ele possui soluções inteiras. Formalmente, o problema pode ser expresso assim:

$$
P(x_1, x_2, \dots, x_n) \, = 0
$$

>Os problemas de Hilbert são uma lista de 23 problemas matemáticos propostos por David Hilbert em 1900, durante o Congresso Internacional de Matemáticos em Paris. Esses problemas abordam questões em várias áreas da matemática e estimularam muitas descobertas ao longo do século XX. Cada problema visava impulsionar a pesquisa e delinear os desafios mais importantes da matemática da época. Alguns dos problemas foram resolvidos, enquanto outros permanecem abertos ou foram provados como indecidíveis, como o **décimo problema de Hilbert**, que pergunta se existe um algoritmo capaz de determinar se um polinômio com coeficientes inteiros possui soluções inteiras.

Em 1970, [Yuri Matiyasevich] \ ,(Yuri Matiyasevich), em colaboração com [Julia Robinson](https://en.wikipedia.org/wiki/Julia_Robinson), [Martin Davis] \ ,(<https://en.wikipedia.org/wiki/Martin_Davis_(mathematician)>) e [Hilary Putnam](https://en.wikipedia.org/wiki/Hilary_Putnam), provou que tal algoritmo não existe. Esse resultado teve implicações profundas na teoria dos números e demonstrou a indecidibilidade de um problema central na matemática.

A equivalência entre o cálculo lambda, as máquinas de Turing e as funções recursivas permitiu estabelecer os limites da computação algorítmica. O Problema da Parada e outros resultados indecidíveis, como o décimo problema de Hilbert, mostraram que existem problemas além do alcance dos algoritmos.

A **Tese de Church-Turing** formalizou essa ideia, afirmando que qualquer função computável pode ser expressa por um dos modelos computacionais mencionados, Máquina de Turing, recursão e o cálculo lambda[^cita6]. Essa tese forneceu a base rigorosa necessária ao desenvolvimento da Ciência da Computação, permitindo a demonstração da existência de problemas não solucionáveis por algoritmos.

## 1.4. Limitações do Cálculo Lambda e Sistemas Avançados

O cálculo lambda é poderoso. Ele pode expressar qualquer função computável. Mas tem limitações: **não tem tipos nativos** ou qualquer sistema de tipos. Tudo é função. Números, booleanos, estruturas de dados são codificados como funções; **Não tem estado mutável**. cada expressão produz um novo resultado. Não modifica valores existentes. Isso é uma vantagem em alguns cenários, mas agrega complexidade a definição de algoritmos; **não tem controle de fluxo direto**, _Loops_ e condicionais são simulados com funções recursivas.

Apesar de o cálculo lambda ser chamado de _a menor linguagem de programação_ a criação de algoritmos sem controle de fluxo não é natural para programadores, ou matemáticos, nativos do mundo imperativo.

Por fim, o cálculo lambda **pode ser ineficiente**. Por mais que doa confessar isso. Mas temos que admitir que codificações como números de Church podem levar a cálculos lentos. Performance nunca foi um objetivo.

Sistemas mais avançados de cálculo lambda abordam algumas das deficiências do cálculo lambda expandindo, provando conceitos, criando novos sistemas lógicos, ou criando ferramentas de integração. Entre estes sistemas, a leitora poderia considerar:

1. **Sistemas de tipos**: o cálculo lambda tipado adiciona tipos. **O Sistema F**, por exemplo, permite polimorfismo. A função $\Lambda \alpha. \lambda x:\alpha.\;x$ é polimórfica e funciona para qualquer tipo $\alpha\,$.

2. **Efeitos colaterais**: o cálculo lambda com efeitos colaterais permite mutação e I/O. A função $\text{let}\;x = \text{ref}\;0\;\text{in}\;x := !x + 1$ cria uma referência mutável e providencia um incremento.

3. **Construções imperativas**: algumas extensões adicionam estruturas de controle diretas. Este é o caso de $\text{if}\;b\;\text{then}\;e_1\;\text{else}\;e_2\,$. Neste caso, temos um condicional direto, não implementado como uma função.

4. **Otimizações**: implementações eficientes usam representações otimizadas. A função $2 + 3 \rightarrow 5$ usa aritmética tradicional, não números de Church. Aqui, a observadora leitora já deve ter percebido que, neste livro, quando encontrarmos uma operação aritmética, vamos tratá-la como tal.

Estas extensões agregam funcionalidade e transformam o cálculo lambda em uma ferramenta matemática mais flexível. Muitas vezes com o objetivo de criar algoritmos, facilitar o uso de linguagens de programação baseadas em cálculo lambda no universo fora da matemática.

## 1.5. Notações e Convenções

O cálculo lambda utiliza uma notação específica para representar funções, variáveis, termos e operações. Abaixo estão as notações e convenções, além de algumas expansões necessárias para a compreensão completa.

### 1.4.1. Símbolos Básicos

- **$\lambda$**: indica a definição de uma função anônima. Por exemplo, $\lambda x.\;x + 1$ define uma função que recebe $x$ e retorna $x + 1\,$.
  
- **Variáveis**: letras minúsculas, como $x\,$, $y\,$, $z\,$, representam variáveis no cálculo lambda.

- **Termos**: letras maiúsculas, como $M\,$, $N\,$, representam termos ou expressões lambda.

- **Aplicação de função**: a aplicação de uma função a um argumento é representada como $(M\;N)\,$, onde $M$ é uma função e $N$ é o argumento. Quando há múltiplas aplicações, como em $((M\;N)\;P)\,$, elas são processadas da esquerda para a direita.

- **Redução**: a seta $\rightarrow$ indica o processo de avaliação, ou redução, de uma expressão lambda. Por exemplo, $(\lambda x.\;x + 1)\;2 \rightarrow 3\,$. Indica que depois da aplicação e substituição a função chegará a $3\,$.

- **redução-$beta$**: a notação $ \rightarrow_\beta $ é usada para indicar a redução beta, um passo específico de substituição em uma expressão lambda. Exemplo: $(\lambda x.\;x + 1)\;2 \rightarrow_\beta 3\,$. A redução beta será a substituição de $x$ por $2\,$, resultando em $2+1$ e finalmente em $3\,$.

- **Equivalência de termos**: o símbolo $\equiv$ denota equivalência entre termos. Dois termos $M \equiv N$ são considerados estruturalmente equivalentes.

### 1.4.2. Tipagem e Contexto

- **Contexto de Tipagem ($\Gamma$)**: representa o contexto de tipagem, que é um conjunto de associações entre variáveis e seus tipos. Por exemplo, $\Gamma = \{ x: \text{Nat}, y: \text{Bool} \}\,$. Dentro de um contexto $\Gamma\,$, um termo pode ter um tipo associado: $\Gamma \vdash M : A$ significa que no contexto $\Gamma\,$, o termo $M$ tem tipo $A\, \,$.

- **Julgamento de tipo**: o símbolo $\vdash$ é utilizado para julgar o tipo de um termo dentro de um contexto de tipagem. Por exemplo, $\Gamma \vdash M : A$ significa que, no contexto $\Gamma\,$, o termo $M$ tem o tipo $A\,$.

- **Tipagem explícita**: usamos $x : A$ para declarar que a variável $x$ tem tipo $A\,$. Por exemplo, $n : \text{Nat}$ indica que $n$ é do tipo número natural ($\text{Nat}$).

### 1.4.3. Funções de Alta Ordem e Abstrações

- **Funções de Alta Ordem**: funções que recebem outras funções como argumentos ou retornam funções como resultado são chamadas de funções de alta ordem. Por exemplo, $(\lambda f. \lambda x.\;f(f\;x))$ é uma função de alta ordem que aplica $f$ duas vezes ao argumento $x\,$.

- **Abstrações Múltiplas**: abstrações aninhadas podem ser usadas para criar expressões mais complexas, como $(\lambda x.\;(\lambda y.\;x + y))\,$. Esse termo define uma função que retorna outra função.

### 1.4.4. Variáveis Livres e Ligadas

- **Variáveis Livres**: uma variável $x$ é considerada livre em uma expressão lambda se não estiver ligada a um operador $\lambda\,$. A notação $FV(M)$ é usada para representar o conjunto de variáveis livres, _Free Variables_, em um termo $M\,$. Por exemplo, em $\lambda y.\;x + y\,$, a variável $x$ é livre e $y$ é ligada.

- **Variáveis Ligadas**: uma variável é considerada ligada se estiver associada a um operador $\lambda\,$. Por exemplo, em $\lambda x.\;x + 1\,$, a variável $x$ é ligada. A notação $BV(M)$ representa o conjunto das variáveis ligadas, _Bound Variable_, no termo $M\,$.

### 1.4.5. Operações Aritméticas

O cálculo lambda permite incluir operações aritméticas dentro das expressões. Por exemplo:

- **Adição**: $x + 1\,$, onde $x$ é uma variável e $+$ é a operação de soma.
- **Multiplicação**: $x \times 2\,$, onde $\times$ representa a multiplicação.
- **Potência**: $x^2\,$, onde o operador de potência eleva $x$ ao quadrado.
- **Operações compostas**: exemplos incluem $x^2 + 2x + 1$ e $x \times y\,$, que seguem as regras usuais de aritmética.

### 1.4.6. Expansões Específicas

- **Notação de Tuplas e Produtos Cartesianos**: o produto cartesiano de conjuntos pode ser representado por notações como $(A \times B) \rightarrow C\,$, que denota uma função que recebe um par de elementos de $A$ e $B$ e retorna um valor em $C\,$.

- **Funções Recursivas**: funções recursivas podem ser descritas usando notação lambda. Um exemplo comum é a definição da função de fatoriais: $f = \lambda n.\;\text{if}\;n = 0\;\text{then}\;1\;\text{else}\;n \times f(n - 1)\,$.

### 1.4.7. Notações Alternativas

- **Parênteses Explícitos**: frequentemente os parênteses são omitidos por convenção, mas podem ser adicionados para clareza em expressões mais complexas, como $((M\;N)\;P)\,$.

- **Reduções Sequenciais**: Quando múltiplas reduções são realizadas, pode-se usar notação como $M \rightarrow_\beta N \rightarrow_\beta P\,$, para descrever o processo completo de avaliação.

### 1.4.8. Convenção de Nomes e Variáveis Livres e Ligadas

No cálculo lambda, as variáveis têm escopo léxico. O escopo é determinado pela estrutura sintática do termo, não pela ordem de avaliação. Uma variável é **ligada** quando aparece dentro do escopo de uma abstração que a introduz. Por exemplo: em $\lambda x.\lambda y.x\;y\,$, tanto $x$ quanto $y$ estão ligadas e em $\lambda x.(\lambda x.\;x)\;x\,$, ambas as ocorrências de $x$ estão ligadas, mas a ocorrência interna (no termo $\lambda x.\;x$) _sombreia_ a externa.

**Uma variável é livre quando não está ligada por nenhuma abstração**. por exemplo: em $\lambda x.\;x\;y\,$, $x$ está ligada, mas $y$ está livre. Ou ainda, em $(\lambda x.\;x)\;y\,$, $y$ está livre.

O conjunto de variáveis livres de um termo $E\,$, denotado por $FV(E)\,$, pode ser definido recursivamente:

1. $FV(x) = \{x\}$
2. $FV(\lambda x.\;E) = FV(E) \setminus \{x\}$
3. $FV(E\;N) = FV(E) \cup FV(N)$

Formalmente dizemos que para qualquer termo termo lambda $M\,$, o conjunto $FV(M)$ de variáveis livres de $M$ e o conjunto $BV(M)$ de variáveis ligadas em $M$ são definidos de forma indutiva da seguinte:

1. Se $M = x$ (uma variável), então:
   - $FV(x) = \{x\}$
   - $BV(x) = \emptyset$

2. Se $M = (M_1 M_2)\,$, então:
   - $FV(M) = FV(M_1) \cup FV(M_2)$
   - $BV(M) = BV(M_1) \cup BV(M_2)$

3. Se $M = (\lambda x: M_1)\,$, então:
   - $FV(M) = FV(M_1) \setminus \{x\}$
   - $BV(M) = BV(M_1) \cup \{x\}$

Se $x \in FV(M_1)\,$, dizemos que as ocorrências da variável $x$ ocorrem no escopo de $\lambda\,$. Um termo lambda $M$ é fechado se $FV(M) = \emptyset\,$, ou seja, se não possui variáveis livres.

O que a atenta leitora não deve perder de vista é que **as variáveis ligadas são somente marcadores de posição**, de modo que elas podem ser renomeadas livremente sem alterar o comportamento de redução do termo, desde que não entrem em conflito com as variáveis livres.

Os termos $\lambda x:\;(x(\lambda y:\;x(y\;x))$ e $\lambda x:\;(x(\lambda z: x\;(z\;x))$ devem ser considerados equivalentes. Da mesma forma, os termos $\lambda x:\; (x\;(\lambda y:\; x\;(y\;x))$ e $\lambda w:\; (w\;(\lambda z:\; w\;(z\;w))$ serão considerados equivalentes. No cálculo lambda está definido um conjunto de regras, redução-$\alpha$ que determina como podemos renomear variáveis.

**Exemplo**:

$$
FV\left((\lambda x: yx)z\right) = \{y, z\}, \quad BV\left((\lambda x: yx)z\right) = \{x\}
$$

e

$$
FV\left((\lambda xy: yx)zw\right) = \{z, w\}, \quad BV\left((\lambda xy: yx)zw\right) = \{x, y\}.
$$

A amável leitora deve entender o conceito de variáveis livres e ligadas observando uma convenção importante no cálculo lambda que diz que podemos renomear variáveis ligadas, _Bound Variables_, sem alterar o significado do termo, desde que não capturemos variáveis livres, _Free Variables_, durante o processo de renomeação. **Esta operação é chamada de redução-$\alpha$** e é estudada com mais fervor em outra parte do livro. Neste momento, podemos dizer que essa renomeação não deve alterar o comportamento ou o significado da função, desde que seja feita com cuidado evitando a captura de variáveis livres. A afoita leitora pode avaliar os exemplos a seguir:

**Exemplo 1**: renomeação segura de variáveis ligadas. Considere a expressão:

$$\lambda x.\lambda y.x\;y$$

Nesta expressão, temos duas abstrações aninhadas. A primeira, $\lambda x\,$, define uma função que recebe $x$ como argumento. A segunda, $\lambda y\,$, define uma função que recebe $y\,$. O termo $x\;y$ é a aplicação de $x$ ao argumento $y\,$. Este termo pode ser visto na árvore sintática a seguir:

$$
\begin{array}{c}
\lambda x \\
\downarrow \\
\lambda $Y$ \\
\downarrow \\
@ \\
\diagup \quad \diagdown \\
x \quad \quad \quad y
\end{array}
$$

A observadora leitora já deve ter percebido que podemos realizar uma **redução-$\alpha$** para renomear as variáveis ligadas sem alterar o significado da expressão. Como não há variáveis livres aqui, podemos renomear $x$ para $z$ e $y$ para $w$:

$$\lambda x.\lambda y.x\;y \to_\alpha \lambda z.\lambda w.z\;w$$

As variáveis ligadas $x$ e $y$ foram renomeadas para $z$ e $w\,$, respectivamente, mas o significado da função permanece o mesmo: ela ainda aplica o primeiro argumento ao segundo. Este é um exemplo de renomeação correta, sem captura de variáveis livres.

**Exemplo 2**: problema de captura de variáveis livres. Para entender este problema, vejamos o segundo exemplo:

$$\lambda x.\;x\;y \neq_\alpha \lambda y.\;y\;y$$

No primeiro termo, $y$ é uma variável livre, ou seja, não está ligada por uma abstração $\lambda$ dentro da expressão e pode representar um valor externo. Se tentarmos renomear $x$ para $y\,$, acabamos capturando a variável livre $y$ em uma abstração. No segundo termo, $y$ se torna uma variável ligada dentro da abstração $\lambda y\,$, o que altera o comportamento do termo. O termo original dependia de $y$ como uma variável livre, mas no segundo termo, $y$ está ligada e aplicada a si mesma:

$$\lambda x.\;x\;y \neq_\alpha \lambda y.\;y\;y$$

No termo original, $y$ poderia ter um valor externo fornecido de outro contexto. No termo renomeado, $y$ foi capturada e usada como uma variável ligada, o que altera o comportamento do termo. Este é um exemplo de renomeação incorreta por captura de uma variável livre, mudando o significado do termo original.

## 1.6. Exercícios
  
**1**: Escreva uma função lambda para representar a identidade, que retorna o próprio argumento.

   **Solução**: a função identidade é simplesmente: $\lambda x.\;x\,$, essa função retorna o argumento $x\,$.

**2**: Escreva uma função lambda que representa uma constante, sempre retornando o número $5\,$, independentemente do argumento.

   **Solução**: a função constante pode ser representada por: $\lambda x.\;5\,$, uma função que sempre retorna $5\,$, independentemente de $x\,$.

**3**: Dado $\lambda x.\;x + 2\,$, aplique a função ao número $3\,$.

   **Solução**: substituímos $x$ por $3$ e teremos: $(\lambda x.\;x + 2)\;3 = 3 + 2 = 5$

**4**: Simplifique a expressão $(\lambda x. \lambda y.\;x)(5)(6)\,$.

   **Solução**: primeiro, aplicamos a função ao valor $5\,$, o que resulta na função $\lambda y. 5\,$. Agora, aplicamos essa nova função ao valor $6$:

   $$(\lambda y. 5)\, 6 = 5$$

   O resultado é $5\,$.

**5**: Simplifique a expressão $(\lambda x.\;x)(\lambda y.\;y)\,$.

   **Solução**: aplicamos a função $\lambda x.\;x$ à função $\lambda y.\;y$:

   $$(\lambda x.\;x)(\lambda y.\;y) \, = \lambda y.\;y$$

   A função $\lambda y.\;y$ é a identidade e o resultado é a própria função identidade.

**6**: Aplique a função $\lambda x. \lambda y.\;x + y$ aos valores $3$ e $4\,$.

   **Solução**: aplicamos a função a $3$ e depois a $4$:

   $$(\lambda x. \lambda y.\;x + y)\;3 = \lambda y. 3 + y$$

   Agora aplicamos $4$:

   $$(\lambda y. 3 + y)\;4 = 3 + 4 = 7$$

   O resultado é $7\,$.

**7**: A função $\lambda x. \lambda y.\;x$ é uma função de primeira ordem ou segunda ordem?

   **Solução**: a função $\lambda x. \lambda y.\;x$ é uma função de segunda ordem, pois é uma função que retorna outra função.

**8**: Defina uma função lambda que troca a ordem dos argumentos de uma função de dois argumentos.

   **Solução**: essa função pode ser definida como:

   $$\lambda f. \lambda x. \lambda y. f\;y\;x$$

   Ela aplica a função $f$ aos argumentos $y$ e $x\,$, trocando a ordem.

**9**: Dada a função $\lambda x.\;x\;x\,$, por que ela não pode ser aplicada a si mesma diretamente?

   **Solução**: se aplicarmos $\lambda x.\;x\;x$ a si mesma, teremos:

   $$(\lambda x.\;x\;x)(\lambda x.\;x\;x)$$

   Isso resultaria em uma aplicação infinita da função a si mesma, o que leva a um comportamento indefinido ou a um erro de recursão infinita.

**10**: Aplique a função $\lambda x.\;x\;x$ ao valor $2\,$.

   **Solução**: substituímos $x$ por $2$:

   $$(\lambda x.\;x\;x)\;2 = 2 \times 2 = 4$$

   O resultado é $4\,$.

# 2. Sintaxe e Semântica

O cálculo lambda usa uma notação simples para definir e aplicar funções. Ele se baseia em três elementos principais: _variáveis, abstrações e aplicações_.

**As variáveis representam valores que podem ser usados em expressões. Uma variável é um símbolo que pode ser substituído por um valor ou outra expressão**. Por exemplo, $x$ é uma variável que pode representar qualquer valor.

**A abstração é a definição de uma função**. No cálculo lambda, uma abstração é escrita usando a notação $\lambda\,$, seguida de uma variável, um ponto e uma expressão. Por exemplo:

$$\lambda x.\;x^2 + 2x + 1$$

**Aqui, $\lambda x.$ indica que estamos criando uma função de $x$**. A expressão $x^2 + 2x + 1$ é o corpo da função. A abstração define uma função anônima que pode ser aplicada a um argumento.

**A aplicação é o processo de usar uma função em um argumento**. No cálculo lambda, representamos a aplicação de uma função a um argumento colocando-os lado a lado. Por exemplo, se tivermos a função $\lambda x.\;x + 1\;$ e quisermos aplicá-la ao valor $2\,$, escrevemos:

$$(\lambda x.\;x + 1)\;2$$

**O resultado da aplicação é a substituição da variável $x$ pelo valor $2\,$,** resultando em $2 + 1$ equivalente a $3\,$. Outros exemplos interessantes de função são a **função identidade**, que retorna o próprio valor e que é escrita como $\lambda x.\;x$ e uma função que some dois números e que pode ser escrita como $\lambda x. \lambda y.\;(x + y)\,$.

No caso da função que soma dois números, $\lambda x. \lambda y.\;(x + y)\,$, temos duas abstrações $\lambda x$ e $\lambda y\,$, cada uma com sua própria variável. Logo, $\lambda x. \lambda y.\;(x + y)$ precisa ser aplicada a dois argumentos. Tal como: $\lambda x. \lambda y.\;(x + y)\;3\;4\,$.

Formalmente dizemos que:

1. Se $x$ é uma variável, então $x$ é um termo lambda.

2. Se $M$ e $N$ são termos lambda, então $(M\; N)$ é um termo lambda chamado de aplicação.

3. Se $E$ é um termo lambda, e $x$ é uma variável, então a expressão $(λx. E)$ é um termo lambda chamado de abstração lambda.

Esses elementos básicos, _variáveis, abstração e aplicação_, formam a base do cálculo lambda. Eles permitem definir e aplicar funções de forma simples sem a necessidade de nomes ou símbolos adicionais.

## 2.1. Estrutura Sintática - Gramática

O cálculo lambda é um sistema formal para representar computação baseado na abstração de funções e sua aplicação. Sua sintaxe é simples, porém expressiva. Enfatizando a simplicidade. Tudo é uma expressão, ou termo, e existem três tipos de termos:

1. **Variáveis**: representadas por letras minúsculas como $x\,$, $y\,$, $z\,$. As variáveis não possuem valor intrínseco, como acontece nas linguagens imperativa. Variáveis atuam como espaços reservados para entradas potenciais de funções.

2. **Aplicação**: a aplicação $(M\;N)$ indica a aplicação da função $M$ ao argumento $N\,$. A aplicação é associativa à esquerda, então $M\;N\;P$ é interpretado como $((M\;N)\;P)\,$.

3. **Abstração**: a abstração $(\lambda x.\;E)$ representa uma função que tem $x$ como parâmetro e $E$ como corpo. O símbolo $\lambda$ indica que estamos definindo uma função. Por exemplo, $(\lambda x.\;x)$ é a função identidade.

**A abstração é a base do cálculo lambda**. Ela permite criar funções anonimas. **Um conceito importante relacionado à abstração é a distinção entre variáveis livres e ligadas**. Uma variável é **ligada** se aparece no escopo de uma abstração lambda que a define. Em $(\lambda x.\;x\;y)\,$, $x$ é uma variável ligada. Por outro lado, uma variável é **livre** se não estiver ligada a nenhuma abstração. No exemplo anterior, $y$ é uma variável livre.

A distinção entre variáveis livres e ligadas permitirá o entendimento da operação de substituição no cálculo lambda. A substituição é a base do processo de computação no cálculo lambda. O poder computacional do cálculo lambda está na forma como esses elementos simples podem ser combinados para expressar operações complexas como valores booleanos, estruturas de dados e até mesmo recursão usando esses os conceitos básicos, _variáveis, abstração e aplicação_, e a existência, ou não, de variáveis ligadas. Formalmente, podemos definir a sintaxe do cálculo lambda usando uma gramática representada usando sintaxe da [Forma de Backus-Naur](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form) (BNF):

$$
\begin{align*}
\text{termo} &::= \text{variável} \\
&\;|\;\text{constante} \\
&\;|\;\lambda . \text{variável}.\;\text{termo} \\
&\;|\;\text{termo}\;\text{termo} \\
&\;|\;(\text{termo})
\end{align*}
$$

A gentil leitora pode facilitar o entendimento de abstrações e aplicações se pensar em um termo lambda como sendo uma árvore, cuja forma corresponde à forma como o termo aplica as regras de produção da gramática. Chamamos a árvore criada pela derivação das regras de produção de de árvore sintática ou árvore de derivação. Para um dado termo $M\,$, qualquer, está árvore terá vértices rotulados por $\lambda x$ ou $@\,$, enquanto as folhas serão rotuladas por variáveis.

Indutivamente, podemos definir que a árvore de construção de uma variável $x$ é somente uma folha, rotulada por $x\,$. A árvore de construção de uma abstração $\lambda x.\;E$ consistirá em um vértice rotulado por $\lambda x$ com uma única subárvore, que é a árvore de construção de $E\,$. Por fim, a árvore de construção de uma aplicação $E\;N$ consistirá em um vértice rotulado por $@$ com duas subárvores: a subárvore esquerda é a árvore de construção de $E$ e a subárvore direita é a árvore de construção de $N\,$. Por exemplo, a árvore de construção do termo $\lambda x \lambda y.\;x\;y\;\lambda z.\;y\;z$ será:

$$
\begin{array}{c}
\lambda x \\
\downarrow \\
\lambda Y \\
\downarrow \\
\diagup \quad \diagdown \\
\begin{array}{cc}
\quad x \quad & \quad \lambda z \quad \\
& \downarrow \\
& \begin{array}{cc}
\;y &\;z
\end{array}
\end{array}
\end{array}
$$

Neste texto, vamos dar prioridade a derivação gramatical, deixando as árvores para suporte a explicação das abstrações. Pelo sim, pelo não, vamos ver mais dois exemplos.

**Exemplo 1**: Representação da abstração $\lambda x.\;x\;x$

Antes de vermos a árvore, podemos analisar a estrutura do termo $\lambda x.\;x\;x\,$. Nesta expressão, o termo $\lambda x$ indica que $x$ é o parâmetro da função e o corpo da função é $x\;x\,$, a aplicação de $x$ a si mesmo.

Agora que a curiosa leitora entendeu a expressão construir a árvore.

$$
\begin{array}{c}
\lambda x \\
\downarrow \\
\begin{array}{c}
@ \\
\diagup \quad \diagdown \\
x \quad \quad \quad x
\end{array}
\end{array}
$$

Esta árvore é composta de um vértice raiz, no topo, $\lambda x\,$, indicando a abstração de $x\,$. Logo em seguida, a leitora pode ver o vértice de aplicação $@$ no meio da árvore representando que $x$ está sendo aplicado a $x\,$. Finalmente, as folhas da árvore são as variáveis $x$ à esquerda e à direita do vértice de aplicação, correspondendo às duas ocorrências de $x$ no corpo da função.

**Exemplo 2**: Representação da aplicação $(\lambda x.\;x + 1)\;2$

Outra vez podemos começar com a estrutura do termo $(\lambda x.\;x + 1)\;2\,$. A expressão $\lambda x.\;x + 1$ define uma função que recebe $x$ como argumento e retorna $x + 1\,$. O termo $2$ é o argumento que é passado para a função. Consequentemente, a aplicação $(\lambda x.\;x + 1)\;2$ envolve a substituição de $x$ por $2$ no corpo da função, o que resultará na expressão $2 + 1\,$. Esta função é representada pela árvore:

$$
\begin{array}{c}
@ \\
\diagup \quad \diagdown \\
\begin{array}{c}
\lambda x & \quad 2 \\
\downarrow \\
x + 1
\end{array}
\end{array}
$$

A árvore sintática representa a aplicação de uma função a um argumento. O vértice de aplicação é representado por $@$ e suas subárvores mostram a função e o argumento.

Em detalhes, temos: o vértice de aplicação, $@\,$, no topo da árvore indica uma aplicação. Não poderia ser diferente já que a aplicação de uma função a um argumento é sempre representada por esse vértice. O galho à esquerda, e abaixo do vértice de aplicação, representa $\lambda x.\;x + 1\,$. Essa parte da árvore representa a função que toma $x$ como parâmetro e retorna $x + 1\,$. Abaixo de $\lambda x\,$, está o corpo da função, $x + 1\,$, que mostra como o argumento $x$ é manipulado. No galho a direita do vértice de aplicação está o valor $2\,$. Este é o argumento que é aplicado a função. Ou, sendo mais claro, o valor $2$ substituirá $x$ no corpo da função. Por último, temos o corpo da função no nível mais baixo, sob $\lambda x\,$, vemos o corpo da função $x + 1\,$. No processo de redução, esse termo é avaliado como $2 + 1\,$, resultando em $3\,$.

Como vimos, a gramática é simples, poucas regras de produção e poucos símbolos. Entretanto, a combinação destas regras pode criar termos complexos. Para facilitar o entendimento a semântica do cálculo lambda pode ser dividida em **semântica operacional** e **semântica denotacional**. A semântica denotacional especifica o significado dos termos da linguagem enquanto a semântica operacional especifica o que acontece quado estes termos são executados. A esperta leitora deve estar pensando que a semântica operacional é mais direta já que, podemo relacionar a semântica operacional com a forma como as expressões, ou uma linguagem formal deve funcionar.

## 2.2. Semântica Operacional

A semântica operacional é uma abordagem rigorosa para descrever o comportamento de linguagens formai, especificando como as expressões de uma linguagem são avaliadas. No caso de linguagens de programação é a semântica operacional que define como os programas irão funcionar.

No contexto do cálculo lambda, a semântica operacional se concentra em como os termos são transformados por meio de uma sequência de reduções. As reduções operam sobre a estrutura sintática dos termos, permitindo a análise detalhada do processo de avaliação, desde a aplicação de funções até a substituição de variáveis.

Abaixo, são apresentadas as principais reduções operacionais utilizadas no cálculo lambda:

1. Redução Beta: A regra que define a ação de aplicação e chamada de _redução beta ou redução-$beta$_. Usamos a redução beta quando uma função é aplicada a um argumento. Neste caso, a redução beta substitui a variável ligada no corpo da função pelo argumento fornecido:

   $$(\lambda x.\;e_1)\;e_2\;\rightarrow\;e_1[x := e_2]$$

   Isso significa que aplicamos a função $\lambda x.\;e_1$ ao argumento $e_2\,$, substituindo $x$ por $e_2$ em $e_1\,$.

   **Exemplo**: considere o termo:

   $$(\lambda x.\;x^2)\;3\;\rightarrow\;3^2$$

   Existem duas estratégias para realização da redução beta:

   1. **Ordem normal**: reduzimos a aplicação mais à esquerda e mais externa primeiro. Essa estratégia sempre encontra a forma normal, se esta existir.

      **Exemplo**: considere o termo:

      $$(\lambda x.\;(\lambda y.\;y + x)\;2)\;(3 + 4)$$

      Não reduzimos $3 + 4$ imediatamente. Aplicamos a função externa:

      $$(\lambda x.\;(\lambda y.\;y + x)\;2)\;7$$

      Substituímos $x$ por $7$ em $(\lambda y.\;y + x)\;2$:

      $$(\lambda y.\;y + 7)\;2$$

      Aplicamos a função interna:

      $$2 + 7 \rightarrow 9$$

   2. **Ordem aplicativa**: avaliamos primeiro os subtermos (argumentos) antes de aplicar a função.

      **Exemplo:**

      $$(\lambda x.\;(\lambda y.\;y + x)\;2)\;(3 + 4)$$

      Avaliamos $3 + 4$:

      $$(\lambda x.\;(\lambda y.\;y + x)\;2)\;7$$

      Substituímos $x$ por $7$:

      $$(\lambda y.\;y + 7)\;2$$

      Avaliamos $2 + 7$:

      $$9$$

2. Redução alfa ou redução-$\alpha$: esta redução determina as regras que permitem renomear variáveis ligadas na esperança de evitar conflitos.

   **Exemplo**:

   $$\lambda x.\;x + 1 \rightarrow \lambda y.\;y + 1$$

3. Redução eta ou redução-$\eta$: esta redução define as regras de captura a equivalência entre funções que produzem os mesmos resultados.

   **Exemplo:**

   $$\lambda x.\;f(x) \rightarrow f$$

Essas regras garantem que a avaliação seja consistente. Por fim, mas não menos importante, o **Teorema de Church-Rosser** parece implicar que, **se uma expressão pode ser reduzida de várias formas então todas chegarão à mesma forma normal, se existir**[^cita5].

>No cálculo lambda, podemos dizer que um termo está em _forma normal_ quando não é possível realizar mais nenhuma redução beta sobre ele. Ou seja, é um termo que não contém nenhum _redex_, expressão redutível e, portanto, não pode ser simplificado ou reescrito de nenhuma outra forma. Formalmente: um termo $M$ está em forma normal se:
>
>$$\forall N \, : \, M \not\rightarrow N$$
>
>Isso significa que não existe nenhum termo $N$ tal que o termo $M$ possa ser reduzido a $N\,$.
>
>**No cálculo lambda, um termo pode não ter uma forma normal se o processo de redução continuar indefinidamente sem nunca alcançar um termo irredutível. Isso acontece devido à possibilidade de _loops_ infinitos ou recursões que não terminam. Os termos com esta característica são conhecidos como **termos divergentes**.

## 2.3. Substituição

A substituição é a operação estrutural do cálculo lambda. Ela funciona substituindo uma variável livre por um termo, e sua formalização evita a captura de variáveis, garantindo que ocorra de forma correta. A substituição é definida recursivamente:

1. $[N/x] x\;N$
2. $[N/x] y\;y, \quad \text{se}\;x \neq y$
3. $[N/x]\;(M_1 \, M_2) ([N/x]M_1)([N/x]M_2)$
4. $[N/x]\;(\lambda $Y$ \, M) \lambda $Y$ \, ([N/x]M), \quad \text{se} ; x \neq $Y$ \quad \text{e} \quad $Y$ \notin FV(N)$

Aqui, $FV(N)$ é o conjunto de variáveis livres, _Free Variable_ de $N\,$. A condição $y \notin FV(N)$ é necessária para evitar a captura de variáveis livres.

Formalmente dizemos que: para qualquer termo lambda $M$, o conjunto $FV(M)$ de variáveis livres de $M$ e o conjunto $BV(M)$ de variáveis ligadas em $M$ serão definidos de forma indutiva:

1. Se $M = x$ (uma variável), então:
   - $FV(x) = \{x\}$
   - $BV(x) = \emptyset$

2. Se $M = (M_1 M_2)$, então:
   - $FV(M) = FV(M_1) \cup FV(M_2)$
   - $BV(M) = BV(M_1) \cup BV(M_2)$

3. Se $M = (\lambda x: M_1)$, então:
   - $FV(M) = FV(M_1) \setminus \{x\}$
   - $BV(M) = BV(M_1) \cup \{x\}$

Se $x \in FV(M_1)$, dizemos que as ocorrências da variável $x$ ocorrem no escopo de $\lambda$. **Um termo lambda $M$ é fechado se $FV(M) = \emptyset$, ou seja, se não possui variáveis livres**.

A atenta leitora não deve perder de vista **que as variáveis ligadas são unicamente marcadores de posição**, de modo que elas podem ser renomeadas livremente sem alterar o comportamento deste termo durante a substituição, desde que não entrem em conflito com as variáveis livres. Por exemplo, os termos $\lambda x:\;(x(\lambda y:\;x(y\;x))$ e $\lambda x:\;(x(\lambda z: x\;(z\;x))$ devem ser considerados equivalentes. Da mesma forma, os termos $\lambda x:\; (x\;(\lambda y:\; x\;(y\;x))$ e $\lambda w:\; (w\;(\lambda z:\; w\;(z\;w))$ devem ser considerados equivalentes.

**Exemplo**:

$$
FV\left((\lambda x: yx)z\right) = \{y, z\}, \quad BV\left((\lambda x: yx)z\right) = \{x\}
$$

e

$$
FV\left((\lambda xy: yx)zw\right) = \{z, w\}, \quad BV\left((\lambda xy: yx)zw\right) = \{x, y\}.
$$

Podemos pensar na substituição como um processo de _buscar e substituir_ em uma expressão, mas com algumas regras especiais. Lendo estas regras em bom português teríamos:

- A regra 1 (**Regra de Substituição Direta**): $[N/x]\,x = N$ indica que a variável $x$ será substituída pelo termo $N\,$. **Esta é a regra fundamenta a substituição**. De forma mais intuitiva podemos dizer que esta regra significa que se encontrarmos exatamente a variável que estamos procurando, substituímos por nosso novo termo. Por exemplo, em $[3/x]\,x\,$, substituímos $x$ por $3\,$.

- A regra 2 (**Regra de Variável Livre**): $[N/x]\,y = y\,$, se $x \neq y\,$, está correta ao indicar que as variáveis que não são $x$ permanecem inalteradas. Ou seja, se durante a substituição de uma variável encontramos uma variável diferente, deixamos como está. Por exemplo: na substituição $[3/x]\,y\,$, $y$ permanece $y$

- A regra 3 (**Regra de Distribuição da Substituição**): $[N/x]\;(M_1\;M_2)\,=\,([N/x]M_1)([N/x]M_2)$ define corretamente a substituição em uma aplicação de termos. O que quer dizer que, se estivermos substituindo em uma aplicação de função, fazemos a substituição em ambas as partes. Por exemplo: em $[3/x]\;(x\;y)\,$, substituímos em $x$ e $y$ separadamente, resultando em $(3\;y)\,$.

- A regra 4 (**Regra de Evitação de Captura de Variáveis**): $[N/x]\;(\lambda y.\;M) \, = \lambda y.\;([N/x]M)\,$, se $x \neq y$ e $y \notin FV(N)\,$, está bem formulada, indicando que a variável vinculada $y$ não será substituída se $x \neq y$ e $y$ não estiverem no conjunto de variáveis livres de $N\,$, o que evita a captura de variáveis. Em uma forma mais intuitiva podemos dizer que se encontrarmos uma abstração lambda, temos que ter cuidado: se a variável ligada for a mesma que estamos substituindo, paramos; se for diferente, substituímos no corpo, mas só se for seguro (sem captura de variáveis). Por exemplo: em $[3/x]\;(\lambda y.\;x)\,$, substituímos $x$ no corpo, resultando em $\lambda y.\;3\,$.

Para que a esforçada leitora possa fixar o entendimento destes conceitos, considere o seguinte exemplo:

$$[y/x]\;(\lambda y.\;x) \neq \lambda y.\;y$$

Se realizarmos a substituição diretamente, a variável livre $y$ será capturada, alterando o significado do termo original. Para evitar isso, utilizamos a **substituição com evasão de captura**. Isto é feito com a aplicando a redução-$\alpha$ para as variáveis ligadas que possam causar conflito. Considere:

$$[y/x]\;(\lambda y.\;x) \, = \lambda z.\;[y/x]\;([z/y]x) \, = \lambda z.\;y$$

Neste processo, a variável ligada $y$ foi renomeada como $z$ antes de realizar a substituição, evitando assim a captura da variável livre $y\,$.

Outro exemplo ilustrativo:

$$[z/x]\;(\lambda z.\;x) \neq \lambda z.\;z$$

Se fizermos a substituição diretamente, a variável $z$ livre em $x$ será capturada pela abstração $\lambda z\,$, modificando o significado do termo. A solução correta é renomear a variável ligada antes da substituição:

$$[z/x]\;(\lambda z.\;x) \, = \lambda w.\;[z/x]\;([w/z]x) \, = \lambda w.\;z$$

Este procedimento assegura que a variável livre $z$ em $x$ não seja capturada pela abstração $\lambda z\,$, preservando o significado original do termo.

**Exemplo 1**: Substituição direta sem captura de variável livre

   $$[a/x]\;(x + y) \, = a + y$$

   Neste caso, substituímos a variável $x$ pelo termo $a$ na expressão $x + y\,$, resultando em $a + y\,$. Não há risco de captura de variáveis livres, pois $y$ não está ligada a nenhuma abstração e permanece livre na expressão resultante.

**Exemplo 2**: Substituição direta mantendo variáveis livres

   $$[b/x]\;(x\;z) \, = b\;z$$

   Aqui, substituímos $x$ por $b$ na expressão $x\;z\,$, obtendo $b\;z\,$. A variável $z$ permanece livre e não ocorre captura, pois não está sob o escopo de nenhuma abstração lambda que a ligue.

**Exemplo 3**: Evasão de captura com renomeação de variável ligada

   $$[y/x]\;(\lambda y.\;x) \, = \lambda z.\;[y/x]\;([z/y]x) \, = \lambda z.\;y$$

   Neste exemplo, se realizássemos a substituição diretamente, a variável livre $y$ em $x$ seria capturada pela abstração $\lambda y\,$, alterando o significado da expressão. Para evitar isso, seguimos os passos:

   1. **Renomeação (Redução Alfa)**: Renomeamos a variável ligada $y$ para $z$ na abstração, obtendo $\lambda z.\, [z/y]x\,$.

   2. **Substituição**: Aplicamos $[y/x]\;(x)\,$, resultando em $y\,$.

   3. **Resultado Final**: A expressão torna-se $\lambda z.\;y \,$, onde $y$ permanece livre.

   Evitamos a captura da variável livre $y$ pela abstração lambda.

**Exemplo 4**: Evasão de captura para preservar o significado da expressão

   $$[w/x]\;(\lambda w.\;x) \, = \lambda v.\;[w/x]\;([v/w]x) \, = \lambda v.\;w$$

   Neste caso, a substituição direta capturaria a variável livre $w$ em $x\,$. Para prevenir isso:

   1. **Renomeação (Redução Alfa)**: Renomeamos a variável ligada $w$ para $v\,$, obtendo $\lambda v.\;[v/w]x\,$.

   2. **Substituição**: Aplicamos $[w/x]\;(x)\,$, resultando em $w\,$.

   3. **Resultado Final**: A expressão fica $\lambda v.\;w\,$, mantendo $w$ como variável livre.

Assim, garantimos que a variável livre $w$ não seja capturada, preservando o significado original da expressão.

Ao aplicar a **evasão de captura** por meio da renomeação de variáveis ligadas (redução-$\alpha$), asseguramos que as substituições não alterem o comportamento semântico das expressões, mantendo as variáveis livres intactas.

### 2.3.1. Relação da Substituição com Outros Conceitos do Cálculo Lambda

A substituição não é um processo autônomo. Esta integrada às reduções $\alpha$ e $\beta\,$, ao conceito de variáveis ligadas, ao conceito de recursão e ponto fixo. A atenta leitora já deve ter percebido que sem esta integração a substituição não seria possível, ou não possibilitaria as reduções.

Começando com a redução-$beta$: a redução $beta$ utiliza a substituição como parte de seu mecanismo. A regra de redução beta é definida como:

$$(\lambda x.M)N \to_\beta [N/x]M$$

Neste caso, $[N/x]M$ indica a substituição de todas as ocorrências livres de $x$ em $M$ por $N\,$. Por exemplo:

$$(\lambda x.x + 2)\;3 \to_\beta [3/x]\;(x + 2) \, = 3 + 2$$

A computação, ou avaliação de expressões, envolve uma série de reduções beta, cada uma requerendo substituições. Considere a expressão:

$$(\lambda x.\lambda y.x + y)\;3\;4$$

 Sua avaliação procede assim:

$$(\lambda x.\lambda y.x + y)\;3\;4 \to_\beta (\lambda y.\;3 + y)\;4 \to_\beta 3 + 4 = 7$$

A redução-$\alpha\,$, por outro lado, é usada durante a substituição para evitar a captura de variáveis. Por exemplo:

$$[y/x]\;(\lambda y.\;x) \, = \lambda z.\;[y/x]\;([z/y]x) \, = \lambda z.\;y$$

Neste caso, a redução-$\alpha$ é usada para renomear $y$ para $z$ antes da substituição.

Finalmente, a substituição $[N/x]M\,$, regra primordial da substituição, afeta exclusivamente as ocorrências livres de $x$ em $M\,$. As ocorrências ligadas de $x$ em $M$ permanecem inalteradas. Esta é a relação entre a substituição e os conceitos de variáveis livres e ligadas.

Além destes conceitos que já vimos, a amável leitora, deve ter em mente que a substituição está relacionada com os conceitos de recursão e ponto fixo, que ainda não estudamos neste livro. De fato, a substituição é usada na definição de combinadores de ponto fixo, como o combinador Y:

$$Y = \lambda f. \, (\lambda x.\;f(x\;x))(\lambda x.\;f(x\;x))$$

A aplicação deste combinador envolve substituições que permitem a definição de funções recursivas.

A substituição interage com estes conceitos no funcionamento do cálculo lambda e em sua aplicação em linguagens de programação funcionais. A atenta leitora deve dedicar algum tempo para analisar o código em Haskell, a seguir.

A implementação da substituição em Haskell pode ajudar a concretizar os conceitos teóricos do cálculo lambda. Vamos analisar o código sugerido:

```haskell
data Expr = Var String | App Expr Expr | Lam String Expr
  deriving (Eq, Show)

-- Função de substituição que inclui a redução alfa para evitar captura
substitute :: String -> Expr -> Expr -> Expr
substitute x n (Var y)
  | x == $Y$    = n
  | otherwise = Var y
substitute x n (App e1 e2) \, = App (substitute x n e1) (substitute x n e2)
substitute x n (Lam $Y$ e)
  | $Y$ == x = Lam $Y$ e  -- Variável ligada é a mesma que estamos substituindo
  | $Y$ `elem` freeVars n =  -- Risco de captura, aplicar redução alfa
      let y' = freshVar $Y$ (n : e : [])
          e' = substitute $Y$ (Var y') e
      in Lam y' (substitute x n e')
  | otherwise = Lam $Y$ (substitute x n e)

-- Função para obter as variáveis livres em uma expressão
freeVars :: Expr -> [String]
freeVars (Var x) = [x]
freeVars (App e1 e2) = freeVars e1 ++ freeVars e2
freeVars (Lam x e) = filter (/= x) (freeVars e)

-- Função para gerar um novo nome de variável que não cause conflitos
freshVar :: String -> [Expr] -> String
freshVar x exprs = head $ filter (`notElem` allVars) candidates
  where
    allVars = concatMap (\e -> freeVars e ++ boundVars e) exprs
    candidates = [x ++ show n | n <- [1..]]

-- Função para obter as variáveis ligadas em uma expressão
boundVars :: Expr -> [String]
boundVars (Var _) \, = []
boundVars (App e1 e2) \, = boundVars e1 ++ boundVars e2
boundVars (Lam x e) \, = x : boundVars e
```

Este código possui algumas características que requerem a nossa atenção. Começando com a definição de tipos:

```haskell
data Expr = Var String | App Expr Expr | Lam String Expr
  deriving (Eq, Show)
```

Esta linha define, em Haskell, o tipo de dados `Expr` que representa expressões do cálculo lambda: `Var String`: representa uma variável; `App Expr Expr`: representa a aplicação de uma função e `Lam String Expr`: representa uma abstração lambda.

A seguir, no código, temos a assinatura e a definição da função de substituição que inclui a redução-$\alpha$:

```haskell
substitute :: String -> Expr -> Expr -> Expr
```

A função `substitute` implementa a substituição $[N/x]M\,$. Ela recebe três argumentos: a variável a ser substituída (`x`); o termo substituto (`n`) e a expressão na qual fazer a substituição (`Expr`).

Agora, que definimos a assinatura da função `substitute` vamos analisar cada um dos seus casos:

1. **Substituição em Variáveis**:

   ```haskell
   substitute x n (Var y)
     | x == $Y$    = n
     | otherwise = Var y
   ```

   Se a variável `y` é a mesma que estamos substituindo (`x`), retornamos o termo substituto `n`. Isto corresponde à **regra 1** da substituição formal. Caso contrário, mantemos a variável original `y` inalterada, conforme a regra 2.

2. **Substituição em Aplicações**:

   ```haskell
   substitute x n (App e1 e2) \, = App (substitute x n e1) (substitute x n e2)
   ```

   Aplicamos a substituição recursivamente em ambos os termos da aplicação. Isto reflete a regra 3 da substituição formal.

3. **Substituição em Abstrações Lambda**:

   ```haskell
   substitute x n (Lam $Y$ e)
     | $Y$ == x = Lam $Y$ e  -- Variável ligada é a mesma que estamos substituindo
     | $Y$ `elem` freeVars n =  -- Risco de captura, aplicar redução alfa
         let y' = freshVar $Y$ (n : e : [])
             e' = substitute $Y$ (Var y') e
         in Lam y' (substitute x n e')
     | otherwise = Lam $Y$ (substitute x n e)
   ```

   Este é o caso mais complexo e corresponde à **regra 4** da substituição formal. Aqui, temos três subcasos:

   1. Se a variável ligada `y` é a mesma que estamos substituindo (`x`). Não fazemos nada, pois `x` está _sombreada_ pela ligação de `y`.

   2. Se `y` está nas variáveis livres do termo substituto `n`: existe o risco de **captura de variável livre**. Para evitar isso, aplicamos a **redução-$\alpha$**, renomeando `y` para um novo nome `y'` que não cause conflito. Utilizamos a função `freshVar` para gerar um novo nome que não esteja nas variáveis livres ou ligadas das expressões envolvidas. Realizamos a substituição no corpo `e` após a renomeação.

   3. Caso contrário: substituímos recursivamente no corpo da abstração `e`, mantendo `y` inalterado.

4. **Função para Variáveis Livres**:

   ```haskell
   freeVars :: Expr -> [String]
   freeVars (Var x) \, = [x]
   freeVars (App e1 e2) \, = freeVars e1 ++ freeVars e2
   freeVars (Lam x e) \, = filter (/= x) (freeVars e)
   ```

   Esta função calcula o conjunto de variáveis livres em uma expressão, essencial para evitar a captura de variáveis durante a substituição.

5. **Função para Gerar Novos Nomes de Variáveis**:

   ```haskell
   freshVar :: String -> [Expr] -> String
   freshVar x exprs = head $ filter (`notElem` allVars) candidates
     where
       allVars = concatMap (\e -> freeVars e ++ boundVars e) exprs
       candidates = [x ++ show n | n <- [1..]]
   ```

   A função `freshVar` gera um novo nome de variável (`y'`) que não está presente em nenhuma das variáveis livres ou ligadas das expressões fornecidas.

   Isso é crucial para a redução-$\alpha\,$, garantindo que o novo nome não cause conflitos.

6. **Função para Variáveis Ligadas**:

   ```haskell
   boundVars :: Expr -> [String]
   boundVars (Var _) \, = []
   boundVars (App e1 e2) \, = boundVars e1 ++ boundVars e2
   boundVars (Lam x e) \, = x : boundVars e
   ```

   Esta função auxilia `freshVar` ao fornecer o conjunto de variáveis ligadas em uma expressão.

Implementando a redução-$\alpha$ no código, conseguimos evitar a captura de variáveis livres durante a substituição, conforme ilustrado nos exemplos anteriores. Vamos ver um exemplo de evasão de captura com renomeação de variável ligada. Considere o termo:

$$[y/x]\;(\lambda y.\;x) \, = \lambda z.\;y$$

No código Haskell, este caso seria processado da seguinte forma:

   1. **Detectar o Risco de Captura**: a variável ligada `y` está presente nas variáveis livres do termo substituto `n` (que é `y`). Portanto, precisamos aplicar a redução-$\alpha\,$.

   2. **Aplicar redução-$\alpha$**: utilizamos `freshVar` para gerar um novo nome, digamos `z`. Renomeamos `y` para `z` na abstração, e substituímos `y` por `z` no corpo.

   3. **Realizar a Substituição**: substituímos `x` por `y` no corpo renomeado.

   4. **Resultado Final**: a expressão resultante é `\lambda z.\;y`, onde `y` permanece livre.

Neste ponto, se a amável leitora se perdeu no Haskell, deve voltar as definições formais da substituição e tentar fazer o paralelo entre as definições formais e o código em Haskell. A importância desta implementação está na demonstração de como os conceitos teóricos do cálculo lambda podem ser traduzidos para código executável, fornecendo uma ponte entre a teoria e a prática.

### 2.3.2. Exercícios de Substituição

**1**: Realize a substituição $[3/x]\;(x + y)\,$.

   **Solução**:

   1. Começando observando que a função de substituição indica que estamos substituindo $x$ por $3$ na expressão $x + y\,$.

   2. Aplicamos a regra formal 3 da substituição:

      $$[N/x]\;(M_1 M_2) \, = ([N/x]M_1)([N/x]M_2)$$

      Neste caso, $M_1 = x\,$, $M_2 = y\,$, e a operação $+$ é tratada como uma aplicação.

   3. Substituímos em $M_1$: $[3/x]\,x = 3$ (pela regra 1).

   4. Substituímos em $M_2$: $[3/x]\,y = y$ (pela regra 2, pois $x \neq y$).

   5. Reconstruímos a expressão: $(3) + (y)\,$.

   O resultado de $[3/x] \ ,(x + y)$ é $3 + y\,$. Ou seja, a variável $x$ foi substituída por $3\,$, enquanto $y$ permaneceu inalterada por ser uma variável diferente de $x\,$.

**2**: Realize a substituição $[(\lambda z.\;z)/x]\;(\lambda y.\;x\;y)\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $(\lambda z.\;z)$ na expressão $\lambda y.\;x\;y\,$.

   2. Começamos aplicando a regra formal 4 da substituição:

      $$[N/x]\;(\lambda y.\;M) \, = \lambda y.\;([N/x]M)$$

      pois $x \neq y$ e $y \notin FV((\lambda z.\;z))$$

   3. Agora mudamos o foco para a substituição dentro do corpo da abstração:

      $$[(\lambda z.\;z)/x]\,(x\;y)$$

   4. Aplicamos a regra 3 para a aplicação dentro do corpo:

      $$([(\lambda z.\;z)/x]\,x)([(\lambda z.\;z)/x]\,y)$$

   5. Resolvemos cada parte:
      - $[(\lambda z.\;z)/x]\,x = \lambda z.\;z$ (pela regra 1)
      - $[(\lambda z.\;z)/x]\,y = y$ (pela regra 2, pois $x \neq y$)

   6. Reconstruímos a expressão: $\lambda y.\;((\lambda z.\;z) y)$

   O resultado da substituição $[(\lambda z.\;z)/x]\;(\lambda y.x\;y)$ é $\lambda y.((\lambda z.\;z) y)\,$. Neste caso, a ocorrência livre de $x$ no corpo da abstração foi substituída por $(\lambda z.\;z)\,$. A variável $y$ permaneceu ligada e não foi afetada pela substituição.

**3**: Realize a substituição $[y/x]\;(\lambda y.\;x)\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $y$ na expressão $\lambda y.x\,$. Este é um caso onde precisamos ter cuidado com a captura de variáveis.

   2. Não podemos aplicar diretamente a regra 4, pois $y \in FV(y)\,$. Para evitar a captura de variáveis, realizamos uma redução-$\alpha$ primeiro: $\lambda y.x \to_\alpha \lambda z.x\,$.

   3. Agora podemos aplicar a substituição com segurança: $[y/x]\;(\lambda x.\;z\,$.

   4. Aplicamos a regra 4: $[y/x]\;(\lambda x.\;z = \lambda z.\;([y/x]\,x)$

   5. Resolvemos a substituição no corpo: $\lambda z.\;([y/x]\,x) \, = \lambda z.\;y$ (pela regra 1)

   O resultado de $[y/x]\;(\lambda y.\;x)$ é $\lambda z.\;y\,$. Para evitar a captura da variável livre $y$ que estamos introduzindo, primeiro renomeamos a variável ligada $y$ para $z\,$, redução-$\alpha\,$. Depois, realizamos a substituição normalmente, resultando em uma abstração que retorna a variável livre $y\,$.

**4**: Realize a substituição $[(\lambda x.\;x\;x)/y] \ ,(y\;z)\,$.

**Solução**:

   1. Estamos substituindo $y$ por $(\lambda x.\, x\;x)$ na expressão $y\;z\,$. Este é um caso de substituição em uma aplicação.

   2. Aplicamos a regra 3: $[(\lambda x.\, x\;x)/y] \ ,(y\;z) \, = ([(\lambda x.\, x\;x)/y]\,y)([(\lambda x.\, x\;x)/y]z)$

   3. Resolvemos a primeira parte: $[(\lambda x.\, x\;x)/y]\,y = (\lambda x.\, x\;x)$ (pela regra 1)

   4. Resolvemos a segunda parte: $[(\lambda x.\, x\;x)/y]z = z$ (pela regra 2, pois $y \neq z$)

   5. Reconstruímos a expressão: $((\lambda x.\;x\;x) z)$

   O resultado de $[(\lambda x.\, x\;x)/y] \ ,(y\;z)$ é $((\lambda x.\, x\;x) z)\,$. A variável $y$ foi substituída pela abstração $(\lambda x.\, x\;x)\,$, enquanto $z$ permaneceu inalterado.

**5**: Realize a substituição $[a/x]\;(\lambda y.\lambda x.\;y\;x)\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $a$ na expressão $\lambda y.\lambda x.\;y\;x\,$. Temos uma abstração aninhada aqui.

   2. Aplicamos a regra 4 para a abstração externa:

      $$[a/x]\;(\lambda y.\lambda x.\;y\;x) \, = \lambda y.([a/x]\;(\lambda x.\;y\;x))$$

   3. Para a abstração interna, não precisamos substituir, pois a variável ligada $x$ _sombreia_ a substituição:  

      $$\lambda y.(\lambda x.\;y\;x)$$

   4. O resultado permanece inalterado.

   O resultado de $[a/x]\;(\lambda y.\lambda x.\;y\;x)$ é $\lambda y.\lambda x.\;y\;x\,$. A substituição não afetou a expressão devido ao sombreamento da variável $x$ na abstração interna.

**6**: Realize a substituição $[(\lambda z.\;z)/x]\;(x (\lambda y.\;y\;x))\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $(\lambda z.\;z)$ na expressão $x (\lambda y.\;y\;x)\,$. Esta é uma aplicação onde $x$ aparece livre duas vezes.

   2. Aplicamos a regra 3:

      $$[(\lambda z.\;z)/x]\;(x (\lambda y.\;y\;x)) \, = ([(\lambda z.\;z)/x]\,x) ([(\lambda z.\;z)/x]\;(\lambda y.\;y\;x))$$

   3. Resolvemos a primeira parte: $[(\lambda z.\;z)/x]\,x = (\lambda z.\;z)$ (pela regra 1)

   4. Para a segunda parte, aplicamos a regra 4:

      $$[(\lambda z.\;z)/x]\;(\lambda y.\;y\;x) \, = \lambda y.([(\lambda z.\;z)/x]\;(x\;y))$$

   5. Aplicamos a regra 3 novamente dentro da abstração:

      $$\lambda y.\;(([(\lambda z.\;z)/x]\,x)([(\lambda z.\;z)/x]\,y))$$

   6. Resolvemos: $\lambda y.((\lambda z.\;z)y)$

   7. Reconstruímos a expressão completa: $((\lambda z.\;z) (\lambda y.\;((\lambda z.\;z)\;y)))$

   O resultado de $[(\lambda z.\;z) / x] \ ,(x (\lambda y.x\;y))$ é $((\lambda z.\;z) (\lambda y.\;((\lambda z.\;z)y)))\,$. Todas as ocorrências livres de $x$ foram substituídas por $(\lambda z.\;z)\,$.

**7**: Realize a substituição $[y/x]\;(\lambda y.\;(\lambda x.\;y))\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $y$ na expressão $\lambda y.(\lambda x.\;y)\,$. Este caso requer atenção para evitar captura de variáveis.

   2. Aplicamos a regra 4 para a abstração externa. Como $y$ é a variável ligada e ao termo de substituição, precisamos fazer uma redução-$\alpha$ primeiro:

      $$\lambda y.(\lambda x.\;y) \to_\alpha \lambda z.\;(\lambda x.z)$$

   3. Agora podemos aplicar a substituição com segurança:

      $$[y/x]\;(\lambda z.\;(\lambda x.\;z))$$

   4. Aplicamos a regra 4: $\lambda z.([y/x]\;(\lambda x.\;z))$

   5. Para a abstração interna, não precisamos substituir, pois $x$ está ligado: $\lambda z.\;(\lambda x.\;z)$

   O resultado de $[y/x]\;(\lambda y.\;(\lambda x.\;y))$ é $\lambda z.\;(\lambda x.\;z)\,$. A redução-$\alpha$ foi necessária para evitar a captura da variável $y\,$, e a substituição não afetou o corpo interno devido à ligação de $x\,$.

**8**: Realize a substituição $[(\lambda x.\;y\;x)/z] \ ,(\lambda y.\;z\;y)\,$.

   **Solução**:

   1. Estamos substituindo $z$ por $(\lambda x.\;y\;x)$ na expressão $\lambda y.\;z\;y\,$. Temos que ter cuidado com a possível captura de variáveis.

   2. Aplicamos a regra 4:

      $$[(\lambda x.\;y\;x)/z] \ ,(\lambda y.\;z\;y) \, = \lambda y'.([(\lambda x.\;x\;y)/z] \ ,(zy'))$$

      Note que fizemos uma redução-$\alpha$ preventiva, renomeando $y$ para $y'$ para evitar possível captura.

   3. Agora aplicamos a regra 3 no corpo da abstração:

      $$\lambda y'.\;(([(\lambda x.\;y\;x)/z]z)([(\lambda x.\;x\;y)/z]y'))$$

   4. Resolvemos a primeira parte: $[(\lambda x.\;x\;y)/z]z = (\lambda x.\;x\;y)$ (pela regra 1)

   5. Resolvemos a segunda parte: $[(\lambda x.\;x\;y)/z]y' = y'$ (pela regra 2, pois $z \neq y'$)

   6. Reconstruímos a expressão: $\lambda y'.((\lambda x.\;x\;y)y')$

   O resultado de $[(\lambda x.\;x\;y)/z] \ ,(\lambda y.\;z\;y)$ é $\lambda y'.\;((\lambda x.\;x\;y)y')\,$. A redução-$\alpha$ preventiva evitou a captura de variáveis, e a substituição foi realizada corretamente no corpo da abstração.

**9**: Realize a substituição $[(\lambda x.\;x)/y] \ ,(\lambda x.\;y\;x)\,$.

   **Solução**:

   1. Estamos substituindo $y$ por $(\lambda x.\;x)$ na expressão $\lambda x.\;y\;x\,$. Precisamos ter cuidado com a variável ligada $x\,$.

   2. Aplicamos a regra 4:

      $$[(\lambda x.x)/y] \ ,(\lambda x.\;y\;x) \, = \lambda x'.\;([(\lambda x.\;x)/y] \ ,(yx'))$$

      Realizamos uma redução-$\alpha$ preventiva, renomeando $x$ para $x'\,$$

   3. Aplicamos a regra 3 no corpo da abstração: $\lambda x'.\;(([(\lambda x.\;x)/y]\,y)([(\lambda x.\;x)/y]x'))$

   4. Resolvemos a primeira parte: $[(\lambda x.\;x)/y]\,y = (\lambda x.\;x)$ (pela regra 1)

   5. Resolvemos a segunda parte: $[(\lambda x.\;x)/y]x' = x'$ (pela regra 2, pois $y \neq x'$)

   6. Reconstruímos a expressão: $\lambda x'.\;((\lambda x.\;x)\;x')$

   O resultado de $[(\lambda x.\;x)/y] \ ,(\lambda x.\;y\;x)$ é $\lambda x'.\;((\lambda x.\;x)x')\,$. A redução-$\alpha$ preventiva evitou conflitos com a variável ligada $x\,$, e a substituição foi realizada corretamente.

**10**: Realize a substituição $[(\lambda z.\;z\;w)/x]\;(\lambda y.\;\lambda w.\;x\;y\;w)\,$.

   **Solução**:

   1. Estamos substituindo $x$ por $(\lambda z.\;z\;w)$ na expressão $\lambda y.\lambda w.\;x\;y\;w\,$. Temos que considerar as variáveis ligadas $y$ e $w\,$.

   2. Aplicamos a regra 4 para a abstração externa:

      $$[(\lambda z.\;z\;w)/x]\;(\lambda y.\;\lambda w.x\;y\;w) \, = \lambda y.\;([(\lambda z.\;z\;w)/x]\;(\lambda w.\;x\;y\;w))$$

   3. Aplicamos a regra 4 novamente para a abstração interna:

      $$\lambda y.\lambda w'.\;([(\lambda z.\;z\;w)/x]\;(\;y\;xw'))$$

      Note que fizemos uma redução-$\alpha\,$, renomeando $w$ para $w'$ para evitar captura.

   4. Agora aplicamos a regra 3 no corpo da abstração mais interna:

      $$\lambda y.\lambda w'.\;(([(\lambda z.\;z\;w)/x]\,x)([(\lambda z.\;z\;w)/x]\,y)([(\lambda z.\;z\;w)/x]w'))$$

   5. Resolvemos cada parte:
      - $[(\lambda z.\;z\;w)/x]\,x = (\lambda z.\;z\;w)$ (pela regra 1)
      - $[(\lambda z.\;z\;w)/x]\,y = y$ (pela regra 2, pois $x \neq y$)
      - $[(\lambda z.\;z\;w)/x]w' = w'$ (pela regra 2, pois $x \neq w'$)

   6. Reconstruímos a expressão: $\lambda y.\lambda w'.\;((\lambda z.\;z\;w)\;y\;w')$

   O resultado de $[(\lambda z.\;z.\;w)/x]\;(\lambda y.\lambda w.\;x\;y\;w)$ é $\lambda y.\lambda w'.((\lambda z.\;z.\;w)\;y\;w')\,$. A redução-$\alpha$ preventiva na variável $w$ evitou a captura, e a substituição foi realizada corretamente, preservando a estrutura da abstração dupla.

## 2.4. Semântica Denotacional no Cálculo Lambda

A semântica denotacional é uma abordagem matemática para atribuir significados formais às expressões de uma linguagem formal, como o cálculo lambda.

Na semântica denotacional, cada expressão é mapeada para um objeto matemático que representa seu comportamento computacional. Isso fornece uma interpretação abstrata da computação, permitindo analisar e provar propriedades sobre programas com rigor.

No contexto do cálculo lambda, o domínio semântico é construído como um conjunto de funções e valores. O significado de uma expressão é definido por sua interpretação nesse domínio, utilizando um ambiente $\rho$ que associa variáveis a seus valores.

A interpretação denotacional é formalmente definida pelas seguintes regras:

1. **Variáveis**:

   $$[x]_\rho = \rho(x)$$

   O significado de uma variável $x$ é o valor associado a ela no ambiente $\rho\,$.Intuitivamente podemos entender esta regra como: quando encontramos uma variável $x\,$, consultamos o ambiente $\rho$ para obter seu valor associado.

   **Exemplo**: suponha um ambiente $\rho$ onde $\rho(x) \, = 5\,$.

   $$[x]_\rho = \rho(x) \, = 5$$

   Assim, o significado da variável $x$ é o valor $5$ no ambiente atual.

2. **Abstrações Lambda**:

   $$[\lambda x.\;e]_\rho = f$$

   Onde $f$ é uma função tal que:

   $$f(v) \, = [e]_{\rho[x \mapsto v]}$$

   Isso significa que a interpretação de $\lambda x.\;e$ é uma função que, dado um valor $v\,$, avalia o corpo $e$ no ambiente onde $x$ está associado a $v\,$. Em bom português esta regra significa que uma abstração $\lambda x.\;e$ representa uma função anônima. Na semântica denotacional, mapeamos essa abstração para uma função matemática que, dado um valor de entrada, produz um valor de saída. Neste caso, teremos dois passos:

   1. **Definição da Função $f$**: A abstração é interpretada como uma função $f\,$, onde para cada valor de entrada $v\,$, calculamos o significado do corpo $e$ no ambiente estendido $\rho[x \mapsto v]\,$.

   2. **Ambiente Estendido**: O ambiente $\rho[x \mapsto v]$ é igual a $\rho\,$, exceto que a variável $x$ agora está associada ao valor $v\,$.

   **Exemplo**:

   Considere a expressão $\lambda x.\;x + 1\,$.

   Interpretação:

   $$[\lambda x.\;x + 1]_\rho = f$$

   Onde $f(v) \, = [x + 1]_{\rho[x \mapsto v]} = v + 1\,$.

   Significado: A abstração é interpretada como a função que incrementa seu argumento em 1.

3. **Aplicações**:

   $$[e_1\;e_2]_\rho = [e_1]_\rho\left([e_2]_\rho\right)$$

   O significado de uma aplicação $e_1\;e_2$ é obtido aplicando o valor da expressão $e_1$ (que deve ser uma função) ao valor da expressão $e_2\,$. Para interpretar uma aplicação $e_1\;e_2\,$, avaliamos ambas as expressões e aplicamos o resultado de $e_1$ ao resultado de $e_2\,$. Neste cenário temos três passos:

   1. **Avaliar $e_1$**: Obtemos $[e_1]_\rho\,$, que deve ser uma função.
  
   2. **Avaliar $e_2$**: Obtemos $[e_2]_\rho\,$, que é o argumento para a função.
  
   3. **Aplicar**: Calculamos $[e_1]_\rho\left([e_2]_\rho\right)\,$.

   **Exemplo**: considere a expressão $(\lambda x.\;x + 1)\;4\,$. Seguiremos três passos:

   **Passo 1**: Interpretar $\lambda x.\;x + 1\,$.

   $$[\lambda x.\;x + 1]_\rho = f, \quad \text{onde} \quad f(v) \, = v + 1$$

   **Passo 2**: Interpretar $4\,$.

   $$[4]_\rho = 4$$

   **Passo 3**: Aplicar $f$ a $4\,$.

   $$[(\lambda x.\;x + 1)\;4]_\rho = f(4) \, = 4 + 1 = 5$$

   A expressão inteira é interpretada como o valor $5\,$.

### 2.4.1. Ambiente $\rho$ e Associação de Variáveis

O ambiente $\rho$ armazena as associações entre variáveis e seus valores correspondentes. Especificamente, $\rho$ é uma função que, dado o nome de uma variável, retorna seu valor associado. Ao avaliarmos uma abstração, estendemos o ambiente com uma nova associação utilizando $[x \mapsto v]\,$.

**Exemplo de Atualização**:

- Ambiente inicial: $\rho = \{ $Y$ \mapsto 2 \}$

- Avaliando $\lambda x.\;x + y$ com $x = 3$:

- Novo ambiente: $\rho' = \rho[x \mapsto 3] = \{ $Y$ \mapsto 2, x \mapsto 3 \}$

- Avaliamos $x + y$ em $\rho'$:

$$[x + y]_{\rho'} = \rho'(x) + \rho'(y) \, = 3 + 2 = 5$$

A semântica denotacional facilita o entendimento do comportamento dos programas sem se preocupar com detalhes de implementação. Permite demonstrar formalmente que um programa satisfaz determinadas propriedades. Na semântica denotacional o significado de uma expressão complexa é construído a partir dos significados de suas partes.

A experta leitora deve concordar que exemplos, facilitam o entendimento e nunca temos o suficiente.

**Exemplo 1**: Com Variáveis Livres: considere a expressão $\lambda x.\;x + y\,$, onde $y$ é uma variável livre.

- Ambiente Inicial: $\rho = \{ $Y$ \mapsto 4 \}$
- Interpretação da Abstração:

$$
  [\lambda x.\;x + y]_\rho = f, \quad \text{onde} \quad f(v) \, = [x + y]_{\rho[x \mapsto v]} = v + 4
$$

- Aplicação: Avaliando $f(6)\,$, obtemos $6 + 4 = 10\,$.

**Exemplo 2**: Aninhamento de Abstrações. Considere $\lambda x.\;\lambda y.\;x + y\,$.

- Interpretação:

  - Primeiro, interpretamos a abstração externa:

   $$
   [\lambda x.\;\lambda y.\;x + y]_\rho = f, \quad \text{onde} \quad f(v) \, = [\lambda y.\;x + y]_{\rho[x \mapsto v]}
   $$

  - Agora, interpretamos a abstração interna no ambiente estendido:

   $$
   f(v) \, = g, \quad \text{onde} \quad g(w) \, = [x + y]_{\rho[x \mapsto v, $Y$ \mapsto w]} = v + w
   $$

- Aplicação:

  - Avaliando $((\lambda x.\;\lambda y.\;x + y)\;3)\;5$:

    - $f(3) \, = g\,$, onde $g(w) \, = 3 + w$
    - $g(5) \, = 3 + 5 = 8$

A semântica denotacional oferece um sistema matemático de atribuir significados às expressões do cálculo lambda. Ao mapear expressões para objetos matemáticos, valores e funções, podemos analisar programas de forma precisa e rigorosa. Entender essas regras permite uma compreensão mais profunda de como funções e aplicações funcionam no cálculo lambda.

Conceitos da semântica denotacional são fundamentais em linguagens funcionais modernas, como Haskell e OCaml.

Ferramentas baseadas em semântica denotacional podem ser usadas para verificar propriedades de programas, como terminação e correção.

Finalmente, a atenta leitora pode perceber que a semântica denotacional permite pensar em expressões lambda como funções matemáticas. Já a semântica operacional foca nos passos da computação.

>Observe que a **Semântica Operacional** é geralmente mais adequada para descrever a execução procedural de linguagens que usam passagem por referência, pois permite capturar facilmente como os estados mudam durante a execução. Por outro lado, a **Semântica Denotacional** é mais alinhada com linguagens puras, que preferem passagem por cópia, evitando efeitos colaterais e garantindo que o comportamento das funções possa ser entendido matematicamente.
>
>Existe uma conexão direta entre a forma como a semântica de uma linguagem é modelada e o mecanismo de passagem de valor que a linguagem suporta. Linguagens que favorecem efeitos colaterais tendem a ser descritas de forma mais natural por semântica operacional, enquanto aquelas que evitam efeitos colaterais são mais bem descritas por semântica denotacional.
>
>No caso do cálculo lambda, a semântica denotacional é preferida. O cálculo lambda é uma linguagem puramente funcional sem efeitos colaterais. A semântica denotacional modela suas expressões como funções matemáticas. Isso está em alinhamento com a natureza do cálculo lambda. Embora a semântica operacional possa descrever os passos de computação, a semântica denotacional fornece uma interpretação matemática abstrata adequada para linguagens que evitam efeitos colaterais.

## 2.5. Avaliação dos Exercícios de Semântica Denotacional

**1**: Dada a função lambda $\lambda x.\;x + 2\,$, aplique-a ao valor $5$ e calcule o resultado.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;x + 2$$

   2. Aplicando a função ao valor $5$:

      $$f(5) \, = (\lambda x.\;x + 2)\;5$$

   3. Substituir $x$ por $5$ no corpo da função:

      $$x + 2 \rightarrow 5 + 2$$

   4. Calculando o resultado:

      $$5 + 2 = 7$$

**2**: Escreva uma expressão lambda que represente a função $f(x, y) \, = x^2 + y^2\,$, e aplique-a aos valores $x = 3$ e $y = 4\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x^2 + y^2$$

   2. Aplicando a função aos valores $x = 3$ e $y = 4$:

      $$f(3)(4) \, = (\lambda x.\;\lambda y.\;x^2 + y^2)\;3\;4$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $3$:

      $$f(3) \, = \lambda y.\;3^2 + y^2$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $4$:

      $$f(3)(4) \, = 3^2 + 4^2$$

   5. Calculando o resultado:

         $$9 + 16 = 25$$

**3**: Crie uma expressão lambda para a função identidade $I(x) \, = x$ e aplique-a ao valor $10\,$.

   **Solução:**

   1. Definindo a função identidade:

      $$I = \lambda x.\;x$$

   2. Aplicando a função ao valor $10$:

      $$I(10) \, = (\lambda x.\;x)\;10$$

   3. Substituir $x$ por $10$ no corpo da função:

      $$x \rightarrow 10$$

   4. Resultado:

      $$10$$

**4**: Defina uma função lambda que aceita um argumento $x$ e retorna o valor $x^3 + 1\,$. Aplique a função ao valor $2\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;x^3 + 1$$

   2. Aplicando a função ao valor $2$:

      $$f(2) \, = (\lambda x.\;x^3 + 1)\;2$$

   3. Substituir $x$ por $2$ no corpo da função:

      $$2^3 + 1$$

   4. Calculando o resultado:

      $$8 + 1 = 9$$

**5**: Escreva uma função lambda que represente a soma de dois números, ou seja, $f(x, y) \, = x + y\,$, e aplique-a aos valores $x = 7$ e $y = 8\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x + y$$

   2. Aplicando a função aos valores $x = 7$ e $y = 8$:

      $$f(7)(8) \, = (\lambda x.\;\lambda y.\;x + y)\;7\;8$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $7$:

      $$f(7) \, = \lambda y.\;7 + y$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $8$:

      $$f(7)(8) \, = 7 + 8$$

   5. Calculando o resultado:

         $$15$$

**6**: Crie uma função lambda para a multiplicação de dois números, ou seja, $f(x, y) \, = x \times y\,$, e aplique-a aos valores $x = 6$ e $y = 9\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x \times y$$

   2. Aplicando a função aos valores $x = 6$ e $y = 9$:

      $$f(6)(9) \, = (\lambda x.\;\lambda y.\;x \times y)\;6\;9$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $6$:

      $$f(6) \, = \lambda y.\;6 \times y$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $9$:

      $$f(6)(9) \, = 6 \times 9$$

   5. Calculando o resultado:

      $$54$$

**7**: Dada a expressão lambda $\lambda x.\;\lambda y.\;x^2 + 2\;y\;x + y^2\,$, aplique-a aos valores $x = 1$ e $y = 2$ e calcule o resultado.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x^2 + 2\;y\;x + y^2$$

   2. Aplicando a função aos valores $x = 1$ e $y = 2$:

      $$f(1)(2) \, = (\lambda x.\;\lambda y.\;x^2 + 2\;y\;x + y^2)\;1\;2$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $1$:

      $$f(1) \, = \lambda y.\;1^2 + 2 \times 1 \times $Y$ + y^2$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $2$:

      $$f(1)(2) \, = 1^2 + 2 \times 1 \times 2 + 2^2$$

   5. Calculando o resultado:

      $$1 + 4 + 4 = 9$$

**8**: Escreva uma função lambda que aceite dois argumentos $x$ e $y$ e retorne o valor de $x - y\,$. Aplique-a aos valores $x = 15$ e $y = 5\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x - y$$

   2. Aplicando a função aos valores $x = 15$ e $y = 5$:

      $$f(15)(5) \, = (\lambda x.\;\lambda y.\;x - y)\;15\;5$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $15$:

      $$f(15) \, = \lambda y.\;15 - y$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $5$:

      $$f(15)(5) \, = 15 - 5$$

   5. Calculando o resultado:

         $$10$$

**9**: Defina uma função lambda que represente a divisão de dois números, ou seja, $f(x, y) \, = \dfrac{x}{y}\,$, e aplique-a aos valores $x = 20$ e $y = 4\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;\dfrac{x}{y}$$

   2. Aplicando a função aos valores $x = 20$ e $y = 4$:

      $$f(20)(4) \, = (\lambda x.\;\lambda y.\;\dfrac{x}{y})\;20\;4$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $20$:

      $$f(20) \, = \lambda y.\;\dfrac{20}{y}$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $4$:

      $$f(20)(4) \, = \dfrac{20}{4}$$

   5. Calculando o resultado:

      $$5$$

**10**: Escreva uma função lambda que calcule a função $f(x, y) \, = x^2 - y^2\,$, e aplique-a aos valores $x = 9$ e $y = 3\,$.

   **Solução:**

   1. Definindo a função lambda:

      $$f = \lambda x.\;\lambda y.\;x^2 - y^2$$

   2. Aplicando a função aos valores $x = 9$ e $y = 3$:

      $$f(9)(3) \, = (\lambda x.\;\lambda y.\;x^2 - y^2)\;9\;3$$

   3. Primeira aplicação: aplicamos $\lambda x$ ao valor $9$:

      $$f(9) \, = \lambda y.\;9^2 - y^2$$

   4. Segunda aplicação: aplicamos $\lambda y$ ao valor $3$:

      $$f(9)(3) \, = 9^2 - 3^2$$

   5. Calculando o resultado:

      $$81 - 9 = 72$$

# 3. Técnicas de Redução, Confluência e Combinadores

As técnicas de redução no cálculo lambda são mecanismos para simplificar e avaliar expressões lambda. Estas incluem a redução-$\alpha$ e a redução beta, que são utilizadas para manipular e computar expressões lambda. Essas técnicas são relevantes tanto para a teoria quanto para a implementação prática de sistemas baseados em lambda, incluindo linguagens de programação funcional. A compreensão dessas técnicas permite entender como funções são definidas, aplicadas e transformadas no contexto do cálculo lambda. A redução-$\alpha$ lida com a renomeação de variáveis ligadas, enquanto a redução beta trata da aplicação de funções a argumentos.

O Teorema de Church-Rosser, conhecido como propriedade de confluência local, estabelece a consistência do processo de redução no cálculo lambda. _currying_, por sua vez, é uma técnica que transforma funções com múltiplos argumentos em uma sequência de funções de um único argumento. Os combinadores, como `S`, `K`, e `I`, são expressões lambda sem variáveis livres que permitem a construção de funções complexas a partir de blocos básicos. Esses conceitos complementam as técnicas de redução e formam a base teórica para a manipulação e avaliação de expressões no cálculo lambda.

## 3.1. Redução Alfa

A redução-$\alpha\,$, ou _alpha reduction_, é o processo de renomear variáveis ligadas em termos lambda, para preservar o comportamento funcional dos termos. **Dois termos são equivalentes sob redução-$\alpha$ se diferirem unicamente nos nomes de suas variáveis ligadas**.

A atenta leitora deve considerar um termo lambda $\lambda x.\;E\,$, onde $E$ é o corpo do termo. A redução-$\alpha$ permitirá a substituição da variável ligada $x$ por outra variável, digamos $y\,$, desde que $y$ não apareça livre em $E\,$. O termo resultante é $\lambda y.\;E[x \mapsto y]\,$, onde a notação $E[x \mapsto y]$ indica a substituição de todas as ocorrências de $x$ por $y$ em $E\,$. Formalmente:

Seja $\lambda x.\;E$ um termo lambda, teremos:

$$\lambda x.\;E \to_\alpha \lambda y.\;E[x \mapsto y]$$

com a condição:

$$y \notin \text{FV}(E)$$

Onde $\text{FV}(E)$ representa o conjunto de variáveis livres em $E\,$, e $E[x \mapsto y]$ indica o termo resultante da substituição de todas as ocorrências da variável $x$ por $y$ em $E\,$, respeitando as ligações de variáveis para evitar a captura. A substituição $E[x \mapsto y]$ é definida formalmente por indução na estrutura de $E\,$. As possibilidades que devemos analisar são:

1. Se $E$ é uma variável, e for igual a $x\,$, a substituição resulta em $y$; caso contrário, $E[x \mapsto y]$ é o próprio $E\,$.

2. Se $E$ é uma aplicação $E_1\;E_2\,$, a substituição é aplicada a ambos os componentes, ou seja, $E[x \mapsto y] = E_1[x \mapsto y]\;E_2[x \mapsto y]\,$.

3. Se $E$ é uma abstração $\lambda z.\;E'\,$, a situação depende da relação entre $z$ e $x\,$.

   - Se $z$ é igual a $x\,$, então $E[x \mapsto y]$ é $\lambda z.\;E'\,$, pois $x$ está ligada por $\lambda z$ e não deve ser substituída dentro de seu próprio escopo.
  
   - Se $z$ é diferente de $x\,$, e $y$ não aparece livre em $E'$ e $z$ é diferente de $y\,$, então $E[x \mapsto y]$ é $\lambda z.\;E'[x \mapsto y]\,$.

4. Se $y$ aparece livre em $E'$ ou $z$ é igual a $y\,$, é necessário renomear a variável ligada $z$ para uma nova variável $w$ que não apareça em $E'$ nem em $y\,$, reescrevendo $E$ como $\lambda w.\;E'[z \mapsto w]$ e então procedendo com a substituição: $E[x \mapsto y] = \lambda w.\;E'[z \mapsto w][x \mapsto y]\,$.

Finalmente temos que a condição $y \notin \text{FV}(E)$ é a forma de evitar a captura de variáveis livres durante a substituição, garantindo que o conjunto de variáveis livres permaneça inalterado e que a semântica do termo seja preservada.

Usamos A redução-$\alpha$ para evitar a captura de variáveis livres durante a substituição na redução beta. Ao substituir um termo $N$ em um termo $E\,$, é possível que variáveis livres em $N$ tornem-se ligadas em $E\,$, o que irá alterar o significado semântico do termo. Para evitar isso, é necessário renomear as variáveis ligadas em $E$ para novas variáveis que não conflitem com as variáveis livres em $N\,$.

**Exemplo 1**: Considere o termo:

$$(\lambda x.\;x)\;(\lambda y.\;y)$$

Este termo é uma aplicação de função com duas funções identidade. Podemos começar com a estrutura do termo: uma função externa $\lambda x.\;x$ aplicada a um argumento que é uma função, $\lambda y.\;y\,$.

Uma vez que entendemos a estrutura, podemos fazer a análise das variáveis ligadas. Na função externa, $(\lambda x.\;x)\,$, $x$ é uma variável ligada. No argumento, $(\lambda y.\;y)\,$, $y$ é uma variável ligada.

Nosso próximo passo é verificar o escopo das variáveis. No termo original, $x$ está ligada no escopo de $\lambda x.\;x$ e $y$ está ligada no escopo de $\lambda y.\;y$

Em resumo, no termo original, $(\lambda x.\;x)\;(\lambda y.\;y)\,$, as variáveis ligadas $x$ e $y$ estão em escopos diferentes. Não há sobreposição ou conflito entre os escopos de $x$ e $y\,$. Ou seja, a substituição de $x$ por $(\lambda y.\;y)\,$, durante a aplicação, não causará captura de variáveis. Neste caso, não há necessidade de aplicar a redução-$\alpha\,$. As variáveis $x$ e $y$ podem permanecer com seus nomes originais sem causar ambiguidade ou conflito.

Neste ponto, já sabemos que não há necessidade de redução-$\alpha$ o que simplifica o processo de avaliação do termo. Logo, a aplicação, redução-$beta\,$, pode ser aplicada diretamente. Neste caso, substituímos todas as ocorrências de $x$ no corpo da abstração externa pelo argumento $\lambda y.\;y\,$, como $x$ aparece uma vez no corpo, o resultado é simplesmente $\lambda y.\;y\,$.

Este exemplo ilustra uma situação em que, apesar de termos múltiplas variáveis ligadas ($x$ e $y$), suas definições em escopos distintos e não sobrepostos eliminam a necessidade de redução-$\alpha\,$.

Com o Exemplo 2 podemos perceber que a necessidade de redução-$\alpha$ não depende somente da presença de variáveis ligadas, mas depende de como seus escopos interagem no termo como um todo.

**Exemplo 2**: Considere o termo:

$$(\lambda x.\;\lambda x.\;x)\;y$$

Observe que neste termo, a variável $x$ está ligada duas vezes em escopos diferentes, duas abstrações lambda. Para evitar confusão, podemos aplicar a redução-$\alpha$ para renomear uma das variáveis ligadas. Podemos aplicar na abstração interna ou na abstração externa. Vejamos:

1. Renomear a variável ligada interna:

   $$\lambda x.\;\lambda x.\;x \to_\alpha \lambda x.\;\lambda z.\;z$$

   Esta escolha é interessante por alguns motivos. O primeiro é que esta redução preserva a semântica do termo mantendo o significado original da função externa intacto. A redução da abstração interna preserva o escopo mínimo de mudança. Alterando o escopo mais interno, minimizamos o impacto em possíveis referências externas.

   A escolha pela abstração interna mantém a clareza da substituição. Durante a aplicação, redução-$beta\,$, ficará evidente que $y$ irá substituir o $x$ externo, enquanto $z$ permanece inalterado. Por fim, a redução da abstração interna é consistente com as práticas de programação e reflete o princípio de menor surpresa, mantendo variáveis externas estáveis. Por último, a escolha da abstração interna previne a captura acidental de variáveis livres.

2. Renomear a variável ligada externa:

   $$\lambda x.\;\lambda x.\;x \to_\alpha \lambda z.\;\lambda x.\;x$$

   A escolha pela abstração externa implica no risco de alteração semântica, correndo o risco de mudar o comportamento se o termo for parte de uma expressão maior que referencia $x\,$. Outro risco está na perda de informação estrutural. Isso será percebido após a aplicação, redução-$\beta\,$, ($\lambda x.\;x$), perde-se a informação sobre a abstração dupla original. Existe ainda possibilidade de criarmos uma confusão de escopos que pode acarretar uma interpretações errônea sobre qual variável está sendo referenciada em contextos mais amplos.

   Há uma razão puramente empírica. A escolha pela abstração externa contraria as práticas comuns ao cálculo lambda, onde as variáveis externas geralmente permanecem estáveis. Por fim, a escolha pela abstração externa reduz a rastreabilidade das transformações em sistemas de tipos ou em sistemas de análise estáticas.

A perspicaz leitora deve ter percebido o esforço para justificar a aplicação da redução-$\alpha$ a abstração interna. Agora que a convenci, podemos fazer a aplicando, β-redução, após a abordagem 1:

$$(\lambda x.\;\lambda z.\;z)\;y \to_\beta \lambda z.\;z$$

Esta redução resulta em uma expressão que preserva a estrutura essencial do termo original, mantendo a abstração interna intacta e substituindo a variável externa, conforme esperado na semântica do cálculo lambda.

### 3.1.1. Formalização da Equivalência Alfa

A equivalência alfa é um conceito do cálculo lambda que estabelece quando dois termos são considerados essencialmente idênticos, diferindo exclusivamente nos nomes de suas variáveis ligadas. Formalmente dizemos que **dois termos lambda $M$ e $N$ são considerados alfa-equivalentes, denotado por $M \equiv_\alpha N\,$, se um pode ser obtido do outro por meio de uma sequência finita de renomeações consistentes de variáveis ligadas**.

Podemos definir a equivalência alfa considerando as três estruturas básicas do cálculo lambda:

1. **Variáveis**:
   $x \equiv_\alpha x$

2. **Aplicação**:
   Se $E_1 \equiv_\alpha F_1$ e $E_2 \equiv_\alpha F_2\,$, então $(E_1\;E_2) \equiv_\alpha (F_1\;F_2)$

3. **Abstração**:
   Se $E \equiv_\alpha F[y/x]$ e $y$ não ocorre livre em $E\,$, então $\lambda x.\;E \equiv_\alpha \lambda y. F$

Onde $F[y/x]$ indica a substituição de todas as ocorrências livres de $x$ por $y$ no corpo $F\,$.

Para que a equivalência alfa seja válida, precisamos seguir três regras básicas: **garantir que as variáveis ligas a um termo lambda sejam renomadas**; **ao renomear uma variável ligada aplicamos a renomeação a todas as ocorrências da variável dentro de seu escopo**, corpo $E\,$, devem ser substituídas pelo novo nome; e finalmente, **o novo nome escolhido para a variável ligada não deve aparecer livre no corpo $E$** onde a substituição for aplicada.

Do ponto de vista da análise relacional, a relação $\equiv_\alpha$ é uma relação de equivalência, o que significa que ela possui as propriedades de: **Reflexividade**,  significando que para todo termo $M\,$, $M \equiv_\alpha M\,$. Ou seja, **todo termo é alfa equivalente a si mesmo**; **Simetria**, se $M \equiv_\alpha N\,$, então $N \equiv_\alpha M\,$. **Todos os termos equivalentes, são equivalentes entre si**; e **Transitividade**, se $M \equiv_\alpha N$ e $N \equiv_\alpha P\,$, então $M \equiv_\alpha P\,$. **Se dois termos, $M$ e $N$ são equivalentes entre si e um deles e equivalente a um terceiro termo, então o outro será equivalente ao terceiro termo**, como pode ser visto na Figura 3.1.1.A.

![Diagrama mostrando a propriedade da transitividade:](/assets/images/alfaEquiv.webp)
_Diagrama apresentando a transitividade da $\equiv_\alpha$._{: legenda}

A amável leitora pode estudar os exemplos a seguir para entender os conceitos da $\equiv_\alpha$.

**Exemplo 1**:

$$\lambda x.\;E \equiv_\alpha \lambda y.\;E[y/x]$$

   Neste termo, $E$ representa o corpo da função, e $E[y/x]$ representa a substituição de $x$ por $y$ em $E\,$.

**Exemplo 2**:

$$\lambda x. \lambda y.\;E \equiv_\alpha \lambda a. \lambda b.\;E[a/x][b/y]$$

   Aqui $E[a/x][b/y]$ representa o corpo $E$ com $x$ substituído por $a$ e $y$ por $b\,$.

**Exemplo 3**:

$$\lambda x.\;E \equiv_\alpha \lambda z.\;E[z/x]$$

   Essa expressão indica que a abstração lambda $\lambda x. , E$ é alfa-equivalente à abstração $\lambda z. , E[z/x]\,$, onde a variável ligada $x$ foi substituída por $z\,$. Esse processo de renomeação não afeta as variáveis livres em $E\,$. Se em $E$ existir uma variável $y\,$, esta variável seria livre e não seria afetada.

**Exemplo 4**: Renomeação Inválida. Considere a expressão:

$$\lambda x. \lambda y.\;E$$

   Uma tentativa inválida de renomeação seria:

$$\lambda x. \lambda y.\;E \not\equiv_\alpha \lambda x. \lambda x.\;E[x/y]$$

   Esta renomeação é inválida porque: **Viola a regra de captura de variáveis**: ao renomear $y$ para $x\,$, estamos potencialmente capturando ocorrências livres de $x$ que possam existir em $E$; **Provoca Perda de distinção**: as variáveis $x$ e $y\,$, que originalmente eram distintas, agora se tornaram a mesma variável, alterando potencialmente o significado da expressão; e **Altera a estrutura de escopo**: o escopo da variável $x$ externa foi efetivamente estendido para incluir o que antes era o escopo de $y\,$. Para ilustrar com um exemplo concreto desse problema, considere:

$$\lambda x.\;\lambda y.\;x + y$$

   A renomeação inválida resultaria em:

$$\lambda x.\;\lambda x.\;x + x$$

   Observe que o significado da expressão mudou. Na expressão original, $x$ e $y$ poderiam ter valores diferentes, mas na versão renomeada incorretamente, elas são forçadas a ter o mesmo valor.

   Uma renomeação válida seria:

$$\lambda x.\;\lambda y.\;x + $Y$ \equiv_\alpha \lambda x.\;\lambda z.\;x + z$$

   Aqui, renomeamos $y$ para $z\,$, mantendo a distinção entre as variáveis e preservando a estrutura e o significado da expressão original.

#### 3.1.1.1. Importância para a redução-$beta$

A equivalência alfa é impacta na correção da aplicação da redução-$beta\,$. Vamos analisar o seguinte exemplo:

$$(\lambda x. \lambda y.\;E)\;y$$

Se aplicássemos a redução-$beta$ diretamente, obteríamos $\lambda y.\;E[y/x]\,$, o que poderia mudar o significado da expressão se $y$ ocorrer livre em $E\,$. Para evitar isso, primeiro aplicamos uma redução-$\alpha$:

$$(\lambda x. \lambda y.\;E)\;y \equiv_\alpha (\lambda x.\;\lambda z.\;E[z/y])\;y$$

Agora podemos aplicar a redução-$beta$ com segurança:

$$(\lambda x. \lambda z.\;E[z/y])\;y \to_\beta \lambda z.\;E[z/y][y/x]$$

Este exemplo ilustra como a equivalência alfa permite o uso seguro da redução beta e da substituição no cálculo lambda, preservando o significado pretendido do corpo $E$ da função original.

### 3.1.2. Exercícios de Redução Alfa

**1**: Aplique a redução-$\alpha$ para renomear a variável da expressão $\lambda x.\;x + 2$ para $z\,$.

   **Solução:** Substituímos a variável ligada $x$ por $z$:

   $$\lambda x.\;x + 2 \to_\alpha \lambda z.\;z + 2$$

**2**: Renomeie a variável ligada $y$ na expressão $\lambda x. \lambda y.\;x + y$ para $w\,$.

   **Solução:** A redução-$\alpha$ renomeia $y$ para $w$:

   $$\lambda x. \lambda y.\;x + $Y$ \to_\alpha \lambda x.\;\lambda w.\;x + w$$

**3**: Aplique a redução-$\alpha$ para renomear a variável $z$ na expressão $\lambda z.\;z^2$ para $a\,$.

   **Solução:** Substituímos $z$ por $a$:

   $$\lambda z.\;z^2 \to_\alpha \lambda a. a^2$$

**4**: Renomeie a variável $f$ na expressão $\lambda f. \lambda x.\;f(x)$ para $g\,$, utilizando a redução-$\alpha\,$.

   **Solução:** Substituímos $f$ por $g$:

   $$\lambda f. \lambda x.\;f(x) \to_\alpha \lambda g. \lambda x.\;g(x)$$

**5**: Na expressão $\lambda x.\;(\lambda x.\;x + 1)\;x\,$, renomeie a variável ligada interna $x$ para $z\,$.

   Para entender este exercício podemos começar analisando a estrutura inicial. Na expressão original, $\lambda x.\;(\lambda x.\;x + 1)\;x\,$, temos uma abstração externa $\lambda x$ e uma abstração interna $\lambda x\,$. Observe que a variável $x$ está sendo usada em dois escopos diferentes.

   O problema aparece porque a repetição de $x$ cria ambiguidade já que o $x$ interno _sombreia_ o $x$ externo. Neste caso, vamos a redução-$\alpha\,$.

   Para aplicar a redução-$\alpha$ renomeamos o $x$ interno para $z\,$, preservando o significado original enquanto remove a ambiguidade.

   $$\lambda x.\;(\lambda x.\;x + 1) x \to_\alpha \lambda x.\;(\lambda z.\;z + 1) x$$

   Após a redução-$\alpha$ temos $x$ no escopo externo e $z$ no escopo interno. Ou seja, o $x$ após o parêntese se refere ao $x$ do escopo externo.

**6**: Aplique a redução-$\alpha$ na expressão $\lambda x. \lambda y.\;x \times y$ renomeando $x$ para $a$ e $y$ para $b\,$.

   **Solução:** Substituímos $x$ por $a$ e $y$ por $b$:

   $$\lambda x. \lambda y.\;x \times $Y$ \to_\alpha \lambda a. \lambda b. a \times b$$

**7**: Renomeie a variável ligada $y$ na expressão $\lambda x.\;(\lambda y.\;y + x)$ para $t\,$.

   **Solução:** Substituímos $y$ por $t$:

   $$\lambda x.\;(\lambda y.\;y + x) \to_\alpha \lambda x.\;(\lambda t. t + x)$$

**8**: Aplique a redução-$\alpha$ na expressão $\lambda f. \lambda x.\;f(x + 2)$ renomeando $f$ para $h\,$.

   **Solução:** Substituímos $f$ por $h$:

   $$\lambda f. \lambda x.\;f(x + 2) \to_\alpha \lambda h. \lambda x.\;h(x + 2)$$

**9**: Na expressão $\lambda x.\;(\lambda y.\;x - y)\,$, renomeie a variável $y$ para $v$ utilizando a redução-$\alpha\,$.

   **Solução:** Substituímos $y$ por $v$:

   $$\lambda x.\;(\lambda y.\;x - y) \to_\alpha \lambda x.\;(\lambda v.\;x - v)$$

**10**: Aplique a redução-$\alpha$ na expressão $\lambda x.\;(\lambda z.\;z + x)\;z\,$, renomeando $z$ na função interna para $w\,$.

   Podemos começar observando a estrutura original. Neste caso, temos uma abstração externa $\lambda x\,$, uma abstração interna $\lambda z$ e um $z$ livre após o parêntese.

   Uma vez que a estrutura está clara podemos avaliar as variáveis: o $z$ na função interna é uma variável ligada. Contudo, o $z$ após o parêntese é uma variável livre, não está ligada a nenhum $\lambda\,$.

   Neste caso, a aplicação da redução-$\alpha$ renomeará o $z$ ligado, na função interna, para $w$ enquanto o $z$ livre permanece inalterado.

   $$\lambda x.\;(\lambda z.\;z + x)\;z \to_\alpha \lambda x.\;(\lambda w. w + x)\;z$$

   Após a aplicação da redução-$\alpha\,$, temos $w$ como variável ligada na função interna. O $z$ livre permanece, mas agora está claramente diferenciado do $w$ ligado. Esta redução evita possíveis confusões entre a variável ligada e a variável livre enquanto mantém a semântica original da expressão.

### 3.1.3. Convenções Práticas: Convenção de Variáveis de Barendregt

Na prática, a redução-$\alpha$ é frequentemente aplicada implicitamente durante as substituições no cálculo lambda. A convenção das variáveis de [Barendregt](https://en.wikipedia.org/wiki/Henk_Barendregt)[^cita8] **estabelece que todas as variáveis ligadas em um termo devem ser escolhidas de modo que sejam distintas entre si e distintas de quaisquer variáveis livres presentes no termo**. Essa convenção elimina a necessidade de renomeações explícitas frequentes e simplifica a manipulação dos termos lambda.

A partir da Convenção de Barendregt, a definição de substituição pode ser simplificada. Em particular, ao realizar a substituição $[N/x]\;(\lambda y.\;M)\,$, podemos escrever:

$$[N/x]\;(\lambda y.\;M) \, = \lambda y.\;[N/x]M$$

Assumindo implicitamente que, se necessário, a variável ligada $y$ é renomeada para evitar conflitos, garantindo que $y \neq x$ e que $y$ não apareça livre em $N\,$. Isso significa que não precisamos nos preocupar com a captura de variáveis livres durante a substituição, pois a convenção assegura que as variáveis ligadas são sempre escolhidas de forma apropriada. Permitido que tratemos os termos alfa-equivalentes como se fossem idênticos. Por exemplo, podemos considerar os seguintes termos como iguais:

$$\lambda x.\;\lambda y.\;x\;y = \lambda a.\;\lambda b.\;a\;b$$

Ambos representam a mesma função, diferindo unicamente nos nomes das variáveis ligadas. Essa abordagem simplifica significativamente a manipulação de termos lambda, pois não precisamos constantemente lidar com conflitos de nomes ou realizar reduções alfa explícitas. Podemos focar nas reduções beta e na estrutura funcional dos termos, sabendo que a escolha dos nomes das variáveis ligadas não afeta o comportamento das funções representadas.

### 3.1.4. Exercícios de Substituição, Redução Alfa e Convenção de Barendregt

**1**: Aplique a substituição $[y/x]\,x$ e explique o processo.

   **Solução:** A substituição de $x$ por $y$ é direta:

$$[y/x]\,x = y$$

**2**: Aplique a substituição $[y/x]\;(\lambda x.\;x + 1)$ e explique por que a substituição não ocorre.

   **Solução:** A variável $x$ está ligada dentro da abstração $\lambda x \,$, então a substituição não afeta o corpo da função:

   $$[y/x]\;(\lambda x.\;x + 1) \, = \lambda x.\;x + 1$$

**3**: Aplique a substituição $[z/x]\;(\lambda z.\;x + z)\,$. Utilize redução-$\alpha$ para evitar captura de variáveis.

   **Solução:** A substituição direta causaria captura de variáveis. Aplicamos a redução-$\alpha$ para renomear $z$ antes de fazer a substituição:

   $$[z/x]\;(\lambda z.\;x + z) \, = \lambda w.\;z + w$$

**4**: Considere a expressão $(\lambda x. \lambda y.\;x + y) z\,$. Aplique a substituição $[z/x]$ e explique a necessidade de redução-$\alpha\,$.

   **Solução:** Como $x$ não está ligada, podemos realizar a substituição sem necessidade de alfa. A expressão resultante será:

   $$[z/x]\;(\lambda x. \lambda y.\;x + y) \, = \lambda y.\;z + y$$

**5**: Aplique a substituição $[z/x]\;(\lambda z.\;x + z)$ sem realizar a redução-$\alpha\,$. O que ocorre?

   **Solução:** Se aplicarmos diretamente a substituição sem evitar a captura, a variável $z$ é capturada e a substituição resultará incorretamente em:

   $$[z/x]\;(\lambda z.\;x + z) \, = \lambda z.\;z + z$$

**6**: Considere a expressão $(\lambda x. \lambda y.\;x + y) (\lambda z.\;z \times z)\,$. Aplique a substituição $[(\lambda z.\;z \times z)/x]$ e use a convenção de Barendregt.

   **Solução:** Aplicamos a substituição:

   $$[(\lambda z.\;z \times z)/x]\;(\lambda x. \lambda y.\;x + y) \, = \lambda y.\;(\lambda z.\;z \times z) + y$$

   Com a convenção de Barendregt, variáveis ligadas não entram em conflito.

**7**: Aplique a redução-$\alpha$ na expressão $\lambda x. \lambda y.\;x + y$ para renomear $ x $ e $ $Y$ $ para $ a $ e $ b \,$, respectivamente, e aplique a substituição $[3/a]\,$.

   **Solução:** Primeiro, aplicamos a redução-$\alpha$:

   $$\lambda x. \lambda y.\;x + $Y$ \to_\alpha \lambda a. \lambda b. a + b$$

   Agora, aplicamos a substituição:

   $$[3/a] \ ,(\lambda a. \lambda b. a + b) \, = \lambda b. 3 + b$$

**8**: Aplique a convenção de Barendregt na expressão $\lambda x.\;(\lambda x.\;x + 1) x$ antes de realizar a substituição $[y/x]\,$.

   **Solução:** Aplicando a convenção de Barendregt, renomeamos a variável ligada interna para evitar conflitos:

   $$\lambda x.\;(\lambda x.\;x + 1) x \to_\alpha \lambda x.\;(\lambda z.\;z + 1) x$$

   Agora, aplicamos a substituição:

   $$[y/x]\;(\lambda x.\;(\lambda z.\;z + 1) x) \, = \lambda x.\;(\lambda z.\;z + 1) y$$

**9**: Aplique a redução-$\alpha$ na expressão $\lambda x.\;(\lambda y.\;x + y)\,$, renomeando $ $Y$ $ para $ z \,$, e depois aplique a substituição $[5/x]\,$.

   **Solução:** Primeiro, aplicamos a redução-$\alpha$:

   $$\lambda x.\;(\lambda y.\;x + y) \to_\alpha \lambda x.\;(\lambda z.\;x + z)$$

   Agora, aplicamos a substituição:

   $$[5/x]\;(\lambda x.\;(\lambda z.\;x + z)) \, = \lambda z. 5 + z$$

**10**: Aplique a substituição $[y/x]\;(\lambda x.\;x + z)$ e explique por que a convenção de Barendregt nos permite evitar a redução-$\alpha$ neste caso.

   **Solução:** Como $x$ é ligado e não há conflitos com variáveis livres, a substituição não afeta o termo, e a convenção de Barendregt garante que não há necessidade de renomeação:

   $$[y/x]\;(\lambda x.\;x + z) \, = \lambda x.\;x + z$$

**11**: Considere o termo $[z/x]\;(\lambda y.\;x + (\lambda x.\;x + y))\,$. Aplique a substituição e a redução-$\alpha$ se necessário.

   **Solução:** Como há um conflito com a variável $x$ no corpo da função, aplicamos redução-$\alpha$ antes da substituição:

   $$\lambda y.\;x + (\lambda x.\;x + y) \to_\alpha \lambda y.\;x + (\lambda w. w + y)$$

   Agora, aplicamos a substituição:

   $$[z/x]\;(\lambda y.\;x + (\lambda w. w + y)) \, = \lambda y.\;z + (\lambda w. w + y)$$

**12**: Aplique a substituição $[y/x]\;(\lambda z.\;x + z)$ onde $ z \notin FV(y)\,$, e explique o processo.

   **Solução:** Como não há conflitos de variáveis livres e ligadas, aplicamos a substituição diretamente:

   $$[y/x]\;(\lambda z.\;x + z) \, = \lambda z. $Y$ + z$$

**13**: Aplique a substituição $[z/x]\;(\lambda y.\;x \times y)$ onde $ z \in FV(x)\,$. Utilize a convenção de Barendregt.

   **Solução:** Como $z$ não causa conflito de variáveis livres ou ligadas, aplicamos a substituição diretamente:

   $$[z/x]\;(\lambda y.\;x \times y) \, = \lambda y.\;z \times y$$

   A convenção de Barendregt garante que não precisamos renomear variáveis.

**14**: Aplique a redução-$\alpha$ na expressão $\lambda x.\;(\lambda y.\;x + y)$ e renomeie $ $Y$ $ para $ t \,$, depois aplique a substituição $[2/x]\,$.

   **Solução:** Primeiro aplicamos a redução-$\alpha$:

   $$\lambda x.\;(\lambda y.\;x + y) \to_\alpha \lambda x.\;(\lambda t.\;x + t)$$

   Agora, aplicamos a substituição:

   $$[2/x]\;(\lambda x.\;(\lambda t.\;x + t)) \, = \lambda t. 2 + t$$

**15**: Aplique a substituição $[y/x]\;(\lambda x.\;x + (\lambda z.\;x + z))$ e explique por que não é necessário aplicar a redução-$\alpha\,$.

   **Solução:** Como a variável $x$ está ligada e não entra em conflito com outras variáveis, a substituição não altera o termo:

   $$[y/x]\;(\lambda x.\;x + (\lambda z.\;x + z)) \, = \lambda x.\;x + (\lambda z.\;x + z)$$

## 3.2. redução Beta

A redução beta é o mecanismo de computação do cálculo lambda que **permite simplificar expressões por meio da aplicação de funções aos seus argumentos**. As outras reduções $\beta$ e $\eta$ são mecanismos de transformação que facilitam, ou possibilitam, a redução-$beta\,$.Formalmente, a redução beta é definida como:

$$(\lambda x.\;E)\;N \to_\beta [x/N]E$$

A notação $[x/N]M$ representa a substituição de todas as ocorrências livres da variável $x$ no termo $E$ pelo termo $N\,$. Eventualmente, quando estudamos semântica denotacional, ou provas formais, usamos a notação $E[x := y]\,$.

A substituição indicada em uma redução-$beta$ deve ser realizada com cuidado para evitar a captura de variáveis livres em $N$ que possam se tornar ligadas em $E$ após a substituição. Para evitar a captura de varáveis livres, pode ser necessário realizar uma redução-$\alpha$ antes de começar a redução beta, renomeando variáveis ligadas em $E$ que possam entrar em conflito com variáveis livres em $N\,$, Figura 3.2.A.

![Diagrama mostrando uma função aplicada a um valor, a regra formal da redução beta e a forma normal obtida](/assets/images/beta.webp)
_3.2.A: Exemplo de Redução Beta_{: legenda}

Considere, por exemplo, o termo $E = (\lambda y.\;x + y)$ e o objetivo de substituir $x$ por $N = y\,$. Se fizermos a substituição diretamente, obteremos:

$$[x/y]E = (\lambda y.\;y + y)$$

Nesse caso, a variável livre $y$ em $N$ tornou-se ligada devido ao $\lambda y$ em $E\,$, resultando em captura de variável e alterando o significado original da expressão. Para evitar a captura, aplicamos uma redução-$\alpha$ ao termo $E\,$, renomeando a variável ligada $y$ para uma nova variável que não apareça em $N\,$, como $z$:

$$E = (\lambda y.\;x + y) \quad \rightarrow_\beta \quad (\lambda z.\;x + z)$$

Agora, ao substituir $x$ por $y\,$, obtemos:

$$[x/y]E = (\lambda z.\;y + z)$$

A variável livre $y$ permanece livre após a substituição, e a captura é evitada. Outro exemplo é o termo $E = (\lambda x.\;\lambda y.\;x + y)$ com $N = y\,$. Tentando substituir diretamente, temos:

$$[x/y]E = (\lambda y.\;y + y)$$

Aqui, a variável livre $y$ em $N$ foi capturada pelo $\lambda y$ interno. Para prevenir isso, realizamos uma redução-$\alpha$ renomeando a variável ligada $y$ para $z$:

$$E = (\lambda x.\;\lambda y.\;x + y) \quad \xrightarrow{\text{alfa}} \quad (\lambda x.\;\lambda z.\;x + z)$$

Procedendo com a substituição:

$$[x/y]E = (\lambda z.\;y + z)$$

Assim, a variável livre $y$ não é capturada, preservando o significado da expressão original.

Em suma, ao realizar a substituição $[x/N]E\,$, é essencial garantir que as variáveis livres em $N$ permaneçam livres após a substituição e não sejam capturadas por variáveis ligadas em $E\,$. A persistente leitora deve avaliar os exemplos a seguir:

**Exemplo 1**: considere a expressão:

$$(\lambda x.\;x + y)\;3$$

Aplicando a redução beta:

$$(\lambda x.\;x + y)\;3 \to_\beta 3 + y$$

Neste caso, substituímos $x$ por $3$ no corpo da função $x + y\,$, resultando em $3 + y\,$. A variável $y$ permanece inalterada por ser uma variável livre.

**Exemplo 2**: se houver risco de captura de variáveis, é necessário realizar uma redução-$\alpha$ antes. Por exemplo:

$$(\lambda x.\;\lambda y.\;x + y)\;y$$

Aplicando a redução beta diretamente:

$$(\lambda x.\;\lambda y.\;x + y)\;y \to_\beta \lambda y.\;y + y$$

Aqui, a variável livre $y$ no argumento foi capturada pela variável ligada $y$ na função interna. Para evitar isso, realizamos uma redução-$\alpha$ renomeando a variável ligada $y$ para $z$:

$$\lambda x.\;\lambda z.\;x + z$$

Agora podemos aplicar a redução beta:

$$(\lambda x.\;\lambda z.\;x + z)\;y \to_\beta \lambda z.\;y + z$$

Assim, evitamos a captura da variável livre $y\,$, mantendo o significado original da expressão.

**Exemplo 3**: considere a expressão:

$$(\lambda x.\;x+1)2$$

 Aplicando a redução beta:

$$(\lambda x.\;x+1)2 \to_\beta [2/x]\;(x+1) \, = 2+1 = 3$$

 Aqui, o valor $2$ é substituído pela variável $x$ na expressão $x + 1\,$, resultando em $2 + 1 = 3\,$.

**Exemplo 4**: um exemplo mais complexo envolvendo uma função de ordem superior:

   $$(\lambda f.\lambda x.\;f\;(f\;x))(\lambda y.\;y*2)\;3$$

   Reduzindo passo a passo:

   $$(\lambda f.\lambda x.\;f(f\;x))(\lambda y.\;y \times 2) \, 3$$

   $$\to_\beta (\lambda x.(\lambda y.\;y \times 2)((\lambda y.\;y \times 2) x))\;3$$

   $$\to_\beta (\lambda y.\;y \times 2)((\lambda y.\;y \times 2)\;3)$$

   $$\to_\beta (\lambda y.\;y \times 2)\;(3 \times 2)$$

   $$\to_\beta (\lambda y.\;y \times 2)\;(6)$$

   $$\to_\beta 6 \times 2$$

   $$= 12$$

Neste exemplo, aplicamos primeiro a função $(\lambda f.\lambda x.\;f\;(f\;x))$ ao argumento $(\lambda y.\;y \times 2)\,$, resultando em uma expressão que aplica duas vezes a função de duplicação ao número $3\,$, obtendo $12\,$.

### 3.2.1. Exercícios de redução-$beta$

**1**: Aplique a redução beta na expressão $(\lambda x.\;x + 1)\;5\,$.

   **Solução:** Aplicamos a substituição de $x$ por $5$ no corpo da função:

   $$(\lambda x.\;x + 1)\;5 \to_\beta [5/x]\;(x + 1) \, = 5 + 1 = 6$$

**2**: Simplifique a expressão $(\lambda x. \lambda y.\;x + y)\;2\;3 $ utilizando a redução beta.

   **Solução:** Primeiro, aplicamos $2$ ao parâmetro $x\,$, e depois $3$ ao parâmetro $y$:

   $$(\lambda x. \lambda y.\;x + y)\;2\;3 \to_\beta (\lambda y.\;2 + y)\;3 \to_\beta 2 + 3 = 5$$

**3**: Aplique a redução beta na expressão $(\lambda f. \lambda x.\;f(f\;x)) (\lambda y.\;y + 1)\;4\,$.

   **Solução:** Primeiro aplicamos $(\lambda y.\;y + 1)$ a $f\,$, e depois $4$ a $x$:

   $$(\lambda f. \lambda x.\;f(f\;x))\;(\lambda y.\;y + 1)\;4$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;y + 1)\;( (\lambda y.\;y + 1) x))\;4$$

   $$\to_\beta (\lambda y.\;y + 1)((\lambda y.\;y + 1)\;4)$$

   $$\to_\beta (\lambda y.\;y + 1)(4 + 1)$$

   $$\to_\beta (\lambda y.\;y + 1)(5)$$

   $$\to_\beta 5 + 1 = 6$$

**4**: Reduza a expressão $(\lambda x. \lambda y.\;x \times y)\;3\;4$ utilizando a redução beta.

   **Solução:** Primeiro aplicamos $3$ a $x$ e depois $4$ a $y$:

   $$(\lambda x. \lambda y.\;x \times y)\;3\;4 \to_\beta (\lambda y.\;3 \times y)\;4 \to_\beta 3 \times 4 = 12$$

**5**: Aplique a redução beta na expressão $(\lambda x. \lambda y.\;x - y)\;10\;6\,$.

   **Solução:** Aplicamos a função da seguinte forma:

   $$(\lambda x. \lambda y.\;x - y)\;10\;6 \to_\beta (\lambda y.\;10 - y)\;6 \to_\beta 10 - 6 = 4$$

**6**: Reduza a expressão $(\lambda f.\;f(2)) (\lambda x.\;x + 3)$ utilizando a redução beta.

   **Solução:** Primeiro aplicamos $(\lambda x.\;x + 3)$ a $f\,$, e depois aplicamos $2$ a $x$:

   $$(\lambda f.\;f(2)) (\lambda x.\;x + 3) \to_\beta (\lambda x.\;x + 3)\;(2) \to_\beta 2 + 3 = 5$$

**7**: Simplifique a expressão $(\lambda f. \lambda x.\;f(x + 2)) (\lambda y.\;y \times 3)\;4 $ utilizando a redução beta.

   **Solução:** Primeiro aplicamos $(\lambda y.\;y \times 3)$ a $f$ e depois $4$ a $x$:

   $$(\lambda f. \lambda x.\;f(x + 2)) (\lambda y.\;y \times 3)\;4$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;y \times 3)(x + 2))\;4$$

   $$\to_\beta (\lambda y.\;y \times 3)(4 + 2)$$

   $$\to_\beta (6 \times 3) \, = 18$$

**8**: Aplique a redução beta na expressão $(\lambda x. \lambda y.\;x^2 + y^2)\;(3 + 1)\;(2 + 2)\,$.

   **Solução:** Primeiro simplificamos as expressões internas e depois aplicamos as funções:

   $$(\lambda x. \lambda y.\;x^2 + y^2)\;(3 + 1)\;(2 + 2)$$

   $$\to_\beta (\lambda x. \lambda y.\;x^2 + y^2)\;4\;4$$

   $$\to_\beta (\lambda y.\;4^2 + y^2)\;4$$

   $$\to_\beta 16 + 4^2 = 16 + 16 = 32$$

**9**: Reduza a expressão $(\lambda f. \lambda x.\;f(f(x))) (\lambda y.\;y + 2)\;3$ utilizando a redução beta.

   **Solução:** Aplicamos a função duas vezes ao argumento:

   $$(\lambda f. \lambda x.\;f(f(x))) (\lambda y.\;y + 2)\;3$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;y + 2)((\lambda y.\;y + 2) x))\;3$$

   $$\to_\beta (\lambda y.\;y + 2)((\lambda y.\;y + 2)\;3)$$

   $$\to_\beta (\lambda y.\;y + 2)(3 + 2)$$

   $$\to_\beta (\lambda y.\;y + 2)(5)$$

   $$\to_\beta 5 + 2 = 7$$

**10**: Reduza a expressão $(\lambda x. \lambda y.\;x - 2 \times y) (6 + 2)\;3$ utilizando a redução beta.

   **Solução:** Primeiro simplificamos as expressões e depois aplicamos as funções:

   $$(\lambda x. \lambda y.\;x - 2 \times y)\;(6 + 2)\;3$$

   $$\to_\beta (\lambda x. \lambda y.\;x - 2 \times y)\;8\;3$$

   $$\to_\beta (\lambda y.\;8 - 2 \times y)\;3$$

   $$\to_\beta 8 - 2 \times 3 = 8 - 6 = 2$$

## 3.3. Redução Eta

A redução-$\eta$ é uma das três formas de redução no cálculo lambda, juntamente com as reduções alfa e beta. A redução-$\eta$ captura a ideia de extensionalidade, permitindo simplificar termos que representam a mesma função pelo comportamento externo. Formalmente, a redução-$\eta$ é definida pela seguinte regra:

$$\lambda x.\;f\;x \to_\eta f \quad \text{se } x \notin \text{FV}(f)$$

Neste caso, esta expressão é formada por: uma função $\lambda x.\;f\;x$ que recebe um argumento $x$ e aplica a função $f$ a $x$; $\text{FV}(f)$ representa o conjunto de variáveis livres em $f$ e a condição $x \notin \text{FV}(f)$ garante que $x$ não aparece livre em $f\,$, evitando a captura de variáveis.

A redução-$\eta$ permite eliminar a abstração $\lambda x$ quando a função aplica $f$ diretamente a $x\,$, resultando que $\lambda x.\;f\;x$ é equivalente a comportamentalmente a $f$. Isso ocorre quando $x$ não aparece livre em $f\,$, garantindo que a eliminação da abstração não altera o significado do termo. Podemos dizer que a redução-$\eta$ expressa o princípio de que duas funções são iguais se, para todos os argumentos, elas produzem os mesmos resultados. Se uma função $\lambda x.\;f\;x$ aplica $f$ diretamente ao seu argumento $x\,$, e $x$ não aparece livre em $f\,$, então $\lambda x.\;f\;x$ pode ser reduzido a $f\,$.

**Exemplo 1**: Considere o termo:

$$\lambda x.\;f\;x$$

Se $f = \lambda y.\;y + 2$ e $x$ não aparece livre em $f\,$, a redução-$\eta$ pode ser aplicada:

$$\lambda x.\;f\;x \to_\eta f = \lambda y.\;y + 2$$

Assim, $\lambda x.\;f\;x$ reduz-se a $f\,$.

**Exemplo 2**: Considere o termo:

$$\lambda x.\;(\lambda y.\;y^2)\;x$$

Como $x$ não aparece livre em $f = \lambda y.\;y^2\,$, a redução-$\eta$ pode ser aplicada:

$$\lambda x.\;f\;x \to_\eta f = \lambda y.\;y^2$$

Portanto, $\lambda x.\;(\lambda y.\;y^2)\;x$ reduz-se a $\lambda y.\;y^2\,$.

Se $x$ aparece livre em $f\,$, a redução-$\eta$ não é aplicável, pois a eliminação de $\lambda x$ alteraria o comportamento do termo. Por exemplo, se $f = \lambda y.\;x + y\,$, onde $x$ é uma variável livre em $f\,$, então:

$$\lambda x.\;f\;x = \lambda x.\;(\lambda y.\;x + y)\;x \to_\beta \lambda x.\;x + x$$

Neste caso, não é possível aplicar a redução-$\eta$ para obter $f\,$, pois $x$ aparece livre em $f\,$, e remover $\lambda x$ deixaria $x$ indefinida.

A condição $x \notin \text{FV}(f)$ é crucial. Se $x$ aparecer livre em $f\,$, a redução-$\eta$ não pode ser aplicada, pois a remoção da abstração $\lambda x$ poderia alterar o significado do termo.

**Exemplo Contrário**: Considere a expressão:

$$\lambda x.\;x\;x$$

Aqui, $x$ aparece livre no corpo $x\;x\,$. Não podemos aplicar a redução-$\eta$ para obter $x\,$, pois isso alteraria o comportamento da função.

### 3.3.1. Propriedade de Extensionalidade

A redução-$\eta$ está relacionada ao conceito de **extensionalidade** em matemática, onde duas funções são consideradas iguais se produzem os mesmos resultados para todos os argumentos. No cálculo lambda, a redução-$\eta$ formaliza esse conceito, permitindo a simplificação de funções que são extensionais.

>Em matemática, o conceito de extensionalidade refere-se à ideia de que dois objetos são considerados iguais se têm as mesmas propriedades externas ou observáveis. No contexto das funções, a extensionalidade implica que duas funções $f$ e $g$ são consideradas iguais se, para todo argumento $x$ em seu domínio comum, $f(x) \, = g(x)\,$. Isso significa que a identidade de uma função é determinada pelos seus valores de saída para cada entrada possível, e não pela sua definição interna ou pela forma como é construída.
>
>A extensionalidade é um princípio em várias áreas da matemática, incluindo teoria dos conjuntos e lógica matemática. Na teoria dos conjuntos, o axioma da extensionalidade afirma que dois conjuntos são iguais se e somente se contêm exatamente os mesmos elementos. No cálculo lambda e na programação funcional, a extensionalidade se manifesta através de conceitos como a redução-$\eta\,$, que permite tratar funções que produzem os mesmos resultados para todas as entradas como equivalentes, independentemente de suas estruturas internas.

Para ilustrar o conceito de extensionalidade e sua relação com a redução-$\eta\,$, a esforçada leitora deve considerar os seguintes exemplos:

**Exemplo 1**: Suponha que temos duas funções no cálculo lambda:

$$f = \lambda x.\;x^2 + 2x + 1$$

e

$$g = \lambda x.\;(x + 1)^2$$

Embora $f$ e $g$ tenham definições diferentes, podemos demonstrar que elas produzem o mesmo resultado para qualquer valor de $x$:

$$
f(x) \, = x^2 + 2x + 1 \\
g(x) \, = (x + 1)^2 = x^2 + 2x + 1
$$

Portanto, para todo $x\,$, $f(x) \, = g(x)\,$, o que significa que $f$ e $g$ são extensionais, apesar de suas diferentes expressões internas.

**Exemplo 2**: Considere as funções:

$$
h = \lambda x.\;f\;x \\
k = f
$$

Se $x$ não aparece livre em $f\,$, a redução-$\eta$ nos permite afirmar que $h$ é equivalente a $k$:

$$h = \lambda x.\;f\;x \to_\eta f = k$$

Isso mostra que, embora $h$ seja definido como uma função que aplica $f$ a $x\,$, ela é extensionamente igual a $f$ em si, reforçando o princípio de que a forma interna da função é menos relevante do que seu comportamento externo.

**Exemplo 3**: No contexto da programação funcional, suponha que temos a seguinte função em Haskell:

```haskell
-- Definição explícita
doubleList :: [Int] -> [Int]
doubleList xs = map (\x -> x * 2) xs
```

Aplicando a redução-$\eta$ e considerando o conceito de extensionalidade, podemos simplificar a função:

```haskell
-- Após a redução eta
doubleList :: [Int] -> [Int]
doubleList = map (* 2)
```

Apesar das diferenças na implementação, ambas as versões de `doubleList` produzem os mesmos resultados para qualquer lista de inteiros fornecida, demonstrando que são extensionais.

**Exemplo 4**: Na teoria dos conjuntos, considere:

$$
A = \{ x \mid x \text{ é um número natural par} \} \\
B = \{ 2n \mid n \in \mathbb{N} \}
$$

Ambos os conjuntos $A$ e $B$ contêm exatamente os mesmos elementos. Pelo axioma da extensionalidade, concluímos que $A = B\,$, mesmo que suas definições sejam diferentes.

Em resumo, a extensionalidade nos permite focar no comportamento observável das funções ou conjuntos, em vez de suas definições internas. No cálculo lambda, a redução-$\eta$ é a ferramenta que formaliza esse princípio, simplificando termos que representam funções com o mesmo efeito para todos os argumentos. Ao aplicar a redução-$\eta\,$, estamos essencialmente afirmando que a forma como a função é construída é irrelevante se seu comportamento externo permanece inalterado.

### 3.3.2. Expansão $\eta$

Além da redução-$\eta\,$, existe a **expansão $\eta$**, que é a operação inversa da redução-$\eta$:

$$f \to_\eta \lambda x.\;f\;x \quad \text{se } x \notin \text{FV}(f)$$

A expansão $\eta$ pode ser útil para transformar termos em formas que facilitam outras reduções ou para provar equivalências entre funções. Neste caso, a equivalência lambda ($\equiv$) é a relação de equivalência gerada pelas reduções alfa, beta e $\eta\,$. Dois termos $M$ e $N$ são equivalentes lambda ($M \equiv N$) se podem ser transformados um no outro por uma sequência finita de reduções e expansões alfa, beta e $\eta\,$.

**Exemplo**: considere a definição de uma função identidade:

$$I = \lambda x.\;x$$

E uma função $F$ definida como:

$$F = \lambda x.\;I\;x$$

Aplicando a redução-$\eta$ a $F$ podemos observar que $I\;x = (\lambda y.\;y)\;x \to_\beta x\,$, ou seja:

$$F = \lambda x.\;I\;x \equiv \lambda x.\;x$$

Ou seja, $F$ pode ser reduzido a $I\,$. No entanto, usando a redução-$\eta$ diretamente:

$$F = \lambda x.\;I\;x \to_\eta I \quad \text{(se } x \notin \text{FV}(I))$$

Como $x$ não aparece livre em $I\,$, podemos aplicar a redução-$\eta$ para obter $F = I\,$.

### 3.3.3. Relação entre redução-$\eta$ e outras Formas de Redução

No cálculo lambda, as reduções alfa, beta e $\eta$ formam um conjunto integrado de transformações que permitem manipular e simplificar expressões lambda. Cada uma dessas reduções desempenha um papel específico, mas elas frequentemente interagem e complementam-se mutuamente.

Vamos começar com a redução-$\alpha\,$, que lida com a renomeação de variáveis ligadas, pode ser necessária antes de aplicar a redução-$\eta$ para evitar conflitos de nomes. Por exemplo:

$$\lambda x. (\lambda y.\;x\;y) x \to_\alpha \lambda x. (\lambda z.\;x z) x \to_\eta \lambda x.\;x$$

Neste caso, a redução-$\alpha$ foi aplicada primeiro para renomear $y$ para $z\,$, evitando a captura de variável, antes de aplicar a redução-$\eta\,$.

A interação entre redução-$\eta$ e beta é particularmente interessante. Em alguns casos, a redução-$\eta$ pode simplificar expressões após a aplicação da redução beta:

$$(\lambda f. \lambda x. f x) (\lambda y.\;y + 1) \to_\beta \lambda x. (\lambda y.\;y + 1) x \to_\eta \lambda y.\;y + 1$$

Aqui, a redução beta é aplicada primeiro, seguida pela redução-$\eta$ para simplificar ainda mais a expressão.

Nesta interação entre as reduções precisamos tomar cuidado com a ordem de aplicação das reduções pode afetar o resultado  e a eficiência do processo de redução. Em linhas gerais, podemos seguir as seguintes regras:

1. A redução-$\alpha$ é aplicada conforme necessário para evitar conflitos de nomes.

2. A redução beta é frequentemente aplicada após a aplicação alfa para realizar computações.

3. A redução-$\eta$ é aplicada para simplificar a estrutura da função após as reduções beta.

**O processo de aplicar todas as reduções possíveis (alfa, beta e $\eta$) até que nenhuma outra redução seja possível é chamado de normalização. A forma normal de um termo lambda é única, se existir, independentemente da ordem das reduções, graças ao Teorema de Church-Rosser.**

**Exemplo Integrado**: Considere a seguinte expressão:

$$(\lambda x. \lambda y.\;x) (\lambda z.\;z)$$

Podemos aplicar as reduções na seguinte ordem:

1. Redução beta: $(\lambda x. \lambda y.\;x) (\lambda z.\;z) \to_\beta \lambda y. (\lambda z.\;z)$

2. redução-$\eta$: $\lambda y. (\lambda z.\;z) \to_\eta \lambda z.\;z$

O resultado , $\lambda z.\;z\,$, é a função identidade, obtida através da aplicação combinada de reduções beta e $\eta\,$.

Entender a interação entre estas formas de redução é crucial para manipular eficientemente expressões lambda e para compreender a semântica de linguagens de programação funcional baseadas no cálculo lambda.

### 3.3.4. Relação entre a redução-$\eta$ e a Programação Funcional

A redução-$\eta$ é frequentemente utilizada em programação funcional para simplificar código e torná-lo mais conciso. Em Haskell, essa técnica é particularmente comum. Vejamos alguns exemplos práticos:

**Exemplo 1**: Simplificação de Funções

Em Haskell, podemos usar a redução-$\eta$ para simplificar definições de funções:

```haskell
-- Antes da redução-$\eta$
addOne :: Int -> Int
addOne x = (+ 1) x

-- Após a redução-$\eta$
addOne :: Int -> Int
addOne = (+ 1)
```

Neste exemplo, definimos uma função `addOne` que adiciona $1$ a um número inteiro. Vamos entender como a redução-$\eta$ é aplicada aqui:

Na versão antes da redução-$\eta$: `addOne x = (+ 1) x` define uma função que toma um argumento `x`. Enquanto `(+ 1)` é uma função parcialmente aplicada em Haskell, equivalente a `\y -> $Y$ + 1`. A função `addOne` aplica `(+ 1)` ao argumento `x`.

A redução-$\eta$ nos permite simplificar esta definição: observamos que `x` aparece como o último argumento tanto no lado esquerdo (`addOne x`) quanto no lado direito (`(+ 1) x`) da equação. A redução-$\eta$ permite remover este argumento `x` de ambos os lados.

Após a redução-$\eta$ temos `addOne = (+ 1)` é a forma simplificada. Isso significa que `addOne` é definida como sendo exatamente a função `(+ 1)`.

No cálculo lambda, temos:

$$\lambda x. (\lambda y.\;y + 1) x \to_\eta \lambda y.\;y + 1$$

Em Haskell, isso se traduz em remover o argumento $x$ e a aplicação deste argumento. Graças a redução-$\eta\,$, as duas versões de `addOne` são funcionalmente idênticas. Para qualquer entrada `n`, tanto `addOne n` quanto `(+ 1) n` produzirão o mesmo resultado.

**Exemplo 2**: Composição de Funções

```haskell
-- Antes da redução-$\eta$
compose :: (b -> c) -> (a -> b) -> a -> c
compose f g x = f (g x)

-- Após a redução-$\eta$
compose :: (b -> c) -> (a -> b) -> a -> c
compose f g = f . g
```

Este exemplo demonstra como a redução-$\eta$ pode simplificar a definição de uma função de composição. Detalhando temos:

1. Antes da redução-$\eta$ temos: - `compose f g x = f (g x)` define uma função que toma três argumentos: `f`, `g`, e `x`. Onde `f` é uma função de tipo `b -> c`, `g` é uma função de tipo `a -> b` e `x` é um valor de tipo `a`. Ou seja, a função aplica `g` a `x`, e então aplica `f` ao resultado.

2. Aplicando a redução-$\eta$ observamos que `x` aparece como o último argumento tanto no lado esquerdo (`compose f g x`) quanto no lado direito (`f (g x)`) da equação. A redução-$\eta$ nos permite remover este argumento `x` de ambos os lados.

3. Após a redução-$\eta$ temos: `compose f g = f . g` é a forma simplificada. Neste caso, o operador `.` em Haskell representa a composição de funções. `(f . g)` é equivalente a `\x -> f (g x)`.

No cálculo lambda, temos:

$$\lambda f. \lambda g. \lambda x.\;f\;(g\;x) \to_\eta \lambda f. \lambda g.\;(f \circ g)$$

Onde $\circ$ representa a composição de funções. Em Haskell, isso se traduz em remover o argumento `x` e usar o operador de composição `.`. Desta forma, as duas versão de `compose` são funcionalmente idênticas. Ou seja, para quaisquer funções `f` e `g` e um valor `x`, tanto `compose f g x` quanto `(f . g) x` produzirão o mesmo resultado.

A versão após a redução-$\eta$ expressa mais diretamente o conceito de composição de funções. Já que, elimina a necessidade de mencionar explicitamente o argumento `x`, focando na relação entre as funções `f` e `g`. A redução-$\eta$ neste caso, não somente simplifica a sintaxe, mas destaca a natureza da composição de funções como uma operação sobre funções, em vez de uma operação sobre os valores que essas funções processam.

**Exemplo 3**: Funções de Ordem Superior

```haskell
-- Antes da redução-$\eta$
map' :: (a -> b) -> [a] -> [b]
map' f xs = map f xs

-- Após a redução-$\eta$
map' :: (a -> b) -> [a] -> [b]
map' = map
```

Este exemplo demonstra como a redução-$\eta$ pode simplificar a definição de uma função que envolve outra função de ordem superior. Detalhando o processo temos:

   1. Antes da redução-$\eta$: `map' f xs = map f xs` define uma função `map'` que toma dois argumentos: `f` (uma função) e `xs` (uma lista). Esta função simplesmente aplica a função `map` padrão do Haskell com os mesmos argumentos.

   2. Aplicando a redução-$\eta$: observamos que tanto `f` quanto `xs` aparecem na mesma ordem no lado esquerdo e direito da equação. A redução-$\eta$ nos permite remover ambos os argumentos.

   3. Após a redução-$\eta$: `map' = map` é a forma simplificada. Isso define `map'` como sendo exatamente a função `map` padrão do Haskell.

No cálculo lambda, temos:

$$\lambda f. \lambda xs. (\text{map}\;f\;xs) \to_\eta \lambda f. \lambda xs. \text{map}\;f \to_\eta \text{map}$$

Cada passo remove um argumento, resultando na função `map` por si só.

Do ponto de vista da equivalência funcional temos: `map' f xs` e `map f xs` são funcionalmente idênticas para quaisquer `f` e `xs`. Finalmente, a versão reduzida `map' = map` expressa que `map'` é exatamente a mesma função que `map`. Esta redução mostra que `map'` não adiciona nenhuma funcionalidade extra à `map`. Em um cenário real, se não houver necessidade de uma função wrapper, a definição de `map'` poderia ser completamente omitida, usando-se diretamente `map`.

A redução-$\eta$ neste caso, revela que `map'` é um alias para `map`. Isso demonstra como a redução-$\eta$ pode ajudar a identificar e eliminar definições de funções redundantes, levando a um código mais conciso e direto.

**Exemplo 4**: Funções Parcialmente Aplicadas

```haskell
-- Antes da redução-$\eta$
sumList :: [Int] -> Int
sumList xs = foldr (+) 0 xs

-- Após a redução-$\eta$
sumList :: [Int] -> Int
sumList = foldr (+) 0
```

Este exemplo demonstra como a redução-$\eta$ pode simplificar a definição de uma função que usa aplicação parcial. Vamos detalhar o processo:

1. Antes da redução-$\eta$: temos `sumList xs = foldr (+) 0 xs` define uma função `sumList` que toma uma lista de inteiros `xs` como argumento. A função usa `foldr` (fold right) com o operador `+` e o valor inicial `0` para somar todos os elementos da lista.

2. Aplicando a redução-$\eta$: observamos que `xs` aparece como o último argumento tanto no lado esquerdo (`sumList xs`) quanto no lado direito (`foldr (+) 0 xs`) da equação. A redução-$\eta$ nos permite remover este argumento `xs` de ambos os lados.

3. Após a redução-$\eta$: `sumList = foldr (+) 0` é a forma simplificada. Isso define `sumList` como a aplicação parcial de `foldr` com os argumentos `(+)` e `0`.

No cálculo lambda, temos:

$$\lambda xs. (\text{foldr}\;(+)\;0\;xs) \to_\eta \text{foldr}\;(+)\;0$$

O argumento `xs` é removido, deixando a aplicação parcial de `foldr`.

Podemos ver, outra vez, como isso funciona em programação funcional: ambas as versões de `sumList` são funcionalmente idênticas. Para qualquer lista `xs`, tanto `sumList xs` quanto `foldr (+) 0 xs` produzirão o mesmo resultado. Como nos exemplos anteriores, a versão após a redução-$\eta$ expressa mais diretamente que `sumList` é uma especialização de `foldr`. Elimina a necessidade de mencionar explicitamente o argumento `xs`, focando na operação de soma em si.

Neste exemplo, `foldr (+) 0` é uma função parcialmente aplicada. Ela espera receber uma lista como seu último argumento. O que demonstra como a redução-$\eta$ pode revelar e tirar proveito da aplicação parcial em Haskell. A redução-$\eta$ neste caso, além de simplificar a sintaxe, destaca o conceito de aplicação parcial em programação funcional. Ela mostra como `sumList` pode ser definida como uma especialização de `foldr`, pronta para ser aplicada a uma lista de inteiros.

**Exemplo 5**. Operadores Infixos

```haskell
-- Antes da redução-$\eta$
divideBy :: Int -> Int -> Int
divideBy x\;y = x `div` y

-- Após a redução-$\eta$
divideBy :: Int -> Int -> Int
divideBy = div
```

Este exemplo demonstra como a redução-$\eta$ pode simplificar a definição de uma função que utiliza um operador infixo. Vamos detalhar o processo:

1. Antes da redução-$\eta$ temos: `divideBy x\;y = x` ``div`` `y` que define uma função que usa o operador infixo `div` para divisão inteira. Onde ``div`` é a notação infixa para a função `div`.

   Em Haskell, operadores infixos são funções que são normalmente usadas entre dois argumentos. Qualquer função de dois argumentos pode ser usada como um operador infixo colocando-a entre crases (\`). Neste caso, `x` ``div`` `y` é equivalente a `div x\;y`.

2. Aplicando a redução-$\eta$ observamos que `x` e `y` aparecem na mesma ordem no lado esquerdo e direito da equação. Logo, a redução-$\eta$ nos permite remover ambos os argumentos.

3. Após a redução-$\eta$ temos: `divideBy = div` é a forma simplificada. Isso define `divideBy` como sendo exatamente a função ``div``.

Se considerarmos o cálculo lambda teremos:

$$\lambda x. \lambda y. (x\;\text{div}\;y) \to_\eta \lambda x. \lambda y. \text{div}\;x\;y \to_\eta \text{div}$$

Cada passo remove um argumento, resultando na função `div` por si só. As expressões `divideBy x\;y` e `div x\;y` são funcionalmente idênticas para quaisquer `x` e `y`. A versão reduzida `divideBy = div` deixa claro que `divideBy` é exatamente a mesma função que `div`.

Observe que graças a redução-$\eta$ a definição se torna mais concisa e direta. Revelando que `divideBy` não adiciona nenhuma funcionalidade extra à `div`. Permitindo que `divideBy` seja usado tanto de forma infixa quanto prefixa, assim como `div`. Neste caso, a redução-$\eta$ mostra com um operador infixo pode ser simplificada para revelar sua equivalência direta com a função de divisão padrão. Isso ilustra a flexibilidade do Haskell em tratar operadores e funções intercambiavelmente.

### 3.3.5. 6. Funções Anônimas

```haskell
-- Antes da redução-$\eta$
processList :: [Int] -> [Int]
processList = map (\x -> x * 2)

-- Após a redução-$\eta$
processList :: [Int] -> [Int]
processList = map (* 2)
```

Este exemplo demonstra como a redução-$\eta$ pode simplificar o uso de funções anônimas, conhecidas como lambdas em Haskell. Vamos detalhar o processo:

1. Antes da redução-$\eta$ temos: `processList = map (\x -> x * 2)` define uma função que aplica `map` a uma função anônima. Ou seja, a função anônima `\x -> x * 2` multiplica cada elemento por 2.

2. Aplicando a redução-$\eta$: observamos que a função anônima `\x -> x * 2` pode ser reescrita como uma aplicação parcial do operador `*`. `(\x -> x * 2)` é equivalente a `(* 2)`.

Após a redução-$\eta$ temos: `processList = map (* 2)` é a forma simplificada. Onde `(* 2)` é uma seção em Haskell, uma forma de aplicação parcial para operadores infixos.

No cálculo lambda temos:

$$\lambda x.\;x * 2 \to_\eta (*\;2)$$

Onde o argumento `x` é removido, deixando a aplicação parcial do operador `*`.

No caso da programação funcional, as duas versões de `processList` são funcionalmente idênticas. Para qualquer lista de inteiros, tanto `map (\x -> x * 2)` quanto `map (* 2)` produzirão o mesmo resultado.

Observe que a versão após a redução-$\eta$ é mais concisa e expressiva. Eliminando a necessidade de criar uma função anônima explícita e, ao mesmo tempo, aproveitando a notação de seção do Haskell para operadores infixos.

Finalmente, talvez a amável leitora deva saber que uma seção é uma forma de aplicar parcialmente um operador infixo. Ou seja, `(* 2)` é equivalente a `\x -> x * 2`. Seções podem ser formadas com o operando à esquerda `(2 *)` ou à direita `(* 2)`. Neste caso, a redução-$\eta$ simplifica a sintaxe e demonstra como aproveitar as características da linguagem Haskell, como operadores infixos e seções, para criar código mais conciso e expressivo.

### 3.3.6. A redução-$\eta$ e a Otimização de Compiladores

A redução-$\eta\,$, além de ser um conceito teórico do cálculo lambda e uma técnica de refatoração em programação funcional, tem implicações significativas na otimização de código por compiladores. Ela oferece várias oportunidades para melhorar a eficiência do código gerado, especialmente em linguagens funcionais.

Uma das principais aplicações da redução-$\eta$ na otimização é a eliminação de funções intermediárias desnecessárias. Por exemplo, uma função definida como `f x\;y = g (h x) y` pode ser otimizada para `f = g . h`. Esta simplificação reduz a criação de _closures_ e o número de chamadas de função, resultando em código mais eficiente.

>No contexto da otimização de compiladores e da redução-$\eta$, _closures_ são estruturas de dados que combinam uma função com seu ambiente léxico. Elas surgem quando uma função captura variáveis do escopo externo, permitindo seu acesso mesmo após o fim do escopo original. Em linguagens funcionais, _closures_ são usadas para implementar funções de ordem superior e currying. Do ponto de vista da otimização, a criação e manutenção de _closures_ podem impactar o uso de memória e o desempenho. A redução-$\eta$ pode eliminar _closures_ desnecessárias, como quando uma função apenas repassa argumentos para outra sem modificá-los. Nesse caso, o compilador pode substituir a closure intermediária por uma referência direta à função original, melhorando o uso de memória e o tempo de execução.

A redução-$\eta$ facilita o _inlining_ de funções, uma técnica onde o compilador substitui chamadas de função por seu corpo. Por exemplo, uma definição como `map' f = map f` pode levar o compilador a fazer inline de `map'`, substituindo-a diretamente por ``map``. Isso melhora o desempenho enquanto reduz a alocação de memória para _closures_, o que é particularmente benéfico em linguagens com coleta de lixo, _garbage collector_, como é o cado do Haskell.

Em linguagens que utilizam _currying_extensivamente, a redução-$\eta$ pode otimizar a aplicação parcial de funções. Uma expressão como `addOne = (+) 1` pode ser otimizada para evitar a criação de uma _closure_ intermediária, melhorando tanto o uso de memória quanto o desempenho.

A fusão de funções é outra área onde a redução-$\eta$ pode ser útil. Ela pode facilitar a combinação de múltiplas funções em uma única passagem, como transformar `sum . map (*2)` em uma única função que multiplica e soma em uma operação. Isso reduz o _overhead_ de iterações múltiplas sobre estruturas de dados.

A redução-$\eta$ simplifica a análise de fluxo de dados, permitindo que o compilador rastreie mais facilmente como os valores são usados e transformados. Isso pode levar a otimizações mais eficazes em nível de código de máquina. Em alguns casos, pode até transformar chamadas não-tail em chamadas _tail_, permitindo a otimização de _tail call_, crucial para linguagens que garantem essa otimização, como Scheme.

A simplificação da estrutura das funções através da redução-$\eta$ pode resultar em código de máquina mais eficiente e mais fácil de otimizar posteriormente. Isso pode ajudar na especialização de funções polimórficas, levando a implementações mais eficientes para tipos específicos.

No contexto de otimizações inter-procedurais, a redução-$\eta$ pode facilitar a análise e otimização de funções através de limites de módulos, permitindo otimizações mais abrangentes.

É importante notar que a aplicação dessas otimizações deve ser equilibrada com outros fatores, como tempo de compilação e tamanho do código gerado. Em alguns casos, a redução-$\eta$ pode interferir com outras otimizações ou com a legibilidade do código de depuração. Compiladores modernos para linguagens funcionais, como o GHC para Haskell, incorporam a redução-$\eta$ como parte de um conjunto mais amplo de técnicas de otimização.

Em suma, a redução-$\eta$ desempenha um papel importante na otimização de compiladores, especialmente para linguagens funcionais, contribuindo significativamente para a geração de código mais eficiente e performático.

## 3.4. Teorema de Church-Rosser

Um dos obstáculos enfrentado por Church durante o desenvolvimento do cálculo lambda dizia respeito a consistência do processo de redução. Ou seja, provar que um termo lambda mesmo que reduzido de formas diferentes, chegaria a mesma forma normal, caso esta forma existisse. Em busca desta consistência, Church e [J. Barkley Rosser](https://en.wikipedia.org/wiki/J._Barkley_Rosser), seu estudante de doutorado, formularam o teorema que viria a ser chamado de **Teorema de Church-Rosser**[^cita5]. Este teorema, chamado de propriedade da confluência local, garante a consistência e a previsibilidade do sistema de redução beta, afirmando que, **independentemente da ordem em que as reduções beta são aplicadas, o resultado , se existir, é o mesmo** Figura 3.4.A.

![Um diagrama com um termo principal, M e dois caminhos de redução chegando ao mesmo ponto](/assets/images/conflu.webp)
_Figura 3.4.A: Diagrama da Propriedade de Confluência determinada pelo Teorema de Church-Rosser_{: legenda}

Formalmente teremos:

$$\forall M, N_1, N_2\;(\text{se}\;M \twoheadrightarrow_\beta N_1\;\text{e}\;M \twoheadrightarrow_\beta N_2,\;\text{então existe um}\;P\;\text{tal que}\;N_1 \twoheadrightarrow_\beta P\;\text{e}\;N_2 \twoheadrightarrow_\beta P).$$

Ou:

$$
(M \twoheadrightarrow_\beta N_1 \land M \twoheadrightarrow_\beta N_2) \implies \exists P\;(N_1 \twoheadrightarrow_\beta P \land N_2 \twoheadrightarrow_\beta P)
$$

Onde $\twoheadrightarrow_\beta$ representa uma sequência, possivelmente vazia, de reduções beta. Com um pouco menos de formalidade, podemos ler o enunciado do Teorema de Church-Rosser como:

**Se um termo $M$ pode ser reduzido a $N_1$ e $N_2$ exclusivamente em um passo, então existe um termo $P$ tal que $N_1$ e $N_2$ podem ser reduzidos a $P\,$.**

>A prova de Barendregt utiliza o Lema de Newman, que afirma que um sistema de reescrita é confluentemente terminante se for fortemente normalizante e localmente confluentemente terminante. A prova pode ser dividida em três partes principais:
>
>1. Confluência Local: a confluência local é definida da seguinte forma:
>
>Se $M$ é um termo no cálculo lambda e pode ser reduzido em um passo para dois termos distintos $N_1$ e $N_2\,$, então existe um termo comum $ P$ tal que $N_1$ e $N_2$ podem ser reduzidos em um número finito de passos para $P\,$. Formalmente:
>
>$$M \rightarrow N_1 \quad \text{e} \quad M \rightarrow N_2 \implies \exists P \, : \, N_1 \twoheadrightarrow P \quad \text{e} \quad N_2 \twoheadrightarrow P
$$
>
>Por exemplo: considere o termo $ M = (\lambda x. x \, x) (\lambda x. x \, x)\,$. Esse termo pode ser reduzido de duas formas diferentes:
>
>1. Redução da aplicação externa: $(\lambda x.\;x\;x) (\lambda x.\;x\;x) \rightarrow (\lambda x.\;x\;x) (\lambda x.\;x\;x)$ (permanece o mesmo)
>
>2. Redução da aplicação interna: $(\lambda x.\;x\;x) (\lambda x.\;x\;x) \rightarrow (\lambda x.\;x\;x)$
>
>     No entanto, ambos os caminhos eventualmente se reduzem ao mesmo termo $(\lambda x.\;x\;x)\,$, o que ilustra a confluência local.
>
>3. Confluência Global: a confluência global é estabelecida ao aplicar o Lema de Newman, que afirma que:
>
>     1. Se um sistema de reescrita é _fortemente normalizante_, ou seja, todas as sequências de reduções terminam em um termo normal, e
>
>     2. Se o sistema é _localmente confluentemente terminante_, então ele é _globalmente confluentemente terminante_.
>
>Para aplicar o Lema de Newman no cálculo lambda, é necessário provar duas coisas: a _Normalização forte_, todos os termos no cálculo lambda podem eventualmente ser reduzidos a uma forma normal (caso exista) e a _Confluência local_, que demonstrei anteriormente.
>
>Como o cálculo lambda satisfaz ambas as condições, ele é confluente e terminante globalmente.
>
>A prova completa envolve mostrar que, mesmo quando existem múltiplos >redexes, subtermos que podem ser reduzidos, a ordem de redução não interfere no resultado . Barendregt utiliza as técnicas de _reescrita paralela_ e _substituição simultânea_ para lidar com as reduções múltiplas.
>
>A reescrita paralela envolve a ideia de aplicar todas as reduções possíveis de um termo ao mesmo tempo. Por exemplo, se um termo $M$ contém dois redexes diferentes, como $(\lambda x.\;x)\;(\lambda y.\;y)\,$, a reescrita paralela reduz ambos os redexes simultaneamente:
>
>$$M = (\lambda x.\;x)\;(\lambda y.\;y) \rightarrow (\lambda y.\;y)$$
>
>Essa abordagem simplifica a prova de confluência, pois elimina a necessidade de considerar todas as possíveis sequências de redução.
>
>Já substituição simultânea é usada para manter a consistência ao aplicar várias reduções ao mesmo tempo. Por exemplo, se temos um termo $(\lambda x.\;M)\;N\,$, a substituição simultânea permite que o termo $M[N/x]$ seja avaliado sem considerar ordens de substituição diferentes.
>
>A prova de confluência de Barendregt é considerada elegante devido à sua simplicidade e clareza ao estruturar a demonstração de confluência no cálculo lambda. Notadamente porque: assegura a consistência do cálculo lambda, permite que linguagens de programação baseadas no cálculo lambda sejam previsíveis e determinísticas e tem implicações diretas na teoria da prova, nas linguagens de programação funcional e na lógica computacional. [^cita8]
>

O Teorema de Church-Rosser ao estabelecer que o cálculo lambda é um sistema _confluente_, estabelece que, embora possam existir diferentes caminhos de redução a partir de um termo inicial, todos os caminhos levam a um resultado comum. além de provar a consistência do cálculo lambda, O Teorema de Church-Rosser teve impacto na prova da existência da unicidade da forma normal e da independência da estratégia de redução.

Finalmente podemos dizer que **o cálculo lambda puro é consistente porque como não é possível derivar termos contraditórios, ou ambíguos**. A consistência, por sua vez, implica na confiabilidade em sistemas formais e linguagens de programação funcionais.

Por sua vez, a _Unicidade da Forma Normal_ é uma consequência imediata da consistência. Se um termo $M$ possui uma forma normal, um termo irredutível, então essa forma normal é única. Isso assegura que o processo de computação no cálculo lambda é determinístico em termos do resultado . Até aqui, vimos que o cálculo lambda é consistente e determinístico.

A última consequência do Teorema de Church-Rosser, a _Independência da Estratégia de Redução_ garante que a ordem, ou estratégia, com que as reduções beta são aplicadas não afeta a forma normal final de um termo. Isso permite flexibilidade na implementação de estratégias de avaliação, como avaliação preguiçosa (_lazy evaluation_) ou avaliação estrita (_eager evaluation_).

Finalmente, podemos dizer que o Teorema de Church-Rosser é o suporte necessário para que a amável leitora possa entender que o cálculo lambda é consistente, determinístico e flexível. A esforçada leitora de acompanhar os seguintes exemplos:

**Exemplo 1**: redução de uma Função Aplicada a Dois Argumentos. Considere o termo:

$$M_1 = (\lambda x.\;(\lambda y.\;x + y))\;3\;2$$

Caminho 1: reduza o termo externo primeiro:

$$M_1 = (\lambda x.\;(\lambda y.\;x + y))\;3\;2$$

$$\to_\beta (\lambda y.\;3 + y)\;2$$

Agora, aplique a segunda função:

$$\to_\beta 3 + 2 = 5$$

Caminho 2: reduza o termo interno primeiro:

$$M_1 = (\lambda x.\;(\lambda y.\;x + y))\;3\;2$$

$$\to_\beta (\lambda x.\;(\lambda y.\;x + y))\;3$$

Reduza o termo externo:

$$\to_\beta (\lambda y.\;3 + y)\;2$$

Finalmente, aplique a segunda função:

$$\to_\beta 3 + 2 = 5$$

Neste exemplo, ambos os caminhos de redução resultam na forma normal $5\,$.

**Exemplo 2**: redução de uma Função de Ordem Superior. Considere o termo:

$$M_2 = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;1$$

Caminho 1: reduza o termo externo primeiro:

$$M_2 = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;1$$

$$\to_\beta (\lambda x.\;(\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;x))\;1$$

Aplique a função interna:

$$\to_\beta (\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;1)$$

Aplique a primeira função:

$$\to_\beta (\lambda y.\;y + 1)\;(1 + 1)$$

Aplique a segunda função:

$$\to_\beta 2 + 1 = 3$$

Caminho 2: reduza o termo interno primeiro:

$$M_2 = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;1$$

$$\to_\beta (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)$$

Aplique a função externa:

$$\to_\beta (\lambda x.\;(\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;x))\;1$$

Agora, aplique a função interna:

$$\to_\beta (\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;1)$$

Aplique a primeira função:

$$\to_\beta (\lambda y.\;y + 1)\;(1 + 1)$$

Aplique a segunda função:

$$\to_\beta 2 + 1 = 3$$

Neste exemplo, ambos os caminhos de redução resultam na forma normal $3\,$.

**Exemplo 3**: considere o termo lambda:

$$M = (\lambda x.\;x\;x) (\lambda x.\;x\;x)$$

Este termo pode ser reduzido de formas diferentes:

Caminho 1: reduza o termo externo primeiro:

$$\begin{align*}
M &= (\lambda x.\;x\;x) (\lambda x.\;x\;x) \\
&\to_\beta (\lambda x.\;x\;x) (\lambda x.\;x\;x)
\end{align*}$$

Se tomarmos o caminho 1, este processo continuará indefinidamente, indicando que não há forma normal.

Caminho 2: reduza o termo interno primeiro:

$$\begin{align*}
M &= (\lambda x.\;x\;x) (\lambda x.\;x\;x) \\
&\to_\beta (\lambda x.\;x\;x) (\lambda x.\;x\;x)
\end{align*}$$

Novamente, o processo é infinito e não temos uma forma normal.

Neste caso, ambos os caminhos levam a um processo de redução infinito, mas o Teorema de Church-Rosser assegura que, se houvesse uma forma normal, ela seria única, independentemente do caminho escolhido.

A confluência garantida pelo Teorema de Church-Rosser é análoga a um rio com vários afluentes que eventualmente convergem para o mesmo oceano. Ou se preferir a antiga expressão latina _Omnes viae Romam ducunt_ Não importa qual caminho a água siga, ou que estrada a nômade leitora pegue, ela acabará chegando ao mesmo destino. No contexto do cálculo lambda, isso significa que diferentes sequências de reduções beta não causam ambiguidades no resultado  da computação.

O Teorema de Church-Rosser fornece uma base teórica para otimizações de compiladores e interpretadores, garantindo que mudanças na ordem de avaliação não alterem o resultado . Tem impacto na teoria da computação já que a confluência é uma propriedade desejável em sistemas de reescrita de termos, assegurando a consistência lógica e a previsibilidade dos sistemas formais. Em sistemas de provas formais e lógica matemática o Teorema de Church-Rosser ajuda a garantir que as demonstrações não levem a contradições.

### 3.4.1. Exercícios Relacionados ao Teorema de Church-Rosser

**1**: Reduza o termo a seguir usando dois caminhos diferentes de redução:

$$M = (\lambda x.\;(\lambda y.\;x \times y))\;5\;3$$

   **Solução:**
   Caminho 1: aplique a primeira função:

   $$M = (\lambda x.\;(\lambda y.\;x \times y))\;5\;3$$

   $$\to_\beta (\lambda y.\;5 \times y)\;3$$

   Agora, aplique a segunda função:

   $$\to_\beta 5 \times 3 = 15$$

   Caminho 2: Primeiro, aplique a segunda função:

   $$M = (\lambda x.\;(\lambda y.\;x \times y))\;5\;3$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;x \times y))\;5$$

   Agora, aplique a primeira função:

   $$\to_\beta (\lambda y.\;5 \times y)\;3$$

   Aplique a segunda função:

   $$\to_\beta 5 \times 3 = 15$$

   Ambos os caminhos levam à forma normal $15\,$.

**2**: Mostre dois caminhos distintos de redução para o termo:

$$M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;2$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;2$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;y + 1) ((\lambda y.\;y + 1) x))\;2$$

   Aplique a função interna:

   $$\to_\beta (\lambda y.\;y + 1) ((\lambda y.\;y + 1)\;2)$$

   $$\to_\beta (\lambda y.\;y + 1)\;(2 + 1)$$

   $$\to_\beta 3 + 1 = 4$$

   Caminho 2: primeiro, aplique a função interna:

   $$M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)\;2$$

   $$\to_\beta (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y + 1)$$

   Agora, aplique a função externa:

   $$
   \to_\beta (\lambda x.\;(\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;x))\;2
   $$

   $$\to_\beta (\lambda y.\;y + 1)\;((\lambda y.\;y + 1)\;2)$$

   $$\to_\beta (\lambda y.\;y + 1)\;(2 + 1)$$

   $$\to_\beta 3 + 1 = 4$$

   Ambos os caminhos levam à forma normal $4\,$.

**3**. Mostre que, independentemente do caminho escolhido, o termo resulta na mesma forma normal.

$$M = (\lambda x.\;(\lambda y.\;x + y))\;7\;8$$

   **Solução:**
   Caminho 1: aplique a primeira função:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;7\;8$$
   $$\to_\beta (\lambda y.\;7 + y)\;8$$

   Aplique a segunda função:

   $$
   \to_\beta 7 + 8 = 15
   $$

   Caminho 2: primeiro, aplique a segunda função:

   $$
   M = (\lambda x.\;(\lambda y.\;x + y))\;7\;8 \to_\beta (\lambda x.\;(\lambda y.\;x + y))\;7
   $$

   Agora, aplique a função:

   $$\to_\beta (\lambda y.\;7 + y)\;8$$

   $$\to_\beta 7 + 8 = 15$$

   Ambos os caminhos levam à forma normal $15\,$.

**4**: Prove que o termo satisfaz a confluência para diferentes sequências de reduções.

$$
M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 1)
$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 1)$$

   $$\to_\beta (\lambda x.\;x + 1) ((\lambda x.\;x + 1)\;3)$$

   Aplique a função interna:

   $$\to_\beta (\lambda x.\;x + 1)\;(3 + 1)$$

   $$\to_\beta 4 + 1 = 5$$

   Caminho 2: aplique a função interna primeiro:

   $$
   M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 1) \to_\beta (\lambda f.\;f (f 3))
   $$

   Agora, aplique a função externa:

   $$\to_\beta (\lambda x.\;x + 1) ((\lambda x.\;x + 1)\;3)$$

   $$\to_\beta (\lambda x.\;x + 1)\;(3 + 1)$$

   $$\to_\beta 4 + 1 = 5$$

   Ambos os caminhos levam à forma normal $5\,$.

**5**: Identifique a forma normal do termo abaixo usando diferentes estratégias de redução.

$$
M = (\lambda x.\;(\lambda y.\;x + y))\;6\;4
$$

   **Solução:**
   Caminho 1: aplique a primeira função:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;6\;4$$

   $$\to_\beta (\lambda y.\;6 + y)\;4$$

   Aplique a segunda função:

   $$
   \to_\beta 6 + 4 = 10
   $$

   Caminho 2: aplique a segunda função primeiro:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;6\;4$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;x + y))\;6$$

   Agora, aplique a função:

   $$\to_\beta (\lambda y.\;6 + y)\;4 \to_\beta 6 + 4 = 10$$

   Ambos os caminhos resultam na forma normal $10\,$.

**6**: Considere o termo abaixo e prove que a forma normal é única, independentemente da estratégia de redução.

$$
M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y^2)\;2
$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y^2)\;2$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;y^2)\;((\lambda y.\;y^2)\;x))\;2$$

   Aplique a função interna:

   $$\to_\beta (\lambda y.\;y^2)\;((\lambda y.\;y^2)\;2)$$

   $$\to_\beta (\lambda y.\;y^2)\;(2^2)$$

   $$\to_\beta 4^2 = 16$$

   Caminho 2: aplique a função interna primeiro:

   $$M = (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y^2)\;2$$

   $$\to_\beta (\lambda f.\;(\lambda x.\;f (f x)))\;(\lambda y.\;y^2)$$

   Agora, aplique a função externa:

   $$\to_\beta (\lambda x.\;(\lambda y.\;y^2)\;((\lambda y.\;y^2)\;x))\;2$$

   $$\to_\beta (\lambda y.\;y^2)\;((\lambda y.\;y^2)\;2)$$

   $$\to_\beta (\lambda y.\;y^2)\;(2^2)$$

   $$\to_\beta 4^2 = 16$$

   Ambos os caminhos resultam na forma normal $16\,$.

**7**: Dado o termo abaixo, mostre que a forma normal é alcançada com diferentes estratégias de redução e que a unicidade da forma normal se mantém.

$$
M = (\lambda x.\;(\lambda y.\;x + y))\;8\;2
$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;8\;2$$

   $$\to_\beta (\lambda y.\;8 + y)\;2$$

   $$\to_\beta 8 + 2 = 10$$

   Caminho 2: aplique a função interna primeiro:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;8\;2$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;x + y))\;8$$

   Agora, aplique a função:

   $$\to_\beta (\lambda y.\;8 + y)\;2$$

   $$\to_\beta 8 + 2 = 10$$

   Ambos os caminhos resultam na forma normal $10\,$.

**8**: Considere o termo abaixo. Demonstre que a forma normal é única mesmo utilizando diferentes estratégias de redução.

$$M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 4)$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 4)$$

   $$\to_\beta (\lambda x.\;x + 4) ((\lambda x.\;x + 4)\;3)$$

   Aplique a função interna:

   $$\to_\beta (\lambda x.\;x + 4)\;(3 + 4)$$

   $$\to_\beta 7 + 4 = 11$$

   Caminho 2: aplique a função interna primeiro:

   $$M = (\lambda f.\;f (f 3))\;(\lambda x.\;x + 4)$$

   $$\to_\beta (\lambda f.\;f (f 3))$$

   Agora, aplique a função externa:

   $$\to_\beta (\lambda x.\;x + 4) ((\lambda x.\;x + 4)\;3)$$

   $$\to_\beta (\lambda x.\;x + 4)\;(3 + 4)$$

   $$\to_\beta 7 + 4 = 11$$

   Ambos os caminhos resultam na forma normal $11\,$.

**9**: Dado o termo a seguir, identifique a forma normal e mostre que ela é única, mesmo utilizando diferentes estratégias de redução.

$$M = (\lambda f.\;f (f 1))\;(\lambda x.\;x + 2)$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda f.\;f (f 1))\;(\lambda x.\;x + 2)$$

   $$\to_\beta (\lambda x.\;x + 2) ((\lambda x.\;x + 2)\;1)$$

   Aplique a função interna:

   $$\to_\beta (\lambda x.\;x + 2)\;(1 + 2)$$

   $$\to_\beta 3 + 2 = 5$$

   Caminho 2: aplique a função interna primeiro:

   $$M = (\lambda f.\;f (f 1))\;(\lambda x.\;x + 2)$$

   $$\to_\beta (\lambda f.\;f (f 1))$$

   Agora, aplique a função externa:

   $$\to_\beta (\lambda x.\;x + 2) ((\lambda x.\;x + 2)\;1)$$

   $$\to_\beta (\lambda x.\;x + 2)\;(1 + 2)$$

   $$\to_\beta 3 + 2 = 5$$

   Ambos os caminhos resultam na forma normal $5\,$.

**10**. Considere o termo abaixo. Prove que ele satisfaz a confluência e que a forma normal é única para diferentes estratégias de redução.

$$M = (\lambda x.\;(\lambda y.\;x + y))\;9\;(\lambda z.\;z \times 2)\;4$$

   **Solução:**
   Caminho 1: aplique a função externa:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;9\;(\lambda z.\;z \times 2)\;4$$

   $$\to_\beta (\lambda y.\;9 + y)\;((\lambda z.\;z \times 2)\;4)$$

   Aplique a função interna:

   $$\to_\beta (\lambda y.\;9 + y)\;(4 \times 2)$$

   $$\to_\beta (\lambda y.\;9 + y)\;8$$

   $$\to_\beta 9 + 8 = 17$$

   Caminho 2: aplique a função interna primeiro:

   $$M = (\lambda x.\;(\lambda y.\;x + y))\;9\;(\lambda z.\;z \times 2)\;4$$

   $$\to_\beta (\lambda x.\;(\lambda y.\;x + y))\;9$$

   Agora, aplique a função externa:

   $$\to_\beta (\lambda y.\;9 + y)\;((\lambda z.\;z \times 2)\;4)$$

   $$\to_\beta (\lambda y.\;9 + y)\;8$$

   $$\to_\beta 9 + 8 = 17$$

   Ambos os caminhos resultam na forma normal $17\,$.

## 3.5. Currying

O cálculo Lambda assume intrinsecamente que uma função possui um único argumento. Esta é a ideia por trás da aplicação. Como a atenta leitora deve lembrar: dado um termo $M$ visto como uma função e um argumento $N\,$, o termo $(M\;N)$ representa o resultado de aplicar $M$ ao argumento $N\,$. A avaliação é realizada pela redução-$\beta\,$.

**Embora o cálculo lambda defina funções unárias estritamente, aqui, não nos limitaremos a essa regra para facilitar o entendimento dos conceitos de substituição e aplicação.**

O conceito de _currying_vem do trabalho do matemático [Moses Schönfinkel](https://en.wikipedia.org/wiki/Moses_Sch%C3%B6nfinkel), que iniciou o estudo da lógica combinatória nos anos 1920. Mais tarde, Haskell Curry popularizou e expandiu essas ideias. O cálculo lambda foi amplamente influenciado por esses estudos, tornando o _currying_uma parte essencial da programação funcional e da teoria dos tipos.

Por exemplo, uma função de dois argumentos $f(x, y)$ pode ser convertida em uma sequência de funções $f'(x)(y)\,$. Aqui, $f'(x)$ retorna uma nova função que aceita $y$ como argumento. Assim, uma função que requer múltiplos parâmetros pode ser aplicada parcialmente, fornecendo alguns argumentos de cada vez, resultando em uma nova função que espera os argumentos restantes. Ou, com um pouco mais de formalidade: uma função de $n$ argumentos é vista como uma função de um argumento que toma uma função de $n - 1$ argumentos como argumento.

Considere uma função $f$ que aceita dois argumentos: $f(x, y)$ a  versão _currificada_ desta função será:

   $$F = \lambda x.(\lambda y.\ ; f(x, y))$$

Agora, $F$ é uma função que aceita um argumento $x$ e retorna outra função que aceita $y\,$. Podemos ver isso com um exemplo: suponha que temos uma função que soma dois números: $soma(x, y) = x + y\,$. A versão _currificada_ seria:

$$add = \lambda x.\;(\lambda y.\;(x + y))$$

Isso significa que $add$ é uma função que recebe um argumento $x$ e retorna outra função $\lambda y.(\;x + y)\,$. Esta função resultante espera um segundo argumento $y$ para calcular a soma de $x$ e $y\,$.

Quando aplicamos $add$ ao argumento $3\,$, obteremos:

   $$(add \; 3) = (\lambda x.\;(\lambda y.\;(x + y))) \; 3$$

Nesse ponto, estamos substituindo $x$ por $3$ na função externa, resultando em:

   $$\lambda y.\;(3 + y)$$

Isso é uma nova função que espera um segundo argumento $y\,$. Agora, aplicamos o segundo argumento, $4\,$, à função resultante:

   $$(\lambda y.\;(3 + y))\;4$$

Substituímos $y$ por $4\,$, obtendo:

   $$3 + 4$$

Finalmente:

   $$7$$

Assim, $(add \;3) \;4$ é avaliado para $7$ após a aplicação sequencial de argumentos à função currificada. A Figura 3.5.A, apresenta a aplicação $(add \; 3) = (\lambda x.\;(\lambda y.\;(x + y))) \; 3$ que explicamos acima.

![Diagrama da função add currificada como explicado anteriormente](/assets/images/curry.webp)
_Figura 3.5.A: Diagrama mostrando o processo de _currying_em Cálculo lambda_{: legenda}

No _currying_, uma função que originalmente recebe dois argumentos, como $f: \mathbb{N} \times \mathbb{N} \rightarrow \mathbb{N}\,$, é transformada em uma função que recebe um argumento e retorna outra função. O resultado é uma função da forma $f': \mathbb{N} \rightarrow (\mathbb{N} \rightarrow \mathbb{N})\,$. Assim, $f'$ recebe o primeiro argumento e retorna uma nova função que espera o segundo argumento para realizar o cálculo final.

Podemos representar essa transformação de forma mais abstrata usando a notação da teoria dos conjuntos. Uma função que recebe dois argumentos é representada como $\mathbb{N}^{\mathbb{N} \times \mathbb{N}}\,$, o que significa "o conjunto de todas as funções que mapeiam pares de números naturais para números naturais". Quando fazemos *currying*, essa função é transformada em $(\mathbb{N}^{\mathbb{N}})^{\mathbb{N}}\,$, o que significa "o conjunto de todas as funções que mapeiam um número natural para outra função que, por sua vez, mapeia um número natural para outro". Assim, temos uma cadeia de funções aninhadas.

Podemos fazer uma analogia com a álgebra:

$$(m^n)^p = m^{n \cdot p}$$

Aqui, elevar uma potência a outra potência equivale a multiplicar os expoentes. Similarmente, no currying, estruturamos as funções de forma aninhada, mas o resultado é equivalente, independentemente de aplicarmos todos os argumentos de uma vez ou um por um. Portanto, o currying cria um isomorfismo entre as funções dos tipos:

$$f : (A \times B) \to C$$

e

$$g : A \to (B \to C)$$

Este _isomorfismo_ significa que as duas formas são estruturalmente equivalentes e podem ser convertidas uma na outra sem perda de informação ou alteração do comportamento da função. A função $f$ que recebe um par de argumentos $(a, b)$ é equivalente à função $g$ que, ao receber $a\,$, retorna uma nova função que espera $b\,$, permitindo que os argumentos sejam aplicados um por vez.

Podemos entender melhor o conceito de _currying_dentro de um contexto mais abstrato, o da teoria das categorias. A teoria das categorias é uma área da matemática que busca generalizar e estudar relações entre diferentes estruturas matemáticas através de objetos e mapeamentos entre eles, chamados de morfismos. No caso do _currying_, ele se encaixa no conceito de uma _categoria fechada cartesiana (CCC)_. Uma _CCC_ é uma categoria que possui certas propriedades que tornam possível a definição e manipulação de funções de forma abstrata, incluindo a existência de produtos, como pares ordenados de elementos, e exponenciais, que são equivalentes ao conjunto de todas as funções entre dois objetos.

No contexto do _currying_, uma _CCC_ permite que funções multivariadas sejam representadas similarmente as funções que aceitam um argumento por vez. Por exemplo, quando representamos uma função como $f: A \times B \to C\,$, estamos dizendo que $f$ aceita um par de argumentos e retorna um valor. Com _currying_, essa função pode ser reestruturada como $g: A \to (B \to C)\,$, onde $g$ aceita um argumento $a\,$, e retorna uma nova função que aceita um argumento $b$ para então retornar o valor final. Esse tipo de reestruturação só é possível porque as operações básicas de uma CCC, como produto e exponenciais, garantem que esse tipo de transformação é sempre viável e consistente. Assim, a noção de categoria fechada cartesiana formaliza a ideia de que funções podem ser aplicadas um argumento de cada vez, sem perder a equivalência com uma aplicação de múltiplos argumentos simultâneos.

Essa estrutura abstrata da teoria das categorias ajuda a explicar por que o _currying_é uma ferramenta tão natural no cálculo lambda e na programação funcional. No cálculo lambda, todas as funções são unárias por definição; qualquer função que precise de múltiplos argumentos é, na verdade, uma cadeia de funções unárias que se encaixam umas nas outras. Esse comportamento é um reflexo direto das propriedades de uma categoria fechada cartesiana. Cada vez que transformamos uma função multivariada em uma sequência de funções aninhadas, estamos explorando a propriedade exponencial dessa categoria, que se comporta de forma semelhante à exponenciação que conhecemos na álgebra. A identidade $(m^n)^p = m^{n \cdot p}$ é um exemplo de como uma estrutura aninhada pode ser vista de forma equivalente a uma única operação combinada.

Entender o _currying_como uma parte de uma categoria fechada cartesiana nos permite uma visão mais profunda sobre como a programação funcional e o cálculo lambda operam. O _currying_não é simplesmente uma técnica prática para simplificar a aplicação de funções; é, na verdade, uma manifestação de uma estrutura matemática mais ampla, que envolve composição, abstração e a criação de novas funções a partir de funções existentes. Essa perspectiva ajuda a conectar o ato prático de currificar funções com a teoria abstrata que fundamenta essas operações, revelando a elegância que há na reestruturação de funções e na capacidade de manipular argumentos um por um.

Voltando ao cálculo lambda, a atenta leitora deve lembrar que todas as funções são unárias, por definição. Assim, funções que parecem aceitar múltiplos argumentos são, na verdade, uma cadeia de funções que retornam outras funções. Vamos ilustrar isso com um exemplo usando a concatenação de strings.

Para começar, vamos definir uma função que concatena duas strings:

$$\text{concat} = \lambda s_1.\; (\lambda s_2.\; s_1 + s_2)$$

Aqui, $\lambda s_1$ cria uma função que recebe a primeira string $s_1$ e retorna outra função $\lambda s_2\,$, que então recebe a segunda string $s_2$ e realiza a concatenação. Quando aplicamos o primeiro argumento, obtemos:

$$(\text{concat}\;\text{"Hello, "}) = \lambda s_2.\;(\text{"Hello, "} + s_2)$$

Agora, podemos aplicar o segundo argumento:

$$(\lambda s_2.\;(\text{"Hello, "} + s_2))\;\text{"World!"} = \text{"Hello, World!"}$$

Outro exemplo seria uma função que cria uma saudação personalizada. Primeiro, vamos definir a função currificada:

$$\text{saudacao} = \lambda nome.\; (\lambda saudacao.\; saudacao + ", " + nome)$$

Aplicando o primeiro argumento:

$$(\text{saudacao}\;\text{"Alice"}) = \lambda saudacao.\;(\text{saudacao} + ", " + \text{"Alice"})$$

E, em seguida, aplicando o segundo argumento:

$$(\lambda saudacao.\;(\text{saudacao} + ", " + \text{"Alice"}))\;\text{"Bom dia"} = \text{"Bom dia, Alice"}$$

Esses exemplos mostram como o _currying_facilita a aplicação parcial de funções, especialmente em contextos onde queremos criar funções específicas a partir de funções mais gerais, aplicando somente alguns dos, ou um argumento inicialmente. Obtendo mais flexibilidade e modularidade no desenvolvimento de nossas funções.

### 3.5.1. _currying_em Haskell

Haskell implementa o _currying_por padrão para todas as funções. Isso significa que cada função em Haskell tecnicamente aceita somente um argumento, mas pode ser aplicada parcialmente para criar novas funções. Podemos definir uma função de múltiplos argumentos assim:

```haskell
add :: Int -> Int -> Int
add x $Y$ = x + y
```

Essa definição é equivalente a:

```haskell
add :: Int -> (Int -> Int)
add = \x -> (\y -> x + y)
```

Aqui, `add` é uma função que aceita um `Int` e retorna uma função que aceita outro `Int` e retorna a soma.

A aplicação parcial é trivial em Haskell. Sempre podemos criar novas funções simplesmente não fornecendo todos os argumentos:

```haskell
addCinco :: Int -> Int
addCinco = add 5

resultado :: Int
resultado = addCinco 3  -- Retorna 8
```

Além da definição de funções usando _currying_e da aplicação parcial. O uso do _currying_no Haskell torna as funções de ordem superior naturais em Haskell:

```haskell
aplicaDuasVezes :: (a -> a) -> a -> a
aplicaDuasVezes f x = f (f x)

incrementaDuasVezes :: Int -> Int
incrementaDuasVezes = aplicaDuasVezes (+1)

resultado :: Int
resultado = incrementaDuasVezes 5  -- Retorna 7
```

Operadores infixos são funções binárias (que aceitam dois argumentos) escritas entre seus operandos. Por exemplo, `+`, `-`, `*`, `/` são operadores infixos comuns. Em Haskell, operadores infixos podem ser facilmente convertidos em funções currificadas usando seções.

Seções são uma característica do Haskell que permite a aplicação parcial de operadores infixos. Elas são uma forma concisa de criar funções anônimas a partir de operadores binários.

1. **Definição**:
   Uma seção é criada ao colocar um operador infixo e um de seus operandos entre parênteses. Isso cria uma função que espera o operando faltante.

```haskell
dobra :: Int -> Int
dobra = (*2)

metade :: Float -> Float
metade = (/2)
```

Finalmente em Haskell o uso do _currying_permite escrever código mais conciso e expressivo. Enquanto facilita a criação de funções especializadas a partir de funções mais gerais e torna a composição de funções mais natural e intuitiva.

### 3.5.2. Exercícios Currying

**1**: escreva uma expressão lambda que representa a função $f(x, y) \, = x + y$ usando currying. Aplique-a aos valores $x = 4$ e $y = 5\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;x + y\,$. Aplicando $x = 4$ e $y = 5$:

   $$(\lambda x. \lambda y.\;x + y)\;4\;5 = 4 + 5 = 9$$

**2**: transforme a função $f(x, y, z) \, = x \times $Y$ + z$ em uma expressão lambda usando _currying_e aplique-a aos valores $x = 2\,$, $y = 3\,$, e $z = 4\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;\lambda z.\;x \times $Y$ + z\,$. Aplicando $x = 2\,$, $y = 3\,$, e $z = 4$:

   $$(\lambda x. \lambda y.\;\lambda z.\;x \times $Y$ + z)\;2\;3\;4 = 2 \times 3 + 4 = 6 + 4 = 10$$

**3**: crie uma função curried que representa $f(x, y) \, = x^2 + y^2\,$. Aplique a função a $x = 1$ e $y = 2\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;x^2 + y^2\,$. Aplicando $x = 1$ e $y = 2$:

   $$(\lambda x. \lambda y.\;x^2 + y^2)\;1\;2 = 1^2 + 2^2 = 1 + 4 = 5$$

**4**: converta a função $f(x, y) \, = \frac{x}{y}$ em uma expressão lambda usando _currying_e aplique-a aos valores $x = 9$ e $y = 3\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;\frac{x}{y}\,$. Aplicando $x = 9$ e $y = 3$:

   $$(\lambda x. \lambda y.\;\frac{x}{y})\;9\;3 = \frac{9}{3} = 3$$

**5**: defina uma função curried que calcule a diferença entre dois números, ou seja, $f(x, y) \, = x - y\,$, e aplique-a aos valores $x = 8$ e $y = 6\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;x - y\,$. Aplicando $x = 8$ e $y = 6$:

   $$(\lambda x. \lambda y.\;x - y)\;8\;6 = 8 - 6 = 2$$

**6**: crie uma função curried para calcular a área de um retângulo, ou seja, $f(l, w) \, = l \times w\,$, e aplique-a aos valores $l = 7$ e $w = 5\,$.

   **Solução:** A função curried é $\lambda l. \lambda w. l \times w\,$. Aplicando $l = 7$ e $w = 5$:

   $$(\lambda l. \lambda w. l \times w)\;7\;5 = 7 \times 5 = 35$$

**7**: transforme a função $f(x, y) \, = x^y$(potência) em uma expressão lambda usando _currying_e aplique-a aos valores $x = 2$ e $y = 3\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;x^y\,$. Aplicando $x = 2$ e $y = 3$:

   $$(\lambda x. \lambda y.\;x^y)\;2\;3 = 2^3 = 8$$

**8**: defina uma função curried que represente a multiplicação de três números, ou seja, $f(x, y, z) \, = x \times $Y$ \times z\,$, e aplique-a aos valores $x = 2\,$, $y = 3\,$, e $z = 4\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;\lambda z.\;x \times $Y$ \times z\,$. Aplicando $x = 2\,$, $y = 3\,$, e $z = 4$:

   $$(\lambda x. \lambda y.\;\lambda z.\;x \times $Y$ \times z)\;2\;3\;4 = 2 \times 3 \times 4 = 24$$

**9**: transforme a função $f(x, y) \, = x + 2y$ em uma expressão lambda curried e aplique-a aos valores $x = 1$ e $y = 4\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;x + 2y\,$. Aplicando $x = 1 $ e $ $Y$ = 4$:

   $$(\lambda x. \lambda y.\;x + 2y)\;1\;4 = 1 + 2 \times 4 = 1 + 8 = 9$$

**10**: crie uma função curried para representar a soma de três números, ou seja, $f(x, y, z) \, = x + $Y$ + z\,$, e aplique-a aos valores $x = 3\,$, $y = 5\,$, e $z = 7\,$.

   **Solução:** A função curried é $\lambda x. \lambda y.\;\lambda z.\;x + $Y$ + z\,$. Aplicando $x = 3\,$, $y = 5\,$, e $z = 7$:

   $$(\lambda x. \lambda y.\;\lambda z.\;x + $Y$ + z)\;3\;5\;7 = 3 + 5 + 7 = 15$$

### 3.5.3. Ordem Normal e Estratégias de Avaliação

A ordem em que as reduções beta são aplicadas pode afetar tanto a eficiência quanto a terminação do cálculo. Existem duas principais estratégias de avaliação:

1. **Ordem Normal**: Sempre reduz o redex mais externo à esquerda primeiro. Essa estratégia garante encontrar a forma normal de um termo, se ela existir. Na ordem normal, aplicamos a função antes de avaliar seus argumentos.

2. **Ordem Aplicativa**: Nesta estratégia, os argumentos são reduzidos antes da aplicação da função. Embora mais eficiente em alguns casos, pode não terminar em expressões que a ordem normal resolveria.

A Figura 3.5.3.A apresenta um diagrama destas duas estratégias de avaliação.

![](/assets/images/normvsaplic.webp)
 _Figura 3.5.3.A: Diagrama de Aplicação nas Ordens Normal e Aplicativa_.{: legenda}

Talvez a atenta leitora entenda melhor vendo as reduções sendo aplicadas:

   $$(\lambda x.\;y)(\lambda z.\;z\;z)$$

1. **Ordem Normal**: A função $(\lambda x.\;y)$ é aplicada diretamente ao argumento $(\lambda z.\;z\;z)\,$, resultando em:

   $$(\lambda x.\;y)(\lambda z.\;z\;z) \to_\beta y$$

Aqui, não precisamos avaliar o argumento, pois a função simplesmente retorna $y\,$.

2. **Ordem Aplicativa**: Primeiro, tentamos reduzir o argumento $(\lambda z.\;z\;z)\,$, resultando em uma expressão que se auto-aplica indefinidamente, causando um loop infinito:

   $$(\lambda x.\;y)(\lambda z.\;z\;z) \to_\beta (\lambda x.\;y)((\lambda z.\;z\;z)(\lambda z.\;z\;z)) \to_\beta ...$$

Este exemplo mostra que a ordem aplicativa pode levar a uma não terminação em termos onde a ordem normal poderá encontrar uma solução.

## 3.6. Combinadores e Funções Anônimas

Os combinadores tem origem no trabalho de [Moses Schönfinkel](https://en.wikipedia.org/wiki/Moses_Sch%C3%B6nfinkel). Em um artigo de 1924 Moses Schönfinkel define uma família de combinadores incluindo os combinadores padrão $S\,$, $K$ e $I$ e demonstra que apenas $S$ e $K$ são necessários[^cite3]. Um conjunto dos combinadores iniciais pode ser visto na Tabela 3.6.A:

| Abreviação Original | Função Original em Alemão    | Tradução para o Inglês     | Expressão Lambda                       | Abreviação Atual |
|---------------------|-----------------------------|----------------------------|----------------------------------------|-----------------|
| $I$                 | Identitätsfunktion           | função identidade         | $\lambda x.\;x$                         | $I$             |
| $K$                 | Konstanzfunktion             | função de constância      | $\lambda\;y\;x.\;x$                        | $C$             |
| $T$                 | Vertauschungsfunktion        | função de troca           | $\lambda\;y\;xz.\;z\;y\;x$                     | $C$             |
| $Z$                 | Zusammensetzungsfunktion     | função de composição      | $\lambda\;y\;xz.\;xz(yz)$                  | $B$             |
| $S$                 | Verschmelzungsfunktion       | função de fusão           | $\lambda\;y\;xz.\;xz(yz)$                  | $S$             |

_Tabela 3.6.A: Relação dos Combinadores Originais._{: legenda}

A Figura 3.6.A mostra as definições dos combinadores $I\,$, $K\,$, $S\,$, e uma aplicação de exemplo de cada um.

![A figura mostra os combinadores I, K e S em notação lambda e a aplicação destes combinadores em exemplos simples.](/assets/images/comb.webp)
_Figura 3.6.A: Definição e Aplicação dos Combinadores $I\,$, $K\,$, $S$_{: legenda}

Schönfinkel apresentou combinadores para representar as operações da lógica de primeiro grau, um para o [traço de Sheffer](https://en.wikipedia.org/wiki/Sheffer_stroke), _NAND_, descoberto em 1913, e outro para a quantificação.

>Em funções booleanas e no cálculo proposicional, o _traço de Sheffer_ é uma operação lógica que representa a negação da conjunção. Essa operação é expressa em linguagem comum como _não ambos_. Ou seja, dados dois operandos, ao menos um deles deve ser falso. Em termos técnicos, essa operação é chamada de _não-conjunção_, _negação alternativa_ ou _NAND_, dependendo do texto onde estão sendo analisados. Esta operação simplesmente nega a conjunção dos operandos e esta é a origem da nomenclatura _NAND_ a abreviação de _Not AND_.
>
>Esta operação foi introduzida pelo filósofo e lógico [Henry Maurice Sheffer](https://en.wikipedia.org/wiki/Henry_M._Sheffer), por isso o nome, em 1913.
>
>O trabalho que definiu o traço de Sheffer demonstrou que todas as operações booleanas podem ser expressas usando somente a operação _NAND_, simplificando a lógica proposicional. Em lógica de primeira ordem representamos esta a operação _NAND_ por $ \mid \,$, $\uparrow\,$, ou $\overline{\wedge}\,$. Não é raro que neófitos confundam a representação do traço de Sheffer com $\vert \vert\,$, que normalmente é usado para representar disjunção. A precavida leitora deve tomar cuidado com isso.
>
>Formalmente, a operação $p \mid q$ pode ser expressa como:
>
>  $$p \mid q = \neg (p \land q) $$
>
>Indicando que a operação do Traço de Sheffer é verdadeira quando a proposição $p \land q$ é falsa. Quando não ambos $p$ e $q$ são verdadeiros.
>
>O Traço de Sheffer possuí as seguintes propriedades:
>
>1. **Universalidade**: a operação _NAND_ é uma operação lógica _universal_, significando que qualquer função booleana pode ser construída apenas com _NANDs_. Isso é particularmente importante em eletrônica digital. A popularidade dessa porta em circuitos digitais se deve à sua versatilidade e à sua implementação relativamente simples em termos de hardware.
>
>2. **Identidade**: O traço de Sheffer é auto-dual e pode representar qualquer outra operação lógica. Podemos representar a disjunção, a conjunção, a negação, o condicional ou a bicondicional através de combinações específicas de _NANDs_.
>
>A importância da _NAND_ pode ser verificada construindo uma operação de negação ($\neg p$) usando o traço de Sheffer, podemos fazê-lo com a seguinte expressão:
>
>  $$\neg p = p \mid p $$
>
>Neste caso, $p \mid p$ significa "não ambos $p$ e $p$", ou seja, simplesmente $\neg p\,$.
>
>| $p$ | $p \mid p$ | $\neg p$ |
>|-----|------------|----------|
>|  V  |     F      |    F     |
>|  F  |     V      |    V     |
>
>Quando $p$ é verdadeiro ($V$): $p \mid p$ é falso, pois o operador _NAND_ aplicado a dois valores verdadeiros resulta em falso. Portanto, $\neg p$ é falso. Quando $p$ é falso ($F$): $p \mid p$ é verdadeiro, pois o operador _NAND_ aplicado a dois valores falsos resulta em verdadeiro. Portanto, $\neg p$ é verdadeiro.
>
>Este exemplo simples ilustra como a operação _NAND_ pode ser usada como um bloco de construção para criar outras operações lógicas.

Schönfinkel, inspirado em Sheffer, buscou reduzir a lógica de predicados ao menor número possível de elementos, e, anos mais tarde, descobriu-se que os quantificadores _para todo_ e _existe_ da lógica de predicados se comportam como abstrações lambda.

>O que Schönfinkel e seus sucessores descobriram é que **os quantificadores universais ($\forall$) e existenciais ($\exists$) podem ser modelados como abstrações lambda**. A estrutura da lógica de predicados é compatível com as regras de abstração e aplicação do cálculo lambda.
>
>O quantificador universal $\forall x. P(x)$ pode ser interpretado como uma função que, dada uma variável $x\,$, retorna verdadeiro para todos os valores de $x$ que satisfazem $P(x)\,$. Esta interpretação esta alinhada com o conceito de abstração lambda, onde uma função recebe um argumento e retorna um valor dependendo desse argumento. Em termos de cálculo lambda, poderíamos expressar o quantificador universal:
>
>  $$\forall x. P(x) \equiv (\lambda x. P(x)) $$
>
>Aqui, a função $\lambda x. P(x)$ é uma abstração que, para cada $x\,$, verifica a verdade de $P(x)\,$.
>
>Da mesma forma, o quantificador existencial $\exists x. P(x)$ pode ser interpretado como a aplicação de uma função que verifica se existe algum valor de $x$ que torna $P(x)$ verdadeiro. Novamente, isso pode ser modelado como uma abstração lambda:
>
>  $$\exists x. P(x) = (\lambda x. \neg P(x)) $$
>
>Essa correspondência revela a natureza fundamental das abstrações lambda e sua aplicação além do cálculo lambda puro.

Para nós, neste momento, um combinador é uma _expressão lambda_ fechada, ou seja, sem variáveis livres. Isso significa que todas as variáveis usadas no combinador estão ligadas dentro da própria expressão.

A perceptiva leitora deve observar que o poder dos combinadores surge de que eles permitem criar funções complexas usando blocos simples, sem a necessidade de referenciar variáveis externas aos blocos.

Começamos com o combinador $K\,$, definido como:

$$K = \lambda x.\lambda y.\;x$$

Este combinador é uma função de duas variáveis que sempre retorna o primeiro argumento, ignorando o segundo. Ele representa o conceito de uma função constante. As funções constante sempre retornam o mesmo valor. No cálculo lambda o combinador $K$ sempre retorna o primeiro argumento independentemente do segundo.

Por exemplo, $KAB$ reduz para $A\,$, sem considerar o valor de $B$:

$$KAB = (\lambda x.\lambda y.x)AB \rightarrow_\beta (\lambda y.A)B \rightarrow_\beta A$$

Os três combinadores, a seguir, são referenciados como básicos e estruturais para a definição de funções compostas em cálculo lambda:

1.**Combinador I (Identidade)**:

$$I = \lambda x.\;x$$

 O combinador identidade retorna o valor que recebe como argumento, sem modificá-lo.

 **Exemplo**: aplicando o combinador $I$ a qualquer valor, ele retornará esse mesmo valor:

$$I\;5 \rightarrow_\beta 5$$

 Outro exemplo:

$$I\;(\lambda y.\;y + 1) \rightarrow_\beta \lambda y.\;y + 1$$

2.**Combinador K (ou C de Constante)**:

$$K = \lambda x.\lambda y.x$$

 Este combinador ignora o segundo argumento e retorna o primeiro.

 **Exemplo**: Usando o combinador $K$ com dois valores:

$$K\;7\;4 \rightarrow_\beta (\lambda x.\lambda y.\;x)\;7\;4 \rightarrow_\beta (\lambda y.\;7)\;4 \rightarrow_\beta 7$$

 Aqui, o valor $7$ é retornado, e o valor $4$ ignorando.

3.**Combinador S (Substituição)**:

$$S = \lambda f.\lambda g.\lambda x.\;fx(gx)$$

 Este combinador é mais complexo, pois aplica a função $f$ ao argumento $x$ e, simultaneamente, aplica a função $g$ a $x\,$, passando o resultado de $g(x)$ como argumento para $f\,$.

 **Exemplo**: Vamos aplicar o combinador $S$ com as funções $f = \lambda z.\;z^2$ e $g = \lambda z.\;z + 1\,$, e o valor $3$:

$$S\;(\lambda z.\;z^2)\;(\lambda z.\;z + 1)\;3$$

 Primeiro, substituímos $f$ e $g$:

$$\rightarrow_\beta (\lambda x.(\lambda z.\;z^2)\;x\;((\lambda z.\;z + 1)\;x))\;3$$

 Agora, aplicamos as funções:

$$\rightarrow_\beta (\lambda z.\;z^2)\;3\;((\lambda z.\;z + 1)\;3)$$

$$\rightarrow_\beta 3^2\;(3 + 1)$$

$$\rightarrow_\beta 9\;4$$

 Assim, $S\;(\lambda z.\;z^2)\;(\lambda z.\;z + 1)\;3$ resulta em $9\,$.

Finalmente, a lista de combinadores do cálculo lambda é um pouco mais extensa [^cita2]:

| Nome | Definição e Comentários |
|------|-------------------------|
| **S** | $\lambda x [\lambda $Y$ [\lambda z [x z (y\;z)]]]\,$. Lembre-se que $x z (y\;z)$ deve ser entendido como a aplicação $(x z)(y\;z)$ de $x z$ a $y\;z\,$. O combinador $S$ pode ser entendido como um operador de _substituir e aplicar_: $z$ _intervém_ entre $x$ e $y$; em vez de aplicar $x$ a $y\,$, aplicamos $x z$ a $y\;z\,$. |
| **K** | $\lambda x [\lambda $Y$ [x]]\,$. O valor de $K M$ é a função constante cujo valor para qualquer argumento é simplesmente $M\,$. |
| **I** | $\lambda x [x]\,$. A função identidade. |
| **B** | $\lambda x [\lambda $Y$ [\lambda z [x (y\;z)]]]\,$. Lembre-se que $x\;y\;z$ deve ser entendido como $(x\;y) z\,$, então este combinador não é uma função identidade trivial. |
| **C** | $\lambda x [\lambda $Y$ [\lambda z [x z y]]]\,$. Troca um argumento. |
| **T** | $\lambda x [\lambda $Y$ [x]]\,$. Valor verdadeiro lógico (True). Idêntico a $K\,$. Veremos mais tarde como essas representações dos valores lógicos desempenham um papel na fusão da lógica com o cálculo lambda. |
| **F** | $\lambda x [\lambda $Y$ [y]]\,$. Valor falso lógico (False). |
| **ω** | $\lambda x [x\;x]\,$. Combinador de autoaplicação. |
| **Ω** | $\omega \omega\,$. Autoaplicação do combinador de autoaplicação. Reduz para si mesmo. |
| **Y** | $\lambda f [(\lambda x [f (x\;x)]) (\lambda x [f (x\;x)])]\,$. O combinador paradoxal de Curry. Para todo termo lambda $X\,$, temos: $Y X \triangleright (\lambda x [X (x\;x)]) (\lambda x [X (x\;x)]) \triangleright X ((\lambda x [X (x\;x)]) (\lambda x [X (x\;x)]))\,$. A primeira etapa da redução mostra que $Y X$ reduz ao termo de aplicação $(\lambda x [X (x\;x)]) (\lambda x [X (x\;x)])\,$, que reaparece na terceira etapa. Assim, $Y$ tem a propriedade curiosa de que $Y X$ e $X (Y X)$ reduzem a um termo comum. |
| **Θ** | $(\lambda x [\lambda f [f (x\;x f)]]) (\lambda x [\lambda f [f (x\;x f)]])\,$. O combinador de ponto fixo de Turing. Para todo termo lambda $X\,$, $Θ X$ reduz para $X (Θ X)\,$, o que pode ser confirmado manualmente. (O combinador paradoxal de Curry $Y$ não tem essa propriedade.) |

_Tabela 3.6.B: Definições e Observações sobre os Combinadores._{: legenda}

No cálculo lambda as funções são anônimas. Desta forma, sempre é possível construir funções sem a atribuição nomes explícitos. Aqui estamos próximos da álgebra e longe das linguagens de programação imperativas, baseadas na Máquina de Turing. Isso é possível, como a atenta leitora deve lembrar, graças a existência das  _abstrações lambda_:

$$\lambda x.\;(\lambda y.\;y)\;x$$

A abstração lambda a cima, representa uma função que aplica a função identidade ao seu argumento $x\,$. Nesse caso, a função interna $\lambda y.\;y$ é aplicada ao argumento $x\,$, e o valor resultante é simplesmente $x\,$, já que a função interna é a identidade. Estas funções inspiraram a criação de funções anônimas e alguns operadores em linguagens de programação imperativas. Como as funções _arrow_ em JavaScript ou às funções _lambdas_ em C++ e Python.

Os combinadores ampliam a utilidade das funções lambda permitem a criação de funções complexas sem o uso de variáveis nomeadas. Esse processo, conhecido como _abstração combinatória_, elimina a necessidade de variáveis explícitas, focando em operações com funções. Podemos ver um exemplo de combinador de composição, denotado como $B\,$, definido por:

$$B = \lambda f.\lambda g.\lambda x.\;f\;(g\;x)$$

Aqui, $B$ aplica a função $f$ ao resultado da função $g\,$, ambas aplicadas a $x\,$. Esse é um exemplo clássico de um combinador que não utiliza variáveis explícitas e demonstra o poder do cálculo lambda puro, onde toda computação pode ser descrita através de combinações.

Podemos ver um outro exemplo na construção do combinador _Mockingbird_, ou $M\,$. Um combinador que aplica uma função a si mesma, definido por:

$$M = \lambda f.\;f\;f$$

Sua função é replicar a aplicação de uma função sobre si mesma, o que é fundamental em certas construções dentro do cálculo lambda, mas não se relaciona com o comportamento do combinador de composição.

Mesmo correndo o risco de ser redundante e óbvio preciso destacar que combinadores podem ser combinados. A expressão $S\;(K\;S)\;K$ é uma combinação de combinadores que possui um comportamento interessante. Podemos a analisar a estrutura do termo $S\;(K\;S)\;K$ observando que: o combinador $S$ é definido como $S = \lambda f.\lambda g.\lambda x.\;f\;x\;(g\;x)\,$, que aplica a função $f$ ao argumento $x\,$, e depois aplica $g$ ao mesmo argumento $x\,$, combinando os resultados; e o combinador $K$ é definido como $K = \lambda x.\lambda y.\;x\,$, que retorna sempre o primeiro argumento, ignorando o segundo. Voltando ao termo original:

$$S\;(K\;S)\;K$$

A amável leitora deve ficar atenta a redução:

1. Primeiro, aplicamos $S$ ao primeiro argumento $(K\;S)$ e ao segundo argumento $K$:

   $$S(KS)K \rightarrow \lambda x.\;(K\;S)\;x (K\;x)$$

2. O termo $(KS)$ é aplicado a $x\,$, o que nos dará:

   $$(KS) x = (\lambda y.\;S)\;x = S$$

3. Agora, temos:

   $$S(KS)K \rightarrow \lambda x.\;S\;(K\;x)$$

Neste ponto, o combinador $S$ permanece, e o segundo termo $K\;x$ simplesmente retorna a constante $K\,$. O que resulta dessa combinação é uma forma que pode ser usada em certos contextos onde se deseja replicar parte do comportamento de duplicação de funções, similar ao combinador _Mockingbird_, mas com características próprias.

A capacidade de expressar qualquer função computável usando somente combinadores é formalizada pelo _teorema da completude combinatória_. **Este teorema afirma que qualquer expressão lambda pode ser transformada em uma expressão equivalente utilizando os combinadores $S$ e $K$**.

 **Exemplo 1**: Definindo uma função constante com o combinador $K$

 O combinador $K$ pode ser usado para criar uma função constante. A função criada sempre retorna o primeiro argumento, independentemente do segundo.

 Definimos a função constante:

$$f = K\;A = \lambda x.\lambda y. ; x A = \lambda y.\;A$$

 Quando aplicamos $f$ a qualquer valor, o resultado é sempre $A\,$, pois o segundo argumento é ignorado.

 **Exemplo 2**: Definindo a aplicação de uma função com o combinador $S$

 O combinador $S$ permite aplicar uma função a dois argumentos e combiná-los. Ele pode ser usado para definir uma função que aplica duas funções diferentes ao mesmo argumento e, em seguida, combina os resultados.

 Definimos a função composta:

$$f = S\;g\;h = \lambda x.\;(g x)(h x)$$

 Aqui, $g$ e $h$ são duas funções que recebem o mesmo argumento $x\,$. O resultado é a combinação das duas funções aplicadas ao mesmo argumento.

$$f A = (\lambda x.(g x)(h x)) A \rightarrow_\beta (g A)(h A)$$

A remoção de variáveis nomeadas simplifica a computação. Este é um dos pontos centrais da teoria dos combinadores.

Em linguagens funcionais como Haskell, essa característica é usada para criar expressões modulares e compostas. Isso traz clareza e concisão ao código.

### 3.6.1. Exercícios sobre Combinadores e Funções Anônimas

**1**: Defina o combinador de ponto fixo de Curry, conhecido como o combinador $ $Y$ \,$, e aplique-o à função $ f(x) \, = x + 1 \,$. Explique o que ocorre.

   **Solução:** O combinador $ $Y$ $ é definido como:

   $$Y = \lambda f. (\lambda x.\;f(x\;x)) (\lambda x.\;f(x\;x))$$

   Aplicando-o à função $ f(x) \, = x + 1 $:

   $$Y (\lambda x.\;x + 1) \to (\lambda x.\;(\lambda x.\;x + 1)(x\;x)) (\lambda x.\;(\lambda x.\;x + 1)(x\;x))$$

   Este processo gera uma recursão infinita, pois a função continua chamando a si mesma.

**2**: Aplique o combinador $Y$ à função $f(x) \, = x \times 2$ e calcule as duas primeiras iterações do ponto fixo.

   **Solução:** Aplicando o combinador $Y$ a $f(x) \, = x \times 2$:

   $$Y (\lambda x.\;x \times 2)$$

   As duas primeiras iterações seriam:

   $$x_1 = 2$$

   $$x_2 = 2 \times 2 = 4$$

**3**: Mostre como o combinador $ $Y$ $ pode ser aplicado para encontrar o ponto fixo da função $ f(x) \, = x^2 - 1 \,$.

   **Solução:** Aplicando o combinador $Y$ à função $f(x) \, = x^2 - 1$:

   $$Y (\lambda x.\;x^2 - 1)$$

   A função continuará sendo aplicada indefinidamente, mas o ponto fixo é a solução de $x = x^2 - 1\,$, que leva ao ponto fixo $x = P = \frac{1 + \sqrt{5}}{2}$(a razão áurea).

**4**: Use o combinador de ponto fixo para definir uma função recursiva que calcula o fatorial de um número.

   **Solução:** A função fatorial pode ser definida como:

   $$f = \lambda f. \lambda n.\;(n = 0 ? 1 : n \times f\;(n-1))$$

   Aplicando o combinador $ $Y$ $:

   $$Y(f) \, = \lambda n.\;(n = 0 ? 1 : n \times Y\;(f)\;(n-1))$$

   Agora podemos calcular o fatorial de um número, como $ 3! = 3 \times 2 \times 1 = 6 \,$.

**5**: Utilize o combinador $ $Y$ $ para definir uma função recursiva que calcula a sequência de Fibonacci.

   **Solução:** A função para Fibonacci pode ser definida como:

   $$f = \lambda f. \lambda n.\;(n = 0 ? 0 : (n = 1 ? 1 : f\;(n-1) + f\;(n-2)))$$

   Aplicando o combinador $ $Y$ $:

   $$Y\;(f) \, = \lambda n.\;(n = 0 ? 0 : (n = 1 ? 1 : Y\;(f)\;(n-1) + Y\;(f)\;(n-2)))$$

   Agora podemos calcular Fibonacci, como $F_5 = 5\,$.

**6**: Explique por que o combinador $Y$ é capaz de gerar funções recursivas, mesmo em linguagens sem suporte nativo para recursão.

   **Solução:** O combinador $Y$ cria recursão ao aplicar uma função a si mesma. Ele transforma uma função aparentemente sem recursão em uma recursiva ao introduzir auto-aplicação. Essa técnica é útil em linguagens onde a recursão não é uma característica nativa, pois o ponto fixo permite que a função se chame indefinidamente.

**7**: Mostre como o combinador $Y$ pode ser aplicado à função exponencial $f(x) \, = 2^x$ e calcule a primeira iteração.

   **Solução:** Aplicando o combinador $Y$ à função exponencial $f(x) \, = 2^x$:

   $$Y (\lambda x.\;2^x)$$

   A primeira iteração seria:

   $$x_1 = 2^1 = 2$$

**8**: Aplique o combinador de ponto fixo para encontrar o ponto fixo da função $f(x) \, = \frac{1}{x} + 1\,$.

   **Solução:** Para aplicar o combinador $Y$ a $f(x) \, = \frac{1}{x} + 1\,$, encontramos o ponto fixo ao resolver $x = \frac{1}{x} + 1\,$. O ponto fixo é a solução da equação quadrática, que resulta em $x = P\,$, a razão áurea.

**9**: Utilize o combinador $Y$ para definir uma função recursiva que soma os números de $1$ até $n\,$.

   **Solução:** A função de soma até $n$ pode ser definida como:

   $$f = \lambda f. \lambda n.\;(n = 0 ? 0 : n + f\;(n-1))$$

   Aplicando o combinador $Y$:

   $$Y(f) \, = \lambda n.\;(n = 0 ? 0 : n + Y\;(f)\;(n-1))$$

   Agora podemos calcular a soma, como $\sum_{i=1}^{3} = 3 + 2 + 1 = 6\,$.

**10**: Aplique o combinador $Y$ para definir uma função recursiva que calcula o máximo divisor comum (MDC) de dois números.

   **Solução:** A função MDC pode ser definida como:

   $$f = \lambda f. \lambda a. \lambda b.\;(b = 0 ? a : f\;(b, a \% b))$$

   Aplicando o combinador $Y$:

   $$Y(f) \, = \lambda a. \lambda b.\;(b = 0 ? a : Y\;(f)\;(b, a \% b))$$

   Agora podemos calcular o MDC, como $\text{MDC}(15, 5) \, = 5 \,$.

**11**: Aplique o combinador identidade $ I = \lambda x.\;x $ ao valor $ 10 \,$.

   **Solução:** Aplicamos o combinador identidade:

   $$I\;10 = (\lambda x.\;x)\;10 \rightarrow_\beta 10$$

**12**: Aplique o combinador $K = \lambda x. \lambda y.\;x$ aos valores $3$ e $7\,$. O que ocorre?

   **Solução:** Aplicamos $K$ ao valor $3$ e depois ao valor $7$:

   $$K\;3\;7 = (\lambda x. \lambda y.\;x)\;3\;7 \rightarrow*\beta (\lambda y.\;3)\;7 \rightarrow*\beta 3$$

**13**: Defina a expressão $ S(\lambda z.\;z^2)(\lambda z.\;z + 1)\;4 $ reduzindo-a passo a passo.

   **Solução:** Aplicamos o combinador $ S = \lambda f. \lambda g. \lambda x.\;f(x)\;(g\;(x))$ às funções $f = \lambda z.\;z^2 $ e $ g = \lambda z.\;z + 1\,$, e ao valor $4$:

   $$S(\lambda z.\;z^2)(\lambda z.\;z + 1)\;4$$

   Primeiro, aplicamos as funções:

   $$(\lambda f. \lambda g. \lambda x.\;f\;(x)\;(g\;(x)))(\lambda z.\;z^2)(\lambda z.\;z + 1)\;4$$

   Agora, substituímos e aplicamos as funções a $4$:

   $$(\lambda z.\;z^2)\;4 ((\lambda z.\;z + 1)\;4) \rightarrow_\beta 4^2\;(4 + 1) \, = 16 \times 5 = 80$$

**14**: Aplique o combinador identidade $I$ a uma função anônima $\lambda y.\;y + 2$ e explique o resultado.

   **Solução:** Aplicamos o combinador identidade $I$ à função anônima:

   $$I (\lambda y.\;y + 2) \, = (\lambda x.\;x) (\lambda y.\;y + 2) \rightarrow_\beta \lambda y.\;y + 2$$

   O combinador identidade retorna a própria função, sem modificações.

**15**: Reduza a expressão $K\;(I\;7)\;9$:

   **Solução:** Aplicamos $I$ a $7\,$, que resulta em $7\,$, e depois aplicamos $K$:

   $$K\;(I\;7)\;9 = K\;7\;9 = (\lambda x. \lambda y.\;x)\;7\;9 \rightarrow*\beta (\lambda y.\;7)\;9 \rightarrow*\beta 7$$

**16**: Aplique o combinador $K$ à função $\lambda z.\;z \times z $ e o valor $5\,$. O que ocorre?

 **Solução:** Aplicamos o combinador $K$ à função e ao valor:

   $$K\;(\lambda z.\;z \times z)\;5 = (\lambda x. \lambda y.\;x)\;(\lambda z.\;z \times z)\;5 \rightarrow*\beta (\lambda y.\;\lambda z.\;z \times z)\;5 \rightarrow*\beta \lambda z.\;z \times z$$

 O combinador $K$ descarta o segundo argumento, retornando a função original $\lambda z.\;z \times z\,$.

**17**: Construa uma função anônima que soma dois números sem usar nomes de variáveis explícitas, usando somente combinadores $ S $ e $ K \,$.

   **Solução:** Usamos o combinador $S$ para aplicar duas funções ao mesmo argumento:

   $$S\;(K\;(3))\;(K\;(4)) \, = (\lambda f. \lambda g. \lambda x.\;f\;(x)\;(g\;(x)))\;(K\;(3))(K\;(4))$$

   Aplicamos $f$ e $g$:

   $$\rightarrow*\beta (\lambda x.\;K(3)(x)(K(4)(x))) \rightarrow*\beta (\lambda x.\;3 + 4) \, = 7$$

**18**: Reduza a expressão $S\;K\;K$ e explique o que o combinador $S\;(K)\;(K)$ representa.

   **Solução:** Aplicamos o combinador $ S $:

   $$S\;K\;K = (\lambda f. \lambda g. \lambda x.\;f\;(x)\;(g\;(x)))\;K\;K$$

   Substituímos $ f $ e $ g $ por $ K $:

   $$= (\lambda x.\;K(x)(K(x)))$$

   Aplicamos $K$:

   $$= \lambda x.\;(\lambda y.\;x)( (\lambda y.\;x)) \rightarrow_\beta \lambda x.\;x$$

   Portanto, $ S(K)(K)$ é equivalente ao combinador identidade $ I \,$.

**19**: Explique por que o combinador $ K $ pode ser usado para representar constantes em expressões lambda.

   **Solução:** O combinador $K = \lambda x. \lambda y.\;x$ descarta o segundo argumento e retorna o primeiro. Isso significa que qualquer valor aplicado ao combinador $K$ é mantido como constante, independentemente de quaisquer outros argumentos fornecidos. Por isso, o combinador $K$ pode ser usado para representar constantes, uma vez que sempre retorna o valor do primeiro argumento, ignorando os subsequentes.

**20**: Reduza a expressão $S\;(K\;S)\;K$ e explique o que esta combinação de combinadores representa.

   **Solução:** Aplicamos o combinador $S$:

   $$S(KS)K = (\lambda f. \lambda g. \lambda x.\;f\;(x)\;(g\;(x)))\;K\;S\;K$$

   Substituímos $f = KS$ e $g = K$:

   $$=\lambda x.\;K\;S\;(x)\;(K\;(x))$$

   Aplicamos $K\;S$ e $K$:

   $$K\;S\;(x) \, = (\lambda x. \lambda y.\;x)\;S\;(x) \, = S$$

   $$K(x) \, = \lambda y.\;x$$

   Portanto:

   $$S\;(K\;S)\;K = S$$

   Essa combinação de combinadores representa a função de substituição $S\,$.

## 3.7. Estratégias de Avaliação no Cálculo Lambda

**As estratégias de avaliação determinam como expressões são computadas**. Essas estratégias de avaliação terão impacto na implementação de linguagens de programação. Diferentes abordagens para a avaliação de argumentos e funções podem resultar em diferentes características de desempenho.

### 3.7.1. Avaliação por Valor vs Avaliação por Nome

No contexto do cálculo lambda e linguagens de programação, existem duas principais abordagens para avaliar expressões:

1. **Avaliação por Valor**: Nesta estratégia, os argumentos são avaliados antes de serem passados para uma função. O cálculo é feito de forma estrita, ou seja, os argumentos são avaliados imediatamente. Isso corresponde à **ordem aplicativa de redução**, onde a função é aplicada após a avaliação completa de seus argumentos. A vantagem desta estratégia é que ela pode ser mais eficiente em alguns contextos, pois o argumento é avaliado somente uma vez.

   **Exemplo**: Considere a expressão $(\lambda x.\;x + 1) (2 + 3)\,$.

   Na **avaliação por valor**, primeiro o argumento $2 + 3$ é avaliado para $5\,$, e em seguida a função é aplicada:

   $$(\lambda x.\;x + 1)\;5 \rightarrow 5 + 1 \rightarrow 6$$

2. Avaliação por Nome: Argumentos são passados para a função sem serem avaliados imediatamente. A avaliação ocorre quando o argumento é necessário. Esta estratégia corresponde à **ordem normal de redução**, em que a função é aplicada diretamente e o argumento só é avaliado quando estritamente necessário. Uma vantagem desta abordagem é que ela pode evitar avaliações desnecessárias, especialmente em contextos onde certos argumentos nunca são utilizados.

   **Exemplo**:
   Usando a mesma expressão $\lambda x.\;x + 1) (2 + 3)\,$, com **avaliação por nome**, a função seria aplicada sem avaliar o argumento de imediato:

   $$(\lambda x.\;x + 1) (2 + 3) \rightarrow (2 + 3) + 1 \rightarrow 5 + 1 \rightarrow 6$$

### 3.7.2. Exercícios sobre Estratégias de Avaliação

**1**: Considere a expressão $(\lambda x.\;x + 1) (2 + 3)\,$. Avalie-a usando a estratégia de**avaliação por valor**.

   **Solução:** Na avaliação por valor, o argumento é avaliado antes de ser aplicado à função:

   $$(2 + 3) \rightarrow 5$$

   Agora, aplicamos a função:

   $$(\lambda x.\;x + 1)\;5 \rightarrow 5 + 1 \rightarrow 6$$

**2**: Use a **avaliação por nome**na expressão $(\lambda x.\;x + 1) (2 + 3)$ e explique o processo.

   **Solução:** Na avaliação por nome, o argumento é passado diretamente para a função:

   $$(\lambda x.\;x + 1) (2 + 3) \rightarrow (2 + 3) + 1 \rightarrow 5 + 1 \rightarrow 6$$

**3**: A expressão $(\lambda x.\;x \times x) ((2 + 3) + 1)$ é dada. Avalie-a usando a **avaliação por valor**.

   **Solução:** Primeiro, avaliamos o argumento:

   $$(2 + 3) + 1 \rightarrow 5 + 1 \to 6$$

   Agora, aplicamos a função:

   $$(\lambda x.\;x \times x)\;6 \rightarrow 6 \times 6 \to 36$$

**4**: Aplique a **avaliação por nome** na expressão $(\lambda x.\;x \times x) ((2 + 3) + 1)$ e explique cada passo.

   **Solução:** Usando avaliação por nome, o argumento não é avaliado imediatamente:

   $$(\lambda x.\;x \times x) ((2 + 3) + 1) \rightarrow ((2 + 3) + 1) \times ((2 + 3) + 1)$$

   Agora, avaliamos o argumento quando necessário:

   $$(5 + 1) \times (5 + 1) \to 6 \times 6 \to 36$$

**5**: Considere a expressão $(\lambda x.\;x + 1) ( (\lambda y.\;y + 2)\;3)\,$. Avalie-a usando a **ordem aplicativa de redução** (avaliação por valor).

   **Solução:** Primeiro, avaliamos o argumento $(\lambda y.\;y + 2)\;3 $:

   $$(\lambda y.\;y + 2)\;3 \rightarrow 3 + 2 \to 5$$

   Agora, aplicamos $ 5 $ à função:

   $$(\lambda x.\;x + 1)\;5 \rightarrow 5 + 1 \to 6$$

**6**: Aplique a **ordem normal de redução** (avaliação por nome) na expressão $(\lambda x.\;x + 1) ( (\lambda y.\;y + 2)\;3)\,$.

   **Solução:** Usando a ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

   $$(\lambda x.\;x + 1) ( (\lambda y.\;y + 2)\;3) \rightarrow ( (\lambda y.\;y + 2)\;3) + 1$$

   Agora, avaliamos o argumento:

   $$(3 + 2) + 1 \to 5 + 1 \to 6$$

**7**: Considere a expressão $(\lambda x.\;x + 1) (\lambda y.\;y + 2)\,$. Avalie-a usando **avaliação por valor** e explique por que ocorre um erro ou indefinição.

   **Solução:** Na avaliação por valor, tentaríamos primeiro avaliar o argumento $\lambda y.\;y + 2 \,$. No entanto, esse é um termo que não pode ser avaliado diretamente, pois é uma função. Logo, a expressão não pode ser reduzida, resultando em um erro ou indefinição, já que a função não pode ser aplicada diretamente sem um argumento concreto.

**8**: Aplique a **avaliação por nome** na expressão $(\lambda x.\;x + 1) (\lambda y.\;y + 2)\,$.

   **Solução:** Na avaliação por nome, passamos o argumento sem avaliá-lo:

   $$(\lambda x.\;x + 1) (\lambda y.\;y + 2) \rightarrow (\lambda y.\;y + 2) + 1$$

   Como a função $\lambda y.\;y + 2 $ não pode ser somada diretamente a um número, a expressão resultante é indefinida ou produzirá um erro.

**9**: Dada a expressão $(\lambda x. \lambda y.\;x + y) (2 + 3)\;4 \,$, aplique a **ordem aplicativa de redução**.

   **Solução:** Primeiro, avaliamos o argumento $ 2 + 3 $:

   $$2 + 3 \to 5$$

   Agora, aplicamos a função $(\lambda x. \lambda y.\;x + y)$:

   $$(\lambda x. \lambda y.\;x + y)\;5 4 \rightarrow (\lambda y.\;5 + y)\;4 \rightarrow 5 + 4 \to 9$$

**10**: Use a **ordem normal de redução** para avaliar a expressão $(\lambda x. \lambda y.\;x + y) (2 + 3)\;4 \,$.

   **Solução:** Na ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

   $$(\lambda x. \lambda y.\;x + y) (2 + 3)\;4 \rightarrow (\lambda y.\;(2 + 3) + y)\;4$$

   Agora, avaliamos os argumentos:

   $$(5) + 4 \to 9$$

# 4. Estratégias de Redução

No cálculo lambda, a ordem em que as expressões são avaliadas define o processo de redução dos termos. As duas estratégias mais comuns para essa avaliação são a _estratégia normal e a _estratégia aplicativa_.

Na _estratégia normal_, as expressões mais externas são reduzidas antes das internas. Já na _estratégia aplicativa_, os argumentos de uma função são reduzidos primeiro, antes de aplicar a função.

Essas estratégias influenciam o resultado e o comportamento do processo de computação, especialmente em expressões que podem divergir ou não possuir valor definido. Vamos ver estas estratégias com atenção.

## 4.1. Ordem Normal (Normal-Order)

Na **ordem normal**, a redução prioriza o _redex_ mais externo à esquerda (redução externa). Essa estratégia é garantida para encontrar a forma normal de um termo, caso ela exista. Como o argumento não é avaliado de imediato, é possível evitar o cálculo de argumentos que nunca serão utilizados, tornando-a equivalente à _avaliação preguiçosa_ em linguagens de programação.

Uma vantagem da ordem normal é que sempre que encontramos a forma normal de um termo, se existir, podemos evitar a avaliação de argumentos desnecessários. Melhorando a eficiência do processo em termos de espaço.

Por outro lado, a ordem normal pode ser ineficiente em termos de tempo, já que, acabamos por reavaliar expressões várias vezes quando elas são necessárias repetidamente.

 **Exemplo 1**: considere a expressão:

$$(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w))$$

 Na ordem normal, a redução ocorre:

$$(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w)) \to_\beta \lambda y.\;y$$

 O argumento $((\lambda z.\;z\;z) (\lambda w. w w))$ não é avaliado, pois ele nunca é utilizado no corpo da função.

**Exemplo 2**: Considere a expressão:

$$M = (\lambda f.\;(\lambda x.\;f\;(x\;x))\;(\lambda x.\;f\;(x\;x)))\;(\lambda y.\;y + 1)$$

Vamos reduzir $M$ usando a ordem normal.

**Passo 1**: Identificamos o redex mais externo à esquerda:

$$\underline{(\lambda f.\;(\lambda x.\;f\;(x\;x))\;(\lambda x.\;f\;(x\;x)))\;(\lambda y.\;y + 1)}$$

Aplicamos a redução beta ao redex, substituindo $f$ por $(\lambda y.\;y + 1)$:

$$\to_\beta (\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))\;(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))$$

**Passo 2**: Novamente, identificamos o redex mais externo à esquerda:

$$\underline{(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))\;(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))}$$

Aplicamos a redução beta, substituindo $x$ por $(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))$:

$$\to_\beta (\lambda y.\;y + 1)\;\left( \underline{(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))\;(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))} \right)$$

**Passo 3**: Dentro do argumento, identificamos o redex mais externo:

$$\underline{(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))\;(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))}$$

Aplicamos a redução beta novamente, substituindo $x$ por $(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))$:

$$\to_\beta (\lambda y.\;y + 1)\;\left( (\lambda y.\;y + 1)\;\left( \underline{(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))\;(\lambda x.\;(\lambda y.\;y + 1)\;(x\;x))} \right) \right)$$

**Passo 4**: Observamos que o processo está se repetindo. A cada aplicação da redução beta, o termo dentro dos parênteses permanece o mesmo, indicando um **loop infinito**:

$$\to_\beta (\lambda y.\;y + 1)\;\left( (\lambda y.\;y + 1)\;\left( (\lambda y.\;y + 1)\;\left( \cdots \right) \right) \right)$$

Na estratégia de ordem normal, a redução do termo $M$ não termina, pois entra em um ciclo infinito de reduções. Não é possível alcançar uma forma normal para $M$ usando esta estratégia, já que continuaremos expandindo o termo indefinidamente sem simplificá-lo a um resultado . Este exemplo ilustra como a ordem normal pode levar a reduções infinitas em certos casos, especialmente quando lidamos com termos autoreferenciados ou combinadores que causam expansão infinita.

Observamos que a expressão começa a repetir a si mesma, indicando um ciclo infinito. Contudo, na ordem normal, como o argumento não é necessário para o resultado , a redução pode ser concluída sem avaliá-lo.

**Exemplo 2**: Considere a expressão:

   $$M = (\lambda x.\;(\lambda y.\;x))\;\left( (\lambda z.\;z + z)\;3 \right)\;\left( (\lambda w.\;w\;w)\;(\lambda w.\;w\;w) \right)$$

   Vamos reduzir $M$ usando a ordem normal.

   1. Identificamos o redex mais externo à esquerda:

      $$\underline{(\lambda x.\;(\lambda y.\;x))\;\left( (\lambda z.\;z + z)\;3 \right)}\;\left( (\lambda w.\;w\;w)\;(\lambda w.\;w\;w) \right)$$

      Aplicamos a redução beta ao redex, substituindo $x$ por $\left( (\lambda z.\;z + z)\;3 \right)$:

      $$\to_\beta\;(\lambda y.\;\left( (\lambda z.\;z + z)\;3 \right))\;\left( (\lambda w.\;w\;w)\;(\lambda w.\;w\;w) \right)$$

   2. Observamos que a função resultante não utiliza o argumento $y$ no corpo da função. Portanto, o segundo argumento $\left( (\lambda w.\;w\;w)\;(\lambda w.\;w\;w) \right)$ não é avaliado na ordem normal, pois não é necessário.

   3. Calculamos a expressão $(\lambda z.\;z + z)\;3$ no corpo da função:

      $$(\lambda z.\;z + z)\;3\;\to_\beta\;3 + 3 = 6$$

   4. Substituímos o resultado no corpo da função:

      $$\lambda y.\;6$$

   Este é o resultado  da redução na ordem normal.

### 4.1.1. Exercícios de Ordem Normal

**1**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;x)\;((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))$$

   **Solução**: na **ordem normal**, reduzimos o redex mais externo à esquerda primeiro. Começamos por identificar o redex mais externo:

   $$(\lambda x.\;x)\;\underline{((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))}$$

   Neste ponto, o termo resultante será:

   $$M' = (\lambda y.\;y\;y)\;(\lambda y.\;y\;y)$$

   Novamente, identificamos o redex mais externo à esquerda:

   $$\underline{(\lambda y.\;y\;y)\;(\lambda y.\;y\;y)}$$

   Aplicamos a redução beta:

   $$\to_\beta (\lambda y.\;y\;y)\;(\lambda y.\;y\;y)$$

   Observamos que o termo repete-se indefinidamente, indicando uma redução infinita. Ou seja, aplicando a estratégia de ordem normal, continuamos a reduzir o redex mais externo à esquerda. Neste caso, a redução não termina, indicando que o termo não possui forma normal.

**2**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;\lambda y.\;x)\;((\lambda z.\;z + 1)\;5)\;((\lambda w.\;w \times 2)\;3)$$

   **Solução**: começamos identificando o redex mais externo:

   $$\underline{(\lambda x.\;\lambda y.\;x)\;((\lambda z.\;z + 1)\;5)}\;((\lambda w.\;w \times 2)\;3)$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \lambda y.\;\underline{((\lambda z.\;z + 1)\;5)}$$

   O termo resultante será:

   $$M' = \lambda y.\;((\lambda z.\;z + 1)\;5)$$

   Identificamos o próximo redex mais externo à esquerda no corpo:

   $$\lambda y.\;\underline{((\lambda z.\;z + 1)\;5)}$$

   Aplicamos a redução beta:

   $$\to_\beta \lambda y.\;5 + 1$$

   Realizamos a operação aritmética e encontramos a forma normal:

   $$\lambda y.\;6$$

   O argumento $((\lambda w.\;w \times 2)\;3)$ não é avaliado, pois não é utilizado no corpo da função resultante.

**3**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda f.\;f\;5)\;(\lambda x.\;x \times x)$$

   **Solução**: identificamos o redex mais externo:

   $$\underline{(\lambda f.\;f\;5)\;(\lambda x.\;x \times x)}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{(\lambda x.\;x \times x)\;5}$$

   Identificamos o próximo redex:

   $$\underline{(\lambda x.\;x \times x)\;5}$$

   Aplicamos a redução beta:

   $$\to_\beta 5 \times 5$$

   Realizamos a operação aritmética:

   $$25$$

   O termo reduzido na ordem normal resulta em $25\,$, que é a forma normal.

**4**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;x\;4)\;(\lambda y.\;y\;\times\;y)$$

   **Solução**: na **ordem normal**, reduzimos o redex mais externo à esquerda primeiro. Começamos identificando o redex mais externo:

   $$\underline{(\lambda x.\;x\;4)\;(\lambda y.\;y\;\times\;y)}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{(\lambda y.\;y\;\times\;y)\;4}$$

   Identificamos o próximo redex mais externo à esquerda:

   $$\underline{(\lambda y.\;y\;\times\;y)\;4}$$

   Aplicamos a redução beta:

   $$\to_\beta 4\;\times\;4$$

   Realizamos a operação aritmética:

   $$16$$

   O termo reduzido na ordem normal resulta em $16\,$, nossa forma normal.

**5**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;\lambda y.\;y\;x)\;((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))\;5$$

   **Solução**: identificamos o redex mais externo:

   $$\underline{(\lambda x.\;\lambda y.\;y\;x)\;((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))}\;5$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \lambda y.\;y\;\underline{((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))}$$

   Observamos que o argumento $x = ((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))$ é um termo que se reduz infinitamente. No entanto, na ordem normal, só o avaliamos se necessário.

   Aplicamos a função resultante ao argumento $5$:

   $$(\lambda y.\;y\;((\lambda z.\;z\;z)\;(\lambda z.\;z\;z)))\;5 \to_\beta \underline{5\;((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))}$$

   Aqui, precisamos avaliar $((\lambda z.\;z\;z)\;(\lambda z.\;z\;z))$ para continuar. No entanto, esse termo não possui forma normal e leva a uma redução infinita. Portanto, a redução não termina, indicando que o termo não possui forma normal na ordem normal.

**6**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;\lambda y.\;x\;y)\;(\lambda w.\;w + 2)\;3$$

   **Solução**: identificamos o redex mais externo:

   $$\underline{(\lambda x.\;\lambda y.\;x\;y)\;(\lambda w.\;w + 2)}\;3$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \lambda y.\;(\lambda w.\;w + 2)\;y$$

   Aplicamos a função resultante ao argumento $3$:

   $$(\lambda y.\;(\lambda w.\;w + 2)\;y)\;3 \to_\beta \underline{(\lambda w.\;w + 2)\;3}$$

   Aplicamos a redução beta ao redex:

   $$\to_\beta 3 + 2$$

   Realizamos a operação aritmética:

   $$5$$

   O termo reduzido na ordem normal resulta em $5\,$, que é a forma normal.

**7**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;x)\;((\lambda y.\;y + y)\;4)$$

   **Solução**: na **ordem normal**, reduzimos o redex mais externo à esquerda primeiro. Começamos identificando o redex mais externo:

   $$(\lambda x.\;x)\;\underline{((\lambda y.\;y + y)\;4)}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{((\lambda y.\;y + y)\;4)}$$

   Agora, identificamos o próximo redex:

   $$\underline{(\lambda y.\;y + y)\;4}$$

   Aplicamos a redução beta:

   $$\to_\beta 4 + 4 = 8$$

   O termo reduzido na ordem normal resulta em $8\,$, que é a forma normal.

**8**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;5)\;((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))$$

   **Solução**: na **ordem normal**, reduzimos o redex mais externo à esquerda primeiro.

   Identificamos o redex mais externo:

   $$\underline{(\lambda x.\;5)\;((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta 5$$

   O termo reduzido na ordem normal resulta em $5\,$, que é a forma normal.

   O argumento $((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))$ não é avaliado, evitando uma redução infinita.

**9**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;x\;2)\;(\lambda y.\;3)$$

   **Solução**: identificamos o redex mais externo:

   $$\underline{(\lambda x.\;x\;2)\;(\lambda y.\;3)}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{(\lambda y.\;3)\;2}$$

   Aplicamos a redução beta ao redex interno:

   $$\to_\beta 3$$

   O termo reduzido na ordem normal resulta em $3\,$, que é a forma normal.

**10**: Reduza o seguinte termo usando a estratégia de ordem normal:

$$M = (\lambda x.\;x\;3)\;(\lambda y.\;y + 2)$$

   **Solução**: identificamos o redex mais externo:

   $$\underline{(\lambda x.\;x\;3)\;(\lambda y.\;y + 2)}$$

   Aplicamos a redução beta ao redex externo:

   $$\to_\beta \underline{(\lambda y.\;y + 2)\;3}$$

   Aplicamos a redução beta ao redex interno:

   $$\to_\beta 3 + 2 = 5$$

   O termo reduzido na ordem normal resulta em $5\,$, nossa forma normal.

## 4.2. Ordem Aplicativa (Applicative-Order)

Na **ordem aplicativa**, a estratégia de redução no cálculo lambda consiste em avaliar primeiro os argumentos de uma função antes de aplicar a função em si. Isso significa que a redução ocorre das partes mais internas para as mais externas (redução interna). Essa abordagem corresponde à **avaliação estrita**, onde os argumentos são completamente avaliados antes da aplicação da função.

A ordem aplicativa é utilizada em muitas linguagens de programação, especialmente nas imperativas e em algumas funcionais, como ML e Scheme. Uma vantagem dessa estratégia é que, quando o resultado de um argumento é utilizado várias vezes no corpo da função, a avaliação prévia evita reavaliações redundantes, podendo ser mais eficiente em termos de tempo. No entanto, a ordem aplicativa pode levar a problemas de não-terminação em casos onde a ordem normal encontraria uma solução. E pode resultar em desperdício de recursos ao avaliar argumentos que não são necessários para o resultado  da função.

**Exemplo 1**: Considere a expressão:

$$M = (\lambda x.\;x)\;((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))$$

Na **ordem aplicativa**, avaliamos primeiro o argumento: Avaliamos o argumento $N = ((\lambda y.\;y\;y)\;(\lambda y.\;y\;y))$:

Aplicamos a redução beta:

$$(\lambda y.\;y\;y)\;(\lambda y.\;y\;y) \to_\beta (\lambda y.\;y\;y)\;(\lambda y.\;y\;y)$$

Observamos que o termo se repete indefinidamente, resultando em uma **redução infinita**. Como o argumento não pode ser completamente avaliado, a aplicação da função não ocorre, e a redução não termina.

Na **ordem normal**, a função $(\lambda x.\;x)$ não utiliza o argumento além de retornar o próprio argumento. No entanto, na ordem aplicativa, a avaliação do argumento impede a conclusão da redução.

**Exemplo 2**: considere a expressão:

$$M = (\lambda x.\;\lambda y.\;x)\;\left( (\lambda z.\;z + 1)\;5 \right)\;\left( (\lambda w.\;w \times 2)\;3 \right)$$

Na ordem aplicativa, procedemos da seguinte forma: avaliamos o primeiro argumento:

 Calculamos $A = (\lambda z.\;z + 1)\;5$:

$$(\lambda z.\;z + 1)\;5 \to_\beta 5 + 1 = 6$$

Avaliamos o segundo argumento:

Calculamos $B = (\lambda w.\;w \times 2)\;3$:

$$(\lambda w.\;w \times 2)\;3 \to_\beta 3 \times 2 = 6$$

Aplicamos a função ao primeiro argumento avaliado:

$$(\lambda x.\;\lambda y.\;x)\;6 \to_\beta \lambda y.\;6$$

Aplicamos a função resultante ao segundo argumento avaliado:

$$(\lambda y.\;6)\;6 \to_\beta 6$$

O resultado é $6\,$. Note que ambos os argumentos foram avaliados, embora o segundo argumento não seja utilizado no resultado . Isso exemplifica como a ordem aplicativa pode desperdiçar recursos ao avaliar argumentos desnecessários.

**Exemplo 3**: considere a expressão:

$$M = (\lambda x.\;42)\;\left( (\lambda y.\;y\;y)\;(\lambda y.\;y\;y) \right)$$

Na ordem aplicativa, avaliamos primeiro o argumento:

Avaliamos o argumento $N = (\lambda y.\;y\;y)\;(\lambda y.\;y\;y)$:

Aplicamos a redução beta:

$$(\lambda y.\;y\;y)\;(\lambda y.\;y\;y) \to_\beta (\lambda y.\;y\;y)\;(\lambda y.\;y\;y) \to_\beta \cdots$$

O termo entra em uma **redução infinita**.

Como o argumento não pode ser completamente avaliado, a aplicação da função não ocorre, e a redução não termina. Na **ordem normal**, a função $(\lambda x.\;42)$ não utiliza o argumento $x\,$, portanto, o resultado seria imediatamente $42\,$, sem necessidade de avaliar o argumento que causa a não-terminação.

**Exemplo 4**: considere a expressão:

$$M = (\lambda f.\;f\;(f\;2))\;(\lambda x.\;x \times x)$$

Na ordem aplicativa, procedemos assim:

Avaliamos o argumento $N = (\lambda x.\;x \times x)\,$, que é uma função e não requer avaliação adicional.

Aplicamos a função externa ao argumento:

$$(\lambda f.\;f\;(f\;2))\;(\lambda x.\;x \times x) \to_\beta (\lambda x.\;x \times x)\;((\lambda x.\;x \times x)\;2)$$

Avaliamos o argumento interno $(\lambda x.\;x \times x)\;2$:

Aplicamos a redução beta:

$$(\lambda x.\;x \times x)\;2 \to_\beta 2 \times 2 = 4$$

Aplicamos a função externa ao resultado:

$$(\lambda x.\;x \times x)\;4 \to_\beta 4 \times 4 = 16$$

O resultado é $16\,$. Neste caso, a ordem aplicativa é eficiente, pois avalia os argumentos necessários e evita reavaliações.

A escolha entre ordem aplicativa e ordem normal depende do contexto e das necessidades específicas da computação. Em situações onde todos os argumentos são necessários e podem ser avaliados sem risco de não-terminação, a ordem aplicativa pode ser preferível. No entanto, quando há possibilidade de argumentos não terminarem ou não serem necessários, a ordem normal oferece uma estratégia mais segura.

### 4.2.1. Exercícios sobre Ordem Normal e Aplicativa

**1**: Aplique a ordem normal à expressão $(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w))\,$.

   **Solução:**
   A ordem normal prioriza a redução externa:

   $$(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w)) \rightarrow_\beta \lambda y.\;y$$

   O argumento $((\lambda z.\;z\;z) (\lambda w. w w))$ nunca é avaliado.

**2**: Reduza a expressão $(\lambda x. \lambda y.\;x) ((\lambda z.\;z + 1)\;5)$ usando a ordem normal.

   **Solução:**
   Na ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

   $$(\lambda x. \lambda y.\;x) ((\lambda z.\;z + 1)\;5) \rightarrow_\beta \lambda y.\;((\lambda z.\;z + 1)\;5)$$

   O argumento não é avaliado porque a função não o utiliza.

**3**: Considere a expressão $(\lambda x. \lambda y.\;y + 1) ((\lambda z.\;z\;z) (\lambda w. w w))\,$. Avalie-a usando ordem normal.

   **Solução:**
   A ordem normal evita a avaliação do argumento:

   $$(\lambda x. \lambda y.\;y + 1) ((\lambda z.\;z\;z) (\lambda w. w w)) \rightarrow_\beta \lambda y.\;y + 1$$

   O termo $((\lambda z.\;z\;z) (\lambda w. w w))$ nunca é avaliado.

**4**: Aplique a ordem normal na expressão $(\lambda x.\;x) ((\lambda z.\;z\;z) (\lambda w. w w))\,$.

   **Solução:**
   Primeiro aplicamos a função sem avaliar o argumento:

   $$(\lambda x.\;x) ((\lambda z.\;z\;z) (\lambda w. w w)) \rightarrow_\beta ((\lambda z.\;z\;z) (\lambda w. w w))$$

   Agora a expressão é indefinida, pois avaliaremos uma expressão sem fim.

**5**: Reduza a expressão $(\lambda x.\;3) ((\lambda z.\;z + 1)\;5)$ utilizando a ordem normal.

   **Solução:**
   Na ordem normal, o argumento não é avaliado:

   $$(\lambda x.\;3) ((\lambda z.\;z + 1)\;5) \rightarrow_\beta 3$$

   O argumento $((\lambda z.\;z + 1)\;5)$ nunca é avaliado.

**6**: Avalie a expressão $(\lambda x. \lambda y.\;x) ((\lambda z.\;z + 1)\;5)$ usando ordem aplicativa.

   **Solução:**
   Na ordem aplicativa, o argumento é avaliado primeiro:

   $$(\lambda z.\;z + 1)\;5 \rightarrow_\beta 6$$

   Agora aplicamos a função:

   $$(\lambda x. \lambda y.\;x)\;6 \rightarrow_\beta \lambda y.\;6$$

**7**: Aplique a ordem aplicativa à expressão $(\lambda x.\;x) ((\lambda z.\;z\;z) (\lambda w. w w))\,$.

   **Solução:**
   Na ordem aplicativa, o argumento é avaliado primeiro, o que leva a um loop sem fim:

   $$((\lambda z.\;z\;z) (\lambda w. w w)) \rightarrow*\beta (\lambda w. w w) (\lambda w. w w) \rightarrow*\beta ...$$

   A expressão entra em uma recursão infinita.

**8**: Reduza a expressão $(\lambda x.\;x \times 2) ((\lambda z.\;z + 3)\;4)$ usando ordem aplicativa.

   **Solução:**
   Primeiro, o argumento $(\lambda z.\;z + 3)\;4 $ é avaliado:

   $$(\lambda z.\;z + 3)\;4 \rightarrow_\beta 4 + 3 \to 7$$

   Agora aplicamos a função:

   $$(\lambda x.\;x \times 2)\;7 \rightarrow_\beta 7 \times 2 \to 14$$

**9**: Considere a expressão $(\lambda x.\;x + 1) (\lambda y.\;y + 2)\,$. Avalie-a usando ordem aplicativa e explique o resultado.

   **Solução:**
   Na ordem aplicativa, tentamos avaliar o argumento primeiro:

   $$(\lambda y.\;y + 2) \rightarrow_\beta \lambda y.\;y + 2$$

   Como o argumento não pode ser avaliado (é uma função), o resultado não pode ser reduzido, levando a um erro ou indefinição.

**10**: Aplique a ordem aplicativa à expressão $(\lambda x.\;x + 1) ((\lambda z.\;z + 2)\;3)\,$.

   **Solução:**
   Primeiro avaliamos o argumento:

   $$(\lambda z.\;z + 2)\;3 \rightarrow_\beta 3 + 2 \to 5$$

   Agora aplicamos a função:

   $$(\lambda x.\;x + 1)\;5 \rightarrow_\beta 5 + 1 \to 6$$

**11**: Compare a avaliação da expressão $(\lambda x.\;2) ((\lambda z.\;z\;z) (\lambda w. w w))$ usando ordem normal e ordem aplicativa.

   Solução (Ordem Normal):
   A ordem normal evita a avaliação do argumento:

   $$(\lambda x.\;2) ((\lambda z.\;z\;z) (\lambda w. w w)) \rightarrow_\beta 2$$

   Solução (Ordem Aplicativa):
   Na ordem aplicativa, o argumento é avaliado, levando a um loop sem fim.

**12**: Considere a expressão $(\lambda x. \lambda y.\;x + y) ((\lambda z.\;z + 1)\;3)\;4 \,$. Avalie usando ordem normal e ordem aplicativa.

   Solução (Ordem Normal):
   Aplicamos a função sem avaliar o argumento:

   $$(\lambda x. \lambda y.\;x + y) ((\lambda z.\;z + 1)\;3)\;4 \rightarrow_\beta (\lambda y.\;((\lambda z.\;z + 1)\;3) + y)\;4$$

   Agora avaliamos o argumento:

   $$((3 + 1) + 4) \to 8$$

   Solução (Ordem Aplicativa):
   Na ordem aplicativa, avaliamos o argumento primeiro:

   $$(\lambda z.\;z + 1)\;3 \rightarrow_\beta 4$$

   Agora aplicamos a função:

   $$(\lambda x. \lambda y.\;x + y)\;4 4 \rightarrow_\beta 4 + 4 \to 8$$

**13**: Aplique ordem normal e ordem aplicativa à expressão:

   $$(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w))\;3 \,$$

   Solução (Ordem Normal):
   A função é aplicada sem avaliar o argumento:

   $$(\lambda x. \lambda y.\;y) ((\lambda z.\;z\;z) (\lambda w. w w))\;3 \rightarrow_\beta \lambda y.\;y$$

   Agora aplicamos a função:

   $$(\lambda y.\;y)\;3 \rightarrow_\beta 3$$

   Solução (Ordem Aplicativa):
   Na ordem aplicativa, o argumento é avaliado, resultando em um loop infinito.

**14**: Avalie a expressão $(\lambda x.\;x) ((\lambda z.\;z + 1)\;3)$ usando ordem normal e ordem aplicativa.

   Solução (Ordem Normal):
   A função é aplicada sem avaliar o argumento:

   $$(\lambda x.\;x) ((\lambda z.\;z + 1)\;3) \rightarrow*\beta ((\lambda z.\;z + 1)\;3) \rightarrow*\beta 4$$

   Solução (Ordem Aplicativa):
   Na ordem aplicativa, o argumento é avaliado primeiro:

   $$(\lambda z.\;z + 1)\;3 \rightarrow_\beta 4$$

   Agora aplicamos a função:

   $$(\lambda x.\;x)\;4 \rightarrow_\beta 4$$

**15**: Reduza a expressão $(\lambda x.\;x) (\lambda y.\;y + 2)$ usando ordem normal e ordem aplicativa.

   Solução (Ordem Normal):
   Aplicamos a função sem avaliar o argumento:

   $$(\lambda x.\;x) (\lambda y.\;y + 2$$

## 4.3. Impactos em Linguagens de Programação

Haskell é uma linguagem de programação que utiliza **avaliação preguiçosa**, que corresponde à **ordem normal**. Isso significa que os argumentos só são avaliados quando absolutamente necessários, o que permite trabalhar com estruturas de dados potencialmente infinitas.

**Exemplo 1**:

 ```haskell
 naturals = [0..]
 take 5 naturals
 -- Retorna [0,1,2,3,4]
 ```

 Aqui, a lista infinita `naturals` nunca é totalmente avaliada. Somente os primeiros 5 elementos são calculados, graças à avaliação preguiçosa.

Já o JavaScript usa avaliação estrita. Contudo, oferece suporte à avaliação preguiçosa por meio de geradores, que permitem gerar valores sob demanda.

**Exemplo 2**:

 ```javascript
 function* naturalNumbers() {
 let n = 0;
 While (True) yield n++;
 }

 const gen = naturalNumbers();
 console.log(gen.next().value); // Retorna 0
 console.log(gen.next().value); // Retorna 1
 ```

O código JavaScript do exemplo utiliza um gerador para criar uma sequência infinita de números naturais, produzidos sob demanda, um conceito semelhante à avaliação preguiçosa (_lazy evaluation_). Assim como na ordem normal de redução, onde os argumentos são avaliados só, e quando necessários, o gerador `naturalNumbers()` só avalia e retorna o próximo valor quando o método `next()` é chamado. Isso evita a criação imediata de uma sequência infinita e permite o uso eficiente de memória, computando os valores exclusivamente quando solicitados, como ocorre na avaliação preguiçosa.

# 5. Equivalência Lambda e Definição de Igualdade

No cálculo lambda, a noção de equivalência vai além da simples comparação sintática entre dois termos. Ela trata de quando dois termos podem ser considerados **igualmente computáveis** ou **equivalentes** em um sentido mais profundo, independentemente de suas formas superficiais. Esta equivalência tem impactos na otimizações de programas, verificação de tipos e raciocínio em linguagens funcionais.

Dois termos lambda $M$ e $N$ são considerados equivalentes, denotado por $M\to_\beta N\,$, se podemos transformar um no outro através de uma sequência, possivelmente vazia de:

1. **$\alpha$-reduções**: que permitem a renomeação de variáveis ligadas, assegurando que a identidade de variáveis internas não afeta o comportamento da função.

2. **$\beta$-reduções**: que representam a aplicação de uma função ao seu argumento, o princípio básico da computação no cálculo lambda.

3. **$\eta$-reduções**: que expressam a extensionalidade de funções, permitindo igualar duas funções que se comportam da mesma forma quando aplicadas a qualquer argumento.

>Extensionalidade refere-se ao princípio de que objetos ou funções são iguais se têm o mesmo efeito em todos os contextos possíveis. Em lógica, duas funções são consideradas extensionais se, para todo argumento, elas produzem o mesmo resultado. Em linguística, extensionalidade se refere a expressões cujo significado é determinado exclusivamente por seu valor de referência, sem levar em conta contexto ou conotação.

Formalmente, a relação $\to_\beta$ é a menor relação de equivalência que satisfaz as seguintes propriedades:

1. **redução-$beta$**: $(\lambda x.\;M)N \to_\beta M[N/x]$

   Isto significa que a aplicação de uma função $(\lambda x.\;M)$ a um argumento $N$ resulta na substituição de todas as ocorrências de $x$ em $M$ pelo valor $N\,$.

2. **$\eta$-redução**: $\lambda x.\;Mx\to_\beta M\,$, se $x$ não ocorre livre em $M$

   A $\eta$-redução captura a ideia de extensionalidade. Se uma função $\lambda x.\;Mx$ aplica $M$ a $x$ sem modificar $x\,$, ela é equivalente a $M\,$.

3. **Compatibilidade com abstração**: Se $M\to_\beta M'\,$, então $\lambda x.\;M\to_\beta \lambda x.\;M'$

   Isto garante que se dois termos são equivalentes, então suas abstrações, funções que os utilizam, serão equivalentes.

4. **Compatibilidade com aplicação**: Se $M\to_\beta M'$ e $N\to_\beta N'\,$, então $M\;N\to_\beta M'N'$

   Esta regra mostra que a equivalência se propaga para as aplicações de funções, mantendo a consistência da equivalência.

É importante notar que a ordem em que as reduções são aplicadas não afeta o resultado , devido à propriedade de Church-Rosser do cálculo lambda. Isso garante que, independentemente de como o termo é avaliado, se ele tem uma forma normal, a avaliação eventualmente a encontrará.

A relação $\to_\beta$ é uma **relação de equivalência**, o que significa que ela possui três propriedades: é uma relação **Reflexiva**. Ou seja, para todo termo $M\,$, temos que $M\to_\beta M\,$. O que significa que qualquer termo é equivalente a si mesmo, o que é esperado; é uma relação **Simétrica**. Isso significa que se $M\to_\beta N\,$, então $N\to_\beta M\,$. Se um termo $M$ pode ser transformado em $N\,$, então o oposto é similarmente verdade. E, finalmente, é uma relação **Transitiva**. Neste caso, se $M\to_\beta N$ e $N\to_\beta P\,$, então $M\to_\beta P\,$. Isso implica que, se podemos transformar $M$ em $N$ e $N$ em $P\,$, então podemos transformar diretamente $M$ em $P\,$.

A equivalência $\to_\beta$ influencia o raciocínio sobre programas em linguagens funcionais, permitindo substituições e otimizações que preservam o significado computacional. As propriedades da equivalência $\to_\beta$ garantem que podemos substituir termos equivalentes em qualquer contexto, sem alterar o significado ou o resultado da computação. Em termos de linguagens de programação, isso permite otimizações e refatorações que preservam a correção do programa.

Neste ponto, a leitora deve estar ansiosa para ver alguns exemplos de equivalência.

1. **Identidade e aplicação trivial**:

   **Exemplo 1**:

   $$\lambda x.(\lambda y.\;y)x \to_\beta \lambda x.\;x$$

   Aqui, a função interna $\lambda y.\;y$ é a função identidade, que simplesmente retorna o valor de $x\,$. Após a aplicação, obtemos $\lambda x.\;x\,$, a função identidade.

   **Exemplo 2**:

   $$\lambda z.(\lambda w.w)z \to_\beta \lambda z.\;z$$

   Assim como no exemplo original, a função interna $\lambda w.w$ é a função identidade. Após a aplicação, o valor de $z$ é retornado.

   **Exemplo 3**:

   $$\lambda a.(\lambda b.b)a \to_\beta \lambda a.a$$

   A função $\lambda b.b$ é aplicada ao valor $a\,$, retornando o próprio $a\,$. Isso demonstra mais uma aplicação da função identidade.

2. **Função constante**:

   **Exemplo 1**:

   $$(\lambda x.\lambda y.x)M\;N \to_\beta M$$

   Neste exemplo, a função $\lambda x.\lambda y.x$ retorna sempre seu primeiro argumento, ignorando o segundo. Aplicando isso a dois termos $M$ e $N\,$, o resultado é simplesmente $M\,$.

   **Exemplo 2**:

   $$(\lambda a.\lambda b.a)P Q \to_\beta P$$

   A função constante $\lambda a.\lambda b.a$ retorna sempre o primeiro argumento ($P$), ignorando $Q\,$.

   **Exemplo 3**:

   $$(\lambda u.\lambda v.u)A B \to_\beta A$$

   Aqui, o comportamento é o mesmo: o primeiro argumento ($A$) é retornado, enquanto o segundo ($B$) é ignorado.

3. **$\eta$-redução**:

   **Exemplo 1**:

   $$\lambda x.(\lambda y.M)x \to_\beta \lambda x.\;M[x/y]$$

   Se $x$ não ocorre livre em $M\,$, podemos usar a $\eta$-redução para _encurtar_ a expressão, pois aplicar $M$ a $x$ não altera o comportamento da função. Este exemplo mostra como podemos internalizar a aplicação, eliminando a dependência desnecessária de $x\,$.

   **Exemplo 2**:

   $$\lambda x.(\lambda z.N)x \to_\beta \lambda x.N[x/z]$$

   Similarmente, se $x$ não ocorre em $N\,$, a $\eta$-redução simplifica a expressão para $\lambda x.N\,$.

   **Exemplo 3**:

   $$\lambda f.(\lambda g.P)f \to_\beta \lambda f.P[f/g]$$

   Aqui, a $\eta$-redução elimina a aplicação de $f$ em $P\,$, resultando em $\lambda f.P\,$.

4. **Termo $\Omega$(não-terminante)**:

   **Exemplo 1**:

   $$(\lambda x.\;\, x\;x)(\lambda x.\;\, x\;x) \to_\beta (\lambda x.\;\, x\;x)(\lambda x.\;\, x\;x)$$

   Este é o famoso _combinador $\Omega$_, que se reduz a si mesmo indefinidamente, criando um ciclo infinito de auto-aplicações. Apesar de não ter forma normal (não termina), ele é equivalente a si mesmo por definição.

   **Exemplo 2**:

   $$(\lambda f.\;f\;f)(\lambda f.\;f\;f) \to_\beta (\lambda f.\;f\;f)(\lambda f.\;f\;f)$$

   Assim como o combinador $\Omega\,$, este termo cria um ciclo infinito de auto-aplicação.

   **Exemplo 3**:

   $$(\lambda u.\;u\;u)(\lambda u.\;u\;u) \to_\beta (\lambda u.\;u\;u)(\lambda u.\;u\;u)$$

   Outra variação do combinador $\Omega\,$, que resulta em uma redução infinita sem forma normal.

5. **Composição de funções**:

   **Exemplo 1**:

   $$(\lambda f.\lambda g.\lambda x.\;f\;(g\;x))\;M\;N \to_\beta \lambda x.\;M\;(N\;x)$$

   Neste caso, a composição de duas funções, $M$ e $N\,$, é expressa como uma função que aplica $N$ ao argumento $x\,$, e então aplica $M$ ao resultado. A redução demonstra como a composição de funções pode ser representada e simplificada no cálculo lambda.

   **Exemplo 2**:

   $$(\lambda f.\lambda g.\lambda y.\;f\;(g\;y))\;A\;B \to_\beta \lambda y.\;A\;(B\;y)$$

   A composição de $A$ e $B$ é aplicada ao argumento $y\,$, e o resultado de $By$ é então passado para $A\,$.

   **Exemplo 3**:

   $$(\lambda h.\lambda k.\lambda z.\;h\;(k\;z))\;P\;Q \to_\beta \lambda z.\;P\;(Q\;z)$$

   Similarmente, a composição de $P$ e $Q$ é aplicada ao argumento $z\,$, e o resultado de $Qz$ é passado para $P\,$.

## 5.1. Equivalência Lambda e seu Impacto em Linguagens de Programação

A equivalência lambda influencia o desenvolvimento e a otimização de linguagens funcionais como Haskell e OCaml. Essa ideia de equivalência cria uma base forte para pensar na semântica dos programas de forma abstrata. Isso é essencial para a verificação formal e otimização automática.

Dois termos lambda $M$ e $N$ são considerados equivalentes, denotado por $M\to_\beta N\,$, se é possível transformar um no outro através de uma sequência (possivelmente vazia) de:

1. $\alpha$ - reduções (renomeação de variáveis ligadas)
2. $\beta$-reduções (aplicação de funções)
3. $\eta$-conversões (extensionalidade de funções)

Formalmente:

$$
\begin{align*}
&\text{1. } (\lambda x.\;M)\;N\to_\beta M\;[N/x] \text{ (redução-$beta$)} \\
&\text{2. } \lambda x.\;Mx\to_\beta M, \text{ se $x$ não ocorre livre em $M$($\eta$-redução)} \\
&\text{3. Se } M\to_\beta M' \text{, então } \lambda x.\;M\to_\beta \lambda x.\;M' \text{ (compatibilidade com abstração)} \\
&\text{4. Se } M\to_\beta M' \text{ e } N\to_\beta N' \text{, então } M\;N\to_\beta M'\;N' \text{ (compatibilidade com aplicação)}
\end{align*}
$$

Talvez algumas aplicações em linguagem Haskell ajude a fixar os conceitos.

1. **Eliminação de Código Redundante**

   A equivalência lambda permite a substituição de expressões por versões mais simples sem alterar o comportamento do programa. Por exemplo:

   ```haskell
   -- Antes da otimização
   let x = (\y -> $Y$ + 1)\;5 in x * 2
   -- Após a otimização (equivalente)
   let x = 6 in x * 2
   ```

   Aqui, o compilador pode realizar a redução-$beta$ $(\lambda y.\;y + 1)\;5\to_\beta 6$ em tempo de compilação, simplificando o código.

2. **Transformações Seguras de Código**

   Os Compiladores podem aplicar refatorações automáticas baseadas em equivalências lambda. Por exemplo:

   ```haskell
   -- Antes da transformação
   map (\x -> f (g x)) xs

   -- Após a transformação (equivalente)
   map (f . g) xs
   ```

   Esta transformação, baseada na lei de composição $f \circ g \equiv \lambda x.\;f(g(x))\,$, pode melhorar a eficiência e legibilidade do código.

3. Inferência de Tipos

   A equivalência lambda é crucial em sistemas de tipos avançados. Considere:

   ```haskell
   -- Definição de uma função polimórfica f
   f :: (a -> b) -> ([a] -> [b])
   f g = map g
   -- Uso de f com a função show
   h :: ([Int] -> [String])
   h = f show

   -- Exemplo de uso
   main :: IO ()
   main = do
   let numbers = [1, 2, 3, 4, 5]
   print $ h numbers
   ```

   Neste exemplo:

   1. A função `f` é definida de forma polimórfica. Ela aceita uma função `g` de tipo `a -> b` e retorna uma função que mapeia listas de a para listas de `b`.

   2. A implementação de `f` usa `map`, que aplica a função `g` a cada elemento de uma lista.

   3. A função `h` é definida como uma aplicação de `f` à função show.

   4. O sistema de tipos de Haskell realiza as seguintes inferências: `show` tem o tipo `Show a \Rightarrow a \rightarrow String`. Ao aplicar `f` a show, o compilador infere que `a = Int` e `b = String`. Portanto, `h` tem o tipo `[Int] -> [String]`.

   Esta inferência demonstra como a equivalência lambda é usada pelo sistema de tipos: `f show` é equivalente a `map show`. O tipo de `map show` é inferido como `[Int] -> [String]`. No `main`, podemos ver um exemplo de uso de `h`, que converte uma lista de inteiros em uma lista de _strings_.

   O sistema de tipos usa equivalência lambda para inferir que `f show` é um termo válido do tipo `[Int] -> [String]`.

4. Avaliação Preguiçosa

Em Haskell, a equivalência lambda fundamenta a avaliação preguiçosa:

```haskell
 expensive_computation :: Integer
 expensive_computation = sum [1..1000000000]

 lazy_example :: Bool -> Integer
 lazy_example condition =
 let x = expensive_computation
 in if condition then x + 1 else 0

 main :: IO ()
 main = do
 putStrLn _Avaliando com condition = True:_
 print $ lazy_example True
 putStrLn _Avaliando com condition = False:_
 print $ lazy_example False
```

Neste exemplo: `expensive_computation` é uma função que realiza um cálculo custoso (soma dos primeiros 1 bilhão de números inteiros).

`lazy_example` é uma função que demonstra a avaliação preguiçosa. Ela aceita um argumento `booleano condition`. Dentro de `lazy_example`, `x` é definido como `expensive_computation`, mas devido à avaliação preguiçosa, este cálculo não é realizado imediatamente.

Se `condition for True`, o programa calculará `x + 1`, o que forçará a avaliação de `expensive_computation`. Se `condition for False`, o programa retornará `0`, e `expensive_computation` nunca é avaliado.

Ao executar este programa, a persistente leitora verá que: quando `condition` é _True_, o programa levará um tempo considerável para calcular o resultado. Quando `condition` é _False_, o programa retorna instantaneamente, pois `expensive_computation` não é avaliado.

Graças à equivalência lambda e à avaliação preguiçosa, `expensive_computation` só é avaliado se `condition` for verdadeira.

A equivalência Lambda, ainda que seja importante, não resolve todos os problemas possíveis. Alguns dos desafios estão relacionados com:

1. **Indecidibilidade**: Determinar se dois termos lambda são equivalentes é um problema indecidível em geral. Compiladores devem usar heurísticas e aproximações.

2. **Efeitos Colaterais**: Em linguagens com efeitos colaterais, a equivalência lambda pode não preservar a semântica do programa. Por exemplo:

   ```haskell
   -- Estas expressões não são equivalentes em presença de efeitos colaterais
   f1 = (\x -> putStrLn (_Processando _ ++ show x) >> return (x + 1))
   f2 = \x -> do
   putStrLn (_Processando _ ++ show x)
   return (x + 1)

   main = do
      let x = f1 5
      $Y$ <- x
      print y

      let z = f2 5
      w <- z
      print w
   ```

   Neste exemplo, `f1` e `f2` parecem equivalentes do ponto de vista do cálculo lambda puro. No entanto, em Haskell, que tem um sistema de I/O baseado em _monads_, elas se comportam diferentemente:

   - `f1` cria uma ação de I/O que, quando executada, imprimirá a mensagem e retornará o resultado.
   - `f2` de igual forma cria uma ação de I/O, mas a mensagem é impressa imediatamente quando `f2` for chamada.

   Ao executar este programa, a incansável leitora verá que a saída para `f1` e `f2` é diferente devido ao momento em que os efeitos colaterais (impressão) ocorrem.

3. **Complexidade Computacional**: Mesmo quando decidível, verificar equivalências pode ser computacionalmente caro, exigindo um equilíbrio entre otimização e tempo de compilação.

# 6. Funções Recursivas e o Combinador Y

No cálculo lambda, uma linguagem puramente funcional, não há uma forma direta de definir funções recursivas. Isso acontece porque, ao tentar criar uma função que se refere a si mesma, como o fatorial, acabamos com uma definição circular que o cálculo lambda puro não consegue resolver. Uma tentativa ingênua de definir o fatorial seria:

$$
\text{fac} = \lambda n.\;\text{if } (n = 0)\;\text{then } 1\;\text{else } n \times (\text{fac}\;(n - 1))
$$

Aqui, $\text{fac}$ aparece nos dois lados da equação, criando uma dependência circular. No cálculo lambda puro, não existem nomes ou atribuições; tudo se baseia em funções anônimas. _Portanto, não é possível referenciar $\text{fac}$ dentro de sua própria definição._

No cálculo lambda, todas as funções são anônimas. Não existem variáveis globais ou nomes fixos para funções. As únicas formas de vincular variáveis são:

- **Abstração lambda**: $\lambda x.\;e\,$, onde $x$ é um parâmetro e $e$ é o corpo da função.
- **Aplicação de função**: $(f\;a)\,$, onde $f$ é uma função e $a$ é um argumento.

Não há um mecanismo para definir uma função que possa se referenciar diretamente. Na definição:

$$
\text{fac} = \lambda n.\;\text{if } (n = 0)\;\text{then } 1\;\text{else } n \times (\text{fac}\;(n - 1))
$$

queremos que $\text{fac}$ possa chamar a si mesma. Mas no cálculo lambda puro:

1. **Não há nomes persistentes**: O nome $\text{fac}$ do lado esquerdo não está disponível no corpo da função à direita. Nomes em abstrações lambda são parâmetros locais.

2. **Variáveis livres devem ser vinculadas**: $\text{fac}$ aparece livre no corpo e não está ligada a nenhum parâmetro ou contexto. Isso viola as regras do cálculo lambda.

3. **Sem referência direta a si mesmo**: Não se pode referenciar uma função dentro de si mesma, pois não existe um escopo que permita isso.

Considere uma função simples no cálculo lambda:

$$\text{função} = \lambda x.\;x + 1$$

Esta função está bem definida. Mas, se tentarmos algo recursivo:

$$\text{loop} = \lambda x.\;(\text{loop}\;x)$$

O problema é o mesmo: $\text{loop}$ não está definido dentro do corpo da função. Não há como a função chamar a si mesma sem um mecanismo adicional.

Em linguagens de programação comuns, definimos funções recursivas porque o nome da função está disponível dentro do escopo. Em Haskell, por exemplo:

```haskell
fac :: Int -> Int
fac 0 = 1
fac n = n * fac (n - 1)
```

Aqui, o nome `fac` está disponível em um dos casos da função para ser chamado recursivamente. No cálculo lambda, essa forma de vinculação não existe.

## 6.1. O Combinador $Y$ como Solução

Para contornar essa limitação, usamos o conceito de **ponto fixo**. Um ponto fixo de uma função $F$ é um valor $X$ tal que $F(X) \, = X\,$. No cálculo lambda, esse conceito é implementado por meio de combinadores de ponto fixo, sendo o mais conhecido o combinador $Y\,$, atribuído a Haskell Curry.

O combinador $Y$ é definido como:

$$Y = \lambda f. (\lambda x.\;f\;(x\;x))\;(\lambda x.\;f\;(x\;x))$$

Para ilustrar o funcionamento do Y-combinator na prática, vamos implementá-lo em Haskell e usá-lo para definir a função fatorial:

```haskell
-- Definição do Y-combinator
y :: (a -> a) -> a
y f = f (y f)

-- Definição da função fatorial usando o Y-combinator
factorial :: Integer -> Integer
factorial = $Y$ $\f n -> if n == 0 then 1 else n * f (n - 1)

main :: IO ()
main = do
 print $ factorial 5 -- Saída: 120
 print $ factorial 10 -- Saída: 3628800
```

Neste exemplo, o Y-combinator (y) é usado para criar uma versão recursiva da função fatorial sem a necessidade de defini-la recursivamente de forma explícita. A função factorial é criada aplicando $Y$ a uma função que descreve o comportamento do fatorial, mas sem se referir diretamente a si mesma.
Podemos estender este exemplo para outras funções recursivas, como a sequência de Fibonacci:

```haskell
fibonacci :: Integer -> Integer
fibonacci = $Y$ $\f n -> if n <= 1 then n else f (n - 1) + f (n - 2)

main :: IO ()
main = do
 print $ map fibonacci [0..10] -- Saída: [0,1,1,2,3,5,8,13,21,34,55]
```

O Y-combinator, ou combinador-Y, tem uma propriedade interessante que a esforçada leitora deve entender:

$$Y\;F = F\;(Y\;F)$$

Isso significa que $Y\;F$ é um ponto fixo de $F\,$, permitindo que definamos funções recursivas sem a necessidade de auto-referência explícita. Quando aplicamos o combinador $Y$ a uma função $F\,$, ele retorna uma versão recursiva de $F\,$.

Matematicamente, o combinador $Y$ cria a recursão ao forçar a função $F$ a se referenciar indiretamente. O processo ocorre:

1. Aplicamos o combinador $Y$ a uma função $F\,$.

2. O $Y$ retorna uma função que, ao ser chamada, aplica $F$ a si mesma repetidamente.

3. Essa recursão acontece até que uma condição de término, como o caso base de uma função recursiva, seja atingida.

Com o combinador $Y\,$, não precisamos declarar explicitamente a recursão. O ciclo de auto-aplicação é gerado automaticamente, transformando qualquer função em uma versão recursiva de si mesma.

## 6.2. Exemplo de Função Recursiva: Fatorial

Usando o combinador $Y\,$, podemos definir corretamente a função fatorial no cálculo lambda. O fatorial de um número $n$ será:

$$
\text{factorial} = Y\;(\lambda f. \lambda n. \text{if}\;(\text{isZero}\;n)\;1\;(\text{mult}\;n\;(f\;(\text{pred}\;n))))
$$

Aqui, utilizamos funções auxiliares como $\text{isZero}\,$, $\text{mult}$ (multiplicação), e $\text{pred}$ (predecessor), todas definíveis no cálculo lambda. O combinador $Y$ cuida da recursão, aplicando a função a si mesma até que a condição de parada ($n = 0$) seja atendida. Vamos ver isso com mais detalhes usando o combinador $Y$ para definir $\text{fac}$

1. **Defina uma função auxiliar que recebe como parâmetro a função recursiva**:

   $$
   \text{Fac} = \lambda f.\;\lambda n.\;\text{if } (n = 0)\;\text{then } 1\;\text{else } n \times (f\;(n - 1))
   $$

   Aqui, $\text{Fac}$ é uma função que, dado um função $f\,$, retorna outra função que calcula o fatorial usando $f$ para a chamada recursiva.

2. **Aplique o combinador $Y$ a $\text{Fac}$ para obter a função recursiva**:

   $$\text{fac} = Y\;\text{Fac}$$

 Agora, $\text{fac}$ é uma função que calcula o fatorial de forma recursiva.

O combinador $Y$ aplica $\text{Fac}$ a si mesmo para $\text{fac}$ se expande indefinidamente, permitindo as chamadas recursivas sem referência direta ao nome da função.

**Exemplo 1**:

   Vamos calcular $\text{fac}\;2$ usando o combinador Y.

   $$Y = \lambda f.\;(\lambda x.\;f\;(x\;x))\;(\lambda x.\;f\;(x\;x))$$

   Função Fatorial:

   $$\text{fatorial} = Y\;\left (\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))) \right)$$

   Expansão da Definição de $\text{fatorial}$:

   Aplicamos $Y$ à função $\lambda f.\;\lambda n.\;\ldots$:

   $$\text{fatorial} = Y\;(\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))$$

   Então, teremos:

   $$\text{fatorial}\;2 = \left( Y\;(\lambda f.\;\lambda n.\;\ldots) \right)\;2$$

   Expandindo o Combinador Y:

   A amável leitora deve lembrar que o Combinador $Y$ é definido como:

   $$Y\;g = (\lambda x.\;g\;(x\;x))\;(\lambda x.\;g\;(x\;x))$$

   Aplicando $Y$ à função $g = \lambda f.\;\lambda n.\;\ldots$:

   $$Y\;g = (\lambda x.\;g\;(x\;x))\;(\lambda x.\;g\;(x\;x))$$

   Portanto,

   $$\text{fatorial} = (\lambda x.\;(\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))\;(x\;x))\;(\lambda x.\;(\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))\;(x\;x))$$

   Aplicando $\text{fatorial}$ a 2

   $$\text{fatorial}\;2 = \left( (\lambda x.\;\ldots)\;(\lambda x.\;\ldots) \right)\;2$$

   Simplificando as Aplicações:

   Primeira Aplicação:

   $$\text{fatorial}\;2 = \left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right)\;2$$

   Onde $F = \lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1)))\,$.

   Aplicando o Primeiro: $\lambda x$

   $$\left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right) \, = F\;\left (\left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right) \right)$$

   A atenta leitora deve ter notado que temos uma autorreferência:

   $$M = \left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right)$$

   Portanto,

   $$\text{fatorial}\;2 = F\;M\;2$$

   Aplicando $F$ com $M$ e $n = 2$:

   $$F\;M\;2 = (\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))\;M\;2$$

   Então,

   $$\text{if}\;(2 = 0)\;1\;(2 \times (M\;(2 - 1)))$$

   Como $2 \ne 0\,$, calculamos:

   $$\text{fatorial}\;2 = 2 \times (M\;1)$$

   Calculando $M\;1$:

   Precisamos calcular $M\;1\,$, onde $M$ será:

   $$M = \left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right)$$

   Então,

   $$M\;1 = \left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right)\;1 = F\;M\;1$$

   Novamente, temos:

   $$\text{fatorial}\;2 = 2 \times (F\;M\;1)$$

   Aplicando $F$ com $M$ e $n = 1$:

   $$F\;M\;1 = (\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))\;M\;1$$

   Então,

   $$\text{if}\;(1 = 0)\;1\;(1 \times (M\;(1 - 1)))$$

   Como $1 \ne 0\,$, temos:

   $$F\;M\;1 = 1 \times (M\;0)$$

   Calculando $M\;0$:

   $$M\;0 = \left (\lambda x.\;F\;(x\;x) \right)\;\left (\lambda x.\;F\;(x\;x) \right)\;0 = F\;M\;0$$

   Aplicando $F$ com $n = 0$:

   $$F\;M\;0 = (\lambda f.\;\lambda n.\;\text{if}\;(n = 0)\;1\;(n \times (f\;(n - 1))))\;M\;0$$

   Como $0 = 0\,$, temos:

   $$F\;M\;0 = 1$$

   Concluindo os Cálculos:

   $$M\;0 = 1$$

   $$F\;M\;1 = 1 \times 1 = 1$$

   $$\text{fatorial}\;2 = 2 \times 1 = 2$$

   Portanto, o cálculo do fatorial de 2 será:

   $$\text{fatorial}\;2 = 2$$

**Exemplo 2**:

   Agora, vamos verificar o cálculo de $\text{fatorial}\;3$ seguindo o mesmo procedimento.

   Aplicando $\text{fatorial}$ a 3:

   $$\text{fatorial}\;3 = F\;M\;3$$

   Onde $F$ e $M$ são como definidos anteriormente.

   Aplicando $F$ com $n = 3$:

   $$\text{if}\;(3 = 0)\;1\;(3 \times (M\;(3 - 1)))$$

   Como $3 \ne 0\,$, temos:

   $$\text{fatorial}\;3 = 3 \times (M\;2)$$

   Calculando $M\;2$:

   Seguindo o mesmo processo, teremos:

   $$M\;2 = F\;M\;2$$

   $$F\;M\;2 = 2 \times (M\;1)$$

   $$M\;1 = F\;M\;1$$

   $$F\;M\;1 = 1 \times (M\;0)$$

   $$M\;0 = F\;M\;0 = 1$$

   Calculando os Valores:

   $$M\;0 = 1$$

   $$F\;M\;1 = 1 \times 1 = 1$$

   $$M\;1 = 1$$

   $$F\;M\;2 = 2 \times 1 = 2$$

   $$M\;2 = 2$$

   $$\text{fatorial}\;3 = 3 \times 2 = 6$$

   Portanto, o cálculo do fatorial de 3 será:

   $$\text{fatorial}\;3 = 6$$

### 6.2.1. Função Fatorial Usando Funções de Ordem Superior

Vamos rever o combinador $Y\,$, desta vez, usando funções de ordem superior. Como cada função de ordem superior encapsula um conjunto de abstrações lambda reduzindo o número de reduções-$\beta$ necessário. Começamos definindo algumas funções:

1. **$\text{isZero}$**:

   $$\text{isZero} = \lambda n.\;n\;(\lambda x.\;\text{False})\;\text{True}$$

   Esta função deve retornar $TRUE$ se for aplicada a $0$ e $False$ para qualquer outro valor. Como podemos ver a seguir:
   Vamos avaliar a função $\text{isZero}$ aplicada primeiro ao número zero e depois ao número um para verificar seu funcionamento.

   1. Aplicando ao zero ($\text{isZero}\;0$):

      $$\begin{align*}
      \text{isZero}\;0 &= (\lambda n.\;n\;(\lambda x.\;\text{False})\;\text{True})\;(\lambda s.\lambda z.\;z) \\
      &\to_\beta (\lambda s.\lambda z.\;z)\;(\lambda x.\;\text{False})\;\text{True} \\
      &\to_\beta (\lambda z.\;z)\;\text{True} \\
      &\to_\beta \text{True}
      \end{align*}$$

   2. Aplicando ao um ($\text{isZero}\;1$):

      $$\begin{align*}
      \text{isZero}\;1 &= (\lambda n.\;n\;(\lambda x.\;\text{False})\;\text{True})\;(\lambda s.\lambda z.\;s\;z) \\
      &\to_\beta (\lambda s.\lambda z.\;s\;z)\;(\lambda x.\;\text{False})\;\text{True} \\
      &\to_\beta (\lambda z.\;(\lambda x.\;\text{False})\;z)\;\text{True} \\
      &\to_\beta (\lambda x.\;\text{False})\;\text{True} \\
      &\to_\beta \text{False}
      \end{align*}$$

2. **$\text{mult}$:**

   $$\text{mult} = \lambda m.\;\lambda n.\;\lambda f.\;m\;(n\;f)$$

   Esta função deve multiplicar dois números naturais. Podemos ver o resultado da sua aplicação se:
  
   1. Substituir $m$ por $2$ e $n$ por $3$
   2. Avaliar todas as reduções beta
   3. Verificar se o resultado é $6$ em notação de Church

   Com estas condições satisfeitas, a curiosa leitora pode ver o resultado da aplicação $\text{mult}\; 3\;2$:

   $$\begin{align*}
   \text{mult}\;2\;3 &= (\lambda m.\;\lambda n.\;\lambda f.\;m\;(n\;f))\;(\lambda s.\;\lambda z.\;s\;(s\;z))\;(\lambda s.\;\lambda z.\;s\;(s\;(s\;z))) \\
   \\
   &\text{Primeiro, substituímos $m$ por $2$:} \\
   &\to_\beta (\lambda n.\;\lambda f.\;(\lambda s.\;\lambda z.\;s\;(s\;z))\;(n\;f))\;(\lambda s.\;\lambda z.\;s\;(s\;(s\;z))) \\
   \\
   &\text{Agora, substituímos $n$ por $3$:} \\
   &\to_\beta \lambda f.\;(\lambda s.\;\lambda z.\;s\;(s\;z))\;((\lambda s.\;\lambda z.\;s\;(s\;(s\;z)))\;f) \\
   \\
   &\text{Calculamos $(3\;f)$, que aplica $f$ três vezes:} \\
   &\to_\beta \lambda f.\;(\lambda s.\;\lambda z.\;s\;(s\;z))\;(\lambda z.\;f\;(f\;(f\;z))) \\
   \\
   &\text{Agora aplicamos esta função duas vezes:} \\
   &\to_\beta \lambda f.\;\lambda z.\;(\lambda z.\;f\;(f\;(f\;z)))\;((\lambda z.\;f\;(f\;(f\;z)))\;z) \\
   \\
   &\text{Continuamos as reduções:} \\
   &\to_\beta \lambda f.\;\lambda z.\;f\;(f\;(f\;(f\;(f\;(f\;z))))) \\
   \end{align*}$$

   O resultado $\lambda f.\;\lambda z.\;f\;(f\;(f\;(f\;(f\;(f\;z)))))$ é a representação do número $6$ em numerais de Church, mostrando que $2 \times 3 = 6$. A função aplica $f$ exatamente seis vezes ao argumento $z$, que é precisamente a definição do numeral de Church para $6$. Copy

3. **$\text{pred}$ (Predecessor):**

   $$\text{pred} = \lambda n.\;\lambda f.\;\lambda x.\;n\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;(\lambda u.\;x)\;(\lambda u.\;u)$$

   Vamos avaliar a validade da função $\text{pred}$ aplicando a função predecessor de $2$. Esta função deve devolver o número natural imediatamente anterior ao número aplicado. A avaliação envolve várias etapas de redução beta:

   $$\begin{align*}
   \text{pred}\;2 &= (\lambda n.\;\lambda f.\;\lambda x.\;n\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;(\lambda u.\;x)\;(\lambda u.\;u))\;(\lambda s.\;\lambda z.\;s\;(s\;z)) \\
   \\
   &\text{Substituímos $n$ por $2$:} \\
   &\to_\beta \lambda f.\;\lambda x.\;(\lambda s.\;\lambda z.\;s\;(s\;z))\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;(\lambda u.\;x)\;(\lambda u.\;u) \\
   \\
   &\text{Agora aplicamos $2$ à primeira função:} \\
   &\to_\beta \lambda f.\;\lambda x.\;(\lambda z.\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;((\lambda g.\;\lambda h.\;h\;(g\;f))\;z))\;(\lambda u.\;x)\;(\lambda u.\;u) \\
   \\
   &\text{Reduzimos a aplicação interna:} \\
   &\to_\beta \lambda f.\;\lambda x.\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;(\lambda g.\;\lambda h.\;h\;(g\;f))\;(\lambda u.\;x)\;(\lambda u.\;u) \\
   \\
   &\text{Continuamos as reduções:} \\
   &\to_\beta \lambda f.\;\lambda x.\;(\lambda h.\;h\;((\lambda g.\;\lambda h.\;h\;(g\;f))\;f))\;(\lambda u.\;x)\;(\lambda u.\;u) \\
   \\
   &\text{Aplicamos à $(\lambda u.\;x)$:} \\
   &\to_\beta \lambda f.\;\lambda x.\;(\lambda u.\;x)\;((\lambda g.\;\lambda h.\;h\;(g\;f))\;f)\;(\lambda u.\;u) \\
   \\
   &\text{Continuamos reduzindo:} \\
   &\to_\beta \lambda f.\;\lambda x.\;f\;(f\;x) \\
   \end{align*}$$

   O resultado $\lambda f.\;\lambda x.\;f\;(f\;x)$ é a representação do número $1$ em numerais de Church, mostrando que o predecessor de $2$ é $1$. A função $\text{pred}$ aplica $f$ exatamente uma vez a menos que o número original, que é precisamente o que desejávamos provar.

Agora vamos definir a função fatorial usando o Combinador $Y$ e as funções de ordem superior que acabamos de definir:

$$\text{fatorial} = Y\;\left (\lambda f.\;\lambda n.\;\text{if}\;(\text{isZero}\;n)\;1\;(\text{mult}\;n\;(f\;(\text{pred}\;n))) \right)$$

**Exemplo 1**: Primeiro relembremos a definição do fatorial usando o combinador de ponto fixo:

   $$\text{fatorial} = Y\;(\lambda f.\;\lambda n.\;\text{if}\;(\text{isZero}\;n)\;1\;(\text{mult}\;n\;(f\;(\text{pred}\;n))))$$

   Agora vamos calcular $\text{fatorial}\;2$ passo a passo:

   1. Iniciando com $n = 2$:

      $$\text{fatorial}\;2 = \text{if}\;(\text{isZero}\;2)\;1\;(\text{mult}\;2\;(\text{fatorial}\;(\text{pred}\;2)))$$

   Vamos detalhar cada passo das reduções:

   2. Avaliando $\text{isZero}\;2$:

     $$\begin{align*}
     \text{isZero}\;2 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;s\;(s\;z)) \\
     &\to_\beta (\lambda s.\lambda z.\;s\;(s\;z))\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{false}
     \end{align*}$$

     Como retornou $\text{false}$, avaliamos:

     $$\begin{align*}
     &= \text{mult}\;2\;(\text{fatorial}\;(\text{pred}\;2)) \\
     &= \text{mult}\;2\;(\text{fatorial}\;1) \quad \text{(pois mostramos anteriormente que $\text{pred}\;2 = 1$)}
     \end{align*}$$

   3. Para $\text{fatorial}\;1$, primeiro avaliamos $\text{isZero}\;1$:

     $$\begin{align*}
     \text{isZero}\;1 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;s\;z) \\
     &\to_\beta (\lambda s.\lambda z.\;s\;z)\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{false}
     \end{align*}$$

     Então calculamos:

     $$\begin{align*}
     \text{fatorial}\;1 &= \text{mult}\;1\;(\text{fatorial}\;(\text{pred}\;1)) \\
     &= \text{mult}\;1\;(\text{fatorial}\;0) \quad \text{(pois $\text{pred}\;1 = 0$)}
     \end{align*}$$

   4. Para $\text{fatorial}\;0$, avaliamos $\text{isZero}\;0$:

     $$\begin{align*}
     \text{isZero}\;0 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;z) \\
     &\to_\beta (\lambda s.\lambda z.\;z)\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{true}
     \end{align*}$$

     Como retornou $\text{true}$, temos:
     $$\text{fatorial}\;0 = 1$$

   5. Agora substituímos de volta:

     $$\begin{align*}
     \text{fatorial}\;1 &= \text{mult}\;1\;(\text{fatorial}\;0) \\
     &= \text{mult}\;1\;1 \\
     &= 1 \quad \text{(pela definição de $\text{mult}$)}
     \end{align*}$$

     E finalmente:

     $$\begin{align*}
     \text{fatorial}\;2 &= \text{mult}\;2\;(\text{fatorial}\;1) \\
     &= \text{mult}\;2\;1 \\
     &= 2 \quad \text{(pela definição de $\text{mult}$)}
     \end{align*}$$

Assim, mostramos que $\text{fatorial}\;2 = 2$, mostrando cada passo da recursão e como os valores são calculados de volta até o resultado final.
A função fatorial é um exemplo clássico de recursão. Mas, não é a única.

Podemos definir uma função de exponenciação recursiva, para calcular $m^n$ usando o cálculo lambda, como:

   $$\text{power} = Y\;(\lambda f. \lambda m. \lambda n. \text{if}\;(\text{isZero}\;n)\;1\;(\text{mult}\;m\;(f\;m\;(\text{pred}\;n))))$$

Assim como fizemos no fatorial, o combinador $Y$ irá permitir a definição recursiva sem auto-referência explícita. Vamos calcular $2^2$ usando a função $\text{power}$ detalhando cada redução. Começamos aplicando $\text{power}\;2\;2$:

   1. Primeira aplicação ($n = 2$):

     $$\begin{align*}
     \text{power}\;2\;2 &= \text{if}\;(\text{isZero}\;2)\;1\;(\text{mult}\;2\;(\text{power}\;2\;(\text{pred}\;2))) \\
     \\
     \text{isZero}\;2 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;s\;(s\;z)) \\
     &\to_\beta (\lambda s.\lambda z.\;s\;(s\;z))\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{false}
     \end{align*}$$

     Como $\text{isZero}\;2$ retorna $\text{false}$, continuamos:

     $$= \text{mult}\;2\;(\text{power}\;2\;(\text{pred}\;2))$$

   2. Calculando $\text{pred}\;2$:

     $$\begin{align*}
     \text{pred}\;2 &\to_\beta 1 \quad \text{(como mostramos anteriormente)}
     \end{align*}$$

   3. Segunda aplicação ($n = 1$):

     $$\begin{align*}
     \text{power}\;2\;1 &= \text{if}\;(\text{isZero}\;1)\;1\;(\text{mult}\;2\;(\text{power}\;2\;(\text{pred}\;1))) \\
     \\
     \text{isZero}\;1 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;s\;z) \\
     &\to_\beta (\lambda s.\lambda z.\;s\;z)\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{false}
     \end{align*}$$

     Como $\text{isZero}\;1$ retorna $\text{false}$, continuamos:

     $$= \text{mult}\;2\;(\text{power}\;2\;(\text{pred}\;1))$$

   4. Calculando $\text{pred}\;1$:

     $$\begin{align*}
     \text{pred}\;1 &\to_\beta 0
     \end{align*}$$

   5. Terceira aplicação ($n = 0$):

     $$\begin{align*}
     \text{power}\;2\;0 &= \text{if}\;(\text{isZero}\;0)\;1\;(\text{mult}\;2\;(\text{power}\;2\;(\text{pred}\;0))) \\
     \\
     \text{isZero}\;0 &= (\lambda n.\;n\;(\lambda x.\;\text{false})\;\text{true})\;(\lambda s.\lambda z.\;z) \\
     &\to_\beta (\lambda s.\lambda z.\;z)\;(\lambda x.\;\text{false})\;\text{true} \\
     &\to_\beta \text{true}
     \end{align*}$$

     Como $\text{isZero}\;0$ retorna $\text{true}$:

     $$\text{power}\;2\;0 = 1$$

   6. Substituindo de volta:

     $$\begin{align*}
     \text{power}\;2\;1 &= \text{mult}\;2\;(\text{power}\;2\;0) \\
     &= \text{mult}\;2\;1 \\
     &\to_\beta 2
     \end{align*}$$

     E finalmente:

     $$\begin{align*}
     \text{power}\;2\;2 &= \text{mult}\;2\;(\text{power}\;2\;1) \\
     &= \text{mult}\;2\;2 \\
     &\to_\beta 4
     \end{align*}$$

Assim, mostramos que $\text{power}\;2\;2 = 4$, ou seja, $2^2 = 4$ em números de Church.

## 6.3. A Função de Ackermann

A Função de [Ackermann](https://en.wikipedia.org/wiki/Wilhelm_Ackermann), $\phi$, definida em cálculo lambda usando o combinador Y, ocupa uma posição singular na teoria da computação. Ela foi descoberta durante as investigações sobre computabilidade no início do século XX, quando matemáticos buscavam entender os limites do que poderia ser calculado mecanicamente. No cálculo lambda, ela é expressa como:

$$\text{Ack} = Y\;(\lambda f.\;\lambda m.\;\lambda n.\;\text{if}\;(\text{isZero}\;m)\;(\text{succ}\;n)\;(\text{if}\;(\text{isZero}\;n)\;(f\;(\text{pred}\;m)\;1)\;(f\;(\text{pred}\;m)\;(f\;m\;(\text{pred}\;n)))))$$

Sua importância é devida porque esta Função de Ackermann foi a primeira função demonstrada ser computável, porém não primitiva recursiva. As funções recursivas primitivas eram consideradas suficientes para expressar qualquer cálculo matemático prático, pois incluíam operações básicas como adição, multiplicação, exponenciação e suas generalizações.

>Funções recursivas primitivas são uma classe de funções que podem ser construídas a partir de funções básicas usando apenas composição e recursão primitiva. Sem muita profundidade, podemos definir as funções recursivas primitivas são funções que podem ser computadas usando apenas laços _For_.
>
>Formalmente, começamos com:
>
>1. A função sucessor: $S(n) = n + 1$
>2. A função zero constante: $Z(n) = 0$
>3. As funções de projeção: $P^n_i(x_1,...,x_n) = x_i$
>
>E então permitimos duas operações para construir novas funções:
>
>1. **Composição**: Se $g$ é uma função de $k$ variáveis e $h_1,...,h_k$ são funções de $n$ variáveis, então podemos formar:
>
>  $$f(x_1,...,x_n) = g(h_1(x_1,...,x_n),...,h_k(x_1,...,x_n))$$
>
>2. **Recursão Primitiva**: Se $g$ é uma função de $n$ variáveis e $h$ é uma função de $n+2$ variáveis, então podemos definir $f$ por:
>
>  $$\begin{align*}
     f(x_1,...,x_n,0) &= g(x_1,...,x_n) \\
     f(x_1,...,x_n,y+1) &= h(x_1,...,x_n,y,f(x_1,...,x_n,y))
   \end{align*}$$
>
>Usando estas regras, podemos construir funções como:
>- Adição: $a + b$ (usando recursão em $b$ com $g(a) = a$ e $h(a,b,c) = S(c)$)
>- Multiplicação: $a \times b$ (usando recursão em $b$ com $g(a) = 0$ e $h(a,b,c) = c + a$)
>- Exponenciação: $a^b$ (usando recursão em $b$ com $g(a) = 1$ e $h(a,b,c) = c \times a$)Vamos detalhar cada uma destas construções:
>
>1. Para a adição $a + b$:
>  - Fazemos recursão em $b$
>  - O caso base $g(a) = a$ significa que $a + 0 = a$
>  - No passo recursivo, $h(a,b,c) = S(c)$ onde $c$ é o resultado de $a + b$, significa que:
>
>  $$\begin{align*}
>  a + (b+1) &= S(a + b)
>  \end{align*}$$
>  
>2. Para a multiplicação $a \times b$:
>  - Fazemos recursão em $b$
>  - O caso base $g(a) = 0$ significa que $a \times 0 = 0$
>  - No passo recursivo, $h(a,b,c) = c + a$ onde $c$ é o resultado de $a \times b$, significa que:
>
>  $$\begin{align*}
>  a \times (b+1) &= (a \times b) + a
>  \end{align*}$$
>
>3. Para a exponenciação $a^b$:
>  - Fazemos recursão em $b$
>  - O caso base $g(a) = 1$ significa que $a^0 = 1$
>  - No passo recursivo, $h(a,b,c) = c \times a$ onde $c$ é o resultado de $a^b$, significa que:
>
>  $$\begin{align*}
>  a^{b+1} &= a^b \times a
>  \end{align*}$$

A função de Ackermann é significativa por demonstrar os limites das funções recursivas primitivas. Ela possui duas propriedades cruciais:

1. **Crescimento Ultra-Rápido**:

  - Para qualquer função primitiva recursiva $f(n)$, existe algum $k$ onde $A(2,k)$ cresce mais rapidamente que $f(n)$

  - Por exemplo, $A(4,2)$ já é um número tão grande que excede a quantidade de átomos no universo observável

2. **Recursão Não-Primitiva**:

  - Em funções recursivas primitivas, o valor recursivo $f(x_1,...,x_n,y)$ só pode aparecer como último argumento de $h$

  - A Ackermann usa o valor recursivo para gerar novos valores recursivos. Por exemplo:

  $$\begin{align*}
  A(m,n) &= n + 1 &\text{se } m = 0 \\
  A(m,n) &= A(m-1,1) &\text{se } m > 0 \text{ e } n = 0 \\
  A(m,n) &= A(m-1,A(m,n-1)) &\text{se } m > 0 \text{ e } n > 0
  \end{align*}$$
  
Este resultado demonstra que existem funções computáveis que não podem ser capturadas pelo esquema de recursão primitiva, motivando definições mais gerais de computabilidade como as Máquinas de Turing. Esta descoberta estabeleceu uma hierarquia de funções computáveis, onde funções como fatorial e exponenciação ocupam níveis relativamente baixos, enquanto a função de Ackermann transcende toda esta hierarquia. Em termos práticos, sua implementação em cálculo lambda serve como um teste rigoroso para sistemas que lidam com recursão profunda, pois mesmo valores pequenos como $\text{Ack}\;4\;2$ geram números astronomicamente grandes. Isto a torna uma ferramenta valiosa para testar otimizações de compiladores e explorar os limites práticos da computação recursiva.

A Função de Ackermann:

$$\text{Ack} = Y\;(\lambda f.\;\lambda m.\;\lambda n.\;\text{if}\;(\text{isZero}\;m)\;(\text{succ}\;n)\;(\text{if}\;(\text{isZero}\;n)\;(f\;(\text{pred}\;m)\;1)\;(f\;(\text{pred}\;m)\;(f\;m\;(\text{pred}\;n)))))$$

É composta por:  

1. **Estrutura Base**:

  $$\text{Ack} = Y\;(\lambda f.\;\lambda m.\;\lambda n.\;[\text{expressão}])$$

  Onde: $Y$ é o combinador de ponto fixo que permite a recursão; $f$ é a própria função (recursão) e $m$ e $n$ são os argumentos da função

2. **Primeiro Caso** (quando $m = 0$):

  $$\text{if}\;(\text{isZero}\;m)\;(\text{succ}\;n)$$

  Onde: se $m = 0$, retorna $n + 1$
  
  Exemplo: $\text{Ack}(0,n) = n + 1$

3. **Segundo Caso** (quando $m > 0$ e $n = 0$):

   $$\text{if}\;(\text{isZero}\;n)\;(f\;(\text{pred}\;m)\;1)$$
  
   Onde: se $n = 0$, calcula $\text{Ack}(m-1,1)$

   Exemplo: $\text{Ack}(m,0) = \text{Ack}(m-1,1)$

4. **Terceiro Caso** (quando $m > 0$ e $n > 0$):

   $$f\;(\text{pred}\;m)\;(f\;m\;(\text{pred}\;n))$$

   Aqui calculamos $\text{Ack}(m-1,\text{Ack}(m,n-1))$
   É uma recursão dupla:
      1. Calcula $\text{Ack}(m,n-1)$ internamente
      2. Usa esse resultado como segundo argumento para $\text{Ack}(m-1,\_)$

5. **Funções Auxiliares**:
  - $\text{isZero}$: testa se um número é zero
  - $\text{pred}$: retorna o predecessor de um número
  - $\text{succ}$: retorna o sucessor de um número

As funções $\text{isZero}$ e $\text{pred}$ foram estudadas anteriormente, falta $\text{succ}$.

   $$\text{succ} = \lambda n.\lambda f.\lambda x.f(n\;f\;x)$$

Aqui temos: $n$ é um número de Church ($\lambda f.\lambda x.f^n(x)$) e o resultado será o resultado será o número de Church imediatamente posterior a $n$, representando $n + 1$. Para entender esta função vamos realizar a redução completa de $\text{succ}\;2$, onde $2 = \lambda f.\lambda x.f(f(x))$:

   $$\begin{align*}
   \text{succ}\;2 &= (\lambda n.\lambda f.\lambda x.f(n\;f\;x))\;(\lambda f.\lambda x.f(f(x))) \\
   &\rightarrow_\beta \lambda f.\lambda x.f((\lambda f.\lambda x.f(f(x)))\;f\;x) \\
   &\rightarrow_\beta \lambda f.\lambda x.f((\lambda x.f(f(x)))\;x) \\
   &\rightarrow_\beta \lambda f.\lambda x.f(f(f(x))) \\
   \end{align*}$$

O resultado $\lambda f.\lambda x.f(f(f(x)))$ é a representação de Church do número $3$, o número imediatamente posterior a $2$. Se a curiosa leitora desejar, pode verificar que este é realmente o número $3$, se aplicá-lo a qualquer função $f$ e argumento $x$:

$$(\lambda f.\lambda x.f(f(f(x))))\;g\;a \rightarrow_\beta^* g(g(g(a)))$$

Finalmente podemos fazer uma aplicação da Função de Ackermann usando apenas cálculo lambda puro:

Vamos calcular $\text{Ack}(1,1)$ usando cálculo lambda puro. Para clareza, usaremos a seguinte notação: números de Church: $\underline{n}$ representa o número $n$; $Y$ é o combinador de ponto fixo e omitiremos algumas reduções do $Y$ por clareza.

   $$\begin{align*}
   \text{Ack}(1,1) &= Y\;(\lambda f.\;\lambda m.\;\lambda n.\;\text{if}\;(\text{isZero}\;m)\;(\text{succ}\;n)\;(\text{if}\;(\text{isZero}\;n)\;(f\;(\text{pred}\;m)\;\underline{1})\;(f\;(\text{pred}\;m)\;(f\;m\;(\text{pred}\;n)))))\;\underline{1}\;\underline{1} \\
   &\rightarrow_\beta \text{if}\;(\text{isZero}\;\underline{1})\;(\text{succ}\;\underline{1})\;(\text{if}\;(\text{isZero}\;\underline{1})\;(\text{Ack}\;(\text{pred}\;\underline{1})\;\underline{1})\;(\text{Ack}\;(\text{pred}\;\underline{1})\;(\text{Ack}\;\underline{1}\;(\text{pred}\;\underline{1})))) \\
   &\rightarrow_\beta \text{if}\;(\text{false})\;(\text{succ}\;\underline{1})\;(\text{if}\;(\text{false})\;(\text{Ack}\;\underline{0}\;\underline{1})\;(\text{Ack}\;\underline{0}\;(\text{Ack}\;\underline{1}\;\underline{0}))) \\
   &\rightarrow_\beta \text{Ack}\;\underline{0}\;(\text{Ack}\;\underline{1}\;\underline{0}) \\
   &\rightarrow_\beta \text{Ack}\;\underline{0}\;(\text{Ack}\;\underline{0}\;\underline{1}) \\
   &\rightarrow_\beta \text{Ack}\;\underline{0}\;(\text{succ}\;\underline{1}) \\
   &\rightarrow_\beta \text{Ack}\;\underline{0}\;\underline{2} \\
   &\rightarrow_\beta \text{succ}\;\underline{2} \\
   &\rightarrow_\beta \underline{3}
   \end{align*}$$

Portanto, $\text{Ack}(1,1) = 3$. A observadora leitora de perceber que: o primeiro argumento $m = 1$ é não-zero, então avaliamos o segundo `if`; o segundo argumento $n = 1$ é não-zero, então calculamos $\text{Ack}(0,\text{Ack}(1,0))$; $\text{Ack}(1,0)$ reduz para $\text{Ack}(0,1)$ que reduz para $2$. Finalmente, $\text{Ack}(0,2)$ reduz para $3$

Sem usar a notação de números de Church, é possível simplificar a Função de Ackermann:

   $$\begin{align*}
   \text{Ack} = Y\;(&\lambda f.\;\lambda m.\;\lambda n.\\
   &\lambda s.\;\lambda z.\\
   &m\;(\lambda x.n\;(\lambda y.f\;x\;(f\;m\;y\;s\;z)\;s\;z)\;(\lambda y.f\;x\;(\lambda s.\lambda z.z)\;s\;z))\;(\lambda x.n\;s\;z))
   \end{align*}$$

Onde: $Y$ é o combinador de ponto fixo $\lambda f.(\lambda x.f(x\;x))(\lambda x.f(x\;x))$; cada número $n$ é representado como $\lambda s.\lambda z.s^n(z)$; o teste $\text{isZero}$ está embutido em $m$ e $n$ através da aplicação de booleanos de Church; a redução de $\text{pred}$ acontece naturalmente na aplicação $x$ dentro de $m$; finalmente, o $\text{succ}$ está representado pela própria aplicação de $s$;

Por exemplo, para $\text{Ack}(1,1)$:

   $$\begin{align*}
   \text{Ack}\;&(\lambda s.\lambda z.s(z))\;(\lambda s.\lambda z.s(z)) \\
   &\rightarrow_\beta^* \lambda s.\lambda z.s(s(s(z)))
   \end{align*}$$

Esta é a forma onde cada operação é reduzida aos seus elementos mais básicos de abstração e aplicação.

A capacidade do cálculo lambda de expressar a função de Ackermann usando apenas funções anônimas e o combinador Y demonstra sua completude computacional de uma forma que vai além das meras definições recursivas de operações aritméticas básicas. Ela mostra que mesmo funções que transcendem a recursão primitiva podem ser expressas no minimalismo do cálculo lambda.

# 7. Estruturas de Dados Simples

As estruturas de dados são componentes da ciência da computação, e sua representação em diferentes modelos computacionais permite compreender suas propriedades e manipulações. Este capítulo explora números naturais e valores booleanos, na ótica do cálculo lambda. Começamos investigando como os números naturais são codificados usando apenas funções, através dos números de Church, e suas operações aritméticas básicas. Em seguida, exploramos como a lógica proposicional é implementada no mesmo framework.

No cálculo lambda, os números naturais são representados como funções de ordem superior que aplicam uma função $f$ a um argumento $x$ um número específico de vezes. Por exemplo, o número $2$ é representado como $\lambda s. \lambda z. s\;(s\;z)$, que aplica a função $s$ duas vezes ao argumento $z$. Esta representação permite implementar todas as operações aritméticas como manipulações dessas funções.

A lógica proposicional é construída começando com a representação dos valores verdade como funções que selecionam entre dois argumentos: $\text{True} = \lambda x. \lambda y. x$ e $\text{False} = \lambda x. \lambda y. y$. A partir dessa base, todas as operações lógicas - negação, conjunção, disjunção e outras - são implementadas como combinações dessas funções fundamentais, demonstrando como o cálculo lambda codifica não apenas números, mas também lógica e controle de fluxo.

## 7.1 Representação de Números Naturais no Cálculo Lambda

Estudar cálculo lambda após a álgebra abstrata nos faz pensar numa relação entre a representação dos números naturais por Church e a definição dos números naturais de [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor). Embora estejam em contextos teóricos diferentes, ambos tentam capturar a essência dos números naturais com estruturas básicas distintas.

Cantor é conhecido por seu trabalho na teoria dos conjuntos. Ele definiu os números naturais como um conjunto infinito e ordenado, começando do $0$ e progredindo com o operador sucessor. Para Cantor, cada número natural é construído a partir do anterior por meio de uma sucessão bem definida. O número $1$ é o sucessor de $0\,$, o número $2$ é o sucessor de $1\,$, e assim por diante. Esta estrutura fornece uma base sólida para a aritmética dos números naturais, especialmente na construção de conjuntos infinitos e na cardinalidade.

Enquanto Cantor desenvolveu sua teoria baseada em conjuntos e sucessores, Church adotou uma abordagem funcional. Em vez de tratar os números como elementos de um conjunto, Church os define como funções que operam sobre outras funções. Isso permite realizar a aritmética de forma puramente funcional. Esta diferença reflete duas abstrações dos números naturais, ambas capturando sua essência recursiva, mas com ferramentas matemáticas diferentes.

A ligação entre essas abordagens está no conceito de sucessor e na construção incremental e recursiva dos números. Embora Cantor e Church tenham trabalhado em áreas distintas — Cantor na teoria dos conjuntos e Church no cálculo lambda —, ambos representam os números naturais como entidades geradas de forma recursiva.

Agora podemos explorar os números de Church.

Os números de Church são uma representação elegante dos números naturais no cálculo lambda puro. Essa representação além de permitir a codificação dos números naturais, permite a implementação de operações aritméticas.

A ideia central dos números de Church é representar o número $n$ como uma função $f\,$. Essa função aplica outra função $f$ $n$ vezes a um argumento $x\,$. Formalmente, o número de Church para $n$ será:

$$n = \lambda s. \lambda z.\;s^n\;(z)$$

Aqui, $s^n(z)$ significa aplicar $s$ a $z\,$, $n$ vezes. $s$ representa o sucessor, e $z$ representa zero. Essa definição captura a essência dos números naturais: zero é a base, e cada número seguinte é obtido aplicando a função sucessor repetidamente.

Os primeiros números naturais podem ser representados:

- $0 = \lambda s. \lambda z.\;z$
- $1 = \lambda s. \lambda z. s\;(z)$
- $2 = \lambda s. \lambda z. s\;(s\;(z))$
- $3 = \lambda s. \lambda z. s\;(s\;(s\;(z)))$

A representação dos números naturais no cálculo lambda permite definir funções que operam sobre esses números.

A função sucessor, que incrementa um número natural como sucessor de outro número natural, é definida como:

$$\text{succ} = \lambda n. \lambda s. \lambda z.\;s\;(n\;s\;z)$$

A **função sucessor** é essencial para entender como os números naturais são criados. No entanto, para que os números de Church sejam considerados números naturais, é preciso implementar operações aritméticas no cálculo lambda puro. Para entender como isso funciona na prática, vamos aplicar $\text{succ}$ ao número $2$:

$$
\begin{aligned}
\text{succ }\;2 &= (\lambda n. \lambda s. \lambda z.\;s(n\;s\;z)) (\lambda s. \lambda z.\;s\;(s\;(z))) \\
&= \lambda s. \lambda z.\;s((\lambda s. \lambda z.\;s\;(s\;(z)))\;s\;z) \\
&= \lambda s. \lambda z.\;s\;(s\;(s\;(z))) \\
&= 3
\end{aligned}
$$

A **adição** entre dois números de Church, $m$ e $n\,$, pode ser definida como:

$$\text{add} = \lambda m. \lambda n. \lambda s. \lambda z.\;m\;s\;(n\;s\;z)$$

Essa definição funciona aplicando $n$ vezes a função $s$ ao argumento $z\,$, representando o número $n\,$. Em seguida, aplica $m$ vezes a função $s$ ao resultado anterior, somando $m$ e $n\,$. Isso itera sobre $n$ e depois sobre $m\,$, combinando as duas operações para obter a soma. Vamos ver um exemplo:

Vamos usar os números de Church para calcular a soma de $2 + 3$ em cálculo lambda puro. Primeiro, representamos os números $2$ e $3$ usando as definições de números de Church:

$$2 = \lambda s. \lambda z.\;s\;(s\;z)$$
$$3 = \lambda s. \lambda z.\;s\;(s\;(s\;z))$$

Agora, aplicamos a função de adição de números de Church:

$$\text{add} = \lambda m. \lambda n. \lambda s. \lambda z.\;m\;s\;(n\;s\;z)$$

Substituímos $m$ por $2$ e $n$ por $3$:

$$\text{add}\;2\;3 = (\lambda m. \lambda n. \lambda s. \lambda z.\;m\;s\;(n\;s\;z))\;2\;3$$

Expandimos a função:

$$= \lambda s. \lambda z. 2\;s\;(3\;s\;z)$$

Substituímos as representações de $2$ e $3$:

$$= \lambda s. \lambda z. (\lambda s. \lambda z. s\;(s\;z))\;s\;( (\lambda s. \lambda z. s\;(s\;(s\;z)))\;s\;z)$$

Primeiro, resolvemos o termo $3\;s\;z$:

$$= s\;(s\;(s\;z))$$

Agora, aplicamos $2\;s$ ao resultado:

$$= s\;(s\;(s\;(s\;(s\;z))))$$

E chegamos ao aninhamento de cinco funções o que representa $5\,$, ou seja, o resultado de $2 + 3$ em cálculo lambda puro.

A **multiplicação** de dois números de Church, $m$ e $n\,$, é expressa assim:

$$\text{mult} = \lambda m. \lambda n. \lambda s. m\;(n\;s)$$

Nesse caso, a função $n\;s$ aplica $s\,$, $n$ vezes, representando o número $n\,$. Então, aplicamos $m$ vezes essa função, resultando em $s$ sendo aplicada $m \times n$ vezes. A multiplicação, portanto, é obtida através da repetição combinada da aplicação de $s\,$. Vamos usar os números de Church para calcular a multiplicação de $2 \times 2$ em cálculo lambda puro. Primeiro, representamos o número $2$ usando a definição de números de Church:

$$2 = \lambda s. \lambda z.\;s\;(s\;z)$$

Agora, aplicamos a função de multiplicação de números de Church:

$$\text{mult} = \lambda m. \lambda n. \lambda s.\;m\;(n\;s)$$

Substituímos $m$ e $n$ por $2$:

$$\text{mult}\;2\;2 = (\lambda m. \lambda n. \lambda s.\;m\;(n\;s))\;2\;2$$

Expandimos a função:

$$= \lambda s. 2\;(2\;s)$$

Substituímos a representação de $2$:

$$= \lambda s. (\lambda s. \lambda z. s\;(s\;z))\;(2\;s)$$

Agora resolvemos o termo $2\;s$:

$$2\;s = \lambda z.\;s\;(s\;z)$$

Substituímos isso na expressão original:

$$= \lambda s. (\lambda z.\;s\;(s\;z))$$

Agora aplicamos a função $2\;s$ mais uma vez:

$$= \lambda s. \lambda z.\;s\;(s\;(s\;(s\;z)))$$

O que representa $4\,$, ou seja, o resultado de $2 \times 2$ em cálculo lambda puro.

A exponenciação, por sua vez, é dada pela fórmula:

$$\text{exp} = \lambda b. \lambda e.\;e\;b$$

Aqui, a função $e\;b$ aplica $b\,$, $e$ vezes. Como $b$ já é um número de Church, aplicar $e$ vezes sobre ele significa calcular $b^e\,$. A exponenciação é realizada repetindo a aplicação da base $b$ sobre si mesma, $e$ vezes. Vamos usar os números de Church para calcular a exponenciação $2^2$ em cálculo lambda puro. Primeiro, representamos o número $2$ usando a definição de números de Church:

$$2 = \lambda s. \lambda z.\;s\;(s\;z)$$

Agora, aplicamos a função de exponenciação de números de Church:

$$\text{exp} = \lambda b. \lambda e.\;e\;b$$

Substituímos $b$ e $e$ por $2$:

$$\text{exp}\;2\;2 = (\lambda b. \lambda e.\;e\;b)\;2\;2$$

Expandimos a função:

$$= 2\;2$$

Agora substituímos a representação de $2$ no lugar de $e$:

$$= (\lambda s. \lambda z.\;s\;(s\;z))\;2$$

Agora aplicamos $2$ a $b$:

$$= (\lambda z.\;2\;(2\;z))$$

Substituímos a definição de $2$ para ambos os termos:

$$= \lambda z. (\lambda s. \lambda z.\;s\;(s\;z))\;(\lambda s. \lambda z. s\;(s\;z))\;z$$

Aplicamos a função de $2$:

$$= \lambda z. \lambda s.\;s\;(s\;(s\;(s\;z)))$$

O que representa $4\,$, ou seja, o resultado de $2^2$ em cálculo lambda puro.

Agora, que vimos três operações básicas, podemos expandir o conceito de números de Church e incluir mais operações aritméticas.

A subtração pode ser definida de forma mais complexa, utilizando combinadores avançados como o **combinador de predecessor**. A definição é a seguinte:

$$
\text{pred} = \lambda n. \lambda f. \lambda x.\;n (\lambda g. \lambda h.\;h\;(g\;f)) (\lambda u.\;x) (\lambda u.\;u)
$$

Esta função retorna o predecessor de $n\,$, ou seja, o número $n - 1\,$. Vamos ver um exemplo de aplicação da função $\text{pred}$ e calcular o predecessor de $3\,$.

$$
\begin{aligned}
\text{pred }\;3 &= (\lambda n. \lambda f. \lambda x.\;n (\lambda g. \lambda h.\;h\;(g\;f)) (\lambda u.\;x) (\lambda u. u)) (\lambda s. \lambda z.\;s(\;s(\;s\;(z)))) \\
&= \lambda f. \lambda x.\;(\lambda s. \lambda z.\;s(\;s(\;s\;(z)))) (\lambda g. \lambda h.\;h\;(g\;f)) (\lambda u.\;x) (\lambda u.\;u) \\
&= \lambda f. \lambda x.\;f\;(f\;(x)) \\
&= 2
\end{aligned}
$$

Podemos definir a divisão como uma sequência de subtrações sucessivas e construir uma função $\text{div}$ que calcule quocientes utilizando $\text{pred}$ e $\text{mult}\,$. A expansão para números inteiros pode ser feita definindo funções adicionais para lidar com números negativos.

Para definir números negativos, e assim representar o conjunto dos números inteiros, em cálculo lambda puro, usamos **pares de Church**. Cada número é representado por dois contadores: um para os sucessores (números positivos) e outro para os predecessores (números negativos). Assim, podemos simular a existência de números negativos. O número zero é definido assim:

$$\text{zero} = \lambda s. \lambda p.\;p$$

Um número positivo, como $2\,$, seria definido assim:

$$\text{positive-2} = \lambda s. \lambda p.\;s\;(s\;p)$$

Um número negativo, como $-2\,$, seria:

$$\text{negative-2} = \lambda s. \lambda p.\;p\;(p\;s)$$

Para manipular esses números, precisamos de funções de sucessor e predecessor. A função de sucessor, que incrementa um número, é definida assim:

$$\text{succ} = \lambda n. \lambda s. \lambda p.\;s\;(n\;s\;p)$$

A função de predecessor, que decrementa um número, será:

$$\text{pred} = \lambda n. \lambda s. \lambda p. p\;(n\;s\;p)$$

Com essas definições, podemos representar e manipular números positivos e negativos. Sucessores são aplicados para aumentar números positivos, e predecessores são usados para aumentar números negativos.

Agora que temos uma representação para números negativos, vamos calcular a soma de $3 + (-2)$ usando cálculo lambda puro. Primeiro, representamos os números $3$ e $-2$ usando as definições de pares de Church:

$$3 = \lambda s. \lambda p.\;s\;(s\;(s\;p))$$
$$-2 = \lambda s. \lambda p.\;p\;(p\;s)$$

A função de adição precisa ser adaptada para lidar com pares de Church (positivos e negativos). Aqui está a nova definição de adição:

$$\text{add} = \lambda m. \lambda n. \lambda s. \lambda p.\;m\;s\;(n\;s\;p)$$

Substituímos $m$ por $3$ e $n$ por $-2$:

$$\text{add}\;3\;(-2) \, = (\lambda m. \lambda n. \lambda s. \lambda p.\;m\;s\;(n\;s\;p))\;3\;(-2)$$

Expandimos a função:

$$= \lambda s. \lambda p.\;3\;s\;(-2\;s\;p)$$

Substituímos a representação de $3$ e $-2$:

$$= \lambda s. \lambda p. (\lambda s. \lambda p.\;s\;(s\;(s\;p)))\;s\;((\lambda s. \lambda p.\;p\;(p\;s))\;s\;p)$$

Resolvemos primeiro o termo $-2\;s\;p$:

$$= p\;(p\;s)$$

Agora aplicamos o termo $3\;s$ ao resultado:

$$= s\;(s\;(p\;s))$$

O que representa $1\,$, ou seja, o resultado de $3 + (-2)$ em cálculo lambda puro.

Para implementar a divisão corretamente no cálculo lambda puro, precisamos do combinador $Y\,$, que permite recursão. O combinador $Y$ veremos o combinador $Y$ com mais cuidado em outra parte deste texto, por enquanto, basta saber que ele é definido assim:

$$Y = \lambda f. (\lambda x. f\;(x\;x)) (\lambda x. f\;(x\;x))$$

Agora, podemos definir a função de divisão com o combinador $Y\,$. Primeiro, definimos a lógica de subtração repetida:

$$\text{divLogic} = \lambda f. \lambda m. \lambda n. \lambda s. \lambda z.\;\text{ifZero}\;(\text{sub}\;m\;n)\;z\;(\lambda q. \text{succ}\;(f\;(\text{sub}\;m\;n)\;n\;s\;z))$$

Aplicamos o combinador $Y$ para permitir a recursão:

$$\text{div} = Y\;\text{divLogic}$$

Para chegar a divisão precisamos utilizar o combinador $Y$ que permite a recursão necessária para subtrair repetidamente o divisor do dividendo. A função $\text{ifZero}\,$, que não definimos anteriormente usada para verificar se o resultado da subtração é zero ou negativo. A função $\text{sub}$ subtrai $n$ de $m\,$. Finalmente usamos a função $\text{succ}$ para contar quantas vezes subtraímos o divisor.

A capacidade de representar números e operações aritméticas no cálculo lambda mostra que ele pode expressar computações lógicas e funcionais e manipulações concretas de números. Isso mostra que o cálculo lambda pode representar qualquer função computável, como afirma a Tese de Church-Turing.

Ao definir números inteiros — positivos, negativos, e zero — e as operações de adição, subtração, multiplicação e divisão, o cálculo lambda mostra que pode ser uma base teórica para linguagens de programação que manipulam dados numéricos. Isso ajuda a entender a relação entre matemática e computação.

Definir números inteiros e operações básicas no cálculo lambda demonstra a universalidade do sistema. Ao mostrar que podemos modelar aritmética e computação de forma puramente funcional, o cálculo lambda prova que pode representar qualquer função computável, inclusive as aritméticas.

Já que modelamos a aritmética podemos avançar para a implementação desses conceitos em Haskell, começando pelos números de Church:

```haskell
-- Números de Church em Haskell
type Church a = (a -> a) -> a -> a

zero :: Church a
zero = \f -> \x -> x

one :: Church a
one = \f -> \x -> f\;x

two :: Church a
two = \f -> \x -> f (f\;x)

three :: Church a
three = \f -> \x -> f (f (f\;x))

-- Função sucessor
succ' :: Church a -> Church a
succ' n = \f -> \x -> f (n f\;x)

-- Adição
add :: Church a -> Church a -> Church a
add m n = \f -> \x -> m f (n f\;x)

-- Multiplicação
mult :: Church a -> Church a -> Church a
mult m n = \f -> m (n f)

-- Conversão para Int
toInt :: Church Int -> Int
toInt n = n (+1) 0

-- Testes
main = do
 print (toInt zero) -- Saída: 0
 print (toInt one) -- Saída: 1
 print (toInt two) -- Saída: 2
 print (toInt three) -- Saída: 3
 print (toInt (succ' two)) -- Saída: 3
 print (toInt (add two three)) -- Saída: 5
 print (toInt (mult two three)) -- Saída: 6
```

Haskell é uma linguagem de programação funcional que, talvez, a curiosa leitora ainda não a conheça. Por isso, vamos explorar a implementação das operações aritméticas dos números de Church (adição e multiplicação) em C++20 e Python. Vamos ver como essas operações são transformações e como podemos implementá-las em linguagens imperativas.

A função sucessor aplica uma função $f$ a um argumento $z$ uma vez a mais do que o número existente. Aqui está uma implementação da função sucessor em C++20.

```cpp
# 6. include <iostream> // Standard library for input and output
# 7. include <functional> // Standard library for std::function, used for higher-order functions

// Define a type alias `Church` that represents a Church numeral.
// A Church numeral is a higher-order function that takes a function f and returns another function.
using Church = std::function<std::function<int(int)>(std::function<int(int)>)>;

// Define the Church numeral for 0.
// `zero` is a lambda function that takes a function `f` and returns a lambda that takes an integer `x` and returns `x` unchanged.
// This is the definition of 0 in Church numerals, which means applying `f` zero times to `x`.
Church zero = [] \ ,(auto f) {
 return [f] \ ,(int x) { return x; }; // Return the identity function, applying `f` zero times.
};

// Define the successor function `succ` that increments a Church numeral by 1.
// `succ` is a lambda function that takes a Church numeral `n` (a number in Church encoding) and returns a new function.
// The new function applies `f` to the result of applying `n(f)` to `x`, effectively adding one more application of `f`.
Church succ = [] \ ,(Church n) {
 return [n] \ ,(auto f) {
 return [n, f] \ ,(int x) {
 return f(n(f)(x)); // Apply `f` one more time than `n` does.
 };
 };
};

// Convert a Church numeral to a standard integer.
// `to_int` takes a Church numeral `n`, applies the function `[] \ ,(int x) { return x + 1; }` to it,
// which acts like a successor function in the integer world, starting from 0.
int to_int(Church n) {
 return n([] \ ,(int x) { return x + 1; })(0); // Start from 0 and apply `f` the number of times encoded by `n`.
}

int main() {
 // Create the Church numeral for 1 by applying `succ` to `zero`.
 auto one = succ(zero);

 // Create the Church numeral for 2 by applying `succ` to `one`.
 auto two = succ(one);

 // Output the integer representation of the Church numeral `two`.
 std::cout << _Sucessor de 1: _ << to_int(two) << std::endl;

 return 0; // End of program
}
```

Para finalizar podemos fazer uma implementação em python 3.x:

```python
# Define the Church numeral for 0.
# `zero` is a function that takes a function `f` and returns another function that takes an argument `x` and simply returns `x`.
# This represents applying `f` zero times to `x`, which is the definition of 0 in Church numerals.
def zero(f):
 return lambda x: x # Return the identity function, applying `f` zero times.

# Define the successor function `succ` that increments a Church numeral by 1.
# `succ` is a function that takes a Church numeral `n` (a function) and returns a new function.
# This new function applies `f` one additional time to the result of `n(f)(x)`, effectively adding 1.
def succ(n):
 return lambda f: lambda x: f(n(f)(x)) # Apply `f` one more time than `n` does.

# Convert a Church numeral to a standard integer.
# `to_int` takes a Church numeral `church_n` and applies the function `lambda x: x + 1` to it,
# which mimics the successor function for integers, starting from 0.
def to_int(church_n):
 return church_n(lambda x: x + 1)(0) # Start from 0 and apply `f` the number of times encoded by `church_n`.

# Create the Church numeral for 1 by applying `succ` to `zero`.
one = succ(zero)

# Create the Church numeral for 2 by applying `succ` to `one`.
two = succ(one)

# Output the integer representation of the Church numeral `two`.
print(_Sucessor de 1:_, to_int(two)) # This will print 2, the successor of 1 in Church numerals.
```

Poderíamos refazer todas as operações em cálculo lambda que demonstramos anteriormente em cálculo lambda puro, contudo este não é o objetivo deste texto.

A representação dos números naturais no cálculo lambda mostra como um sistema simples de funções pode codificar estruturas matemáticas complexas. Essa representação nos permite uma visão sobre a natureza da computação e a expressividade de sistemas baseados em funções. Isso prova a universalidade do cálculo lambda, mostrando que ele pode representar funções e dados.

Complementarmente a representação dos números naturais no cálculo lambda serve de base para sistemas de tipos em linguagens de programação funcionais. Ela mostra como abstrações matemáticas podem ser codificadas em funções puras. Embora linguagens como Haskell não usem diretamente os números de Church, o conceito de representar dados como funções é essencial. Em Haskell, por exemplo, listas são manipuladas com funções de ordem superior que se parecem com os números de Church.

Os números de Church mostram como o cálculo lambda pode codificar dados complexos e operações usando exclusivamente funções. Eles dão uma base sólida para entender computação e abstração em linguagens de programação.

## 7.2. Representação da Lógica Proposicional no Cálculo Lambda

O cálculo lambda oferece uma representação formal para lógica proposicional, similar aos números de Church para os números naturais. Neste cenário é possível codificar valores verdade e operações lógicas como termos lambda. Essa abordagem permite que operações booleanas sejam realizadas através de expressões funcionais.

Para entender o impacto desta representação podemos começar com os dois valores verdade, _True_ (Verdadeiro) e _False_ (Falso), que podem ser representados na forma de funções de ordem superior como:

   **Verdadeiro**: $\text{True} = \lambda x. \lambda y.\;x$

   **Falso**: $\text{False} = \lambda x. \lambda y.\;y$

Aqui, _True_ é uma função que quando aplicada a dois argumentos, retorna o primeiro, enquanto _False_ aplicada aos mesmos dois argumentos retornará o segundo. Tendo definido os termos para verdadeiro e falso, todas as operações lógicas podem ser construídas.

A esperta leitora deve concordar que é interessante, para nossos propósitos, começar definindo as operações estruturais da lógica proposicional: negação (_NOT_), conjunção (_AND_), disjunção (_OR_) e completar com a disjunção exclusiva (_XOR_) e a condicional (_IF-THEN-ELSE_).

### 7.2.1. Negação

A operação de **negação**, _NOT_ ou $\lnot$ inverte o valor de uma proposição lógica atendendo a Tabela Verdade 19.1.1.A:

| $A$    | $\text{Not } A$ |
|--------|----------------|
| True   | False          |
| False  | True           |

_Tabela Verdade 19.1.1.A. Operação de negação._{: legenda}

Utilizando o cálculo lambda a operação de negação pode ser definida como a seguinte função de ordem superior nomeada:

   $$\text{Not} = \lambda b.\;b\;\text{False}\;\text{True}$$

A função $\text{Not}$ recebe um argumento booleano $b$ e se $b$ for _True_, $\text{Not}$ _False_; caso contrário, retorna _True_. Para ver isso acontecer com funções nomeadas podemos avaliar $\text{Not}\;\text{True}$:

   $$
   \begin{align*}
   \text{Not}\;\text{True} &= (\lambda b.\; b\; \text{False}\; \text{True})\; \text{True} \\
   &\to_\beta \text{True}\; \text{False}\; \text{True} \\
   &= (\lambda x.\; \lambda y.\; x)\; \text{False}\; \text{True} \\
   &\to_\beta (\lambda y.\; \text{False})\; \text{True} \\
   &\to_\beta \text{False}
   \end{align*}
   $$

A persistente leitora irá apreciar esta mesma avaliação sem usar qualquer função nomeada, ou seja cálculo lambda puro:

   $$
   \begin{align*}
   &(\lambda b.\; b\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; x) \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x) \\
   &\to_\beta (\lambda y.\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; x) \\
   &\to_\beta (\lambda x.\; \lambda y.\; y) \\
   &\to_\beta \lambda x.\; \lambda y.\; y
   \end{align*}
   $$

### 7.2.2. Conjunção

A operação de **conjunção** é uma operação binária que retorna _True_ unicamente se ambos os operandos forem _True_ obedecendo a Tabela Verdade 19.1.1.B.

| $A$    | $B$    | $A \land B$ |
|--------|--------|------------|
| True   | True   | True       |
| True   | False  | False      |
| False  | True   | False      |
| False  | False  | False      |

_Tabela Verdade 19.1.1.B. Operação conjunção._{: legenda}

No cálculo lambda, a operação de conjunção pode ser expresso por:

   $$\text{And} = \lambda x. \lambda y.\;x\;y\;\text{False}$$

Vamos avaliar uma conjunção aplicada a dois verdadeiros, usando funções nomeadas $\text{And}\;\text{True}\;\text{False}$ passo a passo:

   $$
   \begin{align*}
   &\text{And}\;\text{True}\;\text{False} \\
   &= (\lambda x. \lambda y.\;x\;y\;\text{False})\;\text{True}\;\text{False} \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{And}$ por suas definições em cálculo lambda:} \\
   &= (\lambda x. \lambda y.\;x\;y\;(\lambda x. \lambda y.\;y))\;(\lambda x. \lambda y.\;x)\;(\lambda x. \lambda y.\;y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $(\lambda x. \lambda y.\;x)$ na função $\text{And}$:} \\
   &\to_\beta (\lambda y.\;(\lambda x. \lambda y.\;x)\;y\;(\lambda x. \lambda y.\;y))\;(\lambda x. \lambda y.\;y) \\
   \\
   &\text{Nesta etapa, a substituição de $x$ por $(\lambda x. \lambda y.\;x)$ resulta em uma nova função que depende de $y\,$. A expressão interna aplica $\text{True}$($\lambda x. \lambda y.\;x$) ao argumento $y$ e ao $\text{False}$($\lambda x. \lambda y.\;y$).} \\
   \\
   &\text{Agora, aplicamos a segunda redução beta, substituindo $y$ por $(\lambda x. \lambda y.\;y)$:} \\
   &\to_\beta (\lambda x. \lambda y.\;x)\;(\lambda x. \lambda y.\;y)\;(\lambda x. \lambda y.\;y) \\
   \\
   &\text{A substituição de $y$ por $\text{False}$ resulta na expressão acima. Aqui, $\text{True}$ é aplicada ao primeiro argumento $\text{False}\,$, ignorando o segundo argumento.} \\
   \\
   &\text{Aplicamos a próxima redução beta, aplicando $\lambda x. \lambda y.\;x$ ao primeiro argumento $(\lambda x. \lambda y.\;y)$:} \\
   &\to_\beta \lambda y.\;(\lambda x. \lambda y.\;y) \\
   \\
   &\text{Neste ponto, temos uma função que, quando aplicada a $y\,$, sempre retorna $\text{False}\,$, já que $\lambda x. \lambda y.\;x$ retorna o primeiro argumento.} \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, que ignora o argumento de $\lambda y$ e retorna diretamente $\text{False}$:} \\
   &\to_\beta \lambda x. \lambda y.\;y \\
   \\
   &\text{Esta é exatamente a definição de $\text{False}$ no cálculo lambda.} \\
   \\
   &\text{Portanto, o resultado será:} \\
   &= \text{False}
   \end{align*}
   $$

Esta aplicação fica mais simples se usarmos funções nomeadas:

   $$
   \begin{align*}
   \text{And}\;\text{True}\;\text{False} &= (\lambda x. \lambda y.\;x\;y\;\text{False})\;\text{True}\;\text{False} \\
   &\to_\beta (\lambda y.\;\text{True}\;y\;\text{False})\;\text{False} \\
   &\to_\beta \text{True}\;\text{False}\;\text{False} \\
   &= (\lambda x. \lambda y.\;x)\;\text{False}\;\text{False} \\
   &\to_\beta (\lambda y.\;\text{False})\;\text{False} \\
   &\to_\beta \text{False}
   \end{align*}
   $$

A insistente leitora pode avaliar a conjunção usando unicamente o cálculo lambda puro, sem nenhuma função nomeada.

   $$
   \begin{align*}
   &\text{And}\;\text{True}\;\text{False} \\
   &= (\lambda x.\; \lambda y.\; x\; y\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda y.\; (\lambda x.\; \lambda y.\; x)\; y\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Agora, aplicamos a terceira redução beta, aplicando $\text{True}$ ao primeiro argumento $\text{False}$:} \\
   &\to_\beta \lambda y.\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Neste ponto, temos uma função que sempre retorna $\text{False}\,$, já que $\text{True}$ ignora o segundo argumento.} \\
   \\
   &\text{Aplicamos a última redução beta, que retorna diretamente $\text{False}$:} \\
   &\to_\beta \lambda x.\; \lambda y.\; $Y$ \\
   \\
   &\text{Esta é exatamente a definição de $\text{False}$ no cálculo lambda.} \\
   \\
   &\text{Portanto, o resultado será:} \\
   &= \text{False}
   \end{align*}
   $$

### 7.2.3. Disjunção

A operação de **disjunção** retorna _True_ se pelo menos um dos operandos for _True_ em obediência a Tabela Verdade 19.1.1.C.

| $A$    | $B$    | $A \lor B$ |
|--------|--------|-----------|
| True   | True   | True      |
| True   | False  | True      |
| False  | True   | True      |
| False  | False  | False     |

_Tabela Verdade 19.1.1.C. Operação disjunção._{: legenda}

Em cálculo lambda puro a operação de disjunção pode ser definida por:

   $$\text{Or} = \lambda x. \lambda y.\;x\;\text{True}\;y$$

Vamos avaliar $\text{Or}\;\text{True}\;\text{False}$ usando somente funções nomeadas:

   $$
   \begin{align*}
   \text{Or}\;\text{True}\;\text{False} &= (\lambda x. \lambda y.\;x\;\text{True}\;y)\;\text{True}\;\text{False} \\
   &\to_\beta (\lambda y.\;\text{True}\;\text{True}\;y)\;\text{False} \\
   &\to_\beta \text{True}\;\text{True}\;\text{False} \\
   &= (\lambda x. \lambda y.\;x)\;\text{True}\;\text{False} \\
   &\to_\beta (\lambda y.\;\text{True})\;\text{False} \\
   &\to_\beta \text{True}
   \end{align*}
   $$

Vamos refazer esta mesma aplicação, porém em cálculo lambda puro:

   $$
   \begin{align*}
   \text{And}\;\text{True}\;\text{False} &= (\lambda x.\; \lambda y.\; x\; y\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{And}$ por suas definições em cálculo lambda:} \\
   &= (\lambda x.\; \lambda y.\; x\; y\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda y.\; (\lambda x.\; \lambda y.\; x)\; y\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a terceira redução beta, aplicando $\lambda x.\; \lambda y.\; x$ ao primeiro argumento $\text{False}$:} \\
   &\to_\beta \lambda y.\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Neste ponto, a função resultante é $\lambda y.\; y\,$, que é a definição de $\text{False}\,$.} \\
   \\
   &\text{Portanto, o resultado será:} \\
   &= \text{False}
   \end{align*}
   $$

### 7.2.4. Disjunção Exclusiva

A operação _Xor_ (ou **disjunção exclusiva**) retorna _True_ se um, e somente um, dos operandos for _True_ e obedece a Tabela Verdade 19.1.1.D.

| $A$    | $B$    | $A \oplus B$ |
|--------|--------|-------------|
| True   | True   | False       |
| True   | False  | True        |
| False  | True   | True        |
| False  | False  | False       |

_Tabela Verdade 19.1.1.D. Operação disjunção exclusiva_{: legenda}

Sua definição no cálculo lambda é dada por:

$$\text{Xor} = \lambda b. \lambda c. b\;(\text{Not}\;c)\;c$$

Para entender, podemos, novamente, avaliar $\text{Xor}\;\text{True}\;\text{False}$ usando funções nomeadas:

   $$
   \begin{align*}
   \text{Xor}\;\text{True}\;\text{False} &= (\lambda b. \lambda c. b\;(\text{Not}\;c)\;c)\;\text{True}\;\text{False} \\
   &\to_\beta (\lambda c. \text{True}\;(\text{Not}\;c)\;c)\;\text{False} \\
   &\to_\beta \text{True}\;(\text{Not}\;\text{False})\;\text{False} \\
   &\to_\beta \text{True}\;\text{True}\;\text{False} \\
   &= (\lambda x. \lambda y.\;x)\;\text{True}\;\text{False} \\
   &\to_\beta (\lambda y.\;\text{True})\;\text{False} \\
   &\to_\beta \text{True}
   \end{align*}
   $$

Para manter a tradição, vamos ver esta aplicação em cálculo lambda puro:

   $$
   \begin{align*}
   \text{Xor}\;\text{True}\;\text{False} &= (\lambda b.\; \lambda c.\; b\; (\text{Not}\; c)\; c)\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}\,$, $\text{Not}$ e $\text{Xor}$ por suas definições em cálculo lambda:} \\
   &= (\lambda b.\; \lambda c.\; b\; ((\lambda b.\; b\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x))\; c)\; c)\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda c.\; (\lambda x.\; \lambda y.\; x)\; ((\lambda b.\; b\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x))\; c)\; c)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $c$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; ((\lambda b.\; b\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; y))\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a próxima redução beta na expressão $\text{Not False}\,$, que é $(\lambda b.\; b\; \text{False}\; \text{True})\; \text{False}$:} \\
   &= (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a quarta redução beta, aplicando $\text{True}$ $(\lambda x.\; \lambda y.\; x)$ ao primeiro argumento:} \\
   &\to_\beta (\lambda y.\; \lambda x.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, que retorna $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &= \lambda x.\; \lambda y.\; x \\
   \\
   &\text{Portanto, o resultado é $\text{True}\,$.}
   \end{align*}
   $$

### 7.2.5. Implicação, ou condicional

A operação **implicação** ou *condicional*, retorna _True_ ou _False_, conforme a Tabela Verdade 19.1.1.E. A implicação é verdadeira quando a premissa é falsa ou quando tanto a premissa quanto a conclusão são verdadeiras.

| $A$    | $B$    | $A \to B$ |
|--------|--------|----------|
| True   | True   | True     |
| True   | False  | False    |
| False  | True   | True     |
| False  | False  | True     |

_Tabela Verdade 19.1.1.E. Operação de implicação._{: legenda}

A operação de implicação pode ser definida no cálculo lambda como:

   $$\lambda a.\; \lambda b.\; a\; b\; \text{True}$$

Essa definição de implicação retorna _False_ sempre que a premissa ($a$) é _True_ e a conclusão ($b$) é _False_. Nos demais casos, retorna _True_.

Novamente podemos ver uma aplicação da implicação usando funções nomeadas:

   $$\begin{align*}
   \text{Implicação}\;\text{True}\;\text{False} &= (\lambda a.\; \lambda b.\; a\; b\; \text{True})\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{Implicação}$ por suas definições:} \\
   &= (\lambda a.\; \lambda b.\; a\; b\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $a$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda b.\; (\lambda x.\; \lambda y.\; x)\; b\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $b$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x) \\
   \\
   &\text{Aplicamos a próxima redução beta, aplicando $\text{True}$ $(\lambda x.\; \lambda y.\; x)$ ao primeiro argumento:} \\
   &\to_\beta \lambda y.\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Neste ponto, o resultado é $\text{False}$ $(\lambda x.\; \lambda y.\; y)\,$.} \\
   \\
   &\text{Portanto, o resultado será:} \\
   &= \text{False}
   \end{align*}$$

E, ainda mantendo a tradição, vamos ver a mesma aplicação em cálculo lambda puro:

   $$\begin{align*}
   \text{Implicação}\;\text{True}\;\text{False} &= (\lambda a.\; \lambda b.\; a\; b\; \text{True})\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{Implicação}$ por suas definições em cálculo lambda:} \\
   &= (\lambda a.\; \lambda b.\; a\; b\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $a$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda b.\; (\lambda x.\; \lambda y.\; x)\; b\; (\lambda x.\; \lambda y.\; x))\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $b$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; x)\; (\lambda x.\; \lambda y.\; y)\; (\lambda x.\; \lambda y.\; x) \\
   \\
   &\text{Aplicamos a terceira redução beta, aplicando $\lambda x.\; \lambda y.\; x$ ao primeiro argumento $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta \lambda y.\; (\lambda x.\; \lambda y.\; y) \\
   \\
   &\text{Aplicamos a última redução beta, resultando em $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta \lambda x.\; \lambda y.\; $Y$ \\
   \\
   &\text{Portanto, o resultado será:} \\
   &= \text{False}
   \end{align*}$$

### 7.2.6. Operação IF-THEN-ELSE

A operação _IF-THEN-ELSE_ não é uma das operações da lógica proposicional. _IF-THEN-ELSE_ é, provavelmente, a mais popular estrutura de controle de fluxo em linguagens de programação. Em cálculo lambda podemos expressar esta operação por:

$$\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y$$

Nesta definição: $b$ representa a condição booleana (_True_ ou _False_), $b$ representa o operador condicional do _IF-THAN-ELSE_. O $x$ representa o valor retornado se $b$ for _True_ e $y$ representa o valor retornado se $b$ for _False_.

Vamos aplicar a estrutura condicional para a expressão em dois casos distintos:

1. Aplicação de _IF-THEN-ELSE_ a _True_ $x$ $y$

   $$\begin{align*}
   \text{IF-THEN-ELSE}\;\text{True}\;x\;y &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; x)\; x\; $Y$ \\
   \\
   &\text{Substituímos $\text{True}$ e $\text{IF-THEN-ELSE}$ por suas definições em cálculo lambda:} \\
   &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; x)\; x\; $Y$ \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; (\lambda x.\; \lambda y.\; x)\; x\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, aplicando $\text{True}$ $(\lambda x.\; \lambda y.\; x)$ ao argumento $x$:} \\
   &\to_\beta (\lambda y.\; x) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, retornando $x$:} \\
   &\to_\beta x
   \end{align*}$$

   Outra vez, para não perder o hábito, vamos ver esta mesma aplicação em Cálculo Lambda Puro:

   $$\begin{align*}
   \text{IF-THEN-ELSE}\;\text{True}\;x\;y &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; x)\; x\; $Y$ \\
   \\
   &\text{Substituímos $\text{True}$ e $\text{IF-THEN-ELSE}$ por suas definições:} \\
   &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; x)\; x\; $Y$ \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{True}$ $(\lambda x.\; \lambda y.\; x)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; (\lambda x.\; \lambda y.\; x)\; x\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, aplicando $\text{True}$ ao argumento $x$:} \\
   &\to_\beta (\lambda y.\; x) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, retornando $x$:} \\
   &\to_\beta x
   \end{align*}$$

2. Aplicação de _IF-THEN-ELSE_ a _False_ $x$ $y$

   $$\begin{align*}
   \text{IF-THEN-ELSE}\;\text{False}\;x\;y &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; y)\; x\; $Y$ \\
   \\
   &\text{Substituímos $\text{False}$ e $\text{IF-THEN-ELSE}$ por suas definições em cálculo lambda:} \\
   &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; y)\; x\; $Y$ \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; (\lambda x.\; \lambda y.\; y)\; x\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, aplicando $\text{False}$ $(\lambda x.\; \lambda y.\; y)$ ao argumento $y$:} \\
   &\to_\beta (\lambda y.\; y) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, retornando $y$:} \\
   &\to_\beta y
   \end{align*}$$

   Eu sei que a amável leitora não esperava por essa. Mas, eu vou refazer esta aplicação em cálculo lambda puro.

   $$\begin{align*}
   \text{IF-THEN-ELSE}\;\text{False}\;x\;y &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; y)\; x\; $Y$ \\
   \\
   &\text{Substituímos $\text{False}$ e $\text{IF-THEN-ELSE}$ por suas definições:} \\
   &= (\lambda b.\; \lambda x.\; \lambda y.\; b\; x\; y)\; (\lambda x.\; \lambda y.\; y)\; x\; $Y$ \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{False}$ $(\lambda x.\; \lambda y.\; y)$:} \\
   &\to_\beta (\lambda x.\; \lambda y.\; (\lambda x.\; \lambda y.\; y)\; x\; y) \\
   \\
   &\text{Aplicamos a segunda redução beta, aplicando $\text{False}$ ao argumento $y$:} \\
   &\to_\beta (\lambda y.\; y) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta, retornando $y$:} \\
   &\to_\beta y
   \end{align*}$$

No cálculo lambda, o _controle de fluxo_ é feito inteiramente através de funções e substituições. A estrutura _IF-THEN-ELSE_ introduz decisões condicionais a este sistema lógico. Permitindo escolher entre dois resultados diferentes com base em uma condição booleana.

Afirmamos anteriormente que o cálculo lambda é um modelo de computação completo, capaz de expressar qualquer computação que uma máquina de Turing pode executar. Para alcançar essa completude, é necessário ser capaz de tomar decisões. A função **IF-THEN-ELSE** é uma forma direta de conseguir esta funcionalidade.

## 7.3. Exercícios de Lógica Proposicional em Cálculo Lambda

**1**: Avalie a expressão $\lnot P$ onde $P$ é verdadeiro, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   \text{NOT}\;\text{True} &= (\lambda b.\; b\;\text{False}\;\text{True})\;\text{True} \\
   \\
   &\text{Substituímos $\text{True}$ e $\text{NOT}$ por suas definições:} \\
   &= (\lambda b.\; b\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x))\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{True}$ $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a segunda redução beta:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta:} \\
   &\to_\beta \lambda x.\;\lambda y.\;y
   \end{align*}$$

**2**: Calcule $P \land Q$ onde $P$ é verdadeiro e $Q$ é falso, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   \text{AND}\;\text{True}\;\text{False} &= (\lambda x.\;\lambda y.\;x\;y\;\text{False})\;\text{True}\;\text{False} \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{AND}$ por suas definições:} \\
   &= (\lambda x.\;\lambda y.\;x\;y\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $\text{True}$ $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;x)\;y\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $\text{False}$ $(\lambda x.\;\lambda y.\;y)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a terceira redução beta:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta:} \\
   &\to_\beta \lambda x.\;\lambda y.\;y
   \end{align*}$$

**3**: Avalie $P \lor Q$ onde $P$ é falso e $Q$ é verdadeiro, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   \text{OR}\;\text{False}\;\text{True} &= (\lambda x.\;\lambda y.\;x\;\text{True}\;y)\;\text{False}\;\text{True} \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}$ e $\text{OR}$ por suas definições:} \\
   &= (\lambda x.\;\lambda y.\;x\;(\lambda x.\;\lambda y.\;x)\;y)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $\text{False}$ $(\lambda x.\;\lambda y.\;y)$:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x)\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $\text{True}$ $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a terceira redução beta:} \\
   &\to_\beta (\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta:} \\
   &\to_\beta \lambda x.\;\lambda y.\;x
   \end{align*}$$

**4**: Calcule $P \oplus P$ onde $P$ é verdadeiro, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   \text{XOR}\;\text{True}\;\text{True} &= (\lambda b.\;\lambda c.\;b\;(\text{NOT}\;c)\;c)\;\text{True}\;\text{True} \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{NOT}$ e $\text{XOR}$ por suas definições:} \\
   &= (\lambda b.\;\lambda c.\;b\;((\lambda x.\;x\;\text{False}\;\text{True})\;c)\;c)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{True}$ $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda c.\;(\lambda x.\;\lambda y.\;x)\;((\lambda x.\;x\;\text{False}\;\text{True})\;c)\;c)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $c$ por $\text{True}$ $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;((\lambda x.\;x\;\text{False}\;\text{True})\;(\lambda x.\;\lambda y.\;x))\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a terceira redução beta, calculando $\text{NOT}\;\text{True}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a quarta redução beta:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta:} \\
   &\to_\beta \lambda x.\;\lambda y.\;y
   \end{align*}$$

**5**: Avalie a expressão $(\lnot (P \land Q)) \lor (R \oplus S)$ onde $P$ é verdadeiro, $Q$ é falso, $R$ é falso e $S$ é verdadeiro, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   &(\text{NOT}\;(\text{AND}\;\text{True}\;\text{False}))\;\text{OR}\;(\text{XOR}\;\text{False}\;\text{True}) \\
   \\
   &\text{Calculamos primeiro $\text{AND}\;\text{True}\;\text{False}$:} \\
   &= (\text{NOT}\;\text{False})\;\text{OR}\;(\text{XOR}\;\text{False}\;\text{True}) \\
   \\
   &\text{Calculamos $\text{NOT}\;\text{False}$:} \\
   &= \text{True}\;\text{OR}\;(\text{XOR}\;\text{False}\;\text{True}) \\
   \\
   &\text{Calculamos $\text{XOR}\;\text{False}\;\text{True}$:} \\
   &= \text{True}\;\text{OR}\;\text{True} \\
   \\
   &\text{Finalmente, calculamos $\text{True}\;\text{OR}\;\text{True}$:} \\
   &\to_\beta \lambda x.\;\lambda y.\;x
   \end{align*}$$

**6**: Calcule a expressão $P \rightarrow (Q \land R) \lor (S \lor T)$ onde $P$ é verdadeiro, $Q$ é falso, $R$ é verdadeiro, $S$ é verdadeiro e $T$ é falso, usando funções nomeadas.

   **Soluçao**:

   $$\begin{align*}
   &\text{IF-THEN-ELSE}\;\text{True}\;(\text{AND}\;\text{False}\;\text{True})\;(\text{OR}\;\text{True}\;\text{False}) \\
   \\
   &\text{Substituímos $\text{True}\,$, $\text{False}\,$, $\text{IF-THEN-ELSE}\,$, $\text{AND}$ e $\text{OR}$ por suas definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\lambda x.\;\lambda y.\;x)\;((\lambda x.\;\lambda y.\;x\;y\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x))\;((\lambda x.\;\lambda y.\;x\;\text{True}\;y)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)) \\
   \\
   &\text{Aplicamos as reduções beta para calcular $\text{AND}\;\text{False}\;\text{True}$:} \\
   &\to_\beta (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;((\lambda x.\;\lambda y.\;x\;\text{True}\;y)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)) \\
   \\
   &\text{Aplicamos as reduções beta para calcular $\text{OR}\;\text{True}\;\text{False}$:} \\
   &\to_\beta (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;x) \\
   \\
   &\text{Aplicamos a redução beta final:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;y)
   \end{align*}$$

**7**: Avalie a expressão $(P \rightarrow Q) \land (\lnot P)$ onde $P$ é verdadeiro e $Q$ é falso, em cálculo lambda puro.

   **Soluçao**:

   $$\begin{align*}
   &(\lambda b.\;\lambda c.\;b\;(\lambda x.\;\lambda y.\;y)\;c)\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $(\lambda x.\;\lambda y.\;x)$:} \\
   &\to_\beta (\lambda c.\;(\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;c)\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $c$ por $(\lambda x.\;\lambda y.\;y)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\lambda x.\;\lambda y.\;y)\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Aplicamos a terceira redução beta:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y))\;(\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Finalmente, aplicamos a última redução beta:} \\
   &\to_\beta \lambda x.\;\lambda y.\;y
   \end{align*}$$

**8**: Calcule $(P \rightarrow Q) \rightarrow ((Q \rightarrow R) \rightarrow (P \rightarrow R))$ onde $P\,$, $Q\,$, e $R$ são verdadeiros, em cálculo lambda puro.

   **Soluçao**:

   $$\begin{align*}
   &(\lambda f.\;\lambda x.\;\lambda y.\;f\;x\;(f\;x\;y))\;(\lambda a.\;\lambda b.\;a)\;(\lambda p.\;\lambda q.\;p)\;(\lambda m.\;\lambda n.\;n) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $f$ por $(\lambda a.\;\lambda b.\;a)$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;(\lambda a.\;\lambda b.\;a)\;x\;((\lambda a.\;\lambda b.\;a)\;x\;y))\;(\lambda p.\;\lambda q.\;p)\;(\lambda m.\;\lambda n.\;n) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $x$ por $(\lambda p.\;\lambda q.\;p)$:} \\
   &\to_\beta (\lambda y.\;(\lambda a.\;\lambda b.\;a)\;(\lambda p.\;\lambda q.\;p)\;((\lambda a.\;\lambda b.\;a)\;(\lambda p.\;\lambda q.\;p)\;y))\;(\lambda m.\;\lambda n.\;n) \\
   \\
   &\text{Aplicamos a terceira redução beta, substituindo $y$ por $(\lambda m.\;\lambda n.\;n)$:} \\
   &\to_\beta (\lambda a.\;\lambda b.\;a)\;(\lambda p.\;\lambda q.\;p)\;((\lambda a.\;\lambda b.\;a)\;(\lambda p.\;\lambda q.\;p)\;(\lambda m.\;\lambda n.\;n)) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta \lambda p.\;\lambda q.\;p
   \end{align*}$$

**9**: Avalie a expressão $(P \rightarrow (Q \rightarrow R)) \rightarrow ((P \rightarrow Q) \rightarrow (P \rightarrow R))$ onde $P\,$, $Q\,$, e $R$ são verdadeiros, em cálculo lambda puro.

   **Solução**:

   $$\begin{align*}
   &(\lambda x.\;\lambda y.\;\lambda z.\;x\;z\;(y\;z))\;(\lambda a.\;\lambda b.\;a)\;(\lambda c.\;\lambda d.\;d)\;(\lambda e.\;\lambda f.\;e) \\
   \\
   &\text{Substituímos $P\,$, $Q\,$, $R$ e as implicações por suas definições:} \\
   &= (\lambda x.\;\lambda y.\;\lambda z.\;x\;z\;(y\;z))\;(\lambda a.\;\lambda b.\;a)\;(\lambda c.\;\lambda d.\;d)\;(\lambda e.\;\lambda f.\;e) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $(\lambda a.\;\lambda b.\;a)$:} \\
   &\to_\beta (\lambda y.\;\lambda z.\;(\lambda a.\;\lambda b.\;a)\;z\;(y\;z))\;(\lambda c.\;\lambda d.\;d)\;(\lambda e.\;\lambda f.\;e) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $(\lambda c.\;\lambda d.\;d)$:} \\
   &\to_\beta (\lambda z.\;(\lambda a.\;\lambda b.\;a)\;z\;((\lambda c.\;\lambda d.\;d)\;z))\;(\lambda e.\;\lambda f.\;e) \\
   \\
   &\text{Aplicamos a terceira redução beta, substituindo $z$ por $(\lambda e.\;\lambda f.\;e)$:} \\
   &\to_\beta (\lambda a.\;\lambda b.\;a)\;(\lambda e.\;\lambda f.\;e)\;((\lambda c.\;\lambda d.\;d)\;(\lambda e.\;\lambda f.\;e)) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda b.\;(\lambda e.\;\lambda f.\;e))\;((\lambda c.\;\lambda d.\;d)\;(\lambda e.\;\lambda f.\;e)) \\
   &\to_\beta \lambda e.\;\lambda f.\;e \\
   \\
   &\text{O resultado é equivalente a True.}
   \end{align*}$$

**10**: Calcule a expressão $((P \rightarrow P) \rightarrow P) \rightarrow P$ onde $P$ é verdadeiro, em cálculo lambda puro.

   **Solução**:

   $$\begin{align*}
   &(\lambda x.\;\lambda y.\;x\;(x\;y))\;(\lambda a.\;\lambda b.\;a\;(a\;b))\;(\lambda p.\;\lambda q.\;p) \\
   \\
   &\text{Substituímos $P$ e as implicações por suas definições:} \\
   &= (\lambda x.\;\lambda y.\;x\;(x\;y))\;(\lambda a.\;\lambda b.\;a\;(a\;b))\;(\lambda p.\;\lambda q.\;p) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $x$ por $(\lambda a.\;\lambda b.\;a\;(a\;b))$:} \\
   &\to_\beta (\lambda y.\;(\lambda a.\;\lambda b.\;a\;(a\;b))\;((\lambda a.\;\lambda b.\;a\;(a\;b))\;y))\;(\lambda p.\;\lambda q.\;p) \\
   \\
   &\text{Aplicamos a segunda redução beta, substituindo $y$ por $(\lambda p.\;\lambda q.\;p)$:} \\
   &\to_\beta (\lambda a.\;\lambda b.\;a\;(a\;b))\;((\lambda a.\;\lambda b.\;a\;(a\;b))\;(\lambda p.\;\lambda q.\;p)) \\
   \\
   &\text{Aplicamos a terceira redução beta:} \\
   &\to_\beta (\lambda a.\;\lambda b.\;a\;(a\;b))\;(\lambda b.\;(\lambda p.\;\lambda q.\;p)\;((\lambda p.\;\lambda q.\;p)\;b)) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta \lambda b.\;(\lambda b.\;(\lambda p.\;\lambda q.\;p)\;((\lambda p.\;\lambda q.\;p)\;b))\;((\lambda b.\;(\lambda p.\;\lambda q.\;p)\;((\lambda p.\;\lambda q.\;p)\;b))\;b) \\
   &\to_\beta \lambda b.\;(\lambda p.\;\lambda q.\;p)\;((\lambda p.\;\lambda q.\;p)\;((\lambda b.\;(\lambda p.\;\lambda q.\;p)\;((\lambda p.\;\lambda q.\;p)\;b))\;b)) \\
   &\to_\beta \lambda b.\;\lambda q.\;(\lambda p.\;\lambda q.\;p) \\
   &\to_\beta \lambda b.\;\lambda q.\;\lambda q'.\;q \\
   \\
   &\text{O resultado é equivalente a True.}
   \end{align*}$$

**11**: Calculando o valor absoluto de um número. Avalie $\text{ABS} \; x$ onde $x = -5\,$, utilizando uma definição em cálculo lambda com $\text{IF-THEN-ELSE}\,$.

   **Solução**:

   $$\begin{align*}
   \text{ABS} \; (-5) &= \text{IF-THEN-ELSE} \; (x < 0) \; (-x) \; x \\
   \\
   &\text{Definições:} \\
   &\text{IF-THEN-ELSE} = (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y) \\
   &\text{TRUE} = (\lambda x.\;\lambda y.\;x), \; \text{FALSE} = (\lambda x.\;\lambda y.\;y) \\
   \\
   &\text{Substituímos as definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(-(-5))\;(-5) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{TRUE}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;\text{TRUE}\;x\;y)\;(-(-5))\;(-5) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;x)\;(-(-5))\;y)\;(-5) \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(-(-5))\;(-5) \\
   &\to_\beta \lambda y.\;(-(-5)) \\
   &= 5 \\
   \\
   &\text{O resultado é 5.}
   \end{align*}$$

**12**: Verificando se um número é par. Calcule $\text{IS-EVEN} \; x$ onde $x = 4\,$, utilizando uma definição em cálculo lambda com $\text{IF-THEN-ELSE}\,$.

   **Soluçao**:

   $$\begin{align*}
   \text{IS-EVEN} \; 4 &= \text{IF-THEN-ELSE} \; (4 \mod 2 = 0) \; \text{TRUE} \; \text{FALSE} \\
   \\
   &\text{Substituímos as definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(\text{TRUE})\;(\text{FALSE}) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{TRUE}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;\text{TRUE}\;x\;y)\;(\text{TRUE})\;(\text{FALSE}) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;x)\;(\text{TRUE})\;y)\;(\text{FALSE}) \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\text{TRUE})\;(\text{FALSE}) \\
   &\to_\beta \lambda y.\;(\text{TRUE}) \\
   \\
   &\text{O resultado é TRUE.}
   \end{align*}$$

**13**: Calculando o máximo de dois números. Calcule $\text{MAX} \; x \; y$ onde $x = 7$ e $y = 10\,$, utilizando uma definição em cálculo lambda com $\text{IF-THEN-ELSE}\,$.

   **Solução**:

   $$\begin{align*}
   \text{MAX} \; 7 \; 10 &= \text{IF-THEN-ELSE} \; (7 > 10) \; 7 \; 10 \\
   \\
   &\text{Substituímos as definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{FALSE})\;7\;10 \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{FALSE}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;\text{FALSE}\;x\;y)\;7\;10 \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;y)\;7\;y)\;10 \\
   &\to_\beta (\lambda x.\;\lambda y.\;y)\;7\;10 \\
   &\to_\beta \lambda y.\;10 \\
   \\
   &\text{O resultado é 10.}
   \end{align*}$$

**14**: Calculando o sinal de um número. Calcule $\text{SIGN} \; x$ onde $x = -8\,$, utilizando uma definição em cálculo lambda com $\text{IF-THEN-ELSE}\,$.

   **Solução**:

   $$\begin{align*}
   \text{SIGN} \; (-8) &= \text{IF-THEN-ELSE} \; (x < 0) \; (-1) \; (\text{IF-THEN-ELSE} \; (x = 0) \; 0 \; 1) \\
   \\
   &\text{Substituímos as definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(-1)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{FALSE})\;0\;1) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{TRUE}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;\text{TRUE}\;x\;y)\;(-1)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{FALSE})\;0\;1) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;x)\;(-1)\;y)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{FALSE})\;0\;1) \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(-1)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{FALSE})\;0\;1) \\
   &\to_\beta \lambda y.\;(-1) \\
   \\
   &\text{O resultado é -1.}
   \end{align*}$$

**15**: Verificando se um número está dentro de um intervalo. Calcule $\text{IN-RANGE} \; x \; \text{LOW} \; \text{HIGH}$ onde $x = 5\,$, $\text{LOW} = 3$ e $\text{HIGH} = 7\,$, utilizando uma definição em cálculo lambda com $\text{IF-THEN-ELSE}\,$.

   **Solução**:

   $$\begin{align*}
   \text{IN-RANGE} \; 5 \; 3 \; 7 &= \text{IF-THEN-ELSE} \; (5 \geq 3) \; (\text{IF-THEN-ELSE} \; (5 \leq 7) \; \text{TRUE} \; \text{FALSE}) \; \text{FALSE} \\
   \\
   &\text{Substituímos as definições:} \\
   &= (\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(\text{TRUE})\;(\text{FALSE}))\;(\text{FALSE}) \\
   \\
   &\text{Aplicamos a primeira redução beta, substituindo $b$ por $\text{TRUE}$:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;\text{TRUE}\;x\;y)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(\text{TRUE})\;(\text{FALSE}))\;(\text{FALSE}) \\
   \\
   &\text{Aplicamos a segunda redução beta:} \\
   &\to_\beta (\lambda y.\;(\lambda x.\;\lambda y.\;x)\;((\lambda b.\;\lambda x.\;\lambda y.\;b\;x\;y)\;(\text{TRUE})\;(\text{TRUE})\;(\text{FALSE}))\;y)\;(\text{FALSE}) \\
   \\
   &\text{Aplicamos as reduções beta restantes:} \\
   &\to_\beta (\lambda x.\;\lambda y.\;x)\;(\text{TRUE})\;(\text{FALSE}) \\
   &\to_\beta \lambda y.\;(\text{TRUE}) \\
   \\
   &\text{O resultado é TRUE.}
   \end{align*}$$

# 8. Estruturas de Dados Compostas

Embora o cálculo lambda puro não possua estruturas de dados nativas, podemos representá-las usando funções. Um exemplo clássico é a codificação de listas no estilo de Church, que nos permite aplicar recursão a essas estruturas.

Como a amável leitora deve lembrar, O cálculo lambda é um sistema formal para expressar computação baseada em abstração e aplicação de funções. Sendo Turing completo, o cálculo lambda pode expressar qualquer computação, ou estrutura de dados,  realizável. Podemos representar estas estruturas usando funções. Esta seção explora como listas e tuplas são representadas e manipuladas no cálculo lambda puro.

Para nos mantermos na mesma linha de raciocínio, vamos lembrar que:

   1. **Listas**: Representam coleções ordenadas de elementos, potencialmente infinitas.

   2. **Tuplas**: Representam coleções finitas e heterogêneas de elementos. Nesta seção, tuplas representarão pares ordenados.

No cálculo lambda, estas estruturas são representadas usando funções de ordem superior. Por exemplo, uma lista $[1, 2, 3]$ em cálculo lambda puro é representada como:

   $$\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil})) $$

Uma tupla $(3, 4)$ é representada como:

   $$\lambda f.\;F\;3\;4 $$

## 8.1. Listas

Para definirmos uma lista precisamos do conceito de lista vazia, que aqui será representado por $\text{nil}$ e uma função de construção de listas, $\text{cons}$:

1. Lista vazia ($\text{nil}$):

   $$\text{nil} = \lambda c.\,\lambda n.\;n$$

   Esta função ignora o primeiro argumento e retorna o segundo representando a lista vazia.

2. Construtor de lista ($\text{cons}$):

   $$\text{cons} = \lambda h. \lambda t. \lambda c. \, \lambda n.\;c\;h\;(t\;c\;n)$$

   O construtor recebe um elemento $h$ e uma lista $t\,$, e cria uma nova lista com $h$ na frente de $t\,$.

Com $\text{nil}$ e $\text{cons}\,$, podemos criar e manipular listas. Por exemplo, a lista $[1, 2, 3]$ será representada como:

   $$\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil}))$$

Esta lista está diagramada na Figura 20.1.B:

![](/assets/images/list.webp)
_Figura 6.1.B: Diagrama de uma lista em cálculo lambda_{: legenda}

Quando a leitora olha para o diagrama e para a função que representa a lista $[1,2,3]$ em cálculo lambda imagina que existe um abismo entre a sua ideia de lista e a função que encontramos. Não perca as esperanças, não é tão complicado quanto parece. Só trabalhoso. Chegamos a esta função começando com a lista vazia:

   $$\text{nil} = \lambda c. \,\lambda n.\;n$$

Adicionamos o elemento 3:

   $$\text{cons}\;3\;\text{nil} = (\lambda h. \lambda t. \lambda c. \, \lambda n.\;c\;h\;(t\;c\;n))\;3\;(\lambda c. \, \lambda n.\;n)$$

Após a redução-$beta\,$, temos:

   $$\lambda c. \, \lambda n.\;c\;3\;((\lambda c. \, \lambda n.\;n)\;c\;n)$$

Adicionamos o elemento 2:

   $$\text{cons}\;2\;(\text{cons}\;3\;\text{nil}) \, = (\lambda h. \lambda t. \lambda c. \, \lambda n.\;c\;h\;(t\;c\;n))\;2\;(\lambda c. \, \lambda n.\;c\;3\;((\lambda c. \, \lambda n.\;n)\;c\;n))$$

Após a redução-$beta\,$, obtemos:

   $$\lambda c. \, \lambda n.\;c\;2\;((\lambda c. \, \lambda n.\;c\;3\;((\lambda c. \, \lambda n.\;n)\;c\;n))\;c\;n)$$

Finalmente, adicionamos o elemento 1:

   $$\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil})) \, = (\lambda h. \lambda t. \lambda c. \, \lambda n.\;c\;h\;(t\;c\;n))\;1\;(\lambda c. \, \lambda n.\;c\;2\;((\lambda c. \, \lambda n.\;c\;3\;((\lambda c. \, \lambda n.\;n)\;c\;n))\;c\;n))$$

Após a redução-$beta\,$, a representação final será:

   $$\lambda c. \, \lambda n.\;c\;1\;((\lambda c. \, \lambda n.\;c\;2\;((\lambda c. \, \lambda n.\;c\;3\;((\lambda c. \, \lambda n.\;n)\;c\;n))\;c\;n))\;c\;n)$$

Esta é a representação completa da lista $[1, 2, 3]$ em cálculo lambda puro. Esta representação permite operações recursivas sobre listas, como mapear funções ou calcular comprimentos. A curiosa leitora pode matar a curiosidade vendo a definição de uma de duas funções em cálculo lambda puro para lidar com listas:

### 8.1.1. Função Comprimento de lista (Length)

Vamos criar uma função para calcular o comprimento de uma lista usando o combinador $Y$:

   $$\text{length} = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{succ}\;(f\;t))\;0)$$

Aqui, $\text{succ}$ é a função que retorna o sucessor de um número, e o corpo da função aplica-se recursivamente até que a lista seja esvaziada.

**Exemplo 1**: Vamos calcular o Comprimento da Lista $[1, 2, 3]$ em Cálculo Lambda Puro, usando $\text{length}$:

   Definição de $\text{length}$:

   $$\text{length} = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{succ}\;(f\;t))\;0)$$

   Representação da lista $[1, 2, 3]$:

   $$[1, 2, 3] = \lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n))$$

   Aplicamos $\text{length}$ à lista:

   $$\text{length}\;[1, 2, 3] = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{succ}\;(f\;t))\;0)\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   O combinador $Y$ permite a recursão. Após aplicá-lo, obtemos:

   $$(\lambda l. l\;(\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))\;0)\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   Aplicamos a função à lista:

   $$(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))\;(\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))\;0$$

   Reduzimos, aplicando $c$ e $n$:

   $$(\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))\;1\;((\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))\;2\;((\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))\;3\;0))$$

   Reduzimos cada aplicação de $(\lambda h. \lambda t. \text{succ}\;(\text{length}\;t))$:

   $$\text{succ}\;(\text{succ}\;(\text{succ}\;(\text{length}\;\text{nil})))$$

   Sabemos que $\text{length}\;\text{nil} = 0\,$, então:

   $$\text{succ}\;(\text{succ}\;(\text{succ}\;0))$$

   Cada $\text{succ}$ incrementa o número por 1, então o resultado é 3.

   Portanto, $\text{length}\;[1, 2, 3] = 3$ em cálculo lambda puro.

#### 8.1.1.1. Função Soma (Sum) dos Elementos de uma Lista

Da mesma forma que fizemos com a função comprimento, podemos definir uma função para somar os elementos de uma lista:

   $$\text{sum} = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{add}\;h\;(f\;t))\;0)$$

Essa função percorre a lista somando os elementos, aplicando recursão via o combinador $Y$ até que a lista seja consumida.

**Exemplo 1**: Vamos aplicar esta função à lista $[1, 2, 3]$:

   Representação da lista $[1, 2, 3]$:

   $$[1, 2, 3] = \lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n))$$

   Aplicamos $\text{sum}$ à lista:

   $$\text{sum}\;[1, 2, 3] = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{add}\;h\;(f\;t))\;0)\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   O combinador $Y$ permite a recursão. Após aplicá-lo, obtemos:

   $$(\lambda l. l\;(\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))\;0)\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   Aplicamos a função à lista:

   $$(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))\;(\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))\;0$$

   Reduzimos, aplicando $c$ e $n$:

   $$(\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))\;1\;((\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))\;2\;((\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))\;3\;0))$$

   Reduzimos cada aplicação de $(\lambda h. \lambda t. \text{add}\;h\;(\text{sum}\;t))$:

   $$\text{add}\;1\;(\text{add}\;2\;(\text{add}\;3\;(\text{sum}\;\text{nil})))$$

   Sabemos que $\text{sum}\;\text{nil} = 0\,$, então:

   $$\text{add}\;1\;(\text{add}\;2\;(\text{add}\;3\;0))$$

   Realizamos as adições de dentro para fora:

   $$\text{add}\;1\;(\text{add}\;2\;3)$$
   $$\text{add}\;1\;5$$
   $$6$$

   Portanto, $\text{sum}\;[1, 2, 3] = 6$ em cálculo lambda puro.

#### 8.1.1.2. Funções Head e Tail em Cálculo Lambda Puro

As funções Head e Tail são funções uteis na manipulação de listas, principalmente em linguagens funcionais. Vamos definir estas funções em cálculo lambda puro. Começando pela função $\text{head}$:

   $$\text{head} = \lambda l.\;l\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x)$$

Em seguida, temos a função $\text{tail}$:

   $$\text{tail} = \lambda l. l\;(\lambda h. \lambda t. t)\;(\lambda x.\;x)$$

**Exemplo 1**: Aplicação à lista [1, 2, 3]

   Aplicação de Head: Aplicamos Head à lista:

   $$\text{head}\;[1, 2, 3] = (\lambda l. l\;(\lambda h. \lambda t. h)\;(\lambda x.\;x))\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   Reduzimos:

   $$(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))\;(\lambda h. \lambda t. h)\;(\lambda x.\;x)$$

   Aplicamos $c$ e $n$:

   $$(\lambda h. \lambda t. h)\;1\;((\lambda h. \lambda t. h)\;2\;((\lambda h. \lambda t. h)\;3\;(\lambda x.\;x)))$$

   Reduzimos:

   $$1$$

   Portanto, $\text{head}\;[1, 2, 3] = 1\,$.

   Aplicação de Tail:

   Aplicamos Tail à lista:

   $$\text{tail}\;[1, 2, 3] = (\lambda l. l\;(\lambda h. \lambda t. t)\;(\lambda x.\;x))\;(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))$$

   Reduzimos:

   $$(\lambda c. \, \lambda n.\;c\;1\;(c\;2\;(c\;3\;n)))\;(\lambda h. \lambda t. t)\;(\lambda x.\;x)$$

   Aplicamos $c$ e $n$:

   $$(\lambda h. \lambda t. t)\;1\;((\lambda h. \lambda t. t)\;2\;((\lambda h. \lambda t. t)\;3\;(\lambda x.\;x)))$$

   Reduzimos:

   $$\lambda c. \, \lambda n.\;c\;2\;(c\;3\;n)$$

   Portanto, $\text{tail}\;[1, 2, 3] = [2, 3]\,$.

Listas são um tipo de dado composto útil para a maior parte das linguagens de programação. Não poderíamos deixar de definir listas em cálculo lambda puro para exemplificar a possibilidade da criação de algoritmos em cálculo lambda. Outra estrutura indispensável são as tuplas.

## 8.2. Tuplas em Cálculo Lambda Puro

Definimos uma tupla de dois elementos, que pode representar um par ordenado, como:

   $$(x, y) \, = \lambda f.\;F\;x\;y$$

A tupla $(3,4)$ é representada assim:

   $$(3, 4) \, = \lambda f.\;F\;3\;4$$

Para que uma tupla seja útil, precisamos ser capazes de trabalhar com seus elementos individualmente. Para isso, podemos definir duas funções: $\text{first}$ e $\text{follow}\,$.

### 8.2.1. Função First

A função First retorna o primeiro elemento da tupla:

   $$\text{first} = \lambda p. p\;(\lambda x. \lambda y.\;x)$$

**Exemplo**: Aplicação a $(3,4)$:

   $$\text{first}\;(3, 4) \, = (\lambda p. p\;(\lambda x. \lambda y.\;x))\;(\lambda f.\;F\;3\;4)$$

   Redução:

   $$(\lambda f.\;F\;3\;4)\;(\lambda x. \lambda y.\;x)$$

   $$(\lambda x. \lambda y.\;x)\;3\;4$$

   $$3$$

### 8.2.2. Função Last

A função Last retorna o último elemento da tupla:

   $$\text{last} = \lambda p. p\;(\lambda x. \lambda y.\;y)$$

   **Exemplo 2**: Aplicação a $(3,4)$:

   $$\text{last}\;(3, 4) \, = (\lambda p. p\;(\lambda x. \lambda y.\;y))\;(\lambda f.\;F\;3\;4)$$

   Redução:

   $$(\lambda f.\;F\;3\;4)\;(\lambda x. \lambda y.\;y)$$

   $$(\lambda x. \lambda y.\;y)\;3\;4$$

   $$4$$

## 8.3. Exercícios de Listas e Tuplas

**1**: Dada a definição de lista vazia (`nil`) e o construtor de lista (`cons`):

   $$\text{nil} = \lambda c.\,\lambda n.\;n $$

   $$\text{cons} = \lambda h. \lambda t. \lambda c. \lambda n.\;c\;h\;(t\;c\;n) $$

**2**: Construa a lista $[1, 2]$ em cálculo lambda.

   **Solução**: vamos construir a lista detalhadamente:

   1. Começamos com a lista vazia:

      $$\text{nil} $$

   2. Adicionamos o elemento 2:

      $$\text{cons}\;2\;\text{nil} $$

   3. Finalmente, adicionamos o elemento 1:

      $$\text{cons}\;1\;(\text{cons}\;2\;\text{nil}) $$

   Portanto, a representação final da lista $[1, 2]$ em cálculo lambda será:

   $$\text{cons}\;1\;(\text{cons}\;2\;\text{nil}) $$

**2**: Dada a definição de tupla de dois elementos:

   $$(x, y) = \lambda f. f\;x\;y $$

   E as funções `first` e `last`:

   $$\text{first} = \lambda p. p\;(\lambda x. \lambda y.\;x) $$

   $$\text{last} = \lambda p. p\;(\lambda x. \lambda y.\;y) $$

**3**: Crie a tupla $(3, 4)$ e aplique as funções `first` e `last` a ela.

   **Solução**:

   1. Criação da tupla $(3, 4)$:

      $$(3, 4) = \lambda f. f\;3\;4 $$

   2. Aplicação da função `first`:

      $$\text{first}\;(3, 4) = (\lambda p. p\;(\lambda x. \lambda y.\;x))\;(\lambda f. f\;3\;4) $$

      Redução:

      $$= (\lambda f. f\;3\;4)\;(\lambda x. \lambda y.\;x) $$

      $$= (\lambda x. \lambda y.\;x)\;3\;4 $$

      $$= 3 $$

   3. Aplicação da função `last`:

      $$\text{last}\;(3, 4) = (\lambda p. p\;(\lambda x. \lambda y.\;y))\;(\lambda f. f\;3\;4) $$

      Redução:
      $$= (\lambda f. f\;3\;4)\;(\lambda x. \lambda y.\;y) $$

      $$= (\lambda x. \lambda y.\;y)\;3\;4 $$

      $$= 4 $$

**3**. Dada a função `head` para listas:

   $$\text{head} = \lambda l.\;l\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x) $$

Aplique esta função à lista $[5, 6, 7]$ construída no cálculo lambda.

   **Solução**:

   1. Primeiro, construímos a lista $[5, 6, 7]$:

      $$[5, 6, 7] = \text{cons}\;5\;(\text{cons}\;6\;(\text{cons}\;7\;\text{nil})) $$

   2. Agora, aplicamos a função `head` a esta lista:

      $$\text{head}\;[5, 6, 7] = (\lambda l.\;l\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x))\;(\text{cons}\;5\;(\text{cons}\;6\;(\text{cons}\;7\;\text{nil}))) $$

   3. Redução:

      $$= (\text{cons}\;5\;(\text{cons}\;6\;(\text{cons}\;7\;\text{nil})))\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x) $$

      $$= (\lambda c. \lambda n.\;c\;5\;((\text{cons}\;6\;(\text{cons}\;7\;\text{nil}))\;c\;n))\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x) $$

      $$= (\lambda h. \lambda t.\;h)\;5\;((\text{cons}\;6\;(\text{cons}\;7\;\text{nil}))\;(\lambda h. \lambda t.\;h)\;(\lambda x.\;x)) $$

      $$= 5 $$

   Portanto, `head [5, 6, 7] = 5`.

**4**: Dada a função `tail` para listas:

   $$\text{tail} = \lambda l. l\;(\lambda h. \lambda t. t)\;(\lambda x.\;x) $$

Aplique esta função à lista $[3, 4, 5]$ construída no cálculo lambda.

   **Solução**:

   1. Primeiro, construímos a lista $[3, 4, 5]$:

   $$[3, 4, 5] = \text{cons}\;3\;(\text{cons}\;4\;(\text{cons}\;5\;\text{nil})) $$

   2. Agora, aplicamos a função `tail` a esta lista:

   $$\text{tail}\;[3, 4, 5] = (\lambda l. l\;(\lambda h. \lambda t. t)\;(\lambda x.\;x))\;(\text{cons}\;3\;(\text{cons}\;4\;(\text{cons}\;5\;\text{nil}))) $$

   3. Redução:

   $$= (\text{cons}\;3\;(\text{cons}\;4\;(\text{cons}\;5\;\text{nil})))\;(\lambda h. \lambda t. t)\;(\lambda x.\;x) $$

   $$= (\lambda c. \lambda n.\;c\;3\;((\text{cons}\;4\;(\text{cons}\;5\;\text{nil}))\;c\;n))\;(\lambda h. \lambda t. t)\;(\lambda x.\;x) $$

   $$= (\lambda h. \lambda t. t)\;3\;((\text{cons}\;4\;(\text{cons}\;5\;\text{nil}))\;(\lambda h. \lambda t. t)\;(\lambda x.\;x)) $$

   $$= (\text{cons}\;4\;(\text{cons}\;5\;\text{nil}))\;(\lambda h. \lambda t. t)\;(\lambda x.\;x) $$

   $$= \lambda c. \lambda n.\;c\;4\;((\text{cons}\;5\;\text{nil})\;c\;n) $$

   Portanto, `tail [3, 4, 5] = [4, 5]`.

**5**: Dada a definição de tupla de três elementos:

   $$(x, y, z) = \lambda f. f\;x\;y\;z $$

E a função `second` para obter o segundo elemento:

   $$\text{second} = \lambda p. p\;(\lambda x. \lambda y. \lambda z.\;y) $$

Crie a tupla $(5, 6, 7)$ e aplique a função `second` a ela.

   **Solução**:

   1. Criação da tupla $(5, 6, 7)$:

   $$(5, 6, 7) = \lambda f. f\;5\;6\;7 $$

   2. Aplicação da função `second`:

   $$\text{second}\;(5, 6, 7) = (\lambda p. p\;(\lambda x. \lambda y. \lambda z.\;y))\;(\lambda f. f\;5\;6\;7) $$

   Redução:

   $$= (\lambda f. f\;5\;6\;7)\;(\lambda x. \lambda y. \lambda z.\;y) $$

   $$= (\lambda x. \lambda y. \lambda z.\;y)\;5\;6\;7 $$

   $$= 6 $$

   Portanto, `second (5, 6, 7) = 6`.

**6**: Considere a função `map` para listas em cálculo lambda:

   $$\text{map} = Y\;(\lambda f. \lambda g. \lambda l. l\;(\lambda h. \lambda t. \text{cons}\;(g\;h)\;(f\;g\;t))\;\text{nil}) $$

Onde $Y$ é o combinador de ponto fixo. Aplique a função `map` à lista $[1, 2, 3]$ com a função $g = \lambda x. x + 1\,$.

   **Solução**:

   1. Primeiro, construímos a lista $[1, 2, 3]$:

   $$[1, 2, 3] = \text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil})) $$

   2. Definimos a função $g$:

   $$g = \lambda x. x + 1 $$

   3. Aplicamos `map g [1, 2, 3]`:

   $$\text{map}\;g\;[1, 2, 3] = Y\;(\lambda f. \lambda g. \lambda l. l\;(\lambda h. \lambda t. \text{cons}\;(g\;h)\;(f\;g\;t))\;\text{nil})\;g\;(\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil}))) $$

   4. Redução (simplificada):

   $$= \text{cons}\;(g\;1)\;(\text{map}\;g\;(\text{cons}\;2\;(\text{cons}\;3\;\text{nil}))) $$

   $$= \text{cons}\;2\;(\text{cons}\;(g\;2)\;(\text{map}\;g\;(\text{cons}\;3\;\text{nil}))) $$

   $$= \text{cons}\;2\;(\text{cons}\;3\;(\text{cons}\;(g\;3)\;(\text{map}\;g\;\text{nil}))) $$

   $$= \text{cons}\;2\;(\text{cons}\;3\;(\text{cons}\;4\;\text{nil})) $$

   Portanto, `map (λx. x + 1) [1, 2, 3] = [2, 3, 4]`.

**7**: Considere a função `filter` para listas em cálculo lambda:

   $$\text{filter} = Y\;(\lambda f. \lambda p. \lambda l. l\;(\lambda h. \lambda t. \text{if}\;(p\;h)\;(\text{cons}\;h\;(f\;p\;t))\;(f\;p\;t))\;\text{nil}) $$

Onde $Y$ é o combinador de ponto fixo e $\text{if}$ é definido como:

   $$\text{if} = \lambda c. \lambda t. \lambda f. c\;t\;f $$

Aplique a função `filter` à lista $[1, 2, 3, 4, 5]$ com o predicado $p = \lambda x. \text{isEven}\;x\,$, onde $\text{isEven}$ retorna verdadeiro para números pares.

   **Solução**:

   1. Construímos a lista $[1, 2, 3, 4, 5]$:

   $$[1, 2, 3, 4, 5] = \text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;(\text{cons}\;4\;(\text{cons}\;5\;\text{nil})))) $$

   2. Definimos o predicado $p$:

   $$p = \lambda x. \text{isEven}\;x $$

   3. Aplicamos `filter p [1, 2, 3, 4, 5]`:

   $$\text{filter}\;p\;[1, 2, 3, 4, 5] = Y\;(\lambda f. \lambda p. \lambda l. l\;(\lambda h. \lambda t. \text{if}\;(p\;h)\;(\text{cons}\;h\;(f\;p\;t))\;(f\;p\;t))\;\text{nil})\;p\;(\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;(\text{cons}\;4\;(\text{cons}\;5\;\text{nil}))))) $$

   4. Redução (simplificada):

   $$= \text{if}\;(p\;1)\;(\text{cons}\;1\;(\text{filter}\;p\;[2, 3, 4, 5]))\;(\text{filter}\;p\;[2, 3, 4, 5]) $$

   $$= \text{filter}\;p\;[2, 3, 4, 5] $$

   $$= \text{cons}\;2\;(\text{filter}\;p\;[3, 4, 5]) $$

   $$= \text{cons}\;2\;(\text{cons}\;4\;(\text{filter}\;p\;[5])) $$

   $$= \text{cons}\;2\;(\text{cons}\;4\;\text{nil}) $$

   Portanto, `filter (λx. isEven x) [1, 2, 3, 4, 5] = [2, 4]`.

**8**: Dada a definição de lista de tuplas de dois elementos:

   $$[(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)] = \text{cons}\;(\lambda f. f\;x_1\;y_1)\;(\text{cons}\;(\lambda f. f\;x_2\;y_2)\;(...\;(\text{cons}\;(\lambda f. f\;x_n\;y_n)\;\text{nil})...)) $$

E a função `sumPairs` que soma os elementos de cada par em uma lista de pares:

$$\text{sumPairs} = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{cons}\;(h\;(\lambda x. \lambda y. x + y))\;(f\;t))\;\text{nil})$$

Aplique a função `sumPairs` à lista $[(1, 2), (3, 4), (5, 6)]\,$.

   **Solução**:

   1. Construímos a lista $[(1, 2), (3, 4), (5, 6)]$:

   $$[(1, 2), (3, 4), (5, 6)] = \text{cons}\;(\lambda f. f\;1\;2)\;(\text{cons}\;(\lambda f. f\;3\;4)\;(\text{cons}\;(\lambda f. f\;5\;6)\;\text{nil})) $$

   2. Aplicamos `sumPairs` à lista:

   $$
   \text{sum\_pairs} = Y\;(\lambda f. \lambda l. l\;(\lambda h. \lambda t. \text{cons}\;(h\;(\lambda x. \lambda y. x + y))\;(f\;t))\;\text{nil})
   $$

   3. Redução (simplificada):

   $$= \text{cons}\;((\lambda f. f\;1\;2)\;(\lambda x. \lambda y. x + y))\;(\text{sumPairs}\;(\text{cons}\;(\lambda f. f\;3\;4)\;(\text{cons}\;(\lambda f. f\;5\;6)\;\text{nil}))) $$

   $$= \text{cons}\;3\;(\text{cons}\;((\lambda f. f\;3\;4)\;(\lambda x. \lambda y. x + y))\;(\text{sumPairs}\;(\text{cons}\;(\lambda f. f\;5\;6)\;\text{nil}))) $$

   $$= \text{cons}\;3\;(\text{cons}\;7\;(\text{cons}\;((\lambda f. f\;5\;6)\;(\lambda x. \lambda y. x + y))\;(\text{sumPairs}\;\text{nil}))) $$

   $$= \text{cons}\;3\;(\text{cons}\;7\;(\text{cons}\;11\;\text{nil})) $$

   Portanto, `sumPairs [(1, 2), (3, 4), (5, 6)] = [3, 7, 11]`.

**9**: Considere a função `foldRight`, eventualmente conhecida como `reduce`) para listas em cálculo lambda:

   $$\text{foldRight} = Y\;(\lambda f. \lambda g. \lambda a. \lambda l. l\;(\lambda h. \lambda t. g\;h\;(f\;g\;a\;t))\;a) $$

Onde $Y$ é o combinador de ponto fixo. Use `foldRight` para implementar a função `length` que calcula o comprimento de uma lista.

   **Solução**:

   1. Para implementar `length` usando `foldRight`, precisamos de uma função $g$ que incremente um contador para cada elemento da lista, e um valor inicial $a$ de 0. Definimos:

   $$g = \lambda x. \lambda acc. \text{succ}\;acc $$
   $$a = 0 $$

   2. Agora, podemos definir `length` como:

   $$\text{length} = \lambda l.\;\text{foldRight}\;(\lambda x. \lambda acc.\;\text{succ}\;acc)\;0\;l$$

   3. Para testar, vamos aplicar `length` à lista $[1, 2, 3, 4]$:

   $$\text{length}\;[1, 2, 3, 4] = \text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;(\text{cons}\;1\;(\text{cons}\;2\;(\text{cons}\;3\;(\text{cons}\;4\;\text{nil})))) $$

   4. Redução (simplificada):

   $$= (\lambda x. \lambda acc. \text{succ}\;acc)\;1\;(\text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;[2, 3, 4])$$

   $$= \text{succ}\;(\text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;[2, 3, 4])$$

   $$= \text{succ}\;(\text{succ}\;(\text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;[3, 4]))$$

   $$= \text{succ}\;(\text{succ}\;(\text{succ}\;(\text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;[4])))$$

   $$= \text{succ}\;(\text{succ}\;(\text{succ}\;(\text{succ}\;(\text{foldRight}\;(\lambda x. \lambda acc. \text{succ}\;acc)\;0\;\text{nil})))) $$

   $$= \text{succ}\;(\text{succ}\;(\text{succ}\;(\text{succ}\;0))) $$

Portanto, `length [1, 2, 3, 4] = 4`.

**10**. Dada a definição de árvore binária em cálculo lambda:

   $$\text{leaf} = \lambda v. \lambda f. f\;v\;\text{nil}\;\text{nil} $$

   $$\text{node} = \lambda v. \lambda l. \lambda r. \lambda f. f\;v\;l\;r $$

E a função `treeMap` que aplica uma função a todos os valores em uma árvore:

   $$
   \text{treeMap} = Y\;(\lambda f. \lambda g. \lambda t. t\;(\lambda v. \lambda l. \lambda r. \text{node}\;(g\;v)\;(f\;g\;l)\;(f\;g\;r)))
   $$

   Aplique `treeMap` à árvore $\text{node}\;1\;(\text{leaf}\;2)\;(\text{leaf}\;3)$ com a função $g = \lambda x. x * 2\,$.

   **Solução**:

   1. Definimos a árvore:

   $$\text{tree} = \text{node}\;1\;(\text{leaf}\;2)\;(\text{leaf}\;3) $$

   2. Definimos a função $g$:

   $$g = \lambda x. x * 2 $$

   3. Aplicamos `treeMap g tree`:

   $$
   \text{treeMap}\;g\;\text{tree} = Y\;(\lambda f. \lambda g. \lambda t. t\;(\lambda v. \lambda l. \lambda r. \text{node}\;(g\;v)\;(f\;g\;l)\;(f\;g\;r)))\;g\;(\text{node}\;1\;(\text{leaf}\;2)\;(\text{leaf}\;3))
   $$

   4. Redução (simplificada):

   $$= \text{node}\;(g\;1)\;(\text{treeMap}\;g\;(\text{leaf}\;2))\;(\text{treeMap}\;g\;(\text{leaf}\;3))$$

   $$= \text{node}\;2\;(\text{leaf}\;(g\;2))\;(\text{leaf}\;(g\;3))$$

   $$= \text{node}\;2\;(\text{leaf}\;4)\;(\text{leaf}\;6)$$

   Portanto, `treeMap (λx. x * 2) (node 1 (leaf 2) (leaf 3)) = node 2 (leaf 4) (leaf 6)`.

# 9. Cálculo Lambda e Haskell

Haskell implementa diretamente conceitos do cálculo lambda. Vejamos alguns exemplos:

1. **Funções Lambda**: em Haskell, funções lambda são criadas usando a sintaxe \x -> ..., que é análoga à notação $\lambda x.$ do cálculo lambda.

   ```haskell
   -- Cálculo lambda: \lambda x.\;x
   identidade = \x -> x
   -- Cálculo lambda: \lambda x.\lambda y.x
   constante = \x -> \y -> x
   -- Uso:
   main = do
   print (identidade 5) -- Saída: 5
   print (constante 3 4) -- Saída: 3
   ```

2. **Aplicação de Função**: a aplicação de função em Haskell é semelhante ao cálculo lambda, usando justaposição:

   ```haskell
   -- Cálculo lambda: (\lambda x.\;x+1)\;5
   incrementar = (\x -> x + 1)\;5
   main = print incrementar -- Saída: 6
   ```

3. *currying*: Haskell usa _currying_por padrão, permitindo aplicação parcial de funções:

   ```haskell
   -- Função de dois argumentos
   soma :: Int -> Int -> Int
   soma x\;y = x + y
   -- Aplicação parcial
   incrementar :: Int -> Int
   incrementar = soma 1

   main = do
   print (soma 2 3) -- Saída: 5
   print (incrementar 4) -- Saída: 5
   ```

4. **Funções de Ordem Superior**: Haskell suporta funções de ordem superior, um dos conceitos do cálculo lambda:

   ```haskell
   -- map é uma função de ordem superior
   dobrarLista :: [Int] -> [Int]
   dobrarLista = map (\x -> 2 * x)

   main = print (dobrarLista [1,2,3]) -- Saída: [2,4,6]
   ```

5. **Codificação de Dados**: no cálculo lambda puro, não existem tipos de dados primitivos além de funções. Haskell, sendo uma linguagem prática, fornece tipos de dados primitivos, mas ainda permite codificações similares às do cálculo lambda.

6. **Booleanos**: no cálculo lambda, os booleanos podem ser codificados como:

$$
   \begin{aligned}
   \text{True} &= \lambda x.\lambda y.x \\
   \text{False} &= \lambda x.\lambda y.\;y\\
   \end{aligned}
$$

   Em Haskell, podemos implementar isso como:

   ```haskell
   True :: a -> a -> a
   True = \x -> \y -> x

   False :: a -> a -> a
   False = \x -> \y -> y

   -- Função if-then-else
   if' :: (a -> a -> a) -> a -> a -> a
   if' b t e = b t e

   main = do
      print (if' True _verdadeiro_ _falso_) -- Saída: _verdadeiro_
      print (if' False _verdadeiro_ _falso_) -- Saída: _falso_
   ```

7. Números Naturais: os números naturais podem ser representados usando a codificação de Church:

$$
   \begin{aligned}
   0 &= \lambda f.\lambda x.\;x \\
   1 &= \lambda f.\lambda x.f\;x \\
   2 &= \lambda f.\lambda x.f (f\;x) \\
   3 &= \lambda f.\lambda x.f (f (f\;x))
   \end{aligned}
$$

   Em Haskell, teremos:

   ```haskell
   type Church a = (a -> a) -> a -> a

   zero :: Church a
   zero = \f -> \x -> x

   succ' :: Church a -> Church a
   succ' n = \f -> \x -> f (n f\;x)

   one :: Church a
   one = succ' zero

   two :: Church a
   two = succ' one
   -- Converter para Int
   toInt :: Church Int -> Int
   toInt n = n (+1) 0
   main = do
      print (toInt zero) -- Saída: 0
      print (toInt one) -- Saída: 1
      print (toInt two) -- Saída: 2
   ```

O cálculo lambda é a base teórica para as linguagens de programação que usam o paradigma da programação funcional, especialmente em Haskell. Mas, para isso, precisamos considerar os tipos.

# 10. Cálculo Lambda Simplesmente Tipado

O Cálculo Lambda Simplesmente Tipado é uma extensão do cálculo lambda não tipado que introduz uma estrutura de tipos. Este sistema aborda questões de consistência lógica encontradas no cálculo lambda não tipado, como o termo $\omega = \lambda x.\;x\;x\,$, que leva a reduções infinitas.

Uma característica do Cálculo Lambda Simplesmente Tipado é sua relação com a lógica e a computação, exemplificada pela Correspondência de Curry-Howard. Esta correspondência estabelece uma conexão entre provas matemáticas e programas de computador.

Geralmente não percebemos que, na matemática, uma a definição de uma função inclui a determinação dos tipos de dados que ela recebe e dos tipos de dados que ela devolve. Por exemplo, a função de quadrado aceita números inteiros $n$ como entradas e produz números inteiros $n^2$ como saídas. Considere uma função para determinar se um número é zero ou não, $isZero\,$, esta função aceitará números inteiros e produzirá valores booleanos como resposta. Fazemos isso, quase instintivamente, sem definir os domínios relacionados com os valores sobre os quais aplicaremos a função e com o resultado desta aplicação.

Nesta seção, examinaremos os elementos do Cálculo Lambda Simplesmente Tipado, incluindo sua sintaxe, regras de tipagem e propriedades. Exploraremos como este sistema se relaciona com o design de linguagens de programação e a verificação formal de programas. A discussão incluirá exemplos matemáticos e práticos para ilustrar como os conceitos teóricos se traduzem em construções de programação e raciocínio lógico. Mas, antes precisamos entender como chegamos até aqui.

## 10.1. A Teoria dos Tipos Simples

A **Teoria dos Tipos Simples**, desenvolvida por Alonzo Church na década de 1940, representa um marco na história da lógica matemática e da Ciência da Computação. Criada para resolver problemas de inconsistência no cálculo lambda não tipado, essa teoria introduziu um framework robusto para formalizar o raciocínio matemático e computacional, abordando paradoxos semelhantes ao **paradoxo de Russell** na teoria dos conjuntos. A Teoria dos Tipos Simples foi uma das primeiras soluções práticas para garantir que expressões lambda fossem bem formadas, evitando contradições lógicas e permitindo cálculos confiáveis.

O cálculo lambda não tipado, proposto por Church na década de 1930, ofereceu um modelo poderoso de computabilidade, mas sua flexibilidade permitiu a formulação de termos paradoxais, como o **combinador Y** (um fixpoint combinator) e o termo**$\omega = \lambda x.\;x\;x$**, que resulta em reduções infinitas. Esses termos paradoxais tornavam o cálculo lambda inconsistente, uma vez que permitiam a criação de expressões que não convergiam para uma forma normal, gerando loops infinitos.

O problema era análogo aos paradoxos que surgiram na teoria dos conjuntos ingênua, como o paradoxo de Russell. A solução proposta por Church envolvia restringir o cálculo lambda através da introdução de tipos, criando um sistema onde exclusivamente combinações de funções e argumentos compatíveis fossem permitidas, prevenindo a criação de termos paradoxais.

### 10.1.1. Fundamentos da Teoria dos Tipos Simples

A ideia central da **Teoria dos Tipos Simples** é organizar as expressões lambda em uma hierarquia de tipos que impõe restrições sobre a formação de termos. Isso garante que termos paradoxais, como $\omega\,$, sejam automaticamente excluídos. A estrutura básica da teoria é composta por:

1. **Tipos Base**: Esses são os tipos diretamente relacionados com o hardware, como $\text{Bool}$ para valores booleanos e $\text{Nat}$ para números naturais. Esses tipos representam os elementos básicos manipulados pelo sistema.

2. **Tipos de Função**: Se $A$ e $B$ são tipos, então $A \rightarrow B$ representa uma função que recebe um valor do tipo $A$ e retorna um valor do tipo $B\,$. Esta construção é crucial para definir funções no cálculo lambda tipado.

3. **Hierarquia de Tipos**: Os tipos formam uma hierarquia estrita. Tipos base estão na camada inferior, enquanto os tipos de função, que podem tomar funções como argumentos e retornar funções como resultados, estão em níveis superiores. Isso evita que funções sejam aplicadas a si mesmas de forma paradoxal, como em $\lambda x.;x;x,$.

O **sistema de tipos** no cálculo lambda tipado simples é definido por um conjunto de regras que especificam como os tipos podem ser atribuídos aos termos. Essas regras garantem que as expressões sejam consistentes e bem formadas. Estas regras são:

- **Regra da Variável**: Se uma variável $x$ tem o tipo $A$ no contexto $\Gamma\,$, então ela é bem tipada:

$$
\frac{x : A \in \Gamma}{\Gamma \vdash x : A}
$$

- **Regra da Abstração**: Se, no contexto $\Gamma\,$, assumimos que $x$ tem tipo $A$ e podemos derivar que $M$ tem tipo $B\,$, então $\lambda x : A . M$ é uma função bem tipada que mapeia de $A$ para $B$:

$$
\frac{\Gamma, x : A \vdash M : B}{\Gamma \vdash (\lambda x : A . M) : A \rightarrow B}
$$

- **Regra da Aplicação**: Se $M$ é uma função do tipo $A \rightarrow B$ e $N$ é um termo do tipo $A\,$, então a aplicação $M\;N$ resulta em um termo do tipo $B$:

$$
\frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M\;N : B}
$$

Vamos voltar as essas regras com mais cuidado no futuro, por enquanto basta entender que a **Teoria dos Tipos Simples** apresenta várias propriedades importantes que a tornam um sistema robusto para lógica e computação:

1. **Consistência**: Ao contrário do cálculo lambda não tipado, o sistema de tipos simples é consistente. Isso significa que nem todas as proposições podem ser provadas, e o sistema não permite a formação de paradoxos.

2. **Normalização Forte**: Todo termo bem tipado no cálculo lambda simples possui uma forma normal, e qualquer sequência de reduções eventualmente termina. Essa propriedade garante que os cálculos são finitos e que todos os termos se resolvem em uma forma final.

3. **Preservação de Tipos (Subject Reduction)**: Se um termo $M$ tem tipo $A$ e $M$ é reduzido para $N\,$, então $N$ terá o tipo $A\,$. Isso garante que a tipagem é preservada durante as operações de redução.

4. **Decidibilidade da Tipagem**: É possível determinar, de forma algorítmica, se um termo é bem tipado e, em caso afirmativo, qual é o seu tipo. Essa propriedade é crucial para a verificação automática de programas e provas.

A **Teoria dos Tipos Simples** influenciou diversas áreas da Ciência da Computação e da lógica matemática. Ela impactou linguagens de programação, verificação formal, semântica de linguagens e lógica computacional. Nos sistemas de tipos modernos, como os usados em linguagens funcionais como ML e Haskell, suas raízes estão diretamente ligadas à Teoria dos Tipos Simples. A tipagem estática, derivada dessa teoria, é usada para detectar erros antes da execução do programa.

Na verificação formal, a Teoria dos Tipos Simples fornece a base para sistemas de prova assistida por computador, como **Coq** e **Isabelle**, que permitem a formalização de teoremas matemáticos e sua verificação automática. A Teoria dos Tipos Simples contribui para a semântica formal das linguagens de programação, oferecendo o rigor necessário para descrever o comportamento das construções de linguagem.

A Teoria dos Tipos Simples é ligada à _Correspondência de Curry-Howard_, que estabelece uma relação entre proposições lógicas e tipos, e entre provas e programas. Esta correspondência trata da conexão entre lógica e computação, reforçando o papel dos tipos na verificação de propriedades em sistemas computacionais e matemáticos.

A **Teoria dos Tipos Simples** tem limitações. Uma delas é a expressividade limitada, pois o sistema não pode expressar diretamente conceitos como indução, usados em contextos puramente matemáticos. Outra limitação é a ausência de polimorfismo, já que não há suporte nativo para funções polimórficas, que operam de forma genérica sobre múltiplos tipos.

Para superar essas limitações, surgiram várias extensões para a Teoria dos Tipos Simples teoria. Os sistemas de tipos polimórficos, como o **Sistema F** de [Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard), introduzem quantificação sobre tipos, permitindo a definição de funções polimórficas. A teoria dos tipos dependentes foi desenvolvida permitindo que tipos dependam de valores, o que aumenta a expressividade e possibilita raciocínios mais complexos. A Teoria dos Tipos homotópica conecta a teoria dos tipos com a topologia algébrica, oferecendo novos insights sobre a matemática e a computação.

## 10.2. Estruturas de Dados e Segurança de Tipos

A presença de tipos não altera de forma alguma a avaliação de uma expressão. Usaremos os tipos para restringir quais expressões iremos avaliar. Especificamente, o sistema de tipos para o cálculo lambda simplesmente tipado assegura que qualquer programa bem tipado não correrá o risco de ficar preso em um _loop_ infinito, ou simplesmente preso.

No cálculo lambda não tipado estendido com booleanos, podemos encontrar termos bem formados que ficam _presos_ - ou seja, não são valores, mas não podem ser reduzidos. Por exemplo, considere o termo:

$\text{True}\;(\lambda x.\;x)$

Este termo é uma aplicação, então não é um valor. No entanto, não pode ser reduzido, pois nenhuma das regras de redução se aplica. Já que não é uma aplicação de abstração, então a regra $\beta$ não se aplica. Finalmente, não é uma expressão condicional (if-then-else), então as regras de redução para booleanos não se aplicam.

Outro exemplo será:

$\text{if}\;(\lambda x.\;x)\;\text{then}\;\text{True}\;\text{else}\;\text{False}$

Este termo fica _preso_ porque a condição do `if` não é um booleano, mas uma abstração. Não há regra de redução que possa ser aplicada a este termo.

Por outro lado, Um loop infinito ocorre quando um termo pode ser reduzido indefinidamente sem nunca chegar a um valor. Um exemplo clássico é o termo omega:

   $(\lambda x.\;x\;x)\;(\lambda x.\;x\;x)$

   Este termo reduz a si mesmo indefinidamente:

   $(\lambda x.\;x\;x)\;(\lambda x.\;x\;x) \to (\lambda x.\;x\;x)\;(\lambda x.\;x\;x) \to \ldots$

Em uma linguagem de programação real, estes seriam considerados erros de tipo. Por exemplo, em Haskell, tentar definir funções equivalentes resultaria em erros de compilação:

```haskell
stuck1 = True (\x -> x)
stuck2 = if (\x -> x) then True else False
```

Ou seja, o cálculo lambda não tipado é poderoso. Ele expressa todas as funções computáveis. Mas tem limites. Algumas expressões no cálculo lambda não tipado levam a paradoxos. O termo $\omega = \lambda x.\;x\;x$ aplicado a si mesmo resulta em redução infinita:

$$(\lambda x.\;x\;x) (\lambda x.\;x\;x) \to (\lambda x.\;x\;x) (\lambda x.\;x\;x) \to ...$$

A existência deste _loop_ infinito é um problema. Torna o sistema inconsistente. Podemos resolver esta inconsistência adicionando tipos aos termos. Os tipos restringem como os termos se combinam evitando paradoxos e laços infinitos. Uma vez que os tipos tenham sido acrescentados, teremos o cálculo lambda tipado.

No cálculo lambda tipado, cada termo terá um tipo e funções terão tipos no formato $A \to B\,$, Significando que recebem uma entrada do tipo $A$ e retornam uma saída do tipo $B\,$.

O termo $\omega$ não é válido no cálculo lambda tipado. O sistema de tipos o rejeita. tornando o sistema consistente, garantindo que as operações terminem, evitando a recursão infinita. Desta forma, o cálculo lambda tipado se torna a base para linguagens de programação tipadas, garantindo que os programas sejam bem comportados e terminem.

A adoção de tipos define quais dados são permitidos como argumentos e quais os tipos de resultados uma função pode gerar. Essas restrições evitam a aplicação indevida de funções a si mesmas e o uso de expressões malformadas, garantindo consistência e prevenindo paradoxos.

No sistema de tipos simples, variáveis têm tipos atribuídos, no formato $x\;:\;A\,$, onde $A$ é o tipo de $x\,$. As funções são descritas por sua capacidade de aceitar um argumento de um tipo e retornar um valor de outro tipo. Uma função que aceita uma entrada do tipo $A$ e retorna um valor do tipo $B$ é escrita como $A \rightarrow B\,$. Permitindo que funções recebam argumentos de um tipo e retornem outro tipo de dado de acordo com a necessidade.

Podemos simplificar o conceito de tipos a dois conceitos:

1. **Tipos básicos**, como $\text{Bool}$(booleanos) ou $\text{Nat}$(números naturais).

2. **Tipos de função**, como $A \rightarrow B\,$, que representam funções que mapeiam valores de $A$ para $B\,$.

Considere a expressão $\lambda x.\;x + 1\,$. No cálculo lambda tipado, essa função é válida se $x$ for de um tipo numérico, como $x : \text{Nat}\,$, neste caso, considerando $1$ com um literal natural. A função seria tipada e sua assinatura a definirá como uma função que aceita um número natural e retorna um número natural:

$$\lambda x : \text{Nat}.\;x + 1 : \text{Nat} \rightarrow \text{Nat}$$

Isso assegura que somente valores do tipo $\text{Nat}$ possam ser aplicados a essa função, evitando a aplicação incorreta de argumentos não numéricos.

Com um pouco mais de formalidade, vamos considerar um conjunto de tipos básicos. Usaremos a letra grega $\tau$(_tau_) minúscula para indicar um tipo básico. O conjunto de tipos simples é definido pela seguinte gramática BNF:

$$A,B ::= \tau \mid A \rightarrow B \mid A \times B \mid 1$$

O significado pretendido desses tipos é o seguinte: tipos base são estruturas simples como os  tipos de inteiro e booleano. O tipo $A \rightarrow B$ é o tipo de funções de $A$ para $B\,$. O tipo $A \times B$ é o tipo de tuplas $\langle x, $Y$ \rangle\,$, onde $x$ tem tipo $A$ e $y$ tem tipo $B\,$. A notação $\langle x, $Y$ \rangle$ foi introduzida para representar um par de termos $M$ e $N\,$. Permitindo que o cálculo lambda tipado manipule funções e estruturas de dados compostas.

O tipo $1$ é um tipo de um elemento literal, um tipo especial que contém exatamente um elemento, semelhante ao conceito de _tipo simples_ em algumas linguagens de programação. Isso é útil para representar valores que não carregam informação significativa, mas que precisam existir para manter a consistência do sistema de tipos.

Vamos adotar uma regra de precedência: $\times$ tem precedência sobre $\rightarrow\,$, e $\rightarrow$ associa-se à direita. Assim, $A \times B \rightarrow C$ é $(A \times B) \rightarrow C\,$, e $A \rightarrow B \rightarrow C$ é $A \rightarrow (B \rightarrow C)$[^cita7].

O conjunto de termos lambda tipados puros e brutos é definido pela seguinte BNF:

Termos brutos: $M,N ::= x \mid M N \mid \lambda x^A.M \mid \langle M,N \rangle \mid \pi_1M \mid \pi_2M \mid *$

Onde temos $x$ representando variáveis; $M N$ representando aplicação de função; $\lambda x^A.M$ representando abstração lambda com anotação de tipo; $\langle M,N \rangle$ representando tuplas; $\pi_1M$ e $\pi_2M$ representam projeções de tuplas e $*$ representando o elemento único do tipo $1\,$. Desta forma, a amável leitora notará que definimos a sintaxe básica dos termos no cálculo lambda tipado, antes de qualquer verificação de tipo ou análise semântica. Por isso, os chamamos estes tipos simples ou brutos. Estes termos são chamados de _brutos_ porque  representam a estrutura sintática pura dos termos, sem garantia de que sejam _bem tipados_.

Esta sintaxe, simples, pode incluir termos que não são válidos no sistema de tipos, mas que seguem a gramática definida. Ainda assim, usaremos esta gramática como o ponto de partida para o processo de verificação de tipos e, posteriormente, para análise semântica. Posteriormente estes termos _brutos_ serão submetidos a um conjunto de regras de tipagem que determinará se as expressões  estão bem formadas no sistema de tipos do cálculo lambda simplesmente tipado.

Diferentemente do que fizemos no cálculo lambda não tipado, vamos adicionar uma sintaxe especial para pares. Especificamente, $\langle M,N \rangle$ é um par de termos, $\pi_iM$ é uma projeção, com a intenção de que $\pi_i\langle M_1,M_2 \rangle = M_i\,$, usadas para extrair os componentes de um par. Especificamente, a intenção é que $\pi_1\langle M_1,M_2 \rangle$ resulte $M_1$ e $\pi_2\langle M_1,M_2 \rangle$ resulte em $M_2\,$. Criando uma regra que permite o acesso aos elementos individuais de um par.

Adicionamos um termo $*\,$, que é o único elemento do tipo $1\,$. Outra mudança em relação ao cálculo lambda não tipado é que agora escrevemos $\lambda x^A.M$ para uma abstração lambda para indicar que $x$ tem tipo $A\,$. No entanto, às vezes omitiremos os sobrescritos e escreveremos $\lambda x.\;M$ como antes.

Esta gramática permite que as abstrações lambda incluam anotações de tipo na forma $\lambda  x:\tau. M\,$, indicando explicitamente que a variável $x$ tem o tipo $\tau\,$. Isso permite que o sistema verifique se as aplicações de função são feitas corretamente e se os termos são bem tipados.

Embora as anotações de tipo sejam importantes, às vezes os tipos podem ser omitidos, escrevendo-se simplesmente $\lambda  x. M\,$. Isso ocorre quando o tipo de $x$ é claro a partir do contexto ou quando não há ambiguidade, facilitando a leitura e a escrita das expressões.

Em resumo as sintáticas permitem que o cálculo lambda tipado:

- **Represente Estruturas de Dados Complexas**: Com a capacidade de manipular pares e projeções, é possível representar dados mais complexos além de funções puras, aproximando o cálculo lambda das necessidades práticas de linguagens de programação.

- **Garanta a Segurança de Tipos**: As anotações de tipo em variáveis e a sintaxe enriquecida ajudam a prevenir erros, como a aplicação indevida de funções ou a formação de expressões paradoxais, assegurando que termos bem tipados sejam considerados válidos.

As noções de variáveis livres e ligadas e redução-$\alpha$ são definidas como no cálculo lambda não tipado; novamente identificamos termos $\alpha$-equivalentes.

## 10.3. Sintaxe do Cálculo Lambda Tipado

O cálculo lambda tipado estende o cálculo lambda não tipado, adicionando uma estrutura de tipos que restringe a formação e a aplicação de funções. Essa extensão preserva os princípios do cálculo lambda, mas introduz um sistema de tipos que promove maior consistência e evita paradoxos lógicos. Enquanto no cálculo lambda não tipado as funções podem ser aplicadas livremente a qualquer argumento, o cálculo lambda tipado impõe restrições que garantem que as funções sejam aplicadas a argumentos compatíveis com seu tipo.

No cálculo lambda tipado, as expressões são construídas a partir de três elementos principais: variáveis, abstrações e aplicações. Esses componentes definem a estrutura básica das funções e seus argumentos, e a adição de tipos funciona como um mecanismo de segurança, assegurando que as funções sejam aplicadas de forma correta. Uma variável $x\,$, por exemplo, é anotada com um tipo específico como $x : A\,$, onde $A$ pode ser um tipo básico como $\text{Nat}$ ou $\text{Bool}\,$, ou um tipo de função como $A \rightarrow B\,$.

### 10.3.1. Gramática e Regras de Produção

A gramática do cálculo lambda tipado pode ser definida formalmente usando a notação de Backus-Naur Form (BNF), uma forma comum de descrever regras de construção de linguagens formais, que usamos antes. Aqui está uma versão simplificada:

$$
\begin{align*}
\text{tipo} &::= \text{tipo-base} \\
&\;|\;\text{tipo} \rightarrow \text{tipo} \\
&\;|\;(\text{tipo})

\\[10pt]

\text{tipo-base} &::= \text{Nat} \\
&\;|\;\text{Bool}

\\[10pt]

\text{termo} &::= \text{variável} \\
&\;|\;\text{constante} \\
&\;|\;\lambda \text{variável} : \text{tipo}.\;\text{termo} \\
&\;|\;\text{termo}\;\text{termo} \\
&\;|\;(\text{termo})

\\[10pt]

\text{variável} &::= x\;|\;y\;|\;z\;|\;\ldots

\\[10pt]

\text{constante} &::= 0\;|\;1\;|\;2\;|\;\ldots \\
&\;|\;\text{True}\;|\;\text{False}
\end{align*}
$$

Estas regras de produção definem a estrutura sintática do cálculo lambda tipado. Elas especificam como os termos e tipos válidos podem ser construídos. Neste conjunto de regras de produção temos:

1. **Variáveis**: Representadas por letras minúsculas ($x\,$, $y\,$, $z$). Cada variável tem um tipo associado.

2. **Tipos**:
   - Tipos base: $\text{Nat}\,$, $\text{Bool}\,$, etc.
   - Tipos de função: $A \rightarrow B\,$, onde $A$ e $B$ são tipos.

3. **Abstrações**: Representadas como $\lambda x:A. t\,$, onde:
   - $x$ é a variável ligada
   - $A$ é o tipo da variável $x$
   - $t$ é o corpo da abstração (um termo)

4. **Aplicações**: Representadas como $(t_1\;t_2)\,$, onde $t_1$ e $t_2$ são termos.

5. **Parênteses**: Usados para agrupar expressões complexas e definir a precedência.

Vamos ver alguns exemplos de uso da gramática para definição de expressões em cálculo lambda tipado:

**Exemplo 1**: Construção de um tipo de função:

$$\text{Nat} \rightarrow \text{Bool}$$

   Este é um tipo válido, representando uma função que recebe um `Nat` e retorna um `Bool`.

**Exemplo 2**: Construção de uma abstração lambda:

$$\lambda x : \text{Nat}.\;x$$

   Esta é uma função de identidade, bem tipada, para números naturais, e seu resultado ao ser aplicada a um valor é o próprio valor.

**Exemplo 3**: Construção de uma aplicação:

$$(\lambda x : \text{Nat}.\;x)\;5$$

   Aqui, aplicamos a função de identidade ao número $5\,$, e o resultado da aplicação é $5\,$.

**Exemplo 4**: Construção de um termo mais complexo:

$$(\lambda f : (\text{Nat} \rightarrow \text{Nat}). \lambda x : \text{Nat}. f\;(f\;x))$$

   Este termo representa uma função que recebe uma função $f$ de $Nat$ para $Nat$ e retorna uma nova função que aplica $f$ duas vezes ao argumento $x\,$.

Além da construção de funções e abstrações tipadas, o básico para a criação de expressões no cálculo lambda tipado, a gramática pode ser usada para validar expressões. Vamos fazer uma derivação, para validar a expressão lambda tipada:

$$(\lambda x : \text{Nat}. \lambda $Y$ : \text{Bool}.\;x)\;3\;\text{True}$$

   1. Começamos com o termo completo:

      $$\text{termo} \rightarrow \text{termo}\;\text{termo}$$

   2. Expandimos o primeiro termo:

      $$(\lambda x : \text{Nat}. \lambda $Y$ : \text{Bool}.\;x)\;3\;\text{True}$$

      $$\text{termo} \rightarrow (\text{termo})\;\text{termo}\;\text{termo}$$

   3. Dentro dos parênteses, temos uma abstração:

      $$\text{termo} \rightarrow (\lambda \text{variável} : \text{tipo}.\;\text{termo})\;\text{termo}\;\text{termo}$$

   4. Expandimos o corpo da primeira abstração:

      $$\text{termo} \rightarrow (\lambda x : \text{Nat}. \lambda \text{variável} : \text{tipo}.\;\text{termo})\;\text

### 7.3.2. Semântica Estática (Sistema de Tipos)

A semântica do cálculo lambda tipado define o significado das expressões e como elas são avaliadas. Ela consiste em duas partes principais: a semântica estática (sistema de tipos) e a semântica dinâmica (regras de redução).

O sistema de tipos do cálculo lambda tipado é responsável por atribuir tipos às expressões e garantir que exclusivamente expressões bem tipadas sejam aceitas. Já passamos por estas regras antes. Contudo, para manter o contexto vamos a última vez:

As regras de tipagem no cálculo lambda tipado são geralmente expressas através da inferência natural. Abaixo, estas regras são detalhadas, sempre partindo de premissas em direção a conclusão.

#### 7.3.2.1. Regra da Variável

A regra da variável afirma que, se uma variável $x$ tem tipo $A$ no contexto $\Gamma\,$, então podemos derivar que $x$ tem tipo $A$ nesse contexto:

$$\frac{x : A \in \Gamma}{\Gamma \vdash x : A}$$

Essa regra formaliza a ideia de que, se sabemos que $x$ tem tipo $A$ a partir do contexto, então $x$ pode ser usada em expressões como um termo de tipo $A\,$.

#### 10.3.2.2. Regra de Abstração

A regra de abstração define o tipo de uma função. Se, assumindo que $x$ tem tipo $A\,$, podemos derivar que $M$ tem tipo $B\,$, então a abstração $\lambda x : A . M$ tem o tipo $A \rightarrow B\,$. Formalmente:

$$\frac{\Gamma, x : A \vdash M : B}{\Gamma \vdash (\lambda x : A . M) : A \rightarrow B}$$

Essa regra assegura que a função $\lambda x : A . M$ é corretamente formada e mapeia valores do tipo $A$ para resultados do tipo $B\,$.

#### 10.3.2.3. Regra de Aplicação

A regra de aplicação governa a forma como funções são aplicadas a seus argumentos. Se $M$ é uma função do tipo $A \rightarrow B$ e $N$ é um termo do tipo $A\,$, então a aplicação $M\;N$ tem tipo $B$:

$$\frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M\;N : B}$$

Essa regra garante que, ao aplicar uma função $M$ a um argumento $N\,$, a aplicação resulta em um termo do tipo esperado $B\,$.

Em Haskell, a aplicação de função é direta e o sistema de tipos verifica automaticamente a compatibilidade:

```haskell
increment :: Int -> Int
increment x = x + 1

result :: Int
result = increment 5  -- Retorna 6
```

Neste exemplo, `increment` tem tipo `Int -> Int` (equivalente a $A \rightarrow B$), e `5` tem tipo `Int` (equivalente a $A$). A aplicação `increment 5` resulta em um `Int` (equivalente a $B$), demonstrando a regra de aplicação na prática.

Se tentássemos aplicar `increment` a um argumento de tipo incompatível, como em `increment True`, obteríamos um erro de tipo, ilustrando como o sistema de tipos em Haskell implementa a segurança garantida pela regra de aplicação do cálculo lambda tipado.

#### 7.3.2.4. Semântica Dinâmica (Regras de Redução)

A semântica dinâmica define como as expressões são avaliadas. O principal mecanismo de avaliação é a redução beta:

1. **redução-$beta$**:

$$(\lambda x:A. t)\;s \rightarrow t[x := s]$$

   Onde $t[x := s]$ denota a substituição de todas as ocorrências livres de $x$ em $t$ por $s\,$. Isso corresponde à aplicação de uma função ao seu argumento.

   Exemplo:
$$(\lambda x:\text{Nat}.\;x + 1)\;5 \rightarrow 5 + 1 \rightarrow 6$$

2. **redução-$\eta$** (uma forma de extensionalidade):

$$\lambda x:A. (f\;x) \rightarrow f$$

   Se $x$ não ocorre livre em $f\,$, a redução-$\eta$ indica que uma função $\lambda x:A. (f\;x)$ é equivalente a $f\,$, refletindo o princípio de que duas funções que produzem os mesmos resultados para todos os argumentos são indistinguíveis.

A aplicação das regras de tipagem, semântica estática, e das reduções, semântica dinâmica leva ao surgimento de um conjunto de propriedades semânticas:

1. **Preservação de Tipos** (Subject Reduction): Se $\Gamma \vdash t : A$ e $t \rightarrow t'\,$, então $\Gamma \vdash t' : A\,$.

   Esta propriedade garante que a redução preserva os tipos, assegurando que a avaliação de um termo bem tipado sempre resulta em um termo do mesmo tipo. Considere o seguinte termo no cálculo lambda simplesmente tipado:

$$(\lambda x: \text{Nat}.\;x + 1)\;3$$

   Aqui, temos uma função que incrementa um número natural ($x + 1$) e a aplicamos ao número $3\,$. A tipagem desse termo pode ser verificada: o termo $\lambda x: \text{Nat}.\;x + 1$ tem tipo $\text{Nat} \rightarrow \text{Nat}\,$, pois é uma função que recebe um número natural e retorna outro número natural; o número $3$ tem o tipo $\text{Nat}\,$. Agora, aplicamos a **regra de aplicação**:

$$
   \frac{\Gamma \vdash (\lambda x: \text{Nat}.\;x + 1) : \text{Nat} \rightarrow \text{Nat} \quad \Gamma \vdash 3 : \text{Nat}}{\Gamma \vdash (\lambda x: \text{Nat}.\;x + 1)\;3 : \text{Nat}}
$$

   Após a aplicação, o termo é reduzido usando a **redução beta**:

$$(\lambda x: \text{Nat}.\;x + 1)\;3 \rightarrow 3 + 1 \rightarrow 4$$

   Como resultado, o termo final é $4\,$, que tem tipo $\text{Nat}\,$. A preservação de tipos garante que, ao longo da redução, o tipo do termo permaneceu como $\text{Nat}\,$.

   Em Haskell, a preservação de tipos é garantida pelo sistema de tipos estático e pelo compilador. Considere o seguinte exemplo:

   ```haskell
   data Bool = True | False

   not :: Bool -> Bool
   not True = False
   not False = True

   alwaysBool :: Bool
   alwaysBool = not (not True)
   ```

   Neste exemplo, a função `not` tem o tipo `Bool -> Bool`, o que corresponde a $\text{Bool} \rightarrow \text{Bool}$ no cálculo lambda tipado. O compilador Haskell garante que: `not True` tem tipo `Bool` e  `not (not True)` tem tipo `Bool`

   Assim, a expressão `alwaysBool` é garantida pelo sistema de tipos a sempre retornar um valor do tipo `Bool`, independentemente das reduções intermediárias. Isso ilustra a preservação de tipos em ação:

$$\frac{\Gamma \vdash \text{not} : \text{Bool} \rightarrow \text{Bool} \quad \Gamma \vdash \text{True} : \text{Bool}}{\Gamma \vdash \text{not True} : \text{Bool}}$$

   E subsequentemente:

$$\frac{\Gamma \vdash \text{not} : \text{Bool} \rightarrow \text{Bool} \quad \Gamma \vdash \text{not True} : \text{Bool}}{\Gamma \vdash \text{not (not True)} : \text{Bool}}$$

   O sistema de tipos do Haskell assegura que todas as operações preservam os tipos esperados, refletindo a propriedade de preservação de tipos do cálculo lambda tipado.

2. **Normalização Forte**: Todo termo bem tipado em certos sistemas de tipos, como o cálculo lambda simplesmente tipado, tem uma sequência finita de reduções que leva a uma forma normal (um termo que não pode ser mais reduzido). Considere o seguinte termo:

$$
   (\lambda f: \text{Nat} \rightarrow \text{Nat}. \lambda x: \text{Nat}. f (f\;x))\;(\lambda y: \text{Nat}. $Y$ + 1)\;0
$$

   Este termo descreve uma função que aplica outra função $f$ duas vezes a um argumento $x\,$. Aplicamos essa função à função que incrementa $y$ e ao valor $0\,$. Vamos ver como o termo se reduz,

   Primeiro, aplicamos:

  $$\lambda f: \text{Nat} \rightarrow \text{Nat}. \lambda x: \text{Nat}. f (f\;x)$ à função $\lambda y: \text{Nat}. $Y$ + 1$$

  $$(\lambda f: \text{Nat} \rightarrow \text{Nat}. \lambda x: \text{Nat}. f (f\;x))\;(\lambda y: \text{Nat}. $Y$ + 1)
      \rightarrow \lambda x: \text{Nat}. (\lambda y: \text{Nat}. $Y$ + 1) ((\lambda y: \text{Nat}. $Y$ + 1)\;x)$$

      Agora, aplicamos essa função ao valor $0$:

  $$(\lambda x: \text{Nat}. (\lambda y: \text{Nat}. $Y$ + 1) ((\lambda y: \text{Nat}. $Y$ + 1)\;x))\;0
      \rightarrow (\lambda y: \text{Nat}. $Y$ + 1) ((\lambda y: \text{Nat}. $Y$ + 1)\;0)$$

      Avaliando a primeira aplicação:

  $$(\lambda y: \text{Nat}. $Y$ + 1)\;0 \rightarrow 0 + 1 \rightarrow 1$$

      Avaliando a segunda aplicação:

  $$(\lambda y: \text{Nat}. $Y$ + 1)\;1 \rightarrow 1 + 1 \rightarrow 2$$

      O termo foi completamente reduzido para $2\,$, e não há mais reduções possíveis. Esse é o estado irreduzível ou a _forma normal_ do termo. A _normalização forte_ garante que, neste sistema de tipos, qualquer termo bem tipado eventualmente chegará a uma forma normal, sem laços infinitos.

3. **Church-Rosser** (Confluência): Se um termo pode ser reduzido de duas formas diferentes, então existe uma forma comum que ambas as reduções eventualmente alcançarão. Isso garante que a ordem de avaliação não afeta o resultado . Para entender, considere o seguinte termo lambda tipado:

$$(\lambda x:\text{Nat}. \lambda y:\text{Nat}.\;x + y)\;3\;((\lambda z:\text{Nat}.\;z * 2)\;2)$$

Este termo pode ser reduzido por dois caminhos:

1. Reduzindo a aplicação externa primeiro:

$$(\lambda x:\text{Nat}. \lambda y:\text{Nat}.\;x + y)\;3\;((\lambda z:\text{Nat}.\;z * 2)\;2)$$

$$\rightarrow (\lambda y:\text{Nat}. 3 + y)\;((\lambda z:\text{Nat}.\;z * 2)\;2)$$

$$\rightarrow 3 + ((\lambda z:\text{Nat}.\;z * 2)\;2)$$

$$\rightarrow 3 + (2 * 2)$$

$$\rightarrow 3 + 4$$

$$\rightarrow 7$$

2. Reduzindo a aplicação interna primeiro:

$$(\lambda x:\text{Nat}. \lambda y:\text{Nat}.\;x + y)\;3\;((\lambda z:\text{Nat}.\;z * 2)\;2)$$

$$\rightarrow (\lambda x:\text{Nat}. \lambda y:\text{Nat}.\;x + y)\;3\;(2 * 2)$$

$$\rightarrow (\lambda x:\text{Nat}. \lambda y:\text{Nat}.\;x + y)\;3\;4$$

$$\rightarrow (\lambda y:\text{Nat}. 3 + y)\;4$$

$$\rightarrow 3 + 4$$

$$\rightarrow 7$$

Observe que, independentemente da ordem em que as reduções são aplicadas, chegamos ao mesmo resultado : $7\,$. Esta propriedade atesta que a ordem de avaliação em um programa lambda tipado não afeta o resultado , desde que o programa termine. Esta propriedade implica na confiabilidade e previsibilidade dos sistemas baseados no cálculo lambda tipado, como muitas linguagens de programação funcional.

A semântica do cálculo lambda tipado tem implicações para a teoria da computação e o design de linguagens de programação. Ela promove a segurança de tipos, assegurando que programas bem tipados não causam erros de tipo durante a execução. Por exemplo, em uma linguagem com tipagem estática baseada no cálculo lambda tipado, uma expressão como `1 + True` seria rejeitada em tempo de compilação, evitando erros em tempo de execução. Esse cálculo serve como base para linguagens funcionais tipadas, como Haskell e ML, que herdaram suas propriedades formais. A correspondência de Curry-Howard une programas e provas matemáticas, onde tipos correspondem a proposições e termos a provas, unificando lógica e computação. O cálculo lambda tipado oferece uma base sólida para sistemas de verificação formal de programas, permitindo provas rigorosas de correção. Na prática, isso permite a criação de software crítico com alto grau de confiabilidade, como em sistemas de controle de voo ou protocolos de segurança criptográfica.

Nos sistemas onde a propriedade de Church-Rosser, confluência, é válida, o cálculo lambda assegura que o resultado  de um programa seja determinístico, independentemente da estratégia de avaliação. Isso permite que compiladores realizem otimizações, já que a ordem de avaliação não altera o resultado . Por exemplo, em uma expressão como $(\lambda x.\;x + 1)\;(2 + 3)\,$, o compilador pode escolher avaliar $(2 + 3)$ primeiro ou aplicar a função $\lambda x.\;x + 1$ diretamente a $(2 + 3)\,$, sabendo que o resultado é o mesmo. Em sistemas confluentes, o raciocínio equacional se torna uma ferramenta útil, permitindo equivalências entre expressões e facilitando provas e transformações de programas.

A propriedade de Church-Rosser varia conforme o sistema de cálculo lambda tipado. No _cálculo lambda simplesmente tipado_, a confluência é garantida, pois o sistema se baseia em tipos básicos e tipos de função, sem construções que poderiam causar ambiguidades na redução. Em sistemas com tipos dependentes, como o Cálculo das Construções, a propriedade tende a se manter, embora a prova de confluência seja mais complexa devido à interação entre termos e tipos. Em sistemas que modelam computações com estado ou efeitos colaterais, a confluência pode ser perdida, pois a mutação de estado pode fazer com que a ordem das operações altere o resultado . Nos sistemas com recursão irrestrita, a confluência permanece, mas a normalização forte é comprometida, pois nem todos os termos têm uma forma normal. Entretanto, os diferentes caminhos de redução podem ainda convergir, se a convergência for possível. Extensões que incluem operadores adicionais, como paralelismo ou concorrência, podem comprometer a confluência, pois nesses casos a ordem de avaliação pode impactar o resultado .

A semântica do cálculo lambda tipado estabelece uma ligação entre lógica, teoria dos tipos e programação, influenciando o design de linguagens modernas e técnicas de verificação formal. **Embora a propriedade de Church-Rosser seja desejável, ela não é universal em todos os sistemas de cálculo lambda tipado**. No contexto de linguagens de programação, há um equilíbrio entre garantias teóricas, como a confluência, e a necessidade de maior expressividade. Isso exige uma avaliação cuidadosa das características de cada sistema ao considerar suas propriedades de redução e avaliação, já que muitas linguagens do mundo real incluem características que podem violar a confluência, como efeitos colaterais e paralelismo.

A semântica do cálculo lambda tipado tem implicações profundas: na **segurança de Tipos**, assegurando que programas bem tipados não causarão erros de tipo durante a execução; servindo de **Base para Linguagens Funcionais**. Muitas linguagens funcionais tipadas, como Haskell e ML, são baseadas no cálculo lambda tipado. A **correspondência de Curry-Howard** estabelece uma conexão entre programas e provas matemáticas, onde tipos correspondem a proposições e termos a provas. Finalmente, a **verificação Formal** fornece uma base para o desenvolvimento de sistemas de verificação formal de programas.

A semântica do cálculo lambda tipado, portanto, define o comportamento de programas, estabelecendo uma ponte entre lógica, teoria dos tipos e programação, influenciando profundamente o design de linguagens de programação modernas e técnicas de verificação formal.

### 10.3.3. Abstrações Lambda e Tipos

No cálculo lambda tipado, as abstrações são expressas na forma $\lambda x : A.\;E\,$, onde $x$ é uma variável de tipo $A$ e $E$ é a expressão cujo resultado dependerá de $x\,$. O tipo dessa abstração é dado por $A \rightarrow B\,$, onde $B$ é o tipo do resultado de $E\,$. Por exemplo, a abstração $\lambda x : \text{Nat}.\;x + 1$ define uma função que aceita um argumento do tipo $\text{Nat}$(número natural) e retorna outro número natural. Nesse caso, o tipo da abstração é $\text{Nat} \rightarrow \text{Nat}\,$, o que significa que a função mapeia um número natural para outro número natural.

$$\lambda x : \text{Nat}.\;x + 1 : \text{Nat} \rightarrow \text{Nat}$$

As variáveis no cálculo lambda tipado podem ser livres ou ligadas. Variáveis livres são aquelas que não estão associadas a um valor específico dentro do escopo da função, enquanto variáveis ligadas são aquelas definidas no escopo da abstração. Esse conceito de variáveis livres e ligadas é familiar na lógica de primeira ordem e tem grande importância na estruturação das expressões lambda.

### 10.3.4. Aplicações de Funções

A aplicação de funções segue a mesma sintaxe do cálculo lambda não tipado, mas no cálculo tipado é restrita pelos tipos dos termos envolvidos. Se uma função $f$ tem o tipo $A \rightarrow B\,$, então ela só pode ser aplicada a um termo $x$ do tipo $A\,$. A aplicação de $f$ a $x$ resulta em um termo do tipo $B\,$. Um exemplo simples seria a aplicação da função de incremento $\lambda x : \text{Nat}.\;x + 1$ ao número 2:

$$(\lambda x : \text{Nat}.\;x + 1)\;2 \rightarrow 3$$

Aqui, a função de tipo $\text{Nat} \rightarrow \text{Nat}$ é aplicada ao número $2\,$, e o resultado é o número $3\,$, que é do tipo $\text{Nat}\,$.

### 10.3.5. Substituição e Redução

A operação de substituição no cálculo lambda tipado segue o mesmo princípio do cálculo não tipado, com a adição de restrições de tipo. Quando uma função é aplicada a um argumento, a variável vinculada à função é substituída pelo valor do argumento na expressão. Formalmente, a substituição de $N$ pela variável $x$ em $E$ é denotada por $[N/x]E\,$, indicando que todas as ocorrências livres de $x$ em $E$ devem ser substituídas por $N\,$.

A redução no cálculo lambda tipado segue a estratégia de redução-$beta\,$, onde aplicamos a função ao seu argumento e substituímos a variável ligada pelo valor fornecido. Um exemplo clássico de redução-$beta$ seria:

$$(\lambda x : \text{Nat}.\;x + 1)\;2 \rightarrow 2 + 1 \rightarrow 3$$

Esse processo de substituição e simplificação é a forma de computação de expressões no cálculo lambda tipado, e é usado na avaliação de programas em linguagens de programação funcionais.

## 10.4. Regras de Tipagem

Antes de apresentarmos as regras formais do cálculo lambda tipado, é importante entender como chegamos a este sistema de tipos. O desenvolvimento do sistema de tipos foi um processo gradual, partindo de ideias simples e evoluindo para um sistema mais expressivo.

O sistema de tipos do cálculo lambda tipado evoluiu gradualmente a partir de ideias mais simples. Inicialmente, poderíamos considerar um sistema ingênuo com unicamente dois tipos: $\text{bool}$ para valores booleanos e $\to$ para funções. Neste sistema primitivo, $T := \text{bool} \mid \to\,$, qualquer função seria simplesmente representada pelo tipo $\to\,$.

Este sistema é excessivamente simplista. Considere as funções $\lambda x. \text{True}$ e $\lambda x. \lambda y. \text{False}\,$. Ambas teriam o tipo $\to\,$, apesar de serem diferentes - a primeira retorna imediatamente um booleano, enquanto a segunda retorna outra função.

Para resolver essa limitação, refinamos nossa ideia de tipos de função. Em vez de um tipo genérico $\to\,$, introduzimos tipos de função da forma $T_1 \to T_2\,$, onde $T_1$ é o tipo do input e $T_2$ é o tipo do output. Nossa definição de tipos agora se torna recursiva: $T := \text{bool} \mid T \to T\,$.

Esta definição recursiva nos permite construir tipos mais complexos. Por exemplo, $(\text{bool} \to \text{bool}) \to \text{bool}$ representa uma função que aceita outra função (que mapeia booleanos para booleanos) e retorna um booleano.

Com este sistema refinado, podemos diferenciar nossas funções anteriores: $\lambda x. \text{True}$ teria o tipo $T \to \text{bool}$ para qualquer tipo $T\,$, enquanto $\lambda x. \lambda y. \text{False}$ teria o tipo $T_1 \to (T_2 \to \text{bool})$ para quaisquer tipos $T_1$ e $T_2\,$.

Este desenvolvimento nos leva a um sistema de tipos mais expressivo, capaz de capturar nuances importantes sobre o comportamento das funções. No entanto, ainda existem limitações. Por exemplo, não podemos expressar funções polimórficas como a função identidade $\lambda x.\;x\,$, que deve funcionar para qualquer tipo. Estas limitações motivarão desenvolvimentos futuros, como o polimorfismo paramétrico, que estudaremos mais adiante.

As regras de tipagem no cálculo lambda tipado fornecem um sistema formal para garantir que as expressões sejam bem formadas. As principais regras são:

1. **Regra da Variável**: Se uma variável $x$ possui o tipo $\tau$ no contexto $\Gamma\,$, então podemos derivar que $\Gamma \vdash x : A\,$.

   O contexto de tipagem, denotado por `$\Gamma $`, é um conjunto de associações entre variáveis e seus tipos. Formalmente temos:

$$
   \frac{}{\;\Gamma \vdash x : A} \quad \text{se } (x : \tau) \in \Gamma
$$

   Isso significa que, se a variável $x$ tem o tipo $\Tau$ no contexto $\Gamma\,$, então podemos derivar que $\Gamma \vdash x : \tau\,$. Uma variável é bem tipada se seu tipo está definido em determinado contexto.

   - **Contexto de Tipagem ($\Gamma$)**: É um conjunto de pares $(x : \tau)$ que associa as variáveis aos seus respectivos tipos. Por exemplo, $\Gamma = \{ x : \text{Int},\;y : \text{Bool} \}\,$.

   - **Julgamento de Tipagem (`$\Gamma \vdash x : \tau$)**: Lê-se _sob o contexto $\Gamma\,$, a variável $x$ tem tipo $\tau$_.

   Considere o contexto:

$$
   \Gamma = \{ x : \text{Nat},\;y : \text{Bool} \}
$$

   Aplicando a Regra da Variável: Como $(x : \text{Nat}) \in \Gamma$`, podemos afirmar que:

$$
   \Gamma \vdash x : \text{Nat}
$$

   Similarmente, como `$(y : \text{Bool}) \in \Gamma$:

$$
   \Gamma \vdash $Y$ : \text{Bool}
$$

   Isso mostra que, dentro do contexto $\Gamma\,$, as variáveis $x$ e $y$ têm os tipos $\text{Nat}$ e $\text{Bool}\,$, respectivamente.

   A Regra da Variável fundamenta a tipagem sendo a base para atribuição de tipos a expressões mais complexas. Sem essa regra, não seria possível inferir os tipos das variáveis em expressões. Essa regra garante a consistência do sistema  asseguramos que as variáveis são usadas de acordo com seus tipos declarados, evitamos erros de tipagem e comportamentos imprevistos.

2. **Regra de Abstração**: Se sob o contexto $\Gamma\,$, temos que $\Gamma, x:\tau \vdash E:B\,$, então podemos derivar que $\Gamma \vdash (\lambda x:A.E) : A \rightarrow B\,$.

   A **Regra de Abstração** no cálculo lambda tipado permite derivar o tipo de uma função lambda baseada no tipo de seu corpo e no tipo de seu parâmetro. Formalmente, a regra é expressa como:

$$\frac{\Gamma,\;x:\tau\;\vdash\;E:B}{\;\Gamma\;\vdash\;(\lambda x:\tau.\;E) : A \rightarrow B}$$

   Isso significa que, se no contexto $\Gamma\,$, ao adicionar a associação $x:\tau\,$, podemos derivar que $E$ tem tipo $B\,$, então podemos concluir que a abstração lambda $(\lambda x:\tau.\;E)$ tem tipo $A \rightarrow B$ no contexto original $\Gamma\,$.

   A linguagem Haskell permite definir funções anônimas, chamadas de funções lambda, de forma similar ao cálculo lambda simplesmente tipado:

   ```haskell
   multiplyBy2 :: Int -> Int
   multiplyBy2 = \x -> x * 2
   ```

   Esta definição em Haskell é equivalente a $\lambda x:\text{Int}.\;x * 2$ no cálculo lambda tipado. O tipo Int -> Int corresponde a $A \rightarrow B\,$, onde tanto $A$ quanto $B$ são Int. O sistema de tipos do Haskell infere automaticamente que x é do tipo Int baseado no contexto da multiplicação.
   Podemos usar esta função assim:

   ```haskell
   result :: Int
   result = multiplyBy2 3  -- Retorna 6
   ```

   Este exemplo demonstra como a abstração lambda do cálculo tipado se traduz diretamente para uma linguagem de programação funcional moderna.

   Novamente temos o contexto de tipagem $\Gamma$ indicando Conjunto de associações entre variáveis e seus tipos. O julgamento é feito por _sob o contexto $\Gamma\,$, a expressão $M$ tem tipo $B$_. Finalmente, existe uma adição ao contexto definida ao considerar a variável $x$ com tipo $\tau\,$, expandimos o contexto para $\Gamma,\;x:\tau\,$.

   A Regra de Abstração define Tipos de Funções: Permite derivar o tipo de uma função lambda a partir dos tipos de seu parâmetro e de seu corpo, enquanto parece assegurar a coerência por garantir que a função está bem tipada e que pode ser aplicada a argumentos do tipo correto.

3. **Regra de Aplicação**: Se $\Gamma \vdash M : \tau \rightarrow B$ e $\Gamma \vdash N : \tau\,$, então podemos derivar que $\Gamma \vdash (M\;N) : B\,$.

   A **Regra de Aplicação** no cálculo lambda tipado permite determinar o tipo de uma aplicação de função com base nos tipos da função e do argumento. Formalmente, a regra é expressa como:

$$
   \frac{\Gamma\;\vdash\;M : \tau \rightarrow B \quad \Gamma\;\vdash\;N : \tau}{\;\Gamma\;\vdash\;(M\;N) : B}
$$

   Isso significa que, se no contexto $\Gamma$ podemos derivar que $M$ tem tipo $A \rightarrow B$ e que $N$ tem tipo $A\,$, então podemos concluir que a aplicação $(M\;N)$ tem tipo $B$ no contexto $\Gamma\,$.

   Em Haskell, a aplicação de função é direta e o sistema de tipos verifica automaticamente a compatibilidade:

   ```haskell
   increment :: Int -> Int
   increment x = x + 1

   result :: Int
   result = increment 5  -- Retorna 6
   ```

   Neste exemplo, increment tem tipo `Int -> Int` (equivalente a $A \rightarrow B$), e 5 tem tipo Int (equivalente a $A$). A aplicação increment $5$ resulta em um `Int` (equivalente a $B$), demonstrando a regra de aplicação na prática.

   Analisando temos, novamente, o contexto de tipagem $\Gamma$), 0 julgamentos de Tipagem $\Gamma\;\vdash\;M : \tau \rightarrow B$: A expressão $M$ é uma função que leva um argumento do tipo $\tau$ e retorna um resultado do tipo $B\,$. Finalmente $\Gamma\;\vdash\;N : \tau$: A expressão $N$ é um argumento do tipo $\tau\,$.
   Ou seja, $\Gamma\;\vdash\;(M\;N) : B$: A aplicação da função $M$ ao argumento $N$ resulta em um termo do tipo $B\,$.

   Esta regra Permite Compor funções e argumentos determinando como funções tipadas podem ser aplicadas a argumentos tipados para produzir resultados tipados. Mais que isso, mostra que as funções são aplicadas a argumentos do tipo correto, evitando erros de tipagem. Esta regra estabelece que, se temos uma função que espera um argumento de um certo tipo e temos um argumento desse tipo, então a aplicação da função ao argumento é bem tipada e seu tipo é o tipo de retorno da função, garantindo a segurança e a coerência do sistema de tipos.

Essas regras fornecem a base para a derivação de tipos em expressões complexas no cálculo lambda tipado, garantindo que cada parte da expressão esteja correta e que a aplicação de funções seja válida.

### 10.4.1. Exemplos das regras de tipagem

**Exemplo 1**: Regra da Variável

   Considere o contexto:

$$\Gamma = \{ x : \text{Nat},\;y : \text{Bool} \}$$

   Aplicando a Regra da Variável teremos:

   Como $(x : \text{Nat}) \in \Gamma\,$, então:

$$\Gamma \vdash x : \text{Nat}$$

   Como $(y : \text{Bool}) \in \Gamma\,$, então:

$$\Gamma \vdash $Y$ : \text{Bool}$$

**Exemplo**: Regra de Abstração

   Considere a função:

$$\lambda x:\text{Nat}.\;x + 1$$

   Aplicação da regra:

   No contexto $\Gamma$ estendido com $x:\text{Nat}$:

$$\Gamma,\;x:\text{Nat} \vdash x + 1 : \text{Nat}$$

   Aplicando a Regra de Abstração:

$$\frac{\Gamma,\;x:\text{Nat} \vdash x + 1 : \text{Nat}}{\;\Gamma \vdash (\lambda x:\text{Nat}.\;x + 1) : \text{Nat} \rightarrow \text{Nat}}$$

**Exemplo**: Regra de Aplicação

   Considere $M = \lambda x:\text{Nat}.\;x + 1$ e $N = 5\,$.

   Tipagem da função $M$:

$$\Gamma \vdash M : \text{Nat} \rightarrow \text{Nat}$$

   Tipagem do argumento $N$:

$$\Gamma \vdash 5 : \text{Nat}$$

   Aplicando a Regra de Aplicação:

$$\frac{\Gamma\;\vdash\;M : \text{Nat} \rightarrow \text{Nat} \quad \Gamma\;\vdash\;5 : \text{Nat}}{\;\Gamma\;\vdash\;M\;5 : \text{Nat}}$$

### 10.4.2. Exercícios Regras de Tipagem no Cálculo Lambda

**1**: Dado o contexto:

$$\Gamma = \{ z : \text{Bool} \}$$

Use a Regra da Variável para derivar o tipo de $z$ no contexto $\Gamma\,$.

   **Solução**: dela Regra da Variável:

   $$\frac{z : \text{Bool} \in \Gamma}{\Gamma \vdash z : \text{Bool}}$$

   Portanto, no contexto $\Gamma\,$, $z$ tem tipo $\text{Bool}\,$.

**2**: Considere a função:

$$\lambda y:\text{Nat}.\;y \times 2$$

Usando a Regra de Abstração, mostre que esta função tem o tipo $\text{Nat} \rightarrow \text{Nat}\,$.

   **Solução**: sabemos que $y \times 2$ é uma operação que, dado $y$ de tipo $\text{Nat}\,$, retorna um $\text{Nat}\,$.

   Aplicando a Regra de Abstração:

   $$\frac{\Gamma, y:\text{Nat} \vdash $Y$ \times 2 : \text{Nat}}{\Gamma \vdash \lambda y:\text{Nat}.\;y \times 2 : \text{Nat} \rightarrow \text{Nat}}$$

   Portanto, a função tem tipo $\text{Nat} \rightarrow \text{Nat}\,$.

**3**. No contexto vazio $\Gamma = \{\}\,$, determine se a seguinte aplicação é bem tipada usando a Regra de Aplicação:

$$(\lambda x:\text{Bool}.\;x)\;\text{True}$$

   **Solução**: aplicando a Regra de Aplicação:

   1. $\Gamma \vdash \lambda x:\text{Bool}.\;x : \text{Bool} \rightarrow \text{Bool}\,$.

   2. $\Gamma \vdash \text{True} : \text{Bool}\,$.

   3. Como os tipos correspondem, podemos concluir:

   $$
   \frac{\Gamma \vdash \lambda x:\text{Bool}.\;x : \text{Bool} \rightarrow \text{Bool} \quad \Gamma \vdash \text{True} : \text{Bool}}{\Gamma \vdash (\lambda x:\text{Bool}.\;x)\;\text{True} : \text{Bool}}
   $$

   A aplicação é bem tipada e tem tipo $\text{Bool}\,$.

**4**: Dado o contexto:

   $$\Gamma = \{ f : \text{Nat} \rightarrow \text{Nat},\;n : \text{Nat} \}$$

Use a Regra de Aplicação para mostrar que $f\;n$ tem tipo $\text{Nat}\,$.

   **Solução**: aplicando a **Regra de Aplicação**:

   1. Do contexto, $\Gamma \vdash f : \text{Nat} \rightarrow \text{Nat}\,$.

   2. Do contexto, $\Gamma \vdash n : \text{Nat}\,$.

   3. Portanto:

   $$\frac{\Gamma \vdash f : \text{Nat} \rightarrow \text{Nat} \quad \Gamma \vdash n : \text{Nat}}{\Gamma \vdash f\;n : \text{Nat}}$$

   Assim, $f\;n$ tem tipo $\text{Nat}\,$.

**5**: Usando as regras de tipagem, determine o tipo da expressão:

   $$\lambda f:\text{Nat} \rightarrow \text{Bool}.\;\lambda n:\text{Nat}.\;f\;n$$

   **Solução**: Queremos encontrar o tipo da função $\lambda f:\text{Nat} \rightarrow \text{Bool}.\;\lambda n:\text{Nat}.\;f\;n\,$.

   1. No contexto $\Gamma\,$, adicionamos $f:\text{Nat} \rightarrow \text{Bool}\,$.

   2. Dentro da função, adicionamos $n:\text{Nat}\,$.

   3. Sabemos que $\Gamma, f:\text{Nat} \rightarrow \text{Bool}, n:\text{Nat} \vdash f\;n : \text{Bool}\,$.

   4. Aplicando a Regra de Abstração para $n$:

   $$\frac{\Gamma, f:\text{Nat} \rightarrow \text{Bool}, n:\text{Nat} \vdash f\;n : \text{Bool}}{\Gamma, f:\text{Nat} \rightarrow \text{Bool} \vdash \lambda n:\text{Nat}.\;f\;n : \text{Nat} \rightarrow \text{Bool}}$$

   5. Aplicando a Regra de Abstração para $f$:

   $$\frac{\Gamma \vdash \lambda n:\text{Nat}.\;f\;n : \text{Nat} \rightarrow \text{Bool}}{\Gamma \vdash \lambda f:\text{Nat} \rightarrow \text{Bool}.\;\lambda n:\text{Nat}.\;f\;n : (\text{Nat} \rightarrow \text{Bool}) \rightarrow (\text{Nat} \rightarrow \text{Bool})}$$

   Portanto, o tipo da expressão é $(\text{Nat} \rightarrow \text{Bool}) \rightarrow (\text{Nat} \rightarrow \text{Bool})\,$.

**6**: No contexto:

   $$\Gamma = \{ x : \text{Nat} \times \text{Bool} \}$$

 Utilize a Regra da Variável para derivar o tipo de $x$ em $\Gamma\,$.

   **Solução**: pela Regra da Variável:

   $$\frac{x : \text{Nat} \times \text{Bool} \in \Gamma}{\Gamma \vdash x : \text{Nat} \times \text{Bool}}$$

   Portanto, $x$ tem tipo $\text{Nat} \times \text{Bool}$ no contexto $\Gamma\,$.

**7**: Mostre, usando a Regra de Abstração, que a função:

$$\lambda p:\text{Nat} \times \text{Bool}.\;\pi_1\;p$$

   Tem o tipo $(\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}\,$.

   **Solução**:

   1. No contexto $\Gamma\,$, adicionamos $p:\text{Nat} \times \text{Bool}\,$.

   2. A operação $\pi_1\;p$ extrai o primeiro componente do par, portanto:

  $$\Gamma, p:\text{Nat} \times \text{Bool} \vdash \pi_1\;p : \text{Nat}$$

   3. Aplicando a Regra de Abstração:

   $$\frac{\Gamma, p:\text{Nat} \times \text{Bool} \vdash \pi_1\;p : \text{Nat}}{\Gamma \vdash \lambda p:\text{Nat} \times \text{Bool}.\;\pi_1\;p : (\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}}$$

   Portanto, a função tem tipo $(\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}\,$.

**8**: No contexto vazio, determine se a seguinte aplicação é bem tipada:

$$(\lambda x:\text{Nat}.\;x + 1)\;\text{True}$$

   Explique qual regra de tipagem é violada se a aplicação não for bem tipada.

   **Solução**:

   1. Temos $\Gamma \vdash \lambda x:\text{Nat}.\;x + 1 : \text{Nat} \rightarrow \text{Nat}\,$.

   2. E, $\Gamma \vdash \text{True} : \text{Bool}\,$.

   3. Pela Regra de Aplicação, para que a aplicação seja bem tipada, o tipo do argumento deve corresponder ao tipo esperado pela função:

   $$\frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M\;N : B}$$

   4. Aqui, $M$ espera um argumento do tipo $\text{Nat}\,$, mas $N$ é de tipo $\text{Bool}\,$.

   5. Como $\text{Nat} \neq \text{Bool}\,$, a aplicação não é bem tipada.

   A Regra de Aplicação foi violada porque o tipo do argumento fornecido não corresponde ao tipo esperado pela função.

**9**: Dado o termo:

$$M = \lambda x:\text{Bool}.\;\lambda y:\text{Bool}.\;x \land y$$

Determine o tipo de $M$ usando as regras de tipagem.

   **Solução**:

   1. No contexto $\Gamma\,$, adicionamos $x:\text{Bool}\,$.

   2. Dentro da função, adicionamos $y:\text{Bool}\,$.

   3. A expressão $x \land y$ tem tipo $\text{Bool}\,$.

   4. Aplicando a Regra de Abstração para $y$:

   $$\frac{\Gamma, x:\text{Bool}, y:\text{Bool} \vdash x \land $Y$ : \text{Bool}}{\Gamma, x:\text{Bool} \vdash \lambda y:\text{Bool}.\;x \land $Y$ : \text{Bool} \rightarrow \text{Bool}}$$

   5. Aplicando a Regra de Abstração para $x$:

   $$\frac{\Gamma \vdash \lambda y:\text{Bool}.\;x \land $Y$ : \text{Bool} \rightarrow \text{Bool}}{\Gamma \vdash \lambda x:\text{Bool}.\;\lambda y:\text{Bool}.\;x \land $Y$ : \text{Bool} \rightarrow (\text{Bool} \rightarrow \text{Bool})}$$

   Portanto, o tipo de $M$ é $\text{Bool} \rightarrow \text{Bool} \rightarrow \text{Bool}\,$.

**10**. Utilize as regras de tipagem para mostrar que a expressão:

   $$(\lambda f:\text{Nat} \rightarrow \text{Nat}.\;f\;(f\;2))\;(\lambda x:\text{Nat}.\;x + 3)$$

   Tem tipo $\text{Nat}\,$.

   **Solução**:

   1. Primeiro, analisamos a função:

   $$\lambda f:\text{Nat} \rightarrow \text{Nat}.\;f\;(f\;2)$$

   Dentro desta função, $f : \text{Nat} \rightarrow \text{Nat}\,$.
      - Sabemos que $2 : \text{Nat}\,$.
      - Então $f\;2 : \text{Nat}\,$.
      - Consequentemente, $f\;(f\;2) : \text{Nat}\,$.

   2. Portanto, a função tem tipo:

      $$(\text{Nat} \rightarrow \text{Nat}) \rightarrow \text{Nat}$$

   3. Agora, consideramos o argumento:

      $$\lambda x:\text{Nat}.\;x + 3$$

   Esta função tem tipo $\text{Nat} \rightarrow \text{Nat}\,$.

   4. Aplicando a **Regra de Aplicação**:

   $$\frac{\Gamma \vdash \lambda f:\text{Nat} \rightarrow \text{Nat}.\;f\;(f\;2) : (\text{Nat} \rightarrow \text{Nat}) \rightarrow \text{Nat} \quad \Gamma \vdash \lambda x:\text{Nat}.\;x + 3 : \text{Nat} \rightarrow \text{Nat}}{\Gamma \vdash (\lambda f:\text{Nat} \rightarrow \text{Nat}.\;f\;(f\;2))\;(\lambda x:\text{Nat}.\;x + 3) : \text{Nat}}$$

   Assim, a expressão completa tem tipo $\text{Nat}\,$.

## 10.5. Conversão e Redução no Cálculo Lambda Tipado

No cálculo lambda tipado, os processos de conversão e redução são essenciais para a manipulação e simplificação de expressões, garantindo que as transformações sejam consistentes com a estrutura de tipos. Essas operações permitem entender como as funções são aplicadas e como as expressões podem ser transformadas mantendo a segurança e a consistência do sistema tipado.

### 10.5.1. redução-$beta$

A**redução-$beta$**é o mecanismo central de computação no cálculo lambda tipado. Ela ocorre quando uma função é aplicada a um argumento, substituindo todas as ocorrências da variável ligada pelo valor do argumento na expressão. Formalmente, se temos uma abstração $\lambda x : A . M$ e aplicamos a um termo $N$ do tipo $A\,$, a redução-$beta$ é expressa como:

$$(\lambda x : A . M)\;N \rightarrow_\beta M[N/x]$$

onde $M[N/x]$ denota a substituição de todas as ocorrências livres de $x$ em $M$ por $N\,$. A redução-$beta$ é o passo básico da computação no cálculo lambda, e sua correta aplicação preserva os tipos das expressões envolvidas.

Por exemplo, considere a função de incremento aplicada ao número $2$:

$$(\lambda x : \text{Nat} .\;x + 1)\;2 \rightarrow_\beta 2 + 1 \rightarrow 3$$

Aqui, a variável $x$ é substituída pelo valor $2$ e, em seguida, a expressão é simplificada para $3\,$. No cálculo lambda tipado, a redução-$beta$ garante que os tipos sejam preservados, de modo que o termo final é do tipo $\text{Nat}\,$, assim como o termo original.

### 10.5.2. Conversões $\alpha$ e $\eta$

Além da redução-$beta\,$, existem duas outras formas importantes de conversão no cálculo lambda: a **redução-$\alpha$** e a **$\eta$-redução**.

- **redução-$\alpha$**: Esta operação permite a renomeação de variáveis ligadas, desde que a nova variável não conflite com variáveis livres. Por exemplo, as expressões $\lambda x : A .\;x$ e $\lambda $Y$ : A . y$ são equivalentes sob redução-$\alpha$:

$$\lambda x : A .\;x \equiv_\alpha \lambda $Y$ : A . y$$

 A redução-$\alpha$ é importante para evitar a captura de variáveis durante o processo de substituição, garantindo que a renomeação de variáveis ligadas não afete o comportamento da função.

- **$\eta$-redução**: A $\eta$-redução expressa o princípio de extensionalidade, que afirma que duas funções são idênticas se elas produzem o mesmo resultado para todos os argumentos. Formalmente, a $\eta$-redução permite que uma abstração lambda da forma $\lambda x : A . f\;x$ seja convertida para $f\,$, desde que $x$ não ocorra livre em $f$:

$$\lambda x : A . f\;x \rightarrow_\eta f$$

Esta propriedade se reflete em linguagens funcionais como Haskell, que suportam naturalmente funções de ordem superior. Por exemplo:

```haskell
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

increment :: Int -> Int
increment = (+1)

result :: Int
result = applyTwice increment 5
-- Retorna 7
```

Neste exemplo, `applyTwice` é uma função de ordem superior que toma uma função `f` do tipo `a -> a` e um valor `x` do tipo `a`, e aplica `f` duas vezes a `x`. A função increment é passada como argumento para `applyTwice`, demonstrando como funções podem ser tratadas como valores de primeira classe.

A $\eta$-redução simplifica as funções removendo abstrações redundantes, tornando as expressões mais curtas e mais diretas.

### 10.5.3. Normalização e Estratégias de Redução

Uma das propriedades mais importantes do cálculo lambda tipado é a **normalização forte**, que garante que todo termo bem tipado pode ser reduzido até uma **forma normal**, uma expressão que não pode mais ser simplificada. Isso significa que qualquer sequência de reduções, eventualmente, terminará, o que contrasta com o cálculo lambda não tipado, onde reduções infinitas são possíveis.

Existem diferentes estratégias de redução que podem ser aplicadas ao calcular expressões no cálculo lambda tipado:

1. **Redução por ordem normal**: Nessa estratégia, reduzimos sempre o redex mais à esquerda e mais externo primeiro. Essa abordagem garante que, se existir uma forma normal, ela é encontrada.

2. **Redução por ordem de chamada (call-by-name)**: Nesta estratégia, exclusivamente os termos que são necessários para a computação são reduzidos. Isso implementa uma avaliação _preguiçosa_, comum em linguagens funcionais como Haskell.

3. **Redução por valor (call-by-value)**: Nesta estratégia, os argumentos são completamente reduzidos antes de serem aplicados às funções. Isso é típico de linguagens com avaliação estrita, como OCaml ou ML.

Todas essas estratégias são **normalizantes**no cálculo lambda tipado, ou seja, alcançarão uma forma normal, se ela existir, devido à normalização forte.

### 10.5.4. Preservação de Tipos e Segurança

Um dos princípios do cálculo lambda tipado é a **preservação de tipos** durante a redução, conhecida como _subject reduction_. Essa propriedade assegura que, se um termo $M$ tem um tipo $A$ e $M$ é reduzido a $N$ através de redução-$beta\,$, então $N$ terá o tipo $A\,$. Formalmente:

$$
\frac{\Gamma \vdash M : A \quad M \rightarrow_\beta N}{\Gamma \vdash N : A}
$$

Essa propriedade, combinada com a **propriedade de progresso**, que afirma que todo termo bem tipado ou é um valor ou pode ser reduzido, estabelece a segurança do sistema de tipos no cálculo lambda tipado. Isso garante que, durante a computação, nenhum termo incorreto em termos de tipo é gerado.

### 10.5.5. Confluência e Unicidade da Forma Normal

O cálculo lambda tipado possui a propriedade de **confluência**, conhecida como **propriedade de Church-Rosser**. A palavra confluência, no nosso contexto, significa que, se um termo $M$ pode ser reduzido de duas formas diferentes para dois termos $N_1$ e $N_2\,$, sempre existirá um termo comum $P$ tal que $N_1$ e $N_2$ poderão ser reduzidos a $P$:

$$
M \to N*1 \quad M \rightarrow^* N*2 \quad \Rightarrow \quad \exists P : N_1 \rightarrow^* P \quad N*2 \rightarrow^* P
$$

A confluência, combinada com a normalização forte, garante a **unicidade da forma normal** para termos bem tipados. Isso significa que, independentemente da ordem de redução escolhida, um termo bem tipado sempre converge para a mesma forma normal, garantindo consistência e previsibilidade no processo de redução.

## 10.6. Propriedades do Cálculo Lambda Tipado

A partir das regras de tipagem, podemos definir um conjunto de propriedades da tipagem no cálculo lambda. Essas propriedades têm implicações tanto para a teoria quanto para a prática da programação. Vamos destacar:

1. **Normalização forte**: Todo termo bem tipado possui uma forma normal e qualquer sequência de reduções eventualmente termina. Isso garante que as reduções de expressões no cálculo lambda tipado sempre produzirão um resultado , eliminando loops infinitos. É importante notar que a propriedade de normalização forte é garantida no cálculo lambda simplesmente tipado. Em sistemas de tipos mais avançados, como aqueles que suportam recursão ou tipos dependentes, a normalização forte pode não se aplicar, podendo existir termos bem tipados que não convergem para uma forma normal.

   Formalmente, se $\Gamma \vdash M : \tau\,$, então existe uma sequência finita de reduções $M \rightarrow_\beta M_1 \rightarrow_\beta ... \rightarrow_\beta M_n$ onde $M_n$ está em forma normal.

   **Exemplo**: considere o termo $(\lambda x:\text{Nat}.\;x + 1)\;2\,$. Este termo é bem tipado e reduz para $3$ em um número finito de passos:

$$(\lambda x:\text{Nat}.\;x + 1)\;2 \rightarrow_\beta 2 + 1 \rightarrow 3$$

2. **Preservação de tipos** (_subject reduction_): se uma expressão $M$ possui o tipo $A$ sob o contexto $\Gamma\,$, e $M$ pode ser reduzido para $N$ pela regra redução-$beta$ ($M \rightarrow_\beta N$), então $N$ possui o tipo $A\,$. Essa propriedade é essencial para garantir que as transformações de termos dentro do sistema de tipos mantenham a consistência tipológica.

   Formalmente: Se $\Gamma \vdash M : \tau$ e $M \rightarrow_\beta N\,$, então $\Gamma \vdash N : \tau\,$.

   **Exemplo**: se $\Gamma \vdash (\lambda x:\text{Nat}.\;x + 1)\;2 : \text{Nat}\,$, então após a redução, teremos $\Gamma \vdash 3 : \text{Nat}\,$.

3. **Decidibilidade da tipagem**: um algoritmo pode decidir se uma expressão possui um tipo válido no sistema de tipos, o que é uma propriedade crucial para a análise de tipos em linguagens de programação.

   Isso significa que existe um procedimento efetivo que, dado um contexto $\Gamma$ e um termo $M\,$, pode determinar se existe um tipo $A$ tal que $\Gamma \vdash M : \tau\,$.

   **Exemplo**: um algoritmo de verificação de tipos pode determinar que:
   - $\lambda x:\text{Nat}.\;x + 1$ tem tipo $\text{Nat} \rightarrow \text{Nat}$
   - $(\lambda x:\text{Nat}.\;x) \text{True}$ não é bem tipado

4. **Progresso**: uma propriedade adicional importante é o progresso. Se um termo é bem tipado e não está em forma normal, então existe uma redução que pode ser aplicada a ele.

   Formalmente: Se $\Gamma \vdash M : \tau$ e $M$ não está em forma normal, então existe $N$ tal que $M \rightarrow_\beta N\,$.

   Esta propriedade, junto com a preservação de tipos, garante que termos bem tipados ou estão em forma normal ou podem continuar sendo reduzidos sem ficarem _presos_ em um estado intermediário.

Estas propriedades juntas garantem a consistência e a robustez do sistema de tipos do cálculo lambda tipado, fornecendo uma base sólida para o desenvolvimento de linguagens de programação tipadas e sistemas de verificação formal.

### 10.6.1. Exercícios de Propriedades do Cálculo Lambda Tipado

**1**: Considere o termo $(\lambda x:\text{Nat}.\;x + 1)\;2\,$. Mostre a sequência de reduções que leva este termo à sua forma normal, ilustrando a propriedade de normalização forte.

   **Solução**:

$$(\lambda x:\text{Nat}.\;x + 1)\;2 \rightarrow_\beta 2 + 1 \rightarrow 3$$

   O termo reduz à sua forma normal, $3\,$, em um número finito de passos.

**2**: Dado o termo $(\lambda f:\text{Nat}\rightarrow\text{Nat}. \lambda x:\text{Nat}. f (f x)) (\lambda y:\text{Nat}. $Y$ + 1)\;2\,$, mostre que ele é bem tipado e reduz para um valor do tipo $\text{Nat}\,$.

   **Solução**: 0 termo é bem tipado: $(\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat}\rightarrow\text{Nat}$

   Redução:

$$\begin{aligned}
   &(\lambda f:\text{Nat}\rightarrow\text{Nat}. \lambda x:\text{Nat}. f (f x)) (\lambda y:\text{Nat}. $Y$ + 1)\;2 \\
   &\rightarrow_\beta (\lambda x:\text{Nat}. (\lambda y:\text{Nat}. $Y$ + 1) ((\lambda y:\text{Nat}. $Y$ + 1) x))\;2 \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. $Y$ + 1) ((\lambda y:\text{Nat}. $Y$ + 1)\;2) \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. $Y$ + 1) (2 + 1) \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. $Y$ + 1)\;3 \\
   &\rightarrow_\beta 3 + 1 \\
   &\rightarrow 4
   \end{aligned}$$

   O resultado  (4) é do tipo $\text{Nat}\,$, ilustrando a preservação de tipos.

**3**. Explique por que o termo $(\lambda x:\text{Bool}.\;x + 1)$ não é bem tipado. Como isso se relaciona com a propriedade de decidibilidade da tipagem?

   **Solução**: o termo não é bem tipado porque tenta adicionar 1 a um valor booleano, o que é uma operação inválida. A decidibilidade da tipagem permite que um algoritmo detecte este erro de tipo, rejeitando o termo como mal tipado.

**4**: Considere o termo:

$$
   M = (\lambda x:\text{Nat}.\;\text{if } x = 0\;\text{then}\;1\;\text{else}\;x \times ((\lambda y:\text{Nat}.\;y - 1)\;x))
$$

   Este termo calcula $x$ multiplicado por $x - 1\,$. Mostre que ele satisfaz a propriedade de **preservação de tipos** para uma entrada específica.

   **Solução**: vamos aplicar o termo $M$ a $x = 3$ e verificar a preservação de tipos durante as reduções.

   1. **Tipagem Inicial:**

      Tipo de $M$:

      $$\Gamma \vdash M : \text{Nat} \rightarrow \text{Nat}$$

      $M$ é uma função que recebe um $\text{Nat}$ e retorna um $\text{Nat}\,$.

      Tipo do Argumento $3$:

      $$\Gamma \vdash 3 : \text{Nat}$$

   2. **Aplicação da Função:**

      Aplicação:

      $$M\;3 = (\lambda x:\text{Nat}.\;\text{if } x = 0\;\text{then}\;1\;\text{else}\;x \times ((\lambda y:\text{Nat}.\;y - 1)\;x))\;3$$

      Redução por redução-$beta$:

      $$\rightarrow_\beta \text{if } 3 = 0\;\text{then}\;1\;\text{else}\;3 \times ((\lambda y:\text{Nat}.\;y - 1)\;3)$$

   3. **Avaliação da Condicional:**

      Como $3 \neq 0\,$, seguimos para o ramo _else_:

      $$\rightarrow 3 \times ((\lambda y:\text{Nat}.\;y - 1)\;3)$$

   4. **Redução do Parêntese Interno:**

      Aplicação da Função Interna:

      $$(\lambda y:\text{Nat}.\;y - 1)\;3 \rightarrow_\beta 3 - 1 = 2$$

      Atualização da Expressão:

      $$\rightarrow 3 \times 2$$

   5. **Cálculo Final:**

      Multiplicação:

      $$3 \times 2 = 6$$

      Tipo do Resultado:

      $$\Gamma \vdash 6 : \text{Nat}$$

   **Demonstrando a Preservação de Tipos:**

   Antes da Redução: a função $M$ tem tipo $\text{Nat} \rightarrow \text{Nat}$; o argumento $3$ tem tipo $\text{Nat}\,$. Portanto, pela **Regra de Aplicação**:

   $$\frac{\Gamma\;\vdash\;M : \text{Nat} \rightarrow \text{Nat} \quad \Gamma\;\vdash\;3 : \text{Nat}}{\;\Gamma\;\vdash\;M\;3 : \text{Nat}}$$

   Durante as Reduções, cada passo manteve o tipo $\text{Nat}$:

   - $\Gamma \vdash 3 \times 2 : \text{Nat}$
  
   - $\Gamma \vdash 6 : \text{Nat}$

   O termo inicial $M\;3$ tem tipo $\text{Nat}\,$. Após as reduções, o resultado $6$ tem tipo $\text{Nat}\,$. Portanto, a propriedade de preservação de tipos é satisfeita.

   **Observação**: como usamos as regras de tipagem.

  1. **Regra da Variável**: definimos os tipos das variáveis $x$ e $y$ como $\text{Nat}\,$.

  2. **Regra de Abstração**: As funções anônimas $\lambda x:\text{Nat}.\;\dots$ e $\lambda y:\text{Nat}.\;y - 1$ são tipadas como $\text{Nat} \rightarrow \text{Nat}\,$.

  3. **Regra de Aplicação**: Aplicamos as funções aos argumentos correspondentes, mantendo a consistência de tipos.

Repita este exercício. Ele demonstra como as reduções em um termo bem tipado mantêm o tipo consistente.

**5**: Dê um exemplo de um termo que não satisfaz a propriedade de progresso no cálculo lambda não tipado, mas que seria rejeitado no cálculo lambda tipado.

   **Solução**: considere o termo $(\lambda x.\;x\;x) (\lambda x.\;x\;x)\,$. No cálculo lambda não tipado, este termo reduz infinitamente para si mesmo:

   $$(\lambda x.\;x\;x) (\lambda x.\;x\;x) \rightarrow_\beta (\lambda x.\;x\;x) (\lambda x.\;x\;x) \rightarrow_\beta \cdots$$

   No cálculo lambda tipado, este termo seria rejeitado porque não é possível atribuir um tipo consistente para $x$ em $x\;x\,$.

**6**: Explique como a propriedade de normalização forte garante que não existem loops infinitos em programas bem tipados no cálculo lambda tipado.

   **Solução**: a normalização forte garante que toda sequência de reduções de um termo bem tipado eventualmente termina em uma forma normal. Isso implica que não pode haver loops infinitos, pois se houvesse, a sequência de reduções nunca terminaria, contradizendo a propriedade de normalização forte.

**7**: Considere o termo $(\lambda x:\text{Nat}\rightarrow\text{Nat}.\;x 3) (\lambda y:\text{Nat}. $Y$ * 2)\,$. Mostre que este termo satisfaz as propriedades de preservação de tipos e progresso.

   **Solução**: preservação de tipos: O termo inicial tem tipo $\text{Nat}\,$. Após a redução:

   $$(\lambda x:\text{Nat}\rightarrow\text{Nat}.\;x 3) (\lambda y:\text{Nat}. $Y$ \times 2) \rightarrow_\beta (\lambda y:\text{Nat}. $Y$ \times 2)\;3 \rightarrow_\beta 3 \times 2 \rightarrow 6$$

   O resultado  $6$ ainda é do tipo $\text{Nat}\,$.

   Progresso: O termo inicial não está em forma normal e pode ser reduzido, como mostrado acima.

**8**: Explique por que a decidibilidade da tipagem é importante para compiladores de linguagens de programação tipadas.

   **Solução**: a decidibilidade da tipagem permite que compiladores verifiquem estaticamente se um programa está bem tipado. Isso é crucial para detectar erros de tipo antes da execução do programa, melhorando a segurança e a eficiência. Sem esta propriedade, seria impossível garantir que um programa está livre de erros de tipo em tempo de compilação.

**9**: Dê um exemplo de um termo que é bem tipado no cálculo lambda tipado, mas que não teria uma representação direta em uma linguagem sem tipos de ordem superior (como C).

   **Solução**: considere o termo:

   $$\lambda f:(\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat}. f (\lambda x:\text{Nat}.\;x + 1)$$

   Este termo tem tipo $((\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat})\rightarrow\text{Nat}\,$. Ele representa uma função que toma como argumento outra função (que por sua vez aceita uma função como argumento). Linguagens sem tipos de ordem superior, como C, não podem representar diretamente funções que aceitam ou retornam outras funções.

**10**. Como a propriedade de preservação de tipos contribui para a segurança de execução em linguagens de programação baseadas no cálculo lambda tipado?

   **Solução**: a preservação de tipos garante que, à medida que um programa é executado (ou seja, à medida que os termos são reduzidos), os tipos dos termos permanecem consistentes. Isso significa que operações bem tipadas no início da execução permanecerão bem tipadas durante toda a execução. Esta propriedade previne erros de tipo em tempo de execução, contribuindo significativamente para a segurança de execução ao garantir que operações inválidas (como tentar adicionar um booleano a um número) nunca ocorrerão durante a execução de um programa bem tipado.

O cálculo lambda tipado, com suas regras de tipagem e propriedades de normalização, oferece um conjunto de ferramentas para analisar programas e suas características. Esta estrutura formal permite a construção de programas com certas garantias e revela uma relação entre a lógica proposicional e os sistemas de tipos.

Esta relação é formalizada na Correspondência de Curry-Howard, que estabelece uma conexão entre programas e provas matemáticas. A correspondência liga duas áreas da matemática e da Ciência da Computação. Como pode ser visto na Figura 8.6.1.A.

![Um diagrama de blocos mostrando a relação entre a aplicação de função e a lógica proposicional](/assets/images/churchRosser.webp)
_Figura 7.6.1.A: Diagrama da Relação entre cálculo lambda e lógica proposicional mostrando a importância do Teorema de Church-Rosser._{: legenda}

Na Correspondência de Curry-Howard, os tipos em linguagens de programação podem ser vistos como proposições lógicas, e os programas bem tipados como provas dessas proposições. Esta perspectiva fornece uma forma de analisar sistemas de tipos e oferece abordagens para o desenvolvimento de software e a verificação formal de programas.

## 10.7. Correspondência de Curry-Howard

A Correspondência de Curry-Howard, conhecida como Isomorfismo de Curry-Howard estabelece uma profunda conexão entre tipos em linguagens de programação e proposições em lógica construtiva.

O isomorfismo de Curry-Howard tem raízes no trabalho realizado por um conjunto de pesquisadores ao longo do século XX. Contudo, [Haskell Curry](https://en.wikipedia.org/wiki/Haskell_Curry), em 1934, foi o primeiro a observar uma conexão entre a lógica combinatória e os tipos de funções, notando que os tipos dos combinadores correspondiam a tautologias na lógica proposicional.

Um longo hiato se passou, até que [William Howard](https://en.wikipedia.org/wiki/William_Alvin_Howard), em 1969, expandiu esta observação para um isomorfismo completo entre lógica intuicionista e cálculo lambda tipado, mostrando que as regras de dedução natural correspondiam às regras de tipagem no cálculo lambda simplesmente tipado.

>A lógica intuicionista é um sistema formal de lógica desenvolvido por [Arend Heyting](https://en.wikipedia.org/wiki/Arend_Heyting), baseado nas ideias do matemático [L.E.J. Brouwer](https://en.wikipedia.org/wiki/L._E._J._Brouwer). Diferentemente da lógica clássica, a lógica intuicionista rejeita o princípio do terceiro excluído (A ou não-A) e a lei da dupla negação (não-não-A implica A). Ela exige provas construtivas, onde a existência de um objeto matemático só é aceita se houver um método para construí-lo. Esta abordagem tem implicações profundas na matemática e na Ciência da Computação, especialmente na teoria dos tipos e na programação funcional, onde se alinha naturalmente com o conceito de computabilidade.

A correspondência foi posteriormente generalizada por [Jean-Yves Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard) e [John C. Reynolds](https://en.wikipedia.org/wiki/John_C._Reynolds), que independentemente, em 1971-72, estenderam o isomorfismo para incluir a quantificação de segunda ordem. Eles demonstraram que o **Sistema F** (cálculo lambda polimórfico) corresponde à lógica de segunda ordem, estabelecendo assim as bases para uma compreensão profunda da relação entre lógica e computação. Estas descobertas tiveram um impacto no desenvolvimento de linguagens de programação e sistemas de prova assistidos por computador.

Assim, chegamos aos dias atuais com a correspondência Curry-Howard tendo implicações tanto para a teoria da computação quanto para o desenvolvimento de linguagens de programação. Vamos examinar os principais aspectos deste isomorfismo:

1. Proposições como tipos

   Na lógica construtiva, **a verdade de uma proposição é equivalente à sua demonstrabilidade**. Isso significa que para uma proposição $P$ ser verdadeira, deve existir uma prova de $P\,$.

   Formalmente: Uma proposição $P$ é verdadeira se, e somente se, existe uma prova $p$ de $P\,$, denotada por $p : P\,$. Ou seja, a proposição _existe um número primo maior que 100_ é verdadeira porque podemos fornecer uma prova construtiva, como o número 101 e uma demonstração de que este é um número primo.

2. Provas como programas

   **As regras de inferência na lógica construtiva correspondem às regras de tipagem em linguagens de programação**. Assim, um programa bem tipado pode ser visto como uma prova de sua especificação tipo.

   Formalmente: Se $\Gamma \vdash e : \tau\,$, então $e$ pode ser interpretado como uma prova da proposição representada por $\tau$ no contexto $\Gamma\,$. Por exemplo, um programa $e$ do tipo $\text{Nat} \rightarrow \text{Nat}$ é uma prova da proposição _existe uma função dos números naturais para os números naturais_.

3. Correspondência entre conectivos lógicos e tipos

   Existe uma correspondência direta entre conectivos lógicos e construtores de tipos:

   1. Conjunção ($\wedge$) \, = Tipo produto ($\times$)
   2. Disjunção ($\vee$) \, = Tipo soma ($+$)
   3. Implicação ($\rightarrow$) \, = Tipo função ($\rightarrow$)
   4. Quantificação universal ($\forall$) \, = Polimorfismo paramétrico ($\forall$)

   A proposição $P_1 \wedge P_2 \rightarrow P_3$ corresponde ao tipo $\tau_1 \times \tau_2 \rightarrow \tau_3\,$.

4. Invalidade e tipos inabitados

   Uma proposição falsa na lógica construtiva corresponde a um tipo inabitado na teoria de tipos.

   Formalmente: Uma proposição $P$ é falsa se e somente se não existe termo $e$ tal que $e : P\,$. O tipo $\forall X.\;x$ é inabitado, correspondendo a uma proposição falsa na lógica.

Estas correspondências fornecem uma base sólida para o desenvolvimento de linguagens de programação com sistemas de tipos expressivos e para a verificação formal de programas. Por outro lado, provar a veracidade destas quatro correspondências é desafiador. Vamos nos arriscar com a terceira delas, correspondência entre conectivos lógicos e tipos. Nos limitando ao item três, a relação entre a implicação lógica e o tipo da função. Principalmente por esta ser a base da correspondência de Curry-Howard. Provar que a implicação lógica corresponde ao tipo função estabelece o fundamento para as outras correspondências.

### 10.7.1. Provando: Implicação Lógica para Tipo de Função

Nas lógicas construtivista e proposicional, a implicação $A \rightarrow B$ significa que, se $A$ é verdadeiro, então $B$ deve ser verdadeiro. Uma prova de $A \rightarrow B$ consiste em assumir $A$ e derivar $B\,$.

>A lógica construtivista e a lógica proposicional diferem significativamente em seus princípios e métodos. Enquanto a lógica proposicional aceita a lei do terceiro excluído ($P \vee \neg P$) e provas por contradição, a lógica construtivista as rejeita em certos contextos, exigindo construções explícitas para provar existência. A lógica construtivista tem forte conexão com a teoria da computação, interpretando quantificadores de forma distinta e enfatizando métodos de prova algorítmicos. Essas diferenças impactam a formulação de teoremas matemáticos e têm implicações importantes para a fundação da matemática e Ciência da Computação.

Usando a matemática, podemos partir da definição da implicação:

$$A \rightarrow B \equiv$$

Que lemos como _Se $A\,$, então $B$_. Depois, podemos lembrar da regra da introdução da implicação:

$$\frac{[A] \vdash B}{A \rightarrow B}$$

Esta regra afirma que se podemos derivar $B$ assumindo $A\,$, então podemos concluir $A \rightarrow B\,$. No cálculo lambda tipado, definimos uma função $f: A \rightarrow B$ como:

$$f \equiv \lambda x:A. t$$

onde $t$ é um termo do tipo $B$ que pode depender de $x\,$.

Usando a regra de tipagem para abstração lambda:

$$\frac{\Gamma, x:A \vdash t:B}{\Gamma \vdash (\lambda x:A. t): A \rightarrow B}$$

Podemos remover a implicação usando o modus ponens:

$$\frac{A \rightarrow B \quad A}{B}$$

Voltando ao cálculo lambda, esta regra corresponde à aplicação de função:

$$\frac{f: A \rightarrow B \quad a: A}{f(a): B}$$

Estas regras estabelecem uma correspondência direta entre a implicação lógica e o tipo de função no cálculo lambda tipado.

Talvez seja mais fácil entender esta prova se usarmos Haskell. Começando com a definição de tipos correspondentes a $A$ e $B$:

```haskell
data A = A
data B = B
```

Podemos definir uma função correspondente à implicação $A \rightarrow B$:

```haskell
f :: A -> B
f = \x -> t
  where
    t :: B
    t = B  -- Aqui, t é um termo do tipo B que pode depender de x
```

agora podemos aplicar esta função, o que correspondente à eliminação da implicação:

```haskell
a :: A
a = A

resultado :: B
resultado = f a
```

Esta implementação em Haskell demonstra como os conceitos da prova formal se traduzem em código executável. A função `f` representa a implicação $A \rightarrow B\,$, e sua aplicação a um valor do tipo `A` produz um resultado do tipo `B`, espelhando a regra de eliminação da implicação na lógica.

A correspondência entre implicação lógica e tipos de função é evidente tanto na prova formal quanto na implementação Haskell. A abstração lambda $\lambda x:A. t$ na matemática corresponde diretamente à função `\x -> t` em Haskell, ambas representando uma função de $A$ para $B\,$, que é a contraparte direta de uma prova de $A \rightarrow B$ na lógica.

Esta demonstração usando matemática e programação, da correspondência entre implicação lógica e tipos de função ilustra como os conceitos da lógica proposicional se mapeiam diretamente para estruturas no cálculo lambda tipado e em linguagens de programação funcional, fornecendo uma base para entendimento da relação profunda entre lógica e programação. Este entendimento deve permitir a _apreciação da expressividade e o poder do cálculo lambda tipado como um sistema formal para raciocínio sobre programas e provas_. Talvez, esta seja a frase mais importante de todo este texto, para a carreira dos jovens cientistas da computação.

### 10.7.2. Sistema de Tipos

No cálculo lambda tipado, os tipos podem ser básicos ou compostos. Tipos básicos incluem, por exemplo, $\text{Bool}\,$, que representa valores booleanos, e $\text{Nat}\,$, que denota números naturais. Tipos de função são construídos a partir de outros tipos; $A \rightarrow B$ denota uma função que mapeia valores do tipo $A$ para valores do tipo $B\,$. Por exemplo, em Haskell, podemos definir uma função simples que representa $\lambda x:\text{Nat}.\;x + 1$:

```haskell
increment :: Int -> Int
increment x = x + 1
```
O sistema de tipos tem uma estrutura recursiva, permitindo a construção de tipos complexos a partir de tipos mais simples.

A tipagem de variáveis assegura que cada variável esteja associada a um tipo específico. Uma variável $x$ do tipo $A$ é denotada como $x : A\,$. Isso implica que $x$ só pode ser associado a valores que respeitem as regras do tipo $A\,$, restringindo o comportamento da função.

Um **contexto de tipagem**, representado por $\Gamma\,$, é um conjunto de associações entre variáveis e seus tipos. O contexto fornece informações necessárias sobre as variáveis livres em uma expressão, facilitando o julgamento de tipos. Por exemplo, um contexto $\Gamma = \{x : A, $Y$ : B\}$ indica que, nesse ambiente, a variável $x$ tem tipo $A$ e a variável $y$ tem tipo $B\,$. Os contextos são essenciais para derivar os tipos de expressões mais complexas.

### 10.7.3. Normalização Forte e Fraca

O cálculo lambda é um sistema minimalista. Porém, forte e consistente. Uma característica importante para a criação de linguagens de programação, que podemos destacar é a normalização. Existem dois tipos de normalização:

1. **Normalização fraca**: todo termo tem uma forma normal e Você vai chegar lá eventualmente.

2. **Normalização forte**: toda sequência de reduções termina em forma normal. Não importa como a atenta leitora aplique a redução, se não errar vai alcançar uma forma normal, se essa forma normal existir.

A normalização forte é, para nós, de suprema importância. É o que almejamos em todos os ambientes computacionais, garante que todos os programas terminem após um número finito de passos, evitando _loops_ infinitos e comportamentos imprevisíveis. No cálculo lambda simplesmente tipado, essa propriedade está assegurada, o que é algo notável.

Como em qualquer resultado matemático, a normalização precisa ser provada. Para simplificar, apresentaremos uma prova informal. Embora não seja a mais elegante, capturaremos a essência do argumento. Para tanto:

1. Atribuímos um _tamanho_ a cada termo.
2. Mostramos que cada redução torna o termo menor.
3. Como não é possível diminuir para sempre, podemos parar.

Os detalhes são complicados. Mas essa é a ideia principal. Aqui está um esboço da função de tamanho:

$$size(\lambda x:A.M) \, = size(M) + 1$$

$$size(MN) \, = size(M) + size(N) + 1$$

Cada redução-$beta$ diminui o termo. Observe que não é possível reduzir para sempre. Então, a insistente leitora terá que parar em algum ponto, se existir, este ponto será uma forma normal.

A prática da normalização impregna todo o Haskell. Talvez porque a normalização garante a terminação de programas bem tipados. Sáo programas sem _loops_ infinitos, sem falhas inesperadas, ou comportamentos indeterminados. Simplesmente funções puras que chegam a um resultado .

### 10.7.4. Exemplos

Vamos ver a normalização em ação com três estruturas que vimos antes:

1. Função identidade:

 $(\lambda x:A.x)\;M \to M$

 Um passo e Acabou. Forma normal atingida.

2. Função constante:

 $(\lambda x:A.\lambda y:B.x)MN \to (\lambda y:B.M)N \to M$

 Dois passos. E forma normal atingida.

3. Números de Church:

 $2\;3 \equiv (\lambda f.\lambda x.f(fx))(\lambda y.yyy) \to_\beta \lambda x.(\lambda y.yyy)((\lambda y.yyy)x) \to_\beta \lambda x.(\, x\;xx)(\, x\;xx) \to_\beta \lambda x.\, x\;x\, x\;x\, x\;x\, x\;xx$

 Vários passos. Mas termina. Isso é a normalização em ação.

A normalização é poderosa. É a espinha dorsal da programação funcional. É o que torna essas linguagens seguras e previsíveis.

Lembre-se: no cálculo lambda, tudo termina. Essa é a beleza. Esse é o poder da normalização.

### 10.7.5. Termos Bem Tipados e Segurança do Sistema

Um termo é considerado **bem tipado** se sua derivação de tipo pode ser construída usando as regras de tipagem formais. A tipagem estática é uma característica importante do cálculo lambda tipado, pois permite detectar erros de tipo durante o processo de compilação, antes mesmo de o programa ser executado. Isso é essencial para a segurança e confiabilidade dos sistemas, já que garante que funções não sejam aplicadas a argumentos incompatíveis.

O sistema de tipos do cálculo lambda tipado exclui automaticamente termos paradoxais como o combinador $\omega = \lambda x.\;x\;x\,$. Para que $\omega$ fosse bem tipado, a variável $x$ precisaria ter o tipo $A \rightarrow A$ e ao mesmo tempo o tipo $A\,$, o que é impossível. Assim, a auto-aplicação de funções é evitada, garantindo a consistência do sistema.

### 10.7.6. Propriedades do Sistema de Tipos

O cálculo lambda tipado apresenta várias propriedades que reforçam a robustez do sistema:

-**Normalização Forte**: Todo termo bem tipado tem uma forma normal, ou seja, pode ser reduzido até uma forma final através de redução-$beta\,$. Isso implica que termos bem tipados não entram em loops infinitos de computação.

-**Preservação de Tipos (Subject Reduction)**: Se $\Gamma \vdash M : A$ e $M \rightarrow_\beta N\,$, então $\Gamma \vdash N : A\,$. Isso garante que a tipagem é preservada durante a redução de termos, assegurando a consistência dos tipos ao longo das transformações.

-**Progresso**: Um termo bem tipado ou é um valor (isto é, está em sua forma final), ou pode ser reduzido. Isso significa que termos bem tipados não ficam presos em estados intermediários indeterminados.

-**Decidibilidade da Tipagem**: É possível determinar, usando algorítimos, se um termo é bem tipado e, em caso afirmativo, qual é o seu tipo. Essa propriedade é essencial para a verificação automática de tipos em sistemas formais e linguagens de programação.

### 10.7.7. Correspondência de Curry-Howard

A **correspondência de Curry-Howard** estabelece uma relação profunda entre o cálculo lambda tipado e a lógica proposicional intuicionista. Sob essa correspondência, termos no cálculo lambda tipado são vistos como provas, e tipos são interpretados como proposições. Em particular:

-Tipos correspondem a proposições.
-Termos correspondem a provas.
-Normalização de termos corresponde à normalização de provas.

Por exemplo, o tipo $A \rightarrow B$ pode ser interpretado como a proposição lógica _se $A\,$, então $B$_, e um termo deste tipo representa uma prova dessa proposição. Essa correspondência fornece a base para a verificação formal de programas e para a lógica assistida por computador.

O Cálculo Lambda Simplesmente Tipado fornece uma base formal para o estudo de linguagens de programação tipadas e sistemas de verificação formal. Suas propriedades, como normalização forte, preservação de tipos e decidibilidade da tipagem, têm implicações tanto para a teoria quanto para a prática da Ciência da Computação. A correspondência entre o cálculo lambda tipado e linguagens de programação modernas pode ser observada em várias construções. Por exemplo, em Haskell:

```haskell
identity :: a -> a
identity = \x -> x

applyFunction :: (a -> b) -> a -> b
applyFunction f x = f x

compose :: (b -> c) -> (a -> b) -> a -> c
compose f g = \x -> f (g x)
```

Estas implementações em Haskell refletem diretamente os conceitos do cálculo lambda tipado. A função `identity` corresponde à abstração $\lambda x:A.\;x\,$, `applyFunction` demonstra a regra de aplicação, e `compose` ilustra como funções de ordem superior são tratadas no sistema de tipos.

A Correspondência de Curry-Howard estabelece uma conexão profunda entre o cálculo lambda tipado e a lógica proposicional, unificando os conceitos de computação e prova formal. Esta correspondência tem implicações para o desenvolvimento de assistentes de prova baseados em tipos, a derivação de programas a partir de especificações formais e a verificação formal de propriedades de programas.

O estudo do cálculo lambda tipado e suas extensões continua a influenciar o design de linguagens de programação, sistemas de tipos avançados e métodos formais para o desenvolvimento de software. À medida que a complexidade dos sistemas de software aumenta, os princípios derivados do cálculo lambda tipado tornam-se cada vez mais relevantes para garantir a correção e a segurança dos programas.

# 11. Notas e Referências

[^cita1]: Schönfinkel, Moses. **Über die Bausteine der mathematischen Logik**. _Mathematische Annalen_, vol. 92, no. 1-2, 1924, pp. 305-316.

[^cita2]: Malpas, J., Davidson, D., **The Stanford Encyclopedia of Philosophy (Winter 2012 Edition)**, Edward N.\;zalta and Uri Nodelman (eds.), URL = <https://plato.stanford.edu/entries/lambda-calculus/#Com>.

[^cita3]: DOMINUS, M., **Why is the S combinator an S?**, URL = <https://blog.plover.com/math/combinator-s.html>.

[^cita4]: CARDONE, Felice; HINDLEY, J. Roger. **History of Lambda-calculus and Combinatory Logic**. Swansea University Mathematics Department Research Report No. MRRS-05-06, 2006. URL = <https://hope.simons-rock.edu/~pshields/cs/cmpt312/cardone-hindley.pdfl>.

[^cita5]: Alonzo Church and J.B. Rosser. **Some properties of conversion**. Transactions of the American Mathematical Society, 39(3):472–482, May 1936. <https://www.ams.org/journals/tran/1936-039-03/S0002-9947-1936-1501858-0/S0002-9947-1936-1501858-0.pdf>

[^cita6]: Alan Turing. **On computable numbers, with an application to the entscheidungsproblem**. Proceedings of the London Mathematical Society, 42:230–265, 1936. Published 1937.

[^cita7]: SELINGER, Peter. **Lecture Notes on the Lambda Calculus**. Department of Mathematics and Statistics, Dalhousie University, Halifax, Canada.

[^cita8]: BARENDREGT, H. P. (1984). **The Lambda Calculus: Its Syntax and Semantics**. North-Holland.

[^cita9]: WIGDERSON, Avi. **Mathematics and Computation: A Theory Revolutionizing Technology and Science.** Princeton University Press, 2019.
