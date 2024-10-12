---
layout: post
title: O Cálculo Lambda - Fundamentos da Computação Funcional
author: Frank
categories:
   - Matemática
   - Linguagens Formais
   - Lógica Matemática
tags:
   - Matemática
   - Linguagens Formais
image: assets/images/calculolambda.jpg
description: Introdução ao cálculo lambda.
slug: calculo-lambda-fundamentos-da-computacao-funcional
keywords:
   - Cálculo Lambda
   - Code Comparison
rating: 5
published: 2024-09-08T21:19:20.392Z
draft: 2024-09-08T21:19:20.392Z
featured: true
toc: true
preview: Este guia apresenta o cálculo lambda. Começamos com os fundamentos teóricos e seguimos para as aplicações práticas em linguagens de programação funcionais. Explicamos abstração, aplicação e recursão. Mostramos exemplos de *currying* e combinadores de ponto fixo. O cálculo lambda é uma base para a computação funcional.
beforetoc: Este guia apresenta o cálculo lambda. Começamos com os fundamentos teóricos e seguimos para as aplicações práticas em linguagens de programação funcionais. Explicamos abstração, aplicação e recursão. Mostramos exemplos de *currying* e combinadores de ponto fixo. O cálculo lambda é uma base para a computação funcional.
lastmod: 2024-10-12T01:31:14.026Z
date: 2024-09-08T21:19:30.955Z
---

# Introdução, História e Motivações e Limites

O cálculo lambda é uma teoria formal para expressar computação por meio da visão de funções como fórmulas. Um sistema para manipular funções como sentenças, desenvolvido por Alonzo [Church](https://en.wikipedia.org/wiki/Alonzo_Church) sob uma visão extensionista das funções na década de 1930. Nesta teoria usamos funções para representar todos os dados e operações. Em cálculo lambda, tudo é uma função e uma função simples será parecida com:

$$ \lambda x. x + 1 $$

Esta função adiciona $1$ ao seu argumento. O $\lambda$ indica que estamos definindo uma função.

Na teoria da computação definida por Church com o cálculo lambda existem três componentes básicos: as variáveis: $x$, $y$, $z$; as abstrações $\lambda x. E$, onde $E$ é uma expressão lambda e a aplicação $(E \, M)$, onde $E$ e $M$ são expressões lambda. Com estes três componentes e o cálculo lambda é possível expressar qualquer função computacionalmente possível.

A década de 1930 encerrou a busca pela consistência da matemática iniciada nas última décadas do século XIX. Neste momento histórico os matemáticos buscavam entender os limites da computação. Questionavam: Quais problemas podem ser resolvidos por algoritmos? Existem problemas não computáveis?

Estas questões surgiram como consequência dos trabalhos no campo da lógica e da lógica combinatória que despontaram no final do século XIX e começo do século XX. Em um momento crítico, Church ofereceu respostas, definindo que as funções computáveis são aquelas que podem ser expressas em cálculo lambda. Um exemplo simples de função computável seria:

$$ \text{soma} = \lambda m. \lambda n. \, m + n $$

Esta função soma dois números. **Todas as funções lambda são, por definição unárias e anônimas**. Assim, a função acima está sacrificando o rigor matemático para facilitar o entendimento. Esta é uma liberdade que será abusada descaradamente neste texto sempre com a esperança que estando mais próximo do que aprendemos nos ciclos básicos de estudo, será mais simples de entender.

O trabalho de Church estabeleceu limites claros para computação, ajudando a revelar o que é e o que não é computável. Sobre esta formalização foi construída a ciência da computação. Seu objetivo era entender e formalizar a noção de _computabilidade_. Church buscava um modelo matemático preciso para computabilidade. Nesta busca ele criou uma forma de representar funções e operações matemáticas de forma abstrata, usando como base a lógica combinatória desenvolvida anos antes [^cita4].

Na mesma época, [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) desenvolveu a [máquina de Turing](https://en.wikipedia.org/wiki/Turing_machine), uma abordagem diferente para tratar a computabilidade. Apesar das diferenças, essas duas abordagens provaram ser equivalentes e, juntas, estabeleceram os alicerces da teoria da computação moderna. O objetivo de Church era capturar o conceito de _cálculo efetivo_[^cita5]. Seu trabalho foi uma das primeiras tentativas de formalizar matematicamente o ato de computar. Mais tarde, a equivalência entre o cálculo lambda e a máquina de Turing consolidou a ideia de que ambos podiam representar qualquer função computável, levando à formulação da [Tese de Church-Turing](https://en.wikipedia.org/wiki/Church%E2%80%93Turing_thesis). Afirmando que qualquer função computável pode ser resolvida pela máquina de touring e, equivalentemente, pelo cálculo lambda, fornecendo uma definição matemática precisa do que é, ou não, computável. 

A partir do meio da década de 1930, vários matemáticos e lógicos, como [Church](https://en.wikipedia.org/wiki/Alonzo_Church), [Turing](https://en.wikipedia.org/wiki/Alan_Turing), [Gödel](https://en.wikipedia.org/wiki/Kurt_G%C3%B6del) e [Post](https://en.wikipedia.org/wiki/Emil_Leon_Post), desenvolveram modelos diferentes para formalizar a computabilidade. Cada um desses modelos abordou o problema de uma perspectiva exclusiva.

Church propôs o cálculo lambda para descrever funções de forma simbólica, usando a _abstração lambda_. Esse modelo representa funções como estruturas de primeira classe formalizando a computabilidade em termos de funções e variáveis.

Em 1936, [Alan Turing](https://en.wikipedia.org/wiki/Alan_Turing) propôs a máquina de Turing. Essa máquina, conceitual, é formadas por uma fita infinita que pode ser lida e manipulada por uma cabeça de leitura/escrita, seguindo um conjunto de regras e se movendo entre estados fixos.

A visão de Turing apresentava uma abordagem mecânica da computação, complementando a perspectiva simbólica de Church. Church havia provado que algumas funções não são computáveis. O _problema da parada_ é um exemplo famoso:

$$ \text{parada} = \lambda f. \lambda x. \text{("f(x) para?")} $$

Church mostrou que esta função não pode ser expressa no cálculo lambda e, consequentemente, não pode ser computada. Church e Turing, não trabalharam sozinhos.

[Kurt Gödel](https://en.wikipedia.org/wiki/Kurt_G%C3%B6del) contribuiu com a ideia de funções recursivas, uma abordagem algébrica que define a computação por meio de funções primitivas e suas combinações. Ele explorou a computabilidade a partir de uma perspectiva aritmética, usando funções que podem ser definidas recursivamente. Essa visão trouxe uma base numérica e algébrica para o conceito de computabilidade.

Em paralelo, [Emil Post](https://en.wikipedia.org/wiki/Emil_Leon_Post) desenvolveu os sistemas de reescrita, baseados em regras de substituição de strings. Embora menos conhecido, o trabalho de Post foi importante para a teoria das linguagens formais e complementou as outras abordagens, fornecendo uma visão baseada em regras de substituição.

Apesar das diferenças estruturais entre o cálculo lambda, as máquinas de Turing, as funções recursivas e os sistemas de Post, todos esses modelos têm o mesmo poder computacional. Uma função que não for computável em um destes modelos, não o será em todos os outros. E neste ponto temos uma base sólida para a ciência da computação.

## A Inovação de Church: Abstração Funcional

O trabalho de Alonzo Church é estruturado sobre a ideia de _abstração funcional_. Esta abstração permitiu tratar funções como estruturas de primeira classe. Estruturas que podem ser passadas como argumentos, retornadas como resultados e usadas em expressões composta, assim como qualquer outro valor na álgebra tradicional.

No cálculo lambda, uma função é escrita como $\lambda x . E$. Aqui, $\lambda$ indica que é uma função, $x$ é a variável ligada, onde a função será aplicada, e $E$ é o corpo da função. Por exemplo, a função que soma $1$ a um número será escrita como $\lambda x . \, x + 1$. Isso possibilita a manipulação direta de funções, sem a necessidade de linguagens ou estruturas rígidas.

A abstração funcional também criou o conceito de **funções anônimas**. Hoje, muitas linguagens modernas, como Haskell, Lisp, Python e JavaScript, adotam essas funções como parte das ferramentas disponíveis em sua sintaxe. Tais funções são conhecidas como _lambda functions_ ou _arrow functions_.

A abstração funcional possibilitou a criação de operações de combinação, um conceito da lógica combinatória. Estas operações de combinação são representadas na aplicação de combinadores que, por sua vez, definem como combinar funções. No cálculo lambda, e nas linguagens funcionais, os combinadores, como o **combinador Y**, facilitam a prova de conceitos matemáticos ou, permitem acrescentar funcionalidades ao cálculo lambda. O combinador $Y$, por exemplo, permite o uso de recursão em funções. O combinador $Y$, permitiu provar a equivalência entre o Cálculo lambda, a máquina de touring e a recursão de Gödel. Solidificando a noção de computabilidade.

Na notação matemática clássica, as funções são representadas usando símbolos de variáveis e operadores. Por exemplo, uma função quadrática pode ser escrita como:

$$ f(x) = x^2 + 2x + 1 $$

Essa notação é direta e representa um relação matemática entre dois conjuntos. Descrevendo o resultado da aplicação da relação a um dos elementos de un conjunto, encontrando o elemento relacionado no outro. A definição da função não apresenta o processo de computação necessário. O cálculo lambda, por outro lado, descreve um processo de aplicação e transformação de variáveis. Enquanto a Máquina de Turing descreve a computação de forma mecânica, o cálculo lambda foca na transformação de expressões.

Linguagens de programação modernas, como Python ou JavaScript, têm suas próprias formas de representar funções. Por exemplo, em Python, uma função pode ser representada assim:

```python
def f(x):
 return x**2 + 2*x + 1
```

Essa notação equivale à notação matemática clássica, porém permite controle sobre o fluxo de execução e manipulação de dados. As linguagens funcionais representam funções a partir das estruturas do cálculo lambda. Neste caso, funções são tratadas como elementos fundamentais e a aplicação de funções é central. Muitos textos acadêmicos dizem que funções são cidadãos de primeira classe.

**No cálculo lambda, usamos _abstração_ e _aplicação_ para criar e aplicar funções.** Na criação de uma função que soma dois números, escrita como:

$$ \lambda x. \lambda y. \, (x + y) $$

A notação $\lambda$ indica que estamos criando uma função anônima. Essa abstração explícita é menos comum na notação matemática clássica na qual, geralmente definimos funções nomeadas.

## Limitações do Cálculo Lambda e Sistemas Avançados

O cálculo lambda é poderoso. Ele pode expressar qualquer função computável. Mas tem limitações: **não tem tipos nativos**. Tudo é função. Números, booleanos, estruturas de dados - todos são codificados como funções; **Não tem estado mutável**. Cada expressão produz um novo resultado. Não modifica valores existentes. Isso é uma vantagem em alguns cenários, mas agrega complexidade a definição de algoritmos; **não tem controle de fluxo direto**. Loops e condicionais são simulados com funções recursivas. Apesar de ser chamado de _a menor linguagem de programação_ a criação de algoritmos sem controle de fluxo não é natural para programadores, e matemáticos, nativos do mundo imperativo; **pode ser ineficiente**. Codificações como números de Church podem levar a cálculos lentos. Performance nunca foi um objetivo.

Sistemas mais avançados de cálculo lambda abordam essas limitações:

1. **sistemas de tipos**: O cálculo lambda tipado adiciona tipos. O Sistema F permite polimorfismo:

   $$ \Lambda \alpha. \lambda x:\alpha. \, x $$

   Esta função é polimórfica. Funciona para qualquer tipo $\alpha$. Veremos cálculo lambda tipado, quanto ao Sistema F, ainda não tenho certeza.

2. **Efeitos colaterais**: O cálculo lambda com efeitos colaterais permite mutação e I/O:

   $$ \text{let} \; x = \text{ref} \; 0 \; \text{in} \; x := !x + 1 $$

   Esta expressão cria uma referência mutável e a incrementa.

3. **Construções imperativas**: Algumas extensões adicionam estruturas de controle diretas:

   $$ \text{if} \; b \; \text{then} \; e_1 \; \text{else} \; e_2 $$

   Este é um condicional direto, não codificado como função.

4. **Otimizações**: Implementações eficientes usam representações otimizadas:

   $$ 2 + 3 \rightarrow 5 $$

 Este cálculo usa aritmética tradicional, não números de Church.

Estas extensões agregam funcionalidade e transformam o cálculo lambda em uma ferramenta matemática mais flexível. Facilitam a criação de algoritmos e utilização do cálculo lambda na criação de linguagens funcionais.

## Notação e Convenções

O cálculo lambda usa uma notação específica para representar funções e operações. Aqui estão os elementos fundamentais:

### Símbolos Básicos

- $\lambda$: Indica a definição de uma função. Por exemplo, $\lambda x. x + 1$ define uma função que adiciona 1 ao seu argumento.

- $x, y, z$: Letras minúsculas geralmente representam variáveis.

- $M, N$: Letras maiúsculas geralmente representam termos ou expressões lambda.

- $(M \, N)$: Parênteses indicam a aplicação de uma função $M$ a um argumento $N$.

- $\rightarrow$: Usado para indicar redução ou avaliação. Por exemplo, $(\lambda x. x + 1) \, 2 \rightarrow 3$.

- $\rightarrow_\beta$: Indica especificamente uma redução beta.

- $\equiv$: Indica equivalência entre termos.

- $\Gamma$: Representa um contexto de tipagem, um conjunto de associações entre variáveis e seus tipos.

- $\vdash$: Usado em julgamentos de tipo. Por exemplo, $\Gamma \vdash M : A$ significa que no contexto $\Gamma$, o termo $M$ tem tipo $A$.

### Convenções de Escrita

- Abstrações múltiplas podem ser abreviadas: $\lambda x. \lambda y. M$ pode ser escrito como $\lambda x y. M$.

- Aplicações são associativas à esquerda: $M \, N \, P$ significa $((M \, N) \, P)$.

- Tipos de função são associativos à direita: $A \rightarrow B \rightarrow C$ significa $A \rightarrow (B \rightarrow C)$.

### Notações Especiais

- $[N/x]M$: Denota a substituição de todas as ocorrências livres de $x$ em $M$ por $N$.

- $FV(M)$: Representa o conjunto de variáveis livres no termo $M$.

- $\alpha$-conversão: Renomeação de variáveis ligadas, denotada por $\equiv_\alpha$.

- $\eta$-conversão: Expressa a extensionalidade de funções, denotada por $\rightarrow_\eta$.

Estas notações e convenções formam a base da linguagem formal do cálculo lambda, permitindo a expressão precisa de funções e suas transformações.


## Convenção de Nomes e Variáveis Livres e Ligadas

No cálculo lambda, as variáveis têm escopo léxico. O escopo é determinado pela estrutura sintática do termo, não pela ordem de avaliação.

Uma variável é **ligada** quando aparece dentro do escopo de uma abstração que a introduz. Por exemplo:

- Em $\lambda x.\lambda y.x \, y$, tanto $x$ quanto $y$ estão ligadas.
- Em $\lambda x.(\lambda x. \, x) \, x$, ambas as ocorrências de $x$ estão ligadas, mas a ocorrência interna (no termo $\lambda x. \, x$) "esconde" a externa.

**Uma variável é livre quando não está ligada por nenhuma abstração**. Por exemplo:

- Em $\lambda x. \, x \, y$, $x$ está ligada, mas $y$ está livre.
- Em $(\lambda x. \, x) \, y$, $y$ está livre.

O conjunto de variáveis livres de um termo $E$, denotado por $FV(E)$, pode ser definido recursivamente:

$$
\begin{align*}
FV(x) &= \{x\} \\
FV(\lambda x. \, E) &= FV(E) \setminus \{x\} \\
FV(E \, N) &= FV(E) \cup FV(N)
\end{align*}
$$

Uma convenção importante no cálculo lambda é que podemos renomear variáveis ligadas sem alterar o significado do termo, desde que não capturemos variáveis livres. **Esta operação é chamada de $\alpha$-conversão**. Por exemplo:

$$\lambda x.\lambda y.x \, y \to_\alpha \lambda z.\lambda w.z \, w$$

Devemos ter cuidado para não capturar variáveis livres:

$$\lambda x. \, x \, y \neq_\alpha \lambda y. \, y \, y$$

No segundo termo, a variável livre $y$ foi capturada, o que altera o significado do termo.

## Diferença entre abstração e aplicação

A abstração e a aplicação são os dois mecanismos fundamentais do cálculo lambda. Cada um tem um papel distinto. **A abstração $ \lambda x. \, E$ define uma função**. Aqui, $x$ é o parâmetro e $E$ é o corpo da função. Por exemplo:

- $ \lambda x. \, x + 5$ define uma função que soma $5$ ao seu argumento.

- $\lambda f. \lambda x. \, f \, (f \, x)$ define uma função que aplica o primeiro argumento duas vezes ao segundo.

**A abstração cria funções no cálculo lambda**. A aplicação $M \, N$ aplica uma função a um argumento. Aqui, $M$ é a função e $N$ é o argumento. Por exemplo:

- $ (\lambda x. \, x + 5) \, 3$ aplica a função $ \lambda x. \, x + 5$ ao valor $3$.

- $(\lambda f. \lambda x. \, f \, (f \, x)) \, (\lambda y. \, y * 2) \, 3$ aplica a função de composição dupla à função de duplicação e ao número $3$.

## Exercícios
  
**1**: Escreva uma função lambda para representar a identidade, que retorna o próprio argumento.

**Solução**: a função identidade é simplesmente: $\lambda x. x$$, essa função retorna o argumento $x$.

**2**: Escreva uma função lambda que representa uma constante, sempre retornando o número $5$, independentemente do argumento.

**Solução**: a função constante pode ser representada por: $\lambda x. 5$, uma função que sempre retorna $5$, independentemente de $x$.

**3**: Dado $\lambda x. x + 2$, aplique a função ao número $3$.

**Solução**: substituímos $x$ por $3$ e teremos: $(\lambda x. x + 2) 3 = 3 + 2 = 5$

**4**: Simplifique a expressão $(\lambda x. \lambda y. x)(5)(6)$.

**Solução**: primeiro, aplicamos a função ao valor $5$, o que resulta na função $\lambda y. 5$. Agora, aplicamos essa nova função ao valor $6$:

$$
(\lambda y. 5) 6 = 5
$$

O resultado final é $5$.

**5**: Simplifique a expressão $(\lambda x. x)(\lambda y. y)$.

**Solução**: aplicamos a função $\lambda x. x$ à função $\lambda y. y$:

$$
(\lambda x. x)(\lambda y. y) = \lambda y. y
$$

A função $\lambda y. y$ é a identidade e o resultado final é a própria função identidade.

**6**: aplique a função $\lambda x. \lambda y. x + y$ aos valores $3$ e $4$.

**Solução**: Aplicamos a função a $3$ e depois a $4$:

$$
(\lambda x. \lambda y. x + y) 3 = \lambda y. 3 + y
$$

Agora aplicamos $4$:

$$
(\lambda y. 3 + y) 4 = 3 + 4 = 7
$$

O resultado final é $7$.

**7**: A função $\lambda x. \lambda y. x$ é uma função de primeira ordem ou segunda ordem?

**Solução**: a função $\lambda x. \lambda y. x$ é uma função de segunda ordem, pois é uma função que retorna outra função.

**8**: Defina uma função lambda que troca a ordem dos argumentos de uma função de dois argumentos.

**Solução**: essa função pode ser definida como:

$$
\lambda f. \lambda x. \lambda y. f \, y \, x
$$

Ela aplica a função $f$ aos argumentos $y$ e $x$, trocando a ordem.

**9**: Dada a função $\lambda x. x \, x$, por que ela não pode ser aplicada a si mesma diretamente?

**Solução**: se aplicarmos $\lambda x. x \, x$ a si mesma, teremos:

$$
(\lambda x. x \, x)(\lambda x. x \, x)
$$

Isso resultaria em uma aplicação infinita da função a si mesma, o que leva a um comportamento indefinido ou a um erro de recursão infinita.

**10**: Aplique a função $\lambda x. x \, x$ ao valor $2$.

**Solução**: substituímos $x$ por $2$:

$$
(\lambda x. x \, x) 2 = 2 \times 2 = 4
$$

O resultado final é $4$.

# Sintaxe e Semântica

O cálculo lambda usa uma notação simples para definir e aplicar funções. Ele se baseia em três elementos principais: _variáveis, abstrações e aplicações_.

**As variáveis representam valores que podem ser usados em expressões. Uma variável é um símbolo que pode ser substituído por um valor ou outra expressão**. Por exemplo, $x$ é uma variável que pode representar qualquer valor.

**A abstração é a definição de uma função**. No cálculo lambda, uma abstração é escrita usando a notação $\lambda$, seguida de uma variável, um ponto e uma expressão. Por exemplo:

$$ \lambda x. \, x^2 + 2x + 1 $$

**Aqui, $\lambda x.$ indica que estamos criando uma função de $x$**. A expressão $x^2 + 2x + 1$ é o corpo da função. A abstração define uma função anônima que pode ser aplicada a um argumento.

**A aplicação é o processo de usar uma função em um argumento**. No cálculo lambda, representamos a aplicação de uma função a um argumento colocando-os lado a lado. Por exemplo, se tivermos a função $ \lambda x. \, x + 1\,$ e quisermos aplicá-la ao valor $2$, escrevemos:

$$ (\lambda x. \, x + 1) \, 2 $$

**O resultado da aplicação é a substituição da variável $x$ pelo valor $2$,** resultando em $2 + 1$ equivalente a $3$. Outros exemplos interessantes são:

- **Identidade**: A função identidade, que retorna o próprio valor, é escrita como $ \lambda x. \, x$.

- **Soma de Dois Números**: Uma função que soma dois números pode ser escrita como $ \lambda x. \lambda y. \, (x + y)$. Temos duas abstrações $\lambda x$ e $\lambda y$, com duas variáveis. Logo, $ \lambda x. \lambda y. \, (x + y)$ precisa ser aplicada a dois argumentos. Tal como: $ \lambda x. \lambda y. \, (x + y) 3 4$.

Esses elementos básicos, _variáveis, abstração e aplicação_, formam a base do cálculo lambda. Eles permitem definir e aplicar funções de forma simples sem a necessidade de nomes ou símbolos adicionais.


## Estrutura Sintática - Gramática

O cálculo lambda é um sistema formal para representar computação baseado na abstração de funções e sua aplicação. Sua sintaxe é simples e poderosa em termos de expressão. Enfatizando a simplicidade. Tudo é uma expressão (ou termo) e existem apenas três tipos de termos:

1. **Variáveis**: Representadas por letras minúsculas como $x$, $y$, $z$. As variáveis não possuem valor intrínseco, como em linguagens como Python ou C++. Atuam como espaços reservados para entradas potenciais de funções.

2. **Aplicação**: A aplicação $(M \, N)$ indica a aplicação da função $M$ ao argumento $N$. A aplicação é associativa à esquerda, então $M \, N \, P$ é interpretado como $((M \, N) \, P)$.

3. **Abstração**: A abstração $ (\lambda x. \, E)$ representa uma função que tem $x$ como parâmetro e $E$ como corpo. O símbolo $\lambda$ indica que estamos definindo uma função. Por exemplo, $ (\lambda x. \, x)$ é a função identidade.

**A abstração é central no cálculo lambda**. Ela permite criar funções anonimamente, sem a necessidade de nomeá-las.

**Um conceito importante relacionado à abstração é a distinção entre variáveis livres e ligadas**:

- Uma variável é **ligada** se aparece no escopo de uma abstração lambda que a define. Em $(\lambda x. \, x y)$, $x$ é uma variável ligada.

- Uma variável é **livre** se não está ligada por nenhuma abstração. No exemplo anterior, $y$ é uma variável livre.

A distinção entre variáveis livres e ligadas é indispensável para entender a substituição no cálculo lambda. A substituição é a base do processo de computação no cálculo lambda. O poder do cálculo lambda está na forma como esses elementos simples podem ser combinados para expressar operações complexas como valores booleanos, estruturas de dados e até mesmo recursão usando apenas esses os conceitos básicos, _variáveis, abstração e aplicação_, e a ligação de variáveis. Formalmente, podemos definir a sintaxe do cálculo lambda usando uma gramática na [Forma de Backus-Naur](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form) (BNF) como:

$$
\begin{align*}
\text{termo} &::= \text{variável} \\
&\, |\, \text{constante} \\
&\, |\, \lambda . \text{variável}. \, \text{termo} \\
&\, |\, \text{termo}\, \text{termo} \\
&\, |\, (\text{termo})
\end{align*}
$$

Uma forma de facilitar o entendimento de abstrações e aplicações é pensar em um termo como uma árvore, cuja forma corresponde à maneira como o termo corresponde à gramática. Podemos chamar este tipo de árvore de árvore sintática. Esta árvore será a representação, em um grafo deste termo. em formato de árvore. Para um dado termo $s$ está árvore terá vértices rotulados por $λx$ ou $@$, e folhas rotuladas por variáveis. Ela é dada indutivamente por:

1. A árvore de construção de uma variável $x$ é uma única folha, rotulada por $x$.

2. A árvore de construção de uma abstração $λx.s$ consiste em um nó rotulado por $λx$ com uma única subárvore, que é a árvore de construção de $s$.

3. A árvore de construção de uma aplicação $s\,t# consiste em um nó rotulado por $@$ com duas subárvores: a subárvore esquerda é a árvore de construção de $s$ e a subárvore direita é a árvore de construção de $t$.

Por exemplo, a árvore de construção do termo $λxy.xλz.yz$ será: 

 $$
\begin{array}{c}
\lambda x \\
\downarrow \\
\lambda y \\
\downarrow \\
\diagup \quad \diagdown \\
\begin{array}{cc}
\quad x \quad & \quad \lambda z \quad \\
& \downarrow \\
& \begin{array}{cc}
\, y & \, z
\end{array}
\end{array}
\end{array}
$$

Neste texto, vamos dar prioridade a derivação gramatical, e evitaremos as árvores. Contudo, como as árvores são excelentes para visualização e entendimento, vamos ver mais dois exemplos.

**Exemplo 1: Representação da abstração $\lambda x.\ x\ x$**

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

**Exemplo 2**: Representação da aplicação $(\lambda x.\ x + 1)\ 2$

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

Como vimos, a gramática é simples, poucas regras de produção e poucos símbolos. Por outro lado, a semântica do cálculo lambda pode ser dividida em **semântica operacional** e **semântica denotacional**.

## Semântica Operacional

A semântica operacional descreve como as expressões são avaliadas passo a passo. A principal regra é a **redução beta ($\beta$-redução)**. Ela ocorre quando uma função é aplicada a um argumento. A redução beta substitui a variável ligada no corpo da função pelo argumento fornecido:

$$(\lambda x.\, e_1)\ e_2\ \rightarrow\ e_1[x := e_2]$$

Isso significa que aplicamos a função $\lambda x.\, e_1$ ao argumento $e_2$, substituindo $x$ por $e_2$ em $e_1$.

**Exemplo:**

$$(\lambda x.\, x^2)\ 3\ \rightarrow\ 3^2$$

Existem duas estratégias principais para realizar a redução beta:

1. **Ordem normal**: Reduzimos a aplicação mais à esquerda e mais externa primeiro. Essa estratégia sempre encontra a forma normal, se existir.

   **Exemplo:**

   $$(\lambda x.\, (\lambda y.\, y + x)\ 2)\ (3 + 4)$$

   Não reduzimos $3 + 4$ imediatamente. Aplicamos a função externa:

   $$(\lambda x.\, (\lambda y.\, y + x)\ 2)\ 7$$

   Substituímos $x$ por $7$ em $(\lambda y.\, y + x)\ 2$:

   $$(\lambda y.\, y + 7)\ 2$$

   Aplicamos a função interna:

   $$2 + 7 \rightarrow 9$$

2. **Ordem aplicativa**: Avaliamos primeiro as subexpressões (argumentos) antes de aplicar a função.

   **Exemplo:**

   $$(\lambda x.\, (\lambda y.\, y + x)\ 2)\ (3 + 4)$$

   Avaliamos $3 + 4$:

   $$(\lambda x.\, (\lambda y.\, y + x)\ 2)\ 7$$

   Substituímos $x$ por $7$:

   $$(\lambda y.\, y + 7)\ 2$$

   Avaliamos $2 + 7$:

   $$9$$

   Além da redução beta, existem as seguintes conversões:

- **$\alpha$-conversão**: Renomeia variáveis ligadas para evitar conflitos.

 **Exemplo:**

 $$\lambda x.\, x + 1 \rightarrow \lambda y.\, y + 1$$

- **$\eta$-conversão**: Captura a equivalência entre funções que produzem os mesmos resultados.

 **Exemplo:**

 $$\lambda x.\, f(x) \rightarrow f$$

Essas regras garantem que a avaliação seja consistente. Por fim, mas não menos importante, o **Teorema de Church-Rosser** assegura que, **se uma expressão pode ser reduzida de várias maneiras então todas chegarão à mesma forma normal, se existir**[^cita5].

## Substituição

A substituição é a operação central no cálculo lambda. Ela funciona substituindo uma variável livre por um termo, e sua formalização evita a captura de variáveis, garantindo que ocorra de forma correta. A substituição é definida de maneira recursiva:

1. $[N/x]x = N$
2. $[N/x]y = y$, se $x
eq y$
3. $[N/x](M_1 M_2) = ([N/x]M_1)([N/x]M_2)$
4. $[N/x](\lambda y.M) = \lambda y.([N/x]M)$, se $x \neq y$ e $y \notin FV(N)$

Aqui, $FV(N)$ é o conjunto de variáveis livres de $N$. A condição $y \notin FV(N)$ é necessária para evitar a captura de variáveis livres. Considere, por exemplo:

$$[y/x](\lambda y. \, x) \neq \lambda y. \, y$$

Nesse caso, uma substituição direta capturaria a variável livre $y$, alterando o significado do termo. Para evitar isso, utilizamos a **substituição com evasão de captura**. Considere:

$$[y/x](\lambda y. \, x) = \lambda z.\, [y/x]([z/y]x) = \lambda z.\, y$$

Renomeamos a variável ligada $y$ para $z$ antes de realizar a substituição, evitando a captura da variável livre $y$.

Outro exemplo relevante:

$$[z/x](\lambda z.\, x) \neq \lambda z. \, z$$

Se fizermos a substituição diretamente, a variável $z$ será capturada, mudando o significado do termo. A solução correta é renomear a variável ligada antes da substituição:

$$[z/x](\lambda z.\, x) = \lambda w.\, [z/x]([w/z]x) = \lambda w.\, z$$

Este processo garante que a variável livre $z$ não seja capturada pela abstração $\lambda z$.

 **Exemplo 1**: Substituição direta sem captura de variável livre

 $$[a/x](x + y) = a + y$$

 Aqui, substituímos a variável $x$ pelo termo $a$, resultando em $a + y$. Não há risco de captura, pois $y$ não está ligada.

 **Exemplo 2**: Substituição direta de variáveis livres

 $$[b/x](x \, z) = b \, z$$

 Nesse exemplo, substituímos $x$ por $b$, resultando em $b \, z$. A variável $z$ permanece livre.

 **Exemplo 3**: Evasão de captura com renomeação

 $$[y/x](\lambda y.\, x) = \lambda z.\, [y/x]([z/y]x) = \lambda z.\, y$$

 Renomeamos a variável ligada $y$ para $z$ antes de realizar a substituição, evitando que a variável livre $y$ seja capturada.

 **Exemplo 4**: Evasão de captura para preservar significado

 $$[w/x](\lambda w.\, x) = \lambda v.\, [w/x]([v/w]x) = \lambda v.\, w$$

 Aqui, renomeamos a variável ligada $w$ para $v$ antes de fazer a substituição, garantindo que a variável livre $w$ não seja capturada.

## Semântica Denotacional

Na semântica denotacional, cada expressão é mapeada para um objeto matemático. Isso fornece uma interpretação mais abstrata da computação. O domínio é construído como um conjunto de funções. O significado de uma expressão é definido por sua interpretação nesse domínio.

A interpretação denotacional é definida por:

- $[x]_\rho = \rho(x)$

- $[\lambda x.\, e]_\rho = f$, onde $f(v) = [e]_{\rho[x \mapsto v]}$

- $[e_1\ e_2]_\rho = [e_1]_\rho([e_2]_\rho)$

**Exemplo:**

Para a expressão $(\lambda x.\, x + 1)\ 2$, interpretamos $\lambda x.\, x + 1$ como uma função que adiciona 1. Aplicando a 2, obtemos 3.

A semântica denotacional permite pensar em expressões lambda como funções matemáticas. Já a semântica operacional foca nos passos da computação.

>Observe que a **Semântica Operacional** é geralmente mais adequada para descrever a execução passo a passo de linguagens que usam passagem por referência, pois permite capturar facilmente como os estados mudam durante a execução. Por outro lado, a **Semântica Denotacional** é mais alinhada com linguagens puras, que preferem passagem por cópia, evitando efeitos colaterais e garantindo que o comportamento das funções possa ser entendido em termos de funções matemáticas.
>
>Existe uma conexão direta entre a forma como a semântica de uma linguagem é modelada e o mecanismo de passagem de valor que a linguagem suporta. Linguagens que favorecem efeitos colaterais tendem a ser descritas de forma mais natural por semântica operacional, enquanto aquelas que evitam efeitos colaterais são mais bem descritas por semântica denotacional.
>
>No caso do cálculo lambda, a semântica denotacional é preferida. O cálculo lambda é uma linguagem puramente funcional sem efeitos colaterais. A semântica denotacional modela suas expressões como funções matemáticas. Isso está em alinhamento com a natureza do cálculo lambda. Embora a semântica operacional possa descrever os passos de computação, a semântica denotacional fornece uma interpretação matemática abstrata adequada para linguagens que evitam efeitos colaterais.

## Exercícios de Semântica Denotacional

**1**: Dada a função lambda $ \lambda x. \, x + 2 $, aplique-a ao valor 5 e calcule o resultado.

 **Solução:**
 Aplicando a função ao valor $5$, temos:

 $$(\lambda x. \, x + 2) \, 5 = 5 + 2 = 7$$

**2**: Escreva uma expressão lambda que represente a função $f(x, y) = x^2 + y^2$, e aplique-a aos valores $x = 3$ e $y = 4$.

 **Solução:**
 A função pode ser representada como $\lambda x. \lambda y. \, x^2 + y^2$. Aplicando $x = 3$ e $y = 4$:

 $$ (\lambda x. \lambda y. \, x^2 + y^2) \, 3 \, 4 = 3^2 + 4^2 = 9 + 16 = 25 $$

**3**: Crie uma expressão lambda para a função identidade $I(x) = x$ e aplique-a ao valor $10$.

 **Solução:**
 A função identidade é $\lambda x. \, x$. Aplicando ao valor 10:

 $$(\lambda x. \, x) \, 10 = 10$$

**4**: Defina uma função lambda que aceita um argumento $x$ e retorna o valor $x^3 + 1$. Aplique a função ao valor $2$.

 **Solução:**
 A função lambda é $\lambda x. \, x^3 + 1$. Aplicando ao valor 2:

 $$(\lambda x. \, x^3 + 1) \, 2 = 2^3 + 1 = 8 + 1 = 9$$

**5**: Escreva uma função lambda que represente a soma de dois números, ou seja, $f(x, y) = x + y$, e aplique-a aos valores $x = 7$ e $y = 8$.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, x + y$. Aplicando $x = 7$ e $y = 8$:

 $$(\lambda x. \lambda y. \, x + y) \, 7 \, 8 = 7 + 8 = 15$$

**6**: Crie uma função lambda para a multiplicação de dois números, ou seja, $f(x, y) = x \cdot y$, e aplique-a aos valores $x = 6$ e $y = 9$.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, x \cdot y$. Aplicando $x = 6$ e $y = 9$:

 $$(\lambda x. \lambda y. \, x \cdot y) 6 9 = 6 \cdot 9 = 54$$

**7**: Dada a expressão lambda $\lambda x. \lambda y. \, x^2 + 2xy + y^2$, aplique-a aos valores $x = 1$ e $y = 2$ e calcule o resultado.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, x^2 + 2xy + y^2$. Aplicando $x = 1$ e $y = 2$:

 $$(\lambda x. \lambda y. \, x^2 + 2xy + y^2) 1 2 = 1^2 + 2(1)(2) + 2^2 = 1 + 4 + 4 = 9$$

**8**: Escreva uma função lambda que aceite dois argumentos $x$ e $y$ e retorne o valor de $x - y$. Aplique-a aos valores $x = 15$ e $y = 5$.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, x - y$. Aplicando $ x = 15 $ e $ y = 5 $:

 $$(\lambda x. \lambda y. \, x - y) \, 15 \, 5 = 15 - 5 = 10$$

**9**: Defina uma função lambda que represente a divisão de dois números, ou seja, $ f(x, y) = \frac{x}{y} $, e aplique-a aos valores $x = 20$ e $y = 4$.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, \frac{x}{y}$. Aplicando $x = 20$ e $y = 4$:

 $$(\lambda x. \lambda y. \, \frac{x}{y}) \, 20 \, 4 = \frac{20}{4} = 5$$

**10**: Escreva uma função lambda que calcule a função $f(x, y) = x^2 - y^2$, e aplique-a aos valores $x = 9$ e $y = 3$.

 **Solução:**
 A função lambda é $\lambda x. \lambda y. \, x^2 - y^2$. Aplicando $x = 9$ e $y = 3$:

 $$(\lambda x. \lambda y. \, x^2 - y^2) \, 9 \, 3 = 9^2 - 3^2 = 81 - 9 = 72$$

## Redução (Alfa-Redução)

A redução $\alpha$ (ou $\alpha$-conversão) é o processo de renomear variáveis ligadas. Isso garante que funções que diferem apenas nos nomes de suas variáveis ligadas sejam tratadas como equivalentes. Formalmente, temos:

$$\lambda x.M \to_\alpha \lambda y.[y/x]\,E$$

Aqui, $[y/x]\,E$ significa substituir todas as ocorrências livres de $x$ em $M$ por $y$, onde $y$ não ocorre livre em $E$. Essa condição é essencial para evitar a captura de variáveis livres.

**Exemplo**:

$$\lambda x.\lambda y.x \, y \to_\alpha \lambda z.\lambda y.z \, y \to_\alpha \lambda w.\lambda v.w \, v$$

A redução $\alpha$ é importante por:

1. **Evitar conflitos de nomes** durante operações como a redução $\Beta$, garantindo que variáveis ligadas não interfiram com variáveis livres.

2. **Uniformizar funções** que diferem apenas nos nomes de suas variáveis ligadas, simplificando a identificação de equivalências semânticas.

3. **Ser a base para escopos léxicos** em linguagens de programação, onde renomear variáveis ligadas assegura a correta correspondência entre variáveis e seus valores.

A redução $\alpha$ está intimamente ligada ao conceito de escopo léxico em linguagens de programação. **O escopo léxico garante que o significado de uma variável seja determinado por sua posição no texto do programa, não pela ordem de execução.** A redução $\alpha$ assegura que podemos renomear variáveis sem alterar o comportamento do programa, desde que respeitemos as regras de escopo.

Em linguagens funcionais como Haskell ou OCaml, a redução $\alpha$ ocorre implicitamente. Por exemplo, as seguintes definições são tratadas como equivalentes em Haskell:

```haskell
f = \x -> x + 1
f = \y -> y + 1
```

Ambas representam a mesma função, e a renomeação da variável não altera seu comportamento.

### Exercícios de Redução Alfa no Cálculo Lambda

**1**: Aplique a redução alfa para renomear a variável da expressão $\lambda x. \, x + 2$ para $z$.

 **Solução:** Substituímos a variável ligada $x$ por $z$:

 $$\lambda x. \, x + 2 \to_\alpha \lambda z. \, z + 2$$

**2**: Renomeie a variável ligada $ y $ na expressão $\lambda x. \lambda y. \, x + y$ para $w$.

 **Solução:** A redução alfa renomeia $y$ para $w$:

 $$ \lambda x. \lambda y. \, x + y \to_\alpha \lambda x. \, \lambda w. \, x + w $$

**3**: Aplique a redução alfa para renomear a variável $z$ na expressão $\lambda z. \, z^2$ para $a$.

 **Solução:** Substituímos $z$ por $a$:

 $$ \lambda z. \, z^2 \to_\alpha \lambda a. a^2 $$

**4**: Renomeie a variável $f$ na expressão $\lambda f. \lambda x. \, f(x)$ para $g$, utilizando a redução alfa.

 **Solução:** Substituímos $f$ por $g$:

 $$ \lambda f. \lambda x. \, f(x) \to_\alpha \lambda g. \lambda x. \, g(x) $$

**5**: Na expressão $\lambda x. \, (\lambda x. \, x + 1) \, x$, renomeie a variável ligada interna $x$ para $z$.

 **Solução:** Substituímos a variável ligada interna $x$ por $z$:

 $$ \lambda x. \, (\lambda x. \, x + 1) x \to_\alpha \lambda x. \, (\lambda z. \, z + 1) x $$

**6**: Aplique a redução alfa na expressão $\lambda x. \lambda y. \, x \cdot y$ renomeando $x$ para $a$ e $y$ para $b$.

 **Solução:** Substituímos $x$ por $a$ e $y$ por $b$:

 $$\lambda x. \lambda y. \, x \cdot y \to_\alpha \lambda a. \lambda b. a \cdot b$$

**7**: Renomeie a variável ligada $y$ na expressão $\lambda x. \, (\lambda y. \, y + x)$ para $t$.

 **Solução:** Substituímos $y$ por $t$:

 $$\lambda x. \, (\lambda y. \, y + x) \to_\alpha \lambda x. \, (\lambda t. t + x)$$

**8**: Aplique a redução alfa na expressão $\lambda f. \lambda x. \, f(x + 2)$ renomeando $f$ para $h$.

 **Solução:** Substituímos $f$ por $h$:

 $$\lambda f. \lambda x. \, f(x + 2) \to_\alpha \lambda h. \lambda x. \, h(x + 2)$$

**9**: Na expressão $\lambda x. \, (\lambda y. \, x - y)$, renomeie a variável $y$ para $v$ utilizando a redução alfa.

 **Solução:** Substituímos $y$ por $v$:

 $$\lambda x. \, (\lambda y. \, x - y) \to_\alpha \lambda x. \, (\lambda v. \, x - v)$$

**10**: Aplique a redução alfa na expressão $\lambda x. \, (\lambda z. \, z + x) \, z$, renomeando $z$ na função interna para $w$.

 **Solução:** Substituímos $z$ na função interna por $w$:

 $$ \lambda x. \, (\lambda z. \, z + x) \, z \to_\alpha \lambda x. \, (\lambda w. w + x) \, z$$

### Redução Alfa e Substituição

A redução alfa é intimamente ligada à substituição. Muitas vezes precisamos renomear variáveis antes de realizar substituições para evitar conflitos de nomes. Por exemplo:

$$(\lambda x.\, \lambda y.\, x)\, y$$

Para reduzir este termo corretamente, renomeamos a variável $y$ na abstração interna, evitando conflito com o argumento:

$$(\lambda x.\, \lambda y.\, x)y\to_\alpha (\lambda x.\, \lambda z.\, x)y \rightarrow_\beta \lambda z.\, y$$

Sem a redução alfa, teríamos obtido incorretamente $\lambda y. \, y$, o que mudaria o comportamento da função.

A redução alfa é essencial para evitar ambiguidades, especialmente em casos onde variáveis ligadas compartilham nomes com variáveis livres.

### Convenções Práticas: Convenção de Variáveis de Barendregt

Na prática, a redução alfa é aplicada implicitamente durante as substituições. A _convenção de variável de Barendregt_ estabelece que todas as variáveis ligadas em um termo devem ser distintas entre si e das variáveis livres. Isso elimina a necessidade de renomeações explícitas frequentes e simplifica a manipulação de termos no cálculo lambda.

Com essa convenção, podemos simplificar a definição de substituição para:

$$[N/x](\lambda y.\, M) = \lambda y.\, ([N/x]M)$$

assumindo implicitamente que $y$ será renomeado, se necessário. Ou seja, a convenção de Barendregt nos permite tratar termos alfa-equivalentes como idênticos. Por exemplo, podemos considerar os seguintes termos como iguais:

$$\lambda x.\, \lambda y.\, x y = \lambda a.\, \lambda b.\, a b$$

Isso simplifica muito a manipulação de termos lambda, pois não precisamos nos preocupar constantemente com conflitos de nomes.

### Exercícios de Substituição, Redução Alfa e Convenção de Barendregt

**1**: Aplique a substituição $[y/x]x$ e explique o processo.

 **Solução:** A substituição de $x$ por $y$ é direta:

 $$ [y/x]x = y $$

**2**: Aplique a substituição $[y/x] (\lambda x. \, x + 1)$ e explique por que a substituição não ocorre.

 **Solução:** A variável $x$ está ligada dentro da abstração $ \lambda x $, então a substituição não afeta o corpo da função:

 $$ [y/x] (\lambda x. \, x + 1) = \lambda x. \, x + 1 $$

**3**: Aplique a substituição $[z/x](\lambda z. \, x + z)$. Utilize redução alfa para evitar captura de variáveis.

 **Solução:** A substituição direta causaria captura de variáveis. Aplicamos a redução alfa para renomear $z$ antes de fazer a substituição:

 $$ [z/x](\lambda z. \, x + z) = \lambda w. z + w $$

**4**: Considere a expressão $ (\lambda x. \lambda y. \, x + y) z $. Aplique a substituição $ [z/x] $ e explique a necessidade de redução alfa.

 **Solução:** Como $x$ não está ligada, podemos realizar a substituição sem necessidade de alfa. A expressão resultante é:

 $$ [z/x] (\lambda x. \lambda y. \, x + y) = \lambda y. \, z + y $$

**5**: Aplique a substituição $ [z/x](\lambda z. \, x + z) $ sem realizar a redução alfa. O que ocorre?

 **Solução:** Se aplicarmos diretamente a substituição sem evitar a captura, a variável $z$ será capturada e a substituição resultará incorretamente em:

 $$ [z/x](\lambda z. \, x + z) = \lambda z. \, z + z $$

**6**: Considere a expressão $ (\lambda x. \lambda y. \, x + y) (\lambda z. \, z \cdot z) $. Aplique a substituição $ [(\lambda z. \, z \cdot z)/x] $ e use a convenção de Barendregt.

 **Solução:** Aplicamos a substituição:

 $$ [(\lambda z. \, z \cdot z)/x] (\lambda x. \lambda y. \, x + y) = \lambda y. \, (\lambda z. \, z \cdot z) + y $$

 Com a convenção de Barendregt, variáveis ligadas não entram em conflito.

**7**: Aplique a redução alfa na expressão $ \lambda x. \lambda y. \, x + y $ para renomear $ x $ e $ y $ para $ a $ e $ b $, respectivamente, e aplique a substituição $ [3/a] $.

**Solução:** Primeiro, aplicamos a redução alfa:

 $$ \lambda x. \lambda y. \, x + y \to_\alpha \lambda a. \lambda b. a + b $$

 Agora, aplicamos a substituição:

 $$ [3/a](\lambda a. \lambda b. a + b) = \lambda b. 3 + b $$

**8**: Aplique a convenção de Barendregt na expressão $ \lambda x. \, (\lambda x. \, x + 1) x $ antes de realizar a substituição $ [y/x] $.

 **Solução:** Aplicando a convenção de Barendregt, renomeamos a variável ligada interna para evitar conflitos:

 $$ \lambda x. \, (\lambda x. \, x + 1) x \to_\alpha \lambda x. \, (\lambda z. \, z + 1) x $$

 Agora, aplicamos a substituição:

 $$ [y/x] (\lambda x. \, (\lambda z. \, z + 1) x) = \lambda x. \, (\lambda z. \, z + 1) y $$

**9**: Aplique a redução alfa na expressão $ \lambda x. \, (\lambda y. \, x + y) $, renomeando $ y $ para $ z $, e depois aplique a substituição $ [5/x] $.

 **Solução:** Primeiro, aplicamos a redução alfa:

 $$ \lambda x. \, (\lambda y. \, x + y) \to_\alpha \lambda x. \, (\lambda z. \, x + z) $$

 Agora, aplicamos a substituição:

 $$ [5/x] (\lambda x. \, (\lambda z. \, x + z)) = \lambda z. 5 + z $$

**10**: Aplique a substituição $ [y/x] (\lambda x. \, x + z) $ e explique por que a convenção de Barendregt nos permite evitar a redução alfa neste caso.

 **Solução:** Como $x$ é ligado e não há conflitos com variáveis livres, a substituição não afeta o termo, e a convenção de Barendregt garante que não há necessidade de renomeação:

 $$ [y/x] (\lambda x. \, x + z) = \lambda x. \, x + z $$

**11**: Considere o termo $ [z/x] (\lambda y. \, x + (\lambda x. \, x + y)) $. Aplique a substituição e a redução alfa se necessário.

 **Solução:** Como há um conflito com a variável $x$ no corpo da função, aplicamos redução alfa antes da substituição:

 $$ \lambda y. \, x + (\lambda x. \, x + y) \to_\alpha \lambda y. \, x + (\lambda w. w + y) $$

 Agora, aplicamos a substituição:

 $$ [z/x] (\lambda y. \, x + (\lambda w. w + y)) = \lambda y. \, z + (\lambda w. w + y) $$

**12**: Aplique a substituição $ [y/x](\lambda z. \, x + z) $ onde $ z \notin FV(y) $, e explique o processo.

 **Solução:** Como não há conflitos de variáveis livres e ligadas, aplicamos a substituição diretamente:

 $$ [y/x](\lambda z. \, x + z) = \lambda z. y + z $$

**13**: Aplique a substituição $ [z/x] (\lambda y. \, x \cdot y) $ onde $ z \in FV(x) $. Utilize a convenção de Barendregt.

 **Solução:** Como $z$ não causa conflito de variáveis livres ou ligadas, aplicamos a substituição diretamente:

 $$ [z/x] (\lambda y. \, x \cdot y) = \lambda y. \, z \cdot y $$

 A convenção de Barendregt garante que não precisamos renomear variáveis.

**14**: Aplique a redução alfa na expressão $ \lambda x. \, (\lambda y. \, x + y) $ e renomeie $ y $ para $ t $, depois aplique a substituição $ [2/x] $.

 **Solução:** Primeiro aplicamos a redução alfa:

 $$ \lambda x. \, (\lambda y. \, x + y) \to_\alpha \lambda x. \, (\lambda t. \, x + t) $$

 Agora, aplicamos a substituição:

 $$ [2/x] (\lambda x. \, (\lambda t. \, x + t)) = \lambda t. 2 + t $$

**15**: Aplique a substituição $ [y/x] (\lambda x. \, x + (\lambda z. \, x + z)) $ e explique por que não é necessário aplicar a redução alfa.

 **Solução:** Como a variável $x$ está ligada e não entra em conflito com outras variáveis, a substituição não altera o termo:

 $$ [y/x] (\lambda x. \, x + (\lambda z. \, x + z)) = \lambda x. \, x + (\lambda z. \, x + z) $$

## Redução Beta

### Teorema de Church-Rosser

O **Teorema de Church-Rosser**, também conhecido como propriedade de confluência, é um resultado fundamental no cálculo lambda. Ele afirma que:

Se um termo $M$ pode ser reduzido para $N_1$ e $N_2$ por sequências de reduções beta, então existe um termo $P$ tal que tanto $N_1$ quanto $N_2$ podem ser reduzidos para $P$.

Formalmente:

Se $M \twoheadrightarrow_\beta N_1$ e $M \twoheadrightarrow_\beta N_2$, então existe $P$ tal que $N_1 \twoheadrightarrow_\beta P$ e $N_2 \twoheadrightarrow_\beta P$.

Onde $\twoheadrightarrow_\beta$ denota zero ou mais reduções beta.

Este teorema tem várias consequências importantes:

1. **Unicidade da Forma Normal**: Se um termo tem uma forma normal, ela é única.

2. **Independência da Estratégia de Redução**: A forma normal de um termo (se existir) não depende da ordem em que as reduções são aplicadas.

3. **Consistência**: Não é possível derivar termos contraditórios no cálculo lambda puro.

#### Exercícios de Redução Beta no Cálculo Lambda

**1**: Aplique a redução beta na expressão $ (\lambda x. \, x + 1) 5 $.

 **Solução:** Aplicamos a substituição de $ x $ por $ 5 $ no corpo da função:

 $$ (\lambda x. \, x + 1) 5 \to\_\beta [5/x](x + 1) = 5 + 1 = 6 $$

**2**: Simplifique a expressão $ (\lambda x. \lambda y. \, x + y) 2 3 $ utilizando a redução beta.

 **Solução:** Primeiro, aplicamos $ 2 $ ao parâmetro $ x $, e depois $ 3 $ ao parâmetro $ y $:

 $$ (\lambda x. \lambda y. \, x + y) 2 3 \to*\beta (\lambda y. \, 2 + y) 3 \to*\beta 2 + 3 = 5 $$

**3**: Aplique a redução beta na expressão $ (\lambda f. \lambda x. \, f(f \, x)) (\lambda y. \, y + 1) 4 $.

 **Solução:** Primeiro aplicamos $ (\lambda y. \, y + 1) $ a $ f $, e depois $ 4 $ a $ x $:

 1. $ (\lambda f. \lambda x. \, f(f \, x)) (\lambda y. \, y + 1) 4 $
 2. $ \to\_\beta (\lambda x. \, (\lambda y. \, y + 1)( (\lambda y. \, y + 1) x)) 4 $
 3. $ \to\_\beta (\lambda y. \, y + 1)( (\lambda y. \, y + 1) 4) $
 4. $ \to\_\beta (\lambda y. \, y + 1)(4 + 1) $
 5. $ \to\_\beta (\lambda y. \, y + 1)(5) $
 6. $ \to\_\beta 5 + 1 = 6 $

**4**: Reduza a expressão $ (\lambda x. \lambda y. \, x \cdot y) 3 4 $ utilizando a redução beta.

 **Solução:** Primeiro aplicamos $ 3 $ a $ x $ e depois $ 4 $ a $ y $:

 $$ (\lambda x. \lambda y. \, x \cdot y) 3 4 \to*\beta (\lambda y. \, 3 \cdot y) 4 \to*\beta 3 \cdot 4 = 12 $$

**5**: Aplique a redução beta na expressão $ (\lambda x. \lambda y. \, x - y) 10 6 $.

 **Solução:** Aplicamos a função da seguinte forma:

 $$ (\lambda x. \lambda y. \, x - y) 10 6 \to*\beta (\lambda y. \, 10 - y) 6 \to*\beta 10 - 6 = 4 $$

**6**: Reduza a expressão $ (\lambda f. f(2)) (\lambda x. \, x + 3) $ utilizando a redução beta.

 **Solução:** Primeiro aplicamos $ (\lambda x. \, x + 3) $ a $ f $, e depois aplicamos $ 2 $ a $ x $:

 $$ (\lambda f. f(2)) (\lambda x. \, x + 3) \to*\beta (\lambda x. \, x + 3)(2) \to*\beta 2 + 3 = 5 $$

**7**: Simplifique a expressão $ (\lambda f. \lambda x. \, f(x + 2)) (\lambda y. \, y \cdot 3) 4 $ utilizando a redução beta.

 **Solução:** Primeiro aplicamos $ (\lambda y. \, y \cdot 3) $ a $ f $ e depois $ 4 $ a $ x $:

 1. $ (\lambda f. \lambda x. \, f(x + 2)) (\lambda y. \, y \cdot 3) 4 $
 2. $ \to\_\beta (\lambda x. \, (\lambda y. \, y \cdot 3)(x + 2)) 4 $
 3. $ \to\_\beta (\lambda y. \, y \cdot 3)(4 + 2) $
 4. $ \to\_\beta (6 \cdot 3) = 18 $

**8**: Aplique a redução beta na expressão $ (\lambda x. \lambda y. \, x^2 + y^2) (3 + 1) (2 + 2) $.

 **Solução:** Primeiro simplificamos as expressões internas e depois aplicamos as funções:

 1. $ (\lambda x. \lambda y. \, x^2 + y^2) (3 + 1) (2 + 2) $
 2. $ \to\_\beta (\lambda x. \lambda y. \, x^2 + y^2) 4 4 $
 3. $ \to\_\beta (\lambda y. \, 4^2 + y^2) 4 $
 4. $ \to\_\beta 16 + 4^2 = 16 + 16 = 32 $

**9**: Reduza a expressão $ (\lambda f. \lambda x. \, f(f(x))) (\lambda y. \, y + 2) 3 $ utilizando a redução beta.

 **Solução:** Aplicamos a função duas vezes ao argumento:

 1. $ (\lambda f. \lambda x. \, f(f(x))) (\lambda y. \, y + 2) 3 $
 2. $ \to\_\beta (\lambda x. \, (\lambda y. \, y + 2)( (\lambda y. \, y + 2) x)) 3 $
 3. $ \to\_\beta (\lambda y. \, y + 2)( (\lambda y. \, y + 2) 3) $
 4. $ \to\_\beta (\lambda y. \, y + 2)(3 + 2) $
 5. $ \to\_\beta (\lambda y. \, y + 2)(5) $
 6. $ \to\_\beta 5 + 2 = 7 $$

**10**: Reduza a expressão $ (\lambda x. \lambda y. \, x - 2 \cdot y) (6 + 2) 3 $ utilizando a redução beta.

 **Solução:** Primeiro simplificamos as expressões e depois aplicamos as funções:

 1. $ (\lambda x. \lambda y. \, x - 2 \cdot y) (6 + 2) 3 $
 2. $ \to\_\beta (\lambda x. \lambda y. \, x - 2 \cdot y) 8 3 $
 3. $ \to\_\beta (\lambda y. \, 8 - 2 \cdot y) 3 $
 4. $ \to\_\beta 8 - 2 \cdot 3 = 8 - 6 = 2 $

## Currying

**Currying** é uma técnica no cálculo lambda em que uma função com múltiplos argumentos é transformada em uma sequência de funções unárias. Cada função aceita um único argumento e retorna outra função que aceita o próximo argumento, até que todos os argumentos sejam fornecidos.

**Em cálculo lambda puro, todas as funções recebem um, e apenas um, argumento. Não me ative a esta regra de forma estrita, para facilitar o entendimento do processo de substituição e aplicação.**

O conceito de **currying** vem do trabalho do matemático [Moses Schönfinkel](https://en.wikipedia.org/wiki/Moses_Sch%C3%B6nfinkel), que iniciou o estudo da lógica combinatória nos anos 1920. Mais tarde, Haskell Curry popularizou e expandiu essas ideias, dando nome à técnica. O cálculo lambda foi amplamente influenciado por esses estudos, tornando o currying uma parte essencial da programação funcional e da teoria dos tipos.

Por exemplo, uma função de dois argumentos $f(x, y)$ pode ser convertida em uma sequência de funções $f'(x)(y)$. Aqui, $f'(x)$ retorna uma nova função que aceita $y$ como argumento. Assim, uma função que requer múltiplos parâmetros pode ser aplicada parcialmente, fornecendo apenas alguns argumentos de cada vez, resultando em uma nova função que espera os argumentos restantes.

Formalmente, o processo de **currying** pode ser descrito como um isomorfismo entre funções do tipo.

$$f : (A \times B) \to C$ e $g : A \to (B \to C)$$

 **Exemplo**:

 Considere a seguinte função que soma dois números:

 $$\text{add}(x, y) = x + y$$

 Essa função pode ser _Curryed_ da seguinte forma:

 $$\text{add}(x) = \lambda y. \, (x + y)$$

 Aqui, $\text{add}(x)$ é uma função que aceita $y$ como argumento e retorna a soma de $x$ e $y$. Isso permite a aplicação parcial da função:

 $$\text{add}(2) = \lambda y. \, (2 + y)$$

 Agora, $\text{add}(2)$ é uma função que aceita um argumento e retorna esse valor somado a 2.

### Propriedades e Vantagens do Currying

1. **Aplicação Parcial**: _Currying_ permite que funções sejam aplicadas parcialmente, o que pode simplificar o código e melhorar a reutilização. Em vez de aplicar todos os argumentos de uma vez, pode-se aplicar apenas alguns e obter uma nova função que espera os argumentos restantes.

2. **Flexibilidade**: Permite compor funções mais facilmente, combinando funções parciais em novos contextos sem a necessidade de redefinições.

3. **Isomorfismo com Funções Multivariadas**: Em muitos casos, funções que aceitam múltiplos argumentos podem ser tratadas como funções que aceitam um único argumento e retornam outra função. Essa correspondência torna o _Currying_ uma técnica natural para linguagens funcionais.

**No cálculo lambda, toda função é, por definição, uma função unária, o que significa que toda função no cálculo lambda já está implicitamente _Curryed_**. Funções de múltiplos argumentos são definidas como uma cadeia de funções que retornam outras funções. Uma função que soma dois números no cálculo lambda pode ser definida como:

$$\text{add} = \lambda x. \lambda y. \, x + y$$

Aqui, $\lambda x$ define uma função que aceita $x$ como argumento e retorna uma nova função $\lambda y$ que aceita $y$ e retorna a soma $x + y$. Quando aplicada, temos:

$$(\text{add} \, 2) \, 3 = (\lambda x. \lambda y. \, x + y) \, 2 \, 3$$

A aplicação funciona da seguinte forma:

$$ (\lambda x. \lambda y. \, x + y) \, 2 = \lambda y. \, 2 + y$$

E, em seguida:

$$ (\lambda y. \, 2 + y) \, 3 = 2 + 3 = 5$$

Esse é um exemplo claro de como _Currying_ permite a aplicação parcial de funções no cálculo lambda puro. Outro exemplo mais complexo seria uma função de multiplicação:

$$\text{mult} = \lambda x. \lambda y. \, x \times y$$

Aplicando parcialmente:

$$(\text{mult} \, 3) = \lambda y. \, 3 \times y$$

Agora, podemos aplicar o segundo argumento:

$$ (\lambda y. \, 3 \times y) \, 4 = 3 \times 4 = 12$$

Esses exemplos ilustram como o _Currying_ é um conceito fundamental no cálculo lambda, permitindo a definição e aplicação parcial de funções. Mas, ainda não vimos tudo.

#### Exercícios Currying

**1**: escreva uma expressão lambda que representa a função $f(x, y) = x + y$ usando currying. Aplique-a aos valores $x = 4$ e $y = 5$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, x + y$. Aplicando $x = 4$ e $y = 5$:

 $$(\lambda x. \lambda y. \, x + y) \, 4 \, 5 = 4 + 5 = 9$$

**2**: transforme a função $f(x, y, z) = x \cdot y + z$ em uma expressão lambda usando currying e aplique-a aos valores $x = 2$, $y = 3$, e $z = 4$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, \lambda z. \, x \cdot y + z$. Aplicando $x = 2$, $y = 3$, e $z = 4$:

 $$(\lambda x. \lambda y. \, \lambda z. \, x \cdot y + z) \, 2 \, 3 \, 4 = 2 \cdot 3 + 4 = 6 + 4 = 10$$

**3**: crie uma função curried que representa $f(x, y) = x^2 + y^2$. Aplique a função a $x = 1$ e $y = 2$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, x^2 + y^2$. Aplicando $x = 1$ e $y = 2$:

 $$ (\lambda x. \lambda y. \, x^2 + y^2) \, 1 \, 2 = 1^2 + 2^2 = 1 + 4 = 5 $$

**4**: converta a função $f(x, y) = \frac{x}{y}$ em uma expressão lambda usando currying e aplique-a aos valores $x = 9$ e $y = 3$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, \frac{x}{y}$. Aplicando $x = 9$ e $y = 3$:

 $$(\lambda x. \lambda y. \, \frac{x}{y}) \, 9 \, 3 = \frac{9}{3} = 3$$

**5**: defina uma função curried que calcule a diferença entre dois números, ou seja, $f(x, y) = x - y$, e aplique-a aos valores $x = 8$ e $y = 6$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, x - y$. Aplicando $x = 8$ e $y = 6$:

 $$(\lambda x. \lambda y. \, x - y) \, 8 \, 6 = 8 - 6 = 2$$

**6**: crie uma função curried para calcular a área de um retângulo, ou seja, $f(l, w) = l \cdot w$, e aplique-a aos valores $l = 7$ e $w = 5$.

 **Solução:** A função curried é $\lambda l. \lambda w. l \cdot w$. Aplicando $l = 7$ e $w = 5$:

 $$(\lambda l. \lambda w. l \cdot w) \, 7 \, 5 = 7 \cdot 5 = 35$$

**7**: transforme a função $f(x, y) = x^y$ (potência) em uma expressão lambda usando currying e aplique-a aos valores $x = 2$ e $y = 3$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, x^y$. Aplicando $x = 2$ e $y = 3$:

 $$(\lambda x. \lambda y. \, x^y) \, 2 \, 3 = 2^3 = 8$$

**8**: defina uma função curried que represente a multiplicação de três números, ou seja, $f(x, y, z) = x \cdot y \cdot z$, e aplique-a aos valores $x = 2$, $y = 3$, e $z = 4$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, \lambda z. \, x \cdot y \cdot z$. Aplicando $x = 2$, $y = 3$, e $z = 4$:

 $$ (\lambda x. \lambda y. \, \lambda z. \, x \cdot y \cdot z) \, 2 \, 3 \, 4 = 2 \cdot 3 \cdot 4 = 24$$

**9**: transforme a função $f(x, y) = x + 2y$ em uma expressão lambda curried e aplique-a aos valores $x = 1$ e $y = 4$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, x + 2y$. Aplicando $x = 1 $ e $ y = 4$:

 $$(\lambda x. \lambda y. \, x + 2y) \, 1 \, 4 = 1 + 2 \cdot 4 = 1 + 8 = 9$$

**10**: crie uma função curried para representar a soma de três números, ou seja, $f(x, y, z) = x + y + z$, e aplique-a aos valores $x = 3$, $y = 5$, e $z = 7$.

 **Solução:** A função curried é $\lambda x. \lambda y. \, \lambda z. \, x + y + z$. Aplicando $x = 3$, $y = 5$, e $z = 7$:

 $$(\lambda x. \lambda y. \, \lambda z. \, x + y + z) \, 3 \, 5 \, 7 = 3 + 5 + 7 = 15$$

A redução beta é o mecanismo fundamental de computação no cálculo lambda, permitindo a simplificação de expressões através da aplicação de funções a seus argumentos.

Formalmente, a redução beta é definida como:

$$(\lambda x.\,E)\, N \to_\beta [N/x]\, M$$

Onde $[N/x]\, M$ denota a substituição de todas as ocorrências livres de $x$ em $M$ por $N$. Isso reflete o processo de aplicação de uma função, onde substituímos o parâmetro formal $x$ pelo argumento $N$ no corpo da função $M$. Note que a substituição deve ser feita de maneira a evitar a captura de variáveis livres. Isso pode exigir a renomeação de variáveis ligadas (redução alfa) antes da substituição.

 **Exemplos**:

 Considere a expressão:

 $$(\lambda x. \, x+1)2$$

 Aplicando a redução beta:

 $$(\lambda x. \, x+1)2 \to_\beta [2/x](x+1) = 2+1 = 3$$

 Aqui, o valor $2$ é substituído pela variável $x$ na expressão $x + 1$, resultando em $2 + 1 = 3$.

Agora, um exemplo mais complexo envolvendo uma função de ordem superior:

 $$(\lambda f.\lambda x. \, f \, (f \, x))(\lambda y. \, y*2) \, 3$$

 Reduzindo passo a passo:

 $$(\lambda f.\lambda x.\, f(f \, x))(\lambda y.\, y \cdot 2)3$$

 $$\to_\beta (\lambda x.(\lambda y.\, y \cdot 2)((\lambda y.\, y \cdot 2) x)) \, 3$$

 $$\to_\beta (\lambda y.\, y \cdot 2)((\lambda y.\, y \cdot 2)\, 3)$$

 $$\to_\beta (\lambda y.\, y \cdot 2) \, (3 \cdot 2)$$

 $$\to_\beta (\lambda y.\, y \cdot 2) \, (6)$$

 $$\to_\beta 6 \cdot 2$$

 $$= 12$$

Neste exemplo, aplicamos primeiro a função $(\lambda f.\lambda x.\, f \, (f \, x))$ ao argumento $(\lambda y.\, y*2)$, resultando em uma expressão que aplica duas vezes a função de duplicação ao número $3$, obtendo $12$.

### Ordem Normal e Estratégias de Avaliação

A ordem em que as reduções beta são aplicadas pode afetar tanto a eficiência quanto a terminação do cálculo. Existem duas principais estratégias de avaliação:

1. **Ordem Normal**: Sempre reduz o redex mais externo à esquerda primeiro. Essa estratégia garante encontrar a forma normal de um termo, se ela existir. Na ordem normal, aplicamos a função antes de avaliar seus argumentos.

2. **Ordem Aplicativa**: Nesta estratégia, os argumentos são reduzidos antes da aplicação da função. Embora mais eficiente em alguns casos, pode não terminar em expressões que a ordem normal resolveria.

Por exemplo, considere a expressão:

$$(\lambda x.\, y)(\lambda z.\, z \, z)$$

- **Ordem Normal**: A função $(\lambda x.\, y)$ é aplicada diretamente ao argumento $(\lambda z.\, z \, z)$, resultando em:

 $$(\lambda x.\, y)(\lambda z.\, z \, z) \to_\beta y$$

 Aqui, não precisamos avaliar o argumento, pois a função simplesmente retorna $y$.

- **Ordem Aplicativa**: Primeiro, tentamos reduzir o argumento $(\lambda z.\, z \, z)$, resultando em uma expressão que se auto-aplica indefinidamente, causando um loop infinito:

 $$(\lambda x.\, y)(\lambda z.\, z \, z) \to_\beta (\lambda x.\, y)((\lambda z.\, z\, z)(\lambda z.\, z\, z)) \to_\beta ...$$

Este exemplo mostra que a ordem aplicativa pode levar a uma não terminação, enquanto a ordem normal encontra uma solução.

## Combinadores e Funções Anônimas

Os combinadores também tem origem no trabalho de [Moses Schönfinkel](https://en.wikipedia.org/wiki/Moses_Sch%C3%B6nfinkel). Em um artigo de 1924 de Moses Schönfinkel[^cita1]. Nele, ele define uma família de combinadores incluindo os padrões $S$, $K$ e $I$ e demonstra que apenas $S$ e $K$ são necessários[^cite3]. Seu conjunto inicial de combinadores inclui:

| Abreviação Original | Função Original em Alemão    | Tradução para o Inglês     | Expressão Lambda                       | Abreviação Atual |
|---------------------|-----------------------------|----------------------------|----------------------------------------|-----------------|
| $I$                 | Identitätsfunktion           | "função identidade"         | $\lambda x. x$                         | $I$             |
| $C$                 | Konstanzfunktion             | "função de constância"      | $\lambda xy. x$                        | $K$             |
| $T$                 | Vertauschungsfunktion        | "função de troca"           | $\lambda xyz. zxy$                     | $C$             |
| $Z$                 | Zusammensetzungsfunktion     | "função de composição"      | $\lambda xyz. xz(yz)$                  | $B$             |
| $S$                 | Verschmelzungsfunktion       | "função de fusão"           | $\lambda xyz. xz(yz)$                  | $S$             |

Schönfinkel também tinha combinadores que representavam operações lógicas, um para o [traço de Sheffer](https://en.wikipedia.org/wiki/Sheffer_stroke)(NAND), descoberto em 1913, e outro para a quantificação, porém, nenhum dos dois interessa neste momento. Contudo, Lembre-se de que qualquer circuito booleano pode ser construído apenas com portas NAND. Schönfinkel buscou, de maneira análoga, reduzir a lógica de predicados ao menor número possível de elementos, e, anos mais tarde, descobriu-se que os quantificadores "para todo" e "existe" da lógica de predicados se comportam como abstrações lambda.

Para nós, neste momento, um combinador é uma _expressão lambda_ fechada, ou seja, sem variáveis livres. Isso significa que todas as variáveis usadas no combinador estão ligadas dentro da própria expressão. Combinadores são elementos fundamentais da teoria do cálculo lambda, eles permitem criar funções complexas usando apenas blocos simples, sem a necessidade de referenciar variáveis externas.

Começamos com o combinador $K$, definido como:

$$K = \lambda x.\lambda y. \, x$$

Este combinador é uma função de duas variáveis que sempre retorna o primeiro argumento, ignorando o segundo. Ele representa o conceito de uma função constante. As funções constante sempre retornam o mesmo valor. No cálculo lambda o combinador $K$ sempre retorna o primeiro argumento independentemente do segundo.

Por exemplo, $KAB$ reduz para $A$, sem considerar o valor de $B$:

$$KAB = (\lambda x.\lambda y.x)AB \rightarrow_\beta (\lambda y.A)B \rightarrow_\beta A$$

Existem três combinadores considerados como fundamentais na construção de funções no cálculo lambda:

1.**Combinador I (Identidade)**:

 $$I = \lambda x. \, x$$

 O combinador identidade retorna o valor que recebe como argumento, sem modificá-lo.

 **Exemplo**: aplicando o combinador $I$ a qualquer valor, ele retornará esse mesmo valor:

 $$I \, 5 \rightarrow_\beta 5$$

 Outro exemplo:

 $$I \, (\lambda y. \, y + 1) \rightarrow_\beta \lambda y. \, y + 1$$

2.**Combinador K (ou C de Constante)**:

 $$K = \lambda x.\lambda y.x$$

 Este combinador ignora o segundo argumento e retorna o primeiro.

 **Exemplo**: Usando o combinador $K$ com dois valores:

 $$K \, 7 \, 4 \rightarrow_\beta (\lambda x.\lambda y. \, x) \, 7 \, 4 \rightarrow_\beta (\lambda y. \, 7) \, 4 \rightarrow_\beta 7$$

 Aqui, o valor $7$ será retornado, e o valor $4$ ignorando.

3.**Combinador S (Substituição)**:

 $$S = \lambda f.\lambda g.\lambda x. \, fx(gx)$$

 Este combinador é mais complexo, pois aplica a função $f$ ao argumento $x$ e, simultaneamente, aplica a função $g$ a $x$, passando o resultado de $g(x)$ como argumento para $f$.

 **Exemplo**: Vamos aplicar o combinador $S$ com as funções $f = \lambda z. \, z^2$ e $g = \lambda z. \, z + 1$, e o valor $3$:

 $$S \, (\lambda z. \, z^2) \, (\lambda z. \, z + 1) \, 3$$

 Primeiro, substituímos $f$ e $g$:

 $$\rightarrow_\beta (\lambda x.(\lambda z. \, z^2) \, x \, ((\lambda z. \, z + 1) \, x)) \, 3$$

 Agora, aplicamos as funções:

 $$\rightarrow_\beta (\lambda z. \, z^2) \, 3 \, ((\lambda z. \, z + 1) \, 3)$$

 $$\rightarrow_\beta 3^2 \, (3 + 1)$$

 $$\rightarrow_\beta 9 \, 4$$

 Assim, $S \, (\lambda z. \, z^2) \, (\lambda z. \, z + 1) \, 3$ resulta em $9$.

Finalmente, a lista de combinadores do cálculo lambda é um pouco mais extensa [^cita2]:

| Nome | Definição e Comentários |
|------|-------------------------|
| **S** | $\lambda x [\lambda y [\lambda z [x z (y z)]]]$. Lembre-se que $x z (y z)$ deve ser entendido como a aplicação $(x z)(y z)$ de $x z$ a $y z$. O combinador $S$ pode ser entendido como um operador de "substituir e aplicar": $z$ "intervém" entre $x$ e $y$; em vez de aplicar $x$ a $y$, aplicamos $x z$ a $y z$. |
| **K** | $\lambda x [\lambda y [x]]$. O valor de $K M$ é a função constante cujo valor para qualquer argumento é simplesmente $M$. |
| **I** | $\lambda x [x]$. A função identidade. |
| **B** | $\lambda x [\lambda y [\lambda z [x (y z)]]]$. Lembre-se que $x y z$ deve ser entendido como $(x y) z$, então este combinador não é uma função identidade trivial. |
| **C** | $\lambda x [\lambda y [\lambda z [x z y]]]$. Troca um argumento. |
| **T** | $\lambda x [\lambda y [x]]$. Valor verdadeiro lógico (True). Idêntico a $K$. Veremos mais tarde como essas representações dos valores lógicos desempenham um papel na fusão da lógica com o cálculo lambda. |
| **F** | $\lambda x [\lambda y [y]]$. Valor falso lógico (False). |
| **ω** | $\lambda x [x x]$. Combinador de autoaplicação. |
| **Ω** | $\omega \omega$. Autoaplicação do combinador de autoaplicação. Reduz para si mesmo. |
| **Y** | $\lambda f [(\lambda x [f (x x)]) (\lambda x [f (x x)])]$. O combinador paradoxal de Curry. Para todo termo lambda $X$, temos: $Y X \triangleright (\lambda x [X (x x)]) (\lambda x [X (x x)]) \triangleright X ((\lambda x [X (x x)]) (\lambda x [X (x x)]))$. A primeira etapa da redução mostra que $Y X$ reduz ao termo de aplicação $(\lambda x [X (x x)]) (\lambda x [X (x x)])$, que reaparece na terceira etapa. Assim, $Y$ tem a propriedade curiosa de que $Y X$ e $X (Y X)$ reduzem a um termo comum. |
| **Θ** | $(\lambda x [\lambda f [f (x x f)]]) (\lambda x [\lambda f [f (x x f)]])$. O combinador de ponto fixo de Turing. Para todo termo lambda $X$, $Θ X$ reduz para $X (Θ X)$, o que pode ser confirmado manualmente. (O combinador paradoxal de Curry $Y$ não tem essa propriedade.) |

No cálculo lambda é possível a construção de funções sem a necessidade de atribuir nomes explícitos. Aqui estamos próximos da álgebra e longe das linguagens de programação tradicionais, baseadas na Máquina de Turing. Isso se deve as _abstrações lambda_:

$$\lambda x. \, (\lambda y. \, y) \, x$$

A abstração lambda a cima, representa uma função que aplica a função identidade ao seu argumento $x$. Nesse caso, a função interna $\lambda y. \, y$ é aplicada ao argumento $x$, e o valor resultante é simplesmente $x$, já que a função interna é a identidade. Estas funções inspiraram a criação de funções anônimas e alguns operadores em linguagens de programação modernas. As funções _arrow_ em JavaScript ou às _lambdas_ em Python.

Uma propriedade notável dos combinadores é permitir a expressão de funções complexas sem o uso de variáveis nomeadas. Esse processo, conhecido como _abstração combinatória_, elimina a necessidade de variáveis explícitas, focando apenas em operações com funções. Um exemplo disso é a composição de funções definida por:

$$S \, (K \, S) \, K$$

Essa expressão representa o combinador de composição, comumente denotado como $B$, definido por:

$$B = \lambda f.\lambda g.\lambda x. \, f \, (g \, x)$$

Aqui, $B$ é construído inteiramente a partir dos combinadores $S$ e $K$, sem o uso de variáveis explícitas. Isso demonstra o poder do cálculo lambda puro, onde toda computação pode ser descrita através de combinações.

A capacidade de expressar qualquer função computável usando apenas combinadores é formalizada pelo _teorema da completude combinatória_. **Este teorema afirma que qualquer expressão lambda pode ser transformada em uma expressão equivalente utilizando apenas os combinadores $S$ e $K$**.

 **Exemplo 1**: Definindo uma função constante com o combinador $K$

 O combinador $K$ pode ser usado para criar uma função constante. A função criada sempre retorna o primeiro argumento, independentemente do segundo.

 Definimos a função constante:

 $$f = K \, A = \lambda x.\lambda y. ; x A = \lambda y. \, A$$

 Quando aplicamos $f$ a qualquer valor, o resultado será sempre $A$, pois o segundo argumento é ignorado.

 ****Exemplo 2**: Definindo a aplicação de uma função com o combinador $S$

 O combinador $S$ permite aplicar uma função a dois argumentos e combiná-los. Ele pode ser usado para definir uma função que aplica duas funções diferentes ao mesmo argumento e, em seguida, combina os resultados.

 Definimos a função composta:

 $$f = S \, g \, h = \lambda x. \, (g x)(h x)$$

 Aqui, $g$ e $h$ são duas funções que recebem o mesmo argumento $x$. O resultado é a combinação das duas funções aplicadas ao mesmo argumento.

 $$f A = (\lambda x.(g x)(h x)) A \rightarrow_\beta (g A)(h A)$$

A remoção de variáveis nomeadas simplifica a computação. Este é um dos pontos centrais da teoria dos combinadores.

Em linguagens funcionais como Haskell, essa característica é usada para criar expressões modulares e compostas. Isso traz clareza e concisão ao código.

### Exercícios sobre Combinadores e Funções Anônimas

**1**: Defina o combinador de ponto fixo de Curry, conhecido como o combinador $ Y $, e aplique-o à função $ f(x) = x + 1 $. Explique o que ocorre.

 **Solução:** O combinador $ Y $ é definido como:

 $$ Y = \lambda f. (\lambda x. \, f(x \, x)) (\lambda x. \, f(x \, x)) $$

 Aplicando-o à função $ f(x) = x + 1 $:

 $$ Y (\lambda x. \, x + 1) \to (\lambda x. \, (\lambda x. \, x + 1)(x \, x)) (\lambda x. \, (\lambda x. \, x + 1)(x \, x)) $$

 Este processo gera uma recursão infinita, pois a função continua chamando a si mesma.

**2**: Aplique o combinador $Y$ à função $f(x) = x \cdot 2$ e calcule as duas primeiras iterações do ponto fixo.

 **Solução:** Aplicando o combinador $Y$ a $f(x) = x \cdot 2$:

 $$Y (\lambda x. \, x \cdot 2)$$

 As duas primeiras iterações seriam:

 $$x_1 = 2$$
 $$x_2 = 2 \cdot 2 = 4$$

**3**: Mostre como o combinador $ Y $ pode ser aplicado para encontrar o ponto fixo da função $ f(x) = x^2 - 1 $.

 **Solução:** Aplicando o combinador $Y$ à função $f(x) = x^2 - 1$:

 $$Y (\lambda x. \, x^2 - 1)$$

 A função continuará sendo aplicada indefinidamente, mas o ponto fixo é a solução de $x = x^2 - 1$, que leva ao ponto fixo $x = \phi = \frac{1 + \sqrt{5}}{2}$ (a razão áurea).

**4**: Use o combinador de ponto fixo para definir uma função recursiva que calcula o fatorial de um número.

 **Solução:** A função fatorial pode ser definida como:

 $$ f = \lambda f. \lambda n. \, (n = 0 ? 1 : n \cdot f \, (n-1)) $$

 Aplicando o combinador $ Y $:

 $$ Y(f) = \lambda n. \, (n = 0 ? 1 : n \cdot Y \, (f) \, (n-1)) $$

 Agora podemos calcular o fatorial de um número, como $ 3! = 3 \cdot 2 \cdot 1 = 6 $.

**5**: Utilize o combinador $ Y $ para definir uma função recursiva que calcula a sequência de Fibonacci.

 **Solução:** A função para Fibonacci pode ser definida como:

 $$ f = \lambda f. \lambda n. \, (n = 0 ? 0 : (n = 1 ? 1 : f \, (n-1) + f \, (n-2))) $$

 Aplicando o combinador $ Y $:

 $$ Y \, (f) = \lambda n. \, (n = 0 ? 0 : (n = 1 ? 1 : Y \, (f) \, (n-1) + Y \, (f) \, (n-2))) $$

 Agora podemos calcular Fibonacci, como $F_5 = 5$.

**6**: Explique por que o combinador $Y$ é capaz de gerar funções recursivas, mesmo em linguagens sem suporte nativo para recursão.

 **Solução:** O combinador $Y$ cria recursão ao aplicar uma função a si mesma. Ele transforma uma função aparentemente sem recursão em uma recursiva ao introduzir auto-aplicação. Essa técnica é útil em linguagens onde a recursão não é uma característica nativa, pois o ponto fixo permite que a função se chame indefinidamente.

**7**: Mostre como o combinador $Y$ pode ser aplicado à função exponencial $f(x) = 2^x$ e calcule a primeira iteração.

 **Solução:** Aplicando o combinador $Y$ à função exponencial $f(x) = 2^x$:

 $$ Y (\lambda x. \, 2^x) $$

 A primeira iteração seria:

 $$x_1 = 2^1 = 2$$

**8**: Aplique o combinador de ponto fixo para encontrar o ponto fixo da função $f(x) = \frac{1}{x} + 1$.

 **Solução:** Para aplicar o combinador $Y$ a $f(x) = \frac{1}{x} + 1$, encontramos o ponto fixo ao resolver $x = \frac{1}{x} + 1$. O ponto fixo é a solução da equação quadrática, que resulta em $x = \phi$, a razão áurea.

**9**: Utilize o combinador $Y$ para definir uma função recursiva que soma os números de $1$ até $n$.

 **Solução:** A função de soma até $n$ pode ser definida como:

 $$ f = \lambda f. \lambda n. \, (n = 0 ? 0 : n + f \, (n-1)) $$

 Aplicando o combinador $Y$:

 $$ Y(f) = \lambda n. \, (n = 0 ? 0 : n + Y \, (f) \, (n-1)) $$

 Agora podemos calcular a soma, como $\sum\_{i=1}^{3} = 3 + 2 + 1 = 6$.

**10**: Aplique o combinador $Y$ para definir uma função recursiva que calcula o máximo divisor comum (MDC) de dois números.

 **Solução:** A função MDC pode ser definida como:

 $$ f = \lambda f. \lambda a. \lambda b. \, (b = 0 ? a : f \, (b, a \% b)) $$

 Aplicando o combinador $Y$:

 $$ Y(f) = \lambda a. \lambda b. \, (b = 0 ? a : Y \, (f) \, (b, a \% b)) $$

 Agora podemos calcular o MDC, como $\text{MDC}(15, 5) = 5 $.

**11**: Aplique o combinador identidade $ I = \lambda x. \, x $ ao valor $ 10 $.

 **Solução:** Aplicamos o combinador identidade:

 $$ I \, 10 = (\lambda x. \, x) \, 10 \rightarrow\_\beta 10 $$

**12**: Aplique o combinador $K = \lambda x. \lambda y. \, x$ aos valores $3$ e $7$. O que ocorre?

 **Solução:** Aplicamos $K$ ao valor $3$ e depois ao valor $7$:

 $$ K \, 3 \, 7 = (\lambda x. \lambda y. \, x) \, 3 \, 7 \rightarrow*\beta (\lambda y. \, 3) \, 7 \rightarrow*\beta 3 $$

**13**: Defina a expressão $ S(\lambda z. \, z^2)(\lambda z. \, z + 1) 4 $ e reduza-a passo a passo.

 **Solução:** Aplicamos o combinador $ S = \lambda f. \lambda g. \lambda x. \, f(x) \, (g \, (x)) $ às funções $f = \lambda z. \, z^2 $ e $ g = \lambda z. \, z + 1$, e ao valor $4$:

 $$S(\lambda z. \, z^2)(\lambda z. \, z + 1) \, 4$$

 Primeiro, aplicamos as funções:

 $$(\lambda f. \lambda g. \lambda x. \, f \, (x) \, (g \, (x)))(\lambda z. \, z^2)(\lambda z. \, z + 1) 4 $$

 Agora, substituímos e aplicamos as funções a $4$:

 $$(\lambda z. \, z^2) 4 ((\lambda z. \, z + 1) 4) \rightarrow\_\beta 4^2 \, (4 + 1) = 16 \cdot 5 = 80 $$

**14**: Aplique o combinador identidade $I$ a uma função anônima $\lambda y. \, y + 2$ e explique o resultado.

 **Solução:** Aplicamos o combinador identidade $I$ à função anônima:

 $$ I (\lambda y. \, y + 2) = (\lambda x. \, x) (\lambda y. \, y + 2) \rightarrow\_\beta \lambda y. \, y + 2 $$

 O combinador identidade retorna a própria função, sem modificações.

**15**: Reduza a expressão $K \, (I \, 7) \, 9$ passo a passo.

 **Solução:** Aplicamos $I$ a $7$, que resulta em $7$, e depois aplicamos $K$:

 $$ K \, (I \, 7) \, 9 = K \, 7 \, 9 = (\lambda x. \lambda y. \, x) \, 7 \, 9 \rightarrow*\beta (\lambda y. \, 7) \, 9 \rightarrow*\beta 7 $$

**16**: Aplique o combinador $K$ à função $\lambda z. \, z \cdot z $ e o valor $5$. O que ocorre?

 **Solução:** Aplicamos o combinador $K$ à função e ao valor:

 $$ K \, (\lambda z. \, z \cdot z) \, 5 = (\lambda x. \lambda y. \, x) \, (\lambda z. \, z \cdot z) \, 5 \rightarrow*\beta (\lambda y. \, \lambda z. \, z \cdot z) 5 \rightarrow*\beta \lambda z. \, z \cdot z $$

 O combinador $K$ descarta o segundo argumento, retornando a função original $\lambda z. \, z \cdot z$.

**17**: Construa uma função anônima que soma dois números sem usar nomes de variáveis explícitas, apenas usando combinadores $ S $ e $ K $.

 **Solução:** Usamos o combinador $S$ para aplicar duas funções ao mesmo argumento:

 $$ S \, (K \, (3)) \, (K \, (4)) = (\lambda f. \lambda g. \lambda x. \, f \, (x) \, (g \, (x))) \, (K \, (3))(K \, (4)) $$

 Aplicamos $f$ e $g$:

 $$ \rightarrow*\beta (\lambda x. \, K(3)(x)(K(4)(x))) \rightarrow*\beta (\lambda x. \, 3 + 4) = 7$$

**18**: Reduza a expressão $S \, K \, K$ e explique o que o combinador $S \, (K) \, (K)$ representa.

 **Solução:** Aplicamos o combinador $ S $:

 $$ S \, K \, K = (\lambda f. \lambda g. \lambda x. \, f \, (x) \, (g \, (x))) \, K \, K $$

 Substituímos $ f $ e $ g $ por $ K $:

 $$ = (\lambda x. \, K(x)(K(x))) $$
 Aplicamos $ K $:

 $$ = \lambda x. \, (\lambda y. \, x)( (\lambda y. \, x)) \rightarrow\_\beta \lambda x. \, x $$

 Portanto, $ S(K)(K) $ é equivalente ao combinador identidade $ I $.

**19**: Explique por que o combinador $ K $ pode ser usado para representar constantes em expressões lambda.

 **Solução:** O combinador $K = \lambda x. \lambda y. \, x$ descarta o segundo argumento e retorna o primeiro. Isso significa que qualquer valor aplicado ao combinador $K$ será mantido como constante, independentemente de quaisquer outros argumentos fornecidos. Por isso, o combinador $K$ pode ser usado para representar constantes, uma vez que sempre retorna o valor do primeiro argumento, ignorando os subsequentes.

**20**: Reduza a expressão $S \, (K \, S) \, K$ e explique o que esta combinação de combinadores representa.

 **Solução:** Aplicamos o combinador $S$:

 $$ S(KS)K = (\lambda f. \lambda g. \lambda x. \, f \, (x) \, (g \, (x))) \, K \, S \, K $$

 Substituímos $f = KS$ e $g = K$:

 $$= \lambda x. \, K \, S \, (x) \, (K \, (x))$$

 Aplicamos $K \, S$ e $K$:

 $$K \, S \, (x) = (\lambda x. \lambda y. \, x) \, S \, (x) = S$$

 $$K(x) = \lambda y. \, x$$

 Portanto:

 $$S \, (K \, S) \, K = S$$

 Essa combinação de combinadores representa a função de substituição $S$.

## Estratégias de Avaliação no Cálculo Lambda

**As estratégias de avaliação determinam como expressões são computadas**. Essas estratégias de avaliação também terão impacto na implementação de linguagens de programação. Diferentes abordagens para a avaliação de argumentos e funções podem resultar em diferentes características de desempenho.

### Avaliação por Valor vs Avaliação por Nome

No contexto do cálculo lambda e linguagens de programação, existem duas principais abordagens para avaliar expressões:

1. **Avaliação por Valor**: Nesta estratégia, os argumentos são avaliados antes de serem passados para uma função. O cálculo é feito de forma estrita, ou seja, os argumentos são avaliados imediatamente. Isso corresponde à **ordem aplicativa de redução**, onde a função é aplicada apenas após a avaliação completa de seus argumentos. A vantagem desta estratégia é que ela pode ser mais eficiente em alguns contextos, pois o argumento é avaliado apenas uma vez.

   **Exemplo**: Considere a expressão $ (\lambda x. \, x + 1) (2 + 3) $.

   Na **avaliação por valor**, primeiro o argumento $2 + 3$ é avaliado para $5$, e em seguida a função é aplicada:

   $$ (\lambda x. \, x + 1) 5 \rightarrow 5 + 1 \rightarrow 6 $$

2. Avaliação por Nome**: Argumentos são passados para a função sem serem avaliados imediatamente. A avaliação ocorre apenas quando o argumento é necessário. Esta estratégia corresponde à **ordem normal de redução**, em que a função é aplicada diretamente e o argumento só é avaliado quando estritamente necessário. Uma vantagem desta abordagem é que ela pode evitar avaliações desnecessárias, especialmente em contextos onde certos argumentos nunca são utilizados.

   **Exemplo**:
   Usando a mesma expressão $ \lambda x. \, x + 1) (2 + 3) $, com **avaliação por nome**, a função seria aplicada sem avaliar o argumento de imediato:

   $$ (\lambda x. \, x + 1) (2 + 3) \rightarrow (2 + 3) + 1 \rightarrow 5 + 1 \rightarrow 6 $$

### Exercícios sobre Estratégias de Avaliação

**1**: Considere a expressão $ (\lambda x. \, x + 1) (2 + 3) $. Avalie-a usando a estratégia de**avaliação por valor**.

**Solução:** Na avaliação por valor, o argumento é avaliado antes de ser aplicado à função:

 $$(2 + 3) \rightarrow 5$$
 Agora, aplicamos a função:

 $$(\lambda x. \, x + 1) 5 \rightarrow 5 + 1 \rightarrow 6 $$

**2**: Use a **avaliação por nome**na expressão $ (\lambda x. \, x + 1) (2 + 3) $ e explique o processo.

**Solução:** Na avaliação por nome, o argumento é passado diretamente para a função:

 $$ (\lambda x. \, x + 1) (2 + 3) \rightarrow (2 + 3) + 1 \rightarrow 5 + 1 \rightarrow 6 $$

**3**: A expressão $ (\lambda x. \, x \cdot x) ((2 + 3) + 1) $ é dada. Avalie-a usando a **avaliação por valor**.

**Solução:** Primeiro, avaliamos o argumento:

 $$ (2 + 3) + 1 \rightarrow 5 + 1 \to 6 $$
 Agora, aplicamos a função:

 $$ (\lambda x. \, x \cdot x) 6 \rightarrow 6 \cdot 6 \to 36 $$

**4**: Aplique a **avaliação por nome** na expressão $ (\lambda x. \, x \cdot x) ((2 + 3) + 1) $ e explique cada passo.

**Solução:** Usando avaliação por nome, o argumento não é avaliado imediatamente:

 $$ (\lambda x. \, x \cdot x) ((2 + 3) + 1) \rightarrow ((2 + 3) + 1) \cdot ((2 + 3) + 1) $$

 Agora, avaliamos o argumento quando necessário:

 $$ (5 + 1) \cdot (5 + 1) \to 6 \cdot 6 \to 36 $$

**5**: Considere a expressão $ (\lambda x. \, x + 1) ( (\lambda y. \, y + 2) 3) $. Avalie-a usando a **ordem aplicativa de redução** (avaliação por valor).

**Solução:** Primeiro, avaliamos o argumento $ (\lambda y. \, y + 2) 3 $:

 $$ (\lambda y. \, y + 2) 3 \rightarrow 3 + 2 \to 5 $$
 Agora, aplicamos $ 5 $ à função:

 $$ (\lambda x. \, x + 1) 5 \rightarrow 5 + 1 \to 6 $$

**6**: Aplique a **ordem normal de redução** (avaliação por nome) na expressão $ (\lambda x. \, x + 1) ( (\lambda y. \, y + 2) 3) $.

**Solução:** Usando a ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

 $$ (\lambda x. \, x + 1) ( (\lambda y. \, y + 2) 3) \rightarrow ( (\lambda y. \, y + 2) 3) + 1 $$

 Agora, avaliamos o argumento:

 $$ (3 + 2) + 1 \to 5 + 1 \to 6 $$

**7**: Considere a expressão $ (\lambda x. \, x + 1) (\lambda y. \, y + 2) $. Avalie-a usando **avaliação por valor** e explique por que ocorre um erro ou indefinição.

**Solução:** Na avaliação por valor, tentaríamos primeiro avaliar o argumento $ \lambda y. \, y + 2 $. No entanto, esse é um termo que não pode ser avaliado diretamente, pois é uma função. Logo, a expressão não pode ser reduzida, resultando em um erro ou indefinição, já que a função não pode ser aplicada diretamente sem um argumento concreto.

**8**: Aplique a **avaliação por nome** na expressão $ (\lambda x. \, x + 1) (\lambda y. \, y + 2) $.

**Solução:** Na avaliação por nome, passamos o argumento sem avaliá-lo:

 $$ (\lambda x. \, x + 1) (\lambda y. \, y + 2) \rightarrow (\lambda y. \, y + 2) + 1 $$

 Como a função $ \lambda y. \, y + 2 $ não pode ser somada diretamente a um número, a expressão resultante será indefinida ou produzirá um erro.

**9**: Dada a expressão $ (\lambda x. \lambda y. \, x + y) (2 + 3) 4 $, aplique a **ordem aplicativa de redução**.

**Solução:** Primeiro, avaliamos o argumento $ 2 + 3 $:

 $$ 2 + 3 \to 5 $$

 Agora, aplicamos a função $ (\lambda x. \lambda y. \, x + y) $:

 $$ (\lambda x. \lambda y. \, x + y) 5 4 \rightarrow (\lambda y. \, 5 + y) 4 \rightarrow 5 + 4 \to 9 $$

**10**: Use a **ordem normal de redução** para avaliar a expressão $ (\lambda x. \lambda y. \, x + y) (2 + 3) 4 $.

**Solução:** Na ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

 $$ (\lambda x. \lambda y. \, x + y) (2 + 3) 4 \rightarrow (\lambda y. \, (2 + 3) + y) 4 $$

 Agora, avaliamos os argumentos:

 $$ (5) + 4 \to 9 $$

# Estratégias de Redução

## Ordem Normal (Normal-Order)

Na **ordem normal**, a redução prioriza o _redex_ mais externo à esquerda (redução externa). Essa estratégia é garantida para encontrar a forma normal de um termo, caso ela exista. Além disso, como o argumento não é avaliado de imediato, é possível evitar o cálculo de argumentos que nunca serão utilizados, tornando-a equivalente à _avaliação preguiçosa_ em linguagens de programação.

- **Vantagens**: sempre encontra a forma normal de um termo, se ela existir; pode evitar a avaliação de argumentos desnecessários, melhorando a eficiência em termos de espaço.

- **Desvantagens**: pode ser ineficiente em termos de tempo, pois reavalia expressões várias vezes quando elas são necessárias repetidamente.

 **Exemplo**:
 Considere a expressão:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) $$

 Na **ordem normal**, a redução ocorre da seguinte maneira:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) \to\_\beta \lambda y. \, y $$

 O argumento $((\lambda z. \, z \, z) (\lambda w. w w))$ não é avaliado, pois ele nunca é utilizado no corpo da função.

## Ordem Aplicativa (Applicative-Order)

Na **ordem aplicativa**, os argumentos de uma função são avaliados antes da aplicação da função em si (redução interna). Esta é a estratégia mais comum em linguagens de programação imperativas e em algumas funcionais. Ela pode ser mais eficiente em termos de tempo, pois garante que os argumentos são avaliados apenas uma vez.

- **Vantagens**: Pode ser mais eficiente quando o resultado de um argumento é utilizado várias vezes, pois evita a reavaliação.

- **Desvantagens**: Pode resultar em não-terminação em casos onde a ordem normal encontraria uma solução. Além disso, pode desperdiçar recursos ao avaliar argumentos que não são necessários.

**Exemplo**:
 Utilizando a mesma expressão:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) $$

 Na **ordem aplicativa**, primeiro o argumento $((\lambda z. \, z \, z) (\lambda w. w w))$ é avaliado antes da aplicação da função:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) \to*\beta (\lambda x. \lambda y. \, y) ((\lambda w. w w) (\lambda w. w w)) \to*\beta ... $$

 Isso leva a uma avaliação infinita, uma vez que a expressão $((\lambda w. w w) (\lambda w. w w))$ entra em um loop sem fim.

### Exercícios sobre Ordem Normal e Aplicativa

**1**: Aplique a **ordem normal** à expressão $ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) $.

**Solução:**
 A ordem normal prioriza a redução externa:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) \rightarrow\_\beta \lambda y. \, y $$

 O argumento $((\lambda z. \, z \, z) (\lambda w. w w))$ nunca é avaliado.

**2**: Reduza a expressão $ (\lambda x. \lambda y. \, x) ((\lambda z. \, z + 1) 5) $ usando a **ordem normal**.

**Solução:**
 Na ordem normal, aplicamos a função sem avaliar o argumento imediatamente:

 $$ (\lambda x. \lambda y. \, x) ((\lambda z. \, z + 1) 5) \rightarrow\_\beta \lambda y. \, ((\lambda z. \, z + 1) 5) $$

 O argumento não é avaliado porque a função não o utiliza.

**3**: Considere a expressão $ (\lambda x. \lambda y. \, y + 1) ((\lambda z. \, z \, z) (\lambda w. w w)) $. Avalie-a usando **ordem normal**.

**Solução:**
 A ordem normal evita a avaliação do argumento:

 $$ (\lambda x. \lambda y. \, y + 1) ((\lambda z. \, z \, z) (\lambda w. w w)) \rightarrow\_\beta \lambda y. \, y + 1 $$

 O termo $((\lambda z. \, z \, z) (\lambda w. w w))$ nunca é avaliado.

**4**: Aplique a **ordem normal** na expressão $ (\lambda x. \, x) ((\lambda z. \, z \, z) (\lambda w. w w)) $.

**Solução:**
 Primeiro aplicamos a função sem avaliar o argumento:

 $$ (\lambda x. \, x) ((\lambda z. \, z \, z) (\lambda w. w w)) \rightarrow\_\beta ((\lambda z. \, z \, z) (\lambda w. w w)) $$

 Agora a expressão é indefinida, pois avaliaremos uma expressão sem fim.

**5**: Reduza a expressão $ (\lambda x. \, 3) ((\lambda z. \, z + 1) 5) $ utilizando a **ordem normal**.

**Solução:**
 Na ordem normal, o argumento não é avaliado:

 $$ (\lambda x. \, 3) ((\lambda z. \, z + 1) 5) \rightarrow\_\beta 3 $$

 O argumento $((\lambda z. \, z + 1) 5)$ nunca é avaliado.

**6**: Avalie a expressão $ (\lambda x. \lambda y. \, x) ((\lambda z. \, z + 1) 5) $ usando **ordem aplicativa**.

**Solução:**
 Na ordem aplicativa, o argumento é avaliado primeiro:

 $$ (\lambda z. \, z + 1) 5 \rightarrow\_\beta 6 $$

 Agora aplicamos a função:

 $$ (\lambda x. \lambda y. \, x) 6 \rightarrow\_\beta \lambda y. \, 6 $$

**7**: Aplique a **ordem aplicativa** à expressão $ (\lambda x. \, x) ((\lambda z. \, z \, z) (\lambda w. w w)) $.

**Solução:**
 Na ordem aplicativa, o argumento é avaliado primeiro, o que leva a um loop sem fim:

 $$ ((\lambda z. \, z \, z) (\lambda w. w w)) \rightarrow*\beta (\lambda w. w w) (\lambda w. w w) \rightarrow*\beta ... $$

 A expressão entra em uma recursão infinita.

**8**: Reduza a expressão $ (\lambda x. \, x \cdot 2) ((\lambda z. \, z + 3) 4) $ usando **ordem aplicativa**.

**Solução:**
 Primeiro, o argumento $ (\lambda z. \, z + 3) 4 $ é avaliado:

 $$ (\lambda z. \, z + 3) 4 \rightarrow\_\beta 4 + 3 \to 7 $$

 Agora aplicamos a função:

 $$ (\lambda x. \, x \cdot 2) 7 \rightarrow\_\beta 7 \cdot 2 \to 14 $$

**9**: Considere a expressão $ (\lambda x. \, x + 1) (\lambda y. \, y + 2) $. Avalie-a usando **ordem aplicativa** e explique o resultado.

**Solução:**
 Na ordem aplicativa, tentamos avaliar o argumento primeiro:

 $$ (\lambda y. \, y + 2) \rightarrow\_\beta \lambda y. \, y + 2 $$

 Como o argumento não pode ser avaliado (é uma função), o resultado não pode ser reduzido, levando a um erro ou indefinição.

**10**: Aplique a **ordem aplicativa** à expressão $ (\lambda x. \, x + 1) ((\lambda z. \, z + 2) 3) $.

**Solução:**
 Primeiro avaliamos o argumento:

 $$ (\lambda z. \, z + 2) 3 \rightarrow\_\beta 3 + 2 \to 5 $$

 Agora aplicamos a função:

 $$ (\lambda x. \, x + 1) 5 \rightarrow\_\beta 5 + 1 \to 6 $$

**11**: Compare a avaliação da expressão $ (\lambda x. \, 2) ((\lambda z. \, z \, z) (\lambda w. w w)) $ usando **ordem normal** e **ordem aplicativa**.

**Solução (Ordem Normal):**
 A ordem normal evita a avaliação do argumento:

 $$ (\lambda x. \, 2) ((\lambda z. \, z \, z) (\lambda w. w w)) \rightarrow\_\beta 2 $$

**Solução (Ordem Aplicativa):**
 Na ordem aplicativa, o argumento é avaliado, levando a um loop sem fim.

**12**: Considere a expressão $ (\lambda x. \lambda y. \, x + y) ((\lambda z. \, z + 1) 3) 4 $. Avalie usando **ordem normal** e **ordem aplicativa**.

**Solução (Ordem Normal):**
 Aplicamos a função sem avaliar o argumento:

 $$ (\lambda x. \lambda y. \, x + y) ((\lambda z. \, z + 1) 3) 4 \rightarrow\_\beta (\lambda y. \, ((\lambda z. \, z + 1) 3) + y) 4 $$

 Agora avaliamos o argumento:

 $$ ((3 + 1) + 4) \to 8 $$

**Solução (Ordem Aplicativa):**
Na ordem aplicativa, avaliamos o argumento primeiro:

 $$ (\lambda z. \, z + 1) 3 \rightarrow\_\beta 4 $$

 Agora aplicamos a função:

 $$ (\lambda x. \lambda y. \, x + y) 4 4 \rightarrow\_\beta 4 + 4 \to 8 $$

**13**: Aplique **ordem normal** e **ordem aplicativa** à expressão $ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) 3 $.

**Solução (Ordem Normal):**
 A função é aplicada sem avaliar o argumento:

 $$ (\lambda x. \lambda y. \, y) ((\lambda z. \, z \, z) (\lambda w. w w)) 3 \rightarrow\_\beta \lambda y. \, y $$

 Agora aplicamos a função:

 $$ (\lambda y. \, y) 3 \rightarrow\_\beta 3 $$

**Solução (Ordem Aplicativa):**
 Na ordem aplicativa, o argumento é avaliado, resultando em um loop infinito.

**14**: Avalie a expressão $ (\lambda x. \, x) ((\lambda z. \, z + 1) 3) $ usando **ordem normal** e **ordem aplicativa**.

**Solução (Ordem Normal):**
 A função é aplicada sem avaliar o argumento:

 $$ (\lambda x. \, x) ((\lambda z. \, z + 1) 3) \rightarrow*\beta ((\lambda z. \, z + 1) 3) \rightarrow*\beta 4 $$

**Solução (Ordem Aplicativa):**
 Na ordem aplicativa, o argumento é avaliado primeiro:

 $$ (\lambda z. \, z + 1) 3 \rightarrow\_\beta 4 $$

 Agora aplicamos a função:

 $$ (\lambda x. \, x) 4 \rightarrow\_\beta 4 $$

**15**: Reduza a expressão $ (\lambda x. \, x) (\lambda y. \, y + 2) $ usando **ordem normal** e **ordem aplicativa**.

**Solução (Ordem Normal):**
 Aplicamos a função sem avaliar o argumento:

 $$ (\lambda x. \, x) (\lambda y. \, y + 2$$

## Impactos em Linguagens de Programação

Haskell é uma linguagem de programação que utiliza **avaliação preguiçosa**, que corresponde à **ordem normal**. Isso significa que os argumentos só são avaliados quando absolutamente necessários, o que permite trabalhar com estruturas de dados potencialmente infinitas.

**Exemplo**:

 ```haskell
 naturals = [0..]
 take 5 naturals -- Retorna [0,1,2,3,4]
 ```

 Aqui, a lista infinita `naturals` nunca é totalmente avaliada. Somente os primeiros 5 elementos são calculados, graças à avaliação preguiçosa.

 A linguagem OCaml, por padrão, usa avaliação estrita (ordem aplicativa), mas fornece suporte para avaliação preguiçosa através do módulo Lazy.

**Exemplo**:

 ```ocaml
 let lazy_value = lazy (expensive_computation())
 let result = Lazy.force lazy_value
 ```

 Nesse exemplo, `expensive_computation` só será avaliada quando `Lazy.force` for chamado. Até então, o valor é mantido em um estado _preguiçoso_, economizando recursos se o valor nunca for utilizado.

Já o JavaScript usa apenas avaliação estrita. Contudo, oferece suporte à avaliação preguiçosa por meio de geradores, que permitem gerar valores sob demanda.

**Exemplo**:

 ```javascript
 function* naturalNumbers() {
 let n = 0;
 While (true) yield n++;
 }

 const gen = naturalNumbers();
 console.log(gen.next().value); // Retorna 0
 console.log(gen.next().value); // Retorna 1
 ```

O código JavaScript do exemplo utiliza um gerador para criar uma sequência infinita de números naturais, produzidos sob demanda, um conceito semelhante à avaliação preguiçosa (_lazy evaluation_). Assim como na ordem normal de redução, onde os argumentos são avaliados apenas quando necessários, o gerador `naturalNumbers()` só avalia e retorna o próximo valor quando o método `next()` é chamado. Isso evita a criação imediata de uma sequência infinita e permite o uso eficiente de memória, computando os valores apenas quando solicitados, como ocorre na avaliação preguiçosa.

# Equivalência Lambda e Definição de Igualdade

No cálculo lambda, a noção de equivalência vai além da simples comparação sintática entre dois termos. Ela trata de quando dois termos podem ser considerados **igualmente computáveis** ou **equivalentes** em um sentido mais profundo, independentemente de suas formas superficiais. Esta equivalência é central para otimizações de programas, verificação de tipos e raciocínio em linguagens funcionais.

Dois termos lambda $M$ e $N$ são considerados equivalentes, denotado por $M\to_\beta N$, se podemos transformar um no outro através de uma sequência (possivelmente vazia) de:

1. **$\alpha$-conversões**: que permitem a renomeação de variáveis ligadas, assegurando que a identidade de variáveis internas não afeta o comportamento da função.

2. **$\beta$-reduções**: que representam a aplicação de uma função ao seu argumento, o princípio básico da computação no cálculo lambda.

3. **$\eta$-conversões**: que expressam a extensionalidade[^nota5] de funções, permitindo igualar duas funções que se comportam da mesma forma quando aplicadas a qualquer argumento.

Formalmente, a relação $\to_\beta$ é a menor relação de equivalência que satisfaz as seguintes propriedades fundamentais:

1. **$\beta$-redução**: $ (\lambda x.M)N \to_\beta M[N/x] $

   Isto significa que a aplicação de uma função $ (\lambda x.M) $ a um argumento $N$ resulta na substituição de todas as ocorrências de $x$ em $M$ pelo valor $N$.

2. **$\eta$-conversão**: $\lambda x. \, Mx\to_\beta M$, se $x$ não ocorre livre em $M$

   A $\eta$-conversão captura a ideia de extensionalidade. Se uma função $\lambda x.Mx$ aplica $M$ a $x$ sem modificar $x$, ela é equivalente a $M$.

3. **Compatibilidade com abstração**: Se $M\to_\beta M'$, então $\lambda x. \, M\to_\beta \lambda x.M'$

   Isto garante que se dois termos são equivalentes, então suas abstrações (funções que os utilizam) também serão equivalentes.

4. **Compatibilidade com aplicação**: Se $M\to_\beta M'$ e $N\to_\beta N'$, então $M \, N\to_\beta M'N'$

   Esta regra assegura que a equivalência se propaga para as aplicações de funções, mantendo a consistência da equivalência.

É importante notar que a ordem em que as reduções são aplicadas não afeta o resultado final, devido à propriedade de Church-Rosser do cálculo lambda. Isso garante que, independentemente de como o termo é avaliado, se ele tem uma forma normal, a avaliação eventualmente a encontrará.

A relação $\to_\beta$ é uma **relação de equivalência**, o que significa que ela possui três propriedades fundamentais:

1. **Reflexiva**: Para todo termo $M$, temos que $M\to_\beta M$. Isto significa que qualquer termo é equivalente a si mesmo, o que é esperado.

2. **Simétrica**: Se $M\to_\beta N$, então $N\to_\beta M$. Se um termo $M$ pode ser transformado em $N$, então o oposto também é verdade.

3. **Transitiva**: Se $M\to_\beta N$ e $N\to_\beta P$, então $M\to_\beta P$. Isso implica que, se podemos transformar $M$ em $N$ e $N$ em $P$, então podemos transformar diretamente $M$ em $P$.

A equivalência $\to_\beta$ é fundamental para o raciocínio sobre programas em linguagens funcionais, permitindo substituições e otimizações que preservam o significado computacional. As propriedades da equivalência $\to_\beta$ garantem que podemos substituir termos equivalentes em qualquer contexto, sem alterar o significado ou o resultado da computação. Em termos de linguagens de programação, isso permite otimizações e refatorações que preservam a correção do programa.

Vamos ver alguns exemplos de equivalência.

1. **Identidade e aplicação trivial**:

 **Exemplo 1**:

 $$ \lambda x.(\lambda y. \, y)x \to\_\beta \lambda x. \, x $$

 Aqui, a função interna $\lambda y. \, y$ é a função identidade, que simplesmente retorna o valor de $x$. Após a aplicação, obtemos $\lambda x. \, x$, que também é a função identidade.

 **Exemplo 2**:

 $$ \lambda z.(\lambda w.w)z \to\_\beta \lambda z.z $$

 Assim como no exemplo original, a função interna $\lambda w.w$ é a função identidade. Após a aplicação, o valor de $z$ é retornado.

 **Exemplo 3**:

 $$ \lambda a.(\lambda b.b)a \to\_\beta \lambda a.a $$

 A função $\lambda b.b$ é aplicada ao valor $a$, retornando o próprio $a$. Isso demonstra mais uma aplicação da função identidade.

2. **Função constante**:

 **Exemplo 1**:

 $$ (\lambda x.\lambda y.x)M \, N \to\_\beta M $$

 Neste exemplo, a função $\lambda x.\lambda y.x$ retorna sempre seu primeiro argumento, ignorando o segundo. Aplicando isso a dois termos $M$ e $N$, o resultado é simplesmente $M$.

 **Exemplo 2**:

 $$ (\lambda a.\lambda b.a)P Q \to\_\beta P $$

 A função constante $\lambda a.\lambda b.a$ retorna sempre o primeiro argumento ($P$), ignorando $Q$.

 **Exemplo 3**:

 $$ (\lambda u.\lambda v.u)A B \to\_\beta A $$

 Aqui, o comportamento é o mesmo: o primeiro argumento ($A$) é retornado, enquanto o segundo ($B$) é ignorado.

3. **$\eta$-conversão**:

 **Exemplo 1**:

 $$ \lambda x.(\lambda y.M)x \to\_\beta \lambda x.M[x/y] $$

 Se $x$ não ocorre livre em $M$, podemos usar a $\eta$-conversão para "encurtar" a expressão, pois aplicar $M$ a $x$ não altera o comportamento da função. Este exemplo mostra como podemos internalizar a aplicação, eliminando a dependência desnecessária de $x$.

 **Exemplo 2**:

 $$ \lambda x.(\lambda z.N)x \to\_\beta \lambda x.N[x/z] $$

 Similarmente, se $x$ não ocorre em $N$, a $\eta$-conversão simplifica a expressão para $\lambda x.N$.

 **Exemplo 3**:

 $$ \lambda f.(\lambda g.P)f \to\_\beta \lambda f.P[f/g] $$

 Aqui, a $\eta$-conversão elimina a aplicação de $f$ em $P$, resultando em $\lambda f.P$.

4. **Termo $\Omega$ (não-terminante)**:

 **Exemplo 1**:

 $$ (\lambda x. \, xx)(\lambda x. \, xx) \to\_\beta (\lambda x. \, xx)(\lambda x. \, xx) $$

 Este é o famoso _combinador $\Omega$_, que se reduz a si mesmo indefinidamente, criando um ciclo infinito de auto-aplicações. Apesar de não ter forma normal (não termina), ele é equivalente a si mesmo por definição.

 **Exemplo 2**:

 $$ (\lambda f.\, f\,f)(\lambda f.\,f\,f) \to\_\beta (\lambda f.\,f\,f)(\lambda f.\,f\,f) $$

 Assim como o combinador $\Omega$, este termo também cria um ciclo infinito de auto-aplicação.

 **Exemplo 3**:

 $$ (\lambda u.\,u\,u)(\lambda u.\,u\,u) \to\_\beta (\lambda u.\,u\,u)(\lambda u.\,u\,u) $$

 Outra variação do combinador $\Omega$, que também resulta em uma redução infinita sem forma normal.

5. **Composição de funções**:

 **Exemplo 1**:

 $$ (\lambda f.\lambda g.\lambda x.\,f\,(g\,x))\,M \, N \to\_\beta \lambda x.\,M \, (N \, x)$$

 Neste caso, a composição de duas funções, $M$ e $N$, é expressa como uma função que aplica $N$ ao argumento $x$, e então aplica $M$ ao resultado. A redução demonstra como a composição de funções pode ser representada e simplificada no cálculo lambda.

 **Exemplo 2**:

 $$ (\lambda f.\lambda g.\lambda y.\,f\,(g\,y))\,A\,B \to\_\beta \lambda y.\,A\,(B\,y) $$

 A composição de $A$ e $B$ é aplicada ao argumento $y$, e o resultado de $By$ é então passado para $A$.

 **Exemplo 3**:

 $$(\lambda h.\lambda k.\lambda z.\,h\,(k\,z))\,P\,Q \to_\beta \lambda z.\,P\,(Q\,z)$$

 Similarmente, a composição de $P$ e $Q$ é aplicada ao argumento $z$, e o resultado de $Qz$ é passado para $P$.

## Equivalência Lambda e seu Impacto em Linguagens de Programação

A equivalência lambda influencia o desenvolvimento e a otimização de linguagens funcionais como Haskell e OCaml. Essa ideia de equivalência dá uma base forte para pensar na semântica dos programas de forma abstrata. Isso é essencial para a verificação formal e otimização automática.

Dois termos lambda $M$ e $N$ são considerados equivalentes, denotado por $M\to_\beta N$, se é possível transformar um no outro através de uma sequência (possivelmente vazia) de:

1. $\alpha$-conversões (renomeação de variáveis ligadas)
2. $\beta$-reduções (aplicação de funções)
3. $\eta$-conversões (extensionalidade de funções)

Formalmente:

$$
\begin{align*}
&\text{1. } (\lambda x. \, M) \, N\to*\beta M \, [N/x] \text{ ($\beta$-redução)} \\
&\text{2. } \lambda x. \, Mx\to*\beta M, \text{ se $x$ não ocorre livre em $M$ ($\eta$-conversão)} \\
&\text{3. Se } M\to*\beta M' \text{, então } \lambda x. \, M\to*\beta \lambda x. \, M' \text{ (compatibilidade com abstração)} \\
&\text{4. Se } M\to*\beta M' \text{ e } N\to*\beta N' \text{, então } M \, N\to\_\beta M' \, N' \text{ (compatibilidade com aplicação)}
\end{align*}
$$

Talvez algumas aplicações em linguagem Haskell ajude a fixar os conceitos.

1. **Eliminação de Código Redundante**

 A equivalência lambda permite a substituição de expressões por versões mais simples sem alterar o comportamento do programa. Por exemplo:

 ```haskell
 -- Antes da otimização
 let x = (\y -> y + 1) 5 in x * 2
 -- Após a otimização (equivalente)
 let x = 6 in x * 2
 ```

 Aqui, o compilador pode realizar a $\beta$-redução $ (\lambda y. \, y + 1) 5\to_\beta 6$ em tempo de compilação, simplificando o código.

2. **Transformações Seguras de Código**

 Os Compiladores podem aplicar refatorações automáticas baseadas em equivalências lambda. Por exemplo:

 ```haskell
 -- Antes da transformação
 map (\x -> f (g x)) xs

 -- Após a transformação (equivalente)
 map (f . g) xs
 ```

 Esta transformação, baseada na lei de composição $f \circ g \equiv \lambda x. \, f(g(x))$, pode melhorar a eficiência e legibilidade do código.

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

 Esta inferência demonstra como a equivalência lambda é usada pelo sistema de tipos: `f show` é equivalente a `map show`. O tipo de `map show` é inferido como `[Int] -> [String]`. No `main`, vemos um exemplo de uso de `h`, que converte uma lista de inteiros em uma lista de _strings_.

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
 putStrLn "Avaliando com condition = True:"
 print $ lazy_example True
 putStrLn "Avaliando com condition = False:"
 print $ lazy_example False
```

Neste exemplo: `expensive_computation` é uma função que realiza um cálculo custoso (soma dos primeiros 1 bilhão de números inteiros).

`lazy_example` é uma função que demonstra a avaliação preguiçosa. Ela aceita um argumento `booleano condition`. Dentro de `lazy_example`, `x` é definido como `expensive_computation`, mas devido à avaliação preguiçosa, este cálculo não é realizado imediatamente.

Se `condition for True`, o programa calculará `x + 1`, o que forçará a avaliação de `expensive_computation`. Se `condition for False`, o programa retornará `0`, e `expensive_computation` nunca será avaliado.

Ao executar este programa, você verá que: quando `condition` é `True`, o programa levará um tempo considerável para calcular o resultado. Quando `condition` é `False`, o programa retorna instantaneamente, pois `expensive_computation` não é avaliado.

Graças à equivalência lambda e à avaliação preguiçosa, `expensive_computation` só será avaliado se `condition` for verdadeira.

A equivalência Lambda, ainda que muito importante, não resolve todos os problemas possíveis. Alguns dos desafios estão relacionados com:

1. **Indecidibilidade**: Determinar se dois termos lambda são equivalentes é um problema indecidível em geral. Compiladores devem usar heurísticas e aproximações.

2. **Efeitos Colaterais**: Em linguagens com efeitos colaterais, a equivalência lambda pode não preservar a semântica do programa. Por exemplo:

 ```haskell
 -- Estas expressões não são equivalentes em presença de efeitos colaterais
 f1 = (\x -> putStrLn ("Processando " ++ show x) >> return (x + 1))
 f2 = \x -> do
 putStrLn ("Processando " ++ show x)
 return (x + 1)

 main = do
 let x = f1 5
 y <- x
 print y

 let z = f2 5
 w <- z
 print w
 ```

 Neste exemplo, `f1` e `f2` parecem equivalentes do ponto de vista do cálculo lambda puro. No entanto, em Haskell, que tem um sistema de I/O baseado em _monads_, elas se comportam diferentemente:

 - `f1` cria uma ação de I/O que, quando executada, imprimirá a mensagem e retornará o resultado.
 - `f2` também cria uma ação de I/O, mas a mensagem será impressa imediatamente quando `f2` for chamada.

 Ao executar este programa, você verá que a saída para `f1` e `f2` é diferente devido ao momento em que os efeitos colaterais (impressão) ocorrem.

3. **Complexidade Computacional**: Mesmo quando decidível, verificar equivalências pode ser computacionalmente caro, exigindo um equilíbrio entre otimização e tempo de compilação.

## Números de Church

Estudar cálculo lambda após a álgebra abstrata nos faz pensar numa relação entre a representação dos números naturais por Church e a definição dos números naturais de [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor). Embora estejam em contextos teóricos diferentes, ambos tentam capturar a essência dos números naturais com estruturas básicas distintas.

Cantor é conhecido por seu trabalho na teoria dos conjuntos. Ele definiu os números naturais como um conjunto infinito e ordenado, começando do $0$ e progredindo com o operador sucessor. Para Cantor, cada número natural é construído a partir do anterior por meio de uma sucessão bem definida. O número $1$ é o sucessor de $0$, o número $2$ é o sucessor de $1$, e assim por diante. Esta estrutura fornece uma base sólida para a aritmética dos números naturais, especialmente na construção de conjuntos infinitos e na cardinalidade.

Enquanto Cantor desenvolveu sua teoria baseada em conjuntos e sucessores, Church adotou uma abordagem funcional. Em vez de tratar os números como elementos de um conjunto, Church os define como funções que operam sobre outras funções. Isso permite realizar a aritmética de maneira puramente funcional. Esta diferença reflete duas maneiras de abstrair os números naturais, ambas capturando sua essência recursiva, mas com ferramentas matemáticas diferentes.

A ligação entre essas abordagens está no conceito de sucessor e na construção incremental e recursiva dos números. Embora Cantor e Church tenham trabalhado em áreas distintas — Cantor na teoria dos conjuntos e Church no cálculo lambda —, ambos representam os números naturais como entidades geradas de forma recursiva.

Agora podemos explorar os números de Church.

## Representação de Números Naturais no Cálculo Lambda

Os números de Church são uma representação elegante dos números naturais no cálculo lambda puro. Essa representação além de permitir a codificação dos números naturais, permite a implementação de operações aritméticas.

A ideia central dos números de Church é representar o número $n$ como uma função $f$. Essa função aplica outra função $f$ $n$ vezes a um argumento $x$. Formalmente, o número de Church para $n$ é:

$$n = \lambda s. \lambda z. \, s^n\, (z)$$

Aqui, $s^n(z)$ significa aplicar $s$ a $z$, $n$ vezes. $s$ representa o sucessor, e $z$ representa zero. Essa definição captura a essência dos números naturais: zero é a base, e cada número seguinte é obtido aplicando a função sucessor repetidamente.

Os primeiros números naturais podem ser representados da seguinte maneira:

- $0 = \lambda s. \lambda z. \, z$
- $1 = \lambda s. \lambda z. s \, (z)$
- $2 = \lambda s. \lambda z. s \, (s \, (z))$
- $3 = \lambda s. \lambda z. s \, (s \, (s \, (z)))$

A representação dos números naturais no cálculo lambda permite definir funções que operam sobre esses números.

A função sucessor, que incrementa um número natural como sucessor de outro número natural, é definida como:

$$ \text{succ} = \lambda n. \lambda s. \lambda z. \, s \, (n \, s \, z) $$

A **função sucessor** é essencial para entender como os números naturais são criados. No entanto, para que os números de Church sejam considerados números naturais, é preciso implementar operações aritméticas no cálculo lambda puro. Para entender como isso funciona na prática, vamos aplicar $\text{succ}$ ao número $2$:

$$
\begin{aligned}
\text{succ } \, 2 &= (\lambda n. \lambda s. \lambda z.\, s(n \, s \, z)) (\lambda s. \lambda z.\, s\, (s\, (z))) \\
&= \lambda s. \lambda z.\, s((\lambda s. \lambda z.\, s\, (s\, (z))) \, s \, z) \\
&= \lambda s. \lambda z. \, s\, (s\, (s\, (z))) \\
&= 3
\end{aligned}
$$

A **adição** entre dois números de Church, $m$ e $n$, pode ser definida como:

$$\text{add} = \lambda m. \lambda n. \lambda s. \lambda z. \, m \, s \, (n \, s \, z)$$

Essa definição funciona aplicando $n$ vezes a função $s$ ao argumento $z$, representando o número $n$. Em seguida, aplica $m$ vezes a função $s$ ao resultado anterior, somando $m$ e $n$. Isso itera sobre $n$ e depois sobre $m$, combinando as duas operações para obter a soma. Vamos ver um exemplo:

Vamos usar os números de Church para calcular a soma de $2 + 3$ em cálculo lambda puro. Primeiro, representamos os números $2$ e $3$ usando as definições de números de Church:

$$2 = \lambda s. \lambda z. \, s \, (s \, z)$$
$$3 = \lambda s. \lambda z. \, s \, (s \, (s \, z))$$

Agora, aplicamos a função de adição de números de Church:

$$ \text{add} = \lambda m. \lambda n. \lambda s. \lambda z. \, m \, s \, (n \, s \, z) $$

Substituímos $m$ por $2$ e $n$ por $3$:

$$ \text{add} \, 2 \, 3 = (\lambda m. \lambda n. \lambda s. \lambda z. \, m \, s \, (n \, s \, z)) \, 2 \, 3 $$

Expandimos a função:

$$ = \lambda s. \lambda z. 2 \, s \, (3 \, s \, z) $$

Substituímos as representações de $2$ e $3$:

$$ = \lambda s. \lambda z. (\lambda s. \lambda z. s \, (s \, z)) \, s \, ( (\lambda s. \lambda z. s \, (s \, (s \, z))) \, s \, z) $$

Primeiro, resolvemos o termo $3 \, s \, z$:

$$ = s \, (s \, (s \, z)) $$

Agora, aplicamos $2 \, s$ ao resultado:

$$ = s \, (s \, (s \, (s \, (s \, z)))) $$

E chegamos ao aninhamento de cinco funções o que representa $5$, ou seja, o resultado de $2 + 3$ em cálculo lambda puro.

A **multiplicação** de dois números de Church, $m$ e $n$, é expressa assim:

$$\text{mult} = \lambda m. \lambda n. \lambda s. m \, (n \, s)$$

Nesse caso, a função $n \, s$ aplica $s$, $n$ vezes, representando o número $n$. Então, aplicamos $m$ vezes essa função, resultando em $s$ sendo aplicada $m \times n$ vezes. A multiplicação, portanto, é obtida através da repetição combinada da aplicação de $s$. Vamos usar os números de Church para calcular a multiplicação de $2 \times 2$ em cálculo lambda puro. Primeiro, representamos o número $2$ usando a definição de números de Church:

$$ 2 = \lambda s. \lambda z. \, s \, (s \, z) $$

Agora, aplicamos a função de multiplicação de números de Church:

$$ \text{mult} = \lambda m. \lambda n. \lambda s. \, m \, (n \, s) $$

Substituímos $m$ e $n$ por $2$:

$$ \text{mult} \, 2 \, 2 = (\lambda m. \lambda n. \lambda s. \, m \, (n \, s)) \, 2 \, 2 $$

Expandimos a função:

$$ = \lambda s. 2 \, (2 \, s) $$

Substituímos a representação de $2$:

$$ = \lambda s. (\lambda s. \lambda z. s \, (s \, z)) \, (2 \, s) $$

Agora resolvemos o termo $2 \, s$:

$$ 2 \, s = \lambda z. \, s \, (s \, z) $$

Substituímos isso na expressão original:

$$ = \lambda s. (\lambda z. \, s \, (s \, z)) $$

Agora aplicamos a função $2 \, s$ mais uma vez:

$$ = \lambda s. \lambda z. \, s \, (s \, (s \, (s \, z))) $$

O que representa $4$, ou seja, o resultado de $2 \times 2$ em cálculo lambda puro.

A exponenciação, por sua vez, é dada pela fórmula:

$$ \text{exp} = \lambda b. \lambda e. \, e \, b $$

Aqui, a função $e \, b$ aplica $b$, $e$ vezes. Como $b$ já é um número de Church, aplicar $e$ vezes sobre ele significa calcular $b^e$. Dessa forma, a exponenciação é realizada repetindo a aplicação da base $b$ sobre si mesma, $e$ vezes. Vamos usar os números de Church para calcular a exponenciação $2^2$ em cálculo lambda puro. Primeiro, representamos o número $2$ usando a definição de números de Church:

$$ 2 = \lambda s. \lambda z. \, s \, (s \, z) $$

Agora, aplicamos a função de exponenciação de números de Church:

$$ \text{exp} = \lambda b. \lambda e. \, e \, b $$

Substituímos $b$ e $e$ por $2$:

$$ \text{exp} \, 2 \, 2 = (\lambda b. \lambda e. \, e \, b) \, 2 \, 2 $$

Expandimos a função:

$$ = 2 \, 2 $$

Agora substituímos a representação de $2$ no lugar de $e$:

$$ = (\lambda s. \lambda z. \, s \, (s \, z)) \, 2 $$

Agora aplicamos $2$ a $b$:

$$ = (\lambda z. \, 2 \, (2 \, z)) $$

Substituímos a definição de $2$ para ambos os termos:

$$ = \lambda z. (\lambda s. \lambda z. \, s \, (s \, z)) \, (\lambda s. \lambda z. s \, (s \, z)) \, z $$

Aplicamos a função de $2$:

$$ = \lambda z. \lambda s. \, s \, (s \, (s \, (s \, z))) $$

O que representa $4$, ou seja, o resultado de $2^2$ em cálculo lambda puro.

Agora, que vimos três operações básicas, podemos expandir o conceito de números de Church e incluir mais operações aritméticas.

A subtração pode ser definida de forma mais complexa, utilizando combinadores avançados como o **combinador de predecessor**. A definição é a seguinte:

$$
\text{pred} = \lambda n. \lambda f. \lambda x. \, n (\lambda g. \lambda h.\, h\, (g\, f)) (\lambda u.\, x) (\lambda u.\, u)
$$

Esta função retorna o predecessor de $n$, ou seja, o número $n - 1$. Vamos ver um exemplo de aplicação da função $\text{pred}$ e calcular o predecessor de $3$.

$$
\begin{aligned}
\text{pred } \, 3 &= (\lambda n. \lambda f. \lambda x. \, n (\lambda g. \lambda h.\, h\, (g\, f)) (\lambda u. \, x) (\lambda u. u)) (\lambda s. \lambda z.\, s(\, s(\, s\, (z)))) \\
&= \lambda f. \lambda x. \, (\lambda s. \lambda z.\, s(\, s(\, s\, (z)))) (\lambda g. \lambda h.\, h \, (g\, f)) (\lambda u.\, x) (\lambda u.\, u) \\
&= \lambda f. \lambda x. \, f\, (f\, (x)) \\
&= 2
\end{aligned}
$$

Podemos definir a divisão como uma sequência de subtrações sucessivas e construir uma função $\text{div}$ que calcule quocientes utilizando $\text{pred}$ e $\text{mult}$. A expansão para números inteiros também pode ser feita definindo funções adicionais para lidar com números negativos.

Para definir números negativos, e assim representar o conjunto dos números inteiros, em cálculo lambda puro, usamos **pares de Church**. Cada número é representado por dois contadores: um para os sucessores (números positivos) e outro para os predecessores (números negativos). Assim, podemos simular a existência de números negativos. O número zero é definido assim:

$$\text{zero} = \lambda s. \lambda p. \, p$$

Um número positivo, como $2$, seria definido assim:

$$\text{positive-2} = \lambda s. \lambda p. \, s \, (s \, p)$$

Um número negativo, como $-2$, seria:

$$\text{negative-2} = \lambda s. \lambda p. \, p \, (p \, s)$$

Para manipular esses números, precisamos de funções de sucessor e predecessor. A função de sucessor, que incrementa um número, é definida assim:

$$\text{succ} = \lambda n. \lambda s. \lambda p. \, s \, (n \, s \, p)$$

A função de predecessor, que decrementa um número, é:

$$\text{pred} = \lambda n. \lambda s. \lambda p. p \, (n \, s \, p)$$

Com essas definições, podemos representar e manipular números positivos e negativos. Sucessores são aplicados para aumentar números positivos, e predecessores são usados para aumentar números negativos.

Agora que temos uma representação para números negativos, vamos calcular a soma de $3 + (-2)$ usando cálculo lambda puro. Primeiro, representamos os números $3$ e $-2$ usando as definições de pares de Church:

$$3 = \lambda s. \lambda p. \, s \, (s \, (s \, p))$$
$$-2 = \lambda s. \lambda p. \, p \, (p \, s)$$

A função de adição precisa ser adaptada para lidar com pares de Church (positivos e negativos). Aqui está a nova definição de adição:

$$\text{add} = \lambda m. \lambda n. \lambda s. \lambda p. \, m \, s \, (n \, s \, p)$$

Substituímos $m$ por $3$ e $n$ por $-2$:

$$\text{add} \, 3 \, (-2) = (\lambda m. \lambda n. \lambda s. \lambda p. \, m \, s \, (n \, s \, p)) \, 3 \, (-2)$$

Expandimos a função:

$$= \lambda s. \lambda p. \, 3 \, s \, (-2 \, s \, p)$$

Substituímos a representação de $3$ e $-2$:

$$= \lambda s. \lambda p. (\lambda s. \lambda p. \, s \, (s \, (s \, p))) \, s \, ((\lambda s. \lambda p. \, p \, (p \, s)) \, s \, p)$$

Resolvemos primeiro o termo $-2 \, s \, p$:

$$= p \, (p \, s) $$

Agora aplicamos o termo $3 \, s$ ao resultado:

$$= s \, (s \, (p \, s))$$

O que representa $1$, ou seja, o resultado de $3 + (-2)$ em cálculo lambda puro.

Para implementar a divisão corretamente no cálculo lambda puro, precisamos do combinador $Y$, que permite recursão. O combinador $Y$ veremos o combinador $Y$ com mais cuidado em outra parte deste texto, por enquanto, basta saber que ele é definido assim:

$$ Y = \lambda f. (\lambda x. f \, (x \, x)) (\lambda x. f \, (x \, x)) $$

Agora, podemos definir a função de divisão com o combinador $Y$. Primeiro, definimos a lógica de subtração repetida:

$$ \text{divLogic} = \lambda f. \lambda m. \lambda n. \lambda s. \lambda z. \, \text{ifZero} \, (\text{sub} \, m \, n) \, z \, (\lambda q. \text{succ} \, (f \, (\text{sub} \, m \, n) \, n \, s \, z)) $$

Aplicamos o combinador Y para permitir a recursão:

$$ \text{div} = Y \, \text{divLogic} $$

Para chegar a divisão precisamos utilizar o combinador $Y$ que permite a recursão necessária para subtrair repetidamente o divisor do dividendo. A função $\text{ifZero}$, que não definimos anteriormente usada para verificar se o resultado da subtração é zero ou negativo. A função $\text{sub}$ subtrai $n$ de $m$. Finalmente usamos a função $\text{succ}$ para contar quantas vezes subtraímos o divisor.

A capacidade de representar números e operações aritméticas no cálculo lambda mostra que ele pode expressar não só computações lógicas e funcionais, mas também manipulações concretas de números. Isso mostra que o cálculo lambda pode representar qualquer função computável, como afirma a Tese de Church-Turing.

Ao definir números inteiros — positivos, negativos, e zero — e as operações de adição, subtração, multiplicação e divisão, o cálculo lambda mostra que pode ser uma base teórica para linguagens de programação que manipulam dados numéricos. Isso ajuda a entender a relação entre matemática e computação.

Definir números inteiros e operações básicas no cálculo lambda demonstra a universalidade do sistema. Ao mostrar que podemos modelar aritmética e computação de forma puramente funcional, o cálculo lambda prova que pode representar qualquer função computável, inclusive as aritméticas.

Já que modelamos a aritmética podemos avançar para a implementação desses conceitos em Haskell, começando pelos números de Church:

```haskell
-- Números de Church em Haskell
type Church a = (a -> a) -> a -> a

zero :: Church a
zero = \f -> \x -> x

one :: Church a
one = \f -> \x -> f \, x

two :: Church a
two = \f -> \x -> f (f \, x)

three :: Church a
three = \f -> \x -> f (f (f \, x))

-- Função sucessor
succ' :: Church a -> Church a
succ' n = \f -> \x -> f (n f \, x)

-- Adição
add :: Church a -> Church a -> Church a
add m n = \f -> \x -> m f (n f \, x)

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

Haskell é a principal linguagem de programação funcional. Talvez muitos de vocês não a conheçam. Por isso, vamos explorar a implementação das operações aritméticas dos números de Church (adição e multiplicação) em C++20 e Python. Vamos ver como essas operações são transformações e como podemos implementá-las nessas duas linguagens que são meu foco no momento.

A função sucessor aplica uma função $f$ a um argumento $z$ uma vez a mais do que o número existente. Aqui está uma implementação da função sucessor em C++20.

```cpp
# include <iostream> // Standard library for input and output
# include <functional> // Standard library for std::function, used for higher-order functions

// Define a type alias `Church` that represents a Church numeral.
// A Church numeral is a higher-order function that takes a function f and returns another function.
using Church = std::function<std::function<int(int)>(std::function<int(int)>)>;

// Define the Church numeral for 0.
// `zero` is a lambda function that takes a function `f` and returns a lambda that takes an integer `x` and returns `x` unchanged.
// This is the definition of 0 in Church numerals, which means applying `f` zero times to `x`.
Church zero = [](auto f) {
 return [f](int x) { return x; }; // Return the identity function, applying `f` zero times.
};

// Define the successor function `succ` that increments a Church numeral by 1.
// `succ` is a lambda function that takes a Church numeral `n` (a number in Church encoding) and returns a new function.
// The new function applies `f` to the result of applying `n(f)` to `x`, effectively adding one more application of `f`.
Church succ = [](Church n) {
 return [n](auto f) {
 return [n, f](int x) {
 return f(n(f)(x)); // Apply `f` one more time than `n` does.
 };
 };
};

// Convert a Church numeral to a standard integer.
// `to_int` takes a Church numeral `n`, applies the function `[](int x) { return x + 1; }` to it, 
// which acts like a successor function in the integer world, starting from 0.
int to_int(Church n) {
 return n([](int x) { return x + 1; })(0); // Start from 0 and apply `f` the number of times encoded by `n`.
}

int main() {
 // Create the Church numeral for 1 by applying `succ` to `zero`.
 auto one = succ(zero);

 // Create the Church numeral for 2 by applying `succ` to `one`.
 auto two = succ(one);

 // Output the integer representation of the Church numeral `two`.
 std::cout << "Sucessor de 1: " << to_int(two) << std::endl;

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
print("Sucessor de 1:", to_int(two)) # This will print 2, the successor of 1 in Church numerals.
```

Poderíamos refazer todas as operações em cálculo lambda que demonstramos anteriormente em cálculo lambda puro, contudo este não é o objetivo deste texto.

A representação dos números naturais no cálculo lambda mostra como um sistema simples de funções pode codificar estruturas matemáticas complexas. Ela também nos dá uma visão sobre a natureza da computação e a expressividade de sistemas baseados em funções. Isso prova a universalidade do cálculo lambda, mostrando que ele pode representar funções e dados.

Essa representação também serve de base para sistemas de tipos em linguagens de programação funcionais. Ela mostra como abstrações matemáticas podem ser codificadas em funções puras. Embora linguagens como Haskell não usem diretamente os números de Church, o conceito de representar dados como funções é essencial. Em Haskell, por exemplo, listas são manipuladas com funções de ordem superior que se parecem com os números de Church.

Os números de Church mostram como o cálculo lambda pode codificar dados complexos e operações usando apenas funções. Eles dão uma base sólida para entender computação e abstração em linguagens de programação.

## Lógica Proposicional no Cálculo Lambda

O cálculo lambda oferece uma representação formal para lógica proposicional, similar aos números de Church para os números naturais. Ele pode codificar valores de verdade e operações lógicas como funções. Essa abordagem permite que operações booleanas sejam realizadas através de expressões funcionais. Neste caso, os dois valores de verdade fundamentais, _True_ (Verdadeiro) e _False_ (Falso), podem ser representados da seguinte maneira:

- **True**: $\text{True} = \lambda x. \lambda y. \, x$

- **False**: $\text{False} = \lambda x. \lambda y. \, y$

Aqui, _True_ é uma função que quando aplicada a dois argumentos, retorna o primeiro, enquanto _False_ retorna o segundo. Estes são os fundamentos sobre os quais todas as operações lógicas podem ser construídas. Podemos começar com as operações fundamentais da lógica proposicional: negação (Not), conjunção (And), disjunção (Or), disjunção exclusiva (Xor) e condicional (If-Then-Else).

A operação de **negação**, que inverte o valor de uma proposição, pode ser definida como:

$$\text{Not} = \lambda b. \, b \, \text{False} \, \text{True}$$

Esta função recebe um valor booleano $b$. Se $b$ for _True_, ela retorna _False_; caso contrário, retorna _True_.

Podemos avaliar $\text{Not} \, \text{True}$ como exemplo:

$$
\begin{align*}
\text{Not} \, \text{True} &= (\lambda b. \, b \, \text{False} \, \text{True}) \, \text{True} \\
&\to_\beta \text{True} \, \text{False} \, \text{True} \\
&= (\lambda x. \lambda y. \, x) \, \text{False} \, \text{True} \\
&\to_\beta (\lambda y. \, \text{False}) \, \text{True} \\
&\to_\beta \text{False}
\end{align*}
$$

A operação de **conjunção** retorna _True_ apenas se ambos os operandos forem _True_. No cálculo lambda, isso pode ser expresso como:

$$\text{And} = \lambda x. \lambda y. \, x \, y \, \text{False}$$

Vamos avaliar $\text{And} \, \text{True} \, \text{False}$ primeiro:

$$
\begin{align*}
&\text{And} \, \text{True} \, \text{False} \\
&= (\lambda x. \lambda y. \, x \, y \, \text{False}) \, \text{True} \, \text{False} \\
\\
&\text{Substituímos $\text{True}$, $\text{False}$ e $\text{And}$ por suas definições em cálculo lambda:} \\
&= (\lambda x. \lambda y. \, x \, y \, (\lambda x. \lambda y. \, y)) \, (\lambda x. \lambda y. \, x) \, (\lambda x. \lambda y. \, y) \\
\\
&\text{Aplicamos a primeira redução beta, substituindo $x$ por $ (\lambda x. \lambda y. \, x)$ na função $\text{And}$:} \\
&\to_\beta (\lambda y. \, (\lambda x. \lambda y. \, x) \, y \, (\lambda x. \lambda y. \, y)) \, (\lambda x. \lambda y. \, y) \\
\\
&\text{Nesta etapa, a substituição de $x$ por $ (\lambda x. \lambda y. \, x)$ resulta em uma nova função que depende de $y$. A expressão interna aplica $\text{True}$ ($ \lambda x. \lambda y. \, x$) ao argumento $y$ e ao $\text{False}$ ($ \lambda x. \lambda y. \, y$).} \\
\\
&\text{Agora, aplicamos a segunda redução beta, substituindo $y$ por $ (\lambda x. \lambda y. \, y)$:} \\
&\to_\beta (\lambda x. \lambda y. \, x) \, (\lambda x. \lambda y. \, y) \, (\lambda x. \lambda y. \, y) \\
\\
&\text{A substituição de $y$ por $\text{False}$ resulta na expressão acima. Aqui, $\text{True}$ é aplicada ao primeiro argumento $\text{False}$, ignorando o segundo argumento.} \\
\\
&\text{Aplicamos a próxima redução beta, aplicando $ \lambda x. \lambda y. \, x$ ao primeiro argumento $ (\lambda x. \lambda y. \, y)$:} \\
&\to_\beta \lambda y. \, (\lambda x. \lambda y. \, y) \\
\\
&\text{Neste ponto, temos uma função que, quando aplicada a $y$, sempre retorna $\text{False}$, já que $ \lambda x. \lambda y. \, x$ retorna o primeiro argumento.} \\
\\
&\text{Finalmente, aplicamos a última redução beta, que ignora o argumento de $\lambda y$ e retorna diretamente $\text{False}$:} \\
&\to_\beta \lambda x. \lambda y. \, y \\
\\
&\text{Esta é exatamente a definição de $\text{False}$ no cálculo lambda.} \\
\\
&\text{Portanto, o resultado final é:} \\
&= \text{False}
\end{align*}
$$

Podemos entender a **conjunção** de uma forma mais fácil se usarmos as funções de ordem superior definidas por Church para $True$ e $False:

$$
\begin{align*}
\text{And} \, \text{True} \, \text{False} &= (\lambda x. \lambda y. \, x \, y \, \text{False}) \, \text{True} \, \text{False} \\
&\to_\beta (\lambda y. \, \text{True} \, y \, \text{False}) \, \text{False} \\
&\to_\beta \text{True} \, \text{False} \, \text{False} \\
&= (\lambda x. \lambda y. \, x) \, \text{False} \, \text{False} \\
&\to_\beta (\lambda y. \, \text{False}) \, \text{False} \\
&\to_\beta \text{False}
\end{align*}
$$

A operação de **disjunção** retorna _True_ se pelo menos um dos operandos for _True_. Ela pode ser definida assim:

$$
\text{Or} = \lambda x. \lambda y. \, x \, \text{True} \, y
$$

Vamos avaliar $\text{Or} \, \text{True} \, \text{False}$:

$$
\begin{align*}
\text{Or} \, \text{True} \, \text{False} &= (\lambda x. \lambda y. \, x \, \text{True} \, y) \, \text{True} \, \text{False} \\
&\to_\beta (\lambda y. \, \text{True} \, \text{True} \, y) \, \text{False} \\
&\to_\beta \text{True} \, \text{True} \, \text{False} \\
&= (\lambda x. \lambda y. \, x) \, \text{True} \, \text{False} \\
&\to_\beta (\lambda y. \, \text{True}) \, \text{False} \\
&\to_\beta \text{True}
\end{align*}
$$

A operação _Xor_ (ou **disjunção exclusiva**) retorna _True_ se um, e apenas um, dos operandos for _True_. Sua definição no cálculo lambda será dada por:

$$
\text{Xor} = \lambda b. \lambda c. b \, (\text{Not} \, c) \, c
$$

Para entender, podemos avaliar $\text{Xor} \, \text{True} \, \text{False}$:

$$
\begin{align*}
\text{Xor} \, \text{True} \, \text{False} &= (\lambda b. \lambda c. b \, (\text{Not} \, c) \, c) \, \text{True} \, \text{False} \\
&\to_\beta (\lambda c. \text{True} \, (\text{Not} \, c) \, c) \, \text{False} \\
&\to_\beta \text{True} \, (\text{Not} \, \text{False}) \, \text{False} \\
&\to_\beta \text{True} \, \text{True} \, \text{False} \\
&= (\lambda x. \lambda y. \, x) \, \text{True} \, \text{False} \\
&\to_\beta (\lambda y. \, \text{True}) \, \text{False} \\
&\to_\beta \text{True}
\end{align*}
$$

A operação **condicional**, ou **implicação**, também conhecida como _If-Then-Else_, pode ser definida no cálculo lambda como:

$$
\text{If} = \lambda b. \lambda x. \lambda y. \, b \, x \, y
$$

Essa operação retorna $x$ se $b$ for _True_ e $y$ se $b$ for _False_.

Novamente, a avaliação de um exemplo pode melhorar o entendimento. Vamos avaliar $\text{If} \, \text{True} \, A \, B$:

$$
\begin{align*}
\text{If} \, \text{True} \, A \, B &= (\lambda b. \lambda x. \lambda y. \, b \, x \, y) \, \text{True} \, A \, B \\
&\to_\beta (\lambda x. \lambda y. \, \text{True} \, x \, y) \, A \, B \\
&\to_\beta (\lambda y. \, \text{True} \, A \, y) \, B \\
&\to_\beta \text{True} \, A \, B \\
&= (\lambda x. \lambda y. \, x) \, A \, B \\
&\to_\beta (\lambda y. \, A) \, B \\
&\to_\beta A
\end{align*}
$$

Como a **conjunção** é importante para a construção de linguagens lógicas, vamos avaliar $\text{Not} \, (\text{And} \, \text{True} \, \text{False})$:

$$
\begin{align*}
\text{Not} \, (\text{And} \, \text{True} \, \text{False}) &= (\lambda b. b \, \text{False} \, \text{True}) \, ( (\lambda x. \lambda y. \, x \, y \, \text{False}) \, \text{True} \, \text{False}) \\
&\to_\beta (\lambda b. b \, \text{False} \, \text{True}) \, ( (\lambda y. \, \text{True} \, y \, \text{False}) \, \text{False}) \\
&\to_\beta (\lambda b. b \, \text{False} \, \text{True}) \, (\text{True} \, \text{False} \, \text{False}) \\
&\to_\beta (\lambda b. b \, \text{False} \, \text{True}) \, (\lambda x. \lambda y. \, x) \, \text{False} \, \text{False} \\
&\to_\beta \text{False}
\end{align*}
$$

Como resultado, a expressão retorna _False_, como esperado.

## Funções Recursivas e o Combinador Y no Cálculo Lambda

No cálculo lambda, uma linguagem puramente funcional, não há uma forma direta de definir funções recursivas. Isso acontece porque, ao tentar criar uma função que se refere a si mesma, como o fatorial, acabamos com uma definição circular que o cálculo lambda puro não consegue resolver. Uma tentativa ingênua de definir o fatorial seria:

$$
\text{fac} = \lambda n.\, \text{if } (n = 0) \, \text{then } 1 \, \text{else } n \cdot (\text{fac} \, (n - 1))
$$

Aqui, $\text{fac}$ aparece nos dois lados da equação, criando uma dependência circular. No cálculo lambda puro, não existem nomes ou atribuições; tudo se baseia em funções anônimas. _Portanto, não é possível referenciar $\text{fac}$ dentro de sua própria definição._

No cálculo lambda, todas as funções são anônimas. Não existem variáveis globais ou nomes fixos para funções. As únicas formas de vincular variáveis são:

- **Abstração lambda**: $\lambda x.\, e$, onde $x$ é um parâmetro e $e$ é o corpo da função.
- **Aplicação de função**: $(f\, a)$, onde $f$ é uma função e $a$ é um argumento.

Não há um mecanismo para definir uma função que possa se referenciar diretamente. Na definição:

$$
\text{fac} = \lambda n.\, \text{if } (n = 0) \, \text{then } 1 \, \text{else } n \cdot (\text{fac} \, (n - 1))
$$

queremos que $\text{fac}$ possa chamar a si mesma. Mas no cálculo lambda puro:

1. **Não há nomes persistentes**: O nome $\text{fac}$ do lado esquerdo não está disponível no corpo da função à direita. Nomes em abstrações lambda são apenas parâmetros locais.

2. **Variáveis livres devem ser vinculadas**: $\text{fac}$ aparece livre no corpo e não está ligada a nenhum parâmetro ou contexto. Isso viola as regras do cálculo lambda.

3. **Sem referência direta a si mesmo**: Não se pode referenciar uma função dentro de si mesma, pois não existe um escopo que permita isso.

Considere uma função simples no cálculo lambda:

$$\text{função} = \lambda x.\, x + 1$$

Esta função está bem definida. Mas, se tentarmos algo recursivo:

$$\text{loop} = \lambda x.\, (\text{loop}\, x)$$

O problema é o mesmo: $\text{loop}$ não está definido dentro do corpo da função. Não há como a função chamar a si mesma sem um mecanismo adicional.

Em linguagens de programação comuns, definimos funções recursivas porque o nome da função está disponível dentro do escopo. Em Python, por exemplo:

```python
def fac(n):
 if n == 0:
 return 1
 else:
 return n * fac(n - 1)
```

Aqui, o nome `fac` está disponível dentro da função. No cálculo lambda, essa forma de vinculação não existe.

### O Combinador $Y$ como Solução

Para contornar essa limitação, usamos o conceito de **ponto fixo**. Um ponto fixo de uma função $F$ é um valor $X$ tal que $F(X) = X$. No cálculo lambda, esse conceito é implementado por meio de combinadores de ponto fixo, sendo o mais conhecido o combinador $Y$, atribuído a Haskell Curry.

O combinador $Y$ é definido como:

$$Y = \lambda f. (\lambda x. \, f \, (x \, x)) \, (\lambda x. \, f \, (x \, x))$$

Para ilustrar o funcionamento do Y-combinator na prática, vamos implementá-lo em Haskell e usá-lo para definir a função fatorial:

```haskell
-- Definição do Y-combinator
y :: (a -> a) -> a
y f = f (y f)

-- Definição da função fatorial usando o Y-combinator
factorial :: Integer -> Integer
factorial = y $ \f n -> if n == 0 then 1 else n * f (n - 1)

main :: IO ()
main = do
 print $ factorial 5 -- Saída: 120
 print $ factorial 10 -- Saída: 3628800
```

Neste exemplo, o Y-combinator (y) é usado para criar uma versão recursiva da função fatorial sem a necessidade de defini-la recursivamente de forma explícita. A função factorial é criada aplicando y a uma função que descreve o comportamento do fatorial, mas sem se referir diretamente a si mesma.
Podemos estender este exemplo para outras funções recursivas, como a sequência de Fibonacci:

```haskell
fibonacci :: Integer -> Integer
fibonacci = y $ \f n -> if n <= 1 then n else f (n - 1) + f (n - 2)

main :: IO ()
main = do
 print $ map fibonacci [0..10] -- Saída: [0,1,1,2,3,5,8,13,21,34,55]
```

Além disso, o Y-combinator, ou combinador-Y, tem uma propriedade muito interessante:

$$Y \, F = F \, (Y \, F)$$

Isso significa que $Y \, F$ é um ponto fixo de $F$, permitindo que definamos funções recursivas sem a necessidade de auto-referência explícita. Quando aplicamos o combinador $Y$ a uma função $F$, ele retorna uma versão recursiva de $F$.

Matematicamente, o combinador $Y$ cria a recursão ao forçar a função $F$ a se referenciar indiretamente. O processo ocorre da seguinte maneira:

1. Aplicamos o combinador $Y$ a uma função $F$.
2. O $Y$ retorna uma função que, ao ser chamada, aplica $F$ a si mesma repetidamente.
3. Essa recursão acontece até que uma condição de término, como o caso base de uma função recursiva, seja atingida.

Com o combinador $Y$, não precisamos declarar explicitamente a recursão. O ciclo de auto-aplicação é gerado automaticamente, transformando qualquer função em uma versão recursiva de si mesma.

### Exemplo de Função Recursiva: Fatorial

Usando o combinador $Y$, podemos definir corretamente a função fatorial no cálculo lambda. O fatorial de um número $n$ é:

$$
\text{factorial} = Y \, (\lambda f. \lambda n. \text{if} \, (\text{isZero} \, n) \, 1 \, (\text{mult} \, n \, (f \, (\text{pred} \, n))))
$$

Aqui, utilizamos funções auxiliares como $\text{isZero}$, $\text{mult}$ (multiplicação), e $\text{pred}$ (predecessor), todas definíveis no cálculo lambda. O combinador $Y$ cuida da recursão, aplicando a função a si mesma até que a condição de parada ($n = 0$) seja atendida. Vamos ver isso com mais detalhes usando o combinador $Y$ para definir $\text{fac}$

1. **Defina uma função auxiliar que recebe como parâmetro a função recursiva**:

 $$
 \text{Fac} = \lambda f.\, \lambda n.\, \text{if } (n = 0) \, \text{then } 1 \, \text{else } n \cdot (f\, (n - 1))
 $$

 Aqui, $\text{Fac}$ é uma função que, dado um função $f$, retorna outra função que calcula o fatorial usando $f$ para a chamada recursiva.

2. **Aplique o combinador $Y$ a $\text{Fac}$ para obter a função recursiva**:

 $$\text{fac} = Y\, \text{Fac}$$

 Agora, $\text{fac}$ é uma função que calcula o fatorial de forma recursiva.

O combinador $Y$ aplica $\text{Fac}$ a si mesmo de maneira que $\text{fac}$ se expande indefinidamente, permitindo as chamadas recursivas sem referência direta ao nome da função.

**Exemplo 1**:

Vamos calcular $\text{fac}\, 2$ usando o combinador Y.

 **Combinador Y**:

 $$Y = \lambda f.\, (\lambda x.\, f\, (x\, x))\, (\lambda x.\, f\, (x\, x))$$

 **Função Fatorial**:

 $$\text{fatorial} = Y\, \left (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))) \right)$$

 **Expansão da Definição de $\text{fatorial}$**:

 Aplicamos $Y$ à função $\lambda f.\, \lambda n.\, \ldots$:

 $$\text{fatorial} = Y\, (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))$$

 Então,

 $$\text{fatorial}\, 2 = \left( Y\, (\lambda f.\, \lambda n.\, \ldots) \right)\, 2$$

 **Expandindo o Combinador Y**:

 O Combinador Y é definido como:

 $$Y\, g = (\lambda x.\, g\, (x\, x))\, (\lambda x.\, g\, (x\, x))$$

 Aplicando $Y$ à função $g = \lambda f.\, \lambda n.\, \ldots$:

 $$Y\, g = (\lambda x.\, g\, (x\, x))\, (\lambda x.\, g\, (x\, x))$$

 Portanto,

 $$\text{fatorial} = (\lambda x.\, (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))\, (x\, x))\, (\lambda x.\, (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))\, (x\, x))$$

 **Aplicando $\text{fatorial}$ a 2**

 Agora, calculamos $\text{fatorial}\, 2$:

 $$\text{fatorial}\, 2 = \left( (\lambda x.\, \ldots)\, (\lambda x.\, \ldots) \right)\, 2$$

 **Simplificando as Aplicações**:

 Vamos simplificar a expressão passo a passo.

 **Primeira Aplicação**:

 $$\text{fatorial}\, 2 = \left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right)\, 2$$

 Onde $F = \lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1)))$.

 **Aplicando o Primeiro**: $\lambda x$

 $$\left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right) = F\, \left (\left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right) \right)$$

 Note que temos uma autorreferência aqui. Vamos denotar:

 $$M = \left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right)$$

 Portanto,

 $$\text{fatorial}\, 2 = F\, M\, 2$$

 **Aplicando $F$ com $M$ e $n = 2$:**

 $$F\, M\, 2 = (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))\, M\, 2$$

 Então,

 $$\text{if}\, (2 = 0)\, 1\, (2 \times (M\, (2 - 1)))$$

 Como $2 \ne 0$, calculamos:

 $$\text{fatorial}\, 2 = 2 \times (M\, 1)$$

 **Calculando $M\, 1$:**

 Precisamos calcular $M\, 1$, onde $M$ é:

 $$M = \left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right)$$

 Então,

 $$M\, 1 = \left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right)\, 1 = F\, M\, 1$$

 Novamente, temos:

 $$\text{fatorial}\, 2 = 2 \times (F\, M\, 1)$$

 **Aplicando $F$ com $M$ e $n = 1$:**

 $$F\, M\, 1 = (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))\, M\, 1$$

 Então,

 $$\text{if}\, (1 = 0)\, 1\, (1 \times (M\, (1 - 1)))$$

 Como $1 \ne 0$, temos:

 $$F\, M\, 1 = 1 \times (M\, 0)$$

 **Calculando $M\, 0$:**

 $$M\, 0 = \left (\lambda x.\, F\, (x\, x) \right)\, \left (\lambda x.\, F\, (x\, x) \right)\, 0 = F\, M\, 0$$

 Aplicando $F$ com $n = 0$:

 $$F\, M\, 0 = (\lambda f.\, \lambda n.\, \text{if}\, (n = 0)\, 1\, (n \times (f\, (n - 1))))\, M\, 0$$

 Como $0 = 0$, temos:

 $$F\, M\, 0 = 1$$

 **Concluindo os Cálculos:**

 $$M\, 0 = 1$$

 $$F\, M\, 1 = 1 \times 1 = 1$$

 $$\text{fatorial}\, 2 = 2 \times 1 = 2$$

 Portanto, o cálculo do fatorial de 2 é:

 $$\text{fatorial}\, 2 = 2$$

**Exemplo 2**:

Agora, vamos verificar o cálculo de $\text{fatorial}\, 3$ seguindo o mesmo procedimento.

 **Aplicando $\text{fatorial}$ a 3**:

 $$\text{fatorial}\, 3 = F\, M\, 3$$

 Onde $F$ e $M$ são como definidos anteriormente.

 **Aplicando $F$ com $n = 3$**

 $$\text{if}\, (3 = 0)\, 1\, (3 \times (M\, (3 - 1)))$$

 Como $3 \ne 0$, temos:

 $$\text{fatorial}\, 3 = 3 \times (M\, 2)$$

 **Calculando $M\, 2$**:

 Seguindo o mesmo processo:

 $$M\, 2 = F\, M\, 2$$

 $$F\, M\, 2 = 2 \times (M\, 1)$$

 $$M\, 1 = F\, M\, 1$$

 $$F\, M\, 1 = 1 \times (M\, 0)$$

 $$M\, 0 = F\, M\, 0 = 1$$

 **Calculando os Valores**:

 $$M\, 0 = 1$$

 $$F\, M\, 1 = 1 \times 1 = 1$$

 $$M\, 1 = 1$$

 $$F\, M\, 2 = 2 \times 1 = 2$$

 $$M\, 2 = 2$$

 $$\text{fatorial}\, 3 = 3 \times 2 = 6$$

 Portanto, o cálculo do fatorial de 3 é:

 $$\text{fatorial}\, 3 = 6$$

### Usando Funções de Ordem Superior

Vamos rever o combinador $Y$, desta vez, usando funções de ordem superior. Começamos definindo algumas funções de ordem superior.

 **$\text{isZero}$**:

 $$\text{isZero} = \lambda n.\, n\, (\lambda x.\, \text{false})\, \text{true}$$

 **$\text{mult}$:**

 $$\text{mult} = \lambda m.\, \lambda n.\, \lambda f.\, m\, (n\, f)$$

 **$\text{pred}$ (Predecessor):**

 $$\text{pred} = \lambda n.\, \lambda f.\, \lambda x.\, n\, (\lambda g.\, \lambda h.\, h\, (g\, f))\, (\lambda u.\, x)\, (\lambda u.\, u)$$

Agora vamos definir a função fatorial usando o Combinador Y e as funções acima:

$$\text{fatorial} = Y\, \left (\lambda f.\, \lambda n.\, \text{if}\, (\text{isZero}\, n)\, 1\, (\text{mult}\, n\, (f\, (\text{pred}\, n))) \right)$$

**Exemplo 1**: Vamos verificar se $\text{fatorial}\, 2 = 2$ usando estas definições.

 **Aplicação da Função:**

 $$\text{fatorial}\, 2 = F\, M\, 2$$

 Onde $F$ e $M$ são definidos de forma análoga.

 **Aplicando $F$ com $n = 2$:**

 $$\text{if}\, (\text{isZero}\, 2)\, 1\, (\text{mult}\, 2\, (M\, (\text{pred}\, 2)))$$

 Como $\text{isZero}\, 2$ é $\text{false}$, continuamos:

 Calcule $\text{pred}\, 2 = 1$
 Calcule $M\, 1$

 **Recursão:**

 $$M\, 1 = F\, M\, 1$$

 $$\text{fatorial}\, 1 = \text{mult}\, 1\, (M\, (\text{pred}\, 1))$$

 **Caso Base:**

 $$\text{pred}\, 1 = 0$$

 $$\text{isZero}\, 0 = \text{true}$$

 então,

 $$\text{fatorial}\, 0 = 1$$

 **Calculando os Valores:**

 $$\text{fatorial}\, 1 = \text{mult}\, 1\, 1 = 1$$

 $$\text{fatorial}\, 2 = \text{mult}\, 2\, 1 = 2$$

A função fatorial é um exemplo clássico de recursão. Entretanto, podemos definir uma função de exponenciação, recursiva, para calcular $m^n$:

$$\text{power} = Y \, (\lambda f. \lambda m. \lambda n. \text{if} \, (\text{isZero} \, n) \, 1 \, (\text{mult} \, m \, (f \, m \, (\text{pred} \, n))))$$

Assim como no fatorial, o combinador $Y$ permite a definição recursiva sem auto-referência explícita. Mas, esta demonstração ficará a seu cargo.

## Representação de Valores e Computações

Uma das características principais do cálculo lambda é representar valores, dados e computações complexas, usando apenas funções. Até números e _booleanos_ são representados de forma funcional. Um exemplo indispensável é a representação dos números naturais, chamada **Numerais de Church**:

$$
\begin{align*}
0 &= \lambda s. \, \lambda z. \, z \\
1 &= \lambda s. \, \lambda z. \, s \, z \\
2 &= \lambda s. \, \lambda z. s \, (s \, z) \\
3 &= \lambda s. \, \lambda z. \, s \, (s (s \, z))
\end{align*}
$$

Voltaremos a esta notação mais tarde. O importante é que essa codificação permite que operações aritméticas sejam definidas inteiramente em termos de funções. Por exemplo, a função sucessor pode ser expressa como:

$$
\text{succ} = \lambda n. \, \lambda s. \, \lambda z. \, s \, (n \, s \, z)
$$

Assim, operações como adição e multiplicação também podem ser construídas de maneira funcional, respeitando a estrutura funcional do cálculo lambda.

Um dos resultados mais profundos da formalização da computabilidade, utilizando o cálculo lambda e as máquinas de Turing, foi a identificação de problemas _indecidíveis_. Problemas para os quais não podemos decidir se o algoritmo que os resolve irá parar em algum ponto, ou não.

O exemplo mais emblemático é o problema da parada, formulado por Alan Turing em 1936. O problema da parada questiona se é possível construir um algoritmo que, dado qualquer programa e uma entrada, determine se o programa eventualmente terminará ou continuará a executar indefinidamente. Em termos formais, essa questão pode ser expressa como:

$$
\text{Existe } f : \text{Programa} \times \text{Entrada} \rightarrow \{\text{Para}, \text{NãoPara}\}?
$$

Turing demonstrou, por meio de um argumento de diagonalização, que tal função $f$ não pode existir. Esse resultado mostra que não é possível determinar, de forma algorítmica, o comportamento de todos os programas para todas as possíveis entradas..

Outro problema indecidível, elucidado pelas descobertas em computabilidade, é o _décimo problema de Hilbert_[^nota1]. Esse problema questiona se existe um algoritmo que, dado um polinômio com coeficientes inteiros, possa determinar se ele possui soluções inteiras. Formalmente, o problema pode ser expresso assim:

$$
P(x_1, x_2, \dots, x_n) = 0
$$

Em 1970, [Yuri Matiyasevich](Yuri Matiyasevich), em colaboração com [Julia Robinson](https://en.wikipedia.org/wiki/Julia_Robinson), [Martin Davis](<https://en.wikipedia.org/wiki/Martin_Davis_(mathematician)>) e [Hilary Putnam](https://en.wikipedia.org/wiki/Hilary_Putnam), provou que tal algoritmo não existe. Esse resultado teve implicações profundas na teoria dos números e demonstrou a indecidibilidade de um problema central na matemática.

A equivalência entre o cálculo lambda, as máquinas de Turing e as funções recursivas permitiu estabelecer os limites da computação algorítmica. O problema da parada e outros resultados indecidíveis, como o décimo problema de Hilbert, mostraram que existem problemas além do alcance dos algoritmos.

A **Tese de Church-Turing** formalizou essa ideia, afirmando que qualquer função computável pode ser expressa por um dos modelos computacionais mencionados, Máquina de Turing, recursão e o cálculo lambda[^cita6]. Essa tese forneceu a base rigorosa necessária ao desenvolvimento da ciência da computação, permitindo a demonstração da existência de problemas não solucionáveis por algoritmos.

## O Cálculo Lambda e a Lógica

O cálculo lambda possui uma relação direta com a lógica matemática, especialmente através do **isomorfismo de Curry-Howard**. Esse isomorfismo cria uma correspondência entre provas matemáticas e programas computacionais. Em termos simples, uma prova de um teorema é um programa que constrói um valor a partir de uma entrada, e provar teoremas equivale a computar funções.

Essa correspondência deu origem ao paradigma das _provas como programas_[^nota2] . O cálculo lambda define computações e serve como uma linguagem para representar e verificar a correção de algoritmos. Esse conceito se expandiu na pesquisa moderna e fundamenta muitos assistentes de prova e linguagens de programação com sistemas de tipos avançados, como o **Sistema F**[^nota3] e o **Cálculo de Construções**[^nota4].

O cálculo lambda continua a influenciar a ciência da computação. O desenvolvimento do cálculo lambda tipado levou à criação de sistemas de tipos complexos, fundamentais para a verificação formal de software e para linguagens de programação modernas, como Haskell, Coq e Agda. Esses sistemas garantem propriedades de programas, como segurança e correção, utilizando princípios do cálculo lambda.

O cálculo lambda não é apenas um conceito teórico abstrato; ele possui implicações práticas, especialmente na programação funcional. Ele é o alicerce teórico sobre o qual muitas linguagens de programação funcional se apoiam. Linguagens como Lisp, Haskell, OCaml e F# incorporam princípios do cálculo lambda. Exemplos incluem:

1. **Funções como cidadãos de primeira classe**: No cálculo lambda, funções são valores. Podem ser passadas como argumentos, retornadas como resultados e manipuladas livremente. Isso é um princípio central da programação funcional.

2. **Funções de ordem superior**: O cálculo lambda permite a criação de funções que operam sobre outras funções. Isso se traduz em conceitos como `map`, `filter` e `reduce` em linguagens funcionais.

3. **Currying**: A técnica de transformar uma função com múltiplos argumentos em uma sequência de funções de um único argumento é natural no cálculo lambda.

4. **Avaliação preguiçosa (_lazy_)**: Embora não faça parte do cálculo lambda puro, a semântica de redução do cálculo lambda inspirou o conceito de avaliação preguiçosa em linguagens como Haskell.

5. **Recursão**: Definir funções recursivas é essencial em programação funcional. No cálculo lambda, isso é feito com combinadores de ponto fixo.

A correspondência [Curry-Howard](https://groups.seas.harvard.edu/courses/cs152/2021sp/lectures/lec15-curryhoward.pdf), também conhecida como **isomorfismo proposições-como-tipos**, estabelece uma relação entre sistemas de tipos em linguagens de programação e sistemas lógicos. Especificamente, ela indica que programas correspondem a provas, tipos correspondem a proposições lógicas e a avaliação de programas corresponde à simplificação de provas. Isso fornece uma base teórica para a relação entre programação e lógica matemática, influenciando o desenvolvimento de linguagens de programação e sistemas de prova formais.

# Estruturas de Dados Compostas

Embora o cálculo lambda puro não possua estruturas de dados nativas, podemos representá-las usando funções. Um exemplo clássico é a codificação de listas no estilo de Church, que nos permite aplicar recursão a essas estruturas.

## Representação de Listas no Cálculo Lambda

No cálculo lambda, representamos listas usando funções. Esta codificação permite manipular listas e aplicar funções recursivas.

Definimos dois elementos básicos:

1. Lista vazia ($\text{nil}$):

 $$ \text{nil} = \lambda c. \lambda n. n $$

 Esta função ignora o primeiro argumento e retorna o segundo.

2. Construtor de lista ($\text{cons}$):

 $$ \text{cons} = \lambda h. \lambda t. \lambda c. \lambda n. c \, h \, (t \, c \, n) $$

O construtor recebe um elemento $h$ e uma lista $t$. Ele cria uma nova lista com $h$ na frente de $t$.

Com $\text{nil}$ e $\text{cons}$, podemos criar e manipular listas. Por exemplo, a lista $[1, 2, 3]$ é representada como:

 $$ \text{cons} \, 1 \, (\text{cons} \, 2 \, (\text{cons} \, 3 \, \text{nil})) $$

Chegamos a essa representação da seguinte forma:

 **Começamos com a lista vazia**:

 $$ \text{nil} = \lambda c. \lambda n. n $$

 **Adicionamos o elemento 3**:

 $$ \text{cons} \, 3 \, \text{nil} = (\lambda h. \lambda t. \lambda c. \lambda n. c \, h \, (t \, c \, n)) \, 3 \, (\lambda c. \lambda n. n) $$

 Após a redução $\beta$, temos:

 $$ \lambda c. \lambda n. c \, 3 \, ((\lambda c. \lambda n. n) \, c \, n) $$

 **Adicionamos o elemento 2**:

 $$ \text{cons} \, 2 \, (\text{cons} \, 3 \, \text{nil}) = (\lambda h. \lambda t. \lambda c. \lambda n. c \, h \, (t \, c \, n)) \, 2 \, (\lambda c. \lambda n. c \, 3 \, ((\lambda c. \lambda n. n) \, c \, n)) $$

 Após a redução $\beta$, obtemos:

 $$ \lambda c. \lambda n. c \, 2 \, ((\lambda c. \lambda n. c \, 3 \, ((\lambda c. \lambda n. n) \, c \, n)) \, c \, n) $$

 **Finalmente, adicionamos o elemento 1**:

 $$ \text{cons} \, 1 \, (\text{cons} \, 2 \, (\text{cons} \, 3 \, \text{nil})) = (\lambda h. \lambda t. \lambda c. \lambda n. c \, h \, (t \, c \, n)) \, 1 \, (\lambda c. \lambda n. c \, 2 \, ((\lambda c. \lambda n. c \, 3 \, ((\lambda c. \lambda n. n) \, c \, n)) \, c \, n)) $$

 Após a redução $\beta$, a representação final é:

 $$ \lambda c. \lambda n. c \, 1 \, ((\lambda c. \lambda n. c \, 2 \, ((\lambda c. \lambda n. c \, 3 \, ((\lambda c. \lambda n. n) \, c \, n)) \, c \, n)) \, c \, n) $$

Esta é a representação completa da lista $[1, 2, 3]$ em cálculo lambda puro. Esta representação permite operações recursivas sobre listas, como mapear funções ou calcular comprimentos. Podemos, por exemplo, definir uma função para calcular o comprimento de listas

### Função Comprimento (Length)

Vamos definir uma função para calcular o comprimento de uma lista usando o combinador $Y$:

$$\text{length} = Y \, (\lambda f. \lambda l. l \, (\lambda h. \lambda t. \text{succ} \, (f \, t)) \, 0)$$

Aqui, $\text{succ}$ é a função que retorna o sucessor de um número, e o corpo da função aplica-se recursivamente até que a lista seja esvaziada.

**Exemplo 1**: Vamos calcular o Comprimento da Lista $[1, 2, 3]$ em Cálculo Lambda Puro, usando $\text{length}$:

 **Definição de $\text{length}$**:

 $$ \text{length} = Y \, (\lambda f. \lambda l. l \, (\lambda h. \lambda t. \text{succ} \, (f \, t)) \, 0) $$

 **Representação da lista $[1, 2, 3]$**:

 $$ [1, 2, 3] = \lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n)) $$

 **Aplicamos $\text{length}$ à lista**:

 $$ \text{length} \, [1, 2, 3] = Y \, (\lambda f. \lambda l. l \, (\lambda h. \lambda t. \text{succ} \, (f \, t)) \, 0) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 O combinador $Y$ permite a recursão. Após aplicá-lo, obtemos:

 $$ (\lambda l. l \, (\lambda h. \lambda t. \text{succ} \, (\text{length} \, t)) \, 0) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 **Aplicamos a função à lista**:

 $$ (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) \, (\lambda h. \lambda t. \text{succ} \, (\text{length} \, t)) \, 0 $$

 Reduzimos, aplicando $c$ e $n$:

 $$ (\lambda h. \lambda t. \text{succ} \, (\text{length} \, t)) \, 1 \, ((\lambda h. \lambda t. \text{succ} \, (\text{length} \, t)) \, 2 \, ((\lambda h. \lambda t. \text{succ} \, (\text{length} \, t)) \, 3 \, 0)) $$

 Reduzimos cada aplicação de $(\lambda h. \lambda t. \text{succ} \, (\text{length} \, t))$:

 $$ \text{succ} \, (\text{succ} \, (\text{succ} \, (\text{length} \, \text{nil}))) $$

 Sabemos que $\text{length} \, \text{nil} = 0$, então:

 $$ \text{succ} \, (\text{succ} \, (\text{succ} \, 0)) $$

 Cada $\text{succ}$ incrementa o número por 1, então o resultado final é 3.

 Portanto, $\text{length} \, [1, 2, 3] = 3$ em cálculo lambda puro.

#### Função Soma (Sum) dos Elementos de uma Lista

Da mesma forma que fizemos antes, podemos definir uma função para somar os elementos de uma lista:

$$\text{sum} = Y \, (\lambda f. \lambda l. l \, (\lambda h. \lambda t. \text{add} \, h \, (f \, t)) \, 0)$$

Essa função percorre a lista somando os elementos, aplicando recursão via o combinador $Y$ até que a lista seja consumida.

**Exemplo 1**: Vamos aplicar esta função à lista $[1, 2, 3]$:

 **Representação da lista $[1, 2, 3]$**:

 $$ [1, 2, 3] = \lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n)) $$

 **Aplicamos $\text{sum}$ à lista**:

 $$ \text{sum} \, [1, 2, 3] = Y \, (\lambda f. \lambda l. l \, (\lambda h. \lambda t. \text{add} \, h \, (f \, t)) \, 0) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 O combinador $Y$ permite a recursão. Após aplicá-lo, obtemos:

 $$ (\lambda l. l \, (\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t)) \, 0) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 **Aplicamos a função à lista**:

 $$ (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) \, (\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t)) \, 0 $$

 **Reduzimos, aplicando $c$ e $n$**:

 $$ (\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t)) \, 1 \, ((\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t)) \, 2 \, ((\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t)) \, 3 \, 0)) $$

 Reduzimos cada aplicação de $(\lambda h. \lambda t. \text{add} \, h \, (\text{sum} \, t))$:

 $$ \text{add} \, 1 \, (\text{add} \, 2 \, (\text{add} \, 3 \, (\text{sum} \, \text{nil}))) $$

 Sabemos que $\text{sum} \, \text{nil} = 0$, então:

 $$ \text{add} \, 1 \, (\text{add} \, 2 \, (\text{add} \, 3 \, 0)) $$

 **Realizamos as adições de dentro para fora**:

 $$ \text{add} \, 1 \, (\text{add} \, 2 \, 3) $$
 $$ \text{add} \, 1 \, 5 $$
 $$ 6 $$

Portanto, $\text{sum} \, [1, 2, 3] = 6$ em cálculo lambda puro.

#### Funções Head e Tail em Cálculo Lambda Puro

As funções Head e Tail são funções uteis na manipulação de listas, principalmente em linguagens funcionais. Vamos definir estas funções em cálculo lambda puro. Começando pela função $\text{head}$:

$$ \text{head} = \lambda l. \, l \, (\lambda h. \lambda t. \, h) \, (\lambda x. \, x) $$

Em seguida, temos a função $\text{tail}$:

$$ \text{tail} = \lambda l. l \, (\lambda h. \lambda t. t) \, (\lambda x. \, x) $$

**Exemplo 1**: Aplicação à lista [1, 2, 3]

 **Aplicação de Head**: Aplicamos Head à lista:

 $$ \text{head} \, [1, 2, 3] = (\lambda l. l \, (\lambda h. \lambda t. h) \, (\lambda x. \, x)) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 Reduzimos:

 $$ (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) \, (\lambda h. \lambda t. h) \, (\lambda x. \, x) $$

 Aplicamos $c$ e $n$:

 $$ (\lambda h. \lambda t. h) \, 1 \, ((\lambda h. \lambda t. h) \, 2 \, ((\lambda h. \lambda t. h) \, 3 \, (\lambda x. \, x))) $$

 Reduzimos:

 $$ 1 $$

Portanto, $\text{head} \, [1, 2, 3] = 1$.

 **Aplicação de Tail**:

 Aplicamos Tail à lista:

 $$ \text{tail} \, [1, 2, 3] = (\lambda l. l \, (\lambda h. \lambda t. t) \, (\lambda x. \, x)) \, (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) $$

 Reduzimos:

 $$ (\lambda c. \lambda n. c \, 1 \, (c \, 2 \, (c \, 3 \, n))) \, (\lambda h. \lambda t. t) \, (\lambda x. \, x) $$

 Aplicamos $c$ e $n$:

 $$ (\lambda h. \lambda t. t) \, 1 \, ((\lambda h. \lambda t. t) \, 2 \, ((\lambda h. \lambda t. t) \, 3 \, (\lambda x. \, x))) $$

 Reduzimos:

 $$ \lambda c. \lambda n. c \, 2 \, (c \, 3 \, n) $$

Portanto, $\text{tail} \, [1, 2, 3] = [2, 3]$.

Listas são um tipo de dado composto útil para a maior parte das linguagens de programação. Não poderíamos deixar de definir listas em cálculo lambda puro para exemplificar a possibilidade da criação de algoritmos em cálculo lambda. Outra estrutura indispensável são as tuplas.

### Tuplas em Cálculo Lambda Puro

Definimos uma tupla de dois elementos como:

$$ (x, y) = \lambda f. f \, x \, y $$

A tupla $(3,4)$ será representada assim:

$$ (3, 4) = \lambda f. f \, 3 \, 4$$

Para que uma tupla seja realmente útil, precisamos ser capazes de trabalhar com seus elementos individualmente. Para isso, podemos definir duas funções: $\text{first}$ e $\text{follow}$.

#### Função First

A função First retorna o primeiro elemento da tupla:

$$ \text{first} = \lambda p. p \, (\lambda x. \lambda y. \, x) $$

 **Exemplo**: Aplicação a $(3,4)$:

 $$\text{first} \, (3, 4) = (\lambda p. p \, (\lambda x. \lambda y. \, x)) \, (\lambda f. f \, 3 \, 4) $$

 Redução:

 $$ (\lambda f. f \, 3 \, 4) \, (\lambda x. \lambda y. \, x) $$

 $$ (\lambda x. \lambda y. \, x) \, 3 \, 4 $$

 $$ 3 $$

#### Função Last

A função Last retorna o último elemento da tupla:

$$\text{last} = \lambda p. p \, (\lambda x. \lambda y. y)$$

 **Exemplo 2**: Aplicação a $(3,4)$:

 $$\text{last} \, (3, 4) = (\lambda p. p \, (\lambda x. \lambda y. y)) \, (\lambda f. f \, 3 \, 4)$$

 Redução:

 $$ (\lambda f. f \, 3 \, 4) \, (\lambda x. \lambda y. y) $$

 $$ (\lambda x. \lambda y. y) \, 3 \, 4 $$

 $$ 4 $$

## Cálculo Lambda e Haskell

Haskell implementa diretamente muitos conceitos do cálculo lambda. Vejamos alguns exemplos:

1. Funções Lambda: em Haskell, funções lambda são criadas usando a sintaxe \x -> ..., que é análoga à notação $\lambda x.$ do cálculo lambda.

   ```haskell
   -- Cálculo lambda: λx. \, x
   identidade = \x -> x
   -- Cálculo lambda: λx.λy.x
   constante = \x -> \y -> x
   -- Uso:
   main = do
   print (identidade 5) -- Saída: 5
   print (constante 3 4) -- Saída: 3
   ```

2. Aplicação de Função: a aplicação de função em Haskell é semelhante ao cálculo lambda, usando justaposição:

   ```haskell
   -- Cálculo lambda: (λx. \, x+1) 5
   incrementar = (\x -> x + 1) 5
   main = print incrementar -- Saída: 6
   ```

3. Currying: Haskell usa currying por padrão, permitindo aplicação parcial de funções:

   ```haskell
   -- Função de dois argumentos
   soma :: Int -> Int -> Int
   soma x y = x + y
   -- Aplicação parcial
   incrementar :: Int -> Int
   incrementar = soma 1

   main = do
   print (soma 2 3) -- Saída: 5
   print (incrementar 4) -- Saída: 5
   ```

4. Funções de Ordem Superior: Haskell suporta funções de ordem superior, um conceito fundamental do cálculo lambda:

   ```haskell
   -- map é uma função de ordem superior
   dobrarLista :: [Int] -> [Int]
   dobrarLista = map (\x -> 2 * x)

   main = print (dobrarLista [1,2,3]) -- Saída: [2,4,6]
   ```

5. Codificação de Dados: no cálculo lambda puro, não existem tipos de dados primitivos além de funções. Haskell, sendo uma linguagem prática, fornece tipos de dados primitivos, mas ainda permite codificações similares às do cálculo lambda.

6. Booleanos: no cálculo lambda, os booleanos podem ser codificados como:

   $$
   \begin{aligned}
   \text{true} &= \lambda x.\lambda y.x \\
   \text{false} &= \lambda x.\lambda y. \, y\\
   \end{aligned}
   $$

   Em Haskell, podemos implementar isso como:

   ```haskell
   true :: a -> a -> a
   true = \x -> \y -> x

   false :: a -> a -> a
   false = \x -> \y -> y

   -- Função if-then-else
   if' :: (a -> a -> a) -> a -> a -> a
   if' b t e = b t e

   main = do
   print (if' true "verdadeiro" "falso") -- Saída: "verdadeiro"
   print (if' false "verdadeiro" "falso") -- Saída: "falso"
   ```

7. Números Naturais: os números naturais podem ser representados usando a codificação de Church:

   $$
   \begin{aligned}
   0 &= \lambda f.\lambda x. \, x \\
   1 &= \lambda f.\lambda x.f \, x \\
   2 &= \lambda f.\lambda x.f (f \, x) \\
   3 &= \lambda f.\lambda x.f (f (f \, x))
   \end{aligned}
   $$

   Em Haskell, teremos:

   ```haskell
   type Church a = (a -> a) -> a -> a

   zero :: Church a
   zero = \f -> \x -> x

   succ' :: Church a -> Church a
   succ' n = \f -> \x -> f (n f \, x)

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

O cálculo lambda é a base teórica para muitos conceitos da programação funcional, especialmente em Haskell. Mas, para isso, precisamos considerar os tipos.

# Cálculo Lambda Tipado

Geralmente não percebemos que, na matemática, uma a definição de uma função inclui a determinação dos tipos de dados que ela recebe e dos tipos de dados que ela devolve. Por exemplo, a função de quadrado aceita números inteiros $n$ como entradas e produz números inteiros $n^2$ como saídas. Uma função de teste de zero $isZero$ aceitará números inteiros e produzirá valores booleanos como resposta. Fazemos isso, quase instintivamente, ou explicitamente, definindo os domínio. Podemos estender este conceito ao cálculo lambda.

O cálculo lambda não tipado é poderoso. Ele expressa todas as funções computáveis. Mas tem limites. Algumas expressões no cálculo lambda não tipado levam a paradoxos. O termo $\omega = \lambda x. \, x \, x$ aplicado a si mesmo resulta em redução infinita:

$$(\lambda x. \, x \, x) (\lambda x. \, x \, x) \to (\lambda x. \, x \, x) (\lambda x. \, x \, x) \to ...$$

A existência deste _loop_ infinito é um problema. Torna o sistema inconsistente. Podemos resolver esta inconsistência adicionando tipos aos termos. Os tipos restringem como os termos se combinam evitando paradoxos e laços infinitos. Uma vez que os tipos tenham sido acrescentados, teremos o cálculo lambda tipado.

No cálculo lambda tipado, cada termo terá um tipo e funções terão tipos no formato $A \to B$, Significando que recebem uma entrada do tipo $A$ e retornam uma saída do tipo $B$.

O termo $\omega$ não é válido no cálculo lambda tipado. O sistema de tipos o rejeita. tornando o sistema consistente, garantindo que as operações terminem, evitando a recursão infinita. Desta forma, o cálculo lambda tipado se torna a base para linguagens de programação tipadas, garantindo que os programas sejam bem comportados e terminem.

A adoção de tipos define quais dados são permitidos como argumentos e quais os tipos de resultados uma função pode gerar. Essas restrições evitam a aplicação indevida de funções a si mesmas e o uso de expressões malformadas, garantindo consistência e prevenindo paradoxos.

No sistema de tipos simples, variáveis têm tipos atribuídos, no formato $x\, :\, A$, onde $A$ é o tipo de $x$. As funções são descritas por sua capacidade de aceitar um argumento de um tipo e retornar um valor de outro tipo. Uma função que aceita uma entrada do tipo $A$ e retorna um valor do tipo $B$ é escrita como $A \rightarrow B$. Permitindo que funções recebam argumentos de um tipo e retornem outro tipo de dado de acordo com a necessidade.

Podemos simplificar o conceito de tipos a dois conceitos:

- **Tipos básicos**, como $\text{Bool}$ (booleanos) ou $\text{Nat}$ (números naturais).

- **Tipos de função**, como $A \rightarrow B$, que representam funções que mapeiam valores de $A$ para $B$.

Considere a expressão $ \lambda x. \, x + 1$. No cálculo lambda tipado, essa função será válida se $x$ for de um tipo numérico, como $x : \text{Nat}$, neste caso considerando $1$ com um literal natural. Sendo assim, a função seria tipada e sua assinatura a definirá como uma função que aceita um número natural e retorna um número natural:

$$\lambda x : \text{Nat}. \, x + 1 : \text{Nat} \rightarrow \text{Nat}$$

Isso assegura que apenas valores do tipo $\text{Nat}$ possam ser aplicados a essa função, evitando a aplicação incorreta de argumentos não numéricos.

Com um pouco mais de formalidade, vamos considerar um conjunto de tipos básicos. Usaremos a letra grega $\tau$ ("tau") minúscula para indicar um tipo básico. O conjunto de tipos simples será definido pela seguinte gramática BNF:

 $$A,B ::= \tau \mid A \rightarrow B \mid A \times B \mid 1$$

O significado pretendido desses tipos é o seguinte: tipos base são estruturas simples como os  tipos de inteiro e booleano. O tipo $A \rightarrow B$ é o tipo de funções de $A$ para $B$. O tipo $A \times B$ é o tipo de tuplas $\langle x, y \rangle$, onde $x$ tem tipo $A$ e $y$ tem tipo $B$. A notação $\langle x, y \rangle$ foi introduzida para representar um par de termos $M$ e $N$. Permitindo que o cálculo lambda tipado manipule não apenas funções, mas também estruturas de dados compostas.

O tipo $1$ é um tipo de um elemento literal, um tipo especial que contém exatamente um elemento, semelhante ao conceito de _tipo simples_ em algumas linguagens de programação. Isso é útil para representar valores que não carregam informação significativa, mas que precisam existir para manter a consistência do sistema de tipos.

Vamos adotar uma regra de precedência: $\times$ tem precedência sobre $\rightarrow$, e $\rightarrow$ associa-se à direita. Assim, $A \times B \rightarrow C$ é $(A \times B) \rightarrow C$, e $A \rightarrow B \rightarrow C$ é $A \rightarrow (B \rightarrow C)$[^cita7].

O conjunto de termos lambda tipados puros e brutos será definido pela seguinte BNF:

Termos brutos: $M,N ::= x \mid M N \mid \lambda x^A.M \mid \langle M,N \rangle \mid \pi_1M \mid \pi_2M \mid *$

Onde:

- $x$ representa variáveis
- $M N$ representa aplicação de função
- $\lambda x^A.M$ representa abstração lambda com anotação de tipo
- $\langle M,N \rangle$ representa pares
- $\pi_1M$ e $\pi_2M$ representam projeções de pares
- $*$ representa o elemento único do tipo $1$

Aqui definimos a sintaxe básica dos termos no cálculo lambda tipado, antes de qualquer verificação de tipo ou análise semântica. Por isso, os chamamos de simples ou brutos.

Estes termos são chamados de "brutos" porque  representam a estrutura sintática pura dos termos, sem garantia de que sejam bem tipados. Esta sintaxe pode incluir termos que não são válidos no sistema de tipos, mas que seguem a gramática básica. Usaremos esta gramática como o ponto de partida para o processo de verificação de tipos e análise semântica. Estes termos brutos serão posteriormente submetidos a regras de tipagem para determinar se são bem formados no sistema de tipos do cálculo lambda tipado.

Diferentemente do que fizemos no cálculo lambda não tipado, adicionamos aqui uma sintaxe especial para pares. Especificamente, $\langle M,N \rangle$ é um par de termos, $\pi_iM$ é uma projeção, com a intenção de que $\pi_i\langle M_1,M_2 \rangle = M_i$, usadas para extrair os componentes de um par. Especificamente, a intenção é que $\pi_1\langle M_1,M_2 \rangle$ resulte $M_1$ e $\pi_2\langle M_1,M_2 \rangle$ resulte em $M_2$. Criando uma regra que permite o acesso aos elementos individuais de um par.

Além disso, adicionamos um termo $*$, que é o único elemento do tipo $1$. Outra mudança em relação ao cálculo lambda não tipado é que agora escrevemos $\lambda x^A.M$ para uma abstração lambda para indicar que $x$ tem tipo $A$. No entanto, às vezes omitiremos os sobrescritos e escreveremos $\lambda x.M$ como antes.

Esta gramática permite que as abstrações lambda incluam anotações de tipo na forma $λ x:\tau. M$, indicando explicitamente que a variável $x$ tem o tipo $\tau$. Isso permite que o sistema verifique se as aplicações de função são feitas corretamente e se os termos são bem tipados.

Embora as anotações de tipo sejam importantes, às vezes os tipos podem ser omitidos, escrevendo-se simplesmente $λ x. M$. Isso ocorre quando o tipo de $x$ é claro a partir do contexto ou quando não há ambiguidade, facilitando a leitura e a escrita das expressões.

Em resumo as sintáticas permitem que o cálculo lambda tipado:

- **Represente Estruturas de Dados Complexas**: Com a capacidade de manipular pares e projeções, é possível representar dados mais complexos além de funções puras, aproximando o cálculo lambda das necessidades práticas de linguagens de programação.

- **Garanta a Segurança de Tipos**: As anotações de tipo em variáveis e a sintaxe enriquecida ajudam a prevenir erros, como a aplicação indevida de funções ou a formação de expressões paradoxais, assegurando que apenas termos bem tipados sejam considerados válidos.

As noções de variáveis livres e ligadas e $\alpha$-conversão são definidas como no cálculo lambda não tipado; novamente identificamos termos $\alpha$-equivalentes.

## Regras de Tipagem

As regras de tipagem no cálculo lambda tipado fornecem um sistema formal para garantir que as expressões sejam bem formadas. As principais regras são:

1. **Regra da Variável**: Se uma variável $x$ possui o tipo $\tau$ no contexto $\Gamma$, então podemos derivar que $\Gamma \vdash x : A$.

   O contexto de tipagem, denotado por `$ \Gamma $`, é um conjunto de associações entre variáveis e seus tipos. Formalmente temos:

   $$
   \frac{}{\, \Gamma \vdash x : A} \quad \text{se } (x : \tau) \in \Gamma
   $$

   Isso significa que, se a variável $x$ tem o tipo $\Tau$ no contexto $\Gamma$, então podemos derivar que $\Gamma \vdash x : \tau$. Em outras palavras, uma variável é bem tipada se seu tipo está definido em determinado contexto.

   - **Contexto de Tipagem ($\Gamma$)**: É um conjunto de pares $(x : \tau)$ que associa as variáveis aos seus respectivos tipos. Por exemplo, $\Gamma = \{ x : \text{Int},\ y : \text{Bool} \}$.

   - **Julgamento de Tipagem (`$\Gamma \vdash x : \tau$)**: Lê-se "sob o contexto $\Gamma$, a variável $x$ tem tipo $\tau$".

   Considere o contexto:

   $$
   \Gamma = \{ x : \text{Nat},\ y : \text{Bool} \}
   $$

   Aplicando a Regra da Variável: Como $(x : \text{Nat}) \in \Gamma$`, podemos afirmar que:

   $$
   \Gamma \vdash x : \text{Nat}
   $$

   Similarmente, como `$(y : \text{Bool}) \in \Gamma$:

   $$
   \Gamma \vdash y : \text{Bool}
   $$

   Isso mostra que, dentro do contexto $\Gamma$, as variáveis $x$ e $y$ têm os tipos $\text{Nat}$ e $\text{Bool}$, respectivamente.

   A Regra da Variável fundamenta a tipagem sendo a base para atribuição de tipos a expressões mais complexas. Sem essa regra, não seria possível inferir os tipos das variáveis em expressões. Além disso, essa regra garante a consistência do sistema  asseguramos que as variáveis são usadas de acordo com seus tipos declarados, evitamos erros de tipagem e comportamentos imprevistos.

2. **Regra de Abstração**: Se sob o contexto $\Gamma$, temos que $\Gamma, x:\tau \vdash M:B$, então podemos derivar que $\Gamma \vdash (\lambda x:A.M) : A \rightarrow B$.

   A **Regra de Abstração** no cálculo lambda tipado permite derivar o tipo de uma função lambda baseada no tipo de seu corpo e no tipo de seu parâmetro. Formalmente, a regra é expressa como:

   $$
   \frac{\Gamma,\, x:\tau \, \vdash\ M:B}{\, \Gamma\ \vdash\ (\lambda x:\tau.\ M) : A \rightarrow B}
   $$

   Isso significa que, se no contexto $\Gamma$, ao adicionar a associação $x:\tau$, podemos derivar que $M$ tem tipo $B$, então podemos concluir que a abstração lambda $(\lambda x:\tau.\, M)$ tem tipo $A \rightarrow B$ no contexto original $\Gamma$. Novamente temos o contexto de tipagem $\Gamma$ indicando Conjunto de associações entre variáveis e seus tipos. O julgamento será feito por _sob o contexto $\Gamma$, a expressão $M$ tem tipo $B$_. Finalmente, existe uma adição ao contexto definida ao considerar a variável $x$ com tipo $\tau$, expandimos o contexto para $\Gamma,\, x:\tau$.

   A Regra de Abstração define Tipos de Funções: Permite derivar o tipo de uma função lambda a partir dos tipos de seu parâmetro e de seu corpo, enquanto assegura Coerência por garantir que a função está bem tipada e que pode ser aplicada a argumentos do tipo correto.

3. **Regra de Aplicação**: Se $\Gamma \vdash M : \tau \rightarrow B$ e $\Gamma \vdash N : \tau$, então podemos derivar que $\Gamma \vdash (M \, N) : B$.

   A **Regra de Aplicação** no cálculo lambda tipado permite determinar o tipo de uma aplicação de função com base nos tipos da função e do argumento. Formalmente, a regra é expressa como:

   $$
   \frac{\Gamma\ \vdash\ M : \tau \rightarrow B \quad \Gamma\ \vdash\ N : \tau}{\, \Gamma\ \vdash\ (M\ N) : B}
   $$

   Isso significa que, se no contexto $\Gamma$ podemos derivar que $M$ tem tipo $A \rightarrow B$ e que $N$ tem tipo $A$, então podemos concluir que a aplicação $(M\ N)$ tem tipo $B$ no contexto $\Gamma$.

   Analisando temos, novamente, o contexto de tipagem $\Gamma$), 0 julgamentos de Tipagem $\Gamma\ \vdash\ M : \tau \rightarrow B$: A expressão $M$ é uma função que leva um argumento do tipo $\tau$ e retorna um resultado do tipo $B$. Finalmente $\Gamma\ \vdash\ N : \tau$: A expressão $N$ é um argumento do tipo $\tau$.
   Ou seja, $\Gamma\ \vdash\ (M\ N) : B$: A aplicação da função $M$ ao argumento $N$ resulta em um termo do tipo $B$.

   Esta regra Permite Compor funções e argumentos determinando como funções tipadas podem ser aplicadas a argumentos tipados para produzir resultados tipados. Também assegura que as funções são aplicadas a argumentos do tipo correto, evitando erros de tipagem. Esta regra estabelece que, se temos uma função que espera um argumento de um certo tipo e temos um argumento desse tipo, então a aplicação da função ao argumento é bem tipada e seu tipo é o tipo de retorno da função. Isso é fundamental para a construção de programas bem tipados no cálculo lambda tipado, garantindo a segurança e a coerência do sistema de tipos.

Essas regras fornecem a base para a derivação de tipos em expressões complexas no cálculo lambda tipado, garantindo que cada parte da expressão esteja correta e que a aplicação de funções seja válida.

### Exemplos das regras de tipagem

**Exemplo 1**: Regra da Variável

   Considere o contexto:

   $$
   \Gamma = \{ x : \text{Nat},\ y : \text{Bool} \}
   $$

   Aplicando a Regra da Variável teremos:

   Como $(x : \text{Nat}) \in \Gamma$, então:

   $$
   \Gamma \vdash x : \text{Nat}
   $$

   Como $(y : \text{Bool}) \in \Gamma$, então:

   $$
   \Gamma \vdash y : \text{Bool}
   $$

**Exemplo**: Regra de Abstração

   Considere a função:

   $$
   \lambda x:\text{Nat}.\ x + 1
   $$

   Aplicação da regra:

   No contexto $\Gamma$ estendido com $x:\text{Nat}$:

   $$
   \Gamma,\, x:\text{Nat} \vdash x + 1 : \text{Nat}
   $$

   Aplicando a Regra de Abstração:

   $$
   \frac{\Gamma,\, x:\text{Nat} \vdash x + 1 : \text{Nat}}{\, \Gamma \vdash (\lambda x:\text{Nat}.\ x + 1) : \text{Nat} \rightarrow \text{Nat}}
   $$

**Exemplo**: Regra de Aplicação

   Considere $M = \lambda x:\text{Nat}.\ x + 1$ e $N = 5$.

   Tipagem da função $M$:

   $$
   \Gamma \vdash M : \text{Nat} \rightarrow \text{Nat}
   $$

   Tipagem do argumento $N$:

   $$
   \Gamma \vdash 5 : \text{Nat}
   $$

   Aplicando a Regra de Aplicação:

   $$
   \frac{\Gamma\ \vdash\ M : \text{Nat} \rightarrow \text{Nat} \quad \Gamma\ \vdash\ 5 : \text{Nat}}{\, \Gamma\ \vdash\ M\, 5 : \text{Nat}}
   $$

### Exercícios Regras de Tipagem no Cálculo Lambda

**1**. Dado o contexto:

   $$
   \Gamma = \{ z : \text{Bool} \}
   $$

   Use a **Regra da Variável** para derivar o tipo de $z$ no contexto $\Gamma$.

   **Solução**: dela **Regra da Variável**:

   $$
   \frac{z : \text{Bool} \in \Gamma}{\Gamma \vdash z : \text{Bool}}
   $$

   Portanto, no contexto $\Gamma$, $z$ tem tipo $\text{Bool}$.

**2**. Considere a função:

   $$
   \lambda y:\text{Nat}.\ y \times 2
   $$

   Usando a **Regra de Abstração**, mostre que esta função tem o tipo $\text{Nat} \rightarrow \text{Nat}$.

   **Solução**: sSabemos que $y \times 2$ é uma operação que, dado $y$ de tipo $\text{Nat}$, retorna um $\text{Nat}$.

   Aplicando a **Regra de Abstração**:

   $$
   \frac{\Gamma, y:\text{Nat} \vdash y \times 2 : \text{Nat}}{\Gamma \vdash \lambda y:\text{Nat}.\ y \times 2 : \text{Nat} \rightarrow \text{Nat}}
   $$

   Portanto, a função tem tipo $\text{Nat} \rightarrow \text{Nat}$.

**3**. No contexto vazio $\Gamma = \{\}$, determine se a seguinte aplicação é bem tipada usando a **Regra de Aplicação**:

   $$
   (\lambda x:\text{Bool}.\ x)\ \text{true}
   $$

   **Solução**: aplicando a **Regra de Aplicação**:

   1. $\Gamma \vdash \lambda x:\text{Bool}.\ x : \text{Bool} \rightarrow \text{Bool}$.

   2. $\Gamma \vdash \text{true} : \text{Bool}$.

   3. Como os tipos correspondem, podemos concluir:

   $$
   \frac{\Gamma \vdash \lambda x:\text{Bool}.\ x : \text{Bool} \rightarrow \text{Bool} \quad \Gamma \vdash \text{true} : \text{Bool}}{\Gamma \vdash (\lambda x:\text{Bool}.\ x)\ \text{true} : \text{Bool}}
   $$

   A aplicação é bem tipada e tem tipo $\text{Bool}$.

**4**. Dado o contexto:

   $$
   \Gamma = \{ f : \text{Nat} \rightarrow \text{Nat},\ n : \text{Nat} \}
   $$

   Use a **Regra de Aplicação** para mostrar que $f\ n$ tem tipo $\text{Nat}$.

   **Solução**: aplicando a **Regra de Aplicação**:

   1. Do contexto, $\Gamma \vdash f : \text{Nat} \rightarrow \text{Nat}$.

   2. Do contexto, $\Gamma \vdash n : \text{Nat}$.

   3. Portanto:

   $$
   \frac{\Gamma \vdash f : \text{Nat} \rightarrow \text{Nat} \quad \Gamma \vdash n : \text{Nat}}{\Gamma \vdash f\ n : \text{Nat}}
   $$

   Assim, $f\ n$ tem tipo $\text{Nat}$.

**5**. Usando as regras de tipagem, determine o tipo da expressão:

   $$
   \lambda f:\text{Nat} \rightarrow \text{Bool}.\ \lambda n:\text{Nat}.\ f\ n
   $$

**Solução**: Queremos encontrar o tipo da função $\lambda f:\text{Nat} \rightarrow \text{Bool}.\ \lambda n:\text{Nat}.\ f\ n$.

   1. No contexto $\Gamma$, adicionamos $f:\text{Nat} \rightarrow \text{Bool}$.

   2. Dentro da função, adicionamos $n:\text{Nat}$.

   3. Sabemos que $\Gamma, f:\text{Nat} \rightarrow \text{Bool}, n:\text{Nat} \vdash f\ n : \text{Bool}$.

   4. Aplicando a **Regra de Abstração** para $n$:

   $$
   \frac{\Gamma, f:\text{Nat} \rightarrow \text{Bool}, n:\text{Nat} \vdash f\ n : \text{Bool}}{\Gamma, f:\text{Nat} \rightarrow \text{Bool} \vdash \lambda n:\text{Nat}.\ f\ n : \text{Nat} \rightarrow \text{Bool}}
   $$

   5. Aplicando a **Regra de Abstração** para $f$:

   $$
   \frac{\Gamma \vdash \lambda n:\text{Nat}.\ f\ n : \text{Nat} \rightarrow \text{Bool}}{\Gamma \vdash \lambda f:\text{Nat} \rightarrow \text{Bool}.\ \lambda n:\text{Nat}.\ f\ n : (\text{Nat} \rightarrow \text{Bool}) \rightarrow (\text{Nat} \rightarrow \text{Bool})}
   $$

   Portanto, o tipo da expressão é $(\text{Nat} \rightarrow \text{Bool}) \rightarrow (\text{Nat} \rightarrow \text{Bool})$.

**6**. No contexto:

   $$
   \Gamma = \{ x : \text{Nat} \times \text{Bool} \}
   $$

   Utilize a **Regra da Variável** para derivar o tipo de $x$ em $\Gamma$.

   **Solução**: pela **Regra da Variável**:

   $$
   \frac{x : \text{Nat} \times \text{Bool} \in \Gamma}{\Gamma \vdash x : \text{Nat} \times \text{Bool}}
   $$

   Portanto, $x$ tem tipo $\text{Nat} \times \text{Bool}$ no contexto $\Gamma$.

**7**. Mostre, usando a **Regra de Abstração**, que a função:

   $$
   \lambda p:\text{Nat} \times \text{Bool}.\ \pi_1\ p
   $$

   Tem o tipo $(\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}$.

   **Solução**:

   1. No contexto $\Gamma$, adicionamos $p:\text{Nat} \times \text{Bool}$.

   2. A operação $\pi_1\ p$ extrai o primeiro componente do par, portanto:

      $$
      \Gamma, p:\text{Nat} \times \text{Bool} \vdash \pi_1\ p : \text{Nat}
      $$

   3. Aplicando a **Regra de Abstração**:

   $$
   \frac{\Gamma, p:\text{Nat} \times \text{Bool} \vdash \pi_1\ p : \text{Nat}}{\Gamma \vdash \lambda p:\text{Nat} \times \text{Bool}.\ \pi_1\ p : (\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}}
   $$

   Portanto, a função tem tipo $(\text{Nat} \times \text{Bool}) \rightarrow \text{Nat}$.

**8**. No contexto vazio, determine se a seguinte aplicação é bem tipada:

   $$
   (\lambda x:\text{Nat}.\ x + 1)\ \text{true}
   $$

   Explique qual regra de tipagem é violada se a aplicação não for bem tipada.

   **Solução**:

   1. Temos $\Gamma \vdash \lambda x:\text{Nat}.\ x + 1 : \text{Nat} \rightarrow \text{Nat}$.

   2. Também, $\Gamma \vdash \text{true} : \text{Bool}$.

   3. Pela **Regra de Aplicação**, para que a aplicação seja bem tipada, o tipo do argumento deve corresponder ao tipo esperado pela função:

   $$
   \frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M\ N : B}
   $$

   4. Aqui, $M$ espera um argumento do tipo $\text{Nat}$, mas $N$ é de tipo $\text{Bool}$.

   5. Como $\text{Nat} \neq \text{Bool}$, a aplicação não é bem tipada.

   A **Regra de Aplicação** é violada porque o tipo do argumento fornecido não corresponde ao tipo esperado pela função.

**9**. Dado:

   $$
   M = \lambda x:\text{Bool}.\ \lambda y:\text{Bool}.\ x \land y
   $$

   Determine o tipo de $M$ usando as regras de tipagem.

   **Solução**:

   1. No contexto $\Gamma$, adicionamos $x:\text{Bool}$.

   2. Dentro da função, adicionamos $y:\text{Bool}$.

   3. A expressão $x \land y$ tem tipo $\text{Bool}$.

   4. Aplicando a **Regra de Abstração** para $y$:

      $$
      \frac{\Gamma, x:\text{Bool}, y:\text{Bool} \vdash x \land y : \text{Bool}}{\Gamma, x:\text{Bool} \vdash \lambda y:\text{Bool}.\ x \land y : \text{Bool} \rightarrow \text{Bool}}
      $$

   5. Aplicando a **Regra de Abstração** para $x$:

   $$
   \frac{\Gamma \vdash \lambda y:\text{Bool}.\ x \land y : \text{Bool} \rightarrow \text{Bool}}{\Gamma \vdash \lambda x:\text{Bool}.\ \lambda y:\text{Bool}.\ x \land y : \text{Bool} \rightarrow (\text{Bool} \rightarrow \text{Bool})}
   $$

   Portanto, o tipo de $M$ é $\text{Bool} \rightarrow \text{Bool} \rightarrow \text{Bool}$.

**10**. Utilize as regras de tipagem para mostrar que a expressão:

   $$
   (\lambda f:\text{Nat} \rightarrow \text{Nat}.\ f\ (f\ 2))\ (\lambda x:\text{Nat}.\ x + 3)
   $$

   Tem tipo $\text{Nat}$.

   **Solução**:

   1. Primeiro, analisamos a função:

      $$
      \lambda f:\text{Nat} \rightarrow \text{Nat}.\ f\ (f\ 2)
      $$

      - Dentro desta função, $f : \text{Nat} \rightarrow \text{Nat}$.
      - Sabemos que $2 : \text{Nat}$.
      - Então $f\ 2 : \text{Nat}$.
      - Consequentemente, $f\ (f\ 2) : \text{Nat}$.

   2. Portanto, a função tem tipo:

      $$
      (\text{Nat} \rightarrow \text{Nat}) \rightarrow \text{Nat}
      $$

   3. Agora, consideramos o argumento:

      $$
      \lambda x:\text{Nat}.\ x + 3
      $$

      - Esta função tem tipo $\text{Nat} \rightarrow \text{Nat}$.

   4. Aplicando a **Regra de Aplicação**:

      $$
      \frac{\Gamma \vdash \lambda f:\text{Nat} \rightarrow \text{Nat}.\ f\ (f\ 2) : (\text{Nat} \rightarrow \text{Nat}) \rightarrow \text{Nat} \quad \Gamma \vdash \lambda x:\text{Nat}.\ x + 3 : \text{Nat} \rightarrow \text{Nat}}{\Gamma \vdash (\lambda f:\text{Nat} \rightarrow \text{Nat}.\ f\ (f\ 2))\ (\lambda x:\text{Nat}.\ x + 3) : \text{Nat}}
      $$

   Assim, a expressão completa tem tipo $\text{Nat}$.

## Propriedades do Cálculo Lambda Tipado

A partir das regras de tipagem, podemos definir um conjunto de propriedades da tipagem no cálculo lambda. Essas propriedades têm implicações tanto para a teoria quanto para a prática da programação. Vamos destacar:

1. **Normalização forte**: Todo termo bem tipado possui uma forma normal e qualquer sequência de reduções eventualmente termina. Isso garante que as reduções de expressões no cálculo lambda tipado sempre produzirão um resultado final, eliminando loops infinitos. É importante notar que a propriedade de normalização forte é garantida no cálculo lambda simplesmente tipado. Em sistemas de tipos mais avançados, como aqueles que suportam recursão ou tipos dependentes, a normalização forte pode não se aplicar, podendo existir termos bem tipados que não convergem para uma forma normal.

   Formalmente, se $\Gamma \vdash M : \tau$, então existe uma sequência finita de reduções $M \rightarrow_\beta M_1 \rightarrow_\beta ... \rightarrow_\beta M_n$ onde $M_n$ está em forma normal.

   **Exemplo**: considere o termo $(\lambda x:\text{Nat}. x + 1) 2$. Este termo é bem tipado e reduz para $3$ em um número finito de passos:

   $$(\lambda x:\text{Nat}. x + 1) 2 \rightarrow_\beta 2 + 1 \rightarrow 3$$

2. **Preservação de tipos** (_subject reduction_): se uma expressão $M$ possui o tipo $A$ sob o contexto $\Gamma$, e $M$ pode ser reduzido para $N$ pela regra $\beta$-redução ($M \rightarrow_\beta N$), então $N$ também possui o tipo $A$. Essa propriedade é essencial para garantir que as transformações de termos dentro do sistema de tipos mantenham a consistência tipológica.

   Formalmente: Se $\Gamma \vdash M : \tau$ e $M \rightarrow_\beta N$, então $\Gamma \vdash N : \tau$.

   **Exemplo**: se $\Gamma \vdash (\lambda x:\text{Nat}. x + 1) 2 : \text{Nat}$, então após a redução, teremos $\Gamma \vdash 3 : \text{Nat}$.

3. **Decidibilidade da tipagem**: um algoritmo pode decidir se uma expressão possui um tipo válido no sistema de tipos, o que é uma propriedade crucial para a análise de tipos em linguagens de programação.

   Isso significa que existe um procedimento efetivo que, dado um contexto $\Gamma$ e um termo $M$, pode determinar se existe um tipo $A$ tal que $\Gamma \vdash M : \tau$.

   **Exemplo**: um algoritmo de verificação de tipos pode determinar que:
   - $\lambda x:\text{Nat}. x + 1$ tem tipo $\text{Nat} \rightarrow \text{Nat}$
   - $(\lambda x:\text{Nat}. x) \text{true}$ não é bem tipado

4. **Progresso**: uma propriedade adicional importante é o progresso. Se um termo é bem tipado e não está em forma normal, então existe uma redução que pode ser aplicada a ele.

   Formalmente: Se $\Gamma \vdash M : \tau$ e $M$ não está em forma normal, então existe $N$ tal que $M \rightarrow_\beta N$.

   Esta propriedade, junto com a preservação de tipos, garante que termos bem tipados ou estão em forma normal ou podem continuar sendo reduzidos sem ficarem "presos" em um estado intermediário.

Estas propriedades juntas garantem a consistência e a robustez do sistema de tipos do cálculo lambda tipado, fornecendo uma base sólida para o desenvolvimento de linguagens de programação tipadas e sistemas de verificação formal.

### Exercícios de Propriedades do Cálculo Lambda Tipado

**1**. Considere o termo $(\lambda x:\text{Nat}. x + 1) 2$. Mostre a sequência de reduções que leva este termo à sua forma normal, ilustrando a propriedade de normalização forte.

   **Solução**:

   $$
   (\lambda x:\text{Nat}. x + 1) 2 \rightarrow_\beta 2 + 1 \rightarrow 3
   $$

   O termo reduz à sua forma normal, $3$, em um número finito de passos.

**2**. Dado o termo $(\lambda f:\text{Nat}\rightarrow\text{Nat}. \lambda x:\text{Nat}. f (f x)) (\lambda y:\text{Nat}. y + 1) 2$, mostre que ele é bem tipado e reduz para um valor do tipo $\text{Nat}$.

   **Solução**: 0 termo é bem tipado: $(\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat}\rightarrow\text{Nat}$

   Redução:

   $$
   \begin{aligned}
   &(\lambda f:\text{Nat}\rightarrow\text{Nat}. \lambda x:\text{Nat}. f (f x)) (\lambda y:\text{Nat}. y + 1) 2 \\
   &\rightarrow_\beta (\lambda x:\text{Nat}. (\lambda y:\text{Nat}. y + 1) ((\lambda y:\text{Nat}. y + 1) x)) 2 \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. y + 1) ((\lambda y:\text{Nat}. y + 1) 2) \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. y + 1) (2 + 1) \\
   &\rightarrow_\beta (\lambda y:\text{Nat}. y + 1) 3 \\
   &\rightarrow_\beta 3 + 1 \\
   &\rightarrow 4
   \end{aligned}
   $$

   O resultado final (4) é do tipo $\text{Nat}$, ilustrando a preservação de tipos.

**3**. Explique por que o termo $(\lambda x:\text{Bool}. x + 1)$ não é bem tipado. Como isso se relaciona com a propriedade de decidibilidade da tipagem?

   **Solução**: o termo não é bem tipado porque tenta adicionar 1 a um valor booleano, o que é uma operação inválida. A decidibilidade da tipagem permite que um algoritmo detecte este erro de tipo, rejeitando o termo como mal tipado.

**4**. Considere o termo:

   $$
   M = (\lambda x:\text{Nat}.\ \text{if } x = 0\ \text{then}\ 1\ \text{else}\ x \times ((\lambda y:\text{Nat}.\ y - 1)\ x))
   $$

   Este termo calcula $x$ multiplicado por $x - 1$. Mostre que ele satisfaz a propriedade de **preservação de tipos** para uma entrada específica.

   **Solução:** vamos aplicar o termo $M$ a $x = 3$ e verificar a preservação de tipos durante as reduções.

   1. **Tipagem Inicial:**

      - Tipo de $M$:

      $$
      \Gamma \vdash M : \text{Nat} \rightarrow \text{Nat}
      $$

      $M$ é uma função que recebe um $\text{Nat}$ e retorna um $\text{Nat}$.

      Tipo do Argumento $3$:

      $$
      \Gamma \vdash 3 : \text{Nat}
      $$

   2. **Aplicação da Função:**

      Aplicação:

      $$
      M\ 3 = (\lambda x:\text{Nat}.\ \text{if } x = 0\ \text{then}\ 1\ \text{else}\ x \times ((\lambda y:\text{Nat}.\ y - 1)\ x))\ 3
      $$

      Redução por $\beta$-redução:

      $$
      \rightarrow_\beta \text{if } 3 = 0\ \text{then}\ 1\ \text{else}\ 3 \times ((\lambda y:\text{Nat}.\ y - 1)\ 3)
      $$

   3. **Avaliação da Condicional:**

      Como $3 \neq 0$, seguimos para o ramo "else":

      $$
      \rightarrow 3 \times ((\lambda y:\text{Nat}.\ y - 1)\ 3)
      $$

   4. **Redução do Parêntese Interno:**

      Aplicação da Função Interna:

      $$
      (\lambda y:\text{Nat}.\ y - 1)\ 3 \rightarrow_\beta 3 - 1 = 2
      $$

      Atualização da Expressão:

      $$
      \rightarrow 3 \times 2
      $$

   5. **Cálculo Final:**

      Multiplicação:

      $$
      3 \times 2 = 6
      $$

      Tipo do Resultado:

      $$
      \Gamma \vdash 6 : \text{Nat}
      $$

   **Demonstrando a Preservação de Tipos:**

   Antes da Redução: a função $M$ tem tipo $\text{Nat} \rightarrow \text{Nat}$; o argumento $3$ tem tipo $\text{Nat}$. Portanto, pela **Regra de Aplicação**:

   $$
   \frac{\Gamma\ \vdash\ M : \text{Nat} \rightarrow \text{Nat} \quad \Gamma\ \vdash\ 3 : \text{Nat}}{\, \Gamma\ \vdash\ M\ 3 : \text{Nat}}
   $$

   Durante as Reduções, cada passo manteve o tipo $\text{Nat}$:

- $\Gamma \vdash 3 \times 2 : \text{Nat}$
  
- $\Gamma \vdash 6 : \text{Nat}$

  O termo inicial $M\ 3$ tem tipo $\text{Nat}$. Após as reduções, o resultado $6$ também tem tipo $\text{Nat}$. Portanto, a propriedade de preservação de tipos é satisfeita.

   **Observação:** uso das regras de tipagem.

  1. **Regra da Variável**: definimos os tipos das variáveis $x$ e $y$ como $\text{Nat}$.

  2. **Regra de Abstração**: As funções anônimas $\lambda x:\text{Nat}.\ \dots$ e $\lambda y:\text{Nat}.\ y - 1$ são tipadas como $\text{Nat} \rightarrow \text{Nat}$.

  3. **Regra de Aplicação**: Aplicamos as funções aos argumentos correspondentes, mantendo a consistência de tipos.

  Este exercício demonstra como as reduções em um termo bem tipado mantêm o tipo consistente.

**5**. Dê um exemplo de um termo que não satisfaz a propriedade de progresso no cálculo lambda não tipado, mas que seria rejeitado no cálculo lambda tipado.

   **Solução**: considere o termo $(\lambda x. x x) (\lambda x. x x)$. No cálculo lambda não tipado, este termo reduz infinitamente para si mesmo:

   $$
   (\lambda x. x x) (\lambda x. x x) \rightarrow_\beta (\lambda x. x x) (\lambda x. x x) \rightarrow_\beta ...
   $$

   No cálculo lambda tipado, este termo seria rejeitado porque não é possível atribuir um tipo consistente para $x$ em $x x$.

**6**. Explique como a propriedade de normalização forte garante que não existem loops infinitos em programas bem tipados no cálculo lambda tipado.

   **Solução**: a normalização forte garante que toda sequência de reduções de um termo bem tipado eventualmente termina em uma forma normal. Isso implica que não pode haver loops infinitos, pois se houvesse, a sequência de reduções nunca terminaria, contradizendo a propriedade de normalização forte.

**7**. Considere o termo $(\lambda x:\text{Nat}\rightarrow\text{Nat}. x 3) (\lambda y:\text{Nat}. y * 2)$. Mostre que este termo satisfaz as propriedades de preservação de tipos e progresso.

   **Solução**: preservação de tipos: O termo inicial tem tipo $\text{Nat}$. Após a redução:

   $$
   (\lambda x:\text{Nat}\rightarrow\text{Nat}. x 3) (\lambda y:\text{Nat}. y * 2) \rightarrow_\beta (\lambda y:\text{Nat}. y * 2) 3 \rightarrow_\beta 3 * 2 \rightarrow 6
   $$

   O resultado final $6$ ainda é do tipo $\text{Nat}$.

   Progresso: O termo inicial não está em forma normal e pode ser reduzido, como mostrado acima.

**8**. Explique por que a decidibilidade da tipagem é importante para compiladores de linguagens de programação tipadas.

   **Solução**: a decidibilidade da tipagem permite que compiladores verifiquem estaticamente se um programa está bem tipado. Isso é crucial para detectar erros de tipo antes da execução do programa, melhorando a segurança e a eficiência. Sem esta propriedade, seria impossível garantir que um programa está livre de erros de tipo em tempo de compilação.

**9**. Dê um exemplo de um termo que é bem tipado no cálculo lambda tipado, mas que não teria uma representação direta em uma linguagem sem tipos de ordem superior (como C).

   **Solução**: considere o termo:

   $$
   \lambda f:(\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat}. f (\lambda x:\text{Nat}. x + 1)
   $$

   Este termo tem tipo $((\text{Nat}\rightarrow\text{Nat})\rightarrow\text{Nat})\rightarrow\text{Nat}$. Ele representa uma função que toma como argumento outra função (que por sua vez aceita uma função como argumento). Linguagens sem tipos de ordem superior, como C, não podem representar diretamente funções que aceitam ou retornam outras funções.

**10**. Como a propriedade de preservação de tipos contribui para a segurança de execução em linguagens de programação baseadas no cálculo lambda tipado?

   **Solução**: a preservação de tipos garante que, à medida que um programa é executado (ou seja, à medida que os termos são reduzidos), os tipos dos termos permanecem consistentes. Isso significa que operações bem tipadas no início da execução permanecerão bem tipadas durante toda a execução. Esta propriedade previne erros de tipo em tempo de execução, contribuindo significativamente para a segurança de execução ao garantir que operações inválidas (como tentar adicionar um booleano a um número) nunca ocorrerão durante a execução de um programa bem tipado.

## Correspondência de Curry-Howard

A Correspondência de Curry-Howard, também conhecida como Isomorfismo de Curry-Howard estabelece uma profunda conexão entre tipos em linguagens de programação e proposições em lógica construtiva.

O isomorfismo de Curry-Howard tem raízes no trabalho realizado por um conjunto de pesquisadores ao longo do século XX. [Haskell Curry](https://en.wikipedia.org/wiki/Haskell_Curry), em 1934, foi o primeiro a observar uma conexão entre a lógica combinatória e os tipos de funções, notando que os tipos dos combinadores correspondiam a tautologias na lógica proposicional.

Um longo hiato se passou, até que [William Howard](https://en.wikipedia.org/wiki/William_Alvin_Howard), em 1969, expandiu esta observação para um isomorfismo completo entre lógica intuicionista e cálculo lambda tipado, mostrando que as regras de dedução natural correspondiam às regras de tipagem no cálculo lambda simplesmente tipado.

A correspondência foi posteriormente generalizada por [Jean-Yves Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard) e [John C. Reynolds](https://en.wikipedia.org/wiki/John_C._Reynolds), que independentemente, em 1971-72, estenderam o isomorfismo para incluir a quantificação de segunda ordem. Eles demonstraram que o Sistema F (cálculo lambda polimórfico) corresponde à lógica de segunda ordem, estabelecendo assim as bases para uma compreensão profunda da relação entre lógica e computação. Estas descobertas tiveram um impacto no desenvolvimento de linguagens de programação e sistemas de prova assistidos por computador.

Assim, chegamos aos dias atuais com a correspondência Curry-Howard tendo implicações tanto para a teoria da computação quanto para o desenvolvimento de linguagens de programação. Vamos examinar os principais aspectos deste isomorfismo:

### 1. Proposições como tipos

Na lógica construtiva, **a verdade de uma proposição é equivalente à sua demonstrabilidade**. Isso significa que para uma proposição $\phi$ ser verdadeira, deve existir uma prova de $\phi$.

Formalmente: Uma proposição $\phi$ é verdadeira se, e somente se, existe uma prova $p$ de $\phi$, denotada por $p : \phi$.

**Exemplo**: A proposição "existe um número primo maior que 100" é verdadeira porque podemos fornecer uma prova construtiva, como o número 101 e uma demonstração de que este é um número primo.

### 2. Provas como programas

**As regras de inferência na lógica construtiva correspondem às regras de tipagem em linguagens de programação**. Assim, um programa bem tipado pode ser visto como uma prova de sua especificação tipo.

Formalmente: Se $\Gamma \vdash e : \tau$, então $e$ pode ser interpretado como uma prova da proposição representada por $\tau$ no contexto $\Gamma$.

**Exemplo**: Um programa $e$ do tipo $\text{Nat} \rightarrow \text{Nat}$ é uma prova da proposição "existe uma função dos números naturais para os números naturais".

### 3. Correspondência entre conectivos lógicos e tipos

Existe uma correspondência direta entre conectivos lógicos e construtores de tipos:

1. Conjunção ($\wedge$) = Tipo produto ($\times$)
2. Disjunção ($\vee$) = Tipo soma ($+$)
3. Implicação ($\rightarrow$) = Tipo função ($\rightarrow$)
4. Quantificação universal ($\forall$) = Polimorfismo paramétrico ($\forall$)

**Exemplo**: A proposição $\phi_1 \wedge \phi_2 \rightarrow \phi_3$ corresponde ao tipo $\tau_1 \times \tau_2 \rightarrow \tau_3$.

### 4. Invalidade e tipos inabitados

Uma proposição falsa na lógica construtiva corresponde a um tipo inabitado na teoria de tipos.

Formalmente: Uma proposição $\phi$ é falsa se e somente se não existe termo $e$ tal que $e : \phi$.

**Exemplo**: O tipo $\forall X. X$ é inabitado, correspondendo a uma proposição falsa na lógica.

Estas correspondências fornecem uma base sólida para o desenvolvimento de linguagens de programação com sistemas de tipos expressivos e para a verificação formal de programas.

## Sintaxe do Cálculo Lambda Tipado

O cálculo lambda tipado estende o cálculo lambda não tipado, adicionando uma estrutura de tipos que restringe a formação e a aplicação de funções. Essa extensão preserva os princípios fundamentais do cálculo lambda, mas introduz um sistema de tipos que assegura maior consistência e evita paradoxos lógicos. Enquanto no cálculo lambda não tipado as funções podem ser aplicadas livremente a qualquer argumento, o cálculo lambda tipado impõe restrições que garantem que as funções sejam aplicadas apenas a argumentos compatíveis com seu tipo.

No cálculo lambda tipado, as expressões são construídas a partir de três elementos principais: variáveis, abstrações e aplicações. Esses componentes definem a estrutura básica das funções e seus argumentos, e a adição de tipos funciona como um mecanismo de segurança, assegurando que as funções sejam aplicadas de forma correta. Uma variável $x$, por exemplo, é anotada com um tipo específico como $x : A$, onde $A$ pode ser um tipo básico como $\text{Nat}$ ou $\text{Bool}$, ou um tipo de função como $A \rightarrow B$.

### Abstrações Lambda e Tipos

No cálculo lambda tipado, as abstrações são expressas na forma $\lambda x : A. M$, onde $x$ é uma variável de tipo $A$ e $M$ é a expressão cujo resultado dependerá de $x$. O tipo dessa abstração é dado por $A \rightarrow B$, onde $B$ é o tipo do resultado de $M$. Por exemplo, a abstração $\lambda x : \text{Nat}. \, x + 1$ define uma função que aceita um argumento do tipo $\text{Nat}$ (número natural) e retorna outro número natural. Nesse caso, o tipo da abstração é $\text{Nat} \rightarrow \text{Nat}$, o que significa que a função mapeia um número natural para outro número natural.

$$\lambda x : \text{Nat}. \, x + 1 : \text{Nat} \rightarrow \text{Nat}$$

As variáveis no cálculo lambda tipado podem ser livres ou ligadas. Variáveis livres são aquelas que não estão associadas a um valor específico dentro do escopo da função, enquanto variáveis ligadas são aquelas definidas no escopo da abstração. Esse conceito de variáveis livres e ligadas é familiar na lógica de primeira ordem e tem grande importância na estruturação das expressões lambda.

### Aplicações de Funções

A aplicação de funções segue a mesma sintaxe do cálculo lambda não tipado, mas no cálculo tipado é restrita pelos tipos dos termos envolvidos. Se uma função $f$ tem o tipo $A \rightarrow B$, então ela só pode ser aplicada a um termo $x$ do tipo $A$. A aplicação de $f$ a $x$ resulta em um termo do tipo $B$. Um exemplo simples seria a aplicação da função de incremento $\lambda x : \text{Nat}. \, x + 1$ ao número 2:

$$(\lambda x : \text{Nat}. \, x + 1) \, 2 \rightarrow 3$$

Aqui, a função de tipo $\text{Nat} \rightarrow \text{Nat}$ é aplicada ao número $2$, e o resultado é o número $3$, que também é do tipo $\text{Nat}$.

### Regras de Tipagem

As regras de tipagem no cálculo lambda tipado são fundamentais para garantir que as expressões sejam bem formadas. Estas regras estabelecem a maneira como os tipos são atribuídos às variáveis, abstrações e aplicações. As regras principais incluem:

-**Regra da Variável**: Se uma variável $x$ tem tipo $A$ em um contexto $\Gamma$, podemos afirmar que $\Gamma \vdash x : A$.

-**Regra de Abstração**: Se, no contexto $\Gamma$, temos que $\Gamma, x : A \vdash M : B$, então $\Gamma \vdash (\lambda x : A. M) : A \rightarrow B$.

-**Regra de Aplicação**: Se $\Gamma \vdash M : A \rightarrow B$ e $\Gamma \vdash N : A$, então $\Gamma \vdash (M \, N) : B$.

Essas regras fornecem as bases para derivar tipos em expressões mais complexas, garantindo que as aplicações de funções e os argumentos sigam uma lógica de tipos consistente.

### Substituição e Redução

A operação de substituição no cálculo lambda tipado segue o mesmo princípio do cálculo não tipado, com a adição de restrições de tipo. Quando uma função é aplicada a um argumento, a variável vinculada à função é substituída pelo valor do argumento na expressão. Formalmente, a substituição de $N$ pela variável $x$ em $M$ é denotada por $[N/x]M$, indicando que todas as ocorrências livres de $x$ em $M$ devem ser substituídas por $N$.

A redução no cálculo lambda tipado segue a estratégia de $\beta$-redução, onde aplicamos a função ao seu argumento e substituímos a variável ligada pelo valor fornecido. Um exemplo clássico de $\beta$-redução seria:

$$(\lambda x : \text{Nat}. \, x + 1) \, 2 \rightarrow 2 + 1 \rightarrow 3$$

Esse processo de substituição e simplificação é a base para a computação de expressões no cálculo lambda tipado, e é fundamental para a avaliação de programas em linguagens de programação funcionais.

## Propriedades do Cálculo Lambda Tipado

O cálculo lambda tipado tem algumas propriedades importantes que o distinguem do cálculo não tipado. Uma dessas propriedades é a **normalização forte**, que garante que todo termo bem tipado possui uma forma normal, e que qualquer sequência de reduções eventualmente terminará. Outra propriedade é a **preservação de tipos**, que assegura que se um termo $M$ tem tipo $A$ e $M \rightarrow_\beta N$, então $N$ também terá o tipo $A$. Além disso, a tipagem no cálculo lambda tipado é **decidível**, o que significa que existe um algoritmo para determinar se um termo tem ou não um tipo válido.

## Regras de Tipagem no Cálculo Lambda Tipado

As regras de tipagem no cálculo lambda tipado formam a espinha dorsal de um sistema que assegura a consistência e a correção das expressões. A tipagem previne a formação de termos paradoxais e, ao mesmo tempo, estabelece uma base sólida para o desenvolvimento de linguagens de programação seguras e de sistemas de prova assistida por computador. Ao impor que variáveis e funções sejam usadas apenas em conformidade com seus tipos, o cálculo lambda tipado garante que a aplicação de funções a argumentos ocorra de maneira correta.

### Sistema de Tipos

No cálculo lambda tipado, os tipos podem ser básicos ou compostos. Tipos básicos incluem, por exemplo, $\text{Bool}$, que representa valores booleanos, e $\text{Nat}$, que denota números naturais. Tipos de função são construídos a partir de outros tipos; $A \rightarrow B$ denota uma função que mapeia valores do tipo $A$ para valores do tipo $B$. O sistema de tipos, portanto, tem uma estrutura recursiva, permitindo a construção de tipos complexos a partir de tipos mais simples.

A tipagem de variáveis assegura que cada variável esteja associada a um tipo específico. Uma variável $x$ do tipo $A$ é denotada como $x : A$. Isso implica que $x$ só pode ser associado a valores que respeitem as regras do tipo $A$, restringindo o comportamento da função.

Um **contexto de tipagem**, representado por $\Gamma$, é um conjunto de associações entre variáveis e seus tipos. O contexto fornece informações necessárias sobre as variáveis livres em uma expressão, facilitando o julgamento de tipos. Por exemplo, um contexto $\Gamma = \{x : A, y : B\}$ indica que, nesse ambiente, a variável $x$ tem tipo $A$ e a variável $y$ tem tipo $B$. Os contextos são essenciais para derivar os tipos de expressões mais complexas.

### Normalização Forte e Fraca

O cálculo lambda é um sistema minimalista. Mas tem muita força. É a base da programação funcional. E possui propriedades poderosas. Uma característica importante para a criação de linguagens de programação é a normalização. Existem dois tipos de normalização:

1. Normalização fraca: Todo termo tem uma forma normal. Você vai chegar lá eventualmente.

2. Normalização forte: Toda sequência de reduções termina. Não importa como você reduz, vai alcançar uma forma normal.

A normalização forte é o que realmente interessa. É o que queremos.

No cálculo lambda simplesmente tipado, temos normalização forte. Isso é algo belo. Como qualquer coisa na matemática, a normalização precisa ser provada. Para facilitar, vamos ver uma prova informal. A prova não é elegante. Mas aqui está a essência:

1. Atribuímos um "tamanho" a cada termo.
2. Mostramos que cada redução torna o termo menor.
3. Como não dá para diminuir para sempre, você tem que parar.

Os detalhes são complicados. Mas essa é a ideia. Aqui está um esboço da função de tamanho:

$$size(\lambda x:A.M) = size(M) + 1$$

$$size(MN) = size(M) + size(N) + 1$$

Cada $\beta$-redução diminui o termo. Não dá para reduzir para sempre. Então, você tem que parar.

A normalização não é só teoria. Ela é prática.

Em Haskell, a normalização garante a terminação de programas bem tipados. Nada de loops infinitos. Nada de falhas. Apenas funções puras que terminam.

O OCaml também usa isso. O sistema de tipos garante a normalização forte. É uma rede de segurança para programadores.

### Exemplos

Vamos ver a normalização em ação:

1. Função identidade:

 $(\lambda x:A.x)M \to M$

 Um passo. Acabou. Forma normal atingida.

2. Função constante:

 $(\lambda x:A.\lambda y:B.x)MN \to (\lambda y:B.M)N \to M$

 Dois passos. Forma normal atingida.

3. Números de Church:

 $2 3 \equiv (\lambda f.\lambda x.f(fx))(\lambda y.yyy) \to_\beta \lambda x.(\lambda y.yyy)((\lambda y.yyy)x) \to_\beta \lambda x.(xxx)(xxx) \to_\beta \lambda x.xxxxxxxxx$

 Vários passos. Mas termina. Isso é a normalização em ação.

A normalização é poderosa. É a espinha dorsal da programação funcional. É o que torna essas linguagens seguras e previsíveis.

Lembre-se: no cálculo lambda, tudo termina. Essa é a beleza. Esse é o poder da normalização.

### Regras de Tipagem Fundamentais

As regras de tipagem no cálculo lambda tipado são geralmente expressas através da inferência natural. Abaixo, as regras fundamentais são detalhadas, sempre partindo de premissas para uma conclusão.

#### Regra da Variável

A regra da variável afirma que, se uma variável $x$ tem tipo $A$ no contexto $\Gamma$, então podemos derivar que $x$ tem tipo $A$ nesse contexto:

$$\frac{x : A \in \Gamma}{\Gamma \vdash x : A}$$

Essa regra formaliza a ideia de que, se sabemos que $x$ tem tipo $A$ a partir do contexto, então $x$ pode ser usada em expressões como um termo de tipo $A$.

#### Regra de Abstração

A regra de abstração define o tipo de uma função. Se, assumindo que $x$ tem tipo $A$, podemos derivar que $M$ tem tipo $B$, então a abstração $\lambda x : A . M$ tem o tipo $A \rightarrow B$. Formalmente:

$$\frac{\Gamma, x : A \vdash M : B}{\Gamma \vdash (\lambda x : A . M) : A \rightarrow B}$$

Essa regra assegura que a função $\lambda x : A . M$ é corretamente formada e mapeia valores do tipo $A$ para resultados do tipo $B$.

#### Regra de Aplicação

A regra de aplicação governa a forma como funções são aplicadas a seus argumentos. Se $M$ é uma função do tipo $A \rightarrow B$ e $N$ é um termo do tipo $A$, então a aplicação $M \, N$ tem tipo $B$:

$$\frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M \, N : B}$$

Essa regra garante que, ao aplicar uma função $M$ a um argumento $N$, a aplicação resulta em um termo do tipo esperado $B$.

### Termos Bem Tipados e Segurança do Sistema

Um termo é considerado **bem tipado** se sua derivação de tipo pode ser construída usando as regras de tipagem formais. A tipagem estática é uma característica importante do cálculo lambda tipado, pois permite detectar erros de tipo durante o processo de compilação, antes mesmo de o programa ser executado. Isso é essencial para a segurança e confiabilidade dos sistemas, já que garante que funções não sejam aplicadas a argumentos incompatíveis.

Além disso, o sistema de tipos do cálculo lambda tipado exclui automaticamente termos paradoxais como o combinador $\omega = \lambda x. \, x \, x$. Para que $\omega$ fosse bem tipado, a variável $x$ precisaria ter o tipo $A \rightarrow A$ e ao mesmo tempo o tipo $A$, o que é impossível. Assim, a auto-aplicação de funções é evitada, garantindo a consistência do sistema.

### Propriedades do Sistema de Tipos

O cálculo lambda tipado apresenta várias propriedades que reforçam a robustez do sistema:

-**Normalização Forte**: Todo termo bem tipado tem uma forma normal, ou seja, pode ser reduzido até uma forma final através de $\beta$-redução. Isso implica que termos bem tipados não entram em loops infinitos de computação.

-**Preservação de Tipos (Subject Reduction)**: Se $\Gamma \vdash M : A$ e $M \rightarrow_\beta N$, então $\Gamma \vdash N : A$. Isso garante que a tipagem é preservada durante a redução de termos, assegurando a consistência dos tipos ao longo das transformações.

-**Progresso**: Um termo bem tipado ou é um valor (isto é, está em sua forma final), ou pode ser reduzido. Isso significa que termos bem tipados não ficam presos em estados intermediários indeterminados.

-**Decidibilidade da Tipagem**: É possível determinar, de maneira algorítmica, se um termo é bem tipado e, em caso afirmativo, qual é o seu tipo. Essa propriedade é essencial para a verificação automática de tipos em sistemas formais e linguagens de programação.

### Correspondência de Curry-Howard

A **correspondência de Curry-Howard** estabelece uma relação profunda entre o cálculo lambda tipado e a lógica proposicional intuicionista. Sob essa correspondência, termos no cálculo lambda tipado são vistos como provas, e tipos são interpretados como proposições. Em particular:

-Tipos correspondem a proposições.
-Termos correspondem a provas.
-Normalização de termos corresponde à normalização de provas.

Por exemplo, o tipo $A \rightarrow B$ pode ser interpretado como a proposição lógica "se $A$, então $B$", e um termo deste tipo representa uma prova dessa proposição. Essa correspondência fornece a base para a verificação formal de programas e para a lógica assistida por computador.

## Conversão e Redução no Cálculo Lambda Tipado

No cálculo lambda tipado, os processos de conversão e redução são essenciais para a manipulação e simplificação de expressões, garantindo que as transformações sejam consistentes com a estrutura de tipos. Essas operações são fundamentais para entender como as funções são aplicadas e como as expressões podem ser transformadas mantendo a segurança e a consistência do sistema tipado.

### Redução $\beta$

A**$\beta$-redução**é o mecanismo central de computação no cálculo lambda tipado. Ela ocorre quando uma função é aplicada a um argumento, substituindo todas as ocorrências da variável ligada pelo valor do argumento na expressão. Formalmente, se temos uma abstração $\lambda x : A . M$ e aplicamos a um termo $N$ do tipo $A$, a $\beta$-redução é expressa como:

$$(\lambda x : A . M) \, N \rightarrow\_\beta M[N/x]$$

onde $M[N/x]$ denota a substituição de todas as ocorrências livres de $x$ em $M$ por $N$. A $\beta$-redução é o passo básico da computação no cálculo lambda, e sua correta aplicação preserva os tipos das expressões envolvidas.

Por exemplo, considere a função de incremento aplicada ao número $2$:

$$(\lambda x : \text{Nat} . \, x + 1) \, 2 \rightarrow\_\beta 2 + 1 \rightarrow 3$$

Aqui, a variável $x$ é substituída pelo valor $2$ e, em seguida, a expressão é simplificada para $3$. No cálculo lambda tipado, a $\beta$-redução garante que os tipos sejam preservados, de modo que o termo final também é do tipo $\text{Nat}$, assim como o termo original.

### Conversões $\alpha$ e $\eta$

Além da $\beta$-redução, existem duas outras formas importantes de conversão no cálculo lambda: a**$\alpha$-conversão**e a**$\eta$-conversão**.

-**$\alpha$-conversão**: Esta operação permite a renomeação de variáveis ligadas, desde que a nova variável não conflite com variáveis livres. Por exemplo, as expressões $\lambda x : A . \, x$ e $\lambda y : A . y$ são equivalentes sob $\alpha$-conversão:

$$\lambda x : A . \, x \equiv\_\alpha \lambda y : A . y$$

 A $\alpha$-conversão é importante para evitar a captura de variáveis durante o processo de substituição, garantindo que a renomeação de variáveis ligadas não afete o comportamento da função.

-**$\eta$-conversão**: A $\eta$-conversão expressa o princípio de extensionalidade, que afirma que duas funções são idênticas se elas produzem o mesmo resultado para todos os argumentos. Formalmente, a $\eta$-conversão permite que uma abstração lambda da forma $\lambda x : A . f \, x$ seja convertida para $f$, desde que $x$ não ocorra livre em $f$:

$$\lambda x : A . f \, x \rightarrow\_\eta f$$

 A $\eta$-conversão simplifica as funções removendo abstrações redundantes, tornando as expressões mais curtas e mais diretas.

### Normalização e Estratégias de Redução

Uma das propriedades mais importantes do cálculo lambda tipado é a **normalização forte**, que garante que todo termo bem tipado pode ser reduzido até uma **forma normal**, uma expressão que não pode mais ser simplificada. Isso significa que qualquer sequência de reduções, eventualmente, terminará, o que contrasta com o cálculo lambda não tipado, onde reduções infinitas são possíveis.

Existem diferentes estratégias de redução que podem ser aplicadas ao calcular expressões no cálculo lambda tipado:

1. **Redução por ordem normal**: Nessa estratégia, reduzimos sempre o redex mais à esquerda e mais externo primeiro. Essa abordagem garante que, se existir uma forma normal, ela será encontrada.

2. **Redução por ordem de chamada (call-by-name)**: Nesta estratégia, apenas os termos que realmente são necessários para a computação são reduzidos. Isso implementa uma avaliação "preguiçosa", comum em linguagens funcionais como Haskell.

3. **Redução por valor (call-by-value)**: Nesta estratégia, os argumentos são completamente reduzidos antes de serem aplicados às funções. Isso é típico de linguagens com avaliação estrita, como OCaml ou ML.

Todas essas estratégias são **normalizantes**no cálculo lambda tipado, ou seja, alcançarão uma forma normal, se ela existir, devido à normalização forte.

### Preservação de Tipos e Segurança

Um princípio fundamental no cálculo lambda tipado é a **preservação de tipos** durante a redução, também conhecida como **subject reduction**. Essa propriedade assegura que, se um termo $M$ tem um tipo $A$ e $M$ é reduzido a $N$ através de $\beta$-redução, então $N$ também terá o tipo $A$. Formalmente:

$$
\frac{\Gamma \vdash M : A \quad M \rightarrow\_\beta N}{\Gamma \vdash N : A}
$$

Essa propriedade, combinada com a **propriedade de progresso**, que afirma que todo termo bem tipado ou é um valor ou pode ser reduzido, estabelece a segurança do sistema de tipos no cálculo lambda tipado. Isso garante que, durante a computação, nenhum termo incorreto em termos de tipo será gerado.

### Confluência e Unicidade da Forma Normal

O cálculo lambda tipado possui a propriedade de **confluência**, também conhecida como **propriedade de Church-Rosser**. Confluência significa que, se um termo $M$ pode ser reduzido de duas maneiras diferentes para dois termos $N_1$ e $N_2$, sempre existirá um termo comum $P$ tal que $N_1$ e $N_2$ poderão ser reduzidos a $P$:

$$
M \to N*1 \quad M \rightarrow^* N*2 \quad \Rightarrow \quad \exists P : N_1 \rightarrow^* P \quad N*2 \rightarrow^* P
$$

A confluência, combinada com a normalização forte, garante a **unicidade da forma normal** para termos bem tipados. Isso significa que, independentemente da ordem de redução escolhida, um termo bem tipado sempre converge para a mesma forma normal, garantindo consistência e previsibilidade no processo de redução.

## A Teoria dos Tipos Simples

A **Teoria dos Tipos Simples**, desenvolvida por Alonzo Church na década de 1940, representa um marco na história da lógica matemática e da ciência da computação. Criada para resolver problemas de inconsistência no cálculo lambda não tipado, essa teoria introduziu um framework robusto para formalizar o raciocínio matemático e computacional, abordando paradoxos semelhantes ao **paradoxo de Russell** na teoria dos conjuntos. A teoria dos tipos simples foi uma das primeiras soluções práticas para garantir que expressões lambda fossem bem formadas, evitando contradições lógicas e permitindo cálculos confiáveis.

O cálculo lambda não tipado, proposto por Church na década de 1930, ofereceu um modelo poderoso de computabilidade, mas sua flexibilidade permitiu a formulação de termos paradoxais, como o **combinador Y** (um fixpoint combinator) e o termo**$\omega = \lambda x. \, x\ x$**, que resulta em reduções infinitas. Esses termos paradoxais tornavam o cálculo lambda inconsistente, uma vez que permitiam a criação de expressões que não convergiam para uma forma normal, gerando loops infinitos.

O problema era análogo aos paradoxos que surgiram na teoria dos conjuntos ingênua, como o paradoxo de Russell. A solução proposta por Church envolvia restringir o cálculo lambda através da introdução de tipos, criando um sistema onde apenas combinações de funções e argumentos compatíveis fossem permitidas, prevenindo a criação de termos paradoxais.

### Fundamentos da Teoria dos Tipos Simples

A ideia central da **Teoria dos Tipos Simples** é organizar as expressões lambda em uma hierarquia de tipos que impõe restrições sobre a formação de termos. Isso garante que termos paradoxais, como $\omega$, sejam automaticamente excluídos. A estrutura básica da teoria é composta por:

1.**Tipos Base**: Esses são os tipos fundamentais, como $\text{Bool}$ para valores booleanos e $\text{Nat}$ para números naturais. Esses tipos representam os elementos básicos manipulados pelo sistema.

2.**Tipos de Função**: Se $A$ e $B$ são tipos, então $A \rightarrow B$ representa uma função que recebe um valor do tipo $A$ e retorna um valor do tipo $B$. Esta construção é crucial para definir funções no cálculo lambda tipado.

3.**Hierarquia de Tipos**: Os tipos formam uma hierarquia estrita. Tipos base estão na camada inferior, enquanto os tipos de função, que podem tomar funções como argumentos e retornar funções como resultados, estão em níveis superiores. Isso evita que funções sejam aplicadas a si mesmas de maneira paradoxal, como em $\lambda x . \, x \, x$.

### Sistema de Tipos e Regras de Tipagem

O **sistema de tipos** no cálculo lambda tipado simples é definido por um conjunto de regras que especificam como os tipos podem ser atribuídos aos termos. Essas regras garantem que as expressões sejam consistentes e bem formadas. As três regras fundamentais são:

-**Regra da Variável**: Se uma variável $x$ tem o tipo $A$ no contexto $\Gamma$, então ela é bem tipada:

$$
\frac{x : A \in \Gamma}{\Gamma \vdash x : A}
$$

-**Regra da Abstração**: Se, no contexto $\Gamma$, assumimos que $x$ tem tipo $A$ e podemos derivar que $M$ tem tipo $B$, então $\lambda x : A . M$ é uma função bem tipada que mapeia de $A$ para $B$:

$$
\frac{\Gamma, x : A \vdash M : B}{\Gamma \vdash (\lambda x : A . M) : A \rightarrow B}
$$

-**Regra da Aplicação**: Se $M$ é uma função do tipo $A \rightarrow B$ e $N$ é um termo do tipo $A$, então a aplicação $M \, N$ resulta em um termo do tipo $B$:

$$
\frac{\Gamma \vdash M : A \rightarrow B \quad \Gamma \vdash N : A}{\Gamma \vdash M \, N : B}
$$

Essas regras garantem que as expressões sejam tipadas corretamente e que o sistema evite inconsistências lógicas, como a auto-aplicação de funções.

### Propriedades Fundamentais

A **Teoria dos Tipos Simples** apresenta várias propriedades importantes que a tornam um sistema robusto para lógica e computação:

1. **Consistência**: Ao contrário do cálculo lambda não tipado, o sistema de tipos simples é consistente. Isso significa que nem todas as proposições podem ser provadas, e o sistema não permite a formação de paradoxos.

2. **Normalização Forte**: Todo termo bem tipado no cálculo lambda simples possui uma forma normal, e qualquer sequência de reduções eventualmente termina. Essa propriedade garante que os cálculos são finitos e que todos os termos se resolvem em uma forma final.

3. **Preservação de Tipos (Subject Reduction)**: Se um termo $M$ tem tipo $A$ e $M$ é reduzido para $N$, então $N$ também terá o tipo $A$. Isso garante que a tipagem é preservada durante as operações de redução.

4. **Decidibilidade da Tipagem**: É possível determinar, de forma algorítmica, se um termo é bem tipado e, em caso afirmativo, qual é o seu tipo. Essa propriedade é crucial para a verificação automática de programas e provas.

### Impacto e Aplicações

A **Teoria dos Tipos Simples** influenciou diversas áreas da ciência da computação e da lógica matemática:

1. **Linguagens de Programação**: Sistemas de tipos modernos, como os usados em linguagens funcionais como ML e Haskell, são derivados da teoria dos tipos simples. A tipagem estática ajuda a detectar erros antes da execução do programa, aumentando a segurança do software.

2. **Verificação Formal**: A teoria dos tipos simples fornece a base para sistemas de prova assistida por computador, como**Coq**e**Isabelle**, que permitem a formalização de teoremas matemáticos e sua verificação automática.

3. **Semântica de Linguagens**: A teoria dos tipos simples contribui para a semântica formal das linguagens de programação, oferecendo uma maneira rigorosa de descrever o comportamento das construções de linguagem.

4. **Lógica Computacional**: A teoria dos tipos simples é intimamente ligada à**correspondência de Curry-Howard**, que estabelece uma relação entre proposições lógicas e tipos, e entre provas e programas. Esta correspondência é central para entender a conexão entre lógica e computação.

### Limitações e Extensões

Embora poderosa, a **Teoria dos Tipos Simples** tem limitações:

1. **Expressividade Limitada**: O sistema não pode expressar diretamente conceitos como indução, que são importantes em muitos contextos matemáticos.

2. **Ausência de Polimorfismo**: Não há suporte nativo para funções polimórficas, que operam de forma genérica sobre múltiplos tipos.

Para superar essas limitações, surgiram várias extensões da teoria:

1. **Sistemas de Tipos Polimórficos**: Como o **Sistema F** de [Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard), que introduz quantificação sobre tipos, permitindo a definição de funções polimórficas.

2. **Teoria dos Tipos Dependentes**: Extensões que permitem que tipos dependam de valores, aumentando significativamente a expressividade e permitindo raciocínios mais complexos.

3. **Teoria dos Tipos Homotópica**: Uma extensão recente que conecta a teoria dos tipos com a topologia algébrica, oferecendo novos insights sobre a matemática e a computação.

# Notas e Referências

[^nota1]: Os problemas de Hilbert são uma lista de 23 problemas matemáticos propostos por David Hilbert em 1900, durante o Congresso Internacional de Matemáticos em Paris. Esses problemas abordam questões fundamentais em várias áreas da matemática e estimularam muitas descobertas ao longo do século XX. Cada problema visava impulsionar a pesquisa e delinear os desafios mais importantes da matemática da época. Alguns dos problemas foram resolvidos, enquanto outros permanecem abertos ou foram provados como indecidíveis, como o **décimo problema de Hilbert**, que pergunta se existe um algoritmo capaz de determinar se um polinômio com coeficientes inteiros possui soluções inteiras.

[^nota2]: O paradigma de _provas como programas_ é uma correspondência entre demonstrações matemáticas e programas de computador, também conhecida como **correspondência de Curry-Howard**. Segundo esse paradigma, cada prova em lógica formal corresponde a um programa e cada tipo ao qual uma prova pertence corresponde ao tipo de dado que um programa manipula. Essa ideia estabelece uma ponte entre a lógica e a teoria da computação, permitindo a formalização de demonstrações como estruturas computáveis e o desenvolvimento de sistemas de prova automáticos e seguros.

[^nota3]: O **Sistema F**, também conhecido como cálculo lambda polimórfico de segunda ordem, é uma extensão do cálculo lambda que permite quantificação universal sobre tipos. Desenvolvido por [Jean-Yves Girard](https://en.wikipedia.org/wiki/Jean-Yves_Girard) e [John Reynolds](https://en.wikipedia.org/wiki/John_C._Reynolds) de forma independente, o Sistema F é fundamental para a teoria da tipagem em linguagens de programação, permitindo expressar abstrações mais poderosas, como tipos genéricos e polimorfismo paramétrico. Ele serve como uma base para a formalização de muitos sistemas de tipos usados em linguagens funcionais modernas.

[^nota4]: O **Cálculo de Construções** é um sistema formal que combina elementos do cálculo lambda e da teoria dos tipos para fornecer uma base para a lógica construtiva. Ele foi desenvolvido por [Thierry Coquand](https://en.wikipedia.org/wiki/Thierry_Coquand) e é uma extensão do Sistema F, com a capacidade de definir tipos dependentes e níveis mais complexos de abstração. O cálculo de construções é a base da linguagem **Coq**, um assistente de prova amplamente utilizado para formalizar demonstrações matemáticas e desenvolver software verificado.

[^nota5]: Extensionalidade refere-se ao princípio de que objetos ou funções são iguais se têm o mesmo efeito em todos os contextos possíveis. Em lógica, duas funções são consideradas extensionais se, para todo argumento, elas produzem o mesmo resultado. Em linguística, extensionalidade se refere a expressões cujo significado é determinado exclusivamente por seu valor de referência, sem levar em conta contexto ou conotação.

[^nota6]: A lógica intuicionista é um sistema formal de lógica desenvolvido por [Arend Heyting](https://en.wikipedia.org/wiki/Arend_Heyting), baseado nas ideias do matemático [L.E.J. Brouwer](https://en.wikipedia.org/wiki/L._E._J._Brouwer). Diferentemente da lógica clássica, a lógica intuicionista rejeita o princípio do terceiro excluído (A ou não-A) e a lei da dupla negação (não-não-A implica A). Ela exige provas construtivas, onde a existência de um objeto matemático só é aceita se houver um método para construí-lo. Esta abordagem tem implicações profundas na matemática e na ciência da computação, especialmente na teoria dos tipos e na programação funcional, onde se alinha naturalmente com o conceito de computabilidade.

[^cita1]: Schönfinkel, Moses. "Über die Bausteine der mathematischen Logik." *Mathematische Annalen*, vol. 92, no. 1-2, 1924, pp. 305-316.

[^cita2]: Malpas, J., “Donald Davidson”, The Stanford Encyclopedia of Philosophy (Winter 2012 Edition), Edward N. Zalta and Uri Nodelman (eds.), URL = <https://plato.stanford.edu/entries/lambda-calculus/#Com>.

[^cita3]: DOMINUS, Mark, Why is the S combinator an S?, URL = <https://blog.plover.com/math/combinator-s.html>.

[^cita4]: CARDONE, Felice; HINDLEY, J. Roger. *History of Lambda-calculus and Combinatory Logic*. Swansea University Mathematics Department Research Report No. MRRS-05-06, 2006. URL = <https://hope.simons-rock.edu/~pshields/cs/cmpt312/cardone-hindley.pdfl>.

[^cita5]: Alonzo Church and J.B. Rosser. Some properties of conversion.  
*Transactions of the American Mathematical Society*, 39(3):472–482, May 1936.

[^cita6]: Alan Turing. On computable numbers, with an application to the entscheidungsproblem.  
*Proceedings of the London Mathematical Society*, 42:230–265, 1936. Published 1937.

[^cita7]: SELINGER, Peter. *Lecture Notes on the Lambda Calculus*. Department of Mathematics and Statistics, Dalhousie University, Halifax, Canada.
