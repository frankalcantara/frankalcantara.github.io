---
layout: post
title: Brevíssima Introdução a Ciência da Computação
author: Frank
categories:
    - artigo
    - Matemática
    - disciplina
tags:
    - algoritmos
    - Matemática
    - eng. software
image: ""
featured: false
rating: 5
description: Uma brevíssima introdução ao conceito de monad e seu uso em linguagens de programação.
date: 2023-09-29T21:53:10.540Z
preview: A Ciência da Computação é a arte de aplicar a matemática para utilizar máquinas na resolução de problemas. Nesta jornada vamos ver o mínimo de matemática necessário para entender como fazer magia com computadores.
keywords: ""
slug: brevissima-introducao-a-ciencia-da-computacao
draft: 2023-09-29T21:53:10.540Z
published: false
lastmod: 2025-05-06T11:04:17.750Z
---
Nesta jornada exploraremos as profundezas da álgebra abstrata, começando nos conceitos de conjuntos, nos levando a uma análise de relações e funções. Este é o ponto de partida, onde lançamos as bases para uma compreensão mais profunda das estruturas matemáticas que virão a seguir, desde as simples _magmas_ até os mais complexos anéis, corpos, monads e tipos. À medida que avançarmos o terreno se tornará mais complexo e intrigante. Explorarmos categorias, morfismos e, tipos e tipos paramétricos.

Com uma estrutura bem planejada, este percurso não só serve como uma introdução à Ciência da Computação, mas também como uma fundação sólida para os estudos mais avançados de linguagens de programação e algoritmos. Prepare-se para uma viagem que irá expandir o seu conhecimento, e a sua apreciação pela beleza e elegância que a matemática pode oferecer à resolução de problemas.

Contrariando o senso comum, a Ciência da Computação não nasce dos esforços de [Pascal](https://en.wikipedia.org/wiki/Blaise_Pascal), [Leibniz](https://en.wikipedia.org/wiki/Gottfried_Wilhelm_Leibniz) ou [Babagge](https://en.wikipedia.org/wiki/Charles_Babbage). Ainda que seus trabalhos sejam importantes, e sirvam de base para a matemática desenvolvida no final do século XIX, a Ciência da Computação tem data de nascimento, 12 de novembro de 1937, dia em que [Alan Touring](https://en.wikipedia.org/wiki/Alan_Turing) publicou [_On Computable Numbers, with an Application to the Entscheidungsproblem_](https://londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/plms/s2-42.1.230#:~:text=On%20Computable%20Numbers%2C%20with%20an,Mathematical%20Society%20%2D%20Wiley%20Online%20Library). Quando terminar de estudar este trabalho, a amável leitora há de concordar comigo. Para chegarmos lá, vamos andar para frente e para trás na história, como se este humilde timoneiro controlasse uma máquina do tempo. Faremos isso para criar um conjunto ordenado de porquês, antes de entendermos cada como relacionado a Ciência da Computação e por falar em conjuntos.

## Os Conjuntos são os Alicerces que Precisamos

No século XVII, matemáticos como [John Wallis](https://en.wikipedia.org/wiki/John_Wallis) e [Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton) já utilizavam conjuntos de forma informal em suas obras. [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler), no século XVIII, introduziu noções primitivas de conjuntos e funções em seus trabalhos. Esta é a base. Aqui começa o estudo dos conjuntos, uma base que, se a doce leitora quiser, pode ser trilhada até a aritmética grega, mas que iria muito além das nossas ambições. Esta base forneceu a estrutura sobre a qual [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor) se apoiou.

O ano de 1874 assistiu [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor) introduzir ao mundo à teoria dos conjuntos. Primeiramente delineada em [_Über eine Eigenschaft des Inbegriffes aller reellen algebraischen Zahlen_](https://link.springer.com/chapter/10.1007/978-3-7091-9516-1_2) (Sobre uma propriedade do conjunto de todos os números algébricos reais), o trabalho impactou fortemente o mundo da matemática a partir do [Journal de Crelle](https://www.degruyter.com/journal/key/crll/html). Em 1879, Cantor solidificou suas ideias espalhando-as através de seis artigos reveladores, impactantes e controversos no [Mathematische Annalen](https://www.springer.com/journal/208). Essas obras marcariam o terreno fértil para o desenvolvimento da matemática moderna.

Cantor não hesitou. Ele revelou que os números reais ultrapassavam os naturais em quantidade, desvendou os números transfinitos e a aritmética cardinal e forjou conjuntos infinitos não enumeráveis, inacreditavelmente desafiadores.

Foi uma caminhada solitária. O ceticismo sombreava cada passo, a comunidade matemática murmurava entre surpresa, medo e inveja. Mas a determinação de Cantor prevaleceu, plantando firmemente sua teoria como uma espinha dorsal não reconhecida, mas indispensável, da matemática moderna, ramificando-se silenciosamente mas poderosamente em análises, topologias, teoria dos grafos e medidas. À Teoria de Conjuntos de Cantor damos o nome de Inocente, por permitir a definição de conjuntos de forma não axiomática. Mas, afinal, o que é um conjunto?

Um conjunto é uma coleção de coisas, qualquer coisa. Como um rebanho de ovelhas, um cardume de peixes, ou os livros na sua biblioteca. Conjuntos podem ser finitos ou infinitos. Aliás, esta distinção cabe a Cantor. Um exemplo de conjunto finito é o rebanho de ovelhas, um exemplo de conjunto infinito é o conjunto dos números naturais $\mathbb{N}$ que contém os valores representados por $0$, $1$, $2$, $3$ e assim por diante, sem fim. Só a matemática para permitir o entendimento destes conceitos.

> Na matemática a arte de fazer perguntas tem mais valor que resolver problemas. Georg Cantor

Um conjunto é uma coleção bem definida de objetos distintos. Os elementos de um conjunto podem ser qualquer coisa: números, pessoas, figuras geométricas. Qualquer coisa pode ser um elemento e, independente do conjunto, e dos seus elementos, na matemática todos os elementos serão representados por símbolos. Dizemos que, Se $a$ é um elemento do conjunto $A$, nós escrevemos $a \in A$. Este $\in$ é o símbolo de pertencimento, dizemos que $a$ pertence a $A$. Mas, não para aí. A história dos conjuntos na matemática é longa.

Por exemplo, para definir o conjunto de todos os Números inteiros pares podemos definir o conjunto $Par$ como:

$$Par = \{x: x é divisível por 2}$$

Que lemos $Par$ é o conjunto de todos os elementos $x$ tal que $x \text{é divisível por 2}$

Na alvorada do século XX, a matemática navegava em águas turbulentas, enfrentando paradoxos e inconsistências. A necessidade de uma fundação sólida para a matemática era evidente, dando início à jornada da teoria dos conjuntos. E, neste ponto, daremos nosso primeiro salto no tempo. Desta vez, para o futuro, vamos deixar o final do século XIX, pular alguns anos importantes como 1900 e 1903, que visitaremos em breve e focar no ano de 1908. 

Em 1908, [Ernst Zermelo](https://en.wikipedia.org/wiki/Ernst_Zermelo) lançou a primeira pedra, criando uma teoria dos conjuntos para navegar através dos paradoxos que assolavam a matemática da época. Ele introduziu axiomas cruciais, incluindo o controverso axioma da escolha, na tentativa de fornecer um porto seguro para a matemática. Seu objetivo era resolver os paradoxos que haviam sido encontrados na Teoria Ingênua de Cantor. Precisamos entender estes trabalhos porque no tempo em que a leitora lê este trabalho, esta é a teoria que suporta a matemática. No entanto, nossa jornada esta longe de terminar.

### Teoria de Zermelo-Fraenkel

Em 1922, [Abraham Fraenkel](https://en.wikipedia.org/wiki/Abraham_Fraenkel) e [Thoralf Skolem](https://en.wikipedia.org/wiki/Thoralf_Skolem) refinaram e expandiram a teoria de Zermelo, dando origem à teoria de **Zermelo-Fraenkel**, conhecida como **ZF**. Eles decidiram abandonar o axioma da escolha, uma decisão motivada por debates fervorosos sobre sua validade. Matemáticos são criaturas de personalidade forte.

O axioma da escolha não seria esquecido. Ele ressurgiu, demonstrando ser uma ferramenta poderosa para construir conjuntos e provar teoremas que, de outra forma, permaneceriam inalcançáveis. Este axioma, que permite formar um novo conjunto selecionando um elemento de cada conjunto em uma coleção, encontrou seu lugar na teoria de Zermelo-Fraenkel com o axioma da escolha, conhecida como **ZFC**.

A **ZFC**, com o axioma da escolha a bordo, navegou para águas mais profundas, permitindo a prova de teoremas significativos e facilitando a construção de conjuntos complexos. Enquanto isso, a teoria **ZF** continuou sua própria jornada, explorando territórios inexplorados sem o auxílio do axioma da escolha.

A matemática, como um navio resistente, continua sua jornada através dos mares tumultuados, com as teorias ZF e **ZFC** servindo como bússolas confiáveis, guiando os matemáticos em sua busca incessante por compreensão e descoberta. Infelizmente, os matemáticos ainda não têm consistência, ou concordância sobre qual conjunto de axiomas define a Teoria Zermelo-Fraenkel, porém, com certeza ela será sustentada pela beleza dos seguintes axiomas:

1. **Axioma da Extensão**: para todos os conjuntos $A$ e $B$, $A = B$ se e somente se para todo $x$, $x \in A$ se e somente se $x \in B$. O que pode ser expresso com mais formalidade por:
     $$ \forall u(u \in X = u \in Y) \Rightarrow X = Y $$
   - Este axioma ajuda a definir a igualdade de conjuntos de uma forma clara e precisa. É fundamental para evitar ambiguidades e inconsistências ao trabalhar com conjuntos. 
   - **Exemplo**: Se $A = \{1, 2\}$ e $B = \{1, 2\}$, então $A = B$.
   - Sem o axioma da extensão, poderia haver ambiguidade na representação de conjuntos. Por exemplo, os conjuntos $A = \{1, 2, 3\}$ e $B = \{3, 2, 1\}$ poderiam ser considerados diferentes devido à ordem dos elementos. Além disso, sem uma definição clara de igualdade de conjuntos, poderia haver inconsistências ao definir subconjuntos. Se tivermos um conjunto $C = \{1, 2\}$ e um conjunto $D = \{2, 1\}$, algum inocente poderia argumentar que $C$ não é um subconjunto de $D$ devido à diferença na ordem dos elementos. Na ausência do axioma da extensão, operações de conjuntos como união e interseção poderiam resultar em inconsistências. A união de $A = \{1, 2\}$ e $B = \{2, 1\}$ poderia ser questionada e alguns duvidariam se a união resultaria em um conjunto com dois ou quatro elementos.

2. **Axioma do Conjunto Vazio**: existe um conjunto $\emptyset$ tal que para todo $x$, $x \notin \emptyset$.
   - Este axioma introduz o conceito de um conjunto sem elementos, que serve como um bloco de construção fundamental na teoria dos conjuntos. É necessário para definir operações como interseção e diferença de conjuntos.

3. **Axioma do Par**: para todos os conjuntos $A$ e $B$, existe um conjunto $C$ tal que para todo $x$, $x \in C$ se e somente se $x = a$ ou $x = b$. Que podemos formalizar como:
     $$ \forall a \forall b \exists c \forall x(x \in C = (x = a \vee x = b)) $$
   - Este axioma permite a construção de conjuntos a partir de outros conjuntos existentes. Facilita a construção e manipulação de conjuntos em teoremas e definições. Este axioma estabelece que, para qualquer par de conjuntos $A$ e $B$, existe um conjunto $C$ que contém exatamente $A$ e $B$ como seus elementos. Pode ser que $A$ e $B$ sejam iguais, nesse caso, o conjunto $C$ terá um único elemento. Este axioma garante a existência de um conjunto que contém exatamente dois elementos específicos, que podem ser conjuntos eles mesmos. Não importa se os elementos são iguais ou diferentes, o conjunto resultante terá, no máximo, dois elementos distintos.
   - **Exemplo 1**: se $A = \{1\}$ e $B = \{2\}$, então $C = \{\{1\}, \{2\}\}$.
   - **Exemplo 2**: se $A = \{1, 2, 3\} \) e \( B = \{3, 4, 5\}$ então $C = \{ \{1, 2, 3\}, \{3, 4, 5\} \}.

4. **Axioma da União**: para todo conjunto $X$, existe um conjunto $Y$ tal que para todo $x$, $x \in Y$ se e somente se existe um conjunto $Z$ tal que $Z \in X$ e $x \in Z$. Expresso por:
     $$ \forall X \exists Y \forall u(u \in Y = \exists z(z \in X \wedge u \in z)) $$
   - Este axioma facilita a união de conjuntos. É uma operação fundamental na teoria dos conjuntos, permitindo a combinação de conjuntos de uma forma estruturada.
   - **Exemplo**: Se $X = \{\{1\}, \{2\}\}$, então $Y = \{1, 2\}$.

5. **Axioma do Conjunto Potência**: para todo conjunto $X$, existe um conjunto $Y$ tal que para todo $Z$, $Z \in Y$ se e somente se $Z \subseteq X$. Que pode ser escrito formalmente por:
     $$ \forall X \exists Y \forall u(u \in Y = u \subseteq X) $$
   - Este axioma permite a construção de conjuntos de conjuntos. É essencial para explorar as propriedades e estruturas dos conjuntos em um nível mais profundo. Define a existência dos subconjuntos já que garante que para qualquer conjunto $X$, existe um conjunto $Y$ que contém todos os subconjuntos de $X$. Ou seja, $X$ é um subconjunto de $Y$. 
   - **Exemplo**: Se $X = \{1, 2\}$, então $Y = \{\emptyset, \{1\}, \{2\}, \{1, 2\}\}$ é o conjunto que contém todos os subconjuntos de $X$. Observe que $\emptyset$ é um subconjunto de $X$, ainda que ele não tenha sido explicitado.

6. **Axioma da Separação** (ou Especificação): para todo conjunto $X$ e toda propriedade $P(x)$ expressa por uma fórmula, existe um conjunto $Y$ tal que para todo $x$, $x \in Y$ se e somente se $x \in X$ e $P(x)$ é verdadeira. Formalmente teremos:
     $$ \forall X \forall p \exists Y \forall u(u \in Y = (u \in X \wedge \phi(u,p))) $$
   - Este axioma permite a criação de subconjuntos a partir de conjuntos existentes usando uma propriedade específica, explicitada pela fórmula $P(x)$. Facilita a manipulação e análise de conjuntos através da especificação de propriedades desejadas.
   - **Exemplo**: se $X = \{1, 2, 3\}$ e $P(x)$ representa $x > 1$, então $Y = \{2, 3\}$.
  
7. **Axioma da Fundação**: todo conjunto $X$ é bem fundado, o que significa que não existe uma sequência infinita descendente de elementos em $X$. Que pode ser escrito como:
     $$ \forall S[S \neq \emptyset \Rightarrow (\exists x \in S)S \cap x = \emptyset] $$
   Este axioma evita a formação de loops infinitos e estruturas recursivas sem fim dentro dos conjuntos. Este axioma afirma que todo conjunto $X$ é bem fundado, o que significa que não existe uma sequência infinita descendente de elementos em $X$. Para entender melhor, considere que estamos tentando formar uma sequência infinita descendente de conjuntos, onde cada conjunto contém o próximo na sequência, como $x_1 \in x_2 \in x_3 \in \ldots$. O Axioma da Fundação proíbe a existência de tal sequência, garantindo que não podemos ter uma cadeia infinita de conjuntos aninhados dessa forma. Este axioma garante que os conjuntos têm uma _base_ ou _fundação_, prevenindo a formação de conjuntos que não têm um elemento _mínimo_, o que poderia levar a contradições, como o Paradoxo de Russell.
   - **Exemplo**: Não é possível ter uma sequência como $x_1 \in x_2 \in x_3 \in \ldots$ em um conjunto.

8. **Axioma da Infinitude**: existe um conjunto $X$ tal que $\emptyset \in X$ e para todo $x \in X$, $x \cup \{x\} \in X$. Formalmente teremos:
     $$ \exists S[\emptyset \in S \wedge [\forall x \in S](x \cup \{x\} \in S)] $$
   Este axioma introduz a noção de infinitude na teoria dos conjuntos. É a base para a construção de conjuntos infinitos, como os números naturais.
   - **Exemplo**: O conjunto dos números naturais, $\mathbb{N}$, satisfaz este axioma.

9. **Axioma da Escolha**: dada uma coleção de conjuntos não vazios, é possível formar um novo conjunto selecionando um elemento de cada conjunto na coleção. O que formalmente seria expresso por:
        $$ \forall x \in a \exists A(x,y) \Rightarrow \exists y \forall x \in a A(x,y(x)) $$
   Este axioma é fundamental para muitos teoremas e construções na matemática. Permite a seleção de elementos em uma coleção de conjuntos, facilitando a construção de novos conjuntos e provas.
   - **Exemplo**: Dada uma coleção de conjuntos $\{A_1, A_2, \ldots, A_n\}$, é possível formar um conjunto $B = \{x_1, x_2, \ldots, x_n\}$ onde $x_i \in A_i$ para cada $i$.

Neste texto nós usaremos a **ZFC** sempre que for conveniente e, se encontrarmos alguma inconsistência durante a construção deste raciocínio, destacaremos que bússola usaremos para justificar nossa decisão. Dito isso, já começando, para que uma coleção de itens seja um conjunto, ela deve obedecer as seguintes propriedades:

- **Determinação**: os elementos de um conjunto são bem definidos;
- **Não ordenação**: a ordem dos elementos não importa;
- **Eliminação de repetidos**: cada elemento aparece apenas uma vez;

Entre os conceitos mais importantes que estudaremos, devidos a Cantor, destaca-se o conceito de _cardinalidade_, $#$. A cardinalidade do conjunto $A$ que pode ser representada por $#_A$ ou $|A|$ representa o número de elementos em $A$.

Exemplos:
$|\varnothing| = 0$, a cardinalidade do conjunto vazio, $\varnothing$ é zero;
$|\{1\}| = 1$, um conjunto unitário;
$|\{1, 2\}| = 2$, um conjunto com dois elementos do conjunto dos Naturais $\mathbb{N}$.

Podemos fazer operações entre conjuntos. Exemplos importantes de operações em conjuntos incluem: união ($\cup$), interseção ($\cap$), diferença (-), produto cartesiano ($\times$).

### Operações entre Conjuntos

**$\textbf{União}$**: a união de dois conjuntos $A$ e $B$, denotada por $A \cup B$, é o conjunto que contém todos os elementos que estão em $A$ ou em $B$.

$$A \cup B = \{x : x \in A \text{ ou } x \in B\}$$

Lembrando do **Axioma da Extensão**, sua existência elimina qualquer ambiguidade na união dos conjuntos $A = \{1, 2\}$ e $B = \{2, 1\}$. De acordo com este axioma, _dois conjuntos são iguais se e somente se eles têm os mesmos elementos_, independentemente da ordem em que os elementos são apresentados. Portanto, os conjuntos $A$ e $B$ são, na verdade, iguais, já que contêm os mesmos elementos, 1 e 2. Graças a isso, quando aplicamos uma operação de união entre $A$ e $B$, não existe dúvidas de que o resultado seria um conjunto com apenas dois elementos, $\{1, 2\}$.

**$\textbf{Interseção}$**: a interseção de dois conjuntos $A$ e $B$, denotada por $A \cap B$, é o conjunto que contém os elementos que estão simultaneamente em $A$ e em $B$.

$$A \cap B = \{x : x \in A \text{ e } x \in B\}$$

**$\textbf{Diferença}$**: a diferença de dois conjuntos $A$ e $B$, denotada por $A - B$, é o conjunto dos elementos que estão em $A$ mas não estão em $B$.

$$A - B = \{x : x \in A \text{ e } x \notin B\}$$

**$\textbf{Produto Cartesiano}$**: o produto cartesiano de dois conjuntos $A$ e $B$, denotado por $A \times B$, é o conjunto formado por todas as ordens $(a, b)$ tal que $a \in A$ e $b \in B$.

$$A \times B = \{(a,b) : a \in A, b \in B\}$$

### Aritmética a Partir de Conjuntos

Com a cardinalidade de um conjunto em mente seremos capazes de definir algumas operações aritméticas. Por exemplo, partindo do conjunto dos números Naturais, $\mathbb{N}$, poderíamos dizer:

Zero é o vazio. Não tem nada. Nada mesmo. Em conjuntos, isso é representado por $A=\{\}$. Um conjunto com cardinalidade $#=0$. O número $1$ é diferente. Ele contém o zero. Ou melhor, o conjunto vazio. Em conjuntos, isso é $B=\{\{\}\}$.

Agora o jogo começa. Cada novo número é o velho mais ele mesmo. Este é o conceito de sucessão. Assim, dois é um e um. Em conjuntos, isso é $C=\{\{\}, \{\{\}\}\}$. Três é dois e um. Em conjuntos, isso é $\{\{\}, \{\{\}\}, \{\{\}, \{\{\}\}\}$. E podemos representar todos os números naturais apenas com conjuntos.

Cantor nos mostrou o caminho enquanto Zermelo-Fraenkel pavimentaram este caminho para nosso uso. Nesta estrada cada número pode ser representado como o conjunto de todos os conjuntos que estão antes dele. Quatro é três, dois e um. E zero. Sempre com zero. Não esqueçam do zero. Em conjuntos, isso seria $\{\{\}, \{\{\}\}, \{\{\}, \{\{\}\}\}, \{\{\}, \{\{\}\}, \{\{\}, \{\{\}\}\}\}$.

Este caminho não tem fim. Não para. Nunca para. Isso é o que Cantor viu. Há sempre um próximo número. Sempre um conjunto maior. Números naturais são simples. Mas são profundos. Eles começam com nada e vão até o infinito. Tudo graças aos conjuntos.

Eu, pouco humilde e muito audacioso, discordo de Cantor em alguns aspectos, o começo, zero e o fim, infinito. Mas, vou deixar esta discussão para uma outra hora, em outro local, com outros objetivos.

Nós podemos simplificar a representação de Cantor, substituindo os conjuntos de conjuntos que ele viu pelos símbolos que usamos para os números naturais desde que [Fibonacci](https://en.wikipedia.org/wiki/Fibonacci) começou a usá-los. Se fizermos isso, teremos:

1. $0 = \{\}$ (conjunto vazio)
2. $1 = \{0\}$ (o conjunto que contém apenas o conjunto vazio)
3. $2 = \{0, 1\}$
4. $3 = \{0, 1, 2\}$
5. E assim por diante...

E assim sucessivamente. Para todo o sempre. A cardinalidade de cada conjunto representa um número natural, desde que os elementos deste conjunto sejam os números naturais menores, ou iguais ao número natural que queremos representar. Ancorados no porto desta representação podemos vislumbrar os mares infinitos das operações aritméticas básicas. Usando mais Cantor Zermelo-Fraenkel.

### Operações Aritméticas Ingênuas

Quatro operações, e apenas quatro operações servem de guia para toda a aritmética. Para nós, nesta jornada de neófitos, nos bastam estes quatro pilares da matemática.

#### 1. Adição

A adição de dois números naturais $a$ e $b$ é definida pela união dos conjuntos que representam $a$ e $b$, e tomando a cardinalidade do conjunto resultante:

$$a + b = | A \cup B |$$

logo teríamos:

$$2 + 3 = |\{0, 1\} \cup \{0, 1, 2\}| = |\{0, 1, 2\}| = 3$$

#### 2. Subtração

A subtração de dois números naturais $a$ e $b$, desde que $a \gt b$, já que estamos falando de números naturais é definida pela diferença dos conjuntos que representam $a$ e $b$, e tomando a cardinalidade do conjunto resultante:

$$a - b = | A - B |$$

Sendo assim:

$$3 - 2 = |\{0, 1, 2\} - \{0, 1\}| = |\{2\}| = 1$$

#### 3. Multiplicação

A multiplicação de dois números naturais $a$ e $b$ é definida através da operação de adição repetida:

$$a \times b = a + a + \ldots + a \, (\text{b vezes})$$

O que nos levaria a:

$$2 \times 3 = 2 + 2 + 2 = |\{0, 1\} \cup \{0, 1\} \cup \{0, 1\}| = |\{0, 1\}| = 2$$

#### 4. Divisão

A divisão de dois números naturais $a$ e $b$, desde que $b \neq 0$, porque a divisão por zero não é possível, é definida como o número de vezes que $b$ pode ser subtraído de $a$ até que o resultado seja menor que $b$:

$$a \div b = q \, (\text{se } a = q \times b + r \, \text{ e } r < b)$$

Desta forma teríamos:

$$6 \div 2 = 3 \, (\text{porque } 6 = 3 \times 2 + 0)$$

Esta foi uma viagem tranquila, nosso navio ficou atado ao porto da ingenuidade. Conceitos foram apresentados sem a formalidade matemática. Navios ficam seguros nos portos mas, não foi para isso que eles foram construídos. Para realmente enfrentar nossa jornada teremos que navegar segundo Zermelo-Fraenkel.

Nesta jornada vamos voltar a definição dos números naturais com a sintaxe definida por Zermelo-Fraenkel.

1. $0 = \{\}$ (conjunto vazio)
2. $1 = \{0\} = \{\{\}\}$
3. $2 = \{0, 1\} = \{\{\}, \{\{\}\}\}$
4. $3 = \{0, 1, 2\} = \{\{\}, \{\{\}\}, \{\{\}, \{\{\}\}\}\}$
5. E assim por diante...

A ideia é que o número natural $n$ é o conjunto de todos os números naturais menores que $n$. E deste ponto podemos derivar as operações aritméticas novamente, com um pouco mais de rigor. Esta viagem será mais tranquila, independente do mar que navegarmos se tivermos a bussola correta. Neste caso, a função sucessor

A função sucessor, denotada por $S(n)$, é a função que atribui a cada número natural $n$ o próximo número natural. Utilizando os conjuntos como definimos anteriormente, a função sucessor pode ser definida como:

$$S(n) = n \cup \{n\}$$

O que nos permite criar uma sequência de exemplos:

1. $S(0) = 0 \cup \{0\} = \{\} \cup \{\{\}\} = \{\{\}\} = 1$
2. $S(1) = 1 \cup \{1\} = \{\{\}\} \cup \{\{\{\}\}\} = \{\{\}, \{\{\}\}\} = 2$
3. $S(2) = 2 \cup \{2\} = \{\{\}, \{\{\}\}\} \cup \{\{\{\}, \{\{\}\}\}\} = \{\{\}, \{\{\}\}, \{\{\}, \{\{\}\}\} = 3$
4. E assim por diante...

A função sucessor é a função que permite demonstrar a criação dos números naturais com o rigor da matemática, a partir da teoria dos conjuntos. E agora, podemos definir as operações aritméticas novamente, desta feita, com um pouco mais de rigor matemático.

#### Operações Aritméticas com um Pouco de Rigor Matemático

Vamos ficar com as quatro operações que vimos antes. Isso há de permitir a comparação entre as duas formas que escolhemos para a definição de operações.

##### Adição

A adição pode ser definida recursivamente:

1. $a + 0 = a$
2. $a + S(b) = S(a + b)$

Onde $S(b)$ representa o sucessor de $b$, que é $b \cup \{b\}$.

##### Multiplicação

A multiplicação também pode ser definida recursivamente:

1. $a \times 0 = 0$
2. $a \times S(b) = a \times b + a$

##### Subtração

A subtração é um pouco mais complexa, porque requer a introdução do conceito de predecessor:

1. $a - 0 = a$
2. $a - S(b) = Pred(a - b)$

Onde $Pred(b)$ é o predecessor de $b$, que é o maior número natural menor que $b$.

##### Divisão

A divisão pode ser definida usando uma abordagem de subtração repetida:

1. $a \div b = q$ se $a = q \times b + r$ e $0 \leq r < b$

Essas definições garantem um tratamento formal e rigoroso das operações aritméticas na teoria dos conjuntos.

### Fechamento

Antes de progredirmos, um outro conceito bate a nossa porta com a força daqueles que sabem seu lugar no Universo, um conceito que será fundamental para o entendimento das estruturas algébricas que marcam nosso caminho até os monads. O conceito de fechamento.

Aqui, neste humilde texto, ficaremos restritos ao conceito de fechamento que relaciona os elementos de um conjunto a uma operação específica e deixaremos o fechamento topológico para um outro dia chuvoso em que nos calhe a sina de escrever.

Dizemos que um conjunto é fechado sob uma operação especifica se, ao aplicarmos esta operação aos elementos do conjunto encontramos outro elemento do mesmo conjunto.  

Formalmente diremos que um conjunto $A$ é dito fechado sob uma operação binária $\circ$ se, para todo $x, y \in A$, o resultado da operação $x \circ y$  também pertence a $A$. Fazendo uso da linguagem da matemática podemos dizer que:

$$\forall x, y \in A, \: x \circ y \in A$$

Alguns exemplos são necessários:

- **Adição**: O conjunto dos números inteiros,  $\mathbb{Z}$,  é fechado sob a operação de adição, pois a soma de dois números inteiros é sempre um número inteiro. O mesmo pode ser dito sobre a subtração. Mas o conjunto dos números inteiros,  $\mathbb{Z}$, será aberto sob a divisão. Consegue entender isso?

- **Multiplicação**: O conjunto dos números naturais,  $\mathbb{N}$, é fechado sob a operação de multiplicação, pois o produto de dois números naturais é sempre um número natural. E, novamente, este conjunto será aberto, sob a divisão e sob a subtração.

O fechamento é uma característica importante dos conjuntos que os relacionam as operações que podem ser realizadas sobre os seus elementos. E possui suas próprias características ]interessantes.

**Fechamento de Subconjuntos**: Se um subconjunto $B$ de $A$ é fechado sob uma operação definida em $A$, então $B$ é um subconjunto fechado de $A$.

**Fechamento com Relação a Composição de Funções**: Se temos duas funções $f: A \rightarrow B$ e $g: B \rightarrow C$, a composição $g \circ f: A \rightarrow C$ é uma operação fechada em $A$.

Esta última propriedade há de fazer sentido quando revermos funções. Mas, antes, relações.

  There is no friend as loyal as a book.

## Relações

> There is no friend as loyal as a book. Ernest Hemingway

Hemingway atribui aos elementos do conjunto livros, uma relação com o ser humano, mais intensa que a relação de amizade que podemos ter com outro ser humano. Uma citação para provocar reflexão e consideração, mas, ainda assim, uma relação.

Com isso entramos no reino das relações, o trazemos para a fria teoria dos conjuntos. lembramo-nos de que, como os livros na citação de Hemingway, cada elemento num conjunto possui uma _lealdade_ característica, uma relação, com outros elementos dentro de uma estrutura matemática formal. Uma relação que cabe a você, criador, ou observador, de um determinado conjunto determinar ou observar. Como Hemingway fez com seus amigos e livros.

Na teoria dos conjuntos de Zermelo-Fraenkel, uma relação é definida como um conjunto de pares ordenados. A relação deve ser vista como uma conexão entre elementos de dois conjuntos. Ainda que o segundo destes conjuntos seja o próprio primeiro conjunto. E, neste caso, onde relacionamos os conjuntos com elementos do mesmo conjunto, teremos uma relação endógina.

Independentemente dos conjuntos envolvidos, podemos definir uma relação $R$ de um conjunto $A$ para um conjunto $B$ (denotado por $R \subseteq A \times B$) como um subconjunto resultado do produto cartesiano de $A$ e $B$. A descrição formal de uma relação $R$ pode ser expressa da seguinte forma:

**Definição Formal**: a relação $R$ de $A$ em $B$ é um conjunto de pares ordenados onde o primeiro elemento do par pertence ao conjunto $A$ e o segundo elemento do par pertence ao conjunto $B$:

$$R = \{ (a, b) | a \in A, b \in B \}$$

Relações entre elementos de conjuntos possuem propriedades específicas que nos permitem entender a relação em si. Dependendo das características específicas dos pares ordenados em uma relação, uma relação poderá ser:

- **Reflexiva**: Uma relação $R$ em $A$ é dita reflexiva se, para todo $a \in A$, $(a, a) \in R$.
- **Simétrica**: Uma relação $R$ em $A$ é dita simétrica se, para todo $(a, b) \in R$, também temos $(b, a) \in R$.
- **Anti-Simétrica**: Uma relação $R$ em $A$ é dita anti-simétrica se, para todo $(a, b) \in R$ e $(b, a) \in R$, temos que $a = b$. Em outras palavras, se dois elementos diferentes são relacionados um com o outro, eles não podem ser relacionados na ordem inversa.
- **Transitiva**: Uma relação $R$ em $A$ é dita transitiva se, para todo $(a, b) \in R$ e $(b, c) \in R$, também temos $(a, c) \in R$.

Com estas relações abrimos um caminho que nos leva a categorizar relações. A nós, por enquanto, nesta jornada, interessam:

1. **Relação de Equivalência**: Uma relação $\sim$ em um conjunto $A$ é chamada de relação de equivalência se satisfaz as seguintes propriedades:
   - **Reflexiva**: Para todo $a \in A$, temos $a \sim a$.
   - **Simétrica**: Para todos $a, b \in A$, se $a \sim b$, então $b \sim a$.
   - **Transitiva**: Para todos $a, b, c \in A$, se $a \sim b$ e $b \sim c$, então $a \sim c$.

   **Exemplo**: Considere o conjunto de todas as linhas no plano. Definimos uma relação de equivalência $\sim$ tal que, para quaisquer duas linhas $l_1$ e $l_2$, $l_1 \sim l_2$ se e somente se $l_1$ é paralela a $l_2$. Esta relação é reflexiva (uma linha é paralela a si mesma), simétrica (se $l_1$ é paralela a $l_2$, então $l_2$ é paralela a $l_1$) e transitiva (se $l_1$ é paralela a $l_2$ e $l_2$ é paralela a $l_3$, então $l_1$ é paralela a $l_3$).

2. **Relação de Ordem Total**: Uma relação $\leq$ em um conjunto $A$ é chamada de relação de ordem total quando satisfaz as seguintes propriedades:
   - **Reflexiva**: Para todo $a \in A$, temos $a \leq a$.
   - **Antissimétrica**: Para todos $a, b \in A$, se $a \leq b$ e $b \leq a$, então $a = b$.
   - **Transitiva**: Para todos $a, b, c \in A$, se $a \leq b$ e $b \leq c$, então $a \leq c$.
   - **Totalidade**: Para todos $a, b \in A$, ou $a \leq b$ ou $b \leq a$.

   **Exemplo**: A relação "menor ou igual a" ($\leq$) em $\mathbb{R}$ é uma relação de ordem total.

3. **Relação de Isomorfismo**: Na álgebra abstrata, uma relação $\cong$ entre duas estruturas algébricas é chamada de isomorfismo se existe uma bijeção $f$ entre elas que preserva as operações (isto é, a estrutura) definidas nas estruturas.

   **Exemplo 1**: Considere dois grupos $(G, \cdot)$ e $(H, \ast)$. Uma função $f: G \to H$ é um isomorfismo de grupos se:
   - Para todos $a, b \in G$, $f(a \cdot b) = f(a) \ast f(b)$.
   - $f$ é uma bijeção.

   **Exemplo 2**: No contexto de grafos, dois grafos $G$ e $H$ são isomorfos se existe uma bijeção $f: V(G) \to V(H)$ tal que dois vértices $u$ e $v$ são adjacentes em $G$ se, e somente se, $f(u)$ e $f(v)$ são adjacentes em $H$.

Estas definições formais e exemplos ilustram a profundidade das relações na exploração e compreensão de estruturas matemáticas complexas, como os conjuntos, que nos guiaram até aqui. Contudo, quando aplicamos uma relação a um elemento de um determinado conjunto, estamos, na verdade, aplicando uma função a este elemento. Podemos dizer que uma função de $A$ para $B$ é uma _relação especial_ que satisfaça a propriedade funcional, ou seja, para cada $a \in A$, existe exatamente um $b \in B$ tal que $(a, b) \in R$.

Eu sei que já usamos funções ao logo deste texto. E agora, finalmente, chegou a hora de falarmos sobre elas.

## Funções

Em um dia ensolarado, imagine uma jornada tranquila de um ponto a outro, dentro de seu bairro, talvez para comprar pão. Uma jornada onde cada passo é guiado por um propósito claro, um destino definido. No universo da matemática, funções agem como esses passos conscientes, guiando-nos de um valor a outro, com precisão e determinação. Uma função é uma forma de aplicar uma relação entre um ponto inicial e um objetivo.

Podemos conceber as funções matemáticas como entidades singulares, destiladas até sua essência mais pura, interligando pontos em uma paisagem vasta e muitas vezes inexplorada. Nesta paisagem os pontos, são elementos de conjuntos e a função e a aplicação de uma relação entre estes elementos.

As funções funcionam como sentenças concisas que nos permitem narrar histórias complexas com clareza e profundidade. Uma entidade que nos permite tecer conexões significativas entre elementos distintos em um conjunto, abrindo portas para explorações mais profundas no reino da matemática.

Na matemática, uma **função** é uma relação entre dois conjuntos que associa cada elemento do primeiro conjunto (_domínio_) a exatamente um elemento do segundo conjunto (_contra-domínio_). Formalmente, uma função $f$ de um conjunto $A$ para um conjunto $B$, que denotaremos como $f: A \rightarrow B$ pode ser definida como:

**Definição de Função**: uma função $f: A \rightarrow B$ é um conjunto de pares ordenados $(a, b)$, onde $a \in A$ e $b \in B$, tal que para cada $a \in A$, existe exatamente um $b \in B$ tal que $(a, b) \in f$. Formalmente,

$$f = \{ (a, b) | a \in A, b \in B \}$$

Uma função precisa obedecer um conjunto de propriedades:

- **Bem Definida**: Para cada $a \in A$, existe um único $b \in B$ tal que $(a, b) \in f$.

- **Imagem**: A imagem de $a$ sob $f$ é o único elemento $b$ em $B$ tal que $(a, b) \in f$, denotado por $f(a) = b$.

Talvez alguns exemplos ajudem a jogar luz sobre as trevas da dúvida:

- **Função Identidade**: $f: A \rightarrow A$ definida por $f(a) = a$ para todo $a \in A$.

- **Função Constante**: $f: A \rightarrow B$ definida por $f(a) = c$ para todo $a \in A$, onde $c$ é um elemento fixo de $B$.

Na nossa jornada, se olharmos com atenção na direção das funções encontraremos diferentes classes de funções que delineiam de forma distinta as relações entre os conjuntos. Cada classe de função carrega consigo características próprias e significativas.

### Classificações das Funções

1. **Função Injetora (ou Injeção)**:

   Uma função $f: A \rightarrow B$ é dita _injetora_ se elementos distintos do conjunto $A$ são levados a elementos distintos do conjunto $B$. Formalmente,

   $$\forall a_1, a_2 \in A, \, (a_1 \neq a_2) \Rightarrow (f(a_1) \neq f(a_2))$$

   Isso significa que elementos distintos do conjunto de domínio são mapeados para elementos distintos do conjunto contra-domínio. No gráfico de uma função _injetora_, não existem duas linhas horizontais que intersectam o gráfico em dois pontos diferentes.

   **Exemplo**: A função $f: \mathbb{R} \rightarrow \mathbb{R}$ dada por $f(x) = 2x + 1$ é uma injeção.

2. **Função Sobrejetora (ou Sobrejeção)**:

   Uma função $f: A \rightarrow B$ é chamada _sobrejetora_ se a imagem da função é igual ao contra-domínio, isto é, cada elemento de $B$ é imagem de pelo menos um elemento de $A$. Formalmente,

   $$\forall b \in B, \, \exists a \in A \, \text{ tal que } \, f(a) = b$$

   O que significa que todo elemento do conjunto contra-domínio é imagem de pelo menos um elemento do conjunto de domínio. No gráfico de uma função _sobrejetora_, cada valor de $y$ no contra-domínio é atingido pelo gráfico da função.

   **Exemplo**: A função $f: \mathbb{R} \rightarrow \mathbb{R}$ dada por $f(x) = x^3$ é uma sobrejeção.

3. **Função Bijetora (ou Bijecção)**:

   Uma função é dita _bijetora_ se é ao mesmo tempo _injetora_ e _sobrejetora_. Formalmente,

   $$\forall b \in B, \, \exists! a \in A \, \text{ tal que } \, f(a) = b$$

   Significando que cada elemento do conjunto de domínio é mapeado para um elemento único no conjunto contra-domínio e vice-versa. No gráfico de uma função _bijetora_, cada linha horizontal e vertical intersecta o gráfico em exatamente um ponto.

   **Exemplo**: A função $f: \mathbb{R} \rightarrow \mathbb{R}$ dada por $f(x) = x$ é uma bijeção.

4. **Função Constante**:

   Como mencionado anteriormente, uma função constante é aquela que associa todos os elementos de $A$ a um único elemento de $B$.

   **Exemplo**: A função $f: \mathbb{R} \rightarrow \mathbb{R}$ dada por $f(x) = c$, onde $c$ é uma constante, é uma função constante.

## Um Ponto Importante na nossa Jornada

Em um determinado momento da história, no alvorecer do século XX, quando ainda se ouviam retumbantes as vozes do século XIX, a comunidade matemática enfrentava uma crise nas suas mais profundas fundações. As fissuras na edificação da matemática clássica começaram a surgir, revelando paradoxos que ameaçavam a estabilidade e consistência de teorias consolidadas.

Na teoria dos conjuntos de Cantor, e em outros pontos, encontramos paradoxos cruéis espreitando nas esquinas das teorias. Vários caminhos novos surgiram diretamente deste momento histórico entre centenas de trabalhos vamos destacar dois. Simplesmente porque seus trabalhos interessam aos temas que abordo neste segundo semestre de 2023:

- **[Gottlob Frege](https://en.wikipedia.org/wiki/Gottlob_Frege)**: uma das mentes mais influentes da época, lançou as bases da lógica moderna, mas também se aventurou em criar uma linguagem formal robusta que pudesse servir como alicerce para toda a matemática. Frege concebeu a linguagem da Lógica de Primeira Ordem, enriquecida com quantificadores e predicados, uma ferramenta que viria a ser central na formalização das teorias matemáticas. Um mergulho mais profundo neste tópico pode ser encontrado em [neste artigo](https://frankalcantara.com/introducao-programacao-logica/), onde exploramos a influência e o legado de Frege na programação lógica.

- **[Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell)**: contemporâneo de Frege, Russell foi outra figura central nesta revolução conceitual. Em sua busca por uma fundação mais sólida para a matemática, Russell propôs a teoria dos tipos, um sistema que buscava resolver os paradoxos surgidos da teoria dos conjuntos, estabelecendo uma hierarquia de tipos para evitar auto-referências problemáticas. Seu trabalho, profundamente interligado com a lógica e a filosofia, abriu novos caminhos para entender e estruturar formalismos matemáticos. Estamos considerando a possibilidade de explorar a teoria dos tipos em mais detalhes em um futuro texto, que poderá ser encontrado [aqui](https://frankalcantara.com/2023-08-21-a-teoria-dos-tipos-e-o-cálculo-lambda-a-simplicidade-e-o-segredo-da-felicidade/) e dependendo de para onde os ventos da investigação nos levem. Em algum momento, ainda neste texto exploremos um pouco da Teoria dos Tipos. Que soprem os ventos!

Por ora, ancoramos na Álgebra Abstrata. O próximo porto: _magmas_.

## Magmas

Cuidadosa e meticulosamente, como um capitão que leva seu barco pelas rotas que deseja, conduzi nossa jornada por meio de operações binárias. Pura maldade, sabendo meu destino não quis permitir desvios que o levassem a mares revoltos e tornassem nossa jornada mais longa. É bela a matemática quando esta está domada. E, ainda com as operações binárias a nossa frente uma nova estrutura algébrica desponta no horizonte. Os _magmas_.

No âmbito da álgebra abstrata, os _magmas_ surgem como uma resposta natural à necessidade de estudar estruturas com operações binária, sem exigir propriedades restritivas demais, proporcionando assim um terreno fértil para a investigação das propriedades mais fundamentais das operações binárias.

Embora seja desafiador apontar um momento histórico exato para a emergência do conceito de _magma_, podemos situar a sua solidificação como conceito matemático no início do século XX. Aquele momento turbulento durante o qual a matemática estava se formalizando e diversificando rapidamente. Os matemáticos estavam à procura de estruturas que pudessem generalizar e unificar diversos fenômenos matemáticos, e a introdução de conceitos como grupos, anéis e corpos, na álgebra havia pavimentado o caminho para a definição de estruturas mais gerais, como os _magmas_.

Um _magma_ é um conjunto com uma operação. Como uma fazenda com uma máquina. A máquina pega dois elementos do conjunto e produz um novo elemento. Por exemplo, em um conjunto de inteiros a operação pode ser a adição. A máquina adição pega 2 e 3. Produz 5.

### Por que os magmas?

O conceito de _magma_ permite uma abordagem ampla e geral das operações binárias, sendo apenas necessário que haja uma operação fechada definida no conjunto. Esta generalidade nos concede a liberdade de explorar e entender uma variedade vasta de operações sem sermos restringidos por propriedades adicionais. No estudo de estruturas mais complexas, às vezes é benéfico despir as propriedades adicionais e estudar a estrutura nua, um _magma_, para ganhar entendimento sobre as propriedades fundamentais que governam essa estrutura.

A nomenclatura _magma_, de origem grega _μάγμα_, em álgebra abstrata é creditada ao matemático francês [Nicolas Bourbaki](https://en.wikipedia.org/wiki/Nicolas_Bourbaki), um pseudônimo coletivo usado por um grupo de matemáticos franceses no século XX. Existem textos que relacionam o _magma_ a uma estrutura chamada de _groupid_. Os dois termos se referem a tupla formada por um conjunto e a operação binária sob a qual este conjunto é fechado. Com vantagem para o temo _magma_ que só possui esta conotação sobre o termo _groupid_ que tem outros significados na álgebra.

Existem ainda os que acreditam que o temo tenha sido escolhido porque, em Francês, idioma materno do Grupo Bourbaki, a palavra _magma tem uma relação com algo que se move de forma desordenada, ou com uma pilha de bagunça_.Bagunça pode ser pura, mas nunca será simples.

>"Deixe $E$ ser um conjunto. Uma função $f$ de $E x E$ em $E$ é chamada de _lei de composição em $E$_. O valor $f(x, y)$ de $f$ para um par ordenado $(x, y)$ [pertencente a] $E \times E$ é chamado de composição de $x$ e $y$ sob essa lei. Um conjunto com uma **lei de composição é chamado de magma**". [^1]

Não tenho certeza quanto os motivos da escolha deste termo, [e parece que ninguém tem](https://english.stackexchange.com/questions/63210/etymology-of-magma-in-abstract-algebra). Este professorzinho de subúrbio, tende a concordar que este termo, _magma_, tenha sido escolhido por representar uma metáfora com o _magma_ geológico. Se não por precisão metafórica ao menos pela beleza da alegoria.

Semelhante ao _magma_ geológico, que é uma forma primordial e fundida de rocha, a estrutura de _magma_ na álgebra abstrata representa uma estrutura _não refinada_, possuindo apenas a operação binária mais básica sem a imposição de outras propriedades estruturais. Podemos estender esta analogia, destacando que, assim como o _magma_ geológico tem o potencial de se solidificar em várias formas rochosas complexas, a estrutura de _magma_ na álgebra pode ser _solidificada_ com a adição de propriedades adicionais para formar estruturas algebraicas mais complexas e estruturadas, como semigrupos, grupos, e anéis.

Veremos a utilidade do estudo e aplicação de _magmas_ fora da álgebra pura, quando aplicarmos a álgebra a:

1. **Teoria de Grafos**: algumas operações definidas sobre conjuntos de grafos podem formar uma estrutura de composição que pode ser melhor entendida com _magma_. Isso pode ser útil na análise de propriedades de grafos e na construção de novas classes de grafos.

2. **Ciência da Computação**: no campo da ciência da computação, a teoria de autômatos e linguagens formais muitas vezes utiliza conceitos de álgebra abstrata. Aqui, os _magmas_ podem surgir no estudo de operações sobre strings ou outras estruturas de dados.

3. **Física**: especialmente na teoria quântica de campos, as estruturas algébricas podem desempenhar um papel significativo. Os _magmas_ podem surgir como uma forma de explorar operações binárias em espaços de estados físicos ou em outras estruturas encontradas na física teórica.

4. **Matemática Discreta**: os _magmas_ podem ser usados para estudar operações binárias em conjuntos finitos, fornecendo insights sobre a estrutura e propriedades desses conjuntos.

### Magmas Segundo a Formalidade Matemática

Sem a formalidade a matemática perde sua beleza e caminha próxima a inutilidade. Assim, precismos aplicar um pouco de formalidade ao conceito de _magma_.

**Definição**: um **_magma_** é uma tupla formada por um conjunto não vazio $M$ fechado sob uma operação binária $\cdot : M \times M \to M$. Ou seja, para quaisquer dois elementos $a, b \in M$, $a \cdot b$ também pertence a $M$.

#### Propriedades e Características

Os _magmas_ não necessitam satisfazer nenhuma propriedade particular além do fechamento, o que os torna estruturas bastante gerais e amplas. No entanto, podem ser investigadas propriedades adicionais que um _magma_ pode possuir, tais como:

1. **Associatividade**: Um _magma_ é dito associativo se, para todos $a, b, c \in M$, temos que $(a \cdot b) \cdot c = a \cdot (b \cdot c)$.
2. **Comutatividade**: Um _magma_ é dito comutativo se, para todos $a, b \in M$, temos que $a \cdot b = b \cdot a$.

#### Exemplos de _magmas_

1. **Conjunto de inteiros com a operação de adição**:

   Considere o conjunto de inteiros $\mathbb{Z}$ e a operação de adição $+$. A tupla $(\mathbb{Z}, +)$ forma um _magma_, já que a adição de dois inteiros quaisquer resulta em outro inteiro.

   **Propriedades**:
   - Associativo: Sim, pois $(a+b)+c = a+(b+c)$, para todos $a, b, c \in \mathbb{Z}$.
   - Comutativo: Sim, pois $a+b = b+a$, para todos $a, b \in \mathbb{Z}$.

2. **Conjunto de matrizes $n \times n$ com a operação de multiplicação**:

   Considere o conjunto de todas as matrizes $n \times n$ sobre um campo $F$, denotado por $M_n(F)$, e a operação de multiplicação de matrizes. $(M_n(F), \cdot)$ forma um _magma_, pois o produto de duas matrizes quaisquer $n \times n$ resulta em outra matriz $n \times n$.

   **Propriedades**:
   - Associativo: Sim, pois $(A \cdot B) \cdot C = A \cdot (B \cdot C)$, para todas $A, B, C \in M_n(F)$.
   - Comutativo: Não, em geral a multiplicação de matrizes não é comutativa, ou seja, $A \cdot B \neq B \cdot A$ para algumas matrizes $A, B \in M_n(F)$.

#### Contra Exemplos: Estruturas que Não Formam Magmas

1. **Conjunto dos números naturais com a operação de subtração**:

   Considere o conjunto dos números naturais $\mathbb{N}$ e a operação de subtração $-$. Esta estrutura não forma um _magma_, pois a subtração de dois números naturais nem sempre resulta em um número natural. Esta operação pode resultar em um número negativo, e números negativos não pertencem ao conjunto dos números naturais.

   **Contra Exemplo**:
   - $5 - 7 = -2$, e $-2 \notin \mathbb{N}$, demonstrando que o conjunto não é fechado sob a operação de subtração.

2. **Conjunto das matrizes $2 \times 2$ com uma operação não comutativa e não associativa**:

   Considere o conjunto de todas as matrizes $2 \times 2$ sobre os números reais, e defina uma nova operação binária $\star$ da seguinte forma:

   $$ A \star B = AB - BA $$

   Onde $A$ e $B$ são matrizes $2 \times 2$ e $AB$ e $BA$ representam a multiplicação usual de matrizes. Esta operação não é nem comutativa nem associativa, o que pode ser demonstrado com exemplos específicos de matrizes.

   **Contra Exemplo**:

   Considere as matrizes $2 \times 2$, $A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ e $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

   - **Não Comutativa**:
     $$A \star B = AB - BA \neq BA - AB = B \star A$$

   - **Não Associativa**:
     Para demonstrar que a operação não é associativa, podemos introduzir uma terceira matriz, $C = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, e verificar que
     $$ (A \star B) \star C \neq A \star (B \star C)$$

Verdades seja dita. Tal qual o magma geológico não permeia o dia a dia do homem comum, a chance é grande de que a amável leitora não se depare com este termo em territórios além da matemática pura. Escasso, direto, sem adornos. E mesmo na linda aridez da matemática este conceito serve apenas de alicerce para a definição de leis que o entendimento de operações entre elementos de um mesmo conjunto e a criação de novas estruturas.

   ---
     [^1]: BOURBAKI, N. Elementos de Matemática, Álgebra I, Capítulos 1-3. Paris: Hermann; Massachusetts: Addison-Wesley, 1974. p. 1.
