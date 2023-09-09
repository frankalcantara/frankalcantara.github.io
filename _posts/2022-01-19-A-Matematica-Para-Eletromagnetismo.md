---
layout: post
title: "A Fórmula da Atração: a Matemática do Eletromagnetismo"
author: Frank
categories:
  - Matemática
  - Eletromagnetismo
tags:
  - Matemática
  - Física
  - Eletromagnetismo
image: assets/images/eletromag1.jpg
description: Entenda como a matemática fundamenta o eletromagnetismo e suas aplicações práticas em um artigo acadêmico destinado a estudantes de ciência e engenharia.
slug: formula-da-atracao-matematica-eletromagnetismo
keywords:
  - cálculo vetorial
  - Eletromagnetismo
  - Matemática
  - poesia
  - álgebra vetorial
rating: 5
---

O Eletromagnetismo é a lei, o ordenamento. Como uma divindade antiga que rege a existência e os movimentos do nosso universo. Duas forças, elétrica e magnética, em uma dança interminável, moldam tudo, de um grão de poeira a um oceano de estrelas, até o mesmo dispositivo que você usa para decifrar essas palavras deve sua existência e funcionamento ao Eletromagnetismo.

Imagem de [Asimina Nteliou](https://pixabay.com/users/asimina-1229333/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2773167) de [Pixabay](https://pixabay.com//?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2773167)

> "Nesta longa vida eu aprendi que toda a nossa ciência se comparada com a realidade é primitiva e infantil. Ainda assim, **é a coisa mais preciosa que temos**." Albert Einstein

- [Álgebra Linear](#álgebra-linear)
  - [Vetores, os compassos de tudo que há e haverá](#vetores-os-compassos-de-tudo-que-há-e-haverá)
    - [Vetores Unitários](#vetores-unitários)
  - [Multiplicação por Escalar](#multiplicação-por-escalar)
  - [Vetor Oposto](#vetor-oposto)
  - [Adição e Subtração de Vetores](#adição-e-subtração-de-vetores)
  - [Vetores Posição e Distância](#vetores-posição-e-distância)
  - [Produto Escalar](#produto-escalar)
  - [Produto Vetorial](#produto-vetorial)
    - [A Identidade de Jacobi](#a-identidade-de-jacobi)
  - [Usando a Álgebra Vetorial no Eletromagnetismo](#usando-a-álgebra-vetorial-no-eletromagnetismo)
    - [Lei de Coulomb](#lei-de-coulomb)
- [Cálculo Vetorial](#cálculo-vetorial)
  - [Campos Vetoriais](#campos-vetoriais)
  - [Gradiente](#gradiente)
    - [Significado do Gradiente](#significado-do-gradiente)
    - [Propriedades do Gradiente](#propriedades-do-gradiente)
  - [Divergência](#divergência)
    - [Fluxo e a Lei de Gauss](#fluxo-e-a-lei-de-gauss)
    - [Teorema da Divergência](#teorema-da-divergência)
    - [Propriedades da Divergência](#propriedades-da-divergência)

Trataremos de linhas de força invisíveis que se entrelaçam, tangenciam e interferem umas nas outras, formando o tecido do Cosmos e o fluxo da vida, tão real quanto a terra sob os pés ou o ar que respiramos, e como este último, completamente invisíveis.

O estudo do Eletromagnetismo será uma batalha própria, individual, dura. É a esperança lançar luz sobre o desconhecido, descobrir as regras que governam a vida e o universo, e então aproveitar essas regras para criar, para progredir, para sobreviver. Não é para os fracos de coração, nem para aqueles que buscam respostas fáceis. É para aqueles que não temem o desconhecido, para os que se levantam diante do abismo do desconhecido e dizem: "eu irei entender". É um desafio, uma luta, um chamado. E, como em qualquer luta, há perdas, há dor, mas também há vitórias, triunfos e, no final de tudo, compreensão. Esta é uma guerra que começou a milhares de anos, ainda que estejamos interessados apenas nas últimas batalhas, as mais recentes.

Quando o século XIX caminhava para seu final, um homem, [James Clerk Maxwell](https://en.wikipedia.org/wiki/James_Clerk_Maxwell), orquestrou as danças dos campos elétrico e do magnético em uma sinfonia de equações. Desenhando na tela do universo, Maxwell delineou a interação dessas forças com o espaço e a matéria. Sua obra, extraordinária em todos os aspectos, destaca-se pela simplicidade refinada e pela beleza lírica. Um balé de números, símbolos e equações que desliza pela folha, fluido e elegante como um rio.

Mas essa beleza, essa simplicidade, não é acessível a todos. Ela é um jardim murado, reservado àqueles que conquistaram o direito de entrar através de estudo e compreensão. Sem o conhecimento apropriado, seja da física que fundamenta o universo ou da matemática que o descreve, as equações de Maxwell são como flores de pedra: frias, inalteráveis, sem vida. Com esse entendimento, no entanto, elas florescem em cores e formas maravilhosas, vivas e palpitantes com significado.

É aqui que embarcamos na nossa jornada, uma exploração através desse jardim de pedra e sombra, para encontrar a beleza escondida nele. Neste artigo, nosso foco estará na matemática que compõe essas equações, o esqueleto que sustenta a carne e o sangue da física. Não estaremos preocupados com as aplicações práticas ou físicas dessas equações. Essas são preocupações para outro momento, outra jornada. Aqui, nosso interesse está no abstrato, no puro, na dança dos números e símbolos que compõem as equações do eletromagnetismo.

Considere este texto como a liberação da toa, o início da sua jornada em um mar de conhecimento. Uma viagem em busca do conhecimento mais estruturante do Universo. Talvez você chegue lá, talvez não.

Lágrimas de decepção não o encontrarão em cada porto. Mesmo que não chegue ao destino desejado. Cada porto de entendimento lhe trará a luz do conhecimento ao final você será uma pessoa diferente. São mares revoltos, não será fácil. Nada que vale a pena é! E, como diria [o poeta](https://en.wikipedia.org/wiki/Fernando_Pessoa):

> "...Tudo vale a pena Se a alma não é pequena...". Fernando Pessoa.

# Álgebra Linear

Área da matemática envolvida com o espaço, vetores e seu baile atemporal, ritmado por regras intrínsecas. Vetores e Matrizes, soldados organizados em linhas e colunas, cada um contando histórias de variáveis e transformações. Divergências, gradientes e rotacionais, gestos majestosos na dança do cálculo vetorial. Tudo tão complexo quanto a vida, tão real quanto a morte, tão honesto quanto o mar, profundo, impiedoso e direto. 

>O mar bravo só respeita rei. [Arnaud Rodrigues / Chico Anísio](https://www.letras.com/baiano-os-novos-caetanos/1272051/)

O espaço será definido por vetores, cheio de mistério e beleza. A Análise vetorial será a bússola do navegante, guiando-o através do vasto oceano do desconhecido. A cada dia, a cada cálculo, desvendaremos um pouco mais desse infinito, mapearemos um pouco mais desse oceano de números, direções, sentidos e valores, entendemos um pouco mais de como o Universo dança ao som da álgebra linear e da análise vetorial.

## Vetores, os compassos de tudo que há e haverá

Vetores, feixes silenciosos de informação, conduzem o entendimento além do simples tamanho. São como bússolas com uma medida, apontando com determinação e direção para desvendar os segredos das grandezas que precisam mais do que só a magnitude. Vetores, abstrações matemáticas que usamos para entender as gradezas que precisam de direção e sentido além da pura magnitude. Usamos os vetores para ir além das grandezas escalares.

As grandezas escalares, aquelas que podem ser medidas como medimos a massa de um peixe, o tempo que demora para o sol se pôr ou a velocidade de um veleiro cortando a paisagem em linha quase reta. Cada uma é um número único, uma quantidade, um fato em si mesmo contendo todo o conhecimento necessário para seu entendimento.

São as contadoras de histórias silenciosas do mundo, falando de tamanho, de quantidade, de intensidade. E, como um bom whisky, sua força reside na simplicidade. Ainda assim as grandezas escalares que oferecem uma medida da verdade.

As grandezas vetoriais, por outro lado são complexas, diversas e intrigantes. Vetores são abstrações matemáticas usadas para entender estas grandezas vetoriais, guerreiras da direção e do sentido. Navegam o mar da matemática com uma clareza de propósito que vai além da mera magnitude. Elas possuem uma seta, uma bússola, que indica para onde se mover. Sobrepujam o valor em si com uma direção, um sentido, uma indicação, uma seta.

A seta, uma extensão de seu ser, representa sua sua orientação. Aponta o caminho para a verdade, mostrando não apenas o quanto, mas também o onde. Seu indica sua magnitude, o quanto, sua essência. Assim vetores escondem intensidade, direção e sentido em uma única entidade, fugaz e intrigante.

As grandezas vetoriais são como o vento, cuja direção e força você sente, mas cuja essência não se pode segurar. Elas são como o rio, cujo fluxo e direção moldam a paisagem. São essenciais para entender o mundo em movimento, o mundo de forças, velocidades e acelerações. Elas dançam nas equações do eletromagnetismo, desenham os padrões da física e guiam os marinheiros na imensidão do desconhecido. No mar da compreensão, grandezas vetoriais são a bússola e o vento, dando não apenas escala, mas também orientação e sentido à nossa busca pelo conhecimento. Como é belo o idioma de Machado de Assis, mas, de tempos em tempos, temos que recorrer as imagens.

![Três vetores no plano cartesiano](/assets/images/vetorPlano1.jpeg){:class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 1 - Três vetores aleatórios no plano $(x,y$).</legend>

Toda esta poesia pode ser resumida na geometria de uma seta com origem e destino em um espaço multidimensional contendo informações de direção, sentido e intensidade. Três setas, três vetores, $A$, $B$ e $C$, em um plano. Nesta jornada, não seremos limitados pela frieza da geometria. Buscamos a grandeza da álgebra. Na álgebra vetores são representados por operações entre outros vetores.

Na física moderna usamos os vetores como definido por [Dirac](https://en.wikipedia.org/wiki/Paul_Dirac) (1902-1984), que chamamos de Vetores Ket, ou simplesmente ket. Não aqui, pelo menos não por enquanto. Aqui utilizaremos a representação vetorial como definida por [Willard Gibbs](https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs) (1839–1903) no final do Século XIX. Adequada ao estudo clássico do Eletromagnetismo. O estudo das forças que tecem campos vetoriais que abraçam a própria estrutura do Universo. Invisíveis porém implacáveis.

Entender esses campos, então, é uma forma de começar a entender o universo. É ler a história que está sendo escrita nas linhas invisíveis de força. É mergulhar no mar profundo do desconhecido, e emergir com um conhecimento novo e precioso. É se tornar um tradutor da linguagem cósmica, um leitor das marcas deixadas pelas forças em seus campos. É, em resumo, a essência da ciência. E é essa ciência, esse estudo dos campos e das forças que neles atuam, que iremos explorar.

Para lançar as pedras fundamentais do nosso conhecimento representaremos os vetores por meio de letras latinas maiúsculas $A, B, C, ...$ frias. Estes vetores são elementos de um espaço vetorial $\textbf{V}$, também representado por letras latinas, desta feita em negrito. Nossos espaços vetoriais serão sempre representados em três dimensões. O espaço que procuramos é o nosso, o espaço onde vivemos, a forma como percebemos o universo e assim ficaremos limitados a três dimensões.

Não é qualquer espaço, é um espaço específico, limitado à realidade e limitante das operações que podemos fazer para defini-lo assim, nosso estudo se fará a partir de um espaço vetorial que satisfaça às seguintes condições:

1. o espaço vetorial $\textbf{V}$ seja fechado em relação a adição. Isso quer dizer que para cada par de vetores $A$ e $B$ pertencentes a $\textbf{V}$ existe um, e somente um, vetor $C$ que representa a soma de $A$ e $B$ e que também pertence ao espaço vetorial $\textbf{V}$, dizemos que: $\exists A \in \textbf{V} \wedge \exists B \in \textbf V \therefore \exists A+B=C \in \textbf V$;

2. a adição seja associativa: $(A+B)+C = A+(B+C)$;

3. existe um vetor zero: a adição deste vetor zero a qualquer vetor $A$ resulta no próprio vetor $A$, inalterado, imutável. De tal forma que: $\forall A \in \textbf V \space \space \exists \wedge 0 \in \space \textbf V \space \therefore \space  0+A=A$;

4. existe um vetor negativo $-A$ de forma que a soma de um vetor com seu vetor negativo resulta no vetor zero. Tal que: $\exists -A \in V \space \space \vert \space \space -A+A=0 $;

5. o espaço vetorial $V$ seja fechado em relação a multiplicação por um escalar, um valor sem direção ou sentido, de tal forma que para todo e qualquer elemento $c$ do conjunto dos números complexos $\mathbb{C}$ multiplicado por um vetor $A$ do espaço vetorial $\textbf{V}$ existe um, e somente um vetor $cA$ que também pertence ao espaço vetorial $\textbf{V}$. Tal que: $\exists  \space c \in \mathbb{C} \space \space \wedge \space \space \exists  \space A \in V \space \space \therefore \space \space \exists  \space cA \in \textbf{V}$;

6. Existe um escalar neutro 1: tal que a multiplicação de qualquer vetor $A$ por $1$ resulta em $A$. Ou seja: $\exists \space 1 \in \mathbb{R} \space \space \wedge \space \space \exists  \space A \in \textbf{V} \space \space \vert \space \space 1A = A$;

É preciso manter a atenção voltada para a verdade da hierarquia que rege o mundo dos conjuntos. O conjunto dos números reais $\mathbb{R}$ é um subconjunto do conjunto dos números imaginários $\mathbb{C}=\{a+bi \space \space a.b \in \mathbb{R}\}$. Esta relação contenção determina que o conjunto $\mathbb{R}$, o conjunto dos números reais, se visto de forma mais abrangente, representa de forma concisa, todos os números imaginários cuja parte imaginária é igual a zero. Se usarmos a linguagem da matemática dizemos que $\mathbb{R}=\{a+bi \space \space \vert \space \space a.b \in \mathbb{R} \wedge b=0\}$.

A representação algébrica dos vetores definida por [Willard Gibbs](https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs) (1839–1903) indica que um vetor em um espaço $\textbf{V}$ qualquer é, pura e simplesmente, o resultado de operações realizadas entre os vetores que definem os componentes deste espaço vetorial.

Nosso espaço $\textbf{V}$ terá três dimensões então precisamos escolher um conjunto de coordenadas, que definam os pontos deste espaço e usar estes pontos para definir os componentes vetoriais que usaremos para definir todos os vetores do espaço $\textbf{V}$.

Maxwell, seguindo os passos de [Newton](https://en.wikipedia.org/wiki/Isaac_Newton), também se apoiou nos ombros de gigantes. E eis que em nossa jornada nos defrontamos com um destes gigantes. Em meados do Século XVII, [René Descartes](https://plato.stanford.edu/entries/descartes/) criou um sistema de coordenadas para definir o espaço que conhecemos que prevaleceu contra o tempo e a evolução e até hoje leva seu nome latino: Sistema de Coordenadas Cartesianas.

No caso do Sistema de Coordenadas Cartesianas, o espaço será limitado por três eixos, perpendiculares e ortogonais e pelos valores das coordenadas $(x,y.z)$ colocadas sobre estes eixos. Do ponto de vista da Álgebra Vetorial, para cada um destes eixos teremos um vetor de comprimento unitário. São estes vetores, que chamamos de vetores unitários e identificamos por $(a_x, a_y, a_z)$ respectivamente, Sendo vetores unitários eles têm magnitude $1$ e estão orientados segundo os eixos cartesianos $(x,y,z)$.

A mágica acontece quando dizemos que todos os vetores do espaço vetorial $\textbf{V}$ podem ser representados por somas dos vetores unitários $(a_x, a_y, a_z)$ quando multiplicados independentemente por fatores escalares. De fato, qualquer vetor no espaço será o produto de um vetor unitário por um escalar.

### Vetores Unitários

Um vetor $B$ qualquer tem magnitude, direção e sentido. A magnitude, também chamada de intensidade, ou módulo, será representada por $\vert B \vert$. Definiremos um vetor unitário $a$ na direção $B$ por $a_B$ de tal forma que:

$$ a_B=\frac{B}{|B|} $$

Um vetor unitário $a_B$ é um vetor que tem a mesma direção e sentido de $B$ com magnitude $1$ logo o módulo, ou magnitude, ou ainda comprimento de $a_b$ será representado por:

$$\vert a_B \vert=1$$

Agora que conhecemos os vetores unitários podemos entender a ciência, quase mágica, que sustenta a Álgebra Vetorial e faz com que todos os conceitos geométricos que suportam a existência de vetores possam ser representados algebricamente em um espaço, desde que este espaço seja algebricamente definido.

Em um sistema de coordenadas tridimensionais e ortogonais podemos expressar qualquer vetor na forma da soma dos seus componentes unitários ortogonais. Qualquer vetor, independente da sua direção, sentido, ou magnitude pode ser representado pela soma dos os vetores unitários que representam as direções, eixos e coordenadas, do sistema de coordenadas escolhido. A cada fator desta soma damos o nome de componente vetorial, ou simplesmente componente. Existirá um componente para cada dimensão do sistema de coordenadas e estes componentes são relativos ao sistema de coordenadas que escolhermos para representar o espaço $\textbf{V}$.

Como somos marinheiros de primeira viagem, navegamos de dia, em mares conhecidos mantendo a terra a vista. Neste caso, começaremos com o Sistema de Coordenadas Cartesianas. Um sistema de coordenadas conhecido, seguro e fácil de representar. Não será difícil visualizar um espaço vetorial definido neste sistema já que é o espaço em que vivemos. A sala de sua casa tem uma largura $x$, um comprimento $y$ e uma altura $z$. No Sistema de Coordenadas Cartesianas a representação de um vetor $B$ qualquer, segundo seus componentes unitários e ortogonais será dada por:

$$B=b_xa_x+b_ya_y+b_za_z$$

Nesta representação, $b_x$, $b_y$, $b_z$ representam os fatores escalares que devemos usar para multiplicar os vetores unitários $a_x$, $a_y$, $a_z$ de forma que a soma destes vetores represente o vetor $B$ no espaço $\Bbb{R}^3$.

Ao longo deste artigo vamos chamar $b_x$, $b_y$, $b_z$ de componentes vetoriais nas direções $x$, $y$, $z$, ou de projeções de $B$ nos eixos $x$, $y$, $z$. A prova da equivalência entre os componentes e as projeções sobre os eixos pertence ao domínio da geometria que ficou no porto na hora em que começamos esta viagem.

A beleza e a simplicidade do Sistema de Coordenadas Cartesianas é também a sua maldição. Em muitos problemas o uso deste sistema torna a matemática desnecessariamente torturante. Neste caso podemos recorrer a qualquer outro sistema de coordenadas. Com a única condição de termos três dimensões ortogonais entre si. Por exemplo, poderíamos definir nosso vetor $B$ como:

$$B=b_xa_x+b_ya_y+b_za_z$$

$$B=b_ra_r+b_\phi a_\phi+b_za_z$$

$$B=b_ra_r+b_\phi a_\phi+b_\theta a_\theta$$

Respectivamente para os Sistemas de Coordenadas Cartesianas, Cilíndricas e Esféricas. Sistemas de coordenadas diferentes para o mesmo espaço. Há que existir uma forma de garantir que o vetor $B$ tenha sempre a mesma magnitude, a mesma direção e o mesmo sentido não importando o sistema de coordenadas escolhido em um dado momento. Existe, não tenha dúvidas quanto a isso. Se for necessário, mergulharemos nestas técnicas de conversão entre sistemas.

O mar, e a matemática, são imprevisíveis. Não é raro que, uma vez que o sistema de coordenadas tenha sido definido, para um determinado conjunto de problemas, os vetores sejam representados apenas por seus componentes vetoriais, se qualquer referência aos vetores unitários. Assim, no Sistema de Coordenadas Cartesianas o vetor $B=3a_x+a_y-a_z$ pode ser representado apenas por $B=(3,1, -1)$.

Quando representamos um vetor por seus componentes ortogonais, podemos calcular sua magnitude utilizando os fatores escalares multiplicadores de cada vetor unitário.

Dado o vetor $B=b_xa_x+b_ya_y+b_za_z$ sua magnitude será dada por $\vert B \vert=\sqrt{b_x^2+b_y^2+b_z^2}$. Desta forma poderemos encontrar o vetor unitário $a_B$ de $B$ por:

$$a_B=\frac{B}{ \vert B \vert }= \frac{b_xa_x+b_ya_y+b_za_z}{ \sqrt{b_x^2+b_y^2+b_z^2}}$$

Talvez toda essa rigidez da matemática desvaneça diante dos seus olhos com um exemplo.

<p class="exp">
<b>Exemplo 1:</b> calcule o vetor unitário $a_A$ do vetor $A=a_x-3a_y+2a_z$. <br><br>
<b>Solução:</b>

\[a_A=\frac{a_xa_x+a_ya_y+a_za_z}{\sqrt{a_x^2+a_y^2+a_z^2}}\]

\[a_A=\frac{a_x-3a_y+2a_z}{\sqrt{1^2+(-3)^2+2^2}}=\frac{a_x-3a_y+2a_z}{3,7416}\]

\[a_A=0,2672a_x-0,8018a_y+0,5345a_z\]

</p>
## Multiplicação por Escalar

Um escalar é um número, um valor, frio, simples e direto. A informação contida no escalar não precisa de direção e sentido. São escalares todos os números reais $(\mathbb{R})$, inteiros $(\mathbb{Z})$ e naturais $(\mathbb{N})$. Os números complexos $(\mathbb{C})$ precisam de um pouco mais de atenção.

Os números complexos, $\mathbb{C}$ contém informações que podem ser associadas a direção e sentido mas, não são vetores. São como peixes em um lago. A parte real é como a distância que o peixe nada para leste ou oeste. A parte imaginária é o quanto ele nada para norte ou sul. Eles podem mover-se em duas direções, mas não são como o vento ou um rio, que têm uma direção e um sentido claros. Os números complexos, eles são mais como os peixes - nadam por aí, sem preocupação com a direção. Eles são escalares, não vetores.

Todavia, tal como um pescador marca a posição de um peixe pelo quão longe está da margem e em que ângulo, podemos fazer o mesmo com números complexos. Chamamos de magnitude a distância até a origem e ângulo é a direção que aponta para eles. Ainda assim, não confunda isso com a direção e o sentido de um vetor na física. É uma comparação, nada mais.

É importante entender que números complexos, $\mathbb{C}$, possuem um conceito relacionado a magnitude e fase, ângulo na representação polar, em que um número complexo $c$ pode ser representado como $r*e^{iθ}$, onde $r$ é a magnitude (ou o módulo) do número complexo, e $θ$ é a fase (ou o argumento), que pode ser pensada como a direção do número complexo no plano complexo. Mas, novamente, o conceito de direção usado aqui não é o mesmo conceito de direção quando nos referimos a vetores. É apenas uma analogia matemática. Sim! A matemática tem analogias. Voltaremos aos números complexos quando for conveniente. Vamos apenas guardar em nossa caixa de ferramentas a noção de que um número é um escalar. Uma informação de valor e apenas isso.

A multiplicação de um vetor $B$ por um escalar implica na multiplicação de cada um dos componentes $b$ desse vetor por este escalar. Os escalares que usaremos, neste artigo, serão elementos do conjunto dos números reais $\Bbb{R}$. Sem esquecer que, como vimos antes, os elementos dos conjunto dos números reais $\Bbb{R}$ são definidos como elementos do conjunto dos números complexos $\Bbb{C}$ a mesma definição que utilizamos quando explicitamos as regras de formação do espaço vetorial $\textbf{ V}$ que definimos para nossos vetores.

A multiplicação de cada componente por um escalar e simples e quase não requer um exemplo. Quase.

<p class="exp">
<b>Exemplo 2:</b> considere o vetor $V=2a_x+4a_y-a_z$ e calcule $3,3V$ e $V/2$: <br><br>
<b>Solução:</b>

\[3.3V=(3,3)(2)a_x+(3,3)(4)a_y+(3,3)(-1)a_z\]

\[3.3V=6,6a_x+13,2a_y-3,3a_z\]

\[\frac{V}{2}=(\frac{1}{2})(2)a_x+(\frac{1}{2})(4)a_y+(\frac{1}{2})(-1)a_z\]

\[\frac{V}{2}=a_x+2a_y-\frac{1}{2}a_z\]

</p>

A multiplicação por escalar é comutativa, associativa, distributiva e fechada em relação ao zero e ao elemento neutro. Se tivermos os escalares $m$ e $n$ e os vetores $A$ e $B$, as propriedades da multiplicação por um escalar serão dadas por:

1. **comutatividade:** a ordem dos fatores não afeta o produto. Portanto, se você multiplicar um vetor por um escalar, receberá o mesmo resultado, independentemente da ordem. Ou seja, $m(A) = (A)m$.

2. **associatividade:** a forma como os fatores são agrupados não afeta o produto. Portanto, se você multiplicar um vetor por um produto de escalares, receberá o mesmo resultado, independentemente de como os fatores são agrupados. Ou seja, $(mn)A = m(nA)$.

3. **distributividade:** a multiplicação por escalar é distributiva em relação à adição de vetores e de escalares. Portanto, se você multiplicar a soma de dois vetores por um escalar, o resultado será o mesmo que se você multiplicar cada vetor pelo escalar e somar os resultados. Ou seja, $m(A + B)=mA + mB$. Da mesma forma, se você multiplicar um vetor pela soma de dois escalares, o resultado será o mesmo que se você multiplicar o vetor por cada escalar e somar os resultados. Ou seja, $(m + n)A = mA + nA$.

4. **Fechada em relação ao zero e ao elemento neutro:** Multiplicar qualquer vetor por zero resulta no vetor zero. Ou seja, $0A = 0$. E multiplicar qualquer vetor por $1$ (o elemento neutro da multiplicação escalar) resulta no mesmo vetor. Ou seja, $1A = A$.

<p class="exp">
<b>Em resumo:</b> <br><br>
\[mA=Am\]

\[m(nA) = (mn)A\]

\[m(A+B) = mA+mB\]

\[(A+B)n = nA+nB\]

\[1A=A\]

\[0A=0\]

</p>
## Vetor Oposto

A multiplicação de um vetor pelo escalar $-1$ é especial.
Chamamos de Vetor Oposto ao vetor $A$ ao vetor que tem a mesma intensidade, a mesma direção e sentido oposto ao sentido de $A$. Um Vetor Oposto é o resultado da multiplicação de um vetor pelo escalar $-1$. Logo:

$$-1A = -A$$

## Adição e Subtração de Vetores

Olhe para os pássaros no céu. Os vetores são como o rastro de um pássaro no céu, mostrando não apenas quão longe voou, mas também a direção que escolheu. Representam forças, esses ventos invisíveis que movem o mundo, que também são assim. Eles têm tamanho e direção, forças são vetores no universo da Álgebra Linear.

Como os pássaros no céu, os vetores também podem se juntar, ou se afastar. A soma, a subtração, fazem parte do seu voo. Alguns podem achar útil imaginar isso, recorrendo a geometria, como um paralelogramo, uma forma com lados paralelos que mostra como um vetor soma ao outro.

![Soma de Vetores com a Regra do Paralelogramo](/assets/images/SomaVetores.jpeg){:class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 2 - Regra do Paralelogramo - Soma geométrica de Vetores.</legend>

Eu não vou lhe guiar em um passeio pelo mundo das formas e linhas, não aqui, não agora. Mas lembre-se que a geometria, embora silenciosa e imóvel, sempre está lá, embaixo de tudo, o esqueleto invisível que dá forma ao nosso mundo.

A matemática irascível, nos força a dizer que o espaço vetorial $\textbf{V}$ é fechado em relação a soma de vetores. A soma de vetores é feita componente a componente, paulatinamente. Assim, se considerarmos os vetores $A$ e $B$ poderemos encontrar um vetor $C$ que seja a soma de $A$ e $B$ representada por $C=A+B$ por:

$$C=A+B=(A_x a_x+A_y a_y+A_z a_z)+(B_x a_x+B_y a_y+B_z a_z)$$

$$C=A+B=(A_x+B_x)a_x+(A_y+B_y)a_y+(A_y+B_y)a_z$$

<p class="exp">
<b>Exemplo 3:</b> se $A=5a_x-3a_y+a_z$ e $B=a_x+4a_y-7a_z$. Calcule $C=A+B$. <br><br>
<b>Solução</b>

\[C=A+B=(5a_x-3a_y+a_z)+(1a_x+4a_y-7a_z)\]

\[C=A+B=(5+1)a_x+(-3+4)a_y+(1-7)a_z \]

\[C= 6a_x+a_y-6a_z\]

</p>

A Subtração será uma soma em que o segundo operando será o vetor oposto do operando original. Assim:

$$C=A-B=A+(-B)=A+(-1B)$$

Talvez um exemplo torne a subtração óbvia.

<p class="exp">
<b>Exemplo 4:</b> considere $A=5a_x-3a_y+a_z$ e $B=1a_x+4a_y-7a_z$ e calcule $C=A-B$. <br><br>
<b>Solução:</b>

\[C=A-B=(5a_x-3a_y+a_z)+(-1(1a_x+4a_y-7a_z))\]

\[C=A-B=(5a_x-3a_y+a_z)+(-1a_x-4a_y+7a_z)\]

\[C=A-B=4a_x-7a_y+8a_z\]

</p>

A adição e subtração de vetores obedecem a certas propriedades matemáticas. Se considerarmos os vetores $A$, $B$ e $C$, e o escalar $m$, temos:

1. **comutatividade da adição de vetores:** a ordem dos vetores na adição não afeta o resultado final. Portanto, $A + B = B + A$. A subtração, entretanto, não é comutativa, ou seja, $A - B ≠ B - A$. A comutatividade é como uma dança onde a ordem dos parceiros não importa.

2. **associatividade da adição de vetores:** a forma como os vetores são agrupados na adição não afeta o resultado final. Assim, $(A + B) + C = A + (B + C)$. A associatividade é como um grupo de amigos que se reúne. Não importa a ordem de chegada o resultado é uma festa. A subtração, entretanto, não é associativa, ou seja, $(A - B) - C ≠ A - (B - C)$.

3. **Distributividade da multiplicação por escalar em relação à adição de vetores:** Se você multiplicar a soma de dois vetores por um escalar, o resultado será o mesmo que se você multiplicar cada vetor pelo escalar e somar os resultados. Isto é, $m*(A + B) = mA + mB$.

Essas propriedades são fundamentais para a manipulação de vetores em muitos campos da física e da matemática.

<p class="exp">
<b>Em Resumo:</b> <br><br>
\[A+B=B+A\]

\[A+(B+C)=(A+B)+C\]

\[m(A+B)=mA+mC\]

<b>Importante:</b> a subtração não é comutativa nem associativa.

</p>
## Vetores Posição e Distância

Um vetor posição, também conhecido como vetor ponto, é uma ferramenta útil para descrever a posição de um ponto no espaço em relação a um ponto de referência (geralmente a origem do sistema de coordenadas). Como uma flecha que começa na origem, o coração do sistema de coordenadas, onde $x$, $y$, e $z$ são todos zero, $(0,0,0)$, e termina em um ponto $P$ no espaço. Este ponto $P$ tem suas próprias coordenadas - digamos, $x$, $y$, e $z$.

O vetor posição $R$ que vai da origem até este ponto $P$ será representado por $R_P$. Se as coordenadas de $P$ são $(x, y, z)$, então o vetor posição $R_P$ será:

$$R_p = xa_x + ya_y + za_z$$

O que temos aprendido, na nossa jornada, até o momento, sobre vetores é simplesmente uma forma diferente de olhar para a mesma coisa. Soma de vetores unitários, $a_x$, $a_y$, $a_z$, que define um vetor em qualquer direção que escolhemos de uma maneira diferente, define o vetor posição, a seta que parte do zero - a origem - e se estende até qualquer ponto no espaço. Está tudo conectado, cada parte fazendo sentido à luz da outra. Assim, aprendemos a entender o espaço ao nosso redor, uma vetor de cada vez. Não podemos nos limitar a origem como ponto de partida de todos os vetores. Entre dois pontos quaisquer no espaço, $P$ e $Q$ é possível traçar um vetor. Um vetor que chamaremos de vetor distância $D$.

Dois pontos no espaço, $P$ e $Q$, são como dois pontos num mapa. Cada um tem seu próprio vetor posição - seu próprio caminho da origem, o centro do mapa, até onde eles estão. Chamamos esses caminhos de $R_P$ e $R_Q$. Linhas retas que partem da origem, o centro do mapa e chegam a $P$ e $Q$. Usando para definir estes pontos os vetores posição a partir da origem.

Agora, você quer encontrar a distância entre $P$ e $Q$, não o caminho do centro do mapa até $P$ ou $Q$, mas o caminho direto de $P$ até $Q$. Este caminho será o vetor distância $D$.

Como encontramos $D$? Subtraímos. $$D$$ será a diferença entre $R_Q$ e $R_P$. É como pegar o caminho de $Q$ ao centro do mapa e subtrair o caminho de $P$ ao centro do mapa. O que sobra é o caminho de $P$ até $Q$.

$$D = R_Q - R_P$$

$D$, a distância entre $P$ e $Q$, será geometricamente representado por uma seta apontando de $P$ para $Q$. O comprimento dessa seta é a distância entre $P$ e $Q$. É um conceito simples, mas poderoso. Uma maneira de conectar dois pontos em um espaço, uma forma de enxergar todo espaço. Definindo qualquer vetor a partir dos vetores posição da sua própria origem e destino.

<p class="exp">
<b>Exemplo: 5</b> considerando que $P$ esteja no ponto $(3,2,-1)$ e $Q$ esteja no ponto $(1,-2,3)$. Logo, o vetor distância $D_{PQ}$ será dado por: <br><br>
<b>Solução:</b>
\[D_{PQ} = R_P - R_Q \]

Logo:

\[D_{PQ} = (P_x-Q_x)a_x + (P_y-Q_y)a_y+(P_z-Q_z)a_z \]

\[D_{PQ} = (3-1)a_x+(3-(-2))a_y+((-1)-3)a_z \]

\[D_{PQ} = 2a_x+5a_y-4a_z \]

</p>

<p class="exp">
<b>Exemplo 6:</b> dados os pontos $P_1(4,4,3)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$. (a) Especifique o vetor $A$ que se estende da origem até o ponto $P_1$. (b) Determine um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos $P_1$ e $P_2$. (c) Calcule o perímetro do triângulo formado pelos pontos $P_1$, $P_2$ e $P_3$.
<br><br>
<b>Solução:</b><br>
<b>(a)</b> o vetor $A$ será o vetor posição do ponto $P_1(4,3,2)$ dado por:

\[A = 4a_x+4a_y+3a_z\]

<b>(b)</b> para determinar um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos $P_1$ e $P_2$ precisamos primeiro encontrar este ponto médio $P_M$. Então:

\[P_M=\frac{P_1+P_2}{2} =\frac{(4,4,3)+(-2,0,5)}{2}\]

\[P_M=\frac{(2,4,8)}{2} = (1, 2, 4)\]

\[P_M=a_x+2a_y+4a_z\]

Para calcular o vetor unitário na direção do vetor $P_M$ teremos:

\[a\_{P_M}=\frac{(1, 2, 4)}{|(1, 2, 4)|} = \frac{(1, 3, 4)}{\sqrt{1^2+2^2+4^2}}\]

\[a\_{P_M}=0.22a_x+0.45a_y+0.87a_z\]

<b>(c)</b> finalmente, para calcular o perímetro do triângulo formado por: $P_1(4,4,3)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$, precisaremos somar os módulos dos vetores distância ente $P_1(4,3,2)$ e $P_2(-2,0,5)$, $P_2(-2,0,5)$ e $P_3(7,-2,1)$ e $P_3(7,-2,1)$ e $P_1(4,3,2)$.

\[ \vert P_1P_2 \vert = \vert (4,4,3)-(-2,0,5) \vert = \vert (6,4,-2) \vert \]

\[ \vert P_1P_2 \vert = \sqrt{6^2+4^2+2^2}=7,48\]

\[ \vert P_2P_3 \vert = \vert (-2,0,5)-(7,-2,1) \vert = \vert (-9,2,-4) \vert \]

\[ \vert P_2P_3 \vert = \sqrt{9^2+2^2+4^2}=10,05\]

\[ \vert P_3P_1 \vert = \vert (7,-2,1)-(4,4,3) \vert = \vert (3,-6,-2) \vert \]

\[ \vert P_3P_1 \vert = \sqrt{3^2+6^2+6^2}=7 \]

Sendo assim o perímetro será:

\[ \vert P_1P_2 \vert + \vert P_2P_3 \vert + \vert P_3P_1 \vert =7,48+10,05+7=24.53 \]

</p>

## Produto Escalar

Há um jeito de juntar dois vetores - setas no espaço - e obter algo diferente: um número, algo simples, sem direção. Isso é o Produto Escalar. **O resultado do produto escalar entre dois vetores é um valor escalar**.

O Produto Escalar opera dois vetores e resulta em um número. No nosso espaço vetorial $\textbf {V}$ um número real. Esse número tem algo especial: ele não se mexe. Não importa como você vire ou gire o espaço, o número permanece o mesmo. Ele é invariante.

Na física, chamamos esses números de escalares. São quantidades que não precisam saber para onde estão apontando. Elas apenas são. Um exemplo? A massa de um objeto. Não importa como você vire ou gire o objeto, sua massa permanece a mesma.

Aqui está o segredo: as partes de um vetor são escalares. Se você quebrar um vetor em seus componentes unitário - cada uma apontando ao longo de um eixo - essas partes mudam quando você gira o espaço. Elas são sensíveis, se moldam e se adaptam. Elas não são invariáveis. O Produto Escalar é simples e invariável, mas essencial para entender o espaço e como as coisas se movem dentro dele.

Usando a linguagem da matemática, direta e linda, dizemos que dados os vetores $A$ e $B$, **o produto escalar entre $A$ e $B$ resultará em uma quantidade escalar** e será representado por $A\cdot B$. Trigonometricamente o produto escalar entre $A$ e $B$ será dado por:

$$A\cdot B = |A||B|cos(\theta_{AB})$$

E esta será a equação analítica, trigonométrica, geométrica, do Produto Escalar. Observe que o Produto Escalar corresponde a projeção do vetor $A$ em $B$. Se não estiver vendo esta projeção deve voltar a geometria, e mergulhar na trigonometria básica, para iluminar este conhecimento. Assim, na prosa de [Camões](https://en.wikipedia.org/wiki/Lu%C3%ADs_de_Cam%C3%B5es) dizemos que o Produto Escalar entre dois vetores $A$ e $B$ é o produto entre o produto das magnitudes destes vetores e o cosseno do menor ângulo entre eles.

Vetores são como flechas atiradas no vazio do espaço. E como flechas, podem seguir diferentes caminhos.

Alguns vetores correm paralelos, como flechas lançadas lado a lado, nunca se encontrando. Eles seguem a mesma direção, compartilham o mesmo curso, mas nunca se cruzam. Sua jornada é sempre paralela, sempre ao lado. O ângulo entre eles é $zero$ seu cosseno será então $1$.

Outros vetores são transversais, como flechas que cortam o espaço em ângulos retos, ângulos de $90^0$. Eles não seguem a mesma direção, nem o mesmo caminho. Eles se interceptam, mas em ângulos precisos, limpos, cortando o espaço como uma grade. O cosseno entre estes vetores é $0$.

E ainda há aqueles vetores que se cruzam em qualquer ângulo, como flechas lançadas de pontos distintos, cruzando o espaço de formas únicas. Eles podem se encontrar, cruzar caminhos em um único ponto, ou talvez nunca se encontrem. Eles desenham no espaço uma dança de possibilidades, um balé de encontros e desencontros. Aqui, o cosseno não pode ser determinado antes de conhecermos os vetores.

Assim, como flechas no espaço, vetores desenham caminhos - paralelos, transversais ou se cruzando em qualquer ângulo. Eles são a linguagem do espaço, a escrita das distâncias e direções. Eles são os contadores de histórias do espaço tridimensional.

A matemática destila estes conceitos simplesmente como: se temos um vetor $A$ e um vetor $B$ teremos o Produto Escalar entre eles dado por:

$$A\cdot B = A_x B_x+A_y B_y+A_z B_z$$

E saímos da equação analítica, geométrica, e voltamos aos mares tranquilos e conhecidos da Álgebra Linear. Talvez um exemplo ajude a acender a luz do entendimento.

<p class="exp">
<b>Exemplo 7:</b> dados os vetores $A=3a_x+4a_y+a_z$ e $B=a_x+2a_y-5a_z$ encontre o ângulo $\theta$ entre $A$ e $B$.
<br><br>
<b>Solução:</b><br>
Para calcular o ângulo vamos usar a equação analítica, ou trigonométrica, do Produto Escalar:

\[A\cdot B =|A||B|cos(\theta)\]

Logo vamos precisar dos módulos dos vetores e do Produto Escalar entre eles. Sendo assim:

\[A\cdot B = (3,4,1)\cdot(1,2,-5) \]

\[A\cdot B = (3)(1)+(4)(2)+(1)(-5)=6\]

Calculando os módulos de $A$ e $B$, teremos:

\[ \vert A \vert = \vert (3,4,1) \vert =\sqrt{3^2+4^2+1^2}=5,1\]

\[ \vert B \vert = \vert (1,2,-5) \vert =\sqrt{1^2+2^2+5^2}=5,48\]

Com o Produto Escalar e os módulos dos vetores podemos aplicar nossa equação analítica:

\[ A\cdot B =|A||B|cos(\theta)\] logo:

\[ 6 =(5,1)(5,48)cos(\theta) \therefore cos(\theta) = \frac{6}{27,95}=0,2147 \]

\[ \theta = arccos(0,2147)=77,6^0 \]

</p>

Até agora, estivemos estudando um espaço de três dimensões, traçando vetores que se estendem em comprimento, largura e altura no Espaço Cartesiano. Isso serve para algumas coisas, como o estudo do eletromagnetismo, da dança de forças e campos que tecem o tecido do nosso mundo físico. Mas nem sempre é o bastante.

A verdade é que o universo é mais complexo do que as três dimensões que podemos tocar e ver. Há mundos além deste, mundos que não podemos ver, não podemos tocar, mas podemos imaginar. Para esses mundos, precisamos de mais. Muito mais.

Álgebra vetorial é a ferramenta que usamos para desenhar estes mundos. Com ela, podemos expandir nosso pensamento para além das três dimensões, para espaços de muitas dimensões. Espaços que são mais estranhos, mais complicados, mas também mais ricos em possibilidades.

Então, talvez seja hora de reescrever nossa definição de Produto Vetorial, a hora de expandir horizontes. Não apenas para o espaço tridimensional, mas para todos os espaços que podem existir. Isso é o que a álgebra vetorial permite. Isso é o que a álgebra vetorial é: uma linguagem para desenhar mundos, de três dimensões ou mais.

Generalizando o produto escalar entre dois vetores $A$ e $B$ com $N$ dimensões teremos:

$$A\cdot B = \sum\limits_{i=1}\limits^{N} a_ib_i$$

Onde $i$ é o número de dimensões. Assim, se $i =3$ e chamamos estas dimensões $x$, $y$, $z$ respectivamente para $i=1$, $i=2$ e $i=3$ teremos:

$$A\cdot B = \sum\limits_{i=1}\limits^{3} a_ib_i = a_1b_1 +a_2b_2 + a_3b_3 $$

Ou, substituindo os nomes das dimensões:

$$A\cdot B = a_xb_x +a_yb_y + a_zb_z $$

1. **Comutatividade:** o Produto Escalar tem uma beleza simples quase rítmica. Como a batida de um tambor ou o toque de um sino, ele se mantém o mesmo não importa a ordem. Troque os vetores - a seta de $A$ para $B$ ou a flecha de $B$ para $A$ - e você obtém o mesmo número, o mesmo escalar. Isso é o que significa ser comutativo. Ou seja: $A\cdot B = B\cdot A$

2. **Distributividade:** o Produto Escalar também é como um rio dividindo-se em afluentes. Você pode distribuí-lo, espalhá-lo, dividir um vetor por muitos. Adicione dois vetores e multiplique-os por um terceiro - você pode fazer isso de uma vez ou pode fazer um por vez. O Produto Escalar não se importa. Ele dá o mesmo número, a mesma resposta. Isso é ser distributivo. Dessa forma teremos: $A\cdot (B+C) = A\cdot B +A\cdot C$

Portanto, o Produto Escalar é constante e flexível. Ele se mantém o mesmo e se adapta, tudo de uma vez. Essas são suas regras, sua batida, seu fluxo. E isso é o que o torna uma ferramenta tão poderosa, tão versátil, em nosso esforço para entender o espaço e as coisas dentro dele.

Veja um vetor $A$. Uma seta solitária estendendo-se no espaço. Agora, imagine colocar outra seta exatamente igual, exatamente no mesmo lugar. Duas Setas juntas, $A$ e $A$, sem nenhum ângulo entre elas. Quando multiplicamos esses vetores, o Produto Escalar, obtemos a magnitude de $A$ ao quadrado, ou $\vert A \vert ^2$.

Por que? Porque o ângulo $\theta$ entre um vetor e ele mesmo é $zero$. E o cosseno de zero é $1$. Assim:

$$A\cdot A = \vert A \vert^2$$

Para simplificar, vamos dizer que $A^2$ é o mesmo que $ \vert A \vert ^2$. Uma notação, uma abreviação para o comprimento, magnitude, de $A$ ao quadrado.

Então, aqui está a lição: um vetor e ele mesmo, lado a lado, são definidos pela magnitude do próprio vetor, ao quadrado. É um pequeno pedaço de sabedoria, um truque, uma ferramenta. Mantenha esta ferramenta na sua caixa de ferramentas preferidas.

Assim como as ondas em uma praia, indo e voltando, de tempos em tempos precisamos rever nossas ferramentas e o conhecimento que construímos com elas. Em todos os sistemas de coordenadas que usamos para definir o espaço $\textbf{V}$ os vetores unitários são ortogonais. Setas no espaço que se cruzam em um ângulo reto. Este ângulo reto, esta ortogonalidade, garante duas propriedades interessantes.

$$a_x\cdot a_y=a_x\cdot a_z=a_y\cdot a_z=0$$

$$a_x\cdot a_x=a_y\cdot a_y=a_z\cdot a_z=1$$

A primeira garante que o Produto Escalar entre quaisquer dois componentes vetoriais ortogonais é $zero$, a segunda que o Produto Escalar entre os mesmos dois componentes vetoriais é $1$. Essas são duas garantias, duas verdades que podemos segurar firmes enquanto navegamos pelo vasto oceano do espaço vetorial. Como um farol em uma noite tempestuosa, elas nos guiam e nos ajudam a entender o indescritível. Mais que isso, serão ferramentas para transformar o muito difícil em muito fácil.

Estas propriedades podem ser expressas usando o [Delta de Kronecker](https://en.wikipedia.org/wiki/Kronecker_delta) definido por [Leopold Kronecker](https://en.wikipedia.org/wiki/Leopold_Kronecker)(1823–1891). O Delta de Kronecker é uma forma de representar por índices as dimensões do espaço vetorial escolhido. Uma generalização, para levarmos a Álgebra Linear ao seu potencial máximo, sem abandonar os limites que definimos para o estudo do Eletromagnetismo. Sendo assim, teremos:

$$
\begin{equation}
  \delta_{\mu \upsilon}=\begin{cases}
    1, se \space\space \mu = \upsilon .\\
    0, se \space\space \mu \neq \upsilon.
  \end{cases}
\end{equation}
$$

Usando o Delta de Kronecker podemos escrever as propriedades dos componentes ortogonais unitários em relação ao produto escalar como:

$$a_\mu \cdot a_\upsilon = \delta_{\mu \upsilon}$$

Que será útil na representação computacional de vetores e no entendimento de transformações vetoriais em espaços com mais de $3$ dimensões.

## Produto Vetorial

Imagine dois vetores, $A$ e $B$, como setas lançadas no espaço. Agora, imagine desenhar um paralelogramo com as magnitudes de $A$ e $B$ como lados. O Produto Vetorial de $A$ e $B$, representado por $A \times B$, é como uma seta disparada diretamente para fora desse paralelogramo, tão perfeitamente perpendicular quanto um mastro em um navio.

**A magnitude, o comprimento dessa seta, é a área do paralelogramo formado por $A$ e $B$**. É um número simples, mas importante. Descreve o quão longe essa flecha se estende no espaço. O comprimento do vetor resultado do Produto Vetorial. **O resultado do Produto Vetorial entre dois vetores é um vetor.**

Agora, essa Seta, esse Produto Vetorial, pode ser definido por uma equação analítica, geométrica, trigonométrica:

$$A \times B = \vert A \vert  \vert B \vert sen(\theta_{AB}) a_n$$

Onde $a_n$ representa o vetor unitário na direção perpendicular ao plano formado pelo paralelogramo formado por $A$ e $B$.
É uma fórmula simples, mas poderosa. Ela nos diz como calcular o Produto Vetorial, como determinar a direção, o sentido e a intensidade desta seta, lançada ao espaço.

A direção dessa seta, $a_n$, é decidida pela regra da mão direita. Estenda a mão, seus dedos apontando na direção de $A$. Agora, dobre seus dedos na direção de $B$. Seu polegar, erguido, aponta na direção de $a_n$, na direção do Produto Vetorial.

Assim, o Produto Vetorial determina uma forma de conectar dois vetores, $A$ e $B$, e criar algo novo: um terceiro vetor, lançado diretamente para fora do plano criado por $A$ e $B$. E esse vetor, esse Produto Vetorial, tem tanto uma magnitude - a área do paralelogramo - quanto uma direção - decidida pela regra da mão direita. É uma forma de entender o espaço tridimensional. E como todas as coisas na álgebra vetorial, é simples, mas poderoso.

$$A \times B = \vert A \vert  \vert B \vert sen(\theta_{AB}) a_n$$

É uma equação poderosa e simples, útil, muito útil, mas geométrica, trigonométrica e analítica. Algebricamente o Produto Vetorial pode ser encontrado usando uma matriz. As matrizes são os sargentos do exército da Álgebra Vetorial, úteis mas trabalhosas e cheias de regras. Considerando os vetores $A=A_x a_x+A_y a_y+A_z a_z$ e $B=B_x a_x+B_y a_y+B_z a_z$ o Produto Vetorial $A\times B$ será encontrado resolvendo a matriz:

$$
A\times B=\begin{vmatrix}
a_x & a_y & a_z\\
A_x & A_y & A_z\\
B_x & B_y & B_z
\end{vmatrix}
$$

A matriz será sempre montada desta forma. A primeira linha om os vetores unitários, a segunda com o primeiro operando, neste caso os componentes de $A$ e na terceira com os componentes de $B$. A Solução deste produto será encontrada, mais facilmente com o Método dos Cofatores. Para isso vamos ignorar a primeira linha.

Ignorando também a primeira coluna, a coluna do vetor unitário $a_x$ resta uma matriz composta de:

$$
\begin{vmatrix}
A_y & A_z\\
B_y & B_z
\end{vmatrix}
$$

O Esta matriz multiplicará o vetor unitário $a_x$. Depois vamos construir outras duas matrizes como esta. A segunda será encontrada quando ignorarmos a coluna referente ao unitário $a_y$, que multiplicará o oposto do vetor $a_y$.

$$
\begin{vmatrix}
A_x & A_z\\
B_x & B_z
\end{vmatrix}
$$

Finalmente ignoramos a coluna referente ao vetor unitário $a_z$ para obter:

$$
\begin{vmatrix}
A_x & A_y\\
B_x & B_y
\end{vmatrix}
$$

Que será multiplicada por $a_z$. Colocando tudo junto, em uma equação matricial teremos:

$$
A\times B=\begin{vmatrix}
a_x & a_y & a_z\\
A_x & A_y & A_z\\
B_x & B_y & B_z
\end{vmatrix}=\begin{vmatrix}
A_y & A_z\\
B_y & B_z
\end{vmatrix}a_x-\begin{vmatrix}
A_x & A_z\\
B_x & B_z
\end{vmatrix}a_y+\begin{vmatrix}
A_x & A_y\\
B_x & B_y
\end{vmatrix}a_z
$$

Cuide o negativo no segundo termo como cuidaria do leme do seu barco, sua vida depende disso e o resultado do Produto Vetorial Também. Uma vez que a equação matricial está montada. Cada matriz pode ser resolvida usando a [Regra de Sarrus](https://en.wikipedia.org/wiki/Rule_of_Sarrus) (multiplicação cruzada). Assim, nosso Produto Vetorial será simplificado por:

$$A\times B=(A_y B_z- A_z B_y)a_x-(A_x B_z-A_z B_x)a_y+(A_x B_y-A_y B_x)a_z$$

Cuidado com os determinantes, o Chapeleiro não ficou louco por causa do chumbo, muito usado na fabricação de chapéus. Ficou louco [resolvendo determinantes](https://www.johndcook.com/blog/2023/07/10/lewis-carroll-determinants/). Talvez um exemplo afaste a insanidade tempo suficiente para você continuar lendo.

<p class="exp">
<b>Exemplo 8:</b> dados os vetores $A=a_x+2a_y+3a_z$ e $B=4a_x+5a_y-6a_z$. (a) Calcule o Produto Vetorial entre $A$ e $B$. (b) Encontre o ângulo $\theta$ entre $A$ e $B$.
<br><br>
<b>Solução:</b><br>
(a) Vamos começar com o Produto Vetorial:

\[
A\times B=\begin{vmatrix}
a_x & a_y & a_z\\
A_x & A_y & A_z\\
B_x & B_y & B_z
\end{vmatrix} = \begin{vmatrix}
a_x & a_y & a_z\\
1 & 2 & 3\\
4 & 5 & -6
\end{vmatrix}
\]

Que será reduzida a:

\[
A \times B= \begin{vmatrix}
2 & 3\\
5 & -6
\end{vmatrix}a_x-\begin{vmatrix}
1 & 3\\
4 & -6
\end{vmatrix}a_y+\begin{vmatrix}
1 & 2\\
4 & 5
\end{vmatrix}a_z
\]

Usando Sarrus em cada uma destas matrizes teremos:

\[A\times B=(2(-6) - 3(5))a_x-(1(-6)-3(4))a_y+(1(5)-2(4))a_z\]

\[A\times B=-27a_x+18a_y-3a_z\]

Esta foi a parte difícil, agora precisamos dos módulos, magnitudes, dos vetores $A$ e $B$.

\[\vert A \vert = \sqrt{1^2+2^2+3^2} = \sqrt{14} \approx 3.74165\]

\[\vert B \vert = \sqrt{4^2+5^2+6^2} = \sqrt{77} \approx 8.77496\]

Para calcular o ângulo vamos usar a equação analítica, ou trigonométrica, do Produto Vetorial:

\[A \times B = \vert A \vert  \vert B \vert sen(\theta_{AB}) a_n\]

A forma mais fácil de resolver este problema é aplicar o módulo aos dois lados da equação. Se fizermos isso, teremos:

\[\vert A \times B \vert = \vert A \vert  \vert B \vert sen(\theta_{AB}) \vert a_n \vert \]

Como $a_n$ é um vetor unitário, por definição $\vert a_n \vert = 1$ logo:

\[\vert A \times B \vert = \vert A \vert  \vert B \vert sen(\theta_{AB})\]

Ou, para ficar mais claro:

\[sen(\theta_{AB}) = \frac{\vert A \times B \vert}{\vert A \vert \vert B \vert}\]

Os módulos de $A$ e $B$ já tenos, precisamos apenas do módulo de $A\times B$.

\[
\vert A\times B \vert = \sqrt{27^2+16^2+3^2} = \sqrt{994} \approx 31.5298
\]

Assim o seno do ângulo $\theta_{AB}$ será dado por:

\[sen(\theta_{AB}) = \frac{\sqrt{994}}{(\sqrt{14})(\sqrt{77})} \approx \frac{31.5298}{(3.74165)(8.77496)}\]

\[sen(\theta_{AB}) = 0.960316\]

\[ \theta_{AB} =73.8^0 \]

</p>

O Produto Vetorial é como uma dança entre vetores. E como todas as danças tem características únicas e interessantes:

1. **Comutatividade:** no universo dos vetores, há uma dança estranha acontecendo. $A \times B$ e $B \times A$ não são a mesma coisa, eles não trocam de lugar facilmente como dançarinos em um salão de baile. Em vez disso, eles são como dois boxeadores em um ringue, um o espelho do outro, mas em direções opostas. Assim, $A \times B$ é o oposto de $B \times A$. Assim: $ A \times B =-B \times A$

2. **Associatividade:** imagine três dançarinos: $A$, $B$ e $C$. A sequência de seus passos importa. $A$ dançando com $B$, depois com $C$, não é o mesmo que $A$ dançando com o resultado de $B$ e $C$ juntos. Assim como a dança, a ordem dos parceiros importa. O Produto Vetorial não é associativo. Desta forma: $A \times (B \times C) \neq (A \times B) \times C$

3. **Distributividade:** existe um aspecto familiar. Quando $A$ dança com a soma de $B$ e $C$, é a mesma coisa que $A$ dançando com $B$ e depois com $C$. Distributividade, uma velha amiga da aritmética, aparece aqui, guiando a dança. O que pode ser escrito como: $A \times (B+C) = A \times B + A \times C$

4. **Multiplicação por Escalar:** imagine que temos dois vetores, firme e diretos, apontando em suas direções particulares no espaço. Chamamos eles de $A$ e $B$. Esses dois, em uma dança matemática, se entrelaçam em um produto vetorial, formando um terceiro vetor, o $C$, perpendicular a ambos $A$ e $B$. Mais que isso, perpendicular ao paralelogramo formado por $$A$$ e $B$. Ainda mais, perpendicular ao plano formado por $A$ e $B$.

Portanto, a dança do Produto Vetorial é peculiar e intrigante, não troca de lugar como a dança tradicional e a sequência de seus passos importa, mas ela acolhe a velha regra da distributividade. Uma dança peculiar no palco da matemática.

Agora entra em cena um escalar, $k$, um número simples, porém carregado de influência. Ele se aproxima do produto vetorial e o muda, mas não de maneira selvagem ou imprevisível, e sim com a precisão de um relojoeiro. A magnitude do produto vetorial é esticada ou contraída pelo escalar, dependendo de seu valor. Isto pode ser escrito matematicamente como:

$$k(A \times B) = (kA) \times B = A \times (kB)$$

Porém, como o norte em uma bússola, a direção do produto vetorial não se altera. O resultado é um novo vetor, $D$, que é um múltiplo escalar do original $C$. O vetor $D$ carrega a influência do escalar $k$, mas mantém a orientação original de $C$.

Por fim, precisamos tirar para dançar os vetores unitários. Estrutura de formação dos nossos sistemas de coordenadas. Como Produto Vetorial $A\times B$ produz um vetor ortogonal ao plano formado por $A$ e $B$ a aplicação deste conceito a dois dos vetores unitários de um sistema de coordenadas irá produzir o terceiro vetor deste sistema. Sendo assim, observando o Sistema de Coordenadas Cartesianas teremos:

$$a_x\times a_y = a_z$$

$$a_x\times a_z = a_y$$

$$a_y\times a_z = a_x$$

Mais uma ferramenta que precisamos manter à mão. Um conjunto de regras que irão simplificar equações e iluminar o desconhecido de forma mais simples, quase natural.

### A Identidade de Jacobi

Associatividade, a dança entre três vetores no espaço infinito chamou a atenção de [Carl Gustav Jacob Jacobi](https://en.wikipedia.org/wiki/Carl_Gustav_Jacob_Jacobi) que entrando nesta roda encontrou uma dança de ritmo constante.

$$A \times (B  \times  C) + B  \times  (C  \times  A) + C  \times (A  \times B) = 0$$

Em Álgebra Linear, a Identidade de Jacobi é a marca da existência de uma [Álgebra de Lie](https://en.wikipedia.org/wiki/Lie_algebra), um pilar para descrever simetrias contínuas, presentes em domínios da física como a teoria quântica de campos e a teoria geral da relatividade. Talvez, se o tempo estiver bom e eu me sentir forte, embarquemos algum dia em uma jornada pelos mares crespos da teoria quântica dos campos.

Em física, a identidade de Jacobi serve como uma regra fundamental para análises que envolvem a conservação do momento angular. Ela é uma ferramenta no entendimento da rotação de corpos rígidos, descrita pela equação de [Euler](https://en.wikipedia.org/wiki/Leonhard_Euler), que descrevem o movimento de um corpo rígido no espaço e um protagonista em funções na mecânica quântica.

Na geometria diferencial, a Identidade de Jacobi é vital para a criação dos [Campos de Jacobi](https://pt.wikipedia.org/wiki/Campo_de_Jacobi), uma expansão das [Superfícies de Riemann](https://en.wikipedia.org/wiki/Riemann_surface) que permite a inclusão de simetrias adicionais.

Então, mesmo que a identidade de Jacobi possa parecer um conjunto abstrato de símbolos matemáticos, ela é, na verdade, uma ferramenta que guia inúmeras investigações em ciência e matemática, como um farol confiável que nos leva adiante na procura constante pelo conhecimento.

## Usando a Álgebra Vetorial no Eletromagnetismo

Em um mundo onde a ciência se entrelaça com a arte, a álgebra vetorial se ergue como uma ponte sólida entre o visível e o invisível. Neste ponto da nossa jornada, navegaremos pelas correntes do eletromagnetismo, uma jornada onde cada vetor conta uma história, cada produto escalar revela uma conexão profunda, e cada produto vetorial desvenda um mistério. A matemática da Álgebra Vetorial é a ferramenta que nos guiará.

Prepare-se para uma surpresa olhe com cuidado e verá como a matemática se torna poesia, desvendando os segredos do universo elétrico e magnético. Esta rota promete uma jornada de descoberta, compreensão e surpresa. Começaremos pelo mais básico de todos os básicos, a Lei de Coulomb.

### Lei de Coulomb

No ano da glória de 1785, um pesquisador francês, [Charles-Augustin de Coulomb](https://en.wikipedia.org/wiki/Charles-Augustin_de_Coulomb)Formulou, empiricamente uma lei para definir a intensidade da força exercida por uma carga elétrica $Q$ sobre outra dada por: 

$$
F_{21} = K_e \frac{Q_1Q_2}{R^2}
$$

Esta era uma lei empírica, baseada na observação dos efeitos de cargas diferentes em uma balança de torção. [Henry Cavendish](https://en.wikipedia.org/wiki/Charles-Augustin_de_Coulomb) chegou a mesma equação, de forma independente alguns anos depois.E até o trabalho de [Michael Faraday](https://en.wikipedia.org/wiki/Michael_Faraday) sobre as linhas de força, esta equação era suficiente. Quase 100 anos depois de Coulomb, matemáticos como Gauss, Hamilton, Gibbs e Maxwell deram a esta lei uma roupagem vetorial. 

$$
F_{21} = \frac{1}{4\pi \epsilon_0 \epsilon_r} \frac{Q_1Q_2}{R^2} a_{21} 
$$

Nesta equação: 

- $F_{21}$ é a força que é aplicada sobre a carga 2, $Q_2$,  devido a existência da carga 1, $Q_1$.
- $\epslion_0$ representa a permissividade do vácuo, medida em Farads por metro ($F/m$). 
- $\epslion_r$ representa a permissividade do meio onde as cargas estão, um valor escalar e sem unidade.
- $4\pi $ surge da existência da força em todos os pontos do espaço, uma esfera que se estende da carga até o infinito. 
- $Q_1Q_2$ representa o produto entre as intensidades das cargas que no Sistema Internacional de Unidades são medidas em Coulombs ($C$). 
- $a_{21}$ representa o vetor unitário com origem em $Q1$ e destino em $Q2$.
# Cálculo Vetorial

Cálculo vetorial, soa como algo saído de uma história de ficção científica. Mas é mais terra-a-terra do que podemos imaginar de longe. Trata-se uma técnica para lidar com quantidades que têm tanto magnitude quanto direção de forma contínua. Velocidade. Força. Fluxo de um rio, Campos Elétricos, Campos Magnéticos. Coisas que não apenas têm um tamanho, mas também uma direção, um sentido. Não sei se já falei sobre isso, são as grandezas que chamamos de vetoriais e representamos por vetores.

A beleza do cálculo vetorial perceptível na sua capacidade de descrever o mundo físico profunda e significativamente. 

Considere um campo de trigo balançando ao vento. O vento não está apenas soprando com uma certa força, mas também em uma certa direção. O cálculo vetorial nos permitirá entender fenômenos como esse e transformá-los em ferramentas de inovação e sucesso.

O cálculo vetorial é construído sobre três operações fundamentais: o gradiente, a divergência e o rotacional. O gradiente nos diz a direção e a taxa na qual uma quantidade está mudando. A divergência nos diz o quanto um campo está se espalhando de um ponto. E o rotacional nos dá uma medida da rotação ou vorticidade de um campo.

Se tivermos uma função escalar $\mathbf{f}$, o gradiente de $\mathbf{f}$ será dado por:

$$
\nabla \mathbf{f} = \left( \frac{\partial \mathbf{f}}{\partial x}, \frac{\partial \mathbf{f}}{\partial y}, \frac{\partial \mathbf{f}}{\partial z} \right)
$$

E se tivermos um campo vetorial $ \mathbf{F} = F_x a_x + F_y a_y + F_z a_x $, a divergência de $\mathbf{F}$ é dada por:

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

E o rotacional de $\mathbf{F}$ é:

$$
\nabla \times \mathbf{F} = \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z} \right) a_x - \left( \frac{\partial F_z}{\partial x} - \frac{\partial F_x}{\partial z} \right) a_i + \left( \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y} \right) a_z
$$

A única coisa que pode encher seus olhos de lágrimas é o sal trazido pela maresia, não o medo do Cálculo Vetorial. Então, não se intimide por estas equações herméticas, quase esotéricas. O Cálculo Vetorial é apenas conjunto de ferramentas, como um canivete suíço, que nos ajuda a explorar e entender o mundo ao nosso redor. Nós vamos abrir cada ferramenta deste canivete e aprender a usá-las.

## Campos Vetoriais

Quando olhamos as grandezas escalares, traçamos Campos Escalares. Como uma planície aberta, eles se estendem no espaço, sem direção, mas com magnitude, definidos por uma função $\mathbf{f}(x,y,z)$, onde $x$, $y$, $z$ pertencem a um universo de triplas de números reais. Agora, para as grandezas vetoriais, moldamos Campos Vetoriais, definidos por funções vetoriais $\mathbf{F}(x,y,z)$, onde $x$, $y$, $z$ são componentes vetoriais. Em outras palavras, representamos Campos Vetoriais no espaço como um sistema onde cada ponto do espaço puxa um vetor.

Imagine-se em um rio, a correnteza o arrastando, conduzindo seu corpo. A correnteza aplica uma força sobre seu corpo. O rio tem uma velocidade, uma direção. Em cada ponto, ele te empurra de uma forma diferente. Isso é um campo vetorial. Ele é como um mapa, com forças distribuídas, representadas por setas desenhadas para te orientar. Mas essas setas não são meras orientações. Elas têm um comprimento, uma magnitude, e uma direção e um sentido. Elas são vetores. E o mapa completo, deste rio com todas as suas setas, descreverá um campo vetorial.

Em cada ponto no espaço, o campo vetorial tem um vetor. Os vetores podem variar de ponto para ponto. Pense de novo no rio. Em alguns lugares, a correnteza é forte e rápida. Em outros, é lenta e suave. Cada vetor representará essa correnteza em um ponto específico. E o campo vetorial representará o rio todo.

Frequentemente, Campos Vetoriais são chamados para representar cenas do mundo físico: a ação das forças na mecânica, o desempenho dos campos elétricos e magnéticos no Eletromagnetismo, o fluxo de fluidos na dinâmica dos fluidos. Em cada ponto, as coordenadas $(x, y, z)$ são protagonistas, ao lado das funções escalares $P$, $Q$ e $R$. O vetor resultante no palco tem componentes nas direções $x$, $y$ e $z$, representadas pelos atores coadjuvantes, os vetores unitários $(a_x, a_y, a_z)$.

Imaginar um campo vetorial no palco do espaço tridimensional é tarefa árdua que requer visão espacial, coisa para poucos. Para aqueles que já trilharam os caminhos árduos da geometria e do desenho tridimensional Se nosso palco for bidimensional, poderemos colocar os vetores em um plano, selecionar alguns pontos e traçar estes vetores. Neste caso voltaremos nossa atenção e esforço para trabalhar com apenas os componentes $x$ e $y$ e o campo vetorial será definido por uma função dada por:

$$\mathbf{F}(x, y) = (P(x, y), Q(x, y))$$

Uma função, uma definição direta, e simples, ainda assim, sem nenhum apelo visual. Mas somos insistentes e estamos estudando matemática, a rota que nos levará ao horizonte do Eletromagnetismo. Que nasce na carga elétrica, fenômeno simples, estrutural e belo que cria forças que se espalham por todo universo. Vamos pegar duas cargas de mesma intensidade e colocar no nosso palco.

![Campo Vetorial devido a duas cargas elétricas](/assets/images/CampoVetorial1.jpeg){:class="lazyimg"}

<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 3 - Diagrama de um campo vetorial em duas dimensões.</legend>

Agora podemos ver o Campo Vetorial, simples, com poucos pontos escolhidos no espaço e duas cargas pontuais representadas por círculos. Um vermelho, quente, para indicar a carga positiva outro azul, frio, para indicar a carga negativa. Treine a vista. Seja cuidadoso, detalhista. E verá a interação das forças em todos os pontos do espaço.

O Campo Elétrico, o Campo Vetorial que a figura apresenta, surge, na força da própria definição, na carga elétrica positiva por isso os vetores apontam para fora, para longe desta carga, divergem. E são drenados pela carga elétrica negativa, as setas apontam diretamente para ela, convergem. Em todos os pontos que escolhi para plotar em todo o espaço do plano desenhado, você pode ver o efeito das forças criadas por esta carga. Em alguns pontos um vetor está exatamente sobre o outro, eles se anulam, em todos os outros pontos do espaço se somam.

Visualizar um Campo Vetorial é como assistir a uma peça, com cada vetor como um ator em um gráfico. Cada vetor é um personagem desenhado com uma linha direcionada, geralmente com uma seta, atuando com direção e magnitude. Mas essa peça é complexa e exige tempo e paciência para ser compreendida. Uma abordagem mais simples seria tomar um ponto de teste no espaço e desenhar algumas linhas entre a origem do Campo Vetorial e esse ponto, traçando assim os principais pontos da trama.

O Campo Vetorial requer cuidado, carinho e atenção, ele está em todos os pontos do espaço. Contínuo e muitas vezes, infinito. Trabalhar com a continuidade e com o infinito requer mãos calejadas e fortes. Teremos que recorrer a Newton e [Leibniz](https://en.wikipedia.org/wiki/Gottfried_Wilhelm_Leibniz) e ao Cálculo Integral e Diferencial. Não tema! Ainda que muitos se acovardem frente a continuidade este não será nosso destino. Vamos conquistar integrais e diferenciais como [Odisseu](https://en.wikipedia.org/wiki/Odysseus) conquistou [Troia](https://en.wikipedia.org/wiki/Trojan_War), antes de entrar em batalha vamos afiar espadas, lustrar escudos e lanças, na forma de gradiente, divergência e rotacional.

## Gradiente

Imagine-se no topo de uma montanha, cercado por terreno acidentado. Seu objetivo é descer a montanha, mas o caminho não é claramente marcado. Você olha ao redor, tentando decidir para qual direção deve seguir. O gradiente é como uma bússola que indica a direção de maior inclinação. Se você seguir o gradiente, estará se movendo na direção de maior declividade. Se a velocidade for importante é nesta direção que descerá mais rápido.

Agora, vamos trazer um pouco de matemática para esta metáfora. Em um espaço de múltiplas dimensões. Imagine uma montanha com muitos picos e vales, e você pode se mover em qualquer direção, o gradiente de uma função em um determinado ponto é um vetor que aponta na direção de maior variação desta função. Se a função tem múltiplas dimensões, **o gradiente é o vetor que resulta da aplicação das derivadas parciais da função**.

Se tivermos uma função $\mathbf{f}(x, y)$, uma função escalar, o gradiente de $\mathbf{f}$ será dado por:

$$
\nabla \mathbf{f} = \left( \frac{\partial \mathbf{f}}{\partial x}, \frac{\partial \mathbf{f}}{\partial y} \right)
$$

Assim como a bússola na montanha, o gradiente nos mostra a direção à seguir para maximizar, ou minimizar, a função. É uma ferramenta importante na matemática e na física, especialmente em otimização e aprendizado de máquina. Mas não tire seus olhos do ponto mais importante: **o gradiente é uma operação que aplicada a uma função escalar devolve um vetor**. Em três dimensões, usando o Sistema de Coordenadas Cartesianas teremos:

$$
\nabla \mathbf{f} = \left( \frac{\partial \mathbf{f}}{\partial x}, \frac{\partial \mathbf{f}}{\partial y}, \frac{\partial \mathbf{f}}{\partial z} \right)
$$

onde $\frac{\partial \mathbf{f} }{\partial x}$, $\frac{\partial \mathbf{f}}{\partial y}$, e $\frac{\partial \mathbf{f}}{\partial z}$ são as derivadas parciais de $\mathbf{f}$ com respeito a $x$, $y$, e $z$ respectivamente.

Só a expressão **Derivadas parciais** pode fazer o coração bater mais rápido. O medo não o guiará aqui. As derivadas parciais são como velhos amigos que você ainda não conheceu.

Imagine-se em uma grande pradaria. O vento está soprando, carregando consigo o cheiro da grama e da terra. Você está livre para caminhar em qualquer direção. Para o norte, onde o sol se põe, ou para o sul, onde a floresta começa. Cada passo que você dá muda a paisagem ao seu redor, mas de maneiras diferentes dependendo da direção em que você escolheu caminhar.

A derivada parcial é apenas essa ideia, vestida com a roupa do cálculo. Ela apenas quer saber: e se eu der um pequeno passo para o norte, ou seja, mudar um pouco $x$, como a paisagem, nossa função, vai mudar? Ou o se for para o sul, ou em qualquer outra direção que escolher.

Então, em vez de temer as derivadas parciais, podemos vê-las como uma ferramentas úteis que nos ajudem a entender a terra sob nossos pés, o vento, a água que flui, o Campo Elétrico, entender a função que estamos usando para descrever o fenômeno que queremos entender. Com as derivadas parciais, podemos entender melhor o terreno onde pisamos, saber para onde estamos indo e como chegar lá. E isso é bom. Não é?

Uma derivada parcial de uma função de várias variáveis revela a taxa na qual a função muda quando pequenas alterações são feitas em apenas uma das incógnitas da função, mantendo todas as outras constantes. O conceito é semelhante ao conceito de derivada em cálculo de uma variável, entretanto agora estamos considerando funções com mais de uma incógnita.

Por exemplo, se temos uma função $\mathbf{f}(x, y)$, a derivada parcial de $\mathbf{f}$ em relação a $x$ (denotada por $\frac{\partial \mathbf{f}}{\partial x}$ mede a taxa de variação de $\mathbf{f}$ em relação a pequenas mudanças em $x$, mantendo $y$ constante. Da mesma forma, $\frac{\partial \mathbf{f}}{\partial y}$ mede a taxa de variação de $\mathbf{f}$ em relação a pequenas mudanças em $y$, mantendo $x$ constante. Em três dimensões, a derivada parcial em relação uma das dimensões é a derivada de $\mathbf{f}$ enquanto mantemos as outras constantes. Nada mais que a repetição, dimensão a dimensão da derivada em relação a uma dimensão enquanto as outras são constantes.

**O gradiente mede a taxa em que o Campo Escalar varia em uma determinada direção.** Para clarear e afastar a sombra das dúvidas, nada melhor que um exemplo.

<p class="exp">
<b>Exemplo 9:</b> considerando o Campo Escalar dado por $\mathbf{f}(x,y) = 10sin(\frac{x^2}{5})+4y$, (a) calcule a intensidade do campo no ponto $P(2,3)$, (b) o gradiente deste campo no ponto $P$.  
<br><br>
<b>Solução:</b><br>

(a) A intensidade em um ponto é trivial, trata-se apenas da aplicação das coordenadas do ponto desejado na função do campo. Sendo assim:

\[\mathbf{f}(x,y) = 10sin(\frac{x^2}{5})+4y\]

\[\mathbf{f}(2,3) = 10sin(\frac{2^2}{5})a_x+4(3)a_y\]

\[\mathbf{f}(2,3) = 7.17356a_x+12a_y\]

(b) agora precisamos calcular o gradiente. O gradiente de uma função $\mathbf{f}(x, y)$ é um vetor que consiste nas derivadas parciais da função com respeito a cada uma de suas variáveis que representam suas coordenadas. <br><br>

Vamos calcular as derivadas parciais de $\mathbf{f}$ com respeito a $x$ e $y$, passo a passo:<br><br>

Primeiro, a derivada parcial de $f$ com respeito a $x$ é dada por:

\[
\frac{\partial \mathbf{f}}{\partial x} = \frac{\partial}{\partial x} \left[10\sin\left(\frac{x^2}{5}\right) + 4y\right]
\]

Nós podemos dividir a expressão em duas partes e calcular a derivada de cada uma delas separadamente. A derivada de uma constante é zero, então a derivada de $4y$ com respeito a $x$ é zero. Agora, vamos calcular a derivada do primeiro termo:

\[
\frac{\partial}{\partial x} \left[10\sin\left(\frac{x^2}{5}\right)\right] = 10\cos\left(\frac{x^2}{5}\right) \cdot \frac{\partial}{\partial x} \left[\frac{x^2}{5}\right]
\]

Usando a regra da cadeia, obtemos:

\[
10\cos\left(\frac{x^2}{5}\right) \cdot \frac{2x}{5} = \frac{20x}{5}\cos\left(\frac{x^2}{5}\right) = 4x\cos\left(\frac{x^2}{5}\right)
\]

Portanto, a derivada parcial de $\mathbf{f}$ com respeito a $x$ é:

\[
\frac{\partial \mathbf{f}}{\partial x} = 4x\cos\left(\frac{x^2}{5}\right)
\]

Agora, vamos calcular a derivada parcial de $\mathbf{f}$ com respeito a $y$:

\[
\frac{\partial \mathbf{F}}{\partial y} = \frac{\partial}{\partial y} \left[10\sin\left(\frac{x^2}{5}\right) + 4y\right]
\]

Novamente, dividindo a expressão em duas partes, a derivada do primeiro termo com respeito a $y$ é zero (pois não há $y$ no termo), e a derivada do segundo termo é $4$. Portanto, a derivada parcial de $\mathbf{f}$ com respeito a $y$ é:

\[
\frac{\partial \mathbf{F}}{\partial y} = 4
\]

Assim, o gradiente de $\mathbf{f}$ é dado por:

\[
\nabla \mathbf{F} = \left[\frac{\partial \mathbf{F}}{\partial x}, \frac{\partial \mathbf{F}}{\partial y}\right] = \left(4x\cos\left(\frac{x^2}{5}\right), 4\right)
\]

E esta é a equação que define o gradiente. Para saber o valor do gradiente no ponto $P$ tudo que precisamos é aplicar o ponto na equação então:

\[
\nabla \mathbf{F}(2,3) = \left( 4(2)\cos\left(\frac{2^2}{5}\right), 4\right) = \left(  5.57365, 4 \right)
\]

Ao derivarmos parcialmente o Campo Vetorial $\mathbf{f}$ escolhemos nosso Sistema de Coordenadas. Sendo assim:

\[
\nabla \mathbf{f}(2,3) = 5.57365 a_x+ 4a_y
\]
</p>

Assim como um navegador considera a variação da profundidade do oceano em diferentes direções para traçar a rota mais segura, a derivada parcial nos ajuda a entender como uma função se comporta quando mudamos suas variáveis de entrada. O gradiente é a forma de fazermos isso em todas as dimensões, derivando em uma incógnita de cada vez.

### Significado do Gradiente

Em qualquer ponto $P$ o gradiente é um vetor que aponta na direção da maior variação de um Campo Escalar neste ponto. Nós podemos voltar ao exemplo 8 e tentar apresentar isso de uma forma mais didática. Primeiro o gráfico do Campo Escalar dado por: $\mathbf{F}(x,y) = 10sin(\frac{x^2}{5})+4y$.

![Gráfico do Campo Escalar](/assets/images/Func1Grad.jpeg){:# class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 4 - Gráfico de um Campo Escalar $f(x,y)$.</legend>

Nesta imagem é possível ver a variação do campo $\mathbf{f}(x,y)$$ eu escolhi uma função em $$\mathbf{f}(x,y)$ no domínio dos $\mathbb{R}^2$ por ser mais fácil de desenhar e visualizar, toda a variação fica no domínio de $z$. Podemos plotar o gradiente na superfície criada pelo campo $\mathbf{f}(x,y)$.

![Gráfico do Campo Escalar mostrando a intensidade do gradiente ](/assets/images/func1Grad2.jpeg){:class="lazyimg"}
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Figura 5 - Gráfico de um Campo Escalar $f(x,y) representando o Gradiente$.</legend>

Nesta imagem, em cada ponto, a cor da superfície foi definida de acordo com a intensidade do gradiente. Quanto menor esta intensidade, mais próximo do vermelho. Quanto maior, mais próximo do Azul. Veja que a variação é maior nas bordas de descida ou subida e menor nos picos e vales. Coisas características da derivação.

E é só isso. Se você entendeu até aqui, entendeu o gradiente e já sabe aplicá-lo. Eu disse que lágrimas seriam evitadas.

### Propriedades do Gradiente

O reino do gradiente é o reino dos Campos Escalares, o gradiente tem características matemáticas distintas que o guiam em sua exploração:

1. **Linearidade**: O gradiente é uma operação linear. Isso significa que para quaisquer campos escalares $f$ e $g$, e quaisquer constantes $a$ e $b$, temos:

$$
  \nabla (af + bg) = a \nabla f + b \nabla g
$$

O gradiente de uma soma de funções é a soma dos gradientes das funções, cada um ponderado por sua respectiva constante.

2. **Produto por Escalar**: O gradiente de uma função escalar multiplicada por uma constante é a constante vezes o gradiente da função. Para uma função escalar $f$ e uma constante $a$, teremos:

$$
\nabla (af) = a \nabla f
$$

3. **Regra do Produto**: Para o produto de duas funções escalares $f$ e $g$, a regra do produto para o gradiente é dada por:

$$
\nabla (fg) = f \nabla g + g \nabla f
$$

Esta é a versão para gradientes da regra do produto para derivadas no cálculo unidimensional.

4. **Regra da Cadeia**: Para a função composta $f(g(x))$, a regra da cadeia para o gradiente será dada por:

$$
\nabla f(g(x)) = (\nabla g(x)) f'(g(x))
$$

Esta é a extensão da regra da cadeia familiar do cálculo unidimensional.

Estas propriedades, como as leis imutáveis da física, regem a conduta do gradiente em sua jornada através dos campos escalares. No palco do eletromagnetismo, o gradiente desempenha um papel crucial na descrição de como os Campos Elétrico e Magnético variam no espaço.

1. **Campo Elétrico e Potencial Elétrico**: o campo elétrico é o gradiente negativo do potencial elétrico. Isso significa que o Campo Elétrico aponta na direção de maior variação do potencial elétrico, formalmente expresso como:

$$
 \mathbf{E} = -\nabla V
$$

Aqui, $\mathbf{E}$ é o Campo Elétrico e $V$ é o potencial elétrico. O gradiente, portanto, indica a encosta, o aclive, mais íngreme que uma partícula carregada experimentaria ao mover-se no Campo Elétrico.

2. **Campo Magnético**: o Campo Magnético não é o gradiente de nenhum potencial escalar, **O Campo Magnético é um campo vetorial cuja divergência é zero**. No entanto, em situações estáticas ou de baixas frequências, pode-se definir um potencial vetorial $\mathbf{A}$ tal que:

$$ \mathbf{B} = \nabla \times \mathbf{A} $$

Essas propriedades do gradiente são como setas, apontando o caminho através das complexidades do eletromagnetismo. O gradiente é a ferramenta mais simples do nosso canivete suíço do cálculo vetorial.

## Divergência

Seu barco sempre será pequeno perto do oceano e da força do vento. Você sente o vento em seu rosto, cada sopro, uma força direcional, um vetor com magnitude e direção. Todo o oceano e a atmosfera acima dele compõem um campo vetorial, com o vento soprando em várias direções, forças aplicadas sobre o seu barco.

No oceano, o tempo é um caprichoso mestre de marionetes, manipulando o clima com uma rapidez alucinante. Agora, em uma tempestade, existem lugares onde o vento parece convergir, como se estivesse sendo sugado para dentro. Em outros lugares, parece que o vento está explodindo para fora. Esses são os pontos de divergência e convergência do campo vetorial do vento.

Um lugar onde o vento está sendo sugado para dentro tem uma divergência negativa - o vento está "fluido" para dentro mais do que está saindo. Um lugar onde o vento está explodindo para fora tem uma divergência positiva - o vento está saindo mais do que está entrando. Este é o conceito que aplicamos as cargas elétricas, o campo elétrico diverge das cargas positivas e converge para as negativas. Isto porque assim convencionamos há séculos, quando começamos e estudar o Eletromagnetismo.

Matematicamente, **a divergência é uma maneira de medir esses comportamentos de "expansão" ou "contração" de um campo vetorial**. Para um campo vetorial em três dimensões, $\mathbf{F} = f_x a_x + f_y a_y + f_z a_z$, a divergência é calculada como:

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

A divergência, então, é como a "taxa de expansão" do vento em um determinado ponto - mostra se há mais vento saindo ou entrando em uma região específica do espaço, um lugar. assim como a sensação que temos no meio de uma tempestade. **A divergência é o resultado do produto escalar entre o operador $\nabla$ e o Campo Vetorial. O resultado da divergência é uma função escalar que dá a taxa na qual o fluxo do campo vetorial está se expandindo ou contraindo em um determinado ponto**.

Sendo um pouco mais frio podemos dizer que a divergência é um operador diferencial que atua sobre um Campo Vetorial para produzir um Campo Escalar. Em termos físicos, a divergência em um ponto específico de um Campo Vetorial representa a fonte ou dreno no ponto: uma divergência positiva indica que neste ponto existe uma fonte, ou fluxo de vetores para fora, divergindo. Enquanto uma divergência negativa indica um dreno ou fluxo para dentro, convergindo.

### Fluxo e a Lei de Gauss

O fluxo, nas margens do cálculo vetorial, é **uma medida da quantidade de campo que passa através de uma superfície**. Imagine um rio, com a água fluindo com velocidades e direções variadas. Cada molécula de água tem uma velocidade - um vetor - e toda a massa de água compõe um Campo Vetorial.

Se você colocar uma rede no rio, o fluxo do campo de água através da rede seria uma medida de quanta água está passando por ela. Para um campo vetorial $\mathbf{F}$ e uma superfície $S$ com vetor normal dado por $\mathbf{n}$, o fluxo será definido, com a formalidade da matemática, como:

$$
\iint_S (\mathbf{F} \cdot \mathbf{n}) \, dS
$$

Uma integral dupla, integral de superfície onde $dS$ é o elemento diferencial de área da superfície, e o produto escalar $\mathbf{F} \cdot \mathbf{n}$ mede o quanto do campo está fluindo perpendicularmente à superfície.

Agora, a divergência entra em cena como a versão local do fluxo. Se encolhermos a rede até que ela seja infinitesimalmente pequena, o fluxo através da rede se tornará infinitesimal e será dado pela divergência do campo de água no ponto onde a rede está. Matematicamente, isso é expresso na Lei de [Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss):

$$
\nabla \cdot \mathbf{F} = \frac{d (\text{Fluxo})}{dV}
$$

Onde, $V$ é o volume da região, e **a Lei de Gauss afirma que a divergência de um campo vetorial em um ponto é igual à taxa de variação do fluxo do campo através de uma superfície que envolve o ponto**.

### Teorema da Divergência

Imagine-se como um explorador atravessando o vasto terreno do cálculo vetorial. Você se depara com duas paisagens: a superfície e o volume. Cada uma tem suas próprias características e dificuldades, mas há uma ponte que as conecta, uma rota que permite viajar entre elas. Esta é a Lei de Gauss.

A Lei de Gauss, ou o Teorema da Divergência, é a ponte que interliga dois mundos diferentes. Ela afirma que, para um dado campo vetorial $\mathbf{F}$, a integral de volume da divergência do campo vetorial sobre um volume $V$ é igual à integral de superfície do campo vetorial através da superfície $S$ que delimita o volume $V$:

$$
\iiint_V (\nabla \cdot \mathbf{F}) \, dV = \iint_S (\mathbf{F} \cdot \mathbf{n}) \, dS
$$

Uma integral tripla igual a uma integral dupla. Aqui, $dV$ é um pedaço infinitesimalmente pequeno de volume dentro de $V$, e $dS$ é um pedaço infinitesimalmente pequeno da superfície $S$, respectivamente elemento infinitesimal de volume e área. O vetor $\mathbf{n}$ é um vetor normal apontando para fora da superfície.

Com a Lei de Gauss, podemos ir e voltar entre a superfície e o volume, entre o plano e o volume. Esta é a beleza e o poder da matemática: a linguagem e as ferramentas para navegar pelos mais complexos terrenos.

### Propriedades da Divergência

No universo dos campos vetoriais, a divergência tem propriedades matemáticas distintas que servem como marcos na paisagem:

1. **Linearidade**: A divergência é uma operação linear. Isso significa que para quaisquer campos vetoriais $\mathbf{F}$ e $mathbf{G}$, e quaisquer escalares $a$ e $b$, temos:

$$
    \nabla \cdot (a\mathbf{F} + b\mathbf{G}) = a (\nabla \cdot \mathbf{F}) + b (\nabla \cdot \mathbf{G})
$$

A divergência de uma soma é a soma das divergências, com cada divergência ponderada por seu respectivo escalar.

2. **Produto por Escalar**: A divergência de um campo vetorial multiplicado por um escalar é o escalar vezes a divergência do campo vetorial. Para um campo vetorial $\mathbf{F}$ e um escalar $a$, temos:

$$
    \nabla \cdot (a\mathbf{F}) = a (\nabla \cdot \mathbf{F})
$$

3. **Divergência de um Produto**: A divergência de um produto de um campo escalar $\phi$ e um campo vetorial $\mathbf{F}$ é dado por:

$$
\nabla \cdot (\phi \mathbf{F}) = \phi (\nabla \cdot \mathbf{F}) + \mathbf{F} \cdot (\nabla \phi)
$$

Este é o análogo vetorial do produto de regra para derivadas no cálculo unidimensional.

4. **Divergência do Rotação**: A divergência do rotacional de qualquer campo vetorial é sempre zero:

$$
   \nabla \cdot (\nabla \times \mathbf{F}) = 0
$$

Esta propriedade é um reflexo do fato de que as linhas de campo do rotacional de um campo vetorial são sempre fechadas, sem início ou fim. Não se preocupe, ainda, já, já, chegaremos ao rotacional.

Essas propriedades são como as leis inabaláveis que governam o comportamento da divergência em sua jornada através dos campos vetoriais.
