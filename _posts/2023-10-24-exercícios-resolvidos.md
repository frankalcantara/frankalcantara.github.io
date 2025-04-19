---
layout: post
title: Exercícios s
author: Frank
categories:
    - Matemática
    - Eletromagnetismo
tags:
    - Matemática
    - Física
    - Eletromagnetismo
image: ""
featured: 2023-10-24T23:42:05.134Z
rating: 0
description: ""
date: 2023-10-24T23:42:05.134Z
preview: ""
keywords: ""
published: false
slug: null
lastmod: 2025-04-19T00:23:52.084Z
---

### Exercício 1

Em uma tarde quente em um bar à beira-mar, um velho pescador conversava com um jovem aprendiz sobre os vetores. "Eles são como o vento, têm direção, sentido e intensidade", disse o pescador. "Imagine dois pontos no mar, e queremos saber a direção e força do vento entre eles". Ele desenhou no chão com um pedaço de carvão os pontos: A(1,2,3) e B(-1,-2,3). "Agora", ele perguntou, "como determinamos o vetor entre esses dois pontos?"

**Solução:** para determinar o vetor entre dois pontos, usamos a seguinte fórmula:

Sejam os pontos $A(x_1,y_1,z_1)$ e $B(x_2,y_2,z_2)$ dois pontos no espaço. O vetor $\vec{V}$ entre eles será dado por:

$$\vec{V} = (x_2 - x_1, y_2 - y_1, z_2 - z_1)$$

Substituindo pelos pontos dados:

$$v = (-1 - 1, -2 - 2, 3 - 3)$$

$$v = (-2, -4, 0)$$

Assim, o vetor entre os pontos $A(1,2,3)$ e $B(-1,-2,3)$ é $\vec{V} = (-2, -4, 0)$.

### Exercício 2

Você é um capitão de um pequeno barco de pesca, perdido em alto mar. Sua bússola, impressa em um plano cartesiano, mostra a direção para o porto seguro como um vetor $\, \vec{a} = (4, 3, -1)$. Este vetor contém a direção e a força dos ventos e correntes que você deve enfrentar. Sua tarefa é simplificar essa informação em um vetor unitário que aponte a direção exata para o porto. Lembre-se, um vetor unitário tem magnitude $1$ e aponta na mesma direção e sentido do vetor original. Utilize suas habilidades em álgebra vetorial para encontrar esse vetor unitário e aponte seu barco para casa.

**Solução:** para encontrar o vetor unitário correspondente ao vetor $\, \vec{a} = (4, 3, -1)$, primeiro precisamos calcular a magnitude do vetor $\, \vec{a}$.

1. Cálculo da Magnitude de $\, \vec{a}$: a magnitude do vetor $\, \vec{a}$ é dada por:

 $$
 |\, \vec{a}| = \sqrt{4^2 + 3^2 + (-1)^2}
 $$

 $$
 |\, \vec{a}| = \sqrt{16 + 9 + 1}
 $$

 $$
 |\, \vec{a}| = \sqrt{26}
 $$

 $$
 |\, \vec{a}| \approx 5.099
 $$

2. Cálculo do Vetor Unitário: para encontrar o vetor unitário $\, \vec{a}$, nós dividimos cada componente do vetor $\, \vec{a}$ pela sua magnitude:

 $$
 \vec{a} = \frac{\, \vec{a}}{|\, \vec{a}|}
 $$

 $$
 \vec{a} = \frac{(4, 3, -1)}{5.099}
 $$

 $$
 \vec{a} \approx (0.784, 0.588, -0.196)
 $$

O vetor unitário correspondente ao vetor $\, \vec{a}$ é aproximadamente $(0.784, 0.588, -0.196)$. Este é o vetor que você deve seguir para apontar seu barco na direção certa para o porto seguro, independente das condições dos ventos e correntes.

Se a amável leitora conseguiu entender de que forma as forças devidas a ventos e correntes estão implícitas na definição do vetor $\, \vec{a}$, já entendeu o que é um vetor unitário e como ele pode ser usado para simplificar as informação contidas em um sistema de forças.

### Exercício 3

Em um antigo mapa de um navegador solitário, as distâncias eram indicadas apenas por unidades, sem definição específica de sua medida, como se fossem passos ou palmos. Naqueles tempos, a precisão não era tão exigente, e os navegadores costumavam confiar em seus instintos e habilidades de observação. Nesse mapa peculiar, o navegador anotou:

1. Um trajeto que começa em seu ponto de partida, marcado como a origem, e vai até um ponto de interesse $A = (-3, 4, 5)$.
2. Um vetor unitário $b$ que, também a partir da origem, aponta na direção de um segundo ponto de interesse, $B$, e é representado por $\vec{b} = \frac{(-2, 1, 3)}{2}$.
3. Ele também fez uma anotação de que a distância entre os dois pontos de interesse, $A$ e $B$, era de 12 unidades. Talvez essa fosse a distância que ele precisava viajar em um dia para chegar ao ponto $B$ antes do anoitecer. Talvez fosse apenas um sonho, um destino que nunca foi percorrido. Não sabemos, mas talvez seja possível determinar as coordenada exatas do ponto $B$ no mapa. Dado essas informações, qual seria a localização exata do ponto $B$ no mapa?

**Solução:** primeiramente, encontramos o vetor unitário $\, \vec{a}$ referente ao vetor que sai da origem e chega no ponto $A$:

$$\, \vec{a} = \frac{\, \vec{a} }{|\, \vec{a}|} = \frac{\, \vec{a} }{\sqrt{x^2 + y^2 + z^2} } = \frac{1}{\sqrt{x^2 + y^2 + z^2} } \, \vec{a} $$

Para o ponto $B$, o vetor unitário é dado por:

$$\vec{B} = \frac{1}{2}k(-2, 1, 3)$$

Nesta equação o termo $k$ é um valor escalar que representa a magnitude de um dos vetores que está na direção do vetor unitário $\vec{b}$. Portanto, ao variar $k$, obtemos todos os possíveis vetores, de diferentes magnitudes, que estão na direção especificada pelo vetor unitário $\frac{1}{2}(-2, 1, 3)$. **como um vetor unitário só determina uma direção e um sentido exite um número infinito de vetores múltiplos de um dado vetor unitário**. Nossa tarefa é encontrar qual destes vetores satisfaz o enunciado. Ou, em outras palavras, $\vec{b}$ é o vetor unitário na direção de todos os vetores múltiplos escalares dele mesmo. Expandindo:

$$\vec{B} = \frac{-2k}{2} \, \vec{a}_x + \frac{k}{2} \, \vec{a}_y + \frac{3k}{2} \, \vec{a}_z$$

A distância entre dois pontos no espaço cartesiano é dada pela fórmula da distância euclidiana. Para lembrar, vamos considerar dois pontos no espaço tridimensional:
$ A(x_1, y_1, z_1) \) e \( B(x_2, y_2, z_2) $.

A distância $D$ entre os pontos $A$ e $B$ é dada por:

$$ D = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2} $$

Esta fórmula é uma extensão direta do teorema de Pitágoras. Se estivéssemos considerando pontos em um plano bidimensional, a fórmula seria:

$$ D = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $$

E, de forma geral, para um espaço $n$-dimensional, a distância é dada por:

$$ D = \sqrt{\sum_{i=1}^{n} (x_{2i} - x_{1i})^2} $$

Onde $x_{1i}$ e $x_{2i}$ são as coordenadas dos pontos nas diferentes dimensões.

Podemos pensar de uma forma um pouco diferente. A distância entre dois pontos no espaço é o comprimento, magnitude, do vetor que liga estes dois pontos. Se a amável leitora considerar os dois pontos originais do enunciado $A(x_1, y_1, z_1)$ e $B(x_2, y_2, z_2)$ no espaço, o vetor que liga esses dois pontos será dado por:

$$ \vec{AB} = (x_2 - x_1, y_2 - y_1, z_2 - z_1) $$

O módulo (ou magnitude) desse vetor será precisamente a distância entre os pontos $A$ e $B$, e será calculado por:

$$ |\vec{AB}| = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2} $$

Esta fórmula coincide com a fórmula da distância euclidiana usada anteriormente. Portanto, a distância entre dois pontos no espaço é equivalente à magnitude do vetor que conecta esses dois pontos. Se substituirmos os valores que temos:

$$\sqrt{(-3 - \frac{2k}{2})^2 + (4 + \frac{k}{2})^2 + (5 + \frac{3k}{2})^2} = 12$$

$$\sqrt{(-3 - k)^2 + (4 + \frac{k}{2})^2 + (5 + \frac{3k}{2})^2} = 12$$

$$(-3 - k)^2 + (4 + \frac{k}{2})^2 + (5 + \frac{3k}{2})^2 = 144 $$

Vamos tratar cada termo desta equação separada e cuidadosamente.

1. Expanda $(-3 - k)^2$ em $ 9 + 6k + k^2 $;

2. Expanda $(4 + \frac{k}{2})^2$:

 Primeiro, $4 + \frac{k}{2}$ se torna $4 + 0.5B$.
 Agora, $(4 + 0.5k)^2$ é: $16 + 4k + 0.25k^2$.

3. Expanda $(5 + \frac{3k}{2})^2$:

 Primeiro, $5 + \frac{3k}{2}$ se torna $5 + 1.5k$.
 Agora, $(5 + 1.5k)^2$ será: $25 + 15k + 2.25k^2$

Combinando todas essas expansões, teremos:

$$k^2 + 6k + 9 + 0.25k^2 + 4k + 16 + 2.25k^2 + 15k + 25 = 144$$

Somando os termos semelhantes:

$$3.5k^2 + 25k + 50 = 144$$

Esta é a equação após a combinação e expansão completa dos termos.

$$3.5k^2 + 24k - 94 = 0$$

Cujas raízes serão:

$$k = -\frac{2}{7} \left( 12 + \sqrt{473} \right) \approx 2.7853$$

$$k = \frac{2}{7} \left( \sqrt{473} - 12 \right)\approx -9.6424$$

São duas soluções possíveis. Se escolhermos a raiz negativa, estaremos dizendo que o ponto $B$ está na direção oposta de $A$:

$$\vec{B} = \frac{-2k}{2} \, \vec{a}_x + \frac{k}{2} \, \vec{a}_y + \frac{3k}{2} \, \vec{a}_z$$

E substituir $B$ por $2.7853$ teremos?

$$\vec{B} = \frac{-2(2.7853)}{2} \, \vec{a}_x + \frac{(2.7853)}{2} \, \vec{a}_y + \frac{3(2.7853)}{2} \, \vec{a}_z$$

$$\vec{B} =-2.7853\, \vec{a}_x + 1.39265 \, \vec{a}_y + 4,1779 \, \vec{a}_z$$

Se estivéssemos no mundo real, em um cenário que incluísse todas as características do ambiente, as forças devidas aos ventos e correntes seriam vetores separados, possivelmente variáveis em função do tempo e da sua posição no espaço. Estes vetores afetariam tanto a direção quanto a velocidade do barco e teriam que ser considerados continuamente para garantir uma rota otimizada até o porto. Contudo, para tornar o problema mais simples e atrativo, podemos considerar que todas as variações imagináveis de ventos e correntes estão representadas nas forças que atuam em cada um dos eixos cartesianos, forçando o barco em uma determinada direção criando o vetor indicado por nossa bússola. O vetor unitário, usado dessa forma estripa estas influências complexas mostrando apenas a direção e sentidos que devem ser seguidos. Assim, permitimos que mesmo o mais inexperiente marinheiro, destes mares matemáticos, perceba a beleza da matemática.

### Exercício 4

Alice é uma engenheira trabalhando no projeto de construção de uma ponte. As forças aplicadas sobre um pilar foram simplificadas até que serem reduzidas a dois vetores: $\vec{F}_1 = 4\, \vec{a}_x + 3\, \vec{a}_y$ e $\vec{F}_2 = -1\, \vec{a}_x + 2\, \vec{a}_y$ a força aplicada ao pilar será o resultado da subtração entre os vetores. Alice precisa saber qual será a força resultante após aplicar uma correção de segurança ao vetor $\vec{F}_2$ multiplicando-o por 2. O trabalho de Alice é definir as características físicas deste pilar, o seu é ajudar Alice com estes cálculos.

**Solução:** começando observando que os dois vetores tem componente $\, \vec{a}_z$ zerado, isto significa que as forças já foram simplificadas até um plano. Depois multiplicamos $\vec{F}_2$ por 2:

$$
2\vec{F}_2 = 2(-1\, \vec{a}_x + 2\, \vec{a}_y) = -2\, \vec{a}_x + 4\, \vec{a}_y
$$

Agora, subtraímos esse novo vetor para encontrar a força resultante $\vec{F}_{\text{res}}$:

$$
\vec{F}_{\text{res}} = \vec{F}_1 - 2\vec{F}_2 = (4\, \vec{a}_x + 3\, \vec{a}_y) - (-2\, \vec{a}_x + 4\, \vec{a}_y)
$$

$$
\vec{F}_{\text{res}} = 6\, \vec{a}_x - \, \vec{a}_y
$$

Portanto, a força resultante após a correção de segurança será $\vec{F}_{\text{res}} = 6\, \vec{a}_x - \, \vec{a}_y$.

### Exercício 5

Larissa é uma física estudando o movimento de uma partícula em um campo elétrico. Ela reduziu o problema a dois vetores representando as velocidades da partícula em um momento específico:
$\vec{V}_1 = 6\, \vec{a}_x - 4\, \vec{a}_y + 2\, \vec{a}_z$ e $\vec{V}_2 = 12\, \vec{a}_x + 8\, \vec{a}_y - 4\, \vec{a}_z$. Larissa precisa qual será a velocidade média da partícula se ele considerar que $\vec{V}_2$ deve ser dividido por $2$ graças ao efeito de uma força estranha ao sistema agindo sobre uma das partículas. Para ajudar Larissa ajude-a a determinar a velocidade média, sabendo que esta será dada pela soma destes vetores após a correção dos efeitos da força estranha ao sistema.

**Solução:** vamos começar fazendo a correção definida no enunciado dividindo $\vec{V}_2$ por $2$:

$$
\frac{\vec{V}_2}{2} = \frac{12\, \vec{a}_x + 8\, \vec{a}_y - 4\, \vec{a}_z}{2} = 6\, \vec{a}_x + 4\, \vec{a}_y - 2\, \vec{a}_z
$$

Agora, adicionamos esse novo vetor para encontrar a velocidade média $\vec{V}_{\text{média}}$:

$$
\vec{V}_{\text{média}} = \vec{V}_1 + \frac{\vec{V}_2}{2} = (6\, \vec{a}_x - 4\, \vec{a}_y + 2\, \vec{a}_z) + (6\, \vec{a}_x + 4\, \vec{a}_y - 2\, \vec{a}_z)
$$

$$
\vec{V}_{\text{média}} = 12\, \vec{a}_x
$$

O problema descrito no enunciado acabou por provocar o cancelamento dos componentes de velocidade nos eixos $y$ e $z$, deixando a velocidade média da partícula apenas no componente $x$.

### Exercício 6

Marcela é uma física experimental realizando um experimento em um laboratório de pesquisas em um projeto para estudar o movimento de partículas subatômicas. As velocidades das partículas $A$ e $B$ são representadas pelos vetores $\vec{v}_A$ e $\vec{v}_B$, definidos por:

$$ \vec{v}_A = -10\, \vec{a}_x + 4\, \vec{a}_y - 8\, \vec{a}_z \, \text{m/s} $$

$$ \vec{v}_B = 8\, \vec{a}_x + 7\, \vec{a}_y - 2\, \vec{a}_z \, \text{m/s} $$

Marcela precisa calcular a velocidade resultante $\vec{v}_R$ das partículas $A$ e $B$ sabendo que neste ambiente os as velocidades das partículas são afetadas por forças provenientes de campos externos que foram modeladas na equação $\vec{v}_R = 3\vec{v}_A - 4\vec{v}_B$. Qual o vetor unitário que determina a direção e o sentido de $\vec{v}_R$ nestas condições?

**Solução:** primeiro, vamos realizar a operação $3\vec{v}_A - 4\vec{v}_B$ para calcular $\vec{v}_R$:

$$
\vec{v}_R = 3\vec{v}_A - 4\vec{v}_B
$$

Agora, substituindo os valores dos vetores \$\vec{v}_A$ e $\vec{v}_B$:

$$
\vec{v}_R = 3(-10\, \vec{a}_x + 4\, \vec{a}_y - 8\, \vec{a}_z) - 4(8\, \vec{a}_x + 7\, \vec{a}_y - 2\, \vec{a}_z)
$$

$$
\vec{v}_R = (-30\, \vec{a}_x + 12\, \vec{a}_y - 24\, \vec{a}_z) - (32\, \vec{a}_x + 28\, \vec{a}_y - 8\, \vec{a}_z)
$$

$$
\vec{v}_R = (-30\, \vec{a}_x + 12\, \vec{a}_y - 24\, \vec{a}_z) + (-32\, \vec{a}_x - 28\, \vec{a}_y - 8\, \vec{a}_z)
$$

Simplificando o resultado da soma:

$$
\vec{v}_R = (-30\, \vec{a}_x - 32\, \vec{a}_x) + (12\, \vec{a}_y - 28\, \vec{a}_y) + (-24\, \vec{a}_z - 8\, \vec{a}_z)
$$

$$
\vec{v}_R = -62\, \vec{a}_x - 16\, \vec{a}_y - 32\, \vec{a}_z
$$

Para encontrar a direção e o sentido de $\vec{v}_R$ em termos de um vetor unitário $\vec{v}_R$, podemos encontrar o vetor unitário dividindo $\vec{v}_R$ pelo seu módulo:

$$
\vec{v}_R = \frac{\vec{v}_R}{ \vert  \vec{v}_R \vert }
$$

$$
 \vert  \vec{v}_R \vert = \sqrt{(-62)^2 + (-16)^2 + (-32)^2}
$$

$$
 \vert  \vec{v}_R \vert = \sqrt{3844 + 256 + 1024}
$$

$$
 \vert  \vec{v}_R \vert = \sqrt{5124}
$$

$$
 \vert  \vec{v}_R \vert = 2\sqrt{1281} \, \text{m/s}
$$

Logo teremos:

$$
\vec{v}_R = \frac{-62\, \vec{a}_x - 16\, \vec{a}_y - 32\, \vec{a}_z}{2\sqrt{1281} }
$$

Portanto, a direção e o sentido de $\vec{v}_R$ em termos de um vetor unitário:

$$
\vec{v}_R = \frac{-62\, \vec{a}_x - 16\, \vec{a}_y - 32\, \vec{a}_z}{2\sqrt{1281} }
$$

### Exercício 7

Tudo é relativo! A amável leitora já deve ter ouvido esta frase. Uma mentira, das mais vis deste nossos tempos. Tudo é relativo, na física! Seria mais honesto. Não existe qualquer documento, artigo, livro, ou entrevista onde [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) tenha dito tal sandice. Ainda assim, isso é repetido a exaustão. Não por nós. Nós dois estamos em busca da verdade do conhecimento. E aqui, neste ponto, entra o conceito de Einstein: as leis da física são as mesmas independente do observador. Isso quer dizer que, para entender um fenômeno, precisamos criar uma relação entre o observador e o fenômeno. Dito isso, considere que você está observando um trem que corta da direita para esquerda seu campo de visão em velocidade constante $\vec{V}_t = 10 \text{km/h}$. Nesse trem, um passageiro atravessa o vagão perpendicularmente ao movimento do trem em uma velocidade dada por $\vec{V}_p = 2 \text{km/h}$. Qual a velocidade deste passageiro para você, que está colocada de forma perfeitamente perpendicular ao movimento do trem?

**Solução** vamos observar os dados apresentados no enunciado:

- $\vec{V}_t$ é a velocidade do trem em relação ao solo, que é $10 \, \text{km/h}$ na direção $x$ como o enunciado não especificou, coloquei o eixo $x$ com o sentido positivo orientado na direção em que o trem está se movendo.
- $\vec{V}_p$ é a velocidade do passageiro em relação ao trem, que é $2 \, \text{km/h}$ na direção $y$. Isto porque o passageiro está se movimentando em uma direção perpendicular ao trem. Coloquei o sentido positivo do eixo $y$ na direção do movimento do passageiro.
- Como não há nenhuma informação no enunciado sobre variação de altura, podemos ignorar o eixo $z$. Esta é uma análise que faremos em duas dimensões.
- $\vec{V}_o$ é a velocidade do observador em relação ao solo, que queremos calcular e pode ser representada como $\vec{V}_o = (o_x\, \vec{a}_x + o_y\, \vec{a}_y + o_z \, \vec{a}_z)$.

Observe que o enunciado garante que o passageiro está se movendo perpendicularmente a direção do movimento do trem. A velocidade do passageiro em relação ao observador que está no solo, perpendicular ao movimento do trem, será a soma da velocidade do trem em relação ao solo com a velocidade do passageiro em relação ao trem:

$$ \vec{V}_{o} = \vec{V}_t + \vec{V}_p $$

Resolvendo para $o_x$ e $o_y$, obtemos:

$$ o_x = 10 \, \text{km/h} $$

$$ o_y = 2 \, \text{km/h} $$

$$ \vec{V}_o = 10 \vec{a}_x + 2 \vec{a}_y $$

A amável leitora deve perceber que no fenômeno descrito no enunciado, tanto o trem quanto o passageiros estão se movendo em direções perpendiculares entre si. Os vetores que representam as duas velocidades têm apenas uma componente. Contudo, visto do solo, o passageiro está se momento segundo um vetor com componentes tanto no eixo $x$ quanto no eixo $y$. Foi esta relatividade na observação do fenômeno que Einstein estudou.

### Exercício 8

**Solução:** vamos tornar o exercício 7 mais interessante: considere que você está observando um trem que corta da direita para esquerda seu campo de visão em velocidade constante $\vec{V}_t = 10 \text{km/h}$ subindo uma ladeira com inclinação de $25^\circ$. Nesse trem, um passageiro atravessa o vagão perpendicularmente ao movimento do trem em uma velocidade dada por $\vec{V}_p = 2 \text{km/h}$. Qual a velocidade deste passageiro para você, que está colocada de forma perfeitamente perpendicular ao movimento do trem?

Para calcular a velocidade do passageiro em relação a você, que está perpendicular ao movimento do trem, consideramos a seguinte situação:

- O trem se move com uma velocidade de magnitude de $10 \, \text{km/h}$ a um ângulo de $25$ graus em relação ao eixo $x$ positivo, no plano $(x,z)$.
- O passageiro se move na direção $y$ positivo com uma velocidade de $2 \, \text{km/h}$.

Primeiro, projetamos a velocidade do trem nos eixos $x$ e $z$ para encontrarmos a representação algébrica desta velocidade:

- A componente $x$ da velocidade do trem $ \vec{V}_{t_x}$ será dada pela projeção desta magnitude no eixo $x$:

 $$ \vec{V}_{t_x} = \vec{V}_t \cdot \cos(25^\circ) \approx 8.83 \, \text{km/h} $$

- A componente $z$ da velocidade do trem $\vec{V}_{t_z}$ será dada por:

 $$ \vec{V}_{t_z} = \vec{V}_t \cdot \sin(25^\circ) \approx 4.15 \, \text{km/h} $$

Logo a velocidade do trem será dada por:

$$ \vec{V}_t = (\vec{V}_{t_x}\, \vec{a}_x + 0 \, \vec{a}_y + \vec{V}_{t_z} \, \vec{a}_z) \text{km/h} $$

Considerando que o passageiro se move na direção $y$ positivo com uma velocidade de $2 \text{km/h}$, a velocidade do passageiro em relação ao observador $\vec{V}_{o}$ será:

$$ \vec{V}_{p} = 0 \, \vec{a}_x + 2 \, \vec{a}_y + 0 \vec{z} \text{km/h}$$

Com estas velocidades podemos calcular a velocidade do passageiro em relação ao observador somando a velocidade do trem em relação ao observador $\vec{V}_t$ e a velocidade do passageiro em relação ao trem $\vec{V}_p$:

$$\vec{V}_o = \vec{V}_{p} + \vec{V}_t $$

Substituindo os valores:

$$ \vec{V}_o = (8.83 \, \vec{a}_x + 2 \, \vec{a}_y + 4.15 \, \vec{a}_z) \, \text{km/h} $$

### Exercício 9

Considere um sistema de referência onde as distâncias são dimensionadas apenas por unidades abstratas, sem especificação de unidades de medida. Nesse sistema, dois vetores são dados. O vetor $\, \vec{a}$ inicia na origem e termina no ponto $P$ com coordenadas $(8, -1, -5)$. Temos também um vetor unitário $\vec{c}$ que parte da origem em direção ao ponto $Q$, e é representado por $\frac{1}{3}(1, -3, 2)$. Se a distância entre os pontos $P$ e $Q$ é igual a 15 unidades, determine as coordenadas do ponto $Q$.

**Solução**: vamos encontrar o vetor distância $\vec{R}$ entre os pontos $P$ e $Q$.

$$\vec{R} = \, \vec{a} - 15\vec{c}$$

As coordenadas do vetor $\, \vec{a}$ são $(8, -1, -5)$ e as do vetor unitário $\vec{c}$ são $\frac{1}{3}(1, -3, 2) = (\frac{1}{3}, -1, \frac{2}{3})$.

Substituindo na equação, temos:

$$\vec{R} = (8, -1, -5) - 15 \left(\frac{1}{3}, -1, \frac{2}{3}\right)$$

$$\vec{R} = (8, -1, -5) - (5, 15, 10)$$

$$\vec{R} = (3, -16, -15)$$

Agora, vamos usar o vetor $\vec{R}$ para encontrar as coordenadas do ponto $Q$. O vetor $\vec{R}$ tem origem no ponto $Q$ e aponta para o ponto $P$, então para encontrar $Q$ precisamos fazer:

$$\text{Coordenadas de } Q = \text{Coordenadas de } P - \vec{R}$$

$$\text{Coordenadas de } Q = (8, -1, -5) - (3, -16, -15)$$

$$\text{Coordenadas de } Q = (5, 15, 10)$$

Isso só é possível porque tanto as coordenadas de $Q$ quanto as coordenadas de $P$ são, na verdade, os componentes dos Vetores Posição da origem até $P$ e $Q$, respectivamente. Desta forma, as coordenadas do ponto $Q$ são $(5, 15, 10)$.

### Exercício 10

Considere os pontos $\mathbf{P}$ e $\mathbf{Q}$ localizados em $(1, 3, 2)$ e $(4, 0, -1)$, respectivamente. Calcule: (a) O vetor posição $\vec{P}$; (b) O vetor distância de $P$ para $Q$, $\vec{PQ}$; (c) A distância entre $P$ e $Q$; (d) Um vetor paralelo a $\vec{PQ}$ com magnitude de 10.

**Solução:**

(a) O vetor posição $\vec{P}$ é simplesmente o vetor que vai da origem, $\mathbf{O}$, até o ponto $\mathbf{P}$. Ele pode ser representado como:

$$
\vec{P} = P-O = ((1-0), (3-0), (2-0)) = (1,3,2)
$$

(b) O vetor distância de $\mathbf{P}$ para $\mathbf{Q}$, $\vec{PQ}$, é calculado como $\vec{Q} - \vec{P}$. Os vetores posição para $\mathbf{P}$ e $\mathbf{Q}$ são $(1, 3, 2)$ e $ (4, 0, -1)$ respectivamente. Logo:

$$
\vec{PQ} = (4, 0, -1) - (1, 3, 2) = (3, -3, -3)
$$

(c) A distância, $\vec{D}$, entre $\mathbf{P}$ e $\mathbf{Q}$ será o módulo do vetor $\vec{PQ}$, calculado como:

$$
\vec{D} = \vert  \vec{PQ} \vert = \sqrt{3^2 + (-3)^2 + (-3)^2} = \sqrt{9 + 9 + 9} \approx 5.1962
$$

(d) Um vetor paralelo a $\vec{PQ}$ com magnitude de $10$ pode ser encontrado ao normalizar $\vec{PQ}$:

$$
\vec{pq} = \frac{\vec{PQ}}{ \vert  \vec{PQ} \vert  } = \frac{(3, -3, -3)}{\sqrt{27}}
$$

$$
\vec{pq} = (\frac{3}{\sqrt{27}}, -\frac{3}{\sqrt{27}}, -\frac{3}{\sqrt{27}})
$$

Agora multiplicamos pela magnitude desejada de $1$:

$$
\vec{v} = 10 \vec{pq}
$$

$$
\vec{v} = (\frac{30}{\sqrt{27}}, -\frac{30}{\sqrt{27}}, -\frac{30}{\sqrt{27}})
$$

$$
\vec{v} \approx (5.7735, -5.7735, -5.7735)
$$

### Exercício 11

Em um novo projeto de engenharia civil para a construção de uma estrutura triangular inovadora, foram demarcados três pontos principais para as fundações. Esses pontos, determinados por estudos topográficos e geotécnicos, foram identificados como $\mathbf{A}(4, 0, 3)$, $\mathbf{B}(-2, 3, -4)$ e $\mathbf{C}(1, 3, 1)$ em um espaço tridimensional utilizando o Sistema de Coordenadas Cartesianas. A equipe de engenheiros precisa compreender a relação espacial entre esses pontos, pois isto impacta diretamente na distribuição das cargas e na estabilidade da estrutura.

Seu desafio será determinar o o ângulo $\theta_{BAC}$ entre estes vetores para a análise estrutural, pois determina o direcionamento das forças na fundação.

**Solução:** para encontrar $\vec{AB}$:

$$
\vec{AB} = \mathbf{B} - \mathbf{A} = ((-2 - 4), (3 - 0), (-4 - 3)) = (-6, 3, -7)
$$

Para encontrar $\vec{AC}$:

$$
\vec{AC} = \mathbf{C} - \mathbf{A} = ((1 - 4), (3 - 0), (1 - 3)) = (-3, 3, -2)
$$

A partir da equação analítica do Produto Escalar podemos dizer que o cosseno do ângulo entre dois vetores é dado por:

$$
\cos(\theta_{BAC}) = \frac{\vec{AB} \cdot \vec{AC}}{ \vert  \vec{AB} \vert  \vert  \vec{AC} \vert }
$$

Onde o Produto Escalar será dada por:

$$
\vec{AB} \cdot \vec{AC} = (-6 \times -3) + (3 \times 3) + (-7 \times -2) = 18 + 9 + 14 = 41
$$

A magnitude de $\vec{AB}$ será:

$$
 \vert  \vec{AB}| = \sqrt{(-6)^2 + 3^2 + (-7)^2} = \sqrt{36 + 9 + 49} = \sqrt{94}
$$

A magnitude de $\vec{AC}$ é:

$$
 \vert  \vec{AC}| = \sqrt{(-3)^2 + 3^2 + (-2)^2} = \sqrt{9 + 9 + 4} = \sqrt{22}
$$

Sendo assim:

$$
\cos(\theta_{BAC}) = \frac{41}{\sqrt{94 \times 22}}
$$

$$
\theta_{BAC} = \cos^{-1} \left( \frac{41}{\sqrt{94 \times 22}} \right) \approx \cos^{-1} 0.901589
$$

$$
\theta_{BAC} \approx 25.63^\circ
$$

### Exercício 12

Considere o vetor $\vec{F} = (x, y, z)$ perpendicular ao vetor $\vec{G} = (2, 3, 1)$. Sabendo que $\vec{F} \cdot \vec{F} = 9$. Determine os componentes que definem o vetor $\vec{F}$.

**Solução:** primeiro vamos determinar a relação entre os vetores $\vec{F}$ e $\vec{G}$ usando o Produto Escalar. Sabemos que se dois vetores são perpendiculares (ortogonais) entre si, o Produto Escalar entre eles é zero.

$$\vec{F} \cdot \vec{G} = x \times 2 + y \times 3 + z \times 1 = 0$$

Resultando em:

$$2x + 3y + z = 0 \quad \text{...(i)}$$

A segunda informação, $\vec{F} \cdot \vec{F} = 9$, indica que a magnitude ao quadrado de $\vec{F}$ é 9.

$$\vec{F} \cdot \vec{F} = x^2 + y^2 + z^2 = 9 \quad \text{...(ii)}$$

Utilizando a equação (i), podemos expressar $z$ em função de $x$ e $y$:

$$z = -2x - 3y \quad \text{...(iii)}$$

Substituindo a expressão para $z$ da equação (iii) na equação (ii):

$$x^2 + y^2 + (-2x - 3y)^2 = 9$$

Expandido, obtemos:

$$x^2 + y^2 + 4x^2 + 9y^2 + 12xy = 9$$

Combinando termos semelhantes, chegamos à seguinte expressão:

$$5x^2 + 10y^2 + 12xy - 9 = 0$$

A partir das equações fornecidas:

1) $2x + 3y + z = 0 \quad \text{...(i)}$

2) $x^2 + y^2 + z^2 = 9 \quad \text{...(ii)}$

3) $z = -2x - 3y \quad \text{...(iii)}$

Substituindo $z$ de (3) em (2):

$$x^2 + y^2 + (-2x - 3y)^2 = 9$$

Expandindo e agrupando os termos:

$$5x^2 + 10y^2 + 12xy = 9$$

$$5x^2 + 12xy + 10y^2 - 9 = 0$$

Vamos expressar $y$ em termos de $x$ a partir de (1):

$$y = \frac{-2x - z}{3}$$

Usando (3) para $z$, obtemos:

$$y = \frac{-2x + 2x + 3y}{3}$$

$$y = \frac{y}{3}$$

Isso implica que $y = 0$.

Substituindo $y = 0$ em (3), obtemos $z = -2x$.

Substituindo $y = 0$ e $z = -2x$ em (2):

$$5x^2 = 9$$

$$x^2 = \frac{9}{5}$$

$$x = \sqrt{\frac{9}{5} } = \frac{3}{\sqrt{5} }$$

Como $x$ pode ser positivo ou negativo:

$$x = \pm \frac{3}{\sqrt{5} }$$

$$z = \mp \frac{6}{\sqrt{5} }$$

As soluções para $\vec{F}$ são:

$$\vec{F_1} = \left(\frac{3}{\sqrt{5} }, 0, -\frac{6}{\sqrt{5} }\right)$$

$$\vec{F_2} = \left(-\frac{3}{\sqrt{5} }, 0, \frac{6}{\sqrt{5} }\right)$$

### Exercício 13

Calcule o Produto Escalar de $\vec{C} = \vec{A} - \vec{B}$ com ele mesmo.

**Solução:** primeiro, expandimos $\vec{C} \cdot \vec{C}$ usando a definição de $\vec{C}$ dadas no enunciado e as propriedades do produto escalar.

$$
\vec{C}\cdot \vec{C} = (\vec{A} - \vec{B}) \cdot (\vec{A} - \vec{B})
$$

O produto escalar é distributivo, o que nos permite expandir para:

$$
\vec{C}\cdot \vec{C} = \vec{A} \cdot \vec{A} - \vec{A} \cdot \vec{B} - \vec{B} \cdot \vec{A} + \vec{B} \cdot \vec{B}
$$

Note que $\vec{A} \cdot \vec{B}$ e $\vec{B} \cdot \vec{A}$ são iguais devido à propriedade comutativa do produto escalar.

Vamos converter os termos para magnitudes e ângulos usando a relação $\vec{A} \cdot \vec{A} = \vec{A}^2$ e $\vec{A} \cdot \vec{B} = \vert  \vec{A} \vert  \vert  \vec{B} \vert  \cos \theta$, onde $\theta$ é o ângulo entre $\vec{A}$ e $\vec{B}$.

$$
\vec{C}^2 = \vec{A}^2 + \vec{B}^2 - 2 \vert  \vec{A} \vert  \vert  \vec{B} \vert  \cos \theta
$$

Substituímos $\vec{A} \cdot \vec{B}$ e $\vec{B} \cdot \vec{A}$ por $ \vert  \vec{A} \vert  \vert  \vec{B} \vert  \cos \theta$ e simplificamos a expressão para obter $\vec{C}^2$.

Chegamos à expressão final para $\vec{C}^2$ em termos das magnitudes de $\vec{A}$ e $\vec{B}$ e do ângulo $\theta$ entre eles.

$$ \vec{C}^2 = \vec{A}^2 + \vec{B}^2 - 2 \vert  \vec{A} \vert  \vert  \vec{B} \vert  cos \theta $$

E esta é a forma vetorial da _Lei dos Cossenos_. 

A Lei dos Cossenos é uma equação usada em trigonometria para relacionar os lados de um triângulo com um de seus ângulos. Ela é particularmente útil para encontrar um lado de um triângulo quando conhecemos os outros dois lados e o ângulo entre eles. A Lei dos Cossenos é geralmente expressa da seguinte forma para um triângulo com lados de comprimentos $a$, $b$, e $c$, e ângulo $\theta$ oposto ao lado $c$:

$$
c^2 = a^2 + b^2 - 2ab \cos(\gamma)
$$

Esta fórmula pode ser rearranjada para resolver qualquer um dos lados ou ângulos do triângulo, e também pode ser adaptada para as outras duas versões, correspondendo aos outros dois ângulos do triângulo:

$$
a^2 = b^2 + c^2 - 2bc \cos(\alpha)
$$

$$
b^2 = a^2 + c^2 - 2ac \cos(\beta)
$$

A Lei dos Cossenos é uma extensão do [Teorema de Pitágoras](https://en.wikipedia.org/wiki/Pythagorean_theorem) para triângulos que não são retângulos. Quando o ângulo $\theta$ é de $90^\circ$, o $\cos(90^\circ) = 0$, e a equação se reduz ao Teorema de Pitágoras:

$$
c^2 = a^2 + b^2
$$

### Exercício 14

Considerando a equação analítica do Produto escalar, $\vec{A}\cdot \vec{B} = \vert  \, \vec{a} \vert  \vert \vec{B} \vert  cos(\theta)$, e a equação analítica do Produto Vetorial, $\vec{A} \times \vec{A} = \vert  \vec{A} \vert  \vert  \vec{B} \vert  sen(\theta_{AB})$ prove que tanto o Produto Escalar quanto o Produto Vetorial são operações distributivas.

**Solução:** começando pela prova de Distributividade do Produto Escalar, vamos considerar três vetores arbitrários $\vec{A}$, $\vec{B}$, e $\vec{C}$. Queremos mostrar que:

$$
\vec{A} \cdot (\vec{B} + \vec{C}) = \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C}
$$

Primeiro, vamos expandir $\vec{B} + \vec{C}$ em termos de suas componentes:

$$
\vec{B} + \vec{C} = (B_x + C_x) \, \vec{a}_x + (B_y + C_y) \, \vec{a}_u + (B_z + C_z) \, \vec{a}_z
$$

Agora, podemos calcular $\vec{A} \cdot (\vec{B} + \vec{C})$ usando as componentes dos vetores:

$$
\vec{A} \cdot (\vec{B} + \vec{C}) = A_x (B_x + C_x) + A_y (B_y + C_y) + A_z (B_z + C_z)
$$

$$
\vec{A} \cdot (\vec{B} + \vec{C}) = A_x B_x + A_x C_x + A_y B_y + A_y C_y + A_z B_z + A_z C_z
$$

$$
\vec{A} \cdot (\vec{B} + \vec{C}) = (A_x B_x + A_y B_y + A_z B_z) + (A_x C_x + A_y C_y + A_z C_z)
$$

$$
\vec{A} \cdot (\vec{B} + \vec{C}) = \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C}
$$

Mostramos que $\vec{A} \cdot (\vec{B} + \vec{C}) = \vec{A} \cdot \vec{B} + \vec{A} \cdot \vec{C}$, provando assim que o produto escalar é distributivo.

Para provar que o produto vetorial é distributivo vamos, novamente, considerar três vetores arbitrários $\vec{A}$, $\vec{B}$, e $ \vec{C}$. Queremos mostrar que:

$$
\vec{A} \times (\vec{B} + \vec{C}) = \vec{A} \times \vec{B} + \vec{A} \times \vec{C}
$$

Começamos expandindo $\vec{B} + \vec{C}$ em termos de suas componentes:

$$
\vec{B} + \vec{C} = (B_x + C_x) \vec{a}_x + (B_y + C_y) \vec{a}_y + (B_z + C_z) \vec{a}_z
$$

Agora, vamos calcular $\vec{A} \times (\vec{B} + \vec{C})$ usando as componentes dos vetores. O produto vetorial em termos das componentes é dado por:

$$
\vec{A} \times (\vec{B} + \vec{C}) = (A_y (B_z + C_z) - A_z (B_y + C_y)) \,\vec{a}_x + (A_z (B_x + C_x) - A_x (B_z + C_z)) \, \vec{a}_y \\ + (A_x (B_y + C_y) - A_y (B_x + C_x)) \, \vec{a}_z
$$

Expandindo, obtemos:

$$
\vec{A} \times (\vec{B} + \vec{C}) = \vec{a}_x (A_y B_z + A_y C_z - A_z B_y - A_z C_y) + \vec{a}_y (A_z B_x + A_z C_x - A_x B_z - A_x C_z) \\ + \vec{a}_z (A_x B_y + A_x C_y - A_y B_x - A_y C_x)
$$

Agrupando os termos, temos:

$$
\vec{A} \times (\vec{B} + \vec{C}) = (A_y B_z - A_z B_y) \hat{a}_x + (A_z B_x - A_x B_z) \vec{a}_y + (A_x B_y - A_y B_x) \vec{a}_z + \\ (A_y C_z - A_z C_y) \vec{a}_x + (A_z C_x - A_x C_z) \vec{a}_y + (A_x C_y - A_y C_x) \vec{a}_z
$$

$$
\vec{A} \times (\vec{B} + \vec{C}) = \vec{A} \times \vec{B} + \vec{A} \times \vec{C}
$$

Portanto, mostramos que $ \vec{A} \times (\vec{B} + \vec{C}) = \vec{A} \times \vec{B} + \vec{A} \times \vec{C} $, provando assim que o Produto Vetorial é distributivo.

