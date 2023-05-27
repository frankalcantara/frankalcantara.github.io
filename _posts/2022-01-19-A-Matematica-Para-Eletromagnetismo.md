---
layout: post
title: "A Fórmula da Atração: a matemática do Eletromagnetismo"
author: Frank
categories: [Matemática, Eletromagnetismo]
tags: [Matemática, Física, Eletromagnetismo]
image: assets/images/eletromag1.jpg
description: "Entenda como a matemática fundamenta o eletromagnetismo e suas aplicações práticas em um artigo acadêmico destinado a estudantes de ciência e engenharia."
---

Imagem de <a href="https://pixabay.com/users/asimina-1229333/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2773167">Asimina Nteliou</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2773167">Pixabay</a>

<blockquote>"Este é um trabalho em andamento. Escrevo quando posso e posto em seguida. Falta Muito!</blockquote>
<blockquote>"Nesta longa vida eu aprendi que toda a nossa ciência se comparada com a realidade é primitiva e infantil. Ainda assim, <b>é a coisa mais preciosa que temos</b>." Albert Einstein</blockquote>
<p>
Tudo que nos cerca é devido ao Eletromagnetismo. Tudo! Trata-se do estudo das forças fundamentais que definem nosso Universo. Desde a sua própria existência até o dispositivo que você está usando, tudo está intimamente relacionado aos efeitos do campo elétrico, do campo magnético e das interações entre eles.
</p>
<p>
No final do Século XIX, James Clerk Maxwell sintetizou a relação entre estas forças em um conjunto de equações matemáticas que explicam matematicamente como o eletromagnetismo afeta o espaço e a matéria. O trabalho de Maxwell, extraordinário de qualquer ponto de vista, surpreende pela simplicidade e beleza. Entretanto, requer um profundo conhecimento tando de física quanto de matemática para seu entendimento. Neste artigo, exploraremos a base matemática necessária ao conhecimento do eletromagnetismo, sem nos preocuparmos com a física onde esta matemática será aplicada.
</p>
<p>
Considere este texto como o primeiro passo da sua jornada. Tenha coragem,calma e perseverança. <b>Tudo que posso dizer é que vai doer e não será rápido</b>. Este artigo destina-se a estudantes de ciência e engenharia, mas pode ser relevante para qualquer pessoa interessada em aprender sobre a matemática como ferramenta para o Eletromagnetismo e ter um vislumbre da complexidade do Universo.
</p>

<h3>Álgebra Linear</h3>
<h4>Vetores, tudo depende dos vetores</h4>

<p>
Vetores são abstrações matemáticas usadas para o entendimento de grandezas que precisam das informações de magnitude, direção e sentido para sua definição. Estas grandezas, que chamamos de grandezas vetoriais, são a base do estudo do eletromagnetismo. Geometricamente, um vetor será representado por uma seta, com origem, destino e comprimento. Na física moderna usamos os vetores como definido por <a href="https://en.wikipedia.org/wiki/Paul_Dirac" target="_blanck">Dirac (1902-1984) (1839–1903)</a>, que chamamos de Vetores Ket, ou simplesmente ket. Neste artigo utilizaremos a forma diádica de representação vetorial como definida por <a href="https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs" target="_blanck">Willard Gibbs (1839–1903)</a> no final do Século XIX, que é mais adequada ao estudo do Eletromagnetismo Clássico. Neste artigo representaremos os vetores \(A, B, C, ...\) como elementos de um espaço vetorial \(V\) em três dimensões desde que este espaço satisfaça as seguintes condições: 
<ol>
    <li>o espaço vetorial \(V\) seja fechado em relação a adição: para cada par de vetores \(A\) e \(B\) pertencentes a \(V\) existe um, e somente um, vetor \(C\) que representa a soma de \(A\) e \(B\). Tal que: \(\exists A \in V\space\space\space \wedge\space\space\space \exists B \in V \space\space|\space\space \exists A+B=C \in V\).</li>
    <li>A adição seja associativa: \((A+B)+C = A+(B+C)\)</li>
    <li>Existe um vetor zero: a adição deste vetor zero a qualquer vetor \(A\) resulta em \(A\), tal que: \(\forall A \in V \space\space \exists 0 \in V \space\space | \space\space 0+A=A\).</li>
    <li>Existe um vetor negativo \(-A\) de forma que a soma de um vetor com seu vetor negativo, ou oposto, resulta no vetor zero. Tal que: \(\exists -A \in V \space\space | \space\space -A+A=0 \).</li>
    <li>o espaço vetorial \(V\) seja fechado em relação a multiplicação por escalar: para todo e qualquer elemento \(c\) do conjunto dos números complexos \(\Bbb{C}\) multiplicado por um vetor \(A\) do espaço vetorial \(V\) existe um, e somente um vetor \(cA\) que também pertence ao espaço vetorial \(V\). Tal que: \(\exists c \in \Bbb{C} \space\space \wedge \space \space \exists A \in V \space\space |\space\space \exists cA \in V\).</li>
    <li>Existe um escalar neutro 1: tal que a multiplicação de qualquer vetor \(A\) por \(1\) resulta em \(A\). Ou seja: \(\exists 1 \in \Bbb{R} \space\space \wedge \space\space \exists A \in V \space\space | \space\space 1A = A\).</li>
</ol> 
</p>
<p>
O conjunto dos números reais \(\Bbb{R}\) é um subconjunto do conjunto dos números imaginários \(\Bbb{C}=\{a+bi\space\space a.b \in \Bbb{R}\}\), e representará todos os números imaginários de parte imaginária é igual a zero, \(\Bbb{R}=\{a+bi \space\space|\space\space a.b \in \Bbb{R}\wedge b=0\}\). A representação algébrica dos vetores definida por <a href="https://en.wikipedia.org/wiki/Josiah_Willard_Gibbs" target="_blanck">Willard Gibbs (1839–1903)</a> requer a existência de componentes vetoriais para cada um dos eixos do sistema de coordenadas escolhido. No caso do Sistema de Coordenadas Cartesianas, espaço formado pelas coordenadas \((x,y.z)\), teremos três vetores que formarão esta base de representação. Estes vetores serão os vetores unitários \((a_x,a_y,a_z)\), que estarão orientados segundo os eixos cartesianos com comprimento unitário.
</p>
<h5>Vetor Unitário</h5>
<p>Um vetor \(B\) qualquer tem magnitude, direção e sentido. A magnitude, também chamada de intensidade, ou módulo, será representada por \(|B|\). Definiremos um vetor unitário \(a_B\) como:
$$a_B=\frac{B}{|B|}$$
Um vetor unitário \(a_B\) é um vetor que tem a mesma direção e sentido de \(B\) com magnitude \(1\) logo \(|a_B|=1\).
</p>
<p>
 Em um sistema de coordenadas tridimensionais e ortogonais podemos expressar qualquer vetor na forma da soma dos seus componentes unitários ortogonais. Estes components são relativos ao sistema de coordenadas que escolhemos e escolhemos o sistema de coordenadas mais adequado a solução do problema específico que estamos estudando. Usando o Sistema de Coordenadas Cartesianas a representação de um vetor \(B\) segundo seus componentes unitários e ortogonais será dada por:
$$B=B_xa_x+B_ya_y+B_za_z$$
Onde, \(B_x,B_y,B_z\) representam os fatores que devem multiplicar os vetores unitários \(a_x, a_y, a_z\) de forma que a soma destes vetores represente o vetor \(B\) no espaço \(\Bbb{R}^3\). Ao longo deste artigo vamos chamar \(B_x,B_y,B_z\) de componentes vetoriais nas direções \(x,y,z\), ou de projeções de \(B\) nos eixos \(x,y,z\). As direções \(x,y,z\) foram escolhidas quando optamos pelo Sistema de Coordenadas Cartesianas. Estas direções, ou eixos, do Sistema de Coordenadas Cartesianas implicam na existência dos vetores unitários \(a_x, a_y, a_z\) um para cada eixo ortogonal todos com módulo \(1\). Se utilizarmos sistemas de coordenadas diferentes os vetores unitários deverão ser adequados aos eixos deste sistema. Assim teremos: 
 $$B=B_xa_x+B_ya_y+B_za_z$$
 $$B=B_ra_r+B_\phi a_\phi+B_za_z$$
 $$B=B_ra_r+B_\phi a_\phi+B_\theta a_\theta$$
 Respectivamente para os sistemas de coordenadas cartesianas, cilíndricas e esféricas. Não é raro que, uma vez que o sistema de coordenadas tenha sido definido os vetores sejam representados apenas por seus componentes. Desta forma, no Sistema de Coordenadas Cartesianas \(B=3a_x+a_y-a_z\) pode ser representado apenas por \(B=(3,1,-1)\).
 </p>
 <p>
 Quando representamos um vetor por seus componentes ortogonais, podemos calcular sua magnitude utilizando os fatores multiplicadores de cada componente. Assim, dado o vetor \(B=B_xa_x+B_ya_y+B_za_z\) sua magnitude será dada por \(|B|=\sqrt{B_x^2+B_y^2+B_z^2}\). Desta forma poderemos encontrar o vetor unitário de \(B\) por: 
 $$a_B=\frac{B}{|B|}= \frac{B_xa_x+B_ya_y+B_za_z}{\sqrt{B_x^2+B_y^2+B_z^2}}$$
</p>
<p class="exp">
Exemplo: calcule o vetor unitário \(a_A\) do vetor \(A=a_x-3a_y+2a_z\).
$$a_A=\frac{A_xa_x+A_ya_y+A_za_z}{\sqrt{A_x^2+A_y^2+A_z^2}}$$
$$a_A=\frac{a_x-3a_y+2a_z}{\sqrt{1^2+(-3)^2+2^2}}=\frac{a_x-3a_y+2a_z}{3,7416}$$
$$a_A=0,2672a_x-0,8018a_y+0,5345a_z$$
</p>
<h5>Multiplicação por Escalar</h5>
<p>
Um escalar é uma grandeza que não precisa de direção e sentido. A multiplicação de um vetor por um escalar implica na multiplicação de cada um dos componentes desse vetor por este escalar. Os escalares que usaremos, neste artigo, serão elementos do conjunto dos números reais \(\Bbb{R}\), mas, como vimos antes, estes escalares são definidos como elementos do conjunto dos números complexos \(\Bbb{C}\) que utilizamos na definição do espaço vetorial \(V\) que definimos para nossos vetores.
</p>
<p class="exp">
Exemplo: considere o vetor \(V=2a_x+4a_y-a_z\) e calcule \(3,3V\) e \(V/2\)
$$3.3V=(3,3)(2)a_x+(3,3)(4)a_y+(3,3)(-1)a_z=6,6a_x+13,2a_y-3,3a_z$$
$$\frac{V}{2}=(\frac{1}{2})(2)a_x+(\frac{1}{2})(4)a_y+(\frac{1}{2})(-1)a_z = a_x+2a_y-\frac{1}{2}a_z$$
</p>
<p>
A Multiplicação por Escalar é comutativa, associativa, distributiva fechada em relação ao zero e ao elemento neutro. Assim se tivermos os escalares \(m\) e \(n\) e os vetores \(A\) e \(B\) teremos:
$$mA=Am$$
$$m(nA) = (mn)A$$
$$m(A+B) = mA+mB$$
$$(A+B)n = nA+nB$$
$$1A=A$$
$$0A=0$$
</p>
<h5>Vetor Oposto</h5>
<p>Chamamos de Vetor Oposto a \(A\) ao vetor que tem a mesma intensidade, a mesma direção e sentido oposto ao sentido de \(A\). Um Vetor Oposto é o resultado da multiplicação de um vetor pelo escalar \(-1\). Logo:
$$-1A = -A$$
</p>
<h5>Adição e Subtração de Vetores</h5>
<p>Vetores podem ser somados, ou subtraídos, geometricamente por meio da regra do paralelogramo. Algebricamente dizemos que o espaço vetorial \(V\) é fechado em relação a soma. A soma de vetores é feita componente a componente. Assim, se considerarmos os vetores \(A\) e \(B\) poderemos encontrar um vetor \(C\) que seja a soma \(A+B\) por: 
$$C=A+B=(A_x a_x+A_y a_y+A_z a_z)+(B_x a_x+B_y a_y+B_z a_z)$$
$$C=A+B=(A_x+B_x)a_x+(A_y+B_y)a_y+(A_y+B_y)a_z$$
</p>
<p class="exp">
Exemplo: se \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) calcule \(C=A+B\).
$$C=A+B=(5a_x-3a_y+a_z)+(1a_x+4a_y-7a_z)=(5+1)a_x+(-3+4)a_y+(1-7)a_z $$
$$C= 6a_x+a_y-6a_z$$
</p>
<p>A Subtração será uma soma em que o segundo operando será o vetor oposto do operando original. Assim:
$$C=A-B=A+(-B)=A+(-1B)$$
</p>
<p class="exp">
Exemplo: considere \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) e calcule \(C=A-B\).
$$C=A-B=(5a_x-3a_y+a_z)+(-1(1a_x+4a_y-7a_z))$$
$$C=A-B=(5a_x-3a_y+a_z)+(-1a_x-4a_y+7a_z)=4a_x-7a_y+8a_z$$
</p>
<p>Tanto a adição quanto a subtração de vetores são Comutativas, Associativas e Distributivas. Assim, considerando os vetores \(A\), \(B\) e \(C\) e o escalar \(m\), teremos:
$$A+B=B+A$$
$$A+(B+C)=(A+B)+C$$
$$m(A+B)=mA+mC$$
</p>
<h5>Vetores Distância e Ponto</h5>
<p>Em qualquer sistema de coordenadas vamos chamar de vetor ponto, ou vetor posição, \(R\). Ao vetor que liga a origem do sistema de coordenadas a um ponto \(P\) qualquer no espaço. De tal forma que \(R_p\) será dado por:
$$R_P = (P_x-0)a_x+(P_y-0)a_y+(P_z-0)a_z = P_x a_x+P_y a_y+P_z a_z$$
Ou seja, o vetor ponto é um vetor que indica, no espaço, onde um determinado ponto está. 
</p>
<p>Ainda, em qualquer sistema de coordenadas, o vetor distância, \(D\), é o vetor que mede o deslocamento entre dois pontos no espaço.

<p class="exp">
Exemplo: considerando que \(P\) esteja no ponto \((3,2,-1)\) e \(Q\) esteja no ponto \((1,-2,3)\), o vetor distância \(D_{pq}\) será dado por:
$$D_{pq} = R_p - R_q$$
Logo: 
$$D_{pq} = (P_x-Q_x)a_x+(P_y-Q_y)a_y+(P_z-Q_z)a_z $$
$$D_{pq} = (3-1)a_x+(3-(-2))a_y+((-1)-3)a_z$$
$$D_{pq} = 2a_x+5a_y-4a_z$$
</p>
<p class="exp">
Exemplo: Dados os pontos \(P_1(4,4,3)\), \(P_2(-2,0,5)\) e \(P_3(7,-2,1)\). Especifique o vetor \(A\) que se estende da origem até o ponto \(P_1\). Determine um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos \(P_1\) e \(P_2\). Calcule o perímetro do triângulo formado pelos pontos \(P_1\), \(P_2\) e \(P_3\).
<br><br>
O vetor \(A\) será o vetor posição do ponto \(P_1(4,3,2)\). 
$$A = 4a_x+4a_y+3a_z$$
Para determine um vetor unitário que parte da origem e atinge o ponto médio do segmento de reta formado pelos pontos \(P_1\) e \(P_2\) precisaremos encontrar este ponto médio \(P_M\). 
$$P_M=\frac{P_1+P_2}{2} =\frac{(4,4,3)+(-2,0,5)}{2}$$
$$P_M=\frac{(2,4,8)}{2} = (1, 2, 4)$$
$$P_M=1a_x+2a_y+4a_z)$$
Para calcular o vetor unitário na direção do vetor \(P_M\) teremos:
$$a_{P_M}=\frac{(1, 2, 4)}{|(1, 2, 4)|} = \frac{(1, 3, 4)}{\sqrt{1^2+2^2+4^2}} =0.22a_x+0.45a_y+0.87a_z$$
Finalmente, para calcular o perímetro do triângulo formado por \(P_1(4,4,3)\), \(P_2(-2,0,5)\) e \(P_3(7,-2,1)\) precisaremos somar os módulos dos vetores distância ente \(P_1(4,3,2)\) e \(P_2(-2,0,5)\), \(P_2(-2,0,5)\) e \(P_3(7,-2,1)\) e \(P_3(7,-2,1)\) e \(P_1(4,3,2)\).
$$|P_1P_2| =|(4,4,3)-(-2,0,5)|=|(6,4,-2)|=\sqrt{6^2+4^2+2^2}=7,48$$
$$|P_2P_3| =|(-2,0,5)-(7,-2,1)|=|(-9,2,-4)|=\sqrt{9^2+2^2+4^2}=10,05$$
$$|P_3P_1| =|(7,-2,1)-(4,4,3)|=|(3,-6,-2)|=\sqrt{3^2+6^2+6^2}=7$$
Sendo assim o perímetro será: 
$$|P_1P_2|+|P_2P_3|+|P_3P_1| =7,48+10,05+7=24.53$$
</p>
<h5>Produto Escalar</h5>
<p>
O Produto Escalar é uma operação entre dois vetores que resulta em um valor escalar. Um valor escalar é um número do conjuntos dos números reais \(\Bbb{R}\) representa uma quantidade invariante em todas as transformações rotacionais possíveis. Números escalares representam, na física, grandezas que não precisam das informações de direção e sentido para seu entendimento. Observe que o componentes ortogonais e individuais da representação algébrica de um vetor não são escalares. Estes componentes são sensíveis as operações de transformação vetorial, como veremos. 
</p>
<p>
Dados os vetores \(A\) e \(B\), <b>o produto escalar entre \(A\) e \(B\) resulta em uma quantidade escalar</b> e será representado por \(A\cdot B\). Trigonometricamente o produto escalar entre \(A\) e \(B\) será dado por: 
$$A\cdot B = |A||B|cos(\theta_{AB})$$
O que corresponde a projeção do vetor \(A\) em \(B\). O produto escalar entre dois vetores \(A\) e \(B\) é o produto entre o produto das magnitudes destes vetores e o cosseno do menor ângulo entre eles. Algebricamente, se \(A=A_x a_x+A_y a_y+A_z a_z\) e \(B=B_x a_x+B_y a_y+B_z a_z\) então teremos:
$$A\cdot B = A_x B_x+A_y B_y+A_z B_z$$
</p>
<p class="exp">
Exemplo: dados os vetores \(A=3a_x+4a_y+a_z\) e \(B=a_x+2a_y-5a_z\) encontre o ângulo \(\theta \) entre \(A\) e \(B\).
<br><br>
Para calcular o ângulo vamos usar a forma trigonométrica do Produto Escalar: \(A\cdot B =|A||B|cos(\theta)\). Logo vamos precisar dos módulos dos vetores e do Produto Escalar entre eles. Sendo assim: 
$$A\cdot B = (3,4,1)\cdot(1,2,-5) = (3)(1)+(4)(2)+(1)(-5)=6$$
$$|A| = |(3,4,1)|=\sqrt{3^2+4^2+1^2}=5,1$$
$$|B| = |(1,2,-5)|=\sqrt{1^2+2^2+5^2}=5,48$$
Com o Produto Escalar e os módulos dos vetores podemos aplicar \(A\cdot B =|A||B|cos(\theta)\) logo
$$6 =(5,1)(5,48)cos(\theta) \therefore cos(\theta) = \frac{6}{27,95}=0,2147$$
$$\theta = arccos(0,2147)=77,6^0$$
</p>

<p> Generalizando o produto escalar entre dois vetores \(A\) e \(B\) com \(N\) dimensões teremos:
$$A\cdot B = \sum\limits_{i=1}\limits^{N} a_ib_i$$
</p>
<p>
O Produto Escalar é comutativo e distributivo. Sendo assim, teremos: 
$$A\cdot B = B\cdot A$$
$$A\cdot (B+C) = A\cdot B +A\cdot C$$
Além disso, teremos \(A\cdot A = |A|^2\) já que o ângulo entre um vetor e ele mesmo é zero e \(cos(0) = 1\). Por padrão, consideramos que \(A^2 = |A|^2\).
</p>
<p>Os vetores unitários são ortogonais entre si, em todos os sistemas de coordenadas que usaremos neste artigo. Esta ortogonalidade garante duas propriedades interessantes do Produto Escalar, a saber:
$$a_x\cdot a_y=a_x\cdot a_z=a_y\cdot a_z=0$$
$$a_x\cdot a_x=a_y\cdot a_y=a_z\cdot a_z=1$$
</p>
<p>
Estas propriedades podem ser expressas usando o <i>Delta de Kronecker</i> definido por <a href="https://en.wikipedia.org/wiki/Leopold_Kronecker" target="_blanck">Leopold Kronecker (1823–1891)</a>. O <i>Delta de Kronecker</i> é uma forma de representar por índices as dimensões do espaço dimensional escolhido. Sendo assim, teremos: 
$$
\begin{equation}
  \delta_{\mu \upsilon}=\begin{cases}
    1, se \space\space \mu = \upsilon .\\
    0, se \space\space \mu \neq \upsilon.
  \end{cases}
\end{equation}
$$ 
</p>
<p>
Usando o <i>Delta de Kronecker</i> podemos escrever as propriedades dos componentes ortogonais unitários em relação ao produto escalar como: 
$$a_\mu \cdot a_\upsilon = \delta_{\mu \upsilon}$$
Que será útil na representação computacional de vetores e no entendimento de transformações vetoriais em espaços com mais de \(3\) dimensões.
</p>

<h5>Produto Vetorial</h5>
<p>O Produto Vetorial entre dois vetores \(A\) e \(B\), representado por \(A\times B\) é um vetor cuja magnitude representa a área do paralelogramo formado por \(A\) e \(B\) e cuja direção é perpendicular ao plano deste paralelogramo, \(a_n\). Geometricamente o Produto Vetorial será dado por: 
$$A\times B = |A||B|sen\theta_{AB}\space a_n$$
A direção de \(a_n\) pode ser determinada pela regra da mão direita. Colocando seus dedos esticados na direção do vetor \(A\) ao fechar estes dedos na direção do vetor \(B\), seu polegar estará apontando na direção do sentido positivo de \(a_n\).   
</p>
<p>Algebricamente o Produto Vetorial pode ser encontrado usando uma matriz. Considerando os vetores \(A=A_x a_x+A_y a_y+A_z a_z\) e \(B=B_x a_x+B_y a_y+B_z a_z\) o Produto Vetorial \(A\times B\) será encontrado resolvendo a matriz:
$$A\times B=\begin{vmatrix}
a_x & a_y & a_z\\
A_x & A_y & A_z\\
B_x & B_y & B_z
\end{vmatrix}$$
Logo:
$$A\times B=\begin{vmatrix}
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
\end{vmatrix}a_z$$
$$A\times B=(A_y B_z- A_z B_y)a_x-(A_x B_z-A_z B_x)a_y+(A_x B_y-A_y B_x)a_z$$
</p>
<p>O Produto Vetorial não é comutativo. Logo \(A\times B \neq B\times A). Contudo é anticomutativo. Isto quer dizer que  \(A\times B=-B\times A). O Produto Vetorial também não é distributivo. Ou seja: 
$$A\times (B\times C) \neq (A\times B)\times C$$
O Produto Vetorial é distributivo. Logo: 
$$A\times (B+C) = A\times B + A\times C$$
</p>
<p>Como o Produto Vetorial \(A\times B\) produz um vetor ortogonal ao plano formado por \(A\) e \(B\) a aplicação deste conceito a dois dos vetores unitários de um sistema de coordenadas irá produzir o terceiro vetor deste sistema. Sendo assim, teremos: 
$$a_x\times a_y = a_z$$
$$a_x\times a_z = a_y$$
$$a_y\times a_z = a_x$$
</p>
<h5>Produto Escalar Triplo</h5>
<p>Considerando os vetores \(A\), \(B\) e \(B\) definiremos o produto escalar triplo como:
$$A\cdot (B\times C)= B\cdot (C\times A) = C\times (A\times B)$$
Que pode ser calculado usando uma matrix por: 
$$A\cdot (B\times C) =\begin{vmatrix}
A_x & A_y & A_z\\
B_x & B_y & B_z\\
C_x & C_y & C_z
\end{vmatrix}$$
Na matrix que usamos para calcular o produto escalar triplo não estão colocados os vetores unitários do sistema de coordenadas o resultado desta operação será um escalar. 
</p>
<p>O resultado do Produto Escalar Triplo é o volume de um paralelepípedo formado pelos três vetores envolvidos na operação. Observe que este volume pode ser negativo, ou não, dependendo apenas da ordem dos vetores neste produto. Assim, podemos  dizer que o resultado do Produto Escalar Triplo é um <i>pseudoescalar</i>. Em álgebra linear, um pseudoescalar é um valor que pode mudar o sinal devido a alguma mudança de sinal de uma das suas coordenadas espaciais. No nosso caso, o volume pode mudar de sinal de acordo com a ordem dos vetores na operação de Produto Escalar Triplo. Ainda assim, esta mudança de sinal não inclui qualquer informação de direção ou sentido para o resultado do Produto Escalar Triplo.</p>

<h5>Produto Vetorial Triplo</h5>
<p>Considerando os vetores \(A\), \(B\) e \(B\) definiremos o produto vetorial triplo como:
$$A\times (B\times C) = B(A\cdot C)- C$$
</p>
<p class="exp">
Considere os vetores \(A = ( 1, 2, 3 )\), \(B = ( 2, 3, 1 )\) e \(C = ( 3, 1, 2 )\). Vamos calcular o produto vetorial triplo \( A \times (B \times C) \).

Calculando o produto vetorial de \(B\) e \(C\)

$$B \times C = \begin{vmatrix} a_x & a_y & a_z \\ 2 & 3 & 1 \\ 3 & 1 & 2 \end{vmatrix} = (3a_x - 7a_y + 7a_z)$$

Multiplicando o vetor \(A\) pelo resultado do passo 1

$$A \times (B \times C) = \begin{vmatrix} a_x & a_y & a_z \\ 1 & 2 & 3 \\ 3a_x & -7a_y & 7a_z \end{vmatrix} = a_x\begin{vmatrix} 2 & 3 \\ -7 & 7 \end{vmatrix} - a_y\begin{vmatrix} 1 & 3 \\ -7 & 7 \end{vmatrix} + a_z\begin{vmatrix} 1 & 2 \\ -7 & -7 \end{vmatrix}$$

Resolvendo as determinantes

$$\begin{aligned} \begin{vmatrix} 2 & 3 \\ -7 & 7 \end{vmatrix} &= (2 \times 7) - (3 \times (-7)) = 35 + 21 = 56 \\ \begin{vmatrix} 1 & 3 \\ -7 & 7 \end{vmatrix} &= (1 \times 7) - (3 \times (-7)) = 7 + 21 = 28 \\ \begin{vmatrix} 1 & 2 \\ -7 & -7 \end{vmatrix} &= (1 \times (-7)) - (2 \times (-7)) = -7 + 14 = 7 \end{aligned}$$

Simplificando

$$A \times (B \times C) = a_x(56) - a_y(28) + a_z(7) = 56a_x - 28a_y + 7a_z$$

Portanto, o produto vetorial triplo \(A \times (B \times C)\) é igual a \(56a_x - 28a_y + 7a_z\).

</p>

<h4>Campos Vetoriais</h4>
<p>Grandezas escalares formam Campos Escalares no espaço que podem ser definidos apenas como uma função \(f(x,y,z)\) onde \(x,y,z \in \Bbb{R}^3\). Grandezas vetoriais formam Campos Vetoriais que serão definidos por funções vetoriais \(V(x,y,z)\). Ou, em outras palavras representamos Campos Vetoriais no espaço como uma função de três variáveis. 
</p>
<p>
Campos vetoriais são uma abstração matemática e visual de funções vetoriais que descrevem como vetores variam no espaço. Este espaço pode ser definido em qualquer número de dimensões maior que \(2\). Em outras palavras, um Campo Vetorial é uma função que associa um vetor a cada ponto de um espaço especifico, representando uma grandeza com direção e magnitude. 
Campos vetoriais são comumente usados para representar fenômenos físicos, como forças em mecânica, campos elétricos e magnéticos em eletromagnetismo, e fluxos de fluidos em dinâmica dos fluidos. Uma função vetorial será representada como:

$$
\mathbf{F}(x, y, z) = (P(x, y, z), Q(x, y, z), R(x, y, z))
$$

onde \(\mathbf{F}\) é o Campo Vetorial, \((x, y, z)\) são as coordenadas de um ponto no espaço e \(P\), \(Q\) e \(R\) são funções escalares que dependem das coordenadas \((x, y, z)\). O vetor resultante em cada ponto terá componentes nas direções \(x\), \(y\) e \(z\) representadas pelos vetores unitários \((a_x, a_y, a_z)\).

Em um espaço bidimensional, a função vetorial terá apenas componentes \(x\) e \(y\):

$$
\mathbf{F}(x, y) = (P(x, y), Q(x, y))
$$

Um Campo Vetorial pode ser visualizado através de gráficos de vetores, onde cada vetor é representado por uma reta orientada, geralmente com uma seta, com direção e magnitude. Estes gráficos são complexos e, geralmente, demandam um tempo grande para a interpretação do Campo Vetorial. Uma forma mais simples é definir um ponto de testes no espaço e traçar poucas linhas entre a origem do Campo Vetorial e este ponto.

</p>
