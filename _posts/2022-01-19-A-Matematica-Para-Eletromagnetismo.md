---
layout: post
title: "A Fórmula da Atração: a matemática por trás do Eletromagnetismo"
author: Frank
categories: [Matemática, Eletromagnetismo]
tags: [Matemática, Física, Eletromagnetismo]
image: assets/images/eletromag1.webp
description: "Entenda como a matemática fundamenta o eletromagnetismo e suas aplicações práticas em um artigo acadêmico destinado a estudantes de ciência e engenharia."
---

<blockquote>"Este é um trabalho em andamento. Escrevo quando posso e posto em seguida. Falta Muito!</blockquote>

<p>
Tudo que nos cerca é devido ao Eletromagnetismo. Tudo! Trata-se do estudo das forças fundamentais que definem nosso Universo. Desde a sua própria existência até o dispositivo que você está usando, tudo está intimamente relacionado aos efeitos do campo elétrico, do campo magnético e das interações entre eles.
</p>
<p>
No final do Século XIX, James Clerk Maxwell sintetizou a relação entre estas forças em um conjunto de equações matemáticas que explicam matematicamente como o eletromagnetismo afeta o espaço e a matéria. O trabalho de Maxwell, extraordinário de qualquer ponto de vista, surpreende pela simplicidade e beleza. Entretanto, requer um profundo conhecimento tando de física quanto de matemática para seu entendimento. Neste artigo, exploraremos a base matemática necessária ao conhecimento do eletromagnetismo, sem nos preocuparmos com a física onde esta matemática será aplicada.
</p>
<p>Considere este texto como o primeiro passo da sua jornada. Tenha coragem,calma e perseverança. Tudo que posso dizer é que vai doer e não será rápido.</p>

<blockquote>"Nesta longa vida eu aprendi que toda a nossa ciência se comparada com a realidade é primitiva e infantil. Ainda assim, <b>é a coisa mais preciosa que temos</b>." Albert Einstein</blockquote>

Imagem de <a href="https://pixabay.com/users/asimina-1229333/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2773167">Asimina Nteliou</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2773167">Pixabay</a>

<p>
Este artigo destina-se a estudantes de ciência e engenharia, mas pode ser relevante para qualquer pessoa interessada em aprender sobre a matemática como ferramenta para entender o Universo.
</p>

<h4>Álgebra Linear - Vetores, tudo depende dos vetores</h4>

<p>
Vetores são abstrações matemáticas usadas para o entendimento de grandezas que precisam das informações de magnitude, direção e sentido para sua definição e uso. Estas grandezas, as grandezas vetoriais, são a base do estudo do eletromagnetismo. Geometricamente, um vetor será representado por uma seta, com origem, destino e comprimento. A álgebra linear e a trigonometria são os campos da matemática mais adequados ao estudo de vetores. Aqui Vamos ignorar a trigonometria sempre que possível, focando preferencialmente na álgebra linear. Começando por definir vetor unitário.
</p>
<h5>Vetor Unitário</h5>
<p>Um vetor \(V\) qualquer tem magnitude, direção e sentido. A magnitude, também chamada de intensidade, ou módulo, será representada por \(|V|\). Definiremos um vetor unitário \(a_V\) como:
$$a_V=\frac{V}{|V|}$$
Um vetor unitário \(a_V\) é um vetor que tem a mesma direção e sentido de \(V\) com magnitude \(1\) logo \(|V|=1\).
</p>
<p>
 Podemos expressar qualquer vetor na forma da soma dos seus componentes unitários ortogonais. Estes components são relativos ao sistema de coordenadas que escolhemos para resolver um problema específico. Usando o Sistema de Coordenadas Cartesianas a representação de um vetor \(V\) segundo seus componentes unitários e ortogonais será dada por:
$$V=V_xa_x+V_ya_y+V_za_z$$
Onde, \(V_x,V_y,V_z\) representam os fatores que devem multiplicar os vetores unitários \(a_x, a_y, a_z\) de forma que a soma destes vetores represente o vetor \(V\) no espaço \(\Bbb{R}^3\). Ao longo deste artigo vamos chamar \(V_x,V_y,V_z\) de componentes vetoriais nas direções \(x,y,z\). As direções \(x,y,z\) foram escolhidas quando usamos o Sistema de Coordenadas Cartesianas. Estas direções, ou eixos, Sistema de Coordenadas Cartesianas implicam na existência dos vetores unitários \(a_x, a_y, a_z\). Se utilizarmos sistemas de coordenadas diferentes os vetores unitários deverão ser adequados aos eixos deste sistema. Assim teremos: 
 $$V=V_xa_x+V_ya_y+V_za_z$$
 $$V=V_ra_r+V_\phi a_\phi+V_za_z$$
 $$V=V_ra_r+V_\phi a_\phi+V_\theta a_\theta$$
 Respectivamente para os sistemas de coordenadas cartesianas, cilíndricas e esféricas. Não é raro que, uma vez que o sistema de coordenadas tenha sido definido os vetores sejam representados apenas por seus componentes. Desta forma, \(V=3a_x+a_y-a_z\) pode ser representado apenas por \(V=(3,1,-1)\).
 </p>
 <p>
 Quando representamos um vetor por seus componentes ortogonais, podemos calcular sua magnitude utilizando os fatores multiplicadores de cada componente. Assim, dado o vetor \(V=V_xa_x+V_ya_y+V_za_z\) sua magnitude será dada por \(|V|=\sqrt(V_x^2+V_y^2+V_z^2)\). Desta forma poderemos encontrar o vetor unitário de \(V\) por: 
 $$a_V=\frac{V_xa_x+V_ya_y+V_za_z}{\sqrt(V_x^2+V_y^2+V_z^2)}$$
</p>
<p class="exp">
Exemplo: calcule o vetor unitário \(a_A\) do vetor \(A=a_x-3a_y+2a_z\).
$$a_A=\frac{A_xa_x+A_ya_y+A_za_z}{\sqrt(A_x^2+A_y^2+A_z^2)}$$
$$a_A=\frac{a_x-3a_y+2a_z}{\sqrt(1^2+(-3)^2+2^2)}=\frac{a_x-3a_y+2a_z}{3,7416}$$
$$a_A=0,2672a_x-0,8018a_y+0,5345a_z$$
</p>
<h5>Multiplicação por Escalar</h5>
<p>
Um escalar é uma grandeza que não precisa de direção e sentido. A multiplicação de um vetor por um escalar implica na multiplicação de cada um dos componentes desse vetor por este escalar. 
</p>
<p class="exp">
Exemplo: considere o vetor \(V=2a_x+4a_y-a_z\) e calcule \(3,3V\) e \(V/2\)
$$3.3V=(3,3)(2)a_x+(3,3)(4)a_y+(3,3)(-1)a_z=6,6a_x+13,2a_y-3,3a_z$$
$$\frac{V}{2}=(\frac{1}{2})(2)a_x+(\frac{1}{2})(4)a_y+(\frac{1}{2})(-1)a_z = a_x+2a_y-\frac{1}{2}a_z$$
</p>
A Multiplicação por Escalar é Comutativa, Associativa e Distributiva. Assim se tivermos os escalares \(m\) e \(n\) e o vetor \(V\) teremos:
$$mV=Vm$$
$$m(nV) = (mn)V$$
</p>
<p>Chamamos de Vetor Oposto a \(V\) ao vetor que tem a mesma intensidade, a mesma direção e sentido oposto ao sentido de \(V\). Um Vetor Oposto é o resultado da multiplicação de um vetor pelo escalar \(-1\). Logo:
</p>
<h5>Adição e Subtração de Vetores</h5>
<p>Vetores podem ser somados, ou subtraídos, geometricamente por meio da regra do paralelogramo. Algebricamente a soma, ou subtração de vetores é feita componente a componente. Assim, se considerarmos os vetores \(A\) e \(B\) poderemos encontrar um vetor \(C\) que seja a soma \(A+B\) por: 
$$C=A+B=(A_x a_x+A_y a_y+A_z a_z)+(B_x a_x+B_y a_y+B_z a_z)$$
$$C=A+B=(A_x+B_x)a_x+(A_y+B_y)a_y+(A_y+B_y)a_z$$
</p>
<p class="exp">
Exemplo: se \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) calcule \(C=A+B\).
$$C=A+B=(5a_x-3a_y+a_z)+(1a_x+4a_y-7a_z)=(5+1)a_x+(-3+4)a_y+(1-7)a_z $$
$$C= 6a_x+a_y-6a_z$$
</p>
<p>A Subtração será uma soma em que o segundo operando será o vetor oposto do operando original. Assim:
$$C=A-B=A+(-B)=A+(-1\times B)$$
</p>
<p class="exp">
Exemplo: considere \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) e calcule \(C=A-B\).
$$C=A-B=(5a_x-3a_y+a_z)+(-1\times (1a_x+4a_y-7a_z))$$
$$C=A-B=(5a_x-3a_y+a_z)+(-1a_x-4a_y+7a_z)=4a_x-7a_y+8a_z$$
</p>
<p>Tanto a adição quanto a subtração de vetores são Comutativas, Associativas e Distributivas. Assim, considerando os vetores \(A\), \(B\) e \(C\) e o escalar \(m\), teremos:
$$A+B=B+A$$
$$A+(B+C)=(A+B)+C$$
$$m(A+B)=mA+mC$$
</p>
<h5>Vetores Distância e Ponto</h5>
<p>Em qualquer sistema de coordenadas vamos chamar de vetor ponto, ou vetor posição, \(r\). Ao vetor que liga a origem do sistema de coordenadas a um ponto \(P\) qualquer no espaço. De tal forma que \(r_p\) será dado por:
$$r_P = (P_x-0)a_x+(P_y-0)a_y+(P_z-0)a_z = P_x a_x+P_y a_y+P_z a_z$$
Ou seja, o vetor ponto é um vetor que indica, no espaço, onde um determinado ponto está e, até o momento é o vetor que vimos neste artigo. 
</p>
<p>Ainda, em qualquer sistema de coordenadas, o vetor distância, \(d\), é o vetor que mede o deslocamento entre dois pontos no espaço. Então, considerando que \(P\) esteja no ponto \((3,2,-1)\) e \(Q\) esteja no ponto \((1,-2,3)\), o vetor distância \(d_{PQ}\) será dado por:
$$d_{PQ} = r_P - r_Q$$
Logo: 
$$d_{PQ} = (P_x-Q_x)a_x+(P_y-Q_y)a_y+(P_z-Q_z)a_z $$
$$d_{PQ} = (3-1)a_x+(3-(-2))a_y+((-1)-3)a_z$$
$$d_{PQ} = 2a_x+5a_y-4a_z$$
</p>
<h5>Produto Escalar</h5>
<p>O Produto Escalar é uma operação entre dois vetores que resulta em um valor escalar. Dados os vetores \(A\) e \(B\), o produto escalar entre eles será representado por \(A\cdot B\). Neste ponto precisamos recorrer a geometria. Geometricamente o produto escalar entre \(A\) e \(B\) será dado por: 
$$A\cdot B = |A||B|cos\theta_{AB}$$
O produto escalar entre dois vetores \(A\) e \(B\) é o produto das magnitudes destes vetores e o cosseno do menor ângulo entre eles. Algebricamente, se \(A=A_x a_x+A_y a_y+A_z a_z\) e \(B=B_x a_x+B_y a_y+B_z a_z\) então teremos:
$$A\cdot B = A_x B_x+A_y B_y+A_z B_z$$
Por exemplo: se \(A=3a_x-4a_y+a_z\) e \(B=3a_x+2a_y-3a_z\) então:
$$A\cdot B = (3)(3)+(-4)(2)+(1)(-3)= 9-8-3=-2$$
O Produto Escalar é comutativo e distributivo. Sendo assim, teremos: 
$$A\cdot B = B\cdot A$$
$$A\cdot (B+C) = A\cdot B +A\cdot C$$
Além disso, teremos \(A\cdot A = |A|^2\) já que o ângulo entre um vetor e ele mesmo é zero e \(cos0^0 = 1\). Por padrão, consideramos que \(A^2 = |A|^2).
</p>
<p>Os vetores unitários são ortogonais entre si, em todos os sistemas de coordenadas que usaremos neste artigo. Esta ortogonalidade garante duas propriedades interessantes do Produto Escalar, a saber:
$$a_x\cdot a_y=a_x\cdot a_z=a_y\cdot a_z=0$$
$$a_x\cdot a_x=a_y\cdot a_y=a_z\cdot a_z=1$$
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
<p>O resultado do Produto Escalar Triplo é o volume de um paralelepípedo formado pelos três vetores envolvidos na operação. Observe que este volume pode ser negativo, ou não, dependendo apenas da ordem dos vetores neste produto. Assim, seria mais correto dizer que o resultado do Produto Escalar Triplo é um <i>pseudoescalar</i>. Em álgebra linear, um pseudoescalar é um valor que pode mudar o sinal devido a alguma mudança de sinal de uma das coordenadas espaciais. No nosso caso, o volume pode mudar de sinal de acordo com a ordem dos vetores na operação de Produto Escalar Triplo. Ainda assim, esta mudança de sinal não inclui qualquer informação de direção ou sentido para o resultado do Produto Escalar Triplo.</p>

<h5>Produto Vetorial Triplo</h5>
<p>Considerando os vetores \(A\), \(B\) e \(B\) definiremos o produto vetorial triplo como:
$$A\times (B\times C) = B(A\cdot C)- C$$

</p>
<p>Grandezas escalares formam Campos Escalares no espaço que podem ser definidos apenas como uma função \(f(x,y,z)\) onde \(x,y,z \in \Bbb{R}^3\). Grandezas vetoriais formam Campos Vetoriais que serão definidos por funções vetoriais \(V(x,y,z)\).
</p>
