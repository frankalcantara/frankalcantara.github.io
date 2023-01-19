---
layout: post
title: "A Fórmula da Atração: a matemática por trás do Eletromagnetismo"
author: Frank
categories: [opinião]
tags: [opinião, covid]
image: assets/images/eletro1.jpg
description: "Entenda como a matemática fundamenta o eletromagnetismo e suas aplicações práticas em um artigo acadêmico destinado a estudantes de ciência e engenharia."
---

<p> Tudo que nos cerca é devido a existência do Eletromagnetismo. Tudo! Desde sua existência até o dispositivo que você está usando está intimamente relacionado aos efeitos do campo elétrico e do campo magnético. No final do Século XIX, James Clerk Maxwell sintetizou a relação entre estas forças da natureza em um conjunto de equações matemáticas. O trabalho de Maxwell extraordinário de qualquer ponto de vista, ainda hoje, surpreende a todos pela precisão matemática. Neste artigo, vou passar alguns dos conceitos matemáticos que você precisa entender antes de se aprofundar no eletromagnetismo. Tenha coragem e calma. Tudo que posso dizer é que vai doer!</p>

<blockquote>"Nesta longa vida eu aprendi que toda a nossa ciência se comparada com a realidade é primitiva e infantil. Ainda assim, é a coisa mais preciosa que temos." Albert Einstein</blockquote>

<p>Este artigo explora como a matemática suporta o eletromagnetismo, um campo da física fundamental para a compreensão de fenômenos como a eletricidade e o magnetismo. Aqui você vai encontrar uma visão geral dos princípios matemáticos fundamentais que são usados para descrever os fenômenos eletromagnéticos. Destina-se a estudantes de ciência e engenharia, mas pode ser interessante para qualquer pessoa interessada em aprender sobre como a matemática está presente no mundo à nossa volta.</p>

<h4>Vetores, tudo depende dos vetores</h4>

<p>Vetores são abstrações matemáticas que permitem o entendimento de grandezas que, por sua vez, precisam das informações de grandeza, direção e sentido para seu próprio entendimento. Estas grandezas são as grandezas vetoriais. O eletromagnetismo só revela seus segredos na forma de vetores e campos vetoriais. Geometricamente, um vetor será representado por uma seta, com origem, destino e comprimento. Vamos ignorar a geometria sempre que possível e nos concentrar na álgebra.</p>

<h5>Vetor Unitário</h5>
<p>Um vetor \(V\) qualquer tem magnitude, direção e sentido. A magnitude, também chamada de intensidade, ou módulo será representada por \(|V|\). Sendo assim, definiremos um vetor unitário \(a_V\) como:
$$a_V=\frac{V}{|V|}$$
Um vetor unitário \(a_V\) é um vetor que tem a mesma direção e sentido de \(V\) com magnitude \(1\) logo \(|V|=1\).
</p>
<p>Grandezas escalares formam Campos Escalares no espaço que podem ser definidos apenas como uma função \(f(x,y,z)\) onde \(x,y,z \in \Bbb{R}^3\). Grandezas vetoriais formam Campos Vetoriais que serão definidos por funções vetoriais \(V(x,y,z)\). Ou na forma da soma dos componentes unitários relativos ao sistema de coordenadas escolhido. Usando o Sistema de Coordenadas Cartesianas a representação de um vetor \(V\) segundo seus componentes unitários e ortogonais será dada por:
$$V=V_xa_x+V_ya_y+V_za_z$$
Onde, \(V_x,V_y,V_z\) representam os fatores que devem multiplicar os vetores unitários \(a_x, a_y, a_z\) de forma que a soma destes vetores represente o vetor \(V\) no espaço \(\Bbb{R}^3\). Ao longo deste artigo vamos chamar \(V_x,V_y,V_z\) de componentes vetoriais nas direções \(x,y,z\). As direções \(x,y,z\) foram escolhidas porque estamos usando o Sistema de Coordenadas Cartesianas e indicam que cada vetor unitário corresponderá a uma das direções do Sistema de Coordenadas Cartesianas. Se utilizarmos sistemas de coordenadas diferentes os componentes vetoriais deverão ser adequados as estes sistemas. Assim teremos: 
 $$V=V_xa_x+V_ya_y+V_za_z$$
 $$V=V_ra_r+V_\phi a_\phi+V_za_z$$
 $$V=V_ra_r+V_\phi a_\phi+V_\theta a_\theta$$
 Respectivamente para os sistemas de coordenadas cartesianas, cilíndricas e esféricas. Quando representamos um vetor por seus componentes ortogonais, podemos calcular sua magnitude utilizando os fatores multiplicadores de cada componente assim, dado o vetor \(V=V_xa_x+V_ya_y+V_za_z\) sua magnitude será dada por \(|V|=\sqrt(V_x^2+V_y^2+V_z^2)\). Desta forma poderemos encontrar o vetor unitário de \(V\) por: 
 $$a_V=\frac{V_xa_x+V_ya_y+V_za_z}{\sqrt(V_x^2+V_y^2+V_z^2)}$$
</p>
<h5>Multiplicação por Escalar</h5>
<p>Um escalar é uma grandeza que não precisa de direção e sentido. Logo um escalar é um número real. Sendo assim, teremos:
$$B=3.3 \times V=3.3 \times V_xa_x+3.3 \times V_ya_y+3.3 \times V_za_z$$
Ou ainda: 
$$C=\frac{V}{2}=\frac{1}{2}\times V_xa_x+\frac{1}{2}\times V_ya_y+\frac{1}{2}\times V_za_z$$
A Multiplicação por Escalar é Comutativa e Associativa. Assim se tivermos os escalares \(m\) e \(n\) e o vetor \(V\) teremos:
$$mV=Vm$$
$$m(nV) = (mn)V$$</p>
<p>Chamamos de Vetor Oposto a \(V\) ao vetor que tem a mesma intensidade, a mesma direção e sentido oposto ao sentido de \(V\). Um Vetor Oposto é o resultado da multiplicação de um vetor pelo escalar \(-1\). Logo:
</p>
<h5>Soma e Subtração de Vetores</h5>
<p>Vetores podem ser somados, ou subtraídos, geometricamente por meio da regra do paralelogramo. Algebricamente a soma, ou subtração de vetores é feita componente a componente. Assim, se considerarmos os vetores \(A\) e \(B\) poderemos encontrar um vetor \(C\) que seja a soma \(A+B\) por: 
$$C=A+B=(A_x a_x+A_y a_y+A_z a_z)+(B_x a_x+B_y a_y+B_z a_z)=(A_x+B_x)a_x+(A_y+B_y)a_y+(A_y+B_y)a_z$$
Por exemplo, se : \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) então:
$$C=A+B=(5a_x-3a_y+a_z)+(1a_x+4a_y-7a_z)=(5+1)a_x+(-3+4)a_y+(1-7)a_z \therefore C= 6a_x+a_y-6a_z$$
</p>
<p>A Subtração será uma soma em que o segundo operando será o vetor oposto do operando original. Assim:
$$C=A-B=A+(-B)=A+(-1\times B)$$
 se : \(A=5a_x-3a_y+a_z\) e \(B=1a_x+4a_y-7a_z\) então:
$$C=A-B=(5a_x-3a_y+a_z)+(-1\times (1a_x+4a_y-7a_z))$$
$$C=A-B=(5a_x-3a_y+a_z)+(-1a_x-4a_y+7a_z)=4a_x-7a_y+8a_z$$
<\p>


<span>Foto de: <a href="https://unsplash.com/pt-br/fotografias/_kdTyfnUFAc">Alessandro Bianchi</a> on <a href="https://unsplash.com/s/photos/covid?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
