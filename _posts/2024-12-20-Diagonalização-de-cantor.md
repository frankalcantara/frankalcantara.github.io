---
layout: post
title: Em Busca da Diagonalização de Cantor
author: Frank
categories:
    - artigo
    - Matemática
    - disciplina
tags:
    - algoritmos
    - Matemática
    - lógica
image: assets/images/cantor.webp
featured: false
rating: 5
description: Tradução de dois artigos de Georg Cantor para o português como falado no Brasil.
date: 2024-12-19T20:42:55.811Z
preview: Para colocar ordem na casa, a tradução de dois artigos de Georg Cantor sobre a não numerabilidade dos números reais.
keywords: |-
    Georg Cantor
    Diagonalização
    Computabilidade
toc: false
published: true
beforetoc: Para colocar ordem na casa, a tradução de dois artigos de Georg Cantor sobre a não numerabilidade dos números reais.
lastmod: 2025-05-06T11:04:17.858Z
---

Triste é a sina de quem pretende entender.

Enquanto escrevia o livro sobre cálculo lambda, me deparei com alguns problemas surpreendentes. Não imaginava quanto seria difícil encontrar a sequência dos fatos que levaram a criação da ciência da computação.

Por exemplo, as informações sobre a origem da prova por contradição, usando a técnica de diagonalização. A maioria das fontes dá a entender que Cantor criou esta técnica para provar que os números reais não podem ser contados. Cresci ouvindo isso.

Pesquisando para o livro percebi que esta não é bem a verdade. Cantor já havia provado não ser possível contar os números reais quando publicou a ideia da diagonalização. E sequer foi o primeiro a fazer isso.

Então, resolvi traduzir alguns artigos seminais sobre o assunto. Como não coloquei as traduções no livro aqui estão.

___

## Prova de Cantor de 1874 sobre a Não-Denumerabilidade

Esta é uma tradução livre da prova de [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor), publicada em 1874, que demonstra a **não-denumerabilidade dos números reais**. O texto original, em alemão, pode ser encontrado online no artigo ["*Über eine Eigenschaft des Inbegriffes aller reellen algebraischen Zahlen*"](https://www.digizeitschriften.de/download/pdf/243919689_0077/log14.pdf)" e foi publicado em 1874 no periódico *Journal für die Reine und Angewandte Mathematik* (conhecido como *Crelle’s Journal*), Vol. 77, páginas 258–262.

Este texto, em português, como falado no Brasil, foi criado com ajuda de ferramentas de inteligência artificial para traduzir do alemão e corrigido usando traduções feitas para o inglês disponíveis na Internet. Esta é uma tradução livre, e comentada segundo as vozes da minha cabeça.

Antes da tradução: numerabilidade, em matemática, representa a qualidade de um conjunto ser numerável. Ou seja, que os elementos deste conjunto possam ser listados em uma sequência finita. A palavra denumerabilidade tem o mesmo sentido.

### Sobre uma Propriedade do Conjunto de Todos os Números Algébricos Reais - Por Georg Cantor, 1874

Por um número algébrico real entende-se um número real $\omega$ que satisfaz uma equação não-constante da forma:

$$\begin{equation}a_0 \omega^n + a_1 \omega^{n-1} + \dots + a_n = 0\end{equation}$$

na qual $n$, $a_0$, $a_1, \dots, a_n$ são inteiros. Aqui, os números $n$ e $a_0$ são positivos, os coeficientes $a_0, a_1, \dots, a_n$ não possuem fatores comuns, e a equação acima é irredutível. Com essas considerações, é verdade, de acordo com princípios conhecidos de aritmética e álgebra, que a equação (1) é completamente determinada pelos números algébricos reais que a satisfazem. Por outro lado, se $n$ é o grau da equação, então ela é satisfeita por no máximo $n$ números algébricos reais $\omega$.

Os números algébricos reais constituem, em sua totalidade, um conjunto de números denotado por $(\Omega)$. É evidente que $(\Omega)$ possui a propriedade de que, em qualquer intervalo ao redor de um número dado $\alpha$, existem infinitos números de $(\Omega)$ dentro desse intervalo. À primeira vista, pode parecer que seja possível correlacionar o conjunto $(\Omega)$, de forma bijetiva, ao conjunto $(\nu)$ de todos os inteiros positivos $\nu$. Ou seja, cada número algébrico real $\omega$ corresponderia a um inteiro positivo distinto, e, inversamente, cada inteiro positivo $\nu$ corresponderia a um número algébrico real único $\omega$. Em outras palavras, pode-se conceber $(\Omega)$ como uma sequência infinita ordenada:

$$\begin{equation}
\omega_1, \omega_2, \dots, \omega_\nu, \dots
\end{equation}
$$

na qual todos os números reais algébricos aparecem e cada número ocupa uma posição definida pela subscrição. Dada uma definição que estabeleça tal correspondência, ela pode ser modificada livremente. Na Seção 1, descreverei uma forma que considero a menos complicada. Para ilustrar uma aplicação dessa propriedade do conjunto de todos os números algébricos reais, mostrarei, na Seção 2, que, para qualquer sequência arbitrária de números reais da forma (2), é possível encontrar números $\eta$ que não pertencem à sequência, mesmo em um intervalo arbitrário $(\alpha, \beta)$. Combinando os resultados dessas duas seções, obtemos uma nova prova do teorema, primeiramente demonstrado por [Liouville](https://pt.wikipedia.org/wiki/Joseph_Liouville)[^1], de que em qualquer intervalo $(\alpha, \beta)$ existem infinitos números reais transcendentes (isto é, não algébricos).

Além disso, o teorema da Seção 2 explica porque conjuntos de números reais que formam um chamado **contínuo** (como todos os números reais entre $0$ e $1$) não podem ser correlacionados bijetivamente ao conjunto $(\nu)$. Desta forma, é possível identificar a diferença essencial entre um chamado contínuo e um conjunto como o dos números algébricos reais.

#### Seção 1

Retornemos à equação (1), que é satisfeita por um número algébrico $\omega$ e que, sob as condições estipuladas, é completamente determinada. A soma dos valores absolutos de seus coeficientes, mais o número $n - 1$ (onde $n$ é o grau de $\omega$), será chamada de **altura do número $\omega$**. Denotemos esta altura por $N$. Assim, na notação usual, temos:

$$\begin{equation}
N = n - 1 + |a_0| + |a_1| + \dots + |a_n|
\end{equation}$$

A altura $N$ é, portanto, um inteiro positivo bem definido para cada número algébrico real $\omega$. Inversamente, para qualquer valor inteiro positivo de $N$, existem apenas um número finito de números algébricos reais com altura $N$. Se chamarmos esse número de $\Phi(N)$, então, por exemplo, temos:

$$\begin{equation}
\Phi(1) = 1, \quad \Phi(2) = 2, \quad \Phi(3) = 4, \text{ e assim por diante.}
\end{equation}
$$

Os números do conjunto $(\Omega)$, isto é, o conjunto de todos os números algébricos reais, podem então ser organizados da seguinte forma: tomamos como primeiro número $\omega_1$ o único número com altura $N = 1$; para $N = 2$, existem 2 números algébricos reais, que são ordenados de acordo com o tamanho, sendo designados por $\omega_2, \omega_3$; para $N = 3$, há 4 números, novamente ordenados por tamanho. De forma geral, após todos os números em $(\Omega)$ até uma altura $N = N_k$ terem sido enumerados e atribuídos a posições definidas, os números algébricos reais com altura $N = N_k + 1$ seguem a ordem de tamanho. Assim, obtemos o conjunto $(\Omega)$ de todos os números algébricos reais na forma:

$$
\begin{equation}
\omega_1, \omega_2, \omega_3, \dots
\end{equation}
$$

Com respeito a essa ordenação, podemos nos referir ao número algébrico real de índice $\nu$. Nenhum membro do conjunto foi omitido.

#### Seção 2

Dada qualquer definição de uma sequência infinita de números reais mutuamente distintos, da forma:

$$\begin{equation}
\omega_1, \omega_2, \omega_3, \dots
\end{equation}
$$

então, em qualquer intervalo $(\alpha, \beta)$ arbitrariamente escolhido, existe um número $\eta$ (e, consequentemente, infinitos números desse tipo) que pode ser mostrado que **não ocorre** na sequência acima. Vamos agora provar este fato.

Consideremos o intervalo $(\alpha, \beta)$, onde $\alpha < \beta$, arbitrariamente escolhido. Os dois primeiros números de nossa sequência que estão no interior deste intervalo (com exceção das extremidades) podem ser designados por $\alpha'$ e $\beta'$, com $\alpha' < \beta'$. Da mesma forma, designamos os dois primeiros números de nossa sequência que estão no interior de $(\alpha', \beta')$ por $\alpha''$ e $\beta''$, onde $\alpha'' < \beta''$. Continuamos assim para determinar o próximo intervalo $(\alpha''', \beta''')$, e assim por diante.

Dessa forma, obtemos as sequências $\alpha', \alpha'', \dots$ e $\beta', \beta'', \dots$, cujos índices aumentam continuamente. Os números $\alpha', \alpha'', \dots$ estão sempre aumentando em valor, enquanto os números $\beta', \beta'', \dots$ estão sempre diminuindo. Além disso, os intervalos $(\alpha', \beta')$, $(\alpha'', \beta'')$, $(\alpha''', \beta''')$, etc., são aninhados, com cada um contendo todos os seguintes.

Existem agora dois casos possíveis:

##### Primeiro caso

O número de intervalos formados é finito. Neste caso, seja o último intervalo $(\alpha_k, \beta_k)$. Como, no interior deste intervalo, pode haver no máximo um número da sequência $(\omega_1, \omega_2, \omega_3, \dots)$, podemos escolher um número $\eta$ deste intervalo que não pertence à sequência. Assim, o teorema está provado para este caso.

##### Segundo caso

O número de intervalos formados é infinito. Neste caso, os números $\alpha, \alpha', \alpha'', \dots$, por estarem sempre aumentando de valor, mas não crescendo infinitamente, possuem um valor limite determinado $\alpha_\infty$. O mesmo vale para os números $\beta, \beta', \beta'', \dots$, que possuem um limite determinado $\beta_\infty$, por estarem sempre diminuindo de valor.

Além disso, ocorre que $\alpha_\infty = \beta_\infty$. Este é o caso típico quando lidamos com o conjunto $(\Omega)$ de todos os números algébricos reais. A partir da definição dos intervalos, podemos verificar que o número $\eta = \alpha_\infty = \beta_\infty$ não pode estar contido na sequência $(\omega_1, \omega_2, \omega_3, \dots)$[^2].

Se, no entanto, $\alpha_\infty < \beta_\infty$, então todo número $\eta$ no interior do intervalo $(\alpha_\infty, \beta_\infty)$, ou nas suas extremidades, satisfaz a condição de não pertencer à sequência $(\omega_1, \omega_2, \omega_3, \dots)$.

#### Conclusão

Os teoremas demonstrados neste artigo admitem extensões em diversas direções, das quais mencionaremos apenas uma:

> "Seja $\omega_1, \omega_2, \dots, \omega_n, \dots$ uma sequência finita ou infinita de números que são linearmente independentes entre si (isto é, de forma que nenhuma equação da forma $a_1 \omega_1 + a_2 \omega_2 + \dots + a_n \omega_n = 0$, com coeficientes inteiros não todos nulos, seja possível), e seja $(\Omega)$ o conjunto de todos os números $\Omega$ que podem ser representados como funções racionais com coeficientes inteiros dos números $\omega$. Então, em qualquer intervalo $(\alpha, \beta)$, existem infinitos números que não estão contidos em $(\Omega)$."

De fato, pode-se verificar, por um método de prova semelhante ao da Seção 1, que o conjunto $(\Omega)$ pode ser concebido na forma sequencial:

$$\begin{equation}
\omega_1, \omega_2, \omega_3, \dots
\end{equation}
$$

A partir disso, e considerando a Seção 2, a verdade do teorema segue diretamente.

#### Casos Especiais

Um caso especial notável do teorema acima (onde a sequência $\omega_1, \omega_2, \dots$ é finita e o grau das funções racionais que geram o conjunto $(\Omega)$ é pré-determinado) foi demonstrado com base em princípios de Galois pelo Sr. B. Minnigerode. (Veja *Math. Annalen*, Vol. 4, p. 497).

___

___ 

Esta é uma tradução livre do trabalho de [Georg Cantor](https://en.wikipedia.org/wiki/Georg_Cantor), publicada em 1890, que apresenta a diagonalização. O texto original, em alemão, foi publicado em 1890 no periódico *Jahresbericht der Deutschen Mathematiker-Vereinigung*, Vol. 1, páginas 72–78 e pode ser encontrado online no artigo ["*Ueber eine elementare Frage der Mannigfaltigkeitslehre*"](https://gdz.sub.uni-goettingen.de/id/PPN37721857X_0001?tify=%7B%22pages%22%3A%5B83%5D%2C%22pan%22%3A%7B%22x%22%3A0.497%2C%22y%22%3A0.762%7D%2C%22view%22%3A%22info%22%2C%22zoom%22%3A0.516%7D)". 

Novamente, a tradução do alemão foi realizada com apoio de ferramentas de inteligência artificial e corrigido com base em artigos diversos escritos em inglês, disponíveis na interneta.

___

## Sobre uma Questão Elementar da Teoria das Manifestações - Georg Cantor, 1891

### Seção 1

No ensaio intitulado **Sobre uma Propriedade da Coleção de Todos os Números Algébricos Reais** (Journ. Math., Vol. 77, p. 258), é provável que, pela primeira vez, tenha sido provado que existem **manifestações infinitas** que não podem ser correlacionadas de forma única com o conjunto de todos os inteiros finitos $1, 2, 3, \dots, \nu, \dots$, ou, como eu digo, que não possuem a mesma *potência* da série $1, 2, 3, \dots, \nu, \dots$.

Como mostrado na Seção 2 desse trabalho, segue-se diretamente que, por exemplo, o conjunto de todos os números reais de qualquer intervalo $(a, b)$ não pode ser imaginado na forma sequencial:

$$\omega_1, \omega_2, \dots, \omega_\nu, \dots \tag{1}$$

Contudo, é possível fornecer uma prova muito mais simples desse teorema, que é independente da consideração de números irracionais.

Se $m$ e $w$ forem dois caracteres mutuamente exclusivos, consideramos uma coleção $M$ de elementos $E = (x_1, x_2, \dots, x_\nu, \dots)$, que dependem de um número infinito de coordenadas $x_1, x_2, \dots, x_\nu, \dots$, onde cada uma dessas coordenadas é ou $m$ ou $w$[^3].

$M$ é a totalidade de todos os elementos $E$.

Os elementos de $M$ incluem, por exemplo, os seguintes três:

1. $E_I = (m, m, m, m, \dots)$  
2. $E_{II} = (w, w, w, w, \dots)$  
3. $E_{III} = (m, w, m, w, \dots)$  

Afirmo agora que tal manifestação $M$ não possui a mesma potência da série $1, 2, 3, \dots, \nu, \dots$.

Isso segue da seguinte proposição:

> Se $E_1, E_2, \dots, E_\nu, \dots$ forem quaisquer sequências infinitas de elementos da manifestação $M$, então sempre existe um elemento $E_0$ de $M$ que não coincide com nenhum $E_\nu$.

### Prova

Suponha que:

$$E_1 = (a_{1,1}, a_{1,2}, \dots, a_{1,\nu}, \dots),$$

$$E_2 = (a_{2,1}, a_{2,2}, \dots, a_{2,\nu}, \dots),$$

$$\dots$$

$$E_\mu = (a_{\mu,1}, a_{\mu,2}, \dots, a_{\mu,\nu}, \dots).$$

Aqui, os $a_{\mu,\nu}$ são ou $m$ ou $w$. Agora definimos uma sequência $b_1, b_2, \dots, b_\nu, \dots$ de forma que $b_\nu$ seja ou $m$ ou $w$ e **diferente de $a_{\nu,\nu}$**.

- Se $a_{\nu,\nu} = m$, então $b_\nu = w$.  
- Se $a_{\nu,\nu} = w$, então $b_\nu = m$.

Considere agora o elemento:

$$E_0 = (b_1, b_2, b_3, \dots)$$

de $M$. É fácil verificar que a equação:

$$E_0 = E_\mu$$

não pode ser satisfeita para nenhum valor inteiro positivo de $\mu$. Caso contrário, para o $\mu$ dado e para todos os valores inteiros de $\nu$, teríamos:

$$b_\nu = a_{\mu,\nu},$$

e, em particular,

$$b_\mu = a_{\mu,\mu},$$

o que é contraditório pela definição de $b_\nu$.

Portanto, segue-se imediatamente que a totalidade de todos os elementos de $M$ não pode ser colocada na forma sequencial:

$$E_1, E_2, \dots, E_\nu, \dots$$

Caso contrário, seríamos confrontados com a contradição de que um elemento $E_0$ é simultaneamente membro de $M$ e não é membro de $M$.

**Esta prova é notável não apenas por sua grande simplicidade, mas especialmente porque o princípio nela seguido pode ser estendido sem mais para a proposição geral de que as cardinalidades de conjuntos bem-definidos não têm um máximo ou, o que é o mesmo, que para qualquer conjunto dado $L$ pode ser colocado ao seu lado outro conjunto $M$ que tem cardinalidade maior que $L$.**

Seja, por exemplo, $L$ um *continuum linear*, digamos o conjunto de todas as grandezas numéricas reais $z$ que são $\geq 0$ e $\leq 1$.

Entenda-se por $M$ o conjunto de todas as funções unívocas $f(x)$ que assumem apenas os dois valores $0$ ou $1$, enquanto $x$ percorre todos os valores reais que são $\geq 0$ e $\leq 1$.

Que $M$ não tem cardinalidade menor que $L$, segue do fato de que se podem indicar subconjuntos de $M$ que têm a mesma cardinalidade que $L$, por exemplo, o subconjunto que consiste de todas as funções de $x$ que para um único valor $x_0$ de $x$ tem o valor $1$, para todos os outros valores de $x$ tem o valor $0$.

$M$ também não tem a mesma cardinalidade que $L$, pois caso contrário o conjunto $M$ poderia ser colocado em relação um-a-um com a variável $z$, e $M$ poderia ser pensado na forma de uma função unívoca das duas variáveis $x$ e $z$: $\phi(x,z)$ de modo que através de cada especialização de $z$ um elemento $f(x) = \phi(x,z)$ de $M$ é obtido e inversamente cada elemento $f(x)$ de $M$ surge de $\phi(x,z)$ através de uma única especialização determinada de $z$. Mas isto leva a uma contradição. Pois se entendermos por $g(x)$ aquela função unívoca de $x$ que assume apenas os valores 0 ou 1 e para cada valor de $x$ é diferente de $\phi(x,\bar{x})$, então por um lado $g(x)$ é um elemento de $M$, por outro lado $g(x)$ não pode surgir através de nenhuma especialização $z = z_0$ de $\phi(x,z)$, porque $\phi(z_0,z_0)$ é diferente de $g(z_0)$.

Sendo assim, a cardinalidade de $M$ nem menor nem igual àquela de $L$, segue-se que ela é maior que a cardinalidade de $L$. (Cf. Crelle's Journal, Bd. 84 S. 242.)

Já mostrei nos *Fundamentos de uma teoria geral dos conjuntos* (Leipzig 1883; Math. Annalen Bd. 21) por meios completamente diferentes que as cardinalidades não têm máximo; lá foi até mesmo provado que o conjunto de todas as cardinalidades, quando pensamos estas últimas ordenadas segundo sua grandeza, forma um *conjunto bem ordenado*, de modo que na natureza após cada cardinalidade segue uma próxima maior.

___

No último artigo Cantor apresenta uma demonstração mais simples do que hoje conhecemos como a técnica da diagonalização. O artigo estabelece que:

1. Existem conjuntos infinitos de diferentes tamanhos, cardinalidades;
2. Não existe um conjunto infinito de cardinalidade máxima;
3. O conjunto das cardinalidades é bem ordenado ;
4. Para cada cardinalidade, existe uma cardinalidade maior.

A prova apresentada por Cantor é notável por sua simplicidade e elegância, usando apenas conceitos básicos para demonstrar um resultado profundo sobre a natureza do infinito.

A amável leitora pode se sentir mais confortável usando a notação contemporânea:

Seja $M$ um conjunto cujos elementos são sequências infinitas de dois símbolos $m$ e $w$. Suponha que $M$ possa ser enumerado: 

$$E_1, E_2, ..., E_ν, ...$$ 

de tal forma que cada 

$$E_μ = (a_{μ,1}, a_{μ,2}, ..., a_{μ,ν}, ...)$$

Cantor constrói um elemento $E_0 = (b_1, b_2, b_3, ...)$ que difere de cada $E_μ$ em pelo menos uma posição:

$$
b_ν = \begin{cases}
w & \text{se } a_{ν,ν} = m \\
m & \text{se } a_{ν,ν} = w
\end{cases}
$$

Este $E_0$ está em $M$ mas não pode estar na enumeração. E temos a  contradição.

___

[^1]: NT: a vida inteira o sujeito escuta que Cantor provou que os números reais eram incontáveis. Ai o cara vai traduzir a prova e descobre que o próprio Cantor chama de "outra prova".

[^2]: Se o número $\eta$ estivesse contido em nossa sequência, teríamos $\eta = \omega_p$, onde $p$ é um índice específico. Isso, no entanto, é impossível, pois $\omega_p$ não está no interior do intervalo $(\alpha^{(p)}, \beta^{(p)})$, enquanto o número $\eta$ está no interior desse intervalo, de acordo com sua definição.

[^3]: NT: $m$ e $w$ são uma forma interessante de definir um conjunto binário de elementos.
