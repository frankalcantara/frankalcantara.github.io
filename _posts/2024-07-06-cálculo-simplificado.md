---
layout: post
title: Cálculo Simplificado
author: Frank
categories:
  - matemática
tags:
  - matemática
  - cálculo
  - álgebra
  - exercícios
  - problemas resolvidos
image: assets/images/prog_dynamic.jpeg
featured: false
rating: 5
description: Versão atualizada de Cálculo Made Easy com novos exercícios e exemplos. Aprenda cálculo de forma intuitiva desde os fundamentos até tópicos avançados.
date: 2024-07-06T19:31:18.251Z
preview: Cálculo Simplificado é a versão atualizada e ampliada do clássico Cálculo Made Easy. Este livro foi revisado e enriquecido com novos exercícios e exemplos práticos para facilitar ainda mais o entendimento dos conceitos de cálculo. Ideal para estudantes e profissionais que desejam dominar o cálculo de forma intuitiva e eficaz, este guia aborda desde os fundamentos até tópicos mais avançados, sempre com explicações claras e acessíveis. Descubra como o cálculo pode ser descomplicado e aplicado em diversas áreas do conhecimento.
keywords: Este livro é essencial para aqueles que buscam uma compreensão profunda e acessível do cálculo. A versão atualizada de Cálculo Simplificado não só mantém a fidelidade ao texto original de Cálculo Made Easy, mas também enriquece a experiência do leitor com explicações de detalhes pitorescos do início do século XX. Além disso, as unidades, teoremas e técnicas de resolução de problemas foram modernizados para refletir os avanços contemporâneos. Esta abordagem combinada proporciona uma visão completa e relevante, facilitando o aprendizado e a aplicação do cálculo nos dias atuais.
toc: true
published: false
slug: calculo-simplificado
draft: 2024-07-06T19:30:40.256Z
lastmod: 2024-09-28T14:02:20.355Z
---

O livro **Calculus Made Easy** de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson) foi colocado em domínio público, tanto no Brasil quanto nos EUA. Este é um dos melhores livros introdutórios de cálculo já escrito. Simples, direto e abrangente. Sem nenhuma modéstia, ou vergonha na cara, escolhi este livro para tradução, atualização e expansão. Na esperança de criar um material atualizado para o ensino e fomento do cálculo entre alguns estudantes de língua portuguesa. Vou traduzir, atualizar, comentar e expandir o conteúdo. Contudo, como o livro é ótimo, algumas coisas precisam ser mantidas o mais original possível por curiosidade e pelo valor cultural. Um bom exemplo é a brincadeira da capa:

> O QUE UM TOLO PODE FAZER, OUTRO TAMBÉM PODE.
> (Provérbio Simiano Antigo.)

Em 1910 quando a primeira edição de **Calculus Made Easy** foi publicada, ainda era aceitável e interessante, brincar e se divertir quando estudávamos matemática, ou ciência. Talvez seja por isso que este livro seja lembrado por tantos professores de língua inglesa, com carinho.

Vamos ao prólogo de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson), com o máximo de fidelidade que esse tolo consegue usar para traduzir e escrever.

> Considerando quantos tolos conseguem calcular, é surpreendente que se pense que é uma tarefa difícil, ou tediosa, para que qualquer outro tolo possa aprender a dominar os mesmos truques.
> Alguns truques de cálculo são inacreditavelmente fáceis. Por outro lado, alguns são enormemente difíceis. Os tolos que escrevem os livros de texto de matemática avançada, que são na maioria, tolos inteligentes, raramente se dão ao trabalho de mostrar como os cálculos fáceis podem ser fáceis. Pelo contrário, eles parecem desejar impressioná-lo com sua tremenda astúcia, abordando o cálculo da forma mais difícil.
> Sendo eu mesmo um sujeito notavelmente estúpido, tive que desaprender as dificuldades, e agora peço aos meus colegas tolos para ensinar as partes que não são difíceis. Domine o que veremos aqui e o resto seguirá. O que um tolo pode fazer, outro também pode.

Deste ponto em diante, sempre que eu apenas traduzir o livro original, a tradução estará destacada. Todo o resto do livro será versionado. Este tolo que escreve, muito mais tolo que o prof. Silvanus, também prefere o fácil, ainda que sofra atração pelo desafiador. Vou usar o texto do prof. Silvanus para privilegiar o fácil. Mesmo que consiga, cálculo em especial e matemática em geral, só se aprende com caderno, lápis e horas de cadeira resolvendo exercícios.

Se a interessada leitora chegou até aqui. Boa sorte e sucesso.

## I. PARA LIVRÁ-LO DOS MEDOS MITOLÓGICOS

O terror preliminar, que sufoca a maioria dos alunos iniciantes, impedindo-os de sequer tentar aprender cálculo, pode ser abolido de uma vez por todas, simplesmente declarando qual é o significado, em termos de bom senso, dos dois símbolos principais usados no cálculo. Esses símbolos terríveis são:

1. $d \space \space$ que significa meramente _um pequeno pedaço de_. Ou se preferir, com um pouco mais de formalidade _uma fração muito pequena de_. Assim, $dx$ significa um pedaço muito pequeno de $x$; ou $du$ significa um pedacinho muito pequeno de $u$. Matemáticos tradicionais acham mais educado dizer _um elemento de_ em vez de _um pedacinho de_, ou ainda uma _fração infinitesimal de_. Esta coisa de _infinitesimal_ quer dizer que é tão pequeno que quase se confunde com o zero. Você verá que esses pequenos pedaços podem ser considerados infinitamente pequenos.

2. $\int  \space \space$ que é apenas um **S** longo, e pode ser lido, se você preferir, como _a soma de_. Assim, $\int dx$ significa a soma de todos os pequenos pedaços de $x$; ou $\int dt$ significa a soma de todos os pequenos pedaços de $t$. Nossos amigos matemáticos chamam esse símbolo de _a integral de_. Qualquer tolo pode ver que se $x$ for considerado como composto por muitos pequenos pedaços, cada um dos quais é chamado de $dx$, se você somá-los, obterá, como resultado, $x$. A soma de todos os $dx$ é a mesma coisa que $x$. A palavra _integral_, que os matemáticos preferem, simplesmente significa _o todo_ e tem o mesmo sentido de _somar todas as pequenas partes que compõem o todo_.

Para criar uma imagem desta soma de pequenos pedaços, pense na duração de uma hora. Você sempre poderá pensar neste intervalo de tempo como sendo um intervalo dividido em $3600$ pequenos pedaços chamados de segundos. O todo, sinônimo de _a integral_ dos segundos é a hora e a hora é a soma dos seus $3600$ pequenos pedaços somados.

A partir de agora, quando você encontrar uma expressão que começa com esse símbolo aterrorizante, $\int $, você saberá que ele está lá apenas para lhe dar a instrução de que a partir deste momento, você deverá realizar uma operação de adição de todos os pequenos pedaços indicados pelo símbolo $\int $.

E cálculo é só isso!

## II. SOBRE DIFERENTES GRAUS DE PEQUENEZ

Pequenez, ou o grau de quanto as coisas são pequenas, é uma palavra rara no português coloquial do Brasil. Que muitas vezes é usada para indicar ausência de moral, mas que originalmente era a qualidade do pequeno. Indicando que um pequeno pode ser menor que outro. Pequenez será usada neste texto com seu sentido original e descobriremos que em nossos estudos de cálculo teremos que lidar com quantidades pequenas de vários graus de pequenez. A palavra pequenez nos permitirá entender que uma pequena fração, ou pequeno pedaço, pode ser menor que o outro.

É crucial compreendermos em quais situações poderemos considerar certas quantidades tão diminutas que estas se tornem negligenciáveis em nossas análises. Essas quantidades são tão pequenas que seu impacto será insignificante no contexto geral. No entanto, é importante notar que esta "insignificância", ou pequenez, é sempre relativa e depende da escala do problema em questão. O que é considerado desprezível em um contexto pode ser significativo em outro, portanto, a decisão de omitir ou não uma quantidade deve sempre levar em conta a magnitude relativa em relação aos outros elementos envolvidos no problema.

Antes de imergirmos no mundo das regras e formalidades da matemática, vamos considerar alguns casos familiares.

Existem $60$ minutos em uma hora, $24$ horas em um dia e $7$ dias em uma semana. Portanto, existem $1440$ minutos no dia e $10080$ minutos na semana. Obviamente, $1$ minuto é uma quantidade muito pequena de tempo comparada a uma semana.

Nossos antepassados desenvolveram um sistema de medição do tempo baseado em subdivisões. Eles consideraram uma fração que julgavam pequena em comparação com uma hora e a chamaram de _minuto_, do latim "prima minuta", significando "diminuído" ou "muito pequeno". Essa unidade representava, e ainda representa, um sexagésimo de uma hora.

Conforme a necessidade de medir intervalos ainda menores surgiu, eles subdividiram cada minuto em $60$ partes iguais. Essas unidades menores, durante o reinado da **Rainha Elizabeth I** no século XVI, foram denominadas _segundos minutos_. O termo "segundo" vem dessa expressão, que significa literalmente "a segunda diminuição". Ou seja, a sexagésima parte da sexagésima parte da hora.

Essa divisão sexagesimal do tempo, que persiste até hoje, tem suas raízes nos sistemas numéricos das antigas Suméria e Babilônia e reflete a longa história da nossa busca por medições de tempo cada vez mais precisas. E, principalmente, serve como um exemplo de grandezas de graus diferentes de pequenez que todos conseguem entender.

> A expressão latina "prima minuta", significando primeira parte pequena, usada por [Ptolomeu](https://en.wikipedia.org/wiki/Ptolemy) para dividir o círculo em $60$ partes. A palavra segundo, origina-se na expressão latina "secunda minuta" {: ntr}

Agora, se um minuto é tão pequeno em comparação com uma semana, quão menor é um segundo?

> Antes de seguirmos, precisamos de um pouco de contexto para entender o universo monetário na época do prof. Silvanus, pré-decimal e britânico, porque vou manter a analogia original de graus de pequenez que ele usou em seu livro.
>
> A libra esterlina (£1) era a unidade principal, dividida em 240 pennies. O "farthing", a menor unidade, valia 1/4 de um penny ou 1/960 de uma libra, derivando seu nome do inglês antigo "feorthing" (quarta parte). Havia também o "halfpenny", valendo metade de um penny. O "sovereign" ou soberano, uma moeda de ouro, equivalia a uma libra e era usado como reserva de valor e no mercado internacional. Embora não mencionada pelo prof. Silvanus, a "guinea" ou guinéu, valendo 21 xelins (uma libra e um xelim), era comumente usada para bens de luxo. >
>
> Eu não resisti a tentação e este complexo sistema monetário, parte integrante da vida cotidiana inglesa da virada do século XX, foi mantido neste texto por seu valor histórico, cultural e, principalmente porque é interessante. O "farthing", usado para transações de baixíssimo valor, foi desmonetizado em 1961, décadas após a escrita do livro.

Voltando ao conceito de pequenez. Pense em um "farthing" em comparação com um soberano. Ele, o "farthing", era pouco maior que a milésima parte da Libra. Um "farthing" a mais, ou a menos, terá pouca importância em comparação com um soberano e, certamente, pode ser considerado uma quantidade insignificante. Porém, podemos comparar um "farthing" com £1000.

Relativamente a £1000, o "farthing" não tem nem importância que tem em relação teria para um soberano. Mesmo um soberano pode ser relativamente insignificante quando comparado com valores que se contem aos milhões de libras.

O que a leitora deve ter percebido é que se escolhermos uma fração numérica como constituindo a proporção que, para qualquer propósito, chamaremos de pequeno. Será possível identificar outras frações de graus maiores de pequenez.

Voltando a medida do tempo, se a fração $\frac{1}{60}$ for considerada como uma _parte pequena_ de um tempo específico, como uma semana, ou um mês. Então, $\frac{1}{60}$ de $\frac{1}{60}$ (sendo uma _pequena fração de uma pequena fração_) será considerada uma _quantidade pequena da segunda ordem de pequenez_.

> Os matemáticos falam sobre a segunda ordem de "magnitude" (isto é: grandeza) quando realmente querem dizer segunda ordem de _pequenez_. Não se deixe confundir.

Ou, se para um propósito qualquer, pegássemos $1$ por cento (isto é: $\frac{1}{100}$) como uma _fração pequena_, então $1$ por cento de $1$ por cento (isto é: $\frac{1}{10.000}$) seria uma fração de _segunda ordem de grandeza, ou pequenez se preferir_. E $\frac{1}{1.000.000}$ seria uma pequena fração da _terceira ordem de grandeza_, representando $1$ por cento de $1$ por cento de $1$ por cento.

Por fim, suponha que para um propósito específico, deveremos considerar $\frac{1}{1.000.000}$ como _pequena_. Considere um cronômetro de primeira categoria que não deve perder, ou ganhar, mais de meio minuto em um ano, este cronômetro deverá manter o tempo com uma precisão de $1$ parte em $1.051.200$ que é próximo de um milionésimo. Agora, se para o tal propósito considerarmos $\frac{1}{1.000.000}$ como nossa quantidade pequena, então $\frac{1}{1.000.000}$ de $\frac{1}{1.000.000}$, ou seja, $\frac{1}{1.000.000.000.000}$ (ou um trilionésimo) será uma quantidade pequena da _segunda ordem de grandeza_, e pode ser completamente desconsiderado.

Veja que quanto menor for uma quantidade pequena em si, mais insignificante se torna a correspondente quantidade pequena da segunda ordem de grandeza. Portanto, sabemos que _Em nossas análises, poderemos justificadamente negligenciar as quantidades pequenas de segunda, terceira ou ordens superiores, desde que a quantidade pequena de primeira ordem seja considerada suficientemente insignificante por si só_.

O que definirá o que é "suficientemente pequena" depende do contexto do problema e da precisão requerida. Esta abordagem nos permite simplificar cálculos complexos, focando apenas nas quantidades que têm um impacto significativo no resultado final. No entanto, é crucial exercer julgamento cuidadoso ao decidir quais termos omitir, sempre considerando as implicações dessa simplificação na precisão global de nossa solução. A escolha deste valor que poderá ser desprezado irá depender do problema. Contudo, será um valor suficientemente pequeno para permitir que o problema possa ser resolvido com o cálculo.

Uma coisa importante, que não podemos esquecer, é que quantidades insignificantes, quando sujeitas a multiplicação por outro fator, podem se tornar relevantes. Isso ocorre se o fator de multiplicação seja, em si, grande em relação ao problema. Mesmo um "farthing" se torna importante se for multiplicado por valores na casa dos milhares ou milhões de vezes.

No cálculo, escrevemos $dx$ para representar uma fração de primeira ordem de pequenez de $x$. Essas coisas como $dx$, e $du$, e $dy$, serão chamadas de _diferenciais_, e lidas como o diferencial de $x$, ou o diferencial de $u$, ou o diferencial de $y$, conforme o caso. Ou ainda, podem ser chamados de a derivada de $x$, a derivada de $y$ ou a derivada de$u$. Mesmo que $dx$ seja um pedacinho de $x$, e considerado relativamente pequeno em si mesmo, para resolver o nosso problema, não podemos considerar que quantidades como $x \cdot dx$, ou $x^2 \, dx$, ou $a^x \, dx$ sejam insignificantes. Por outro lado, sem nenhuma dúvida, podemos ter certeza que $dx \cdot dx$ será insignificante. _No caso de $dx \cdot dx$ teremos uma quantidade pequena da segunda ordem_.

Um exemplo muito simples servirá como ilustração.

Vamos pensar em $x$ como uma quantidade que irá crescer apenas um pedacinho de modo que, em algum momento, teremos $x + dx$, onde $dx$ será um incremento pequeno, adicionado pelo crescimento. Fazendo o quadrado de $x + dx$ teremos $x^2 + 2x \, dx + (dx)^2$. Neste caso, segundo termo, $2x \, dx$ não é desprezível porque é uma quantidade de primeira ordem enquanto o terceiro termo, $(dx)^2$ é um termo de segunda ordem de pequenez, já que $(dx)^2 = dx\cdot dx$.

Usando frações para deixar mais claro, se a leitora fizer $dx$ ter um valor numérico, digamos, $\frac{1}{60}$ de $x$, então o segundo termo em $x^2 + 2x \, dx + (dx)^2$ seria $\frac{2x^2}{60}$, enquanto o terceiro termo seria $\frac{1}{3600}$. O terceiro termo é claramente menos significativo que o segundo.

Indo além, podemos considerar nosso pedacinho $dx$ como $\frac{1}{1000}$ de $x$, então o segundo termo de $x^2 + 2x \, dx + (dx)^2$ será $\frac{2x^2}{1000}$, enquanto o terceiro termo será apenas $\frac{1}{1.000.000}$.

_Aquilo que consideraremos pequeno irá depender do problema que desejamos solucionar, mas no cálculo o pequeno sempre será $dx$_.

![]({{ site.baseurl }}/assets/images/calc*Fig1.jpg)
\_Figura 1.1 - Um quadrado acrescido de $dx$.*

Talvez, uma ilustração usando alguns conceitos da geometria básica possa ajudar a visualização destes conceitos de níveis diferentes de grandeza.

Desenhe um quadrado cujo lado será $x$, Fig. 1.1.a. Suponha que o quadrado cresça devido a adição de uma pequena quantidade, um $dx$, ao seu tamanho em cada direção,Fig. 1.1.b. O quadrado ampliado será composto pelo quadrado original de área $x^2$, mais os dois retângulos na parte superior e na direita, cada um com área $xdx$ (ou juntos $2xdx$), e o pequeno quadrado no canto superior direito que é $(dx)^2$. Na Fig. 1b, representei $dx$ como uma fração muito grande de $x$, cerca de $\frac{1}{10}$, por razões didáticas e gráficas. Suponha que eu considere $dx$ como $\frac{1}{100}$, aproximadamente a espessura de uma linha desenhada com uma caneta fina. Se fizer isso, você não poderá ver o quadrado no canto superior direito. Entretanto, ele terá uma área de apenas $\frac{1}{10,000}$ de $x^2$. Claramente, neste caso, $(dx)^2$ será desprezível.

Vamos considerar outra analogia, desta vez, usando algo que dói na parte mais importante do corpo. O bolso.

Suponha que um milionário dissesse ao seu secretário: — na próxima semana, eu lhe darei uma pequena comissão de qualquer valor monetário que eu receba. Suponha que o secretário dissesse ao seu filho: — eu lhe darei uma parte de tudo que eu receber. Finalmente, suponha que a fração em cada caso seja $\frac{1}{100}$ (um centésimo). Se o Sr. Milionário receber $£1000$ na próxima semana, o secretário receberá $£10$ e o seu filho, um centésimo disso, equivalente a $2$ xelins.

> Um "xelin" é um 20 avos de uma libra, então, £10 correspondem a 200 "xelins". Um centésimo de 200 xelins\* é 2 "xelins".

Dez libras seriam uma quantidade pequena em comparação com $£1000$; mas dois xelins é insignificante. Uma quantidade muito pequena, uma quantidade de segunda ordem de grandeza. Qual seria esta relação se a fração, em vez de ser $\frac{1}{100}$, fosse $\frac{1}{1000}$ (um milésimo)? Neste caso, quando o Sr. Milionário recebesse suas $£1000$, o Sr. Secretário receberia apenas $£1$, e o garoto menos de um "farthing"!

O espirituoso Dean Swift[^1]{#nt1} uma vez escreveu:{#nt1}

> So, Nat'ralists observe, a Flea
> Hath smaller Fleas that on him prey.
> And these have smaller Fleas to bite 'em,
> And so proceed ad infinitum.

Talvez um boi possa se preocupar com uma pulga, uma criatura da primeira ordem de pequenez. Mas, provavelmente um boi não se incomodará com a pulga da pulga. Se esta criatura existir será uma criatura de segunda ordem de pequenez e, portanto insignificante para o boi. Talvez, mesmo uma quantidade gigantesca de pulgas de pulgas não teria muita importância para um boi.

E chega de analogias sobre a relatividade do que pode ser desprezado.

## III. SOBRE CRESCIMENTOS RELATIVOS

Durante todo estudo de cálculo, lidaremos com quantidades que estão variando, aumentando ou diminuindo, e com as suas taxas de variação.

Classificaremos todas as quantidades em duas classes: _constantes_ e _variáveis_. Aquelas que considerarmos de valor fixo, e chamaremos de _constantes_, e denotaremos algebricamente usando letras do início do alfabeto latino, como $a$, $b$ ou $c$. Enquanto aquelas que consideramos capazes de crescer ou diminuir, ou ainda, como dizem os matemáticos, capazes de "variar", serão denotadas por letras do final do alfabeto latino, como $x$, $y$, $z$, $u$, $v$, $w$ ou, muitas vezes, $t$.

Além disso, geralmente lidaremos com mais de uma variável ao mesmo tempo e seremos levados a considerar a forma como uma variável depende da outra. Por exemplo, para um determinado problema precisaremos considerar a forma como a altura atingida por um projétil depende do tempo necessário para atingir essa altura. Ou, seremos convidados a investigar um retângulo de área dada e a descobrir de que forma um aumento, ainda que pequeno, no comprimento dele implicará em uma redução na sua largura, de forma a manter a área constante. Ou ainda, ficaremos intrigados como uma variação qualquer na inclinação de uma escada implicará em qual variação na altura que ela atinge. Todos são problemas comuns ao estudo da física e geometria.

Suponha que tenhamos duas variáveis que dependem uma da outra. São variáveis que ocorrem em problemas tais que uma variação qualquer em uma delas causará uma alteração no valor da outra. _Vamos chamar uma das variáveis de $x$, a variável que não depende, e portanto é independente e de $y$ a variável que depende_.

Para manter esta linha de raciocínio, suponha agora que façamos $x$ variar, ou seja, alteraremos, ou imaginaremos que o valor de $x$ foi alterado, adicionando a ela uma fração muito pequena do seu valor. Fração esta que já chamamos de $dx$. Assim, faremos $x$ se tornar $x + dx$. Então, porque $x$ foi alterado, $y$ também terá sido alterado, e terá se tornado $y + dy$. O pequeno $dy$ poderá ser em alguns casos positivo, aumentando, em outros negativo, diminuindo. Contudo, não será, exceto em casos raros, do mesmo tamanho que $dx$.

![]({{ site.baseurl }}/assets/images/calc*Fig2.jpg)
\_Figura 2.1 - Um crescimento $dx$ em um triângulo.*{: class="legend"}

Para ilustrar, vamos analisar dois exemplos:

1. Vamos fazer com que $x$ e $y$ sejam, respectivamente, a base e a altura de um triângulo retângulo, Figura 2.1.a, cuja inclinação da hipotenusa esteja fixada em $30^\circ$. Supondo que este triângulo seja capaz de se expandir e ainda manter seus ângulos constantes. Quando a base do triângulo crescer de modo a se tornar $x + dx$, a altura se tornará $y + dy$. Como visto na Figura 2.2.a. Neste cenário, o aumento de $x$ resulta em um aumento de $y$. Observe ainda que o triângulo menor, cuja altura é $dy$, e cuja base é $dx$, é semelhante ao triângulo original. Deve ser óbvio que o valor da razão $\frac{dy}{dx}$ será o mesmo da razão $\frac{y}{x}$. Como o ângulo é $30^\circ$, é constante, veremos que:

$$
\frac{dy}{dx} = \frac{1}{1.73}.
$$

2. Considere a Figura 2.2.b, onde uma escada $AB$ (em lilás) de comprimento fixo está apoiada contra uma parede. Definiremos $x$ como a distância horizontal entre a base da escada (ponto A) e a parede, enquanto $y$ representará a altura alcançada pela escada na parede (ponto B). Observe que $y$ depende diretamente de $x$. Intuitivamente, se afastarmos a base da escada da parede (aumentando $x$), a extremidade superior da escada descerá (diminuindo $y$), como ilustrado na Figura 2.2.b (em azul). Matematicamente, podemos expressar isso da seguinte forma: se incrementarmos $x$ para $x + dx$, $y será reduzido para $y - dy$. Em outras palavras, um incremento positivo em $x$ resulta em um incremento negativo em $y$, demonstrando a relação inversa entre estas variáveis no contexto deste problema geométrico. Parece razoável, mas quão razoável? Suponha que a escada fosse tão longa que, quando a extremidade inferior $A$ estivesse a $50$ centímetros da parede, a extremidade superior $B$ alcançasse $4.5$ metros do chão. Se você puxar a extremidade inferior $1$ centímetro a mais, quanto a extremidade superior descerá?

Vamos colocar tudo em metros: $x = 0.5 \, m$ e $y = 4.5 \, m$. Agora, o incremento de $x$ que chamamos de $dx$ é de $0.01 \, m$. Sendo assim, $x + dx = 0.51 \, m$. Entendemos que o $x$, a variável independente, variar $x$ significa variar $y$.

Precisamos achar $y$ irá variar. Sabemos que a nova altura será $y - dy$, sem dúvida. Mas, temos que calcular este $dy$. Se calcularmos a altura pelo Teorema de Pitágoras poderemos encontrar o valor de $dy$. O comprimento da escada é:

$$
\sqrt{(4.5)^2 + (0.5)^2} = 4.52769 \text{ metros}.
$$

Claramente, então, a nova altura, que é $y - dy$, será calculada por:

$$
(y - dy)^2 = (4.52769)^2 - (0.51)^2 = 20.50026 - 0.2601 = 20.24016,
$$

$$
y - dy = \sqrt{20.24016} = 4.49557 \text{ metros}.
$$

Agora $y = 4.50000$, então $dy$ é $4.50000 - 4.49557 = 0.00443$ metros.

Vimos que fazer $dx$ aumentar de $0.01$ metros resultou em fazer $dy$ sofrer uma redução de $0.00443$ metros.

A razão de $dy$ para $dx$ pode ser declarada da seguinte forma:

$$
\frac{dy}{dx} = \frac{0.00443}{0.01} = 0.443.
$$

Ou, se preferirmos, como no enunciado eu escolhi $1 \, \test{cm}$ de deslocamento, poderemos considerar esta fração, como o nosso valor pequeno. Se fo assim, poderemos dizer que:

$$
\frac{dy}{dx} = \frac{0.443}{1} = 0.443.
$$

Também é possível perceber que, exceto em uma posição particular, $dy$ terá um tamanho diferente de $dx$. Que posição é essa?[^2]{#nt2}

Retornando ao cálculo diferencial, nosso foco nos dois casos anteriores se concentra em um conceito intrigante: a razão entre $dy$ e $dx$ quando ambos se aproximam de valores infinitamente pequenos. Mais precisamente, estamos investigando a proporção que $dy$ mantém em relação a $dx$ à medida que estas quantidades se tornam infinitesimais. Note que só podemos encontrar essa razão, $\frac{dy}{dx}$, quando $y$ e $x$ estão relacionados de alguma forma. Geralmente, nos problemas que estudamos escolhemos as variáveis de modo que sempre que $x$ varia, $y$ também varie.

Comparando os dois exemplos: No Exemplo 1, o triângulo apresenta uma relação diretamente proporcional entre sua base ($x$) e altura ($y$). Um aumento na base resulta em um aumento correspondente na altura. Já no Exemplo 2, a escada demonstra uma relação inversamente proporcional entre a distância de sua base à parede ($x$) e a altura alcançada ($y$). Conforme afastamos a base da escada da parede, a altura diminui, inicialmente de forma gradual. No entanto, à medida que $x$ continua aumentando, a taxa de diminuição de $y$ irá, progressivamente, acelerar. Esta diferença nas relações entre $x$ e $y$ nos dois exemplos ilustra como variações similares na variável independente ($x$) podem produzir efeitos drasticamente diferentes na variável dependente ($y$), dependendo da natureza específica do problema e da relação matemática envolvida.

Nos exemplos que vimos, as relações entre $x$ e $y$ podem ser perfeitamente definidas usando um pouco de álgebra e geometria. Se fizermos a análise destas relações, encontraremos $\frac{y}{x} = \tan 30^\circ$ para o Exemplo 1 e $x^2 + y^2 = L^2$ para o Exemplo 2, desde que $L$ seja o comprimento da escada. Finalmente, $\frac{dy}{dx}$ terá apenas o significado que encontramos em cada caso.

Um contra exemplo da importância da escolha das grandezas envolvidas no problema pode ser visto se $x$ for, como antes, a distância do pé da escada à parede e $y$ seja, em vez da altura alcançada, o comprimento horizontal da parede, ou número de tijolos na parede, ou ainda o número de anos desde que a parede foi construída. Neste contra exemplo, qualquer mudança em $x$ não causará nenhuma mudança em $y$; neste caso, $\frac{dy}{dx}$ não terá qualquer significado. Portanto, neste casos de grandezas para o $y$, não será possível encontrar uma expressão que relacione a variável independente $x$ com a variável dependente $y$. As grandezas que escolhemos, o comprimento horizontal da parede, ou número de tijolos na parede, ou ainda o número de anos desde que a parede foi construída não são dependentes de $x$.

Sempre que usarmos diferenciais $dx$, $dy$, $dz$, etc., a existência de algum tipo de relação entre $x$, $y$, $z$, será implícita, e essa relação será chamada de função de $x$, $y$, $z$, etc. As duas expressões dadas acima, $\frac{y}{x} = \tan 30^\circ$ e $x^2 + y^2 = l^2$, são funções de $x$ em $y$. A expressões de relação entre $x$ e $y$, que achamos nos Exemplos 1 e 2, contêm implicitamente os meios necessários a expressão $x$ em termos de $y$ ou $y$ em termos de $x$, e por essa razão são chamadas de "funções implícitas" em $x$ e $y$. As relações que encontramos podem ser colocadas nas formas:

$$y = x \tan 30^\circ \quad \text{ou} \quad x = \frac{y}{\tan 30^\circ}$$

$$\text{e} \quad y = \sqrt{l^2 - x^2} \quad \text{ou} \quad x = \sqrt{l^2 - y^2}$$.

Essas últimas expressões afirmam explicitamente o valor de $x$ em termos de $y$, ou de $y$ em termos de $x$ e, por essa razão, são chamadas de "funções explícitas" de $x$ ou $y$. Dessa forma, $x^2 + 3 = 2y - 7$ é uma função implícita em $x$ e $y$. A função $x^2 + 3 = 2y - 7$ pode ser escrita $y = \frac{x^2 + 10}{2}$ (função explícita de $x$) ou $x = \sqrt{2y - 10}$ (função explícita de $y$).

Uma função explícita em variáveis como $x$, $y$, $z$ é uma expressão cujo valor muda conforme essas variáveis mudam. O resultado calculado dessa função é denominado "variável dependente", pois seu valor depende das outras variáveis na função. Estas últimas são chamadas de "variáveis independentes", já que seus valores não são determinados pela função. Frequentemente, escolhemos $x$ como variável independente e $y$ como variável dependente. Por exemplo, na função $u = x^2 \sin{θ}$, $x$ e $θ$ são variáveis independentes, enquanto $u$ é a variável dependente.

Em certas situações, a relação precisa entre quantidades $x$, $y$, $z$ pode ser desconhecida ou inconveniente de expressar explicitamente. Nesses casos, sabemos ou podemos afirmar que existe alguma relação entre essas variáveis, de modo que a alteração de uma afeta as outras. Indicamos a existência de tal função usando notações como $f(x, y, z)$ para funções implícitas, ou $x = f(y, z)$, $y = f(x, z)$, $z = f(x, y)$ para funções explícitas. Às vezes, usamos letras como $F$ ou $\Phi$ no lugar de $f$. Assim, $y = f(x)$, $y = F(x)$ e $y = \Phi(x)$ significam que $y$ depende de $x$ de alguma forma não especificada, seja por desconhecimento ou por conveniência.

Chamamos a razão $\frac{dy}{dx}$ de "derivada de $y$ com respeito a $x$". Porém, formalmente deveríamos chamar de "coeficiente diferencial de $y$ com respeito a $x$". Para nós derivada de $y$ com respeito a $x$ será suficiente. Ainda assim, é um nome científico e solene, para uma coisa tão simples. Neste texto, não vamos nos assustar com nomes solenes. Em vez de nos assustarmos, passaremos à coisa simples, encontrar a razão $\frac{dy}{dx}$.

Na álgebra comum que você aprendeu na escola, a leitora estava sempre tentando encontrar alguma quantidade desconhecida que você chamava de $x$, ou $y$. Agora você terá que aprender a procurar estas quantidades usando formas de cálculo novas. O processo de encontrar o valor de $\frac{dy}{dx}$ é chamado de _diferenciação_. Não perca de vista que o que desejamos é o valor dessa razão quando tanto $dy$ quanto $dx$ são indefinidamente pequenos e relacionados um com o outro, esta é a última vez que vou lembrar isso. Juro!

_No cálculo estamos preocupados em encontrar a razão $\frac{dy}{dx}$ quando $dx$ e $dy$ são tão infinitesimalmente pequenos que tendem a zero_. Droga, jurei em falso!

Agora que tiramos o medo, e mostramos que os conceitos envolvidos são muito simples, podemos aprender como encontrar a razão $\frac{dy}{dx}$.

### Como ler diferenciais

Estimada leitora, é crucial que você não confunda $dx$ com o produto de $d$ e $x$. O símbolo $d$ não é um fator multiplicativo, mas sim um operador que indica "um elemento infinitesimal de" ou "uma variação infinitamente pequena de" qualquer quantidade que o segue. Assim, $dx$ representa uma porção infinitesimal da variável $x$. Na linguagem matemática, pronunciamos $dx$ como "de-xis". Compreender corretamente este conceito é essencial para dominar os princípios do cálculo e evitar erros comuns de interpretação.

Coeficientes diferenciais de segunda ordem, que serão estudados mais tarde, serão representados: $\frac{d^2y}{dx^2}$, o que é lido "de-dois-ipsilon de-xis-quadrado", significando que operação de diferenciar $y$ em relação a $x$ foi, ou tem que ser, feita duas vezes consecutivas. Esta é a notação desenvolvida por Leibnitz.

Outra maneira de indicar que uma função foi diferenciada é colocando uma aspa simples depois do símbolo da função. Assim, se $y = F(x)$, o que significa que $y$ é uma função não especificada em $x$, podemos escrever $f'(x)$ em vez de $\frac{d(f(x))}{dx}$, ou ainda $\frac{d}{dy}f(x)$, que eu gosto mais porque separa o coeficiente diferencial. Da mesma forma, podemos usar $f''(x)$ para significar que a função original $f(x)$ foi, ou terá que ser, diferenciada duas vezes consecutivas em relação a $x$.

Eu vou usar as notações $\frac{d}{dx}f(x)$, $\frac{dy}{dx}f(x)$ e $f'(x)$ livremente. Para que a leitora se sinta confortável com qualquer notação.

## IV. CASOS MAIS SIMPLES

A partir dos princípios fundamentais da álgebra, podemos diferenciar algumas expressões algébricas simples. Como queremos deixar tudo simples e fácil, eu vou usar apenas a álgebra que que a leitora deve no ensino médio, ou antes.

### CASO 1

Vamos começar com uma função simples $y = x^2$. Antes de começarmos _lembre-se de que os conceitos mais importantes para entender o cálculo é são as ideias de acréscimo e decréscimo_. Os matemáticos chamam isso de variação. Como $y$ é igual $x^2$, parece trivial observa que se $x$ cresce, $x^2$ também crescerá. É isso que significa ser igual.

O que queremos descobrir é a proporção entre o crescimento de $y$ e o crescimento de $x$. Em outras palavras: _nossa tarefa será descobrir a razão entre $dy$ e $dx$, que representamos na expressão $\frac{dy}{dx}$_.

Vamos fazer $x$ crescer apenas $dx$ e se tornar $x + dx$; da mesma forma, $y$ crescerá se tornará $y + dy$. Segundo a função $y = x^2$, será verdade que o $y$ aumentado será igual ao quadrado do $x$ aumentado. Escrevendo isso algebricamente teremos:

$$
y + dy = (x + dx)^2
$$

Resolvendo o quadrado $(x + dx)^2$, obteremos:

$$
y + dy = x^2 + 2x \cdot dx + (dx)^2
$$

O que $(dx)^2$ significa? Lembre-se de que $dx$ significava uma parte, um fração, muito pequena de $x$. Então, $(dx)^2$ significará uma fração de uma fração $x$; isto é, conforme explicado anteriormente, é uma quantidade da segunda ordem de pequenez, ou grandeza. Portanto, $(dx)^2$ pode ser descartado por ser insignificante em comparação com os outros termos. Ou seja:

$$
y + dy = x^2 + 2x \cdot dx
$$

Como sabemos que $y = x^2$, afinal é a relação original, vamos subtrair $y$ de um lado e $x^2$ da equação e ficamos com:

$$
dy = 2x \cdot dx
$$

Como estamos buscando uma razão, dividindo $dy$ por $dx$, por liberdade poética, e simplicidade, não que seja uma divisão, encontramos:

$$
\frac{dy}{dx} = 2x
$$

Finalmente encontramos a razão do crescimento de $y$ dado um crescimento em $x$. Neste caso, $2x$. E a leitora, se repetiu estes passos no caderno acaba de encontrar sua primeira derivada. Bom motivo para comer um chocolate.

> _Nota_ – Esta razão $\frac{dy}{dx}$ é o resultado de diferenciar $y$ com respeito a $x$. Diferenciar significa encontrar o coeficiente diferencial. Suponha que tivéssemos outra função de $x$, como, por exemplo, $u = 7x^2 + 3$. Então, se nos mandassem diferenciar isso com respeito a $x$, deveríamos encontrar $\frac{dx}{du}$, ou, o que é a mesma coisa que $\frac{d(7x^2 + 3)}{du}$. Muitas vezes teremos casos em que o tempo será a variável independente, por exemplo: $y = b + \frac{1}{2}at^2$. logo, se alguém a pedir para diferenciar esta função teríamos que encontrar o coeficiente diferencial de $y$ com respeito a $t$. Logo, nosso trabalho seria tentar encontrar $\frac{dy}{dt}$, isto é, encontrar $\frac{d(b + \frac{1}{2}at^2)}{dt}$.

Um bom exemplo numérico pode clarear os pensamentos e mostrar que cálculo não é o bicho papão debaixo da cama.

Considere novamente a função $y = x^2$. Suponha que $x$ tenha o valor inicial de $100$, resultando em $y = 10.000$. Agora, vamos aumentar o valor de $x$ em uma unidade, ou seja, $dx = 1$, de modo que $x$ se torne $101$. O novo valor de $y$ será $101^2 = 10.201$.

No entanto, podemos desprezar o termo de segunda ordem (o $1$ no final de $10.201$). Isso nos permite arredondar o novo valor de $y$ para $10.200$.

Assim, o aumento em $y$ (denotado por $dy$) é de aproximadamente $200$. Essa aproximação pode ser suficientemente precisa para diversos fins práticos.

_Observe que a qualidade da aproximação depende da magnitude da variação em $x$ (neste caso, $dx = 1$) em relação ao valor inicial de $x$ (neste caso, $x = 100$). Quanto menor for a variação em relação ao valor inicial, mais precisa será a aproximação_. A expressão desta relação será:

$$
\frac{dy}{dx} = \frac{200}{1} = 200
$$

De acordo com o trabalho algébrico do parágrafo anterior, havíamos encontrado $\frac{dy}{dx} = 2x$. E assim, para $x = 100$ e $2x = 200$.

A atenta leitora dirá: &#8212; Negligenciamos uma unidade inteira e eu acho que uma unidade não é coisa que se negligencie.

E não estará errada. Sempre podemos tentar novamente, desta vez vamos tornar $dx$ ainda menor.

Tentaremos com $dx = \frac{1}{10}$. Então $x + dx = 100,1$, e

$$
(x + dx)^2 = 100,1 \cdot 100,1 = 10.020,01.
$$

Agora, o último dígito $1$ é apenas um milionésimo de $10.000$ e é totalmente insignificante; então podemos tomar $10.020$ sem o pequeno decimal no final. E isso faz $dy = 20$; e

$$
\frac{dy}{dx} = \frac{20}{0,1} = 200,
$$

o que confirma o $\frac{dy}{dx} = 2x$ que encontramos algebricamente.

### CASO 2

Agora é a sua vez. A esforçada leitora deve tentar diferenciar $y = x^3$ usando a mesma técnica do CASO 1, deixando $y$ crescer para $y + dy$, enquanto $x$ cresce para $x + dx$. Vá lá! Eu aguardo.

Se a amável leitora tiver tentado, terá encontrado algo semelhante à:

$$
y + dy = (x + dx)^3
$$

Resolvendo este cubo, obteremos:

$$
y + dy = x^3 + 3x^2 \cdot dx + 3x(dx)^2 + (dx)^3
$$

Agora, como sabemos que podemos negligenciar quantidades da segunda e terceira ordens de grandeza, sempre que $dy$ e $dx$ são considerados infinitamente pequenos, $(dx)^2$ e $(dx)^3$ se tornarão infinitamente menores que $dy$ e $dx$. Assim, considerando-os insignificantes, teremos:

$$
y + dy = x^3 \cdot dx
$$

Como $y = x^3$; podemos subtrair $y$ dos dois lados encontrando:

$$
dy = 3x^2 \cdot dx
$$

Mas, o que queremos é uma relação. Logo:

$$
\frac{dy}{dx} = 3x^2
$$

### CASO 3

Este também é seu. tente diferenciar $y = x^4$. Começando exatamente como fizemos antes. Eu não me incomodo de aguardar.

A leitora deve ter feito tanto $y$ quanto $x$ crescerem um pouco. Neste caso, teremos:

$$
y + dy = (x + dx)^4
$$

Resolvendo a quarta potência, obteremos:

$$
y + dy = x^4 + 4x^3 \cdot dx + 6x^2 (dx)^2 + 4x(dx)^3 + (dx)^4
$$

Eliminando os termos que contêm as maiores potências de $dx$, por serem insignificantes, teremos:

$$
y + dy = x^4 + 4x^3 \cdot dx
$$

Subtraindo o original $y = x^4$, ficamos com:

$$
dy = 4x^3 \cdot dx
$$

Como o que buscamos é uma relação:

$$
\frac{dy}{dx} = 4x^3
$$

Todos os casos que analisamos até agora são muito simples, desde que você saiba manipular polinômios e tenha compreendido por que podemos desprezar termos em que $dx$ está elevado a potências maiores que $1$. Vamos organizar os resultados em na Tabela 4.1 para tentar identificar um padrão geral. Usaremos duas colunas: uma para os valores de $y$ e outra para os valores correspondentes de $\frac{dy}{dx}$:

| y     | $\frac{dy}{dx}$ |
| ----- | --------------- |
| $x^2$ | $2x$            |
| $x^3$ | $3x^2$          |
| $x^4$ | $4x^3$          |

_Tabela 4.1 - Uma comparação entre $x^n$ e sua derivada, $dx$._{: class="legend"}

Cuidadosamente observe a Tabela 4.1. Consegue achar uma relação entre $x$ e $\frac{dy}{dx}$ nesta tabela. Olhe cuidadosamente, a operação de diferenciar parece ter tido o efeito de diminuir a potência de $x$ em $1$ sempre que aplicada. Por exemplo: no último caso, reduzindo $x^4$ para $x^3$), e ao mesmo tempo, esta operação parece estar multiplicando $x$ pelo mesmo valor da potência original. Sem pressa, olhe os casos 1, 2 e 3, na Tabela 4.1 e verifique se este parágrafo faz sentido para você.

Se as observações sobre a Tabela 4.1 fizerem sentido para você, podemos conjecturar como outras potências serão diferenciadas. Se esta regra que inferimos estiver certa você deve achar que diferenciar $x^5$ resultaria em $5x^4$, ou que diferenciar $x^6$ resultaria $6x^5$. Se, a amável leitora, pensou assim, parabéns! É exatamente isso. Entretanto, todo mundo mente, e podemos conferir. Vamos tentar com $y = x^5$. Neste caso, usando o mesmo processo, teremos:

$$
y + dy = (x + dx)^5
$$

$$
= x^5 + 5x^4 \cdot dx + 10x^3 (dx)^2 + 10x^2 (dx)^3 + 5x (dx)^4 + (dx)^5.
$$

Ignorando todos os termos contendo pequenas quantidades de ordens superiores, teremos:

$$
y + dy = x^5 + 5x^4 \cdot dx,
$$

Subtraindo $y = x^5$, dos dois lados do igual, obteremos:

$$
dy = 5x^4 \cdot dx,
$$

ou seja,

$$
\frac{dy}{dx} = 5x^4,
$$

Exatamente como vimos observando a Tabela 4.1. Seguindo logicamente nossa observação, devemos concluir que, se quisermos lidar com qualquer potência, que chamaremos de $n$, poderíamos abordá-la da mesma maneira e seguir a mesma regra. Vamos fazer $y = x^n$ para generalizar, aplicando de forma matemática a regra que inferimos, teremos:

$$
\frac{dy}{dx} = nx^{(n-1)}.
$$

Por exemplo, se fizermos $n = 8$, então $y = x^8$; e diferenciarmos esta expressão, teremos:

$$
\frac{dy}{dx} = 8x^7.
$$

De fato, a atenta leitora acabou de aprender a sua primeira regra de diferenciação.

_Esta regra é chamada de Regra da Potência e diz que: diferenciar $x^n$ resultará $n \space x^{n-1}$, sendo válida todos os casos onde $n$ é um número inteiro e positivo_. Podemos demonstrar a validade da Regra da Potência acrescentando uma pequena fração de $x$ ao próprio $x$ e expandir $(x + dx)^n$ usando o Teorema Binomial de Newton:

#### Comprovando a Regra da Potência

O Teorema Binomial de Newton[^2]{#nt3} nos diz que podemos expandir a expressão $(x + dx)^8$ em uma série de termos envolvendo coeficientes binomiais. Aplicando a fórmula geral para o Teorema Binomial:

$$
(x + dx)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} (dx)^k.
$$

Onde $\binom{n}{k}$ é o coeficiente binomial dado por:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}.
$$

Que representa a combinação de $k$ de $8$ em $8$. Vamos considerar um exemplo prático para expandir $(x + dx)^8$ usando o Teorema Binomial.

##### Exemplo: Expansão de $(x + dx)^8$

A expansão de $(x + dx)^8$ usando o Teorema Binominal será dada por:

$$
(x + dx)^8 = \sum_{k=0}^{8} \binom{8}{k} x^{8-k} (dx)^k.
$$

Vamos calcular cada termo da soma individual e cuidadosamente:

1. Para $k = 0$: $\binom{8}{0} x^{8-0} (dx)^0 = 1 \cdot x^8 \cdot 1 = x^8$.

2. Para $k = 1$: $\binom{8}{1} x^{8-1} (dx)^1 = 8 \cdot x^7 \cdot dx = 8x^7 dx$.

3. Para $k = 2$: $\binom{8}{2} x^{8-2} (dx)^2 = 28 \cdot x^6 \cdot (dx)^2 = 28x^6 (dx)^2$.

4. Para $k = 3$: $\binom{8}{3} x^{8-3} (dx)^3 = 56 \cdot x^5 \cdot (dx)^3 = 56x^5 (dx)^3$.

5. Para $k = 4$: $\binom{8}{4} x^{8-4} (dx)^4 = 70 \cdot x^4 \cdot (dx)^4 = 70x^4 (dx)^4$.

6. Para $k = 5$: $\binom{8}{5} x^{8-5} (dx)^5 = 56 \cdot x^3 \cdot (dx)^5 = 56x^3 (dx)^5$.

7. Para $k = 6$: $\binom{8}{6} x^{8-6} (dx)^6 = 28 \cdot x^2 \cdot (dx)^6 = 28x^2 (dx)^6$.

8. Para $k = 7$: $\binom{8}{7} x^{8-7} (dx)^7 = 8 \cdot x^1 \cdot (dx)^7 = 8x (dx)^7$.

9. Para $k = 8$: $\binom{8}{8} x^{8-8} (dx)^8 = 1 \cdot x^0 \cdot (dx)^8 = (dx)^8$.

Somando todos os termos, obtemos a expansão completa:

$$
(x + dx)^8 = x^8 + 8x^7 dx + 28x^6 (dx)^2 + 56x^5 (dx)^3 + 70x^4 (dx)^4 + 56x^3 (dx)^5 + 28x^2 (dx)^6 + 8x (dx)^7 + (dx)^8.
$$

Portanto, o polinômio equivalente a $(x + dx)^8$ será:

$$
(x + dx)^8 = x^8 + 8x^7 dx + 28x^6 (dx)^2 + 56x^5 (dx)^3 + 70x^4 (dx)^4 + 56x^3 (dx)^5 + 28x^2 (dx)^6 + 8x (dx)^7 + (dx)^8.
$$

Para simplificar podemos usar o mesmo método que usamos nos exemplos numéricos e calcular a derivada de $y=x^8$. Como já sabemos a expansão de $(x+dx)$^8 podemos começar removendo todas as potências de $dx$ de ordem maior que $1$, logo:

$$
y+dy = x^8+8x^7
$$

Novamente, se removermos $y$ dos dois lados o sinal encontraremos o resultado esperado, que confirma nossa regra:

$$
\frac{dy}{dx} = 8x^7
$$

Infelizmente, a questão de saber se esta regra é verdadeira para casos onde $n$ tem valores negativos, ou fracionários, ainda requer considerações adicionais.

### Caso de uma potência negativa

Vamos fazer que nossa função seja $y = x^{-2}$. Como sou otimista, vamos proceder exatamente como fizemos antes e ver onde chegamos.

$$
y + dy = (x + dx)^{-2},
$$

$$
= x^{-2} \left(1 + \frac{dx}{x}\right)^{-2}.
$$

O binômio com o Teorema Binomial, obteremos:

$$
= x^{-2} \left[1 - \frac{2 \, dx}{x} + \frac{2(2 + 1)}{1 \cdot 2} \left(\frac{dx}{x}\right)^2 - \text{etc.}\right],
$$

$$
= x^{-2} - 2x^{-3} \cdot dx + 3x^{-4} (dx)^2 - 4x^{-5} (dx)^3 + \text{etc.}.
$$

Negligenciando as pequenas quantidades de ordens superiores de grandeza, teremos:

$$
y + dy = x^{-2} - 2x^{-3} \cdot dx.
$$

Subtraindo o original $y = x^{-2}$, encontramos:

$$
dy = -2x^{-3} \cdot dx,
$$

$$
\frac{dy}{dx} = -2x^{-3}.
$$

E, veja que nosso resultado ainda está de acordo com a Regra da Potência.

### Caso de uma potência fracionária

Vamos tentar com $y = x^{\frac{1}{2}}$. Então, mais uma vez, vamos tentar como antes:

$$
y + dy = (x + dx)^{\frac{1}{2}} = x^{\frac{1}{2}} \left(1 + \frac{dx}{x}\right)^{\frac{1}{2}},
$$

$$
= \sqrt{x} + \frac{1}{2} \frac{dx}{\sqrt{x}} - \frac{1}{8} \frac{(dx)^2}{x \sqrt{x}} + \text{termos com potências mais altas} dx.
$$

Subtraindo o original $y = x^{\frac{1}{2}}$, e ignorando as potências mais altas, teremos:

$$
dy = \frac{1}{2} \frac{dx}{\sqrt{x}} = \frac{1}{2} x^{-\frac{1}{2}} \cdot dx,
$$

e, finalmente teremos:

$$
\frac{dy}{dx} = \frac{1}{2} x^{-\frac{1}{2}}.
$$

O que novamente concorda com a regra que inferimos.

Para resumir: chegamos à seguinte regra: _Para diferenciar $x^n$, multiplique pela potência e reduza a potência em um, resultando em $nx^{n-1}$_.

### EXERCÍCIOS I

Resolva os exercícios a seguir usando apenas as técnicas algébricas que vimos até o momento:

1. $y = x^{13} $   Resposta: $\frac{dy}{dx} =13x^{12}$.
2. $y = x^{-\frac{3}{2}}$ Resposta: $\frac{dy}{dx} = -\frac{3}{2} x^{-\frac{5}{2}}$.
3. $y = x^{2a} $   Resposta: $\frac{dy}{dx} =2a x^{2a-1}$.
4. $u = t^{2.4} $  Resposta: $\frac{dy}{dx} =2.4 t^{1.4}$.
5. $z = \sqrt[3]{u} $  Resposta: $z = \frac{dy}{dx} =\frac{1}{3} u^{-\frac{2}{3}}$.
6. $y = \sqrt[3]{x^{-5}} $   Resposta:  $\frac{dy}{dx} =-\frac{5}{3} x^{-\frac{8}{3}}$.
7. $u = \sqrt{\frac{1}{x^8}} $   Resposta: $\frac{dy}{dx} =\frac{du}{dx} = -4x^{-5}$.
8. $y = 2x^{a} $   Resposta: $\frac{dy}{dx} = 2a x^{a-1}$.
9. $y = \sqrt[3]{x^3} $  Resposta: $\frac{dy}{dx} = 1$.
10. $y = \sqrt{\frac{1}{x^m}} $  Resposta: $\frac{dy}{dx} = -\frac{m}{2} x^{-\frac{m+2}{2}}$.

Se resolveu os exercícios e chegou neste ponto, a amável leitora deve ser capaz de diferenciar funções que sejam apenas potências de $x$. Viu como é fácil usando apenas a álgebra que aprendeu no ensino médio?

## V. PRÓXIMA ETAPA. O QUE FAZEMOS COM CONSTANTES

Até agora, em nossas funções, geralmente consideramos $x$ como uma quantidade que aumenta, e, como resultado, $y$ também aumenta. Usamos $x$ para representar qualquer quantidade que podemos variar. E, considerando a variação de $x$ como uma espécie de causa, consideramos a variação resultante de $y$ como um efeito. Em outras palavras, o valor de $y$ depende do valor de $x$. Tanto $x$ quanto $y$ são variáveis, mas $x$ é a variável que controlamos, a variável independente, enquanto $y$ é a variável dependente. Em todo o capítulo anterior, buscamos regras para determinar a proporção entre a variação em $y$ (dependente) e a variação em $x$ (independente).

Nosso próximo passo será descobrir como as constantes (valores que não mudam quando $x$ ou $y$ mudam) afetam o processo de diferenciação.

### Constantes Adicionadas

Vamos começar estudando um caso simples: o caso da constante adicionada a uma função. Neste caso, poderemos ter funções como:

$$
y = x^3 + 5.
$$

Assim como antes, vamos supor que $x$ varie para $x + dx$ e $y$ varie para $y + dy$.

Então: $y + dy = (x + dx)^3 + 5$, que expandido será:

$$= x^3 + 3x^2 dx + 3x(dx)^2 + (dx)^3 + 5 $$.

Descartando as pequenas quantidades de ordens superiores, teremos:

$$y + dy = x^3 + 3x^2 \cdot dx + 5 $$.

Subtraindo $y = x^3 + 5$, teremos:

$$dy = 3x^2 dx $$,

$$\frac{dy}{dx} = 3x^2 $$.

Portanto, o $5$, o valor da constante adicionada a função, desapareceu completamente. o $5$ não acrescentou nada ao crescimento de $x$ e, sendo assim, não entra no coeficiente diferencial. Se tivéssemos colocado $7$, $800$, ou qualquer outro número, no lugar do $5$, o resultado seria o mesmo. Quando usamos a letra $a$, ou $b$, ou $c$ para representar qualquer constante, esta letra simplesmente desaparecerá quando diferenciarmos a função.

Só para ressaltar: se a constante adicional tivesse sido de valor negativo, como $-5$ ou $-b$, também teria desaparecido. E chegamos a outra regra importante: _a derivada da constante é zero_

### Constantes Multiplicadas

Vamos considerar um experimento simples. Seja $y = 7x^2$. Prosseguindo como antes, obteremos:

$$y + dy = 7(x + dx)^2 $$

$$= 7(x^2 + 2x \cdot dx + (dx)^2) $$

$$= 7x^2 + 14x \cdot dx + 7(dx)^2 $$.

Então, subtraindo $y = 7x^2$, e ignorando o último termo, teremos:

$$dy = 14x \cdot dx $$

$$\frac{dy}{dx} = 14x $$.

Vamos ilustrar este exemplo traçando os gráficos das equações $y = 7x^2$ e $\frac{dy}{dx} = 14x$, atribuindo a $x$ um conjunto de valores sucessivos, 0, 1, 2, 3, etc., e encontrando os valores correspondentes de $y$ e de $\frac{dy}{dx}$.

Esses valores podem tabulados da seguinte forma:

| $x$             | 0   | 1   | 2   | 3   | 4   | 5   | -1  | -2  | -3  |
| --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $y$             | 0   | 7   | 28  | 63  | 112 | 175 | 7   | 28  | 63  |
| $\frac{dy}{dx}$ | 0   | 14  | 28  | 42  | 56  | 70  | -14 | -28 | -42 |

_Tabela 5.1 - Comparação entre $f(x)=7x^2$ e sua derivada $\frac{dy}{dx}$._{: class="legend"}

O que nos permite gerar o Gráfico 5.1:

![]({{ site.baseurl }}/assets/images/graf-51.jpg){: class="lazyimg"}
_Gráfico 5.1 - Um crescimento $dx$ em um triângulo._{: class="legend"}

No Gráfico 5.1 vemos os valores de $x$, os correspondentes valores de $y$ e as duas curvas que nos interessam: em azul a função $f(x)=y=7x^2$ e em laranja sua derivada, a relação $\frac{dy}{dx} = 14x$.

No Gráfico 5.1 compare cuidadosamente as duas curvas traçadas. Observe que a altura da ordenada, eixo $y$, e verifique por inspeção que a altura da ordenada da curva derivada, em laranja, é proporcional à inclinação da curva original, em azul. À esquerda da origem, onde a curva da função original inclina negativamente (isto é, para baixo da esquerda para a direita, onde a função é decrescente), as ordenadas correspondentes da curva derivada são negativas.

Agora, se olharmos novamente o Caso 1, veremos que diferenciar $x^2$ nos dará $2x$. Assim, o coeficiente diferencial de $7x^2$ é $7$ vezes maior que o de $x^2$. Se tivéssemos tomado $8x^2$, o coeficiente diferencial teria sido $8$ vezes maior que o coeficiente diferencial de $x^2$. Se fizermos a função $y = ax^2$, obteremos:

$$
\frac{dy}{dx} = a \cdot 2x.
$$

Se tivéssemos começado com $y = ax^n$, deveríamos ter encontrado:

$$
\frac{dy}{dx} = a \cdot nx^{n-1}.
$$

Então, qualquer multiplicação por uma constante reaparece como uma mera multiplicação do coeficiente diferencial quando a função é diferenciada. Que pode ser escrito na forma de regra: _a derivada do produto entre uma constante e uma função é produto da constante e a derivada da função_. O que é verdadeiro para a multiplicação será igualmente verdadeiro para a divisão. Se no exemplo acima, tivéssemos usado a constante $\frac{1}{7}$ em vez de $7$, teríamos o mesmo $\frac{1}{7}$ no resultado após a diferenciação.

**Alguns Exemplos Adicionais**. A seguir estão alguns exemplos, totalmente resolvidos, que permitirão que você domine completamente o processo de diferenciação aplicado a expressões algébricas comuns, e permitirão que você resolva por si só os exercícios sugeridos no final deste capítulo. Refaça-os, cuidadosamente, em um caderno.

1. Diferencie $y = \frac{x^5}{7} - \frac{3}{5}$. Onde:

$$
\frac{3}{5}
$$

É uma constante adicionada e, como tal, desaparece. Sendo assim, Podemos escrever:

$$
\frac{dy}{dx} = \frac{1}{7} \cdot x^{5},
$$

ou, usando a Regra da Potência:

$$
\frac{dy}{dx} = \frac{1}{7} \cdot 5 \cdot x^{5-1},
$$

o que finalmente resulta em:

$$
\frac{dy}{dx} = \frac{5}{7} x^4.
$$

1. Diferencie $y = a \sqrt{x} - \frac{1}{2} \sqrt{a}$.

O termo $\frac{1}{2} \sqrt{a}$ desaparece, sendo uma constante adicionada. Como $a \sqrt{x}$, pode ser escrito como $ax^{\frac{1}{2}}$, teremos:

$$
\frac{dy}{dx} = a \cdot \frac{1}{2} \cdot x^{\frac{1}{2} - 1} = \frac{a}{2} \cdot x^{-\frac{1}{2}},
$$

ou

$$
\frac{dy}{dx} = \frac{a}{2 \sqrt{x}}.
$$

3. Se $ay + bx = by - ax + (x + y) \sqrt{a^2 - b^2}$, encontre o coeficiente diferencial de $y$ em relação a $x$.

Uma expressão desse tipo precisará de um pouco mais de conhecimento do que adquirimos até agora. No entanto, sempre vale a pena verificar se a expressão pode ser colocada em uma forma mais simples.

Primeiro, devemos tentar colocá-la na forma$y=$ alguma expressão envolvendo apenas$x$. Já que é a forma que já sabemos derivar.

A expressão $ay + bx = by - ax + (x + y) \sqrt{a^2 - b^2}$ pode ser escrita como:

$$
(a - b)y + (a + b)x = (x + y) \sqrt{a^2 - b^2}.
$$

Elevando ao quadrado, obtemos

$$
(a - b)^2 y^2 + (a + b)^2 x^2 + 2(a + b)(a - b)xy = (x^2 + y^2 + 2xy)(a^2 - b^2),
$$

o que pode ser simplificado em:

$$
(a - b)^2 y^2 + (a + b)^2 x^2 = x^2 (a^2 - b^2) + y^2 (a^2 - b^2),
$$

ou

$$
[(a - b)^2 - (a^2 - b^2)] y^2 = [(a^2 - b^2) - (a + b)^2] x^2.
$$

ou seja:

$$
2b(b - a)y^2 = -2b(b + a)x^2,
$$

portanto,

$$
y = \sqrt{\frac{a + b}{a - b}} x \quad \text{e} \quad \frac{dy}{dx} = \sqrt{\frac{a + b}{a - b}}.
$$

Crie alguns outros exemplos para si mesmo e tente diferenciá-los. E vamos ver alguns problemas de outras disciplinas, onde usamos a diferenciação:

4. O volume de um cilindro de raio $r$ e altura $h$ é dado pela fórmula $V = \pi r^2 h$. Encontre a taxa de variação do volume em relação ao raio quando $r = 5.5 \, \text{cm}$ centímetros e $h = 20\, \text{cm}$. Se $r = h$, encontre as dimensões do cilindro para que uma mudança de $1\, \text{cm}$ no raio cause uma mudança de $400 \, \text{cm}^3$ no volume.

O problema pede para encontrar a taxa de variação do volume de um cilindro em relação ao seu raio, dado que o volume é $V = \pi r^2 h$. Em seguida, pede as dimensões do cilindro para que uma mudança de $1 \, \text{cm}$ no raio cause uma mudança de $400 \, \text{cm}^3$ no volume, quando $r = h$.

Podemos começar com a taxa de variação do volume em relação ao raio. Esta taxa de variação será obtida pela derivada:

$$\frac{d(\pi r^2 h)}{dr} = 2 \pi rh$$

Substituindo $r = 5.5\, \text{cm}$ e $h = 20\, \text{cm}$, obteremos:

$$\frac{dV}{dr} = 2 \pi (5.5)(20) = 691.15 \, \text{cm}^3/\text{cm}$$

Isso quer dizer que, para um cilindro com raio de $5.5\, \text{cm}$ e altura de $20\, \text{cm}$, o volume aumenta aproximadamente $691.15 \, \text{cm}^3$ para cada aumento de $1\, \text{cm}$ no raio. Observe que esta taxa de variação é específica para os valores dados de $$e $h$.

A segunda parte do problema pede as dimensões do cilindro para uma mudança específica no volume:

Se $r = h$, teremos:

$$\frac{d(\pi r^2 h)}{dr} = 2 \pi r^2$$

Queremos que uma mudança de $1\, \text{cm}$ no raio cause uma mudança de $400 \, \text{cm}^3$ no volume, ou seja:

$$2 \pi r^2 = 400$$

Resolvendo para $r$:

$$r = \sqrt{\frac{400}{2 \pi}} = 7.98 \, \text{cm}$$

Como $r = h$, as dimensões do cilindro são:

$$r = h = 7.98 \, \text{cm}$$

5. A leitura, $\theta$, de um "Pirômetro de Radiação Féry"[^4]{#nt4} está relacionada à temperatura em graus Celsius $t$ do corpo observado pela relação dada por:

$$
\frac{\theta}{\theta_1} = \left( \frac{t}{t_1} \right)^4
$$

Onde $\theta_1$ é a leitura correspondente a uma temperatura conhecida $t_1$ do corpo observado. Compare a sensibilidade do pirômetro às temperaturas $800^\circ \text{C}$, $1000^\circ \text{C}$, $1200^\circ \text{C}$, dado que ele leu $2$ quando a temperatura era $1000^\circ \text{C}$.
Precisamos lembrar que a sensibilidade é a taxa de variação da leitura com a temperatura, ou seja, $\frac{d \theta}{dt}$.

A fórmula pode ser escrita como:

$$
\theta = \frac{\theta_1}{t_1^4} t^4 = \frac{2 t^4}{1000^4},
$$

onde $\theta_1 = 2$ e $t_1 = 1000$. Simplificando a equação, teremos:

$$
\theta = \frac{2 t^4}{(10^3)^4} = \frac{2 t^4}{10^{12}} = \frac{t^4}{5 \cdot 10^{11}}.
$$

Agora, diferenciamos $\theta$ em relação a $t$:

$$
\frac{d\theta}{dt} = \frac{d}{dt} \left( \frac{t^4}{5 \cdot 10^{11}} \right) = \frac{4 t^3}{5 \cdot 10^{11}}.
$$

Vamos calcular a sensibilidade para $t = 800$, $1000$ e $1200$:

Para $t = 800$:

$$
\frac{d\theta}{dt} \bigg|_{t=800} = \frac{4 \cdot 800^3}{5 \cdot 10^{11}} = \frac{4 \cdot 512 \cdot 10^6}{5 \cdot 10^{11}} = \frac{2048 \cdot 10^6}{5 \cdot 10^{11}} = 0,04096.
$$

Para $t = 1000$:

$$
\frac{d\theta}{dt} \bigg|_{t=1000} = \frac{4 \cdot 1000^3}{5 \cdot 10^{11}} = \frac{4 \cdot 10^9}{5 \cdot 10^{11}} = 0,08.
$$

Para $t = 1200$:

$$
\frac{d\theta}{dt} \bigg|_{t=1200} = \frac{4 \cdot 1200^3}{5 \cdot 10^{11}} = \frac{4 \cdot 1.728 \cdot 10^9}{5 \cdot 10^{11}} = \frac{6.912 \cdot 10^9}{5 \cdot 10^{11}} = 0,13824.
$$

A sensibilidade será aproximadamente o dobro em $1000^\circ \text{C}$ com relação a $800^\circ \text{C}$, e torna-se aproximadamente $1,73$ vezes maior novamente até $1200^\circ \text{C}$.

### EXERCÍCIOS II

Aplique os conceitos de derivação que vimos até agora para resolver os seguintes exercícios e problemas:

1. $ y = ax^3 + 6$. Resposta: $\frac{dx}{dy} = 3ax^2$
2. $ y = 13x^{\frac{3}{2}} - c$. Resposta: $\frac{dx}{dy} = \frac{39}{2} x^{\frac{1}{2}}$
3. $ y = 12x^{\frac{1}{2}} + c^{\frac{1}{2}}$. Resposta: $\frac{dx}{dy} = 6x^{-\frac{1}{2}} $
4. $ y = c^{\frac{1}{2}} x^{\frac{1}{2}}$. Resposta: $\frac{dx}{dy} = \frac{1}{2}c^{\frac{1}{2}}x^{-\frac{1}{2}}$
5. $ u = \frac{az^n - 1}{c}$. Resposta: $frac{du}{dy} = \frac{an}{c}z^{n-1}$
6. $ y = 1,18t^2 + 22,4$. Resposta: $frac{dx}{dy} = $2,36t$

7. Se $l_t$ e $l_0$ forem os comprimentos de uma barra de ferro nas temperaturas $ t^\circ \text{C}$ e $0^\circ \text{C}$, respectivamente, então $l_t = l_0 (1 + 0.000012t)$. Encontre a variação do comprimento da barra por grau Celsius. Resposta: $frac{dx}{dy} = 0.000012\cdot l_0$

8. Foi constatado que se $c$ for a potência de uma lâmpada incandescente e $V$ for a voltagem, $c = aV^b$, onde $a$ e $b$ são constantes. Encontre a taxa de variação da potência luminosa com a voltagem e calcule a mudança de potência luminosa por volt em 80, 100 e 120 volts no caso de uma lâmpada para a qual $a = 0.5 \cdot 10^{-10}$ e $b = 6$. Resposta: $
\frac{dC}{dV} = abV^{b-1}, \, 0.98, \, 3.00 \, \text{e} \, 7.47 \, \text {watts por volt, respectivamente}$

9. A frequência $n$ de vibração de uma corda de diâmetro $D$, comprimento $L$ e gravidade específica $\sigma$, esticada com uma força $T$, é dada por

$$
n = \frac{1}{DL} \sqrt{\frac{gT}{\pi \sigma}}.
$$

Encontre a taxa de variação da frequência quando $D$, $L$, $\sigma$ e $T$ são variáveis independentes.

Resposta:

$$
\frac{dn}{dD} = -\frac{1}{LD^2} \sqrt{\frac{gT}{\pi \sigma}}, \quad \frac{dn}{dL} = -\frac{1}{DL^2} \sqrt{\frac{gT}{\pi \sigma}},
$$

$$
\frac{dn}{d\sigma} = -\frac{1}{2DL} \sqrt{\frac{gT}{\pi \sigma^3}}, \quad \frac{dn}{dT} = \frac{1}{2DL} \sqrt{\frac{g}{\pi \sigma T}}.
$$

10. A maior pressão externa $P$ que um tubo pode suportar sem colapsar é dada por

$$
P = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3}{D^3},
$$

onde $E$ e $\sigma$ são constantes, $t$ é a espessura do tubo e $D$ é seu diâmetro. (Esta fórmula assume que $4t$ é pequeno em comparação com $D$.)

a) Compare a taxa em que $P$ varia para uma pequena mudança de espessura e para uma pequena mudança de diâmetro ocorrendo separadamente.

b) Compare a taxa na qual $P$ varia para uma pequena mudança de espessura e para uma pequena mudança de diâmetro ocorrendo separadamente.

Resposta:

$$
\frac{\text{Taxa de variação de } P \text{ quando } t \text{ varia}}{\text{Taxa de variação de } P \text{ quando } D \text{ varia}} = -\frac{D}{t}
$$

11. Encontre, a partir dos primeiros princípios, a taxa na qual os seguintes variam em relação a uma mudança no raio:
    (a) - a circunferência de um círculo de raio $r$;
    (b) - a área de um círculo de raio $r$;
    (c) - a área lateral de um cone de dimensão inclinada $l$;
    (d) - o volume de um cone de raio $r$ e altura $h$;
    (e) - a área de uma esfera de raio $r$;
    (f) - o volume de uma esfera de raio $r$.

Resposta: $ \quad 2\pi, \, 2\pi r, \, \pi l, \, \frac{2}{3} \pi rh, \, 8\pi r, \, 4\pi r^2$

12. O comprimento $L$ de uma barra de ferro na temperatura $T$ é dado por

$$
L = l_t \left[ 1 + 0.000012 (T - t) \right],
$$

onde $l_t$ é o comprimento na temperatura $t$. Encontre a taxa de variação do diâmetro $D$ de um pneu de ferro adequado para ser encolhido em uma roda, quando a temperatura $T$ varia.

Resposta: $ \quad \frac{dD}{dT} = \frac{0.000012 l_t}{\pi}$

## VI. ADIÇÕES, SUBTRAÇÕES, PRODUTOS e QUOCIENTES

Agora que a amável leitora já sabe como diferenciar funções algébricas simples, tais como $x^2 + c$ ou $ax^4$, temos que aprender como lidar com as operações aritméticas entre duas ou mais funções. Começando pela Adição.

### Adição e Subtração

Por exemplo, seja:

$$y = (x^2 + c) + (ax^4 + b) $$

Qual será a $\frac{dy}{dx}$ de $y = (x^2 + c) + (ax^4 + b) $? Como devemos proceder com essa nova tarefa?

A resposta a esta questão é imediata: basta diferenciar cada termo separadamente, um após o outro, assim, teremos:

$$\frac{dy}{dx} = 2x + 4ax^3. \ (\text{Resposta}) $$

Se você tiver dúvidas sobre a correção do cálculo acima, tente um caso mais geral, resolvendo-o pelos princípios fundamentais que vimos anteriormente. Parece complicado, mas não é. veja:

Seja $y = u + v$, onde $u$ é qualquer função de $x$, e $v$ qualquer outra função de $x$. Então, permitindo que $x$ aumente para $x + dx$, $y$ aumentará para $y + dy$; e $u$ aumentará para $u + du$; e $v$ para $v + dv$. Neste caso, teremos:

$$y + dy = u + du + v + dv$$.

Subtraindo o original $y = u + v$, obteremos:

$$dy = du + dv$$,

e dividindo por $dx$, obteremos:

$$\frac{dy}{dx} = \frac{du}{dx} + \frac{dv}{dx}$$.

O que justifica nosso procedimento. Ou seja, você deve diferenciar cada função separadamente e somar os resultados. Esta é a _Regra da Soma que pode ser expressa como a derivada da soma de funções e a soma das derivadas das funções_.

Se voltarmos ao exemplo do parágrafo anterior e colocarmos os valores das duas funções estão sendo somadas, usando uma das notações mostradas anteriormente para a diferenciação, teremos:

$$\frac{dy}{dx} = \frac{d}{dx}(x^2 + c) + \frac{d}{dx}(ax^4 + b) $$

$$\frac{dy}{dx}= 2x + 4ax^3 $$

Exatamente como encontramos antes.

Se tivéssemos três funções em $x$, que poderíamos chamar de $u$, $v$ e $w$, de modo que:

$$y = u + v + w$$.

Sendo assim:

$$\frac{dy}{dx} = \frac{du}{dx} + \frac{dv}{dx} + \frac{dw}{dx}$$.

A leitora já deve ter percebido que não há problema algum quanto a subtração de funções. _Se a função $f$ qualquer tiver um sinal negativo, seu coeficiente diferencial também seria negativo_. Assim, ao diferenciar $y = u - v$, obteremos:

$$\frac{dy}{dx} = \frac{du}{dx} - \frac{dv}{dx}$$.

### Produto e quociente

Quando tratamos de **Produtos de Funções**, a simplicidade começa a se dirigir a janela. Mas, não sai. Suponha que a amável leitora tivesse que diferenciar a expressão:

$$y = (x^2 + c) \cdot (ax^4 + b)$$,

Por onde, a esforçada leitora começaria?

Posso adiantar que o resultado não será $2x \cdot 4ax^3$, pois é fácil ver que nem $c \cdot ax^4$, nem $x^2 \cdot b$ teriam sido considerados nesse resultado. Na verdade, existem duas formas de resolvermos este problema.

#### Primeira forma

Esta é a forma instintiva, de quem conhece álgebra. O instinto diz para fazer a multiplicação das funções e depois de resolver a diferenciação. Desta forma: multiplicamos $x^2 + c$ e $ax^4 + b$. Obtendo: $ax^6 + acx^4 + bx^2 + bc$.

Agora que não temos multiplicação de funções, e podemos diferenciar começando com a regra da soma, que vimos acima, obtendo:

$$\frac{dy}{dx} = 6ax^5 + 4acx^3 + 2bx$$.

#### Segunda forma

Existe uma regra para diferenciar produtos de funções que frequentemente é mais eficiente do que multiplicar as funções e depois diferenciar o resultado. Considere a função:

$$y = u \cdot v$$,

onde tanto $u$ quanto $v$ são funções de $x$. É crucial entender que _$u$ e $v$ são duas funções distintas, ambas dependentes de $x$_.

Neste cenário específico, se $x$ variar para $x + dx$, as outras variáveis mudarão da seguinte forma:

- $y$ se tornará $y + dy$
- $u$ se tornará $u + du$
- $v$ se tornará $v + dv$

Esta relação só é válida quando $u$ e $v$ são funções de $x$, e é a base para a regra do produto na diferenciação.

Como $x$ variou para $x+dx$, teremos:

$$y + dy = (u + du) \cdot (v + dv) $$.

Agora, fazendo o produto $(u + du) \cdot (v + dv)$ obteremos:

$$= u \cdot v + u \cdot dv + v \cdot du + du \cdot dv$$.

Não deixe escapar que $ du \cdot dv $ é uma quantidade pequena de segunda ordem de grandez, e, portanto, como fizemos até agora, pode ser descartada. Logo:

$$y + dy = u \cdot v + u \cdot dv + v \cdot du$$.

A seguir, subtraindo o original $ y = u \cdot v $, teremos:

$$dy = u \cdot dv + v \cdot du $$,

e, dividindo por $dx$, obtemos o resultado:

$$\frac{dy}{dx} = u \frac{dv}{dx} + v \frac{du}{dx}$$.

Este esforço algébrico permitiu encontrar uma nova regra de diferenciação: _para diferenciar o produto de duas funções, multiplique cada função pelo coeficiente diferencial da outra, e some os dois produtos assim obtidos. Que pode ser memorizada usando a seguinte estrutura: $uv = vdu+udv$ que se lê: $u$ vezes $v$ é igual a $vdu$ mais $udv$_. Que chamaremos de Regra do produto.

Certamente a atenta leitora notou que esse processo equivale a tratar $u$ como constante enquanto diferencia $v$; depois tratar $v$ como constante enquanto diferencia $u$. Assim, o coeficiente diferencial total $\frac{dy}{dx}$ será a soma desses dois tratamentos. Agora que temos uma regra para diferenciar um produto de funções podemos voltar a $y = (x^2 + c) \cdot (ax^4 + b)$.

Queremos diferenciar o produto:

$$(x^2 + c) \cdot (ax^4 + b)$$.

Precisamos de $u$ e $v$. Para tal, basta fazer $(x^2 + c) = u$ e $(ax^4 + b) = v$. Usando a Regra do Produto de Funções, podemos escrever:

$$\frac{dy}{dx} = (x^2 + c) \frac{d(ax^4 + b)}{dx} + (ax^4 + b) \frac{d(x^2 + c)}{dx}$$,

$$= (x^2 + c) 4ax^3 + (ax^4 + b) 2x $$,

$$= 4ax^5 + 4acx^3 + 2ax^5 + 2bx $$,

$$\frac{dy}{dx} = 6ax^5 + 4acx^3 + 2bx $$,

O que confirma o resultado que encontramos a0 diferenciarmos usando a primeira forma, quando multiplicamos os binômios antes de derivar.

Para diferenciar quocientes, consideraremos a seguinte função:

$$y = \frac{bx^5 + c}{x^2 + a}$$

Neste caso, não é possível simplificar a divisão algebricamente, pois $x^2 + a$ não divide $bx^5 + c$ e não há fatores comuns. Portanto, devemos aplicar os princípios que aprendemos para desenvolver uma regra geral para a diferenciação de quocientes.

Comecemos simplificando o problema usando as funções genéricas $u$ e $v$:

$$y = \frac{u}{v}$$

onde $u$ e $v$ são funções de $x$. Quando $x$ se torna $x + dx$, teremos:

$$y + dy = \frac{u + du}{v + dv}$$

Agora, realizamos uma divisão algébrica:

$$\frac{u + du}{v + dv} = \frac{u + du}{v} \cdot \frac{1}{1 + \frac{dv}{v}} = \frac{u + du}{v} \cdot (1 - \frac{dv}{v}) = \frac{u + du}{v} - \frac{(u + du) \cdot dv}{v^2}$$

Expandindo:

$$\frac{u + du}{v} - \frac{u \cdot dv + du \cdot dv}{v^2}$$

$$= \frac{u + du}{v} - \frac{u \cdot dv}{v^2} - \frac{du \cdot dv}{v^2}$$

Dividindo por $dx$:

$$\frac{dy}{dx} = \frac{du}{dx} \cdot \frac{1}{v} + \frac{u}{v} \cdot \frac{dv}{dx} - \frac{u \cdot dv}{v^2 \cdot dx} - \frac{du \cdot dv}{v^2 \cdot dx}$$

Os termos contendo $du \cdot dv$ são de segunda ordem e podem ser negligenciados. Assim:

$$y + dy = \frac{u}{v} + \frac{du}{v} - \frac{u \cdot dv}{v^2}$$

Que pode ser reescrito como:

$$= \frac{u}{v} + \frac{v \cdot du - u \cdot dv}{v^2}$$

Subtraindo o $y$ original:

$$dy = \frac{v \cdot du - u \cdot dv}{v^2}$$

Portanto:

$$\frac{dy}{dx} = \frac{v \frac{du}{dx} - u \frac{dv}{dx}}{v^2}$$

Esta equação final fornece a regra para diferenciar um quociente de duas funções:

1. Multiplique o denominador pelo coeficiente diferencial do numerador.
2. Subtraia o produto do numerador pelo coeficiente diferencial do denominador.
3. Divida o resultado pelo quadrado do denominador.

Em resumo: _"$u$ dividido por $v$ é igual a $vdu$ menos $udv$, tudo dividido por $v^2$"_.

Finalmente podemos tentar diferenciar a função que propomos no início deste dilema: $y = \frac{bx^5 + c}{x^2 + a}$. Vamos considerar que: $bx^5 + c = u$ e $x^2 + a = v$. Se fizermos assim, teremos:

$$\frac{dy}{dx} = \frac{(x^2 + a) \frac{d(bx^5 + c)}{dx} - (bx^5 + c) \frac{d(x^2 + a)}{dx}}{(x^2 + a)^2} $$

$$= \frac{(x^2 + a)(5bx^4) - (bx^5 + c)(2x)}{(x^2 + a)^2} $$

$$\frac{dy}{dx} = \frac{3bx^6 + 5abx^4 - 2cx}{(x^2 + a)^2}. \ (\text{Resposta}) $$

A resolução da diferencial de quocientes é, muitas vezes tediosa e trabalhosa, mas não é difícil! Alguns exemplos adicionais totalmente resolvidos, apresentados a seguir, devem permitir que a amável leitora, sedimente este processo. Este é o momento de ser paciente e persistente. Refaça todos os exemplos no seu caderno.

### Exemplos

1. Diferenciar $ y = \frac{a}{b^2}x^3 - \frac{a^2}{b}x + \frac{a^2}{b^2} $.

   Sendo uma constante, o último termo $\frac{a^2}{b^2}$ desaparece, e teremos:

   $$\frac{dy}{dx} = \frac{a}{b^2} \cdot 3 \cdot x^{3-1} - \frac{a^2}{b} \cdot 1 \cdot x^{1-1}$$.

   Como $ x^{1-1} = x^0 = 1 $ obteremos:

   $$\frac{dy}{dx} = \frac{3a}{b^2} x^2 - \frac{a^2}{b}$$.

2. Diferenciar $ y = 2a\sqrt{bx^3} - \frac{3b\sqrt{a}}{x} - 2\sqrt{ab}$.

Colocando $ x $ na forma de fração, obteremos:

$$y = 2a\sqrt{b}x^{\frac{3}{2}} - 3b\sqrt{a}x^{-1} - 2\sqrt{ab}$$.

Agora, teremos:

$$\frac{dy}{dx} = 2a\sqrt{b} \cdot \frac{3}{2} \cdot x^{\frac{3}{2} - 1} - 3b\sqrt{a} \cdot (-1) \cdot x^{-1-1}$$,

ou, finalmente:

$$\frac{dy}{dx} = 3a\sqrt{bx} + \frac{3b\sqrt{a}}{x^2}$$.

3. Diferenciar $ z = 1,8 \sqrt[3]{\frac{1}{\theta^2}} - \frac{4,4}{\sqrt[5]{\theta}} - 27^\circ $.

Essa função pode ser escrita como: $ z = 1,8 \theta^{-\frac{2}{3}} - 4,4 \theta^{-\frac{1}{5}} - 27^\circ $.

Os, por serem constantes, $ 27^\circ $ desaparecem, e teremos:

$$\frac{dz}{d\theta} = 1.8 \cdot -\frac{2}{3} \cdot \theta^{-\frac{2}{3}-1} - 4.4 \cdot \left(-\frac{1}{5}\right) \theta^{-\frac{1}{5}-1}$$

ou:

$$\frac{dz}{d\theta} = -1,2 \theta^{-\frac{5}{3}} + 0,88 \theta^{-\frac{6}{5}}$$,

ou,

$$\frac{dz}{d\theta} = \frac{0,88}{\sqrt[5]{\theta^6}} - \frac{1,2}{\sqrt[3]{\theta^5}} $$

4. Diferenciar $ v = (3t^2 - 1,2t + 1)^3$.

Uma forma direta de resolver essa diferencial será explicada mais tarde. Contudo, com o que já sabemos, podemos resolve-la sem nenhuma dificuldade. Desenvolvendo o cubo, obtemos

$$v = 27t^6 - 32,4t^5 + 39,96t^4 - 23,328t^3 + 13,32t^2 - 3,6t + 1$$,

portanto,

$$\frac{dv}{dt} = 162t^5 - 162t^4 + 159.84t^3 - 69.984t^2 + 26.64t - 3.6$$.

5. Diferenciar $ y = (2x - 3)(x + 1)^2 $.

$$\frac{dy}{dx} = (2x - 3) \frac{d[(x + 1)(x + 1)]}{dx} + (x + 1)^2 \frac{d(2x - 3)}{dx} $$

$$= (2x - 3) \left[ (x + 1) \frac{d(x + 1)}{dx} + (x + 1) \frac{d(x + 1)}{dx} \right] + (x + 1)^2 \frac{d(2x - 3)}{dx} $$

$$= 2(x + 1) [(2x - 3) + (x + 1)] = 2(x + 1)(3x - 2) $$

ou, simplesmente, multiplique e depois diferencie.

6. Diferenciar $ y = 0.5x^3(x - 3) $.

$$\frac{dy}{dx} = 0.5 \left[ x^3 \frac{d(x - 3)}{dx} + (x - 3) \frac{d(x^3)}{dx} \right] $$

$$= 0.5 \left[ x^3 + (x - 3) \cdot 3x^2 \right] = 2x^3 - 4.5x^2$$.

Valem as mesmas observações que fizemos no exemplo anterior.

7. Diferenciar $ w = \left( \theta + \frac{1}{\theta} \right) \left( \sqrt{\theta} + \frac{1}{\sqrt{\theta}} \right) $.

Podemos escrever esta função como:

$$w = (\theta + \theta^{-1})(\theta^{\frac{1}{2}} + \theta^{-\frac{1}{2}})$$.

O que nos levará a:

$$\frac{dw}{d\theta} = (\theta + \theta^{-1}) \frac{d(\theta^{\frac{1}{2}} + \theta^{-\frac{1}{2}})}{d\theta} + (\theta^{\frac{1}{2}} + \theta^{-\frac{1}{2}}) \frac{d(\theta + \theta^{-1})}{d\theta} $$

$$= (\theta + \theta^{-1}) \left( \frac{1}{2} \theta^{-\frac{1}{2}} - \frac{1}{2} \theta^{-\frac{3}{2}} \right) + (\theta^{\frac{1}{2}} + \theta^{-\frac{1}{2}})(1 - \theta^{-2}) $$

$$= \frac{1}{2} (\theta^{\frac{1}{2}} + \theta^{-\frac{3}{2}} - \theta - \theta^{-\frac{5}{2}}) + \frac{1}{2} \left( \theta^{\frac{1}{2}} - \theta^{-\frac{1}{2}} \right) $$

$$= \frac{3}{2} \left( \sqrt{\theta} - \frac{1}{\sqrt[5]{\theta}} \right) + \frac{1}{2} \left( \frac{1}{\sqrt{\theta}} - \frac{1}{\sqrt[3]{\theta}} \right)$$.

Este resultado poderia ter sido obtido mais simplesmente multiplicando os dois fatores primeiro e diferenciando depois. Este processo, no entanto, nem sempre é possível. Veja, por exemplo, o exemplo 8, no qual a regra para diferenciar um produto _deve_ ser usada.

8. Diferenciar $ y = \frac{a}{1 + a\sqrt{x} + a^2 x} $.

$$\frac{dy}{dx} = \frac{(1 + ax^{\frac{1}{2}} + a^2 x) \cdot 0 - a \frac{d(1 + ax^{\frac{1}{2}} + a^2 x)}{dx}}{(1 + a\sqrt{x} + a^2 x)^2} $$

$$= -\frac{a \left(\frac{1}{2} ax^{-\frac{1}{2}} + a^2 \right)}{(1 + a\sqrt{x} + a^2 x)^2}$$.

9. Diferenciar $ y = \frac{x^2}{x^2 + 1} $.

$$\frac{dy}{dx} = \frac{(x^2 + 1) \cdot 2x - x^2 \cdot 2x}{(x^2 + 1)^2} = \frac{2x}{(x^2 + 1)^2}$$.

10. Diferenciar $ y = \frac{a + \sqrt{x}}{a - \sqrt{x}} $.

Na forma de índice, $ y = \frac{a + x^{\frac{1}{2}}}{a - x^{\frac{1}{2}}} $.

$$\frac{dy}{dx} = \frac{(a - x^{\frac{1}{2}}) \left(\frac{1}{2} x^{-\frac{1}{2}}\right) - (a + x^{\frac{1}{2}}) \left(-\frac{1}{2} x^{-\frac{1}{2}}\right)}{(a - x^{\frac{1}{2}})^2} $$

$$= \frac{a - x^{\frac{1}{2}} + a + x^{\frac{1}{2}}}{2(a - x^{\frac{1}{2}})^2 x^{\frac{1}{2}}}$$,

portanto,

$$\frac{dy}{dx} = \frac{a}{(a - \sqrt{x})^2 \sqrt{x}}$$.

11. Diferenciar:

$$\theta = \frac{1 - a\sqrt[3]{t^2}}{1 + a\sqrt[3]{t^3}}$$

Primeiro, vamos reescrever as raízes cúbicas usando notação de expoente fracionário:

$$\theta = \frac{1 - a(t^2)^{\frac{1}{3}}}{1 + a(t^3)^{\frac{1}{3}}}$$

Simplificando os expoentes:

$$\theta = \frac{1 - at^{\frac{2}{3}}}{1 + at^{\frac{3}{3}}} = \frac{1 - at^{\frac{2}{3}}}{1 + at^1} = \frac{1 - at^{\frac{2}{3}}}{1 + at}$$

Agora podemos aplicar a regra do quociente, onde $u = 1 - at^{\frac{2}{3}}$ e $v = 1 + at$:

$$\frac{d\theta}{dt} = \frac{v\frac{du}{dt} - u\frac{dv}{dt}}{v^2}$$

Calculando as derivadas:

$\frac{du}{dt} = -\frac{2}{3}at^{-\frac{1}{3}}$ e $\frac{dv}{dt} = a$

Substituindo na fórmula:

$$\frac{d\theta}{dt} = \frac{(1 + at) \left(-\frac{2}{3}at^{-\frac{1}{3}}\right) - (1 - at^{\frac{2}{3}}) (a)}{(1 + at)^2}$$

Expandindo:

$$= \frac{-\frac{2}{3}at^{-\frac{1}{3}} - \frac{2}{3}a^2t^{\frac{2}{3}} - a + at^{\frac{2}{3}}}{(1 + at)^2}$$

Simplificando:

$$= \frac{-\frac{2}{3}at^{-\frac{1}{3}} - a - \frac{2}{3}a^2t^{\frac{2}{3}} + at^{\frac{2}{3}}}{(1 + at)^2}$$

$$= \frac{-2at^{-\frac{1}{3}} - 3a - 2a^2t^{\frac{2}{3}} + 3at^{\frac{2}{3}}}{3(1 + at)^2}$$

Fatorando $a$:

$$\frac{d\theta}{dt} = \frac{-a(2t^{-\frac{1}{3}} + 3 + 2at^{\frac{2}{3}} - 3t^{\frac{2}{3}})}{3(1 + at)^2}$$

Esta é a forma final da derivada. Se desejarmos, podemos retornar à notação de raiz cúbica:

$$\frac{d\theta}{dt} = \frac{-a(2\frac{1}{\sqrt[3]{t}} + 3 + 2a\sqrt[3]{t^2} - 3\sqrt[3]{t^2})}{3(1 + at)^2}$$

12. Um reservatório de seção transversal quadrada tem lados inclinados em um ângulo de $45°$ com a vertical. O lado da base é $200\, \text{m}$. Encontre uma expressão para a quantidade que entra ou sai quando a profundidade da água varia em $1\, \text{m}$; portanto, encontre, em litros, a quantidade retirada por hora quando a profundidade é reduzida de $14$ para $10\, \text{m}$ em 24 horas.

O volume de um tronco de pirâmide de altura $H$, e de bases $A$ e $a$, será:

$$V = \frac{H}{3} (A + a + \sqrt{Aa})$$.

Vê-se facilmente que, sendo a inclinação $45°$, se a profundidade for $h$, o comprimento do lado da superfície quadrada da água é $200 + 2h$ metros, de modo que o volume de água será:

$$\frac{h}{3} \left[ 200^2 + (200 + 2h)^2 + 200(200 + 2h) \right] = 40,000h + 800h^2 + \frac{4h^3}{3}$$.

A taxa de variação do volume com relação à profundidade $h$ é dada por:

$$\frac{dV}{dh} = 40,000 + 1600h + 4h^2 $$,

em metros cúbicos por metro de variação de profundidade. O nível médio de $14$ para $10$ metros será $12$ metros. Quando $ h = 12 $,

$$\frac{dV}{dh} = 40,000 + 1600 \cdot 12 + 4 \cdot 12^2 = 50,176 \, \text{m}^3$$.

A quantidade de água em metros cúbicos que sai do reservatório ao se reduzir a profundidade de $4$ metros em $24$ horas será dada por:

$$
\Delta V \approx \frac{dV}{dh} \bigg|_{h=12} \cdot \Delta h = 50,176 \, \text{m}^3/\text{m} \cdot 4 \, \text{m} = 200,704 \, \text{m}^3
$$.

  Convertendo metros cúbicos para litros ($1$ metro cúbico $= 1000$ litros):

  $$200,704 \, \text{m}^3 = 200,704,000 \, \text{litros}$$.

  Portanto, a quantidade retirada por hora é de


$$

\frac{200,704 \, \text{m}^3 \cdot 1000 \, \text{litros/m}^3}{24 \, \text{horas}} = 8,362,666.67 \, \text{litros/hora}

$$
.

13. A pressão absoluta, em atmosferas, $P$, do vapor saturado na temperatura $t^\circ \, \text{C}$. foi determinada por [Dulong](https://en.wikipedia.org/wiki/Pierre_Louis_Dulong) como sendo $P = \left( \frac{40 + t}{140} \right)^5$ desde que $t$ esteja acima de $80^\circ \, \text{C}$. Encontre a taxa de variação da pressão com a temperatura a $100^\circ \, \text{C}$.

  Expanda o numerador usando o Teorema Binomial de Newton:

  $$P = \frac{1}{140^5} (40^5 + 5 \cdot 40^4 t + 10 \cdot 40^3 t^2 + 10 \cdot 40^2 t^3 + 5 \cdot 40t^4 + t^5)$$,

  portanto

  $$\frac{dP}{dt} = \frac{1}{537,824 \cdot 10^5} (5 \cdot 40^4 + 20 \cdot 40^3 t + 30 \cdot 40^2 t^2 + 20 \cdot 40t^3 + 5t^4)$$

  Quando $ t = 100 $ a função resulta em $ 0.036 $ atmosferas por grau centrígrado de variação de temperatura.

### EXERCÍCIOS III

1. Diferencie:

  (a) $ u = 1 + x + \frac{x^2}{1 \cdot 2} + \frac{x^3}{1 \cdot 2 \cdot 3} + \cdots. $
(b) $ y = ax^2 + bx + c. $
(c) $ y = (x + a)^2. $
(d) $ y = (x + a)^3. $

  Resposta: (a) $1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \frac{x^4}{24} + \cdots$; (b) $2ax + b$.; (c) $2x + 2a$.; (d) $3x^2 + 6ax + 3a^2$.

2. Se $ w = at - \frac{1}{2}bt^2 $, encontre $\frac{dw}{dt}$.

  Resposta: $\frac{dw}{dt} = a - bt$.

3. Encontre o coeficiente diferencial de:

  $$y = (x + \sqrt{-1}) \cdot (x - \sqrt{-1}) $$

  Resposta: $\frac{dy}{dx} = 2x$.

4. Diferencie:

  $$y = (197x - 34x^2) \cdot (7 + 22x - 83x^3) $$

  Resposta: $14110x^4 - 65404x^3 - 2244x^2 + 8192x + 1379$

5. Se $ x = (y + 3) \cdot (y + 5) $, encontre $\frac{dx}{dy}$.

  Resposta: $\frac{dx}{dy} = 2y + 8$.

6. Diferencie $ y = 1.3709x \cdot (112.6 + 45.202x^2) $.

  Resposta: $185.9022654x^2 + 154.36334$.

  Encontre os coeficientes diferenciais de

7. $ y = \frac{2x + 3}{3x + 2}. $

  Resposta: $\frac{-5}{(3x + 2)^2}$

8. $ y = \frac{1 + x + 2x^2 + 3x^3}{1 + x + 2x^2}. $

  Resposta: $\frac{6x^4 + 6x^3 + 9x^2}{(1 + x + 2x^2)^2}$.

9. $ y = \frac{ax + b}{cx + d}. $

  Resposta: $\frac{ad - bc}{(cx + d)^2}$.

10. $ y = \frac{x^n + a}{x^{-n} + b}. $

  Resposta: $\frac{anx^{-n-1} + bnx^{n-1} + 2nx^{-1}}{(x^{-n} + b)^2}$.

11. A temperatura $ t $ do filamento de uma lâmpada elétrica incandescente está relacionada à corrente que passa pela lâmpada pela relação:

  $$C = a + bt + ct^2 $$

  Encontre uma expressão que forneça a variação da corrente correspondente a uma variação de temperatura.

  Resposta: $b + 2ct$.

12. As seguintes fórmulas foram propostas para expressar a relação entre a resistência elétrica $ R $ de um fio na temperatura $ t^\circ $ C, e a resistência $ R_0 $ desse mesmo fio a $ 0^\circ $ Centígrados, sendo $ a $, $ b $, $ c $ constantes.

  $$R = R_0 (1 + at + bt^2)$$.
$$R = R_0 (1 + at + b\sqrt{t})$$.
$$R = R_0 (1 + at + bt^2)^{-1}$$.

  Encontre a taxa de variação da resistência em relação à temperatura conforme dada por cada uma dessas fórmulas.

  Resposta: $R_0(a + 2bt), \quad R_0 \left( a + \frac{b}{2\sqrt{t}} \right), \quad - \frac{R_0(a + 2bt)}{(1 + at + bt^2)^2} \quad \text{ou} \quad \frac{R^2(a + 2bt)}{R_0}$.

13. A força eletromotriz $ E $ de um certo tipo de célula padrão tem sido encontrada variando com a temperatura $ t $ de acordo com a relação

  $$E = 1.4340 \left[ 1 - 0.000814(t - 15) + 0.000007(t - 15)^2 \right] \text{ volts}$$.

  Encontre a variação da força eletromotriz por grau, a $ 15^\circ $, $ 20^\circ $ e $ 25^\circ $.

  Resposta: $1.4340(0.000014t - 0.001024), \quad -0.00117, \quad -0.00107, \quad -0.00097$.

14. A força eletromotriz necessária para manter um arco elétrico de comprimento $ l $ com uma corrente de intensidade $ i $ foi encontrada pela Sra. Ayrton como sendo

  $$E = a + bl + \frac{c + kl}{i}$$,

  onde $ a $, $ b $, $ c $, $ k $ são constantes. Encontre uma expressão para a variação da força eletromotriz (a) com relação ao comprimento do arco; (b) com relação à intensidade da corrente.

  Resposta: $\frac{dE}{dl} = b + \frac{k}{i}, \quad \frac{dE}{di} = - \frac{c + kl}{i^2}$.

## VII. DIFERENCIAÇÃO SUCESSIVA

A diferenciação sucessiva de uma função é um conceito fundamental no cálculo. Este processo envolve aplicar o operador de derivação repetidamente à mesma função, gerando uma série de novas funções, cada uma representando uma taxa de variação de ordem superior. Para compreender melhor este conceito e a terminologia associada, vamos examinar passo a passo o efeito de diferenciar uma função múltiplas vezes. Ilustraremos este processo com alguns exemplos específicos, permitindo-nos observar os padrões emergentes e as implicações matemáticas de cada derivação sucessiva.

Vamos fazer com que $y = x^5$.

Primeira diferenciação, usando a regra da potência:


$$

5x^4

$$

Segunda diferenciação, derivando a equação resultante da primeira derivada, usando a regra da potência:


$$

5 \cdot 4x^3 = 20x^3

$$

Terceira diferenciação, derivando a equação resultante da segunda derivada, novamente usando a regra da potência:


$$

5 \cdot 4 \cdot 3x^2 = 60x^2

$$

Quarta diferenciação, aqui a atenta leitora já percebeu o padrão:


$$

5 \cdot 4 \cdot 3 \cdot 2x = 120x

$$

Quinta diferenciação:


$$

5 \cdot 4 \cdot 3 \cdot 2 \cdot 1 = 120

$$

Sexta diferenciação:


$$

0

$$

Mesmo que eu não tenha destacado isso, estamos, eventualmente usando uma notação para as funções. Já empregamos, aqui ou ali, o símbolo geral $f(x)$ para indicar função em $x$. Aqui, o símbolo $f()$ é lido como "função de", sem especificar qual função particular está sendo considerada. Assim, a declaração $y = f(x)$ apenas nos diz que $y$ é uma função de $x$; pode ser $x^2$ ou $ax^n$, ou $\cos x$ ou qualquer outra função mais complicada, ou assustadora, de $x$.

*O símbolo correspondente para o coeficiente diferencial, definido por [Lagrange](https://pt.wikipedia.org/wiki/Joseph-Louis_Lagrange), é $f'(x)$, que é mais simples de escrever do que $\frac{dy}{dx}$. Isso é chamado de "função derivada" de $x$. A notação específica $\frac{d}{dx} f(x)$ foi criado por [Leibnitz](https://pt.wikipedia.org/wiki/Gottfried_Leibniz)*.

Suponha que diferenciássemos novamente; obteremos a "segunda função derivada" ou o segundo coeficiente diferencial, ou ainda , a segunda derivada, que será denotado por $f''(x)$; e assim por diante.

Agora vamos generalizar.

Seja $y = f(x) = x^n$.

Primeira diferenciação, $f'(x) = nx^{n-1}$.

Segunda diferenciação, $f''(x) = n(n-1)x^{n-2}$.

Terceira diferenciação, $f'''(x) = n(n-1)(n-2)x^{n-3}$.

Quarta diferenciação, $f''''(x) = n(n-1)(n-2)(n-3)x^{n-4}$.

etc., etc.

Mas esta não é a única maneira de indicar diferenciações sucessivas. Pois, se a função original for $y = f(x)$;

diferenciando uma vez, obteremos:


$$

\frac{dy}{dx} = f'(x);

$$

diferenciando duas vezes, obteremos:


$$

\frac{d}{dx} \left( \frac{dy}{dx} \right) = f''(x);

$$

e isso é mais convenientemente escrito como:


$$

\frac{d^2 y}{(dx)^2}, \quad \text{ou mais usualmente} \quad \frac{d^2 y}{dx^2}.

$$

Similarmente, podemos escrever o resultado da diferenciação três vezes como:


$$

\frac{d^3 y}{dx^3} = f'''(x).

$$

### Exemplos

Agora, vamos tentar $y = f(x) = 7x^4 + 3.5x^3 - \frac{1}{2}x^2 + x - 2$.

Primeira derivada:


$$

\frac{dy}{dx} = f'(x) = 28x^3 + 10.5x^3 - x + 1,

$$

Segunda derivada:


$$

\frac{d^2 y}{dx^2} = f''(x) = 84x^2 + 21x - 1,

$$

Terceira derivada:


$$

\frac{d^3 y}{dx^3} = f'''(x) = 168x + 21,

$$

Quarta derivada:


$$

\frac{d^4 y}{dx^4} = f''''(x) = 168,

$$


$$

\frac{d^5 y}{dx^5} = f''''(x) = 0.

$$

De maneira semelhante, se $y = \phi(x) = 3x(x^2 - 4)$,

Primeira derivada:


$$

\phi'(x) = \frac{dy}{dx} = 3 \left[ x \cdot 2x + (x^2 - 4) \cdot 1 \right] = 3(3x^2 - 4),

$$

Segunda derivada:


$$

\phi''(x) = \frac{d^2 y}{dx^2} = 3 \cdot 6x = 18x,

$$

Terceira derivada:


$$

\phi'''(x) = \frac{d^3 y}{dx^3} = 18,

$$

Quarta derivada:


$$

\phi''''(x) = \frac{d^4 y}{dx^4} = 0.

$$

### Exercícios IV

Encontre $\frac{dy}{dx}$ e $\frac{d^2 y}{dx^2}$ para as seguintes expressões:

(1) $y = 17x + 12x^2$.

(2) $y = \frac{x^2 + a}{x + a}$.

(3) $y = 1 + \frac{x}{1} + \frac{x^2}{1 \cdot 2} + \frac{x^3}{1 \cdot 2 \cdot 3} + \frac{x^4}{1 \cdot 2 \cdot 3 \cdot 4}$.

(4) Encontre a segunda e terceira derivadas nas funções derivadas no Exercício III e nos exemplos dados.

## VIII. QUANDO O TEMPO VARIA

Alguns dos problemas mais importantes resolvidos com o cálculo são aqueles em que o tempo é a variável independente, e temos que averiguar os valores de alguma outra grandeza que varia quando o tempo varia. Algumas coisas crescem à medida que o tempo passa; outras coisas diminuem. A distância que um trem percorreu desde seu ponto de partida continua aumentando à medida que o tempo passa. As árvores crescem mais altas à medida que os anos passam. Qual está crescendo a uma taxa maior: uma planta de $30 \, \text{cm}$ de altura que em um mês chega a $4 \, \text{cm}$, ou uma árvore de $12 \, \text{m}$ de altura que em um ano se chega a $14 \, \text{m}$?

Neste capítulo, vamos fazer uso, descabido e descontrolado, da palavra **taxa**. Usaremos a palavra taxa para nos referir a uma relação de proporcionalidade entre duas variáveis. Neste Capítulo uma das variáveis será o tempo. Se um carro passa voando por nós a, digamos $30 \, \text{m}$ por segundo, um simples cálculo mental nos mostrará que isso equivale – enquanto durar – a uma taxa de $1800 \, \text{m}$ por minuto, ou mais de $100 \, \text{Km}$ por hora.

Agora, em que sentido é verdadeiro que uma velocidade de $30 \, \text{m}$ por segundo é o mesmo que $1800 \, \text{m}$por minuto? Trinta metros não são a mesma coisa que 1800 metros, nem um segundo é a mesma coisa que um minuto. O que queremos dizer ao afirmar que a **taxa** é a mesma, é: **que a proporção entre a distância percorrida e o tempo gasto para percorrê-la é a mesma em ambos os casos**.

Vejamos outro exemplo: um homem pode ter apenas alguns Reais em sua posse e, mesmo assim, ser capaz de gastar dinheiro em uma taxa de milhões por ano, desde que continue gastando dinheiro a essa taxa por poucos minutos. Vamos ver como esta mágica funciona. Porque, para ser honesto, fiquei bem interessado.

Suponha que você entrega um Real no balcão para pagar por algumas balas, e suponha que a operação dura exatamente um segundo. Então, durante essa breve operação, você está queimando seu dinheiro à taxa de $1$ Real por segundo, que é a mesma taxa que $60,00$ Reais por minuto, ou $3.600,00$ Reais por hora, ou $86.400,00$ Reais por dia, ou ainda $2.592.000,00$ Reais por ano! Entretanto, se você tiver $100,00$ Reais no bolso só poderá continuar gastando dinheiro à taxa de $2.592.000,00$ Reais por ano por apenas $1$ minutos e $40$ segundos.

Juntos vamos colocar essas ideias em notação diferencial. Começamos fazendo que $y$ represente o dinheiro, e com que $t$ represente o tempo.

Se você está gastando dinheiro, e a quantidade que você gasta em um curto período de tempo $dt$ é chamada de $dy$, a **taxa** de gasto será $\frac{dy}{dt}$, ou melhor, deve ser escrita com um sinal de menos, como $-\frac{dy}{dt}$, porque $dy$ é um decréscimo, não um acréscimo.

Infelizmente, para nossos objetivos, dinheiro não é um bom exemplo para o cálculo. Isso porque geralmente vem e vai aos saltos e não por um fluxo contínuo – você pode ganhar $100.000,00$ Reais por ano, mas este salário não será depositado todos os dias, o dia todo, em um fluxo contínuo de pequenas quantidades. Seu salário é depositado, semanalmente, mensalmente, trimestralmente, em blocos. Além disso seus gastos também são feitos em pagamentos repentinos. Neste caso, *dizemos que este processo é um processo discreto, onde não existe continuidade entre dois pontos específicos no tempo*.

Uma ilustração mais adequada as necessidades do Cálculo e a ideia de taxa, que quero passar, será fornecida pelo estudo da velocidade de um corpo em movimento.

Vamos considerar uma situação próxima da realidade: de Londres (estação Euston) para Liverpool são $320\, \text{Km}$. Se um trem sai de Londres às $7$ horas, e chega a Liverpool às $11$ horas. A leitora deve ter percebido que este trem percorreu os $320\, \text{Km}$ em $4\, \text{h}$. Assim, a taxa média de velocidade deste trem foi de $80\, \text{Km/h}$. Já que: $ \frac{320}{4} = 80 $.

Neste ponto, a atenta leitora está fazendo uma comparação mental entre a distância percorrida e o tempo gasto para percorrê-la. Você está dividindo um pelo outro. Para ver se eu não errei e para formar a estrutura cognitiva necessária para entender o problema. Na notação que estamos introduzindo teremos, se $y$ é toda a distância, e $t$ todo o tempo, claramente a taxa média será $\frac{y}{t}$.

Se olharmos com cuidado o percurso deste trem, veremos que a velocidade não foi constante o tempo todo. Ao partir, durante a aceleração e durante a desaceleração no final da viagem, a velocidade tera sido menor. Provavelmente em algum ponto, talvez ao descer uma colina, a velocidade tenha sido superior a $80\, \text{Km/h}$. Finalmente, se em qualquer mínima fração de tempo $dt$, o elemento correspondente mínimo de distância percorrida foi $dy$, então naquela parte da viagem a velocidade foi $\frac{dy}{dt}$. *A taxa na qual uma grandeza, neste caso a distância, está mudando em relação a outra grandeza, neste caso o tempo, será adequadamente expressa se declararmos o coeficiente diferencial de uma em relação à outra*.

A velocidade, corretamente expressa de acordo com a física e a matemática, será a taxa na qual uma distância muito pequena em qualquer direção está sendo percorrida. Portanto a velocidade pode ser escrita como:

$$v = \frac{dy}{dt}$$.

Se a velocidade $v$ não for constante ou ela está aumentando ou diminuindo. *A taxa na qual uma velocidade está variando é chamada de aceleração quando positiva e desaceleração quando negativa*. Se um corpo em movimento está, em qualquer instante, ganhando mais velocidade, $dv$, em um elemento de tempo, $dt$, então a aceleração $a$ em cada instante infinitesimal pode ser escrita por:


$$

a = \frac{dv}{dt}.

$$

Como já vimos, $dv$ é corretamente representada por $d\left(\frac{dy}{dt}\right)$. Assim, podemos definir a aceleração como:


$$

a = \frac{d\left(\frac{dy}{dt}\right)}{dt}.

$$

A leitora deve observar que a aceleração é a derivada da derivada da distância pelo tempo. Que podemos representar por $a = \frac{d^2 y}{dt^2}$. Que podemos ler como *a aceleração é o segundo coeficiente diferencial do deslocamento, em relação ao tempo ou, a segunda derivada do deslocamento em relação ao tempo*. Onde o que estamos chamando de deslocamento é a distância percorrida. Assim sendo, a aceleração é a taxa de variação da velocidade em relação ao tempo e será medida, no sistema internacional de unidades por metros por segundo por segundo ou $\text{m/s}$.

Quando um trem começa a se mover, sua velocidade $v$ é pequena, mas está aumentando rapidamente. O trem está sendo acelerado pelo esforço do motor. Assim, seu coeficiente $\frac{d^2 y}{dt^2}$ será grande. Quando atinge sua velocidade máxima e não está mais sendo acelerado, não há variação de velocidade. logo $\frac{d^2 y}{dt^2}$ cai para zero. Neste ponto dizemos que a velocidade tende a ser constante. Finalmente, quando o trem se aproxima do seu local de parada, sua velocidade começa a diminuir. Pode ocorrer que a velocidade diminua muito rapidamente se, por exemplo, os freios forem acionados. Durante esse período de desaceleração, ou redução de velocidade, o valor de $\frac{dv}{dt}$, ou de $\frac{d^2 y}{dt^2}$ será negativo.

Para acelerar uma massa $m$ é necessária a aplicação contínua de força. A força necessária para acelerar uma massa é proporcional à essa massa, e à aceleração que está sendo imposta. Portanto, podemos escrever a força $f$ como:


$$

f = ma,

$$

ou:


$$

f = m \frac{dv}{dt};

$$

ou ainda:


$$

f = m \frac{d^2 y}{dt^2}.

$$

*O produto da massa pela velocidade em que está se movendo é chamado de momento*, e é simbolizado por $mv$. Se diferenciarmos o momento em relação ao tempo, obteremos $\frac{d(mv)}{dt}$, a taxa de variação do momento. Mas, como $m$ é uma quantidade constante, podemos escrever $m \frac{dv}{dt}$, que sabemos, pelo que vimos antes, é o mesmo que a força, $f$. Ou seja, *a força pode ser expressa tanto como massa vezes aceleração, quanto como taxa de variação do momento*.

O produto da massa pela velocidade em que está se movendo é chamado de *momento*, e é simbolizado por $mv$.

Se diferenciarmos o momento em relação ao tempo, obteremos $\frac{d(mv)}{dt}$, que é a taxa de variação do momento. Como $m$ é uma quantidade constante, podemos escrever $m \frac{dv}{dt}$. Pelo que vimos antes, sabemos que isso é o mesmo que a força, $f$. Portanto, *a força pode ser expressa tanto como massa vezes aceleração, quanto como taxa de variação do momento*.

Agora, vamos falar sobre trabalho. Se uma força é empregada para mover algo contra uma força contrária igual e oposta, esta força irá realizar *trabalho*. *A quantidade de trabalho realizado é medida pelo produto da força pela distância (em sua própria direção) através da qual seu ponto de aplicação se move*. Assim, se uma força $f$ se move para frente através de um comprimento $y$, o trabalho realizado, que podemos chamar de $w$, será:


$$

w = f \cdot y,

$$

onde consideramos $f$ como uma força constante. Se a força varia em partes infinitesimais de uma distância $y$, teremos que encontrar uma expressão para o valor dessa força. Se $f$ for a força ao longo do pequeno elemento de comprimento $dy$, a quantidade de trabalho realizado será $f \cdot dy$. Mas como $dy$ é apenas um elemento infinitesimal de comprimento, um elemento infinitesimal de trabalho será realizado. Como usamos $w$ para trabalho, então um elemento infinitesimal de trabalho será representada $dw$; e teremos:


$$

dw = f \cdot dy;

$$

que, se lembrarmos de $f=ma$ pode ser escrito:


$$

dw = ma \cdot dy;

$$

ou, se lembramos que a velocidade é $\frac{d^2 y}{dt^2}$, podemos escrever:


$$

dw = m \frac{dv}{dt} \cdot dy;

$$

ou ainda, podemos lembrar que a velocidade é a derivada segunda da distância em relação ao tempo e teremos:


$$

dw = m \frac{d^2 y}{dt^2} \cdot dy;

$$

Agora podemos fazer um pouco de mágica algébrica. Vamos começar, dividindo ambos os lados da equação por $dy$:


$$

\frac{dw}{dy} = m \frac{d^2 y}{dt^2}

$$

A equação resultante é:


$$

\frac{dw}{dy} = m \frac{d^2 y}{dt^2}

$$

Em muitos contextos de física, a expressão $m \frac{d^2 y}{dt^2}$ é conhecida como a força $f$. Portanto, podemos substituir essa expressão por $f$:


$$

\frac{dw}{dy} = f

$$

Isso nos leva a uma terceira definição de *força*. Quando uma força está sendo usada para produzir um deslocamento em uma direção específica, podemos defini-la assim:

*A força (nessa direção) é igual à quantidade de trabalho realizado por unidade de deslocamento naquela direção*.

Em outras palavras, a força é a taxa entre a variação trabalho realizado e a variação da distância percorrida na direção da força. Nesta definição, a palavra *taxa* é usada no sentido de razão ou proporção.

[Sir Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton), que descobriu os métodos do cálculo quase simultaneamente a Leibniz, tinha uma abordagem particular. Newton considerava todas as quantidades variáveis como *fluxos*. O que hoje chamamos de coeficiente diferencial, ele via como a taxa de fluxo, ou *fluxion* da quantidade em questão.

Newton não usava a notação $dy$, $dx$, e $dt$ (essa notação é de Leibniz). Para Newton, se $y$ fosse uma quantidade variável ou "fluente", sua taxa de variação (ou "fluxion") era simbolizada por $\dot{y}$. Se $x$ fosse a variável, sua *fluxion* era $\dot{x}$. O ponto acima da letra indicava que ela havia sido diferenciada.

A notação de Newton, no entanto, tinha limitações. Ela não especificava a variável independente em relação à qual a diferenciação foi feita, o que limitou sua adoção. Por outro lado, a notação de Leibniz oferecia mais clareza. Por exemplo, $\frac{dy}{dt}$ mostra claramente que $y$ é diferenciado em relação a $t$, e $\frac{dy}{dx}$ indica que $y$ é diferenciado em relação a $x$.

Em comparação, $\dot{y}$ na notação de Newton poderia significar $\frac{dy}{dx}$, $\frac{dy}{dt}$, $\frac{dy}{dz}$, ou qualquer outra variável, dependendo do contexto. A notação de Leibniz, sendo mais informativa, acabou prevalecendo. Mais que isso, *na opinião deste pobre escriba, Newton entendeu o cálculo, usou o cálculo, mas foi Leibnitz que realmente criou o cálculo*.

No entanto, a notação de Newton ainda pode ser útil hoje em dia. Sua simplicidade pode ser vantajosa se usada exclusivamente para casos onde o *tempo* é a variável independente, os problemas que Newton estudou. Nesse caso específico, $\dot{y}$ significa $\frac{dy}{dt}$, $\dot{u}$ significa $\frac{du}{dt}$, e $\ddot{x}$ significa $\frac{d^2 x}{dt^2}$.

Adotando esta notação "fluxional", podemos escrever as equações mecânicas que estudamos nos parágrafos acima, da seguinte forma:

| distância  | $x$   |
|--------------|-------------|
| velocidade   | $v = \dot{x}$ |
| aceleração   | $a = \dot{v} = \ddot{x}$ |
| força  | $f = m\dot{v} = m\ddot{x}$ |
| trabalho   | $w = x \cdot m\ddot{x}$ |

> O Autor original, [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson) era inglês e o texto original foi publicado no Reino Unido. Que defende até hoje a primazia de Newton na descoberta do Cálculo. Principalmente porque, no século XVII, o presidente da "The Royal Society of London for Improving Natural Knowledge", também conhecida como Royal Society, uma das instituições científicas, ainda hoje uma das instituições científicas mais prestigiadas do mundo, julgou o trabalho de Leibnitz como plágio. Quase esquecia, na época o presidente da Royal Society era o próprio Newton. Hoje já estamos suficientemente distantes no tempo para sermos mais justos com Leibnitz.

Só nos resta estudar alguns exemplos para entender melhor estes conceitos.

### Exemplos

1. Um corpo se move de tal forma que a distância $x$ (em metros), que ele percorre a partir de um certo ponto $O$, é dada pela relação $x = 0.2t^2 + 10.4$, onde $t$ é o tempo em segundos decorridos desde um certo instante. Encontre a velocidade e a aceleração $5$ segundos após o corpo começar a se mover, e também encontre os valores correspondentes quando a distância percorrida é de $100$ metros. Encontre também a velocidade média durante os primeiros $10$ segundos de seu movimento. (Suponha que as distâncias e o movimento para a direita sejam positivos.).

  Resolvendo teremos:


$$

x = 0.2t^2 + 10.4;

$$


$$

v = \dot{x} = \frac{dx}{dt} = 0.4t;

$$

  e


$$

a = \ddot{x} = \frac{d^2 x}{dt^2} = 0.4 \quad \text{constante}.

$$

  Quando $t = 0$, $x = 10.4$ e $v = 0$. O corpo partiu inicialmente de um ponto 10.4 metros à direita do ponto $O$; e o tempo foi contado a partir do instante em que o corpo começou a se mover.

  Quando $t = 5$, $v = 0.4 \cdot 5 = 2 \text{m/s}$; $a = 0.4 \text{m/s}^2$.

  Quando $x = 100$, $100 = 0.2t^2 + 10.4$, ou $t^2 = 448$, e $t = 21.17$ segundos; $v = 0.4 \cdot 21.17 = 8.468 \text{m/s}$.

  Quando $t = 10$,

  distância percorrida $= 0.2 \cdot 10^2 + 10.4 - 10.4 = 20 \text{m}$.

  velocidade média $= \frac{20}{10} = 2 \text{m/s}$.

  É a mesma velocidade que a velocidade no meio do intervalo, $t = 5$; pois, com a aceleração sendo constante, a velocidade variou uniformemente de zero quando $t = 0$ para $4 \, \text{m/s}$ quando $t = 10$.

2. No problema acima, vamos supor que tivéssemos:


$$

x = 0.2t^2 + 3t + 10.4.

$$

  Neste caso teríamos:


$$

v = \dot{x} = \frac{dx}{dt} = 0.4t + 3;

$$


$$

a = \ddot{x} = \frac{d^2 x}{dt^2} = 0.4 \quad \text{constante}.

$$

  Quando $t = 0$, $x = 10.4$ e $v = 3 \text{ m/s}$, o tempo é contado a partir do instante em que o corpo passou por um ponto $10.4$ metros do ponto $O$, sua velocidade sendo então já $3 \text{ m/s}$. Para encontrar o tempo decorrido desde que o corpo começou a se mover, façamos $v = 0$; então $0.4t + 3 = 0$, ou $t = -\frac{3}{0.4} = -7.5$ segundos. O corpo começou a se mover $7.5$ segundos antes do início da observação. $5$ segundos depois disso teremos $t = -2.5$ e $v = 0.4 \cdot -2.5 + 3 = 2 \text{m/s}$.

  Quando $x = 100 \text{m}$,


$$

100 = 0.2t^2 + 3t + 10.4; \quad \text{ou} \quad t^2 + 15t - 448 = 0;

$$

  portanto


$$

t = 14.95 \text{ s}, \quad v = 0.4 \cdot 14.95 + 3 = 8.98 \text{ m/s}.

$$

  Para encontrar a distância percorrida durante os primeiros 10 segundos do movimento, é necessário saber quão longe o corpo estava do ponto $O$ quando começou.

  Quando $t = -7.5$,


$$

x = 0.2 \cdot (-7.5)^2 - 3 \cdot 7.5 + 10.4 = -0.85 \text{m},

$$

  isso é 0.85 m à esquerda do ponto $O$.

  Agora, quando $t = 2.5$,


$$

x = 0.2 \cdot 2.5^2 + 3 \cdot 2.5 + 10.4 = 19.15 \text{ m}.

$$

  Então, em $10$ segundos, a distância percorrida foi $19.15 + 0.85 = 20 \text{ m}$, e a velocidade média:


$$

\text{velocidade média} = \frac{20}{10} = 2 \text{ m/s}.

$$

3. Considere um problema similar quando a distância é dada por $x = 0.2t^2 - 3t + 10.4$. Então $v = 0.4t - 3$, $a = 0.4 \quad \text{constante}$. Quando $t = 0$, $x = 10.4$ como antes, e $v = -3$; de modo que o corpo estava se movendo na direção oposta ao seu movimento nos casos anteriores. Como a aceleração é positiva, no entanto, vemos que essa velocidade diminuirá à medida que o tempo passa, até que se torne zero, quando $v = 0$ ou $0.4t - 3 = 0$; ou $t = 7.5$ s. Após isso, a velocidade torna-se positiva; e 5 segundos após o corpo começar, $t = 12.5$, e


$$

v = 0.4 \cdot 12.5 - 3 = 2 \text{ m/s}.

$$

  Resolvendo:

  Quando $x = 100 \text{ m}$,


$$

100 = 0.2t^2 - 3t + 10.4; \quad \text{ou} \quad t^2 - 15t - 448 = 0,

$$

  e $t = 29.95$; $v = 0.4 \cdot 29.95 - 3 = 8.98 \text{ m/s}$.

  Quando $v$ é zero,


$$

x = 0.2 \cdot 7.5^2 - 3 \cdot 7.5 + 10.4 = -0.85 \text{ m},

$$

  informando-nos que o corpo se move de volta para 0.85 m além do ponto $O$ antes de parar. Dez segundos depois


$$

t = 17.5 \quad \text{e} \quad x = 0.2 \cdot 17.5^2 - 3 \cdot 17.5 + 10.4 = 19.15.

$$

  A distância percorrida $= 0.85 + 19.15 = 20.0 \text{ m}$, e a velocidade média é novamente $2 \text{ m/s}$.

4. Considere um problema similar quando a distância é dada por $x = 0.2t^3 - 3t^2 + 10.4$. Então $v = 0.6t^2 - 6t$ e $a = 1.2t - 6$. A aceleração não é mais constante.

  Vamos começar analisando as condições iniciais:

  Quando $t = 0$, teremos:

  - $x = 0.2(0)^3 - 3(0)^2 + 10.4 = 10.4$ metros
- $v = 0.6(0)^2 - 6(0) = 0$ m/s
- $a = 1.2(0) - 6 = -6$ m/s²

  - Posição inicial: o corpo começa a $10,4$ metros à direita do ponto $O$.
- Velocidade inicial: o corpo está inicialmente em repouso (velocidade zero).
- Aceleração inicial: a aceleração inicial é negativa ($-6 \, \text{m/s²}$), o que significa que o corpo começará a se mover para a esquerda, em direção ao ponto $O$.

  A aceleração não é constante, mas varia linearmente com o tempo. Isso significa que a velocidade não mudará a uma taxa constante.

  - Velocidade: a velocidade será negativa enquanto a aceleração for negativa. A velocidade se tornará zero em algum momento e depois se tornará positiva.
- Posição: a posição do corpo diminuirá inicialmente (movimento para a esquerda), atingirá um valor mínimo e depois começará a aumentar (movimento para a direita).

  O corpo começa em repouso a $10,4$ metros à direita do ponto $O$. Devido à aceleração negativa inicial, ele começa a se mover para a esquerda. A velocidade aumenta em módulo (mantendo-se negativa) até que a aceleração se torne zero. Após esse ponto, a aceleração se torna positiva, a velocidade diminui em módulo (ainda negativa) até se tornar zero, e então o corpo inverte o sentido do movimento e passa a se mover para a direita com velocidade crescente.

5. Se temos $x = 0.2t^3 - 3t + 10.4$, então $v = 0.6t^2 - 3$, e $a = 1.2t$.

  Quando $t = 0$, $x = 10.4$; $v = -3$; $a = 0$.

  O corpo está se movendo em direção ao ponto $O$ com uma velocidade de $3 \, \text{m/s}$, e nesse instante a velocidade é uniforme.

  Vemos que as condições do movimento podem sempre ser determinadas a partir da equação tempo-distância e suas primeiras e segundas derivadas. Nos dois últimos exemplos a velocidade média durante os primeiros $10$ segundos e durante os $5$ segundos após o início não será mais a mesma, pois a velocidade não está aumentando uniformemente, já que a aceleração não é mais constante.

  ![]({{ site.baseurl }}/assets/images/calc_Fig3.jpg)
*Figura 5.1 - Gráfico das grandezas do Exemplo 5.*{: class="legend"}

6. O ângulo $\theta$ (em radianos) percorrido por uma roda é dado por $\theta = 3 + 2t - 0.1t^3$, onde $t$ é o tempo em segundos a partir de um determinado instante; encontre a velocidade angular $\omega$ e a aceleração angular $\alpha$: (a) após 1 segundo; (b) depois que a roda tiver realizado uma revolução. Em que momento a roda está em repouso e quantas revoluções ela realizou até esse instante?

  Escrevendo para a aceleração usando a notação de Newton, teremos:


$$

\omega = \dot{\theta} = \frac{d\theta}{dt} = 2 - 0.3t^2, \quad \alpha = \ddot{\theta} = \frac{d^2\theta}{dt^2} = -0.6t.

$$

  Quando $t = 0$, $\theta = 3$; $\omega = 2 \text{ rad/s}$; $\alpha = 0$.

  Quando $t = 1$,


$$

\omega = 2 - 0.3 = 1.7 \text{ rad/s}; \quad \alpha = -0.6 \text{ rad/s}^2.

$$

  Isso é uma desaceleração; a roda está diminuindo a velocidade.

  Depois de 1 revolução,


$$

\theta = 2\pi = 6.28; \quad 6.28 = 3 + 2t - 0.1t^3.

$$

  Traçando o gráfico, $\theta = 3 + 2t - 0.1t^3$ Figura 10.4, podemos obter o valor ou os valores de $t$ para os quais $\theta = 6.28$; estes são 2.11 e 3.03 (há um terceiro valor negativo).

  Quando $t = 2.11$,


$$

\theta = 6.28; \quad \omega = 2 - 1.34 = 0.66 \text{ rad/s}; \quad \alpha = -1.27 \text{ rad/s}^2.

$$

  Quando $t = 3.03$,


$$

\theta = 6.28; \quad \omega = 2 - 2.754 = -0.754 \text{ rad/s}; \quad \alpha = -1.82 \text{ rad/s}^2.

$$

  A velocidade é invertida. A roda está evidentemente em repouso entre esses dois instantes; está em repouso quando $\omega = 0$, isto é, quando $0 = 2 - 0.3t^3$, ou quando $t = 2.58$ segundos, ela realizou


$$

\frac{\theta}{2\pi} = \frac{3 + 2 \cdot 2.58 - 0.1 \cdot 2.58^3}{6.28} = 1.025 \text{ revoluções}.

$$

  ![]({{ site.baseurl }}/assets/images/angular_motion_graphs.jpg)
*Figura 5.2 - Gráfico das grandezas do Exemplo 6.*{: class="legend"}

### Exercícios V

1. Se $y = a + bt^2 + ct^4$, encontre $\frac{dy}{dt}$ e $\frac{d^2 y}{dt^2}$.

   Resposta:
$$

\frac{dy}{dt} = 2bt + 4ct^3; \quad \frac{d^2 y}{dt^2} = 2b + 12ct^2

$$
.

2. Um corpo caindo livremente no espaço descreve em $t$ segundos um espaço $s$, em metros, expresso pela equação $s = 16t^2$. Desenhe uma curva mostrando a relação entre $s$ e $t$. Também determine a velocidade do corpo nos seguintes tempos a partir do seu ponto de partida: $t = 2$ segundos; $t = 4.6$ segundos; $t = 0,01$ segundo.

  Resposta:

  $64$; $147,2$; e $0.32$ metros por segundo.

3. Se $x = at - \frac{1}{2}gt^2$, encontre $\dot{x}$ e $\ddot{x}$.

  Resposta:

  $x = a - gt; \quad \ddot{x} = -g$.

4. Se um corpo se move de acordo com a lei


$$

s = 12 - 4.5t + 6.2t^2

$$
,

   encontre sua velocidade quando $t = 4$ segundos; $s$ sendo em metros.

  Resposta:

  $45,1$ metros por segundo.

5. Encontre a aceleração do corpo mencionado no exemplo anterior. A aceleração é a mesma para todos os valores de $t$?

  Resposta:

  $12,4$ metros por segundo por segundo. Sim.

6. O ângulo $\theta$ (em radianos) percorrido por uma roda giratória está relacionado com o tempo $t$ (em segundos) decorrido desde o início; pela lei


$$

\theta = 2,1 - 3,2t + 4,8t^2

$$
.

  Encontre a velocidade angular (em radianos por segundo) dessa roda quando $1 \frac{1}{2}$ segundos se passaram. Encontre também sua aceleração angular.

  Resposta:

  Velocidade angular $= 11,2$ radianos por segundo; aceleração angular $= 9.6$ radianos por segundo ao quadrado.

7. Um deslizante se move de tal forma que, durante a primeira parte de seu movimento, sua distância $s$ em metros do ponto de partida é dada pela expressão


$$

s = 6.8t^3 - 10.8t; \quad t \text{ em segundos}

$$
.

   Encontre a expressão para a velocidade e a aceleração a qualquer momento; e, portanto, encontre a velocidade e a aceleração após 3 segundos.

  Resposta:

  $v = 20,4t^2 - 10,8$. $a = 40,8t$. $172,8$ cm/s, $122,4$ cm/s².

8. O movimento de um balão ascendente é tal que sua altura $h$, em quilômetros, é dada a qualquer instante pela expressão $h = 0,5 + \frac{1}{10} \sqrt[3]{t - 125}$; $t$ sendo em segundos.

   Encontre uma expressão para a velocidade e a aceleração a qualquer momento. Desenhe curvas para mostrar a variação da altura, velocidade e aceleração durante os primeiros dez minutos da ascensão.

   Resposta:


$$

v = \frac{1}{30 \sqrt[3]{(t - 125)^2}}, \quad a = -\frac{1}{45 \sqrt[5]{(t - 125)^5}}

$$

9. Uma pedra é lançada para baixo na água e sua profundidade $p$ em metros em qualquer instante $t$ segundos após atingir a superfície da água é dada pela expressão


$$

p = \frac{4}{4 + t^2} + 0,8t - 1

$$
.

   Encontre uma expressão para a velocidade e a aceleração a qualquer momento. Encontre a velocidade e a aceleração após 10 segundos.

   Resposta:


$$

v = 0,8 - \frac{8t}{(4 + t^2)^2}, \quad a = \frac{24t^2 - 32}{(4 + t^2)^3}, \quad 0,7926 \text{ m/s e } 0,00211 \text{ m/s}^2

$$

10. Um corpo se move de tal forma que os espaços descritos no tempo $t$ a partir da partida são dados por $s = t^n$, onde $n$ é uma constante. Encontre o valor de $n$ quando a velocidade é dobrada do quinto ao décimo segundo; encontre também quando a velocidade é numericamente igual à aceleração ao final do décimo segundo.

  Resposta:

  $n = 2, \quad n = 11$.

## IX. UMA NOVA ESTRATÉGIA: A Regra da Cadeia

Não é raro ficar perplexo quando encontramos uma expressão para diferenciar que é muito complicada para ser resolvida diretamente. No ponto em que estamos, muito complicadas serão todas as expressões que não se encaixem nas regras que já vimos. Por exemplo, a equação:

$$y = (x^2 + a^2)^{3/2}$$,

não se encaixa em nenhuma das regras simples que vimos até agora, eu não sei tirar a raiz quadrada do cubo de $(x^2 + a^2)$ e não tenho paciência para isso. Ao nosso socorro vem a **Regra da Cadeia** uma solução para lá de elegante para resolver diferenciais de funções complicadas.

Para entender a regra da cadeia, vamos começar introduzindo uma variável nova. Como estou com a criatividade em alta, usaremos $u$, para representar a expressão interna:

$$y = u^{3/2}$$,

uma equação que a amável leitora conseguirá derivar facilmente, usando apenas a regra da potência. Neste caso, terá feito algo parecido com:

$$\frac{dy}{du} = \frac{3}{2} u^{1/2}$$.

Feito isso, eu sei o resultado da diferencial de $u$ em relação a $u$. Porém, $u$ substituiu a expressão original, $(x^2 + a^2)$. Para terminar precisamos derivar $u$ em relação a $x$:

$$u = x^2 + a^2$$,

usando as Regras da Soma e a da Potência esta derivação resultará em:

$$\frac{du}{dx} = 2x$$.

Já que $a^2$ é constante e derivada de constante é zero. *Finalmente, aplicamos a regra da cadeia*:


$$

\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}

$$
,

Isto é determinar que a derivada $\frac{dy}{dx}$ de $(x^2 + a^2)$ é a derivada de $y$ em relação a $u$, multiplicada pela derivada de $u$ em relação a $x$. O que resultará em:


$$

\frac{dy}{dx} = \frac{3}{2} u^{1/2} \cdot 2x
= \frac{3}{2} (x^2 + a^2)^{1/2} \cdot 2x
= 3x (x^2 + a^2)^{1/2}

$$
.

Assim, encontramos a derivada da função original! A regra da cadeia nos permite "quebrar" funções compostas em partes mais simples, usando uma variável intermediária, e derivá-las passo a passo. E isso é tudo, a mágica está feita.

Quando encontrarmos uma função complicada, que não parece fácil derivar usando as regras mais simples, podemos adotar uma abordagem alternativa. Primeiramente, substituímos esta função por uma nova variável, como $u$, $v$, ou qualquer outra de nossa preferência. Em seguida, derivamos em relação a esta nova variável. Após obter este resultado inicial, substituímos a função original no resultado da primeira derivação. O próximo passo será derivar em relação à variável inicial, seja ela $x$ ou qualquer outra. Por fim, aplicamos a Regra da Cadeia para obter a derivada final. Este método, conhecido como substituição ou mudança de variável, nos permite lidar com funções complexas de maneira mais sistemática e compreensível.

Tendo entendido o processo, podemos generalizar a Regra da Cadeia.

Essas funções que eu chamei de complicadas, na sua maioria, são funções compostas. Se tivermos uma função composta $y = f(g(x))$, no nosso exemplo $$y = (x^2 + a^2)^{3/2}$$, podemos usar a Regra da Cadeia para encontrar sua derivada:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$,

esta é a forma simplificada da regra da cadeia, muito usada para memorização.
Em outras palavras, derivamos a função "externa" $f$ em relação ao seu argumento $g(x)$, e multiplicamos pela derivada da função "interna" $g$ em relação a $x$. A leitora deve me perdoar, acabei usando um pouco de formalidade matemática. Eu disse que ia ser fácil. Sendo assim, vamos tentar melhorar isso.

Se tivermos uma função composta $y = f(g(x))$. Que no nosso caso, $y = (x^2 + a^2)^{3/2}$, $g(x) = x^2 + a^2$ (função interna) e $f(x) = x^{3/2}$ (função externa), podemos usar a regra da cadeia para encontrar sua derivada:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

Em outras palavras, *a Regra da Cadeia implica que devemos derivar a função "externa" $f$ em relação ao seu argumento $g(x)$, e multiplicamos pela derivada da função "interna" $g$ em relação a $x$*.

Oh,Deus! Não, não é complicado nem difícil. Aos poucos, quando você estiver lidando com senos, cossenos, e funções exponenciais, irá se apaixonar pela regra da cadeia.

Vamos praticar a Regra da Cadeia com alguns exemplos, resolvidos passo a passo.

1. Diferencie $y = \sqrt{a + x}$.

  Vamos fazer $a + x = u$. Dessa forma, podemos encontrar a derivada de $u$ em relação a $x$:


$$

\frac{du}{dx} = 1

$$
.

  Agora, substituímos $u$ na expressão de $y$:


$$

y = u^{1/2}

$$
.

  Aqui estamos trabalhando com uma função composta $y = f(g(x))$, onde $g(x) = u = a + x$ (função interna) e $f(u) = u^{1/2}$ (função externa). Em seguida, encontramos a derivada de $y$ em relação a $u$:


$$

\frac{dy}{du} = \frac{1}{2} u^{-1/2} = \frac{1}{2} (a + x)^{-1/2}

$$
.

  Aplicamos a Regra da Cadeia para encontrar a derivada de $y$ em relação a $x$:


$$

\frac{dy}{dx} = f'(g(x)) \cdot g'(x) = \frac{dy}{du} \cdot \frac{du}{dx} = \frac{1}{2} (a + x)^{-1/2} \cdot 1 = \frac{1}{2} (a + x)^{-1/2}

$$
.

  Portanto, a derivada de $y = \sqrt{a + x}$ em relação a $x$ é


$$

\frac{dy}{dx} = \frac{1}{2} (a + x)^{-1/2}

$$
.

2. Diferencie $y = \frac{1}{\sqrt{a + x^2}}$.

  Vamos fazer $a + x^2 = u$. Dessa forma, podemos encontrar a derivada de $u$ em relação a $x$:


$$

\frac{du}{dx} = 2x

$$
.

  Agora, substituímos $u$ na expressão de $y$:


$$

y = u^{-1/2}

$$
.

  Novamente, temos uma função composta $y = f(g(x))$, onde $g(x) = a + x^2$ (função interna) e $f(u) = u^{-1/2}$ (função externa). Em seguida, encontramos a derivada de $y$ em relação a $u$:


$$

\frac{dy}{du} = -\frac{1}{2} u^{-3/2}

$$
.

  Aplicamos a regra da cadeia para encontrar a derivada de $y$ em relação a $x$:


$$

\frac{dy}{dx} = f'(g(x)) \cdot g'(x) = \frac{dy}{du} \cdot \frac{du}{dx} = -\frac{1}{2} u^{-3/2} \cdot 2x = -\frac{x}{(a + x^2)^{3/2}}

$$
.

  Portanto, a derivada de $y = \frac{1}{\sqrt{a + x^2}}$ em relação a $x$ é


$$

\frac{dy}{dx} = -\frac{x}{(a + x^2)^{3/2}}

$$
.

3. Diferencie $y = \left(m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}}\right)^a$.

  Nessa altura do campeonato, a atenta leitora já percebeu que todos os exemplos serão de funções compostas e que a parte que requer mais atenção é a escolha da função interna e da função externa. Neste exemplo temos uma função composta $y = f(g(x))$, onde $g(x) = m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}}$ (função interna) e $f(u) = u^a$ (função externa).

  Vamos fazer $m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}} = u$. Dessa forma, podemos encontrar a derivada de $u$ em relação a $x$:


$$

\frac{du}{dx} = -\frac{2}{3}nx^{-1/3} - \frac{4}{3}px^{-7/3}

$$
.

  Agora, substituímos $u$ na expressão de $y$:


$$

y = u^a

$$

  Em seguida, encontramos a derivada de $y$ em relação a $u$:


$$

\frac{dy}{du} = au^{a-1}

$$
.

  Aplicamos a regra da cadeia para encontrar a derivada de $y$ em relação a $x$:


$$

\frac{dy}{dx} = f'(g(x)) \cdot g'(x) = \frac{dy}{du} \cdot \frac{du}{dx} = a \left( m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}} \right)^{a-1} \left( -\frac{2}{3}nx^{-1/3} - \frac{4}{3}px^{-7/3} \right)

$$
.

  Portanto, a derivada de $y = \left(m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}}\right)^a$ em relação a $x$ é


$$

\frac{dy}{dx} = -a \left( m - nx^{\frac{2}{3}} + \frac{p}{x^{\frac{4}{3}}} \right)^{a-1} \left( \frac{2}{3}nx^{-1/3} + \frac{4}{3}px^{-7/3} \right)

$$
.

  A leitora também já deve ter percebido que a estratégia é sempre a mesma, quase como um algoritmo.

4. Diferencie $y = \frac{1}{\sqrt{x^3 - a^2}}$.

  Fazendo $u = x^3 - a^2$.


$$

\frac{du}{dx} = 3x^2; \quad y = u^{-1/2}; \quad \frac{dy}{du} = -\frac{1}{2} u^{-3/2}.

$$


$$

\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = -\frac{3x^2}{2\sqrt{(x^3 - a^2)^3}}.

$$

5. Diferencie $y = \sqrt{\frac{1 - x}{1 + x}}$.

  Este exemplo é bem mais trabalhoso. Vamos reescrever como $y = \left(\frac{1 - x}{1 + x}\right)^{1/2}$.

  Para aplicar a regra da cadeia, vamos definir as funções internas e externas.

  Primeiro, defina $u = \frac{1 - x}{1 + x}$. Então, a função externa é $y = u^{1/2}$.

  Para encontrar a derivada de $y$ em relação a $x$, precisamos aplicar a regra da cadeia:


$$

\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.

$$

  Primeiro, encontramos $\frac{dy}{du}$:


$$

\frac{dy}{du} = \frac{1}{2} u^{-1/2} = \frac{1}{2} \left( \frac{1 - x}{1 + x} \right)^{-1/2}.

$$

  Agora, encontramos $\frac{du}{dx}$ usando a Regra do Quociente:


$$

u = \frac{1 - x}{1 + x}.

$$

  Aplicamos a Regra do Quociente:


$$

\frac{du}{dx} = \frac{(1 + x) \frac{d}{dx} (1 - x) - (1 - x) \frac{d}{dx} (1 + x)}{(1 + x)^2}.

$$

  Calculando as derivadas:


$$

\frac{d}{dx} (1 - x) = -1,

$$


$$

\frac{d}{dx} (1 + x) = 1.

$$

  Portanto,


$$

\frac{du}{dx} = \frac{(1 + x) (-1) - (1 - x) (1)}{(1 + x)^2} = \frac{-(1 + x) - (1 - x)}{(1 + x)^2} = \frac{-1 - x - 1 + x}{(1 + x)^2} = \frac{-2}{(1 + x)^2}.

$$

  Agora, combinamos as duas derivadas usando a regra da cadeia:


$$

\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = \frac{1}{2} \left( \frac{1 - x}{1 + x} \right)^{-1/2} \cdot \frac{-2}{(1 + x)^2}.

$$

  Simplificando:


$$

\frac{dy}{dx} = \frac{1}{2} \left( \frac{1 + x}{1 - x} \right)^{1/2} \cdot \frac{-2}{(1 + x)^2} = -\frac{(1 + x)^{1/2}}{(1 - x)^{1/2}} \cdot \frac{1}{(1 + x)^2} = -\frac{(1 + x)^{1/2}}{(1 - x)^{1/2} (1 + x)^2}.

$$

  Portanto, a derivada de $y = \sqrt{\frac{1 - x}{1 + x}}$ em relação a $x$ é:


$$

\frac{dy}{dx} = -\frac{1}{(1 + x) \sqrt{1 - x^2}}.

$$

  Poderíamos ter resolvido esta derivação usando apenas a Regra do Quociente. Comece por reescrever a função como $y = \frac{(1 - x)^{1/2}}{(1 + x)^{1/2}}$ e aplique a Regra do Quociente. Vamos ver:

  Para encontrar a derivada de $y$ em relação a $x$, usaremos a Regra do Quociente, que diz que a derivada de $\frac{u}{v}$ é dada por:


$$

\frac{d}{dx} \left( \frac{u}{v} \right) = \frac{v \frac{du}{dx} - u \frac{dv}{dx}}{v^2}.

$$

  Neste caso, $u = (1 - x)^{1/2}$ e $v = (1 + x)^{1/2}$.

  Primeiro, encontramos as derivadas de $u$ e $v$ em relação a $x$:


$$

\frac{du}{dx} = \frac{d}{dx} (1 - x)^{1/2} = \frac{1}{2} (1 - x)^{-1/2} \cdot (-1) = -\frac{1}{2\sqrt{1 - x}}

$$
,


$$

\frac{dv}{dx} = \frac{d}{dx} (1 + x)^{1/2} = \frac{1}{2} (1 + x)^{-1/2} \cdot (1) = \frac{1}{2\sqrt{1 + x}}

$$
.

  Agora, aplicamos a Regra do Quociente:


$$

\frac{dy}{dx} = \frac{(1 + x)^{1/2} \left(-\frac{1}{2\sqrt{1 - x}}\right) - (1 - x)^{1/2} \left(\frac{1}{2\sqrt{1 + x}}\right)}{(1 + x)}

$$
.

  Simplificando os termos no numerador:


$$

\frac{dy}{dx} = \frac{(1 + x)^{1/2} \left(-\frac{1}{2\sqrt{1 - x}}\right) - (1 - x)^{1/2} \left(\frac{1}{2\sqrt{1 + x}}\right)}{(1 + x)}

$$
.


$$

\frac{dy}{dx} = \frac{-\frac{1}{2} (1 + x)^{1/2} \cdot (1 - x)^{-1/2} - \frac{1}{2} (1 - x)^{1/2} \cdot (1 + x)^{-1/2}}{(1 + x)}

$$
.


$$

\frac{dy}{dx} = \frac{-\frac{(1 + x)}{2 \sqrt{1 - x} \sqrt{1 + x}} - \frac{(1 - x)}{2 \sqrt{1 - x} \sqrt{1 + x}}}{(1 + x)}

$$
.


$$

\frac{dy}{dx} = \frac{-\frac{(1 + x) + (1 - x)}{2 \sqrt{1 - x} \sqrt{1 + x}}}{(1 + x)}

$$
.


$$

\frac{dy}{dx} = \frac{-\frac{2}{2 \sqrt{1 - x^2}}}{(1 + x)}

$$
.

  Portanto, a derivada de $y = \sqrt{\frac{1 - x}{1 + x}}$ em relação a $x$ é


$$

\frac{dy}{dx} = -\frac{1}{(1 + x) \sqrt{1 - x^2}}

$$
.

6. Diferencie $y = \sqrt{\frac{x^3}{1 + x^2}}$.

  Podemos escrever a função como:


$$

y = x^{3/2} (1 + x^2)^{-1/2}

$$
.

  Para encontrar a derivada de $y$ em relação a $x$, aplicamos a regra do produto:


$$

\frac{dy}{dx} = \frac{3}{2} x^{1/2} (1 + x^2)^{-1/2} + x^{3/2} \cdot \frac{d[(1 + x^2)^{-1/2}]}{dx}

$$
.

  Diferenciando $(1 + x^2)^{-1/2}$, aplicamos a Regra da Cadeia:


$$

\frac{d[(1 + x^2)^{-1/2}]}{dx} = -\frac{1}{2} (1 + x^2)^{-3/2} \cdot 2x = -\frac{x}{\sqrt{(1 + x^2)^3}}

$$
.

  de modo que:


$$

\frac{dy}{dx} = \frac{3\sqrt{x}}{2\sqrt{1 + x^2}} - \frac{\sqrt{x^5}}{\sqrt{(1 + x^2)^3}} = \frac{\sqrt{x}(3 + x^2)}{2\sqrt{(1 + x^2)^3}}

$$
.

7. Diferencie  $y = (x + \sqrt{x^2 + x + a})^3$

  Para derivar a função, aplicaremos a Regra da Cadeia. Primeiramente, definimos:

  $$u = x + \sqrt{x^2 + x + a} $$.

  Assim, a função original se torna:

  $$y = u^3 $$.

  A derivada de $u$ em relação a $x$ é obtida pelas Regras da Soma e da Cadeia:

  $$\frac{du}{dx} = 1 + \frac{d}{dx} (\sqrt{x^2 + x + a}) $$.

  Para derivar o termo da raiz quadrada, aplicamos novamente a Regra da Cadeia:

  $$\frac{d}{dx} (\sqrt{x^2 + x + a}) = \frac{1}{2}(x^2 + x + a)^{-1/2} \cdot (2x + 1) = \frac{2x + 1}{2\sqrt{x^2 + x + a}} $$.

  Portanto:

  $$\frac{du}{dx} = 1 + \frac{2x + 1}{2\sqrt{x^2 + x + a}} $$.

  A derivada de $y$ em relação a $u$ é dada pela regra da potência:

  $$\frac{dy}{du} = 3u^2 = 3(x + \sqrt{x^2 + x + a})^2 $$.

  Finalmente, pela Regra da Cadeia:

  $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} $$.

  Substituindo os valores encontrados:

  $$\frac{dy}{dx} = 3(x + \sqrt{x^2 + x + a})^2 \cdot \left(1 + \frac{2x + 1}{2\sqrt{x^2 + x + a}}\right) $$.

  A derivada de $y = (x + \sqrt{x^2 + x + a})^3$ em relação a $x$ é:

  $$\frac{dy}{dx} = 3(x + \sqrt{x^2 + x + a})^2 \left(1 + \frac{2x + 1}{2\sqrt{x^2 + x + a}}\right) $$.

8. Diferencie $y = \sqrt{\frac{a^2 + x^2}{a^2 - x^2}} \sqrt[3]{\frac{a^2 - x^2}{a^2 + x^2}}$.

  Primeiro, simplificamos a expressão:


$$

y = \frac{(a^2 + x^2)^{1/2} (a^2 - x^2)^{1/3}}{(a^2 - x^2)^{1/2} (a^2 + x^2)^{1/3}} = (a^2 + x^2)^{1/6} (a^2 - x^2)^{-1/6}.

$$

  Aplicamos a regra do produto para encontrar a derivada de $y$ em relação a $x$:

  A derivada de um produto de funções é a soma das derivadas de cada função multiplicada pela outra função não derivada.


$$

\frac{dy}{dx} = (a^2 + x^2)^{1/6} \frac{d[(a^2 - x^2)^{-1/6}]}{dx} + \frac{d[(a^2 + x^2)^{1/6}]}{dx} (a^2 - x^2)^{-1/6}.

$$

  Vamos fazer $u = (a^2 - x^2)^{-1/6}$ e $v = (a^2 - x^2)$:

  Redefinimos partes da expressão para aplicar a regra da cadeia.


$$

u = v^{-1/6}; \quad \frac{du}{dv} = -\frac{1}{6} v^{-7/6}; \quad \frac{dv}{dx} = -2x.

$$

  Usamos a regra da cadeia para encontrar $\frac{du}{dx}$:


$$

\frac{du}{dx} = \frac{du}{dv} \cdot \frac{dv}{dx} = -\frac{1}{3} x (a^2 - x^2)^{-7/6}.

$$

  Vamos fazer $w = (a^2 + x^2)^{1/6}$ e $z = (a^2 + x^2)$:

  Redefinimos outra parte da expressão para aplicar a regra da cadeia novamente.


$$

w = z^{1/6}; \quad \frac{dw}{dz} = \frac{1}{6} z^{-5/6}; \quad \frac{dz}{dx} = 2x.

$$

  Usamos a regra da cadeia para encontrar $\frac{dw}{dx}$:


$$

\frac{dw}{dx} = \frac{dw}{dz} \cdot \frac{dz}{dx} = \frac{1}{3} x (a^2 + x^2)^{-5/6}.

$$

  Portanto, combinamos as derivadas usando a regra do produto para obter a derivada final de $y$ em relação a $x$:


$$

\frac{dy}{dx} = (a^2 + x^2)^{1/6} \cdot \frac{x}{3(a^2 - x^2)^{7/6}} + \frac{x}{3(a^2 - x^2)^{1/6} (a^2 + x^2)^{5/6}}.

$$

  ou


$$

\frac{dy}{dx} = \frac{x}{3} \left[ \frac{\sqrt[6]{a^2 + x^2}}{(a^2 - x^2)^{7/6}} + \frac{1}{\sqrt[6]{(a^2 - x^2)} (a^2 + x^2)^{5/6}} \right].

$$

9. Diferencie $y^n$ em relação a $y^5$.

  Aplicamos a regra da derivada de uma potência:


$$

\frac{d(y^n)}{d(y^5)} = \frac{ny^{n-1}}{5y^4} = \frac{n}{5} y^{n-5}.

$$

10. Encontre o primeiro e o segundo coeficiente diferencial de $y = \frac{x}{b} \sqrt{(a - x)x}$.

  Primeiro, aplicamos a regra do produto:


$$

\frac{dy}{dx} = \frac{x}{b} \frac{d\left[(a - x)x\right]^{1/2}}{dx} + \frac{\sqrt{(a - x)x}}{b}.

$$

  Vamos fazer $[(a - x)x]^{1/2} = u$ e $[(a - x)x] = w$; então $u = w^{1/2}$:

  Redefinimos partes da expressão para aplicar a regra da cadeia.


$$

\frac{du}{dw} = \frac{1}{2} w^{-1/2} = \frac{1}{2\sqrt{(a - x)x}}.

$$

  Encontramos $\frac{dw}{dx}$:


$$

\frac{dw}{dx} = a - 2x.

$$

  Usamos a regra da cadeia para encontrar $\frac{du}{dx}$:


$$

\frac{du}{dx} = \frac{du}{dw} \cdot \frac{dw}{dx} = \frac{a - 2x}{2\sqrt{(a - x)x}}.

$$

  Portanto,


$$

\frac{dy}{dx} = \frac{x(a - 2x)}{2b\sqrt{(a - x)x}} + \frac{\sqrt{(a - x)x}}{b} = \frac{x(3a - 4x)}{2b\sqrt{(a - x)x}}.

$$

  Agora, encontramos o segundo coeficiente diferencial $\frac{d^2 y}{dx^2}$:


$$

\frac{d^2 y}{dx^2} = \frac{2b\sqrt{(a - x)x}(3a - 8x) - (3ax - 4x^2)b(a - 2x)}{(a - x)x}.

$$

  Simplificando,


$$

\frac{d^2 y}{dx^2} = \frac{3a^2 - 12ax + 8x^2}{4b(a - x)\sqrt{(a - x)x}}.

$$

### Exercícios VI

Diferencie as seguintes funções:

(1) $y = \sqrt{x^2 + 1}$.

(2) $y = \sqrt{x^2 + a^2}$.

(3) $y = \frac{1}{\sqrt{a + x}}$.

(4) $y = \frac{a}{\sqrt{a - x^2}}$.

(5) $y = \frac{\sqrt{x^2 - a^2}}{x^2}$.

(6) $y = \frac{\sqrt[3]{x^4 + a}}{\sqrt[3]{x^3 + a}}$.

(7) $y = \frac{a^2 + x^2}{(a + x)^2}$.

(8) Diferencie $y^5$ em relação a $y^2$.

(9) Diferencie $y = \frac{\sqrt{1 - \theta^2}}{1 - \theta}$.

### Voltando a Regra da Cadeia #########################Parei aqui

A Regra da Cadeia, que utilizamos para derivar funções compostas, pode ser facilmente estendida para lidar com três ou mais elos na cadeia de composição  de funções. Em vez de apenas duas funções, podemos ter uma composição de várias funções, como:

$$y = f(g(h(x)))$$

Nesse caso, a derivada de $y$ em relação a $x# será calculada da seguinte forma:

$$dy/dx = dy/dz \cdot dz/dw \cdot dw/dx$$

Onde: $z = g(h(x))$ e $w = h(x)$

Em outras palavras, derivamos a função mais externa ($f$) em relação à sua variável ($z$), depois multiplicamos pela derivada da função intermediária ($g$) em relação à sua variável ($w$), e assim por diante, até chegarmos à derivada da função mais interna ($h$) em relação a $x$. Para entendermos, nada melhor que um exemplo.

Suponha que tenhamos a função:

$$y = sen(cos(x^2))$$

Podemos aplicar a Regra da Cadeia estendida da seguinte forma:

Primeiro precisamos identificar as funções. Neste caso, optaremos por:

- $$f(z) = sen(z)$$
- $$g(w) = cos(w)$$
- $$h(x) = x^2$$

Em seguida poderemos calcular as derivadas:

- $$dy/dz = cos(z)$$
- $$dz/dw = -sen(w)$$
- $$dw/dx = 2x$$

Finalmente podemos aplicar a Regra da Cadeia estendida:

   $$dy/dx = cos(z) \cdot (-sen(w)) \cdot 2x$$

   $$dy/dx = cos(cos(x^2)) \cdot (-sen(x^2)) \cdot 2x$$

A astuta leitora deve observar que a regra da cadeia pode ser estendida para quantas funções forem necessárias, sempre multiplicando as derivadas de cada função em relação à sua respectiva variável. Novamente, antes que esforçada leitora feche o livro e saia correndo, vamos ver alguns exemplos:

1. Dado:


$$

z = 3x^4; \quad v = \frac{7}{z^2}; \quad y = \sqrt{1 + v}

$$

  Encontre $\frac{dv}{dx}$:

  Solução:

  Primeiro, encontramos $\frac{dv}{dz}$ e $\frac{dz}{dx}$:


$$

\frac{dz}{dx} = \frac{d}{dx}(3x^4) = 12x^3

$$


$$

\frac{dv}{dz} = \frac{d}{dz}\left(\frac{7}{z^2}\right) = -\frac{14}{z^3}

$$

  Agora, utilizando a Regra da Cadeia:


$$

\frac{dv}{dx} = \frac{dv}{dz} \cdot \frac{dz}{dx} = -\frac{14}{z^3} \cdot 12x^3 = -\frac{168x^3}{z^3}

$$

  Substituindo $z = 3x^4$:


$$

\frac{dv}{dx} = -\frac{168x^3}{(3x^4)^3} = -\frac{168x^3}{27x^{12}} = -\frac{168}{27x^9} = -\frac{56}{9x^9}

$$

2. Dado:


$$

t = \frac{1}{5\sqrt{\theta}}; \quad x = t^3 + \frac{t}{2}; \quad v = \frac{7x^2}{\sqrt{x - 1}}

$$

  Encontre $\frac{dv}{d\theta}$:

  Solução:

  Primeiro, encontramos $\frac{dx}{dt}$ e $\frac{dt}{d\theta}$:


$$

\frac{dx}{dt} = \frac{d}{dt}\left(t^3 + \frac{t}{2}\right) = 3t^2 + \frac{1}{2}

$$


$$

\frac{dt}{d\theta} = \frac{d}{d\theta}\left(\frac{1}{5\sqrt{\theta}}\right) = -\frac{1}{10\theta^{3/2}}

$$

  Agora, encontramos $\frac{dv}{dx}$:


$$

\frac{dv}{dx} = \frac{d}{dx}\left(\frac{7x^2}{\sqrt{x - 1}}\right) = \frac{7x(5x - 6)}{3\sqrt{(x - 1)^4}}

$$

  Utilizando a Regra da Cadeia:


$$

\frac{dv}{d\theta} = \frac{dv}{dx} \cdot \frac{dx}{dt} \cdot \frac{dt}{d\theta} = \frac{7x(5x - 6)}{3\sqrt{(x - 1)^4}} \cdot \left(3t^2 + \frac{1}{2}\right) \cdot -\frac{1}{10\theta^{3/2}}

$$

  Simplificando:


$$

\frac{dv}{d\theta} = \frac{7x(5x - 6)(3t^2 + \frac{1}{2})}{30\sqrt[3]{(x - 1)^4}\sqrt[3]{\theta}}

$$

  Substituindo $x$ e $t$ pelos seus valores em termos de $\theta$.

3. Dado:


$$

\theta = \frac{3a^2 x}{\sqrt{x^3}}; \quad \omega = \frac{\sqrt{1 - \theta^2}}{1 + \theta}; \quad \phi = \sqrt{3} - \frac{1}{\omega\sqrt{2}}

$$

  Encontre $\frac{d\phi}{dx}$:

  Primeiro, encontramos $\frac{d\theta}{dx}$, $\frac{d\omega}{d\theta}$, e $\frac{d\phi}{d\omega}$:


$$

\theta = \frac{3a^2 x^{-1}}{\sqrt{x^3}} = \frac{3a^2}{x^{5/2}}

$$


$$

\frac{d\theta}{dx} = \frac{d}{dx}\left(\frac{3a^2}{x^{5/2}}\right) = -\frac{15a^2}{2x^{7/2}}

$$


$$

\frac{d\omega}{d\theta} = \frac{d}{d\theta}\left(\frac{\sqrt{1 - \theta^2}}{1 + \theta}\right) = -\frac{1}{(1 + \theta)\sqrt{1 - \theta}}

$$


$$

\frac{d\phi}{d\omega} = \frac{d}{d\omega}\left(\sqrt{3} - \frac{1}{\omega\sqrt{2}}\right) = \frac{1}{\sqrt{2}\omega^2}

$$

  Utilizando a regra da cadeia:


$$

\frac{d\phi}{dx} = \frac{d\phi}{d\omega} \cdot \frac{d\omega}{d\theta} \cdot \frac{d\theta}{dx} = \frac{1}{\sqrt{2}\omega^2} \cdot -\frac{1}{(1 + \theta)\sqrt{1 - \theta}} \cdot -\frac{15a^2}{2x^{7/2}}

$$

  Simplificando:


$$

\frac{d\phi}{dx} = \frac{15a^2}{2\sqrt{2}\omega^2 x^{7/2}(1 + \theta)\sqrt{1 - \theta}}

$$

  Substituindo $\omega$ e $\theta$ pelos seus valores.

  Sabemos que:
$$

\theta = \frac{3a^2}{x^{5/2}}

$$

  Então, substituímos $\theta$ na expressão para $\omega$:


$$

\omega = \frac{\sqrt{1 - \left(\frac{3a^2}{x^{5/2}}\right)^2}}{1 + \frac{3a^2}{x^{5/2}}} = \frac{\sqrt{1 - \frac{9a^4}{x^5}}}{1 + \frac{3a^2}{x^{5/2}}}

$$

  Substituímos $\omega$ na expressão para $\frac{d\phi}{dx}$:


$$

\frac{d\phi}{dx} = \frac{15a^2}{2\sqrt{2} \left( \frac{\sqrt{1 - \frac{9a^4}{x^5}}}{1 + \frac{3a^2}{x^{5/2}}} \right)^2 x^{7/2} \left(1 + \frac{3a^2}{x^{5/2}}\right) \sqrt{1 - \frac{9a^4}{x^5}}}

$$

  Simplificando ainda mais:


$$

\frac{d\phi}{dx} = \frac{15a^2 (1 + \frac{3a^2}{x^{5/2}})^2}{2\sqrt{2} (1 - \frac{9a^4}{x^5}) x^{7/2} \left(1 + \frac{3a^2}{x^{5/2}}\right) \sqrt{1 - \frac{9a^4}{x^5}}}

$$


$$

\frac{d\phi}{dx} = \frac{15a^2 (1 + \frac{3a^2}{x^{5/2}})}{2\sqrt{2} x^{7/2} (1 - \frac{9a^4}{x^5})^{3/2}}

$$

  Portanto, a expressão final para $\frac{d\phi}{dx}$ é:


$$

\frac{d\phi}{dx} = \frac{15a^2 \left(1 + \frac{3a^2}{x^{5/2}}\right)}{2\sqrt{2} x^{7/2} \left(1 - \frac{9a^4}{x^5}\right)^{3/2}}

$$

### Exercícios VII

A seguir estão alguns exercícios que a esforçada leitora deve resolver para fixar o conteúdo:

1. Se $u = \frac{1}{2}x^3; \quad v = 3(u + u^2); \quad w = \frac{1}{v^2}$, encontre $\frac{dw}{dx}$.

  Resposta:


$$

\frac{dw}{dx} = \frac{3x^2 (3 + 3x^3)}{27\left(\frac{1}{2}x^3 + \frac{1}{4}x^6\right)^3}.

$$

2. Se $y = 3x^2 + \sqrt{2}; \quad z = \sqrt{1 + y}; \quad v = \frac{1}{\sqrt{3} + 4z}$, encontre $\frac{dv}{dx}$.

  Resposta:


$$

\frac{dv}{dx} = \frac{12x}{\sqrt{1 + \sqrt{2} + 3x^2} \left( \sqrt{3 + 4\sqrt{1 + \sqrt{2} + 3x^2}} \right)^2}.

$$

3. Se $y = \frac{x^3}{\sqrt{3}}; \quad z = (1 + y)^2; \quad u = \frac{1}{\sqrt{1 + z}}$, encontre $\frac{du}{dx}$.

  Resposta:


$$

\frac{du}{dx} = \frac{x^2\left(\sqrt{3 + x^3}\right)}{\sqrt{\left[1 + \left(\frac{x^3}{\sqrt{3}}\right)\right]^3}}.

$$

## CAPÍTULO X. SIGNIFICADO GEOMÉTRICO DA DIFERENCIAÇÃO

Em cálculo e nas disciplinas que o usa como ferramenta, conhecer o significado geométrico do coeficiente diferencial pode ser muito útil. Porém, antes de continuarmos lembre-se que o coeficiente diferencial representa uma taxa de variação.

A paciente leitora deve lembrar que qualquer função de $x$, como, por exemplo, $x^2$, ou $\sqrt{x}$, ou $ax + b$, pode ser plotada como uma curva, em um gráfico. Com as facilidades que a informática trouxe, qualquer estudante será capaz de plotar estes gráficos. Se não, ainda é possível, usar uma régua e um lápis e escolhendo um conjunto de valores para a variável independente, $x$, encontrar o valor correspondente em $y$ e plotar a curva. Se a amável leitora preferir usar a linguagem Python, há um exemplo no Apêndice II.

Vamos fazer que $PQR$, na Figura 10.1, ser uma fração da curva de uma função plotada em relação aos eixos de coordenadas $OX$ e $OY$. Considere qualquer ponto $Q$ nesta curva, onde a abcissa do ponto é $x$ e sua ordenada é $y$. Agora observe como $y$ muda quando o valor de $x$ varia. Se $x$ sofre um incremento $dx$, para a direita, é possível observar que $y$, nesta curva em particular, também sofre um incremento $dy$. Isso ocorre porque esta curva em particular é uma curva ascendente, ou crescente.

Em uma curva crescente a razão de $dy$ para $dx$ será a medida do grau de inclinação da curva entre os dois pontos $Q$ e $T$. A atenta leitora pode ver na Figura 10.1 que a curva entre $Q$ e $T$ tem muitas inclinações diferentes, de modo que não é possível, nem correto, falar da inclinação da curva entre $Q$ e $T$. No entanto, se os pontos $Q$ e $T$ estão tão próximos um do outro que uma pequena fração, $QT$, da curva é praticamente reta, então estaremos corretos em dizer que a razão $\frac{dy}{dx}$ é a inclinação da curva ao longo de $QT$. A linha reta $QT$ produzida de ambos os lados toca a curva apenas no comprimento de $QT$, e se este comprimento é infinitesimalmente pequeno, a linha reta tocará a curva em exatamente um, e somente um, ponto. Consequentemente esta linha será a curva tangente à curva $PR$ que estamos analisando. Na Figura 10.1 a linha tangente está alongada além do fração infinitesimal $QT$ em uma linha reta e pontilhada.

![]({{ site.baseurl }}/assets/images/cap10-1.jpg){#figura7}
*Figura 10.1 - Significado Geométrico da derivada.*{: class="legend"}

Esta tangente à curva tem evidentemente a mesma inclinação que a curva tem em $QT$, então $\frac{dy}{dx}$ é a inclinação da tangente à curva no ponto $Q$ para o qual se encontrarmos o valor de $\frac{dy}{dx}$. Releia estes dois últimos parágrafos, e observe a Figura 10.1 até estes conceitos estejam claros e evidentes para você.

É fácil perceber que a expressão curta *a inclinação de uma curva* não tem significado preciso, porque uma curva tem muitas inclinações — de fato, cada pequena porção de uma curva tem uma inclinação ligeiramente diferente da fração anterior. Por outro lado, *a inclinação de uma curva em um ponto é uma coisa perfeitamente definida*. Trata-se da inclinação de uma pequena porção da curva situada exatamente naquele ponto. Esta inclinação é, na verdade, o mesmo que *a inclinação da tangente à curva naquele ponto*.

Observe que se $dx$ é um pequeno passo para a direita, $dy$ será um pequeno passo para cima. Esses passos devem ser considerados tão curtos quanto possível — de fato infinitesimalmente curtos, — embora nos diagramas tenhamos que representá-los por partes que não são infinitesimais, ou passos, o quanto $x$ e $y$ variam não poderiam ser vistos.

Consideraremos, doravante, que $\frac{dy}{dx}$ representa a inclinação da curva, de uma determinada função em qualquer um dos pontos desta função.

*Se uma curva está inclinada para cima a $45^\circ$ em um ponto particular, como na Figura 10.2.a, $dy$ e $dx$ serão iguais, e o valor de $\frac{dy}{dx} = 1$*.

*Se a curva inclina-se para cima acentuadamente com mais que $45^\circ$, como na Figura 10.2.b, $\frac{dy}{dx}$ será maior que $1$*.

*Se a curva inclina-se para cima muito suavemente, como na Figura 10.1.c, $\frac{dy}{dx}$ será uma fração menor que $1$*.

![]({{ site.baseurl }}/assets/images/cap10-2.jpg){#figura7}
*Figura 10.2 - Significado Geométrico da derivada.*{: class="legend"}

*Para uma linha horizontal, ou um ponto horizontal em uma curva, $dy = 0$. Isso significa que um aumento infinitesimal, ou não, em $x$, não modifica o valor de $y$ e, portanto, $\frac{dy}{dx} = 0$*.

*Se uma curva inclina-se para baixo, como na Figura 10.2.d, $dy$ será um passo para baixo, e deve, portanto, ser considerado de valor negativo; assim $\frac{dy}{dx}$ também terá um sinal negativo*.

*Se a "curva" acontecer de ser uma linha reta, como na Figura 10.2.b, o valor de $\frac{dy}{dx}$ será o mesmo em todos os pontos ao longo dela. Em outras palavras, sua inclinação é constante*.

*Se uma curva se inclina mais para cima à medida que avança para a direita, os valores de $\frac{dy}{dx}$ se tornarão cada vez maiores com o aumento da inclinação, como nas Figuras 10.2.a, 10.2.b e 10.2.c*.

*Se uma curva é uma que fica cada vez mais plana à medida que avança, os valores de $\frac{dy}{dx}$ se tornarão cada vez menores à medida que a parte mais plana é atingida, como na Figura 14.

Se uma curva primeiro desce e depois sobe novamente, como na Figura 10.3.b apresentando uma concavidade para cima, então claramente $\frac{dy}{dx}$ será inicialmente negativo, com valores decrescentes à medida que a curva se achata, sendo zero no ponto onde o fundo da depressão da curva é alcançado. Desse ponto em diante $\frac{dy}{dx}$ terá valores positivos que continuarão a aumentar. Em tal caso, $y$ é dito estar em um mínimo. O valor mínimo de $y$ não é necessariamente o menor valor de $y$, mas sim aquele valor de $y$ correspondente ao fundo da depressão.

*A característica de um ponto de mínimo é que $y$ deve imediatamente aumentar em ambos os lados deste ponto. Para o valor particular de $x$ que faz $y$ um mínimo, o valor de $\frac{dy}{dx} = 0$*.

Se uma curva primeiro sobe e depois desce, como na Figura 10.3.b, os valores de $\frac{dy}{dx}$ serão positivos no início; depois zero, quando o cume é alcançado; e então negativos, à medida que a curva desce, como na Figura 16. Nesse caso, diz-se que $y$ passa por um máximo, mas o valor máximo de $y$ não é necessariamente o maior valor de $y$. Para o valor particular de $x$ que faz $y$ um máximo, o valor de $\frac{dy}{dx} = 0$.

*A característica de um ponto de máximo é que $y$ deve imediatamente diminuir em ambos os lados deste ponto. Para o valor particular de $x$ que faz $y$ um máximo, o valor de $\frac{dy}{dx} = 0$*.

![]({{ site.baseurl }}/assets/images/cap10-3.jpg){#figura7}
*Figura 10.3 - Análise da derivada em curvas crescentes e decrescentes.*{: class="legend"}

Se uma curva tem a forma peculiar da Figura 10.4, os valores de $\frac{dy}{dx}$ serão sempre positivos; mas haverá um lugar particular onde a inclinação é menor, onde o valor de $\frac{dy}{dx}$ será um mínimo; ou seja, menos do que é em qualquer outra parte da curva.

![]({{ site.baseurl }}/assets/images/cap10-4.jpg){#figura7}
*Figura 10.4 - Análise da derivada em uma curva com concavidade horizontal.*{: class="legend"}

## CAPITULO XVII - INTEGRAÇÃO

O segredo já foi revelado: o símbolo misterioso $\int$ é apenas um $S$ alongado, significa simplesmente _a soma de_ ou _a soma de todas essas quantidades._ Ele se assemelha ao símbolo $\sum$ (a letra Sigma grega), que também indica soma. Mas há uma diferença prática no uso desses símbolos pelos matemáticos: enquanto $\sum$ é geralmente usado para somar um número finito de quantidades, o sinal de integral $\int$ é usado para somar um vasto número de pequenas quantidades infinitamente pequenas, meros elementos que, juntos, formam o total desejado. Assim, $\int dy = y$, e $\int dx = x$.

Neste ponto, a amável leitora já deve entender que o todo de algo pode ser visto como sendo composto por muitos pequenos pedaços; quanto menores os pedaços, mais deles haverá. Por exemplo, uma linha de uma polegada pode ser vista como composta de 10 partes, cada uma com $\frac{1}{10}$ de polegada; ou de 100 partes, cada uma com $\frac{1}{100}$ de polegada; ou de 1.000.000 de partes, cada uma com $\frac{1}{1.000.000}$ de polegada; ou, levando esse pensamento ao limite, pode ser considerada como composta por um número infinito de elementos, cada um infinitesimalmente pequeno.

A leitora deve se perguntar: qual é a utilidade de pensar assim? Por que não pensar no todo diretamente? A resposta é simples: existem muitos casos em que não se pode calcular a grandeza de uma coisa como um todo sem somar muitas pequenas partes. O processo de _integração_ nos permite calcular totais que, de outra forma, não conseguiríamos estimar diretamente.

Vamos começar considerando alguns casos simples para nos familiarizarmos com essa ideia de somar muitas partes separadas.

Considere a série:

$$1 + \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \frac{1}{32} + \frac{1}{64} + \text{etc.}$$

Aqui, cada termo é metade do anterior. Qual seria o valor total se continuássemos até um número infinito de termos? A maioria dos estudantes de ensino médio sabe que a resposta é $2$. Imagine isso como uma linha, como mostrado na Figura XX. Comece com uma polegada, adicione meia polegada, depois um quarto, depois um oitavo, e assim por diante. Se pararmos em qualquer ponto, sempre faltará uma parte para completar as 2 polegadas; e essa parte será do mesmo tamanho que o último termo adicionado. Por exemplo, se somarmos 1, $\frac{1}{2}$ e $\frac{1}{4}$, faltará $\frac{1}{4}$. Se formos até $\frac{1}{64}$, ainda faltará $\frac{1}{64}$. A peça que falta é sempre igual ao último termo adicionado. Somente com um número infinito de operações atingiremos as $2$ polegadas completas.

Na prática, chegaremos perto quando as partes forem tão pequenas que não possam ser desenhadas, isso acontece por volta do décimo termo, pois o décimo primeiro é $\frac{1}{1024}$. Se formos a um ponto onde nem mesmo uma máquina de medição de precisão detecte a diferença, bastará ir até uns 20 termos. Um microscópio não mostrará o $18º$ termo!

A operação de integração é como pegar todo o lote de uma vez. Existem casos em que a integral nos dará o total exato que resultaria de um número infinito de operações. Nesse sentido, o cálculo integral oferece um caminho rápido e direto para um resultado que, de outra forma, demandaria um trabalho interminável e minucioso. Esta é uma das razões pelas quais é importante aprender a integrar.

### INCLINAÇÕES DE CURVAS E AS PRÓPRIAS CURVAS

Vamos fazer uma investigação preliminar sobre as inclinações das curvas. Já vimos que diferenciar uma curva significa encontrar uma expressão para sua inclinação (ou inclinações em diferentes pontos). Mas, será que podemos fazer o processo inverso? Podemos reconstruir a curva inteira se conhecermos sua inclinação (ou inclinações)?

Voltemos ao caso (2). Aqui temos a curva mais simples: uma linha reta com a equação

$$y = ax + b.$$

[A imagem mostra o gráfico de uma linha reta inclinada. O eixo vertical é marcado como Y e o eixo horizontal como X. A linha intercepta o eixo Y em um ponto marcado como $b$. A inclinação da linha é mostrada com pequenos triângulos retângulos ao longo da linha, indicando a taxa constante de variação.]

FIG. 47.

Sabemos que aqui $b$ representa a altura inicial de $y$ quando $x = 0$, e que $a$, que é o mesmo que $\frac{dy}{dx}$, é a "inclinação" da linha. A linha tem uma inclinação constante. Ao longo dela, os triângulos elementares

[A imagem mostra um pequeno triângulo retângulo com a hipotenusa inclinada. O lado vertical é marcado como $dy$ e o lado horizontal como $dx$.]

têm sempre a mesma proporção entre altura e base. Suponha que consideremos os $dx$'s e $dy$'s com magnitudes finitas, de modo que 10 $dx$'s somem uma polegada. Nesse caso, teríamos dez pequenos triângulos como este:

[A imagem mostra uma série de 10 pequenos triângulos retângulos idênticos, alinhados lado a lado.]

Agora, suponha que a tarefa fosse reconstruir a _curva_ apenas com a informação de que $\frac{dy}{dx} = a$. Como procederíamos? Mantendo os pequenos $d$'s de tamanho finito, poderíamos desenhar 10 triângulos, todos com a mesma inclinação, e então uni-los, ponta a ponta, assim:

[A imagem mostra dois gráficos de linhas retas inclinadas. O eixo vertical é marcado como Y e o eixo horizontal como X. A linha inferior é sólida e começa na origem O, enquanto a linha superior é tracejada e começa em um ponto C acima de O no eixo Y. Ambas as linhas são compostas por uma série de pequenos triângulos retângulos, indicando sua inclinação constante.]

FIG. 48.

Como a inclinação é a mesma para todos, eles se juntam para formar, como na Figura 48, uma linha inclinada com a inclinação correta $\frac{dy}{dx} = a$. Seja considerando os $dy$s e $dx$s como finitos ou infinitamente pequenos, sendo todos iguais, é claro que $\frac{y}{x} = a$, se considerarmos $y$ como o total de todos os $dy$s e $x$ como o total de todos os $dx$s. Mas onde colocar essa linha inclinada? Devemos começar na origem $O$ ou mais acima?

Como a única informação que temos é sobre a inclinação, não temos instruções sobre a altura exata acima de $O$. Na verdade, a altura inicial é indeterminada. A inclinação permanece a mesma, independentemente da altura inicial. Então, faremos uma tentativa do que pode ser o desejado e iniciaremos a linha inclinada a uma altura $C$ acima de $O$. Ou seja, temos a equação:

$$y = ax + C.$$

Agora fica claro que, nesse caso, a constante adicionada indica o valor particular que $y$ assume quando $x = 0$.

Vamos agora para um caso mais complicado: uma linha cuja inclinação não é constante, mas que se torna cada vez mais acentuada. Suponha que a inclinação aumente em proporção ao crescimento de $x$. Em símbolos, isso é expresso por:

$$\frac{dy}{dx} = ax.$$

Para um exemplo concreto, tome $a = \frac{1}{5}$, o que nos dá:

$$\frac{dy}{dx} = \frac{1}{5}x.$$

Para entender melhor, vamos calcular alguns valores da inclinação para diferentes valores de $x$ e desenhar pequenos diagramas correspondentes. Quando

[A imagem mostra uma tabela com três colunas e seis linhas. A primeira coluna exibe valores de $x$ de 0 a 5. A segunda coluna mostra os valores correspondentes de $\frac{dy}{dx}$, calculados como $0.2x$. A terceira coluna contém pequenos diagramas de triângulos retângulos, representando visualmente a inclinação para cada valor de $x$.]

| $x$   | $\frac{dy}{dx}$ | Diagrama                          |
|-------|------------------|-----------------------------------|
| $x = 0$ | $\frac{dy}{dx} = 0,$   | [linha horizontal]              |
| $x = 1$ | $\frac{dy}{dx} = 0.2,$ | [triângulo com inclinação leve] |
| $x = 2$ | $\frac{dy}{dx} = 0.4,$ | [triângulo mais inclinado]      |
| $x = 3$ | $\frac{dy}{dx} = 0.6,$ | [triângulo ainda mais inclinado]|
| $x = 4$ | $\frac{dy}{dx} = 0.8,$ | [triângulo muito inclinado]     |
| $x = 5$ | $\frac{dy}{dx} = 1.0.$ | [triângulo com inclinação de 45°]|

Os diagramas na terceira coluna mostram triângulos retângulos com inclinações crescentes, correspondendo aos valores calculados de $\frac{dy}{dx}$.

Agora tente juntar as peças, posicionando cada uma de modo que o meio de sua base esteja na posição correta à direita, encaixando-se nos cantos; como mostrado na Figura 49. O resultado, obviamente, não é uma curva suave, mas uma aproximação de uma. Se tivéssemos usado partes com metade do comprimento, e duas vezes mais numerosas, como na Figura 50, teríamos uma aproximação melhor. Para uma curva perfeita, precisaríamos que cada $dx$ e seu correspondente $dy$ fossem infinitamente pequenos e infinitamente numerosos.

[A imagem mostra dois gráficos, rotulados como Fig. 49 e Fig. 50.]

**Fig. 49:**
[Um gráfico mostra uma curva aproximada formada por segmentos de reta. O eixo Y está à esquerda e o eixo X embaixo. A curva começa na origem O e sobe de forma cada vez mais íngreme até um ponto P. Há 5 segmentos de reta, cada um representando um intervalo de $x = 1$, com triângulos retângulos mostrando a inclinação crescente.]

**Fig. 50:**
[Um gráfico semelhante ao anterior, mas com segmentos de reta menores e mais numerosos, resultando em uma aproximação mais suave da curva. Também começa em O e termina em P, mas com 10 segmentos de reta, cada um representando um intervalo de $x = 0.5$.]

Então, qual deve ser o valor de qualquer $y$? Claramente, em qualquer ponto $P$ da curva, o valor de $y$ é a soma de todos os pequenos $dy$s de 0 até aquele ponto, ou seja, $\int dy = y$. Como cada $dy$ é igual a $\frac{1}{5}x \cdot dx$, segue que o $y$ total será a soma de todos esses pedaços como $\frac{1}{5}x \cdot dx$, ou, como devemos escrever, $\int \frac{1}{5}x \cdot dx$.

Se $x$ fosse constante, $\int \frac{1}{5}x \cdot dx$ seria o mesmo que $\frac{1}{5}x \int dx$, ou $\frac{1}{5}x^2$. Mas $x$ começa em 0 e aumenta até o valor de $x$ no ponto $P$, então seu valor médio de 0 até esse ponto é $\frac{1}{2}x$. Assim, $\int \frac{1}{5}x dx = \frac{1}{10}x^2$, resultando em $y = \frac{1}{10}x^2$.

Como no caso anterior, é necessário adicionar uma constante indeterminada $C$, pois não foi especificado a que altura a curva começa acima da origem quando $x = 0$. Portanto, a equação da curva mostrada na Figura 51 é:

$$y = \frac{1}{10}x^2 + C.$$

[A imagem mostra um gráfico rotulado como Fig. 51. O eixo vertical é marcado como Y e o horizontal como X. Uma curva parabólica é desenhada, começando em um ponto $C$ acima da origem $O$ no eixo Y. A curva se torna mais íngreme à medida que se move para a direita. Uma linha vertical tracejada marca um ponto na curva, com o intervalo entre essa linha e o eixo Y marcado como $x$, e o intervalo entre o eixo X e o ponto na curva marcado como $y$.]

EXERCÍCIOS XVI

(1) Encontre a soma final de $\frac{2}{3} + \frac{1}{3} + \frac{1}{6} + \frac{1}{12} + \frac{1}{24}$ + etc.

(2) Mostre que a série $1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \frac{1}{5} - \frac{1}{6} + \frac{1}{7}$ etc., é convergente, e encontre sua soma até 8 termos.

(3) Se $\log_e(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4}$ + etc., encontre $\log_e 1,3$.

(4) Seguindo um raciocínio similar ao explicado neste capítulo, encontre $y$,

  (a)se $\frac{dy}{dx} = \frac{1}{4}x$; (b)se $\frac{dy}{dx} = \cos x$.

(5) Se $\frac{dy}{dx} = 2x + 3$, encontre $y$.

RESPOSTAS

(1) $1\frac{1}{3}$.

(2) 0,6344.

(3) 0,2624.

(4) (a) $y = \frac{1}{8}x^2 + C$; (b) $y = \sin x + C$.

(5) $y = x^2 + 3x + C$.

## CAPÍTULO XVIII INTEGRAÇÃO COMO O REVERSO DA DIFERENCIAÇÃO

Diferenciar é o processo pelo qual, quando $y$ é dado como uma função de $x$, podemos encontrar $\frac{dy}{dx}$.

Como qualquer operação matemática, o processo de diferenciação deve poder ser revertido. Por exemplo, se diferenciar $y = x^4$ encontrará $\frac{dy}{dx} = 4x^3$, então começar com $\frac{dy}{dx} = 4x^3$ e reverter o processo deverá resultar em $y = x^4$. Mas aqui aparece um ponto curioso: poderíamos obter $\frac{dy}{dx} = 4x^3$ a partir de $x^4$, $x^4 + a$, $x^4 + c$, ou $x^4$ com qualquer constante adicionada.

Isso deixa claro que, ao ir de $\frac{dy}{dx}$ para $y$, devemos prever a possibilidade de uma constante adicionada, cujo valor permanece indeterminado até ser confirmado de outra forma. Assim, se diferenciar $x^n$ resulta em $nx^{n-1}$, reverter de $\frac{dy}{dx} = nx^{n-1}$ nos dá $y = x^n + C$, onde $C$ representa a constante indeterminada possível.


## NOTAS DE RODAPÉ

[:1]:SWIFT, Dean. **On Poetry: a Rhapsody**, p. 20, impresso em 1733 — geralmente citado incorretamente.

[:2]:Nesta posição a relação entre $dx$ e $dy$ será 1, isso irá ocorrer quando o ângulo formado entre a escada e o chão for de $45^\circ$.

[:3]:A forma clássica do teorema binomial, com coeficientes binomiais e combinações, foi desenvolvida, ao longo do tempo, por matemáticos como Al-Karaji, Jia Xian e Omar Khayyam antes de Newton. Isaac Newton, por sua vez, generalizou o teorema para incluir potências fracionárias e negativas, expandindo seu alcance e aplicações.

[:4]:O Pirômetro de Radiação Féry é um dispositivo projetado para medir altas temperaturas detectando a radiação térmica emitida por um objeto. Inventado por Charles Féry no início do século XX, foi um avanço significativo na pirometria, a ciência da medição de temperatura.

## APÊNDICE I - Binômio de Newton

O Teorema Binomial de Newton é uma fórmula matemática que permite expandir expressões da forma $(x + y)^n$, onde $n$ é um número. A fórmula tem o nome de Newton porque ele generalizou o teorema binomial para potências não inteiras. Contudo, o conceito de coeficientes binomiais e suas propriedades foram estudados por vários matemáticos ao longo da história:

1. **[Al-Karaji](https://en.wikipedia.org/wiki/Al-Karaji) (c. 953 – c. 1029)**: Este matemático persa apresentou uma prova por indução do teorema binomial para inteiros positivos e explorou propriedades relacionadas a coeficientes binomiais.
2. **[Jia Xian](https://en.wikipedia.org/wiki/Jia_Xian) (c. 1010 – c. 1070)**: Este matemático chinês desenvolveu o triângulo de Pascal, que facilita o cálculo de coeficientes binomiais.
3. **[Omar Khayyam](https://en.wikipedia.org/wiki/Omar_Khayyam) (1048 – 1131)**: Outro matemático persa que trabalhou em binômios e triângulos aritméticos.
4. **[Blaise Pascal](https://en.wikipedia.org/wiki/Blaise_Pascal) (1623 – 1662)**: Popularizou o triângulo aritmético na Europa, conhecido hoje como triângulo de Pascal.
5. **[Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton) (1642 – 1727)**: Generalizou o teorema binomial para potências não inteiras, ampliando significativamente seu campo de aplicação.

O Teorema Binomial de Newton fornece duas formas principais para expandir potências de binômios.

### 1. Expansão Clássica (para inteiros positivos $n$)

Para um número inteiro positivo $n$, a expansão do binômio $(x + y)^n$ é dada por:


$$

(x + y)^n = \sum\_{k=0}^{n} \binom{n}{k} x^{n-k} y^k

$$

ou de forma equivalentemente, alguns livros grafam como:


$$

(x + y)^n = \sum\_{k=0}^{n} C(n, k) x^{n-k} y^k

$$

Onde $\binom{n}{k}$ ou $C(n, k)$ é o coeficiente binomial, calculado como:


$$

\binom{n}{k} = \frac{n!}{k!(n-k)!}

$$

ou


$$

C(n, k) = \frac{n!}{k!(n-k)!}

$$

### 2. Expansão Generalizada (para qualquer número real ou complexo $n$)

Newton generalizou o teorema binomial para incluir potências fracionárias e negativas. A fórmula geral para a expansão do binômio $(x + y)^n$, onde $n$ pode ser qualquer número real ou complexo, é:


$$

(x + y)^n = \sum\_{k=0}^{\infty} \binom{n}{k} x^{n-k} y^k

$$

Onde o coeficiente binomial generalizado $\binom{n}{k}$ é definido por:


$$

\binom{n}{k} = \frac{n (n-1) (n-2) \cdot (n-k+1)}{k!}

$$

ou equivalentemente,


$$

\binom{n}{k} = \frac{n!}{k!(n-k)!}

$$

para inteiros positivos $n$, e


$$

\binom{n}{k} = \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}

$$

onde $\Gamma(z)$ é a função gama, para outros valores de $n$.

Estas duas formas do teorema binomial permitem a expansão de potências de binômios tanto para casos simples de inteiros positivos quanto para casos mais complexos envolvendo expoentes fracionários ou negativos.

## Apêndice II - Traçando Gráficos em Python

Uma forma fácil e interessante de traçar gráficos de funções e usar a linguagem python. A interessada leitora, não precisa sequer instalar um programa, ou a linguagem. Poderá usar um serviço web, entre tantos disponíveis. Eu gosto de usar o Google Colaboratory disponível em: https://colab.research.google.com/ . Neste serviço as células de código já estão preparadas para rodar Python. Imaginando que a leitora irá tentar, segue um exemplo de código para gerar o Grafico 1 do Capítulo X:

  ```Python
  import numpy as np
  import matplotlib.pyplot as plt

  # Definindo a função
  def f(x):
  return x**2 + 2

  # Derivada da função
  def df(x):
  return 2*x

  # Pontos para o gráfico
  x = np.linspace(0, 2, 400)
  y = f(x)

  # Ponto de tangência
  x_tangent = 1
  y_tangent = f(x_tangent)
  slope = df(x_tangent)

  # Gerando o gráfico
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, y, label=r'$y = x^2 + 2$', color='black')

  # Tangente
  tangent_line = slope * (x - x_tangent) + y_tangent
  ax.plot(x, tangent_line, '--', color='gray')

  # Ponto de tangência
  ax.plot(x_tangent, y_tangent, 'ro')

  # Linhas pontilhadas para y_tangent e x_tangent
  ax.plot([x_tangent, x_tangent], [0, y_tangent], 'k--')
  ax.plot([0, x_tangent], [y_tangent, y_tangent], 'k--')

  # Anotações dos pontos com fontes maiores
  ax.annotate(r'$P$', xy=(0, f(0)), xytext=(-0.05, f(0) + 0.1), fontsize=20)
  ax.annotate(r'$Q$', xy=(x_tangent, y_tangent), xytext=(x_tangent - 0.05, y_tangent + 0.1), fontsize=20)
  ax.annotate(r'$R$', xy=(2, f(2)), xytext=(2, f(2) - 0.05), fontsize=20)

  # Anotações dos eixos com fontes maiores
  ax.annotate(r'$O$', xy=(0, 0), xytext=(-0.1, -0.3), fontsize=20)
  ax.annotate(r'$X$', xy=(2, 0), xytext=(2, -0.3), fontsize=20)
  ax.annotate(r'$Y$', xy=(0, 7), xytext=(-0.3, 7), fontsize=20)

  # Configurações do gráfico
  ax.set_xlabel(r'$X$', fontsize=30)
  ax.set_ylabel(r'$Y$', fontsize=30)
  ax.axhline(0, color='black', linewidth=1.5)
  ax.axvline(0, color='black', linewidth=1.5)

  # Adicionando setas nos eixos
  ax.annotate('', xy=(2.2, 0), xytext=(0, 0),
  arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black'))
  ax.annotate('', xy=(0, 7.5), xytext=(0, 0),
  arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black'))

  ax.grid(True, which='both', color='lightgrey', linestyle='--', linewidth=0.5)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)

  # Ajustando as margens para remover a borda
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  # Salvando a figura como JPG
  plt.savefig('tangent_graph.jpg', format='jpg')
  plt.show()
  ```


## EXERCÍCIOS RESOLVIDOS

### Exercícios I

1. $y = x^{13}$


$$

y + dy = (x + dx)^{13}

$$

  Expandindo pelo binômio de Newton, obteremos:


$$

y + dy = x^{13} + 13x^{12} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = x^{13} + 13x^{12} \cdot dx

$$

  Subtraindo $y = x^{13}$:


$$

dy = 13x^{12} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = 13x^{12}

$$

2. $y = x^{-\frac{3}{2}}$


$$

y + dy = (x + dx)^{-\frac{3}{2}}

$$

  Expandindo pelo binômio de Newton:


$$

y + dy = x^{-\frac{3}{2}} - \frac{3}{2} x^{-\frac{5}{2}} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = x^{-\frac{3}{2}} - \frac{3}{2} x^{-\frac{5}{2}} \cdot dx

$$

  Subtraindo $y = x^{-\frac{3}{2}}$:


$$

dy = -\frac{3}{2} x^{-\frac{5}{2}} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = -\frac{3}{2} x^{-\frac{5}{2}}

$$

3. $y = x^{2a}$


$$

y + dy = (x + dx)^{2a}

$$

  Expandindo pelo binômio de Newton:


$$

y + dy = x^{2a} + 2a x^{2a-1} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = x^{2a} + 2a x^{2a-1} \cdot dx

$$

  Subtraindo $y = x^{2a}$:


$$

dy = 2a x^{2a-1} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = 2a x^{2a-1}

$$

4. $u = t^{2.4}$


$$

u + du = (t + dt)^{2.4}

$$

  Expandindo pelo binômio de Newton:


$$

u + du = t^{2.4} + 2.4 t^{1.4} \cdot dt + \text{termos com } (dt)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

u + du = t^{2.4} + 2.4 t^{1.4} \cdot dt

$$

  Subtraindo $u = t^{2.4}$:


$$

du = 2.4 t^{1.4} \cdot dt

$$

  Portanto:


$$

\frac{du}{dt} = 2.4 t^{1.4}

$$

5. $z = \sqrt[3]{u}$


$$

z + dz = (u + du)^{\frac{1}{3}}

$$

  Expandindo pelo binômio de Newton:


$$

z + dz = u^{\frac{1}{3}} + \frac{1}{3} u^{-\frac{2}{3}} \cdot du + \text{termos com } (du)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

z + dz = u^{\frac{1}{3}} + \frac{1}{3} u^{-\frac{2}{3}} \cdot du

$$

  Subtraindo $z = u^{\frac{1}{3}}$:


$$

dz = \frac{1}{3} u^{-\frac{2}{3}} \cdot du

$$

  Portanto:


$$

\frac{dz}{du} = \frac{1}{3} u^{-\frac{2}{3}}

$$

6. $y = \sqrt[3]{x^{-5}}$


$$

y + dy = (x + dx)^{-\frac{5}{3}}

$$

  Expandindo pelo binômio de Newton:


$$

y + dy = x^{-\frac{5}{3}} - \frac{5}{3} x^{-\frac{8}{3}} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = x^{-\frac{5}{3}} - \frac{5}{3} x^{-\frac{8}{3}} \cdot dx

$$

  Subtraindo $y = x^{-\frac{5}{3}}$:


$$

dy = -\frac{5}{3} x^{-\frac{8}{3}} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = -\frac{5}{3} x^{-\frac{8}{3}}

$$

7. $u = \sqrt{\frac{1}{x^8}}$


$$

u + du = (x + dx)^{-4}

$$

  Expandindo pelo binômio de Newton:


$$

u + du = x^{-4} - 4 x^{-5} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

u + du = x^{-4} - 4 x^{-5} \cdot dx

$$

  Subtraindo $u = x^{-4}$:


$$

du = -4 x^{-5} \cdot dx

$$

  Portanto:


$$

\frac{du}{dx} = -4x^{-5}

$$

8. $y = 2x^{a}$


$$

y + dy = 2(x + dx)^{a}

$$

  Expandindo pelo binômio de Newton:


$$

y + dy = 2(x^a + a x^{a-1} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores})

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = 2x^a + 2a x^{a-1} \cdot dx

$$

  Subtraindo $y = 2x^a$:


$$

dy = 2a x^{a-1} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = 2a x^{a-1}

$$

9. $y = \sqrt[3]{x^3}$


$$

y + dy = (x + dx)

$$

  Expandindo:


$$

y + dy = x + dx

$$

  Subtraindo $y = x$:


$$

dy = dx

$$

  Portanto:


$$

\frac{dy}{dx} = 1

$$

10. $y = \sqrt{\frac{1}{x^m}}$


$$

y + dy = (x + dx)^{-\frac{m}{2}}

$$

  Expandindo pelo binômio de Newton:


$$

y + dy = x^{-\frac{m}{2}} - \frac{m}{2} x^{-\frac{m+2}{2}} \cdot dx + \text{termos com } (dx)^2 \text{ e superiores}

$$

  Negligenciando os termos de ordem superior:


$$

y + dy = x^{-\frac{m}{2}} - \frac{m}{2} x^{-\frac{m+2}{2}} \cdot dx

$$

  Subtraindo $y = x^{-\frac{m}{2}}$:


$$

dy = -\frac{m}{2} x^{-\frac{m+2}{2}} \cdot dx

$$

  Portanto:


$$

\frac{dy}{dx} = -\frac{m}{2} x^{-\frac{m+2}{2}}

$$

### Exercícios II

Aplique os conceitos de derivação que vimos até agora para resolver os seguintes exercícios e problemas:

1. $y = ax^3 + 6$:

  Derivando ambos os lados da equação em relação a $x$:


$$

\frac{dy}{dx} = \frac{d}{dx}(ax^3 + 6)

$$

  Observando que $a$ é uma constante, podemos usar a regra da potência para derivar $ax^3$ e a constante somada, resultando em:


$$

\frac{dy}{dx} = 3ax^2 + 0 = 3ax^2

$$

2. $y = 13x^{\frac{3}{2}} - c$:

  Derive ambos os lados da equação em relação a $x$:


$$

\frac{dy}{dx} = \frac{d}{dx}(13x^{\frac{3}{2}} - c)

$$

  Aplique a regra da potência para derivar $13x^{\frac{3}{2}}$ e a constante:


$$

\frac{dy}{dx} = 13 \cdot \frac{3}{2} x^{\frac{3}{2} - 1} = 13 \cdot \frac{3}{2} x^{\frac{1}{2}} = \frac{39}{2} x^{\frac{1}{2}}

$$


$$

\frac{dy}{dx} = \frac{39}{2} x^{\frac{1}{2}}

$$

3. $y = 12x^{\frac{1}{2}} + c^{\frac{1}{2}}$:

  $c^{\frac{1}{2}}$ é um termo constante e, como tal, pode ser ignorado. No outro termo podemos aplicar a regra da potência:


$$

\frac{dy}{dx} = \frac{d}{dx} 12x^{\frac{1}{2}} = 12\frac{1}{2}x^{1-\frac{1}{2}} = 6x^{-\frac{1}{2}}

$$


$$

\frac{dy}{dx} = 6x^{-\frac{1}{2}}

$$

4. $y = c^{\frac{1}{2}} x^{\frac{1}{2}}$:

  Derive ambos os lados da equação em relação a $x$:


$$

\frac{dy}{dx} = \frac{d}{dx}(c^{\frac{1}{2}} x^{\frac{1}{2}})

$$

  Aplique a regra da potência:


$$

\frac{dy}{dx} = c^{\frac{1}{2}} \cdot \frac{1}{2} x^{\frac{1}{2} - 1} = \frac{c^{\frac{1}{2}}}{2} x^{-\frac{1}{2}}

$$


$$

$\frac{dx}{dy} = \frac{1}{2}c^{\frac{1}{2}}x^{-\frac{1}{2}}$

$$

5. $u = \frac{a \space \space z^n - 1}{c}$:

  Derive ambos os lados da equação em relação a $z$:


$$

\frac{du}{dz} = \frac{d}{dz}\left(\frac{a \space \space z^n - 1}{c}\right)

$$

  A constante $\frac{1}{c}$ pode ser fatorada para fora da derivada:


$$

\frac{du}{dz} = \frac{1}{c} \frac{d}{dz}(a \space \space z^n - 1)

$$

  Aplique a regra da potência e a constante:


$$

\frac{du}{dz} = \frac{1}{c} \left( a \space \space \frac{d}{dz}(z^n) - \frac{d}{dz}(1) \right)

$$

  Derive $z^n$:


$$

\frac{d}{dz}(z^n) = n \space \space z^{n-1}

$$

  Substitua a derivada de $z^n$:


$$

\frac{du}{dz} = \frac{1}{c} \left( a \space \space n \space \space z^{n-1} - 0 \right) = \frac{an}{c} z^{n-1}

$$


$$

\frac{du}{dz} = \frac{an}{c} z^{n-1}

$$

6. $y = 1.18 \space \space t^2 + 22.4$:

  Derive ambos os lados da equação em relação a $z$:


$$

\frac{dy}{dt} = \frac{d}{dt}1.18t^2 + 22.4

$$

  A constante $22.4$ pode ser ignorada:


$$

\frac{dy}{dt} = \frac{d}{dt}(1.18t^2)

$$

  Aplique a regra da potência:


$$

\frac{dy}{dt} = 1.18 \space \space 2t^{2-1}

$$

  Portanto, a solução correta é:


$$

\frac{dy}{dt} = $2.36 \space \space t$

$$

7. Se $l_t$ e $l_0$ forem os comprimentos de uma barra de ferro nas temperaturas $ t^\circ \text{C}$ e $0^\circ \text{C}$, respectivamente, então $l_t = l_0  \space \space (1 + 0.000012 \space \space t)$. Encontre a variação do comprimento da barra por grau Celsius. Resposta: $\frac{dx}{dy} = 0.000012 \space \space l_0$

  Dado que $l_t$ e $l_0$ são os comprimentos de uma barra de ferro nas temperaturas $t^\circ \text{C}$ e $0^\circ \text{C}$, respectivamente, a relação entre o comprimento da barra e a temperatura é dada pela equação:

  $$l_t = l_0  \space \space (1 + 0.000012 \space \space t) = l_0 + 0.000012 \space \space l_0 t $$

  Desejamos encontrar a variação do comprimento da barra por grau Celsius, isto é, a taxa de variação do comprimento $l_t$ em relação à temperatura $t$.

  Distribuímos $l_0$ na equação inicial:


$$

l_t = l_0 + 0.000012 \space \space l_0 t

$$

  Derivamos ambos os lados da equação em relação a $t$:


$$

\frac{d l_t}{d t} = \frac{d}{d t} \left( l_0 + 0.000012 \space \space l_0 t \right)

$$

  No primeiro termo temos $l_0$, que é constante e pode ser descartado. Para o segundo termo ($0.000012  \space \space  l_0 t$), que é um produto de uma constante e uma variável podemos aplicar a regra da potência lembrando que $t=t^1$:


$$

\frac{d}{d t} (0.000012 \space \space l_0 t) = 0.000012 \space \space l_0 \cdot (t^{1-1}) = 0.000012 \space \space l_0

$$

  Portanto, a variação do comprimento da barra por grau Celsius é:


$$

\frac{d l_t}{d t} = 0.000012 \space \space l_0

$$

8. Foi constatado que se $c$ for a potência de uma lâmpada incandescente e $V$ for a voltagem, $c = aV^b$, onde $a$ e $b$ são constantes. Encontre a taxa de variação da potência luminosa com a voltagem e calcule a mudança de potência luminosa por volt em 80, 100 e 120 volts no caso de uma lâmpada para a qual $a = 0.5 \cdot 10^{-10}$ e $b = 6$.

  Foi constatado que se $c$ for a potência de uma lâmpada incandescente e $V$ for a voltagem, $c = aV^b$, onde $a$ e $b$ são constantes. Desejamos encontrar a taxa de variação da potência luminosa com a voltagem e calcular a mudança de potência luminosa por volt em 80, 100 e 120 volts para uma lâmpada na qual $a = 0.5 \cdot 10^{-10}$ e $b = 6$.

  Expressão inicial


$$

c = aV^b

$$

  Derivada de $c$ em relação a $V$ aplicando a regra do produto entre constante e função, e a regra da potência:


$$

\frac{dc}{dV} = \frac{d}{dV} (aV^b) = a \cdot \frac{d}{dV} (V^b) = a \cdot bV^{b-1}

$$

  Portanto, a taxa de variação da potência luminosa com a voltagem é:


$$

\frac{dc}{dV} = abV^{b-1}

$$

  Cálculo da mudança de potência luminosa por volt para valores específicos de voltagem:

  Para $a = 0.5 \cdot 10^{-10}$ e $b = 6$:

  Quando $V = 80$ volts:


$$

\frac{dc}{dV} \bigg|\_{V=80} = 0.5 \cdot 10^{-10} \cdot 6 \cdot 80^{5} = 3 \cdot 10^{-10} \cdot 80^5

$$

  Calculando:


$$

80^5 = 3276800000 \quad \Rightarrow \quad 3 \cdot 10^{-10} \cdot 3276800000 = 0.98 \, \text{watts por volt}

$$

  Quando $V = 100$ volts:**


$$

\frac{dc}{dV} \bigg|\_{V=100} = 0.5 \cdot 10^{-10} \cdot 6 \cdot 100^{5} = 3 \cdot 10^{-10} \cdot 100^5

$$

   Calculando:
$$

100^5 = 100000000 \quad \Rightarrow \quad 3 \cdot 10^{-10} \cdot 100000000 = 3.00 \, \text{watts por volt}

$$

  Quando $V = 120$ volts:


$$

\frac{dc}{dV} \bigg|\_{V=120} = 0.5 \cdot 10^{-10} \cdot 6 \cdot 120^{5} = 3 \cdot 10^{-10} \cdot 120^5

$$

  Calculando:


$$

120^5 = 24883200000 \quad \Rightarrow \quad 3 \cdot 10^{-10} \cdot 24883200000 = 7.47 \, \text{watts por volt}

$$

  Portanto, a mudança de potência luminosa por volt será dada por:


$$

\frac{dc}{dV} = abV^{b-1}, \, 0.98, \, 3.00 \, \text{e} \, 7.47 \, \text{watts por volt, respectivamente}

$$

9. A frequência $n$ de vibração de uma corda de diâmetro $D$, comprimento $L$ e gravidade específica $\sigma$, esticada com uma força $T$, é dada por


$$

n = \frac{1}{DL} \sqrt{\frac{gT}{\pi \sigma}}.

$$

Encontre a taxa de variação da frequência quando $D$, $L$, $\sigma$ e $T$ são variáveis independentes.

  Resposta:

  Derivada de $n$ em relação a $D$:

  A frequência $n$ de vibração de uma corda de diâmetro $D$, comprimento $L$ e gravidade específica $\sigma$, esticada com uma força $T$, é dada por:

  $$n = \frac{1}{DL} \sqrt{\frac{gT}{\pi \sigma}} $$

  Desejamos encontrar a taxa de variação da frequência quando $D$, $L$, $\sigma$ e $T$ são variáveis independentes.


$$

n = \frac{1}{DL} \sqrt{\frac{gT}{\pi \sigma}} = \left(\frac{1}{L} \sqrt{\frac{gT}{\pi \sigma}}\right) \cdot D^{-1}

$$

  Aplicando a regra da potência e a regra do produto entre constante e função:


$$

\frac{dn}{dD} = \left(\frac{1}{L} \sqrt{\frac{gT}{\pi \sigma}}\right) \cdot (-1) D^{-2} = -\frac{1}{LD^2} \sqrt{\frac{gT}{\pi \sigma}}

$$

  Portanto:


$$

\frac{dn}{dD} = -\frac{1}{LD^2} \sqrt{\frac{gT}{\pi \sigma}}

$$

  Derivada de $n$ em relação a $L$:


$$

n = \frac{1}{DL} \sqrt{\frac{gT}{\pi \sigma}} = \left(\frac{1}{D} \sqrt{\frac{gT}{\pi \sigma}}\right) \cdot L^{-1}

$$

  Aplicando a regra da potência e a regra do produto entre constante e função:


$$

\frac{dn}{dL} = \left(\frac{1}{D} \sqrt{\frac{gT}{\pi \sigma}}\right) \cdot (-1) L^{-2} = -\frac{1}{DL^2} \sqrt{\frac{gT}{\pi \sigma}}

$$

  Portanto:


$$

\frac{dn}{dL} = -\frac{1}{DL^2} \sqrt{\frac{gT}{\pi \sigma}}

$$

  Derivada de $n$ em relação a $\sigma$:


$$

n = \frac{1}{DL} \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}}

$$

  Reescrevendo $\sqrt{\frac{gT}{\pi \sigma}}$ como $\left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}}$, aplicamos a regra da potência:


$$

\frac{d}{d\sigma} \left( \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}} \right) = \frac{1}{2} \left(\frac{gT}{\pi \sigma}\right)^{-\frac{1}{2}} \cdot \left(-\frac{gT}{\pi \sigma^2}\right) = -\frac{1}{2} \left(\frac{gT}{\pi \sigma}\right)^{-\frac{1}{2}} \cdot \frac{gT}{\pi \sigma^2}

$$


$$

\frac{d}{d\sigma} \left( \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}} \right) = -\frac{1}{2} \left(\frac{gT}{\pi \sigma^3}\right)^{\frac{1}{2}}

$$

  Portanto:


$$

\frac{dn}{d\sigma} = \frac{1}{DL} \cdot -\frac{1}{2} \sqrt{\frac{gT}{\pi \sigma^3}} = -\frac{1}{2DL} \sqrt{\frac{gT}{\pi \sigma^3}}

$$

  Derivada de $n$ em relação a $T$:


$$

n = \frac{1}{DL} \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}}

$$

  Aplicando a regra da potência:


$$

\frac{d}{dT} \left( \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}} \right) = \frac{1}{2} \left(\frac{gT}{\pi \sigma}\right)^{-\frac{1}{2}} \cdot \frac{g}{\pi \sigma}

$$


$$

\frac{d}{dT} \left( \left(\frac{gT}{\pi \sigma}\right)^{\frac{1}{2}} \right) = \frac{1}{2} \left(\frac{g}{\pi \sigma T}\right)^{\frac{1}{2}}

$$

  Portanto:


$$

\frac{dn}{dT} = \frac{1}{DL} \cdot \frac{1}{2} \sqrt{\frac{g}{\pi \sigma T}} = \frac{1}{2DL} \sqrt{\frac{g}{\pi \sigma T}}

$$

  Em resumo, respostas:


$$

\frac{dn}{dD} = -\frac{1}{LD^2} \sqrt{\frac{gT}{\pi \sigma}}, \quad \frac{dn}{dL} = -\frac{1}{DL^2} \sqrt{\frac{gT}{\pi \sigma}},

$$


$$

\frac{dn}{d\sigma} = -\frac{1}{2DL} \sqrt{\frac{gT}{\pi \sigma^3}}, \quad \frac{dn}{dT} = \frac{1}{2DL} \sqrt{\frac{g}{\pi \sigma T}}.

$$

10. A maior pressão externa $P$ que um tubo pode suportar sem colapsar é dada por


$$

P = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3}{D^3},

$$

onde $E$ e $\sigma$ são constantes, $t$ é a espessura do tubo e $D$ é seu diâmetro. (Esta fórmula assume que $4t$ é pequeno em comparação com $D$.)

a) Compare a taxa em que $P$ varia para uma pequena mudança de espessura e para uma pequena mudança de diâmetro ocorrendo separadamente.

b) Compare a taxa na qual $P$ varia para uma pequena mudança de espessura e para uma pequena mudança de diâmetro ocorrendo separadamente.

  Taxa de Variação da Pressão Externa $P$

  A maior pressão externa $P$ que um tubo pode suportar sem colapsar é dada por:


$$

P = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3}{D^3}

$$

  onde $E$ e $\sigma$ são constantes, $t$ é a espessura do tubo e $D$ é seu diâmetro.

  Letra a) Taxa de Variação com Respeito à Espessura $t$

  Considere a expressão inicial:


$$

P = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3}{D^3}

$$

  Aplicando a regra do produto entre constante e função e a regra da potência:


$$

\frac{dP}{dt} = \left( \frac{2E}{1 - \sigma^2} \right) \frac{d}{dt} \left( \frac{t^3}{D^3} \right) = \left( \frac{2E}{1 - \sigma^2} \right) \frac{3t^2}{D^3}

$$

  Portanto:
$$

\frac{dP}{dt} = \left( \frac{2E}{1 - \sigma^2} \right) \frac{3t^2}{D^3}

$$

  Letra b) Taxa de Variação com Respeito ao Diâmetro $D$

  Considere novamente a expressão inicial:


$$

P = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3}{D^3}

$$

  Aplicando a regra do produto entre constante e função e a regra da potência:


$$

\frac{dP}{dD} = \left( \frac{2E}{1 - \sigma^2} \right) \frac{d}{dD} \left( \frac{t^3}{D^3} \right) = \left( \frac{2E}{1 - \sigma^2} \right) \frac{t^3 \cdot (-3) D^{-4}}{1} = \left( \frac{2E}{1 - \sigma^2} \right) \frac{-3t^3}{D^4}

$$

  Portanto:
$$

\frac{dP}{dD} = -\left( \frac{2E}{1 - \sigma^2} \right) \frac{3t^3}{D^4}

$$

  Respostas

  1. Taxa de Variação com Respeito à Espessura $t$:


$$

\frac{dP}{dt} = \left( \frac{2E}{1 - \sigma^2} \right) \frac{3t^2}{D^3}

$$

2. Taxa de Variação com Respeito ao Diâmetro $D$:


$$

\frac{dP}{dD} = -\left( \frac{2E}{1 - \sigma^2} \right) \frac{3t^3}{D^4}

$$

  Fazendo uma análise comparativa:

- A taxa de variação da pressão $P$ com respeito à espessura $t$ é proporcional a $t^2$, ou seja, aumenta rapidamente com $t$.
- A taxa de variação da pressão $P$ com respeito ao diâmetro $D$ é proporcional a $\frac{1}{D^4}$, ou seja, diminui rapidamente com $D$.

  Portanto, uma pequena mudança na espessura $t$ terá um impacto significativo na pressão $P$, enquanto uma pequena mudança no diâmetro $D$ também terá um impacto significativo, mas na direção oposta.

11. Encontre, a partir dos primeiros princípios, a taxa na qual os seguintes variam em relação a uma mudança no raio:

(a) - a circunferência de um círculo de raio $r$;
(b) - a área de um círculo de raio $r$;
(c) - a área lateral de um cone de dimensão inclinada $l$;
(d) - o volume de um cone de raio $r$ e altura $h$;
(e) - a área de uma esfera de raio $r$;
(f) - o volume de uma esfera de raio $r$.

  (a) Circunferência de um círculo de raio $r$

  A circunferência $C$ de um círculo é dada por:

  $$C = 2\pi r$$

  Aplicando a regra da derivada do produto entre constante e função:

  $$\frac{dC}{dr} = \frac{d}{dr}(2\pi r) = 2\pi \cdot \frac{d}{dr}(r) = 2\pi$$

  Portanto, a taxa de variação da circunferência em relação ao raio é:

$$\frac{dC}{dr} = 2\pi$$

  (b) Área de um círculo de raio $r$

  A área $A$ de um círculo é dada por:

  $$A = \pi r^2$$

  Aplicando a regra da potência:

  $$\frac{dA}{dr} = \frac{d}{dr}(\pi r^2) = \pi \cdot 2r = 2\pi r$$

  Portanto, a taxa de variação da área em relação ao raio é:

  $$\frac{dA}{dr} = 2\pi r$$

  (c) Área lateral de um cone de dimensão inclinada $l$

  A área lateral $A$ de um cone é dada por:

  $$A = \pi r l$$

  Aplicando a regra da derivada do produto entre constante e função:

  $$\frac{dA}{dr} = \frac{d}{dr}(\pi r l) = \pi l \cdot \frac{d}{dr}(r) = \pi l$$

  Portanto, a taxa de variação da área lateral em relação ao raio é:

  $$\frac{dA}{dr} = \pi l$$

  (d) Volume de um cone de raio $r$ e altura $h$

  O volume $V$ de um cone é dado por:

  $$V = \frac{1}{3} \pi r^2 h$$

  Aplicando a regra da potência e a regra do produto entre constante e função:

  $$\frac{dV}{dr} = \frac{d}{dr}\left(\frac{1}{3} \pi r^2 h\right) = \frac{1}{3} \pi h \cdot \frac{d}{dr}(r^2) = \frac{1}{3} \pi h \cdot 2r = \frac{2}{3} \pi rh$$

  Portanto, a taxa de variação do volume em relação ao raio é:

  $$\frac{dV}{dr} = \frac{2}{3} \pi rh$$

  (e) Área de uma esfera de raio $r$

  A área $A$ de uma esfera é dada por:

  $$A = 4\pi r^2$$

  Aplicando a regra da potência:

  $$\frac{dA}{dr} = \frac{d}{dr}(4\pi r^2) = 4\pi \cdot 2r = 8\pi r$$

  Portanto, a taxa de variação da área em relação ao raio é:

  $$\frac{dA}{dr} = 8\pi r$$

  (f) Volume de uma esfera de raio $r$

  O volume $V$ de uma esfera é dado por:

  $$V = \frac{4}{3} \pi r^3$$

  Aplicando a regra da potência:

  $$\frac{dV}{dr} = \frac{d}{dr}\left(\frac{4}{3} \pi r^3\right) = \frac{4}{3} \pi \cdot 3r^2 = 4\pi r^2$$

  Portanto, a taxa de variação do volume em relação ao raio é:

  $$\frac{dV}{dr} = 4\pi r^2$$

  Respostas:

  (a) $ \frac{dC}{dr} = 2\pi $
(b) $ \frac{dA}{dr} = 2\pi r $
(c) $ \frac{dA}{dr} = \pi l $
(d) $ \frac{dV}{dr} = \frac{2}{3} \pi rh $
(e) $ \frac{dA}{dr} = 8\pi r $
(f) $ \frac{dV}{dr} = 4\pi r^2 $

(12) O comprimento $L$ de uma barra de ferro na temperatura $T$ é dado por


$$

L = l_t \left[ 1 + 0.000012 (T - t) \right],

$$

  onde $l_t$ é o comprimento na temperatura $t$. Encontre a taxa de variação do diâmetro $D$ de um pneu de ferro adequado para ser encolhido em uma roda, quando a temperatura $T$ varia.

  O comprimento $L$ de uma barra de ferro na temperatura $T$ é dado por:

  $$L = l_t \left[ 1 + 0.000012 (T - t) \right]$$

  onde $l_t$ é o comprimento na temperatura $t$. Desejamos encontrar a taxa de variação do diâmetro $D$ de um pneu de ferro adequado para ser encolhido em uma roda, quando a temperatura $T$ varia.

  $$L = l_t \left[ 1 + 0.000012 (T - t) \right]$$

  Aplicando a regra do produto entre constante e função:

  $$\frac{dL}{dT} = l_t \cdot \frac{d}{dT} \left( 1 + 0.000012 (T - t) \right)$$

  Derivada do termo dentro dos colchetes:

  $$\frac{d}{dT} \left( 1 + 0.000012 (T - t) \right) = 0 + 0.000012 \cdot \frac{d}{dT}(T - t)$$

  Como $t$ é constante, sua derivada em relação a $T$ é zero:**

  $$\frac{d}{dT} (T - t) = \frac{dT}{dT} - \frac{dt}{dT} = 1 - 0 = 1$$

  Portanto:

  $$\frac{d}{dT} \left( 1 + 0.000012 (T - t) \right) = 0.000012$$

  Substituindo na derivada de $L$:

  $$\frac{dL}{dT} = l_t \cdot 0.000012 = 0.000012 l_t$$

  Agora, considerando que o diâmetro $D$ está relacionado ao comprimento $L$ da barra, a taxa de variação do diâmetro $D$ em relação à temperatura $T$ será proporcional à taxa de variação do comprimento $L$. Para um pneu de ferro adequado para ser encolhido em uma roda, assumimos que o diâmetro $D$ varia linearmente com o comprimento $L$:

  $$\frac{dD}{dT} = \frac{dD}{dL} \cdot \frac{dL}{dT}$$

  Sabemos que $D$ é proporcional a $L$ e que a relação entre eles é $D = \frac{L}{\pi}$ (uma suposição de que o pneu tem uma forma circular onde $D = \frac{L}{\pi}$):

  Portanto:

  $$\frac{dD}{dL} = \frac{1}{\pi}$$

  Então:

  $$\frac{dD}{dT} = \frac{1}{\pi} \cdot 0.000012 l_t = \frac{0.000012 l_t}{\pi}$$

  Resposta

  $$\frac{dD}{dT} = \frac{0.000012 l_t}{\pi}$$

### Exercícios III

1. Letra (a): Diferencie $u = 1 + x + \frac{x^2}{1 \cdot 2} + \frac{x^3}{1 \cdot 2 \cdot 3} + \cdots$.

  Começamos com a regra da potência para diferenciar cada termo:


$$

\frac{du}{dx} = \frac{d}{dx} \left(1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots \right)

$$

  Diferenciando cada termo, obteremos:


$$

\frac{du}{dx} = 0 + 1 + \frac{2x}{2} + \frac{3x^2}{6} + \cdots

$$

  Simplificando:


$$

\frac{du}{dx} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots

$$

  A função $u$ é a expansão em [Série de Maclaurin](https://en.wikipedia.org/wiki/Taylor_series) da função exponencial $e^x$. Portanto, a derivada de $u$ em relação a $x$ é a própria função:


$$

\frac{du}{dx} = \frac{d}{dx} (e^x) = e^x

$$

  ou, escrito na forma da série:


$$

\frac{du}{dx} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots

$$

  Letra (b): Diferencie $y = ax^2 + bx + c$.

  Começamos, novamente, com a Regra da Potência e a regra da soma:


$$

\frac{dy}{dx} = \frac{d}{dx} (ax^2) + \frac{d}{dx} (bx) + \frac{d}{dx} (c)

$$

  Diferenciando cada termo, usando a Regra da Soma, obteremos:


$$

\frac{dy}{dx} = 2ax + b + 0

$$

  Simplificando:


$$

\frac{dy}{dx} = 2ax + b

$$

  Letra (c): Diferencie $y = (x + a)^2$.

  Primeiro, expandimos a expressão:


$$

y = (x + a)^2 = x^2 + 2ax + a^2

$$

  Agora, diferenciamos cada termo usando a regra da potência, a regra da multiplicação de função por constante, e a regra da soma:

  Derivada de $x^2$:


$$

\frac{d}{dx} (x^2) = 2x

$$

  Derivada de $2ax$:


$$

\frac{d}{dx} (2ax) = 2a \cdot \frac{d}{dx} (x) = 2a \cdot 1 = 2a

$$

  Derivada de $a^2$:


$$

\frac{d}{dx} (a^2) = 0 \quad \text{(porque $a^2$ é uma constante em relação a $x$)}

$$

  Somando as derivadas:


$$

\frac{dy}{dx} = 2x + 2a + 0 = 2x + 2a

$$

  Portanto, a derivada de $y = (x + a)^2$ é:


$$

\frac{dy}{dx} = 2x + 2a

$$

  Este exercício seria mais simples se usássemos a Regra da Cadeia, que ainda não vimos.

  Letra (d): Diferencie $y = (x + a)^3$.

  Novamente, seria mais simples usando a Regra da Cadeia, mas como ainda não vimos esta regra, começamos expandindo a expressão:


$$

y = (x + a)^3 = x^3 + 3ax^2 + 3a^2x + a^3

$$

  Agora, diferenciamos cada termo usando a regra da potência, a regra da multiplicação de função por constante, e a regra da soma:

  Derivada de $x^3$:


$$

\frac{d}{dx} (x^3) = 3x^2

$$

  Derivada de $3ax^2$:


$$

\frac{d}{dx} (3ax^2) = 3a \cdot \frac{d}{dx} (x^2) = 3a \cdot 2x = 6ax

$$

  Derivada de $3a^2x$:


$$

\frac{d}{dx} (3a^2x) = 3a^2 \cdot \frac{d}{dx} (x) = 3a^2 \cdot 1 = 3a^2

$$

  Derivada de $a^3$:


$$

\frac{d}{dx} (a^3) = 0 \quad \text{(porque $a^3$ é uma constante em relação a $x$)}

$$

  Agora, somamos as derivadas:


$$

\frac{dy}{dx} = 3x^2 + 6ax + 3a^2 + 0 = 3x^2 + 6ax + 3a^2

$$

  Portanto, a derivada de $y = (x + a)^3$ é:


$$

\frac{dy}{dx} = 3x^2 + 6ax + 3a^2

$$

2. Se $w = at - \frac{1}{2}bt^2$, encontre $\frac{dw}{dt}$.

  Usamos a regra da potência e a regra da soma:


$$

\frac{dw}{dt} = \frac{d}{dt} (at) - \frac{d}{dt} \left(\frac{1}{2}bt^2\right)

$$

  Diferenciando cada termo, obteremos:


$$

\frac{dw}{dt} = a - bt

$$

3. Encontre o coeficiente diferencial de:


$$

y = (x + \sqrt{-1}) \cdot (x - \sqrt{-1})

$$

  Primeiro, simplificamos a expressão:


$$

y = x^2 - (\sqrt{-1})^2 = x^2 + 1

$$

  Agora, diferenciamos:


$$

\frac{dy}{dx} = \frac{d}{dx} (x^2 + 1) = 2x + 0 = 2x

$$

4. Diferencie:


$$

y = (197x - 34x^2) \cdot (7 + 22x - 83x^3)

$$

  Usamos a regra do produto:


$$

\frac{dy}{dx} = \frac{d}{dx} (197x - 34x^2) \cdot (7 + 22x - 83x^3) + (197x - 34x^2) \cdot \frac{d}{dx} (7 + 22x - 83x^3)

$$

  Diferenciando cada termo:


$$

\frac{d}{dx} (197x - 34x^2) = 197 - 68x

$$


$$

\frac{d}{dx} (7 + 22x - 83x^3) = 22 - 249x^2

$$

  Substituindo de volta na regra do produto:


$$

\frac{dy}{dx} = (197 - 68x)(7 + 22x - 83x^3) + (197x - 34x^2)(22 - 249x^2)

$$

  Simplificando:


$$

\frac{dy}{dx} = 1379 + 4322x - 14110x^4 - 65404x^3 - 2244x^2 + 8192x

$$

  Rearranjando os termos:


$$

\frac{dy}{dx} = 14110x^4 - 65404x^3 - 2244x^2 + 8192x + 1379

$$

5. Se $x = (y + 3) \cdot (y + 5)$, encontre $\frac{dx}{dy}$.

  Usamos a regra do produto:


$$

\frac{dx}{dy} = \frac{d}{dy} (y + 3) \cdot (y + 5) + (y + 3) \cdot \frac{d}{dy} (y + 5)

$$

  Diferenciando cada termo:


$$

\frac{d}{dy} (y + 3) = 1

- $$

$$
\frac{d}{dy} (y + 5) = 1
$$

Substituindo de volta na regra do produto:

$$
\frac{dx}{dy} = 1 \cdot (y + 5) + (y + 3) \cdot 1
$$

Simplificando:

$$
\frac{dx}{dy} = y + 5 + y + 3 = 2y + 8
$$

6. Diferencie $y = 1.3709x \cdot (112.6 + 45.202x^2)$.

Começamos com a Regra do Produto:

$$
\frac{dy}{dx} = \frac{d}{dx} \left( 1.3709x \right) \cdot (112.6 + 45.202x^2) + 1.3709x \cdot \frac{d}{dx} \left( 112.6 + 45.202x^2 \right)
$$

Diferenciando cada termo:

$$
\frac{d}{dx} \left( 1.3709x \right) = 1.3709
$$

$$
\frac{d}{dx} \left( 112.6 + 45.202x^2 \right) = 0 + 2 \cdot 45.202x = 90.404x
$$

Substituindo de volta na regra do produto:

$$
\frac{dy}{dx} = 1.3709 \cdot (112.6 + 45.202x^2) + 1.3709x \cdot 90.404x
$$

Simplificando:

$$
\frac{dy}{dx} = 1.3709 \cdot 112.6 + 1.3709 \cdot 45.202x^2 + 1.3709 \cdot 90.404x^2
$$

$$
\frac{dy}{dx} = 154.36334 + 1.3709 \cdot 135.606x^2
$$

$$
\frac{dy}{dx} = 154.36334 + 185.9022654x^2
$$

Portanto, a derivada de $y = 1.3709x \cdot (112.6 + 45.202x^2)$ é:

$$
\frac{dy}{dx} = 185.9022654x^2 + 154.36334
$$

7. Diferencie $y = \frac{2x + 3}{3x + 2}$.

Usamos a Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(3x + 2) \cdot \frac{d}{dx}(2x + 3) - (2x + 3) \cdot \frac{d}{dx}(3x + 2)}{(3x + 2)^2}
$$

Diferenciando cada termo:

$$
\frac{d}{dx} (2x + 3) = 2
$$

$$
\frac{d}{dx} (3x + 2) = 3
$$

Substituindo de volta na Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(3x + 2) \cdot 2 - (2x + 3) \cdot 3}{(3x + 2)^2}
$$

Simplificando:

$$
\frac{dy}{dx} = \frac{6x + 4 - 6x - 9}{(3x + 2)^2}
$$

$$
\frac{dy}{dx} = \frac{-5}{(3x + 2)^2}
$$

Portanto, a derivada de $y = \frac{2x + 3}{3x + 2}$ é:

$$
\frac{dy}{dx} = \frac{-5}{(3x + 2)^2}
$$

8. Diferencie $y = \frac{1 + x + 2x^2 + 3x^3}{1 + x + 2x^2}$.

Usamos a Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(1 + x + 2x^2) \cdot \frac{d}{dx}(1 + x + 2x^2 + 3x^3) - (1 + x + 2x^2 + 3x^3) \cdot \frac{d}{dx}(1 + x + 2x^2)}{(1 + x + 2x^2)^2}
$$

Diferenciando cada termo:

$$
\frac{d}{dx} (1 + x + 2x^2 + 3x^3) = 0 + 1 + 4x + 9x^2 = 1 + 4x + 9x^2
$$

$$
\frac{d}{dx} (1 + x + 2x^2) = 0 + 1 + 4x = 1 + 4x
$$

Substituindo de volta na Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(1 + x + 2x^2) \cdot (1 + 4x + 9x^2) - (1 + x + 2x^2 + 3x^3) \cdot (1 + 4x)}{(1 + x + 2x^2)^2}
$$

Simplificando o numerador:

$$
\begin{align*}
& (1 + x + 2x^2)(1 + 4x + 9x^2) - (1 + x + 2x^2 + 3x^3)(1 + 4x) \\
&= (1 + 4x + 9x^2 + x + 4x^2 + 9x^3 + 2x^2 + 8x^3 + 18x^4) \\
& - (1 + 4x + x + 4x^2 + 2x^2 + 8x^3 + 3x^3 + 12x^4) \\
&= 1 + 5x + 15x^2 + 17x^3 + 18x^4 - (1 + 4x + x + 4x^2 + 2x^2 + 8x^3 + 3x^3 + 12x^4) \\
&= 6x^3 + 6x^2 + 9x^4
\end{align*}
$$

Portanto:

$$
\frac{dy}{dx} = \frac{6x^4 + 6x^3 + 9x^2}{(1 + x + 2x^2)^2}
$$

9. Diferencie $y = \frac{ax + b}{cx + d}$.

Usamos a Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(cx + d) \cdot \frac{d}{dx}(ax + b) - (ax + b) \cdot \frac{d}{dx}(cx + d)}{(cx + d)^2}
$$

Diferenciando cada termo:

$$
\frac{d}{dx} (ax + b) = a
$$

$$
\frac{d}{dx} (cx + d) = c
$$

Substituindo de volta na Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(cx + d) \cdot a - (ax + b) \cdot c}{(cx + d)^2}
$$

Simplificando:

$$
\frac{dy}{dx} = \frac{acx + ad - acx - bc}{(cx + d)^2}
$$

$$
\frac{dy}{dx} = \frac{ad - bc}{(cx + d)^2}
$$

Portanto, a derivada de $y = \frac{ax + b}{cx + d}$ é:

$$
\frac{dy}{dx} = \frac{ad - bc}{(cx + d)^2}
$$

10. Diferencie $y = \frac{x^n + a}{x^{-n} + b}$.

Começamos Aplicando a Regra do Quociente ao nosso problema:

$$
\frac{dy}{dx} = \frac{(x^{-n} + b) \cdot \frac{d}{dx}(x^n + a) - (x^n + a) \cdot \frac{d}{dx}(x^{-n} + b)}{(x^{-n} + b)^2}
$$

Agora vamos diferenciar cada termo individualmente. Usando a regra da potência, teremos:

$$
\frac{d}{dx} (x^n + a) = nx^{n-1}
$$

e

$$
\frac{d}{dx} (x^{-n} + b) = -nx^{-n-1}
$$

Substituímos as derivadas na Regra do Quociente.

$$
\frac{dy}{dx} = \frac{(x^{-n} + b) \cdot nx^{n-1} - (x^n + a) \cdot (-nx^{-n-1})}{(x^{-n} + b)^2}
$$

Simplificando.

$$
\frac{dy}{dx} = \frac{nx^{n-1}x^{-n} + bnx^{n-1} + nx^{-n-1}x^n + anx^{-n-1}}{(x^{-n} + b)^2}
$$

Simplificando os expoentes, teremos:

$$
\frac{dy}{dx} = \frac{nx^{-1} + bnx^{n-1} + nx^{-1} + anx^{-n-1}}{(x^{-n} + b)^2}
$$

Combinando os termos semelhantes:

$$
\frac{dy}{dx} = \frac{2nx^{-1} + bnx^{n-1} + anx^{-n-1}}{(x^{-n} + b)^2}
$$

Logo:

$$
\frac{dy}{dx} = \frac{2nx^{-1} + bnx^{n-1} + anx^{-n-1}}{(x^{-n} + b)^2}
$$

11. Encontre a variação da corrente $C$ em relação à temperatura $t$ para a expressão $C = a + bt + ct^2$.

Diferenciamos a expressão com relação a $t$:

$$
\frac{dC}{dt} = \frac{d}{dt} (a + bt + ct^2)
$$

Usando a regra da soma, a regra da constante e a regra da potência:

$$
\frac{dC}{dt} = 0 + b + 2ct
$$

Portanto, a taxa de variação da corrente em relação à temperatura é:

$$
\frac{dC}{dt} = b + 2ct
$$

12. Encontre a taxa de variação da resistência em relação à temperatura para as seguintes expressões.

Primeira equação: $R = R_0 (1 + at + bt^2)$

Diferenciamos a expressão com relação a $t$:

$$
\frac{dR}{dt} = R_0 \frac{d}{dt} (1 + at + bt^2)
$$

Usando a regra da soma, a regra da constante e a regra da potência:

$$
\frac{dR}{dt} = R_0 (0 + a + 2bt)
$$

Portanto:

$$
\frac{dR}{dt} = R_0 (a + 2bt)
$$

Segunda equação: $R = R_0 (1 + at + b\sqrt{t})$

Diferenciamos a expressão com relação a $t$:

$$
\frac{dR}{dt} = R_0 \frac{d}{dt} (1 + at + b\sqrt{t})
$$

Usando a regra da soma, a regra da constante e a regra da raiz quadrada:

Derivada de $1$:

$$
\frac{d}{dt} (1) = 0
$$

Derivada de $at$:

$$
\frac{d}{dt} (at) = a
$$

Derivada de $b\sqrt{t}$:

$$
\frac{d}{dt} (b\sqrt{t}) = b \cdot \frac{1}{2\sqrt{t}} = \frac{b}{2\sqrt{t}}
$$

Portanto:

$$
\frac{dR}{dt} = R_0 (0 + a + \frac{b}{2\sqrt{t}})
$$

Simplificando:

$$
\frac{dR}{dt} = R_0 \left( a + \frac{b}{2\sqrt{t}} \right)
$$

Terceira equação: $R = R_0 (1 + at + bt^2)^{-1}$

Usamos a Regra do Quociente, onde $f = R_0$ e $g = 1 + at + bt^2$:

Primeiro, reescrevemos a expressão usando a Regra do Quociente:

$$
R = \frac{R_0}{1 + at + bt^2}
$$

Diferenciamos a expressão com relação a $t$:

$$
\frac{dR}{dt} = \frac{(1 + at + bt^2) \cdot 0 - R_0 \cdot \frac{d}{dt}(1 + at + bt^2)}{(1 + at + bt^2)^2}
$$

Diferenciamos o denominador:

Derivada de $1$:

$$
\frac{d}{dt} (1) = 0
$$

Derivada de $at$:

$$
\frac{d}{dt} (at) = a
$$

Derivada de $bt^2$:

$$
\frac{d}{dt} (bt^2) = 2bt
$$

Portanto:

$$
\frac{d}{dt} (1 + at + bt^2) = a + 2bt
$$

Substituindo de volta na Regra do Quociente:

$$
\frac{dR}{dt} = \frac{0 - R_0 (a + 2bt)}{(1 + at + bt^2)^2}
$$

Simplificando:

$$
\frac{dR}{dt} = -R_0 \cdot \frac{a + 2bt}{(1 + at + bt^2)^2}
$$

Portanto, a taxa de variação da resistência em relação à temperatura é:

$$
\frac{dR}{dt} = -R_0 \cdot \frac{a + 2bt}{(1 + at + bt^2)^2}
$$

13. A força eletromotriz $E$ varia com a temperatura $t$ de acordo com:

$$
E = 1.4340 \left[ 1 - 0.000814(t - 15) + 0.000007(t - 15)^2 \right]
$$

Encontre a variação da força eletromotriz por grau a $15^\circ$, $20^\circ$ e $25^\circ$.

Diferenciamos a expressão com relação a $t$:

$$
\frac{dE}{dt} = 1.4340 \left[ -0.000814 + 0.000014(t - 15) \right]
$$

Substituímos $t$ pelos valores dados:

Para $t = 15$:

$$
\frac{dE}{dt} \bigg|_{t=15} = 1.4340 \left[ -0.000814 + 0.000014(15 - 15) \right] = -0.00116796
$$

Para $t = 20$:

$$
\frac{dE}{dt} \bigg|_{t=20} = 1.4340 \left[ -0.000814 + 0.000014(20 - 15) \right] = -0.0010976
$$

Para $t = 25$:

$$
\frac{dE}{dt} \bigg|_{t=25} = 1.4340 \left[ -0.000814 + 0.000014(25 - 15) \right] = -0.0010272
$$

14. A força eletromotriz $E$ para manter um arco elétrico de comprimento $l$ com corrente $i$ é dada por:

$$
E = a + bl + \frac{c + kl}{i}
$$

Parte 1: Variação da força eletromotriz com relação ao comprimento do arco $l$

Diferenciamos a expressão com relação a $l$:

$$
\frac{dE}{dl} = b + \frac{d}{dl} \left( \frac{c + kl}{i} \right)
$$

Usamos a regra da derivada de uma constante e a Regra do Quociente:

$$
\frac{d}{dl} \left( \frac{c + kl}{i} \right) = \frac{k}{i}
$$

Portanto:

$$
\frac{dE}{dl} = b + \frac{k}{i}
$$

Parte 2: Variação da força eletromotriz com relação à corrente $i$

Diferenciamos a expressão com relação a $i$:

$$
\frac{dE}{di} = \frac{d}{di} \left( a + bl + \frac{c + kl}{i} \right)
$$

Usamos a regra da derivada de uma constante e a Regra do Quociente:

$$
\frac{d}{di} \left( \frac{c + kl}{i} \right) = -\frac{c + kl}{i^2}
$$

Portanto:

$$
\frac{dE}{di} = -\frac{c + kl}{i^2}
$$

### Exercícios IV

Encontre $\frac{dy}{dx}$ e $\frac{d^2 y}{dx^2}$ para as seguintes expressões:

1. $y = 17x + 12x^2$.

Primeira derivada:

$$
\frac{dy}{dx} = 17 + 24x
$$

Segunda derivada:

$$
\frac{d^2 y}{dx^2} = 24
$$

2. $y = \frac{x^2 + a}{x + a}$.

Primeira derivada, usando a Regra do Quociente:

$$
\frac{dy}{dx} = \frac{(x + a) \cdot 2x - (x^2 + a) \cdot 1}{(x + a)^2} = \frac{2x(x + a) - (x^2 + a)}{(x + a)^2}
$$

Simplificando:

$$
\frac{dy}{dx} = \frac{2x^2 + 2ax - x^2 - a}{(x + a)^2} = \frac{x^2 + 2ax - a}{(x + a)^2}
$$

Segunda derivada:

Usamos a Regra do Quociente novamente:

$$
\frac{d^2 y}{dx^2} = \frac{(x + a)^2 \cdot \frac{d}{dx}(x^2 + 2ax - a) - (x^2 + 2ax - a) \cdot \frac{d}{dx}((x + a)^2)}{((x + a)^2)^2}
$$

Diferenciando o numerador e o denominador:

$$
\frac{d}{dx}(x^2 + 2ax - a) = 2x + 2a
$$

$$
\frac{d}{dx}((x + a)^2) = 2(x + a)
$$

Substituindo e simplificando:

$$
\frac{d^2 y}{dx^2} = \frac{(x + a)^2 \cdot (2x + 2a) - (x^2 + 2ax - a) \cdot 2(x + a)}{(x + a)^4}
$$

$$
= \frac{2(x + a)(x^2 + 2ax - a) - 2(x^2 + 2ax - a)(x + a)}{(x + a)^4}
$$

$$
= \frac{2a(a + 1)}{(x + a)^3}
$$

3. $y = 1 + \frac{x}{1} + \frac{x^2}{1 \cdot 2} + \frac{x^3}{1 \cdot 2 \cdot 3} + \frac{x^4}{1 \cdot 2 \cdot 3 \cdot 4}$.

Já conhecemos esta série. Contudo, vamos resolver novamente.

Começamos com a regra da potência para diferenciar cada termo:

$$
\frac{du}{dx} = \frac{d}{dx} \left(1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots \right)
$$

Diferenciando cada termo, obteremos:

$$
\frac{du}{dx} = 0 + 1 + \frac{2x}{2} + \frac{3x^2}{6} + \cdots
$$

Simplificando:

$$
\frac{du}{dx} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots
$$

A função $u$ é a expansão em [Série de Maclaurin](https://en.wikipedia.org/wiki/Taylor_series) da função exponencial $e^x$. Portanto, a derivada de $u$ em relação a $x$ é a própria função:

$$
\frac{du}{dx} = \frac{d}{dx} (e^x) = e^x
$$

ou, escrito na forma da série:

$$
\frac{du}{dx} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots
$$

Para a segunda, terceira, quarta, etc. derivadas, teremos sempre o mesmo resultado.

4. Encontre as funções derivadas de 2ª e 3ª ordem nos Exercícios III, números 1 a 7, e nos Exemplos dados, números 1 a 7.

Considerando as questões do Exercício III,

1. Letra (a): Diferencie $u = 1 + x + \frac{x^2}{1 \cdot 2} + \frac{x^3}{1 \cdot 2 \cdot 3} + \cdots$.

Primeira Derivada:

$$
\frac{du}{dx} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots
$$

Resposta:

Este exercício está resolvido acima no item 3

Letra (b): Diferencie $y = ax^2 + bx + c$.

Primeira Derivada:

$$
\frac{dy}{dx} = 2ax + b
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 2a
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 0
$$

Letra (c): Diferencie $y = (x + a)^2$.

Primeira Derivada:

$$
\frac{dy}{dx} = 2x + 2a
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 2
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 0
$$

Letra (d): Diferencie $y = (x + a)^3$.

Primeira Derivada:

$$
\frac{dy}{dx} = 3x^2 + 6ax + 3a^2
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 6x + 6a
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 6
$$

2. Se $w = at - \frac{1}{2}bt^2$, encontre $\frac{dw}{dt}$.

Primeira Derivada:

$$
\frac{dw}{dt} = a - bt
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2w}{dt^2} = -b
$$

Terceira Derivada:

$$
\frac{d^3w}{dt^3} = 0
$$

3. Encontre o coeficiente diferencial de:

$$
y = (x + \sqrt{-1}) \cdot (x - \sqrt{-1})
$$

Primeira derivada:

$$
\frac{dy}{dx} = \frac{d}{dx} (x^2 + 1) = 2x + 0 = 2x
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 2
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 0
$$

4. Diferencie:

$$
y = (197x - 34x^2) \cdot (7 + 22x - 83x^3)
$$

Primeira derivada:

$$
\frac{dy}{dx} = 14110x^4 - 65404x^3 - 2244x^2 + 8192x + 1379
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 56440x^3 - 196212x^2 - 4488x + 8192
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 169320x^2 - 392424x - 4488
$$

5. Se $x = (y + 3) \cdot (y + 5)$, encontre $\frac{dx}{dy}$.

Primeira derivada:

$$
\frac{dx}{dy} = y + 5 + y + 3 = 2y + 8
$$

Resposta:
Segunda Derivada:

$$
\frac{d^2x}{dy^2} = 2
$$

Terceira Derivada:

$$
\frac{d^3x}{dy^3} = 0
$$

6. Diferencie $y = 1.3709x \cdot (112.6 + 45.202x^2)$.

Primeira derivada:

$$
\frac{dy}{dx} = 185.9022654x^2 + 154.36334
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2y}{dx^2} = 2 \cdot 185.9022654x = 371.8045308x
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = 371.8045308
$$

7. Diferencie $y = \frac{2x + 3}{3x + 2}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{-5}{(3x + 2)^2}
$$

Resposta:

Segunda Derivada:

Usando a Regra do Quociente, diferenciamo novamente:

$$
\frac{d^2y}{dx^2} = \frac{d}{dx} \left( \frac{-5}{(3x + 2)^2} \right) = \frac{30}{(3x + 2)^3}
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{30}{(3x + 2)^3} \right) = \frac{-270}{(3x + 2)^4}
$$

8. Diferencie $y = \frac{1 + x + 2x^2 + 3x^3}{1 + x + 2x^2}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{6x^4 + 6x^3 + 9x^2}{(1 + x + 2x^2)^2}
$$

Segunda Derivada:

Para a segunda derivada, usamos a Regra do Quociente e produto:

$$
\frac{d^2y}{dx^2} = \frac{d}{dx} \left( \frac{6x^4 + 6x^3 + 9x^2}{(1 + x + 2x^2)^2} \right)
$$

Vamos calcular separadamente a derivada do numerador e do denominador:

Numerador:

$$
\frac{d}{dx} (6x^4 + 6x^3 + 9x^2) = 24x^3 + 18x^2 + 18x
$$

Denominador:

$$
\frac{d}{dx} ((1 + x + 2x^2)^2) = 2(1 + x + 2x^2)(1 + 4x)
$$

Então,

$$
\frac{d^2y}{dx^2} = \frac{(24x^3 + 18x^2 + 18x)(1 + x + 2x^2)^2 - (6x^4 + 6x^3 + 9x^2)2(1 + x + 2x^2)(1 + 4x)}{(1 + x + 2x^2)^4}
$$

Terceira Derivada:

A terceira derivada é encontrada diferenciando a segunda derivada, o que envolve o uso novamente das regras do produto e quociente. Devido à complexidade do processo, a expressão completa será:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{N}{D} \right)
$$

Para simplificação, expandimos:

$$
\frac{d^3y}{dx^3} = \frac{D \cdot \frac{dN}{dx} - N \cdot \frac{dD}{dx}}{D^2}
$$

Vamos calcular os termos separadamente:

Derivada do numerador $\frac{dN}{dx}$:

$$
\frac{dN}{dx} = \frac{d}{dx} \left( (1 + x + 2x^2)^2 (24x^3 + 18x^2 + 18x) \right) - \frac{d}{dx} \left( (6x^4 + 6x^3 + 9x^2) \cdot 2(1 + x + 2x^2)(1 + 4x) \right)
$$

Derivada do denominador $\frac{dD}{dx}$:

$$
\frac{dD}{dx} = 4(1 + x + 2x^2)^3 (1 + 4x)
$$

Substituindo essas derivadas na expressão para a terceira derivada:

$$
\frac{d^3y}{dx^3} = \frac{(1 + x + 2x^2)^4 \cdot \frac{dN}{dx} - N \cdot 4(1 + x + 2x^2)^3 (1 + 4x)}{(1 + x + 2x^2)^8}
$$

9. Diferencie $y = \frac{ax + b}{cx + d}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{ad - bc}{(cx + d)^2}
$$

Resposta:

Segunda Derivada:

Usando a Regra do Quociente, diferenciamo novamente:

$$
\frac{d^2y}{dx^2} = \frac{d}{dx} \left( \frac{ad - bc}{(cx + d)^2} \right) = \frac{2(ad - bc)c}{(cx + d)^3}
$$

Terceira Derivada:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{2(ad - bc)c}{(cx + d)^3} \right) = \frac{-6(ad - bc)c^2}{(cx + d)^4}
$$

10. Diferencie $y = \frac{x^n + a}{x^{-n} + b}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{2nx^{-1} + bnx^{n-1} + anx^{-n-1}}{(x^{-n} + b)^2}
$$

Resposta:

Simplificando a primeira derivada:

$$
\frac{dy}{dx} = \frac{nx^{n-1}x^{-n} + bnx^{n-1} + nx^{-n-1}x^n + anx^{-n-1}}{(x^{-n} + b)^2} = \frac{nx^{-1} + bnx^{n-1} + anx^{-n-1}}{(x^{-n} + b)^2}
$$

Segunda derivada:

Para calcular a segunda derivada, utilizamos a Regra do Quociente novamente:

$$
\frac{d^2y}{dx^2} = \frac{(x^{-n} + b)^2 \cdot \frac{d}{dx}(nx^{-1} + bnx^{n-1} + anx^{-n-1}) - (nx^{-1} + bnx^{n-1} + anx^{-n-1}) \cdot \frac{d}{dx}((x^{-n} + b)^2)}{(x^{-n} + b)^4}
$$

Calculando as derivadas individuais:

Derivada do numerador:

$$
\frac{d}{dx}(nx^{-1} + bnx^{n-1} + anx^{-n-1}) = -nx^{-2} + bn(n-1)x^{n-2} - an(-n-1)x^{-n-2}
$$

Derivada do denominador:

$$
\frac{d}{dx}((x^{-n} + b)^2) = 2(x^{-n} + b)(-nx^{-n-1})
$$

Substituindo e simplificando, teremos:

$$
\frac{d^2y}{dx^2} = \frac{(x^{-n} + b)^2 \cdot (-nx^{-2} + bn(n-1)x^{n-2} + annx^{-n-2}) - (nx^{-1} + bnx^{n-1} + anx^{-n-1}) \cdot 2(x^{-n} + b)(-nx^{-n-1})}{(x^{-n} + b)^4}
$$

Terceira derivada:

Para calcular a terceira derivada, utilizamos a Regra do Quociente novamente, diferenciando a segunda derivada:

$$
\frac{d^3y}{dx^3} = \frac{(x^{-n} + b)^4 \cdot \frac{d}{dx} \left[ (x^{-n} + b)^2 \cdot (-nx^{-2} + bn(n-1)x^{n-2} + annx^{-n-2}) - (nx^{-1} + bnx^{n-1} + anx^{-n-1}) \cdot 2(x^{-n} + b)(-nx^{-n-1}) \right] - \left[ (x^{-n} + b)^2 \cdot (-nx^{-2} + bn(n-1)x^{n-2} + annx^{-n-2}) - (nx^{-1} + bnx^{n-1} + anx^{-n-1}) \cdot 2(x^{-n} + b)(-nx^{-n-1}) \right] \cdot 4(x^{-n} + b)^3 \cdot (-nx^{-n-1})}{(x^{-n} + b)^8}
$$

Vamos simplificar a expressão resultante:

Diferenciamos cada termo no numerador:

$$
\frac{d}{dx} \left[ (x^{-n} + b)^2 \cdot (-nx^{-2} + bn(n-1)x^{n-2} + annx^{-n-2}) \right]
$$

$$
\frac{d}{dx} \left[ (nx^{-1} + bnx^{n-1} + anx^{-n-1}) \cdot 2(x^{-n} + b)(-nx^{-n-1}) \right]
$$

Substituímos as derivadas no numerador da terceira derivada.

Simplificamos os termos resultantes.

A terceira derivada completa é extensa, mas a abordagem geral envolve aplicar as regras do produto e do quociente repetidamente até obter a expressão final.

Portanto, a terceira derivada de $y = \frac{x^n + a}{x^{-n} + b}$ é:

$$
\frac{d^3y}{dx^3} = \frac{(x^{-n} + b)^4 \cdot \text{termo diferenciado} - \text{numerador diferenciado} \cdot 4(x^{-n} + b)^3 \cdot (-nx^{-n-1})}{(x^{-n} + b)^8}
$$

11. Encontre a variação da corrente $C$ em relação à temperatura $t$ para a expressão $C = a + bt + ct^2$.

Primeira derivada:

$$
\frac{dC}{dt} = b + 2ct
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2C}{dt^2} = 2c
$$

Terceira Derivada:

$$
\frac{d^3C}{dt^3} = 0
$$

12. Encontre a taxa de variação da resistência em relação à temperatura para as seguintes expressões.

Primeira equação: $R = R_0 (1 + at + bt^2)$

Primeira derivada:

$$
\frac{dR}{dt} = R_0 (a + 2bt)
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2R}{dt^2} = R_0 \cdot 2b = 2bR_0
$$

Terceira Derivada:

$$
\frac{d^3R}{dt^3} = 0
$$

Segunda equação: $R = R_0 (1 + at + b\sqrt{t})$

Primeira derivada:

$$
\frac{dR}{dt} = R_0 \left( a + \frac{b}{2\sqrt{t}} \right)
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2R}{dt^2} = R_0 \left( 0 - \frac{b}{4t^{3/2}} \right) = -\frac{bR_0}{4t^{3/2}}
$$

Terceira Derivada:

$$
\frac{d^3R}{dt^3} = R_0 \left( \frac{3b}{8t^{5/2}} \right) = \frac{3bR_0}{8t^{5/2}}
$$

Terceira equação: $R = R_0 (1 + at + bt^2)^{-1}$

Primeira derivada:

$$
\frac{dR}{dt} = -R_0 \cdot \frac{a + 2bt}{(1 + at + bt^2)^2}
$$

Resposta:

Segunda Derivada:

Para calcular a segunda derivada, utilizamos a Regra do Quociente e produto:

$$
\frac{d^2R}{dt^2} = \frac{d}{dt} \left( -R_0 \cdot \frac{a + 2bt}{(1 + at + bt^2)^2} \right)
$$

Vamos simplificar a derivada:

$$
\frac{d^2R}{dt^2} = -R_0 \cdot \frac{(1 + at + bt^2)^2 \cdot \frac{d}{dt}(a + 2bt) - (a + 2bt) \cdot \frac{d}{dt}((1 + at + bt^2)^2)}{(1 + at + bt^2)^4}
$$

Calculando as derivadas individuais:

Derivada do numerador:

$$
\frac{d}{dt}(a + 2bt) = 2b
$$

Derivada do denominador:

$$
\frac{d}{dt}((1 + at + bt^2)^2) = 2(1 + at + bt^2)(a + 2bt)
$$

Substituindo e simplificando, teremos:

$$
\frac{d^2R}{dt^2} = -R_0 \cdot \frac{(1 + at + bt^2)^2 \cdot 2b - (a + 2bt) \cdot 2(1 + at + bt^2)(a + 2bt)}{(1 + at + bt^2)^4}
$$

Simplificando mais:

$$
\frac{d^2R}{dt^2} = -R_0 \cdot \frac{2b(1 + at + bt^2)^2 - 2(a + 2bt)^2(1 + at + bt^2)}{(1 + at + bt^2)^4}
$$

$$
\frac{d^2R}{dt^2} = -R_0 \cdot \frac{2b(1 + at + bt^2) - 2(a + 2bt)^2}{(1 + at + bt^2)^3}
$$

Terceira Derivada:

Para calcular a terceira derivada, utilizamos a Regra do Quociente e produto novamente, diferenciando a segunda derivada:

$$
\frac{d^3R}{dt^3} = \frac{d}{dt} \left( -R_0 \cdot \frac{2b(1 + at + bt^2) - 2(a + 2bt)^2}{(1 + at + bt^2)^3} \right)
$$

Vamos simplificar a derivada:

$$
\frac{d^3R}{dt^3} = -R_0 \cdot \frac{(1 + at + bt^2)^3 \cdot \frac{d}{dt}(2b(1 + at + bt^2) - 2(a + 2bt)^2) - (2b(1 + at + bt^2) - 2(a + 2bt)^2) \cdot \frac{d}{dt}((1 + at + bt^2)^3)}{(1 + at + bt^2)^6}
$$

Calculando as derivadas individuais:

Derivada do numerador:

$$
\frac{d}{dt}(2b(1 + at + bt^2) - 2(a + 2bt)^2) = 2b(a + 2bt) + 2b(2t)
$$

Derivada do denominador:

$$
\frac{d}{dt}((1 + at + bt^2)^3) = 3(1 + at + bt^2)^2(a + 2bt)
$$

Substituindo e simplificando, teremos:

$$
\frac{d^3R}{dt^3} = -R_0 \cdot \frac{(1 + at + bt^2)^3 \cdot (2b(a + 2bt) + 2b(2t)) - (2b(1 + at + bt^2) - 2(a + 2bt)^2) \cdot 3(1 + at + bt^2)^2(a + 2bt)}{(1 + at + bt^2)^6}
$$

Simplificando mais:

$$
\frac{d^3R}{dt^3} = -R_0 \cdot \frac{(1 + at + bt^2)^3 \cdot 2b(a + 2bt + 2t) - (2b(1 + at + bt^2) - 2(a + 2bt)^2) \cdot 3(1 + at + bt^2)^2(a + 2bt)}{(1 + at + bt^2)^6}
$$

$$
\frac{d^3R}{dt^3} = -R_0 \cdot \frac{2b(1 + at + bt^2)^3(a + 2bt + 2t) - 3(1 + at + bt^2)^2(2b(1 + at + bt^2) - 2(a + 2bt)^2)(a + 2bt)}{(1 + at + bt^2)^6}
$$

Portanto, a terceira derivada de $R = R_0 (1 + at + bt^2)^{-1}$ é:

$$
\frac{d^3R}{dt^3} = -R_0 \cdot \frac{2b(1 + at + bt^2)^3(a + 2bt + 2t) - 3(1 + at + bt^2)^2(2b(1 + at + bt^2) - 2(a + 2bt)^2)(a + 2bt)}{(1 + at + bt^2)^6}
$$

13. A força eletromotriz $E$ varia com a temperatura $t$ de acordo com:

Primeira derivada:

$$
\frac{dE}{dt} = 1.4340 \left[ -0.000814 + 0.000014(t - 15) \right]
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2E}{dt^2} = 1.4340 \cdot 0.000014
$$

Terceira Derivada:

$$
\frac{d^3E}{dt^3} = 0
$$

14. A força eletromotriz $E$ para manter um arco elétrico de comprimento $l$ com corrente $i$ é dada por:

$$
E = a + bl + \frac{c + kl}{i}
$$

Parte 1: Variação da força eletromotriz com relação ao comprimento do arco $l$

Primeira derivada:

$$
\frac{dE}{dl} = b + \frac{k}{i}
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2E}{dl^2} = 0
$$

Terceira Derivada:

$$
\frac{d^3E}{dl^3} = 0
$$

Parte 2: Variação da força eletromotriz com relação à corrente $i$

Primeira derivada:

$$
\frac{dE}{di} = -\frac{c + kl}{i^2}
$$

Resposta:

Segunda Derivada:

Usando a Regra do Quociente e da potência:

$$
\frac{d^2E}{di^2} = \frac{d}{di} \left( -\frac{c + kl}{i^2} \right)
$$

$$
\frac{d^2E}{di^2} = - (c + kl) \cdot \frac{d}{di} \left( i^{-2} \right)
$$

$$
\frac{d^2E}{di^2} = - (c + kl) \cdot (-2) i^{-3}
$$

$$
\frac{d^2E}{di^2} = 2 \frac{c + kl}{i^3}
$$

Terceira Derivada:

Usando a Regra do Quociente e da potência novamente:

$$
\frac{d^3E}{di^3} = \frac{d}{di} \left( 2 \frac{c + kl}{i^3} \right)
$$

$$
\frac{d^3E}{di^3} = 2 (c + kl) \cdot \frac{d}{di} \left( i^{-3} \right)
$$

$$
\frac{d^3E}{di^3} = 2 (c + kl) \cdot (-3) i^{-4}
$$

$$
\frac{d^3E}{di^3} = -6 \frac{c + kl}{i^4}
$$

Portanto, as derivadas da expressão $E = a + bl + \frac{c + kl}{i}$ são:

Com relação ao comprimento do arco $l$:

Primeira Derivada:

$$
\frac{dE}{dl} = b + \frac{k}{i}
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2E}{dl^2} = 0
$$

Terceira Derivada:

$$
\frac{d^3E}{dl^3} = 0
$$

Com relação à corrente $i$:

Primeira Derivada:

$$
\frac{dE}{di} = -\frac{c + kl}{i^2}
$$

Resposta:

Segunda Derivada:

$$
\frac{d^2E}{di^2} = 2 \frac{c + kl}{i^3}
$$

Terceira Derivada:

$$
\frac{d^3E}{di^3} = -6 \frac{c + kl}{i^4}
$$

Ainda no item 4 do Exercício IV, precisamos calcular a segunda e terceiras derivadas das respostas dos exemplos do Capítulo 6.

1. Diferenciar $y = \frac{a}{b^2}x^3 - \frac{a^2}{b}x + \frac{a^2}{b^2}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{3a}{b^2} x^2 - \frac{a^2}{b}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = \frac{6a}{b^2} x
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = \frac{6a}{b^2}
$$

2. Diferenciar $y = 2a\sqrt{bx^3} - \frac{3b\sqrt{a}}{x} - 2\sqrt{ab}$.

Primeira derivada:

$$
\frac{dy}{dx} = 3a\sqrt{bx} + \frac{3b\sqrt{a}}{x^2}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = \frac{3a\sqrt{b}}{2\sqrt{x}} - \frac{6b\sqrt{a}}{x^3}
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = -\frac{3a\sqrt{b}}{4x^{3/2}} + \frac{18b\sqrt{a}}{x^4}
$$

3. Diferenciar $z = 1.8 \sqrt[3]{\frac{1}{\theta^2}} - \frac{4.4}{\sqrt[5]{\theta}} - 27^\circ$.

Primeira derivada:

$$
\frac{dz}{d\theta} = -1.2 \theta^{-\frac{5}{3}} + 0.88 \theta^{-\frac{6}{5}}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2z}{d\theta^2} = 2.0 \theta^{-\frac{8}{3}} - 1.056 \theta^{-\frac{11}{5}}
$$

Terceira derivada:

$$
\frac{d^3z}{d\theta^3} = -\frac{20}{3} \theta^{-\frac{11}{3}} + \frac{11.616}{5} \theta^{-\frac{16}{5}}
$$

4. Diferenciar $v = (3t^2 - 1.2t + 1)^3$.

Primeira derivada:

$$
\frac{dv}{dt} = 162t^5 - 162t^4 + 159.84t^3 - 69.984t^2 + 26.64t - 3.6
$$

Resposta:

Segunda derivada:

$$
\frac{d^2v}{dt^2} = 810t^4 - 648t^3 + 479.52t^2 - 139.968t + 26.64
$$

Terceira derivada:

$$
\frac{d^3v}{dt^3} = 3240t^3 - 1944t^2 + 959.04t - 139.968
$$

5. Diferenciar $y = (2x - 3)(x + 1)^2$.

Primeira derivada:

$$
\frac{dy}{dx} = 2(x + 1)(3x - 2)
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = 6x + 2
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = 6
$$

6. Diferenciar $y = 0.5x^3(x - 3)$.

Primeira derivada:

$$
\frac{dy}{dx} = 2x^3 - 4.5x^2
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = 6x^2 - 9x
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = 12x - 9
$$

7. Diferenciar $w = \left( \theta + \frac{1}{\theta} \right) \left( \sqrt{\theta} + \frac{1}{\sqrt{\theta}} \right)$.

Primeira derivada:

$$
\frac{dw}{d\theta} = \frac{3}{2} \left( \sqrt{\theta} - \frac{1}{\sqrt[5]{\theta}} \right) + \frac{1}{2} \left( \frac{1}{\sqrt{\theta}} - \frac{1}{\sqrt[3]{\theta}} \right)
$$

Resposta:

Segunda derivada:

$$
\frac{d^2w}{d\theta^2} = \frac{3}{4} \theta^{-\frac{1}{2}} + \frac{3}{10} \theta^{-\frac{6}{5}} - \frac{1}{4} \theta^{-\frac{3}{2}} + \frac{1}{6} \theta^{-\frac{4}{3}}
$$

Terceira derivada:

$$
\frac{d^3w}{d\theta^3} = -\frac{3}{8} \theta^{-\frac{3}{2}} - \frac{18}{50} \theta^{-\frac{11}{5}} + \frac{3}{8} \theta^{-\frac{5}{2}} - \frac{4}{18} \theta^{-\frac{7}{3}}
$$

8. Diferenciar $y = \frac{a}{1 + a\sqrt{x} + a^2 x}$.

Primeira derivada:

$$
\frac{dy}{dx} = -\frac{a \left(\frac{1}{2} ax^{-\frac{1}{2}} + a^2 \right)}{(1 + a\sqrt{x} + a^2 x)^2}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = \frac{a \left( \frac{1}{4} a^2 x^{-3/2} \right) (1 + a\sqrt{x} + a^2 x)^2 + 2a (\frac{1}{2} a x^{-1/2} + a^2) (1 + a\sqrt{x} + a^2 x) a \left( x^{-1/2} + 2 \right)}{(1 + a\sqrt{x} + a^2 x)^4}
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{d^2y}{dx^2} \right)
$$

9. Diferenciar $y = \frac{x^2}{x^2 + 1}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{2x}{(x^2 + 1)^2}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = \frac{2(x^2 + 1)^2 \cdot 2x' - 2x \cdot 2(x^2 + 1) \cdot 2x'}{(x^2 + 1)^4} = \frac{4(x^2 + 1) - 8x^2}{(x^2 + 1)^3}
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{4(x^2 + 1) - 8x^2}{(x^2 + 1)^3} \right)
$$

10. Diferenciar $y = \frac{a + \sqrt{x}}{a - \sqrt{x}}$.

Primeira derivada:

$$
\frac{dy}{dx} = \frac{a}{(a - \sqrt{x})^2 \sqrt{x}}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2y}{dx^2} = \frac{d}{dx} \left( \frac{a}{(a - \sqrt{x})^2 \sqrt{x}} \right)
$$

Terceira derivada:

$$
\frac{d^3y}{dx^3} = \frac{d}{dx} \left( \frac{d^2y}{dx^2} \right)
$$

11. Diferenciar $\theta = \frac{1 - a\sqrt[3]{t^2}}{1 + a\sqrt[3]{t^3}}$.

Primeira derivada:

$$
\frac{d\theta}{dt} = \frac{(1 + a t^{\frac{2}{3}}) \left(-\frac{2}{3} a t^{-\frac{1}{3}}\right) - (1 - a t^{\frac{2}{3}}) \left(\frac{3}{2} a t^{\frac{1}{2}}\right)}{(1 + a t^{\frac{3}{2}})^2}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2\theta}{dt^2} = \frac{d}{dt} \left( \frac{(1 + a t^{\frac{2}{3}}) \left(-\frac{2}{3} a t^{-\frac{1}{3}}\right) - (1 - a t^{\frac{2}{3}}) \left(\frac{3}{2} a t^{\frac{1}{2}}\right)}{(1 + a t^{\frac{3}{2}})^2} \right)
$$

Terceira derivada:

$$
\frac{d^3\theta}{dt^3} = \frac{d}{dt} \left( \frac{d^2\theta}{dt^2} \right)
$$

12. Um reservatório de seção transversal quadrada tem lados inclinados em um ângulo de $45^\circ$ com a vertical. O lado da base é $200\, \text{m}$. Encontre uma expressão para a quantidade que entra ou sai quando a profundidade da água varia em $1\, \text{m}$; portanto, encontre, em litros, a quantidade retirada por hora quando a profundidade é reduzida de $14$ para $10\, \text{m}$ em 24 horas.

Primeira derivada:

$$
\frac{dV}{dh} = 40,000 + 1600h + 4h^2
$$

Resposta:

Segunda derivada:

$$
\frac{d^2V}{dh^2} = 1600 + 8h
$$

Terceira derivada:

$$
\frac{d^3V}{dh^3} = 8
$$

13. A pressão absoluta, em atmosferas, $P$, do vapor saturado na temperatura $t^\circ \, \text{C}$. foi determinada por Dulong como sendo $P = \left( \frac{40 + t}{140} \right)^5$ desde que $t$ esteja acima de $80^\circ \, \text{C}$. Encontre a taxa de variação da pressão com a temperatura a $100^\circ \, \text{C}$.

Primeira derivada:

$$
\frac{dP}{dt} = \frac{5(40 + t)^4}{140^5}
$$

Resposta:

Segunda derivada:

$$
\frac{d^2P}{dt^2} = \frac{20(40 + t)^3}{140^5}
$$

Terceira derivada:

$$
\frac{d^3P}{dt^3} = \frac{60(40 + t)^2}{140^5}
$$

### Exercícios V

1. Se $y = a + bt^2 + ct^4$, encontre $\frac{dy}{dt}$ e $\frac{d^2 y}{dt^2}$.

Resolvendo:

Usando a regra da potência, da constante e da soma:

$$
\frac{dy}{dt} = \frac{d}{dt}(a) + \frac{d}{dt}(bt^2) + \frac{d}{dt}(ct^4)
$$

$$
\frac{dy}{dt} = 0 + 2bt + 4ct^3 = 2bt + 4ct^3
$$

Para a segunda derivada:

$$
\frac{d^2 y}{dt^2} = \frac{d}{dt}(2bt + 4ct^3)
$$

$$
\frac{d^2 y}{dt^2} = 2b + 12ct^2
$$

Resposta:

$$
\frac{dy}{dt} = 2bt + 4ct^3; \quad \frac{d^2 y}{dt^2} = 2b + 12ct^2
$$

2. Um corpo caindo livremente no espaço descreve em $t$ segundos um espaço $s$, em metros, expresso pela equação $s = 16t^2$. Desenhe uma curva mostrando a relação entre $s$ e $t$. Também determine a velocidade do corpo nos seguintes tempos a partir do seu ponto de partida: $t = 2$ segundos; $t = 4.6$ segundos; $t = 0.01$ segundo.

Resolvendo

Usando a regra da potência e da constante:

$$
v = \frac{ds}{dt} = \frac{d}{dt}(16t^2) = 32t
$$

Para os tempos dados:

$$
v(t=2) = 32 \cdot 2 = 64 \text{ m/s}
$$

$$
v(t=4.6) = 32 \cdot 4.6 = 147.2 \text{ m/s}
$$

$$
v(t=0.01) = 32 \cdot 0.01 = 0.32 \text{ m/s}
$$

Resposta:

$64$; $147.2$; e $0.32$ metros por segundo.

3. Se $x = at - \frac{1}{2}gt^2$, encontre $\dot{x}$ e $\ddot{x}$.

Resolvendo:

Usando a regra da constante que multiplica a função, da soma e da subtração:

$$
\dot{x} = \frac{d}{dt}(at) - \frac{d}{dt}\left(\frac{1}{2}gt^2\right)
$$

$$
\dot{x} = a - gt
$$

Para a segunda derivada:

$$
\ddot{x} = \frac{d}{dt}(a - gt) = -g
$$

Resposta:

$x = a - gt; \quad \ddot{x} = -g$

4. Se um corpo se move de acordo com a lei $s = 12 - 4.5t + 6.2t^2$, encontre sua velocidade quando $t = 4$ segundos; $s$ sendo em metros.

Resolvendo:

Usando a regra da potência, da constante, da soma e da subtração:

$$
v = \frac{ds}{dt} = \frac{d}{dt}(12 - 4.5t + 6.2t^2)
$$

$$
v = 0 - 4.5 + 12.4t
$$

Para $t = 4$:

$$
v(t=4) = -4.5 + 12.4 \cdot 4 = -4.5 + 49.6 = 45.1 \text{ m/s}
$$

Resposta:

$45.1$ metros por segundo.

5. Encontre a aceleração do corpo mencionado no exemplo anterior. A aceleração é a mesma para todos os valores de $t$?

Resolvendo:

Usando a regra da potência, da constante e da soma:

$$
a = \frac{dv}{dt} = \frac{d}{dt}(-4.5 + 12.4t) = 12.4
$$

A aceleração é constante.

Resposta:

$12.4$ metros por segundo por segundo. Sim.

6. O ângulo $\theta$ (em radianos) percorrido por uma roda giratória está relacionado com o tempo $t$ (em segundos) decorrido desde o início; pela lei $\theta = 2.1 - 3.2t + 4.8t^2$. Encontre a velocidade angular (em radianos por segundo) dessa roda quando $1 \frac{1}{2}$ segundos se passaram. Encontre também sua aceleração angular.

Resolvendo:

Usando a regra da potência, da constante e da soma:

$$
\omega = \frac{d\theta}{dt} = \frac{d}{dt}(2.1 - 3.2t + 4.8t^2)
$$

$$
\omega = 0 - 3.2 + 9.6t
$$

Para $t = 1.5$:

$$
\omega(t=1.5) = -3.2 + 9.6 \cdot 1.5 = -3.2 + 14.4 = 11.2 \text{ rad/s}
$$

Para a aceleração angular:

$$
\alpha = \frac{d\omega}{dt} = \frac{d}{dt}(-3.2 + 9.6t) = 9.6
$$

Resposta:

Velocidade angular $= 11.2$ radianos por segundo; aceleração angular $= 9.6$ radianos por segundo ao quadrado.

7. Um corpo deslizante se move de tal forma que, durante a primeira parte de seu movimento, sua distância $s$ em metros do ponto de partida é dada pela expressão $s = 6.8t^3 - 10.8t$; $t$ em segundos. Encontre a expressão para a velocidade e a aceleração a qualquer momento; e, portanto, encontre a velocidade e a aceleração após 3 segundos.

Resolvendo:

Usando a regra da potência, da constante e da soma:

$$
v = \frac{ds}{dt} = \frac{d}{dt}(6.8t^3 - 10.8t)
$$

$$
v = 20.4t^2 - 10.8
$$

Para $t = 3$:

$$
v(t=3) = 20.4 \cdot 3^2 - 10.8 = 20.4 \cdot 9 - 10.8 = 183.6 - 10.8 = 172.8 \text{ cm/s}
$$

Para a aceleração:

$$
a = \frac{dv}{dt} = \frac{d}{dt}(20.4t^2 - 10.8) = 40.8t
$$

Para $t = 3$:

$$
a(t=3) = 40.8 \cdot 3 = 122.4 \text{ cm/s}^2
$$

Resposta:

$v = 20.4t^2 - 10.8$. $a = 40.8t$. $172.8$ cm/s, $122.4$ cm/s².

8. O movimento de um balão ascendente é tal que sua altura $h$, em quilômetros, é dada a qualquer instante pela expressão $h = 0.5 + \frac{1}{10} \sqrt[3]{t - 125}$; $t$ sendo em segundos. Encontre uma expressão para a velocidade e a aceleração a qualquer momento. Desenhe curvas para mostrar a variação da altura, velocidade e aceleração durante os primeiros dez minutos da ascensão.

Resolvendo:

Para a velocidade, usamos a regra da potência e da constante que multiplica função:

$$
v = \frac{dh}{dt} = \frac{d}{dt} \left(0.5 + \frac{1}{10} \sqrt[3]{t - 125}\right)
$$

$$
v = \frac{1}{10} \cdot \frac{d}{dt} \left((t - 125)^{1/3}\right)
$$

$$
v = \frac{1}{10} \cdot \frac{1}{3} (t - 125)^{-2/3} \cdot 1
$$

$$
v = \frac{1}{30} (t - 125)^{-2/3}
$$

Para a aceleração:

$$
a = \frac{dv}{dt} = \frac{d}{dt} \left(\frac{1}{30} (t - 125)^{-2/3}\\right)
$$

$$
a = \frac{1}{30} \cdot \frac{d}{dt} \left((t - 125)^{-2/3}\right)
$$

$$
a = \frac{1}{30} \cdot \left(-\frac{2}{3}\right) (t - 125)^{-5/3}
$$

$$
a = -\frac{1}{45} (t - 125)^{-5/3}
$$

Resposta:

$$
v = \frac{1}{30 \sqrt[3]{(t - 125)^2}}, \quad a = -\frac{1}{45 \sqrt[5]{(t - 125)^5}}
$$

9. Uma pedra é lançada para baixo na água e sua profundidade $p$ em metros em qualquer instante $t$ segundos após atingir a superfície da água é dada pela expressão

$$
p = \frac{4}{4 + t^2} + 0.8t - 1
$$

Encontre uma expressão para a velocidade e a aceleração a qualquer momento. Encontre a velocidade e a aceleração após 10 segundos.

Resolvendo:

Para a velocidade:

$$
v = \frac{dp}{dt} = \frac{d}{dt} \left( \frac{4}{4 + t^2} + 0.8t - 1 \right)
$$

Usando a Regra do Quociente e da soma:

$$
v = \frac{d}{dt} \left( \frac{4}{4 + t^2} \right) + \frac{d}{dt} (0.8t) - \frac{d}{dt} (1)
$$

$$
v = \frac{-8t}{(4 + t^2)^2} + 0.8 - 0
$$

$$
v = 0.8 - \frac{8t}{(4 + t^2)^2}
$$

Para a aceleração:

$$
a = \frac{dv}{dt} = \frac{d}{dt} \left( 0.8 - \frac{8t}{(4 + t^2)^2} \right)
$$

$$
a = 0 - \frac{d}{dt} \left( \frac{8t}{(4 + t^2)^2} \right)
$$

Usando a Regra do Quociente:

$$
a = - \left( \frac{(4 + t^2)^2 \cdot 8 - 8t \cdot 2(4 + t^2) \cdot t}{(4 + t^2)^4} \right)
$$

$$
a = - \left( \frac{8(4 + t^2)^2 - 16t^2 (4 + t^2)}{(4 + t^2)^4} \right)
$$

$$
a = - \left( \frac{32 + 8t^2 - 16t^2 - 16t^4}{(4 + t^2)^3} \right)
$$

$$
a = - \frac{24t^2 - 16t^4}{(4 + t^2)^3}
$$

$$
a = \frac{24t^2 - 32}{(4 + t^2)^3}
$$

Para $t = 10$:

$$
v(t=10) = 0.8 - \frac{8 \cdot 10}{(4 + 10^2)^2} = 0.8 - \frac{80}{104^2} \approx 0.7926 \text{ m/s}
$$

$$
a(t=10) = \frac{24 \cdot 10^2 - 32}{(4 + 10^2)^3} = \frac{2400 - 32}{(4 + 100)^3} \approx 0.00211 \text{ m/s}^2
$$

Resposta:

$$
v = 0.8 - \frac{8t}{(4 + t^2)^2}, \quad a = \frac{24t^2 - 32}{(4 + t^2)^3}, \quad 0.7926 \text{ m/s e } 0.00211 \text{ m/s}^2
$$

10. Um corpo se move de tal forma que os espaços descritos no tempo $t$ a partir da partida são dados por $s = t^n$, onde $n$ é uma constante. Encontre o valor de $n$ quando a velocidade é dobrada do quinto ao décimo segundo; encontre também quando a velocidade é numericamente igual à aceleração ao final do décimo segundo.

Resolvendo:

Para a velocidade:

$$
v = \frac{ds}{dt} = \frac{d}{dt}(t^n) = nt^{n-1}
$$

Para a aceleração:

$$
a = \frac{dv}{dt} = \frac{d}{dt}(nt^{n-1}) = n(n-1)t^{n-2}
$$

Quando a velocidade é dobrada do quinto ao décimo segundo:

$$
v(10) = 2v(5)
$$

$$
nt^{n-1} \bigg|_{t=10} = 2 \cdot nt^{n-1} \bigg|_{t=5}
$$

$$
n \cdot 10^{n-1} = 2n \cdot 5^{n-1}
$$

$$
10^{n-1} = 2 \cdot 5^{n-1}
$$

$$
\left(\frac{10}{5}\right)^{n-1} = 2
$$

$$
2^{n-1} = 2
$$

$$
n-1 = 1
$$

$$
n = 2
$$

Quando a velocidade é numericamente igual à aceleração ao final do décimo segundo:

$$
nt^{n-1} = n(n-1)t^{n-2}
$$

$$
t^{n-1} = (n-1)t^{n-2}
$$

$$
t \big|_{t=10} = n-1
$$

$$
10 = n-1
$$

$$
n = 11
$$

Resposta:

$n = 2, \quad n = 11$

### Exercícios VI

Derive as seguintes funções:

1. $y = \sqrt{x^2 + 1}$

Vamos aplicar a regra da cadeia.

Primeiro, reescrevemos a função:

$$y = (x^2 + 1)^{1/2} $$

Derivando em relação a $x$:

$$\frac{dy}{dx} = \frac{1}{2}(x^2 + 1)^{-1/2} \cdot \frac{d}{dx}(x^2 + 1) $$

A derivada de $x^2 + 1$ é $2x$, então:

$$\frac{dy}{dx} = \frac{1}{2}(x^2 + 1)^{-1/2} \cdot 2x = \frac{x}{\sqrt{x^2 + 1}} $$

2. $y = \sqrt{x^2 + a^2}$

Aplicando a regra da cadeia novamente.

Reescrevendo a função:

$$y = (x^2 + a^2)^{1/2} $$

Derivando em relação a $x$:

$$\frac{dy}{dx} = \frac{1}{2}(x^2 + a^2)^{-1/2} \cdot \frac{d}{dx}(x^2 + a^2) $$

A derivada de $x^2 + a^2$ é $2x$, então:

$$\frac{dy}{dx} = \frac{1}{2}(x^2 + a^2)^{-1/2} \cdot 2x = \frac{x}{\sqrt{x^2 + a^2}} $$

3. $y = \frac{1}{\sqrt{a + x}}$

Aplicando a regra da cadeia e do quociente.

Reescrevendo a função:

$$y = (a + x)^{-1/2} $$

Derivando em relação a $x$:

$$\frac{dy}{dx} = -\frac{1}{2}(a + x)^{-3/2} \cdot \frac{d}{dx}(a + x) $$

A derivada de $a + x$ é $1$, então:

$$\frac{dy}{dx} = -\frac{1}{2}(a + x)^{-3/2} = -\frac{1}{2(a + x)^{3/2}} $$

4. $y = \frac{a}{\sqrt{a - x^2}}$

Aplicando a regra da cadeia e do quociente.

Reescrevendo a função:

$$y = a(a - x^2)^{-1/2} $$

Derivando em relação a $x$:

$$\frac{dy}{dx} = a \cdot \left( -\frac{1}{2}(a - x^2)^{-3/2} \cdot \frac{d}{dx}(a - x^2) \right) $$

A derivada de $a - x^2$ é $-2x$, então:

$$\frac{dy}{dx} = a \cdot \left( -\frac{1}{2}(a - x^2)^{-3/2} \cdot (-2x) \right) = \frac{ax}{(a - x^2)^{3/2}} $$

5. $y = \frac{\sqrt{x^2 - a^2}}{x^2}$

Aplicando a regra da cadeia e do quociente.

Reescrevendo a função:

$$y = \frac{(x^2 - a^2)^{1/2}}{x^2} $$

Usando a regra do quociente:

$$\frac{dy}{dx} = \frac{ \left( \frac{1}{2}(x^2 - a^2)^{-1/2} \cdot \frac{d}{dx}(x^2 - a^2) \cdot x^2 - (x^2 - a^2)^{1/2} \cdot \frac{d}{dx}(x^2) \right) }{ (x^2)^2 } $$

A derivada de $x^2 - a^2$ é $2x$ e a derivada de $x^2$ é $2x$, então:

$$\frac{dy}{dx} = \frac{ \left( \frac{1}{2}(x^2 - a^2)^{-1/2} \cdot 2x \cdot x^2 - (x^2 - a^2)^{1/2} \cdot 2x \right) }{ x^4 } $$

Simplificando:

$$\frac{dy}{dx} = \frac{ x(x^2) - 2x(x^2 - a^2)^{1/2} }{ x^4 \cdot (x^2 - a^2)^{1/2} } = \frac{x^3 - 2x(x^2 - a^2)^{1/2}}{x^4 \cdot (x^2 - a^2)^{1/2}} $$

Simplificando ainda mais:

$$\frac{dy}{dx} = \frac{x^3 - 2x(x^2 - a^2)^{1/2}}{x^4 \cdot (x^2 - a^2)^{1/2}} = \frac{x - 2(x^2 - a^2)^{1/2}}{x^3 \cdot (x^2 - a^2)^{1/2}} $$

$$\frac{dy}{dx} = \frac{a^2}{x^3(x^2 - a^2)^{1/2}}$$

6. $y = \frac{\sqrt[3]{x^4 + a}}{\sqrt[3]{x^3 + a}}$

Aplicando a regra da cadeia e do quociente.

Reescrevendo a função:

$$y = \frac{(x^4 + a)^{1/3}}{(x^3 + a)^{1/3}} $$

Usando a regra do quociente:

$$\frac{dy}{dx} = \frac{ (x^3 + a)^{1/3} \cdot \frac{d}{dx}(x^4 + a)^{1/3} - (x^4 + a)^{1/3} \cdot \frac{d}{dx}(x^3 + a)^{1/3} }{ (x^3 + a)^{2/3} } $$

Derivadas:

$$\frac{d}{dx}(x^4 + a)^{1/3} = \frac{1}{3}(x^4 + a)^{-2/3} \cdot 4x^3 $$

$$\frac{d}{dx}(x^3 + a)^{1/3} = \frac{1}{3}(x^3 + a)^{-2/3} \cdot 3x^2 $$

Substituindo as derivadas:

$$\frac{dy}{dx} = \frac{ (x^3 + a)^{1/3} \cdot \frac{1}{3}(x^4 + a)^{-2/3} \cdot 4x^3 - (x^4 + a)^{1/3} \cdot \frac{1}{3}(x^3 + a)^{-2/3} \cdot 3x^2 }{ (x^3 + a)^{2/3} } $$

Simplificando:

$$\frac{dy}{dx} = \frac{ (x^3 + a)^{1/3} \cdot 4x^3 - (x^4 + a)^{1/3} \cdot 3x^2 }{ 3(x^4 + a)^{2/3} \cdot (x^3 + a)^{2/3} } $$

7. $y = \frac{a^2 + x^2}{(a + x)^2}$

Aplicando a regra do quociente.

Derivando em relação a $x$:

$$\frac{dy}{dx} = \frac{ (a + x)^2 \cdot \frac{d}{dx}(a^2 + x^2) - (a^2 + x^2) \cdot \frac{d}{dx}(a + x)^2 }{ ((a + x)^2)^2 } $$

Derivadas:

$$\frac{d}{dx}(a^2 + x^2) = 2x $$

$$\frac{d}{dx}(a + x)^2 = 2(a + x) $$

Substituindo as derivadas:

$$\frac{dy}{dx} = \frac{ (a + x)^2 \cdot 2x - (a^2 + x^2) \cdot 2(a + x) }{ (a + x)^4 } $$

Simplificando:

$$\frac{dy}{dx} = \frac{ 2x(a + x)^2 - 2(a^2 + x^2)(a + x) }{ (a + x)^4 } $$

$$\frac{dy}{dx} = \frac{ 2x(a + x) - 2(a^2 + x^2) }{ (a + x)^3 } $$

$$\frac{dy}{dx} = \frac{2x(a + x) - 2(a^2 + x^2)}{(a + x)^3} = \frac{2ax + 2x^2 - 2a^2 - 2x^2}{(a + x)^3} = -\frac{2a(a - x)}{(a + x)^3}$$

8. Diferencie $y^5$ em relação a $y^2$

Aqui, consideramos $z = y^5$ e $w = y^2$.

Derivada implícita:

$$\frac{dz}{dw} = \frac{dz}{dy} \cdot \frac{dy}{dw} $$

Derivando:

$$\frac{dz}{dy} = 5y^4 $$

$$\frac{dy}{dw} = \frac{1}{2y} $$

Substituindo:

$$\frac{dz}{dw} = 5y^4 \cdot \frac{1}{2y} = \frac{5y^3}{2} $$

9. $y = \frac{\sqrt{1 - \theta^2}}{1 - \theta}$

Aplicando a regra da cadeia e do quociente.

Reescrevendo a função:

$$y = \frac{(1 - \theta^2)^{1/2}}{1 - \theta} $$

Derivando em relação a $\theta$:

$$\frac{dy}{d\theta} = \frac{ (1 - \theta) \cdot \frac{d}{d\theta}(1 - \theta^2)^{1/2} - (1 - \theta^2)^{1/2} \cdot \frac{d}{d\theta}(1 - \theta) }{ (1 - \theta)^2 } $$

Derivadas:

$$\frac{d}{d\theta}(1 - \theta^2)^{1/2} = \frac{1}{2}(1 - \theta^2)^{-1/2} \cdot (-2\theta) = \frac{d}{d\theta}(1 - \theta^2)^{1/2} = -\frac{\theta}{\sqrt{1 - \theta^2}} $$

$$\frac{d}{d\theta}(1 - \theta) = -1 $$

Substituindo as derivadas:

$$\frac{dy}{d\theta} = \frac{ (1 - \theta) \cdot \left( -\frac{\theta}{(1 - \theta^2)^{1/2}} \right) - (1 - \theta^2)^{1/2} \cdot (-1) }{ (1 - \theta)^2 } $$

Simplificando:

$$\frac{dy}{d\theta} = \frac{ -(1 - \theta) \cdot \frac{\theta}{(1 - \theta^2)^{1/2}} + (1 - \theta^2)^{1/2} }{ (1 - \theta)^2 } $$

$$\frac{dy}{d\theta} = \frac{ \theta(1 - \theta) + (1 - \theta^2) }{ (1 - \theta)^2 \cdot (1 - \theta^2)^{1/2} } $$

$$\frac{dy}{d\theta} = \frac{-\theta(1 - \theta)\sqrt{1 - \theta^2} + (1 - \theta^2)}{(1 - \theta)^2 (1 - \theta^2)} = \frac{1}{(1 - \theta)\sqrt{1 - \theta^2}}$$

### Exercícios VII

1. Dado:

$$
u = \frac{1}{2}x^3; \quad v = 3(u + u^2); \quad w = \frac{1}{v^2}
$$

Encontre $\frac{dw}{dx}$:

Primeiro, encontramos $\frac{du}{dx}$, $\frac{dv}{du}$ e $\frac{dw}{dv}$:

$$
\frac{du}{dx} = \frac{d}{dx}\left(\frac{1}{2}x^3\right) = \frac{3}{2}x^2
$$

$$
\frac{dv}{du} = \frac{d}{du}\left(3(u + u^2)\right) = 3(1 + 2u) = 3\left(1 + 2\left(\frac{1}{2}x^3\right)\right) = 3(1 + x^3)
$$

$$
\frac{dw}{dv} = \frac{d}{dv}\left(\frac{1}{v^2}\right) = -\frac{2}{v^3}
$$

Agora, utilizando a regra da cadeia:

$$
\frac{dw}{dx} = \frac{dw}{dv} \cdot \frac{dv}{du} \cdot \frac{du}{dx} = -\frac{2}{v^3} \cdot 3(1 + x^3) \cdot \frac{3}{2}x^2
$$

Simplificando:

$$
\frac{dw}{dx} = -\frac{3x^2 (3 + 3x^3)}{v^3}
$$

Como $v = 3\left(\frac{1}{2}x^3 + \left(\frac{1}{2}x^3\right)^2\right) = 3\left(\frac{1}{2}x^3 + \frac{1}{4}x^6\right)$, teremos:

$$
\frac{dw}{dx} = -\frac{3x^2 (3 + 3x^3)}{27\left(\frac{1}{2}x^3 + \frac{1}{4}x^6\right)^3}
$$

Simplificando novamente, teremos:

$$\frac{1}{2}x^3 + \frac{1}{4}x^6 = \frac{1}{4}x^3(2 + x^3)$$

Elevar o denominador ao cubo:

$$\left(\frac{1}{4}x^3(2 + x^3)\right)^3 = \frac{1}{64}x^9(2 + x^3)^3$$

Simplificar o numerador:

$$3x^2 (3 + 3x^3) = 9x^2(1 + x^3)$$

Substituir as simplificações na expressão original:

$$-\frac{9x^2(1 + x^3)}{27 * \frac{1}{64}x^9(2 + x^3)^3}$$

Simplificar a expressão resultante:

$$-\frac{8(1 + x^3)}{9x^4(1 + x^3/2)^3}$$

2. Dado:

$$
y = 3x^2 + \sqrt{2}; \quad z = \sqrt{1 + y}; \quad v = \frac{1}{\sqrt{3} + 4z}
$$

Encontre $\frac{dv}{dx}$:\*\*

Primeiro, encontramos $\frac{dy}{dx}$, $\frac{dz}{dy}$ e $\frac{dv}{dz}$:

$$
\frac{dy}{dx} = \frac{d}{dx}\left(3x^2 + \sqrt{2}\right) = 6x
$$

$$
\frac{dz}{dy} = \frac{d}{dy}\left(\sqrt{1 + y}\right) = \frac{1}{2\sqrt{1 + y}}
$$

$$
\frac{dv}{dz} = \frac{d}{dz}\left(\frac{1}{\sqrt{3} + 4z}\right) = -\frac{4}{(\sqrt{3} + 4z)^2}
$$

Agora, utilizando a regra da cadeia:

$$
\frac{dv}{dx} = \frac{dv}{dz} \cdot \frac{dz}{dy} \cdot \frac{dy}{dx} = -\frac{4}{(\sqrt{3} + 4z)^2} \cdot \frac{1}{2\sqrt{1 + y}} \cdot 6x
$$

Simplificando:

$$
\frac{dv}{dx} = -\frac{12x}{(\sqrt{3} + 4z)^2 \sqrt{1 + y}}
$$

Como $y = 3x^2 + \sqrt{2}$ e $z = \sqrt{1 + y} = \sqrt{1 + 3x^2 + \sqrt{2}}$, teremos:

$$
\frac{dv}{dx} = \frac{12x}{\sqrt{1 + \sqrt{2} + 3x^2} \left( \sqrt{3 + 4\sqrt{1 + \sqrt{2} + 3x^2}} \right)^2}
$$

3. Dado:

$$
y = \frac{x^3}{\sqrt{3}}; \quad z = (1 + y)^2; \quad u = \frac{1}{\sqrt{1 + z}}
$$

Encontre $\frac{du}{dx}$:\*\*

Primeiro, encontramos $\frac{dy}{dx}$, $\frac{dz}{dy}$ e $\frac{du}{dz}$:

$$
\frac{dy}{dx} = \frac{d}{dx}\left(\frac{x^3}{\sqrt{3}}\right) = \frac{3x^2}{\sqrt{3}}
$$

$$
\frac{dz}{dy} = \frac{d}{dy}\left((1 + y)^2\right) = 2(1 + y)
$$

$$
\frac{du}{dz} = \frac{d}{dz}\left(\frac{1}{\sqrt{1 + z}}\right) = -\frac{1}{2(1 + z)^{3/2}}
$$

Agora, utilizando a regra da cadeia:

$$
\frac{du}{dx} = \frac{du}{dz} \cdot \frac{dz}{dy} \cdot \frac{dy}{dx} = -\frac{1}{2(1 + z)^{3/2}} \cdot 2(1 + y) \cdot \frac{3x^2}{\sqrt{3}}
$$

Simplificando:

$$
\frac{du}{dx} = -\frac{3x^2(1 + y)}{\sqrt{3}(1 + z)^{3/2}}
$$

Como $y = \frac{x^3}{\sqrt{3}}$ e $z = (1 + y)^2 = \left(1 + \frac{x^3}{\sqrt{3}}\right)^2$, teremos:

$$
\frac{du}{dx} = -\frac{3x^2\left(1 + \frac{x^3}{\sqrt{3}}\right)}{\sqrt{3}\left[1 + \left(\frac{x^3}{\sqrt{3}}\right)\right]^{3/2}}
$$

Finalmente, simplificamos:

$$
\frac{du}{dx} = \frac{x^2\left(\sqrt{3 + x^3}\right)}{\sqrt{\left[1 + \left(\frac{x^3}{\sqrt{3}}\right)\right]^3}}
$$

Substituindo valores em $\frac{dw}{dx}$, teremos:

$$
v = 3\left(\frac{1}{2}x^3 + \left(\frac{1}{2}x^3\right)^2\right) = 3\left(\frac{1}{2}x^3 + \frac{1}{4}x^6\right)
$$

Então:

$$
v = \frac{3}{2}x^3 + \frac{3}{4}x^6
$$

Substituindo $v$ em $\frac{dw}{dx}$:

$$
\frac{dw}{dx} = -\frac{3x^2 (3 + 3x^3)}{\left(\frac{3}{2}x^3 + \frac{3}{4}x^6\right)^3}
$$

Substituindo valores em $\frac{dv}{dx}$, teremos:

$$
y = 3x^2 + \sqrt{2}
$$

$$
z = \sqrt{1 + y} = \sqrt{1 + 3x^2 + \sqrt{2}}
$$

$$
v = \frac{1}{\sqrt{3} + 4z}
$$

Substituindo $z$ em $\frac{dv}{dx}$:

$$
\frac{dv}{dx} = \frac{12x}{\sqrt{1 + \sqrt{2} + 3x^2} \left( \sqrt{3 + 4\sqrt{1 + \sqrt{2} + 3x^2}} \right)^2}
$$

Substituindo valores em $\frac{du}{dx}$, teremos:

$$
y = \frac{x^3}{\sqrt{3}}
$$

$$
z = (1 + y)^2 = \left(1 + \frac{x^3}{\sqrt{3}}\right)^2
$$

$$
u = \frac{1}{\sqrt{1 + z}} = \frac{1}{\sqrt{1 + \left(1 + \frac{x^3}{\sqrt{3}}\right)^2}}
$$

Substituindo $z$ em $\frac{du}{dx}$:

$$
\frac{du}{dx} = -\frac{3x^2\left(1 + \frac{x^3}{\sqrt{3}}\right)}{\sqrt{3}\left[1 + \left(\frac{x^3}{\sqrt{3}}\right)\right]^{3/2}}
$$

Simplificando:

$$
\frac{du}{dx} = \frac{x^2\left(\sqrt{3 + x^3}\right)}{\sqrt{\left[1 + \left(\frac{x^3}{\sqrt{3}}\right)\right]^3}}
$$
