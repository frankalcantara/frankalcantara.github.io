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
toc: false
published: false
slug: calculo-simplificado
draft: 2024-07-06T19:30:40.256Z
---

O livro **Calculus Made Easy** de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson) foi colocado em domínio público, tanto no Brasil quanto dos EUA. Este é um dos melhores livros introdutórios de cálculo já escrito. Simples, direto e abrangente. Sem nenhuma modéstia, ou vergonha na cara, escolhi este livro para tradução, atualização e expansão. Vou tentar ser o mais fiel possível ao autor. Mas, vou atualizar todo o conteúdo e expandir alguns conceitos e, principalmente, o número de exercícios resolvidos. Contudo, como o livro é ótimo, algumas coisas não podem ser deixadas de lado, ou versionadas como a observação da capa:

>O QUE UM TOLO PODE FAZER, OUTRO TAMBÉM PODE.
>(Provérbio Simiano Antigo.)

Em 1910 quando **Calculus Made Easy** foi publicado, ainda era aceitável e interessante, brincar e se divertir quando estudávamos matemática, ou ciência. Talvez seja por isso que este livro é lembrado por tantos professores com tanto carinho. Então, vamos ao prólogo de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson), com o máximo de fidelidade que esse tolo consegue usar para traduzir e escrever.

>Considerando quantos tolos conseguem calcular, é surpreendente que se pense que é uma tarefa difícil, ou tediosa, para qualquer outro tolo aprender a dominar os mesmos truques.
>Alguns truques de cálculo são bastante fáceis. Alguns são enormemente difíceis. Os tolos que escrevem os livros de texto de matemática avançada — e eles são na maioria tolos inteligentes — raramente se dão ao trabalho de mostrar como os cálculos fáceis são fáceis. Pelo contrário, eles parecem desejar impressioná-lo com sua tremenda astúcia, abordando-o da maneira mais difícil.
>Sendo eu mesmo um sujeito notavelmente estúpido, tive que desaprender as dificuldades, e agora peço para apresentar aos meus colegas tolos as partes que não são difíceis. Domine estas completamente, e o resto seguirá. O que um tolo pode fazer, outro também pode.

Deste ponto em diante, sempre que eu apenas traduzir o livro original, a tradução estará destacada. Todo o resto do livro será versionado. Isso significa que estarei comentando, atualizando e aprimorando tudo que puder e, nesse ritmo, segue o livro:

## № 1. PARA LIVRÁ-LO DOS TERRORES PRELIMINARES

O terror preliminar, que sufoca a maioria dos meninos da quinta série impedindo-os de sequer tentar aprender cálculo, pode ser abolido de uma vez por todas, simplesmente declarando qual é o significado, em termos de bom senso, dos dois símbolos principais usados no cálculo. Esses símbolos terríveis são:

1. $d$ que significa meramente *um pedacinho de*. Ou se preferir, com um pouco mais de formalidade *uma fração muito pequena de*. Assim, $dx$ significa um pedacinho de $x$; ou $du$ significa um pedacinho de $u$. Matemáticos ordinários acham mais educado dizer *um elemento de* em vez de *um pedacinho de*, ou ainda uma *fração infinitesimal de*. Esta coisa de *infinitesimal* quer dizer que é tão pequeno que quase se confunde com o zero. Você verá que esses pequenos pedaços podem ser considerados infinitamente pequenos.

2. $\int $ que é apenas um *S* longo, e pode ser lido, se você quiser, como *a soma de*. Assim, $\int dx$ significa a soma de todos os pequenos pedaços de $x$; ou $\int dt$ significa a soma de todos os pequenos pedaços de $t$. Matemáticos ordinários chamam esse símbolo de *a integral de*. Qualquer tolo pode ver que se $x$ for considerado como composto por muitos pequenos pedaços, cada um dos quais é chamado de $dx$, se você somá-los, obterá, como resultado, a soma de todos os $dx$'s. A soma de todos os $dx$ é a mesma coisa que $x$. A palavra *integral* simplesmente significa *o todo* e tem o mesmo sentido de *somar todas as pequenas partes que compõem o todo*.

Pensando na duração de uma hora, você pode, se quiser, considerar que este é um intervalo de tempo dividido em 3600 pequenos pedaços chamados de  segundos. O todo dos segundos é a hora e a hora é a soma dos seus $3600$ pequenos pedaços somados.

A partir de agora, quando você encontrar uma expressão que começa com esse símbolo aterrorizante, $\int $, você saberá que ele está lá apenas para lhe dar instruções de que agora você deve realizar a operação soma, sempre e se puder, de todos os pequenos pedaços indicados pelos símbolos que o símbolo do medo, $\int $.

E é só isso!

## № 2.SOBRE DIFERENTES GRAUS DE PEQUENEZ

Pequenez, ou grau de quanto as coisas são pequenas, é uma palavra rara em português. Ainda assim, como descobriremos que em nossos processos de cálculo temos que lidar com quantidades pequenas de vários graus de pequenez. Precisamos desta palavra para entender que um pequeno pode ser menor que o outro.

Teremos também que aprender em que circunstâncias podemos considerar pequenas quantidades tão minúsculas que podemos omiti-las das nossas considerações. Tão pequenas que são insignificantes. E tudo depende da pequenez relativa.

>Esta pequenez relativa é uma tradução literal da palavra originalmente usada por Sylvanos Thompson. Ele estava tentando ser o mais simples possível, para ficar claro, ele explicará que aquilo que consideramos pequeno é relativo ao problema que estamos resolvendo. Ou seja, nem tudo que é pequeno para você será pequeno para mim.{: class="ntr"}

Antes de imergirmos no mundo das quaisquer regras e formalidades da matemática, vamos considerar alguns casos familiares. Existem $60$ minutos em uma hora, $24$ horas em um dia e $7$ dias em uma semana. Portanto, existem $1440$ minutos no dia e $10080$ minutos na semana.

Obviamente, $1$ minuto é uma quantidade muito pequena de tempo comparada a uma semana. Na verdade, nossos antepassados consideravam pequeno em comparação com uma hora, e chamavam de *um minuto,* significando uma fração minúscula — um sexagésimo de uma hora. Quando passaram a exigir subdivisões ainda menores de tempo, dividiram cada minuto em $60$ partes menores, que, nos dias da **Rainha Elizabeth I**, chamavam de *segundos minutos* (isto é, a sexagésima parte da sexagésima parte da hora).

> A palavra minuto tem origem na expressão latina "prima minuta", significando primeira parte pequena, usada por [Ptolomeu](https://en.wikipedia.org/wiki/Ptolemy) para dividir o círculo em 60 partes. A palavra segundo, origina-se na expressão latina "secunda minuta" {: ntr}

Agora, se um minuto é tão pequeno em comparação com uma semana, quão menor em comparação é um segundo!

>Antes de seguirmos, precisamos de um pouco de contexto. Um "farthing" era uma moeda de baixo valor usada na Inglaterra, equivalente a um quarto de um "penny". A palavra "farthing" deriva do inglês antigo "feorthing", que significa quarta parte. No sistema monetário britânico pré-decimal, existiam 240 pennies em uma libra esterlina ($£1$), e portanto, um "farthing" era igual a $\frac{1}{960}$ de uma libra. Além da libra comum existia a libra de ouro, chamada de "sovereign", soberano. Um soberano valia exatamente uma libra, $£1$. Contudo, por ser de ouro, era muito utilizada como reserva de valor e no mercado internacional. Devido ao seu valor extremamente baixo, o "farthing" foi usado para transações de pequeno valor até ser desmonetizado em 1961. SILVANUS P. THOMPSON, nosso autor escreveu este livro em 1910 e neste tempo, ainda existiam "farthings". Eu achei isso interessante demais para suprimir atualizando as unidades monetárias. Faltou ele falar da "Ginea", guinéu, que valia $21$ "xelins", ou seja, valia uma libra e um "xelin" e era usado como moeda para bens de luxo. Quase esqueço! havia ainda o "halfpenny", a metade de um "penny".

Voltando ao conceito de pequenez. Pense em um "farthing" em comparação com um soberano. Ele, o "farthing", pouco mais que $\frac{1}{1000}$ parte. Um "farthing" a mais ou a menos tem pouca importância em comparação com um soberano e, certamente, pode ser considerado uma quantidade muito pequena. Mas compare um "farthing" com £1000. Relativamente a esta quantia, o "farthing" não tem nem importância do que $\frac{1}{1000}$ de um "farthing" teria para um soberano. Mesmo um "soberano" é, relativamente, uma quantidade insignificante quando comparado com valores que se contem aos milhões de libras.

Agora, se escolhermos uma fração numérica como constituindo a proporção que, para qualquer propósito, chamaremos de pequeno, poderemos facilmente identificar outras frações de graus maiores de pequenez. Vamos voltar a medida do tempo se a fração $\frac{1}{60}$ for chamada de uma *parte pequena*, então $\frac{1}{60}$ de $\frac{1}{60}$ (sendo uma *pequena fração de uma pequena fração*) pode ser considerada uma *quantidade pequena da segunda ordem de pequenez*.

>Os matemáticos falam sobre a segunda ordem de "magnitude" (isto é: grandeza) quando realmente querem dizer segunda ordem de *pequenez*. Isso é muito confuso para os iniciantes.{: class="ntr"}

Ou, se para qualquer propósito pegássemos $1$ por cento (isto é: $\frac{1}{100}$) como uma *fração pequena*, então $1$ por cento de $1$ por cento (isto é: $\frac{1}{10.000}$) seria uma fração de *segunda ordem em pequenez*; e $\frac{1}{1.000.000}$ seria uma pequena fração da *terceira ordem em pequenez*, representando $1$ por cento de $1$ por cento de $1$ por cento.

>A melhor tradução do termo usado pelo autor para *smallness*, em português é pequenez, mas este é um termo pouco usado neste começo de século XXI. Na verdade, Thompson está destacando a insignificância das coisas enquanto as dividimos em pedaços cada vez menores e criando o conceito de insignificância relativa.{: class="ntr"}

Por fim, suponha que, para algum propósito específico, devamos considerar $\frac{1}{1.000.000}$ como *pequena*. Assim, se um cronômetro de primeira categoria não deve perder ou ganhar mais de meio minuto em um ano, ele deve manter o tempo com uma precisão de $1$ parte em $1.051.200$. Agora, se, para tal propósito, considerarmos $\frac{1}{1.000.000}$ (ou um milionésimo) como uma quantidade pequena, então $\frac{1}{1.000.000}$ de $\frac{1}{1.000.000}$, ou seja, $\frac{1}{1.000.000.000.000}$ (ou um trilionésimo) será uma quantidade pequena da *segunda ordem de pequenez*, e pode ser completamente desconsiderada.

Vimos que quanto menor for uma quantidade pequena em si, mais insignificante se torna a correspondente quantidade pequena da segunda ordem. Portanto, sabemos que *em todos os casos estamos justificados em negligenciar as pequenas quantidades da segunda, terceira, ou maior ordens, tudo que é necessário é que a quantidade pequena de primeira ordem seja considerada suficientemente pequena em si mesma*. A escolha deste valor irá depender do problema. Contudo, será um valor suficientemente pequeno que permita resolver o problema usando o cálculo.

Outra coisa importante é que não podemos esquecer que pequenas quantidades, quando sujeitas, em nossas equações, a multiplicação por outro fator, podem se tornar relevantes caso o outro fator seja, em si grande para problema. Mesmo um "farthing" se torna importante se for multiplicado por centenas, milhares ou milhões de vezes.

No cálculo, escrevemos $dx$ para representar um pedacinho de $x$, para representar uma fração muito pequena de $x$. Essas coisas como $dx$, e $du$, e $dy$, serão chamadas de *diferenciais*, e lidas como o diferencial de $x$, ou o diferencial de $u$, ou o diferencial de $y$, conforme o caso. Ou ainda, para sermos mais chegados a matemática, serão chamados de a derivada de $x$, a derivada de $y$ ou a derivada de$u$. Mesmo que $dx$ seja um pedacinho de $x$, e considerado relativamente pequeno em si mesmo para resolver o nosso problema, não deduzir que quantidades como $x \cdot dx$, ou $x^2 \, dx$, ou $a^x \, dx$ sejam insignificantes. Mas implica, sem nenhuma dúvida que $dx \times dx$ será insignificante, já que, neste caso teremos uma quantidade pequena da segunda ordem.

Um exemplo muito simples servirá como ilustração.{#volta1}

Vamos pensar em $x$ como uma quantidade que pode crescer apenas um pedacinho de modo que, em algum momento, tenhamos $x + dx$, onde $dx$ é o pequeno incremento adicionado pelo crescimento. O quadrado de $x + dx$ será $x^2 + 2x \, dx + (dx)^2$. O segundo termo não é desprezível porque é uma quantidade de primeira ordem; enquanto o terceiro termo é da segunda ordem de pequenez, $(dx)^2 = dx\times dx$. Para ficar claro, se fizer $dx$ ter um valor numérico, digamos, $\frac{1}{60}$ de $x$, então o segundo termo seria $\frac{2x^2}{60}$, enquanto o terceiro termo seria $\frac{1}{3600}$. Este terceiro termo é claramente menos significativo que o segundo. Mas, se formos além, podemos considerar nosso pedacinho $dx$ como $\frac{1}{1000}$ de $x$, então o segundo termo será $\frac{2x^2}{1000}$, enquanto o terceiro termo será apenas $\frac{1}{1.000.000}$. *Aquilo que consideraremos pequeno irá depender do problema real, mas na matemática, o pequeno sempre será $dx$. Ou a derivada de qualquer outra variável independente*.

![]({{ site.baseurl }}/assets/images/calc_Fig1.jpg)
*Figura 1 - Um quadrado acrescido de $dx$.*

Talvez um pouco de geometria básica possa ajudar.

Desenhe um quadrado cujo lado representaremos como $x$, Fig. 1a. Agora, suponha que o quadrado cresça adicionando uma pequena quantidade, um $dx$, ao seu tamanho em cada direção,Fig. 1b. O quadrado ampliado será composto pelo quadrado original de área $x^2$, mais os dois retângulos na parte superior e na direita, cada um com área $xdx$ (ou juntos $2xdx$), e o pequeno quadrado no canto superior direito que é $(dx)^2$. Na Fig. 1b, representamos $dx$ como uma fração muito grande de $x$, — cerca de $\frac{1}{10}$ por razões didáticas e gráficas. Mas suponha que façamos $dx$ como $\frac{1}{100}$, aproximadamente a espessura de uma linha desenhada com uma caneta fina. Então, o pequeno quadrado no canto superior direito terá uma área de apenas $\frac{1}{10,000}$ de $x^2$, e será praticamente invisível. Claramente, $(dx)^2$ será desprezível sempre que considerarmos o incremento $dx$ suficientemente pequeno.

Vamos considerar uma analogia.

Suponha que um milionário dissesse ao seu secretário: na próxima semana, eu lhe darei uma pequena fração de qualquer valor monetário que eu receba. Suponha que o secretário dissesse ao seu filho: eu lhe darei uma pequena fração de tudo que eu receber. Suponha que a fração em cada caso seja $\frac{1}{100}$ (um centésimo). Agora, se o Sr. Milionário recebesse £1000 durante a próxima semana, o secretário receberia £10 e o garoto 2 xelins.

>Um "xelin" é um 20 avos de uma libra, então, £10 correspondem a 200 "xelins". Um centésimo de 200 xelins* é 2 "xelins".{: class="ntr"}

Dez libras seriam uma quantidade pequena em comparação com £1000; mas dois xelins é realmente uma quantidade muito pequena, de segunda ordem. Mas qual seria esta relação se a fração, em vez de ser $\frac{1}{100}$, fosse $\frac{1}{1000}$ (um milésimo)? Neste caso, quando o Sr. Milionário recebesse suas £1000, o Sr. Secretário receberia apenas £1, e o garoto menos de um "farthing"!

O espirituoso Dean Swift[^1] uma vez escreveu:{#nt1}

>So, Nat'ralists observe, a Flea
>Hath smaller Fleas that on him prey.
>And these have smaller Fleas to bite 'em,
>And so proceed ad infinitum.

Um boi pode se preocupar com uma pulga de tamanho comum, uma criatura da primeira ordem de pequenez. Mas provavelmente não se incomodará com a pulga de uma pulga. Sendo esta uma criatura de segunda ordem de pequenez, seria insignificante. Mesmo uma quantidade gigantesca de pulgas de pulgas não teria muita importância para um boi.

## № 3. SOBRE CRESCIMENTOS RELATIVOS

Durante todo o cálculo, lidaremos com quantidades que estão variando, aumentando ou diminuindo, e com as suas taxas de variação.

Classificaremos todas as quantidades em duas classes: *constantes* e *variáveis*. Aquelas que consideramos de valor fixo, e chamamos de *constantes*, geralmente denotamos algebricamente usando letras do início do alfabeto latino, como $a$, $b$ ou $c$; enquanto aquelas que consideramos capazes de crescer ou diminuir, ou ainda, como dizem os matemáticos, capazes de "variar", denotamos por letras do final do alfabeto latino, como $x$, $y$, $z$, $u$, $v$, $w$ ou, às vezes, $t$.

Além disso, geralmente lidamos com mais de uma variável ao mesmo tempo e somos levados a considerar a forma como uma variável depende da outra: por exemplo, para um problema precisamos considerar a forma como a altura atingida por um projétil depende do tempo necessário para atingir essa altura. Ou somos convidados a investigar um retângulo de área dada e a perguntar de que forma um aumento, ainda que pequeno, no comprimento dele implicará em uma redução na sua largura, de forma a manter a área constante. Ou, ficamos intrigados como uma variação qualquer na inclinação de uma escada implicará qual variação na altura que ela atinge.

Suponha que tenhamos duas variáveis que dependem uma da outra. São variáveis que ocorrem em problemas tais que uma variação qualquer em uma delas causará uma alteração no valor da outra. *Vamos chamar uma das variáveis de $x$, a variável que não depende, e portanto é independente e de $y$ a variável que depende*. Neste caso, diremos que $x$ é a variável independente, $y$ a variável dependente.

Para manter esta linha de raciocínio, suponha agora que façamos $x$ variar, ou seja, alteramos, ou imaginamos que o valor de $x$ foi alterado, adicionando a ela uma fração muito pequena do seu valor. Fração que já chamamos de $dx$. Assim, faremos $x$ se tornar $x + dx$. Então, porque $x$ foi alterado, $y$ também terá sido alterado, e terá se tornado $y + dy$. Aqui, o pouco $dy$ pode ser em alguns casos positivo, em outros negativo; e não será, exceto por um milagre, do mesmo tamanho que $dx$.

![]({{ site.baseurl }}/assets/images/calc_Fig2.jpg){#figura2}
*Figura 2 - Um crescimento $dx$ em um triângulo.*{: class="legend lazyimg"}

Considere dois exemplos.

(1) Vamos fazer com que $x$ e $y$ sejam, respectivamente, a base e a altura de um triângulo retângulo, Fig.2a, cuja inclinação da hipotenusa esteja fixada em $30^\circ$. Se supusermos que este triângulo é capaz de se expandir e ainda manter seus ângulos constantes, então, quando a base crescer de modo a se tornar $x + dx$, a altura se tornará $y + dy$. Como visto na Fig.2a. Neste cenário, o aumento de $x$ resulta em um aumento de $y$. Observe ainda que o pequeno triângulo, cuja altura é $dy$, e cuja base é $dx$, é semelhante ao triângulo original. Deve ser óbvio que o valor da razão $\frac{dy}{dx}$ será o mesmo da razão $\frac{y}{x}$. Como o ângulo é $30^\circ$, é constante, veremos que:{#caso1}

$$
\frac{dy}{dx} = \frac{1}{1.73}.
$$

(2) Agora façamos com que $x$ represente, na Fig.2b, a distância horizontal, a partir de uma parede, da extremidade inferior de uma escada, $AB$ em lilás, de comprimento fixo; neste caso, $y$ será a altura que a escada atinge na parede. Agora, $y$ claramente depende de $x$. É fácil ver que, se puxarmos a extremidade inferior $A$ um pouco mais para longe da parede, a extremidade superior $B$ descerá um pouco [Fig.2b](#figura2 em azul). Com uma pouco mais de formalidade matemática podemos afirmar que: se aumentarmos $x$ para $x + dx$, então $y$ se tornará $y - dy$; isto é, quando $x$ recebe um incremento positivo, o incremento em $y$ será negativo.{#caso2}

Parece razoável, mas quão razoável? Suponha que a escada fosse tão longa que, quando a extremidade inferior $A$ estivesse a $50$ centímetros da parede, a extremidade superior $B$ alcançasse $4.5$ metros do chão. Agora, se você puxasse a extremidade inferior $1$ centímetro a mais, quanto a extremidade superior desceria? Vamos colocar tudo em metros: $x = 0.5 \, m$ e $y = 4.5 \, m$. Agora, o incremento de $x$ que chamamos de $dx$ é de $0.01 \, m$. Sendo assim, $x + dx = 0.51 \, m$. Entendemos que o $x$, a variável independente, variar $x$ significa variar $y$.

Quanto $y$ irá variar? A nova altura será $y - dy$, sem dúvidas, mas temos que calcular este $dy$. Se calcularmos a altura pelo Teorema de Pitágoras poderemos descobrir quanto será $dy$. O comprimento da escada é:

$$
\sqrt{(4.5)^2 + (0.5)^2} = 4.52769 \text{ metros}.
$$

Claramente, então, a nova altura, que é $y - dy$, será tal que

$$
(y - dy)^2 = (4.52769)^2 - (0.51)^2 = 20.50026 - 0.2601 = 20.24016,
$$

$$
y - dy = \sqrt{20.24016} = 4.49557 \text{ metros}.
$$

Agora $y = 4.50000$, então $dy$ é $4.50000 - 4.49557 = 0.00443$ metros.

Vemos que fazer $dx$ um aumento de $0.01$ metros resultou em fazer $dy$ uma diminuição de $0.00443$ metros.

E a razão de $dy$ para $dx$ pode ser declarada da seguinte forma:

$$
\frac{dy}{dx} = \frac{0.00443}{0.01} = 0.443.
$$

ou se preferirmos, como no enunciado eu escolhi $1 cm$ de deslocamento, este será o nosso valor pequeno então podemos dizer que:

$$
\frac{dy}{dx} = \frac{0.443}{1} = 0.443.
$$

Também é possível perceber que, exceto em uma posição particular, $dy$ terá um tamanho diferente de $dx$. Que posição é essa?[^2]{#nt2}

Voltando ao cálculo diferencial, nos dois casos anteriores, estamos buscando uma coisa curiosa, uma mera razão, à saber: a proporção que $dy$ tem em relação à $dx$, quando ambos são indefinidamente pequenos. Ou melhor, estamos procurando uma razão quando ambos, $dx$ e $dy$, são infinitesimais.

Note que só podemos encontrar essa razão, $\frac{dy}{dx}$, quando $y$ e $x$ estão relacionados de alguma forma. Geralmente escolhemos as variáveis de modo que sempre que $x$ varia, $y$ também varia.

No caso 1, recém mencionado, se a base, $x$, do triângulo for aumentada, a altura, $y$, do triângulo também se tornará maior, e no caso 2, se a distância, $x$, do pé da escada à parede for aumentada, a altura, $y$, atingida pela escada diminuirá de maneira correspondente, inicialmente devagar diminuirá muito devagar. Mas, diminuirá mais e mais rapidamente à medida que $x$ se torna maior.

Nos casos que vimos anteriormente, as relações entre $x$ e $y$ podem ser perfeitamente definidas usando um pouco de álgebra e geometria. Se fizermos isso encontraremos $\frac{y}{x} = \tan 30^\circ$ para o caso 1 e $x^2 + y^2 = L^2$ para o caso 2, desde que $L$ seja o comprimento da escada. Finalmente, $\frac{dy}{dx}$ tera apenas o significado que encontramos em cada caso.

Enquanto $x$ é, como antes, a distância do pé da escada à parede, $y$ é, em vez da altura alcançada, o comprimento horizontal da parede, ou número de tijolos na parede, ou ainda o número de anos desde que foi construída, qualquer mudança em $x$ não causará nenhuma mudança em $y$; neste caso, $\frac{dy}{dx}$ não tem significado algum, e não é possível encontrar uma expressão que relacione a variável independente $x$ com a variável dependente $y$.

Sempre que usarmos diferenciais $dx$, $dy$, $dz$, etc., a existência de algum tipo de relação entre $x$, $y$, $z$, etc., é implícita, e essa relação será chamada de "função" em $x$, $y$, $z$, etc. As duas expressões dadas acima, $\frac{y}{x} = \tan 30^\circ$ e $x^2 + y^2 = l^2$, são funções de $x$ e $y$. Tais expressões contêm implicitamente os meios de expressar $x$ em termos de $y$ ou $y$ em termos de $x$, e por essa razão são chamadas de funções implícitas em $x$ e $y$; elas podem ser respectivamente colocadas nas formas:

$$y = x \tan 30^\circ \quad \text{ou} \quad x = \frac{y}{\tan 30^\circ}$$

$$\text{e} \quad y = \sqrt{l^2 - x^2} \quad \text{ou} \quad x = \sqrt{l^2 - y^2}.$$

Essas últimas expressões afirmam explicitamente o valor de $x$ em termos de $y$, ou de $y$ em termos de $x$, e por essa razão são chamadas de "funções explícitas" de $x$ ou $y$. Por exemplo, $x^2 + 3 = 2y - 7$ é uma função implícita em $x$ e $y$. A função $x^2 + 3 = 2y - 7$ pode ser escrita $y = \frac{x^2 + 10}{2}$ (função explícita de $x$) ou $x = \sqrt{2y - 10}$ (função explícita de $y$).

Vimos que uma função explícita em $x$, $y$, $z$, etc., é simplesmente algo cujo valor muda quando $x$, $y$, $z$, etc., estão mudando. Por causa disso, o valor que encontramos quando calculamos uma função explícita é chamado de "variável dependente*. muitas vezes optamos por funções em $x$ logo $y$ é a variável dependente pois depende do valor das outras variáveis na função; essas outras variáveis são chamadas de "variáveis independentes" porque seu valor não é determinado pelo valor assumido pela função, muitas vezes optamos por $x$. Por exemplo: se $u = x^2 \sin \theta$, $x$ e $\theta$ são as variáveis independentes, e $u$ é a variável dependente.

Às vezes, a relação exata entre várias quantidades $x$, $y$, $z$ não é conhecida, ou não é conveniente afirmá-la. Nestes casos, conhecemos, ou somos capazes de afirmar que existe algum tipo de relação entre essas variáveis, de modo que não se pode alterar $x$ ou $y$ ou $z$ individualmente sem afetar as outras quantidades. A existência de uma função em $x$, $y$, $z$ é então indicada pela notação $f(x, y, z)$ (função implícita) ou por $x = f(y, z)$, $y = f(x, z)$ ou $z = f(x, y)$ (funções explícitas). Às vezes, a letra $F$ ou $\phi$ é usada em vez de $f$, de modo que $y = f(x)$, $y = F(x)$ e $y = \phi(x)$ significam que o valor de $y$ depende do valor de $x$ de alguma forma que não está sendo afirmada, por não ser conhecida ou conveniente.

Chamamos a razão $\frac{dy}{dx}$ de "derivada de $y$ com respeito a $x$"" formalmente deveríamos chamar de "coeficiente diferencial de $y$ com respeito a $x$", mas derivada basta. Ainda assim, é um nome científico e solene, para uma coisa tão simples. Entretanto, neste texto, não vamos nos assustar com nomes solenes. Em vez de nos assustarmos, simplesmente pronunciaremos uma breve maldição sobre a estupidez de dar nomes longos e complicados e, tendo aliviado nossos corações e mentes, passaremos à coisa simples em si, encontrar a razão $\frac{dy}{dx}$.

Na álgebra comum que você aprendeu na escola, você estava sempre tentando encontrar alguma quantidade desconhecida que você chamava de $x$ ou $y$. Agora você tem que aprender a procurar estas quantidades de uma forma nova. O processo de encontrar o valor de $\frac{dy}{dx}$ é chamado de *diferenciação*. Mas, lembre-se, o que se deseja é o valor dessa razão quando tanto $dy$ quanto $dx$ são indefinidamente pequenos, esta é a última vez que vou lembrar isso. Juro!

Estamos preocupados em encontrar a razão $\frac{dy}{dx}$ quando $dx$ e $dy$ são infinitesimalmente pequenos, tão pequenos que tendem a zero.

Vamos agora aprender como encontrar $\frac{dy}{dx}$.

## Como ler diferenciais

A amável leitora nunca deve cometer o erro de pensar que $dx$ significa $d$ vezes $x$, pois $d$ não é um fator – significa "um elemento de" ou "um pedacinho de" qualquer coisa que se siga com este mesmo sentido. Lê-se $dx$ assim: "de-xis".

Coeficientes diferenciais de segunda ordem, que serão encontrados mais tarde. Serão representados: $\frac{d^2y}{dx^2}$, o que é lido "de-dois-ipsilon de-xis-quadrado", significando que operação de diferenciar $y$ em relação a $x$ foi, ou tem que ser, feita duas vezes consecutivas. Esta é a notação de Leibnitz.

Outra maneira de indicar que uma função foi diferenciada é colocando um acento no símbolo da função. Assim, se $y = F(x)$, o que significa que $y$ é alguma função não especificada em $x$, podemos escrever $f'(x)$ em vez de $\frac{d(f(x))}{dx}$, ou ainda $\frac{d}{dx}f(x)$ que eu gosto mais porque separa o coeficiente diferencial explicitando que ele não tem nenhuma relação com a divisão de $dx$ por $dy$. Da mesma forma, $f''(x)$ significará que a função original $f(x)$ foi diferenciada duas vezes com respeito a $x$.

Vamos usar as estas notações livremente neste texto.

## №4. CASOS MAIS SIMPLES

Agora, vejamos como, a partir dos princípios fundamentais da álgebra, podemos diferenciar algumas expressões algébricas simples. Vamos usar apenas a álgebra que você aprendeu no ensino médio, ou antes.

### CASO 1

Vamos começar com uma função simples $y = x^2$. Agora, *lembre-se de que a noção fundamental do cálculo é a ideia de crescimento, ou decréscimo*. Os matemáticos chamam isso de variação. Agora, como $y$ e $x^2$ são iguais, é claro que se $x$ cresce, $x^2$ também crescerá. E se $x^2$ crescer, então $y$ também crescerá. É isso que significa ser igual.

O que temos que descobrir é a proporção entre o crescimento de $y$ e o crescimento de $x$. Em outras palavras, nossa tarefa é descobrir a razão entre $dy$ e $dx$, que representamos na expressão *encontrar o valor de $\frac{dy}{dx}$*.

Vamos fazer $x$ crescer apenas um pedacinho $dx$ e se tornar $x + dx$; da mesma forma, $y$ crescerá se tornará $y + dy$. Mesmo assim, segundo a função $y = x^2$, ainda será verdade que o $y$ aumentado será igual ao quadrado do $x$ aumentado. Escrevendo isso, temos:

$$
y + dy = (x + dx)^2
$$

Fazendo o quadrado, obtemos:

$$
y + dy = x^2 + 2x \cdot dx + (dx)^2
$$

O que $(dx)^2$ significa? Lembre-se de que $dx$ significava uma parte, fração, muito pequena de $x$. Então, $(dx)^2$ significará uma fração de uma fração $x$; isto é, conforme explicado anteriormente, é uma quantidade da segunda ordem de pequenez. Pode, portanto, ser descartado por ser insignificante em comparação com os outros termos. ou seja:

$$
y + dy = x^2 + 2x \cdot dx
$$

Agora, $y = x^2$; então vamos subtrair isso da equação e ficamos com:

$$
dy = 2x \cdot dx
$$

Como estamos buscando uma razão, dividindo por $dx$, encontramos:

$$
\frac{dy}{dx} = 2x
$$

Encontramos a razão do crescimento de $y$ dado um crescimento em $x$. Neste caso, $2x$.

>*Nota* – Esta razão $\frac{dy}{dx}$ é o resultado de diferenciar $y$ com respeito a $x$. Diferenciar significa encontrar o coeficiente diferencial. Suponha que tivéssemos outra função de $x$, como, por exemplo, $u = 7x^2 + 3$. Então, se nos mandassem diferenciar isso com respeito a $x$, deveríamos encontrar $\frac{du}{dx}$, ou, o que é a mesma coisa, $\frac{d(7x^2 + 3)}{dx}$. Muitas vezes temos casos em que o tempo será a variável independente, neste caso: $y = b + \frac{1}{2}at^2$. Então, se nos mandassem diferenciar esta função teríamos que encontrar seu coeficiente diferencial com respeito a $t$. Logo, nosso trabalho seria tentar encontrar $\frac{dy}{dt}$, isto é, encontrar $\frac{d(b + \frac{1}{2}at^2)}{dt}$.{: class="nota"}

Exemplo numérico.

Suponha $x = 100$ e portanto $y = 10,000$. Então, vamos fazer $x$ crescer até se tornar $101$ (ou seja, $dx = 1$). Então, o $y$ aumentado será $101 \times 101 = 10,201$. Mas se concordarmos que podemos ignorar pequenas quantidades da segunda ordem, pode ser rejeitado em comparação com $10,000$; então podemos arredondar o $y$ aumentado para $10,200$. $y$ cresceu de $10,000$ para $10,200$; o acréscimo em $dy$ será, portanto, $200$.

$$
\frac{dy}{dx} = \frac{200}{1} = 200
$$

De acordo com o trabalho algébrico do parágrafo anterior, encontramos $\frac{dy}{dx} = 2x$. E assim é; para $x = 100$ e $2x = 200$.

Mas, você dirá, negligenciamos uma unidade inteira. Uma unidade não é coisa que se negligencie.

Bem, tente novamente, tornando $dx$ ainda menor.

Tente $dx = \frac{1}{10}$. Então $x + dx = 100.1$, e

$$
(x + dx)^2 = 100.1 \times 100.1 = 10,020.01
$$

Agora, o último dígito $1$ é apenas uma milionésima parte de $10,000$ e é totalmente insignificante; então podemos tomar $10,020$ sem o pequeno decimal no final. E isso faz $dy = 20$; e

$$
\frac{dy}{dx} = \frac{20}{0.1} = 200
$$

o que ainda é o mesmo que $2x$.

### CASO 2

Tente diferenciar $y = x^3$ usando a mesma técnica do CASO 1: deixamos $y$ crescer para $y + dy$, enquanto $x$ cresce para $x + dx$.

Então temos:

$$
y + dy = (x + dx)^3
$$

Abrindo este cubo, obtemos:

$$
y + dy = x^3 + 3x^2 \cdot dx + 3x(dx)^2 + (dx)^3
$$

Agora, como sabemos que podemos negligenciar pequenas quantidades da segunda e terceira ordens, sempre que $dy$ e $dx$ são considerados infinitamente pequenos, $(dx)^2$ e $(dx)^3$ se tornarão infinitamente menores que $dy$ e $dx$. Assim, considerando-os insignificantes, teremos:

$$
y + dy = x^3 + 3x^2 \cdot dx
$$

Como $y = x^3$; podemos subtrair $y$ dos dois lados e termos:

$$
dy = 3x^2 \cdot dx
$$

Mas, o que queremos é uma relação. Logo:

$$
\frac{dy}{dx} = 3x^2
$$

### CASO 3

Tente diferenciar $y = x^4$. Começando exatamente como fizemos antes, faremos tanto $y$ quanto $x$ crescerem um pouco, e teremos:

$$
y + dy = (x + dx)^4
$$

Abrindo a quarta a quarta potência, obtemos:

$$
y + dy = x^4 + 4x^3 \cdot dx + 6x^2 (dx)^2 + 4x(dx)^3 + (dx)^4
$$

Então, eliminando os termos que contêm todas as maiores potências de $dx$, como sendo insignificantes, teremos:

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

Todos os casos que vimos até o momento são muito fáceis, desde que você saiba trabalhar com polinômios e tenha entendido porque podemos desconsiderar termos em que $dx$ está elevados a potências diferentes de $1$. Vamos tabular os resultados para ver se podemos inferir alguma regra geral. Vamos usar duas colunas, os valores de $y$ em uma e os valores correspondentes encontrados para $\frac{dy}{dx}$ na outra; assim:

| y   | $\frac{dy}{dx}$  |
|-----|------------------|
| $x^2$ | $2x$            |
| $x^3$ | $3x^2$          |
| $x^4$ | $4x^3$          |

*Tabela 1 - Uma comparação entre $x^n$ e sua derivada, $dx$.*{: class="legend"}

Apenas observe estes resultados: a operação de diferenciar parece ter tido o efeito de diminuir a potência de $x$ em $1$ (por exemplo, no último caso, reduzindo $x^4$ para $x^3$), e ao mesmo tempo, a operação de diferenciar parece estar multiplicando $x$ pelo mesmo número que pareceu originalmente como a potência. Volte aos CASOS 1, 2 e 3 e observe para ver se é isto que está acontecendo.

Como vimos isso acontecer várias vezes, podemos conjecturar como outras potências serão diferenciadas. Se esta regra observada estiver certa você deve esperar que diferenciar $x^5$ resultaria em $5x^4$, ou diferenciar $x^6$ resultaria $6x^5$. Se, a amável leitora, não estiver convencida, deve tentar diferenciar estes dois monômios.

Vamos tentar com $y = x^5$. Neste caso:

$$
y + dy = (x + dx)^5
$$

$$
= x^5 + 5x^4 \cdot dx + 10x^3 (dx)^2 + 10x^2 (dx)^3 + 5x (dx)^4 + (dx)^5.
$$

Ignorando todos os termos contendo pequenas quantidades de ordens superiores, temos:

$$
y + dy = x^5 + 5x^4 \cdot dx,
$$

Subtraindo $y = x^5$, obtemos:

$$
dy = 5x^4 \cdot dx,
$$

ou seja,

$$
\frac{dy}{dx} = 5x^4,
$$

Exatamente como conjecturamos poucas linhas acima. Seguindo logicamente nossa observação, devemos concluir que, se quisermos lidar com qualquer potência superior – que chamaremos de $n$ – poderíamos abordá-la da mesma maneira e seguir a mesma regra. Vamos fazer $y = x^n$:

$$
\frac{dy}{dx} = nx^{(n-1)}.
$$

Por exemplo, se fizermos $n = 8$, então $y = x^8$; e diferenciarmos esta expressão, teremos:

$$
\frac{dy}{dx} = 8x^7.
$$

De fato, existe uma regra de diferenciação, chamada de  Regra da Potência que diz que  diferenciar $x^n$ resultará $nx^{n-1}$ para todos os casos onde $n$ é um número inteiro e positivo. podemos Expandir $(x + dx)^n$ usando o Teorema Binomial de Newton e demonstrar.

#### Teorema Binomial para Expandir $(x+dx)^8$

O Teorema Binomial de Newton[^2]{#nt3} nos diz que podemos expandir a expressão $(x + dx)^8$ em uma série de termos envolvendo coeficientes binomiais. A fórmula geral para o teorema binomial é:

$$
(x + dx)^8 = \sum_{k=0}^{8} \binom{8}{k} x^{8-k} (dx)^k
$$

Onde $\binom{8}{k}$ é o coeficiente binomial dado por:

$$
\binom{8}{k} = \frac{8!}{k!(8-k)!}
$$

Vamos considerar um exemplo prático para expandir $(x + dx)^8$ usando o teorema binomial.

#### Exemplo: Expansão de $(x + dx)^8$

A expansão de $(x + dx)^8$ é dada por:

$$
(x + dx)^8 = \sum_{k=0}^{8} \binom{8}{k} x^{8-k} (dx)^k
$$

Vamos calcular cada termo da soma individual e cuidadosamente:

1. Para $k = 0$:

    $$
    \binom{8}{0} x^{8-0} (dx)^0 = 1 \cdot x^8 \cdot 1 = x^8
    $$

2. Para $k = 1$:

    $$
    \binom{8}{1} x^{8-1} (dx)^1 = 8 \cdot x^7 \cdot dx = 8x^7 dx
    $$

3. Para $k = 2$:

    $$
    \binom{8}{2} x^{8-2} (dx)^2 = 28 \cdot x^6 \cdot (dx)^2 = 28x^6 (dx)^2
    $$

4. Para $k = 3$:

    $$
    \binom{8}{3} x^{8-3} (dx)^3 = 56 \cdot x^5 \cdot (dx)^3 = 56x^5 (dx)^3
    $$

5. Para $k = 4$:

    $$
    \binom{8}{4} x^{8-4} (dx)^4 = 70 \cdot x^4 \cdot (dx)^4 = 70x^4 (dx)^4
    $$

6. Para $k = 5$:

    $$
    \binom{8}{5} x^{8-5} (dx)^5 = 56 \cdot x^3 \cdot (dx)^5 = 56x^3 (dx)^5
    $$

7. Para $k = 6$:
    $$
    \binom{8}{6} x^{8-6} (dx)^6 = 28 \cdot x^2 \cdot (dx)^6 = 28x^2 (dx)^6
    $$  

8. Para $k = 7$:
    $$
    \binom{8}{7} x^{8-7} (dx)^7 = 8 \cdot x^1 \cdot (dx)^7 = 8x (dx)^7
    $$

9. Para $k = 8$:

    $$
    \binom{8}{8} x^{8-8} (dx)^8 = 1 \cdot x^0 \cdot (dx)^8 = (dx)^8
    $$

Somando todos os termos, obtemos a expansão completa:

$$
(x + dx)^8 = x^8 + 8x^7 dx + 28x^6 (dx)^2 + 56x^5 (dx)^3 + 70x^4 (dx)^4 + 56x^3 (dx)^5 + 28x^2 (dx)^6 + 8x (dx)^7 + (dx)^8
$$

Portanto, o polinômio equivalente a $(x + dx)^8$ é:

$$
(x + dx)^8 = x^8 + 8x^7 dx + 28x^6 (dx)^2 + 56x^5 (dx)^3 + 70x^4 (dx)^4 + 56x^3 (dx)^5 + 28x^2 (dx)^6 + 8x (dx)^7 + (dx)^8
$$

A questão de saber se esta regra que conjecturamos é verdadeira para casos onde $n$ tem valores negativos ou fracionários requer consideração adicional.

#### Caso de uma potência negativa

Deixe $y = x^{-2}$. Então proceda como antes:

$$
y + dy = (x + dx)^{-2}
$$

$$
= x^{-2} \left(1 + \frac{dx}{x}\right)^{-2}.
$$

Expandindo isso com o teorema binomial, obteremos

$$
= x^{-2} \left[1 - \frac{2 \, dx}{x} + \frac{2(2 + 1)}{1 \times 2} \left(\frac{dx}{x}\right)^2 - \text{etc.}\right]
$$

$$
= x^{-2} - 2x^{-3} \cdot dx + 3x^{-4} (dx)^2 - 4x^{-5} (dx)^3 + \text{etc.}
$$

Assim, negligenciando as pequenas quantidades das ordens superiores de pequenez, temos:

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

E isso ainda está de acordo com a regra que conjecturamos acima.

#### Caso de uma potência fracionária

Vamos tentar com $y = x^{\frac{1}{2}}$. Então, como antes,

$$
y + dy = (x + dx)^{\frac{1}{2}} = x^{\frac{1}{2}} \left(1 + \frac{dx}{x}\right)^{\frac{1}{2}}
$$

$$
= \sqrt{x} + \frac{1}{2} \frac{dx}{\sqrt{x}} - \frac{1}{8} \frac{(dx)^2}{x \sqrt{x}} + \text{termos com potências mais altas de } dx.
$$

Subtraindo o original $y = x^{\frac{1}{2}}$, e ignorando as potências mais altas, temos:

$$
dy = \frac{1}{2} \frac{dx}{\sqrt{x}} = \frac{1}{2} x^{-\frac{1}{2}} \cdot dx,
$$

e, finalmente temos:

$$
\frac{dy}{dx} = \frac{1}{2} x^{-\frac{1}{2}}.
$$

Concordando com a regra geral.

Para resumir: chegamos à seguinte regra: *Para diferenciar $x^n$, multiplique pela potência e reduza a potência em um, resultando em $nx^{n-1}$ como resultado*.

### EXERCÍCIOS I

Resolva os Exercícios a seguir usando apenas as técnicas algébricas que vimos até o momento:

1. $y = x^{13}$  Resposta: $\frac{dy}{dx} =13x^{12}$
2. $y = x^{-\frac{3}{2}}$ Resposta: $\frac{dy}{dx} = -\frac{3}{2} x^{-\frac{5}{2}}$
3. $y = x^{2a}$ Resposta: $\frac{dy}{dx} =2a x^{2a-1}$
4. $u = t^{2.4}$ Resposta: $\frac{dy}{dx} =2.4 t^{1.4}$
5. $z = \sqrt[3]{u}$ Resposta: $z = $\frac{dy}{dx} =\frac{1}{3} u^{-\frac{2}{3}}$
6. $y = \sqrt[3]{x^{-5}}$ Resposta:  $\frac{dy}{dx} =-\frac{5}{3} x^{-\frac{8}{3}}$
7. $u = \sqrt{\frac{1}{x^8}}$ Resposta: $\frac{dy}{dx} =\frac{du}{dx} = -4x^{-5}$
8. $y = 2x^{a}$ Resposta: $\frac{dy}{dx} = 2a x^{a-1}$
9. $y = \sqrt[3]{x^3}$ Resposta: $\frac{dy}{dx} = 1$
10. $y = \sqrt{\frac{1}{x^m}}$ Resposta: $\frac{dy}{dx} = -\frac{m}{2} x^{-\frac{m+2}{2}}$

Você agora aprendeu como diferenciar potências de $x$. E viu como é fácil usando apenas a álgebra que aprendeu no ensino médio. Estes exercícios estão resolvidos [aqui](#EX1)

## Nº 5. Próxima Etapa. O Que Fazer com Constantes

Em nossas equações, consideramos \(x\) como crescente, e como resultado de \(x\) estar crescendo, \(y\) também mudou seu valor e cresceu. Nós geralmente pensamos em \(x\) como uma quantidade que podemos variar; e, considerando a variação de \(x\) como uma espécie de causa, consideramos a variação resultante de \(y\) como um efeito. Em outras palavras, consideramos o valor de \(y\) como dependente daquele de \(x\). Tanto \(x\) quanto \(y\) são variáveis, mas \(x\) é aquela que operamos, e \(y\) é a "variável dependente". Em todo o capítulo anterior, temos tentado encontrar regras para a proporção que a variação dependente em \(y\) tem em relação à variação independentemente feita em \(x\).

Nosso próximo passo é descobrir qual efeito no processo de diferenciação é causado pela presença de constantes, isto é, de números que não mudam quando \(x\) ou \(y\) mudam seus valores.

### Constantes Adicionadas

Vamos começar com um caso simples de uma constante adicionada, assim: 

$$
y = x^3 + 5
$$

Assim como antes, suponha que \(x\) cresça para \(x + dx\) e \(y\) cresça para \(y + dy\).





## Notas de Rodapé

[^1]:SWIFT, Dean. **On Poetry: a Rhapsody**, p. 20, impresso em 1733 — geralmente citado incorretamente.

[^2]:Nesta posição a relação entre $dx$ e $dy$ será 1, isso irá ocorrer quando os lados do triângulo formado entre a escada e o chão for de $45^\circ$.

[^3]A forma clássica do teorema binomial, com coeficientes binomiais e combinações, foi desenvolvida, ao longo do tempo, por matemáticos como Al-Karaji, Jia Xian e Omar Khayyam antes de Newton. Isaac Newton, por sua vez, generalizou o teorema para incluir potências fracionárias e negativas, expandindo seu alcance e aplicações.

## Apêndice 1 - Binômio de Newton

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
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k
$$

ou de forma equivalentemente, alguns livros grafam como:

$$
(x + y)^n = \sum_{k=0}^{n} C(n, k) x^{n-k} y^k
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
(x + y)^n = \sum_{k=0}^{\infty} \binom{n}{k} x^{n-k} y^k
$$

Onde o coeficiente binomial generalizado $\binom{n}{k}$ é definido por:

$$
\binom{n}{k} = \frac{n (n-1) (n-2) \cdots (n-k+1)}{k!}
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

## Exercícios Resolvidos

### EXERCÍCIOS I

1. $y = x^{13}$

$$
y + dy = (x + dx)^{13}
$$

Expandindo pelo binômio de Newton, obtemos:

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

Você agora aprendeu como diferenciar potências de $x$. Como é fácil!
