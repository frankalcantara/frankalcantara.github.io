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
slug: 2024-07-06T19:30:41.459Z

draft: 2024-07-06T19:30:40.256Z
---

O livro **Calculus Made Easy** de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson) foi colocado em domínio público, tanto no Brasil quanto dos EUA. Este é um dos melhores livros introdutórios de cálculo já escrito. Simples, direto e abrangente. Sem nenhuma modéstia, ou vergonha na cara, escolhi este livro para tradução, atualização e expansão. Vou tentar ser o mais fiel possível ao autor. Mas, vou atualizar todo o conteúdo e expandir alguns conceitos e, principalmente, o número de exercícios resolvidos. Contudo, como o livro é ótimo, algumas coisas não podem ser deixadas de lado, como a observação da capa:

    O QUE UM TOLO PODE FAZER, OUTRO TAMBÉM PODE.
    (Provérbio Simiano Antigo.)

Em 1910 quando **Calculus Made Easy** foi publicado, ainda era aceitável e interessante, brincar e se divertir quando estudávamos matemática, ou ciência. Talvez seja por isso que este livro é lembrado por tantos professores com tanto carinho. Então, vamos ao prólogo de [SILVANUS P. THOMPSON](https://en.wikipedia.org/wiki/Silvanus_P._Thompson), com o máximo de fidelidade que consegue o tolo que escreve.

"Considerando quantos tolos conseguem calcular, é surpreendente que se pense que é uma tarefa difícil, ou tediosa, para qualquer outro tolo aprender a dominar os mesmos truques.

Alguns truques de cálculo são bastante fáceis. Alguns são enormemente difíceis. Os tolos que escrevem os livros de texto de matemática avançada — e eles são na maioria tolos inteligentes — raramente se dão ao trabalho de mostrar como os cálculos fáceis são fáceis. Pelo contrário, eles parecem desejar impressioná-lo com sua tremenda astúcia, abordando-o da maneira mais difícil.

Sendo eu mesmo um sujeito notavelmente estúpido, tive que desaprender as dificuldades, e agora peço para apresentar aos meus colegas tolos as partes que não são difíceis. Domine estas completamente, e o resto seguirá. O que um tolo pode fazer, outro também pode."

## № 1. PARA LIVRÁ-LO DOS TERRORES PRELIMINARES

O terror preliminar, que sufoca a maioria dos meninos da quinta série de sequer tentar aprender cálculo, pode ser abolido de uma vez por todas simplesmente declarando qual é o significado — em termos de bom senso — dos dois símbolos principais que são usados no cálculo. Esses símbolos terríveis são:

(1) $d$ que significa meramente *um pedacinho de*. Ou se preferir, com um pouco mais de formalidade *uma fração muito pequena de*.

Assim, $dx$ significa um pedacinho de $x$; ou $du$ significa um pedacinho de $u$. Matemáticos comuns acham mais educado dizer *um elemento de* em vez de *um pedacinho de*, ou ainda uma *fração infinitesimal de*. Esta coisa de *infinitesimal* quer dizer que é tão pequeno que quase se confunde com o zero. Você verá que esses pequenos pedaços podem ser considerados infinitamente pequenos.

(2) $\int $ que é apenas um *S* longo, e pode ser chamado (se você quiser) de *a soma de*.

Assim, $\int dx$ significa a soma de todos os pequenos pedaços de $x$; ou $\int dt$ significa a soma de todos os pequenos pedaços de $t$. Matemáticos comuns chamam esse símbolo de *a integral de*. Agora, qualquer tolo pode ver que se $x$ for considerado como composto por muitos pequenos pedaços, cada um dos quais é chamado de $dx$, se você somá-los, obterá a soma de todos os $dx$'s (que é a mesma coisa que o todo de $x$). A palavra *integral* simplesmente significa *o todo* e tem o mesmo sentido de *somar todas as pequenas partes que compõem o todo*.

Se você pensar na duração de uma hora, pode, se quiser, pensar este intervalo de tempo como sendo dividido em 3600 pequenos pedaços chamados segundos. O todo dos 3600 pequenos pedaços somados juntos é uma hora.

Quando você vê uma expressão que começa com esse símbolo aterrorizante, $\int $, você saberá, daqui em diante, que ele está lá apenas para lhe dar instruções de que agora você deve realizar a operação, sempre e se puder, de somar todos os pequenos pedaços indicados pelos símbolos que o seguem.

E é só isso!

## № 2.SOBRE DIFERENTES GRAUS DE PEQUENEZ

Descobriremos que em nossos processos de cálculo temos que lidar com pequenas quantidades de vários graus de pequenez. Um pequeno pode ser menor que outro.

Teremos também que aprender em que circunstâncias podemos considerar pequenas quantidades tão minúsculas que podemos omiti-las das nossas considerações. Tudo depende da pequenez relativa.

    Esta pequenez relativa é uma expressão original de Sylvanos Thompson. Ele estava tentando ser o mais simples possível, para ficar claro, ele explicará que aquilo que consideramos pequeno é relativo ao problema que estamos tentando resolver. Ou, nem tudo que é pequeno para você o será para mim.

Antes de fixarmos quaisquer regras, pensemos em alguns casos familiares. Existem $60$ minutos em uma hora, $24$ horas no dia, $7$ dias na semana. Portanto, existem $1440$ minutos no dia e $10080$ minutos na semana.

Obviamente, $1$ minuto é uma quantidade muito pequena de tempo comparada a uma semana inteira. Na verdade, nossos antepassados consideravam pequeno em comparação com uma hora, e chamavam de *um minuto,* significando uma fração minúscula — um sexagésimo de uma hora. Quando passaram a exigir subdivisões ainda menores de tempo, dividiram cada minuto em $60$ partes menores, que, nos dias da **Rainha Elizabeth I**, chamavam de *segundos minutos* (isto é, a sexagésima parte da sexagésima parte da hora).

Agora, se um minuto é tão pequeno em comparação com um dia inteiro, quão menor em comparação é um segundo!

    Aqui precisamos de um pouco de contexto antes de prosseguirmos s: Um *farthing* era uma moeda de baixo valor usada na Inglaterra, equivalente a um quarto de um *penny*. A palavra *farthing* deriva do inglês antigo *feorthing*, que significa *quarta parte*. No sistema monetário britânico pré-decimal, havia 240 pennies em uma libra esterlina (£1), e portanto, um *farthing* era igual a $\frac{1}{960}$ de uma libra. Havia ainda uma libra de ouro, chamada de *soberano*. Devido ao seu valor extremamente baixo, o *farthing* foi usado para transações de pequeno valor até ser desmonetizado em 1961. SILVANUS P. THOMPSON, nosso autor escreveu este livro em 1910 e neste tempo, ainda existiam *farthings*. Eu achei isso interessante demais para suprimir ou atualizar.

Novamente, pense em um *farthing* em comparação com um *soberano*: ele, o *farthing* vale menos que $\frac{1}{1000}$ parte. Um *farthing* a mais ou a menos tem pouca importância em comparação com um *soberano* e, certamente, pode ser considerado uma quantidade *pequena*. Mas compare um *farthing* com £1000: relativamente a esta quantia maior, o *farthing* não tem mais importância do que $\frac{1}{1000}$ de um *farthing* teria para um soberano. Mesmo um *soberano* de ouro é relativamente uma quantidade insignificante na riqueza de um milionário.

Agora, se fixarmos qualquer fração numérica como constituindo a proporção que, para qualquer propósito, chamamos relativamente pequena, podemos facilmente declarar outras frações de um grau mais elevado de pequenez. Assim, se, para o propósito do tempo, $\frac{1}{60}$ for chamada de uma *fração pequena*, então $\frac{1}{60}$ de $\frac{1}{60}$ (sendo uma *pequena fração de uma pequena fração*) pode ser considerada uma *quantidade pequena da segunda ordem de pequenez*.

*Os matemáticos falam sobre a segunda ordem de "magnitude" (isto é: grandeza) quando realmente querem dizer segunda ordem de *pequenez*. Isso é muito confuso para os iniciantes.

Ou, se para qualquer propósito pegássemos $1$ por cento (isto é: $\frac{1}{100}$) como uma *fração pequena*, então $1$ por cento de $1$ por cento (isto é: $\frac{1}{10.000}$) seria uma pequena fração ainda menor de *segunda ordem em redução*; e $\frac{1}{1.000.000}$ seria uma pequena fração da *terceira ordem em redução*, sendo $1$ por cento de $1$ por cento de $1$ por cento.

    A melhor tradução do termo usado pelo autor para *smallness*, em português é pequenez, mas este é um termo pouco usado neste começo de século XXI. Na verdade, Thompson está destacando a insignificância das coisas enquanto as dividimos em pedaços cada vez menores mantendo a mesma ordem de divisão.

Por fim, suponha que, para algum propósito muito preciso, devamos considerar $\frac{1}{1.000.000}$ como *pequeno*. Assim, se um cronômetro de primeira categoria não deve perder ou ganhar mais de meio minuto em um ano, ele deve manter o tempo com uma precisão de $1$ parte em $1.051.200$. Agora, se, para tal propósito, considerarmos $\frac{1}{1.000.000}$ (ou um milionésimo) como uma quantidade pequena, então $\frac{1}{1.000.000}$ de $\frac{1}{1.000.000}$, ou seja, $\frac{1}{1.000.000.000.000}$ (ou um trilionésimo) será uma quantidade pequena da *segunda ordem de pequenez*, e pode ser completamente desconsiderada.

Então, vemos que quanto menor for uma quantidade pequena em si, mais insignificante se torna a correspondente quantidade pequena da segunda ordem. Portanto, sabemos que *em todos os casos estamos justificados em negligenciar as pequenas quantidades da segunda ou terceira (ou maior) ordens, se apenas considerarmos a pequena quantidade da primeira ordem suficientemente pequena em si mesma*. Um valor suficientemente pequeno que permita resolver o problema que lhe interessa resolver com o cálculo.

Mas, deve ser lembrado, que pequenas quantidades, se ocorrerem em nossas expressões como fatores multiplicados por algum outro fator, podem se tornar relevantes se o outro fator for em si grande. Mesmo um *farthing* se torna importante se for multiplicado por centenas, milhares ou milhões de vezes.

No cálculo, escrevemos $dx$ para representar um pequeno pedaço de $x$. Essas coisas como $dx$, e $du$, e $dy$, são chamadas de *diferenciais*, e lidas como o diferencial de $x$, ou o diferencial de $u$, ou o diferencial de $y$, conforme o caso. Ou ainda, a derivada de $x$, a derivada de $y$ ou a derivada de$u$. Se $dx$ for um pequeno pedaço de $x$, e relativamente pequeno em si mesmo, isso não implica que tais quantidades como $x \cdot dx$, ou $x^2 \, dx$, ou $a^x \, dx$ sejam insignificantes. Mas implica que $dx \times dx$ seria insignificante, já que seria uma quantidade pequena da segunda ordem.

Um exemplo muito simples servirá como ilustração.

Vamos pensar em $x$ como uma quantidade que pode crescer um pouco de modo a se tornar $x + dx$, onde $dx$ é o pequeno incremento adicionado pelo crescimento. O quadrado de $x + dx$ será $x^2 + 2xdx + (dx)^2$. O segundo termo não é desprezível porque é uma quantidade de primeira ordem; enquanto o terceiro termo é da segunda ordem de pequenez, $(dx)^2 = dx\times dx$. Para ficar claro, se fizer $dx$ ter um valor numérico, digamos, $\frac{1}{60}$ de $x$, então o segundo termo seria $\frac{2x^2}{60}$, enquanto o terceiro termo seria $\frac{1}{3600}$. Este último termo é claramente menos importante que o segundo. Mas, se formos além, podemos considerar $dx$ como $\frac{1}{1000}$ de $x$, então o segundo termo será $\frac{2x^2}{1000}$, enquanto o terceiro termo será apenas $\frac{1}{1.000.000}$.

**FIG 1 E FIG 2  Já está no One drive**

Geometricamente, isso pode ser representado da seguinte forma: Desenhe um quadrado (Fig. 1) cujo lado representaremos como $x$. Agora, suponha que o quadrado cresça adicionando uma pequena quantidade, um $dx$, ao seu tamanho em cada direção. O quadrado ampliado será composto pelo quadrado original $x^2$, os dois retângulos na parte superior e na direita, cada um com área $xdx$ (ou juntos $2xdx$), e o pequeno quadrado no canto superior direito que é $(dx)^2$. Na Figura 2, representamos $dx$ como uma fração muito grande de $x$ — cerca de $\frac{1}{5}$ por razões didáticas e gráficas. Mas suponha que façamos $dx$ como $\frac{1}{100}$ — aproximadamente a espessura de uma linha desenhada com uma caneta fina. Então, o pequeno quadrado no canto superior direito terá uma área de apenas $\frac{1}{10,000}$ de $x^2$, e será praticamente invisível. Claramente, $(dx)^2$ é desprezível sempre que considerarmos o incremento $dx$ como suficientemente pequeno.

FIG 3

Vamos considerar uma analogia.

Suponha que um milionário dissesse ao seu secretário: na próxima semana, eu lhe darei uma pequena fração de qualquer valor monetário que eu receba. Suponha que o secretário dissesse ao seu filho: eu lhe darei uma pequena fração do que eu receber. Suponha que a fração em cada caso seja $\frac{1}{100}$. Agora, se o Sr. Milionário recebesse £1000 durante a próxima semana, o secretário receberia £10 e o garoto 2 xelins.

    Um *xelin* é um 20 avos de uma libra, então, £10 correspondem a 200 *xelins*. Um centésimo de 200 *xelins* é 2 *xelins*.

Dez libras seriam uma quantidade pequena em comparação com £1000; mas dois xelins é realmente uma quantidade muito pequena, uma quantidade de segunda ordem. Mas qual seria a desproporção se a fração, em vez de ser $\frac{1}{100}$, fosse $\frac{1}{1000}$? Então, enquanto o Sr. Milionário recebesse suas £1000, o Sr. Secretário receberia apenas £1, e o garoto menos de um *farthing*!

O espirituoso Dean Swift* uma vez escreveu:

So, Nat'ralists observe, a Flea
Hath smaller Fleas that on him prey.
And these have smaller Fleas to bite 'em,
And so proceed ad infinitum.

*On Poetry: a Rhapsody* (p. 20), impresso em 1733 — geralmente citado incorretamente.

Um boi pode se preocupar com uma pulga de tamanho comum — uma pequena criatura da primeira ordem de pequenez. Mas provavelmente não se incomodaria com a pulga de uma pulga; sendo da segunda ordem de pequenez, seria insignificante. Mesmo uma grande quantidade de pulgas de pulgas não teria muita importância para o boi.

## № 3. SOBRE CRESCIMENTOS RELATIVOS

Durante todo o cálculo, lidamos com quantidades que estão variando, aumentando ou diminuindo e com suas taxas de variação. Classificaremos todas as quantidades em duas classes: *constantes* e *variáveis*. Aquelas que consideramos de valor fixo, e chamamos de *constantes*, geralmente denotamos algebricamente por letras do início do alfabeto latino, como $a$, $b$ ou $c$; enquanto aquelas que consideramos capazes de crescer, ou (como dizem os matemáticos) de "variar", denotamos por letras do final do alfabeto, como $x$, $y$, $z$, $u$, $v$, $w$ ou, às vezes, $t$.

Além disso, geralmente lidamos com mais de uma variável ao mesmo tempo e pensamos na forma como uma variável depende da outra: por exemplo, pensamos na forma como a altura atingida por um projétil depende do tempo necessário para atingir essa altura. Ou somos convidados a considerar um retângulo de área dada e a perguntar de quer forma qualquer aumento no comprimento dele implicará uma diminuição correspondente na largura dele, para manter a área constante. Ou pensamos na forma como qualquer variação na inclinação de uma escada causará uma variação na altura que ela atinge.

Suponha que tenhamos duas variáveis que dependem uma da outra. Uma alteração em uma causará uma alteração na outra, *por causa* dessa dependência. Vamos chamar uma das variáveis de $x$, e a outra que depende dela de $y$.

    Dizemos que $x$ é a variável independente, $y$ a variável dependente.

Suponha que façamos $x$ variar, ou seja, alteramos ou imaginamos que ela foi alterada, adicionando a ela um pequeno pedaço do seu valor que chamamos de $dx$. Assim, estamos fazendo $x$ se tornar $x + dx$. Então, porque $x$ foi alterado, $y$ também terá sido alterado, e terá se tornado $y + dy$. Aqui, o pouco $dy$ pode ser em alguns casos positivo, em outros negativo; e não será (exceto por um milagre) do mesmo tamanho que $dx$.

**FALTA A FIGURA 4**

Considere dois exemplos.

(1) Vamos fazer com que $x$ e $y$ sejam, respectivamente, a base e a altura de um triângulo retângulo (Fig. 4), cuja inclinação do outro lado está fixada em $30^\circ$. Se supusermos que este triângulo se expande e ainda mantém seus ângulos constantes, então, quando a base cresce de modo a se tornar $x + dx$, a altura se torna $y + dy$. Aqui, o aumento de $x$ resulta em um aumento de $y$. O pequeno triângulo, cuja altura é $dy$, e cuja base é $dx$, é semelhante ao triângulo original; e é óbvio que o valor da razão $\frac{dy}{dx}$ é o mesmo da razão $\frac{y}{x}$. Como o ângulo é $30^\circ$, será visto que aqui

$$
\frac{dy}{dx} = \frac{1}{1.73}.
$$

(2) Agora façamos $x$ representar, na Figura 5, a distância horizontal, a partir de uma parede, da extremidade inferior de uma escada, $AB$, de comprimento fixo; neste caso, $y$ será a altura que a escada atinge na parede. Agora, $y$ claramente depende de $x$. É fácil ver que, se puxarmos a extremidade inferior $A$ um pouco mais para longe da parede, a extremidade superior $B$ descerá um pouco. Vamos afirmar isso com um pouco mais de formalidade. Se aumentarmos $x$ para $x + dx$, então $y$ se tornará $y - dy$; isto é, quando $x$ recebe um incremento positivo, o incremento que resulta em $y$ é negativo.

**FALTA A FIGURA 5**

Sim, mas quanto? Suponha que a escada fosse tão longa que, quando a extremidade inferior $A$ estivesse a $50$ centímetros da parede, a extremidade superior $B$ alcançasse $4.5$ metros do chão. Agora, se você puxasse a extremidade inferior $2.5$ centímetros a mais, quanto a extremidade superior desceria? Coloque tudo em metros: $x = 0.5$ metros, $y = 4.5$ metros. Agora, o incremento de $x$ que chamamos de $dx$ é de $0.025$ metros: ou $x + dx = 0.525$ metros.

Quanto $y$ será diminuído? A nova altura será $y - dy$. Se calcularmos a altura pelo Teorema de Euclides I. 47, então poderemos descobrir quanto será $dy$. O comprimento da escada é

$$
\sqrt{(4.5)^2 + (0.5)^2} = 4.52769 \text{ metros}.
$$

Claramente, então, a nova altura, que é $y - dy$, será tal que

$$
(y - dy)^2 = (4.52769)^2 - (0.525)^2 = 20.50026 - 0.27563 = 20.22463,
$$
$$
y - dy = \sqrt{20.22463} = 4.49715 \text{ metros}.
$$

Agora $y = 4.50000$, então $dy$ é $4.50000 - 4.49715 = 0.00285$ metros.

Vemos que fazer $dx$ um aumento de $0.025$ metros resultou em fazer $dy$ uma diminuição de $0.00504$ metros.

E a razão de $dy$ para $dx$ pode ser declarada da seguinte forma:

$$
\frac{dy}{dx} = \frac{0.00285}{0.025} = 0.11392.
$$

Também é fácil ver que (exceto em uma posição particular) $dy$ terá um tamanho diferente de $dx$.

Agora, usando o cálculo diferencial, estamos buscando uma coisa curiosa, uma mera razão, a saber: a proporção que $dy$ tem em relação a $dx$ quando ambos são indefinidamente pequenos, dizemos quando ambos, $dx$ e $dy$, são infinitesimais.

Deve-se notar aqui que só podemos encontrar essa razão $\frac{dy}{dx}$ quando $y$ e $x$ estão relacionados de alguma forma, de modo que sempre que $x$ varia, $y$ também varia. No primeiro exemplo recém mencionado, se a base $x$ do triângulo for aumentada, a altura $y$ do triângulo também se tornará maior, e no segundo exemplo, se a distância $x$ do pé da escada à parede for aumentada, a altura $y$ atingida pela escada diminuirá de maneira correspondente, inicialmente devagar, mas mais e mais rapidamente à medida que $x$ se torna maior. Nestes casos, a relação entre $x$ e $y$ é perfeitamente definida, podendo ser expressa matematicamente, sendo $\frac{y}{x} = \tan 30^\circ$ e $x^2 + y^2 = l^2$ (onde $l$ é o comprimento da escada) respectivamente, e $\frac{dy}{dx}$ tem o significado que encontramos em cada caso.

Se, enquanto $x$ é, como antes, a distância do pé da escada à parede, $y$ é, em vez da altura alcançada, o comprimento horizontal da parede, ou o número de tijolos nela, ou o número de anos desde que foi construída, qualquer mudança em $x$ naturalmente não causaria nenhuma mudança em $y$; neste caso, $\frac{dy}{dx}$ não tem significado algum, e não é possível encontrar uma expressão para isso.

Sempre que usamos diferenciais $dx$, $dy$, $dz$, etc., a existência de algum tipo de relação entre $x$, $y$, $z$, etc., é implícita, e essa relação é chamada de *função* em $x$, $y$, $z$, etc.; as duas expressões dadas acima, por exemplo, a saber $\frac{y}{x} = \tan 30^\circ$ e $x^2 + y^2 = l^2$, são funções de $x$ e $y$. Tais expressões contêm implicitamente (isto é, contêm sem mostrar distintamente) os meios de expressar $x$ em termos de $y$ ou $y$ em termos de $x$, e por essa razão são chamadas de funções implícitas em $x$ e $y$; elas podem ser respectivamente colocadas nas formas

$$y = x \tan 30^\circ \quad \text{ou} \quad x = \frac{y}{\tan 30^\circ}$$

$$\text{e} \quad y = \sqrt{l^2 - x^2} \quad \text{ou} \quad x = \sqrt{l^2 - y^2}.$$

Essas últimas expressões afirmam explicitamente o valor de $x$ em termos de $y$, ou de $y$ em termos de $x$, e por essa razão são chamadas de *funções explícitas* de $x$ ou $y$. Por exemplo, $x^2 + 3 = 2y - 7$ é uma função implícita em $x$ e $y$; pode ser escrita $y = \frac{x^2 + 10}{2}$ (função explícita de $x$) ou $x = \sqrt{2y - 10}$ (função explícita de $y$).

Vemos que uma função explícita em $x$, $y$, $z$, etc., é simplesmente algo cujo valor muda quando $x$, $y$, $z$, etc., estão mudando. Por causa disso, o valor da função explícita é chamado de *variável dependente*, pois depende do valor das outras quantidades variáveis na função; essas outras variáveis são chamadas de *variáveis independentes* porque seu valor não é determinado pelo valor assumido pela função. Por exemplo, se $u = x^2 \sin \theta$, $x$ e $\theta$ são as variáveis independentes, e $u$ é a variável dependente.

Às vezes, a relação exata entre várias quantidades $x$, $y$, $z$ ou não é conhecida ou não é conveniente afirmá-la; é apenas conhecido, ou conveniente afirmar, que há algum tipo de relação entre essas variáveis, de modo que não se pode alterar $x$ ou $y$ ou $z$ individualmente sem afetar as outras quantidades; a existência de uma função em $x$, $y$, $z$ é então indicada pela notação $F(x, y, z)$ (função implícita) ou por $x = F(y, z)$, $y = F(x, z)$ ou $z = F(x, y)$ (funções explícitas). Às vezes, a letra $f$ ou $\phi$ é usada em vez de $F$, de modo que $y = F(x)$, $y = f(x)$ e $y = \phi(x)$ geralmente significam a mesma coisa, ou seja, que o valor de $y$ depende do valor de $x$ de alguma maneira que não está sendo afirmada.

Chamamos a razão $\frac{dy}{dx}$ de *derivada de $y$ com respeito a $x$* formalmente deveríamos chamar de *coeficiente diferencial de $y$ com respeito a $x$*, mas derivada basta. Ainda assim, é um nome científico, solene, para uma coisa muito simples. Mas não vamos nos assustar com nomes solenes, quando as coisas em si são tão fáceis. Em vez de nos assustarmos, simplesmente pronunciaremos uma breve maldição sobre a estupidez de dar nomes longos e complicados; e, tendo aliviado nossas mentes, passaremos à coisa simples em si, encontrar a razão $\frac{dy}{dx}$.

Na álgebra comum que você aprendeu na escola, você estava sempre tentando encontrar alguma quantidade desconhecida que você chamava de $x$ ou $y$. Agora você tem que aprender a procurar estas quantidades de uma forma nova. O processo de encontrar o valor de $\frac{dy}{dx}$ é chamado de *diferenciação*. Mas, lembre-se, o que se deseja é o valor dessa razão quando tanto $dy$ quanto $dx$ são indefinidamente pequenos, esta é a última vez que vou lembrar isso: estamos preocupados em encontrar a razão $\frac{dy}{dx}$ quando $dx$ e $dy$ são infinitesimalmente pequenos, tão pequenos que tendem a zero.

Vamos agora aprender como buscar $\frac{dy}{dx}$.

## Como ler diferenciais

A amável leitora nunca deve cometer o erro de pensar que $dx$ significa $d$ vezes $x$, pois $d$ não é um fator – significa *um elemento de* ou *um pedacinho de* qualquer coisa que se siga com este mesmo sentido. Lê-se $dx$ assim: *de-xis*.

Caso a leitora não tenha ninguém para guiá-lo em tais assuntos, pode-se simplesmente dizer que se lê os coeficientes diferenciais da seguinte maneira. O coeficiente diferencial $\frac{dy}{dx}$ é lido *de-ipsilon de-xis*. Ou seja, $\frac{du}{dt}$ é lido *de-u de-te*. Caramba! Como é difícil tentar escrever a pronúncia.

Coeficientes diferenciais de segunda ordem, que serão encontrados mais tarde. Serão representados: $\frac{d^2y}{dx^2}$, o que é lido *de-dois-ipsilon de-xis-quadrado*, significando que operação de diferenciar $y$ em relação a $x$ foi, ou tem que ser, realizada duas vezes consecutivas. Esta é a notação de Leibnitz.

Outra maneira de indicar que uma função foi diferenciada é colocando um acento no símbolo da função. Assim, se $y = F(x)$, o que significa que $y$ é alguma função não especificada de $x$ (veja aqui), podemos escrever $F'(x)$ em vez de $\frac{d(F(x))}{dx}$. Da mesma forma, $F''(x)$ significará que a função original $F(x)$ foi diferenciada duas vezes com respeito a $x$. Esta é a notação da Lagrange.

Vamos usar as duas notações livremente neste texto.
