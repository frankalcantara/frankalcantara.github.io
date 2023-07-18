---
title: Introdução a Programação Lógica
layout: post
author: Frank
description: Uma aventura pelo universo matemático que fundamenta a programação lógica.
date: 2023-07-13T02:50:56.534Z
preview: ""
image: assets/images/prolog1.jpeg
tags:
  - Lógica
  - Programação Lógica
  - Prolog
categories:
  - disciplina
  - Lógica
  - Material de Aula
  - Matemática
rating: 5
slug: introducao-programacao-logica
keywords:
  - lógica
  - Programação
  - programação lógica
draft: true
---

A Programação Lógica é artefato de raciocínio capaz de ensinar um detetive computadorizado a resolver os mais intricados mistérios, permitindo que se preocupe apenas com o "o que" e deixando o "como" a cargo da máquina. É a base de alguns modelos computacionais que estão mudando o mundo.

> "Logic programming is the future of artificial intelligence." - [Marvin Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky)

# Introdução

Imagine, por um momento, que estamos explorando o universo dos computadores, mas em vez de sermos os comandantes, capazes de ditar todos os passos do caminho, nós fornecemos as diretrizes gerais e deixamos que o computador deduza o caminho. Pode parecer estranho, a princípio, para quem está envolvido com as linguagens do Paradigma Imperativo. Acredite, ou não, isso é exatamente o que a Programação Lógica faz.

Em vez de sermos forçados a ordenar cada detalhe do processo de solução de um problema, a Programação Lógica permite que declaremos o que queremos, e então deixar o computador fazer o trabalho de encontrar os detalhes e processos necessários para resolver cada problema.

Na **Programação Imperativa** partimos de uma determinada expressão e seguimos um conjunto de instruções até encontrar o resultado desejado. O programador fornece um conjunto de instruções que definem o fluxo de controle e modificam o estado da máquina. O foco está em **como** o problema deve ser resolvido passo a passo. Exemplos de linguagens imperativas incluem H++, Java e Python.

Na Programação Lógica, um dos paradigmas da **Programação Descritiva** usamos a dedução. Na Programação Descritiva, o programador fornece uma descrição lógica, ou funcional, de **o que** deve ser feito, sem especificar o fluxo de controle. O foco está no problema, não na solução. Exemplos incluem SQL, Prolog e Haskell. Na Programação Lógica, partimos de uma conjectura e, de acordo com um conjunto específico de regras, tentamos construir uma prova para esta conjectura.

Uma conjectura é uma suposição, ou proposição que é acreditada ser verdadeira mas ainda não foi provada. Na linguagem natural, conjecturas são frequentemente expressas como declarações que precisam de confirmação adicional. Na Lógica de Primeira Ordem, as conjecturas são tratadas como sentenças que são propostas para serem verdadeiras. Essas sentenças podem ser analisadas e testadas usando as regras e estruturas da Lógica de Primeira Ordem.

![Diagrama de Significado de Conjeturas](/assets/images/conjecturas.jpeg)

Em resumo: **Imperativa:** focada no processo, no "como" chegar à solução; **Descritiva:** focada no problema em si, no "o que" precisa ser feito.

A escolha entre estes paradigmas dependerá da aplicação e do estilo do programador. Mas o futuro parece cada vez mais orientado para linguagens declarativas e descritivas, que permitem ao programador concentrar-se no problema, não nos detalhes da solução. Efeito que parece ser evidente se considerarmos os avanços recentes no campo da inteligência artificial.

Em nossa exploração, vamos começar com a Lógica de Primeira Ordem, a qual iremos subdividir em elementos menores e interligados. É importante notar que muitos no campo acadêmico podem não distinguir as sutilezas que diferenciam a Lógica de Primeira Ordem da Lógica Predicativa. Neste Proposição, iremos decompor a Lógica de Primeira Ordem em suas partes componentes, examinando cada uma como uma entidade distinta. E, para iniciar nossa jornada, utilizaremos a Lógica Proposicional como alicerce para estabelecer o raciocínio.

A Lógica Proposicional é um tipo de linguagem matemática suficientemente rica para expressar muitos dos problemas que precisamos resolver e suficientemente gerenciável para que os computadores possam lidar com ela. Uma ferramenta útil tanto ao homem quanto a máquina. Quando esta ferramenta estiver conhecida mergulharemos no espírito da Lógica de Primeira Ordem, a Lógica Predicativa, ou Lógica de Predicados, e então poderemos fazer sentido do mundo.

Vamos enfrentar a inferência e a dedução, duas ferramentas para extração de conhecimento de declarações lógicas. Voltando a metáfora do Detetive, podemos dizer que a inferência é quase como um detetive que tira conclusões a partir de pistas: você tem algumas verdades e precisa descobrir outras verdades que são consequências diretas das primeiras verdades.

Vamos falar da Cláusula de Horn, um conceito um pouco mais estranho. Uma regra que torna todos os problemas expressos em lógica mais fácies de resolver. É como uma receita de bolo que, se corretamente seguida, torna o processo de cozinhar muito mais simples.

No final do dia, tudo que queremos, desde os tempos de [Gödel](https://en.wikipedia.org/wiki/Kurt_Gödel), [Turing](https://en.wikipedia.org/wiki/Alan_Turing) e [Church](https://en.wikipedia.org/wiki/Alonzo_Church) é que nossas máquinas sejam capazes de resolver problemas complexos com o mínimo de interferência nossa. Queremos que eles pensem, ou pelo menos, que simulem o pensamento. E a Programação Lógica é uma maneira deveras interessante de perseguir este objetivo.

A Programação Lógica aparece em meados dos anos 1970 como uma evolução dos esforços das pesquisas sobre a prova computacional de teoremas matemáticos e inteligência artificial. Deste esforço surgiu a esperança de que poderíamos usar a lógica como um linguagem de programação, em inglês, "programming logic" ou Prolog. Este artigo faz parte de uma série sobre a Programação Lógica, partiremos da base matemática e chegaremos ao Prolog.

# Lógica de Primeira Ordem

A Lógica de Primeira Ordem é um dos fundamentos essenciais da ciência da computação e, consequentemente, da programação. Essa matemática permite quantificar sobre objetos, fazer declarações que se aplicam a todos os membros de um conjunto ou a um membro particular desse conjunto. Por outro lado, nos impede de quantificar diretamente sobre predicados ou funções.

Essa restrição não é um defeito, mas sim um equilíbrio cuidadoso entre poder expressivo e simplicidade computacional. Dá-nos uma maneira de formular uma grande variedade de problemas, sem tornar o processo de resolução desses problemas excessivamente complexo.

A Lógica de Primeira Ordem é o nosso ponto de partida, nossa base, nossa pedra fundamental. Uma forma poderosa e útil de olhar para o universo, não tão complicada que seja hermética a olhos leigos, mas suficientemente complexa para permitir a descoberta de alguns dos mistérios da matemática e, no processo, resolver alguns problemas práticos.

A Lógica de Primeira Ordem consiste de uma linguagem, consequentemente criada sobre um alfabeto $\Sigma$, de um conjunto de axiomas e de um conjunto de regras de inferência. Esta linguagem consiste de todas as fórmulas bem formadas da teoria da Lógica Proposicional e predicativa. O conjunto de axiomas é um subconjunto do conjunto de fórmulas bem formadas acrescido e, finalmente, um conjunto de regras de inferência.

O alfabeto $\Sigma$ pode ser dividido em conjuntos de símbolos agrupados por classes:

**variáveis, constantes e símbolos de pontuação**: vamos usar os símbolos do alfabeto latino em minúsculas e alguns símbolos de pontuação. Destaque-se os símbolos $($ e $)$, parenteses, que usaremos para definir a prioridade de operações.

Vamos usar os símbolos $u$, $v$, $w$, $x$, $y$ e $z$ para indicar variáveis e $a$, $b$, $c$, $d$ e $e$ para indicar constantes.

**Funções**: usaremos os símbolos $\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$ e $\mathbf{i}$ para indicar funções.

**Predicados**: usaremos os símbolos $P$, $Q$, $\mathbf{r}$ e $S$ para indicar predicados.

**Operadores**: usaremos os símbolos tradicionais da Lógica Proposicional: $\neg$ (negação), $\wedge$ (disjunção, _and_), $\vee$ (conjunção, _or_), $\rightarrow$ (implicação) e $\leftrightarrow$ (equivalência).

**Quantificadores**: nos manteremos no limite da tradição matemática e usar $\exists$ (quantificador existencial) e $\forall$ (quantificador universal).

**Fórmulas Bem Formadas**: usaremos letras do alfabeto latino, maiúsculas para representar as Fórmulas Bem Formadas: $F$, $G$, $I$, $J$, $K$.

Na lógica matemática, uma Fórmula Bem Formada, também conhecida como expressão bem formada, é uma sequência **finita** de símbolos que é formada de acordo com as regras gramaticais de uma linguagem lógica específica.

Em lógica de primeira ordem, uma Fórmula Bem Formada é uma expressão que **só pode ser** verdadeira ou falsa. As Fórmulas Bem Formadas são compostas de quantificadores, variáveis, constantes, predicados, e conectivos lógicos, e devem obedecer a regras específicas de sintaxe.

Em qualquer linguagem matemática regra sintática mais importante é a precedência das operações, uma espécie de receita. Que deve ser seguida à letra. Vamos nos restringir a seguinte ordem de precedência:

$$\neg, \forall, \exists, \wedge, \vee, \rightarrow, \leftrightarrow$$

Dando maior precedência a $\neg$ e a menor a $\leftrightarrow$.

O uso dos parenteses e da ordem de precedência, com parcimônia, muita parcimônia, permite que possamos escrever $(\forall x(\exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))))$ ou $\forall x \exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))$ que são a mesma fórmula bem Formada. Escolha a opção que seja mais fácil de ler e entender.

Nesta linguagem cada sentença, ou preposição, deve ser verdadeira ou falsa, nunca verdadeira e falsa ao mesmo tempo e nada diferente de verdadeiro ou falso.

Para que uma sentença, ou preposição, seja verdadeira ela precisa ser logicamente verdadeira. Uma sentença que deve ser falsa é uma sentença contraditória.

Assim como aprendemos nossa língua materna reconhecendo padrões, repetições e regularidades, também reconhecemos Fórmulas Bem Formadas por seus padrões característicos. Os símbolos estarão dispostos de forma organizada e padronizada em termos sobre os quais serão aplicadas operações, funções e quantizadores.

Termos são variáveis, constantes ou mesmo funções aplicadas a termos e seguem um pequeno conjunto de regras:

1. uma variável $x$ é um termo em sí;
2. uma constante $a$ é um termo em si;
3. se $\mathbf{f}$ é uma função de termos $(t_1, ... t_2)$ então $\mathbf{f}(t_1, ... t_2)$ é um termo.

Cada proposição, ou sentença, na Lógica Proposicional é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. 'A chuva cai', 'O sol brilha' - cada uma dessas proposições é verdadeira ou falsa como uma unidade. Um átomo, elemento básico e fundamental de todas as expressões. Também, mas tarde, chamaremos de átomos a todo predicado aplicado aos termos de uma fórmula. Assim, também precisamos definir os predicados.

1. se $\mathbf{p}$ é um predicado de termos $(t_1, ... t_2)$ então $\mathbf{p}(t_1, ... t_2)$ é um fórmula bem Formada, um átomo.
2. se $F$ e $G$ são Fórmulas Bem Formadas então: $\neg F$, $F\wedge G$, $F \vee G$, $F \rightarrow G$ e $F \leftrightarrow G$ são Fórmulas Bem Formadas.
3. se $F$ é uma fórmula bem Formada e $x$ uma variável então $\exists x F$ e $\forall x F$ são Fórmulas Bem Formadas.

Por fim, podemos dizer que as Fórmulas Bem Formadas: respeitam regras de precedência entre conectivos, parênteses e quantificadores;
não apresentam problemas como variáveis livres não quantificadas e, principalmente, são unívocas, sem ambiguidade na interpretação.

Finalmente podemos dizer que a linguagem da Lógica de Primeira Ordem é o conjunto de todas as Fórmulas Bem Formadas incluindo os campos de estudo da Lógica Proposicional e da Lógica de Predicados. Termos e átomos costurados em uma teia onde cada termo, ou átomo, é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. 'A chuva cai', 'O sol brilha' - cada uma dessas proposições é verdadeira ou falsa como uma unidade. As operações lógicas são as pontes que conectam essas ilhas, permitindo-nos construir estruturas mais complexas de razão.

## Lógica Proposicional

Esse sistema, por vezes chamado de álgebra booleana, fundamental para o desenvolvimento da computação, é uma verdadeira tapeçaria de possibilidades. Na Lógica Proposicional, declarações atômicas, que só podem ter valores os verdadeiro, $T$, ou falso $F$, são entrelaçadas em declarações compostas cuja veracidade, segundo as regras desse cálculo, depende dos valores de verdade das declarações atômicas que as compõem quando sujeitas aos operadores, ou conectivos, que definimos anteriormente.

Vamos representar essas declarações atômicas por literais $F$, $G$, $X_1$, $X_2$ etc., e suas negações por $\neg F$, $\neg G$, $\neg X_1$, $\neg X_2$ etc. Todos os símbolos individuais e suas negações são conhecidas como literais.

As declarações atômicas e compostas são costuradas por conectivos para produzir declarações compostas, cujo valor de verdade depende dos valores de verdade das declarações componentes. Os conectivos que consideramos inicialmente, cuja Tabela Verdade será dada por:

<table style="margin-left: auto;
  margin-right: auto; text-align:center;">
  <tr style="border-top: 2px solid gray; border-bottom: 1px solid gray;">
    <th style="width:8%; border-right: 1px solid gray;">$F$</th>
    <th style="width:8%; border-right: double gray;">$G$</th> 
    <th style="width:16.8%; border-right: 1px solid gray;">$F \vee G$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$F \wedge G$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$\neg F$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$F \rightarrow G$</th>
    <th style="width:16.8%;">$F \leftrightarrow G$</th>
  </tr>
  <tr style="background-color: #eeeeee;">
    <td style="width:8%; border-right: 1px solid gray;">T</td>
    <td style="width:8%; border-right: double gray;">T</td> 
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%;">T</td>
  </tr>
  <tr>
    <td style="width:8%; border-right: 1px solid gray;">T</td>
    <td style="width:8%; border-right: double gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%;">F</td>
  </tr>
  <tr style="background-color: #eeeeee;">
    <td style="width:8%; border-right: 1px solid gray;">F</td>
    <td style="width:8%; border-right: double gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%;">F</td> 
  </tr>
  <tr style="border-bottom: 2px solid gray;">
    <td style="width:8%; border-right: 1px solid gray;">F</td>
    <td style="width:8%; border-right: double gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">F</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%; border-right: 1px solid gray;">T</td>
    <td style="width:16.8%;">T</td>
  </tr>
</table>
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Tabela 1 - Tabela Verdade, operadores básicos.</legend>

Quando aplicamos a Tabela Verdade a uma declaração composta, obtemos um procedimento capaz de determinar se declaração composta é verdadeira ou falsa. Para isso, tudo que temos que fazer é aplicar, segundo as regras de procedência, a Tabela Verdade a expressão, simplificando-a. Uma alternativa mais simples que a aplicação algébrica dos axiomas da Lógica Proposicional.

O operador $\vee$, também chamado de ou inclusivo é verdade apenas quando ambos os termos são verdadeiros. Diferindo de um operador, que por não ser básico e fundamenta, não consta da nossa lista, ou exclusivo, $\oplus$, falso se ambos os termos forem verdadeiros.

O condicional $\rightarrow$ não representa a implicação em nenhum sentido causal. Em particular, ele é definido como verdadeiro quando nenhum dos termos é verdadeiro, e é falso apenas quando o termo antecedente é verdadeiro e o consequente falso.

O bicondicional $\leftrightarrow$ equivale a ambos os componentes terem o mesmo valor-verdade. Todos os operadores, ou conectivos, conectam duas declarações, exceto $\neg$ que se aplica a apenas um termo.

Ainda observando a Tabela Verdade acima, é fácil perceber que se tivermos $4$ termos, em vez de $2$, teremos $2^4 = 16$. Se para uma determinada Fórmula Bem Formada todas os resultados forem verdadeiros, $T$, teremos uma _tautologia_, se todos forem falsos, $F$ uma _contradição_.

Uma _tautologia_ é uma fórmula que é sempre verdadeira, não importa a interpretação ou atribuição de valores às variáveis. Em programação lógica, tautologias representam verdades universais sobre o domínio do problema. Já uma _contradição_ é sempre falsa. Na Programação Lógica, contradições indicam inconsistências ou impossibilidades lógicas no domínio modelado.

Identificar tautologias permite simplificar expressões e fazer inferências válidas automaticamente. Reconhecer contradições evita tentar provar algo logicamente impossível.

Linguagens de programação que usamo a Programação Lógica usam _unificação_ e resolução para fazer deduções. Tautologias geram cláusulas vazias que simplificam esta resolução. Em problemas de _satisfatibilidade_, se obtivermos uma contradição, sabemos que as premissas são insatisfatíveis. Segure as lágrimas e o medo. Os termos _unificação_ e _satisfatibilidade_ serão explicados assim que sejam necessários. Antes disso, precisamos falar de equivalências. E para isso vamos incluir um metacaractere no nosso alfabeto. O caractere $\equiv$ que não faz parte da nossa linguagem, mas permitirá o entendimento das principais equivalências.

<table style="width: 100%; margin: auto; border-collapse: collapse;">
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%; border-top: 2px solid #666666;">$F \wedge G \equiv G \wedge F$</td>
        <td style="text-align: center; width: 30%; border-top: 2px solid #666666;">Comutatividade da Conjunção</td>
        <td style="text-align: center; width: 20%;border-top: 2px solid #666666;">(1)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \vee G \equiv G \vee F$</td>
        <td style="text-align: center; width: 30%;">Comutatividade da Disjunção</td>
        <td style="text-align: center; width: 20%;">(2)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \wedge (G \vee H) \equiv (F \wedge G) \vee (F \wedge H)$</td>
        <td style="text-align: center; width: 30%;">Distributividade da Conjunção sobre a Disjunção</td>
        <td style="text-align: center; width: 20%;">(3)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \vee (G \wedge H) \equiv (F \vee G) \wedge (F \vee H)$</td>
        <td style="text-align: center; width: 30%;">Distributividade da Disjunção sobre a Conjunção</td>
        <td style="text-align: center; width: 20%;">(4)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$\neg (F \wedge G) \equiv \neg F \vee \neg G$</td>
        <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
        <td style="text-align: center; width: 20%;">(5)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$\neg (F \vee G) \equiv \neg F \wedge \neg G$</td>
        <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
        <td style="text-align: center; width: 20%;">(6)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \rightarrow G \equiv \neg F \vee G$</td>
        <td style="text-align: center; width: 30%;">Definição de Implicação</td>
        <td style="text-align: center; width: 20%;">(7)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \leftrightarrow G \equiv (F \rightarrow G) \wedge (G \rightarrow F)$</td>
        <td style="text-align: center; width: 30%;">Definição de Equivalência</td>
        <td style="text-align: center; width: 20%;">(8)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \rightarrow G \equiv \neg G \rightarrow \neg F$</td>
        <td style="text-align: center; width: 30%;">Lei da Contra positiva</td>
        <td style="text-align: center; width: 20%;">(9)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \wedge \neg F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Contradição</td>
        <td style="text-align: center; width: 20%;">(10)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \vee \neg F \equiv T$</td>
        <td style="text-align: center; width: 30%;">Lei da Exclusão</td>
        <td style="text-align: center; width: 20%;">(11)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$\neg(\neg F) \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Dupla Negação</td>
        <td style="text-align: center; width: 20%;">(12)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade</td>
        <td style="text-align: center; width: 20%;">(13)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \wedge T \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(14)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \wedge F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei do Domínio para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(15)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \vee T \equiv T$</td>
        <td style="text-align: center; width: 30%;">Lei do Domínio para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(16)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \vee F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(17)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$F \wedge F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Idempotência para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(18)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$F \vee F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Idempotência para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(19)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$(F \wedge G) \wedge H \equiv F \wedge (G \wedge H)$</td>
        <td style="text-align: center; width: 30%;">Associatividade da Conjunção</td>
        <td style="text-align: center; width: 20%;">(20)</td>
    </tr>
    <tr style="background-color: #f2f2f2;border-bottom: 2px solid #666666;">
        <td style="text-align: center; width: 50%;">$(F \vee G) \vee H \equiv F \vee (G \vee H)$</td>
        <td style="text-align: center; width: 30%;">Associatividade da Disjunção</td>
        <td style="text-align: center; width: 20%;">(21)</td>
    </tr>
</table>
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Tabela 2 - Equivalências em Lógica Proposicional.</legend>

Estas equivalências permitem validar Fórmulas Bem Formadas sem o uso de uma Tabela Verdade. São muitas as equivalências que existem, estas são as mais comuns. Talvez, alguns exemplos de validação de Fórmulas Bem Formadas, clareiem o caminho que precisamos seguir:

**Exemplo 1**: $F \wedge (G \vee (F \wedge H))$

Simplificação:

$$
 \begin{align*}
 F \wedge (G \vee (F \wedge H)) &\equiv (F \wedge G) \vee (F \wedge (F \wedge H)) && \text{(Distributividade da Conjunção sobre a Disjunção, 3)} \\
 &\equiv (F \wedge G) \vee (F \wedge H) && \text{(Lei da Idempotência para a Conjunção, 18)}
 \end{align*}
$$

**Exemplo 2**: $F \rightarrow (G \wedge (H \vee F))$

Simplificação:

$$
 \begin{align*}
 F \rightarrow (G \wedge (H \vee F)) &\equiv \neg F \vee (G \wedge (H \vee F)) && \text{(Definição de Implicação, 7)} \\
 &\equiv (\neg F \vee G) \wedge (\neg F \vee (H \vee F)) && \text{(Distributividade da Disjunção sobre a Conjunção, 4)} \\
 &\equiv (\neg F \vee G) \wedge (H \vee \neg F \vee F) && \text{(Comutatividade da Disjunção, 2)} \\
 &\equiv (\neg F \vee G) \wedge T && \text{(Lei da Exclusão, 11)} \\
 &\equiv \neg F \vee G && \text{(Lei da Identidade para a Conjunção, 14)}
 \end{align*}
$$

**Exemplo 3**: $\neg (F \wedge (G \rightarrow H))$

Simplificação:

$$
 \begin{align*}
 \neg (F \wedge (G \rightarrow H)) &\equiv \neg (F \wedge (\neg G \vee H)) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg F \vee \neg (\neg G \vee H) && \text{(Lei de De Morgan, 5)} \\
 &\equiv \neg F \vee (G \wedge \neg H) && \text{(Lei de De Morgan, 6)}
 \end{align*}
$$

**Exemplo 4**: $\neg ((F \rightarrow G) \wedge (H \rightarrow I))$

Simplificação:

$$
 \begin{align*}
 \neg ((F \rightarrow G) \wedge (H \rightarrow I)) &\equiv \neg ((\neg F \vee G) \wedge (\neg H \vee I)) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg (\neg F \vee G) \vee \neg (\neg H \vee I) && \text{(Lei de De Morgan, 5)} \\
 &\equiv (F \wedge \neg G) \vee (H \wedge \neg I) && \text{(Lei de De Morgan, 6)}
 \end{align*}
$$

**Exemplo 5**: $(F \rightarrow G) \vee (H \rightarrow I) \vee (E \rightarrow F)$

Simplificação:

$$
 \begin{align*}
 (F \rightarrow G) \vee (H \rightarrow I) \vee (E \rightarrow F) &\equiv (\neg F \vee G) \vee (\neg H \vee I) \vee (\neg E \vee F) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg F \vee G \vee \neg H \vee I \vee \neg E \vee F && \text{(Comutatividade da Disjunção, 2)}
 \end{align*}
$$

**Exemplo 6:**
$F \wedge (G \vee (H \rightarrow I)) \vee (\neg E \leftrightarrow F)$

Simplificação:

$$
\begin{align*}
F \wedge (G \vee (H \rightarrow I)) \vee (\neg E \leftrightarrow F) &\equiv F \wedge (G \vee (\neg H \vee I)) \vee ((\neg E \rightarrow F) \wedge (F \rightarrow \neg E)) && \text{(Definição de Implicação, 7)}\\
&\equiv (F \wedge G) \vee (F \wedge (\neg H \vee I)) \vee ((\neg E \vee F) \wedge (\neg F \vee \neg E)) && \text{(Distributividade da Conjunção sobre a Disjunção, 3)}\\
&\equiv (F \wedge G) \vee (F \wedge (\neg H \vee I)) \vee (F \vee \neg E) && \text{(Lei da Contrapositiva, 9)}
\end{align*}
$$

**Exemplo 7:**
$\neg(F \vee (G \wedge \neg H)) \leftrightarrow ((I \vee E) \rightarrow (F \wedge G))$

Simplificação:

$$
\begin{align*}
\neg(F \vee (G \wedge \neg H)) \leftrightarrow ((I \vee E) \rightarrow (F \wedge G)) &\equiv (\neg F \wedge \neg(G \wedge \neg H)) \leftrightarrow ((\neg I \wedge \neg E) \vee (F \wedge G)) && \text{(Definição de Implicação, 7)} \\
&\equiv (\neg F \wedge (G \vee H)) \leftrightarrow (\neg I \vee \neg E \vee (F \wedge G)) && \text{(Lei de De Morgan, 6)}
\end{align*}
$$

**Exemplo 8:**
$\neg(F \leftrightarrow G) \vee ((H \rightarrow I) \wedge (\neg E \vee \neg F))$

Simplificação:

$$
\begin{align*}
\neg(F \leftrightarrow G) \vee ((H \rightarrow I) \wedge (\neg E \vee \neg F)) &\equiv \neg((F \rightarrow G) \wedge (G \rightarrow F)) \vee ((\neg H \vee I) \wedge (\neg E \vee \neg F)) && \text{(Definição de Equivalência, 8)}\\
&\equiv (\neg(F \rightarrow G) \vee \neg(G \rightarrow F)) \vee ((\neg H \vee I) \wedge (\neg E \vee \neg F)) && \text{(Lei de De Morgan, 5)}\\
&\equiv ((F \wedge \neg G) \vee (G \wedge \neg F)) \vee ((\neg H \vee I) \wedge (\neg E \vee \neg F)) && \text{(Lei de De Morgan, 6)}
\end{align*}
$$

**Exemplo 9:**
$(F \wedge G) \vee ((\neg H \leftrightarrow I) \rightarrow (\neg E \wedge F))$

Simplificação:

$$
\begin{align*}
(F \wedge G) \vee ((\neg H \leftrightarrow I) \rightarrow (\neg E \wedge F)) &\equiv (F \wedge G) \vee ((\neg(\neg H \leftrightarrow I)) \vee (\neg E \wedge F)) && \text{(Definição de Implicação, 7)}\\
&\equiv (F \wedge G) \vee ((H \leftrightarrow I) \vee (\neg E \wedge F)) && \text{(Lei da Dupla Negação, 12)}\\
&\equiv (F \wedge G) \vee (((H \rightarrow I) \wedge (I \rightarrow H)) \vee (\neg E \wedge F)) && \text{(Definição de Equivalência, 8)}
\end{align*}
$$

**Exemplo 10:**  
$\neg(F \wedge (G \vee H)) \leftrightarrow (\neg(I \rightarrow E) \vee \neg(F \rightarrow G))$

Simplificação:

$$
\begin{align*}
\neg(F \wedge (G \vee H)) \leftrightarrow (\neg(I \rightarrow E) \vee \neg(F \rightarrow G)) &\equiv (\neg F \vee \neg(G \vee H)) \leftrightarrow ((I \wedge \neg E) \vee (F \wedge \neg G)) && \text{(Definição de Implicação, 7)}\\
&\equiv (\neg F \vee (\neg G \wedge \neg H)) \leftrightarrow ((I \wedge \neg E) \vee (F \wedge \neg G)) && \text{(Lei de De Morgan, 6)}
\end{align*}
$$

A Lógica Proposicional é a estrutura mais simples e, ainda assim, fundamentalmente profunda que usamos para fazer sentido do universo. Imagine um universo de verdades e falsidades, onde cada proposição é um átomo indivisível que detém uma verdade única e inalterada. Neste cosmos de lógica, estas proposições são as estrelas, e as operações lógicas - conjunção, disjunção, negação, implicação, e bi-implicação - são as forças gravitacionais que as unem em constelações mais complexas de significado.

A Lógica Proposicional, enquanto subcampo da lógica matemática, é essencial para a forma como entendemos e interagimos com o mundo ao nosso redor. Ela fornece a base para a construção de argumentos sólidos e para a avaliação da validade de proposições. Originadas na necessidade humana de descobrir a verdade e diminuir os conflitos a partir da lógica. No entanto, a beleza da Lógica Proposicional se estende além do campo da filosofia e do discurso. Ela é a fundação da Álgebra de Boole, a qual, por sua vez, é a base para o design de circuitos eletrônicos e a construção de computadores modernos. Graças a uma ideia de [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon). as operações básicas da Álgebra de Boole - AND, OR, NOT - são os componentes fundamentais dos sistemas digitais que formam o núcleo dos computadores, telefones celulares, e de fato, de toda a nossa era digital. A Lógica Proposicional é a base sobre a qual construímos todo o edifício do raciocínio lógico. É como a tabela periódica para os químicos ou as leis de Newton para os físicos. É simples, elegante e, acima de tudo, poderosa. A partir dessa fundação, podemos começar a explorar os reinos mais profundos da lógica e do pensamento.

Nossa jornada pela Lógica Proposicional nos levou a uma compreensão mais profunda de como as proposições podem ser expressas e manipuladas. No entanto, a complexidade dessas proposições pode variar significativamente, e pode ser útil simplificar ou padronizar a forma como representamos essas proposições. Principalmente se estamos pensando em fazer circuitos digitais, onde a normalização de circuitos é um fator preponderante na determinação dos custos. É aqui que entram as formas normais.

## Lógica Predicativa

A Lógica Predicativa, coração e espírito da Lógica de Primeira Ordem, nos leva um passo além da Lógica Proposicional. Em vez de se concentrar apenas em proposições completas que são verdadeiras ou falsas, a lógica predicativa nos permite expressar proposições sobre "objetos" e as relações entre eles. Ela nos permite falar de maneira mais rica e sofisticada sobre o mundo.

Se você se lembra, na Lógica Proposicional, cada proposição é um átomo indivisível. Por exemplo, 'A chuva cai' ou 'O sol brilha'. Cada uma dessas proposições é verdadeira ou falsa como uma unidade. Na lógica predicativa, no entanto, podemos olhar para dentro dessas proposições. Podemos falar sobre o sujeito - a chuva, o sol - e o predicado - cai, brilha. E podemos quantificar sobre eles: para todos os dias, existe um momento em que o sol brilha.

Enquanto a Lógica Proposicional pode ser vista como a aritmética do verdadeiro e do falso, a lógica predicativa é a álgebra do raciocínio. Ela nos permite manipular proposições de maneira muito mais rica e expressiva. Com ela, podemos começar a codificar partes substanciais da matemática e da ciência, levando-nos mais perto de nossa busca para decifrar o cosmos, um símbolo de lógica de cada vez.

### Introdução aos Predicados

Um predicado é como uma luneta que nos permite observar as propriedades de uma entidade. É uma lente através da qual podemos ver se uma entidade particular possui ou não uma característica específica. Por exemplo, ao observar o universo das letras através do telescópio do predicado _ser uma vogal_, percebemos que algumas entidades, como $F$ e $I$, possuem essa propriedade, enquanto outras, como $G$ e $H$, não.

Contudo, é importante lembrar que um predicado não é uma afirmação absoluta de verdade ou falsidade. Ao contrário das proposições, os predicados não são declarações completas. Eles são mais parecidos com frases com espaços em branco, aguardando para serem preenchidos. Por exemplo:

1. O \***\*\_\*\*** está saboroso;

2. O \***\*\_\*\*** é vermelho;

3. \***\*\_\*\*** é alto.

Preencha as lacunas, como quiser e faça sentido, e perceba que, em cada caso, estamos atribuindo uma propriedade a um objeto. Esses são exemplos de predicados do nosso cotidiano, que ilustram de maneira simples e objetiva o conceito que queremos abordar.

Aqui, no universo da lógica, os predicados são ferramentas poderosas que permitem analisar o mundo à nossa volta de maneira estruturada e precisa.

Na matemática, um predicado pode ser entendido como uma função que recebe um objeto (ou um conjunto de objetos) e retorna um valor de verdade, ou seja, verdadeiro ou falso. Esta função descreve uma propriedade que o objeto pode possuir.

Um predicado $P$ é uma função que retorna um valor booleano, isto é, $P$ é uma função $P : U \rightarrow \\{\text{Verdadeiro, Falso}\\}$ para um conjunto $U$. Esse conjunto $U$ é chamado de universo ou domínio do discurso, e dizemos que $P$ é um predicado sobre $U$.

Podemos imaginar que o universo $U$ é o conjunto de todos os possíveis argumentos para o qual o predicado $P$ pode ser aplicado. Cada elemento desse universo é testado pelo predicado, que retorna Verdadeiro ou Falso dependendo se o elemento cumpre ou não a propriedade descrita pelo predicado. Dessa forma, podemos entender o predicado como uma espécie de filtro, ou critério, que é aplicado ao universo $U$, separando os elementos que cumprem uma determinada condição daqueles que não a cumprem. Esta é uma maneira de formalizar e estruturar nossas observações e declarações sobre o mundo ao nosso redor, tornando-as mais precisas e permitindo que as manipulemos de maneira lógica e consistente.

Para que este conceito fique mais claro, suponha que temos um conjunto de números $U = \\{1, 2, 3, 4, 5\\}$ e um predicado $P(u)$, que dizemos unário, que afirma _u é par_. Neste caso, a variável $u$ é o argumento do predicado $P$. Quando aplicamos este predicado a cada elemento do universo $U$, obtemos um conjunto de valores verdade:

$$
\begin{align}
&P(1) = \text{falso};\\
&P(2) = \text{verdadeiro};\\
&P(3) = \text{falso};\\
&P(4) = \text{verdadeiro};\\
&P(5) = \text{falso}.
\end{align}
$$

Assim, vemos que o predicado $P(u)$ dado por _u é par_ é uma propriedade que alguns números do conjunto $U$ possuem, e outros não. Vale notar que, na lógica de predicados, a função que define um predicado pode ter múltiplos argumentos. Por exemplo, podemos ter um predicado $Q(x, y)$ que afirma _x é maior que y_. Neste caso, o predicado $Q$ é uma função de dois argumentos que retorna um valor de verdade. Dizemos que $Q(x, y)$ é um predicado binário. Exemplos nos conduzem ao caminho do entendimento:

1. **Exemplo 1**:

   - Universo do discurso: $U = \text{conjunto de todas as pessoas}$.
   - Predicado: $P(x) = \\{ x : x \text{ é um matemático} \\}$;
   - Itens para os quais $P(x)$ é verdadeiro: "Carl Gauss", "Leonhard Euler", "John Von Neumann".

2. **Exemplo 2**:

   - Universo do discurso: $U = \{x \in \mathbb{Z} : x \text{ é par}\}$
   - Predicado: $P(x) = (x > 5)$;
   - Itens para os quais $P(x)$ é verdadeiro: $6$, $8$, $10 ...$.

3. **Exemplo 3**:

   - Universo do discurso: $U = \{x \in \mathbb{R} : x > 0 \text{ e } x < 10\}$
   - Predicado: $P(x) = (x^2 - 4 = 0)$;
   - Itens para os quais $P(x)$ é verdadeiro: $2$, $-2$.

4. **Exemplo 4**:

   - Universo do discurso: $U = \\{x \in \mathbb{N} : x \text{ é um múltiplo de } 3\\}$
   - Predicado: $P(x) = (\text{mod}(x, 2) = 0)$;
   - Itens para os quais $P(x)$ é verdadeiro: $6$, $12$, $18 ...$.

5. **Exemplo 5**:

   - Universo do discurso: $U = \{(x, y) \in \mathbb{R}^2 : x \neq y\}$
   - Predicado: $P(x, y) = (x < y)$;
   - Itens para os quais $P(x, y)$ é verdadeiro: $(1, 2)$, $(3, 4)$, $(5, 6)$.

O número de argumentos em um predicado será devido apenas ao sentido que queremos dar. A metáfora lógica que estamos construindo. Por exemplo, pense em um predicado ternário $R$ dado por _x está entre y e z_. Quando substituímos $x$, $y$ e $z$ por números específicos podemos validar a verdade, ou não do predicado $R$. Vamos considerar algumas amostras adicionais de predicados baseados na aritmética com uma forma um pouco menos formal, e muito mais prática, de defini-los:

1. $Primo(n)$: o número inteiro positivo $n$ é um número primo.
2. $PotênciaDe(n, k)$: o número inteiro $n$ é uma potência exata de $k : n = ki$ para algum $i \in \mathbb{Z} ≥ 0$.
3. $somaDeDoisPrimos(n)$: o número inteiro positivo $n$ é igual à soma de dois números primos.

Em 1, 2 e 3 os predicados estão definidos com mnemônicos. Assim, aumentamos a legibilidade e tornamos mais fácil o seu entendimento. Parece simples, apenas um mnemônico como identificador. Mas, pense cuidadosamente e será capaz de vislumbrar a flexibilidade que os predicados adicionam a abstração lógica. Ainda assim, falta alguma coisa.

## Regras de Inferência

Regras de inferência são esquemas que proporcionam a estrutura para derivações lógicas. Base da tomada de decisão computacional. Elas definem os passos legítimos que podem ser aplicados a uma ou mais proposições, sejam elas atômicas ou Fórmulas Bem Formadas, para produzir uma proposição nova. Em outras palavras, uma regra de inferência é uma transformação sintática de Formas Bem Formadas que preserva a verdade.

Aqui uma regra de inferência será representada por:

$$
\frac{P_1, P_2, ..., P_n}{C}\\
$$

Onde o conjunto formado $P_1, P_2, ..., P_n$, chamado de Proposição, ou antecedente, $\Gamma$, e $C$, chamado de conclusão, ou consequente, são Formulas Bem Formadas. A regra significa que se o Proposição é verdadeiro então a conclusão $C$ também é verdadeira.

Eu vou tentar usar Proposição / conclusão. Entretanto já vou me desculpando se escapar um antecendente / consequente. Será por mera força do hábito.

A representação que usamos é conhecida como sequência de dedução, é uma forma de indicar que se o Proposição, colocado acima da linha horizontal for verdadeiro, estamos dizendo que todas as preposições $P_1, P_2, ..., P_n$ são verdadeiras e todas as proposições colocas abaixo da linha, conclusão, também serão verdadeiras.

As regras de inferência são o alicerce da lógica dedutiva e do estudo das demonstrações matemáticas. Elas permitem que raciocínios complexos sejam quebrados em passos mais simples, cada um dos quais pode ser justificado pela aplicação de uma regra de inferência. Algumas das regras de inferência mais utilizadas estão listadas a seguir:

### Modus Ponens

A regra do Modus Ponens permite inferir uma conclusão a partir de uma implicação e de sua premissa antecedente. Se temos uma implicação $F \rightarrow G$, e sabemos que $F$ é verdadeiro, então podemos concluir que $G$ também é verdadeiro.

$$
F \rightarrow G
$$

$$
\begin{aligned}
&F\\
\hline
&G\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _se chover, $(F)$, então, $(\rightarrow)$, a rua ficará molhada, $(G)$_;
- Proposição 2: _está chovendo, $(F)$ é verdadeira_.
- Conclusão: logo, _a rua ficará molhada, $(G)$_.

Algumas aplicações do Modus Ponens:

- Derivar ações de regras e leis condicionais. Por exemplo:

  - Proposição: _se a velocidade, $V$, é maior que $80 km/h$, então é uma infração de trânsito, $IT$_.
  - Proposição: _joão está dirigindo, $D$, a $90 km/h$_.
  - Conclusão: logo, _João cometeu uma infração de trânsito_.

  $$
  V > 80 \rightarrow IT
  $$

  $$
  \begin{aligned}
  &D = 90\\
  \hline
  &IT
  \end{aligned}
  $$

- Aplicar implicações teóricas e chegar a novas conclusões. Por exemplo:

  - Proposição: _se um número é par, $P$, então é divisível por 2, $D2$_.
  - Proposição: _128 é par_.
  - Conclusão: logo, _128 é divisível por 2_.

  $$
  x \text{ é par} \rightarrow \text{divisível por dois}
  $$

  $$
  \begin{aligned}
  &128 \text{ é par}\\
  \hline
  &128 \text{ é divisível por 2}
  \end{aligned}
  $$

- Fazer deduções lógicas em matemática e ciência. Por exemplo:

  - Proposição: _se dois lados de um triângulo têm o mesmo comprimento, então o triângulo é isósceles_.
  - Proposição: _o triângulo $ABC$ tem os lados $AB$, $AC$ e $BC$ do mesmo comprimento_.
  - Conclusão: logo, _o triângulo $ABC$ é isósceles_.

  $$
  \begin{aligned}
  &AB = AC\\
  &AB = AC \\
  &AB=BC\text{ no triângulo} ABC\\
  \hline
  &\text{o triângulo } ABC \text{ é isósceles}
  \end{aligned}
  $$

- Tirar conclusões com base no raciocínio condicional na vida cotidiana. Por exemplo:

  - Proposição: _se hoje não chover, então irei à praia_.
  - Proposição: _Hoje não choveu_.
  - Conclusão: logo, _irei à praia_.

  $$
  \neg (\text{chover hoje}) \rightarrow \text{ir à praia}
  $$

  $$
  \begin{aligned}
  &\neg (\text{choveu hoje})\\
  \hline
  &(\text{ir à praia})
  \end{aligned}
  $$

### Modus Tollens

A regra do Modus Tollens permite inferir a negação da premissa antecedente a partir de uma implicação e da negação de sua premissa consequente.Se temos uma implicação $F \rightarrow G$, e sabemos que $G$ é falso (ou seja, $\neg G$), então podemos concluir que $F$ também é falso.

$$
F \rightarrow G
$$

$$
\begin{aligned}
&\neg G\\
\hline
&\neg F\\
\end{aligned}
$$

Em linguagem natural:

- Proposição 1: _se uma pessoa tem 18 anos ou mais_, $(F)$, _então_, $(\rightarrow)$ _ela pode votar_, $(G)$;
- Proposição 2: _Maria não pode votar_ $(\neg G)$
- Conclusão: logo, _Maria não tem 18 anos ou mais_, $(\neg F)$.

Algumas aplicações do Modus Tollens:

- Refutar teorias mostrando que suas previsões são falsas. Por exemplo:

  - Proposição: _se a teoria da geração espontânea, $TG$ é correta, insetos irão se formar em carne deixada exposta ao ar, $I$_.
  - Proposição: _insetos não se formam em carne deixada exposta ao ar_.
  - Conclusão: logo, _a teoria da geração espontânea_ é falsa.

  $$
  TG \rightarrow I
  $$

  $$
  \begin{aligned}
  \neg I\\
  \hline
  \neg TG
  \end{aligned}
  $$

- Identificar inconsistências ou contradições em raciocínios. Por exemplo:

  - Proposição: _se João, $J$, é mais alto, $>$, que Maria $M$, então Maria não é mais alta que João_.
  - Proposição: _Maria é mais alta que João_.
  - Conclusão: logo, _o raciocínio é inconsistente_.

  $$
  (J>M) \rightarrow \neg(M>J)
  $$

  $$
  \begin{aligned}
  (M>J)\\
  \hline
  \neg(J>M)
  \end{aligned}
  $$

- Fazer deduções lógicas baseadas na negação da conclusão. Por exemplo:

  - Proposição: _se hoje, $H$, é sexta-feira, $Se$, amanhã é sábado $Sa$_.
  - Proposição: _Amanhã não é sábado_.
  - Conclusão: logo, _hoje não é sexta-feira_.

  $$
  (H=Se) \rightarrow (A=Sa)
  $$

  $$
  \begin{aligned}
  \neg(A=(Sa)\\
  \hline
  \neg(H=Se)
  \end{aligned}
  $$

- Descobrir causas de eventos por eliminação de possibilidades. Por exemplo:

  - Proposição: _se a tomada está com defeito, $D$ a lâmpada não acende $L$_.
  - Proposição: _a lâmpada não acendeu_.
  - Conclusão:  logo, _a tomada deve estar com defeito_.

  $$
  D \rightarrow \neg L
  $$

  $$
  \begin{aligned}
  &\neg L\\
  \hline
  &D
  \end{aligned}
  $$

### Dupla Negação

A regra da Dupla Negação permite eliminar uma dupla negação, inferindo a afirmação original. A negação de uma negação é equivalente à afirmação original. Esta regra é importante para simplificar expressões lógicas.

$$
\neg \neg F
$$

$$
\begin{aligned}
&\neg \neg F\\
\hline
F\\
\end{aligned}
$$

$$
\begin{aligned}
&F\\
\hline
&\neg \neg F\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _não é verdade, $(\neg)$, que Maria não, $(\neg)$, está feliz, $(F)$_.
- Conclusão: logo, _Maria está feliz, $(F)$_.

A dupla negação pode parecer desnecessária, mas ela tem algumas aplicações na lógica:

- Simplifica expressões logicas: remover duplas negações ajuda a simplificar e a normalizar expressões complexas, tornando-as mais fáceis de analisar. Por exemplo, transformar _não é verdade que não está chovendo_ em simplesmente _está chovendo_.

$$
\neg \neg \text{Está chovendo} \Leftrightarrow \text{Está chovendo}
$$

- Preserva o valor de verdade: inserir ou remover duplas negações não altera o valor de verdade original de uma proposição. Isso permite transformar proposições em formas logicamente equivalentes.

- Auxilia provas indiretas: em provas por contradição, ou contrapositiva, introduzir uma dupla negação permite assumir o oposto do que se quer provar e derivar uma contradição. Isso, indiretamente, prova a proposição original.

- Conecta lógica proposicional e de predicados: em lógica de predicados, a negação de quantificadores universais e existenciais envolve dupla negação. Por exemplo, a negação de _todo $x$ é $P$_ é _existe algum $x$ tal que não é $P$_.

$$
\neg \forall x P(x) \Leftrightarrow \exists x \neg P(x)
$$

- Permite provar equivalências: uma identidade ou lei importante na lógica é que a dupla negação de uma proposição é logicamente equivalente à proposição original. A regra da dupla negação permite formalmente provar essa equivalência.

$$
\neg \neg P \Leftrightarrow P
$$

### Adição

A regra da Adição permite adicionar uma disjunção a uma afirmação, resultando em uma nova disjunção verdadeira. Esta regra é útil para introduzir alternativas em nosso raciocínio dedutivo.

$$
F
$$

$$
\begin{aligned}
&F\\
\hline
&F \vee G\\
\end{aligned}
$$

$$
\begin{aligned}
&G\\
\hline
&F \vee G\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _o céu está azul, $(F)$_.
- Conclusão: logo, _o céu está azul ou gatos podem voar, $(F \lor G)$_;

A regra da Adição permite introduzir uma disjunção em uma prova ou argumento lógico. Especificamente, ela nos permite inferir uma disjunção $F \vee G$ a partir de uma das afirmações disjuntivas ($F$ ou $G$) individualmente.

Alguns usos e aplicações importantes da regra da Adição:

- Introduzir alternativas ou possibilidades em um argumento: por exemplo, dado que _João está em casa_, podemos concluir que _João está em casa ou no trabalho_. E expandir este ou o quanto seja necessário para explicitar os lugares onde joão está.

- Combinar afirmações em novas disjunções: dadas duas afirmações quaisquer $F$ e $G$, podemos inferir que $F$ ou $G$ é verdadeiro.

- Criar casos ou opções exaustivas em uma prova: podemos derivar uma disjunção que cubra todas as possibilidades relevantes. Lembre-se do pobre _joão_.

- Iniciar provas por casos: ao assumir cada disjuntiva separadamente, podemos provar teoremas por casos exaustivos.

- Realizar provas indiretas: ao assumir a negação de uma disjunção, podemos chegar a uma contradição e provar a disjunção original.

A regra da Adição amplia nossas capacidades de prova e abordagem de problemas.

### Modus Tollendo Ponens

O Modus Tollendo Ponens permite inferir uma disjunção a partir da negação da outra disjunção.

Dada uma disjunção $F \vee G$:

- Se $\neg F$, então $G$
- Se $\neg G$, então $F$

Esta regra nos ajuda a chegar a conclusões a partir de disjunções, por exclusão de alternativas.

$$
F \vee G
$$

$$
\begin{aligned}
&\neg F\\
\hline
&G\\
\end{aligned}
$$

$$
\begin{aligned}
&\neg G\\
\hline
&F\\
\end{aligned}
$$

Em linguagem natural:

- Proposição 1: _ou o céu está azul ou a grama é roxa_.
- Proposição 2: _a grama não é roxa_.
- Conclusão: logo, _o céu está azul_

Algumas aplicações do Modus Tollendo Ponens:

- Derivar ações a partir de regras disjuntivas. Por exemplo:

  - Proposição: _ou João vai à praia, $P$ ou João vai ao cinema, $C$_.
  - Proposição: _João não vai ao cinema_, $\neg C$.
  - Conclusão: logo, _João vai à praia_.

$$
\begin{aligned}
&P \vee C\\
&\neg C\\
\hline
&P
\end{aligned}
$$

- Simplificar casos em provas por exaustão. Por exemplo:

  - Proposição: _o número é par, $P$, ou ímpar, $I$_.
  - Proposição: _o número não é ímpar, $\neg P$_.
  - Conclusão: logo, _o número é par_.

$$
\begin{aligned}
&P \vee I\\
&\neg I\\
\hline
&P
\end{aligned}
$$

- Eliminar opções em raciocínio dedutivo. Por exemplo:

  - Proposição: _ou João estava em casa, $C$, ou João estava no trabalho, $T$_.
  - Proposição: _João não estava em casa_.
  - Conclusão: logo, _João estava no trabalho_.

$$
\begin{aligned}
&C \vee T\\
&\neg C\\
\hline
&T
\end{aligned}
$$

- Fazer prova indireta da disjunção. Por exemplo:

  - Proposição: _1 é par, $1P$, ou 1 é ímpar, $1I$_.
  - Proposição: _1 não é par_.
  - Conclusão: logo, _1 é ímpar_.

$$
\begin{aligned}
&1P \vee 1I\\
&\neg 1P\\
\hline
&1I
\end{aligned}
$$

### Adjunção

A regra da Adjunção permite combinar duas afirmações em uma conjunção. Esta regra é útil para juntar duas premissas em uma única afirmação conjuntiva.

$$
F
$$

$$
G
$$

$$
\begin{aligned}
&F\\
&G\\
\hline
&F \land G\\
\end{aligned}
$$

Em linguagem natural:

- Contexto
- proposição 1: _o céu está azul_.
- proposição 2: _os pássaros estão cantando_.
- Conclusão: logo, _o céu está azul e os pássaros estão cantando_.

Algumas aplicações da Adjunção:

- Combinar proposições relacionadas em argumentos. Por exemplo:

  - Proposição: _o céu está nublado, $N$_.
  - Proposição: _está ventando, $V$_.
  - Conclusão: logo, _o céu está nublado e está ventando_.

$$
\begin{aligned}
&N\\
&V\\
\hline
&N \land V
\end{aligned}
$$

- Criar declarações conjuntivas complexas. Por exemplo:

  - Proposição: _1 é número natural, $N_1$_.
  - Proposição: _2 é número natural $N_2$_.
  - Conclusão: logo, _1 é número natural **e** 2 é número natural_.

$$
\begin{aligned}
&N_1\\
&N_2\\
\hline
&N_1 \land N_2
\end{aligned}
$$

- Derivar novas informações da interseção de fatos conhecidos. Por exemplo:

  - Proposição: _o gato está em cima do tapete, $GT$_.
  - Proposição: _o rato está em cima do tapete, $RT$_.
  - Conclusão: logo, _o gato **e** o rato estão em cima do tapete_.

$$
\begin{aligned}
&GT\\
&RT\\
\hline
&G_T \land R_T
\end{aligned}
$$

- Fazer deduções lógicas baseadas em múltiplas proposições. Por exemplo:

  - Proposição: _2 + 2 = 4_
  - Proposição: _4 x 4 = 16_
  - Conclusão: logo, _$(2 + 2 = 4) ∧ (4 × 4 = 16)$_

$$
\begin{aligned}
&2 + 2 = 4\\
&4 \times 4 = 16\\
\hline
&2 + 2 = 4 \\land 4 \\times 4 = 16
\\end{aligned}
$$

### Simplificação

A regra da Simplificação permite inferir uma conjunção a partir de uma conjunção composta. Esta regra nos permite derivar ambos os elementos de uma conjunção, a partir da afirmação conjuntiva.

$$
F \land G
$$

$$
\begin{aligned}
&F \land G\\
\hline
&F\\
\end{aligned}
$$

$$
\begin{aligned}
&F \land G\\
\hline
&G\\
\end{aligned}
$$

Em linguagem natural:

- Contexto
  - proposição: _o céu está azul e os pássaros estão cantando_
  - Conclusão: logo, _o céu está azul. E os pássaros estão cantando_.

Algumas aplicações da Simplificação:

- Derivar elementos de conjunções complexas. Por exemplo:

  - Proposição: _hoje está chovendo, $C$, e fazendo frio, $F$_.
  - Conclusão: logo, _está chovendo_.

$$
\begin{aligned}
&C \land F\\
\hline
&C
\end{aligned}
$$

- Simplificar provas baseadas em conjunções. Por exemplo:

  - Proposição: _2 é par, $2P$, e 3 é ímpar, $3P$_.
  - Conclusão: logo, _3 é ímpar, $3I$_.

$$
\begin{aligned}
&2P \land 3I\\
\hline
&3I
\end{aligned}
$$

- Inferir detalhes específicos de declarações complexas. Por exemplo:

  - Proposição: _o gato está dormindo, $D$, e ronronando, $R$_.
  - Conclusão: logo, _o gato está ronronando_.

$$
\begin{aligned}
&D \land R\\
\hline
&R
\end{aligned}
$$

- Derivar informações de premissas conjuntivas. Por exemplo:

  - Proposição: _está chovendo, $J$, e o jogo foi cancelado, $C$_.
  - Conclusão: logo, _o jogo foi cancelado_.

$$
\begin{aligned}
&C \land J\\
\hline
&J
\end{aligned}
$$

### Bicondicionalidade\*\*

A regra da Bicondicionalidade permite inferir uma bicondicional a partir de duas condicionais. Esta regra nos permite combinar duas implicações para obter uma afirmação de equivalência lógica.

$$
F \rightarrow G
$$

$$
G \rightarrow F
$$

$$
\begin{aligned}
&F \rightarrow G\\
&G \rightarrow F\\
\hline
&F \leftrightarrow G\\
\end{aligned}
$$

Em linguagem natural:

- Contexto

  - proposição _1: se está chovendo, então a rua está molhada_.
  - proposição _2: se a rua está molhada, então está chovendo_.
  - Conclusão: logo, _está chovendo se e somente se a rua está molhada_.

  Algumas aplicações da Bicondicionalidade:

- Inferir equivalências lógicas a partir de implicações bidirecionais. Por exemplo:

  - Proposição: _se chove, $C$ então a rua fica molhada, $M$_.
  - Proposição: _se a rua fica molhada, então chove_.
  - Conclusão: logo, _chove se e somente se a rua fica molhada_.

$$
\begin{aligned}
&C \rightarrow M\\
&M \rightarrow C\\
\hline
&C \leftrightarrow M
\end{aligned}
$$

- Simplificar relações recíprocas. Por exemplo:

  - Proposição: _se um número é múltiplo de 2, $M2$ então é par, $P$_.
  - Proposição: _se um número é par, então é múltiplo de 2_.
  - Conclusão: logo, _um número é par se e somente se é múltiplo de 2_.

$$
\begin{aligned}
&M2 \rightarrow P\\
&P \rightarrow M2\\
\hline
&P \leftrightarrow M2
\end{aligned}
$$

- Estabelecer equivalências matemáticas. Por exemplo:

  - Proposição: _se $x^2 = 25$, então $x = 5$_.
  - Proposição: _se $x = 5$, então $x^2 = 25$_.
  - Conclusão: logo, _$x^2 = 25$ se e somente se $x = 5$_.

$$
\begin{aligned}
&(x^2 = 25) \\rightarrow (x = 5)\\
&(x = 5) \rightarrow (x^2 = 25)\\
\\hline
&(x^2 = 25) \\leftrightarrow (x = 5)
\\end{aligned}
$$

- Provar relações de definição mútua. Por exemplo:

  - Proposição: _se figura é um quadrado, $Q$, então tem 4 lados iguais, $4L$_.
  - Proposição: _se figura tem 4 lados iguais, é um quadrado_.
  - Conclusão: logo, _figura é quadrado se e somente se tem 4 lados iguais_.

$$
\begin{aligned}
&Q \rightarrow 4L\\
&4L \rightarrow Q\\
\hline
&Q \leftrightarrow 4L
\end{aligned}
$$

### Equivalência

A regra da Equivalência permite inferir uma afirmação ou sua negação a partir de uma bicondicional. Esta regra nos permite aplicar bicondicionais para derivar novas afirmações baseadas nas equivalências lógicas.

$$
F \leftrightarrow G
$$

$$
\begin{aligned}
&F \leftrightarrow G\\\\
&F\\
\hline
&G\\
\end{aligned}
$$

$$
\begin{aligned}
&F \leftrightarrow G\\\\
&G\\
\hline
&F\\
\end{aligned}
$$

$$
F \leftrightarrow G
$$
$$
\begin{aligned}
&\neg F\\
\hline
&\neg G\\
\end{aligned}
$$

$$
F \leftrightarrow G
$$
$$
\begin{aligned}
&\neg G\\
\hline
&\neg F\\
\end{aligned}
$$

Em linguagem natural:

- proposição 1: _está chovendo se e somente se a rua está molhada_.
- proposição 2: _está chovendo_.
- Conclusão: logo, _a rua está molhada_.

Algumas aplicações da Equivalência:

- Inferir fatos de equivalências estabelecidas. Por exemplo:

  - Proposição: _o número é par, $P$ se e somente se for divisível por 2, $D2$_.
  - Proposição: _156 é divisível por 2_.
  - Conclusão: logo, _156 é par_.

$$
\begin{aligned}
&P \leftrightarrow D2\\\\
&D2(156)\\
\hline
&P(156)
\end{aligned}
$$

- Derivar negações de equivalências. Por exemplo:

  - Proposição: _$x$ é negativo se e somente se $x < 0$_.
  - Proposição: _$x$ não é negativo_.
  - Conclusão: logo, _$x$ não é menor que $0$_.
$$
  N \leftrightarrow (x < 0)\\\\
$$
$$
\begin{aligned}
&\neg N\\
\hline
&\neg (x < 0)
\end{aligned}
$$

- Fazer deduções baseadas em definições. Por exemplo:

  - Proposição: _número ímpar é definido como não divisível por $2$_.
  - Proposição: _$9$ não é divisível por $2$_.
  - Conclusão: logo, _$9$ é ímpar_.

$$
&I \leftrightarrow \neg D_2
$$
$$
\begin{aligned}
&\neg D_2(9)\\
\hline
&I(9)
\end{aligned}
$$

## Quantificadores

Para termos uma linguagem lógica suficientemente flexível precisaremos ser capazes de dizer quando o predicado $P$ ou $Q$ é verdadeiro para muitos valores diferentes de seus argumentos. Neste sentido, vincularemos as variáveis aos predicados usando quantificadores, que indicam que a afirmação que estamos fazendo se aplica a todos os valores da variável (quantificação universal), ou se aplica a poucos, ou um, (quantificação existencial). Na lógica de predicados, usaremos esses quantificadores para fazer declarações sobre todo um universo de discurso, ou para afirmar que existe pelo menos um membro que satisfaz uma determinada propriedade neste universo.

Vamos desmistificar trazendo estes conceitos para as nossas experiências humanas e sociais. Imagine que você está em uma festa e o anfitrião lhe pede para verificar se todos os convidados têm algo para beber. Você começa a percorrer a sala, verificando cada pessoa. Se você encontrar pelo menos uma pessoa sem bebida, você pode imediatamente dizer _nem todos têm bebidas_. Mas, se você verificar cada convidado e todos eles tiverem algo para beber, você pode dizer com confiança _todos têm bebidas_. Este é o conceito do quantificador universal, matematicamente representado por $\forall$, que lemos como _para todo_.

A festa continua e o anfitrião quer saber se alguém na festa está bebendo champanhe. Desta vez, assim que você encontrar uma pessoa com champanhe, você pode responder imediatamente _sim, alguém está bebendo champanhe_. Você não precisa verificar todo mundo para ter a resposta correta. Este é o conceito do quantificador existencial, denotado por $\exists$, que lemos _existe algum_.

Voltando a matemática, considere o universo de todos os números inteiros $\mathbb{Z}$. Podemos usar o quantificador universal, $\forall$, para fazer a declaração _para todo número inteiro $x$, $x$ é maior ou igual a zero ou $x$ é menor que zero_. Usando o quantificador existencial, $\exists$, podemos dizer _existe algum número inteiro x, tal que x é igual a zero_.

Os quantificadores nos permitem fazer declarações gerais ou específicas sobre os membros de um universo de discurso, de uma forma que seria difícil ou impossível sem eles.

### Universo do Discurso

O universo do discurso, $U$, também chamado de **universo**, é o conjunto de objetos de interesse em um determinado cenário lógico. O universo do discurso é importante porque as proposições na Lógica de Predicados são declarações sobre objetos de um universo.

O universo,$U$, é o domínio das variáveis das nossas Fórmulas Bem Formadas. O universo do discurso pode ser o conjunto dos números reais, $\mathbb{R}$ o conjunto dos inteiros,$\mathbb{z}$, o conjunto de todos os alunos em uma sala de aula que usam camisa amarela, ou qualquer outro conjunto que você defina. Na prática, o universo costuma ser deixado implícito e deveria ser óbvio a partir do contexto. Se não for o caso, precisa ser explicitado.  

Por exemplo, se estamos interessados em proposições sobre números naturais, $\mathbb{N}$, o universo do discurso é o conjunto $\mathbb{N} = \{0, 1, 2, 3,...\}$. Já se estamos interessados em proposições sobre alunos de uma sala de aula, o universo do discurso poderia ser o conjunto $U = \{\text{Paulo}, \text{Ana}, ...\}$.

### Quantificador Universal

O quantificador universal $\forall$, lê-se _para todo_, indica que uma afirmação deve ser verdadeira para todos os valores de uma variável dentro de um universo de valores permitidos. Por exemplo, a preposição clássica _todos os humanos são mortais_ poderia ser escrita, em notação matemática, $\forall x : Humano(x) \rightarrow Mortal(x)$. Ou com predicado um pouco mais matemático, teríamos se $x$ é positivo então $x + 1$ é positivo, pode ser escrito $\forall x : x > 0 \rightarrow x + 1 > 0$. E pronto! Aqui temos quantificadores, Lógica Predicativa, Lógica Proposicional e Teoria dos Conjuntos.

Recorremos a teoria dos conjuntos para tornar o universo do discurso mais explícito, a notação de pertencimento é útil nesta definição. Um exemplo bom exemplo desta prática seria:

$$\forall x \in \mathbb{Z} : x > 0 \rightarrow x + 1 > 0$$

Isso é logicamente equivalente a escrever:

$$\forall x \\{x \in \mathbb{Z} \rightarrow (x > 0 \rightarrow x + 1 > 0)\\}$$

ou a escrever:

$$\forall x (x \in \mathbb{Z} \land x > 0) \rightarrow x + 1 > 0$$

Espero que concordemos que a forma curta deixa mais claro que a intenção de $x \in \mathbb{Z}$ é restringir o intervalo de $x$.

A afirmação $\forall x P(x)$ é, de certa forma a operação $\wedge$, _and_, em todo o universo do discurso. Se pensarmos assim, o predicado:

$$\forall x \in \mathbb{N} : P(x)$$

Pode ser escrito como:

$$P(0) \land P(1) \land P(2) \land P(3) \land \ldots$$

Onde $P(0), P(1), P(2), P(3) ...$ representam a aplicação do predicado $P$ a todos os elementos $x$ do conjunto \mathbb{Z}.

Em Lógica Proposicional, não podemos escrever expressões com um número infinito de termos, como a expansão em conjunções que fizemos. No entanto, podemos usar esta interpretação informal para entender o significado por trás de $\forall x: P(x)$.

O quantificador universal $\forall x P(x)$ afirma que a proposição $P(x)$ é verdadeira para todo valor possível que $x$ pode assumir. Uma forma de interpretar isso é pensar em $x$ como uma variável que pode ter qualquer valor, fornecido por um adversário qualquer. Seu trabalho é mostrar que não importa qual valor esse adversário escolher para $x$, a proposição $P(x)$ será sempre verdadeira.

Para validar $\forall x P(x)$ escolheremos o pior caso possível para $x$ - todo valor que suspeitamos poderia fazer $P(x)$ falso. Se você consegue provar que $P(x)$ é verdadeira neste caso específico, então $\forall x P(x)$ deve ser verdadeira. Novamente, vamos recorrer a exemplos na esperança de explicitar este conhecimento.

1. **Exemplo 1**: "Todos os números reais são maiores que 0." (Universo do discurso: $x \in \mathbb{R}$)

   $$\forall x\,(Número(x) \rightarrow x > 0)$$

2. **Exemplo 2**: "Todos os triângulos em um plano euclidiano têm a soma dos ângulos internos igual a 180 graus." (Universo do discurso: $x$ é um triângulo em um plano euclidiano)

   $$\forall x\,(Triângulo(x) \rightarrow \Sigma_{i=1}^3 ÂnguloInterno_i(x) = 180^\circ)$$

3. **Exemplo 3**: "Todas as pessoas com mais de 18 anos podem tirar carteira de motorista." (Universo do discurso: $x$ é uma pessoa)

   $$\forall x\,(Pessoa(x) \land Idade(x) > 18 \rightarrow PodeTirarCarteira(x))$$

4. **Exemplo 4**: "Todo número par maior que 2 pode ser escrito como a soma de dois números primos." (Universo do discurso: $x \in \mathbb{Z}$

   $$\forall x\,(Par(x) \land x > 2 \rightarrow \exists a\exists b\, (Primo(a) \land Primo(b) \land x = a + b))$$

5. **Exemplo 5**: "Para todo número natural, se ele é múltiplo de 4 e múltiplo de 6, então ele também é múltiplo de 12." (Universo do discurso: $x \in \mathbb{N}$)

   $$\forall x\,((\exists a\in\Bbb N\,(x = 4a) \land \exists b\in\Bbb N\,(x = 6b)) \rightarrow \exists c\in\Bbb N\,(x = 12c))$$

O quantificador universal nos permite definir uma Fórmula Bem Formada representando todos os elementos de um conjunto, universo do discurso, dada uma qualidade específica, predicado. Nem sempre isso é suficiente.

### Quantificador Existencial

O quantificador existencial, $\exists$ nos permite fazer afirmações sobre a existência de objetos com certas propriedades, sem precisarmos especificar exatamente quais objetos são esses. Vamos tentar remover os véus da dúvida com um exemplo simples.

Consideremos a sentença: "existem humanos mortais". Com um pouco mais de detalhe e matemática, podemos escrever isso como: existe pelo menos um $x$ tal que $x$ é humano e mortal. Para escrever a mesma sentença com precisão matemática teremos:

$$\exists x : \text{Humano}(x) \land \text{Mortal}(x)$$

Lendo por partes: existe um $x$, tal que $x$ é humano _E_ $x$ é mortal. Em outras palavras, existe pelo menos um humano que é mortal.

Note duas coisas importantes:

1. Nós não precisamos dizer exatamente quem é esse humano mortal. Só afirmamos que existe um. O operador $\exists$ captura essa ideia.

2. Usamos _E_ ($\land$), não implicação ($\rightarrow$). Se usássemos $\rightarrow$, a afirmação ficaria muito mais fraca. Veja:

$$\exists x: \text{Humano}(x) \rightarrow \text{Mortal}(x)$$

Que pode ser lido como: "existe um $x$ tal que, _SE_ $x$ é humano, _ENTÃO_ $x$ é mortal". Essa afirmação é verdadeira em qualquer universo que contenha um unicórnio de bolinhas roxas imortal. Porque o unicórnio não é humano, então $\text{Humano}(\text{unicórnio})$ é falsa, e a implicação $\text{Humano}(x) \rightarrow \text{Mortal}(x)$ é verdadeira independente do consequente. Não entendeu? Volte dois parágrafos e leia novamente. Repita!

Portanto, é crucial usar o operador $\land$, e não $\rightarrow$ quando trabalhamos com quantificadores existenciais. O $\land$ garante que a propriedade se aplica ao objeto existente definido pelo $\exists$.

Assim como o quantificador universal, $\forall$, o quantificador existencial, $\exists$, também pode ser restrito a um universo específico, usando a notação de pertencimento:

$$\exists x \in \mathbb{Z}: x = x^2$$

Que afirma a existência de pelo menos um inteiro $x$ tal que $x$ é igual ao seu quadrado. Novamente, não precisamos dizer qual é esse inteiro, apenas que ele existe dentro do conjunto dos inteiros. Existe?

De forma geral, o quantificador existencial serve para fazer afirmações elegantes sobre a existência de objetos com certas propriedades, sem necessariamente conhecermos ou elencarmos todos esses objetos. Isso agrega mais qualidade a representação do mundo real que podemos fazer com a Lógica de Primeira Ordem.

Estudando o quantificador universal encontramos duas equivalências interessantes:

$$\lnot \forall x : P(x) \leftrightarrow \exists x : \lnot P(x)$$

$$\lnot \exists x : P(x) \leftrightarrow \forall x : \lnot P(x)$$

Essas equivalências são essencialmente as versões quantificadas das Leis de De Morgan: a primeira diz que se você quer a que nem todos os humanos são mortais, isso é equivalente a encontrar algum humano que não é mortal. A segunda diz que para mostrar que nenhum humano é mortal, você tem que mostrar que todos os humanos não são mortais.

Podemos representar uma declaração $\exists x P(x)$ como uma expressão _OU_. Por exemplo, $\exists x \in \mathbb{N} : P(x)$ poderia ser reescrito como:

$$P(0) \lor P(1) \lor P(2) \lor P(3) \lor \ldots$$

E lembramos o problema que encontramos quando fizemos isso com o quantificador $\forall$: não podemos representar fórmulas sem fim em Lógica de Primeira Ordem. Mas, aqui também esta notação nos permite entender melhor o quantificador existencial.

A expansão de $\exists$ usando $\lor$ afirma que a proposição $P(x)$ é verdadeira se pelo menos um valor de $x$ dentro do universo definido atender ao predicado $P$. O que esta expansão está dizendo é que existe pelo menos um número natural $x$ tal que $P(x)$ é verdadeiro. Não precisamos saber exatamente qual é esse $x$. Apenas que ele existe dentro de $\mathbb{N}$.

O quantificador existencial não especifica o objeto, apenas afirma que existe um objeto com aquela propriedade, dentro do universo determinado. Isso permite fazer afirmações elegantes sobre a existência de objetos com certas características, certas qualidades, ou ainda, certos predicados, sem necessariamente conhecermos exatamente quais são esses objetos.

Portanto, mesmo que não possamos de fato escrever uma disjunção infinita na Lógica de Primeira Ordem, essa expansão informal transmite de forma simples e intuitiva o significado do quantificador existencial.

# Da Matemática para a Linguagem Natural

Ao ler Fórmula Bem Formada contendo quantificadores, **lemos da esquerda para a direita**.

$\forall x$ pode ser lido como _para todo objeto $x$ no universo do discurso onde este objeto está implícito, o seguinte se mantém_ e $\exists x$ pode ser lido como _existe um objeto $x$ no universo que satisfaz o seguinte_ ou ainda _para algum objeto $x$ no universo, o seguinte se mantém_. A forma como lê-mos determina nosso entendimento da Fórmula Bem Formada.

A conversão de uma Fórmula Bem Formada em sentença, não necessariamente resulta em boas expressões em linguagem natural. Apesar disso, sempre devemos começar lendo estas Fórmulas Bem Formadas. Primeiro tente fazer a leitura correta e depois aprimores as sentenças sem alterar os valores lógicos.

Por exemplo: seja $U$ o universo o conjunto de aviões e seja $F(x,y)$ o predicado  denotando _$x$ voa mais rápido que $y$_, poderemos ter: 

- $\forall x\forall y F(x,y)$ pode ser lido como _Para todo avião $x$ se mantém: $x$ é mais rápido que todo ( no sentido de qualquer) avião $y$_.

- $\exists x\forall y F(x,y)$ pode ser lido inicialmente como _Para algum avião $x$ se mantém: para todo avião $y$, $x$ é mais rápido que $y$_.

- $\forall x\exists y F(x,y)$ representa _Existe um avião $x$ que satisfaz o seguinte: (ou tal que) para todo avião $y$, $x$ é mais rápido que $y$_. 

- $\exists x\exists y F(x,y)$ se lê _Para algum avião $x$ existe um avião $y$ tal que $x$ é mais rápido que $y$_.

### Ordem de Aplicação dos Quantificadores

Quando mais de uma variável é quantificada em uma fbf como $\forall y\forall x P(x,y)$, elas são aplicadas de dentro para fora, ou seja, a mais próxima da fórmula atômica é aplicada primeiro. Assim, $\forall y\forall x P(x,y)$ se lê "existe um $y$ tal que para todo $x$, $P(x,y)$ se mantém" ou "para algum $y$, $P(x,y)$ se mantém para todo $x$".

As posições dos mesmos tipos de quantificadores podem ser trocadas sem afetar o valor lógico, desde que não haja quantificadores do outro tipo entre os que serão trocados.

Por exemplo, $\forall x\forall y\forall z P(x,y,z)$ é equivalente a $\forall y\forall x\forall z P(x,y,z)$, $\forall z\forall y\forall x P(x,y,z)$, etc. O mesmo vale para o quantificador existencial.

No entanto, as posições de quantificadores de tipos diferentes **não** podem ser trocadas. Por exemplo, $\forall x\exists y P(x,y)$ **não** é equivalente a $\exists y\forall x P(x,y)$. Por exemplo, seja $P(x,y)$ representando $x < y$ para o conjunto dos números como universo. Então, $\forall x\exists y P(x,y)$ se lê "para todo número $x$, existe um número $y$ que é maior que $x$", o que é verdadeiro, enquanto $\exists y\forall x P(x,y)$ se lê "existe um número que é maior que todo (qualquer) número", o que não é verdadeiro.

## Regras de Inferência usando Quantificadores

### Repetição

  A regra de Repetição permite repetir uma afirmação. Esta regra é útil para propagar premissas em uma prova formal.

$$
F
$$

$$
\begin{aligned}
&F\\
\hline
&F\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _o céu está azul_.
- Conclusão: logo, _o céu está azul_.

  Algumas aplicações da Repetição:

- Reafirmar premissas em provas longas. Por exemplo:

  - Proposição: _todos os homens,$H(x)$, são mortais, M(x)$_.
  - Conclusão: logo, _todos os homens são mortais_.

$$
\begin{aligned}
&\forall x(H(x) \rightarrow M(x))\\
\hline
&\forall x(H(x) \rightarrow M(x))
\end{aligned}
$$

- Introduzir suposições em provas indiretas. Por exemplo:

  - Proposição: _suponha que $(2 + 2 = 5)$_.
  - Conclusão: logo, _(2 + 2 = 5)$_.

$$
\begin{aligned}
&2 + 2 = 5\\
\hline
&2 + 2 = 5
\end{aligned}
$$

- Derivar instâncias de generalizações. Por exemplo:

  - Proposição: _para todo $x$, $x + 0 = x$_.
  - Conclusão: logo, _$2 + 0 = 2$.

$$
\begin{aligned}
&\forall x(x + 0 = x)\\
\hline
&2 + 0 = 2
\end{aligned}
$$

### Instanciação Universal

A regra de Instanciação Universal permite substituir a variável em uma afirmação universalmente quantificada por um termo concreto. Esta regra nos permite derivar casos particulares a partir de afirmações gerais.

$$
\forall x \: P(x)
$$

$$
\begin{aligned}
&\forall x \: P(x)\\
\hline
&P(a)\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _todos os cachorros latem_.
- Conclusão: logo, _Rex late_.

Algumas aplicações da Instanciação Universal:

- Derivar casos concretos de proposições universais. Por exemplo:

  - Proposição: _todos os mamíferos respiram ar_.
  - Conclusão: logo, _a baleia respira ar_.

$$
\begin{aligned}
&\forall x(M(x) \rightarrow R(x))\\
\hline
&R(b)
\end{aligned}
$$

- Aplicar regras e princípios gerais. Por exemplo:

  - Proposição: _todos os triângulos têm 180 graus internos_.
  - Conclusão: logo, _o triângulo $ABC$ tem 180 graus_.

$$
\begin{aligned}
&\forall t(T(t) \rightarrow 180^\circ(t))\\
\hline
&180^\circ(\triangle ABC)
\end{aligned}
$$

- Testar propriedades em membros de conjuntos. Por exemplo:

  - Proposição: _todo inteiro é maior que seu antecessor_.
  - Conclusão: logo, _$5$ é maior que $4$_.

$$
\\begin{aligned}
&\forall n(\mathbb{Z}(n) \rightarrow (n > n-1))\\
\\hline
&5 > 4
\\end{aligned}
$$

### Generalização Existencial**

A regra de Generalização Existencial permite inferir que algo existe a partir de uma afirmação concreta. Esta regra nos permite generalizar de exemplos específicos para a existência geral.

$$
P(a)
$$

$$
\begin{aligned}
P(a)\\
\hline
\exists x \: P(x)\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _Rex é um cachorro_.
- Conclusão: logo, _existe pelo menos um cachorro_.

Algumas aplicações da Generalização Existencial (regra 12):

- Inferir existência a partir de exemplos concretos. Por exemplo:

  - Proposição: _o urânio-235 é radioativo_.
  - Conclusão: logo, _existe pelo menos um elemento químico radioativo_.

$$
\begin{aligned}
&R(u_{235})\\
\hline
&\exists x R(x)
\end{aligned}
$$

- Concluir que uma propriedade não é vazia. Por exemplo:

  - Proposição: _$7$ é um número primo_.
  - Conclusão: logo, _existe pelo menos um número primo_.

$$
\begin{aligned}
&P(7)\\
\hline
&\exists x P(x)
\end{aligned}
$$

- Inferir a existência de soluções para problemas. Por exemplo:

  - Proposição: _$x = 2$ satisfaz a equação $x + 3 = 5$_.
  - Conclusão: logo, _existe pelo menos uma solução para essa equação_.

$$
\begin{aligned}
&S(2)\\
\hline
&\exists x S(x)
\end{aligned}
$$

### Instanciação Existencial**

A regra de Instanciação Existencial permite introduzir um novo termo como instância de uma variável existencialmente quantificada. Esta regra nos permite derivar exemplos de afirmações existenciais.

$$
\exists x P(x)
$$

$$
\begin{aligned}
&\exists x P(x)\\
\hline
&P(b)\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _existe um cachorro com rabo curto_.
- Conclusão: logo, _Rex tem rabo curto_.

Algumas aplicações da Instanciação Existencial (regra 13):

- Derivar exemplos de existência previamente estabelecida. Por exemplo:

  - Proposição: _existem estrelas maiores que o Sol_.
  - Conclusão: logo, _Alpha Centauri é maior que o Sol_.

$$
\begin{aligned}
&\exists x (E(x) \land M(x, s))\\
\hline
&M(a, s)
\end{aligned}
$$

- Construir modelos satisfatíveis para predicados existenciais. Por exemplo:

  - Proposição: _existem pessoas mais velhas que $25$ anos_.
  - Conclusão: logo, _John tem 30 anos_.

$$
\begin{aligned}
&\exists x (P(x) \land V(x, 25))\\
\hline
&P(j) \land V(j, 30)
\end{aligned}
$$

- Provar que conjuntos não estão vazios. Por exemplo:

- Proposição: _existem números reais maiores que $2$_.
- Conclusão: logo, _$5$ é um número real maior que $2$_.

$$
\begin{aligned}
&\exists x (R(x) \land M(x, 2))\\
\hline
&R(5) \land M(5, 2)
\end{aligned}
$$

## Formas Normais

As formas normais, em sua essência, são um meio de trazer ordem e consistência à maneira como representamos proposições na Lógica Proposicional. Elas oferecem uma estrutura formalizada para expressar proposições, uma convenção que simplifica a comparação, análise e simplificação de proposições lógicas.

Consideremos, por exemplo, a tarefa de comparar duas proposições para determinar se são equivalentes. Sem uma forma padronizada de representar proposições, essa tarefa pode se tornar complexa e demorada. No entanto, ao utilizar as formas normais, cada proposição é expressa de uma maneira padrão, tornando a comparação direta e simples. Além disso, as formas normais também desempenham um papel crucial na simplificação de proposições. Ao expressar uma proposição em sua forma normal, é mais fácil identificar oportunidades de simplificação, removendo redundâncias ou simplificando a estrutura lógica. As formas normais não são apenas uma ferramenta para lidar com a complexidade da Lógica Proposicional, mas também uma metodologia que facilita a compreensão e manipulação de proposições lógicas.

Existem várias formas normais na Lógica Proposicional, cada uma com suas próprias regras e aplicações. Aqui estão algumas das principais:

1. **Forma Normal Negativa (FNN)**: Uma proposição está na Forma Normal Negativa se as operações de negação $\neg$ aparecerem apenas imediatamente antes das variáveis. Isso é conseguido aplicando as leis de De Morgan e eliminando as duplas negações.

2. **Forma Normal Conjuntiva (FNC)**: Uma proposição está na Forma Normal Conjuntiva se for uma conjunção, operação _E_, $\wedge$, de uma ou mais cláusulas, onde cada cláusula é uma disjunção, operação _OU_, $\vee$, de literais. Em outras palavras, é uma série de cláusulas conectadas por _Es_, onde cada cláusula é composta de variáveis conectadas por _OUs_.

3. **Forma Normal Disjuntiva (FND)**: Uma proposição está na Forma Normal Disjuntiva se for uma disjunção de uma ou mais cláusulas, onde cada cláusula é uma conjunção de literais. Ou seja, é uma série de cláusulas conectadas por _ORs_, onde cada cláusula é composta de variáveis conectadas por _ANDs_.

4. **Forma Normal Prenex (FNP)**: Uma proposição está na Forma Normal Prenex se todos os quantificadores, para a Lógica de Primeira Ordem, estiverem à esquerda, precedendo uma matriz quantificadora livre. Esta forma é útil na Lógica de Primeira Ordem e na teoria da prova.

5. **Forma Normal Skolem (FNS)**: Na Lógica de Primeira Ordem, uma fórmula está na Forma Normal de Skolem se estiver na Forma Normal Prenex e se todos os quantificadores existenciais forem eliminados. Isto é realizado através de um processo conhecido como Skolemização.

Nosso objetivo é rever a matemática que suporta a Programação Lógica, entre as principais formas normais, para este objetivo, precisamos destacar duas formas normais:

1. **Forma Normal Conjuntiva (FNC)**: A Forma Normal Conjuntiva é importante na Programação Lógica porque muitos sistemas de inferência, como a resolução, funcionam em fórmulas que estão na FNC. Além disso, os programas em Prolog, A linguagem de Programação Lógica que escolhemos, são essencialmente cláusulas na FNC.

2. **Forma Normal de Skolem (FNS)**: A Forma Normal de Skolem é útil na Programação Lógica porque a Skolemização, o processo de remover quantificadores existenciais transformando-os em funções de quantificadores universais, permite uma forma mais eficiente de representação e processamento de fórmulas lógicas. Essa forma normal é frequentemente usada em Lógica de Primeira Ordem e teoria da prova, ambas fundamentais para a Programação Lógica.

Embora outras formas normais possam ter aplicações em áreas específicas da Programação Lógica, a FNC e a FNS são provavelmente as mais amplamente aplicáveis e úteis nesse Proposição. Começando com a Forma Normal Conjuntiva.

Se considerarmos as propriedades associativas apresentadas nas linhas 20 e 21 da Tabela 2, podemos escrever uma sequência de conjunções, ou disjunções, sem precisarmos de parênteses. Sendo assim:

$$((F \wedge (G \wedge H)) \wedge I)$$

Pode ser escrita como:

$$F \wedge G \wedge H \wedge I$$
