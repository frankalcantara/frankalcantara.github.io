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

Na Programação Imperativa partimos de uma determinada expressão e seguimos um conjunto de instruções até encontrar o resultado desejado. Na Programação Lógica, usamos a dedução. Partimos de uma conjectura e, de acordo com um conjunto específico de regras, tentamos construir uma prova para esta conjectura. Esta prova, não é simples, o [Grande Teorema de Fermat tomou 357 anos para ser provado](https://en.wikipedia.org/wiki/Wiles%27s_proof_of_Fermat%27s_Last_Theorem).

Na **Programação Imperativa**, o programador fornece um conjunto de instruções que definem o fluxo de controle e modificam o estado da máquina. O foco está em **como** o problema deve ser resolvido passo a passo. Exemplos de linguagens imperativas incluem C++, Java e Python.

Já na **Programação Descritiva**, o programador fornece uma descrição lógica de **o que** deve ser feito, sem especificar o fluxo de controle. O foco está no problema, não na solução. Exemplos incluem SQL, Prolog e Haskell.

Enquanto linguagens imperativas são procedurais, as descritivas são declarativas. O programador imperativo tem que pensar como um computador, já o descritivo precisa descrever a lógica do problema.

Em resumo:

- **Imperativa:** focada no processo, no "como" chegar à solução.
- **Descritiva:** focada no problema em si, no "o que" precisa ser feito.

A escolha dependerá da aplicação e do estilo do programador. Mas o futuro parece cada vez mais orientado para linguagens declarativas e descritivas, que permitem ao programador concentrar-se no problema, não nos detalhes da solução. Efeito que parece ser evidente se consideramos os avanços recentes no campo da inteligência artificial.

Na nossa jornada o caminho começará na Lógica de Primeira Ordem, que aqui dividiremos em partes menores e interdependentes. Ainda que a maior parte da academia não perceba as nuances que separam a Lógica de Primeira Ordem da Lógica Predicativa. Aqui, separaremos a Lógica de Primeira Ordem em seus compoentes aninhados olhando cada parte como se individual fosse. E, na manhã da nossa jornada, usaremos a Lógica Proposicional para construir o raciocínio.

A Lógica Proposicional é um tipo de linguagem matemática suficientemente rica para expressar muitos dos problemas que precisamos resolver e suficientemente gerenciável para que os computadores possam lidar com ela. Uma ferramenta útil tanto ao homem quanto a máquina. Quando esta ferramenta estiver conhecida mergulharemos no espírito da Lógica de Primeira Ordem, a Lógica Predicativa, ou Lógica de Predicados, e então poderemos fazer sentido do mundo.

Vamos enfrentar a inferência e a dedução, duas ferramentas para extração de conhecimento de declarações lógicas. Voltando a metáfora do Detetive, podemos dizer que a inferência é quase como um detetive que tira conclusões a partir de pistas: você tem algumas verdades e precisa descobrir outras verdades que são consequências diretas das primeiras verdades.

Vamos falar da Cláusula de Horn, um conceito um pouco mais estranho. Uma regra que torna todos os problemas expressos em lógica mais fácies de resolver. É como uma receita de bolo que, se corretamente seguida, torna o processo de cozinhar muito mais simples.

No final do dia, tudo que queremos, desde os tempos de [Gödel](https://en.wikipedia.org/wiki/Kurt_Gödel), [Turing](https://en.wikipedia.org/wiki/Alan_Turing) e [Church](https://en.wikipedia.org/wiki/Alonzo_Church) é que nossas máquinas sejam capazes de resolver problemas complexos com o mínimo de interferência nossa. Queremos que eles pensem, ou pelo menos, que simulem o pensamento. E a Programação Lógica é uma maneira deveras interessante de perseguir este objetivo.

A Programação Lógica aparece em meados dos anos 1970 como uma evolução dos esforços das pesquisas sobre a prova computacional de teoremas matemáticos e inteligência artificial. Deste esforço surgiu a esperança de que poderíamos usar a lógica como um linguagem de programação, em inglês, "programming logic" ou Prolog. Este artigo faz parte de uma série sobre a Programação Lógica, partiremos da base matemática e chegaremos ao Prolog.

# 1. Lógica de Primeira Ordem

A lógica de primeira ordem é um dos fundamentos essenciais da ciência da computação e, consequentemente, da programação. Essa matemática permite quantificar sobre objetos, fazer declarações que se aplicam a todos os membros de um conjunto ou a um membro particular desse conjunto. Por outro lado, nos impede de quantificar diretamente sobre predicados ou funções.

Usar a lógica de primeira ordem é como olhar para as estrelas em uma noite clara. Nós podemos ver e quantificar as estrelas individuais (objetos), mas não podemos quantificar diretamente sobre as constelações (predicados ou funções).

Essa restrição não é um defeito, mas sim um equilíbrio cuidadoso entre poder expressivo e simplicidade computacional. Dá-nos uma maneira de formular uma grande variedade de problemas, sem tornar o processo de resolução desses problemas excessivamente complexo.

A lógica de primeira ordem é o nosso ponto de partida, nossa base, nossa pedra fundamental. Uma forma poderosa e útil de olhar para o universo, não tão complicada que seja hermética a olhos leigos, mas suficientemente complexa para permitir a descoberta de alguns dos mistérios da matemática e, no processo, resolver alguns problemas práticos.

A Lógica de primeira ordem consiste de uma linguagem, consequentemente criada sobre um alfabeto $\Sigma$, de um conjunto de axiomas e de um conjunto de regras de inferência. Esta linguagem consiste de todas as fórmulas bem formadas da teoria da Lógica Proposicional e predicativa. O conjunto de axiomas é um subconjunto do conjunto de fórmulas bem formadas acrescido e, finalmente, um conjunto de regras de inferência.

O alfabeto $\Sigma$ pode ser dividido em conjuntos de símbolos agrupados por classes:

**variáveis, constantes e símbolos de pontuação**: vamos usar os símbolos do alfabeto latino em minúsculas e alguns símbolos de pontuação. Destaque-se os símbolos $($ e $)$, parenteses, que usaremos para definir a prioridade de operações.

Vamos usar os símbolos $u$, $v$, $w$, $x$, $y$ e $z$ para indicar variáveis e $a$, $b$, $c$, $d$ e $e$ para indicar constantes.

**Funções**: usaremos os símbolos $\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$ e $\mathbf{i}$ para indicar funções.

**Predicados**: usaremos os símbolos $P$, $Q$, $\mathbf{r}$ e $S$ para indicar predicados.

**Operadores**: usaremos os símbolos tradicionais da Lógica Proposicional: $\neg$ (negação), $\wedge$ (disjunção, _and_), $\vee$ (conjunção, _or_), $\rightarrow$ (implicação) e $\leftrightarrow$ (equivalência).

**Quantificadores**: nos manteremos no limite da tradição matemática e usar $\exists$ (quantificador existencial) e $\forall$ (quantificador universal).

**Fórmulas Bem Formatadas**: usaremos letras do alfabeto latino, maiúsculas para representar as Fórmulas Bem Formatadas: $F$, $G$, $I$, $J$, $K$.

Em qualquer linguagem matemática, a precedência das operações é como uma receita. Que deve ser seguida à letra para garantir o sucesso. Vamos definir uma ordem de precedência para garantir a legibilidade das nossas Fórmulas Bem Formatadas:

$$\neg, \forall, \exists, \wedge, \vee, \rightarrow, \leftrightarrow$$

Com a maior ordem de precedência dada a $\neg$ e a menor $\leftrightarrow$.

O uso dos parenteses e da ordem de precedência com parcimônia, permite que possamos escrever $(\forall x(\exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))))$ ou $\forall x \exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))$ que são a mesma fórmula bem formatada. Escolha a opção que seja mais fácil de ler e entender.

**Fórmulas Bem Formatadas** são conjuntos de termos e operações seguindo a ordem de precedência definida anteriormente e as regras de cada operação.

As Fórmulas Bem Formatadas são como as frases de um idioma novo, com sua própria gramática e vocabulário. Assim como aprendemos nossa língua nativa desde a infância, também precisamos aprender as regras desse novo idioma para nos comunicarmos nele.

Esse idioma não é falado por pessoas, e sim por computadores. É a linguagem da lógica matemática. E como toda linguagem, tem seus próprios símbolos - as letras que formam seu alfabeto - suas regras de gramática e suas expressões bem construídas.

As Fórmulas Bem Formatadas são as frases "gramaticalmente corretas" nessa linguagem lógico-matemática. Frases que respeitam as regras sintáticas, que têm significado claro e preciso dentro desse sistema formal. Frases que podem ser "entendidas" por um computador.

Assim como aprendemos nossa língua materna reconhecendo padrões e regularidades, também reconhecemos Fórmulas Bem Formatadas por seus padrões característicos. Os símbolos estão dispostos de maneira organizada, em uma construção step-by-step que lembra a recursão das formigas ao construir seus formigueiros.

Termos são variáveis, constantes ou mesmo funções aplicadas a termos e seguem um pequeno conjunto de regras:

1. uma variável $x$ é um termo em sí;
2. uma constante $a$ é um termo em si;
3. se $\mathbf{f}$ é uma função de termos $(t_1, ... t_2)$ então $\mathbf{f}(t_1, ... t_2)$ é um termo.

Cada proposição na Lógica Proposicional é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. 'A chuva cai', 'O sol brilha' - cada uma dessas proposições é verdadeira ou falsa como uma unidade. Um átomo, elemento básico e fundamental de todas as expressões. Também, mas tarde, chamaremos de átomos a todo predicado aplicado aos termos de uma fórmula. Assim, também precisamos definir os predicados.

1. se $\mathbf{p}$ é um predicado de termos $(t_1, ... t_2)$ então $\mathbf{p}(t_1, ... t_2)$ é um fórmula bem formatada, um átomo.
2. se $F$ e $G$ são Fórmulas Bem Formatadas então: $\neg F$, $F\wedge G$, $F \vee G$, $F \rightarrow G$ e $F \leftrightarrow G$ são Fórmulas Bem Formatadas.
3. se $F$ é uma fórmula bem formatada e $x$ uma variável então $\exists x F$ e $\forall x F$ são Fórmulas Bem Formatadas.

Por fim, podemos dizer que as Fórmulas Bem Formatadas: respeitam regras de precedência entre conectivos, parênteses e quantificadores;
não apresentam problemas como variáveis livres não quantificadas e, principalmente, são unívocas, sem ambiguidade na interpretação.

Finalmente podemos dizer que a linguagem da Lógica de Primeira Ordem é o conjunto de todas as Fórmulas Bem Formatadas incluindo os campos de estudo da Lógica Proposicional e da Lógica de Predicados. Termos e átomos costurados em uma teia onde cada termo, ou átomo, é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. 'A chuva cai', 'O sol brilha' - cada uma dessas proposições é verdadeira ou falsa como uma unidade. As operações lógicas são as pontes que conectam essas ilhas, permitindo-nos construir estruturas mais complexas de razão.

## Lógica Proposicional

Esse sistema, por vezes chamado de álgebra booleana, fundamental para o desenvolvimento da computação, é uma verdadeira tapeçaria de possibilidades. Na Lógica Proposicional, declarações atômicas, que só podem ter valores os verdadeiro, $T$, ou falso $F$, são entrelaçadas em declarações compostas cuja veracidade, segundo as regras desse cálculo, depende dos valores de verdade das declarações atômicas que as compõem quando sujeitas aos operadores, ou conectivos, que definimos anteriormente.

Vamos representar essas declarações atômicas por literais $A$, $B$, $X_1$, $X_2$ etc., e suas negações por $\neg A$, $\neg B$, $\neg X_1$, $\neg X_2$ etc. Todos os símbolos individuais e suas negações são conhecidas como literais.

As declarações atômicas e compostas são costuradas por conectivos para produzir declarações compostas, cujo valor de verdade depende dos valores de verdade das declarações componentes. Os conectivos que consideramos inicialmente, cuja Tabela Verdade será dada por:

<table style="margin-left: auto;
  margin-right: auto; text-align:center;">
  <tr style="border-top: 2px solid gray; border-bottom: 1px solid gray;">
    <th style="width:8%; border-right: 1px solid gray;">$A$</th>
    <th style="width:8%; border-right: double gray;">$B$</th> 
    <th style="width:16.8%; border-right: 1px solid gray;">$A \vee B$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$A \wedge B$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$\neg A$</th>
    <th style="width:16.8%; border-right: 1px solid gray;">$A \rightarrow B$</th>
    <th style="width:16.8%;">$A \leftrightarrow B$</th>
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

Ainda observando a Tabela Verdade acima, é fácil perceber que se tivermos $4$ termos, em vez de $2$, teremos $2^4 = 16$. Se para uma determinada Fórmula Bem Formatada todas os resultados forem verdadeiros, $T$, teremos uma _tautologia_, se todos forem falsos, $F$ uma _contradição_.

Uma _tautologia_ é uma fórmula que é sempre verdadeira, não importa a interpretação ou atribuição de valores às variáveis. Em programação lógica, tautologias representam verdades universais sobre o domínio do problema. Já uma _contradição_ é sempre falsa. Na Programação Lógica, contradições indicam inconsistências ou impossibilidades lógicas no domínio modelado.

Identificar tautologias permite simplificar expressões e fazer inferências válidas automaticamente. Reconhecer contradições evita tentar provar algo logicamente impossível.

Linguagens de programação que usamo a Programação Lógica usam _unificação_ e resolução para fazer deduções. Tautologias geram cláusulas vazias que simplificam esta resolução. Em problemas de _satisfatibilidade_, se obtivermos uma contradição, sabemos que as premissas são insatisfatíveis. Segure as lágrimas e o medo. Os termos _unificação_ e _satisfatibilidade_ serão explicados assim que sejam necessários. Antes disso, precisamos falar de equivalências. E para isso vamos incluir um metacaractere no nosso alfabeto. O caractere $\equiv$ que não faz parte da nossa linguagem, mas permitirá o entendimento das principais equivalências.

<table style="width: 100%; margin: auto; border-collapse: collapse;">
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%; border-top: 2px solid #666666;">$A \wedge B \equiv B \wedge A$</td>
        <td style="text-align: center; width: 30%; border-top: 2px solid #666666;">Comutatividade da Conjunção</td>
        <td style="text-align: center; width: 20%;border-top: 2px solid #666666;">(1)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \vee B \equiv B \vee A$</td>
        <td style="text-align: center; width: 30%;">Comutatividade da Disjunção</td>
        <td style="text-align: center; width: 20%;">(2)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \wedge (B \vee C) \equiv (A \wedge B) \vee (A \wedge C)$</td>
        <td style="text-align: center; width: 30%;">Distributividade da Conjunção sobre a Disjunção</td>
        <td style="text-align: center; width: 20%;">(3)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \vee (B \wedge C) \equiv (A \vee B) \wedge (A \vee C)$</td>
        <td style="text-align: center; width: 30%;">Distributividade da Disjunção sobre a Conjunção</td>
        <td style="text-align: center; width: 20%;">(4)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$\neg (A \wedge B) \equiv \neg A \vee \neg B$</td>
        <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
        <td style="text-align: center; width: 20%;">(5)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$\neg (A \vee B) \equiv \neg A \wedge \neg B$</td>
        <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
        <td style="text-align: center; width: 20%;">(6)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \rightarrow B \equiv \neg A \vee B$</td>
        <td style="text-align: center; width: 30%;">Definição de Implicação</td>
        <td style="text-align: center; width: 20%;">(7)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \leftrightarrow B \equiv (A \rightarrow B) \wedge (B \rightarrow A)$</td>
        <td style="text-align: center; width: 30%;">Definição de Equivalência</td>
        <td style="text-align: center; width: 20%;">(8)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \rightarrow B \equiv \neg B \rightarrow \neg A$</td>
        <td style="text-align: center; width: 30%;">Lei da Contra positiva</td>
        <td style="text-align: center; width: 20%;">(9)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \wedge \neg A \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei da Contradição</td>
        <td style="text-align: center; width: 20%;">(10)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \vee \neg A \equiv T$</td>
        <td style="text-align: center; width: 30%;">Lei da Exclusão</td>
        <td style="text-align: center; width: 20%;">(11)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$\neg(\neg A) \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Dupla Negação</td>
        <td style="text-align: center; width: 20%;">(12)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade</td>
        <td style="text-align: center; width: 20%;">(13)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \wedge T \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(14)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \wedge F \equiv F$</td>
        <td style="text-align: center; width: 30%;">Lei do Domínio para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(15)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \vee T \equiv T$</td>
        <td style="text-align: center; width: 30%;">Lei do Domínio para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(16)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \vee F \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Identidade para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(17)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$A \wedge A \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Idempotência para a Conjunção</td>
        <td style="text-align: center; width: 20%;">(18)</td>
    </tr>
    <tr style="background-color: #f2f2f2;">
        <td style="text-align: center; width: 50%;">$A \vee A \equiv A$</td>
        <td style="text-align: center; width: 30%;">Lei da Idempotência para a Disjunção</td>
        <td style="text-align: center; width: 20%;">(19)</td>
    </tr>
    <tr>
        <td style="text-align: center; width: 50%;">$(A \wedge B) \wedge C \equiv A \wedge (B \wedge C)$</td>
        <td style="text-align: center; width: 30%;">Associatividade da Conjunção</td>
        <td style="text-align: center; width: 20%;">(20)</td>
    </tr>
    <tr style="background-color: #f2f2f2;border-bottom: 2px solid #666666;">
        <td style="text-align: center; width: 50%;">$(A \vee B) \vee C \equiv A \vee (B \vee C)$</td>
        <td style="text-align: center; width: 30%;">Associatividade da Disjunção</td>
        <td style="text-align: center; width: 20%;">(21)</td>
    </tr>
</table>
<legend style="font-size: 1em;
  text-align: center;
  margin-bottom: 20px;">Tabela 2 - Equivalências em Lógica Proposicional.</legend>

Estas equivalências permitem validar Fórmulas Bem Formatadas sem o uso de uma Tabela Verdade. São muitas as equivalências que existem, estas são as mais comuns. Talvez, alguns exemplos de validação de Fórmulas Bem Formatadas, clareiem o caminho que precisamos seguir:

**Exemplo 1**: $A \wedge (B \vee (A \wedge C))$

Simplificação:

$$
 \begin{align*}
 A \wedge (B \vee (A \wedge C)) &\equiv (A \wedge B) \vee (A \wedge (A \wedge C)) && \text{(Distributividade da Conjunção sobre a Disjunção, 3)} \\
 &\equiv (A \wedge B) \vee (A \wedge C) && \text{(Lei da Idempotência para a Conjunção, 18)}
 \end{align*}
$$

**Exemplo 2**: $A \rightarrow (B \wedge (C \vee A))$

Simplificação:

$$
 \begin{align*}
 A \rightarrow (B \wedge (C \vee A)) &\equiv \neg A \vee (B \wedge (C \vee A)) && \text{(Definição de Implicação, 7)} \\
 &\equiv (\neg A \vee B) \wedge (\neg A \vee (C \vee A)) && \text{(Distributividade da Disjunção sobre a Conjunção, 4)} \\
 &\equiv (\neg A \vee B) \wedge (C \vee \neg A \vee A) && \text{(Comutatividade da Disjunção, 2)} \\
 &\equiv (\neg A \vee B) \wedge T && \text{(Lei da Exclusão, 11)} \\
 &\equiv \neg A \vee B && \text{(Lei da Identidade para a Conjunção, 14)}
 \end{align*}
$$

**Exemplo 3**: $\neg (A \wedge (B \rightarrow C))$

Simplificação:

$$
 \begin{align*}
 \neg (A \wedge (B \rightarrow C)) &\equiv \neg (A \wedge (\neg B \vee C)) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg A \vee \neg (\neg B \vee C) && \text{(Lei de De Morgan, 5)} \\
 &\equiv \neg A \vee (B \wedge \neg C) && \text{(Lei de De Morgan, 6)}
 \end{align*}
$$

**Exemplo 4**: $\neg ((A \rightarrow B) \wedge (C \rightarrow D))$

Simplificação:

$$
 \begin{align*}
 \neg ((A \rightarrow B) \wedge (C \rightarrow D)) &\equiv \neg ((\neg A \vee B) \wedge (\neg C \vee D)) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg (\neg A \vee B) \vee \neg (\neg C \vee D) && \text{(Lei de De Morgan, 5)} \\
 &\equiv (A \wedge \neg B) \vee (C \wedge \neg D) && \text{(Lei de De Morgan, 6)}
 \end{align*}
$$

**Exemplo 5**: $(A \rightarrow B) \vee (C \rightarrow D) \vee (E \rightarrow F)$

Simplificação:

$$
 \begin{align*}
 (A \rightarrow B) \vee (C \rightarrow D) \vee (E \rightarrow F) &\equiv (\neg A \vee B) \vee (\neg C \vee D) \vee (\neg E \vee F) && \text{(Definição de Implicação, 7)} \\
 &\equiv \neg A \vee B \vee \neg C \vee D \vee \neg E \vee F && \text{(Comutatividade da Disjunção, 2)}
 \end{align*}
$$

**Exemplo 6:**
$A \wedge (B \vee (C \rightarrow D)) \vee (\neg E \leftrightarrow F)$

Simplificação:

$$
\begin{align*}
A \wedge (B \vee (C \rightarrow D)) \vee (\neg E \leftrightarrow F) &\equiv A \wedge (B \vee (\neg C \vee D)) \vee ((\neg E \rightarrow F) \wedge (F \rightarrow \neg E)) && \text{(Definição de Implicação, 7)}\\
&\equiv (A \wedge B) \vee (A \wedge (\neg C \vee D)) \vee ((\neg E \vee F) \wedge (\neg F \vee \neg E)) && \text{(Distributividade da Conjunção sobre a Disjunção, 3)}\\
&\equiv (A \wedge B) \vee (A \wedge (\neg C \vee D)) \vee (F \vee \neg E) && \text{(Lei da Contrapositiva, 9)}
\end{align*}
$$

**Exemplo 7:**
$\neg(A \vee (B \wedge \neg C)) \leftrightarrow ((D \vee E) \rightarrow (F \wedge G))$

Simplificação:

$$
\begin{align*}
\neg(A \vee (B \wedge \neg C)) \leftrightarrow ((D \vee E) \rightarrow (F \wedge G)) &\equiv (\neg A \wedge \neg(B \wedge \neg C)) \leftrightarrow ((\neg D \wedge \neg E) \vee (F \wedge G)) && \text{(Definição de Implicação, 7)} \\
&\equiv (\neg A \wedge (B \vee C)) \leftrightarrow (\neg D \vee \neg E \vee (F \wedge G)) && \text{(Lei de De Morgan, 6)}
\end{align*}
$$

**Exemplo 8:**
$\neg(A \leftrightarrow B) \vee ((C \rightarrow D) \wedge (\neg E \vee \neg F))$

Simplificação:

$$
\begin{align*}
\neg(A \leftrightarrow B) \vee ((C \rightarrow D) \wedge (\neg E \vee \neg F)) &\equiv \neg((A \rightarrow B) \wedge (B \rightarrow A)) \vee ((\neg C \vee D) \wedge (\neg E \vee \neg F)) && \text{(Definição de Equivalência, 8)}\\
&\equiv (\neg(A \rightarrow B) \vee \neg(B \rightarrow A)) \vee ((\neg C \vee D) \wedge (\neg E \vee \neg F)) && \text{(Lei de De Morgan, 5)}\\
&\equiv ((A \wedge \neg B) \vee (B \wedge \neg A)) \vee ((\neg C \vee D) \wedge (\neg E \vee \neg F)) && \text{(Lei de De Morgan, 6)}
\end{align*}
$$

**Exemplo 9:**
$(A \wedge B) \vee ((\neg C \leftrightarrow D) \rightarrow (\neg E \wedge F))$

Simplificação:

$$
\begin{align*}
(A \wedge B) \vee ((\neg C \leftrightarrow D) \rightarrow (\neg E \wedge F)) &\equiv (A \wedge B) \vee ((\neg(\neg C \leftrightarrow D)) \vee (\neg E \wedge F)) && \text{(Definição de Implicação, 7)}\\
&\equiv (A \wedge B) \vee ((C \leftrightarrow D) \vee (\neg E \wedge F)) && \text{(Lei da Dupla Negação, 12)}\\
&\equiv (A \wedge B) \vee (((C \rightarrow D) \wedge (D \rightarrow C)) \vee (\neg E \wedge F)) && \text{(Definição de Equivalência, 8)}
\end{align*}
$$

**Exemplo 10:**  
$\neg(A \wedge (B \vee C)) \leftrightarrow (\neg(D \rightarrow E) \vee \neg(F \rightarrow G))$

Simplificação:

$$
\begin{align*}
\neg(A \wedge (B \vee C)) \leftrightarrow (\neg(D \rightarrow E) \vee \neg(F \rightarrow G)) &\equiv (\neg A \vee \neg(B \vee C)) \leftrightarrow ((D \wedge \neg E) \vee (F \wedge \neg G)) && \text{(Definição de Implicação, 7)}\\
&\equiv (\neg A \vee (\neg B \wedge \neg C)) \leftrightarrow ((D \wedge \neg E) \vee (F \wedge \neg G)) && \text{(Lei de De Morgan, 6)}
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

Um predicado é como uma luneta que nos permite observar as propriedades de uma entidade. É uma lente através da qual podemos ver se uma entidade particular possui ou não uma característica específica. Por exemplo, ao observar o universo das letras através do telescópio do predicado _ser uma vogal_, percebemos que algumas entidades, como $A$ e $I$, possuem essa propriedade, enquanto outras, como $B$ e $C$, não.

Contudo, é importante lembrar que um predicado não é uma afirmação absoluta de verdade ou falsidade. Ao contrário das proposições, os predicados não são declarações completas. Eles são mais parecidos com frases com espaços em branco, aguardando para serem preenchidos. Por exemplo:

1. O \_\_\_\_\_\_\_\_\_ está saboroso;

2. O \_\_\_\_\_\_\_\_\_ é vermelho;

3. \_\_\_\_\_\_\_\_\_ é alto.

Preencha as lacunas, como quiser e faça sentido, e perceba que, em cada caso, estamos atribuindo uma propriedade a um objeto. Esses são exemplos de predicados do nosso cotidiano, que ilustram de maneira simples e objetiva o conceito que queremos abordar.

Aqui, no universo da lógica, os predicados são ferramentas poderosas que permitem analisar o mundo à nossa volta de maneira estruturada e precisa.

Na matemática, um predicado pode ser entendido como uma função que recebe um objeto (ou um conjunto de objetos) e retorna um valor de verdade, ou seja, verdadeiro ou falso. Esta função descreve uma propriedade que o objeto pode possuir.

Um predicado $P$ é uma função que retorna um valor booleano, isto é, $P$ é uma função $P \vert U \rightarrow \\{\text{Verdadeiro, Falso}\\}$ para um conjunto $U$. Esse conjunto $U$ é chamado de universo ou domínio do discurso, e dizemos que $P$ é um predicado sobre $U$.

Podemos imaginar que o universo $U$ é o conjunto de todos os possíveis argumentos para o qual o predicado $P$ pode ser aplicado. Cada elemento desse universo é testado pelo predicado, que retorna Verdadeiro ou Falso dependendo se o elemento cumpre ou não a propriedade descrita pelo predicado. Dessa forma, podemos entender o predicado como uma espécie de filtro, ou critério, que é aplicado ao universo $U$, separando os elementos que cumprem uma determinada condição daqueles que não a cumprem. Esta é uma maneira de formalizar e estruturar nossas observações e declarações sobre o mundo ao nosso redor, tornando-as mais precisas e permitindo que as manipulemos de maneira lógica e consistente.

Para que este conceito fique mais claro, suponha que temos um conjunto de números $U = \\{1, 2, 3, 4, 5\\}$ e um predicado $P(u)$, que dizemos unário, que afirma _u é par_. Neste caso, a variável $u$ é o argumento do predicado $P$. Quando aplicamos este predicado a cada elemento do universo $U$, obtemos um conjunto de valores verdade:

$$
\begin{align}
P(1) = \text{falso};\\
P(2) = \text{verdadeiro};\\
P(3) = \text{falso};\\
P(4) = \text{verdadeiro};\\
P(5) = \text{falso}.
\end{align}
$$

Assim, vemos que o predicado $P(u)$ dado por _u é par_ é uma propriedade que alguns números do conjunto $U$ possuem, e outros não. Vale notar que, na lógica de predicados, a função que define um predicado pode ter múltiplos argumentos. Por exemplo, podemos ter um predicado $Q(x, y)$ que afirma _x é maior que y_. Neste caso, o predicado $Q$ é uma função de dois argumentos que retorna um valor de verdade. Dizemos que $Q(x, y)$ é um predicado binário. Exemplos nos conduzem ao caminho do entendimento:

1. **Exemplo 1**:

   - Universo do discurso: $U = \text{conjunto de todas as pessoas}$.
   - Predicado: $P(x) = \\{ x \vert x \text{ é um matemático} \\}$;
   - Itens para os quais $P(x)$ é verdadeiro: "Carl Gauss", "Leonhard Euler", "John Von Neumann".

2. **Exemplo 2**:

   - Universo do discurso: $U = \{x \in \mathbb{Z} \vert x \text{ é par}\}$
   - Predicado: $P(x) = (x > 5)$;
   - Itens para os quais $P(x)$ é verdadeiro: $6$, $8$, $10 ...$.

3. **Exemplo 3**:

   - Universo do discurso: $U = \{x \in \mathbb{R} \vert x > 0 \text{ e } x < 10\}$
   - Predicado: $P(x) = (x^2 - 4 = 0)$;
   - Itens para os quais $P(x)$ é verdadeiro: $2$, $-2$.

4. **Exemplo 4**:

   - Universo do discurso: $U = \\{x \in \mathbb{N} \vert x \text{ é um múltiplo de } 3\\}$
   - Predicado: $P(x) = (\text{mod}(x, 2) = 0)$;
   - Itens para os quais $P(x)$ é verdadeiro: $6$, $12$, $18 ...$.

5. **Exemplo 5**:

   - Universo do discurso: $U = \{(x, y) \in \mathbb{R}^2 \vert x \neq y\}$
   - Predicado: $P(x, y) = (x < y)$;
   - Itens para os quais $P(x, y)$ é verdadeiro: $(1, 2)$, $(3, 4)$, $(5, 6)$.

O número de argumentos em um predicado será devido apenas ao sentido que queremos dar. A metáfora lógica que estamos construindo. Por exemplo, pense em um predicado ternário $R$ dado por _x está entre y e z_. Quando substituímos $x$, $y$ e $z$ por números específicos podemos validar a verdade, ou não do predicado $R$. Vamos considerar algumas amostras adicionais de predicados baseados na aritmética com uma forma um pouco menos formal, e muito mais prática, de defini-los:

1. $Primo(n)$: o número inteiro positivo $n$ é um número primo.
2. $PotênciaDe(n, k)$: o número inteiro $n$ é uma potência exata de $k \vert n = ki$ para algum $i \in \mathbb{Z} ≥ 0$.
3. $somaDeDoisPrimos(n)$: o número inteiro positivo $n$ é igual à soma de dois números primos.

Em 1, 2 e 3 os predicados estão definidos com mnemônicos. Assim, aumentamos a legibilidade e tornamos mais fácil o seu entendimento. Parece simples, apenas um mnemônico como identificador. Mas, pense cuidadosamente e será capaz de vislumbrar a flexibilidade que os predicados adicionam a abstração lógica. Ainda assim, falta alguma coisa.

## Quantificadores

Para termos uma linguagem lógica suficientemente flexível precisaremos ser capazes de dizer quando o predicado $P$ ou $Q$ é verdadeiro para muitos valores diferentes de seus argumentos. Neste sentido, vincularemos as variáveis aos predicados usando quantificadores, que indicam que a afirmação que estamos fazendo se aplica a todos os valores da variável (quantificação universal), ou se aplica a poucos, ou um, (quantificação existencial). Na lógica de predicados, usaremos esses quantificadores para fazer declarações sobre todo um universo de discurso, ou para afirmar que existe pelo menos um membro que satisfaz uma determinada propriedade neste universo.

Vamos desmistificar trazendo estes conceitos para as nossas experiências humanas e sociais. Imagine que você está em uma festa e o anfitrião lhe pede para verificar se todos os convidados têm algo para beber. Você começa a percorrer a sala, verificando cada pessoa. Se você encontrar pelo menos uma pessoa sem bebida, você pode imediatamente dizer _nem todos têm bebidas_. Mas, se você verificar cada convidado e todos eles tiverem algo para beber, você pode dizer com confiança _todos têm bebidas_. Este é o conceito do quantificador universal, matematicamente representado por $\forall$, que lemos como _para todo_.

A festa continua e o anfitrião quer saber se alguém na festa está bebendo champanhe. Desta vez, assim que você encontrar uma pessoa com champanhe, você pode responder imediatamente _sim, alguém está bebendo champanhe_. Você não precisa verificar todo mundo para ter a resposta correta. Este é o conceito do quantificador existencial, denotado por $\exists$, que lemos _existe algum_.

Voltando a matemática, considere o universo de todos os números inteiros $\mathbb{Z}$. Podemos usar o quantificador universal, $\forall$, para fazer a declaração _para todo número inteiro $x$, $x$ é maior ou igual a zero ou $x$ é menor que zero_. Usando o quantificador existencial, $\exists$, podemos dizer _existe algum número inteiro x, tal que x é igual a zero_.

Os quantificadores nos permitem fazer declarações gerais ou específicas sobre os membros de um universo de discurso, de uma forma que seria difícil ou impossível sem eles.

### Quantificador Universal

O quantificador universal $\forall$, lê-se _para todo_, indica que uma afirmação deve ser verdadeira para todos os valores de uma variável dentro de um universo de valores permitidos. Por exemplo, a preposição clássica _todos os humanos são mortais_ poderia ser escrita, em notação matemática, $\forall x \vert Humano(x) \rightarrow Mortal(x)$. Ou com predicado um pouco mais matemático, teríamos se $x$ é positivo então $x + 1$ é positivo, pode ser escrito $\forall x \vert x > 0 \rightarrow x + 1 > 0$. E pronto! Aqui temos quantificadores, Lógica Predicativa, Lógica Proposicional e Teoria dos Conjuntos.

Recorremos a teoria dos conjuntos para tornar o universo do discurso mais explícito, a notação de pertencimento é útil nesta definição. Um exemplo bom exemplo desta prática seria:

$$\forall x \in \mathbb{Z} \vert x > 0 \rightarrow x + 1 > 0$$

Isso é logicamente equivalente a escrever:

$$\forall x \\{x \in \mathbb{Z} \rightarrow (x > 0 \rightarrow x + 1 > 0)\\}$$

ou a escrever:

$$\forall x (x \in \mathbb{Z} \land x > 0) \rightarrow x + 1 > 0$$

Espero que concordemos que a forma curta deixa mais claro que a intenção de $x \in \mathbb{Z}$ é restringir o intervalo de $x$.

A afirmação $\forall x P(x)$ é, de certa forma a operação $\wedge$, _and_, em todo o universo do discurso. Se pensarmos assim, o predicado:

$$\forall x \in \mathbb{N} \vert P(x)$$

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

O quantificador universal nos permite definir uma Fórmula Bem Formatada representando todos os elementos de um conjunto, universo do discurso, dada uma qualidade específica, predicado. Nem sempre isso é suficiente.

### Quantificador Existencial

O quantificador existencial $\exists$ (lê-se "existe") diz que uma afirmação deve ser verdadeira para pelo menos um valor da variável. Então "algum humano é mortal" se torna $\exists x : Humano(x) \land Mortal(x)$. Observe que usamos E ao invés de implicação aqui; a afirmação $\exists x : Humano(x) \rightarrow Mortal(x)$ faz a afirmação muito mais fraca de que "existe alguma coisa $x$, tal que se $x$ é humano, então $x$ é mortal", o que é verdadeiro em qualquer universo que contém um pinguim roxo imortal - já que não é humano, $Humano(pinguim) \rightarrow Mortal(pinguim)$ é verdadeiro.

Assim como com $\forall$, $\exists$ pode ser limitado a um universo explícito com a notação de pertencimento a conjunto, por exemplo, $\exists x \in \mathbb{Z} : x = x^2$. Isso é equivalente a escrever $\exists x : x \in \mathbb{Z} \land x = x^2$.

A fórmula $\exists x : P(x)$ é equivalente a um OU muito grande, de forma que $\exists x \in \mathbb{N} : P(x)$ poderia ser reescrito como $P(0) \lor P(1) \lor P(2) \lor P(3) \lor \ldots$ Novamente, você normalmente não pode escrever uma expressão como esta se há termos infinitos, mas transmite a ideia.

2.3.2.3 Negação e quantificadores

As seguintes equivalências valem:

$\lnot \forall x : P(x) \leftrightarrow \exists x : \lnot P(x).$

$\lnot \exists x : P(x) \leftrightarrow \forall x : \lnot P(x).$

Essas são essencialmente as versões quantificadoras das leis de De Morgan: a primeira diz que se você quer mostrar que nem todos os humanos são mortais, isso é equivalente a encontrar algum humano que não é mortal. A segunda diz que para mostrar que nenhum humano é mortal, você tem que mostrar que todos os humanos não são mortais.

## Formas Normais

As formas normais, em sua essência, são um meio de trazer ordem e consistência à maneira como representamos proposições na Lógica Proposicional. Elas oferecem uma estrutura formalizada para expressar proposições, uma convenção que simplifica a comparação, análise e simplificação de proposições lógicas.

Consideremos, por exemplo, a tarefa de comparar duas proposições para determinar se são equivalentes. Sem uma forma padronizada de representar proposições, essa tarefa pode se tornar complexa e demorada. No entanto, ao utilizar as formas normais, cada proposição é expressa de uma maneira padrão, tornando a comparação direta e simples. Além disso, as formas normais também desempenham um papel crucial na simplificação de proposições. Ao expressar uma proposição em sua forma normal, é mais fácil identificar oportunidades de simplificação, removendo redundâncias ou simplificando a estrutura lógica. As formas normais não são apenas uma ferramenta para lidar com a complexidade da Lógica Proposicional, mas também uma metodologia que facilita a compreensão e manipulação de proposições lógicas.

Existem várias formas normais na Lógica Proposicional, cada uma com suas próprias regras e aplicações. Aqui estão algumas das principais:

1. **Forma Normal Negativa (FNN)**: Uma proposição está na Forma Normal Negativa se as operações de negação $\neg$ aparecerem apenas imediatamente antes das variáveis. Isso é conseguido aplicando as leis de De Morgan e eliminando as duplas negações.

2. **Forma Normal Conjuntiva (FNC)**: Uma proposição está na Forma Normal Conjuntiva se for uma conjunção, operação _AND_, $\wedge$, de uma ou mais cláusulas, onde cada cláusula é uma disjunção, operação _OR_, $\vee$, de literais. Em outras palavras, é uma série de cláusulas conectadas por _ANDs_, onde cada cláusula é composta de variáveis conectadas por _ORs_.

3. **Forma Normal Disjuntiva (FND)**: Uma proposição está na Forma Normal Disjuntiva se for uma disjunção de uma ou mais cláusulas, onde cada cláusula é uma conjunção de literais. Ou seja, é uma série de cláusulas conectadas por _ORs_, onde cada cláusula é composta de variáveis conectadas por _ANDs_.

4. **Forma Normal Prenex (FNP)**: Uma proposição está na Forma Normal Prenex se todos os quantificadores, para a lógica de primeira ordem, estiverem à esquerda, precedendo uma matriz quantificadora livre. Esta forma é útil na lógica de primeira ordem e na teoria da prova.

5. **Forma Normal Skolem (FNS)**: Na lógica de primeira ordem, uma fórmula está na Forma Normal de Skolem se estiver na Forma Normal Prenex e se todos os quantificadores existenciais forem eliminados. Isto é realizado através de um processo conhecido como Skolemização.

Nosso objetivo é rever a matemática que suporta a Programação Lógica, entre as principais formas normais, para este objetivo, precisamos destacar duas formas normais:

1. **Forma Normal Conjuntiva (FNC)**: A Forma Normal Conjuntiva é importante na Programação Lógica porque muitos sistemas de inferência, como a resolução, funcionam em fórmulas que estão na FNC. Além disso, os programas em Prolog, A linguagem de Programação Lógica que escolhemos, são essencialmente cláusulas na FNC.

2. **Forma Normal de Skolem (FNS)**: A Forma Normal de Skolem é útil na Programação Lógica porque a Skolemização, o processo de remover quantificadores existenciais transformando-os em funções de quantificadores universais, permite uma forma mais eficiente de representação e processamento de fórmulas lógicas. Essa forma normal é frequentemente usada em lógica de primeira ordem e teoria da prova, ambas fundamentais para a Programação Lógica.

Embora outras formas normais possam ter aplicações em áreas específicas da Programação Lógica, a FNC e a FNS são provavelmente as mais amplamente aplicáveis e úteis nesse contexto. Começando com a Forma Normal Conjuntiva.

Se considerarmos as propriedades associativas apresentadas nas linhas 20 e 21 da Tabela 2, podemos escrever uma sequência de conjunções, ou disjunções, sem precisarmos de parênteses. Sendo assim:

$$((A \wedge (B \wedge C)) \wedge D)$$

Pode ser escrita como:

$$A \wedge B \wedge C \wedge D$$
