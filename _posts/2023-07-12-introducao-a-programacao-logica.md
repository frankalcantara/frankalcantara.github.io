---
title: "Decifrando Mistérios: A Jornada da Programação Lógica"
layout: post
author: Frank
description: Uma aventura pelo universo matemático que fundamenta a Programação Lógica.
date: 2023-07-13T02:50:56.534Z
preview: ""
image: assets/images/prolog1.webp
tags:
   - Lógica
   - Programação Lógica
   - Prolog
categories:
   - disciplina
   - Lógica
   - material de Aula
   - matemática
rating: 5
slug: decifrando-misterios-jornada-da-programacao-logica
keywords:
   - lógica
   - Programação
   - Programação Lógica
draft: true
toc: true
lastmod: 2025-05-21T18:59:57.679Z
beforetoc: A Programação Lógica é artefato de raciocínio capaz de ensinar um detetive computadorizado a resolver os mais intricados mistérios, permitindo que se preocupe apenas com o _o que_ e deixando o _como_ a cargo da máquina. Um paradigma de programação onde não precisamos atentar para os estados da máquina e podemos nos concentrar no problema que queremos resolver. Esta é a base de alguns dos modelos computacionais que estão mudando o mundo, na revolução da Inteligência Artificial.
---

> "Logic programming is the future of artificial intelligence." - [Marvin Minsky](https://en.wikipedia.org/wiki/marvin_minsky){: class="epigraph"}

## Introdução

Imagine, por um momento, que estamos explorando o universo dos computadores, mas em vez de sermos os comandantes, capazes de ditar todos os passos do caminho, nós fornecemos as diretrizes gerais e deixamos que o computador deduza o caminho. Pode parecer estranho para quem está envolvido com as linguagens do Paradigma Imperativo. Acredite ou não, isso é exatamente o que a Programação Lógica faz.

Em vez de sermos forçados a ordenar cada detalhe do processo de solução de um problema, a Programação Lógica permite que declaremos o que queremos, e então deixemos o computador fazer o trabalho de encontrar os detalhes e processos necessários para resolver cada problema.

Na **Programação Imperativa** partimos de uma determinada expressão e seguimos um conjunto de instruções até encontrar o resultado desejado. O programador fornece um conjunto de instruções que definem o fluxo de controle e modificam o estado da máquina a cada passo. O foco está em **como** o problema deve ser resolvido. Exemplos de linguagens imperativas incluem C++, Java e Python.

Na Programação Descritiva, o programador fornece uma descrição lógica ou funcional, **do que** deve ser feito, sem especificar o fluxo de controle. O foco está no problema, não na solução. Exemplos incluem SQL, Prolog e Haskell. Na Programação Lógica, partimos de uma hipótese e, de acordo com um conjunto específico de regras, tentamos construir uma prova para esta hipótese.

Na Programação Lógica, um dos paradigmas da **Programação Descritiva** usamos a dedução para resolver problemas.

_Uma hipótese é uma suposição, expressa na forma de proposição, que é acreditada ser verdadeira, mas que ainda não foi provada_. Uma sentença declarativa que precisa ser verificada em busca da sua validação. Na linguagem natural, conjecturas são frequentemente expressas como declarações. Na Lógica de Primeira Ordem, serão proposições e as proposições serão tratadas como sentenças que foram criadas para serem verificadas na busca da verdade. Para testar a verdade expressa nestas sentenças usaremos as ferramentas da própria Lógica de Primeira Ordem.

![Diagrama de Significado de Conjecturas](/assets/images/conjecturas.webp)

Em resumo: **programação imperativa** focada no processo, no _como_ chegar à solução; **programação descritiva** focada no problema em si, no _o que_ precisa ser feito. Eu, sempre que posso escolho uma linguagem descritiva. Não há glória, nem honra nesta escolha apenas as lamúrias da opinião pessoal.

Sua escolha, pessoal e intransferível, entre estes paradigmas dependerá da aplicação que será construída, tanto quanto dependerá do estilo do programador. Contudo, o futuro parece cada vez mais orientado para linguagens descritivas, que permitam ao programador concentrar-se no problema, não nos detalhes da solução. Efeito que parece ser evidente se considerarmos os avanços da segunda década no século XXI no campo da Inteligência Artificial. Este documento contém a base matemática que suporta o entendimento da programação lógica e um pouco de Prolog, como linguagem de programação para solução de problemas. Será uma longa jornada.

Em nossa jornada, percorreremos a **Lógica de Primeira Ordem**. Esta será a nossa primeira rota, que iremos subdividir em elementos interligados e interdependentes e, sem dúvida, de mesma importância e valor: a _lógica Proposicional_ e a _lógica Predicativa_. Não deixe de notar que muitos dos nossos companheiros de viagem, aqueles restritos à academia, podem não entender as sutilezas desta divisão.

Pretensioso este timoneiro tenta não ser. Partiremos da _Lógica Proposicional_ com esperança de encontrar bons ventos que nos levem até o Prolog.

_A **Lógica Proposicional** é um tipo de linguagem matemática, suficientemente rica para expressar os problemas que precisamos resolver e suficientemente simples para que computadores possam lidar com ela. Quando esta ferramenta estiver conhecida mergulharemos na alma da **Lógica de Primeira Ordem**, a **Lógica Predicativa**, ou Lógica de Predicados, e então poderemos fazer sentido do mundo real de forma clara e bela_.

Vamos enfrentar a inferência e a dedução, duas ferramentas para extração de conhecimento de declarações lógicas. Voltando a metáfora do Detetive, podemos dizer que a inferência é quase como um detetive que tira conclusões a partir de pistas: teremos algumas verdades, nossas pistas, e precisaremos descobrir outras verdades, consequências diretas das primeiras verdades, para encontrar o que procuramos de forma incontestável. A verdade da lógica não abarca opiniões ou contestações. É linda e inquestionável.

Nossos mares não serão brandos, mas não nos furtaremos a enfrentar as especificidades da **Cláusula de Horn**, um conceito um pouco mais estranho. Uma regra que torna todos os problemas expressos em lógica mais fácies de resolver. Como um mapa que, se seguido corretamente, torna o processo de descobrir a verdade mais simples. Muito mais simples, até mesmo passível de automatização.

No final do dia, cansados, porém felizes, vamos entender que, desde os tempos de [Gödel](https://en.wikipedia.org/wiki/Kurt_Gödel), [Turing](https://en.wikipedia.org/wiki/Alan_Turing) e [Church](https://en.wikipedia.org/wiki/Alonzo_Church), tudo que queremos é que nossas máquinas sejam capazes de resolver problemas complexos com o mínimo de interferência nossa. Queremos que elas pensem, ou pelo menos, que simulem o pensamento. Aqui, neste objetivo, entre as pérolas mais reluzentes da evolução humana destaca-se a Programação Lógica.

Como diria [Newton](https://en.wikipedia.org/wiki/Isaac_Newton) chegamos até aqui porque nos apoiamos nos ombros de gigantes. O termo Programação Lógica aparece em meados dos anos 1970 como uma evolução dos esforços nas pesquisas sobre a prova computacional de teoremas matemáticos e Inteligência Artificial. O homem querendo fazer máquinas capazes de raciocinar como o homem. Deste esforço surgiu a esperança de que poderíamos usar a lógica como uma linguagem de programação, em inglês, _programming logic_, ou Prolog. Aqui está a base deste conhecimento.

## Lógica de Primeira Ordem

A Lógica de Primeira Ordem é uma estrutura básica da ciência da computação e da programação. Ela nos permite discursar e raciocinar com precisão sobre os elementos - podemos fazer afirmações sobre todo um grupo, ou sobre um único elemento em particular. No entanto, tem suas limitações. Na Lógica de Primeira Ordem clássica não podemos fazer afirmações diretas sobre predicados ou funções. Entretanto, algumas extensões, como a Lógica de Segunda Ordem, permitem fazer afirmações sobre predicados e funções.

Essa restrição não é um defeito, mas sim um equilíbrio cuidadoso entre poder expressivo e simplicidade computacional. Dá-nos uma forma de formular uma grande variedade de problemas, sem tornar o processo de resolução desses problemas excessivamente complexo.

A Lógica de Primeira Ordem é o nosso ponto de partida, nossa base, a pedra fundamental. Uma forma poderosa e útil de olhar para o universo, não tão complicada que seja hermética a olhos leigos, mas suficientemente complexa para permitir a descoberta de alguns dos mistérios da matemática e, no processo, resolver alguns problemas práticos.

A Lógica de Primeira Ordem consiste de uma linguagem, consequentemente criada a partir de um alfabeto $\Sigma$, de um conjunto de axiomas e de um conjunto de regras de inferência. Esta linguagem consiste de todas as fórmulas bem formadas da teoria da Lógica Proposicional e predicativa. O conjunto de axiomas é um subconjunto do conjunto de fórmulas bem formadas acrescido e, finalmente, um conjunto de regras de inferência.

O alfabeto $\Sigma$ que estamos definindo poderá ser dividido em classes formadas por conjuntos de símbolos agrupados por semelhança. Assim:

1. **variáveis, constantes e símbolos de pontuação**: vamos usar os símbolos do alfabeto latino em minúsculas e alguns símbolos de pontuação. Destaque-se os símbolos $($ e $)$, parênteses, que usaremos para definir a prioridade de operações. Vamos usar os símbolos $U$, $V$, $w$, $x$, $y$ e $z$ para indicar variáveis e $a$, $b$, $c$, $d$ e $e$ para indicar constantes.

2. **funções**: usaremos os símbolos $\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$ e $\mathbf{i}$ Para indicar funções.

3. **predicados**: usaremos letras do alfabeto latino, maiúsculas $P$, $Q$, $R$ e $S$, ou simplesmente _strings_ como $\text{MaiorQue}$ ou $\text{IgualA}$ para indicar predicados. Sempre começando com letras maiúsculas.

4. **operadores**: usaremos os símbolos tradicionais da Lógica Proposicional: $\neg$ (negação), $\wedge $ (conjunção, **AND**), $\vee $ (disjunção, _or_), $\rightarrow$ (implicação) e $\leftrightarrow$ (equivalência).

5. **quantificadores**: seguiremos, de perto, a tradição matemática usando $\exists $ (quantificador existencial) e $\forall $ (quantificador universal).

6. **Fórmulas Bem Formadas**: usaremos para representar as Fórmulas Bem Formadas: $P$, $Q$, $R$, $S$, $T$.

Na lógica matemática, uma Fórmula Bem Formada, ou Expressão Bem Formada, é uma sequência **finita** de símbolos formada de acordo com as regras gramaticais de uma linguagem formal especificamente desenvolvida para a redação das fórmulas da lógica.

_Em Lógica de Primeira Ordem, uma Fórmula Bem Formada é uma expressão que **só pode ser** verdadeira ou falsa_. As Fórmulas Bem Formadas são compostas de símbolos que representam quantificadores, variáveis, constantes, predicados, e conectivos lógicos. Cuja distribuição e uso seguirão as regras sintáticas, gramaticais e semânticas da linguagem da lógica. Aprender lógica é aprender esta linguagem.

Em qualquer linguagem matemática, sem dúvida, a regra sintática mais importante é a precedência das operações, uma espécie de receita indexada. Que deve ser seguida à letra. Neste texto, vamos nos restringir a seguinte ordem de precedência:

$$\neg, \forall, \exists, \wedge, \vee, \rightarrow, \leftrightarrow$$

Dando maior precedência a $\neg$ (negação) e a menor a $\leftrightarrow$ (equivalência).

O uso dos parênteses e da ordem de precedência requer cautela, muita cautela. Os parênteses permitem que possamos escrever $(\forall x(\exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))))$ ou $\forall x \exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))\,$ duas expressões diferentes que são a mesma Fórmula Bem Formada. Escolha a opção que seja mais fácil de ler,entender e explicar.

Na linguagem da lógica cada sentença, ou proposição, deve ser verdadeira ou falsa, nunca pode ser verdadeira e falsa ao mesmo tempo, e não pode ser algo diferente de verdadeiro ou falso. Para que uma sentença, ou proposição, seja verdadeira ela precisa ser logicamente verdadeira. Uma sentença contraditória é aquela que é sempre falsa, independentemente da interpretação.

Da mesma forma que aprendemos nossa língua materna reconhecendo padrões, repetições e regularidades, também reconhecemos Fórmulas Bem Formadas por seus padrões característicos. os símbolos estarão dispostos de forma organizada em termos sobre os quais se aplicam operações, funções e quantificadores.

Termos são variáveis, constantes ou mesmo funções aplicadas a termos e seguem um pequeno conjunto de regras:

1. uma variável $x$ é um termo em si;
2. uma constante $A$ é um termo em si; uma proposição que a contenha será verdadeira $(T)$ ou falsa $(F)$;
3. se $\mathbf{f}$ é uma função de termos $(t_1, ... t_n)$ então $\mathbf{f}(t_1, ... t_n)$ é um termo.

**Cada proposição, ou sentença, na Lógica Proposicional é um fato fundamental e indivisível**. _A chuva cai_, _O sol brilha_ - cada uma dessas proposições é verdadeira ou falsa por si só, como uma unidade, um átomo, elemento básico e fundamental de todas as expressões. Mais tarde, chamaremos de átomos a todo predicado aplicado aos termos de uma fórmula. Assim, precisamos definir os predicados.

1. se $P$ é um predicado de termos $(t_1, ... t_n)$ então $P(t_1, ... t_n)$ é uma Fórmula Bem Formada, um átomo.
2. se $P$ e $Q$ são Fórmulas Bem Formadas então: $\neg P$, $P\wedge Q$, $P \vee Q$, $P \rightarrow Q$ e $P \leftrightarrow Q$ são Fórmulas Bem Formadas.
3. se $P$ é uma Fórmula Bem Formada e $x$ uma variável então $\exists x P(x)$ e $\forall x P(x)$ são Fórmulas Bem Formadas.

Podemos dizer que as Fórmulas Bem Formadas respeitam as regras de precedência entre conectivos, parênteses e quantificadores; não apresentam problemas como variáveis livres não quantificadas e, principalmente, são unívocas, sem ambiguidade na interpretação.

Finalmente podemos definir a linguagem da Lógica de Primeira Ordem como o conjunto de todas as Fórmulas Bem Formadas criadas a partir dos campos de estudo da Lógica Proposicional e da Lógica de Predicados. Termos e átomos interligados em uma teia, onde cada termo ou átomo é como uma ilha de verdade. _A chuva cai_, _O sol brilha_. Cada uma dessas proposições é verdadeira ou falsa, em si, uma unidade, como uma ilha. As operações lógicas são as pontes que conectam essas ilhas, permitindo-nos construir as estruturas mais complexas da razão.

## Lógica Proposicional

Esse sistema, também chamado de álgebra booleana, fundamental para o desenvolvimento da computação, é uma verdadeira tapeçaria de possibilidades. **Na Lógica Proposicional, declarações atômicas, que só podem ter valores verdadeiro, $T$, ou falso $F$, são entrelaçadas em declarações compostas cuja veracidade, segundo as regras desse cálculo, depende dos valores de verdade das declarações atômicas que as compõem quando sujeitas aos operadores, ou aos conectivos, que definimos anteriormente**.

Vamos representar essas declarações atômicas por literais $A$, $B$, $X_1$, $X_2$ etc., e suas negações por $\neg A$, $\neg B$, $\neg X_1$, $\neg X_2$ etc. Todos os símbolos individuais e suas negações são conhecidos como literais.

Na Lógica Proposicional, as fórmulas são conhecidas como Fórmulas Bem Formadas. Elas podem ser atômicas ou compostas. Nas fórmulas compostas, um operador principal liga duas fórmulas atômicas ou duas compostas.

As declarações atômicas e compostas são costuradas por conectivos para produzir declarações compostas, cujo valor de verdade depende dos valores de verdade das declarações componentes. Os conectivos que consideramos inicialmente, e suas tabelas verdade serão:

| $P$ | $Q$ | $P\vee Q$ | $P\wedge Q$ | $\neg P$ | $P\rightarrow Q$ | $P\leftrightarrow Q$ | $P\oplus Q$ |
|---|---|---|---|---|---|---|---|
| T | T | $T$ | $T$ | $F$ | $T$ | $T$ | $F$ |
| T | F | $T$ | $F$ | $F$ | $F$ | $F$ | $T$ |
| F | T | $T$ | $F$ | $T$ | $T$ | $F$ | $T$ |
| $F$ | $F$ | $F$ | $F$ | $T$ | $T$ | $T$ | $F$ |

_Tabela 1 - Tabela Verdade, operadores básicos._{: class="legend"}

Quando usamos a Tabela Verdade em uma declaração composta, podemos ver se ela é verdadeira ou falsa. Basta seguir as regras de precedência e aplicar a Tabela Verdade, simplificando a expressão. É uma alternativa mais direta do que o uso dos axiomas da Lógica Proposicional.

O operador $\vee$, também chamado de ou inclusivo, é verdade quando pelo menos um dos termos é verdadeiro. Diferindo de um operador, que por não ser básico e fundamental, não consta da nossa lista, chamado de ou exclusivo, $\oplus$, falso se ambos os termos forem iguais, ou verdadeiros ou falsos.

O condicional $\rightarrow$ não implica em causalidade. O condicional $\rightarrow$ é falso apenas quando o antecedente é verdadeiro e o consequente é falso.

O bicondicional $\leftrightarrow$ equivale a ambos os componentes terem o mesmo valor-verdade. Todos os operadores, ou conectivos, conectam duas declarações, exceto $\neg$ que se aplica a apenas um termo.

Cada operador com sua própria aridade:

| No Argumentos | Aridade | Exemplos |
|---|---|---|
| 0 | Nulo | $5$, $False$, Constantes |
| 1 | Unário | $P(x)$, $7x$ |
| 2 | Binário | $x \vee y$, $c \wedge y$ |
| 3 | Ternário | if $P$ then $Q$ else $R$, $(P \rightarrow Q) \wedge (\neg P \rightarrow R)$ |

_Tabela 2 - Aridade dos Operadores da Lógica Proposicional._{: class="legend"}

Ainda observando a Tabela 1, que contem a Tabela Verdade dos operadores da Lógica Proposicional, é fácil perceber que se tivermos quatro termos diferentes, em vez de dois, teremos $2^4 = 16$ linhas. Independente do número de termos, se para uma determinada Fórmula Bem Formada todos os resultados forem verdadeiros, $T$, teremos uma _tautologia_, se todos forem falsos, $F$ uma _contradição_.

**Uma tautologia é uma fórmula que é sempre verdadeira, não importa os valores dados às variáveis**. Na Programação Lógica, tautologias são verdades universais no domínio do problema. Uma contradição é uma fórmula que é sempre falsa, independente dos valores das variáveis. Em Programação Lógica, contradições mostram inconsistências ou impossibilidades lógicas no domínio.

Identificar tautologias permite simplificar expressões e fazer inferências válidas automaticamente. Reconhecer contradições evita o custo de tentar provar algo logicamente impossível.

Linguagens de programação que usam a Programação Lógica usam **unificação** e resolução para fazer deduções. Tautologias geram cláusulas vazias que simplificam esta resolução. Em problemas de **satisfatibilidade**, se obtivermos uma contradição, sabemos que as premissas são insatisfatíveis. Segure as lágrimas e o medo. Os termos **unificação** e **satisfatibilidade** serão explicados assim que sejam necessários. Antes disso, precisamos falar de _equivalências_. Para isso vamos incluir um metacaractere no alfabeto da nossa linguagem: o caractere $\equiv$ que permitirá o entendimento das principais equivalências da Lógica Proposicional explicitadas a seguir:

| Expressão Lógica Equivalente | Nome da Lei/Propriedade | Ref. |
|---|---|---|
| $P \land Q \equiv Q \land P$ | Comutatividade da Conjunção | (1) |
| $P \lor Q \equiv Q \lor P$ | Comutatividade da Disjunção | (2) |
| $P \land (Q \lor R) \equiv (P \land Q) \lor (P \land R)$ | Distributividade da Conjunção sobre a Disjunção | (3) |
| $P \lor (Q \land R) \equiv (P \lor Q) \land (P \lor R)$ | Distributividade da Disjunção sobre a Conjunção | (4) |
| $\neg (P \land Q) \equiv \neg P \lor \neg Q$ | Lei de De Morgan | (5) |
| $\neg (P \lor Q) \equiv \neg P \land \neg Q$ | Lei de De Morgan | (6) |
| $P \rightarrow Q \equiv \neg P \lor Q$ | Definição de Implicação | (7) |
| $P \leftrightarrow Q \equiv (P \rightarrow Q) \land (Q \rightarrow P)$ | Definição de Equivalência | (8) |
| $P \rightarrow Q \equiv \neg Q \rightarrow \neg P$ | Lei da Contrapositiva | (9) |
| $P \land \neg P \equiv F$ | Lei da Contradição | (10) |
| $P \lor \neg P \equiv T$ | Lei do Terceiro Excluído | (11) |
| $\neg(\neg P) \equiv P$ | Lei da Dupla Negação | (12) |
| $P \equiv P$ | Lei da Identidade | (13) |
| $P \land T \equiv P$ | Lei da Identidade para a Conjunção | (14) |
| $P \land F \equiv F$ | Lei do Domínio para a Conjunção | (15) |
| $P \lor T \equiv T$ | Lei do Domínio para a Disjunção | (16) |
| $P \lor F \equiv P$ | Lei da Identidade para a Disjunção | (17) |
| $(P \land Q) \land R \equiv P \land (Q \land R)$ | Associatividade da Conjunção | (18) |
| $(P \lor Q) \lor R \equiv P \lor (Q \lor R)$ | Associatividade da Disjunção | (19) |
| $P \land P \equiv P$ | Idempotência da Conjunção | (20) |
| $P \lor P \equiv P$ | Idempotência da Disjunção | (21) |

_Tabela 3 - Equivalências em Lógica Proposicional._{: class="legend"}

Como essas equivalências permitem validar Fórmulas Bem Formadas sem o uso de uma tabela verdade. Uma coisa interessante seria tentar provar cada uma delas. Mas, isso fica, por enquanto, a cargo da amável leitora.

AAs equivalências que mencionei surgiram quase naturalmente enquanto escrevia, mais por hábito e necessidade do que por um raciocínio organizado. Existem muitas equivalências, mas essas são as que uso com mais frequência. Talvez, alguns exemplos de validação de Fórmulas Bem Formadas, usando apenas as equivalências da Tabela 3, possam inflar as velas do conhecimento e nos guiar pelo caminho que devemos seguir:

**Exemplo 1**: $P \wedge (Q \vee (P \wedge R))$

$$
 \begin{align*}
 P \wedge (Q \vee (P \wedge R)) &\equiv (P \wedge Q) \vee (P \wedge (P \wedge R)) && \text{Distributividade da Conjunção sobre a Disjunção (3)} \\
 &\equiv (P \wedge Q) \vee ((P \wedge P) \wedge R) && \text{Associatividade da Conjunção (20)} \\
 &\equiv (P \wedge Q) \vee (P \wedge R) && \text{Idempotência da Conjunção (P} \wedge \text{P} \equiv \text{P)}
 \end{align*}
$$

**Nota**: A lei da Idempotência ($P \wedge P \equiv P$) não está na Tabela 3.

**Exemplo 2**:$P\rightarrow (Q \wedge (R \vee P))$

$$
 \begin{align*}
 P \rightarrow (Q \wedge (R \vee P)) &\equiv \neg P \vee (Q \wedge (R \vee P)) && \text{(7)} \\
 &\equiv (\neg P \vee Q) \wedge (\neg P \vee (R \vee P)) && \text{(4)} \\
 &\equiv (\neg P \vee Q) \wedge (R \vee \neg P \vee P) && \text{(2)} \\
 &\equiv (\neg P \vee Q) \wedge T && \text{(11)} \\
 &\equiv \neg P \vee Q && \text{(14)}
 \end{align*}
$$

**Exemplo 3**: $\neg (P \wedge (Q \rightarrow R))$

$$
 \begin{align*}
 \neg (P \wedge (Q \rightarrow R)) &\equiv \neg (P \wedge (\neg Q \vee R)) && \text{(7)} \\
 &\equiv \neg P \vee \neg (\neg Q \vee R) && \text{(5)} \\
 &\equiv \neg P \vee (Q \wedge \neg R) && \text{(6)}
 \end{align*}
$$

**Exemplo 4**: $\neg ((P \rightarrow Q) \wedge (R \rightarrow S))$

$$
 \begin{align*}
 \neg ((P \rightarrow Q) \wedge (R \rightarrow S)) &\equiv \neg ((\neg P \vee Q) \wedge (\neg R \vee S)) && \text{(7)} \\
 &\equiv \neg (\neg P \vee Q) \vee \neg (\neg R \vee S) && \text{(5)} \\
 &\equiv (P \wedge \neg Q) \vee (R \wedge \neg S) && \text{(6)}
 \end{align*}
$$

**Exemplo 5**: $(P \rightarrow Q) \vee (R \rightarrow S) \vee (E \rightarrow P)$

$$
 \begin{align*}
 (P \rightarrow Q) \vee (R \rightarrow S) \vee (E \rightarrow P) &\equiv (\neg P \vee Q) \vee (\neg R \vee S) \vee (\neg E \vee P) && \text{(7)} \\
 &\equiv \neg P \vee Q \vee \neg R \vee S \vee \neg E \vee P && \text{(2)}\\
 &\equiv TRUE \vee Q \vee \neg R \vee S \vee \neg E && \text{(11)}\\
 &\equiv TRUE \vee Q \vee \neg R \vee S \vee \neg E && \text{(11)}\\
 &\equiv TRUE
 \end{align*}
$$

**Exemplo 6**: $P\wedge (Q \vee (R \rightarrow S)) \vee (\neg E \leftrightarrow P)$

Vamos simplificar a expressão passo a passo, indicando as leis da Tabela 3 utilizadas:

$$
\begin{align*}
&P\wedge (Q \vee (R \rightarrow S)) \vee (\neg E \leftrightarrow P) \\
&\equiv P\wedge (Q \vee (\neg R \vee S)) \vee ((\neg E \rightarrow P) \wedge (P \rightarrow \neg E)) && \text{Definição de Implicação (7), Definição de Equivalência (8)} \\
&\equiv P\wedge (Q \vee \neg R \vee S) \vee ((\neg (\neg E) \vee P) \wedge (\neg P \vee \neg E)) && \text{Associatividade da Disjunção (21), Definição de Implicação (7)} \\
&\equiv P\wedge (Q \vee \neg R \vee S) \vee ((E \vee P) \wedge (\neg P \vee \neg E)) && \text{Lei da Dupla Negação (12)} \\
&\equiv (P\wedge Q) \vee (P\wedge \neg R) \vee (P\wedge S) \vee ((E \vee P) \wedge (\neg P \vee \neg E)) && \text{Distributividade da Conjunção sobre a Disjunção (3), aplicada repetidamente} \\
&\equiv (P\wedge Q) \vee (P\wedge \neg R) \vee (P\wedge S) \vee (E \wedge (\neg P \vee \neg E)) \vee (P \wedge (\neg P \vee \neg E)) && \text{Distributividade da Disjunção sobre a Conjunção (4)} \\
&\equiv (P\wedge Q) \vee (P\wedge \neg R) \vee (P\wedge S) \vee (E \wedge \neg P) \vee (E \wedge \neg E) \vee (P \wedge \neg P) \vee (P \wedge \neg E) && \text{Distributividade da Conjunção sobre a Disjunção (3)} \\
&\equiv (P\wedge Q) \vee (P\wedge \neg R) \vee (P\wedge S) \vee (E \wedge \neg P) \vee F \vee F \vee (P \wedge \neg E) && \text{Lei da Contradição (e.g., } E \wedge \neg E \equiv F \text{, similar à (10))} \\
&\equiv (P\wedge Q) \vee (P\wedge \neg R) \vee (P\wedge S) \vee (E \wedge \neg P) \vee (P \wedge \neg E) && \text{Lei da Identidade para a Disjunção (17)}
\end{align*}
$$

Este exemplo ilustra como múltiplas leis podem ser aplicadas. A cuidadosa leitora pode verificar cada passo com atenção. A simplificação completa pode ser extensa.

**Exemplo 7**: determinar se a fórmula $\neg(P \lor (Q \land \neg R)) \leftrightarrow ((S \lor E) \rightarrow (P \land Q))$ é uma equivalência lógica.

Para analisar esta suposta equivalência, vamos simplificar ambos os lados separadamente:

**Lado Esquerdo**:
$$\begin{align*}
\neg(P \lor (Q \land \neg R)) &\equiv \neg P \land \neg(Q \land \neg R) & \text{(Lei de De Morgan)} \\
&\equiv \neg P \land (\neg Q \lor \neg\neg R) & \text{(Lei de De Morgan)} \\
&\equiv \neg P \land (\neg Q \lor R) & \text{(Dupla Negação)}
\end{align*}$$

**Lado Direito**:
$$\begin{align*}
((S \lor E) \rightarrow (P \land Q)) &\equiv \neg(S \lor E) \lor (P \land Q) & \text{(Eliminação da Implicação)} \\
&\equiv (\neg S \land \neg E) \lor (P \land Q) & \text{(Lei de De Morgan)}
\end{align*}$$

Como podemos ver, os resultados finais $\neg P \land (\neg Q \lor R)$ e $(\neg S \land \neg E) \lor (P \land Q)$ têm formas diferentes e envolvem variáveis diferentes. Claramente, estas expressões não são logicamente equivalentes, a menos que existam restrições adicionais entre as variáveis $P$, $Q$, $R$, $S$ e $E$, que não foram especificadas.

**Conclusão**: As expressões não são logicamente equivalentes.

**Exemplo 8**:
$\neg(P \leftrightarrow Q) \vee ((R \rightarrow S) \wedge (\neg E \vee \neg P))$

$$
\begin{align*}
\neg(P \leftrightarrow Q) \vee ((R \rightarrow S) \wedge (\neg E \vee \neg P)) &\equiv \neg((P \rightarrow Q) \wedge (Q \rightarrow P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(8)}\\
&\equiv (\neg(P \rightarrow Q) \vee \neg(Q \rightarrow P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(5)}\\
&\equiv ((P \wedge \neg Q) \vee (Q \wedge \neg P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(6)}
\end{align*}
$$

**Exemplo 9**: $(P \wedge Q) \vee ((\neg R \leftrightarrow S) \rightarrow (\neg E \wedge P))$

Vamos simplificar a expressão. Para clareza, podemos denotar $A \equiv (\neg R \leftrightarrow S)$.
$$
\begin{align*}
&(P \wedge Q) \vee ((\neg R \leftrightarrow S) \rightarrow (\neg E \wedge P)) \\
&\equiv (P \wedge Q) \vee (\neg (\neg R \leftrightarrow S) \vee (\neg E \wedge P)) && \text{Definição de Implicação (7)} \\
&\equiv (P \wedge Q) \vee (\neg ((\neg R \rightarrow S) \wedge (S \rightarrow \neg R)) \vee (\neg E \wedge P)) && \text{Definição de Equivalência (8)} \\
&\equiv (P \wedge Q) \vee (\neg (\neg (\neg R) \vee S) \vee \neg ( \neg S \vee \neg R) \vee (\neg E \wedge P)) && \text{Lei de De Morgan (5), Definição de Implicação (7) (aplicada duas vezes)} \\
&\equiv (P \wedge Q) \vee (\neg (R \vee S) \vee \neg ( \neg S \vee \neg R) \vee (\neg E \wedge P)) && \text{Lei da Dupla Negação (12)} \\
&\equiv (P \wedge Q) \vee ((\neg R \wedge \neg S) \vee (S \wedge R) \vee (\neg E \wedge P)) && \text{Lei de De Morgan (6) (aplicada duas vezes), Lei da Dupla Negação (12)}
\end{align*}
$$

A atenta leitora pode notar que a negação de uma equivalência $\neg(X \leftrightarrow Y)$ também pode ser expressa como $(X \wedge \neg Y) \vee (\neg X \wedge Y)$.

**Exemplo 10**: $\neg(P \wedge (Q \vee R)) \leftrightarrow (\neg(S \rightarrow E) \vee \neg(P \rightarrow Q))$

Vamos simplificar ambos os lados da equivalência:
Lado Esquerdo (LE): $\neg(P \wedge (Q \vee R))$

$$
\begin{align*}
\text{LE} &\equiv \neg P \vee \neg(Q \vee R) && \text{Lei de De Morgan (5)} \\
&\equiv \neg P \vee (\neg Q \wedge \neg R) && \text{Lei de De Morgan (6)}
\end{align*}
$$

Lado Direito (LD): $\neg(S \rightarrow E) \vee \neg(P \rightarrow Q)$

$$
\begin{align*}
\text{LD} &\equiv \neg(\neg S \vee E) \vee \neg(\neg P \vee Q) && \text{Definição de Implicação (7) (aplicada duas vezes)} \\
&\equiv (S \wedge \neg E) \vee (P \wedge \neg Q) && \text{Lei de De Morgan (6) (aplicada duas vezes), Lei da Dupla Negação (12)}
\end{align*}
$$

Portanto, a expressão original é equivalente a:

$$
(\neg P \vee (\neg Q \wedge \neg R)) \leftrightarrow ((S \wedge \neg E) \vee (P \wedge \neg Q))
$$

Não foram utilizadas substituições temporárias como $F$ ou $G$ para manter a clareza.

A lógica proposicional é essencial para entendermos o mundo. É a base de argumentos sólidos e da avaliação de proposições. Nasceu da necessidade humana de buscar a verdade e resolver conflitos com a lógica. Mas sua beleza vai além da filosofia, do discurso e da matemática. É a fundação da álgebra de [George Boole](https://en.wikipedia.org/wiki/George_Boole), que sustenta o design de circuitos eletrônicos e a construção dos computadores modernos.

_Em sua dissertação de final de curso, [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) usou a álgebra booleana para simplificar circuitos de controle. Desde então, as operações básicas dessa álgebra — **AND**, **OR**, **NOT** — tornaram-se os blocos fundamentais dos sistemas digitais. Elas formam o núcleo dos computadores, dos celulares e, na verdade, de toda a nossa civilização digital. A lógica proposicional é a base de todo o raciocínio lógico. Como a tabela periódica para químicos ou as leis de Newton para físicos. Ela é simples, elegante e poderosa_.

Tão importante quanto o impacto da **lógica proposicional** na tecnologia digital é seu papel no pensamento racional, na tomada de decisões e na prova de teoremas. Neste caminho, nosso guia são as **regras de inferência**.

### Regras de Inferência

Regras de inferência são esquemas que proporcionam a estrutura para derivações lógicas. Base da tomada de decisão computacional. Elas definem os passos legítimos que podem ser aplicados a uma ou mais proposições, sejam elas atômicas ou Fórmulas Bem Formadas, para produzir uma proposição nova. Em outras palavras, uma regra de inferência é uma transformação sintática de Formas Bem Formadas que preserva a verdade.

Aqui uma regra de inferência será representada por:

$$\frac{P_1, P_2, ..., P_n}{C},$$

ou, eventualmente por:

$$P_1, P_2, ..., P_n \vdash C.$$

Onde o conjunto formado $P_1, P_2, ..., P_n$, chamado de contexto, ou antecedente, $\Gamma$, e $C$, chamado de conclusão, ou consequente, são Formulas Bem Formadas. A regra significa que se as proposições que constituem a conjunção expressa no contexto é verdadeira então a conclusão $C$, consequência, também será verdadeira.

Eu vou tentar usar contexto e conclusão. Mas a compassiva leitora deve me perdoar se eu escapar para antecedente e consequente. É apenas o hábito. 

Quando estudamos lógica, chamamos de **argumento** uma lista de proposições, que aqui são as premissas. Elas vêm seguidas de uma palavra ou expressão (portanto, consequentemente, desta forma) e de outra proposição, que chamamos de conclusão. A forma que usamos para representar isso é chamada de sequência de dedução. É uma forma de mostrar que, se a proposição colocada acima da linha horizontal for verdadeira, então estamos afirmando que todas as proposições $P_1, P_2, ..., P_n$ acima da linha são verdadeiras. E, por isso, a proposição abaixo da linha, a conclusão, também será verdadeira.

**As regras de inferência são o alicerce da lógica dedutiva e das provas matemáticas. Elas permitem que raciocínios complexos sejam divididos em passos simples, onde cada passo é justificado pela aplicação de uma regra de inferência**. A seguir, estão algumas das regras de inferência mais usadas:

#### Modus Ponens

A regra do **Modus Ponens** permite inferir uma conclusão a partir de uma implicação e de sua premissa antecedente. Se temos uma implicação $P\rightarrow Q$, e sabemos que $P$ é verdadeiro, então podemos concluir que $Q$ também é verdadeiro.

$$P \rightarrow Q$$

$$
\begin{aligned}
&P\\
\hline
&Q\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _se chover, $(P)$, então, $(\rightarrow)$, a rua ficará molhada, $(Q)$_;
- Proposição 2: _está chovendo, $(P)$ é verdadeira_.
- Conclusão: logo, _a rua ficará molhada, $(Q)$_.

Algumas aplicações do _Modus Ponens_:

- Derivar ações de regras e leis condicionais. Por exemplo:

  - Proposição: _se a velocidade, $V$, é maior que $80 \text{km/h}$, então é uma infração de trânsito, $IT$_.
  - Proposição: _joão está dirigindo, $ d$, A $90 \text{km/h}$_.
  - Conclusão: logo, _João cometeu uma infração de trânsito_.

$$V > 80 \rightarrow IT$$

$$
 \begin{aligned}
 &D = 90\\
 \hline
 &IT
 \end{aligned}
$$

- Aplicar implicações teóricas e chegar a novas conclusões. Por exemplo:

  - Proposição: _se um número é par, $P$, então é divisível por 2, $ d2$_.
  - Proposição: _128 é par_.
  - Conclusão: logo, _128 é divisível por 2_.

$$ x \text{ é par} \rightarrow \text{divisível por dois}$$

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
 &(AB = AC) \wedge (AB=CB) \text{ no triângulo} ABC\\
 \hline
 &\text{o triângulo } ABC \text{ é isósceles}
 \end{aligned}
$$

- Tirar conclusões com base no raciocínio condicional na vida cotidiana. Por exemplo:

  - Proposição: _se hoje não chover, então irei à praia_.
  - Proposição: _Hoje não choveu_.
  - Conclusão: logo, _irei à praia_.

$$\neg (\text{chover hoje}) \rightarrow \text{ir à praia}$$

$$
 \begin{aligned}
 &\neg (\text{choveu hoje})\\
 \hline
 &(\text{ir à praia})
 \end{aligned}
$$

#### Modus Tollens

A regra do Modus Tollens permite inferir a negação da premissa antecedente a partir de uma implicação e da negação de sua premissa consequente. Se temos uma implicação $P\rightarrow Q$, e sabemos que $Q$ é falso (ou seja, $\neg G$), então podemos concluir que $P$ também é falso.

$$P \rightarrow Q$$

$$
\begin{aligned}
&\neg Q\\
\hline
&\neg P\\
\end{aligned}
$$

Em linguagem natural:

- Proposição 1: _se uma pessoa tem 18 anos ou mais_, $(P)$, _então_, $(\rightarrow)$ _ela pode votar_, $(Q)$;
- Proposição 2: _maria não pode votar_$(\neg Q)$;
- Conclusão: logo, _maria não tem 18 anos ou mais_, $(\neg P)$.

Algumas aplicações do Modus Tollens:

- Refutar teorias mostrando que suas previsões são falsas. Por exemplo:

  - Proposição: _se a teoria da geração espontânea, $TG$ é correta, insetos irão se formar em carne deixada exposta ao ar, $I$_.
  - Proposição: _insetos não se formam em carne deixada exposta ao ar_.
  - Conclusão: logo, _a teoria da geração espontânea_ é falsa.

$$TG \rightarrow I$$

$$
 \begin{aligned}
 \neg I\\
 \hline
 \neg TG
 \end{aligned}
$$

- Identificar inconsistências ou contradições em raciocínios. Por exemplo:

  - Proposição: _se João, $J$, é mais alto, $>$, que Maria $m $, então Maria não é mais alta que João_.
  - Proposição: _Maria é mais alta que João_.
  - Conclusão: logo, _o raciocínio é inconsistente_.

$$(J>M) \rightarrow \neg(M>J)$$

$$
 \begin{aligned}
 (M>J)\\
 \hline
 \neg(J>M)
 \end{aligned}
$$

- Fazer deduções lógicas baseadas na negação da conclusão. Por exemplo:

  - Proposição: _se hoje, $H$, é sexta-feira, $se$, amanhã é sábado $SA$_.
  - Proposição: _amanhã não é sábado_.
  - Conclusão: logo, _hoje não é sexta-feira_.

$$(H=Se) \rightarrow (A=SA)$$

$$
 \begin{aligned}
 \neg(A=(Sa)\\
 \hline
 \neg(H=Se)
 \end{aligned}
$$

- Descobrir causas de eventos por eliminação de possibilidades. Por exemplo:

  - Proposição: _se a tomada está com defeito, $D$A lâmpada não acende $L$_.
  - Proposição: _a lâmpada não acendeu_.
  - Conclusão: logo, _a tomada deve estar com defeito_.

$$D \rightarrow \neg L$$

$$
 \begin{aligned}
 &\neg L\\
 \hline
 &D
 \end{aligned}
$$

#### Dupla Negação

A regra da Dupla Negação permite eliminar uma dupla negação, inferindo a afirmação original. A negação de uma negação é equivalente à afirmação original. Esta regra é importante para simplificar expressões lógicas.

$$\neg \neg F$$

$$
\begin{aligned}
&\neg \neg F\\
\hline
&F\\
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

- Proposição: _não é verdade, $(\neg Q)$, que maria não, $(\neg Q)$, está feliz, $(P)$_.
- Conclusão: logo, _maria está feliz, $(P)$_.

A dupla negação pode parecer desnecessária, mas ela tem algumas aplicações na lógica:

- Simplifica expressões logicas: remover duplas negações ajuda a simplificar e a normalizar expressões complexas, tornando-as mais fáceis de analisar. Por exemplo, transformar _não é verdade que não está chovendo_ em simplesmente _está chovendo_.

$$\neg \neg \text{Está chovendo} \Leftrightarrow \text{Está chovendo}$$

- Preserva o valor de verdade: inserir ou remover duplas negações não altera o valor de verdade original de uma proposição. Isso permite transformar proposições em formas logicamente equivalentes.

- Auxilia provas indiretas: em provas por contradição, ou contrapositiva, introduzir uma dupla negação permite assumir o oposto do que se quer provar e derivar uma contradição. Isso, indiretamente, prova a proposição original.

- Conecta Lógica Proposicional e de predicados: em Lógica Predicativa, a negação de quantificadores universais e existenciais envolve dupla negação. Por exemplo, a negação de _todo $x$ é $P$_ é _existe algum $x$ tal que $P(x)$ não é verdadeiro_.

$$\neg \forall x P(x) \Leftrightarrow \exists x \neg P(x)$$

- Permite provar equivalências: uma identidade ou lei importante na lógica é que a dupla negação de uma proposição é logicamente equivalente à proposição original. A regra da dupla negação permite formalmente provar essa equivalência.

$$\neg \neg P \Leftrightarrow P$$

#### Adição

A regra da Adição permite adicionar uma disjunção a uma afirmação, resultando em uma nova disjunção verdadeira. Esta regra é útil para introduzir alternativas em nosso raciocínio dedutivo.

$$F$$

$$\begin{aligned}
&F\\
\hline
&F \vee G\\
\end{aligned}$$

$$\begin{aligned}
&G\\
\hline
&F \vee G\\
\end{aligned}$$

Em linguagem natural:

- Proposição: _o céu está azul, $(P)$_;
- Conclusão: logo, _o céu está azul ou gatos podem voar, $(P \lor Q)$_.

A regra da Adição permite introduzir uma disjunção em uma prova ou argumento lógico. Especificamente, ela nos permite inferir uma disjunção $P\vee Q$A partir de uma das afirmações disjuntivas ($P$ ou $Q$) individualmente.

Alguns usos e aplicações importantes da regra da Adição:

- Introduzir alternativas ou possibilidades em um argumento: por exemplo, dado que _João está em casa_, podemos concluir que _João está em casa OR no trabalho_. E expandir este _OR_ o quanto seja necessário para explicitar os lugares onde joão está.

- Combinar afirmações em novas disjunções: dadas duas afirmações quaisquer $P$ e $Q$, podemos inferir que $P$ ou $Q$ é verdadeiro.

- Criar casos ou opções exaustivas em uma prova: podemos derivar uma disjunção que cubra todas as possibilidades relevantes. Lembre-se do pobre _joão_.

- Iniciar provas por casos: ao assumir cada disjuntiva separadamente, podemos provar teoremas por casos exaustivos.

- Realizar provas indiretas: ao assumir a negação de uma disjunção, podemos chegar a uma contradição e provar a disjunção original.

A regra da Adição amplia nossas capacidades de prova e abordagem de problemas.

#### Modus Tollendo Ponens

O Modus Tollendo Ponens permite inferir uma disjunção a partir da negação da outra disjunção.

Dada uma disjunção $P\vee Q$:

- Se $\neg P$, então $Q$
- Se $\neg Q$, então $P$

Esta regra nos ajuda a chegar a conclusões a partir de disjunções, por exclusão de alternativas.

$$P \vee Q$$

$$\begin{aligned}
&\neg P\\
\hline
&Q\\
\end{aligned}$$

$$\begin{aligned}
&\neg Q\\
\hline
&P\\
\end{aligned}$$

Em linguagem natural:

- Proposição 1: _ou o céu está azul ou a grama é roxa_;
- Proposição 2: _a grama não é roxa_;
- Conclusão: logo, _o céu está azul_.

Algumas aplicações do Modus Tollendo Ponens:

- Derivar ações a partir de regras disjuntivas. Por exemplo:

  - Proposição: _ou João vai à praia, $P$ ou João vai ao cinema, $c$_;
  - Proposição: _João não vai ao cinema_, $\neg C$;
  - Conclusão: logo, _João vai à praia_.

$$P \vee C$$

$$\begin{aligned}
&\neg C\\
\hline
&P
\end{aligned}$$

- Simplificar casos em provas por exaustão. Por exemplo:

  - Proposição: _o número é par, $P$, ou ímpar, $I$_;
  - Proposição: _o número não é ímpar, $\neg P$_;
  - Conclusão: logo, _o número é par_.

$$P \vee I$$

$$\begin{aligned}
&\neg I\\
\hline
&P
\end{aligned}$$

- Eliminar opções em raciocínio dedutivo. Por exemplo:

  - Proposição: _ou João estava em casa, $c$, ou João estava no trabalho, $t$_;
  - Proposição: _João não estava em casa_;
  - Conclusão: logo, _João estava no trabalho_.

$$C \vee T$$

$$\begin{aligned}
&\neg C\\
\hline
&T
\end{aligned}$$

- Fazer prova indireta da disjunção. Por exemplo:

  - Proposição: _1 é par, $1P$, ou 1 é ímpar, $1I$_;
  - Proposição: _1 não é par_;
  - Conclusão: logo, _1 é ímpar_.

$$1P \vee 1I$$

$$\begin{aligned}
&\neg 1P\\
\hline
&1I
\end{aligned}$$

#### Adjunção

A regra da Adjunção permite combinar duas afirmações em uma conjunção. Esta regra é útil para juntar duas premissas em uma única afirmação conjuntiva.

$$F$$

$$G$$

$$\begin{aligned}
&F\\
&G\\
\hline
&F \land G\\
\end{aligned}$$

Em linguagem natural:

- proposição 1: _o céu está azul_;
- proposição 2: _os pássaros estão cantando_;
- Conclusão: logo, _o céu está azul e os pássaros estão cantando_.

Algumas aplicações da Adjunção:

- Combinar proposições relacionadas em argumentos. Por exemplo:

  - Proposição: _o céu está nublado, $n$_;
  - Proposição: _está ventando, $V$_;
  - Conclusão: logo, _o céu está nublado e está ventando_.

$$\begin{aligned}
&N\\
&V\\
\hline
&N \land V
\end{aligned}$$

- Criar declarações conjuntivas complexas. Por exemplo:

  - Proposição: _1 é número natural, $n1$_;
  - Proposição: _2 é número natural $n2$_;
  - Conclusão: logo, _1 é número natural **e** 2 é número natural_.

$$\begin{aligned}
&N1\\
&N2\\
\hline
&N1 \land N2
\end{aligned}$$

- Derivar novas informações da interseção de fatos conhecidos. Por exemplo:

  - Proposição: _o gato está em cima do tapete, $gT$_;
  - Proposição: _o rato está em cima do tapete, $rT$_;
  - Conclusão: logo, _o gato **e** o rato estão em cima do tapete_.

$$\begin{aligned}
&GT\\
&RT\\
\hline
&G_T \land R_T
\end{aligned}$$

- Fazer deduções lógicas baseadas em múltiplas proposições. Por exemplo:

  - Proposição: _2 + 2 = 4_;
  - Proposição: _4 x 4 = 16_;
  - Conclusão: logo, _$(2 + 2 = 4) ∧ (4 × 4 = 16)$_.

$$\begin{aligned}
&(2 + 2 = 4)\\
&(4 \times 4 = 16)\\
\hline
&(2 + 2 = 4) \land (4 \times 4 = 16)
\end{aligned}$$

#### Simplificação

A regra da Simplificação permite inferir uma conjunção a partir de uma conjunção composta. Esta regra nos permite derivar ambos os elementos de uma conjunção, a partir da afirmação conjuntiva.

$$F \land G$$

$$\begin{aligned}
&F \land G\\
\hline
&F\\
\end{aligned}$$

$$\begin{aligned}
&F \land G\\
\hline
&G\\
\end{aligned}$$

Em linguagem natural:

- proposição: _o céu está azul e os pássaros estão cantando_;
- Conclusão: logo, _o céu está azul. E os pássaros estão cantando_.

Algumas aplicações da Simplificação:

- Derivar elementos de conjunções complexas. Por exemplo:

  - Proposição: _hoje está chovendo, $c$, e fazendo frio, $F$_;
  - Conclusão: logo, _está chovendo_.

$$\begin{aligned}
&C \land F\\
\hline
&C
\end{aligned}$$

- Simplificar provas baseadas em conjunções. Por exemplo:

  - Proposição: _2 é par, $2P$, e 3 é ímpar, $3P$_;
  - Conclusão: logo, _3 é ímpar, $3I$_.

$$\begin{aligned}
&2P \land 3I\\
\hline
&3I
\end{aligned}$$

- Inferir detalhes específicos de declarações complexas. Por exemplo:

  - Proposição: _o gato está dormindo, $d$, e ronronando, $R$_;
  - Conclusão: logo, _o gato está ronronando_.

$$\begin{aligned}
&D \land R\\
\hline
&R
\end{aligned}$$

- Derivar informações de premissas conjuntivas. Por exemplo:

  - Proposição: _está chovendo, $J$, e o jogo foi cancelado, $c$_;
  - Conclusão: logo, _o jogo foi cancelado_.

$$\begin{aligned}
&C \land J\\
\hline
&J
\end{aligned}$$

#### Bicondicionalidade

A regra da Bicondicionalidade permite inferir uma bicondicional a partir de duas condicionais. Esta regra nos permite combinar duas implicações para obter uma afirmação de equivalência lógica.

$$F \rightarrow G$$

$$G \rightarrow F$$

$$\begin{aligned}
&G \rightarrow F\\
\hline
&F \leftrightarrow G\\
\end{aligned}$$

Em linguagem natural:

- proposição _1: se está chovendo, então a rua está molhada_;
- proposição _2: se a rua está molhada, então está chovendo_;
- Conclusão: logo, _está chovendo se e somente se a rua está molhada_.

Algumas aplicações da Bicondicionalidade:

- Inferir equivalências lógicas a partir de implicações bidirecionais. Por exemplo:

  - Proposição: _se chove, $c$ então a rua fica molhada, $m$_;
  - Proposição: _se a rua fica molhada, então chove_;
  - Conclusão: logo, _chove se e somente se a rua fica molhada_.

$$C \rightarrow M$$

$$\begin{aligned}
&M \rightarrow C\\
\hline
&C \leftrightarrow M
\end{aligned}$$

- Simplificar relações recíprocas. Por exemplo:

  - Proposição: _se um número é múltiplo de 2, $M2$ então é par, $P$_;
  - Proposição: _se um número é par, então é múltiplo de 2_;
  - Conclusão: logo, _um número é par se e somente se é múltiplo de 2_.

$$P \rightarrow M2$$

$$\begin{aligned}
&M2 \rightarrow P\\
\hline
&P \leftrightarrow M2
\end{aligned}$$

- Estabelecer equivalências matemáticas. Por exemplo:

  - Proposição: _se $x^2 = 25$, então $x = 5$_;
  - Proposição: _se $x = 5$, então $x^2 = 25$_;
  - Conclusão: logo, _$x^2 = 25$ se e somente se $x = 5$_.

$$(x^2 = 25) \rightarrow (x = 5)$$

$$\begin{aligned}
&(x = 5) \rightarrow (x^2 = 25)\\
\hline
&(x^2 = 25) \leftrightarrow (x = 5)
\end{aligned}$$

- Provar relações de definição mútua. Por exemplo:

  - Proposição: _se figura é um quadrado, $Q$, então tem 4 lados iguais, $4L$_;
  - Proposição: _se figura tem 4 lados iguais, é um quadrado_;
  - Conclusão: logo, _figura é quadrado se e somente se tem 4 lados iguais_.

$$Q \rightarrow 4L$$

$$\begin{aligned}
&4L \rightarrow Q\\
\hline
&Q \leftrightarrow 4L
\end{aligned}$$

#### Equivalência

A regra da Equivalência permite inferir uma afirmação ou sua negação a partir de uma bicondicional. Esta regra nos permite aplicar bicondicionais para derivar novas afirmações baseadas nas equivalências lógicas.

$$F \leftrightarrow G$$

$$\begin{aligned}
&F\\
\hline
&G\\
\end{aligned}$$

$$F \leftrightarrow G$$

$$\begin{aligned}
&G\\
\hline
&F\\
\end{aligned}$$

$$F \leftrightarrow G$$

$$\begin{aligned}
&\neg F\\
\hline
&\neg G\\
\end{aligned}$$

$$F \leftrightarrow G$$

$$\begin{aligned}
&\neg G\\
\hline
&\neg F\\
\end{aligned}$$

Em linguagem natural:

- proposição 1: _está chovendo se e somente se a rua está molhada_;
- proposição 2: _está chovendo_;
- Conclusão: logo, _a rua está molhada_.

Algumas aplicações da Equivalência:

1. Inferir fatos de equivalências estabelecidas. Por exemplo:

   - Proposição: _o número é par, $P$ se e somente se for divisível por 2, $d2$_;
   - Proposição: _156 é divisível por 2_;
   - Conclusão: logo, _156 é par_.

   $$P \leftrightarrow D2$$

   $$\begin{aligned}
   &D2(156)\\
   \hline
   &P(156)
   \end{aligned}$$

2. Derivar negações de equivalências. Por exemplo:

   - Proposição: _$x$ é negativo se e somente se $x < 0$_;
   - Proposição: _$x$ não é negativo_;
   - Conclusão: logo, _$x$ não é menor que $0$_.

   $$ N \leftrightarrow (x < 0)$$

   $$\begin{aligned}
   &\neg N\\
   \hline
   &\neg (x < 0)
   \end{aligned}$$

3. Fazer deduções baseadas em definições. Por exemplo:

   - Proposição: _número ímpar é definido como não divisível, $nD2$, por $2$_;
   - Proposição: _$9$ não é divisível por $2$_;
   - Conclusão: logo, _$9$ é ímpar_.

   $$I \leftrightarrow \neg ND2$$

   $$\begin{aligned}
   &\neg D_2(9)\\
   \hline
   &I(9)
   \end{aligned}$$

  | Regra | Descrição | Fórmula |
   |---|---|---|
   | Modus Ponens | Se $P \rightarrow Q$ e $P$ são verdadeiros, então $Q$ também é verdadeiro. | $\frac{P, P \rightarrow Q}{Q}$ |
   | Modus Tollens | Se $P \rightarrow Q$ e $\neg Q$ são verdadeiros, então $\neg P$ também é verdadeiro. | $\frac{\neg Q, P \rightarrow Q}{\neg P}$ |
   | Dupla Negação | A negação de uma negação é equivalente à afirmação original. | $\frac{\neg \neg P}{P}$ |
   | Adição | Se $P$ é verdadeiro, então $P \vee Q$ também é verdadeiro. | $\frac{P}{P \vee Q}$ |
   | Adjunção | Se $P$ e $Q$ são verdadeiros, então $P \wedge Q$ é verdadeiro. | $\frac{P, Q}{P \wedge Q}$ |
   | Simplificação | Se $P \wedge Q$ é verdadeiro, então $P$ (ou $Q$) é verdadeiro. | $\frac{P \wedge Q}{P}$ |
   | Bicondicionalidade | Se $P \leftrightarrow Q$, então $P \rightarrow Q$ e $Q \rightarrow P$ são verdadeiros. | $\frac{P \leftrightarrow Q}{P \rightarrow Q, Q \rightarrow P}$ |
  
  _Tabela 4 - Resumo dos métodos de inferência._{: class="legend"}

### Classificação das Fórmulas Proposicionais

Podemos classificar fórmulas proposicionais de acordo com suas propriedades semânticas, analisando suas tabelas-verdade. Seja $R$ uma fórmula proposicional:

- $R$ é **satisfatível** se sua Tabela Verdade contém pelo menos uma linha verdadeira. Considere:$P\wedge Q$.

$$\begin{array}{|c|c|c|}
 \hline
 P & Q & P \land Q \\
 \hline
 F & F & F \\
 \hline
 F & T & F \\
 \hline
 T & F & F \\
 \hline
 T & T & T \\
 \hline
 \end{array}$$

- $R$ é **insatisfatível** se sua Tabela Verdade contém apenas linhas falsas. Exemplo:$P\wedge \neg P$.
- $R$ é **falsificável** se sua Tabela Verdade contém pelo menos uma linha falsa. Exemplo:$P\wedge Q$.
- $R$ é **válida** se sua Tabela Verdade contém apenas linhas verdadeiras. Exemplo:$P\vee \neg P$.

Note que:

- Se $A$ é válida, então $A$ é satisfatível.
- Se $A$ é insatisfatível, então $A$ é falsificável.

Fórmulas válidas são importantes na lógica proposicional, representando argumentos sempre verdadeiros independentemente da valoração de suas variáveis proposicionais atômicas. Na verdade, esta classificação será importante para:

1. **Análise de Argumentos**: Se uma argumentação lógica pode ser representada por uma fórmula que é insatisfatível, então sabemos que o argumento é inválido ou inconsistente. Isso é frequentemente usado em lógica e filosofia para analisar a validade dos argumentos.

2. **Prova de Teoremas**: Na prova de teoremas, essas classificações são úteis. Quando estamos tentando provar que uma fórmula é uma tautologia, podemos usar essas classificações para simplificar a tarefa. Podemos mostrar que a negação da fórmula é insatisfatível, mostrando que a fórmula original é uma tautologia.

3. **Simplificação de Fórmulas**: Na simplificação de fórmulas, essas classificações também são úteis. Se temos uma fórmula complexa e podemos mostrar que uma parte dela é uma tautologia, podemos simplificar a fórmula removendo essa parte. Similarmente, se uma parte da fórmula é uma contradição (ou seja, é insatisfatível), sabemos que a fórmula inteira é insatisfatível.

4. **Construção de Argumentos**: Na construção de argumentos, estas classificações são úteis para garantir que os argumentos são válidos. Se estamos construindo um argumento e podemos mostrar que ele é representado por uma fórmula que é satisfatível (mas não uma tautologia), sabemos que existem algumas circunstâncias em que o argumento é válido e outras em que não é.

## Provas

A matemática respira prova. Nenhuma sentença matemática tem qualquer valor se não for provada. As verdades da aritmética devem ser estabelecidas com rigor lógico; as conjecturas da geometria, confirmadas por construtos infalíveis. Cada novo teorema se ergue sobre os ombros de gigantes – um edifício de razão cuidadosamente erigido.

A beleza da lógica proposicional é revelar, nas entranhas da matemática, um método para destilar a verdade. Seus símbolos e regras exaltam nosso raciocínio e nos elevam da desordem da intuição. Com poucos elementos simples – variáveis, conectivos, axiomas – podemos capturar verdades absolutas no âmbito do pensamento simbólico.

Considere um sistema proposicional, com suas Fórmulas Bem Formadas, suas transformações válidas. Ainda que simples, vemos nesse sistema o que há de profundo na natureza da prova. Seus teoremas irradiam correção; suas demonstrações, poder dedutivo. Dentro deste sistema austero reside a beleza em uma estética hermética, mas que desvelada faz brilhar a luz da razão e do entendimento.

### Contrapositivas e Recíprocas

As implicações são um problema, do ponto de vista da matemática. Sentenças do tipo _se...então_ induzem uma conclusão. Provar estas sentenças é uma preocupação constante da matemática [^3]. Dada uma implicação, existem duas fórmulas relacionadas que ocorrem com tanta frequência que possuem nomes especiais: contrapositivas e recíprocas. Antes de mergulharmos em contrapositivas, precisamos visitar alguns portos.

### Logicamente Equivalente

Vamos imaginar um mundo de fórmulas que consistem apenas em duas proposições:$P$ e $Q$. Usando os operadores da Lógica Proposicional podemos escrever um número muito grande de fórmulas diferentes combinando estas duas proposições.

A coisa interessante sobre as fórmulas que conseguimos criar com apenas duas proposições é que cada uma dessas fórmulas tem uma Tabela Verdade com exatamente quatro linhas, $2^2=4$. Mesmo que isso pareça surpreendente, só existem dezesseis configurações possíveis para a última coluna de todas as Tabelas Verdades de todas as tabelas que podemos criar, $2^4=16$. Como resultado, muitas fórmulas compartilham a mesma configuração final em suas Tabelas Verdade. Todas as fórmulas que possuem a mesma configuração na última coluna são equivalentes.Terei ouvido um viva?

Com um pouco mais de formalidade podemos dizer que: considere as proposições $A$ e $B$. Estas proposições serão ditas logicamente equivalentes se, e somente se, a proposição $A \Leftrightarrow B$ for uma tautologia.

**Exemplo: 1** Vamos mostrar que $P\rightarrow Q$ é logicamente equivalente a $\neg Q \rightarrow \neg P$.

**Solução**: Para isso, verificaremos se a coluna do conectivo principal na Tabela Verdade para a proposição bicondicional $(P\rightarrow Q) \leftrightarrow (\neg Q \rightarrow \neg P)$ contém apenas valores verdadeiros:

$$
\begin{array}{|c|c|c|c|c|c|}
 \hline
 P & Q & P \rightarrow Q & \neg Q & \neg P & \neg Q \rightarrow \neg P & (P \rightarrow Q) \leftrightarrow (\neg Q \rightarrow \neg P) \\
 \hline
 F & F & T & T & T & T & T \\
 \hline
 F & T & T & F & T & T & T \\
 \hline
 T & F & F & T & F & F & T \\
 \hline
 T & T & T & F & F & T & T \\
 \hline
 \end{array}
$$

Como a coluna da operação principal $(P\rightarrow Q) \leftrightarrow (\neg Q \rightarrow \neg P)$ contém apenas valores verdadeiros ($T$), a proposição bicondicional é uma tautologia. Consequentemente, as fórmulas $P\rightarrow Q$ e $\neg Q \rightarrow \neg P$ são logicamente equivalentes.

**Exemplo 2**: Vamos mostrar que $P \land Q$ não é logicamente equivalente a $P \lor Q$.

**Solução**
Para mostrar que $P \land Q$ não é logicamente equivalente a $P \lor Q$, precisamos verificar se a proposição bicondicional $(P \land Q) \leftrightarrow (P \lor Q)$ é uma tautologia. Se não for uma tautologia, então as duas fórmulas não são logicamente equivalentes.

Construindo a Tabela Verdade (usando T para Verdadeiro e F para Falso):

$$
\begin{array}{|c|c|c|c|c|}
\hline
P & Q & P \land Q & P \lor Q & (P \land Q) \leftrightarrow (P \lor Q) \\
\hline
T & T & T & T & T \\
T & F & F & T & F \\
F & T & F & T & F \\
F & F & F & F & T \\
\hline
\end{array}
$$

Como a última coluna da Tabela Verdade para $(P \land Q) \leftrightarrow (P \lor Q)$ não contém apenas valores $T$ (há ocorrências de $F$), a proposição bicondicional não é uma tautologia. Portanto, $P \land Q$ e $P \lor Q$ não são logicamente equivalentes.

**Exemplo 3**: Vamos mostrar que $P\rightarrow Q$ é logicamente equivalente a $\neg P \lor Q$.

**Solução**
Verificando a Tabela Verdade:

$$
\begin{array}{|c|c|c|c|c|c|}
 \hline
 P & Q & \neg P & \neg P \lor Q & P \rightarrow Q \leftrightarrow \neg P \lor Q\\
 \hline
 V & V & F & V & V\\
 \hline
 V & F & F & F & V\\
 \hline
 F & V & V & V & V\\
 \hline
 F & F & V & V & V\\ \hline
 \end{array}
$$

Neste caso $P\rightarrow Q$ e $\neg P \lor Q$ são logicamente equivalentes.

Em resumo, duas fórmulas $P$ e $Q$, atômicas, ou não, são equivalentes se quando $P$ for verdadeiro, $Q$ também será e vice-versa. Agora que já sabemos o que significa _logicamente equivalentes_ podemos entender o que é uma proposição contrapositiva.

### Contrapositiva

A contrapositiva de uma implicação é obtida invertendo-se o antecedente e o consequente da implicação original e negando-os. Por exemplo, considere a seguinte implicação: _se chove, então a rua fica molhada_ sua contrapositiva poderia ser: _se a rua não está molhada, então não choveu_. Sejam $P$ e $Q$ fórmulas proposicionais derivadas de uma sentença do tipo _se ... então_. A implicação $P\rightarrow Q$ representa a sentença Se $P$, então $Q$. Neste caso, A contrapositiva de $P\rightarrow Q$ será dada por:

$$
\begin{aligned}
\lnot Q \rightarrow \lnot P
\end{aligned}
$$

A contrapositiva pode ser lida como _se não $Q$, então não $P$_. Em outras palavras estamos dizendo: _Se $Q$ é falso, então $P$ é falso_. A contrapositiva de uma fórmula é importante porque, frequentemente, é mais fácil provar a contrapositiva de uma fórmula que a própria fórmula. E, como a contrapositiva é logicamente equivalente a sua formula, provar a contrapositiva é provar a fórmula. Como a contrapositiva de uma implicação e a própria implicação são logicamente equivalentes, se provamos uma, a outra está provada. Além disso, a contrapositiva preserva a validade das implicações proposicionais. Finalmente, observe que a contrapositiva troca o antecedente pelo negação do consequente e vice-versa.

**Exemplo 1**:

A contrapositiva de $P\rightarrow (Q \lor R)$ é $\lnot(Q \lor R) \rightarrow \neg P$.

**Exemplo 2**:
Dizemos que uma função é injetora se $x \neq y $implica $f(x) \neq f(y)$. A contrapositiva desta implicação é: se $f(x) = f(y)$ então $x = y$.

O Exemplo 2 é uma prova de conceito. Normalmente é mais fácil assumir $f(x) = f(y)$ e deduzir $x = y$ do que assumir $x \neq y$ e deduzir $f(x) \neq f(y)$. Isto pouco tem a ver com funções e muito com o fato de que $x \neq y$ geralmente não é uma informação útil.

O que torna a contrapositiva importante é que toda Fórmula Bem Formada é logicamente equivalente à sua contrapositiva. Consequentemente, se queremos provar que uma função é injetora, é suficiente provar que se $f(x) = f(y)$ então $x = y$.

A contrapositiva funciona para qualquer declaração condicional, e matemáticos gastam muito tempo provando declarações condicionais.

O que não podemos esquecer de jeito nenhum é que toda fórmula condicional terá a forma $P\rightarrow Q$. Mostramos que isso é logicamente equivalente a $\lnot Q \rightarrow \lnot P$ verificando a Tabela Verdade para a declaração bicondicional construída a partir dessas fórmulas. E que para obter a contrapositiva basta inverter antecedente e consequente e negar ambos. mantendo a relação lógica entre os termos da implicação.

### Recíproca

A recíproca, também conhecida como _conversa_ por alguns acadêmicos brasileiros, é obtida apenas invertendo antecedente e consequente. Então, considerando a recíproca da condicional$P\rightarrow Q$ será $ q \rightarrow P$. Destoando da contrapositiva a recíproca não é necessariamente equivalente à implicação original. Além disso, a contrapositiva preserva a equivalência lógica, a recíproca não.

**Exemplo 1**:
A conversa de $P\rightarrow (Q \lor R)$ será $(Q \lor R) \rightarrow P$.

**Exemplo 2**:
Dizemos que uma função é bem definida se cada entrada tem uma saída única. Assim, uma função é bem definida se $x = y$ implica $f(x) = f(y)$. Observe estas fórmulas:

1. $f(x)$ é bem definida significa que $x = y \rightarrow f(x) = f(y)$.

2. $f(x)$ é injetora significa que $f(x) = f(y) \rightarrow x = y$.

Podemos ver que _$f(x)$ é bem definida_ é a recíproca de _$f(x)$ é injetora_.

Para provar uma bicondicional como _o número é primo se e somente se o número é ímpar_, um matemático frequentemente prova _se o número é primo, então o número é ímpar_ e depois prova a recíproca, _se o número é ímpar, então o número é primo_. Nenhuma dessas etapas pode ser pulada, pois uma implicação e sua recíproca podem não ser logicamente equivalentes. Por exemplo, pode-se facilmente mostrar que _se o número é par, então o número é divisível por 2_ não é logicamente equivalente à sua recíproca _se o número é divisível por 2, então o número é par_. Algumas fórmulas como _se 5 é ímpar, então 5 é ímpar_ são equivalentes às suas recíprocas por coincidência. Para resumir, uma implicação é sempre equivalente à sua contrapositiva, mas pode não ser equivalente à sua recíproca.

### Análise de Argumentos

Quando vimos regras de inferência, sem muitos floreios, definimos argumentos. mas, sem usar a palavra argumento em nenhum lugar. Vamos voltar um pouco. Definiremos um argumento proposicionalmente como sendo uma regra de inferência, então um argumento será definido por um conjunto de proposições. Quando estamos analisando argumentos chamamos as proposições de premissas logo:

$$\frac{P_1, P_2, ..., P_n}{C}$$

Onde o conjunto formado $P_1, P_2, ..., P_n$, chamado de antecedente, e $ c$, chamado de conclusão. Dizemos que o argumento será válido, só e somente se, a implicação definida por $P_1, P_2, ..., P_n \rightarrow C$ for uma tautologia. Neste caso, é muito importante percebermos que a conclusão de um argumento logicamente válido não é necessariamente verdadeira. A única coisa que a validade lógica garante é que se todas as premissas forem verdadeiras, a conclusão será verdadeira.

Podemos recuperar as regras de inferência e observá-las pelo ponto de vista da análise de argumentos. Se fizermos isso, vamos encontrar alguns formatos comuns:

**Modus Ponens**: se é verdade que se eu estudar para o exame $P$, então eu passarei no exame, $Q$, e também é verdade que eu estudei para o exame $P$, então podemos concluir que eu passarei no exame $Q$.

matematicamente, sejam $P$ e $Q$ Proposições. A forma do _Modus Ponens_ é a seguinte:

$$
\begin{align*}
& \quad P \rightarrow Q \quad \text{(Se P, então Q)} \\
& \quad P \quad \text{(P é verdadeiro)} \\
\hline
& \quad Q \quad \text{(Portanto, Q é verdadeiro)}
\end{align*}
$$

Cuja Tabela Verdade será:

$$
\begin{array}{|c|c|c|}
\hline
P & Q & P \rightarrow Q \\
\hline
T & T & T \\
T & F & F \\
F & T & T \\
F & F & T \\
\hline
\end{array}
$$

SSe olharmos para a primeira linha, se $P$ é verdadeiro e $P→ Q$ é verdadeiro, então $Q$ é necessariamente verdadeiro, o que é exatamente a forma de Modus Ponens.

**Modus Tollens** : se é verdade que se uma pessoa é um pássaro $P$, então essa pessoa pode voar $Q$, e também é verdade que essa pessoa não pode voar $\neg Q$, então podemos concluir que essa pessoa não é um pássaro $\neg P$. Ou:

Sejam $P$ e $Q$ Proposições. A forma do Modus Tollens é a seguinte:

$$
\begin{align*}
& \quad P \rightarrow Q \quad \text{(Se P, então Q)} \\
& \quad \neg Q \quad \text{(Q é falso)} \\
\hline
& \quad \neg P \quad \text{(Portanto, P é falso)}
\end{align*}
$$

Cuja Tabela Verdade será dada por:

$$
\begin{array}{|c|c|c|c|c|}
\hline
P & Q & \neg Q & P \rightarrow Q & \neg P \\
\hline
T & T & F & T & F \\
T & F & T & F & F \\
F & T & F & T & T \\
F & F & T & T & T \\
\hline
\end{array}
$$

Se olharmos para a quarta linha, se $Q$ é falso e $P\rightarrow Q$ é verdadeiro, então $P$ é necessariamente falso, o que é exatamente a forma de Modus Tollens.

**Silogismo Hipotético** : _se é verdade que se eu acordar cedo $P$, então eu irei correr $Q$, e também é verdade que se eu correr $Q$, então eu irei tomar um café da manhã saudável $R$, podemos concluir que se eu acordar cedo $P$, então eu irei tomar um café da manhã saudável $R$_.

matematicamente teremos: sejam $P$, $Q$ e $R$ Proposições. A forma do Silogismo Hipotético é a seguinte:

$$
\begin{align*}
& \quad P \rightarrow Q \quad \text{(Se P, então Q)} \\
& \quad Q \rightarrow R \quad \text{(Se Q, então R)} \\
\hline
& \quad P \rightarrow R \quad \text{(Portanto, se P, então R)}
\end{align*}
$$

Cuja Tabela Verdade será:

$$
\begin{array}{|c|c|c|c|c|}
\hline
P & Q & R & P \rightarrow Q & Q \rightarrow R & P \rightarrow R \\
\hline
T & T & T & T & T & T \\
T & T & F & T & F & F \\
T & F & T & F & T & T \\
T & F & F & F & T & T \\
F & T & T & T & T & T \\
F & T & F & T & F & T \\
F & F & T & T & T & T \\
F & F & F & T & T & T \\
\hline
\end{array}
$$

Se olharmos para a primeira linha, se $P$ é verdadeiro, $P\rightarrow Q$ é verdadeiro e $ q \rightarrow r $ é verdadeiro, então $P\rightarrow r $ é necessariamente verdadeiro, o que é exatamente a forma de Silogismo Hipotético.

**Silogismo Disjuntivo**: _se é verdade que ou eu vou ao cinema $P$ ou eu vou ao teatro $Q$, e também é verdade que eu não vou ao cinema $\neg P$, então podemos concluir que eu vou ao teatro $Q$_. Ou, com um pouco mais de formalidade:

Sejam $P$ e $Q$ Proposições. A forma do Silogismo Disjuntivo é a seguinte:

$$
\begin{align*}
& \quad P \lor Q \quad \text{(P ou Q)} \\
& \quad \neg P \quad \text{(não P)} \\
\hline
&\quad Q \quad \text{(Portanto, Q)}
\end{align*}
$$

A Tabela Verdade será:

$$
\begin{array}{|c|c|c|c|}
\hline
P & Q & \neg P & P \lor Q \\
\hline
T & T & F & T \\
T & F & F & T \\
F & T & T & T \\
F & F & T & F \\
\hline
\end{array}
$$

Se olharmos para a terceira linha, se $P$ é falso e $P\vee Q$ é verdadeiro, então $Q$ é necessariamente verdadeiro, o que é exatamente a forma de Silogismo Disjuntivo.

Não podemos esquecer: um argumento só é válido se, e somente se, a proposição condicional que o expresse seja uma tautologia. Agora podemos definir um sistema de prova.

### Finalmente, um Sistema de Prova

Ainda estamos no domínio da Lógica Proposicional e vamos definir um sistema de prova simples e direto chamado de $\mathfrak{L}$ desenvolvido por [John Lemmon](https://en.wikipedia.org/wiki/John_Lemmon) na primeira parte do século XX. Vamos construir a prova e, sintaticamente, em cada linha da nossa prova teremos:

- **um axioma** de $\mathfrak{L}$. Um axioma é uma fórmula ou proposição que é aceita como verdadeira primitivamente, sem necessidade de demonstração. Por exemplo: $(p \rightarrow q) \rightarrow ((q \rightarrow r) \rightarrow (p \rightarrow r))$;
- **o resultado da aplicação do _Modus Ponens_**;
- **uma hipótese**, na forma de fórmula;
- **ou um lema**, uma proposição auxiliar demonstrável utilizada como passo intermediário na prova. Por exemplo: a derivação de fórmulas menores.

**Axiomas** são preposições consideradas como verdades, são absolutos. **Lemas** são passos intermediários no processo de prova, pequenos teoremas já provados e, finalmente temos o **teorema**: representado por $\varphi$. Um teorema é uma fórmula demonstrável a partir de axiomas, lemas e das regras de inferência do sistema. Vamos começar dos axiomas.

Existem três axiomas no sistema $\mathfrak{L}$. Estes axiomas formam a base do sistema dedutivo $\mathfrak{L}$ em lógica proposicional. Eles capturam propriedades fundamentais das implicações que permitem derivar teoremas válidos.

**Axioma 1**: $A \rightarrow (B \rightarrow A)$, este axioma estabelece que se $A$ é verdadeiro, então a implicação $B \rightarrow A$ também é verdadeira, independentemente de $B$. Isso porque a implicação $B \rightarrow A$ só será falsa se $B$ for verdadeiro e $A$ falso, o que não pode ocorrer se $A$ é inicialmente verdadeiro.

**Axioma 2**: $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \rightarrow B) \rightarrow (A \rightarrow C))$, este axioma captura a transitividade das implicações, estabelecendo que se a implicação $A \rightarrow B$ e $B \rightarrow C$ são verdadeiras, então $A \rightarrow C$ também é verdadeira.

**Axioma 3**: $(\lnot B \rightarrow \lnot A) \rightarrow ((\lnot B \rightarrow A) \rightarrow B)$, este axioma garante que se de $\lnot B$ Podemos inferir tanto $\lnot A$ quanto $A$, então $B$ deve ser verdadeiro. Isso porque $B$ e $\lnot B$ não podem ser verdadeiros simultaneamente.

Além dos axiomas, usaremos apenas uma regra de inferência, o _Modus Ponens_. O _Modus Ponens_ está intimamente relacionado à proposição $(P \wedge (P \rightarrow Q)) \rightarrow Q$. Tanto a proposição quando a regra de inferência, de certa forma, dizem: "se $P$ e $P\rightarrow Q$ são verdadeiros, então $Q$ é verdadeiro". Esta proposição é um exemplo de uma tautologia, porque é verdadeira para cada configuração de $P$ e $Q$. A diferença é que esta tautologia é uma única proposição, enquanto o _Modus Ponens_ é uma regra de inferência que nos permite deduzir novas proposições a partir proposições já provadas.

Nos resta apenas destacar a última linha de uma prova. No sistema $\mathfrak{L}$A última fórmula será chamada de teorema. Representaremos como $\vdash A$ se $A$ for um teorema. Escrevemos $B_1, B_2, ..., B_n \vdash_L A$ só, e somente só, $A$Puder ser provado em $\mathfrak{L}$A partir das fórmulas dadas $B_1, B_2, ..., B_n$. Onde:

- $A$: Fórmula que é um teorema;

- $ g_1, ..., G_n$: Fórmulas que servem como premissas;

- $\vdash_L$: Símbolo para indicar _demonstrável em $\mathfrak{L}$_;

- escrevemos $\mathfrak{L} A$ Para indicar que $A$ é demonstrável no sistema $\mathfrak{L}$.

Talvez tudo isso fique mais claro se fizermos algumas provas.

**Prova 1**: nosso teorema é $A \rightarrow A$

1. $A \rightarrow ((A \rightarrow A) \rightarrow A)$ (Axioma 1 com $A := A$ e $B := (A \rightarrow A)$)

Aqui usamos o primeiro axioma de $\mathfrak{L}$, que tem a forma $(A \rightarrow (B \rightarrow A))$. Para tanto usamos $A := A$ e $B := (A \rightarrow A)$ para fazer a correspondência com o axioma, obtendo a fórmula na linha. Observe que usamos o símbolo $:=$, um símbolo que não faz parte do nosso alfabeto e aqui está sendo usado com o sentido _substituído por_. Até na matemática usamos licenças poéticas.

1. $(A \rightarrow ((A \rightarrow A) \rightarrow A)) \rightarrow ((A \rightarrow (A \rightarrow A)) \rightarrow (A \rightarrow A))$ (Axioma 2 com $A := A$, $B := (A \rightarrow A)$ e $ c := A$)

   A segunda linha usa o segundo axioma de $\mathfrak{L}$, que é $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \rightarrow B) \rightarrow (A \rightarrow C))$. O autor substituiu $A := A$, $B := (A \rightarrow A)$ e $ c := A$ Para obter a fórmula na linha.

2. $((A \rightarrow (A \rightarrow A)) \rightarrow (A \rightarrow A))$ (_Modus Ponens_ aplicado às linhas 1 e 2)

   Finalmente aplicamos a regra de _Modus Ponens_, que diz que se temos $A$ e também temos $A \rightarrow B$, então podemos deduzir $B$. As linhas 1 e 2 correspondem a $A$ e $A \rightarrow B$, respectivamente, e ao aplicar _Modus Ponens_, obtemos $B$, que é a fórmula na linha 3.

3. $(A \rightarrow (A \rightarrow A))$ (Axioma 1 com $A := A$ e $B := A$)

   De forma similar à primeira linha, a quarta linha usa o primeiro axioma com $A := A$ e $B := A$.

4. $(A \rightarrow A)$(_Modus Ponens_ aplicado às linhas 3 e 4)

   Finalmente, aplicamos o _Modus Ponens_ às linhas 3 e 4 para obter a fórmula na última linha, que é o teorema que tentamos provar.

   Então, o primeiro teorema está correto e podemos escrever $\vdash \mathfrak{L} A$.

**Prova 2**: vamos tentar provar $\vdash (\lnot B \rightarrow B) \rightarrow B$
**Prova 2**: vamos provar $\vdash (\lnot B \rightarrow B) \rightarrow B$

Para esta prova, utilizaremos o Teorema 1 ($\vdash A \rightarrow A$) como um lema e o Axioma 3.
O Axioma 3 é: $(\lnot X \rightarrow \lnot Y) \rightarrow ((\lnot X \rightarrow Y) \rightarrow X)$.

1. $\lnot B \rightarrow \lnot B$
   (Lema, Teorema 1 com $A$ substituído por $\lnot B$)

2. $(\lnot B \rightarrow \lnot B) \rightarrow ((\lnot B \rightarrow B) \rightarrow B)$
   (Instância do Axioma 3, onde $X$ do axioma é $B$ da nossa meta, e $Y$ do axioma é $B$ da nossa meta. Substituindo $X$ por $B$ e $Y$ por $B$ no Axioma 3: $(\lnot B \rightarrow \lnot B) \rightarrow ((\lnot B \rightarrow B) \rightarrow B)$)

3. $((\lnot B \rightarrow B) \rightarrow B)$
   (Modus Ponens aplicado às linhas 1 e 2. A linha 1 é o antecedente da implicação na linha 2.)

4. $(\lnot B \rightarrow B) \rightarrow B$ (Aplicação do Axioma 1 com $A := \lnot B$ e $B := B$)

   Finalmente, aplicamos o primeiro axioma, que é $A \rightarrow (B \rightarrow A)$, substituindo $A$ Por $\lnot B$ e $B$ Por $B$ Para obter o teorema que estamos tentando provar.

**Prova 3**: vamos tentar novamente, desta vez com $\vdash ((A \land B) \rightarrow C)$

1. $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \land B) \rightarrow C)$ (Suposto axioma com $A := A$, $B := B$ e $ c := C$)

   Aqui estamos assumindo que a fórmula $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \land B) \rightarrow C)$ é um axioma. No entanto, esta fórmula **não** é um axioma do sistema $\mathfrak{L}$. Portanto, esta tentativa de provar o teorema é inválida desde o início.

2. $A \rightarrow (B \rightarrow C)$ (Hipótese)

   Aqui estamos introduzindo uma hipótese, que é permissível. No entanto, uma hipótese deve ser descartada antes do final da prova e, nesta tentativa de prova, não é.

3. $(A \land B) \rightarrow C$ (_Modus Ponens_ aplicado às linhas 1 e 2)

   Finalmente, tentamos aplicar a regra de inferência _Modus Ponens_ às linhas 1 e 2 para obter $(A \land B) \rightarrow C$. No entanto, como a linha 1 é inválida, esta aplicação de _Modus Ponens_ também é inválida.

Portanto, esta tentativa de provar o teorema $(A \land B) \rightarrow C$ **falha** porque faz suposições inválidas e usa regras de inferência de forma inválida.

Esta última tentativa de prova é interessante. Para o teorema $(A \land B) \rightarrow C$, não é possível provar diretamente no sistema $\mathfrak{L}$ sem a presença de axiomas adicionais ou a introdução de hipóteses adicionais. Que não fazem parte do sistema $\mathfrak{L}$.

O sistema $\mathfrak{L}$ é baseado em axiomas específicos (que utilizam os conectivos $\rightarrow$ e $\lnot$) e em uma única regra de inferência (_Modus Ponens_), como vimos. O teorema $((A \land B) \rightarrow C)$ não pode ser derivado diretamente apenas a partir dos axiomas do sistema $\mathfrak{L}$ porque o conectivo de conjunção ($\land$) não é primitivo no sistema $\mathfrak{L}$ e não pode ser definido ou introduzido usando apenas os axiomas fornecidos e o Modus Ponens sem regras adicionais ou definições para $\land$. Os axiomas de $\mathfrak{L}$ focam na implicação e na negação.

Se tivéssemos acesso a axiomas ou regras de inferência adicionais que lidam com a conjunção, ou se você tem permissão para introduzir hipóteses adicionais (por exemplo, você pode introduzir $A \land B \rightarrow C$ como uma hipótese), então a prova pode ser possível. Em alguns sistemas de lógica, a conjunção pode ser definida em termos de negação e disjunção, e neste caso, o teorema pode ser provável.

Com as ferramentas que vimos até agora, podemos tentar provar o teorema $((A \land B) \rightarrow C)$ usando uma Tabela Verdade:

$$
\begin{array}{|c|c|c|c|c|}
\hline
A & B & C & A \land B & (A \land B) \rightarrow C \\
\hline
T & T & T & T & T \\
T & T & F & T & F \\
T & F & T & F & T \\
T & F & F & F & T \\
F & T & T & F & T \\
F & T & F & F & T \\
F & F & T & F & T \\
F & F & F & F & T \\
\hline
\end{array}
$$

Como podemos ver, a coluna final, que representa o teorema $(A \land B) \rightarrow C$, não é sempre verdadeira. Isso significa que a proposição $(A \land B) \rightarrow C$ não é uma tautologia, existe uma situação, quando $A$ e $B$ são verdadeiros, mas $ c$ é falso, em que a proposição inteira é falsa. Basta isso para que o teorema seja falso.

A nossa terceira prova mostra os limites do sistema $\mathfrak{L}$, o que pode dar uma falsa impressão sobre o a capacidade deste sistema de prova. Vamos tentar melhorar isso.

#### Lema

Considere nossa primeira prova, provamos $A \rightarrow A$ e, a partir deste momento, $A \rightarrow A$ se tornou um Lema. Um lema é uma afirmação que é provada não como um fim em si mesma, mas como um passo útil para a prova de outros teoremas.

Em outras palavras, um lema é um resultado menor que serve de base para um resultado maior. Uma vez que um lema é provado, ele pode ser usado em provas subsequentes de teoremas mais complexos. Em geral, um lema é menos geral e menos notável do que um teorema.

Considere o seguinte Teorema: $\vdash_L (\lnot B \rightarrow B) \rightarrow B$, podemos prová-lo da seguinte forma:

1. $\lnot B \rightarrow \lnot B$ - Lembrando que $A := \lnot B$ do Teorema 1

2. $(\lnot B \rightarrow \lnot B) \rightarrow ((\lnot B \rightarrow B) \rightarrow B)$ - Decorrente do Axioma 3, onde $A := \lnot B$ e $B := B$

3. $((\lnot B \rightarrow B) \rightarrow B)$- Através do _Modus Ponens_
   Justificativa: Linhas 1 e 2

A adoção de lemas é, na verdade, um mecanismo útil para economizar tempo e esforço. Ao invés de replicar o Teorema 1 na primeira linha dessa prova, nós poderíamos, alternativamente, copiar as 5 linhas da prova original do Teorema 1, substituindo todos os casos de $A$ Por $\lnot B$. As justificativas seriam mantidas iguais às da prova original do Teorema 1. A prova resultante, então, consistiria exclusivamente de axiomas e aplicações do _Modus Ponens_. No entanto, uma vez que a prova do Teorema 1 já foi formalmente documentada, parece redundante replicá-la aqui. E eis o motivo da existência e uso dos lemas.

#### Hipóteses

Hipóteses são suposições ou proposições feitas como base para o raciocínio, sem a suposição de sua veracidade. Elas são usadas como pontos de partida para investigações ou pesquisas científicas. Essencialmente uma hipótese é uma teoria ou ideia que você pode testar de alguma forma. Isso significa que, através de experimentação e observação, uma hipótese pode ser provada verdadeira ou falsa.

Por exemplo, se você observar que uma planta está morrendo, pode formar a hipótese de que ela não está recebendo água suficiente. Para testar essa hipótese, você pode dar mais água à planta e observar se ela melhora. Se melhorar, isso suporta sua hipótese. Se não houver mudança, isso sugere que sua hipótese pode estar errada, e você pode então formular uma nova hipótese para testar.

Na lógica proposicional, uma hipótese é uma proposição (ou afirmação) que é assumida como verdadeira para o propósito de argumentação ou investigação. Obviamente, pode ser uma fórmula atômica, ou complexa, desde que seja uma Fórmula Bem Formada.

Em um sistema formal de provas, como o sistema $\mathfrak{L}$ uma hipótese é um ponto de partida para um processo de dedução. O objetivo é usar as regras do sistema para deduzir novas proposições a partir das hipóteses. Se uma proposição puder ser deduzida a partir das hipóteses usando as regras do sistema, dizemos que essa proposição é uma consequência lógica das hipóteses.
Se temos as hipóteses $P$ e $P\rightarrow Q$, podemos deduzir $Q$ usando o _Modus Ponens_. Nesse caso, $Q$ seria uma consequência lógica das hipóteses.

No contexto do sistema de provas $\mathfrak{L}$ e considerando apenas a lógica proposicional, **uma hipótese é uma proposição ou conjunto de proposições assumidas como verdadeiras, a partir das quais outras proposições podem ser logicamente deduzidas**.

**Exemplo 1**: considere o seguinte argumento:

$$
\begin{align*}
A \rightarrow (B \rightarrow C) \\
A \rightarrow B \\
\hline
A \rightarrow C
\end{align*}
$$

Aplicando o processo de dedução do Sistema $\mathfrak{L}$, teremos:

$$
\begin{align*}
& A \rightarrow (B \rightarrow C) &\text{Hipótese} \\
& A \rightarrow B &\text{Hipótese}\\
& (A \rightarrow (B \rightarrow C)) \rightarrow ((A \rightarrow B) \rightarrow (A \rightarrow C)) &\text{Axioma 2}\\
& (A \rightarrow B) \rightarrow (A \rightarrow C) & \text{Modus Ponens, linhas 1 e 3} \\
& A \rightarrow C & \text{Modus Ponens, linhas 2 e 4}\\
\end{align*}
$$

Neste exemplo, vamos o uso das Hipóteses. No processo de dedução, as hipóteses devem ser usadas na forma como são declaradas. O que as torna diferentes dos lemas.

Neste ponto, podemos voltar um pouco e destacar um constructor importante na programação imperativa: _se...então_ representando por $P\rightarrow Q$, uma implicação. Que pode ser lido como hipótese $P$ e conclusão $Q$.

## Lógica Predicativa

> A lógica é a técnica que usamos para adicionar convicção à verdade. Jean de la Bruyere{: class="epigraph"}

A Lógica Predicativa, coração e espírito da Lógica de Primeira Ordem, nos leva um passo além da Lógica Proposicional. Em vez de se concentrar apenas em proposições completas que são verdadeiras ou falsas, a lógica predicativa nos permite expressar proposições sobre objetos e as relações entre eles. Ela nos permite falar de forma mais rica e sofisticada sobre o mundo.

Vamos lembrar que na Lógica Proposicional, cada proposição é um átomo indivisível. Por exemplo, 'A chuva cai' ou 'O sol brilha'. Cada uma dessas proposições é verdadeira ou falsa como uma unidade. Na lógica predicativa, no entanto, podemos olhar para dentro dessas proposições. Podemos falar sobre o sujeito - a chuva, o sol - e o predicado - cai, brilha. Podemos quantificar sobre eles: para todos os dias, existe um momento em que o sol brilha.

Enquanto a Lógica Proposicional pode ser vista como a aritmética do verdadeiro e do falso, a lógica predicativa é a álgebra do raciocínio. Ela nos permite manipular proposições de forma muito mais rica e expressiva. Com ela, podemos começar a codificar partes substanciais da matemática e da ciência, levando-nos mais perto de nossa busca para decifrar o cosmos, um símbolo de lógica de cada vez.

### Introdução aos Predicados

Um predicado é como uma luneta que nos permite observar as propriedades de uma entidade. Um conjunto de lentes através do qual podemos ver se uma entidade particular possui ou não uma característica específica. A palavra predicado foi importada do campo da linguística e tem o mesmo significado: qualidade; característica. Por exemplo, ao observar o universo das letras através do telescópio do predicado _ser uma vogal_, percebemos que algumas entidades deste conjunto, como $A$ e $I $, possuem essa propriedade, enquanto outras, como $ g$ e $H$, não.

Um predicado não é uma afirmação absoluta de verdade ou falsidade. Divergindo das proposições, os predicados não são declarações completas. Pense neles como aquelas sentenças com espaços em branco, aguardando para serem preenchidos, que só têm sentido completo quando preenchidas:

1. O \_\_\_\_\_\_\_ está saboroso;

2. O \_\_\_\_\_\_\_ é vermelho;

3. \_\_\_\_\_\_\_ é alto.

Preencha as lacunas, como quiser desde que faça sentido, e perceba que, em cada caso, ao preencher estamos atribuindo uma qualidade a um objeto. Esses são exemplos de predicados do nosso cotidiano, que sinteticamente o conceito que queremos abordar. Na lógica, os predicados são artefatos que possibilitam examinar o mundo ao nosso redor de forma organizada e exata.

Um predicado pode ser entendido como uma função que recebe um objeto (ou um conjunto de objetos) e retorna um valor de verdade, $\{\text{verdadeiro ou falso}\}$. Esta função descreve uma propriedade que o objeto pode possuir. Isto é, se $P$ é uma função $P: U \rightarrow \\{\text{Verdadeiro, Falso}\\}$ Para um determinado conjunto $ u$ qualquer. Esse conjunto $ u$ é chamado de _universo ou domínio do discurso_, e dizemos que $P$ é um predicado sobre $ u$.

### Universo do Discurso

O universo do discurso, $U$, também chamado de **universo**, ou domínio, é o conjunto de objetos de interesse em um determinado cenário lógico para uma análise específica. O universo do discurso é importante porque as proposições na Lógica de Predicados serão declarações sobre objetos de um universo.

O universo, $U$, é o domínio das variáveis das nossas Fórmulas Bem Formadas. O universo do discurso pode ser o conjunto dos números reais, $\mathbb{R}$ o conjunto dos inteiros, $\mathbb{z}$, o conjunto de todos os alunos em uma sala de aula que usam camisa amarela, ou qualquer outro conjunto que definamos. Na prática, o universo costuma ser deixado implícito e deveria ser óbvio a partir do contexto. Se não for o caso, precisa ser explicitado.

Se estamos interessados em proposições sobre números naturais, $\mathbb{N}$, o universo do discurso é o conjunto $\mathbb{N} = \{0, 1, 2, 3,...\}$, um conjunto infinito. Já se estamos interessados em proposições sobre alunos de uma sala de aula, o universo do discurso poderia ser o conjunto $ u = \{\text{Paulo}, \text{Ana}, ...\}$, um conjunto finito.

Para que este conceito fique mais claro, suponha que temos um conjunto de números $U = \\{1, 2, 3, 4, 5\\}$ e um predicado $P(u)$, que dizemos unário por ter um, e somente um, argumento, que afirma _u é par_. Ao aplicarmos este predicado a cada elemento do universo $U$, obtemos um conjunto de valores verdade:

$$
\begin{align}
&P(1) = \text{falso};\\
&P(2) = \text{verdadeiro};\\
&P(3) = \text{falso};\\
&P(4) = \text{verdadeiro};\\
&P(5) = \text{falso}.
\end{align}
$$

Vemos que o predicado $P(u)$ dado por _u é par_ é uma propriedade que alguns números do conjunto $ u$ Possuem, e outros não. Vale notar que na Lógica Predicativa, a função que define um predicado pode ter múltiplos argumentos. Por exemplo, podemos ter um predicado $Q(x, y)$ que afirma _x é maior que y_. Neste caso, o predicado $Q$ é uma função de dois argumentos que retorna um valor de verdade. Dizemos que $Q(x, y)$ é um predicado binário. Exemplos nos conduzem ao caminho do entendimento:

1. **Exemplo 1**:

   - Universo do discurso: $U = \text{conjunto de todas as pessoas}$.
   - Predicado:$P(x) = \\{ x : x \text{ é um matemático} \\}$;
   - Itens para os quais $P(x)$ é verdadeiro: Carl Gauss, Leonhard Euler, John Von Neumann.

2. **Exemplo 2**:

   - Universo do discurso: $U = \{x \in \mathbb{Z} : x \text{ é par}\}$
   - Predicado: $Q(x) = (x > 5)$;
   - Itens para os quais $Q(x)$ é verdadeiro: $6 $, $8 $, $10 ...$.

3. **Exemplo 3**:

   - Universo do discurso: $U = \{x \in \mathbb{R} : x > 0 \text{ e } x < 10\}$
   - Predicado: $R(x) = (x^2 - 4 = 0)$;
   - Itens para os quais $R(x)$ é verdadeiro: $2$, $-2$.

4. **Exemplo 4**:

   - Universo do discurso: $U = \\{x \in \mathbb{N} : x \text{ é um múltiplo de } 3\\}$
   - Predicado: $S(x) = (\text{mod}(x, 2) = 0)$;
   - Itens para os quais $S(x)$ é verdadeiro: $6$, $12$, $18 \ldots $.

5. **Exemplo 5**:

   - Universo do discurso: $U = \{(x, y) \in \mathbb{R}^2 : x \neq y\}$
   - Predicado: $P(x, y) = (x < y)$;
   - Itens para os quais $P(x, y)$ é verdadeiro: $(1, 2)$, $(3, 4)$, $(5, 6)$.

### Entendendo Predicados

A aridade do predicado, número de argumentos, é limitado pela análise lógica que estamos fazendo. Considere um predicado ternário, $R$, dado por _x está entre y e z_. Quando substituímos $x$, $y$ e $z$ Por números específicos podemos validar a verdade do predicado $R$. Vamos considerar alguns exemplos adicionais de predicados baseados na aritmética e defini-los com menos formalidade e mais legibilidade:

1. $ Primo(n)$: o número inteiro positivo $ n$ é um número primo.
2. $ PotênciaDe (n, k)$: o número inteiro $ n$ é uma potência exata de $k : n = ki$ Para algum $i \in \mathbb{Z} ≥ 0$.
3. $ somaDeDoisPrimos(n)$: o número inteiro positivo $ n$ é igual à soma de dois números primos.

Em 1, 2 e 3 os predicados estão definidos com mnemônicos aumentando a legibilidade e melhorando nossa capacidade de manter o universo implícito. O uso de predicados, e da Lógica Proposicional, permite a escrita de sentenças menos ambíguas para a definição de conceitos lógicos em formato matemático. Por exemplo: se $x$ é um ancestral de $y$ e $y$ é um ancestral de $z$ então $x$ é um ancestral de $z$; que, se consideramos o predicado $AncestralDe $ Pode ser escrito como $AncestralDe (x,y) \wedge ancestralDe (y,z) \rightarrow ancestralDe (x,z)$. Ainda assim, falta alguma coisa. Algo que permita aplicar os predicados a um conjunto de elementos dentro do universo do discurso. É aqui que entram os quantificadores.

### Quantificadores

Embora a Lógica Proposicional seja um bom ponto de partida, a maioria das afirmações interessantes em matemática contêm variáveis definidas em domínios maiores do que apenas $\\{\text{Verdadeiro}, \text{Falso}\\}$. Por exemplo, a afirmação _$x \text{é uma potência de } 2$_ não é uma proposição. Não temos como definir a verdade dessa afirmação até conhecermos o valor de $x$. Se $P(x)$ é definido como a afirmação _$x \text{é uma potência de } 2$_, então $P(8)$ é verdadeiro e $P(7)$ é falso.

Para termos uma linguagem lógica que seja suficientemente flexível para representar os problemas que encontramos no Universo real, o Universo em que vivemos, precisaremos ser capazes de dizer quando o predicado $P$ ou $Q$ é verdadeiro para valores diferentes em seus argumentos. Para tanto, vincularemos as variáveis aos predicados usando operadores para indicar quantidade, chamados de quantificadores.

Os quantificadores indicam se a sentença que estamos criando se aplica a todos os valores possíveis do argumento, _quantificação universal_, ou se esta sentença se aplica a um valor específico, _quantificação existencial_. Usaremos esses quantificadores para fazer declarações sobre **todos os elementos** de um universo de discurso específico, ou para afirmar que existe **pelo menos um elemento** do universo do discurso que satisfaz uma determinada qualidade.

Vamos remover o véu da dúvida usando como recurso metafórico uma experiência humana, social, comum e popular: imaginemos estar em uma festa e o anfitrião lhe pede para verificar se todos os convidados têm algo para beber. Você, prestativo e simpático, começa a percorrer a sala, verificando cada pessoa. Se você encontrar pelo menos uma pessoa sem bebida, você pode imediatamente dizer _nem todos têm bebidas_. mas, se você verificar cada convidado e todos eles tiverem algo para beber, você pode dizer com confiança _todos têm bebidas_. Este é o conceito do quantificador universal, matematicamente representado por $\forall$, que lemos como _para todo_.

A festa continua e o anfitrião quer saber se alguém na festa está bebendo champanhe. Desta vez, assim que você encontrar uma pessoa com champanhe, você pode responder imediatamente _sim, alguém está bebendo champanhe_. Você não precisa verificar todo mundo para ter a resposta correta. Este é o conceito do quantificador existencial, denotado por $\exists $, que lemos _existe algum_.

Os quantificadores nos permitem fazer declarações gerais, ou específicas, sobre os membros de um universo de discurso, de uma forma que seria difícil, ou impossível, sem estes operadores especiais.

#### Quantificador Universal

O quantificador universal $\forall$, lê-se _para todo_, indica que uma afirmação deve ser verdadeira para todos os valores de uma variável dentro de um universo de discurso definido para a criação de uma sentença contendo um predicado qualquer. Por exemplo, a proposição clássica _todos os humanos são mortais_ pode ser escrita como $\forall x Humano(x) \rightarrow Mortal(x)$. Ou recorrendo a um exemplo com mais de rigor matemático, teríamos o predicado se _$x$ é positivo então $x + 1 $ é positivo_, que pode ser escrito $\forall x (x > 0 \rightarrow x + 1 > 0)$. Neste último exemplo temos Quantificadores, Lógica Predicativa, Lógica Proposicional e Teoria dos Conjuntos em uma sentença.

O quantificador universal pode ser representado usando apenas a Lógica Proposicional, com uma pequena trapaça. A afirmação $\forall x P(x)$ é, de certa forma, a operação $\wedge $, **AND** aplicada a todos os elementos do universo do discurso. Ou seja, o predicado:

$$\forall x \{x:\in \mathbb{N}\} : P(x)$$

Pode ser escrito como:

$$P(0) \land P(1) \land P(2) \land P(3) \land \ldots $$

Onde $P(0), P(1), P(2), P(3) \ldots $ representam a aplicação do predicado $P$A todos os elementos $x$ do conjunto $\mathbb{N}$. A trapaça fica por conta de que, em Lógica Proposicional, não podemos escrever expressões com um número infinito de termos. Portanto, a expansão em conjunções de um predicado $P$ em um Universo de Discurso, $ u$, não é uma Fórmula Bem Formada se a cardinalidade de $ u$ for infinita. De qualquer forma, podemos usar esta interpretação informal para entender o significado de $\forall x P(x)$.

A representação do Quantificador Universal como uma conjunção **não é uma Fórmula Bem Formada** a não ser que o Universo do Discurso seja não infinito. Neste caso, teremos uma conjunção que chamaremos de **Conjunção Universal**:

$$\forall x (P(x) \land Q(x))$$

Isso significa que para todo $x$ no domínio, as propriedades $P$, $Q$, e outras listadas são todas verdadeiras. É uma forma de expressar que todas as condições listadas são verdadeiras para cada elemento no domínio. Esta fórmula será usada para simplificar sentenças, ou para criar formas normais.

Vamos voltar um pouco. O quantificador universal $\forall x P(x)$Afirma que a proposição $P(x)$ é verdadeira para todo, e qualquer, valor possível de $x$ como elemento de um conjunto, $u$. Uma forma de interpretar isso é pensar em $x$ como uma variável que pode ter qualquer valor dentro do universo do discurso.

Para validar $\forall x P(x)$ escolhemos o pior caso possível para $x$, todos os valores que suspeitamos possa fazer $P(x)$ falso. Se conseguirmos provar que $P(x)$ é verdadeira nestes casos específicos, então $\forall x P(x)$ deve ser verdadeira. Novamente, vamos recorrer a exemplos na esperança de explicitar este conceito.

**Exemplo 1**: todos os números reais são maiores que 0. (Universo do discurso: $\{x \in \mathbb{R}\}$)

$$\forall x (x \in \mathbb{R} \rightarrow x > 0)$$

> Observe que este predicado, apesar de estar corretamente representado, é $Falso$.

**Exemplo 2**: todos os triângulos em um plano euclidiano têm a soma dos ângulos internos igual a 180 graus. (Universo do discurso: $x$ é um triângulo em um plano euclidiano)

$$\forall x (Triângulo(x) \rightarrow \Sigma_{i=1}^3 ÂnguloInterno_i(x) = 180^\circ)$$

**Exemplo 3**: todas as pessoas com mais de 18 anos podem tirar carteira de motorista." (Universo do discurso: $x$ é uma pessoa no Brasil)

$$\forall x (Pessoa(x) \land Idade (x) \geq 18 \rightarrow PodeTirarCarteira(x))$$

**Exemplo 4**: todo número par maior que 2 pode ser escrito como a soma de dois números primos. (Universo do discurso: $\{x \in \mathbb{Z}\}$

$$\forall x\,(Par(x) \land x > 2 \rightarrow \exists a\exists b\, (Primo(a) \land Primo(b) \land x = a + b))$$

**Exemplo 5**: para todo número natural, se ele é múltiplo de 4 e múltiplo de 6, então ele também é múltiplo de 12. (Universo do discurso: $\{x \in \mathbb{N}\}$)

$$\forall x\,((\exists a\in\Bbb N\,(x = 4a) \land \exists b\in\Bbb N\,(x = 6b)) \rightarrow \exists c\in\Bbb N\,(x = 12c))$$

O quantificador universal nos permite definir uma Fórmula Bem Formada representando todos os elementos de um conjunto, um universo do discurso, em relação a uma qualidade específica, um predicado. Esta é um artefato lógico interessante, mas não suficiente.

Usamos, preferencialmente, a implicação, $\to$, com o quantificador universal, $\forall$, para indicar que uma propriedade vale para todos os elementos de um domínio, Porque permite afirmar que _para todo $x$, se $P(x)$ for verdadeira, então $Q(x)$ também será verdadeira_. Isso permite que $P(x)$ seja falsa para alguns $x$, mas a implicação como um todo permanece verdadeira. Ou, em outras palavras, quando usamos uma implicação, como $P(x) \rightarrow Q(x)$, estamos dizendo que _se $P(x)$ for verdadeira, então $Q(x)$ também será verdadeira_. A implicação é uma forma lógica que permite conectar duas proposições, onde a veracidade de $Q(x)$ depende da veracidade de $P(x)$.

> Importante: A implicação $P(x) \rightarrow Q(x)$ é considerada verdadeira em qualquer dos seguintes casos:
>
> $P(x)$ é verdadeira e $Q(x)$ é verdadeira.
> $P(x)$ é falsa, independentemente de $Q(x)$.
> O ponto-chave é o segundo caso: se $P(x)$ for falsa, a implicação $P(x) \rightarrow Q(x)$ ainda é verdadeira, não importa o valor de $Q(x)$.

Essa preferência não é arbitrária, mas baseada nas limitações que os outros conectivos apresentam quando combinados com o quantificador universal. Porém, uma análise de todos os operadores pode ser interessante para sedimentar os conceitos.

Comecemos com a conjunção. Quando usamos $∀x(P(x) ∧ Q(x))$, estamos afirmando que para todo $x$, tanto $P(x)$ quanto $Q(x)$ são verdadeiros. Isso é extremamente restritivo e raramente reflete situações do mundo real. Por exemplo, se disséssemos _Todos os animais são mamíferos e podem voar_, estaríamos fazendo uma afirmação falsa, pois nem todos os animais são mamíferos e nem todos podem voar. Outro exemplo seria _Todos os números são pares e primos_, o que é claramente falso, pois nenhum número (exceto 2) satisfaz ambas as condições simultaneamente.

A disjunção, por outro lado, é muito fraca quando combinada com o quantificador universal. $∀x(P(x) ∨ Q(x))$ afirma que para todo $x$, ou $P(x)$ ou $Q(x)$ (ou ambos) são verdadeiros. Isso geralmente não captura relações condicionais úteis. Por exemplo, _Todo número é par ou ímpar_ é uma afirmação verdadeira, mas não nos diz muito sobre a relação entre paridade e números. Da mesma forma, _Toda pessoa é alta ou baixa_ é uma afirmação de tal amplitude que se torna quase sem sentido, pois não fornece informações úteis sobre a altura das pessoas.

A equivalência ($\leftrightarrow$) com o quantificador universal também apresenta problemas. $∀x(P(x) \leftrightarrow Q(x))$ afirma que para todo $x$, $P(x)$ é verdadeiro se e somente se $Q(x)$ for verdadeiro. Isso é uma condição muito forte e raramente é satisfeita em situações reais. Por exemplo, _Um número é par se e somente se é divisível por 4_ é falso, pois há números pares que não são divisíveis por $4$ (como $2$ e $6$). Outro exemplo seria _Uma pessoa é feliz se e somente se é rica_, o que claramente não reflete a realidade complexa da felicidade e riqueza.

Por outro lado, a implicação ($\to$) oferece várias vantagens quando usada com o quantificador universal. $∀x(P(x) \to Q(x))$ nos permite expressar relações condicionais de forma mais flexível e precisa. Por exemplo, _Para todo número, se é par, então não é primo (exceto 2)_ é uma afirmação verdadeira e informativa. Outro exemplo seria _Para toda pessoa, se é médico, então tem formação universitária_. Esta formulação permite exceções (pode haver pessoas com formação universitária que não são médicos) e captura uma regra geral de forma precisa.

A implicação também tem a vantagem de ser verdadeira quando o antecedente ($P(x)$) é falso, o que é útil para expressar regras gerais. Por exemplo, em _Para todo x, se x é um quadrado perfeito, então x é positivo_, a implicação é verdadeira mesmo para números negativos (que não são quadrados perfeitos), mantendo a regra geral válida.

Espero que tenha ficado claro. A implicação, quando combinada com o quantificador universal, oferece um equilíbrio entre flexibilidade e precisão que os outros conectivos lógicos não conseguem alcançar. Ela permite expressar relações condicionais, acomoda exceções e captura regras gerais de forma mais eficaz, tornando-a a escolha preferida em muitas situações da lógica formal e da matemática.

#### Quantificador Existencial

O quantificador existencial, $\exists $ nos permite fazer afirmações sobre a existência de objetos com certas propriedades, sem precisarmos especificar exatamente quais objetos são esses. Vamos tentar remover os véus da dúvida com um exemplo simples.

Consideremos a sentença: _existem humanos mortais_. Com um pouco mais de detalhe e matemática, podemos escrever isso como: existe pelo menos um $x$ tal que $x$ é humano e mortal. Para escrever a mesma sentença com precisão matemática teremos:

$$\exists x \text{Humano}(x) \land \text{Mortal}(x)$$

Lendo por partes: _existe um $x$, tal que $x$ é humano AND $x$ é mortal_. Em outras palavras, existe pelo menos um humano que é mortal.

Note duas coisas importantes:

1. Nós não precisamos dizer exatamente quem é esse humano mortal. Só afirmamos que existe um. O operador $\exists $ captura essa ideia.

2. Usamos **AND** ($\land $), não implicação ($\rightarrow $). Se usássemos $\rightarrow $, a afirmação ficaria muito mais fraca. Veja:

$$\exists x \text{Humano}(x) \rightarrow \text{Mortal}(x)$$

Que pode ser lido como: _existe um $x$ tal que, SE $x$ é humano, ENTÃO $x$ é mortal_. Essa afirmação é verdadeira em qualquer universo que contenha um unicórnio de bolinhas roxas imortal. Porque o unicórnio não é humano, então $\text{Humano}(\text{unicórnio})$ é falsa, e a implicação $\text{Humano}(x) \rightarrow \text{Mortal}(x)$ é verdadeira. Não entendeu? Volte dois parágrafos e leia novamente. Repita!

Portanto, é importante usar o operador $\land $, e não $\rightarrow $ quando trabalhamos com quantificadores existenciais. O $\land $ garante que a propriedade se aplica ao objeto existente definido pelo $\exists $. Contudo, podemos melhorar um pouco isso:

A conjunção, $\land$, é frequentemente empregada com o quantificador existencial, $\exists$, para expressar a presença de ao menos um elemento em determinado conjunto que possui múltiplas características simultaneamente. Isso nos possibilita declarar que _há no mínimo um $x$ para o qual tanto $P(x)$ quanto $Q(x)$ são válidas_. Tal afirmação confirma a existência de pelo menos um elemento que atende a ambos os critérios. Dito de outra forma, ao utilizarmos uma conjunção, como em $P(x) \land Q(x)$, estamos afirmando que _existe ao menos um $x$ onde $P(x)$ é verdadeiro e, ao mesmo tempo, $Q(x)$ também o é_. A conjunção funciona como um operador lógico que une duas proposições, onde a validade da asserção existencial depende da ocorrência simultânea de $P(x)$ e $Q(x)$ para, no mínimo, um $x$.

> No contexto do quantificador existencial $\exists x$, a conjunção $P(x) \land Q(x)$ é tida como verdadeira se, e apenas se:
>
> Houver ao menos um $x$ para o qual tanto $P(x)$ quanto $Q(x)$ são verdadeiras.
> Caso não exista tal $x$, a afirmação existencial é considerada falsa.
> Observe que basta a existência de um único elemento satisfazendo ambas as condições para validar a afirmação existencial.

Esta predileção não é fortuita, mas fundamentada na aptidão da conjunção em expressar com exatidão a existência de elementos dotados de múltiplos atributos concomitantes. No entanto, uma avaliação dos demais operadores pode ser proveitosa para consolidar esses conceitos.

Iniciemos com a implicação. Ao empregarmos $\exists x(P(x) \to Q(x))$, declaramos a existência de ao menos um $x$ tal que, se $P(x)$ for verdadeiro, então $Q(x)$ também o será. Esta formulação é menos elucidativa que a conjunção no âmbito existencial, pois seria verdadeira mesmo se $P(x)$ fosse falso para todo $x$. Ilustrando: _Há um número que, se for ímpar, é múltiplo de 2_ é verdadeiro (pois é válido para números pares), mas não esclarece se realmente existe um número ímpar que é múltiplo de 2.

A disjunção aliada ao quantificador existencial, $\exists x(P(x) \lor Q(x))$, assevera a existência de pelo menos um $x$ que satisfaz $P(x)$ ou $Q(x)$ (ou ambos). Embora útil em certos contextos, geralmente é menos robusta que a conjunção para afirmar a existência de elementos com múltiplas propriedades. Por exemplo: _Existe um número que é negativo ou racional_ é verdadeiro, mas não nos informa se há um número que é ambos.

A equivalência ($\leftrightarrow$) com o quantificador existencial também pode ser problemática. $\exists x(P(x) \leftrightarrow Q(x))$ afirma a existência de ao menos um $x$ para o qual $P(x)$ é verdadeiro se e somente se $Q(x)$ for verdadeiro. Isso pode ser útil em alguns casos, mas frequentemente é mais restritivo do que o necessário. Por exemplo: _Existe um número que é positivo se e somente se é inteiro_ é verdadeiro (o número 1 satisfaz isso), mas não captura a existência de números que são apenas positivos ou apenas inteiros.

Em contrapartida, a conjunção ($\land$) apresenta diversas vantagens quando utilizada com o quantificador existencial. $\exists x(P(x) \land Q(x))$ nos permite afirmar a existência de elementos que possuem múltiplas propriedades simultaneamente. Por exemplo: _Existe um número que é positivo e par_ é uma afirmação verdadeira e informativa (o número 2 satisfaz ambas as condições). Outro exemplo seria _Existe uma substância que é líquida e condutora de eletricidade_. Esta formulação afirma claramente a existência de substâncias com ambas as características.

A conjunção também tem a vantagem de ser falsa quando não há elementos que satisfaçam ambas as condições, o que é útil para expressar a inexistência de certos tipos de elementos. Por exemplo: _Existe um número que é natural e negativo simultaneamente_ é falso, indicando corretamente que não há tais números.

Em suma, a conjunção, quando associada ao quantificador existencial, proporciona um meio preciso e informativo de expressar a existência de elementos com múltiplos atributos. Ela permite afirmar a presença de elementos que atendem a condições simultâneas, tornando-se a opção preferencial em diversas situações da lógica formal e da matemática quando se trata de asserções existenciais.

Assim como o quantificador universal, $\forall $, o quantificador existencial, $\exists $ , também pode ser restrito a um universo específico, usando a notação de pertencimento:

$$\exists x \in \mathbb{Z} : x = x^2$$

Esta sentença afirma a existência de pelo menos um inteiro $x$ tal que $x$ é igual ao seu quadrado. Novamente, não precisamos dizer qual é esse inteiro, apenas que ele existe dentro do conjunto dos inteiros. Existe?

De forma geral, o quantificador existencial serve para fazer afirmações elegantes sobre a existência de objetos com certas qualidades, sem necessariamente conhecermos ou elencarmos todos esses objetos. Isso agrega mais qualidade a representação do mundo real que podemos fazer com a Lógica de Primeira Ordem.

Talvez, alguns exemplos possam ajudar no seu entendimento:

**Exemplo 1**: existe um mamífero que não respira ar.

$$\exists x (mamífero(x) \land \neg RespiraAr(x))$$

**Exemplo 2**: existe uma equação do segundo grau com uma raiz real.

$$\exists x (\text{Eq2Grau}(x) \land |\text{RaízesReais}(x)| \leq 1)$$

**Exemplo 3**: existe um número primo que é par.

$$\exists x (Primo(x) \land Par(x))$$

**Exemplo 4**: existe um quadrado perfeito que pode ser escrito como o quadrado de um número racional.

$$\exists x (QuadPerfeito(x) \land \exists a \in \mathbb{Q} \ (x = a^2))$$

**Exemplo 5**: existe um polígono convexo em que a soma dos ângulos internos não é igual A$(n-2)\cdot180^{\circ}$.

$$\exists x (\text{PolígonoConvexo}(x) \land \sum_{i=1}^{n} \text{ÂnguloInterno}_i(x) \neq (n-2)\cdot 180^{\circ})$$

> Novamente, observe que este predicado é $falso$. Todos os polígonos convexos têm a soma dos ângulos internos igual a $(n−2)cdot 180$, onde $𝑛$ é o número de lados do polígono.

### Equivalências Interessantes

Estudando o quantificador universal encontramos duas equivalências interessantes:

$$\lnot \forall x P(x) \leftrightarrow \exists x \lnot P(x)$$

$$\lnot \exists x P(x) \leftrightarrow \forall x \lnot P(x)$$

Essas equivalências são essencialmente as versões quantificadas das **Leis de De Morgan**. A primeira diz que nem todos os humanos são mortais, isso é equivalente a encontrar algum humano que não é mortal. A segunda diz que para mostrar que nenhum humano é mortal, temos que mostrar que todos os humanos não são mortais.

Podemos representar uma declaração $\exists x P(x)$ como uma expressão _OU_. Por exemplo, $\exists x \in \mathbb{N} : P(x)$ Poderia ser reescrito como:

$$P(0) \lor P(1) \lor P(2) \lor P(3) \lor \ldots $$

Lembrado do problema que encontramos quando fizemos isso com o quantificador $\forall $: não podemos representar fórmulas sem fim em Lógica de Primeira Ordem. mas, novamente esta notação, ainda que inválida, nos permite entender melhor o quantificador existencial. Caso o Universo do Discurso seja não infinito, limitado e contável, teremos a **Disjunção Existencial** uma expressão na lógica de primeiro grau que afirma que existe pelo menos um elemento em um domínio que satisfaz uma ou mais propriedades. A forma geral de uma disjunção existencial é:

$$\exists x (P(x) \lor Q(x))$$

Isso significa que existe pelo menos um $x$ no domínio que satisfaz a propriedade $P$, ou a propriedade $Q$, ou ambas, ou outras propriedades listadas. É uma forma de expressar que pelo menos uma das condições listadas é verdadeira para algum elemento no domínio.

A expansão de $\exists $ usando $\lor $ destaca que a proposição $P(x)$ é verdadeira se pelo menos um valor de $x$ dentro do universo do discurso atender ao predicado $P$. O que a expansão de exemplo está dizendo é que existe pelo menos um número natural $x$ tal que $P(x)$ é verdadeiro. Não precisamos saber exatamente qual é esse $x$. Apenas que existe um elemento dentro de $\mathbb{N}$ que atende o predicado.

O quantificador existencial não especifica o objeto dentro do universo determinado. Esse operador permite fazer afirmações elegantes sobre a existência de objetos com certas características, certas qualidades, ou ainda, certos predicados, sem necessariamente conhecermos exatamente quais são esses objetos.

### Dos Predicados à Linguagem Natural

Ao ler uma Fórmula Bem Formada contendo quantificadores, a ordem da leitura é geralmente da esquerda para a direita. A interpretação precisa, no entanto, depende crucialmente da ordem e do tipo dos quantificadores.

Por exemplo, $\forall x$ pode ser lido como "_para todo objeto $x$ no universo do discurso (onde este objeto está implícito), o seguinte se mantém_". Já o quantificador $\exists x$ pode ser lido como "_existe um objeto $x$ no universo que satisfaz o seguinte_" ou "_para algum objeto $x$ no universo, o seguinte se mantém_".

Converter uma Fórmula Bem Formada em uma sentença fluida em linguagem natural nem sempre é direto, mas é um passo valioso para a compreensão. Vamos considerar $U$ como o universo do discurso (o conjunto de todos os aviões já fabricados) e $F(x,y)$ como o predicado que denota "$x$ voa mais rápido que $y$". Analisemos algumas combinações de quantificadores:

1. **$\forall x \forall y F(x,y)$**

   - **Leitura literal**: Para todo avião $x$, e para todo avião $y$, $x$ voa mais rápido que $y$.

   - **Significado**: Esta afirmação é muito forte. Ela diz que cada avião no universo é mais rápido que todos os aviões no universo (incluindo ele mesmo, a menos que $F(x,x)$ seja definido como falso ou que se adicione $x \neq y$). Se o universo tiver mais de um avião, esta afirmação provavelmente será falsa, pois implicaria, por exemplo, que $A$ é mais rápido que $B$ e $B$ é mais rápido que $A$ simultaneamente.

2. **$\exists x \forall y F(x,y)$**

   - **Leitura literal**: Existe um avião $x$ tal que, para todo avião $y$, $x$ voa mais rápido que $y$.

   - **Significado**: Esta afirmação diz que existe pelo menos um avião que é mais rápido que todos os outros (e, novamente, dependendo da definição de $F(x,x)$, mais rápido que ele mesmo). Em outras palavras, existe um "avião mais rápido absoluto".

3. **$\forall x \exists y F(x,y)$**

   - **Leitura literal**: Para todo avião $x$, existe um avião $y$ tal que $x$ voa mais rápido que $y$.

   - **Significado**: Esta afirmação diz que para qualquer avião que escolhermos, podemos encontrar algum avião $y$ que é mais lento que $x$. Se $y$ pode ser igual a $x$, a afirmação é trivialmente verdadeira se $F(x,x)$ for verdadeiro para algum $x$. Se $y$ deve ser diferente de $x$, isso significaria que não existe um "avião mais lento absoluto" (a menos que o universo seja finito e ordenado de forma cíclica, ou que $F(x,y)$ permita que $x$ seja mais rápido que "nada" se $y$ for o mais lento). Uma interpretação comum é que, para cada avião, há outro que ele supera em velocidade.

4. **$\exists x \exists y F(x,y)$**

   - **Leitura literal**: Existe um avião $x$ e existe um avião $y$ tal que $x$ voa mais rápido que $y$.

   - **Significado**: Esta é a afirmação mais fraca entre as quatro. Ela simplesmente diz que a relação "voa mais rápido que" não é vazia; ou seja, há pelo menos um par de aviões $(x,y)$ onde $x$ é mais rápido que $y$.

É fundamental perceber que **estas quatro sentenças têm significados lógicos distintos e geralmente não expressam o mesmo contexto**. A ordem dos quantificadores, especialmente quando misturamos $\forall$ e $\exists$, altera drasticamente o significado da afirmação. Por exemplo, $\exists x \forall y F(x,y)$ (existe um avião mais rápido que todos) é uma afirmação muito mais forte e diferente de $\forall x \exists y F(x,y)$ (para cada avião, existe um mais lento).

Ao traduzir da lógica para a linguagem natural ou vice-versa, a precisão na interpretação da ordem e do tipo dos quantificadores é essencial. A prática leva a uma maior fluidez nesse processo de tradução e compreensão.

### Exercícios de Conversão de Linguagem Natural em Expressões Predicativas

**Sentença 1**: _Todo matemático que é professor tem alunos que são brilhantes e interessados._

$$
\forall x ((\text{Matemático}(x) \wedge \text{Professor}(x)) \rightarrow \exists y (\text{Aluno}(y) \wedge \text{Brilhante}(y) \wedge \text{Interessado}(y) \wedge \text{Ensina}(x, y)))
$$

$$
\forall x (\text{Matemático}(x) \rightarrow (\text{Professor}(x) \rightarrow \exists y (\text{Aluno}(y) \wedge \text{Brilhante}(y) \wedge \text{Interessado}(y) \wedge \text{Ensina}(x, y))))
$$

**Sentença 2**: _Alguns engenheiros não são nem ricos nem felizes._

$$\exists x (\text{Engenheiro}(x) \wedge \neg (\text{Rico}(x) \vee \text{Feliz}(x)))$$

$$\exists x (\text{Engenheiro}(x) \wedge \neg\text{Rico}(x) \wedge \neg\text{Feliz}(x))$$

**Sentença 3**: _Todos os planetas que têm água possuem vida ou têm potencial para vida._

$$
\forall x (\text{Planeta}(x) \wedge \text{TemÁgua}(x) \rightarrow (\text{TemVida}(x) \vee \text{TemPotencialParaVida}(x)))
$$

$$
\forall x (\text{Planeta}(x) \rightarrow (\text{TemÁgua}(x) \rightarrow (\text{TemVida}(x) \vee \text{TemPotencialParaVida}(x))))
$$

**Sentença 4**: _Nenhum cientista que é cético acredita em todos os mitos._

$$
\neg \exists x (Cientista(x) \wedge Cético(x) \wedge \forall y (Mito(y) \rightarrow Acredita(x,y)))
$$

$$
\forall x ((\text{Cientista}(x) \wedge \text{Cético}(x)) \rightarrow \exists y (\text{Mito}(y) \wedge \neg \text{Acredita}(x, y)))
$$

$$
\forall x (\text{Cientista}(x) \rightarrow (\text{Cético}(x) \rightarrow \exists y (\text{Mito}(y) \wedge \neg \text{Acredita}(x, y))))
$$

**Sentença 5**: _Alguns filósofos que escrevem sobre ética também leem ou estudam psicologia._

$$
\exists x (\text{Filósofo}(x) \wedge \text{EscreveSobreÉtica}(x) \wedge (\text{Lê}(x, \text{"Psicologia"}) \vee \text{Estuda}(x, \text{"Psicologia"})))
$$

$$
\exists x (\text{Filósofo}(x) \wedge \text{EscreveSobreÉtica}(x) \rightarrow (\text{Lê}(x, \text{"Psicologia"}) \vee \text{Estuda}(x, \text{"Psicologia"})))
$$

$$
\exists x (\text{Filósofo}(x) \land \text{EscreveSobreÉtica}(x) \land (\text{Lê}(x) \lor \text{"Psicologia"}(x)))
$$

**Sentença 6**: _Para todo escritor, existe pelo menos um livro que ele escreveu e que é tanto criticado quanto admirado._

$$
\forall x (\text{Escritor}(x) \rightarrow \exists y (\text{Livro}(y) \wedge \text{Escreveu}(x, y) \wedge \text{Criticado}(y) \wedge \text{Admirado}(y)))
$$

$$
\exists x (\text{Escritor}(x) \wedge \exists y (\text{Livro}(y) \wedge \text{Escreveu}(x, y) \wedge (\text{Criticado}(y) \wedge \text{Admirado}(y))))
$$

$$
\forall x \exists y (\text{Escritor}(x) \land \text{Escreveu}(x, y) \rightarrow (\text{criticado}(y) \land \text{Admirado}(y)))
$$

### Exercícios de Conversão de Expressões Predicativas em Linguagem Natural

**1. Fórmula Lógica**: $\forall x (\text{Humano}(x) \rightarrow (\text{Mortal}(x) \wedge \text{Racional}(x)))  
$

- Predicados:

  - $Humano(x)$: _$x$ é um humano_.
  - $Mortal(x)$: _$x$ é mortal_.
  - $Racional(x)$: _$x$ é racional_.

- **Sentença em Português**: Todo humano é mortal e racional.

**~2. Fórmula Lógica**:$\exists y (\text{Livro}(y) \wedge (\text{Interessante}(y) \vee \text{Complicado}(y)))
$

- Predicados:

  - $Livro(y)$: _y é um livro_.
  - $Interessante(y)$: _y é interessante_.
  - $Complicado(y)$: _y é complicado_.

- **Sentença em Português**: Existe pelo menos um livro que é interessante ou complicado.

**3. Fórmula Lógica**:$\forall x \forall y (\text{Amigos}(x, y) \rightarrow (\text{Confiável}(x) \wedge \text{Honra}(x)))$

- Predicados:

  - $Amigos(x, y)$: _x é amigo de y_.
  - $Confiável(x)$: _x é confiável_.
  - $Honra(x)$: _x honra y_.

- **Sentença em Português**: Todo amigo de alguém é confiável e honra o amigo.

**4. Fórmula Lógica**:$\exists x \exists y (\text{Animal}(x) \wedge \text{Planta}(y) \wedge \text{Convive}(x, y))
$

- Predicados:

  - $Animal(x)$: _x é um animal_.
  - $Planta(y)$: _y é uma planta_.
  - $Convive(x, y)$: _x e y convivem_.

- **Sentença em Português**: Existe pelo menos um animal e uma planta que convivem no mesmo ambiente.

**5. Fórmula Lógica**:$\forall x \exists y (\text{Professor}(x) \rightarrow (\text{Disciplina}(y) \wedge \text{Leciona}(x, y)))$

- Predicados:

  - $Professor(x)$: _x é um professor_.
  - $Disciplina(y)$: _y é uma disciplina_.
  - $Leciona(x, y)$: _x leciona y_.

- **Sentença em Português**: Para todo professor, existe pelo menos uma disciplina que ele leciona.

**6. Fórmula Lógica**:$\exists x \forall y (\text{Músico}(x) \wedge (\text{Instrumento}(y) \rightarrow \text{Toca}(x, y)))$

- Predicados:

  - $Músico(x)$: _x é um músico_.
  - $Instrumento(y)$: _y é um instrumento_.
  - $Toca(x, y)$: _x toca y_.

- **Sentença em Português**: Existe pelo menos um músico que, se algo é um instrumento, então ele toca esse instrumento.

### Exemplos Extras de conversão de sentenças em predicados

1. **Todos os pássaros voam e todos os peixes nadam.**

   $$\forall x (Pássaro(x) \rightarrow Voa(x)) \land \forall y (Peixe(y) \rightarrow Nada(y))$$

2. **Todos os estudantes estudam ou todos os professores ensinam.**

   $$\forall x (Estudante(x) \rightarrow Estuda(x)) \lor \forall y (Professor(y) \rightarrow Ensina(y))$$

3. **Todos os cães latem e todos os gatos miam, mas nem todos os animais fazem barulho.**

   $$\forall x (Cão(x) \rightarrow Late(x)) \land \forall y (Gato(y) \rightarrow Mia(y)) \land \neg \forall z (Animal(z) \rightarrow FazBarulho(z))$$

4. **Se todos os carros são vermelhos, então todos os caminhões são azuis.**

   $$\forall x (Carro(x) \rightarrow Vermelho(x)) \rightarrow \forall y (Caminhão(y) \rightarrow Azul(y))$$

5. **Todos os planetas orbitam uma estrela e todos os asteroides orbitam o sol.**

   $$\forall x (Planeta(x) \rightarrow OrbitaEstrela(x)) \land \forall y (Asteroide(y) \rightarrow OrbitaSol(y))$$

6. **Alguns pássaros não voam.**

   $$\exists x (Pássaro(x) \land \neg Voa(x))$$

7. **Existe pelo menos um estudante que não estuda**.

   $$\exists x (Estudante(x) \land \neg Estuda(x))$$

8. **Há algum animal que não faz barulho**.

   $$\exists x (Animal(x) \land \neg FazBarulho(x))$$

9. **Existe um carro que não é vermelho**.

   $$\exists x (Carro(x) \land \neg Vermelho(x))$$

10. **Há um planeta que não orbita uma estrela**.

    $$\exists x (Planeta(x) \land \neg \exists y (Estrela(y) \land Orbita(x, y)))$$

11. Todos os pássaros voam, mas existe um animal que não voa.

    $$\forall x (Pássaro(x) \rightarrow Voa(x)) \land \exists y (Animal(y) \land \neg Voa(y))$$

12. Para cada estudante, existe um professor que o ensina.

    $$\forall x (Estudante(x) \rightarrow \exists y (Professor(y) \land Ensina(y, x)))$$

13. Existe um cão que late para todos os gatos.

    $$\exists x (Cão(x) \land \forall y (Gato(y) \rightarrow Late(x, y)))$$

14. Para cada carro vermelho, existe um caminhão azul.

    $$\forall x (Carro(x) \land Vermelho(x) \rightarrow \exists y (Caminhão(y) \land Azul(y)))$$

15. Todos os planetas orbitam uma estrela, e existe um asteroide que orbita o sol.

    $$(\forall x (Planeta(x) \rightarrow \exists y (Estrela(y) \land Orbita(x, y)))) \land (\exists z (Asteroide(z) \land Orbita(z, Sol)))$$

### Exemplos Extras de Conversão de Predicados em Sentenças

1. $\forall x (Gato(x) \rightarrow (Peludo(x) \land Dorminhoco(x)))$

   $$\text{Todo gato é peludo e dorminhoco.}$$

2. $\forall y (Árvore(y) \rightarrow (Verde(y) \land Grande(y)))$

   $$\text{Toda árvore é verde e grande.}$$

3. $(\forall x (Cidade(x) \rightarrow Populosa(x))) \rightarrow (\forall y (País(y) \rightarrow Populoso(y)))$

   $$\text{Se toda cidade é populosa, então todo país é populoso.}$$

4. $\forall x (Criança(x) \rightarrow (Inocente(x) \land Curiosa(x))) \land \neg \exists y (Adulto(y) \land (Inocente(y) \land Curioso(y)))$

   $$\text{Toda criança é inocente e curiosa, e não existe um adulto que seja inocente e curioso.}$$

5. $\forall x (Ave(x) \rightarrow Voa(x)) \land \forall y (Peixe(y) \rightarrow Nada(y))$

   $$\text{Toda ave voa e todo peixe nada.}$$

6. $\exists x (Pessoa(x) \land Feliz(x))$

   $$\text{Existe uma pessoa que é feliz.}$$

7. $\exists y (Livro(y) \land Interessante(y) \land \neg Longo(y))$

   $$\text{Há um livro que é interessante e não é longo.}$$

8. $\exists x (Estudante(x) \land (\forall y (Disciplina(y) \rightarrow Gosta(x, y))))$

   $$\text{Existe um estudante que gosta de todas as disciplinas.}$$

9. $\exists x (Carro(x) \land Rápido(x)) \land \exists y (Carro(y) \land \neg Rápido(y))$

   $$\text{Existe um carro que é rápido, e existe um carro que não é rápido.}$$

10. $\neg \exists x (Político(x) \land Honesto(x))$

    $$\text{Não existe um político que seja honesto.}$$

11. $$\forall x (Cachorro(x) \rightarrow (\exists y (Pessoa(y) \land Dono(y, x))))$$

    $$\text{Todo cachorro tem uma pessoa que é seu dono.}$$

12. $$\exists x (Música(x) \land (\forall y (Pessoa(y) \rightarrow Gosta(y, x))))$$

    $$\text{Existe uma música que todas as pessoas gostam.}$$

13. $$\forall x (Estudante(x) \rightarrow (\exists y (Professor(y) \land Ensina(y, x))))$$

    $$\text{Para todo estudante, existe um professor que o ensina.}$$

14. $$(\exists x (Médico(x) \land Competente(x))) \land (\forall y (Médico(y) \rightarrow Ocupado(y)))$$

    $$\text{Existe um médico que é competente, e todo médico é ocupado.}$$

15. $$(\forall x (Artista(x) \rightarrow Criativo(x))) \rightarrow (\exists y (Pintor(y) \land Criativo(y)))$$

    $$\text{Se todo artista é criativo, então existe um pintor que é criativo.}$$

### Ordem de Aplicação dos Quantificadores

Quando mais de uma variável é quantificada em uma Fórmula Bem Formada como $\forall y\forall x P(x,y)$, elas são aplicadas de dentro para fora, ou seja, a mais próxima da fórmula atômica é aplicada primeiro. Assim, $\forall y\forall x P(x,y)$ se lê _existe um $y$ tal que para todo $x$, $P(x,y)$ se mantém_ ou _para algum $y$, $P(x,y)$ se mantém para todo $x$_.

As posições dos mesmos tipos de quantificadores podem ser trocadas sem afetar o valor lógico, desde que não haja quantificadores do outro tipo entre os que serão trocados.

Por exemplo, $\forall x\forall y\forall z P(x,y,z)$ é equivalente a $\forall y\forall x\forall z P(x,y,z)$, $\forall z\forall y\forall x P(x,y,z)$. O mesmo vale para o quantificador existencial.

No entanto, as posições de quantificadores de tipos diferentes **não** podem ser trocadas. Por exemplo, $\forall x\exists y P(x,y)$ **não** é equivalente A$\exists y\forall x P(x,y)$. Por exemplo, seja $P(x,y)$ representando $x < y$ Para o conjunto dos números como universo. Então, $\forall x\exists y P(x,y)$ se lê _para todo número $x$, existe um número $y$ que é maior que $x$_, o que é verdadeiro, enquanto $\exists y\forall x P(x,y)$ se lê _existe um número que é maior que todo (qualquer) número_, o que não é verdadeiro.

#### Negação dos Quantificadores

Existe uma equivalência entre as negações dos quantificadores. De tal forma que:

1. **Negação do Quantificador Universal ($\forall $)**: A negação de uma afirmação universal significa que existe pelo menos um caso no Universo do Discurso, onde a afirmação não é verdadeira. Isso pode ser expresso pela seguinte equivalência:

   $$\neg \forall x \, P(x) \equiv \exists x \, \neg P(x)$$

   Em linguagem natural podemos entender como: negar que _para todos os $x$, $P(x)$ é verdadeiro_ é equivalente a afirmar que _existe algum $x$ tal que $P(x)$ não é verdadeiro_.

2. **Negação do Quantificador Existencial ( $\exists $ )**: A negação de uma afirmação existencial significa que a afirmação não é verdadeira para nenhum caso no Universo do Discurso. Isso pode ser expresso pela seguinte equivalência:

$$\neg \exists x \, P(x) \equiv \forall x \, \neg P(x)$$

Ou seja, negar que _existe algum $x$ tal que $P(x)$ é verdadeiro_ é equivalente a afirmar que _para todos os $x$, $P(x)$ não é verdadeiro_.

Vamos tentar entender estas negações. Considere as expressões $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$. Essas fórmulas se aplicam a qualquer predicado $P$, e possuem o mesmo valor de verdade para qualquer $P$.

Na lógica proposicional, poderíamos simplesmente verificar isso com uma tabela verdade, mas aqui, não podemos. Não existem proposições, conectadas por $\land $, $\lor $, para construir uma tabela e não é possível determinar o valor verdade de forma genérica para uma determinada variável.

Vamos tentar entender isso com linguagem natural: afirmar que $\neg (\forall x P(x))$ é verdadeiro significa que não é verdade que $P(x)$ se aplica a todas as possíveis entidades $x$. Deve haver alguma entidade $A$ Para a qual$P(a)$ é falso. Como $P(a)$ é falso, $\neg P(a)$ é verdadeiro. Isso significa que $\exists x (\neg P(x))$ é verdadeiro. Portanto, a verdade de $\neg (\forall x P(x))$implica a verdade de $\exists x (\neg P(x))$.

Se $\neg (\forall x P(x))$ é falso, então $\forall x P(x)$ é verdadeiro. Como $P(x)$ é verdadeiro para todos os $x$, $\neg P(x)$ é falso para todos os $x$. Logo, $\exists x (\neg P(x))$ é falso.

Os valores de verdade de $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$ são os mesmos. Como isso é verdadeiro para qualquer predicado $P$, essas duas fórmulas são logicamente equivalentes, e podemos escrever $\neg (\forall x P(x)) \equiv \exists x (\neg P(x))$.

Muita lógica? Que tal se tentarmos novamente, usando um pouco mais de linguagem natural.
Considere as expressões lógicas $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$. Para ilustrar essas fórmulas, vamos usar um exemplo com um predicado $P(x)$ que se aplica a uma entidade $x$ se _$x$ é feliz_.

A expressão $\forall x P(x)$ significa que todos são felizes. A negação dessa afirmação, $\neg (\forall x P(x))$, equivale logicamente a $\exists x (\neg P(x))$, ou seja, existe pelo menos um indivíduo que não é feliz.

A expressão $\exists x (\neg P(x))$ significa que _existe alguém que não está feliz_. Você pode ver que isso é apenas outra forma de expressar a ideia contida em $\neg (\forall x P(x))$.

A afirmação de que _não é verdade que todos estão felizes_ implica que deve haver alguém que não está feliz. Se a primeira afirmação é falsa (ou seja, todos estão felizes), então a segunda afirmação também deve ser falsa.

Portanto, as duas fórmulas têm o mesmo valor verdade. Elas são logicamente equivalentes e podem ser representadas como $\neg (\forall x P(x)) \equiv \exists x (\neg P(x))$. Esta equivalência reflete uma relação profunda e intuitiva em nosso entendimento de declarações sobre entidades em nosso mundo.

| Expressão | Equivalência |
|---|---|
| $\forall x P(x)$ | $\neg \exists x \neg P(x)$ |
| $\exists x \, P(x)$ | $\neg \forall x \, \neg P(x)$ |
| $\neg \forall x \, P(x)$ | $\exists x \, \neg P(x)$ |
| $\neg \exists x \, P(x)$ | $\forall x \, \neg P(x)$ |

_Tabela 5 - Equivalências entre Quantificadores._</legend>

### Regras de Inferência usando Quantificadores

As regras de inferência com quantificadores lidam especificamente com as proposições que envolvem quantificadores. Estas regras nos permitem fazer generalizações ou especificações, transformando proposições universais em existenciais, e vice-versa. Compreender essas regras é essencial para aprofundar o entendimento da estrutura da lógica, o que nos permite analisar e construir argumentos mais complexos de forma precisa e coerente.

Nos próximos tópicos, exploraremos essas regras em detalhes, observando como elas interagem com os quantificadores universal e existencial.

#### Repetição

A regra de Repetição permite repetir uma afirmação. Esta regra é útil para propagar premissas em uma prova formal.

$$F$$

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

  - Proposição: _todos os homens, $H(x)$, são mortais, M(x)$_.
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
  - Conclusão: logo, \_$2 + 0 = 2$.

$$
\begin{aligned}
&\forall x(x + 0 = x)\\
\hline
&2 + 0 = 2
\end{aligned}
$$

#### Instanciação Universal

A regra de Instanciação Universal permite substituir a variável em uma afirmação universalmente quantificada por um termo concreto. Esta regra nos permite derivar casos particulares a partir de afirmações gerais.

$$\forall x P(x)$$

$$\begin{aligned}
&\forall x P(x)\\
\hline
&P(a)\\
\end{aligned}$$

Em linguagem natural:

- Proposição: _todos os humanos são mortais_.
- Conclusão: logo, _Sócrates é mortal_. Assumindo que Sócrates é humano.

**Exemplo completo de aplicação**:

- **Premissa 1**: todos os mamíferos respiram ar: $\forall x(M(x) \rightarrow R(x))$;
- **Premissa 2**: a baleia é um mamífero: $M(b)$;
- **Aplicação da Instanciação Universal à Premissa 1**: $M(b) \rightarrow R(b)$;
- **Aplicação de Modus Ponens**:
  
  $$\begin{aligned}
  &M(b) \rightarrow R(b)\\
  &M(b)\\
  \hline
  &R(b)
  \end{aligned}$$

- **Conclusão**: logo, a baleia respira ar: $R(b)$

Algumas aplicações da Instanciação Universal:

- Aplicar regras e princípios gerais. Por exemplo:

  - Proposição: _todos os triângulos têm 180 graus internos_: $\forall t(T(t) \rightarrow 180^\circ(t))$;
  - Premissa adicional: _ABC é um triângulo_: $T(\text{Triângulo }ABC)$;
  - Aplicação da Instanciação Universal: $T(\text{Triângulo }ABC) \rightarrow 180^\circ(\text{Triângulo }ABC)$;
  - Aplicação de Modus Ponens:
  
  $$\begin{aligned}
  &T(\text{Triângulo }ABC) \rightarrow 180^\circ(\text{Triângulo }ABC)\\
  &T(\text{Triângulo }ABC)\\
  \hline
  &180^\circ(\text{Triângulo }ABC)
  \end{aligned}$$

  - Conclusão: logo, _o triângulo $ABC$ tem 180 graus_.

- Testar propriedades em membros de conjuntos. Por exemplo:

  - Proposição: _todo inteiro é maior que seu antecessor_: $\forall n (\mathbb{Z}(n) \rightarrow (n > n-1))$;
  - Premissa adicional: _5 é um inteiro_: $\mathbb{Z}(5)$;
  - Aplicação da Instanciação Universal: $\mathbb{Z}(5) \rightarrow (5 > 5-1)$;
  - Aplicação de Modus Ponens:
  
  $$\begin{aligned}
  &\mathbb{Z}(5) \rightarrow (5 > 5-1)\\
  &\mathbb{Z}(5)\\
  \hline
  &5 > 4
  \end{aligned}$$
  
  - Conclusão: logo, $5$ é maior que $4$.

#### Generalização Existencial

A regra de Generalização Existencial permite inferir que algo existe a partir de uma afirmação concreta. Esta regra nos permite generalizar de exemplos específicos para a existência geral.

$$P(a)$$

$$
\begin{aligned}
P(a)\\
\hline
\exists x P(x)\\
\end{aligned}
$$

Em linguagem natural:

- Proposição: _Rex é um cachorro_.
- Conclusão: logo, _existe pelo menos um cachorro_.

Algumas aplicações da Generalização Existencial:

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

  - Proposição: _$7 $ é um número primo_.
  - Conclusão: logo, _existe pelo menos um número primo_.

$$
\begin{aligned}
&P(7)\\
\hline
&\exists x P(x)
\end{aligned}
$$

- Inferir a existência de soluções para problemas. Por exemplo:

  - Proposição: _$x = 2 $ satisfaz a equação $x + 3 = 5 $_.
  - Conclusão: logo, _existe pelo menos uma solução para essa equação_.

$$
\begin{aligned}
&S(2)\\
\hline
&\exists x S(x)
\end{aligned}
$$

#### Instanciação Existencial

A regra de Instanciação Existencial permite introduzir um novo termo como instância de uma variável existencialmente quantificada. Esta regra nos permite derivar exemplos de afirmações existenciais.

$$\exists x P(x)$$

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

Algumas aplicações da Instanciação Existencial:

- Derivar exemplos de existência previamente estabelecida. Por exemplo:

  - Proposição: _existem estrelas, $ e $, maiores, $M $, que o Sol, $s $_.
  - Conclusão: logo, _Alpha Centauri, $A$, é maior que o Sol_.

$$
\begin{aligned}
&\exists x (e (x) \land M(x, s))\\
\hline
&M(a, s)
\end{aligned}
$$

- Construir modelos satisfatíveis para predicados existenciais. Por exemplo:

  - Proposição: _existem pessoas mais velhas que $25$Anos_.
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

## Análise de Argumentos Lógicos em Textos do Cotidiano

Uma das coisas mais importantes que a amável leitora irá aprender neste documento é que a lógica não é apenas uma disciplina acadêmica, mas uma ferramenta poderosa para analisar e compreender argumentos em textos do cotidiano. A lógica nos ajuda a identificar premissas e conclusões, avaliar a validade de argumentos e entender como as ideias estão interconectadas. A lógica é útil e pode ser a diferença entre um argumento persuasivo e um argumento falacioso. Além disso, o uso da lógica para entender textos do cotidiano criará as estruturas cognitivas necessários para a construção de soluções computacionais para os problemas mais complexos da atualidade.

A análise de argumentos lógicos em textos do cotidiano envolve a identificação de premissas e conclusões, bem como a formalização dessas relações em lógica proposicional ou lógica de predicados.

Deste ponto tem diante, caberá a esforçada leitora, aplicar os conceitos aprendidos neste documento para analisar textos de livros, notícias, especificações de sistemas e outros documentos. Comece vendo os textos de exemplo a seguir: 

### Exemplo 1: Avanço Científico na Medicina (2022)

"Em 2022, pesquisadores descobriram que a vacina contra a malária aprovada pela Organização Mundial da Saúde pode ajudar a salvar centenas de milhares de vidas por ano. É também a primeira vacina do mundo para uma infecção parasitária. A malária mata mais de 600.000 pessoas anualmente, a maioria crianças menores de cinco anos."

#### Premissas e Conclusão

**Premissas**:

1. A malária mata mais de 600.000 pessoas anualmente, majoritariamente crianças menores de cinco anos;
2. A vacina contra a malária foi aprovada pela Organização Mundial da Saúde;
3. A vacina contra a malária é a primeira vacina do mundo para uma infecção parasitária.

**Conclusão**:

- A vacina contra a malária pode ajudar a salvar centenas de milhares de vidas por ano.

#### Formalização Lógica

**Lógica Proposicional**:

- $P$: A malária mata mais de 600.000 pessoas anualmente, majoritariamente crianças menores de cinco anos;
- $Q$: A vacina contra a malária foi aprovada pela OMS;
- $R$: A vacina contra a malária é a primeira vacina do mundo para uma infecção parasitária;
- $S$: A vacina contra a malária pode ajudar a salvar centenas de milhares de vidas por ano.

Estrutura do argumento: $(P \land Q \land R) \rightarrow S$

**Lógica de Predicados**:

- $m$: malária;
- $v$: vacina contra a malária;
- $Mata(x, y)$: x mata y pessoas anualmente;
- $MajoriaCriancas(x)$: a maioria das vítimas de x são crianças menores de cinco anos;
- $Aprovada(x)$: x foi aprovada pela OMS;
- $Primeira(x)$: x é a primeira vacina para infecção parasitária;
- $PodeSalvar(x, y)$: x pode salvar y vidas por ano.

**Formalização**:

1. $Mata(m, 600000) \land MajoriaCriancas(m)$;
2. $Aprovada(v)$;
3. $Primeira(v)$;
4. $[Mata(m, 600000) \land MajoriaCriancas(m) \land Aprovada(v) \land Primeira(v)] \rightarrow PodeSalvar(v, \text{"centenas de milhares"})$.

#### Análise da Validade

Este argumento não segue uma forma lógica estritamente válida. A relação entre as premissas e a conclusão depende de conhecimentos médicos implícitos.

Estrutura implícita:

- Se uma doença mata muitas pessoas e existe uma vacina aprovada contra essa doença, então essa vacina pode salvar muitas vidas;
- A malária mata muitas pessoas;
- Existe uma vacina aprovada contra a malária;
- Logo, a vacina contra a malária pode salvar muitas vidas.

Esta estrutura se aproxima de um modus ponens, mas depende de uma premissa implícita.

#### Análise da Solidez

As premissas são verificáveis e consideradas verdadeiras:

- A mortalidade por malária é confirmada por dados epidemiológicos da OMS;
- A aprovação da vacina pela OMS é um fato verificável;
- Ser a primeira vacina para infecção parasitária é historicamente verificável.

A conclusão é razoável no contexto médico, mas sua solidez completa dependeria de dados específicos sobre a eficácia da vacina.

### Exemplo 2: Inovação Tecnológica Nuclear (2023)

"O campo da fusão nuclear teve um grande avanço em 2023. A fusão nuclear é uma reação química que produz uma grande quantidade de calor que pode ser usada para gerar energia. É o mesmo processo que alimenta o sol. A reação química é produzida por dois núcleos atômicos leves que se combinam e formam um único núcleo atômico leve mais pesado. Isso produz uma grande quantidade de energia."

#### Premissas e Conclusão

**Premissas**:

1. A fusão nuclear é uma reação que ocorre quando dois núcleos atômicos leves se combinam formando um único núcleo mais pesado;
2. Esta reação produz grande quantidade de calor;
3. O calor pode ser usado para gerar energia;
4. A fusão nuclear é o mesmo processo que alimenta o sol.

**Conclusão**:

- O campo da fusão nuclear teve um grande avanço em 2023.

#### Formalização Lógica

**Lógica Proposicional**:

- $P$: A fusão nuclear é uma reação onde núcleos leves se combinam formando um núcleo mais pesado;
- $Q$: A fusão nuclear produz grande quantidade de calor;
- $R$: O calor pode ser usado para gerar energia;
- $S$: A fusão nuclear é o mesmo processo que alimenta o sol;
- $T$: O campo da fusão nuclear teve um grande avanço em 2023.

Estrutura do argumento: $(P \land Q \land R \land S) \rightarrow T$

**Lógica de Predicados**:

- $FusaoNuclear(x)$: $x$ é um processo de fusão nuclear;
- $Reacao(x, y, z)$: $x$ é uma reação onde $y$ se combina formando $z$;
- $Produz(x, y)$: $x$ produz $y$;
- $PodeGerarEnergia(x)$: $x$ pode ser usado para gerar energia;
- $AlimentaSol(x)$: $x$ é o processo que alimenta o sol;
- $TeveAvanco(x, y, z)$: o campo $x$ teve um avanço de grau $y$ no ano $z$.

**Formalização**:

1. $\forall x [FusaoNuclear(x) \rightarrow Reacao(x, \text{"núcleos leves"}, \text{"núcleo mais pesado"})]$;
2. $\forall x [FusaoNuclear(x) \rightarrow Produz(x, \text{"grande quantidade de calor"})]$;
3. $\forall x [Produz(x, \text{"grande quantidade de calor"}) \rightarrow PodeGerarEnergia(x)]$;
4. $\forall x [FusaoNuclear(x) \rightarrow AlimentaSol(x)]$;
5. $TeveAvanco(\text{"campo da fusão nuclear"}, \text{"grande"}, 2023)$.

#### Análise da Validade

Este argumento apresenta uma estrutura incomum, pois a conclusão não é derivada logicamente das premissas apresentadas. As premissas descrevem o que é a fusão nuclear e suas características, mas não estabelecem uma relação lógica com o avanço mencionado.

Sob análise de dedução natural, o argumento não é válido, pois a conclusão não é uma consequência lógica das premissas fornecidas.

#### Análise da Solidez

Como o argumento não é formalmente válido, não pode ser considerado sólido. Entretanto, suas premissas são majoritariamente verdadeiras:

- A definição de fusão nuclear como combinação de núcleos leves é cientificamente precisa;
- A produção de calor e seu potencial energético são verdadeiros;
- A fusão nuclear realmente alimenta o sol.

Há um erro conceitual no texto: a fusão nuclear é descrita como "reação química", quando na verdade é uma reação nuclear, comprometendo a precisão científica do texto.

### Exemplo 3: Economia Global (2023)

"As economias avançadas devem desacelerar de 2,6% em 2022 para 1,5% em 2023 e 1,4% em 2024, à medida que o aperto da política começa a surtir efeito. A inflação global deverá diminuir constantemente, de 8,7% em 2022 para 6,9% em 2023 e 5,8% em 2024, devido a uma política monetária mais rígida auxiliada por preços mais baixos das commodities internacionais."

#### Premissas e Conclusão

**Premissas**:

1. O aperto da política (monetária) está começando a surtir efeito;
2. Está sendo implementada uma política monetária mais rígida;
3. Os preços das commodities internacionais estão mais baixos.

**Conclusões**:

1. As economias avançadas devem desacelerar de $2,6\%$ em 2022 para $1,5\%$ em 2023 e $1,4\%$ em 2024;
2. A inflação global deverá diminuir constantemente, de $8,7\%$ em 2022 para $6,9\%$ em 2023 e $5,8\%$ em 2024.

#### Formalização Lógica

**Lógica Proposicional**:

- $P$: O aperto da política monetária está surtindo efeito;
- $Q$: Está sendo implementada uma política monetária mais rígida;
- $R$: Os preços das commodities internacionais estão mais baixos;
- $S$: As economias avançadas desacelerarão para $1,5\%$ em 2023 e $1,4\%$ em 2024;
- $T$: A inflação global diminuirá para $6,9\%$ em 2023 e $5,8\%$ em 2024.

Estrutura do argumento: $(P \land Q \land R) \rightarrow (S \land T)$

**Lógica de Predicados**:

- $ApertoSurteEfeito(x)$: o aperto da política monetária $x$ está surtindo efeito;
- $PoliticaRigida(x)$: $x$ é uma política monetária rígida;
- $PrecosBaixos(x)$: os preços de $$x$ estão baixos;
- $Desacelerar(x, y, z)$: a economia $x$ desacelerará para taxa $y$ no ano $z$;
- $DiminuirInflacao(x, y, z)$: a inflação $x$ diminuirá para taxa $y$ no ano $z$.

**Formalização**:

1. $ApertoSurteEfeito(\text{"política monetária"})$;
2. $PoliticaRigida(\text{"política monetária atual"})$;
3. $PrecosBaixos(\text{"commodities internacionais"})$;
4. $[ApertoSurteEfeito(\text{"política monetária"}) \land PoliticaRigida(\text{"política monetária atual"})] \rightarrow Desacelerar(\text{"economias avançadas"}, 1.5\%, 2023) \land Desacelerar(\text{"economias avançadas"}, 1.4\%, 2024)$;
5. $[PoliticaRigida(\text{"política monetária atual"}) \land PrecosBaixos(\text{"commodities internacionais"})] \rightarrow DiminuirInflacao(\text{"global"}, 6.9\%, 2023) \land DiminuirInflacao(\text{"global"}, 5.8\%, 2024)$.

#### Análise da Validade

Este argumento segue uma estrutura causal que pode ser analisada pela forma lógica:

- Se $X$ causa $Y$, e $X$ está ocorrendo, então $Y$ ocorrerá;
- $X$ está ocorrendo;
- Portanto, $Y$ ocorrerá.

Esta estrutura segue o padrão de modus ponens, que é uma forma de argumento válida.

#### Análise da Solidez

A validade lógica do argumento foi estabelecida, mas sua solidez depende da veracidade das premissas:

1. A eficácia do aperto monetário é uma afirmação empírica que requer verificação com dados econômicos;
2. A implementação de política monetária mais rígida era geralmente verdadeira no contexto de 2023;
3. A afirmação sobre preços mais baixos de commodities depende do período específico e das commodities consideradas.

As conclusões são previsões específicas cuja solidez dependeria da veracidade das premissas, da robustez dos modelos econômicos e da ausência de fatores externos imprevistos.

Em economia, relações causais são geralmente probabilísticas, tornando a solidez do argumento contingente a condições específicas.

### Exercício de Análise de Argumentos Lógicos

**Objetivo**: aplicar técnicas de lógica proposicional e de predicados para analisar descrições e especificações de sistemas computacionais, traduzindo-as para a linguagem formal e avaliando sua consistência lógica como base para decisões de implementação.

**Descrição**: na engenharia de software, especificações e requisitos de sistemas são frequentemente descritos em linguagem natural, o que pode levar a ambiguidades, inconsistências e interpretações equivocadas. A análise lógica formal dessas descrições pode ajudar a identificar tais problemas e proporcionar uma base sólida para o desenvolvimento de soluções computacionais.
Nesta tarefa, você atuará como "Arquiteto Lógico de Sistemas" para traduzir especificações em linguagem natural para modelos lógicos formais.

#### Exercício 1: Sistema de Autenticação Biométrica

**Fragmento de Texto Original**: O sistema de autenticação biométrica deve permitir o acesso a usuários autorizados por meio de reconhecimento facial ou impressão digital. Se um usuário não conseguir autenticar por nenhum dos métodos biométricos, o sistema deve oferecer como alternativa a autenticação por senha. Caso ocorram três tentativas falhas consecutivas por qualquer método, o acesso do usuário deve ser temporariamente bloqueado por 30 minutos por motivos de segurança.

**Solução**: A tarefa é analisar o fragmento de texto e formalizá-lo em lógica proposicional e lógica de predicados, identificando premissas, conclusões e avaliando a validade e solidez do argumento.

**Premissas**:

1. Usuários autorizados podem se autenticar por reconhecimento facial;
2. Usuários autorizados podem se autenticar por impressão digital;
3. Se a autenticação biométrica falhar, o usuário pode usar senha;
4. Três tentativas falhas consecutivas levam ao bloqueio temporário;
5. O bloqueio temporário dura 30 minutos.

**Conclusões**:

- O sistema deve bloquear o acesso após três tentativas falhas consecutivas;
- O sistema deve permitir múltiplos métodos de autenticação.

**Formalização**

**Lógica Proposicional**:

- $A$: O usuário está autorizado;
- $F$: O usuário autentica com reconhecimento facial;
- $D$: O usuário autentica com impressão digital;
- $S$: O usuário autentica com senha;
- $T$: Ocorreram três tentativas falhas consecutivas;
- $B$: O acesso do usuário está bloqueado temporariamente.

**Estrutura do argumento**:

1. $A \rightarrow (F \lor D \lor S)$;
2. $\neg(F \lor D) \rightarrow S$;
3. $T \rightarrow B$.

**Lógica de Predicados**:

- $Usuario(x)$: $x$ é um usuário;
- $Autorizado(x)$: $x$ é autorizado;
- $AutenticaFacial(x)$: $x$ autentica por reconhecimento facial;
- $AutenticaDigital(x)$: $x$ autentica por impressão digital;
- $AutenticaSenha(x)$: $x$ autentica por senha;
- $TentativasFalhas(x, n)$: $x$ teve $n$ tentativas falhas consecutivas;
- $Bloqueado(x, t)$: $x$ está bloqueado por $t$ minutos.

**Formalização**:

1. $\forall x [Autorizado(x) \rightarrow (AutenticaFacial(x) \lor AutenticaDigital(x) \lor AutenticaSenha(x))]$;
2. $\forall x [(Usuario(x) \land \neg(AutenticaFacial(x) \lor AutenticaDigital(x))) \rightarrow AutenticaSenha(x)]$;
3. $\forall x [TentativasFalhas(x, 3) \rightarrow Bloqueado(x, 30)]$.

**Análise da Validade**: o argumento é válido em termos de lógica proposicional e de predicados. A estrutura segue formas lógicas válidas:

1. A primeira relação estabelece uma disjunção inclusiva (OR) de métodos de autenticação disponíveis para usuários autorizados;

2. A segunda relação segue a forma $(P \land \neg Q) \rightarrow R$, que é válida: se um usuário não consegue autenticar pelos métodos biométricos, então deve poder usar senha;

3. A terceira relação segue a forma $P \rightarrow Q$, um modus ponens: se ocorrerem três tentativas falhas, então o bloqueio é implementado.

**Análise da Solidez**:

As premissas são razoáveis no contexto de sistemas de autenticação modernos:

1. A disponibilidade de múltiplos métodos de autenticação aumenta a usabilidade;
2. A provisão de métodos alternativos quando os biométricos falham é uma prática comum;
3. O bloqueio após múltiplas tentativas falhas é um mecanismo de segurança padrão.

As conclusões derivadas são sólidas no contexto de sistemas de autenticação e seguem práticas recomendadas de segurança digital.

#### Exercício 2: Processamento de Pagamentos Online

**Fragmento de Texto Original**: O sistema de pagamentos online deve processar transações com cartões de crédito, cartões de débito e carteiras digitais. Quando uma transação é iniciada, o sistema verifica primeiro se há fundos suficientes. Se houver fundos suficientes, o sistema realiza a verificação de segurança. Uma transação só é aprovada se ambas as verificações forem bem-sucedidas. Caso contrário, a transação é rejeitada e o cliente recebe uma notificação com o motivo da falha.

**Solução**: A tarefa é analisar o fragmento de texto e formalizá-lo em lógica proposicional e lógica de predicados, identificando premissas, conclusões e avaliando a validade e solidez do argumento.

**Premissas**:

1. O sistema processa transações com cartões de crédito, cartões de débito e carteiras digitais;
2. Quando uma transação é iniciada, o sistema verifica a disponibilidade de fundos;
3. Se há fundos suficientes, o sistema realiza verificação de segurança;
4. Uma transação é aprovada apenas se as verificações de fundos e segurança forem bem-sucedidas;
5. Transações rejeitadas geram notificações com o motivo da falha.

**Conclusão**:

- Se uma verificação de fundos ou segurança falhar, a transação será rejeitada.

**Formalização**

**Lógica Proposicional**:

- $C$: A transação é com cartão de crédito;
- $D$: A transação é com cartão de débito;
- $W$: A transação é com carteira digital;
- $F$: Há fundos suficientes;
- $S$: A verificação de segurança é bem-sucedida;
- $A$: A transação é aprovada;
- $R$: A transação é rejeitada;
- $N$: O cliente recebe notificação.

Estrutura do argumento:

1. $(C \lor D \lor W)$ (A transação é feita por um dos métodos aceitos);
2. $F \rightarrow S$ (Se há fundos, realiza-se verificação de segurança);
3. $(F \land S) \rightarrow A$ (Se há fundos e a verificação de segurança é bem-sucedida, a transação é aprovada);
4. $\neg(F \land S) \rightarrow (R \land N)$ (Se não há fundos ou a verificação falha, a transação é rejeitada e há notificação).

**Lógica de Predicados**:

- $Transacao(x)$: $x$ é uma transação;
- $Metodo(x, y)$: a transação $x$ utiliza o método de pagamento $y$;
- $TemFundos(x)$: a transação $x$ tem fundos suficientes;
- $VerificacaoSeguranca(x)$: a transação $x$ passa na verificação de segurança;
- $Aprovada(x)$: a transação $x$ é aprovada;
- $Rejeitada(x)$: a transação $x$ é rejeitada;
- $Notifica(x, y)$: o sistema notifica o cliente sobre $y$ relacionado à transação $x$.

**Formalização**:

1. $\forall x [Transacao(x) \rightarrow (Metodo(x, \text{"crédito"}) \lor Metodo(x, \text{"débito"}) \lor Metodo(x, \text{"carteira digital"}))]$;
2. $\forall x [Transacao(x) \rightarrow (TemFundos(x) \rightarrow VerificacaoSeguranca(x))]$;
3. $\forall x [Transacao(x) \rightarrow ((TemFundos(x) \land VerificacaoSeguranca(x)) \rightarrow Aprovada(x))]$;
4. $\forall x [Transacao(x) \rightarrow (\neg(TemFundos(x) \land VerificacaoSeguranca(x)) \rightarrow (Rejeitada(x) \land \exists y Notifica(x, y)))]$.

**Análise da Validade**:

O argumento é válido logicamente. As relações causais seguem formas lógicas consistentes:

1. A primeira premissa estabelece os métodos de pagamento aceitos, formando uma disjunção inclusiva;
2. A relação entre verificação de fundos e verificação de segurança segue um fluxo condicional válido;
3. A aprovação da transação requer a conjunção (AND) de condições, seguindo o padrão $(P \land Q) \rightarrow R$;
4. A rejeição da transação ocorre pela negação da conjunção, usando a lei de De Morgan: $\neg(P \land Q) \equiv \neg P \lor \neg Q$;

**Análise da Solidez**:

As premissas são sólidas no contexto de sistemas de processamento de pagamentos:

1. Os métodos de pagamento mencionados são comuns em sistemas reais;
2. A verificação de fundos antes de processamento é uma prática padrão;
3. As verificações de segurança são essenciais em transações financeiras;
4. A notificação em caso de falha é uma boa prática para experiência do usuário.

A conclusão derivada é sólida e consistente com o funcionamento esperado de um sistema de pagamentos seguro e funcional.

#### Exercício 3: Sistema de Gerenciamento de Estoque

**Fragmento de Texto Original**: "O sistema de gerenciamento de estoque deve monitorar continuamente os níveis de produtos. Quando o estoque de um produto cai abaixo do limite mínimo configurado, o sistema deve gerar automaticamente uma ordem de reabastecimento. Se o produto estiver marcado como 'crítico', a ordem deve ser enviada com prioridade alta. Caso contrário, a ordem segue o fluxo padrão. Qualquer produto que não tenha movimento de venda por mais de 90 dias deve ser marcado para revisão de demanda."

**Solução**: A tarefa é analisar o fragmento de texto e formalizá-lo em lógica proposicional e lógica de predicados, identificando premissas, conclusões e avaliando a validade e solidez do argumento.

**Premissas**:

1. O sistema monitora continuamente os níveis de estoque dos produtos;
2. Existe um limite mínimo configurado para cada produto;
3. Ordens de reabastecimento são geradas quando o estoque cai abaixo do limite mínimo;
4. Produtos podem ser marcados como 'críticos';
5. Produtos críticos recebem prioridade alta no reabastecimento;
6. Produtos sem movimento de venda por mais de 90 dias são marcados para revisão.

**Conclusões**:

- Se o estoque de um produto cai abaixo do limite e o produto é crítico, uma ordem de reabastecimento com prioridade alta é gerada;
- Se o estoque de um produto cai abaixo do limite e o produto não é crítico, uma ordem de reabastecimento padrão é gerada;
- Se um produto não tem vendas por mais de 90 dias, ele deve ser revisado.

**Formalização**

**Lógica Proposicional**:

- $M$: O sistema monitora os níveis de estoque;
- $B$: O estoque está abaixo do limite mínimo;
- $O$: Uma ordem de reabastecimento é gerada;
- $C$: O produto é marcado como crítico;
- $P$: A ordem é enviada com prioridade alta;
- $F$: A ordem segue fluxo padrão;
- $N$: O produto não tem movimento de venda por mais de 90 dias;
- $R$: O produto é marcado para revisão de demanda.

**Estrutura do argumento**:

1. $M$;
2. $B \rightarrow O$;
3. $(B \land C) \rightarrow (O \land P)$;
4. $(B \land \neg C) \rightarrow (O \land F)$;
5. $N \rightarrow R$.

**Lógica de Predicados**:

- $Produto(x)$: $x$ é um produto;
- $Monitora(x)$: o sistema monitora o estoque de $x$;
- $AbaixoLimite(x)$: o estoque de $x$ está abaixo do limite mínimo;
- $Critico(x)$: $x$ é marcado como crítico;
- $GeraOrdem(x, y)$: o sistema gera uma ordem de reabastecimento para $x$ com prioridade $y$;
- $SemVendas(x, d)$: $x$ não tem vendas por $d$ dias;
- $MarcarRevisao(x)$: $x$ é marcado para revisão de demanda.

**Formalização**:

1. $\forall x [Produto(x) \rightarrow Monitora(x)]$;
2. $\forall x [Produto(x) \land AbaixoLimite(x) \rightarrow \exists y \, GeraOrdem(x, y)]$;
3. $\forall x [Produto(x) \land AbaixoLimite(x) \land Critico(x) \rightarrow GeraOrdem(x, \text{"alta"})]$;
4. $\forall x [Produto(x) \land AbaixoLimite(x) \land \neg Critico(x) \rightarrow GeraOrdem(x, \text{"normal"})]$;
5. $\forall x [Produto(x) \land SemVendas(x, 90) \rightarrow MarcarRevisao(x)]$.

**Análise da Validade**:

O argumento é logicamente válido. A estrutura segue padrões lógicos consistentes:

1. A relação entre níveis de estoque e geração de ordens segue um modus ponens;
2. A distinção entre produtos críticos e não críticos usa corretamente a conjunção e a negação;
3. A condição para revisão de demanda segue uma implicação simples.

As regras de negócio são representadas por condicionais bem formados, sem contradições ou ambiguidades lógicas.

**Análise da Solidez**:

As premissas são sólidas no contexto de sistemas de gerenciamento de estoque:

1. O monitoramento contínuo de estoque é uma funcionalidade essencial desses sistemas;
2. O conceito de limite mínimo para reabastecimento é uma prática comum;
3. A priorização de produtos críticos é uma estratégia logística válida;
4. A revisão de produtos sem movimentação é uma prática de otimização de estoque reconhecida.

As conclusões derivadas são sólidas e refletem procedimentos operacionais padrão em gerenciamento de estoque e logística.

## Análise 4: Sistema de Recomendação de Conteúdo

### Texto Original
"O sistema de recomendação deve analisar o histórico de visualizações, preferências explícitas e comportamento de navegação de cada usuário. Com base nesses dados, o sistema calcula um score de relevância para cada item de conteúdo disponível. Itens com score acima de 0,7 são recomendados ao usuário. No entanto, se o usuário já visualizou um item nos últimos 30 dias, este não deve ser recomendado novamente, independentemente do score. Adicionalmente, se o usuário deu um feedback negativo a um conteúdo similar, o score desse tipo de conteúdo deve ser reduzido em 0,3 pontos."

**Solução**: A tarefa é analisar o fragmento de texto e formalizá-lo em lógica proposicional e lógica de predicados, identificando premissas, conclusões e avaliando a validade e solidez do argumento.

**Premissas**:
1. O sistema analisa o histórico de visualizações, preferências explícitas e comportamento de navegação.
2. Um score de relevância é calculado para cada item de conteúdo.
3. Itens com score acima de 0,7 são recomendados.
4. Itens visualizados nos últimos 30 dias não são recomendados, independente do score.
5. Feedback negativo a conteúdo similar reduz o score em 0,3 pontos.

**Conclusões**:
- Um item será recomendado se seu score for maior que 0,7 E não tiver sido visualizado nos últimos 30 dias.
- O feedback negativo do usuário influencia o cálculo do score para itens similares.

### Formalização Lógica

**Lógica Proposicional**:
- $A$: O sistema analisa dados do usuário.
- $C$: O sistema calcula scores de relevância.
- $S$: O item tem score acima de 0,7.
- $V$: O item foi visualizado nos últimos 30 dias.
- $R$: O item é recomendado ao usuário.
- $F$: O usuário deu feedback negativo a conteúdo similar.
- $D$: O score é reduzido em 0,3 pontos.

Estrutura do argumento:
1. $A \rightarrow C$
2. $(S \land \neg V) \rightarrow R$
3. $V \rightarrow \neg R$
4. $F \rightarrow D$

**Lógica de Predicados**:
- $Usuario(u)$: u é um usuário
- $Item(i)$: i é um item de conteúdo
- $AnalisaDados(u)$: o sistema analisa dados do usuário u
- $Score(i, s)$: o item i tem score s
- $Visualizado(u, i, d)$: o usuário u visualizou o item i nos últimos d dias
- $Recomendado(u, i)$: o item i é recomendado ao usuário u
- $FeedbackNegativo(u, t)$: o usuário u deu feedback negativo ao tipo de conteúdo t
- $Similar(i, t)$: o item i é similar ao tipo de conteúdo t
- $ReducaoScore(i, v)$: o score do item i é reduzido em v pontos

**Formalização**:

1. $\forall u [Usuario(u) \rightarrow AnalisaDados(u)]$;
2. $\forall u \forall i [Usuario(u) \land Item(i) \land Score(i, s) \land s > 0.7 \land \neg Visualizado(u, i, 30) \rightarrow Recomendado(u, i)]$;
3. $\forall u \forall i [Usuario(u) \land Item(i) \land Visualizado(u, i, 30) \rightarrow \neg Recomendado(u, i)]$;
4. $\forall u \forall i \forall t [Usuario(u) \land Item(i) \land Similar(i, t) \land FeedbackNegativo(u, t) \rightarrow ReducaoScore(i, 0.3)]$.

**Análise da Validade**:

O argumento é logicamente válido. As regras de recomendação seguem formas lógicas bem definidas:

1. A relação entre análise de dados e cálculo de scores é uma implicação simples;
2. A condição para recomendação usa corretamente a conjunção entre score alto e não visualização recente;
3. A exclusão de itens já visualizados é uma implicação direta;
4. A redução de score baseada em feedback é uma relação causal válida.

A estrutura lógica representa adequadamente as regras condicionais do sistema de recomendação.

**Análise da Solidez**:

As premissas são sólidas no contexto de sistemas de recomendação modernos:

1. A utilização de histórico, preferências e comportamento de navegação é uma prática padrão;
2. O uso de scores de relevância é uma abordagem quantitativa comum;
3. A prevenção de recomendações repetitivas é uma boa prática de experiência do usuário;
4. A consideração de feedback negativo reflete sistemas adaptativos reais.

As conclusões derivadas são sólidas e representam um sistema de recomendação funcional que equilibra relevância, novidade e preferências do usuário.

#### Exercício 5: Sistema de Detecção de Fraudes

**Fragmento Texto Original**: "O sistema de detecção de fraudes deve analisar cada transação em tempo real. Uma transação é marcada como suspeita se atender a pelo menos um dos seguintes critérios: valor acima do padrão histórico do cliente, localização geográfica incomum, ou múltiplas tentativas em curto período de tempo. Se dois ou mais critérios forem atendidos simultaneamente, a transação é automaticamente bloqueada e enviada para revisão manual. Caso contrário, se apenas um critério for atendido, o cliente recebe uma notificação de confirmação. Se o cliente não confirmar em 5 minutos, a transação é bloqueada preventivamente."

**Solução**: A tarefa é analisar o fragmento de texto e formalizá-lo em lógica proposicional e lógica de predicados, identificando premissas, conclusões e avaliando a validade e solidez do argumento.

**Premissas**:

1. O sistema analisa cada transação em tempo real;
2. Critérios de suspeita: valor acima do padrão, localização incomum, múltiplas tentativas;
3. Uma transação é suspeita se atende a pelo menos um dos critérios;
4. Uma transação é automaticamente bloqueada se atende a dois ou mais critérios;
5. Se apenas um critério for atendido, o cliente recebe notificação para confirmação;
6. Se não houver confirmação em 5 minutos, a transação é bloqueada.

**Conclusões**:

- Se múltiplos critérios de suspeita são atendidos, a transação é bloqueada sem intervenção do cliente;
- Se um único critério é atendido, a transação depende de confirmação do cliente;
- Toda transação suspeita é ou bloqueada automaticamente ou requer confirmação.

**Formalização**

**Lógica Proposicional**:

- $R$: O sistema analisa transações em tempo real;
- $V$: A transação tem valor acima do padrão histórico;
- $L$: A transação ocorre em localização geográfica incomum;
- $M$: Há múltiplas tentativas em curto período;
- $S$: A transação é marcada como suspeita;
- $B$: A transação é bloqueada automaticamente;
- $N$: O cliente recebe notificação de confirmação;
- $C$: O cliente confirma a transação em 5 minutos;
- $P$: A transação é bloqueada preventivamente.

**Estrutura do argumento**:

1. $(V \lor L \lor M) \rightarrow S$;
2. $[(V \land L) \lor (V \land M) \lor (L \land M)] \rightarrow B$;
3. $[S \land \neg((V \land L) \lor (V \land M) \lor (L \land M))] \rightarrow N$;
4. $(N \land \neg C) \rightarrow P$.

**Lógica de Predicados**:

- $Transacao(t)$: $t$ é uma transação;
- $AnalisaTempoReal(t)$: a transação $t$ é analisada em tempo real;
- $ValorAlto(t)$: a transação $t$ tem valor acima do padrão histórico;
- $LocalizacaoIncomum(t)$: a transação $t$ ocorre em localização incomum;
- $MultiplasTentativas(t)$: há múltiplas tentativas para a transação $t$;
- $Suspeita(t)$: a transação $t$ é marcada como suspeita;
- $Bloqueada(t)$: a transação $t$ é bloqueada automaticamente;
- $EnviaNotificacao(t)$: uma notificação é enviada para confirmar $t$;
- $Confirma(t, m)$: a transação $t$ é confirmada dentro de m minutos;
- $BloqueioPreventivo(t)$: a transação $t$ recebe bloqueio preventivo.

**Formalização**:

1. $\forall t [Transacao(t) \rightarrow AnalisaTempoReal(t)]$;
2. $\forall t [Transacao(t) \land (ValorAlto(t) \lor LocalizacaoIncomum(t) \lor MultiplasTentativas(t)) \rightarrow Suspeita(t)]$;
3. $\forall t [Transacao(t) \land ((ValorAlto(t) \land LocalizacaoIncomum(t)) \lor (ValorAlto(t) \land MultiplasTentativas(t)) \lor (LocalizacaoIncomum(t) \land MultiplasTentativas(t))) \rightarrow Bloqueada(t)]$;
4. $\forall t [Transacao(t) \land Suspeita(t) \land \neg((ValorAlto(t) \land LocalizacaoIncomum(t)) \lor (ValorAlto(t) \land MultiplasTentativas(t)) \lor (LocalizacaoIncomum(t) \land MultiplasTentativas(t))) \rightarrow EnviaNotificacao(t)]$;
5. $\forall t [Transacao(t) \land EnviaNotificacao(t) \land \neg Confirma(t, 5) \rightarrow BloqueioPreventivo(t)]$.

**Análise da Validade**:

O argumento é logicamente válido. A estrutura representa corretamente o processo de decisão do sistema:

1. A definição de transação suspeita usa uma disjunção ($OR$) adequada;
2. A condição para bloqueio automático usa corretamente conjunções ($AND$) para representar a combinação de critérios;
3. A notificação em caso de suspeita única é representada por uma conjunção com uma negação de múltiplos critérios;
4. O bloqueio preventivo após falta de confirmação segue uma implicação lógica válida.

O sistema de regras é coerente, sem contradições ou ambiguidades lógicas.

**Análise da Solidez**:

As premissas são sólidas no contexto de sistemas de detecção de fraudes:

1. A análise em tempo real é essencial para sistemas antifraude eficazes;
2. Os critérios mencionados são indicadores comuns de atividades potencialmente fraudulentas;
3. A escalação baseada na quantidade de indicadores segue práticas reais de segurança;
4. O envolvimento do cliente para confirmação é uma prática que equilibra segurança e usabilidade;
5. O tempo limite para confirmação é uma medida preventiva razoável.

As conclusões derivadas são sólidas e representam um sistema de detecção de fraudes que equilibra detecção automática, envolvimento do cliente e proteção preventiva.

## Problemas Interessantes Resolvidos com Lógica Proposicional e Predicativa

Aqui estão cinco quebra-cabeças clássicos juntamente com suas soluções usando Lógica de Primeira Ordem

1. **Quebra-cabeça: O Mentiroso e o Verdadeiro**
   Você encontra dois habitantes: $A$ e $B$. Você sabe que um sempre diz a verdade e o outro sempre mente, mas você não sabe quem é quem. Você pergunta a $A$, _Você é o verdadeiro?_ A responde, mas você não consegue ouvir a resposta dele. $B$ então te diz, _A disse que ele é o mentiroso_.

   **Solução**: $A$ deve ser o verdadeiro e $B$ deve ser o mentiroso. Se $B$fosse o verdadeiro, ele nunca diria que é o mentiroso. Portanto, $B$ deve ser o mentiroso e $A$ deve ser o verdadeiro, independentemente do que $B$ disse.

   Usando apenas lógica proposicional teremos:

   **Definições**:
   VA: A é o verdadeiro
   MA: A é o mentiroso
   VB: B é o verdadeiro
   MB: B é o mentiroso
   RA: A respondeu "Sim" à pergunta "Você é o verdadeiro?"

   **Axiomas**:

   1. $VA \lor MA$ (A é verdadeiro ou mentiroso);
   2. $\neg(VA \land MA)$ (A não é ambos verdadeiro e mentiroso);
   3. $VB \lor MB$ (B é verdadeiro ou mentiroso);
   4. $\neg(VB \land MB)$ (B não é ambos verdadeiro e mentiroso);
   5. $VA \to \neg VB$ (Se A é verdadeiro, B não é verdadeiro);
   6. $VA \to RA$ (Se A é verdadeiro, ele respondeu "Sim");
   7. $MA \to \neg RA$ (Se A é mentiroso, ele respondeu "Não");
   8. $VB \to (B \text{ diz } \neg RA)$ (Se B é verdadeiro, ele diz a verdade sobre a resposta de A);
   9. $MB \to (B \text{ diz } RA)$ (Se B é mentiroso, ele mente sobre a resposta de A).

   **Fato observado**:

   $$B \text{ diz } \neg RA$$

   **Prova**:

   1. $B \text{ diz } \neg RA$ (Fato observado)
   2. $(VB \land \neg RA) \lor (MB \land RA)$ (Por 8, 9 e 1)
   3. Suponha $MA$:
      3.1. $\neg RA$ (Por 7)
      3.2. $VB$ (Por 3, 4 e 5)
      3.3. Mas isto contradiz 2, pois teríamos $(VB \land RA)$
   4. Portanto, $\neg MA$ (Por reductio ad absurdum)
   5. $VA$ (Por 1 e 4)

   Conclusão:

   $$VA \land \neg MA$$

   $A$ é o verdadeiro e não é o mentiroso.

   Usando lógica de primeiro grau teremos:

   **Definições**:

   1. $V(x)$: $x$ é o verdadeiro;

   2. $M(x)$: $x$ é o mentiroso;

   3. $R(x)$: $x$ respondeu "Sim" à pergunta "Você é o verdadeiro?";

   4. $D(x, p)$: $x$ diz que p é verdadeiro.

   **Axiomas**:

   1. $\forall x (V(x) \lor M(x))$ (Todo x é verdadeiro ou mentiroso);

   2. $\forall x (V(x) \to \neg M(x))$ (Ninguém é ambos verdadeiro e mentiroso);

   3. $\forall x (V(x) \to R(x))$ (Se x é verdadeiro, x responde "Sim");

   4. $\forall x (M(x) \to \neg R(x))$ (Se x é mentiroso, x responde "Não");

   5. $\forall x \forall y \forall p (V(x) \to (D(x, p) \leftrightarrow p))$ (Se x é verdadeiro, x diz p se e somente se p é verdadeiro);

   6. $\forall x \forall y \forall p (M(x) \to (D(x, p) \leftrightarrow \neg p))$ (Se x é mentiroso, x diz p se e somente se p é falso).

   **Fatos observados**:

   $$D(B, \neg R(A))$$

   **Prova**:

   1. $D(B, \neg R(A))$ (Fato observado);

   2. $V(A) \lor M(A)$ (Por 1);

   3. Suponha $M(A)$:
      3.1. $\neg R(A)$ (Por 4);
      3.2. $V(B)$ (Pois apenas um é mentiroso, por 1 e 2);
      3.3. $D(B, \neg R(A)) \leftrightarrow \neg R(A)$ (Por 5);
      3.4. $\neg R(A)$ (Por 1 e 3.3);
      3.5. Mas isto contradiz 3.1 e 3.4.

   4. Portanto, $\neg M(A)$ (Por reductio ad absurdum)

   5. $V(A)$ (Por 2 e 4)

   **Conclusão**:
   $$V(A) \land \neg M(A)$$

   $A$ é o verdadeiro e não é o mentiroso.

2. **Quebra-cabeça: As Três Lâmpadas**
   Existem três lâmpadas incandescentes em uma sala, e existem três interruptores fora da sala. Você pode manipular os interruptores o quanto quiser, mas só pode entrar na sala uma vez. Como você pode determinar qual interruptor opera qual lâmpada?

   **Solução**: ligue um interruptor e espere um pouco. Então desligue esse interruptor e ligue um segundo interruptor. Entre na sala. A lâmpada que está acesa corresponde ao segundo interruptor. A lâmpada que está desligada e quente corresponde ao primeiro interruptor. A lâmpada que está desligada e fria corresponde ao terceiro interruptor.

   Usando Lógica de Primeira Ordem:
   Vamos denotar os interruptores como $s1, s2, s3$ e as lâmpadas como $b1, b2, b3$. Podemos definir predicados $On(b, s)$ e $Hot(b)$.

   $$On(b1, s2) \land Hot(b2) \land \neg (On(b3) \lor Hot(b3))$$

3. **Quebra-cabeça: O Agricultor, a Raposa, o Ganso e o Grão**
   Um agricultor quer atravessar um rio e levar consigo uma raposa, um ganso e um saco de grãos. O barco do agricultor só lhe permite levar um item além dele mesmo. Se a raposa e o ganso estiverem sozinhos, a raposa comerá o ganso. Se o ganso e o grão estiverem sozinhos, o ganso comerá o grão. Como o agricultor pode levar todas as suas posses para o outro lado do rio?

   **Solução**: o agricultor leva o ganso através do rio primeiro, deixando a raposa e o grão no lado original. Ele deixa o ganso no outro lado e volta para pegar a raposa. Ele deixa a raposa no outro lado, mas leva o ganso de volta ao lado original para pegar o grão. Ele deixa o grão com a raposa no outro lado. Finalmente, ele retorna ao lado original mais uma vez para pegar o ganso.

   Usando Lógica de Primeira Ordem:
   Podemos definir predicados $mesmoLado(x, y)$ e $come (x, y)$.
   A solução envolve a sequência de ações que mantêm as seguintes condições:

   $$\neg (mesmoLado(Raposa, Ganso) \land \neg mesmoLado(Raposa, Fazendeiro))$$

   $$\neg (mesmoLado(Ganso, Grãos) \land \neg mesmoLado(Ganso, Fazendeiro))$$

4. **Quebra-cabeça: O Problema da Ponte e da Tocha**
   Quatro pessoas chegam a um rio à noite. Há uma ponte estreita, mas ela só pode conter duas pessoas de cada vez. Eles têm uma tocha e, por ser noite, a tocha tem que ser usada ao atravessar a ponte. A pessoa $A$ Pode atravessar a ponte em um minuto, $B$ em dois minutos, $C$ em cinco minutos e $D$ em oito minutos. Quando duas pessoas atravessam a ponte juntas, elas devem se mover no ritmo da pessoa mais lenta. Qual é a forma mais rápida para todos eles atravessarem a ponte?

   **Solução**: primeiro, $A$ e $B$Atravessam a ponte, o que leva 2 minutos. $A$ então pega a tocha e volta para o lado original, levando 1 minuto. $A$ fica no lado original enquanto $C$ e $D$Atravessam a ponte, levando 8 minutos. $B$ então pega a tocha e volta para o lado original, levando 2 minutos. Finalmente, $A$ e $B$Atravessam a ponte novamente, levando 2 minutos. No total, teremos $2+1+8+2+2=15$ minutos.

   Usando Lógica de Primeira Ordem:
   Vamos denotar o tempo que cada pessoa leva para atravessar a ponte como $t_A, T_B, T_C, T_D$ e o tempo total como $t$. O problema pode ser representado da seguinte forma:

   $$(T_A + T_B + T_A + T_C + T_D + T_B + T_A) \leq T$$

   Substituindo os valores dos tempos resulta em $15 \leq T$.

5. **Quebra-cabeça: O Problema de Monty Hall**
   Em um programa de game show, os concorrentes tentam adivinhar qual das três portas contém um prêmio valioso. Depois que um concorrente escolhe uma porta, o apresentador, que sabe o que está por trás de cada porta, abre uma das portas não escolhidas para revelar uma cabra, representando nenhum prêmio. O apresentador então pergunta ao concorrente se ele quer mudar sua escolha para a outra porta não aberta ou ficar com sua escolha inicial. O que o concorrente deve fazer para maximizar suas chances de ganhar o prêmio?

   **Solução**: o concorrente deve sempre mudar sua escolha. Inicialmente, a chance do prêmio estar atrás da porta escolhida é $1/3$ e a chance de estar atrás de uma das outras portas é $2/3$. Depois que o apresentador abre uma porta para revelar uma cabra, a chance do prêmio estar atrás da porta não escolhida e não aberta ainda é $2/3$.

   Usando Lógica de Primeira Ordem:
   Vamos denotar as portas como $d1, d2, d3$ e o prêmio como $P$. Podemos definir um predicado $contemPremio(d)$. A solução pode ser representada pela seguinte condição:

   $$(contemPremio(d1) \land \neg contemPremio(d2) \land \neg contemPremio(d3)) \\ \lor (contemPremio(d2)  \land \neg contemPremio(d1) \land \neg contemPremio(d3)) \\ \lor (contemPremio(d3) \land \neg contemPremio(d1) \land \neg contemPremio(d2))$$

   Esta condição afirma que o prêmio está exatamente atrás de uma das portas, e o concorrente deve mudar sua escolha depois que uma das portas é aberta para revelar nenhum prêmio.

### O Mistério da Mansão Hollow – Um Desafio para Detetives Lógicos**

**Objetivo**: Aplicar os princípios da lógica proposicional e de predicados para analisar um conjunto complexo de informações, identificar contradições, realizar deduções formais e solucionar um enigma.

**Descrição**:
A esforçada leitora foi convidada a investigar um intrigante mistério ocorrido na antiga Mansão Hollow. O renomado inventor, Sir Henry Clithering, desapareceu em circunstâncias suspeitas, deixando para trás uma série de pistas, depoimentos de funcionários e familiares, e alguns bilhetes enigmáticos. A polícia local está confusa com a quantidade de informações, algumas aparentemente contraditórias. Você deve montar uma equipe de detetives e superar o famoso detetive Hercule Poirot. Para isso deverá:

1. Analisar cuidadosamente todo o material fornecido (descrições de personagens, mapa da mansão, horários, depoimentos, bilhetes).

2. Formalizar as informações relevantes utilizando sentenças da lógica proposicional e, quando aplicável, da lógica de predicados.

3. Construir tabelas-verdade e/ou aplicar regras de inferência para verificar a consistência das informações e deduzir novos fatos.

4. Identificar o(s) responsável(is) pelo desaparecimento do Sr. Blackwood (ou determinar o que de fato aconteceu), justificando cada passo da sua conclusão com base nas deduções lógicas realizadas.

A seguir a descrição do caso, o mapa da mansão e os depoimentos dos envolvidos.

#### **O Mistério da Mansão Hollow: O Desaparecimento de Sir Henry Clithering**

**Data do Incidente**: Segunda-feira, 12 de Maio de 2025
**Local**: Mansão Hollow, uma propriedade rural isolada.
**Vítima (Desaparecido)**: Sir Henry Clithering, renomado inventor, 58 anos.

**1. Descrições dos Personagens**:

- **Sir Henry Clithering**: O inventor desaparecido. Gênio excêntrico e recluso, conhecido por sua mente brilhante e comportamento imprevisível. Estava trabalhando febrilmente em um novo projeto secreto chamado "Quimera".

- **Sra. Eleanor Clithering (50 anos)**: Esposa de Arthur. Uma mulher elegante e ambiciosa, visivelmente preocupada com a reputação e fortuna da família. Ela teme que o comportamento errático de Arthur possa arruiná-los.

- **Dr. Alistair Finch (45 anos)**: Um cientista brilhante, antigo protegido de Arthur, mas que se tornou seu principal rival acadêmico e comercial. Chegou à mansão no dia do desaparecimento, alegando buscar uma reconciliação e possível colaboração.

- **Miss Clara Evans (28 anos)**: A jovem e inteligente assistente pessoal de Arthur. Dedicada e leal, trabalhava em estreita colaboração com ele no projeto "Quimera" e conhecia muitos de seus segredos.

- **Sr. Reginald "Reggie" Croft (65 anos)**: O mordomo, trabalha para a família Blackwood há mais de trinta anos. É um homem discreto, observador e extremamente leal à memória do falecido pai de Arthur, mas demonstra certa reserva em relação ao próprio Arthur.

- **Sra. Beatrice Croft (62 anos)**: Esposa de Reggie, a cozinheira da mansão. Conhece todos os cantos da casa e os hábitos de seus ocupantes. É prática e não se deixa levar por fantasias.

**2. Mapa da Mansão Hollow (Descrição Textual)**:

A Mansão Hollow é uma construção vitoriana de dois andares, com um vasto terreno.

- **Térreo**:

  - **Hall de Entrada**: Amplo, com piso de mármore, uma imponente escadaria de carvalho que leva ao andar superior. Portas levam à biblioteca (esquerda), sala de estar (direita) e, ao fundo, um corredor para a sala de jantar e a ala de serviço/cozinha.
  
  - **Biblioteca**: Paredes forradas de estantes com livros antigos e científicos. Uma grande escrivaninha de mogno, poltronas de couro e uma lareira. Duas janelas altas com vista para o jardim da frente.
  
  - **Sala de Estar**: Mobiliário luxuoso, mas um pouco antiquado. Um piano de cauda, lareira e janelas com vista para o jardim lateral e o gazebo.
  
  - **Sala de Jantar**: Uma longa mesa de jantar polida, prataria reluzente. Acesso direto à cozinha.
  
  - **Cozinha**: Grande e funcional, com uma mesa rústica ao centro. Portas para a despensa, os aposentos dos Croft e uma saída para o jardim dos fundos/horta.
  
  - **Laboratório do Sr. Blackwood**: Localizado no final de um corredor isolado, partindo do hall, perto da escada de serviço. A porta possui uma fechadura especial de alta segurança projetada pelo próprio Arthur. O interior é um caos organizado de equipamentos eletrônicos, protótipos mecânicos, quadros com equações e ferramentas. Possui uma única janela reforçada que dá para o jardim dos fundos. Este é o local principal da investigação inicial.
  
  - **Escritório do Sr. Blackwood**: Uma sala menor, anexa ao laboratório, acessível apenas por uma porta dentro do laboratório. Mais organizada, com arquivos, patentes, um cofre e um computador.

- **Andar Superior**:
  
  - **Quarto Principal (Sr. e Sra. Blackwood)**: Espaçoso, com uma grande cama de dossel, penteadeira, armários embutidos e um banheiro privativo. Uma varanda com vista para o jardim da frente.
  
  - **Quarto de Hóspedes**: Onde Dr. Finch deixou seus pertences (embora não tenha passado a noite). Confortável, com uma cama de solteiro, escrivaninha e janela para o jardim lateral.
  
  - **Quarto de Clara Evans**: Menor e mais simples, localizado perto da escada de serviço, com vista para os fundos.
  
  - **Aposentos do Mordomo e da Cozinheira (Sr. e Sra. Croft)**: Localizados na ala de serviço, acima da cozinha.

- **Exterior**:

  - **Jardim da Frente**: Um gramado bem cuidado com um caminho circular de cascalho que leva à porta principal. Ladeado por sebes altas.
  
  - **Jardim Lateral**: Menos formal, com um gazebo antigo coberto de hera e canteiros de rosas.
  
  - **Jardim dos Fundos**: Uma área mais extensa e um pouco mais selvagem, com árvores antigas, uma pequena horta cultivada pela Sra. Croft e, nos limites da propriedade, uma velha estufa de vidro abandonada e parcialmente coberta por vegetação.

**3. Linha do Tempo (Segunda-feira, 12 de Maio de 2025)**:

- **08:00**: Café da manhã servido na sala de jantar. Sra. Blackwood preside. Sr. Blackwood não comparece, o que, segundo Sra. Blackwood, era comum quando ele estava imerso em trabalho.

- **09:00**: Clara Evans leva uma bandeja com café e torradas para o laboratório do Sr. Blackwood.

- **10:00**: Dr. Alistair Finch chega pontualmente à Mansão Hollow. É recebido pelo mordomo, Sr. Croft, e anunciado à Sra. Blackwood.

- **10:15 - 11:00 (aprox.)**: Dr. Finch e Sra. Blackwood conversam na sala de estar.

- **11:00**: Sra. Blackwood acompanha Dr. Finch até a porta do laboratório do Sr. Blackwood. Ela bate. Uma voz abafada, identificada por ela como sendo de Arthur, diz: "Estou no meio de algo crítico! Não me perturbem agora!". Dr. Finch parece contrariado.

- **11:05 - 13:00**: Período crucial com movimentações diversas e álibis a serem verificados.

- **13:00**: O almoço é servido. Sr. Blackwood novamente não aparece.

- **14:00**: Sra. Blackwood, demonstrando crescente preocupação, pede a Sr. Croft que vá verificar pessoalmente o Sr. Blackwood em seu laboratório.

- **14:05**: Sr. Croft dirige-se ao laboratório. Encontra a porta especial entreaberta. A fechadura de alta segurança parece ter sido arranhada (marcas de tentativa de arrombamento), mas está destrancada (possivelmente aberta corretamente após a tentativa de arrombamento). O interior do laboratório está em grande desordem: papéis e diagramas espalhados pelo chão, algumas ferramentas fora do lugar, uma cadeira virada. Sr. Blackwood não está em lugar nenhum. A janela dos fundos do laboratório está destrancada e aberta. Não há sinais óbvios de luta violenta (ex: sangue).

- **14:15**: Sra. Blackwood, após ser informada por Sr. Croft, instrui-o a chamar a polícia local.

- **17:00**: A notícia do desaparecimento e a natureza peculiar do caso chegam aos ouvidos de Hercule Poirot, que está concluindo um caso em uma cidade vizinha. Ele informa que só poderá dedicar-se ao mistério da Mansão Hollow na manhã seguinte. (Vocês têm até lá para resolver!)

**4. Depoimentos Iniciais (Coletados apressadamente pelo Sargento Davis, da polícia local)**:

- **Sra. Eleanor Clithering**:
  
  - "Arthur estava impossível nas últimas semanas, totalmente absorvido pelo tal projeto 'Quimera'. Falava coisas sem sentido sobre revolucionar o mundo, mas também sobre pessoas que queriam roubá-lo. Ele sempre foi um pouco... dramático."
  
  - "Quando bati à porta do laboratório às 11:00, ouvi claramente Arthur dizer para não ser perturbado. Sim, a voz parecia um pouco abafada, mas era ele. Dr. Finch estava ao meu lado."
  
  - "Depois disso, subi para meus aposentos para descansar e escrever algumas cartas. Não vi mais o Dr. Finch até a hora do almoço."
  
  - "A fechadura do laboratório é uma invenção do próprio Arthur. Apenas ele possuía a chave mestra. Ouvi dizer que Clara talvez soubesse algum truque para abri-la, mas forçá-la... faria um barulho terrível, não acha?"
  
  - "Desaparecer assim... não é do feitio de Arthur, a menos que seja parte de algum plano mirabolante dele. Ou então algo terrível aconteceu."

- **Dr. Alistair Finch**:
  
  - "Eu vim em uma missão de paz, acreditem. Nossas divergências passadas foram puramente intelectuais. Eu esperava que pudéssemos colaborar. A ideia de roubar o trabalho de Arthur é um insulto."
  
  - "Sim, a Sra. Blackwood me acompanhou até a porta do laboratório. Ouvi uma voz masculina dizer para não sermos inoportunos. Não posso jurar que era Arthur, a voz estava abafada, como disse a Sra. Blackwood."
  
  - "Após a recusa, senti-me um pouco desconfortável. Decidi caminhar pelos jardins para espairecer, entre aproximadamente 11:05 e 12:45. Andei pela frente da casa e também pelo jardim lateral, perto do gazebo. O tempo estava agradável."
  
  - "Não vi ninguém suspeito. Vi o mordomo, Sr. Croft, por um instante, perto da entrada lateral da casa, por volta das 11:20. Ele parecia estar carregando uma caixa ou algo similar em direção à parte de trás da casa ou à adega."
  
  - "Eu nunca tocaria na fechadura do laboratório de Arthur sem permissão. Seria uma violação imperdoável da ética científica."

- **Miss Clara Evans**:
  
  - "Sr. Blackwood estava muito pressionado, mas também excitado com o 'Quimera'. Ele dizia que mudaria tudo. Ele confiava em mim implicitamente."
  
  - "Sim, ele temia que o Dr. Finch, ou outros, pudessem tentar se apropriar de suas descobertas. Ele tomava muitas precauções."
  
  - "Quando levei seu café às 09:00, ele estava um pouco agitado, mas lúcido. Disse-me: 'Clara, hoje é um dia de grandes decisões. Lembre-se dos nossos protocolos.'"
  
  - "Entre 11:00 e 13:00, estive principalmente no escritório anexo ao laboratório, compilando dados. A porta entre o escritório e o laboratório estava fechada na maior parte do tempo para que ele tivesse silêncio. Saí brevemente, por volta das 11:30, para ir à biblioteca buscar o 'Compêndio de Ligas Metálicas Raras'. Fiquei lá uns 15, talvez 20 minutos. Não cruzei com ninguém no corredor ou na biblioteca."
  
  - "O laboratório tem um bom isolamento acústico, especialmente com a porta do escritório fechada. Não ouvi nenhum barulho de arrombamento. A fechadura especial é complexa; apenas Sr. Blackwood tinha a chave. Eu conheço o procedimento de abertura manual de emergência, mas é uma sequência demorada e específica."

- **Sr. Reginald "Reggie" Croft (Mordomo)**:
  
  - "Dr. Finch chegou às 10:00. Parecia um pouco nervoso, na minha opinião. Ele e a patroa conversaram na sala de estar por um bom tempo."
  
  - "Por volta das 11:00, eu estava no hall polindo a prata, e ouvi as vozes da Sra. Blackwood e do Dr. Finch perto do corredor do laboratório. Não prestei muita atenção ao que foi dito. Logo depois, vi a Sra. Blackwood subir a escadaria principal."
  
  - "De fato, por volta das 11:15, eu estava transportando uma caixa de garrafas de vinho da entrada de serviço lateral para a adega no porão. Nesse momento, vi o Dr. Finch caminhando pelo jardim da frente, perto do portão principal. Ele olhava muito para o relógio."
  
  - "Quando fui chamado pela Sra. Blackwood às 14:00, encontrei a porta do laboratório como descrito: entreaberta, com arranhões na fechadura, mas destrancada. O Sr. Blackwood era metódico. Se ele não queria ser perturbado, ele trancava a porta de uma forma que ninguém entraria."

- **Sra. Beatrice Croft (Cozinheira)**:
  
  - "Da cozinha, não se ouve muito do resto da casa, a menos que seja uma gritaria. Estive ocupada com o almoço toda a manhã."
  
  - "Sr. Blackwood não aparecer para as refeições não era novidade quando estava às voltas com suas invenções malucas."
  
  - "Uma coisa estranha: Miss Evans passou rapidamente pela cozinha por volta das 12:50. Parecia muito pálida e apressada. Perguntei se estava tudo bem, e ela murmurou algo sobre ir verificar se o Sr. Blackwood queria que o almoço fosse servido no laboratório. Ela voltou alguns minutos depois, ainda mais pálida, e disse que ele não tinha respondido aos chamados dela na porta do laboratório e que era melhor não insistir. Achei estranho ela não ter comentado isso com a Sra. Blackwood imediatamente, antes do alarme oficial."
  
  - "A janela do laboratório? Sim, dá para uma parte mais isolada d-o jardim dos fundos, perto da minha horta. Se alguém pulou por ali, e se esgueirou pelas árvores, poderia sumir sem ser visto da casa principal."

**5. Pistas e Bilhetes Enigmáticos**:

- **Pista 1: Papel Amassado na Lixeira do Laboratório**:
  Um pequeno pedaço de papel de anotações, claramente arrancado de um bloco maior, contém a seguinte mensagem escrita à mão por Sr. Blackwood (caligrafia confirmada):
  
  > "Se A implica B, e o Corvo visita o Ninho, então a Hipótese se confirma. A negação do consequente é o único caminho seguro. Sigma Ativado."

- **Pista 2: Objeto Encontrado no Chão do Laboratório, Perto da Mesa Principal**:
  Um pequeno e incomum botão de metal fosco, com um desenho de uma engrenagem estilizada. Não parece pertencer a nenhuma roupa do Sr. Blackwood, nem faz parte do vestuário usual dos funcionários.

- **Pista 3: Anotação na Margem de um Livro na Biblioteca**:
  No livro "Compêndio de Ligas Metálicas Raras" (o mesmo que Clara Evans mencionou ter pego), na página sobre o Bismuto, há uma pequena anotação a lápis, quase imperceptível:

   > "Onde o passado encontra o futuro, a reflexão é a chave. $(\neg P \lor Q)$ é equivalente a ?"
    A caligrafia parece ser de Sr. Blackwood.

- **Pista 4: Marca Estranha no Batente da Janela Aberta do Laboratório**:
  Do lado de fora do batente da janela do laboratório, há uma leve marca de fuligem ou graxa escura, como se algo metálico e sujo tivesse sido apoiado ali brevemente.

- **Pista 5: Na Estufa Abandonada (Jardim dos Fundos)**:
  Dentro da estufa, sobre uma bancada empoeirada, alguém desenhou com o dedo na poeira um símbolo: um triângulo equilátero com um pequeno círculo no centro. Ao lado do desenho, um único fósforo queimado. Não há outras pegadas recentes visíveis devido ao solo irregular e coberto de folhas secas.

## Formas Normais

As formas normais, em sua essência, são um meio de trazer ordem e consistência à forma como representamos proposições na Lógica Proposicional. Elas oferecem uma estrutura formalizada para expressar proposições, uma convenção que simplifica a comparação, análise, entendimento e simplificação de proposições lógicas.

Consideremos, por exemplo, a tarefa de comparar duas proposições para determinar se são equivalentes. Sem uma forma padronizada de representar proposições, essa tarefa pode se tornar complexa e demorada. No entanto, ao utilizar as formas normais, cada proposição é expressa de uma forma padrão, tornando a comparação direta e simples. Além disso, as formas normais também desempenham um papel importante na simplificação de proposições. Ao expressar uma proposição em sua forma normal, é mais fácil identificar oportunidades de simplificação, removendo redundâncias ou simplificando a estrutura lógica. As formas normais não são apenas uma ferramenta para lidar com a complexidade da Lógica Proposicional, mas também uma metodologia que facilita a compreensão e manipulação de proposições lógicas.

Existem várias formas normais na Lógica Proposicional, cada uma com suas próprias regras e aplicações. Aqui estão algumas das principais:

1. **Forma Normal Negativa (PNN)**: Uma proposição está na Forma Normal Negativa se as operações de negação $\neg $Aparecerem apenas imediatamente antes das variáveis. Isso é conseguido aplicando as leis de De Morgan e eliminando as duplas negações.

   $$\neg (A \wedge B) \equiv (\neg A \vee \neg B)$$

2. **Forma Normal Conjuntiva (PNC)**: Uma proposição está na Forma Normal Conjuntiva se for uma conjunção, operação _E_, $\wedge $, de uma ou mais cláusulas, onde cada cláusula é uma disjunção, operação _OU_, $\vee $, de literais. Em outras palavras, é uma série de cláusulas conectadas por _Es_, onde cada cláusula é composta de variáveis conectadas por _OUs_.

   $$(A \vee B) \wedge (C \vee D) \equiv (A \wedge C) \vee (A \wedge D) \vee (B \wedge C) \vee (B \wedge D)$$

3. **Forma Normal Disjuntiva (PND)**: uma proposição está na Forma Normal Disjuntiva se for uma disjunção de uma ou mais cláusulas, onde cada cláusula é uma conjunção de literais. Ou seja, é uma série de cláusulas conectadas por **ORs**, onde cada cláusula é composta de variáveis conectadas por **ANDs**.

   $$(A \wedge B) \vee (C \wedge D) \equiv (A \vee C) \wedge (A \vee D) \wedge (B \vee C) \wedge (B \vee D)$$

4. **Forma Normal Prenex (PNP)**: uma proposição está na Forma Normal Prenex se todos os quantificadores, para a Lógica de Primeira Ordem, estiverem à esquerda, precedendo uma matriz quantificadora livre. Esta forma é útil na Lógica de Primeira Ordem e na teoria da prova.

   $$\exists x \forall y (P(x,y) \wedge Q(y)) \equiv \forall y \exists x (P(x,y) \wedge Q(y))$$

5. **Forma Normal Skolem (PNS)**: na Lógica de Primeira Ordem, uma fórmula está na Forma Normal de Skolem se estiver na Forma Normal Prenex e se todos os quantificadores existenciais forem eliminados. Isto é realizado através de um processo conhecido como Skolemização.

   $$\forall x (P(x,y)) \equiv P(x, f(x))$$

Nosso objetivo é rever a matemática que suporta a Programação Lógica, entre as principais formas normais, para este objetivo, precisamos destacar duas formas normais:

1. **Forma Normal Conjuntiva (FNC)**: a Forma Normal Conjuntiva é importante na Programação Lógica porque muitos sistemas de inferência, como a resolução, funcionam em fórmulas que estão na FNC. Além disso, os programas em Prolog, A linguagem de Programação Lógica que escolhemos, são essencialmente cláusulas na FNC.

2. **Forma Normal de Skolem (FNS)**: a Forma Normal de Skolem é útil na Programação Lógica porque a Skolemização, o processo de remover quantificadores existenciais transformando-os em funções de quantificadores universais, permite uma forma mais eficiente de representação e processamento de fórmulas lógicas. Essa forma normal é frequentemente usada em Lógica de Primeira Ordem e teoria da prova, ambas fundamentais para a Programação Lógica.

Embora outras formas normais possam ter aplicações em áreas específicas da Programação Lógica, a FNC e a FNS são provavelmente as mais amplamente aplicáveis e úteis nesse Proposição. Começando com a Forma Normal Conjuntiva.

Se considerarmos as propriedades associativas apresentadas nas linhas 20 e 21 da Tabela 2, podemos escrever uma sequência de conjunções, ou disjunções, sem precisarmos de parênteses. Sendo assim, teremos:

$$((P \wedge (Q \wedge R)) \wedge S)$$

Pode ser escrita como:

$$P\wedge Q \wedge R \wedge s $$

### Forma Normal Negativa (FNN)

A Forma Normal Negativa é uma representação canônica de fórmulas lógicas em que as negações são aplicadas apenas aos átomos da fórmula e não a expressões mais complexas. Em outras palavras, a negação está _empurrada para dentro_ o máximo possível. A FNN é útil por sua simplicidade e é frequentemente um passo intermediário na conversão para outras formas normais.

Uma fórmula está na Forma Normal Negativa se:

- todos os operadores de negação $\neg $ são aplicados diretamente aos átomos, variáveis ou constantes.
- usaremos apenas a negação $\neg $, a conjunção $\land $, e a disjunção $\lor $.

Converter uma fórmula para a FNN envolve os seguintes passos:

1. **Eliminar os Bicondicionais**: substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
2. **Eliminar Implicações**: substitua todas as ocorrências de implicação $A \rightarrow B$ Por $\neg A \lor B$.
3. **Aplicar as Leis de De Morgan**: Use as leis de De Morgan para mover as negações para dentro, aplicando:
   - $\neg (A \land B) \rightarrow \neg A \lor \neg B$
   - $\neg (A \lor B) \rightarrow \neg A \land \neg B$
4. **Eliminar Dupla Negação**: Substitua qualquer dupla negação $\neg \neg A$ Por $A$.

#### Exemplo 1: Converta a fórmula $\neg (A \land (B \rightarrow C))$ Para FNN

1. Eliminar Implicações: $\neg (A \land (\neg B \lor C))$
2. Aplicar De Morgan: $\neg A \lor (B \land \neg C)$
3. Eliminar Dupla Negação: $\neg A \lor (B \land \neg C)$(já está na FNN)

#### Exemplo 2: Converta a fórmula $(A \rightarrow B) \land \neg (C \lor D)$ Para FNN

1. Eliminar Implicações: $(\neg A \lor B) \land \neg (C \lor D)$;
2. Aplicar De Morgan: $(\neg A \lor B) \land (\neg C \land \neg D)$;
3. Eliminar Dupla Negação: $(\neg A \lor B) \land (\neg C \land \neg D)$ (já está na FNN).

### Forma Normal Disjuntiva (FND)

A Forma Normal Disjuntiva é uma representação canônica de fórmulas lógicas em que a fórmula é escrita como uma disjunção de conjunções. Trata-se uma forma canônica útil para a análise e manipulação de fórmulas lógicas e é comumente usada em algoritmos de raciocínio lógico.

Uma fórmula está na Forma Normal Disjuntiva se puder ser escrita como:

$$(C_1 \land C_2 \land \ldots) \lor (D_1 \land D_2 \land \ldots) \lor$$

Na qual, cada $C_i$ e $D_i$ é um literal. Ou seja, é uma variável ou sua negação. Com um pouco mais de formalidade matemática podemos afirmar que uma Fórmula Bem Formada está na Forma Normal Disjuntiva quando está na forma:

$$\bigvee_{i=1}^{m} \left( \bigwedge_{j=1}^{n} L_{ij} \right)$$

Converter uma fórmula para a FND geralmente envolve os seguintes passos:

1. **Eliminar os Bicondicionais**: substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
2. **Eliminar Implicações**: substitua todas as ocorrências de implicação $A \rightarrow B$ Por $\neg A \lor B$.
3. **Aplicar as Leis de De Morgan**: use as leis de De Morgan para mover as negações para dentro, aplicando:

   - $\neg (A \land B) \rightarrow \neg A \lor \neg B$
   - $\neg (A \lor B) \rightarrow \neg A \land \neg B$

4. **Eliminar Dupla Negação**: Substitua qualquer dupla negação $\neg \neg A$ Por $A$.
5. **Aplicar a Lei Distributiva**: Use a lei distributiva para expandir a fórmula, transformando-a em uma disjunção de conjunções.

#### Exemplo 1

   $$(A \rightarrow B) \land (C \lor \neg (D \land E))$$

1. Eliminar Implicações

   $$(A \rightarrow B) \land (C \lor \neg (D \land E)) \rightarrow (\neg A \lor B) \land (C \lor \neg (D \land E))$$

2. Aplicar De Morgan

   $$(\neg A \lor B) \land (C \lor \neg D \lor \neg E)$$

3. Distribuir a Disjunção

   $$(\neg A \lor B) \land C \lor (\neg A \lor B) \land \neg D \lor (\neg A \lor B) \land \neg E$$

#### Exemplo 2

$$(\neg A \land (B \rightarrow C)) \lor (D \land \neg (E \rightarrow F))$$

1. Eliminar Implicações

   $$(\neg A \land (\neg B \lor C)) \lor (D \land \neg (\neg E \lor F)) \rightarrow (\neg A \land (\neg B \lor C)) \lor (D \land (E \land \neg F))$$

2. Distribuir a Disjunção

   $$(\neg A \land \neg B \lor \neg A \land C) \lor (D \land E \land \neg F)$$

3. Distribuir a Disjunção Novamente

   $$\neg A \land \neg B \lor \neg A \land C \lor D \land E \land \neg F$$

#### Exemplo 3

$$(p \rightarrow q) \rightarrow (r \vee s)$$

1. Remover as implicações ($\rightarrow$):

   $$p \rightarrow q \equiv \neg p \vee q$$

2. Substituir a expressão original com a equivalência encontrada no passo 1:

   $$(\neg p \vee q) \rightarrow (r \vee s)$$

3. Aplicar novamente a equivalência para remover a implicação:

   $$\neg (\neg p \vee q) \vee (r \vee s)$$

4. Aplicar a lei de De Morgan para expandir a negação:

   $$(p \wedge \neg q) \vee (r \vee s)$$

#### Exemplo 4

$$(p \rightarrow q) \rightarrow (\neg r \vee s)$$

1. Primeiro, vamos eliminar as implicações, usando a equivalência $p \rightarrow q \equiv \neg p \vee q$:

   $$(p \rightarrow q) \rightarrow (\neg r \vee s)$$

   Substituindo a implicação interna, temos:

   $$(\neg p \vee q) \rightarrow (\neg r \vee s)$$

2. Agora, vamos eliminar a implicação externa, usando a mesma equivalência:

   $$\neg (\neg p \vee q) \vee (\neg r \vee s)$$

3. Em seguida, aplicamos a lei de De Morgan para expandir a negação:

   $$(p \wedge \neg q) \vee (\neg r \vee s)$$

#### Exemplo 5

$$\neg(p \land q) \rightarrow (r \leftrightarrow s)$$

$$\begin{align*}
\quad 1. & \quad \neg(p \land q) \rightarrow (r \leftrightarrow s) \\
\quad 2. & \quad \neg(p \land q) \rightarrow ((r \rightarrow s) \land (s \rightarrow r)) \, \text{ (Substituindo a equivalência por suas implicações)} \\
\quad 3. & \quad \neg(p \land q) \rightarrow ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Convertendo as implicações em disjunções)} \\
\quad 4. & \quad (\neg (p \land q)) \lor ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Aplicando a equivalência } p \rightarrow q \equiv \neg p \lor q \text{)} \\
\quad 5. & \quad (\neg p \lor \neg q) \lor ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Aplicando a De Morgan em } \neg(p \land q) \text{)} \\
\quad 6. & \quad (\neg p \lor \neg q \lor \neg r \lor s) \land (\neg p \lor \neg q \lor \neg s \lor r) \, \text{ (Aplicando a distributividade para obter a FND)}
\end{align*}$$

A Forma Normal Disjuntiva é útil porque qualquer fórmula lógica pode ser representada desta forma, e a representação é única (à exceção da ordem dos literais e cláusulas).

### Forma Normal Conjuntiva (FNC)

A Forma Normal Conjuntiva é uma representação canônica de fórmulas lógicas em que a fórmula é escrita como uma conjunção de disjunções. Em outras palavras, é uma expressão lógica na forma de uma _conjunção de disjunções_. É uma forma canônica útil para a análise e manipulação de fórmulas lógicas e é comumente usada em algoritmos de raciocínio lógico e simplificação de fórmulas.

Uma fórmula está na Forma Normal Conjuntiva se puder ser expressa na forma:

$$(D_1 \lor D_2 \lor \ldots \lor D_n) \land (E_1 \lor E_2 \lor \ldots \lor E_m) \land \ldots$$

Na qual, $D_1, \ldots , D_n$ e $ e_1, \ldots ,E_n $ representam átomos. Podemos dizer que a Forma Normal Conjuntiva acontece quando a Fórmula Bem Formada está na forma:

$$\bigwedge_{i=1}^{m} \left( \bigvee_{j=1}^{n} L_{ij} \right)$$

Converter uma fórmula para a Forma Normal Conjuntiva, já incluindo os conceitos de Skolemização, envolve os seguintes passos:

1. **Eliminar os Bicondicionais**: substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
2. **Eliminar Implicações**: substitua todas as ocorrências de implicação $A \rightarrow B$ Por $\neg A \lor B$.
3. **Colocar a Negação no Interior dos parênteses**: Use as leis de De Morgan para mover as negações para dentro, aplicando:

   - $\neg (\forall x A) \equiv \exists x \neg A$;
   - $\neg (\exists x A) \equiv \forall x \neg A$;
   - $\neg (A \land B) \rightarrow \neg A \lor \neg B$;
   - $\neg (A \lor B) \rightarrow \neg A \land \neg B$.

4. **Eliminar Dupla Negação**: Substitua qualquer dupla negação $\neg \neg A$ Por $A$.
5. **Skolemização**: todas as variáveis existenciais será substituída por uma Constante de Skolem, ou uma Função de Skolem das variáveis universais relacionadas.

   - $\exists x Bonito(x)$ será transformado em $Bonito(g1)$ onde $g1$ é uma Constante de Skolem;
   - $\forall x Pessoa(x) \rightarrow Coração(x) \wedge Feliz(x,y)$ se torna $\forall x Pessoa(x) \rightarrow Coração(H(x))\wedge Feliz(x,H(x))$, onde $H$ é uma função de Skolem.

6. Remova todos os Quantificadores Universais. $\forall x Pessoa(x)$ se torna $Pessoa(x)$.

7. **Aplicar a Lei Distributiva**: Use a lei distributiva para expandir a fórmula, transformando-a em uma conjunção de disjunções. Substituindo $\wedge$ por $\vee$.

#### Exemplo 1 

$$(A \land B) \rightarrow (C \lor D)$$

1. Eliminar Implicações\*:

   $$\neg (A \land B) \lor (C \lor D) \rightarrow (\neg A \lor \neg B) \lor (C \lor D)$$

2. Distribuir a Disjunção:

   $$(\neg A \lor \neg B \lor C \lor D)$$

#### Exemplo 2

$$(A \land \neg B) \lor (\neg C \land D) \rightarrow (E \lor F)$$

1. Eliminar Implicações:

   $$\neg ((A \land \neg B) \lor (\neg C \land D)) \lor (E \lor F) \rightarrow \neg (A \land \neg B) \land \neg (\neg C \land D) \lor (E \lor F)$$

2. Aplicar De Morgan:

   $$(\neg A \lor B) \land (C \lor \neg D) \lor (E \lor F)$$

3. Distribuir a Disjunção:

   $$(\neg A \lor B \lor E \lor F) \land (C \lor \neg D \lor E \lor F)$$

#### Exemplo 3

$$(p \wedge (q \vee r)) \vee (\neg p \wedge \neg q)$$

1. Aplicar a lei distributiva ($\wedge$ sobre $\vee$) no primeiro termo para obter uma Forma Normal Disjuntiva (FND) da expressão:

    $$(p \wedge q) \vee (p \wedge r) \vee (\neg p \wedge \neg q)$$

2. Aplicar a lei distributiva ($\vee$ sobre $\wedge$) para transformar a FND em uma conjunção de disjunções.

    A expressão $( (p \wedge q) \vee (p \wedge r) ) \vee (\neg p \wedge \neg q)$ pode ser reescrita como:

    $$((p \wedge q) \vee (p \wedge r) \vee \neg p) \wedge ((p \wedge q) \vee (p \wedge r) \vee \neg q)$$

    Simplificando cada um dos termos principais da conjunção:

    - O primeiro termo, $((p \wedge q) \vee (p \wedge r) \vee \neg p)$, simplifica para:

        $$(\neg p \vee q \vee r)$$

    - O segundo termo, $((p \wedge q) \vee (p \wedge r) \vee \neg q)$, simplifica para:

        $$(p \vee \neg q)$$

3. Finalmente a Forma Normal Conjuntiva (FNC):

    $$(\neg p \vee q \vee r) \wedge (p \vee \neg q)$$

#### Exemplo 4

$$ \neg ((p \wedge q) \vee \neg (r \wedge s))$$

1. Aplicando a Lei de De Morgan na expressão inteira:

   $$
   \begin{align*}
   \neg ((p \wedge q) \vee \neg (r \wedge s)) &\equiv \neg (p \wedge q) \wedge (r \wedge s) \quad \text{(Lei de De Morgan)}
   \end{align*}
   $$

2. aplicando a Lei de De Morgan nos termos internos:

   $$
   \begin{align*}
   \neg (p \wedge q) \wedge (r \wedge s) &\equiv (\neg p \vee \neg q) \wedge (r \wedge s) \quad \text{(Lei de De Morgan)}
   \end{align*}
   $$

#### Exemplo 5

$$\neg (((p \rightarrow q) \rightarrow p) \rightarrow p)$$

1. Eliminar Implicações. Utilizando a equivalência $p \rightarrow q \equiv \neg p \lor q $:

   $$\neg(\neg(\neg p \lor q)\lor p)\lor p$$

2. Aplicar Leis de De Morgan:

   $$((p \land \neg q) \lor p) \land \neg p$$

3. Simplificamos a expressão usando propriedades como $p \lor p \equiv p$ e, em seguida, redistribuímos os termos para alcançar a Forma Normal Conjuntiva:

   $$(p \land (\neg q \lor p)) \land \neg p$$

4. Aplicamos as propriedades comutativa e associativa para organizar os termos de uma forma mais apresentável:

   $$(\neg q \lor p) \land (p \land \neg p)$$

5. Identificamos e simplificamos contradições na expressão usando $p \land \neg p \equiv \bot$, levando a:

   $$(\neg q \lor p) \land \bot$$

6. Por último, aplicamos a identidade com a contradição $$\bot \land p \equiv \bot$ para obter a expressão final:

   $$\bot \text{False}$$

#### Exemplo 6

$$(p \rightarrow q) \leftrightarrow (p \rightarrow r)$$

1. Começamos pela definição de equivalência e implicação:

   $$(p \rightarrow q) \leftrightarrow (p \rightarrow r)$$

2. Aplicamos as definições de implicação:

   $$(\neg p \lor q) \leftrightarrow (\neg p \lor r)$$

3. Agora, aplicamos a definição de equivalência, transformando-a em uma conjunção de duas implicações:

   $$((\neg p \lor q) \rightarrow (\neg p \lor r)) \land ((\neg p \lor r) \rightarrow (\neg p \lor q))$$

4. Em seguida, aplicamos a definição de implicação novamente para cada uma das implicações internas:

   $$(\neg (\neg p \lor q) \lor (\neg p \lor r)) \land (\neg (\neg p \lor r) \lor (\neg p \lor q))$$

5. Vamos aplicar a lei de De Morgan e a lei da dupla negação para simplificar a expressão:

   $$((p \land \neg q) \lor (\neg p \lor r)) \land ((p \land \neg r) \lor (\neg p \lor q))$$

6. Aplicando a lei distributiva para desenvolver cada conjunção interna em disjunções:

   $$((p \lor (\neg p \lor r)) \land (\neg q \lor (\neg p \lor r))) \land ((p \lor (\neg p \lor q)) \land (\neg r \lor (\neg p \lor q)))$$

A aplicação das equivalências não é, nem de longe, a única forma de percorrer a rota da conversão de uma Fórmula Bem Formada em Forma Normal Conjuntiva.

### Forma Normal Prenex

A Forma Normal Prenex é uma padronização para fórmulas da lógica de primeiro grau. Nela, todos os quantificadores são deslocados para a frente da fórmula, deixando a matriz da fórmula livre de quantificadores. A Forma Normal Prenex é vantajosa por três razões fundamentais:

1. **Facilitação da Manipulação Lógica**: ao separar os quantificadores da matriz, a Forma Normal Prenex simplifica a análise e manipulação da estrutura lógica da fórmula;

2. **Preparação para Outras Formas Normais**: Serve como uma etapa intermediária valiosa na conversão para outras formas normais, como as Forma Normal Conjuntiva e Forma Normal Disjuntiva;

3. **Uso em Provas Automáticas**: é amplamente empregada em métodos de prova automática, tornando o raciocínio sobre quantificadores mais acessível.

Considere o seguinte exemplo, partindo da fórmula original: $\exists x \forall y (P(x,y) \wedge Q(y))$

Na Forma Prenex, esta fórmula será representada:

$$\forall y \exists x (P(x,y) \wedge Q(y))$$

Uma fórmula na Forma Normal Prenex segue uma estrutura específica definida por:

$$Q_1 x_1 \, Q_2 x_2 \, \ldots \, Q_n x_n \, M(x_1, x_2, \ldots, x_n)$$

Nessa estrutura:

- $Q_i$ são quantificadores, podendo ser universais $\forall$ ou existenciais $\exists$;
- $x_i$ são as variáveis vinculadas pelos quantificadores;
- $M(x_1, x_2, \ldots, x_n)$ representa a matriz da fórmula, uma expressão lógica sem quantificadores.

Converter uma fórmula para a Forma Normal Prenex envolve os seguintes passos:

1. **Eliminar Implicações**: substitua todas as ocorrências de implicação por disjunções e negações;

2. **Mover Negações para Dentro**: use as leis de De Morgan para mover as negações para dentro dos quantificadores e proposições;

3. **Padronizar Variáveis**: certifique-se de que as variáveis ligadas a diferentes quantificadores sejam distintas;

4. **Eliminar Quantificadores Existenciais**: substitua os quantificadores existenciais por constantes ou funções Skolem, dependendo do contexto;

5. **Mover Quantificadores para Fora**: mova todos os quantificadores para a esquerda da expressão, mantendo a ordem relativa dos quantificadores universais e existenciais.

A Forma Normal Prenex é uma representação canônica de fórmulas da lógica de primeiro grau que separa claramente os quantificadores da matriz da fórmula. Ela é uma ferramenta valiosa na lógica e na teoria da prova, e sua compreensão é fundamental para trabalhar com lógica de primeiro grau.

#### Regras de Equivalência Prenex

A Forma Prenex de uma fórmula lógica com quantificadores permite mover todos os quantificadores para o início da fórmula. Existem algumas regras de equivalência que preservam a Forma Prenex quando aplicadas a uma fórmula:

**1. Comutatividade de quantificadores do mesmo tipo**: a ordem dos quantificadores do mesmo tipo pode ser trocada em uma fórmula na Forma Prenex. Por exemplo:

$$\forall x \forall y \ P(x,y) \Leftrightarrow \forall y \forall x \ P(x,y)$$

Isso ocorre porque a ordem dos quantificadores universais $\forall x$ e $\forall y$ não altera o significado lógico da fórmula. Essa propriedade é conhecida como comutatividade dos quantificadores.

**2. Associatividade de quantificadores do mesmo tipo**: quantificadores do mesmo tipo podem ser agrupados de forma associativa em uma Forma Prenex. Por exemplo:

$$\forall x \forall y \forall z \ P(x,y,z) \Leftrightarrow \forall x (\forall y \forall z \ P(x,y,z))$$

Novamente, o agrupamento dos quantificadores universais não muda o significado da fórmula. Essa é a propriedade associativa.

**3. Distributividade de quantificadores sobre operadores lógicos**: os quantificadores podem ser distribuídos sobre operadores lógicos como $\wedge, \vee, \rightarrow$:

$$\forall x (P(x) \vee Q(x)) \Leftrightarrow (\forall x \ P(x)) \vee (\forall x \ Q(x))$$

Isso permite mover o quantificador para dentro do escopo do operador lógico. A equivalência se mantém pois a ordem de quantificação e operação não se altera.

#### Conversão para Formas Normais Conjuntiva (FNC) e Disjuntiva (FND)

**1. Eliminar Implicações**: substitua todas as ocorrências de implicação da forma $A \rightarrow B$ Por $\neg A \lor B$.

**2. Mover a Negação para Dentro**: use as leis de De Morgan para mover a negação para dentro dos quantificadores e das proposições. Aplique as seguintes transformações:

- $\neg \forall x P(x) \rightarrow \exists x \neg P(x)$
- $\neg \exists x P(x) \rightarrow \forall x \neg P(x)$

**3. Padronizar Variáveis**: certifique-se de que as variáveis ligadas a diferentes quantificadores sejam distintas, renomeando-as se necessário.

**4. Eliminar os Quantificadores Existenciais**: substitua cada quantificador existencial $\exists x$ Por um novo termo constante ou Função Skolem, dependendo das variáveis livres em seu escopo. Para eliminar os quantificadores existenciais, é necessário introduzir novos termos: Constantes ou Funções Skolem.

1. **Se o quantificador existencial não tem quantificadores universais à sua esquerda**:
   Substitua $\exists x P(x)$ Por $P(c)$, onde $c$ é uma nova constante.

2. **Se o quantificador existencial tem quantificadores universais à sua esquerda**:
   Substitua $\exists x P(x)$ Por $P(f(y_1, y_2, \ldots, y_n))$, onde $f$ é uma nova função Skolem, e $y_1, y_2, \ldots, y_n$ são as variáveis universais à esquerda do quantificador existencial.

**5. Mover os Quantificadores Universais para Fora**: mova todos os quantificadores universais para fora, para a esquerda da expressão. Isso cria uma Forma Prenex da fórmula.

**6. Eliminar os Quantificadores Universais**: remova os quantificadores universais, deixando apenas a matriz da fórmula. Isso resulta em uma fórmula livre de quantificadores. Após a eliminação dos quantificadores existenciais e a movimentação de todos os quantificadores universais para fora (Forma Prenex), a eliminação dos quantificadores universais é simples:

1. **Remova os quantificadores universais da fórmula**:
   Se você tem uma fórmula da forma $\forall x P(x)$, simplesmente remova o quantificador $\forall x$, deixando apenas a matriz da fórmula $P(x)$.

2. **Trate as variáveis como variáveis livres**:
   As variáveis que eram ligadas pelo quantificador universal agora são tratadas como variáveis livres na matriz da fórmula.

**7. Conversão para FNC**:

1. Use as leis distributivas para mover as conjunções para dentro e as disjunções para fora.
2. Certifique-se de que a fórmula esteja na forma de uma conjunção de disjunções (cláusulas).

**8. Conversão para FND**:

1. Use as leis distributivas para mover as disjunções para dentro e as conjunções para fora.
2. Certifique-se de que a fórmula esteja na forma de uma disjunção de conjunções.

#### Exemplo 1 - Duas fórmulas logicamente equivalentes

Vamos considerar duas fórmulas logicamente equivalentes, uma na Forma Prenex e outra não considere a fórmula original:

$$\forall x \exists y (P(x) \rightarrow Q(y))$$

Se convertida para a Forma Prenex teremos:

$$\exists y \forall x (P(x) \rightarrow Q(y))$$

Cuja a equivalência pode ser provada por meio do seguinte raciocínio: seja $I$ uma interpretação booleana das variáveis $P$ e $Q$. Suponha $I$ satisfaz $\forall x \exists y (P(x) \rightarrow Q(y))$. Logo, para todo $x$ no domínio, existe um $y$ tal que: se $P(x)$ é verdadeiro, então $Q(y)$ também é verdadeiro. Isso é equivalente a dizer: existe um $y$, tal que para todo $x$, se $P(x)$ é verdadeiro, $Q(y)$ também é verdadeiro. Ou seja, $I$ também satisfaz: $\exists y \forall x (P(x) \rightarrow Q(y))$. Por um raciocínio simétrico, o oposto também é verdadeiro. Portanto, as fórmulas são logicamente equivalentes.

#### Exemplo 2 - Fórmula sem Forma Prenex

$$\forall x (P(x) \rightarrow \exists y Q(x,y))$$

Não pode ser convertida à Forma Prenex pois o quantificador $\exists y$ está dentro do escopo de de uma implicação ($\rightarrow$).

>A conversão para Forma Normal Conjuntiva é útil para métodos de prova. A conversão para Forma Normal Disjuntiva é menos comum, mas pode ser útil em alguns contextos de análise lógica. **CUIDADO: a eliminação dos quantificadores pode alterar a interpretação da fórmula em alguns modelos, mas é útil porque preserva a satisfatibilidade**.

### Usando a Tabela-Verdade para Gerar Formas Normais

Em meio à precisão rígida da lógica proposicional, a tabela verdade surge como nossa bússola fiel. Com ela, discernimos, sem rodeios, os caminhos para as Formas Normais Conjuntiva e Disjuntiva. Cortamos através da névoa de possibilidades, fixando nosso olhar nas linhas nítidas onde a verdade ou a falsidade se manifestam. Encaramos, então, a fórmula que se descortina diante de nós.

Considere a Fórmula Bem Formada dada por: $(A \lor B) \rightarrow (C \land \neg A)$, se encontrarmos sua Tabela Verdade, podemos encontrar, tanto a Forma Normal Conjuntiva quanto a Forma Normal Disjuntiva. Bastando fixar nosso olhar na verdade, ou na falsidade.

#### Gerando a Forma Normal Disjuntiva

Para transformar $(A \lor B) \rightarrow (C \land \neg A)$ na sua Forma Normal Conjuntiva, como um cozinheiro de bordo, devemos seguir rigidamente, os seguintes passos:

1. Criar a Tabela-Verdade

   $$
   \begin{array}{cccc|c|c|c}
   A & B & C & \neg A & A \lor B & C \land \neg A & (A \lor B) \rightarrow (C \land \neg A) \\
   \hline
   T & T & T & F & T & F & F \\
   T & T & F & F & T & F & F \\
   T & F & T & F & T & F & F \\
   T & F & F & F & T & F & F \\
   F & T & T & T & T & T & T \\
   F & T & F & T & T & F & F \\
   F & F & T & T & F & T & T \\
   F & F & F & T & F & T & T \\
   \end{array}
   $$

2. Identificar as Linhas com Resultado Verdadeiro

   As linhas 5, 7 e 8 têm resultado verdadeiro.

3. Construir a FND usando as linhas com resultados verdadeiros:

Neste passo, nosso objetivo é construir uma expressão que seja verdadeira nas linhas 5, 7 e 8 (as linhas onde o resultado é verdadeiro), e falsa em todos os outros casos. Para fazer isso, criamos uma disjunção (uma expressão _OR_) para cada linha verdadeira que reflete as condições das variáveis nesta linha, e então unimos essas disjunções com uma conjunção (uma operação **AND**) para criar a Forma Normal Disjuntiva desejada:

a. **Primeiro Termo Correspondente a Linha 5: $(\neg A \land B \land C)$**
Este termo é verdadeiro quando $A$ é falso, $B$ é verdadeiro e $C$ é verdadeiro, o que corresponde à linha 5 da tabela.

b. **Segundo Termo Correspondente a Linha 7: $(\neg A \land \neg B \land C)$**
Este termo é verdadeiro quando $A$ é falso, $B$ é falso e $C$ é verdadeiro, o que corresponde à linha 7 da tabela.

c. **Terceiro Correspondente a Linha 8: $(\neg A \land \neg B \land \neg C)$**
Este termo é verdadeiro quando $A$ é falso, $B$ é falso e $C$ é falso, o que corresponde à linha 8 da tabela.

Finalmente, unimos estes termos com operações OR ($\lor$) para criar a expressão FND completa:

$$
(A \lor B) \rightarrow (C \land \neg A) = (\neg A \land B \land C) \lor (\neg A \land \neg B \land C) \lor (\neg A \land \neg B \land \neg C)
$$

A expressão acima será verdadeira se qualquer um dos termos (ou seja, qualquer uma das linhas 5, 7 ou 8 da tabela) for verdadeiro, garantindo que a expressão capture exatamente as condições em que $(A \lor B) \rightarrow (C \land \neg A)$ é verdadeira de acordo com a tabela-verdade.

#### Gerando a Forma Normal Conjuntiva

Partindo da mesma tabela verdade da expressão $(A \lor B) \rightarrow (C \land \neg A)$, nossa bússola nesta fase da jornada, precisaremos voltar nosso olhar cuidadoso para as linhas com resultado falso e então teremos:

1. Identificar as Linhas com Resultado Falso

   As linhas $1$, $2$, $3$, $4$ e $6$ têm resultado falso.

2. Construir a Forma Normal Conjuntiva: para cada linha falsa, criaremos uma disjunção que represente a negação da linha e as combinaremos com uma conjunção. Como um pescador que cria uma rede entrelaçando fios com nós. A construção dos termos disjuntivos considerará as variáveis que tornam a fórmula falsa na respectiva linha da Tabela verdade:

   - Linha 1: $(\neg A \lor \neg B \lor \neg C \lor A)$;
   - Linha 2: $(\neg A \lor \neg B \lor C \lor A)$;
   - Linha 3: $(\neg A \lor B \lor \neg C \lor A)$;
   - Linha 4: $(\neg A \lor B \lor C \lor A)$;
   - Linha 6: $(A \lor \neg B \lor C \lor \neg A)$.

   Combinando-os com uma conjunção, temos a Forma Normal Conjuntiva:

   $$
   \begin{align*}
   (A \lor B) \rightarrow (C \land \neg A) &\equiv (\neg A \lor \neg B \lor \neg C \lor A) \\
   &\land (\neg A \lor \neg B \lor C \lor A) \\
   &\land (\neg A \lor B \lor \neg C \lor A) \\
   &\land (\neg A \lor B \lor C \lor A) \\
   &\land (A \lor \neg B \lor C \lor \neg A)
   \end{align*}
   $$

Lamentavelmente, as tabelas verdade não têm utilidade na Lógica de Primeira Ordem quando usamos predicados e quantificadores. Skolemização e Forma Normal Prenex são as rotas que precisaremos dominar para desvendar esse enigma.

## Skolemização

A Skolemização é uma técnica usada na Lógica de Primeira Ordem para eliminar quantificadores existenciais em fórmulas. Consiste em substituir as variáveis existenciais por Constantes ou Funções Skolem. Considere a fórmula a seguir com um quantificador universal e um existencial:

$$\forall x \exists y P(x,y)$$

Ao aplicar a Skolemização, a variável existencial $y$ é substituída por uma Função de Skolem $f(x)$:

$$P(x,f(x))$$

Para uma fórmula com dois quantificadores universais e dois existenciais:

$$\forall x \forall z \exists y \exists w R(x,y,z,w)$$

A Skolemização resultará em:

$$\forall x \forall z R(x,f(x),z,g(x,z))$$

Onde $f(x)$ e $ g(x,z)$ são Funções Skolem introduzidas para substituir as variáveis existenciais $y$ e $w $ respectivamente. A escolha entre usar uma Constante Skolem ou uma Função Skolem durante a Skolemização depende do escopo dos quantificadores na fórmula original. Aqui estão as regras e passos para realizar a Skolemização de forma mais explicativa:

**Passo 1: Identificar os Quantificadores Existenciais**: comece identificando os quantificadores existenciais na fórmula.

**Passo 2: Determinar se a Variável Existencial Depende de Variáveis Universais**: para cada variável ligada a um quantificador existencial, determinamos se ela depende ou não de alguma variável universal. Isso significa verificar se existem quantificadores universais que _dominam_ a variável existencial. Se a variável existencial não depende de variáveis universais, usamos uma Constante de Skolem. Caso contrário, usamos uma Função de Skolem que leva como parâmetros as variáveis universais que a dominam.

**Passo 3: Substituir as Variáveis Existenciais**: agora, substituímos todas as variáveis existenciais na fórmula original de acordo com as decisões tomadas no Passo 2. Se usarmos Constantes de Skolem, substituímos as variáveis existenciais diretamente pelas constantes. Se usarmos Funções de Skolem, substituímos as variáveis existenciais pelas funções de Skolem aplicadas às variáveis universais apropriadas.

**Exemplo 1**: considere a Fórmula Bem Formada dada por: $\forall x \exists y \ P(x,y)$

1. Identificamos o quantificador existencial que introduz a variável $y$.

2. A variável $y$ não depende de nenhuma variável universal, então usamos uma Constante de Skolem, digamos $a$. A fórmula se torna:

   $$\forall x \ P(x,a)$$

**Exemplo 2**: considere a fórmula original: $\forall x \forall z \exists y \ Q(x,y,z)$

1. Identificamos o quantificador existencial que introduz a variável $y$.

2. A variável $y$ depende de duas variáveis universais, $x$ e $z$. Portanto, usamos uma Função de Skolem, digamos $f(x,z)$. A fórmula se torna:

   $$\forall x \forall z \ Q(x,f(x,z),z)$$

Substituímos $y$ por $f(x,z)$, que é uma função que depende das variáveis universais $x$ e $z$.

Em resumo, a Skolemização simplifica fórmulas quantificadas, eliminando quantificadores existenciais e substituindo variáveis por Constantes ou Funções de Skolem, dependendo de sua relação com quantificadores universais. Isso auxilia na conversão de fórmulas quantificadas para a Forma Normal Conjuntiva e na simplificação da lógica.

### Exemplos de conversão em formas normais, conjuntiva e disjuntiva

a) Todos os alunos estudam ou alguns professores ensinam matemática

**Lógica de Primeiro Grau**:

$$\forall x(\text{Aluno}(x) \rightarrow \text{Estuda}(x)) \lor \exists y(\text{Professor}(y) \land \text{EnsinaMatemática}(y))$$

**Forma Normal Conjuntiva (FNC)**:

1. Convertendo a implicação:

   $$\neg \text{Aluno}(x) \lor \text{Estuda}(x)$$

2. Adicionando a disjunção existencial:

   $$(\neg \text{Aluno}(x) \lor \text{Estuda}(x)) \land (\text{Professor}(y) \land \text{EnsinaMatemática}(y))$$

**Forma Normal Disjuntiva (FND)**:

1. Negando o consequente do implicador:

   $$\text{Aluno}(x) \land \neg \text{Estuda}(x)$$

2. Adicionando a conjunção existencial negada:

   $$(\text{Aluno}(x) \land \neg \text{Estuda}(x)) \lor (\neg \text{Professor}(y) \lor \neg \text{EnsinaMatemática}(y))$$

b) Algum aluno estuda e todo professor ensina

**Lógica de Primeiro Grau**:

$$\exists x(\text{Aluno}(x) \land \text{Estuda}(x)) \land \forall y(\text{Professor}(y) \rightarrow \text{Ensina}(y))$$

**Forma Normal Conjuntiva (FNC)**:

1. Convertendo a implicação:

   $$\neg \text{Professor}(y) \lor \text{Ensina}(y)$$

2. Adicionando a conjunção existencial:

   $$(\text{Aluno}(x) \land \text{Estuda}(x)) \land (\neg \text{Professor}(y) \lor \text{Ensina}(y))$$

**Forma Normal Disjuntiva (FND)**:

1. Negando a conjunção existencial:

   $$\neg \text{Aluno}(x) \lor \neg \text{Estuda}(x)$$

2. Adicionando a conjunção negada do consequente do implicador:

   $$(\neg \text{Aluno}(x) \lor \neg \text{Estuda}(x)) \lor (\text{Professor}(y) \land \neg \text{Ensina}(y))$$

c) Todo estudante é inteligente ou algum professor é sábio

**Lógica de Primeiro Grau**:

$$\forall x(\text{Estudante}(x) \rightarrow \text{Inteligente}(x)) \lor \exists y(\text{Professor}(y) \land \text{Sábio}(y))$$

**Forma Normal Conjuntiva (FNC)**:

1. Convertendo a implicação:

   $$\neg \text{Estudante}(x) \lor \text{Inteligente}(x)$$

2. Adicionando a disjunção existencial:

   $$(\neg \text{Estudante}(x) \lor \text{Inteligente}(x)) \land (\text{Professor}(y) \land \text{Sábio}(y))$$

**Forma Normal Disjuntiva (FND)**:

1. Negando o consequente do implicador:

   $$\text{Estudante}(x) \land \neg \text{Inteligente}(x)$$

2. Adicionando a conjunção existencial negada:

   $$(\text{Estudante}(x) \land \neg \text{Inteligente}(x)) \lor (\neg \text{Professor}(y) \lor \neg \text{Sábio}(y))$$

d) Todo animal corre ou algum pássaro voa

**Lógica de Primeiro Grau**:

$$\forall x(\text{Animal}(x) \rightarrow \text{Corre}(x)) \lor \exists y(\text{Pássaro}(y) \land \text{Voa}(y))$$

**Forma Normal Conjuntiva (FNC)**:

1. Convertendo a implicação:

   $$\neg \text{Animal}(x) \lor \text{Corre}(x)$$

2. Adicionando a disjunção existencial:

   $$(\neg \text{Animal}(x) \lor \text{Corre}(x)) \land (\text{Pássaro}(y) \land \text{Voa}(y))$$

**Forma Normal Disjuntiva (FND)**:

1. Negando o consequente do implicador:

   $$\text{Animal}(x) \land \neg \text{Corre}(x)$$

2. Adicionando a conjunção existencial negada:

   $$(\text{Animal}(x) \land \neg \text{Corre}(x)) \lor (\neg \text{Pássaro}(y) \lor \neg \text{Voa}(y))$$

## Mundos na Lógica de Primeira Ordem

A lógica de primeira ordem, também conhecida como lógica de predicados de primeira ordem, emergiu no final do século XIX e início do século XX, principalmente através dos trabalhos de Gottlob Frege, Bertrand Russell e Alfred North Whitehead. Essa lógica foi desenvolvida como uma extensão da lógica proposicional, permitindo a representação de afirmações mais complexas sobre objetos e suas relações. A lógica de primeira ordem tornou-se uma ferramenta fundamental na matemática, filosofia e ciência da computação, especialmente na formalização de sistemas dedutivos e na fundamentação da matemática.

A capacidade de definir "mundos" ou estruturas dentro da lógica de primeira ordem é que permite modelar e analisar sistemas complexos. Esses mundos representam interpretações ou modelos que atribuem significado às fórmulas lógicas, permitindo verificar a validade de argumentos, provar teoremas e desenvolver sistemas de inteligência artificial. Na ciência da computação, por exemplo, a lógica de primeira ordem é usada em linguagens de programação declarativas, sistemas de banco de dados e na verificação de software.

### Definição Formal de um Mundo

Na lógica de primeira ordem, um **mundo** ou **modelo** é uma estrutura que consiste em:

1. **Domínio de Discurso ($D$)**: Um conjunto não vazio de objetos sobre os quais as variáveis quantificadas podem se referir.
   Exemplo: $D = \{1, 2, 3, 4, 5\}$ (um domínio de números inteiros de 1 a 5)

2. **Símbolos de Constantes**: Elementos específicos do domínio que são nomeados.
   Exemplo: $a = 1$, $b = 3$ (onde $a$ e $b$ são constantes que se referem a elementos específicos do domínio)

3. **Símbolos de Função**: Mapeamentos de elementos do domínio para outros elementos dentro do domínio.
   Exemplo: $f(x) = x + 1$ (uma função que mapeia cada elemento do domínio para seu sucessor)

4. **Símbolos de Predicado**: Propriedades ou relações que podem ser atribuídas aos elementos do domínio.
   Exemplo: $P(x)$: "x é par", $R(x, y)$: "x é menor que y"

5. **Interpretação**: Uma função que atribui significado aos símbolos não lógicos (constantes, funções e predicados) em termos do domínio.
   Exemplo:
   - $I(a) = 1$
   - $I(f(2)) = 3$
   - $I(P) = \{2, 4\}$
   - $I(R) = \{(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)\}$

Um modelo $M$ para uma linguagem $L$ é então definido como $M = (D, I)$, onde $D$ é o domínio e $I$ é a interpretação.

Neste exemplo, temos um modelo $M$ onde:

$$M = (\{1, 2, 3, 4, 5\}, I)$$

com $I$ definido como acima. Este modelo representa um "mundo" onde podemos fazer afirmações sobre números inteiros de 1 a 5, suas relações de ordem e paridade.

### Construção de Mundos

Vamos ilustrar a definição acima com um exemplo concreto.

**Domínio de Objetos ($D$)**:

$$D = \{ a, b, c \}$$

**Onde**: $a$, $b$ e $c$ são objetos distintos no domínio.

**Símbolos de Constante**: $e$: representa um elemento específico do domínio.

**Símbolos de Função**: $f(x)$: "o melhor amigo de x."

**Símbolos de Predicado**:

- $P(x)$: "x é uma pessoa."
- $Q(x)$: "x é um animal."
- $R(x, y)$: "x gosta de y."

**Interpretação no Mundo**: atribuímos significado aos símbolos não lógicos:

- $I(e) = a$ (a constante $e$ refere-se ao objeto $a$)
- $I(f)(a) = b$ (o melhor amigo de $a$ é $b$)
- $I(f)(b) = c$ (o melhor amigo de $b$ é $c$)
- $I(f)(c) = a$ (o melhor amigo de $c$ é $a$)
- $P(a)$ é verdadeiro (a é uma pessoa).
- $P(b)$ é verdadeiro (b é uma pessoa).
- $P(c)$ é falso (c não é uma pessoa).
- $Q(c)$ é verdadeiro (c é um animal).
- $R(a, c)$ é verdadeiro (a gosta de c).
- $R(b, c)$ é verdadeiro (b gosta de c).
- $R(a, b)$ é falso (a não gosta de b).

**Representação Formal do Mundo**:

As informações acima podem ser formalizadas através das seguintes fórmulas:

1. $P(a) \land P(b) \land \neg P(c)$: a e b são pessoas; c não é.
2. $Q(c)$: c é um animal.
3. $R(a, c) \land R(b, c) \land \neg R(a, b)$: a e b gostam de c; a não gosta de b.
4. $f(a) = b \land f(b) = c \land f(c) = a$: representação da função "melhor amigo".
5. $e = a$: a constante $e$ refere-se ao objeto $a$.

Este mundo agora inclui não apenas predicados, mas também uma constante $e$ e uma função $f$, enriquecendo a estrutura e as relações entre os objetos do domínio.

O mundo que definimos acima, embora simples, ilustra vários conceitos importantes da lógica de primeira ordem:

1. **Domínio Finito**: Nosso domínio $D = \{a, b, c\}$ é finito, o que facilita a compreensão, mas é importante notar que domínios em lógica de primeira ordem podem ser infinitos.

2. **Relações entre Objetos**: Através dos predicados $P$, $Q$, e $R$, estabelecemos propriedades e relações entre os objetos. Isso demonstra como a lógica de primeira ordem pode capturar informações estruturadas sobre um conjunto de entidades.

3. **Funções**: A introdução da função $f$ (melhor amigo) mostra como podemos mapear objetos do domínio para outros objetos do mesmo domínio, criando relações mais complexas.

4. **Constantes Nomeadas**: A constante $e$ ilustra como podemos nos referir diretamente a elementos específicos do domínio.

5. **Expressividade**: Mesmo com apenas três objetos, três predicados, uma função e uma constante, somos capazes de expressar uma variedade de fatos e relações.

**Limitações do Exemplo**:

1. **Escala**: Em aplicações reais, os domínios e conjuntos de predicados e funções são geralmente muito maiores e mais complexos.

2. **Tipos de Objetos**: Nosso exemplo mistura pessoas e animais no mesmo domínio. Em modelos mais sofisticados, poderíamos usar tipos ou sortes para distinguir diferentes categorias de objetos.

3. **Relações Temporais**: Este modelo é estático. Em muitas aplicações, precisaríamos representar como as relações mudam ao longo do tempo.

4. **Incerteza**: A lógica de primeira ordem clássica lida com afirmações definitivamente verdadeiras ou falsas. Não há representação direta de probabilidades ou incertezas.

**Extensões Possíveis**: para tornar este mundo mais rico e realista, poderíamos:

1. Adicionar mais objetos ao domínio.
2. Introduzir predicados mais complexos, como $Irmão(x,y)$ ou $MaisVelho(x,y)$.
3. Definir funções adicionais, como $Idade(x)$ ou $Pai(x)$.
4. Incorporar axiomas que expressem regras gerais sobre o mundo, como $\forall x (P(x) \rightarrow \neg Q(x))$ (nada pode ser simultaneamente uma pessoa e um animal).

Este exemplo simplificado serve como um ponto de partida para entender como modelos mais complexos podem ser construídos na lógica de primeira ordem para representar conhecimento e raciocinar sobre domínios mais sofisticados.

### Aplicações e Importância

A definição de mundos na lógica de primeira ordem tem aplicações fundamentais em diversas áreas, abrangendo desde a matemática pura até as ciências aplicadas e a engenharia, passando pela biologia e economia. Na matemática, essa abordagem suporta a prova de teoremas, onde modelos são utilizados para verificar a consistência de sistemas axiomáticos e construir contraexemplos. A teoria dos modelos, um ramo importante da lógica matemática, se dedica ao estudo das relações entre estruturas matemáticas e as linguagens formais que as descrevem. Além disso, nos fundamentos da matemática, a lógica de primeira ordem desempenha um papel central na formalização de conceitos matemáticos, como exemplificado pela Teoria dos Conjuntos de Zermelo-Fraenkel com o Axioma da Escolha (ZFC).

### A Teoria dos Modelos

A teoria dos modelos estuda as relações entre estruturas matemáticas e as linguagens formais que as descrevem. Vamos considerar um exemplo simples, onde analisamos a relação entre uma estrutura numérica e a linguagem formal que a descreve.

Seja $M = (D, I)$ um modelo onde:

$$D = \{0, 1, 2, 3, 4, 5\}$$

Este domínio representa um conjunto de números inteiros de $0$ a $5$. A interpretação $I$ atribui significados aos símbolos não lógicos:

1. **Função de Adição ($+$)**: mapeia pares de elementos do domínio para sua soma.

   $$ I(+) : (x, y) \mapsto (x + y \mod 6)$$$
   (A adição é feita com módulo $6$).

2. **Símbolo de Constante**: a constante $c = 3$.

3. **Predicado de Paridade**: $P(x)$ significa "x é par".

   $$ I(P) = \{0, 2, 4\} $$

Com isso, podemos construir fórmulas na linguagem formal e verificar se são satisfeitas no modelo $M$.

**Regras**:

1. A soma de dois números pares é sempre par:

   $$ \forall x \forall y (P(x) \land P(y) \rightarrow P(x + y)) $$

   Esta fórmula é verdadeira em $M$.

2. O número $3$ não é par:

   $$ \neg P(3) $$

   Esta fórmula também é verdadeira em $M$, pois $3 \notin \{0, 2, 4\}$.

3. A adição em $M$ é comutativa:

   $$ \forall x \forall y (x + y = y + x) $$

   Esta fórmula é verdadeira, uma vez que a adição em $M$ é comutativa no módulo $6$.

Neste exemplo, a **estrutura matemática** $M$ é um conjunto de números inteiros de $0$ a $5$ com a operação de adição módulo $6$. As **fórmulas na linguagem formal** são expressões que descrevem propriedades de números, como paridade e comutatividade da adição.

A teoria dos modelos nos permite verificar se essas fórmulas são satisfeitas em $M$. O estudo dessas relações entre fórmulas e estruturas é central na lógica matemática e fundamenta muitas áreas, como a álgebra e a aritmética, além de fornecer ferramentas para analisar a consistência de teorias matemáticas.

As ciências cognitivas constituem outro campo que faz uso extensivo do conceito de mundos. A modelagem cognitiva se baseia na representação formal de processos de raciocínio e tomada de decisão, enquanto a psicologia do raciocínio estuda como os seres humanos realizam inferências lógicas, muitas vezes comparando o raciocínio humano com os princípios formais da lógica. A engenharia de sistemas também faz uso do conceito de mundos. A especificação de requisitos e a modelagem de domínio se apoiam na capacidade de descrever formalmente sistemas complexos e suas interações, bem como representar conhecimento específico de domínio em diversos sistemas de engenharia. Entretanto, precisamos destacar duas áreas importantes para este trabalho: a ciência da computação e a linguística computacional.

#### Aplicações na Ciência da Computação

Na ciência da computação, as aplicações são vastas e variadas. No campo da inteligência artificial, a representação de conhecimento se beneficia enormemente da capacidade de modelar domínios complexos para sistemas especialistas e agentes inteligentes. O planejamento automatizado utiliza a descrição de estados do mundo e ações para resolver problemas, enquanto o processamento de linguagem natural depende da análise semântica de textos e da compreensão de contexto. Em bancos de dados, a modelagem conceitual e as consultas semânticas se apoiam fortemente em princípios lógicos para descrever formalmente esquemas e expressar consultas complexas. A verificação de software também se beneficia, com métodos formais sendo empregados para especificar e verificar propriedades de sistemas, e técnicas de model checking permitindo a verificação automática de propriedades em sistemas de estados finitos.

##### Exemplo 1 - Diagnóstico Médico

Em sistemas especialistas de diagnóstico médico, a capacidade de definir e manipular mundos lógicos permite:

1. **Raciocínio sobre cenários hipotéticos**: um sistema especialista pode criar um mundo lógico $M = (D, I)$ representando um paciente com sintomas específicos:

   $$D = \{p, f, t, d, c, g, a\}$$

   Onde $p$ representa o paciente, $f$ (febre), $t$ (tosse), $d$ (dor de cabeça), $c$ (COVID-19), $g$ (gripe), e $a$ (alergia) são elementos do domínio.

   A interpretação $I$ define predicados como:

   - $S(x,y)$: "x tem sintoma y"
   - $D(x,z)$: "x tem doença z"
   - $T(x,w)$: "x fez teste w"

   O sistema pode então raciocinar sobre um cenário hipotético onde:

   $$S(p,f) \land S(p,t) \land \neg S(p,d)$$

   Este mundo representa um paciente com febre e tosse, mas sem dor de cabeça.

2. **Planejamento de ações em ambientes complexos**: baseado no mundo atual, o sistema pode planejar uma sequência de testes diagnósticos. Por exemplo, podemos definir uma função de ação $A(x,y)$ que representa "realizar ação y no paciente x".

   O sistema pode usar regras como:

   $$\forall x (S(x,f) \land S(x,t) \rightarrow A(x, \text{"testar_covid"}))$$

   $$\forall x (S(x,t) \land \neg S(x,f) \rightarrow A(x, \text{"testar_alergia"}))$$

   Assim, no nosso cenário hipotético, o sistema recomendaria testar para COVID-19.

3. **Inferência de novas informações a partir de dados existentes**: o sistema pode usar regras de inferência para derivar novos fatos. Por exemplo:

   $$\forall x (S(x,f) \land S(x,t) \land T(x, \text{"covid_positivo"}) \rightarrow D(x,c))$$

   $$\forall x (S(x,f) \land S(x,t) \land T(x, \text{"covid_negativo"}) \land T(x, \text{"gripe_positivo"}) \rightarrow D(x,g))$$

   Se adicionarmos ao nosso mundo $T(p, \text{"covid_positivo"})$, o sistema pode inferir $D(p,c)$, concluindo que o paciente tem COVID-19.

4. **Validação de consistência em bases de conhecimento**: o sistema pode verificar se o diagnóstico proposto é consistente com o conhecimento existente. Por exemplo, podemos ter uma regra de consistência:

   $$\forall x \neg(D(x,c) \land D(x,g))$$

   Esta regra afirma que um paciente não pode ter COVID-19 e gripe simultaneamente. Se o sistema tentar adicionar $D(p,g)$ ao mundo onde já existe $D(p,c)$, ele detectará uma inconsistência.

   Além disso, o sistema pode usar regras de integridade mais complexas, como:

   $$\forall x (D(x,c) \rightarrow \exists y (S(x,y) \land (y = f \lor y = t \lor y = d)))$$

   Esta regra afirma que se um paciente tem COVID-19, ele deve ter pelo menos um dos sintomas: febre, tosse ou dor de cabeça.

Neste exemplo expandido, o mundo lógico permite ao sistema especialista:

   1. Representar e raciocinar sobre o estado de saúde do paciente;
   2. Planejar testes diagnósticos baseados em regras predefinidas;
   3. Fazer inferências sobre possíveis doenças usando regras lógicas;
   4. Garantir a consistência do diagnóstico através de verificações de integridade.

##### Exemplo 2 - Robô de Limpeza

Em sistemas de planejamento para robôs autônomos, a capacidade de definir e manipular mundos lógicos permite:

1. **Raciocínio sobre cenários hipotéticos**: um sistema de IA para um robô de limpeza pode criar um mundo lógico $M = (D, I)$ representando o estado de um ambiente:

   $$D = \{r, s1, s2, s3, s4, p1, p2, l, d\}$$

   Onde $r$ representa o robô, $s1$ a $s4$ são setores do ambiente, $p1$ e $p2$ são tipos de sujeira (por exemplo, poeira e líquido), $l$ é o carregador, e $d$ é a lixeira.

   A interpretação $I$ define predicados como:

   - $Em(x,y)$: "x está em y";
   - $Sujo(x,y)$: "x está sujo com y";
   - $Limpo(x)$: "x está limpo";
   - $TemFerramenta(x,y)$: "x tem a ferramenta para limpar y".

   O sistema pode raciocinar sobre um cenário hipotético onde:

   $$Em(r,s1) \land Sujo(s2,p1) \land Sujo(s3,p2) \land Limpo(s4) \land TemFerramenta(r,p1)$$

   Este mundo representa um robô no setor 1, com setores 2 e 3 sujos, setor 4 limpo, e o robô equipado para limpar poeira.

2. **Planejamento de ações em ambientes complexos**: baseado no mundo atual, o sistema pode planejar uma sequência de ações de limpeza. Definimos uma função de ação $A(x,y,z)$ que representa "x realiza ação y no local z".

   O sistema pode usar regras como:

   $$\forall x,y,z \left( Em(x,y) \land Sujo(z, p_1) \land TemFerramenta(x, p_1) \land y \neq z \rightarrow A(x, \text{"mover"}, z) \right)$$

   Assumindo que `p_1` é uma constante conhecida no universo do discurso, a fórmula está sintaticamente correta. Mas se `p_1` for uma variável, ela deve ser quantificada. Como está, a fórmula **é ambígua** e **potencialmente inválida**.

   $$\forall x,y (Em(x,y) \land Sujo(y,p1) \land TemFerramenta(x,p1) \rightarrow A(x, \text{"limpar"}, y))$$

   Assim, no nosso cenário, o sistema planejaria mover o robô para o setor 2 e então limpá-lo.

3. **Inferência de novas informações a partir de dados existentes**: o sistema pode usar regras de inferência para atualizar o estado do mundo após ações. Por exemplo:

   $$\forall x,y (A(x, \text{"limpar"}, y) \land Sujo(y,p1) \land TemFerramenta(x,p1) \rightarrow Limpo(y))$$

   $$\forall x,y,z (A(x, \text{"mover"}, z) \land Em(x,y) \rightarrow Em(x,z) \land \neg Em(x,y))$$

   Após a ação de limpeza no setor 2, o sistema inferiria $Limpo(s2)$, atualizando o estado do mundo.

4. **Validação de consistência em bases de conhecimento**: o sistema pode verificar se o estado do mundo é consistente após cada ação. Por exemplo, podemos ter regras de consistência:

   $$\forall x \neg(Limpo(x) \land Sujo(x,p1))$$

   $$\forall x,y,z (Em(x,y) \land Em(x,z) \rightarrow y = z)$$

   A primeira regra afirma que um setor não pode estar limpo e sujo ao mesmo tempo. A segunda garante que o robô só pode estar em um lugar de cada vez.

   Além disso, o sistema pode usar regras de integridade mais complexas, como:

   $$\forall x ((\exists y Sujo(x,y)) \rightarrow \neg Limpo(x))$$

   Esta regra afirma que se um setor está sujo com qualquer tipo de sujeira, ele não pode ser considerado limpo.

Neste exemplo, o mundo lógico permite ao sistema de IA do robô de limpeza:

   1. Representar e raciocinar sobre o estado do ambiente e do próprio robô;
   2. Planejar ações de limpeza baseadas em regras predefinidas e no estado atual;
   3. Fazer inferências sobre os resultados das ações, atualizando o estado do mundo;
   4. Garantir a consistência do estado do mundo através de verificações de integridade.

Este uso sofisticado da lógica de primeira ordem demonstra como sistemas de IA podem manipular informações complexas e realizar raciocínios avançados em domínios de planejamento e execução de tarefas autônomas.

#### Aplicações na Linguística Computacional

Na linguística computacional, a semântica formal emprega a lógica de primeira ordem para modelar o significado de sentenças e discursos em linguagens naturais. As gramáticas formais, por sua vez, se beneficiam dessa abordagem na descrição da estrutura sintática de linguagens, e a análise do discurso utiliza esses princípios para representar contexto e relações entre sentenças em textos.

##### Exemplo 1 - Gramática Formal

Na linguística, particularmente no estudo de gramáticas formais, a lógica de primeira ordem pode ser usada para definir e analisar estruturas sintáticas. Considere o seguinte exemplo de um mundo lógico representando uma gramática simplificada:

Seja $M = (D, I)$ um modelo onde:

$$
D = \{s, np, vp, n, v, det, \text{"o"}, \text{"gato"}, \text{"caça"}, \text{"rato"}\}
$$

Onde $s$ (sentença), $np$ (sintagma nominal), $vp$ (sintagma verbal), $n$ (substantivo), $v$ (verbo), $det$ (determinante) são categorias sintáticas, e "o", "gato", "caça", "rato" são palavras.

A interpretação $I$ define predicados e funções como:

1. $Categoria(x, y)$: "x é uma palavra da categoria sintática y"
2. $Compõe(x, y, z)$: "x é composto por y seguido de z"
3. $Precede(x, y)$: "x precede imediatamente y na sentença"

Podemos definir regras gramaticais usando fórmulas lógicas:

1. Regra para sintagma nominal:

   $$\forall x \forall y (Categoria(x, det) \land Categoria(y, n) \land Precede(x, y) \rightarrow \exists z (Compõe(z, x, y) \land Categoria(z, np)))$$

2. Regra para sintagma verbal:

   $$\forall x (Categoria(x, v) \rightarrow \exists y (Compõe(y, x, x) \land Categoria(y, vp)))$$

3. Regra para sentença:

   $$\forall x \forall y (Categoria(x, np) \land Categoria(y, vp) \land Precede(x, y) \rightarrow \exists z (Compõe(z, x, y) \land Categoria(z, s)))$$

4. Atribuição de categorias às palavras:

   $$Categoria(\text{"o"}, det)$$

   $$Categoria(\text{"gato"}, n)$$

   $$Categoria(\text{"caça"}, v)$$

   $$Categoria(\text{"rato"}, n)$$

Agora, podemos usar este mundo lógico para:

1. **Analisar estruturas sintáticas**: dada a sequência de palavras "o gato caça o rato", podemos usar as regras para derivar sua estrutura sintática:

   $$Precede(\text{"o"}, \text{"gato"}) \land Precede(\text{"gato"}, \text{"caça"}) \land Precede(\text{"caça"}, \text{"o"}) \land Precede(\text{"o"}, \text{"rato"})$$

   A partir disso e das regras, podemos inferir:

   $$\exists np_1 (Compõe(np_1, \text{"o"}, \text{"gato"}) \land Categoria(np_1, np))$$

   $$\exists vp (Compõe(vp, \text{"caça"}, \text{"caça"}) \land Categoria(vp, vp))$$

   $$\exists np_2 (Compõe(np_2, \text{"o"}, \text{"rato"}) \land Categoria(np_2, np))$$

   $$\exists s (Compõe(s, np_1, vp) \land Categoria(s, s))$$

2. **Verificar a gramaticalidade de sentenças**: podemos verificar se uma sequência de palavras forma uma sentença válida ao tentar derivar um $s$ usando as regras.

3. **Gerar sentenças gramaticais**: podemos usar as regras para gerar todas as sentenças possíveis de um certo comprimento.

4. **Estudar ambiguidades**: poderíamos estender o modelo para lidar com ambiguidades estruturais, por exemplo, adicionando regras para sintagmas preposicionais.

Este exemplo demonstra como a lógica de primeira ordem pode ser usada para formalizar e raciocinar sobre estruturas gramaticais, permitindo análises sintáticas rigorosas e geração de sentenças gramaticalmente corretas.

> Um sintagma é um grupo de palavras que, juntas, formam uma unidade dentro de uma frase e desempenham uma função sintática específica. Cada sintagma tem um núcleo (ou "cabeça"), que é o elemento mais importante dentro do grupo e define o tipo de sintagma. O sintagma pode ser constituído apenas pelo núcleo ou por outras palavras que o acompanham, chamadas modificadores ou complementos. Existem diferentes tipos de sintagmas, dependendo da classe gramatical do núcleo:
>
> 1. Sintagma Nominal (SN): Tem um substantivo como núcleo. Exemplo: o gato preto (o núcleo é gato, um substantivo).
> 2. Sintagma Verbal (SV): Tem um verbo como núcleo. Exemplo: corre rápido (o núcleo é corre, um verbo).
> 3. Sintagma Adjetival (SAdj): Tem um adjetivo como núcleo. Exemplo: muito feliz (o núcleo é feliz, um adjetivo).
> 4. Sintagma Adverbial (SAdv): Tem um advérbio como núcleo. Exemplo: muito rapidamente (o núcleo é rapidamente, um advérbio).
> 5. Sintagma Preposicional (SP): Tem uma preposição seguida de um complemento, que pode ser um sintagma nominal ou outro. Exemplo: com cuidado (o núcleo é com, uma preposição).

#### Exemplos Aplicação da Lógica de Primeira Ordem em Biologia e Economia

A lógica de primeira ordem é uma ferramenta fundamenta para modelar e raciocinar sobre sistemas complexos. A seguir, a atenta leitora poderá estudar dois exemplos práticos de como a lógica de primeira ordem pode ser aplicada em biologia e economia.

##### Exemplo 1 - Sistemas Biológicos

Na biologia, a lógica de primeira ordem pode ser usada para modelar sistemas biológicos e suas interações. Considere o seguinte e de um mundo lógico representando uma cadeia alimentar simplificada.

Seja $M = (D, I)$ um modelo onde:

$$D = \{c, h, a, p, f\}$$

Onde $c$ (cobra), $h$ (gavião), $a$ (antílope), $p$ (planta), $f$ (fruto) são organismos.

A interpretação $I$ define predicados como:

   1. $Come(x, y)$: "x come y";
   2. $Herbívoro(x)$: "x é herbívoro";
   3. $Carnívoro(x)$: "x é carnívoro";
   4. $Produtor(x)$: "x é produtor";

Podemos usar a lógica para descrever as interações alimentares:

1. Regras de herbívoros:

   $$ \forall x (Herbívoro(x) \rightarrow \exists y (Come(x, y) \land Produtor(y))) $$

   (Um herbívoro come apenas produtores).

2. Regras de carnívoros:

   $$ \forall x (Carnívoro(x) \rightarrow \exists y (Come(x, y) \land Herbívoro(y))) $$

   (Um carnívoro come apenas herbívoros).

Atribuição de categorias aos organismos:

$$Herbívoro(a), Produtor(p), Produtor(f), Carnívoro(c), Carnívoro(h)$$

Agora, podemos usar este mundo lógico para:

   1. **Analisar interações tróficas**: Por exemplo, $Come(c, a)$ significa que a cobra come o antílope;

   2. **Verificar coerência ecológica**: As regras acima garantem que um herbívoro não comerá um carnívoro, e que um carnívoro não comerá plantas.

##### Exemplo 2 - Modelagem Econômica

Na economia, a lógica de primeira ordem pode ser aplicada para modelar mercados e interações econômicas. Considere o seguinte exemplo de um mundo lógico representando um mercado simples com consumidores e produtos.

Seja $M = (D, I)$ um modelo onde:

$$D = \{c_1, c_2, p_1, p_2, m\}$$

Onde $c_1$ e $c_2$ são consumidores, $p_1$ e $p_2$ são produtos, e $m$ é o mercado.

A interpretação $I$ define predicados como:

1. $Compra(x, y)$: "x compra o produto y"
2. $Disponível(y, m)$: "o produto y está disponível no mercado"
3. $Dinheiro(x, z)$: "o consumidor x tem dinheiro z"

Podemos usar a lógica para descrever transações no mercado:

1. Regra de compra:

   $$ \forall x \forall y (Dinheiro(x, z) \land Disponível(y, m) \land z \geq \text{Preço}(y) \rightarrow Compra(x, y)) $$

   (Um consumidor compra um produto se tiver dinheiro suficiente e o produto estiver disponível).

Atribuição de valores:

$$Dinheiro(c_1, 100), Dinheiro(c_2, 50), Disponível(p_1, m), Disponível(p_2, m)$$

Agora, podemos usar este mundo lógico para:

   1. **Analisar transações**: Por exemplo, $Compra(c_1, p_1)$ significa que o consumidor $c_1$ comprou o produto $p_1$;

   2. **Verificar restrições econômicas**: As regras garantem que um consumidor só pode comprar um produto se tiver dinheiro suficiente e se o produto estiver disponível no mercado.

Essa ampla gama de aplicações demonstra a versatilidade e a importância fundamental da definição de mundos na lógica de primeira ordem, estabelecendo-a como uma ferramenta essencial para o avanço do conhecimento e da tecnologia em múltiplas disciplinas.
A importância da definição de mundos na lógica de primeira ordem reside em sua capacidade de:

   1. Fornecer um framework rigoroso para representar conhecimento estruturado;
   2. Permitir raciocínio automatizado sobre informações complexas;
   3. Facilitar a comunicação precisa de ideias abstratas entre diferentes disciplinas;
   4. Servir como base para o desenvolvimento de sistemas inteligentes e adaptativos.

À medida que os sistemas se tornam mais complexos e as demandas por inteligência artificial aumentam, a habilidade de definir e trabalhar com mundos lógicos torna-se cada vez mais importante para o avanço tecnológico e científico.

#### Exercício de Aplicação da Lógica de Primeira Ordem

A seguir, a esforçada leitora terá a oportunidade de ver dois exercícios práticos que envolvem a aplicação da lógica de primeira ordem. O primeiro exercício aborda a coloração de um grafo, enquanto o segundo envolve grafos parcialmente coloridos. Ambos os exercícios são projetados para serem resolvidos sem o uso de funções, utilizando apenas relações e variáveis.

##### Exercício 1 - Coloração de um Grafo

Imagine que você está trabalhando como engenheiro de redes para uma grande empresa de tecnologia. Sua tarefa é planejar as conexões entre os servidores da empresa, garantindo que as comunicações entre eles não criem conflitos. O problema consiste em garantir que os servidores diretamente conectados não utilizem o mesmo canal de comunicação (representado por uma cor). Você tem, no máximo, $n$ servidores e deseja utilizar menos de $k+1$ canais de comunicação, respeitando que cada servidor pode se conectar diretamente a um número limitado de outros servidores, cujo limite é dado pelo grau de conexão $m$.

**Descrição do Problema**:

- **Servidor**: Representado como um nó em um grafo;
- **Conexão direta**: Representada como uma aresta entre dois nós;
- **Cor**: Representa o canal de comunicação atribuído a um servidor. Dois servidores diretamente conectados não podem compartilhar o mesmo canal;
- **Grau de um servidor**: O número de conexões diretas que ele tem com outros servidores;
- **Grau de conexão da rede**: O maior grau entre os servidores da rede.

O objetivo é determinar uma forma de atribuir um canal de comunicação a cada servidor de forma que não haja conflitos de comunicação entre servidores diretamente conectados, utilizando menos de $k+1$ canais.

**Solução**: vamos usar lógica de primeira ordem para modelar este problema sem utilizar funções, apenas relações e variáveis.

- um predicado binário $Cor(x, c)$, onde $x$ é um servidor e $c$ é uma cor/canal;
- um predicado unário $Servidor(x)$, que significa que $x$ é um servidor;
- um predicado binário $Conexao(x, y)$, que significa que $x$ está diretamente conectado a $y$.

**Regras ou Axiomas**:

1. Dois servidores diretamente conectados não podem usar o mesmo canal de comunicação:

   $$ \forall x \forall y \forall c: (Servidor(x) \land Servidor(y) \land Conexao(x, y) \land Cor(x, c) \rightarrow \neg Cor(y, c)) $$

2. Cada servidor deve receber exatamente uma cor:

   $$ \forall x: (Servidor(x) \rightarrow \exists c: Cor(x, c)) $$

   $$ \forall x \forall c1 \forall c2: (Servidor(x) \land Cor(x, c1) \land Cor(x, c2) \rightarrow c1 = c2) $$

3. Restrição de grau para um servidor (no máximo $m$ conexões):

   $$ \forall x: (Servidor(x) \rightarrow \neg\exists x_1,...,x_{m+1}: (\bigwedge_{i=1}^{m+1} Conexao(x, x_i) \land \bigwedge_{i \neq j} x_i \neq x_j)) $$

4. Número máximo de cores utilizadas (menos de $k+1$):

   $$ \neg\exists c_1,...,c_{k+1}: (\bigwedge_{i=1}^{k+1} (\exists x: Servidor(x) \land Cor(x, c_i)) \land \bigwedge_{i \neq j} c_i \neq c_j) $$

**Consultas Possíveis**:

Com esse modelo, você pode fazer as seguintes consultas:

1. **Verificar se dois servidores estão diretamente conectados**:

   - Consulta: $Conexao(a, b)$;
   - Resposta: **True** se o servidor $a$ estiver diretamente conectado ao servidor $b$, **False** caso contrário.

2. **Verificar qual canal de comunicação (cor) foi atribuído a um servidor**:

   - Consulta: $Cor(a, c)$;
   - Resposta: **True** se o servidor $a$ usa o canal $c$, **False** caso contrário.

3. **Verificar se dois servidores conectados têm cores diferentes**:

   - Consulta: $Conexao(a, b) \land \forall c: (Cor(a, c) \rightarrow \neg Cor(b, c))$;
   - Resposta: **True** se os servidores $a$ e $b$ estiverem diretamente conectados e tiverem cores diferentes, **False** se eles compartilharem a mesma cor ou não estiverem conectados.

4. **Verificar se um servidor tem mais de $m$ conexões diretas**:

   - Consulta: $\exists x_1,...,x_{m+1}: (\bigwedge_{i=1}^{m+1} Conexao(a, x_i) \land \bigwedge_{i \neq j} x_i \neq x_j)$;
   - Resposta: **True** se o servidor $a$ tiver mais de $m$ servidores diretamente conectados, **False** caso contrário.

5. **Verificar se a coloração da rede é válida**:

   - Consulta: $\forall x \forall y \forall c: (Servidor(x) \land Servidor(y) \land Conexao(x, y) \land Cor(x, c) \rightarrow \neg Cor(y, c))$;
   - Resposta: **True** se todos os servidores diretamente conectados tiverem cores diferentes, **False** se houver algum conflito de cores.

##### Exercício 2 - Grafos Parcialmente Coloridos

Dado um conjunto não vazio e finito de cores $\{c_1, \dots, c_k\}$, um grafo direcionado parcialmente colorido é uma estrutura $\langle N, R, C \rangle$ na qual:

- $N$ é um conjunto não vazio de nós;
- $R$ é uma relação binária sobre $N$;
- $C$ associa cores aos nós (nem todos os nós são necessariamente coloridos, e cada nó tem no máximo uma cor).

Forneça uma linguagem de Lógica de Primeira Ordem e um conjunto de axiomas que formalizem grafos parcialmente coloridos. Mostre que todo modelo dessa teoria corresponde a um grafo parcialmente colorido, e vice-versa. Para cada uma das seguintes propriedades, escreva uma fórmula que seja verdadeira apenas nos grafos que satisfazem a propriedade:

   1. Nós conectados não têm a mesma cor;
   2. O grafo contém apenas dois nós amarelos;
   3. Começando de um nó vermelho, pode-se alcançar um nó verde em no máximo 4 passos;
   4. Para cada cor, existe pelo menos um nó com essa cor;
   5. O grafo é composto por $|C|$ subgrafos disjuntos e não vazios, um para cada cor.

**Solução**:

- Um predicado binário $edge$, onde $edge(n, m)$ significa que o nó $n$ está conectado ao nó $m$;
- Um predicado binário $color$, onde $color(n, x)$ significa que o nó $n$ tem a cor $x$;
- As constantes $yellow$, $green$, $red$.

**Axiomas e Regras**:

1. Cada nó tem no máximo uma cor:

   $$ \forall n \forall x: (color(n, x) \rightarrow \neg \exists y: (y \neq x \land color(n, y))) $$

2. Nós conectados não têm a mesma cor:

   $$ \forall n \forall m \forall x: (edge(n, m) \land color(n, x) \rightarrow \neg color(m, x)) $$

3. O grafo contém apenas dois nós amarelos:

   $$ \exists n \exists n': (color(n, yellow) \land color(n', yellow) \land n \neq n' \land \forall m: (m \neq n \land m \neq n' \rightarrow \neg color(m, yellow))) $$

4. Começando de um nó vermelho, pode-se alcançar um nó verde em no máximo 4 passos:
   Primeiro, definimos a relação de alcançabilidade em até k passos:

   $$ reach_k(n, m, 0) \leftrightarrow n = m $$

   $$ reach_k(n, m, k+1) \leftrightarrow reach_k(n, m, k) \lor \exists x (edge(n, x) \land reach_k(x, m, k)) $$

   Então, a propriedade 3 é expressa como:

   $$ \forall n (color(n, red) \rightarrow \exists m (reach_k(n, m, 4) \land color(m, green))) $$

5. Para cada cor, existe pelo menos um nó com essa cor:

   $$ \forall x \exists n: color(n, x) $$

6. O grafo é composto por $|C|$ subgrafos disjuntos e não vazios, um para cada cor:

   $$ \forall x \exists n: color(n, x) \land $$

   $$ \forall n \exists x: color(n, x) \land $$

   $$ \forall n \forall m \forall x \forall y ((color(n, x) \land color(m, y) \land x \neq y) \rightarrow \neg reach_k(n, m, \infty)) $$

   Onde $\infty$ representa um número suficientemente grande para cobrir todo o grafo.

**Consultas possíveis**:

1. Verificar se dois nós estão conectados:

   - Consulta: $edge(a, b)$;
   - Resposta: **True** se o nó $a$ está conectado ao nó $b$, **False** caso contrário.

2. Verificar a cor de um nó:

   - Consulta: $color(a, x)$;
   - Resposta: **True** se o nó $a$ tem a cor $x$, **False** caso contrário.

3. Verificar se um nó é alcançável a partir de outro em até k passos:

   - Consulta: $reach_k(a, b, k)$;
   - Resposta: **True** se o nó $b$ é alcançável a partir do nó $a$ em até $k$ passos, **False** caso contrário.

4. Contar o número de nós de uma determinada cor:

   - Consulta: $\exists n_1, ..., n_m: (\bigwedge_{i=1}^m color(n_i, x) \land \bigwedge_{i \neq j} n_i \neq n_j \land \forall n: (color(n, x) \rightarrow \bigvee_{i=1}^m n = n_i))$;
   - Resposta: O maior valor de $m$ para o qual esta fórmula é verdadeira é o número de nós da cor $x$.

5. Verificar se o grafo é totalmente colorido:
   - Consulta: $\forall n \exists x: color(n, x)$;
   - Resposta: **True** se todos os nós têm uma cor atribuída, **False** caso contrário.

##### Exercício 3 - Minesweeper [:2]

O jogo **Minesweeper** foi inventado por [Robert Donner](<https://en.wikipedia.org/wiki/Robert_Donner_(disambiguation)>) em 1989. O objetivo do jogo é limpar um campo minado sem detonar uma mina. A tela do jogo consiste em um campo retangular de quadrados. Cada quadrado pode ser limpo, ou descoberto, clicando nele. Se um quadrado contendo uma mina for clicado, o jogo termina. Se o quadrado não contém uma mina, uma das duas coisas acontece: (1) Um número entre 1 e 8 aparece, indicando o número de quadrados adjacentes contendo minas, ou (2) nenhum número aparece; nesse caso, não há minas nas células adjacentes.

Forneça, em uma linguagem de Lógica de Primeira Ordem, um mundo que permita formalizar o conhecimento de um jogador em um estado do jogo. Nessa linguagem, você deve ser capaz de formalizar o seguinte conhecimento:

1. Existem exatamente $n$ minas no campo minado.
2. Se uma célula contém o número 1, então há exatamente uma mina nas células adjacentes.
3. Mostre, por meio de dedução, que deve haver uma mina na posição (3,3) no estado do jogo da figura a seguir.

![]({{ site.baseurl }}/assets/images/mines.webp){: class="lazyimg"}
_Figura 1 - Um estado do jogo Minesweeper._{: class="legend"}

**Solução**:

1. Um predicado unário $mine$, onde $mine(x)$ significa que a célula $x$ contém uma mina;
2. Um predicado binário $adj$, onde $adj(x, y)$ significa que a célula $x$ é adjacente à célula $y$;
3. Um predicado binário $contains$, onde $contains(x, n)$ significa que a célula $x$ contém o número $n$.

**Regras e Axiomas**:

1. Existem exatamente $n$ minas no jogo:

   $$ \exists x*1 \dots \exists x_n \left( \bigwedge*{i=1}^{n} mine(x*i) \land \forall y (mine(y) \rightarrow \bigvee*{i=1}^{n} y = x_i) \right) $$

2. Se uma célula contém o número 1, então há exatamente uma mina nas células adjacentes:

   $$ \forall x: (contains(x, 1) \rightarrow \exists z: (adj(x, z) \land mine(z) \land \forall y: (adj(x, y) \land mine(y) \rightarrow y = z))) $$

3. Mostre por meio de dedução que deve haver uma mina na posição (3,3):

   De acordo com a figura acima, temos:

   1. $contains((2, 2), 1)$;

   2. $\neg mine((1, 1)) \land \neg mine((1, 2)) \land \neg mine((1, 3))$;

   3. $\neg mine((2, 1)) \land \neg mine((2, 2)) \land \neg mine((2, 3))$;

   4. $\neg mine((3, 1)) \land \neg mine((3, 2))$.

      Podemos deduzir:

   5. $\exists z: (adj((2, 2), z) \land mine(z) \land \forall y: (adj((2, 2), y) \land mine(y) \rightarrow y = z))$ (de a e axioma 2)

   6. $mine((1, 1)) \lor mine((1, 2)) \lor mine((1, 3)) \lor mine((2, 1)) \lor mine((2, 2)) \lor mine((2, 3)) \lor mine((3, 1)) \lor mine((3, 2)) \lor mine((3, 3))$ (de e)

   7. $mine((3, 3))$ (de b, c, d e f)

##### Exercício 4 - Conexões Aéreas

Imagine que você é responsável pela gestão de voos entre várias cidades brasileiras. A tarefa envolve criar uma representação formal das conexões aéreas entre essas cidades, considerando diferentes tipos de voos, como voos domésticos e internacionais, e as restrições específicas que regulam essas conexões. O objetivo é formalizar essas conexões de forma que se possa responder a perguntas sobre as rotas disponíveis e as restrições envolvidas.

**Descrição do Problema**:

- **Cidades brasileiras**: representadas como nós de um grafo;
- **Voos diretos**: representados como arestas que conectam duas cidades diretamente (sem escalas intermediárias);
- **Tipos de voos**: diferentes categorias de voos, como domésticos (doméstico) e internacionais (internacional), com restrições sobre onde eles podem operar.
- **Cidades pequenas**: algumas cidades são classificadas como pequenas, e certas restrições se aplicam a essas cidades.

**Solução**:

- As constantes $SP$, $RJ$, $BSB$, $FLN$, $MAO$ são identificadores das cidades São Paulo, Rio de Janeiro, Brasília, Florianópolis, Manaus;
- As constantes $Domestico$, $Internacional$ são os identificadores dos tipos de voo;
- O predicado unário $Aviao(x)$ significa que $x$ é um avião;
- O predicado unário $Cidade(x)$ significa que $x$ é uma cidade;
- O predicado unário $CidadePequena(x)$ significa que $x$ é uma cidade pequena;
- O predicado binário $TipoVoo(x, y)$ significa que o voo $x$ é do tipo $y$;
- O predicado binário $PertenceEstado(x, y)$ significa que a cidade $x$ está no estado $y$;
- O predicado ternário $ConexaoDireta(x, y, z)$ significa que o voo $x$ conecta diretamente as cidades $y$ e $z$ (sem escalas intermediárias).

**Regras e Axiomas**:

1. Um avião tem exatamente um tipo de voo:

   $$ \forall x (Aviao(x) \rightarrow \exists y (TipoVoo(x, y))) \land \forall x y z (TipoVoo(x, y) \land TipoVoo(x, z) \rightarrow y = z) $$

2. O tipo Internacional é diferente do tipo Doméstico:

   $$ \neg (Internacional = Domestico) $$

3. Uma cidade está associada a exatamente um estado:

   $$ \forall x (Cidade(x) \rightarrow \exists y (PertenceEstado(x, y))) \land \forall x y z (PertenceEstado(x, y) \land PertenceEstado(x, z) \rightarrow y = z) $$

4. Cidades pequenas são cidades:

   $$ \forall x (CidadePequena(x) \rightarrow Cidade(x)) $$

5. Se uma cidade $a$ está conectada a uma cidade $b$, então $b$ também está conectada a $a$:

   $$ \forall x y (\exists z ConexaoDireta(z, x, y) \rightarrow \exists z ConexaoDireta(z, y, x)) $$

6. Definição das constantes de cidade:

   $$ Cidade(SP) \land Cidade(RJ) \land Cidade(BSB) \land Cidade(FLN) \land Cidade(MAO) $$

**Axiomas específicos**:

1. Não há conexão direta de São Paulo para Manaus:

   $$ \neg \exists x ConexaoDireta(x, SP, MAO) $$

2. Existe um voo doméstico de São Paulo para Manaus que faz escalas em Brasília, Rio de Janeiro e Florianópolis:

   $$ \exists x (ConexaoDireta(x, SP, BSB) \land ConexaoDireta(x, BSB, RJ) \land ConexaoDireta(x, RJ, FLN) \land ConexaoDireta(x, FLN, MAO) \land TipoVoo(x, Domestico)) $$

3. Voos domésticos conectam cidades brasileiras:

   $$ \forall x y z (TipoVoo(x, Domestico) \rightarrow (ConexaoDireta(x, y, z) \rightarrow (Cidade(y) \land Cidade(z)))) $$

4. Voos internacionais não fazem escalas em cidades pequenas:

   $$ \forall x y z (ConexaoDireta(x, y, z) \land TipoVoo(x, Internacional) \rightarrow \neg CidadePequena(y) \land \neg CidadePequena(z)) $$

**Consultas Possíveis**:

1. **Verificar se há uma conexão direta entre duas cidades**:

   - Consulta: $ConexaoDireta(a, b, c)$;
   - Resposta: **True** se o voo $a$ conecta diretamente as cidades $b$ e $c$, **False** caso contrário.

2. **Verificar o tipo de voo de um avião**:

   - Consulta: $TipoVoo(a, x)$;
   - Resposta: **True** se o avião $a$ opera o tipo de voo $x$, **False** caso contrário.

3. **Verificar se duas cidades estão no mesmo estado**:

   - Consulta: $PertenceEstado(a, b)$;
   - Resposta: **True** se a cidade $a$ está no estado $b$, **False** caso contrário.

4. **Verificar se um voo faz escalas apenas em cidades grandes**:

   - Consulta: $\forall y z (ConexaoDireta(a, y, z) \rightarrow (\neg CidadePequena(y) \land \neg CidadePequena(z)))$;
   - Resposta: **True** se o voo $a$ não faz escalas em cidades pequenas, **False** caso contrário.

5. **Verificar se uma cidade pequena está conectada por um voo**:

   - Consulta: $\exists x (CidadePequena(y) \land ConexaoDireta(x, y, z))$;
   - Resposta: **True** se a cidade pequena $y$ está conectada por um voo a alguma outra cidade, **False** caso contrário.

##### Exercício 5 - Jogo de Damas Brasileiro

O jogo de damas brasileiro é jogado em um tabuleiro de 64 casas (pretas e brancas), onde dois jogadores competem com 12 peças cada (denominadas **comuns**). Um jogador tem peças pretas e o outro, peças brancas. O objetivo do jogo é capturar todas as peças do adversário ou impossibilitar os movimentos do adversário.

Quando o jogo começa, as peças de cada jogador são posicionadas nas 12 casas pretas mais próximas a eles, sendo que as casas brancas não são utilizadas durante o jogo. As peças se movem apenas diagonalmente, permanecendo nas casas pretas. O jogador com peças pretas sempre faz o primeiro movimento.

**Movimentos**:

Existem quatro tipos fundamentais de movimento: o movimento comum de uma peça, o movimento comum de uma dama, o movimento de captura de uma peça e o movimento de captura de uma dama.

- **Movimento comum de uma peça**: A peça é movida diagonalmente para frente, à esquerda ou à direita, para uma casa vazia adjacente;
- **Movimento comum de uma dama**: A dama (uma peça que alcançou a última fileira e foi promovida) pode se mover diagonalmente em qualquer direção (frente, trás, esquerda ou direita);
- **Captura**: Quando uma peça (comum ou dama) tem uma peça adversária adjacente, e a casa imediatamente além está vazia, a peça adversária pode ser capturada ao "pular" sobre ela, removendo-a do tabuleiro. Se uma peça puder realizar capturas múltiplas consecutivas, ela deve fazê-lo.

**Objetivo**:

O jogador vence ao capturar todas as peças do adversário ou ao impossibilitar os movimentos de seu oponente.

**Formalização em Lógica de Primeira Ordem**:

- O predicado unário $square(x)$ significa que $x$ é uma casa do tabuleiro;
- O predicado unário $piece(x)$ significa que $x$ é uma peça;
- O predicado unário $white(x)$ significa que $x$ é branca;
- O predicado unário $black(x)$ significa que $x$ é preta;
- O predicado unário $common(x)$ significa que $x$ é uma peça comum;
- O predicado unário $dama(x)$ significa que $x$ é uma dama;
- O predicado binário $empty(x, t)$ significa que a casa $x$ está vazia no tempo $t$;
- O predicado binário $contain(x, y, t)$ significa que a casa $x$ contém a peça $y$ no tempo $t$;
- O predicado binário $capture(x, y, t)$ significa que a peça $x$ capturou a peça $y$ no tempo $t$;
- O predicado binário $adjacent(x, y)$ significa que as casas $x$ e $y$ são adjacentes;
- O predicado unário $turn(x, t)$ significa que é a vez do jogador $x$ no tempo $t$;
- O predicado binário $lastRow(x, y)$ significa que a casa $x$ está na última fileira para o jogador com cor $y$.

**Regras e Axiomas**:

1. Cada peça é branca ou preta:

   $$ \forall x: (piece(x) \rightarrow (white(x) \lor black(x))) $$

2. Cada peça é uma peça comum ou uma dama:

   $$ \forall x: (piece(x) \rightarrow (common(x) \lor dama(x))) $$

3. As casas brancas estão sempre vazias:

   $$ \forall x: (square(x) \land white(x) \rightarrow \forall t: empty(x, t)) $$

4. Em cada instante do jogo, as casas pretas estão vazias ou contêm uma peça:

   $$ \forall x: (square(x) \land black(x) \rightarrow \forall t: (empty(x, t) \lor \exists y: contain(x, y, t))) $$

5. No início do jogo (instante zero), há exatamente 12 peças brancas e 12 peças pretas no tabuleiro:

   $$ \exists p*1, \dots, p*{12}, q*1, \dots, q*{12}: (\bigwedge\*{i=1}^{12} (piece(p_i) \land white(p_i) \land piece(q_i) \land black(q_i)) \land $$

   $$ \forall x: (piece(x) \land white(x) \rightarrow \bigvee*{i=1}^{12} x = p*i) \land $$

   $$ \forall x: (piece(x) \land black(x) \rightarrow \bigvee\*{i=1}^{12} x = q_i)) $$

6. Movimento de peça comum:

   $$ \forall x, y, p, t: (common(p) \land contain(x, p, t) \land empty(y, t) \land adjacent(x, y) \land turn(color(p), t) \rightarrow contain(y, p, t+1) \land empty(x, t+1)) $$

7. Movimento de dama:

   $$ \forall x, y, p, t: (dama(p) \land contain(x, p, t) \land empty(y, t) \land turn(color(p), t) \rightarrow contain(y, p, t+1) \land empty(x, t+1)) $$

8. Captura:

   $$ \forall x, y, z, p_1, p_2, t: (piece(p_1) \land piece(p_2) \land color(p_1) \neq color(p_2) \land contain(x, p_1, t) \land contain(y, p_2, t) \land empty(z, t) \land adjacent(x, y) \land adjacent(y, z) \land turn(color(p_1), t) \rightarrow capture(p_1, p_2, t) \land contain(z, p_1, t+1) \land empty(x, t+1) \land empty(y, t+1)) $$

9. Promoção a dama:

   $$ \forall x, p, t: (common(p) \land contain(x, p, t) \land lastRow(x, color(p)) \rightarrow dama(p)) $$

10. Vitória:

$$ \forall t: (\neg \exists x: (piece(x) \land white(x) \land contain(y, x, t)) \lor \neg \exists x: (piece(x) \land black(x) \land contain(y, x, t)) \lor $$

$$ \neg \exists x, y: (piece(x) \land contain(y, x, t) \land turn(color(x), t) \land ((\exists z: (empty(z, t) \land adjacent(y, z))) \lor (\exists w, z: (piece(w) \land color(w) \neq color(x) \land contain(z, w, t) \land adjacent(y, z) \land \exists v: (empty(v, t) \land adjacent(z, v)))))) \rightarrow gameOver(t)) $$

**Consultas Possíveis**:

1. **Verificar se uma casa está vazia no tempo $t$**:

   - Consulta: $empty(a, t)$;
   - Resposta: **True** se a casa $a$ está vazia no tempo $t$, **False** caso contrário.

2. **Verificar qual peça está em uma casa no tempo $t$**:

   - Consulta: $contain(a, p, t)$;
   - Resposta: **True** se a peça $p$ está na casa $a$ no tempo $t$, **False** caso contrário.

3. **Verificar se uma peça capturou outra no tempo $t$**:

   - Consulta: $capture(x, y, t)$;
   - Resposta: **True** se a peça $x$ capturou a peça $y$ no tempo $t$, **False** caso contrário.

4. **Verificar o número total de peças de uma cor no tabuleiro**:

   - Consulta: $\exists p_1, \dots, p_n: (\bigwedge_{i=1}^n (piece(p_i) \land color(p_i)) \land \forall x: (piece(x) \land color(x) \rightarrow \bigvee_{i=1}^n x = p_i))$;
   - Resposta: O valor $n$ corresponde ao número total de peças da cor especificada no tabuleiro naquele momento.

5. **Verificar se o jogo terminou**:

   - Consulta: $gameOver(t)$;
   - Resposta: **True** se o jogo terminou no tempo $t$, **False** caso contrário.

6. **Verificar de quem é a vez de jogar**:

   - Consulta: $turn(x, t)$;
   - Resposta: **True** se é a vez do jogador $x$ no tempo $t$, **False** caso contrário.

7. **Verificar se uma peça comum foi promovida a dama**:

   - Consulta: $\exists t_1, t_2: (t_1 < t_2 \land common(p, t_1) \land dama(p, t_2))$;
   - Resposta: **True** se a peça $p$ foi promovida de comum para dama em algum momento do jogo, **False** caso contrário.

##### Exercício 6 - Sudoku

O Sudoku é um jogo de lógica jogado em um tabuleiro de 9x9, que é dividido em 9 regiões menores de 3x3. O objetivo do jogo é preencher todas as 81 casas do tabuleiro com números de 1 a 9, respeitando as seguintes regras:

1. Cada número de 1 a 9 deve aparecer exatamente uma vez em cada linha;
2. Cada número de 1 a 9 deve aparecer exatamente uma vez em cada coluna;
3. Cada número de 1 a 9 deve aparecer exatamente uma vez em cada uma das 9 regiões 3x3.

O jogo começa com algumas casas já preenchidas, e o jogador deve completar as casas restantes de forma a obedecer essas regras.

**Solução**:

- O predicado unário $cell(x)$ significa que $x$ é uma célula do tabuleiro;
- O predicado binário $value(x, v)$ significa que a célula $x$ contém o valor $v$, onde $v$ é um número de $1$ a $9$;
- O predicado binário $inRow(x, r)$ significa que a célula $x$ está na linha $r$, onde $r$ é um número de $1$ a $9$;
- O predicado binário $inColumn(x, c)$ significa que a célula $x$ está na coluna $c$, onde $c$ é um número de $1$ a $9$;
- O predicado binário $inRegion(x, z)$ significa que a célula $x$ está na região $z$, onde $z$ é um número de $1$ a $9$ representando uma das $9$ regiões $3\times 3$.

**Regras e Axiomas**:

1. Cada célula tem exatamente um valor entre $1$ e $9$:

   $$\forall x: (cell(x) \rightarrow \exists! v: (1 \leq v \leq 9 \land value(x, v)))$$

2. Cada linha contém os números de $1$ a $9$ exatamente uma vez:

   $$\forall r \forall v: (1 \leq r \leq 9 \land 1 \leq v \leq 9 \rightarrow \exists! x: (inRow(x, r) \land value(x, v)))$$

3. Cada coluna contém os números de $1$ a $9$ exatamente uma vez:

   $$\forall c \forall v: (1 \leq c \leq 9 \land 1 \leq v \leq 9 \rightarrow \exists! x: (inColumn(x, c) \land value(x, v)))$$

4. Cada região 3x3 contém os números de $1$ a $9$ exatamente uma vez:

   $$\forall z \forall v: (1 \leq z \leq 9 \land 1 \leq v \leq 9 \rightarrow \exists! x: (inRegion(x, z) \land value(x, v)))$$

5. Células na mesma linha não podem ter o mesmo valor:

   $$\forall x_1 \forall x_2 \forall v \forall r: (x_1 \neq x_2 \land value(x_1, v) \land value(x_2, v) \land inRow(x_1, r) \land inRow(x_2, r) \rightarrow \bot)$$

6. Células na mesma coluna não podem ter o mesmo valor:

   $$\forall x_1 \forall x_2 \forall v \forall c: (x_1 \neq x_2 \land value(x_1, v) \land value(x_2, v) \land inColumn(x_1, c) \land inColumn(x_2, c) \rightarrow \bot)$$

7. Células na mesma região não podem ter o mesmo valor:

   $$\forall x_1 \forall x_2 \forall v \forall z: (x_1 \neq x_2 \land value(x_1, v) \land value(x_2, v) \land inRegion(x_1, z) \land inRegion(x_2, z) \rightarrow \bot)$$

8. Cada célula está em exatamente uma linha, uma coluna e uma região:

   $$\forall x: (cell(x) \rightarrow \exists! r \exists! c \exists! z: (inRow(x, r) \land inColumn(x, c) \land inRegion(x, z)))$$

**Consultas Possíveis**:

1. **Verificar se uma célula está preenchida com um determinado valor no tabuleiro**:

   - Consulta: $value(x, v)$;
   - Resposta: **True** se a célula $x$ contém o valor $v$, **False** caso contrário.

2. **Verificar se uma linha contém todos os números de 1 a 9**:

   - Consulta: $\forall v (1 \leq v \leq 9 \rightarrow \exists! x: (inRow(x, r) \land value(x, v)))$;
   - Resposta: **True** se a linha $r$ contém todos os números de 1 a 9, **False** caso contrário.

3. **Verificar se uma coluna contém todos os números de 1 a 9**:

   - Consulta: $\forall v (1 \leq v \leq 9 \rightarrow \exists! x: (inColumn(x, c) \land value(x, v)))$;
   - Resposta: **True** se a coluna $c$ contém todos os números de 1 a 9, **False** caso contrário.

4. **Verificar se uma região 3x3 contém todos os números de 1 a 9**:

   - Consulta: $\forall v (1 \leq v \leq 9 \rightarrow \exists! x: (inRegion(x, z) \land value(x, v)))$;
   - Resposta: **True** se a região $z$ contém todos os números de 1 a 9, **False** caso contrário.

##### Exercício 7 - Torre de Hanói

No jogo **Torre de Hanói**, três postes são dados, e discos de tamanhos diferentes são empilhados no primeiro poste em ordem crescente de tamanho (o menor no topo). O objetivo do jogo é mover todos os discos para o terceiro poste, usando o segundo poste como auxiliar, sob as seguintes condições [^2]:

1. Somente um disco pode ser movido de cada vez;
2. Nenhum disco pode ser colocado sobre um disco menor.

**Regras e Axiomas**:

1. Formalize a regra de que apenas um disco pode ser movido de cada vez;
2. Formalize a regra de que nenhum disco pode ser colocado sobre um disco menor;
3. Formalize a condição de vitória, isto é, todos os discos estão no terceiro poste.

**Solução**:

- O predicado unário $disk(x)$ significa que $x$ é um disco;
- O predicado unário $peg(x)$ significa que $x$ é um poste;
- O predicado ternário $on(x, y, t)$ significa que, no tempo $t$, o disco $x$ está diretamente sobre o disco $y$;
- O predicado ternário $at(x, p, t)$ significa que, no tempo $t$, o disco $x$ está no poste $p$;
- O predicado ternário $move(d, p, t)$ significa que, no tempo $t$, o disco $d$ foi movido para o poste $p$;
- O predicado unário $smallest(x)$ significa que $x$ é o disco de menor tamanho;
- O predicado binário $larger(x, y)$ significa que o disco $x$ é maior que o disco $y$.

**Axiomas**:

1. **Apenas um disco pode ser movido de cada vez**:

   $$\forall t \exists! d \exists p: move(d, p, t)$$

   Este axioma afirma que, para cada tempo $t$, existe exatamente um disco $d$ e um poste $p$ tal que $move(d, p, t)$ é verdadeiro. Isso garante que apenas um disco é movido em cada instante.

2. **Movimento afeta o estado do jogo**:

   $$\forall d \forall p \forall t: (move(d, p, t) \rightarrow at(d, p, t+1))$$

   Se um disco $d$ é movido para o poste $p$ no tempo $t$, então no tempo $t+1$, o disco $d$ está no poste $p$.

3. **Estado dos discos no tempo seguinte**:

   $$\forall d \forall p \forall t: \left[ at(d, p, t+1) \leftrightarrow \left( [at(d, p, t) \land \neg \exists p': move(d, p', t)] \lor move(d, p, t) \right) \right]$$

   Um disco $d$ está no poste $p$ no tempo $t+1$ se ele já estava no poste $p$ no tempo $t$ e não foi movido no tempo $t$, ou se ele foi movido para o poste $p$ no tempo $t$.

4. **Nenhum disco pode ser colocado sobre um disco menor**:

   $$\forall d_1 \forall d_2 \forall t: (on(d_1, d_2, t) \rightarrow larger(d_1, d_2))$$

   Este axioma garante que, em qualquer momento $t$, se o disco $d_1$ está sobre o disco $d_2$, então $d_1$ é maior que $d_2$.

5. **Definição da relação de tamanho entre os discos**:

   - **Irreflexividade**:

   $$\forall x: \neg larger(x, x)$$

   - **Transitividade**:

   $$\forall x \forall y \forall z: (larger(x, y) \land larger(y, z) \rightarrow larger(x, z))$$

   - **Anti-simetria**:

   $$\forall x \forall y: (larger(x, y) \rightarrow \neg larger(y, x))$$

   Estes axiomas definem $larger$ como uma relação de ordem estrita entre os discos.

6. **Condição de vitória: todos os discos estão no terceiro poste**:

   $$\exists t \forall d: (disk(d) \rightarrow at(d, peg_3, t))$$

   Este axioma define a condição de vitória: existe um instante $t$ em que todos os discos estão no terceiro poste ($peg_3$).

7. **Não há movimentos após a vitória**:

   $$\forall t' > t, \forall d, \forall p: \neg move(d, p, t')$$

   Após o tempo $t$ em que a condição de vitória é alcançada, não ocorrem mais movimentos.

8. **Cada disco está em exatamente um poste em cada momento**:

   $$\forall d \forall t: (disk(d) \rightarrow \exists! p: (peg(p) \land at(d, p, t)))$$

   Este axioma garante que cada disco está em exatamente um poste em cada momento do jogo.

9. **Relação entre $on$ e $at$**:

   $$\forall d_1 \forall d_2 \forall p \forall t: (on(d_1, d_2, t) \rightarrow at(d_1, p, t) \land at(d_2, p, t))$$

   Se um disco $d_1$ está sobre um disco $d_2$, ambos estão no mesmo poste $p$ no tempo $t$.

10. **Estrutura de pilha sem ciclos**:

    - **Aciclicidade da relação $on$**:

    $$
    \forall d_1 \forall d_2 \forall t: (on(d_1, d_2, t) \rightarrow \neg on(d_2, d_1, t))
    $$

    _Isto garante que não existem ciclos na relação de "estar sobre"._

11. **Condições para $on$ e a base do poste**:

    - Um disco pode estar diretamente no poste sem nenhum disco abaixo:

    $$
    \forall d \forall p \forall t: \left( at(d, p, t) \land \neg \exists d': on(d, d', t) \right) \rightarrow \text{$d$ está na base ou é o único disco no poste $p$}
    $$

    Este axioma assegura que, se não há nenhum disco abaixo de $d$, então $d$ está na base da pilha ou é o único disco no poste $p$.

**Consultas Possíveis**:

1. **Verificar se um disco está em um determinado poste no tempo $t$**:

   - Consulta: $at(d, p, t)$;
   - Resposta: _Verdadeiro_ se o disco $d$ está no poste $p$ no tempo $t$, _Falso_ caso contrário.

2. **Verificar se um disco está sobre outro no tempo $t$**:

   - Consulta: $on(d_1, d_2, t)$;
   - Resposta: _Verdadeiro_ se o disco $d_1$ está sobre o disco $d_2$ no tempo $t$, _Falso_ caso contrário.

3. **Verificar se o disco $d_1$ é maior que o disco $d_2$**:

   - Consulta: $larger(d_1, d_2)$;
   - Resposta: _Verdadeiro_ se o disco $d_1$ é maior que o disco $d_2$, _Falso_ caso contrário.

4. **Verificar se o jogo foi vencido no tempo $t$**:

   - Consulta: $\forall d: (disk(d) \rightarrow at(d, peg_3, t))$;
   - Resposta: _Verdadeiro_ se todos os discos estão no terceiro poste no tempo $t$, _Falso_ caso contrário.

5. **Verificar se um disco foi movido para um poste em um determinado instante**:

   - Consulta: $move(d, p, t)$;
   - Resposta: _Verdadeiro_ se o disco $d$ foi movido para o poste $p$ no tempo $t$, _Falso_ caso contrário.

##### Exercício 8 - Modelo de Família com Meios-Irmãos

**Variáveis Proposicionais**:

Para pessoas:

- $P_i$: Pessoa $$i$ (na qual, $i$ é um identificador único);
- $H_i$: Pessoa $i$ é homem;
- $M_i$: Pessoa $i$ é mulher.

Para relações:

- $PaiDe(i,j)$: Pessoa $i$ é pai de pessoa $j$;
- $MaeDe(i,j)$: Pessoa $i$ é mãe de pessoa $j$;
- $FilhoDe(i,j)$: Pessoa $i$ é filho de pessoa $j$;
- $FilhaDe(i,j)$: Pessoa $i$ é filha de pessoa $j$;
- $IrmaoDe(i,j)$: Pessoa $i$ é irmão de pessoa $j$;
- $IrmaDe(i,j)$: Pessoa $i$ é irmã de pessoa $j$;
- $MeioIrmaoDe(i,j)$: Pessoa $i$ é meio-irmão de pessoa $j$;
- $MeioIrmaDe(i,j)$: Pessoa $i$ é meia-irmã de pessoa $j$.

**Regras do Modelo**:

1. Cada pessoa é homem ou mulher, mas não ambos:

   $$ \forall i, P_i \rightarrow (H_i \oplus M_i) $$

2. Relações de paternidade e maternidade:

   $$ \forall i,j, PaiDe(i,j) \rightarrow (H_i \land (FilhoDe(j,i) \lor FilhaDe(j,i))) $$

   $$ \forall i,j, MaeDe(i,j) \rightarrow (M_i \land (FilhoDe(j,i) \lor FilhaDe(j,i))) $$

3. Relações de filiação:

   $$ \forall i,j, FilhoDe(i,j) \rightarrow (H_i \land (PaiDe(j,i) \lor MaeDe(j,i))) $$

   $$ \forall i,j, FilhaDe(i,j) \rightarrow (M_i \land (PaiDe(j,i) \lor MaeDe(j,i))) $$

4. Relações de irmandade:

   $$ \forall i,j, IrmaoDe(i,j) \rightarrow (H_i \land \exists k, (PaiDe(k,i) \land PaiDe(k,j)) \land \exists l, (MaeDe(l,i) \land MaeDe(l,j)) \land (i \neq j)) $$

   $$ \forall i,j, IrmaDe(i,j) \rightarrow (M_i \land \exists k, (PaiDe(k,i) \land PaiDe(k,j)) \land \exists l, (MaeDe(l,i) \land MaeDe(l,j)) \land (i \neq j)) $$

5. Relações de meio-irmandade:

   $$ \forall i,j, MeioIrmaoDe(i,j) \rightarrow (H_i \land (((\exists k, PaiDe(k,i) \land PaiDe(k,j)) \oplus (\exists l, MaeDe(l,i) \land MaeDe(l,j))) \land (i \neq j))) $$

   $$ \forall i,j, MeioIrmaDe(i,j) \rightarrow (M_i \land (((\exists k, PaiDe(k,i) \land PaiDe(k,j)) \oplus (\exists l, MaeDe(l,i) \land MaeDe(l,j))) \land (i \neq j))) $$

6. Uma pessoa não pode ser seu próprio pai ou mãe:

   $$ \forall i, \lnot PaiDe(i,i) \land \lnot MaeDe(i,i) $$

7. Uma pessoa não pode ser irmão ou meio-irmão de si mesma:

   $$ \forall i, \lnot IrmaoDe(i,i) \land \lnot IrmaDe(i,i) \land \lnot MeioIrmaoDe(i,i) \land \lnot MeioIrmaDe(i,i) $$

8. Simetria nas relações de irmandade:

   $$ \forall i,j, IrmaoDe(i,j) \leftrightarrow IrmaoDe(j,i) $$

   $$ \forall i,j, IrmaDe(i,j) \leftrightarrow IrmaDe(j,i) $$

   $$ \forall i,j, MeioIrmaoDe(i,j) \leftrightarrow MeioIrmaoDe(j,i) $$

   $$ \forall i,j, MeioIrmaDe(i,j) \leftrightarrow MeioIrmaDe(j,i) $$

9. Uma pessoa não pode ser simultaneamente irmão e meio-irmão de outra:

   $$ \forall i,j, \lnot(IrmaoDe(i,j) \land MeioIrmaoDe(i,j)) \land \lnot(IrmaDe(i,j) \land MeioIrmaDe(i,j)) $$

Neste caso podemos definir um dos estados do mundo: para representar que $P1$ é pai de $P2$ e $P3$, $P4$ é mãe de $P2$, $P5$ é mãe de $P3$, e $P2$ e $P3$ são meios-irmãos:

$$
\begin{align*}
P1 \land P2 \land P3 \land P4 \land P5 \land \\
H_1 \land H_2 \land H_3 \land M_4 \land M_5 \land \\
PaiDe(1,2) \land PaiDe(1,3) \land \\
MaeDe(4,2) \land MaeDe(5,3) \land \\
FilhoDe(2,1) \land FilhoDe(2,4) \land \\
FilhoDe(3,1) \land FilhoDe(3,5) \land \\
MeioIrmaoDe(2,3) \land MeioIrmaoDe(3,2)
\end{align*}
$$

**Consultas Possíveis**::

1. **Verificar se uma pessoa existe no mundo**:

   - Consulta: $P_i$;
   - Resposta: Verdadeiro se a pessoa i existe no mundo, Falso caso contrário.

2. **Verificar o sexo de uma pessoa**:

   - Consulta: $H_i$ ou $M_i$;
   - Resposta: Verdadeiro se a pessoa i é homem (H_i) ou mulher (M_i), Falso caso contrário.

3. **Verificar relação de paternidade**:

   - Consulta: $PaiDe(i,j)$;
   - Resposta: Verdadeiro se a pessoa i é pai da pessoa j, Falso caso contrário.

4. **Verificar relação de maternidade**:

   - Consulta: $MaeDe(i,j)$;
   - Resposta: Verdadeiro se a pessoa i é mãe da pessoa j, Falso caso contrário.

5. **Verificar se duas pessoas são irmãos**:

   - Consulta: $IrmaosDe(i,j)$;
   - Resposta: Verdadeiro se as pessoas i e j são irmãos (mesmo pai e mesma mãe), Falso caso contrário.

6. **Verificar se duas pessoas são meios-irmãos**:

   - Consulta: $MeiosIrmaosDe(i,j)$;
   - Resposta: Verdadeiro se as pessoas i e j são meios-irmãos (mesmo pai OU mesma mãe, mas não ambos), Falso caso contrário.

7. **Encontrar o pai de uma pessoa**:

   - Consulta: $\exists x, PaiDe(x,i)$;
   - Resposta: Verdadeiro se existe um pai para a pessoa i, Falso caso contrário;
   - Para obter o pai específico: $x$ tal que $PaiDe(x,i)$ é verdadeiro.

8. **Encontrar a mãe de uma pessoa**:

   - Consulta: $\exists x, MaeDe(x,i)$;
   - Resposta: Verdadeiro se existe uma mãe para a pessoa i, Falso caso contrário;
   - Para obter a mãe específica: $x$ tal que $MaeDe(x,i)$ é verdadeiro.

9. **Verificar se duas pessoas têm o mesmo pai**:

   - Consulta: $\exists x, (PaiDe(x,i) \land PaiDe(x,j))$;
   - Resposta: Verdadeiro se as pessoas i e j têm o mesmo pai, Falso caso contrário.

10. **Verificar se duas pessoas têm a mesma mãe**:

    - Consulta: $\exists x, (MaeDe(x,i) \land MaeDe(x,j))$;
    - Resposta: Verdadeiro se as pessoas i e j têm a mesma mãe, Falso caso contrário.

11. **Contar o número de filhos de uma pessoa**:

    - Consulta: $\text{Contagem}(\{j : PaiDe(i,j) \lor MaeDe(i,j)\})$;
    - Resposta: O número de filhos da pessoa $i$.

12. **Verificar se uma pessoa é filho único**:

    - Consulta: $\lnot \exists j, (j \neq i \land (IrmaosDe(i,j) \lor MeiosIrmaosDe(i,j)))$;
    - Resposta: Verdadeiro se a pessoa i não tem irmãos nem meios-irmãos, Falso caso contrário.

##### Exercício 9 - Jogo Pedra, Papel e Tesoura

O jogo **Pedra, Papel e Tesoura** é um jogo simples entre dois jogadores, onde cada jogador escolhe uma das três opções: Pedra, Papel ou Tesoura. As regras são:

**Variáveis Proposicionais**:

Para jogadas:

- $P_i$: Jogador i escolheu Pedra;
- $A_i$: Jogador i escolheu Papel;
- $T_i$: Jogador i escolheu Tesoura.

Para resultados:

- $V_i$: Jogador i venceu;
- $E$: O jogo terminou em empate.

**Regras do Mundo**:

1. Cada jogador faz exatamente uma jogada:

   $$ \forall i, ((P_i \lor A_i \lor T_i) \land \lnot(P_i \land A_i) \land \lnot(P_i \land T_i) \land \lnot(A_i \land T_i)) $$

2. Condições de vitória para o Jogador $1$:

   $$ V_1 \leftrightarrow ((P_1 \land T_2) \lor (T_1 \land A_2) \lor (A_1 \land P_2)) $$

3. Condições de vitória para o Jogador $2$:

   $$ V_2 \leftrightarrow ((P_2 \land T_1) \lor (T_2 \land A_1) \lor (A_2 \land P_1)) $$

4. Condição de empate:

   $$ E \leftrightarrow ((P_1 \land P_2) \lor (A_1 \land A_2) \lor (T_1 \land T_2)) $$

5. O jogo tem exatamente um resultado:

   $$ (V_1 \lor V_2 \lor E) \land \lnot(V_1 \land V_2) \land \lnot(V_1 \land E) \land \lnot(V_2 \land E) $$

6. Não é possível que ambos os jogadores vençam:

   $$ \lnot(V_1 \land V_2) $$

**Consultas Possíveis**:

1. **Verificar a jogada de um jogador**:

   - Consulta: $P_i$, $A_i$, ou $T_i$;
   - Resposta: Verdadeiro se o Jogador i escolheu a jogada correspondente, Falso caso contrário.

2. **Verificar o vencedor**:

   - Consulta: $V_1$ ou $V_2$;
   - Resposta: Verdadeiro se o Jogador correspondente venceu, Falso caso contrário.

3. **Verificar se houve empate**:

   - Consulta: $E$;
   - Resposta: Verdadeiro se o jogo terminou em empate, Falso caso contrário.

4. **Determinar o resultado do jogo**:

   - Consulta:

     $$
     resultado = \begin{cases}
       1 & \text{se } V_1 \\
       2 & \text{se } V_2 \\
       0 & \text{se } E
     \end{cases}
     $$

   - Resposta:

     - $0$ se o jogo terminou em empate;
     - $1$ se o Jogador 1 venceu;
     - $2$ se o Jogador 2 venceu.

5. **Verificar se um jogador escolheu uma jogada específica e venceu**:

   - Consulta: $(P_i \land V_i)$, $(A_i \land V_i)$, ou $(T_i \land V_i)$.
   - Resposta: Verdadeiro se o Jogador i escolheu a jogada específica e venceu, Falso caso contrário.

6. **Verificar se o jogo foi válido**:

   - Consultas:

     $$((P_1 \lor A_1 \lor T_1) \land \lnot(P_1 \land A_1) \land \lnot(P_1 \land T_1) \land \lnot(A_1 \land T_1)) \land$$

     $$((P_2 \lor A_2 \lor T_2) \land \lnot(P_2 \land A_2) \land \lnot(P_2 \land T_2) \land \lnot(A_2 \land T_2)) \land$$

     $$((V_1 \lor V_2 \lor E) \land \lnot(V_1 \land V_2) \land \lnot(V_1 \land E) \land \lnot(V_2 \land E))$$

   - Resposta: verdadeiro se o jogo seguiu todas as regras (uma jogada por jogador e um único resultado), Falso caso contrário.

**Exemplo de um estado válido deste Mundo**:

$$P_1 \land T_2 \land V_1 \land \lnot V_2 \land \lnot E \land \\
   \lnot A_1 \land \lnot T_1 \land \lnot P_2 \land \lnot A_2$$

Este mundo representa um jogo onde:

- O Jogador $1$ escolheu Pedra;
- O Jogador $2$ escolheu Tesoura;
- O Jogador $1$ venceu;
- Não houve empate.

##### Exercício 10 - Mundo de Ginásio de Esportes

Elabore um mundo para um ginásio de esportes. O modelo deve incluir atletas, modalidades esportivas, treinadores, e competições. Considere que um atleta pode praticar múltiplas modalidades, um treinador pode especializar-se em uma ou mais modalidades, e uma competição envolve uma modalidade específica com vários atletas participantes. Crie consultas para responder se algum atleta pratica todas as modalidades, se algum treinador é especializado em todas as modalidades e mais duas a seu critério.

**Fatos**:

- $A(x)$: $x$ é um atleta;
- $M(x)$: $x$ é uma modalidade esportiva;
- $T(x)$: $x$ é um treinador;
- $C(x)$: $x$ é uma competição;
- $Pratica(x,y)$: atleta $x$ pratica a modalidade $y$;
- $Especializa(x,y)$: treinador $x$ é especializado na modalidade $y$;
- $Participa(x,y)$: atleta $x$ participa da competição $y$;
- $EnvolveModalidade(x,y)$: competição $x$ envolve a modalidade $y$.

**Regras**:

1. Todo atleta pratica pelo menos uma modalidade:

   $$ \forall x(A(x) \rightarrow \exists y(M(y) \land Pratica(x,y))) $$

2. Todo treinador é especializado em pelo menos uma modalidade:

   $$ \forall x(T(x) \rightarrow \exists y(M(y) \land Especializa(x,y))) $$

3. Toda competição envolve exatamente uma modalidade:

   $$ \forall x(C(x) \rightarrow \exists! y(M(y) \land EnvolveModalidade(x,y))) $$

4. Um atleta só pode participar de uma competição se praticar a modalidade envolvida:

   $$ \forall x \forall y(Participa(x,y) \rightarrow \exists z(M(z) \land Pratica(x,z) \land EnvolveModalidade(y,z))) $$

**Consultas**:

1. Verificar se um atleta pratica uma modalidade específica:

   - Consulta: `Pratica(atleta,modalidade)`;
   - Resposta: Verdadeiro se o atleta pratica a modalidade, Falso caso contrário.

2. Verificar se um treinador é especializado em uma modalidade específica:

   - Consulta: `Especializa(treinador,modalidade)`;
   - Resposta: Verdadeiro se o treinador é especializado na modalidade, Falso caso contrário.

3. Verificar se um atleta participa de uma competição específica:

   - Consulta: `Participa(atleta,competicao)`;
   - Resposta: Verdadeiro se o atleta participa da competição, Falso caso contrário.

4. Verificar se uma competição envolve uma modalidade específica:

   - Consulta: `EnvolveModalidade(competicao,modalidade)`;
   - Resposta: Verdadeiro se a competição envolve a modalidade, Falso caso contrário.

5. Verificar se existe um atleta que pratica todas as modalidades:

   - Consulta: $\exists x(A(x) \land \forall y(M(y) \rightarrow Pratica(x,y)))$;
   - Resposta: Verdadeiro se existe um atleta que pratica todas as modalidades, Falso caso contrário.

6. Verificar se existe um treinador especializado em todas as modalidades:

   - Consulta: $\exists x(T(x) \land \forall y(M(y) \rightarrow Especializa(x,y)))$;
   - Resposta: Verdadeiro se existe um treinador especializado em todas as modalidades, Falso caso contrário.

7. Verificar se existe uma modalidade praticada por todos os atletas:

   - Consulta: $\exists y(M(y) \land \forall x(A(x) \rightarrow Pratica(x,y)))$;
   - Resposta: Verdadeiro se existe uma modalidade praticada por todos os atletas, Falso caso contrário.

8. Verificar se existe uma competição em que todos os atletas participam:

   - Consulta: $\exists y(C(y) \land \forall x(A(x) \rightarrow Participa(x,y)))$;
   - Resposta: Verdadeiro se existe uma competição com participação de todos os atletas, Falso caso contrário.

9. Verificar se um atleta está qualificado para participar de uma competição específica:

   - Consulta: $\exists z(M(z) \land Pratica(atleta,z) \land EnvolveModalidade(competicao,z))$;
   - Resposta: Verdadeiro se o atleta pratica a modalidade envolvida na competição, Falso caso contrário.

10. Verificar se existe um treinador especializado na modalidade de uma competição específica:
    - Consulta: $\exists x \exists y(T(x) \land M(y) \land Especializa(x,y) \land EnvolveModalidade(competicao,y))$;
    - Resposta: Verdadeiro se existe um treinador especializado na modalidade da competição, Falso caso contrário.

## Cláusulas de Horn

A **Cláusula de Horn** foi nomeada em homenagem ao matemático e lógico americano [Alfred Horn](https://en.wikipedia.org/wiki/Alfred_Horn), que a introduziu em [um artigo publicado em 1951](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/on-sentences-which-are-true-of-direct-unions-of-algebras1/DF348CB269B06D6702DA3AE4DCF38C39). O contexto histórico e a motivação para a introdução da Cláusula de Horn são profundamente enraizados na solução do Problema da Decidibilidade. Na primeira metade do século XX, a lógica matemática estava focada na questão da decidibilidade: determinar se uma afirmação lógica é verdadeira ou falsa de forma algorítmica.

Não demorou muito para os matemáticos perceberem que a Lógica de Primeira Ordem é poderosa, mas pode ser ineficientes para resolver os problemas relacionados ao Problema da Decidibilidade. A busca por formas mais eficientes de resolução levou ao estudo de subconjuntos restritos da Lógica de Primeira Ordem, onde a decidibilidade poderia ser alcançada de forma mais eficiente. Aqui, eficiência significa o menor custo computacional, no menor tempo.

Alfred Horn identificou um desses subconjuntos em seu artigo de 1951, introduzindo o que agora é conhecido como **Cláusula de Horn**. Ele mostrou que esse subconjunto particular tem propriedades interessantes que permitem a resolução em tempo polinomial, tornando-o atraente para aplicações práticas.

Se prepare vamos ver porque $P \lor \neg Q \lor \neg R $ é uma Cláusula de Horn e $P \lor Q \lor \neg R$ não é.

### Definição da Cláusula de Horn

A **Cláusula de Horn** é uma disjunção de literais que contém, no máximo, um literal positivo.
Existem algumas formas equivalentes de representar Cláusulas de Horn:

1.  **Forma Disjuntiva**:
    $\neg A_1 \lor \neg A_2 \lor \ldots \lor \neg A_k \lor B$
    Aqui, $A_1, \ldots, A_k, B$ são proposições atômicas (átomos). Os literais $\neg A_i$ são negativos, e $B$ é o único literal positivo (se presente). Se não houver literal positivo, a cláusula é uma meta ou consulta. Se não houver literais negativos, é um fato.

2.  **Forma Implicativa (mais comum em programação lógica)**:
    $(A_1 \land A_2 \land \ldots \land A_k) \rightarrow B$
    Esta forma é equivalente à disjuntiva acima.
    * $A_1, \ldots, A_k$ são os átomos no corpo (antecedente) da implicação.
    * $B$ é o átomo na cabeça (consequente) da implicação.

**Tipos de Cláusulas de Horn (baseado na forma disjuntiva $\neg A_1 \lor \ldots \lor \neg A_k \lor B$)**:

* **Fatos (Cláusulas Unitárias Positivas)**: Têm exatamente um literal positivo e nenhum literal negativo ($k=0$).

   * Forma disjuntiva: $B$
   * Forma implicativa: $\top \rightarrow B$ (ou simplesmente $B.$ em Prolog)
   * Exemplo: $Mortal(socrates)$.

* **Regras (Cláusulas Definidas)**: Têm exatamente um literal positivo ($B$) e um ou mais literais negativos ($\neg A_i$, com $k \ge 1$).

    * Forma disjuntiva: $\neg A_1 \lor \ldots \lor \neg A_k \lor B$
    * Forma implicativa: $(A_1 \land \ldots \land A_k) \rightarrow B$ (ou $B \leftarrow A_1, \ldots, A_k.$ em Prolog)
    * Exemplo: $Homem(x) \rightarrow Mortal(x)$ (Cláusula: $\neg Homem(x) \lor Mortal(x)$).

* **Metas ou Consultas (Cláusulas Negativas)**: Não têm nenhum literal positivo (a parte $B$ está ausente, ou é $\bot$ - falso), e um ou mais literais negativos ($k \ge 1$).
    * Forma disjuntiva: $\neg A_1 \lor \ldots \lor \neg A_k$
    * Forma implicativa: $(A_1 \land \ldots \land A_k) \rightarrow \bot$ (ou $\leftarrow A_1, \ldots, A_k.$ em Prolog)
    * Exemplo: $\leftarrow Mortal(socrates)$ (Cláusula: $\neg Mortal(socrates)$).

* **Cláusula Nula (ou Vazia)**: Representa uma contradição. Não possui literais. Surge quando uma consulta é refutada.
    * Forma disjuntiva: $\Box$ ou $\bot$

Para entender melhor, imagine que estamos construindo um cenário mental fundamentado na lógica para construir o entendimento de um problema, uma espécie de paisagem mental onde as coisas fazem sentido. Nesse cenário, as Cláusulas de Horn serão os tijolos fundamentais que usaremos para construir estruturas lógicas.

**1. Fatos**: os fatos são como pedras fundamentais desse cenário. Eles são afirmações simples e diretas que dizem como as coisas são. Considere, por exemplo: _O céu é azul_, $P$ e _A grama é verde_$Q$. Essas são verdades que não precisam de justificativa. Elas simplesmente são. os Fatos são axiomas.

**2. Regras**: as regras são um pouco mais intrigantes. Elas são como as regras de um jogo que definem como as coisas se relacionam umas com as outras. _Se não chover, a grama não ficará molhada._ Essa é uma regra. Ela nos diz o que esperar se certas condições forem atendidas. As regras são como os conectores em nosso mundo lógico, ligando fatos e permitindo que façamos inferências. Elas são o motor que nos permite raciocinar e descobrir novas verdades a partir das que já conhecemos. Por exemplo:

- $\neg P \land \neg Q \rightarrow R$: _Se não chover, $P$ e não ventar, $Q$, então faremos um piquenique, $R$_.
- $\neg A \land \neg B \land \neg C \rightarrow D$: _Se $A$, $B$ e $C$ forem falsos, então $D$ é verdadeiro_.

**3. Metas ou Consultas**: finalmente, temos as metas ou consultas. Essas são as perguntas que fazemos ao nosso mundo lógico. _Está chovendo?_, _A grama está molhada?_ São os caminhos que usaremos para explorar o cenário criado, olhando ao redor e tentando entender o que está acontecendo. As consultas são a forma de interagir com nosso mundo lógico, usando os fatos e regras que estabelecemos para encontrar respostas e alcançar objetivos. Por exemplo:

- $\neg P \land \neg Q$: _É verdade que hoje não está chovendo e não está ventando?_
- $\neg X \land \neg Y \land \neg Z$: _$x$, $Y$ e $Z $ são falsos?_

Podemos tentar avaliar alguns exemplos de uso de Fatos, Regras e Consultas:

#### Exemplo 1 - Sistema de Recomendação de Roupas

Imagine que estamos construindo um sistema lógico para recomendar o tipo de roupa que uma pessoa deve vestir com base no clima. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**1. Fatos**: primeiro, estabelecemos os fatos, as verdades básicas do cenário que descreve nosso problema. Neste exemplo, os fatos poderiam ser informações sobre o clima atual.

- **Fato 1**: Está ensolarado. (Representado como $s$);
- **Fato 2**: A temperatura está acima de 20°C. (Representado como $T$).

Você pode criar todos os fatos necessários a descrição do seu problema.

**2. Regras**: em seguida, definimos as regras que descrevem como as coisas se relacionam. Essas regras nos dizem o tipo de roupa apropriada com base no clima.

- **Regra 1**: Se está ensolarado e a temperatura está acima de 20°C, use óculos de sol. ($\neg S \land \neg T \rightarrow O $);
- **Regra 2**: Se está ensolarado, use chapéu. ($\neg S \rightarrow C$);
- **Regra 3**: Se a temperatura está acima de 20°C, use camiseta. ($\neg T \rightarrow A$).

Você pode criar todas as regras que achar importante para definir o comportamento no cenário que descreve o problema.

**3. Consultas (Metas)**: agora, podemos fazer consultas ao nosso sistema para obter recomendações de roupas.

- **Consulta 1**: Está ensolarado e a temperatura está acima de 20°C. O que devo vestir? ($\neg S \land \neg T$);

As consultas representam todas as consultas que podem ser feitas neste cenário. A esforçada leitora deve criar quantas consultas achar necessário para entender o problema.

**4. Resolução**: usando os fatos e regras, podemos resolver a consulta:

1. Está ensolarado e a temperatura está acima de 20°C (_Fato_);
2. Portanto, use óculos de sol (_Regra 1_);
3. Portanto, use chapéu (_Regra 2_);
4. Portanto, use camiseta (_Regra 3_).

Neste exemplo, as Cláusulas de Horn nos permitiram representar o conhecimento sobre o clima e as regras para escolher roupas. Os fatos forneceram a base de conhecimento, as regras permitiram inferências lógicas, e a consulta nos permitiu explorar o sistema para obter recomendações práticas.

#### Exemplo 2 - Sistema de Diagnóstico Médico

Imagine que estamos construindo um sistema lógico para diagnosticar doenças com base em sintomas, histórico médico e outros fatores relevantes. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**1. Fatos**: começamos estabelecemos os fatos, que são as informações conhecidas sobre o paciente.

- **Fato 1**: O paciente tem febre. (Representado como $F$);
- **Fato 2**: O paciente tem tosse. (Representado como $T$);
- **Fato 3**: O paciente viajou recentemente para uma área endêmica. (Representado como $V$);
- **Fato 4**: O paciente foi vacinado contra a gripe. (Representado como $ g$).

**2. Regras**: em seguida, definimos as regras que descrevem as relações entre sintomas, histórico médico e possíveis doenças.

- **Regra 1**: Se o paciente tem febre e tosse, mas foi vacinado contra a gripe, então pode ter resfriado comum. ($\neg F \land \neg T \land G \rightarrow R$);
- **Regra 2**: Se o paciente tem febre, tosse e viajou para uma área endêmica, então pode ter malária. ($\neg F \land \neg T \land \neg V \rightarrow M $);
- **Regra 3**: Se o paciente tem febre e tosse, mas não foi vacinado contra a gripe, então pode ter gripe. ($\neg F \land \neg T \land \neg G \rightarrow I $).

**3. Consultas**: agora, podemos fazer consultas ao nosso sistema para obter diagnósticos possíveis.

- **Consulta 1**: O paciente tem febre, tosse, viajou para uma área endêmica e foi vacinado contra a gripe. Qual é o diagnóstico? ($\neg F \land \neg T \land \neg V \land G$);

**4. Resolução**: usando os fatos e regras, podemos resolver a consulta:

1. O paciente tem febre, tosse, viajou para uma área endêmica e foi vacinado contra a gripe (_Fatos_);
2. Portanto, o paciente pode ter resfriado comum (_Regra 1_);
3. Portanto, o paciente pode ter malária (_Regra 2_).

**5. Conclusão**: este exemplo ilustra como as Cláusulas de Horn podem ser usadas em um contexto mais complexo, como um sistema de diagnóstico médico. A mesma abordagem pode ser aplicada a outros domínios, como diagnósticos de falhas em máquinas, sistemas legais, planejamento financeiro e muito mais.

## Exemplo 3 - Mundo Núcleo Familiar (Lógica de Primeira Ordem)

O exemplo a seguir apresenta um mundo que representa uma família e suas relações, apresentado usando a sintaxe da lógica de primeira ordem (FOL).

**Fatos**: os fatos são representados como predicados aplicados a constantes em FOL.

- $Homem(joão)$;
- $Homem(pedro)$;
- $Mulher(maria)$;
- $Mulher(ana)$;
- $Progenitor(joão, pedro)$;
- $Progenitor(maria, pedro)$;
- $Progenitor(joão, ana)$;
- $Progenitor(maria, ana)$.

**Regras**:

**1. Pai**:

* Forma Implicativa:

    $$\forall X \forall Y (Homem(X) \land Progenitor(X, Y) \rightarrow Pai(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y (\neg Homem(X) \lor \neg Progenitor(X, Y) \lor Pai(X, Y))$$

**2. Mãe**:

* Forma Implicativa:

    $$\forall X \forall Y (Mulher(X) \land Progenitor(X, Y) \rightarrow Mae(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y (\neg Mulher(X) \lor \neg Progenitor(X, Y) \lor Mae(X, Y))$$

**3. Meio-Irmão**: com pelo menos um progenitor em comum e não sendo a mesma pessoa.

* Forma Implicativa:

    $$\forall X \forall Y \forall Z (Homem(X) \land Progenitor(Z, X) \land Progenitor(Z, Y) \land X \neq Y \rightarrow MeioIrmao(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall Z (\neg Homem(X) \lor \neg Progenitor(Z, X) \lor \neg Progenitor(Z, Y) \lor X = Y \lor MeioIrmao(X, Y))$$

**4. Meio-Irmã**: com pelo menos um progenitor em comum e não sendo a mesma pessoa.

* Forma Implicativa:

    $$\forall X \forall Y \forall Z (Mulher(X) \land Progenitor(Z, X) \land Progenitor(Z, Y) \land X \neq Y \rightarrow MeioIrma(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall Z (\neg Mulher(X) \lor \neg Progenitor(Z, X) \lor \neg Progenitor(Z, Y) \lor X = Y \lor MeioIrma(X, Y))$$

**5. Irmão**: com ambos os pais em comum e não sendo a mesma pessoa.

* Forma Implicativa:

    $$\forall X \forall Y \forall P \forall M (Homem(X) \land Pai(P,X) \land Pai(P,Y) \land Mae(M,X) \land Mae(M,Y) \land X \neq Y \rightarrow Irmao(X,Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall P \forall M (\neg Homem(X) \lor \neg Pai(P,X) \lor \neg Pai(P,Y) \lor \neg Mae(M,X) \lor \neg Mae(M,Y) \lor X = Y \lor Irmao(X,Y))$$

**6. Irmã**: com ambos os pais em comum e não sendo a mesma pessoa.

* Forma Implicativa:

    $$\forall X \forall Y \forall P \forall M (Mulher(X) \land Pai(P,X) \land Pai(P,Y) \land Mae(M,X) \land Mae(M,Y) \land X \neq Y \rightarrow Irma(X,Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall P \forall M (\neg Mulher(X) \lor \neg Pai(P,X) \lor \neg Pai(P,Y) \lor \neg Mae(M,X) \lor \neg Mae(M,Y) \lor X = Y \lor Irma(X,Y))$$

**7. Avô**:

* Forma Implicativa:

    $$\forall X \forall Y \forall Z (Homem(X) \land Progenitor(X, Z) \land Progenitor(Z, Y) \rightarrow Avo(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall Z (\neg Homem(X) \lor \neg Progenitor(X, Z) \lor \neg Progenitor(Z, Y) \lor Avo(X, Y))$$

**8. Avó**:

* Forma Implicativa:

    $$\forall X \forall Y \forall Z (Mulher(X) \land Progenitor(X, Z) \land Progenitor(Z, Y) \rightarrow Avo(X, Y))$$

* Forma Clausal:

    $$\forall X \forall Y \forall Z (\neg Mulher(X) \lor \neg Progenitor(X, Z) \lor \neg Progenitor(Z, Y) \lor Avo(X, Y))$$

**Consultas (Metas)**:

1. `pai(joão, pedro)`

   Para verificar se $Pai(joão, pedro)$ é uma consequência lógica da base de conhecimento, tenta-se provar que a base de conhecimento junto com $\neg Pai(joão, pedro)$ leva a uma contradição ($\bot$). A meta é, portanto:
   $$\neg Pai(joão, pedro)$$

2. `irmão(pedro, ana)`

   Para verificar se $Irmao(pedro, ana)$ é verdadeiro:
   $$\neg Irmao(pedro, ana)$$

3. `avó(X, ana)`

   Para perguntar se "Existe uma avó X para Ana?", a consulta seria $\exists X (Avo(X, ana) \land Mulher(X))$.
   A forma de meta para refutação seria tentar provar que a base de conhecimento junto com $\forall X (\neg Avo(X, ana) \lor \neg Mulher(X))$ leva a uma contradição.

Em um sistema de prova por refutação, adicionamos a negação da consulta à base de conhecimento e tentamos derivar uma contradição ($\bot$). As representações das metas como negações em FOL estão corretas nesse contexto.
#### Exemplo 4 - Torre de Hanói

A **Torre de Hanói** é um quebra-cabeça matemático que consiste em três postes e um número de discos de diferentes tamanhos que podem deslizar sobre qualquer poste. O quebra-cabeça começa com os discos empilhados em ordem decrescente de tamanho no primeiro poste, o menor disco no topo. O objetivo é mover toda a pilha para o último poste, obedecendo às seguintes regras:

**Predicados**:

- $Disco(x)$: $x$ é um disco;
- $Poste(x)$: $x$ é um poste;
- $Menor(x)$: $x$ é o disco menor;
- $Maior(x, y)$: o disco $x$ é maior que o disco $y$;
- $Em(x, y)$: o disco $x$ está no poste $y$;
- $Sobre(x, y)$: o disco $x$ está sobre o disco $y$.

**Fatos (Cláusulas de Horn Unitárias)**:

1. $Disco(d_1)$;
2. $Disco(d_2)$;
3. $Disco(d_3)$;
4. $Poste(p_1)$;
5. $Poste(p_2)$;
6. $Poste(p_3)$;
7. $Menor(d_1)$;
8. $Maior(d_2, d_1)$;
9. $Maior(d_3, d_2)$.

**Regras (Cláusulas de Horn Não-Unitárias)**:

1. Movimento válido:

   $$\neg Disco(x) \lor \neg Poste(y) \lor \neg Poste(z) \lor \neg Em(x, y) \lor \neg DiscoNoTopo(x, y) \lor \neg DiscoNoTopo(u, z) \lor \neg Maior(x, u) \lor MovimentoValido(x, y, z)$$

2. Condição de vitória:

   $$\neg Disco(x) \lor \neg Disco(y) \lor \neg Disco(z) \lor \neg Em(x, p_3) \lor \neg Em(y, p_3) \lor \neg Em(z, p_3) \lor Vitoria()$$

3. Disco válido (nenhum disco maior sobre um menor):

   $$\neg Sobre(x, y) \lor \neg Maior(x, y) \lor DiscoValido(x, y)$$

4. Movimento único:

   $$\neg Disco(x) \lor \neg Disco(y) \lor \neg Poste(z) \lor \neg Poste(w) \lor \neg MovimentoValido(y, z, w) \lor x = y \lor MovimentoUnico(x)$$

5. Estado inicial:

   $$\neg Em(d_1, p_1) \lor \neg Em(d_2, p_1) \lor \neg Em(d_3, p_1) \lor \neg Sobre(d_3, d_2) \lor \neg Sobre(d_2, d_1) \lor EstadoInicial()$$

6. Disco no topo:

   $$\neg Disco(x) \lor \neg Poste(y) \lor \neg Em(x, y) \lor \neg Disco(z) \lor \neg Em(z, y) \lor \neg Sobre(z, x) \lor DiscoNoTopo(x, y)$$

**Consultas (Metas)**:

1. Verificar se um movimento é válido:

   $$\neg MovimentoValido(x, y, z)$$

2. Verificar se o jogo foi vencido:

   $$\neg Vitoria()$$

3. Verificar se um disco pode estar sobre outro:

   $$\neg DiscoValido(x, y)$$

4. Verificar se apenas um disco está sendo movido:

   $$\neg MovimentoUnico(x)$$

5. Verificar o estado inicial:

   $$\neg EstadoInicial()$$

6. Verificar se um disco está no topo de um poste:

   $$\neg DiscoNoTopo(x, y)$$

### Quantificadores em Cláusulas de Horn

Os quantificadores podem ser incluídos nas Cláusulas de Horn. Contudo, é importante notar que a forma padrão de Cláusulas de Horn em programação lógica geralmente lida com quantificação de forma implícita. A quantificação universal é comum e é geralmente assumida em regras, enquanto a quantificação existencial é muitas vezes tratada através de fatos específicos ou construção de termos.

Precisamos tomar cuidado porque a inclusão explícita de quantificadores pode levar a uma Lógica de Primeira Ordem mais rica, permitindo expressões mais complexas e poderosas. No entanto, isso também pode aumentar a complexidade do raciocínio e da resolução.

O quantificador universal (representado por $\forall $) afirma que uma propriedade é verdadeira para todos os membros de um domínio. Em Cláusulas de Horn, isso é geralmente representado implicitamente através de regras gerais que se aplicam a todos os membros de um conjunto. Por exemplo, considere a regra: _Todos os pássaros podem voar_. Em uma Cláusula de Horn, isso pode ser representado como:

Em programação lógica e Cláusulas de Horn, a quantificação é frequentemente tratada implicitamente.

**Quantificador Universal em Regras**: considere a afirmação: "_Para todo x, se x é um pássaro, então x pode voar_."

* **Fórmula em Lógica de Primeira Ordem (LPO)**: $\forall x (\text{Pássaro}(x) \rightarrow \text{Voa}(x))$
* **Forma clausal (Cláusula de Horn equivalente)**: $\neg \text{Pássaro}(x) \lor \text{Voa}(x)$
    (Aqui, $x$ é implicitamente quantificado universalmente.)
* **Representação em Prolog**: `voa(X) :- passaro(X).`
    (A variável `X` é implicitamente quantificada universalmente.)

**Quantificador Existencial e Fatos**: considere a afirmação: "_Existe um pássaro que não pode voar_."
* **Fórmula em LPO**: $\exists x (\text{Pássaro}(x) \land \neg \text{Voa}(x))$
* **Tratamento em sistemas de Cláusulas de Horn**: Afirmações existenciais puras como $\exists x \Phi(x)$ não são diretamente representadas como regras de Cláusula de Horn. Para incorporar tal conhecimento, se o indivíduo específico for conhecido, ele é afirmado como um conjunto de fatos.
    Por exemplo, se sabemos que "Pengu" é um pássaro e não voa:
    * **Fatos em Prolog**:
        `passaro(pengu).`
        `nao_voa(pengu).` (ou `voa(pengu) :- fail.`)

* Se a existência é conhecida mas o indivíduo não é nomeado, em processos de prova teórica (como resolução), a Skolemização substituiria $x$ por uma nova constante (constante de Skolem), resultando em: $\text{Pássaro}(c) \land \neg \text{Voa}(c)$. Estes seriam então fatos no sistema: $\text{Pássaro}(c).$ e $\neg \text{Voa}(c).$ (ou um predicado para a negação).

### Conversão de Fórmulas

Seja uma fórmula bem formada arbitrária da Lógica Proposicional. Alguns passos podem ser aplicados para obter uma cláusula de Horn equivalente:

1. Converter a fórmula para Forma Normal Conjuntiva (FNC), obtendo uma conjunção de disjunções
2. Aplicar as seguintes técnicas em cada disjunção:

   - Inverter a polaridade de literais positivos extras;
   - Adicionar literais negativos que preservem a satisfatibilidade;
   - Dividir em cláusulas menores se necessário.

3. Simplificar a fórmula final obtida.

#### Exemplo 1: dada a fórmula

$$(P \land Q) \lor (P \land R)$$

Passos:

1. Converter para FNC: $(P \lor Q) \land (P \lor R)$;
2. Inverter P em uma das disjunções: $(P \lor Q) \land (\neg P \lor R)$;
3. Adicionar literal negativo: $(P \lor Q \lor \neg S) \land (\neg P \lor R \lor \neg T)$;
4. Simplificar: $\neg S \lor P \land \neg T \lor r $.

A sequência destes passos permite encontrar uma conjunção de cláusulas de Horn equivalente à fórmula original.

#### Transformação de Forma Normal Conjuntiva (FNC) para Cláusulas de Horn

A Forma Normal Conjuntiva é uma conjunção de disjunções de literais. Uma Cláusula de Horn é um tipo especial de cláusula que contém no máximo um literal positivo. Considere que o objetivo das Cláusulas de Horn é criar um conjunto de Fórmulas Bem Formadas, divididas em Fatos, Regras e Consultas para permitir a resolução de problemas então, a transformação de uma FNC para Cláusulas de Horn pode incorrer em alguns problemas:

- **Perda de Informação**: Nem todas as cláusulas em FNC podem ser transformadas em Cláusulas de Horn. Para minimizar este risco atente para as regras de equivalência que vimos anteriormente.
- **Complexidade**: A transformação pode ser complexa e requer uma análise cuidadosa da lógica e do contexto.

**Etapas de Transformação**

1. **Converter para FNC**: Se a fórmula ainda não estiver em Forma Normal Conjuntiva, converta-a para Forma Normal Conjuntiva usando as técnicas descritas anteriormente;
2. **Identificar Cláusulas de Horn**: Verifique cada cláusula na Forma Normal Conjuntiva. Se uma cláusula contém no máximo um literal positivo, ela já é uma Cláusula de Horn;
3. **Transformar Cláusulas Não-Horn**: Se uma cláusula contém mais de um literal positivo, ela não pode ser diretamente transformada em uma Cláusula de Horn sem perder informações.

##### Exemplo 1: vamos considerar a seguinte fórmula bem formada

$$(A \rightarrow B) \land (B \lor C)$$

1. **Converter para FNC**:

   - Elimine a implicação: $(\neg A \lor B) \land (B \lor C)$;
   - A fórmula já está em Forma Normal Conjuntiva.

2. **Identificar Cláusulas de Horn**:

   - Ambas as cláusulas são Cláusulas de Horn, pois cada uma contém apenas um literal positivo.

3. **Resultado**:

   - A fórmula em Cláusulas de Horn é: $(\neg A \lor B) \land (B \lor C)$

#### Problemas interessantes resolvidos com a Cláusula de Horn

**Problema 1 - O Mentiroso e o Verdadeiro**:: Você encontra dois habitantes: $A$ e $B$. Você sabe que um sempre diz a verdade e o outro sempre mente, mas você não sabe quem é quem. Você consulta a $A$, _Você é o verdadeiro?_ A responde, mas você não consegue ouvir a resposta dele. $B$ então te diz, _A disse que ele é o mentiroso_.

**Fatos**:

$mentiroso(A)$
$verdadeiro(B)$

**Regra**:

$$
\forall x \forall y (mentiroso(x) \wedge consulta(y, \text{Você é o verdadeiro?}) → Responde (x, \text{Sou o mentiroso}))
$$

**Consulta**:

$$ responde (A, \text{Sou o mentiroso})?$$

**Problema 2 - As Três Lâmpadas**: existem três lâmpadas incandescentes em uma sala, e existem três interruptores fora da sala. Você pode manipular os interruptores o quanto quiser, mas só pode entrar na sala uma vez. Como você pode determinar qual interruptor opera qual lâmpada?

**Fatos**:

$Interruptor(s_1)$
$Interruptor(s_2)$
$Interruptor(s_3)$

$Lâmpada(b_1)$
$Lâmpada(b_2)$
$Lâmpada(b_3)$

**Regras**:

$$\forall x \forall y (Interruptor(x) \wedge Ligado(x) \wedge Lâmpada(y) \rightarrow Acende (y))$$

$$\forall x (Lâmpada(x) \wedge FoiLigada(x) \wedge AgoraDesligada(x) \rightarrow EstáQuente (x))$$

**Consulta**:

$$Acende (b_2, s_2)?$$

$$ estáQuente (b_1)?$$

**Problema 3 - O Agricultor, a Raposa, o Ganso e o Grão**: um agricultor quer atravessar um rio e levar consigo uma raposa, um ganso e um saco de grãos. O barco do agricultor só lhe permite levar um item além dele mesmo. Se a raposa e o ganso estiverem sozinhos, a raposa comerá o ganso. Se o ganso e o grão estiverem sozinhos, o ganso comerá o grão. Como o agricultor pode levar todas as suas posses para o outro lado do rio?

**Fatos**:

- $Raposa(r)$;
- $Ganso(g)$;
- $Grao(gr)$.

Nestes fatos a atenta leitora deve observar que '$r$' é uma raposa, '$g$' é um ganso, e '$gr$' é um saco de grãos.

**Regras**:

1. Se $x$ é uma Raposa e $y$ é um Ganso, e $x$ e $y$ estão sozinhos, então $x$ come $y$:

$$\forall x \forall y (Raposa(x) \land Ganso(y) \land Sozinhos(x, y) \rightarrow Come(x, y))$$

2. Se $x$ é um Ganso e $y$ é Grão, e $x$ e $y$ estão sozinhos, então $x$ come $y$:

$$\forall x \forall y (Ganso(x) \land Grao(y) \land Sozinhos(x, y) \rightarrow Come(x, y))$$

O predicado $Sozinhos(Item1, Item2)$ significaria que $Item1$ e $Item2$ estão em uma margem do rio sem o agricultor. O predicado $Come(Predador, Presa)$ significa que o predador come a presa.

**Consulta**:

As consultas visam verificar se, em um determinado estado da travessia, certas condições de não comer são satisfeitas. Para um sistema de prova, estas seriam as metas a serem mantidas verdadeiras, ou suas negações a serem evitadas.

1. A raposa $r$ não come o ganso $g$ (ou seja, é falso que $r$ come $g$)?

$$\neg Come(r, g)$$

2. O ganso $g$ não come o grão $gr$ (ou seja, é falso que $g$ come $gr$)?

$$\neg Come(g, gr)$$

Estas consultas, no contexto da resolução do problema, representam estados seguros que devem ser mantidos durante toda a travessia. A solução do problema envolve encontrar uma sequência de movimentos que leve todos ao outro lado do rio sem nunca satisfazer as condições de $Come(r,g)$ ou $Come(g,gr)$ quando o agricultor não está presente para supervisionar.

**Problema 4 - A Ponte e a Tocha**: quatro pessoas chegam a um rio à noite. Há uma ponte estreita, mas ela só pode conter duas pessoas de cada vez. Eles têm uma tocha e, por ser noite, a tocha tem que ser usada ao atravessar a ponte. A pessoa A pode atravessar a ponte em um minuto, B em dois minutos, C em cinco minutos e D em oito minutos. Quando duas pessoas atravessam a ponte juntas, elas devem se mover no ritmo da pessoa mais lenta. Qual é a forma mais rápida para todos eles atravessarem a ponte?

**Fatos (tempos)**:

- $tempo(a, 1)$;
- $tempo(b, 2)$;
- $tempo(c, 5)$;
- $tempo(d, 8)$.

**Regras**:

1. Regra para determinar qual pessoa é mais lenta:

   $$\neg tempo(X, TX) \lor \neg tempo(Y, TY) \lor \neg maior(TX, TY) \lor mais\_lento(X, Y, X)$$

   $$\neg tempo(X, TX) \lor \neg tempo(Y, TY) \lor \neg maior(TY, TX) \lor mais\_lento(X, Y, Y)$$

2. Regra para calcular o tempo quando duas pessoas atravessam juntas:

   $$\neg mais\_lento(X, Y, Z) \lor \neg tempo(Z, T) \lor tempo\_travessia(X, Y, T)$$

3. Relações "maior que" definidas como fatos:

   - $maior(2, 1)$;
   - $maior(5, 1)$;
   - $maior(5, 2)$;
   - $maior(8, 1)$;
   - $maior(8, 2)$;
   - $maior(8, 5)$.

4. Regras para representar o plano de travessia:

   $$\neg atravessa\_ida(X, Y, T1) \lor \neg volta(Z, T2) \lor \neg atravessa\_ida(W, V, T3) \lor \neg volta(U, T4) \lor \neg atravessa\_ida(S, R, T5) \lor travessia\_completa(T1+T2+T3+T4+T5)$$

   Onde as variáveis representam as pessoas que atravessam em cada fase da solução.

**Consulta**:

$$travessia\_completa(15)?$$

Esta consulta verifica se existe um plano de travessia que soma exatamente 15 minutos, representando a solução ótima para o problema.

**Problema 5 - O Problema de Monty Hall**: em um programa de game show, os concorrentes tentam adivinhar qual das três portas contém um prêmio valioso. Depois que um concorrente escolhe uma porta, o apresentador, que sabe o que está por trás de cada porta, abre uma das portas não escolhidas para revelar uma cabra (representando nenhum prêmio). O apresentador então pergunta ao concorrente se ele quer mudar sua escolha para a outra porta não aberta ou ficar com sua escolha inicial. O que o concorrente deve fazer para maximizar suas chances de ganhar o prêmio?

**Fatos**:

- $Porta(d_1)$;
- $Porta(d_2)$;
- $Porta(d_3)$.

**Regras**:

$$\forall x Prêmio(x) \rightarrow Porta(x)$$

$$\forall x \forall y (Porta(x) \wedge Porta(y) \wedge x \neq y \rightarrow \neg Prêmio(x) \vee \neg Prêmio(y))$$

**Consulta**:

$$\exists x (Porta(x) \wedge \neg Revelada(x) \wedge x \neq PortaEscolhida \rightarrow Prêmio(x))?$$

## O Prolog Entra em Cena

O Prolog é uma linguagem de programação lógica que utiliza Cláusulas de Horn para representar e manipular conhecimento. A sintaxe e a semântica do Prolog são diretamente mapeadas para Cláusulas de Horn:

- **Fatos**: Em Prolog, fatos são representados como cláusulas sem antecedentes. Por exemplo, o fato _John é humano_ pode ser representado como _humano(john)_.
- **Regras**: As regras em Prolog são representadas como implicações, onde os antecedentes são literais negativos e o consequente é o literal positivo. Por exemplo, a regra _Se X é humano, então X é mortal_ pode ser representada como _mortal(X) :- humano(X)_.
- **Consultas**: As consultas em Prolog são feitas ao sistema para inferir informações com base nos fatos e regras definidos. Por exemplo, a consulta "Quem é mortal?" pode ser representada como _?- mortal(X)_.

O Prolog utiliza um mecanismo de resolução baseado em Cláusulas de Horn para responder a consultas. Ele aplica uma técnica de busca em profundidade para encontrar uma substituição de variáveis que satisfaça a consulta.

#### Exemplo 1: O mais simples possível

**Fatos**:

```prolog
homem(joão).
mulher(maria).
```

Os fatos indicam que "João é homem" e "maria é mulher".

**Regra**:

```prolog
mortal(X) :- homem(X).
```

A regra estabelece que "Se $X$ é homem, então $X$ é mortal". O símbolo $:-$ representa implicação.

**Consulta**:

```prolog
mortal(joão).
```

A consulta verifica se "João é mortal", aplicando a regra definida anteriormente. O Prolog responderá **True** (verdadeiro ou $\top$) pois a regra se aplica dado o fato de que João é homem.

#### Exemplo 2: Sistema de Recomendação de Roupas em Prolog

Imagine que estamos construindo um sistema lógico simples em Prolog para recomendar o tipo de roupa que uma pessoa deve vestir com base no clima. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**Fatos**: primeiro, estabelecemos os fatos, que são as verdades básicas sobre o mundo. Neste caso, os fatos podem ser informações sobre o clima atual.

- **Fato 1**: está ensolarado.

```prolog
 ensolarado.
```

- **Fato 2**: a temperatura está acima de 20°C.

```prolog
 temperatura_acima_de_20.
```

**Regras**: em seguida, definimos as regras que descrevem como as coisas se relacionam. Essas regras nos dizem o tipo de roupa apropriada com base no clima.

- **Regra 1**: se está ensolarado e a temperatura está acima de 20°C, use óculos de sol.

```prolog
 óculos_de_sol :- ensolarado, temperatura_acima_de_20.
```

- **Regra 2**: se está ensolarado, use chapéu.

```prolog
 chapéu :- ensolarado.
```

- **Regra 3**: se a temperatura está acima de 20°C, use camiseta.

```prolog
 camiseta :- temperatura_acima_de_20.
```

Agora, podemos fazer consultas ao nosso sistema para obter recomendações de roupas.

- **Consulta 1**: está ensolarado e a temperatura está acima de 20°C. O que devo vestir?

```prolog
 ?- óculos_de_sol, chapéu, camiseta.
```

#### Exemplo 3: Torre de Hanói

```prolog
% Fatos
disco(d1).
disco(d2).
disco(d3).
poste(p1).
poste(p2).
poste(p3).
menor(d1).
maior(d2, d1).
maior(d3, d2).

% Regras (Cláusulas de Horn)

% Um disco está em um poste
em(D, P) :- disco(D), poste(P).

% Um disco está sobre outro
sobre(D1, D2) :- disco(D1), disco(D2), maior(D1, D2).

% Movimento válido
movimento_valido(D, P1, P2) :-
    em(D, P1),
    poste(P2),
    P1 \= P2,
    \+ (em(D2, P2), menor(D2, D)).

% Condição de vitória
vitoria :-
    disco(D1),
    disco(D2),
    disco(D3),
    em(D1, p3),
    em(D2, p3),
    em(D3, p3).

% Regra de que nenhum disco pode estar sobre um disco menor
disco_valido(D1, D2) :-
    disco(D1),
    disco(D2),
    maior(D1, D2).

% Apenas um disco pode ser movido de cada vez
movimento_unico(D) :-
    disco(D),
    \+ (disco(D2), D \= D2, movimento_valido(D2, _, _)).

% Estado inicial
estado_inicial :-
    em(d1, p1),
    em(d2, p1),
    em(d3, p1),
    sobre(d3, d2),
    sobre(d2, d1).

% Consultas possíveis
% ?- movimento_valido(D, P1, P2).
% ?- vitoria.
% ?- disco_valido(D1, D2).
% ?- movimento_unico(D).
% ?- estado_inicial.
```

##### Exemplo 4: O Narrador é seu próprio avô

O problema do narrador que é seu próprio avô é um exemplo clássico de raciocínio lógico e relações familiares. O problema envolve a construção de uma base de conhecimento que representa as relações familiares e a aplicação de regras lógicas para determinar se o narrador realmente é seu próprio avô. Este exemplo foi publicado por [Niklaus Wirth](https://en.wikipedia.org/wiki/Niklaus_Wirth) em seu livro _Algorithms + Data Structures = Programs_ [^1] fazendo referência a um problema que havia sido publicado em um jornal de Zürich em 1922, que cito em tradução livre a seguir:

Casei com uma viúva (vamos chamá-la de W) que tem uma filha adulta (chame-a de D). Meu pai (F), que nos visitava com bastante frequência, apaixonou-se pela minha enteada e casou-se com ela. Por isso, meu pai se tornou meu genro e minha enteada se tornou minha madrasta. Alguns meses depois, minha esposa deu à luz um filho (S1), que se tornou cunhado do meu pai, e meu tio. A esposa do meu pai, ou seja, minha enteada, também teve um filho (S2). Em outras palavras, para todos os efeitos, eu sou meu próprio avo.

Usando este relato como base podemos criar uma base de conhecimento em Prolog, incluir algumas regras, e finalmente verificar se é verdade que o **narrador** é o seu próprio avô.

```prolog
 % predicados
homem(narrador).
homem(f).
homem(s1).
homem(s2).

% Predicados para relações baseadas em casamentos
parentesco_legal(narrador,w).
parentesco_legal(narrador,f).

% relações de parentesco, filhos, netos de sangue
parentesco(w,d).
parentesco(f,narrador).
parentesco(narrador,s1).
parentesco(f,s2).

% Regras para definir, pai, padrasto e avo
pai(X,Y) :- homem(X), parentesco(X,Y).
padrasto(X,Y) :-  homem(X), parentesco_legal(X,Y).
avo(X,Z) :- (pai(X,Y); padrasto(X,Y)), (pai(Y,Z) ; padrasto(Y,Z)).

%pergunte se o narrador é avo dele mesmo avo(narrador,narrador)
```

## Glossário

1. **Álgebra de Boole**: Sistema algébrico usado na lógica matemática, baseado nos valores verdadeiro (1) e falso (0).

2. **Antecedente**: Em uma implicação $P \rightarrow Q$, $P$ é o antecedente.

3. **Aridade**: Número de argumentos que uma função ou predicado aceita.

4. **Argumento**: Lista de proposições (premissas) seguidas de uma conclusão.

5. **Associatividade**: Propriedade onde $(a * b) * c = a * (b * c)$ para um operador $*$.

6. **Átomo**: Proposição indivisível ou predicado aplicado a termos em uma fórmula.

7. **Axioma**: Fórmula ou proposição aceita como verdadeira sem necessidade de demonstração.

8. **Bicondicional** ($\leftrightarrow$): Operador lógico que indica equivalência entre duas proposições.

9. **Cardinalidade**: Número de elementos em um conjunto.

10. **Cláusula**: Disjunção de literais, como $P \vee Q \vee \neg R$.

11. **Cláusula de Horn**: Disjunção de literais com no máximo um literal positivo.

12. **Comutatividade**: Propriedade onde $a * b = b * a$ para um operador $*$.

13. **Conclusão**: Em um argumento, a proposição final que se deriva das premissas.

14. **Conjunção** ($\wedge$): Operador lógico "E".

15. **Consequente**: Em uma implicação $P \rightarrow Q$, $Q$ é o consequente.

16. **Constante**: Símbolo que representa um objeto específico no domínio do discurso.

17. **Constante de Skolem**: Termo introduzido para eliminar quantificadores existenciais.

18. **Contradição**: Fórmula que é sempre falsa, independentemente dos valores de suas variáveis.

19. **Contrapositiva**: Para uma implicação $P \rightarrow Q$, sua contrapositiva é $\neg Q \rightarrow \neg P$.

20. **Dedução**: Processo de derivar conclusões lógicas a partir de premissas.

21. **Disjunção** ($\vee$): Operador lógico "OU".

22. **Distributividade**: Propriedade onde $a * (b + c) = (a * b) + (a * c)$ para operadores $*$ e $+$.

23. **Domínio do Discurso**: Conjunto de objetos sobre os quais as variáveis quantificadas podem se referir.

24. **Dupla Negação**: Princípio onde $\neg \neg P \equiv P$.

25. **Equivalência Lógica** ($\equiv$): Relação entre duas fórmulas que têm o mesmo valor verdade para todas as interpretações.

26. **Escopo**: Parte de uma fórmula à qual um quantificador ou operador se aplica.

27. **Fato**: Na programação lógica, afirmação considerada verdadeira sem condições.

28. **Falseabilidade**: Propriedade de uma hipótese que pode ser provada falsa.

29. **Forma Normal Conjuntiva** (FNC): Fórmula que é uma conjunção de cláusulas, onde cada cláusula é uma disjunção de literais.

30. **Forma Normal Disjuntiva** (FND): Fórmula que é uma disjunção de conjunções de literais.

31. **Forma Normal Negativa** (FNN): Fórmula onde as negações aparecem apenas imediatamente antes das variáveis proposicionais.

32. **Forma Normal Prenex**: Fórmula onde todos os quantificadores estão no início, seguidos por uma matriz sem quantificadores.

33. **Forma Normal Skolem**: Forma Normal Prenex onde todos os quantificadores existenciais foram eliminados.

34. **Fórmula Atômica**: Fórmula que consiste em um predicado aplicado a termos.

35. **Fórmula Bem Formada**: Sequência de símbolos que segue as regras de formação da linguagem lógica.

36. **Função**: Mapeamento de um conjunto de argumentos para um valor único.

37. **Função de Skolem**: Função introduzida para eliminar quantificadores existenciais que dependem de variáveis universalmente quantificadas.

38. **Idempotência**: Propriedade onde $a * a = a$ para um operador $*$.

39. **Implicação** ($\rightarrow$): Operador lógico "SE...ENTÃO".

40. **Indução Matemática**: Método de prova que envolve um caso base e um passo indutivo.

41. **Inferência**: Processo de derivar novas informações a partir de informações existentes.

42. **Instanciação**: Substituição de uma variável por um termo específico.

43. **Interpretação**: Atribuição de significado aos símbolos de uma linguagem formal.

44. **Leis de De Morgan**: $\neg(P \wedge Q) \equiv (\neg P \vee \neg Q)$ e $\neg(P \vee Q) \equiv (\neg P \wedge \neg Q)$.

45. **Lema**: Proposição auxiliar demonstrável utilizada como passo intermediário na prova de um teorema.

46. **Literal**: Variável proposicional ou sua negação.

47. **Lógica de Primeira Ordem**: Sistema formal para representar e raciocinar sobre propriedades de objetos e relações entre eles.

48. **Lógica Proposicional**: Sistema lógico que lida com proposições e suas inter-relações.

49. **Meta-linguagem**: Linguagem usada para descrever outra linguagem.

50. **Modelo**: Interpretação que satisfaz um conjunto de fórmulas.

51. **Modus Ponens**: Regra de inferência: $P, P \rightarrow Q \vdash Q$.

52. **Modus Tollens**: Regra de inferência: $\neg Q, P \rightarrow Q \vdash \neg P$.

53. **Negação** ($\neg$): Operador lógico que inverte o valor de verdade de uma proposição.

54. **Predicado**: Função que mapeia objetos a valores de verdade.

55. **Premissa**: Proposição a partir da qual se deriva uma conclusão em um argumento.

56. **Prolog**: Linguagem de programação baseada na Lógica de Primeira Ordem e Cláusulas de Horn.

57. **Prova**: Sequência de passos lógicos que demonstra a verdade de uma proposição.

58. **Quantificador Existencial** ($\exists$): Símbolo lógico que significa "existe pelo menos um".

59. **Quantificador Universal** ($\forall$): Símbolo lógico que significa "para todo".

60. **Recíproca**: Para uma implicação $P \rightarrow Q$, sua recíproca é $Q \rightarrow P$.

61. **Redução ao Absurdo**: Método de prova que assume a negação da conclusão e deriva uma contradição.

62. **Refutação**: Prova da falsidade de uma proposição.

63. **Regra**: Na programação lógica, implicação que define como derivar novos fatos.

64. **Resolução**: Regra de inferência usada em provas automatizadas.

65. **Satisfatibilidade**: Propriedade de uma fórmula que é verdadeira para pelo menos uma interpretação.

66. **Semântica**: Estudo do significado em linguagens formais e naturais.

67. **Silogismo**: Forma de raciocínio dedutivo com duas premissas e uma conclusão.

68. **Sintaxe**: Conjunto de regras que definem as sequências bem formadas em uma linguagem.

69. **Skolemização**: Processo de eliminação de quantificadores existenciais em uma fórmula lógica.

70. **Tabela Verdade**: Tabela que mostra os valores de verdade de uma fórmula para todas as combinações possíveis de seus componentes.

71. **Tautologia**: Fórmula que é sempre verdadeira, independentemente dos valores de suas variáveis.

72. **Teoria**: Conjunto de fórmulas em um sistema lógico.

73. **Teorema**: Afirmação que pode ser provada como verdadeira dentro de um sistema lógico.

74. **Termo**: Constante, variável ou função aplicada a outros termos.

75. **Unificação**: Processo de encontrar substituições que tornam dois termos idênticos.

76. **Universo de Herbrand**: Conjunto de todos os termos básicos que podem ser construídos a partir das constantes e funções de uma linguagem de primeira ordem.

77. **Universo do Discurso**: Conjunto de todas as entidades sobre as quais as variáveis em uma fórmula lógica podem assumir valores.

78. **Validade**: Propriedade de um argumento onde a conclusão é verdadeira sempre que todas as premissas são verdadeiras.

79. **Variável**: Símbolo que representa um objeto não especificado no domínio do discurso.

80. **Variável Livre**: Variável em uma fórmula que não está ligada a nenhum quantificador.

---

[^1]: WIRTH, Niklaus. **Algorithms and Data Structures**. [S.l.]: [s.n.], [s.d.]. Disponível em: https://cdn.preterhuman.net/texts/math/Data_Structure**AND**Algorithms/Algorithms%20and%20Data%20Structures%20-%20Niklaus%20Wirth.pdf.

[^2]: GHIDINI, C., & Serafini, L. (2013-2014). **Mathematical Logic Exercises**. Disponível em: https://disi.unitn.it/~ldkr/ml2014/ExercisesBooklet.pdf.

[^3]: GALLIER, J. H. (1986). **Logic for Computer Science**. [S.l.]: [s.n.], [s.d.]. Disponível em: https://perso.liris.cnrs.fr/alain.mille/enseignements/Master_PRO/BIA/BIA_2012_2013/logic_gallier.pdf

## Referências

1. BOOLOS, G.; BURGESS, J.; JEFFREY, R. **Computability and Logic**. 5. ed. Cambridge: Cambridge University Press, 2007.

2. CHANG, C.C.; KEISLER, H.J. **Model Theory**. Amsterdam: North-Holland, 1990.

3. EBBINGHAUS, H.D.; FLUM, J. **Finite Model Theory**. 2. ed. Berlin: Springer, 2006.

4. GALLIER, J.H. **Logic for Computer Science: Foundations of Automatic Theorem Proving**. 2. ed. Mineola: Dover Publications, 2015.

5. GENESERETH, M.; NILSSON, N. **Logical Foundations of Artificial Intelligence**. San Francisco: Morgan Kaufmann, 1987.

6. INTERNATIONAL MONETARY FUND. World Economic Outlook, October 2023: **Navigating Global Divergences**. IMF, out. 2023. Disponível em: https://www.imf.org/en/Publications/WEO/Issues/2023/10/10/world-economic-outlook-october-2023. Acesso em: 17 mai. 2025.

7. KRIPKE, S. **Naming and Necessity**. Cambridge: Harvard University Press, 1980.

8. MANNA, Z. **Verification of Computer Programs**. Cambridge: MIT Press, 1974.

9. MDPI BLOG. **Five Breakthrough Moments in Science and Technology in 2022**. MDPI Blog, 23 jan. 2023. Disponível em: https://blog.mdpi.com/2023/01/23/breakthroughs-in-2022/. Acesso em: 17 mai. 2025.

10. MIT TECHNOLOGY REVIEW. **10 Breakthrough Technologies 2022**. MIT Technology Review, 23 fev. 2022. Disponível em: https://www.technologyreview.com/2022/02/23/1045416/10-breakthrough-technologies-2022/. Acesso em: 17 mai. 2025.

11. QUINE, W.V.O. **Word and Object**. Cambridge: MIT Press, 1960.

12. RUSSELL, S.; NORVIG, P. **Artificial Intelligence: A Modern Approach**. 4. ed. Upper Saddle River: Pearson, 2020.

13. VAN HARMELEN, F.; LIFSCHITZ, V.; PORTER, B. (Ed.). **Handbook of Knowledge Representation**. Amsterdam: Elsevier, 2008.

14. WIRTH, N. **Algorithms + Data Structures = Programs**. 3. ed. Englewood Cliffs: Prentice-Hall, 1976.
