---
title: "Decifrando Mistérios: A Jornada da Programação Lógica"
layout: post
author: Frank
description: Uma aventura pelo universo matemático que fundamenta a Programação Lógica.
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
 - material de Aula
 - matemática
rating: 5
slug: introducao-programacao-logica
keywords:
 - lógica
 - Programação
 - Programação Lógica
draft: true
---

A Programação Lógica é artefato de raciocínio capaz de ensinar um detetive computadorizado a resolver os mais intricados mistérios, permitindo que se preocupe apenas com o _o que_ e deixando o _como_ a cargo da máquina. Um paradigma de programação onde não precisamos atentar para os estados da máquina e podemos nos concentrar no problema que queremos resolver. Esta é a base de alguns dos modelos computacionais que estão mudando o mundo, na revolução da Inteligência Artificial.

> "Logic programming is the future of artificial intelligence." - [Marvin Minsky](https://en.wikipedia.org/wiki/marvin_Minsky)

- [Introdução](#introdução)
- [Lógica de Primeira Ordem](#lógica-de-primeira-ordem)
  - [Lógica Proposicional](#lógica-proposicional)
  - [Regras de Inferência](#regras-de-inferência)
    - [_Modus Ponens_](#modus-ponens)
    - [_Modus Tollens_](#modus-tollens)
    - [Dupla Negação](#dupla-negação)
    - [Adição](#adição)
    - [_Modus Tollendo Ponens_](#modus-tollendo-ponens)
    - [Adjunção](#adjunção)
    - [Simplificação](#simplificação)
    - [Bicondicionalidade](#bicondicionalidade)
    - [Equivalência](#equivalência)
  - [Classificando Fórmulas Proposicionais](#classificando-fórmulas-proposicionais)
- [Um Sistema de Prova](#um-sistema-de-prova)
  - [Contrapositivas e Recíprocas](#contrapositivas-e-recíprocas)
    - [Logicamente Equivalente](#logicamente-equivalente)
    - [Contrapositiva](#contrapositiva)
    - [Recíproca](#recíproca)
  - [Análise de Argumentos](#análise-de-argumentos)
  - [Finalmente, um Sistema de Prova](#finalmente-um-sistema-de-prova)
    - [Lema](#lema)
    - [Hipóteses](#hipóteses)
- [Lógica Predicativa](#lógica-predicativa)
  - [Introdução aos Predicados](#introdução-aos-predicados)
  - [Universo do Discurso](#universo-do-discurso)
    - [Entendendo Predicados](#entendendo-predicados)
  - [Quantificadores](#quantificadores)
  - [Quantificador Universal](#quantificador-universal)
  - [Quantificador Existencial](#quantificador-existencial)
    - [Equivalências Interessantes](#equivalências-interessantes)
  - [Dos Predicados à Linguagem Natural](#dos-predicados-à-linguagem-natural)
    - [Exercícios de Conversão de Linguagem Natural em Expreessões Predicativas](#exercícios-de-conversão-de-linguagem-natural-em-expreessões-predicativas)
    - [Exercícios de Conversão de Expressões Predicativas em Linguagem Natural](#exercícios-de-conversão-de-expressões-predicativas-em-linguagem-natural)
  - [Ordem de Aplicação dos Quantificadores](#ordem-de-aplicação-dos-quantificadores)
    - [Negação dos Quantificadores](#negação-dos-quantificadores)
  - [Regras de Inferência usando Quantificadores](#regras-de-inferência-usando-quantificadores)
    - [Repetição](#repetição)
    - [Instanciação Universal](#instanciação-universal)
    - [Generalização Existencial](#generalização-existencial)
    - [Instanciação Existencial](#instanciação-existencial)
  - [Problemas Interessantes Resolvidos com Lógica Proposicional e Predicativa](#problemas-interessantes-resolvidos-com-lógica-proposicional-e-predicativa)
    - [Exemplos Extras de conversão de sentenças em predicados](#exemplos-extras-de-conversão-de-sentenças-em-predicados)
    - [Exemplos Extras de Conversão de Predicados em Sentenças](#exemplos-extras-de-conversão-de-predicados-em-sentenças)
  - [Formas Normais](#formas-normais)
    - [Forma Normal Negativa (FNN)](#forma-normal-negativa-fnn)
      - [Estrutura da Forma Normal Negativa](#estrutura-da-forma-normal-negativa)
      - [Conversão para Forma Normal Negativa](#conversão-para-forma-normal-negativa)
      - [Exemplo 1: Converta a fórmula $\\neg (A \\land (B \\rightarrow C))$ Para FNN](#exemplo-1-converta-a-fórmula-neg-a-land-b-rightarrow-c-para-fnn)
      - [Exemplo 2: Converta a fórmula $(A \\rightarrow B) \\land \\neg (C \\lor D)$ Para FNN](#exemplo-2-converta-a-fórmula-a-rightarrow-b-land-neg-c-lor-d-para-fnn)
    - [Forma Normal Disjuntiva (FND)](#forma-normal-disjuntiva-fnd)
      - [Estrutura da Forma Normal Disjuntiva](#estrutura-da-forma-normal-disjuntiva)
      - [Conversão para Forma Normal Disjuntiva](#conversão-para-forma-normal-disjuntiva)
      - [Exemplos de Conversão para a Forma Normal Disjuntiva (Proposicional)](#exemplos-de-conversão-para-a-forma-normal-disjuntiva-proposicional)
    - [Forma Normal Conjuntiva (FNC)](#forma-normal-conjuntiva-fnc)
      - [Estrutura da Forma Normal Conjuntiva](#estrutura-da-forma-normal-conjuntiva)
      - [Conversão para Forma Normal Conjuntiva](#conversão-para-forma-normal-conjuntiva)
      - [Exemplos de Conversão para Forma Normal Conjuntiva](#exemplos-de-conversão-para-forma-normal-conjuntiva)
    - [Usando a Tabela-Verdade para Gerar Formas Normais](#usando-a-tabela-verdade-para-gerar-formas-normais)
      - [Gerando a Forma Normal Disjuntiva](#gerando-a-forma-normal-disjuntiva)
      - [Gerando a Forma Normal Conjuntiva](#gerando-a-forma-normal-conjuntiva)
    - [Skolemização](#skolemização)
    - [Forma Normal Prenex](#forma-normal-prenex)
      - [Estrutura da Forma Normal Prenex](#estrutura-da-forma-normal-prenex)
      - [Conversão para Forma Normal Prenex](#conversão-para-forma-normal-prenex)
      - [Regras de Equivalência Prenex](#regras-de-equivalência-prenex)
    - [Conversão para Formas Normais Conjuntiva (FNC) e Disjuntiva (FND)](#conversão-para-formas-normais-conjuntiva-fnc-e-disjuntiva-fnd)
      - [Exemplos Interessantes da Forma Prenex](#exemplos-interessantes-da-forma-prenex)
      - [Observações Importantes](#observações-importantes)
      - [Exemplos de conversão em formas normais, conjuntiva e disjuntiva](#exemplos-de-conversão-em-formas-normais-conjuntiva-e-disjuntiva)
  - [Cláusula de Horn](#cláusula-de-horn)
    - [Definição da Cláusula de Horn](#definição-da-cláusula-de-horn)
      - [Forma Geral](#forma-geral)
      - [Tipos de Cláusulas de Horn](#tipos-de-cláusulas-de-horn)
      - [Exemplo 1: Sistema de Recomendação de Roupas](#exemplo-1-sistema-de-recomendação-de-roupas)
      - [Exemplo 2: Sistema de Diagnóstico Médico](#exemplo-2-sistema-de-diagnóstico-médico)
    - [Quantificadores em Cláusulas de Horn](#quantificadores-em-cláusulas-de-horn)
      - [Usando o Quantificador Universal em Cláusulas de Horn](#usando-o-quantificador-universal-em-cláusulas-de-horn)
      - [Usando o Quantificador Existencial em Cláusulas de Horn](#usando-o-quantificador-existencial-em-cláusulas-de-horn)
    - [Conversão de Fórmulas](#conversão-de-fórmulas)
      - [Exemplo: dada a fórmula](#exemplo-dada-a-fórmula)
      - [Transformação de Forma Normal Conjuntiva (FNC) para Cláusulas de Horn](#transformação-de-forma-normal-conjuntiva-fnc-para-cláusulas-de-horn)
      - [Etapas de Transformação](#etapas-de-transformação)
      - [Problemas interessantes resolvidos com a Cláusula de Horn](#problemas-interessantes-resolvidos-com-a-cláusula-de-horn)
    - [Cláusulas de Horn e o Prolog](#cláusulas-de-horn-e-o-prolog)
      - [Exemplo 1: O mais simples possível](#exemplo-1-o-mais-simples-possível)
      - [Exemplo 2: Sistema de Recomendação de Roupas em Prolog](#exemplo-2-sistema-de-recomendação-de-roupas-em-prolog)
    - [Um Problema Interessante](#um-problema-interessante)

# Introdução

Imagine, por um momento, que estamos explorando o universo dos computadores, mas em vez de sermos os comandantes, capazes de ditar todos os passos do caminho, nós fornecemos as diretrizes gerais e deixamos que o computador deduza o caminho. Pode parecer estranho, a princípio, para quem está envolvido com as linguagens do Paradigma Imperativo. Acredite, ou não, isso é exatamente o que a Programação Lógica faz.

Em vez de sermos forçados a ordenar cada detalhe do processo de solução de um problema, a Programação Lógica permite que declaremos o que queremos, e então deixar o computador fazer o trabalho de encontrar os detalhes e processos necessários para resolver cada problema.

Na **Programação Imperativa** partimos de uma determinada expressão e seguimos um conjunto de instruções até encontrar o resultado desejado. O programador fornece um conjunto de instruções que definem o fluxo de controle e modificam o estado da máquina a cada passo. O foco está em **como** o problema deve ser resolvido. Exemplos de linguagens imperativas incluem C++, Java e Python.

Na Programação Descritiva, o programador fornece uma descrição lógica, ou funcional, de **o que** deve ser feito, sem especificar o fluxo de controle. O foco está no problema, não na solução. Exemplos incluem SQL, Prolog e Haskell. Na Programação Lógica, partimos de uma conjectura e, de acordo com um conjunto específico de regras, tentamos construir uma prova para esta conjectura.

Na Programação Lógica, um dos paradigmas da **Programação Descritiva** usamos a dedução para resolver problemas.

_Uma conjectura é uma suposição, expressa na forma de proposição, que é acreditada ser verdadeira, mas ainda não foi provada_. Uma sentença declarativa que precisa ser verificada em busca da sua validação. Na linguagem natural, conjecturas são frequentemente expressas como declarações. Na Lógica de Primeira Ordem, serão proposições e as proposições serão tratadas como sentenças que foram criadas para serem verificadas na busca da verdade. Para testar a verdade expressa nestas sentenças usaremos as ferramentas da própria Lógica de Primeira Ordem.

![Diagrama de Significado de Conjecturas](/assets/images/conjecturas.jpeg)

Em resumo: **programação imperativa** focada no processo, no _como_ chegar à solução; **programação descritiva** focada no problema em si, no _o que_ precisa ser feito. Eu, sempre que posso escolho uma linguagem descritiva. Não há glória, nem honra nesta escolha apenas as lamúrias da opinião pessoal.

Sua escolha, pessoal e intransferível, entre estes paradigmas dependerá da aplicação que será construída, tanto quanto dependerá do estilo do programador. Contudo, o futuro parece cada vez mais orientado para linguagens descritivas, que permitam ao programador concentrar-se no problema, não nos detalhes da solução. Efeito que parece ser evidente se considerarmos os avanços da segunda década no século XX no campo da Inteligência Artificial. Este documento contém a base matemática que suporta o entendimento da programação lógica e um pouco de Prolog, como linguagem de programação para solução de problemas. Será uma longa jornada.

Em nossa jornada, percorreremos a **Lógica de Primeira Ordem**. Esta será a nossa primeira rota, que iremos subdividir em elementos interligados e interdependentes e, sem dúvida, de mesma importância e valor: a _lógica Proposicional_ e a _lógica Predicativa_. Não deixe de notar que muitos dos nossos companheiros de viagem, aqueles restritos a academia, podem não entender as sutilezas desta divisão. A estes, deixo a justificativa, meio rota e meio esfarrapada da necessidade do uso da didática para a estruturação do aprendizado. Pobre do professor que ignora as mazelas enfrentadas por seus alunos. Condenado está a falar às paredes.

Pretensioso este timoneiro tenta não ser. Partiremos da _Lógica Proposicional_ com esperança de encontrar bons ventos que nos levem até o Prolog.

A _Lógica Proposicional_ é um tipo de linguagem matemática, suficientemente rica para expressar os problemas que precisamos resolver e suficientemente simples para que computadores possam lidar com ela. Ferramenta tão útil ao homem quanto a máquina. Quando esta ferramenta estiver conhecida mergulharemos na alma da _Lógica de Primeira Ordem_, a _Lógica Predicativa_, ou Lógica de Predicados, e então poderemos fazer sentido do mundo real de forma clara e bela.

Vamos enfrentar a inferência e a dedução, duas ferramentas para extração de conhecimento de declarações lógicas. Voltando a metáfora do Detetive, podemos dizer que a inferência é quase como um detetive que tira conclusões a partir de pistas: teremos algumas verdades, nossas pistas, e precisaremos descobrir outras verdades, consequências diretas das primeiras verdades, para encontrar o que procuramos de forma incontestável. A verdade da lógica não abarca opiniões ou contestações. É linda e incontestável.

Nossos mares não serão brandos, mas não nos furtaremos a enfrentar  as especifidades da **Cláusula de Horn**, um conceito um pouco mais estranho. Uma regra que torna todos os problemas expressos em lógica mais fácies de resolver. Como um mapa que, se corretamente seguido, torna o processo de descobrir a verdade mais simples. Muito mais simples, eventualmente passível de automatização.

No final do dia, cansados, porém felizes, vamos entender que, desde os tempos de [Gödel](https://en.wikipedia.org/wiki/Kurt_Gödel), [Turing](https://en.wikipedia.org/wiki/Alan_TurinQ) e [Church](https://en.wikipedia.org/wiki/Alonzo_ChurcR), tudo que queremos é que nossas máquinas sejam capazes de resolver problemas complexos com o mínimo de interferência nossa. Queremos que elas pensem, ou pelo menos, que simulem o pensamento. Aqui, neste objetivo, entre as pérolas mais reluzentes da evolução humana destaca-se a Programação Lógica.

Como diria [Newton](https://en.wikipedia.org/wiki/Isaac_Newton) chegamos até aqui porque nos apoiamos nos ombros de gigantes. O termo Programação Lógica aparece em meados dos anos 1970 como uma evolução dos esforços nas pesquisas sobre a prova computacional de teoremas matemáticos e Inteligência Artificial. O homem querendo fazer máquinas capazes de raciocinar como o homem. Deste esforço surgiu a esperança de que poderíamos usar a lógica como uma linguagem de programação, em inglês, _programming logic_, ou Prolog. Aqui está a base deste conhecimento.

# Lógica de Primeira Ordem

A Lógica de Primeira Ordem é uma estrutura básica da ciência da computação e da programação. Ela nos permite discursar e raciocinar com precisão sobre os elementos - podemos fazer afirmações sobre todo um grupo, ou sobre um único elemento em particular. No entanto, tem suas limitações - não podemos usá-la para fazer afirmações diretas sobre predicados ou funções.

Essa restrição não é um defeito, mas sim um equilíbrio cuidadoso entre poder expressivo e simplicidade computacional. Dá-nos uma maneira de formular uma grande variedade de problemas, sem tornar o processo de resolução desses problemas excessivamente complexo.

A Lógica de Primeira Ordem é o nosso ponto de partida, nossa base, nossa pedra fundamental. Uma forma poderosa e útil de olhar para o universo, não tão complicada que seja hermética a olhos leigos, mas suficientemente complexa para permitir a descoberta de alguns dos mistérios da matemática e, no processo, resolver alguns problemas práticos.

A Lógica de Primeira Ordem consiste de uma linguagem, consequentemente criada sobre um alfabeto $\Sigma $, de um conjunto de axiomas e de um conjunto de regras de inferência. Esta linguagem consiste de todas as fórmulas bem formadas da teoria da Lógica Proposicional e predicativa. O conjunto de axiomas é um subconjunto do conjunto de fórmulas bem formadas acrescido e, finalmente, um conjunto de regras de inferência.

O alfabeto $\Sigma$ que estamos definindo poderá ser dividido em classes formadas por conjuntos de símbolos agrupados por similaridade. Assim:

1. **variáveis, constantes e símbolos de pontuação**: vamos usar os símbolos do alfabeto latino em minúsculas e alguns símbolos de pontuação. Destaque-se os símbolos $($ e $)$, parenteses, que usaremos para definir a prioridade de operações. Vamos usar os símbolos $U$, $V$, $w$, $x$, $y$ e $z$ Para indicar variáveis e $a$, $b$, $c$, $d$ e $e$ Para indicar constantes.

2. **funções**: usaremos os símbolos $\mathbf{f}$, $\mathbf{g}$, $\mathbf{h}$ e $\mathbf{i}$ Para indicar funções.

3. **predicados**: usaremos letras do alfabeto latino, maiúsculas  $P$, $Q$, $R$ e $S$, ou simplesmente _strigs_ como $\text{Azul}$ ou $\text{Sábio}$ para indicar predicados. Sempre começando com letras maiúsculas.

4. **operadores**: usaremos os símbolos tradicionais da Lógica Proposicional: $\neg$ (negação), $\wedge $ (conjunção, _and_), $\vee $ (disjunção, _or_), $\rightarrow$ (implicação) e $\leftrightarrow$ (equivalência).

5. **quantificadores**: nos manteremos no limite da tradiçãoletras do alfabeto latino, maiúsculas  matemática usando $\exists $ (quantificador existencial) e $\forall $ (quantificador universal).

6. **Fórmulas Bem Formadas**: usaremos para representar as Fórmulas Bem Formadas: $P$, $Q$, $R$, $S$, $T$.

Na lógica matemática, uma Fórmula Bem Formada, também conhecida como expressão bem formada, é uma sequência **finita** de símbolos que é formada de acordo com as regras gramaticais de uma linguagem lógica específica.

Em Lógica de Primeira Ordem, uma Fórmula Bem Formada é uma expressão que **só pode ser** verdadeira ou falsa. As Fórmulas Bem Formadas são compostas de quantificadores, variáveis, constantes, predicados, e conectivos lógicos, e devem obedecer a regras específicas de sintaxe.

Em qualquer linguagem matemática regra sintática mais importante é a precedência das operações, uma espécie de receita. Que deve ser seguida à letra. Vamos nos restringir a seguinte ordem de precedência:

$$\neg, \forall, \exists, \wedge, \vee, \rightarrow, \leftrightarrow$$

Dando maior precedência A $\neg$ e a menor A $\leftrightarrow$.

O uso os parenteses e da ordem de precedência requer parcimônia, muita parcimônia. Os parênteses permitem que possamos escrever $(\forall x(\exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))))$ ou $\forall x \exists y (\mathbf{p}(x,y)\rightarrow \mathbf{q}(x))$ que são a mesma Fórmula Bem Formada. Escolha a opção que seja mais fácil de ler e entender.

Nesta linguagem cada sentença, ou preposição, deve ser verdadeira ou falsa, nunca verdadeira e falsa ao mesmo tempo e nada diferente de verdadeiro ou falso. Para que uma sentença, ou preposição, seja verdadeira ela precisa ser logicamente verdadeira. Uma sentença que deve ser falsa é uma sentença contraditória.

Da mesma forma que  aprendemos nossa língua materna reconhecendo padrões, repetições e regularidades, também reconhecemos Fórmulas Bem Formadas por seus padrões característicos. Os símbolos estarão dispostos de forma organizada e padronizada em termos sobre os quais serão aplicadas operações, funções e quantificadores.

Termos são variáveis, constantes ou mesmo funções aplicadas a termos e seguem um pequeno conjunto de regras:

1. uma variável $x$ é um termo em si;
2. uma constante $A$ é um termo em si que será verdadeira $(T)$ ou falsA$(P)$;
3. se $\mathbf{f}$ é uma função de termos $(t_1, ... t_n)$ então $\mathbf{f}(t_1, ... t_n)$ é um termo.

Cada proposição, ou sentença, na Lógica Proposicional é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. _A chuva cai_, _O sol brilha_ - cada uma dessas proposições é verdadeira ou falsa como uma unidade. Um átomo, elemento básico e fundamental de todas as expressões. Também, mas tarde, chamaremos de átomos a todo predicado aplicado aos termos de uma fórmula. Também precisamos definir os predicados.

1. se $P$ é um predicado de termos $(t_1, ... t_n)$ então $P(t_1, ... t_n)$ é uma Fórmula Bem Formada, um átomo.
2. se $P$ e $Q$ são Fórmulas Bem Formadas então: $\neg P$, $P\wedge Q$, $P \vee Q$, $P \rightarrow Q$ e $P \leftrightarrow Q$ são Fórmulas Bem Formadas.
3. se $P$ é uma Fórmula Bem Formada e $x$ uma variável então $\exists x P(x)$ e $\forall x P(x)$ são Fórmulas Bem Formadas.

Podemos dizer que as Fórmulas Bem Formadas: respeitam regras de precedência entre conectivos, parênteses e quantificadores; não apresentam problemas como variáveis livres não quantificadas e, principalmente, são unívocas, sem ambiguidade na interpretação.

Finalmente podemos definir a linguagem da Lógica de Primeira Ordem como o conjunto de todas as Fórmulas Bem Formadas criadas incluindo os campos de estudo da Lógica Proposicional e da Lógica de Predicados. Termos e átomos costurados em uma teia onde cada termo, ou átomo, é como uma ilha isolada de verdade, um fato fundamental que não pode ser dividido em partes menores. _A chuva cai_, _O sol brilha_. Cada uma dessas proposições é verdadeira ou falsa, em si, uma unidade, como uma ilha. As operações lógicas são as pontes que conectam essas ilhas, permitindo-nos construir as estruturas mais complexas da razão.

## Lógica Proposicional

Esse sistema, por vezes chamado de álgebra booleana, fundamental para o desenvolvimento da computação, é uma verdadeira tapeçaria de possibilidades. Na Lógica Proposicional, declarações atômicas, que só podem ter valores os verdadeiro, $t$, ou falso $F$, são entrelaçadas em declarações compostas cuja veracidade, segundo as regras desse cálculo, depende dos valores de verdade das declarações atômicas que as compõem quando sujeitas aos operadores, ou conectivos, que definimos anteriormente.

Vamos representar essas declarações atômicas por literais $A$, $B$, $X_1$, $X_2$ etc., e suas negações por $\neg A$, $\neg B$, $\neg X_1$, $\neg X_2$ etc. Todos os símbolos individuais e suas negações são conhecidas como literais.

Na Lógica Proposicional, as fórmulas, chamadas de Fórmulas Bem Formadas, podem ser atômicas, ou compostas. Existe um operador, ou conectivo lógico, principal, que conecta várias Fórmulas Bem Formadas de forma recursiva.

As declarações atômicas e compostas são costuradas por conectivos para produzir declarações compostas, cujo valor de verdade depende dos valores de verdade das declarações componentes. Os conectivos que consideramos inicialmente, e suas Tabelas Verdade serão:

<table style="margin-left: auto; margin-right: auto; text-align:center;">
 <tr style="border-top: 2px solid gray; border-bottom: 1px solid gray;">
 <th style="border-right: 1px solid gray;">$P$</th>
 <th style="border-right: double gray;">$Q$</th>
 <th style="width:16.8%; border-right: 1px solid gray;">$P\vee Q$</th>
 <th style="width:16.8%; border-right: 1px solid gray;">$P\wedge Q$ </th>
 <th style="width:16.8%; border-right: 1px solid gray;">$\neg P$</th>
 <th style="width:16.8%; border-right: 1px solid gray;">$P\rightarrow Q$</th>
 <th style="width:16.8%; border-right: 1px solid gray;">$P\leftrightarrow Q$</th>
 <th style="width:16.8%;">$P\oplus Q$</th>
 </tr>
 <tr style="background-color: #eeeeee;">
 <td style="border-right: 1px solid gray;">T</td>
 <td style="border-right: double gray;">T</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">T</td>
 <td style="width:16.8%;">F</td>
 </tr>
 <tr>
 <td style="border-right: 1px solid gray;">T</td>
 <td style="border-right: double gray;">F</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">F</td>
 <td style="width:16.8%;">T</td>
 </tr>
 <tr style="background-color: #eeeeee;">
 <td style="border-right: 1px solid gray;">F</td>
 <td style="border-right: double gray;">T</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">F</td>
 <td style="width:16.8%;">T</td>
 </tr>
 <tr style="border-bottom: 2px solid gray;">
 <td style="border-right: 1px solid gray;">F</td>
 <td style="border-right: double gray;">F</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$F$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">$t$</td>
 <td style="width:16.8%; border-right: 1px solid gray;">T</td>
 <td style="width:16.8%;">F</td>
 </tr>
</table>
<legend style="font-size: 1em;
 text-align: center;
 margin-bottom: 20px;">Tabela 1 - Tabela Verdade, operadores básicos.</legend>

Quando aplicamos a Tabela Verdade a uma declaração composta, obtemos um procedimento capaz de determinar se declaração composta é verdadeira ou falsa. Para isso, tudo que temos que fazer é aplicar, segundo as regras de procedência, a Tabela Verdade a expressão, simplificando-a. Uma alternativa mais simples que a aplicação algébrica dos axiomas da Lógica Proposicional.

O operador $\vee $, também chamado de ou inclusivo, é verdade apenas quando ambos os termos são verdadeiros. Diferindo de um operador, que por não ser básico e fundamental, não consta da nossa lista, chamado de ou exclusivo, $\oplus $, falso se ambos os termos forem verdadeiros.

O condicional $\rightarrow $ não representa a implicação em nenhum sentido causal. Em particular, ele é definido como verdadeiro quando nenhum dos termos é verdadeiro, e é falso apenas quando o termo antecedente é verdadeiro e o consequente falso.

O bicondicional $\leftrightarrow$ equivale a ambos os componentes terem o mesmo valor-verdade. Todos os operadores, ou conectivos, conectam duas declarações, exceto $\neg $ que se aplica a apenas um termo.

Cada operador com sua própria Aridade:

<table style="margin-left: auto;
 margin-right: auto; text-align:center;">

<tr  style="border-top: 2px solid gray; border-bottom: 1px solid gray;">
<th style="border-right: 1px solid gray;">No Argumentos</th>  
<th style="border-right: 1px solid gray;">Aridade</th>
<th style="border-right: 1px solid gray; white-space: nowrap;">Exemplos</th>
</tr>

<tr style="background-color: #eeeeee;">
<td style="border-right: 1px solid gray;">0</td>
<td style="border-right: 1px solid gray;">Nullary</td>
<td style="border-right: 1px solid gray; white-space: nowrap;">$5$, $False $, Constantes</td>
</tr>

<tr style="background-color: #ffffff;">  
<td style="border-right: 1px solid gray;">1</td>
<td style="border-right: 1px solid gray;">Unary</td>
<td style="border-right: 1px solid gray; white-space: nowrap;">$P(x)$, $7x$</td>
</tr>

<tr style="background-color: #eeeeee;">
<td style="border-right: 1px solid gray;">2</td>
<td style="border-right: 1px solid gray;">Binary</td>
<td style="border-right: 1px solid gray; white-space: nowrap;">$x \vee y$, $ c \wedge y$</td>
</tr>

<tr style="border-bottom: 2px solid gray; background-color: #ffffff;">
<td style="width:45%; border-right: 1px solid gray;">3</td>  
<td style="width:45%; border-right: 1px solid gray;">Ternary</td>
<td style="width:45%; border-right: 1px solid gray; white-space: nowrap;">if$P$ then $Q$ else $R$, $(P \rightarrow Q) \wedge (\neg P \rightarrow R)$</td>
</tr>
</table>
<legend style="font-size: 1em;
 text-align: center;
 margin-bottom: 20px;">Tabela 2 - Aridade dos Operadores da Lógica Proposicional.</legend>

Ainda observando a Tabela 1, que contem a Tabela Verdade dos operadores da Lógica Proposicional, é fácil perceber que se tivermos $4$ termos diferentes, em vez de $2$, teremos $2^4 = 16$ linhas. Independente do número de termos, se para uma determinada Fórmula Bem Formada todas os resultados forem verdadeiros, $T$, teremos uma _tautologia_, se todos forem falsos, $f$ uma _contradição_.

Uma _tautologia_ é uma fórmula que é sempre verdadeira, não importando atribuição de valores às variáveis. Em Programação Lógica, tautologias representam verdades universais sobre o domínio do problema. Já uma _contradição_ é sempre falsa. Na Programação Lógica, contradições indicam inconsistências ou impossibilidades lógicas no domínio do problema.

Identificar tautologias permite simplificar expressões e fazer inferências válidas automaticamente. Reconhecer contradições evita o custo de tentar provar algo logicamente impossível.

Linguagens de programação que usam a Programação Lógica usam _unificação_ e resolução para fazer deduções. Tautologias geram cláusulas vazias que simplificam esta resolução. Em problemas de _satisfatibilidade_, se obtivermos uma contradição, sabemos que as premissas são insatisfatíveis. Segure as lágrimas e o medo. Os termos _unificação_ e _satisfatibilidade_ serão explicados assim que sejam necessários. Antes disso, precisamos falar de _equivalências_. Para isso vamos incluir um metacaractere no alfabeto da nossa linguagem: o caractere $\equiv $ que permitirá o entendimento das principais equivalências da Lógica Proposicional explicitadas a seguir:

<table style="width: 100%; margin: auto; border-collapse: collapse;">
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%; border-top: 2px solid #666666;">$P\wedge Q \equiv Q \wedge P$</td>
  <td style="text-align: center; width: 30%; border-top: 2px solid #666666;">Comutatividade da Conjunção</td>
  <td style="text-align: center; width: 20%;border-top: 2px solid #666666;">(1)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\vee Q \equiv Q \vee P$</td>
  <td style="text-align: center; width: 30%;">Comutatividade da Disjunção</td>
  <td style="text-align: center; width: 20%;">(2)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\wedge (Q \vee R) \equiv (P \wedge Q) \vee (P \wedge R)$</td>
  <td style="text-align: center; width: 30%;">Distributividade da Conjunção sobre a Disjunção</td>
  <td style="text-align: center; width: 20%;">(3)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\vee (Q\wedge R) \equiv (P \vee Q) \wedge (P \vee R)$</td>
  <td style="text-align: center; width: 30%;">Distributividade da Disjunção sobre a Conjunção</td>
  <td style="text-align: center; width: 20%;">(4)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;"> $\neg (P \wedge Q) \equiv \neg P \vee \neg Q$</td>
  <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
  <td style="text-align: center; width: 20%;">(5)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;"> $\neg (P \vee Q) \equiv \neg P \wedge \neg Q$</td>
  <td style="text-align: center; width: 30%;">Lei de De Morgan</td>
  <td style="text-align: center; width: 20%;">(6)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\rightarrow Q \equiv \neg P \vee Q$</td>
  <td style="text-align: center; width: 30%;">Definição de Implicação</td>
  <td style="text-align: center; width: 20%;">(7)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\leftrightarrow Q \equiv (P \rightarrow Q) \wedge (Q \rightarrow P)$</td>
  <td style="text-align: center; width: 30%;">Definição de Equivalência</td>
  <td style="text-align: center; width: 20%;">(8)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\rightarrow Q \equiv \neg Q \rightarrow \neg P$</td>
  <td style="text-align: center; width: 30%;">Lei da Contra positiva</td>
  <td style="text-align: center; width: 20%;">(9)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\wedge \neg P \equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Contradição</td>
  <td style="text-align: center; width: 20%;">(10)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\vee \neg P \equiv T$</td>
  <td style="text-align: center; width: 30%;">Lei da Exclusão</td>
  <td style="text-align: center; width: 20%;">(11)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;"> $\neg(\neg P) \equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Dupla Negação</td>
  <td style="text-align: center; width: 20%;">(12)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Identidade</td>
  <td style="text-align: center; width: 20%;">(13)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\wedge T \equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Identidade para a Conjunção</td>
  <td style="text-align: center; width: 20%;">(14)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\wedge F \equiv F$</td>
  <td style="text-align: center; width: 30%;">Lei do Domínio para a Conjunção</td>
  <td style="text-align: center; width: 20%;">(15)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\vee T \equiv T$</td>
  <td style="text-align: center; width: 30%;">Lei do Domínio para a Disjunção</td>
  <td style="text-align: center; width: 20%;">(16)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\vee F \equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Identidade para a Disjunção</td>
  <td style="text-align: center; width: 20%;">(17)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;">$P\wedge F \equiv F$</td>
  <td style="text-align: center; width: 30%;">Lei da Idempotência para a Conjunção</td>
  <td style="text-align: center; width: 20%;">(18)</td>
 </tr>
 <tr style="background-color: #f2f2f2;">
  <td style="text-align: center; width: 50%;">$P\vee F \equiv P$</td>
  <td style="text-align: center; width: 30%;">Lei da Idempotência para a Disjunção</td>
  <td style="text-align: center; width: 20%;">(19)</td>
 </tr>
 <tr>
  <td style="text-align: center; width: 50%;"> $(P \wedge Q) \wedge R \equiv P \wedge (Q \wedge R)$</td>
  <td style="text-align: center; width: 30%;">Associatividade da Conjunção</td>
  <td style="text-align: center; width: 20%;">(20)</td>
 </tr>
 <tr style="background-color: #f2f2f2;border-bottom: 2px solid #666666;">
  <td style="text-align: center; width: 50%;"> $(P \vee Q) \vee R \equiv P \vee (Q \vee R)$</td>
  <td style="text-align: center; width: 30%;">Associatividade da Disjunção</td>
  <td style="text-align: center; width: 20%;">(21)</td>
 </tr>
</table>
<legend style="font-size: 1em; text-align: center;
 margin-bottom: 20px;">Tabela 3 - Equivalências em Lógica Proposicional.</legend>

Como essas equivalências permitem validar Fórmulas Bem Formadas sem o uso de uma Tabela Verdade. Uma coisa interessante seria tentar provar cada uma delas.

As equivalências que listei pipocaram quase espontaneamente enquanto estava escrevendo este texto, mais por hábito e necessidade que por algum raciocínio organizado. São muitas as equivalências que existem, estas são as mais comuns no meu cotidiano. Talvez, alguns exemplos de validação de Fórmulas Bem Formadas usando apenas as equivalências apresentadas na Tabela 3, estufar as velas do conhecimento e nos levar pela rota que precisamos seguir:

**Exemplo 1**:$P \wedge (Q \vee (P \wedge R))$

$$
 \begin{align*}
 P \wedge (Q \vee (P \wedge R)) &\equiv (P \wedge Q) \vee (P \wedge (P \wedge R)) && \text{(3)} \\
 &\equiv (P \wedge Q) \vee (P \wedge R) && \text{(18)}
 \end{align*}
$$

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

**Exemplo 6:**
$P\wedge (Q \vee (R \rightarrow S)) \vee (\neg E \leftrightarrow P)$

$$
\begin{align*}
P \wedge (Q \vee (R \rightarrow S)) \vee (\neg E \leftrightarrow P) &\equiv P \wedge (Q \vee (\neg R \vee S)) \vee ((\neg E \wedge P) \vee (E \wedge \neg P)) && \text{(1)}\\
&\equiv (P \wedge Q) \vee (P \wedge (\neg R \vee S)) \vee ((\neg E \wedge P) \vee (E \wedge \neg P)) && \text{(2)}\\
&\equiv (P \wedge Q) \vee (P \wedge \neg R) \vee (P \wedge S) \vee (\neg E \wedge P) \vee (E \wedge \neg P) && \text{(3)}
\end{align*}
$$

**Exemplo 7:**
 $\neg(P \vee (Q \wedge \neg R)) \leftrightarrow ((S \vee E) \rightarrow (P \wedge Q))$

$$
\begin{align*}
\neg(P \vee (Q \wedge \neg R)) \leftrightarrow ((S \vee E) \rightarrow (P \wedge Q)) &\equiv (\neg P \wedge \neg(Q \wedge \neg R)) \leftrightarrow ((\neg S \wedge \neg E) \vee (P \wedge Q)) && \text{(7)} \\
&\equiv (\neg P \wedge (Q \vee R)) \leftrightarrow (\neg S \vee \neg E \vee (P \wedge Q)) && \text{(L6)}
\end{align*}
$$

**Exemplo 8:**
 $\neg(P \leftrightarrow Q) \vee ((R \rightarrow S) \wedge (\neg E \vee \neg P))$

$$
\begin{align*}
\neg(P \leftrightarrow Q) \vee ((R \rightarrow S) \wedge (\neg E \vee \neg P)) &\equiv \neg((P \rightarrow Q) \wedge (Q \rightarrow P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(8)}\\
&\equiv (\neg(P \rightarrow Q) \vee \neg(Q \rightarrow P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(5)}\\
&\equiv ((P \wedge \neg Q) \vee (Q \wedge \neg P)) \vee ((\neg R \vee S) \wedge (\neg E \vee \neg P)) && \text{(6)}
\end{align*}
$$

**Exemplo 9:**
 $(P \wedge Q) \vee ((\neg R \leftrightarrow S) \rightarrow (\neg E \wedge P))$

$$
\begin{align*}
(P \wedge Q) \vee ((\neg R \leftrightarrow S) \rightarrow (\neg E \wedge P)) &\equiv (P \wedge Q) \vee ((\neg(\neg R \leftrightarrow S)) \vee (\neg E \wedge P)) && \text{(7)}\\
&\equiv (P \wedge Q) \vee ((H \leftrightarrow I) \vee (\neg E \wedge P)) && \text{(12)}\\
&\equiv (P \wedge Q) \vee (((H \rightarrow I) \wedge (I \rightarrow R)) \vee (\neg E \wedge P)) && \text{(8)}
\end{align*}
$$

**Exemplo 10:**
 $\neg(P \wedge (Q \vee R)) \leftrightarrow (\neg(S \rightarrow E) \vee \neg(P \rightarrow Q))$

$$
\begin{align*}
\neg(P \wedge (Q \vee R)) \leftrightarrow (\neg(S \rightarrow E) \vee \neg(P \rightarrow Q)) &\equiv (\neg P \vee \neg(Q \vee R)) \leftrightarrow ((S \wedge \neg E) \vee (P \wedge \neg Q)) && \text{(7)}\\
&\equiv (\neg F \vee (\neg G \wedge \neg R)) \leftrightarrow ((S \wedge \neg E) \vee (P \wedge \neg Q)) && \text{(6)}
\end{align*}
$$

A Lógica Proposicional é a estrutura mais simples e profunda que usamos para fazer sentido do universo. Imagine um universo de verdades e falsidades, onde cada proposição é um átomo indivisível que detém uma verdade única e inalterada. Neste cosmos de lógica, estas proposições são as estrelas, e as operações lógicas - conjunção, disjunção, negação, implicação, e bi-implicação - são as forças gravitacionais que as unem em constelações mais complexas de significado.

A Lógica Proposicional é essencial para a forma como entendemos e interagimos com o mundo ao nosso redor. Ela fornece a base para a construção de argumentos sólidos e para a avaliação da validade de proposições. Originadas na necessidade humana de descobrir a verdade e diminuir os conflitos a partir da lógica. No entanto, a beleza da Lógica Proposicional se estende além dos campos da filosofia, do discurso e da matemática. Ela é a fundação da Álgebra de [George Boole](https://en.wikipedia.org/wiki/George_Boole), a qual, por sua vez, é a base para o design de circuitos eletrônicos e a construção de computadores modernos.

Em seu trabalho de conclusão de curso, [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) aplicou a Álgebra de Boole à simplificação de circuitos de controle. Desde então, as operações básicas da Álgebra de Boole - AND, OR, NOT - são os componentes fundamentais dos sistemas digitais que formam o núcleo dos computadores, telefones celulares e, de fato, de toda a nossa civilização essencialmente digital. A Lógica Proposicional a base sobre a qual construímos todo o edifício do raciocínio lógico. É como a tabela periódica para os químicos ou as leis de Newton para os físicos. É simples, elegante e, acima de tudo, poderosa.

Tão importante quanto o impacto da _Lógica Proposicional_ na tecnologia digital é a sua importância no pensamento racional, tomada de decisão e prova de teoremas. Neste destino, nosso teodolito são as _Regras de Inferência_.

## Regras de Inferência

Regras de inferência são esquemas que proporcionam a estrutura para derivações lógicas. Base da tomada de decisão computacional. Elas definem os passos legítimos que podem ser aplicados a uma ou mais proposições, sejam elas atômicas ou Fórmulas Bem Formadas, para produzir uma proposição nova. Em outras palavras, uma regra de inferência é uma transformação sintática de Formas Bem Formadas que preserva a verdade.

Aqui uma regra de inferência será representada por:

$$
\frac{P_1, P_2, ..., P_n}{C}\\
$$

Onde o conjunto formado $P_1, P_2, ..., P_n$, chamado de contexto, ou antecedente, $\Gamma $, e $ c$, chamado de conclusão, ou consequente, são Formulas Bem Formadas. A regra significa que se o Proposição é verdadeiro então a conclusão $ c$ também é verdadeira.

Eu vou tentar usar o par contexto e conclusão. Entretanto já vou me desculpando se escapar um antecendente e consequente. Será por mera força do hábito. Quando estudamos lógica chamamos de _argumento_ a uma lista de proposições, neste caso chamadas de premissas, seguidas de uma palavra, ou expressão (portanto, consequentemente, desta forma), e de outra proposição, neste caso, chamada de conclusão.

A representação que usamos é conhecida como sequência de dedução, é uma forma de indicar que se o Proposição, colocado acima da linha horizontal for verdadeiro, estamos dizendo que todas as preposições $P_1, P_2, ..., P_n$ são verdadeiras e todas as proposições colocas abaixo da linha, conclusão, também serão verdadeiras.

As regras de inferência são o alicerce da lógica dedutiva e do estudo das demonstrações matemáticas. Elas permitem que raciocínios complexos sejam quebrados em passos mais simples, cada um dos quais pode ser justificado pela aplicação de uma regra de inferência. Algumas das regras de inferência mais utilizadas estão listadas a seguir:

### _Modus Ponens_

A regra do _Modus Ponens_ permite inferir uma conclusão a partir de uma implicação e de sua premissa antecedente. Se temos uma implicação $P\rightarrow Q $, e sabemos que $P$ é verdadeiro, então podemos concluir que $Q$ também é verdadeiro.

$$
P \rightarrow Q
$$

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

  - Proposição: _se a velocidade, $V$, é maior que $80 km/h$, então é uma infração de trânsito, $IT$_.
  - Proposição: _joão está dirigindo, $ d$, A$90 km/h$_.
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

  - Proposição: _se um número é par, $P$, então é divisível por 2, $ d2$_.
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
 &(AB = AC) \wedge (AB=CB) \text{ no triângulo} ABC\\
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

### _Modus Tollens_

A regra do _Modus Tollens_ permite inferir a negação da premissa antecedente a partir de uma implicação e da negação de sua premissa consequente.Se temos uma implicação $P\rightarrow Q$, e sabemos que $Q$ é falso (ou seja, $\neg G$), então podemos concluir que $P$ também é falso.

$$
P \rightarrow Q
$$

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

Algumas aplicações do _Modus Tollens_:

- Refutar teorias mostrando que suas previsões são falsas. Por exemplo:

  - Proposição: _se a teoria da geração espontânea, $ tG$ é correta, insetos irão se formar em carne deixada exposta ao ar, $I$_.
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

  - Proposição: _se João, $J$, é mais alto, $>$, que mariA$m $, então maria não é mais alta que João_.
  - Proposição: _maria é mais alta que João_.
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

  - Proposição: _se hoje, $H$, é sexta-feira, $ se $, amanhã é sábado $ sA$_.
  - Proposição: _amanhã não é sábado_.
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

  - Proposição: _se a tomada está com defeito, $D$A lâmpada não acende $L$_.
  - Proposição: _a lâmpada não acendeu_.
  - Conclusão: logo, _a tomada deve estar com defeito_.

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

$$
\neg \neg \text{Está chovendo} \Leftrightarrow \text{Está chovendo}
$$

- Preserva o valor de verdade: inserir ou remover duplas negações não altera o valor de verdade original de uma proposição. Isso permite transformar proposições em formas logicamente equivalentes.

- Auxilia provas indiretas: em provas por contradição, ou contrapositiva, introduzir uma dupla negação permite assumir o oposto do que se quer provar e derivar uma contradição. Isso, indiretamente, prova a proposição original.

- Conecta Lógica Proposicional e de predicados: em Lógica Predicativa, a negação de quantificadores universais e existenciais envolve dupla negação. Por exemplo, a negação de _todo $x$ é $P$0_ é _existe algum $x$ tal que não é $P$_.

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

- Proposição: _o céu está azul, $(P)$_.
- Conclusão: logo, _o céu está azul ou gatos podem voar, $(P \lor Q)$_;

A regra da Adição permite introduzir uma disjunção em uma prova ou argumento lógico. Especificamente, ela nos permite inferir uma disjunção $P\vee Q$A partir de uma das afirmações disjuntivas ($P$ ou $Q$) individualmente.

Alguns usos e aplicações importantes da regra da Adição:

- Introduzir alternativas ou possibilidades em um argumento: por exemplo, dado que _João está em casa_, podemos concluir que _João está em casa OR no trabalho_. E expandir este _OR_ o quanto seja necessário para explicitar os lugares onde joão está.

- Combinar afirmações em novas disjunções: dadas duas afirmações quaisquer $P$ e $Q$, podemos inferir que $P$ ou $Q$ é verdadeiro.

- Criar casos ou opções exaustivas em uma prova: podemos derivar uma disjunção que cubra todas as possibilidades relevantes. Lembre-se do pobre _joão_.

- Iniciar provas por casos: ao assumir cada disjuntiva separadamente, podemos provar teoremas por casos exaustivos.

- Realizar provas indiretas: ao assumir a negação de uma disjunção, podemos chegar a uma contradição e provar a disjunção original.

A regra da Adição amplia nossas capacidades de prova e abordagem de problemas.

### _Modus Tollendo Ponens_

O _Modus Tollendo Ponens_ permite inferir uma disjunção a partir da negação da outra disjunção.

Dada uma disjunção $P\vee Q$:

- Se $\neg P$, então $Q$
- Se $\neg Q$, então $P$

Esta regra nos ajuda a chegar a conclusões a partir de disjunções, por exclusão de alternativas.

$$
P \vee Q
$$

$$
\begin{aligned}
&\neg P\\
\hline
&Q\\
\end{aligned}
$$

$$
\begin{aligned}
&\neg Q\\
\hline
&P\\
\end{aligned}
$$

Em linguagem natural:

- Proposição 1: _ou o céu está azul ou a grama é roxa_.
- Proposição 2: _a grama não é roxa_.
- Conclusão: logo, _o céu está azul_

Algumas aplicações do _Modus Tollendo Ponens_:

- Derivar ações a partir de regras disjuntivas. Por exemplo:

  - Proposição: _ou João vai à praia, $P$ ou João vai ao cinema, $ c$_.
  - Proposição: _João não vai ao cinema_, $\neg C$.
  - Conclusão: logo, _João vai à praia_.

$$
P \vee C
$$

$$
\begin{aligned}
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
P \vee I
$$

$$
\begin{aligned}
&\neg I\\
\hline
&P
\end{aligned}
$$

- Eliminar opções em raciocínio dedutivo. Por exemplo:

  - Proposição: _ou João estava em casa, $ c$, ou João estava no trabalho, $t$_.
  - Proposição: _João não estava em casa_.
  - Conclusão: logo, _João estava no trabalho_.

$$
C \vee T
$$

$$
\begin{aligned}
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
1P \vee 1I\\
$$

$$
\begin{aligned}
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

- proposição 1: _o céu está azul_.
- proposição 2: _os pássaros estão cantando_.
- Conclusão: logo, _o céu está azul e os pássaros estão cantando_.

Algumas aplicações da Adjunção:

- Combinar proposições relacionadas em argumentos. Por exemplo:

  - Proposição: _o céu está nublado, $ n$_.
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

  - Proposição: _1 é número natural, $ n1$_.
  - Proposição: _2 é número natural $ n2$_.
  - Conclusão: logo, _1 é número natural **e** 2 é número natural_.

$$
\begin{aligned}
&N1\\
&N2\\
\hline
&N1 \land N2
\end{aligned}
$$

- Derivar novas informações da interseção de fatos conhecidos. Por exemplo:

  - Proposição: _o gato está em cima do tapete, $ gT$_.
  - Proposição: _o rato está em cima do tapete, $ rT$_.
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
&(2 + 2 = 4)\\
&(4 \times 4 = 16)\\
\hline
&(2 + 2 = 4) \land (4 \times 4 = 16)
\end{aligned}
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

- proposição: _o céu está azul e os pássaros estão cantando_
- Conclusão: logo, _o céu está azul. E os pássaros estão cantando_.

Algumas aplicações da Simplificação:

- Derivar elementos de conjunções complexas. Por exemplo:

  - Proposição: _hoje está chovendo, $ c$, e fazendo frio, $F$_.
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

  - Proposição: _o gato está dormindo, $ d$, e ronronando, $R$_.
  - Conclusão: logo, _o gato está ronronando_.

$$
\begin{aligned}
&D \land R\\
\hline
&R
\end{aligned}
$$

- Derivar informações de premissas conjuntivas. Por exemplo:

  - Proposição: _está chovendo, $J$, e o jogo foi cancelado, $ c$_.
  - Conclusão: logo, _o jogo foi cancelado_.

$$
\begin{aligned}
&C \land J\\
\hline
&J
\end{aligned}
$$

### Bicondicionalidade

A regra da Bicondicionalidade permite inferir uma bicondicional a partir de duas condicionais. Esta regra nos permite combinar duas implicações para obter uma afirmação de equivalência lógica.

$$
F \rightarrow G
$$

$$
G \rightarrow F
$$

$$
\begin{aligned}
&G \rightarrow F\\
\hline
&F \leftrightarrow G\\
\end{aligned}
$$

Em linguagem natural:

- proposição _1: se está chovendo, então a rua está molhada_.
- proposição _2: se a rua está molhada, então está chovendo_.
- Conclusão: logo, _está chovendo se e somente se a rua está molhada_.

 Algumas aplicações da Bicondicionalidade:

- Inferir equivalências lógicas a partir de implicações bidirecionais. Por exemplo:

  - Proposição: _se chove, $ c$ então a rua fica molhada, $m $_.
  - Proposição: _se a rua fica molhada, então chove_.
  - Conclusão: logo, _chove se e somente se a rua fica molhada_.

$$
C \rightarrow M
$$

$$
\begin{aligned}
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
P \rightarrow M2
$$

$$
\begin{aligned}
&M2 \rightarrow P\\
\hline
&P \leftrightarrow M2
\end{aligned}
$$

- Estabelecer equivalências matemáticas. Por exemplo:

  - Proposição: _se $x^2 = 25$, então $x = 5$_.
  - Proposição: _se $x = 5$, então $x^2 = 25$_.
  - Conclusão: logo, _$x^2 = 25$ se e somente se $x = 5$_.

$$
(x^2 = 25) \rightarrow (x = 5)
$$

$$
\begin{aligned}
&(x = 5) \rightarrow (x^2 = 25)\\
\hline
&(x^2 = 25) \leftrightarrow (x = 5)
\end{aligned}
$$

- Provar relações de definição mútua. Por exemplo:

  - Proposição: _se figura é um quadrado, $Q$, então tem 4 lados iguais, $4L$_.
  - Proposição: _se figura tem 4 lados iguais, é um quadrado_.
  - Conclusão: logo, _figura é quadrado se e somente se tem 4 lados iguais_.

$$
Q \rightarrow 4L
$$

$$
\begin{aligned}
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
&F\\
\hline
&G\\
\end{aligned}
$$

$$
F \leftrightarrow G
$$

$$
\begin{aligned}
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

1. Inferir fatos de equivalências estabelecidas. Por exemplo:

- Proposição: _o número é par, $P$ se e somente se for divisível por 2, $ d2$_.
- Proposição: _156 é divisível por 2_.
- Conclusão: logo, _156 é par_.

$$
P \leftrightarrow D2
$$

$$
\begin{aligned}
&D2(156)\\
\hline
&P(156)
\end{aligned}
$$

2. Derivar negações de equivalências. Por exemplo:

- Proposição: _$x$ é negativo se e somente se $x < 0$_.
- Proposição: _$x$ não é negativo_.
- Conclusão: logo, _$x$ não é menor que $0$_.

$$
 N \leftrightarrow (x < 0)
$$

$$
\begin{aligned}
&\neg N\\
\hline
&\neg (x < 0)
\end{aligned}
$$

3. Fazer deduções baseadas em definições. Por exemplo:

- Proposição: _número ímpar é definido como não divisível, $ nD2$, por $2$_.
- Proposição: _$9$ não é divisível por $2$_.
- Conclusão: logo, _$9$ é ímpar_.

$$
I \leftrightarrow \neg ND2
$$

$$
\begin{aligned}
&\neg D_2(9)\\
\hline
&I(9)
\end{aligned}
$$

## Classificando Fórmulas Proposicionais

Podemos classificar fórmulas proposicionais de acordo com suas propriedades semânticas, analisando suas tabelas-verdade. Seja $A$ uma fórmula proposicional:

- $A$ é **satisfatível** se sua Tabela Verdade contém pelo menos uma linha verdadeira. Considere:$P\wedge Q$.

$$
\begin{array}{|c|c|c|}
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
 \end{array}
$$

- $A$ é **insatisfatível** se sua Tabela Verdade contém apenas linhas falsas. Exemplo:$P\wedge \neg p$.
- $A$ é **falsificável** se sua Tabela Verdade contém pelo menos uma linha falsa. Exemplo:$P\wedge q$.
- $A$ é **válida** se sua Tabela Verdade contém apenas linhas verdadeiras. Exemplo:$P\vee \neg p$.

Note que:

- Se $A$ é válida, então $A$ é satisfatível.
- Se $A$ é insatisfatível, então $A$ é falsificável.

Fórmulas válidas são importantes na lógica proposicional, representando argumentos sempre verdadeiros independentemente da valoração de suas variáveis proposicionais atômicas. Na verdade, esta classificação será importante para:

1. **Análise de Argumentos**: Se uma argumentação lógica pode ser representada por uma fórmula que é insatisfatível, então sabemos que o argumento é inválido ou inconsistente. Isso é frequentemente usado em lógica e filosofia para analisar a validade dos argumentos.

2. **Prova de Teoremas**: Na prova de teoremas, essas classificações são úteis. Quando estamos tentando provar que uma fórmula é uma tautologia, podemos usar essas classificações para simplificar a tarefa. Podemos mostrar que a negação da fórmula é insatisfatível, mostrando que a fórmula original é uma tautologia.

3. **Simplificação de Fórmulas**: Na simplificação de fórmulas, essas classificações também são úteis. Se temos uma fórmula complexa e podemos mostrar que uma parte dela é uma tautologia, podemos simplificar a fórmula removendo essa parte. Similarmente, se uma parte da fórmula é uma contradição (ou seja, é insatisfatível), sabemos que a fórmula inteira é insatisfatível.

4. **Construção de Argumentos**: Na construção de argumentos, estas classificações são úteis para garantir que os argumentos são válidos. Se estamos construindo um argumento e podemos mostrar que ele é representado por uma fórmula que é satisfatível (mas não uma tautologia), sabemos que existem algumas circunstâncias em que o argumento é válido e outras em que não é.

# Um Sistema de Prova

A matemática respira prova. Nenhuma sentença matemática tem qualquer valor se não for provada. As verdades da aritmética devem ser estabelecidas com rigor lógico; as conjecturas da geometria, confirmadas por construtos infalíveis. Cada novo teorema se ergue sobre os ombros de gigantes – um edifício de razão cuidadosamente erigido.

A beleza da lógica proposicional é revelar, nas entranhas da matemática, um método para destilar a verdade. Seus símbolos e regras exaltam nosso raciocínio e nos elevam da desordem da intuição. Com poucos elementos simples – variáveis, conectivos, axiomas – podemos capturar verdades absolutas no âmbito do pensamento simbólico.

Considere um sistema proposicional, com suas Fórmulas Bem Formadas, suas transformações válidas. Ainda que simples, vemos nesse sistema o que há de profundo na natureza da prova. Seus teoremas irradiam correção; suas demonstrações, poder dedutivo. Dentro deste sistema austero reside a beleza em uma estética hermética, mas que desvelada faz brilhar a luz da razão e do entendimento.

## Contrapositivas e Recíprocas

As implicações são um problema, do ponto de vista da matemática. Sentenças do tipo _se...então_ induzem uma conclusão. Provar estas sentenças é uma preocupação constante da matemática. Dada uma implicação, existem duas fórmulas relacionadas que ocorrem com tanta frequência que possuem nomes especiais: contrapositivas e recíprocas. Antes de mergulharmos em contrapositivas, precisamos visitar alguns portos.

### Logicamente Equivalente

Vamos imaginar um mundo de fórmulas que consistem apenas em duas proposições:$P$ e $Q$. Usando os operadores da Lógica Proposicional podemos escrever um número muito grande de fórmulas diferentes combinando estas duas proposições.

A coisa interessante sobre as fórmulas que conseguimos criar com apenas duas proposições é que cada uma dessas fórmulas tem uma Tabela Verdade com exatamente quatro linhas, $2^2=4$. Mesmo que isso pareça surpreendente, só existem dezesseis configurações possíveis para a última coluna de todas as Tabelas Verdades de todas as tabelas que podemos criar, $2^4=16$. Como resultado, muitas fórmulas compartilham a mesma configuração final em suas Tabelas Verdade. Todas as fórmulas que possuem a mesma configuração na última coluna são equivalentes.Terei ouvido um viva?

Com um pouco mais de formalidade podemos dizer que: considere as proposições $A$ e $B$. Estas proposições serão ditas logicamente equivalentes se, e somente se, a proposição $A \Leftrightarrow B$ for uma tautologia.

 **Exemplo: 1** Vamos mostrar que $P\rightarrow Q$ é logicamente equivalente A$\neg Q \rightarrow \neg P$.

 **Solução:** Para isso, verificaremos se a coluna do conectivo principal na Tabela Verdade para a proposição bicondicional formada por essas duas fórmulas contém apenas valores verdadeiros:

$$
\begin{array}{|c|c|c|c|c|}
 \hline
 P & Q & P \implies Q & \lnot Q \implies \lnot P & P \implies Q \iff \lnot Q \implies \lnot P \\
 \hline
 F & F & T & T & T \\
 \hline
 F & T & T & F & T \\
 \hline
 T & F & F & T & T \\
 \hline
 T & T & T & T & T \\
 \hline
 \end{array}
 $$

 Como a coluna da operação principal de $P\implies Q \iff \lnot Q \implies \lnot P$ contém apenas valores verdadeiros, a proposição bicondicional é uma tautologia, consequentemente e as fórmulas $P\implies Q$ e $\lnot Q \implies \lnot P$ são logicamente equivalentes.

 **Exemplo 2:** Vamos mostrar que $P\land Q$ não é logicamente equivalente A$P\lor Q$.

 **Solução**
 Verificando a Tabela Verdade:

$$
\begin{array}{|c|c|c|c|c|}
 \hline
 P & Q & P \land Q & P \lor Q & P \land Q \iff P \lor Q \\ \hline
 V & V & V & V & F \\ \hline
 V & F & F & V & F \\ \hline
 F & V & F & V & F \\ \hline
 F & F & F & F & F \\ \hline
 \end{array}
 $$

 Consequentemente, as fórmulas $P\land Q$ não são logicamente equivalentes $P\lor Q$.

 **Exemplo 3:** Vamos mostrar que $P\rightarrow Q$ é logicamente equivalente A$\neg P \lor Q$.

 **Solução**
 Verificando a Tabela Verdade:

$$
\begin{array}{|c|c|c|c|c|c|}
 \hline
 P & Q & \neg P & \neg P \lor Q & P \rightarrow Q \iff \neg P \lor Q\\
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

A contrapositiva pode ser lida como _se não $Q$, então não $P$_. Em outras palavras estamos dizendo: _Se $Q$ é falso, então $P$ é falso_. A contrapositiva de uma fórmula é importante porque, frequentemente, é mais fácil provar a contrapositiva de uma fórmula que a própria fórmula. E, como a contrapositiva é logicamente equivalente a sua formula, provar a contrapositiva é provar a fórmula. Como a contrapositiva de uma implicação e a própria implicação são logicamente equivalentes, se provamos uma, a outra está provada. Além disso, a contrapositva preserva a validade das implicações proposicionais. Finalmente, observe que a contrapositiva troca o antecedente pelo negação do consequente e vice-versa.

 **Exemplo 1:**
 A contrapositiva de $P\rightarrow (Q \lor R)$ é $\lnot(Q \lor R) \rightarrow \lnot P$.

 **Exemplo 2:**
 Dizemos que uma função é injetora se $x \neq y $implica $f(x) \neq f(y)$. A contrapositiva desta implicação é: se $f(x) = f(y)$ então $x = y$.

O Exemplo 2 é uma prova de conceito. Normalmente é mais fácil assumir $f(x) = f(y)$ e deduzir $x = y$ do que assumir $x \neq y$ e deduzir $f(x) \neq f(y)$. Isto pouco tem a ver com funções e muito com o fato de que $x \neq y$ geralmente não é uma informação útil.

O que torna a contrapositiva importante é que toda Fórmula Bem Formada é logicamente equivalente à sua contrapositiva. Consequentemente, se queremos provar que uma função é injetora, é suficiente provar que se $f(x) = f(y)$ então $x = y$.

A contrapositiva funciona para qualquer declaração condicional, e matemáticos gastam muito tempo provando declarações condicionais.

O que não podemos esquecer de jeito nenhum é que toda fórmula condicional terá a forma $P\rightarrow Q$. Mostramos que isso é logicamente equivalente A$\lnot Q \rightarrow \lnot P$ verificando a Tabela Verdade para a declaração bicondicional construída a partir dessas fórmulas. E que para obter a contrapositiva basta inverter antecedente e consequente e negar ambos. mantendo a relação lógica entre os termos da implicação.

### Recíproca

A recíproca, também conhecida como _conversa_ por alguns acadêmicos brasileiros, é obtida apenas invertendo antecedente e consequente. Então, considerando a recíproca da condicional$P\rightarrow Q$ será $ q \rightarrow P$. Destoando da contrapositiva a recíproca não é necessariamente equivalente à implicação original. Além disso, a contrapositiva preserva a equivalência lógica, a recíproca não.

 **Exemplo 1:**
 A conversa de $P\rightarrow (Q \lor R)$ será $(Q \lor R) \rightarrow P$.

 **Exemplo 2:**
 Dizemos que uma função é bem definida se cada entrada tem uma saída única. Assim, uma função é bem definida se $x = y$ implica $f(x) = f(y)$. Observe estas fórmulas:

 1. $f(x)$ é bem definida significa que $x = y \rightarrow f(x) = f(y)$.

 2. $f(x)$ é injetora significa que $f(x) = f(y) \rightarrow x = y$.

 Podemos ver que _$f(x)$ é bem definida_ é a recíproca de _$f(x)$ é injetora_.

Para provar uma bicondicional como _o número é primo se e somente se o número é ímpar_, um matemático frequentemente prova _se o número é primo, então o número é ímpar_ e depois prova a recíproca, _se o número é ímpar, então o número é primo_. Nenhuma dessas etapas pode ser pulada, pois uma implicação e sua recíproca podem não ser logicamente equivalentes. Por exemplo, pode-se facilmente mostrar que _se o número é par, então o número é divisível por 2_ não é logicamente equivalente à sua recíproca _se o número é divisível por 2, então o número é par_. Algumas fórmulas como _se 5 é ímpar, então 5 é ímpar_ são equivalentes às suas recíprocas por coincidência. Para resumir, uma implicação é sempre equivalente à sua contrapositiva, mas pode não ser equivalente à sua recíproca.

## Análise de Argumentos

Quando vimos regras de inferência, sem muitos floreios, definimos argumentos. mas, sem usar a palavra argumento em nenhum lugar. Vamos voltar um pouco. Definiremos um argumento proposicionalmente como sendo uma regra de inferência, então um argumento será definido por um conjunto de proposições. Quando estamos analisando argumentos chamamos as proposições de premissas logo:

$$
\frac{P_1, P_2, ..., P_n}{C}\\
$$

Onde o conjunto formado $P_1, P_2, ..., P_n$, chamado de antecedente, e $ c$, chamado de conclusão. Dizemos que o argumento será válido, só e somente se, a implicação definida por $P_1, P_2, ..., P_n \rightarrow C$ for uma tautologia. Neste caso, é muito importante percebermos que a conclusão de um argumento logicamente válido não é necessariamente verdadeira. A única coisa que a validade lógica garante é que se todas as premissas forem verdadeiras, a conclusão será verdadeira.

Podemos recuperar as regras de inferência e observá-las pelo ponto de vista da análise de argumentos. Se fizermos isso, vamos encontrar alguns formatos comuns:

**_Modus Ponens_**: _se é verdade que se eu estudar para o exame $P$, então eu passarei no exame, $Q$, e também é verdade que eu estudei para o exame $P$, então podemos concluir que eu passarei no exame $Q$_.

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

Se olharmos para a primeira linha, se $P$ é verdadeiro e $P→ Q$ é verdadeiro, então $Q$ é necessariamente verdadeiro, o que é exatamente a forma de _Modus Ponens_.

**_Modus Tollens_** : _se é verdade que se uma pessoa é um pássaro $P$, então essa pessoa pode voar $Q$, e também é verdade que essa pessoa não pode voar $\neg Q$, então podemos concluir que essa pessoa não é um pássaro $\neg P$. Ou:

Sejam $P$ e $Q$ Proposições. A forma do _Modus Tollens_ é a seguinte:

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

Se olharmos para a segunda linha, se $Q$ é falso e $P\rightarrow Q$ é verdadeiro, então $P$ é necessariamente falso, o que é exatamente a forma de _Modus Tollens_.

**Silogismo Hipotético** : _se é verdade que se eu acordar cedo $P$, então eu irei correr $Q$, e também é verdade que se eu correr $Q$, então eu irei tomar um café da manhã saudável $R$, podemos concluir que se eu acordar cedo $P$, então eu irei tomar um café da manhã saudável $R$_.
matematicamente teremos:

Sejam $P$, $Q$ e $R$ Proposições. A forma do Silogismo Hipotético é a seguinte:

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

## Finalmente, um Sistema de Prova

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

Além dos axiomas, usaremos apenas uma regra de inferência, o _Modus Ponens_. O _Modus Ponens_ está intimamente relacionado à proposição $(P \wedge (P \rightarrow Q)) \rightarrow Q$. Tanto a preposição quando a regra de inferência, de certa forma, dizem: "se $P$ e $P\rightarrow Q$ são verdadeiros, então $Q$ é verdadeiro". Esta proposição é um exemplo de uma tautologia, porque é verdadeira para cada configuração de $P$ e $Q$. A diferença é que esta tautologia é uma única proposição, enquanto o _Modus Ponens_ é uma regra de inferência que nos permite deduzir novas proposições a partir proposições já provadas.

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

3. $((A \rightarrow (A \rightarrow A)) \rightarrow (A \rightarrow A))$ (_Modus Ponens_ aplicado às linhas 1 e 2)

 Finalmente aplicamos a regra de _Modus Ponens_, que diz que se temos $A$ e também temos $A \rightarrow B$, então podemos deduzir $B$. As linhas 1 e 2 correspondem a $A$ e $A \rightarrow B$, respectivamente, e ao aplicar _Modus Ponens_, obtemos $B$, que é a fórmula na linha 3.

4. $(A \rightarrow (A \rightarrow A))$ (Axioma 1 com $A := A$ e $B := A$)

 De maneira similar à primeira linha, a quarta linha usa o primeiro axioma com $A := A$ e $B := A$.

5. $(A \rightarrow A)$(_Modus Ponens_ aplicado às linhas 3 e 4)

 Finalmente, aplicamos o _Modus Ponens_ às linhas 3 e 4 para obter a fórmula na última linha, que é o teorema que tentamos provar.

Então, o primeiro teorema está correto e podemos escrever $\vdash \mathfrak{L} A$.

**Prova 2**: vamos tentar provar $\vdash (\lnot B \rightarrow B) \rightarrow B$

1. $\lnot B \rightarrow \lnot B$ (Aplicação do Teorema 1 com $A := \lnot B$)

Aqui aplicamos o Teorema 1 (que é $A \rightarrow A$) substituindo $A$ Por $\lnot B$.

2. $((\lnot B \rightarrow \lnot B) \rightarrow ((\lnot B \rightarrow B) \rightarrow \lnot B))$ (Aplicação do Axioma 2 com $A := \lnot B$, $B := \lnot B$, e $ c := B$)

Agora aplicamos o segundo axioma, que é $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \rightarrow B) \rightarrow (A \rightarrow C))$, substituindo $A$ Por $\lnot B$, $B$ Por $\lnot B$ e $ c$ Por $B$.

3. $(\lnot B \rightarrow B) \rightarrow \lnot B$ (Aplicação do _Modus Ponens_ às linhas 1 e 2)

A regra de _Modus Ponens_ nos diz que se temos $A$ e também temos $A \rightarrow B$, então podemos deduzir $B$. As linhas 1 e 2 correspondem a $A$ e $A \rightarrow B$, respectivamente. Ao aplicar _Modus Ponens_, obtemos $B$, que é a fórmula nesta linha.

4. $(\lnot B \rightarrow B) \rightarrow B$ (Aplicação do Axioma 1 com $A := \lnot B$ e $B := B$)

Finalmente, aplicamos o primeiro axioma, que é $A \rightarrow (B \rightarrow A)$, substituindo $A$ Por $\lnot B$ e $B$ Por $B$ Para obter o teorema que estamos tentando provar.

**Prova 3**: vamos tentar novamente, desta vez com $\vdash ((A \land B) \rightarrow C)$

1. $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \land B) \rightarrow C)$ (Suposto axioma com $A := A$, $B := B$ e $ c := C$)

Aqui estamos assumindo que a fórmula $(A \rightarrow (B \rightarrow C)) \rightarrow ((A \land B) \rightarrow C)$ é um axioma. No entanto, esta fórmula **não** é um axioma do sistema $\mathfrak{L}$. Portanto, esta tentativa de provar o teorema é inválida desde o início.

2. $A \rightarrow (B \rightarrow C)$ (Hipótese)

Aqui estamos introduzindo uma hipótese, que é permissível. No entanto, uma hipótese deve ser descartada antes do final da prova e, nesta tentativa de prova, não é.

3. $(A \land B) \rightarrow C$ (_Modus Ponens_ aplicado às linhas 1 e 2)

Finalmente, tentamos aplicar a regra de inferência _Modus Ponens_ às linhas 1 e 2 para obter $(A \land B) \rightarrow C$. No entanto, como a linha 1 é inválida, esta aplicação de _Modus Ponens_ também é inválida.

Portanto, esta tentativa de provar o teorema $(A \land B) \rightarrow C$ **falha** porque faz suposições inválidas e usa regras de inferência de maneira inválida.

Esta última prova é interessante. Para o teorema $(A \land B) \rightarrow C$, não é possível provar diretamente no sistema $\mathfrak{L}$ sem a presença de axiomas adicionais ou a introdução de hipóteses adicionais. Que não fazem parte do sistema $\mathfrak{L}$.

O sistema $\mathfrak{L}$ é baseado em axiomas específicos e em uma única regra de inferência (_Modus Ponens_), como vimos. O teorema $((A \land B) \rightarrow C)$ não pode ser derivado apenas a partir dos axiomas do sistema $\mathfrak{L}$, pois a conjunção. Ou seja, o operador _OR_, ou $\lor $, disjunção, não está presente em nenhum dos axiomas do sistema $\mathfrak{L}$.

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

### Lema

Considere nossa primeira prova, provamos $A \rightarrow A$ e, a partir deste momento, $A \rightarrow A$ se tornou um Lema. Um lema é uma afirmação que é provada não como um fim em si mesma, mas como um passo útil para a prova de outros teoremas.

Em outras palavras, um lema é um resultado menor que serve de base para um resultado maior. Uma vez que um lema é provado, ele pode ser usado em provas subsequentes de teoremas mais complexos. Em geral, um lema é menos geral e menos notável do que um teorema.

Considere o seguinte Teorema: $\vdash_L (\lnot B \rightarrow B) \rightarrow B$, podemos prová-lo da seguinte forma:

1. $\lnot B \rightarrow \lnot B$ - Lembrando que $A := \lnot B$ do Teorema 1

2. $(\lnot B \rightarrow \lnot B) \rightarrow ((\lnot B \rightarrow B) \rightarrow B)$ - Decorrente do Axioma 3, onde $A := \lnot B$ e $B := B$

3. $((\lnot B \rightarrow B) \rightarrow B)$- Através do _Modus Ponens_
Justificativa: Linhas 1 e 2

A adoção de lemas é, na verdade, um mecanismo útil para economizar tempo e esforço. Ao invés de replicar o Teorema 1 na primeira linha dessa prova, nós poderíamos, alternativamente, copiar as 5 linhas da prova original do Teorema 1, substituindo todos os casos de $A$ Por $\lnot B$. As justificativas seriam mantidas iguais às da prova original do Teorema 1. A prova resultante, então, consistiria exclusivamente de axiomas e aplicações do _Modus Ponens_. No entanto, uma vez que a prova do Teorema 1 já foi formalmente documentada, parece redundante replicá-la aqui. E eis o motivo da existência e uso dos lemas.

### Hipóteses

Hipóteses são suposições ou proposições feitas como base para o raciocínio, sem a suposição de sua veracidade. Elas são usadas como pontos de partida para investigações ou pesquisas científicas. Essencialmente uma hipótese é uma teoria ou ideia que você pode testar de alguma forma. Isso significa que, através de experimentação e observação, uma hipótese pode ser provada verdadeira ou falsa.

Por exemplo, se você observar que uma planta está morrendo, pode formar a hipótese de que ela não está recebendo água suficiente. Para testar essa hipótese, você pode dar mais água à planta e observar se ela melhora. Se melhorar, isso suporta sua hipótese. Se não houver mudança, isso sugere que sua hipótese pode estar errada, e você pode então formular uma nova hipótese para testar.

Na lógica proposicional, uma hipótese é uma proposição (ou afirmação) que é assumida como verdadeira para o propósito de argumentação ou investigação. Obviamente, pode ser uma fórmula atômica, ou complexa, desde que seja uma Fórmula Bem Formada.

Em um sistema formal de provas, como o sistema $\mathfrak{L}$ uma hipótese é um ponto de partida para um processo de dedução. O objetivo é usar as regras do sistema para deduzir novas proposições a partir das hipóteses. Se uma proposição puder ser deduzida a partir das hipóteses usando as regras do sistema, dizemos que essa proposição é uma consequência lógica das hipóteses.
Se temos as hipóteses $P$ e $P\rightarrow Q$, podemos deduzir $Q$ usando o _Modus Ponens_. Nesse caso, $Q$ seria uma consequência lógica das hipóteses.

No contexto do sistema de provas $\mathfrak{L}$ e considerando apenas a lógica proposicional, **uma hipótese é uma proposição ou conjunto de proposições assumidas como verdadeiras, a partir das quais outras proposições podem ser logicamente deduzidas**.

**Exemplo 1:** considere o seguinte argumento:

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

# Lógica Predicativa

>A lógica é a técnica que usamos para adicionar convicção à verdade.
>Jean de la Bruyere

A Lógica Predicativa, coração e espírito da Lógica de Primeira Ordem, nos leva um passo além da Lógica Proposicional. Em vez de se concentrar apenas em proposições completas que são verdadeiras ou falsas, a lógica predicativa nos permite expressar proposições sobre objetos e as relações entre eles. Ela nos permite falar de maneira mais rica e sofisticada sobre o mundo.

Vamos lembrar que na Lógica Proposicional, cada proposição é um átomo indivisível. Por exemplo, 'A chuva cai' ou 'O sol brilha'. Cada uma dessas proposições é verdadeira ou falsa como uma unidade. Na lógica predicativa, no entanto, podemos olhar para dentro dessas proposições. Podemos falar sobre o sujeito - a chuva, o sol - e o predicado - cai, brilha. Podemos quantificar sobre eles: para todos os dias, existe um momento em que o sol brilha.

Enquanto a Lógica Proposicional pode ser vista como a aritmética do verdadeiro e do falso, a lógica predicativa é a álgebra do raciocínio. Ela nos permite manipular proposições de maneira muito mais rica e expressiva. Com ela, podemos começar a codificar partes substanciais da matemática e da ciência, levando-nos mais perto de nossa busca para decifrar o cosmos, um símbolo de lógica de cada vez.

## Introdução aos Predicados

Um predicado é como uma luneta que nos permite observar as propriedades de uma entidade. Um conjunto de lentes através do qual podemos ver se uma entidade particular possui ou não uma característica específica. A palavra predicado foi importada do campo da linguística e tem o mesmo significado: qualidade; característica. Por exemplo, ao observar o universo das letras através do telescópio do predicado _ser uma vogal_, percebemos que algumas entidades deste conjunto, como $A$ e $I $, possuem essa propriedade, enquanto outras, como $ g$ e $H$, não.

Um predicado não é uma afirmação absoluta de verdade ou falsidade. Divergindo das proposições, os predicados não são declarações completas. Pense neles como aquelas sentenças com espaços em branco, aguardando para serem preenchidos, que só têm sentido completo quando preenchidas:

1. O \_\_\_\_\_\_\_ está saboroso;

2. O \_\_\_\_\_\_\_ é vermelho;

3. \_\_\_\_\_\_\_ é alto.

Preencha as lacunas, como quiser desde que faça sentido, e perceba que, em cada caso, ao preencher estamos atribuindo uma qualidade a um objeto. Esses são exemplos de predicados do nosso cotidiano, que sinteticamente o conceito que queremos abordar. Na lógica, os predicados são artefatos que possibilitam examinar o mundo ao nosso redor de forma organizada e exata.

Um predicado pode ser entendido como uma função que recebe um objeto (ou um conjunto de objetos) e retorna um valor de verdade, $\{\text{verdadeiro ou falso}\}$. Esta função descreve uma propriedade que o objeto pode possuir. Isto é, se $P$ é uma função $P: U \rightarrow \\{\text{Verdadeiro, Falso}\\}$ Para um determinado conjunto $ u$ qualquer. Esse conjunto $ u$ é chamado de _universo ou domínio do discurso_, e dizemos que $P$ é um predicado sobre $ u$.

## Universo do Discurso

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

## Quantificadores

Embora a Lógica Proposicional seja um bom ponto de partida, a maioria das afirmações interessantes em matemática contêm variáveis definidas em domínios maiores do que apenas $\\{\text{Verdadeiro}, \text{Falso}\\}$. Por exemplo, a afirmação _$x \text{é uma potência de } 2$_ não é uma proposição. Não temos como definir a verdade dessa afirmação até conhecermos o valor de $x$. Se $P(x)$ é definido como a afirmação _$x \text{é uma potência de } 2$_, então $P(8)$ é verdadeiro e $P(7)$ é falso.

Para termos uma linguagem lógica que seja suficientemente flexível para representar os problemas que encontramos no Universo real, o Universo em que vivemos, precisaremos ser capazes de dizer quando o predicado $P$ ou $Q$ é verdadeiro para valores diferentes em seus argumentos. Para tanto, vincularemos as variáveis aos predicados usando operadores para indicar quantidade, chamados de quantificadores.

Os quantificadores indicam se a sentença que estamos criando se aplica a todos os valores possíveis do argumento, _quantificação universal_, ou se esta sentença se aplica a um valor específico, _quantificação existencial_. Usaremos esses quantificadores para fazer declarações sobre **todos os elementos** de um universo de discurso específico, ou para afirmar que existe **pelo menos um elemento** do universo do discurso que satisfaz uma determinada qualidade.

Vamos remover o véu da dúvida usando como recurso metafórico uma experiência humana, social, comum e popular: imaginemos estar em uma festa e o anfitrião lhe pede para verificar se todos os convidados têm algo para beber. Você, prestativo e simpático, começa a percorrer a sala, verificando cada pessoa. Se você encontrar pelo menos uma pessoa sem bebida, você pode imediatamente dizer _nem todos têm bebidas_. mas, se você verificar cada convidado e todos eles tiverem algo para beber, você pode dizer com confiança _todos têm bebidas_. Este é o conceito do quantificador universal, matematicamente representado por $\forall$, que lemos como _para todo_.

A festa continua e o anfitrião quer saber se alguém na festa está bebendo champanhe. Desta vez, assim que você encontrar uma pessoa com champanhe, você pode responder imediatamente _sim, alguém está bebendo champanhe_. Você não precisa verificar todo mundo para ter a resposta correta. Este é o conceito do quantificador existencial, denotado por $\exists $, que lemos _existe algum_.

Os quantificadores nos permitem fazer declarações gerais, ou específicas, sobre os membros de um universo de discurso, de uma forma que seria difícil, ou impossível, sem estes operadores especiais.

## Quantificador Universal

O quantificador universal $\forall$, lê-se _para todo_, indica que uma afirmação deve ser verdadeira para todos os valores de uma variável dentro de um universo de discurso definido para a criação de uma sentença contendo um predicado qualquer. Por exemplo, a preposição clássica _todos os humanos são mortais_ pode ser escrita como $\forall x Humano(x) \rightarrow Mortal(x)$. Ou recorrendo a um exemplo com mais de rigor matemático, teríamos o predicado se _$x$ é positivo então $x + 1 $ é positivo_, que pode ser escrito $\forall x (x > 0 \rightarrow x + 1 > 0)$. Neste último exemplo temos Quantificadores, Lógica Predicativa, Lógica Proposicional e Teoria dos Conjuntos em uma sentença.

O quantificador universal pode ser representado usando apenas a Lógica Proposicional, com uma pequena trapaça. A afirmação $\forall x P(x)$ é, de certa forma, a operação $\wedge $, _AND_ aplicada a todos os elementos do universo do discurso. Ou seja, o predicado:

$$\forall x \{x:\in \mathbb{N}\} : P(x)$$

Pode ser escrito como:

$$P(0) \land P(1) \land P(2) \land P(3) \land \ldots $$

Onde $P(0), P(1), P(2), P(3) \ldots $ representam a aplicação do predicado $P$A todos os elementos $x$ do conjunto $\mathbb{N}$. A trapaça fica por conta de que, em Lógica Proposicional, não podemos escrever expressões com um número infinito de termos. Portanto, a expansão em conjunções de um predicado $P$ em um Universo de Discurso, $ u$, não é uma Fórmula Bem Formada se a cardinalidade de $ u$ for infinita. De qualquer forma, podemos usar esta interpretação informal para entender o significado de $\forall x P(x)$.

A representação do Quantificador Universal como uma conjunção **não é uma Fórmula Bem Formada** a não ser que o Universo do Discurso seja não infinito. Neste caso, teremos uma conjunção que chamaremos de **Conjunção Universal**:

$$
\forall x (P(x) \land Q(x))
$$

Isso significa que para todo $x$ no domínio, as propriedades $P$, $Q$, e outras listadas são todas verdadeiras. É uma maneira de expressar que todas as condições listadas são verdadeiras para cada elemento no domínio. Esta fórmula será usada para simplificar sentenças, ou para criar formas normais.

Vamos voltar um pouco. O quantificador universal $\forall x P(x)$Afirma que a proposição $P(x)$ é verdadeira para todo, e qualquer, valor possível de $x$ como elemento de um conjunto $ u$. Uma forma de interpretar isso é pensar em $x$ como uma variável que pode ter qualquer valor dentro do universo do discurso.

Para validar $\forall x P(x)$ escolhemos o pior caso possível para $x$, todos os valors que suspeitamos possa fazer $P(x)$ falso. Se conseguirmos provar que $P(x)$ é verdadeira nestes casos específicos, então $\forall x P(x)$ deve ser verdadeira. Novamente, vamos recorrer a exemplos na esperança de explicitar este conceito.

**Exemplo 1**: todos os números reais são maiores que 0. (Universo do discurso: $\{x \in \mathbb{R}\}$)

$$\forall x (x \in \mathbb{R} \rightarrow x > 0)$$

>Observe que este predicado, apesar de estar corretamente representado, é $Falso$.

**Exemplo 2**: todos os triângulos em um plano euclidiano têm a soma dos ângulos internos igual a 180 graus. (Universo do discurso: $x$ é um triângulo em um plano euclidiano)

$$\forall x (Triângulo(x) \rightarrow \Sigma_{i=1}^3 ÂnguloInterno_i(x) = 180^\circ)$$

**Exemplo 3**: todas as pessoas com mais de 18 anos podem tirar carteira de motorista." (Universo do discurso: $x$ é uma pessoa no Brasil)

$$\forall x (Pessoa(x) \land Idade (x) \geq 18 \rightarrow PodeTirarCarteira(x))$$

**Exemplo 4**: todo número par maior que 2 pode ser escrito como a soma de dois números primos. (Universo do discurso: $\{x \in \mathbb{Z}\}$

$$\forall x\,(Par(x) \land x > 2 \rightarrow \exists a\exists b\, (Primo(a) \land Primo(b) \land x = a + b))$$

**Exemplo 5**: para todo número natural, se ele é múltiplo de 4 e múltiplo de 6, então ele também é múltiplo de 12. (Universo do discurso: $\{x \in \mathbb{N}\}$)

$$\forall x\,((\exists a\in\Bbb N\,(x = 4a) \land \exists b\in\Bbb N\,(x = 6b)) \rightarrow \exists c\in\Bbb N\,(x = 12c))$$

O quantificador universal nos permite definir uma Fórmula Bem Formada representando todos os elementos de um conjunto, um universo do discurso, em relação a uma qualidade específica, um predicado. Esta é um artefato lógico interessante, mas não suficiente.

## Quantificador Existencial

O quantificador existencial, $\exists $ nos permite fazer afirmações sobre a existência de objetos com certas propriedades, sem precisarmos especificar exatamente quais objetos são esses. Vamos tentar remover os véus da dúvida com um exemplo simples.

Consideremos a sentença: _existem humanos mortais_. Com um pouco mais de detalhe e matemática, podemos escrever isso como: existe pelo menos um $x$ tal que $x$ é humano e mortal. Para escrever a mesma sentença com precisão matemática teremos:

$$\exists x \text{Humano}(x) \land \text{Mortal}(x)$$

Lendo por partes: _existe um $x$, tal que $x$ é humano AND $x$ é mortal_. Em outras palavras, existe pelo menos um humano que é mortal.

Note duas coisas importantes:

1. Nós não precisamos dizer exatamente quem é esse humano mortal. Só afirmamos que existe um. O operador $\exists $  captura essa ideia.

2. Usamos _AND_ ($\land $), não implicação ($\rightarrow $). Se usássemos $\rightarrow $, a afirmação ficaria muito mais fraca. Veja:

$$\exists x \text{Humano}(x) \rightarrow \text{Mortal}(x)$$

Que pode ser lido como: _existe um $x$ tal que, SE $x$ é humano, ENTÃO $x$ é mortal_. Essa afirmação é verdadeira em qualquer universo que contenha um unicórnio de bolinhas roxas imortal. Porque o unicórnio não é humano, então $\text{Humano}(\text{unicórnio})$ é falsa, e a implicação $\text{Humano}(x) \rightarrow \text{Mortal}(x)$ é verdadeira. Não entendeu? Volte dois parágrafos e leia novamente. Repita!

Portanto, é crucial usar o operador $\land $, e não $\rightarrow $ quando trabalhamos com quantificadores existenciais. O $\land $ garante que a propriedade se aplica ao objeto existente definido pelo $\exists $.

Assim como o quantificador universal, $\forall $, o quantificador existencial, $\exists $ , também pode ser restrito a um universo específico, usando a notação de pertencimento:

$$\exists x \in \mathbb{Z} : x = x^2$$

Esta sentença afirma a existência de pelo menos um inteiro $x$ tal que $x$ é igual ao seu quadrado. Novamente, não precisamos dizer qual é esse inteiro, apenas que ele existe dentro do conjunto dos inteiros. Existe?

De forma geral, o quantificador existencial serve para fazer afirmações elegantes sobre a existência de objetos com certas qualidades, sem necessariamente conhecermos ou elencarmos todos esses objetos. Isso agrega mais qualidade a representação do mundo real que podemos fazer com a Lógica de Primeira Ordem.

Talvez, alguns exemplos possam ajudar no seu entendimento:

**Exemplo 1**: existe um mamífero que não respira ar.

$$
\exists x (mamífero(x) \land \neg RespiraAr(x))
$$

**Exemplo 2**: existe uma equação do segundo grau com três raízes reais.

$$
\exists x (Eq2Grau(x) \land |\text{RaízesReais}(x)| = 3)
$$

**Exemplo 3**: existe um número primo que é par.

$$
\exists x (Primo(x) \land Par(x))
$$

**Exemplo 4**: existe um quadrado perfeito que pode ser escrito como o quadrado de um número racional.

$$
\exists x (QuadPerfeito(x) \land \exists a \in \mathbb{Q} \ (x = a^2))
$$

**Exemplo 5**: existe um polígono convexo em que a soma dos ângulos internos não é igual A$(n-2)\cdot180^{\circ}$.

$$
\exists x (\text{PolígonoConvexo}(x) \land \sum_{i=1}^{n} \text{ÂnguloInterno}_i(x) \neq (n-2)\cdot 180^{\circ})
$$

>Novamente, observe que este predicado é $falso$. Todos os polígonos convexos têm a soma dos ângulos internos igual a $(n−2)cdot 180$, onde $𝑛$ é o número de lados do polígono.

### Equivalências Interessantes

Estudando o quantificador universal encontramos duas equivalências interessantes:

$$\lnot \forall x P(x) \leftrightarrow \exists x \lnot P(x)$$

$$\lnot \exists x P(x) \leftrightarrow \forall x \lnot P(x)$$

Essas equivalências são essencialmente as versões quantificadas das **Leis de De Morgan**. A primeira diz que nem todos os humanos são mortais, isso é equivalente a encontrar algum humano que não é mortal. A segunda diz que para mostrar que nenhum humano é mortal, temos que mostrar que todos os humanos não são mortais.

Podemos representar uma declaração $\exists x P(x)$ como uma expressão _OU_. Por exemplo, $\exists x \in \mathbb{N} : P(x)$ Poderia ser reescrito como:

$$P(0) \lor P(1) \lor P(2) \lor P(3) \lor \ldots $$

Lembrado do problema que encontramos quando fizemos isso com o quantificador $\forall $: não podemos representar fórmulas sem fim em Lógica de Primeira Ordem. mas, novamente esta notação, ainda que inválida, nos permite entender melhor o quantificador existencial. Caso o Universo do Discurso seja não infinito, limitado e contável, teremos a **Disjunção Existencial** uma expressão na lógica de primeiro grau que afirma que existe pelo menos um elemento em um domínio que satisfaz uma ou mais propriedades. A forma geral de uma disjunção existencial é:

$$
\exists x (P(x) \lor Q(x))
$$

Isso significa que existe pelo menos um $x$ no domínio que satisfaz a propriedade $P$, ou a propriedade $Q$, ou ambas, ou outras propriedades listadas. É uma maneira de expressar que pelo menos uma das condições listadas é verdadeira para algum elemento no domínio.

A expansão de $\exists $ usando $\lor $ destaca que a proposição $P(x)$ é verdadeira se pelo menos um valor de $x$ dentro do universo do discurso atender ao predicado $P$. O que a expansão de exemplo está dizendo é que existe pelo menos um número natural $x$ tal que $P(x)$ é verdadeiro. Não precisamos saber exatamente qual é esse $x$. Apenas que existe um elemento dentro de $\mathbb{N}$ que atende o predicado.

O quantificador existencial não especifica o objeto dentro do universo determinado. Esse operador permite fazer afirmações elegantes sobre a existência de objetos com certas características, certas qualidades, ou ainda, certos predicados, sem necessariamente conhecermos exatamente quais são esses objetos.

## Dos Predicados à Linguagem Natural

Ao ler Fórmula Bem Formada contendo quantificadores, **lemos da esquerda para a direita**. Por exemplo, $\forall x$ Pode ser lido como _para todo objeto $x$ no universo do discurso onde este objeto está implícito, o seguinte se mantém_. Por outro lado, o quantificador $\exists x$ Pode ser lido como _existe um objeto $x$ no universo que satisfaz o seguinte_ ou ainda _para algum objeto $x$ no universo, o seguinte se mantém_. A forma como lê-mos determina como entenderemos as Fórmulas Bem Formadas que incluam quantificadores.

A conversão de uma Fórmula Bem Formada em sentença, não necessariamente resulta em boas expressões em linguagem natural. Apesar disso, para entender a sentença o melhor caminho passa sempre pela leitura, em linguagem natural da Fórmula Bem Formada. Por exemplo: sejA$ u$, universo do discurso, o conjunto de todos os aviões já fabricados e sejA$F(x,y)$ o predicado denotando _$x$ voa mais rápido que $y$_, poderemos ter:

- $\forall x \forall y F(x,y)$ Pode ser lido como _Para todo avião $x$: $x$ é mais rápido que todo (no sentido de qualquer) avião $y$_.

- $\exists x \forall y F(x,y)$ Pode ser lido inicialmente como _Para algum avião $x$ que para todo avião $y$, $x$ é mais rápido que $y$_.

- $\forall x \exists y F(x,y)$ representa _Existe um avião $x$ ou tal que para todo avião $y$, $x$ é mais rápido que $y$_.

- $\exists x \exists y F(x,y)$ se lê _Para algum avião $x$ existe um avião $y$ tal que $x$ é mais rápido que $y$_.

As quatro sentenças expressam o mesmo contexto, embora sejam redigidas de maneiras distintas. Ao escrevermos, optamos pela forma mais transparente segundo nossa própria opinião. Quando a situação é de leitura, a escolha não existe, é necessário entender, e nesse cenário, a recomendação seria começar pela escrita da sentença em linguagem natural. Trata-se de um processo, e com o passar do tempo, torna-se mais simples.

### Exercícios de Conversão de Linguagem Natural em Expreessões Predicativas

**Sentença 1**: _Todo matemático que é professor tem alunos que são brilhantes e interessados._
  
$$
\forall x ((\text{Matemático}(x) \wedge \text{Professor}(x)) \rightarrow \exists y (\text{Aluno}(y) \wedge \text{Brilhante}(y) \wedge \text{Interessado}(y) \wedge \text{Ensina}(x, y)))
$$

$$
\forall x (\text{Matemático}(x) \rightarrow (\text{Professor}(x) \rightarrow \exists y (\text{Aluno}(y) \wedge \text{Brilhante}(y) \wedge \text{Interessado}(y) \wedge \text{Ensina}(x, y))))
$$

**Sentença 2**: _Alguns engenheiros não são nem ricos nem felizes._
  
$$
\exists x (\text{Engenheiro}(x) \wedge \neg (\text{Rico}(x) \vee \text{Feliz}(x)))
$$
  
$$
\exists x (\text{Engenheiro}(x) \wedge \neg\text{Rico}(x) \wedge \neg\text{Feliz}(x))
$$

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

## Ordem de Aplicação dos Quantificadores

Quando mais de uma variável é quantificada em uma Fórmula Bem Formada como $\forall y\forall x P(x,y)$, elas são aplicadas de dentro para fora, ou seja, a mais próxima da fórmula atômica é aplicada primeiro. Assim, $\forall y\forall x P(x,y)$ se lê _existe um $y$ tal que para todo $x$, $P(x,y)$ se mantém_ ou _para algum $y$, $P(x,y)$ se mantém para todo $x$_.

As posições dos mesmos tipos de quantificadores podem ser trocadas sem afetar o valor lógico, desde que não haja quantificadores do outro tipo entre os que serão trocados.

Por exemplo, $\forall x\forall y\forall z P(x,y,z)$ é equivalente A$\forall y\forall x\forall z P(x,y,z)$, $\forall z\forall y\forall x P(x,y,z)$. O mesmo vale para o quantificador existencial.

No entanto, as posições de quantificadores de tipos diferentes **não** podem ser trocadas. Por exemplo, $\forall x\exists y P(x,y)$ **não** é equivalente A$\exists y\forall x P(x,y)$. Por exemplo, sejA$P(x,y)$ representando $x < y$ Para o conjunto dos números como universo. Então, $\forall x\exists y P(x,y)$ se lê _para todo número $x$, existe um número $y$ que é maior que $x$_, o que é verdadeiro, enquanto $\exists y\forall x P(x,y)$ se lê _existe um número que é maior que todo (qualquer) número_, o que não é verdadeiro.

### Negação dos Quantificadores

Existe uma equivalência entre as negações dos quantificadores. De tal forma que:

1. **Negação do Quantificador Universal ($\forall $):** A negação de uma afirmação universal significa que existe pelo menos um caso no Universo do Discurso, onde a afirmação não é verdadeira. Isso pode ser expresso pela seguinte equivalência:

$$\neg \forall x \, P(x) \equiv \exists x \, \neg P(x)$$

Em linguagem natural podemos entender como: negar que _para todos os $x$, $P(x)$ é verdadeiro_ é equivalente a afirmar que _existe algum $x$ tal que $P(x)$ não é verdadeiro_.

2. **Negação do Quantificador Existencial ( $\exists $ ):** A negação de uma afirmação existencial significa que a afirmação não é verdadeira para nenhum caso no Universo do Discurso. Isso pode ser expresso pela seguinte equivalência:

$$\neg \exists x \, P(x) \equiv \forall x \, \neg P(x)$$

Ou seja, negar que _existe algum $x$ tal que $P(x)$ é verdadeiro_ é equivalente a afirmar que _para todos os $x$, $P(x)$ não é verdadeiro_.

Vamos tentar entender estas negações. Considere as expressões $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$. Essas fórmulas se aplicam a qualquer predicado $P$, e possuem o mesmo valor de verdade para qualquer $P$.

Na lógica proposicional, poderíamos simplesmente verificar isso com uma tabela verdade, mas aqui, não podemos. Não existem proposições, conectadas por $\land $, $\lor $, para construir uma tabela e não é possível determinar o valor verdade de forma genérica para uma determinada variável.

Vamos tentar entender isso com linguagem natural: afirmar que $\neg (\forall x P(x))$ é verdadeiro significa que não é verdade que $P(x)$ se aplica a todas as possíveis entidades $x$. Deve haver alguma entidade $A$ Para a qual$P(a)$ é falso. Como $P(a)$ é falso, $\neg P(a)$ é verdadeiro. Isso significa que $\exists x (\neg P(x))$ é verdadeiro. Portanto, a verdade de $\neg (\forall x P(x))$implica a verdade de $\exists x (\neg P(x))$.

Se $\neg (\forall x P(x))$ é falso, então $\forall x P(x)$ é verdadeiro. Como $P(x)$ é verdadeiro para todos os $x$, $\neg P(x)$ é falso para todos os $x$. Logo, $\exists x (\neg P(x))$ é falso.

Os valores de verdade de $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$ são os mesmos. Como isso é verdadeiro para qualquer predicado $P$, essas duas fórmulas são logicamente equivalentes, e podemos escrever $\neg (\forall x P(x)) \equiv \exists x (\neg P(x))$.

Muita lógica? Que tal se tentarmos novamente, usando um pouco mais de linguagem natural.
Considere as expressões lógicas $\neg (\forall x P(x))$ e $\exists x (\neg P(x))$. Para ilustrar essas fórmulas, vamos usar um exemplo com um predicado $P(x)$ que se aplica a uma entidade $x$ se _$x$ é feliz_.

A expressão $\forall x P(x)$ significa que _todos são felizes_, enquanto $\neg (\forall x P(x))$ significa que _não é verdade que todos são felizes_. Ou seja, deve haver pelo menos uma pessoa que não está feliz.

A expressão $\exists x (\neg P(x))$ significa que _existe alguém que não está feliz_. Você pode ver que isso é apenas outra maneira de expressar a ideia contida em $\neg (\forall x P(x))$.

A afirmação de que _não é verdade que todos estão felizes_ implica que deve haver alguém que não está feliz. Se a primeira afirmação é falsa (ou seja, todos estão felizes), então a segunda afirmação também deve ser falsa.

Portanto, as duas fórmulas têm o mesmo valor verdade. Elas são logicamente equivalentes e podem ser representadas como $\neg (\forall x P(x)) \equiv \exists x (\neg P(x))$. Esta equivalência reflete uma relação profunda e intuitiva em nosso entendimento de declarações sobre entidades em nosso mundo.

<table style="width: 100%; margin: auto; border-collapse: collapse;">
  <tr>
    <th style="text-align: center; background-color: #eeeeee;">Expressão</th>
    <th style="text-align: center; background-color: #eeeeee;">Equivalência</th>
  </tr>
  <tr>
    <td style="text-align: center; width: 50%;">
    $\forall x P(x)$</td>
    <td style="text-align: center; width: 50%;">
    $\neg \exists x \neg P(x)$</td>
  </tr>
  <tr style="background-color: #eeeeee;">
    <td style="text-align: center; width: 50%;">$\exists x \, P(x)$</td>
    <td style="text-align: center; width: 50%;" >$\neg \forall x \, \neg P(x)$</td>
  </tr>
  <tr>
    <td style="text-align: center; width: 50%;" >$\neg \forall x \, P(x)$</td>
    <td style="text-align: center; width: 50%;" >$\exists x \, \neg P(x)$</td>
  </tr>
  <tr style="border-bottom: 2px solid gray;">
    <td style="text-align: center; width: 50%;">$\neg \exists x \, P(x)$</td>
    <td style="text-align: center; width: 50%;" >$\forall x \, \neg P(x)$</td>
  </tr>
</table>
<legend style="font-size: 1em; text-align: center;
 margin-bottom: 20px;">Tabela 4 - Equivalências entre Quantificadores.</legend>

## Regras de Inferência usando Quantificadores

As regras de inferência com quantificadores lidam especificamente com as proposições que envolvem quantificadores. Estas regras nos permitem fazer generalizações ou especificações, transformando proposições universais em existenciais, e vice-versa. Compreender essas regras é essencial para aprofundar o entendimento da estrutura da lógica, o que nos permite analisar e construir argumentos mais complexos de maneira precisa e coerente.

Nos próximos tópicos, exploraremos essas regras em detalhes, observando como elas interagem com os quantificadores universal e existencial.

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
\forall x P(x)
$$

$$
\begin{aligned}
&\forall x P(x)\\
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
  - Conclusão: logo, _o triângulo $ABC $ tem 180 graus_.

$$
\begin{aligned}
&\forall t(T(t) \rightarrow 180^\circ(t))\\
\hline
&180^\circ(\text{Triângulo} ABC)
\end{aligned}
$$

- Testar propriedades em membros de conjuntos. Por exemplo:

  - Proposição: _todo inteiro é maior que seu antecessor_.
  - Conclusão: logo, _$5 $ é maior que $4$_.

$$
\begin{aligned}
&\forall n (\mathbb{Z}(n) \rightarrow (n > n-1))\\
\hline
&5 > 4
\end{aligned}
$$

### Generalização Existencial

A regra de Generalização Existencial permite inferir que algo existe a partir de uma afirmação concreta. Esta regra nos permite generalizar de exemplos específicos para a existência geral.

$$
P(a)
$$

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

### Instanciação Existencial

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

## Problemas Interessantes Resolvidos com Lógica Proposicional e Predicativa

Aqui estão cinco quebra-cabeças clássicos juntamente com suas soluções usando Lógica de Primeira Ordem

1. **Quebra-cabeça: O Mentiroso e o Verdadeiro**
 Você encontra dois habitantes: $A$ e $B$. Você sabe que um sempre diz a verdade e o outro sempre mente, mas você não sabe quem é quem. Você pergunta a $A$, _Você é o verdadeiro?_ A responde, mas você não consegue ouvir a resposta dele. $B$ então te diz, _A disse que ele é o mentiroso_.

 **Solução**: $B$ deve ser o verdadeiro e $A$ deve ser o mentiroso. Se $A$fosse o verdadeiro, ele nunca diria que é o mentiroso. Portanto, $A$ deve ser o mentiroso e $B$ deve ser o verdadeiro, independentemente do que $A$ disse.

 Usando Lógica de Primeira Ordem: Vamos denotar a resposta de $A$ como $A$. Então o predicado $falaVerdade (A, a)$ será falso porque um verdadeiro nunca pode dizer que é um mentiroso. Portanto, $\neg falaVerdade (A, a)$ e daí, $falaVerdade (B, \neg a)$.

1. **Quebra-cabeça: As Três Lâmpadas**
 Existem três lâmpadas incandescentes em uma sala, e existem três interruptores fora da sala. Você pode manipular os interruptores o quanto quiser, mas só pode entrar na sala uma vez. Como você pode determinar qual interruptor opera qual lâmpada?

 **Solução**: ligue um interruptor e espere um pouco. Então desligue esse interruptor e ligue um segundo interruptor. Entre na sala. A lâmpada que está acesa corresponde ao segundo interruptor. A lâmpada que está desligada e quente corresponde ao primeiro interruptor. A lâmpada que está desligada e fria corresponde ao terceiro interruptor.

 Usando Lógica de Primeira Ordem:
 Vamos denotar os interruptores como $s1, s2, s3$ e as lâmpadas como $b1, b2, b3$. Podemos definir predicados $On(b, s)$ e $Hot(b)$.

$$On(b1, s2) \land Hot(b2) \land \neg (On(b3) \lor Hot(b3))$$

1. **Quebra-cabeça: O Agricultor, a Raposa, o Ganso e o Grão**
 Um agricultor quer atravessar um rio e levar consigo uma raposa, um ganso e um saco de grãos. O barco do agricultor só lhe permite levar um item além dele mesmo. Se a raposa e o ganso estiverem sozinhos, a raposa comerá o ganso. Se o ganso e o grão estiverem sozinhos, o ganso comerá o grão. Como o agricultor pode levar todas as suas posses para o outro lado do rio?

 **Solução**: o agricultor leva o ganso através do rio primeiro, deixando a raposa e o grão no lado original. Ele deixa o ganso no outro lado e volta para pegar a raposa. Ele deixa a raposa no outro lado, mas leva o ganso de volta ao lado original para pegar o grão. Ele deixa o grão com a raposa no outro lado. Finalmente, ele retorna ao lado original mais uma vez para pegar o ganso.

 Usando Lógica de Primeira Ordem:
 Podemos definir predicados $mesmoLado(x, y)$ e $come (x, y)$.
 A solução envolve a sequência de ações que mantêm as seguintes condições:

$$\neg (mesmoLado(Raposa, Ganso) \land \neg mesmoLado(Raposa, Fazendeiro))$$

$$\neg (mesmoLado(Ganso, Grãos) \land \neg mesmoLado(Ganso, Fazendeiro))$$

4. **Quebra-cabeça: O Problema da Ponte e da Tocha**
 Quatro pessoas chegam a um rio à noite. Há uma ponte estreita, mas ela só pode conter duas pessoas de cada vez. Eles têm uma tocha e, por ser noite, a tocha tem que ser usada ao atravessar a ponte. A pessoa $A$ Pode atravessar a ponte em um minuto, $B$ em dois minutos, $C$ em cinco minutos e $D$ em oito minutos. Quando duas pessoas atravessam a ponte juntas, elas devem se mover no ritmo da pessoa mais lenta. Qual é a maneira mais rápida para todos eles atravessarem a ponte?

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

  $$\text{Todo cachorro tem uma pessoa que é seu dono.}

12. $$\exists x (Música(x) \land (\forall y (Pessoa(y) \rightarrow Gosta(y, x))))$$

  $$\text{Existe uma música que todas as pessoas gostam.}$$

13. $$\forall x (Estudante(x) \rightarrow (\exists y (Professor(y) \land Ensina(y, x))))$$

  $$\text{Para todo estudante, existe um professor que o ensina.}$$

14. $$(\exists x (Médico(x) \land Competente(x))) \land (\forall y (Médico(y) \rightarrow Ocupado(y)))$$

  $$\text{Existe um médico que é competente, e todo médico é ocupado.}$$

15. $$(\forall x (Artista(x) \rightarrow Criativo(x))) \rightarrow (\exists y (Pintor(y) \land Criativo(y)))$$

  $$\text{Se todo artista é criativo, então existe um pintor que é criativo.}$$  

## Formas Normais

As formas normais, em sua essência, são um meio de trazer ordem e consistência à maneira como representamos proposições na Lógica Proposicional. Elas oferecem uma estrutura formalizada para expressar proposições, uma convenção que simplifica a comparação, análise e simplificação de proposições lógicas.

Consideremos, por exemplo, a tarefa de comparar duas proposições para determinar se são equivalentes. Sem uma forma padronizada de representar proposições, essa tarefa pode se tornar complexa e demorada. No entanto, ao utilizar as formas normais, cada proposição é expressa de uma maneira padrão, tornando a comparação direta e simples. Além disso, as formas normais também desempenham um papel crucial na simplificação de proposições. Ao expressar uma proposição em sua forma normal, é mais fácil identificar oportunidades de simplificação, removendo redundâncias ou simplificando a estrutura lógica. As formas normais não são apenas uma ferramenta para lidar com a complexidade da Lógica Proposicional, mas também uma metodologia que facilita a compreensão e manipulação de proposições lógicas.

Existem várias formas normais na Lógica Proposicional, cada uma com suas próprias regras e aplicações. Aqui estão algumas das principais:

1. **Forma Normal Negativa (PNN)**: Uma proposição está na Forma Normal Negativa se as operações de negação $\neg $Aparecerem apenas imediatamente antes das variáveis. Isso é conseguido aplicando as leis de De Morgan e eliminando as duplas negações.

$$
\neg (A \wedge B) \equiv (\neg A \vee \neg B)
$$

2. **Forma Normal Conjuntiva (PNC)**: Uma proposição está na Forma Normal Conjuntiva se for uma conjunção, operação _E_, $\wedge $, de uma ou mais cláusulas, onde cada cláusula é uma disjunção, operação _OU_, $\vee $, de literais. Em outras palavras, é uma série de cláusulas conectadas por _Es_, onde cada cláusula é composta de variáveis conectadas por _OUs_.

$$
(A \vee B) \wedge (C \vee D) \equiv (A \wedge C) \vee (A \wedge D) \vee (B \wedge C) \vee (B \wedge D)
$$

3. **Forma Normal Disjuntiva (PND)**: uma proposição está na Forma Normal Disjuntiva se for uma disjunção de uma ou mais cláusulas, onde cada cláusula é uma conjunção de literais. Ou seja, é uma série de cláusulas conectadas por _ORs_, onde cada cláusula é composta de variáveis conectadas por _ANDs_.

$$
(A \wedge B) \vee (C \wedge D) \equiv (A \vee C) \wedge (A \vee D) \wedge (B \vee C) \wedge (B \vee D)
$$

4. **Forma Normal Prenex (PNP)**: uma proposição está na Forma Normal Prenex se todos os quantificadores, para a Lógica de Primeira Ordem, estiverem à esquerda, precedendo uma matriz quantificadora livre. Esta forma é útil na Lógica de Primeira Ordem e na teoria da prova.

$$
\exists x \forall y (P(x,y) \wedge Q(y)) \equiv \forall y \exists x (P(x,y) \wedge Q(y))
$$

5. **Forma Normal Skolem (PNS)**: na Lógica de Primeira Ordem, uma fórmula está na Forma Normal de Skolem se estiver na Forma Normal Prenex e se todos os quantificadores existenciais forem eliminados. Isto é realizado através de um processo conhecido como Skolemização.

$$
\forall x (P(x,y)) \equiv P(x, f(x))
$$

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

#### Estrutura da Forma Normal Negativa

Uma fórmula está na Forma Normal Negativa se:

- todos os operadores de negação $\neg $ são aplicados diretamente aos átomos, variáveis ou constantes.
- usaremos apenas a negação $\neg $, a conjunção $\land $, e a disjunção $\lor $.

#### Conversão para Forma Normal Negativa

Converter uma fórmula para a FNN envolve os seguintes passos:

1. **Eliminar os Bicondicionais:** substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
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

#### Estrutura da Forma Normal Disjuntiva

Uma fórmula está na Forma Normal Disjuntiva se puder ser escrita como:

$$(C_1 \land C_2 \land \ldots) \lor (D_1 \land D_2 \land \ldots) \lor$$

Onde cada $C_i$ e $D_i$ é um literal. Ou seja, é uma variável ou sua negação. Com um pouco mais de formalidade matemática podemos afirmar que uma Fórmula Bem Formada está na Forma Normal Disjuntiva quando está na forma:

$$\bigvee_{i=1}^{m} \left( \bigwedge_{j=1}^{n} L_{ij} \right)$$

#### Conversão para Forma Normal Disjuntiva

Converter uma fórmula para a FND geralmente envolve os seguintes passos:

1. **Eliminar os Bicondicionais:** substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
2. **Eliminar Implicações**: substitua todas as ocorrências de implicação $A \rightarrow B$ Por $\neg A \lor B$.
3. **Aplicar as Leis de De Morgan**: use as leis de De Morgan para mover as negações para dentro, aplicando:

   - $\neg (A \land B) \rightarrow \neg A \lor \neg B$
   - $\neg (A \lor B) \rightarrow \neg A \land \neg B$

4. **Eliminar Dupla Negação**: Substitua qualquer dupla negação $\neg \neg A$ Por $A$.
5. **Aplicar a Lei Distributiva**: Use a lei distributiva para expandir a fórmula, transformando-a em uma disjunção de conjunções.

#### Exemplos de Conversão para a Forma Normal Disjuntiva (Proposicional)

**Exemplo 1**: $(A \rightarrow B) \land (C \lor \neg (D \land E))$

1. Eliminar Implicações

    $$(A \rightarrow B) \land (C \lor \neg (D \land E)) \rightarrow (\neg A \lor B) \land (C \lor \neg (D \land E))$$

2. Aplicar De Morgan

    $$(\neg A \lor B) \land (C \lor \neg D \lor \neg E)$$

3. Distribuir a Disjunção

    $$(\neg A \lor B) \land C \lor (\neg A \lor B) \land \neg D \lor (\neg A \lor B) \land \neg E$$

**Exemplo 2**: $(\neg A \land (B \rightarrow C)) \lor (D \land \neg (E \rightarrow F))$

1. Eliminar Implicações

    $$(\neg A \land (\neg B \lor C)) \lor (D \land \neg (\neg E \lor F)) \rightarrow (\neg A \land (\neg B \lor C)) \lor (D \land (E \land \neg F))$$

2. Distribuir a Disjunção

    $$(\neg A \land \neg B \lor \neg A \land C) \lor (D \land E \land \neg F)$$

3. Distribuir a Disjunção Novamente

    $$\neg A \land \neg B \lor \neg A \land C \lor D \land E \land \neg F$$

**Exemplo 3**: $(p \rightarrow q) \rightarrow (r \vee s)$

1. Remover as implicações ($\rightarrow$):

    $$p \rightarrow q \equiv \neg p \vee q$$

2. Substituir a expressão original com a equivalência encontrada no passo 1:

    $$(\neg p \vee q) \rightarrow (r \vee s)$$

3. Aplicar novamente a equivalência para remover a implicação:

    $$\neg (\neg p \vee q) \vee (r \vee s)$$

4. Aplicar a lei de De Morgan para expandir a negação:

    $$(p \wedge \neg q) \vee (r \vee s)$$   

**Exemplo 4**:: $(p \rightarrow q) \rightarrow (\neg r \vee s)$

1. Primeiro, vamos eliminar as implicações, usando a equivalência $p \rightarrow q \equiv \neg p \vee q$:

   $$(p \rightarrow q) \rightarrow (\neg r \vee s)$$

   Substituindo a implicação interna, temos:

   $$(\neg p \vee q) \rightarrow (\neg r \vee s)$$

2. Agora, vamos eliminar a implicação externa, usando a mesma equivalência:

   $$\neg (\neg p \vee q) \vee (\neg r \vee s)$$

3. Em seguida, aplicamos a lei de De Morgan para expandir a negação:

   $$(p \wedge \neg q) \vee (\neg r \vee s)$$

**Exemplo 5**:  $\neg(p \land q) \rightarrow (r \leftrightarrow s)$

$$
\begin{align*}
1. & \quad \neg(p \land q) \rightarrow (r \leftrightarrow s) \\
2. & \quad \neg(p \land q) \rightarrow ((r \rightarrow s) \land (s \rightarrow r)) \, \text{ (Substituindo a equivalência por suas implicações)} \\
3. & \quad \neg(p \land q) \rightarrow ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Convertendo as implicações em disjunções)} \\
4. & \quad (\neg \neg(p \land q)) \lor ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Aplicando a equivalência } p \rightarrow q \equiv \neg p \lor q \text{)} \\
5. & \quad (p \land q) \lor ((\neg r \lor s) \land (\neg s \lor r)) \, \text{ (Aplicando a dupla negação)} \\
6. & \quad (p \land q) \lor (r \land s) \lor (\neg r \land \neg s) \, \text{ (Aplicando a distributividade para obter a FND)}
\end{align*}
$$

A Forma Normal Disjuntiva é útil porque qualquer fórmula lógica pode ser representada desta forma, e a representação é única (à exceção da ordem dos literais e cláusulas).

### Forma Normal Conjuntiva (FNC)

A Forma Normal Conjuntiva é uma representação canônica de fórmulas lógicas em que a fórmula é escrita como uma conjunção de disjunções. Em outras palavras, é uma expressão lógica na forma de uma _conjunção de disjunções_. É uma forma canônica útil para a análise e manipulação de fórmulas lógicas e é comumente usada em algoritmos de raciocínio lógico e simplificação de fórmulas.

#### Estrutura da Forma Normal Conjuntiva

Uma fórmula está na Forma Normal Conjuntiva se puder ser expressa na forma:

$$
(D_1 \lor D_2 \lor \ldots \lor D_n) \land (E_1 \lor E_2 \lor \ldots \lor E_m) \land \ldots
$$

Onde $D_1, \ldots , D_n$ e $ e_1, \ldots ,E_n $ representam átomos. Podemos dizer que a Forma Normal Conjuntiva acontece quando a Fórmula Bem Formada está na forma:

$$
\bigwedge_{i=1}^{m} \left( \bigvee_{j=1}^{n} L_{ij} \right)
$$

#### Conversão para Forma Normal Conjuntiva

Converter uma fórmula para a Forma Normal Conjuntiva, já incluindo os conceitos de Skolemização, envolve os seguintes passos:

1. **Eliminar os Bicondicionais:** substitua todas as ocorrências de $A\leftrightarrow B$ Por $A\rightarrow B \wedge B\rightarrow A$.
2. **Eliminar Implicações**: substitua todas as ocorrências de implicação $A \rightarrow B$ Por $\neg A \lor B$.
3. **Colocar a Negação no Interior dos Parenteses**: Use as leis de De Morgan para mover as negações para dentro, aplicando:

     - $\neg (\forall x A) \equiv \exists x \neg A$
     - $\neg (\exists x A) \equiv \forall x \neg A$
     - $\neg (A \land B) \rightarrow \neg A \lor \neg B$
     - $\neg (A \lor B) \rightarrow \neg A \land \neg B$

4. **Eliminar Dupla Negação**: Substitua qualquer dupla negação $\neg \neg A$ Por $A$.
5. **Skolemização**: todas as variáveis existenciais será substituída por uma Constante de Skolem, ou uma Função de Skolem das variáveis universais relacionadas.

     - $\exists x Bonito(x)$ será transformado em $Bonito(g1)$ onde $g1$ é uma Constante de Skolem.
     - $\forall x Pessoa(x) \rightarrow Coração(x) \wedge Feliz(x,y)$ se torna $\forall x Pessoa(x) \rightarrow Coração(H(x))\wedge Feliz(x,H(x))$, onde $H$ é uma função de Skolem.

6. Remova todos os Quantificadores Universais. $\forall x Pessoa(x)$ se torna $Pessoa(x)$.

7. **Aplicar a Lei Distributiva**: Use a lei distributiva para expandir a fórmula, transformando-a em uma conjunção de disjunções. Substituindo $\wedge$ por $\vee$.

#### Exemplos de Conversão para Forma Normal Conjuntiva

**Exemplo 1**: $(A \land B) \rightarrow (C \lor D)$ 

1. Eliminar Implicações**:
  
   $$\neg (A \land B) \lor (C \lor D) \rightarrow (\neg A \lor \neg B) \lor (C \lor D)$$

2. Distribuir a Disjunção:
  
    $$(\neg A \lor \neg B \lor C \lor D)$$

**Exemplo 2**: $(A \land \neg B) \lor (\neg C \land D) \rightarrow (E \lor F)$ 

1. Eliminar Implicações:
  
    $$\neg ((A \land \neg B) \lor (\neg C \land D)) \lor (E \lor F) \rightarrow \neg (A \land \neg B) \land \neg (\neg C \land D) \lor (E \lor F)$$

2. Aplicar De Morgan:
  
    $$(\neg A \lor B) \land (C \lor \neg D) \lor (E \lor F)$$

3. Distribuir a Disjunção:
  
    $$(\neg A \lor B \lor E \lor F) \land (C \lor \neg D \lor E \lor F)$$

**Exemplo 3**: $(p \wedge (q \vee r)) \vee (\neg p \wedge \neg q)$

1. Aplicar a lei distributiva para expandir a expressão:  

    $$(p \wedge q) \vee (p \wedge r) \vee (\neg p \wedge \neg q)$$

2. Transformando a expressão em uma conjunção de disjunções. Podemos fazer isso aplicando novamente a lei distributiva:

   $$(p \wedge q) \vee \neg p) \wedge ( (p \wedge q) \vee \neg q) \wedge ( (p \wedge r) \vee \neg p) \wedge ( (p \wedge r) \vee \neg q)$$

3. Finalmente a Forma Normal Conjuntiva

  $$((p \wedge q) \vee \neg p) \wedge ((p \wedge q) \vee \neg q) \wedge ((p \wedge r) \vee \neg p) \wedge (p \wedge r) \vee \neg q)$$

**Exemplo 4**: $ \neg ((p \wedge q) \vee \neg (r \wedge s)) $

1. Aplicando a Lei de De Morgan na expressão inteira:

    $$\begin{align*}
    \neg ((p \wedge q) \vee \neg (r \wedge s)) &\equiv \neg (p \wedge q) \wedge (r \wedge s) \quad \text{(Lei de De Morgan)}
    \end{align*}$$

2. aplicando a Lei de De Morgan nos termos internos:

    $$
    \begin{align*}
    \neg (p \wedge q) \wedge (r \wedge s) &\equiv (\neg p \vee \neg q) \wedge (r \wedge s) \quad \text{(Lei de De Morgan)}
    \end{align*}  
    $$

**Exemplo 5**: $\neg (((p \rightarrow q) \rightarrow p) \rightarrow p)$

1. Eliminar Implicações. Utilizando a equivalência $p \rightarrow q \equiv \neg p \lor q $:

   $$\neg(\neg(\neg p \lor q)\lor p)\lor p$$

2. Aplicar Leis de De Morgan:

   $$((p \land \neg q) \lor p) \land \neg p$$

3. Simplificamos a expressão usando propriedades como $p \lor p \equiv p$ e, em seguida, redistribuímos os termos para alcançar a forma normal conjuntiva:

   $$(p \land (\neg q \lor p)) \land \neg p$$

4. Aplicamos as propriedades comutativa e associativa para organizar os termos de uma maneira mais apresentável:

   $$(\neg q \lor p) \land (p \land \neg p)$$

5. Identificamos e simplificamos contradições na expressão usando $p \land \neg p \equiv \bot$, levando a:

   $$(\neg q \lor p) \land \bot$$

6. Por último, aplicamos a identidade com a contradição $$\bot \land p \equiv \bot$ para obter a expressão final:

   $$\bot \text{False}$$

**Exemplo 6**: $(p \rightarrow q) \leftrightarrow (p \rightarrow r)$

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

Neste passo, nosso objetivo é construir uma expressão que seja verdadeira nas linhas 5, 7 e 8 (as linhas onde o resultado é verdadeiro), e falsa em todos os outros casos. Para fazer isso, criamos uma disjunção (uma expressão _OR_) para cada linha verdadeira que reflete as condições das variáveis nesta linha, e então unimos essas disjunções com uma conjunção (uma operação _AND_) para criar a Forma Normal Disjuntiva desejada:

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

    - Linha 1: $(\neg A \lor \neg B \lor \neg C \lor A)$
    - Linha 2: $(\neg A \lor \neg B \lor C \lor A)$
    - Linha 3: $(\neg A \lor B \lor \neg C \lor A)$
    - Linha 4: $(\neg A \lor B \lor C \lor A)$
    - Linha 6: $(A \lor \neg B \lor C \lor \neg A)$

    Combinando-os com uma conjunção, temos a Forma Normal Conjuntiva:

    $$(A \lor B) \rightarrow (C \land \neg A) \equiv (\neg A \lor \neg B \lor \neg C \lor A) \\ \land (\neg A \lor \neg B \lor C \lor A) \land (\neg A \lor B \lor \neg C \lor A) \\ \land (\neg A \lor B \lor C \lor A) \land (A \lor \neg B \lor C \lor \neg A)$$

Lamentavelmente, as tabelas verdade não têm utilidade na Lógica de Primeira Ordem quando usamos predicados e quantificadores. Skolemização e Forma Normal Prenex são as rotas que precisaremos dominar para desvendar esse enigma.

### Skolemização

A Skolemização é uma técnica usada na Lógica de Primeira Ordem para eliminar quantificadores existenciais em fórmulas. Consiste em substituir as variáveis existenciais por Constantes ou Funções Skolem. Considere a fórmula a seguir com um quantificador universal e um existencial:

$$\forall x \exists y P(x,y)$$

Ao aplicar a skolemização, a variável existencial $y$ é substituída por uma Função de Skolem $f(x)$:

$$P(x,f(x))$$

Para uma fórmula com dois quantificadores universais e dois existenciais:

$$\forall x \forall z \exists y \exists w R(x,y,z,w)$$

A skolemização resultará em:

$$\forall x \forall z R(x,f(x),z,g(x,z))$$

Onde $f(x)$ e $ g(x,z)$ são Funções Skolem introduzidas para substituir as variáveis existenciais $y$ e $w $ respectivamente. A escolha entre usar uma Constante Skolem ou uma Função Skolem durante a skolemização depende do escopo dos quantificadores na fórmula original. Aqui estão as regras e passos para realizar a skolemização de maneira mais explicativa:

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

Em resumo, a skolemização simplifica fórmulas quantificadas, eliminando quantificadores existenciais e substituindo variáveis por Constantes ou Funções de Skolem, dependendo de sua relação com quantificadores universais. Isso auxilia na conversão de fórmulas quantificadas para a Forma Normal Conjuntiva e na simplificação da lógica.

### Forma Normal Prenex

A Forma Normal Prenex é uma padronização para fórmulas da lógica de primeiro grau. Nela, todos os quantificadores são deslocados para a frente da fórmula, deixando a matriz da fórmula livre de quantificadores. A Forma Normal Prenex é vantajosa por três razões fundamentais:

1. **Facilitação da Manipulação Lógica**: ao separar os quantificadores da matriz, a Forma Normal Prenex simplifica a análise e manipulação da estrutura lógica da fórmula.

2. **Preparação para Outras Formas Normais**: Serve como uma etapa intermediária valiosa na conversão para outras formas normais, como as Forma Normal Conjuntiva e Forma Normal Disjuntiva.

3. **Uso em Provas Automáticas**: é amplamente empregada em métodos de prova automática, tornando o raciocínio sobre quantificadores mais acessível.

Considere o seguinte exemplo, partindo da fórmula original: $\exists x \forall y (P(x,y) \wedge Q(y))$

Na Forma Prenex, esta fórmula será representada:

$$
\forall y \exists x (P(x,y) \wedge Q(y))
$$

#### Estrutura da Forma Normal Prenex

Uma fórmula na Forma Normal Prenex segue uma estrutura específica definida por:

$$
Q_1 x_1 \, Q_2 x_2 \, \ldots \, Q_n x_n \, M(x_1, x_2, \ldots, x_n)
$$

Nessa estrutura:

- $Q_i$ são quantificadores, podendo ser universais $\forall$ ou existenciais $\exists$.
- $x_i$ são as variáveis vinculadas pelos quantificadores.
- $M(x_1, x_2, \ldots, x_n)$ representa a matriz da fórmula, uma expressão lógica sem quantificadores.

#### Conversão para Forma Normal Prenex

Converter uma fórmula para a Forma Normal Prenex  envolve os seguintes passos:

1. **Eliminar Implicações**: substitua todas as ocorrências de implicação por disjunções e negações.

2. **Mover Negações para Dentro**: use as leis de De Morgan para mover as negações para dentro dos quantificadores e proposições.

3. **Padronizar Variáveis**: certifique-se de que as variáveis ligadas a diferentes quantificadores sejam distintas.

4. **Eliminar Quantificadores Existenciais**: substitua os quantificadores existenciais por constantes ou funções Skolem, dependendo do contexto.

5. **Mover Quantificadores para Fora**: mova todos os quantificadores para a esquerda da expressão, mantendo a ordem relativa dos quantificadores universais e existenciais.

A Forma Normal Prenex  é uma representação canônica de fórmulas da lógica de primeiro grau que separa claramente os quantificadores da matriz da fórmula. Ela é uma ferramenta valiosa na lógica e na teoria da prova, e sua compreensão é fundamental para trabalhar com lógica de primeiro grau.

#### Regras de Equivalência Prenex

A Forma Prenex de uma fórmula lógica com quantificadores permite mover todos os quantificadores para o início da fórmula. Existem algumas regras de equivalência que preservam a Forma Prenex quando aplicadas a uma fórmula:

**1. Comutatividade de quantificadores do mesmo tipo**: a ordem dos quantificadores do mesmo tipo pode ser trocada em uma fórmula na Forma Prenex. Por exemplo:

$$
\forall x \forall y \ P(x,y) \Leftrightarrow \forall y \forall x \ P(x,y)
$$

Isso ocorre porque a ordem dos quantificadores universais $\forall x$ e $\forall y$ não altera o significado lógico da fórmula. Essa propriedade é conhecida como comutatividade dos quantificadores.

**2. Associatividade de quantificadores do mesmo tipo**: quantificadores do mesmo tipo podem ser agrupados de forma associativa em uma Forma Prenex. Por exemplo:

$$
\forall x \forall y \forall z \ P(x,y,z) \Leftrightarrow \forall x (\forall y \forall z \ P(x,y,z))
$$

Novamente, o agrupamento dos quantificadores universais não muda o significado da fórmula. Essa é a propriedade associativa.

**3. Distributividade de quantificadores sobre operadores lógicos**: os quantificadores podem ser distribuídos sobre operadores lógicos como $\wedge, \vee, \rightarrow$:

$$
\forall x (P(x) \vee Q(x)) \Leftrightarrow (\forall x \ P(x)) \vee (\forall x \ Q(x))
$$

Isso permite _mover_ o quantificador para dentro do escopo do operador lógico. A equivalência se mantém pois a ordem de quantificação e operação não se altera.

### Conversão para Formas Normais Conjuntiva (FNC) e Disjuntiva (FND)

**1. Eliminar Implicações**: substitua todas as ocorrências de implicação da forma $A \rightarrow B$ Por $\neg A \lor B$.

**2. Mover a Negação para Dentro**: use as leis de De Morgan para mover a negação para dentro dos quantificadores e das proposições. Aplique as seguintes transformações:

- $\neg \forall x P(x) \rightarrow \exists x \neg P(x)$
- $\neg \exists x P(x) \rightarrow \forall x \neg P(x)$

**3. Padronizar Variáveis**: certifique-se de que as variáveis ligadas a diferentes quantificadores sejam distintas, renomeando-as se necessário.

**4. Eliminar os Quantificadores Existenciais**: substitua cada quantificador existencial $\exists x$ Por um novo termo constante ou Função Skolem, dependendo das variáveis livres em seu escopo. Para eliminar os quantificadores existenciais, é necessário introduzir novos termos: Constantes ou Funções Skolem.

1. **Se o quantificador existencial não tem quantificadores universais à sua esquerda:**
 SubstituA$\exists x P(x)$ Por $P(c)$, onde $c$ é uma nova constante.

2. **Se o quantificador existencial tem quantificadores universais à sua esquerda:**
 SubstituA$\exists x P(x)$ Por $P(f(y_1, y_2, \ldots, y_n))$, onde $f$ é uma nova função Skolem, e $y_1, y_2, \ldots, y_n$ são as variáveis universais à esquerda do quantificador existencial.

**5. Mover os Quantificadores Universais para Fora**: mova todos os quantificadores universais para fora, para a esquerda da expressão. Isso cria uma Forma Prenex da fórmula.

**6. Eliminar os Quantificadores Universais**: remova os quantificadores universais, deixando apenas a matriz da fórmula. Isso resulta em uma fórmula livre de quantificadores. Após a eliminação dos quantificadores existenciais e a movimentação de todos os quantificadores universais para fora (Forma Prenex), a eliminação dos quantificadores universais é simples:

1. **Remova os quantificadores universais da fórmula:**
 Se você tem uma fórmula da forma $\forall x P(x)$, simplesmente remova o quantificador $\forall x$, deixando apenas a matriz da fórmula $P(x)$.

2. **Trate as variáveis como variáveis livres:**
 As variáveis que eram ligadas pelo quantificador universal agora são tratadas como variáveis livres na matriz da fórmula.

**7. Conversão para FNC**:

1. Use as leis distributivas para mover as conjunções para dentro e as disjunções para fora.
2. Certifique-se de que a fórmula esteja na forma de uma conjunção de disjunções (cláusulas).

**8. Conversão para FND**:

1. Use as leis distributivas para mover as disjunções para dentro e as conjunções para fora.
2. Certifique-se de que a fórmula esteja na forma de uma disjunção de conjunções.

#### Exemplos Interessantes da Forma Prenex

**Exemplo 1**: duas fórmulas logicamente equivalentes, uma na Forma Prenex e outra não considere a fórmula original:

$$
\forall x \exists y (P(x) \rightarrow Q(y))  
$$

Se convertida para a Forma Prenex teremos:

$$
\exists y \forall x (P(x) \rightarrow Q(y))
$$

Cuja a equivalência pode ser provada por meio do seguinte raciocínio: sejA$I$ uma interpretação booleana das variáveis $P$ e $Q$. SuponhA$I$ satisfaz $\forall x \exists y (P(x) \rightarrow Q(y))$. Logo, para todo $x$ no domínio, existe um $y$ tal que: se $P(x)$ é verdadeiro, então $Q(y)$ também é verdadeiro. Isso é equivalente a dizer: existe um $y$, tal que para todo $x$, se $P(x)$ é verdadeiro, $Q(y)$ também é verdadeiro. Ou seja, $I$ também satisfaz: $\exists y \forall x (P(x) \rightarrow Q(y))$. Por um raciocínio simétrico, o oposto também é verdadeiro. Portanto, as fórmulas são logicamente equivalentes.

**Exemplo 2**: Fórmula sem Forma Prenex:

$$
\forall x (P(x) \rightarrow \exists y Q(x,y))
$$

Não pode ser convertida à Forma Prenex pois o quantificador $\exists y$ está dentro do escopo de $\rightarrow$.

#### Observações Importantes

A conversão para Forma Normal Conjuntiva é útil para métodos de prova. A conversão para Forma Normal Disjuntiva é menos comum, mas pode ser útil em alguns contextos de análise lógica. **CUIDADO: a eliminação dos quantificadores pode alterar a interpretação da fórmula em alguns modelos, mas é útil porque preserva a satisfatibilidade**.

#### Exemplos de conversão em formas normais, conjuntiva e disjuntiva

a) Todos os alunos estudam ou alguns professores ensinam matemática

**Lógica de Primeiro Grau**:

$$\forall x(\text{Aluno}(x) \rightarrow \text{Estuda}(x)) \lor \exists y(\text{Professor}(y) \land \text{EnsinaMatemática}(y))$$

**Forma Normal Conjuntiva (FNC)**:

1.Convertendo a implicação:

$$\neg \text{Aluno}(x) \lor \text{Estuda}(x)$$

2.Adicionando a disjunção existencial:

$$(\neg \text{Aluno}(x) \lor \text{Estuda}(x)) \land (\text{Professor}(y) \land \text{EnsinaMatemática}(y))$$

**Forma Normal Disjuntiva (FND)**:

1.Negando o consequente do implicador:

$$\text{Aluno}(x) \land \neg \text{Estuda}(x)$$

2.Adicionando a conjunção existencial negada:

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

1. Adicionando a disjunção existencial:

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

## Cláusula de Horn

A **Cláusula de Horn** foi nomeada em homenagem ao matemático e lógico americano [Alfred Horn](https://en.wikipedia.org/wiki/Alfred_Horn), que a introduziu em [um artigo publicado em 1951](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/on-sentences-which-are-true-of-direct-unions-of-algebras1/DF348CB269B06D6702DA3AE4DCF38C39). O contexto histórico e a motivação para a introdução da Cláusula de Horn são profundamente enraizados na solução do Problema da Decidibilidade. Na primeira metade do século XX, a lógica matemática estava focada na questão da decidibilidade: determinar se uma afirmação lógica é verdadeira ou falsa de maneira algorítmica.

Não demorou muito para os matemáticos perceberem que a Lógica de Primeira Ordem é poderosa, mas pode ser ineficientes para resolver os problemas relacionados ao Problema da Decidibilidade. A busca por formas mais eficientes de resolução levou ao estudo de subconjuntos restritos da Lógica de Primeira Ordem, onde a decidibilidade poderia ser alcançada de maneira mais eficiente. Aqui, eficiência significa o menor custo computacional, no menor tempo.

Alfred Horn identificou um desses subconjuntos em seu artigo de 1951, introduzindo o que agora é conhecido como **Cláusula de Horn**. Ele mostrou que esse subconjunto particular tem propriedades interessantes que permitem a resolução em tempo polinomial, tornando-o atraente para aplicações práticas.

Se prepare vamos ver porque $P \lor \neg Q \lor \neg R $ é uma Cláusula de Horn e $P \lor Q \lor \neg R$ não é.

### Definição da Cláusula de Horn

A **Cláusula de Horn** é uma forma especial de cláusula na Lógica de Primeira Ordem. Ela é caracterizada por **ter no máximo um literal positivo**.

#### Forma Geral

Uma Cláusula de Horn pode ser representada pela fórmula dada por:

$$
\bigwedge_{i=1}^{n} \neg P_i \rightarrow P
$$

onde:

-$P_i$ são literais positivos. Um literal positivo é uma proposição atômica. Pode haver no máximo um literal positivo.
-$P$ é um literal positivo ou uma contradição (falso).
-$n$ é o número de literais negativos na cláusula. Os literais negativos são representados por $\neg P_i$. Ou seja, os literais negativos são as negações de proposições atômicas. Podem haver zero ou mais literais negativos.

#### Tipos de Cláusulas de Horn

A Cláusula de Horn pode ser classificada em três tipos principais:

1. **Nula**: uma cláusula vazia;
2. **Fatos**: não há literais negativos, apenas um literal positivo. Exemplo:$P$.
3. **Regras**: um ou mais literais negativos e exatamente um literal positivo. Eventualmente chamamos as Regras de Cláusulas Definidas; Exemplo: $\neg P \land \neg Q \rightarrow R$.
4. **Metas ou Consultas**: um ou mais literais negativos e nenhum literal positivo. As cláusulas de meta contém apenas literais negativos. Exemplo: $\neg P \land \neg Q$.

Para entender melhor, imagine que estamos construindo um cenário mental fundamentado na lógica para construir o entendimento de um problema, uma espécie de paisagem mental onde as coisas fazem sentido. Nesse cenário, as Cláusulas de Horn serão os tijolos fundamentais que usaremos para construir estruturas lógicas.

**1. Fatos**: os fatos são como pedras fundamentais desse cenário. Eles são afirmações simples e diretas que dizem como as coisas são. Considere, por exemplo: _O céu é azul_, $P$ e _A grama é verde_$Q$. Essas são verdades que não precisam de justificativa. Elas simplesmente são. os Fatos são axiomas.

**2. Regras**: as regras são um pouco mais intrigantes. Elas são como as regras de um jogo que definem como as coisas se relacionam umas com as outras. _Se não chover, a grama não ficará molhada._ Essa é uma regra. Ela nos diz o que esperar se certas condições forem atendidas. As regras são como os conectores em nosso mundo lógico, ligando fatos e permitindo que façamos inferências. Elas são o motor que nos permite raciocinar e descobrir novas verdades a partir das que já conhecemos. Por exemplo:

- $\neg P \land \neg Q \rightarrow R$: _Se não chover, $P$ e não ventar, $Q$, então faremos um piquenique, $R$_.
- $\neg A \land \neg B \land \neg C \rightarrow D$: _Se $A$, $B$ e $C$ forem falsos, então $D$ é verdadeiro_.

**3. Metas ou Consultas**: finalmente, temos as metas ou consultas. Essas são as perguntas que fazemos ao nosso mundo lógico. _Está chovendo?_, _A grama está molhada?_ São os caminhos que usaremos para explorar o cenário criado, olhando ao redor e tentando entender o que está acontecendo. As consultas são a maneira de interagir com nosso mundo lógico, usando os fatos e regras que estabelecemos para encontrar respostas e alcançar objetivos. Por exemplo:

- $\neg P \land \neg Q$: _É verdade que hoje não está chovendo e não está ventando?_
- $\neg X \land \neg Y \land \neg Z$: _$x$, $Y$ e $Z $ são falsos?_

Podemos tentar avaliar alguns exemplos de uso de Fatos, Regras e Consultas:

#### Exemplo 1: Sistema de Recomendação de Roupas

Imagine que estamos construindo um sistema lógico para recomendar o tipo de roupa que uma pessoa deve vestir com base no clima. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**1. Fatos**: primeiro, estabelecemos os fatos, as verdades básicas do cenário que descreve nosso problema. Neste exemplo, os fatos poderiam ser informações sobre o clima atual.

- **Fato 1**: Está ensolarado. (Representado como $s $)
- **Fato 2**: A temperatura está acima de 20°C. (Representado como $T$)

Você pode criar todos os fatos necessários a descrição do seu problema.

**2. Regras**: em seguida, definimos as regras que descrevem como as coisas se relacionam. Essas regras nos dizem o tipo de roupa apropriada com base no clima.

- **Regra 1**: Se está ensolarado e a temperatura está acima de 20°C, use óculos de sol. ($\neg S \land \neg T \rightarrow O $)
- **Regra 2**: Se está ensolarado, use chapéu. ($\neg S \rightarrow C$)
- **Regra 3**: Se a temperatura está acima de 20°C, use camiseta. ($\neg T \rightarrow A$)

Você pode criar todas as regras que achar importante para definir o comportamento no cenário que descreve o problema.

**3. Consultas (Metas)**: agora, podemos fazer consultas ao nosso sistema para obter recomendações de roupas.

- **Consulta 1**: Está ensolarado e a temperatura está acima de 20°C. O que devo vestir? ($\neg S \land \neg T$)

As consultas representam todas as consultas que podem ser feitas neste cenário. Crie quantas consultas achar necessário.

**4. Resolução**: usando os fatos e regras, podemos resolver a consulta:

1. Está ensolarado e a temperatura está acima de 20°C (_Fato_).
2. Portanto, use óculos de sol (_Regra 1_).
3. Portanto, use chapéu (_Regra 2_).
4. Portanto, use camiseta (_Regra 3_).

Neste exemplo, as Cláusulas de Horn nos permitiram representar o conhecimento sobre o clima e as regras para escolher roupas. Os fatos forneceram a base de conhecimento, as regras permitiram inferências lógicas, e a consulta nos permitiu explorar o sistema para obter recomendações práticas.

#### Exemplo 2: Sistema de Diagnóstico Médico

Imagine que estamos construindo um sistema lógico para diagnosticar doenças com base em sintomas, histórico médico e outros fatores relevantes. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**1. Fatos**: começamos estabelecemos os fatos, que são as informações conhecidas sobre o paciente.

- **Fato 1**: O paciente tem febre. (Representado como $F$)
- **Fato 2**: O paciente tem tosse. (Representado como $T$)
- **Fato 3**: O paciente viajou recentemente para uma área endêmica. (Representado como $V$)
- **Fato 4**: O paciente foi vacinado contra a gripe. (Representado como $ g$)

**2. Regras**: em seguida, definimos as regras que descrevem as relações entre sintomas, histórico médico e possíveis doenças.

- **Regra 1**: Se o paciente tem febre e tosse, mas foi vacinado contra a gripe, então pode ter resfriado comum. ($\neg F \land \neg T \land G \rightarrow R$)
- **Regra 2**: Se o paciente tem febre, tosse e viajou para uma área endêmica, então pode ter malária. ($\neg F \land \neg T \land \neg V \rightarrow M $)
- **Regra 3**: Se o paciente tem febre e tosse, mas não foi vacinado contra a gripe, então pode ter gripe. ($\neg F \land \neg T \land \neg G \rightarrow I $)

**3. Consultas**: agora, podemos fazer consultas ao nosso sistema para obter diagnósticos possíveis.

- **Consulta 1**: O paciente tem febre, tosse, viajou para uma área endêmica e foi vacinado contra a gripe. Qual é o diagnóstico? ($\neg F \land \neg T \land \neg V \land G$)

**4. Resolução**: usando os fatos e regras, podemos resolver a consulta:

1. O paciente tem febre, tosse, viajou para uma área endêmica e foi vacinado contra a gripe (_Fatos_).
2. Portanto, o paciente pode ter resfriado comum (_Regra 1_).
3. Portanto, o paciente pode ter malária (_Regra 2_).

**5. Conclusão**: este exemplo ilustra como as Cláusulas de Horn podem ser usadas em um contexto mais complexo, como um sistema de diagnóstico médico. A mesma abordagem pode ser aplicada a outros domínios, como diagnósticos de falhas em máquinas, sistemas legais, planejamento financeiro e muito mais.

### Quantificadores em Cláusulas de Horn

Os quantificadores podem ser incluídos nas Cláusulas de Horn. Contudo, é importante notar que a forma padrão de Cláusulas de Horn em programação lógica geralmente lida com quantificação de maneira implícita. A quantificação universal é comum e é geralmente assumida em regras, enquanto a quantificação existencial é muitas vezes tratada através de fatos específicos ou construção de termos.

Precisamos tomar cuidado porque a inclusão explícita de quantificadores pode levar a uma Lógica de Primeira Ordem mais rica, permitindo expressões mais complexas e poderosas. No entanto, isso também pode aumentar a complexidade do raciocínio e da resolução.

#### Usando o Quantificador Universal em Cláusulas de Horn

O quantificador universal (representado por $\forall $) afirma que uma propriedade é verdadeira para todos os membros de um domínio. Em Cláusulas de Horn, isso é geralmente representado implicitamente através de regras gerais que se aplicam a todos os membros de um conjunto. Por exemplo, considere a regra: _Todos os pássaros podem voar_. Em uma Cláusula de Horn, isso pode ser representado como:

- **Regra**: Se é um pássaro, então pode voar. ( $\forall x, \neg \text{Pássaro}(x) \rightarrow \text{Voa}(x)$)

#### Usando o Quantificador Existencial em Cláusulas de Horn

O quantificador existencial (representado por $\exists $ ) afirma que existe pelo menos um membro de um Universo de Discurso, ou domínio, para o qual uma propriedade é verdadeira. Em Cláusulas de Horn, isso pode ser representado através de fatos específicos ou regras que afirmam a existência de algo. Por exemplo, considere a afirmação: _Existe um pássaro que não pode voar_. Em uma Cláusula de Horn, isso pode ser representado como:

- **Fato**: Existe um pássaro que não pode voar. ( $\exists x, \text{Pássaro}(x) \land \neg \text{Voar}(x)$)

### Conversão de Fórmulas

Seja uma fórmula bem formada arbitrária da Lógica Proposicional. Alguns passos podem ser aplicados para obter uma cláusula de Horn equivalente:

1. Converter a fórmula para Forma Normal Conjuntiva (FNC), obtendo uma conjunção de disjunções
2. Aplicar as seguintes técnicas em cada disjunção:

   - Inverter a polaridade de literais positivos extras;
   - Adicionar literais negativos que preservem a satisfatibilidade;
   - Dividir em cláusulas menores se necessário.

3. Simplificar a fórmula final obtida.

#### Exemplo: dada a fórmula

$$(P \land Q) \lor (P \land R)$$

Passos:

1. Converter para FNC: $(P \lor Q) \land (P \lor R)$
2. Inverter P em uma das disjunções: $(P \lor Q) \land (\neg P \lor R)$
3. Adicionar literal negativo: $(P \lor Q \lor \neg S) \land (\neg P \lor R \lor \neg T)$
4. Simplificar: $\neg S \lor P \land \neg T \lor r $

A sequência destes passos permite encontrar uma conjunção de cláusulas de Horn equivalente à fórmula original.

#### Transformação de Forma Normal Conjuntiva (FNC) para Cláusulas de Horn

A Forma Normal Conjuntiva é uma conjunção de disjunções de literais. Uma Cláusula de Horn é um tipo especial de cláusula que contém no máximo um literal positivo. Considere que o objetivo das Cláusulas de Horn é criar um conjunto de Fórmulas Bem Formadas, divididas em Fatos, Regras e Consultas para permitir a resolução de problemas então, a transformação de uma FNC para Cláusulas de Horn pode incorrer em alguns problemas:

- **Perda de Informação**: Nem todas as cláusulas em FNC podem ser transformadas em Cláusulas de Horn. Para minimizar este risco atente para as regras de equivalência que vimos anteriormente.
- **Complexidade**: A transformação pode ser complexa e requer uma análise cuidadosa da lógica e do contexto.

#### Etapas de Transformação

1. **Converter para FNC**: Se a fórmula ainda não estiver em Forma Normal Conjuntiva, converta-a para Forma Normal Conjuntiva usando as técnicas descritas anteriormente.
2. **Identificar Cláusulas de Horn**: Verifique cada cláusula na Forma Normal Conjuntiva. Se uma cláusula contém no máximo um literal positivo, ela já é uma Cláusula de Horn.
3. **Transformar Cláusulas Não-Horn**: Se uma cláusula contém mais de um literal positivo, ela não pode ser diretamente transformada em uma Cláusula de Horn sem perder informações.

**Exemplo**: vamos considerar a seguinte fórmula bem formada:

$$(A \rightarrow B) \land (B \lor C)$$

1. **Converter para FNC**:

- Elimine a implicação: $(\neg A \lor B) \land (B \lor C)$
- A fórmula já está em Forma Normal Conjuntiva.

2. **Identificar Cláusulas de Horn**:

- Ambas as cláusulas são Cláusulas de Horn, pois cada uma contém apenas um literal positivo.

3. **Resultado**:

- A fórmula em Cláusulas de Horn é: $(\neg A \lor B) \land (B \lor C)$

#### Problemas interessantes resolvidos com a Cláusula de Horn

**Problema 1 - O Mentiroso e o Verdadeiro:**: Você encontra dois habitantes: $A$ e $B$. Você sabe que um sempre diz a verdade e o outro sempre mente, mas você não sabe quem é quem. Você consulta a $A$, _Você é o verdadeiro?_ A responde, mas você não consegue ouvir a resposta dele. $B$ então te diz, _A disse que ele é o mentiroso_.

**Fatos**:

$mentiroso(A)$
$verdadeiro(B)$

**Regra**:

$$
\forall x \forall y (mentiroso(x) \wedge consulta(y, \text{Você é o verdadeiro?}) → Responde (x, \text{Sou o mentiroso}))
$$

**Consulta**:

$$ responde (A, \text{Sou o mentiroso})?$$

**Problema 2 - As Três Lâmpadas:** existem três lâmpadas incandescentes em uma sala, e existem três interruptores fora da sala. Você pode manipular os interruptores o quanto quiser, mas só pode entrar na sala uma vez. Como você pode determinar qual interruptor opera qual lâmpada?

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

**Problema 3 - O Agricultor, a Raposa, o Ganso e o Grão:** um agricultor quer atravessar um rio e levar consigo uma raposa, um ganso e um saco de grãos. O barco do agricultor só lhe permite levar um item além dele mesmo. Se a raposa e o ganso estiverem sozinhos, a raposa comerá o ganso. Se o ganso e o grão estiverem sozinhos, o ganso comerá o grão. Como o agricultor pode levar todas as suas posses para o outro lado do rio?

**Fatos**:
´´´ prolog
raposa(r)
ganso(g)
grao(gr)

**Regras**:

$$\forall x \forall y (Raposa(x) \wedge Ganso(y) \wedge Sozinhos(x, y) \rightarrow Come (x, y))$$

$$\forall x \forall y (Ganso(x) \wedge Grão(y) \wedge Sozinhos(x, y) \rightarrow Come (x, y))$$

**Consulta**:

$$¬Come (r, g)?$$
$$¬Come (g, gr)?$$

**Problema 4 - A Ponte e a Tocha:** quatro pessoas chegam a um rio à noite. Há uma ponte estreita, mas ela só pode conter duas pessoas de cada vez. Eles têm uma tocha e, por ser noite, a tocha tem que ser usada ao atravessar a ponte. A pessoa A pode atravessar a ponte em um minuto, B em dois minutos, C em cinco minutos e D em oito minutos. Quando duas pessoas atravessam a ponte juntas, elas devem se mover no ritmo da pessoa mais lenta. Qual é a maneira mais rápida para todos eles atravessarem a ponte?

**Fatos (tempos)**:

$tempo(a, 1)$
$tempo(b, 2)$
$tempo(c, 5)$
$tempo(d, 8)$

**Regra**:

$$\forall x \forall y (AtravessaCom(x, y) \rightarrow TempoTotal(Máximo(Tempo(x), Tempo(y))))$$

**Consulta**:

$$tempoTotal(15)?$$

**Problema 5 - O Problema de Monty Hall:** em um programa de game show, os concorrentes tentam adivinhar qual das três portas contém um prêmio valioso. Depois que um concorrente escolhe uma porta, o apresentador, que sabe o que está por trás de cada porta, abre uma das portas não escolhidas para revelar uma cabra (representando nenhum prêmio). O apresentador então pergunta ao concorrente se ele quer mudar sua escolha para a outra porta não aberta ou ficar com sua escolha inicial. O que o concorrente deve fazer para maximizar suas chances de ganhar o prêmio?

**Fatos**:

$ Porta(d_1)$
$ Porta(d_2)$
$ Porta(d_3)$

**Regras**:

$$\forall x Prêmio(x) \rightarrow Porta(x)$$

$$\forall x \forall y (Porta(x) \wedge Porta(y) \wedge x \neq y \rightarrow \neg Prêmio(x) \vee \neg Prêmio(y))$$

**Pergunta**:

$$\exists x (Porta(x) \wedge \neg Revelada(x) \wedge x \neq PortaEscolhida \rightarrow Prêmio(x))?$$

### Cláusulas de Horn e o Prolog

O Prolog é uma linguagem de programação lógica que utiliza Cláusulas de Horn para representar e manipular conhecimento. A sintaxe e a semântica do Prolog são diretamente mapeadas para Cláusulas de Horn:

- **Fatos**: Em Prolog, fatos são representados como cláusulas sem antecedentes. Por exemplo, o fato _John é humano_ pode ser representado como _humano(john)_.
- **Regras**: As regras em Prolog são representadas como implicações, onde os antecedentes são literais negativos e o consequente é o literal positivo. Por exemplo, a regra _Se X é humano, então X é mortal_ pode ser representada como _mortal(X) :- humano(X)_.
- **Consultas**: As consultas em Prolog são feitas ao sistema para inferir informações com base nos fatos e regras definidos. Por exemplo, a consulta "Quem é mortal?" pode ser representada como _?- mortal(X)_.

O Prolog utiliza um mecanismo de resolução baseado em Cláusulas de Horn para responder a consultas. Ele aplica uma técnica de busca em profundidade para encontrar uma substituição de variáveis que satisfaça a consulta.

#### Exemplo 1: O mais simples possível

**Fatos:**

``` prolog
homem(joão).
mulher(maria).
```

Os fatos indicam que "João é homem" e "maria é mulher".

**Regra:**

``` prolog
mortal(X) :- homem(X).
```

A regra estabelece que "Se X é homem, então X é mortal". O símbolo :- representa implicação.

**Consulta:**  

``` prolog
mortal(joão).
 ```

A consulta verifica se "João é mortal", aplicando a regra definida anteriormente. O Prolog responderá _True_ (verdadeiro ou $\top$) pois a regra se aplica dado o fato de que João é homem.

#### Exemplo 2: Sistema de Recomendação de Roupas em Prolog

Imagine que estamos construindo um sistema lógico simples em Prolog para recomendar o tipo de roupa que uma pessoa deve vestir com base no clima. Vamos usar Cláusulas de Horn para representar o conhecimento e a lógica do sistema.

**Fatos**: primeiro, estabelecemos os fatos, que são as verdades básicas sobre o mundo. Neste caso, os fatos podem ser informações sobre o clima atual.

- **Fato 1**: está ensolarado.

``` prolog
 ensolarado.
 ```

- **Fato 2**: a temperatura está acima de 20°C.

``` prolog
 temperatura_acima_de_20.
```

**Regras**: em seguida, definimos as regras que descrevem como as coisas se relacionam. Essas regras nos dizem o tipo de roupa apropriada com base no clima.

- **Regra 1**: se está ensolarado e a temperatura está acima de 20°C, use óculos de sol.

``` prolog
 óculos_de_sol :- ensolarado, temperatura_acima_de_20.
```

- **Regra 2**: se está ensolarado, use chapéu.

``` prolog
 chapéu :- ensolarado.
```

- **Regra 3**: se a temperatura está acima de 20°C, use camiseta.

``` prolog
 camiseta :- temperatura_acima_de_20.
```

**Consultas**: agora, podemos fazer consultas ao nosso sistema para obter recomendações de roupas.

- **Consulta 1**: está ensolarado e a temperatura está acima de 20°C. O que devo vestir?

``` prolog
 ?- óculos_de_sol, chapéu, camiseta.
```

### Um Problema Interessante

[Niklaus Wirth](https://en.wikipedia.org/wiki/Niklaus_Wirth) em seu livro _Algorithms + Data Structures = Programs_ [^1] cita um problema interessante que foi publicado em um jornal de _Zürich_ em 1922, que cito em tradução livre a seguir: 

>Casei com uma viúva (vamos chamá-la de W) que tem uma filha adulta (chame-a de D). Meu pai (F), que nos visitava com bastante frequência, apaixonou-se pela minha enteada e casou-se com ela. Por isso, meu pai se tornou meu genro e minha enteada se tornou minha madrasta. Alguns meses depois, minha esposa deu à luz um filho (S1), que se tornou cunhado do meu pai, e meu tio. A esposa do meu pai, ou seja, minha enteada, também teve um filho (S2). Em outras palavras, para todos os efeitos, eu sou meu próprio avo.

Usando este relato como base podemos criar uma base de conhecimento em Prolog, incluir algumas regras, e finalmente verificar se é verdade que o _narrador_ é o seu próprio avô. 

``` prolog
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

---
[^1]:WIRTH, Niklaus. Algorithms and Data Structures. [S.l.]: [s.n.], [s.d.]. Disponível em: <https://cdn.preterhuman.net/texts/math/Data_Structure_And_Algorithms/Algorithms%20and%20Data%20Structures%20-%20Niklaus%20Wirth.pdf>. Acesso em: [data de acesso].
