---
layout: post
title: "MDP: Casos Reais"
author: frank
categories: []
tags: []
image: ""
featured: false
rating: 0
description: ""
date: null
preview: ""
keywords: ""
toc: false
published: false
beforetoc: ""
lastmod: 2025-02-15T09:17:24.789Z
---

A curiosa leitora deve estar decepcionada. Quando comecei esta jornada eu prometi que estudaríamos Reinforcement Learning e, até agora, nada. Só o básico fundamental. Peço desculpas.

Eu estou me divertindo enquanto escrevo e ainda descubro dúvidas que nem sabia que tinha. Analisando os artigos anteriores: [1](https://frankalcantara.com/reinforcement-learning-hist%C3%B3ria/), [2](https://frankalcantara.com/entendendo-mdp/). [3](https://frankalcantara.com/um-mundo-em-uma-grade/) e [4](https://frankalcantara.com/resolvendo-o-grid-world/) percebi que o texto carece de aplicações práticas.

Pesquisando, encontrei alguns casos interessantes de aplicações de **MDP** em situações da vida real que foram acadêmica e cientificamente descritos em artigos publicados. Escolhi aqueles que continham mais dados e fossem compatíveis com os meus interesses pessoais. A compadecida leitora há de perdoar este pobre autor egoísta.

## Manutenção Ótima de Turbinas Eólicas

Na indústria, a Manutenção Preditiva Industrial busca otimizar a manutenção de equipamentos, antecipando falhas e minimizando o tempo de inatividade. Em um cenário de **MDP**, os estados representam os diferentes níveis de desgaste de um equipamento. As ações incluem realizar manutenção, continuar a operação ou substituir uma peça. As recompensas são medidas em termos do balanço entre os custos de manutenção e o tempo de operação contínua.

Este é um tema que tem impacto na eficiência do processo industrial e, consequentemente, na competitividade. Um caso real de matemática, ciência se usar o termo de forma geral, aplicada para impactar no mercado. O Artigo mais destacado neste tema destaca um caso emblemático envolvendo  a **General Electric (GE)**, que utiliza **MDPs** para otimizar a manutenção de turbinas eólicas, considerando fatores como desgaste, condições climáticas e demanda de energia[^1].

[^1]:WU, Yan-Ru; ZHAO, Hong-Shan. **Optimization Maintenance of Wind Turbines Using Markov Decision Processes**. Disponível em: https://www.researchgate.net/publication/241177157_Optimization_maintenance_of_wind_turbines_using_Markov_decision_processes. Acesso em: 12 fev. 2025.

Este sistema permite à **GE** realizar manutenções de forma mais eficiente, prolongando a vida útil das turbinas e reduzindo custos operacionais.

Não vou discutir o artigo. Vou transformar os dados do artigo em um exercícios de MDP e resolver, depois comparamos os nossos dados com os dados dos pesquisadores.

> A sagaz leitora pode anotar essa ideia: a melhor forma, tanto para sociedade, quanto para você, de entender qualquer artigo científico é refazer a pesquisa ou, no mínimo, refazer todos os cálculos.

### Exercício: Manutenção Ótima de Turbinas Eólicas via MDP

Uma empresa de energia eólica busca otimizar sua política de manutenção de [caixas de engrenagem](https://www.energy.gov/eere/wind/how-wind-turbine-works-text-version) (*gearboxes*) de turbinas eólicas usando **Processos de Decisão de Markov (MDP)**. Similar ao **Grid World**, onde um agente navega em uma grade buscando maximizar recompensas, *aqui temos um sistema que navega entre estados de deterioração buscando minimizar custos*.

#### Estados ($S$)

O sistema possui $7$ estados, mais simples que os $12$ estados que usamos no exemplo do  **Grid World**:

1. Estado perfeito, nenhum desgaste. Análogo ao estado inicial do Grid World;
2. Desgaste leve;
3. Desgaste moderado;
4. Desgaste avançado;
5. Desgaste severo;
6. Falha por deterioração. Um estado de recompensa negativa, análogo ao estado terminal negativo que usamos no **Grid Word**;
7. Falha aleatória, segundo a Distribuição de Poisson. Neste caso, temos dois estados com recompensas negativas;

>No contexto de manutenção de turbinas eólicas, além da deterioração natural do equipamento, o modelo considera falhas aleatórias que podem ocorrer independentemente do estado de desgaste.
>
>O artigo especifica uma taxa de falha aleatória $\lambda_0 = 0.0027$, equivalente a aproximadamente uma falha por ano.
>
>A **distribuição de Poisson**[^2] é uma distribuição de probabilidade discreta que modela o número de eventos que ocorrem em um intervalo fixo de tempo ou espaço, sob as seguintes condições:
>
>1. **Eventos independentes**: Cada evento ocorre independentemente dos outros;
>2. **Taxa constante**: A taxa média de ocorrência dos eventos ($\lambda$) é constante ao longo do tempo ou espaço;
>3. **Número infinito de tentativas**: Em qualquer intervalo, há um número potencialmente infinito de oportunidades para que um evento ocorra.
>
>>A probabilidade de observar $k$ eventos em um intervalo é dada por:
>
>$$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
>
>Nesta equação temos:
>
>- $k$ é o número de eventos observados.
>- $\lambda$ é a taxa média de ocorrência dos eventos.
>- $e$ é a base do logaritmo natural.
>- $k!$ é o fatorial de $k$.
>
>Para nosso MDP, isto significa que em qualquer estado não-terminal $s$, existe uma probabilidade $\lambda_0$ de transição para o estado de falha aleatória (estado 7):
>
>$$ P(s' = 7|s, a) = \lambda_0 = 0.0027 $$
>
Esta probabilidade será incorporada na matriz de transição $P(s'|s,a)$ como uma transição possível de cada estado não-terminal para o estado $7$, considerando três possibilidades em cada estado:
>
>- Deterioração natural (com taxa $\lambda$)
>- Falha aleatória (com taxa $\lambda_0$)
>- Permanência no estado atual (com taxa $1 - \lambda - \lambda_0$)
>
> A atenta leitora deve notar que esta é uma simplificação do modelo real. Na prática, equipamentos muito desgastados podem ter taxas de falha aleatória maiores. Porém, para manter a propriedade de Markov, assumimos $\lambda_0$ constante independente do estado.

[^2]: DEVORE, Jay L. **Probability and Statistics for Engineering and the Sciences**. 9. ed. Boston: Cengage Learning, 2016.

#### Ações ($A$)

Em cada estado não-terminal, as seguintes ações estão disponíveis. Novamente, podemos fazer uma analogia com o **Grid World**. Quando estudamos **MDP** com o **Grid Word** usamos uma ação para cada direção possível:

- $NA$: Nenhuma ação;
- $MM$: Manutenção menor, retorna ao estado anterior;
- $PM$: Manutenção preventiva; retorna ao estado perfeito;
- $CM$: Manutenção corretiva; após falha, retorna ao estado perfeito.

#### Dinâmica do Sistema

#### Parâmetros Operacionais

- Turbina: $5MW$, fator de capacidade $0.4$, implicando em $2MW$ efetivos;
- Custo da energia: $€0.5/kWh$;
- Horizonte de tempo: $100.000$ horas;
- Tempo de reparo após falha: $15$ dias;

#### Taxas de Transição

- Deterioração entre estados: $\lambda = 0.0012$ (1/833 dias);
- Falha aleatória: $\lambda_0 = 0.0027$ (1/ano);
- Fator de desconto: $\gamma = 0.9$ (análogo ao **Grid World**);

#### Probabilidades de Manutenção

Para $MM$ em estados $2-5$:

| Estado       | Prob. Transição |
|-------------|-----------------|
| Anterior    | $0.7$            |
| $-2$ estados  | $0.2$            |
| $-3$ estados  | $0.1$            |

Esse $-2$ representa dois estados antes do atual. O mesmo vale para o $-3$

Para $PM$ em estados $2-5$:

| Estado    | Prob. Transição |
|-----------|-----------------|
| Perfeito  | $0.9$            |
| Estado 2  | $0.09$           |
| Estado 3  | $0.01$           |

#### Sistema de Recompensas (Custos)

Análogo ao sistema de recompensas do **Grid World** acrescido da informação de custos:

- Inspeção: $€200$, análogo ao custo por passo do Grid World;
- Manutenção menor ($MM$): $€3.000$;
- Manutenção preventiva ($PM$): $€7.500$;
- Manutenção corretiva (CM): $€150.000$;
- Perda por falha: $€180.000$, análogo à penalidade do estado terminal negativo;
- Estado não detectado: $€1.000/hora$.

### Intervalos de Inspeção

- Mínimo: $500$ horas;
- Máximo: $20.000$ horas;
- Discretizado em $20$ intervalos iguais.

A perspicaz leitora deve ter ficado curiosa com estes intervalos de inspeção.

Quando estudamos o **Grid World**, ressaltei que a cada passo o agente observa perfeitamente seu estado atual. O agente sempre sabe em qual célula está. Isso significa que o **Grid World** é um ambiente com observabilidade total.

Nas condições apresentadas no artigo[^1] não existe a observabilidade total. O intervalo de inspeção representa o tempo entre observações do estado real do sistema. Só é possível saber o estado da turbina quando são realizadas inspeções. O que torna este problema mais complexo que o **Grid Word**. Este é o custo que temos que pagar para bisbilhotar o mundo real.

Como, entre inspeções, o sistema evolui sem que saibamos seu estado exato, o custo/benefício de fazer inspeções mais ou menos frequentes precisa ser considerado.

A amável leitora deve considerar que, se não conhecemos a forma como o sistema evolui, uma falha pode ocorrer entre inspeções sem ser detectada.

No **Grid World** isso seria como se o agente só pudesse observar sua posição a cada $N$ passos aleatórios e o o agente pudesse cair em estados terminais negativos sem saber.

Por isso os autores do artigo precisaram otimizar também o intervalo entre inspeções.

### Questões

1. **Modelagem do Sistema**
   a) Formule a matriz de transição $P(s'|s,a)$ para cada ação
   b) Defina a função de recompensa $R(s,a,s')$
   c) Identifique os estados terminais e suas características

2. **Equações de Bellman**
   a) Escreva a equação para a função valor-estado $V^*(s)$
   b) Escreva a equação para a função valor-ação $Q^*(s,a)$
   c) Como estas equações se comparam com as do Grid World?

3. **Solução por Iteração de Valor**
   a) Inicialize os valores de estado
   b) Aplique o processo iterativo usando a equação de Bellman
   c) Mostre o cálculo detalhado para a primeira iteração no estado perfeito

4. **Solução por Iteração de Política**
   a) Defina uma política inicial
   b) Realize a avaliação de política
   c) Execute a melhoria de política
   d) Compare a convergência com o caso do Grid World

5. **Análise de Resultados**
   a) Compare a política ótima encontrada com:
      - Manutenção preventiva a cada 2.500h (€6.38/h)
      - Manutenção preventiva a cada 5.000h (€4.84/h)
   b) Analise como os diferentes custos influenciam a política ótima
   c) Como a inclusão de falhas aleatórias afeta a solução comparada ao Grid World?

## Dados para Verificação

Para validar sua solução, considere os seguintes pontos de verificação:

- Custo médio ótimo esperado: €2.113/h
- Número típico de iterações até convergência: 4
- Estados 6 e 7 são sempre tratados com manutenção corretiva

## Solução

### Questão 1

#### Matriz de Transição $P(s'|s,a)$

Primeiro, vamos entender como construir a matriz $P(s'|s,NA)$. Para cada estado $s$, precisamos determinar as probabilidades de transição para todos os possíveis estados $s'$ quando nenhuma ação é tomada.

1. **Estados não-terminais (1-5)**

Para cada estado $i$ de $1$ a $5$:

- Probabilidade de permanecer no mesmo estado $i$:
     $P(s' = i|s = i) = 1 - \lambda - \lambda_0 = 0.9961$
- Probabilidade de deteriorar para próximo estado $i+1$:
     $P(s' = i+1|s = i) = \lambda = 0.0012$
- Probabilidade de falha aleatória (ir para estado 7):
     $P(s' = 7|s = i) = \lambda_0 = 0.0027$
- Todas outras probabilidades são 0

1. **Estados terminais (6-7)**

- Estado 6 (falha por deterioração):
     $P(s' = 6|s = 6) = 1$ (absorvente)
- Estado 7 (falha aleatória):
     $P(s' = 7|s = 7) = 1$ (absorvente)

Por exemplo, para o estado 1:

- $P(1|1) = 0.9961$ (permanecer)
- $P(2|1) = 0.0012$ (deteriorar)
- $P(7|1) = 0.0027$ (falha aleatória)
- $P(3|1) = P(4|1) = P(5|1) = P(6|1) = 0$ 

Considerando:

- Taxa de deterioração: $\lambda = 0.0012$
- Taxa de falha aleatória: $\lambda_0 = 0.0027$
- Probabilidade de permanecer: $1 - \lambda - \lambda_0 = 0.9961$

$$ P(s'|s,NA) = \begin{bmatrix}
0.9961 & 0.0012 & 0 & 0 & 0 & 0 & 0.0027 \\
0 & 0.9961 & 0.0012 & 0 & 0 & 0 & 0.0027 \\
0 & 0 & 0.9961 & 0.0012 & 0 & 0 & 0.0027 \\
0 & 0 & 0 & 0.9961 & 0.0012 & 0 & 0.0027 \\
0 & 0 & 0 & 0 & 0.9961 & 0.0012 & 0.0027 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} $$

#### Para Manutenção Menor ($MM$)

A matriz $P(s'|s,MM)$ é construída considerando que:

1. **MM só é aplicável nos estados 2-5**
Para cada estado $i$ de 2 a 5:

- Probabilidade de retornar ao estado 1: $P(1|i) = 0.7$
- Probabilidade de retornar ao estado 2: $P(2|i) = 0.2$
- Probabilidade de retornar ao estado 3: $P(3|i) = 0.1$
- Todas outras probabilidades são 0

2. **Estado 1 e estados 6-7**

- Marcados como NA pois a ação não é aplicável

Por exemplo, para o estado 3:

- $P(1|3) = 0.7$ (retorno ao estado perfeito)
- $P(2|3) = 0.2$ (retorno ao estado 2)
- $P(3|3) = 0.1$ (permanecer no estado 3)
- $P(4|3) = P(5|3) = P(6|3) = P(7|3) = 0$

Aplicável apenas aos estados $2-5$:

$$ P(s'|s,MM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

Onde NA indica que a ação não é aplicável neste estado.

### 3. Matriz para Manutenção Preventiva (PM)

A matriz $P(s'|s,PM)$ segue uma lógica similar:

1. **PM só é aplicável nos estados 2-5**
Para cada estado $i$ de 2 a 5:
- Probabilidade de retornar ao estado 1: $P(1|i) = 0.9$
- Probabilidade de retornar ao estado 2: $P(2|i) = 0.09$
- Probabilidade de retornar ao estado 3: $P(3|i) = 0.01$
- Todas outras probabilidades são 0

2. **Estado 1 e estados 6-7**
- Marcados como NA pois a ação não é aplicável

Aplicável apenas aos estados $2-5$:

$$ P(s'|s,PM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

### 4. Matriz para Manutenção Corretiva (CM)

A matriz $P(s'|s,CM)$ é a mais simples:

1. **CM só é aplicável nos estados 6-7**
Para estados 6 e 7:
- Probabilidade de retornar ao estado 1: $P(1|i) = 1$
- Todas outras probabilidades são 0

2. **Estados 1-5**
- Marcados como NA pois a ação não é aplicável

Aplicável apenas aos estados $6-7$:

$$ P(s'|s,CM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix} $$

#### Verificação de Consistência

Para cada matriz, devemos verificar se:

1. Todas as linhas somam 1 (propriedade fundamental de probabilidades)
2. Não existem probabilidades negativas
3. Estados marcados como NA não têm transições definidas

1. **Função de Recompensa $R(s,a,s')$**

A função de recompensa é definida como:

$$ R(s,a,s',t) = C_{insp}(s) + C_{ação}(a) + C_{falha}(s') + C_{estado}(s) \cdot t $$

Onde:
- $C_{insp}(s) = €200$ para estados não terminais
- $C_{ação}(a)$ =
 * $NA$: $€0$
 * $MM$: $€3.000$
 * $PM$: $€7.500$
 * $CM$: $€150.000$
- $C_{falha}(s') = €180.000$ se $s'$ é estado $6$ ou $7$
- $C_{estado}(s) \cdot t = €1.000/h \times t$ para estados não terminais, onde $t$ é o intervalo entre inspeções

3. **Estados Terminais**

a) Estado 6 (Falha por Deterioração)
- Características:
 * Estado absorvente sem deterioração natural
 * Requer manutenção corretiva ($CM$)
 * Custo total = $€330.000$ ($€150.000 + €180.000$)
 * Transição determinística para estado $1$ apenas após $CM$
 * Permanece no estado 6 se nenhuma ação é tomada
 * Não possui custos de inspeção ou estado não detectado

b) Estado 7 (Falha Aleatória)
- Características:
 * Estado absorvente sem deterioração natural;
 * Requer manutenção corretiva ($CM$);
 * Custo total = $€330.000$ ($€150.000 + €180.000$);
 * Transição determinística para estado $1$ apenas após $CM$;
 * Permanece no estado 7 se nenhuma ação é tomada;
 * Não possui custos de inspeção ou estado não detectado.

## Solução da Questão 1: Modelagem do Sistema de Manutenção de Turbinas Eólicas

### Introdução

O problema de manutenção de turbinas eólicas pode ser modelado como um MDP com 7 estados representando diferentes níveis de deterioração e falha. Vamos construir passo a passo as matrizes de transição para cada ação possível e a função de recompensa associada.

Antes de começarmos, devemos lembrar que os estados são: 

- Estado 1: Condição perfeita
- Estado 2: Desgaste leve
- Estado 3: Desgaste moderado
- Estado 4: Desgaste avançado
- Estado 5: Desgaste severo
- Estado 6: Falha por deterioração
- Estado 7: Falha aleatória (Poisson)


### 1. Construção das Matrizes de Transição

#### 1.1 Matriz para Nenhuma Ação (NA)

A primeira matriz que construiremos é $P(s'|s,NA)$, que representa as transições naturais do sistema quando nenhuma ação de manutenção é tomada.

**Parâmetros Iniciais:**
- Taxa de deterioração: $\lambda = 0.0012$ (1/833 dias)
- Taxa de falha aleatória: $\lambda_0 = 0.0027$ (1/ano)
- Probabilidade de permanecer: $1 - \lambda - \lambda_0 = 0.9961$

**Processo de Construção:**

1. **Estados não-terminais (1-5)**: Para cada estado $i$ de 1 a 5:
   - Permanecer no mesmo estado: $P(s' = i|s = i) = 0.9961$
   - Deteriorar para próximo estado: $P(s' = i+1|s = i) = 0.0012$
   - Falha aleatória: $P(s' = 7|s = i) = 0.0027$
   - Demais probabilidades: 0

2. **Estados terminais (6-7)**:
   - Estado 6: $P(s' = 6|s = 6) = 1$ (absorvente)
   - Estado 7: $P(s' = 7|s = 7) = 1$ (absorvente)

Isto resulta na matriz:

$$ P(s'|s,NA) = \begin{bmatrix}
0.9961 & 0.0012 & 0 & 0 & 0 & 0 & 0.0027 \\
0 & 0.9961 & 0.0012 & 0 & 0 & 0 & 0.0027 \\
0 & 0 & 0.9961 & 0.0012 & 0 & 0 & 0.0027 \\
0 & 0 & 0 & 0.9961 & 0.0012 & 0 & 0.0027 \\
0 & 0 & 0 & 0 & 0.9961 & 0.0012 & 0.0027 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} $$

**Verificação:**
- Cada linha soma 1: $0.9961 + 0.0012 + 0.0027 = 1$
- Probabilidades não-negativas
- Estados terminais são absorventes.

#### 1.2 Matriz para Manutenção Menor (MM)

A matriz $P(s'|s,MM)$ representa as transições quando aplicamos manutenção menor. Esta ação só é possível nos estados 2 a 5.

**Processo de Construção:**

1. **Estados Aplicáveis (2-5)**:
  - Probabilidade de retorno ao estado perfeito (1): $P(1|i) = 0.7$
  - Probabilidade de retorno ao estado 2: $P(2|i) = 0.2$
  - Probabilidade de retorno ao estado 3: $P(3|i) = 0.1$
  - Todas outras probabilidades: 0

2. **Estados Não-Aplicáveis (1,6,7)**:
  - Marcados como NA (Não Aplicável)

A matriz resultante é:

$$ P(s'|s,MM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

**Explicação dos Valores:**
- $0.7$ representa alta probabilidade de recuperação total
- $0.2$ indica chance moderada de recuperação parcial
- $0.1$ representa possibilidade de recuperação limitada
- Soma das probabilidades para cada estado aplicável = 1

#### 1.3 Matriz para Manutenção Preventiva (PM)

A matriz $P(s'|s,PM)$ representa transições após manutenção preventiva, que também só é aplicável nos estados 2 a 5, mas com maior probabilidade de retorno ao estado perfeito.

**Processo de Construção:**

1. **Estados Aplicáveis (2-5)**:
  - Probabilidade de retorno ao estado 1: $P(1|i) = 0.9$
  - Probabilidade de retorno ao estado 2: $P(2|i) = 0.09$
  - Probabilidade de retorno ao estado 3: $P(3|i) = 0.01$
  - Todas outras probabilidades: 0

2. **Estados Não-Aplicáveis (1,6,7)**:
  - Marcados como NA

A matriz resultante é:

$$ P(s'|s,PM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

**Comparação com MM:**
- PM tem maior probabilidade de retorno ao estado perfeito (0.9 vs 0.7)
- Menor probabilidade de estados intermediários
- Custo mais elevado que MM

#### 1.4 Matriz para Manutenção Corretiva (CM)

A matriz $P(s'|s,CM)$ representa transições após manutenção corretiva, que é aplicável apenas nos estados de falha (6 e 7).

**Processo de Construção:**

1. **Estados Aplicáveis (6-7)**:
  - Probabilidade de retorno ao estado 1: $P(1|i) = 1$
  - Todas outras probabilidades: 0

2. **Estados Não-Aplicáveis (1-5)**:
  - Marcados como NA

A matriz resultante é:

$$ P(s'|s,CM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix} $$

**Características Importantes:**
- Determinística (probabilidade 1 de retorno ao estado 1)
- Mais cara que outras ações
- Única opção nos estados de falha

### 2. Construção da Função de Recompensa

A função de recompensa total $R(s,a,s',t)$ é decomposta em custos imediatos e custos dependentes do tempo:

#### 2.1 Custos Imediatos

$$ C_{immediate}(s,a,s') = C_{insp}(s) + C_{ação}(a) + C_{falha}(s') $$

Onde:

1. **Custo de Inspeção** $C_{insp}(s)$:

$$ C_{insp}(s) = \begin{cases}
200, & \text{se } s \in \{1,2,3,4,5\} \\
0, & \text{se } s \in \{6,7\}
\end{cases} $$

2. **Custo da Ação** $C_{ação}(a)$:

$$ C_{ação}(a) = \begin{cases}
0, & \text{se } a = NA \\
3.000, & \text{se } a = MM \\
7.500, & \text{se } a = PM \\
150.000, & \text{se } a = CM
\end{cases} $$

3. **Custo de Falha** $C_{falha}(s')$:

$$ C_{falha}(s') = \begin{cases}
180.000, & \text{se } s' \in \{6,7\} \\
0, & \text{caso contrário}
\end{cases} $$

#### 2.2 Custos Dependentes do Tempo

$$ C_{time}(s,t) = C_{estado}(s) \cdot t $$

Onde:

- $C_{estado}(s) = €1.000/h$ para estados não terminais
- $t$ é o intervalo de inspeção em horas
- Intervalo: $t \in [500, 20.000]$ horas

Além disso, devemos lembrar que:

- Intervalo mínimo: 500h
- Intervalo máximo: 20.000h
- Discretização: 20 intervalos iguais de ((20.000 - 500)/20) = 975h

#### 2.3 Função de Recompensa Total

$$ R(s,a,s',t) = C_{immediate}(s,a,s') + C_{time}(s,t) $$

**Exemplo de Cálculo:**

Para estado 2, ação MM, próximo estado 1, intervalo 1000h:

C_immediate = €200 + €3.000 + €0 = €3.200
C_time = €1.000/h × 1000h = €1.000.000
R(2,MM,1,1000) = €3.200 + €1.000.000 = €1.003.200

### 3. Verificação Final do Modelo

#### 3.1 Verificação das Matrizes de Transição

- **Propriedade Estocástica**: Todas as linhas das matrizes somam 1 para estados aplicáveis
- **Não-negatividade**: Todas as probabilidades são não-negativas
- **Estados Absorventes**: Estados 6 e 7 são corretamente modelados como absorventes sob NA
- **Consistência das Ações**: Cada ação é aplicável apenas nos estados definidos

#### 3.2 Verificação da Função de Recompensa

- **Custos Não-negativos**: Todos os custos são positivos ou zero
- **Consistência Temporal**: Custos por hora são proporcionais ao intervalo de inspeção
- **Discretização Adequada**: Os 20 intervalos de inspeção são operacionalmente factíveis
- **Custo Total**: Corretamente combina custos imediatos e temporais

#### 3.3 Propriedades do MDP

- **Horizonte Infinito**: O modelo pode ser executado indefinidamente
- **Estacionariedade**: As probabilidades de transição não mudam com o tempo
- **Observabilidade**: O estado do sistema é conhecido após cada inspeção
- **Markoviano**: Decisões dependem apenas do estado atual

Este MDP modela adequadamente o problema de manutenção de turbinas eólicas, capturando:
1. Deterioração natural do equipamento
2. Possibilidade de falhas aleatórias
3. Diferentes opções de manutenção
4. Estrutura de custos realista
5. Intervalos de inspeção práticos
