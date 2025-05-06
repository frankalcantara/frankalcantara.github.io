---
layout: post
title: "MDP: Casos Reais - Manutenção de Turbinas Eólicas"
author: Frank
categories:
    - artigo
    - Matemática
    - Inteligência Artificial
tags:
    - algoritmos
    - C++
    - inteligência artificial
    - resolução de problemas
    - paradigmas de programação
image: assets/images/deep1.webp
featured: false
rating: 5
description: Aplicação prática do MDP de um artigo científico, detalhado de forma didática. Muita matemática.
date: 2025-02-23T14:35:42.669Z
preview: Usando o MDP para resolver problemas da vida real. Neste caso um problema de manutenção de turbinas eólicas.
keywords: Reinforcement Learning, Markov Decision Processes (MDP), Manutenção de Turbinas Eólicas, Manutenção Preditiva Industrial, General Electric (GE), Grid World, Otimização de Manutenção, Distribuição de Poisson, Equações de Bellman, Iteração de Valor, Iteração de Política, Custo de Manutenção, Turbinas Eólicas, Estados de Deterioração, Manutenção Preventiva, Manutenção Corretiva, Intervalos de Inspeção, Aplicações de MDP, Eficiência Industrial, Redução de Custos Operacionais
toc: true
published: true
lastmod: 2025-05-06T11:04:17.964Z
slug: mdp-casos-reais-manutencao-de-turbinas-eolicas
---

A curiosa leitora deve estar decepcionada. Quando comecei esta jornada eu prometi que estudaríamos Reinforcement Learning e, até agora, nada. Só o básico fundamental. Peço desculpas.

Eu estou me divertindo enquanto escrevo e ainda descubro dúvidas que nem sabia que tinha. Analisando os artigos anteriores: [1](https://frankalcantara.com/reinforcement-learning-hist%C3%B3ria/), [2](https://frankalcantara.com/entendendo-mdp/). [3](https://frankalcantara.com/um-mundo-em-uma-grade/) e [4](https://frankalcantara.com/resolvendo-o-grid-world/) percebi que o texto carece de aplicações práticas.

Pesquisando, encontrei alguns casos interessantes de aplicações de **MDP** em situações da vida real que foram acadêmica e cientificamente descritos em artigos publicados. Escolhi aqueles que continham mais dados e fossem compatíveis com os meus interesses pessoais. A compadecida leitora há de perdoar este pobre autor egoísta.

## Manutenção Ótima de Turbinas Eólicas

Na indústria, a Manutenção Preditiva Industrial busca otimizar a manutenção de equipamentos, antecipando falhas e minimizando o tempo de inatividade. Em um cenário de **MDP**, os estados representam os diferentes níveis de desgaste de um equipamento. As ações incluem realizar manutenção, continuar a operação ou substituir uma peça. As recompensas são medidas em termos do balanço entre os custos de manutenção e o tempo de operação contínua.

Este é um tema que tem impacto na eficiência do processo industrial e, consequentemente, na competitividade. Um caso real de matemática, ciência se usar o termo de forma geral, aplicada para impactar no mercado. O artigo que apareceu de forma mais relevante nas duas buscas que fiz sobre o uso prático de **MDP** destaca um caso emblemático envolvendo  a **General Electric (GE)**, que utiliza **MDPs** *para otimizar a manutenção de turbinas eólicas, considerando fatores como desgaste, condições climáticas e demanda de energia*[^1].

[^1]:WU, Yan-Ru; ZHAO, Hong-Shan. **Optimization Maintenance of Wind Turbines Using Markov Decision Processes**. Disponível em: https://www.researchgate.net/publication/241177157_Optimization_maintenance_of_wind_turbines_using_Markov_decision_processes. Acesso em: 12 fev. 2025.

Este sistema permite à **GE** realizar manutenções de forma mais eficiente, prolongando a vida útil das turbinas e reduzindo custos operacionais.

O artigo "Optimization Maintenance of Wind Turbines Using Markov Decision Processes"[^1] utiliza o modelo *Semi-Markov Decision Process (**SMDP**)*. O **SMDP** é uma extensão do **Markov Decision Process (MDP)** que permite modelar sistemas onde o tempo gasto em cada estado segue uma distribuição de probabilidade arbitrária.

### Diferenças entre MDP e SMDP

| Característica | MDP | SMDP |
|---|---|---|
| Tempo de Transição | Fixo (discreto) | Distribuição de probabilidade arbitrária (contínuo) |
| Ações | Tomadas em intervalos de tempo fixos | Tomadas em cada mudança de estado |
| Modelagem de Manutenção | Menos adequado para problemas com tempos variáveis entre inspeções e ações de manutenção | Mais adequado para problemas com tempos variáveis entre inspeções e ações de manutenção |

O artigo utiliza o modelo **SMDP** por ser mais adequado para capturar a natureza contínua e estocástica do processo de deterioração da caixa de engrenagens, além de permitir a otimização dos intervalos de inspeção e das ações de manutenção.

Ainda assim, transformar os dados do artigo em um exercícios de **MDP** e tentar resolver de forma didática, comparando com o **Grid World**. Servirá como banho de vida real e introdução ao **SMDP**.

>A sagaz leitora pode anotar essa ideia: a melhor forma, tanto para sociedade, quanto para você, de entender qualquer artigo científico é refazer a pesquisa ou, no mínimo, refazer todos os cálculos.

### Exercício: Manutenção Ótima de Turbinas Eólicas via MDP

Uma empresa de energia eólica busca otimizar sua política de manutenção de [caixas de engrenagem](https://www.energy.gov/eere/wind/how-wind-turbine-works-text-version) (*gearboxes*) de turbinas eólicas usando **Processos de Decisão de Markov (MDP)**. Similar ao **Grid World**, no qual, um agente navega em uma grade buscando maximizar recompensas, *aqui temos um sistema que navega entre estados de deterioração buscando minimizar custos*.

#### Construindo o Modelo

##### Estados ($S$)

O sistema possui $7$ estados, mais simples que os $12$ estados que usamos no exemplo do  **Grid World**:

1. Estado perfeito, nenhum desgaste. Análogo ao estado inicial do Grid World;
2. Desgaste leve;
3. Desgaste moderado;
4. Desgaste avançado;
5. Desgaste severo;
6. Falha por deterioração. Um estado de recompensa negativa, análogo ao estado terminal negativo que usamos no **Grid World**;
7. Falha aleatória, segundo a Distribuição de Poisson. Neste caso, temos dois estados com recompensas negativas;

>No contexto de manutenção de turbinas eólicas, além da deterioração natural do equipamento, o modelo considera falhas aleatórias que podem ocorrer independentemente do estado de desgaste.
>
>O artigo[^1] especifica uma taxa de falha aleatória $\lambda_0 = 0.0027$, equivalente a aproximadamente uma falha por ano.
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
>$$ P(s' = 7 \vert  s, a) = \lambda_0 = 0.0027 $$
>
Esta probabilidade será incorporada na matriz de transição $P(s' \vert  s,a)$ como uma transição possível de cada estado não-terminal para o estado $7$, considerando três possibilidades em cada estado:
>
>- Deterioração natural (com taxa $\lambda$)
>- Falha aleatória (com taxa $\lambda_0$)
>- Permanência no estado atual (com taxa $1 - \lambda - \lambda_0$)
>
> A atenta leitora deve notar que esta é uma simplificação do modelo real. Na prática, equipamentos muito desgastados podem ter taxas de falha aleatória maiores. Porém, para manter a propriedade de Markov, assumimos $\lambda_0$ constante independente do estado.

[^2]: DEVORE, Jay L. **Probability and Statistics for Engineering and the Sciences**. 9. ed. Boston: Cengage Learning, 2016.

##### Ações ($A$)

Em cada estado não-terminal, as seguintes ações estão disponíveis. Novamente, podemos fazer uma analogia com o **Grid World**. Quando estudamos **MDP** com o **Grid World** usamos uma ação para cada direção possível:

- $NA$: Nenhuma ação;
- $MM$: Manutenção menor, alta probabilidade de retorno ao estado perfeito;
- $PM$: Manutenção preventiva; retorna ao estado perfeito;
- $CM$: Manutenção corretiva; após falha, retorna ao estado perfeito.

##### Dinâmica do Sistema

###### Parâmetros Operacionais

- Turbina: $5MW$, fator de capacidade $0.4$, implicando em $2MW$ efetivos;
- Custo da energia: $0.5/kWh$;
- Horizonte de tempo: $100.000$ horas;
- Tempo de reparo após falha: $15$ dias;

###### Taxas de Transição

- Deterioração entre estados: $\lambda = 0.0012$ (1/833 dias);
- Falha aleatória: $\lambda_0 = 0.0027$ (1/ano);
- Fator de desconto: $\gamma = 0.9$ (análogo ao **Grid World**);

###### Probabilidades de Manutenção

Para $MM$ em estados $2-5$:

| Estado   | Prob. Transição |
|----------|-----------------|
| Estado 1 | $0.7$           |
| Estado 2 | $0.2$           |
| Estado 3 | $0.1$           |

Esse $-2$ representa dois estados antes do atual. O mesmo vale para o $-3$

Para $PM$ em estados $2-5$:

| Estado    | Prob. Transição |
|-----------|-----------------|
| Perfeito  | $0.9$           |
| Estado 2  | $0.09$          |
| Estado 3  | $0.01$          |

###### Sistema de Recompensas (Custos)

Análogo ao sistema de recompensas do **Grid World** acrescido da informação de custos:

- Inspeção: $200$, análogo ao custo por passo do Grid World;
- Manutenção menor ($MM$): $3.000$;
- Manutenção preventiva ($PM$): $7.500$;
- Manutenção corretiva (CM): $150.000$;
- Perda por falha: $180.000$, análogo à penalidade do estado terminal negativo;
- Estado não detectado: $1.000/hora$.

De forma mais clara, teremos:

| Ação | Significado            | Custo   |
|------|------------------------|---------|
| NA   | Não fazer nada         | 0       |
| MM   | Manutenção menor       | 3.000   |
| PM   | Manutenção preventiva  | 7.500   |
| CM   | Troca completa         | 150.000 |
| --   | Estado não detectado   | 1.000   |

#### Intervalos de Inspeção

- Mínimo: $500$ horas;
- Máximo: $20.000$ horas;
- Discretizado em $20$ intervalos iguais.

A perspicaz leitora deve ter ficado curiosa com estes intervalos de inspeção.

Quando estudamos o **Grid World**, ressaltei que a cada passo o agente observa perfeitamente seu estado atual. O agente sempre sabe em qual célula está. Isso significa que o **Grid World** é um ambiente com observabilidade total.

Nas condições apresentadas no artigo[^1] não existe a observabilidade total. O intervalo de inspeção representa o tempo entre observações do estado real do sistema. Só é possível saber o estado da turbina quando são realizadas inspeções. O que torna este problema mais complexo que o **Grid World**. Este é o custo que temos que pagar para bisbilhotar o mundo real.

Como, entre inspeções, o sistema evolui sem que saibamos seu estado exato, o custo/benefício de fazer inspeções mais ou menos frequentes precisa ser considerado.

A amável leitora deve considerar que, se não conhecemos a forma como o sistema evolui, uma falha pode ocorrer entre inspeções sem ser detectada.

No **Grid World** isso seria como se o agente só pudesse observar sua posição a cada $N$ passos aleatórios e o o agente pudesse cair em estados terminais negativos sem saber.

Por isso os autores do artigo[^1] precisaram otimizar também o intervalo entre inspeções.

### Questões

1. **Modelagem do Sistema**
   a) Formule a matriz de transição $P(s' \vert  s,a)$ para cada ação;
   b) Defina a função de recompensa $R(s,a,s')$;
   c) Identifique os estados terminais e suas características.

2. **Equações de Bellman**
   a) Escreva a equação para a função valor-estado $V^{*}(s)$;
   b) Escreva a equação para a função valor-ação $Q^*(s,a)$;
   c) Como estas equações se comparam com as do Grid World?

3. **Solução por Iteração de Valor**
   a) Inicialize os valores de estado;
   b) Aplique o processo iterativo usando a equação de Bellman;
   c) Mostre o cálculo detalhado para a primeira iteração no estado perfeito.

4. **Solução por Iteração de Política**
   a) Defina uma política inicial;
   b) Realize a avaliação de política;
   c) Execute a melhoria de política;
   d) Compare a convergência com o caso do Grid World.

5. **Análise de Resultados**
   a) Compare a política ótima encontrada com:
      - Manutenção preventiva a cada 2.500h (6.38/h);
      - Manutenção preventiva a cada 5.000h (4.84/h);
   b) Analise como os diferentes custos influenciam a política ótima;
   c) Como a inclusão de falhas aleatórias afeta a solução comparada ao Grid World?

### Dados para Verificação

Para validar sua solução, considere os seguintes pontos de verificação:

- Custo médio ótimo esperado: 2.113/h;
- Número típico de iterações até convergência: $4$;
- Estados $6$ e $7$ são sempre tratados com manutenção corretiva;

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

### Construção das Matrizes de Transição

#### Matriz para Nenhuma Ação \text{(NA)}

A primeira matriz que construiremos é $P(s' \vert  s,NA)$, que representa as transições naturais do sistema quando nenhuma ação de manutenção é tomada.

**Parâmetros Iniciais:**

- Taxa de deterioração: $\lambda = 0.0012$ (1/833 dias);
- Taxa de falha aleatória: $\lambda_0 = 0.0027$ (1/ano);
- Probabilidade de permanecer: $1 - \lambda - \lambda_0 = 0.9961$.

**Processo de Construção:**

1. **Estados não-terminais (1-5)**: Para cada estado $i$ de 1 a 5:

   - Permanecer no mesmo estado: $P(s' = i \vert  s = i) = 0.9961$;
   - Deteriorar para próximo estado: $P(s' = i+1 \vert  s = i) = 0.0012$;
   - Falha aleatória: $P(s' = 7 \vert  s = i) = 0.0027$;
   - Demais probabilidades: $0$.

2. **Estados terminais (6-7)**:

   - Estado 6: $P(s' = 6 \vert  s = 6) = 1$ (absorvente);
   - Estado 7: $P(s' = 7 \vert  s = 7) = 1$ (absorvente).

Isto resulta na matriz:

$$ P(s' \vert  s,NA) = \begin{bmatrix}
0.9961 & 0.0012 & 0 & 0 & 0 & 0 & 0.0027 \\
0 & 0.9961 & 0.0012 & 0 & 0 & 0 & 0.0027 \\
0 & 0 & 0.9961 & 0.0012 & 0 & 0 & 0.0027 \\
0 & 0 & 0 & 0.9961 & 0.0012 & 0 & 0.0027 \\
0 & 0 & 0 & 0 & 0.9961 & 0.0012 & 0.0027 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} $$

**Verificação:**

- Cada linha soma $$1$: $0.9961 + 0.0012 + 0.0027 = 1$;
- Probabilidades não-negativas;
- Estados terminais são absorventes.

### Matriz para Manutenção Menor $\text{(MM)}$

A matriz $P(s' \vert  s,MM)$ representa as transições quando aplicamos manutenção menor. Esta ação só é possível nos estados 2 a 5.

**Processo de Construção:**

1. **Estados Aplicáveis (2-5)**:

  - Probabilidade de retorno ao estado perfeito (1): $P(1 \vert  i) = 0.7$;
  - Probabilidade de retorno ao estado 2: $P(2 \vert  i) = 0.2$;
  - Probabilidade de retorno ao estado 3: $P(3 \vert  i) = 0.1$;
  - Todas outras probabilidades: $0$.

2. **Estados Não-Aplicáveis (1,6,7)**:

  - Marcados como $\text{(NA)}$ (Não Aplicável).

A matriz resultante é:

$$ P(s' \vert  s,MM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
0.7 & 0.2 & 0.1 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

**Explicação dos Valores:**

- $0.7$ representa a alta probabilidade de retorno ao estado perfeito (estado 1);
- $0.2$ indica a chance moderada de permanecer ou retornar ao estado de desgaste leve (estado 2);
- $0.1$ representa a possibilidade de transição para o estado de desgaste moderado (estado 3);
- A soma das probabilidades para cada estado aplicável é igual a $1$.

#### Matriz para Manutenção Preventiva $\text{(PM)}$

A matriz $P(s' \vert  s,PM)$ representa transições após manutenção preventiva, que também só é aplicável nos estados 2 a 5, mas com maior probabilidade de retorno ao estado perfeito.

**Processo de Construção:**

1. **Estados Aplicáveis (2-5)**:

  - Probabilidade de retorno ao estado 1: $P(1 \vert  i) = 0.9$;
  - Probabilidade de retorno ao estado 2: $P(2 \vert  i) = 0.09$;
  - Probabilidade de retorno ao estado 3: $P(3 \vert  i) = 0.01$;
  - Todas outras probabilidades: $0$.

2. **Estados Não-Aplicáveis (1,6,7)**:

  - Marcados como \text{(NA)}.

A matriz resultante é:

$$ P(s' \vert  s,PM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
0.9 & 0.09 & 0.01 & 0 & 0 & 0 & 0 \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA}
\end{bmatrix} $$

**Comparação com \text{(MM)}:**

- PM tem maior probabilidade de retorno ao estado perfeito ($0.9$ vs $0.7$);
- Menor probabilidade de estados intermediários;
- Custo mais elevado que \text{(MM)}

#### Matriz para Manutenção Corretiva \text{(CM)}

A matriz $P(s' \vert  s,CM)$ representa transições após manutenção corretiva, que é aplicável apenas nos estados de falha ($6$ e $7$).

**Processo de Construção:**

1. **Estados Aplicáveis (6-7)**:
  - Probabilidade de retorno ao estado 1: $P(1 \vert  i) = 1$
  - Todas outras probabilidades: 0

2. **Estados Não-Aplicáveis (1-5)**:
  - Marcados como NA

A matriz resultante é:

$$ P(s' \vert  s,CM) = \begin{bmatrix}
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
\text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} & \text{NA} \\
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix} $$

**Características Importantes:**

- Determinística (probabilidade $1$ de retorno ao estado $1$);
- Mais cara que outras ações;
- Única opção nos estados de falha.

### Construção da Função de Recompensa

A função de recompensa total $R(s,a,s',t)$ é decomposta em custos imediatos e custos dependentes do tempo:

#### Custos Imediatos

$$ C_{immediate}(s,a,s') = C_{insp}(s) + C_{ação}(a) + C_{falha}(s') $$

Nesta equação temos:

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

#### Custos Dependentes do Tempo

$$ C_{time}(s,t) = C_{estado}(s) \cdot t $$

Neste caso, temos:

- $C_{estado}(s) = 1.000/h$ para estados não terminais;
- $t$ é o intervalo de inspeção em horas;
- Intervalo: $t \in [500, 20.000]$ horas.

Além disso, devemos lembrar que:

- Intervalo mínimo: $500h$;
- Intervalo máximo: $20.000h$;
- Discretização: $20$ intervalos iguais de $((20.000 - 500)/20) = 975h$;

#### Função de Recompensa Total

$$R(s,a,s',t) = C_{immediate}(s,a,s') + C_{time}(s,t)$$

**Exemplo de Cálculo:**

Para estado $2$, ação \text{(MM)}, próximo estado $1$, intervalo $1000h$:

$$C_immediate = 200 + 3.000 + 0 = 3.200$$

$$C_time = 1.000/h \times 1000h = 1.000.000$$

$$R(2,MM,1,1000) = 3.200 + 1.000.000 = 1.003.200$$

### Verificação do Modelo

#### Verificação das Matrizes de Transição

- **Propriedade Estocástica**: todas as linhas das matrizes somam 1 para estados aplicáveis;
- **Não-negatividade**: todas as probabilidades são não-negativas;
- **Estados Absorventes**: estados $6$ e $7$ são corretamente modelados como absorventes sob \text{(NA)};
- **Consistência das Ações**: cada ação é aplicável apenas nos estados definidos.

#### Verificação da Função de Recompensa

- **Custos Não-negativos**: todos os custos são positivos ou zero;
- **Consistência Temporal**: custos por hora são proporcionais ao intervalo de inspeção;
- **Discretização Adequada**: os $20$ intervalos de inspeção são operacionalmente factíveis;
- **Custo Total**: corretamente combina custos imediatos e temporais.

#### Propriedades do MDP

- **Horizonte Infinito**: o modelo pode ser executado indefinidamente;
- **Estacionariedade**: as probabilidades de transição não mudam com o tempo;
- **Observabilidade**: o estado do sistema é conhecido após cada inspeção;
- **Markoviano**: decisões dependem apenas do estado atual.

Ou, em outras palavras, a atenta leitora deve perceber que este exercício modela adequadamente o problema de manutenção de turbinas eólicas, capturando:

1. Deterioração natural do equipamento;
2. Possibilidade de falhas aleatórias;
3. Diferentes opções de manutenção;
4. Estrutura de custos realista;
5. Intervalos de inspeção práticos.

## Solução da Questão 2: Equações de Bellman

Agora vamos resolver a Questão 2, que exige a formulação das equações de Bellman para este problema de manutenção de turbinas eólicas e uma comparação com o Grid World.

### Equação para a Função Valor-Estado $V^{*}(s)$

A função valor-estado $V^{*}(s)$ representa o valor esperado de longo prazo (neste caso, custo mínimo) a partir do estado $s$, seguindo a política ótima $\pi^{*}$. Para um horizonte infinito com desconto, a equação de Bellman é dada por:

$$
V^{*}(s) = \min_{a \in A_s} \left[ \sum_{s' \in S} P(s'\mid s,a)[R(s,a,s') + \gamma V^{*}(s')] \right]
$$

Na qual, temos:

- $S$ é o conjunto de estados $\{1, 2, 3, 4, 5, 6, 7\}$;
- $A_s$ é o conjunto de ações disponíveis no estado $s$ (ex.: $\{NA\}$ em $s=1$, $\{NA, MM, PM\}$ em $s=2,3,4,5$, $\{CM\}$ em $s=6,7$);
- $R(s,a,s')$ é a recompensa imediata (custo) associada à transição do estado $s$ para $s'$ ao executar a ação $a$;
- $P(s'\mid s,a)$ é a probabilidade de transição de $s$ para $s'$ dado $a$;
- $\gamma = 0.9$ é o fator de desconto, conforme especificado.

Para o problema que estamos estudando, como os intervalos de inspeção $t$ são variáveis otimizáveis, a equação se torna dependente de $t$. Definimos uma função de recompensa estendida $R(s,a,t)$ que representa o custo esperado ao escolher a ação $a$ no estado $s$ com intervalo de inspeção $t$:

$$
R(s,a,t) = C_{insp}(s) + C_{ação}(a) + \sum_{s' \in S} P(s'\mid s,a) C_{falha}(s') + C_{estado}(s) \cdot t
$$

Esta função de recompensa estendida incorpora o valor esperado de $R(s,a,s')$ sobre todos os possíveis estados seguintes $s'$, acrescido do custo dependente do tempo. Assim, a equação de Bellman adaptada torna-se:

$$
V^{*}(s) = \min_{a \in A_s, t \in T_s} \left[ R(s,a,t) + \gamma \sum_{s' \in S} P(s'\mid s,a) V^{*}(s') \right]
$$

Neste caso, temos:

- $T_s$ é o conjunto de intervalos de inspeção possíveis ($500h$ a $20.000h$, discretizados em $20$ intervalos);

**Exemplo**: Para o estado $s=2$, com ações $\{NA, MM, PM\}$, a equação seria:

$$
V^{*}(2) = \min_{a \in \{NA, MM, PM\}, t \in T_2} \left[ R(2,a,t) + \gamma \sum_{s' \in \{1,2,3,7\}} P(s'\mid 2,a) V^{*}(s') \right]
$$

### Equação para a Função Valor-Ação $Q^*(s,a)$

A função valor-ação $Q^*(s,a)$ representa o custo esperado ao tomar a ação $a$ no estado $s$ e seguir a política ótima daí em diante. Ela é definida como:

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \vert  s,a) V^{*}(s')
$$

Incorporando o intervalo de inspeção $t$, temos:

$$
Q^*(s,a,t) = R(s,a,t) + \gamma \sum_{s' \in S} P(s' \vert  s,a) V^{*}(s')
$$

**Exemplo**: Para $s=2$ e $a=MM$, com um intervalo $t$ específico:

$$
Q^*(2,MM,t) = R(2,MM,t) + \gamma \left[ 0.7 V^{*}(1) + 0.2 V^{*}(2) + 0.1 V^{*}(3) \right]
$$

Nesta equação temos:

- $R(2,MM,t) = 200 + 3000 + 0 + 1000 \cdot t = 3200 + 1000t$ (sem custo de falha imediato);
- $\gamma = 0.9$.

A política ótima $\pi^{*}(s)$ será encontrada escolhendo a ação e intervalo que minimizam $Q^*$:

$$
\pi^{*}(s) = \arg\min_{a \in A_s, t \in T_s} Q^*(s,a,t)
$$

### Comparação com as Equações do Grid World

No **Grid World** apresentado nos artigos anteriores, as equações de Bellman são mais simples devido às seguintes características:

1. **Observabilidade Total**: no **Grid World**, o agente sempre sabe seu estado a cada passo, enquanto aqui o estado só é conhecido após as inspeções. Isso provocou a adição da variável $t$ às equações, tornando-as mais complexas.

2. **Ações Simples**: no **Grid World**, ações são movimentos ($\text{Norte}$, $\text{Sul}$, $\text{Leste}$, $\text{Oeste}$), com recompensas fixas por passo $(-1)$ e recompensas terminais ($+1$ ou $-1$). Aqui, ações envolvem manutenção com custos variados e dependentes de $t$.

3. **Horizonte e Recompensas**: o **Grid World** geralmente tem um horizonte finito, rodamos o modelo até atingir um estado terminal. No problema de manutenção das turbinas eólicas assumimos um horizonte infinito com desconto ($\gamma = 0.9$), refletindo a operação contínua da turbina.

**Equação do Grid World**:

$$
V^{*}(s) = \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} P(s' \vert  s,a) V^{*}(s') \right]
$$

Esta equação usa maximização (recompensas positivas) versus minimização aqui (custos negativos) e não inclui $t$, pois cada passo é uma decisão imediata.

**Similaridades**: as duas equações usam o princípio de otimalidade de Bellman. Ambas dependem de probabilidades de transição $P(s' \vert  s,a)$ e recompensas $R(s,a)$.

**Diferenças**: neste exercício, $R(s,a,t)$ é dinâmico com $t$, enquanto no **Grid World** é constante por passo. A inclusão de falhas aleatórias, segundo a distribuição de Poisson, e custos de estado não detectado adiciona complexidade ausente no **Grid World**.

A esforçada leitora deve atentar para o fato que, enquanto o **Grid World** é um ambiente controlado e abstrato, este problema reflete a realidade com incertezas e custos variáveis, exigindo uma adaptação mais sofisticada das equações de Bellman.

## Solução da Questão 3: Solução por Iteração de Valor

Vamos resolver a Questão 3, que exige a aplicação do método de iteração de valor para encontrar a função valor-estado ótima $V^{*}(s)$ no problema de manutenção de turbinas eólicas. Este método atualiza iterativamente os valores de cada estado até convergir para a solução ótima, usando a equação de Bellman.

### Inicializar os Valores de Estado

No método de iteração de valor, começamos com uma estimativa inicial para os valores de cada estado. Como estamos minimizando custos, o equivalente as recompensas negativas, uma escolha comum é inicializar todos os valores como zero, representando um cenário otimista inicial:

$$
V_0(s) = 0 \quad \text{para todo } s \in S = \{1, 2, 3, 4, 5, 6, 7\}
$$

Isso significa que teremos:

- $V_0(1) = 0$
- $V_0(2) = 0$
- $V_0(3) = 0$
- $V_0(4) = 0$
- $V_0(5) = 0$
- $V_0(6) = 0$
- $V_0(7) = 0$

A sagaz leitora deve notar que esta inicialização é arbitrária e outras escolhas, tais como custos altos iniciais poderiam ser usadas, mas $V_0(s) = 0$ simplifica os cálculos iniciais.

### Aplicar o Processo Iterativo Usando a Equação de Bellman

O método de iteração de valor atualiza os valores de estado em cada iteração $k$ usando a equação de Bellman adaptada para este problema, que inclui o intervalo de inspeção $t$:

$$
V_{k+1}(s) = \min_{a \in A_s, t \in T_s} \left[ R(s,a,t) + \gamma \sum_{s' \in S} P(s' \vert  s,a) V_k(s') \right]
$$

Nesta equação temos:

- $A_s$ é o conjunto de ações disponíveis no estado $s$ ($\{NA\}$ para $s=1$, $\{NA, MM, PM\}$ para $s=2,3,4,5$, $\{CM\}$ para $s=6,7$);
- $T_s$ é o conjunto de intervalos de inspeção ($500h$ a $20.000h$, discretizados em 20 intervalos de 975h cada);
- $R(s,a,t) = C_{insp}(s) + C_{ação}(a) + \sum_{s' \in S} P(s' \vert  s,a) C_{falha}(s') + C_{estado}(s) \cdot t$;
- $\gamma = 0.9$ é o fator de desconto;
- $P(s' \vert  s,a)$ é a probabilidade de transição definida nas matrizes da Questão 1.

O processo iterativo continua até que a diferença entre $V_{k+1}(s)$ e $V_k(s)$ seja menor que um limiar pequeno. Por exemplo, $\epsilon = 0.01$, indicando convergência.

Para simplificar os cálculos iniciais, assumiremos que o intervalo de inspeção $t$ será avaliado em um valor representativo, $t = 500h$, o mínimo, mas na prática, o método otimizaria $t$ em cada iteração. Vamos detalhar a primeira iteração para o estado perfeito como pedido no enunciado.

### Cálculo Detalhado para a Primeira Iteração no Estado Perfeito

Vamos calcular $V_1(1)$, o valor do estado perfeito ($s=1$) na primeira iteração, usando $V_0(s) = 0$ para todos os estados e $t = 500h$ como exemplo inicial.

#### Passo 1: Identificar Ações e Transições

No estado $s=1$, a única ação disponível é $NA$ (Nenhuma Ação). A matriz de transição para $NA$ é:

$$
P(s' \vert  1,NA) = \begin{cases}
0.9961, & \text{se } s' = 1 \\
0.0012, & \text{se } s' = 2 \\
0.0027, & \text{se } s' = 7 \\
0, & \text{caso contrário}
\end{cases}
$$

#### Passo 2: Calcular a Recompensa $R(1,NA,t)$

A função de recompensa é:

$$
R(1,NA,t) = C_{insp}(1) + C_{ação}(NA) + \sum_{s' \in S} P(s' \vert  1,NA) C_{falha}(s') + C_{estado}(1) \cdot t
$$

- $C_{insp}(1) = 200$ (custo de inspeção para estados não terminais);
- $C_{ação}(NA) = 0$ (nenhuma ação não tem custo);
- $C_{falha}(s')$:
  - $C_{falha}(1) = 0$ (sem falha);
  - $C_{falha}(2) = 0$ (sem falha);
  - $C_{falha}(7) = 180.000$ (falha aleatória);
  - $\sum_{s' \in S} P(s' \vert  1,NA) C_{falha}(s') = 0.9961 \cdot 0 + 0.0012 \cdot 0 + 0.0027 \cdot 180000 = 0 + 0 + 486 = 486$;
- $C_{estado}(1) = 1.000/h$ (custo por hora em estado não detectado);
- $t = 500h$;

Substituímos:

$$
R(1,NA,500) = 200 + 0 + 486 + 1000 \cdot 500 = 200 + 486 + 500000 = 500686
$$

#### Passo 3: Calcular o Termo de Desconto

Usamos os valores iniciais $V_0(s')$:

$$
\sum_{s' \in S} P(s' \vert  1,NA) V_0(s') = 0.9961 \cdot V_0(1) + 0.0012 \cdot V_0(2) + 0.0027 \cdot V_0(7)
$$

Como $V_0(1) = V_0(2) = V_0(7) = 0$:

$$
\sum_{s' \in S} P(s' \vert  1,NA) V_0(s') = 0.9961 \cdot 0 + 0.0012 \cdot 0 + 0.0027 \cdot 0 = 0
$$

Com $\gamma = 0.9$:

$$
\gamma \sum_{s' \in S} P(s' \vert  1,NA) V_0(s') = 0.9 \cdot 0 = 0
$$

#### Passo 4: Atualizar $V_1(1)$

Para $s=1$, com apenas uma ação ($NA$):

$$
V_1(1) = R(1,NA,500) + \gamma \sum_{s' \in S} P(s' \vert  1,NA) V_0(s')
$$

Substituímos:

$$
V_1(1) = 500686 + 0 = 500686
$$

#### Passo 5: Reflexão

O valor $V_1(1) = 500686$ reflete o custo esperado na primeira iteração para $t = 500h$. Este valor é alto devido ao termo $C_{estado}(1) \cdot t = 500000$, que representa o custo de permanecer no estado sem detecção por 500 horas. Na prática, precisaríamos testar todos os $t \in T_s$ ($500h$, $1475h$, $...$, $20.000h$) e escolher o mínimo, mas para exemplificar a primeira iteração, usamos $t = 500h$.

Para ilustrar, se usarmos $t = 2000h$:

- $R(1,NA,2000) = 200 + 0 + 486 + 1000 \cdot 2000 = 200 + 486 + 2000000 = 2000686$;
- $V_1(1) = 2000686$.

O valor aumenta com $t$, sugerindo que intervalos menores podem ser preferíveis inicialmente. Nas iterações seguintes, os valores $V_k(s')$ não serão mais zero, afetando a decisão.

A perspicaz leitora deve notar que a iteração de valor exige calcular $V_1(s)$ para todos os estados ($s=2$ a $s=7$), considerando todas as ações e intervalos possíveis. Aqui, focamos em $s=1$ para a primeira iteração, como pedido. O processo completo convergiria após cerca de $4$ iterações, conforme os dados de verificação, com um custo médio ótimo esperado de $2.113/h$, muito menor que esses valores iniciais, indicando a necessidade de otimizar $t$ e as ações nas próximas iterações.

>Neste exemplo, fixamos $t=500h$, mas o modelo exige testar todos os $t \in [500h,20.000h]$ discretizados. Talvez possamos fazer isso, mais tarde, usando C++ 23.

## Solução da Questão 4: Solução por Iteração de Política

Vamos resolver a Questão 4, que exige o uso do método de iteração de política para determinar a política ótima $\pi^{*}(s)$ no problema de manutenção de turbinas eólicas. Este método alterna entre avaliação de política e melhoria de política até convergir para a solução ótima, considerando ações e intervalos de inspeção.

###  Definir uma Política Inicial

No método de iteração de política, começamos com uma política inicial arbitrária $\pi_0(s)$, que mapeia cada estado $s$ para uma ação $a \in A_s$ e um intervalo de inspeção $t \in T_s$. Para simplificar, escolhemos uma política conservadora que evita custos altos iniciais:

- Para $s = 1$: $\pi_0(1) = (NA, 500h)$ (nenhuma ação, inspeção frequente);
- Para $s = 2, 3, 4, 5$: $\pi_0(s) = (MM, 500h)$ (manutenção menor, inspeção frequente);
- Para $s = 6, 7$: $\pi_0(s) = (CM, 500h)$ (manutenção corretiva, inspeção após reparo).

Escolhemos $t = 500h$ (o intervalo mínimo) para garantir observações frequentes no início. A tabela da política inicial será dada por:

| Estado | Ação | Intervalo de Inspeção |
|--------|------|-----------------------|
| 1      | NA   | 500h                  |
| 2      | MM   | 500h                  |
| 3      | MM   | 500h                  |
| 4      | MM   | 500h                  |
| 5      | MM   | 500h                  |
| 6      | CM   | 500h                  |
| 7      | CM   | 500h                  |

A sagaz leitora deve notar que esta é uma escolha inicial simples, e o método ajustará a política nas iterações seguintes.

### Realizar a Avaliação de Política

Na avaliação de política, calculamos os valores $V^{\pi_0}(s)$ para cada estado $s$ sob a política $\pi_0$, resolvendo o sistema de equações lineares derivado da equação de Bellman para uma política fixa:

$$
V^{\pi_0}(s) = R(s, \pi_0(s), t) + \gamma \sum_{s' \in S} P(s' \vert  s, \pi_0(s)) V^{\pi_0}(s')
$$

Nesta equação temos:

- $R(s, a, t)$ representa a recompensa, custo, definida na Questão 1;
- $\gamma = 0.9$ é o fator de desconto;
- $P(s' \vert  s, a)$ vem das matrizes de transição da Questão 1;
- $t = 500h$ conforme $\pi_0$.

Para simplificar, vamos calcular $V^{\pi_0}(1)$ como exemplo (estado perfeito):

- $\pi_0(1) = (NA, 500h)$;
- $R(1, NA, 500) = C_{insp}(1) + C_{ação}(NA) + \sum_{s' \in S} P(s' \vert  1, NA) C_{falha}(s') + C_{estado}(1) \cdot 500$;
  - $C_{insp}(1) = 200$;
  - $C_{ação}(NA) = 0$;
  - $\sum_{s' \in S} P(s' \vert  1, NA) C_{falha}(s') = 0.9961 \cdot 0 + 0.0012 \cdot 0 + 0.0027 \cdot 180000 = 486$ (ver Questão 3);
  - $C_{estado}(1) \cdot 500 = 1000 \cdot 500 = 500000$;
  - $R(1, NA, 500) = 200 + 0 + 486 + 500000 = 500686$.

A equação torna-se:

$$
V^{\pi_0}(1) = 500686 + 0.9 \left[ 0.9961 V^{\pi_0}(1) + 0.0012 V^{\pi_0}(2) + 0.0027 V^{\pi_0}(7) \right]
$$

Para resolver, precisamos de $V^{\pi_0}(s)$ para todos os estados, formando um sistema de 7 equações. Na prática, usamos técnicas numéricas, mas para fins didáticos, isolamos $V^{\pi_0}(1)$ assumindo valores iniciais, por exemplo: $V^{\pi_0}(s) = 0$ para $s \neq 1$ como aproximação inicial:

$$
V^{\pi_0}(1) = 500686 + 0.9 \cdot 0.9961 V^{\pi_0}(1)
$$

$$
V^{\pi_0}(1) - 0.89649 V^{\pi_0}(1) = 500686
$$

$$
0.10351 V^{\pi_0}(1) = 500686
$$

$$
V^{\pi_0}(1) = \frac{500686}{0.10351} \approx 4837106
$$

Esse valor é aproximado e alto devido à simplificação (ignoramos $V^{\pi_0}(2)$ e $V^{\pi_0}(7)$). Na avaliação completa, resolveríamos o sistema inteiro.

### Executar a Melhoria de Política

Na melhoria de política, atualizamos $\pi_1(s)$ escolhendo a ação e intervalo que minimizam o custo esperado com base nos valores $V^{\pi_0}(s)$:

$$
\pi_1(s) = \arg\min_{a \in A_s, t \in T_s} \left[ R(s, a, t) + \gamma \sum_{s' \in S} P(s' \vert  s, a) V^{\pi_0}(s') \right]
$$

Para $s=1$, testamos $NA$ com diferentes $t$, por exemplo: $500h$, $2000h$:

- $t = 500h$: $R(1, NA, 500) + 0.9 \cdot [0.9961 \cdot 4837106 + 0] \approx 500686 + 4333955 = 4834641$;
- $t = 2000h$: $R(1, NA, 2000) = 200 + 486 + 1000 \cdot 2000 = 2000686$;
  - $2000686 + 0.9 \cdot 0.9961 \cdot 4837106 \approx 2000686 + 4333955 = 6334641$.

O custo com $t = 500h$ é menor, então $\pi_1(1) = (NA, 500h)$ inicialmente. Para $s=2$, testaríamos $NA$, $MM$, $PM$ com vários $t$, escolhendo o menor custo. Repetimos para todos os estados.

### Comparação da Convergência com o Caso do Grid World

1. **No Grid World**, a iteração de política converge rapidamente (geralmente em poucas iterações) devido a:

- **Ambiente Simples**: estados limitados, por exemplo: $12$; ações determinísticas, movimentos; recompensas fixas, $-1$ por passo, $+1$ ou $-1$ nos terminais;
- **Observabilidade Total**: o agente sabe seu estado a cada passo, sem intervalos de inspeção;
- **Horizonte Finito**: termina ao alcançar estados terminais.

2. **Nesse exercício**:

- **Complexidade Maior**: $7$ estados, mas ações com probabilidades, por exemplo: $MM$: 0.7 para $1$, $0.2$ para $2$, $0.1$ para $3$ e otimização de $t$, com $20$ opções;
- **Observabilidade Parcial**: intervalos de inspeção $t$ aumentam o espaço de decisão;
- **Horizonte Infinito**: usamos desconto ($\gamma = 0.9$), requerendo mais iterações para estabilizar custos altos.

Os dados de verificação indicam convergência em $4$ iterações com custo médio de $2.113/h$, sugerindo que, apesar da complexidade, o problema converge em um número similar ao **Grid World** para políticas iniciais razoáveis. A incerteza, definida por falhas que seguem a distribuição de Poisson, e custos variáveis tornam a convergência mais sensível à política inicial.

A dedicada leitora deve perceber que a iteração de política é mais trabalhosa neste caso, mas oferece uma abordagem próxima a realidade prática para otimizar manutenção, ajustando ações e intervalos dinamicamente.

## Solução da Questão 5: Análise de Resultados

Vamos resolver a Questão 5, que exige a análise da política ótima encontrada, comparando-a com estratégias fixas, avaliando a influência dos custos e examinando o impacto das falhas aleatórias em relação ao **Grid World**.

### Compare a Política Ótima Encontrada com Manutenção Preventiva a Cada 2.500h (6.38/h) e 5.000h (4.84/h)

Para realizar essa comparação, usamos os dados de verificação fornecidos: o custo médio ótimo esperado da política encontrada por MDP é 2.113/h, obtido após cerca de 4 iterações (ver Questões 3 e 4). As estratégias fixas são:

- **Manutenção preventiva a cada 2.500h**: Custo médio de 6.38/h;
- **Manutenção preventiva a cada 5.000h**: Custo médio de 4.84/h.

#### Comparação Numérica
- **Política Ótima (MDP)**: $2.113/h$;
- **PM a cada 2.500h**: $6.38/h$;
  - Diferença: $6.38 - 2.113 = 4.267/h$;
  - Redução percentual: $\frac{4.267}{6.38} \approx 66.9\%$;
- **PM a cada 5.000h**: $4.84/h$;
  - Diferença: $4.84 - 2.113 = 2.727/h$;
  - Redução percentual: $\frac{2.727}{4.84} \approx 56.3\%$.

#### Interpretação
A política ótima reduz significativamente os custos em comparação com as estratégias fixas:

- **Versus 2.500h**: uma redução de 66.9% mostra que inspeções e manutenções frequentes (a cada 2.500h) geram custos altos desnecessários, possivelmente devido a intervenções excessivas em estados de baixo desgaste.
- **Versus 5.000h**: Uma redução de 56.3% indica que intervalos mais longos diminuem os custos de intervenção, mas ainda não otimizam tão bem quanto o MDP, que ajusta ações e intervalos dinamicamente.

A política ótima, conforme o artigo[^1], usa uma combinação de $NA$, $MM$, $PM$, e $CM$ com intervalos variáveis, por exemplo: $10685h$ para $s=1$, 10815h para $s=2$, evitando manutenções desnecessárias e minimizando o impacto de falhas.

### Analise Como os Diferentes Custos Influenciam a Política Ótima

Os custos definidos no modelo afetam diretamente a política ótima. Vamos analisar cada componente:

- **Custo de Inspeção (200)**:

  - baixo em relação a outros custos, incentivando inspeções frequentes em estados iniciais para evitar falhas não detectadas. Um aumento para 1000, por exemplo, poderia levar a intervalos maiores ($t$), reduzindo inspeções.

- **Custo de Manutenção Menor ($MM$, 3.000)**:
  
  - moderado, tornando $MM$ atrativo em estados intermediários. Se fosse mais caro, a política favoreceria $NA$ ou $PM$.

- **Custo de Manutenção Preventiva ($PM$, 7.500)**:

  - mais alto que $MM$, mas eficaz para estados avançados, pois retorna ao estado perfeito com alta probabilidade, $0.9$. Um custo menor, por exemplo: $5.000$, poderia aumentar seu uso em estados anteriores.
  
- **Custo de Manutenção Corretiva ($CM$, 150.000)**:

  - extremamente alto, restrito a estados de falha. Uma redução poderia não alterar a política, pois $CM$ é inevitável após falhas.

- **Custo de Falha (180.000)**:

  - alto, penalizando transições para $s=6$ ou $s=7$. Um custo menor aumentaria a tolerância a falhas, possivelmente favorecendo $NA$ por mais tempo.

- **Custo de Estado Não Detectado (1.000/h)**:

  - domina os custos totais para intervalos longos. Se reduzido, intervalos maiores seriam preferidos, alterando $t$ na política ótima.

#### Exemplo Prático

Para $s=2$, com $\pi^{*}(2) = (MM, 10815h)$ (baseado no artigo[^1]):

- $R(2, MM, 10815) = 200 + 3000 + 0 + 1000 \cdot 10815 = 10818200$;
- Um custo de estado menor (100/h) reduziria $R$ para 1083200, possivelmente tornando $t = 20000h$ mais atrativo.

A política ótima equilibra esses custos, escolhendo ações e $t$ que minimizam o impacto de falhas caras enquanto controlam os custos de intervenção e inspeção.

### Como a Inclusão de Falhas Aleatórias Afeta a Solução Comparada ao Grid World?

No **Grid World**, a solução é deterministicamente previsível. Não existem falhas aleatórias. Transições dependem apenas da ação. Por exemplo: mover norte leva ao estado ao norte com probabilidade $1$, exceto por limites. Existe um conjunto de recompensas fixas, à saber: $-1$ por passo, $+1$ ou $-1$ nos terminais. Finalmente, no **Grid World** contamos com um *horizonte finito*. A análise termina ao atingir estados terminais, sem incertezas contínuas.

Neste problema de turbinas, começamos com falhas aleatória,  distribuição de Poisson, $\lambda_0 = 0.0027$. Existe uma probabilidade de transição para $s=7$ (falha aleatória) em todos os estados não terminais, independentemente da ação. Isso adiciona risco constante, ausente no **Grid World**. Também enfrentamos um impacto nos custos: o custo de falha ($180.000$) associado a $s=7$ aumenta o valor esperado de $R(s,a,t)$. No **Grid World**, penalidades são pequenas e previsíveis.

>Poderíamos estudar $\lambda_0$ com mais cuidado. Estados de maior desgaste, podem ter $\lambda_0$ maior, isso traria o modelo mais próximo da realidade. Contudo, para **MDP**, essa simplificação parece aceitável. Ainda assim, aqui temos outro ponto de melhoria.

Outra diferença importante entre este problema e o **Grid World** está na observabilidade parcial. As falhas podem ocorrer entre inspeções, exigindo otimização de $t$. No **Grid World**, o agente sabe seu estado a cada passo, eliminando essa incerteza.

Finalmente, no problema descrito no artigo[^1], o horizonte de análise tende ao infinito. O desconto ($\gamma = 0.9$) reflete operação contínua, contrastando com o fim abrupto do **Grid World**. Falhas aleatórias acumulam custos ao longo do tempo.

#### Efeito na Solução

Três pontos precisam ser levados em consideração:

1. **Complexidade**: a política deve balancear o risco de falhas aleatórias (levando a $CM$) contra custos de intervenção ($MM$, $PM$). No **Grid World**, a política simplesmente evita penalidades ($-1$) até atingir a recompensa ($+1$).

2. **Flexibilidade**: intervalos $t$ ajustáveis mitigam falhas inesperadas, enquanto no **Grid World** o agente segue um caminho fixo.

3. **Convergência**: a incerteza das falhas pode exigir mais iterações para estabilizar.

A curiosa leitora deve perceber que as falhas aleatórias transformam o problema em um desafio próximo da realidade, exigindo uma política que antecipe eventos imprevisíveis, algo que o ambiente controlado do **Grid World** não precisa, ou consegue, enfrentar.

## Finalmente, os Finalmentes

A esforçada leitora deveria, usando [C++](https://github.com/Svalorzen/AI-Toolbox), ou [Python](https://pymdptoolbox.readthedocs.io/en/latest/), montar este modelo e refazer estes cálculos. Principalmente, nos pontos que, ao longo do texto, fui marcando como interessante.
