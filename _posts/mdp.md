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
lastmod: 2025-02-12T22:14:35.909Z
---

A curiosa leitora deve estar decepcionada. Quando comecei esta jornada eu prometi que estudaríamos Reinforcement Learning e, até agora, nada. Só o básico fundamental. Peço desculpas.

Eu estou me divertindo enquanto escrevo e ainda descubro dúvidas que nem sabia que tinha. Analisando os artigos anteriores: [1](https://frankalcantara.com/reinforcement-learning-hist%C3%B3ria/), [2](https://frankalcantara.com/entendendo-mdp/). [3](https://frankalcantara.com/um-mundo-em-uma-grade/) e [4](https://frankalcantara.com/resolvendo-o-grid-world/) percebi que o texto carece de aplicações práticas.

Pesquisando, encontrei alguns casos interessantes de aplicações de **MDP** em situações da vida real que foram acadêmica e cientificamente descritos em artigos publicados. Escolhi aqueles que continham mais dados e fossem compatíveis com os meus interesses pessoais. A compadecida leitora há de perdoar este pobre autor egoísta.

## Manutenção Ótima de Turbinas Eólicas

Na indústria, a Manutenção Preditiva Industrial busca otimizar a manutenção de equipamentos, antecipando falhas e minimizando o tempo de inatividade. Em um cenário de **MDP**, os estados representam os diferentes níveis de desgaste de um equipamento. As ações incluem realizar manutenção, continuar a operação ou substituir uma peça. As recompensas são medidas em termos do balanço entre os custos de manutenção e o tempo de operação contínua. Um exemplo emblemático é a **General Electric (GE)**, que utiliza **MDPs** para otimizar a manutenção de turbinas eólicas, considerando fatores como desgaste, condições climáticas e demanda de energia. 

[^1]:WU, Yan-Ru; ZHAO, Hong-Shan. **Optimization Maintenance of Wind Turbines Using Markov Decision Processes**. Disponível em: https://www.researchgate.net/publication/241177157_Optimization_maintenance_of_wind_turbines_using_Markov_decision_processes. Acesso em: 12 fev. 2025.

Este sistema permite à **GE** realizar manutenções de forma mais eficiente, prolongando a vida útil das turbinas e reduzindo custos operacionais.


# Exercício: Manutenção Ótima de Turbinas Eólicas via MDP

## Contexto do Problema
Uma empresa de energia eólica busca otimizar sua política de manutenção de caixas de engrenagem (gearboxes) usando Processos de Decisão de Markov (MDP). Similar ao Grid World, onde um agente navega em uma grade buscando maximizar recompensas, aqui temos um sistema que "navega" entre estados de deterioração buscando minimizar custos.

## Formulação MDP

### Estados ($S$)
O sistema possui 7 estados (comparado aos 12 estados do Grid World):

1. Estado perfeito (análogo ao estado inicial do Grid World)
2. Desgaste leve
3. Desgaste moderado
4. Desgaste avançado
5. Desgaste severo
6. Falha por deterioração (análogo ao estado terminal negativo)
7. Falha aleatória Poisson (segundo estado terminal negativo)

### Ações ($A$)
Em cada estado não-terminal, as seguintes ações estão disponíveis (análogo às quatro direções do Grid World):

- NA: Nenhuma ação
- MM: Manutenção menor (retorna ao estado anterior)
- PM: Manutenção preventiva (retorna ao estado perfeito)
- CM: Manutenção corretiva (após falha, retorna ao estado perfeito)

### Dinâmica do Sistema

#### Parâmetros Operacionais
- Turbina: 5MW, fator de capacidade 0.4 (2MW efetivos)
- Custo da energia: €0.5/kWh
- Horizonte de tempo: 100.000 horas
- Tempo de reparo após falha: 15 dias

#### Taxas de Transição
- Deterioração entre estados: $\lambda = 0.0012$ (1/833 dias)
- Falha aleatória: $\lambda_0 = 0.0027$ (1/ano)
- Fator de desconto: $\gamma = 0.9$ (análogo ao Grid World)

#### Probabilidades de Manutenção
Para MM em estados 2-5:
Estado       Prob. Transição
Anterior     0.7
-2 estados   0.2
-3 estados   0.1
Copy
Para PM em estados 2-5:
Estado       Prob. Transição
Perfeito     0.9
Estado 2     0.09
Estado 3     0.01
Copy
### Sistema de Recompensas (Custos)
Análogo ao sistema de recompensas do Grid World, mas com custos:

- Inspeção: €200 (análogo ao custo por passo do Grid World)
- Manutenção menor (MM): €3.000
- Manutenção preventiva (PM): €7.500
- Manutenção corretiva (CM): €150.000
- Perda por falha: €180.000 (análogo à penalidade do estado terminal negativo)
- Estado não detectado: €1.000/hora

### Intervalos de Inspeção
- Mínimo: 500 horas
- Máximo: 20.000 horas
- Discretizado em 20 intervalos iguais

## Questões

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