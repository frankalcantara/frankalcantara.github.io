---
layout: post
title: Novos Data Centers no Horizonte
author: "Frank "
categories:
    - artigo
    - opinião
tags:
    - Física
    - inteligência artificial
    - opinião
    - Engenharia
rating: 6
description: A Nvidia fez uma proposta pela criação de um novo padrão de distribuição de energia para *data centers*. Este texto analisa essa proposta
date: 2025-11-14T00:58:10.955Z
preview: |
    # texto técnico expandido  
    *arquiteturas de distribuição de energia em *data centers* de ia: análise elétrica, térmica, econômica e formalização acadêmica da transição 54 vdc → 800 vdc*
lastmod: 2025-11-14T05:15:14.648Z
published: false
draft: 2025-11-14T01:11:49.434Z
---

A rápida evolução das tecnologias de treinamento e aceleração para assistentes de inteligência artificial (IA) transformou os *data centers* em sistemas cujo comportamento energético lembra mais instalações industriais de alta densidade do que as salas de servidores que povoam nosso imaginário.

Espera-se que o mercado de servidores de IA comece em **US$ 124,8 bilhões em 2024 e salte para US$ 204,7 bilhões em 2025. Até 2030, prevê-se que seja de US$ 837 bilhões, com algumas estimativas excedendo US$ 1,56 trilhão**. Esse número de 34~39% não é apenas uma estatística; representa uma mudança de paradigma em como a humanidade começou a ver a inteligência como um recurso computacional. A Nvídia lidera este mercado com alguma coisa entre 80% e 94% de participação, e sua influência é tamanha que suas decisões moldam o futuro da infraestrutura de IA.

Além do Cuda, a barreira quase intransponível construída nos últimos 20 ou 30 anos, há a busca constante de performance, eficiência e escalabilidade. 

O [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) integra 36 Superchips, ou 72 GPUs e 36 CPUs, em um único sistema de refrigeração líquida em escala de rack. Com 208 bilhões de transistores, este monstro possui **1.440 petaflops de desempenho com precisão FP4**. A velocidade de inferência 30x mais rápida e a eficiência energética 25x melhorada em comparação com a geração anterior do H100 não foram apenas melhorias de desempenho, mas uma reescrita dos limites físicos. Uma besta faminta por energia, capaz de consumir até 6 megawatts em um único rack.

A [Goldman Sachs Research](https://www.wired.com/story/data-center-ai-boom-us-economy-jobs/) projeta que a demanda global de energia dos data centers saltará 165% até 2030, de 55GW em 2023 para 145GW, com picos intermediários de 92GW em 2027. A IA, que hoje responde por 14% desse consumo, alcançará 27% em 2027 e 36% em 2030 — podendo devorar mais de 40% dos 96GW críticos previstos para 2026. Nos EUA, a fatia dos data centers na demanda elétrica nacional mais que dobrará, de 4% para acima de 8%.

Esse crescimento não é apenas estatístico: é o maior gargalo físico do ecossistema de IA. Cerca de 60% da nova demanda exigirá geração adicional, demandando US$ 720 bilhões em investimentos em transmissão até 2030. A matriz energética será híbrida: 30% ciclo combinado a gás, 30% picos a gás, 27,5% solar e 12,5% eólica — um equilíbrio entre urgência e transição.

A vantagem competitiva se concentrará em FLOPS por watt e resfriamento eficiente. A Nvidia alega que o Blackwell melhora a eficiência energética em 25 vezes; a Super Micro domina resfriamento líquido. O mercado se dividirá: data centers AI-Ready, com energia massiva e circuitos refrigerados, prosperarão; instalações legadas, presas ao ar condicionado, tornar-se-ão relíquias.

Em resumo, o futuro da IA não depende apenas de parâmetros, mas de joules. Quem dominar a termodinâmica da computação liderará a próxima década.

Todo cuidado é pouco quando tentamos prever o futuro. A [McKinsey](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/investing-in-the-rising-data-center-economy) estima que serão necessários US$ 5,2 trilhões em investimentos de capital (CAPEX) até 2030 para atender à demanda de 'computação' dos modelos de IA. Por outro lado, o comitê de investimento global do Morgan Stanley alertou que o atual boom de CAPEX em IA e o mercado de ações que ele sustenta podem estar mais perto do fim estrondoso do que imagina nossa vã imaginação. Embora a necessidade a longo prazo seja clara, existe uma visão cautelosa de que as avaliações de mercado e a velocidade dos gastos podem ter entrado em uma fase de bolha de curto prazo ou, no mínimo, em um período de ajuste e revisão.

## Para Novas Demandas, Novas Tecnologias

Em 2025, a Nvidia propôs uma nova arquitetura de distribuição de energia para *data centers* de IA, denominada [Kyber](https://developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-generation-of-ai-factories/), que utiliza barramentos de 800 VDC em vez dos tradicionais 48–54 VDC. Esta mudança não é trivial; ela reflete a necessidade de lidar com cargas cada vez maiores e os desafios associados à eficiência energética e à dissipação térmica.

Em 2025, dominar a relação entre potência, tensão, corrente, perdas e eficiência tornou-se indispensável, sobretudo porque limites elétricos e térmicos agora restringem tanto quanto limites de interconexão ou memória.

## Para Entender o Fluxo de Energia

O fluxo de energia de um data center moderno pode ser representado com mais rigor usando uma notação que destaca tensões, correntes, perdas e a função de cada estágio. o diagrama abaixo enfatiza a transformação e retificação desde a entrada em média tensão até o a tensão final diretamente na gpu.

[][][][][][][][][][][][][]

Um fluxo semelhante será reapresentado mais adiante para a arquitetura de 800 vdc, permitindo comparação direta.

## Equações detalhadas de cada estágio de conversão
cada estágio de conversão tem uma eficiência associada. chamaremos $ \eta_i $ a eficiência do estágio $ i $. se houver $ n $ estágios, a eficiência total é:
$$ \eta_{\text{total}} = \prod_{i=1}^{n} \eta_i $$
### 3.1 psu (480 vac → 54 vdc)
supor eficiência de 96 por cento:
$$ \eta_{\text{psu}} = 0.96 $$
### 3.2 conversão primária (54 v → 12 v)
eficiência típica de 97 por cento:
$$ \eta_{\text{prim}} = 0.97 $$
### 3.3 conversão secundária (12 v → 5 v)
eficiência próxima de 95 por cento:
$$ \eta_{\text{sec}} = 0.95 $$
### 3.4 pol (5 v → 1 v)
eficiência de 92 por cento:
$$ \eta_{\text{pol}} = 0.92 $$
### 3.5 eficiência total do sistema tradicional
substituindo:
$$ \eta_{\text{total}} = 0.96 \times 0.97 \times 0.95 \times 0.92 $$
$$ \eta_{\text{total}} \approx 0.81 $$
isso significa que, de 120 kw entregues ao rack:
$$ P_{\text{útil}} = 120000 \times 0.81 = 97200\ \text{W} $$
as perdas ficam em:
$$ P_{\text{perda}} = 120000 - 97200 = 22800\ \text{W} $$
mais de 22 kw que se convertem diretamente em calor antes de chegar às gpus.

## 4 análise térmica completa das perdas e projeções por rack
a dissipação térmica total de um rack se decompõe em três parcelas:
1. calor gerado nos chips (gpus e cpus)  
2. calor gerado nos estágios de conversão  
3. calor gerado por perdas resistivas em cabos, conectores e barramentos  
a potência térmica é aproximadamente igual à potência perdida:
$$ Q_{\text{rack}} \approx P_{\text{perdido}} $$
### 4.1 sistema tradicional 54 vdc
considerando o exemplo anterior:
$$ Q_{\text{rack,54V}} \approx 22800\ \text{W} $$
em racks de 150 kw, as perdas totais sobem facilmente para 25–30 kw. racks de 30 kw a mais de perdas térmicas exigem sistemas de refrigeração mais robustos, que consomem mais energia elétrica, elevando o pue (power usage effectiveness).
### 4.2 sistema 800 vdc
a arquitetura kyber reduz drasticamente o número de estágios e melhora a eficiência, levando a perdas totais inferiores a 6 por cento em cenários reais. suponha um rack de 600 kw:
$$ \eta_{\text{kyber}} \approx 0.94 $$
$$ P_{\text{perda}} = 600000 \times (1 - 0.94) = 36000\ \text{W} $$
embora 36 kw pareçam altos, o rack fornece cinco vezes mais computação que o rack de 120 kw do exemplo anterior. comparando perdas relativas:
- sistema 54 v: perdas ≈ 20%  
- sistema 800 v: perdas ≈ 6%  
assim, a dissipação térmica por watt de computação cai substancialmente.

# parte ii: comparação completa de tco entre 54 vdc e 800 vdc
o custo total de propriedade de um data center é composto por:
1. custo do cobre e conectores  
2. custo das psus  
3. custo térmico (capex em chillers, opex em energia)  
4. perdas elétricas (energia não convertida em computação)  
5. manutenção  
6. espaço físico estrutural  
7. custo de engenharia associado à complexidade  
analisemos cada item.
## 5 impacto no tco: análise aprofundada
### 5.1 custo de cobre
- sistema 54 vdc utiliza barras e cabos grossos, frequentemente acima de 100 mm² por fase  
- sistema 800 vdc reduz a largura dos condutores em até 80–90 por cento  
o custo de cobre pode diminuir em milhares de dólares por rack por ano quando considerado o ciclo de substituição e manutenção.
### 5.2 custo das psus
psus tradicionais são caras, volumosas e apresentam taxas de falha que exigem redundância. no modelo kyber:
- psus desaparecem do rack  
- a conversão é centralizada em módulos altamente eficientes  
- a redundância fica mais simples e barata
### 5.3 custo térmico
o q adicional gerado por perdas se relaciona ao consumo de refrigeração. aproximando que o resfriamento consome 0.3 a 0.5 w para cada watt de calor gerado, a diferença entre 20 por cento e 6 por cento de perdas resulta em grandes economias operacionais.
### 5.4 custo das perdas energéticas
em um cluster de 20 mw:
- perdas de 20 por cento equivalem a 4 mw  
- perdas de 6 por cento equivalem a 1.2 mw  
a diferença de 2.8 mw pode representar economia de milhões de dólares por ano em energia.
### 5.5 manutenção
manter dezenas de psus por rack é caro e arriscado. centralizar conversão simplifica substituições e reduz o mtbf global.
### 5.6 ocupação física
liberar 20–30 por cento do volume de um rack significa acomodar mais gpus por metro quadrado, aumentando densidade computacional sem ampliar área física.

# parte iii: versão em estilo de artigo acadêmico / capítulo de livro
## 6 formalização acadêmica: introdução
este capítulo examina, sob perspectiva de engenharia elétrica e ciência da computação de alto desempenho, a transição estrutural entre duas arquiteturas de distribuição de energia para *data centers*: o modelo consolidado baseado em 48–54 vdc e a arquitetura emergente que utiliza barramentos de 800 vdc. essa transição é impulsionada principalmente pelas exigências energéticas crescentes dos aceleradores de ia, cujo consumo por unidade cresce a taxas superiores às melhorias de eficiência elétrica.
## 7 discussão metodológica
as análises são conduzidas por meio de modelagem elétrica convencional, incluindo cálculo de correntes, perdas resistivas e eficiência de conversão, bem como estimativas térmicas derivadas da equivalência entre potência perdida e calor gerado. por fim, inclui-se um componente econômico, considerando o ciclo de vida da infraestrutura.
## 8 resultados
as deduções mostram que o sistema 48–54 vdc, embora adequado até cerca de 120–150 kw por rack, falha em escalabilidade acima dessa faixa. correntes elevadas, perdas resistivas significativas e complexidade estrutural convergem para um limite físico. já a topologia de 800 vdc reduz correntes, simplifica conectores, melhora a eficiência e permite densidades superiores a 600 kw por rack.
## 9 conclusão acadêmica
a transição para alta tensão dc não é mera evolução incremental, mas uma mudança paradigmática necessária para acompanhar o crescimento das cargas de ia. os resultados indicam que sistemas de 800 vdc serão predominantes até o fim da década de 2020, com possibilidades de expansão para tensões ainda mais elevadas conforme a indústria exige densidades de potência acima de 1 mw por rack.

# 10 conclusão unificada
a análise detalhada dos aspectos elétricos, térmicos, econômicos e conceituais evidencia que a migração de infraestruturas de energia de 54 vdc para 800 vdc é tecnicamente inevitável. a nova arquitetura não apenas reduz perdas e simplifica o design físico, mas também sustenta a escalabilidade exigida pela próxima geração de aceleradores. ao integrar alta tensão, conversão centralizada e dispositivos avançados baseados em sic e gan, os *data centers* que adotam esse modelo posicionam-se para uma década de crescimento energético extremo sem comprometer eficiência ou capacidade térmica. trata-se de uma transformação estrutural que acompanha organicamente a evolução da computação de ia em direção a clusters de potência cada vez maior.
ords: |-
datacenters
    Inteligência Artificial
    Nvidia
    Engenharia Elétrica
draft: 2025-11-14T00:58:49.458Z
lastmod: 2025-11-14T00:58:53.988Z
toc: true


## 1 introdução geral

a rápida evolução dos aceleradores de ia transformou os *data centers* em sistemas cujo comportamento energético lembra mais instalações industriais de alta densidade do que salas de servidores tradicionais. em 2025, dominar a relação entre potência, tensão, corrente, perdas e eficiência tornou-se indispensável, sobretudo porque limites elétricos e térmicos agora restringem tanto quanto limites de interconexão ou memória. este texto expande a análise feita anteriormente e acrescenta diagramas técnicos detalhados, deduções matemáticas de eficiência, estimativas térmicas rigorosas, comparações completas de tco e uma reformulação em estilo de artigo acadêmico ou capítulo de livro.



# parte i: visão técnica aprofundada do fluxo de energia

## 2 cadeia energética completa com diagramas técnicos adicionais

o fluxo de energia de um data center moderno pode ser representado com mais rigor usando uma notação que destaca tensões, correntes, perdas e a função de cada estágio. o diagrama abaixo enfatiza a transformação e retificação desde a entrada em média tensão até o pol final no package da gpu.

grid (11–33 kv ac)
↓ transformadores (step-down)
480 vac trifásico
↓ ups (baterias, flywheel, estabilização)
480 vac limpo
↓ pdu
alimentação dos racks
↓ psus (ac → 48–54 vdc)
barramento dc
↓ conversores intermediários (54 v → 12 v)
alimentação de placas
↓ conversores locais (12 v → 5 v / 3.3 v)
alimentação dos módulos
↓ pols (5 v → 0.8–1 v)
tensões finais para gpus/cpus

yaml
Copiar código

um fluxo semelhante será reapresentado mais adiante para a arquitetura de 800 vdc, permitindo comparação direta.



## 3 equações detalhadas de cada estágio de conversão

cada estágio de conversão tem uma eficiência associada. chamaremos $ \eta_i $ a eficiência do estágio $ i $. se houver $ n $ estágios, a eficiência total é:

$$ \eta_{\text{total}} = \prod_{i=1}^{n} \eta_i $$

### 3.1 psu (480 vac → 54 vdc)
supor eficiência de 96 por cento:

$$ \eta_{\text{psu}} = 0.96 $$

### 3.2 conversão primária (54 v → 12 v)
eficiência típica de 97 por cento:

$$ \eta_{\text{prim}} = 0.97 $$

### 3.3 conversão secundária (12 v → 5 v)
eficiência próxima de 95 por cento:

$$ \eta_{\text{sec}} = 0.95 $$

### 3.4 pol (5 v → 1 v)
eficiência de 92 por cento:

$$ \eta_{\text{pol}} = 0.92 $$

### 3.5 eficiência total do sistema tradicional
substituindo:

$$ \eta_{\text{total}} = 0.96 \times 0.97 \times 0.95 \times 0.92 $$

$$ \eta_{\text{total}} \approx 0.81 $$

isso significa que, de 120 kw entregues ao rack:

$$ P_{\text{útil}} = 120000 \times 0.81 = 97200\ \text{W} $$

as perdas ficam em:

$$ P_{\text{perda}} = 120000 - 97200 = 22800\ \text{W} $$

mais de 22 kw que se convertem diretamente em calor antes de chegar às gpus.



## 4 análise térmica completa das perdas e projeções por rack

a dissipação térmica total de um rack se decompõe em três parcelas:

1. calor gerado nos chips (gpus e cpus)  
2. calor gerado nos estágios de conversão  
3. calor gerado por perdas resistivas em cabos, conectores e barramentos  

a potência térmica é aproximadamente igual à potência perdida:

$$ Q_{\text{rack}} \approx P_{\text{perdido}} $$

### 4.1 sistema tradicional 54 vdc

considerando o exemplo anterior:

$$ Q_{\text{rack,54V}} \approx 22800\ \text{W} $$

em racks de 150 kw, as perdas totais sobem facilmente para 25–30 kw. racks de 30 kw a mais de perdas térmicas exigem sistemas de refrigeração mais robustos, que consomem mais energia elétrica, elevando o pue (power usage effectiveness).

### 4.2 sistema 800 vdc

a arquitetura kyber reduz drasticamente o número de estágios e melhora a eficiência, levando a perdas totais inferiores a 6 por cento em cenários reais. suponha um rack de 600 kw:

$$ \eta_{\text{kyber}} \approx 0.94 $$

$$ P_{\text{perda}} = 600000 \times (1 - 0.94) = 36000\ \text{W} $$

embora 36 kw pareçam altos, o rack fornece cinco vezes mais computação que o rack de 120 kw do exemplo anterior. comparando perdas relativas:

- sistema 54 v: perdas ≈ 20%  
- sistema 800 v: perdas ≈ 6%  

assim, a dissipação térmica por watt de computação cai substancialmente.



# parte ii: comparação completa de tco entre 54 vdc e 800 vdc

o custo total de propriedade de um data center é composto por:

1. custo do cobre e conectores  
2. custo das psus  
3. custo térmico (capex em chillers, opex em energia)  
4. perdas elétricas (energia não convertida em computação)  
5. manutenção  
6. espaço físico estrutural  
7. custo de engenharia associado à complexidade  

analisemos cada item.

## 5 impacto no tco: análise aprofundada

### 5.1 custo de cobre

- sistema 54 vdc utiliza barras e cabos grossos, frequentemente acima de 100 mm² por fase  
- sistema 800 vdc reduz a largura dos condutores em até 80–90 por cento  

o custo de cobre pode diminuir em milhares de dólares por rack por ano quando considerado o ciclo de substituição e manutenção.

### 5.2 custo das psus

psus tradicionais são caras, volumosas e apresentam taxas de falha que exigem redundância. no modelo kyber:

- psus desaparecem do rack  
- a conversão é centralizada em módulos altamente eficientes  
- a redundância fica mais simples e barata

### 5.3 custo térmico

o q adicional gerado por perdas se relaciona ao consumo de refrigeração. aproximando que o resfriamento consome 0.3 a 0.5 w para cada watt de calor gerado, a diferença entre 20 por cento e 6 por cento de perdas resulta em grandes economias operacionais.

### 5.4 custo das perdas energéticas

em um cluster de 20 mw:

- perdas de 20 por cento equivalem a 4 mw  
- perdas de 6 por cento equivalem a 1.2 mw  

a diferença de 2.8 mw pode representar economia de milhões de dólares por ano em energia.

### 5.5 manutenção

manter dezenas de psus por rack é caro e arriscado. centralizar conversão simplifica substituições e reduz o mtbf global.

### 5.6 ocupação física

liberar 20–30 por cento do volume de um rack significa acomodar mais gpus por metro quadrado, aumentando densidade computacional sem ampliar área física.



# parte iii: versão em estilo de artigo acadêmico / capítulo de livro

## 6 formalização acadêmica: introdução

este capítulo examina, sob perspectiva de engenharia elétrica e ciência da computação de alto desempenho, a transição estrutural entre duas arquiteturas de distribuição de energia para *data centers*: o modelo consolidado baseado em 48–54 vdc e a arquitetura emergente que utiliza barramentos de 800 vdc. essa transição é impulsionada principalmente pelas exigências energéticas crescentes dos aceleradores de ia, cujo consumo por unidade cresce a taxas superiores às melhorias de eficiência elétrica.

## 7 discussão metodológica

as análises são conduzidas por meio de modelagem elétrica convencional, incluindo cálculo de correntes, perdas resistivas e eficiência de conversão, bem como estimativas térmicas derivadas da equivalência entre potência perdida e calor gerado. por fim, inclui-se um componente econômico, considerando o ciclo de vida da infraestrutura.

## 8 resultados

as deduções mostram que o sistema 48–54 vdc, embora adequado até cerca de 120–150 kw por rack, falha em escalabilidade acima dessa faixa. correntes elevadas, perdas resistivas significativas e complexidade estrutural convergem para um limite físico. já a topologia de 800 vdc reduz correntes, simplifica conectores, melhora a eficiência e permite densidades superiores a 600 kw por rack.

## 9 conclusão acadêmica

a transição para alta tensão dc não é mera evolução incremental, mas uma mudança paradigmática necessária para acompanhar o crescimento das cargas de ia. os resultados indicam que sistemas de 800 vdc serão predominantes até o fim da década de 2020, com possibilidades de expansão para tensões ainda mais elevadas conforme a indústria exige densidades de potência acima de 1 mw por rack.



# 10 conclusão unificada

a análise detalhada dos aspectos elétricos, térmicos, econômicos e conceituais evidencia que a migração de infraestruturas de energia de 54 vdc para 800 vdc é tecnicamente inevitável. a nova arquitetura não apenas reduz perdas e simplifica o design físico, mas também sustenta a escalabilidade exigida pela próxima geração de aceleradores. ao integrar alta tensão, conversão centralizada e dispositivos avançados baseados em sic e gan, os *data centers* que adotam esse modelo posicionam-se para uma década de crescimento energético extremo sem comprometer eficiência ou capacidade térmica. trata-se de uma transformação estrutural que acompanha organicamente a evolução da computação de ia em direção a clusters de potência cada vez maior.
