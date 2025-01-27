---
layout: post
title: "DeepSeek AI: A Revolução na Eficiência que Pode Abalar o Mercado"
author: Frank
categories:
    - artigo
tags:
    - algoritmos
    - linguagem natural
    - Matemática
    - inteligência artificial
image: ""
featured: false
rating: 5
description: "Tamanho não é documento na IA!  Com inovações disruptivas e um orçamento 20x menor, a DeepSeek desafia o status quo.  Prepare-se! "
date: 2025-01-27T13:43:25.513Z
preview: "DeepSeek AI prova: tamanho não é documento na IA!  Com inovações disruptivas e um orçamento 20x menor, eles estão desafiando o status quo.  Prepare-se para um mercado mais acessível e competitivo! "
keywords: inteligência artificial
toc: false
published: true
beforetoc: "Você já ouviu falar da DeepSeek AI? Uma equipe ENXUTA de menos de 200 gênios, desafiou as gigantes da tecnologia e fez o IMPOSSÍVEL: desenvolveram modelos de IA que RIVALIZAM e até SUPERAM o GPT-4 e o Claude... com um orçamento 20 VEZES MENOR! Isso mesmo, você não leu errado! Enquanto Google, OpenAI e companhia gastavam fortunas, a DeepSeek AI repensou a arquitetura tradicional, focando em EFICIÊNCIA, INOVAÇÃO e CÓDIGO ABERTO."
lastmod: 2025-01-27T14:54:40.549Z
slug: deepseek-ai-revolucao-na-eficiencia-pode-abalar-mercado
---

>Esta é uma republicação de um artigo escrito escrito em 25 de janeiro de 2025, no Linkedin.

O cenário da Inteligência Artificial está em constante ebulição, e a recente chegada da DeepSeek AI causou um verdadeiro terremoto. Com um investimento de apenas 557 mil dólares, o equivalente aproximado de 278,8 mil horas de uso de GPU’s H800, com custo padrão de _US$_2.00 por hora, eles conseguiram desenvolver modelos que rivalizam e até superam o GPT-4 e o Claude em diversas tarefas típicas de validação. Ainda que existam rumores de que este custo tenha sido de aproximada mente 5 milhões de dólares. Ainda representa 20 vezes menos que os modelos mais populares. Isso é levanta uma questão interessante: **como, valha me Deus, eles conseguiram isso?**

A equipe da DeepSeek AI repensou a arquitetura, agora tradicional e antiga, focando em estudo, otimização e eficiência. Enquanto a maior parte do mercado está aumentando os modelos e incluindo mais capacidade computacional no processo eles resolveram tentar processos diferentes:

1. **Quantização Inteligente:** em termos simples, a quantização pode ser entendida como arredondamento de números para facilitar os cálculos. Modelos de IA tradicionais (meu Deus, em menos de 3 anos os modelos ficaram tradicionais) geralmente usam números de ponto flutuante de 32 bits (FP32) para representar os pesos e ativações (os parâmetros que definem o modelo). Cada número FP32 é como um valor com alta precisão, similar a usar muitas casas decimais. A DeepSeek AI, por outro lado, utiliza majoritariamente números de ponto flutuante de **8 bits (FP8)**. Isso é análogo a usar menos casas decimais, tornando cada número menor do ponto de vista do armazenamento. Ou seja, no espaço de uma informação, nas arquiteturas tradicionais, a equipe da DeepSeek armazena três informações. Ao reduzir a precisão numérica dessa forma, a DeepSeek AI consegue diminuir drasticamente a quantidade de memória necessária para armazenar e processar o modelo – em muitos casos, uma redução de até 75% no uso de memória, com perda mínima de acurácia no desempenho do modelo. Esta não é uma ideia exclusiva da Deepseek. Para ser honesto, a Microsoft tem um modelo, o BitNet b1.58, um _Transformer_ treinado com uma representação de pesos de **1.58 bits** que resultou nos modelos Phi-3 e Phi-4, que ainda não teve tanto sucesso porque, como todo mundo, Google, Claude, OpenAi, X, o foco da Microsoft estava no modelo de linguagem.
2. **Previsão de Múltiplos Tokens:** Imagine ler um livro palavra por palavra, ou ler por frases inteiras. A DeepSeek AI aplica esse conceito à forma como processa a linguagem. Em vez de prever o próximo token (palavra ou parte de uma palavra) individualmente, como os modelos antigos e ultrapassados, o sistema deles é capaz de prever vários tokens de uma só vez, formando uma sequência mais longa. Isso acelera significativamente o processo de geração de texto, tornando-o cerca de duas vezes mais rápido, mantendo aproximadamente 90% da precisão. Esse ganho de velocidade é crucial para processar grandes volumes de dados de forma eficiente. Essa ideia começa com o Google, passa pela Meta e Microsoft e, principalmente, por um grupo de pesquisa formado por pesquisadores da Berkley, Stanford e CMU.

- **Sistema de Especialistas (MoE - _Mixture of Experts_):** aqui está o pulo do gato. Ao invés de depender de um único modelo gigantesco que tenta "saber de tudo", a DeepSeek AI utiliza uma abordagem mais inteligente e modular. Eles desenvolveram um sistema chamado **"_Mixture of Experts_" (MoE)**, que consiste em múltiplos modelos "especialistas" menores. Cada um desses especialistas é treinado em uma área específica. Um sistema de roteamento inteligente direciona cada tarefa para o especialista mais adequado, ativando-o apenas quando necessário. Mais, ou menos, como o Google Search faz há anos: quando você inicia uma busca, vários sistemas avaliam a busca, devolvendo informações diferentes pertinentes em formatos diferentes. No caso da DeepSeek, pense em uma empresa com um grande número de departamentos muito especializados: ao invés de todos trabalharem em todas as tarefas, cada departamento lida com o que faz melhor. Isso significa que, dos 671 bilhões de parâmetros totais do sistema, apenas 37 bilhões são ativados em um dado momento, otimizando drasticamente o uso de recursos computacionais. E, novamente aqui, eles não foram os únicos nem os primeiros, Google, Microsoft, Nvidia e Meta já publicaram trabalhos usando esta tecnologia que, justiça seja feita, se não me falha a memória ou a vista, começou em um artigo da Google lá nos idos de 2021.

Em resumo, nenhuma tecnologia inédita. Eles só foram melhores e mais competentes. Talvez porque eles tenham usado estas ideias de forma integrada. E aqui está o segredo de 1 trilhão de dólares. O tamanho da equipe de pesquisa e desenvolvimento. Dando uma busca podemos encontrar:

- **Google:** a Google possui uma força de trabalho considerável dedicada à IA, com mais de 200 funcionários focados em tempo integral na operacionalização de práticas responsáveis para o desenvolvimento de IA. Além disso, a Google DeepMind, uma subsidiária focada em pesquisa e desenvolvimento de IA, conta com aproximadamente 2.600 funcionários.
- **OpenAI:** as informações sobre o número de funcionários da OpenAI variam bastante, com estimativas que vão de 375 a 3.531 funcionários. Essa discrepância pode ser resultado do rápido crescimento da empresa e da inclusão de diferentes tipos de funções nas contagens.   E, existe o rumor sobre dezenas de milhares de avaliadores de respostas para o **RLHF** da empresa.
- **Anthropic (Claude):** a Anthropic, criadora do Claude, também apresenta variações nas estimativas de tamanho da equipe, com números que vão de 101 a 1.035 funcionários. Assim como no caso da OpenAI, essa variação pode refletir o crescimento da empresa e a dificuldade em isolar os funcionários que trabalham especificamente no desenvolvimento de modelos de IA.
- **DeepSeek:** a DeepSeek se destaca por sua equipe enxuta, com cerca de 220 funcionários. A empresa inteira é estimada como tendo 220 funcionários. Se compararmos com as outras, não chega a ser uma empresa. Está mais para grupo de trabalho. De novo, a empresa inteira.

Eu jogo minhas fichas neste detalhe. Uma equipe menor, com foco e competência, sem os limites de uma estrutura complexa e megalítica, consegue resultados melhores.

**Resultados Impactantes:**

- **Custo de Treinamento:** Reduzido de US$ 100 milhões para US$ 5 milhões. Na pior avaliação que eu encontrei.  
- **Necessidade de GPU’s:** De 100.000 para 2.000. Isso é, chutar o pau da barraca do mercado de GPU. Segundo consta, a DeepSeek usou 2.048 GPUs H800 em vez das 100.000 ou mais reportadas pelos outros modelos.
- **Custos de API:** 95% mais baratos. Simplesmente tornou as outras API’s economicamente inviáveis, injustificáveis e inúteis.
- **Capacidade de Execução:** Possibilidade de rodar em GPU’s de jogos, em vez de hardware de data center de alto custo.

**O que isso significa para o mercado?** A inovação da DeepSeek AI tem o potencial de democratizar o desenvolvimento de IA. Grandes empresas de tecnologia, que antes tinham a vantagem de seus imensos recursos, agora enfrentam uma competição mais acirrada. As barreiras de entrada diminuem, permitindo que empresas menores e até indivíduos entrem no jogo.

Levante da cadeira e vá trabalhar! Não, as grandes empresas não são invencíveis! São estas as lições que podemos aprender da DeepSeek. Porém, como de Nostradamus e louco, todo mundo tem um pouco. **O que podemos esperar nos próximos meses?**

Um palpite vai na linha da **acessibilidade ampliada ao desenvolvimento de IA**. Eles mostraram que o mercado está aberto e não existe um vencedor definido e você pode sim pesquisar IA com a sua GPU ou pagando muito pouco por GPU’s em nuvem. Basta ser competente e criativo.

Acredito que veremos **um aumento dramático da competição**.  As universidades, notadamente nos EUA, que vinham mostrando sinais de desanimo graças aos custos envolvidos, voltam ao jogo. Muito mais importante que isso: pequenos empreendedores voltam ao jogo. Os investidores também mostravam sinais de desanimo. Afinal, quem pode competir com os investimos de centenas de bilhões de dólares que, até há 15 dias, eram indispensáveis para entrar na brincadeira. Agora, abre-se uma janela. Pode ser que surja outra grande ideia. **Se você é investidor é melhor coçar o bolso**.

Algumas empresas terão que, da noite para o dia, **rever seu modelo de negócio**. A Nvidia parece ser a mais impactada. Hoje, ainda não é possível avaliar o impacto da redução do custo de treinamento no negócio. Pode ser que o investimento de **100 bilhões** **de dólares**, projeto **Stargate** de 21 de janeiro de 2025, anunciado pela OpenAI, Oracle e Softbank seja um tiro no pé, com modelos melhores que rodem em qualquer lugar, ou não. O que sabemos com certeza é que o **Banco da Chi**na, lançou, 25 de janeiro de 2025, um plano de investimento _AI Industry Development Cátion Plan_ com investimento de **137 bilhões de dólares**. Isso foi susto, ou eles sabem alguma coisa que eu não sei?

A DeepSeek AI fez tudo isso com uma equipe de pesquisa e desenvolvimento com **menos de 200 pessoas** e, o mais incrível, **disponibilizou tudo em código aberto**. Isso mostra que a inovação nem sempre vem do tamanho, mas sim da inteligência e da eficiência.  E, aqui está, novamente, **a sua oportunidade**. Depende de você.

**O que você acha das inovações da DeepSeek AI? Quais serão os próximos capítulos dessa revolução? Compartilhe suas opiniões nos comentários!**

**#IA #InteligenciaArtificial #DeepSeekAI #Inovação #Tecnologia #OpenSource #Eficiência #Disrupção #Mercado**

&nbsp;