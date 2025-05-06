---
layout: post
title: "Reinforcement Learning: História"
author: Frank
categories:
    - Matemática
    - Inteligência Artificial
tags:
    - inteligência artificial
    - Matemática
    - resolução de problemas
    - reinforcement learning
image: assets/images/rl1.webp
featured: false
rating: 0
description: A história do Reinforcement Learning, da psicologia do século XIX ao DeepMind.  Ideal para estudantes de IA e ciência da computação!
date: 2025-02-03T17:20:09.061Z
preview: Nos últimos anos a aprendizagem por reforço (*reinforcement learning - *RL**) tem chamado a atenção da mídia que cobre inteligência artificial em todo o mundo. Sua trajetória, no entanto, é uma fascinante tapeçaria tecida com fios de diversas disciplinas, desde a psicologia do século XIX até os algoritmos de raciocínio que estamos vendo nascer.
keywords: Reinforcement Learning, História do RL, Richard Bellman, Q-Learning, Deep Q-Networks, AlphaGo, Processos de Decisão de Markov, Psicologia Comportamental, Dopamina e IA, Deep Reinforcement Learning.
toc: true
published: true
lastmod: 2025-05-06T11:04:17.920Z
draft: 2025-02-03T17:19:36.133Z
---

Nos últimos anos a aprendizagem por reforço (*reinforcement learning - *RL**) tem chamado a atenção da mídia que cobre inteligência artificial em todo o mundo. Sua trajetória, no entanto, é uma fascinante tapeçaria tecida com fios de diversas disciplinas, desde a psicologia do século XIX até os algoritmos de raciocínio que estamos vendo nascer.

> We do not know what the rules of the game are; all we are allowed to do is watch the playing. Of course if we are allowed to watch long enough we may catch on a few rules. - *Richard Feynman*

## Psicologia e Biologia (Século XIX - Meados do Século XX)

A essência do *RL* reside na ideia de aprender com base em recompensas e punições, um conceito profundamente enraizado na psicologia comportamental e na biologia.

Embora [Edward Thorndike](https://www.britannica.com/biography/Edward-L-Thorndike) (1898) seja frequentemente citado por sua **Lei do Efeito**, a qual postula que comportamentos seguidos por recompensas tendem a se repetir, enquanto aqueles seguidos por punições tendem a ser evitados, sendo festejado mundialmente como o pai da teoria de aprendizagem por reforço, também precisamos reconhecer as contribuições de [Ivan Pavlov](https://www.britannica.com/biography/Ivan-Pavlov) (final do séc. XIX, início do séc. XX).

Pavlov, através de seus experimentos de condicionamento clássico, demonstrou como um estímulo neutro (como um sino) pode ser associado a um estímulo incondicionado (como comida) para produzir uma resposta condicionada (salivação). Este trabalho introduziu o conceito fundamental de aprendizado associativo, que, embora diferente do condicionamento operante, pavimentou o caminho para a compreensão de como as conexões entre estímulos e respostas são formadas e influenciam o comportamento[^1].

[^1]: Outra hora escrevo sobre condicionamento operante. Tenho uma história ótima sobre isso.

Posteriormente, [B.F. Skinner](https://www.britannica.com/biography/B-F-Skinner) (1938) expandiu esses conceitos com o condicionamento operante. Seus experimentos com ratos e pombos demonstraram que os animais podiam aprender a associar ações específicas a consequências positivas (reforço) ou negativas (punição). Skinner solidificou a ideia de que o aprendizado é moldado por reforços, um princípio central que se tornaria a pedra angular do *RL*.

<div class="video-wrapper">
   <iframe src="https://www.youtube.com/embed/NeK8GNLylkc?autoplay=0"
          loading="lazy"
          allow="clipboard-write; encrypted-media; picture-in-picture"
          allowfullscreen>
  </iframe>
</div>

Além da psicologia, a **Etologia**, o estudo do comportamento animal em seus ambientes naturais, também contribuiu para o contexto do *RL*. Pesquisadores como [Konrad Lorenz](https://www.britannica.com/biography/Konrad-Lorenz) e [Nikolas Tinbergen](https://www.britannica.com/biography/Nikolaas-Tinbergen) estudaram comportamentos instintivos e padrões fixos de ação, que, embora distintos do Reinforcement Learning, ofereceram insights valiosos sobre o comportamento animal e influenciaram o desenvolvimento de modelos computacionais de comportamento.

Essas ideias pioneiras da psicologia e da biologia lançaram as bases conceituais para o *RL*. Contudo, a atenta leitora deve ter percebido que faltava uma estrutura matemática formal para transformá-las em algoritmos computacionais.

## Processos de Decisão de Markov (**MDPs**) (Anos 1950)

A década de 1950 marcou o início da formalização matemática de problemas de decisão sequencial. [Richard Bellman](https://en.wikipedia.org/wiki/Richard_E._Bellman) (1957) foi um figura central nesse processo, introduzindo o conceito de **programação dinâmica**[^2], uma técnica para resolver   de forma eficiente. Ele formulou o princípio da *otimalidade*, que afirma que a solução ótima de um problema pode ser decomposta em subproblemas ótimos, permitindo a resolução de problemas complexos através da resolução de problemas mais simples. Uma ideia, dividir para conquistar, que começou com [Julius Caesar](https://en.wikipedia.org/wiki/Julius_Caesar), mas teve seu ápice matemático na pesquisa sobre recursão que começou com [Kurt Gödel](https://www.britannica.com/biography/Kurt-Godel). E que, a atenta leitora estudou [aqui](https://amzn.to/3WKq0mk).

![Ilustração mostrando dois professores no quadro negro](/assets/images/rl2.webp)

[^2]: BELLMAN, Richard Ernest. Dynamic Programming. Princeton, New Jersey: Princeton University Press, 1957.

Juntamente com a programação dinâmica, Bellman desenvolveu os **Processos de Decisão de Markov (MDPs)**, que se tornaram a estrutura matemática fundamental para o *RL*. **MDPs** fornecem um quadro formal para modelar problemas de decisão onde o resultado depende tanto das ações do agente quanto de estados aleatórios do ambiente. Neste ponto, não posso deixar de mencionar [Ronald Howard](https://en.wikipedia.org/wiki/Ronald_A._Howard) (1960), que em seu livro *Dynamic Programming and Markov Processes*, aplicou os **MDPs** a problemas práticos de tomada de decisão e ajudou a popularizar essa abordagem.

Dentro do contexto de **MDPs**, dois algoritmos clássicos para encontrar a política ótima emergiram: Iteração de Valor (*Value Iteration*) e Iteração de Política (*Policy Iteration*). Esses algoritmos, baseados nos princípios da programação dinâmica, foram fundamentais para resolver **MDPs** antes do advento de métodos mais escaláveis.

Os **MDPs**, inicialmente aplicados em áreas como controle e economia, forneceram a estrutura matemática necessária para o desenvolvimento do *RL* como o conhecemos hoje.

>Até aqui, a criativa leitora pode aproveitar uma dica: problemas de otimização sequencial.

## Os Primeiros Passos Computacionais: Aprendizado de Máquina (Anos 1960-1980)

Nas décadas de 1960 e 1970, os pesquisadores começaram a explorar como os computadores poderiam aprender através da interação com o ambiente. [Arthur Samuel](https://en.wikipedia.org/wiki/Arthur_Samuel_(computer_scientist)) (1959) deu um passo pioneiro com seu programa de aprendizado para jogar damas. Esse programa, embora rudimentar, ajustava seus parâmetros com base em recompensas (vitórias ou derrotas) e representa um dos primeiros exemplos de Reinforcement Learning em, o que hoje chamamos de inteligência artificial, influenciando inclusive as futuras ideias de Aprendizado por Diferença Temporal.

[Widrow](https://en.wikipedia.org/wiki/Bernard_Widrow) e [Hoff](https://en.wikipedia.org/wiki/Marcian_Hoff) (1960) propuseram o algoritmo [ADALINE](https://github.com/TheAlgorithms/C-Plus-Plus/blob/master/machine_learning/adaline_learning.cpp), que **usava feedback para ajustar pesos em redes neurais simples**. Embora não fosse *RL* no sentido moderno, introduziu a ideia de usar feedback para melhorar o desempenho, um conceito indispensável para o desenvolvimento posterior dos campos de estudo relacionados a inteligência artificial.

Nesse período, também surgiram as primeiras discussões sobre o dilema da exploração-explotação (*exploration-exploitation dilemma*), um conceito central no *RL*. **O agente precisa encontrar um equilíbrio entre explorar novas ações para descobrir melhores recompensas e explorar ações que já se mostraram promissoras**. A Cibernética, um campo interdisciplinar que explora sistemas de controle e comunicação em animais e máquinas, também teve influência nesse período inicial, com conceitos como feedback e controle sendo compartilhados entre a cibernética e o *RL*.

![imagem de um robô perdido em um labirinto](/assets/images/rl3.webp)

Esta área de pesquisa ainda estava em sua primeira infância, mas as sementes para o futuro florescimento do *RL* já haviam sido plantadas.

## O Florescimento do Reinforcement Learning Moderno (Anos 1980-1990)

Os anos 1980 e 1990 testemunharam a consolidação do *RL* como uma disciplina distinta dentro do estudo da inteligência artificial. [Richard Sutton](http://www.incompleteideas.net/) (1988) introduziu os métodos de Aprendizado por Diferença Temporal (**TD**), uma classe de algoritmos que aprendem estimativas de valor com base em diferenças temporais entre previsões sucessivas. A grande contribuição dos métodos **TD** foi a combinação da amostragem e do aprendizado por *bootstrap* do aprendizado por diferença temporal com os conceitos de atualização de valor e política da programação dinâmica. Em RL, o termo *bootstrap* significa que o algoritmo aprende uma estimativa (ex: valor de um estado) e, em seguida, usa essa própria estimativa como alvo para atualizar outras estimativas. É como se o algoritmo se *puxasse pelas próprias botas* para aprender, usando o conhecimento que já possui para aprimorar seu aprendizado[^3]. Isso permitiu o aprendizado online e incremental, sem a necessidade de um modelo completo do ambiente, um avanço significativo em relação aos métodos anteriores.

[^3]:O termo bootstrap tem origem em uma expressão idiomática em inglês do século XIX, que se referia à ação de se erguer ou superar uma dificuldade sem ajuda externa, puxando-se pelas próprias botas (*to pull oneself up by one's bootstraps*). Essa expressão era frequentemente associada à imagem de um indivíduo que, literalmente, se impulsionava para cima puxando as tiras de couro de suas botas.

[Chris Watkins](https://www.cs.rhul.ac.uk/~chrisw/) (1989) desenvolveu o algoritmo **Q-Learning**, um algoritmo off-policy (não dependente diretamente da política atual) que permite que **um agente aprenda uma política ótima sem conhecer o modelo do ambiente**. O **Q-Learning** é um marco importante nesta história. Notadamente porque permitiu que os agentes aprendessem a partir de experiências geradas por qualquer política, aumentando significativamente a flexibilidade e a eficiência do aprendizado. Além do **Q-Learning**, o algoritmo SARSA (*State-Action-Reward-State-Action*), proposto por [Rummery e Niranjan](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) (1994), também ganhou destaque. **SARSA** é um algoritmo *on-policy* que aprende a função *Q* com base na política que está sendo executada atualmente.

Nesse período, [Gerald Tesauro](https://www.researchgate.net/scientific-contributions/Gerald-Tesauro-8269192) (1992) demonstrou o poder do *RL* em jogos complexos com o TD-Gammon, um programa que aprendeu a jogar Gamão (*Backgammon*) em um nível de mestre mundial através da auto-aprendizagem (*self-play*) usando métodos **TD**. Esse trabalho foi um marco importante, demonstrando a capacidade do *RL* de aprender estratégias complexas em domínios desafiadores.

Paralelamente, as conexões entre *RL* e a neurociência começaram a ser exploradas. A descoberta de que o neurotransmissor *dopamina* desempenha uma ação no sistema de recompensa do cérebro forneceu uma base biológica para os mecanismos de Reinforcement Learning[^4].

[^4]: A dopamina, neurotransmissor tradicionalmente associado à recompensa, atua primariamente sinalizando a expectativa de recompensa e não a recompensa em si (SCHULTZ, Wolfram. Neuronal reward and decision signals: from theories to data. Physiological reviews, v. 95, n. 3, p. 853-951, 2015.). No aprendizado humano, ela interage com outros neurotransmissores como serotonina e norepinefrina em um sistema complexo de reforço comportamental (BERKE, Joshua D. What does dopamine mean?. Nature neuroscience, v. 21, n. 6, p. 787-793, 2018.). Estímulos sociais digitais, como feedback em redes sociais, ativam vias dopaminérgicas em conjunto com outros sistemas neurais. O sistema de recompensa intermitente dessas plataformas pode influenciar padrões comportamentais, porém a formação de comportamentos compulsivos envolve também alterações em circuitos de controle inibitório e transmissão glutamatérgica (MONTAGUE, P. Read; HYMAN, Steven E.; COHEN, Jonathan D. Computational roles for dopamine in behavioural control. Nature, v. 431, n. 7010, p. 760-767, 2004.). Esta interação entre neurobiologia e design digital sugere uma relação complexa onde múltiplos sistemas neurais influenciam o aprendizado e comportamento nas redes sociais, não se limitando apenas à via dopaminérgica.

## A Era do Deep Reinforcement Learning (Anos 2000 - Presente)

O advento do novo milênio, juntamente com o aumento exponencial do poder computacional e o desenvolvimento de redes neurais profundas, inaugurou a era do **Deep Reinforcement Learning**. A combinação de *RL* com *Deep Learning* permitiu que os agentes aprendessem representações complexas do ambiente e tomassem decisões em domínios de alta dimensionalidade, algo que era impossível com os métodos tradicionais de *RL*.

Um marco fundamental foi o desenvolvimento do **Deep Q-Networks (DQN)** (2013) por uma equipe do Google DeepMind, liderada por [Volodymyr Mnih](https://www.cs.toronto.edu/~vmnih/). O **DQN** combinou **Q-Learning** com redes neurais profundas para aprender a jogar jogos Atari diretamente a partir de pixels brutos, alcançando desempenho super-humano em vários jogos. O **DQN** popularizou a experiência replay (*replay buffer*), uma técnica proposta inicialmente por [Long-Ji Lin](https://dl.acm.org/profile/81100220663) (1992), que armazena experiências passadas e as reutiliza para treinamento, ajudando a estabilizar o aprendizado com redes neurais profundas. Também é importante destacar o **Double Q-learning** ([Van Hasselt](https://arxiv.org/abs/1509.06461), 2010), que aborda a tendência do Q-learning de superestimar valores de ação, usando dois conjuntos de pesos para separar a seleção da ação da avaliação da ação, tornando o aprendizado mais estável.

Outro feito notável do DeepMind foi o [AlphaGo (2016)](https://en.wikipedia.org/wiki/AlphaGo_versus_Lee_Sedol), que derrotou o campeão mundial de Go, Lee Sedol. O AlphaGo utilizou uma combinação de *RL*, aprendizado supervisionado e busca em árvore Monte Carlo para dominar o Go, um jogo de tabuleiro extremamente complexo, considerado um grande desafio para a IA.

Desde o **DQN**, diversos outros algoritmos de Deep *RL* foram desenvolvidos, expandindo as capacidades do *RL* para diferentes tipos de problemas:

**Deep Deterministic Policy Gradient (DDPG)**: Para espaços de ação contínuos, permitindo a aplicação do *RL* em problemas de controle robótico, por exemplo.
**Trust Region Policy Optimization (TRPO)** e **Proximal Policy Optimization (PPO)**: Algoritmos que melhoram a estabilidade do treinamento em métodos de gradiente de política, tornando-os mais robustos e confiáveis.
**Asynchronous Advantage Actor-Critic (A3C)** e **Advantage Actor-Critic (A2C)**: Métodos actor-critic que usam múltiplas cópias do agente para aprender de forma paralela, acelerando o processo de aprendizado.

E, chegamos ao [DeepSeek-R1](https://frankalcantara.com/deepseek-explicado-de-forma-simples/).

A jornada do Reinforcement Learning é uma história de convergência e inovação. Desde suas raízes na psicologia comportamental e na biologia, passando pela formalização matemática com os **MDPs**, até a revolução do *Deep Learning*, o *RL* se transformou em uma ferramenta poderosa para a construção de agentes inteligentes. Com o rápido progresso em IA e computação, o futuro do *RL* promete ainda mais inovações e impactos transformacionais em diversas áreas da sociedade. Essa história está só começando, e eu também.

Me siga se quiser saber mais sobre *reinforcement learning*. Se tudo correr bem, vou escrever aqui, capítulo por capítulo, um livro texto sobre *reinforcement learning*.
