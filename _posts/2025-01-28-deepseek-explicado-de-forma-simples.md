---
layout: post
title: "Deepseek-R1: explicado de forma simples"
author: Frank
categories:
    - artigo
    - Matemática
    - Inteligência Artificial
tags:
    - algoritmos
    - Matemática
    - inteligência artificial
image: assets/images/deep4.webp
featured: false
rating: 0
description: Uma análise dos artigos sobre o modelo DeepSeek-R1, com explicações um pouco mais detalhadas das principais tecnologias envolvidas
date: 2025-01-29T23:24:18.812Z
preview: Para entender o DeepSeek-R1 e Reinforcement Learning usando como base o artigo de lançamento com um pouco mais de profundidade e didática.
keywords: |-
    DeepSeek-R1
    Reinforcement Learning
    Inteligência Artificial 
    MoE 
toc: true
published: true
lastmod: 2025-04-19T00:31:03.901Z
slug: deepseek-explicado-de-forma-simples
---

Uma das disciplinas que leciono na Pontifícia Universidade Católica do Paraná, **Construção de Interpretadores** engloba o processamento de linguagens formais a naturais. [Dado o terremoto provocado](https://frankalcantara.com/deepseek-ai-revolucao-na-eficiencia-pode-abalar-mercado/) pela [DeepSeek](https://deepseek.com/) com o seu modelo DeepSeek-R1, fiquei curioso e resolvi fazer um apanhado artigos para que as vozes na minha cabeça se acalmem um pouco. Curiosidade mata gato mas excita o pesquisador. Esse é o resultado deste esforço.

A primeira coisa importante a notar é que o DeepSeek-R1 está sob a licença MIT, e que pode ser encontrado no [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1). Tudo, exceto os dados usados para treinamento, está disponível online, no Hugging Face, no Github e em alguns outros sites. 

> A grande questão é: porque não os dados de treinamento? A resposta mais óbvia é: porque aqui está o problema. Mas isso fica para outra discussão[^1].  

O R1 chamou a atenção por empatar, ou bater os modelos antigos e tradicionais.

![Comparação entre os resultados de diversos modelos](/assets/images/deep3.webp)

*Comparação entre os resultados de diversos modelos*{: class="legend"}

> Achei o máximo escrever *modelos antigos e tradicionais* para uma tecnologia de 4 anos, no máximo.

[O R1 quase derrubou a internet por, supostamente, ter sido criado com um custo 20 vezes menor](https://frankalcantara.com/deepseek-ai-revolucao-na-eficiencia-pode-abalar-mercado/).

O que realmente me interessa, já que não tenho acesso aos dados, neste modelo é o uso de *Reinforcement Learning* por eles que foi descaradamente explicitado em vários artigos abertos. Me interessa porque eu tenho falado para os meus alunos que o **próximo salto evolutivo da humanidade será devido a *Reinforcement Learning***. Então, talvez, só talvez, a DeepSeek não me deixe mentir sozinho.

Uma das inovações do DeepSeek-R1 é a adoção da *Group Robust Preference Optimization* (**GRPO**), introduzida no artigo DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models sobre o trabalho de Schulman et.al de 2017 *Group Robust Preference Optimization in Reward-free RLHF*. Essa técnica substitui métodos tradicionais de otimização de políticas, como o *Proximal Policy Optimization* (PPO), apresentado por Schulman et al. em *Proximal Policy Optimization Algorithms*. Simplificando, a **GRPO** permite que o modelo aprenda de forma mais eficaz comparando seu desempenho com o de outros modelos em um grupo, otimizando suas ações para alcançar melhores resultados em tarefas de raciocínio matemático. Essa abordagem torna o processo de treinamento mais eficiente e escalável se comparado com o PPO.

Além da **GRPO**, o DeepSeek-R1 incorpora a *Multi-head Latent Attention* (MLA), uma técnica introduzida no DeepSeek-V3, que, por sua vez, foi inspirada no trabalho de Kitaev, Kaiser e Levskaya em *Reformer: The Efficient Transformer*. A MLA aborda as ineficiências computacionais e de memória associadas ao processamento de sequências longas, especialmente em modelos de linguagem com atenção multi-cabeça. Em termos simples podemos dizer que a MLA melhora a eficiência do modelo ao simplificar a forma como ele processa as informações. Ela projeta as matrizes *Key-Query-Value* (KQV) em um espaço latente de menor dimensão, reduzindo a complexidade computacional e melhorando a eficiência do modelo.

Neste momento você tem duas escolhas claras: sentar em um lugar mais confortável já que vai demorar, ou ir fazer scroll no instagram.

## Fundamentos da Arquitetura

A sopa de letrinhas que precisa ser consumida, morna e vagarosamente, para entender como o DeepSeek-R1 funciona, ainda precisa de algum tempero.  

Algumas das mudanças realizadas pela equipe de DeepSeek, liderada por [Luo Fuli um prodígio com cara de atriz de dorama](https://english.mathrubhumi.com/features/technology/luo-fuli-deepseek-ai-genius-1.10294853),  incluem *Mixture of Experts* (MoE), *Multi-head Latent Attention* (MLA), Quantização FP8 e *Multi-Token Prediction* (MTP). A saber:

### Mixture of Experts (MoE)

O mecanismo *Mixture of Experts* (MoE) ativa apenas um subconjunto dos parâmetros totais dentro de cada bloco *Transformer*, permitindo economias computacionais substanciais enquanto preserva a qualidade do modelo. Esta ativação seletiva é particularmente vantajosa para escalar os parâmetros do modelo sem aumentar proporcionalmente os custos computacionais.

A função *gate* de seleção de especialistas é governada por uma função de porta $G(x)$ que direciona tokens $x$ para especialistas $E_k$, definida como:

$$
G(x) = softmax(W_gx)
$$

Cada token é então processado pelos especialistas selecionados, agregados como:

$$
y = \sum_{k \in K} G_k(x)E_k(x)
$$

Uma perda de balanceamento de carga é adicionada para encorajar utilização igual dos especialistas, reduzindo gargalos computacionais.

Vamos ver um exemplo simplificado de como o MoE funciona na prática. Imagine que temos:

- 3 especialistas ($E_1$, $E_2$, $E_3$)
- Um token de entrada $x$ representando a palavra "computador"

Primeiro, o token passa pela função *gate* $G(x)$, que calcula um score para cada especialista. Vamos dizer que após a transformação $W_gx$ e aplicação do softmax, obtemos:

$$
G(x) = softmax(W_gx) = [0.7, 0.2, 0.1]
$$

Isto significa que:

- Especialista 1 ($E_1$): 70% de ativação
- Especialista 2 ($E_2$): 20% de ativação
- Especialista 3 ($E_3$): 10% de ativação

Agora, suponha que cada especialista processe o token e produza um vetor de características:

$$
\begin{align*}
E_1(x) &= [1.0, -0.5] \\
E_2(x) &= [0.3, 0.8] \\
E_3(x) &= [-0.2, 0.4]
\end{align*}
$$

A saída final será a soma ponderada desses vetores, usando os pesos da função gate:

$$
\begin{align*}
y &= 0.7 \cdot [1.0, -0.5] + 0.2 \cdot [0.3, 0.8] + 0.1 \cdot [-0.2, 0.4] \\
&= [0.7, -0.35] + [0.06, 0.16] + [-0.02, 0.04] \\
&= [0.74, -0.15]
\end{align*}
$$

Agora, imagine que após processar vários tokens, notamos que o Especialista 1 está sendo usado 80% do tempo. Aqui é onde a perda de balanceamento entra em ação:

Para $K = 3$ especialistas, a frequência ideal é $\frac{1}{K} = \frac{1}{3} \approx 0.33$

Calculando a perda de balanceamento para este caso (com $\alpha = 1$):

$$
\begin{align*}
L_{balance} &= \sum_{k=1}^3 (f_k - \frac{1}{3})^2 \\
&= (0.8 - 0.33)^2 + (0.15 - 0.33)^2 + (0.05 - 0.33)^2 \\
&= 0.47^2 + (-0.18)^2 + (-0.28)^2 \\
&= 0.22 + 0.03 + 0.08 \\
&= 0.33
\end{align*}
$$

Este valor alto de $L_{balance}$ indica um desequilíbrio significativo na utilização dos especialistas, e o modelo será penalizado por isso durante o treinamento, incentivando-o a desenvolver uma distribuição mais equilibrada nas próximas iterações.

O MoE funciona essencialmente como um sistema de distribuição de tráfego inteligente, onde o "roteador" (chamado de função de gate ou porta) decide qual especialista ou combinação de especialistas deve processar cada token de entrada. Este roteamento é feito de forma dinâmica e aprendida, não através de regras fixas.

Para entender melhor, podemos fazer uma analogia com um hospital:
Imagine um grande hospital com vários médicos especialistas. Quando um paciente chega, similar a um token de entrada, um enfermeiro de triagem muito experiente, a função de *gate*, avalia rapidamente o caso e decide quais especialistas devem atender o paciente. Alguns casos podem precisar de apenas um especialista, enquanto outros podem requerer uma equipe de diferentes especialidades.

No contexto do DeepSeek-R1, este roteamento é representado matematicamente pela função $G(x)$, que podemos entender como um direcionador que:

1. Recebe um token de entrada $x$
2. Avalia suas características através de uma transformação $W_gx$
3. Usa uma função softmax para gerar probabilidades de encaminhamento para diferentes especialistas
4. Direciona o token para os especialistas mais apropriados

Finalmente temos a  perda de balanceamento de carga. Um mecanismo que evita que alguns especialistas fiquem sobrecarregados enquanto outros ficam ociosos. Para entender este conceito, podemos voltar ao nosso hospital:

Imagine que em um hospital, alguns médicos especialistas começam a receber muito mais pacientes que outros. Por exemplo, um cardiologista está sempre ocupado, atendendo 80% dos pacientes, enquanto um neurologista mal recebe pacientes. Isso cria dois problemas: o cardiologista fica sobrecarregado, podendo causar atrasos e queda na qualidade do atendimento; e o conhecimento do neurologista está sendo desperdiçado.

Para resolver isso, o hospital, nosso sistema MoE, adiciona uma regra especial na função de triagem: se o enfermeiro da triagem, função *gate*, percebe que está enviando muitos pacientes para um mesmo especialista, ele recebe um "feedback negativo", a perda de balanceamento, que o incentiva a distribuir melhor os pacientes. E viva o **Reinforcement Learning**!

Matematicamente, isso é implementado como um termo adicional na função de perda total do modelo:

$$ L_{balance} = \alpha \cdot \sum_{k=1}^K (f_k - \frac{1}{K})^2 $$

Nesta equação:

- $f_k$ representa a frequência com que o especialista $k$ é utilizado;
- $\frac{1}{K}$ é a frequência ideal (distribuição uniforme);
- $\alpha$ é um hiperparâmetro que controla a importância deste balanceamento.

Este tapinha na mão, perda adicional de balanceamento, age como um regulador, penalizando o modelo quando ele desenvolve preferências muito fortes por certos especialistas. Na verdade, o sistema busca minimizar $L_{balance}$, o que naturalmente leva a uma distribuição mais uniforme do trabalho entre os especialistas.

O objetivo final é garantir que todos os especialistas sejam utilizados de forma aproximadamente igual ao longo do tempo, evitando gargalos e maximizando a eficiência. Continuando nossa analogia, é como ter um bom administrador hospitalar que garante que todos os médicos estejam contribuindo de forma equilibrada para o funcionamento do hospital. Nem que ele tenha que chamar a atenção de alguém aqui e ali.

Quando esta perda de balanceamento é combinada com a função principal do MoE:

$$ y = \sum_{k \in K} G_k(x)E_k(x) $$

O sistema completo consegue não apenas rotear tokens para os especialistas mais apropriados, mas também manter uma distribuição saudável de carga de trabalho em todo o modelo.

A beleza deste sistema é que ele é eficiente, adaptativo e escalável: eficiente por só ativar os especialistas necessários para cada token; adaptativo por aprender os padrões de roteamento durante o treinamento e escalável por permitir aumentar o número de parâmetros do modelo sem aumentar proporcionalmente o custo computacional. O MoE é o cara.

### Group Robust Preference Optimization

A *Group Robust Preference Optimization* (**GRPO**) representa uma evolução significativa nos métodos de otimização para modelos de linguagem, substituindo abordagens tradicionais como o *Proximal Policy Optimization* (PPO). Esta técnica introduz um paradigma de aprendizado que compara o desempenho do modelo com outros em um grupo de referência.

A função objetivo do **GRPO** pode ser expressa matematicamente como:

$$
J_{GRPO}(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \min_{i \in G} \frac{p_\theta(y|x)}{p_i(y|x)} \right]
$$

Onde $\theta$ representa os parâmetros do modelo, $G$ é o grupo de referência, e $p_i(y|x)$ é a probabilidade atribuída pelo modelo $i$ à saída $y$ dado o contexto $x$.

Para entender melhor como o **GRPO** funciona na prática, considere um exemplo de raciocínio matemático:

Dado um problema matemático $x$: "Qual é a derivada de $f(x) = x^2$?"

E uma resposta candidata $y$: "$f'(x) = 2x$"

O **GRPO** avalia a resposta comparando com um grupo de modelos de referência:

$$
\begin{align*}
\text{Modelo A}: &P(y|x) = 0.95 \\
\text{Modelo B}: &P(y|x) = 0.92 \\
\text{Modelo C}: &P(y|x) = 0.88
\end{align*}
$$

A otimização **GRPO** busca maximizar:

$$
\min\left\{\frac{0.95}{0.92}, \frac{0.95}{0.88}\right\} = \min\{1.033, 1.080\} = 1.033
$$

Este processo incentiva o modelo a desenvolver robustez em suas previsões, evitando sobre-ajuste a um único critério de avaliação. O **GRPO** é mais eficiente graças a MLA.

### Multi-head Latent Attention (MLA)

O sistema *Multi-head Latent Attention* (MLA), introduzido no DeepSeek-V3, reduz ineficiências computacionais e de memória projetando matrizes *Key-Query-Value* (KQV) em um espaço latente de menor dimensão. O objetivo é diminuir a latência de inferência e os custos computacionais, particularmente para processamento de contexto longo.

Para entender o MLA, podemos usar uma analogia com uma biblioteca. Imagine uma enorme biblioteca universitária com milhões de livros, nosso espaço original. Buscar um livro específico comparando-o com cada livro da biblioteca seria extremamente ineficiente. Em vez disso, a biblioteca usa um sistema de catalogação, nosso espaço latente, que representa cada livro por um código mais compacto, contendo informações essenciais como assunto, autor e localização.

No contexto do MLA, a transformação do mecanismo de atenção começa no espaço original, onde:

$K,Q,V = W_kX,W_qX,W_vX$

Aqui, $X$ representa nossa entrada, como os livros na biblioteca, e $W_k$, $W_q$, $W_v$ são transformações que geram nossas chaves ($K$), consultas ($Q$) e valores ($V$), similar a criar diferentes índices para nossos livros.

Estas representações são então projetadas em um espaço latente $L$ de menor dimensão:

$K_L,Q_L,V_L = W_LK,W_LQ,W_LV$

É como se criássemos um catálogo resumido que mantém apenas as informações mais relevantes, tornando as buscas muito mais eficientes.

Vamos ver como isso funciona na prática. Imagine que temos:

- Uma sequência de entrada com dimensão original de 1024;
- Um espaço latente com dimensão 64;

No espaço original, para calcular a atenção entre uma sequência de $100$ tokens: precisaríamos fazer $100 \times 100 = 10.000$ comparações. Neste caso, cada comparação envolveria vetores de dimensão $1024$.

No espaço latente, ainda fazemos $100 \times 100 = 10.000$ comparações. Mas cada comparação agora usa vetores de dimensão $64$. Implicando em uma redução de $16x$ na quantidade de memória necessária! Esta economia de memória é proporcional à razão entre as dimensões original e latente. Neste caso, $1024/64 = 16$.

A complexidade computacional permanece $O(N^2)$. Porém, as operações são realizadas em vetores de dimensão reduzida, $d_L$, reduzindo o custo computacional real. Em nosso exemplo temos:

1. No espaço de atenção original, com dimensão $d$:
  $$O(n^2d) \text{ operações, onde } n \text{ é o tamanho da sequência}$$

2. No espaço latente, com dimensão reduzida $d_l$:
  $$O(n^2d_l + nd_kd_l) \text{ operações, onde } d_l < d_k$$

Para ilustrar a diferença prática quando $n = 100$, $d_k = 1024$ e $d_l = 64$:

- A primeira expressão resulta em aproximadamente 10 milhões de operações;
- A segunda expressão resulta em aproximadamente 640 mil operações. Eita!

Voltando à nossa analogia da biblioteca, é como se pudéssemos transformar cada livro em um cartão de catálogo compacto, realizar buscas usando apenas estes cartões e acessar o livro completo apenas quando fosse realmente necessário.

O MLA aplica este mesmo princípio, permitindo que modelos processem textos muito mais longos com menos recursos. É como ter uma biblioteca infinitamente mais eficiente, onde podemos encontrar exatamente o que precisamos sem ter que olhar cada livro individualmente. É como ter o melhor dos dois mundos: a riqueza de informação do espaço original com a eficiência do espaço latente. Parece mágica, mas não é.

### FP8 Quantização: Compactando Números de Forma Inteligente

O DeepSeek-R1 utiliza quantização de ponto flutuante de 8 bits (FP8) para reduzir o uso de memória e custos computacionais. Este processo é similar à compressão de imagens digitais: assim como podemos reduzir o tamanho de uma foto mantendo sua qualidade visual, a quantização FP8 reduz o tamanho dos números preservando sua utilidade matemática. Comparado ao formato tradicional de 32 bits (FP32), o FP8 reduz os requisitos de memória em 75% enquanto mantém a estabilidade numérica necessária durante o treinamento e inferência do modelo.

Para realizar esta compressão numérica, o FP8 utiliza uma função de quantização curiosamente simples e deliciosamente elegante:

$$
x_q = clip(round(x/S), -127, 127)
$$

Onde $S$ é um fator de escala que é ajustado dinamicamente com base nos gradientes de perda. Para entender como isso funciona na prática, imaginemos um neurônio em nossa rede que precisa armazenar o valor $0.123456789$. No formato FP32, este número seria armazenado com alta precisão, usando 32 bits de memória. Durante a quantização FP8, primeiro dividimos este valor por um fator de escala, digamos $S = 0.01$, obtendo $12.3456789$. Este número é então arredondado para $12$ e multiplicado novamente por $S$, resultando em $0.12$. [Faz mais sentido em binário](https://frankalcantara.com/precisao-realidade-os-desafios-da-norma-ieee-754-na-computacao-moderna/).

>Este foi só um exemplo, existem vários padrões de quantização FP8, por exemplo:
>
>- FP8-E4M3: este formato oferece um bom equilíbrio entre faixa dinâmica e precisão, sendo adequado para uma variedade de aplicações de aprendizado de máquina.
>
>- FP8-E5M2: este formato prioriza uma faixa dinâmica maior em detrimento da precisão, sendo útil em casos onde a representação de números muito grandes ou muito pequenos é o fator crítico.
>
>No artigo, eles não explicitam que padrão usaram. E eu não vou baixar o código. Minha máquina não roda esse treco.

O processo é similar ao ajuste de zoom em uma câmera fotográfica. O fator de escala $S$ funciona como o zoom, ajustando o nível de detalhe que capturamos. Em regiões do modelo onde precisamos mais precisão, $S$ se ajusta automaticamente para preservar mais detalhes numéricos, como um fotógrafo ajustando o zoom para capturar detalhes importantes em uma cena[^2].

Em um modelo com um bilhão de parâmetros, esta técnica reduz o uso de memória de 4 gigabytes para apenas 1 gigabyte. Além disso, números menores são processados mais rapidamente pelo hardware, e mais dados podem ser mantidos nos caches do processador, acelerando todas as operações do modelo.

O aspecto mais interessante da quantização FP8 é como ela equilibra precisão e eficiência. Em partes do modelo onde pequenas variações numéricas são cruciais, o fator de escala $S$ se ajusta para preservar mais precisão. Em outras regiões, onde variações menores não afetam significativamente o resultado, a quantização pode ser mais agressiva. Este comportamento adaptativo permite que o modelo mantenha seu desempenho mesmo com uma representação numérica mais compacta.

Na prática, isso significa que podemos executar modelos complexos em hardware mais simples e tornar o treinamento mais eficiente. O fator de escala dinâmico, determinado pelos gradientes de perda durante o treinamento, garante que esta compressão numérica não comprometa a capacidade de aprendizado do modelo.

### Multi-Token Prediction (MTP)

O mecanismo *Multi-Token Prediction* (MTP) representa uma mudança fundamental na forma como os modelos de linguagem geram texto, permitindo a previsão simultânea de múltiplos tokens em vez da tradicional abordagem autorregressiva token por token. Esta inovação é particularmente significativa para tarefas que envolvem raciocínio de contexto longo.

A função de predição fundamental do MTP é governada por uma função de probabilidade condicional dada por:

$$
P(y_t|x) = \prod_{t=1}^T P(y_t|y_{<t},x)
$$

A paralelização reduz o número de passos de inferência de $T$ para $T/k$, acelerando a geração em hardware adequado, sendo $k$ o número de tokens previstos simultaneamente em cada etapa.

Na prática é mais, ou menos, assim: imagine que estamos tentando gerar a frase "O gato preto dormiu no sofá". Em uma abordagem tradicional autorregressiva, teríamos:

$$
\begin{align*}
P(\text{"O gato preto dormiu no sofá"}) &= P(\text{"O"}) \times \\
&P(\text{"gato"}|\text{"O"}) \times \\
&P(\text{"preto"}|\text{"O gato"}) \times \\
&P(\text{"dormiu"}|\text{"O gato preto"}) \times \\
&P(\text{"no"}|\text{"O gato preto dormiu"}) \times \\
&P(\text{"sofá"}|\text{"O gato preto dormiu no"})
\end{align*}
$$

Com MTP usando $k=2$ (prevendo dois tokens por vez), teríamos:

$$
\begin{align*}
P(\text{"O gato preto dormiu no sofá"}) &= P(\text{"O gato"}) \times \\
&P(\text{"preto dormiu"}|\text{"O gato"}) \times \\
&P(\text{"no sofá"}|\text{"O gato preto dormiu"})
\end{align*}
$$

A eficiência computacional vem do fato de que, em vez de fazer $6$ passos de inferência, fazemos apenas $3$. Para entender melhor, podemos fazer uma analogia com um processo de tradução humana:

Imagine um tradutor experiente trabalhando em um texto. Um tradutor iniciante pode precisar traduzir palavra por palavra, similar à abordagem autorregressiva tradicional. No entanto, um tradutor experiente frequentemente processa *chunks* ou grupos de palavras simultaneamente, similar ao MTP[^3]. Esta capacidade de processar múltiplos tokens simultaneamente não apenas acelera o processo, mas também pode capturar melhor as dependências semânticas entre palavras próximas.

A implementação do MTP envolve um mecanismo de atenção modificado que permite que o modelo:

1. Mantenha a coerência entre os tokens gerados simultaneamente;
2. Preserve as dependências contextuais importantes;
3. Balanceie a velocidade de geração com a qualidade do texto;

Para garantir a qualidade da geração paralela, o modelo utiliza uma função de perda especial que podemos definir como:

$$
L_{MTP} = -\log P(Y|X) + \alpha L_{consistency}
$$

Na qual $L_{consistency}$ é um termo de regularização que penaliza inconsistências entre tokens gerados simultaneamente, e $\alpha$ é um hiperparâmetro que controla a importância deste termo. Esta é uma versão simplificada; implementações reais podem usar mecanismos de atenção especializados para garantir coerência.

O termo de consistência pode ser calculado como:

$$
L_{consistency} = \sum_{i=1}^{k-1} \|h_i - h_{i+1}\|^2
$$

Neste caso, $h_i$ representa as representações ocultas dos tokens gerados em paralelo.

Para ilustrar o impacto na eficiência, considere um modelo processando um texto de $1000$ tokens:

- Abordagem tradicional: $1000$ passos de inferência;
- MTP com $k=4$: $250$ passos de inferência;
- MTP com $k=8$: $125$ passos de inferência.

No entanto, valores muito altos de $k$ podem levar a degradação na qualidade do texto gerado. O truque está em encontrar um equilíbrio entre velocidade e qualidade.

Similar ao MoE, que otimiza o uso de recursos através da especialização, o MTP otimiza através da paralelização, demonstrando como diferentes estratégias de otimização podem trabalhar em conjunto para criar modelos mais eficientes e capazes.

## Pipeline de Treinamento: da Pré-Treinamento ao Raciocínio

O DeepSeek-R1 emprega um pipeline multi-estágio projetado para maximizar suas capacidades de raciocínio enquanto minimiza custos computacionais. Este processo consiste em estágios distintos, cada um guiado por funções de perda e mecanismos de recompensa específicos para a tarefa.

Nós vamos estudar este treinamento em duas fases distintas.

### Estágio 1: Cold Start com Ajuste Fino Supervisionado (SFT)

O DeepSeek-R1 começa ajustando o modelo V3-Base com exemplos de alta qualidade de *Chain of Thought (CoT)*. Estes exemplos são cuidadosamente curados usando prompting com poucos exemplos, anotação manual e refinamento das saídas do DeepSeek-R1-Zero.

#### Chain of Thought (CoT) e Zero-Shot Chain of Thought

O mecanismo *Chain of Thought* (CoT) representa uma evolução na forma como os modelos de linguagem abordam problemas complexos. Em vez de tentar chegar diretamente a uma resposta, o CoT introduz um processo de raciocínio explícito e passo a passo que mimetiza o pensamento humano. Esta abordagem é particularmente poderosa para tarefas que exigem raciocínio matemático, lógico ou multi-etapas. Áreas do pensamento também conhecidas como: tudo.

Um vislumbre da matemática do CoT deve ajudar a entender como ele funciona. A probabilidade de gerar uma resposta correta $y$ para uma entrada $x$ usando CoT pode ser expressa como:

$$
P(y|x) = \sum_{z \in Z} P(y|z,x)P(z|x)
$$

Nesta equação, $z$ representa os passos intermediários do raciocínio, como quando escrevemos nosso pensamento em um papel ao resolver um problema. Sendo assim, o conjunto $Z$ contém todos os possíveis caminhos de raciocínio que poderíamos seguir.

Para entender melhor como isso funciona na prática, vamos considerar um problema que todos já enfrentamos em sala de aula:

*Em uma sala há 3 mesas. Em cada mesa há 4 vasos. Em cada vaso há 2 flores. Quantas flores há no total?*

Um modelo tradicional poderia tentar pular direto para a resposta. Já com CoT, o processo é composto quase naturalmente:

$$
\begin{align*}
z_1 &: \text{"Primeiro, vamos contar os vasos:"} \\
z_2 &: \text{"3 mesas × 4 vasos = 12 vasos no total"} \\
z_3 &: \text{"Agora, vamos contar as flores:"} \\
z_4 &: \text{"12 vasos × 2 flores = 24 flores no total"} \\
y &: \text{"24 flores"}
\end{align*}
$$

Em cada passo deste processo, o modelo calcula a probabilidade do próximo passo baseado em todos os passos anteriores:

$$
P(z_t|z_{<t},x) = \text{softmax}(f_\theta(z_{<t},x))
$$

Na qual, $f_\theta$ é a função do modelo, parametrizada por $\theta$, que decide qual deve ser o próximo passo do raciocínio.

Para garantir que este raciocínio seja não apenas correto, mas também coerente, o modelo utiliza uma função de perda especialmente projetada:

$$
L_{CoT} = -\left ( \log P(y|x) + \alpha \sum_{t=1}^T \log P(z_t|z_{<t},x)\right )
$$

O hiperparâmetro $\alpha$, característico do DeepSeek-R1, funciona como um professor ajustando o peso entre chegar à resposta certa ($y$) e mostrar o trabalho de forma clara ($z_t$).

A eficácia desta abordagem pode ser medida de várias formas complementares:

1. Precisão da resposta final ($A_f$):

   $$A_f = \frac{\text{Respostas corretas}}{\text{Total de problemas}}$$

2. Coerência dos passos intermediários ($C_z$):

   $$C_z = \frac{1}{T}\sum_{t=1}^T \text{score}(z_t|z_{<t})$$

   Usando embeddings de texto para medir quão bem cada passo segue do anterior.

3. Complexidade do raciocínio ($R_c$):

   $$R_c = \log_2(|Z|)$$

   Que nos diz quão sofisticado é o processo de pensamento. Na prática, Como $Z$ é o espaço de raciocínios e, sendo assim, intratável, podemos usar métricas diferentes como *número médio de passos*, ou *diversidade de caminhos*[^1].

Um passo evolutivo, quase óbvio, deste conceito é o *Zero-Shot Chain of Thought* (Zero-Shot-CoT). Esta variante elimina a necessidade de exemplos específicos de demonstração, confiando apenas em prompts simples como *Vamos pensar sobre isso passo a passo:*. O que se parece muito com a forma como um professor ensina a organização do pensamento para a solução de problemas.

A probabilidade neste contexto se torna:

$$
P(y|x,p) = \sum_{z \in Z} P(y|z,x,p)P(z|x,p)
$$

Aqui, $p$ é o prompt que induz o comportamento de raciocínio passo a passo.

O processo do Zero-Shot-CoT pode ser dividido em duas fases principais:

1. Geração do raciocínio:

    $$P(z|x,p) = \prod_{t=1}^T P(z_t|z_{<t},x,p)$$

2. Extração da resposta:

   $$P(y|z,x,p) = f_\theta(z,x,p)$$

Uma das inovações mais interessantes do Zero-Shot-CoT é o conceito de *prompts de auto-consistência*, onde vários caminhos de raciocínio são gerados e agregados:

$$
y_{final} = \text{mode}\{y_i: y_i = f(z_i), z_i \sim P(z|x,p)\}
$$

Este mecanismo funciona como um comitê de raciocínio, onde diferentes linhas de pensamento são consideradas antes de chegar a uma conclusão.

Sempre existem trocas que precisam ser feitas. No Zero-Shot-Cot o custo computacional aumenta proporcionalmente ao detalhamento do raciocínio:

$$
F = \frac{\text{tokens}_{CoT}}{\text{tokens}_{base}} \approx \frac{\sum_{t=1}^T |z_t| + |y|}{|y|}
$$

$|z_t|$ é o comprimento de cada passo intermediário e $|y|$ é o comprimento da resposta final.

Comparando as abordagens em problemas matemáticos reais:

- Modelos tradicionais: 45% de precisão
- Zero-Shot-CoT: 68% de precisão
- CoT com exemplos: 75% de precisão

Assim como um estudante primeiro aprende a mostrar seu trabalho com exemplos guiados, CoT tradicional, e eventualmente desenvolve a capacidade de estruturar seu próprio raciocínio, Zero-Shot-CoT, os modelos de linguagem estão desenvolvendo formas cada vez mais sofisticadas de pensar e explicar seu pensamento. Há aqui um veio mais valioso que nióbio para o futuro da inteligência artificial, que poucos conseguirão explorar.

#### Cold Start

O conceito de *cold start* (cold start) representa um desafio fundamental em sistemas de aprendizado de máquina. Em sistemas de recomendação, este problema manifesta-se quando precisamos fazer sugestões para novos usuários ou itens sem histórico. Nos Modelos baseados em *Large Language Models* (**LLM**), como o DeepSeek-R1, o desafio é análogo mas distinto: precisamos inicializar o modelo com capacidades de raciocínio estruturado e legível antes do treinamento com *Reinforcement Learning*.

A matemática por trás do cold start em sistemas de recomendação pode ser expressa como:

$$
P(r_{ui}|h) = f_\theta(\text{user}_u, \text{item}_i, h)
$$

Na qual, $r_{ui}$ representa a avaliação prevista do usuário $u$ para o item $i$, dado o histórico limitado $h$.

No contexto do DeepSeek-R1, a formulação análoga seria:

$$
P(y|x,c) = g_\phi(\text{input}_x, \text{context}_c)
$$

Onde $y$ é a resposta gerada, $x$ é a entrada, e $c$ representa o contexto de treinamento inicial.

Para entender melhor como o cold start funciona em modelos de linguagem, vamos considerar um exemplo prático. Imagine um modelo tentando resolver um problema matemático:

*Entrada: Calcule a área de um círculo com raio 5.*

Sem cold start adequado, o modelo poderia gerar:

$$area = 553.14 = 78.5$$

Com cold start estruturado, teríamos:

$$
\begin{align*}
z_1 &: \text{A área de um círculo é dada por } \pi r^2\text{"} \\
z_2 &: \text{Substituindo } r = 5\text{:} \\
z_3 &: \text{Área } = \pi \cdot 5^2 = \pi \cdot 25 \approx 78.54 \\
y &: \text{A área do círculo é aproximadamente 78.54 unidades quadradas}
\end{align*}
$$

O processo de cold start é governado por uma função de perda específica:

$$
L_{cold} = -\sum_{i=1}^n \log P_\theta(o_i|q,\{o_1,\ldots,o_{i-1}\}) + \alpha L_{structure}
$$

Nesta equação temos:

- $o_i$ é o i-ésimo token na sequência de saída;
- $q$ é a consulta de entrada;
- $L_{structure}$ penaliza desvios do formato desejado;
- $\alpha$ controla o peso da penalidade estrutural.

O impacto do cold start pode ser medido através de várias métricas:

1. Legibilidade ($R_s$):

   $$R_s = \frac{1}{N}\sum_{i=1}^N \text{score}(\text{structure}_i)$$

2. Consistência de formato ($F_c$):

   $$F_c = \frac{\text{saídas bem formatadas}}{\text{total de saídas}}$$

3. Qualidade do raciocínio ($Q_r$):

   $$Q_r = \beta R_s + (1-\beta)F_c$$

Nenhuma destas técnicas, em nenhum dos artigos que eu consegui encontrar, está diretamente ligada ao DeepSeek-R1. Isso quer dizer que este é outro ponto que precisa de mais atenção e pesquisa.

Uma inovação particular do DeepSeek-R1 é a estruturação do formato de saída:

```xml
<reasoning_process>
  Passo 1: Identificar a fórmula da área do círculo
  Passo 2: Substituir o valor do raio
  Passo 3: Calcular o resultado
</reasoning_process>
<summary>
  A área do círculo é 78.54 unidades quadradas
</summary>
```

A numeração de passos, não aparece mais. Ainda assim, esta estruturação explícita, associada ao treinamento inicial cuidadoso, parece ter resultado em melhorias significativas. Não é difícil encontrar estes resultado nos artigos publicados pela DeepSeek. Em termos de legibilidade, foi observado um aumento dramático de $45%$ para $92%$, refletindo uma melhora substancial na clareza das saídas do modelo. A consistência de formato, igualmente importante, evoluiu de $38%$ para impressionantes 95%, demonstrando que o modelo aprendeu a manter uma estrutura coerente em suas respostas. Quanto à precisão do raciocínio, elemento fundamental para aplicações práticas, foi  registrado um avanço de 52% para $88%$.

No entanto, existe um balanceamento, uma troca que precisa ser considerada, entre a rigidez estrutural e a flexibilidade de expressão. Cuja relação pode ser expressa matematicamente como:

$$
T = \gamma S + (1-\gamma)E
$$

Nesta equação, $S$ representa o grau de aderência à estrutura predefinida, $E$ quantifica a capacidade de expressão flexível do modelo, e $\gamma$ atua como um parâmetro de balanceamento que permite ajustar a importância relativa destes dois aspectos.

O início a frio em **LLM**, em vez de simplesmente preencher lacunas de dados, busca estabelecer um framework robusto para o raciocínio estruturado. Não é fácil resistir a uma analogia com o processo educacional, onde primeiro estabelecemos fundamentos sólidos de resolução de problemas antes de expor o estudante a desafios mais complexos. E, pelo que podemos ver no mercado, está funcionando.

### Estágio 2: *Reinforcement Learning* - Evoluindo Através da Experiência

O *Reinforcement Learning* (RL) é o coração pulsante do DeepSeek-R1, representando uma mudança evolucionária na forma como os modelos de linguagem natural desenvolvem suas capacidades de raciocínio. Em vez de depender apenas de dados cuidadosamente curados por humanos, o modelo evolui através de um processo orgânico de tentativa e erro, similar a como nós humanos aprendemos através da experiência.

>Eu não queria dizer isso. Mas, eu te disse!
>
>Mentira: queria sim.

Para entender profundamente como o RL funciona, precisamos primeiro compreender sua estrutura matemática. O processo pode ser formalizado como uma decisão de Markov (MDP), definida pela tupla:

$$(S, A, P, R, \gamma)$$

Nesta estrutura, $S$ representa o espaço de estados possíveis do modelo, $A$ é o conjunto de todas as ações que o modelo pode tomar, $P$ captura a dinâmica de como o ambiente responde a essas ações, $R$ é a função que determina as recompensas, e $\gamma$ é um fator que equilibra a importância de recompensas imediatas versus futuras.

O objetivo fundamental do modelo é desenvolver uma política ótima $\pi^*$ que maximize a soma descontada de recompensas futuras:

$$
\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]
$$

Em outras palavras, o modelo busca o máximo possível de recompensas, assim como eu, e você. Este processo está formalmente implementado no DeepSeek-R1 por meio de um sistema sofisticado de recompensas que combina três aspectos fundamentais:

$$
R_{total} = \alpha R_{precision} + \beta R_{format} + \gamma R_{coherence}
$$

Cada componente desta equação tem um propósito específico. A recompensa de precisão ($R_{precision}$) funciona como um professor rigoroso, avaliando a correção objetiva das respostas:

$$
R_{precision} = \frac{1}{N}\sum_{i=1}^N \text{score}(y_i, y_i^*)
$$

Para ilustrar como isto funciona na prática, considere um problema clássico de programação - a implementação da sequência de Fibonacci. Que em python poderia ser resolvido por:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

O modelo recebe recompensas baseadas em testes específicos. Para cada teste da função fibonacci, o modelo é avaliado pela correção de sua resposta:

$$
\begin{align*}
\text{test}_1 &: \text{fibonacci}(0) = 0 \quad \checkmark \\
\text{test}_2 &: \text{fibonacci}(1) = 1 \quad \checkmark \\
\text{test}_3 &: \text{fibonacci}(5) = 5 \quad \checkmark
\end{align*}
$$

Porém, a precisão por si só não é suficiente para desenvolver um modelo verdadeiramente capaz. Por isso, foi introduzido a recompensa de formato ($R_{format}$), que atua como um professor de redação, assegurando que o raciocínio seja bem estruturado e claro:

$$
R_{format} = w_1C(z) + w_2S(z)
$$

Nesta equação estão combinados dois aspectos: a completude do raciocínio, representada por $C(z)$, e a estrutura sintática, capturada por $S(z)$. Os pesos $w_1$ e $w_2$ permitem ajustar a importância relativa de cada aspecto. Você pode fazer um paralelo com um professor que pode enfatizar ora a profundidade do argumento, ora a clareza da apresentação. Para ilustrar, considere este exemplo de um raciocínio bem estruturado:

```xml
<reasoning_process>
1. Primeiro, identificamos o caso base (n ≤ 1)
2. Para outros valores, aplicamos a recursão
3. A função chama a si mesma com (n-1) e (n-2)
</reasoning_process>
<answer>
Função fibonacci implementada com recursão
</answer>
```

A terceira dimensão do aprendizado é capturada pela recompensa de coerência ($R_{coherence}$). Novamente, podemos voltar a metáfora do professor. Desta feita, como um professor de lógica experiente. Este componente irá assegurar que cada passo do raciocínio flua naturalmente para o próximo:

$$
R_{coherence} = \frac{1}{T-1}\sum_{t=1}^{T-1} \text{sim}(z_t, z_{t+1})
$$

O processo de aprendizado em si se desenrola como uma dança cuidadosamente coreografada de ajustes sutis em torno destas recompensas, que podemos expressar matematicamente como:

$$
\theta_{t+1} = \theta_t + \eta \nabla_\theta \mathbb{E}[R_{total}]
$$

Para dimensionar os resultados do sistema de recompensas precisamos recorrer, novamente, aos dados divulgados: em tarefas matemáticas, foi observada uma evolução significativa, com uma precisão inicial de $65%$, o modelo alcança impressionantes $91%$ após $100$ mil iterações de aprendizado. Na qualidade do código gerado foi reportado um progresso similar, com a taxa de testes bem-sucedidos saltando de $55%$ para $89%$. Talvez ainda mais impressionante seja o desenvolvimento da capacidade de estruturação do raciocínio reportada, que atinge níveis quase humanos - aumenta de $72%$ para $96%$, enquanto a coerência lógica evolui de $68%$ para $94%$.

>Finalmente, acho que escrevi umas trinta páginas só para escrever este último parágrafo.

A abordagem holística ao aprendizado espelha profundamente o modo como nós, humanos, desenvolvemos expertise em áreas complexas. Assim como um músico deve equilibrar técnica e expressividade, ou um escritor deve balancear clareza e estilo (estou trabalhando nisso), o DeepSeek-R1 aprende a harmonizar precisão, estrutura e coerência em seu raciocínio. O resultado é um nível de elegância e clareza que torna seu raciocínio não apenas correto, mas genuinamente compreensível para os humanos com quem interage. Pelo menos, esta é a sensação que tive nestes últimos $9$ ou $10$ dias desde que troquei o [Qwen 2](https://qwen2.org/) pelo DeepSeek.

## Otimização de Política Relativa em Grupo (**GRPO**)

A Otimização de Política Relativa em Grupo (**GRPO**) representa uma inovação fundamental na arquitetura do DeepSeek-R1, introduzido no *DeepMath*, é uma alternativa elegante e eficiente aos métodos tradicionais de otimização de política como PPO (*Proximal Policy Optimization*) e DPO (*Direct Preference Optimization*). Para entender como o **GRPO** funciona, vamos começar com sua formulação matemática:

No coração do **GRPO** está uma função objetivo projetada que equilibrar múltiplos objetivos concorrentes:

$$
J_{GRPO}(\theta) = E_{q\sim P(Q),\{o_i\}^G_{i=1}\sim \pi_{\theta_{old}}(O|q)}[\frac{1}{G}\sum_{i=1}^G \min(\rho_i A_i, clip(\rho_i,1-\epsilon,1+\epsilon)A_i) - \beta D_{KL}(\pi_\theta\|\pi_{ref})]
$$

Esta expressão incorpora vários componentes-chave que trabalham em harmonia:

1. **Taxa de Verossimilhança ($\rho_i$)**: a taxa de verossimilhança funciona como um medidor fundamental que compara a probabilidade de gerar uma saída $o_i$ sob a nova política versus a política antiga:

    $$
    \rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}
    $$

    Esta razão atua como um detector sensível de mudanças na política. Quando $\rho_i > 1$, significa que a nova política atribui maior probabilidade à saída $o_i$ em comparação com a política antiga.

2. **Função de Vantagem ($A_i$)**: a função de vantagem introduz um mecanismo sofisticado de normalização que avalia a qualidade relativa das saídas dentro de um grupo:

    $$
    A_i = \frac{r_i - \mu_r}{\sigma_r}
    $$

    Nesta equação temos:

    - $r_i$ representa a recompensa para a saída $i$;
    - $\mu_r = mean(r_1,...,r_G)$ é a média das recompensas do grupo;
    - $\sigma_r = std(r_1,...,r_G)$ é o desvio padrão das recompensas do grupo.

    Esta normalização serve a dois propósitos:

    - cria uma escala significativa para comparar saídas em diferentes contextos;
    - ajuda a estabilizar o treinamento reduzindo o impacto de variações na escala das recompensas.

3. **Mecanismo de Clipping**: a operação de clipping, expressa como $clip(\rho_i,1-\epsilon,1+\epsilon)$, implementa uma estratégia conservadora de atualização de política:

    $$
    clip(\rho_i,1-\epsilon,1+\epsilon) = \begin{cases}
    1-\epsilon & \text{se } \rho_i < 1-\epsilon \\
    \rho_i & \text{se } 1-\epsilon \leq \rho_i \leq 1+\epsilon \\
    1+\epsilon & \text{se } \rho_i > 1+\epsilon
    \end{cases}
    $$

4. **Penalidade de Divergência KL**: o termo de divergência de Kullback-Leibler fornece uma camada adicional de estabilidade:

    $$
    -\beta D_{KL}(\pi_\theta\|\pi_{ref})
    $$

> A divergência de Kullback-Leibler, denotada como $D_{KL}(P\|Q)$, é uma medida fundamental em teoria da informação e aprendizado de máquina que quantifica a diferença entre duas distribuições de probabilidade $P$ e $Q$. Matematicamente expressa como $D_{KL}(P\|Q) = \sum_x P(x)\log(\frac{P(x)}{Q(x)})$, ela pode ser interpretada como o "custo" em bits de informação quando usamos a distribuição $Q$ para aproximar a distribuição verdadeira $P$. No contexto do **GRPO**, ela atua como uma "professora paciente" que gentilmente impede o modelo de se desviar muito drasticamente de uma política conhecida e estável ($\pi_{ref}$), funcionando como um mecanismo de estabilização que promove mudanças graduais e controladas no comportamento do modelo. É importante notar que $D_{KL}$ não é simétrica, ou seja, $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$, uma característica que a torna particularmente útil em contextos onde queremos manter uma direção específica de influência entre as distribuições.

### Implementação Prática

A implementação do **GRPO** no DeepSeek-R1 pode ser visualizada através do seguinte pseudocódigo em python:

```python
def calcular_atualizacao_politica(politica_atual, politica_antiga, saidas, recompensas, epsilon=0.2, beta=0.01):
   # Calcula as taxas de verossimilhança
   rhos = politica_atual.prob(saidas) / politica_antiga.prob(saidas)
   
   # Calcula as vantagens
   vantagens = normalizar_recompensas(recompensas)
   
   # Aplica clipping e penalidade KL
   objetivo_clipado = torch.min(
       rhos * vantagens,
       torch.clamp(rhos, 1-epsilon, 1+epsilon) * vantagens
   )
   
   penalidade_kl = beta * calcular_divergencia_kl(politica_atual, politica_referencia)
   
   return objetivo_clipado.mean() - penalidade_kl
```

No contexto específico do DeepSeek-R1, pode-se dizer que o **GRPO** é responsável por:

1. **Aprendizado Baseado em Grupos**: em vez de avaliar saídas individuais isoladamente, o **GRPO** processa saídas em grupos, permitindo uma estimativa mais robusta das recompensas e melhor eficiência amostral durante o treinamento.

2. **Atualizações Adaptativas**: a combinação de *clipping* e *divergência KL* cria um mecanismo adaptativo de atualização que ajuda a prevenir mudanças bruscas na política do modelo.

3. **Estabilidade**: o **GRPO** incorpora múltiplos mecanismos de estabilização usados para o treinamento de modelos grandes como o DeepSeek-R1. Assim, vantagens normalizadas reduzem a sensibilidade à escala das recompensas, taxas de verossimilhança *clipadas* previnem atualizações extremas e a penalidade de *divergência KL* mantém a consistência da política de recompensas.

Além disso, podemos creditar a abordagem baseada em grupos alguns benefícios computacionais significativos, incluindo: requisitos reduzidos de memória através do processamento em lotes; computação paralela de vantagens e atualizações e a normalização eficiente de recompensas dentro dos grupos.

## Além do **GRPO**: Métodos de Otimização Em LLMs

Os métodos de otimização em *LLMs** evoluíram significativamente nos últimos anos, cada um trazendo abordagens únicas para o desafio de alinhar o comportamento do modelo com objetivos específicos. Vamos explorar as principais alternativas ao **GRPO**, analisando suas características.

### PPO (Proximal Policy Optimization)

O *PPO* e o **GRPO** são métodos genuínos de *Reinforcement Learning*. Seguem o paradigma clássico de *RL* onde um agente interage com um ambiente, recebe recompensas e ajusta sua política com base nessas recompensas. Eles implementam o ciclo completo de exploração-recompensa-ajuste característico do *RL*. O PPO emergiu como uma das primeiras soluções robustas para otimização de política em **LLMs**. Sua função objetivo pode ser expressa matematicamente como:

$$
L^{PPO}(\theta) = \mathbb{E}_t[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]
$$

Nesta função, $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ representa a razão de probabilidade entre a política atual e antiga.

O PPO se destaca pela sua estabilidade, mas enfrenta desafios complexos em ambientes distribuídos, principalmente devido à necessidade de sincronização frequente entre diferentes instâncias de treinamento.

### DPO (Direct Preference Optimization)

O DPO é um método de aprendizado supervisionado com preferências. Embora ele use conceitos inspirados em *RL*, como a otimização de política, ele não segue o ciclo tradicional de *RL*. Em vez disso, trabalha diretamente com dados rotulados de preferências humanas. O DPO introduz uma abordagem mais direta para otimização, baseando-se em preferências humanas explícitas. Sua formulação matemática central é:

$$
L^{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[\log\sigma(r_\theta(x,y_w) - r_\theta(x,y_l))]
$$

Neste caso, $y_w$ e $y_l$ representam, respectivamente, as respostas preferidas e não preferidas, e $r_\theta$ é a função de recompensa aprendida.

### KTO (Kahneman-Tversky Optimization)

O KTO é um método híbrido. Embora use funções de valor similares às encontradas em *RL*, sua abordagem é mais próxima de um método de otimização direta baseado em utilidade. Ele incorpora **princípios da economia comportamental**[^1] para modelar preferências humanas, mas não segue o paradigma tradicional de *RL*.

O KTO representa uma inovação ao incorporar princípios da economia comportamental. Sua função de valor adaptada segue a forma:

$$
V(x) = \begin{cases}
x^\alpha & \text{se } x \geq 0 \\
-\lambda(-x)^\beta & \text{se } x < 0
\end{cases}
$$

Nesta função, $\alpha$, $\beta$ e $\lambda$ são parâmetros que modelam a assimetria na percepção de ganhos e perdas.

### APO (Anchored Preference Optimization)

O APO, assim como o DPO, é mais um método de otimização supervisionada do que *RL* propriamente dito. Ele trabalha com pares contrastantes e usa técnicas de ancoragem para manter a estabilidade durante o treinamento, mas não implementa o ciclo exploração-recompensa característico do *RL*.

O APO introduz uma família de objetivos contrastantes que consideram explicitamente a relação entre o modelo e o conjunto de dados de preferência. Sua formulação matemática para o APO-zero é:

$$
L^{APO}(\theta) = \mathbb{E}_{(x,y_+,y_-)\sim \mathcal{D}}[\log\frac{p_\theta(y_+|x)}{p_\theta(y_-|x)} - \alpha\|p_\theta - p_{\text{ref}}\|_2^2]
$$

Neste caso, $p_{\text{ref}}$ é uma distribuição de referência e $\alpha$ controla a força da ancoragem.

### Análise Comparativa

Cada método apresenta características únicas que o tornam mais adequado para cenários específicos:

1. **Requisitos de Dados**:

   - PPO e **GRPO** requerem apenas um modelo de recompensa;
   - DPO e APO necessitam de dados de preferência emparelhados;
   - KTO funciona com feedback binário simples.

2. **Eficiência Computacional**:

   - **GRPO** se destaca pela eliminação da rede crítica;
   - PPO pode ser computacionalmente intensivo;
   - DPO e APO oferecem bom equilíbrio entre complexidade e desempenho;

3. **Estabilidade de Treinamento**:

   - APO e **GRPO** fornecem maior estabilidade;
   - PPO pode ser instável em configurações distribuídas;
   - KTO e DPO mantêm estabilidade moderada.

4. **Qualidade das Saídas**:

   - APO demonstra resultados superiores em benchmarks desafiadores;
   - **GRPO** excele em tarefas de raciocínio;
   - KTO e DPO mostram forte alinhamento com preferências humanas.

A escolha entre estes métodos frequentemente depende do contexto específico de aplicação, recursos computacionais disponíveis e requisitos de qualidade das saídas. No caso do DeepSeek-R1, a adoção do **GRPO** representa uma escolha equilibrada que prioriza eficiência computacional e qualidade de raciocínio, embora cada uma das alternativas apresente vantagens específicas que podem ser valiosas em diferentes contextos de aplicação.

## Algumas Observações Particulares

O modelo parece ter desenvolvido a capacidade de revisitar e revisar etapas intermediárias durante a resolução de problemas complexos. Este processo de reflexão permite que o modelo avalie criticamente seu próprio raciocínio e faça ajustes quando necessário. Eu, ainda não tinha visto isso. O o1 da OpenAi não mostra todo o raciocínio, mas na parte que ele mostra não aparece qualquer revisão de etapas.

O DeepSeek-R1 demonstrou a habilidade de identificar e corrigir erros em tempo real durante seu processo de raciocínio. Novamente, não tinha visto isso em nenhum outro modelo. Verdade seja dita, no dia que a OpenAi lançou o modelo de US$200,00 por mês, eu cancelei minha conta. Ou seja, pode ser que exista e eu não tenha visto.

Outra coisa interessante que podemos observar lendo as iterações do modelo é que, eventualmente ele para, por um tempo maior e, de repente descobre a resposta correta. O artigo reforça que este comportamento surge de forma espontânea através da interação do modelo com o ambiente de **Reinforcement Learning**, demonstrando a sua capacidade de melhorar a resolução de problemas de forma autônoma.

Finalmente, para quem gosta das técnicas de alta-performance, quando largamos as linguagens de programação e a compilação padrão e tiramos suco de pedra, tudo indica que o ganho de performance e a redução do custo estejam relacionados ao uso desenfreado do [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/). 

O PTX (*Parallel Thread Execution*) é uma representação intermediária entre linguagens de alto nível, como CUDA C/C++, e o código de máquina (SASS - Shader ASSembly) executado pela GPU. Gerado pelo compilador NVCC, ele permite otimizações granulares, como ajuste de registradores e organização de threads, que não são viáveis diretamente em CUDA. Para entender, imagine que o CUDA é como escrever um texto em português, o PTX é uma tradução para o inglês, próximo mas ainda não final, e o SASS é a versão em alemão, código de máquina específico. O PTX funciona como uma 'Assembly portável', oferecendo controle detalhado sem perder compatibilidade entre arquiteturas de GPU. Assim, ele está muito mais próximo do Assembly do que de linguagens como Python. Se não entendeu, isso quer dizer que, dá para mexer no sistema com unhas e dentes.  

## Questões em Aberto

Algumas questões estão abertas servido para fomentar um turbilhão de hipóteses na Xfere. Três me chamam a atenção:

1. **Coleta de Dados**: como foram curados os conjuntos de dados específicos para raciocínio? Compreender as fontes e critérios de seleção de dados permite replicar e melhorar o desempenho do modelo;

2. **Treinamento do Modelo**: nenhum código de treinamento foi liberado pela DeepSeek, deixando incertezas sobre quais hiperparâmetros funcionam melhor e como eles diferem entre famílias e escalas de modelos;

3. **Leis de Escala**: quais são as relações entre os custos de computação e dados no treinamento de modelos de raciocínio? Precisamos conhecer estas relações para otimizar outros modelos.

4. **Destilação**: em modelos de linguagem grandes (LLMs) é uma técnica que visa transferir o conhecimento de um modelo maior e mais complexo (o "professor") para um modelo menor e mais eficiente (o "aluno"). É como se estivéssemos condensando a sabedoria de um especialista em um manual mais conciso, mas igualmente útil. Essa é uma técnica relativamente corrente no desenvolvimento de modelos **LLMs**. O ponto importante aqui é a profundidade desta destilação e o que será considerado justo, ou não.

**Finalmente**: todo este texto foi escrito com ferramentas de inteligência artificial para busca, formatação e revisão: [notebookllm](https://notebooklm.google.com/), [deepseek](https://chat.deepseek.com/), [gemini](https://gemini.google.com/), [claude](https://claude.ai/) e [qwen2.5](https://chat.qwenlm.ai/).

## Referências

1. **DeepSeek-R1: Incentivizing Reasoning Capability in **LLM** via *Reinforcement Learning***

    - arXiv:2501.12948
    - [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

2. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**

    - arXiv:2402.03300
    - [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

3. **DeepSeek-V3 Technical Report**
    - arXiv:2412.19437
    - [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

4. **Open-R1: a fully open reproduction of DeepSeek-R1**
    - Hugging Face Blog
    - [https://huggingface.co/blog/open-r1](https://huggingface.co/blog/open-r1)

5. **Proximal Policy Optimization Algorithms**
    - arXiv:1707.06347v2
    - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

6. **Reformer: The Efficient Transformer**
    - arXiv:2001.04451v1
    - [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)

7. **Group Robust Preference Optimization in Reward-free RLHF**
   - arXiv:2405.20304  
   - [https://arxiv.org/abs/2405.20304](https://arxiv.org/abs/2405.20304)

8. **Attention is All you Need**
    - arXiv:1706.03762
    - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

9. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
    - arXiv:2201.11903
    - [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

10. **Large Language Models are Zero-Shot Reasoners**
    - arXiv:2205.11916
    - [https://arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916)

11. **Secrets of RLHF in Large Language Models Part I: PPO**
    - arXiv:2307.04964
    - [https://arxiv.org/abs/2307.04964](https://arxiv.org/abs/2307.04964)

[^1]: isso é um problema porque a maioria dos modelos usa alguma técnica de web-crawling para recuperar dados da internet. Mais, ou menos, usando a ideia que deu origem ao Google. A maior parte do dado recolhido desta forma tem baixa qualidade. Eu tenho uma, ou duas ideias de como melhorar isso, respeitando todos os direitos autorais. Dada a qualidade da resposta do DeepSeek-R1, tem uma pulga atrás da minha orelha gritando. Eles se tocaram! Olha lá, caiu a ficha!

[^2]: acabei de ter uma ideia que vou deixar escrita aqui só para não perder o fio: fuzzy logic.

[^3]: eu sei que você não vai acreditar em mim. Mas, é assim que você lê. Em grupos de palavras. Aliás, todas as técnicas de leitura rápida se baseiam nisso. 