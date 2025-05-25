---
layout: post
title: Transformers, Do Código à Geração
author: Frank (Adaptado por IA)
categories: |-
    disciplina
    Matemática
    artigo
tags: |-
    Matemática
    inteligência artificial
    processamento de linguagem natural
    transformers
    encoder-decoder
    atenção
    BERT
    GPT
image: assets/images/transformer_architecture.webp
featured: false
rating: 5
description: Explorando a arquitetura completa Encoder-Decoder dos Transformers, o bloco decodificador, o processo de treinamento e suas vastas aplicações.
date: 2025-02-17T22:55:34.524Z
preview: Nesta aula, mergulhamos fundo na arquitetura completa dos Transformers. Desvendamos o bloco decodificador, incluindo a atenção mascarada e a atenção cruzada, e como ele interage com o codificador para gerar sequências. Também abordaremos o treinamento e as diversas aplicações desta poderosa arquitetura.
keywords: |-
    transformers
    encoder-decoder
    decoder block
    masked self-attention
    cross-attention
    processamento de linguagem natural
    treinamento de transformers
    BERT
    GPT
toc: true
published: false
lastmod: 2025-05-25T20:09:24.212Z
---

## Navegando a Arquitetura Completa: Do Código à Geração

Já [discutimos atenção]([link-para-aula-anterior](https://frankalcantara.com/transformers-quatro/)), quando desbravamos os mares do bloco codificador dos Transformers. Vimos como a auto-atenção, do inglês self-attention, especialmente em sua forma multi-cabeça, do inglês  multi-head, permite que o modelo pese a importância de palavras diferentes na sequência de entrada. Exploramos a necessidade da codificação posicional para reintroduzir a ordem das palavras e como as redes *feed-forward* (FFN) processam as representações contextuais geradas. Com esses componentes, *um bloco codificador transforma uma sequência de embeddings de entrada em uma sequência de representações contextuais semanticamente ricas*.

A atenta leitora deve se recordar que o objetivo final de muitos modelos de sequência não é apenas *entender* uma sequência de entrada, mas também *gerar* uma sequência de saída. Pense na tradução automática: uma frase em português (entrada) é transformada em uma frase em inglês (saída). É aqui que a arquitetura completa **Encoder-Decoder** dos Transformers brilha, e o protagonista desta nova etapa da nossa jornada é o **Bloco Decodificador (Decoder Block)**.

Nossa jornada inclui os seguintes destinos:

* Desvendar os componentes do Bloco Decodificador, incluindo suas duas camadas de atenção distintas;

* Entender como múltiplos blocos codificadores e decodificadores são empilhados para formar a arquitetura Transformer completa;

* Obter uma visão geral do processo de treinamento desses modelos complexos;

* Vislumbrar o vasto panorama de aplicações e variações dos Transformers.

Que a perspicaz leitora muna-se de mapas e bússolas. A aventura continua!

## O Bloco Decodificador: Gerando Sequências com Atenção

O bloco decodificador tem a tarefa de gerar a sequência de saída, token por token. Ele opera de maneira **auto-regressiva**, o que significa que a predição de cada token futuro depende dos tokens gerados anteriormente. Por exemplo, ao traduzir "Eu amo pão" para o inglês, após gerar "I", o modelo usará "I" para gerar "love"; depois usará "I love" para gerar "bread".

Para realizar essa tarefa, o bloco decodificador possui uma estrutura ligeiramente mais complexa que o bloco codificador. Um bloco decodificador típico inclui três subcamadas principais, cada uma seguida por uma conexão residual e normalização de camada (Add & Norm), semelhante ao que vimos no encoder:

1.  **Auto-Atenção Multi-Cabeça Mascarada (Masked Multi-Head Self-Attention)**
2.  **Atenção Multi-Cabeça entre Codificador-Decodificador (Encoder-Decoder Multi-Head Attention ou Cross-Attention)**
3.  **Rede Feed-Forward (FFN)**

Vamos analisar cada uma dessas subcamadas.

### 1. Auto-Atenção Multi-Cabeça Mascarada (Masked Multi-Head Self-Attention)

A primeira subcamada do bloco decodificador é uma camada de auto-atenção multi-cabeça. No entanto, há uma modificação fundamental em relação à auto-atenção que vimos no codificador: o **mascaramento**.

**Por que o Mascaramento?**

Como o decodificador opera de forma auto-regressiva, ao prever o token na posição $t$, ele só deve ter acesso aos tokens anteriores (posições $0$ a $t-1$) na sequência de saída que está sendo gerada. Ele não pode "ver o futuro" e usar tokens da posição $t$ ou posteriores. Se permitíssemos isso durante o treinamento, o modelo simplesmente aprenderia a copiar o token alvo da posição $t$, tornando o aprendizado trivial e inútil para a geração real, onde os tokens futuros são desconhecidos.

A máscara garante que a auto-atenção no decodificador só considere as posições anteriores. Isso é implementado adicionando-se uma matriz de máscara aos scores de atenção (antes da função softmax). Essa máscara atribui um valor muito negativo (como $-\infty$) às posições que não devem ser atendidas.

**A Matemática do Mascaramento na Auto-Atenção**

Recordando a equação da atenção escalonada por produto escalar:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

Na auto-atenção mascarada do decodificador, a equação torna-se:

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left( \frac{QK^T + M}{\sqrt{d_k}} \right) V$$

onde $M$ é a matriz de máscara. Para uma sequência de comprimento $L_{out}$ (comprimento da sequência de saída), $M$ é uma matriz $L_{out} \times L_{out}$. Para a predição do token na posição $i$, a máscara $M_{ij}$ é $0$ se $j \le i$ (permitindo atenção à posição atual e anteriores) e $-\infty$ (ou um número negativo muito grande) se $j > i$ (bloqueando atenção a posições futuras).

Quando um valor muito negativo é adicionado aos scores antes do softmax, $e^{\text{score} - \infty} \approx 0$. Assim, os pesos de atenção para as posições futuras mascaradas se tornam efetivamente zero.

**Exemplo Conceitual da Máscara**

Suponha que estamos gerando uma sequência de saída e já temos os tokens $y_0, y_1$. Queremos gerar $y_2$. A camada de auto-atenção mascarada calculará representações contextuais para $y_0, y_1, y_2$ (onde $y_2$ é o token atual sendo processado).

Para a posição $0$ ($y_0$):
* Pode atender a $y_0$. Scores para $y_1, y_2$ são mascarados.

Para a posição $1$ ($y_1$):
* Pode atender a $y_0, y_1$. Scores para $y_2$ são mascarados.

Para a posição $2$ ($y_2$):
* Pode atender a $y_0, y_1, y_2$.

A matriz de máscara $M$ (antes de ser multiplicada por um valor muito negativo) seria algo como:

$$
M = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$
(Esta representação é conceitual; na prática, os $-\infty$ são adicionados aos scores $QK^T$).

Essa camada permite que cada posição na sequência de saída do decodificador atenda às posições anteriores (e a si mesma) na sequência de saída, construindo representações que sabem o que já foi gerado até o momento. As matrizes $Q, K, V$ para esta camada são todas derivadas da sequência de saída do decodificador (ou, mais precisamente, dos embeddings da sequência de saída mais as codificações posicionais).

### 2. Atenção Multi-Cabeça entre Codificador-Decodificador (Cross-Attention)

A segunda subcamada de atenção é onde a mágica da tradução (ou de qualquer tarefa sequência-a-sequência) realmente acontece. É aqui que o decodificador considera a sequência de entrada que foi processada pelo codificador. Esta camada é frequentemente chamada de **cross-attention** (atenção cruzada).

**Função da Cross-Attention**

Enquanto a auto-atenção mascarada permite ao decodificador entender o contexto da sequência de *saída* que ele está construindo, a atenção cruzada permite que ele olhe para a sequência de *entrada* codificada e decida quais partes da entrada são mais relevantes para prever o próximo token da saída.

Por exemplo, ao traduzir a frase em português "O gato sentou no tapete" para o inglês, quando o decodificador está prestes a gerar a palavra "cat", a camada de cross-attention deve permitir que ele preste muita atenção à palavra "gato" na representação da frase de entrada vinda do codificador.

**Matemática da Cross-Attention**

A mecânica da cross-attention é idêntica à da auto-atenção multi-cabeça que já conhecemos:

$$\text{Attention}(Q_D, K_E, V_E) = \text{softmax}\left( \frac{Q_D K_E^T}{\sqrt{d_k}} \right) V_E$$

A diferença fundamental está na origem das matrizes Query (Q), Key (K) e Value (V):

* **Query ($Q_D$)**: Vem da saída da subcamada anterior do decodificador (a camada de auto-atenção mascarada). Ou seja, as queries são baseadas na sequência de saída parcial.
* **Key ($K_E$) e Value ($V_E$)**: Vêm da saída final do *stack de codificadores*. Estas são as representações contextuais da sequência de entrada completa. $K_E$ e $V_E$ são as mesmas para cada etapa de decodificação e para cada bloco decodificador dentro do stack de decodificadores.

A atenta leitora notará que aqui não há necessidade de mascaramento como na auto-atenção do decodificador. O decodificador pode, e deve, ter acesso a todas as partes da sequência de entrada para decidir o que é relevante.

_Figura conceitual (descrição): Um diagrama mostraria o fluxo de dados: A saída da camada de auto-atenção mascarada do decodificador (após Add & Norm) é transformada para gerar as matrizes $Q_D$. As saídas do último bloco do codificador são transformadas (uma única vez antes do processo de decodificação começar, ou de forma idêntica em cada passo/bloco) para gerar as matrizes $K_E$ e $V_E$. Estes são então alimentados na unidade de atenção multi-cabeça._

Esta camada permite que cada token sendo gerado pelo decodificador "consulte" a sequência de entrada codificada e extraia as informações necessárias para a sua própria predição.

### 3. Rede Feed-Forward (FFN)

A terceira subcamada é uma rede feed-forward, idêntica em estrutura àquela usada nos blocos codificadores:

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2 \quad \text{(ou GeLU)}$$

Ela é aplicada independentemente a cada posição da sequência de saída (após a camada de cross-attention e sua respectiva Add & Norm). Sua função é processar adicionalmente as representações obtidas pela atenção cruzada, permitindo transformações mais complexas e não-lineares.

### Montagem do Bloco Decodificador

Assim como no codificador, cada uma dessas três subcamadas (auto-atenção mascarada, cross-attention, FFN) dentro de um bloco decodificador é envolvida por uma conexão residual e uma camada de normalização (Layer Normalization).

O fluxo de dados para um token na posição $t$ através de um bloco decodificador seria:
1.  Entrada: Representação do token $y_t$ da camada anterior do decodificador (ou embedding de $y_t$ + codificação posicional, se for o primeiro bloco).
2.  $x_1 = \text{LayerNorm}(\text{entrada} + \text{MaskedMultiHeadSelfAttention}(\text{entrada}))$
3.  $x_2 = \text{LayerNorm}(x_1 + \text{EncoderDecoderMultiHeadAttention}(Q=x_1, K=K_E, V=V_E))$
4.  Saída do Bloco: $\text{LayerNorm}(x_2 + \text{FFN}(x_2))$

_Figura conceitual (descrição): Um diagrama detalhado do bloco decodificador, mostrando as três subcamadas, as conexões residuais, as camadas de normalização e as entradas/saídas de cada componente, incluindo a entrada $K_E, V_E$ vinda do codificador para a camada de cross-attention._

## A Arquitetura Completa Encoder-Decoder

Agora que entendemos os blocos codificador e decodificador, podemos montar a arquitetura Transformer completa.

A arquitetura original proposta por Vaswani et al. (2017) consiste em:
* Um **stack (pilha) de $N$ blocos codificadores idênticos**. A saída de um bloco codificador é a entrada para o próximo. A entrada para o primeiro bloco é o embedding da sequência de entrada mais a codificação posicional.
* Um **stack (pilha) de $N$ blocos decodificadores idênticos**. A saída de um bloco decodificador é a entrada para o próximo. A entrada para o primeiro bloco é o embedding da sequência de saída (deslocada e com o token de início) mais a codificação posicional.

**Fluxo de Dados Global:**
1.  **Codificação da Entrada**: A sequência de entrada inteira (e.g., "O gato sentou no tapete") é processada pelo stack de $N$ codificadores, resultando em uma sequência de vetores de contexto $\mathbf{C} = (c_1, c_2, ..., c_{L_{in}})$, onde $L_{in}$ é o comprimento da sequência de entrada. Esses vetores $c_i$ são as "Keys" e "Values" ($K_E$ e $V_E$) que serão usados por todas as camadas de cross-attention nos blocos decodificadores.
2.  **Decodificação da Saída (Auto-Regressiva)**: O decodificador gera a sequência de saída token por token.
    * **Passo 1**: O decodificador recebe um token de início especial (e.g., `<SOS>` - Start Of Sequence) como sua primeira entrada. Ele processa esse token através do stack de $N$ decodificadores. A camada de cross-attention em cada bloco decodificador usa $\mathbf{C}$ (saída do encoder).
    * Ao final do stack de decodificadores, a representação do token atual é passada por uma **camada linear final** seguida por uma **função softmax**. A camada linear projeta o vetor de saída do decodificador para a dimensão do vocabulário de saída. O softmax então produz uma distribuição de probabilidade sobre todo o vocabulário, indicando a probabilidade de cada palavra ser o primeiro token real da sequência de saída (e.g., "The").
    * O token com a maior probabilidade (ou um token amostrado dessa distribuição) é escolhido como o primeiro token gerado.
    * **Passo $t$**: Para gerar o $t$-ésimo token da sequência de saída, o decodificador recebe os $t-1$ tokens gerados anteriormente como entrada. Ele os processa através de seus blocos (com auto-atenção mascarada para só ver $y_0, ..., y_{t-1}$). A atenção cruzada novamente consulta $\mathbf{C}$. A camada linear e softmax no final preveem o próximo token $y_t$.
    * Este processo continua até que um token especial de fim de sequência (e.g., `<EOS>` - End Of Sequence) seja gerado, ou até um comprimento máximo de sequência seja atingido.

_Figura conceitual (descrição): Uma visão geral da arquitetura Transformer completa, mostrando o stack de encoders à esquerda, o stack de decoders à direita. As setas indicam o fluxo da sequência de entrada através dos encoders. A saída final do encoder (K, V) é mostrada alimentando cada bloco do decoder na camada de cross-attention. A sequência de saída (Outputs shifted right) é mostrada alimentando a base do decoder, e as predições de probabilidade de saída no topo._

**Camada Linear Final e Softmax**

Após o último bloco decodificador, o vetor de saída para a posição $t$, digamos $\mathbf{d}_t \in \mathbb{R}^{d_{model}}$, é projetado para o tamanho do vocabulário de saída $V_{out}$:

$$\text{Logits}_t = \mathbf{d}_t W_{out} + b_{out}$$

onde $W_{out} \in \mathbb{R}^{d_{model} \times |V_{out}|}$ e $b_{out} \in \mathbb{R}^{|V_{out}|}$. Os $\text{Logits}_t$ são então passados por uma função softmax para obter uma distribuição de probabilidade sobre o vocabulário:

$$P(y_t | y_{<t}, X) = \text{softmax}(\text{Logits}_t)$$

Esta é a probabilidade do próximo token ser cada palavra do vocabulário, dado os tokens de saída anteriores $y_{<t}$ e a sequência de entrada $X$.

## Processo de Treinamento (Visão Geral)

Treinar um modelo Transformer é uma tarefa computacionalmente intensiva que requer grandes volumes de dados (corpus paralelos para tradução, grandes textos para modelagem de linguagem) e recursos de hardware significativos (GPUs/TPUs).

**Função de Perda (Loss Function)**

Para tarefas como tradução automática, o modelo é treinado para maximizar a probabilidade da sequência de saída correta, dado uma sequência de entrada. Isso é tipicamente feito minimizando a **entropia cruzada (cross-entropy)** entre a distribuição de probabilidade prevista pelo modelo e a distribuição real (que é one-hot, ou seja, 1 para o token correto e 0 para todos os outros).

A perda para uma única instância de treinamento (par de sequências entrada $X$ e saída alvo $Y^* = (y_1^*, ..., y_{L_{out}}^*)$) é a soma das perdas de entropia cruzada para cada token na sequência de saída:

$$L(\theta) = - \sum_{t=1}^{L_{out}} \log P(y_t^* | y_{<t}^*, X; \theta)$$

onde $\theta$ representa todos os parâmetros do modelo. $P(y_t^* | y_{<t}^*, X; \theta)$ é a probabilidade que o modelo (com parâmetros $\theta$) atribui ao token alvo correto $y_t^*$ na posição $t$, dados os tokens alvo corretos anteriores $y_{<t}^*$ (ver "Teacher Forcing" abaixo) e a sequência de entrada $X$.

**Teacher Forcing**

Durante o treinamento, em vez de alimentar as próprias previsões anteriores do decodificador para prever o próximo token (o que pode levar a erros que se propagam e dificultam o aprendizado, especialmente no início), é comum usar uma técnica chamada **teacher forcing**.

Com teacher forcing, a entrada para o decodificador em cada passo $t$ é sempre o token alvo *correto* da sequência de treinamento, $y_{t-1}^*$, independentemente do que o modelo previu no passo $t-1$. Isso torna o treinamento mais estável e paralelizável (as predições para todas as posições da sequência de saída podem ser calculadas de uma vez, já que as entradas do decodificador não dependem de suas próprias saídas anteriores).

**Otimização**

A perda $L(\theta)$ é diferenciável em relação aos parâmetros do modelo $\theta$. Portanto, podemos usar algoritmos de otimização baseados em gradiente, como o **Adam (Adaptive Moment Estimation)**, para ajustar os pesos. O gradiente da função de perda é calculado usando o algoritmo de **retropropagação (backpropagation)** através de toda a arquitetura Transformer.

O treinamento envolve processar mini-lotes (mini-batches) de pares de sequências, calcular a perda média para o lote, calcular os gradientes e atualizar os pesos do modelo. Esse processo é repetido por muitas épocas (passes completos sobre o conjunto de treinamento).

**Hiperparâmetros e Regularização**

O treinamento de Transformers envolve o ajuste de muitos hiperparâmetros, como:
* Número de blocos $N$ no encoder e decoder (e.g., $N=6$ no paper original).
* Dimensão do modelo $d_{model}$ (e.g., 512).
* Número de cabeças de atenção (e.g., 8).
* Dimensão da camada interna da FFN $d_{ff}$ (e.g., 2048).
* Taxa de aprendizado, parâmetros do otimizador Adam.
* Técnicas de regularização como dropout (aplicado à saída de cada subcamada antes de ser adicionada à entrada da subcamada e normalizada) para prevenir overfitting.

## Aplicações e Variações (Breve Panorama)

Desde sua introdução, a arquitetura Transformer revolucionou o campo do Processamento de Linguagem Natural e encontrou aplicações em diversas outras áreas.

* **Tradução Automática**: A tarefa original para a qual foi projetado, alcançando resultados estado-da-arte.
* **Modelagem de Linguagem**: Modelos como a série **GPT (Generative Pre-trained Transformer)** da OpenAI são baseados apenas na arquitetura do *decodificador* Transformer. Eles são pré-treinados em vastas quantidades de texto para prever o próximo token em uma sequência. Após o pré-treinamento, podem ser ajustados (fine-tuned) para uma variedade de tarefas de geração de texto, ou usados diretamente para geração zero-shot/few-shot.
* **Compreensão de Linguagem**: Modelos como **BERT (Bidirectional Encoder Representations from Transformers)** do Google utilizam apenas a arquitetura do *codificador* Transformer. BERT é pré-treinado usando objetivos como o *Masked Language Model (MLM)* (prever tokens mascarados aleatoriamente na entrada) e *Next Sentence Prediction (NSP)*. As representações geradas por BERT são altamente contextuais e podem ser usadas como features ou ajustadas para uma ampla gama de tarefas de compreensão de linguagem (NLU), como classificação de texto, resposta a perguntas, reconhecimento de entidades nomeadas.
* **Sumarização de Texto**: Gerar resumos concisos de documentos mais longos.
* **Geração de Diálogo**: Construção de chatbots e agentes conversacionais.
* **Análise de Sentimento**.
* **Além do Texto**: Transformers também foram adaptados para outras modalidades, como visão computacional (Vision Transformer - ViT), processamento de fala e até mesmo biologia (e.g., modelagem de proteínas).

A chave para o sucesso de muitos desses modelos é o paradigma de **pré-treinamento e fine-tuning**. Um modelo Transformer massivo é primeiro pré-treinado em uma tarefa auto-supervisionada em um corpus de dados gigantesco (e.g., prever palavras em texto da web). Isso permite que o modelo aprenda representações ricas e gerais da linguagem. Em seguida, esse modelo pré-treinado pode ser rapidamente adaptado (fine-tuned) com uma quantidade relativamente pequena de dados rotulados para uma tarefa específica, frequentemente alcançando desempenho superior.

## Conclusão da Aula e Próximos Horizontes

Nesta aula, completamos nossa visão geral da arquitetura Transformer padrão, explorando em detalhe o funcionamento do bloco decodificador e como ele interage com o bloco codificador. Vimos como a auto-atenção mascarada e a atenção cruzada são os pilares que permitem ao decodificador gerar sequências de saída coerentes e contextualmente relevantes em relação à entrada. Também tocamos brevemente no processo de treinamento e no impacto transformador desses modelos em diversas aplicações.

A jornada pelo universo dos Transformers é vasta e continua em expansão. A engenhosa leitora pode se aprofundar em:
* Diferentes mecanismos de atenção (e.g., sparse attention, Longformer).
* Técnicas de otimização e escalabilidade para treinar modelos ainda maiores.
* Análise detalhada de modelos específicos como BERT, GPT, T5, etc.
* Considerações éticas e desafios associados a grandes modelos de linguagem.

Com o conhecimento adquirido até aqui, a leitora está bem equipada para navegar por esses tópicos mais avançados e apreciar as contínuas inovações que moldam o futuro da inteligência artificial.

---
**Nota:** As referências bibliográficas e implementações de código C++ para os novos componentes (bloco decodificador, atenção mascarada, cross-attention, arquitetura completa) seriam adicionadas aqui, seguindo o padrão do arquivo original. Dada a complexidade, os exemplos de código C++ para a Aula 2 seriam consideravelmente mais envolvidos, especialmente para uma implementação completa, e poderiam focar em ilustrar os mecanismos chave de forma isolada ou conceitual.