---
layout: post
title: Transformers - Word2Vec, a Ponte para o Contexto
author: frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - Matemática
    - inteligência artificial
    - processamento de linguagem natural
    - word embeddings
    - Word2Vec
    - CBoW
    - SkipGram
    - Negative Sampling
    - Hierarchical Softmax
    - redes neurais
    - atenção
image: assets/images/word2vec_bridge.webp
featured: false
rating: 5
description: Aprofundando no **Word2vec** e suas otimizações, analisando como ele captura a semântica e suas limitações em relação aos modelos contextuais como Transformers.
date: 2025-04-27T18:30:00.000Z
preview: Aprofundando no **Word2vec** (CBoW, SkipGram) e suas otimizações, analisando como ele captura a semântica e suas limitações em relação aos modelos contextuais como Transformers.
keywords: |-
    Transformers
    Word2Vec
    CBoW
    SkipGram
    word embeddings
    representações distribuídas
    redes neurais
    Negative Sampling
    Hierarchical Softmax
    Processamento de Linguagem Natural
    Inteligência Artificial
    Atenção
    Embeddings Estáticos
    Embeddings Contextuais
toc: true
published: false
lastmod: 2025-04-28T14:43:27.693Z
---

## Word2Vec: Aprofundando nas Representações Distribuídas

Nos artigos anteriores, navegamos desde os fundamentos matemáticos até as primeiras tentativas de vetorização como **TF-IDF** e modelos **N-gram**, culminando na introdução dos mecanismos de atenção e uma visão geral dos algoritmos **Word2Vec** (**CBoW** e **SkipGram**) e das redes neurais que os sustentam. Percebemos as limitações das abordagens que ignoram o contexto ou se restringem a vizinhanças locais. Vimos também como as redes neurais rasas podem ser usadas para aprender representações mais ricas.

O **Word2Vec** marcou uma revolução ao propor uma forma eficiente de aprender **embeddings distribuídos**: vetores densos e de baixa dimensionalidade que capturam relações semânticas complexas, baseados na **hipótese distribucional**, o significado de uma palavra emerge do contexto em que ela aparece.

Neste artigo, vamos revisitar e aprofundar o funcionamento do **Word2Vec**, focando nos detalhes do treinamento, nas otimizações que o tornaram prático, nas propriedades dos **embeddings** resultantes e em suas limitações que motivaram o desenvolvimento de modelos contextuais como os Transformers.

## Revisitando **CBoW** e SkipGram: As Duas Faces do Word2Vec

Como vimos em [transformers-cinco]([link_para_transformers_cinco](https://frankalcantara.com/transformers-cinco/)), o **Word2vec** oferece duas arquiteturas principais:

1. **Continuous Bag-of-Words (CBoW)**: pretende prever a palavra-alvo central a partir da média dos vetores das palavras de contexto circundantes. É reconhecidamente eficiente e bom, na literatura, para palavras frequentes.

    * **Entrada**: média dos embeddings das palavras de contexto;
    * **Objetivo**: maximizar $p(w_t \vert Context(w_t))$;
    * **Rede Neural (Simplificada)**: contexto Médio $\rightarrow$ Camada Oculta (Linear) $\rightarrow$ **Softmax** sobre Vocabulário.

2. **SkipGram**: pretende prever as palavras de contexto a partir da palavra-alvo central. É mais lento, mas captura melhor informações sobre palavras raras e relações mais sutis.

    * **Entrada**: Embedding da palavra-alvo;
    * **Objetivo**: Maximizar $\sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} \vert w_t)$;
    * **Rede Neural (Simplificada)**: Palavra Alvo $\rightarrow$ Camada Oculta (Linear) $\rightarrow$ Múltiplos Softmax, um para cada palavra de contexto, sobre Vocabulário.

Ambos os modelos utilizam uma rede neural rasa, como detalhado em [transformers-seis]([link_para_transformers_seis](https://frankalcantara.com/transformers-seis/)), mas o truque é que **não estamos interessados na tarefa de predição em si**. O verdadeiro tesouro são os **pesos aprendidos** na matriz de entrada $W^{(1)}$ (ou $W_{\text{entrada} }$), cujas linhas se tornam os vetores de embedding finais das palavras ($\vec{v}_w$).

É importante que a atenta leitora perceba que **CBoW** e **SkipGram** não são usados simultaneamente, mas representam duas abordagens alternativas dentro do framework Word2Vec. Ambas compartilham o mesmo objetivo fundamental: produzir embeddings distribuídos de alta qualidade, porém atuam em direções opostas. Como podemos ver na Figura 1.

![Diagrama mostrando a relação entre CBOW e SkipGram no framework Word2Vec](/images/word2vec-framework.webp)

_Figura 1: Visão geral do framework Word2Vec destacando a relação entre suas duas arquiteturas alternativas: CBOW (Continuous Bag-of-Words) e SkipGram. O CBOW prevê a palavra-alvo a partir da média dos vetores de contexto, enquanto o SkipGram faz o oposto, prevendo as palavras de contexto a partir da palavra-alvo. Ambas compartilham os mesmos componentes subjacentes e as mesmas técnicas de otimização (Negative Sampling e Hierarchical Softmax), mas diferem na direção da predição, resultando em características distintas em relação ao desempenho com palavras frequentes ou raras._{: class="legend"}

Essa inversão na direção da previsão, aparentemente sutil, resulta em características distintas que tornam cada arquitetura mais adequada para diferentes cenários:

| Cenário | Arquitetura Recomendada | Justificativa |
|---------|------------------------|---------------|
| Palavras frequentes | **CBoW** | O efeito de suavização ao tirar a média de múltiplos contextos beneficia palavras que aparecem frequentemente |
| Palavras raras | **SkipGram** | Cada ocorrência de uma palavra rara gera múltiplos exemplos de treinamento |
| Corpus pequenos | **CBoW** | Melhor generalização quando há poucos exemplos de treinamento |
| Polissemia | **SkipGram** | Melhor capacidade de capturar múltiplos sentidos de palavras |
| Eficiência computacional | **CBoW** | Treinamento mais rápido, especialmente para corpus grandes |
| Tarefas de analogia semântica | **SkipGram** | Superior em capturar relações vetoriais (como no clássico exemplo $\vec{v}_{\text{rei} } - \vec{v}_{\text{homem} } + \vec{v}_{\text{mulher} } \approx \vec{v}_{\text{rainha} }$) |

As duas arquiteturas enfrentam o mesmo desafio computacional: a função **Softmax** na camada de saída requer normalização sobre todo o vocabulário, tornando cada atualização $O( \vert V \vert )$. Para superar esse obstáculo, tanto **CBoW** quanto **SkipGram** podem ser implementados com as mesmas técnicas de otimização:

$$\text{Negative Sampling}: O( \vert V \vert ) \rightarrow O(k), \text{ onde } k \ll  \vert V \vert $$

$$\text{Hierarchical Softmax}: O( \vert V \vert ) \rightarrow O(\log \vert V \vert )$$

Em resumo, a escolha entre **CBoW** e **SkipGram** depende das características específicas da aplicação, sendo complementares em suas forças e fraquezas. O **Word2Vec**, se visto como  um framework, oferece essa flexibilidade permitindo selecionar a arquitetura mais adequada para cada caso de uso.

## O Desafio do Treinamento: **Softmax** e Suas Otimizações

A etapa mais custosa no treinamento ingênuo de **CBoW** e **SkipGram** é a camada de saída Softmax:

$$P(w_O \vert w_I) = \frac{\exp(\vec{v}'_{w_O} \cdot \vec{v}_{w_I})}{\sum_{w \in V} \exp(\vec{v}'_{w} \cdot \vec{v}_{w_I})}$$

O cálculo do denominador $\sum_{w \in V} \exp(\vec{v}'_{w} \cdot \vec{v}_{w_I})$ requer a soma sobre **todo** o vocabulário $V$, tornando cada atualização $O(\vert V \vert)$. Para vocabulários realistas, centenas de milhares ou milhões de palavras, isso é impraticável.

Duas otimizações principais foram propostas para contornar esse problema:

### 1. Negative Sampling

* **Ideia**: Transforma o problema multiclasse (prever a palavra correta dentre $\vert V \vert$ opções) em múltiplos problemas de classificação binária.
  
* **Processo**: Para cada par positivo, palavra alvo, palavra de contexto real, selecionamos $k$ palavras negativas, que não são o contexto real, aleatoriamente do vocabulário (geralmente com uma distribuição enviesada pela frequência, $P_n(w) \propto f(w)^{0.75}$). O modelo será treinado para:
  
  * Maximizar a probabilidade, via **sigmoid**, de que o par positivo seja real;
  * Minimizar a probabilidade, via **sigmoid**, de que os pares negativos sejam reais.

* **Função Objetivo (simplificada para um par positivo e um negativo $w_N$)**:
  
    $$ L = -\log \sigma(\vec{v}'_{w_O} \cdot \vec{v}_{w_I}) - \log \sigma(-\vec{v}'_{w_N} \cdot \vec{v}_{w_I}) $$
  
    O objetivo é levar $\sigma(\text{positivo})$ para 1 e $\sigma(\text{negativo})$ para 0.

* **Complexidade**: $O(k+1)$, onde $k$, tipicamente $5$-$20$, é o número de amostras negativas. Muito mais eficiente que $O(\vert V \vert)$.

* **Implementação**: a atualização de gradiente só precisa considerar os vetores da palavra de entrada, da palavra de saída positiva e das $k$ palavras de saída negativas.

### 2. Hierarchical Softmax

* **Ideia**: organiza o vocabulário em uma árvore binária, geralmente uma **árvore de Huffman**, na qual palavras frequentes ficam mais perto da raiz.

* **Processo**: para prever a probabilidade de uma palavra $w_O$, a rede aprende a navegar da raiz da árvore até a folha correspondente a $w_O$. Em cada nó interno da árvore, uma função **sigmoid** prediz a probabilidade de ir para o filho esquerdo ou direito, usando um vetor associado àquele vértice interno ($v'_n$) e o vetor da palavra de entrada ($v_{w_I}$).

    $$ P(\text{nó filho} \vert n, w_I) = \sigma(\llbracket \text{nó filho é filho esquerdo?} \rrbracket \cdot \vec{v}'_n \cdot \vec{v}_{w_I}) $$

    Neste caso, temos: $\llbracket \cdot \rrbracket$ é $1$ ou $-1$ dependendo da convenção.

* **Probabilidade Final**: A probabilidade $P(w_O \vert w_I)$ é o produto das probabilidades das decisões tomadas ao longo do caminho único da raiz até a folha $w_O$.
  
* **Complexidade**: $O(\log_2 \vert V \vert)$, pois a profundidade da **árvore de Huffman** é logarítmica no tamanho do vocabulário.

* **Implementação**: A atualização de gradiente só envolve os vetores da palavra de entrada e dos nós internos no caminho até a palavra de saída correta.

**Comparação as Técnicas de Otimização:**

| Técnica             | Complexidade | Vantagem Principal                     | Desvantagem Principal                      | Uso Comum       |
|----------------------|--------------|----------------------------------------|--------------------------------------------|-----------------|
| **Softmax** Completo    | $O(\vert V \vert)$    | Matematicamente exato                  | Muito lento para vocabulários grandes     | Raramente usado |
| Negative Sampling   | $O(k)$       | Rápido, bom desempenho geral           | Aproximação, requer ajuste de $k$        | Muito comum     |
| Hierarchical Softmax| $O(\log \vert V \vert)$| Rápido, bom para palavras raras      | Estrutura de árvore complexa             | Menos comum hoje|

Na prática, **Negative Sampling** se tornou a otimização mais popular devido à sua simplicidade e bom desempenho empírico.

## As Propriedades Mágicas dos Embeddings Word2Vec

O resultado do treinamento com Word2Vec (seja **CBoW** ou SkipGram, com NS ou HS) é uma matriz de **embeddings** $W_{\text{entrada} }$, onde cada linha $\vec{v}_w$ é um vetor denso que representa a palavra $w$. Esses vetores exibem propriedades notáveis:

1. **Similaridade Semântica**: Palavras com significados semelhantes tendem a ter vetores próximos no espaço vetorial (alta similaridade de cosseno).

    * `cos_sim(vec("gato"), vec("cachorro"))` > `cos_sim(vec("gato"), vec("carro"))`

2. **Relações Analógicas**: Relações semânticas e sintáticas podem ser capturadas por aritmética vetorial. O exemplo mais famoso é:

    $$ \vec{v}_{\text{rei} } - \vec{v}_{\text{homem} } + \vec{v}_{\text{mulher} } \approx \vec{v}_{\text{rainha} } $$

    Outros exemplos incluem relações de capital-país, tempos verbais, etc.

    $$ \vec{v}_{\text{Paris} } - \vec{v}_{\text{França} } + \vec{v}_{\text{Alemanha} } \approx \vec{v}_{\text{Berlim} } $$

    $$ \vec{v}_{\text{andando} } - \vec{v}_{\text{andar} } + \vec{v}_{\text{nadar} } \approx \vec{v}_{\text{nadando} } $$

![Visualização de embeddings](/assets/images/word-embedding-visualization-improved.webp)

 _Figura 2: Visualização 2D, hipotética, após redução de dimensionalidade, mostrando como palavras semanticamente relacionadas se agrupam no espaço vetorial aprendido pelo Word2Vec._

Essas propriedades tornam os **embeddings Word2Vec** extremamente úteis como representações de entrada para tarefas de Processamento de Linguagem Natural mais complexas.

## Limitações do Word2Vec: A Estaticidade e a Falta de Contexto

Apesar de seu impacto revolucionário, o Word2Vec possui uma limitação fundamental já mencionada em [transformers-seis](link_para_transformers_seis): ele gera embeddings **estáticos**.

* **Um Vetor por Palavra**: Cada palavra no vocabulário tem *apenas um* vetor de embedding associado a ela, independentemente do contexto.
* **Problema da Polissemia**: O modelo não consegue distinguir os diferentes significados de uma palavra polissêmica. A palavra "banco" terá o mesmo vetor em "sentei no banco" e "fui ao banco". O vetor resultante é, na melhor das hipóteses, uma média dos contextos em que a palavra aparece.
* **Sensibilidade ao Contexto Imediato Perdida**: Embora treinado no contexto, o *embedding final* não muda dinamicamente com base nas palavras vizinhas em uma nova frase.

Essa natureza estática limita a capacidade do Word2Vec em tarefas que exigem uma compreensão profunda do contexto específico em que uma palavra é usada.

## A Ponte para os Transformers: Do Estático ao Contextual

As limitações dos embeddings estáticos como os do Word2Vec motivaram o desenvolvimento de **embeddings contextuais**. A ideia é que a representação de uma palavra deve **depender da sentença inteira** em que ela aparece.

É aqui que entram os mecanismos de **atenção** e a arquitetura **Transformer**:

1.  **Embeddings Iniciais**: Modelos como Transformers ainda começam com embeddings estáticos (muitas vezes pré-treinados com algoritmos tipo Word2Vec ou GloVe) para cada palavra/token de entrada.
2.  **Codificação Posicional**: Como a atenção por si só não considera a ordem das palavras, informações posicionais são adicionadas aos embeddings iniciais.
3.  **Mecanismo de Auto-Atenção (Self-Attention)**: Este é o coração do Transformer. Cada palavra calcula scores de atenção com *todas* as outras palavras na sequência (incluindo ela mesma). Esses scores determinam o quanto cada palavra contribui para a representação final de uma palavra específica. Isso permite que o modelo pese dinamicamente a importância das outras palavras com base no contexto atual.
    $$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k} } \right) V $$
4.  **Redes Feed-Forward**: Após a atenção, redes feed-forward processam a representação contextualizada de cada palavra independentemente.
5.  **Múltiplas Camadas**: Empilhando várias camadas de auto-atenção e FFN, o modelo constrói representações cada vez mais ricas e profundamente contextuais.

A saída de um Transformer para uma palavra não é mais um vetor estático, mas um **embedding contextual** que reflete o significado específico daquela palavra naquela sentença particular. A palavra "banco" terá representações diferentes em "sentei no banco" e "fui ao banco" após passar pelas camadas do Transformer.

Portanto, o Word2Vec pode ser visto como um passo crucial que forneceu **representações semânticas iniciais** de alta qualidade, mas estáticas. Os Transformers, utilizando a atenção, constroem sobre esses embeddings iniciais para gerar **representações dinâmicas e contextuais**, superando a principal limitação do Word2Vec e permitindo avanços significativos em PLN.

## Conclusão

O Word2Vec (CBoW e SkipGram), juntamente com otimizações como Negative Sampling e Hierarchical Softmax, representou um salto qualitativo na forma como representamos palavras para máquinas. Ao aprender vetores densos baseados no contexto, ele capturou relações semânticas e analógicas de forma inédita. No entanto, sua natureza estática o impede de lidar com a complexidade e a ambiguidade inerentes à linguagem natural dependente de contexto.

Compreender o Word2Vec e suas limitações é essencial para apreciar a inovação e o poder dos mecanismos de atenção e da arquitetura Transformer, que geram representações contextuais dinâmicas. O Word2Vec foi a ponte necessária que nos levou das representações esparsas e limitadas do passado para a era dos modelos de linguagem contextuais que dominam o PLN hoje. No próximo artigo, mergulharemos mais fundo na arquitetura Transformer, explorando componentes como a Atenção Multi-Cabeça e a Codificação Posicional.

## Referências

*Incluir aqui as referências já citadas nos artigos anteriores (cinco e seis) e adicionar referências específicas se necessário.*