---
layout: post
title: Transformers - Embeddings Distribuídos
author: frank
categories: |-
    disciplina
    Matemática
    artigo
tags: |-
    Matemática
    inteligência artificial
    processamento de linguagem natural
    word embeddings
    **Word2Vec**
    CBoW
    Skipgram
    vetorização
    aprendizado de máquina
image: assets/images/word2vec1.webp
featured: false
rating: 5
description: Explorando os algoritmos **Word2Vec** (CBoW e Skipgram) como ponte entre representações estáticas e os modelos contextuais modernos.
date: 2025-02-10T22:55:34.524Z
preview: Este artigo apresenta os algoritmos **Word2Vec** (CBoW e Skipgram) como avanço fundamental que supera as limitações das representações estáticas como Bag-of-Words e TF-IDF, pavimentando o caminho para os modelos contextuais modernos como os Transformers.
keywords: |-
    transformers
    cadeias de Markov
    modelos de sequência
    processamento de linguagem natural
    matrizes de transição
    matemática
    aprendizado de máquina
    inteligência artificial
    atenção
    rnn
    lstm
toc: true
published: false
lastmod: 2025-04-19T23:25:15.128Z
---

## Superando Limitações: A Necessidade de Representações Distribuídas

Nos artigos anteriores, exploramos técnicas de vetorização como [Bag-of-Words (BoW) e TF-IDF](https://frankalcantara.com/transformers-dois/), bem como [modelos probabilísticos N-gram](https://frankalcantara.com/transformers-desvendando-modelagem-de-sequencias/). Apesar da utilidade dessas abordagens, a atenta leitora pode ver que elas apresentam limitações significativas:

1. **Alta dimensionalidade e esparsidade**: as representações baseadas em BoW e TF-IDF geram vetores extremamente esparsos em espaços de alta dimensão, cardinalidade do vocabulário, resultando em matrizes enormes majoritariamente preenchidas com zeros.

2. **Ausência de semântica**: não capturam relações de similaridade ou analogia entre palavras. Por exemplo, `rei` e `rainha` podem ser tão diferentes quanto `rei` e `maçã`.

3. **Sem generalização**: o modelo não consegue inferir nada sobre palavras que não aparecem no corpus de treinamento ou que aparecem raramente.

4. **Contexto limitado**: mesmo os modelos **N-gram** capturam apenas dependências locais em janelas pequenas, ignorando relações de longo alcance.

Para superar essas limitações, precisamos de representações mais densas e de menor dimensionalidade que capturem relações semânticas entre palavras, permitam generalização para palavras raras ou novas e que tenham capacidade de modelar informações contextuais. Sim. Eu quero a perfeição. Ou o mais perto possível.

O avanço fundamental nessa direção foi introduzido por [Tomas Mikolov](https://en.wikipedia.org/wiki/Tom%C3%A1%C5%A1_Mikolov) e seus colegas no Google em 2013 com o **Word2Vec**, que propôs duas arquiteturas inovadoras para gerar **embeddings distribuídos de palavras**: o **Continuous Bag-of-Words (CBoW)** e o **Skipgram**.

## Embeddings Distribuídos: Nova Perspectiva para Representação de Palavras

Antes que a afeita leitora mergulhe nos mares dos algoritmos específicos dos **embeddings distribuídos**, precisamos tentar entender o conceito fundamental da representações distribuídas.

### Da Representação **One-Hot**  para Representação Distribuída

Em abordagens tradicionais como **Bag-of-Words**, frequentemente utilizamos (implícita ou explicitamente) vetores **one-hot** para representar palavras. Nesta representação, cada palavra corresponde a um vetor de tamanho igual ao vocabulário ($\vert V \vert $), contendo $1$ na posição única associada àquela palavra e $0$ em todas as outras. Nós vimos estas representações inocentes em detalhes no [artigo](https://frankalcantara.com/transformers-dois/).

Considerando **one-hot** poderíamos definir um vocabulário hipotético de $5$ palavras $V = \{\text{gato}, \text{cachorro}, \text{pássaro}, \text{corre}, \text{dorme}\}$, a representação **One-Hot**  seria:

* `gato` = $[1, 0, 0, 0, 0]$
* `cachorro` = $[0, 1, 0, 0, 0]$
* `pássaro` = $[0, 0, 1, 0, 0]$
* `corre` = $[0, 0, 0, 1, 0]$
* `dorme` = $[0, 0, 0, 0, 1]$

Como a atenta leitora deve lembrar, essa representação, apesar de simples, apresenta alguns problemas importantes:

* **Ortogonalidade e Ausência de Semântica**: o produto escalar entre quaisquer dois vetores **One-Hot**  distintos é zero ($v_{palavra1} \cdot v_{palavra2} = 0$). Isso implica que todas as palavras são igualmente diferentes umas das outras em um determinado espaço vetorial. Não há noção de similaridade; `gato` é tão distante de `cachorro` quanto de `corre`.
* **Alta Dimensionalidade e Esparsidade**: a dimensão do vetor cresce linearmente com o tamanho do vocabulário, que, em casos reais, pode facilmente chegar a milhões. Isso resulta em vetores extremamente longos e esparsos, quase inteiramente preenchidos por zeros.
* **Ineficiência Computacional e de Memória**: armazenar e processar esses vetores gigantes e esparsos é computacionalmente caro e ineficiente.

Para superar essas limitações, buscamos representações mais ricas e eficientes. Entram em cena as **representações distribuídas**, também conhecidas como **word embeddings**.

A ideia central por trás dessas representações está alinhada com a **hipótese distribucional**: o significado de uma palavra pode ser inferido a partir dos contextos em que ela costuma aparecer ("*you shall know a word by the company it keeps*", J.R. Firth). Assim, em vez de um único indicador "ligado", usamos vetores densos e de dimensão muito menor (tipicamente 50 a 300 dimensões) para codificar as palavras.

Podemos fazer uma **analogia**: pense na representação **One-Hot**  como um painel com um interruptor dedicado para cada palavra – apenas um pode estar ligado por vez. Já a representação distribuída seria como um painel de mixagem de áudio com vários controles deslizantes ("dimmers"). A identidade de uma palavra é definida pela combinação única das posições de *todos* esses controles.

Chama-se 'distribuída' porque o significado ou as características de uma palavra não estão localizados em uma única dimensão, mas sim **distribuídos** através de *todas* as dimensões do vetor. Cada elemento do vetor contribui um pouco para a representação geral, capturando nuances semânticas e sintáticas.

Voltando ao nosso vocabulário, uma representação distribuída *hipotética* poderia ser:

* `gato` = $[0.2, -0.4, 0.7, -0.2, 0.1, \cdots]$
* `cachorro` = $[0.1, -0.3, 0.8, -0.1, 0.2, \cdots]$
* `pássaro` = $[-0.1, -0.5, 0.3, 0.1, 0.4, \cdots]$
* `corre` = $[0.0, 0.6, -0.1, 0.8, -0.3, \cdots]$
* `dorme` = $[-0.3, 0.2, -0.5, -0.5, 0.6, \cdots]$

Observe como `gato` e `cachorro`, semanticamente próximos, compartilham alguns padrões (valores relativamente altos na 3ª dimensão, baixos na 2ª). Já `corre` e `dorme` (verbos) possuem padrões distintos dos animais. Os valores exatos não importam tanto quanto as *relações* entre os vetores.

É **fundamental** entender que essas dimensões (os elementos do vetor) não recebem rótulos pré-definidos como "animalidade" ou "ação". Pelo contrário, o modelo de treinamento (como o Word2Vec que veremos a seguir) **aprende** essas representações ao analisar vastas quantidades de texto. As dimensões acabam capturando características latentes (ocultas) e relações complexas entre palavras, simplesmente tentando prever palavras em seus contextos. Nós podemos, *a posteriori*, tentar interpretar o que certas dimensões ou combinações delas podem significar, mas elas **emergem** naturalmente do processo de aprendizado.

Essa capacidade de aprender representações significativas a partir do contexto resulta em vetores com propriedades notáveis, ausentes na representação **One-Hot** :

1.  **Similaridade semântica**: Palavras semanticamente similares terão representações vetoriais próximas no espaço vetorial (usando métricas como similaridade de cosseno). Por exemplo, os vetores para `gato` e `cachorro` estarão mais próximos entre si do que do vetor para `telefone`.

2.  **Relações analógicas**: Capturam relações que podem ser expressas por operações vetoriais simples. O exemplo clássico é:
    $$
    \text{vec}(\text{"rei"}) - \text{vec}(\text{"homem"}) + \text{vec}(\text{"mulher"}) \approx \text{vec}(\text{"rainha"})
    $$
    Ou ainda, para capitais e países:
    $$
    \text{vec}(\text{"Paris"}) - \text{vec}(\text{"França"}) + \text{vec}(\text{"Itália"}) \approx \text{vec}(\text{"Roma"})
    $$

3.  **Generalização**: Palavras raras ou mesmo fora do vocabulário de treinamento podem ter embeddings estimados de qualidade razoável se ocorrerem em contextos similares a palavras mais comuns (técnicas como FastText lidam bem com isso).

4.  **Transferência de aprendizado**: Embeddings pré-treinados em grandes volumes de texto (como notícias ou a Wikipédia) podem ser carregados e reutilizados como ponto de partida para diversas tarefas de PLN, mesmo com conjuntos de dados menores para a tarefa específica.

Estas propriedades tornam as representações distribuídas uma ferramenta poderosa e fundamental no Processamento de Linguagem Natural moderno, servindo de base para modelos mais complexos como LSTMs e Transformers. A seguir, exploraremos como o **Word2Vec** consegue aprender esses vetores densos e significativos.

### Propriedades dos Embeddings Distribuídos

As representações distribuídas têm propriedades notáveis:

1. **Similaridade semântica**: Palavras semanticamente similares terão representações vetoriais próximas no espaço. Por exemplo, os vetores para "gato" e "cachorro" estarão mais próximos entre si do que de "telefone".

2. **Relações analógicas**: Capturam relações surpreendentemente complexas que podem ser expressas por operações vetoriais:

  $$\text{vec}(\text{"rei"}) - \text{vec}(\text{"homem"}) + \text{vec}(\text{"mulher"}) \approx \text{vec}(\text{"rainha"})$$
  
  Ou ainda:
  
  $$\text{vec}(\text{"Paris"}) - \text{vec}(\text{"França"}) + \text{vec}(\text{"Itália"}) \approx \text{vec}(\text{"Roma"})$$

3. **Generalização**: Palavras raras podem ter embeddings de qualidade razoável se ocorrerem em contextos similares a palavras mais comuns.

4. **Transferência de aprendizado**: Embeddings treinados em grandes corpus de texto podem ser reutilizados em várias tarefas diferentes.

Estas propriedades emergem naturalmente quando treinamos algoritmos como **Word2Vec**, que aprendem a prever palavras com base em seus contextos (CBoW) ou vice-versa (Skipgram).

## **Word2Vec**: Aprendendo Representações a partir do Contexto

**Word2Vec** não é um único algoritmo, mas uma família de modelos com duas variantes principais: **Continuous Bag-of-Words (CBoW)** e **Skipgram**. Ambos compartilham a mesma ideia fundamental: aprender representações vetoriais de palavras ao prever palavras dentro de uma janela de contexto.

### Princípios Gerais do **Word2Vec**

A ideia central do **Word2Vec** é aproveitar a **hipótese distribucional**: palavras que aparecem em contextos similares tendem a ter significados similares. Assim, o modelo aprende representações vetoriais treinando uma rede neural rasa para uma tarefa de predição.

O processo geral funciona assim:

1. Cada palavra no vocabulário recebe inicialmente um vetor aleatório.
2. Uma rede neural simples, com apenas uma camada oculta, é treinada para uma tarefa de predição.
3. Após o treinamento, os vetores de palavras na camada de entrada (ou, em algumas implementações, na camada oculta) são usados como os embeddings finais.

A genialidade está na simplicidade: não estamos realmente interessados na tarefa de predição em si, mas nos vetores que a rede aprende durante o processo.

### Janela de Contexto

Ambos CBoW e Skipgram operam em janelas de contexto - uma janela deslizante que passa pelo texto, definindo para cada posição uma palavra-alvo e seu contexto. Considere a frase:

> "O gato preto corre pelo jardim"

Com uma janela de tamanho 2, para a palavra-alvo "preto", o contexto seria ["gato", "corre"].

### Arquitetura da Rede Neural

Em sua forma mais básica, a rede neural do **Word2Vec** tem a seguinte estrutura:

- **Camada de entrada**: Representa a(s) palavra(s) de contexto ou a palavra-alvo
- **Camada oculta**: Uma camada linear (sem função de ativação)
- **Camada de saída**: Uma camada softmax que produz probabilidades sobre o vocabulário

As matrizes de pesos entre as camadas são o que realmente nos interessa:

- $W_{entrada}$: Matriz de dimensão $|V| \times d$, onde $|V|$ é o tamanho do vocabulário e $d$ é a dimensão dos embeddings
- $W_{saída}$: Matriz de dimensão $d \times |V|$

### O Problema da Suavização com Softmax

A função softmax na camada de saída é definida como:

$$P(w_O|w_I) = \frac{\exp(v'_{w_O}v_{w_I})}{\sum_{w \in V} \exp(v'_{w}v_{w_I})}$$

Onde:
- $w_O$ é a palavra de saída (a prever)
- $w_I$ é a palavra de entrada
- $v_{w}$ é o vetor de entrada para a palavra $w$
- $v'_{w}$ é o vetor de saída para a palavra $w$

O problema computacional é evidente: o denominador requer uma soma sobre todo o vocabulário (possivelmente milhões de palavras), tornando-se proibitivamente caro para cada atualização.

## Continuous Bag-of-Words (CBoW)

Agora vamos explorar o primeiro modelo do **Word2Vec**: o **Continuous Bag-of-Words (CBoW)**.

### Conceito e Arquitetura do CBoW

No CBoW, o modelo usa o contexto (palavras circundantes) para prever a palavra-alvo central. "Continuous" se refere à natureza densa e contínua dos vetores de embeddings, diferentemente do Bag-of-Words tradicional que usa vetores esparsos.

Dada uma sequência de palavras de treinamento $w_1, w_2, ..., w_T$, o objetivo é maximizar a log-probabilidade:

$$\frac{1}{T} \sum_{t=1}^{T} \log p(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$$

Onde $c$ é o tamanho da janela de contexto.

A arquitetura específica do CBoW é:

1. **Entradas**: Vetores **One-Hot** para cada palavra de contexto
2. **Projeção**: Os vetores das palavras de contexto são projetados para a camada oculta e **somados** (ou tirada a média)
3. **Saída**: Predição da palavra central via softmax

Em termos matemáticos, dado um contexto $Context(w_t)$ para uma palavra-alvo $w_t$, queremos maximizar:

$$p(w_t|Context(w_t)) = \frac{\exp(v'_{w_t}h)}{\sum_{w \in V} \exp(v'_{w}h)}$$

Onde $h = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} v_{w_{t+j}}$ é a representação do contexto (média dos vetores de contexto).

### Exemplo de Treinamento com CBoW

Vamos ilustrar o processo com um exemplo simples. Considere a frase:

> "O gato preto corre pelo jardim"

Com uma janela de tamanho 1, obteríamos os seguintes pares contexto-alvo:

1. ["O"] → "gato"
2. ["O", "preto"] → "gato"
3. ["gato", "corre"] → "preto"
4. ["preto", "pelo"] → "corre"
5. ["corre", "jardim"] → "pelo"
6. ["pelo"] → "jardim"

Para o exemplo 3, o processo seria:

1. Converter "gato" e "corre" em vetores **One-Hot** 
2. Projetá-los para obter seus vetores de embedding $v_{gato}$ e $v_{corre}$
3. Calcular a média: $h = (v_{gato} + v_{corre}) / 2$
4. Multiplicar $h$ pela matriz de saída para obter scores para cada palavra no vocabulário
5. Aplicar softmax para obter probabilidades
6. Calcular a perda (erro) baseada na diferença entre a previsão e a palavra-alvo real "preto"
7. Propagar o erro de volta para atualizar os embeddings

### Vantagens e Desvantagens do CBoW

**Vantagens**:

- É eficiente para palavras frequentes, pois suaviza o ruído tirando a média de vários contextos
- Geralmente mais rápido de treinar que o Skipgram
- Bom para corpus menores

**Desvantagens**:

- A média dos vetores de contexto pode diluir informações específicas
- Não trata tão bem palavras raras quanto o Skipgram

## Skipgram

O modelo Skipgram inverte a tarefa do CBoW: em vez de usar o contexto para prever a palavra-alvo, usa a palavra-alvo para prever cada palavra do contexto.

### Conceito e Arquitetura do Skipgram

No Skipgram, para cada palavra-alvo, tentamos prever cada uma das palavras de contexto separadamente. Dada uma sequência de palavras de treinamento $w_1, w_2, ..., w_T$, o objetivo é maximizar a log-probabilidade:

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

Onde:
- $c$ é o tamanho da janela de contexto
- $p(w_{t+j} | w_t)$ é calculado usando softmax:

$$p(w_O | w_I) = \frac{\exp(v'_{w_O}v_{w_I})}{\sum_{w \in V} \exp(v'_{w}v_{w_I})}$$

Com $w_I$ sendo a palavra de entrada (alvo) e $w_O$ a palavra de saída (contexto).

A arquitetura do Skipgram é:

1. **Entrada**: Vetor **One-Hot** para a palavra-alvo
2. **Projeção**: O vetor é projetado para a camada oculta (embedding)
3. **Saída**: Múltiplos softmax, um para cada posição de contexto, prevendo a palavra naquela posição

Na prática, em vez de múltiplos softmax, tratamos cada par palavra-alvo/palavra-contexto como um exemplo de treinamento individual.

### Exemplo de Treinamento com Skipgram

Voltando à nossa frase de exemplo:

> "O gato preto corre pelo jardim"

Com uma janela de tamanho 1, o Skipgram geraria os seguintes pares alvo-contexto:

1. "O" → ["gato"]
2. "gato" → ["O", "preto"]
3. "preto" → ["gato", "corre"]
4. "corre" → ["preto", "pelo"]
5. "pelo" → ["corre", "jardim"]
6. "jardim" → ["pelo"]

Note que o número de exemplos de treinamento é o mesmo, mas agora "gato" aparece em 3 pares diferentes. Para o exemplo 3, o processo seria:

1. Converter "preto" em vetor **One-Hot** 
2. Projetá-lo para obter seu vetor de embedding $v_{preto}$
3. Para cada palavra de contexto ("gato" e "corre"), calcular a probabilidade dela ocorrer dado $v_{preto}$
4. Calcular a perda combinada
5. Propagar o erro de volta para atualizar os embeddings

### Vantagens e Desvantagens do Skipgram

**Vantagens**:

- Melhor para palavras raras, pois cada ocorrência gera múltiplos exemplos de treinamento
- Captura melhor múltiplos sentidos de palavras
- Funciona bem mesmo com corpus menores

**Desvantagens**:

- Computacionalmente mais intensivo, especialmente para palavras frequentes
- Gera muitos mais exemplos de treinamento, podendo tornar o treinamento mais lento

## Otimizações Críticas para **Word2Vec**

Na prática, treinar os modelos CBoW e Skipgram como descritos acima seria computacionalmente inviável para corpus grandes. Mikolov et al. introduziram duas otimizações críticas que tornaram esses algoritmos práticos.

### Negative Sampling

O problema principal do softmax é a necessidade de calcular um denominador que soma sobre todo o vocabulário. O **Negative Sampling** aproxima esta tarefa transformando-a em múltiplos problemas de classificação binária.

Para cada exemplo positivo (palavra-alvo e palavra de contexto real), amostramos $k$ exemplos negativos (palavras aleatórias que não são o contexto real). O objetivo se torna:

$$\log \sigma(v'_{w_O}v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v'_{w_i}v_{w_I})]$$

Onde:
- $\sigma$ é a função sigmoide $\sigma(x) = \frac{1}{1+e^{-x}}$
- $P_n(w)$ é a distribuição de amostragem (tipicamente proporcional a $f(w)^{0.75}$, onde $f(w)$ é a frequência da palavra)

Este método reduz dramaticamente a quantidade de computação, de $O(|V|)$ para $O(k)$ onde $k$ é tipicamente entre 5-20.

### Hierarchical Softmax

Outra alternativa ao softmax completo é o **Hierarchical Softmax**, que usa uma árvore binária de Huffman para representar o vocabulário. Cada palavra é uma folha, e o caminho da raiz até a folha representa uma sequência de decisões binárias.

A probabilidade de uma palavra é o produto das probabilidades de cada decisão ao longo do caminho:

$$p(w|w_I) = \prod_{j=1}^{L(w)-1} \sigma([[n(w,j+1) = ch(n(w,j))]] \cdot v'_{n(w,j)}v_{w_I})$$

Onde:
- $L(w)$ é o comprimento do caminho até $w$
- $n(w,j)$ é o $j$-ésimo nó no caminho
- $ch(n)$ é o filho esquerdo de $n$
- $[[x]]$ é 1 se $x$ é verdadeiro, -1 caso contrário

A complexidade computacional é reduzida para $O(\log |V|)$, tornando-se muito mais eficiente.

## Implementação em C++

Vamos agora ver uma implementação simplificada do algoritmo Skipgram com Negative Sampling em C++20. Por questões de espaço, focaremos nos componentes principais.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>

// Estrutura para representar a palavra e seu embedding
struct Word {
   std::string word;
   std::vector<float> vector;
   std::vector<float> gradient;
   long long count;
};

class **Word2Vec** {
private:
   // Parâmetros do modelo
   int vector_size;
   int window_size;
   int negative_samples;
   float learning_rate;
   
   // Vocabulário e contagens
   std::unordered_map<std::string, int> word_to_index;
   std::vector<Word> vocabulary;
   long long total_words;
   
   // Gerador de números aleatórios
   std::mt19937 rng;
   std::uniform_int_distribution<int> uniform_window;
   std::discrete_distribution<int> negative_sampler;
   
   // Inicializa embeddings com valores aleatórios
   void initializeEmbeddings() {
       std::uniform_real_distribution<float> dist(-0.5f / vector_size, 0.5f / vector_size);
       
       for (auto& word : vocabulary) {
           word.vector.resize(vector_size);
           word.gradient.resize(vector_size, 0.0f);
           
           for (int i = 0; i < vector_size; i++) {
               word.vector[i] = dist(rng);
           }
       }
   }
   
   // Treina um par de palavras (target e context) com negative sampling
   void trainPair(int target_idx, int context_idx) {
       // Lógica de negative sampling
       // 1. Atualizar para o par positivo (target, context)
       float dot_product = 0.0f;
       for (int i = 0; i < vector_size; i++) {
           dot_product += vocabulary[target_idx].vector[i] * vocabulary[context_idx].vector[i];
       }
       
       float probability = sigmoid(dot_product);
       float gradient = learning_rate * (1.0f - probability);
       
       // Atualizar vetores
       for (int i = 0; i < vector_size; i++) {
           vocabulary[context_idx].gradient[i] += gradient * vocabulary[target_idx].vector[i];
           vocabulary[target_idx].gradient[i] += gradient * vocabulary[context_idx].vector[i];
       }
       
       // 2. Atualizar para pares negativos (target, sampled)
       for (int n = 0; n < negative_samples; n++) {
           int negative_idx = sampleNegative(target_idx);
           
           dot_product = 0.0f;
           for (int i = 0; i < vector_size; i++) {
               dot_product += vocabulary[target_idx].vector[i] * vocabulary[negative_idx].vector[i];
           }
           
           probability = sigmoid(dot_product);
           gradient = learning_rate * (0.0f - probability);
           
           // Atualizar vetores
           for (int i = 0; i < vector_size; i++) {
               vocabulary[negative_idx].gradient[i] += gradient * vocabulary[target_idx].vector[i];
               vocabulary[target_idx].gradient[i] += gradient * vocabulary[negative_idx].vector[i];
           }
       }
       
       // Aplicar gradientes
       for (int i = 0; i < vector_size; i++) {
           vocabulary[target_idx].vector[i] += vocabulary[target_idx].gradient[i];
           vocabulary[context_idx].vector[i] += vocabulary[context_idx].gradient[i];
           
           // Resetar gradientes
           vocabulary[target_idx].gradient[i] = 0.0f;
           vocabulary[context_idx].gradient[i] = 0.0f;
       }
   }
   
   // Função sigmoide
   float sigmoid(float x) {
       return 1.0f / (1.0f + std::exp(-x));
   }
   
   // Amostra uma palavra negativa (diferente da target)
   int sampleNegative(int target_idx) {
       int negative_idx;
       do {
           negative_idx = negative_sampler(rng);
       } while (negative_idx == target_idx);
       
       return negative_idx;
   }

public:
   **Word2Vec**(int vector_size = 100, int window_size = 5, 
            int negative_samples = 5, float learning_rate = 0.025)
       : vector_size(vector_size), window_size(window_size),
         negative_samples(negative_samples), learning_rate(learning_rate),
         total_words(0), rng(std::random_device{}()) {
       
       uniform_window = std::uniform_int_distribution<int>(1, window_size);
   }
   
   // Constrói vocabulário a partir do corpus
   void buildVocabulary(const std::vector<std::vector<std::string>>& corpus) {
       std::unordered_map<std::string, long long> word_counts;
       
       // Contar ocorrências de cada palavra
       for (const auto& sentence : corpus) {
           for (const auto& word : sentence) {
               word_counts[word]++;
               total_words++;
           }
       }
       
       // Construir vocabulário com palavras que atingem o limiar mínimo
       int index = 0;
       for (const auto& [word, count] : word_counts) {
           Word w;
           w.word = word;
           w.count = count;
           vocabulary.push_back(w);
           word_to_index[word] = index++;
       }
       
       // Preparar distribuição para negative sampling (f(w)^0.75)
       std::vector<double> sampling_weights;
       for (const auto& word : vocabulary) {
           sampling_weights.push_back(std::pow(word.count / (double)total_words, 0.75));
       }
       
       negative_sampler = std::discrete_distribution<int>(
           sampling_weights.begin(), sampling_weights.end());
       
       // Inicializar embeddings
       initializeEmbeddings();
   }
   
   // Treina o modelo com Skipgram e Negative Sampling
   void trainSkipgram(const std::vector<std::vector<std::string>>& corpus, int epochs = 5) {
       for (int epoch = 0; epoch < epochs; epoch++) {
           // Para cada sentença no corpus
           for (const auto& sentence : corpus) {
               // Para cada palavra na sentença
               for (size_t i = 0; i < sentence.size(); i++) {
                   // Obter índice da palavra alvo
                   int target_idx = word_to_index[sentence[i]];
                   
                   // Determinar tamanho da janela para esta palavra
                   int current_window = uniform_window(rng);
                   
                   // Treinar com palavras de contexto
                   for (int j = -current_window; j <= current_window; j++) {
                       if (j == 0) continue; // Pular a própria palavra-alvo
                       
                       size_t context_pos = i + j;
                       if (context_pos >= sentence.size()) continue;
                       
                       // Obter índice da palavra de contexto
                       int context_idx = word_to_index[sentence[context_pos]];
                       
                       // Treinar o par (target, context)
                       trainPair(target_idx, context_idx);
                   }
               }
           }
           
           // Ajustar taxa de aprendizado para próxima época
           learning_rate *= 0.9f;
           
           std::cout << "Época " << epoch + 1 << "/" << epochs << " completa" << std::endl;
       }
   }
   
   // Salva os embeddings em um arquivo
   void saveEmbeddings(const std::string& filename) {
       std::ofstream file(filename);
       if (!file.is_open()) {
           std::cerr << "Erro ao abrir arquivo para escrita" << std::endl;
           return;
       }
       
       // Escrever cabeçalho: número de palavras e dimensão dos vetores
       file << vocabulary.size() << " " << vector_size << std::endl;
       
       // Escrever cada palavra e seu vetor
       for (const auto& word : vocabulary) {
           file << word.word;
           for (float val : word.vector) {
               file << " " << val;
           }
           file << std::endl;
       }
       
       file.close();
   }
   
   // Encontra palavras mais similares a uma palavra dada
   std::vector<std::pair<std::string, float>> findSimilar(const std::string& word, int top_n = 10) {
       if (word_to_index.find(word) == word_to_index.end()) {
           std::cerr << "Palavra não encontrada no vocabulário" << std::endl;
           return {};
       }
       
       int word_idx = word_to_index[word];
       const auto& word_vector = vocabulary[word_idx].vector;
       
       std::vector<std::pair<std::string, float>> similarities;
       
       // Calcular similaridade com todas as palavras
       for (size_t i = 0; i < vocabulary.size(); i++) {
           if (i == word_idx) continue; // Pular a própria palavra
           
           float similarity = cosineSimilarity(word_vector, vocabulary[i].vector);
           similarities.push_back({vocabulary[i].word, similarity});
       }
       
       // Ordenar por similaridade (decrescente)
       std::sort(similarities.begin(), similarities.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
       
       // Retornar top_n resultados
       if (similarities.size() > top_n) {
           similarities.resize(top_n);
       }
       
       return similarities;
   }
   
   // Calcula similaridade de cosseno entre dois vetores
   float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
       float dot_product = 0.0f;
       float norm_a = 0.0f;
       float norm_b = 0.0f;
       
       for (size_t i = 0; i < a.size(); i++) {
           dot_product += a[i] * b[i];
           norm_a += a[i] * a[i];
           norm_b += b[i] * b[i];
       }
       
       norm_a = std::sqrt(norm_a);
       norm_b = std::sqrt(norm_b);
       
       if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
       
       return dot_product / (norm_a * norm_b);
   }
};

// Função principal
int main() {
   // Exemplo de corpus simplificado
   std::vector<std::vector<std::string>> corpus = {
       {"o", "gato", "preto", "corre", "pelo", "jardim"},
       {"o", "cachorro", "late", "para", "o", "gato"},
       {"gatos", "e", "cachorros", "são", "anim