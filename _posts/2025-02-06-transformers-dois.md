---
layout: post
title: Transformers - A Vetorização Básica
author: frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - C++
    - Matemática
    - inteligência artificial
image: assets/images/transformers2.webp
featured: false
rating: 5
description: Técnicas de vetorização mais básicas, como a frequência de termos e o modelo Bag of Words e TF-IDF
date: 2025-02-09T22:55:34.524Z
preview: Vamos aprender as técnicas de vetorização mais básicas, como a frequência de termos e o modelo Bag of Words. Vamos discutir como essas técnicas são usadas para representar textos como vetores numéricos, permitindo que os computadores processem e analisem linguagem natural.
keywords: |-
    C++
    Matemática
    inteligência artificial
    vetorização
    frequência de termos
    bag of words
    NLP
    processamento de linguagem natural
    transformers
    embeddings
toc: true
published: true
lastmod: 2025-04-08T20:45:31.136Z
---

Neste artigo aprenderemos as técnicas de vetorização mais básicas, como a frequência de termos e o modelo *Bag of Words* (BoW). Vamos discutir como essas técnicas são usadas para representar textos como vetores numéricos, permitindo que os computadores processem e analisem linguagem natural.

## Técnicas Básicas de Vetorização

Existem diversas técnicas para transformar textos em números, nessa viagem, os primeiros portos que visitaremos estão fortemente construídos sobre as técnicas primitivas de representação matemática de elementos linguísticos. Eu vou fazer o máximo de esforço para seguir um fluxo crescente de dificuldade. A atenta leitora não deve se assustar se achar que eu perdi o rumo. Se eu parecer perdido é porque lembrei de algo que precisa ser visto antes que possamos deixar um porto em direção a outro. As passagens serão compradas sempre que eu achar que ficou claro.

### Frequência de Termos

Uma das formas mais básicas, quase intuitiva. de representar texto numericamente é através da frequência de termos. *A ideia é contar quantas vezes cada palavra aparece em um documento ou conjunto de documentos* e criar um vocabulário[^2].

[^2]: O termo "vocabulário" é frequentemente usado em linguística e processamento de linguagem natural para se referir ao conjunto de palavras ou termos que são relevantes para um determinado contexto ou tarefa. Em muitos casos, o vocabulário é construído a partir de um *corpus* de texto, onde cada palavra única é considerada um termo do vocabulário. O vocabulário pode variar em tamanho e complexidade, dependendo do domínio e da aplicação específica.

Antes de prosseguirmos com a vetorização de textos, precisamos formalizar o conceito de *corpus*. Em linguística computacional e processamento de linguagem natural, um *corpus* (plural *corpora*) é uma coleção estruturada de textos. Formalmente, definimos:

$$
\text{Corpus} = \{D_1, D_2, \ldots, D_N\}
$$

Nesta definição, cada $D_i$ é um documento individual e $N$ é o número total de documentos no *corpus*.

Cada documento $D_i$ é uma sequência ordenada de *tokens* ou palavras, que podem incluir palavras, números, sinais de pontuação e outros elementos significativos. Assim, podemos representar um documento $D_i$ como uma sequência de *tokens*:

Em processamento de linguagem natural, *um token é a menor unidade de análise em um texto*. Um token pode ser uma palavra, um número, um sinal de pontuação ou qualquer outra unidade significativa. Por exemplo, na frase "O gato preto.", temos 4 tokens: "O", "gato", "preto" e ".". O processo de dividir um texto em *tokens* é chamado de *tokenização*. A escolha de como definir e separar *tokens* é uma decisão importante que afeta todo o processamento subsequente. Considere, para ilustrar, a palavra composta "guarda-chuva" pode ser considerada como um único token ou como três tokens ("guarda", "-", "chuva"), dependendo das regras de tokenização adotadas. Em modelos modernos de processamento de linguagem natural, como o GPT e BERT, os tokens podem ser ainda mais granulares, incluindo partes de palavras, chamadas de subpalavras (*subwords*).

$$
D_i = (w_1, w_2, \ldots, w_{m_i})
$$

Neste caso, $w_j$ representa o $j$-ésimo token do documento e $m_i$ é o comprimento do documento $D_i$.

O vocabulário global $V$ do *corpus* será o conjunto de todos os tokens únicos que aparecem em qualquer documento. Dado por:

$$
V = \bigcup_{i=1}^N \{w : w \text{ ocorre em } D_i\}
$$

Em nossos exemplos de vetorização, trabalharemos tanto com documentos individuais quanto com o *corpus* completo, dependendo da técnica específica sendo aplicada. A cardinalidade do vocabulário $|V|$ determinará a dimensionalidade das nossas representações vetoriais.

Neste cenário vamos considerar que a palavra vocabulário se refere ao conjunto de todas as palavras únicas presentes em um *corpus* de texto, ou em um único documento, dependendo da técnica de representação que você está utilizando.

Pense no vocabulário como sendo uma lista organizada de todas as palavras distintas que o seu modelo de linguagem conhece, ou que são relevantes para a tarefa que está estudando. *Um vocabulário será sempre um conjunto finito de termos sem repetição*.

Podemos definir a técnica de frequência de termos como: seja $D$ um documento, definimos o vocabulário $V$ de $D$ como o conjunto de todas as palavras, $w$, únicas em $D$. De tal forma que: para cada palavra $w \in V$, a frequência de termos $f(w, D)$ é o número de vezes que $w$ ocorre em $D$.

Para entender essa definição, imagine que temos um texto curto $D_1$: "O gato preto subiu no telhado. O gato dorme no telhado".

O vocabulário $V_{D_1}$ de $D_1$ será dado por:

$$V_{D_1} = \{ \text{"o"}, \text{"gato"}, \text{"preto"}, \text{"subiu"}, \text{"no"}, \text{"telhado"}, \text{"dorme"} \}$$

Já que temos:

* o documento: "O gato preto subiu no telhado. O gato dorme no telhado".
* o vocabulário: $V_{D_1} = \{ \text{"o"}, \text{"gato"}, \text{"preto"}, \text{"subiu"}, \text{"no"}, \text{"telhado"}, \text{"dorme"} \}$.
* Calculamos a frequência, $f$, de cada termo em $V_{D_1}$ e $D_1$ apenas contando quantas vezes os termos do vocabulário aparecem no texto. Sendo assim, temos:

$$
\begin{align*}
& f(\text{"o"}, D_1) = 2\\
& f(\text{"gato"}, D_1) = 2\\
& f(\text{"preto"}, D_1) = 1\\
& f(\text{"subiu"}, D_1) = 1\\
& f(\text{"no"}, D_1) = 2\\
& f(\text{"telhado"}, D_1) = 2\\
& f(\text{"dorme"}, D_1) = 1
\end{align*}
$$

Com essa contagem, podemos representar o documento $D_1$ como um vetor de frequências $\vec{v}_{D_1}$. Se ordenarmos as palavras de $V_{D_1}$ alfabeticamente, por exemplo., $V'_{D_1} = \{ \text{"dorme"}, \text{"gato"}, \text{"no"}, \text{"o"}, \text{"preto"}, \text{"subiu"}, \text{"telhado"} \}$, então o vetor de frequências será dado por:

$$
\vec{v}_{D_1} = \begin{bmatrix} 1 \\ 2 \\ 2 \\ 2 \\ 1 \\ 1 \\ 2 \end{bmatrix}
$$

A sagaz leitora deve perceber que o vetor de frequência $\vec{v}_{D_1}$ reside no espaço vetorial inteiro $\mathbb{Z}^{\vert V \vert}$, no qual, temos:

* $\vec{v}_{D_1}$ denota o vetor de frequência do documento $D_1$.
* $\mathbb{Z}$ representa o conjunto dos números inteiros, indicando que cada componente do vetor $\vec{v}_{D_1}$ é um número inteiro (neste caso, uma contagem de frequência).
* $\vert V \vert$ representa a cardinalidade do vocabulário $V$, que é o número total de palavras únicas no vocabulário. Este valor $\vert V \vert$ define a dimensionalidade do espaço vetorial.

Em notação matemática, podemos expressar isso como:

$$\vec{v}_{D_1} \in \mathbb{Z}^{\vert V \vert}$$

Essa representação simples captura algumas informações sobre a importância relativa das palavras no texto. Palavras que aparecem com mais frequência podem ser consideradas mais relevantes para o conteúdo do documento. Isso pouco.

#### Limitações da Representação Vetorial por Frequência de Palavras

A representação de textos usando apenas vetores de frequência apresenta limitações que comprometem sua eficácia no processamento de linguagem natural. Para compreender melhor estas limitações, vamos tentar entender onde estão as falhas.

Considere um documento $D$ representado por um vetor de frequências $\vec{v}_D$, onde cada componente $v_i$ corresponde à frequência de uma palavra $w_i$ no vocabulário. Matematicamente, temos:

$$\vec{v}_D = [f(w_1,D), f(w_2,D), ..., f(w_n,D)]^T$$

onde $f(w_i,D)$ é a frequência da palavra $w_i$ no documento $D$.

O primeiro problema é a perda completa da estrutura sequencial da linguagem. Em português, por exemplo, a ordem das palavras é indispensável para o significado. Por exemplo, considere as frases:

"João ama Maria" e "Maria ama João"

Usando o algoritmo de frequência de termos, ambas as frases produziriam exatamente o mesmo vetor de frequência:

$$\vec{v} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$$

As componentes do vetor $\vec{v}$ correspondem às frequências de "João", "ama" e "Maria", respectivamente. Em um mundo perfeito, "João ama Maria" e "Maria ama João". Porém, além do mundo não ser perfeito, como diria [Drummond](https://www.letras.mus.br/carlos-drummond-de-andrade/460652/), o sentido de "João ama Maria" é muito diferente do sentido de "Maria ama João".

Um segundo problema é a ausência de relações semânticas entre palavras. Palavras com significados similares ou com significados relacionados são tratadas como completamente independentes. Por exemplo, nas frases:

"O carro é veloz" e "O automóvel é rápido"

Embora estas frases sejam semanticamente equivalentes, seus vetores de frequência seriam completamente diferentes e ortogonais no espaço vetorial, sem qualquer medida de similaridade entre elas. Como as palavras são tratadas como dimensões independentes, e não há sobreposição de termos entre "carro"/"automóvel" e "veloz"/"rápido", é possível que os vetores sejam ortogonais. A atenta leitora pode verificar este caso acompanhando o exemplo a seguir:

**Exemplo 1:** analisando os documentos compostos por "O carro é veloz" e "O automóvel é rápido":

1. $D_1$: "O carro é veloz"
2. $D_2$: "O automóvel é rápido"

Os documentos $D_1$ e $D_2$ forma o *corpus* da nossa análise. Primeiro, construímos o vocabulário global $V_{global}$ unindo todas as palavras únicas em $D_1$ e $D_2$:

$V_{global} = \{ \text{"o"}, \text{"carro"}, \text{"é"}, \text{"veloz"}, \text{"automóvel"}, \text{"rápido"} \}$

Ordenando alfabeticamente o vocabulário para definir a ordem das dimensões dos vetores:

$V'_{global} = \{ \text{"automóvel"}, \text{"carro"}, \text{"é"}, \text{"o"}, \text{"rápido"}, \text{"veloz"} \}$

Agora, criamos os vetores de frequência $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$ para cada documento, usando a ordem de $V'_{global}$. Lembre-se que cada posição do vetor corresponde a uma palavra em $V'_{global}$, e o valor é a frequência daquela palavra no documento.

Para $D_1$ = "O carro é veloz", teremos:

$$
\begin{align*}
& \text{automóvel}: 0\\
& \text{carro}: 1\\
& \text{é}: 1\\
& \text{o}: 1\\
& \text{rápido}: 0\\
& \text{veloz}: 1
\end{align*}
$$

Portanto, o vetor de frequência para $D_1$ será dado por:

$$
\vec{v}_{D_1} = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 1 \\ 0 \\ 1 \end{bmatrix}
$$

Para $D_2$ = "O automóvel é rápido", teremos:

$$\begin{align*}
& \text{automóvel}: 1 \\
& \text{carro}: 0 \\
& \text{é}: 1 \\
& \text{o}: 1 \\
& \text{rápido}: 1 \\
& \text{veloz}: 0 \\
\end{align*}$$

Portanto, o vetor de frequência para $D_2$ será:

$$
\vec{v}_{D_2} = \begin{bmatrix} 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0 \end{bmatrix}
$$

Podemos usar estes vetores para verificar a similaridade entre eles usando o produto escalar. Começamos pela ortogonalidade.

>Em álgebra linear, dois vetores $\vec{u}$ e $\vec{v}$ são ditos **ortogonais** se o seu produto escalar for igual a zero. Matematicamente, a ortogonalidade é definida como:
>
>$$
\vec{u} \cdot \vec{v} = \vec{u}^T \vec{v} = \sum_{i=1}^{n} u_i v_i = 0
$$
>
>na qual $n$ é a dimensão dos vetores.

Vamos calcular o produto escalar dos vetores de frequência $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$:

$$
\vec{v}_{D_1} \cdot \vec{v}_{D_2} = (0 \times 1) + (1 \times 0) + (1 \times 1) + (1 \times 1) + (0 \times 1) + (1 \times 0) = 0 + 0 + 1 + 1 + 0 + 0 = 2
$$

Neste caso, o produto escalar de $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$ é 2, que é **diferente de zero**. Portanto, os vetores $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$ **não são ortogonais**. Ou seja existe alguma similaridade direcional entre eles. Estes vetores não são ortogonais porque compartilham os termos "o" e "é".

O produto escalar sendo positivo indica que os vetores tendem a apontar na mesma direção. Quanto maior o valor positivo, maior a similaridade em termos de direção.

Vetores ortogonais não possuem qualquer similaridade direcional. Porém, mesmo que os vetores não sejam ortogonais, a similaridade medida apenas pelo produto escalar, ou outras medidas baseadas em frequência, ainda seria limitada.

Vetores de frequência capturam sobreposições de palavras literais, mas não a **relação semântica** subjacente. "Carro" e "automóvel", "veloz" e "rápido" são sinônimos, mas em representações de frequência, eles são tratados como palavras completamente distintas. *A semântica, sentido da palavra no documento, se perde*.

A questão da polissemia[^4] também é completamente ignorada na representação vetorial por frequência. Considere a palavra "banco" nas frases:

"Sentei no banco da praça" e "Fui ao banco sacar dinheiro"

O vetor de frequências trataria estas ocorrências do termo *banco* como idênticas, apesar de seus significados serem drasticamente diferentes. Matematicamente, isto significa que estamos mapeando diferentes significados para o mesmo componente do vetor, perdendo informação semântica.

[^4]: A palavra polissemia refere-se ao fenômeno linguístico em que uma única palavra possui múltiplos significados ou sentidos relacionados, mas distintos, dependendo do contexto em que é utilizada. Por exemplo, a palavra "banco" pode significar uma instituição financeira, um assento para sentar-se, ou um conjunto de dados, entre outros sentidos. Esta multiplicidade semântica constitui um desafio significativo para sistemas de processamento de linguagem natural, que precisam determinar qual significado específico está sendo empregado em cada ocorrência da palavra.

Outro problema sério é o tratamento de negações. As frases:

"O filme é bom" e "O filme não é bom"

Produziriam vetores muito similares, diferindo apenas pela presença do termo "não". A natureza oposta de seus significados seria praticamente indistinguível na representação vetorial. Os dois vetores terão, praticamente, a mesma direção e amplitude. Como a atenta leitora pode ver no exemplo a seguir:

**Exemplo 2**: vamos analisar os documentos "O filme é bom" e "O filme não é bom".

Começamos definindo os documentos:

1. $D_1$: "O filme é bom"
2. $D_2$: "O filme não é bom"

Construímos o vocabulário global $V_{global}$ a partir de todas as palavras únicas presentes em $D_1$ e $D_2$:

$V_{global} = \{ \text{"o"}, \text{"filme"}, \text{"é"}, \text{"bom"}, \text{"não"} \}$

Dado o vocabulário, podemos ordená-lo alfabeticamente para definir a ordem das dimensões dos vetores:

$V'_{global} = \{ \text{"bom"}, \text{"é"}, \text{"filme"}, \text{"não"}, \text{"o"} \}$

A cardinalidade do vocabulário é $\vert V'_{global}\vert  = 5$, o que significa que nossos vetores de frequência terão 5 dimensões.

Neste ponto, podemos criar os vetores de frequência $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$ para cada documento, seguindo a ordem das palavras em $V'_{global}$.

Para $D_1$ = "O filme é bom":

$$
\begin{align*}
& \text{bom}: 1 \\
& \text{é}: 1\\
& \text{filme}: 1\\
& \text{não}: 0\\
& \text{o}: 1
\end{align*}
$$

O vetor de frequência para $D_1$ é:

$$
\vec{v}_{D_1} = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 0 \\ 1 \end{bmatrix}
$$

Para $D_2$ = "O filme não é bom":

$$
\begin{align*}
& \text{bom}: 1 \\
& \text{é}: 1 \\
& \text{filme}: 1 \\
& \text{não}: 1 \\
& \text{o}: 1
\end{align*}
$$

O vetor de frequência para $D_2$ é:

$$
\vec{v}_{D_2} = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}
$$

Uma vez que os vetores estejam definidos, podemos analisar os vetores $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$ em termos de produto escalar e magnitude, e discutir a similaridade entre eles.

**a) Produto Escalar**: calculamos o produto escalar de $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$:

$$
\vec{v}_{D_1} \cdot \vec{v}_{D_2} = (1 \times 1) + (1 \times 1) + (1 \times 1) + (0 \times 1) + (1 \times 1) = 1 + 1 + 1 + 0 + 1 = 4
$$

Para serem ortogonais, o produto escalar deveria ser zero. Como $\vec{v}_{D_1} \cdot \vec{v}_{D_2} = 4 \neq 0$, os vetores **não são ortogonais**.

O produto escalar é $4$, um valor positivo e relativamente alto, considerando as magnitudes dos vetores. Isso **sugere alguma similaridade** entre os vetores no espaço vetorial. Eles tendem a mesma direção.

**b) Magnitudes**: calculamos as magnitudes de $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$:

$$
\vert \vec{v}_{D_1}\vert = \sqrt{1^2 + 1^2 + 1^2 + 0^2 + 1^2} = \sqrt{4} = 2
$$

$$
\vert \vec{v}_{D_2}\vert = \sqrt{1^2 + 1^2 + 1^2 + 1^2 + 1^2} = \sqrt{5} \approx 2.236
$$

As magnitudes são próximas, indicando que ambos os vetores têm comprimentos similares no espaço vetorial definido pelo vocabulário global.

Embora o produto escalar e as magnitudes sugiram alguma similaridade entre $\vec{v}_{D_1}$ e $\vec{v}_{D_2}$, a leitora pode notar a **discrepância em termos de significado semântico**.

- $D_1$: "O filme é bom" expressa uma **avaliação positiva** sobre o filme.
- $D_2$: "O filme não é bom" expressa uma **avaliação negativa**, ou no mínimo, não positiva, sobre o filme, essencialmente com o sentido **oposto** ao sentido de $D_1$.

**A representação vetorial de frequência, neste caso, falha em capturar essa oposição semântica.** Os vetores são considerados *similares* em termos de produto escalar porque eles compartilham muitas palavras em comum ("o", "filme", "é", "bom"). A presença da palavra "não" em $D_2$, que inverte completamente o sentido, tem um impacto quase insignificante na representação vetorial que usa a técnica de frequência.

Ainda há um aspecto particularmente problemático na representação vetorial por frequência: a questão da dimensionalidade.

Para um vocabulário de tamanho $\vert V \vert$, cada documento é representado em um espaço $\mathbb{R}^{\vert V \vert}$. Neste caso, $\mathbb{R}$ refere-se a um espaço vetorial de dimensão $\vert V \vert $, o tamanho do vocabulário. Isso significa que cada documento é representado como um vetor em um espaço de alta dimensão, no qual cada dimensão corresponde a uma palavra do vocabulário. Para um vocabulário de $10.000$ palavras, cada documento seria representado como um vetor em um espaço de $10.000$ dimensões.

Como usaremos apenas a frequência, teremos vetores extremamente esparsos, nos quais a maioria dos componentes será zero. Esta característica não só é computacionalmente ineficiente, mas também dificulta a identificação de similaridades entre documentos.

Um vetor esparso é um vetor onde a **maioria das suas componentes é zero**. Em coleções de documentos, como livros, artigos e processos, se usarmos os nossos vetores de frequência, a maioria das palavras do vocabulário não estará presente em um documento específico. Portanto, para um documento $D$, a maioria das entradas em $\vec{v}_D$ será zero.

Trabalhar com vetores em espaços de alta dimensão (milhares, milhões ou bilhões de dimensões) é **computacionalmente caro**. Armazenar e processar vetores esparsos com tantas dimensões exige recursos significativos de memória e tempo de processamento. Nestas condições, a noção de *distância* e *similaridade* torna-se menos intuitiva e, principalmente, menos discriminativa. Vetores esparsos em alta dimensão tendem a ser *distantes* uns dos outros, mesmo que semanticamente relacionados. Em resumo: *vetores esparsos dificultam o cálculo de medidas de similaridade robustas e significativas*.

#### Perda de Informação Contextual

Para verificar a qualidade da representação por frequência e quantificar a perda de informação contextual devido à esparsidade e à alta dimensionalidade, podemos expressar esta perda de informação por meio de uma **função de perda $L(\vec{v}_D)$**.Podemos definir a função de perda $L(\vec{v}_D)$ como:

$$L(\vec{v}_D) = H(D) - I(\vec{v}_D)$$

Nesta função, temos:

- **$L(\vec{v}_D)$**: representa a **perda de informação contextual** ao representar o documento $D$ pelo vetor de frequência $\vec{v}_D$. Idealmente, queremos que $L(\vec{v}_D)$ seja o menor possível, indicando que a representação vetorial preserva o máximo de informação relevante.

- **$H(D)$**: é a **entropia de informação do documento original $D$**. A entropia $H(D)$ é uma medida teórica da quantidade total de informação contida no documento $D$. Ela quantifica a incerteza ou aleatoriedade inerente ao documento. Em termos simples, documentos mais complexos e informativos tendem a ter entropia mais alta.

> A definição precisa de entropia em contextos textuais pode ser complexa e depender de como a informação é modelada, mas conceitualmente representa a "riqueza" informacional do texto original. A quantidade de termos diferentes que aparecem no documento, a frequência de cada termo e a distribuição de probabilidade dos termos são fatores que influenciam a entropia. Documentos com uma variedade maior de palavras e uma distribuição mais uniforme entre elas tendem a ter maior entropia.

- **$I(\vec{v}_D)$**: é a **informação preservada na representação vetorial $\vec{v}_D$**. $I(\vec{v}_D)$ mede quanta informação do documento original $D$ é efetivamente capturada e retida na representação vetorial $\vec{v}_D$. Representações mais eficazes devem maximizar $I(\vec{v}_D)$.

A função $L(\vec{v}_D) = H(D) - I(\vec{v}_D)$ expressa a perda de informação como a **diferença** entre a informação total original ($H(D)$) e a informação que conseguimos reter na representação vetorial ($\vec{v}_D$).

Para representações como vetores de frequência (e **BoW**, **TF-IDF** que veremos a seguir), a perda $L(\vec{v}_D)$ é **substancial**. Isso ocorre porque, como discutimos, essas representações descartam muita informação contextual importante (ordem das palavras, relações semânticas, nuances, etc.). A entropia original $H(D)$ é alta (o documento contém muita informação), mas a informação preservada $I(\vec{v}_D)$ é relativamente baixa, resultando em uma perda significativa.

A perda de informação $L(\vec{v}_D)$ tende a **aumentar com a complexidade semântica do texto**. Textos mais complexos, com nuances, ironia, sarcasmo, metáforas, etc., dependem ainda mais do contexto para a compreensão. Representações simples como vetores de frequência falham ainda mais em capturar a riqueza informacional desses textos, resultando em uma perda de informação ainda maior.

A consciência dessas limitações motivou o desenvolvimento de técnicas mais avançadas, como **word embeddings** (Word2Vec, GloVe, FastText) e **modelos de linguagem contextuais** (BERT, GPT, *Transformers*). Vamos chegar lá. Todavia, por enquanto temos que entender algumas outras formas de vetorização que servirão de contextualização e base para o entendimento dos modelos mais avançados.

### Bag of Words (BoW)

O modelo **Bag of Words (BoW)**, ou *saco de palavras*, é uma variação da representação por frequência. O **BoW** também se baseia na frequência de palavras, mas com uma abordagem que ignora a ordem e a estrutura gramatical das palavras no texto. Este algoritmo trata cada documento como um *saco* de palavras, onde apenas a presença e a frequência das palavras importam.

O algoritmo **Bag of Words (BoW)** surgiu da área de recuperação de informação e processamento de linguagem natural. Embora a ideia de representar textos como "sacos de palavras" tenha evoluído ao longo do tempo, um dos trabalhos seminais que formalizou e popularizou essa abordagem é frequentemente creditado a [Gerard Salton](https://en.wikipedia.org/wiki/Gerard_Salton) e seus colaboradores no contexto do sistema SMART (*System for the Mechanical Analysis and Retrieval of Text*) [^5], [^6].

[^5]: SALTON, G.; LESK, M. E. **The SMART Automatic Document Retrieval System—An Illustration**. Communications of the ACM, v. 8, n. 9, p. 391-398, 1965. Disponível em: https://apastyle.apa.org/blog/citing-online-works. Acesso em: 27 de fevereiro de 2025.

[^6]:SALTON, G.; WONG, A.; YANG, C. S. **A vector space model for automatic indexing**. Communications of the ACM, v. 18, n. 11, p. 613-620, 1975. Disponível em: https://apastyle.apa.org/blog/citing-online-works. Acesso em: 10 de março de 2025.

Para entender o **BoW** matematicamente, vamos considerar um conjunto de documentos, *corpus*, $Docs = \{D_1, D_2, ..., D_N\}$. Para construir o **BoW**, nossa primeira ação será construir um vocabulário global $V_{global}$ que conterá todas as palavras únicas que existem em todos os documentos de $Docs$:

$$
V_{global} = \bigcup_{i=1}^N V_{D_i}
$$

onde $V_{D_i}$ é o vocabulário do documento $D_i$.

Para cada documento $D_i \in Docs$, a representação **BoW** é um vetor $\vec{bow}_{D_i}$ de tamanho $|V_{global}|$. Cada posição $j$ em $\vec{bow}_{D_i}$ corresponde à $j$-ésima palavra $w_j$ em $V_{global}$, e o valor nessa posição será a frequência $f(w_j, D_i)$ da palavra no documento:

$$
\vec{bow}_{D_i} = [f(w_1, D_i), f(w_2, D_i), ..., f(w_{|V_{global}|}, D_i)]^T
$$

Para ilustrar, considere um pequeno *corpus* com dois documentos:

$D_1$: "O gato preto subiu no telhado"
$D_2$: "O cachorro correu no gramado"

O vocabulário global $V_{global}$ será:

$$V_{global} = \{\text{"o"}, \text{"gato"}, \text{"preto"}, \text{"subiu"}, \text{"no"}, \text{"telhado"}, \text{"cachorro"}, \text{"correu"}, \text{"gramado"}\}$$

Ordenando alfabeticamente (uma prática recomendada para padronização):

$$V'_{global} = \{\text{"cachorro"}, \text{"correu"}, \text{"gato"}, \text{"gramado"}, \text{"no"}, \text{"o"}, \text{"preto"}, \text{"subiu"}, \text{"telhado"}\}$$

Os vetores **BoW** para cada documento serão:

$$
\vec{bow}_{D_1} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}
\quad
\vec{bow}_{D_2} = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

A atenta leitora deve observar que cada vetor $\vec{bow}_{D_i}$ reside no espaço vetorial inteiro $\mathbb{Z}^{|V_{global}|}$. Estes vetores tendem a ser esparsos, ou seja, a maioria de suas componentes é zero, especialmente quando o vocabulário global é grande.

O **BoW** mantém todas as limitações da representação por frequência de termos que discutimos anteriormente:

1. **Perda da ordem das palavras**: As frases "João ama Maria" e "Maria ama João" produzem exatamente o mesmo vetor **BoW**.

2. **Ausência de relações semânticas**: As frases "O carro é veloz" e "O automóvel é rápido", embora semanticamente equivalentes, produzem vetores completamente diferentes.

3. **Alta dimensionalidade**: Para um vocabulário de tamanho $|V_{global}|$, cada documento é representado em um espaço $\mathbb{R}^{|V_{global}|}$, levando a vetores extremamente esparsos.

A sagaz leitora pode pensar que essas limitações tornam o **BoW** inútil. Contudo, apesar dessas limitações, o **BoW** é surpreendentemente eficaz em muitas tarefas de processamento de linguagem natural, especialmente quando combinado com outras técnicas que veremos adiante, como o **TF-IDF**.

**Exemplo 1**: usando o C++

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include <iomanip>

// Classe para implementar o modelo Bag of Words (BoW)
class BagOfWords {
private:
    std::vector<std::string> vocabulary;              // Vocabulário global ordenado
    std::unordered_map<std::string, int> wordToIndex; // Mapeamento de palavras para índices
    std::vector<Eigen::VectorXi> documentVectors;     // Vetores BoW dos documentos

public:
    // Função para construir o vocabulário global a partir de documentos já tokenizados
    void buildVocabulary(const std::vector<std::vector<std::string>>& documents) {
        std::set<std::string> uniqueWords;
        
        // Adicionar todas as palavras ao conjunto de palavras únicas
        for (const auto& doc : documents) {
            for (const auto& word : doc) {
                uniqueWords.insert(word);
            }
        }
        
        // Converter o conjunto para um vetor ordenado
        vocabulary.assign(uniqueWords.begin(), uniqueWords.end());
        std::sort(vocabulary.begin(), vocabulary.end());
        
        // Criar mapeamento de palavras para índices
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            wordToIndex[vocabulary[i]] = i;
        }
    }
    
    // Função para criar os vetores BoW para cada documento
    void createBowVectors(const std::vector<std::vector<std::string>>& documents) {
        documentVectors.clear();
        
        for (const auto& doc : documents) {
            // Inicializar um vetor de zeros com o tamanho do vocabulário
            Eigen::VectorXi bowVector = Eigen::VectorXi::Zero(vocabulary.size());
            
            // Contar a frequência de cada palavra no documento
            for (const auto& word : doc) {
                if (wordToIndex.find(word) != wordToIndex.end()) {
                    int index = wordToIndex[word];
                    bowVector(index)++;
                }
            }
            
            documentVectors.push_back(bowVector);
        }
    }
    
    // Função para inicializar o modelo com um conjunto de documentos tokenizados
    void fit(const std::vector<std::vector<std::string>>& documents) {
        buildVocabulary(documents);
        createBowVectors(documents);
    }
    
    // Função para obter o vetor BoW de um novo documento tokenizado
    Eigen::VectorXi transform(const std::vector<std::string>& document) {
        Eigen::VectorXi bowVector = Eigen::VectorXi::Zero(vocabulary.size());
        
        for (const auto& word : document) {
            if (wordToIndex.find(word) != wordToIndex.end()) {
                int index = wordToIndex[word];
                bowVector(index)++;
            }
        }
        
        return bowVector;
    }
    
    // Função para calcular a similaridade de cosseno entre dois vetores
    double cosineSimilarity(const Eigen::VectorXi& v1, const Eigen::VectorXi& v2) {
        // Converter para double para cálculos de ponto flutuante
        Eigen::VectorXd v1d = v1.cast<double>();
        Eigen::VectorXd v2d = v2.cast<double>();
        
        double dotProduct = v1d.dot(v2d);
        double norm1 = v1d.norm();
        double norm2 = v2d.norm();
        
        if (norm1 < 1e-10 || norm2 < 1e-10) return 0.0;
        return dotProduct / (norm1 * norm2);
    }
    
    // Função para imprimir a matriz de documentos x termos (DTM)
    void printDocumentTermMatrix() {
        std::cout << "Matriz de Documentos x Termos (Document-Term Matrix):\n\n";
        
        // Imprimir cabeçalho com os termos do vocabulário
        std::cout << std::setw(10) << "Documento";
        for (const auto& word : vocabulary) {
            std::cout << std::setw(10) << word;
        }
        std::cout << "\n";
        
        // Imprimir linha para cada documento
        for (size_t i = 0; i < documentVectors.size(); ++i) {
            std::cout << std::setw(10) << "Doc " + std::to_string(i+1);
            for (int j = 0; j < documentVectors[i].size(); ++j) {
                std::cout << std::setw(10) << documentVectors[i](j);
            }
            std::cout << "\n";
        }
    }
    
    // Função para imprimir informações sobre o modelo
    void printInfo() {
        std::cout << "Modelo Bag of Words (BoW)\n";
        std::cout << "-----------------------\n";
        std::cout << "Tamanho do vocabulário: " << vocabulary.size() << " palavras\n";
        std::cout << "Número de documentos: " << documentVectors.size() << "\n\n";
        
        std::cout << "Vocabulário global (ordenado alfabeticamente):\n";
        std::cout << "{ ";
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            std::cout << "\"" << vocabulary[i] << "\"";
            if (i < vocabulary.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " }\n\n";
    }
    
    // Função para comparar dois documentos
    void compareDocuments(int doc1Index, int doc2Index) {
        if (doc1Index >= documentVectors.size() || doc2Index >= documentVectors.size()) {
            std::cout << "Índice de documento inválido.\n";
            return;
        }
        
        double similarity = cosineSimilarity(documentVectors[doc1Index], 
                                             documentVectors[doc2Index]);
        
        std::cout << "Similaridade entre Doc " << (doc1Index+1) << " e Doc " 
                  << (doc2Index+1) << ": " << std::fixed << std::setprecision(4) 
                  << similarity << "\n";
    }
    
    // Getters para fins de análise
    const std::vector<std::string>& getVocabulary() const { return vocabulary; }
    const std::vector<Eigen::VectorXi>& getDocumentVectors() const { return documentVectors; }
};

int main() {
    // Documentos já tokenizados do exemplo no texto
    std::vector<std::vector<std::string>> documents = {
        {"o", "gato", "preto", "subiu", "no", "telhado"},
        {"o", "cachorro", "correu", "no", "gramado"}
    };
    
    // Criar e treinar o modelo BoW
    BagOfWords bow;
    bow.fit(documents);
    
    // Imprimir informações sobre o modelo
    bow.printInfo();
    
    // Imprimir a matriz de documentos x termos
    bow.printDocumentTermMatrix();
    
    // Calcular similaridade entre os documentos
    std::cout << "\nSimilaridade entre documentos:\n";
    bow.compareDocuments(0, 1);
    
    // Demonstrar o uso do modelo com um novo documento
    std::vector<std::string> newDoc = {"o", "gato", "preto", "desceu", "do", "telhado"};
    std::cout << "\nNovo documento: [\"o\", \"gato\", \"preto\", \"desceu\", \"do\", \"telhado\"]\n";
    
    Eigen::VectorXi newDocVector = bow.transform(newDoc);
    
    // Imprimir o vetor BoW do novo documento
    std::cout << "Vetor BoW do novo documento:\n[ ";
    for (int i = 0; i < newDocVector.size(); ++i) {
        std::cout << newDocVector(i);
        if (i < newDocVector.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]\n\n";
    
    // Comparar o novo documento com os existentes
    std::cout << "Similaridade com Doc 1: " << std::fixed << std::setprecision(4) 
              << bow.cosineSimilarity(newDocVector, bow.getDocumentVectors()[0]) << "\n";
    std::cout << "Similaridade com Doc 2: " << std::fixed << std::setprecision(4) 
              << bow.cosineSimilarity(newDocVector, bow.getDocumentVectors()[1]) << "\n";
    
    // Demonstrar como a ordem das palavras não afeta o vetor BoW
    std::vector<std::string> reorderedDoc = {"telhado", "o", "gato", "preto", "subiu", "no"};
    std::cout << "\nDocumento com palavras reordenadas:\n";
    std::cout << "[\"telhado\", \"o\", \"gato\", \"preto\", \"subiu\", \"no\"]\n";
    
    Eigen::VectorXi reorderedVector = bow.transform(reorderedDoc);
    
    std::cout << "Vetor BoW do documento reordenado:\n[ ";
    for (int i = 0; i < reorderedVector.size(); ++i) {
        std::cout << reorderedVector(i);
        if (i < reorderedVector.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " ]\n\n";
    
    // Verificar que os vetores BoW são idênticos, independente da ordem
    bool areEqual = (reorderedVector - bow.getDocumentVectors()[0]).norm() == 0;
    std::cout << "Os vetores BoW do Doc 1 e do documento reordenado são idênticos? "
              << (areEqual ? "Sim" : "Não") << "\n";
    
    return 0;
}
```

Este exemplo reimplementa o modelo **Bag of Words (BoW)** assumindo que os documentos são simples, como os que usamos nos exemplos teóricos. A implementação foca exclusivamente na representação matemática e nas funcionalidades essenciais do **BoW**.

1. **Representação matemática pura**:

   - A classe `BagOfWords` implementa diretamente a formulação matemática apresentada no texto:
   - Construção do vocabulário global: $V_{global} = \bigcup_{i=1}^N V_{D_i}$
   - Criação de vetores BoW: $\vec{bow}_{D_i} = [f(w_1, D_i), f(w_2, D_i), ..., f(w_{|V_{global}|}, D_i)]^T$

2. **Uso da biblioteca Eigen**:

   - Utilizamos `Eigen::VectorXi` para representar os vetores de frequência de termos
   - Aproveitamos operações eficientes como `.dot()` e `.norm()` para o cálculo de similaridade de cosseno
   - A conversão de tipos de `VectorXi` para `VectorXd` é feita através do método `.cast<double>()`

3. **Estruturas de dados eficientes**:

   - Usamos `std::set` para obter palavras únicas ao construir o vocabulário
   - Implementamos um mapeamento eficiente de palavras para índices com `std::unordered_map`
   - A representação vetorial permite operações matemáticas rápidas nos documentos

4. **Demonstração de limitações do BoW**:

   - Adicionamos um exemplo específico para mostrar como a ordem das palavras não afeta o vetor BoW
   - Verificamos matematicamente que documentos com palavras em ordens diferentes produzem vetores idênticos

No código do exemplo, implementamos as seguintes funções principais:

- **`buildVocabulary()`**: constrói o vocabulário global ordenado alfabeticamente
- **`createBowVectors()`**: cria os vetores de frequência de termos para cada documento
- **`transform()`**: converte um novo documento em um vetor BoW usando o vocabulário existente
- **`cosineSimilarity()`**: calcula a similaridade entre dois documentos, usando a fórmula:

$$
\text{sim}(\vec{a},\vec{b}) = \cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}\vert\vec{b}|} = \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2}\sqrt{\sum_{i=1}^n b_i^2}}
$$

- **`printDocumentTermMatrix()`**: Visualiza a matriz de documentos por termos

Por fim, na função `main()`:

1. usamos os documentos do exemplo no texto: "O gato preto subiu no telhado" e "O cachorro correu no gramado";
2. construímos o modelo BoW e visualizamos a matriz de documentos por termos;
3. calculamos a similaridade entre os documentos;
4. demonstramos a aplicação do modelo a um novo documento;
5. ilustramos de forma prática uma das limitações fundamentais do BoW: a perda da ordem das palavras.

Esta implementação é concisa e focada nos aspectos matemáticos do **BoW**. Com o único objetivo de permitir que a esforçada leitora possa entender como passar a matemática para código em C++.

Caberá a criativa leitora a criar sua própria interpretação.

### **TF-IDF** (Term Frequency-Inverse Document Frequency)

O **TF-IDF (Term Frequency-Inverse Document Frequency)** tem origem nos trabalhos de Spärck[^7] e Salton[^8] quase como uma evolução natural do modelo **BoW** com o objetivo de resolver um dos seus problemas fundamentais: a dominância de palavras muito frequentes. O **TF-IDF** teve impacto significativo na área de recuperação de informação por conseguir ponderar a importância das palavras levando em consideração, além da frequência em um determinado documento, a raridade no *corpus*.

[^7]: SPÄRCK JONES, K. S. A statistical interpretation of term specificity and its application in retrieval. Journal of Documentation, v. 28, n. 1, p. 11–21, 1972.

[^8]: SALTON, G.; BUCKLEY, C. Term-weighting approaches in automatic text retrieval. Information Processing & Management, v. 24, n. 5, p. 513–523, 1988.

Formalmente, seja $Docs = \{D_1, D_2, ..., D_N\}$ um *corpus* com $N$ documentos. O **TF-IDF** combinará duas métricas complementares para cada palavra $w$ em um documento $D_i$. Da seguinte forma:

1. **Term Frequency (TF)**: captura a importância local da palavra no documento;
2. **Inverse Document Frequency (IDF)**: mede a importância global da palavra no corpus.

Uma forma interessante de entender o **TF-IDF** é dividir o algoritmo em suas partes constituintes e entender cada parte em separado. Dividir para vencer!

#### Term Frequency (TF)

O cálculo da frequência de termos, ou **Term Frequency (TF)**, pode ser realizado de diferentes maneiras, cada uma com suas próprias características e aplicações.

A forma mais simples de calcular o **TF** é usar a frequência bruta, $raw$, também chamada de contagem absoluta:

$$
\text{TF}_{raw}(w, D_i) = f(w, D_i)
$$

Nesta definição, $f(w, D_i)$ representa o número de vezes que a palavra $w$ aparece no documento $D_i$.

**Exemplo 1**: considere os dois documentos $D_1$ e $D_2$, a seguir:

- $D_1$: "O pequeno gato preto viu outro gato preto";
- $D_2$: "Gato".

Para a palavra "gato", teríamos:

- $\text{TF}_{raw}(\text{"gato"}, D_1) = 2$;
- $\text{TF}_{raw}(\text{"gato"}, D_2) = 1$.

A frequência bruta, embora simples, apresenta um problema: ela não considera o tamanho do documento. No exemplo 1 acima, "gato" representa $100\%$ das palavras em $D_2$, mas apenas $25\%$ das palavras em $D_1$. A frequência bruta não consegue capturar essa diferença de importância relativa.

A frequência bruta pode ser enganosa, especialmente quando comparamos documentos de tamanhos diferentes. Em um documento pequeno, uma palavra pode ter uma frequência bruta alta, mas isso não significa que ela seja mais importante do que em um documento maior.

**Exemplo 2**: vamos tentar novamente usando um *corpus* com três documentos. Considere:

- $D_1$: "O gato preto caça o rato preto";
- $D_2$: "O rato branco corre do gato";
- $D_3$: "O cachorro late para o gato preto".

Para este *corpus*, teremos:

Para a palavra "gato":

- $\text{TF}_{raw}(\text{"gato"}, D_1) = 1$
- $\text{TF}_{raw}(\text{"gato"}, D_2) = 1$
- $\text{TF}_{raw}(\text{"gato"}, D_3) = 1$

Para a palavra "preto":

- $\text{TF}_{raw}(\text{"preto"}, D_1) = 2$
- $\text{TF}_{raw}(\text{"preto"}, D_2) = 0$
- $\text{TF}_{raw}(\text{"preto"}, D_3) = 1$

Para resolver a limitação da frequência bruta, utilizamos a frequência normalizada que pode ser definida como:

$$
\text{TF}(w, D_i) = \frac{f(w, D_i)}{\sum_{w' \in D_i} f(w', D_i)}
$$

O denominador $\sum_{w' \in D_i} f(w', D_i)$ representa o número total de palavras no documento $D_i$.

**Exemplo 3**: novamente considere o *corpus* dado por:  

$D_1$: "O pequeno gato preto viu outro gato preto";
$D_2$: "Gato".

Para a palavra "gato", teríamos:

Para $D_1$: "O pequeno gato preto viu outro gato preto" (8 palavras total)

- $f(\text{"gato"}, D_1) = 2$;
- $\sum_{w' \in D_1} f(w', D_1) = 8$;
- $\text{TF}(\text{"gato"}, D_1) = \frac{2}{8} = 0.25$.

Para $D_2$: "Gato" (1 palavra total)

- $f(\text{"gato"}, D_2) = 1$;
- $\sum_{w' \in D_2} f(w', D_1) = 1$;
- $\text{TF}(\text{"gato"}, D_2) = \frac{1}{1} = 1.0$.

A sagaz leitora pode ver que a frequência normalizada captura melhor a importância relativa da palavra em cada documento. Em $D_2$, "gato" tem frequência normalizada $1.0$, indicando que representa $100\%$ do documento, enquanto em $D_1$ tem frequência $0.25$, representando $25\%$ do documento.

**Exemplo 4**: voltando ao *corpus* com três documentos, teríamos:

Para $D_1$ (8 palavras total):

$$\text{TF}(\text{"preto"}, D_1) = \frac{2}{8} = 0.25$$

Para $D_2$ (7 palavras total):

$$\text{TF}(\text{"gato"}, D_2) = \frac{1}{7} \approx 0.143$$

Para $D_3$ (7 palavras total):

$$\text{TF}(\text{"preto"}, D_3) = \frac{1}{7} \approx 0.143$$

Além das frequências bruta e normalizada, existem outras formulações importantes do **TF**, criadas para resolver problemas específicos na representação de documentos.

##### Frequência Logarítmica

A frequência logarítmica é definida como:

$$
\text{TF}_{log}(w, D_i) = 1 + \log(f(w, D_i))
$$

Esta formulação é útil quando temos palavras com frequências muito diferentes em um mesmo documento. O logaritmo ajuda a *comprimir essas diferenças, evitando que palavras muito frequentes dominem completamente a representação*.

**Exemplo 5**: considere o documento:

$D_1$: "O cachorro late. O cachorro corre. O cachorro pula. O gato dorme."

Calculando **TF** para as palavras "cachorro" e "gato", teremos:

- $f(\text{"cachorro"}, D_1) = 3$;
- $f(\text{"gato"}, D_1) = 1$.

Com frequência bruta, a diferença é $3:1$

- $\text{TF}_{raw}(\text{"cachorro"}, D_1) = 3$;
- $\text{TF}_{raw}(\text{"gato"}, D_1) = 1$.

Com frequência logarítmica, a diferença é menor:

- $\text{TF}_{log}(\text{"cachorro"}, D_1) = 1 + \log(3) \approx 1.48$;
- $\text{TF}_{log}(\text{"gato"}, D_1) = 1 + \log(1) = 1$.

A frequência logarítmica reduz a razão de $3:1$ para aproximadamente $1.48:1$, suavizando a dominância de "cachorro". A redução da razão de $3:1$ para $1.48:1$ através da transformação logarítmica é matematicamente significativa porque reflete melhor como os humanos processam informação linguística. Uma palavra que aparece $3$ vezes em um texto não é necessariamente $3$ vezes mais importante para o significado do documento do que uma palavra que aparece apenas uma vez.

**Exemplo 6**: voltando ao *corpus* de três documentos, teremos:

- $D_1$: "O gato preto caça o rato preto";
- $D_2$: "O rato branco corre do gato";
- $D_3$: "O cachorro late para o gato preto".

Para "preto" em $D_1$:

$$
\text{TF}_{log}(\text{"preto"}, D_1) = 1 + \log(2) \approx 1.301
$$

Para "gato" em $D_2$:

$$
\text{TF}_{log}(\text{"gato"}, D_2) = 1 + \log(1) = 1
$$

Esta propriedade de compressão do logaritmo é particularmente valiosa em processamento de linguagem natural porque segue a Lei de Weber-Fechner, que estabelece que a percepção humana de diferenças tende a ser logarítmica em relação ao estímulo real: da mesma forma que não percebemos uma luz com $300$ lumens como sendo 3 vezes mais brilhante que uma com $100$ lumens, também não interpretamos uma palavra que aparece $3$ vezes em um texto como tendo o triplo da importância semântica de uma palavra que aparece uma única vez.

>A propriedade de compressão logarítmica, frequentemente utilizada em processamento de linguagem natural, encontra justificativa na **Lei de Weber-Fechner**. Esta lei psicofísica fundamental descreve a relação entre a magnitude física de um estímulo e a percepção humana da intensidade desse estímulo.
>
>Em essência, a Lei de Weber-Fechner postula que a percepção sensorial é proporcional ao logaritmo da intensidade real do estímulo. Matematicamente, isso pode ser expresso como:
>
>$$ P = k \cdot \log(S/S_0) $$
>
>Na qual, temos:
>
>- $P$ representa a magnitude da percepção.
>- $S$ é a intensidade do estímulo.
>- $S_0$ é o limiar de detecção do estímulo (o menor estímulo>que pode ser percebido).
>- $k$ é uma constante de proporcionalidade, dependente do tipo de estímulo e da modalidade sensorial.
>
>A **Lei de Weber-Fechner** implica que a nossa percepção de mudanças em estímulos não é linear, mas sim logarítmica. Em vez de percebermos aumentos aritméticos na intensidade de um estímulo como aumentos lineares na percepção, percebemos aumentos geométricos como aumentos aritméticos.
>
>No contexto do processamento de linguagem natural e, especificamente, no algoritmo **TF-IDF**, essa lei se torna relevante ao justificar a aplicação do logaritmo na frequência dos termos (**TF**) e na frequência inversa nos documentos (**IDF**). Assim como a percepção humana não escala linearmente com a intensidade da luz, também não interpretamos a importância de uma palavra em um texto de forma linear com sua frequência bruta.
>
>Por exemplo, uma luz com $300$ lumens não será percebida como três vezes mais brilhante que uma com $100$ lumens. De maneira análoga, uma palavra que aparece $3$ vezes em um documento não será necessariamente considerada três vezes mais importante semanticamente do que uma palavra que aparece apenas uma vez. A aplicação do logaritmo na frequência dos termos e documentos no **TF-IDF** busca modelar essa percepção humana não linear da importância, dando menos peso a aumentos lineares na frequência e enfatizando a presença da palavra, mesmo que não seja extremamente frequente.
>
>Embora o trabalho de [Weber](https://www.britannica.com/biography/Ernst-Heinrich-Weber)[^9] tenha dado início a pequisas que levou a Lei Weber-Fechner. [Gustav Theodor Fechner](https://www.britannica.com/biography/Gustav-Fechner) expandiu e formalizou as ideias de [Ernst Heinrich Weber](https://www.britannica.com/biography/Ernst-Heinrich-Weber) em sua obra posterior[^10]. A formulação matemática mais completa e a denominação da lei são geralmente atribuídas a Fechner.
>
>Este princípio psicofísico, portanto, oferece uma base teórica para a utilização da compressão logarítmica em técnicas de processamento de linguagem natural como **TF-IDF**, alinhando o processamento computacional de texto com a forma como os humanos percebem e interpretam a informação.

[^9]WEBER, E. H. **De tactu: Annotationes anatomicae et physiologicae**. Lipsiae: Koehler, 1834.

[^10]: FECHNER, G. T. **Elemente der Psychophysik.** Leipzig: Breitkopf und Härtel, 1860.

##### Frequência Binária

A frequência binária simplifica a representação para apenas presença ou ausência:

$$
\text{TF}_{binary}(w, D_i) = \begin{cases} 1 & \text{se } f(w, D_i) > 0 \\ 0 & \text{caso contrário} \end{cases}
$$

Esta formulação é útil quando a presença de uma palavra é mais importante que sua frequência. Por exemplo, quando estamos interessados na classificação de textos por tópicos ou na análise de palavras-chave.

**Exemplo 7**: para que a amável leitora entenda, considere dois documentos:

- $D_1$: "Python é uma linguagem de programação. Python é versátil. Python é popular";
- $D_2$: "Java é uma linguagem de programação".

Se calcularmos **TF** a palavra "linguagem", teremos:

- $\text{TF}_{raw}(\text{"linguagem"}, D_1) = 1$;
- $\text{TF}_{raw}(\text{"linguagem"}, D_2) = 1$;
- $\text{TF}_{binary}(\text{"linguagem"}, D_1) = 1$;
- $\text{TF}_{binary}(\text{"linguagem"}, D_2) = 1$.

Para a palavra "Python":

- $\text{TF}_{raw}(\text{"Python"}, D_1) = 3$;
- $\text{TF}_{raw}(\text{"Python"}, D_2) = 0$;
- $\text{TF}_{binary}(\text{"Python"}, D_1) = 1$;
- $\text{TF}_{binary}(\text{"Python"}, D_2) = 0$.

A frequência binária indica apenas que $D_1$ é sobre Python e $D_2$ não é, ignorando a repetição da palavra.

**Exemplo 8**: podemos aplicar a frequência binária ao **corpus** com três documentos que usamos anteriormente:

$D_1$: "O gato preto caça o rato preto";
$D_2$: "O rato branco corre do gato";
$D_3$: "O cachorro late para o gato preto".

Para "rato":

- $\text{TF}_{binary}(\text{"rato"}, D_1) = 1$ (presente);
- $\text{TF}_{binary}(\text{"rato"}, D_2) = 1$ (presente);
- $\text{TF}_{binary}(\text{"rato"}, D_3) = 0$ (ausente).

##### Frequência Aumentada (Augmented Frequency)

A frequência aumentada é definida como:

$$
\text{TF}_{aug}(w, D_i) = 0.5 + 0.5 \times \frac{f(w, D_i)}{\max_{w' \in D_i} f(w', D_i)}
$$

Nesta fórmula temos:

- $f(w, D_i)$ é a frequência bruta do termo $w$ no documento $D_i$;
- $max_{w' ∈ D_i} f(w', D_i)$ é a frequência do termo mais frequente em $D_i$;
- $0,5$ é um fator de amortecimento que impede que termos que aparecem muitas vezes dominem completamente a ponderação.

Usamos esta fórmula quando queremos considerar a frequência relativa das palavras, mas evitar que documentos longos dominem apenas por seu tamanho. O termo $0.5$ garante um peso mínimo para qualquer palavra presente.

**Exemplo 9**: considere o documento: $D_1$: "O gato preto viu outro gato preto. O gato dormiu."

Para este documento:

- $f(\text{"gato"}, D_1) = 3$ (palavra mais frequente);
- $f(\text{"preto"}, D_1) = 2$;
- $f(\text{"dormiu"}, D_1) = 1$.

Calculando $\text{TF}_{aug}$:

- $\text{TF}_{aug}(\text{"gato"}, D_1) = 0.5 + 0.5 \times \frac{3}{3} = 1.0$;
- $\text{TF}_{aug}(\text{"preto"}, D_1) = 0.5 + 0.5 \times \frac{2}{3} \approx 0.83$;
- $\text{TF}_{aug}(\text{"dormiu"}, D_1) = 0.5 + 0.5 \times \frac{1}{3} \approx 0.67$.

Observe que mesmo "dormiu", que aparece apenas uma vez, recebe um peso considerável $(0.67)$ devido ao termo base $0.5$.

**Exemplo 10**: usando o **corpus** com três documentos:

- $D_1$: "O gato preto caça o rato preto";
- $D_2$: "O rato branco corre do gato";
- $D_3$: "O cachorro late para o gato preto".

Para $D_1$, a palavra mais frequente aparece 2 vezes ("preto"):

Para "gato":

$$
\text{TF}_{aug}(\text{"gato"}, D_1) = 0.5 + 0.5 \times \frac{1}{2} = 0.75
$$

Para "preto":

$$
\text{TF}_{aug}(\text{"preto"}, D_1) = 0.5 + 0.5 \times \frac{2}{2} = 1.0
$$

A frequência aumentada se destaca sistemas de recuperação de informação e mecanismos de busca onde temos documentos de tamanhos muito diferentes, como por exemplo, ao comparar artigos científicos com resumos ou *abstracts*.

Ao utilizar a fórmula da frequência aumentada, garantimos que mesmo palavras que aparecem apenas uma vez ainda recebam um peso mínimo de $0.5$, enquanto a palavra mais frequente no documento recebe peso $1.0$, criando assim uma distribuição mais equilibrada que é valiosa quando precisamos comparar documentos de diferentes comprimentos ou quando queremos evitar que documentos longos dominem os resultados graças ao seu tamanho. Um problema comum em sistemas de busca acadêmica, bibliotecas digitais e bases de dados documentais, onde a variação no tamanho dos documentos pode ser significativa.

Além das quatro formas que estudamos ($\text{TF}_{raw}$, $\text{TF}_{log}$, $\text{TF}_{binary}$ e $\text{TF}_{aug}$), existem várias outras formulações para o cálculo do **TF**. Uma variação particularmente interessante é a frequência $K$-modificada, definida como:

$$ \text{TF}_K(w, D_i) = K + (1-K) \times \frac{f(w, D_i)}{\max_{w' \in D_i} f(w', D_i)} $$

Na qual, teremos:

- $K$ é um parâmetro ajustável no intervalo $[0,1]$;
- $f(w, D_i)$ é a frequência bruta do termo $w$ no documento $D_i$;
- $\max_{w' \in D_i} f(w', D_i)$ é a frequência bruta máxima de qualquer termo em $D_i$;
- $(1-K)$ serve como um fator de escala para a normalização da frequência do termo;

Observe que esta fórmula é uma generalização da frequência aumentada, que é um caso especial onde $K = 0.5$. Outra variação importante é a frequência logarítmica suavizada:

$$
\text{TF}_{log-smooth}(w, D_i) = \begin{cases} 1 + \log(1 + \log(f(w, D_i))) & \text{se } f(w, D_i) > 0 \\ 0 & \text{caso contrário} \end{cases}
$$

que aplica uma dupla transformação logarítmica para uma suavização ainda mais agressiva. Também existe a frequência normalizada por comprimento:

$$
\text{TF}_{length}(w, D_i) = \frac{f(w, D_i)}{\sqrt{\sum_{w' \in D_i} f(w', D_i)^2}}
$$

que usa a norma euclidiana do documento como fator de normalização, sendo particularmente útil quando trabalhamos com vetores em espaços de alta dimensão. A escolha entre estas diferentes formulações geralmente depende do domínio específico e das características do corpus, sendo comum em sistemas modernos o uso de várias formulações em conjunto, combinadas através de técnicas de *ensemble* ou votação ponderada.

>Técnicas de *ensemble* e votação ponderada são métodos de combinação de diferentes abordagens para obter resultados mais robustos e confiáveis. Imagine um grupo de especialistas médicos analisando um caso complexo. Cada um tem sua expertise e perspectiva única, e a decisão final considera a opinião de todos, dando mais peso aos especialistas com mais experiência em aspectos específicos do caso. No contexto do **TF-IDF**, estas técnicas funcionam de maneira similar: diferentes fórmulas de **TF** são calculadas para o mesmo texto (como $\text{TF}_{log}$, $\text{TF}_{aug}$, etc.), e o resultado final é uma combinação ponderada desses valores. Por exemplo, podemos dar mais peso ao $\text{TF}_{aug}$ quando trabalhamos com documentos de tamanhos muito diferentes, e mais peso ao $\text{TF}_{log}$ quando lidamos com palavras de frequências muito variadas. Esta abordagem é particularmente poderosa porque combina as vantagens de diferentes formulações, compensando as limitações individuais de cada método e produzindo resultados mais estáveis e precisos em uma variedade maior de situações.
>
>Quase deixo passar! A palavra *ensemble* vem do francês e significa conjunto, ou em conjunto. No contexto de análise de dados e aprendizado de máquina, uma técnica de *ensemble* refere-se a um método que combina múltiplos modelos ou abordagens diferentes para criar uma solução mais robusta. Como uma orquestra que, não por coincidência, também usa a palavra *ensemble*.
>
>Em uma orquestra cada instrumento contribui com seu som único, e juntos criam uma música mais rica e complexa do que qualquer instrumento sozinho poderia produzir. Da mesma forma, quando falamos de *ensemble* no contexto do **TF-IDF**, estamos falando de combinar diferentes fórmulas de cálculo do **TF** ou do **IDF**, cada uma com suas características específicas, para obter um resultado mais preciso e confiável.

##### Escolhendo a Formulação Adequada

No momento de definir sua aplicação, a analítica leitora deve considerar:

1. **Natureza dos Documentos**:

   - Documentos longos → Frequência normalizada ou aumentada;
   - Documentos curtos → Frequência bruta pode ser adequada.

2. **Objetivo da Análise**:

   - Classificação de tópicos → Frequência binária;
   - Recuperação de informação → Frequência logarítmica;
   - Análise de similaridade → Frequência aumentada.

3. **Características do Vocabulário**:

   - Palavras com frequências muito diferentes → Frequência logarítmica;
   - Palavras-chave importantes → Frequência binária ou aumentada.

A esforçada leitora deverá experimentar diferentes formulações em seu conjunto de dados específico, avaliando o impacto de cada uma no desempenho final do seu sistema. *Triste é a sina daquela que pesquisa e aprende*.

Um pouco de código em C++ deve ajudar a amável leitora a entender melhor o conceito de **TF**. O código tenta seguir as fórmulas matemáticas apresentadas no texto, demonstrando como traduzir expressões matemáticas para código C++.

Não deixe de notar que a implementação está simplificada e não considera tarefas de pré-processamento como tokenização, remoção de *stopwords* ou *stemming*, que seriam importantes em aplicações reais e que serão discutidos em detalhes mais adiante.

O código a seguir utiliza a biblioteca [Eigen](https://eigen.tuxfamily.org/dox/), utilizada em aplicações científicas e de aprendizado de máquina. A biblioteca é leve e fácil de usar, permitindo operações eficientes com matrizes e vetores em alto desempenho e com precisão numérica.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

// Função para calcular diferentes versões do TF para um documento
class TermFrequency {
public:
    // Calcula a frequência bruta (raw)
    static double raw(const std::vector<std::string>& document, const std::string& term) {
        double count = 0;
        for (const auto& word : document) {
            if (word == term) {
                count++;
            }
        }
        return count;
    }
    
    // Calcula a frequência normalizada
    static double normalized(const std::vector<std::string>& document, const std::string& term) {
        double count = raw(document, term);
        return count / document.size();
    }
    
    // Calcula a frequência logarítmica
    static double logarithmic(const std::vector<std::string>& document, const std::string& term) {
        double count = raw(document, term);
        return count > 0 ? 1.0 + std::log(count) : 0.0;
    }
    
    // Calcula a frequência binária
    static double binary(const std::vector<std::string>& document, const std::string& term) {
        for (const auto& word : document) {
            if (word == term) {
                return 1.0;
            }
        }
        return 0.0;
    }
    
    // Calcula a frequência aumentada (augmented)
    static double augmented(const std::vector<std::string>& document, const std::string& term) {
        std::unordered_map<std::string, int> termCounts;
        
        // Contar ocorrências de cada termo
        for (const auto& word : document) {
            termCounts[word]++;
        }
        
        // Encontrar o termo mais frequente
        int maxCount = 0;
        for (const auto& [_, count] : termCounts) {
            maxCount = std::max(maxCount, count);
        }
        
        int termCount = termCounts[term];
        return 0.5 + 0.5 * (static_cast<double>(termCount) / maxCount);
    }
};

// Exemplo de uso
int main() {
    // Documento do Exemplo 3 no texto
    std::vector<std::string> doc1 = {"o", "pequeno", "gato", "preto", "viu", "outro", "gato", "preto"};
    std::vector<std::string> doc2 = {"gato"};
    
    std::string term = "gato";
    
    std::cout << "Para o termo '" << term << "':\n";
    
    std::cout << "Documento 1:\n";
    std::cout << "  TF_raw: " << TermFrequency::raw(doc1, term) << "\n";
    std::cout << "  TF_normalized: " << TermFrequency::normalized(doc1, term) << "\n";
    std::cout << "  TF_logarithmic: " << TermFrequency::logarithmic(doc1, term) << "\n";
    std::cout << "  TF_binary: " << TermFrequency::binary(doc1, term) << "\n";
    std::cout << "  TF_augmented: " << TermFrequency::augmented(doc1, term) << "\n";
    
    std::cout << "Documento 2:\n";
    std::cout << "  TF_raw: " << TermFrequency::raw(doc2, term) << "\n";
    std::cout << "  TF_normalized: " << TermFrequency::normalized(doc2, term) << "\n";
    std::cout << "  TF_logarithmic: " << TermFrequency::logarithmic(doc2, term) << "\n";
    std::cout << "  TF_binary: " << TermFrequency::binary(doc2, term) << "\n";
    std::cout << "  TF_augmented: " << TermFrequency::augmented(doc2, term) << "\n";
    
    return 0;
}
```

#### Inverse Document Frequency (IDF)

A Frequência Inversa nos Documentos (**IDF**, do inglês *Inverse Document Frequency*) foi necessária para criar um nível de refinamento para a medida de frequência de termos (**TF**).  Enquanto o **TF** quantifica a importância de um termo dentro de um documento específico, ele não considera a distribuição desse termo em uma determinada coleção de documentos.

*A ideia que sustenta a criação do **IDF** é que termos que aparecem frequentemente em todos os documentos de uma coleção são menos relevantes para distinguir documentos uns dos outros*. Termos como "o", "e", "de", por exemplo, são muito comuns na maioria dos textos em português, mas raramente carregam um peso semântico suficientemente significativo na diferenciação entre dois documento. Atribuir alta importância a essas palavras com base apenas em sua frequência dentro de um documento, **TF**, pode levar a resultados de busca e análise de texto aquém do esperado para uma função específica.

O conceito de **IDF** foi introduzido por [Karen Spärck Jones](https://www.historyofdatascience.com/karen-sparck-jones-the-search-engineer-enabler/) em seu artigo seminal de 1972, intitulado *A statistical interpretation of term specificity and its application in retrieval*[^11]. Neste trabalho, Spärck Jones argumentou que a especificidade de um termo, ou seja, quão distintivo ele é dentro de uma coleção de documentos, deveria ser levada em conta para melhorar a recuperação de informação. O **IDF** é uma medida dessa especificidade.

[^11]: SPÄRCK JONES, K. S. **A statistical interpretation of term specificity and its application in retrieval**. *Journal of Documentation*, v. 28, n. 1, p. 11–21, 1972.

O cálculo do **IDF** é definido pela seguinte fórmula:

$$
\text{IDF}(w, Docs) = \log \left( \frac{|Docs|}{|\{D_j \in Docs: w \in D_j\}|} \right)
$$

Na qual, temos:

- $|Docs|$ representa o número total de documentos na coleção;
- $|\{D_j \in Docs: w \in D_j\}|$ é o número de documentos que contêm a palavra $w$. Este valor representa a frequência documental da palavra $w$;
- $\log$ é o logaritmo natural (base $e$).

**Exemplo 1**: considere o **corpus** com três documentos que temos usado:

$D_1$: "O gato preto caça o rato preto"
$D_2$: "O rato branco corre do gato"
$D_3$: "O cachorro late para o gato preto"

Para "gato" (aparece nos 3 documentos):

$$\text{IDF}(\text{"gato"}, Docs) = \log \left(\frac{3}{3}\right) = \log(1) = 0$$

Para "preto" (aparece em 2 documentos):

$$\text{IDF}(\text{"preto"}, Docs) = \log \left(\frac{3}{2}\right) \approx 0.405$$

Para "cachorro" (aparece em 1 documento):

$$\text{IDF}(\text{"cachorro"}, Docs) = \log \left(\frac{3}{1}\right) \approx 1.099$$

A função logarítmica na fórmula do **IDF** desempenha dois papéis importantes que precisamos destacar:

1. **Suavização da divisão**: a aplicação do logaritmo atenua o impacto de grandes variações no denominador. Isso é particularmente útil quando se lida com coleções de documentos muito grandes, onde a frequência documental de alguns termos pode ser muito baixa, levando a valores de IDF muito altos sem a suavização logarítmica.

2. **Garantia de baixo peso para palavras comuns**: a função logarítmica garante que palavras muito comuns, que aparecem em um grande número de documentos (fazendo com que o denominador $|\{D_j \in Docs: w \in D_j\}|$ se aproxime de $|Docs|$), resultem em valores de **IDF** próximos de zero.  Idealmente, se uma palavra aparece em todos os documentos, o termo dentro do logaritmo se torna 1, e $\log(1) = 0$, atribuindo um **IDF** nulo a essa palavra, indicando sua falta de poder discriminatório.

Em resumo, o **IDF** atua como um fator de ponderação que diminui a importância de termos que são comuns em muitos documentos e aumenta a importância de termos que são raros e, portanto, mais específicos e discriminativos dentro da coleção. Ao combinar o **IDF** com o **TF**, o algoritmo **TF-IDF** consegue identificar termos que são importantes em um documento específico e, ao mesmo tempo, distintivos em relação à coleção de documentos como um todo, melhorando significativamente a eficácia em tarefas como recuperação de informação e análise de texto.

Existem diferentes versões da fórmula **IDF**. Embora o princípio básico permaneça o mesmo, reduzir o peso de termos que aparecem frequentemente em toda a coleção de documentos, existem variações para abordar nuances específicas ou melhorar o desempenho em certos cenários. Aqui está uma análise de algumas versões comuns de **IDF**:

##### IDF Básico (conforme descrito anteriormente)

Esta é a versão mais fundamental e amplamente utilizada, frequentemente referida como IDF padrão.

$$
\text{IDF}(w, Docs) = \log \left( \frac{|Docs|}{|\{D_j \in Docs: w \in D_j\}|} \right)
$$

**Exemplo 2**: voltando ao **corpus** com três documentos:

$D_1$: "O gato preto caça o rato preto"
$D_2$: "O rato branco corre do gato"
$D_3$: "O cachorro late para o gato preto"

Para "gato" (aparece nos 3 documentos):

$$\text{IDF}(\text{"gato"}, Docs) = \log \left(\frac{3}{3}\right) = \log(1) = 0$$

Para "preto" (aparece em 2 documentos):

$$\text{IDF}(\text{"preto"}, Docs) = \log \left(\frac{3}{2}\right) \approx 0.405$$

Para "cachorro" (aparece em 1 documento):

$$\text{IDF}(\text{"cachorro"}, Docs) = \log \left(\frac{3}{1}\right) \approx 1.099$$

Esta é uma versão simples de entender e implementar, funciona bem em muitos casos gerais. Todavia, pode ser instável quando um termo não aparece em nenhum documento da coleção. Neste caso, o denominador torna-se zero, levando à divisão por zero. Além disso, não leva em conta a frequência do termo nos documentos onde ele aparece, apenas a contagem de documentos.

##### IDF com Suavização (IDF+1)

Para evitar a divisão por zero quando um termo não está presente em nenhum documento e para atenuar ligeiramente o efeito **IDF** para termos muito raros, um fator de suavização é frequentemente adicionado ao denominador. Uma abordagem comum é adicionar $1$ à contagem da frequência documental resultando em:

$$
\text{IDF}(w, Docs) = \log \left( \frac{|Docs|}{1 + |\{D_j \in Docs: w \in D_j\}|} \right)
$$

**Exemplo 3**: novamente com o **corpus** de três documentos:

$D_1$: "O gato preto caça o rato preto"
$D_2$: "O rato branco corre do gato"
$D_3$: "O cachorro late para o gato preto"

Para "gato" (aparece nos 3 documentos):

$$\text{IDF}(\text{"gato"}, Docs) = \log \left(\frac{3}{3}\right) = \log(1) = 0$$

Para "preto" (aparece em 2 documentos):

$$\text{IDF}(\text{"preto"}, Docs) = \log \left(\frac{3}{2}\right) \approx 0.405$$

Para "cachorro" (aparece em 1 documento):

$$\text{IDF}(\text{"cachorro"}, Docs) = \log \left(\frac{3}{1}\right) \approx 1.099$$

Esta fórmula impede a divisão por zero, fornece um valor **IDF** não nulo mesmo para termos não presentes na coleção e reduz ligeiramente o impacto de termos muito raros. Trata-se de uma formulação mais estável numericamente do que o **IDF** básico. Esta fórmula também não considera a frequência do termo dentro dos documentos.

##### IDF Probabilístico

Esta versão visa estimar a probabilidade de um documento ser relevante dado um termo[^12], e vice-versa. Uma fórmula IDF probabilística comum é:

$$
\text{IDF}(w, Docs) = \log \left( \frac{|Docs| - |\{D_j \in Docs: w \in D_j\}|}{|\{D_j \in Docs: w \in D_j\}|} \right)
$$

Ou, por vezes, vista como:

$$
\text{IDF}(w, Docs) = \log \left( \frac{N - df_w}{df_w} \right) = \log \left( \frac{N}{df_w} - 1 \right)
$$

Nesta fórmula, $N = |Docs|$ e $df_w = |\{D_j \in Docs: w \in D_j\}|$, frequência documental da palavra $w$.

[^12]: ROBERTSON, S. E.; JONES, K. S. **Relevance weighting of search terms.** journal of the American Society for Information Science, v. 45, n. 3, p. 129–146, 1994.

**Exemplo 4**: usando o **corpus** de três documentos:

- $D_1$: "O gato preto caça o rato preto";
- $D_2$: "O rato branco corre do gato";
- $D_3$: "O cachorro late para o gato preto".

Para "gato":

$$
\text{IDF}(\text{"gato"}, Docs) = \log \left(\frac{3 - 3}{3}\right) = \log(0)
$$

(Indefinido, mostrando uma limitação desta formulação)

Para "preto":

$$
\text{IDF}(\text{"preto"}, Docs) = \log \left(\frac{3 - 2}{2}\right) = \log(0.5) \approx -0.301
$$

O **IDF probabilístico** com sua interpretação probabilística, potencialmente reflete melhor o caráter informativo de termos raros em alguns cenários. Contudo, pode resultar em valores **IDF** negativos se o termo aparecer em mais de metade dos documentos, o que pode não ser intuitivo. Além disso, o **IDF** probabilístico é sensível à frequência documental.

##### IDF Máximo

Esta variação define um limite superior para o valor **IDF**. Baseia-se na ideia de que, após um certo ponto, aumentar os valores **IDF** para termos extremamente raros pode não ser benéfico e pode até ser prejudicial. Uma abordagem é usar o valor IDF máximo observado na coleção como o limite prático da sua aplicação:

$$
\text{IDF}_{\text{max}}(w, Docs) = \min \left( \text{IDF}(w, Docs), \max_{w' \in Vocabulary} \{\text{IDF}(w', Docs)\} \right)
$$

Neste caso, $\text{IDF}(w, Docs)$ é tipicamente o **IDF** básico ou **IDF+1**.

O **IDF Máximo**, ou **IDF Max**, limita a influência de termos extremamente raros, melhorando potencialmente a robustez e prevenindo o *overfitting* a ruído nos dados. Porém, introduz um corte relativamente arbitrário, o que pode ser prejudicial em alguns conjuntos de dados.

##### IDF Normalizado por Comprimento (Length-Normalized IDF)

Em alguns casos, o comprimento do documento pode influenciar a frequência do termo e a frequência documental. Documentos mais longos têm maior probabilidade de conter qualquer termo dado, potencialmente inflando as frequências documentais para termos comuns. A normalização por comprimento pode ser aplicada também ao IDF, embora seja menos comum do que a normalização por comprimento para TF. Uma abordagem é normalizar a frequência documental pelo comprimento médio do documento:

Esta versão é menos formalmente definida e mais um conceito. Geralmente envolveria o ajuste da contagem da frequência documental com base no comprimento do documento, mas as fórmulas específicas variam.

- **Prós:** Potencialmente mitiga o efeito do viés do comprimento do documento nos valores IDF;
- **Contras:** Adiciona complexidade, e o benefício pode ser marginal em comparação com a normalização por comprimento aplicada ao TF.

##### Escolhendo a versão IDF

A esforçada leitora terá alguma dificuldade de encontrar a versão do **IDF** adequada ao seu projeto. Isso irá requerer empiricismo e paciência. Contudo, para a recuperação de texto de propósito geral, o **IDF básico ou IDF+1** são frequentemente suficientes e fornecem um bom equilíbrio entre simplicidade e eficácia. Da mesma forma, o **IDF probabilístico** deve ser considerado quando uma interpretação probabilística é desejada, ou quando se lida com conjuntos de dados nos quais se espera que termos raros sejam altamente informativos.

Finalmente, considere que o **IDF máximo** pode ser útil em cenários onde a robustez a *outliers* e termos extremamente raros seja importante e o **IDF normalizado por comprimento**, embora seja menos comum, pode ser avaliado se, durante os seus testes, o comprimento do documento for suspeito de ser um fator de confusão significativo.

Novamente, para que a esforçada leitora não se perca na matemática pura, um pouco de código em C++ pode ajudar.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <Eigen/Dense>

class InverseDocumentFrequency {
public:
    // Calcula IDF básico
    static double basic(const std::vector<std::vector<std::string>>& corpus, const std::string& term) {
        double N = corpus.size(); // número total de documentos
        double df = 0;           // document frequency
        
        for (const auto& document : corpus) {
            for (const auto& word : document) {
                if (word == term) {
                    df++;
                    break; // contamos o documento apenas uma vez
                }
            }
        }
        
        // Evitar divisão por zero
        if (df == 0) return 0.0;
        
        return std::log(N / df);
    }
    
    // Calcula IDF com suavização (IDF+1)
    static double smooth(const std::vector<std::vector<std::string>>& corpus, const std::string& term) {
        double N = corpus.size();
        double df = 0;
        
        for (const auto& document : corpus) {
            std::unordered_set<std::string> uniqueTerms(document.begin(), document.end());
            if (uniqueTerms.find(term) != uniqueTerms.end()) {
                df++;
            }
        }
        
        return std::log(N / (1 + df));
    }
    
    // Calcula IDF probabilístico
    static double probabilistic(const std::vector<std::vector<std::string>>& corpus, const std::string& term) {
        double N = corpus.size();
        double df = 0;
        
        for (const auto& document : corpus) {
            std::unordered_set<std::string> uniqueTerms(document.begin(), document.end());
            if (uniqueTerms.find(term) != uniqueTerms.end()) {
                df++;
            }
        }
        
        // O IDF probabilístico pode ter problemas se df = N
        if (df == N) return 0.0;
        
        return std::log((N - df) / df);
    }
    
    // Calcula todas as variantes do IDF para um conjunto de termos
    static void calculateAllVariants(const std::vector<std::vector<std::string>>& corpus, 
                                    const std::vector<std::string>& terms) {
        std::cout << "Termo\t\tIDF Básico\tIDF Suavizado\tIDF Probabilístico\n";
        std::cout << "-----------------------------------------------------------\n";
        
        for (const auto& term : terms) {
            double basic_idf = basic(corpus, term);
            double smooth_idf = smooth(corpus, term);
            double prob_idf = probabilistic(corpus, term);
            
            std::cout << term << "\t\t" 
                     << basic_idf << "\t\t" 
                     << smooth_idf << "\t\t";
            
            // Verificar se o IDF probabilístico está definido
            if (std::isnan(prob_idf) || std::isinf(prob_idf)) {
                std::cout << "Indefinido";
            } else {
                std::cout << prob_idf;
            }
            
            std::cout << "\n";
        }
    }
};

int main() {
    // Corpus do Exemplo 1 na seção IDF
    std::vector<std::vector<std::string>> corpus = {
        {"o", "gato", "preto", "caça", "o", "rato", "preto"},
        {"o", "rato", "branco", "corre", "do", "gato"},
        {"o", "cachorro", "late", "para", "o", "gato", "preto"}
    };
    
    std::vector<std::string> terms = {"gato", "preto", "cachorro", "rato"};
    
    InverseDocumentFrequency::calculateAllVariants(corpus, terms);
    
    return 0;
}
```

Este exemplo implementa três variantes do cálculo de IDF discutidas no texto: **IDF básico**, **IDF com suavização** e **IDF probabilístico**.

A classe `InverseDocumentFrequency` encapsula estes métodos de cálculo. A implementação usa estruturas de dados eficientes como `unordered_set` para verificar rapidamente a presença de termos em documentos, evitando contagens duplicadas. O método `calculateAllVariants` gera uma tabela formatada que mostra os valores de **IDF** para cada termo usando as diferentes formulações, facilitando a comparação.

Note que o código inclui tratamento para o caso em que o **IDF probabilístico** pode se tornar indefinido, quando um termo aparece em todos os documentos, mostrando como lidar com casos especiais que podem surgir nas aplicações reais.

#### Cálculo do TF-IDF

Finalmente chegamos a informação que nos interessa. A pontuação final **TF-IDF** será o resultado do produto das duas métricas:

$$
\text{TF-IDF}(w, D_i, Docs) = \text{TF}(w, D_i) \times \text{IDF}(w, Docs)
$$

Na qual, temos:

- $w$ é um termo (palavra) do nosso vocabulário $V$;
- $D_i$ é o $i$-ésimo documento em nossa coleção;
- $Docs$ é nossa coleção completa de documentos, *corpus*;
- $\text{TF}(w, D_i)$ representa a frequência do termo $w$ no documento $D_i$;
- \text{IDF}(w, Docs) é a frequência inversa do documento w no corpus.

Esta fórmula tem propriedades matemáticas interessantes:

1. Se uma palavra ocorre muito em um documento ($\text{TF}$ alto) mas é comum no *corpus* ($\text{IDF}$ baixo), sua pontuação **TF-IDF** será moderada;
2. Se uma palavra ocorre muito em um documento ($\text{TF}$ alto) e é rara no *corpus* ($\text{IDF}$ alto), sua pontuação **TF-IDF** será alta;
3. Se uma palavra ocorre pouco em um documento ($\text{TF}$ baixo), sua pontuação **TF-IDF** será baixa independentemente do $\text{IDF}$.

Contudo, precisamos voltar a matemática. Antes que a perspicaz leitora possa ver um exemplo detalhado do uso do **TF-IDF** precisamos ver **Similaridade de Cosseno**.

##### Similaridade de Cosseno

A **Similaridade de Cosseno** é uma medida que permite quantificar o quão similares dois vetores são, baseando-se no ângulo entre eles. Em um espaço vetorial de documentos, onde cada dimensão representa um termo, a **Similaridade de Cosseno** *fornece uma medida da similaridade semântica entre os documentos, independente do tamanho dos documentos*.

Para dois vetores $\vec{a}$ e $\vec{b}$, a **Similaridade de Cosseno** será definida como:

$$
\text{sim}(\vec{a},\vec{b}) = \cos(\theta) = \frac{\vec{a} \cdot \vec{b} }{|\vec{a}| |\vec{b}|} = \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \sqrt{\sum_{i=1}^n b_i^2} }
$$

Nesta fórmula temos:

- $\vec{a}$ e $\vec{b}$ são vetores $n$-dimensionais;
- $a_i$ e $b_i$ são os $i$-ésimos componentes dos vetores $\vec{a}$ e $\vec{b}$ respectivamente;
- $n$ é a dimensão dos vetores (tamanho do vocabulário);
- $\theta$ é o ângulo entre os dois vetores;
- $|\vec{a}|$ e $|\vec{b}|$ representam as normas euclidianas dos vetores.

O resultado da aplicação desta fórmula irá variar entre $-1$ e $1$, de tal forma que:

- $1$ indica vetores apontando na mesma direção, documentos muito similares;
- $0$ indica vetores perpendiculares, documentos sem relação;
- $-1$ indica vetores em direções opostas, documentos completamente diferentes.

No contexto de análise de documentos em processamento de linguagem natural, como trabalhamos com frequências não-negativas, o resultado estará sempre entre $0$ e $1$.

**Exemplo 1**: comparação Simples. Considere um **corpus** formado por três documentos curtos:

- $D_1$: "O gato caça rato";
- $D_2$:"O gato dorme muito";
- $D_3$: "O rato come queijo".

Após aplicar **TF-IDF** e considerando apenas os termos $[\text{"gato", "rato", "caça", "dorme", "come", "queijo"}]$, obtemos os vetores:

- $\vec{D_1} = [0.4, 0.4, 0.5, 0, 0, 0]$;
- $\vec{D_2} = [0.4, 0, 0, 0.5, 0, 0]$;
- $\vec{D_3} = [0, 0.4, 0, 0, 0.5, 0.4]$.

Calculando a similaridade entre $\vec{D_1}$ e $\vec{D_2}$:

$$
\text{sim}(\vec{D_1},\vec{D_2}) = \frac{(0.4 \times 0.4) + 0 + 0 + 0 + 0 + 0}{\sqrt{0.4^2 + 0.4^2 + 0.5^2} \sqrt{0.4^2 + 0^2 + 0^2 + 0.5^2} } \approx 0.45
$$

Enquanto entre $\vec{D_1}$ e $\vec{D_3}$:

$$\text{sim}(\vec{D_1},\vec{D_3}) = \frac{0 + (0.4 \times 0.4) + 0 + 0 + 0 + 0} { \sqrt{0.4^2 + 0.4^2 + 0.5^2} \sqrt{0.4^2 + 0^2 + 0.5^2 + 0.4^2} } \approx 0.25$$

**Exemplo 2**: aplicação em Busca de Documentos

Imagine agora uma consulta $q$ = "gato caçador". Após processamento TF-IDF:

$\vec{Q} = [0.6, 0, 0.4, 0, 0, 0]$

Calculando a similaridade com cada documento:

$$ \text{sim}(\vec{Q},\vec{D_1}) = \frac{(0.6 \times 0.4) + 0 + (0.4 \times 0.5)}{\sqrt{0.6^2 + 0.4^2} \sqrt{0.4^2 + 0.4^2 + 0.5^2} } \approx 0.78 $$

$$ \text{sim}(\vec{Q},\vec{D_2}) = \frac{(0.6 \times 0.4)}{\sqrt{0.6^2 + 0.4^2} \sqrt{0.4^2 + 0.5^2} }\approx 0.41 $$

$$ \text{sim}(\vec{Q},\vec{D_3}) = 0 $$

Neste caso, o sistema ranquearia os documentos na ordem: $D_1 > D_2 > D_3$, o que faz sentido intuitivamente, já que $D_1$ contém tanto "gato" quanto um termo relacionado à caça.

A **Similaridade de Cosseno** é particularmente útil em recuperação de informação porque é independente do tamanho dos documentos, considera a direção dos vetores, não sua magnitude, permite comparações rápidas em grandes conjuntos de documentos e funciona bem com vetores esparsos, característicos de representações **TF-IDF**.

**Exemplo 3**: similaridade de cosseno em C++.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iomanip>

// Função para calcular a similaridade de cosseno entre dois vetores
double cosineSimilarity(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    double dotProduct = v1.dot(v2);
    double norm1 = v1.norm();
    double norm2 = v2.norm();
    
    if (norm1 == 0 || norm2 == 0) return 0.0;
    return dotProduct / (norm1 * norm2);
}

// Função para mostrar a matriz de similaridade entre documentos
void printSimilarityMatrix(const std::vector<Eigen::VectorXd>& documents) {
    int n = documents.size();
    Eigen::MatrixXd similarityMatrix = Eigen::MatrixXd::Zero(n, n);
    
    // Calcular similaridade entre todos os pares de documentos
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            similarityMatrix(i, j) = cosineSimilarity(documents[i], documents[j]);
        }
    }
    
    // Imprimir matriz de similaridade
    std::cout << "Matriz de Similaridade de Cosseno:\n";
    std::cout << std::setw(10) << " ";
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(10) << "Doc " + std::to_string(i+1);
    }
    std::cout << "\n";
    
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(10) << "Doc " + std::to_string(i+1);
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                     << similarityMatrix(i, j);
        }
        std::cout << "\n";
    }
}

int main() {
    // Exemplo 1 da seção "Similaridade de Cosseno"
    // Vetores TF-IDF para os documentos:
    // D1: "O gato caça rato"
    // D2: "O gato dorme muito"
    // D3: "O rato come queijo"
    
    // Ordem dos termos: "gato", "rato", "caça", "dorme", "come", "queijo"
    Eigen::VectorXd D1(6);
    D1 << 0.4, 0.4, 0.5, 0.0, 0.0, 0.0;
    
    Eigen::VectorXd D2(6);
    D2 << 0.4, 0.0, 0.0, 0.5, 0.0, 0.0;
    
    Eigen::VectorXd D3(6);
    D3 << 0.0, 0.4, 0.0, 0.0, 0.5, 0.4;
    
    // Consulta "gato caçador"
    Eigen::VectorXd Q(6);
    Q << 0.6, 0.0, 0.4, 0.0, 0.0, 0.0;
    
    std::vector<Eigen::VectorXd> documents = {D1, D2, D3};
    
    // Calcular e imprimir a matriz de similaridade entre documentos
    printSimilarityMatrix(documents);
    
    // Calcular similaridade entre a consulta e cada documento
    std::cout << "\nSimilaridade entre a consulta 'gato caçador' e os documentos:\n";
    for (int i = 0; i < documents.size(); ++i) {
        double sim = cosineSimilarity(Q, documents[i]);
        std::cout << "Documento " << (i+1) << ": " << std::fixed << std::setprecision(4) << sim << "\n";
    }
    
    // Verificação manual das similaridades
    std::cout << "\nVerificação manual das similaridades usando a fórmula:\n";
    
    double sim_D1_D2 = D1.dot(D2) / (D1.norm() * D2.norm());
    std::cout << "sim(D1, D2) = " << std::fixed << std::setprecision(4) << sim_D1_D2 << "\n";
    
    double sim_D1_D3 = D1.dot(D3) / (D1.norm() * D3.norm());
    std::cout << "sim(D1, D3) = " << std::fixed << std::setprecision(4) << sim_D1_D3 << "\n";
    
    double sim_Q_D1 = Q.dot(D1) / (Q.norm() * D1.norm());
    std::cout << "sim(Q, D1) = " << std::fixed << std::setprecision(4) << sim_Q_D1 << "\n";
    
    return 0;
}
```

Utilizamos os vetores **TF-IDF** apresentados no **Exemplo 1** da seção **Similaridade de Cosseno** do texto.
O código implementa:

1. **Função de similaridade de cosseno**: calcula o produto escalar dos vetores normalizado pelos seus comprimentos (normas), usando as operações otimizadas da biblioteca Eigen.

2. **Matriz de similaridade**: calcula e exibe uma matriz que mostra a similaridade entre todos os pares de documentos, facilitando a visualização de quais documentos são mais semelhantes entre si.

3. **Similaridade com consulta**: demonstra como calcular a similaridade entre uma consulta "gato caçador" e cada documento do corpus.

4. **Verificação manual**: implementa o cálculo detalhado da fórmula para confirmação dos resultados.

A implementação utiliza os recursos da biblioteca Eigen para operações vetoriais eficientes, como produto escalar (`dot`) e cálculo de normas (`norm`). Isso facilita a implementação das fórmulas matemáticas apresentadas no texto.

##### Exemplo Detalhado de TF-IDF

Vamos analisar um exemplo completo usando um **corpus** com apenas dois documentos:

- $D_1$: "O gato preto subiu no telhado. O gato dorme no telhado";
- $D_2$: "O telhado é preto".

Primeiro, definimos o vocabulário global ordenado:

$$
V'_{global} = \{ \text{"dorme"}, \text{"é"}, \text{"gato"}, \text{"no"}, \text{"o"}, \text{"preto"}, \text{"subiu"}, \text{"telhado"} \}
$$

Agora, vamos calcular o **TF-IDF** para algumas palavras chave:

**Para "gato" em $D_1$:**

1. Cálculo do TF:

   - $f(\text{"gato"}, D_1) = 2$ (frequência bruta);
   - Total de palavras em $D_1 = 10$;
   - $\text{TF}(\text{"gato"}, D_1) = \frac{2}{10} = 0.2$.

2. Cálculo do IDF:

   - $|Docs| = 2$;
   - $|\{D_j \in Docs: \text{"gato"} \in D_j\}| = 1$;
   - $\text{IDF}(\text{"gato"}, Docs) = \log(\frac{2}{1}) \approx 0.301$.

3. **TF-IDF** final:

   - $\text{TF-IDF}(\text{"gato"}, D_1) = 0.2 \times 0.301 \approx 0.0602$.

**Para "telhado" em $D_2$:**

1. Cálculo do TF:

   - $f(\text{"telhado"}, D_2) = 1$;
   - Total de palavras em $D_2 = 4$;
   - $\text{TF}(\text{"telhado"}, D_2) = \frac{1}{4} = 0.25$.

2. Cálculo do IDF:

   - $|Docs| = 2$;
   - $|\{D_j \in Docs: \text{"telhado"} \in D_j\}| = 2$;
   - $\text{IDF}(\text{"telhado"}, Docs) = \log(\frac{2}{2}) = 0$.

3. **TF-IDF** final:

   - $\text{TF-IDF}(\text{"telhado"}, D_2) = 0.25 \times 0 = 0$.

A sagaz leitora deve notar que "telhado" recebeu pontuação zero em $D_2$ porque aparece em todos os documentos do corpus, ilustrando como o **TF-IDF** penaliza palavras muito comuns.

Para um documento $D_i$, podemos construir um vetor **TF-IDF** $\vec{tfidf}_{D_i} \in \mathbb{R}^{|V_{global}|}$:

$$
\vec{tfidf}_{D_i} = [\text{TF-IDF}(w_1, D_i), \text{TF-IDF}(w_2, D_i), ..., \text{TF-IDF}(w_{|V_{global}|}, D_i)]^T
$$

Esta representação vetorial é útil em tarefas relacionadas com a recuperação de informação, classificação de documentos, agrupamento de textos similares e sistemas de recomendação baseados em conteúdo. Ou, em outras palavras, todas as atividades comuns, e básicas, do processamento de linguagem natural.

O **TF-IDF** resolve parcialmente o problema de palavras muito frequentes do **BoW**, mas ainda mantém algumas limitações, como a perda da ordem das palavras e das relações semânticas.

Para este pobre autor, a perda de conteúdo semântico é a mais preocupante e o principal motivo de continuarmos este texto. Mas, antes, vamos ver um exemplo completo.

**Exemplo 1: **TF-IDF** para Recuperação de Informação

Para ilustrar o uso do **TF-IDF** em recuperação de informação, vamos construir um exemplo passo a passo. Imagine que estamos construindo um sistema de busca simplificado para uma pequena coleção de documentos.

- **Documento 1:** "O gato caçador pula no telhado";
- **Documento 2:** "Cachorro late para o gato no quintal";
- **Documento 3:** "Pássaro voa alto no céu azul".

Começamos criando um vocabulário simplificado, considerando todas as palavras em minúsculas e removendo pontuações. Nosso vocabulário será dado por:

$$
V = \{ \text{"gato", "caçador", "pula", "telhado", "cachorro", "late", "quintal", "pássaro", "voa", "alto", "céu", "azul"}\}
$$

**1. Cálculo da Frequência do Termo (TF):**

Para cada documento e cada termo do vocabulário, calculamos a Frequência do Termo (**TF**).  Vamos usar a frequência bruta do termo no documento.

| Termo      | Documento 1 (TF) | Documento 2 (TF) | Documento 3 (TF) |
| :--------- | :---------------: | :---------------: | :---------------: |
| "gato"       |         1         |         1         |         0         |
| "caçador"    |         1         |         0         |         0         |
| "pula"       |         1         |         0         |         0         |
| "telhado"    |         1         |         0         |         0         |
| "cachorro"   |         0         |         1         |         0         |
| "late"       |         0         |         1         |         0         |
| "quintal"    |         0         |         1         |         0         |
| "pássaro"    |         0         |         0         |         1         |
| "voa"        |         0         |         0         |         1         |
| "alto"       |         0         |         0         |         1         |
| "céu"        |         0         |         0         |         1         |
| "azul"       |         0         |         0         |         1         |

**2. Cálculo da Frequência Inversa nos Documentos (IDF):**

Agora, calculamos o IDF para cada termo do vocabulário usando a fórmula básica:

$$
\text{IDF}(w, Docs) = \log \left( \frac{|Docs|}{|\{D_j \in Docs: w \in D_j\}|} \right)
$$

Onde $|Docs| = 3$ (número total de documentos).

| Termo      | Document Frequency (DF) | IDF (aproximado) |
| :--------- | :---------------------: | :---------------: |
| "gato"       |            2            |      $\log(3/2) \approx 0.18$      |
| "caçador"    |            1            |      $\log(3/1) \approx 0.48$      |
| "pula"       |            1            |      $\log(3/1) \approx 0.48$      |
| "telhado"    |            1            |      $\log(3/1) \approx 0.48$      |
| "cachorro"   |            1            |      $\log(3/1) \approx 0.48$      |
| "late"       |            1            |      $\log(3/1) \approx 0.48$      |
| "quintal"    |            1            |      $\log(3/1) \approx 0.48$      |
| "pássaro"    |            1            |      $\log(3/1) \approx 0.48$      |
| "voa"        |            1            |      $\log(3/1) \approx 0.48$      |
| "alto"       |            1            |      $\log(3/1) \approx 0.48$      |
| "céu"        |            1            |      $\log(3/1) \approx 0.48$      |
| "azul"       |            1            |      $\log(3/1) \approx 0.48$      |

*Observação: Utilizamos o logaritmo natural (base $$e$$) para simplificação. Em aplicações reais, a base do logaritmo não altera o ranqueamento, apenas a escala dos valores.*

**3. Cálculo do TF-IDF:**

Multiplicamos o **TF** de cada termo em cada documento pelo seu IDF correspondente para obter a matriz **TF-IDF**:

| Termo      | Documento 1 (**TF-IDF**) | Documento 2 (**TF-IDF**) | Documento 3 (**TF-IDF**) |
| :--------- | :--------------------: | :--------------------: | :--------------------: |
| "gato"       |        $1 \times 0.18 \approx 0.18$        |        $1 \times 0.18 \approx 0.18$        |         $0$          |
| "caçador"    |        $1 \times 0.48 \approx 0.48$        |         $0$          |         $0$          |
| "pula"       |        $1 \times 0.48 \approx 0.48$        |         $0$          |         $0$          |
| "telhado"    |        $1 \times 0.48 \approx 0.48$        |         $0$          |         $0$          |
| "cachorro"   |         $0$          |        $1 \times 0.48 \approx 0.48$        |         $0$          |
| "late"       |         $0$          |        $1 \times 0.48 \approx 0.48$        |         $0$          |
| "quintal"    |         $0$          |        $1 \times 0.48 \approx 0.48$        |         $0$          |
| "pássaro"    |         $0$          |         $0$          |        $1 \times 0.48 \approx 0.48$        |
| "voa"        |         $0$          |         $0$          |        $1 \times 0.48 \approx 0.48$        |
| "alto"       |         $0$          |         $0$          |        $1 \times 0.48 \approx 0.48$        |
| "céu"        |         $0$          |         $0$          |        $1 \times 0.48 \approx 0.48$        |
| "azul"       |         $0$          |         $0$          |        $1 \times 0.48 \approx 0.48$        |

**4. Consulta do Usuário**: suponha que o usuário faça a seguinte consulta: "gato no telhado".

Primeiro, precisamos criar um vetor da consulta usando o mesmo algoritmo que usamos antes. Ou seja, calculamos o vetor **TF-IDF** para a consulta, usando o mesmo **IDF** calculado para os documentos. Vamos assumir $TF=1$ para cada termo presente na consulta e $0$ para os demais termos do vocabulário.

| Termo      | Consulta (TF-IDF) |
| :--------- | :-----------------: |
| "gato"       |      $1 \times 0.18 \approx 0.18$      |
| "telhado"    |      $1 \times 0.48 \approx 0.48$      |
| "caçador"    |         $0$         |
| "pula"       |         $0$         |
| "cachorro"   |         $0$         |
| "late"       |         $0$         |
| "quintal"    |         $0$         |
| "pássaro"    |         $0$         |
| "voa"        |         $0$         |
| "alto"       |         $0$         |
| "céu"        |         $0$         |
| "azul"       |         $0$         |

**5. Cálculo da Similaridade do Cosseno**: calculamos a similaridade do cosseno entre o vetor da consulta e o vetor **TF-IDF** de cada documento.

1. **Documento 1 vs. Consulta:**

   Vetor Doc 1:  $[0.18, 0.48, 0.48, 0.48, 0, 0, 0, 0, 0, 0, 0, 0]$

   Vetor Consulta: $[0.18, 0, 0, 0.48, 0, 0, 0, 0, 0, 0, 0, 0]$

   Similaridade (Doc 1, Consulta) $\approx \frac{(0.18 \times 0.18) + (0.48 \times 0.48)}{\sqrt{(0.18^2 + 0.48^2 + 0.48^2 + 0.48^2)} \times \sqrt{(0.18^2 + 0.48^2)}} \approx 0.92$

2. **Documento 2 vs. Consulta:**

   Vetor Doc 2:  $[0.18, 0, 0, 0, 0.48, 0.48, 0.48, 0, 0, 0, 0, 0]$

   Vetor Consulta: $[0.18, 0, 0, 0.48, 0, 0, 0, 0, 0, 0, 0, 0]$

   Similaridade (Doc 2, Consulta) $\approx \frac{(0.18 \times 0.18)}{\sqrt{(0.18^2 + 0.48^2 + 0.48^2 + 0.48^2)} \times \sqrt{(0.18^2 + 0.48^2)}} \approx 0.16$

3. **Documento 3 vs. Consulta:**

   Vetor Doc 3:  $[0, 0, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48]$

   Vetor Consulta: $[0.18, 0, 0, 0.48, 0, 0, 0, 0, 0, 0, 0, 0]$

   Similaridade (Doc 3, Consulta) $\approx 0$ (produto escalar é zero)

**6. Ranqueamento**: com base na similaridade do cosseno, os documentos devem ranqueados em ordem decrescente de similaridade. O que resulta em:

1. **Documento 1:** Similaridade $\approx 0.92$;
2. **Documento 2:** Similaridade $\approx 0.16$;
3. **Documento 3:** Similaridade $\approx 0$.

Finalmente, o **Documento 1** deve ser considerado o mais relevante para a consulta "gato no telhado", o que faz sentido intuitivamente, pois ele contém ambos os termos da consulta e tem uma concentração maior desses termos em relação aos outros documentos. O **TF-IDF**, neste exemplo simplificado, consegue capturar a relevância do **Documento 1** para a consulta, demonstrando seu princípio de funcionamento na recuperação de informação.

**Exemplo 2**: trabalhando com C++.

Este exemplo apresenta uma implementação completa do algoritmo **TF-IDF** utilizando a biblioteca Eigen para operações matriciais. A classe **TFIDF** encapsula todo o processamento, desde a construção do vocabulário até o cálculo da matriz **TF-IDF** e a busca de documentos similares.

Principais componentes e técnicas utilizadas:

1. **Representação vetorial**: utilizamos a biblioteca Eigen para criar e manipular matrizes e vetores, o que facilita os cálculos de similaridade e produtos escalares;

2. **Construção do vocabulário**: o método `buildVocabulary()` extrai todos os termos únicos do corpus e cria um mapeamento entre termos e índices para acesso eficiente;

3. **Cálculo de TF-IDF**: implementamos o cálculo de **TF normalizada** e **IDF suavizada**, seguindo as fórmulas apresentadas no texto.

4. **Matriz TF-IDF**: o método `computeTFIDFMatrix()` calcula a matriz TF-IDF completa, onde cada linha representa um documento e cada coluna um termo do vocabulário.
Similaridade de cosseno: Implementamos o cálculo da **similaridade de cosseno** para comparar documentos entre si ou com uma consulta.
Processamento de consultas: O método `processQuery()` converte uma consulta de texto em um vetor **TF-IDF** e calcula sua similaridade com todos os documentos do corpus.

Na função `main()`, demonstramos o uso da classe com o exemplo do corpus de três documentos mencionado no texto. Imprimimos a matriz **TF-IDF** resultante e testamos uma consulta "gato preto", mostrando os documentos ordenados por relevância.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>

class TFIDF {
private:
    std::vector<std::vector<std::string>> corpus;
    std::vector<std::string> vocabulary;
    Eigen::MatrixXd tfidfMatrix;
    
    // Mapeia termos para índices no vocabulário
    std::unordered_map<std::string, int> termToIndex;
    
public:
    TFIDF(const std::vector<std::vector<std::string>>& documents) : corpus(documents) {
        buildVocabulary();
        computeTFIDFMatrix();
    }
    
    // Constrói o vocabulário a partir do corpus
    void buildVocabulary() {
        std::unordered_set<std::string> uniqueTerms;
        
        for (const auto& document : corpus) {
            for (const auto& term : document) {
                uniqueTerms.insert(term);
            }
        }
        
        vocabulary.assign(uniqueTerms.begin(), uniqueTerms.end());
        std::sort(vocabulary.begin(), vocabulary.end()); // Ordenar para consistência
        
        // Criar mapeamento de termos para índices
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            termToIndex[vocabulary[i]] = i;
        }
    }
    
    // Calcula o Term Frequency (TF) normalizado para um termo em um documento
    double calculateTF(const std::vector<std::string>& document, const std::string& term) {
        double count = 0;
        for (const auto& word : document) {
            if (word == term) {
                count++;
            }
        }
        
        return count / document.size(); // Frequência normalizada
    }
    
    // Calcula o Inverse Document Frequency (IDF) para um termo
    double calculateIDF(const std::string& term) {
        double N = corpus.size();
        double documentFrequency = 0;
        
        for (const auto& document : corpus) {
            for (const auto& word : document) {
                if (word == term) {
                    documentFrequency++;
                    break;
                }
            }
        }
        
        // Aplicar suavização para evitar divisão por zero
        return std::log(N / (1 + documentFrequency));
    }
    
    // Computa a matriz TF-IDF completa
    void computeTFIDFMatrix() {
        int numDocs = corpus.size();
        int numTerms = vocabulary.size();
        
        // Inicializar matriz TF-IDF com zeros
        tfidfMatrix = Eigen::MatrixXd::Zero(numDocs, numTerms);
        
        // Calcular valores TF-IDF para cada documento e termo
        for (int i = 0; i < numDocs; ++i) {
            const auto& document = corpus[i];
            
            for (int j = 0; j < numTerms; ++j) {
                const auto& term = vocabulary[j];
                double tf = calculateTF(document, term);
                double idf = calculateIDF(term);
                
                tfidfMatrix(i, j) = tf * idf;
            }
        }
    }
    
    // Calcula a similaridade de cosseno entre dois vetores
    double cosineSimilarity(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
        double dotProduct = v1.dot(v2);
        double norm1 = v1.norm();
        double norm2 = v2.norm();
        
        if (norm1 == 0 || norm2 == 0) return 0.0;
        return dotProduct / (norm1 * norm2);
    }
    
    // Processa uma consulta e retorna os documentos mais similares
    std::vector<std::pair<int, double>> processQuery(const std::vector<std::string>& query) {
        // Criar vetor TF-IDF para a consulta
        Eigen::VectorXd queryVector = Eigen::VectorXd::Zero(vocabulary.size());
        
        for (const auto& term : query) {
            if (termToIndex.find(term) != termToIndex.end()) {
                int index = termToIndex[term];
                double tf = static_cast<double>(std::count(query.begin(), query.end(), term)) / query.size();
                double idf = calculateIDF(term);
                queryVector(index) = tf * idf;
            }
        }
        
        // Calcular similaridade com cada documento
        std::vector<std::pair<int, double>> similarities;
        for (int i = 0; i < corpus.size(); ++i) {
            Eigen::VectorXd docVector = tfidfMatrix.row(i);
            double similarity = cosineSimilarity(queryVector, docVector);
            similarities.push_back({i, similarity});
        }
        
        // Ordenar por similaridade (ordem decrescente)
        std::sort(similarities.begin(), similarities.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return similarities;
    }
    
    // Exibe a matriz TF-IDF
    void printTFIDFMatrix() {
        std::cout << "Matriz TF-IDF (" << corpus.size() << " documentos x " 
                 << vocabulary.size() << " termos):\n\n";
        
        // Imprimir cabeçalho do vocabulário
        std::cout << std::setw(10) << " ";
        for (const auto& term : vocabulary) {
            std::cout << std::setw(10) << term;
        }
        std::cout << "\n";
        
        // Imprimir valores da matriz
        for (int i = 0; i < corpus.size(); ++i) {
            std::cout << "Doc " << i << ":" << std::setw(4) << " ";
            for (int j = 0; j < vocabulary.size(); ++j) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << tfidfMatrix(i, j);
            }
            std::cout << "\n";
        }
    }
    
    // Getter para o vocabulário
    const std::vector<std::string>& getVocabulary() const {
        return vocabulary;
    }
};

int main() {
    // Exemplo do corpus com três documentos usado no texto
    std::vector<std::vector<std::string>> corpus = {
        {"o", "gato", "preto", "caça", "o", "rato", "preto"},
        {"o", "rato", "branco", "corre", "do", "gato"},
        {"o", "cachorro", "late", "para", "o", "gato", "preto"}
    };
    
    TFIDF tfidf(corpus);
    
    // Imprimir vocabulário
    std::cout << "Vocabulário: ";
    for (const auto& term : tfidf.getVocabulary()) {
        std::cout << term << " ";
    }
    std::cout << "\n\n";
    
    // Imprimir matriz TF-IDF
    tfidf.printTFIDFMatrix();
    
    // Testar uma consulta
    std::vector<std::string> query = {"gato", "preto"};
    std::cout << "\nConsulta: ";
    for (const auto& term : query) {
        std::cout << term << " ";
    }
    std::cout << "\n\n";
    
    // Processar consulta e mostrar resultados
    auto results = tfidf.processQuery(query);
    std::cout << "Resultados da busca (ordenados por relevância):\n";
    for (const auto& [docId, similarity] : results) {
        std::cout << "Documento " << docId << ": Similaridade = " 
                 << std::fixed << std::setprecision(4) << similarity << "\n";
        
        // Mostrar o conteúdo do documento
        std::cout << "  Conteúdo: ";
        for (const auto& term : corpus[docId]) {
            std::cout << term << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```

#### **TF-IDF** em C++ Na Prática

O código a seguir implementa um sistema de recuperação de informação utilizando o algoritmo **TF-IDF**. Ele inclui pré-processamento de texto, construção de vocabulário, cálculo da matriz **TF-IDF**, e similaridade de cosseno entre documentos e consultas.

A esforçada leitora deve notar que o código é um exemplo simplificado e não inclui todas as funcionalidades de um sistema de recuperação de informação completo. Usando técnicas que ainda não discutimos. No entanto, ele fornece uma base sólida para entender como o **TF-IDF** pode ser implementado em C++.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <Eigen/Dense>

// Classe para pré-processamento de texto
class TextPreprocessor {
public:
    // Converte texto para minúsculas e remove pontuação
    static std::string normalize(const std::string& text) {
        std::string result;
        for (char c : text) {
            if (std::isalpha(c)) {
                result += std::tolower(c);
            } else if (c == ' ' || c == '\t' || c == '\n') {
                result += ' ';
            }
        }
        return result;
    }
    
    // Divide texto em tokens (palavras)
    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        return tokens;
    }
    
    // Lista de stopwords em português, muito simplificada
    static const std::set<std::string>& getStopwords() {
        static const std::set<std::string> stopwords = {
            "o", "a", "os", "as", "um", "uma", "uns", "umas",
            "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
            "em", "para", "por", "com", "sem", "sob", "e", "ou", "mas"
        };
        return stopwords;
    }
    
    // Remove stopwords de um documento tokenizado
    static std::vector<std::string> removeStopwords(const std::vector<std::string>& tokens) {
        std::vector<std::string> filtered;
        const auto& stopwords = getStopwords();
        
        for (const auto& token : tokens) {
            if (stopwords.find(token) == stopwords.end()) {
                filtered.push_back(token);
            }
        }
        
        return filtered;
    }
    
    // Processa um texto completo: normaliza, tokeniza e remove stopwords
    static std::vector<std::string> process(const std::string& text) {
        std::string normalized = normalize(text);
        std::vector<std::string> tokens = tokenize(normalized);
        return removeStopwords(tokens);
    }
};

// Sistema de Recuperação de Informação baseado em TF-IDF
class InformationRetrievalSystem {
private:
    std::vector<std::string> rawDocuments;               // Documentos originais
    std::vector<std::vector<std::string>> processedDocs; // Documentos processados
    std::vector<std::string> vocabulary;                 // Vocabulário global
    std::unordered_map<std::string, int> termToIndex;    // Mapa de termos para índices
    Eigen::MatrixXd tfidfMatrix;                         // Matriz TF-IDF
    
public:
    // Adiciona um novo documento ao sistema
    void addDocument(const std::string& document) {
        rawDocuments.push_back(document);
        processedDocs.push_back(TextPreprocessor::process(document));
    }
    
    // Constrói o vocabulário a partir dos documentos processados
    void buildVocabulary() {
        std::set<std::string> uniqueTerms;
        
        for (const auto& doc : processedDocs) {
            for (const auto& term : doc) {
                uniqueTerms.insert(term);
            }
        }
        
        vocabulary.assign(uniqueTerms.begin(), uniqueTerms.end());
        std::sort(vocabulary.begin(), vocabulary.end());
        
        // Criar mapeamento de termos para índices
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            termToIndex[vocabulary[i]] = i;
        }
    }
    
    // Calcula a frequência do termo (TF) com suavização logarítmica
    double calculateTF(const std::vector<std::string>& document, const std::string& term) {
        int count = 0;
        for (const auto& word : document) {
            if (word == term) {
                count++;
            }
        }
        
        return count > 0 ? 1.0 + std::log(count) : 0.0;
    }
    
    // Calcula a frequência inversa do documento (IDF) com suavização
    double calculateIDF(const std::string& term) {
        int N = processedDocs.size();
        int df = 0;
        
        for (const auto& doc : processedDocs) {
            if (std::find(doc.begin(), doc.end(), term) != doc.end()) {
                df++;
            }
        }
        
        return std::log(N / (1.0 + df));
    }
    
    // Calcula a matriz TF-IDF para todos os documentos
    void computeTFIDFMatrix() {
        int numDocs = processedDocs.size();
        int numTerms = vocabulary.size();
        
        tfidfMatrix = Eigen::MatrixXd::Zero(numDocs, numTerms);
        
        for (int i = 0; i < numDocs; ++i) {
            const auto& doc = processedDocs[i];
            
            for (int j = 0; j < numTerms; ++j) {
                const auto& term = vocabulary[j];
                double tf = calculateTF(doc, term);
                double idf = calculateIDF(term);
                tfidfMatrix(i, j) = tf * idf;
            }
        }
    }
    
    // Calcula o vetor TF-IDF para uma consulta
    Eigen::VectorXd calculateQueryVector(const std::string& query) {
        std::vector<std::string> processedQuery = TextPreprocessor::process(query);
        Eigen::VectorXd queryVector = Eigen::VectorXd::Zero(vocabulary.size());
        
        for (const auto& term : processedQuery) {
            if (termToIndex.find(term) != termToIndex.end()) {
                int index = termToIndex[term];
                double tf = calculateTF(processedQuery, term);
                double idf = calculateIDF(term);
                queryVector(index) = tf * idf;
            }
        }
        
        return queryVector;
    }
    
    // Calcula a similaridade de cosseno entre dois vetores
    double cosineSimilarity(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
        double dotProduct = v1.dot(v2);
        double norm1 = v1.norm();
        double norm2 = v2.norm();
        
        if (norm1 < 1e-10 || norm2 < 1e-10) return 0.0;
        return dotProduct / (norm1 * norm2);
    }
    
    // Busca por documentos relevantes para uma consulta
    std::vector<std::pair<int, double>> search(const std::string& query, int topK = 3) {
        // Verificar se há documentos no sistema
        if (processedDocs.empty()) {
            return {};
        }
        
        // Calcular vetor TF-IDF para a consulta
        Eigen::VectorXd queryVector = calculateQueryVector(query);
        
        // Calcular similaridades com todos os documentos
        std::vector<std::pair<int, double>> results;
        for (int i = 0; i < processedDocs.size(); ++i) {
            Eigen::VectorXd docVector = tfidfMatrix.row(i);
            double similarity = cosineSimilarity(queryVector, docVector);
            results.push_back({i, similarity});
        }
        
        // Ordenar resultados por similaridade (decrescente)
        std::sort(results.begin(), results.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Retornar apenas os top-K resultados mais relevantes
        if (results.size() > topK) {
            results.resize(topK);
        }
        
        return results;
    }
    
    // Inicializa o sistema com um conjunto de documentos
    void initialize(const std::vector<std::string>& documents) {
        // Limpar estado atual
        rawDocuments.clear();
        processedDocs.clear();
        vocabulary.clear();
        termToIndex.clear();
        
        // Adicionar e processar documentos
        for (const auto& doc : documents) {
            addDocument(doc);
        }
        
        // Construir vocabulário e matriz TF-IDF
        buildVocabulary();
        computeTFIDFMatrix();
    }
    
    // Mostrar informações sobre o sistema
    void printInfo() {
        std::cout << "Sistema de Recuperação de Informação\n";
        std::cout << "------------------------------------\n";
        std::cout << "Número de documentos: " << rawDocuments.size() << "\n";
        std::cout << "Tamanho do vocabulário: " << vocabulary.size() << " termos\n";
        
        std::cout << "\nAmostra do vocabulário (primeiros 20 termos):\n";
        int count = 0;
        for (const auto& term : vocabulary) {
            std::cout << term << " ";
            if (++count >= 20) break;
        }
        std::cout << "\n\n";
    }
    
    // Mostrar documentos originais
    void printDocuments() {
        std::cout << "Documentos no sistema:\n";
        std::cout << "---------------------\n";
        for (int i = 0; i < rawDocuments.size(); ++i) {
            std::cout << "Documento " << i << ": " << rawDocuments[i] << "\n";
        }
        std::cout << "\n";
    }
    
    // Mostrar resultados da busca de forma amigável
    void displaySearchResults(const std::string& query, 
                             const std::vector<std::pair<int, double>>& results) {
        std::cout << "Resultados da busca para: \"" << query << "\"\n";
        std::cout << "----------------------------------------\n";
        
        if (results.empty()) {
            std::cout << "Nenhum resultado encontrado.\n\n";
            return;
        }
        
        for (int i = 0; i < results.size(); ++i) {
            int docId = results[i].first;
            double score = results[i].second;
            
            std::cout << (i+1) << ". [Score: " << std::fixed << std::setprecision(4) << score << "] ";
            std::cout << "Documento " << docId << ": " << rawDocuments[docId] << "\n";
        }
        std::cout << "\n";
    }
};

// Função para demonstrar o sistema com o exemplo do texto
void demonstrateWithExampleCorpus() {
    std::vector<std::string> corpus = {
        "O gato preto caça o rato preto",
        "O rato branco corre do gato",
        "O cachorro late para o gato preto"
    };
    
    InformationRetrievalSystem irSystem;
    irSystem.initialize(corpus);
    
    irSystem.printInfo();
    irSystem.printDocuments();
    
    // Demonstrar busca com consultas variadas
    std::vector<std::string> queries = {
        "gato preto",
        "rato",
        "cachorro late",
        "animal"
    };
    
    for (const auto& query : queries) {
        auto results = irSystem.search(query);
        irSystem.displaySearchResults(query, results);
    }
}

int main() {
    std::cout << "Demonstração do Sistema de Recuperação de Informação baseado em TF-IDF\n\n";
    demonstrateWithExampleCorpus();
    
    return 0;
}
```

Este exemplo apresenta uma implementação completa e prática de um sistema de recuperação de informação baseado no algoritmo **TF-IDF**, utilizando C++20 e a biblioteca Eigen. O sistema implementa todas as etapas necessárias para a construção de um mecanismo de busca textual:

1. **Pré-processamento de texto**: a classe `TextPreprocessor` implementa técnicas básicas de processamento de linguagem natural, que ainda não foram discutidas no texto. À saber:

   - Normalização: remoção de pontuação, conversão para minúsculas;
   - Tokenização: divisão do texto em palavras;
   - Remoção de stopwords: palavras muito comuns como artigos e preposições.

2. **Indexação**: o sistema constrói um vocabulário global a partir dos documentos e cria uma matriz **TF-IDF** utilizando a biblioteca Eigen.

3. **Implementação do TF** com suavização logarítmica
Implementação do **IDF com suavização**.

4. **Representação vetorial** eficiente usando matrizes esparsas.

5. **Busca**: implementa o processamento de consultas e cálculo de similaridade:

    - Conversão da consulta em um vetor **TF-IDF**;
    - Cálculo da similaridade de cosseno entre a consulta e todos os documentos;
    - Ordenação dos resultados por relevância.

6. **Interface de usuário**: fornece métodos para visualizar informações do sistema e resultados de busca.

Na função `demonstrateWithExampleCorpus()`, utilizamos o corpus de três documentos mencionado no texto como exemplo, e demonstramos buscas com diferentes consultas para mostrar como o sistema ranqueia os documentos por relevância.

Esta implementação é uma aplicação prática dos conceitos teóricos apresentados no texto, mostrando como o **TF-IDF** pode ser usado para construir um sistema de recuperação de informação funcional.

### One-Hot Encoding

Chegamos ao **One-Hot Encoding**. Embora menos comum para representar diretamente *textos inteiros* em tarefas de processamento de linguagem natural mais sofisticados, o **One-Hot Encoding** é frequentemente usado como um passo inicial para representar *palavras individuais* ou *caracteres* numericamente em processamentos complexos.

Em **One-Hot Encoding**, cada palavra, ou caractere, do vocabulário $V$, será representada por um vetor binário $\vec{e}_w$. Neste caso, o tamanho do vetor será igual ao tamanho do vocabulário, $\vert V \vert$. Para cada palavra $w_i \in V$, o vetor $\vec{e}_{w_i}$ terá todos os valores como $0$, exceto na posição $i$ correspondente à palavra $w_i$ no vocabulário, onde o valor será $1$.

Formalmente, teremos: se $V = \{w_1, w_2, ..., w_{\vert V \vert }\}$ é o vocabulário ordenado, então o **One-Hot Encoding** para a palavra $w_i$ é um vetor $\vec{e}_{w_i} \in \mathbb{R}^{\vert V \vert }$ tal que:

$$
(\vec{e}_{w_i})_j =
\begin{cases}
    1 & \text{se } j = i \\
    0 & \text{se } j \neq i
\end{cases}
$$

Para o vocabulário $V'_{global} = \{ \text{"dorme"}, \text{"é"}, \text{"gato"}, \text{"no"}, \text{"o"}, \text{"preto"}, \text{"subiu"}, \text{"telhado"} \}$., o **One-Hot Encoding** seria:

- $\vec{e}_{\text{"dorme"}} = [1, 0, 0, 0, 0, 0, 0, 0]^T$
- $\vec{e}_{\text{"é"}} = [0, 1, 0, 0, 0, 0, 0, 0]^T$
- $\vec{e}_{\text{"gato"}} = [0, 0, 1, 0, 0, 0, 0, 0]^T$
- $\vec{e}_{\text{"no"}} = [0, 0, 0, 1, 0, 0, 0, 0]^T$
- $\vec{e}_{\text{"o"}} = [0, 0, 0, 0, 1, 0, 0, 0]^T$
- $\vec{e}_{\text{"preto"}} = [0, 0, 0, 0, 0, 1, 0, 0]^T$
- $\vec{e}_{\text{"subiu"}} = [0, 0, 0, 0, 0, 0, 1, 0]^T$
- $\vec{e}_{\text{"telhado"}} = [0, 0, 0, 0, 0, 0, 0, 1]^T$

One-hot encoding é extremamente simples e garante que cada palavra seja representada de forma única e independente. É frequentemente usado como entrada para modelos de aprendizado profundo, especialmente em camadas iniciais de redes neurais.

Os produtos escalares são especialmente úteis quando estamos trabalhando com nossas representações one-hot de palavras. O produto escalar de qualquer vetor one-hot com ele mesmo é um.

Produto escalar de vetores correspondentes (Exemplo com vetores one-hot):

Considere o vetor $a = [0, 1, 0]$. O produto escalar consigo mesmo é:

$$
a \cdot a = (0 * 0) + (1 * 1) + (0 * 0) = 0 + 1 + 0 = 1
$$

E o produto escalar de qualquer vetor one-hot com qualquer outro vetor one-hot é zero.

Produto escalar de vetores não correspondentes (Exemplo com vetores one-hot):

Considere os vetores $a = [0, 1, 0]$ e $b = [1, 0, 0]$. O produto escalar é:

$$
a \cdot b = (0 * 1) + (1 * 0) + (0 * 0) = 0 + 0 + 0 = 0
$$

Os dois exemplos anteriores mostram como os produtos escalares podem ser usados para medir a similaridade. Como outro exemplo, considere um vetor de valores que representa uma combinação de palavras com diferentes pesos. Uma palavra codificada one-hot pode ser comparada com ele usando o produto escalar para mostrar o quão fortemente essa palavra é representada.

#### Exemplo de One-Hot Encoding em C++ 20

O código a seguir implementa o **One-Hot Encoding** em C++20. Ele utiliza a biblioteca Eigen para operações matriciais e demonstra como criar representações one-hot para um conjunto de palavras.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm> // Para std::sort
#include <Eigen/Dense> // Para Eigen::VectorXd
#include <iomanip>     // Para std::fixed, std::setprecision

// Classe para realizar One-Hot Encoding de palavras
class OneHotEncoder {
private:
    std::vector<std::string> vocabulary;            // Vocabulário ordenado
    std::unordered_map<std::string, int> termToIndex; // Mapa termo -> índice
    size_t vocabularySize;                          // Tamanho do vocabulário

public:
    // Construtor padrão
    OneHotEncoder() : vocabularySize(0) {}

    // Constrói o vocabulário a partir de um corpus (lista de documentos tokenizados)
    void fit(const std::vector<std::vector<std::string>>& corpus) {
        std::set<std::string> uniqueTerms;
        for (const auto& doc : corpus) {
            for (const auto& term : doc) {
                uniqueTerms.insert(term);
            }
        }

        vocabulary.assign(uniqueTerms.begin(), uniqueTerms.end());
        std::sort(vocabulary.begin(), vocabulary.end()); // Garante ordem consistente
        vocabularySize = vocabulary.size();

        termToIndex.clear();
        for (size_t i = 0; i < vocabularySize; ++i) {
            termToIndex[vocabulary[i]] = i;
        }
        std::cout << "Vocabulário construído com " << vocabularySize << " termos únicos.\n";
    }

    // Transforma um termo em seu vetor One-Hot correspondente
    // Retorna um vetor de zeros se o termo for desconhecido (OOV - Out Of Vocabulary)
    Eigen::VectorXd transform(const std::string& term) const {
        // Cria um vetor de zeros do tamanho do vocabulário
        Eigen::VectorXd oneHotVector = Eigen::VectorXd::Zero(vocabularySize);

        // Procura o índice do termo no mapa
        auto it = termToIndex.find(term);
        if (it != termToIndex.end()) {
            // Se encontrado, define a posição correspondente como 1.0
            oneHotVector(it->second) = 1.0;
        } else {
            // Opcional: Imprimir aviso para termos OOV
            // std::cerr << "Aviso: Termo '" << term << "' não encontrado no vocabulário (OOV).\n";
        }
        return oneHotVector;
    }

    // Retorna o vocabulário construído
    const std::vector<std::string>& getVocabulary() const {
        return vocabulary;
    }

    // Retorna o tamanho do vocabulário
    size_t getVocabularySize() const {
        return vocabularySize;
    }

    // Imprime o vocabulário e seus índices
    void printVocabulary() const {
         std::cout << "\nVocabulário (Termo -> Índice):\n";
         std::cout << "-----------------------------\n";
         for(size_t i = 0; i < vocabulary.size(); ++i) {
              std::cout << "\"" << vocabulary[i] << "\" -> " << i << "\n";
         }
         std::cout << std::endl;
    }
};

// Função auxiliar para imprimir um vetor Eigen de forma mais legível
void printEigenVector(const Eigen::VectorXd& vec, const std::string& name) {
    std::cout << name << " (dim=" << vec.size() << "): [";
    // Imprime apenas alguns elementos para vetores longos, se necessário
    int max_print = 20;
    for (int i = 0; i < vec.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (vec.size() > max_print && i >= max_print / 2 && i < vec.size() - max_print / 2) {
             if (i == max_print / 2) std::cout << "...";
             continue;
        }
        std::cout << vec(i);
    }
    std::cout << "]\n";
}


int main() {
    // Corpus de exemplo do texto (TF-IDF Exemplo 1)
    std::vector<std::vector<std::string>> corpus = {
        {"o", "gato", "preto", "caça", "o", "rato", "preto"},
        {"o", "rato", "branco", "corre", "do", "gato"},
        {"o", "cachorro", "late", "para", "o", "gato", "preto"}
    };

    // 1. Criar e treinar o encoder
    OneHotEncoder encoder;
    encoder.fit(corpus);
    encoder.printVocabulary();

    // 2. Obter vetores One-Hot para alguns termos
    std::cout << "Vetores One-Hot:\n";
    std::cout << "----------------\n";
    Eigen::VectorXd vecGato = encoder.transform("gato");
    Eigen::VectorXd vecPreto = encoder.transform("preto");
    Eigen::VectorXd vecCachorro = encoder.transform("cachorro");
    Eigen::VectorXd vecOOV = encoder.transform("animal"); // Termo fora do vocabulário

    printEigenVector(vecGato, "OneHot('gato')");
    printEigenVector(vecPreto, "OneHot('preto')");
    printEigenVector(vecCachorro, "OneHot('cachorro')");
    printEigenVector(vecOOV, "OneHot('animal') [OOV]"); // Deve ser um vetor de zeros

    // 3. Verificar propriedades do produto escalar mencionadas no texto
    std::cout << "\nPropriedades do Produto Escalar:\n";
    std::cout << "-------------------------------\n";

    // Produto escalar de um vetor com ele mesmo (deve ser 1)
    double dotSelfGato = vecGato.dot(vecGato);
    std::cout << "Produto escalar (gato · gato): " << std::fixed << std::setprecision(1) << dotSelfGato << " (Esperado: 1.0)\n";

    // Produto escalar de vetores diferentes (deve ser 0)
    double dotGatoPreto = vecGato.dot(vecPreto);
    std::cout << "Produto escalar (gato · preto): " << std::fixed << std::setprecision(1) << dotGatoPreto << " (Esperado: 0.0)\n";

    double dotGatoCachorro = vecGato.dot(vecCachorro);
    std::cout << "Produto escalar (gato · cachorro): " << std::fixed << std::setprecision(1) << dotGatoCachorro << " (Esperado: 0.0)\n";

    // Produto escalar com vetor OOV (deve ser 0)
    double dotGatoOOV = vecGato.dot(vecOOV);
    std::cout << "Produto escalar (gato · OOV): " << std::fixed << std::setprecision(1) << dotGatoOOV << " (Esperado: 0.0)\n";

    // 4. Exemplo: Usar produto escalar para "extrair" presença (simples)
    //    Criando um vetor simples que representa "presença de gato e cachorro"
    Eigen::VectorXd presenceVector = vecGato + vecCachorro;
    printEigenVector(presenceVector, "Vetor Presença (gato + cachorro)");

    double weightGato = vecGato.dot(presenceVector);
    double weightPreto = vecPreto.dot(presenceVector);
    double weightCachorro = vecCachorro.dot(presenceVector);

    std::cout << "\nPeso extraído via Produto Escalar:\n";
    std::cout << "  Peso de 'gato' no vetor presença: " << weightGato << "\n";
    std::cout << "  Peso de 'preto' no vetor presença: " << weightPreto << "\n";
    std::cout << "  Peso de 'cachorro' no vetor presença: " << weightCachorro << "\n";

    return 0;
}
```

Pensando no nosso progresso acho que, no próximo artigo, teremos que voltar a matemática.
