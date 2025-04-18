---
layout: post
title: Transformers - Prestando Atenção
author: frank
categories: |-
    disciplina
    Matemática
    artigo
tags: |-
    Matemática
    inteligência artificial
    processamento de linguagem natural
    modelos de sequência
    Cadeias de Markov
    matrizes de transição
    aprendizado de máquina
    atenção
image: assets/images/atencao1.webp
featured: false
rating: 5
description: ""
date: 2025-02-10T22:55:34.524Z
preview: Neste artigo, mergulhamos na modelagem de sequências textuais. Partimos das Cadeias de Markov, **N-gram**s, e suas limitações, construindo gradualmente a intuição para modelos mais sofisticados capazes de capturar dependências de longo alcance, fundamentais para a arquitetura Transformer.
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
published: true
lastmod: 2025-04-18T19:48:28.758Z
---

## Superando Limitações Locais: Construindo a Ponte para a Atenção

No [artigo anterior]([link-para-o-artigo-anterior](https://frankalcantara.com/transformers-desvendando-modelagem-de-sequencias/)), navegamos pelos modelos probabilísticos clássicos para sequências, como as **Cadeias de Markov** e os **Modelos **N-gram****. Vimos como estes modelos são capazes de  capturar a dependência local estimando a probabilidade de uma palavra $w_t$ com base em suas $N-1$ vizinhas imediatas, $P(w_t  \vert  w_{t-N+1}, ..., w_{t-1})$. Essas são técnicas importantes capazes de fornecer representações ricas, como a **Vetorização por Razão de Verossimilhança**, que compara padrões locais de um documento com os padrões gerais do corpus. Mas, nem todos os mares são calmos.

>A **Propriedade de Markov** é um conceito fundamental em processos estocásticos e modelagem de sequências. Esta propriedade estabelece que a probabilidade de um estado futuro depende apenas do estado presente, e não de estados anteriores da sequência.
>
>Em termos mais formais, para uma sequência de variáveis aleatórias (como palavras em um texto) $X_1, X_2, ..., X_n$, a Propriedade de Markov afirma que:
>
>$$P(X_{n+1} = x | X_1 = x_1, X_2 = x_2, ..., X_n = x_n) = P(X_{n+1} = x | X_n = x_n)$$

A própria natureza desses modelos, encapsulada na **Propriedade de Markov**, impõe *uma limitação significativa: a dificuldade em capturar dependências de longo alcance*. Esta limitação é precisamente o que os modelos baseados em mecanismos de atenção, como os **Transformers**, procuram superar, *permitindo que cada palavra na sequência preste atenção a qualquer outra palavra*, independentemente da distância entre elas.

Para superar a **Propriedade de Markov** e entender o significado completo, ou prever a próxima palavra, em situações da linguagem real, precisamos relacionar palavras que estão muito distantes entre si em uma determinada sequência de texto.

A atenta leitora deve considerar que, a solução inocente, aumentar a ordem $N$ nos modelos **N-gram** para tentar alcançar contextos mais longos, será impraticável devido à esparsidade dos dados que resultará desta solução. Afinal, não deve ser difícil de entender que as combinações de $N$ palavras ficam cada vez mais raras e que o número de estados possíveis, irá crescer quase exponencialmente, a maldição da dimensionalidade. O que nos deixa com uma pulga atrás de orelha:

**Como podemos capturar dependências de longo alcance sem aumentar a ordem dos N-grams?**

Neste artigo, percorreremos uma rota entre a visão local dos **N-grams** e os mecanismos mais sofisticados que permitem aos **Transformers** lidar com essas dependências. Nosso primeiro porto será conceitual. Iremos navegar pela ideia de **Agregação de Características de Pares**. Esta técnica irá permitir que possamos considerar a influência de todos os pares formados pela palavra atual, $w_t$, e cada palavra $w_i$ que a precede ($i < t$), efetivamente "saltando" sobre o contexto intermediário.

Minha esperança é que a compassiva leitora seja capaz de entender como essa agregação funciona, suas vantagens e limitações. Serão justamente estas limitações do modelo de **Agregação de Características de Pares** que impulsionarão nossa jornada até a introdução dos conceitos de **atenção seletiva** e **mascaramento**. Este será o porto mais importante desta jornada. Quando chegarmos lá será possível ver como o conceito de foco, manter a atenção no que interessa, pode ser implementada através de operações matriciais e como a informação contextual resultante pode ser processada por **Redes Feed-Forward (FFN)**.

Com cuidado e bom tempo, ao final desta jornada, teremos desvendado a intuição teórica e os componentes tecnológico que sustentam a revolução trazida pelos **Transformers*.

### Agregação de Características de Pares

Ao que parece, aumentar a ordem $N$ nos modelos **N-gram/Markovianos** é uma estratégia limitada para capturar o contexto necessário em linguagem natural devido à maldição da dimensionalidade e à esparsidade dos dados. Ou seja, precisamos de uma abordagem diferente para lidar com dependências que podem se estender por muitas palavras.

A atenta leitora não deve esquecer que queremos transformar linguagem natural, no formato de texto, em algo que o computador possa manipular. Nossa escolha ainda está no processo de vetorização de textos. Deste ponto em diante, a criativa leitora deve considerar que nosso sistema precisa lidar com frases complexas. Entretanto, em benefício do entendimento, vamos considerar as frases a seguir e que cada uma ocorre com igual probabilidade, $50\%$:

* $D_1$ = `Verifique o log do programa e descubra se ele foi executado, por favor.`;
* $D_2$ = `Verifique o log da bateria e descubra se ela acabou, por favor.`.

Neste cenário, para determinar a sequência correta de palavras após `descubra se`, precisamos resolver a referência pronominal. Se o `log` mencionado anteriormente era do `programa` (substantivo masculino), o pronome correto é `ele` e a ação subsequente é `foi executado`. Se o log era da `bateria` (substantivo feminino), o pronome correto é `ela` e a ação é `acabou`. A palavra importante, `programa` ou `bateria`, que determinará o pronome e o verbo seguintes está significativamente distante na sequência de palavras. Um modelo de Markov tradicional exigiria uma ordem $N$ inviável, $N > 8$, para capturar essa dependência com [os modelos que vimos antes](https://frankalcantara.com/transformers-desvendando-modelagem-de-sequencias/).

Para superar a visão estritamente local dos modelos **N-gram** (discutidos em detalhe no [artigo anterior](https://frankalcantara.com/transformers-desvendando-modelagem-de-sequencias/)), foram, ao longo do tempo, propostas algumas alternativas interessantes. Vamos nos concentrar em uma abordagem que mantém a inspiração de analisar interações entre pares de palavras, mas que introduz um certo grau de flexibilidade. Quase como se o modelo estivesse fazendo pilates e esticando-se para alcançar palavras mais distantes.

A ideia central será: ao tentar prever a palavra que segue a palavra atual, $w_t$, em vez de depender apenas do contexto imediatamente anterior, como no par $(w_{t-1}, w_t)$ para bigramas ou a janela fixa dos **N-grams**, *vamos considerar a influência potencial de todas as palavras $w_i$ que apareçam antes de $w_t$ na sequência*. Em outras palavras, o que estamos propondo é um método que permita analisar a contribuição de cada par $(w_i, w_t)\;$, onde o índice $i$ varia desde o início da sequência até a posição anterior a $t$ para todo $i : 0 \le i < t$.

Olhar o problema por essa perspectiva irá permitir saltar sobre o texto intermediário que pode existir entre $w_i$ e $w_t$. *Ao fazer isso, abrimos a possibilidade de capturar dependências e relações semânticas de longo alcance, que são inacessíveis aos modelos **N-gram** tradicionais devido à sua janela de contexto fixa e local*.

Essa mudança conceitual irá implicar em uma reinterpretação da matriz de transição. Neste caso, as linhas da matriz não representarão mais um estado probabilístico: o contexto imediato $w_{t-1}$ ou $[w_{t-2}, w_{t-1}]$ onde as probabilidades de transição devem somar $1$. Em vez disso, cada linha poderá ser vista como uma **característica** (*feature*) que será definida por um par específico $(w_i, w_t)$ que ocorreu na sequência. O valor na coluna $j$ dessa linha passa a representar um **voto** ou o **peso** que essa característica específica atribui à palavra $w_j$ como sendo a próxima palavra $(w_{t+1})$.

A amável leitora vai ouvir falar muito em *feature*, tanto na academia como no populacho. O conceito de *feature* como sendo uma característica do texto anterior a uma determinada palavra, que irá permitir definir qual palavra a seguirá é suficientemente importante no processamento de linguagem natural que, praticamente, se tornou uma medida de desempenho. Pense sobre as *features* sobre isso como se Cada par $(w_i, w_t)$ seja uma característica que possui um determinado grau de contribuição na previsão da próxima palavra. O valor associado a essa característica, que chamaremos de voto ou peso, indica o quanto o par  $(w_i, w_t)$ contribui para a definição de cada palavra candidata a ser a próxima.

![Votação de características de pares com saltos](/assets/images/Votosabstrata.webp)
_Figura 1: Modelo conceitual hipotético baseado em pares com saltos. As linhas representam características (pares como `(programa, executado)` ou `(bateria, executado)`). Os valores apresentados são "votos" para a próxima palavra (ex: "por"). Apenas pesos não-zero relevantes são mostrados. A ilustração foca na predição da palavra após `executado` no contexto da primeira frase._{: class="legend"}

A Figura 1 ilustra a ideia de *feature* como fator para a previsão da palavra após `executado`, no primeiro exemplo de frase: `... log do programa ... ele foi executado ...`. Várias características, pares formados por `executado` e palavras anteriores como `verifique`, `o`, `log`, `do`, `programa`, `ele`, `foi`, etc.) estão ativas. Cada uma delas vota nas possíveis próximas palavras. A palavra com mais votos, ou com peso maior, será a escolhida como a próxima palavra. No exemplo, o par `(programa, executado)` tem um voto alto para `por`, enquanto o par `(bateria, executado)` tem um voto baixo, ou zero. Isso significa que, no contexto da frase, `por` é uma escolha mais provável do que `favor` ou outras palavras. Assim, temos:

* Características como `(programa, executado)` dariam um voto forte para `por`, pois essa sequência (`programa ... executado por favor`) é plausível e ocorre na primeira frase.

* Características como `(bateria, executado)` dariam voto zero, ou muito baixo, para `por`, pois essa combinação não ocorre no nosso exemplo, Na segunda frase temos `bateria ... ela acabou`.

* Características menos informativas, como `(o, executado)` ou `(log, executado)`, podem ter votos distribuídos, ou votos mais fracos. Ver apenas `o` ou `log` antes de `executado` não ajuda muito a distinguir entre as frases originais ou a prever a palavra seguinte.

Para fazer uma previsão, somamos os votos de todas as características ativas, pares formados pela palavra atual e todas as palavras anteriores na sequência específica, para cada palavra candidata a ser a próxima. A palavra com a maior soma de votos será a palavra escolhida.

No exemplo `Verifique o log do programa e descubra se ele foi executado`: as características ativas relevantes para prever a palavra após `executado` incluem pares como `(verifique, executado)`, `(o, executado)`, `(log, executado)`, `(do, executado)`, `(programa, executado)`, `(e, executado)`, `(descubra, executado)`, `(se, executado)`, `(ele, executado)`, `(foi, executado)`. Se somarmos os votos hipotéticos, onde o par informativo `(programa, executado)` tem um voto alto para `por`, e outros pares menos informativos têm votos menores ou distribuídos, a palavra `por` provavelmente acumulará o total de votos mais alto, tornando-se a previsão correta para esta sequência.

![Votação de características de pares com saltos](/assets/images/saltos.webp)
_Figura 2: A figura ilustra o mecanismo de votação de características hipotético para predição da próxima palavra em um modelo baseado em pares com saltos. No exemplo, no contexto da frase `Verifique o log do programa e descubra se ele foi executado`, está sendo prevista a palavra que segue `executado` ._{: class="legend"}

Esta abordagem, embora ainda baseada em pares, oferece uma forma de incorporar contexto de longo alcance de forma seletiva. A implementação pode usar estruturas de dados semelhantes às do modelo de segunda ordem, mas a lógica de treinamento e predição muda para refletir a soma de votos, ou pesos, de múltiplos pares ativos.

Para entender como os valores de votos apresentados na Figura 2 são calculados, precisamos detalhar a matemática conceitual por trás da abordagem de Agregação de Características de Pares. *É importante notar que os valores na figura são ilustrativos, hipotéticos, que foram criados para que a atenta leitora possa entender a ideia de que diferentes pares têm pesos diferentes*. *Um corpus de treinamento real conteria muito mais dados*.

Todo o conceito que vimos até aqui pode ser reduzido a três passos:

1. **Coleta de Evidências (Contagem)**: durante a fase de treinamento, que no nosso caso conceitual será apenas analisar as frases de exemplo, o modelo observa todas as ocorrências de sequências de três palavras $(w_i, w_t, w_{t+1})$. Para cada par $(w_i, w_t)$ que aparece na sequência, ele registra qual palavra $w_{t+1}$ o seguiu. Nesta fase, mantemos uma contagem, $C(w_i, w_t, w_{t+1})\;$, de quantas vezes vimos a palavra $w_{t+1}$ aparecer imediatamente após a palavra $w_t$, dado que $w_i$ apareceu em alguma posição anterior a $t$. Se as sequências tiverem pesos, como no exemplo onde cada frase tem peso $0.5$, somamos esses pesos em vez de apenas contar $1$ para cada ocorrência.

2. **Normalização por Par (Cálculo dos Votos)**: para cada par específico $(w_i, w_t)$ que ocorreu no treinamento, calculamos o voto que este par dá para uma possível próxima palavra $w_k$. Esse voto é a frequência relativa, ou probabilidade condicional estimada, de $w_k$ ocorrer após $(w_i, w_t)\;$, baseada nas contagens. A fórmula para o voto será dada por:

    $$\text{Voto}(w_k  \vert  w_i, w_t) = \frac{C(w_i, w_t, w_k)}{\sum_{w'} C(w_i, w_t, w')}$$

    Na qual, a soma no denominador $\sum_{w'} C(w_i, w_t, w')\;\,$ será feita sobre todas as palavras $w'$ que foram observadas seguindo o par $(w_i, w_t)$ no corpus de treinamento. Isso garante que a soma dos votos de um par específico $(w_i, w_t)$ para todas as possíveis palavras seguintes seja $1$.

    No exemplo hipotético apresentado na Figura 2 vemos que considerando o par $(programa, executado)\;$, o voto para `por` será $0.8$ e para `favor` o voto será $0.1$. Isso implica que, no corpus hipotético usado para gerar a figura, $80\%$ das vezes em que a sequência continha `... programa ... executado ...`, a palavra seguinte foi `por`, e $10\%$ das vezes foi `favor`. Os $10\%$ restantes foram outras palavras não apresentadas na figura.

    Se fizermos uma análise de contraste com as frases do nosso corpus de treinamento: considerando as duas frases dadas, o par $(programa, executado)$ ocorre uma única vez, seguido por `por`. Portanto, um cálculo estrito baseado apenas nessas duas frases resultaria em:

    $$\text{Voto}(\text{por} \vert \text{programa}, \text{executado})\;\; = 1.0$$

    e

    $$\text{Voto}(\text{favor} \vert \text{programa}, \text{executado})\;\; = 0.0$$

    Isso confirma que os valores da figura são ilustrativos de um cenário de dados mais rico. Afinal, $1$ e $0$ são extremos, não tem nenhuma graça e o modelo real deverá ter uma distribuição mais suave entre as palavras candidatas.

3. **Agregação na Predição**: quando queremos prever a palavra após $w_t$ em uma nova sequência, um documento qualquer que não está no corpus de treinamento, identificamos todos os pares $(w_i, w_t)$ formados pela palavra atual $w_t$ e cada palavra anterior $w_i$ na sequência. Em seguida, somamos os votos pré-calculados durante o treinamento, $\text{Voto}(w_k \vert w_i, w_t)\;\,$ para cada palavra candidata $w_k$:

    $$Score(w_k  \vert  \text{sequência até } w_t) = \sum_{i \text{ tal que } w_i \text{ precede } w_t} Voto(w_k  \vert  w_i, w_t)$$

    A palavra $w_k$ com o maior $Score$ agregado é a previsão do modelo. A Figura 2 ilustra a etapa antes dessa agregação final, mostrando os votos individuais $\text{Voto}(w_k  \vert  w_i, w_t)$ para $w_t = \text{"executado"}$ e vários $w_i$.

Eu parti de um exemplo simples, livre, leve e solto, para que a esforçada leitora tenha uma chance maior de entender a ideia. Mas, deve ser possível imaginar que essa abordagem pode ser aplicada a sequências muito mais longas e complexas. A ideia é que, ao considerar todos os pares $(w_i, w_t)\;$, podemos capturar dependências de longo alcance que seriam impossíveis com um modelo **N-gram** tradicional. Talvez um exemplo mais complexo ajude a fixar a ideia.

#### Exemplo Detalhado: Modelo de Agregação de Características de Pares

Mesmo que o título desta seção seja exemplo detalhado, eu vou ignorar que no mundo real, passaríamos o corpus por alguns processos de preparação de texto antes da aplicação de qualquer algoritmo. Sendo assim, para este exemplo a esforçada leitora deve considerar um **corpus de treinamento** com os $5$ documentos a seguir:

1. $D_1 =\;$ `Verifique o log do programa e descubra se ele foi executado, por favor.`;
2. $D_2 =\;$`Verifique o log da bateria e descubra se ela acabou, por favor.`;
3. $D_3 =\;$`O programa foi executado com sucesso, por isso não precisa verificar novamente.`;
4. $D_4 =\;$`A bateria foi substituída, por isso está funcionando corretamente.`;
5. $D_5 =\;$`Ele executou o programa, por isso obteve os resultados esperados.`.

Cada documento tem peso igual a $0.2$ no corpus. Isso quer dizer que estes documentos são igualmente prováveis no nosso sistema. Podemos calcular explicitamente as probabilidades e valores para o nosso modelo.

##### 1. Coleta de Evidências (Contagem)

Primeiro, precisamos contar todas as ocorrências de triplas $(w_i, w_t, w_{t+1})\;$, na qual:

* $w_i$ é uma palavra anterior na sequência;
* $w_t$ é a palavra atual para a qual queremos prever a próxima;
* $w_{t+1}$ é a palavra que segue $w_t$.

Para este exemplo, vamos tentar prever a palavra após a palavra `executado` em diferentes contextos.

1. Contagens para $(w_i, \text{executado}, w_{t+1})$

| $w_i$ | $w_t$ | $w_{t+1}$ | Ocorrências | Peso Total |
|-------|-------|------------|-------------|------------|
| programa | executado | por | 1 | 0.2 |
| foi | executado | por | 1 | 0.2 |
| ele | executado | por | 1 | 0.2 |
| foi | executado | com | 1 | 0.2 |
| programa | executado | com | 1 | 0.2 |

$$\\$$

##### 2. Normalização por Par (Cálculo dos Votos)

Para cada par $(w_i, w_t)\;$, calculamos o voto para cada possível próxima palavra $w_k$ usando:

$$\text{Voto}(w_k \vert w_i, w_t) = \frac{C(w_i, w_t, w_k)}{\sum_{w'} C(w_i, w_t, w')}$$

1. Cálculo para o par $(\text{programa}, \text{executado})$

    $$\text{Voto}(\text{por} \vert \text{programa}, \text{executado}) = \frac{0.2}{0.2 + 0.2} = \frac{0.2}{0.4} = 0.5$$

    $$\text{Voto}(\text{com} \vert \text{programa}, \text{executado}) = \frac{0.2}{0.2 + 0.2} = \frac{0.2}{0.4} = 0.5$$

2. Cálculo para o par $(\text{foi}, \text{executado})$

    $$\text{Voto}(\text{por} \vert \text{foi}, \text{executado}) = \frac{0.2}{0.2 + 0.2} = \frac{0.2}{0.4} = 0.5$$

    $$\text{Voto}(\text{com} \vert \text{foi}, \text{executado}) = \frac{0.2}{0.2 + 0.2} = \frac{0.2}{0.4} = 0.5$$

3. Cálculo para o par $(\text{ele}, \text{executado})$

    $$\text{Voto}(\text{por} \vert \text{ele}, \text{executado}) = \frac{0.2}{0.2} = 1.0$$

    $$\text{Voto}(\text{com} \vert \text{ele}, \text{executado}) = \frac{0}{0.2} = 0.0$$

##### 3. Agregação na Predição

Nosso próximo passo é prever a próxima palavra após `executado` em uma nova sequência:

`Verifique se o programa foi executado ...`

Antes de qualquer coisa a atenta leitora deve observar que `Verifique se o programa foi executado ...` não faz parte do corpus de treinamento. Portanto, não temos certeza de como o modelo irá se comportar. Vamos analisar os pares $(w_i, w_t)$ formados por `executado` e as palavras anteriores na sequência. Neste caso, as palavras anteriores relevantes são: `verifique`, `se`, `o`, `programa`, `foi`.

Focaremos nos pares que já vimos no treinamento:

| Par $(w_i, w_t)$ | Voto para `por` | Voto para `com` |
|------------------|-----------------|-----------------|
| (programa, executado) | 0.5 | 0.5 |
| (foi, executado) | 0.5 | 0.5 |

Agregando estes votos:

$$\text{Score}(\text{por}) = 0.5 + 0.5 = 1.0$$

$$\text{Score}(\text{com}) = 0.5 + 0.5 = 1.0$$

Neste caso, o modelo não consegue distinguir claramente entre `por` e `com` baseado apenas nas duas palavras-chave. Isso demonstra uma limitação do modelo quando apenas alguns pares são informativos.

Se incluirmos mais contexto, como em outra sequência nova:

`Verifique se ele executou o programa ...`

Então temos:

| Par $(w_i, w_t)$ | Voto para `por` | Voto para `com` |
|------------------|-----------------|-----------------|
| (ele, executado) | 1.0 | 0.0 |

Neste caso:

$$\text{Score}(\text{por}) = 1.0$$

$$\text{Score}(\text{com}) = 0.0$$

O modelo prevê claramente `por` como a próxima palavra.

Este exemplo ilustra como a presença de palavras-chave distintivas (`ele` vs. `programa` ou `foi`) pode alterar significativamente a previsão, demonstrando como o modelo captura dependências de longo alcance.

#### Implementação em C++

O código C++ apresentado abaixo, encapsulado na classe `PairwiseFeatureAggregator`, implementa precisamente os passos que acabamos de detalhar no exemplo. O método `addSequence` corresponde à coleta de evidências (Passo 1), `normalizeVotes` executa o cálculo dos votos normalizados por par (Passo 2), e `predictNextWord` realiza a agregação desses votos para efetuar a predição em novas sequências (Passo 3).

```cpp
#include <iostream>        ///< Para entrada e saída padrão (std::cout).
#include <vector>          ///< Para contêiner std::vector usado no armazenamento de sequências.
#include <string>          ///< Para std::string, usado em palavras e mensagens.
#include <unordered_map>   ///< Para std::unordered_map, usado no armazenamento de transições.
#include <map>             ///< Para std::map, usado na ordenação de predições.
#include <set>             ///< Para std::set, usado no armazenamento do vocabulário.
#include <iomanip>         ///< Para std::fixed e std::setprecision, usados na formatação de saída.

/**
 * @class PairwiseFeatureAggregator
 * @brief Uma classe para modelar transições de palavras com base em pares com saltos.
 *
 * Esta classe implementa um modelo que associa pares de palavras (anterior, atual) a palavras seguintes,
 * acumulando pesos (votos) para prever a próxima palavra em uma sequência. Os pesos são normalizados
 * para cada par (anterior, atual) como probabilidades condicionais.
 */
class PairwiseFeatureAggregator {
private:
    /**
     * @brief Estrutura para armazenar transições: transitions[palavra_anterior][palavra_atual][proxima_palavra] -> peso.
     */
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          std::unordered_map<std::string, double>>> transitions;
    std::set<std::string> vocabulary; ///< Vocabulário global de palavras.

public:
    /**
     * @brief Adiciona uma sequência ao modelo, acumulando pesos para transições.
     * @param sequence Vetor de strings representando a sequência de palavras.
     * @param weight Peso a ser atribuído à sequência (padrão é 1.0).
     */
    void addSequence(const std::vector<std::string>& sequence, double weight = 1.0) {
        if (sequence.size() < 3) return; // Necessário pelo menos (anterior, atual, próxima)

        // Adicionar palavras ao vocabulário
        for (const auto& word : sequence) {
            vocabulary.insert(word);
        }

        // Para cada posição atual, associar pares (anterior, atual) à próxima palavra
        for (size_t current_pos = 1; current_pos < sequence.size() - 1; ++current_pos) {
            const std::string& current_word = sequence[current_pos];
            const std::string& next_word = sequence[current_pos + 1];

            for (size_t prev_pos = 0; prev_pos < current_pos; ++prev_pos) {
                const std::string& prev_word = sequence[prev_pos];
                transitions[prev_word][current_word][next_word] += weight;
            }
        }
    }

    /**
     * @brief Normaliza os pesos (votos) para cada par (anterior, atual) como probabilidades.
     */
    void normalizeVotes() {
        for (auto& [prev_word, inner_map1] : transitions) {
            for (auto& [current_word, inner_map2] : inner_map1) {
                double total_votes = 0.0;
                for (auto const& [next_word, vote] : inner_map2) {
                    total_votes += vote;
                }
                if (total_votes > 0) {
                    for (auto& [next_word, vote] : inner_map2) {
                        inner_map2[next_word] /= total_votes;
                    }
                }
            }
        }
    }

    /**
     * @brief Prediz a próxima palavra com base na sequência fornecida.
     * @param sequence Vetor de strings representando a sequência de entrada.
     * @return Mapa ordenado de palavras candidatas para a próxima palavra e seus pesos acumulados.
     */
    std::map<std::string, double> predictNextWord(const std::vector<std::string>& sequence) const {
        std::map<std::string, double> accumulated_votes;
        if (sequence.empty()) return accumulated_votes;

        const std::string& current_word = sequence.back();
        for (size_t i = 0; i < sequence.size() - 1; ++i) {
            const std::string& prev_word = sequence[i];
            auto it1 = transitions.find(prev_word);
            if (it1 != transitions.end()) {
                auto it2 = it1->second.find(current_word);
                if (it2 != it1->second.end()) {
                    for (const auto& [next_word, vote] : it2->second) {
                        accumulated_votes[next_word] += vote;
                    }
                }
            }
        }
        return accumulated_votes;
    }

    /**
     * @brief Obtém o vocabulário do modelo.
     * @return Referência constante ao conjunto de palavras do vocabulário.
     */
    const std::set<std::string>& getVocabulary() const {
        return vocabulary;
    }

    /**
     * @brief Exibe as predições ordenadas por peso.
     * @param predictions Mapa de palavras candidatas e seus pesos.
     * @param sequence Vetor de strings representando a sequência de entrada.
     */
    void printPredictions(const std::map<std::string, double>& predictions, const std::vector<std::string>& sequence) const {
        std::cout << "Predição (soma de votos normalizados) após sequência terminando com '" << sequence.back() << "':\n";
        std::multimap<double, std::string, std::greater<double>> sorted_predictions;
        for (const auto& [word, vote] : predictions) {
            sorted_predictions.insert({vote, word});
        }
        for (const auto& [vote, word] : sorted_predictions) {
            std::cout << "  " << word << ": " << std::fixed << std::setprecision(2) << vote << "\n";
        }
    }
};

/**
 * @brief Função principal que demonstra o uso da classe PairwiseFeatureAggregator.
 *
 * Este programa cria um modelo de predição de palavras com base em pares com saltos, treina o modelo
 * com duas sequências representando verificações de log de um programa e uma bateria, normaliza os
 * pesos, e testa a predição da próxima palavra para as sequências fornecidas.
 *
 * @return 0 em caso de execução bem-sucedida.
 */
int main() {
    // Criar o modelo
    PairwiseFeatureAggregator model; ///< Instância do modelo de agregação de características.

    // Definir sequências de treinamento
    std::vector<std::string> sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado", "por", "favor"
    };
    std::vector<std::string> sequence2 = {
        "verifique", "o", "log", "da", "bateria", "e", "descubra", "se", "ela", "acabou", "por", "favor"
    };

    // Adicionar sequências com pesos iguais (50% cada)
    model.addSequence(sequence1, 0.5);
    model.addSequence(sequence2, 0.5);

    // Normalizar os pesos
    model.normalizeVotes();

    // Testar predições para a sequência 1 (terminando em "executado")
    std::vector<std::string> test_sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado"
    };
    auto predictions1 = model.predictNextWord(test_sequence1);
    model.printPredictions(predictions1, test_sequence1);

    // Testar predições para a sequência 2 (terminando em "acabou")
    std::vector<std::string> test_sequence2 = {
        "verifique", "o", "log", "da", "bateria", "e", "descubra", "se", "ela", "acabou"
    };
    auto predictions2 = model.predictNextWord(test_sequence2);
    model.printPredictions(predictions2, test_sequence2);

    return 0;
}

```

#### Considerações importantes

Embora esta abordagem de pares com saltos e votação nos permita considerar contexto de longo alcance, ela enfatiza um característica negativa. Ao somar votos de muitas características (pares), a contribuição das poucas características *realmente* informativas (como `(programa, executado)` no nosso exemplo) pode ser diluída pelo "ruído" das características menos úteis (como `(o, executado)`). A diferença entre o total de votos para a palavra correta e as incorretas pode ser pequena, tornando o modelo menos confiável e menos robusto. A esse problema chamamos de **diluição do sinal**.

Além dessa questão fundamental da **diluição do sinal**, a abordagem de agregação irrestrita de pares, na prática, enfrenta outros desafios que limitam sua aplicabilidade em cenários mais complexos:

* **Complexidade Computacional e de Memória**: a estrutura de dados usada para armazenar as contagens ou votos, como o `std::unordered_map` triplamente aninhado no código C++ (aaargh!), pode se tornar inaceitavelmente grande para corpus com vocabulários extensos. Pior ainda, durante a predição para uma sequência de comprimento $T$, o modelo precisa potencialmente considerar e somar votos de $O(T^2)$ pares $(w_i, w_t)$. *Isso torna o método computacionalmente caro e difícil de escalar para as sequências longas* frequentemente encontradas em tarefas de Processamento de Linguagem Natural do mundo real.

* **Interpretação da Pontuação Final (Normalização)**: a normalização é realizada individualmente para cada par $(w_i, w_t)\;$, garantindo que $\sum_{w_k} \text{Voto}(w_k \vert w_i, w_t) = 1$. No entanto, a pontuação final para uma palavra candidata $w_k$, calculada como $Score(w_k) = \sum_{i \text{ t.q.} w_i \text{ precede } w_t} \text{Voto}(w_k  \vert  w_i, w_t)\;$, é uma simples soma dessas probabilidades condicionais. O resultado $Score(w_k)$ não representa mais uma probabilidade bem calibrada; a soma $\sum_{w_k} Score(w_k)$ não é necessariamente $1$. *A pontuação final funciona como um ranking, onde valores mais altos são melhores, mas perde uma interpretação probabilística direta sobre a confiança da previsão*.

**A forma que encontramos para superar estas dificuldades inclui tentar fazer o modelo prestar atenção dinamicamente nas características, pares ou, mais geralmente, nas palavras anteriores, que são mais relevantes para a previsão atual, mitigando o ruído e controlando a complexidade.**

### Mascaramento e Atenção Seletiva: Focando no que Importa

*A solução para a diluição dos votos é introduzir um mecanismo que permita ao modelo **prestar atenção seletiva** às características mais informativas, ignorando ou diminuindo o peso das demais.* Podemos fazer isso através de uma técnica que chamaremos de **mascaramento**.

Imagine que, para a tarefa de prever a palavra após `executado` no contexto da frase sobre o `programa`, pudéssemos saber, ou de alguma forma aprender, que as características mais importantes são aquelas que envolvem `programa` e talvez `foi`, enquanto outras como `verifique`, `o`, `log`, etc., são menos preditivas neste ponto específico.

Podemos criar uma **máscara**. Esta máscara será, essencialmente, um vetor ou conjunto de pesos, um para cada característica (ou para cada palavra anterior que forma um par com a palavra atual). A máscara atribuirá peso $1$, ou um peso alto, às características/palavras anteriores consideradas importantes e peso $0$, ou um peso baixo, às demais.

![Atividades de características mascaradas](/assets/images/mascara1.webp)
_Figura 3: Aplicação de uma máscara conceitual. A máscara (à direita) atribui peso 1 para "programa" e "bateria" (assumindo que estas são as palavras-chave distintivas) e 0 para as outras. Ao multiplicar os pesos das características ativas (centro) pela máscara, apenas as características relevantes ("programa, executado" neste caso) mantêm seu peso (resultado à esquerda)._{: class="legend"}

Para aplicar a máscara aos "votos" que calculamos na seção anterior, realizamos uma multiplicação elemento a elemento (produto Hadamard) entre o vetor de votos de cada possível palavra seguinte e a máscara apropriada, ou, de forma equivalente, aplicamos a máscara ao *contexto* antes de calcular os votos acumulados. Qualquer voto originado de uma característica/par mascarado (peso $0$ na máscara) é zerado.

>**O Produto de Hadamard**
>
>O produto de Hadamard entre duas matrizes $A$ e $B$ de mesmas dimensões, denotado por $A \circ B$ ou $A \odot B$, é uma operação que multiplica os elementos correspondentes das duas matrizes:
>
>$$(A \circ B)_{ij} = A_{ij} \cdot B_{ij}$$
>
>Ao contrário da multiplicação matricial padrão, esta operação preserva as dimensões originais e é computacionalmente eficiente. No contexto da atenção seletiva em **Transformers**, o produto de Hadamard é utilizado para aplicar a máscara ao vetor de características:
>
>$$\text{Características}_{\text{mascaradas}} = \text{Características}_{\text{originais}} \circ \text{Máscara}$$
>
>Onde a máscara contém valores binários ($0$ ou $1$) ou pesos contínuos entre 0 e 1 que determinam a importância relativa de cada elemento. Esta operação permite que o modelo "filtre" características irrelevantes (multiplicando por $0$) enquanto mantém as relevantes (multiplicando por 1 ou por um peso não-nulo).

A máscara funciona como se estivesse apagando temporariamente as partes menos relevantes da nossa matriz conceitual de votos, deixando apenas as conexões que realmente importam para a decisão atual.

![Matriz de transição mascarada](/assets/images/masked-transition-matrix.webp)
_Figura 4: Matriz de votos/transição conceitual após a aplicação da máscara. Apenas as linhas/características consideradas relevantes (ex: envolvendo "programa" ou "bateria") permanecem ativas, tornando a previsão muito mais direcionada._{: class="legend"}

*Após aplicar a máscara, a soma dos votos torna-se mais decisiva_. No nosso exemplo, se a máscara destacar apenas a característica `(programa, executado)`, o total de votos para `por` virá predominantemente, ou exclusivamente, dessa característica, enquanto os votos para outras palavras serão zerados ou drasticamente reduzidos. _Este resultado provoca que a confiança do modelo na previsão correta aumenta*.

**Este processo de focar seletivamente em partes específicas do input (as palavras anteriores mais relevantes, neste caso) para tomar uma decisão sobre o output (a próxima palavra) é a intuição fundamental por trás do mecanismo de ATENÇÃO.**

O artigo seminal *Attention is All You Need* (Vaswani et al., 2017) introduziu uma forma específica e poderosa de implementar essa ideia, que se tornou a base dos modelos **Transformer**.

>A origem do termo está diretamente ligada à capacidade do modelo de transformar representações de sequências (como texto) por meio de mecanismos de autoatenção (self-attention), sem depender de redes neurais recorrentes (RNNs) ou convolucionais (CNNs).

A persistente leitora deve observar que o que descrevemos até agora é uma aproximação conceitual para construir o entendimento. Entretanto, assim como fizemos anteriormente, vamos recorrer a um exemplo mais rigoroso de como o mascaramento funciona matematicamente em um cenário próximo da realidade.

#### Exemplo Detalhado: Mascaramento em Corpus Realista

Consideremos um vocabulário $V = \{w_1, w_2, ..., w_{ \vert V \vert }\}$ e uma sequência $S = [w_5, w_{17}, w_3, w_{42}, w_{11}]$. Vamos focar na posição atual $t=4$ (palavra $w_{11}$) e considerar todas as posições anteriores $i \in \{0, 1, 2, 3\}$.

1. **Representação vetorial (embeddings)**

    >Em processamento de linguagem natural, um **embedding** é uma representação vetorial de palavras, frases ou outros elementos linguísticos em um espaço contínuo de baixa dimensão. É uma técnica fundamental que permite converter palavras (elementos discretos) em vetores numéricos $\vec{w} \in \mathbb{R}^d$ que capturem suas propriedades semânticas e relações com outras palavras.
    >
    >As principais características dos embeddings são:
    >
    >1. **Representação densa em vetor**: Cada palavra $w$ é mapeada para um vetor $\vec{w} = [w_1, w_2, ..., w_d]$ de números reais, tipicamente com dimensões entre $50 \leq d \leq 300$ elementos (em vez de um vetor one-hot esparso com milhares ou milhões de dimensões).
    >
    >2. **Preservação de similaridade semântica**: Palavras com significados semelhantes ficam próximas no espaço vetorial. A similaridade é frequentemente medida usando a similaridade de cosseno:
    >
    >   $$\text{sim}(\vec{w}_i, \vec{w}_j) = \frac{\vec{w}_i \cdot \vec{w}_j}{ \vert \vec{w}_i \vert  \cdot  \vert \vec{w}_j \vert }$$
    >
    >3. **Captura de relações analógicas**: Embeddings bem treinados podem capturar relações como "rei está para rainha assim como homem está para mulher" através de operações vetoriais simples:
    >
    >   $$\vec{v}_{\text{rei}} - \vec{v}_{\text{homem}} + \vec{v}_{\text{mulher}} \approx \vec{v}_{\text{rainha}}$$
    >
    >4. **Aprendizado por contexto**: Os embeddings são geralmente aprendidos observando como as palavras aparecem em contextos semelhantes em grandes corpus de texto, otimizando uma função objetivo como:
    >
    >  $$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j}  \vert  w_t)$$
    >
    >na qual, $c$ é o tamanho da janela de contexto e $p(w_{t+j}  \vert  w_t)$ é a probabilidade de observar a palavra $w_{t+j}$ dado $w_t$.
    >
    Alguns modelos populares de word embeddings incluem:
    >
    >* **Word2Vec**: Desenvolvido pelo Google em 2013, usando redes neurais para prever palavras vizinhas (**skipgram**) ou a palavra atual a partir das vizinhas (**CBOW**).
    >* **GloVe**: Desenvolvido por Stanford, combinando estatísticas globais de co-ocorrência $X_{ij}$ com aprendizado local de contexto:
    >
    >$$J = \sum_{i,j=1}^{ \vert V \vert } f(X_{ij})(\vec{w}_i^T\vec{w}_j + b_i + b_j - \log X_{ij})^2$$
    >
    >* **FastText**: Desenvolvido pelo Facebook, considera subpalavras (**N-grams** de caracteres) para lidar melhor com palavras raras e morfologia:
    >
    > $$\vec{w} = \frac{1}{ \vert G_w \vert } \sum_{g \in G_w} \vec{z}_g$$
    onde $G_w$ é o conjunto de **N-grams** na palavra $w$ e $\vec{z}_g$ é o vetor do **N-gram** $g$.
    >
    >Em modelos como os Transformers, os embeddings são apenas o primeiro passo. Cada token (palavra ou subpalavra) é primeiro convertido em um vetor de embedding e depois processado através das camadas de atenção e feed-forward para produzir representações contextualizadas $\mathbf{h}_i^l$ que capturam o significado da palavra no contexto específico em que aparece:
    >
    >$$\mathbf{h}_i^l = \text{TransformerLayer}_l(\mathbf{h}_i^{l-1}, \mathbf{h}_{\neq i}^{l-1})$$
    >
    >na qual, $\mathbf{h}_i^0 = \text{Embedding}(w_i) + \text{PositionalEncoding}(i)$
    >

    Voltando ao nosso exemplo, cada palavra é representada por um vetor de embedding $\mathbf{e}_i \in \mathbb{R}^d$, onde $d$ é a dimensão do embedding. Assumamos $d=4$ para simplicidade:

    $$\mathbf{e}_0 = \mathbf{e}_{w_5} = [0.2, -0.3, 0.1, 0.5]$$

    $$\mathbf{e}_1 = \mathbf{e}_{w_{17}} = [0.4, 0.1, -0.2, 0.3]$$

    $$\mathbf{e}_2 = \mathbf{e}_{w_3} = [-0.1, 0.5, 0.3, 0.2]$$

    $$\mathbf{e}_3 = \mathbf{e}_{w_{42}} = [0.6, 0.2, -0.4, 0.1]$$

    $$\mathbf{e}_4 = \mathbf{e}_{w_{11}} = [0.3, 0.4, 0.2, -0.1]$$

2. **Cálculo de Query, Key, Value**:

    A conversão de embeddings em vetores **Query**, **Key** e **Value** por meio de transformações lineares é parte fundamental do mecanismo de atenção. Essas transformações são essenciais para permitir que o modelo aprenda a focar em diferentes partes do input, dependendo do contexto e da tarefa.

    Notadamente porque os embeddings originais $\mathbf{X} \in \mathbb{R}^{n \times d}$ representam palavras em um espaço semântico geral. As transformações lineares permitem projetar esses embeddings em subespaços especializados:

    $$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

    de tal forma que $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d_k}$ são matrizes de parâmetros treináveis.

    Cada projeção serve a um propósito específico:

    * $\mathbf{Q}$ (Query): Codifica como uma palavra "busca" informações relevantes;
    * $\mathbf{K}$ (Key): Determina como uma palavra "responde" a buscas;
    * $\mathbf{V}$ (Value): Contém a informação semântica efetiva a ser propagada.

    Sem estas transformações, o mecanismo de atenção seria limitado à equação:

    $$\text{Attention}(\mathbf{X}, \mathbf{X}, \mathbf{X}) = \text{softmax}\left(\frac{\mathbf{X}\mathbf{X}^T}{\sqrt{d}}\right)\mathbf{X}$$

    O uso de parâmetros separados $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ aumenta significativamente os graus de liberdade do modelo:

    $$\text{Attention}(\mathbf{X}\mathbf{W}^Q, \mathbf{X}\mathbf{W}^K, \mathbf{X}\mathbf{W}^V) = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}^Q(\mathbf{X}\mathbf{W}^K)^T}{\sqrt{d_k}}\right)\mathbf{X}\mathbf{W}^V$$

    ***As matrizes $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ são parâmetros aprendidos durante o treinamento***, permitindo que o modelo adapte dinamicamente:

    1. Quais aspectos dos embeddings são importantes para consultas;
    2. Quais aspectos tornam um token "consultável" por outros;
    3. Quais informações devem ser transmitidas quando um token é consultado.

    Além disso, as transformações lineares permitem mapear embeddings de dimensão $d$ para vetores $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ de dimensão $d_k$, potencialmente diferente de $d$. Isso é útil para reduzir a complexidade computacional, especialmente em tarefas de atenção multi-cabeça, onde cada cabeça pode ter uma dimensão diferente:

    $$\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d_k}$$

    Isso possibilita controlar a complexidade computacional e a capacidade representacional do mecanismo de atenção.

    Voltando ao nosso exemplo: usamos transformações lineares para converter embeddings em vetores Query, Key e Value:

    $$\mathbf{Q} = \mathbf{E}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{E}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{E}\mathbf{W}^V$$

    Onde $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d_k}$ são matrizes de parâmetros aprendidas. Assumiremos $d_k = 3$ e matrizes simplificadas:

    $$\mathbf{W}^Q = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.1 \\ 0.2 & 0.3 & 0.4 \\ 0.5 & 0.1 & 0.2 \end{bmatrix}, \mathbf{W}^K = \begin{bmatrix} 0.3 & 0.2 & 0.1 \\ 0.1 & 0.4 & 0.3 \\ 0.5 & 0.2 & 0.3 \\ 0.2 & 0.1 & 0.5 \end{bmatrix}, \mathbf{W}^V = \begin{bmatrix} 0.2 & 0.3 & 0.1 \\ 0.4 & 0.2 & 0.3 \\ 0.1 & 0.5 & 0.2 \\ 0.3 & 0.1 & 0.4 \end{bmatrix}$$

    Calculando $\mathbf{q}_4$ (Query para posição atual), $\mathbf{k}_i$ (Key para cada posição) e $\mathbf{v}_i$ (Value para cada posição):

    $$\mathbf{q}_4 = \mathbf{e}_4\mathbf{W}^Q = [0.3, 0.4, 0.2, -0.1] \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.1 \\ 0.2 & 0.3 & 0.4 \\ 0.5 & 0.1 & 0.2 \end{bmatrix} = [0.23, 0.31, 0.21]$$

    De modo similar, calculamos $\mathbf{k}_i$ para $i \in \{0, 1, 2, 3, 4\}$ e $\mathbf{v}_i$ para $i \in \{0, 1, 2, 3, 4\}$.

3. **Cálculo dos scores de atenção (sem mascaramento)**:

    Os scores de atenção, similaridade entre $\mathbf{q}_4$ e cada $\mathbf{k}_i$ são:

    $$s_{4,i} = \frac{\mathbf{q}_4 \cdot \mathbf{k}_i}{\sqrt{d_k}}$$

    Por exemplo:

    $$s_{4,0} = \frac{[0.23, 0.31, 0.21] \cdot [0.23, 0.05, 0.22]}{\sqrt{3}} = \frac{0.1262}{\sqrt{3}} = 0.0729$$

    Similarmente, calculamos $s_{4,1} = 0.1039$, $s_{4,2} = 0.1501$, $s_{4,3} = 0.0592$ e $s_{4,4} = 0.1327$

4. **Aplicação de mascaramento explícito**:

    A ideia do mascaramento é zerar scores de posições não relevantes. Vamos considerar que apenas as posições 0 e 2 são relevantes para a posição atual. Criamos uma máscara $\mathbf{M}$:

    $$\mathbf{M}_4 = [1, 0, 1, 0, 1]$$

    Aplicamos a máscara aos scores:

    $$\tilde{s}_{4,i} = s_{4,i} \cdot \mathbf{M}_{4,i}$$

    Resultando em:

    $$\tilde{s}_{4,0} = 0.0729 \cdot 1 = 0.0729$$

    $$\tilde{s}_{4,1} = 0.1039 \cdot 0 = 0$$

    $$\tilde{s}_{4,2} = 0.1501 \cdot 1 = 0.1501$$

    $$\tilde{s}_{4,3} = 0.0592 \cdot 0 = 0$$

    $$\tilde{s}_{4,4} = 0.1327 \cdot 1 = 0.1327$$

5. **Normalização via softmax**:

    A função **softmax** é uma transformação matemática fundamental em aprendizado de máquina, especialmente em redes neurais e modelos de linguagem. Ela converte um vetor de números reais em uma distribuição de probabilidade.

    Para um vetor $\mathbf{z} = [z_1, z_2, \ldots, z_n]$, a função softmax é definida como:

    $$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

    Esta função tem propriedades importantes:

    1. **Normalização**: os valores resultantes somam $1$, permitindo sua interpretação como probabilidades:

        $$\sum_{i=1}^{n} \text{softmax}(\mathbf{z})_i = 1$$

    2. **Diferenciabilidade**: a função é suave e tem derivadas bem definidas em todos os pontos, facilitando o treinamento por gradiente descendente:

        $$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = 
        \begin{cases} 
        \text{softmax}(\mathbf{z})_i(1-\text{softmax}(\mathbf{z})_i) & \text{se } i = j \\
        -\text{softmax}(\mathbf{z})_i\text{softmax}(\mathbf{z})_j & \text{se } i \neq j
        \end{cases}$$

        3. **Não-linearidade**: Transforma relações lineares em não-lineares, permitindo que redes neurais aprendam mapeamentos complexos.

        4. **Amplificação**: Enfatiza valores maiores e suprime menores. Se $z_i \gg z_j$, então $\text{softmax}(\mathbf{z})_i \gg \text{softmax}(\mathbf{z})_j$.

    3. **Escala invariante**: Adicionar uma constante $c$ a todos os elementos não altera o resultado:

        $$\text{softmax}([z_1+c, z_2+c, \ldots, z_n+c])_i = \text{softmax}([z_1, z_2, \ldots, z_n])_i$$

        Esta propriedade é frequentemente explorada para estabilidade numérica:

        $$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i-\max(\mathbf{z})}}{\sum_{j=1}^{n} e^{z_j-\max(\mathbf{z})}}$$

    No contexto do mecanismo de atenção em Transformers, a softmax normaliza os scores de atenção em pesos que somam $1$:

    $$\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^{n} \exp(s_{ik})}$$

    onde $s_{ij}$ é o score de similaridade entre as posições $i$ e $j$.

    No nosso exemplo os pesos de atenção normalizados são obtidos aplicando softmax aos scores mascarados:

    $$\alpha_{4,i} = \frac{\exp(\tilde{s}_{4,i})}{\sum_{j=0}^{4} \exp(\tilde{s}_{4,j})}$$

    Calculando:

    $$\alpha_{4,0} = \frac{\exp(0.0729)}{\exp(0.0729) + \exp(0) + \exp(0.1501) + \exp(0) + \exp(0.1327)}$$

    $$\alpha_{4,0} = \frac{1.0756}{1.0756 + 1 + 1.1620 + 1 + 1.1419} = 0.2518$$

    Similarmente:

    $$\alpha_{4,1} = 0.2341$$

    $$\alpha_{4,2} = 0.2720$$

    $$\alpha_{4,3} = 0.2341$$

    $$\alpha_{4,4} = 0.2674$$

6. **Combinação ponderada dos Values**:

    Finalmente, o vetor de contexto para a posição $4$ é:

    $$\mathbf{c}_4 = \sum_{i=0}^{4} \alpha_{4,i} \mathbf{v}_i$$

    $$\mathbf{c}_4 = 0.2518 \cdot \mathbf{v}_0 + 0 \cdot \mathbf{v}_1 + 0.2720 \cdot \mathbf{v}_2 + 0 \cdot \mathbf{v}_3 + 0.2674 \cdot \mathbf{v}_4$$

    Este vetor de contexto $\mathbf{c}_4$ agora contém informações das posições relevantes (0, 2, 4), com as posições não relevantes (1, 3) efetivamente excluídas pelo mascaramento.

#### Implementação em C++

Vejamos como o mascaramento pode ser implementado em C++, aplicando a máscara *antes* de acumular os votos (demonstração conceitual):

```cpp
#include <iostream>         ///< Para entrada e saída padrão (std::cout).
#include <vector>          ///< Para contêiner std::vector usado no armazenamento de sequências.
#include <string>          ///< Para std::string, usado em palavras e mensagens.
#include <unordered_map>   ///< Para std::unordered_map, usado no armazenamento de transições.
#include <map>             ///< Para std::map, usado na ordenação de predições.
#include <set>             ///< Para std::set, usado no armazenamento do vocabulário e máscaras.
#include <iomanip>         ///< Para std::fixed e std::setprecision, usados na formatação de saída.
#include <cmath>           ///< Para std::max (não usado diretamente, mantido por compatibilidade).

/**
 * @class PairwiseFeatureAggregator
 * @brief Uma classe para modelar transições de palavras com base em pares com saltos, com suporte a mascaramento.
 *
 * Esta classe associa pares de palavras (anterior, atual) a palavras seguintes, acumulando pesos (votos)
 * para prever a próxima palavra em uma sequência. Suporta mascaramento para considerar apenas palavras
 * anteriores específicas como relevantes. Os pesos são normalizados como probabilidades condicionais
 * para cada par (anterior, atual).
 */
class PairwiseFeatureAggregator {
private:
    /**
     * @brief Estrutura para armazenar transições: transitions[palavra_anterior][palavra_atual][proxima_palavra] -> peso.
     */
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          std::unordered_map<std::string, double>>> transitions;
    std::set<std::string> vocabulary; ///< Vocabulário global de palavras.

public:
    /**
     * @brief Adiciona uma sequência ao modelo, acumulando pesos para transições.
     * @param sequence Vetor de strings representando a sequência de palavras.
     * @param weight Peso a ser atribuído à sequência (padrão é 1.0).
     */
    void addSequence(const std::vector<std::string>& sequence, double weight = 1.0) {
        if (sequence.size() < 3) return; // Necessário pelo menos (anterior, atual, próxima)

        // Adicionar palavras ao vocabulário
        for (const auto& word : sequence) {
            vocabulary.insert(word);
        }

        // Para cada posição atual, associar pares (anterior, atual) à próxima palavra
        for (size_t current_pos = 1; current_pos < sequence.size() - 1; ++current_pos) {
            const std::string& current_word = sequence[current_pos];
            const std::string& next_word = sequence[current_pos + 1];

            for (size_t prev_pos = 0; prev_pos < current_pos; ++prev_pos) {
                const std::string& prev_word = sequence[prev_pos];
                transitions[prev_word][current_word][next_word] += weight;
            }
        }
    }

    /**
     * @brief Normaliza os pesos (votos) para cada par (anterior, atual) como probabilidades.
     */
    void normalizeVotes() {
        for (auto& [prev_word, inner_map1] : transitions) {
            for (auto& [current_word, inner_map2] : inner_map1) {
                double total_votes = 0.0;
                for (auto const& [next_word, vote] : inner_map2) {
                    total_votes += vote;
                }
                if (total_votes > 0) {
                    for (auto& [next_word, vote] : inner_map2) {
                        inner_map2[next_word] /= total_votes;
                    }
                }
            }
        }
    }

    /**
     * @brief Prediz a próxima palavra com base na sequência fornecida, considerando uma máscara de palavras anteriores.
     * @param sequence Vetor de strings representando a sequência de entrada.
     * @param relevant_prev_words Conjunto de palavras anteriores consideradas relevantes (máscara).
     * @return Mapa ordenado de palavras candidatas para a próxima palavra e seus pesos acumulados.
     */
    std::map<std::string, double> predictNextWordWithMask(
        const std::vector<std::string>& sequence,
        const std::set<std::string>& relevant_prev_words) const {
        std::map<std::string, double> accumulated_votes;
        if (sequence.empty()) return accumulated_votes;

        const std::string& current_word = sequence.back();
        for (size_t i = 0; i < sequence.size() - 1; ++i) {
            const std::string& prev_word = sequence[i];
            if (relevant_prev_words.count(prev_word) == 0) {
                continue; // Pula palavras anteriores não relevantes
            }

            auto it1 = transitions.find(prev_word);
            if (it1 != transitions.end()) {
                auto it2 = it1->second.find(current_word);
                if (it2 != it1->second.end()) {
                    for (const auto& [next_word, vote] : it2->second) {
                        accumulated_votes[next_word] += vote;
                    }
                }
            }
        }
        return accumulated_votes;
    }

    /**
     * @brief Obtém o vocabulário do modelo.
     * @return Referência constante ao conjunto de palavras do vocabulário.
     */
    const std::set<std::string>& getVocabulary() const {
        return vocabulary;
    }
};

/**
 * @brief Exibe as predições ordenadas por peso.
 * @param predictions Mapa de palavras candidatas e seus pesos.
 */
void displayPredictions(const std::map<std::string, double>& predictions) {
    std::multimap<double, std::string, std::greater<double>> sorted_predictions;
    for (const auto& [word, vote] : predictions) {
        sorted_predictions.insert({vote, word});
    }
    if (sorted_predictions.empty()) {
        std::cout << "  (Nenhuma previsão gerada)\n";
    } else {
        for (const auto& [vote, word] : sorted_predictions) {
            std::cout << "  " << word << ": " << std::fixed << std::setprecision(2) << vote << "\n";
        }
    }
}

/**
 * @brief Função principal que demonstra o uso da classe PairwiseFeatureAggregator com mascaramento.
 *
 * Este programa cria um modelo de predição de palavras com base em pares com saltos, treina o modelo
 * com duas sequências representando verificações de log de um programa e uma bateria, normaliza os
 * pesos, e testa a predição da próxima palavra para uma sequência de teste com e sem máscaras de
 * palavras anteriores relevantes.
 *
 * @return 0 em caso de execução bem-sucedida.
 */
int main() {
    // Criar o modelo
    PairwiseFeatureAggregator model; ///< Instância do modelo de agregação de características.

    // Definir sequências de treinamento
    std::vector<std::string> sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado", "por", "favor"
    };
    std::vector<std::string> sequence2 = {
        "verifique", "o", "log", "da", "bateria", "e", "descubra", "se", "ele", "acabou", "por", "favor"
    };

    // Adicionar sequências com pesos iguais (50% cada)
    model.addSequence(sequence1, 0.5);
    model.addSequence(sequence2, 0.5);

    // Normalizar os pesos
    model.normalizeVotes();

    // Definir sequência de teste
    std::vector<std::string> test_sequence = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado"
    };

    // Predição sem máscara (considerando todas as palavras anteriores)
    std::cout << "Predição SEM máscara explícita (considerando todos os pares):\n";
    std::set<std::string> all_prev_words;
    for (size_t i = 0; i < test_sequence.size() - 1; ++i) {
        all_prev_words.insert(test_sequence[i]);
    }
    auto predictions_no_mask = model.predictNextWordWithMask(test_sequence, all_prev_words);
    displayPredictions(predictions_no_mask);

    // Predição com máscara (focando apenas em "programa")
    std::cout << "\nPredição COM máscara (focando apenas em 'programa' como palavra anterior relevante):\n";
    std::set<std::string> mask_relevant = {"programa"};
    auto predictions_with_mask = model.predictNextWordWithMask(test_sequence, mask_relevant);
    displayPredictions(predictions_with_mask);

    // Predição com máscara (focando apenas em "bateria")
    std::cout << "\nPredição COM máscara (focando apenas em 'bateria'):\n";
    std::set<std::string> mask_relevant_bateria = {"bateria"};
    auto predictions_with_mask_bateria = model.predictNextWordWithMask(test_sequence, mask_relevant_bateria);
    displayPredictions(predictions_with_mask_bateria);

    return 0;
}
```

A implementação em C++ ilustra como o mascaramento pode ser aplicado para focar apenas nas palavras relevantes, permitindo que o modelo faça previsões mais precisas. O código é modular e pode ser facilmente adaptado para diferentes sequências e máscaras.

A atenta leitora já deve ter entendido a necessidade da atenção e a intuição que suporta esta ideia: focar seletivamente no que é relevante usando mascaramento/ponderação. A próxima pergunta que a curiosa leitora precisa fazer é: *como esse processo de seleção e ponderação é implementado de forma eficiente e, crucialmente, aprendido pelos modelos?*

Aqui despontam as operações matriciais que definem a atenção nos **Transformers**.

### Atenção como Multiplicação de Matrizes: Aprendendo a Focar

Vamos considerar que esperta leitora já entendeu a intuição da atenção como um mecanismo de foco seletivo, usando mascaramento ou ponderação para destacar informações relevantes. Precisamos tentar encontrar uma forma de implementar isso de forma eficiente e, crucialmente, que permita ao modelo *aprender* quais informações são relevantes em cada contexto. Isso quer dizer que: para ser eficiente, a máscara não pode ser fixa. A máscara precisa ser criada de acordo com o contexto atual da palavra para a qual estamos tentando prever a próxima palavra e com o contexto das palavras que vieram antes.

Para que os modelos possam aprender esses padrões de atenção e para que o cálculo seja eficiente em hardware moderno, como GPUs e TPUs, buscamos expressar todo o processo através de **operações de matrizes diferenciáveis**. Isso permite que usemos algoritmos como a retropropagação (_backpropagation_) para ajustar os pesos do modelo.

Trabalhamos a intuição da atenção como sendo um mecanismo de foco seletivo, usando mascaramento ou ponderação para destacar informações relevantes. Para que os modelos possam *aprender* esses padrões de atenção e para que o cálculo seja eficiente em hardware moderno, como GPUs e TPUs, buscamos expressar todo o processo através de **operações de matrizes diferenciáveis**. Isso permite que usemos algoritmos como a retropropagação (_backpropagation_) para ajustar os pesos do modelo.

A ideia central agora é substituir o processo de aplicação de máscara que discutimos, que funciona como uma `tabela de consulta` implícita para selecionar relevância na forma de um cálculo matricial. Em vez de apenas selecionar características, vamos calcular um **peso de atenção** para cada palavra anterior em relação à palavra atual. Esse peso determinará quanta atenção a palavra atual deve dar a cada palavra anterior ao construir seu vetor de contexto.

O processo geralmente envolve três componentes principais, derivados da representação vetorial (embedding) de cada palavra na sequência:

1. **Query (Consulta - Q)**: um vetor que representa a palavra/posição atual, atuando como uma "sonda" para buscar informações relevantes.
2. **Key (Chave - K)**: um vetor associado a cada palavra na sequência (incluindo as anteriores), que pode ser "comparado" com a Query para determinar a relevância.
3. **Value (Valor - V)**: um vetor associado a cada palavra na sequência, contendo a informação que será efetivamente passada adiante se a palavra for considerada relevante.

A relevância entre uma Query (palavra atual $t$) e uma Key (palavra anterior $i$) é calculada medindo a **similaridade** entre $Q_t$ e $K_i$. Uma forma comum e eficiente de fazer isso é através do **produto escalar (dot product)**. Podemos calcular todos os scores de similaridade para a palavra $t$ em relação a todas as palavras anteriores $i$ (e a própria $t$) de uma vez só usando multiplicação de matrizes:

$$ \text{Scores}_t = Q_t \cdot K^T $$

Onde $Q_t$ é o vetor query da palavra $t$, e $K$ é uma matriz onde cada linha $K_i$ é o vetor chave da palavra $i$. O resultado $\text{Scores}_t$ é um vetor onde cada elemento $j$ representa a similaridade bruta entre a query $t$ e a chave $j$.

![Consulta de máscara por multiplicação de matrizes](/assets/images/mask-query-matrix-multiplication-fixed.webp)
_Figura 5: Processo conceitual de consulta de atenção. A Query (Q) da palavra atual interage com as Keys (K) das palavras anteriores (e da atual) para gerar scores de atenção. (Nota: A figura original ilustrava uma busca de máscara; aqui reinterpretamos como cálculo de scores QK^T)._{: class="legend"}

Esses scores brutos precisam ser normalizados para se tornarem pesos de atenção que somam 1. Isso é feito aplicando a função **softmax**. Além disso, no artigo original do Transformer, os scores são escalonados por $\sqrt{d_k}$ (onde $d_k$ é a dimensão dos vetores Key/Query) antes do softmax para estabilizar os gradientes durante o treinamento:

$$ \text{AttentionWeights}_t = \text{softmax}\left( \frac{Q_t \cdot K^T}{\sqrt{d_k}} \right) $$

O resultado $\text{AttentionWeights}_t$ é um vetor de pesos, onde cada peso $\alpha_{ti}$ indica quanta atenção a palavra $t$ deve prestar à palavra $i$.

Finalmente, o **vetor de contexto** para a palavra $t$, $C_t$, é calculado como uma **soma ponderada** dos vetores Value ($V$) de todas as palavras, usando os pesos de atenção calculados:

$$ C_t = \sum_{i} \alpha_{ti} V_i $$

Este processo inteiro pode ser expresso de forma compacta para todas as palavras de uma sequência simultaneamente usando matrizes $Q$, $K$, $V$ (onde cada linha representa uma palavra):

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

![Equação de atenção destacando QKT](/assets/images/attention-equation-visualization.webp)
_Figura 6: A equação de atenção completa. O termo $QK^T$ calcula a similaridade, o softmax normaliza em pesos, e estes ponderam os vetores Value (V)._{: class="legend"}

Crucialmente, as matrizes $Q$, $K$, e $V$ não são os embeddings originais das palavras. Elas são obtidas aplicando **transformações lineares aprendíveis** (matrizes de pesos $W_Q, W_K, W_V$) aos embeddings de entrada. Isso permite que o modelo aprenda *quais aspectos* das palavras são relevantes para atuar como query, key ou value em diferentes contextos.

Este mecanismo de atenção é poderoso porque:

* Captura dependências independentemente da distância.
* Os cálculos são paralelizáveis sobre a sequência (ao contrário das RNNs).
* O modelo aprende a determinar as relações de relevância dinamicamente.

### Processando o Contexto Ponderado: A Rede Feed-Forward

Após o mecanismo de atenção calcular o vetor de contexto $C_t$ para cada palavra $t$ (que agora contém informação da própria palavra $t$ misturada com informações ponderadas de outras palavras relevantes na sequência), precisamos processar essa rica representação contextual.

O objetivo é transformar $C_t$ em uma saída que possa ser usada para a tarefa final (como prever a próxima palavra) ou que sirva como entrada para a próxima camada do modelo Transformer. Essa transformação é realizada por uma **Rede Neural Feed-Forward (FFN)**, aplicada independentemente a cada posição $t$ da sequência.

Embora tenhamos usado a analogia de "características de pares" na seção anterior, a FFN nos**Transformers**é mais genérica e poderosa. Tipicamente, ela consiste em duas camadas lineares com uma função de ativação não-linear entre elas, como ReLU (Rectified Linear Unit) ou GeLU (Gaussian Error Linear Unit):

$$ \text{FFN}(C_t) = \text{ReLU}(C_t W_1 + b_1) W_2 + b_2 $$

Onde $W_1, b_1, W_2, b_2$ são matrizes de pesos e vetores de bias aprendíveis. A primeira camada geralmente expande a dimensão do vetor $C_t$, e a segunda camada a projeta de volta para a dimensão original esperada pela próxima camada ou pela saída do modelo.

![Diagrama da camada de rede neural](/assets//images/ffn-layer-diagram.webp)
_Figura 7: Diagrama conceitual de uma camada de rede neural. A FFN nos**Transformers**aplica transformações semelhantes (lineares + não-linearidade) ao vetor de contexto de cada posição._{: class="legend"}

A não-linearidade (ReLU/GeLU) é crucial, pois permite que a FFN aprenda transformações complexas e não apenas combinações lineares das informações presentes no vetor de contexto $C_t$. Embora possamos *imaginar* que a FFN poderia aprender a detectar combinações específicas como "bateria, executado" (como no exemplo manual abaixo), na prática ela aprende representações mais abstratas e úteis para a tarefa.

![Cálculo da característica](/assets/images/feature-calculation-diagram.webp)
_Figura 8: Ilustração de como uma camada linear (multiplicação por matriz de pesos W) pode, em princípio, ser configurada para detectar a presença/ausência de certas combinações no vetor de entrada (que seria o $C_t$ após a atenção). A ReLU subsequente ajudaria a "ativar" essas características detectadas._{: class="legend"}

O código C++ abaixo demonstra a aplicação de uma camada linear seguida por ReLU, como parte de uma FFN:

```cpp
#include <iostream>         ///< Para entrada e saída padrão (std::cout).
#include <vector>          ///< Para contêiner std::vector usado no armazenamento de nomes de features.
#include <string>          ///< Para std::string, usado em nomes de features e mensagens.
#include <unordered_map>   ///< Para std::unordered_map (não utilizado diretamente neste código, mantido por compatibilidade).
#include <Eigen/Dense>     ///< Para a biblioteca Eigen, usada em operações com matrizes e vetores.
#include <cmath>           ///< Para std::max, usado na função ReLU.

/**
 * @brief Aplica a função ReLU (Rectified Linear Unit) a um único valor.
 * @param x O valor de entrada.
 * @return O valor após a aplicação de ReLU, max(0, x) (double).
 */
double relu(double x) {
    return std::max(0.0, x);
}

/**
 * @brief Aplica a função ReLU elemento a elemento em um vetor Eigen.
 * @param vec O vetor de entrada (Eigen::VectorXd).
 * @return Um novo vetor com a função ReLU aplicada a cada elemento (Eigen::VectorXd).
 */
Eigen::VectorXd reluVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd result = vec;
    for (int i = 0; i < result.size(); ++i) {
        result(i) = relu(result(i));
    }
    return result;
}

/**
 * @class FeedForwardLayer
 * @brief Uma classe que implementa uma camada feedforward com ativação ReLU.
 *
 * Esta classe simula uma camada de uma rede neural feedforward (FFN), aplicando uma transformação
 * linear (W * x + b) seguida de uma ativação ReLU. Utiliza a biblioteca Eigen para operações matriciais
 * e vetoriais.
 */
class FeedForwardLayer {
private:
    Eigen::MatrixXd weights;        ///< Matriz de pesos (W) da camada.
    Eigen::VectorXd bias;           ///< Vetor de bias (b) da camada.
    std::vector<std::string> featureNames; ///< Nomes hipotéticos das features na saída.

public:
    /**
     * @brief Construtor que inicializa a camada com pesos, bias e nomes de features.
     * @param w Matriz de pesos (Eigen::MatrixXd).
     * @param b Vetor de bias (Eigen::VectorXd).
     * @param names Vetor de strings contendo nomes das features de saída.
     */
    FeedForwardLayer(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, const std::vector<std::string>& names)
        : weights(w), bias(b), featureNames(names) {}

    /**
     * @brief Aplica a transformação da camada (linear + ReLU) a um vetor de entrada.
     * @param input O vetor de entrada (Eigen::VectorXd).
     * @return O vetor de saída após a transformação linear e ativação ReLU (Eigen::VectorXd).
     */
    Eigen::VectorXd forward(const Eigen::VectorXd& input) const {
        Eigen::VectorXd hiddenOutput = weights * input + bias;
        return reluVector(hiddenOutput);
    }

    /**
     * @brief Exibe as features ativadas na saída da camada.
     * @param output O vetor de saída após a transformação (Eigen::VectorXd).
     */
    void printActivatedFeatures(const Eigen::VectorXd& output) const {
        std::cout << "Features Ativadas (após ReLU) com nomes hipotéticos:\n";
        for (size_t i = 0; i < output.size(); ++i) {
            if (output(i) > 0) {
                std::string name = (i < featureNames.size()) ? featureNames[i] : "Desconhecida";
                std::cout << "  " << name << ": " << output(i) << "\n";
            }
        }
    }
};

/**
 * @brief Função principal que demonstra o uso da classe FeedForwardLayer.
 *
 * Este programa simula uma camada feedforward com ativação ReLU, utilizando um vetor de contexto
 * hipotético como entrada. A camada é configurada com pesos manuais para detectar combinações de
 * features como "bateria", "executado", e outras, e os resultados são exibidos para análise.
 *
 * @return 0 em caso de execução bem-sucedida.
 */
int main() {
    // Definir dimensões
    int context_dim = 4;  ///< Dimensão do vetor de contexto (entrada).
    int hidden_dim = 8;   ///< Dimensão da camada oculta (saída).

    // Simular vetor de contexto C_t
    Eigen::VectorXd context_vector = Eigen::VectorXd::Zero(context_dim);
    context_vector(0) = 1.0; // Feature "bateria" ativa
    context_vector(1) = 0.0; // Feature "programa" inativa
    context_vector(2) = 1.0; // Feature "executado" ativa
    context_vector(3) = 1.0; // Termo de bias

    // Definir pesos e bias da camada
    Eigen::MatrixXd W1(hidden_dim, context_dim);
    Eigen::VectorXd b1(hidden_dim);
    W1.setZero();
    b1.setZero();

    // Configurar pesos manuais
    W1.row(0) << 1.0, 0.0, 0.0, 0.0; // Detecta "bateria"
    W1.row(1) << 0.0, 1.0, 0.0, 0.0; // Detecta "programa"
    W1.row(2) << 0.0, 0.0, 1.0, 0.0; // Detecta "executado"
    W1.row(3) << 0.0, 0.0, 0.0, 1.0; // Detecta bias
    W1.row(6) << 1.0, 0.0, 1.0, -1.0; // Detecta "bateria" E "executado"
    W1.row(7) << 0.0, 1.0, 1.0, -1.0; // Detecta "programa" E "executado"

    // Definir nomes hipotéticos para as features
    std::vector<std::string> hidden_feature_names = {
        "feat_bateria", "feat_programa", "feat_executado", "feat_bias",
        "?", "?", "feat_bat_exec", "feat_prog_exec"
    };

    // Criar a camada feedforward
    FeedForwardLayer layer(W1, b1, hidden_feature_names); ///< Instância da camada FFN.

    // Aplicar a transformação
    Eigen::VectorXd hidden_output = W1 * context_vector + b1;
    Eigen::VectorXd activated_output = layer.forward(context_vector);

    // Exibir resultados
    std::cout << "Vetor de Contexto (Input C_t):\n" << context_vector.transpose() << "\n\n";
    std::cout << "Saída da Camada Linear 1 (Antes de ReLU):\n" << hidden_output.transpose() << "\n\n";
    std::cout << "Saída após ReLU (Input para Camada Linear 2):\n" << activated_output.transpose() << "\n\n";

    // Exibir features ativadas
    layer.printActivatedFeatures(activated_output);

    return 0;
}
```

Portanto, um bloco típico de um Transformer consiste na aplicação do mecanismo de **auto-atenção** (para calcular o vetor de contexto $C_t$ para cada posição $t$, olhando para toda a sequência) seguido pela aplicação da **Rede Feed-Forward** (para processar cada $C_t$ independentemente). Frequentemente, conexões residuais e normalização de camada (Layer Normalization) são adicionadas em torno desses dois sub-blocos para facilitar o treinamento de redes profundas.

### Conclusão e Perspectivas

Nesta jornada através da modelagem de sequências, partimos das Cadeias de Markov (modelos **N-gram**), reconhecendo sua simplicidade mas também suas limitações inerentes na captura de contexto de longo alcance. Vimos como a ideia conceitual de "pares com saltos" e "votação" nos levou à necessidade de um foco seletivo, que materializamos na intuição do **mascaramento** e da **atenção seletiva**.

Crucialmente, observamos como esse mecanismo de atenção pode ser implementado de forma eficiente e aprendível usando **operações matriciais** ($Q, K, V$ e a equação de atenção), permitindo ao modelo ponderar dinamicamente a relevância de diferentes partes da sequência. Finalmente, vimos como o vetor de contexto resultante é processado por uma **Rede Feed-Forward (FFN)**, completando os dois componentes principais de um bloco Transformer.

A perspicaz leitora percebeu que construímos os fundamentos conceituais que justificam a arquitetura proposta em "Attention is All You Need". Os**Transformers**abandonaram a recorrência das RNNs/LSTMs em favor da atenção paralelizável, permitindo treinar modelos muito maiores em mais dados e alcançando resultados estado-da-arte em inúmeras tarefas de Processamento de Linguagem Natural.

Claro, há mais detalhes na arquitetura completa do Transformer que não cobrimos aqui. No próximo artigo, pretendemos mergulhar mais fundo:

* **Atenção Multi-Cabeça (Multi-Head Attention)**: Como o modelo aprende a prestar atenção a diferentes aspectos da sequência simultaneamente.
* **Codificação Posicional (Positional Encoding)**: Como a informação sobre a ordem das palavras, perdida pela atenção que trata a sequência como um conjunto, é reintroduzida.
* **Arquitetura Completa Encoder-Decoder**: Como esses blocos são empilhados e combinados para tarefas como tradução automática.
* **Aplicações e Variações**: Uma visão geral do impacto dos**Transformers**e modelos derivados (BERT, GPT, etc.).

Os conceitos que exploramos – modelagem sequencial, captura de contexto, e atenção seletiva – formam a base não apenas dos Transformers, mas de grande parte da pesquisa atual em inteligência artificial. Compreendê-los é essencial para navegar neste campo fascinante.

## Referências Bibliográficas

BAHDANAU, D.; CHO, K.; BENGIO, Y. **Neural machine translation by jointly learning to align and translate**. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego. Proceedings [...]. San Diego: ICLR, 2015.

BENGIO, Y. et al. **A neural probabilistic language model**. Journal of Machine Learning Research, v. 3, p. 1137-1155, 2003.

BROWN, P. F. et al. **Class-based **N-gram** models of natural language**. Computational Linguistics, v. 18, n. 4, p. 467-479, 1992.

CHOMSKY, N. **Three models for the description of language**. IRE Transactions on Information Theory, v. 2, n. 3, p. 113-124, 1956.

DEVLIN, J. et al. BERT: **pre-training of deep bidirectional transformers for language understanding**. In: CONFERENCE OF THE NORTH AMERICAN CHAPTER OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. Proceedings [...]. Minneapolis: ACL, 2019. p. 4171-4186.

GRAVES, A. **Supervised sequence labelling with recurrent neural networks**. Heidelberg: Springer, 2012. (Studies in Computational Intelligence, v. 385).

HOCHREITER, S.; SCHMIDHUBER, J. **Long short-term memory**. Neural Computation, v. 9, n. 8, p. 1735-1780, 1997.

KINGMA, D. P.; BA, J. Adam: **a method for stochastic optimization**. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego. Proceedings [...]. San Diego: ICLR, 2015.

LUONG, M. T.; PHAM, H.; MANNING, C. D. **Effective approaches to attention-based neural machine translation**. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING, 2015, Lisbon. Proceedings [...]. Lisbon: ACL, 2015. p. 1412-1421.

MIKOLOV, T. et al. **Distributed representations of words and phrases and their compositionality**. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 26., 2013, Lake Tahoe. Proceedings [...]. Lake Tahoe: NIPS, 2013. p. 3111-3119.

RABINER, L. R. **A tutorial on hidden Markov models and selected applications in speech recognition**. Proceedings of the IEEE, v. 77, n. 2, p. 257-286, 1989.

SUTSKEVER, I.; VINYALS, O.; LE, Q. V. **Sequence to sequence learning with neural networks**. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 27., 2014, Montreal. Proceedings [...]. Montreal: NIPS, 2014. p. 3104-3112.

VASWANI, A. et al. **Attention is all you need**. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. Proceedings [...]. Long Beach: NIPS, 2017. p. 5998-6008.