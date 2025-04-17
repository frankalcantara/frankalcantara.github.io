---
layout: post
title: Transformers- Prestando Atenção
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
image: assets/images/multimatriz1.webp
featured: false
rating: 5
description: ""
date: 2025-02-10T22:55:34.524Z
preview: Neste artigo, mergulhamos na modelagem de sequências textuais. Partimos das Cadeias de Markov, N-grams, e suas limitações, construindo gradualmente a intuição para modelos mais sofisticados capazes de capturar dependências de longo alcance, fundamentais para a arquitetura Transformer.
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
lastmod: 2025-04-17T18:49:21.287Z
---
## Superando Limitações Locais: Agregação de Características de Pares



Como vimos, aumentar a ordem $N$ nos modelos **N-gram**/Markovianos é uma estratégia limitada para capturar o contexto necessário em linguagem natural devido à maldição da dimensionalidade e à esparsidade dos dados. Precisamos de uma abordagem diferente para lidar com dependências que podem se estender por muitas palavras.

Imagine que nosso sistema agora precisa lidar com frases mais complexas, cada uma ocorrendo com igual probabilidade (50%):

* Verifique o log do **programa** e descubra se ele foi **executado**, por favor.
* Verifique o log da **bateria** e descubra se ele **acabou**, por favor.

Neste exemplo, para determinar se a palavra após "ele" deve ser "foi" (seguido de "executado") ou "acabou", precisamos olhar para trás e identificar se o log era do "programa" ou da "bateria". A palavra crucial ("programa" ou "bateria") está 8 posições antes da palavra que queremos prever ("foi" ou "acabou"). Um modelo de Markov de oitava ordem seria computacionalmente inviável ($N^8$ estados!).

Podemos, então, pensar em uma estratégia diferente, ainda inspirada na ideia de olhar pares de palavras (como na segunda ordem), mas com mais flexibilidade. E se, para prever a palavra seguinte à palavra atual $w_t$, considerássemos não apenas o par $(w_{t-1}, w_t)$, mas *todos* os pares $(w_i, w_t)$ onde $w_i$ é qualquer palavra que apareceu *antes* de $w_t$ na sequência ($i < t$)? Isso nos permite "saltar" sobre palavras intermediárias e potencialmente capturar relações de longo alcance.

Essa mudança conceitual implica uma reinterpretação fundamental da nossa "matriz de transição". As linhas da matriz não representam mais um estado probabilístico (o contexto imediato $w_{t-1}$ ou $[w_{t-2}, w_{t-1}]$) onde as probabilidades de transição devem somar 1. Em vez disso, cada linha pode ser vista como uma **característica (feature)** definida por um par específico $(w_i, w_t)$ que ocorreu na sequência. O valor na coluna $j$ dessa linha passa a representar um **voto** ou **peso** que essa característica específica atribui à palavra $w_j$ como sendo a próxima palavra ($w_{t+1}$).

![Votação de características de pares com saltos](assets/images/second-order-skip1.webp)
_Figura 5: Modelo conceitual baseado em pares com saltos. As linhas representam características (pares como `(programa, executado)` ou `(bateria, executado)`). Os valores são "votos" para a próxima palavra (ex: "por"). Apenas pesos não-zero relevantes são mostrados._{: class="legend"}

A Figura 5 ilustra isso para prever a palavra após "executado" (no primeiro exemplo de frase). Várias características (pares formados por "executado" e palavras anteriores como "verifique", "o", "log", "do", "programa", etc.) estão ativas. Cada uma delas "vota" nas possíveis próximas palavras.

Neste modelo conceitual:

* Características como `(programa, executado)` dariam um voto forte para "por".
* Características como `(bateria, executado)` (que não ocorreria na primeira frase) dariam voto zero para "por".
* Características menos informativas, como `(o, executado)` ou `(log, executado)`, podem ter votos distribuídos ou votos mais fracos (ex: 0.5 para "por", 0.5 para "favor" - indicando que apenas ver "o" ou "log" antes de "executado" não ajuda muito a decidir a próxima palavra específica *neste contexto*).

Para fazer uma previsão, somamos os votos de *todas* as características ativas (pares formados pela palavra atual e *todas* as palavras anteriores na sequência específica) para cada palavra candidata a ser a próxima. A palavra com a maior soma de votos é a escolhida.

No exemplo "Verifique o log do programa e descubra se ele foi executado":

* As características ativas relevantes seriam `(verifique, executado)`, `(o, executado)`, `(log, executado)`, `(do, executado)`, `(programa, executado)`, `(e, executado)`, `(descubra, executado)`, `(se, executado)`, `(ele, executado)`, `(foi, executado)`.
* Se somarmos os votos hipotéticos (baseados na Figura 5, onde pares irrelevantes podem ter voto 0.5 e `(programa, executado)` tem voto 1 para "por"), a palavra "por" acumularia um total de votos mais alto (ex: 5 votos, como no texto original) comparado a outras palavras, tornando-se a previsão correta.

Esta abordagem, embora ainda baseada em pares, oferece uma maneira de incorporar contexto de longo alcance de forma seletiva. A implementação pode usar estruturas de dados semelhantes às do modelo de segunda ordem, mas a lógica de treinamento e predição muda para refletir a soma de "votos" de múltiplos pares ativos.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map> // Para ordenar a saída
#include <set> // Para vocabulário
#include <iomanip>

// Classe para implementar o modelo conceitual baseado em pares com saltos
// (Renomeado de SkipGramMarkovModel para clareza)
class PairwiseFeatureAggregator {
private:
    // transitions[palavra_anterior][palavra_atual] -> mapa{proxima_palavra: voto/peso}
    // Note: Estes "votos" são aprendidos/contados; a normalização pode variar dependendo da estratégia.
    // Aqui, vamos manter a lógica original de normalizar por par (w_prev, w_curr) como se fosse prob.
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          std::unordered_map<std::string, double>>> transitions;
    std::set<std::string> vocabulary;

public:
    // Adicionar uma sequência ao modelo (contagem de ocorrências)
    // O 'weight' pode ser usado para dar mais importância a certas sequências
    void addSequence(const std::vector<std::string>& sequence, double weight = 1.0) {
        if (sequence.size() < 3) return; // Precisamos de pelo menos (prev, current, next)

        // Adicionar todas as palavras ao vocabulário
        for (const auto& word : sequence) {
            vocabulary.insert(word);
        }

        // Para cada posição 'current_pos', considerar (current_word, next_word)
        // e associar com *todas* as palavras anteriores 'prev_word'.
        for (size_t current_pos = 1; current_pos < sequence.size() - 1; ++current_pos) {
            const std::string& current_word = sequence[current_pos];
            const std::string& next_word = sequence[current_pos + 1];

            // Iterar sobre todas as palavras anteriores a 'current_pos'
            for (size_t prev_pos = 0; prev_pos < current_pos; ++prev_pos) {
                const std::string& prev_word = sequence[prev_pos];
                // Acumula o 'voto' da característica (prev_word, current_word) para a next_word
                transitions[prev_word][current_word][next_word] += weight;
            }
        }
    }

    // Normalizar os 'votos' (opcional, pode ser interpretado como probabilidade P(next | prev, current))
    // Mantendo a lógica original do artigo: normaliza para cada par (prev, current).
    void normalizeVotes() {
        for (auto& [prev_word, inner_map1] : transitions) {
            for (auto& [current_word, inner_map2] : inner_map1) {
                double total_votes = 0.0;
                for (auto const& [next_word, vote] : inner_map2) {
                    total_votes += vote;
                }
                if (total_votes > 0) {
                    for (auto& [next_word, vote] : inner_map2) {
                        inner_map2[next_word] /= total_votes; // Normaliza
                    }
                }
            }
        }
    }


    // Prever a próxima palavra somando os votos de todos os pares relevantes na sequência atual
    std::map<std::string, double> predictNextWord(const std::vector<std::string>& sequence) const {
        if (sequence.empty()) return {};

        std::map<std::string, double> accumulated_votes; // Usar map para ordenar a saída
        const std::string& current_word = sequence.back(); // A última palavra da sequência dada

        // Considerar todos os pares (prev_word, current_word) formados com palavras anteriores na sequência
        for (size_t i = 0; i < sequence.size() - 1; ++i) {
            const std::string& prev_word = sequence[i];

            // Verificar se temos 'votos' registrados para este par (prev_word, current_word)
            auto it1 = transitions.find(prev_word);
            if (it1 != transitions.end()) {
                auto it2 = it1->second.find(current_word);
                if (it2 != it1->second.end()) {
                    // Acumular os votos para cada possível próxima palavra
                    for (const auto& [next_word, vote] : it2->second) {
                        accumulated_votes[next_word] += vote;
                    }
                }
            }
        }

        return accumulated_votes;
    }

    // Obter o vocabulário do modelo
    const std::set<std::string>& getVocabulary() const {
        return vocabulary;
    }
};

int main() {
    // Criar o modelo
    PairwiseFeatureAggregator model;

    // Adicionar sequências de treinamento
    std::vector<std::string> sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado", "por", "favor"
    };
    std::vector<std::string> sequence2 = {
        "verifique", "o", "log", "da", "bateria", "e", "descubra", "se", "ele", "acabou", "por", "favor"
    };

    // Adicionar com pesos iguais (50% frequência)
    model.addSequence(sequence1, 0.5);
    model.addSequence(sequence2, 0.5);

    // Normalizar os 'votos' (interpretando como P(next | prev, current) aqui)
    model.normalizeVotes();

    // Testar predições para a sequência 1 (terminando em "executado")
    std::vector<std::string> test_sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado"
    };

    std::cout << "Predição (soma de votos normalizados) após sequência 1 (termina com 'executado'):\n";
    auto predictions1 = model.predictNextWord(test_sequence1);

    // Ordenar os resultados por valor (voto acumulado), decrescente
    std::multimap<double, std::string, std::greater<double>> sorted_predictions1;
    for (const auto& [word, vote] : predictions1) {
        sorted_predictions1.insert({vote, word});
    }

    for (const auto& [vote, word] : sorted_predictions1) {
        std::cout << "  " << word << ": " << std::fixed << std::setprecision(2) << vote << "\n";
    }

     // Testar predições para a sequência 2 (terminando em "acabou")
     // (Para fins de demonstração, usaríamos uma sequência de teste terminando em 'acabou')
     // std::vector<std::string> test_sequence2 = { ... "acabou" };
     // auto predictions2 = model.predictNextWord(test_sequence2);
     // ... exibir predictions2 ...

    return 0;
}
```

Embora esta abordagem de "pares com saltos" e "votação" nos permita considerar contexto de longo alcance, ela introduz um novo desafio. Ao somar votos de *muitas* características (pares), a contribuição das poucas características *realmente* informativas (como `(programa, executado)` no nosso exemplo) pode ser diluída pelo "ruído" das características menos úteis (como `(o, executado)`). A diferença entre o total de votos para a palavra correta e as incorretas pode ser pequena, tornando o modelo menos confiante e robusto.

Como podemos aprimorar isso? Precisamos de uma forma de fazer o modelo focar dinamicamente nas características (pares) que são mais relevantes para a previsão atual.

### Mascaramento e Atenção Seletiva: Focando no que Importa

A solução para a diluição dos votos é introduzir um mecanismo que permita ao modelo **prestar atenção seletiva** às características mais informativas, ignorando ou diminuindo o peso das demais. Podemos fazer isso através de **mascaramento**.

Imagine que, para a tarefa de prever a palavra após "executado" no contexto da frase sobre o "programa", pudéssemos saber (ou aprender) que as características mais importantes são aquelas que envolvem "programa" e talvez "foi", enquanto outras como "verifique", "o", "log", etc., são menos preditivas *neste ponto específico*.

Podemos criar uma **máscara**, que é essencialmente um vetor ou conjunto de pesos, um para cada característica (ou para cada palavra anterior que forma um par com a palavra atual). A máscara atribuiria peso 1 (ou um peso alto) às características/palavras anteriores consideradas importantes e peso 0 (ou um peso baixo) às demais.

![Atividades de características mascaradas](assets/images/masked-features1.webp)
_Figura 6: Aplicação de uma máscara conceitual. A máscara (à direita) atribui peso 1 para "programa" e "bateria" (assumindo que estas são as palavras-chave distintivas) e 0 para as outras. Ao multiplicar os pesos das características ativas (centro) pela máscara, apenas as características relevantes ("programa, executado" neste caso) mantêm seu peso (resultado à esquerda)._{: class="legend"}

Para aplicar a máscara aos "votos" que calculamos na seção anterior, realizamos uma multiplicação elemento a elemento (produto Hadamard) entre o vetor de votos de cada possível palavra seguinte e a máscara apropriada, ou, de forma equivalente, aplicamos a máscara ao *contexto* antes de calcular os votos acumulados. Qualquer voto originado de uma característica/par mascarado (peso 0 na máscara) é zerado.

O efeito disso é como "apagar" temporariamente as partes menos relevantes da nossa matriz conceitual de votos, deixando apenas as conexões que realmente importam para a decisão atual.

![Matriz de transição mascarada](assets/images/masked-transition1.webp)
_Figura 7: Matriz de votos/transição conceitual após a aplicação da máscara. Apenas as linhas/características consideradas relevantes (ex: envolvendo "programa" ou "bateria") permanecem ativas, tornando a previsão muito mais direcionada._{: class="legend"}

Após aplicar a máscara, a soma dos votos torna-se muito mais decisiva. No nosso exemplo, se a máscara destacar apenas a característica `(programa, executado)`, o total de votos para "por" virá predominantemente (ou exclusivamente) dessa característica, enquanto os votos para outras palavras (ou vindos de outras características) serão zerados ou drasticamente reduzidos. A confiança do modelo na previsão correta aumenta significativamente.

**Este processo de focar seletivamente em partes específicas do input (as palavras anteriores mais relevantes, neste caso) para tomar uma decisão sobre o output (a próxima palavra) é a intuição fundamental por trás do mecanismo de ATENÇÃO.** O artigo seminal "Attention is All You Need" (Vaswani et al., 2017) introduziu uma forma específica e poderosa de implementar essa ideia, que se tornou a base dos modelos Transformer. O que descrevemos aqui é uma aproximação conceitual para construir o entendimento.

Vejamos como o mascaramento pode ser implementado em C++, aplicando a máscara *antes* de acumular os votos (demonstração conceitual):

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <iomanip>
#include <cmath> // Para std::max em ReLU (será usado depois)
// #include <Eigen/Dense> // Eigen será mais útil na próxima seção

// --- Supondo que a classe PairwiseFeatureAggregator foi definida acima ---
class PairwiseFeatureAggregator {
private:
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          std::unordered_map<std::string, double>>> transitions;
    std::set<std::string> vocabulary;
public:
    void addSequence(const std::vector<std::string>& sequence, double weight = 1.0) { /* ... implementação ... */
        if (sequence.size() < 3) return;
        for (const auto& word : sequence) { vocabulary.insert(word); }
        for (size_t current_pos = 1; current_pos < sequence.size() - 1; ++current_pos) {
            const std::string& current_word = sequence[current_pos];
            const std::string& next_word = sequence[current_pos + 1];
            for (size_t prev_pos = 0; prev_pos < current_pos; ++prev_pos) {
                const std::string& prev_word = sequence[prev_pos];
                transitions[prev_word][current_word][next_word] += weight;
            }
        }
    }
    void normalizeVotes() { /* ... implementação ... */
         for (auto& [prev_word, inner_map1] : transitions) {
            for (auto& [current_word, inner_map2] : inner_map1) {
                double total_votes = 0.0;
                for (auto const& [next_word, vote] : inner_map2) { total_votes += vote; }
                if (total_votes > 0) {
                    for (auto& [next_word, vote] : inner_map2) { inner_map2[next_word] /= total_votes; }
                }
            }
        }
    }
    // Versão modificada para aceitar uma máscara de palavras anteriores relevantes
    std::map<std::string, double> predictNextWordWithMask(
        const std::vector<std::string>& sequence,
        const std::set<std::string>& relevant_prev_words // Máscara: palavras anteriores a considerar
    ) const {
        if (sequence.empty()) return {};

        std::map<std::string, double> accumulated_votes;
        const std::string& current_word = sequence.back();

        for (size_t i = 0; i < sequence.size() - 1; ++i) {
            const std::string& prev_word = sequence[i];

            // >>> APLICAÇÃO DA MÁSCARA AQUI <<<
            // Só considera o par se a prev_word for relevante
            if (relevant_prev_words.count(prev_word) == 0) {
                continue; // Pula esta palavra anterior (mascarada)
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
     const std::set<std::string>& getVocabulary() const { return vocabulary; }
};


// Função auxiliar apenas para exibir mapas ordenados
void displayPredictions(const std::map<std::string, double>& predictions) {
    std::multimap<double, std::string, std::greater<double>> sorted_predictions;
    for (const auto& [word, vote] : predictions) {
        sorted_predictions.insert({vote, word});
    }
     if (sorted_predictions.empty()){
         std::cout << "  (Nenhuma previsão gerada)\n";
    } else {
        for (const auto& [vote, word] : sorted_predictions) {
            std::cout << "  " << word << ": " << std::fixed << std::setprecision(2) << vote << "\n";
        }
    }
}


// ----- Demonstração de Mascaramento (Continuação do main anterior) -----
int main() {
    // Criar e treinar o modelo
    PairwiseFeatureAggregator model;
    std::vector<std::string> sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado", "por", "favor"
    };
     std::vector<std::string> sequence2 = {
        "verifique", "o", "log", "da", "bateria", "e", "descubra", "se", "ele", "acabou", "por", "favor"
    };
    model.addSequence(sequence1, 0.5);
    model.addSequence(sequence2, 0.5);
    model.normalizeVotes(); // Normaliza os votos acumulados

    // Sequência de teste
    std::vector<std::string> test_sequence1 = {
        "verifique", "o", "log", "do", "programa", "e", "descubra", "se", "ele", "foi", "executado"
    };

    // Predição SEM máscara (conceitualmente, máscara = todas as palavras anteriores)
    // Para obter o resultado sem máscara explícita, precisaríamos da função original predictNextWord
    // Vamos simular isso chamando predictNextWordWithMask com todas as palavras anteriores como relevantes
     std::cout << "Predição SEM máscara explícita (considerando todos os pares):\n";
     std::set<std::string> all_prev_words;
     for(size_t i=0; i<test_sequence1.size()-1; ++i) all_prev_words.insert(test_sequence1[i]);
     auto predictions_no_mask = model.predictNextWordWithMask(test_sequence1, all_prev_words);
     displayPredictions(predictions_no_mask);


    // Predição COM máscara (focando apenas em 'programa' como palavra anterior relevante)
    std::cout << "\nPredição COM máscara (focando apenas em 'programa' como palavra anterior relevante):\n";
    std::set<std::string> mask_relevant = {"programa"}; // Máscara: só 'programa' importa
    auto predictions_with_mask = model.predictNextWordWithMask(test_sequence1, mask_relevant);
    displayPredictions(predictions_with_mask);

    // Predição COM máscara (focando apenas em 'bateria' - que não está na seq. de teste)
    std::cout << "\nPredição COM máscara (focando apenas em 'bateria'):\n";
    std::set<std::string> mask_relevant_bateria = {"bateria"};
    auto predictions_with_mask_bateria = model.predictNextWordWithMask(test_sequence1, mask_relevant_bateria);
    displayPredictions(predictions_with_mask_bateria); // Deve gerar pouca ou nenhuma previsão


    return 0;
}
```

Entendemos agora a *necessidade* da atenção (superar contexto local fixo) e a *intuição* por trás dela (focar seletivamente no que é relevante usando mascaramento/ponderação). A próxima pergunta natural é: como esse processo de seleção e ponderação é implementado de forma eficiente e, crucialmente, *aprendido* pelos modelos? É aqui que entram as operações matriciais que definem a atenção nos Transformers.

### Atenção como Multiplicação de Matrizes: Aprendendo a Focar

Entendemos a intuição da atenção como um mecanismo de foco seletivo, usando mascaramento ou ponderação para destacar informações relevantes. Mas como o modelo *aprende* qual máscara aplicar? A máscara não pode ser fixa; ela precisa depender do contexto atual – da palavra para a qual estamos tentando prever a próxima e das palavras que vieram antes.

Para que os modelos possam *aprender* esses padrões de atenção e para que o cálculo seja eficiente em hardware moderno (como GPUs), buscamos expressar todo o processo através de **operações de matrizes diferenciáveis**. Isso permite que usemos algoritmos como a retropropagação (backpropagation) para ajustar os pesos do modelo.

A ideia central é substituir a "tabela de consulta" implícita da máscara por um cálculo matricial. Em vez de apenas "selecionar" características, vamos calcular um **peso de atenção** para cada palavra anterior em relação à palavra atual. Esse peso determinará quanta "atenção" a palavra atual deve dar a cada palavra anterior ao construir seu vetor de contexto.

O processo geralmente envolve três componentes principais, derivados da representação vetorial (embedding) de cada palavra na sequência:

1.  **Query (Consulta - Q):** Um vetor que representa a palavra/posição atual, atuando como uma "sonda" para buscar informações relevantes.
2.  **Key (Chave - K):** Um vetor associado a cada palavra na sequência (incluindo as anteriores), que pode ser "comparado" com a Query para determinar a relevância.
3.  **Value (Valor - V):** Um vetor associado a cada palavra na sequência, contendo a informação que será efetivamente passada adiante se a palavra for considerada relevante.

A relevância entre uma Query (palavra atual $t$) e uma Key (palavra anterior $i$) é calculada medindo a **similaridade** entre $Q_t$ e $K_i$. Uma forma comum e eficiente de fazer isso é através do **produto escalar (dot product)**. Podemos calcular todos os scores de similaridade para a palavra $t$ em relação a todas as palavras anteriores $i$ (e a própria $t$) de uma vez só usando multiplicação de matrizes:

$$ \text{Scores}_t = Q_t \cdot K^T $$

Onde $Q_t$ é o vetor query da palavra $t$, e $K$ é uma matriz onde cada linha $K_i$ é o vetor chave da palavra $i$. O resultado $\text{Scores}_t$ é um vetor onde cada elemento $j$ representa a similaridade bruta entre a query $t$ e a chave $j$.

![Consulta de máscara por multiplicação de matrizes](assets/images/mask-query1.webp)
_Figura 8: Processo conceitual de consulta de atenção. A Query (Q) da palavra atual interage com as Keys (K) das palavras anteriores (e da atual) para gerar scores de atenção. (Nota: A figura original ilustrava uma busca de máscara; aqui reinterpretamos como cálculo de scores QK^T)._{: class="legend"}

Esses scores brutos precisam ser normalizados para se tornarem pesos de atenção que somam 1. Isso é feito aplicando a função **softmax**. Além disso, no artigo original do Transformer, os scores são escalonados por $\sqrt{d_k}$ (onde $d_k$ é a dimensão dos vetores Key/Query) antes do softmax para estabilizar os gradientes durante o treinamento:

$$ \text{AttentionWeights}_t = \text{softmax}\left( \frac{Q_t \cdot K^T}{\sqrt{d_k}} \right) $$

O resultado $\text{AttentionWeights}_t$ é um vetor de pesos, onde cada peso $\alpha_{ti}$ indica quanta atenção a palavra $t$ deve prestar à palavra $i$.

Finalmente, o **vetor de contexto** para a palavra $t$, $C_t$, é calculado como uma **soma ponderada** dos vetores Value ($V$) de todas as palavras, usando os pesos de atenção calculados:

$$ C_t = \sum_{i} \alpha_{ti} V_i $$

Este processo inteiro pode ser expresso de forma compacta para todas as palavras de uma sequência simultaneamente usando matrizes $Q$, $K$, $V$ (onde cada linha representa uma palavra):

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

![Equação de atenção destacando QKT](assets/images/attn-equation1.webp)
_Figura 9: A equação de atenção completa. O termo $QK^T$ calcula a similaridade, o softmax normaliza em pesos, e estes ponderam os vetores Value (V)._{: class="legend"}

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

![Diagrama da camada de rede neural](assets/images/nn-layer1.webp)
_Figura 10: Diagrama conceitual de uma camada de rede neural. A FFN nos**Transformers**aplica transformações semelhantes (lineares + não-linearidade) ao vetor de contexto de cada posição._{: class="legend"}

A não-linearidade (ReLU/GeLU) é crucial, pois permite que a FFN aprenda transformações complexas e não apenas combinações lineares das informações presentes no vetor de contexto $C_t$. Embora possamos *imaginar* que a FFN poderia aprender a detectar combinações específicas como "bateria, executado" (como no exemplo manual abaixo), na prática ela aprende representações mais abstratas e úteis para a tarefa.

![Cálculo da característica](assets/images/feature-calc1.webp)
_Figura 11: Ilustração de como uma camada linear (multiplicação por matriz de pesos W) pode, em princípio, ser configurada para detectar a presença/ausência de certas combinações no vetor de entrada (que seria o $C_t$ após a atenção). A ReLU subsequente ajudaria a "ativar" essas características detectadas._{: class="legend"}

O código C++ abaixo demonstra a aplicação de uma camada linear seguida por ReLU, como parte de uma FFN:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath> // Para std::max (ReLU)

// Função ReLU (Rectified Linear Unit) para um único valor
double relu(double x) {
    return std::max(0.0, x);
}

// Aplicar ReLU elemento a elemento em um vetor Eigen
Eigen::VectorXd reluVector(const Eigen::VectorXd& vec) {
    // Aplica a função relu a cada coeficiente do vetor.
    // Eigen >= 3.4 pode usar .unaryExpr(&relu) ou .cwiseMax(0)
    // Para compatibilidade mais ampla, podemos fazer um loop:
     Eigen::VectorXd result = vec;
     for (int i = 0; i < result.size(); ++i) {
         result(i) = relu(result(i));
     }
     return result;
     // Alternativa moderna (Eigen 3.3+): return vec.cwiseMax(0.0);
}

int main() {
    // Definir um vocabulário hipotético para o input (resultado da atenção)
    // Na prática, seria um vetor denso C_t, não one-hot.
    // Mas para ilustrar a FFN com pesos 'manuais', simulamos um input simples.
    // Dimensão do vetor de contexto C_t (ex: 4)
    int context_dim = 4;

    // Simular um vetor de contexto C_t (ex: representando "bateria", "executado", e um bias)
    Eigen::VectorXd context_vector = Eigen::VectorXd::Zero(context_dim);
    context_vector(0) = 1.0; // Feature "bateria" ativa
    context_vector(1) = 0.0; // Feature "programa" inativa
    context_vector(2) = 1.0; // Feature "executado" ativa
    context_vector(3) = 1.0; // Termo de bias

    // Definir a primeira camada linear da FFN: W1 (dim_saida x dim_entrada) e b1
    // Ex: expandir de dim 4 para dim 8
    int hidden_dim = 8;
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(hidden_dim, context_dim); // Pesos aleatórios (na prática, são aprendidos)
    Eigen::VectorXd b1 = Eigen::VectorXd::Random(hidden_dim);             // Bias aleatório

    // Simular pesos 'manuais' como no artigo original para intuição
    // (Sobrescrevendo os pesos aleatórios)
    W1.setZero(); // Zera a matriz para começar
    b1.setZero();
    // Recriar pesos da Figura 11 (adaptados para W1 que multiplica C_t):
    // Saída 0: detecta "bateria"
    W1.row(0) << 1, 0, 0, 0;
    // Saída 1: detecta "programa"
    W1.row(1) << 0, 1, 0, 0;
     // Saída 2: detecta "executado"
    W1.row(2) << 0, 0, 1, 0;
     // Saída 3: bias (sempre 1 se o bias de entrada for 1)
    W1.row(3) << 0, 0, 0, 1;
     // Saída 4: "bateria" E NÃO "programa" (exemplo mais complexo)
    // W1.row(4) << 1, -1, 0, 0; // Requereria pesos e talvez bias específicos
     // Saída 5: "programa" E NÃO "bateria"
    // W1.row(5) << -1, 1, 0, 0;
     // Saída 6: "bateria" E "executado"
    W1.row(6) << 1, 0, 1, -1; // Ex: ativa se bat=1, exec=1, bias=1 -> 1+1-1=1
    b1(6) = 0; // Ajustar bias se necessário
     // Saída 7: "programa" E "executado"
    W1.row(7) << 0, 1, 1, -1; // Ex: ativa se prog=1, exec=1, bias=1 -> 0+1+1-1=1
    b1(7) = 0;


    // Aplicar a primeira camada: W1 * C_t + b1
    Eigen::VectorXd hidden_output = W1 * context_vector + b1;

    // Aplicar a não-linearidade ReLU
    Eigen::VectorXd activated_output = reluVector(hidden_output);

    // (A segunda camada linear W2, b2 seria aplicada a activated_output depois)

    // Exibir resultados intermediários
    std::cout << "Vetor de Contexto (Input C_t):\n" << context_vector.transpose() << "\n\n";
    std::cout << "Saída da Camada Linear 1 (Antes de ReLU):\n" << hidden_output.transpose() << "\n\n";
    std::cout << "Saída após ReLU (Input para Camada Linear 2):\n" << activated_output.transpose() << "\n\n";

    // Nomes hipotéticos para as features na camada oculta
     std::vector<std::string> hidden_feature_names = {
         "feat_bateria", "feat_programa", "feat_executado", "feat_bias",
         "?", "?", "feat_bat_exec", "feat_prog_exec"
     };
     std::cout << "Features Ativadas (após ReLU) com nomes hipotéticos:\n";
     for (int i = 0; i < activated_output.size(); ++i) {
         if (activated_output(i) > 0) {
             std::cout << "  " << hidden_feature_names[i] << ": " << activated_output(i) << "\n";
         }
     }

    return 0;
}
```

Portanto, um bloco típico de um Transformer consiste na aplicação do mecanismo de **auto-atenção** (para calcular o vetor de contexto $C_t$ para cada posição $t$, olhando para toda a sequência) seguido pela aplicação da **Rede Feed-Forward** (para processar cada $C_t$ independentemente). Frequentemente, conexões residuais e normalização de camada (Layer Normalization) são adicionadas em torno desses dois sub-blocos para facilitar o treinamento de redes profundas.

### Conclusão e Perspectivas

Nesta jornada através da modelagem de sequências, partimos das Cadeias de Markov (modelos **N-gram**), reconhecendo sua simplicidade mas também suas limitações inerentes na captura de contexto de longo alcance. Vimos como a ideia conceitual de "pares com saltos" e "votação" nos levou à necessidade de um foco seletivo, que materializamos na intuição do **mascaramento** e da **atenção seletiva**.

Crucialmente, observamos como esse mecanismo de atenção pode ser implementado de forma eficiente e aprendível usando **operações matriciais** ($Q, K, V$ e a equação de atenção), permitindo ao modelo ponderar dinamicamente a relevância de diferentes partes da sequência. Finalmente, vimos como o vetor de contexto resultante é processado por uma **Rede Feed-Forward (FFN)**, completando os dois componentes principais de um bloco Transformer.

A perspicaz leitora percebeu que construímos os fundamentos conceituais que justificam a arquitetura proposta em "Attention is All You Need". Os**Transformers**abandonaram a recorrência das RNNs/LSTMs em favor da atenção paralelizável, permitindo treinar modelos muito maiores em mais dados e alcançando resultados estado-da-arte em inúmeras tarefas de Processamento de Linguagem Natural.

Claro, há mais detalhes na arquitetura completa do Transformer que não cobrimos aqui. No próximo artigo, pretendemos mergulhar mais fundo:

* **Atenção Multi-Cabeça (Multi-Head Attention):** Como o modelo aprende a prestar atenção a diferentes aspectos da sequência simultaneamente.
* **Codificação Posicional (Positional Encoding):** Como a informação sobre a ordem das palavras, perdida pela atenção que trata a sequência como um conjunto, é reintroduzida.
* **Arquitetura Completa Encoder-Decoder:** Como esses blocos são empilhados e combinados para tarefas como tradução automática.
* **Aplicações e Variações:** Uma visão geral do impacto dos**Transformers**e modelos derivados (BERT, GPT, etc.).

Os conceitos que exploramos – modelagem sequencial, captura de contexto, e atenção seletiva – formam a base não apenas dos Transformers, mas de grande parte da pesquisa atual em inteligência artificial. Compreendê-los é essencial para navegar neste campo fascinante.  7 7 7 7 7 7 7 7 7Y6 7Y7Y