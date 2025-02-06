---
layout: post
title: Entendendo Markov Decision Process
author: Frank
categories:
    - Matemática
    - Inteligência Artificial
tags:
    - inteligência artificial
    - Matemática
    - resolução de problemas
    - reinforcement learning
image: /images/mdp1.webp
featured: false
rating: 5
description: A primeira parte do MDP, com código em C++ 20 e os motivos impulsionaram Markov.
date: 2025-02-05T00:25:52.147Z
preview: A primeira parte do MDP, com código em C++ 20 e os motivos impulsionaram Markov.
toc: true
published: true
keywords: Aprendizado por Reforço, História do RL, MDP, Markov.
beforetoc: ""
lastmod: 2025-02-06T21:45:20.210Z
draft: 2025-02-05T00:25:59.828Z
---

>Cheating is not in the nature of a gambler but in the nature of a loser. *John Scarne*

Reinforcement Learning, do ponto de vista da matemática, começa com a Lei dos Grandes Números.

A Lei dos Grandes Números - **LGN**, um teorema fundamental da teoria das probabilidades que descreve o resultado de realizar o mesmo experimento um grande número de vezes. De acordo com esta lei, a média dos resultados obtidos de um grande número de tentativas se aproxima do valor esperado e tenderá a se tornar mais próximo sempre que o número de tentativas aumentar. Isso quer dizer que: se você repetir um experimento aleatório muitas vezes, de forma independente e sob as mesmas condições, a média dos resultados observados convergirá para o valor teórico esperado (a média populacional).

Considere o exemplo menos criativo possível: lançar uma moeda. Neste caso, teremos que considerar o:

**Valor esperado**: Uma moeda justa tem duas faces (cara e coroa), cada uma com probabilidade de $0,5$ (ou $50\%$). O valor esperado de um lançamento é, portanto, $0,5$.

E, providenciar um conjunto de tentativas:

**Resultados experimentais**:

1. Lançando a moeda 10 vezes, obtive $3$ caras e $7$ coroas (média de $0,3$);

2. Lançando a moeda 100 vezes, obtive $61$ caras e $39$ coroas (média de $0,61$);

3. Lançando a moeda $1.000$ vezes: $486$ caras e $514$ coroas (média de $0,49$);

4. Lançando a moeda 10.000 vezes: $5030$ caras e $4970$ coroas (média de $0,5030$).

Pelo menos, é assim que deveria ser. Como não tenho paciência para lançar, anotar e contar. Considerei ser possível criar um lançamento aleatoriamente justo em C++ então:

```c++
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <format>

class CoinFlipper {
private:
    std::random_device rd;  // Hardware entropy source
    std::mt19937 gen;      // Mersenne Twister generator
    std::bernoulli_distribution dist;  // Distribution for fair coin flip

public:
    CoinFlipper() : gen(rd()), dist(0.5) {}

    bool flip() {
        return dist(gen);
    }d
};

struct FlipStats {
    size_t heads;
    size_t tails;
    double heads_ratio;

    FlipStats(const std::vector<bool>& flips) {
        heads = std::count(flips.begin(), flips.end(), true);
        tails = flips.size() - heads;
        heads_ratio = static_cast<double>(heads) / flips.size();
    }
};

std::vector<bool> perform_flips(size_t n) {
    CoinFlipper flipper;
    std::vector<bool> results(n);

    std::generate(results.begin(), results.end(),
        [&flipper]() { return flipper.flip(); });

    return results;
}

void verify_law_of_large_numbers() {
    const std::vector<size_t> trial_sizes = { 10, 100, 1000, 10000 };

    for (const auto size : trial_sizes) {
        auto results = perform_flips(size);
        FlipStats stats(results);

        std::cout << std::format(
            "Number of flips: {:5} | Heads: {:5} | Tails: {:5} | "
            "Heads ratio: {:.4f} | Deviation from expected: {:.4f}\n",
            size,
            stats.heads,
            stats.tails,
            stats.heads_ratio,
            std::abs(0.5 - stats.heads_ratio)
        );
    }
}

int main() {
    std::cout << "Verifying the Law of Large Numbers with coin flips\n";
    std::cout << "Expected probability: 0.5000\n\n";

    verify_law_of_large_numbers();

    return 0;
}

```

Que, ao rodar, com C++ 20, resulta em:

```shell
Verifying the Law of Large Numbers with coin flips
Expected probability: 0.5000

Number of flips:    10 | Heads:     3 | Tails:     7 | Heads ratio: 0.3000 | Deviation from expected: 0.2000
Number of flips:   100 | Heads:    61 | Tails:    39 | Heads ratio: 0.6100 | Deviation from expected: 0.1100
Number of flips:  1000 | Heads:   486 | Tails:   514 | Heads ratio: 0.4860 | Deviation from expected: 0.0140
Number of flips: 10000 | Heads:  5030 | Tails:  4970 | Heads ratio: 0.5030 | Deviation from expected: 0.0030

```

Se a esforçada leitora já sofreu com a teoria da probabilidade, já passou pela **LGN**. Então, esta mesma lei garante que, neste caso, já ouviu falar que existem duas vertentes desta lei:  a **Lei Fraca dos Grandes Números (LFN)** que afirma que a média amostral converge em probabilidade para o valor esperado. Ou seja, a probabilidade de a média amostral se desviar do valor esperado por mais que um valor pequeno tende a zero à medida que o número de tentativas aumenta; e a **Lei Forte dos Grandes Números (LFGN)** que  Afirma que a média amostral converge quase certamente para o valor esperado. Isso é uma afirmação mais forte que a LFN, significando que, com probabilidade $1$, a média amostral se aproximará arbitrariamente do valor esperado à medida que o número de tentativas aumenta. Os dois casos, ainda que diferentes, confirmam a intuição que é possível a partir do teste prático.

Havia, contudo, um problema.

No início do século XX, o matemático russo [Pavel Nekrasov](https://en.wikipedia.org/wiki/Pavel_Nekrasov) afirmava, aos quatro ventos, que a independência entre variáveis aleatórias era uma condição necessária para a **LGN**. Ele usava essa ideia, em parte, para justificar suas crenças religiosas, sugerindo que o livre-arbítrio é necessário para a estabilidade social e assim, a independência seria fundamental na sociedade. Eu gosto da ideia. Mas, isso é minha veia anarquista pulsando alto.

Em outras palavras, Nekrasov defendia que, em experimentos como o lançamento da moeda, a probabilidade só convergem para o valor esperado porque cada novo lançamento é independente do anterior. O que, novamente, é intuitivamente aceitável.

[Markov](https://www.britannica.com/biography/Andrey-Andreyevich-Markov) discordava enfática e veementemente.

Markov acreditava que a **LGN** deveria valer mesmo para eventos que fossem intrinsecamente dependentes e queria provar isso. Sua motivação era tanto matemática quanto ideológica, buscando desvincular a matemática de interpretações religiosas e filosóficas.

Bons tempos em que se acreditava que a matemática deveria ser independente de crenças, valores e opiniões.

Para demonstrar seu ponto, Markov escolheu um problema concreto: analisar a alternância entre vogais e consoantes no poema épico "[Eugene Onegin](https://en.wikipedia.org/wiki/Eugene_Onegin)", de [Alexander Pushkin](https://pt.wikipedia.org/wiki/Alexandre_Pushkin), 288 páginas em cirílico. Se estive correto, isso significaria que a probabilidade de uma letra ser vogal ou consoante dependeria apenas da letra imediatamente anterior, e não de toda a sequência de letras que veio antes.

>Alfabeto cirílico: $33$ letras: $20$ consoantes (б, в, г, д, ж, з, к, л, м, н, п, р, с, т, ф, х, ц, ч, ш, щ), $10$ vogais (а, е, ё, и, о, у, ы, э, ю, я), uma semivogal/consoante (й), e duas letras modificadoras ou "sinais" (ь, ъ) que alteram a pronúncia de uma consoante anterior ou de uma vogal seguinte.

Markov queria mostrar que, mesmo existindo uma dependência entre letras consecutivas (por exemplo, após uma consoante, é mais provável que venha uma vogal), a frequência relativa de vogais e consoantes ainda convergiria para um valor específico à medida que o texto se alongasse. Essa prova, se possível, validaria a **LGN** mesmo em eventos não independentes.

Em busca da evidência necessária, Markov contou meticulosamente as vogais e consoantes nas primeiras $20.000$ letras do poema, categorizando-as em blocos de $100$ letras. Ele precisava desses dados para estimar as probabilidades de transição que definiriam sua Cadeia de Markov. Sobre estes dados, observou que a probabilidade de uma letra ser vogal ou consoante era fortemente influenciada pela letra precedente, o que ele chamou de *dependência de curto alcance*.  Após uma consoante, a probabilidade de a letra subsequente ser uma vogal era significativamente maior.

Para formalizar suas observações, Markov propôs que a probabilidade de uma letra dependeria exclusivamente da letra imediatamente anterior, e não do histórico completo de letras anteriores. Essa formalização é conhecida como a *propriedade de Markov*, também descrita como a propriedade da ausência de memória, e é o princípio fundamental que define as Cadeias de Markov.

### Cadeias de Markov e a Propriedade de Markov

Markov precisava de um modelo matemático que capturasse a ideia de dependência, mas de uma forma específica. Ele formalizou essa ideia com o que hoje conhecemos como *Propriedade de Markov*, ou ausência de memória. Em termos mais formais, dizemos que:

Para uma sequência de variáveis aleatórias $X_1, X_2, ..., X_n$, a propriedade de Markov estabelece que, para todo $n$:

$$ P(X_{n+1} = x_{n+1} | X_n = x_n, X_{n-1} = x_{n-1}, ..., X_1 = x_1) = P(X_{n+1} = x_{n+1} | X_n = x_n) $$

A atenta leitora deve considerar que essa propriedade diz que o futuro (o estado $X_{n+1}$) depende apenas do presente (o estado $X_n$), e não de todo o passado ($X_{n-1}, X_{n-2}, ..., X_1$). Neste caso, dizemos que o sistema *não tem memória* além do seu estado atual.

Agora podemos chegar a *Cadeia de Markov*.

Uma Cadeia de Markov é uma sequência de variáveis aleatórias que obedece à propriedade de Markov. Ou seja, *é um modelo matemático onde a probabilidade de transição para o próximo estado depende apenas do estado atual*.

Após estabelecer a propriedade de Markov, o próximo passo foi determinar as probabilidades de transição que governariam o comportamento da cadeia. Voltando  ao texto de *Eugene Onegin* e aos dados que havia recolhido, Markov estimou essas probabilidades contando a frequência de ocorrência de cada tipo de transição (vogal para consoante, consoante para vogal, etc.) e construindo uma matriz de transição.

Formalmente dizemos que a dependência entre os estados em uma Cadeia de Markov pode ser integralmente descrita pela sua matriz de transição. De forma que: para um sistema com $n$ estados possíveis, a matriz de transição $P$ é uma matriz $n \times n$, onde cada elemento $p_{ij}$ representa a probabilidade de transição do estado $i$ para o estado $j$:

$$
P = \begin{bmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \
p_{21} & p_{22} & \cdots & p_{2n} \
\vdots & \vdots & \ddots & \vdots \
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{bmatrix}
$$

Neste caso, temos:

- $p_{ij} = P(X_{n+1} = j | X_n = i)$ (a probabilidade de ir para o estado $j$ dado que o estado atual é $i$)
- $0 \le p_{ij} \le 1$ para todos $i$ e $j$ (todos os elementos são probabilidades)
- $\sum_{j=1}^{n} p_{ij} = 1$ para todo $i$ (a soma das probabilidades de transição a partir de um dado estado deve ser 1).

Aplicando as probabilidades de transição determinadas a partir de *Eugene Onegin*, Markov realizou simulações estocásticas e cálculos analíticos para demonstrar que a frequência relativa de vogais e consoantes convergia para um valor específico, independentemente do estado inicial e à medida que a sequência de letras se estendia. Essa demonstração provou que a Lei dos Grandes Números era válida mesmo em sistemas com dependência de curto alcance, como o sistema composto pelas palavras e letras de *Eugene Onegin*. Com isso, Markov não apenas refutou a alegação de Nekrasov de que a independência era um requisito para a LGN, mas também abriu caminho para uma compreensão mais ampla de processos estocásticos.

O trabalho de Markov com "Eugene Onegin" foi um exemplo concreto e poderoso, mas sua intuição ia além da análise literária. Ele percebeu que o princípio da dependência de curto alcance, formalizado na propriedade de Markov, poderia ser *generalizado para qualquer sistema que evoluísse em etapas, onde o estado futuro dependesse apenas do estado presente*.

A primeira vez que estudei MDP, lá nos anos 1980, o professor sugeriu que fossemos a biblioteca, pegássemos uma edição de *Eugene Onegin* e refizéssemos o trabalho de Markov. Como não valia nota, ninguém fez. Os alunos são todos iguais, não importa a era, ou a disciplina. Porém, os tempos são outros.

Não consegui o texto em cirílico completo em txt, mas consegui uma versão antiga com $18327$ letras, eu ignorei a semivogal/consoante (й), e as duas letras modificadoras (ь, ъ) durante a contagem de letras.

O número exato de letras do poema completo, em cirílico, parece variar ligeiramente dependendo do ano da edição. Talvez o idioma tenha passado por reformas e atualizações entre os tempos de Markov e este em que escrevo. Não chequei e não interessa.  O número exato de letras não é relevante para verificarmos o trabalho de Markov. O foco dele estava em uma amostra com $20.000$ letras consecutivas. As nossas $18327$ letras devem ser suficientes. Finalmente, para usar o texto em cirílico, tive que trabalhar em UTF-8 e minerar a web em busca de [uma versão em cirílico](\assets\poema.txt). Não foi fácil.

Para verificar empiricamente as ideias de Markov, e em homenagem aos velhos tempos, implementei a análise em C++ 20. O código completo, juntamente com explicações detalhadas, está apresentado a seguir.

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <ranges>
#include <codecvt>
#include <locale>
#include <iomanip> // Adicionado para std::setprecision

// Define character sets for Russian alphabet
namespace russian {
    // Vogais em hexadecimal UTF-8
    const std::unordered_set<std::string> vowels = {
        "\xD0\xB0",  // а
        "\xD0\xB5",  // е
        "\xD1\x91",  // ё
        "\xD0\xB8",  // и
        "\xD0\xBE",  // о
        "\xD1\x83",  // у
        "\xD1\x8B",  // ы
        "\xD1\x8D",  // э
        "\xD1\x8E",  // ю
        "\xD1\x8F"   // я
    };
    
    // Consoantes em hexadecimal UTF-8
    const std::unordered_set<std::string> consonants = {
        "\xD0\xB1",  // б
        "\xD0\xB2",  // в
        "\xD0\xB3",  // г
        "\xD0\xB4",  // д
        "\xD0\xB6",  // ж
        "\xD0\xB7",  // з
        "\xD0\xBA",  // к
        "\xD0\xBB",  // л
        "\xD0\xBC",  // м
        "\xD0\xBD",  // н
        "\xD0\xBF",  // п
        "\xD1\x80",  // р
        "\xD1\x81",  // с
        "\xD1\x82",  // т
        "\xD1\x84",  // ф
        "\xD1\x85",  // х
        "\xD1\x86",  // ц
        "\xD1\x87",  // ч
        "\xD1\x88",  // ш
        "\xD1\x89"   // щ
    };
}

// Estrutura para armazenar estatísticas de transição
struct TransitionStats {
    size_t vowel_to_vowel = 0;
    size_t vowel_to_consonant = 0;
    size_t consonant_to_vowel = 0;
    size_t consonant_to_consonant = 0;
    size_t total_transitions = 0;
    size_t total_vowels = 0;
    size_t total_consonants = 0;
};

class MarkovAnalysis {
private:
    TransitionStats stats;
    std::unordered_map<std::string, double> transition_probs;

    std::string get_utf8_char(const std::string& str, size_t& pos) {
        std::string result;
        unsigned char c = str[pos];
        
        if ((c & 0x80) == 0) {
            result = str.substr(pos, 1);
            pos += 1;
        } else if ((c & 0xE0) == 0xC0) {
            result = str.substr(pos, 2);
            pos += 2;
        } else if ((c & 0xF0) == 0xE0) {
            result = str.substr(pos, 3);
            pos += 3;
        } else if ((c & 0xF8) == 0xF0) {
            result = str.substr(pos, 4);
            pos += 4;
        }
        return result;
    }

    bool is_vowel(const std::string& c) const {
        return russian::vowels.contains(c);
    }

    bool is_consonant(const std::string& c) const {
        return russian::consonants.contains(c);
    }

public:
    void process_text(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }

        std::string text((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        file.close();

        std::string prev_char;
        bool prev_was_vowel = false;
        bool first_char = true;
        
        for (size_t i = 0; i < text.length();) {
            std::string current_char = get_utf8_char(text, i);
            
            if (!is_vowel(current_char) && !is_consonant(current_char)) {
                continue;
            }

            bool is_vowel_curr = is_vowel(current_char);
            
            if (is_vowel_curr) {
                stats.total_vowels++;
            } else {
                stats.total_consonants++;
            }

            if (!first_char) {
                stats.total_transitions++;
                if (prev_was_vowel && is_vowel_curr) {
                    stats.vowel_to_vowel++;
                } else if (prev_was_vowel && !is_vowel_curr) {
                    stats.vowel_to_consonant++;
                } else if (!prev_was_vowel && is_vowel_curr) {
                    stats.consonant_to_vowel++;
                } else {
                    stats.consonant_to_consonant++;
                }
            }

            prev_char = current_char;
            prev_was_vowel = is_vowel_curr;
            first_char = false;
        }

        if (stats.total_transitions > 0) {
            transition_probs["vv"] = static_cast<double>(stats.vowel_to_vowel) / stats.total_transitions;
            transition_probs["vc"] = static_cast<double>(stats.vowel_to_consonant) / stats.total_transitions;
            transition_probs["cv"] = static_cast<double>(stats.consonant_to_vowel) / stats.total_transitions;
            transition_probs["cc"] = static_cast<double>(stats.consonant_to_consonant) / stats.total_transitions;
        }
    }

    void print_results() const {
        std::cout << "\nMarkov Analysis Results:\n";
        std::cout << "------------------------\n";
        std::cout << "Total characters analyzed: " 
                  << (stats.total_vowels + stats.total_consonants) << "\n";
        std::cout << "Total vowels: " << stats.total_vowels << "\n";
        std::cout << "Total consonants: " << stats.total_consonants << "\n";
        
        double vowel_freq = static_cast<double>(stats.total_vowels) / 
                           (stats.total_vowels + stats.total_consonants);
        std::cout << "Vowel frequency: " << std::fixed << std::setprecision(4) 
                  << vowel_freq << "\n\n";

        std::cout << "Transition Probabilities:\n";
        std::cout << "Vowel to Vowel: " << std::fixed << std::setprecision(4) 
                  << transition_probs.at("vv") << "\n";
        std::cout << "Vowel to Consonant: " << transition_probs.at("vc") << "\n";
        std::cout << "Consonant to Vowel: " << transition_probs.at("cv") << "\n";
        std::cout << "Consonant to Consonant: " << transition_probs.at("cc") << "\n";
    }
};

int main() {
    try {
        MarkovAnalysis analyzer;
        analyzer.process_text("poema.txt");
        analyzer.print_results();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

```shell
Markov Analysis Results:
------------------------
Total characters analyzed: 18327
Total vowels: 7830
Total consonants: 10497
Vowel frequency: 0.4272

Transition Probabilities:
Vowel to Vowel: 0.0438
Vowel to Consonant: 0.3834
Consonant to Vowel: 0.3834
Consonant to Consonant: 0.1893
```

Vamos tentar entender esse código. Começamos com as bibliotecas necessárias para o código:

```cpp
#include <iostream>     // I/O
#include <fstream>      // File handling
#include <string>       // String operations
#include <vector>       // Dynamic arrays
#include <unordered_map>    // Hash maps
#include <unordered_set>    // Hash sets
#include <numeric>      // Numeric operations
#include <ranges>       // C++20 ranges
#include <codecvt>      // Encoding conversion
#include <locale>       // Localization
#include <iomanip>      // Output formatting
```

Precisamos definir as consoantes e vogais do alfabeto russo para isso criei o namespace `russian` para definir estes caracteres em UTF-8:

```cpp
namespace russian {
   const std::unordered_set<std::string> vowels = {
       "\xD0\xB0",  // а
       "\xD0\xB5",  // е
       "\xD1\x91",  // ё
       ...
   };
   
   const std::unordered_set<std::string> consonants = {
       "\xD0\xB1",  // б
       "\xD0\xB2",  // в
       ...
   };
}
```

A estrutura `TransitionStats` armazena todas as contagens necessárias para calcular as probabilidades de transição entre vogais e consoantes:

1. **vowel_to_vowel**: Conta quantas vezes uma vogal é seguida por outra vogal.
  
   - Exemplo: "еа" em "театр" - $P(V|V)$

2. **vowel_to_consonant**: Conta quantas vezes uma vogal é seguida por uma consoante.
  
   - Exemplo: "ит" em "итог" - $P(C|V)$

3. **consonant_to_vowel**: Conta quantas vezes uma consoante é seguida por uma vogal.
  
   - Exemplo: "ра" em "рада" - $P(V|C)$

4. **consonant_to_consonant**: Conta quantas vezes uma consoante é seguida por outra consoante.
  
   - Exemplo: "ст" em "стол" - $P(C|C)$

A estrutura `TransitionStats` também contém as seguintes variáveis totalizadoras:

1. **total_transitions**: Soma total de todas as transições.

   - Matematicamente: `total_transitions = vowel_to_vowel + vowel_to_consonant + consonant_to_vowel + consonant_to_consonant`
  
   - Usado como denominador para calcular probabilidades

2. **total_vowels**: Número total de vogais no texto.
  
   - Usado para calcular a frequência geral de vogais: $f_v = \frac{total\_vowels}{total\_vowels + total\_consonants}$

3. **total_consonants**: Número total de consoantes no texto.
  
   - Usado para calcular a frequência geral de consoantes: $f_c = \frac{total\_consonants}{total\_vowels + total\_consonants}$

As contagens são usadas para construir a matriz de transição $P$:

$$
P = \begin{bmatrix}
\frac{vowel\_to\_vowel}{total\_transitions} & \frac{vowel\_to\_consonant}{total\_transitions} \\
\frac{consonant\_to\_vowel}{total\_transitions} & \frac{consonant\_to\_consonant}{total\_transitions}
\end{bmatrix}
$$

Uma matriz de transição, também conhecida como matriz de probabilidades de transição, é uma estrutura matemática fundamental em Cadeias de Markov que descreve as probabilidades de transição entre estados de um sistema. Formalmente dizemos que para um sistema com $n$ estados, a matriz de transição $P$ é uma matriz $n \times n$ onde cada elemento $p_{ij}$ representa a probabilidade de transição do estado $i$ para o estado $j$:

$$
P = \begin{bmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{bmatrix}
$$

Uma matriz de transição apresenta as seguintes propriedades:

1. Todos os elementos são não-negativos:
  
    $$ p_{ij} \geq 0 \quad \forall i,j $$

2. A soma de cada linha é 1 (probabilidade total):
  
    $$ \sum_{j=1}^n p_{ij} = 1 \quad \forall i $$

Para o texto de *Eugene Onegin*, temos uma matriz $2 \times 2$ pois existem apenas dois estados possíveis (vogal ou consoante):

$$
P = \begin{bmatrix}
P(V|V) & P(V|C) \\
P(C|V) & P(C|C)
\end{bmatrix}
$$

Neste caso, temos:

- $P(V|V)$ é a probabilidade de uma vogal após uma vogal
- $P(V|C)$ é a probabilidade de uma vogal após uma consoante
- $P(C|V)$ é a probabilidade de uma consoante após uma vogal
- $P(C|C)$ é a probabilidade de uma consoante após uma consoante

Por exemplo, se no texto encontramos:

- "еа" em "театр": contribui para $P(V|V)$
- "ит" em "итог": contribui para $P(C|V)$
- "ра" em "рада": contribui para $P(V|C)$
- "ст" em "стол": contribui para $P(C|C)$

A matriz resultante poderia ser algo como:

$$
P = \begin{bmatrix}
0.3 & 0.7 \\
0.6 & 0.4
\end{bmatrix}
$$

Onde cada linha soma $1$, pois representa todas as possibilidades para o próximo estado dado o estado atual.

A contagem das transições é feita pelo fragmento a seguir:

```cpp
// Exemplo de uso durante o processamento do texto
if (prev_was_vowel && is_vowel_curr) {
   stats.vowel_to_vowel++;
} else if (prev_was_vowel && !is_vowel_curr) {
   stats.vowel_to_consonant++;
} else if (!prev_was_vowel && is_vowel_curr) {
   stats.consonant_to_vowel++;
} else {
   stats.consonant_to_consonant++;
}
stats.total_transitions++;

if (is_vowel_curr) {
   stats.total_vowels++;
} else {
   stats.total_consonants++;
}
```

Neste caso, temos:

```cpp
if (prev_was_vowel && is_vowel_curr) {
  stats.vowel_to_vowel++;
}
```

Se a letra anterior era vogal (`prev_was_vowel`) E a letra atual é vogal (`is_vowel_curr`), incrementa a contagem de transições `vogal→vogal`.

Exemplo: Em "еа" (театр):

- `prev_was_vowel = true` (е é vogal)
- `is_vowel_curr = true` (а é vogal)
- Incrementa `vowel_to_vowel`

```cpp
else if (prev_was_vowel && !is_vowel_curr) {
  stats.vowel_to_consonant++;
}
```

Se a letra anterior era vogal E a letra atual é consoante, incrementa a contagem de transições `vogal→consoante`.

Exemplo: Em "ит" (итог):

- `prev_was_vowel = true` (и é vogal)
- `is_vowel_curr = false` (т é consoante)
- Incrementa `vowel_to_consonant`

```cpp
else if (!prev_was_vowel && is_vowel_curr) {
  stats.consonant_to_vowel++;
}
```

Se a letra anterior era consoante E a letra atual é vogal, incrementa a contagem de transições `consoante→vogal`.

Exemplo: Em "ра" (рада):

- `prev_was_vowel = false` (р é consoante)
- `is_vowel_curr = true` (а é vogal)
- Incrementa `consonant_to_vowel`

```cpp
else {
  stats.consonant_to_consonant++;
}
```

Se nenhuma das condições anteriores for verdadeira, significa que temos uma transição `consoante→consoante`.

Exemplo: Em "ст" (стол):

- `prev_was_vowel = false` (с é consoante)
- `is_vowel_curr = false` (т é consoante)
- Incrementa `consonant_to_consonant`

Finalmente em :

```cpp
stats.total_transitions++;
```

Incrementamos o contador total de transições, independente do tipo. Enquanto:

```cpp
if (is_vowel_curr) {
  stats.total_vowels++;
} else {
  stats.total_consonants++;
}
```

Mantém a contagem total de vogais e consoantes no texto. Estas contagens serão usadas para calcular as probabilidades $P(i|j)$ da matriz de transição:

$$ P(i|j) = \frac{count(j \rightarrow i)}{total\_transitions} $$

Por exemplo:

$$ P(V|V) = \frac{vowel\_to\_vowel}{total\_transitions} $$

A classe principal `MarkovAnalysis` implementa a matriz de transição $P$, de forma que:

$$
P_{ij} = P(X_{n+1} = j | X_n = i)
$$

Para $i,j \in \{vogal, consoante\}$

O processamento dos caracteres, que também é implementando na classe `MarkovAnalysis` segue o modelo:

$$
P(X_n | X_{n-1}) = \frac{count(X_{n-1} \rightarrow X_n)}{total\_transitions}
$$

A implementação de `MarkovAnalysis` pode ser vista em:

```cpp
class MarkovAnalysis {
private:
   TransitionStats stats;
   std::unordered_map<std::string, double> transition_probs;

   // UTF-8 character extraction
   std::string get_utf8_char(const std::string& str, size_t& pos) {
       // Character extraction logic
   }

   bool is_vowel(const std::string& c) const {
       return russian::vowels.contains(c);
   }

   bool is_consonant(const std::string& c) const {
       return russian::consonants.contains(c);
   }

public:
   void process_text(const std::string& filename) {
       // Text processing logic
   }

   void print_results() const {
       // Results output
   }
};
```

Uma vez que o código seja executado corretamente, os resultados serão apresentados em duas partes:

1. Frequências absolutas:
  
   - Total de caracteres
   - Total de vogais
   - Total de consoantes

2. Matriz de probabilidades de transição:
  
   - $P(V|V)$ - Vogal para Vogal
   - $P(C|V)$ - Vogal para Consoante
   - $P(V|C)$ - Consoante para Vogal
   - $P(C|C)$ - Consoante para Consoante

Este exemplo em C++ 20 demonstra a tese de Markov de que a Lei dos Grandes Números pode ser aplicada a sequências dependentes, logo:

$$
\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n X_i = \mu
$$

Para uma sequência de variáveis aleatórias $\{X_i\}$ com dependência markoviana.

É importante que a Perspicaz leitora observe que a *Cadeia de Markov*, no código, está representada de forma implícita e explícita em diferentes partes:

1. Implicitamente (na Lógica):

    `process_text` (dentro da classe MarkovAnalysis): a lógica da função `process_text` incorpora a propriedade de Markov. Observe as variáveis `prev_was_vowel` e `current_char`. A função itera pelo texto, e a cada passo, a decisão de incrementar `vowel_to_vowel`, `vowel_to_consonant`, etc., depende apenas do caractere atual (`current_char`, que representa $X_{n+1}$)e do estado anterior (`prev_was_vowel`, que representa $X_n$). Não há nenhuma consideração de caracteres anteriores a `prev_char`. Essa é a propriedade de Markov em ação.

2. Explicitamente (na Estrutura de Dados):

    `transition_probs` (dentro da classe `MarkovAnalysis`): Este `unordered_map` armazena a matriz de transição da Cadeia de Markov. As chaves são strings ("vv", "vc", "cv", "cc") que representam os pares de estados (Vogal-Vogal, Vogal-Consoante, etc.), e os valores são as probabilidades de transição correspondentes $(P(V∣V), P(C∣V), P(V∣C), P(C∣C))$. Esta é a representação explícita da Cadeia de Markov.

    ```cpp
    if (stats.total_transitions > 0) {
        transition_probs["vv"] = static_cast<double>(stats.vowel_to_vowel) / stats.total_transitions;
        transition_probs["vc"] = static_cast<double>(stats.vowel_to_consonant) / stats.total_transitions;
        transition_probs["cv"] = static_cast<double>(stats.consonant_to_vowel) / stats.total_transitions;
        transition_probs["cc"] = static_cast<double>(stats.consonant_to_consonant) / stats.total_transitions;
    }
    ```

Ou seja, enquanto a lógica do código (especialmente em `process_text`) implementa a propriedade de Markov. A matriz de transição, armazenada em `transition_probs`, é a representação explícita da Cadeia de Markov.

Note também, curiosa leitora, que este código constrói a Cadeia de Markov a partir de dados (o texto do poema). Ele não simula a cadeia (ou seja, não gera novas sequências de letras usando as probabilidades de transição). Uma simulação seria um passo adicional, onde você começaria com um estado inicial (digamos, uma vogal) e, a cada passo, usaria as probabilidades em `transition_probs` para escolher aleatoriamente o próximo estado (vogal ou consoante). Começou a ficar interessante. Mas, aí já é russo demais para mim. Vamos estudar Cadeias de Markov e o processo de decisão, na outra parte deste capítulo, quando estudaremos uma grid.
