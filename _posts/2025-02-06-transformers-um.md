---
layout: post
title: Transformers - A Temida Matemática
author: frank
categories:
    - disciplina
    - Matemática
    - artigo
tags:
    - C++
    - Matemática
    - inteligência artificial
image: assets/images/transformers1.webp
featured: false
rating: 5
description: Uma introdução a matemática que suporta a criação de transformers para processamento de linguagem natural com exemplos de código em C++20.
date: 2025-02-09T22:55:34.524Z
preview: Uma introdução a matemática que suporta a criação de transformers para processamento de linguagem natural com exemplos de código em C++20.
keywords:
    - transformers
    - matemática
    - processamento de linguagem natural
    - C++
    - aprendizado de máquina
    - vetores
    - produto escalar
    - álgebra linear
    - embeddings
    - atenção
    - deep learning
    - inteligência artificial
toc: true
published: true
beforetoc: ""
lastmod: 2025-04-02T18:10:09.146Z
---

Neste artigo, a curiosa leitora irá enfrentar os *Transformers*. Nenhuma relação com o o Optimus Prime. Se for estes *Transformers* que está procurando, **o Google falhou com você!**

Neste texto vamos discutir os **Transformers** modelos de aprendizado de máquina que revolucionaram o processamento de linguagem natural (**NLP**). Estas técnicas foram apresentados ao mundo em um artigo intitulado *Attention is All You Need* (Atenção é Tudo que Você Precisa), publicado em 2017[^1] na conferência *Advances in Neural Information Processing Systems (NeurIPS)*. Observe, atenta leitora que isso se deu há quase 10 anos. No ritmo atual, uma eternidade.

O entendimento da linguagem natural por máquinas é, ou era, um desafio importante que beirava o impossível. Este problema parece estar resolvido. Se isso for verdade, terá sido graças as técnicas e algoritmos, criados em torno de aprendizado de máquinas e estatísticas. Ou se preferir, podemos dizer que Usamos algoritmos determinísticos para aplicar técnicas estocásticas em bases de dados gigantescas e assim, romper os limites que haviam sido impostos pela linguística matemática e computacional determinísticas.

Veremos como esses modelos, inicialmente projetados para tradução automática, se tornaram a base para tarefas como geração de texto, como no [GPT-3](https://openai.com/index/gpt-3-apps/), compreensão de linguagem e até mesmo processamento de áudio.

[^1]: VASWANI, Ashish et al. Attention is all you need. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. Proceedings of the [...]. Red Hook: Curran Associates, Inc., 2017. p. 5998-6008. Disponível em: [https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper). Acesso em: 09 fevereiro 2024.

Começaremos com as técnicas de representação mais simples e os conceitos matemáticos fundamentais: produtos escalares e multiplicação de matrizes. E, gradualmente, construiremos nosso entendimento. O que suporta todo este esforço é a esperança que a esforçada leitora possa acompanhar o raciocínio e entender como os *Transformers* funcionam a partir do seu cerne.

Finalmente, os exemplos. O combinado será o seguinte: aqui eu faço em C++ 20. Depois, a leitora faz em Python, C, C++ ou qualquer linguagem que desejar. Se estiver de acordo continuamos.

Para que os computadores processem e compreendam a linguagem humana, é essencial converter texto em representações numéricas. Esse treco burro só entende binário. Dito isso, vamos ter que, de alguma forma, mapear o conjunto dos termos que formam uma linguagem natural no conjunto dos binários que o computador entende. Ou, em outras palavras, temos que representar textos em uma forma matemática que os computadores possam manipular. Essa representação é o que chamamos de vetorização.

### Vetores, os compassos de tudo que há e haverá

Eu usei exatamente esta frase em [um texto sobre eletromagnetismo](https://frankalcantara.com/formula-da-atracao-matematica-eletromagnetismo/#vetores-os-compassos-de-tudo-que-h%C3%A1-e-haver%C3%A1). A ideia, então era explicar eletromagnetismo a partir da matemática. Lá há uma definição detalhada de vetores e todas as suas operações. Aqui, podemos ser um tanto mais diretos. Vetores são os artefatos matemáticos que usamos para explicar o universo.

Um vetor é uma entidade matemática que possui tanto magnitude, ou comprimento, quanto direção. Um vetor pode ser definido como um segmento de reta direcionado na geometria, ou uma sequência ordenada de números, chamados de componentes, na álgebra. A representação depende do contexto. Aqui, vamos nos concentrar na representação algébrica, que é mais comum em programação e computação.

Na geometria, um vetor pode ser visualizado como uma seta em um espaço, por exemplo, em um plano $2D$ ou em um espaço $3D$. O comprimento da seta representa a magnitude, e a direção da seta indica a direção do vetor. Imagine uma seta apontando para cima e para a direita em um plano. Essa seta é um vetor com uma certa magnitude (o comprimento da seta) e uma direção ($45$ graus em relação ao eixo horizontal, por exemplo). A Figura 1 mostra um vetor como usado na matemática e na física.

![uma seta vermelha representando um vetor](/assets/images/vector1.webp)
_Figura 1: Um vetor partindo da origem $O$ com comprimento $\mid V \mid$ e dimensões $V_1$ e $V_2$.{: class="legend"}

Em um sistema algébrico de coordenadas, um vetor pode ser representado como uma tupla. Por exemplo, em um espaço tridimensional, um vetor pode ser escrito como $(x, y, z)$, onde $x$, $y$ e $z$ são as componentes do vetor ao longo dos eixos $x$, $y$ e $z$, respectivamente. Assim, se nos limitarmos a $2D$, o vetor $(2, 3)$ representa um deslocamento de $2$ unidades na direção $x$ e $3$ unidades na direção $y$. Na Figura 1 podemos ver um vetor $V$ partindo da origem $O$ e terminando no ponto $P(V_1, V_2)$.

#### Espaço Vetorial

Para compreender vetores e suas operações, precisamos primeiro entender um conceito algébrico, o conceito de espaço vetorial.

>Um espaço vetorial é uma estrutura matemática que formaliza a noção de operações geométricas como adição de vetores e multiplicação por escalares.

Formalmente, um espaço vetorial sobre um corpo $F$ é um conjunto $V$ no qual há adição de vetores e multiplicação por escalares em $F$, obedecendo axiomas que garantem associatividade, comutatividade, existência de neutro e inverso aditivo, além de compatibilidade entre multiplicação por escalar e estrutura do corpo.

Em processamento de linguagem natural, trabalharemos principalmente com o espaço vetorial real $\mathbb{R}^n$, onde $n$ representa a dimensão do espaço vetorial. Ou, em nosso caso, quantos itens teremos no nosso vetor. Logo, $\mathbb{R}^n$ representa o espaço vetorial que contém todas as $n$-tuplas ordenadas de números reais. Formalmente, definimos $\mathbb{R}^n$ como:

$$
\mathbb{R}^n = \{(x_1, \ldots, x_n) : x_i \in \mathbb{R} \text{ para } i = 1, \ldots, n\}
$$

Quando representarmos palavras (termos), ou documentos, como vetores, estaremos mapeando elementos linguísticos para pontos em um espaço dado por $\mathbb{R}^n$. Neste caso, a dimensão $n$ será determinada pelo método específico de vetorização que escolhermos.

Ao converter textos e elementos linguísticos em representações vetoriais, criamos *word embeddings*. Técnicas que mapeiam palavras ou frases para vetores de números reais. Esta representação tenta capturar tanto o significado semântico quanto as relações contextuais entre palavras em um espaço vetorial contínuo. Um desenvolvimento importante nesse campo são os Mecanismos de Atenção, que utilizam vetores de consulta (*query*), chave (*key*) e valor (*value*) como componentes essenciais. Estes mecanismos constituem o núcleo da arquitetura dos *Transformers*, permitindo que o modelo pondere a importância relativa de diferentes elementos em uma sequência, melhorando significativamente a capacidade de processamento de dependências de longo alcance em textos. Para entender isso, precisamos entender como fazer operações algébricas com vetores.

#### Operações com Vetores

Dado que estejam em um espaço vetorial, os vetores podem ser somados, subtraídos, multiplicados entre si e por escalares. Neste caso, a curiosa leitora deve saber que *escalares são entidades sem direção*. As operações sobre vetores têm interpretações geométricas e algébricas. Focando apenas nas interpretações algébricas, temos:

1. **Soma**: somamos vetores componente a componente. Exemplo: se tivermos $\vec{a}= (1, 2)$ e $\vec{b}= (3, -1)$ então $\vec{a} + \vec {b}$ será dado por $(1, 2) + (3, -1) = (4, 1)$;

2. **Oposto**: Dado um vetor $\vec{v} = (v_1, v_2, \ldots, v_n)$ no espaço $\mathbb{R}^n$, seu oposto será dado por $-\vec{v} = (-v_1, -v_2, \ldots, -v_n)$. Ou seja, o vetor oposto é o vetor que aponta na direção oposta e tem a mesma magnitude. Exemplo: se $\vec{a}= (1, 2)$ o oposto de $\vec{a}$ será dado por $-\vec{a} = (-1, -2)$;

3. **Multiplicação por escalar**: multiplicar um vetor por um escalar altera a sua magnitude, mas não a sua direção, a menos que o escalar seja negativo, caso em que a direção é invertida. Exemplo: dado $\vec{a} = (1, 2)$ o dobro de $\vec{a}$ será dado por $2 * (1, 2) = (2, 4)$;

#### Exemplo em C++ 20

```cpp
#include <iostream>
#include <vector>
#include <numeric> // Para std::inner_product
#include <cmath>   // Para std::sqrt, std::abs
#include <concepts> // Para std::is_arithmetic_v
#include <stdexcept> // Para exceções
#include <string>    // Para std::to_string, std::string
#include <sstream>  // Para formatação de string de floats
#include <iomanip>  // Para std::fixed, std::setprecision
#include <initializer_list> // Para construtor com {}
#include <algorithm> // Para std::abs em Manhattan (usado depois)

// --- Definição da Classe MathVector ---

// Conceito para tipos aritméticos
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Classe genérica para representar um vetor matemático
template<Arithmetic T>
class MathVector {
private:
    std::vector<T> components;
     // Constante pequena para comparações de ponto flutuante
    static constexpr T epsilon = 1e-9;

public:
    // Construtor padrão (vetor vazio)
    MathVector() = default;

    // Construtor a partir de uma lista de inicialização
    MathVector(std::initializer_list<T> init) : components(init) {}

    // Construtor a partir de um std::vector
    explicit MathVector(const std::vector<T>& vec) : components(vec) {}

     // Construtor para criar um vetor de tamanho 'n' com valor inicial 'val'
    explicit MathVector(size_t n, T val = T{}) : components(n, val) {}


    // Retorna o número de dimensões do vetor
    size_t dimensions() const {
        return components.size();
    }

    // Acesso a componentes do vetor (não-const)
    T& operator[](size_t index) {
         if (index >= dimensions()) {
            throw std::out_of_range("Índice fora dos limites do vetor");
        }
        return components[index];
    }

    // Acesso a componentes do vetor (const)
    const T& operator[](size_t index) const {
         if (index >= dimensions()) {
            throw std::out_of_range("Índice fora dos limites do vetor");
        }
        return components[index];
    }

    // Adição de vetores
    MathVector<T> operator+(const MathVector<T>& other) const {
        if (dimensions() != other.dimensions()) {
            throw std::invalid_argument("Não é possível somar vetores de dimensões diferentes");
        }
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) {
            result[i] = components[i] + other[i];
        }
        return result;
    }

    // Subtração de vetores
    MathVector<T> operator-(const MathVector<T>& other) const {
        if (dimensions() != other.dimensions()) {
            throw std::invalid_argument("Não é possível subtrair vetores de dimensões diferentes");
        }
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) {
            result[i] = components[i] - other[i];
        }
        return result;
    }

    // Multiplicação por escalar (vetor * escalar)
    MathVector<T> operator*(T scalar) const {
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) {
            result[i] = components[i] * scalar;
        }
        return result;
    }

    // Vetor oposto (operador unário -)
    MathVector<T> operator-() const {
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) {
            result[i] = -components[i];
        }
        return result;
    }

    // Produto escalar (dot product)
    T dot(const MathVector<T>& other) const {
        if (dimensions() != other.dimensions()) {
            throw std::invalid_argument("Não é possível calcular o produto escalar de vetores de dimensões diferentes");
        }
        return std::inner_product(components.begin(), components.end(), other.components.begin(), T(0));
    }

    // Magnitude (norma L2) do vetor
    T magnitude() const {
        T sum_of_squares = this->dot(*this);
        if constexpr (std::is_integral_v<T>) {
             return static_cast<T>(std::sqrt(static_cast<double>(sum_of_squares)));
        } else {
             return std::sqrt(sum_of_squares);
        }
    }

    // Normalização do vetor (retorna um novo vetor normalizado)
    MathVector<T> normalize() const {
        T mag = magnitude();
        if (std::abs(mag) < epsilon) {
            throw std::domain_error("Não é possível normalizar um vetor de magnitude zero (ou muito próxima de zero)");
        }
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) {
            result[i] = components[i] / mag;
        }
        return result;
    }

    // Representação em string
    std::string to_string() const {
        std::string result = "[";
        for (size_t i = 0; i < dimensions(); ++i) {
             if constexpr (std::is_floating_point_v<T>) {
                 std::stringstream ss;
                 ss << std::fixed << std::setprecision(4) << components[i]; // Controle de precisão
                 result += ss.str();
             } else {
                result += std::to_string(components[i]);
             }
            if (i < dimensions() - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }

    // Permite iteração baseada em range
    auto begin() const { return components.begin(); }
    auto end() const { return components.end(); }
    auto begin() { return components.begin(); }
    auto end() { return components.end(); }

     // Acesso ao vetor subjacente
    const std::vector<T>& get_components() const { return components; }
};

// Multiplicação por escalar (escalar * vetor) - função livre
template<Arithmetic T>
MathVector<T> operator*(T scalar, const MathVector<T>& vec) {
    return vec * scalar; // Reutiliza o operador membro
}

// Função auxiliar para imprimir MathVector (usa MathVector::to_string)
template<Arithmetic T>
void print_mathvector(const MathVector<T>& vec, const std::string& name) {
    std::cout << name << " = " << vec.to_string() << "\n";
}


int main() {
    // Configurar a precisão para saída de números de ponto flutuante
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Exemplo 1: Operações básicas com MathVector\n";
    MathVector<double> a{2, 5, 1};
    MathVector<double> b{3, 1, 4};

    print_mathvector(a, "Vetor a");
    print_mathvector(b, "Vetor b");

    // Adição
    MathVector<double> sum = a + b;
    print_mathvector(sum, "a + b");

    // Subtração
    MathVector<double> diff = a - b;
    print_mathvector(diff, "a - b");

    // Multiplicação por escalar
    MathVector<double> scaled = 2.0 * a; // Usa a função livre operator*
    print_mathvector(scaled, "2 * a");

    // Vetor oposto
    MathVector<double> opposite = -a;
    print_mathvector(opposite, "-a");

    // Exemplo 2: Produto escalar (usando o método da classe)
    std::cout << "\nExemplo 2: Produto escalar\n";
    double dot_product = a.dot(b);
    std::cout << "a · b = " << dot_product << "\n";

    // Exemplo 3: Magnitude e normalização (usando métodos da classe)
    std::cout << "\nExemplo 3: Magnitude e normalização\n";
    double mag_a = a.magnitude();
    std::cout << "Magnitude de a = " << mag_a << "\n";

    try {
        MathVector<double> a_normalized = a.normalize();
        print_mathvector(a_normalized, "a normalizado");
        std::cout << "Magnitude de a normalizado = " << a_normalized.magnitude() << "\n"; // Deve ser ~1.0
    } catch (const std::domain_error& e) {
        std::cerr << "Erro ao normalizar: " << e.what() << '\n';
    }

    return 0;
}
```

### Produto Escalar

Entre as operações entre vetores vamos começar com os produtos escalares, também conhecidos como produto interno. Neste caso, temos uma técnica para multiplicar vetores de forma que o resultado seja um escalar, um número sem dimensão. Para obter o produto escalar, representado por $\cdot$, de dois vetores, multiplicamos seus elementos correspondentes e, em seguida, somamos os resultados das multiplicações. Matematicamente temos:

$$
\text{Se } \vec{a} = [a_1, a_2, ..., a_n] \text{ e } \vec{b} = [b_1, b_2, ..., b_n], \text{ então:}
$$

$$
\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + ... + a_nb_n
$$

A Figura 2 mostra um diagrama desta operação com os dois vetores que usamos acima para facilitar o entendimento.

![dois vetores representados por duas tabelas de uma linha separados por um ponto e um terceiro vetor mostrando as parcelas do produto escalar](/assets/images/dotProd1.webp)

_Figura 2: Entendendo o produto escalar entre dois vetores._{: class="legend"}

**Exemplo 1**: Considerando os vetores  $\vec{a} = [2, 5, 1]$ e $\vec{b} = [3, 1, 4]$. O produto escalar será dado por:

$$
\vec{a} \cdot \vec{b} = (2 * 3) + (5 * 1) + (1 * 4) = 6 + 5 + 4 = 15
$$

O produto escalar também pode ser representado na forma matricial. Se considerarmos os vetores como matrizes, o produto escalar será obtido multiplicando a transposta do primeiro vetor pelo segundo vetor:

$$
\vec{a} \cdot \vec{b} = \vec{a}^T\vec{b} = \begin{bmatrix} a_1 & a_2 & \cdots & a_n \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = \sum_{i=1}^{n} a_i b_i
$$

>A transposta de um vetor é uma operação da álgebra linear que altera a orientação do vetor, convertendo um vetor coluna em um vetor linha ou vice-versa. Formalmente:
>
>- Se $\vec{v}$ é um vetor coluna $\vec{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$, sua transposta $\vec{v}^T$ é um vetor linha $\vec{v}^T = \begin{pmatrix} v_1 & v_2 & \cdots & v_n \end{pmatrix}$
>
>- Se $\vec{v}$ é um vetor linha $\vec{v} = \begin{pmatrix} v_1 & v_2 & \cdots & v_n \end{pmatrix}$, sua transposta $\vec{v}^T$ é um vetor coluna $\vec{v}^T = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$
>
>Em notação matricial, se representarmos um vetor coluna $\vec{v} \in \mathbb{R}^n$ como uma matriz $n \times 1$, sua transposta $\vec{v}^T$ será uma matriz $1 \times n$. A operação de transposição é indicada pelo sobrescrito $T$.
>
>Em sistemas de processamento de linguagem natural, a transposição de vetores é frequentemente utilizada em operações de atenção e em cálculos de similaridade entre vetores de embedding.
>

Para ilustrar, considere os vetores $\vec{a} = [2, 5, 1]$ e $\vec{b} = [3, 1, 4]$. Na forma matricial, temos:

$$
\vec{a}^T\vec{b} = \begin{bmatrix} 2 & 5 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \\ 4 \end{bmatrix} = (2 \times 3) + (5 \times 1) + (1 \times 4) = 15
$$

#### Produto Escalar com Bibliotecas Numéricas

Em Python podemos usar o [JAX](https://github.com/google/jax), uma biblioteca para computação de alta performance com suporte para diferenciação automática, o produto escalar pode ser calculado usando a função `jax.numpy.dot()` ou o operador `@`.

```python
import jax.numpy as jnp

# Defina dois vetores
v1 = jnp.array([1, 2, 3])
v2 = jnp.array([4, 5, 6])

# Calcule o produto escalar usando dot()
dot_product1 = jnp.dot(v1, v2)

# Calcule o produto escalar usando o operador @
dot_product2 = v1 @ v2

print(f"Produto escalar usando dot(): {dot_product1}")
print(f"Produto escalar usando @: {dot_product2}")
```

Em C++ podemos usar o [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), uma biblioteca de álgebra linear de alto desempenho, o produto escalar pode ser calculado usando o método `dot()` ou o operador de multiplicação de matrizes.

```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
    // Defina dois vetores
    Eigen::Vector3d v1(1, 2, 3);
    Eigen::Vector3d v2(4, 5, 6);
    
    // Calcule o produto escalar usando dot()
    double dot_product1 = v1.dot(v2);
    
    // Calcule o produto escalar usando transpose e multiplicação
    double dot_product2 = v1.transpose() * v2;
    
    std::cout << "Produto escalar usando dot(): " << dot_product1 << std::endl;
    std::cout << "Produto escalar usando transpose e multiplicação: " << dot_product2 << std::endl;
    
    return 0;
}
```

#### Exemplo em C++ 20 de Produto Escalar

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <initializer_list>

// --- Definição da Classe MathVector (Necessária para este exemplo) ---
template<typename T> concept Arithmetic = std::is_arithmetic_v<T>;
template<Arithmetic T>
class MathVector {
private:
    std::vector<T> components;
    static constexpr T epsilon = 1e-9;
public:
    MathVector() = default;
    MathVector(std::initializer_list<T> init) : components(init) {}
    explicit MathVector(const std::vector<T>& vec) : components(vec) {}
    explicit MathVector(size_t n, T val = T{}) : components(n, val) {}
    size_t dimensions() const { return components.size(); }
    T& operator[](size_t index) { if (index >= dimensions()) throw std::out_of_range("..."); return components[index]; }
    const T& operator[](size_t index) const { if (index >= dimensions()) throw std::out_of_range("..."); return components[index]; }
    // Métodos essenciais: dot, magnitude, to_string (implementações omitidas por brevidade, assumindo que existem como no Bloco 1)
     T dot(const MathVector<T>& other) const {
        if (dimensions() != other.dimensions()) throw std::invalid_argument("Dimensões diferentes para dot product");
        return std::inner_product(components.begin(), components.end(), other.components.begin(), T(0));
    }
     T magnitude() const { /* ... implementação ... */
        T sum_of_squares = this->dot(*this);
        if constexpr (std::is_integral_v<T>) return static_cast<T>(std::sqrt(static_cast<double>(sum_of_squares)));
        else return std::sqrt(sum_of_squares);
     }
     std::string to_string() const { /* ... implementação ... */
        std::string result = "[";
        for (size_t i = 0; i < dimensions(); ++i) {
             if constexpr (std::is_floating_point_v<T>) { std::stringstream ss; ss << std::fixed << std::setprecision(4) << components[i]; result += ss.str(); }
             else { result += std::to_string(components[i]); }
            if (i < dimensions() - 1) result += ", ";
        }
        result += "]"; return result;
     }
     const std::vector<T>& get_components() const { return components; } // Necessário para Matrix helpers
};
// Função auxiliar para imprimir MathVector
template<Arithmetic T> void print_mathvector(const MathVector<T>& vec, const std::string& name) { std::cout << name << " = " << vec.to_string() << "\n"; }


// --- Definição da Classe Matrix (Mantida como no original) ---
template<Arithmetic T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;
public:
    Matrix(size_t n, size_t m, T initial_value = T{}) : rows(n), cols(m), data(n, std::vector<T>(m, initial_value)) {}
    Matrix(const std::vector<std::vector<T>>& values) { /* ... implementação original ... */
        if (values.empty()) { rows = 0; cols = 0; }
        else {
            rows = values.size(); cols = values[0].size();
            for (const auto& row : values) { if (row.size() != cols) throw std::invalid_argument("Linhas com tamanhos diferentes"); }
            data = values; // Copia os dados
        }
     }
    T& at(size_t i, size_t j) { if (i >= rows || j >= cols) throw std::out_of_range("..."); return data[i][j]; }
    const T& at(size_t i, size_t j) const { if (i >= rows || j >= cols) throw std::out_of_range("..."); return data[i][j]; }
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    Matrix<T> operator*(const Matrix<T>& other) const { /* ... implementação original ... */
        if (cols != other.rows) throw std::invalid_argument("Dimensões incompatíveis");
        Matrix<T> result(rows, other.cols, T{});
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) { result.at(i, j) += data[i][k] * other.data[k][j]; }
            }
        } return result;
     }
    Matrix<T> transpose() const { /* ... implementação original ... */
         Matrix<T> result(cols, rows);
         for (size_t i = 0; i < rows; ++i) { for (size_t j = 0; j < cols; ++j) { result.at(j, i) = data[i][j]; } }
         return result;
     }
    void print(const std::string& name = "") const { /* ... implementação original ... */
         if (!name.empty()) std::cout << name << " =\n";
         for (size_t i = 0; i < rows; ++i) {
             std::cout << "[";
             for (size_t j = 0; j < cols; ++j) { std::cout << std::fixed << std::setprecision(2) << data[i][j]; if (j < cols - 1) std::cout << ", "; }
             std::cout << "]\n";
         }
     }
    T scalar_value() const { if (rows != 1 || cols != 1) throw std::logic_error("Matrix não é 1x1"); return data[0][0]; }

    // Cria um vetor coluna (Matrix) a partir de um MathVector
    static Matrix<T> column_vector_from(const MathVector<T>& vec) {
        Matrix<T> result(vec.dimensions(), 1);
        for (size_t i = 0; i < vec.dimensions(); ++i) {
            result.at(i, 0) = vec[i];
        }
        return result;
    }
    // Cria um vetor linha (Matrix) a partir de um MathVector
    static Matrix<T> row_vector_from(const MathVector<T>& vec) {
        Matrix<T> result(1, vec.dimensions());
        for (size_t i = 0; i < vec.dimensions(); ++i) {
            result.at(0, i) = vec[i];
        }
        return result;
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Demonstração de Produto Escalar e Interpretação usando MathVector\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    MathVector<double> a = {2.0, 5.0, 1.0};
    MathVector<double> b = {3.0, 1.0, 4.0};

    print_mathvector(a, "Vetor a");
    print_mathvector(b, "Vetor b");

    // --- Cálculo Principal usando MathVector::dot ---
    std::cout << "\nCálculo principal via a.dot(b):\n";
    double dot_value_direct = a.dot(b);
    std::cout << "a · b = " << dot_value_direct << "\n";

    // --- Visão Alternativa: Representação Matricial (a^T * b) ---
    std::cout << "\nVisão Alternativa: Representação Matricial (a^T * b):\n";
    try {
        auto a_row = Matrix<double>::row_vector_from(a); // a transposto como matriz linha
        auto b_col = Matrix<double>::column_vector_from(b); // b como matriz coluna

        a_row.print("a como vetor linha (a^T)");
        b_col.print("b como vetor coluna (b)");

        auto dot_result_matrix = a_row * b_col; // Multiplicação de matrizes
        dot_result_matrix.print("\nProduto (a^T * b)");

        double dot_value_matrix = dot_result_matrix.scalar_value(); // Extrai o valor escalar
        std::cout << "Valor do produto escalar (via matriz): " << dot_value_matrix << "\n";
        std::cout << "Verificação manual: " << (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Erro na representação matricial: " << e.what() << '\n';
    }


    // --- Exemplo 2: Extração de componente usando MathVector::dot ---
    std::cout << "\nExemplo 2: Extração da segunda componente de [0.2, 0.7, 0.1] usando dot product\n";
    MathVector<double> base_extract = {0.0, 1.0, 0.0};
    MathVector<double> data_vec = {0.2, 0.7, 0.1};
    print_mathvector(base_extract, "Vetor base (ativa 2a dim)");
    print_mathvector(data_vec, "Vetor de dados");
    double extracted_value = base_extract.dot(data_vec);
    std::cout << "Resultado da extração (base · data): " << extracted_value << "\n";
    std::cout << "Verificação: data[1] = " << data_vec[1] << "\n";


    // --- Exemplo 3: Interpretação geométrica do produto escalar usando MathVector::dot ---
    std::cout << "\nExemplo 3: Interpretação geométrica do produto escalar\n";

    // Vetores similares
    MathVector<double> v_sim1 = {0.8, 0.6};
    MathVector<double> v_sim2 = {0.9, 0.5};
    print_mathvector(v_sim1, "v_sim1");
    print_mathvector(v_sim2, "v_sim2");
    std::cout << "Produto escalar (similares): " << v_sim1.dot(v_sim2) << " (positivo)\n\n";

    // Vetores ortogonais
    MathVector<double> v_ort1 = {1.0, 0.0};
    MathVector<double> v_ort2 = {0.0, 1.0};
    print_mathvector(v_ort1, "v_ort1");
    print_mathvector(v_ort2, "v_ort2");
    std::cout << "Produto escalar (ortogonais): " << v_ort1.dot(v_ort2) << " (zero)\n\n";

    // Vetores opostos
    MathVector<double> v_op1 = {0.7, 0.3};
    MathVector<double> v_op2 = {-0.7, -0.3};
    print_mathvector(v_op1, "v_op1");
    print_mathvector(v_op2, "v_op2");
    std::cout << "Produto escalar (opostos): " << v_op1.dot(v_op2) << " (negativo)\n";

    return 0;
}
```

#### O Produto Escalar e a Similaridade

O produto escalar oferece uma medida quantitativa da similaridade direcional entre dois vetores. Embora não constitua uma métrica de similaridade completa em todos os contextos, fornece informações valiosas sobre o alinhamento vetorial. Em termos gerais, a interpretação do produto escalar $\vec{u} \cdot \vec{v}$ segue estas propriedades:

$$
\vec{u} \cdot \vec{v} = \vert\vec{u}\vert \cdot \vert\vec{v}\vert \cdot \cos(\theta)
$$

Onde $\theta$ representa o ângulo entre os vetores, e podemos observar que:

- $\vec{u} \cdot \vec{v} > 0$: Os vetores apontam em direções geralmente similares (ângulo agudo). Quanto maior o valor positivo, maior a similaridade em termos de direção e magnitude das componentes que se alinham.
- $\vec{u} \cdot \vec{v} = 0$: Os vetores são ortogonais (perpendiculares). Não há similaridade direcional linear entre eles.
- $\vec{u} \cdot \vec{v} < 0$: Os vetores apontam em direções geralmente opostas (ângulo obtuso). Quanto mais negativo, maior a dissimilaridade direcional.

![imagem mostrando três vetores exemplificando os resultados do produto escalar](/assets/images/produto-escalar1.webp)

*Para vetores normalizados (de magnitude unitária), o produto escalar se reduz diretamente ao cosseno do ângulo entre eles, fornecendo uma medida de similaridade no intervalo $[-1, 1]$, frequentemente utilizada em sistemas de recuperação de informação e processamento de linguagem natural.*

**Exemplo 2**: considerando os vetores $\vec{a} = [0, 1, 0]$ e $\vec{b} = [0.2, 0.7, 0.1]$, o produto escalar será:

$$
\vec{a} \cdot \vec{b} = (0 * 0.2) + (1 * 0.7) + (0 * 0.1) = 0 + 0.7 + 0 = 0.7
$$

No exemplo 2, o vetor $\vec{a} = [0, 1, 0]$ pode ser visto como um vetor que *ativa*, ou dá peso máximo, apenas à segunda dimensão, e peso zero às demais. Ao calcular o produto escalar com $\vec{b} = [0.2, 0.7, 0.1]$, estamos essencialmente *extraindo ou medindo* o valor da segunda componente de $b$ (que é $0.7$), ponderado pela *importância, ou peso* que o vetor $a$ atribui a essa dimensão.

Com um pouco mais de formalidade: se temos dois vetores $\vec{u}$ e $\vec{v}$, e você calcula $\vec{u} \cdot \vec{v} = c$, o valor escalar $c$ pode ser interpretado como uma medida de:

- Quanto de $\vec{v}$ "existe" na direção de $\vec{u}$ (e vice-versa);
- O grau de alinhamento ou sobreposição entre os vetores;
- A similaridade entre os padrões representados pelos vetores, no sentido de que componentes importantes em um vetor também são relevantes no outro, com pesos proporcionais aos valores das componentes.

A criativa leitora deve notar que o produto escalar é influenciado tanto pela direção quanto pela magnitude dos vetores.

>*A magnitude de um vetor é dada pela raiz quadrada da soma dos quadrados dos seus componentes*. Isso é equivalente a tirar a raiz quadrada do resultado do produto escalar do vetor com ele mesmo. Para um vetor
>
>$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$
>
>em um espaço $n$-dimensional, a magnitude, eventualmente chamada de *Norma Euclidiana*, representada por $\vert \vec{v}\vert$ será definida por:
>
>$$
\vert \vec{v}\vert  = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

**Exemplo 3**: dado o vetor $\vec{b} = \begin{bmatrix} 0.2 \\ 0.7 \\ 0.1 \end{bmatrix}$, vamos calcular sua magnitude $\vert \vec{b}\vert$:

Podemos resolver este problema em dois passos:

1. **Calcular o produto escalar de $\vec{b}$ consigo mesmo:**

   $$
   \vec{b} \cdot \vec{b} = (0.2 \times 0.2) + (0.7 \times 0.7) + (0.1 \times 0.1) = 0.04 + 0.49 + 0.01 = 0.54
   $$

2. **Extrair a raiz quadrada do resultado:**

   $$
   \vert \vec{b}\vert  = \sqrt{\vec{b} \cdot \vec{b}} = \sqrt{0.54} \approx 0.7348
   $$

Portanto, a magnitude do vetor $\vec{b} = \begin{bmatrix} 0.2 \\ 0.7 \\ 0.1 \end{bmatrix}$ é aproximadamente 0.7348.

A magnitude, pode ter interpretações diferentes em áreas diferentes do conhecimento. Na física, pode representar a intensidade de uma força, ou uma velocidade. No estudo da linguagem natural, a magnitude de um vetor, pode indicar o tamanho do documento em termos de número de palavras, embora não diretamente.

*A atenta leitora deve observar que vetores com magnitudes maiores tendem a ter produtos escalares maiores, mesmo que a direção relativa seja a mesma.*

Com as definições de produto escalar ($\vec{u} \cdot \vec{v}$) e magnitude ($\vert\vec{u}\vert$, $\vert\vec{v}\vert$) em mãos, podemos reorganizar a relação fundamental $\vec{u} \cdot \vec{v} = \vert\vec{u}\vert \cdot \vert\vec{v}\vert \cdot \cos(\theta)$ para isolar o cosseno do ângulo $\theta$. Isso nos fornece diretamente a fórmula da **Similaridade de Cosseno**, uma das métricas mais importantes em processamento de linguagem natural e recuperação de informação para medir a similaridade direcional entre dois vetores:

$$
\text{Similaridade de Cosseno}(\vec{u}, \vec{v}) = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\vert\vec{u}\vert \cdot \vert\vec{v}\vert}
$$

Esta métrica varia no intervalo $[-1, 1]$:
* $1$: Indica que os vetores apontam exatamente na mesma direção (ângulo $0^\circ$).
* $0$: Indica que os vetores são ortogonais (perpendiculares, ângulo $90^\circ$).
* $-1$: Indica que os vetores apontam em direções exatamente opostas (ângulo $180^\circ$).

A grande vantagem da Similaridade de Cosseno é que ela foca puramente na direção dos vetores, ignorando suas magnitudes. Isso é particularmente útil ao comparar documentos de tamanhos diferentes ou *word embeddings*, onde a orientação no espaço semântico é mais importante que a magnitude absoluta do vetor.

Quando estudamos processamento de linguagem natural, a magnitude por si não costuma ser a informação mais importante, ou mais buscada. Geralmente estamos interessados na direção de vetores e na similaridade entre eles.

>A similaridade entre vetores é uma medida de quão semelhantes são dois vetores em termos de direção e magnitude. Essa medida é fundamental em muitas aplicações, como recuperação de informação, recomendação de produtos e análise de sentimentos.

Em alguns casos, a busca da similaridade implica na normalização dos vetores para que a medida de similaridade seja mais afetada pela direção e menos afetada pela magnitude.

>A normalização de um vetor $\vec{v}$ consiste em dividi-lo por sua norma (ou magnitude), resultando em um vetor unitário $\hat{v}$ que mantém a mesma direção, mas possui comprimento 1:
>
>$$\hat{v} = \frac{\vec{v}}{\vert\vec{v}\vert} = \frac{\vec{v}}{\sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}}$$
>
>Neste caso, $\vert\vec{v}\vert$ representa a norma euclidiana do vetor. Quando dois vetores normalizados são comparados através do produto escalar, o resultado varia apenas entre $-1$ e $1$, correspondendo diretamente ao cosseno do ângulo entre eles. Esta abordagem é particularmente útil em aplicações como recuperação de informações, sistemas de recomendação e processamento de linguagem natural, onde a orientação semântica dos vetores é geralmente mais relevante que suas magnitudes absolutas.

>A norma euclidiana, também conhecida como norma $L_2$ ou comprimento euclidiano, é uma função que atribui a cada vetor um valor escalar não-negativo que pode ser interpretado como o "tamanho" ou "magnitude" do vetor. Para um vetor $\vec{v} = (v_1, v_2, \ldots, v_n)$ em $\mathbb{R}^n$, a norma euclidiana é definida como:
>
>$$\vert\vec{v}\vert = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}$$
>
>Esta definição é uma generalização do teorema de Pitágoras para espaços de dimensão arbitrária. Geometricamente, representa a distância do ponto representado pelo vetor à origem do sistema de coordenadas.
>
>A norma euclidiana possui as seguintes propriedades fundamentais:
>
>1. **Não-negatividade**: $\vert\vec{v}\vert \geq 0$ para todo $\vec{v}$, e $\vert\vec{v}\vert = 0$ se e somente se $\vec{v} = \vec{0}$
>2. **Homogeneidade**: $\vert\alpha\vec{v}\vert = |\alpha| \cdot \vert\vec{v}\vert$ para qualquer escalar $\alpha$
>3. **Desigualdade triangular**: $\vert\vec{u} + \vec{v}\vert \leq \vert\vec{u}\vert + \vert\vec{v}\vert$
>
>Estas propriedades fazem da norma euclidiana uma ferramenta essencial em diversos campos, desde geometria e física até aprendizado de máquina e processamento de sinais, onde é utilizada para medir distâncias, calcular erros, e normalizar vetores.

Técnicas como a **Similaridade de Cosseno**, que envolve o produto escalar normalizado pelas magnitudes do vetores, são usadas para isolar a similaridade direcional. Todavia, existem diversas técnicas diferentes para a determinação de um índice para a similaridade entre vetores. Entre elas destacaremos:

1. **Distância Euclidiana**: mede a distância "direta" entre dois pontos no espaço euclidiano. É sensível tanto à direção quanto à magnitude dos vetores.

2. **Distância de Manhattan**: também conhecida como distância $L1$, mede a soma das diferenças absolutas das coordenadas dos vetores.

3. **Distância de Minkowski**: uma generalização das distâncias Euclidiana e de Manhattan, onde a ordem pode ser ajustada para diferentes tipos de distâncias.

4. **Similaridade de Jaccard**: usada principalmente para conjuntos, mede a similaridade como a razão do tamanho da interseção para o tamanho da união dos conjuntos.

5. **Correlação de Pearson**: mede a correlação linear entre dois vetores, variando de $-1$ a $1$. É útil para entender a relação linear entre os componentes dos vetores.

6. **Distância de Mahalanobis**: considera a correlação entre variáveis e é útil quando os vetores têm diferentes escalas ou distribuições.

Algumas já foram usadas processamento de linguagem natural. Outras ainda não. Vamos trabalhar com cada uma delas se, e quando, forem usadas nos algoritmos que estudaremos. A primeira que usaremos será a **Similaridade de Cosseno**. Mas antes, precisamos entender como representar textos como vetores. Para isso, vamos começar com a representação mais simples e intuitiva: a **Frequência de Termos**.

#### Exemplo em C++

```cpp

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <initializer_list>
#include <algorithm> // Para std::abs

// --- Definição da Classe MathVector (Necessária para este exemplo) ---
template<typename T> concept Arithmetic = std::is_arithmetic_v<T>;
template<Arithmetic T>
class MathVector {
private:
    std::vector<T> components;
    static constexpr T epsilon = 1e-9;
public:
    // Construtores, dimensions(), operator[], dot(), magnitude(), normalize(), to_string()
    // (Implementações omitidas por brevidade - assumir como no Bloco 1)
     MathVector() = default;
    MathVector(std::initializer_list<T> init) : components(init) {}
    explicit MathVector(const std::vector<T>& vec) : components(vec) {}
    explicit MathVector(size_t n, T val = T{}) : components(n, val) {}
    size_t dimensions() const { return components.size(); }
    T& operator[](size_t index) { if (index >= dimensions()) throw std::out_of_range("..."); return components[index]; }
    const T& operator[](size_t index) const { if (index >= dimensions()) throw std::out_of_range("..."); return components[index]; }
     T dot(const MathVector<T>& other) const {
        if (dimensions() != other.dimensions()) throw std::invalid_argument("Dimensões diferentes para dot product");
        return std::inner_product(components.begin(), components.end(), other.components.begin(), T(0));
    }
     T magnitude() const {
        T sum_of_squares = this->dot(*this);
        if constexpr (std::is_integral_v<T>) return static_cast<T>(std::sqrt(static_cast<double>(sum_of_squares)));
        else return std::sqrt(sum_of_squares);
     }
      MathVector<T> normalize() const {
        T mag = magnitude();
        if (std::abs(mag) < epsilon) throw std::domain_error("Magnitude zero para normalização");
        MathVector<T> result(dimensions());
        for (size_t i = 0; i < dimensions(); ++i) result[i] = components[i] / mag;
        return result;
     }
     std::string to_string() const {
        std::string result = "[";
        for (size_t i = 0; i < dimensions(); ++i) {
             if constexpr (std::is_floating_point_v<T>) { std::stringstream ss; ss << std::fixed << std::setprecision(4) << components[i]; result += ss.str(); }
             else { result += std::to_string(components[i]); }
            if (i < dimensions() - 1) result += ", ";
        }
        result += "]"; return result;
     }
};

// Função auxiliar para imprimir MathVector
template<Arithmetic T> void print_mathvector(const MathVector<T>& vec, const std::string& name) { std::cout << name << " = " << vec.to_string() << "\n"; }

template<Arithmetic T>
class VectorSimilarity {
private:
    // Verifica se os vetores têm a mesma dimensão
    static bool check_dimensions(const MathVector<T>& a, const MathVector<T>& b) {
        return a.dimensions() == b.dimensions();
    }
    // Constante pequena para comparações de ponto flutuante
    static constexpr T epsilon = 1e-9;

public:
    // Produto escalar (reutiliza MathVector::dot)
    static T dot_product(const MathVector<T>& a, const MathVector<T>& b) {
        // MathVector::dot já verifica dimensões
        return a.dot(b);
    }

    // Similaridade de cosseno
    static T cosine_similarity(const MathVector<T>& a, const MathVector<T>& b) {
        if (!check_dimensions(a, b)) { // Verifica dimensões aqui por clareza
           throw std::invalid_argument("Vetores com dimensões diferentes para similaridade de cosseno");
        }
        T mag_a = a.magnitude();
        T mag_b = b.magnitude();
        if (std::abs(mag_a) < epsilon || std::abs(mag_b) < epsilon) {
             throw std::domain_error("Magnitude zero (ou próxima) em cálculo de similaridade de cosseno");
        }
        T dot = a.dot(b);
        return dot / (mag_a * mag_b);
    }

    // Distância euclidiana
    static T euclidean_distance(const MathVector<T>& a, const MathVector<T>& b) {
        if (!check_dimensions(a, b)) {
            throw std::invalid_argument("Vetores com dimensões diferentes para distância Euclidiana");
        }
        T sum_of_squares = 0;
        for (size_t i = 0; i < a.dimensions(); ++i) {
            T diff = a[i] - b[i];
            sum_of_squares += diff * diff;
        }
        if constexpr (std::is_integral_v<T>) { return static_cast<T>(std::sqrt(static_cast<double>(sum_of_squares))); }
        else { return std::sqrt(sum_of_squares); }
    }

    // Distância de Manhattan (L1)
    static T manhattan_distance(const MathVector<T>& a, const MathVector<T>& b) {
        if (!check_dimensions(a, b)) {
            throw std::invalid_argument("Vetores com dimensões diferentes para distância de Manhattan");
        }
        T sum_of_abs_diff = 0;
        for (size_t i = 0; i < a.dimensions(); ++i) {
            sum_of_abs_diff += std::abs(a[i] - b[i]); // Usa std::abs
        }
        return sum_of_abs_diff;
    }

    // Normalização de um vetor (reutiliza MathVector::normalize)
    static MathVector<T> normalize(const MathVector<T>& vec) {
        // MathVector::normalize já verifica magnitude zero
        return vec.normalize();
    }
};

int main() {
    // Configurar a precisão
    std::cout << std::fixed << std::setprecision(4);

    // Exemplo 1: Vetores alinhados (similares)
    MathVector<double> u1 = {0.5, 0.8, 0.3};
    MathVector<double> v1 = {0.6, 0.9, 0.2}; // Alterado para não ser múltiplo exato
    std::cout << "Exemplo 1: Vetores geralmente alinhados (similares)\n";
    print_mathvector(u1, "u1");
    print_mathvector(v1, "v1");
    std::cout << "Produto escalar: " << VectorSimilarity<double>::dot_product(u1, v1) << "\n";
    std::cout << "Similaridade de cosseno: " << VectorSimilarity<double>::cosine_similarity(u1, v1) << "\n";
    std::cout << "Distância euclidiana: " << VectorSimilarity<double>::euclidean_distance(u1, v1) << "\n";
    std::cout << "Distância de Manhattan: " << VectorSimilarity<double>::manhattan_distance(u1, v1) << "\n";

    // Exemplo 2: Vetores ortogonais (perpendiculares)
    MathVector<double> u2 = {1.0, 0.0, 0.0};
    MathVector<double> v2 = {0.0, 1.0, 0.0};
    std::cout << "\nExemplo 2: Vetores ortogonais (perpendiculares)\n";
    print_mathvector(u2, "u2");
    print_mathvector(v2, "v2");
    std::cout << "Produto escalar: " << VectorSimilarity<double>::dot_product(u2, v2) << "\n";
    std::cout << "Similaridade de cosseno: " << VectorSimilarity<double>::cosine_similarity(u2, v2) << "\n";
    std::cout << "Distância euclidiana: " << VectorSimilarity<double>::euclidean_distance(u2, v2) << "\n";
    std::cout << "Distância de Manhattan: " << VectorSimilarity<double>::manhattan_distance(u2, v2) << "\n";

    // Exemplo 3: Vetores em direções opostas
    MathVector<double> u3 = {0.7, 0.2, -0.3};
    MathVector<double> v3 = {-0.7, -0.2, 0.3}; // Exatamente opostos
    std::cout << "\nExemplo 3: Vetores em direções opostas\n";
    print_mathvector(u3, "u3");
    print_mathvector(v3, "v3");
    std::cout << "Produto escalar: " << VectorSimilarity<double>::dot_product(u3, v3) << "\n";
    std::cout << "Similaridade de cosseno: " << VectorSimilarity<double>::cosine_similarity(u3, v3) << "\n";
    std::cout << "Distância euclidiana: " << VectorSimilarity<double>::euclidean_distance(u3, v3) << "\n";
    std::cout << "Distância de Manhattan: " << VectorSimilarity<double>::manhattan_distance(u3, v3) << "\n";

    // Exemplo 4: Extração de componente (já demonstrado no Bloco 2, mas pode repetir se quiser)
    // MathVector<double> base = {0.0, 1.0, 0.0};
    // MathVector<double> data = {0.2, 0.7, 0.1};
    // std::cout << "\nExemplo 4: Extração de componente usando produto escalar\n";
    // print_mathvector(base, "base");
    // print_mathvector(data, "data");
    // std::cout << "Produto escalar (extrai valor): " << VectorSimilarity<double>::dot_product(base, data) << "\n";

    // Exemplo 5: Normalização usando VectorSimilarity
    MathVector<double> original = {0.2, 0.7, 0.1};
    std::cout << "\nExemplo 5: Normalização de vetores via VectorSimilarity\n";
    print_mathvector(original, "original");
    try {
        MathVector<double> normalized = VectorSimilarity<double>::normalize(original);
        print_mathvector(normalized, "normalizado");
        std::cout << "Magnitude do vetor original: " << original.magnitude() << "\n";
        std::cout << "Magnitude do vetor normalizado: " << normalized.magnitude() << "\n"; // Deve ser ~1.0
    } catch (const std::domain_error& e) {
         std::cerr << "Erro ao normalizar: " << e.what() << '\n';
    }

    return 0;
}
```
Agora que conhecemos o produto escalar, podemos nos aprofundar na matemática que é o coração do processamento de linguagem natural: a multiplicação de matrizes.

### Multiplicação de Matrizes

A multiplicação de matrizes é uma operação fundamental na álgebra linear que aparece constantemente nos modelos de **transformers**. Esta operação irá permitir a combinação de diferentes fontes de textos e transformar representações vetoriais, formando a base de diversas operações nos modelos de processamento de linguagem natural.

A multiplicação de matrizes é uma operação que combina duas matrizes para produzir uma nova matriz. É importante notar que *a multiplicação de matrizes não é comutativa, ou seja, a ordem das matrizes importa*. A multiplicação de matrizes é definida como o produto escalar entre as linhas da primeira matriz e as colunas da segunda matriz.

Formalmente dizemos: sejam $A$ uma matriz de dimensão $m \times n$ e $B$ uma matriz de dimensão $n \times p$. O produto $A \times B$ resultará em uma matriz $C$ de dimensão $m \times p$, onde cada elemento $c_{ij}$ é determinado pelo produto escalar da $i$-ésima linha de $A$ com a $j$-ésima coluna de $B$:

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}
$$

Observe, atenta leitora, que para que a multiplicação de matrizes seja possível, *o número de colunas da primeira matriz deve ser igual ao número de linhas da segunda matriz*. Esta restrição não é arbitrária - ela garante que os produtos escalares entre linhas e colunas sejam bem definidos.

![matriz A multiplicada por matriz B resultando em matriz C](/assets/images/matrix_mult1.webp)

_Figura 3: Visualização da multiplicação de matrizes. Cada elemento $c_{ij}$ da matriz resultante é obtido pelo produto escalar da linha $i$ da matriz $A$ com a coluna $j$ da matriz $B$._{: class="legend"}

Nos modelos **transformer**, a multiplicação de matrizes ocorre com frequência em várias etapas, como:

1. **Atenção**: O mecanismo de atenção utiliza multiplicações de matrizes para calcular as representações de query, key e value.
2. **Embedding de Tokens**: Transformação de tokens discretos em vetores contínuos de alta dimensão.
3. **Projeções Lineares**: Transformações dos vetores de query, key e value no mecanismo de atenção.
4. **Feed-Forward Networks**: Camadas densas que aplicam transformações não-lineares às representações.
5. **Projeções de Saída**: Mapeamento das representações finais para o espaço de saída desejado.

A eficiência dos modelos **transformers** deve-se, em parte, à capacidade de paralelizar estas multiplicações de matrizes em hardware especializado, como GPUs e TPUs.

#### Propriedades Importantes

A multiplicação de matrizes possui algumas propriedades notáveis que a diferenciam da multiplicação de números reais:

1. **Não comutativa**: em geral, $A \times B \neq B \times A$. A ordem das operações importa.
2. **Associativa**: $(A \times B) \times C = A \times (B \times C)$. Podemos calcular multiplicações sucessivas em qualquer ordem.
3. **Distributiva sobre a adição**: $A \times (B + C) = A \times B + A \times C$.
4. **Elemento neutro**: $A \times I = I \times A = A$, onde $I$ é a matriz identidade de dimensão apropriada.

#### Interpretação Geométrica

Geometricamente, a multiplicação por uma matriz pode ser vista como uma transformação linear no espaço vetorial. Estas transformações podem incluir: rotações, mudança de escala, reflexões, cisalhamentos e projeções. Dependendo da matriz, a transformação pode alterar a posição, a forma ou a orientação dos vetores no espaço.

Nos **transformers**, estas transformações são aplicadas para mapear representações vetoriais de um espaço para outro, permitindo que a rede aprenda relações complexas entre os elementos da sequência de entrada.

#### Exemplo Numérico

Nada como um exemplo numérico para fazer a esforçada leitora balançar a poeira. Vamos consider as duas matrizes, $A$ e $B$:

$$
A = \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} \quad \text{e} \quad B = \begin{bmatrix} 1 & 5 \\ 2 & 3 \end{bmatrix}
$$

O produto $C = A \times B$ será:

$$
\begin{align}
c_{11} &= a_{11} \cdot b_{11} + a_{12} \cdot b_{21} = 2 \times 1 + 3 \times 2 = 2 + 6 = 8 \\
c_{12} &= a_{11} \cdot b_{12} + a_{12} \cdot b_{22} = 2 \times 5 + 3 \times 3 = 10 + 9 = 19 \\
c_{21} &= a_{21} \cdot b_{11} + a_{22} \cdot b_{21} = 4 \times 1 + 1 \times 2 = 4 + 2 = 6 \\
c_{22} &= a_{21} \cdot b_{12} + a_{22} \cdot b_{22} = 4 \times 5 + 1 \times 3 = 20 + 3 = 23
\end{align}
$$

Portanto:

$$
C = A \times B = \begin{bmatrix} 8 & 19 \\ 6 & 23 \end{bmatrix}
$$

#### Multiplicação Matriz-Vetor

Um caso especial e extremamente importante, para nossos objetivos, é a multiplicação de uma matriz por um vetor. A perceptiva leitora há de considerar que *um vetor que pode ser visto como uma matriz com apenas uma coluna*. Esta operação é recorrente em praticamente todas as camadas de um modelo **transformer**.

Seja $A$ uma matriz $m \times n$ e $\vec{v}$ um vetor coluna de dimensão $n$. O produto $A\vec{v}$ resulta em um vetor coluna $\vec{w}$ de dimensão $m$:

$$
\vec{w} = A\vec{v} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = 
\begin{bmatrix} 
\sum_{j=1}^{n} a_{1j}v_j \\ 
\sum_{j=1}^{n} a_{2j}v_j \\ 
\vdots \\ 
\sum_{j=1}^{n} a_{mj}v_j
\end{bmatrix}
$$

Cada componente $w_i$ do vetor resultante é o produto escalar da $i$-ésima linha da matriz $A$ com o vetor $\vec{v}$. Este padrão de operação repete-se continuamente no mecanismo de atenção dos *transformers*.

**Exemplo**: Considerando a matriz $A$ definida acima e o vetor $\vec{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$, temos:

$$
A\vec{v} = \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} (2 \times 3) + (3 \times 2) \\ (4 \times 3) + (1 \times 2) \end{bmatrix} = \begin{bmatrix} 6 + 6 \\ 12 + 2 \end{bmatrix} = \begin{bmatrix} 12 \\ 14 \end{bmatrix}
$$

#### Exemplo em C++ 20

```cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    // Construtor para matriz de dimensões m x n com valor inicial
    Matrix(size_t m, size_t n, T initial_value = T{}) 
        : rows(m), cols(n), data(m, std::vector<T>(n, initial_value)) {}
    
    // Construtor a partir de um vector de vectors
    Matrix(const std::vector<std::vector<T>>& values) {
        if (values.empty()) {
            rows = 0;
            cols = 0;
            return;
        }
        
        rows = values.size();
        cols = values[0].size();
        
        // Verifica se todas as linhas têm o mesmo tamanho
        for (const auto& row : values) {
            if (row.size() != cols) {
                throw std::invalid_argument("Todas as linhas devem ter o mesmo número de colunas");
            }
        }
        
        data = values; // Copia os dados
    }
    
    // Acesso aos elementos (com verificação de limites)
    T& at(size_t i, size_t j) {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Índices fora dos limites da matriz");
        }
        return data[i][j];
    }
    
    const T& at(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Índices fora dos limites da matriz");
        }
        return data[i][j];
    }
    
    // Obter dimensões
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    
    // Multiplicação de matrizes
    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Dimensões incompatíveis para multiplicação de matrizes");
        }
        
        Matrix<T> result(rows, other.cols, T{});
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        
        return result;
    }
    
    // Multiplicação por vetor (representado como matriz coluna)
    std::vector<T> multiply_vector(const std::vector<T>& vec) const {
        if (cols != vec.size()) {
            throw std::invalid_argument("Dimensões incompatíveis para multiplicação matriz-vetor");
        }
        
        std::vector<T> result(rows, T{});
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }
        
        return result;
    }
    
    // Impressão formatada da matriz
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << " =\n";
        }
        
        for (size_t i = 0; i < rows; ++i) {
            std::cout << "[";
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::fixed << std::setprecision(2) << data[i][j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Demonstração de Multiplicação de Matrizes\n";
    std::cout << "----------------------------------------\n\n";
    
    // Exemplo 1: Multiplicação de duas matrizes
    Matrix<double> A({
        {2.0, 3.0},
        {4.0, 1.0}
    });
    
    Matrix<double> B({
        {1.0, 5.0},
        {2.0, 3.0}
    });
    
    std::cout << "Exemplo 1: Multiplicação de duas matrizes\n";
    A.print("Matriz A");
    B.print("Matriz B");
    
    try {
        Matrix<double> C = A * B;
        C.print("A * B");
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
    }
    
    // Exemplo 2: Multiplicação matriz-vetor
    std::vector<double> v = {3.0, 2.0};
    
    std::cout << "\nExemplo 2: Multiplicação matriz-vetor\n";
    A.print("Matriz A");
    std::cout << "Vetor v = [" << v[0] << ", " << v[1] << "]\n";
    
    try {
        std::vector<double> result = A.multiply_vector(v);
        std::cout << "A * v = [";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i];
            if (i < result.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
    }
    
    // Exemplo 3: Demonstração de erro (dimensões incompatíveis)
    Matrix<double> D({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });
    
    std::cout << "\nExemplo 3: Tentativa de multiplicação com dimensões incompatíveis\n";
    A.print("Matriz A (2x2)");
    D.print("Matriz D (2x3)");
    
    try {
        Matrix<double> E = D * A; // Isso deve falhar (3 colunas × 2 linhas)
        E.print("D * A");
    } catch (const std::exception& e) {
        std::cout << "Erro (esperado): " << e.what() << '\n';
    }
    
    return 0;
}

Ou, como a preocupada leitora pode preferir, em C++ 20 usando a biblioteca Eigen: 

```cpp
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <string>
#include <stdexcept>

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Demonstração de Multiplicação de Matrizes\n";
    std::cout << "----------------------------------------\n\n";
    
    // Exemplo 1: Multiplicação de duas matrizes
    Eigen::Matrix2d A;
    A << 2.0, 3.0,
         4.0, 1.0;
    
    Eigen::Matrix2d B;
    B << 1.0, 5.0,
         2.0, 3.0;
    
    std::cout << "Exemplo 1: Multiplicação de duas matrizes\n";
    std::cout << "Matriz A =\n" << A << "\n\n";
    std::cout << "Matriz B =\n" << B << "\n\n";
    
    try {
        Eigen::Matrix2d C = A * B;
        std::cout << "A * B =\n" << C << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
    }
    
    // Exemplo 2: Multiplicação matriz-vetor
    Eigen::Vector2d v;
    v << 3.0, 2.0;
    
    std::cout << "Exemplo 2: Multiplicação matriz-vetor\n";
    std::cout << "Matriz A =\n" << A << "\n\n";
    std::cout << "Vetor v =\n" << v << "\n\n";
    
    try {
        Eigen::Vector2d result = A * v;
        std::cout << "A * v =\n" << result << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
    }
    
    // Exemplo 3: Demonstração de erro (dimensões incompatíveis)
    Eigen::MatrixXd D(2, 3);
    D << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    
    std::cout << "Exemplo 3: Tentativa de multiplicação com dimensões incompatíveis\n";
    std::cout << "Matriz A (2x2) =\n" << A << "\n\n";
    std::cout << "Matriz D (2x3) =\n" << D << "\n\n";
    
    try {
        // Verificação explícita de dimensões (para ser didático)
        if (D.cols() != A.rows()) {
            throw std::invalid_argument("Dimensões incompatíveis: D é 2x3 e A é 2x2");
        }
        
        // Eigen lançará uma asserção estática aqui se compilado com verificações
        Eigen::MatrixXd E = D * A;
        std::cout << "D * A =\n" << E << "\n\n";
    } catch (const std::exception& e) {
        std::cout << "Erro (esperado): " << e.what() << '\n';
    }
    
    return 0;
}
```

Agora que vimos o básico da matemática, a vetorização de textos será o tema do próximo artigo.
