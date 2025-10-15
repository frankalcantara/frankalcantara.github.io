---
layout: post
title: Heaps na Standard Template Library do C++23
author: Frank
categories: |-
    Matemática
    artigo
tags: |-
    Algoritmos
    C++
    computação competitiva
    Estruturas de Dados
    Heaps
image: null
featured: false
rating: 5
description: Explicando a estrutura Heap como implementada em C++23.
date: 2025-10-14T11:21:10.761Z
preview: Um estudo detalhado da estrutura heap em c++ como base para um artigo do livro sobre computação competitiva
keywords: |-
    Algoritmos
    computação competitiva
toc: true
published: true
lastmod: 2025-10-14T11:46:43.911Z
draft: 2025-10-14T11:22:19.148Z
---

> Precisa de revisão, esta é uma versão preliminar a partir de um texto antigo em C++ 20. Esta versão foi verificada com o auxílio do Gemini 2.4, Claude 4.5 e Grok 4.1 em 10 de outubro de 2025.

A estrutura de dados **heap** (ou monte) é uma árvore binária especializada que satisfaz a "propriedade do heap". Esta estrutura é fundamental para diversas aplicações em ciência da computação, incluindo algoritmos de ordenação, filas de prioridade, e algoritmos de grafos como Dijkstra e Prim.

### Tipos de Heap

**Max-Heap**: Para qualquer nó `i` diferente da raiz, o valor de `i` é menor ou igual ao valor de seu pai. Isso garante que o elemento de maior valor esteja sempre na raiz da árvore.

**Min-Heap**: A lógica é invertida. Para qualquer nó `i` diferente da raiz, o valor de `i` é maior ou igual ao valor de seu pai, garantindo que o menor elemento esteja na raiz.

### Implementação na STL

A STL do C++ não fornece um contêiner `std::heap` dedicado. Em vez disso, oferece um conjunto de **algoritmos** que operam sobre sequências de acesso aleatório (como `std::vector` ou `std::deque`) para impor e manipular a propriedade do heap. Essa abordagem é mais flexível, permitindo que você escolha o contêiner subjacente mais apropriado para suas necessidades.

Todos os algoritmos de heap estão definidos no cabeçalho `<algorithm>`. A partir do C++20, esses algoritmos foram estendidos para suportar `std::ranges`, permitindo uma integração mais moderna com ranges e views, embora a funcionalidade básica permaneça a mesma.

---

### Representação de um Heap com Contêineres

#### Mapeamento Índice-Nó

A forma canônica de representar uma árvore binária completa ou quase completa em um vetor utiliza o seguinte mapeamento de índices. Para um elemento no índice `i`:

- **Pai**: `(i - 1) / 2`
- **Filho à esquerda**: `2 * i + 1`
- **Filho à direita**: `2 * i + 2`

#### Exemplo de Representação

Considere o seguinte max-heap balanceado:

```
        40
       /  \
      30   20
     / \
    10  25
```

No vetor: `[40, 30, 20, 10, 25]`

- `40` (índice 0): raiz
- `30` (índice 1): filho esquerdo de `40`, calculado como `2*0+1 = 1`
- `20` (índice 2): filho direito de `40`, calculado como `2*0+2 = 2`
- `10` (índice 3): filho esquerdo de `30`, calculado como `2*1+1 = 3`
- `25` (índice 4): filho direito de `30`, calculado como `2*1+2 = 4`

Para ilustrar um heap desbalanceado (quase completo, com o último nó à esquerda), considere este max-heap:

```
          50
         /  \
        40   30
       / \   /
      20  35 10
     /
    15
```

No vetor: `[50, 40, 30, 20, 35, 10, 15]`

- `50` (índice 0): raiz
- `40` (índice 1): filho esquerdo
- `30` (índice 2): filho direito
- `20` (índice 3): filho esquerdo de 40
- `35` (índice 4): filho direito de 40
- `10` (índice 5): filho esquerdo de 30
- `15` (índice 6): filho esquerdo de 20 (o heap é preenchido da esquerda para a direita)

Esta representação é eficiente em memória e permite que os algoritmos naveguem pela "árvore" usando aritmética simples de índices.

---

### Funções da STL para Operações com Heap

A STL fornece seis funções principais para trabalhar com heaps:

1. `std::make_heap()` - Constrói um heap a partir de uma faixa de elementos
2. `std::push_heap()` - Insere um elemento no heap
3. `std::pop_heap()` - Remove o elemento raiz do heap
4. `std::sort_heap()` - Ordena os elementos de um heap
5. `std::is_heap()` - Verifica se uma faixa é um heap válido
6. `std::is_heap_until()` - Encontra a sub-faixa mais longa que forma um heap

---

### 1. `std::make_heap()`

Rearranja os elementos em uma faixa `[first, last)` para que formem um heap válido. Por padrão, cria um max-heap, mas pode criar um min-heap com um comparador customizado.

#### Assinatura

```cpp
template<class RandomIt>
void make_heap(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
void make_heap(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo: Max-Heap e Min-Heap

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

void print_vector(const std::vector<int>& vec, const char* label) {
    std::cout << label;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> data = {20, 30, 40, 25, 15};
    print_vector(data, "Vetor original:      ");

    // 1. Criando um Max-Heap (padrao)
    std::vector<int> max_heap_data = data;
    std::make_heap(max_heap_data.begin(), max_heap_data.end());
    
    print_vector(max_heap_data, "Vetor como max-heap: ");
    std::cout << "O elemento maximo (raiz) e: " << max_heap_data.front() << "\n\n";

    // 2. Criando um Min-Heap
    std::vector<int> min_heap_data = data;
    std::make_heap(min_heap_data.begin(), min_heap_data.end(), std::greater<int>{});

    print_vector(min_heap_data, "Vetor como min-heap: ");
    std::cout << "O elemento minimo (raiz) e: " << min_heap_data.front() << "\n";

    return 0;
}
```

**Saída:**

```shell
Vetor original:      20 30 40 25 15 
Vetor como max-heap: 40 30 20 25 15 
O elemento maximo (raiz) e: 40

Vetor como min-heap: 15 25 20 30 40 
O elemento minimo (raiz) e: 15
```

#### Complexidade

- **Tempo**: $O(N)$ onde $N$ é o número de elementos
- **Espaço Auxiliar**: $O(1)$

---

### 2. `std::push_heap()`

Restaura a propriedade do heap após a adição de um novo elemento ao final do contêiner. A faixa `[first, last-1)` deve ser um heap válido antes da chamada.

#### Processo de Inserção

1. Adicione o elemento ao final do vetor usando `push_back()`
2. Chame `std::push_heap()` para restaurar a propriedade do heap
3. O algoritmo "sobe" o novo elemento na árvore até encontrar sua posição correta

#### Assinatura

```cpp
template<class RandomIt>
void push_heap(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
void push_heap(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void print_vector(const std::vector<int>& vec, const char* label) {
    std::cout << label;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> vec = {40, 30, 20, 10};

    // Garante que o vetor inicial e um heap
    std::make_heap(vec.begin(), vec.end());
    print_vector(vec, "Heap inicial:                ");

    // Passo 1: Adicionar novo elemento ao final
    vec.push_back(50);
    print_vector(vec, "Apos vec.push_back(50):      ");

    // Passo 2: Restaurar a propriedade do heap
    std::push_heap(vec.begin(), vec.end());
    print_vector(vec, "Apos std::push_heap():       ");
    
    std::cout << "Novo maximo: " << vec.front() << "\n";

    return 0;
}
```

**Saída:**

```shell
Heap inicial:                40 30 20 10 
Apos vec.push_back(50):      40 30 20 10 50 
Apos std::push_heap():       50 40 20 10 30 
Novo maximo: 50
```

#### Complexidade

- **Tempo**: $O(\log N)$
- **Espaço Auxiliar**: $O(1)$

---

### 3. `std::pop_heap()`

Move o elemento da raiz (o maior em um max-heap) para o final da faixa e reorganiza os elementos restantes para manter a propriedade do heap em `[first, last-1)`.

#### Processo de Remoção

1. O elemento raiz é trocado com o último elemento
2. A propriedade do heap é restaurada na faixa `[first, last-1)`
3. O elemento removido fica no final do contêiner
4. Use `pop_back()` para remover fisicamente o elemento

#### Assinatura

```cpp
template<class RandomIt>
void pop_heap(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
void pop_heap(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void print_vector(const std::vector<int>& vec, const char* label) {
    std::cout << label;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> vec = {50, 40, 20, 10, 30};
    
    std::make_heap(vec.begin(), vec.end());
    print_vector(vec, "Heap inicial:            ");
  
    // Passo 1: Mover o maior elemento (raiz) para o final
    std::pop_heap(vec.begin(), vec.end());
    print_vector(vec, "Apos std::pop_heap():    ");

    // Passo 2: Remover fisicamente o elemento do vetor
    int removed = vec.back();
    vec.pop_back();
    print_vector(vec, "Apos vec.pop_back():     ");
    
    std::cout << "Elemento removido: " << removed << "\n";
    std::cout << "Novo elemento maximo: " << vec.front() << "\n";

    return 0;
}
```

**Saída:**

```shell
Heap inicial:            50 40 20 10 30 
Apos std::pop_heap():    40 30 20 10 50 
Apos vec.pop_back():     40 30 20 10 
Elemento removido: 50
Novo elemento maximo: 40
```

#### Complexidade

- **Tempo**: $O(\log N)$
- **Espaço Auxiliar**: $O(1)$

---

### 4. `std::sort_heap()`

Transforma um heap em uma faixa ordenada. Após a execução, os elementos não formam mais um heap válido, mas estão ordenados. Para um max-heap, o resultado é uma ordenação crescente.

#### Funcionamento Interno

O algoritmo executa repetidamente `std::pop_heap()` até que toda a faixa seja processada, resultando em uma sequência ordenada.

#### Assinatura

```cpp
template<class RandomIt>
void sort_heap(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
void sort_heap(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void print_vector(const std::vector<int>& vec, const char* label) {
    std::cout << label;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> vec = {20, 30, 40, 25, 15, 10, 35};
    
    std::make_heap(vec.begin(), vec.end());
    print_vector(vec, "Max-heap:        ");
    
    std::sort_heap(vec.begin(), vec.end());
    print_vector(vec, "Apos ordenacao:  ");
    
    // Verificando se ainda e um heap (nao deveria ser)
    bool is_heap = std::is_heap(vec.begin(), vec.end());
    std::cout << "Ainda e um heap? " << (is_heap ? "Sim" : "Nao") << "\n";

    return 0;
}
```

**Saída:**

```shell
Max-heap:        40 30 35 25 15 10 20 
Apos ordenacao:  10 15 20 25 30 35 40 
Ainda e um heap? Nao
```

#### Complexidade

- **Tempo**: $O(N \log N)$
- **Espaço Auxiliar**: $O(1)$

#### Nota sobre Heapsort

Esta função implementa o algoritmo Heapsort, que é um algoritmo de ordenação in-place com complexidade $O(N \log N)$ no pior caso.

---

### 5. `std::is_heap()`

Verifica se uma faixa de elementos satisfaz a propriedade do heap. Retorna `true` se a faixa for um heap válido, `false` caso contrário. Casos edge como heaps vazios ou de tamanho 1 são considerados válidos.

#### Assinatura

```cpp
template<class RandomIt>
bool is_heap(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
bool is_heap(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec1 = {40, 30, 20, 10, 25};
    std::vector<int> vec2 = {40, 50, 20, 10, 25}; // Viola a propriedade
    std::vector<int> vec3 = {10, 20, 30, 40, 50}; // Ordenado mas nao e heap
    std::vector<int> empty_vec; // Heap vazio
    std::vector<int> single_vec = {42}; // Heap de tamanho 1
    
    std::cout << "vec1 e um max-heap? " 
              << (std::is_heap(vec1.begin(), vec1.end()) ? "Sim" : "Nao") << "\n";
    
    std::cout << "vec2 e um max-heap? " 
              << (std::is_heap(vec2.begin(), vec2.end()) ? "Sim" : "Nao") << "\n";
    
    std::cout << "vec3 e um max-heap? " 
              << (std::is_heap(vec3.begin(), vec3.end()) ? "Sim" : "Nao") << "\n";
    
    // Verificando min-heap
    std::cout << "vec3 e um min-heap? " 
              << (std::is_heap(vec3.begin(), vec3.end(), std::greater<int>{}) ? "Sim" : "Nao") << "\n";
    
    // Casos edge
    std::cout << "Heap vazio e valido? " 
              << (std::is_heap(empty_vec.begin(), empty_vec.end()) ? "Sim" : "Nao") << "\n";
    
    std::cout << "Heap de tamanho 1 e valido? " 
              << (std::is_heap(single_vec.begin(), single_vec.end()) ? "Sim" : "Nao") << "\n";

    return 0;
}
```

**Saída:**

```shell
vec1 e um max-heap? Sim
vec2 e um max-heap? Nao
vec3 e um max-heap? Nao
vec3 e um min-heap? Sim
Heap vazio e valido? Sim
Heap de tamanho 1 e valido? Sim
```

#### Complexidade

- **Tempo**: $O(N)$
- **Espaço Auxiliar**: $O(1)$

---

### 6. `std::is_heap_until()`

Retorna um iterador para o primeiro elemento que viola a propriedade do heap. Se toda a faixa for um heap válido, retorna `last`.

#### Assinatura

```cpp
template<class RandomIt>
RandomIt is_heap_until(RandomIt first, RandomIt last);

template<class RandomIt, class Compare>
RandomIt is_heap_until(RandomIt first, RandomIt last, Compare comp);
```

#### Exemplo

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void print_vector(const std::vector<int>& vec, const char* label) {
    std::cout << label;
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<int> vec = {40, 30, 20, 10, 5, 60, 15};
    // Os primeiros 5 elementos formam um heap, mas 60 viola a propriedade
    
    print_vector(vec, "Vetor: ");
    
    auto it = std::is_heap_until(vec.begin(), vec.end());
    
    if (it == vec.end()) {
        std::cout << "Todo o vetor e um heap valido\n";
    } else {
        std::cout << "O heap valido vai ate o indice: " 
                  << std::distance(vec.begin(), it) << "\n";
        std::cout << "Primeiro elemento que viola: " << *it << "\n";
    }
    
    // Exibindo a sub-faixa que e heap
    std::cout << "Sub-faixa que e heap: ";
    for (auto i = vec.begin(); i != it; ++i) {
        std::cout << *i << " ";
    }
    std::cout << "\n";

    return 0;
}
```

**Saída:**

```shell
Vetor: 40 30 20 10 5 60 15 
O heap valido vai ate o indice: 5
Primeiro elemento que viola: 60
Sub-faixa que e heap: 40 30 20 10 5
```

#### Complexidade

- **Tempo**: $O(N)$
- **Espaço Auxiliar**: $O(1)$

---

### Comparadores Customizados

#### Usando Functors e Lambdas

Todos os algoritmos de heap aceitam um comparador customizado, permitindo flexibilidade na definição de prioridade.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Person {
    std::string name;
    int age;
};

int main() {
    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 35},
        {"Diana", 28}
    };
    
    // Max-heap baseado na idade (maior idade na raiz)
    auto age_comparator = <a href="const Person& a, const Person& b" target="_blank" rel="noopener noreferrer nofollow"></a> {
        return a.age < b.age;
    };
    
    std::vector<Person> people_max = people;
    std::make_heap(people_max.begin(), people_max.end(), age_comparator);
    
    std::cout << "Pessoa mais velha: " << people_max.front().name 
              << " (" << people_max.front().age << " anos)\n";
    
    // Min-heap baseado na idade (menor idade na raiz)
    auto age_comparator_min = <a href="const Person& a, const Person& b" target="_blank" rel="noopener noreferrer nofollow"></a> {
        return a.age > b.age;
    };
    
    std::vector<Person> people_min = people;
    std::make_heap(people_min.begin(), people_min.end(), age_comparator_min);
    
    std::cout << "Pessoa mais jovem: " << people_min.front().name 
              << " (" << people_min.front().age << " anos)\n";

    return 0;
}
```

**Saída:**

```shell
Pessoa mais velha: Charlie (35 anos)
Pessoa mais jovem: Bob (25 anos)
```

---

### `std::priority_queue`: Wrapper de Alto Nível

A STL fornece `std::priority_queue`, um adaptador de contêiner que encapsula as operações de heap, oferecendo uma interface mais conveniente.

#### Características

- Contêiner subjacente padrão: `std::vector`
- Comparador padrão: `std::less` (max-heap)
- Interface simplificada: `push()`, `pop()`, `top()`

#### Assinatura

```cpp
template<
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type>
>
class priority_queue;
```

#### Exemplo: Priority Queue Básica

```cpp
#include <iostream>
#include <queue>
#include <vector>

int main() {
    // Max-heap (padrao)
    std::priority_queue<int> max_pq;
    
    max_pq.push(20);
    max_pq.push(30);
    max_pq.push(10);
    max_pq.push(40);
    
    std::cout << "Max Priority Queue:\n";
    while (!max_pq.empty()) {
        std::cout << max_pq.top() << " ";
        max_pq.pop();
    }
    std::cout << "\n\n";
    
    // Min-heap
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    
    min_pq.push(20);
    min_pq.push(30);
    min_pq.push(10);
    min_pq.push(40);
    
    std::cout << "Min Priority Queue:\n";
    while (!min_pq.empty()) {
        std::cout << min_pq.top() << " ";
        min_pq.pop();
    }
    std::cout << "\n";

    return 0;
}
```

**Saída:**

```shell
Max Priority Queue:
40 30 20 10 

Min Priority Queue:
10 20 30 40
```

---

### Caso de Uso: Sistema de Tarefas com Prioridade

#### Implementação Completa

```cpp
#include <iostream>
#include <queue>
#include <string>
#include <vector>

struct Task {
    std::string description;
    int priority;
    
    // Operador de comparacao para o heap
    // Menor valor de priority significa maior prioridade (min-heap)
    bool operator>(const Task& other) const {
        return priority > other.priority;
    }
};

int main() {
    // Priority queue que retorna tarefas com menor valor de prioridade primeiro
    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> task_queue;
    
    // Adicionando tarefas
    task_queue.push({"Responder email", 3});
    task_queue.push({"Corrigir bug critico", 1});
    task_queue.push({"Reuniao de equipe", 2});
    task_queue.push({"Documentar codigo", 4});
    task_queue.push({"Revisar pull request", 2});
    
    std::cout << "Executando tarefas por ordem de prioridade:\n";
    std::cout << "--------------------------------------------\n";
    
    int task_number = 1;
    while (!task_queue.empty()) {
        const Task& current_task = task_queue.top();
        std::cout << task_number++ << ". [P" << current_task.priority << "] " 
                  << current_task.description << "\n";
        task_queue.pop();
    }

    return 0;
}
```

**Saída:**

```shell
Executando tarefas por ordem de prioridade:
--------------------------------------------
1. [P1] Corrigir bug critico
2. [P2] Reuniao de equipe
3. [P2] Revisar pull request
4. [P3] Responder email
5. [P4] Documentar codigo
```

---

### Comparação: Algoritmos de Heap vs Priority Queue

| Aspecto | Algoritmos de Heap | Priority Queue |
|---------|-------------------|----------------|
| Controle | Controle total sobre o contêiner | Interface abstrata |
| Flexibilidade | Pode usar qualquer contêiner de acesso aleatório | Limitado ao contêiner escolhido na instanciação |
| Operações | Requer chamadas explícitas | Interface simplificada |
| Performance | Mesma complexidade assintótica | Mesma complexidade assintótica |
| Uso | Quando você precisa de controle fino | Maioria dos casos de uso |

---

### Considerações de Performance

#### Complexidades Resumidas

| Operação | Complexidade de Tempo | Complexidade de Espaço |
|----------|----------------------|------------------------|
| `make_heap()` | $O(N)$ | $O(1)$ |
| `push_heap()` | $O(\log N)$ | $O(1)$ |
| `pop_heap()` | $O(\log N)$ | $O(1)$ |
| `sort_heap()` | $O(N \log N)$ | $O(1)$ |
| `is_heap()` | $O(N)$ | $O(1)$ |
| `is_heap_until()` | $O(N)$ | $O(1)$ |

#### Escolha do Contêiner Subjacente

- **`std::vector`**: Escolha padrão, boa localidade de cache
- **`std::deque`**: Útil quando inserções/remoções em ambas as extremidades são frequentes
- Evite `std::list`: Não oferece acesso aleatório necessário para heaps

---

### Boas Práticas

1. **Use `std::priority_queue`** para a maioria dos casos de uso, a menos que precise de controle direto sobre o contêiner

2. **Mantenha a invariante**: Após `push_back()`, sempre chame `push_heap()`; antes de `pop_back()`, sempre chame `pop_heap()`

3. **Escolha o comparador correto**: Lembre-se que `std::less` cria um max-heap e `std::greater` cria um min-heap

4. **Reserve memória**: Use `reserve()` no vetor subjacente se souber o tamanho aproximado

5. **Validação durante desenvolvimento**: Use `std::is_heap()` para verificar invariantes durante o desenvolvimento
