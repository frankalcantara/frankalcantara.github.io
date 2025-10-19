---
layout: posts
title: Maps em C++
author: Frank
categories:
    - artigo
    - CPP
tags:
    - algoritmos
    - C++
    - eng. software
    - estrutura de dados
image: assets/images/cpp.webp
featured: false
rating: 0
description: Algumas dicas e características de Maps em C++23
date: 2025-10-19T01:43:39.556Z
preview: Algumas dicas e características de Maps em C++23
keywords: |
    Cpp
    Algoritmos

    Maps
toc: true
published: true
beforetoc: ""
lastmod: 2025-10-19T01:45:35.999Z
draft: 2025-10-19T01:44:15.496Z
---
> Precisa de revisão, esta é uma versão preliminar a partir de um texto antigo em C++ 20. Esta versão foi verificada com o auxílio do Gemini 2.4, Claude 4.5 e Grok 4.1 em 10 de outubro de 2025.


A necessidade de associar chaves a valores é fundamental em programação. Em C++, a biblioteca padrão oferece diversas ferramentas para esta tarefa: `std::map` e `std::multimap` (baseados em comparações), além de `std::unordered_map` e `std::unordered_multimap` (baseados em hashing). A biblioteca Boost adiciona `flat_map`, que oferece trade-offs de desempenho diferentes, e `bimap`, para consultar tanto chaves quanto valores.

Todas essas estruturas compartilham uma característica comum: associam um único tipo de chave a um tipo de valor. Por exemplo, em `std::map<int, std::string>`, o tipo de chave é exclusivamente `int`. Este artigo explora duas situações avançadas: criar maps com dois tipos de chaves e modificar chaves existentes em maps e sets.

## Parte I: Map com Dois Tipos de Chaves

### Motivação

Por que não associar dois tipos de chaves a um único tipo de valor? Um caso de uso prático surge quando conhecemos duas representações da chave no momento da inserção e desejamos consultar o map usando qualquer uma delas.

Por exemplo, em pseudocódigo:

```cpp
// este map aceita chaves do tipo char e int, associando-as a strings
m.insert(key1 = 0, key2 = '0', value = "zero")
m.insert(key1 = 1, key2 = '1', value = "one")
m.insert(key1 = 2, key2 = '2', value = "two")

...

EXPECT_TRUE(m[1] == "one")
EXPECT_TRUE(m['1'] == "one")
```

### Requisitos do Problema

Este problema pode ser abordado de diferentes maneiras. Os únicos requisitos são:

1. As duas representações das chaves devem ser inseridas simultaneamente
2. O valor deve ser consultável por qualquer um dos dois tipos de chave

### Uma Implementação Possível

#### Raciocínio

Uma abordagem é manter dois maps internamente:
- O primeiro mapeia `Key1` para `Key2`
- O segundo mapeia `Key2` para `Value`

Uma inserção no map de chave dupla executa duas inserções:

```cpp
map1: 1 -> '1'
map2: '1' -> "one"
```

Uma consulta por `Key1` realiza duas buscas: primeiro encontra `Key2` correspondente, depois encontra o valor. Uma consulta por `Key2` realiza apenas uma busca direta no segundo map.

#### Decisões de Design

Como não existe um único iterator na coleção, não é possível oferecer o método `find` convencional que retorna um iterator. A solução utiliza `std::optional<Value>` como retorno do método `getValue`, que retorna `std::nullopt` quando a chave não existe.

Esta estrutura não suporta o operador `[]` de `std::map`, pois quando uma chave não existe, seria necessário conhecer ambas as representações para realizar a inserção, mas apenas uma foi fornecida.

#### Implementação

```cpp
template <typename Key1, typename Key2, typename Value>
class doublekey_map
{
public:
    auto size() const
    {
        return key1_key2_.size();
    }
    
    void insert(std::tuple<Key1, Key2, Value> const& entry)
    {
        key1_key2_.insert(std::make_pair(std::get<0>(entry), std::get<1>(entry)));
        key2_value_.insert(std::make_pair(std::get<1>(entry), std::get<2>(entry)));
    }

    std::optional<Value> getValue(Key1 const& key1)
    {
        auto key2 = key1_key2_.find(key1);
        if (key2 == end(key1_key2_)) return std::nullopt;
        
        auto value = key2_value_.find(key2->second);
        if (value == end(key2_value_)) return std::nullopt;
        
        return value->second;
    }

    std::optional<Value> getValue(Key2 const& key2)
    {
        auto value = key2_value_.find(key2);
        if (value == end(key2_value_)) return std::nullopt;

        return value->second;
    }

private:
    std::map<Key1, Key2> key1_key2_;
    std::map<Key2, Value> key2_value_;
};
```

#### Análise de Limitações

Esta solução apresenta trade-offs:

1. Não segue as convenções da STL (sem `begin`, `end`, `find`, operador `[]` ou aliases de tipo)
2. A consulta por `Key1` tem complexidade O(log N) com duas buscas, enquanto a consulta por `Key2` tem O(log N) com uma única busca
3. Usa o dobro de memória comparado a um map simples

## Parte II: Como Modificar Chaves em Maps e Sets

### O Problema

Diferentemente de containers sequenciais como `std::vector`, não é possível simplesmente atribuir um novo valor a uma chave de `std::map`:

```cpp
auto myMap = std::map<std::string, int>{ {"one", 1}, {"two", 2}, {"three", 3} };
myMap.find("two")->first = "dos";  // ERRO!
```

O compilador gera erros porque a chave é `const`. Com `int` como chave, a mensagem é mais clara:

```cpp
error: assignment of read-only member 'std::pair<const int, std::__cxx11::basic_string<char>>::first'
```

Em contraste, modificar um valor compila normalmente:

```cpp
myMap.find("two")->second = 22;  // OK
```

O mesmo problema ocorre com `std::set`:

```cpp
auto mySet = std::set<std::string>{"one", "two", "three"};
*mySet.find("two") = "dos";  // ERRO!
```

### Por Que as Chaves São Imutáveis

`std::map` e `std::set` oferecem duas garantias fundamentais:

1. Mantêm elementos em ordem classificada
2. Garantem unicidade dos elementos (exceto `std::multimap` e `std::multiset`)

Para manter estas invariantes, os containers precisam controlar as posições relativas dos valores. Se você modificasse uma chave diretamente via iterator, o container não seria notificado, quebrando sua estrutura interna.

### A Solução com `extract` (C++17+)

Desde C++17, containers associativos fornecem o método `extract`, que remove um nó da estrutura sem destruí-lo:

```cpp
auto myMap = std::map<std::string, int>{ {"one", 1}, {"two", 2}, {"three", 3} };

auto node = myMap.extract("two");
```

O `extract` tem efeito modificador: o map não contém mais o nó. Verificando o tamanho antes e depois:

```cpp
std::cout << myMap.size() << '\n';  // 3
auto node = myMap.extract("two");
std::cout << myMap.size() << '\n';  // 2
```

Agora você é o único proprietário do nó e pode modificá-lo com segurança. O nó fornece acesso não-const à chave via `key()`:

```cpp
node.key() = "dos";
```

Após modificar a chave, reinsira o nó usando `insert`:

```cpp
myMap.insert(std::move(node));
```

Note o `std::move`, que expressa a transferência de propriedade. O código não compilaria sem ele, pois o nó possui apenas move constructor.

### Tratando Nós Inexistentes

Ao extrair um nó inexistente:

```cpp
auto node = myMap.extract("four");
```

O nó é um objeto válido, mas vazio. Acessar `node.key()` resulta em comportamento indefinido. Sempre verifique com `empty()`:

```cpp
auto node = myMap.extract("two");
if (!node.empty())
{
    node.key() = "dos";
    myMap.insert(std::move(node));
}
```

### O Caso de `std::set`

Para `std::set`, o nó possui o método `value()` em vez de `key()`:

```cpp
auto mySet = std::set<std::string>{"one", "two", "three"};

auto node = mySet.extract("two");
if (!node.empty())
{
    node.value() = "dos";
    mySet.insert(std::move(node));
}
```

### Encapsulando em uma Função Genérica

Para abstrair os detalhes de baixo nível, podemos criar uma função template:

```cpp
template<typename Container>
void replaceKey(Container& container,
                const typename Container::key_type& oldKey,
                const typename Container::key_type& newKey)
{
    auto node = container.extract(oldKey);
    if (!node.empty())
    {
        node.key() = newKey;
        container.insert(std::move(node));
    }
}
```

Esta função funciona para `std::map` e `std::multimap`. Para `std::set`, seria necessária uma implementação separada devido à diferença entre `key()` e `value()`.

Em C++23, com conceitos e `if constexpr`, podemos criar uma versão unificada:

```cpp
template<typename Container>
void replaceKey(Container& container,
                const typename Container::key_type& oldKey,
                const typename Container::key_type& newKey)
{
    auto node = container.extract(oldKey);
    if (!node.empty())
    {
        if constexpr (requires { node.key(); })
            node.key() = newKey;
        else
            node.value() = newKey;
            
        container.insert(std::move(node));
    }
}
```

### Observação Importante sobre Valores em Maps

O nó de map não fornece um acessor `value()`. Se você precisa modificar apenas o valor, faça-o diretamente no map sem extrair o nó. A linguagem previne soluções ineficientes restringindo a interface do nó para maps.

## Conclusão

Estruturas de dados avançadas em C++ requerem compreensão profunda das invariantes mantidas pelos containers. O map com dois tipos de chaves oferece flexibilidade de consulta ao custo de maior complexidade e uso de memória. A modificação de chaves via `extract` demonstra como C++ moderno fornece abstrações seguras e eficientes para operações que anteriormente exigiam cópias desnecessárias.

Em C++23, com conceitos e outras features modernas, podemos criar abstrações ainda mais elegantes que encapsulam a complexidade desses padrões, mantendo a eficiência e a segurança de tipos que caracterizam a linguagem.