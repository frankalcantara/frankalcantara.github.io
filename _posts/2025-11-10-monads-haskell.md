---
layout: post
title: Teoria das Categorias e Monads em Haskell
author: Frank
categories: |-
  Matemática
  programação
  Cálculo Lambda
tags: |-
  Algoritmos
  C++
  Compiladores
  Exercícios 
  Cálculo Lambda
  Haskell
  Teoria das Categorias
image: assets/images/monads.webp
featured: false
rating: 5
description: "Introdução à Teoria das Categorias e Monads em Haskell: conceitos fundamentais, exemplos práticos e aplicações na programação funcional"
date: 2025-11-01T15:01:39.506Z
preview: As monads têm uma função relevante e indispensável na programação funcional, permitindo a composição de efeitos e a manipulação de contextos de forma elegante e segura. Este texto apresenta uma introdução à teoria das categorias direcionada ao entendimento de monads em Haskell, explorando seus conceitos fundamentais, exemplos práticos e aplicações na programação funcional.
keywords: |-
  Algoritmos
  Exercícios
  cálculo lambda
  recursão
  redução beta
  currying
  monads
  categorias
  Teoria das categorias
toc: true
published: true
lastmod: 2025-11-10T21:49:41.671Z
draft: 2025-11-01T15:01:40.919Z
---

A **programação funcional**, especialmente na linguagem **Haskell**, repousa sobre fundamentos matemáticos que, embora abstratos, fornecem uma base precisa para a compreensão de como as máquinas processam informações. Entre estes fundamentos, os conceitos da **Teoria das Categorias** ocupam uma posição central.

Neste texto, a curiosa leitora explorará como ideias puramente matemáticas, **objetos**, **morfismos**, **composição**, **functores**, **transformações naturais** e **monads**, podem ser traduzidos em artefatos de código concretos e poderosos. Nosso objetivo é desmistificar esses termos e demonstrar como eles fornecem ferramentas interessantes para a engenharia de software. Começando, como não poderia deixar de ser, pela própria Teoria das Categorias.

## Teoria das Categorias: a base conceitual

A Teoria das Categorias é um dos ramos da matemática que estuda composição. Ela não se importa com *o que as coisas são, mas sim com como elas se relacionam e se compõem*.

### O que é uma categoria?

Uma **categoria** $\mathcal{C}$ é uma estrutura matemática, similar a um grafo direcionado, composta por:

- **Objetos**: $Obj(\mathcal{C})$. A cuidadosa leitora pode interpretá-los como tipos de dados (ex: `Int`, `String`) ou conjuntos.
- **Morfismos**: $Hom_{\mathcal{C}}(A, B)$. São as "setas" ou transformações entre objetos (ex: funções, como `show :: Int -> String`).
- **Operação de composição**: $\circ : Hom(B,C) \to Hom(A,B) \to Hom(A,C)$. Uma operação que combina morfismos de forma associativa.
- **Morfismo identidade**: $id_A : A \to A$. Para cada objeto $A$, existe uma seta que *não faz nada*.

Esta estrutura pode ser vista na figura a seguir:

![Diagrama de uma categoria com dos objetos, A e B dois morfimos representados por setas e a operação de composição](/assets/images/catego1.webp)

Para que essa estrutura seja formalmente uma categoria, duas leis são necessárias e indispensáveis:

1. **Associatividade**: Para morfismos $f: A \to B$, $g: B \to C$ e $h: C \to D$: $h \circ (g \circ f) = (h \circ g) \circ f$

2. **Identidade**: Para qualquer morfismo $f: A \to B$:$id_B \circ f = f = f \circ id_A$

Essas leis garantem que as composições, caminhos, dentro da categoria se comportem de maneira previsível.

### Exemplo concreto: a categoria Hask 

Em Haskell, existe uma categoria implícita, e idealizada, chamada **Hask**:

| Elemento | Em Hask |
|:---------------|:--------------------------------------|
| **Objetos** | Tipos Haskell (`Int`, `String`, etc.) |
| **Morfismos** | Funções **puras e totais** |
| **Composição** | Operador `(.)` |
| **Identidade** | Função `id` |

A composição de funções em Haskell é feita com o operador `(.)`, definido como:
`(.) :: (b -> c) -> (a -> b) -> (a -> c)`
E a função identidade é:`id :: a -> a`

As leis da categoria são, majoritariamente, garantidas pelo compilador:

```haskell
-- definimos composição por .
-- Associatividade pode ser verificada assim:
h . (g . f) == (h . g) . f

-- Identidade
f . id == f -- à direita
id . f == f -- à esquerda
```

A figura a seguir ilustra a categoria **Hask**:

![Diagrama da categoria Hask representando um morfismo de `Int` para `String`, um morfismo de `String` para `Bool` e a assinatura de tipos da operação de composição](/assets/images/catHask.webp)

>**Nota sobre a pureza de Hask**: a atenta leitora deve observar que, na prática, **Hask** não é uma categoria matemática perfeita. Isso se deve à existência de funções parciais, que falham para certas entradas e valores indefinidos (como `undefined` ou `error "..."`), que violam a propriedade de que _morfismos devem ser totais_. Contudo, ela serve como uma aproximação conceitual poderosa.

### 2.2 Outros Exemplos de Categorias

Embora **Hask** seja o nosso principal objeto de estudo, a Teoria das Categorias ganha vida por meio de exemplos e aplicações, mesmo na álgebra pura.

#### A Categoria **Set**

A categoria **Set** é, talvez, a categoria mais intuitiva, servindo de base para muitas outras.

| Elemento | Em **Set** |
| :--- | :--- |
| **Objetos** | Quaisquer conjuntos (ex: $\{1, 2, 3\}$, $\mathbb{R}$, $\{\text{vermelho}, \text{azul}\}$). |
| **Morfismos** | Funções (totais) entre conjuntos (ex: $f: \mathbb{Z} \to \mathbb{Z}$ definida por $f(x) = x^2$). |
| **Composição** | A composição padrão de funções, $(g \circ f)(x) = g(f(x))$. |
| **Identidade** | A função identidade $id_A(x) = x$ para cada conjunto $A$. |

As leis da categoria são satisfeitas:

1. **Associatividade**: A composição de funções é inerentemente associativa, $h \circ (g \circ f) = (h \circ g) \circ f$.
2. **Identidade**: A função $id$ atua como elemento neutro, $f \circ id_A = f$ e $id_B \circ f = f$.

#### A Categoria **Poset**

Uma categoria **Poset**, de **P**artially **O**rdered **Set**, ou Conjunto Parcialmente Ordenado, é uma construção mais sutil, em que a própria relação de ordem define os morfismos.

Seja $(P, \leq)$ um conjunto $P$ com uma relação de ordem parcial $\leq$, reflexiva, antissimétrica e transitiva. Neste caso, teremos:

| Elemento | Em **Poset** |
| :--- | :--- |
| **Objetos** | Os elementos do conjunto $P$ (ex: $1, 2, 3, \ldots$). |
| **Morfismos** | A própria relação. Existe um morfismo $f: A \to B$ *se, e somente se,* $A \leq B$. |
| **Composição** | A **transitividade** da relação. |
| **Identidade** | A **reflexividade** da relação. |

Para que esta seja uma categoria, as leis devem ser satisfeitas:

1. **Associatividade**: se existe um morfismo $f: A \to B$ (ou seja, $A \leq B$) e $g: B \to C$ (ou seja, $B \leq C$), a composição $g \circ f$ exige um morfismo $h: A \to C$. Neste cenário, a propriedade da **transitividade** ($A \leq B$ e $B \leq C \implies A \leq C$) garante que este morfismo $h$ existe.
2. **Identidade**: para todo objeto $A$, deve existir um morfismo $id_A: A \to A$. A propriedade da **reflexividade** ($A \leq A$) garante que este morfismo de identidade sempre existe.

### Exercícios 1

1. **Análise das Leis**: Por que a lei da identidade é definida como $id_B \circ f = f = f \circ id_A$? Explique por que $id_A \circ f$ (por exemplo) não faria sentido em termos de tipos.
2. **Morfismos**: Na categoria **Hask**, a função `read :: String -> Int` é um morfismo válido? Justifique sua resposta considerando a definição de morfismo em **Hask**. (Dica: o que acontece se `read "oi"` for chamado?)

## Functores: mapeando categorias

Se categorias são *universos* de objetos e morfismos, um _**Functor** é um tradutor que mapeia um universo para outro preservando sua estrutura fundamental_. Formalmente dizemos:

Um **Functor** $F: \mathcal{C} \to \mathcal{D}$ é um mapeamento que preserva a estrutura das categorias $\mathcal{C}$ e $\mathcal{D}$:

- **Mapeia Objetos**: $A \in \mathcal{C} \mapsto F(A) \in \mathcal{D}$
- **Mapeia Morfismos**: $(f: A \to B) \mapsto (F(f): F(A) \to F(B))$

Este mapeamento deve obedecer a duas leis:

1. **Preservação da identidade**:$F(id_A) = id_{F(A)}$

2. **Preservação da composição**:$F(g \circ f) = F(g) \circ F(f)$

### Functores em Haskell

Em Haskell, quase sempre lidamos com **endofunctores**, functores que mapeiam **Hask** para **Hask**. A `typeclass` `Functor` captura essa ideia:

```haskell
class Functor f where\
fmap :: (a -> b) -> f a -> f b

-- `f` é o mapeamento de objetos (ex: `Int`  ->  `Maybe Int`).
-- `fmap` é o mapeamento de morfismos (ex: `(+1)`  ->  uma função que aplica `(+1)` dentro do `Maybe`).

--`fmap` aplica uma função "dentro" de um contexto ou "container" (`f`).
```

Exemplos:

```haskell
fmap (+10) (Just 5) -- Resulta em: Just 15
fmap (+1) [1,2,3] -- Resulta em: [2,3,4]
fmap length ["a","bc"] -- Resulta em: [1,2]
```

As leis do Functor em Haskell são a tradução direta das leis matemáticas:

```haskell
-- Preservação da identidade\
fmap id == id

-- Preservação da composição\
fmap (g . f) == fmap g . fmap f
```

![ilustra hierarquia e assinaturas importantes; coloque esta figura onde inicia aplicative para deixar claro a progressão conceitual.](/assets/images/functor1.webp)

## O Problema do Functor e a Solução Applicative

O `fmap` é excelente, mas tem uma limitação: ele só funciona quando a função, morfismo, está "do lado de fora", pura. Neste caso, precisamos lidar com o que acontece se a própria função estiver dentro do contexto?

```haskell
Just (+5) :: Maybe (Int -> Int)\
Just 10 :: Maybe Int
```

Não podemos usar `fmap`. A assinatura de `fmap` é `(a -> b) -> f a -> f b`, mas o que temos é `f (a -> b)`.

Isso significa que `fmap` espera uma função *pura* como primeiro argumento `(a -> b)`, enquanto no nosso caso a função está *dentro* do mesmo contexto `f`. Em outras palavras, `fmap` consegue aplicar uma transformação sobre um valor encapsulado, mas **não consegue aplicar uma função que também está encapsulada**. 

A atenta leitora deve observar que o obstáculo aqui não é apenas sintático, mas estrutural: `fmap` opera em um único nível de contexto, e o que temos é uma aplicação entre dois valores contextualizados, uma função em `f (a -> b)` e um argumento em `f a`. 

### Applicative: aplicando funções em contextos

Para resolver as limitações do `fmap`, surge a `typeclass` `Applicative`, que estende a `typeclass` `Functor`:

```haskell
class Functor f => Applicative f where
pure :: a -> f a
(<*>) :: f (a -> b) -> f a -> f b

-- `pure`: injeta um valor puro no contexto (o "menor" contexto possível).
-- `(<*>)` (lê-se "ap"): aplica uma função contextualizada a um valor contextualizado.

--Exemplo (resolvendo o problema anterior):

Just (+5) <*> Just 10 -- Resulta em: Just 15

-- E se quisermos somar dois valores em contexto?

pure (+) <*> Just 5 <*> Just 10 -- Resulta em: Just 15

-- Ele também propaga falhas (Nothing)

pure (+) <*> Just 5 <*> Nothing -- Resulta em: Nothing
```

A `typeclass` `Applicative` é excelente para combinar múltiplos valores **independentes** que estão dentro de um mesmo contexto. Ela permite aplicar funções de múltiplos argumentos sem precisar extrair explicitamente os valores do contexto, preservando a pureza e a composicionalidade do código.

O operador `<*>` realiza a aplicação sequencial de funções em contexto a valores também contextualizados, enquanto `pure` injeta uma função ou valor puro nesse contexto para iniciar a cadeia de aplicações.

Por exemplo, considere a monad `Maybe`, que veremos com mais cuidado a seguir e que representa computações que podem falhar:

```haskell
pure (+) <*> Just 5 <*> Just 10
-- Resultado: Just 15
```

Neste caso, tanto a função `(+)` quanto os valores `5` e `10` são combinados dentro do contexto `Maybe`. Se algum deles for `Nothing`, o resultado de toda a expressão também será `Nothing`, mantendo a coerência do contexto:

```haskell
pure (+) <*> Just 5 <*> Nothing
-- Resultado: Nothing
```

Da mesma forma, o comportamento se estende a listas, que representam computações com múltiplos resultados possíveis:

```haskell
pure (*) <*> [1, 2] <*> [10, 100]
-- Resultado: [10, 100, 20, 200]
```

Aqui, o `Applicative` executa todas as combinações possíveis de multiplicação entre os elementos das duas listas, produzindo uma lista com todos os resultados, uma espécie de produto cartesiano funcional.

Em resumo, o `Applicative` generaliza o `Functor`: enquanto `fmap` aplica uma função pura a um valor em contexto, o `Applicative` permite aplicar **funções em contexto a múltiplos valores em contexto**, promovendo composição estruturada e independente dentro de ambientes computacionais.

### Exercícios 2

1. Em Haskell, defina as assinaturas de tipo e implemente exemplos de uso para as funções `pure`, `just` e `Maybe`.

2. **Leis do Functor**: Prove que a implementação de `fmap` para `Maybe` obedece às duas leis do functor.

  ```haskell
  instance Functor Maybe where
  fmap _ Nothing = Nothing
  fmap f (Just x) = Just (f x)
  ```

3. **Leis do Functor para Listas**: Prove que a implementação de `fmap` para listas (`map`) obedece à lei da composição: `fmap (g . f) == fmap g . fmap f`.

4. **Uso de Applicative**: Usando `pure` e `(<*>)`, escreva uma expressão que combine três `Maybe String` em um único `Maybe String` (concatenando-os).

 - `val1 = Just "a"`
 - `val2 = Just "b"`
 - `val3 = Just "c"`
 - (Dica: `pure (++) <*> ...`)

## Computações Dependentes e o Nascimento das Monads

A `Applicative` é poderosa, mas ainda limitada. Ela funciona para computações *independentes*. Mas e se a *próxima* computação depender do *resultado* da computação anterior? Caracterizando funções em *pipeline* onde o resultado de uma etapa influencia a próxima.

Considere este cenário:

```haskell
buscarUsuario :: String -> Maybe User
buscarPermissoes :: User -> Maybe Permissions
```

Não podemos usar `(<*>)`. Precisamos do valor `User` *de dentro* do `Maybe` para poder passá-lo para `buscarPermissoes`.

É para resolver esse **encadeamento dependente** que surge a **Monad**.

### Definição Categórica

Formalmente, uma **Monad** em uma categoria $\mathcal{C}$ é uma tripla $(T, \eta, \mu)$ que consiste em:

1. $T: \mathcal{C} \to \mathcal{C}$ — um **endoFunctor** (ex: `Maybe`).

2. $\eta: I \to T$ — uma transformação natural chamada **unidade** (`pure`/`return`). Ela pega um objeto $A$ e o "injeta" no Functor, $A \to T(A)$.

3. $\mu: T^2 \to T$ — uma transformação natural chamada **multiplicação** (`join`). Ela "achata" um Functor aninhado, $T(T(A)) \to T(A)$.

A estrutura deve obedecer a certas leis, diagramas comutativos, que garantem associatividade e identidade.

## A Monad na Prática: Haskell

A definição matemática é elegante, mas a definição em Haskell é, para muitos, mais prática.

A definição moderna (pós-GHC 7.10) da `typeclass` `Monad` é:

```haskell
class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
-- return = pure (não é mais parte da classe)
```

A atenta leitora deve prestar a atenção à dois pontos importantes:

1. Toda `Monad` é também `Applicative`e, portanto, `Functor`.
2. A definição mínima é apenas o operador `(>>=)` (lê-se *bind*). A função `return` agora é apenas um sinônimo para `pure`.

O operador `(>>=)` é a essência do encadeamento dependente:

```haskell
(>>=) :: m a -> (a -> m b) -> m b
```

Ele pega (1) um valor no contexto `m a`, e (2) uma função `(a -> m b)` que sabe o que fazer com o valor `a` puro. O *bind* cuida de extrair `a` de `m a` e passá-lo para a função.

Exemplos:

```haskell
-- Sucesso:\
Just 5 >>= (\x -> Just (x + 1)) -- Resulta em: Just 6

-- Falha (o 'bind' faz o short-circuit):\
Nothing >>= (\x -> Just (x + 1)) -- Resulta em: Nothing (a função nem é executada)
```

### Relação entre join e bind

As duas definições de Monad, matemática com $\mu$ e Haskell com `(>>=)` são equivalentes. Podemos definir uma em termos da outra:

1. *bind* em termos de `join + fmap`
  Se tivéssemos `join :: m (m a) -> m a`, poderíamos definir bind: `m >>= f = join (fmap f m)` . 

  Análise dos tipos a cuidadosa leitora deve verificar:
  `m :: m a`
  `f :: a -> m b`
  `fmap f m :: m (m b)` (Contexto aninhado!)
  `join (...) :: m b` (Achatado!)

2. join em termos de bind
  Em Haskell, podemos definir `join` facilmente:
  
  ```haskell
  join :: Monad m => m (m a) -> m a\
  join mma = mma >>= id
  ```

  Análise dos tipos:
  `mma :: m (m a)`
  `id :: m a -> m a` (Aqui, o `a` em `(a -> m b)` é `m a`)
  `(>>=)` aplica `id` ao conteúdo `m a` interno, achatando o resultado.

### A Categoria de Kleisli — composição monádica como morfismo puro

Toda `Monad` em uma categoria $\mathcal{C}$ define uma nova categoria, chamada **Categoria de Kleisli**, denotada por $\mathcal{C}_T$, onde $T$ é o endofunctor que representa a monad.

Em termos intuitivos, a categoria de Kleisli é o espaço onde **as funções que retornam valores em contexto** (por exemplo, `a -> Maybe b` ou `a -> IO b`) se comportam como morfismos “puros”. Isso permite raciocinar sobre computações com efeitos da mesma forma que raciocinamos sobre funções puras.

Formalmente, para uma monad $(T, \eta, \mu)$ em $\mathcal{C}$:

- **Objetos:** são os mesmos objetos de $\mathcal{C}$.
- **Morfismos:** para objetos $A$ e $B$, temos  
  $Hom_{\mathcal{C}_T}(A, B) = Hom_{\mathcal{C}}(A, T B)$  
  ou seja, as setas em $\mathcal{C}_T$ são funções do tipo $A \to T B$.
- **Composição:** é definida usando a operação *bind*, ou equivalentemente $\mu$ e o *funtor* $T$:  
  $$
  g \circ_T f = \mu_B \circ T(g) \circ f
  $$
  
  ![visualiza claramente as setas A -> T B e B -> T C e sua composição Kleisli.](/assets/images/Kleisli1.webp)
  
  Em Haskell, essa composição é implementada pelo operador `(>=>)`:
  
  ```haskell
  (>=>) :: Monad m => (a -> m b) -> (b -> m c) -> (a -> m c)
  f >=> g = \x -> f x >>= g
  ```

- **Identidade:** para cada objeto $A$, a identidade é dada por  
  $id_A^T = \eta_A : A \to T A$,  
  que em Haskell corresponde a `return`.

---

#### Intuição: composição de computações com efeitos

Na categoria original $\mathcal{C}$, funções puras compõem-se normalmente:
$$
f : A \to B,\quad g : B \to C \implies g \circ f : A \to C
$$

Na categoria de Kleisli, as setas são funções que retornam valores *em contexto*:
$$
f : A \to T B,\quad g : B \to T C
$$

Como não podemos compor diretamente `f` e `g`, precisamos da estrutura monádica para “encadear” essas computações:
$$
g \circ_T f = \lambda x.\, f(x) >>= g
$$

Assim, `(>=>)` é a composição categórica em $\mathcal{C}_T$. 
Isso justifica matematicamente por que `bind` (`>>=`) é a operação central das Monads — ele é a *composição de morfismos* na categoria de Kleisli.

---

#### Exemplo em Haskell: Maybe e IO como categorias de Kleisli

1. **Com Maybe**

```haskell
f :: Int -> Maybe String
f x = if x > 0 then Just (show x) else Nothing

g :: String -> Maybe Int
g s = if length s < 3 then Just (read s + 1) else Nothing

-- Composição monádica (Kleisli)
h :: Int -> Maybe Int
h = f >=> g
```

Aqui, `h` é a composição de `f` e `g` dentro da categoria de Kleisli de `Maybe`.
O operador `(>=>)` garante que o `Nothing` propague corretamente e que o resultado só exista se todas as etapas anteriores forem bem-sucedidas.

2. **Com IO**

```haskell
lerNumero :: IO Int
lerNumero = do
  putStrLn "Digite um número:"
  readLn

mostrarDobro :: Int -> IO ()
mostrarDobro n = putStrLn ("O dobro é: " ++ show (2 * n))

programa :: IO ()
programa = lerNumero >=> mostrarDobro $ ()
```

No exemplo acima, `lerNumero` e `mostrarDobro` são funções `() -> IO a` e `a -> IO b`. 
Na categoria de Kleisli da monad `IO`, elas podem ser compostas diretamente, o que garante a **sequencialidade pura dos efeitos**.

A categoria de Kleisli formaliza o princípio de que **Monads permitem compor funções com efeitos** dentro de uma estrutura que respeita as leis da composição associativa e da identidade.
Ela mostra que, mesmo quando os efeitos são inevitáveis — exceções, estado, I/O —, a composição continua sendo **um processo matematicamente puro e previsível**.

Em resumo:

| Conceito | Categoria $\mathcal{C}$ | Categoria de Kleisli $\mathcal{C}_T$ |
|:----------|:--------------------------|:-------------------------------------|
| Morfismo | $A \to B$ | $A \to T B$ |
| Identidade | $id_A$ | $\eta_A$ (`return`) |
| Composição | $g \circ f$ | $\mu \circ T(g) \circ f$ (`>=>`) |

A Kleisli é, portanto, o **ambiente natural das Monads** — o espaço onde funções com efeitos podem ser tratadas como morfismos puros, e onde a teoria das categorias revela sua força como modelo formal da programação funcional.

### Leis das Monads (em Haskell)

Toda instância de `Monad` deve obedecer a três leis, análogas às leis categoriais:

| Lei | Código |
|:--------------------------|:---------------------------------------------|
| **Identidade à esquerda** | `return a >>= f == f a` |
| **Identidade à direita** | `m >>= return == m` |
| **Associatividade** | `(m >>= f) >>= g == m >>= (\x -> f x >>= g)` |

Essas leis garantem que o encadeamento de computações é previsível e que `return` é neutro.

### O açúcar sintático do-notation

Encadear `(>>=)` pode ficar visualmente poluído. Chamamos este problema de *o *inferno do callback*. A linguagem Haskell fornece a **do-notation** como um açúcar sintático que é traduzido diretamente para chamadas de `(>>=)`.

Este código com `do`:

```haskell
main_do = do
  x <- Just 5
  y <- Just 10
  return (x + y) -- Resulta em: Just 15
```

É traduzido pelo compilador para este código com `(>>=)`:

```haskell
main_bind =
  Just 5 >>= (\x  -> 
    Just 10 >>= (\y  -> 
      return (x + y)))
```

## Exemplos de Monads Importantes

### Maybe Monad — falhas controladas

Modela computações que podem falhar, propagando `Nothing` (*short-circuit*) automaticamente.

```haskell
safeDiv :: Int -> Int -> Maybe Int
safeDiv _ 0 = Nothing
safeDiv x y = Just (x `div` y)

program = do
  a <- safeDiv 10 2 -- a = 5
  b <- safeDiv a 0 -- b = Nothing
  c <- safeDiv b 1 -- Esta linha nunca executa
return (c + 1) -- O resultado final é Nothing
```

### Either Monad — falhas com mensagem

Similar ao `Maybe`, mas carrega um valor `Left String` no caso de erro.

```haskell
divide :: Double -> Double -> Either String Double
divide _ 0 = Left "Divisão por zero!"
divide x y = Right (x / y)

divide 10 2 >>= (\r -> Right (r * 2)) -- Right 10.0
divide 10 0 >>= (\r -> Right (r * 2)) -- Left "Divisão por zero!"
```

### [] (List) Monad — não determinismo

Modela computações que podem ter *múltiplos* resultados (ou nenhum). O bind (`>>=`) executa a função para cada elemento da lista e concatena os resultados.

```haskell
-- 'do' em Listas = produto cartesiano / "for aninhado"
pairs xs ys = do
  x <- xs -- Para cada x em xs...
  y <- ys -- ...para cada y em ys...
return (x, y) -- ...produza (x, y)

pairs [1,2] [3,4]
-- Resulta em: [(1,3),(1,4),(2,3),(2,4)]
```

Há, [aqui](https://onlinegdb.com/kvjxA75D4a), um código em Haskell para a esforçada leitora explorar os conceitos que acabamos de ver.

### IO Monad — efeitos colaterais puros

A linguagem Haskell é, por definição, **puramente funcional**. Isso significa que uma função não pode modificar o estado global do programa, nem depender de efeitos externos. Em termos matemáticos, cada função é um *morfismo* entre objetos (tipos), obedecendo à propriedade fundamental da pureza: **a mesma entrada sempre gera a mesma saída**.

Mas então surge uma questão inevitável: *como uma linguagem puramente funcional pode interagir com o mundo externo, que é essencialmente impuro?*  
Como ler uma entrada, escrever na tela, acessar o sistema de arquivos, ou gerar um número aleatório sem quebrar a pureza funcional?

A resposta categórica é a **Monad IO**.

#### IO como Functor, Applicative e Monad

O tipo `IO a` não representa o valor `a`, mas uma **descrição pura de uma computação que, quando executada, produzirá um valor de tipo `a` e possivelmente causará efeitos colaterais**. Dessa forma, o programa em Haskell não executa ações diretamente, ele constrói uma árvore de ações que o runtime do Haskell (GHC) executará posteriormente, fora do domínio puro da linguagem.

A `IO` é uma instância das três abstrações fundamentais:

```haskell
instance Functor IO where
  fmap f io = io >>= (return . f)

instance Applicative IO where
  pure = return
  mf <*> mx = do
    f <- mf
    x <- mx
    return (f x)

instance Monad IO where
  (>>=) = bindIO  -- definida internamente no runtime
```

Essas instâncias garantem que o comportamento de `IO` preserve as leis fundamentais de composição da teoria das categorias, permitindo combinar ações sequencialmente sem violar a pureza.

#### Estrutura categórica

Em termos formais, podemos enxergar `IO` como um **endofunctor** $T : \mathcal{C} \to \mathcal{C}$, onde:

- Os **objetos** são tipos puros de Haskell (como `Int`, `String`, `()`);
- Os **morfismos** são funções do tipo `a -> IO b`;
- A unidade $\eta : A \to T(A)$ é a função `return`;
- A multiplicação $\mu : T(T(A)) \to T(A)$ é a operação `join`, que *achata* camadas de ações encadeadas.

Essa estrutura obedece às leis das Monads:

1. **Identidade à esquerda:** `return a >>= f ≡ f a`
2. **Identidade à direita:** `m >>= return ≡ m`
3. **Associatividade:** `(m >>= f) >>= g ≡ m >>= (\x -> f x >>= g)`

Essas leis asseguram que, embora as ações tenham efeitos colaterais, **a composição delas seja puramente determinística** no nível semântico.

#### Encadeamento de ações com IO

O operador `(>>=)` (*bind*) é o responsável por encadear ações de I/O, garantindo a **ordem explícita de execução**. A atenta leitora pode considerar um exemplo clássico de interação com o usuário:

```haskell
main :: IO ()
main = do
  putStrLn "Qual é o seu nome?"
  nome <- getLine
  putStrLn ("Olá, " ++ nome ++ "!")
```

O código acima é matematicamente equivalente a:

```haskell
main_alt :: IO ()
main_alt =
  putStrLn "Qual é o seu nome?" >>= \_ ->
  getLine >>= \nome ->
  putStrLn ("Olá, " ++ nome ++ "!")
```

Observe que `putStrLn` e `getLine` são morfismos do tipo:

```haskell
putStrLn :: String -> IO ()
getLine  :: IO String
```

Cada linha dentro do bloco `do` é, na verdade, uma composição monádica. A notação `do` é apenas uma forma conveniente de encadear operações que retornam `IO`.

#### Separação entre descrição e execução

A pureza é preservada porque a **execução das ações** não ocorre dentro da função, ela está apenas *descrita*. O runtime do Haskell é o responsável por interpretar essa descrição, realizando os efeitos colaterais no mundo real. 

Isso significa que, matematicamente, cada ação `IO a` é um elemento de uma **categoria de Kleisli** associada à Monad `IO`:

$$
\text{Kleisli}(\text{IO}) : A \xrightarrow{f} \text{IO } B
$$

Essa categoria permite que componhamos funções impuras, com efeitos, de maneira pura, através do operador `(>=>)`:

```haskell
(>=>) :: (a -> IO b) -> (b -> IO c) -> (a -> IO c)
f >=> g = \x -> f x >>= g
```

Por exemplo:

```haskell
saudacao :: String -> IO ()
saudacao nome = putStrLn ("Olá, " ++ nome ++ "!")

obterNome :: IO String
obterNome = do
  putStrLn "Digite seu nome:"
  getLine

programa :: IO ()
programa = obterNome >>= saudacao
-- Ou, de forma categórica:
-- programa = obterNome >=> saudacao
```

#### Combinando efeitos

O poder da `IO` Monad aparece ao compor várias ações que produzem e consomem dados. 
Por exemplo, podemos construir um pequeno programa interativo que lê números, os processa e exibe resultados:

```haskell
lerNumero :: String -> IO Int
lerNumero prompt = do
  putStrLn prompt
  input <- getLine
  return (read input)

somaNumeros :: IO ()
somaNumeros = do
  x <- lerNumero "Digite o primeiro número:"
  y <- lerNumero "Digite o segundo número:"
  putStrLn ("A soma é: " ++ show (x + y))
```

Nesse caso, cada chamada de `lerNumero` é um morfismo `() -> IO Int`, e `somaNumeros` é a composição monádica dessas ações. 
Do ponto de vista matemático, estamos compondo morfismos dentro da categoria de Kleisli de `IO`:

$$
() \xrightarrow{lerNumero} \text{IO Int} \xrightarrow{fmap (+)} \text{IO (Int -> Int)} \xrightarrow{<*>} \text{IO Int}
$$

#### Composição e transformação de ações

Além do encadeamento sequencial, podemos transformar o resultado de ações `IO` com `fmap` e `<*>`, pois `IO` também é um `Functor` e um `Applicative`.

```haskell
dobrarEntrada :: IO ()
dobrarEntrada = do
  putStrLn "Digite um número:"
  n <- readLn
  print (n * 2)
```

Pode ser reescrito usando composição funcional pura:

```haskell
dobrarEntrada' :: IO ()
dobrarEntrada' = fmap (*2) readLn >>= print
```

Ou ainda, com aplicação dentro do contexto:

```haskell
dobrarEntrada'' :: IO ()
dobrarEntrada'' = print =<< fmap (*2) readLn
```

Esses exemplos ilustram que **a Monad IO é compatível com o restante da hierarquia Functor–Applicative–Monad**, mantendo as mesmas propriedades de composição funcional.

#### Reflexão categórica final

Do ponto de vista categórico, `IO` é uma *monad de efeitos*, cuja interpretação semântica é dada pelo functor de Kleisli:

$$
T = (\text{ações que produzem efeitos}) : \mathcal{C} \to \mathcal{C}
$$

Através de `bind` e `return`, podemos compor ações de modo associativo, mantendo a semântica pura no nível das transformações de tipos. Assim, o Haskell preserva a pureza da função matemática, enquanto expressa programas que interagem com o mundo real.

Em outras palavras, **`IO` não quebra a pureza de Haskell — ela a estende ao domínio dos efeitos**, fornecendo uma ponte entre o cálculo funcional puro e a realidade impura da execução.

## A Jornada da Abstração

A jornada que a atenta leitora percorreu:
**Categorias → Functores → Applicatives → Monads** é a espinha dorsal da programação funcional moderna.

A Teoria das Categorias não é uma abstração gratuita; ela fornece o vocabulário e as leis que garantem composicionalidade e segurança.

| Conceito Matemático | Estrutura em Haskell | Exemplo |
|:--------------------------|:---------------------|:---------------------------|
| **Objeto** | Tipo | `Int` |
| **Morfismo** | Função pura | `(+1)` |
| **Functor** (endoFunctor) | `Functor` | `fmap (+1) (Just 5)` |
| **Unidade** ($\eta$) | `pure` / `return` | `pure 5 :: Maybe Int` |
| **Multiplicação** ($\mu$) | `join` | `join (Just (Just 5))` |
| **Monad** | `Monad` com `(>>=)` | `Just 5 >>= return . (+1)` |

Finalmente podemos afirmar que Monads não são mágicas: são um padrão de design formal, baseado em matemática rigorosa, para sequenciar computações dependentes de forma pura, segura e composível.

### Exercícios 3

1. **Monad Laws**: Usando a definição da `Maybe` monad, prove a "Identidade à Esquerda" (`return a >>= f == f a`).

2. **do-notation**: Reescreva a seguinte expressão usando `do-notation`:\
 `safeDiv 100 2 >>= (\a -> safeDiv a 5 >>= (\b -> return (b + 1)))`

3. **List Monad**: O que a seguinte expressão do calcula?

``` haskell
 do 
 n <- [1, 2, 3] 
 guard (odd n) -- guard :: Bool -> [()] 
 return (n * 10)
```

(Dica: `guard True = [()]`, `guard False = []`)