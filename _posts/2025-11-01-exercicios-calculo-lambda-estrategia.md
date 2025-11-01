---
layout: post
title: "Praticando Cálculo Lambda: Exercícios e Soluções 2"
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
image: assets/images/girllambda.webp
featured: false
rating: 5
description: Exercícios para prática estratégias de avaliação.
date: 2025-11-01T15:01:39.506Z
preview: Praticar cálculo lambda é essencial para compreender os fundamentos da computação funcional e a teoria dos tipos. Neste post, apresento uma série de exercícios práticos que abrangem conceitos como recursão, redução beta e currying. Cada exercício é acompanhado por uma solução detalhada para facilitar o aprendizado.
keywords: |-
  Algoritmos
  Exercícios
  cálculo lambda
  recursão
  redução beta
  currying
toc: true
published: true
lastmod: 2025-11-01T19:08:34.554Z
draft: 2025-11-01T15:01:40.919Z
---

## Antes de Começar

Para facilitar a compreensão dos conceitos principais, estes exercícios utilizam uma notação que permite operações aritméticas básicas diretamente no cálculo lambda, como adição, subtração, multiplicação, exponenciação e comparações. Em um cálculo lambda puramente teórico, estas operações seriam implementadas usando codificação de Church, mas para os propósitos pedagógicos desta aula, assumiremos que operações como $x + y$, $x \times y$, $x - y$, $x^y$ e comparações como $(n = 0)$ estão disponíveis como primitivas.

Estes dez exercícios exploraram as três principais estratégias de avaliação no cálculo lambda e suas implicações práticas:

**_call-by-value_:** Avalia argumentos antes de passá-los às funções. Simples e previsível, mas pode fazer trabalho desnecessário e pode não terminar mesmo quando um resultado existe.

**Call-by-Name:** Substitui argumentos sem avaliá-los, adiando a computação até que seja necessária. Mais poderosa em termos de terminação, mas pode reavaliar a mesma expressão múltiplas vezes.

**_call-by-need_:** Combina_call-by-name_ com compartilhamento, avaliando cada expressão no máximo uma vez. Oferece o melhor dos dois mundos: evita trabalho desnecessário e não recomputa expressões.


## PARTE 1: ENUNCIADOS

### Exercício 1: Diferença Básica entre _call-by-value_ e_call-by-name_

Considere a seguinte função e aplicação:

$$f = \lambda x. (\lambda y. x)$$
$$expressao = f\ (3 + 4)\ (5 \times 2)$$

a) Realize a redução completa usando estratégia _call-by-value_, mostrando quando cada operação aritmética é avaliada

b) Realize a redução completa usando estratégia_call-by-name_, mostrando quando cada operação aritmética é avaliada

c) Compare os resultados e explique qual estratégia realizou menos operações

### Exercício 2: Função que Ignora Argumentos

Dada a função:

$$constante = \lambda x. (\lambda y. x)$$

Aplique esta função a dois argumentos:

$$resultado = constante\ 5\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

Observe que o segundo argumento $(\lambda z. z\ z)\ (\lambda z. z\ z)$ é um combinador que não termina quando avaliado.

a) Tente reduzir usando _call-by-value_. O que acontece?

b) Reduza usando_call-by-name_. O processo termina?

c) Explique por que as duas estratégias levam a resultados diferentes

### Exercício 3: Currying e Estratégias de Avaliação

Considere a função curried:

$$multiplica\_e\_soma = \lambda x. (\lambda y. (\lambda z. (x \times y) + z))$$

Aplique esta função aos argumentos:

$$resultado = multiplica\_e\_soma\ (2 + 3)\ (4 + 5)\ (6 + 7)$$

a) Realize a redução usando _call-by-value_

b) Realize a redução usando_call-by-name_

c) Conte quantas operações aritméticas foram realizadas em cada estratégia

### Exercício 4: Compartilhamento em _call-by-need_

Considere a expressão:

$$duplica = \lambda x. x + x$$
$$resultado = duplica\ (2 \times 3)$$

a) Reduza usando _call-by-value_

b) Reduza usando_call-by-name_

c) Reduza usando _call-by-need_, indicando quando o compartilhamento ocorre

d) Compare o número de multiplicações realizadas em cada estratégia

### Exercício 5: Recursão com Diferentes Estratégias

Utilizando o combinador Y e a função fatorial:

$$Fatorial = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times f\ (n - 1))$$
$$fatorial = Y\ Fatorial$$

Considere a aplicação:

$$usa\_fatorial = \lambda x. (\lambda y. \text{if}\ (x = 0)\ \text{then}\ 0\ \text{else}\ y)$$
$$resultado = usa\_fatorial\ 0\ (fatorial\ 1000)$$

a) Descreva o que acontece com _call-by-value_

b) Descreva o que acontece com_call-by-name_

c) Explique qual estratégia é mais eficiente neste caso e por quê

### Exercício 6: Aplicação Parcial e Avaliação Estrita vs Preguiçosa

Dada a função curried:

$$processa = \lambda x. (\lambda y. (\lambda z. \text{if}\ (x > 0)\ \text{then}\ y\ \text{else}\ z))$$

E a aplicação:

$$resultado = processa\ (-1)\ (fatorial\ 100)\ 42$$

no qual, `fatorial` é uma função recursiva definida com o combinador Y.

a) Trace a redução usando _call-by-value_, indicando quando `fatorial 100` seria calculado

b) Trace a redução usando_call-by-name_, indicando se `fatorial 100` é calculado

c) Explique como a aplicação parcial interage com a estratégia de avaliação

### Exercício 7: _call-by-need_ e Múltiplas Referências

Considere a função:

$$usa\_tres\_vezes = \lambda x. x + x + x$$

E a aplicação:

$$resultado = usa\_tres\_vezes\ (10 + 20)$$

a) Reduza usando _call-by-value_, contando as operações de adição

b) Reduza usando_call-by-name_, contando as operações de adição

c) Reduza usando _call-by-need_, mostrando explicitamente o compartilhamento e contando as operações

d) Generalize: para uma função que usa um argumento $n$ vezes, como as estratégias se comparam?

### Exercício 8: Recursão Infinita e Ordem de Avaliação

Defina uma função recursiva infinita:

$$infinito = Y\ (\lambda f. \lambda x. f\ x)$$

Considere a expressão condicional:

$$escolhe = \lambda cond. (\lambda a. (\lambda b. \text{if}\ cond\ \text{then}\ a\ \text{else}\ b))$$
$$resultado = escolhe\ \text{true}\ 42\ (infinito\ 0)$$

a) O que acontece com _call-by-value_?

b) O que acontece com_call-by-name_?

c) O que acontece com _call-by-need_?

d) Qual propriedade do cálculo lambda esta situação ilustra?

### Exercício 9: Composição de Funções e Avaliação Preguiçosa

Considere o operador de composição:

$$compose = \lambda f. (\lambda g. (\lambda x. f\ (g\ x)))$$

E as funções:

$$cara = \lambda x. x \times 2$$
$$custosa = \lambda x. fatorial\ x$$

No qual `fatorial` é definido recursivamente. Considere:

$$h = compose\ cara\ custosa$$
$$escolha = \lambda flag. (\lambda val. \text{if}\ flag\ \text{then}\ 0\ \text{else}\ val)$$
$$resultado = escolha\ \text{true}\ (h\ 50)$$

a) Trace a redução usando _call-by-value_

b) Trace a redução usando_call-by-name_

c) Explique como a composição e a estratégia de avaliação interagem

### Exercício 10: Desafio - Fibonacci com Diferentes Estratégias

Considere a definição ingênua de Fibonacci:

$$Fib = \lambda f. (\lambda n. \text{if}\ (n \leq 1)\ \text{then}\ n\ \text{else}\ f\ (n-1) + f\ (n-2))$$
$$fib = Y\ Fib$$

E a função que usa o resultado duas vezes:

$$usa\_dobrado = \lambda x. x \times x$$
$$resultado = usa\_dobrado\ (fib\ 5)$$

a) Descreva quantas chamadas recursivas seriam feitas para calcular `fib 5` uma única vez

b) Com _call-by-value_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

c) Com_call-by-name_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

d) Com _call-by-need_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

e) Explique por que _call-by-need_ oferece vantagem significativa neste cenário

## PARTE 2: ENUNCIADOS COM SOLUÇÕES

### Exercício 1: Diferença Básica entre _call-by-value_ e_call-by-name_

Considere a seguinte função e aplicação:

$$f = \lambda x. (\lambda y. x)$$
$$expressao = f\ (3 + 4)\ (5 \times 2)$$

a) Realize a redução completa usando estratégia _call-by-value_, mostrando quando cada operação aritmética é avaliada

b) Realize a redução completa usando estratégia_call-by-name_, mostrando quando cada operação aritmética é avaliada

c) Compare os resultados e explique qual estratégia realizou menos operações

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, avaliamos os argumentos antes de passá-los à função. Vamos reduzir passo a passo:

**Passo 1:** Avaliação do primeiro argumento

$$expressao = f\ (3 + 4)\ (5 \times 2)$$

Antes de aplicar $f$ ao primeiro argumento, precisamos avaliar $(3 + 4)$:

$$expressao = f\ 7\ (5 \times 2)$$

**Passo 2:** Avaliação do segundo argumento

Antes de aplicar $f\ 7$ ao segundo argumento, avaliamos $(5 \times 2)$:

$$expressao = f\ 7\ 10$$

**Passo 3:** Primeira aplicação de função

Agora aplicamos $f$ ao valor 7:

$$f\ 7 = (\lambda x. (\lambda y. x))\ 7$$

Beta-redução, substituindo $x$ por 7:

$$f\ 7 = \lambda y. 7$$

Portanto:

$$expressao = (\lambda y. 7)\ 10$$

**Passo 4:** Segunda aplicação de função

$$expressao = (\lambda y. 7)\ 10$$

Beta-redução, mas observe que $y$ não aparece no corpo da função, então simplesmente obtemos:

$$expressao = 7$$

**Resultado com CBV:** 7

**Operações realizadas:** Duas operações aritméticas foram executadas: $(3 + 4)$ e $(5 \times 2)$.

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, substituímos os argumentos sem avaliá-los primeiro. Vamos reduzir:

**Passo 1:** Primeira aplicação de função

$$expressao = f\ (3 + 4)\ (5 \times 2)$$

Substituindo a definição de $f$:

$$expressao = (\lambda x. (\lambda y. x))\ (3 + 4)\ (5 \times 2)$$

Beta-redução, substituindo $x$ por $(3 + 4)$ sem avaliar:

$$expressao = (\lambda y. (3 + 4))\ (5 \times 2)$$

**Passo 2:** Segunda aplicação de função

Beta-redução, mas observe que $y$ não aparece no corpo, então o segundo argumento é descartado:

$$expressao = 3 + 4$$

**Passo 3:** Avaliação final

Agora precisamos avaliar a expressão resultante:

$$expressao = 7$$

**Resultado com CBN:** 7

**Operações realizadas:** Apenas uma operação aritmética foi executada: $(3 + 4)$. A operação $(5 \times 2)$ nunca foi avaliada porque o segundo argumento não foi usado no corpo da função.

**Parte c) Comparação**

Ambas as estratégias chegam ao mesmo resultado final: 7. No entanto, há uma diferença importante na eficiência:

_call-by-value_ realizou duas operações aritméticas, avaliando ambos os argumentos antes de aplicá-los à função. Isso incluiu a avaliação de $(5 \times 2)$ que acabou sendo desnecessária, pois o segundo argumento foi descartado pela função.

Call-by-name realizou apenas uma operação aritmética, avaliando apenas $(3 + 4)$. Como a função descarta o segundo argumento (a variável $y$ não aparece no corpo), a expressão $(5 \times 2)$ nunca foi avaliada.

**Conclusão:** Neste exemplo,_call-by-name_ é mais eficiente porque evita computação desnecessária. Quando uma função não usa todos os seus argumentos,_call-by-name_ pode evitar trabalho inútil ao adiar a avaliação até que seja realmente necessária. Este é um dos principais benefícios da avaliação preguiçosa.

### Exercício 2: Função que Ignora Argumentos

Dada a função:

$$constante = \lambda x. (\lambda y. x)$$

Aplique esta função a dois argumentos:

$$resultado = constante\ 5\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

Observe que o segundo argumento $(\lambda z. z\ z)\ (\lambda z. z\ z)$ é um combinador que não termina quando avaliado.

a) Tente reduzir usando _call-by-value_. O que acontece?

b) Reduza usando_call-by-name_. O processo termina?

c) Explique por que as duas estratégias levam a resultados diferentes

#### Solução

**Parte a) Tentativa de redução usando _call-by-value_**

Em _call-by-value_, precisamos avaliar ambos os argumentos antes de aplicá-los à função. Vamos tentar:

**Passo 1:** Avaliação do primeiro argumento

O primeiro argumento é simplesmente 5, que já está em forma normal (não pode ser reduzido mais). Então:

$$resultado = constante\ 5\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

**Passo 2:** Tentativa de avaliação do segundo argumento

Agora precisamos avaliar o segundo argumento: $(\lambda z. z\ z)\ (\lambda z. z\ z)$

Este é o famoso "combinador omega" ou termo que diverge. Vamos ver o que acontece quando tentamos avaliá-lo:

$$(\lambda z. z\ z)\ (\lambda z. z\ z)$$

Aplicando beta-redução, substituímos $z$ por $(\lambda z. z\ z)$:

$$= (\lambda z. z\ z)\ (\lambda z. z\ z)$$

Observe que chegamos exatamente à mesma expressão! Se tentarmos reduzir novamente, obteremos o mesmo resultado infinitamente.

**Conclusão para CBV:** O processo não termina. A avaliação entra em um loop infinito tentando avaliar o segundo argumento. Nunca conseguimos aplicar a função `constante`, pois ficamos presos tentando avaliar o argumento que diverge.

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, não avaliamos os argumentos antes de substituí-los. Vamos reduzir:

**Passo 1:** Primeira aplicação

$$resultado = constante\ 5\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

Substituindo a definição de `constante`:

$$resultado = (\lambda x. (\lambda y. x))\ 5\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

**Passo 2:** Beta-redução para o primeiro argumento

Substituímos $x$ por 5 sem avaliar nada:

$$resultado = (\lambda y. 5)\ ((\lambda z. z\ z)\ (\lambda z. z\ z))$$

**Passo 3:** Beta-redução para o segundo argumento

Observe que $y$ não aparece no corpo da função. Portanto, o segundo argumento é simplesmente descartado:

$$resultado = 5$$

**Conclusão para CBN:** O processo termina com sucesso! Obtemos o resultado 5. O termo divergente nunca foi avaliado porque não era necessário para computar o resultado.

**Parte c) Explicação das diferenças**

As duas estratégias levam a resultados completamente diferentes neste caso:

_call-by-value_ não termina porque tenta avaliar todos os argumentos antes de aplicar a função. Como o segundo argumento é um termo divergente (que reduz a si mesmo infinitamente), a avaliação fica presa em um loop infinito. A função nunca é aplicada, e o fato de que ela descartaria o segundo argumento torna-se irrelevante.

Call-by-name termina com sucesso porque substitui os argumentos sem avaliá-los primeiro. Quando a função é aplicada, o corpo revela que o segundo argumento não é necessário (a variável $y$ não aparece). Portanto, o termo divergente é simplesmente descartado sem nunca ser avaliado.

**Lição importante:** Esta diferença fundamental mostra que_call-by-name_ pode terminar em casos nos quais _call-by-value_ não termina. Na teoria do cálculo lambda, existe um teorema que formaliza isto: se uma expressão tem uma forma normal (um valor final), então a estratégia de avaliação que sempre reduz o redex mais à esquerda (similar a_call-by-name_) eventualmente a encontrará. _call-by-value_ não tem essa garantia.

Esta propriedade torna_call-by-name_ (e _call-by-need_, que é otimizado_call-by-name_) teoricamente mais poderosa em termos de terminação. No entanto, _call-by-value_ é mais previsível em termos de ordem de efeitos colaterais, razão pela qual é comum em linguagens imperativas.

### Exercício 3: Currying e Estratégias de Avaliação

Considere a função curried:

$$multiplica\_e\_soma = \lambda x. (\lambda y. (\lambda z. (x \times y) + z))$$

Aplique esta função aos argumentos:

$$resultado = multiplica\_e\_soma\ (2 + 3)\ (4 + 5)\ (6 + 7)$$

a) Realize a redução usando _call-by-value_

b) Realize a redução usando_call-by-name_

c) Conte quantas operações aritméticas foram realizadas em cada estratégia

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, cada argumento é completamente avaliado antes de ser passado à função.

**Passo 1:** Avaliação do primeiro argumento

$$resultado = multiplica\_e\_soma\ (2 + 3)\ (4 + 5)\ (6 + 7)$$

Avaliamos $(2 + 3) = 5$:

$$resultado = multiplica\_e\_soma\ 5\ (4 + 5)\ (6 + 7)$$

**Passo 2:** Aplicação ao primeiro argumento

$$resultado = (\lambda x. (\lambda y. (\lambda z. (x \times y) + z)))\ 5\ (4 + 5)\ (6 + 7)$$

Beta-redução, substituindo $x$ por 5:

$$resultado = (\lambda y. (\lambda z. (5 \times y) + z))\ (4 + 5)\ (6 + 7)$$

**Passo 3:** Avaliação do segundo argumento

Avaliamos $(4 + 5) = 9$:

$$resultado = (\lambda y. (\lambda z. (5 \times y) + z))\ 9\ (6 + 7)$$

**Passo 4:** Aplicação ao segundo argumento

Beta-redução, substituindo $y$ por 9:

$$resultado = (\lambda z. (5 \times 9) + z)\ (6 + 7)$$

**Passo 5:** Avaliação do terceiro argumento

Avaliamos $(6 + 7) = 13$:

$$resultado = (\lambda z. (5 \times 9) + z)\ 13$$

**Passo 6:** Aplicação ao terceiro argumento

Beta-redução, substituindo $z$ por 13:

$$resultado = (5 \times 9) + 13$$

**Passo 7:** Avaliação da multiplicação

$$resultado = 45 + 13$$

**Passo 8:** Avaliação da adição final

$$resultado = 58$$

**Resultado com CBV:** 58

**Operações aritméticas realizadas:**
- $(2 + 3) = 5$ (adição)
- $(4 + 5) = 9$ (adição)
- $(6 + 7) = 13$ (adição)
- $(5 \times 9) = 45$ (multiplicação)
- $45 + 13 = 58$ (adição)

**Total:** 5 operações (4 adições e 1 multiplicação)

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, os argumentos são substituídos sem avaliação prévia.

**Passo 1:** Primeira aplicação

$$resultado = multiplica\_e\_soma\ (2 + 3)\ (4 + 5)\ (6 + 7)$$

$$resultado = (\lambda x. (\lambda y. (\lambda z. (x \times y) + z)))\ (2 + 3)\ (4 + 5)\ (6 + 7)$$

Beta-redução, substituindo $x$ por $(2 + 3)$ sem avaliar:

$$resultado = (\lambda y. (\lambda z. ((2 + 3) \times y) + z))\ (4 + 5)\ (6 + 7)$$

**Passo 2:** Segunda aplicação

Beta-redução, substituindo $y$ por $(4 + 5)$ sem avaliar:

$$resultado = (\lambda z. ((2 + 3) \times (4 + 5)) + z)\ (6 + 7)$$

**Passo 3:** Terceira aplicação

Beta-redução, substituindo $z$ por $(6 + 7)$ sem avaliar:

$$resultado = ((2 + 3) \times (4 + 5)) + (6 + 7)$$

**Passo 4:** Avaliação da expressão resultante

Agora precisamos avaliar a expressão completa. Seguindo a ordem de precedência (parênteses internos primeiro, multiplicação antes de adição):

Avaliamos $(2 + 3) = 5$:

$$resultado = (5 \times (4 + 5)) + (6 + 7)$$

Avaliamos $(4 + 5) = 9$:

$$resultado = (5 \times 9) + (6 + 7)$$

Avaliamos $(5 \times 9) = 45$:

$$resultado = 45 + (6 + 7)$$

Avaliamos $(6 + 7) = 13$:

$$resultado = 45 + 13$$

Avaliamos a adição final:

$$resultado = 58$$

**Resultado com CBN:** 58

**Operações aritméticas realizadas:**
- $(2 + 3) = 5$ (adição)
- $(4 + 5) = 9$ (adição)
- $(5 \times 9) = 45$ (multiplicação)
- $(6 + 7) = 13$ (adição)
- $45 + 13 = 58$ (adição)

**Total:** 5 operações (4 adições e 1 multiplicação)

**Parte c) Comparação de operações**

Ambas as estratégias realizaram exatamente 5 operações aritméticas e chegaram ao mesmo resultado: 58.

A diferença está no momento em que as operações foram realizadas. Em _call-by-value_, cada argumento foi avaliado no momento em que foi passado à função, resultando em avaliação incremental. Em_call-by-name_, todas as substituições ocorreram primeiro, e depois toda a expressão foi avaliada de uma vez.

**Observação importante:** Neste exemplo específico, ambas as estratégias foram igualmente eficientes porque a função usa todos os seus argumentos exatamente uma vez. A diferença entre as estratégias torna-se significativa quando:

1. Um argumento não é usado (como vimos no Exercício 1), no qual_call-by-name_ evita trabalho desnecessário
2. Um argumento é usado múltiplas vezes (como veremos no Exercício 4), no qual_call-by-name_ pode fazer trabalho redundante

Este exercício demonstra que quando todos os argumentos são usados exatamente uma vez, as duas estratégias têm eficiência comparável, diferindo apenas na ordem de avaliação.

### Exercício 4: Compartilhamento em _call-by-need_

Considere a expressão:

$$duplica = \lambda x. x + x$$

$$resultado = duplica\ (2 \times 3)$$

a) Reduza usando _call-by-value_

b) Reduza usando_call-by-name_

c) Reduza usando _call-by-need_, indicando quando o compartilhamento ocorre

d) Compare o número de multiplicações realizadas em cada estratégia

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, o argumento é avaliado antes de ser passado à função.

**Passo 1:** Avaliação do argumento

$$resultado = duplica\ (2 \times 3)$$

Avaliamos $(2 \times 3) = 6$:

$$resultado = duplica\ 6$$

**Passo 2:** Aplicação da função

$$resultado = (\lambda x. x + x)\ 6$$

Beta-redução, substituindo $x$ por 6:

$$resultado = 6 + 6$$

**Passo 3:** Avaliação final

$$resultado = 12$$

**Resultado com CBV:** 12

**Multiplicações realizadas:** 1 (a multiplicação $2 \times 3$ foi realizada uma vez)

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, o argumento é substituído sem avaliação prévia.

**Passo 1:** Aplicação da função

$$resultado = duplica\ (2 \times 3)$$

$$resultado = (\lambda x. x + x)\ (2 \times 3)$$

Beta-redução, substituindo ambas as ocorrências de $x$ por $(2 \times 3)$:

$$resultado = (2 \times 3) + (2 \times 3)$$

**Passo 2:** Avaliação da primeira multiplicação

$$resultado = 6 + (2 \times 3)$$

**Passo 3:** Avaliação da segunda multiplicação

$$resultado = 6 + 6$$

**Passo 4:** Avaliação da adição

$$resultado = 12$$

**Resultado com CBN:** 12

**Multiplicações realizadas:** 2 (a multiplicação $2 \times 3$ foi realizada duas vezes, uma para cada ocorrência de $x$)

**Parte c) Redução usando _call-by-need_**

_call-by-need_ é como_call-by-name_, mas com compartilhamento: a primeira vez que um argumento precisa ser avaliado, o resultado é memorizado e reutilizado.

**Passo 1:** Aplicação da função

$$resultado = duplica\ (2 \times 3)$$

$$resultado = (\lambda x. x + x)\ (2 \times 3)$$

Beta-redução, mas agora criamos uma referência compartilhada. Vamos representar isso como:

$$\text{let}\ shared\_x = (2 \times 3)\ \text{in}\ shared\_x + shared\_x$$

**Passo 2:** Primeira avaliação de shared_x

Quando encontramos a primeira ocorrência de $shared\_x$, precisamos avaliá-la:

$$shared\_x = (2 \times 3) = 6$$

Agora este valor é memorizado. A expressão se torna:

$$\text{let}\ shared\_x = 6\ \text{in}\ 6 + shared\_x$$

**Passo 3:** Segunda referência a shared_x

Quando encontramos a segunda ocorrência de $shared\_x$, não precisamos recalculá-la. Simplesmente usamos o valor memorizado:

$$6 + 6$$

**Passo 4:** Avaliação final

$$resultado = 12$$

**Resultado com CBNeed:** 12

**Multiplicações realizadas:** 1 (a multiplicação $2 \times 3$ foi realizada apenas uma vez, e o resultado foi compartilhado)

**Parte d) Comparação das estratégias**

Vamos comparar as três estratégias em termos de eficiência:

**_call-by-value_:**
- Multiplicações: 1
- Estratégia: Avalia o argumento uma vez antes de passá-lo
- Vantagem: Simples e previsível
- Desvantagem: Pode avaliar argumentos que nunca serão usados

**Call-by-Name:**
- Multiplicações: 2
- Estratégia: Substitui o argumento textualmente, avaliando cada ocorrência independentemente
- Vantagem: Evita avaliar argumentos não usados
- Desvantagem: Pode reavaliar a mesma expressão múltiplas vezes

**_call-by-need_:**
- Multiplicações: 1
- Estratégia: Substitui o argumento, mas memoriza o resultado da primeira avaliação
- Vantagem: Combina os benefícios das outras duas estratégias
- Desvantagem: Requer overhead de gerenciamento de memória para o compartilhamento

**Análise geral:**

Neste exemplo, _call-by-value_ e _call-by-need_ têm o mesmo desempenho (1 multiplicação), enquanto_call-by-name_ é menos eficiente (2 multiplicações).

_call-by-need_ oferece o melhor dos dois mundos: assim como_call-by-name_, pode evitar avaliar argumentos não usados; mas, diferentemente de_call-by-name_, não reavalia a mesma expressão múltiplas vezes. Esta é a razão pela qual linguagens como Haskell usam _call-by-need_ (lazy evaluation) como estratégia padrão.

A diferença torna-se ainda mais dramática quando o argumento é usado muitas vezes ou quando a computação do argumento é muito custosa. Por exemplo, se a função fosse $\lambda x. x + x + x + x$,_call-by-name_ faria 4 multiplicações, enquanto _call-by-value_ e _call-by-need_ fariam apenas 1.

### Exercício 5: Recursão com Diferentes Estratégias

Utilizando o combinador Y e a função fatorial:

$$Fatorial = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times f\ (n - 1))$$
$$fatorial = Y\ Fatorial$$

Considere a aplicação:

$$usa\_fatorial = \lambda x. (\lambda y. \text{if}\ (x = 0)\ \text{then}\ 0\ \text{else}\ y)$$
$$resultado = usa\_fatorial\ 0\ (fatorial\ 1000)$$

a) Descreva o que acontece com _call-by-value_

b) Descreva o que acontece com_call-by-name_

c) Explique qual estratégia é mais eficiente neste caso e por quê

#### Solução

**Parte a) _call-by-value_**

Em _call-by-value_, todos os argumentos são avaliados antes de serem passados à função.

**Passo 1:** Avaliação do primeiro argumento

$$resultado = usa\_fatorial\ 0\ (fatorial\ 1000)$$

O primeiro argumento é simplesmente 0, que já está em forma normal:

$$resultado = usa\_fatorial\ 0\ (fatorial\ 1000)$$

**Passo 2:** Avaliação do segundo argumento

Agora precisamos avaliar $fatorial\ 1000$ antes de prosseguir. Isso envolve:

$$fatorial\ 1000 = 1000 \times fatorial\ 999$$

$$= 1000 \times (999 \times fatorial\ 998)$$

$$= 1000 \times 999 \times (998 \times fatorial\ 997)$$

Este processo continua recursivamente até:

$$= 1000 \times 999 \times 998 \times ... \times 2 \times 1$$

Esta é uma computação extremamente custosa! O fatorial de 1000 é um número gigantesco com 2568 dígitos. A computação envolve 1000 multiplicações recursivas.

**Passo 3:** Aplicação da função (depois de calcular o fatorial)

Somente depois de calcular completamente $fatorial\ 1000$, podemos aplicar a função:

$$resultado = (\lambda x. (\lambda y. \text{if}\ (x = 0)\ \text{then}\ 0\ \text{else}\ y))\ 0\ (valor\_gigante)$$

Beta-redução para o primeiro argumento:

$$resultado = (\lambda y. \text{if}\ (0 = 0)\ \text{then}\ 0\ \text{else}\ y)\ (valor\_gigante)$$

**Passo 4:** Avaliação condicional

Como $0 = 0$ é verdadeiro, a condicional retorna 0:

$$resultado = 0$$

**Resultado com CBV:** 0 (mas somente depois de calcular $fatorial\ 1000$ completamente)

**Custo:** Extremamente alto - realizamos 1000 multiplicações e calculamos um número gigantesco, apenas para descartá-lo imediatamente.

**Parte b)_call-by-name_**

Em_call-by-name_, os argumentos são substituídos sem avaliação prévia.

**Passo 1:** Primeira aplicação

$$resultado = usa\_fatorial\ 0\ (fatorial\ 1000)$$

$$resultado = (\lambda x. (\lambda y. \text{if}\ (x = 0)\ \text{then}\ 0\ \text{else}\ y))\ 0\ (fatorial\ 1000)$$

Beta-redução, substituindo $x$ por 0:

$$resultado = (\lambda y. \text{if}\ (0 = 0)\ \text{then}\ 0\ \text{else}\ y)\ (fatorial\ 1000)$$

**Passo 2:** Segunda aplicação

Beta-redução, substituindo $y$ por $(fatorial\ 1000)$ sem avaliar:

$$resultado = \text{if}\ (0 = 0)\ \text{then}\ 0\ \text{else}\ (fatorial\ 1000)$$

**Passo 3:** Avaliação condicional

A condição $(0 = 0)$ é verdadeira, então tomamos o ramo `then`:

$$resultado = 0$$

Observe que o ramo `else`, que contém $(fatorial\ 1000)$, nunca é avaliado!

**Resultado com CBN:** 0 (calculado imediatamente, sem avaliar o fatorial)

**Custo:** Mínimo - apenas avaliamos a comparação $(0 = 0)$ e retornamos 0. Nenhuma multiplicação foi realizada.

**Parte c) Comparação de eficiência**

A diferença de eficiência entre as duas estratégias neste caso é dramática:

**_call-by-value_:**
- Realiza 1000 multiplicações recursivas
- Calcula um número com 2568 dígitos
- Usa memória significativa para armazenar o resultado intermediário
- Todo esse trabalho é desperdiçado, pois o valor é imediatamente descartado
- Tempo de execução: muito alto (potencialmente segundos ou mais)

**Call-by-Name:**
- Realiza apenas uma comparação: $(0 = 0)$
- Não calcula o fatorial em momento algum
- Usa memória mínima
- Tempo de execução: instantâneo

**Por que esta diferença ocorre?**

A função `usa_fatorial` tem uma estrutura que torna a avaliação do segundo argumento condicional: ele só é necessário quando o primeiro argumento não é zero. Em linguagens que usam _call-by-value_, o programador precisa ter cuidado para evitar passar computações custosas que podem não ser necessárias. Uma solução comum é usar funções anônimas (thunks) para adiar a computação:

$$usa\_fatorial\_otimizado = \lambda x. (\lambda y. \text{if}\ (x = 0)\ \text{then}\ 0\ \text{else}\ y\ ())$$

na qual $y$ seria uma função que, quando chamada, computa o fatorial.

Em linguagens com_call-by-name_ ou _call-by-need_ (como Haskell), este padrão é automático: expressões são naturalmente adiadas até que sejam necessárias. Isso torna o código mais simples e naturalmente eficiente para casos como este.

**Lição fundamental:**_call-by-name_ (e _call-by-need_) podem ser significativamente mais eficientes quando trabalhamos com estruturas de controle como condicionais, nas quais nem todas as expressões precisam ser avaliadas. Esta é uma das razões pelas quais linguagens funcionais modernas como Haskell preferem avaliação preguiçosa por padrão.

### Exercício 6: Aplicação Parcial e Avaliação Estrita vs Preguiçosa

Dada a função curried:

$$processa = \lambda x. (\lambda y. (\lambda z. \text{if}\ (x > 0)\ \text{then}\ y\ \text{else}\ z))$$

E a aplicação:

$$resultado = processa\ (-1)\ (fatorial\ 100)\ 42$$

na qual `fatorial` é uma função recursiva definida com o combinador Y.

a) Trace a redução usando _call-by-value_, indicando quando `fatorial 100` seria calculado

b) Trace a redução usando_call-by-name_, indicando se `fatorial 100` é calculado

c) Explique como a aplicação parcial interage com a estratégia de avaliação

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, cada argumento é avaliado antes de ser passado à função.

**Passo 1:** Avaliação e aplicação do primeiro argumento

$$resultado = processa\ (-1)\ (fatorial\ 100)\ 42$$

O primeiro argumento, $(-1)$, já está em forma normal:

$$resultado = (\lambda x. (\lambda y. (\lambda z. \text{if}\ (x > 0)\ \text{then}\ y\ \text{else}\ z)))\ (-1)\ (fatorial\ 100)\ 42$$

Beta-redução, substituindo $x$ por $(-1)$:

$$resultado = (\lambda y. (\lambda z. \text{if}\ ((-1) > 0)\ \text{then}\ y\ \text{else}\ z))\ (fatorial\ 100)\ 42$$

**Passo 2:** Avaliação do segundo argumento

Aqui está o problema crítico! Antes de aplicar a função parcialmente construída ao segundo argumento, _call-by-value_ exige que avaliemos $fatorial\ 100$.

$$fatorial\ 100 = 100 \times 99 \times 98 \times ... \times 2 \times 1$$

Esta é uma computação custosa que resulta em um número com 158 dígitos! Vamos chamar este resultado gigante de $F_{100}$.

**Passo 3:** Aplicação do segundo argumento (após calcular o fatorial)

$$resultado = (\lambda y. (\lambda z. \text{if}\ ((-1) > 0)\ \text{then}\ y\ \text{else}\ z))\ F_{100}\ 42$$

Beta-redução, substituindo $y$ por $F_{100}$:

$$resultado = (\lambda z. \text{if}\ ((-1) > 0)\ \text{then}\ F_{100}\ \text{else}\ z)\ 42$$

**Passo 4:** Avaliação e aplicação do terceiro argumento

O terceiro argumento, 42, já está em forma normal:

Beta-redução, substituindo $z$ por 42:

$$resultado = \text{if}\ ((-1) > 0)\ \text{then}\ F_{100}\ \text{else}\ 42$$

**Passo 5:** Avaliação condicional

Avaliamos $(-1) > 0$, que é falso. Portanto, tomamos o ramo `else`:

$$resultado = 42$$

**Resultado com CBV:** 42

**Observação crítica:** Calculamos $fatorial\ 100$ completamente (100 multiplicações, resultando em um número gigante), apenas para descartá-lo! Todo esse trabalho foi desperdiçado porque a condição era falsa.

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, os argumentos são substituídos sem avaliação prévia.

**Passo 1:** Primeira aplicação

$$resultado = processa\ (-1)\ (fatorial\ 100)\ 42$$

$$resultado = (\lambda x. (\lambda y. (\lambda z. \text{if}\ (x > 0)\ \text{then}\ y\ \text{else}\ z)))\ (-1)\ (fatorial\ 100)\ 42$$

Beta-redução, substituindo $x$ por $(-1)$:

$$resultado = (\lambda y. (\lambda z. \text{if}\ ((-1) > 0)\ \text{then}\ y\ \text{else}\ z))\ (fatorial\ 100)\ 42$$

**Passo 2:** Segunda aplicação (sem avaliar fatorial 100)

Beta-redução, substituindo $y$ por $(fatorial\ 100)$ sem avaliar:

$$resultado = (\lambda z. \text{if}\ ((-1) > 0)\ \text{then}\ (fatorial\ 100)\ \text{else}\ z)\ 42$$

**Passo 3:** Terceira aplicação

Beta-redução, substituindo $z$ por 42:

$$resultado = \text{if}\ ((-1) > 0)\ \text{then}\ (fatorial\ 100)\ \text{else}\ 42$$

**Passo 4:** Avaliação condicional

Avaliamos $(-1) > 0$, que é falso. Portanto, tomamos o ramo `else`:

$$resultado = 42$$

**Resultado com CBN:** 42

**Observação importante:** O termo $(fatorial\ 100)$ nunca foi avaliado! Ele foi substituído no corpo da função, mas como a condicional escolheu o ramo `else`, a expressão contendo o fatorial nunca foi necessária.

**Parte c) Interação entre aplicação parcial e estratégia de avaliação**

A aplicação parcial e a estratégia de avaliação interagem de maneiras sutis mas importantes:

**Com _call-by-value_:**

Quando aplicamos parcialmente uma função curried, cada argumento é avaliado no momento em que é fornecido, independentemente de ser usado posteriormente. No nosso exemplo:

1. Aplicamos $(-1)$: este valor é usado na condicional, então sua avaliação faz sentido
2. Aplicamos $(fatorial\ 100)$: este valor é completamente calculado, mesmo que a condicional possa não usá-lo
3. Aplicamos 42: este valor é usado porque a condicional escolheu o ramo `else`

O problema é que a avaliação do segundo argumento ocorreu prematuramente. A estrutura curried não protegeu contra esta avaliação desnecessária.

**Com_call-by-name_:**

A aplicação parcial funciona de forma mais eficiente porque cada argumento é apenas substituído, não avaliado. A avaliação é adiada até que o valor seja realmente necessário. No nosso exemplo:

1. Aplicamos $(-1)$: substituído, e eventualmente avaliado quando necessário para a comparação
2. Aplicamos $(fatorial\ 100)$: apenas substituído, nunca avaliado porque não foi necessário
3. Aplicamos 42: substituído e usado porque a condicional escolheu o ramo `else`

**Lição sobre design de APIs:**

Este exemplo ilustra por que a ordem dos argumentos em funções curried é importante em linguagens com diferentes estratégias de avaliação:

Em linguagens com _call-by-value_, argumentos que são usados condicionalmente ou raramente deveriam vir por último, permitindo que o usuário decida se quer aplicá-los. Alternativamente, deveriam ser encapsulados em funções (thunks) para adiar sua avaliação.

Em linguagens com_call-by-name_ ou _call-by-need_, a ordem dos argumentos é menos crítica para eficiência, pois a avaliação é automaticamente adiada até ser necessária. Isso torna o código mais simples e naturalmente eficiente.

**Observação sobre currying e efeitos colaterais:**

Em linguagens com efeitos colaterais, a diferença entre as estratégias torna-se ainda mais significativa. Com _call-by-value_, efeitos colaterais ocorrem no momento da aplicação. Com_call-by-name_, ocorrem apenas quando o valor é usado (e podem ocorrer múltiplas vezes se o argumento for usado múltiplas vezes). Este é um dos motivos pelos quais linguagens com efeitos colaterais geralmente preferem _call-by-value_, apesar de sua menor eficiência em casos como o deste exercício.

### Exercício 7: _call-by-need_ e Múltiplas Referências

Considere a função:

$$usa\_tres\_vezes = \lambda x. x + x + x$$

E a aplicação:

$$resultado = usa\_tres\_vezes\ (10 + 20)$$

a) Reduza usando _call-by-value_, contando as operações de adição

b) Reduza usando_call-by-name_, contando as operações de adição

c) Reduza usando _call-by-need_, mostrando explicitamente o compartilhamento e contando as operações

d) Generalize: para uma função que usa um argumento $n$ vezes, como as estratégias se comparam?

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, o argumento é completamente avaliado antes de ser passado à função.

**Passo 1:** Avaliação do argumento

$$resultado = usa\_tres\_vezes\ (10 + 20)$$

Avaliamos $(10 + 20) = 30$:

$$resultado = usa\_tres\_vezes\ 30$$

**Contagem:** 1 adição realizada

**Passo 2:** Aplicação da função

$$resultado = (\lambda x. x + x + x)\ 30$$

Beta-redução, substituindo todas as ocorrências de $x$ por 30:

$$resultado = 30 + 30 + 30$$

**Passo 3:** Primeira adição

$$resultado = 60 + 30$$

**Contagem:** 1 adição realizada (total: 2)

**Passo 4:** Segunda adição

$$resultado = 90$$

**Contagem:** 1 adição realizada (total: 3)

**Resultado com CBV:** 90

**Total de adições:** 3 operações
- 1 para calcular o argumento $(10 + 20)$
- 2 para calcular o resultado final ($30 + 30 + 30$)

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, o argumento é substituído sem avaliação prévia.

**Passo 1:** Aplicação da função

$$resultado = usa\_tres\_vezes\ (10 + 20)$$

$$resultado = (\lambda x. x + x + x)\ (10 + 20)$$

Beta-redução, substituindo cada ocorrência de $x$ por $(10 + 20)$:

$$resultado = (10 + 20) + (10 + 20) + (10 + 20)$$

**Passo 2:** Primeira avaliação de $(10 + 20)$

$$resultado = 30 + (10 + 20) + (10 + 20)$$

**Contagem:** 1 adição realizada

**Passo 3:** Segunda avaliação de $(10 + 20)$

$$resultado = 30 + 30 + (10 + 20)$$

**Contagem:** 1 adição realizada (total: 2)

**Passo 4:** Terceira avaliação de $(10 + 20)$

$$resultado = 30 + 30 + 30$$

**Contagem:** 1 adição realizada (total: 3)

**Passo 5:** Primeira adição do resultado

$$resultado = 60 + 30$$

**Contagem:** 1 adição realizada (total: 4)

**Passo 6:** Segunda adição do resultado

$$resultado = 90$$

**Contagem:** 1 adição realizada (total: 5)

**Resultado com CBN:** 90

**Total de adições:** 5 operações

- 3 para calcular $(10 + 20)$ três vezes (uma para cada ocorrência de $x$)
- 2 para calcular o resultado final

**Parte c) Redução usando _call-by-need_**

_call-by-need_ funciona como_call-by-name_, mas memoriza o resultado da primeira avaliação de cada argumento.

**Passo 1:** Aplicação da função com compartilhamento

$$resultado = usa\_tres\_vezes\ (10 + 20)$$

$$resultado = (\lambda x. x + x + x)\ (10 + 20)$$

Beta-redução, mas criamos uma referência compartilhada:

$$\text{let}\ shared\_x = (10 + 20)\ \text{in}\ shared\_x + shared\_x + shared\_x$$

Neste ponto, $shared\_x$ ainda não foi avaliado. É apenas uma referência à expressão $(10 + 20)$.

**Passo 2:** Primeira referência a shared_x - avaliação e memorização

Quando encontramos a primeira ocorrência de $shared\_x$, precisamos avaliá-la:

$$shared\_x = (10 + 20) = 30$$

**Contagem:** 1 adição realizada

Este valor é agora memorizado. Todas as futuras referências a $shared\_x$ usarão este valor sem reavaliá-lo:

$$\text{let}\ shared\_x = 30\ \text{in}\ 30 + shared\_x + shared\_x$$

**Passo 3:** Segunda referência a shared_x - uso do valor memorizado

$$\text{let}\ shared\_x = 30\ \text{in}\ 30 + 30 + shared\_x$$

**Contagem:** 0 adições (apenas recuperamos o valor memorizado)

**Passo 4:** Terceira referência a shared_x - uso do valor memorizado

$$\text{let}\ shared\_x = 30\ \text{in}\ 30 + 30 + 30$$

**Contagem:** 0 adições (novamente, apenas recuperamos o valor memorizado)

**Passo 5:** Primeira adição do resultado

$$60 + 30$$

**Contagem:** 1 adição realizada (total: 2)

**Passo 6:** Segunda adição do resultado

$$resultado = 90$$

**Contagem:** 1 adição realizada (total: 3)

**Resultado com CBNeed:** 90

**Total de adições:** 3 operações
- 1 para calcular $(10 + 20)$ uma única vez (com memorização)
- 2 para calcular o resultado final

**Parte d) Generalização para n usos**

Vamos analisar o comportamento de cada estratégia quando uma função usa seu argumento $n$ vezes, e o argumento requer $k$ operações para ser calculado.

**_call-by-value_:**

- Calcula o argumento uma vez: $k$ operações
- Usa o valor pré-computado $n$ vezes: 0 operações adicionais por uso
- Total de operações extras: $k$ (constante, independente de $n$)

**Call-by-Name:**

- Não calcula o argumento antecipadamente: 0 operações
- Recalcula o argumento cada vez que é usado: $n \times k$ operações
- Total de operações extras: $n \times k$ (linear em $n$)

**_call-by-need_:**

- Calcula o argumento na primeira vez que é usado: $k$ operações
- Usa o valor memorizado nas $n-1$ vezes restantes: 0 operações adicionais
- Total de operações extras: $k$ (constante, independente de $n$)

**Comparação:**

Para um argumento usado $n$ vezes com custo de computação $k$:

| Estratégia    | Operações | Complexidade |
|---------------|-----------|--------------|
| _call-by-value_ | $k$       | $O(k)$       |
|_call-by-name_  | $n \times k$ | $O(nk)$   |
| _call-by-need_  | $k$       | $O(k)$       |

**Conclusões:**

1. **_call-by-value_ e _call-by-need_ têm desempenho idêntico** quando todos os argumentos são usados: ambos calculam cada argumento exatamente uma vez.

2. **Call-by-name é ineficiente** quando argumentos são usados múltiplas vezes, pois recalcula a mesma expressão repetidamente. O custo cresce linearmente com o número de usos.

3. **_call-by-need_ oferece o melhor dos dois mundos:**
   - Como_call-by-name_, evita calcular argumentos que nunca são usados (não demonstrado neste exercício específico, mas vimos em exercícios anteriores)
   - Como _call-by-value_, calcula cada argumento no máximo uma vez, independentemente de quantas vezes é usado
   - Adiciona apenas o overhead de gerenciar o compartilhamento/memorização

**Implicações práticas:**

Em Haskell, que usa _call-by-need_, o programador pode escrever código naturalmente eficiente sem se preocupar se argumentos serão usados múltiplas vezes. A linguagem garante que cada expressão é avaliada no máximo uma vez.

Em linguagens com _call-by-value_ como Python ou JavaScript, o programador tem controle explícito sobre quando valores são computados, mas precisa ter cuidado ao passar funções ou computações custosas como argumentos.

Em linguagens puramente_call-by-name_ (que são raras na prática), o programador precisa ser muito cuidadoso com reutilização de argumentos, potencialmente introduzindo variáveis locais manualmente para evitar recomputação.

### Exercício 8: Recursão Infinita e Ordem de Avaliação

Defina uma função recursiva infinita:

$$infinito = Y\ (\lambda f. \lambda x. f\ x)$$

Considere a expressão condicional:

$$escolhe = \lambda cond. (\lambda a. (\lambda b. \text{if}\ cond\ \text{then}\ a\ \text{else}\ b))$$
$$resultado = escolhe\ \text{true}\ 42\ (infinito\ 0)$$

a) O que acontece com _call-by-value_?

b) O que acontece com_call-by-name_?

c) O que acontece com _call-by-need_?

d) Qual propriedade do cálculo lambda esta situação ilustra?

#### Solução

**Análise da função infinito**

Antes de resolver o exercício, vamos entender o que é a função `infinito`. Aplicando a definição do combinador Y:

$$infinito = Y\ (\lambda f. \lambda x. f\ x)$$

Quando aplicamos `infinito` a um argumento:

$$infinito\ 0 = (\lambda f. \lambda x. f\ x)\ infinito\ 0$$
$$= (\lambda x. infinito\ x)\ 0$$
$$= infinito\ 0$$

Observe que chegamos exatamente à mesma expressão! Esta função diverge: ela se reduz a si mesma infinitamente, nunca chegando a um valor.

**Parte a) _call-by-value_**

Em _call-by-value_, todos os argumentos devem ser completamente avaliados antes de serem passados à função.

**Passo 1:** Avaliação dos argumentos

$$resultado = escolhe\ \text{true}\ 42\ (infinito\ 0)$$

Substituindo a definição de `escolhe`:

$$resultado = (\lambda cond. (\lambda a. (\lambda b. \text{if}\ cond\ \text{then}\ a\ \text{else}\ b)))\ \text{true}\ 42\ (infinito\ 0)$$

**Passo 2:** Aplicação do primeiro argumento

O primeiro argumento, `true`, já está em forma normal:

$$resultado = (\lambda a. (\lambda b. \text{if}\ \text{true}\ \text{then}\ a\ \text{else}\ b))\ 42\ (infinito\ 0)$$

**Passo 3:** Aplicação do segundo argumento

O segundo argumento, 42, já está em forma normal:

$$resultado = (\lambda b. \text{if}\ \text{true}\ \text{then}\ 42\ \text{else}\ b)\ (infinito\ 0)$$

**Passo 4:** Tentativa de avaliação do terceiro argumento

Aqui está o problema! _call-by-value_ exige que avaliemos $(infinito\ 0)$ antes de aplicá-lo à função:

$$infinito\ 0 = (\lambda x. infinito\ x)\ 0 = infinito\ 0 = ...$$

Esta avaliação nunca termina! Entramos em um loop infinito.

**Resultado com CBV:** O programa não termina. Fica eternamente tentando avaliar $(infinito\ 0)$.

**Parte b)_call-by-name_**

Em_call-by-name_, os argumentos são substituídos sem avaliação prévia.

**Passo 1:** Primeira aplicação

$$resultado = escolhe\ \text{true}\ 42\ (infinito\ 0)$$

$$resultado = (\lambda cond. (\lambda a. (\lambda b. \text{if}\ cond\ \text{then}\ a\ \text{else}\ b)))\ \text{true}\ 42\ (infinito\ 0)$$

Beta-redução, substituindo $cond$ por `true`:

$$resultado = (\lambda a. (\lambda b. \text{if}\ \text{true}\ \text{then}\ a\ \text{else}\ b))\ 42\ (infinito\ 0)$$

**Passo 2:** Segunda aplicação

Beta-redução, substituindo $a$ por 42:

$$resultado = (\lambda b. \text{if}\ \text{true}\ \text{then}\ 42\ \text{else}\ b)\ (infinito\ 0)$$

**Passo 3:** Terceira aplicação

Beta-redução, substituindo $b$ por $(infinito\ 0)$ sem avaliar:

$$resultado = \text{if}\ \text{true}\ \text{then}\ 42\ \text{else}\ (infinito\ 0)$$

**Passo 4:** Avaliação condicional

A condição `true` é verdadeira, então tomamos o ramo `then`:

$$resultado = 42$$

Observe que o ramo `else`, que contém $(infinito\ 0)$, nunca é avaliado!

**Resultado com CBN:** 42 (o programa termina com sucesso)

**Parte c) _call-by-need_**

_call-by-need_ funciona exatamente como_call-by-name_ neste caso, mas vamos traçar explicitamente o compartilhamento.

**Passo 1 a 3:** Aplicações e substituições

Os passos são idênticos ao_call-by-name_, criando uma referência compartilhada:

$$\text{let}\ b\_compartilhado = (infinito\ 0)\ \text{in}\ \text{if}\ \text{true}\ \text{then}\ 42\ \text{else}\ b\_compartilhado$$

**Passo 4:** Avaliação condicional

A condição `true` é verdadeira, então tomamos o ramo `then`:

$$resultado = 42$$

A referência $b\_compartilhado$ nunca é forçada (nunca precisamos avaliar seu valor), então $(infinito\ 0)$ nunca é computado.

**Resultado com CBNeed:** 42 (o programa termina com sucesso)

**Observação importante:** Neste caso específico, _call-by-need_ não oferece vantagem sobre_call-by-name_ porque a expressão divergente nunca é usada. A diferença apareceria se a mesma expressão fosse usada múltiplas vezes (_call-by-need_ evitaria recomputação) ou se ela fosse usada mas não divergisse.

**Parte d) Propriedade do cálculo lambda ilustrada**

Esta situação ilustra uma propriedade fundamental conhecida como a **Propriedade de Normalização da Ordem Normal** (ou Teorema de Church-Rosser aplicado à ordem de avaliação).

**Teorema (informal):** Se uma expressão lambda tem uma forma normal (um valor final ao qual pode ser reduzida), então a estratégia de redução que sempre reduz o redex mais à esquerda mais externo (similar a_call-by-name_) eventualmente encontrará essa forma normal.

**Implicações:**

1. **Call-by-name é mais poderosa em termos de terminação:** Se existe alguma sequência de reduções que leva a um resultado,_call-by-name_ (e _call-by-need_) encontrarão esse resultado.

2. **_call-by-value_ pode não terminar mesmo quando um resultado existe:** Como vimos neste exercício, _call-by-value_ pode ficar preso tentando avaliar argumentos que nunca serão usados, mesmo quando a expressão completa tem uma forma normal.

3. **Não há garantia universal de terminação:** Note que o teorema não garante que toda expressão terminará. Algumas expressões simplesmente não têm forma normal (como $infinito\ 0$ sozinho). O teorema apenas garante que se uma forma normal existe,_call-by-name_ a encontrará.

**Exemplo prático desta propriedade:**

Considere a expressão $(\lambda x. 42)\ (infinito\ 0)$:
- Com _call-by-value_: não termina (fica preso tentando avaliar o argumento)
- Com _call-by-name/need_: termina com 42 (o argumento nunca é usado)

Ambas as expressões têm o mesmo "significado" matemático (deveriam retornar 42), mas apenas_call-by-name_ realiza esse significado.

**Contexto histórico:**

Esta propriedade foi uma das motivações originais para o design de linguagens funcionais puras com avaliação preguiçosa, como Haskell. A ideia era criar uma linguagem na qual a semântica correspondesse mais diretamente à teoria matemática do cálculo lambda, na qual a propriedade de normalização da ordem normal é fundamental.

No entanto, _call-by-value_ também tem suas vantagens: é mais fácil raciocinar sobre o desempenho, mais previsível na presença de efeitos colaterais, e na prática, a maioria dos programas em linguagens _call-by-value_ termina adequadamente porque os programadores naturalmente evitam passar expressões divergentes como argumentos.

### Exercício 9: Composição de Funções e Avaliação Preguiçosa

Considere o operador de composição:

$$compose = \lambda f. (\lambda g. (\lambda x. f\ (g\ x)))$$

E as funções:

$$cara = \lambda x. x \times 2$$
$$custosa = \lambda x. fatorial\ x$$

Na qual `fatorial` é definido recursivamente. Considere:

$$h = compose\ cara\ custosa$$
$$escolha = \lambda flag. (\lambda val. \text{if}\ flag\ \text{then}\ 0\ \text{else}\ val)$$
$$resultado = escolha\ \text{true}\ (h\ 50)$$

a) Trace a redução usando _call-by-value_

b) Trace a redução usando_call-by-name_

c) Explique como a composição e a estratégia de avaliação interagem

#### Solução

**Parte a) Redução usando _call-by-value_**

Em _call-by-value_, cada argumento e cada aplicação de função deve ser completamente avaliada antes de prosseguir.

**Passo 1:** Definição de h

$$h = compose\ cara\ custosa$$

$$h = (\lambda f. (\lambda g. (\lambda x. f\ (g\ x))))\ cara\ custosa$$

Beta-redução aplicando `cara`:

$$h = (\lambda g. (\lambda x. cara\ (g\ x)))\ custosa$$

Beta-redução aplicando `custosa`:

$$h = \lambda x. cara\ (custosa\ x)$$

Expandindo as definições de `cara` e `custosa`:

$$h = \lambda x. (\lambda y. y \times 2)\ ((\lambda z. fatorial\ z)\ x)$$

Simplificando:

$$h = \lambda x. (fatorial\ x) \times 2$$

**Passo 2:** Construção da expressão completa

$$resultado = escolha\ \text{true}\ (h\ 50)$$

Expandindo `escolha`:

$$resultado = (\lambda flag. (\lambda val. \text{if}\ flag\ \text{then}\ 0\ \text{else}\ val))\ \text{true}\ (h\ 50)$$

**Passo 3:** Aplicação do primeiro argumento a `escolha`

Beta-redução, substituindo $flag$ por `true`:

$$resultado = (\lambda val. \text{if}\ \text{true}\ \text{then}\ 0\ \text{else}\ val)\ (h\ 50)$$

**Passo 4:** Avaliação do segundo argumento

Aqui está o problema crítico! _call-by-value_ exige que avaliemos $(h\ 50)$ antes de aplicá-lo à função.

$$h\ 50 = (\lambda x. (fatorial\ x) \times 2)\ 50$$

Beta-redução:

$$h\ 50 = (fatorial\ 50) \times 2$$

Agora precisamos calcular $fatorial\ 50$:

$$fatorial\ 50 = 50 \times 49 \times 48 \times ... \times 2 \times 1$$

Este é um número gigantesco! O fatorial de 50 tem 65 dígitos. Após calcular este valor enorme (vamos chamá-lo de $F_{50}$), precisamos multiplicá-lo por 2:

$$h\ 50 = F_{50} \times 2$$

Vamos chamar este resultado de $R_{50}$.

**Passo 5:** Aplicação do segundo argumento (após toda a computação)

$$resultado = (\lambda val. \text{if}\ \text{true}\ \text{then}\ 0\ \text{else}\ val)\ R_{50}$$

Beta-redução:

$$resultado = \text{if}\ \text{true}\ \text{then}\ 0\ \text{else}\ R_{50}$$

**Passo 6:** Avaliação condicional

Como a condição é verdadeira, tomamos o ramo `then`:

$$resultado = 0$$

**Resultado com CBV:** 0 (mas somente depois de calcular $fatorial\ 50$ e multiplicá-lo por 2)

**Custo:** Extremamente alto. Realizamos 50 multiplicações para calcular o fatorial, depois mais uma multiplicação por 2, apenas para descartar o resultado gigante e retornar 0.

**Parte b) Redução usando_call-by-name_**

Em_call-by-name_, os argumentos são substituídos sem avaliação prévia.

**Passo 1:** Definição de h

A composição é construída da mesma forma:

$$h = compose\ cara\ custosa$$

Em_call-by-name_, podemos deixar essa definição na forma compacta ou expandida. Vamos usar a forma expandida para clareza:

$$h = \lambda x. cara\ (custosa\ x)$$

**Passo 2:** Construção da expressão completa

$$resultado = escolha\ \text{true}\ (h\ 50)$$

$$resultado = (\lambda flag. (\lambda val. \text{if}\ flag\ \text{then}\ 0\ \text{else}\ val))\ \text{true}\ (h\ 50)$$

**Passo 3:** Primeira aplicação

Beta-redução, substituindo $flag$ por `true`:

$$resultado = (\lambda val. \text{if}\ \text{true}\ \text{then}\ 0\ \text{else}\ val)\ (h\ 50)$$

**Passo 4:** Segunda aplicação (sem avaliar h 50)

Beta-redução, substituindo $val$ por $(h\ 50)$ sem avaliar:

$$resultado = \text{if}\ \text{true}\ \text{then}\ 0\ \text{else}\ (h\ 50)$$

**Passo 5:** Avaliação condicional

Como a condição `true` é verdadeira, tomamos o ramo `then`:

$$resultado = 0$$

Observe que $(h\ 50)$ nunca foi avaliado! Toda a computação custosa (incluindo $fatorial\ 50$ e a subsequente multiplicação por 2) foi completamente evitada.

**Resultado com CBN:** 0 (calculado imediatamente, sem computar o fatorial)

**Custo:** Mínimo. Apenas avaliamos a condição `true` e retornamos 0. Nenhuma das operações da função composta foi executada.

**Parte c) Interação entre composição e estratégia de avaliação**

A interação entre composição de funções e estratégias de avaliação revela padrões importantes sobre eficiência e design de programas:

**Com _call-by-value_:**

A composição de funções é "estrita" (eager): cada etapa é completamente avaliada antes da próxima. No nosso exemplo:

1. $h$ é definido como a composição de `cara` e `custosa`
2. Quando aplicamos $h\ 50$, primeiro $custosa\ 50$ é completamente avaliado (calculando $fatorial\ 50$)
3. Depois $cara$ é aplicado ao resultado (multiplicação por 2)
4. Só então verificamos se este valor será usado

Este comportamento tem prós e contras:

**Vantagens:**

- Comportamento previsível e simples de entender
- Fácil de depurar (podemos ver o fluxo de valores)
- Adequado para funções com efeitos colaterais (efeitos ocorrem em ordem clara)

**Desvantagens:**

- Trabalho desperdiçado quando o resultado não é usado (como neste exemplo)
- Não permite trabalhar naturalmente com estruturas de dados infinitas
- Requer cuidado do programador para evitar computações custosas desnecessárias

**Com_call-by-name_ (e _call-by-need_):**

A composição de funções é "preguiçosa" (lazy): cada etapa só é avaliada se e quando seu resultado é necessário. No nosso exemplo:

1. $h$ é definido como a composição, mas nada é calculado
2. Quando aplicamos $h\ 50$, a expressão $cara\ (custosa\ 50)$ é apenas formada, não avaliada
3. Esta expressão é passada para `escolha`, ainda sem avaliação
4. `escolha` descarta o valor, então $cara$ e $custosa$ nunca são aplicados

Este comportamento também tem prós e contras:

**Vantagens:**

- Evita automaticamente trabalho desnecessário (como neste exemplo)
- Permite trabalhar naturalmente com estruturas de dados infinitas
- Código é naturalmente modular (componentes podem ser compostos livremente sem preocupação com eficiência)
- Implementa naturalmente o padrão de "curto-circuito"

**Desvantagens:**

- Comportamento menos previsível (difícil saber quando algo será avaliado)
- Pode esconder ineficiências (space leaks)
- Incompatível com efeitos colaterais (ordem dos efeitos torna-se imprevisível)
- Overhead de gerenciar thunks (promessas de computação)

**Padrões de design resultantes:**

Esta diferença leva a diferentes padrões de programação:

Em linguagens com _call-by-value_ (JavaScript, Python, Java), funções de composição são frequentemente implementadas com cuidado especial. É comum usar padrões como:

```shell
compose_lazy = λf. λg. λx. (λ_ -> f (g x))
```

na qual a função interna $(\lambda\_ \rightarrow ...)$ age como um thunk, adiando a computação.

Em linguagens com _call-by-need_ (Haskell), a composição é simples e natural:

```shell
compose = λf. λg. λx. f (g x)
```

e funciona eficientemente porque a avaliação é automaticamente adiada.

**Lição sobre modularidade:**

Este exercício demonstra uma ideia profunda sobre programação funcional: avaliação preguiçosa permite maior modularidade. Com_call-by-name_/need, podemos compor funções livremente sem nos preocupar se o resultado será usado. As preocupações sobre o que calcular e quando calcular são automaticamente separadas, permitindo que escrevamos código mais modular e composicional.

Este foi um dos insights fundamentais que motivou o desenvolvimento de linguagens como Haskell, na qual a avaliação preguiçosa por padrão permite expressar algoritmos de forma muito modular, deixando que o sistema de avaliação decida automaticamente o que realmente precisa ser computado.

### Exercício 10: Desafio - Fibonacci com Diferentes Estratégias

Considere a definição ingênua de Fibonacci:

$$Fib = \lambda f. (\lambda n. \text{if}\ (n \leq 1)\ \text{then}\ n\ \text{else}\ f\ (n-1) + f\ (n-2))$$
$$fib = Y\ Fib$$

E a função que usa o resultado duas vezes:

$$usa\_dobrado = \lambda x. x \times x$$
$$resultado = usa\_dobrado\ (fib\ 5)$$

a) Descreva quantas chamadas recursivas seriam feitas para calcular `fib 5` uma única vez

b) Com _call-by-value_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

c) Com_call-by-name_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

d) Com _call-by-need_, quantas vezes `fib 5` é calculado? Qual o total de chamadas recursivas?

e) Explique por que _call-by-need_ oferece vantagem significativa neste cenário

#### Solução

**Análise preliminar: Árvore de chamadas de Fibonacci**

Antes de abordar as estratégias de avaliação, vamos entender quantas chamadas recursivas são necessárias para calcular `fib 5` uma vez.

A função Fibonacci recursiva tem a seguinte estrutura de chamadas:

```shell
fib 5
├─ fib 4
│  ├─ fib 3
│  │  ├─ fib 2
│  │  │  ├─ fib 1 → 1
│  │  │  └─ fib 0 → 0
│  │  └─ fib 1 → 1
│  └─ fib 2
│     ├─ fib 1 → 1
│     └─ fib 0 → 0
└─ fib 3
   ├─ fib 2
   │  ├─ fib 1 → 1
   │  └─ fib 0 → 0
   └─ fib 1 → 1
```

**Contagem de chamadas:**

- `fib 5`: 1 chamada
- `fib 4`: 1 chamada
- `fib 3`: 2 chamadas
- `fib 2`: 3 chamadas
- `fib 1`: 5 chamadas
- `fib 0`: 3 chamadas

**Total:** 15 chamadas recursivas

Observe a ineficiência: `fib 3` é calculado duas vezes, `fib 2` três vezes, etc. Esta é uma característica da implementação ingênua de Fibonacci.

**Parte a) Chamadas para calcular fib 5 uma vez**

Como demonstrado acima, calcular `fib 5` uma única vez requer 15 chamadas recursivas à função Fibonacci.

**Valores calculados:**

- $fib(0) = 0$
- $fib(1) = 1$
- $fib(2) = 1$
- $fib(3) = 2$
- $fib(4) = 3$
- $fib(5) = 5$

**Parte b) _call-by-value_**

Com _call-by-value_, o argumento é completamente avaliado antes de ser passado à função.

**Passo 1:** Avaliação do argumento

$$resultado = usa\_dobrado\ (fib\ 5)$$

Precisamos avaliar $(fib\ 5)$ completamente antes de aplicar `usa_dobrado`.

Como calculado na parte (a), isso requer 15 chamadas recursivas, resultando no valor 5.

$$resultado = usa\_dobrado\ 5$$

**Passo 2:** Aplicação da função

$$resultado = (\lambda x. x \times x)\ 5$$

Beta-redução:

$$resultado = 5 \times 5 = 25$$

**Resultado com CBV:** 25

**Quantas vezes fib 5 é calculado?** Uma única vez (antes de passar à função)

**Total de chamadas recursivas:** 15 chamadas

**Parte c)_call-by-name_**

Com_call-by-name_, o argumento é substituído sem avaliação prévia, e é reavaliado cada vez que é usado.

**Passo 1:** Aplicação da função

$$resultado = usa\_dobrado\ (fib\ 5)$$

$$resultado = (\lambda x. x \times x)\ (fib\ 5)$$

Beta-redução, substituindo ambas as ocorrências de $x$ por $(fib\ 5)$:

$$resultado = (fib\ 5) \times (fib\ 5)$$

**Passo 2:** Primeira avaliação de fib 5

Precisamos avaliar a primeira ocorrência de $(fib\ 5)$:

Este cálculo requer 15 chamadas recursivas, como demonstrado anteriormente, resultando em 5.

$$resultado = 5 \times (fib\ 5)$$

**Passo 3:** Segunda avaliação de fib 5

Agora precisamos avaliar a segunda ocorrência de $(fib\ 5)$:

Este cálculo novamente requer 15 chamadas recursivas! Toda a árvore de recursão é percorrida novamente, resultando novamente em 5.

$$resultado = 5 \times 5$$

**Passo 4:** Multiplicação final

$$resultado = 25$$

**Resultado com CBN:** 25

**Quantas vezes fib 5 é calculado?** Duas vezes (uma para cada ocorrência de $x$)

**Total de chamadas recursivas:** $15 + 15 = 30$ chamadas

Este é o dobro do necessário! A recomputação completa é extremamente ineficiente.

**Parte d) _call-by-need_**

Com _call-by-need_, o argumento é substituído sem avaliação, mas o resultado da primeira avaliação é memorizado e compartilhado.

**Passo 1:** Aplicação da função com compartilhamento

$$resultado = usa\_dobrado\ (fib\ 5)$$

$$resultado = (\lambda x. x \times x)\ (fib\ 5)$$

Beta-redução, criando uma referência compartilhada:

$$\text{let}\ shared\_fib5 = (fib\ 5)\ \text{in}\ shared\_fib5 \times shared\_fib5$$

Neste ponto, $shared\_fib5$ ainda não foi avaliado.

**Passo 2:** Primeira referência - avaliação e memorização

Quando encontramos a primeira ocorrência de $shared\_fib5$, precisamos avaliá-la:

$$shared\_fib5 = fib\ 5$$

Este cálculo requer 15 chamadas recursivas, resultando em 5. Este valor é agora memorizado:

$$\text{let}\ shared\_fib5 = 5\ \text{in}\ 5 \times shared\_fib5$$

**Passo 3:** Segunda referência - uso do valor memorizado

Quando encontramos a segunda ocorrência de $shared\_fib5$, não recalculamos. Simplesmente usamos o valor memorizado:

$$\text{let}\ shared\_fib5 = 5\ \text{in}\ 5 \times 5$$

**Passo 4:** Multiplicação final

$$resultado = 25$$

**Resultado com CBNeed:** 25

**Quantas vezes fib 5 é calculado?** Uma única vez (com o resultado compartilhado)

**Total de chamadas recursivas:** 15 chamadas (mesmo que _call-by-value_)

**Parte e) Vantagem de _call-by-need_**

_call-by-need_ oferece vantagens significativas neste cenário por combinar o melhor de ambos os mundos:

**Comparação de desempenho:**

| Estratégia    | Cálculos de fib 5 | Chamadas recursivas | Eficiência |
|---------------|-------------------|---------------------|------------|
| _call-by-value_ | 1                 | 15                  | Ótima      |
|_call-by-name_  | 2                 | 30                  | Ruim       |
| _call-by-need_  | 1                 | 15                  | Ótima      |

**Vantagens específicas de _call-by-need_ neste cenário:**

1. **Evita recomputação:** Diferentemente de_call-by-name_, _call-by-need_ memoriza o resultado da primeira avaliação. Quando `usa_dobrado` usa o argumento duas vezes, a segunda referência é gratuita.

2. **Desempenho equivalente a _call-by-value_ para argumentos usados:** Quando um argumento é usado, _call-by-need_ tem o mesmo desempenho que _call-by-value_ (uma avaliação), mas com a vantagem adicional de que se o argumento não fosse usado, não seria calculado.

3. **Escala bem com múltiplas referências:** Se a função usasse o argumento 10 vezes ($x^{10}$),_call-by-name_ faria 150 chamadas recursivas, enquanto _call-by-need_ ainda faria apenas 15.

4. **Transparência referencial preservada:** Diferentemente de técnicas de memoização manual, _call-by-need_ preserva a semântica do cálculo lambda puro, garantindo que programas se comportem corretamente mesmo na presença de compartilhamento.

**Considerações sobre Fibonacci especificamente:**

Vale notar que este exemplo demonstra a ineficiência em dois níveis:

1. **Nível do argumento:**_call-by-name_ recalcula `fib 5` completamente duas vezes (30 chamadas totais)
2. **Nível interno:** A implementação ingênua de Fibonacci já é ineficiente dentro de cada cálculo de `fib 5` (15 chamadas que poderiam ser 6 com memoização adequada)

_call-by-need_ resolve o primeiro problema (compartilhamento entre usos do argumento), mas não resolve automaticamente o segundo (recomputação dentro da própria função recursiva). Para otimizar completamente Fibonacci, precisaríamos de programação dinâmica ou memoização explícita.

No entanto, em Haskell (que usa _call-by-need_), é possível escrever Fibonacci de forma que a preguiça natural da linguagem também otimize as subchamadas, criando uma implementação eficiente sem memoização explícita:

```haskell
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
```

Esta versão calcula cada número de Fibonacci exatamente uma vez, aproveitando a avaliação preguiçosa para criar uma lista infinita memoizada.

Este exercício demonstra que _call-by-need_ oferece garantias de eficiência importantes: cada expressão única é avaliada no máximo uma vez, independentemente de quantas vezes é referenciada. Esta propriedade permite ao programador escrever código naturalmente modular e composicional sem se preocupar com recomputação acidental, um dos principais benefícios da avaliação preguiçosa em linguagens como Haskell.

