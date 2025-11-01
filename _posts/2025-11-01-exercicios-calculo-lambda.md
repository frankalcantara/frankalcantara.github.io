---
layout: post
title: "Praticando Cálculo Lambda: Exercícios e Soluções"
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
description: Exercícios para prática de recursão, redução beta e currying em cálculo lambda.
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
lastmod: 2025-11-01T15:19:09.346Z
draft: 2025-11-01T15:01:40.919Z
---

## Antes de Começar

Para facilitar a compreensão dos conceitos principais, estes exercícios utilizam uma notação que permite operações aritméticas básicas diretamente no cálculo lambda, como adição, subtração, multiplicação, exponenciação e comparações. Em um cálculo lambda puramente teórico, estas operações seriam implementadas usando codificação de Church, mas para os propósitos pedagógicos desta aula, assumiremos que operações como $x + y$, $x \times y$, $x - y$, $x^y$ e comparações como $(n = 0)$ estão disponíveis como primitivas.

Esta abordagem permite focar nos conceitos de currying, beta-redução e recursão com o combinador Y sem a complexidade adicional da codificação numérica.

Estes dez exercícios cobrem os conceitos fundamentais do cálculo lambda estudados até o momento:

1. **Currying:** Transformação de funções de múltiplos argumentos em cadeias de funções unárias (Exercícios 1, 3, 5, 7, 10)

2. **Beta-redução:** Processo de aplicação de funções através da substituição de variáveis (todos os exercícios)

3. **Combinador Y:** Implementação de recursão sem nomes através de auto-aplicação (Exercícios 4, 6, 8, 9, 10)

4. **Aplicação Parcial:** Fixação de alguns argumentos para criar funções especializadas (Exercícios 3, 5, 7, 10)

5. **Composição:** Combinação de funções simples para criar comportamentos complexos (Exercício 9)

## PARTE 1: ENUNCIADOS

Comece por aqui, tente resolver os exercícios antes de conferir as soluções na Parte 2.

### Exercício 1: Currying Básico

Dada a função não-curried que calcula a área de um retângulo:

$$area = \lambda(b, h). b \times h$$

Transforme esta função para sua forma curried e demonstre sua aplicação com $b = 5$ e $h = 8$.

### Exercício 2: Beta-Redução com Funções Curried

Considere a função curried:

$$f = \lambda x. (\lambda y. (\lambda z. x + y \times z))$$

Realize a beta-redução completa da expressão $f\ 3\ 4\ 5$ mostrando cada passo intermediário.

### Exercício 3: Aplicação Parcial e Composição

Defina em cálculo lambda:

a) Uma função `subtrair` curried que receba dois números e retorne a diferença do primeiro pelo segundo

b) Uma função `decrementar` obtida por aplicação parcial de `subtrair`

c) Demonstre a redução de `decrementar 10`

### Exercício 4: Combinador Y - Fatorial

Utilizando o combinador Y:

$$Y = \lambda f. (\lambda x. f\ (x\ x))\ (\lambda x. f\ (x\ x))$$

Defina a função fatorial recursiva e demonstre a redução completa de $fatorial\ 3$ até obter o resultado numérico. Mostre pelo menos os três primeiros passos da expansão recursiva.

### Exercício 5: Currying com Três Argumentos

Considere uma função que calcula o valor final de um investimento com juros compostos:

$$montante = \lambda(c, i, t). c \times (1 + i)^t$$

onde $c$ é o capital inicial, $i$ é a taxa de juros e $t$ é o tempo.

a) Transforme esta função para sua forma curried

b) Crie uma função `investimento_padrao` que fixe a taxa de juros em 0.05 (5%)

c) Crie uma função `investimento_anual` que fixe tanto a taxa (5%) quanto o tempo (1 ano)

d) Demonstre a aplicação de `investimento_anual` com capital inicial de 1000

### Exercício 6: Recursão - Soma dos Primeiros N Números

Utilizando o combinador Y, defina uma função recursiva `soma_n` que calcule a soma dos primeiros $n$ números naturais (isto é, $1 + 2 + 3 + ... + n$).

Demonstre a redução de `soma_n 4` mostrando:

- A definição da função com o combinador Y
- Pelo menos os três primeiros passos da recursão
- O resultado final

### Exercício 7: Currying e Beta-Redução Complexa

Dada a função curried que representa uma operação matemática complexa:

$$g = \lambda x. (\lambda y. (\lambda z. (x + y) \times (y + z)))$$

a) Crie a função `g_parcial` fixando $x = 2$

b) Crie a função `g_mais_parcial` fixando $x = 2$ e $y = 3$

c) Demonstre a beta-redução completa de $g_mais_parcial\ 4$

d) Compare o resultado obtendo o mesmo valor através da aplicação direta $g\ 2\ 3\ 4$

### Exercício 8: Combinador Y - Potência

Defina usando o combinador Y uma função `potencia` que calcule $base^{exp}$, onde ambos são números naturais.

A função deve ser curried, recebendo primeiro a base e depois o expoente.

Demonstre a redução de `potencia 2 3` (que deve resultar em 8), mostrando os principais passos da recursão.

**Dica:** Lembre-se que $b^0 = 1$ e $b^n = b \times b^{n-1}$ para $n > 0$.

### Exercício 9: Composição de Funções Recursivas

Considere as seguintes definições:

$$dobro = \lambda x. x \times 2$$

$$soma\_recursiva = Y\ (\lambda f. \lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + f\ (n - 1))$$

Defina uma função `soma_dobros` que calcule a soma dos dobros dos primeiros $n$ números naturais. Em outras palavras, calcule $2 \times 1 + 2 \times 2 + 2 \times 3 + ... + 2 \times n$.

Demonstre a redução de `soma_dobros 3` (que deve resultar em 12).

### Exercício 10: Desafio - Fibonacci com Currying

Defina uma função `fibonacci_modificado` usando o combinador Y que:

a) Seja curried, recebendo primeiro um multiplicador $m$ e depois o índice $n$

b) Retorne $m \times fib(n)$, onde $fib(n)$ é o n-ésimo número de Fibonacci

Lembre-se que $fib(0) = 0$, $fib(1) = 1$ e $fib(n) = fib(n-1) + fib(n-2)$ para $n > 1$.

Demonstre:

- A definição completa da função
- A criação de uma função especializada `triplo_fib` que multiplica o resultado por 3
- A redução de `triplo_fib 4` (sabendo que $fib(4) = 3$, o resultado deve ser 9)

## PARTE 2: ENUNCIADOS COM SOLUÇÕES

### Exercício 1: Currying Básico

Dada a função não-curried que calcula a área de um retângulo:

$$area = \lambda(b, h). b \times h$$

Transforme esta função para sua forma curried e demonstre sua aplicação com $b = 5$ e $h = 8$.

#### Solução

**Passo 1: Transformação para forma curried**

A transformação consiste em substituir a função que recebe um par de argumentos por funções aninhadas que recebem um argumento por vez:

$$area = \lambda b. (\lambda h. b \times h)$$

Esta notação indica que `area` é uma função que recebe o primeiro argumento $b$ e retorna outra função que aguarda o segundo argumento $h$. Quando ambos os argumentos são fornecidos, a multiplicação é realizada.

**Passo 2: Aplicação do primeiro argumento ($b = 5$)**

$$area\ 5 = (\lambda b. (\lambda h. b \times h))\ 5$$

Aplicando beta-redução (substituímos todas as ocorrências de $b$ por 5):

$$area\ 5 = \lambda h. 5 \times h$$

Neste ponto, obtemos uma função parcialmente aplicada que aguarda apenas a altura.

**Passo 3: Aplicação do segundo argumento ($h = 8$)**

$$area\ 5\ 8 = (\lambda h. 5 \times h)\ 8$$

Aplicando beta-redução novamente (substituímos $h$ por 8):

$$area\ 5\ 8 = 5 \times 8 = 40$$

**Resultado final:** 40

**Observação importante:** A forma curried permite criar funções especializadas. Por exemplo, `area 5` é uma função que calcula a área de retângulos com base 5, aguardando apenas a altura.

### Exercício 2: Beta-Redução com Funções Curried

Considere a função curried:

$$f = \lambda x. (\lambda y. (\lambda z. x + y \times z))$$

Realize a beta-redução completa da expressão $f\ 3\ 4\ 5$ mostrando cada passo intermediário.

#### Solução

**Passo 1: Aplicação do primeiro argumento ($x = 3$)**

$$f\ 3 = (\lambda x. (\lambda y. (\lambda z. x + y \times z)))\ 3$$

Realizando beta-redução, substituímos todas as ocorrências livres de $x$ por 3:

$$f\ 3 = \lambda y. (\lambda z. 3 + y \times z)$$

Após esta redução, obtemos uma função que ainda aguarda dois argumentos: $y$ e $z$.

**Passo 2: Aplicação do segundo argumento ($y = 4$)**

$$f\ 3\ 4 = (\lambda y. (\lambda z. 3 + y \times z))\ 4$$

Realizando beta-redução, substituímos todas as ocorrências livres de $y$ por 4:

$$f\ 3\ 4 = \lambda z. 3 + 4 \times z$$

Agora temos uma função que aguarda apenas o último argumento $z$.

**Passo 3: Aplicação do terceiro argumento ($z = 5$)**

$$f\ 3\ 4\ 5 = (\lambda z. 3 + 4 \times z)\ 5$$

Realizando a última beta-redução, substituímos $z$ por 5:

$$f\ 3\ 4\ 5 = 3 + 4 \times 5$$

**Passo 4: Avaliação aritmética**

Seguindo a precedência de operadores (multiplicação antes de adição):

$$f\ 3\ 4\ 5 = 3 + 20 = 23$$

**Resultado final:** 23

**Observação conceitual:** Este exercício demonstra como a beta-redução funciona em cadeia para funções curried. Cada aplicação de argumento remove uma camada de abstração lambda, até que todos os argumentos sejam consumidos e obtenhamos o valor final.

### Exercício 3: Aplicação Parcial e Composição

Defina em cálculo lambda:

a) Uma função `subtrair` curried que receba dois números e retorne a diferença do primeiro pelo segundo

b) Uma função `decrementar` que subtraia 1 de qualquer número fornecido

c) Demonstre a redução de `decrementar 10`

#### Solução

**Parte a) Definição de `subtrair`**

A função `subtrair` em forma curried recebe o primeiro argumento (minuendo) e retorna uma função que aguarda o segundo argumento (subtraendo):

$$subtrair = \lambda x. (\lambda y. x - y)$$

Esta definição estabelece a ordem: o primeiro argumento será o valor do qual subtraímos, e o segundo argumento é o valor que será subtraído.

**Parte b) Definição de `decrementar`**

Para criar uma função `decrementar` que subtraia 1 de qualquer número, precisamos fornecer ambos os argumentos a `subtrair`. Queremos uma função onde o número variável seja o minuendo e 1 seja o subtraendo fixo:

$$decrementar = \lambda n. subtrair\ n\ 1$$

Expandindo a definição de `subtrair`:

$$decrementar = \lambda n. ((\lambda x. (\lambda y. x - y))\ n)\ 1$$

Note que não podemos usar simplesmente aplicação parcial aqui (como `subtrair 1`) porque isso fixaria o minuendo em 1, resultando em uma função que calcula $1 - y$, o que não é o que desejamos. Por isso, criamos uma nova abstração lambda que aplica os argumentos na ordem correta.

**Parte c) Demonstração da redução de `decrementar 10`**

Vamos reduzir passo a passo:

$$decrementar\ 10 = (\lambda n. ((\lambda x. (\lambda y. x - y))\ n)\ 1)\ 10$$

**Passo 1:** Beta-redução da abstração externa, substituindo $n$ por 10:

$$decrementar\ 10 = ((\lambda x. (\lambda y. x - y))\ 10)\ 1$$

**Passo 2:** Beta-redução da primeira aplicação, substituindo $x$ por 10:

$$decrementar\ 10 = (\lambda y. 10 - y)\ 1$$

**Passo 3:** Beta-redução final, substituindo $y$ por 1:

$$decrementar\ 10 = 10 - 1 = 9$$

**Resultado final:** 9

**Observação sobre ordem de argumentos:** Este exercício ilustra um princípio importante no design de funções curried. A ordem dos argumentos afeta diretamente a facilidade de criar funções especializadas por aplicação parcial. Se tivéssemos definido `subtrair` como $\lambda y. (\lambda x. x - y)$ (recebendo primeiro o subtraendo), então `subtrair 1` seria automaticamente a função decrementar. Em linguagens funcionais, é comum ordenar os argumentos colocando os valores mais propensos a variação por último, facilitando a criação de funções especializadas.

### Exercício 4: Combinador Y - Fatorial

Utilizando o combinador Y:

$$Y = \lambda f. (\lambda x. f\ (x\ x))\ (\lambda x. f\ (x\ x))$$

Defina a função fatorial recursiva e demonstre a redução completa de $fatorial\ 3$ até obter o resultado numérico. Mostre pelo menos os três primeiros passos da expansão recursiva.

#### Solução

**Passo 1: Definição da função fatorial usando o combinador Y**

Primeiro, definimos o "corpo" da função fatorial. Este corpo recebe a própria função como argumento (permitindo a recursão) e o número $n$:

$$F = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times f\ (n - 1))$$

Agora aplicamos o combinador Y para obter a função fatorial recursiva:

$$fatorial = Y\ F$$

Expandindo:

$$fatorial = Y\ (\lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times f\ (n - 1)))$$

**Passo 2: Cálculo de $fatorial\ 3$**

Vamos começar a redução:

$$fatorial\ 3 = (Y\ F)\ 3$$

**Primeira expansão do combinador Y:**

Pela definição do combinador Y, temos:

$$Y\ F = F\ (Y\ F)$$

Portanto:

$$fatorial\ 3 = F\ (Y\ F)\ 3$$

Substituindo a definição de $F$:

$$fatorial\ 3 = (\lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times f\ (n - 1)))\ (Y\ F)\ 3$$

**Beta-redução (substituindo $f$ por $Y\ F$):**

$$fatorial\ 3 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times (Y\ F)\ (n - 1))\ 3$$

**Beta-redução (substituindo $n$ por 3):**

$$fatorial\ 3 = \text{if}\ (3 = 0)\ \text{then}\ 1\ \text{else}\ 3 \times (Y\ F)\ (3 - 1)$$

Como $3 \neq 0$, tomamos o ramo `else`:

$$fatorial\ 3 = 3 \times (Y\ F)\ 2$$

Ou seja:

$$fatorial\ 3 = 3 \times fatorial\ 2$$

**Passo 3: Cálculo de $fatorial\ 2$ (segunda recursão)**

Aplicamos o mesmo processo:

$$fatorial\ 2 = F\ (Y\ F)\ 2$$

$$fatorial\ 2 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times (Y\ F)\ (n - 1))\ 2$$

$$fatorial\ 2 = \text{if}\ (2 = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ F)\ 1$$

Como $2 \neq 0$:

$$fatorial\ 2 = 2 \times (Y\ F)\ 1 = 2 \times fatorial\ 1$$

Substituindo na expressão anterior:

$$fatorial\ 3 = 3 \times (2 \times fatorial\ 1)$$

**Passo 4: Cálculo de $fatorial\ 1$ (terceira recursão)**

$$fatorial\ 1 = F\ (Y\ F)\ 1$$

$$fatorial\ 1 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times (Y\ F)\ (n - 1))\ 1$$

$$fatorial\ 1 = \text{if}\ (1 = 0)\ \text{then}\ 1\ \text{else}\ 1 \times (Y\ F)\ 0$$

Como $1 \neq 0$:

$$fatorial\ 1 = 1 \times (Y\ F)\ 0 = 1 \times fatorial\ 0$$

Substituindo:

$$fatorial\ 3 = 3 \times (2 \times (1 \times fatorial\ 0))$$

**Passo 5: Caso base - $fatorial\ 0$**

$$fatorial\ 0 = F\ (Y\ F)\ 0$$

$$fatorial\ 0 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 1\ \text{else}\ n \times (Y\ F)\ (n - 1))\ 0$$

$$fatorial\ 0 = \text{if}\ (0 = 0)\ \text{then}\ 1\ \text{else}\ 0 \times (Y\ F)\ (-1)$$

Como $0 = 0$, tomamos o ramo `then`:

$$fatorial\ 0 = 1$$

**Passo 6: Avaliação final**

Substituindo o valor do caso base:

$$fatorial\ 3 = 3 \times (2 \times (1 \times 1))$$

$$fatorial\ 3 = 3 \times (2 \times 1)$$

$$fatorial\ 3 = 3 \times 2$$

$$fatorial\ 3 = 6$$

**Resultado final:** 6

**Observação conceitual:** O combinador Y permite expressar recursão sem referências diretas ao nome da função. A "mágica" do combinador Y está na auto-aplicação $(\lambda x. f\ (x\ x))\ (\lambda x. f\ (x\ x))$, que permite que a função se replique infinitamente conforme necessário, mas de forma controlada pelo caso base.

### Exercício 5: Currying com Três Argumentos

Considere uma função que calcula o valor final de um investimento com juros compostos:

$$montante = \lambda(c, i, t). c \times (1 + i)^t$$

onde $c$ é o capital inicial, $i$ é a taxa de juros e $t$ é o tempo.

a) Transforme esta função para sua forma curried

b) Crie uma função `investimento_padrao` que fixe a taxa de juros em 0.05 (5%)

c) Crie uma função `investimento_anual` que fixe tanto a taxa (5%) quanto o tempo (1 ano)

d) Demonstre a aplicação de `investimento_anual` com capital inicial de 1000

#### Solução

**Parte a) Forma curried**

Transformamos a função para receber um argumento por vez através de abstrações lambda aninhadas:

$$montante = \lambda c. (\lambda i. (\lambda t. c \times (1 + i)^t))$$

Esta forma permite aplicação parcial em qualquer estágio, criando funções especializadas.

**Parte b) Função `investimento_padrao`**

Para criar uma função que fixa a taxa de juros em 5%, precisamos decidir a ordem de aplicação dos argumentos. Observando a definição curried, temos três possibilidades de ordenamento. Para maior flexibilidade, podemos querer que o capital venha por último (já que varia mais frequentemente).

Vamos redefinir a ordem dos argumentos para facilitar:

$$montante' = \lambda i. (\lambda t. (\lambda c. c \times (1 + i)^t))$$

Agora criamos `investimento_padrao` fixando $i = 0.05$:

$$investimento\_padrao = montante'\ 0.05$$

Expandindo e reduzindo:

$$investimento\_padrao = (\lambda i. (\lambda t. (\lambda c. c \times (1 + i)^t)))\ 0.05$$

Beta-redução (substituímos $i$ por 0.05):

$$investimento\_padrao = \lambda t. (\lambda c. c \times (1 + 0.05)^t)$$

Simplificando:

$$investimento\_padrao = \lambda t. (\lambda c. c \times 1.05^t)$$

Esta função agora aguarda o tempo e depois o capital.

**Parte c) Função `investimento_anual`**

Partindo de `investimento_padrao`, fixamos $t = 1$:

$$investimento\_anual = investimento\_padrao\ 1$$

$$investimento\_anual = (\lambda t. (\lambda c. c \times 1.05^t))\ 1$$

Beta-redução (substituímos $t$ por 1):

$$investimento\_anual = \lambda c. c \times 1.05^1$$

$$investimento\_anual = \lambda c. c \times 1.05$$

Esta função especializada calcula o montante após um ano com taxa de 5%.

**Parte d) Aplicação com capital inicial de 1000**

$$investimento\_anual\ 1000 = (\lambda c. c \times 1.05)\ 1000$$

Beta-redução (substituímos $c$ por 1000):

$$investimento\_anual\ 1000 = 1000 \times 1.05$$

$$investimento\_anual\ 1000 = 1050$$

**Resultado final:** 1050

**Observação prática:** Este exercício demonstra como currying permite criar uma hierarquia de funções especializadas. Começamos com uma função genérica de três argumentos e criamos versões progressivamente mais específicas:

1. `montante'` - função completamente genérica
2. `investimento_padrao` - fixa a taxa de juros (5%)
3. `investimento_anual` - fixa a taxa (5%) e o período (1 ano)

Cada nível de especialização reduz a flexibilidade, mas aumenta a conveniência para casos de uso específicos. Esta é uma técnica poderosa em programação funcional para criar APIs expressivas e reutilizáveis.

### Exercício 6: Recursão - Soma dos Primeiros N Números

Utilizando o combinador Y, defina uma função recursiva `soma_n` que calcule a soma dos primeiros $n$ números naturais (isto é, $1 + 2 + 3 + ... + n$).

Demonstre a redução de `soma_n 4` mostrando:
- A definição da função com o combinador Y
- Pelo menos os três primeiros passos da recursão
- O resultado final

#### Solução

**Passo 1: Definição da função com o combinador Y**

Primeiro, definimos o corpo da função. A lógica é: se $n = 0$, retornamos 0 (caso base); caso contrário, retornamos $n + soma(n-1)$ (caso recursivo):

$$S = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + f\ (n - 1))$$

Aplicamos o combinador Y para obter a função recursiva:

$$soma\_n = Y\ S$$

Lembrando que:

$$Y = \lambda f. (\lambda x. f\ (x\ x))\ (\lambda x. f\ (x\ x))$$

**Passo 2: Cálculo de $soma\_n\ 4$ - Primeira recursão**

$$soma\_n\ 4 = (Y\ S)\ 4$$

Pela propriedade do combinador Y, sabemos que $Y\ S = S\ (Y\ S)$:

$$soma\_n\ 4 = S\ (Y\ S)\ 4$$

Substituindo a definição de $S$:

$$soma\_n\ 4 = (\lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + f\ (n - 1)))\ (Y\ S)\ 4$$

Beta-redução (substituímos $f$ por $Y\ S$):

$$soma\_n\ 4 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + (Y\ S)\ (n - 1))\ 4$$

Beta-redução (substituímos $n$ por 4):

$$soma\_n\ 4 = \text{if}\ (4 = 0)\ \text{then}\ 0\ \text{else}\ 4 + (Y\ S)\ (4 - 1)$$

Como $4 \neq 0$, tomamos o ramo `else`:

$$soma\_n\ 4 = 4 + (Y\ S)\ 3$$

$$soma\_n\ 4 = 4 + soma\_n\ 3$$

**Passo 3: Cálculo de $soma\_n\ 3$ - Segunda recursão**

$$soma\_n\ 3 = S\ (Y\ S)\ 3$$

$$soma\_n\ 3 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + (Y\ S)\ (n - 1))\ 3$$

$$soma\_n\ 3 = \text{if}\ (3 = 0)\ \text{then}\ 0\ \text{else}\ 3 + (Y\ S)\ 2$$

Como $3 \neq 0$:

$$soma\_n\ 3 = 3 + (Y\ S)\ 2 = 3 + soma\_n\ 2$$

Substituindo na expressão anterior:

$$soma\_n\ 4 = 4 + (3 + soma\_n\ 2)$$

**Passo 4: Cálculo de $soma\_n\ 2$ - Terceira recursão**

$$soma\_n\ 2 = S\ (Y\ S)\ 2$$

$$soma\_n\ 2 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + (Y\ S)\ (n - 1))\ 2$$

$$soma\_n\ 2 = \text{if}\ (2 = 0)\ \text{then}\ 0\ \text{else}\ 2 + (Y\ S)\ 1$$

Como $2 \neq 0$:

$$soma\_n\ 2 = 2 + (Y\ S)\ 1 = 2 + soma\_n\ 1$$

Substituindo:

$$soma\_n\ 4 = 4 + (3 + (2 + soma\_n\ 1))$$

**Passo 5: Cálculo de $soma\_n\ 1$ - Quarta recursão**

$$soma\_n\ 1 = S\ (Y\ S)\ 1$$

$$soma\_n\ 1 = \text{if}\ (1 = 0)\ \text{then}\ 0\ \text{else}\ 1 + (Y\ S)\ 0$$

Como $1 \neq 0$:

$$soma\_n\ 1 = 1 + soma\_n\ 0$$

Substituindo:

$$soma\_n\ 4 = 4 + (3 + (2 + (1 + soma\_n\ 0)))$$

**Passo 6: Caso base - $soma\_n\ 0$**

$$soma\_n\ 0 = S\ (Y\ S)\ 0$$

$$soma\_n\ 0 = \text{if}\ (0 = 0)\ \text{then}\ 0\ \text{else}\ 0 + (Y\ S)\ (-1)$$

Como $0 = 0$, tomamos o ramo `then`:

$$soma\_n\ 0 = 0$$

**Passo 7: Avaliação final**

Substituindo o caso base e avaliando as operações aritméticas:

$$soma\_n\ 4 = 4 + (3 + (2 + (1 + 0)))$$

$$soma\_n\ 4 = 4 + (3 + (2 + 1))$$

$$soma\_n\ 4 = 4 + (3 + 3)$$

$$soma\_n\ 4 = 4 + 6$$

$$soma\_n\ 4 = 10$$

**Resultado final:** 10

**Verificação:** $1 + 2 + 3 + 4 = 10$ ✓

**Observação matemática:** Este exercício implementa a fórmula clássica da soma dos primeiros $n$ naturais: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$. Para $n=4$: $\frac{4 \times 5}{2} = 10$, confirmando nosso resultado.

### Exercício 7: Currying e Beta-Redução Complexa

Dada a função curried que representa uma operação matemática complexa:

$$g = \lambda x. (\lambda y. (\lambda z. (x + y) \times (y + z)))$$

a) Crie a função `g_parcial` fixando $x = 2$

b) Crie a função `g_mais_parcial` fixando $x = 2$ e $y = 3$

c) Demonstre a beta-redução completa de $g\_mais\_parcial\ 4$

d) Compare o resultado obtendo o mesmo valor através da aplicação direta $g\ 2\ 3\ 4$

#### Solução

**Parte a) Criação de `g_parcial` com $x = 2$**

$$g\_parcial = g\ 2$$

$$g\_parcial = (\lambda x. (\lambda y. (\lambda z. (x + y) \times (y + z))))\ 2$$

Beta-redução (substituímos $x$ por 2):

$$g\_parcial = \lambda y. (\lambda z. (2 + y) \times (y + z))$$

Esta função agora aguarda dois argumentos: $y$ e $z$. O primeiro argumento foi fixado em 2.

**Parte b) Criação de `g_mais_parcial` com $x = 2$ e $y = 3$**

$$g\_mais\_parcial = g\_parcial\ 3$$

$$g\_mais\_parcial = (\lambda y. (\lambda z. (2 + y) \times (y + z)))\ 3$$

Beta-redução (substituímos $y$ por 3):

$$g\_mais\_parcial = \lambda z. (2 + 3) \times (3 + z)$$

Simplificando a primeira parte da expressão:

$$g\_mais\_parcial = \lambda z. 5 \times (3 + z)$$

Esta função aguarda apenas um argumento: $z$.

**Parte c) Beta-redução completa de $g\_mais\_parcial\ 4$**

$$g\_mais\_parcial\ 4 = (\lambda z. 5 \times (3 + z))\ 4$$

Beta-redução (substituímos $z$ por 4):

$$g\_mais\_parcial\ 4 = 5 \times (3 + 4)$$

Avaliando a expressão aritmética (primeiro os parênteses):

$$g\_mais\_parcial\ 4 = 5 \times 7$$

$$g\_mais\_parcial\ 4 = 35$$

**Parte d) Aplicação direta $g\ 2\ 3\ 4$**

Vamos reduzir a aplicação completa desde o início para comparar:

$$g\ 2\ 3\ 4 = (\lambda x. (\lambda y. (\lambda z. (x + y) \times (y + z))))\ 2\ 3\ 4$$

**Primeira beta-redução** (aplicamos $x = 2$):

$$g\ 2\ 3\ 4 = (\lambda y. (\lambda z. (2 + y) \times (y + z)))\ 3\ 4$$

**Segunda beta-redução** (aplicamos $y = 3$):

$$g\ 2\ 3\ 4 = (\lambda z. (2 + 3) \times (3 + z))\ 4$$

Simplificando:

$$g\ 2\ 3\ 4 = (\lambda z. 5 \times (3 + z))\ 4$$

**Terceira beta-redução** (aplicamos $z = 4$):

$$g\ 2\ 3\ 4 = 5 \times (3 + 4)$$

$$g\ 2\ 3\ 4 = 5 \times 7 = 35$$

**Comparação dos resultados:**

- Usando aplicação parcial sucessiva: $g\_mais\_parcial\ 4 = 35$
- Usando aplicação direta: $g\ 2\ 3\ 4 = 35$

Os resultados são idênticos, confirmando que a aplicação parcial é equivalente à aplicação completa de todos os argumentos de uma só vez.

**Observação conceitual importante:** Este exercício demonstra um princípio fundamental de currying: a ordem de aplicação dos argumentos não afeta o resultado final. Podemos aplicar todos os argumentos de uma vez ou aplicá-los progressivamente através de funções intermediárias (aplicação parcial). Esta propriedade é garantida pela confluência do cálculo lambda, que assegura que diferentes sequências de reduções levam ao mesmo resultado normal.

### Exercício 8: Combinador Y - Potência

Defina usando o combinador Y uma função `potencia` que calcule $base^{exp}$, onde ambos são números naturais.

A função deve ser curried, recebendo primeiro a base e depois o expoente.

Demonstre a redução de `potencia 2 3` (que deve resultar em 8), mostrando os principais passos da recursão.

**Dica:** Lembre-se que $b^0 = 1$ e $b^n = b \times b^{n-1}$ para $n > 0$.

#### Solução

**Passo 1: Definição da função com o combinador Y**

Precisamos criar uma função recursiva curried. A estrutura será:
- Se o expoente é 0, retornamos 1 (caso base)
- Caso contrário, retornamos $base \times potencia(base, exp-1)$ (caso recursivo)

Como queremos que a função seja curried, precisamos estruturá-la para receber a base primeiro, depois o expoente. Vamos definir o corpo da função:

$$P = \lambda f. (\lambda b. (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ b \times f\ b\ (e - 1)))$$

Aqui, $f$ é a função recursiva, $b$ é a base, e $e$ é o expoente.

Aplicamos o combinador Y:

$$potencia = Y\ P$$

**Passo 2: Cálculo de $potencia\ 2\ 3$ - Primeira aplicação**

$$potencia\ 2\ 3 = (Y\ P)\ 2\ 3$$

Pela propriedade do combinador Y: $Y\ P = P\ (Y\ P)$

$$potencia\ 2\ 3 = P\ (Y\ P)\ 2\ 3$$

Substituindo a definição de $P$:

$$potencia\ 2\ 3 = (\lambda f. (\lambda b. (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ b \times f\ b\ (e - 1))))\ (Y\ P)\ 2\ 3$$

Beta-redução (substituímos $f$ por $Y\ P$):

$$potencia\ 2\ 3 = (\lambda b. (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ b \times (Y\ P)\ b\ (e - 1)))\ 2\ 3$$

Beta-redução (substituímos $b$ por 2):

$$potencia\ 2\ 3 = (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (e - 1))\ 3$$

Beta-redução (substituímos $e$ por 3):

$$potencia\ 2\ 3 = \text{if}\ (3 = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (3 - 1)$$

Como $3 \neq 0$, tomamos o ramo `else`:

$$potencia\ 2\ 3 = 2 \times (Y\ P)\ 2\ 2$$

$$potencia\ 2\ 3 = 2 \times potencia\ 2\ 2$$

**Passo 3: Cálculo de $potencia\ 2\ 2$ - Segunda recursão**

$$potencia\ 2\ 2 = P\ (Y\ P)\ 2\ 2$$

Seguindo o mesmo processo:

$$potencia\ 2\ 2 = (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (e - 1))\ 2$$

$$potencia\ 2\ 2 = \text{if}\ (2 = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ 1$$

Como $2 \neq 0$:

$$potencia\ 2\ 2 = 2 \times (Y\ P)\ 2\ 1 = 2 \times potencia\ 2\ 1$$

Substituindo na expressão anterior:

$$potencia\ 2\ 3 = 2 \times (2 \times potencia\ 2\ 1)$$

**Passo 4: Cálculo de $potencia\ 2\ 1$ - Terceira recursão**

$$potencia\ 2\ 1 = P\ (Y\ P)\ 2\ 1$$

$$potencia\ 2\ 1 = (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (e - 1))\ 1$$

$$potencia\ 2\ 1 = \text{if}\ (1 = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ 0$$

Como $1 \neq 0$:

$$potencia\ 2\ 1 = 2 \times (Y\ P)\ 2\ 0 = 2 \times potencia\ 2\ 0$$

Substituindo:

$$potencia\ 2\ 3 = 2 \times (2 \times (2 \times potencia\ 2\ 0))$$

**Passo 5: Caso base - $potencia\ 2\ 0$**

$$potencia\ 2\ 0 = P\ (Y\ P)\ 2\ 0$$

$$potencia\ 2\ 0 = (\lambda e. \text{if}\ (e = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (e - 1))\ 0$$

$$potencia\ 2\ 0 = \text{if}\ (0 = 0)\ \text{then}\ 1\ \text{else}\ 2 \times (Y\ P)\ 2\ (-1)$$

Como $0 = 0$, tomamos o ramo `then`:

$$potencia\ 2\ 0 = 1$$

**Passo 6: Avaliação final**

Substituindo o caso base e realizando as multiplicações:

$$potencia\ 2\ 3 = 2 \times (2 \times (2 \times 1))$$

$$potencia\ 2\ 3 = 2 \times (2 \times 2)$$

$$potencia\ 2\ 3 = 2 \times 4$$

$$potencia\ 2\ 3 = 8$$

**Resultado final:** 8

**Verificação:** $2^3 = 2 \times 2 \times 2 = 8$ ✓

**Observação sobre currying:** Note que a função `potencia` é curried, o que permite criar funções especializadas facilmente. Por exemplo:

$$potencia\_de\_2 = potencia\ 2$$

Esta função especializada calcula potências de 2. Então `potencia_de_2 3` seria equivalente a $2^3$.

**Observação sobre a recursão:** A recursão ocorre no expoente, não na base. A base permanece constante durante todas as chamadas recursivas, enquanto o expoente é decrementado até atingir zero. Este padrão é típico em implementações recursivas de exponenciação.

### Exercício 9: Composição de Funções Recursivas

Considere as seguintes definições:

$$dobro = \lambda x. x \times 2$$

$$soma\_recursiva = Y\ (\lambda f. \lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ n + f\ (n - 1))$$

Defina uma função `soma_dobros` que calcule a soma dos dobros dos primeiros $n$ números naturais. Em outras palavras, calcule $2 \times 1 + 2 \times 2 + 2 \times 3 + ... + 2 \times n$.

Demonstre a redução de `soma_dobros 3` (que deve resultar em 12).

#### Solução

**Passo 1: Definição da função `soma_dobros`**

Precisamos criar uma função recursiva que, para cada número de 1 a $n$, calcule o dobro e some ao resultado. A estrutura será:
- Se $n = 0$, retornamos 0 (caso base)
- Caso contrário, retornamos $dobro(n) + soma\_dobros(n-1)$ (caso recursivo)

Definimos o corpo da função:

$$SD = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ dobro\ n + f\ (n - 1))$$

Expandindo a definição de `dobro`:

$$SD = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ ((\lambda x. x \times 2)\ n) + f\ (n - 1))$$

Simplificando a aplicação de `dobro`:

$$SD = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + f\ (n - 1))$$

Aplicamos o combinador Y:

$$soma\_dobros = Y\ SD$$

**Passo 2: Cálculo de $soma\_dobros\ 3$ - Primeira recursão**

$$soma\_dobros\ 3 = (Y\ SD)\ 3$$

Pela propriedade do combinador Y:

$$soma\_dobros\ 3 = SD\ (Y\ SD)\ 3$$

Substituindo a definição de $SD$:

$$soma\_dobros\ 3 = (\lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + f\ (n - 1)))\ (Y\ SD)\ 3$$

Beta-redução (substituímos $f$ por $Y\ SD$):

$$soma\_dobros\ 3 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + (Y\ SD)\ (n - 1))\ 3$$

Beta-redução (substituímos $n$ por 3):

$$soma\_dobros\ 3 = \text{if}\ (3 = 0)\ \text{then}\ 0\ \text{else}\ (3 \times 2) + (Y\ SD)\ (3 - 1)$$

Como $3 \neq 0$:

$$soma\_dobros\ 3 = (3 \times 2) + (Y\ SD)\ 2$$

$$soma\_dobros\ 3 = 6 + soma\_dobros\ 2$$

**Passo 3: Cálculo de $soma\_dobros\ 2$ - Segunda recursão**

$$soma\_dobros\ 2 = SD\ (Y\ SD)\ 2$$

$$soma\_dobros\ 2 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + (Y\ SD)\ (n - 1))\ 2$$

$$soma\_dobros\ 2 = \text{if}\ (2 = 0)\ \text{then}\ 0\ \text{else}\ (2 \times 2) + (Y\ SD)\ 1$$

Como $2 \neq 0$:

$$soma\_dobros\ 2 = 4 + (Y\ SD)\ 1 = 4 + soma\_dobros\ 1$$

Substituindo na expressão anterior:

$$soma\_dobros\ 3 = 6 + (4 + soma\_dobros\ 1)$$

**Passo 4: Cálculo de $soma\_dobros\ 1$ - Terceira recursão**

$$soma\_dobros\ 1 = SD\ (Y\ SD)\ 1$$

$$soma\_dobros\ 1 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + (Y\ SD)\ (n - 1))\ 1$$

$$soma\_dobros\ 1 = \text{if}\ (1 = 0)\ \text{then}\ 0\ \text{else}\ (1 \times 2) + (Y\ SD)\ 0$$

Como $1 \neq 0$:

$$soma\_dobros\ 1 = 2 + (Y\ SD)\ 0 = 2 + soma\_dobros\ 0$$

Substituindo:

$$soma\_dobros\ 3 = 6 + (4 + (2 + soma\_dobros\ 0))$$

**Passo 5: Caso base - $soma\_dobros\ 0$**

$$soma\_dobros\ 0 = SD\ (Y\ SD)\ 0$$

$$soma\_dobros\ 0 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (n \times 2) + (Y\ SD)\ (n - 1))\ 0$$

$$soma\_dobros\ 0 = \text{if}\ (0 = 0)\ \text{then}\ 0\ \text{else}\ (0 \times 2) + (Y\ SD)\ (-1)$$

Como $0 = 0$:

$$soma\_dobros\ 0 = 0$$

**Passo 6: Avaliação final**

Substituindo o caso base:

$$soma\_dobros\ 3 = 6 + (4 + (2 + 0))$$

$$soma\_dobros\ 3 = 6 + (4 + 2)$$

$$soma\_dobros\ 3 = 6 + 6$$

$$soma\_dobros\ 3 = 12$$

**Resultado final:** 12

**Verificação:** $2 \times 1 + 2 \times 2 + 2 \times 3 = 2 + 4 + 6 = 12$ ✓

**Observação sobre composição:** Este exercício demonstra como podemos compor uma função simples (`dobro`) com uma estrutura recursiva criada pelo combinador Y. A função `dobro` é aplicada a cada elemento durante a recursão, e os resultados são acumulados através da soma.

**Observação matemática:** A função `soma_dobros` implementa a fórmula: $\sum_{i=1}^{n} 2i = 2\sum_{i=1}^{n} i = 2 \cdot \frac{n(n+1)}{2} = n(n+1)$

Para $n=3$: $3 \times 4 = 12$ ✓

Esta identidade mostra que poderíamos simplificar o cálculo, mas o exercício demonstra como expressar a solução usando recursão e composição de funções no cálculo lambda.

### Exercício 10: Desafio - Fibonacci com Currying

Defina uma função `fibonacci_modificado` usando o combinador Y que:

a) Seja curried, recebendo primeiro um multiplicador $m$ e depois o índice $n$

b) Retorne $m \times fib(n)$, onde $fib(n)$ é o n-ésimo número de Fibonacci

Lembre-se que $fib(0) = 0$, $fib(1) = 1$ e $fib(n) = fib(n-1) + fib(n-2)$ para $n > 1$.

Demonstre:
- A definição completa da função
- A criação de uma função especializada `triplo_fib` que multiplica o resultado por 3
- A redução de `triplo_fib 4` (sabendo que $fib(4) = 3$, o resultado deve ser 9)

#### Solução

**Passo 1: Definição da função Fibonacci básica**

Primeiro, vamos definir a função Fibonacci tradicional usando o combinador Y:

$$Fib = \lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (\text{if}\ (n = 1)\ \text{then}\ 1\ \text{else}\ f\ (n-1) + f\ (n-2)))$$

$$fibonacci = Y\ Fib$$

**Passo 2: Definição da função Fibonacci modificada (curried)**

Para criar uma versão curried que receba primeiro o multiplicador e depois o índice, usaremos composição de funções. Esta é a abordagem mais clara e direta:

$$fibonacci\_modificado = \lambda m. (\lambda n. m \times fibonacci\ n)$$

Esta definição simples cria uma função curried onde primeiro fixamos o multiplicador $m$, e então aplicamos esse multiplicador ao resultado de calcular $fibonacci\ n$. A recursão está contida dentro da função `fibonacci` que já definimos com o combinador Y.

**Passo 3: Criação da função especializada `triplo_fib`**

Aplicamos parcialmente `fibonacci_modificado` fixando o multiplicador em 3:

$$triplo\_fib = fibonacci\_modificado\ 3$$

Expandindo:

$$triplo\_fib = (\lambda m. (\lambda n. m \times fibonacci\ n))\ 3$$

Beta-redução, substituindo $m$ por 3:

$$triplo\_fib = \lambda n. 3 \times fibonacci\ n$$

Esta é uma função que aguarda apenas o índice $n$ e retorna o triplo do n-ésimo número de Fibonacci.

**Passo 4: Cálculo de $triplo\_fib\ 4$**

Primeiro, precisamos calcular $fibonacci\ 4$. Vamos fazer uma redução simplificada, mostrando os valores conhecidos:

**Sub-cálculo: $fibonacci\ 4$**

Sabemos que:
- $fib(0) = 0$
- $fib(1) = 1$
- $fib(2) = fib(1) + fib(0) = 1 + 0 = 1$
- $fib(3) = fib(2) + fib(1) = 1 + 1 = 2$
- $fib(4) = fib(3) + fib(2) = 2 + 1 = 3$

Portanto, $fibonacci\ 4 = 3$

**Cálculo de $triplo\_fib\ 4$:**

$$triplo\_fib\ 4 = (\lambda n. 3 \times fibonacci\ n)\ 4$$

Beta-redução, substituindo $n$ por 4:

$$triplo\_fib\ 4 = 3 \times fibonacci\ 4$$

Substituindo o valor de $fibonacci\ 4$:

$$triplo\_fib\ 4 = 3 \times 3 = 9$$

**Resultado final:** 9

**Passo 5: Demonstração detalhada da recursão de Fibonacci (expansão opcional)**

Para completude, vamos mostrar alguns passos da redução de $fibonacci\ 4$ usando o combinador Y:

$$fibonacci\ 4 = (Y\ Fib)\ 4 = Fib\ (Y\ Fib)\ 4$$

Substituindo a definição de $Fib$:

$$fibonacci\ 4 = (\lambda f. (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (\text{if}\ (n = 1)\ \text{then}\ 1\ \text{else}\ f\ (n-1) + f\ (n-2))))\ (Y\ Fib)\ 4$$

Beta-redução:

$$fibonacci\ 4 = (\lambda n. \text{if}\ (n = 0)\ \text{then}\ 0\ \text{else}\ (\text{if}\ (n = 1)\ \text{then}\ 1\ \text{else}\ (Y\ Fib)\ (n-1) + (Y\ Fib)\ (n-2)))\ 4$$

$$fibonacci\ 4 = \text{if}\ (4 = 0)\ \text{then}\ 0\ \text{else}\ (\text{if}\ (4 = 1)\ \text{then}\ 1\ \text{else}\ (Y\ Fib)\ 3 + (Y\ Fib)\ 2)$$

Como $4 \neq 0$ e $4 \neq 1$:

$$fibonacci\ 4 = (Y\ Fib)\ 3 + (Y\ Fib)\ 2 = fibonacci\ 3 + fibonacci\ 2$$

Sabendo que $fibonacci\ 3 = 2$ e $fibonacci\ 2 = 1$:

$$fibonacci\ 4 = 2 + 1 = 3$$

**Observação sobre currying e especialização:**

Este exercício demonstra um padrão poderoso em programação funcional: criar funções parametrizadas que podem ser facilmente especializadas. A função `fibonacci_modificado` é genérica e permite criar infinitas variações através de aplicação parcial:

- `dobro_fib = fibonacci_modificado 2` calcula o dobro dos números de Fibonacci
- `triplo_fib = fibonacci_modificado 3` calcula o triplo dos números de Fibonacci  
- `identidade_fib = fibonacci_modificado 1` é equivalente à função Fibonacci original
- `meio_fib = fibonacci_modificado 0.5` calcula metade dos números de Fibonacci

**Observação sobre eficiência:**

A implementação recursiva direta de Fibonacci tem complexidade exponencial $O(2^n)$ devido às chamadas redundantes. Por exemplo, ao calcular $fib(4)$, calculamos $fib(3)$ e $fib(2)$, mas $fib(3)$ também calcula $fib(2)$ novamente. Em implementações práticas, técnicas como memoização ou programação dinâmica são usadas para otimizar este cálculo, reduzindo a complexidade para $O(n)$. No entanto, para fins didáticos, a versão recursiva pura demonstra claramente os conceitos do cálculo lambda e a interação entre currying, composição e recursão.

