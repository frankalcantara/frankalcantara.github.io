---
layout: post
title: "First & Follow"
author: Frank
categories:
  - Math
  - Computing Theory
tags:
  - Math
  - Grammar
  - First
  - Follow
  - Parser
image: assets/images/eletromag1.jpg
featured: 2024-03-18T18:35:18.910Z
rating: 0
description: ""
date: 2024-03-18T18:35:18.911Z
preview: ""
keywords: ""
published: 2024-03-18T18:35:18.911Z
slug: 2024-03-18T18:35:18.911Z
type: default
description: Understand how mathematics underpins electromagnetism and its practical applications in an academic article aimed at science and engineering students.
slug: formula-of-attraction-mathematics-supporting-electromagnetism
keywords:
  - Vectorial Calculus
  - Eletromagnetism
  - Math
  - Poetry
  - Vectorial Algebra
rating: 5
published: false
---

# Frist & Follow

Na análise sintática, no contexto de compilação de linguagens de programação, os algoritmos $FRIST$ e $FOLLOW$ são utilizados na construção de tabelas de análise sintática para gramáticas livres de contexto. Esses algoritmos são importantes no processo de análise preditiva, um método de análise sintática *top-down*.

- O algoritmo $FRIST$ é utilizado para determinar o conjunto de terminais que podem aparecer no início de alguma forma derivada de um símbolo não-terminal. Em outras palavras, para um dado não-terminal, o conjunto $FRIST$ contém todos os símbolos terminais que podem começar as cadeias derivadas desse não-terminal.

- O algoritmo $FOLLOW$ é usado para encontrar o conjunto de terminais que podem seguir imediatamente um não-terminal em alguma forma derivada a partir do símbolo inicial da gramática. O conjunto $FOLLOW$ de um não-terminal comtém todos os terminais que podem aparecer logo após o não-terminal em questão, nas derivações da gramática.

Ambos são essenciais na análise sintática LL, uma vez que permitem prever qual regra de produção deve ser aplicada com base no símbolo de entrada atual e no conteúdo da pilha de análise.

## Regras de criação dos conjuntos $FIRST \& FOLLOW$

###Conjunto First

o conjunto $FIRST$ de um símbolo não-terminal é o conjunto de terminais que podem aparecer no início de alguma string derivada desse símbolo.

Para definir o conjunto $FIRST(X)$ para todos os símbolos não-terminais
$X$ de uma gramática que esteja definida por um conjunto de regras de produção, podemos seguir os seguintes passos:

1. **Para símbolos terminais**: o conjunto $FIRST$ é o próprio símbolo terminal. Ou seja, se $a$ é um terminal, então $FIRST(a) = {a}$.

2. **Para um símbolo não-terminal $X$**: olhe para cada regra de produção $X \rightarrow \alpha$ e siga as seguintes regras:

    - Se $\alpha$ é um terminal, adicione $\alpha$ ao conjunto $FIRST(X)$.
    - Se $\alpha$ começa com um símbolo não-terminal $Y$, adicione $FIRST(Y)$ ao $FIRST(X)$, exceto pelo símbolo de vazio $(\varepsilon$) se ele estiver presente.
    - Se $\alpha$ pode derivar em vazio (diretamente ou indiretamente através de outros não-terminais), adicione $\varepsilon$ ao conjunto $FIRST(X)$.

**O símbolo vazio $\varepsilon$ não é sempre um elemento de um conjunto $FIRST$. Ele pode ser um elemento de $FIRST$ se a regra de produção permitir derivação para o vazio, desde que não seja o único elemento do conjunto $FIRST$.**

Repita esses passos até que os conjuntos $FIRST$ de todos os símbolos não-terminais não possam ser alterado.

**Exemplo**: Considere a gramática definida por:

$$
\begin{array}{|c|c|}
\hline
\textbf{Símbolo não-terminal} & \textbf{Regras de produção} \\ \hline
S & aB | bA \\ \hline
A & c | d \\ \hline
B & e | f \\ \hline
\end{array}
$$

Este conjunto de regras de produção permite criar:

$$
\begin{array}{|c|c|c|}
\hline
\textbf{Símbolo} & \textbf{FIRST} & \textbf{Explicação}\\ \hline
S & \{a, b\} & \textbf{S pode ser derivado em "aB" ou "bA"}\\ \hline
A & \{c, d\} & \textbf{A pode ser derivado em "c" ou "d"}\\ \hline
B & \{e, f\} & \textbf{B pode ser derivado em "e" ou "f"}\\ \hline
\end{array}
$$

Logo: $FIRST =\{(S,\{a, b\}),(A,\{c, d\}),(B,\{e, f\})\}$, um conjunto de tuplas.

###Conjunto FOLLOW

O conjunto $FOLLOW$ de um símbolo não-terminal é o conjunto de terminais que podem aparecer imediatamente à direita desse não-terminal em alguma forma sentencial derivada. Para calcular o conjunto $FOLLOW(A)$ para cada não-terminal $A$, siga estes passos:

1. Coloque o símbolo de fim de entrada $(\$)$ no $FOLLOW$ do símbolo inicial da gramática.

  Ao colocar o símbolo de fim de entrada (\$) no $FOLLOW$ do símbolo inicial da gramática, garantimos que o analisador sintático reconheça a última derivação da gramática como válida. Isso significa que o analisador estará preparado para encontrar o símbolo (\$) ao final da string de entrada, indicando que a análise foi concluída com sucesso.

  Em outras palavras, o símbolo (\$) no $FOLLOW$ do símbolo inicial representa a expectativa de que a string de entrada seja completamente processada e que não existam símbolos "soltos" após a última derivada.

2. Para cada produção $X \rightarrow \alpha B \beta$, onde $B$ seja um não-terminal e $\beta$ é uma sequência de símbolos que pode ser vazia:

    - Se $\beta$ não é vazio, adicione todos os símbolos de $FIRST(\beta)$, (exceto por $\varepsilon$, ao $FOLLOW(B)$.
    - Se $\beta$ é vazio ou $FIRST(\beta)$ contém $\varepsilon$, adicione todos os símbolos de $FOLLOW(X)$ ao $FOLLOW(B)$..

Repita esses passos até que os conjuntos $FOLLOW$ de todos os símbolos não-terminais não mudem mais.

**Exemplo**: Considere a gramática definida por:

$$
\begin{array}{|c|c|}
\hline
\textbf{Símbolo não-terminal} & \textbf{Regras de produção} \\ \hline
S & aB | bA \\ \hline
A & c | d \\ \hline
B & e | f \\ \hline
\end{array}
$$

Este conjunto de regras de produção permite criar:

| Símbolo | FOLLOW          | Explicação                                                                                                          |
|---------|-----------------|---------------------------------------------------------------------------------------------------------------------|
| S       | { \$ }          | S é o símbolo inicial, então \$ é o único terminal que pode aparecer à direita de S em uma forma sentencial derivada. |
| A       | { c, d, \$ }    | A pode ser seguido por "c" na regra A -> c, "d" na regra A -> d, ou pelo símbolo de fim de entrada \$ em regras que contêm A. |
| B       | { a, c, d, \$ } | B pode ser seguido por "a" na regra S -> aB, "c" na regra A -> cB, "d" na regra A -> d B, ou pelo símbolo de fim de entrada \$ em regras que contêm B. |


###Construindo a tabela de análise LL(1)

A tabela de análise LL(1) é usada para determinar qual regra de produção aplicar, dada a entrada corrente e o não-terminal no topo da pilha de análise. Para construí-la:

Para cada produção $A\rightarrow \alpha$ faça:

   - Para cada terminal $a$ em $FIRST(\alpha)$, adicione a regra $A\rightarrow \alpha$ à tabela para a combinação $[A,a]$.
   - Se $\varepsilon$ está em $FIRST(\alpha)$, então para cada símbolo $b$ em $FOLLOW(A)$, adicione a regra $A\rightarrow \alpha$ à tabela para a combinação $[A,b]$.
   - Se $\\$$ está em $FOLLOW(A)$, adicione também para $[A, \$]$.

Se há conflitos na tabela (mais de uma regra para a mesma combinação de não-terminal e terminal de entrada), a gramática não é LL(1).

## Exemplo de criação de conjuntos $FIRST \& FOLLOW$
Considerando o seguinte conjunto de regras de produção

$$
\begin{array}{|lc|}
\hline
\textbf{Regras de Produção } \\ \hline
\space \space 𝑅\rightarrow \varepsilon \\ \hline
\space \space 𝑅\rightarrow +𝑆 \\ \hline
\space \space 𝑇\rightarrow 𝐹𝐺 \\ \hline
\space \space 𝐺\rightarrow \varepsilon \\ \hline
\space \space 𝐺\rightarrow ∗𝑇 \\ \hline
\space \space 𝐹\rightarrow 𝑛 \\ \hline
\space \space 𝐹\rightarrow (𝑆) \\ \hline
\end{array}
$$

Crie uma tabela de derivação para um parser LL(1).

### Conjunto FIRST

**Para $F$:**

- $F \rightarrow n$: Adiciona $n$ ao FIRST($F$).
- $F \rightarrow (S)$: Adiciona $($ ao FIRST($F$).

Portanto, FIRST($F$) = \{n, (\}.

**Para $G$:**

- $G \rightarrow \epsilon$: Adiciona $\epsilon$ ao FIRST($G$).
- $G \rightarrow *T$: Adiciona $*$ ao FIRST($G$).

Portanto, FIRST($G$) = \{*, \epsilon\}.

**Para $T$:**

- $T \rightarrow FG$: Como FIRST($F$) inclui $n$ e $($ e $G$ pode ser vazio ($\epsilon$), FIRST($T$) também incluirá $n$ e $($.

Portanto, FIRST($T$) = \{n, (\}.

**Para $R$:**

- $R \rightarrow \epsilon$: Adiciona $\epsilon$ ao FIRST($R$).
- $R \rightarrow +S$: Adiciona $+$ ao FIRST($R$).

Portanto, FIRST($R$) = \{+, \epsilon\}.

**Para $S$:**

- $S \rightarrow TR$: Já que FIRST($T$) inclui $n$ e $($, FIRST($S$) também incluirá $n$ e $($.

Portanto, FIRST($S$) = \{n, (\}.

### Conjunto FOLLOW

Para calcular FOLLOW, consideramos onde cada não-terminal aparece na gramática:

**FOLLOW($S$):**

- $S$ é o símbolo inicial, então inclui \$ no FOLLOW($S$).
- $S$ aparece após $+$ em $R \rightarrow +S$: Adiciona FOLLOW($R$) (que será calculado) a FOLLOW($S$).

FOLLOW($S$) = \{\$\} (e possivelmente mais, dependendo de FOLLOW($R$)).

**FOLLOW($R$):**

- $R$ segue $T$ em $S \rightarrow TR$, então tudo em FOLLOW($S$) (que inclui \$ e mais tarde, outros símbolos) é adicionado a FOLLOW($R$).

FOLLOW($R$) = \{\$\} (e possivelmente mais, dependendo de FOLLOW($S$)).

**FOLLOW($T$):**

- $T$ é seguido por $R$ em $S \rightarrow TR$: Adiciona FIRST($R$) excluindo $\epsilon$ (que é $+$) a FOLLOW($T$). Se $\epsilon$ está em FIRST($R$), também adiciona FOLLOW($S$) (que é \$ e possivelmente mais).

FOLLOW($T$) = \{+, \$\}.

**FOLLOW($G$):**

- $G$ segue $F$ em $T \rightarrow FG$: Adiciona FOLLOW($T$) (que inclui $+$ e \$) a FOLLOW($G$).
- $G$ aparece após $*$ em si mesmo ($G \rightarrow *T$), significando que tudo em FOLLOW($T$) também está em FOLLOW($G$).

FOLLOW($G$) = \{+, \$\}.

**FOLLOW($F$):**

- $F$ é seguido por $G$ em $T \rightarrow FG$: Adiciona FIRST($G$) excluindo $\epsilon$ (que é $*$) a FOLLOW($F$). Se $\epsilon$ está em FIRST($G$), também adiciona FOLLOW($T$) (que é $+$, \$).

- $F$ aparece dentro de $F \rightarrow (S)$ em si mesmo: Isso significa que FOLLOW($F$) também deve incluir $)$ quando $F$ é derivado por $S$.

FOLLOW($F$) = \{*, +, \$, )\}.
### Método para Criar a Tabela LL(1)

1. **Para cada produção da forma $A \rightarrow \alpha$:**
   - Para cada terminal $a$ em $FIRST(\alpha)$, adicione $A \rightarrow \alpha$ à tabela em $[A, a]$.
   - Se $\epsilon$ está em $FIRST(\alpha)$, então para cada símbolo $b$ em $FOLLOW(A)$, adicione $A \rightarrow \alpha$ à tabela em $[A, b]$.
   - Se $\epsilon$ está em $FIRST(\alpha)$ e $FOLLOW(A)$ contém o símbolo de final de arquivo (EOF, representado como $), adicione $A \rightarrow \alpha$ à tabela em $[A, $]$.

### Aplicação ao Exemplo

Vamos aplicar o método para criar a tabela LL(1) para as produções da gramática dada. Aqui estão as produções e os conjuntos $FIRST$ e $FOLLOW$ calculados:

- **Produções:**
  - $S \rightarrow TR$
  - $R \rightarrow \epsilon | +S$
  - $T \rightarrow FG$
  - $G \rightarrow \epsilon | *T$
  - $F \rightarrow n | (S)$

- **Conjuntos FIRST e FOLLOW:**
  - $FIRST(S) = \{n, (\}$
  - $FIRST(R) = \{+, \epsilon\}$
  - $FIRST(T) = \{n, (\}$
  - $FIRST(G) = \{*, \epsilon\}$
  - $FIRST(F) = \{n, (\}$

  - $FOLLOW(S) = \{\$\}$
  - $FOLLOW(R) = \{\$\}$
  - $FOLLOW(T) = \{+, \$\}$
  - $FOLLOW(G) = \{+, \$\}$
  - $FOLLOW(F) = \{*, +, \$, )\}$

### Tabela LL(1)

A tabela tem não-terminais como linhas ($S, R, T, G, F$) e terminais como colunas ($n, (, ), +, *, \$)$. Vamos preencher cada célula conforme as regras.

$$\begin{array}{|c|c|c|c|c|c|c|}
\hline
    & n        & (        & ) & +         & *         & \$ \\
\hline
S   & S \to TR & S \to TR &   &           &           &    \\
\hline
R   &          &          &   & R \to +S  &           & R \to \epsilon \\
\hline
T   & T \to FG & T \to FG &   &           &           &    \\
\hline
G   &          &          &   & G \to \epsilon & G \to *T & G \to \epsilon \\
\hline
F   & F \to n  & F \to (S)&   &           &           &    \\
\hline
\end{array}$$

### Explicação da Tabela

- $S \rightarrow TR$: Como $FIRST(S) = \{n, (\}$, colocamos "S → TR" sob 'n' e '('.
- $R \rightarrow +S | \epsilon$: Com $FIRST(R) = \{+, \epsilon\}$ e $FOLLOW(R) = \{\$\}$, colocamos "R → +S" sob '+' e "R → \epsilon" sob '$', indicando redução a $\epsilon$ se o próximo símbolo for '$'.
- $T \rightarrow FG$: Como $FIRST(T) = \{n, (\}$, colocamos "T → FG" sob 'n' e '('.
- $G \rightarrow *T | \epsilon$: Com $FIRST(G) = \{*, \epsilon\}$ e $FOLLOW(G) = \{+, \$\}$, colocamos "G → *T" sob '*' e "G → \epsilon" sob '+' e '$', já que G pode levar a $\epsilon$.
- $F \rightarrow n | (S)$: Colocamos "F → n" sob 'n' e "F → (S)" sob '('.

As células vazias indicam combinações de entrada para as quais o parser LL(1) não tem regra de produção aplicável, o que pode levar a erros de sintaxe durante a análise. Esta tabela é fundamental para entender como o

## Testando com $n*n$

Para verificar se a string "n*n" faz parte da linguagem gerada pela gramática e se pode ser analisada pelo parser LL(1) usando a tabela de análise que construímos, seguiremos o processo de análise descendente, começando pelo símbolo inicial e utilizando a tabela de análise para guiar a derivação.

Começamos com a pilha contendo o símbolo inicial da gramática, $ S $, e a string de entrada "n*n\$", onde o "$" indica o final da entrada.

1. Olhamos para o topo da pilha $ S $ e o próximo símbolo de entrada ('n').
2. Usamos a tabela de análise para encontrar a produção apropriada, que seria $ S \rightarrow TR $.
3. Substituímos $ S $ pela produção $ TR $ na pilha.
4. O próximo passo é olhar para o topo da pilha $ T $ e o próximo símbolo de entrada ('n').
5. A tabela indica $ T \rightarrow FG $. Substituímos $ T $ por $ FG $.
6. O próximo passo é olhar para o topo da pilha $ F $ e o próximo símbolo de entrada ('n').
7. A tabela indica $ F \rightarrow n $. Substituímos $ F $ por 'n'.
8. Agora a pilha tem 'nG' e a entrada é "n*n". O próximo passo é consumir 'n' da entrada e da pilha, pois eles coincidem.
9. O topo da pilha agora é $ G $ e o próximo símbolo de entrada é '*'.
10. A tabela indica $ G \rightarrow *T $. Substituímos $ G $ por '*T'.
11. Consumimos '*' da entrada e da pilha, pois eles coincidem.
12. O topo da pilha agora é $ T $ novamente e o próximo símbolo de entrada é 'n'.
13. A tabela indica $ T \rightarrow FG $. Substituímos $ T $ por $ FG $.
14. O próximo passo é olhar para o topo da pilha $ F $ e o próximo símbolo de entrada ('n').
15. A tabela indica $ F \rightarrow n $. Substituímos $ F $ por 'n'.
16. Consumimos 'n' da entrada e da pilha, pois eles coincidem.
17. O topo da pilha agora é $ G $ e o próximo símbolo de entrada é '\$'.
18. A tabela indica $ G \rightarrow \epsilon $. Substituímos $ G $ por $ \epsilon $ e removemos $ G $ da pilha.
19. Finalmente, a pilha está vazia e a entrada foi completamente consumida.

Como conseguimos consumir toda a entrada "n\*n$" e esvaziar a pilha sem encontrar nenhum erro, a string "n\*n" é de fato parte da linguagem identificada por este parser LL(1).



## Testando com "n*n+n"

Para analisar a string "n*n+n" usando o parser LL(1) e a tabela de análise construída, seguiríamos o mesmo processo que usamos para a string anterior "n*n". No entanto, desta vez, adicionaremos o terminal '+' na entrada. Vamos ver o passo a passo:

Inicializamos a pilha com o símbolo inicial da gramática, que é $S$, e preparamos a string de entrada como "n*n+n\$", incluindo o sinal de fim de entrada "\$".

1. Olhamos para o topo da pilha $S$ e o primeiro símbolo de entrada ('n'). A tabela de análise nos diz para usar a produção $S \rightarrow TR$.
2. Substituímos $S$ por $TR$ na pilha.
3. Agora o topo da pilha é $T$, e o próximo símbolo de entrada é 'n'. A tabela de análise nos indica que devemos usar $T \rightarrow FG$.
4. Substituímos $T$ por $FG$ na pilha.
5. O topo da pilha agora é $F$, e o próximo símbolo de entrada ainda é 'n'. A tabela de análise indica $F \rightarrow n$, então substituímos $F$ por 'n' e consumimos o 'n' da entrada.
6. Agora o topo da pilha é $G$, e o próximo símbolo de entrada é '*'. A tabela nos diz que devemos usar $G \rightarrow *T$.
7. Substituímos $G$ por '*T' na pilha.
8. Consumimos '*' da entrada.
9. O topo da pilha agora é $T$ novamente, e o próximo símbolo de entrada é 'n'. Novamente, a tabela de análise nos indica que devemos usar $T \rightarrow FG$.
10. Substituímos $T$ por $FG$ na pilha.
11. O topo da pilha agora é $F$, e o próximo símbolo de entrada é 'n'. Usamos a produção $F \rightarrow n$, substituímos $F$ por 'n' e consumimos 'n' da entrada.
12. O topo da pilha agora é $G$, e o próximo símbolo de entrada é '+'. A tabela de análise nos indica que devemos usar $G \rightarrow \epsilon$, então retiramos $G$ da pilha.
13. O topo da pilha agora é $R$, e o próximo símbolo de entrada é '+'. A tabela nos diz que devemos usar $R \rightarrow +S$.
14. Substituímos $R$ por '+S' na pilha.
15. Consumimos '+' da entrada.
16. O topo da pilha agora é $S$, e o próximo símbolo de entrada é 'n'. A tabela de análise nos indica que devemos usar $S \rightarrow TR$.
17. Repetimos o processo para a nova ocorrência de $TR$ na pilha, com a entrada restante sendo 'n$'.

Finalmente, se todos os símbolos da entrada forem consumidos adequadamente e a pilha for esvaziada, então a string "n*n+n" é aceita pela gramática e o parser LL(1).

## Testando com "n-n*n"

Vamos analisar a string "n-n*n" utilizando a tabela de análise LL(1) que foi estabelecida anteriormente, seguindo os passos do parser LL(1):

Começamos com a pilha inicial contendo o símbolo inicial $S$ e a entrada "n-n*n\$".

1. A entrada começa com 'n', então olhamos para $S$ na tabela de análise e aplicamos a produção $S \rightarrow TR$.
2. Substituímos $S$ por $TR$ na pilha.
3. Com 'n' ainda como entrada e $T$ no topo da pilha, aplicamos $T \rightarrow FG$.
4. Substituímos $T$ por $FG$ na pilha.
5. O topo da pilha é $F$ e a entrada é 'n'. Aplicamos $F \rightarrow n$.
6. Consumimos 'n' da entrada, igualando a entrada e a pilha.
7. Agora $G$ está no topo da pilha e o próximo símbolo de entrada é '-', mas na nossa tabela de análise LL(1), não temos uma produção para $G$ quando o próximo símbolo de entrada é '-', o que implica que o parser não pode prosseguir. A gramática parece não incluir produções para lidar com o terminal '-', o que é necessário para analisar expressões com subtração.

Portanto, a string "n-n\*n" **não faz parte da linguagem definida pela gramática fornecida**, porque a gramática não tem regras para lidar com o símbolo '-'. Isso significa que a gramática dada não pode gerar a string fornecida, e o parser LL(1) falhará ao tentar analisá-la. Para que a gramática suporte operações como a subtração, regras adicionais precisariam ser incluídas, o que exigiria uma revisão da definição de $G$ e $R$ para incluir e tratar corretamente o operador '-'.

## Uma Gramática para Aritmética

Considere o seguinte conjunto de regras de produção:

$$
\begin{array}{|lc|}
\hline
\textbf{Regras de Produção } \\ \hline
\space \space S \rightarrow E \\ \hline
\space \space E \rightarrow E + T \,|\, E - T \,|\, T \\ \hline
\space \space T \rightarrow T * F \,|\, T / F \,|\, F \\ \hline
\space \space F \rightarrow n \,|\, (E) \\ \hline
\end{array}
$$

Cuja criação foi baseada nos seguintes conceitos:

- $S$: Símbolo de início da gramática.
- $E$: Representa uma expressão, que pode ser uma soma ou subtração de termos ($T$), ou simplesmente um termo.
- $T$: Representa um termo, que pode ser um produto ou divisão de fatores ($F$), ou simplesmente um fator.
- $F$: Representa um fator, que pode ser um número inteiro ($n$) ou uma expressão entre parênteses.

Esse conjunto de regras de produção define uma gramática que reflete a precedência dos operadores aritméticos, de tal forma que: $*$ e $/$ têm maior precedência que $+$ e $-$, e os parênteses podem ser usados para alterar a ordem de avaliação das expressões.

A recursão à esquerda nas regras para $E$ e $T$ permite a continuação das expressões e termos, respectivamente. A opção de $F\rightarrow (E)$ permite o uso de parênteses para agrupar expressões e alterar a precedência padrão dos operadores.

Note que a representação de $n$ como um número inteiro é abstrata; em uma implementação real, você poderia criar regras adicionais para definir a estrutura de um número inteiro, como uma sequência de dígitos. Ou, se for o caso, criar regras que incluam os números reais e os números inteiros.

### $FIRST \& FOLLOW$

#### Cálculo dos Conjuntos FIRST

##### FIRST(S)

Como $S$ produz $E$ e só $E$, FIRST(S) = FIRST(E).

##### FIRST(E)

- $E$ pode produzir $T$ diretamente, então FIRST(E) inclui FIRST(T).
- Também pode iniciar com $E + T$ ou $E - T$, mas isso eventualmente reduz a $T$, então não adicionamos novos símbolos a FIRST(E) baseado nisso.
- Portanto, FIRST(E) é baseado em FIRST(T).

##### FIRST(T)

- $T$ pode produzir $F$ diretamente, então FIRST(T) inclui FIRST(F).
- Também pode iniciar com $T * F$ ou $T / F$, mas isso eventualmente reduz a $F$, então não adicionamos novos símbolos a FIRST(T) baseado nisso.
- Assim, FIRST(T) = FIRST(F).

##### FIRST(F)

- $F$ pode produzir 'n' diretamente, então 'n' está em FIRST(F).
- $F$ também pode produzir $(E)$, então '(' está em FIRST(F).
- Portanto, FIRST(F) = {'n', '('}.

Com base nas definições acima e entendendo que as produções para $E$ e $T$ são recursivas à esquerda e apenas rearranjam os mesmos conjuntos de símbolos iniciais (ou seja, eles sempre começam com $T$ ou $F$, que levam a 'n' ou '('), podemos concluir:

$$FIRST(S) = FIRST(E) = FIRST(T) = FIRST(F) = {'n', '('}$$

Esses conjuntos FIRST indicam que as strings derivadas de $S$, $E$, $T$, e $F$ podem começar com um número inteiro ou um parêntese aberto, refletindo as opções de iniciar uma expressão aritmética com um número ou uma expressão aninhada entre parênteses.

#### Cálculo dos Conjuntos FOLLOW

##### FOLLOW(S)

- $S$ é o símbolo inicial, então incluímos o símbolo de fim de entrada \$ em FOLLOW(S). Não há produções que levem a $S$ diretamente, então não adicionamos mais nada.
- FOLLOW(S) = {\$}

##### FOLLOW(E)

- $E$ aparece dentro dos parênteses em $F \rightarrow (E)$, então $)$ está em FOLLOW(E).
- Todas as instâncias de $E$ estão seguidas por $+$ ou $-$ quando não são a última coisa numa produção, então $+$ e $-$ estão em FOLLOW(E).
- Como $E$ é o início da gramática (através de $S$), tudo em FOLLOW(S) também está em FOLLOW(E).
- FOLLOW(E) = {+, -, ), \$}

##### FOLLOW(T)

- $T$ é seguido por $+$ e $-$ nas produções de $E$, então $+$ e $-$ estão em FOLLOW(T).
- $T$ também aparece à esquerda de $*$ e $/$ em suas próprias produções, então $*$ e $/$ estão em FOLLOW(T).
- Além disso, qualquer coisa em FOLLOW(E) deve estar em FOLLOW(T), pois $E$ pode terminar com $T$.
- FOLLOW(T) = {+, -, *, /, ), \$}

##### FOLLOW(F)

- $F$ é seguido por $*$ e $/$ em produções de $T$, então $*$ e $/$ estão em FOLLOW(F).
- $F$ também é o último não-terminal nas produções de $T$ e $E$, assim tudo que está em FOLLOW(T) e FOLLOW(E) também deve estar em FOLLOW(F).
- FOLLOW(F) = {+, -, *, /, ), \$}

### Conclusão

Os conjuntos FOLLOW para cada não-terminal são:

- FOLLOW(S) = {\$}
- FOLLOW(E) = {+, -, ), \$}
- FOLLOW(T) = {+, -, *, /, ), \$}
- FOLLOW(F) = {+, -, *, /, ), \$}

Esses conjuntos FOLLOW ajudam a definir os símbolos que podem seguir cada não-terminal na derivação de strings da gramática, orientando a construção de um parser preditivo para expressões aritméticas.
A tabela é organizada com não-terminais nas linhas e terminais (incluindo o final de entrada \$) nas colunas. Vamos preencher as entradas da tabela de acordo com as produções de nossa gramática.

$$
\begin{array}{|l|c|c|c|c|c|c|c|c|}
\hline
 & \textbf{n} & \textbf{(} & \textbf{)} & \textbf{+} & \textbf{-} & \textbf{*} & \textbf{/} & \textbf{\$} \\ \hline
\textbf{S} & S \to E & S \to E & & & & & & \\ \hline
\textbf{E} & E \to TE' & E \to TE' & & E' \to \epsilon & E' \to \epsilon & & & E' \to \epsilon \\ \hline
\textbf{E'} & & & E' \to \epsilon & E' \to +TE' & E' \to -TE' & & & E' \to \epsilon \\ \hline
\textbf{T} & T \to FT' & T \to FT' & & T' \to \epsilon & T' \to \epsilon & & & T' \to \epsilon \\ \hline
\textbf{T'} & & & T' \to \epsilon & T' \to \epsilon & T' \to \epsilon & T' \to *FT' & T' \to /FT' & T' \to \epsilon \\ \hline
\textbf{F} & F \to n & F \to (E) & & & & & & \\ \hline
\end{array}
$$

### Preenchimento da Tabela

- Para $S \rightarrow E$, usamos diretamente na entrada 'n' e '(', que são os únicos símbolos em FIRST(S).
- Para $E \rightarrow T E'$, aplicamos quando o próximo símbolo é 'n' ou '(', que estão em FIRST(T). Para as produções $E \rightarrow E + T$ e $E \rightarrow E - T$, olhamos para FOLLOW(E), colocando essas produções nas colunas '+' e '-', respectivamente, porque esses são os símbolos que podem seguir um $E$ na entrada.
- Para $T$, o raciocínio é similar: $T \rightarrow F T'$ se aplica para 'n' e '(', enquanto $T \rightarrow T * F$ e $T \rightarrow T / F$ são usadas quando os próximos símbolos são '*' e '/', respectivamente, refletindo a precedência dos operadores.
- $F \rightarrow n$ é usado quando o próximo símbolo de entrada é 'n', e $F \rightarrow (E)$ quando o próximo símbolo é '('.

Note que a tabela deixa várias células vazias, indicando erros de parsing para combinações de entrada não-terminais/terminais não cobertas pelas regras da gramática. Além disso, não há entradas para o símbolo de fechamento de parênteses ')' e o final de entrada '$' na maior parte das linhas, pois esses símbolos são tratados via mecanismos de pilha e lookahead no processo de parsing, dependendo do contexto fornecido pelos conjuntos FOLLOW.

### Observações Importantes

1. **Tratamento de Erros:** A tabela de derivação apresentada é didática e oferece um caminho para analisar strings válidas de acordo com uma gramática explicitada por suas **Regras de Produção**. Porém, em um cenário real, entradas inválidas são comuns, e um _parser_ precisa de mecanismos para identificar, reportar e possivelmente corrigir erros sintáticos. Isso irá envolver estratégias diversas como pular _tokens_ até encontrar um que faça sentido no contexto atual ou fornecer mensagens de erro detalhadas que ajudem o usuário a corrigir a entrada.

2. **Manipulação de Símbolos de Fechamento de Escopo:** A gramática envolve o uso de parênteses para delinear o escopo de expressões. A tabela assume que os parênteses serão tratados corretamente como parte das regras de produção. No entanto, a implementação do parser precisa gerenciar explicitamente o escopo aberto e fechado pelos parênteses, garantindo que cada parêntese aberto seja correspondido por um fechamento apropriado e que os escopos sejam aninhados corretamente. Uma técnica para a solução deste problema envolve contagem positiva e negativa.

3. **Precedência e Associatividade de Operadores:** Enquanto a gramática e a tabela LL(1) refletem a precedência dos operadores (por exemplo, `*` e `/` têm maior precedência que `+` e `-`), a implementação precisa garantir que essas regras sejam aplicadas consistentemente para avaliar expressões corretamente. Além disso, a associatividade dos operadores (por exemplo, da esquerda para a direita) também precisa ser considerada ao analisar expressões com múltiplos operadores do mesmo nível de precedência.

4. **Otimizações:** Implementações reais de parsers frequentemente incorporam otimizações para melhorar a eficiência, tanto em termos de velocidade quanto de uso de memória. Isso pode incluir técnicas para reduzir a quantidade de backtracking (se houver) ou para reutilizar resultados de análises de subexpressões comuns.

