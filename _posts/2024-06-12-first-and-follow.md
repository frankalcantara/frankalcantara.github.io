---
layout: post
title: First & Follow
author: Frank
categories:
    - Matemática
    - Linguagens Formais
tags:
    - Matemática
    - Linguagens Formais
    - Compiladores
image: assets/images/first.webp
description: Como construir os conjuntos First  e Follow que irão permitir a criação da tabela de análise para o algoritmo de parser.
slug: first-follow
keywords:
    - parsers
    - first
    - follow
    - derivation
    - derivação
rating: 5
published: 2024-06-12T13:16:53.564Z
draft: 2024-06-12T13:00:39.229Z
preview: Como construir os conjuntos First  e Follow que irão permitir a criação da tabela de análise para o algoritmo de parser.
---

Não dá nem para começar a pensar em criar um *parser* $LL(1)$ se não entender os conjuntos $FIRST$ e $FOLLOW$. [Também não dá para entender estes conjuntos se não souber o que é um *parser* $LL(1)$](https://frankalcantara.com/parsers-ll(1)/) Imagine que você está aprendendo um novo idioma. Para formar frases corretas, você precisará entender quais palavras podem vir antes ou depois de outras. Ou corre o risco de falar como o Yoda. Se quiser evitar ser confundido com um velho alienígena, precisa aprender, no mínimo, a ordem das palavras, muito antes de entender a classe gramatical destas mesmas palavras. Como uma criança aprendendo a falar.

Eu forcei um pouco a barra na metáfora, mas na análise sintática de linguagens livres de contexto, os conjuntos $FIRST$ e $FOLLOW$ desempenham um papel crucial que quase valida minha metáfora. Estes conjuntos ajudam a decifrar a gramática da linguagem de forma determinística determinando as regras de produção que serão  aplicadas aos símbolos da *string* de entrada para garantir que ele faça parte da linguagem.

O conjunto $FIRST$ de um símbolo não-terminal será composto dos símbolos terminais que podem aparecer como **primeiro símbolo** de qualquer sequência de símbolos que seja derivada desse não-terminal. Em outras palavras, o conjunto $FIRST$ indica quais terminais podem iniciar uma declaração válida (frase) dentro da estrutura gramática definida por um não-terminal. Por exemplo, considere uma gramática para definir expressões aritméticas. O não-terminal *EXPR* pode derivar diversas sequências de símbolos, como *2 + 3*, *(4 * 5)*, *x - y*. O conjunto $FIRST$ do não-terminal *EXPR* seria, neste caso específico, ${número, '+', '-', '('}$, pois esses são os símbolos que podem iniciar qualquer expressão aritmética válida nesta gramática até onde podemos saber com as informações passadas neste parágrafo. Uma gramática para todas as expressões aritméticas possíveis teria um conjunto $FIRST$ maior.

O conjunto $FOLLOW$, por sua vez, determina o conjunto de símbolos terminais que podem aparecer **imediatamente após** um não-terminal em alguma derivação da gramática. Ou colocando de outra forma, o conjunto $FOLLOW$ indica quais terminais podem seguir (*follow*) um não-terminal em uma declaração válida da linguagem.

Diferentemente do $FIRST$, que se concentra no início de uma derivação, o $FOLLOW$ analisa a situação em que um não-terminal aparece, considerando as produções diretas do não-terminal e também as produções de outros não terminais que, por ventura, contenham o não-terminal em análise. Por exemplo, considere uma gramática que define declarações de variáveis. O não-terminal *DECLARACAO_VAR* pode ser seguido por diferentes símbolos, dependendo do contexto. Em uma linguagem como o $C$, uma declaração de variável pode terminar com um ponto e vírgula, ser seguida por um operador de atribuição e uma expressão, ou até mesmo ser parte de uma estrutura maior. Neste cenário, o conjunto $FOLLOW$ do não-terminal *DECLARACAO_VAR* incluiria, portanto, o ponto e vírgula ';', o sinal de igual '=', e todos os outros símbolos que podem iniciar uma expressão ou um comando que a linguagem permita ocorrer na mesma linha da declaração da variável.

Os conjuntos $FIRST$ e $FOLLOW$ serão utilizados para construir a Tabela de Derivação $LL(1)$. A forma tecnicamente mais correta seria dizer que estes conjuntos formam a Tabela De Análise $LL(1)$. Entretanto, pobre de mim, prefiro chamar de Tabela de Derivação.

As Tabelas de Derivação são tabelas que guiam o processo de análise sintática descendente preditiva no *parser* $LL(1)$ deterministicamente. Cada célula dessas tabelas corresponde a relação que existe em um par não-terminal, terminal. De forma que o valor da célula apontada por este par indica qual regra de produção deve ser aplicada quando o analisador encontrar este par específico durante a análise preditiva $LL(1)$.

## O Conjunto $FIRST$

O conjunto $FIRST$ de um símbolo não-terminal é o conjunto de todos os terminais que podem aparecer no início de qualquer *string* derivada desse símbolo, incluindo o símbolo vazio ($\varepsilon$) se o não-terminal puder derivar a *string* vazia. Para os símbolos terminais, o elemento do conjunto $FIRST$ será o próprio símbolo terminal.

### Regras de Criação do Conjunto $FIRST$

Para definir o conjunto $FIRST(X)$ para todos os símbolos não-terminais
$X$ de uma gramática que esteja definida por um conjunto de regras de produção, podemos seguir os seguintes passos:

1. **Para símbolos terminais**: o conjunto $FIRST$ é o próprio símbolo terminal. Ou seja, se $a$ é um terminal, então $FIRST(a) = {a}$.

2. **Para um símbolo não-terminal $X$**: olhe para cada regra de produção $X \rightarrow \alpha$ e siga as seguintes regras:

    - Se $\alpha$ é um terminal, adicione $\alpha$ ao conjunto $FIRST(X)$.
    - Se $\alpha$ começa com um símbolo não-terminal $Y$, adicione $FIRST(Y)$ ao $FIRST(X)$, exceto pelo símbolo de vazio $(\varepsilon$) se ele estiver presente.
    - Se $\alpha$ consiste apenas em não-terminais e todos eles podem derivar em vazio (diretamente ou indiretamente), adicione $\varepsilon$ ao conjunto $FIRST(X)$.
    
**O símbolo vazio $\varepsilon$ pertence ao conjunto FIRST(X) se, e somente se, $X$ pode derivar a string vazia (diretamente ou indiretamente).**

Repita esses passos até que os conjuntos $FIRST$ de todos os símbolos não-terminais não possam ser alterado.

### Exemplo 1: Criação de Conjuntos $FIRST$

Considere a gramática definida pelo seguinte conjunto de regras de produção:

$$
\begin{array}{cc}
1. &S \rightarrow aB \vert bA \\
2. &A \rightarrow c \vert d \\
3. &B \rightarrow e \vert f \\
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

Agora que entendemos o algoritmo, podemos tentar criar um pseudocódigo para encontrar os elementos do conjunto $First$.

### Algoritmo para calcular o conjunto FIRST

```pseudo
## Algoritmo para calcular o conjunto FIRST para símbolos não-terminais

# Entrada: Um conjunto de regras de produção P
# Saída: Um dicionário FIRST, onde FIRST[X] é o conjunto FIRST do símbolo não-terminal X

função calcular_FIRST(gramática):
    FIRST = {}  # Inicializa o dicionário FIRST

    # Passo 1: Inicialização para não-terminais
    para cada símbolo não-terminal X na gramática:
        FIRST[X] <- {}

    # Passo 2: Iteração para não-terminais
    mudou = verdadeiro
    enquanto mudou:
        mudou = falso
        para cada regra de produção X \rightarrow Y1 Y2 ... Yn na gramática:
            k = 0
            adicionou_epsilon = verdadeiro
            enquanto k < n e adicionou_epsilon:
                adicionou_epsilon = falso
                Yk = Y[k]

                # Se Yk é terminal, adicionar Yk ao FIRST[X]
                se Yk é terminal:
                    se Yk não está em FIRST[X]:
                        adicionar Yk a FIRST[X]
                        mudou = verdadeiro
                # Se Yk é não-terminal, adicionar FIRST[Yk] ao FIRST[X], exceto \varepsilon
                senão:
                    para cada símbolo t em FIRST[Yk]:
                        se t != "\varepsilon":
                            se t não está em FIRST[X]:
                                adicionar t a FIRST[X]
                                mudou = verdadeiro
                        senão:
                            adicionou_epsilon = verdadeiro
                k = k + 1

            # Se todos os Y1, Y2, ..., Yn podem derivar \varepsilon, adicionar \varepsilon ao FIRST[X]
            se k == n e adicionou_epsilon:
                se "\varepsilon" não está em FIRST[X]:
                    adicionar "\varepsilon" a FIRST[X]
                    mudou = verdadeiro

    retornar FIRST

```

Este pseudocódigo, poderia ser criado em python com um código parecido com este:

```python
def calcular_FIRST(producoes):
    FIRST = {}

    # Passo 1: Inicialização para não-terminais
    # Identificamos todos os símbolos não-terminais presentes nas produções
    nao_terminais = {regra.split('\rightarrow')[0].strip() for regra in producoes}

    # Inicializamos o conjunto FIRST de cada não-terminal como um conjunto vazio
    for nao_terminal in nao_terminais:
        FIRST[nao_terminal] = set()

    mudou = True
    # O loop continua até que não haja mais mudanças nos conjuntos FIRST
    while mudou:
        mudou = False
        # Iteramos por todas as produções da gramática
        for producao in producoes:
            partes = producao.split('\rightarrow')
            X = partes[0].strip()  # Não-terminal do lado esquerdo da produção
            Y = partes[1].strip().split()  # Lista de símbolos do lado direito da produção

            k = 0
            adicionou_epsilon = True  # Flag para controlar a adição de \varepsilon
            # Iteramos sobre os símbolos do lado direito da produção
            while k < len(Y) and adicionou_epsilon:
                adicionou_epsilon = False
                Yk = Y[k]

                # Se Yk é um não-terminal, adicionamos seus FIRST ao FIRST de X
                if Yk in nao_terminais:
                    for simbolo in FIRST[Yk]:
                        if simbolo != "\varepsilon":
                            if simbolo not in FIRST[X]:
                                FIRST[X].add(simbolo)
                                mudou = True
                        else:
                            adicionou_epsilon = True
                else:
                    # Se Yk é um terminal, adicionamos Yk ao FIRST de X
                    if Yk not in FIRST[X]:
                        FIRST[X].add(Yk)
                        mudou = True
                    adicionou_epsilon = False  # Paramos de adicionar se encontramos um terminal
                k += 1

            # Se todos os símbolos Y1, Y2, ..., Yn podem derivar \varepsilon, adicionamos \varepsilon ao FIRST de X
            if k == len(Y) and adicionou_epsilon:
                if "\varepsilon" not in FIRST[X]:
                    FIRST[X].add("\varepsilon")
                    mudou = True

    return FIRST

# Exemplo de uso
producoes = [
    "S \rightarrow a B",
    "S \rightarrow b A",
    "A \rightarrow c",
    "A \rightarrow d",
    "B \rightarrow e",
    "B \rightarrow f"
]

FIRST = calcular_FIRST(producoes)
for nao_terminal in FIRST:
    print(f"FIRST({nao_terminal}) = {FIRST[nao_terminal]}")
```
## O Conjunto $FOLLOW$

O conjunto $FOLLOW$ de um símbolo não-terminal é o conjunto de terminais que podem aparecer imediatamente à direita (após, *follow*) desse não-terminal em alguma forma sentencial derivada, **ou o símbolo de fim de entrada ($) se o não-terminal puder aparecer no final de uma forma sentencial**.

Para definir o conjunto $FOLLOW(A)$ para cada não-terminal $A$, siga estes passos:

1. Coloque o símbolo de fim de entrada $(\$)$ no $FOLLOW$ do símbolo inicial da gramática. Ao colocar o símbolo de fim de entrada (\$) no $FOLLOW$ do símbolo inicial da gramática, garantimos que o analisador sintático reconheça a última derivação da gramática como válida. Isso significa que o analisador estará preparado para encontrar o símbolo ($\$$) ao final da string de entrada, indicando que a análise foi concluída com sucesso.  Em outras palavras, o símbolo ($\$$) no $FOLLOW$ do símbolo inicial representa a expectativa de que a string de entrada seja completamente processada e que não existam símbolos após a última derivada.

2. Para cada produção da forma $A \rightarrow \alpha B \beta$, onde $B$ é um não-terminal:
  
  - Se $\beta$ não deriva $\varepsilon$ (a string vazia), adicione $FIRST(\beta)$ (sem $\varepsilon$) a $FOLLOW(B)$.
  - Se $\beta$ deriva $\varepsilon$ (a string vazia) **ou $\beta$ é a string vazia**, adicione $FOLLOW(A)$ a $FOLLOW(B)$.

Repita esses passos até que os conjuntos $FOLLOW$ de todos os símbolos não-terminais não mudem mais.

**Exemplo**: Considere a gramática definida por:

$$
\begin{array}{cc}
1. & S \rightarrow aB \vert bA \\
2. & A \rightarrow c \vert d \\
3. & B \rightarrow e \vert f \\
\end{array}
$$

**Conjunto FIRST:**

*   $FIRST(S) = \{a, b\}$  (S pode derivar em $aB$ ou $bA$)
*   $FIRST(A) = \{c, d\}$  (A pode derivar em $c$ ou $d$)
*   $FIRST(B) = \{e, f\}$  (B pode derivar em $e$ ou $f$)

**Conjunto FOLLOW:**

1.  $FOLLOW(S) = \{\$\}$ ($S$ é o símbolo inicial)

2.  $FOLLOW(A) = \{\$, c, d\}$
    *   A aparece em: $S \to bA$
    *   Como não há nada após $A$ na produção acima, adicionamos $FOLLOW(S)$ a $FOLLOW(A)$: $\{\$\}$
    *   Além disso, como $A$ aparece após $b$ na produção $S \to bA$, e $B$ pode derivar em $c$ ou $d$ ($B \to c \vert d$), então $c$ e $d$ também podem seguir $A$.

3.  $FOLLOW(B) = \{\$\}$
    *   $B$ aparece em: $S \rightarrow aB$
    *   Como não há nada após $B$ na produção acima, adicionamos $FOLLOW(S)$ a $FOLLOW(B)$: $\{\$\}$

Criamos o conjunto $FIRST$ porque este é necessário para a criação do conjunto $FOLLOW$. Mas, neste momento nos interessa apenas o conjunto $FOLLOW$. O conjunto resultante será:

<table>
  <tr>
    <th>Símbolo</th>
    <th>$FOLLOW$</th>
    <th>Explicação</th>
  </tr>
  <tr>
    <td>$S$</td>
    <td>$\{ \$ \}$</td>
    <td>$S$ é o símbolo inicial, então $\$$ é o único terminal que pode aparecer à direita de $S$ em uma forma sentencial derivada.</td>
  </tr>
  <tr>
    <td>A</td>
    <td>$\{ \$, c, d \}$</td>
    <td>$A$ pode ser seguido por $c$ na regra $A \to c$, $d$ na regra $A \to d$, ou pelo símbolo de fim de entrada $\$$ em regras que contêm $A$.</td>
  </tr>
  <tr>
    <td>$B$</td>
    <td>$\{ a, c, d, \$ \}$</td>
    <td>$B$ pode ser seguido por $a$ na regra $S \to aB$, $c$ na regra $A \to cB$, $d$ na regra $A \to dB$, ou pelo símbolo de fim de entrada $\$$ em regras que contêm $B$.</td>
  </tr>
</table>


## Algoritmo para calcular o conjunto FOLLOW

Assim como fizemos com o $FIRST$ podemos criar um algoritmo para criar o conjunto $FOLLOW$:

```pseudo
# Entrada: Um conjunto de regras de produção P e o símbolo inicial S
# Saída: Um dicionário FOLLOW, onde FOLLOW[X] é o conjunto FOLLOW do símbolo não-terminal X

função calcular_FOLLOW(producoes, simbolo_inicial):
    FOLLOW = {}  # Inicializa o dicionário FOLLOW

    # Passo 1: Inicialização para todos os não-terminais
    para cada símbolo não-terminal X na gramática:
        FOLLOW[X] <- {}

    # Passo 2: Colocar o símbolo de fim de entrada ($) no FOLLOW do símbolo inicial
    FOLLOW[simbolo_inicial] <- {$}

    # Passo 3: Iteração para todos os não-terminais
    mudou = verdadeiro
    enquanto mudou:
        mudou = falso
        para cada regra de produção A \rightarrow \alpha na gramática:
            \alpha = \alpha.split()  # Divide a produção em símbolos individuais
            para cada símbolo B na produção \alpha:
                se B é um não-terminal:
                    # Verifica os símbolos após B na produção
                    para cada símbolo após B na produção:
                        se o símbolo é terminal:
                            se símbolo não está em FOLLOW[B]:
                                adicionar símbolo a FOLLOW[B]
                                mudou = verdadeiro
                            pare
                        senão:
                            para cada símbolo t em FIRST[símbolo]:
                                se t != "\varepsilon":
                                    se t não está em FOLLOW[B]:
                                        adicionar t a FOLLOW[B]
                                        mudou = verdadeiro
                            se "\varepsilon" não está em FIRST[símbolo]:
                                pare
                    # Se não há mais símbolos após B ou todos podem derivar \varepsilon
                    se não há mais símbolos após B ou todos podem derivar \varepsilon:
                        para cada símbolo t em FOLLOW[A]:
                            se t não está em FOLLOW[B]:
                                adicionar t a FOLLOW[B]
                                mudou = verdadeiro

    retornar FOLLOW
```
Agora que temos um pseudo código, podemos partir para o código em Python, para isso é preciso lembrar que vamos precisar do conjunto $FIRST$ para encontrar o conjunto $FOLLOW$.

```python
def calcular_FIRST(producoes):
    FIRST = {}

    # Passo 1: Inicialização para não-terminais
    nao_terminais = {regra.split('\rightarrow')[0].strip() for regra in producoes}
    for nao_terminal in nao_terminais:
        FIRST[nao_terminal] = set()

    mudou = True
    while mudou:
        mudou = False
        for producao in producoes:
            partes = producao.split('\rightarrow')
            X = partes[0].strip()
            Y = partes[1].strip().split()

            k = 0
            adicionou_epsilon = True
            while k < len(Y) and adicionou_epsilon:
                adicionou_epsilon = False
                Yk = Y[k]

                if Yk in nao_terminais:
                    for simbolo in FIRST[Yk]:
                        if simbolo != "\varepsilon":
                            if simbolo not in FIRST[X]:
                                FIRST[X].add(simbolo)
                                mudou = True
                        else:
                            adicionou_epsilon = True
                else:
                    if Yk not in FIRST[X]:
                        FIRST[X].add(Yk)
                        mudou = True
                    adicionou_epsilon = False
                k += 1

            if k == len(Y) and adicionou_epsilon:
                if "\varepsilon" not in FIRST[X]:
                    FIRST[X].add("\varepsilon")
                    mudou = True

    return FIRST

def calcular_FOLLOW(producoes, simbolo_inicial):
    FIRST = calcular_FIRST(producoes)  # Calcula o conjunto FIRST necessário para FOLLOW
    FOLLOW = {}

    # Inicializa FOLLOW para todos os não-terminais
    nao_terminais = {regra.split('\rightarrow')[0].strip() for regra in producoes}
    for nao_terminal in nao_terminais:
        FOLLOW[nao_terminal] = set()

    # Passo 2: Colocar o símbolo de fim de entrada ($) no FOLLOW do símbolo inicial
    FOLLOW[simbolo_inicial].add('$')

    # Passo 3: Iteração para todos os não-terminais
    mudou = True  # Flag para controlar se houve mudanças nos conjuntos FOLLOW
    while mudou:
        mudou = False  # Inicializa a flag como False no início de cada iteração
        for producao in producoes:  # Para cada produção na gramática
            partes = producao.split('\rightarrow')
            A = partes[0].strip()  # Não-terminal do lado esquerdo da produção
            alfa = partes[1].strip().split()  # Símbolos do lado direito da produção

            for i in range(len(alfa)):
                B = alfa[i]  # Símbolo atual na produção
                if B in nao_terminais:  # Se B é um não-terminal
                    beta = alfa[i+1:]  # Símbolos após B na produção
                    if beta:  # Se há símbolos após B
                        # Calcula FIRST(beta) e adiciona ao FOLLOW(B) exceto por \varepsilon
                        for simbolo in calcular_FIRST(['temp \rightarrow ' + ' '.join(beta)]).get('temp', []):
                            if simbolo != '\varepsilon' and simbolo not in FOLLOW[B]:
                                FOLLOW[B].add(simbolo)
                                mudou = True
                        # Se FIRST(beta) contém \varepsilon, adiciona FOLLOW(A) ao FOLLOW(B)
                        if '\varepsilon' in calcular_FIRST(['temp \rightarrow ' + ' '.join(beta)]).get('temp', []):
                            for simbolo in FOLLOW[A]:
                                if simbolo not in FOLLOW[B]:
                                    FOLLOW[B].add(simbolo)
                                    mudou = True
                    else:  # Se não há símbolos após B
                        # Adiciona FOLLOW(A) ao FOLLOW(B) diretamente
                        for simbolo in FOLLOW[A]:
                            if simbolo not in FOLLOW[B]:
                                FOLLOW[B].add(simbolo)
                                mudou = True
    return FOLLOW

# Exemplo de uso
producoes = [
    "S \rightarrow a B",
    "S \rightarrow b A",
    "A \rightarrow c",
    "A \rightarrow d",
    "B \rightarrow e",
    "B \rightarrow f"
]
simbolo_inicial = "S"

FOLLOW = calcular_FOLLOW(producoes, simbolo_inicial)
for nao_terminal in FOLLOW:
    print(f"FOLLOW({nao_terminal}) = {FOLLOW[nao_terminal]}")
```