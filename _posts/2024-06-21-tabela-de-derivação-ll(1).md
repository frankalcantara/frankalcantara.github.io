---
layout: post
title: Tabelas de Derivação (Análise) LL(1)
author: Frank
categories:
    - Matemática
    - Linguagens Formais
tags:
    - Matemática
    - Linguagens Formais
    - Compiladores
image: assets/images/deriva.webp
description: Como criar tabelas de análise para parsers ll(1) usando os conjuntos First e Follow.
slug: tabela-derivacao
keywords:
    - parsers
    - first
    - follow
    - derivation
    - derivação
rating: 5
published: 2024-06-21T19:41:42.444Z
draft: 2024-06-21T19:41:40.386Z
preview: Definição, processo de funcionamento e outras informações necessárias ao entendimento do funcionamento de parsers LL(1)
---

A Tabela de Derivação $LL(1)$, também chamada de tabela da análise LL(1), é uma ferramenta fundamental do funcionamento do *parser* $LL(1)$. Esta tabela será utilizada no processo de *parser* para verificar se um determinado *string* está de acordo com a gramática da linguagem. Esta tabela será usada pelo algoritmo de *parser* para determinar qual regra de produção deverá ser aplicada, considerando o símbolo de entrada corrente e o não-terminal no topo de uma pilha. Este par de símbolos, (terminal / não-terminal) será usado como índice da tabela e determinará qual regra de produção deverá ser utilizada, ou se existe uma inconsistência entre a *string* de entrada e a gramática. Já vimos [o que é um parser](https://frankalcantara.com/parsers-ll(1)/), e como criar [os conjuntos $FIRST$ e $FOLLOW$](https://frankalcantara.com/first-follow/). Agora vamos usar este conhecimento para criar a Tabela de Derivação.

Para construir a Tabela de Derivação $LL(1)$, acrescentaremos um $\$$ no final da *string* de entrada para indicar o seu término e, além disso, seguiremos três regras:

- Para cada terminal $a$ em $FIRST(\alpha)$, adicione a regra $A \to \alpha$ à célula $[A, a]$ da tabela.
- Se $\varepsilon$ está em $FIRST(\alpha)$, adicione a regra $A \to \alpha$ à tabela em $[A, b]$ para cada $b$ em $FOLLOW(A)$.
- Se $ \\$ $ está em $FOLLOW(A)$, adicione também $A \to \alpha$ à célula $[A, \\$]$.

Podemos retornar ao um [exemplo 1 de criação do conjunto $FIRST$](https://frankalcantara.com/first-follow/) e, a partir deste exemplo, criar a Tabela de Derivação correspondente a gramática dada naquele exemplo.

**Exemplo 1**: Considere a gramática definida pelo seguinte conjunto de regras de produção

$$
\begin{array}{cc}
1. & S \rightarrow aB \mid bA \\
2. & A \rightarrow c \mid d \\
3. & B \rightarrow e \mid f \\
\end{array}
$$

A partir deste conjunto de regras de produção podemos definir o seguinte conjunto $FIRST$:

$$
\begin{array}{ccl}
FIRST(S) & = & \{a, b\} \\
FIRST(A) & = & \{c, d\} \\
FIRST(B) & = & \{e, f\} \\
\end{array}
$$

E o conjunto $FOLLOW$ dado por:

$$
\begin{array}{ccl}
FOLLOW(S) & = & \{\$\} \\
FOLLOW(A) & = & \{\$, c, d\} \\
FOLLOW(B) & = & \{\$\} \\
\end{array}
$$

Com estes conjuntos podemos criar uma Tabela de Derivação se seguirmos as regras dadas acima teremos:

- Para $S \to aB$: Como $a$ está em $FIRST(aB)$, adicionamos $S \to aB$ em $[S, a]$.
- Para $S \to bA$: Como $b$ está em $FIRST(bA)$, adicionamos $S \to bA$ em $[S, b]$.
- Para $A \to c$: Como $c$ está em $FIRST(c)$, adicionamos $A \to c$ em $[A, c]$.
- Para $A \to d$: Como $d$ está em $FIRST(d)$, adicionamos $A \to d$ em $[A, d]$.
- Para $B \to e$: Como $e$ está em $FIRST(e)$, adicionamos $B \to e$ em $[B, e]$.
- Para $B \to f$: Como $f$ está em $FIRST(f)$, adicionamos $B \to f$ em $[B, f]$.

O que permite criar a seguinte Tabela de Derivação:

<table border="1">
  <tr>
    <th></th>
    <th>a</th>
    <th>b</th>
    <th>c</th>
    <th>d</th>
    <th>e</th>
    <th>f</th>
    <th>$</th>
  </tr>
  <tr>
    <td>S</td>
    <td>S &rarr; aB</td>
    <td>S &rarr; bA</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>A</td>
    <td></td>
    <td></td>
    <td>A &rarr; c</td>
    <td>A &rarr; d</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>B</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>B &rarr; e</td>
    <td>B &rarr; f</td>
    <td></td>
  </tr>
</table>

Este exemplo é perfeito, para cada par terminal / não-terminal existe apenas uma regra de produção. Infelizmente, quando estamos construindo linguagens livres de contexto, este não é o cenário mais comum.

## Conflitos na Tabela de Derivação

Conflitos na Tabela de Derivação $LL(1)$ ocorrem quando há mais de uma regra de produção associada a um mesmo par indicador (terminal, não-terminal). Esta ambiguidade significa que, ao encontrar esse par, o *parser* $LL(1)$ não conseguirá determinar, de forma única e inequívoca, qual regra deverá aplicar a um determinado símbolo de entrada, tornando a gramática ambígua e inadequada para uso com *parsers* $LL(1)$.

**Exemplo 2**: o conflito. Observe que gramática a seguir foi criada para criar um conflito na Tabela de Derivação. Antes de nos preocuparmos com os tipos de conflito, e como solucioná-los, vamos rever todo o processo de criação de  uma Tabela de Derivação para entender o problema.

$$
\begin{aligned}
1. &\ E \rightarrow T + E \ \vert \ T \\
2. &\ T \rightarrow int \ \vert \ (E)
\end{aligned}
$$

### Conjunto $FIRST$

1. **Para o não-terminal E:**

   - $E \rightarrow T + E$: o primeiro símbolo é $T$. Portanto, incluímos $FIRST(T)$ em $FIRST(E)$.
   - $E \rightarrow T$: o primeiro símbolo é $T$. Portanto, incluímos $FIRST(T)$ em $FIRST(E)$.

2. **Para o não-terminal T:**

   - $T \rightarrow int$: o primeiro símbolo é $int$. Portanto, $FIRST(T)$ inclui $int$.
   - $T \rightarrow (E)$: o primeiro símbolo é $( $. Portanto, $FIRST(T)$ inclui $( $.

Assim, temos:

$$
\begin{aligned}
FIRST(T) &= \{ int, ( \} \\
FIRST(E) &= FIRST(T) = \{ int, ( \}
\end{aligned}
$$

### Conjunto $FOLLOW$

1. **Para o símbolo inicial E:**
   - $FOLLOW(E)$ inclui \$.

2. **Para as produções de E:**
   - $E \rightarrow T + E$:
     - o símbolo $E$ pode ser seguido por $+$, então $+$ está em $FOLLOW(T)$.
     - o símbolo $E$ é o último da produção, então $FOLLOW(E)$ inclui $FOLLOW(E)$.
   - $E \rightarrow T$: o símbolo $T$ é o último da produção, então $FOLLOW(T)$ inclui $FOLLOW(E)$.

3. **Para as produções de T:**
   - $T \rightarrow int$: $int$ é um terminal, não influencia $FOLLOW$.
   - $T \rightarrow (E)$: $E$ pode ser seguido por $)$, então $FOLLOW(E)$ inclui $)$.

Assim, teremos:

$$
\begin{aligned}
FOLLOW(E) &= \{ \$, +, ) \} \\
FOLLOW(T) &= \{ +, \$, ) \}
\end{aligned}
$$

### $Nullable$

Para determinar se algum não-terminal é *nullable*, verificamos se ele pode derivar a string vazia $\epsilon$.

1. **Para E:**
   - $E \rightarrow T + E$: $T$ não é *nullable*, portanto, $E$ não é *nullable* a partir desta produção.
   - $E \rightarrow T$: $T$ não é *nullable*, portanto, $E$ não é *nullable*.

2. **Para T:**
   - $T \rightarrow int$: $int$ não é *nullable*.
   - $T \rightarrow (E)$: $E$ não é *nullable* e $($ é um terminal.

Ou seja, nenhum dos não-terminais é *nullable*:

$$
\begin{aligned}
Nullable(E) &= false \\
Nullable(T) &= false
\end{aligned}
$$

### Resumo dos Conjuntos

$$
\begin{aligned}
FIRST(E) &= \{ int, ( \} \\
FIRST(T) &= \{ int, ( \} \\
FOLLOW(E) &= \{ \$, +, ) \} \\
FOLLOW(T) &= \{ +, \$, ) \} \\
Nullable(E) &= false \\
Nullable(T) &= false
\end{aligned}
$$

O que permite gerar a seguinte Tabela de Derivação:

<table>
  <tr>
    <th>não-terminal</th>
    <th>int</th>
    <th>(</th>
    <th>+</th>
    <th>$</th>
    <th>)</th>
  </tr>
  <tr>
    <td>E</td>
    <td>E &rarr; T</td>
    <td>E &rarr; T</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>T</td>
    <td>T &rarr; int</td>
    <td>T &rarr; (E)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

Nesta tabela podemos ver um conflito explícito. O não-terminal $E$ possui duas produções para os símbolos $int$ e $($.

Sempre que na criação de Tabelas de Derivação existir um conflito estaremos gerando ambiguidades na derivação. A gramática é ambígua, sempre que uma sentença puder ser derivada de duas ou mais formas diferentes, gerando árvores sintáticas diferentes. Por exemplo, na gramática do **Exemplo 2**, a sentença $int + int$ pode ser derivada tanto como $E → T → int$ seguido de $E → T + E → int + int$ quanto como $E → T + E → T + T → int + int$.

## Tipos de Conflitos

Conflitos na tabela $LL(1)$ podem ser classificados em dois tipos principais:

1. **Conflito $FIRST$/$FIRST$**: ocorre quando o conjunto $FIRST$ de duas ou mais produções de um mesmo não-terminal possui um terminal em comum. Na gramática do **Exemplo 2**, as produções $E \to T + E$ e $E \to T$ possuem os terminais $int$ e $($ em seus conjuntos $FIRST$. Ao encontrar $int$ ou $($ na entrada, o *parser* não sabe se deve aplicar a regra que deriva uma expressão com um operador $+$ ou a regra que deriva um termo. Neste momento do processo de *parsing* determinismo saiu pela janela e o *parser* $LL(1)$ é inútil.

2. **Conflito $FIRST$/$FOLLOW$**: ocorre quando uma produção tem $\varepsilon$ (a *string* vazia) em seu conjunto $FIRST$ e o conjunto $FOLLOW$ do não-terminal da produção possui um terminal em comum com o $FIRST$ de outra produção do mesmo não-terminal. Por exemplo, na gramática do **Exemplo 2**, se $E \rightarrow \varepsilon$ fosse uma produção e o conjunto $FOLLOW(E)$ contivesse $+$, que também está no $FIRST$ da produção $E \rightarrow T + E$, ao encontrar o fim de uma expressão (representado por um símbolo em $FOLLOW(E)$), o *parser* não saberia se deve aplicar a regra que deriva apenas um termo ou a regra que deriva uma expressão com um operador $+$.

### Resolução de Conflitos

Conflitos na tabela $LL(1)$ podem ser resolvidos das seguintes formas:

1. **Refatoração da gramática**: A gramática pode ser reescrita para eliminar ambiguidades e recursões à esquerda, evitando assim os conflitos.
2. **Fatoração à esquerda**: Produções com prefixos comuns podem ser fatoradas para que a decisão entre elas possa ser tomada com base em um único símbolo de *lookahead*.
3. **Uso de analisadores mais poderosos**: Se os conflitos não puderem ser resolvidos na gramática, pode ser necessário usar um analisador sintático mais poderoso, como um analisador $LR(1)$ ou $LALR(1)$, que conseguem lidar com gramáticas mais complexas.

As soluções 1 e 2 implicam na modificação da sua gramática, o que ocorre com frequência quando começamos do zero. A solução 3, em linguagens complexas, pode ser a solução adequada, mas implica em mudar de algoritmo de *parser*

**Exemplo 3**: resolução de conflito. No exemplo da gramática anterior, o conflito $FIRST$/$FIRST$ pode ser resolvido fatorando as produções de $E$:

$$
\begin{aligned}
1. &\ E \rightarrow T E' \\
2. &\ E' \rightarrow + E \ \vert \ \epsilon \\
3. &\ T \rightarrow int \ \vert \ (E)
\end{aligned}
$$

Se calcularmos os conjuntos $FIRST$ e $FOLLOW$ novamente, teremos:

### Conjunto $FIRST$

1. **Para o não-terminal E:**
   - $E \rightarrow T E'$: o primeiro símbolo é $T$. Portanto, incluímos $FIRST(T)$ em $FIRST(E)$.

2. **Para o não-terminal E':**
   - $E' \rightarrow + E$: o primeiro símbolo é $+$. Portanto, $FIRST(E')$ inclui $+$.
   - $E' \rightarrow \epsilon$: incluímos $\epsilon$ em $FIRST(E')$.

3. **Para o não-terminal T:**
   - $T \rightarrow int$: o primeiro símbolo é $int$. Portanto, $FIRST(T)$ inclui $int$.
   - $T \rightarrow (E)$: o primeiro símbolo é $( $. Portanto, $FIRST(T)$ inclui $( $.

Assim, teremos:

$$
\begin{aligned}
FIRST(T) &= \{ int, ( \} \\
FIRST(E') &= \{ +, \epsilon \} \\
FIRST(E) &= FIRST(T) = \{ int, ( \}
\end{aligned}
$$

### Conjunto $FOLLOW$

1. **Para o símbolo inicial E:**
   - $FOLLOW(E)$ inclui \$.

2. **Para as produções de E:**
   - $E \rightarrow T E'$:
     - O símbolo $E'$ pode ser seguido por $FOLLOW(E)$.
     - Então, $FOLLOW(E')$ inclui $FOLLOW(E)$.

3. **Para as produções de E':**
   - $E' \rightarrow + E$:
     - O símbolo $E$ pode ser seguido por $FOLLOW(E')$.
     - Então, $FOLLOW(E)$ inclui $FOLLOW(E')$.
   - $E' \rightarrow \epsilon$: não há efeito em $FOLLOW$.

4. **Para as produções de T:**
   - $T \rightarrow int$: $int$ é um terminal, não influencia $FOLLOW$.
   - $T \rightarrow (E)$: $E$ pode ser seguido por $)$, então $FOLLOW(E)$ inclui $)$.

Assim, teremos:

$$
\begin{aligned}
FOLLOW(E) &= \{ \$, ) \} \\
FOLLOW(E') &= \{ \$, ) \} \\
FOLLOW(T) &= \{ +, \$, ) \}
\end{aligned}
$$

### $Nullable$

Para determinar se algum não-terminal é *nullable*, verificamos se ele pode derivar a string vazia $\epsilon$.

1. **Para E:**
   - $E \rightarrow T E'$: $T$ não é *nullable*, portanto, $E$ não é *nullable*.

2. **Para E':**
   - $E' \rightarrow + E$: $+$ é um terminal, portanto, não é *nullable*.
   - $E' \rightarrow \epsilon$: $E'$ é *nullable*.

3. **Para T:**
   - $T \rightarrow int$: $int$ não é *nullable*.
   - $T \rightarrow (E)$: $( $ é um terminal, portanto, não é *nullable*.

Assim, teremos:

$$
\begin{aligned}
Nullable(E) &= false \\
Nullable(E') &= true \\
Nullable(T) &= false
\end{aligned}
$$

### Resumo dos Conjuntos

$$
\begin{aligned}
FIRST(E) &= \{ int, ( \} \\
FIRST(E') &= \{ +, \epsilon \} \\
FIRST(T) &= \{ int, ( \} \\
FOLLOW(E) &= \{ \$, ) \} \\
FOLLOW(E') &= \{ \$, ) \} \\
FOLLOW(T) &= \{ +, \$, ) \} \\
Nullable(E) &= false \\
Nullable(E') &= true \\
Nullable(T) &= false
\end{aligned}
$$

O que permite gerar a seguinte Tabela de Derivação:

<table>
  <tr>
    <th>não-terminal</th>
    <th>int</th>
    <th>(</th>
    <th>+</th>
    <th>$</th>
    <th>)</th>
  </tr>
  <tr>
    <td>E</td>
    <td>E \to T E'</td>
    <td>E \to T E'</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>E'</td>
    <td></td>
    <td></td>
    <td>E' \to + E</td>
    <td>E' \to ε</td>
    <td>E' \to ε</td>
  </tr>
  <tr>
    <td>T</td>
    <td>T \to int</td>
    <td>T \to (E)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

Agora, a decisão entre derivar uma expressão com um operador $+$ ou apenas um termo pode ser tomada com base no próximo símbolo da entrada: se for $+$, aplica-se a regra $E' \rightarrow + E$; caso contrário, aplica-se a regra $E' \rightarrow \varepsilon$.

Assim como fiz nos artigos anteriores, vou sugerir um pseudocódigo, um tanto inocente, para a criação de tabelas de derivação. Acredito que, com um pouco de cuidado, depois que a amável leitora dominar esta técnica possa criar um pseudocódigo mais eficiente. A fé move montanhas.

## Pseudocódigo para a criação da Tabela de Derivação

```Shell

**Entrada:**
- Conjuntos de produções da gramática: `Productions`
- Conjuntos FIRST: `First`
- Conjuntos FOLLOW: `Follow`
- Conjuntos Nullable: `Nullable`
- Conjunto de não-terminais: `NonTerminals`
- Conjunto de terminais: `Terminals`

**Saída:**
- Tabela de Derivação `$LL(1)$`: `ParsingTable`

# Inicialização da Tabela de Derivação $LL(1)$
ParsingTable = {A: {a: None for a in Terminals + ['$']} for A in NonTerminals}

# Para cada produção A \rightarrow \alpha
for A, productions in Productions.items():
    for \alpha in productions:
        # Para cada terminal a em FIRST(\alpha)
        for a in First[\alpha]:
            if a != '\varepsilon':
                # Adicionar a produção A \rightarrow \alpha na célula [A, a]
                if ParsingTable[A][a] is None:
                    ParsingTable[A][a] = A + " \rightarrow " + \alpha
                else:
                    raise ConflictError(f"Conflito na tabela para [{A}, {a}]")

        # Se \varepsilon está em FIRST(\alpha)
        if '\varepsilon' in First[\alpha]:
            # Para cada b em FOLLOW(A)
            for b in Follow[A]:
                # Adicionar a produção A \rightarrow \alpha na célula [A, b]
                if ParsingTable[A][b] is None:
                    ParsingTable[A][b] = A + " \rightarrow " + \alpha
                else:
                    raise ConflictError(f"Conflito na tabela para [{A}, {b}]")

            # Se $ está em FOLLOW(A)
            if '$' in Follow[A]:
                # Adicionar a produção A \rightarrow \alpha na célula [A, $]
                if ParsingTable[A]['$'] is None:
                    ParsingTable[A]['$'] = A + " \rightarrow " + \alpha
                else:
                    raise ConflictError(f"Conflito na tabela para [{A}, $]")

# Função para calcular FIRST de uma string
def compute_first(\alpha):
    if \alpha == '':
        return {'\varepsilon'}
    first_set = set()
    for symbol in \alpha:
        first_set.update(First[symbol] - {'\varepsilon'})
        if '\varepsilon' not in First[symbol]:
            break
    else:
        first_set.add('\varepsilon')
    return first_set
```

Caramba! Só notei agora, eu praticamente escrevo em python quando estou fazendo pseudocódigos. Isto deve ser mal, muito mal... Acho que vou ficar sem sorvete hoje. Em fim, este código pode ser implementado em python por:

```python
# Conjuntos de produções
productions = {
    'E': ['T+E', 'T'],
    'T': ['int', '(E)']
}

# Conjuntos FIRST
first = {
    'E': {'int', '('},
    'T': {'int', '('},
    'T+E': {'int', '('},
    'int': {'int'},
    '(E)': {'('}
}

# Conjuntos FOLLOW
follow = {
    'E': {'$', ')'},
    'T': {'+', '$', ')'}
}

# Conjuntos Nullable
nullable = {
    'E': False,
    'T': False
}

# Conjunto de não-terminais
non_terminals = ['E', 'T']

# Conjunto de terminais
terminals = ['int', '(', ')', '+', '$']

# Inicialização da Tabela de Derivação $LL(1)$
parsing_table = {A: {a: None for a in terminals} for A in non_terminals}

# Função para calcular FIRST de uma string
def compute_first(\alpha):
    if \alpha == '':
        return {'\varepsilon'}
    first_set = set()
    i = 0
    while i < len(\alpha):
        symbol = \alpha[i]
        if symbol in first:
            first_set.update(first[symbol] - {'\varepsilon'})
            if '\varepsilon' not in first[symbol]:
                break
        else:
            # Adiciona todo o terminal se não for um não-terminal
            terminal = ''
            while i < len(\alpha) and (\alpha[i].isalnum() or \alpha[i] in ['_', '+', '(', ')']):
                terminal += \alpha[i]
                i += 1
            first_set.add(terminal)
            if terminal in first and '\varepsilon' in first[terminal]:
                continue
            break
        i += 1
    else:
        first_set.add('\varepsilon')
    return first_set

# Preenchimento da Tabela de Derivação
for A in productions:
    for \alpha in productions[A]:
        first_\alpha = compute_first(\alpha)

        # Para cada terminal em FIRST(\alpha)
        for a in first_\alpha:
            if a != '\varepsilon':
                if a in parsing_table[A]:
                    if parsing_table[A][a] is None:
                        parsing_table[A][a] = f"{A} \rightarrow {\alpha}"
                    else:
                        print(f"Conflito na tabela para [{A}, {a}]")
                else:
                    print(f"Terminal {a} não reconhecido na tabela")

        # Se \varepsilon está em FIRST(\alpha)
        if '\varepsilon' in first_\alpha:
            for b in follow[A]:
                if b in parsing_table[A]:
                    if parsing_table[A][b] is None:
                        parsing_table[A][b] = f"{A} \rightarrow {\alpha}"
                    else:
                        print(f"Conflito na tabela para [{A}, {b}]")
                else:
                    print(f"Terminal {b} não reconhecido na tabela")

# Exibir a Tabela de Derivação de forma legível
import pandas as pd

# Converte os valores None para strings vazias
formatted_parsing_table = {A: {a: (parsing_table[A][a] if parsing_table[A][a] is not None else '') for a in terminals} for A in non_terminals}

# Cria o DataFrame
df = pd.DataFrame(formatted_parsing_table).T

# Reordena as colunas para incluir o símbolo de final de string '$'
df = df[terminals]

# Calcula a largura máxima de cada coluna (incluindo o cabeçalho)
col_widths = {col: max(df[col].astype(str).str.len().max(), len(col)) for col in df.columns}

# Estiliza o DataFrame para uma melhor visualização com bordas, linhas zebradas e ajuste de largura
styled_df = df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black'}).set_table_styles(
    [dict(selector='th', props=[('text-align', 'center'), ('border', '1px solid black')])]
).set_caption("Tabela de Parsing $LL(1)$")

# Define o estilo das células vazias como cinza claro
styled_df.set_properties(subset=pd.IndexSlice[:, :], **{'background-color': 'lightgrey' if v == '' else '' for v in df.values.flatten()})

# Adiciona o estilo zebrado
styled_df.set_table_styles([
    {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'lightblue')]},
    {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]}
])

# Ajusta a largura das colunas
styled_df.set_properties(subset=pd.IndexSlice[:, :], **{f'width': f'{col_widths[col]*8}px' for col in df.columns})

from IPython.display import display

display(styled_df)
```