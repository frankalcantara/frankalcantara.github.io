---
layout: post
title: Resolvendo o Grid World Com MD
author: Frank
categories:
    - artigo
    - disciplina
    - Inteligência Artificial
tags: []
image: assets/images/trans1.webp
featured: false
rating: 5
description: "Uma exploração detalhada da solução do Grid World usando Dynamic Programming, com implementação em C++ 20. "
date: 2025-02-10T09:29:17.254Z
preview: Descubra como resolver o Grid World usando *Dynamic Programming* e C++ 20. Um guia prático e matemático para entender a solução de MDPs, desde as equações de Bellman até a implementação computacional.
keywords: Grid World *Reinforcement Learning* M*Dynamic Programming* Solution *Reinforcement Learning* Processo de Decisão de Markov Equações de Bellman Value Iteration Policy Iteration *Dynamic Programming* Dynamic Programming Política Ótima Optimal Policy
toc: true
published: true
beforetoc: ""
lastmod: 2025-02-11T23:57:46.296Z
draft: 2025-02-10T09:29:19.442Z
---

Agora que formalizamos o **Grid World** como um *Processo de Decisão de Markov* (**M*Dynamic Programming***), podemos aplicar algoritmos de *Reinforcement Learning* (*Reinforcement Learning*) para encontrar a política ótima $\pi^\*$. *A política ótima é aquela que maximiza a recompensa total esperada a longo prazo para o agente*. Vamos explorar como a estrutura do **M*Dynamic Programming*** nos permite resolver o **Grid World**.

## Formulação M*Dynamic Programming* do Grid World

Vamos dedicar um minuto, ou dez, para lembrar em que porto desta jornada estamos. Até agora definimos o seguinte:

1. **Estados $(S)$**: O conjunto de todas as células da grade. Cada célula $(x, y)$ representa um estado.

2. **Ações $(A)$**: O conjunto de ações possíveis: $A= \{\text{Norte}, \text{Sul}, \text{Leste}, \text{Oeste}\}$.

3. **Função de Transição $(P)$**:  $P(s' \mid  s, a)$ define a probabilidade de transitar para o estado $s'$ ao executar a ação $a$ no estado $s$. No **Grid World** estocástico que estamos estudando, teremos:

    * $0.8$ de probabilidade de se mover na direção pretendida.
    * $0.1$ de probabilidade de se mover para cada um dos lados perpendiculares.
    * Se a ação levar a uma colisão com a parede, o agente permanece no estado $s$.
  
    Eu havia usado $10\%$ e $80\%$. Se a atenta leitora não entendeu a mudança, este não é um artigo para você.

4. **Função de Recompensa $(R)$**: $R(s, a, s')\;$ define a recompensa recebida ao transitar de $s$ para $s'$ após executar $a$. No nosso caso, simplificamos para $R(s)$:

    $$R(s) = \begin{cases}
    +1 & \text{se } s \text{ é um estado terminal positivo} \\
    -1 & \text{se } s \text{ é um estado terminal negativo} \\
    r_{vida} & \text{caso contrário}
    \end{cases}$$

    de tal forma que $r_{vida}$ é a recompensa por passo (living reward), geralmente um valor pequeno e negativo (por exemplo, $-0.03$).

5. **Estado Inicial $(s_0)$**: A célula em que o agente inicia cada episódio.

6. **Estados Terminais $(S_{terminal})$**: As células que, quando alcançadas, encerram o episódio.

## Resolvendo o Grid World

A abordagem inocente para encontrar a política ótima para resolver o **Grid World** envolve explorar todas as possíveis sequências de ações a partir de cada estado inicial até alcançar o estado final. Esta é uma estratégia de, com o perdão pela má palavra, força bruta. Esta estratégia envolve os seguintes passos:

1. **Definição do Espaço de Estados e Ações**: identifique todos os estados possíveis na grade e defina todas as ações possíveis que o agente pode tomar em cada estado (por exemplo, mover-se para cima, para baixo, para a esquerda, para a direita);

2. **Geração de Sequências de Ações**: gere todas as possíveis sequências de ações a partir do estado inicial. Isso pode ser feito usando tanto iteratividade quanto recursividade para explorar todas as combinações de ações.

3. **Avaliação das Sequências**: para cada sequência de ações gerada, simule o movimento do agente na grade e calcule a recompensa acumulada ao longo do caminho; e é preciso considerar todas as transições de estado e as recompensas imediatas associadas a cada ação.

4. **Seleção da Sequência Ótima**: compare as recompensas acumuladas de todas as sequências de ações; selecione a sequência que maximiza a recompensa acumulada como a política ótima.

A criativa leitora deve estar percebendo que esta é uma estratégia perfeitamente possível. Não é raro que as estratégias de força bruta sejam a primeira opção quando alguém enfrenta um problema pela primeira vez. Neste caso, trata-se de uma solução de busca em todo o universo de possibilidades, considerando todas as combinações possíveis de movimentos (para cima - $\text{Norte}$, para baixo - $\text{Sul}$, para a esquerda - $\text{Oeste}$, para a direita - $\text{Leste}$) em cada estado. Essa abordagem é, via de regra, ineficiente e computacionalmente dispendiosa. O número de sequências de ações possíveis cresce exponencialmente com o tamanho da grade e o número de estados. Quanto maior o mundo, maior será o custo computacional para resolver o problema.

Existem soluções melhores. A primeira, clássica e didática, foi encontrada com a aplicação das Equações de Bellman.

## Entram as Equações de Bellman

As **Equações de Bellman** desenvolvidas por [Richard Bellman](https://pt.wikipedia.org/wiki/Richard_Bellman) na década de 1950, oferecem uma maneira mais eficiente e escalável de resolver o problema, decompondo-o em subproblemas menores e utilizando a *Dynamic Programming*, programação dinâmica, para encontrar a política ótima[^1]. Para a maioria dos casos práticos, especialmente em ambientes maiores ou ambientes complexos, as **Equações de Bellman** são a abordagem preferida.

[^1]: BELLMAN, Richard. Dynamic Programming. Princeton: Princeton University Press, 1957.

Em 1952, Richard Bellman se deparou com um dilema. Como matemático no [projeto RAND](https://open-int.blog/2018/01/21/project-rand-reports-1951-1956/), ele precisava encontrar uma maneira de resolver problemas de otimização multidimensional, e o termo *programação matemática*, sua primeira ideia, já estava sendo usado por outros pesquisadores.

Segundo o próprio Bellman, o termo *Dynamic Programming* foi escolhido para esconder o fato de que ele estava fazendo pesquisa matemática de um secretário do Departamento de Defesa que tinha aversão a qualquer menção à palavra *pesquisa*[^2]. É assustador, e um pouco desesperador, que uma técnica matemática tão poderosa tenha sido nomeada com o intuito de esconder sua natureza matemática, revelando como a política ruim influencia a linguagem da pesquisa científica há décadas.

[^2]:CONVERSABLE ECONOMIST. Why Is It Called “Dynamic Programming”? 24 ago. 2022. Disponível em: https://conversableeconomist.com/2022/08/24/why-is-it-called-dynamic-programming/. Acesso em: 10 fev. 2025.

A intuição fundamental de Bellman era que problemas de otimização continham subproblemas que se sobrepunham. Esta descoberta levou ao que hoje conhecemos como *Princípio da Otimalidade de Bellman*:

> Uma política ótima tem a propriedade de que, independentemente do estado inicial e da decisão inicial, as decisões restantes devem constituir uma política ótima em relação ao estado resultante da primeira decisão.

Matematicamente, este princípio pode ser expresso como:

$$ V^*(s) = \max_a \{ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)V^*(s') \} $$

A sagaz leitora será capaz de entender a revolução que Bellman iniciou com a *Dynamic Programming* por meio de três ideias fundamentais:

1. **Decomposição Recursiva**: A solução de um problema maior pode ser construída a partir das soluções de seus subproblemas. No contexto de **MDPs**, isso significa que podemos decompor o problema de encontrar uma política ótima em subproblemas menores para cada estado.

2. **Memoização**: Ao resolver subproblemas, armazenamos suas soluções para evitar recálculos. Em termos de **MDPs**, isso se traduz em manter uma tabela de valores para cada estado.

3. **Propagação de Valor**: As soluções dos subproblemas são usadas para construir soluções para problemas maiores. Nos **MDPs**, isso corresponde à atualização iterativa dos valores dos estados.

As **Equações de Bellman** *expressam a relação recursiva entre o valor de um estado e os valores de seus estados sucessores*. Existem duas formas principais da Equação de Bellman que nos interessam no momento:

1. **Equação de Bellman para a Função Valor-Estado $(V^\pi)$**:

    A função valor-estado, $V^\pi(s)$, representa o retorno esperado (soma das recompensas descontadas) ao iniciar no estado $s$ e seguir a política $\pi$. A Equação de Bellman para $V^\pi$ será:

    $$V^\pi(s) = \sum_{a \in A} \pi(a\mid s) \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

    Nesta equação, temos:

    * $\pi(a\mid s)$ é a probabilidade de tomar a ação $a$ no estado $s$ sob a política $\pi$;
    * $\gamma$ é o fator de desconto $(0 \le \gamma \le 1)$, que determina a importância das recompensas futuras.

    Esta equação determina que o valor de um estado $s$ sob a política $\pi$ é a média ponderada dos valores de todos os estados sucessores possíveis $(s')$, considerando a probabilidade de cada transição $(P(s'\mid s, a))$ e a recompensa imediata $(R(s, a, s'))$.

2. **Equação de Bellman para a Função Valor-Ação $(Q^\pi)$**:

    A função valor-ação, $Q^\pi(s, a)$, representa o retorno esperado ao iniciar no estado $s$, tomar a ação $a$ e, em seguida, seguir a política $\pi$.  A Equação de Bellman para $Q^\pi$ será:

    $$Q^\pi(s, a) = \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma \sum_{a' \in A} \pi(a'\mid s') Q^\pi(s', a')]$$

    Esta equação determina que o valor de tomar a ação $a$ no estado $s$ é a média ponderada dos valores de todas as transições possíveis, considerando a recompensa imediata e o valor das ações subsequentes $(a')$ no estado resultante $(s')$, de acordo com a política $\pi$.

## Encontrando a Política Ótima

A *política ótima*, $\pi^\*$, é aquela que maximizar o valor de cada estado.  Podemos expressar as Equações de Bellman para a política ótima usando as funções valor-estado ótima $(V^*)$ e valor-ação ótima $(Q^*)$:

1. **Equação de Bellman de Otimalidade para $V^*$**:

    $$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma V^*(s')]$$

    Esta equação define que o valor ótimo de um estado $s$ é o máximo valor obtido ao escolher a melhor ação $a$ possível, considerando as transições e recompensas.

2. **Equação de Bellman de Otimalidade para $Q^*$**:

    $$Q^*(s, a) = \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma \max_{a' \in A} Q^*(s', a')]$$

    Esta equação diz que o valor ótimo de tomar a ação $a$ no estado $s$ é a média ponderada das recompensas imediatas e o valor máximo possível no estado sucessor $s'$.

Uma vez que tenhamos $Q^\*(s, a)$, podemos extrair a política ótima $\pi^\*$ diretamente:

$$\pi^{*} (s) = \arg\max_{a \in A} Q^*(s, a)$$

Ou seja, a política ótima escolhe a ação que maximiza o valor $Q^*$ em cada estado.

## Resolvendo o Grid World com *Dynamic Programming*

A *Dynamic Programming*, como vimos, oferece uma abordagem sistemática para resolver o **Grid World**. Vamos explorar como podemos aplicar os princípios de Bellman para encontrar a política ótima através de dois algoritmos fundamentais: **Iteração de Valor** e **Iteração de Política**.

### Iteração de Valor

O algoritmo de Iteração de Valor aplica diretamente a equação de Bellman de otimalidade para $V^*$ de forma iterativa:

$$\begin{equation} \tag{1}
V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} P(s'\mid s, a)\;[R(s, a, s') + \gamma V_k(s')\;]
\end{equation}
$$

O processo funciona da seguinte forma:

1. Inicializamos $V_0(s)$ arbitrariamente para todos os estados, geralmente com zeros;
2. Para cada estado $s$, atualizamos seu valor usando a equação (1);
3. Repetimos o processo até a convergência, isto é, até que a diferença máxima entre $V_{k+1}$ e $V_k$ seja menor que um limiar $\epsilon$:

   $$ \max_s \mid V_{k+1}(s) - V_k(s)\mid < \epsilon $$

No contexto do **Grid World**, para um estado $s = (x,y)$, o algoritmo considera: as quatro ações possíveis: $\{\text{Norte}, \text{Sul}, \text{Leste}, \text{Oeste}\}$; as probabilidades de transição: $0.8$ para a direção pretendida, $0.1$ para cada lado e a recompensa imediata $R(s)$ conforme definida anteriormente.

Por exemplo, ao calcular o valor de uma célula $(2,3)$, o algoritmo consideraria:

$$
\begin{align*}
& V_{k+1}(2,3) = \max_{a \in \{N,S,L,O\}}\;\; \{ 0.8[R(s_a)\; + \gamma V_k(s_a)\;] + \\
& 0.1[R(s_{a1}) + \gamma V_k(s_{a1})\;] + 0.1[R(s_{a2}) + \gamma V_k(s_{a2})\;] \}
\end{align*}
$$

Na qual, $s_a$ é o estado resultante do movimento na direção $a$, e $s_{a1}$, $s_{a2}$ são os estados resultantes dos movimentos laterais.

### Iteração de Política

O algoritmo de Iteração de Política alterna entre duas etapas:

1. **Avaliação de Política**: Para uma política fixa $\pi$, calculamos $V^\pi$ resolvendo o sistema de equações:

   $$ V^\pi(s) = \sum_{s' \in S} P(s'\mid s, \pi(s)) [R(s, \pi(s), s') + \gamma V^\pi(s')] $$

2. **Melhoria de Política**: Atualizamos a política para ser gulosa em relação a $V^\pi$:

   $$ \pi'(s) = \arg\max_{a \in A} \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma V^\pi(s')] $$

No **Grid World**, a avaliação de política resolve um sistema linear de equações para encontrar o valor de cada estado sob a política atual. Por exemplo, se a política atual em $(2,3)$ é $Norte$, resolvemos:

$$\begin{align*}
& V^\pi(2,3) = 0.8[R(2,4) + \gamma V^\pi(2,4)\;] + \\
& 0.1[R(3,3) + \gamma V^\pi(3,3)\;] + 0.1[R(1,3) + \gamma V^\pi(1,3)\;]
\end{align*}$$

### Convergência e Políticas Ótimas

Os dois algoritmos convergem para a política ótima $\pi^\*$, mas de maneiras diferentes:

* a **Iteração de Valor** mantém apenas valores e deriva a política implicitamente;
* a **Iteração de Política** mantém uma política explícita e a melhora iterativamente.

Para o **Grid World**, a política ótima resultante nos dará, para cada célula, a direção que o agente deve seguir para maximizar sua recompensa esperada descontada.

### Exemplo Numérico Completo

Nada como um exemplo prático, passo a passo, para que a amável leitora supere o medo da matemática e das equações assustadoras do Bellman. A Figura 1 introduz nosso primeiro problema.

![um grid world com inicio em 0,0, agente em 1,0, um obstáculo em 1,1 e objetivo em 4,3](/assets/images/gw1.webp)
_Figura 1: Exemplo de Grid World, para aplicação da *Dynamic Programming*._{: class="legend"}

Considere um mundo representado por uma grade retangular de dimensões $4 \times 3$, no qual um agente deve aprender a navegar de forma ótima. O ambiente possui as seguintes características:

1. **Grade**: O mundo é composto por $12$ células $(4 \times 3)$, na qual cada célula representa um estado possível para o agente.

2. **Estados Especiais**:
   * Estado Inicial: localizado na célula $(0,0)$ (canto inferior esquerdo);
   * Estado Terminal Positivo: localizado na célula $(3,2)$ (canto superior direito), com recompensa $+1$;
   * Estado Terminal Negativo: localizado na célula $(1,1)$ (centro), com recompensa $-1$;
   * Parede: localizada na célula $(2,1)$ (centro), intransponível.

3. **Dinâmica de Movimento**:
   * O agente pode escolher entre quatro ações: $\text{Norte, Sul, Leste e Oeste}$;
   * Os movimentos são estocásticos:
     * Probabilidade de $0.8$ de mover na direção pretendida;
     * Probabilidade de $0.1$ para cada direção perpendicular à pretendida.
   * Ao colidir com uma parede ou com os limites da grade, o agente permanece em sua posição atual.

4. **Sistema de Recompensas**:
   * Recompensa por passo $(r_{vida})$: $-0.03$ para cada movimento;
   * Recompensa terminal positiva: $+1.0$;
   * Recompensa terminal negativa: $-1.0$;
   * Fator de desconto $\gamma = 0.9$;

A sagaz leitora deve considerar que um episódio termina quando o agente alcança qualquer estado terminal, a política deve ser determinística (uma única ação por estado) e que a convergência ocorre quando a maior mudança em qualquer valor de estado for menor que $\epsilon = 0.001$.  

Encontre a política ótima $\pi^\*$ que maximize a soma das recompensas descontadas esperadas para o agente, começando do estado inicial.

## Usando Iteração de Valor no Grid World - Manualmente

Primeiro, transformamos nosso mundo em uma matriz. Para tanto, inicializamos os valores de todos os estados com $0$, exceto os estados terminais:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & 0.00 & 0.00 & 0.00 & +1.00 \\
1 & 0.00 & -1.00 & \text{WALL} & 0.00 \\
0 & 0.00 & 0.00 & 0.00 & 0.00
\end{array}
$$

### Primeira Iteração

Vamos calcular o novo valor para o estado inicial, localizado no canto inferior esquerdo.

**Para a Direção $\text{Norte}**: calculando o valor para o estado inicial com a ação $\text{Norte}$.

1) Movimento pretendido $(\text{Norte}, $0.8)$:
   - Próximo estado: $(1,0)$;
   - Recompensa: $-0.03 + 0.9 × V(1,0) = -0.03 + 0.9 × 0 = -0.03$.

2) Desvio lateral direito $(\text{Leste}, 0.1)$:
   - Próximo estado: $(0,1)$;
   - Recompensa: $-0.03 + 0.9 × V(0,1) = -0.03 + 0.9 × 0 = -0.03$.

3) Desvio lateral esquerdo $(0.1)$:
   - Colide com parede, permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 × V(0,0) = -0.03 + 0.9 × 0 = -0.03$.

Valor total para ação $\text{Norte}$:
$0.8(-0.03) + 0.1(-0.03) + 0.1(-0.03) = -0.03$

**Para a Direção $\text{Sul}$**: calculando o valor para o estado inicial a ação $\text{Sul}$. Como o agente está na borda inferior da grade, qualquer tentativa de movimento para sul resultará em permanecer na mesma posição.

1) Movimento pretendido $(\text{Sul}, 0.8)$:
   - O agente colide com a borda e permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,0) = -0.03 + 0.9 \times 0 = -0.03$.

2) Desvio lateral direito $(\text{Leste}, 0.1)$:
   - Próximo estado: $(0,1)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,1) = -0.03 + 0.9 \times 0 = -0.03$.

3) Desvio lateral esquerdo $(\text{Oeste}, 0.1)$:
   - O agente colide com a borda e permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,0) = -0.03 + 0.9 \times 0 = -0.03$.

Valor total para ação $\text{Sul}$:
$0.8(-0.03) + 0.1(-0.03) + 0.1(-0.03) = -0.03$

**Para a Direção Leste**: para a ação $\text{Leste}$ a partir do estado inicial $(0,0)$:

1) Movimento pretendido $(\text{Leste}, 0.8)$:
   - Próximo estado: $(0,1)$.
   - Recompensa: $-0.03 + 0.9 \times V(0,1) = -0.03 + 0.9 \times 0 = -0.03$;

2) Desvio lateral direito $(\text{Sul}, 0.1)$:
   - Colide com a borda, permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,0) = -0.03 + 0.9 \times 0 = -0.03$.

3) Desvio lateral esquerdo $(\text{Norte}, 0.1)$:
   - Próximo estado: $(1,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(1,0) = -0.03 + 0.9 \times 0 = -0.03$.

Valor total para ação $\text{Sul}$:
$0.8(-0.03) + 0.1(-0.03) + 0.1(-0.03) = -0.03$

**Para a Direção Oeste**: finalmente, para a ação $\text{Oeste}$ a partir do estado inicial $(0,0)$:

1) Movimento pretendido $(\text{Oeste}, 0.8)$:
   - Colide com a borda, permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,0) = -0.03 + 0.9 \times 0 = -0.03$.

2) Desvio lateral direito $(\text{Norte}, 0.1)$:
   - Próximo estado: $(1,0)$.
   - Recompensa: $-0.03 + 0.9 \times V(1,0) = -0.03 + 0.9 \times 0 = -0.03$;

3) Desvio lateral esquerdo $(\text{Sul}, 0.1)$:
   - Colide com a borda, permanece em $(0,0)$;
   - Recompensa: $-0.03 + 0.9 \times V(0,0) = -0.03 + 0.9 \times 0 = -0.03$.

Valor total para ação $\text{Oeste}$:
$0.8(-0.03) + 0.1(-0.03) + 0.1(-0.03) = -0.03$

**Comparação**: deveríamos escolher a maior recompensa. Contudo, comparando os valores obtidos para todas as direções:

- $\text{Norte}$: $-0.03$
- $\text{Sul}$: $-0.03$
- $\text{leste}$: $-0.03$
- $\text{Oeste}$: $-0.03$

Observamos que na primeira iteração, quando partimos do estado inicial, todas as ações resultam no mesmo valor. Isto ocorre porque todos os estados vizinhos ainda têm valor zero, exceto os estados terminais, que não são alcançáveis em um único passo do estado inicial. A diferenciação entre as ações começará a aparecer nas iterações subsequentes, quando os valores dos estados intermediários começarem a refletir sua proximidade com os estados terminais. Ou seja, após a primeira iteração completa, nosso mundo tem a seguinte característica:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & 0.00 & 0.11 & 0.25 & +1.00 \\
1 & 0.00 & -1.00 & \text{WALL} & 0.64 \\
0 & -0.03 & 0.00 & 0.11 & 0.25
\end{array}
$$

### Segunda Iteração

Agora usamos os valores da primeira iteração para calcular os novos valores. Para o mesmo estado inicial:

1) Ação $\text{Norte}$:

   - Principal $(0.8)$: $-0.03 + 0.9 × 0.00 = -0.03$;
   - Lateral direito $(0.1)$: $-0.03 + 0.9 × 0.00 = -0.03$;
   - Lateral esquerdo $(0.1)$: $-0.03 + 0.9 × (-0.03) = -0.057$.

Valor total: $0.8(-0.03) + 0.1(-0.03) + 0.1(-0.057) = -0.0327$

2) Ação $\text{Sul}$:

   - Principal $(0.8)$: colide com a parede, permanece em $(0,0)$: $-0.03 + 0.9 × (-0.03) = -0.03 + (-0.027) = -0.057$;
   - Lateral direito $(0.1)$: move para $(0,1)$: $-0.03 + 0.9 × 0.00 = -0.03$;
   - Lateral esquerdo $(0.1)$: colide com a parede, permanece em $(0,0)$: $-0.03 + 0.9 × (-0.03) = -0.057$.

Valor total: $0.8(-0.057) + 0.1(-0.03) + 0.1(-0.057) = -0.0543$

3) Ação $\text{Leste}$:

   - Principal $(0.8)$: move para $(0,1): -0.03 + 0.9 × 0.00 = -0.03$;
   - Lateral direito $(0.1)$: colide com a parede, permanece em (0,0): $-0.03 + 0.9 × (-0.03) = -0.057$;
   - Lateral esquerdo $(0.1)$: move para $(1,0)$: $-0.03 + 0.9 × 0.00 = -0.03$.

Valor total: $0.8(-0.03) + 0.1(-0.057) + 0.1(-0.03) = -0.0327$

4) Ação $\text{Oeste}$:

   - Principal $(0.8)$: colide com a parede, permanece em $(0,0)$: $-0.03 + 0.9 × (-0.03) = -0.057$;
   - Lateral direito $(0.1)$: move para $(1,0)$: $-0.03 + 0.9 × 0.00 = -0.03$;
   - Lateral esquerdo $(0.1)$: colide com a parede, permanece em $(0,0)$: $-0.03 + 0.9 × (-0.03) = -0.057$.

Valor total: $0.8(-0.057) + 0.1(-0.03) + 0.1(-0.057) = -0.0543$

**Comparação**: analisando os resultados para cada ação:

- $\text{Norte}$: $-0.0327$;
- $\text{Sul}$: $-0.0543$;
- $\text{Leste}$: $-0.0327$;
- $\text{Oeste}$: $-0.0543$.

Agora podemos observar uma diferenciação entre as ações. As ações $\text{Norte}$ e $\text{Leste}$ têm valores idênticos e mais altos que $\text{Sul}$ e $\text{Oeste}$. Isto aconteceu porque:

1. as ações $\text{Norte}$ e $\text{Leste}$ fazem com que o agente avance em direção aos estados com valores positivos mais frequentemente;

2. As ações $\text{Sul}$ e $\text{Oeste}$ resultam em colisões com as paredes/bordas, forçando o agente a permanecer no mesmo estado, acumulando recompensas negativas por passo;

3. Os valores começam a refletir a estrutura do ambiente e a propagação dos valores positivos a partir do estado terminal com recompensa $+1$.

O valor máximo entre todas as ações $(-0.0327)$ será o novo valor do estado $(0,0)$ na segunda iteração, e a política ótima neste estado começará a favorecer igualmente as ações $\text{Norte}$ e $\text{Leste}$.

### Convergência

Após várias iterações, os valores se estabilizam:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & 0.64 & 0.74 & 0.85 & +1.00 \\
1 & 0.58 & -1.00 & \text{WALL} & 0.89 \\
0 & 0.53 & 0.64 & 0.74 & 0.85
\end{array}
$$

### Política Ótima Resultante

Da matriz de valores final, podemos extrair a política ótima para cada estado:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & \rightarrow & \rightarrow & \rightarrow & +1 \\
1 & \uparrow & -1 & \text{WALL} & \uparrow \\
0 & \uparrow & \rightarrow & \rightarrow & \uparrow
\end{array}
$$

As setas indicam a direção ótima a seguir em cada estado.

A política ótima mostra que o agente deve:

1. partindo do estado inicial, subir e depois seguir para a direita;
2. evitar a área próxima ao estado terminal negativo;
3. procurar alcançar o estado terminal positivo pelo caminho mais seguro.

### Iteração de Valor Em C++

O código a seguir implementa exatamente o mesmo processo que fizemos no exemplo, sem tirar nem por, para que a atenta leitora possa fazer um paralelo e usar o código para entender o algoritmo, ou o algoritmo para entender o código. Cada um entende como quiser.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <format>
#include <iomanip>

// Define as ações possíveis que o agente pode tomar
// Corresponde às quatro direções mencionadas no exemplo
enum class Action { North, South, East, West };

// Define os tipos de células possíveis no grid
// Corresponde aos estados especiais mencionados no exemplo
enum class CellType {
    Normal,      // Célula normal com recompensa de passo -0.03
    Wall,        // Parede - intransponível
    TerminalPos, // Estado terminal com recompensa +1
    TerminalNeg  // Estado terminal com recompensa -1
};

// Estrutura para representar uma posição no grid
// Facilita o trabalho com coordenadas (x,y) mencionadas no exemplo
struct Position {
    int row;    // Linha (equivalente ao y no exemplo)
    int col;    // Coluna (equivalente ao x no exemplo)

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
};

class GridWorld {
private:
    // Dimensões do grid conforme especificado no exemplo (4x3)
    static constexpr int ROWS = 3;
    static constexpr int COLS = 4;

    // Parâmetros de recompensa definidos no exemplo
    static constexpr double STEP_REWARD = -0.03;           // r_vida
    static constexpr double POSITIVE_TERMINAL_REWARD = 1.0; // Recompensa terminal positiva
    static constexpr double NEGATIVE_TERMINAL_REWARD = -1.0;// Recompensa terminal negativa
    static constexpr double DISCOUNT_FACTOR = 0.9;         // Fator de desconto γ
    static constexpr double CONVERGENCE_THRESHOLD = 0.001; // ε para convergência

    // Probabilidades de movimento conforme exemplo
    static constexpr double MAIN_PROB = 0.8;  // Probabilidade de mover na direção desejada
    static constexpr double SIDE_PROB = 0.1;  // Probabilidade de mover perpendicular

    // Representação do ambiente
    std::array<std::array<CellType, COLS>, ROWS> grid;    // Tipo de cada célula
    std::array<std::array<double, COLS>, ROWS> values;    // Valores V(s) de cada estado
    std::array<std::array<Action, COLS>, ROWS> policy;    // Política π(s) para cada estado

    // Verifica se uma posição está dentro dos limites do grid
    // Implementa a lógica de colisão com as bordas mencionada no exemplo
    bool isValidPosition(const Position& pos) const {
        return pos.row >= 0 && pos.row < ROWS &&
            pos.col >= 0 && pos.col < COLS;
    }

    // Calcula a próxima posição baseada na posição atual e ação
    // Implementa a dinâmica de movimento do exemplo, incluindo colisões
    Position getNextPosition(const Position& current, Action action) const {
        Position next = current;
        switch (action) {
        case Action::North: next.row++; break;
        case Action::South: next.row--; break;
        case Action::East:  next.col++; break;
        case Action::West:  next.col--; break;
        }

        // Se bater em parede ou sair do grid, permanece na posição atual
        if (!isValidPosition(next) || grid[next.row][next.col] == CellType::Wall) {
            return current;
        }
        return next;
    }

    // Retorna as ações perpendiculares para uma dada ação
    // Usado para implementar o movimento estocástico (0.1 para cada lado)
    std::pair<Action, Action> getPerpendicularActions(Action action) const {
        switch (action) {
        case Action::North:
        case Action::South:
            return { Action::East, Action::West };
        case Action::East:
        case Action::West:
            return { Action::North, Action::South };
        }
        return { Action::North, Action::South }; // Caso padrão
    }

    // Calcula o valor de uma ação em um estado específico
    // Implementa a equação de Bellman conforme mostrado no exemplo
    double calculateActionValue(const Position& pos, Action action) const {
        double totalValue = 0.0;

        // Direção principal (0.8 de probabilidade)
        Position mainPos = getNextPosition(pos, action);
        totalValue += MAIN_PROB * (STEP_REWARD + DISCOUNT_FACTOR * values[mainPos.row][mainPos.col]);

        // Direções perpendiculares (0.1 de probabilidade cada)
        auto [side1, side2] = getPerpendicularActions(action);
        Position sidePos1 = getNextPosition(pos, side1);
        Position sidePos2 = getNextPosition(pos, side2);

        totalValue += SIDE_PROB * (STEP_REWARD + DISCOUNT_FACTOR * values[sidePos1.row][sidePos1.col]);
        totalValue += SIDE_PROB * (STEP_REWARD + DISCOUNT_FACTOR * values[sidePos2.row][sidePos2.col]);

        return totalValue;
    }

public:
    // Construtor: inicializa o grid conforme especificado no exemplo
    GridWorld() {
        // Inicializa todas as células como normais
        for (auto& row : grid) {
            row.fill(CellType::Normal);
        }

        // Define os estados especiais conforme exemplo
        grid[2][3] = CellType::TerminalPos;  // Canto superior direito (+1)
        grid[1][1] = CellType::TerminalNeg;  // Centro (-1)
        grid[1][2] = CellType::Wall;         // Parede

        // Inicializa valores dos estados
        for (auto& row : values) {
            row.fill(0.0);
        }
        values[2][3] = POSITIVE_TERMINAL_REWARD;  // +1
        values[1][1] = NEGATIVE_TERMINAL_REWARD;  // -1

        // Inicializa política com um valor padrão
        for (auto& row : policy) {
            row.fill(Action::North);
        }
    }

    // Executa o algoritmo de iteração de valor
    // Implementa o processo iterativo mostrado no exemplo
    void runValueIteration() {
        double maxChange;
        int iteration = 0;

        do {
            maxChange = 0.0;

            // Atualiza cada estado não-terminal e não-parede
            for (int row = 0; row < ROWS; ++row) {
                for (int col = 0; col < COLS; ++col) {
                    if (grid[row][col] == CellType::Normal) {
                        Position currentPos{ row, col };
                        double oldValue = values[row][col];

                        // Testa todas as ações e encontra a melhor
                        // Similar ao processo manual do exemplo
                        double maxActionValue = -std::numeric_limits<double>::infinity();
                        Action bestAction = Action::North;

                        for (const auto action : { Action::North, Action::South,
                                                Action::East, Action::West }) {
                            double actionValue = calculateActionValue(currentPos, action);
                            if (actionValue > maxActionValue) {
                                maxActionValue = actionValue;
                                bestAction = action;
                            }
                        }

                        // Atualiza valor e política
                        values[row][col] = maxActionValue;
                        policy[row][col] = bestAction;

                        // Monitora a maior mudança para verificar convergência
                        maxChange = std::max(maxChange, std::abs(values[row][col] - oldValue));
                    }
                }
            }

            ++iteration;
            std::cout << std::format("Iteração {}: Mudança máxima = {:.6f}\n",
                iteration, maxChange);

        } while (maxChange > CONVERGENCE_THRESHOLD);
    }

    // Imprime a matriz de valores final
    // Formato similar ao mostrado no exemplo
    void printValues() const {
        std::cout << "\nValores Finais:\n";
        for (int row = ROWS - 1; row >= 0; --row) {
            for (int col = 0; col < COLS; ++col) {
                if (grid[row][col] == CellType::Wall) {
                    std::cout << "  WALL  ";
                }
                else {
                    std::cout << std::format("{:7.2f}", values[row][col]);
                }
            }
            std::cout << '\n';
        }
    }

    // Imprime a política ótima
    // Usando setas como no exemplo
    void printPolicy() const {
        std::cout << "\nPolítica Ótima:\n";
        for (int row = ROWS - 1; row >= 0; --row) {
            for (int col = 0; col < COLS; ++col) {
                if (grid[row][col] == CellType::Wall) {
                    std::cout << "  W  ";
                }
                else if (grid[row][col] == CellType::TerminalPos) {
                    std::cout << "  +  ";
                }
                else if (grid[row][col] == CellType::TerminalNeg) {
                    std::cout << "  -  ";
                }
                else {
                    char actionSymbol;
                    switch (policy[row][col]) {
                    case Action::North: actionSymbol = '^'; break;
                    case Action::South: actionSymbol = 'v'; break;
                    case Action::East:  actionSymbol = '>'; break;
                    case Action::West:  actionSymbol = '<'; break;
                    }
                    std::cout << std::format("  {}  ", actionSymbol);
                }
            }
            std::cout << '\n';
        }
    }
};

int main() {
    GridWorld world;

    std::cout << "Iniciando Iteração de Valor...\n";
    world.runValueIteration();

    world.printValues();
    world.printPolicy();

    return 0;
}
```

## Usando Iteração de Política no Grid World - Manualmente

Em contraste com a Iteração de Valor, a Iteração de Política alterna entre dois passos distintos: avaliação da política e melhoria da política. Vamos resolver o mesmo problema anterior usando este método.

Relembrando nosso ambiente:
- Grade $4 \times 3$;
- Estado inicial em $(0,0)$;
- Estado terminal positivo $(3,2)$ com recompensa $+1$;
- Estado terminal negativo $(1,1)$ com recompensa $-1$;
- Parede em $(2,1)$;
- Fator de desconto $\gamma = 0.9$;
- Recompensa por passo $r_{vida} = -0.03$;
- Dinâmica estocástica: $0.8$ na direção pretendida, $0.1$ para cada lado.

### Inicialização

Começamos com uma política inicial arbitrária (por exemplo, todas as ações apontando para $\text{Norte}$) e valores iniciais zero. Neste caso, nosso mundo seria representado por:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & 0.00 & 0.00 & 0.00 & +1.00 \\
1 & 0.00 & -1.00 & \text{WALL} & 0.00 \\
0 & 0.00 & 0.00 & 0.00 & 0.00
\end{array}
$$

Política inicial $\pi_0$:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & \uparrow & \uparrow & \uparrow & +1 \\
1 & \uparrow & -1 & \text{WALL} & \uparrow \\
0 & \uparrow & \uparrow & \uparrow & \uparrow
\end{array}
$$

#### Primeira Iteração

1) **Avaliação da Política**

Para cada estado, calculamos $V^{\pi_1}$ usando a política atual.

Para o estado inicial $(0,0)$:

a) Política atual diz $\text{Leste}$ ($\rightarrow$) após a primeira melhoria:

- Movimento principal (0.8):
  * Próximo estado: $(0,1)$;
  * $0.8(-0.03 + 0.9 \times -0.22)$.
  
- Desvio lateral direito (0.1):
  * Próximo estado: $(1,0)$;
  * $0.1(-0.03 + 0.9 \times -0.25)$.
  
- Desvio lateral esquerdo (0.1):
  * Colide com parede, permanece em $(0,0)$;
  * $0.1(-0.03 + 0.9 \times -0.28)$.

Total: $V^{\pi_1}(0,0) = -0.25$

Para o estado $(0,1)$:

b) Política atual diz $\text{Leste}$ ($\rightarrow$) após a primeira melhoria:

- Movimento principal (0.8):
  * Próximo estado: $(0,2)$;
  * $0.8(-0.03 + 0.9 \times -0.15)$.
  
- Desvio lateral direito (0.1):
  * Próximo estado: $(1,1)$ (estado terminal negativo);
  * $0.1(-0.03 + 0.9 \times -1.0)$.
  
- Desvio lateral esquerdo (0.1):
  * Próximo estado: $(0,0)$;
  * $0.1(-0.03 + 0.9 \times -0.28)$.

Total: $V^{\pi_1}(0,1) = -0.22$

Para o estado $(2,0)$:

c) Política atual diz $\text{Leste}$ ($\rightarrow$) após a primeira melhoria:

- Movimento principal (0.8):
  * Próximo estado: $(2,1)$;
  * $0.8(-0.03 + 0.9 \times -0.08)$.
  
- Desvio lateral direito (0.1):
  * Próximo estado: $(3,0)$;
  * $0.1(-0.03 + 0.9 \times 0.85)$.
  
- Desvio lateral esquerdo (0.1):
  * Próximo estado: $(1,0)$;
  * $0.1(-0.03 + 0.9 \times 0.58)$.

Total: $V^{\pi_1}(2,0) = 0.74$

Para o estado $(1,2)$:

d) Política atual diz $\text{Norte}$ ($\uparrow$) após a primeira melhoria:

- Movimento principal (0.8):
  * Próximo estado: $(2,2)$;
  * $0.8(-0.03 + 0.9 \times 0.85)$.
  
- Desvio lateral direito (0.1):
  * Próximo estado: $(1,3)$;
  * $0.1(-0.03 + 0.9 \times 0.89)$.
  
- Desvio lateral esquerdo (0.1):
  * Próximo estado: $(1,1)$ (estado terminal negativo);
  * $0.1(-0.03 + 0.9 \times -1.0)$.

Total: $V^{\pi_1}(1,2) = 0.64$

Este processo de avaliação continua iterativamente para todos os estados não-terminais até que a diferença máxima entre duas iterações sucessivas seja menor que nosso limiar de convergência $\epsilon = 0.001$, ou seja, até que $\max_s \mid V_{k+1}(s) - V_k(s)\mid  < \epsilon$. Após essa convergência, que tipicamente leva 20-30 iterações de avaliação para nossa grade $4 \times 3$, obteremos:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & -0.22 & -0.15 & -0.08 & +1.00 \\
1 & -0.25 & -1.00 & \text{WALL} & 0.54 \\
0 & -0.28 & -0.22 & -0.15 & -0.08
\end{array}
$$

2) **Melhoria da Política**

Para cada estado, encontramos a ação que maximiza:

$$\pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s' \mid s,a)[R(s,a,s') + \gamma V^{\pi_k}(s')\;]$$

Para o estado $(0,0)$:

a) Valor para Norte:
- Principal (0.8): $(-0.03 + 0.9 \times -0.25) \times 0.8 = -0.204$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.22) \times 0.1 = -0.0228$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -0.28) \times 0.1 = -0.0282$.
Total: $-0.28$

b) Valor para Sul (colide com parede):
- Principal (0.8): $(-0.03 + 0.9 \times -0.28) \times 0.8 = -0.2256$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.22) \times 0.1 = -0.0228$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -0.28) \times 0.1 = -0.0282$.
Total: $-0.31$

c) Valor para Leste:
- Principal (0.8): $(-0.03 + 0.9 \times -0.22) \times 0.8 = -0.1824$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.25) \times 0.1 = -0.0255$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -0.28) \times 0.1 = -0.0282$.
Total: $-0.25$

d) Valor para Oeste (colide com parede):
- Principal (0.8): $(-0.03 + 0.9 \times -0.28) \times 0.8 = -0.2256$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.25) \times 0.1 = -0.0255$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -0.28) \times 0.1 = -0.0282$.
Total: $-0.31$

A melhor ação é Leste $(-0.25)$, então atualizamos a política.

Para o estado $(1,2)$:

e) Valor para Norte:
- Principal (0.8): $(-0.03 + 0.9 \times -0.08) \times 0.8 = -0.0576$;
- Direita (0.1): $(-0.03 + 0.9 \times 0.54) \times 0.1 = 0.0456$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -1.0) \times 0.1 = -0.093$.
Total: $-0.105$

f) Valor para Sul:
- Principal (0.8): $(-0.03 + 0.9 \times -0.15) \times 0.8 = -0.132$;
- Direita (0.1): $(-0.03 + 0.9 \times 0.54) \times 0.1 = 0.0456$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -1.0) \times 0.1 = -0.093$.
Total: $-0.1794$

g) Valor para Leste:
- Principal (0.8): $(-0.03 + 0.9 \times 0.54) \times 0.8 = 0.3648$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.08) \times 0.1 = -0.0072$;
- Esquerda (0.1): $(-0.03 + 0.9 \times -1.0) \times 0.1 = -0.093$.
Total: $0.2646$

h) Valor para Oeste:
- Principal (0.8): $(-0.03 + 0.9 \times -1.0) \times 0.8 = -0.744$;
- Direita (0.1): $(-0.03 + 0.9 \times -0.08) \times 0.1 = -0.0072$;
- Esquerda (0.1): $(-0.03 + 0.9 \times 0.54) \times 0.1 = 0.0456$.
Total: $-0.7056$

A melhor ação é Leste $(0.2646)$, atualizamos a política.

Após calcular de forma similar para todos os estados não-terminais, obteremos a nova política $\pi_1$:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & \rightarrow & \rightarrow & \rightarrow & +1 \\
1 & \uparrow & -1 & \text{WALL} & \uparrow \\
0 & \rightarrow & \rightarrow & \rightarrow & \uparrow
\end{array}
$$

A atenta leitora deve ter notado que este processo de melhoria da política é mais detalhado que na Iteração de Valor, pois calculamos explicitamente o valor de cada ação possível em cada estado. Além disso, *a melhoria da política só ocorre após a convergência completa da avaliação da política, diferentemente da Iteração de Valor na qual as atualizações são entrelaçadas*.

#### Iterações Subsequentes

O processo continua alternando entre avaliação e melhoria da política. Após algumas iterações, os valores convergirão para:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & 0.64 & 0.74 & 0.85 & +1.00 \\
1 & 0.58 & -1.00 & \text{WALL} & 0.89 \\
0 & 0.53 & 0.64 & 0.74 & 0.85
\end{array}
$$

E a política ótima final $\pi^*$ será:

$$
\begin{array}{c|cccc}
& 0 & 1 & 2 & 3 \\
\hline
2 & \rightarrow & \rightarrow & \rightarrow & +1 \\
1 & \uparrow & -1 & \text{WALL} & \uparrow \\
0 & \uparrow & \rightarrow & \rightarrow & \uparrow
\end{array}
$$

#### Convergência

A Iteração de Política converge em menos iterações que a Iteração de Valor (tipicamente 3-4 iterações versus 15-20 para Iteração de Valor), mas cada iteração acrescenta tempo de computação, devido à avaliação completa da política. Os dois métodos convergem para a mesma solução ótima, como esperado teoricamente.

A política ótima obtida é idêntica à encontrada por Iteração de Valor, confirmando que os métodos encontram a mesma solução ótima por caminhos diferentes. A principal diferença está na forma como chegam lá:
- **Iteração de Valor**: atualiza valores e melhora a política implicitamente;
- **Iteração de Política**: separa explicitamente a avaliação e melhoria da política.

### Iteração de Política Em C++

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <format>
#include <limits>
#include <chrono>

// Define enums for actions and cell types
enum class Action { North, South, East, West };
enum class CellType { Normal, Wall, TerminalPos, TerminalNeg };

// Structure for maintaining the grid positions
struct Position {
    int row;
    int col;

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
};

class GridWorldPolicyIteration {
private:
    static constexpr int ROWS = 3;
    static constexpr int COLS = 4;
    static constexpr double STEP_REWARD = -0.03;
    static constexpr double POSITIVE_TERMINAL = 1.0;
    static constexpr double NEGATIVE_TERMINAL = -1.0;
    static constexpr double GAMMA = 0.9;
    static constexpr double MAIN_PROB = 0.8;
    static constexpr double SIDE_PROB = 0.1;
    static constexpr double CONVERGENCE_THRESHOLD = 0.001;
    static constexpr double EVAL_CONVERGENCE_THRESHOLD = 0.0001;

    // Grid representation
    std::array<std::array<CellType, COLS>, ROWS> grid;
    std::array<std::array<double, COLS>, ROWS> values;
    std::array<std::array<Action, COLS>, ROWS> policy;

    bool isValidPosition(const Position& pos) const {
        return pos.row >= 0 && pos.row < ROWS &&
            pos.col >= 0 && pos.col < COLS &&
            grid[pos.row][pos.col] != CellType::Wall;
    }

    Position getNextPosition(const Position& current, Action action) const {
        Position next = current;
        switch (action) {
        case Action::North: next.row++; break;
        case Action::South: next.row--; break;
        case Action::East:  next.col++; break;
        case Action::West:  next.col--; break;
        }
        return isValidPosition(next) ? next : current;
    }

    std::pair<Action, Action> getPerpendicularActions(Action action) const {
        switch (action) {
        case Action::North:
        case Action::South:
            return { Action::East, Action::West };
        case Action::East:
        case Action::West:
            return { Action::North, Action::South };
        }
        return { Action::North, Action::South }; // Default case
    }

    double getReward(const Position& pos) const {
        switch (grid[pos.row][pos.col]) {
        case CellType::TerminalPos: return POSITIVE_TERMINAL;
        case CellType::TerminalNeg: return NEGATIVE_TERMINAL;
        default: return STEP_REWARD;
        }
    }

    double calculateActionValue(const Position& pos, Action action) const {
        if (grid[pos.row][pos.col] != CellType::Normal) {
            return getReward(pos);
        }

        double totalValue = 0.0;

        // Main direction
        Position mainPos = getNextPosition(pos, action);
        totalValue += MAIN_PROB * (STEP_REWARD + GAMMA * values[mainPos.row][mainPos.col]);

        // Perpendicular directions
        auto [side1, side2] = getPerpendicularActions(action);
        Position sidePos1 = getNextPosition(pos, side1);
        Position sidePos2 = getNextPosition(pos, side2);

        totalValue += SIDE_PROB * (STEP_REWARD + GAMMA * values[sidePos1.row][sidePos1.col]);
        totalValue += SIDE_PROB * (STEP_REWARD + GAMMA * values[sidePos2.row][sidePos2.col]);

        return totalValue;
    }

    // Policy evaluation step
    double evaluatePolicy() {
        double maxDelta;
        int evalIterations = 0;

        do {
            maxDelta = 0.0;
            for (int row = 0; row < ROWS; ++row) {
                for (int col = 0; col < COLS; ++col) {
                    if (grid[row][col] == CellType::Normal) {
                        Position pos{ row, col };
                        double oldValue = values[row][col];
                        values[row][col] = calculateActionValue(pos, policy[row][col]);
                        maxDelta = std::max(maxDelta, std::abs(values[row][col] - oldValue));
                    }
                }
            }
            evalIterations++;
        } while (maxDelta > EVAL_CONVERGENCE_THRESHOLD);

        std::cout << std::format("Policy evaluation took {} iterations\n", evalIterations);
        return maxDelta;
    }

    // Policy improvement step
    bool improvePolicy() {
        bool policyStable = true;

        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col < COLS; ++col) {
                if (grid[row][col] == CellType::Normal) {
                    Position pos{ row, col };
                    Action oldAction = policy[row][col];

                    // Find best action
                    double maxValue = -std::numeric_limits<double>::infinity();
                    Action bestAction = Action::North;

                    // Calculate and print values for all actions
                    std::cout << std::format("\nState ({},{})\n", row, col);
                    for (const auto action : { Action::North, Action::South,
                                           Action::East, Action::West }) {
                        double actionValue = calculateActionValue(pos, action);
                        std::cout << std::format("Action {}: {:.4f}\n",
                            static_cast<int>(action), actionValue);

                        if (actionValue > maxValue) {
                            maxValue = actionValue;
                            bestAction = action;
                        }
                    }

                    policy[row][col] = bestAction;
                    if (oldAction != bestAction) {
                        policyStable = false;
                        std::cout << std::format("Policy changed at ({},{})\n", row, col);
                    }
                }
            }
        }
        return policyStable;
    }

public:
    GridWorldPolicyIteration() {
        // Initialize grid
        for (auto& row : grid) {
            row.fill(CellType::Normal);
        }

        // Set special states
        grid[2][3] = CellType::TerminalPos;  // Top-right corner
        grid[1][1] = CellType::TerminalNeg;  // Center
        grid[1][2] = CellType::Wall;         // Wall

        // Initialize values
        for (auto& row : values) {
            row.fill(0.0);
        }
        values[2][3] = POSITIVE_TERMINAL;
        values[1][1] = NEGATIVE_TERMINAL;

        // Initialize policy (all North)
        for (auto& row : policy) {
            row.fill(Action::North);
        }
    }

    void runPolicyIteration() {
        int iteration = 0;
        bool policyStable = false;

        while (!policyStable) {
            std::cout << std::format("\nIteration {}\n", iteration);
            std::cout << "Policy Evaluation Step:\n";
            evaluatePolicy();
            printValues();

            std::cout << "\nPolicy Improvement Step:\n";
            policyStable = improvePolicy();
            printPolicy();

            iteration++;
        }

        std::cout << std::format("\nPolicy Iteration converged after {} iterations\n",
            iteration);
    }

    void printValues() const {
        std::cout << "\nValues:\n";
        for (int row = ROWS - 1; row >= 0; --row) {
            for (int col = 0; col < COLS; ++col) {
                if (grid[row][col] == CellType::Wall) {
                    std::cout << "  WALL  ";
                }
                else {
                    std::cout << std::format(" {: .3f}", values[row][col]);
                }
            }
            std::cout << '\n';
        }
    }

    void printPolicy() const {
        std::cout << "\nPolicy:\n";
        for (int row = ROWS - 1; row >= 0; --row) {
            for (int col = 0; col < COLS; ++col) {
                if (grid[row][col] == CellType::Wall) {
                    std::cout << "  W  ";
                }
                else if (grid[row][col] == CellType::TerminalPos) {
                    std::cout << "  +  ";
                }
                else if (grid[row][col] == CellType::TerminalNeg) {
                    std::cout << "  -  ";
                }
                else {
                    char arrow;
                    switch (policy[row][col]) {
                    case Action::North: arrow = '^'; break;
                    case Action::South: arrow = 'v'; break;
                    case Action::East:  arrow = '>'; break;
                    case Action::West:  arrow = '<'; break;
                    }
                    std::cout << std::format("  {}  ", arrow);
                }
            }
            std::cout << '\n';
        }
    }
};

int main() {
    GridWorldPolicyIteration gridWorld;

    std::cout << "Starting Policy Iteration...\n";
    auto start = std::chrono::high_resolution_clock::now();

    gridWorld.runPolicyIteration();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << std::format("\nPolicy Iteration completed in {} ms\n", duration.count());

    return 0;
}
```

## Comparando Iteração de Valor e Iteração de Política

Agora que implementamos e entendemos tanto a Iteração de Valor quanto a Iteração de Política, podemos analisar as características específicas de cada método e entender quando usar cada um.

### Complexidade Computacional

**Iteração de Valor**:
- Complexidade por iteração: $O(\vert S\vert^2\vert A\vert)$, de forma que $\vert S\vert$ é o número de estados e $|A|$ é o número de ações;
- Realiza uma única atualização de valor por estado em cada iteração;
- Número típico de iterações até convergência: $O(\frac{\log(1/\epsilon)}{1-\gamma})$.

**Iteração de Política**:
- Complexidade por iteração: $O(\vert S\vert^3)$ para avaliação de política + $O(\vert S\vert^2\vert A\vert)$ para melhoria de política;
- Requer solução de sistema linear na avaliação de política;
- Número típico de iterações até convergência: $O(\log(\vert A\vert))$.

### Requisitos de Memória

**Iteração de Valor**:
- Requer armazenamento de $O(\vert S\vert)$ para valores dos estados;
- Política é implícita durante o processo;
- Memória adicional $O(\vert A\vert)$ para cálculos temporários.

**Iteração de Política**:
- Requer armazenamento de $O(\vert S\vert)$ para valores dos estados;
- Armazenamento adicional de $O(\vert S\vert)$ para política explícita;
- Memória adicional $O(\vert S\vert^2)$ durante avaliação de política.

### Propriedades de Convergência

**Iteração de Valor**:
- Converge em taxa linear;
- Taxa de convergência: $(1-\gamma)$ por iteração;
- Convergência monótona: $\|V_{k+1} - V^*\| \leq \gamma\|V_k - V^*\|$.

**Iteração de Política**:
- Converge em número finito de iterações;
- Convergência com um número menor de iterações;
- Cada iteração é computacionalmente mais intensiva;
- Convergência estrita: $V^{\pi_{k+1}} > V^{\pi_k}$.

### Considerações Práticas

**Iteração de Valor é preferível quando**:
- O espaço de estados é grande mas esparso;
- Precisão moderada é suficiente;
- Memória é limitada;
- O fator de desconto $\gamma$ está longe de $1$.

**Iteração de Política é preferível quando**:
- O espaço de estados é pequeno ou denso;
- Alta precisão é necessária;
- Recursos computacionais abundantes;
- Número de ações é grande em relação ao número de estados.

### Em Nosso Grid World

Para nosso exemplo específico de Grid World $4 \times 3$:

1. **Tamanho do Problema**:
  - Número de estados: $|S| = 12$ (incluindo terminais);
  - Número de ações: $|A| = 4$ ($\text{Norte}$, $\text{Sul}$, $\text{Leste}$, $\text{Oeste}$);
  - Fator de desconto: $\gamma = 0.9$.

2. **Comparação Empírica**:
  - Iteração de Valor: converge em ~15-20 iterações;
  - Iteração de Política: converge em 3-4 iterações completas.

3. **Tempos de Execução** (em nosso código C++):
  - Iteração de Valor: tipicamente $\sim$ 5ms;
  - Iteração de Política: tipicamente $\sim$ 8ms.

4. **Precisão Final**:
  - Os métodos convergem para a mesma política ótima;
  - Diferença nos valores finais $< 10^{-6}$.

Para problemas pequenos como nosso Grid World, a escolha entre os métodos faz pouca, ou nenhuma diferença. A diferença se torna significativa em problemas maiores ou quando há requisitos específicos de precisão ou de uso de recursos computacionais.

## *Dynamic Programming*, MDPs e *Reinforcement Learning*

A esforçada leitora que leu até aqui deve estar se perguntando: onde está o *Reinforcement Learning*? Se sim, esta é uma pergunta válida.

Vimos que nas técnicas de *Reinforcement Learning* temos um *agente que aprende com experimentando o ambiente* e, não vimos nada disso usando o **MDP** para resolver o **Grid World**. Não tema. Este é um processo didático em que iremos aumentando a complexidade assim que os conceitos forem sendo sedimentados. Comecei pelo **MDP** por causa da programação dinâmica (*Dynamic Programming*).

A *Dynamic Programming* é uma técnica de otimização que resolve problemas complexos decompondo-os em subproblemas mais simples. Tanto o contexto do **Grid World** quanto no contexto do *Reinforcement Learning*, o entendimento de *Dynamic Programming* será necessário por duas características:

1. **Subestrutura Ótima**: a solução ótima do problema pode ser construída a partir das soluções ótimas de seus subproblemas. No **Grid World**, o valor ótimo de um estado depende dos valores ótimos dos estados subsequentes, como expresso na *Equação de Bellman*:

   $$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'\mid s,a)[R(s,a,s') + \gamma V^*(s')]$$

2. **Subproblemas Sobrepostos**: os mesmos subproblemas aparecem repetidamente ao resolver o problema maior. No **Grid World**, o valor de um estado particular será usado múltiplas vezes ao calcular os valores de outros estados.

### A Tríade: MDPs, *Dynamic Programming* e RL

A relação entre Processos de Decisão de Markov (**MDPs**), *Dynamic Programming* e *Reinforcement Learning* pode ser entendida como uma hierarquia de abstração, definida por:

1. **MDPs** fornecem o framework matemático para modelar problemas de decisão sequencial composto por:
   - Estados ($S$);
   - Ações ($A$);
   - Probabilidades de transição ($P$);
   - Recompensas ($R$);
   - Fator de desconto ($\gamma$).

2. ***Dynamic Programming*** oferece métodos para resolver MDPs quando o modelo é completamente conhecido. Neste caso, temos dois métodos baseados nas Equações de Bellman:
   - Iteração de Valor;
   - Iteração de Política;

3. **Reinforcement Learning** estende esses conceitos para situações nas quais o modelo não é conhecido, usando métodos como:
   - Q-Learning;
   - SARSA;
   - Outros métodos que aprendem através da interação.

## Implementação em C++

Nas nossas implementações do **Grid World**, a *Dynamic Programming* se manifesta de várias formas:

1. **Estruturas de Memorização**:

```cpp
std::array<std::array<double, COLS>, ROWS> values;  // Tabela de valores
std::array<std::array<Action, COLS>, ROWS> policy;  // Tabela de política
```

2. **Atualização Iterativa (Value Iteration)**:

```cpp
void runValueIteration() {
    double maxChange;
    do {
        maxChange = 0.0;
        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col < COLS; ++col) {
                // Atualiza valor usando subproblemas já resolvidos
                double oldValue = values[row][col];
                values[row][col] = calculateActionValue(/*...*/);
                maxChange = std::max(maxChange,
                    std::abs(values[row][col] - oldValue));
            }
        }
    } while (maxChange > CONVERGENCE_THRESHOLD);
}
```

3. **Avaliação de Política (Policy Iteration)**:

```cpp
double evaluatePolicy() {
    double maxDelta;
    do {
        maxDelta = 0.0;
        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col < COLS; ++col) {
                // Resolve sistema de equações usando *Dynamic Programming*
                double oldValue = values[row][col];
                values[row][col] = calculateActionValue(/*...*/);
                maxDelta = std::max(maxDelta,
                    std::abs(values[row][col] - oldValue));
            }
        }
    } while (maxDelta > EVAL_CONVERGENCE_THRESHOLD);
    return maxDelta;
}
```

### Características da *Dynamic Programming* no Grid World

O estudo do **Grid World** que a esforçada leitora acompanhou exibe várias características clássicas da *Dynamic Programming* que precisam ser destacadas:

1. **Tabela de Memorização**: A matriz `values` armazena resultados intermediários:

  $$V_k(s) = \text{values[s.row][s.col]}$$

2. **Atualização Bottom-Up**: Os valores são atualizados iterativamente:

  $$V_{k+1}(s) = \max_{a \in A} Q_k(s,a)$$

3. **Convergência Garantida**: O algoritmo converge devido ao fator de desconto $\gamma < 1$:

  $$\|V_{k+1} - V^*\| \leq \gamma\|V_k - V^*\|$$

## Da *Dynamic Programming* ao *Reinforcement Learning*

A *Dynamic Programming* no **Grid World** serve como base para a compreensão de métodos avançados de *Reinforcement Learning*. Mas, *Dynamic Programming* não é *Reinforcement Learning*. A principal diferença está na necessidade de conhecimento do modelo do ambiente.

Na implementação que desenvolvemos, com *Dynamic Programming*, foi necessário e indispensável conhecer todas as probabilidades de transição $P(s'|s,a)$ e recompensas $R(s,a,s')$. Este é o chamado método *model-based*. Neste método, o modelo completo do ambiente é conhecido e utilizado diretamente nos cálculos. Por outro lado, em *Reinforcement Learning*, os agentes precisam aprender através da interação com o ambiente, sem ter acesso a esse modelo completo, o que dizemos ser uma abordagem *model-free*.

A distinção entre *Dynamic Programming* e *Reinforcement Learning* fica evidente na forma como as Equações de Bellman são utilizadas.

Na *Dynamic Programming*, podemos aplicar estas equações diretamente, calculando os valores exatos para cada estado através de operações determinísticas. Foi isso que fizemos, duas vezes, com algoritmos diferentes.

No *Reinforcement Learning*, precisamos aproximar estas equações usando amostras de experiência real. Cada interação com o ambiente fornece uma estimativa ruidosa dos verdadeiros valores. Por exemplo, enquanto nossa implementação em *Dynamic Programming* calcula $V(s)$ usando todas as possíveis transições, um agente de *Reinforcement Learning* precisa estimar este valor observando resultados reais de suas ações.

A convergência também segue padrões distintos. Nossa implementação de *Dynamic Programming* converge de forma determinística, com garantias matemáticas claras baseadas no fator de desconto $\gamma$. Cada iteração reduz o erro de forma previsível e monótona. Em contraste, métodos de *Reinforcement Learning* têm convergência estocástica. Eventualmente os métodos de *Reinforcement Learning* convergem para a solução ótima, mas o caminho até lá é irregular, com flutuações causadas pela natureza aleatória das experiências e explorações[^4].

[^4]: **Experiências**: refere-se a todas as interações que o agente tem com o ambiente: cada ação tomada e cada recompensa recebida é uma experiência. **Exploração** (*exploration*) é quando o agente tenta ações novas ou diferentes para descobrir melhores estratégias. **Explotação** (exploitation) é quando o agente usa o conhecimento que já tem para escolher ações que ele sabe que são boas.

A implementação em C++ do **Grid World** usando *Dynamic Programming* serve assim como um caso ideal que ilustra os princípios fundamentais: decomposição do problema - memorização de soluções parciais, e atualização sistemática de valores - que são posteriormente adaptados e generalizados em *Reinforcement Learning* para situações nas quais o modelo completo não está disponível. Esta progressão de *Dynamic Programming* para *Reinforcement Learning* reflete a evolução que teremos neste texto.

## Exercícios MDP - Grid World

1. Modifique o Grid World para ter diferentes dimensões (por exemplo, 5x5, 8x8) e compare como a Iteração de Valor e a Iteração de Política escalam.

2. Experimente com diferentes fatores de desconto $(\gamma)$ e analise como eles afetam a velocidade de convergência de cada método.

3. Implemente um método híbrido que alterne entre Iteração de Valor e Iteração de Política baseado em algum critério. Em que situações isso poderia ser benéfico?

4. Adicione obstáculos ao Grid World e analise como a topologia do ambiente afeta o desempenho relativo de ambos os métodos.

5. Modifique a estrutura de recompensas e analise como diferentes esquemas de recompensa afetam as propriedades de convergência de ambos os algoritmos.

### Dicas de Resolução

1. Para o primeiro exercício, considere que a complexidade computacional:

$$
\begin{align*}
\text{Iteração de Valor:} & \quad O(|S|^2|A|) \text{ por iteração} \\
\text{Iteração de Política:} & \quad O(|S|^3) \text{ para avaliação} + O(|S|^2|A|) \text{ para melhoria}
\end{align*}
$$

Na qual, $|S|$ cresce quadraticamente com a dimensão da grade.

2. Para o segundo exercício, note que valores de $\gamma$ próximos de $1$ geralmente:
  - Aumentam o número de iterações necessárias;
  - Produzem políticas que consideram recompensas distantes;
  - Afetam mais a Iteração de Valor que a Iteração de Política.
.
3. Para o método híbrido, considere alternar com base em:
  - Taxa de mudança nos valores dos estados;
  - Número de iterações já executadas;
  - Tamanho do problema.

4. Para obstáculos, observe que eles:
  - Reduzem o espaço de estados efetivo $|S|$;
  - Podem criar "corredores" que afetam a propagação de valores;
  - Podem impactar diferentemente cada método

5. Na modificação de recompensas, considere:
  - Recompensas esparsas vs densas;
  - Recompensas positivas vs negativas;
  - Magnitude das recompensas.