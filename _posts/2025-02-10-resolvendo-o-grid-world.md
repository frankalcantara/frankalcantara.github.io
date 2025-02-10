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
description: "Uma exploração detalhada da solução do Grid World usando programação dinâmica, com implementação em C++ 20. "
date: 2025-02-10T09:29:17.254Z
preview: \mid  Descubra como resolver o Grid World usando programação dinâmica e C++ 20. Um guia prático e matemático para entender a solução de MDPs, desde as equações de Bellman até a implementação computacional.
keywords: \mid - Grid World Reinforcement Learning MDP Solution Aprendizado por Reforço Processo de Decisão de Markov Equações de Bellman Value Iteration Policy Iteration Programação Dinâmica Dynamic Programming Política Ótima Optimal Policy
toc: true
published: true
beforetoc: ""
lastmod: 2025-02-10T23:03:07.015Z
draft: 2025-02-10T09:29:19.442Z
---

Agora que formalizamos o **Grid World** como um *Processo de Decisão de Markov* (**MDP**), podemos aplicar algoritmos de *Reinforcement Learning* (**RL**) para encontrar a política ótima $\pi^\*$. *A política ótima é aquela que maximiza a recompensa total esperada a longo prazo para o agente*. Vamos explorar como a estrutura do **MDP** nos permite resolver o **Grid World**.

## Formulação MDP do Grid World

Vamos dedicar um minuto, ou dez, para lembrar em que porto desta jornada estamos. Até agora definimos o seguinte:

1. **Estados $(S)$**: O conjunto de todas as células da grade. Cada célula $(x, y)$ representa um estado.

2. **Ações $(A)$**: O conjunto de ações possíveis: $A= \{\text{Norte}, \text{Sul}, \text{Leste}, \text{Oeste}\}$.

3. **Função de Transição $(P)$**:  $P(s' \mid  s, a)$ define a probabilidade de transitar para o estado $s'$ ao executar a ação $a$ no estado $s$. No nosso **Grid World** estocástico:

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

    Onde $r_{vida}$ é a recompensa por passo (living reward), geralmente um valor pequeno e negativo (por exemplo, $-0.03$).

5. **Estado Inicial $(s_0)$**: A célula onde o agente inicia cada episódio.

6. **Estados Terminais $(S_{terminal})$**: As células que, quando alcançadas, encerram o episódio.

## Resolvendo o Grid World

A abordagem inocente para encontrar a política ótima para resolver o **Grid World** envolve explorar todas as possíveis sequências de ações a partir de cada estado inicial até alcançar o estado final. Esta é uma estratégia de, com o perdão pela má palavra, força bruta. Esta estratégia envolve os seguintes passos:

1. **Definição do Espaço de Estados e Ações**: identifique todos os estados possíveis na grade e defina todas as ações possíveis que o agente pode tomar em cada estado (por exemplo, mover-se para cima, para baixo, para a esquerda, para a direita);

2. **Geração de Sequências de Ações**: gere todas as possíveis sequências de ações a partir do estado inicial. Isso pode ser feito usando tanto iteratividade quanto recursividade para explorar todas as combinações de ações.

3. **Avaliação das Sequências**: para cada sequência de ações gerada, simule o movimento do agente na grade e calcule a recompensa acumulada ao longo do caminho; e considere todas as transições de estado e as recompensas imediatas associadas a cada ação.

4. **Seleção da Sequência Ótima**: compare as recompensas acumuladas de todas as sequências de ações; selecione a sequência que maximiza a recompensa acumulada como a política ótima.

A criativa leitora deve estar percebendo que esta é uma estratégia perfeitamente possível. Geralmente, as estratégias de força bruta são a primeira opção de quem está resolvendo um problema novo. Neste caso, significa considerar todas as combinações possíveis de movimentos (para cima, para baixo, para a esquerda, para a direita) em cada estado. Trata-se de uma solução de busca em todo o universo de possibilidades. Essa abordagem é, via de regra, ineficiente e computacionalmente dispendiosa. O número de sequências de ações possíveis cresce exponencialmente com o tamanho da grade e o número de estados. Quanto maior o mundo, mais custoso será resolver o problema.

Existem soluções melhores. A primeira, clássica e didática, foi encontrada com as Equações de Ballman.

## Entram as Equações de Bellman

As **Equações de Bellman** desenvolvidas por [Richard Bellman](https://pt.wikipedia.org/wiki/Richard_Bellman) na década de 1950, oferecem uma maneira mais eficiente e escalável de resolver o problema, decompondo-o em subproblemas menores e utilizando a programação dinâmica para encontrar a política ótima[^1]. Para a maioria dos casos práticos, especialmente em ambientes maiores ou mais complexos, as **Equações de Bellman** são a abordagem preferida.

[^1]: BELLMAN, Richard. Dynamic Programming. Princeton: Princeton University Press, 1957.

Em 1952, Richard Bellman enfrentava um dilema. Como matemático no [projeto RAND](https://open-int.blog/2018/01/21/project-rand-reports-1951-1956/), ele precisava encontrar uma maneira de resolver problemas de otimização multidimensional, e o termo "programação matemática" já estava sendo usado por outros pesquisadores. Em suas próprias palavras, o termo "programação dinâmica" foi escolhido para esconder o fato de que ele estava fazendo pesquisa matemática de um secretário do Departamento de Defesa que tinha aversão a qualquer menção à palavra *pesquisa*[^2]. É assustador, e um pouco desesperador, que uma técnica matemática tão poderosa tenha sido nomeada com o intuito de esconder sua natureza matemática, revelando como a política ruim influencia a linguagem da ciência há décadas.

[^2]:CONVERSABLE ECONOMIST. Why Is It Called “Dynamic Programming”? 24 ago. 2022. Disponível em: https://conversableeconomist.com/2022/08/24/why-is-it-called-dynamic-programming/. Acesso em: 10 fev. 2025.

A intuição fundamental de Bellman era que problemas de otimização continham subproblemas que se sobrepunham. Esta descoberta levou ao que hoje conhecemos como *Princípio da Otimalidade de Bellman*:

> Uma política ótima tem a propriedade de que, independentemente do estado inicial e da decisão inicial, as decisões restantes devem constituir uma política ótima em relação ao estado resultante da primeira decisão.

Matematicamente, este princípio pode ser expresso como:

$$ V^*(s) = \max_a \{ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)V^*(s') \} $$

A sagaz leitora será capaz de entender a revolução que Bellman iniciou com a programação dinâmica por meio de três ideias fundamentais:

1. **Decomposição Recursiva**: A solução de um problema maior pode ser construída a partir das soluções de seus subproblemas. No contexto de **MDPs**, isso significa que podemos decompor o problema de encontrar uma política ótima em subproblemas menores para cada estado.

2. **Memorização**: Ao resolver subproblemas, armazenamos suas soluções para evitar recálculos. Em termos de **MDPs**, isso se traduz em manter uma tabela de valores para cada estado.

3. **Propagação de Valor**: As soluções dos subproblemas são usadas para construir soluções para problemas maiores. Nos **MDPs**, isso corresponde à atualização iterativa dos valores dos estados.

As **Equações de Bellman** *expressam a relação recursiva entre o valor de um estado e os valores de seus estados sucessores*. Existem duas formas principais da Equação de Bellman que nos interessam no momento:

1. **Equação de Bellman para a Função Valor-Estado $(V^\pi)$**:

    A função valor-estado, $V^\pi(s)$, representa o retorno esperado (soma das recompensas descontadas) ao iniciar no estado $s$ e seguir a política $\pi$. A Equação de Bellman para $V^\pi$ é:

    $$V^\pi(s) = \sum_{a \in A} \pi(a\mid s) \sum_{s' \in S} P(s'\mid s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

    Nesta equação, temos:

    * $\pi(a\mid s)$ é a probabilidade de tomar a ação $a$ no estado $s$ sob a política $\pi$;
    * $\gamma$ é o fator de desconto $(0 \le \gamma \le 1)$, que determina a importância das recompensas futuras.

    Esta equação determina que o valor de um estado $s$ sob a política $\pi$ é a média ponderada dos valores de todos os estados sucessores possíveis $(s')$, considerando a probabilidade de cada transição $(P(s'\mid s, a))$ e a recompensa imediata $(R(s, a, s'))$.

2. **Equação de Bellman para a Função Valor-Ação $(Q^\pi)$**:

    A função valor-ação, $Q^\pi(s, a)$, representa o retorno esperado ao iniciar no estado $s$, tomar a ação $a$ e, em seguida, seguir a política $\pi$.  A Equação de Bellman para $Q^\pi$ é:

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

Ou seja, a política ótima simplesmente escolhe a ação que maximiza o valor $Q^*$ em cada estado.

## Resolvendo o Grid World com Programação Dinâmica

A programação dinâmica, como vimos, oferece uma abordagem sistemática para resolver o **Grid World**. Vamos explorar como podemos aplicar os princípios de Bellman para encontrar a política ótima através de dois algoritmos fundamentais: **Iteração de Valor** e **Iteração de Política**.

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

Onde $s_a$ é o estado resultante do movimento na direção $a$, e $s_{a1}$, $s_{a2}$ são os estados resultantes dos movimentos laterais.

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

Ambos os algoritmos convergem para a política ótima $\pi^\*$, mas de maneiras diferentes:

* a **Iteração de Valor** mantém apenas valores e deriva a política implicitamente;
* a **Iteração de Política** mantém uma política explícita e a melhora iterativamente.

Para o **Grid World**, a política ótima resultante nos dará, para cada célula da grade, a direção que o agente deve seguir para maximizar sua recompensa esperada descontada.

### Exemplo Numérico Completo

Nada como um exemplo prático, passo a passo, para que a amável leitora supere o medo do matemática e das equações assustadoras do Bellman. A Figura 1 introduz nosso primeiro problema.

![um grid world com inicio em 0,0, agente em 1,0, um obstáculo em 1,1 e objetivo em 4,3](/assets/images/gw1.webp)
_Figura 1: Exemplo de Grid World, para aplicação da programação dinâmica._{: class="legend"}

Considere um mundo representado por uma grade retangular de dimensões $4 \times 3$, onde um agente deve aprender a navegar de forma ótima. O ambiente possui as seguintes características:

1. **Grade**: O mundo é composto por $12$ células $(4 \times 3)$, onde cada célula representa um estado possível para o agente.

2. **Estados Especiais**:
   * Estado Inicial: localizado na célula $(0,0)$ (canto inferior esquerdo);
   * Estado Terminal Positivo: localizado na célula $(3,2)$ (canto superior direito), com recompensa $+1$;
   * Estado Terminal Negativo: localizado na célula $(1,1)$ (centro), com recompensa $-1$;
   * Parede: localizada na célula $(2,1)$ (centro), intransponível.

3. **Dinâmica de Movimento**:
   * O agente pode escolher entre quatro ações: $\text{Norte, Sul, Leste e Oeste}$
   * Os movimentos são estocásticos:
     * Probabilidade de $0.8$ de mover na direção pretendida;
     * Probabilidade de $0.1$ para cada direção perpendicular à pretendida.
   * Ao colidir com uma parede ou com os limites da grade, o agente permanece em sua posição atual.

4. **Sistema de Recompensas**:
   * Recompensa por passo $(r_{vida})$: $-0.03$ para cada movimento;
   * Recompensa terminal positiva: $+1.0$;
   * Recompensa terminal negativa: $-1.0$;
   * Fator de desconto $\gamma = 0.9$;

Considere que: um episódio termina quando o agente alcança qualquer estado terminal; a política deve ser determinística (uma única ação por estado); considere convergência quando a maior mudança em qualquer valor de estado for menor que $\epsilon = 0.001$.  

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

Vamos calcular o novo valor para o estado inicial (canto inferior esquerdo).

**Para a Direção $\text{Norte}**: calculando o valor para o estado inicial a ação $\text{Norte}$.

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

1. as ações $\text{Norte}$ e $\text{Leste}$ que o agente avance em direção aos estados com valores positivos mais frequentemente;

2. As ações $\text{Sul}$ e $\text{Oeste}$ resultam em mais colisões com as paredes/bordas, forçando o agente a permanecer no mesmo estado, acumulando recompensas negativas por passo;

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

Onde as setas indicam a direção ótima a seguir em cada estado.

A política ótima mostra que o agente deve:

1. partindo do estado inicial, subir e depois seguir para a direita;
2. evitar a área próxima ao estado terminal negativo;
3. procurar alcançar o estado terminal positivo pelo caminho mais seguro.

### Em C++

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
// Corresponde às quatro direções mencionadas no exercício
enum class Action { North, South, East, West };

// Define os tipos de células possíveis no grid
// Corresponde aos estados especiais mencionados no exercício
enum class CellType {
    Normal,      // Célula normal com recompensa de passo -0.03
    Wall,        // Parede - intransponível
    TerminalPos, // Estado terminal com recompensa +1
    TerminalNeg  // Estado terminal com recompensa -1
};

// Estrutura para representar uma posição no grid
// Facilita o trabalho com coordenadas (x,y) mencionadas no exercício
struct Position {
    int row;    // Linha (equivalente ao y no exercício)
    int col;    // Coluna (equivalente ao x no exercício)

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
};

class GridWorld {
private:
    // Dimensões do grid conforme especificado no exercício (4x3)
    static constexpr int ROWS = 3;
    static constexpr int COLS = 4;

    // Parâmetros de recompensa definidos no exercício
    static constexpr double STEP_REWARD = -0.03;           // r_vida
    static constexpr double POSITIVE_TERMINAL_REWARD = 1.0; // Recompensa terminal positiva
    static constexpr double NEGATIVE_TERMINAL_REWARD = -1.0;// Recompensa terminal negativa
    static constexpr double DISCOUNT_FACTOR = 0.9;         // Fator de desconto γ
    static constexpr double CONVERGENCE_THRESHOLD = 0.001; // ε para convergência

    // Probabilidades de movimento conforme exercício
    static constexpr double MAIN_PROB = 0.8;  // Probabilidade de mover na direção desejada
    static constexpr double SIDE_PROB = 0.1;  // Probabilidade de mover perpendicular

    // Representação do ambiente
    std::array<std::array<CellType, COLS>, ROWS> grid;    // Tipo de cada célula
    std::array<std::array<double, COLS>, ROWS> values;    // Valores V(s) de cada estado
    std::array<std::array<Action, COLS>, ROWS> policy;    // Política π(s) para cada estado

    // Verifica se uma posição está dentro dos limites do grid
    // Implementa a lógica de colisão com as bordas mencionada no exercício
    bool isValidPosition(const Position& pos) const {
        return pos.row >= 0 && pos.row < ROWS &&
            pos.col >= 0 && pos.col < COLS;
    }

    // Calcula a próxima posição baseada na posição atual e ação
    // Implementa a dinâmica de movimento do exercício, incluindo colisões
    Position getNextPosition(const Position& current, Action action) const {
        Position next = current;
        switch (action) {
        case Action::North: next.row++; break;
        case Action::South: next.row--; break;
        case Action::East:  next.col++; break;
        case Action::West:  next.col--; break;
        }

        // Se bater em parede ou sair do grid, permanece na posição atual
        if (!isValidPosition(next) \mid \mid  grid[next.row][next.col] == CellType::Wall) {
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
    // Implementa a equação de Bellman conforme mostrado no exercício
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
    // Construtor: inicializa o grid conforme especificado no exercício
    GridWorld() {
        // Inicializa todas as células como normais
        for (auto& row : grid) {
            row.fill(CellType::Normal);
        }

        // Define os estados especiais conforme exercício
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
    // Implementa o processo iterativo mostrado no exercício
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
                        // Similar ao processo manual do exercício
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
    // Formato similar ao mostrado no exercício
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
    // Usando setas como no exercício
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

Em resumo, resolver o **Grid World** usando **MDP** envolve:

1. Definir formalmente o problema como um **MDP** (estados, ações, transições, recompensas).
2. Usar as Equações de Bellman para expressar a relação entre os valores dos estados/ações.
3. Aplicar um algoritmo de **RL** (Programação Dinâmica, Monte Carlo, Diferença Temporal) para encontrar a política ótima $\pi^\*$ que maximiza a recompensa total esperada.

