# Notação Padrão para Reinforcement Learning

## Fundamentos de Processos Estocásticos

### Cadeias de Markov (Notação Tradicional)

- **Variáveis aleatórias**: $X_1, X_2, ..., X_n$ representando estados em diferentes instantes de tempo
- **Probabilidades de transição**: $P_{ij} = P(X_{n+1} = j \vert   X_n = i)$
- **Propriedade de Markov**: $P(X_{n+1} = x_{n+1} \vert   X_n = x_n, X_{n-1} = x_{n-1}, ..., X_1 = x_1) = P(X_{n+1} = x_{n+1} \vert   X_n = x_n)$
- **Matriz de transição**: $P = [P_{ij}]$ onde cada elemento representa a probabilidade de transição do estado $i$ para o estado $j$

## Notação para Reinforcement Learning (MDP)

### Elementos Básicos

- **Estados**: conjunto $S$, com estados individuais $s, s' \in S$
- **Estado inicial**: $s_0$
- **Estados terminais**: $S_{terminal} \subset S$
- **Ações**: conjunto $A$ ou $A(s)$ (ações disponíveis no estado $s$), com ações individuais $a, a' \in A$
- **Função de transição**: $P(s' \vert   s, a)$ - probabilidade de transitar para o estado $s'$ ao executar a ação $a$ no estado $s$
- **Função de recompensa**: $R(s, a, s')$ - recompensa por transitar do estado $s$ para $s'$ através da ação $a$
- **Recompensa simplificada**: $R(s)$ - quando a recompensa depende apenas do estado (usado em casos específicos)
- **Fator de desconto**: $\gamma \in [0,1]$ - determina a importância de recompensas futuras

### Políticas e Funções Valor

- **Política**: $\pi(a \vert   s)$ - probabilidade de escolher a ação $a$ no estado $s$
- **Política determinística**: $\pi(s)$ - ação a ser tomada no estado $s$
- **Política ótima**: $\pi^*(s)$ ou $\pi^*$
- **Função valor-estado**: $V^\pi(s)$ - valor esperado seguindo política $\pi$ a partir do estado $s$
- **Função valor-ação**: $Q^\pi(s, a)$ - valor esperado executando ação $a$ no estado $s$ e seguindo $\pi$ depois
- **Função valor ótima**: $V^*(s)$ e $Q^*(s, a)$

### Equações de Bellman

- **Equação de Bellman para** $V^\pi$:
$$V^\pi(s) = \sum_{a \in A} \pi(a \vert   s) \sum_{s' \in S} P(s' \vert   s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

- **Equação de Bellman para** $Q^\pi$:
$$Q^\pi(s, a) = \sum_{s' \in S} P(s' \vert   s, a) [R(s, a, s') + \gamma \sum_{a' \in A} \pi(a' \vert   s') Q^\pi(s', a')]$$

- **Equação de Bellman de otimalidade para** $V^*$:
$$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s' \vert   s, a) [R(s, a, s') + \gamma V^*(s')]$$

- **Equação de Bellman de otimalidade para** $Q^*$:
$$Q^*(s, a) = \sum_{s' \in S} P(s' \vert   s, a) [R(s, a, s') + \gamma \max_{a' \in A} Q^*(s', a')]$$

### Algoritmos

- **Iteração de Valor**:
$$V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} P(s' \vert   s, a)[R(s, a, s') + \gamma V_k(s')]$$

- **Iteração de Política**:
  - Avaliação: $V^{\pi_k}(s) = \sum_{s' \in S} P(s' \vert   s, \pi_k(s)) [R(s, \pi_k(s), s') + \gamma V^{\pi_k}(s')]$
  - Melhoria: $\pi_{k+1}(s) = \arg\max_{a \in A} \sum_{s' \in S} P(s' \vert   s, a) [R(s, a, s') + \gamma V^{\pi_k}(s')]$

## Casos Específicos e Extensões

### Para o Problema de Manutenção (Artigo 5)

- **Intervalos de inspeção**: $t \in T_s$, tempo entre observações
- **Função de recompensa estendida**: $R(s, a, s', t)$ - incorpora o intervalo de tempo
- **Componentes de custo**: $C_{immediate}(s, a, s')$, $C_{time}(s, t)$, $C_{insp}(s)$, $C_{ação}(a)$, $C_{falha}(s')$
- **Função valor com otimização de intervalo**:
$$V^*(s) = \min_{a \in A_s, t \in T_s} \left[ R(s, a, t) + \gamma \sum_{s' \in S} P(s' \vert   s, a) V^*(s') \right]$$

## Recomendações para Consistência

1. Manter notação $s, s'$ para estados e $a, a'$ para ações em todo o texto sobre RL
2. Usar barra vertical ($ \vert  $) para probabilidades condicionais: $P(s' \vert   s, a)$ em vez de $P(s'|s,a)$
3. Diferenciar claramente entre $\pi(a \vert   s)$ (política estocástica) e $\pi(s)$ (política determinística)
4. Ao estender o MDP com novas dimensões (como intervalos $t$), documentar explicitamente as alterações na notação