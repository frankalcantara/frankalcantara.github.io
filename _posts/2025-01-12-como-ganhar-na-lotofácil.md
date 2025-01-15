---
layout: post
title: Como Ganhar na LotoFácil Usando Inteligência Artificial
author: Frank
categories:
    - Inteligência Artificial
    - Matemática
    - Generative Adversarial Network
tags:
    - algoritmos
    - eng. software
    - Generative Adversarial Network
    - programação
    - Python
image: assets/images/money.webp
featured: false
rating: 5
description: A descrição de um algoritmo usando uma GAN para gerar palpites para jogar na Lotofácil.
date: 2025-01-12T20:56:51.519Z
preview: Não. Você não vai ganhar na Lotofácil depois de ler este artigo. Porém, pode ser que aprenda alguma coisa sobre redes neurais. Este artigo usa uma versão simples de uma GAN. Mas, mesmo simples, tem um semestre de tecnologia explicitado.
keywords: |
    Artificial Inteligence
    Generative Adversarial Network
    Python
toc: true
published: true
beforetoc: ""
lastmod: 2025-01-15T13:32:39.765Z
slug: como-ganhar-na-lotofacil-usando-inteligencia-artificial
---

Image by <a href="https://nv-sana.mit.edu/">Mit Sana</a>

Você não vai aprender uma receita mágica, um truque ou uma mandinga que faça você ganhar na Lotofácil. Até onde sabemos, os números sorteados são completa e absolutamente aleatórios. Isso é o que a ciência, a matemática e os analistas da Caixa Econômica vão morrer dizendo. E, por mais que me doa dizer isso, esta é a verdade.

Pelo menos por enquanto. Uma forma de melhorar suas chances em um sorteio completamente aleatório é uma mosca azul do estudo da estatística. A mosca azul me lembrou o cisne negro.

Até 1606, mais ou memos, cisne negro era a expressão preferida para fazer um eufemismo sobre algo impossível. Aí os europeus chegaram na Austrália e encontraram um continente chio de cisnes negros. Até 1606 Em 1960, era impossível ir a lua. Em 2005 era impossível foguete dar ré.

Não, eu não acredito que seja possível encontrar um algoritmo para melhorar as chances de ganhar na Lotofácil. Na verdade, acho que isso é impossível. E, justamente por ser impossível é que precisamos fazer.

## Um jogo fácil

AS probabilidades de acertar na Lotofácil são muito maiores que as probabilidades de ganhar na Megasena, por exemplo.

| Prêmio | Mega-Sena | Quina | Lotofácil |
|---|---|---|---|
| 1º Prêmio | Sena (6 números) | Quina (5 números) | 15 números |
| Probabilidade | 1 em 50.063.860  | 1 em 24.040.016  | 1 em 3.268.760  |
| 2º Prêmio | Quina (5 números) | Quadra (4 números) | 14 números |
| Probabilidade | 1 em 154.518 | 1 em 66.185  | 1 em 24.035 |
| 3º Prêmio | Quadra (4 números) | Terno (3 números) | 13 números |
| Probabilidade | 1 em 2.332 | 1 em 866  | 1 em 601 |

Mesmo sendo tão mais fácil, até 20 de dezembro de 2024, eu consegui apenas 11 pontos em um cartão, em todos que eu joguei. Em minha defesa, eu não jogo muito. Uma vez ou outra, quando a Megasena passa de 50 milhões acumulados, eu jogo um ou dois cartões, aposta mínima na Lotofácil.

No dia 20 de dezembro de 2024, eu rodei este script pela primeira vez, joguei dez cartões acertei 1 de 12 pontos e quatro de 11 pontos. *Isso não quer dizer nada. Nenhuma garantia, nenhum indicativo, nada*. Nada! Pura coincidência! Mas, eu fiquei bem feliz. E curioso. E atrevido. Aquela pulga atrás a orelha dizendo: será?

Meu histórico: tenho acertado 11, ou 12 números, todas as vezes que jogo depois do dia 20 de dezembro. Joguei 300 cartões em 10 dias com 5 cartões de 12 e 11 cartões de 11 pontos em dias diferentes. Eu jogo, quando jogo, 10 cartões, dez cartões de uma vez. Além disso, todas as vezes que vou jogar modifico um pouco a rede neural que criei.

Estou perdendo, sem fazer muitas contas, joguei aproximadamente R$300,00 e tem R$126,79 em uma conta que criei só para manter os valores dos prêmios ganhos. *Estou perdendo, feito.* Pensa em um investimento ruim. Entretanto, coincidências me irritam.

O Gibbs disse que coincidências não existem, se não me engano é a regra 39. Então, vamos estudar isso:

## Cálculo de Probabilidades na Lotofácil

Para calcular as chances de obter exatamente 11, 12, 13 ou 14 pontos na Lotofácil ao jogar 10 cartelas diferentes, cada uma com 15 números, utilizamos a fórmula da distribuição hipergeométrica. Eu considerei que todos os cartões tem exatamente 15 números, a aposta mais barata.

A probabilidade de obter exatamente $k$ acertos (pontos) em uma única cartela é dada por:

$$
\begin{equation}
P(X=k) = \frac{\binom{15}{k} \times \binom{10}{15-k}}{\binom{25}{15}}
\end{equation}
$$

Onde $\binom{n}{k}$ é a combinação de $n$ elementos tomados $k$ a $k$.

### Cálculos para cada prêmio

1. Para $k = 11$:

$$
P(X=11) = \frac{\binom{15}{11} \times \binom{10}{4}}{\binom{25}{15}} \approx 0,000344 \text{ ou } 0,0344\%
$$

1. Para $k = 12$:

$$
P(X=12) = \frac{\binom{15}{12} \times \binom{10}{3}}{\binom{25}{15}} \approx 0,000115 \text{ ou } 0,0115\%
$$

1. Para $k = 13$:

$$
P(X=13) = \frac{\binom{15}{13} \times \binom{10}{2}}{\binom{25}{15}} \approx 0,000024 \text{ ou } 0,0024\%
$$

1. Para $k = 14$:

$$
P(X=14) = \frac{\binom{15}{14} \times \binom{10}{1}}{\binom{25}{15}} \approx 0,000003 \text{ ou } 0,0003\%
$$

Eu jogo 10 cartões de 15 de cada vez. Será que isso tem algum impacto na minha chance de ganhar?

## Considerando 10 Cartões Diferentes Em cada Aposta

Ao jogar 10 cartões diferentes, a probabilidade de obter pelo menos uma cartela com exatamente $k$ pontos é aproximadamente 10 vezes a probabilidade de uma única cartela obter $k$ pontos, assumindo que todos os números e cartões são independentes entre si, que é o mínimo de razoabilidade que podemos pensar.

1. Para $k = 11$:

    $$
    P(\text{pelo menos uma cartela com 11 pontos}) \approx 10 \times 0,000344 = 0,00344 \text{ ou } 0,344\%
    $$

2. Para $k = 12$:

    $$
    P(\text{pelo menos uma cartela com 12 pontos}) \approx 10 \times 0,000115 = 0,00115 \text{ ou } 0,115\%
    $$

3. Para $k = 13$:

    $$
    P(\text{pelo menos uma cartela com 13 pontos}) \approx 10 \times 0,000024 = 0,00024 \text{ ou } 0,024\%
    $$

4. Para $k = 14$:

    $$
    P(\text{pelo menos uma cartela com 14 pontos}) \approx 10 \times 0,000003 = 0,00003 \text{ ou } 0,003\%
    $$

## Analisando os Meus Resultados

Em 300 cartões eu acertei: 5 cartões de 12 e 11 cartões de 11 pontos. Para entender isso, eu larguei a calculadora, abri o VSCode e o claude.ai e providenciei um código em Python para calcular a chance de ter obtido os resultados que tive. O cálculo considera:

- 0 cartão com 13 pontos;
- 5 cartões com 12 pontos;
- 11 cartões com 11 pontos;
- 84 cartões com 10 pontos ou menos.

As probabilidades individuais usadas são:

1. $P(13\text{ pontos}) = 2.4 \times 10^{-5}$;
2. $P(12\text{ pontos}) = 1.15 \times 10^{-4}$;
3. $P(11\text{ pontos}) = 3.44 \times 10^{-4}$;
4. $P(\text{outros}) = 1 - P(13) - P(12) - P(11)$.

O cálculo usa a fórmula multinomial:

$$
\begin{equation}
P(X_1=n_1,\ldots,X_k=n_k) = \frac{n!}{n_1!\cdots n_k!}p_1^{n_1}\cdots p_k^{n_k}
\end{equation}
$$

na qual $n$ é o total de cartões e $p_i$ são as probabilidades individuais.

Código em Python para o cálculo, totalmente gerado pelo Claude.ai:

```python
import math
from decimal import Decimal, getcontext

# Configurando precisão para cálculos decimais
getcontext().prec = 100

def calc_single_probabilities():
   """Calcula as probabilidades individuais para cada quantidade de pontos"""
   def combinations(n, k):
       return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
   
   total = combinations(25, 15)
   
   p11 = Decimal(combinations(15, 11) * combinations(10, 4)) / Decimal(total)
   p12 = Decimal(combinations(15, 12) * combinations(10, 3)) / Decimal(total)
   p13 = Decimal(combinations(15, 13) * combinations(10, 2)) / Decimal(total)
   
   return p11, p12, p13

def multinomial_probability(n, outcomes):
   """
   Calcula a probabilidade multinomial
   n: número total de cartões
   outcomes: dicionário com número de cartões para cada resultado
   """
   p11, p12, p13 = calc_single_probabilities()
   p_other = 1 - (p11 + p12 + p13)
   
   numerator = math.factorial(n)
   denominator = 1
   for k in outcomes.values():
       denominator *= math.factorial(k)
   
   coef = Decimal(numerator) / Decimal(denominator)
   
   prob = coef * \
          Decimal(p11) ** outcomes['11'] * \
          Decimal(p12) ** outcomes['12'] * \
          Decimal(p_other) ** outcomes['other']
   
   return float(prob)

# Configuração dos resultados
n = 300  # total de cartões
outcomes = {
   '11': 11,    # cartões com 11 pontos
   '12': 6,     # cartões com 12 pontos
   '13': 0,     # cartões com 13 pontos
   'other': 283 # cartões restantes
}

# Cálculo da probabilidade
probability = multinomial_probability(n, outcomes)
```

$$
P(\text{6 cartões de 12, 13 de 11}) \approx 6.31 \times 10^{-204}
$$

Em formato decimal completo:

$$
\begin{align*}
& 0.0000000000000000000000000000000000000000000000000000000000000000000\\
& 000000000000000000000000000000000000000000000000000000000000000000000\\
& 000000000000000000000000000000000000000000000000000000000000000000000\\
& 000000631%
\end{align*}
$$

Um número também conhecido como zero. Uau! que número lindo!

*Agora sim! Agora podemos dizer, sem sombra de dúvida, que isso não significa nada. É só coincidência. Nada além de coincidência*.

O resultado, parece indicar que o algoritmo que eu fiz aumentou, por alguma razão, a chance de acertar. Mas, isso *não é verdade*.

A probabilidade $P(\text{6 cartões de 12, 13 de 11}) \approx 1.87 \times 10^{-205}$ que encontramos não prova nada quanto a minha chance de ganhar. Apenas mostra quão raro é acontecer a combinação de vitórias e derrotas que tive até agora. Nada além disso e só isso. Mas, se olhar descuidadamente, parece que consegui algo impossível. É com esse tipo de mentira estatística que estão sendo vendidos milhares de métodos seguros para ganhar na Lotofácil.

*Não é o caso. Aqui, não há um método seguro, não há nem um método*. Existe apenas um estudo dos limites do possível. Ainda assim, vou continuar estudando esse problema, mudando o código, testando e apostando.

Quase esqueci. O código.

## O código

Entre os milhares de algoritmos e arquiteturas disponíveis para implementar alguma coisa em inteligência artificial eu escolhi criar uma GAN, *Generative Adversarial Network*. Uma GAN, usa duas redes neurais, uma das quais treinamos com um conjunto de dados, neste caso, os resultados da Lotofácil baixados direto do site da caixa, todos os dias automaticamente. Essa primeira rede, aida durante o seu treinamento, irá treinar uma segunda rede. A segunda rede recebe um ruído, um conjunto de valores totalmente aleatórios e gerar valores que façam parte do conjunto de treinamento.

A arquitetura escolhida é relativamente leve, comparada às GANs profundas usadas na geração de imagens. Mas, foi necessário a adição de pontuação de similaridade e validação das sequências geradas ajuda a garantir a qualidade da saída. O que tornou as coisas um pouco mais complicadas. Mesmo com os recursos de inteligência artificial que temos hoje, Levei quase uma semana para definir a arquitetura.

A GAN, por enquanto, está composta de duas redes treinam simultaneamente: o Gerador cria dados a partir de ruído enquanto o Discriminador aprende a distinguir dados reais de dados gerados. Juntas elas melhoram o resultado usando o treinamento adversarial.

Até o momento cheguei a:

1. **Rede Geradora**:

    ```python

    class Generator(nn.Module):
       def __init__(self, latent_dim=50):
           # Arquitetura:
           # Entrada (50) -> 128 -> 256 -> 128 -> 25
           # Usa BatchNorm e LeakyReLU
    ```

    Essa classe recebe um vetor de ruído de dimensão 50 (espaço latente), usa um estrutura especial para garantir geração de números únicos. Preciso melhora isso, quando gero 50.000 cartões, alguns são repetidos, mais de 300 e menos de 1000. Como é feito com correções de pesos e erros ainda não acertei o ponto.

    Essa classe também implementa o algoritmo de Gumbel-Softmax para saída discreta.

    O *Gumbel-Softmax é uma técnica utilizada para gerar amostras categóricas diferenciáveis em redes neurais*, especialmente útil quando precisamos trabalhar com variáveis discretas em um contexto que requer backpropagation.

    O algoritmo Gumbel-Softmax combina dois elementos: a distribuição Gumbel, que é usada para adicionar ruído que permite amostragem de variáveis categóricas, e a função softmax, que produz uma distribuição de probabilidade suave sobre categorias. Provavelmente é isso que está causando meus repetidos. Mas, eu precisei disso para gerar apenas números inteiros, no espaço da Lotofácil.

    Durante o *forward-pass*, o Gumbel-Softmax aproxima uma amostragem one-hot usando um parâmetro de temperatura $\tau$ que controla o quão próxima a aproximação está de uma distribuição categórica verdadeira - quando $\tau \to 0$, a saída se aproxima de um vetor one-hot, e quando $\tau \to \infty$, a saída se torna mais suave.

    O $\tau$ parâmetro de temperatura que está sendo usado para controlar a nitidez da amostragem. Na próxima versão vou colocar outra rede neural neste código, para treinar o próprio $\tau$.

    A classe Generator usa *BatchNorm, Batch Normalization*, normalização em lotes. Uma técnica que normaliza as ativações de uma camada neural, subtraindo a média do lote, *batch*, e dividindo pelo desvio padrão ($\frac{x - \mu}{\sigma}$). Após esta normalização, o BatchNorm aplica uma transformação linear com parâmetros treináveis ($\gamma$ e $\beta$) que permitem à rede aprender a escala e o deslocamento ideais.

    A ideia, como meu servidor não dispõem de GPU, é conseguir ganhar alguma velocidade de treinamento, com taxas de aprendizado maiores, otimizando a GAN para computação em CPU. A ideia não é minha, essa técnica é comum em GAN's porque tende a estabilizar o treinamento.

    Outro problema que eu ainda tenho. Testei algumas funções de ativação e ainda não escolhe a melhor delas. Talvez não haja uma melhor. Ainda não sei. Preciso testar outras e pesquisar um pouco mais sobre isso.  

    Por enquanto estou usando a *LeakyReLU (Leaky Rectified Linear Unit)*. Uma variante da *ReLU* que, em vez de zerar todos os valores negativos, permite um pequeno gradiente negativo (geralmente 0.01).Formalmente:

    $$
    \begin{equation}
    f(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha x & \text{se } x \leq 0 \end{cases}
    \end{equation}
    $$

    neste caso, $\alpha$ é um pequeno valor positivo, o "leak" do nome. Eu escolhi essa função de ativação depois que descobri que a *ReLU* mata neurônios. Na ReLU padrão, neurônios podem parar de aprender se sempre produzirem ativações negativas. No contexto da GAN, o LeakyReLU é útil porque ajuda a manter gradientes não-nulos durante o treinamento, permitindo um fluxo mais suave de informação através da rede, o que é crítico para o processo adversarial de treinamento. Além disso, está disponível na *torch*:

    ```python
    import torch
    import torch.nn as nn

    # Criando uma camada LeakyReLU com coeficiente de vazamento 0.01
    leaky_relu_layer = nn.LeakyReLU(negative_slope=0.01)

    # Aplicando a camada a um tensor de entrada
    input_tensor = torch.tensor([-10, -5, 0.0, 5, 10])
    output_tensor = leaky_relu_layer(input_tensor)

    # Imprimindo o resultado
    print(output_tensor)  # tensor([-0.1000, -0.0500,  0.0000,  5.0000, 10.0000])
    ```

2. **Rede Discriminadora**:

    ```python
    class Discriminator(nn.Module):
       def __init__(self):
           # Arquitetura:
           # Entrada (15) -> 128 -> 256 -> 128 -> 1
           # Usa Dropout e LeakyReLU</code>
    ```

    E esta rede que recebe as 15 dimensões que importamos dos resultados da Lotofácil. Essa rede usa *dropout*, $0.3$ e uma função de ativação simples: *sigmoid*.

    O *dropout* é a primeira opção para prevenir o *overfitting* durante o treinamento da rede neural. Durante cada passo de treinamento, cada neurônio tem uma probabilidade $p$, no código da GAN, $p=0.3$ ou 30%, de ser temporariamente desligado, *dropout*. Neste caso, sua saída é multiplicada por zero. Matematicamente, para cada neurônio com valor $x$, a saída será dada por:

    $$
    \begin{equation}
    y = \begin{cases} 0 & \text{com probabilidade } p \\ \frac{x}{1-p} & \text{com probabilidade } 1-p \end{cases}
    \end{equation}
    $$

    A divisão por $1-p$ mantém a expectativa do valor de saída constante. Em redes GAN, o *dropout* pode ser usado apenas no discriminador para evitar que ele prejudique o treinamento do gerador.

    A função Sigmoid é uma função de ativação que mapeia qualquer número real para o intervalo $(0,1)$. Sua fórmula é:

    $$
    \begin{equation}
    \sigma(x) = \frac{1}{1 + e^{-x}}
    \end{equation}
    $$

    Ela está sendo usada na última camada do discriminador porque precisamos de uma saída que possa ser interpretada como uma probabilidade, o discriminador precisa estimar a probabilidade de uma amostra ser real ou gerada. A Sigmoid é ideal para esta tarefa porque sua saída está limitada entre 0 e 1. A função (3) é diferenciável o que é útil em sistemas como *backpropagation*.

    Essa é a minha função de ativação preferida. É possível dominar o domínio da *sigmoid* no domínio da probabilidade.

Essa é a classe principal. O script é um pouco mais complexo que isso:

1. **Manipulação de Dados**: temos uma função para manipular os dados dos resultados:

    ```python
    def load_and_split_data(csv_path, test_size=100):
        # Normaliza valores para o intervalo [0,1]
        train_tensor = (train_tensor - 1) / 24.0</code>
    ```

2. **Processo de Treinamento**: no treinamento estou usando o  otimizador Adam com $\beta_1=0.5$, $\beta_2=0.999$; implementa a função de perda padrão e inclui a verificação de números únicos por cartão.

3. **Pós-processamento**:

   ```python
    def denormalize_and_discretize(tensor):
        denorm = tensor * 24.0 + 1.0
        return torch.round(denorm).clamp(1, 25)</code>
    ```

Para melhorar a qualidade dos dados gerados precisaríamos de um arquivo com uns 2 bilhões de linhas ou, mais ou menos, 700.000 anos de sorteios. Exceto, é claro, que se descubra algo novo na matemática ou na física.

Talvez, em algum momento, eu poste o código inteiro, detalhado. Mas, não agora.

## Como isso está indo

Eu resolvi automatizar as coisas. Todos os dias, exceto domingo e segunda, eu baixo o arquivo de resultados da caixa, treino a rede, gero 50.000 cartões. Passo estes cartões por outra rede com uma arquitetura diferente e retiro os 10.000 mais próximos em nenhuma ordem específica. Logo em seguida, os 10 primeiros destes 10.000 são enviados para o meu e-mail.

Outra coisa importante. Antes de treinar a GAN, gerar 50.000, escolher 10.000 e enviar e-mails. O sistema pega o resultado do sorteio de ontem e comparar com os últimos 50.000 cartões gerados. O resultado desta comparação está na tabela a seguir. Nesta tabela, a coluna última geração, contém o dia que a GAN rodou. A coluna versão GAN será automaticamente alterada todas as vezes que eu modificar o código da GAN.

<div id="table-container"></div>

Nesta tabela, tudo parece ok, exceto o dia 09/01/2025. Neste dia, eu causei um bug no script que faz a estatística que contava o número de acertos de 14 pontos em dobro. Deixei na tabela apenas para manter o histórico independente da GAN. Outra coisa, pode ser que apareçam várias linhas no mesmo dia. Neste dia, eu terei rodado o sistema todo mais de uma vez.

Eu sei, eu sei. Uma hora destas vai aparecer um $1$ na coluna dos 15 acertos e não terá sido um dos dez que eu jogo. Neste dia, eu vou chorar!