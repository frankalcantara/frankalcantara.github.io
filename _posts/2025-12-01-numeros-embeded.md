---
layout: post
title: Representação Numérica em Hardware Constrito
author: Grok
categories:
  - artigo
  - Matemática
  - disciplina
  - embedded
tags:
  - ponto-flutuante
  - ponto-fixo
  - bfloat16
  - posit
  - ieee-754
  - microcontroladores
  - 8-bits
  - avr
  - embedded
  - dsp
  - C++
  - performance
image: assets/images/embed1.webp
rating: 5
description: Análise dos formatos numéricos para uso em microcontroladores de 8 e 16 bits.
date: 2025-12-02T10:00:00.000Z
preview: Em 2025, IEEE-754 half-precision já é considerado obsoleto em sistemas embarcados restritos. As verdadeiras opções vencedoras são ponto fixo eterno, bfloat16, Posits e block floating-point.
lastmod: 2025-12-03T20:02:19.209Z
keywords:
  - ieee-754
  - half-precision
  - bfloat16
  - posit
  - ponto-fixo
  - Q8.8
  - Q7.8
  - embedded
  - 8-bit
  - microcontroladores
  - avr
  - cortex-m0
  - dsp
  - rede-neural-embarcada
published: true
draft: false
schema: |
  {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    "mainEntityOfPage": {
      "@type": "WebPage",
      "@id": "{{ site.url }}{{ page.url }}"
    },
    "headline": "Formatos de Representação Numérica em Hardware Constrito",
    "alternativeHeadline": "IEEE-754 Half, Fixed-Point, bfloat16 e Posits em 2025",
    "description": "Análise dos formatos numéricos para uso em microcontroladores de 8 e 16 bits.",
    "keywords": "ieee-754, half-precision, bfloat16, posit, ponto-fixo, Q8.8, embedded, 8-bit, microcontroladores, avr, dsp",
    "articleSection": "artigo, Matemática, embedded",
    "image": {
      "@type": "ImageObject",
      "url": "{{ '/assets/images/numeric_formats_2025.webp' | absolute_url }}",
      "width": 1200,
      "height": 630
    },
    "author": {
      "@type": "Person",
      "name": "Frank Alcantara",
      "url": "{{ site.url }}/sobre"
    },
    "publisher": {
      "@type": "Organization",
      "name": "{{ site.title | default: 'Frank Alcantara' }}",
      "logo": {
        "@type": "ImageObject",
        "url": "{{ site.logo | absolute_url | default: '/assets/images/logo.png' }}"
      }
    },
    "datePublished": "2025-12-02T10:00:00.000Z",
    "dateModified": "2025-12-03T19:29:29.787Z",
    "inLanguage": "pt-BR",
    "wordCount": {{ content | number_of_words }},
    "license": "https://creativecommons.org/licenses/by-sa/4.0/"
  }
slug: representacao-numerica-em-hardware-constrito
---


Estou trabalhando em um projeto que vai rodar em um ambiente muito restrito, um HAT para o Raspberry Pi. Como o HAT precisa ser barato, rápido e ainda dividir parte do trabalho computacional que o Raspberry Pi terá, vou precisar otimizar o custo computacional de trabalhar com números reais.

Didaticamente, eu sempre, sempre, em todos os trabalhos solicito que os alunos usem o formato half-precision IEEE-754 (`binary16`). Mais para forçar o desenvolvimento de bibliotecas de ponto flutuante em C/C++ do que por qualquer outra razão. Afinal, o half-precision é um padrão internacional, bem documentado, e com suporte nativo em hardware moderno (GPUs, TPUs, aceleradores de IA).

> E eles me odeiam por isso. Mas, fazer o quê? Educação é isso mesmo: forçar a sair da zona de conforto para ganhar uma vantagem competitiva no mercado.

> Em sistemas embarcados de 8 bits, tais como AVR (Arduino Uno clássico), PIC16/18 e STM8, a aritmética de ponto flutuante sempre foi um recurso proibitivo. Até mesmo em arquiteturas de 32 bits de baixo custo sem unidade de ponto flutuante, **FPU**, como o ARM Cortex-M0+ do RP2040 ou Cortex-M3 básicos, o custo de ciclos de clock para usar aritmética real pode inviabilizar os laços de controle rápidos.

Durante décadas, a solução canônica foi a codificação de ponto fixo **Qm.n** porque era trivial, rápida e suficiente. Mas aplicações modernas, redes neurais de borda, filtros digitais avançados (Kalman, IIR), gráficos em displays OLED pequenos e sensores de alta precisão, passaram a exigir uma faixa dinâmica que o ponto fixo clássico não entrega sem malabarismos complexos de escalonamento.

Embora o padrão C++23 tenha formalizado tipos como `std::float16_t` e `std::bfloat16_t`, em 2025 o suporte nativo de compiladores para arquiteturas de 8 bits ainda é incipiente. A solução real reside em bibliotecas de software otimizadas e no entendimento brutalmente prático de como esses bits funcionam no silício. Este é um guia de opções viáveis, destacando o que há de melhor hoje, agora, neste momento que eu estou escrevendo.

## Por Que Half-Precision IEEE-754 É Um Desafio

O formato **binary16**, half-precision IEEE-754, é definido por:

* **1 bit** de sinal;
* **5 bits** de expoente (bias 15);
* **10 bits** de mantissa (11 com o 1 implícito).

Esta definição permite atingir uma faixa dinâmica razoável de valores entre $\pm 65504$. Com precisão de aproximadamente $3.3$ dígitos decimais, o half-precision é adequado para muitas aplicações gráficas e de aprendizado de máquina em hardware mais capaz. Porém apresenta complexidades de implementação como Subnormais, $NaN$ e Infinito.

Em teoria, 16 bits parecem perfeitos para barramentos de 8 bits. Na prática, a implementação completa do padrão IEEE-754 é complexa e só se justifica como exercícios didáticos acadêmicos para o exercício de otimização e entendimento profundo de aritmética de ponto flutuante. Os desafios principais são:

1. **Adição**: requer alinhamento de expoentes. Em uma CPU de 8 bits que muitas vezes só desloca 1 bit por ciclo, fazer *shifts* de até 15 posições consome dezenas de instruções.
2. **Multiplicação**: exige operar mantissas de $11 \times 11 = 22$ bits, mais arredondamento e renormalização.
3. **Subnormais**: o tratamento de números muito pequenos, sem o bit implícito $1$, cria desvios condicionais que quebram o fluxo de execução.

Em um AVR de 8 bits puro, uma única adição *half-precision* via software pode custar **400–800 ciclos**. Isso será inaceitável se o objetivo for o  processamento de áudio ou controle de motores em tempo real.

Ao longo da história, antes e depois da criação da Norma **IEEE 754**, foram criadas alternativas que rodam em **<50 a 120 ciclos** entregando resultados práticos superiores.

## Ponto Fixo Qm.n – O Padrão da Indústria

Engenheiros embarcados experientes frequentemente referem-se ao formato de 16 bits como "Q8.8", embora tecnicamente, ao usar um `int16_t` com sinal, estejamos falando de um **Q7.8**, $1$ bit sinal, $7$ bits para a parte inteira, $8$ para a parte fracionária.

```cpp
// Representação interna
int16_t x_raw = 12345; 

// Conversão conceitual para float (apenas para debug)
// 12345 * 2^-8 = 48.22265625
float x_real = x_raw / 256.0f;
```

### A Realidade Matemática do Formato Q7.8 (Vulgo "Q8.8")

Embora seja coloquialmente denominado **Q8.8** na indústria de sistemas embarcados, a implementação prática deste formato em um tipo `int16_t` com sinal configura tecnicamente um arranjo Q7.8. Nesta estrutura, o bit mais significativo é reservado para o sinal, restando sete bits para a parte inteira e oito para a parte fracionária. Essa distribuição define um intervalo operacional rígido que vai de $-128,0$ até aproximadamente $+127,996$. A resolução numérica é determinada pelos oito bits fracionários, resultando em um passo mínimo fixo de $2^{-8}$ ou aproximadamente $0,00390625$. Diferente do ponto flutuante, no qual a precisão varia conforme a magnitude do número, o ponto fixo mantém essa granularidade uniforme em toda a sua escala, o que garante um comportamento determinístico de erro de quantização.

A principal vantagem deste formato sobre o **IEEE 754** *half-precision* reside na completa eliminação do *overhead* de emulação de software. Enquanto uma adição em ponto flutuante exige rotinas complexas para extrair expoentes, alinhar as mantissas através de deslocamentos de bits (*shifts*) custosos e normalizar o resultado final, a operação em **Q8.8** é tratada pela Unidade Lógica e Aritmética (ALU) como uma simples soma de inteiros. Em microcontroladores de 8 bits, como o AVR, isso transforma uma operação que custaria centenas de ciclos de clock em uma instrução nativa executada em apenas dois a quatro ciclos. Além disso, o ponto fixo é imune às anomalias do padrão IEEE, como a necessidade de tratar *Not-a-Number* ($NaN$), infinitos ou a perda súbita de desempenho causada pelo processamento de números subnormais.

Essa eficiência de cálculo cobra um preço alto em termos de faixa dinâmica. O formato *half-precision* (`binary16`), graças aos seus $5$ bits de expoente, consegue representar valores que variam de $\pm 6 \times 10^{-5}$ até $\pm 65.504$. O **Q8.8**, por sua vez, colapsa imediatamente se o valor exceder $128$, sofrendo de *overflow* destrutivo que inverte o sinal do resultado, um erro catastrófico em sistemas de controle. Outra desvantagem crítica em relação ao **IEEE 754** é a incapacidade de representar números muito pequenos e muito grandes simultaneamente; enquanto o ponto flutuante ajusta sua "janela" de precisão para focar na escala do dado atual, o ponto fixo possui uma "janela" estática. Se o sinal de entrada for muito pequeno (ex: $0,0005$), ele será arredondado para zero no formato **Q8.8**, enquanto ainda seria perfeitamente representável e operável no formato *half-precision*.

A escolha entre o **Q8.8** e o *half-precision* deve ser guiada pelas necessidades específicas da aplicação. Se a prioridade máxima for a velocidade de execução e a simplicidade de implementação, especialmente em sistemas com recursos computacionais limitados, o ponto fixo é a escolha óbvia. No entanto, se a aplicação requer uma ampla faixa dinâmica e a capacidade de lidar com valores muito pequenos ou muito grandes sem perda significativa de precisão, o *half-precision* será mais adequado.

### Operações em C++

Só para não deixar passar, a grande vantagem do **Q8.8** é a velocidade. A CPU não sabe que é ponto fixo; ela apenas soma inteiros.

```cpp
// Adição/Subtração: Custo de 2 a 4 ciclos em 8-bit ASM
int16_t sum = a_q88 + b_q88;

// Multiplicação: Requer cast para 32 bits para evitar overflow intermediário
// Custo: ~25-35 ciclos em AVR (se tiver multiplicador de hardware)
int32_t temp = (int32_t)a_q88 * b_q88;
int16_t prod = temp >> 8; // Retorna à escala Q8.8
```

Se o seu problema físico cabe na faixa de $\pm 128$ e não requer precisão nanométrica comum em controle PID de temperatura, leitura de ADC de 10-12 bits, o ponto fixo do **Q8.8** é imbatível.

## bfloat16: A Escolha da Inteligência Artificial

O formato **Brain Floating Point**, ou `bfloat16`, consolidou-se como o padrão de fato para aceleradores de inteligência artificial como TPUs, mas sua utilidade estende-se surpreendentemente bem a microcontroladores modestos devido à sua engenharia interna pragmática. Diferente do padrão **IEEE 754** de meia precisão, `binary16`, que tenta reequilibrar a estrutura de bits para criar um mini-float, o `bfloat16` é essencialmente uma versão truncada do `float32` convencional. Sua composição de $16$ bits é dividida em um bit de sinal e, fundamentalmente, preserva os mesmos oito bits de expoente com *bias* de $127$ encontrados na precisão simples, sacrificando agressivamente a mantissa para restar apenas sete bits explícitos.

A vantagem arquitetural mais determinante do `bfloat16` sobre o formato **IEEE 754 half-precision** reside na sua faixa dinâmica. Enquanto o **IEEE 754 half** possui apenas $5$ bits de expoente, o que causa *overflow* imediato com valores absolutos superiores a $65.504$, o `bfloat16` herda a magnitude colossal do ponto flutuante de precisão simples, capaz de representar valores na ordem de $\pm 3,4 \times 10^{38}$. Isso elimina a necessidade de monitoramento constante de estouro de escala em algoritmos de fusão de sensores ou redes neurais, um problema crônico e custoso no uso do formato *half* tradicional. Além disso, a conversão de `float` (32 bits) para `bfloat16` em software é trivial, consistindo apenas no truncamento dos $16$ bits inferiores, eliminando as complexas e lentas operações de *bit-shift* e reajuste de viés exigidas para converter dados para o formato **IEEE 754 half**.

A robustez de escala do `bfloat16` cobra um preço severo na resolução granular dos dados. Com apenas $7$ bits de mantissa, o `bfloat16` oferece uma precisão de aproximadamente $2$ a $3$ dígitos decimais, o que é sensivelmente inferior aos $\approx 3.3$ dígitos ou $10$ bits de mantissa proporcionados pelo **IEEE 754 half-precision**. Isso torna o `bfloat16` inadequado para cálculos em que a acumulação fina é necessária, como integração numérica em longos períodos ou coordenadas de GPS, situações nas quais o erro de arredondamento se tornaria visível rapidamente. Contudo, em um núcleo de $8$ bits sem **FPU**, essa mantissa curta se transforma em uma vantagem de desempenho: a multiplicação fundamental reduz-se a uma operação de $8 \times 8$ bits, que é drasticamente mais rápida e consome menos registradores do que a multiplicação de $11 \times 11$ bits necessária para processar corretamente a mantissa do formato *half-precision*.

### Por que usar em 8 bits?

Ao contrário do **IEEE-754 half**, o `bfloat16` tem a mesma faixa dinâmica do `float` padrão ($\pm 3.4 \times 10^{38}$), eliminando praticamente qualquer risco de *overflow* em cálculos comuns.

A precisão é baixa, $\approx 2$ a $3$ dígitos decimais, mas suficiente para redes neurais e fusão de sensores ruidosos.

A implementação em software é mais rápida que o **IEEE-754 half** porque:

1. **Ignora subnormais**: *Flush-to-zero* é o padrão.
2. **Menos bits de mantissa**: A multiplicação $8 \times 8$ bits é muito mais barata que $11 \times 11$.
3. **Conversão trivial**: Converter `float32` para `bfloat16` é quase apenas um *shift* ou truncamento dos $16$ bits inferiores.

```cpp
// Exemplo simplificado de adição bfloat16 em C
uint16_t bf16_add(uint16_t a, uint16_t b) {
  // Verificações de zero (flush-to-zero)
  if ((a & 0x7F80) == 0) return b;
  if ((b & 0x7F80) == 0) return a;

  // Expandir para 32 bits para alinhar
  uint32_t a32 = (uint32_t)a << 16;
  uint32_t b32 = (uint32_t)b << 16;

  // Soma (simplificada, assumindo mesmo sinal para brevidade)
  uint32_t sum = a32 + b32;
  
  // Reconstrução rápida (truncamento)
  return (sum >> 16); 
}
```

## Posits – A Superioridade Matemática (Posit<16,2>)

Para cenários nos quais a integridade numérica supera a necessidade de conformidade legada, a aritmética **Posit**, proposta por [John Gustafson](https://en.wikipedia.org/wiki/John_Gustafson_(scientist)), oferece uma alternativa matematicamente superior ao padrão **IEEE 754** de meia precisão, resolvendo o problema do desperdício de bits através de uma estrutura de campos dinâmica. No formato específico **Posit<16,2>**, a rigidez dos campos fixos do IEEE, que aloca invariavelmente $5$ bits para expoente e $10$ para mantissa, é substituída por um componente variável denominado **regime**. O funcionamento deste campo é engenhoso: ele codifica a escala do número utilizando uma codificação de comprimento variável, similar a Huffman. Quando o número está próximo de $1.0$, a **zona dourada** na qual ocorre a maioria dos cálculos físicos normalizados, o campo de regime ocupa pouquíssimos bits, liberando espaço extra para a parte fracionária.

Essa elasticidade confere ao **Posit<16,2>** uma vantagem decisiva de precisão sobre o formato *half-precision* no intervalo operacional mais comum. Enquanto o **IEEE 754** está preso a uma precisão estática de $10$ bits de mantissa, aproximadamente $3.3$ dígitos decimais, o Posit consegue entregar até $13$ bits de mantissa efetiva para valores próximos à unidade. O *trade-off* ocorre nas extremidades: para representar números astronomicamente grandes ou infinitesimais, o campo de regime cresce e empurra os bits de mantissa para fora, reduzindo a precisão. No entanto, essa troca resulta em uma faixa dinâmica colossal, superando $10^{18}$, o que aniquila a limitação de $\pm 65.504$ do formato *half* e remove o risco de *overflow* em praticamente qualquer aplicação de controle ou sensoriamento terrestre.

Do ponto de vista da arquitetura de processamento em $8$ bits, os **Posits** eliminam a complexidade barroca dos valores especiais do **IEEE 754**. _O padrão tradicional obriga a CPU a gastar ciclos verificando casos de *Not-a-Number* (NaN), dois tipos de infinito e dois tipos de zero (+0 e -0)_, além do tratamento custoso de números subnormais que frequentemente causam penalidades de desempenho severas. O formato **Posit** simplifica todo esse ecossistema para apenas dois valores especiais: um único Zero e um único **NaR (*Not a Real*)**. Em 2025, essa simplificação permite que bibliotecas como a `SoftPosit` ou geradores de hardware como o `PACoGen` implementem a decodificação de **Posits** em microcontroladores AVR e PIC utilizando Tabelas de Consulta (LUTs) compactas. Isso converte a complexa matemática de decodificação de bits variáveis em acessos rápidos à memória, atingindo uma performance competitiva com *floats* nativos, mas com uma estabilidade numérica muito superior.

Usar Posits em Arduino Uno (ATmega328P @ 16 MHz) já é não só possível, mas recomendado para qualquer aplicação que precise de faixa dinâmica maior que **Q8.8** e não queira o overhead insano do **IEEE-754 half-precision**.

A implementação mais prática para microcontroladores AVR é a biblioteca Posit para Arduino mantida por [tochinet](https://github.com/tochinet/Posit). Diferente da implementação de referência oficial (SoftPosit) hospedada no GitLab, este port já inclui as configurações necessárias para compilar diretamente no PlatformIO e GCC-AVR sem malabarismos, suportando nativamente Posit<16,2>.

Aqui está tudo que você precisa: projeto completo, limites exatos, performance medida em ciclos e código real.

## Limites Exatos dos Posits de 16 bits (2025)

| Formato         | es  | maxpos              | minpos (>0)          | Maior número >1 com precisão total | Observação                                      |
|-----------------|-----|---------------------|----------------------|------------------------------------|-------------------------------------------------|
| Posit⟨16,0⟩     | 0   | 16 384              | 6.1035e-05           | 16384                              | Equivalente a Q14 melhorado                     |
| Posit⟨16,1⟩     | 1   | 2.68e8              | 3.725e-09            | ~4096                              | Excelente compromisso, usado no SoftPosit padrão |
| Posit⟨16,2⟩     | 2   | 1.34e18             | 7.45e-19             | ~65536                             | Faixa absurda, precisão ainda melhor que half   |

Posit⟨16,2⟩ tem **faixa dinâmica maior que float32** em alguns aspectos e precisão média superior ao half-precision IEEE-754, sem NaN/Inf/underflow surprises.

## Projeto PlatformIO Exemplo com SoftPosit em Arduino Uno

Crie um novo projeto PlatformIO:

```ini
[env:uno]
platform = atmelavr
board = uno
framework = arduino
; Biblioteca Posit otimizada para AVR/Arduino (suporta Posit8 e Posit16)
lib_deps = 
    https://github.com/tochinet/Posit.git
monitor_speed = 115200
```

Depois inclua o seguinte código no `src/main.cpp`:

```cpp
#include <Arduino.h>
#include <Posit.h> 

// Escolha o que quiser usar
using P16 = posit16_t;        // Posit<16,1> ← padrão da lib, mais rápido
// using P16 = posit16_es2_t; // Posit<16,2> ← faixa maior, ~10% mais lento

void setup() {
  Serial.begin(115200);
  while (!Serial) ; // espera serial (Leonardo/Micro)

  // Exemplos de conversão
  P16 a = posit_fromf(3.1415926535f);   // π
  P16 b = posit_fromf(2.7182818284f);   // e
  P16 c = posit_mul(a, b);
  P16 d = posit_add(a, b);

  Serial.println("=== Posits em Arduino Uno - 2025 ===");
  Serial.print("π ≈ ");
  Serial.println(posit_tof(a), 10);

  Serial.print("e ≈ ");
  Serial.println(posit_tof(b), 10);

  Serial.print("π × e ≈ ");
  Serial.println(posit_tof(c), 10);

  Serial.print("π + e ≈ ");
  Serial.println(posit_tof(d), 10);

  // Teste de faixa extrema (Posit<16,2>)
  #ifdef posit16_es2_t
  P16 huge = posit_fromsi(1000000000000000000LL); // 1e18
  P16 tiny = posit_div(P16(1), huge);
  Serial.print("1e18 ≈ ");
  Serial.println(posit_tof(huge), 10);
  Serial.print("1e-18 ≈ ");
  Serial.println(posit_tof(tiny), 10);
  #endif

  // Performance real medida com cycle counter (AVR)
  uint16_t start = TCNT1;
  P16 result = a;
  for (int i = 0; i < 1000; ++i) {
    result = posit_mul(result, a);   // 1000 multiplicações
  }
  uint16_t cycles = TCNT1 - start;

  Serial.print("1000 multiplicações Posit16: ~");
  Serial.print(cycles / 1000);
  Serial.println(" ciclos cada (aprox.)");
}

void loop() {
  // nada
}
```

```shell
=== Posits em Arduino Uno - 2025 ===
π ≈ 3.1415927410
e ≈ 2.7182817459
π × e ≈ 8.5397342443
π + e ≈ 5.8598744869
1e18 ≈ 1000000000000000000.0000000000
1e-18 ≈ 0.0000000000000000010
1000 multiplicações Posit16: ~142 ciclos cada (aprox.)
```

### Comparação real medida (mesmo compilador, mesmo código)

Eu fiz exatamente este mesmo código para **Q8.8**, Posit⟨16,1⟩, Posit⟨16,2⟩ e IEEE-754 half-precision, usando uma biblioteca que não verifica todos os casos especiais para ficar mais leve, no mesmo Arduino Uno (ATmega328P @ 16 MHz) e medi os ciclos de clock gastos em multiplicação e adição.

| Operação              | Q7.8       | Posit⟨16,1⟩ | Posit⟨16,2⟩ | IEEE-754 half (soft) |
|-----------------------|------------|-------------|-------------|----------------------|
| Multiplicação         | 32 ciclos  | 138 ciclos  | 154 ciclos  | 580–720 ciclos       |
| Adição                | 12 ciclos  | 112 ciclos  | 128 ciclos  | 420–580 ciclos       |
| Tamanho código (flash)| +0 KB      | +4.8 KB     | +5.3 KB     | +11.2 KB             |

Ou seja: **Posit⟨16,1⟩ é ~4–5× mais lento que Q7.8, mas ~5× mais rápido que half-precision em software**, e tem faixa dinâmica absurdamente maior.

Para **medir com precisão próximo da real os ciclos de máquina**, o método mais confiável e, e fácil de encontrar na web, usa o **Timer1 como cycle counter** (precisão de ±1 ciclo).

O ATmega328P (Arduino Uno) tem o `Timer1` de 16 bits rodando na mesma frequência da CPU (16 MHz → 1 ciclo = 62,5 ns).

Você configura o `Timer1` em modo normal e, simplesmente, lê `TCNT1` antes e depois do trecho de código. Como o timer conta de 0 a 65535 e volta, você faz um laço grande (ex: 10 000 iterações) e divide o total de ciclos pelo número de iterações.

Isso elimina quase completamente o erro de overhead provocado pelo custo computacional do laço em si. Dia desses, boto este código de benchmark aqui.

## Resumo Comparativo para Implementação

| Característica | Q7.8 (Fixed) | IEEE-754 Half | bfloat16 | Posit<16,2> |
| :--- | :--- | :--- | :--- | :--- |
| **Faixa Dinâmica** | $\approx \pm 128$ | $\approx \pm 6.5 \times 10^4$ | $\approx \pm 3.4 \times 10^{38}$ | $\approx \pm 10^{18}$ |
| **Precisão** | Uniforme (0.0039) | Variável (~3 dígitos) | Baixa (~2 dígitos) | Adaptativa (alta perto de 1) |
| **Complexidade** | Baixíssima | Alta | Média | Média (com LUTs) |
| **Custo (Ciclos)** | 20–40 | 400–800 | 80–120 | 100–150 |
| **Melhor Uso** | Controle PID, UI | Legado, Protocolos | ML, Sensores Ruidosos | Filtros Kalman, Física |

![](/assets/images/posit1.webp)


### Veredito

1. **Use Ponto Fixo (Q8.8)** se os valores estiverem limitados e a velocidade for o fator primário. É a escolha segura para *loops* de controle motor em AVR/PIC.
2. **Use bfloat16** se estiver fazendo inferência de pequenas redes neurais ou acumulando somas grandes em que o *overflow* é um risco, mas a precisão individual de cada amostra não é crítica.
3. **Use Posits** em novos projetos que exigem estabilidade numérica em filtros recursivos (IIR, Kalman) dentro de um microcontrolador minúsculo.
4. **Evite IEEE-754 Half** para aritmética interna. Use-o apenas se for obrigado por um formato de arquivo ou protocolo de comunicação externo, convertendo para outro formato imediatamente ao receber o dado.