---
layout: post
title: Análise da Tecnologia das GPUs NVIDIA
author: Frank Alcantara
categories:
    - artigo
    - GPU
    - NVIDIA
    - hardware
    - IA
    - deep-learning
tags:
    - NVIDIA
    - Blackwell
    - Hopper
    - Rubin
    - Tensor Cores
    - CUDA
    - CuTe
    - WGMMA
    - TMA
    - TMEM
    - arquitetura GPU
    - microarquitetura
    - desempenho
image: assets/images/tecnvida.webp
description: Análise da evolução das arquiteturas NVIDIA desde Volta até Blackwell, com foco nos Tensor Cores, TMA/TMEM, CuTE DSL e previsões para a geração Rubin.
date: 2025-11-23T14:39:47-03:00
lastmod: 2025-12-21T21:05:24.801Z
published: true
draft: 2025-12-21T08:50:21.237Z
keywords:
    - NVIDIA
    - Blackwell
    - Hopper
    - Rubin
    - Tensor Cores
    - WGMMA
    - TMA
    - TMEM
    - CuTE DSL
    - arquitetura GPU
    - microarquitetura NVIDIA
    - GEMM
    - deep learning hardware
    - SIMT
    - programação GPU
slug: analise-da-tecnologia-das-gpus-nvidia
preview: Uma versão comentada de um das melhores análises disponíveis na internet sobre a arquitetura das GPUs Nvidia
---

A primeira ideia sobre este texto foi fazer uma tradução, aumentada e comentada, para o português de uma versão em inglês feita por [Jukan](https://x.com/Jukanlosreve) de uma postagem no WeChat do autor chinês [Zarbot](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Parece confuso e é mesmo.

Depois, a medida que ia traduzindo, fui percebendo que o texto em inglês perdeu algumas nuances do original em Chinês. Mais que isso, são tantas informações em graus diferentes de profundidade que o texto fica difícil de acompanhar. Neste ponto resolvi ir além, fazer a minha própria versão do texto original, direto do Mandarim. Pensa em uma pessoa ambiciosa.

**Em resumo, eu fiz uma versão do original em mandarim para o português, usando como referência a versão em inglês. Que fique claro: não tenho nenhuma associação com o autor, ou com o primeiro tradutor.**

>Eu vou colocar todos os meus pensamentos e pesquisas em blocos de destaque como este. Além, é claro, de incluir todos os links que eu achar necessário para facilitar o entendimento e possível aprendizado. 

Deste ponto em diante, o eu, ou qualquer pronome na primeira pessoa estará se referindo ao autor original do texto em mandarim, Zarbot. 

Como um pouco de honestidade não faz mal a ninguém: este é um texto grande, muito grande e complexo, muito complexo. Se você quiser ler tudo de uma vez, prepare-se para uma maratona.

**O texto original começa com um TL;DR no qual Zarbot se apresenta**, apresentação que eu achei importante manter.

Eu passei algumas semanas para organizar algumas operações **GEMM** no [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) e no [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) usando o [Cute DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html). Ao longo do tempo tive a oportunidade de acompanhar a evolução do [Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/) para o [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) e depois para o [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/). Por coincidência, penúltimo fim de semana, eu participei do Summit de Computação Turing da Huawei e conversei com o Dr. Liao e outras pessoas da equipe Ascend. 

> Zarbot deve estar se referindo ao HUAWEI CONNECT 2025 em Shanghai, China, setembro de 2025.

Depois disso, [Jensen Huang](https://grokipedia.com/page/Jensen_Huang) apresentou o desenvolvimento das tecnologias [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) e [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) [ao vivo na Keynote](https://www.nvidia.com/gtc/dc/keynote/) do GTC (GPU Technology Conference). Neste contexto, preparei uma análise abrangente e uma _previsão da microarquitetura da nova geração_ (É só um palpite, um esforço de adivinhação, não me culpe se eu estiver errado).

>Essa nova arquitetura proposta pela Nvidia foi discutida com foco na distribuição de tensão contínua para alimentação dos racks [neste artigo](https://frankalcantara.com/nvidia-fabrica-de-ia/).

![](/assets/images/tradu1.webp)
**Figura 1**: a tabela apresenta o conjunto de instruções TCGen100 da arquitetura Blackwell organizadas em dois grupos: instruções síncronas para gerenciamento de recursos e sincronização (`alloc`, `dealloc`, `relinquish_alloc_permit`, `fence`, `wait` e `commit`) e instruções assíncronas para operações computacionais e de movimentação de dados (`**MMA**` para multiplicação de matrizes, `cp` para cópia, `shift` para deslocamento, `ld` para leitura e st para escrita). Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**GEMM**, **G**eneral **M**atrix **M**ultiply, representa a operação fundamental de multiplicação de matrizes expressa como:
>
>$$C = \alpha \cdot (A \times B) + \beta \cdot C$$
>
>na qual $A$ é uma matriz $m \times k$, $B$ é uma matriz $k \times n$, e $C$ é uma matriz $m \times n$. Os escalares $\alpha$ e $\beta$ são fatores de escala. Esta operação é a base computacional de praticamente todos os algoritmos de deep learning modernos, representando tipicamente mais de $90\%$ do tempo de execução em redes neurais.
>
>A complexidade computacional de **GEMM** é $O(m \cdot n \cdot k)$, mas a complexidade de acesso à memória é $O(m \cdot k + k \cdot n + m \cdot n)$. Esta discrepância cria o desafio fundamental: quanto maior a matriz, maior a razão `compute-to-memory-access`, permitindo melhor utilização do hardware.
>
>**CuTE DSL**, **Cu**da **TE**mplates **D**omain-**S**pecific **L**anguage: é uma abstração algébrica desenvolvida pela Nvidia para expressar layouts de tensores e operações sobre eles de forma composicional. A abstração fundamental é o conceito de Layout, definido algebricamente como uma função:
>
>$$\text{Layout} : \mathbb{Z}^n \rightarrow \mathbb{Z}$$
>
>Que mapeia coordenadas multidimensionais lógicas para offsets lineares em memória. Um
Layout é especificado por um par (`Shape`, `Stride`), tal que:
>
>- `Shape` é uma tupla hierárquica descrevendo as dimensões lógicas e;
>- `Stride` é uma tupla hierárquica descrevendo os passos em memória.
>
>Por exemplo, o layout de uma matriz $4 \times 8$ armazenada em `row-major` seria:
>
>```shell
>Layout<Shape<_4, _8>, Stride<_8, _1>>
>```
>Em row-major, os elementos de uma linha são armazenados contiguamente (lado a lado) na memória.
>
>A álgebra do **CuTE DSL** permite composição de layouts através de operações como:
>
>1. **Particionamento**: dividir um tensor em tiles menores;
>2. **Composição**: combinar múltiplos layouts;
>3. **Swizzling**: permutações complexas para otimizar acesso à memória.
>
>O aspecto fundamental do **Cute DSL** é abstrair completamente a complexidade da **permutação de endereços de memória** (*swizzling*), necessária para evitar **conflitos de banco** na **memória compartilhada**. Nas arquiteturas Hopper e Blackwell, padrões de permutação de $128 \text{ bits}$ são necessários para maximizar a **taxa de transferência** (ou *throughput*) de acesso à memória, e o **Cute DSL** calcula automaticamente os índices corretos.

Na minha análise, o verdadeiro diferencial competitivo da Nvidia, seu fosso econômico, transcende as polêmicas habituais sobre o ecossistema **CUDA** ou o modelo **SIMT**. A grande vantagem da empresa reside na capacidade de abstrair a complexidade do hardware, resolvendo o que poderíamos chamar de trabalho pesado de engenharia de software de forma transparente dentro da arquitetura. Essa integração vertical, que vai do algoritmo ao silício, equilibra programabilidade e desempenho provocando e inspirando a concorrência. Além disso, o timing de lançamento e a execução de mercado da Nvidia são, inacreditavelmente precisos. Quase cirúrgicos. Ainda assim, toda arquitetura tem seus trade-offs e deficiências.

Em seguida, eu discutiremos alguns dos problemas da Nvidia que afetam tecnologias como a [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), a [Grace](https://www.nvidia.com/pt-br/data-center/grace-cpu/) e a recém-lançado [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/). Depois, assumindo que eu fosse um dos arquitetos da [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference), vamos discutir os caminhos possíveis que a arquitetura irá seguir.

## Um Processo Evolucionário: da Volta ao Blackwell

Começando com a introdução dos **[Tensor Cores](https://www.digitalocean.com/community/tutorials/understanding-tensor-cores)** no [Volta](https://www.nvidia.com/pt-br/data-center/v100/), a arquitetura **SIMT** tradicionalmente definida da Nvidia começou um processo de disrupção. A migração completa da arquitetura só foi finalizada com a geração [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference). _Todo este processo se estendeu por dez anos, representando tanto iterações graduais de hardware quanto inovações progressivas de software_.

>**O Modelo SIMT da NVIDIA**
>
>A NVIDIA denomina seu modelo de paralelismo como **SIMT**, **S**ingle **I**nstruction, **M**ultiple **T**hreads*, Instrução Única, Múltiplos Threads. Este modelo opera entre dois paradigmas academicamente bem definidos:
>
>* **SIMD**, **S**ingle **I**nstruction, **M**ultiple **D**ata: Processamento paralelo de elementos em vetores curtos;
>* **SMT**, **S**imultaneous **M**ulti**T**hreading: Execução paralela de instruções de vários threads independentes (ex: HyperThreading).
>
>O **SIMT** funciona como um híbrido entre processamento vetorial e threading em hardware, focado no equilíbrio entre flexibilidade e eficiência.
>
>Geralmente, modelos menos flexíveis são mais eficientes. Didaticamente, aceitamos duas hierarquias:
>
>* **Flexibilidade**: **SIMD** < **SIMT** < **SMT**
>* **Desempenho (em cargas compatíveis)**: **SIMD** >**SIMT** >**SMT**
>
>Embora se diga que o **SIMT** é um **SIMD** mais flexível, é importante notar a distinção técnica: internamente, a execução dos grupos de threads, chamados de **Warp** s, ainda ocorre de forma sincronizada, com lockstep, similar ao **SIMD**. Pesquisadores frequentemente definem o **SIMT** mais precisamente como **SIMD** com abstração de thread e máscara de execução.
>
>>**Lockstep** é o regime de execução sincronizada no qual todos os threads de um grupo (como um warp) avançam em uníssono, processando a mesma instrução no mesmo ciclo de clock e compartilhando um único contador de programa (PC).
>
>Mesmo que a hierarquia que supomos acima seja útil, alguns arquitetos de sistemas consideram **SMT** e **SIMT** como filosofias opostas: o **SMT** busca maximizar as instruções por ciclo, **I**PC**, **I**nstructions **P**er **C**ycle, de poucos threads complexas, enquanto o **SIMT** sacrifica o **I**PC** individual em favor da vazão total, throughput, de milhares de threads simples.
>
>**SIMT vs. SIMD**: ambos usam o broadcast de instruções para múltiplas unidades de execução, economizando hardware de controle, em ciclo `fetch`/`decode`. No entanto, o modelo da NVIDIA introduz três diferenciais que o **SIMD** clássico não possui:
>
>**1. Instrução Única, Múltiplos Conjuntos de Registradores**: o **SIMD** tradicional exige que o programador gerencie vetores curtos. O **SIMT** permite uma escrita escalar. O código é escrito para uma única thread, usando lógica padrão. Na prática, a **GPU** executa milhares de threads. Um Multiprocessador Streaming, **SM**, gerencia múltiplos núcleos, e todas os threads de um **Warp** executam a mesma instrução simultaneamente, mas **cada thread possui seus próprios registradores** privados.
>>**Nuances e Custos**:
>
>>* **Redundância**: valores idênticos entre threads, tais como ponteiros base, são replicados nos registradores de cada thread, consumindo mais energia que no **SIMD** escalar único.
>
>>* **Tipos de Dados e Eficiência**: no **SIMD**, um registrador de $128 \text{ bits}$ pode armazenar, por exemplo, $16$ elementos de $8 \text{ bits}$. No **SIMT** puro, cada thread processa um item. Se o dado for pequeno, $8 \text{ bits}$, o hardware pode subutilizar a largura do registrador e da ALU.
>
>>* **Arquiteturas** coma Volta e Ampere mitigam isso introduzindo caminhos de execução **SIMD** internos, como **Tensor Cores** e instruções para tipos empacotados, permitindo maior densidade de cálculo quando necessário.
>
>**2. Instrução Única, Múltiplos Endereços**: o **SIMT** permite acessos indiretos à memória ($a[i] = lut[b[i]]$). Enquanto no **SIMD** operações de leitura/escrita dispersa, `gather`/`scatter`, são complexas, no **SIMT** cada thread pode calcular seu próprio endereço de memória.
>
>>**Desafios de Memória**:
>
>>* **DRAM**: se os endereços calculados pelos threads de um **Warp** não forem sequenciais, a **GPU** não consegue realizar a coalescência, agrupamento de acessos. Isso força múltiplas transações de memória, degradando a performance.
>
>>* **Memória Compartilhada**: acessos concorrentes a bancos de memória diferentes na mesma memória compartilhada são rápidos. Se múltiplos threads acessarem o mesmo banco, ocorre conflito e serialização.
>
>**3. Instrução Única, Múltiplos Caminhos de Fluxo**: o **SIMT** suporta divergência de fluxo, `if`/`else`. Se threads de um mesmo **Warp** tomarem caminhos diferentes, o hardware usa uma **máscara de execução**.
>
>>**Execução**: primeiro, executam-se os threads que entraram no ramo `if`, enquanto as outras ficam inativas/mascaradas. Depois, inverte-se a máscara e executam-se os threads do `else`. A correção lógica é mantida, mas a execução torna-se serializada, reduzindo a eficiência proporcionalmente à divergência.
>
>**Estratégias de Ocultação de Latência**
>
>A comparação com o **SMT** revela a estratégia central da **GPU**: usar threads para ocultar a latência, não para maximizar o uso da pipeline de uma única thread.
>
>**SMT** (**CPU**): **CPUs** usam execução fora de ordem, predição de desvio e caches enormes para fazer uma única thread ser veloz. O **SMT** é uma medida auxiliar para preencher lacunas de ociosidade.
>
>**SIMT** (**GPU**): a **GPU** assume que a latência de memória é alta e não tenta combatê-la com caches complexos ou predição. A estratégia é a **força bruta do paralelismo**. Se um **Warp** trava esperando dados da memória, o hardware troca instantaneamente para outro **Warp** pronto.
>
>**Context Switching e Registradores**: diferente de sistemas operacionais tradicionais, a **GPU** não realiza trocas de contexto salvando dados na memória principal, o que seria proibitivamente lento. Neste caso:
>
>* Todos os contextos dos threads ativos devem residir **nos registradores on-chip**. Isso significa que se não houver registradores suficientes para todas os threads desejados, o kernel simplesmente não é iniciado. 
>>No contexto de programação da **CUDA**, um kernel é uma função que, quando chamada pela **CPU**, será executada paralelamente por milhares de threads simultâneas dentro da **GPU**. Um kernel representa a unidade básica de trabalho para o processamento massivo de dados em arquiteturas voltadas para o processamento gráfico.
>* Esse comportamento explica a quantidade massiva de registradores que existem nas **GPUs**, $16 \text{ KB}$ por bloco ou mais. Funcionalmente, esses registradores atuam como uma memória local extremamente rápida e particionada.
>
>Para entender melhor a hierarquia de processamento em uma **GPU**, vejamos os três níveis principais:
>
>>1. kernel: a função principal enviada para a **GPU**, representando o ponto de entrada do programa que será executado no dispositivo.
>>
>>2. **Grid**: a estrutura global que engloba o conjunto total de threads necessárias para processar o kernel.
>>
>>3. **Bloco (Thread Block)**: o **Grid** é dividido em blocos. Cada bloco é uma unidade de escalonamento que reside integralmente em um único multiprocessador da **GPU**. Threads dentro de um mesmo bloco podem cooperar entre si através de sincronização e acesso à memória compartilhada.
>>
>>4. **Warp** **: a unidade fundamental de execução no hardware. Um bloco é subdividido em grupos de 32 threads chamados **Warp** s**. O hardware da **GPU** escalona e executa instruções no nível do **Warp** **; _todos os seus 32 threads executam a mesma instrução simultaneamente_.
>>
>>5. **Thread**: a menor unidade lógica da hierarquia. Cada thread possui seu próprio estado de execução e um conjunto privado de registradores.
>
>**Custos e Limitações**
>
>A simplicidade do hardware, sem predição complexa, em troca de throughput traz limitações claras:
>
>1. **Dependência de Ocupação**: a performance depende de haver **Warp** s suficientes para alternar e ocultar a latência. Sem paralelismo massivo, a **GPU** é ineficiente.
>
>2. **Penalidade por Divergência**: threads não relacionados competem por recursos e a divergência de fluxo quebra o modelo de execução em *lockstep*.
>
>3. **Sincronização**: historicamente, a sincronização era restrita a barreiras dentro do mesmo bloco (`__syncthreads()`). A partir de arquiteturas como Kepler e Volta, surgiram primitivas mais ricas, como os `Cooperative Groups`, `__sync**Warp** ()`, permitindo maior controle. Ainda assim, a sincronização global (entre blocos) ou com o host, **CPU** permanece muito mais restritiva que em ambientes **SMT**, devido à inviabilidade de colocar milhares de threads em espera passiva gerenciada por um sistema operacional.

### Tensor Cores

De uma perspectiva de hardware, olhando desde as instruções **FMA** mais antigas para o vetorizado **DP4A**, depois para a primeira geração de Tensor Cores na Volta (microarquitetura SM70), e subsequentemente Ampere/Hopper/Blackwell, todos eles têm aumentado a escala da multiplicação de matrizes, melhorado a razão entre o custo de computação e o acesso a memória, e suportado formatos de dados de precisão mais baixa.

![](/assets/images/tradu2.webp)
**Figura 2**: Layouts de fragmentos de registradores para instruções **MMA** em diferentes Compute Capabilities. Observe a mudança drástica na organização thread-dado (T#) e o aumento da granularidade do tile, passando do modelo síncrono de $8\times 8\times 4$ da Volta (microarquitetura SM70) para o modelo de **Warp** group $64\times 16\times 16$ da arquitetura Hopper (microarquitetura SM90). Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).
>

| Geração (SM) | Ano de Lançamento | Instrução / Tipo de Dado | Característica Principal (Significado das Siglas) |
| :--- | :---: | :--- | :--- |
| **Pascal (SM61)** | 2016 | **DP4A** | **Dot Product 4-Accumulate**. Instrução para inteiros de $8 \text{ bits}$ (INT8). Realiza o produto escalar de dois vetores de $4$ elementos de $8 \text{ bits}$ e acumula o resultado em um inteiro de $32 \text{ bits}$. Fundamental para as primeiras otimizações de inferência em INT8. |
| **Volta (SM70)** | 2017 | `F32F16F16F32_NT` | **HMMA (Half-precision Matrix Multiply-Accumulate)**. Primeira geração de Tensor Cores. Multiplicação de matrizes de $16 \text{ bits}$ (F16) por $16 \text{ bits}$ (F16), com acumulação em $32 \text{ bits}$ (F32). O sufixo NT indica a transposição dos operandos (Não Transposto x Transposto). O layout é fixo e síncrono. |
| **Ampere (SM80)** | 2020 | `F32F16F16F32_TN` | **MMA** (Matrix Multiply-Accumulate) Assíncrono**. Evolução do **HMMA**. Introduz a capacidade de carregar dados da memória compartilhada de forma assíncrona (`ldmatrix`) antes da computação. O layout TN (Transposto x Não Transposto) mostrado é otimizado para evitar conflitos de bancos de memória compartilhada. |
| **Hopper (SM90)** | 2022 | `F32F16F16_SS` | **WGMMA** (**W**arp **G**roup **M**atrix **M**ultiply-**A**ccumulate)**. Execução por **Warp** group** (grupo de $128$ threads), não apenas por **Warp**. Os operandos A e B residem na **Shared Memory (SS)** e são consumidos diretamente pelos Tensor Cores, sem passar pelo arquivo de registradores dos threads individuais, aumentando muito a eficiência e o tamanho do tile. |
| **Hopper/Blackwell (SM90/100)+** | 2022+ | `FP8` (E4M3 / E5M2) | **Ponto Flutuante de 8-bit**. Novos tipos de dados introduzidos massivamente com A arquitetura Hopper. E4M3 (4 bits expoente, 3 mantissa) para maior alcance dinâmico, e E5M2 (5 bits expoente, 2 mantissa) para maior precisão. Requerem layouts de hardware específicos e, Na arquitetura Blackwell (2024), suporte a microscaling. |
| **Blackwell (SM100)+** | 2024 | `FP4` (E2M1) | **Ponto Flutuante de 4-bit com Microscaling**. Tipo de dado extremamente compacto para inferência de modelos gigantes (**LLMs**). A característica principal é o **Block Scaling**: os dados de $4 \text{ bits}$ são acompanhados por um fator de escala de maior precisão a cada bloco de $16$ ou $32$ elementos. O layout de memória é complexo, intercalando dados e escalas. |

>**FMA: Fused Multiply-Add (Multiplicação e Adição Fusionada)**: trata-se de instrução clássica e fundamental em computação numérica. Ela realiza a operação:
>
>$$D = A \times B + C$$
>
>Na qual $A$, $B$ e $C$ são números escalares, ou elementos correspondentes em um vetor **SIMD** comum. Neste acrônimo, a palavra chave é Fusionada. Em processadores antigos, para fazer $A \times B + C$, o hardware precisava de dois passos:
>
>1. Calcular $Intermediario = A \times B$, e arredondar o resultado;
>2. Calcular $Resultado = Intermediario + C$, e arredondar novamente.
>
>Na instrução **FMA**, a multiplicação e a adição acontecem em **um único passo** no hardware, com apenas um arredondamento final. Isso é importante por dois motivos principais:
>
>1. **Precisão**: como há apenas um arredondamento no final, o resultado é matematicamente mais preciso.
>2. **Desempenho**: ela conta como duas operações de ponto flutuante (**FLOPs**), uma multiplicação e uma adição, mas é executada no tempo de apenas uma instrução. Isso dobra a capacidade de cálculo teórica do processador.
>Nós vimos `FMA <double>` no canto inferior esquerdo da **Figura 2**. Isso representa os núcleos **CUDA** tradicionais, não Tensor Cores, operando com precisão dupla, FP64. Era assim que a maioria da supercomputação científica era feita antes da era da IA profunda.
>
>**MMA**: Matrix Multiply-Accumulate (Multiplicação e Acumulação de Matrizes)**: a **MMA** é a evolução do **FMA** para a era da Inteligência Artificial. É a instrução que define e opera os **Tensor Cores** da NVIDIA.
>
>Em vez de operar com números individuais, a **MMA** opera com **pequenos blocos de matrizes**, chamados de **tiles**, de uma só vez. A operação matemática básica é a mesma, mas em escala matricial:
>
>$$D = A \times B + C$$
>
>Na qual $A$, $B$, $C$ e $D$ agora são **pequenas matrizes**, **tiles**. Por exemplo, na primeira geraçãa Volta (microarquitetura SM70), a operação padrão era multiplicar duas matrizes $4\times 4$ e somar a uma terceira matriz $4\times 4$. Ou seja, cada instrução **MMA** realizava $16$ multiplicações e $15$ adições, totalizando $32$ **FLOPs**, em um único passo. Redes Neurais Profundas, Deep Learning, são, essencialmente, bilhões de multiplicações de matrizes gigantescas. Fazer isso elemento por elemento usando **FMA** seria muito lento.
>
>A instrução **MMA** permite que o hardware, o **Tensor Core**, pegue um bloco inteiro de dados e resolva a multiplicação e soma desse bloco em um único ciclo de clock, ou poucos ciclos, dependendo da complexidade.
>
>**A Evolução do **MMA**: o **MMA** não ficou parado. Conforme vimos na tabela acima, ele evoluiu em três grandes etapas:
>
>1. **HMMA (Volta/Turing)**: **MMA** Síncrono**. Um thread individual era responsável por segurar pedaços específicos da matriz em seus registradores para que o **Tensor Core** operasse.
>2. **MMA** Assíncrono (Ampere)**: O hardware começou a poder buscar os dados da memória enquanto calculava o **MMA** anterior.
>3. **WGMMA (Hopper/microarquitetura SM90)**: **Warp** group **MMA**. A operação **MMA** tornou-se tão grande e complexa que não é mais um único thread ou um único **Warp** que a gerencia, mas um grupo de $128$ threads, **Warp** group** cooperando. Os dados para o **MMA**, no **Hopper** fluem diretamente da memória compartilhada para os Tensor Cores, sem a necessidade de microgerenciamento usando os registradores individuais dos threads.

Olhando as mudanças na precisão numérica e as restrições de área do chip, mostradas a seguir, a geração [Blackwell Ultra (B300)](https://www.tomshardware.com/**PC**-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4) prima por ter começado a cortar o custo computacional necessário aos cálculos em alta precisão.

![](/assets/images/tradu3.webp)
**Figura 3**: Evolução dos formatos de dados suportados pelos Tensor Cores da Nvidia, desde o INT8 no Pascal até o FP4 com microscaling Na arquitetura Blackwell. Observe a tendência de adoção de formatos de menor precisão para maximizar throughput computacional em cargas de trabalho de deep learning. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

O mercado e academia esperam que a geração [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) dobre a escala dos Tensor Cores, estimada em $256 \times N \times 256 \text{ bits}$. Por outro lado, eu acho que veremos uma expansão adicional da **MMA** de **2-CTA** (**C**ooperative **T**hread **A**rray) da arquitetura Blackwell para uma instrução **MMA** conjunta de **4-CTA** no Rubin. No entanto, haverão demandas adicionais para agendamento dentro do **CGA**, **C**ooperative **G**roup **A**rray).

Outro problema trazido pelo aumento na capacidade computacional é a alteração do caminho de acesso aos dados.

Os [Tensor Cores](https://www.digitalocean.com/community/tutorials/understanding-tensor-cores) originais, sa plataforma Volta, começaram a partir da reutilização dos registradores dos [CUDA Cores](https://acecloud.ai/blog/nvidia-cuda-cores-explained/). À medida que a escala dos Tensor Cores, na arquitetura [Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/), se expandiu a eficiência na obtenção dos dados tornou-se crítica. Visando mitigar a pressão de registradores, a Nvidia introduziu a instrução [`cp.async](https://research.meekolab.com/messing-around-with-gpus-again?source=more_articles_bottom_blogs). Uma instrução para mover dados diretamente da memória global para a memória compartilhada. Permitindo contornar o uso dos registradores e, ao mesmo tempo, reduzindo a poluição do cache **L1**. Criaram uma instrução capaz de liberar recursos para computação matemática intensa.

>**Tensor Cores**: _unidades de processamento especializadas, projetadas para acelerar operações de multiplicação e acumulação de matrizes (**MMA**)_, essenciais para tarefas de aprendizado de máquina e inteligência artificial. Estas unidades de processamento permitem computação de alta performance utilizando precisão mista, como **FP16**, **BF16** e **INT8**.
>
>**Pressão de registradores (Register Pressure)**: termo criado para definir uma condição na programação de **GPUs** na qual a demanda por registradores (memória ultra-rápida privada por thread) excede a disponibilidade física. Quando isso ocorre, o compilador é forçado a realizar o derramamento (register spilling) dos dados para a memória local (**LMEM**). Como a **LMEM** significativamente mais lenta, este derramamento, reduz a eficiência do kernel.
>
>**`cp.async**: instrução de cópia assíncrona introduzida na arquitetura [Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/). Essa instrução _permite transferir dados da memória global diretamente para a memória compartilhada **sem passar pelo arquivo de registradores**_. Além de ocultar a latência de memória permitindo computação paralela, sua principal vantagem arquitetural é aliviar a pressão de registradores, liberando-os para cálculos matemáticos.
>> Neste ponto, temos que fazer uma distinção importante:
>>* Conjunto de Registradores (Register Set): Refere-se à visão lógica ou arquitetural. É a lista de registradores que o programador ou o compilador consegue enxergar e utilizar através das instruções (ISA).
>>
>>* Arquivo de Registradores (Register File): Refere-se ao bloco de hardware real. É uma matriz de células de memória (geralmente SRAM) organizada com portas de leitura e escrita específicas. O termo arquivo é usado porque esses registradores são endereçados como uma pequena memória interna extremamente rápida.
>
>**L1** ( **Cache L1**): cache de nível 1, memória rápida próxima aos núcleos de execução. No contexto do `cp.async, a instrução faz os dados contornarem o **L1**, bypass, para evitar a poluição de cache. Ou seja, impedir que novos dados de streaming expulsem dados reutilizáveis que já estão no cache, depositando-os diretamente na memória compartilhada.

Um novo passo evolutivo foi introduzido na arquitetura [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/), com a introdução do **T**ensor **M**emory **A**ccelerator, **[TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)**, permitindo que operandos fossem carregados diretamente na memória compartilhada local, **S**hared **Mem**ory **SMEM**, além de implementar o **C**luster **G**roup **A**ddress, **CGA**, viabilizando **D**istributed **S**hared **M**emory, **DSMEM**. No entanto, neste ponto da evolução, os resultados dos acumuladores ainda residiam no arquivo de registradores para facilitar as operações subsequentes de Epílogo, o que exigia o uso de barreiras de espera, wait barriers, para sincronização.

>As operações de Epílogo referem-se à fase final de execução de um kernel, que ocorre imediatamente após o término do laço principal de computação.
>
>>* **Operações Elemento a Elemento (Fusions)**: é aqui que se aplicam funções de ativação (como ReLU, GELU ou Sigmoid), adição de bias (viés) e escalonamento por constantes ($\alpha$ e $\beta$). Realizar isso no epílogo evita que os dados precisem ser lidos novamente da memória global.
>>* **Conversão de Tipos**: se o cálculo interno foi feito em FP32 para manter a precisão, mas o resultado final deve ser entregue em **FP16** ou **BF16**, essa conversão de formato ocorre no epílogo.
>>* **Redução e Estatísticas**: operações como calcular o valor máximo absoluto, útil para o escalonamento dinâmico em **FP8** são geralmente integradas nesta fase.
>>* **Escrita na Memória Global**: o epílogo organiza os dados que estão nos registradores ou na SMEM e os escreve de forma eficiente, coalescida, na memória de vídeo (**VRAM**), respeitando o layout final desejado.

Finalmente, a arquitetura Blackwell introduz a **T**ensor **Mem**ory (**TMEM**), desacoplando efetivamente o ****Tensor Core**** do **CUDA núcleo**. Na prática, essa separação permite que o ****Tensor Core**** opere quase como um coprocessador dentro do **SM**, o que justifica a necessidade de mecanismos de sincronização mais robustos, como o **MBarrier**, para coordenar o fluxo de dados entre essas unidades independentes. Essa transição reaproveita o mecanismo de **MBarrier** já estabelecido pelas operações assíncronas do **TMA** na arquitetura Hopper. Como pode ser visto na tabela abaixo:

| Arch   | Matrix A  | Matrix B | Matrix D |
| :---   | :---:    | :---:  | :---:  |
| Volta   | RF     | RF    | RF    |
| Ampere  | RF     | RF    | RF    |
| Hopper  | RF / SMEM  | SMEM   | RF    |
| Blackwell | TMEM / SMEM | SMEM   | TMEM   |

>**RF (Register File)**: A memória mais rápida e próxima das unidades de execução. Ocupar o RF com matrizes grandes pode causar pressão de registradores, limitando a ocupação (número de **Warp** s ativos).
>
>**[TMA (Tensor Memory Accelerator)](https://research.colfax-intl.com/tutorial-hopper-tma/)**: Motor de cópia assíncrona introduzido na arquitetura Hopper (H100). Ele gerencia a transferência de dados entre a memória global e a memória compartilhada (SMEM) de forma independente, liberando os threads para outras tarefas e lidando automaticamente com cálculos de endereço.
>
>**[CGA (Cluster Group Architecture)](https://developer.nvidia.com/blog/cooperative-groups/)**: Refere-se à organização em *Thread Block Clusters*. Permite que múltiplos blocos de threads cooperativos, os **C**ooperative **T**hread **A**rray, **CTA**s, sejam agrupados em um Cluster para cooperar na execução, compartilhando dados de forma rápida e sincronizada através da hierarquia de memória.
>
>**DSMEM (Distributed Shared Memory)**: Recurso que permite que a memória compartilhada, **SMEM**, de um bloco seja acessada diretamente por outros blocos dentro do mesmo Cluster. Isso elimina a necessidade de passar pela memória global para trocar dados entre blocos vizinhos.
>
>**TMEM (Tensor Memory)**: Na arquitetura Blackwell, refere-se a uma área de memória dedicada ou um modo de operação que separa o armazenamento de dados dos Tensor Cores do caminho de dados tradicional dos **CUDA** Cores, reduzindo a contenção de registradores e permitindo pipelines mais eficientes.
>
>Com um pouco mais de detalhe, podemos dizer que **MBarrier (Memory Barrier)** é uma primitiva de sincronização em hardware introduzida formalmente na arquitetura Ampere e aprimorada significativamente na arquitetura Hopper (SM90). Diferente de barreiras tradicionais de execução (como `__syncthreads()`), que pausam threads até que todas cheguem a um ponto, o **MBarrier** rastreia a conclusão de **transações de memória assíncronas**.
>
>O funcionamento baseia-se em contagem de transações:
>
>>1. Os threads chegam à barreira e indicam quantos bytes ou transações estão aguardando (expectativa).
>>2. O hardware, como o **TMA**, decrementa o contador da barreira automaticamente à medida que os dados são escritos na memória compartilhada.
>>3. Os threads podem verificar o estado da barreira e dormir ou continuar processando outras coisas até que a contagem chegue a zero.
>
>O **MBarrier** é a peça chave que permite a **Warp** Specialization**: um grupo de **Warp** s pode apenas emitir comandos de cópia, produtores, enquanto outro grupo, consumidores, espera no **MBarrier**. Isso desacopla a latência da memória do fluxo de execução da computação, permitindo pipelines assíncronos mais eficientes tanto na arquitetura Hopper quanto na Blackwell.

![](/assets//images/nvidia-evolui.webp)
**Figura 4**: Evolução das Arquiteturas das GPUs NVIDIA.

Com a introdução da **[TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)**, o [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) eliminou a dependência do arquivo de registradores, **RF**, para os acumuladores, concretizando a separação assíncrona entre as unidades de execução.

### Processamento Assíncrono

O outro aspecto que precisamos considerar é o processamento assíncrono. Quando a geração [Volta](https://www.nvidia.com/en-au/data-center/v100/) introduziu um **P**rogram **C**ounter, **PC** independente para cada thread, marcou o início da execução assíncrona.

![](/assets/images/tradu4.webp)
**Figura 5**: comparação gráfica entre os modelos de execução assíncrona na arquitetura NVIDIA, desde a introdução do **PC** independente por thread na Volta até o mecanismo **MBarrier** na Hopper e Blackwell. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

A partir desse ponto, os threads podiam esperar por mensagens para realizar processamento assíncrono, abrindo uma janela para programação assíncrona em relação às arquiteturas alinhadas por **PC** tradicionais.

![](/assets/images/tradu5.webp)
**Figura 6**: compara o modelo de comunicação entre threads nas arquiteturas de Pascal e Volta. Na arquitetura Pascal (esquerda), o modelo é Lock-Free, significando que os threads não podem esperar por mensagens de outros threads, o que limita a complexidade dos algoritmos de sincronização. Na arquitetura Volta (direita), é introduzido o Independent Thread Scheduling (Escalonamento Independente de Threads), permitindo que threads esperem por mensagens, o que habilita algoritmos livres de inanição e formas mais robustas de comunicação e sincronização entre threads. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>Um ponto que merece destaque: **o PC Independente por Thread da Volta foi o verdadeiro Big Bang da execução assíncrona nas GPUs NVIDIA**.
>
>_Antes da Volta, todos os threads de um **Warp** compartilhavam um único Program Counter, **PC**_. Isso significava que, mesmo com máscara de execução, o hardware ainda avançava o **PC** de forma lockstep. Ou seja, o **Warp** executava as duas ramificações de um `if/else` sequencialmente, mas sempre com o mesmo **PC** para todo o grupo. Era o clássico **SIMT** puro. Esse modelo difere das **CPUs** escalares.
>
>Nas **CPUs** escalares cada thread possui seu próprio **PC** independente. O comportamento pré-Volta assemelha-se ao funcionamento das unidades **SIMD**, como [AVX](https://www.intel.com.br/content/www/br/pt/architecture-and-technology/avx-512-overview.html) no qual uma única instrução controla múltiplos caminhos de dados simultaneamente.
>
>A partir da Volta, a NVIDIA introduziu o **Independent Thread Scheduling**: cada thread passou a ter seu próprio **PC** visível tanto ao programador, quanto ao hardware. Isso mudou tudo:
>
>- deste ponto em diante o scheduler da **GPU** pode reagendar **Warps** de forma flexível, inclusive intercalando instruções de diferentes caminhos de execução dentro do mesmo **Warp**.
>- abriu a porta real para sincronização mais fina, **syncWarp** e **cooperative groups** avançados.
>- e, mais importante para o que vem depois: permitiu que threads individuais **esperassem por eventos assíncronos**, como chegadas de mensagens via grid-sync, ou barreiras assíncronas futuras, sem precisar que todo o **Warp** estivesse no mesmo ponto do código.
>
>Em resumo: a Volta quebrou o dogma do lockstep rígido e criou as condições técnicas para que, anos depois, as arquiteturas Hopper e Blackwell pudessem criar verdadeiras pipelines assíncronas com **TMA**, **MBarrier**, **TMEM**, etc. Sem esse **PC** independente, nada do que veio depois seria possível. Pelo menos, não seria possível com a mesma elegância.
>
>Muita gente acha que a revolução assíncrona começou no Ampere com `cp.async` ou na arquitetura Hopper com **TMA**. Na verdade, a semente foi plantada em 2017, na Volta, e quase ninguém percebeu o quanto isso seria importante.

A NVIDIA introduziu a abstração Cooperative Groups para refinar a coordenação entre os threads, embora a operação dos **Tensor Cores** ainda estivesse vinculada à execução síncrona em todo o **Warp**. Esse cenário evoluiu na arquitetura Ampere, na qual a implementação da instrução `cp.async` tornou o fluxo de suprimento de dados do programa efetivamente assíncrono. Tal progresso estabeleceu os fundamentos para o modelo de Async Thread, permitindo que a movimentação de dados ocorra de forma independente da execução das instruções de cálculo nos **C**ooperative **T**hread **A**rray, **CTA**s.

![](/assets/images/tradu6.webp)
**Figura 7**: Ilustração do modelo de execução assíncrona com `cp.async` na arquitetura Ampere da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**Cooperative Groups: a abstração que finalmente libertou a sincronização nas **GPUs** NVIDIA da tirania dos blocos de threads**: antes do **CUDA 9**, 2017, exatamente na era Volta, a sincronização entre threads era extremamente rígida:
>
>- `__syncthreads()` $\rightarrow$ só dentro do mesmo block;
>- `__sync**Warp** ()` $\rightarrow$ só dentro do mesmo **Warp**, introduzido um pouco antes, mas ainda limitado;
>- Sincronização entre blocks? Praticamente impossível sem o uso de truques horrendos usando a global memory e os loops de polling.
>
>Isso caracterizava um problema porque, à medida que os kernels para multiplicação de matrizes, [GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), e transformers ficavam maiores, querer sincronizar múltiplos blocos que cooperam em um mesmo **tile** grande de matriz se tornou essencial para a performance. Neste contexto, os truques necessários para sincronizar múltiplos blocks eram ineficientes e propensos a erros.
>
>A NVIDIA então lançou a abstração **Cooperative Groups**, permitindo ao programador definir grupos arbitrários de threads que podem sincronizar de forma segura e eficiente:
>
>- `thread_block` $\rightarrow$ equivalente ao antigo `__syncthreads()`, mas agora como objeto;
>- `thread_block_tile` $\rightarrow$ partições estáticas dentro do block (ex: tiles de 32 threads = **Warp**, ou 16, 8, etc.);
>- `grid_group` $\rightarrow$ o santo gral: sincronização entre **todos os blocos do grid inteiro** com uma única chamada `grid.sync()`;
>- `multi_grid_group` $\rightarrow$ sincronização entre múltiplos grids lançados de dispositivos diferentes, usado em ambientes em multi-GPU;
>- E, **o mais poderoso**, grupos dinâmicos $\rightarrow$ `coalesced_group` e `cluster_group, disponíveis a partir da arquitetura Hopper.
>
>A abstração **Cooperative Groups** foi a camada de software que preparou o terreno para tudo que foi criado depois e que esteja relacionado com processamento, paralelo, concorrente e performance.
>
>Sem esta abstração, não teríamos os padrões de **Warp Specialization** da arquitetura Hopper, nem os pipelines assíncronos complexos da arquitetura Blackwell que misturam **producer/consumer Warps** dentro do mesmo **CTA** ou cluster. _Os Cooperative Groups transformaram o modelo **SIMT** rígido em algo muito mais próximo de um **MIMD**_, dando ao programador controle explícito sobre quem sincroniza, o quê, com quem.
>
>**Resumindo**: se o Independent Thread Scheduling da Volta foi o hardware que permitiu a divergência real, Cooperative Groups foi o software que transformou essa disponibilidade em algo útil. Juntos, eles são os verdadeiros pais da execução assíncrona moderna nas **GPUs NVIDIA**.
>
>**MIMD – Multiple Instruction, Multiple Data**: o modelo de paralelismo mais flexível que existe. Cada núcleo/processador executa **instruções diferentes**, em **dados diferentes**, de forma totalmente independente.
>
>**Exemplo prático**: uma **CPU** comum com vários cores rodando programas diferentes ao mesmo tempo, um núcleo no navegador, outro em um jogo, outro no WhatsApp, etc.. Aqui, são múltiplos fluxos de controle (instruções) operando em múltiplos conjuntos de dados. Com objetivos diferentes.
>
>O MIMD é o oposto do SIMD das **GPUs** clássicas. _No **MIMD** cada um faz o que quer, quando quer, com o que quer_. Quando isso acontece para um mesmo objetivo estamos no paraíso do paralelismo. Note, a expressão com um mesmo objetivo. Esta expressão faz com que o exemplo prático acima não seja um exemplo puro de MIMD, mas sim de multiprogramação. Ainda assim, você deve ter entendido a ideia.

A arquitetura Hopper foi além ao introduzir o **MBarrier**. Depois disso, pipelines assíncronos de software e **Warp Specialization** construídos em torno do **MBarrier** se tornaram populares. 

### Evolução da Execução Assíncrona: O Papel dos Proxies na Hopper

A arquitetura Hopper consolidou a execução assíncrona ao introduzir o conceito de **Async Proxy**. Essa inovação segmenta os caminhos de acesso à memória em dois domínios distintos: o **General Proxy**, responsável pelas operações tradicionais de `LD/ST` (Load/Store) no modelo `SIMT`, e o **Async Proxy**, dedicado a coordenar operações de hardware independentes, como as executadas pelo **TMA**. Para assegurar a integridade do sistema, o hardware utiliza barreiras de memória nas quais o fluxo `SIMT` convencional pode aguardar a sinalização de conclusão das transferências assíncronas. Essa estrutura permite combinar a flexibilidade do acesso à memória original com a alta vazão do **TMA**, garantindo que os requisitos de ordenação de memória sejam rigorosamente atendidos.

> **General Proxy vs. Async Proxy**
>
> Na microarquitetura Hopper, o termo **Proxy** refere-se ao agente de hardware encarregado de gerenciar as transações de dados entre os diferentes níveis de memória.
>
>* **General Proxy**: representa o caminho de execução padrão utilizado pelos núcleos |**CUDA**. Ele opera de forma síncrona em relação ao fluxo de instruções da thread; ou seja, quando um thread executa uma instrução de `LD` (Load), os recursos do `SM` (Streaming Multiprocessor) permanecem vinculados ao gerenciamento dessa transação até que o endereço seja resolvido.
>* **Async Proxy**: atua como um agente especializado que despacha a tarefa para o **T**ensor **M**emory **A**ccelerator, **TMA**. Uma vez que a instrução de cópia é emitida, o **Async Proxy** assume a responsabilidade pela movimentação dos dados, permitindo que o **SM** processe outras instruções enquanto a transferência ocorre em segundo plano.
>
> **Ordenação de Memória e Sincronização por Barreira**
>
> A coexistência de dois caminhos de acesso distintos exige um controle rigoroso de **Memory Ordering**. Como o **Async Proxy** opera fora do fluxo sequencial determinado pelo **PC** dos threads, torna-se necessário um ponto de convergência.
>
> O hardware utiliza o mecanismo de **MBarrier** para orquestrar essa dependência. Imagine que $T_{copy}$ seja o tempo final da transação assíncrona e $T_{load}$ seja o momento no qual a thread `SIMT` tenta ler esse dado na memória compartilhada. A barreira de memória garante a seguinte relação lógica:
>
> $$T_{load} \ge T_{copy}$$
>
> Sem essa barreira, os threads do **General Proxy** poderiam acessar endereços na **SMEM** antes que o **TMA** tivesse concluído a escrita, resultando em corrupção de dados ou condições de corrida.

![](/assets/images/tradu7.webp)
**Figura 8**: Representação do modelo de execução assíncrona com **MBarrier** na arquitetura Hopper. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

A arquitetura Hopper tem imperfeições. O **WGMMA** era uma solução temporária, ocupando uma grande quantidade de registradores enquanto também requeria espera síncrona. Portanto, quando A arquitetura Hopper foi lançado, foi explicitamente dito que o **WGMMA** da arquitetura Hopper (SM_90a) não seria compatível com versões anteriores.

![](/assets/images/tradu8.webp)
**Figura 9**: Limitações do **WGMMA** na arquitetura Hopper da NVIDIA, destacando a necessidade de espera síncrona e o consumo elevado de RMEM. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**WGMMA – Warp Group Matrix Multiply-Accumulate**: o **MMA** gigante da arquitetura Hopper. Uma solução transitória, bem cara. Na arquitetura Hopper (SM_90, SM_90a), a NVIDIA precisou escalar drasticamente o tamanho dos tiles que o ****Tensor Core**** conseguia processar de uma só vez. 
>
>A solução encontrada foi o **WGMMA**: em vez de um único **Warp** (32 threads) executar o **MMA** como nas gerações anteriores, agora um **Warp Group**, grupo de 4 **Warps**, equivalente a $128$ threads, executa a operação inteira em conjunto.
>
>Vantagens imediatas:
>- Tiles muito maiores (ex.: 64x128x64) diretamente da shared memory;
>- Enorme aumento de throughput em **GEMM** grandes (os transformers adoraram);
>- Dados fluem direto da **SMEM** para os **Tensor Cores** sem precisar passar pelos registradores de cada thread individual.
>
>Mas o preço foi alto, e por isso o autor original, chama de solução temporária;
>
>- **Consumo brutal de registradores**: mesmo não usando os registradores dos threads para os operandos, os acumuladores ainda ficavam no **RF** implicando em valores altíssimos de register pressure;
>- **Execução síncrona obrigatória**: o **Warp Group** inteiro tinha que esperar a operação terminar antes de prosseguir limitando as pipelines assíncronas.
>
>Por causa dessas limitações internas, a NVIDIA lançou duas versões da arquitetura Hopper:
>
>- SM_90 (H100 padrão) $\rightarrow$ **WGMMA** síncrono, alto uso de registradores;
>- SM_90a (H100 NVL, versão posterior) $\rightarrow$ algumas melhorias, mas ainda síncrono.
>
>**Avisando enfaticamente**: código compilado para SM_90a **não roda** em chips SM_90 antigos. O que, se você ainda não percebeu, implicou na quebra de compatibilidade a frente na mesma geração! Coisa muito rara e grave na indústria.
>
>_O **WGMMA** foi a tecnologia que permitiu a arquitetura Hopper bater records de performance em **LLMs** em 2023-2024, mas foi, para sermos honestos, uma gambiarra histórica_.

Na arquitetura Blackwell, o ****Tensor Core**** também constitui uma operação totalmente assíncrona, reutilizando a construção **MBarrier**. Assim, emitir **TMA** e instruções pode ser feito no nível de thread. No entanto, a alocação e cópia de memória para o **TMEM** ainda requerem manuseio no nível **Warp**. Por outro lado, o mecanismo `ClusterLaunchControl` foi introduzido, fornecendo alguma capacidade de agendamento dinâmico.

![](/assets/images/tradu9.webp)
**Figura 10**: Modelo de execução assíncrona com **TMEM** e **MBarrier** na arquitetura Blackwell da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**ClusterLaunchControl: o mecanismo que finalmente trouxe agendamento dinâmico real para dentro dos Clusters na arquitetura Blackwell**
>
>Na arquitetura Hopper, os **Thread Block Clusters** já existiam, mas o lançamento e o agendamento dos **CTAs** dentro de um cluster eram essencialmente **estáticos**: você definia o tamanho do cluster no despacho do kernel, via `cudaLaunchAttributeClusterDimension`, e todos os blocos eram lançados de uma vez, sem controle fino em tempo de execução.
>
>Na arquitetura Blackwell, a NVIDIA introduziu o **ClusterLaunchControl**, ou controles de lançamento de cluster mais avançados via hardware/registros dedicados, que permite:
>
>- Threads individuais, ou **Warps**, **decidirem dinamicamente** de quais blocos/clusters participar ou quando ativar certas operações cooperativas;
>- Agendamento dinâmico de **Warps** especializados dentro do mesmo cluster (ex.: um **Warp** pode acordar ou lançar tarefas para outros **Warps/CTAs** do cluster de forma assíncrona);
>- Melhor integração com **MBarrier** e **TMA** em escala de cluster, porque agora é possível coordenar produtores/consumidores de dados de forma mais flexível, sem precisar que todo o **Warp** esteja sincronizado o tempo todo.
>
>Isso importa porque mesmo com **TMA** e **TMEM** sendo assíncronos em nível de thread, a alocação de **TMEM** ainda exigia coordenação em nível de **Warp**. _O `ClusterLaunchControl` alivia isso permitindo que a própria **GPU**, ou o programador via instruções, faça **agendamento dinâmico em escala de cluster**, abrindo caminho para padrões ainda mais sofisticados de **Warp Specialization** e pipelines producer-consumer multi-CTA_.
>
>**Resumindo**: se o **MBarrier** foi como um semáforo assíncrono e o **TMEM** como uma memória dedicada, o `ClusterLaunchControl` é o maestro que finalmente permite orquestrar tudo isso de forma dinâmica em tempo de execução, aproximando a **GPU** de um verdadeiro modelo **MIMD** cooperativo em escala de cluster. Lembre-se em todas estas considerações supomos que o céu é azul, o mar está calmo e os ventos são favoráveis.

Podemos então, graças ao `ClusterLaunchControl`, construir padrões de processamento **Warp Specialization** mais complexos.

![](/assets/images/tradu10.webp)
**Figura 11**: Exemplos de padrões de processamento com **Warp** Specialization na arquitetura Blackwell da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

### Layout CuTE

Aqui também temos um artefato de software fantástico, especialmente se considerarmos a forma como abstrai a complexidade do **Swizzle** nas arquiteturas Hopper e Blackwell. Além disso, de uma perspectiva puramente algébrica, esta abstração permite resolver cálculos complexos de fronteiras de Tile/Partition, tornando o código mais intuitivo. Embora para aqueles que não são versados em álgebra, aprender o **CuTE** ainda represente uma curva de aprendizado íngreme. O leitor original começou a discutir a álgebra de Layout **CuTE** no artigo abaixo: ([Referência](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21&poc_token=HL7TImmjizY1ndR3EjvNUMHxZ2wg3Uh_AUlNECLB) como diria meu professor Lobo: te vira, tu não nasceu quadrado, dá seus pulos e se vira para traduzir!)

_Você precisa notar que na arquitetura dual die da arquitetura Blackwell, ou mesmo na arquitetura 4-die do Rubin Ultra, e potencialmente em arquiteturas 3D-DRAM futuras, _essa álgebra simplifica muitos problemas_. E aqui a física dos semicondutores começa a fazer falta.

>**Multi-Die Architectures: por que Blackwell é dual die, Rubin Ultra 4-die e o futuro 3D-DRAM tornam o **CuTE** simplista demais**
>
>Um die é simplesmente o pedaço físico de silício em que os transistores são criados. Devido ao limite físico do reticle, o tamanho máximo que a máquina de litografia da [TSMC](https://www.tsmc.com/english) consegue expor de uma vez só, cerca de $858 \text{ mm}^2$, nenhum dos chips grandes usados atualmente cabe em um único die.
>
>A solução é o design **multi-die**, ou chiplet: vários dies são fabricados separadamente e depois interconectados dentro do mesmo pacote com interlinks de altíssima largura de banda, [NV-HBI](https://wccftech.com/nvidia-blackwell-ai-deep-dive-nv-hbi-fuse-two-ai-gpus-together-5th-gen-tensor-cores-5th-gen-nvlink-spectrum-x/) no caso da NVIDIA).
>
>- **Blackwell (B200/GB200) $\rightarrow$ dual-die**: dois dies de computação, cada um já no limite do reticle, conectados por um interposer com $10 \text{ TB/s}$ de banda bidirecional. Parece um chip único, mas são dois dies colados.
>- **Rubin Ultra (previsto para 2026-2027) $\rightarrow$ 4-die (ou mais)**: vazamentos técnicos e planos de lançamento indicam que a variante Ultra usará quatro dies de computação, acrescidos de dies de I/O e memória, permitindo uma área efetiva muito maior, quase 4 vezes o reticle, com melhor rendimento.
>- **Futuras arquiteturas 3D-DRAM / 3D-stacked**: além dos dies laterais (2D), começa o empilhamento vertical real (3D) de dies de lógica acrescidos de memória [HBM](https://nvidianews.nvidia.com/news/samsung-ai-factory) diretamente em cima uns dos outros ([CoWoS-L](https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm), [HBM4 3D](https://www.supermicro.com/en/glossary/hbm4), etc.), criando caminhos de dados ainda mais complexos entre camadas.
>
>O **CuTE** foi projetado assumindo memória plana e contígua dentro de um único die ou interconexão simples. Quando você tem múltiplos dies, ou camadas 3D, os endereços físicos podem cruzar fronteiras de die com latências e larguras de banda assimétricas. O **swizzle** (permutação de endereços) ótimo em um die pode ser péssimo quando os dados estão no die vizinho. O **CuTE** abstrai tudo como se fosse um tensor único e contíguo. _Funciona. Contudo, esconde complexidades que, em escala de quatro ou ou mais dies, ou em arquiteturas sobrepostas, 3D, podem custar dezenas de porcentos de performance se não forem tratadas manualmente_.
>
>**Resumindo**: é preciso cuidado e atenção quanto mais dies, mais o modelo algébrico bonitinho do **CuTE** começa a simplificar demais a realidade física do hardware. Sem cuidado ele fica bonitinho, mas ordinário. É por isso que o autor original diz "que essa álgebra simplifica muitos problemas demais". A solução é ótima até certo ponto, mas no futuro multi-die extremo vamos precisar de extensões,abstrações novas, ou voltaremos aos truques baratos de código que usávamos nas arquiteturas anteriores.

#### Discutindo as Deficiências da arquitetura Blackwell

Até aqui, tendo me concentrado em coisas boas, neste ponto discutiremos deficiências, alguns problemas. Principalmente para dissipar o misticismo.

##### O Problema SFU do B200

Enquanto a NVIDIA escalava de forma agressiva a performance dos Tensor Cores, foi necessário adicionar uma quantidade maciça de **TMEM**. Ao mesmo tempo, o **DSMEM** implementado através das redes de interconexão entre **GPCs** consumindo uma área significativa do die. Além disso, a decisão de eliminar o **L2 Partitioning**, presente na arquitetura Hopper, acabou por ocupar ainda mais espaço no layout físico do chip. 
 
>**GPC**, **G**raphics **P**rocessing **C**luster**: a unidade estrutural de nível mais alto dentro do silício de uma **GPU NVIDIA**, logo abaixo da interface global do chip. Representa uma subdivisão física que agrupa todos os recursos de processamento. A hierarquia típica de hardware segue a seguinte ordem:
>
>$$
> \text{GPU} \rightarrow \text{GPCs} \rightarrow \text{TPCs} \rightarrow \text{ **SMs**}
>$$
>
>* **GPU**: o chip completo.
>* **GPC**: uma área de processamento que contém sua própria *Raster Engine*, para gráficos, e recursos de roteamento internos.
>* **TPC**, **T**exture **P**rocessing **C**luster: subdivisão dentro do **GPC**.
>* **SM**, **S**treaming **M**ultiprocessor: no qual residem os **CUDA Cores** e **Tensor Cores**.
>
>**No contexto do texto**: o autor original destaca que o **DSMEM**, Memória Compartilhada Distribuída, permite que um **SM** acesse a memória compartilhada de outro **SM**. Fazer isso entre **SMs** que estão dentro do mesmo **GPC** é computacionalmente barato. No entanto, permitir essa comunicação entre **GPCs** diferentes exige uma malha de interconexão física complexa e extensa atravessando o chip. É essa fiação extra, acrescida da lógica de roteamento necessária para cruzar as fronteiras dos clusters, que consumiu áreas significativas do die, competindo por espaço físico com as unidades de lógica de cálculo.
>
>Finalmente temos os **S**treaming **M**ultiprocessor, **SM**: uma unidade fundamental de construção e processamento da arquitetura de uma **GPU**. Pode ser comparado, em uma analogia simplificada, a um núcleo de uma **CPU**, porém projetado para paralelismo massivo. Um **SM** é um bloco arquitetural autônomo que agrupa os principais recursos de execução:
>
>* **CUDA Cores**: unidades para cálculos escalares de ponto flutuante ($FP32$, $FP64$) e inteiros ($INT32$).
>* **Tensor Cores**: unidades especializadas em multiplicação de matrizes.
>* **SFUs (Special Function Units)**: unidades responsáveis por calcular funções transcendentes matemáticas (como $\sin$, $\cos$, $\log$, $\exp$), que são vitais para as funções de ativação e normalização em redes neurais (ex: **Softmax**).
>* **Memória On-chip**: na qual residem o arquivo de registradores (**RF**), **Cache L1** e a Memória Compartilhada.
>
>**No contexto do texto**: O autor aponta um _trade-off arquitetural crítico_. Como as **SFUs** são componentes físicos localizados dentro de cada **SM**, a capacidade total da **GPU** de processar funções transcendentes é diretamente proporcional ao número de **SMs**. Ao reduzir a contagem de **SMs** de $132$ (Hopper) para $80$ (Blackwell) para liberar espaço físico no silício para outras tecnologias,  como **TMEM** e interconexões, a NVIDIA criou um potencial gargalo para operações como o Softmax, a menos que as **S**pecial **F**unction **U**nits, **SFUs** individuais tivessem se tornado muito mais rápidas, o que o texto original afirma não ter ocorrido.

O resultado direto dessas escolhas foi a redução do número de **SMs** por die para apenas $80$, contra $132$ na arquitetura Hopper H100. Infelizmente, as **SFUs**, que rodam junto com os **CUDA Cores**, não receberam nenhum aprimoramento de performance. Na prática, isso cria uma assimetria clara: _as operações **GEMM** ficam absurdamente rápidas, mas o cálculo do **Softmax** dentro do mecanismo de [Attention](https://frankalcantara.com/transformers-quatro/) vira um gargalo visível. Exatamente os algoritmos que mais rodam nos modelos **LLM** de hoje em dia_.

![](/assets/images/tradu11.webp)
**Figura 12**: Comparação da performance de SFU entre as arquiteturas Hopper e Blackwell da NVIDIA, destacando o gargalo do Softmax no B200. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Claro, alguns podem dizer: sem problema, só use Linear Attention. De fato, mudanças recentes nos mecanismos de Attention geraram alguma controvérsia. De um lado, há o [GDN do Qwen-Next](https://qwen3-next.com/) e o [KDA do Kimi Linear](https://www.emergentmind.com/topics/kimi-delta-attention-kda). Do outro lado, o [minmax M2](https://www.minimax.io/news/minimax-m2) abandonou o Linear Attention. Outro caminho é o [MoR do Google/DeepMind](https://medium.com/data-science-in-your-pocket/mixture-of-recursions-mor-google-deepminds-next-big-leap-bye-bye-transformers-ff43ce0a0c04), e rumores sugerem que o Universal Transformer usado no GPT-5 parece ainda estar aprimorando a potência de computação dos blocos de atenção. ([Referência](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494744&idx=1&sn=20f307c5e0fe7c5c5d62a46d81f48646&scene=21&poc_token=HBPUImmjQxGE6_fNqQEDQiPvRKHrZu9cZy_ZpVQI). Mesmo caso do anterior: te vira.) Enquanto isso, o [DSA do DeepSeek-V3.2](https://api-docs.deepseek.com/news/news250929) e o [NSA](https://medium.com/data-science-in-your-pocket/deepseek-native-sparse-attention-advanced-attention-mechanism-for-llms-6ac68fc014ff) anterior seguiram o caminho do Sparse Attention.

Concordo plenamente com a visão que DeepSeek tem adotado: _o Linear Attention não resolve o verdadeiro gargalo dos transformers atuais, que reside no acesso à memória_. A computação pura (FLOPs) escala quase linearmente com mais **Tensor Cores**, mas a largura de banda e a latência de memória não acompanham na mesma proporção. Por isso, acredito que o caminho mais promissor continua sendo o Sparse Attention, seja via padrões de esparsidade aprendidos ou fixos, desde que bem projetados.

>Um artigo recentemente publicado reforça muito essa ideia ao analisar o **S**caled **D**ot-**P**roduct **A**ttention, **SDPA**, sob a ótica de **Optimal Transport**. O autor demonstrou rigorosamente que o forward pass do mecanismo de atenção, em especial o Softmax que gera os pesos de atenção, é **exatamente equivalente** à solução de um problema de One-Sided **E**ntropic **O**ptimal **T**ransport, **EOT**. Ou seja, o Softmax não é apenas uma conveniência prática ou uma aproximação. Ele emerge naturalmente como a solução ótima quando se impõe máxima entropia no plano de transporte.
>
>[Elon Litman, “Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport”, arXiv:2508.08369](https://arxiv.org/abs/2508.08369)
>
>Pensa em um artigo importante que conecta dois campos aparentemente distintos: mecanismos de atenção em deep learning e teoria matemática de transporte ótimo. A demonstração rigorosa de que o forward pass do **SDPA** é equivalente a resolver um problema de **EOT** não apenas fornece uma fundamentação teórica sólida para o uso do Softmax, mas também abre portas para novas interpretações e melhorias. _Preciso muito estudar esse artigo com calma_.

Sob essa perspectiva matemática profunda, fica ainda mais evidente que as **SFUs** precisam obrigatoriamente acompanhar a potência dos **Tensor Cores**. O **Blackwell Ultra (B300)** tenta corrigir exatamente esse problema, embora ao custo de reduzir a performance em precisões altas (FP32/FP64), que são menos críticas para training/inference de **LLMs**.

Por tudo isso, continuo achando que o **B200 e o GB200**, versões padrão da arquitetura Blackwell, não justificam investimentos muito pesados. _O equilíbrio só fica realmente interessante a partir do B300_.

##### Estrutura de Instruções Complexa da arquitetura Blackwell

Na verdade, começando pela arquitetura Hopper, a programação assíncrona se tornou muito complexa, e a introdução do **TMEM** pelo Blackwell adicionou ainda mais complexidade. Por exemplo, todo o conjunto de instruções ****Tensor Core**** `tcgen100` (TCGen05 na figura) tem tanto instruções síncronas quanto assíncronas.

![](/assets/images/tradu12.webp)
**Figura 13**: Estrutura de instruções do **Tensor Core** na arquitetura Blackwell da NVIDIA, destacando a complexidade introduzida pelo **TMEM**. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Por outro lado, a granularidade de emissão de instruções difere. algumas são de granularidade thread, algumas de granularidade **Warp**, e o cenário **2-SM** também precisa ser considerado.

![](/assets/images/tradu13.webp)
**Figura 14**: Granularidade de emissão de instruções na arquitetura Blackwell da NVIDIA, destacando as diferentes granularidades e o cenário 2-SM. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**O Cenário 2-SM na Arquitetura Blackwell**
>
>No contexto das arquiteturas mais recentes da NVIDIA, como a **Blackwell (SM100)**, o cenário **2-SM** refere-se à evolução da unidade de gerenciamento de hardware, na qual dois **SM**, **S**treaming **M**ultiprocessors, passam a operar de forma coordenada ou compartilhada para certas tarefas de alto nível.
>
>Esta mudança é fundamental para entender como a NVIDIA está escalando o desempenho dos **Tensor Cores** sem que a complexidade do agendamento de threads individuais se torne um gargalo.
>
>Nas arquiteturas tradicionais, o **SM** era a unidade soberana de recursos. No cenário **2-SM** introduzido na Blackwell, observamos os seguintes pontos:
>
>1. **SM Duplex / Pairing**: a arquitetura Blackwell organiza os **SM** em pares que compartilham certas estruturas de "front-end", como o cache de instruções de nível 1 (**L1 Instruction Cache**) e a lógica de despacho de instruções.
>2. **MMA de 2-CTA**: como mencionado no texto, a Blackwell introduz instruções **MMA** que podem abranger **2-CTA** (dois blocos de threads cooperativos). Dado que cada **CTA** reside tipicamente em um único **SM**, uma instrução de **2-CTA** exige que dois **SM** operem em sincronia quase perfeita.
>3. **Compartilhamento de TMEM**: a **Tensor Memory (TMEM)**, por ser desacoplada, pode ser acessada de forma que dois **SM** vizinhos dentro de um cluster consigam coordenar a movimentação de dados com maior eficiência do que se estivessem operando de forma totalmente independente.
>
>A tabela a seguir organiza a evolução da granularidade, mostrando como o hardware gerencia o que deve ser executado:
>
>| Nível de Granularidade | Arquitetura de Origem | Unidade de Execução |
>| :--- | :---: | :--- |
>| **Thread** | Volta (SM70) | Cada thread tem seu próprio **PC** e estado individual. |
>| **Warp** | Pascal e anteriores | Grupo fixo de 32 threads executando em **lockstep**. |
>| **Warp Group** | Hopper (SM90) | Grupo de 128 threads (4 **Warps**) cooperando em um único **WGMMA**. |
>| **2-SM / Cluster** | Blackwell (SM100) | Dois ou mais **SM** sincronizados para instruções **MMA** de larga escala (**2-CTA**/**4-CTA**). |
>
>O cenário **2-SM** justifica a introdução de mecanismos de controle mais sofisticados, como o `ClusterLaunchControl`.
>
>À medida que as matrizes operadas pelos **Tensor Cores** crescem, um único **SM** não possui registradores ou memória compartilhada suficientes para manter todos os acumuladores e operandos de um tile gigante de uma só vez. Ao permitir que a instrução seja emitida com granularidade de **2-SM**, a NVIDIA possibilita que o hardware trate os recursos de dois multiprocessadores como um único pool de recursos para aquela operação específica, reduzindo a necessidade de idas e vindas à memória global (**VRAM**).
>
>Esse cenário **2-SM** é o passo intermediário necessário para a transição para instruções **MMA** de **4-CTA**, as quais exigirão a coordenação de quatro **SM**, ou dois pares de **2-SM**, trabalhando em conjunto.

É fácil cometer erros se a sincronização não for bem cuidada. Felizmente, a NVIDIA introduziu várias abstrações de pipeline que escondem grande parte dessa complexidade e evitam muitos do erros que esperados.

Junto com o mecanismo alloc/dealloc do **TMEM**, essas abstrações reduzem a dificuldade de gerenciar a memória quando há dezenas de milhares de threads rodando em paralelo.

Como mostra a Figura 15, abaixo, **Scheduler Warp**, **TMA Warp** e **TC Warp** conseguem operar de forma praticamente single-thread, é como se cada um fosse um thread normal. Só o **Epilogue Warp** ainda precisa do comportamento clássico do **SIMT**.

**Resumindo**: depois que você pega o jeito, não é tão assustador quanto parece, mas exige atenção constante. Eu mesmo, depois de passar tanto tempo escrevendo código assíncrono para **GPU**, já não acho tão complicado. _Só para não deixar passar, este eu é o autor principal do artigo original, não eu que estou versionando. Para este pobre versionador, código assíncrono para **GPU** ainda é um bicho de onze cabeças_.

![](/assets/images/tradu14.webp)
**Figura 15**: Abstrações de pipeline na arquitetura Blackwell da NVIDIA, destacando o funcionamento quase single-thread das diferentes **Warps**. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**Especialização de Warps: Da Execução Monolítica à Decomposição Funcional**
>
> Este conceito refere-se à técnica de **Warp Specialization** (Especialização de Warps), introduzida de forma mais robusta nas arquiteturas Hopper e Blackwell. Em vez de todas as threads de um bloco realizarem a mesma tarefa, os **Warps** são divididos em papéis funcionais distintos, operando em uma estrutura de produtor-consumidor.
>
>**A Abstração de Single-Thread**: quando o autor menciona que os Warps de **Scheduler**, **TMA** e **TC** operam de forma "praticamente single-thread, ele se refere ao **desacoplamento funcional** permitido pelos **Proxies** de hardware:
>
>* **Scheduler Warp**: gerencia o estado da máquina, orquestrando as barreiras de sincronização e o fluxo das instruções.
>* **TMA Warp**: atua como o produtor de dados. Ele instrui o **Tensor Memory Accelerator** a realizar cópias massivas. Do ponto de vista do programador, ele se comporta como uma linha de execução lógica única que "manda" o hardware buscar dados.
>* **TC Warp**: atua como o consumidor. Ele gerencia as operações nos **Tensor Cores**. A complexidade da computação paralela está encapsulada na instrução **MMA**, permitindo que o controle do Warp pareça uma execução sequencial simples.
>
>O hardware esconde a complexidade do **lockstep** interno, permitindo que cada **Warp** especializado foque em sua tarefa sem se preocupar com a sincronização detalhada de cada uma das $32$ threads individuais para aquela função específica.
>
> O **Epilogue Warp** é a fase na qual os resultados processados, frequentemente acumulados em alta precisão, são convertidos, escalonados e finalmente escritos na memória global (**VRAM**). Esta etapa mantém o comportamento **SIMT** clássico pelos seguintes motivos:
>
> 1. **Coalescência de Memória**: Para obter a largura de banda máxima, os acessos à **VRAM** devem ser agrupados. O modelo **SIMT** é ideal para isso, pois permite que 32 threads calculem endereços contíguos e realizem uma única transação de memória unificada.
> 2. **Transformações Elemento-a-Elemento**: Operações de epílogo como a aplicação de funções de ativação ([**ReLU**, **GELU**](https://frankalcantara.com/transformers-quatro/)) ou adição de $bias$ são inerentemente paralelas e independentes por elemento. O paralelismo de dados massivo do **SIMT** é a ferramenta mais eficiente para processar esses milhares de elementos finais antes da escrita.
>
>**Em resumo**: enquanto a coordenação e a computação matricial se tornaram assíncronas e "especializadas", a escrita final de dados ainda depende da força bruta e da organização matemática do modelo vetorial original da NVIDIA para evitar gargalos no barramento de memória.

### Questões da CPU

Embora o [NVLink C2C](https://www.nvidia.com/en-us/data-center/nvlink-c2c/) tenha sido introduzido já na arquitetura Hopper, permitindo que a **CPU** [Grace](https://www.nvidia.com/en-us/data-center/grace-cpu/) se conecte diretamente às **GPUs** Hopper ou Blackwell, o processador Grace ainda tem vários problemas sérios.

>**Os problemas persistentes da **CPU** Grace em 2025 – por que ela ainda é o calcanhar de Aquiles do GB200/GB300**
>
>Mesmo em novembro de 2025, quase dois anos após o anúncio do [Grace Hopper](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) e mais de um ano após os primeiros [GB200](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) em produção, a Grace continua sendo o ponto mais criticado da plataforma NVIDIA para aplicações de inteligência artificial em alta escala. Por exemplo:
>
>1. **O pesadelo dos killer microseconds**: os kernels na arquitetura Blackwell são tão rápidos que muitos rodam em poucas dezenas ou centenas de microssegundos. Quando a **GPU** termina o trabalho e sinaliza a **CPU**, o overhead de interrupção acrescido do tempo de scheduling da **CPU** vira uma fração significativa do tempo total. Em workloads reais de LLM, especialmente com contextos longos ou batchs pequenos, isso pode custar entre $30\%$ e $50\%$ do throughput total. A NVIDIA melhorou um pouco com **CPU-GPU** sincronismo mais fino no [NVLink 5](https://www.amax.com/fifth-generation-nvidia-nvlink/), mas o problema fundamental permanece: a Grace não foi projetada para lidar com latências de microssegundos.
>
>2. **kernel launch ainda lento**: o tempo de lançamento de kernel **CUDA** pelo Grace continua na casa dos $15$ - $25 \, \mu s$, medido por vários benchmarks independentes em 2025. Isso é alguma coisa entre $3$ e $5$ vezes mais lento que um [Xeon](https://www.intel.com.br/content/www/br/pt/products/details/processors/xeon.html) ou [EPYC](https://www.amd.com/en/products/processors/server/epyc.html) moderno. **CUDA Graph** e persistent kernels ajudam em workloads muito regulares, notadamente em treinamento de modelos LLM, mas em ambientes dinâmicos ([vLLM](https://vllm.ai/), [TGI](https://huggingface.co/docs/text-generation-inference/en/index), [Triton](https://developer.nvidia.com/dynamo-triton) com batch variável) o overhead volta a aparecer. Até hoje não existe uma solução completa para isso na Grace.
>
>3. **Cache L2 reduzido (1 MB em vez de 2 MB)**: esse é o pecado original que a NVIDIA nunca consertou. O [Neoverse V2](https://www.arm.com/products/silicon-ip-cpu/neoverse/neoverse-v2) permite até 2 MB de **L2** por núcleo. Os concorrentes [AWS Graviton4](https://aws.amazon.com/pt/ec2/graviton/), [Ampere Altra Max](https://amperecomputing.com/products/processors) e Microsoft Cobalt todos usam $2 \text{ MB}$. A NVIDIA cortou para $1 \text{ MB}$ por razões físicas, a relação de área/die size. Resultando em uma  taxa de **I-cache miss**, falha no cache de instruções, absurdamente alta em código **CUDA** em tempo de execução. Vários clientes relatam que _o Grace gasta entre $20\%$ e $30\%$ dos ciclos em latência de espera no cache de instruções em cargas de trabalho reais de algoritmos de inteligência artificial_.
>
>4. **Network-on-Chip (NoC) em malha com latência alta**: a Grace usa uma malha 2D mesh em vez de ring ou crossbar, ambos mais eficientes. Acesso L3 pode levar vários hops dependendo da localização do núcleo. Em tráfego ScaleOut (NVLink Switch ou InfiniBand/Ethernet), os pacotes de **R**emote **D**irect **M**emory **A**ccess, **RDMA**, têm que atravessar toda a NoC (Network-on-Chip) da Grace antes de chegar na **GPU** via [NVLink-C2C](https://www.nvidia.com/en-us/data-center/nvlink-c2c/). Isso adiciona dezenas de nanossegundos de latência que, em microssegundos totais, viram gargalo visível.
>
>>Para compreender por que a performance em clusters de larga escala pode ser afetada pela Grace, precisamos analisar a viagem que um dado realiza da rede até o processamento final.
>>>1. **A Chegada via Tráfego ScaleOut**: em ambientes de inteligência artificial distribuída, a comunicação entre diferentes servidores (nós) é chamada de ScaleOut. Esse tráfego utiliza tecnologias de altíssima velocidade, como InfiniBand, Ethernet de alta performance ou o NVLink Switch. O objetivo é mover bilhões de parâmetros entre as GPUs do cluster com o mínimo de interferência possível.
>>>2. **O Papel do RDMA**: para otimizar essa movimentação, utiliza-se o **RDMA** (**R**emote **D**irect **M**emory **A**ccess). Esta tecnologia permite que os dados sejam transferidos diretamente de uma memória remota para a memória local sem sobrecarregar a **CPU** com o processamento dos pacotes. Teoricamente, o **RDMA** deveria garantir o caminho mais curto e rápido.
>>>3. **O Labirinto Interno a NoC da Grace**: o problema surge na arquitetura física do chip Grace. Quando o pacote **RDMA** chega da rede externa, ele entra no silício da **CPU**, mas para alcançar a **GPU**, ele precisa atravessar a **NoC** (**N**etwork-**o**n-**C**hip). A NVIDIA optou por uma topologia de Malha 2D (Mesh) para a **NoC** da Grace. Diferente de uma conexão direta (crossbar), a malha exige que o dado salte por diversos roteadores internos entre os núcleos da **CPU**. Cada um desses saltos (hops) adiciona dezenas de nanossegundos à latência total.
>>>4. **A Ponte Final NVLink-C2C**: somente após percorrer esse labirinto interno é que o dado alcança a interface NVLink-C2C (Chip-to-Chip), a ponte de $900 \text{ GB/s}$ que finalmente entrega o pacote à **GPU**.
>
>Embora dezenas de nanossegundos pareçam insignificantes isoladamente, em uma rede de microssegundos, esse atraso acumulado na **NoC** torna-se um gargalo incômodo e indesejado. O dado acaba perdendo tempo navegando pela estrutura física da **CPU** antes de chegar na **GPU**, reduzindo a eficiência do paralelismo massivo exigido pelos modelos de linguagem modernos.
>
>5. **Foco de marketing quase exclusivo em HPC**: até hoje, praticamente todo case de sucesso que a NVIDIA mostra com a Grace é **HPC** tradicional (simulações científicas, clima, CFD). Em algoritmos de inteligência artificial puros, especialmente durante o processo de inferência, os clientes preferem x86 (Xeon + H200/B200 ou EPYC + MI300X) ou até mesmo soluções AMD Instinct + **CPU**. a Grace só brilha mesmo quando o workload é altamente paralelizado e com kernels longos, neste caso, em treinamento de modelos gigantes.
>
>**Resumindo**: em 2025 a Grace continua sendo uma **CPU** muito boa para **HPC**. Porém, apenas aceitável para uso em inteligência artificial. A NVIDIA claramente priorizou densidade e integração NVLink em detrimento de single-thread performance e latência baixa, exatamente o oposto do que os modelos LLMs precisam. Quem quer performance máxima em inferencia ainda prefere sistemas **CPU x86 + GPU NVIDIA** ou migra para soluções que minimizam o papel da **CPU**.

Com a performance da arquitetura Blackwell atingindo níveis sem precedentes, o tempo de execução de diversos kernels caiu para a escala de microssegundos. Esse fenômeno ressuscitou um dilema clássico da arquitetura de sistemas conhecido como **Killer Microseconds**.

A regra geral para o gerenciamento dessas latências pode ser resumida da seguinte forma:

* **Latências de nanossegundos**: indicam que uma espera síncrona simples (`busy waiting`) resolve o problema com eficiência.
* **Latências de milissegundos**: indicam que o custo da troca de contexto (`context switch`) é irrelevante frente ao tempo total de espera.
* **Latências de microssegundos**: representam um pesadelo para a **CPU**, pois são longas demais para manter o processador em espera, mas curtas demais para que o sistema operacional realize a troca de contexto sem que o custo fixo (`overhead`) degrade a performance.

> **Nota Técnica**: O problema dos **Killer Microseconds** ocorre porque o tempo gasto pelo sistema operacional para salvar o estado de um thread e carregar outro é frequentemente similar ou superior ao próprio tempo de execução da tarefa de microssegundos, resultando em uma subutilização severa do hardware.

Mesmo com todas as otimizações assíncronas que a NVIDIA introduziu, a Grace ainda engasga bastante. Dois exemplos concretos:

1. O kernel launch continua lento demais. **CUDA Graphs** e persistent kernels ajudam, mas nem todo workload real permite usá-los.
2. A microarquitetura a Grace tem recortes importantes: apesar de usar o [Neoverse V2](https://www.arm.com/products/silicon-ip-cpu/neoverse/neoverse-v2), o núcleo [ARM](https://www.arm.com/) mais forte da época, a NVIDIA cortou o cache **L2** de $2 \text{ MB}$ para apenas $1 \text{ MB}$. O [Graviton 4](https://aws.amazon.com/pt/ec2/graviton/) da AWS, que usa o mesmo núcleo V2, manteve os $2 \text{ MB}$ de **L2**. Grande parte dos casos de altíssimos índices de erros no cache de instruções, I-Miss Cache, que clientes estão vendo em GB200 podem ser atribuidos a esta redução do cache **L2**.

Não é por acaso que, quando a NVIDIA promove a Grace, o foco é quase exclusivamente em aplicações **HPC** tradicionais. Exatamente as aplicações que menos sofrem com esses gargalos de microssegundos. Cenário em que a Grace realmente brilha.

![](/assets/images/tradu15.webp)
**Figura 16**: Foco de marketing da NVIDIA na **CPU** Grace, destacando a ênfase em aplicações H**PC** tradicionais. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Outro ponto enfatizado é o equilíbrio de maior largura de banda e capacidade de memória, escolhendo [LPDDR5x](https://semiconductor.samsung.com/dram/lpddr/lpddr5x/), e estendendo ainda mais as capacidades de acesso à memória das arquiteturas Hopper e Blackwell via NVLink C2C. Adicionalmente, toda a rede on-chip é uma arquitetura Mesh. A latência de acesso **L3** requer múltiplos hops no **NOC** (Network on Chip), o que tem um impacto significativo.

![](/assets/images/tradu16.webp)
**Figura 17**: Arquitetura Mesh da Network-on-Chip (NoC) na **CPU** Grace, destacando o impacto da latência de acesso L3. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Por outro lado, como o CX7 pareado com o GB200 não tem um Switch **PCIe** embutido, o tráfego ScaleOut RDMA deve atravessar toda a NOC do Grace e depois alcançar o Blackwell via NVLink C2C.

![](/assets/images/tradu17.webp)
**Figura 18**: Tráfego ScaleOut RDMA atravessando a Network-on-Chip (NoC) da **CPU** Grace para alcançar a **GPU** Blackwell via NVLink C2C. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

### Desafios de Integração e o Gargalo de Rede na Arquitetura Grace

Essa configuração arquitetural gera uma série de ineficiências sistêmicas. Visto que a placa de rede **ConnectX-7** (**CX7**) pareada com o GB200 não possui um **Switch PCIe** integrado, o trajeto do tráfego **ScaleOut RDMA** torna-se excessivamente longo e complexo. Os dados são obrigados a atravessar toda a **NoC** (Network-on-Chip) da CPU Grace para, somente então, alcançar o Blackwell via interface **NVLink-C2C**.

Esse "desvio" pela CPU não representa apenas um aumento na distância física, mas um encontro com sérias limitações de memória. O **cache L2 reduzido** da Grace agrava severamente a penalidade de **cache miss** durante essas transferências críticas.

Análises independentes, como as conduzidas pelo site **[Chips and Cheese](https://chipsandcheese.com/)**, demonstram que a latência de memória da Grace supera significativamente a de arquiteturas **x86** tradicionais e é consideravelmente maior que a observada no **Graviton 4**. Esse atraso decorre de dois fatores principais:

* **Topologia 2D Mesh**: o alto número de saltos, hops, necessários para os pacotes navegarem pela malha interna do silício;
* **Noisy Neighbors**: o ruído de contenção gerado por outros núcleos e processos competindo pelos recursos limitados de cache e banda da **NoC**.

![](/images/tradu18.webp)
**Figura 19**: Comparação de latência de memória entre a GH200 e outras arquiteturas, destacando as ineficiências da topologia Mesh e o impacto do ruído de vizinhos. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

A evidência prática desse gargalo já foi documentada em cenários de treinamento de ponta. O relatório técnico [Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72](https://zhuanlan.zhihu.com/p/1965822436948842025) cita explicitamente o **overhead da CPU** como um limitador fundamental de performance.

Essa deficiência não se restringe às **GPUs**; a  **[BlueField-4](https://blogs.nvidia.com/blog/bluefield-4-ai-factory/)**, por ser fundamentada na mesma arquitetura Grace, padece dos mesmos sintomas. Além disso, as gerações de adaptadores **CX8** e **CX9** apresentam falhas de design que indicam que a competência da NVIDIA em redes de altíssima densidade ainda enfrenta desafios significativos de amadurecimento.

>NVIDIA BlueField-4 combines an NVIDIA Grace CPU and NVIDIA ConnectX-9 networking to deliver 6x the compute power and support AI factories up to 4x larger than possible with NVIDIA BlueField-3, accelerating gigascale AI infrastructure.

Grandes provedores de nuvem (hyperscalers) adotaram estratégias distintas para contornar essa barreira de latência:

1. **AWS**: resolveu a questão implementando um **Switch PCIe** adicional no sistema. Isso permite que o tráfego **ScaleOut RDMA** flua diretamente entre a rede e a GPU, contornando completamente a **NoC** da Grace.
2. **Meta**: optou por uma abordagem de redundância, utilizando uma proporção de **1:1** (um processador Grace para cada GPU Blackwell), visando aliviar a carga sobre a malha de interconexão e garantir mais recursos de cache disponíveis para cada fluxo de dados.

Naturalmente, uma parcela considerável dessas limitações será dissipada na arquitetura GB300. Ao traçarmos um paralelo com o **Intel Granite Rapids** (**GNR**), notamos que ele disponibiliza a tecnologia **SNC3** (Sub-NUMA Clustering) como uma tentativa de gerenciar o cache com maior eficiência. Embora o **GNR** também apresente seus próprios desafios na **NoC** que afetam a largura de banda da memória, o ponto central aqui reside na escalabilidade.

A realidade técnica é implacável: assim que a contagem de núcleos ultrapassa um limiar crítico, a complexidade e o impacto da **NoC** tornam-se problemas sistêmicos. Isso ocorre especialmente em **CPUs** de propósito geral que dependem de **coerência de cache**, na qual o tráfego de sinalização para manter os dados consistentes entre centenas de núcleos pode saturar a malha de interconexão. Esse fenômeno é observado mesmo em processadores que não utilizam protocolos de coerência estritos.

Um exemplo histórico emblemático desse colapso de performance ocorreu durante a evolução do **Cisco QFP**, **Q**uantum **F**low **P**rocessor que ou autor relata em primeira mão, quando trabalhava na Cisco Systems, versionado a seguir:

* **Em 2004**: a configuração de $40$ núcleos e $160$ threads operava com estabilidade.
* **Em 2008**: a expansão para $56$ núcleos e $224$ threads ainda se mostrava viável dentro dos limites térmicos e de latência.
* **Terceira Geração (QFP3)**: ao atingir a marca de $224$ núcleos e $896$ threads, o sistema enfrentou uma degradação severa.

>Podemos ilustrar a relação entre o número de núcleos ($n$) e a complexidade de gerenciamento de recursos ($C$) em arquiteturas altamente centralizadas como:
>
>$$ C = f(n^k), \text{ onde } k > 1 $$
>
>Abaixo, os dados comparativos da evolução citada:
>
>| Geração / Ano | Núcleos (Cores) | Threads | Status de Performance |
>| :--- | :---: | :---: | :--- |
>| **QFP (2004)** | 40 | 160 | Estável |
>| **QFP2 (2008)** | 56 | 224 | Estável |
>| **QFP3 (Posterior)** | 224 | 896 | Degradação Severa |

>Essa trajetória demonstra que, quando um único socket de propósito geral tenta suportar centenas de núcleos, os problemas de latência, contenção e ruído na malha de comunicação surgem de forma inevitável. 

O desafio da NVIDIA com a **NoC** da Grace e a integração **2-SM** na Blackwell reflete essa mesma luta física contra o limite do que uma rede interna de silício pode gerenciar de forma eficiente.

### Memória na Arquitetura Blackwell (Dual-Die)

Um dos grandes desafios introduzidos pela arquitetura Blackwell reside na sua natureza **dual-die**. Diferente de um chip monolítico tradicional, a integração de dois dies de computação via **NV-HBI**, **NV**IDIA **H**igh **B**andwidth **I**nterface, cria um ambiente de memória não uniforme.

Quando os dados residem no die oposto ao que o **SM** está operando, o acesso à memória global (**GMEM**) deve atravessar a interconexão física. Esse trajeto gera, inevitavelmente, uma latência superior à de um acesso local. 

>Estatisticamente, podemos expressar essa relação como:
>
>$$\text{Latência}_{\text{Acesso Remoto}} > \text{Latência}_{\text{Acesso Local}}$$
>

_Esse atraso compromete drasticamente a eficiência dos **SMs**, que acabam enfrentando mais ciclos de espera (stalls) enquanto aguardam os dados vindos da **HBM** conectada ao outro die_. Sem uma coordenação precisa, o hardware subutiliza a largura de banda massiva que a arquitetura promete.

### Limitações do CUDA 13.0 e Mitigações Atuais

Até o presente momento, o **CUDA 13.0** ainda não oferece suporte nativo para o agendamento de **CTA** (Cooperative Thread Array) com afinidade de memória. Isso significa que o escalonador global de threads da NVIDIA ainda não "entende" de forma automática a proximidade física entre o bloco de threads e os dados na **GMEM**.

>não há mecanismo automático de agendamento baseado em localidade de dados na **GMEM**; otimizações como thread block clusters (introduzidos no Hopper com CUDA 12/13) melhoram proximidade em nível de hardware (GPC), mas dependem de configuração explícita pelo programador, não de detecção automática pelo escalonador.

Para amenizar esse gargalo, os desenvolvedores têm recorrido ao **CuTE Layout**. Essa abstração permite:

* **Escalonamento de Bancos**: organizar a disposição dos dados para maximizar a reutilização na memória compartilhada (**SMEM**).
* **Swizzling Avançado**: permutar os endereços de memória para alinhar o tráfego com a topologia física dos dies, reduzindo a contenção na **NoC**.

> **Previsão de Roadmap**: é muito provável que a NVIDIA lance, em versões futuras, uma **API de CTA Affinity**. Essa interface permitiria ao programador sugerir ao hardware em qual die o bloco de threads deve ser executado, baseando-se na localização dos dados. Eu deposito bastante confiança nessa evolução.

Já explorei os detalhes técnicos dessa interconexão e os desafios da coerência de cache no artigo abaixo:
> **“Nvidia GB200 Architecture Analysis 4: Análise da arquitetura Blackwell multi-die e coerência de cache”**

#### Tabela Comparativa de Acesso

| Tipo de Acesso | Caminho Físico | Impacto na Performance |
| :--- | :--- | :--- |
| **Local Die** | SM $\rightarrow$ NoC Local $\rightarrow$ HBM Local | Baixa Latência / Alta Eficiência |
| **Cross-Die** | SM $\rightarrow$ NV-HBI $\rightarrow$ NoC Remota $\rightarrow$ HBM Remota | Alta Latência / Risco de Stalls |

## Aqui Começa a Previsão da Arquitetura Vera Rubin (Rubin)

>Eu versionei, mas não alterei as previsões do autor original. Apenas traduzi e adaptei para o português com o meu estilo. Ainda que eu não concorde com todas as previsões.

A transição para a arquitetura Rubin marca o próximo grande salto da NVIDIA. Abaixo, analiso os componentes da **CPU** Vera e as especulações sobre a microarquitetura da **GPU**.

### CPU Vera

Com base no [devboard apresentado no GTC](https://www.nvidia.com/gtc/keynote/), a **CPU Vera** demonstra um foco agressivo em largura de banda e escalonabilidade.

| Especificação | Detalhe Previsto |
| :--- | :--- |
| **Memória** | $8$ canais (largura de banda estimada em $2\times$ a do Grace) |
| **Núcleos** | $88$ cores baseados em Neoverse V3 |
| **I/O** | **PCIe Gen6** com aproximadamente $80$ lanes |
| **Cache L2** | Estimado em $1 \text{ MB}$ por núcleo, apesar do suporte do V3 a $3 \text{ MB}$ |
| **Design** | Multi-die com controladores de memória/PCIe em dies separados |

A raiz do problema de latência permanece física e topológica. Enquanto a leitura do **L1** consome cerca de $3 \text{ ciclos}$, o trajeto para atravessar o **L3** e a **Mesh NoC** ultrapassa facilmente os $120 \text{ ciclos}$. Esse cenário não é catastrófico, mas a latência sistêmica real só será mitigada em uma eventual integração de um processador **Intel x86** equipado com **NVLink-C2C**.

### Especulação sobre a Microarquitetura Rubin

A arquitetura Rubin deve resolver as limitações físicas de área que começaram a sufocar a arquitetura Blackwell.

* **Escalabilidade do Tensor Core**: espera-se que a escala do núcleo dobre na dimensão $M$, exigindo um aumento proporcional na capacidade da <code>TMEM</code>.
* **I/O Die Separado**: como a área do chip atingiu o limite do reticle, a NVIDIA deverá adotar um I/O Diededicado.
* **MMA de 4-SM**: minha aposta é na utilização local de instruções **MMA** de $4-SM$ para maximizar a reutilização de dados.
* **Saturação do GPC**: o formato do **cluster CGA** deve evoluir para $4$, mantendo a estratégia de preencher o **GPC** com clusters densos.

![](/images/tradu19.webp)
**Figura 20**: Especulação sobre a microarquitetura da GPU Rubin, destacando a evolução dos clusters CGA e a integração de um I/O Die dedicado. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Outro tema, o Lossy e Lossless do RDMA... Quando o mundo inteiro está sem perdas, avançar demais também pode causar problemas. Até agora, quando você tem que pensar em Scale Across... Veja o que um pesquisador do Google tem a dizer: Falcon é um dos artigos mais importantes da década.

![](/images/tradu21.webp)
**Figura 21**: Análise do artigo Falcon do Google, destacando a importância do transporte confiável em redes RDMA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

E nós, [eRDMA](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52281/), éramos assim há 3 anos...

"[Falando sobre o artigo de transporte confiável do Google Falcon e análise comparativa do CIPU eRDMA](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495848&idx=2&sn=e55764ca731533c76e55ab4cb0bf25d4&scene=21&poc_token=HA5bSGmjwmOuyDlQqt41uRvJrStqzZ1fqwtol9sF)"

A foto a seguir é a certa, mas infelizmente tem trabalho sujo demais, detalhes diabólicos, pessoas comuns definitivamente não vão conseguir aprender...

![](/images/tradu22.webp)
**Figura 22**: Diagrama da implementação prática de eRDMA, destacando os desafios técnicos envolvidos. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

_No entanto, depois de ser derrotado no mercado, agora aprendi como ser eficiente e, muitas vezes, mesmo entendendo muitas coisas, vou seguindo lentamente o ritmo do mercado e evoluindo de acordo com a mente e a ecologia do usuário_.

>Melhor conselho do escritor original.

Voltando ao assunto, ainda existem muitas diferenças sutis na microarquitetura das **GPUs**, como algumas controvérsias entre **SIMD** e **SIMT**, o framework de escalonamento de Tasks, o design do **MBarrier**, a arquitetura dos núcleos escalares, a interconexão de **NOCs** no chip, etc... A questão mais importante é o equilíbrio entre facilidade de uso e desempenho, e o arquiteto deve garantir que o sistema não atinja um abismo de desempenho (performance cliff) devido a ineficiências de implementação..

Por um lado, é necessário um entendimento profundo do algoritmo: sob a ótica do Transporte Ótimo, o Softmax no SDPA (Scaled Dot-Product Attention) é a solução ideal. Contudo, dada a dificuldade de escalonamento de memória, a escolha recai sobre a Atenção Esparsa, e não sobre a Atenção Linear. Uma vez compreendida essa premissa, percebe-se que reduzir o throughput das SFUs é um equívoco; a Nvidia raramente comete erros de design como o visto no B200, mas, felizmente, a arquitetura foi corrigida no B300.

Para um arquiteto de silício, o maior desafio é projetar para cargas de trabalho que só existirão daqui a $3$ ou $5$ anos. É uma tarefa complexa, mas a NVIDIA possui uma vantagem competitiva baseada em sua capacidade full-stack, o que cria uma barreira de entrada formidável para outros competidores. Talvez a única aposta segura hoje seja a inteligência artificial. Refletindo sobre o que vi na Conferência de Yunqi há pouco tempo, eu ainda me via ironizando minhas próprias previsões.

![](/images/tradu23.webp)
**Figura 23**: Reflexão sobre previsões passadas e a evolução da arquitetura de modelos grandes. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Outra previsão é de meio ano atrás, algumas das quais já aconteceram com antecedência, vamos revisá-la em dois anos.

"[Minke: Prever a arquitetura do modelo grande nos próximos cinco anos?](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494117&idx=1&sn=a0f3f66faff51d407b6f52c02e2577c8&scene=21&poc_token=HB1dSGmjnofcLW52bd3ywl2qSNtQTmqj3E0Crnjn)" 》

Naturalmente, o custo do esforço de engenharia é altíssimo; foram necessários quase 20 anos de iterações para que o ciclo se completasse, e o grande triunfo foi, essencialmente, transpor a lógica dos algoritmos diretamente para o silício. Sob a ótica de semicondutores, detemos o estado da arte no design de protocolos e na implementação física, integrando rede, computação e interconexão. Recordo-me de anos atrás, quando a NVIDIA nos revelou que o roadmap do BlueField-4 e os marcos de desenvolvimento do RDMA já estavam definidos há muito tempo. Podem ficar tranquilos: a próxima geração de chips superará o desempenho do ConnectX-10 sem dificuldades.

Quanto ao paradigma de Scale-up, existem raríssimos especialistas no mundo com o meu nível de profundidade técnica; os diversos trade-offs de arquitetura envolvidos já foram exaustivamente mapeados e analisados por mim há muito tempo.

[Transporte Confiável para RDMA e ScaleUP](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21&poc_token=HJBdSGmjKKOLc81nEZj0Y5rFC3gkyYz4rAirqsXk)

Apesar de desenvolver em **CUDA** há muitos anos, adquiri recentemente um [módulo Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) para dominar integralmente a microarquitetura e o modelo de programação da Blackwell. É possível que ainda existam lacunas no suporte em nível de framework (camada de abstração); contudo, o treinamento de modelos de menor escala ao longo do próximo mês não deve apresentar maiores dificuldades.

Quanto ao algoritmo, fazemos algoritmos quantitativos há muitos anos, e treinamos modelos na Cisco para fazer controle ótimo relacionado ao aprendizado por reforço distribuído, além de algoritmos de grafos para analisar anomalias de equipamentos e acelerar buscas em bancos de dados distribuídos há muitos anos. Quanto à matemática, pode não ser ruim, afinal, estudei dezenas de cursos de matemática no departamento de matemática por vários anos, e basicamente vi claramente a direção em 2014, e ainda estudo álgebra com muita intensidade...

["Fundamentos Matemáticos na Era dos Grandes Modelos"](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586&scene=21#wechat_redirect)

### Algumas sugestões

o que direi se aplica a muitas placas nacionais (Lembre-se o autor original é chinês) e, embora sejam verdades amargas, não as vejam como um ataque aos fabricantes locais.

Na verdade, muitos projetos domésticos, especialmente os da divisão Datacom da Huawei, ainda carecem de maturidade. Em testes centralizados de aquisição, frequentemente encontro cenários de uso real que atingem um abismo de desempenho (performance cliff). Como prezo pelo rigor técnico, sempre que venço um teste, forneço feedback direto aos colegas da Huawei sobre essas limitações; felizmente, os produtos Datacom evoluíram muito e estão bem maduros nos últimos anos.

Discuti esse tema com Liao Bo e a equipe da Ascend no Huawei Turing Technology Summit há duas semanas. O 'diabo mora nos detalhes': por exemplo, na forma como a interconexão dos SMs da NVIDIA é implementada via **CGA**, no design do **MBarrier** e do A**sync Proxy**, e em como isso simplifica o software para acessos assíncronos à memória. A própria abstração da biblioteca **CuTe**. Por que o layout é tão abstrato? Ou detalhes de usabilidade em instruções **tcgen05**, como o motivo de encapsularem o `tcgen05.commit` dentro do `pipeline.consumer`.release, ou a implementação de alloc/dealloc baseada em colunas na **TMEM**.

Uma vez compreendidos, esses designs parecem simples, mas envolvem trade-offs profundos. A facilidade de uso não é apenas uma questão de ecossistema, mas da experiência de entender o que está sendo feito e o porquê. Não há atalhos; é preciso trabalhar com os pés no chão e as mãos na massa. A chamada ultrapassagem em curva pode facilmente resultar em um capotamento. Já lideramos em diversas áreas locais; vamos trabalhar juntos, passo a passo. Como dizia o provérbio: 'Se os estrangeiros conseguem, por que nós não conseguiríamos? Os chineses seriam inferiores?

Na última frase eu suprimi um palavrão, ou dois.