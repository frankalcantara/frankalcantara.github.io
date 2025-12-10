---
layout: post
title: Análise da Tecnologia das GPUs NVIDIA – Hopper → Blackwell → Rubin
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
image: /assets/images/tecnvida.webp
description: Análise da evolução das arquiteturas NVIDIA desde Volta até Blackwell, com foco nos Tensor Cores, TMA/TMEM, CuTEDSL e previsões para a geração Rubin.
date: 2025-11-23T14:39:47-03:00
lastmod: 2025-12-10T00:13:00.460Z
published: false
draft: false
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
schema:
    "@context": https://schema.org
    "@type": Article
    headline: Análise da Tecnologia das GPUs NVIDIA – Hopper → Blackwell → Rubin
    description: Análise profunda da evolução das arquiteturas NVIDIA desde Volta até Blackwell, com foco especial nos Tensor Cores, execução assíncrona, TMA/TMEM, CuTEDSL e previsões para a geração Rubin.
    author:
        "@type": Person
        name: Frank Alcantara
    datePublished: 2025-11-23
    dateModified: 2025-11-27
    publisher:
        "@type": Organization
        name: frankalcantara.com
        logo:
            "@type": ImageObject
            url: https://frankalcantara.com/assets/images/logo.png
    image: https://frankalcantara.com/assets/images/tecnvida.webp
    keywords: NVIDIA, Blackwell, Hopper, Rubin, Tensor Cores, CuTe, WGMMA, TMA, TMEM, arquitetura GPU, deep learning
    wordCount: 5120
    inLanguage: pt-BR
    license: https://creativecommons.org/licenses/by-sa/4.0/
    mainEntityOfPage:
        "@type": WebPage
        "@id": https://frankalcantara.com/2025/11/23/analise-da-tecnologia-das-gpus-nvidia.html
---

A primeira versão deste texto foi uma tradução, aumentada e comentada, para o português de uma versão em inglês feita por [Jukan](https://x.com/Jukanlosreve) de uma postagem no WeChat do autor chinês [Zarbot](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

**Eu fiz uma versão da tradução para o inglês, mas não tenho nenhuma associação com o autor, ou com o primeiro tradutor.**

>Eu vou colocar todos os meus pensamentos e pesquisas em blocos de destaque como este. Além, é claro, de incluir todos os links que eu achar necessário para facilitar o entendimento e possível aprendizado. Para que fique claro, deste ponto em diante, o "eu", ou qualquer pronome na primeira pessoa estará se referindo ao autor original do texto, Zarbot. Finalmente, porque um pouco de honestidade não faz mal a ninguém: é um texto grande, muito grande e complexo, muito complexo. Eu vou tirar, se tiver tempo, uns dez textos deste aqui. Então, se você quiser ler tudo de uma vez, prepare-se para uma maratona.

**Aqui começa a versão: TL;DR**

Eu passei algumas semanas para organizar algumas operações **GEMM** no [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) e no [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) usando o CuteDSL. Eu observei a evolução do Ampere para o Hopper e depois para o Blackwell. Por coincidência, penúltimo fim de semana, eu participei do Summit de Computação Turing da Huawei e conversei com o Dr. Liao e outras pessoas da equipe Ascend. Depois disso, [Jensen Huang](https://grokipedia.com/page/Jensen_Huang) apresentou o quadro de desenvolvimento de engenharia [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) e [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) ao vivo na Keynote do GTC. Portanto, eu preparei uma análise abrangente e uma previsão da microarquitetura da nova geração (É só um palpite, um esforço de adivinhação, não me culpe se eu estiver errado).

>Essa arquitetura foi previamente discutida, focando na distribuição de tensão contínua para alimentação [neste artigo](https://frankalcantara.com/nvidia-fabrica-de-ia/).

![](/assets/images/tradu1.webp)
**Figura 1**: a tabela apresenta o conjunto de instruções TCGen100 do Blackwell organizadas em dois grupos: instruções síncronas para gerenciamento de recursos e sincronização (alloc, dealloc, relinquish_alloc_permit, fence, wait e commit) e instruções assíncronas para operações computacionais e de movimentação de dados (mma para multiplicação de matrizes, cp para cópia, shift para deslocamento, ld para leitura e st para escrita). Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**GEMM**, **G**eneral **M**atrix **M**ultiply, representa a operação fundamental de multiplicação de matrizes expressa como:
>
>$$C = \alpha \cdot (A \times B) + \beta \cdot C$$
>
>na qual $A$ é uma matriz $m \times k$, $B$ é uma matriz $k \times n$, e $C$ é uma matriz $m \times n$. Os escalares $\alpha$ e $\beta$ são fatores de escala. Esta operação é a base computacional de praticamente todos os algoritmos de deep learning modernos, representando tipicamente mais de $90\%$ do tempo de execução em redes neurais.
>
>A complexidade computacional de **GEMM** é $O(m \cdot n \cdot k)$, mas a complexidade de acesso à memória é $O(m \cdot k + k \cdot n + m \cdot n)$. Esta discrepância cria o desafio fundamental: quanto maior a matriz, maior a razão `compute-to-memory-access`, permitindo melhor utilização do hardware.
>
>**CuteDSL**, **CuTE** Domain-Specific Language,:é uma abstração algébrica desenvolvida pela Nvidia para expressar layouts de tensores e operações sobre eles de forma composicional. A abstração fundamental é o
conceito de Layout, definido algebricamente como uma função:
>
>$$\text{Layout} : \mathbb{Z}^n \rightarrow \mathbb{Z}$$
>
>que mapeia coordenadas multidimensionais lógicas para offsets lineares em memória. Um 
Layout é especificado por um par (Shape, Stride), tal que:
>
>- Shape é uma tupla hierárquica descrevendo as dimensões lógicas
>- Stride é uma tupla hierárquica descrevendo os passos em memória
>
>Por exemplo, o layout de uma matriz $4 \times 8$ armazenada em row-major seria:
>
>```shell
>Layout<Shape<_4, _8>, Stride<_8, _1>>
>```
>
>A álgebra do **CuteDSL** permite composição de layouts através de operações como:
>
>1. **Particionamento**: dividir um tensor em tiles menores
>2. **Composição**: combinar múltiplos layouts
>3. **Swizzling**: permutações complexas para otimizar acesso à memória
>
>O aspecto fundamental do **CuteDSL** é que ele abstrai completamente a complexidade do swizzling de memória necessário para evitar bank conflicts na shared memory. No Hopper e Blackwell, padrões de swizzle de 128 bits são necessários para maximizar throughput de acesso à memória, e o **CuteDSL** calcula automaticamente os índices corretos.

Na minha análise, o verdadeiro diferencial competitivo da Nvidia, seu fosso econômico, transcende as polêmicas habituais sobre o ecossistema **CUDA** ou o modelo **SIMT**. A grande vantagem reside na capacidade de abstrair a complexidade do hardware, resolvendo o que poderíamos chamar de trabalho pesado de engenharia de forma transparente dentro da arquitetura. Essa integração vertical, que vai do algoritmo ao silício, equilibra programabilidade e desempenho de uma maneira que inspira o design de novos processadores concorrentes. Além disso, o timing de lançamento e a execução de mercado são cirúrgicos.

Claro, toda arquitetura tem seus trade-offs e deficiências. Em seguida, eu discutiremos muitos problemas da Nvidia que afetam tecnologias como a [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), a [Grace](https://www.nvidia.com/pt-br/data-center/grace-cpu/) e a recém-lançado [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/). Depois, assumindo que eu fosse um arquiteto para a [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference), vamos discutir as evoluções possíveis.

## Um Processo Evolucionário: do Volta ao Blackwell

Começando com a introdução dos **[Tensor Cores](https://www.digitalocean.com/community/tutorials/understanding-tensor-cores)** no [Volta](https://www.nvidia.com/pt-br/data-center/v100/), a arquitetura **SIMT** tradicionalmente definida da Nvidia na verdade começou a sofrer um processo de disrupção. A migração completa da arquitetura só pode ser finalizada com a geração Rubin. Todo o processo se estendeu por dez anos, representando tanto iterações graduais de hardware quanto inovações progressivas de software.

>**O Modelo **SIMT** da NVIDIA**
>
>A NVIDIA denomina seu modelo de paralelismo como ****SIMT****, **S**ingle **I**nstruction, **M**ultiple **T**hreads*, Instrução Única, Múltiplas Threads. Este modelo opera entre dois paradigmas conhecidos:
>
>* ****SIMD****, **S**ingle **I**nstruction, **M**ultiple **D**ata: Processamento paralelo de elementos em vetores curtos.
>* ****SMT****, **S**imultaneous **M**ulti**t**hreading: Execução paralela de instruções de vários threads independentes (ex: HyperThreading).
>
>O **SIMT** funciona como um híbrido entre processamento vetorial e *threading* em hardware, focado no equilíbrio entre flexibilidade e eficiência.
>
>Geralmente, modelos menos flexíveis são mais eficientes. Didaticamente, aceita-se a hierarquia:
>
>* **Flexibilidade**: **SIMD** < **SIMT** < **SMT**
>* **Desempenho (em cargas compatíveis)**: **SIMD** >**SIMT** >**SMT**
>
>Embora se diga que o "**SIMT** é um **SIMD** mais flexível, é importante notar a distinção técnica: internamente, a execução dos grupos de threads, chamados de *warps*, ainda ocorre de forma sincronizada, *lockstep*, similar ao **SIMD**. Pesquisadores frequentemente definem o **SIMT** mais precisamente como "**SIMD** com abstração de thread e máscara de execução.
>
>Além disso, enquanto a hierarquia que supomos acima seja útil, alguns arquitetos de sistemas consideram **SMT** e **SIMT** como filosofias opostas: o **SMT** busca maximizar as instruções por ciclo, **IPC** de poucas threads complexas, enquanto o **SIMT** sacrifica o **IPC** individual em favor da vazão total, throughput, de milhares de threads simples.
>
>**SIMT vs. SIMD**: ambos usam o *broadcast de instruções para múltiplas unidades de execução, economizando hardware de controle, em ciclo fetch/decode*. No entanto, o modelo da NVIDIA introduz três diferenciais que o **SIMD** clássico não possui:
>
>**1. Instrução Única, Múltiplos Conjuntos de Registradores**: o **SIMD** tradicional exige que o programador gerencie vetores curtos. O **SIMT** permite uma escrita escalar: o código é escrito para uma única thread, usando lógica padrão. Na prática, a **GPU** executa milhares de threads. Um Streaming Multiprocessor, **SM**, gerencia múltiplos núcleos, e todas os threads de um warp executam a mesma instrução simultaneamente, mas **cada thread possui seus próprios registradores** privados.
>>**Nuances e Custos**:
>
>>* **Redundância**: valores idênticos entre threads, tais como ponteiros base, são replicados nos registradores de cada thread, consumindo mais energia que no **SIMD** escalar único.
>>* **Tipos de Dados e Eficiência**: no **SIMD**, um registrador de $128 \text{ bits}$ pode armazenar, por exemplo, $16$ elementos de $8 \text{ bits}$. No **SIMT** puro, cada thread processa um item. Se o dado for pequeno, $8 \text{ bits}$, o hardware pode subutilizar a largura do registrador e da ALU.
>>* **Arquiteturas** como Volta e Ampere mitigam isso introduzindo caminhos de execução **SIMD** internos, como **Tensor Cores** e instruções para tipos empacotados, permitindo maior densidade de cálculo quando necessário.
>
>**2. Instrução Única, Múltiplos Endereços**: o **SIMT** permite acessos indiretos à memória ($a[i] = lut[b[i]]$). Enquanto no **SIMD** operações de leitura/escrita dispersa, gather/scatter, são complexas, no **SIMT** cada thread pode calcular seu próprio endereço de memória.
>
>>**Desafios de Memória**:
>
>>* **DRAM**: se os endereços calculados pelas threads de um warp não forem sequenciais, a **GPU** não consegue realizar a coalescência, agrupamento de acessos. Isso força múltiplas transações de memória, degradando a performance.
>>* **Memória Compartilhada**: acessos concorrentes a bancos de memória diferentes na mesma memória compartilhada são rápidos. Se múltiplas threads acessarem o mesmo banco, ocorre conflito e serialização.
>
>**3. Instrução Única, Múltiplos Caminhos de Fluxo**: o **SIMT** suporta divergência de fluxo, $if/else$. Se threads de um mesmo *warp* tomarem caminhos diferentes, o hardware usa uma **máscara de execução**.
>
>>**Execução**: primeiro, executam-se os threads que entraram no ramo $if$, enquanto as outras ficam inativas/mascaradas. Depois, inverte-se a máscara e executam-se os threads do $else$. A correção lógica é mantida, mas a execução torna-se serializada, reduzindo a eficiência proporcionalmente à divergência.
>
>**Estratégias de Ocultação de Latência**
>
>A comparação com o **SMT** revela a estratégia central da **GPU**: usar threads para ocultar a latência, não para maximizar o uso da pipeline de uma única thread.
>
>**SMT** (**CPU**): **CPUs** usam execução fora de ordem, predição de desvio e caches enormes para fazer uma única thread ser veloz. O **SMT** é uma medida auxiliar para preencher lacunas de ociosidade.
>
>**SIMT** (**GPU**): a **GPU** assume que a latência de memória é alta e não tenta combatê-la com caches complexos ou predição. A estratégia é a **força bruta do paralelismo**. Se um warp trava esperando dados da memória, o hardware troca instantaneamente para outro warp pronto.
>
>**Context Switching e Registradores**: diferente de sistemas operacionais tradicionais, a **GPU** não realiza trocas de contexto salvando dados na memória principal, o que seria proibitivamente lento. Neste caso:
>
>* Todos os contextos das threads ativas devem residir **nos registradores on-chip**.
>* Se não houver registradores suficientes para todas os threads desejados, o kernel simplesmente não é iniciado, ou a ocupação é reduzida.
>* Isso explica a quantidade massiva de registradores nas **GPUs**, até 16KB por bloco ou mais. Funcionalmente, esses registradores atuam como uma memória local extremamente rápida e particionada.
>
>**Custos e Limitações**
>
>A simplicidade do hardware, sem predição complexa, em troca de vazão traz limitações claras:
>
>>1. **Dependência de Ocupação**: a performance depende de haver warps suficientes para alternar e ocultar a latência. Sem paralelismo massivo, a **GPU** é ineficiente.
>>2. **Penalidade por Divergência**: threads não relacionados competem por recursos e a divergência de fluxo quebra o modelo de execução em *lockstep*.
>>3.  **Sincronização**: historicamente, a sincronização era restrita a barreiras dentro do mesmo bloco (`__syncthreads()`). A partir de arquiteturas como Kepler e Volta, surgiram primitivas mais ricas, como os `Cooperative Groups`, `__syncwarp()`, permitindo maior controle.  Ainda assim, a sincronização global (entre blocos) ou com o host, **CPU** permanece muito mais restritiva que em ambientes **SMT**, devido à inviabilidade de colocar milhares de threads em espera passiva gerenciada por um sistema operacional.

### Tensor Core

De uma perspectiva de hardware, olhando desde as instruções **FMA** mais antigas para o vetorizado DP4A, depois para a primeira geração Tensor Core no Volta (SM70), e subsequentemente Ampere/Hopper/Blackwell, todos eles têm aumentado a escala da multiplicação de matrizes, melhorado a razão compute-to-memory access e suportado formatos de dados de precisão mais baixa.

![](/assets/images/tradu2.webp)
**Figura 2**: Layouts de fragmentos de registradores para instruções **MMA** em diferentes Compute Capabilities. Observe a mudança drástica na organização thread-dado (T#) e o aumento da granularidade do tile, passando do modelo síncrono de 8x8x4 do Volta (SM70) para o modelo de Warpgroup 64x16x16 do Hopper (SM90). Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).
>

| Geração (SM) | Ano de Lançamento | Instrução / Tipo de Dado | Característica Principal (Significado das Siglas) |
| :--- | :---: | :--- | :--- |
| **Pascal (SM61)** | 2016 | `DP4A` | **Dot Product 4-Accumulate**. Instrução para inteiros de 8-bit (INT8). Realiza o produto escalar de dois vetores de 4 elementos de 8 bits e acumula o resultado em um inteiro de 32 bits. Fundamental para as primeiras otimizações de inferência em INT8. |
| **Volta (SM70)** | 2017 | `F32F16F16F32_NT` | **HMMA (Half-precision Matrix Multiply-Accumulate)**. Primeira geração de Tensor Cores. Multiplicação de matrizes de 16-bit (F16), com acumulação em 32-bit (F32). O sufixo NT indica a transposição dos operandos (Não Transposto x Transposto). O layout é fixo e síncrono. |
| **Ampere (SM80)** | 2020 | `F32F16F16F32_TN` | **MMA (Matrix Multiply-Accumulate) Assíncrono**. Evolução do HMMA. Introduz a capacidade de carregar dados da memória compartilhada de forma assíncrona (`ldmatrix`) antes da computação. O layout TN (Transposto x Não Transposto) mostrado é otimizado para evitar conflitos de bancos de memória compartilhada. |
| **Hopper (SM90)** | 2022 | `F32F16F16_SS` | **WGMMA (Warpgroup Matrix Multiply-Accumulate)**. Execução por **Warpgroup** (grupo de 128 threads), não apenas por Warp. Os operandos A e B residem na **Shared Memory (SS)** e são consumidos diretamente pelos Tensor Cores, sem passar pelo arquivo de registradores das threads individuais, aumentando muito a eficiência e o tamanho do tile. |
| **Hopper/Blackwell (SM90/100)+** | 2022+ | `FP8` (E4M3 / E5M2) | **Ponto Flutuante de 8-bit**. Novos tipos de dados introduzidos massivamente com o Hopper. E4M3 (4 bits expoente, 3 mantissa) para maior alcance dinâmico, e E5M2 (5 bits expoente, 2 mantissa) para maior precisão. Requerem layouts de hardware específicos e, no Blackwell (2024), suporte a *microscaling*. |
| **Blackwell (SM100)+** | 2024 | `FP4` (E2M1) | **Ponto Flutuante de 4-bit com Microscaling**. Tipo de dado extremamente compacto para inferência de modelos gigantes (LLMs). A característica principal é o **Block Scaling**: os dados de 4 bits são acompanhados por um fator de escala de maior precisão a cada bloco de 16 ou 32 elementos. O layout de memória é complexo, intercalando dados e escalas. |

>**FMA: Fused Multiply-Add (Multiplicação e Adição Fusionada)**: trata-se de instrução clássica e fundamental em computação numérica. Ela realiza a operação:
>
>$$D = A \times B + C$$
>
>Na qual $A$, $B$ e $C$ são números escalares, ou elementos correspondentes em um vetor **SIMD** comum. Neste acrônimo, a palavra chave é Fusionada. Em processadores antigos, para fazer $A \times B + C$, o hardware precisava de dois passos:
>
>1. Calcular $Intermediario = A \times B$, e arredondar o resultado.
>2. Calcular $Resultado = Intermediario + C$, e arredondar novamente.
>
>Na instrução **FMA**, a multiplicação e a adição acontecem em **um único passo** no hardware, com apenas um arredondamento final. Isso é importante por dois motivos principais:
>
>1. **Precisão**: como há apenas um arredondamento no final, o resultado é matematicamente mais preciso.
>2. **Desempenho**: ela conta como duas operações de ponto flutuante (**FLOPs**), uma multiplicação e uma adição, mas é executada no tempo de apenas uma instrução. Isso dobra a capacidade de cálculo teórica do processador.
>Nós vimos `FMA <double>` no canto inferior esquerdo da **Figura 2**. Isso representa os núcleos **CUDA** tradicionais, não-Tensor Cores, operando com precisão dupla, FP64. Era assim que a maioria da supercomputação científica era feita antes da era da IA profunda.
>
>**MMA: Matrix Multiply-Accumulate (Multiplicação e Acumulação de Matrizes)**: a **MMA** é a evolução do **FMA** para a era da Inteligência Artificial. É a instrução que define e opera os **Tensor Cores** da NVIDIA.
>
>Em vez de operar com números individuais, a **MMA** opera com **pequenos blocos de matrizes**, chamados de **tiles**, de uma só vez. A operação matemática básica é a mesma, mas em escala matricial:
>
>$$D = A \times B + C$$
>
>Na qual $A$, $B$, $C$ e $D$ agora são **pequenas matrizes**, **tiles**. Por exemplo, na primeira geração Volta (SM70), a operação padrão era multiplicar duas matrizes 4x4 e somar a uma terceira matriz 4x4. Ou seja, cada instrução **MMA** realizava $16$ multiplicações e $15$ adições, totalizando $32$ **FLOPs**, em um único passo. Redes Neurais Profundas (Deep Learning) são, essencialmente, bilhões de multiplicações de matrizes gigantescas. Fazer isso elemento por elemento usando **FMA** seria muito lento.
>
>A instrução **MMA** permite que o hardware (o Tensor Core) pegue um bloco inteiro de dados e resolva a multiplicação e soma desse bloco em um único ciclo de clock (ou poucos ciclos, dependendo da complexidade).
>
>**A Evolução do MMA**: o **MMA** não ficou parado. Conforme vimos na tabela acima, ele evoluiu em três grandes etapas:
>
>1. **HMMA (Volta/Turing)**:  **MMA Síncrono**. Um thread individual era responsável por segurar pedaços específicos da matriz em seus registradores para que o Tensor Core operasse.
>2. **MMA Assíncrono (Ampere)**:  O hardware começou a poder buscar os dados da memória enquanto calculava o **MMA** anterior.
>3. **WGMMA (Hopper/SM90)**:  **Warpgroup MMA**. A operação **MMA** tornou-se tão grande e complexa que não é mais um único thread ou um único Warp que a gerencia, mas um grupo de 128 threads, **Warpgroup** cooperando. Os dados para o **MMA**, no **Hopper** fluem diretamente da memória compartilhada para os Tensor Cores, sem a necessidade de microgerenciamento via registradores individuais das threads.

Olhando as mudanças na precisão numérica, como mostradas abaixo, acompanhadas pelas restrições de área do chip, a geração [Blackwell Ultra (B300)](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4) já começou a cortar o custo computacional necessário aos cálculos de alta precisão.

![](/assets/images/tradu3.webp)
**Figura 3**: Evolução dos formatos de dados suportados pelos Tensor Cores da Nvidia, desde o INT8 no Pascal até o FP4 com microscaling no Blackwell. Observe a tendência de adoção de formatos de menor precisão para maximizar throughput computacional em cargas de trabalho de deep learning. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Espera-se que a geração [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) dobre a escala dos Tensor Cores, estimada em $256 \times N \times 256 \text{ bits}$. Por outro lado, eu acho que veremos uma expansão adicional da **MMA** de 2-**CTA** (**C**ooperative **T**hread **A**rray) do Blackwell para uma instrução **MMA** conjunta de 4-CTA no Rubin. No entanto, haverão demandas adicionais para agendamento dentro do **CGA**, **C**ooperative **G**roup **A**rray).

Outro problema trazido pelo aumento na capacidade computacional é a alteração do path de acesso aos dados. 

Os [Tensor Cores](https://www.digitalocean.com/community/tutorials/understanding-tensor-cores) iniciais, na plataforma Volta, começaram a partir da reutilização dos registradores do [CUDA Core](https://acecloud.ai/blog/nvidia-cuda-cores-explained/). À medida que a escala dos Tensor Cores da arquitetura [Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/) se expandiu, a alimentação eficiente de dados tornou-se crítica, para mitigar a pressão de registradores, eles introduziram o [cp.async](https://research.meekolab.com/messing-around-with-gpus-again?source=more_articles_bottom_blogs) para mover dados diretamente da memória global para a memória compartilhada. Isso permite contornar o arquivo de registradores, evitando o uso de registros temporários para cópia, reduzindo a poluição do cache L1, liberando recursos para computação matemática intensa.

>**Tensor Cores**:  Unidades de processamento especializadas em **GPUs** NVIDIA, projetadas para acelerar operações de multiplicação e acumulação de matrizes (MMA), essenciais para tarefas de aprendizado de máquina e IA. Elas permitem computação de alta performance utilizando precisão mista, como FP16, BF16 e INT8.
>
>**Pressão de registradores (Register Pressure)**:  Condição na programação de **GPUs** onde a demanda por registradores (memória ultra-rápida privada por thread) excede a disponibilidade física. Quando isso ocorre, o compilador é forçado a realizar o "derramamento" (register spilling) dos dados para a memória local (LMEM), que é significativamente mais lenta, reduzindo o desempenho do kernel.
>
>**cp.async**:  Instrução de cópia assíncrona introduzida na arquitetura Ampere (CUDA 11+). Ela permite transferir dados da memória global diretamente para a memória compartilhada **sem passar pelo arquivo de registradores**. Além de ocultar a latência de memória permitindo computação paralela, sua principal vantagem arquitetural é aliviar a pressão de registradores, liberando-os para cálculos matemáticos.
>
>**L1 (Cache L1)**:  Cache de nível 1, memória rápida próxima aos núcleos de execução. No contexto do cp.async, a instrução faz os dados contornarem o L1, bypass, para evitar a "poluição de cache". Ou seja, impedir que novos dados de streaming expulsem dados reutilizáveis que já estão no cache, depositando-os diretamente na memória compartilhada.

A evolução continuou com a arquitetura [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/), que introduziu o **T**ensor **M**emory **A**ccelerator,  **[TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)**, permitindo que operandos fossem carregados diretamente na memória compartilhada, **SMEM**, além de implementar a **CGA** e a **DSMEM**. No entanto, nesta fase, os resultados dos acumuladores ainda residiam no arquivo de registradores para facilitar as operações subsequentes de Epílogo, o que exigia o uso de barreiras de espera, wait barriers, para sincronização. Finalmente, a arquitetura Blackwell introduz o **TMEM**, desacoplando efetivamente o Tensor Core do **CUDA** Core, enquanto reaproveita o mecanismo de **MBarrier** estabelecido pelas operações assíncronas do **TMA**. Como pode ser visto na tabela abaixo:

| Arch      | Matrix A    | Matrix B | Matrix D |
| :---      | :---:       | :---:    | :---:    |
| Volta     | RF          | RF       | RF       |
| Ampere    | RF          | RF       | RF       |
| Hopper    | RF / SMEM   | SMEM     | RF       |
| Blackwell | **TMEM** / SMEM | SMEM     | **TMEM**     |

>**[TMA (Tensor Memory Accelerator)](https://research.colfax-intl.com/tutorial-hopper-tma/)**:  Motor de cópia assíncrona introduzido na arquitetura Hopper (H100). Ele gerencia a transferência de dados entre a memória global e a memória compartilhada (SMEM) de forma independente, liberando as threads para outras tarefas e lidando automaticamente com cálculos de endereço.
>
>**CGA (Cluster Group Architecture)**:  Refere-se à organização em *Thread Block Clusters*. Permite que múltiplos blocos de threads (CTAs) sejam agrupados em um "Cluster" para cooperar na execução, compartilhando dados de forma rápida e sincronizada através da hierarquia de memória.
>
>**DSMEM (Distributed Shared Memory)**:  Recurso que permite que a memória compartilhada (SMEM) de um bloco seja acessada diretamente por outros blocos dentro do mesmo Cluster. Isso elimina a necessidade de passar pela memória global para trocar dados entre blocos vizinhos.
>
>**TMEM (Tensor Memory)**:  Na arquitetura Blackwell, refere-se a uma área de memória dedicada ou um modo de operação que separa o armazenamento de dados dos Tensor Cores do caminho de dados tradicional dos **CUDA** Cores, reduzindo a contenção de registradores e permitindo pipelines mais eficientes.
>
>Com um pouco mais de detalhe, podemos dizer que **MBarrier (Memory Barrier)** é uma primitiva de sincronização em hardware introduzida formalmente na arquitetura Ampere e aprimorada significativamente no Hopper (SM90). Diferente de barreiras tradicionais de execução (como `__syncthreads()`), que pausam threads até que todas cheguem a um ponto, o **MBarrier** rastreia a conclusão de **transações de memória assíncronas**.
>
>O funcionamento baseia-se em contagem de transações:
>
>>1. As threads "chegam" à barreira e indicam quantos bytes ou transações estão aguardando (expectativa).
>>2. O hardware (como o **TMA**) decrementa o contador da barreira automaticamente à medida que os dados são escritos na memória compartilhada.
>>3. As threads podem verificar o estado da barreira e dormir ou continuar processando outras coisas até que a contagem chegue a zero.
>
>No contexto do texto, o **MBarrier** é a peça chave que permite a **Warp Specialization**: um grupo de warps pode apenas emitir comandos de cópia (produtores), enquanto outro grupo (consumidores) espera no MBarrier. Isso desacopla a latência de memória do fluxo de execução da computação, permitindo pipelines assíncronos de altíssima eficiência no Hopper e Blackwell.

Essa evolução tomou aproximadamente sete anos de desenvolvimento, partindo da arquitetura Volta, na qual os Tensor Cores operavam fortemente acoplados ao fluxo de registradores, quase como um add-on, até chegar ao Blackwell. 

Com a introdução da **[TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)**, o [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) eliminou a dependência do arquivo de registradores, RF, para os acumuladores, concretizando uma separação assíncrona entre as unidades de execução. Cada etapa dessa jornada exigiu tanto inovações profundas de hardware quanto avanços significativos nas abstrações de software.

### Processamento Assíncrono

O outro aspecto é o processamento assíncrono. Quando a geração [Volta](https://www.nvidia.com/en-au/data-center/v100/) introduziu um PC, Program Counter, independente para cada Thread, na verdade marcou o início da execução assíncrona.

![](/assets/images/tradu4.webp)
**Figura 4**: Evolução do modelo de execução assíncrona na arquitetura NVIDIA, desde a introdução do PC independente por thread no Volta até o mecanismo **MBarrier** no Hopper e Blackwell. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

A partir desse ponto, os threads podiam esperar por mensagens para realizar processamento assíncrono, abrindo uma janela para programação assíncrona em relação às arquiteturas alinhadas por PC tradicionais.

![](/assets/images/tradu5.webp)
**Figura 5**: Representação do processamento assíncrono na arquitetura NVIDIA, destacando a evolução desde o Volta até o Ampere com a introdução do cp.async. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**Um ponto que merece destaque: o "PC Independente por Thread" do Volta foi o verdadeiro Big Bang da execução assíncrona nas **GPUs** NVIDIA.**
>
>Antes do Volta (SM70), todas as threads de um warp compartilhavam **um único Program Counter (PC)**. Isso significava que, mesmo com máscara de execução, o hardware ainda avançava o PC de forma lockstep: o warp executava as duas ramificações de um `if/else` sequencialmente, mas sempre com o mesmo PC para todo o grupo. Era o clássico SIMT "puro". O mesmo processo que encontramos nas CPUs escalares, nas quais múltiplos threads compartilham o mesmo PC, mas cada thread tem seu próprio conjunto de registradores.
>
>A partir do Volta, a NVIDIA introduziu o **Independent Thread Scheduling**: cada thread passou a ter seu próprio PC visível ao programador,  e ao hardware. Isso mudou tudo:
>
>- deste ponto em diante o scheduler da **GPU** pode reagendar warps de forma mais flexível, inclusive intercalar instruções de diferentes caminhos de execução dentro do mesmo warp.
>- abriu a porta real para sincronização mais fina, syncwarp, cooperative groups avançados.
>- e, mais importante para o que vem depois: permitiu que threads individuais **esperassem por eventos assíncronos**, como chegadas de mensagens via grid-sync, ou barreiras assíncronas futuras, sem precisar que todo o warp estivesse no mesmo ponto do código.
>
>Em resumo: o Volta quebrou o dogma do lockstep rígido e criou as condições técnicas para que, anos depois, Hopper e Blackwell pudessem fazer verdadeiras pipelines assíncronas com TMA, MBarrier, TMEM, etc. Sem esse PC independente, nada do que veio depois seria possível. Pelo menos, não seria possível com a mesma elegância.
>
>Muita gente acha que a revolução assíncrona começou no Ampere com cp.async ou no Hopper com TMA. Na verdade, a semente foi plantada lá em 2017, no Volta, e quase ninguém percebeu na época o quanto isso ia ser importante.

A parte boa é que a Nvidia forneceu a abstração Cooperative Group no software. No entanto, os Tensor Cores ainda requeriam execução síncrona em todo o Warp. Então, começando com a introdução do cp.async no Ampere, o caminho de suprimento de dados de todo o programa efetivamente se tornou assíncrono, que é o conceito de **Async Thread** mencionado pela Nvidia.

![](/assets/images/tradu6.webp)
**Figura 6**: Ilustração do modelo de execução assíncrona com cp.async na arquitetura Ampere da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**Cooperative Groups: a abstração que finalmente libertou a sincronização nas **GPUs** NVIDIA do "tyranny of the thread block"**
>
>Antes do **CUDA** 9, 2017, exatamente na era Volta, a sincronização entre threads era extremamente rígida:
>
>- `__syncthreads()` → só dentro do mesmo block;
>- `__syncwarp()` → só dentro do mesmo warp, introduzido um pouco antes, mas ainda limitado;
>- Sincronização entre blocks? Praticamente impossível sem truques horrendos com global memory e loops de polling. E aqui está a palavra proíbida global.
>
>Isso era um problema gigantesco porque, à medida que os kernels [GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) e transformers ficavam maiores, querer sincronizar múltiplos blocos que cooperam em um mesmo tile grande de matriz se tornou essencial do ponto de vista da performance.
>
>A NVIDIA então lançou **Cooperative Groups**, uma abstração de software que permite ao programador definir grupos arbitrários de threads que podem se sincronizar de forma segura e eficiente:
>
>- `thread_block` → equivalente ao antigo `__syncthreads()`, mas agora como objeto;
>- `thread_block_tile` → partições estáticas dentro do block (ex: tiles de 32 threads = warp, ou 16, 8, etc.);
>- `grid_group` → o santo graal: sincronização entre **todos os blocos do grid inteiro** com uma única chamada `grid.sync()`;
>- `multi_grid_group` → sincronização entre múltiplos grids lançados de dispositivos diferentes, usado em ambientes em multi-GPU;
>- E o mais poderoso: grupos dinâmicos com `coalesced_group` e `cluster_group, disponível a partir do Hopper.
>
>No contexto da evolução que estamos discutindo: Cooperative Groups foi a camada de software que preparou o terreno para tudo que veio depois.
>
>Sem ela, não teríamos os padrões de Warp Specialization do Hopper, nem os pipelines assíncronos complexos do Blackwell que misturam producer/consumer warps dentro do mesmo CTA ou cluster. Ela transformou o modelo SIMT rígido em algo muito mais próximo de um "MIMD dentro do SIMT", dando ao programador controle explícito sobre quem sincroniza com quem.
>
>Resumindo: se o Independent Thread Scheduling do Volta foi o hardware que permitiu a divergência real, Cooperative Groups foi o software que transformou essa possibilidade em algo utilizável e performático. Juntos, eles são os verdadeiros pais da execução assíncrona moderna nas **GPUs** NVIDIA.
>
>**MIMD – Multiple Instruction, Multiple Data**
>
>É o modelo de paralelismo mais flexível que existe:
>
>- Cada núcleo/processador executa **instruções diferentes**;
>- Em **dados diferentes**;
>- De forma totalmente independente.
>
>Também conhecido em Hollywood como o Santo Graal.
>
>Exemplo prático: uma **CPU** comum com vários cores rodando programas diferentes ao mesmo tempo, um núcleo no navegador, outro no jogo, no WhatsApp etc.. Aqui, são múltiplos fluxos de controle (instruções) operando em múltiplos conjuntos de dados. Com objetivos diferentes.
>
>O MIMD é o oposto do SIMD das **GPUs** clássicas.
>
>Resumindo na frase que eu mais uso:
>**MIMD = cada um faz o que quer, quando quer, com o que quer.** Quando isso acontece para um mesmo objetivo estamos no paraíso do paralelismo.

O Hopper deu um passo adiante ao introduzir o MBarrier. Depois disso, pipelines assíncronos de software e Warp Specialization construídos em torno do **MBarrier** se tornaram populares. 

O Hopper introduziu o Async Proxy, distinguindo diferentes caminhos de acesso à memória através de General Proxy e Async Proxy. Para operações Async Proxy, geralmente há uma barreira de memória; o LD/ST (Load/Store) do General Proxy pode esperar por essa barreira para completar, permitindo que operações assíncronas do **TMA** sejam combinadas com o acesso à memória LD/ST **SIMT** original, garantindo requisitos de ordenação de memória.

![](/assets/images/tradu7.webp)
**Figura 7**: Representação do modelo de execução assíncrona com **MBarrier** na arquitetura Hopper da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Claro, o Hopper tem imperfeições. O **WGMMA** era uma solução temporária, ocupando uma grande quantidade de registradores enquanto também requeria espera síncrona. Portanto, quando o Hopper foi lançado, foi explicitamente dito que o **WGMMA** do SM_90a não seria compatível com versões anteriores. Isso teve uma grande desvantagem.

![](/assets/images/tradu8.webp)
**Figura 8**: Limitações do **WGMMA** na arquitetura Hopper da NVIDIA, destacando a necessidade de espera síncrona e o consumo elevado de RMEM. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**WGMMA – Warpgroup Matrix Multiply-Accumulate: o "MMA gigante" do Hopper que foi uma solução transitória, bem cara**
>
>No Hopper (SM90), a NVIDIA precisou escalar drasticamente o tamanho dos tiles que o Tensor Core conseguia processar de uma só vez. A solução foi o **WGMMA**: em vez de um único warp (32 threads) executar o MMA como nas gerações anteriores, agora um **Warpgroup**, grupo de 4 warps = 128 threads, executa a operação inteira em conjunto.
>
>Vantagens imediatas:
>- Tiles muito maiores (ex.: 64x128x64 ou até maiores) diretamente da shared memory;
>- Enorme aumento de throughput em **GEMM** grandes (transformers adoraram);
>- Dados fluem direto da SMEM para os Tensor Cores sem precisar passar por registradores de cada thread individual.
>
>Mas o preço foi alto, e por isso o autor chama de "solução temporária";
>
>- **Consumo brutal de registradores**: mesmo não usando os registradores das threads para os operandos, os acumuladores ainda ficavam no register file implicando em valores altíssimos de register pressure;
>- **Execução síncrona obrigatória**: o warpgroup inteiro tinha que esperar a operação terminar antes de prosseguir → limitava pipelines assíncronos;
>
>Por causa dessas limitações internas, a NVIDIA lançou duas versões do Hopper:
>
>- SM_90 (H100 padrão) → **WGMMA** síncrono, alto uso de registradores;
>- SM_90a (H100 NVL, versão posterior) → algumas melhorias, mas ainda síncrono.
>
>E explicitamente avisou: código compilado para SM_90a **não roda** em chips SM_90 antigos. O que, se ainda não percebeu, implicou na quebra de compatibilidade forward dentro da mesma geração!
>
>O **WGMMA** foi a tecnologia que permitiu ao Hopper bater records de performance em LLMs em 2023-2024, mas foi uma gambiarra heroica.

No Blackwell, o Tensor Core também se tornou uma operação totalmente assíncrona, reutilizando a construção MBarrier. Assim, emitir **TMA** e instruções pode ser feito no nível de Thread. No entanto, a alocação e cópia de memória para o **TMEM** ainda requerem manuseio no nível Warp. Por outro lado, o mecanismo ClusterLaunchControl foi introduzido, fornecendo alguma capacidade de agendamento dinâmico.

![](/assets/images/tradu9.webp)
**Figura 9**: Modelo de execução assíncrona com **TMEM** e **MBarrier** na arquitetura Blackwell da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**ClusterLaunchControl: o mecanismo que finalmente trouxe agendamento dinâmico real para dentro dos Clusters no Blackwell**
>
>No Hopper, os Thread Block Clusters já existiam, mas o lançamento e o agendamento dos CTAs dentro de um cluster eram essencialmente **estáticos**: você definia o tamanho do cluster no launch do kernel, via `cudaLaunchAttributeClusterDimension`, e todos os blocos eram lançados de uma vez, sem controle fino em runtime.
>
>No Blackwell, a NVIDIA introduziu o **ClusterLaunchControl**, ou controles de lançamento de cluster mais avançados via hardware/registros dedicados, que permite:
>
>- Threads individuais ou warps **decidirem dinamicamente** quais blocos/clusters participar ou quando ativar certas operações cooperativas;
>- Agendamento dinâmico de warps especializados dentro do mesmo cluster (ex.: um warp pode "acordar" ou lançar tarefas para outros warps/CTAs do cluster de forma assíncrona);
>- Melhor integração com **MBarrier** e **TMA** em escala de cluster, porque agora é possível coordenar produtores/consumidores de dados de forma mais flexível, sem precisar que todo o warp esteja sincronizado o tempo todo.
>
>Isso importa porque mesmo com **TMA** e **TMEM** sendo assíncronos em nível de thread, a alocação de **TMEM** ainda exigia coordenação em nível de warp. O `ClusterLaunchControl` alivia isso permitindo que a própria **GPU**, ou o programador via instruções, faça **agendamento dinâmico em escala de cluster**, abrindo caminho para padrões ainda mais sofisticados de Warp Specialization e pipelines producer-consumer multi-CTA.
>
>Resumindo: se o **MBarrier** foi o semaforo assíncrono e o **TMEM** foi a memória dedicada, o `ClusterLaunchControl` é o maestro que finalmente permite orquestrar tudo isso de forma dinâmica em runtime, aproximando a **GPU** NVIDIA de um verdadeiro modelo **MIMD** cooperativo em escala de cluster. Lembre-se estamos supondo que voamos em um céu de brigadeiro.

Podemos então construir padrões de processamento Warp Specialization mais complexos.

![](/assets/images/tradu10.webp)
**Figura 10**: Exemplos de padrões de processamento com Warp Specialization na arquitetura Blackwell da NVIDIA. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

### Layout **CuTE** 

Aqui também temos uma abstração de software fantástica, especialmente se considerarmos a forma como abstrai a complexidade do Swizzle no Hopper e no Blackwell. Além disso, de uma perspectiva puramente algébrica, esta abstração permite resolver cálculos complexos de fronteiras de Tile/Partition, tornando o código mais intuitivo, embora para aqueles que não são formados em álgebra, aprender o **CuTE** ainda represente uma curva de aprendizado íngreme. Eu comecei a discutir a álgebra de Layout **CuTE** no artigo abaixo: ([Referência](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21&poc_token=HL7TImmjizY1ndR3EjvNUMHxZ2wg3Uh_AUlNECLB) como diria meu professor Lobo: "te vira, tu não nasceu quadrado, dá seus pulos e traduz sozinho!")

Você precisa notar que na arquitetura dual die do Blackwell, ou mesmo na arquitetura 4-die do Rubin Ultra, e potencialmente em arquiteturas 3D-DRAM futuras, _essa álgebra simplifica muitos problemas demais_.

>**Multi-Die Architectures: por que Blackwell é "dual die", Rubin Ultra "4-die" e o futuro 3D-DRAM tornam o **CuTE** "simplista demais"**
>
>Um "die" é simplesmente o pedaço físico de silício em que os transistores são fabricados. Devido ao limite físico do reticle, o tamanho máximo que a máquina de litografia da [TSMC](https://www.tsmc.com/english) consegue expor de uma vez só, cerca de $858 \text{ mm}^2$, nenhum chip grande atual cabe em um único die.
>
>A solução é o design **multi-die**, ou chiplet: vários dies são fabricados separadamente e depois interconectados dentro do mesmo pacote com interlinks de altíssima largura de banda, [NV-HBI](https://wccftech.com/nvidia-blackwell-ai-deep-dive-nv-hbi-fuse-two-ai-gpus-together-5th-gen-tensor-cores-5th-gen-nvlink-spectrum-x/) no caso da NVIDIA).
>
>- **Blackwell (B200/GB200) → dual-die**: dois dies de computação, cada um já no limite do reticle, conectados por um interposer com $10 \text{ TB/s}$ de banda bidirecional. Parece um chip único, mas são dois dies "colados".
>- **Rubin Ultra (previsto para 2026-2027) → 4-die (ou mais)**: leaks e roadmaps indicam que a variante Ultra usará quatro dies de computação, acrescidos de dies de I/O e memória, permitindo uma área efetiva muito maior, quase 4× reticle, e/ou yields melhores.
>- **Futuras arquiteturas 3D-DRAM / 3D-stacked**: além dos dies laterais (2D), começa o empilhamento vertical real (3D) de dies de lógica acrescidos de memória [HBM](https://nvidianews.nvidia.com/news/samsung-ai-factory) diretamente em cima uns dos outros ([CoWoS-L](https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm), [HBM4 3D](https://www.supermicro.com/en/glossary/hbm4), etc.), criando caminhos de dados ainda mais complexos entre camadas.
>
>O **CuTE** foi projetado assumindo memória "plana" e contígua dentro de um único die ou interconexão simples. Quando você tem múltiplos dies, ou camadas 3D, os endereços físicos podem cruzar fronteiras de die com latências e larguras de banda assimétricas. O swizzle ótimo em um die pode ser péssimo quando os dados estão no die vizinho. O **CuTE** abstrai tudo como se fosse um tensor único e contíguo. Funciona, mas esconde complexidades que, em escala de 4+ dies ou 3D, podem custar dezenas de $\%$ de performance se não forem tratadas manualmente.
>
>Resumindo: é preciso cuidado e atenção quanto mais dies, mais o modelo algébrico bonitinho do **CuTE** começa a simplificar demais a realidade física do hardware. Sem cuidado ele fica bonitinho, mas ordinário. É por isso que o autor diz que "essa álgebra simplifica muitos problemas demais", ela é ótima até certo ponto, mas no futuro multi-die extremo vai precisar de extensões ou abstrações novas.

#### Discutindo as Deficiências do Blackwell

Tendo dito algumas coisas boas, neste ponto discutiremos algumas deficiências, alguns problemas. Principalmente para dissipar o misticismo.

##### O Problema SFU do B200 

Enquanto a NVIDIA escalava de forma agressiva a performance dos Tensor Cores, foi necessário adicionar uma quantidade maciça de **TMEM**. Ao mesmo tempo, o **DSMEM** implementado através das redes de interconexão entre GPCs também consumiu área significativa do die. Além disso, a decisão de eliminar o L2 Partitioning (presente no Hopper) acabou por ocupar ainda mais espaço no layout.  
  
>**GPC (Graphics Processing Cluster)**:  É a unidade estrutural de nível mais alto dentro do silício de uma GPU NVIDIA, logo abaixo da interface global do chip. Representa uma subdivisão física que agrupa recursos de processamento.
>
>A hierarquia típica de hardware segue esta ordem:
>
>$$
> \text{GPU} \rightarrow \text{GPCs} \rightarrow \text{TPCs} \rightarrow \text{SMs}
>$$
>
>* **GPU**:  O chip completo.
>* **GPC**:  Um "bairro" de processamento que contém sua própria *Raster Engine* (para gráficos) e recursos de roteamento internos.
>* **TPC (Texture Processing Cluster)**:  Subdivisão dentro do GPC.
>* **SM (Streaming Multiprocessor)**:  Onde residem os CUDA Cores e Tensor Cores.
>
>**No contexto do texto**:  O autor destaca que o **DSMEM** (Memória Compartilhada Distribuída) permite que um SM acesse a memória compartilhada de outro SM. Fazer isso entre SMs que estão dentro do mesmo GPC é relativamente barato. No entanto, permitir essa comunicação entre **GPCs diferentes** exige uma malha de interconexão física complexa e extensa atravessando o chip. É essa fiação extra e a lógica de roteamento necessária para cruzar as fronteiras dos clusters que "consumiu área significativa do die", competindo por espaço físico com as unidades de lógica de cálculo.
>
>Finalmente temos os **SM (Streaming Multiprocessor)**:  uma unidade fundamental de construção e processamento da arquitetura de uma GPU NVIDIA. Pode ser comparado, em uma analogia simplificada, a um "núcleo" de uma CPU, mas projetado para paralelismo massivo.
>
>Um **SM** é um bloco arquitetural autônomo que agrupa os principais recursos de execução:
>
>* **CUDA Cores**:  Unidades para cálculos escalares de ponto flutuante ($FP32$, $FP64$) e inteiros ($INT32$).
>* **Tensor Cores**:  Unidades especializadas em multiplicação de matrizes.
>* **SFUs (Special Function Units)**:  Unidades responsáveis por calcular funções transcendentes matemáticas (como $\sin$, $\cos$, $\log$, $\exp$), que são vitais para as funções de ativação e normalização em redes neurais (ex: **Softmax**).
>* **Memória On-chip**:  Onde residem o arquivo de registradores (Register File), Cache L1 e a Memória Compartilhada.
>
>**No contexto do texto**:  O autor aponta um *trade-off* arquitetural crítico. Como as **SFUs** são componentes físicos localizados dentro de cada SM, a capacidade total da GPU de processar funções transcendentes é diretamente proporcional ao número de SMs. Ao reduzir a contagem de SMs de 132 (Hopper) para 80 (Blackwell) para liberar espaço físico no silício para outras tecnologias (como TMEM e interconexões), a NVIDIA criou um potencial gargalo para operações como o Softmax, a menos que as SFUs individuais tivessem se tornado muito mais rápidas (o que o texto afirma não ter ocorrido).

O resultado direto dessas escolhas foi a redução do número de SMs por die para apenas **80** (contra 132 no Hopper H100). Infelizmente, as **SFUs** (Special Function Units), que rodam junto com os **CUDA** Cores, não receberam nenhum aprimoramento de performance.  
  
Na prática, isso cria uma assimetria clara: as operações **GEMM** ficam absurdamente rápidas, mas o cálculo do **Softmax** dentro do mecanismo de Attention vira um gargalo visível. Exatamente o que mais roda nos modelos LLM de hoje em dia.

![](/assets/images/tradu11.webp)
**Figura 11**: Comparação da performance de SFU entre as arquiteturas Hopper e Blackwell da NVIDIA, destacando o gargalo do Softmax no B200. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Claro, alguns podem dizer, "Sem problema, só use Linear Attention." De fato, mudanças recentes nos mecanismos de Attention geraram alguma controvérsia. De um lado, há o [GDN do Qwen-Next](https://qwen3-next.com/) e o [KDA do Kimi Linear](https://www.emergentmind.com/topics/kimi-delta-attention-kda). Do outro lado, o [minmax M2](https://www.minimax.io/news/minimax-m2) abandonou o Linear Attn. Outro caminho é o [MoR do Google/DeepMind](https://medium.com/data-science-in-your-pocket/mixture-of-recursions-mor-google-deepminds-next-big-leap-bye-bye-transformers-ff43ce0a0c04), e rumores sugerem que o Universal Transformer usado no GPT-5 parece ainda estar aprimorando a potência de computação dos blocos Attn. ([Referência](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494744&idx=1&sn=20f307c5e0fe7c5c5d62a46d81f48646&scene=21&poc_token=HBPUImmjQxGE6_fNqQEDQiPvRKHrZu9cZy_ZpVQI). Mesmo caso do anterior: te vira.) Enquanto isso, o [DSA do DeepSeek-V3.2](https://api-docs.deepseek.com/news/news250929) e o NSA anterior seguiram o caminho do Sparse Attn.

Concordo plenamente com a visão que DeepSeek tem adotado: _o Linear Attention não resolve o verdadeiro gargalo dos transformers atuais, que é o **acesso à memória**_. A computação pura (FLOPs) escala quase linearmente com mais Tensor Cores, mas a largura de banda e a latência de memória não acompanham na mesma proporção. Por isso, acredito que o caminho mais promissor continua sendo o **Sparse Attention**, seja via padrões de esparsidade aprendidos, ou fixos, bem projetados.

>Um artigo recentemente publicado reforça muito essa ideia ao analisar o Scaled Dot-Product Attention (SDPA) sob a ótica de **Optimal Transport**. O autor demonstrou rigorosamente que o forward pass do mecanismo de atenção, em especial o Softmax que gera os pesos de atenção, é **exatamente equivalente** à solução de um problema de **One-Sided Entropic Optimal Transport** (EOT). Ou seja, o Softmax não é apenas uma “conveniência prática” ou uma aproximação: ele emerge naturalmente como a solução ótima quando se impõe máxima entropia no plano de transporte.
>
>[Elon Litman, “Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport”, arXiv:2508.08369](https://arxiv.org/abs/2508.08369)
>

Sob essa perspectiva matemática profunda, fica ainda mais evidente que as **SFUs** (Special Function Units) precisam obrigatoriamente acompanhar a potência dos Tensor Cores. O **Blackwell Ultra (B300)** tenta corrigir exatamente esse problema, embora ao custo de reduzir a performance em precisões altas (FP32/FP64), que são menos críticas para training/inference de LLMs.

Por tudo isso, continuo achando que o **B200 e o GB200**, versões padrão do Blackwell, não justificam investimentos muito pesados. _O equilíbrio só fica realmente interessante a partir do B300_.

##### Estrutura de Instruções Complexa do Blackwell

Na verdade, começando pelo Hopper, a programação assíncrona se tornou muito complexa, e a introdução do **TMEM** pelo Blackwell adicionou ainda mais complexidade. Por exemplo, todo o conjunto de instruções Tensor Core tcgen100 (TCGen05 na figura) tem tanto instruções síncronas quanto assíncronas.

![](/assets/images/tradu12.webp)
**Figura 12**: Estrutura de instruções do Tensor Core na arquitetura Blackwell da NVIDIA, destacando a complexidade introduzida pelo **TMEM**. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Por outro lado, a granularidade de emissão de instruções difere. algumas são de granularidade thread, algumas de granularidade warp, e o cenário 2-SM também precisa ser considerado.

![](/assets/images/tradu13.webp)
**Figura 13**: Granularidade de emissão de instruções na arquitetura Blackwell da NVIDIA, destacando as diferentes granularidades e o cenário 2-SM. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

É fácil cometer erros se a sincronização não for bem cuidada. Felizmente, a NVIDIA introduziu várias abstrações de pipeline que escondem grande parte dessa complexidade e evitam muitos do bugs que são comumente esperados.

Junto com o mecanismo alloc/dealloc do **TMEM**, essas abstrações reduzem a dificuldade de gerenciar a memória quando há dezenas de milhares de threads rodando em paralelo.

Como mostra a Figura 14, abaixo, Scheduler Warp, TMA Warp e TC Warp conseguem operar de forma praticamente single-thread, é como se cada um fosse uma thread normal. Só o Epilogue Warp ainda precisa do comportamento clássico do SIMT.

Resumindo: depois que você pega o jeito, não é tão assustador quanto parece, mas exige atenção constante. Eu mesmo, depois de passar tanto tempo escrevendo código assíncrono para GPU, já não acho tão complicado.

![](/assets/images/tradu14.webp)
**Figura 14**: Abstrações de pipeline na arquitetura Blackwell da NVIDIA, destacando o funcionamento quase single-thread das diferentes warps. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

### Questões da CPU

Embora o [NVLink C2C](https://www.nvidia.com/en-us/data-center/nvlink-c2c/) tenha sido introduzido já no Hopper, permitindo que a **CPU** [Grace](https://www.nvidia.com/en-us/data-center/grace-cpu/) se conecte diretamente às **GPUs** Hopper ou Blackwell, o processador Grace ainda tem vários problemas sérios.

>**Os problemas persistentes da **CPU** Grace em 2025 – por que ela ainda é o "calcanhar de Aquiles" do GB200/GB300**
>
>Mesmo em novembro de 2025, quase dois anos após o anúncio do Grace Hopper e mais de um ano após os primeiros GB200 em produção, a Grace continua sendo o ponto mais criticado da plataforma NVIDIA para IA em alta escala. Por exemplo:
>
>1. **O pesadelo dos "killer microseconds"**: os kernels no Blackwell são tão rápidos que muitos rodam em poucas dezenas ou centenas de microssegundos. Quando a **GPU** termina o trabalho e sinaliza a CPU, o overhead de interrupção acrescido do tempo de scheduling da **CPU** vira uma fração significativa do tempo total. Em workloads reais de LLM, especialmente com contextos longos ou batch pequeno, isso pode custar entre $30\%$ e $50\%$ do throughput total. A NVIDIA melhorou um pouco com CPU-GPU sync mais fino no NVLink 5, mas o problema fundamental permanece: o Grace não foi projetado para latências de microssegundo.
>
>2. **Kernel launch ainda lento**: o tempo de lançamento de kernel **CUDA** pelo Grace continua na casa dos $15$ - $25 \mu s$, medido por vários benchmarks independentes em 2025. Isso é alguma coisa entre $3$ e $5\times$ mais lento que um [Xeon](https://www.intel.com.br/content/www/br/pt/products/details/processors/xeon.html) ou [EPYC](https://www.amd.com/en/products/processors/server/epyc.html) moderno. **CUDA** Graph e persistent kernels ajudam em workloads muito regulares, notadamente em training, mas em ambientes dinâmicos (vLLM, TGI, Triton com batch variável) o overhead volta a aparecer. Até hoje não existe uma solução completa para isso no Grace.
>
>3. **Cache L2 cortado (1 MB em vez de 2 MB)**: esse é o pecado original que a NVIDIA nunca consertou. O [Neoverse V2](https://www.arm.com/products/silicon-ip-cpu/neoverse/neoverse-v2) permite até 2 MB de L2 por core. [AWS Graviton4](https://aws.amazon.com/pt/ec2/graviton/), [Ampere Altra Max](https://amperecomputing.com/products/processors) e Microsoft Cobalt todos usam 2 MB. A NVIDIA cortou para 1 MB "por razões de área/die size". Resultado: taxa de I-cache miss absurdamente alta em código **CUDA** runtime. Vários clientes relatam que o Grace gasta entre $20\%$ e $30\%$ dos ciclos em stall de I-cache em workloads reais de IA.
>
>4. **Network-on-Chip (NoC) em malha com latência alta**: o Grace usa uma malha 2D mesh em vez de ring ou crossbar, ambos mais eficientes. Acesso L3 pode levar vários hops dependendo da localização do core. Em tráfego ScaleOut (NVLink Switch ou InfiniBand/Ethernet), os pacotes RDMA têm que atravessar toda a NoC (Network-on-Chip) da Grace antes de chegar na **GPU** via NVLink-C2C. Isso adiciona dezenas de nanossegundos de latência que, em microssegundos totais, viram gargalo visível.
>
>5. **Foco de marketing quase exclusivo em HPC**: até hoje, praticamente todo case de sucesso que a NVIDIA mostra com a Grace é HPC tradicional (simulações científicas, clima, CFD). Em IA pura,  especialmente inference, os clientes preferem x86 (Xeon + H200/B200 ou EPYC + MI300X) ou até mesmo soluções AMD Instinct + **CPU**. O Grace só brilha mesmo quando o workload é altamente paralelizado e com kernels longos, neste caso, em treinamento de modelos gigantes.
>
>Resumindo: em 2025 o Grace continua sendo um **CPU** "muito bom para HPC e apenas aceitável para IA". A NVIDIA claramente priorizou densidade e integração NVLink em detrimento de single-thread performance e latência baixa, exatamente o oposto do que os modelos LLMs precisam. Quem quer performance máxima em inferencia ainda prefere sistemas **CPU x86 + GPU NVIDIA** ou migra para soluções que minimizam o papel da **CPU**.

Com o Blackwell ficando absurdamente rápido, o tempo de execução de muitos kernels caiu para o nível de microssegundos, ressuscitando o clássico problema dos “**killer microseconds**”.

Regra geral:

- Latências de nanossegundos implica em espera síncrona simples resolve.
- Latências de milissegundos implica em o custo do context switch é irrelevante.
- Latências de microssegundos implica em vira um pesadelo para qualquer CPU.

Mesmo com todas as otimizações assíncronas que a NVIDIA introduziu, a Grace ainda engasga bastante. Dois exemplos concretos:

1. O kernel launch continua lento demais. **CUDA** Graphs e persistent kernels ajudam, mas nem todo workload real permite usá-los.
2. A microarquitetura a Grace tem recortes importantes: apesar de usar o Neoverse V2 (o core ARM mais forte da época), a NVIDIA cortou o cache L2 de 2 MB para apenas 1 MB. O Graviton 4 da AWS, que usa o mesmo core V2, manteve os 2 MB. Grande parte dos casos de altíssimo I-cache miss rate que clientes estão vendo em GB200 vem exatamente daí.

Não por acaso, quando a NVIDIA promove o Grace, o foco é quase exclusivamente em aplicações HPC tradicionais — exatamente as que menos sofrem com esses gargalos de microssegundos.

![](/assets/images/tradu15.webp)
**Figura 15**: Foco de marketing da NVIDIA na **CPU** Grace, destacando a ênfase em aplicações HPC tradicionais. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Outro ponto enfatizado é o equilíbrio de maior largura de banda e capacidade de memória, escolhendo LPDDR5x, e estendendo ainda mais as capacidades de acesso à memória do Hopper e Blackwell via NVLink C2C. Adicionalmente, toda a rede on-chip é uma arquitetura Mesh. A latência de acesso L3 requer múltiplos hops no NOC (Network on Chip), o que tem um impacto significativo.

![](/assets/images/tradu16.webp)
**Figura 16**: Arquitetura Mesh da Network-on-Chip (NoC) na **CPU** Grace, destacando o impacto da latência de acesso L3. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Por outro lado, como o CX7 pareado com o GB200 não tem um Switch PCIe embutido, o tráfego ScaleOut RDMA deve atravessar toda a NOC do Grace e depois alcançar o Blackwell via NVLink C2C.

![](/assets/images/tradu17.webp)
**Figura 17**: Tráfego ScaleOut RDMA atravessando a Network-on-Chip (NoC) da **CPU** Grace para alcançar a **GPU** Blackwell via NVLink C2C. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Isso gera vários problemas. Por exemplo: quando o tráfego ScaleOut RDMA precisa atravessar toda a NoC do Grace, o cache L2 menor faz a penalidade de cache miss ficar ainda pior. O site Chips and Cheese fez testes mostrando que a latência do Grace é bem maior que a de x86 e também consideravelmente maior que a do Graviton 4, justamente por causa do L2 minúsculo e do barulho dos “noisy neighbors” + ruído da NoC.
Aliás, desabafo rápido: o BlueField-4 baseado em Grace tem exatamente o mesmo problema… a competência da NVIDIA nessa área ainda deixa muito a desejar. Além disso, CX8/CX9 têm defeitos graves de design… Na verdade, um relatório recente do Megatron treinando DeepSeek-V3 no GB200 (“Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72”) também cita esse overhead da CPU como gargalo.
A AWS resolveu o problema colocando um switch PCIe extra para o tráfego ScaleOut RDMA não precisar atravessar a NoC do Grace.
A Meta, por sua vez, está usando proporção 1:1 (um Grace para cada Blackwell) para aliviar isso.
Claro, parte desses problemas some no GB300.
Comparando com o Intel Granite Rapids (GNR), ele oferece a opção SNC3 para lidar melhor com cache. (É claro que o GNR também tem seus dramas com NoC afetando velocidade de memória, mas não vou entrar nisso agora…)
O fato é: quando o número de cores passa de certo ponto, a complexidade e o impacto da NoC viram um problema sério, especialmente em CPUs general-purpose com coerência de cache. Mesmo em processadores sem coerência, já vi isso acontecer. Lá na Cisco, em 2004 fizemos o QFP de 40 cores/160 threads (ok), 2008 56 cores/224 threads (ainda ok), mas na terceira geração com 224 cores/896 threads (QFP3) o negócio desabou feio. Quando um socket general-purpose chega a centenas de cores, o problema aparece inevitavelmente.
2.4 Memória no Blackwell (dual-die)
Outro drama é a arquitetura dual-die. Acesso à memória cruzando dies traz latência maior inevitavelmente. Isso fode a eficiência dos SMs acessando GMEM. O **CUDA** 13.0 ainda não suporta agendamento de CTA com afinidade de memória, mas dá pra amenizar um pouco usando **CuTE** Layout para escalonar os bancos de memória. Claro que no futuro a NVIDIA vai lançar API de CTA affinity… eu boto fé. Já escrevi sobre isso aqui:
“Nvidia GB200 Architecture Analysis 4: Análise do Blackwell multi-die e coerência de cache”
3. Previsão da arquitetura Vera Rubin (agora só chamada Rubin)
3.1 CPU Vera
Pelo devboard que o Jensen mostrou no GTC, a memória da Vera tem 8 canais, então largura de banda deve dobrar em relação ao Grace. Contagem de cores subiu para 88, PCIe Gen6 com uns 80 lanes. O core deve ser Neoverse V3. O V3 suporta até 3 MB de L2, mas sinceramente… duvido que a NVIDIA vá usar mais de 1 MB (no Jetson Thor com V3-AE continua 1 MB só…). E segue multi-die, com controladores de PCIe/Memória em dies separados, igual Graviton 3/4.
O CPU Overhead vai continuar existindo. Raiz do problema: ler do L1 leva 3 ciclos, atravessar L3 + Mesh NoC passa de 120 ciclos fácil. Não é o fim do mundo, mas só vai sumir de verdade quando aparecer um Intel x86 com NVLink-C2C…
3.2 Especulação sobre a arquitetura Rubin
Algumas coisas já são segredo aberto: escala do Tensor Core vai dobrar novamente na dimensão M, e a capacidade do TMEM vai aumentar pra aguentar isso. Mas a área do chip já está no limite. Por isso a NVIDIA vai usar I/O Die separado no Rubin.
Minha aposta: localmente vão usar MMA de 4-SM pra reutilizar dados ainda mais. O formato do cluster CGA vai pra 4, continuando a estratégia do Blackwell de saturar o GPC com clusters. Sobre microarquitetura do SM: adicionar 1 scalar core dentro do SM traria um monte de benefícios e ocupa quase nada de área.
Hoje os descritores de TMA/MMA são gerados quase todos pela CPU host. Como eles mudam pouco, daria pra fazer prefetch, mas a latência do NVLink-C2C ainda é alta (centenas de ciclos). Colocar um scalar core pequeno com Private SMEM de 2~4 KB pra guardar MBarrier seria lindo: você desacopla totalmente o programa assíncrono. Ficaria só: função TC, função TMA e o kernel SIMT tradicional fazendo Epilogue. O scalar core cuidando de Sch warp e gerando descritores. Isso liberaria RMEM, acabaria com Warp Specialization complicada e ainda economizaria ICache.
Com isso daria pra fazer MPMD mais louco, especialmente no Rubin Ultra com 4 dies colados. Dá pra brincar com Green CTX + CTA affinity e usar estruturas on-chip de forma insana. Frameworks tipo Google Pathways já existem há anos pra isso.
Aí alguém vai falar: “Pera, não tá parecendo muito o Ascend 910 da Huawei com Scalar AI CPU + Vector + Tensor Core?”
E eu volto na piada inicial: “Estar um passo à frente te faz pioneiro; estar meio passo à frente te faz deus.”
O sucesso da NVIDIA vem muito de ter arrastado o ecossistema devagarinho desde o Tensor Core do Volta, deu 10 anos pro mundo acompanhar. Técnico adora querer fazer “end-in-mind”. Eu mesmo já queimei a língua várias vezes com isso (em 2018 fazendo edge AI na Cisco… produto só saiu agora em 2025 rs).
Quando você vai muito na frente, vira herege e queima na fogueira tipo Bruno. Exemplo: RDMA Lossy vs Lossless… quando o mundo inteiro tava no Lossless, a gente já tinha eRDMA pronto 3 anos antes do Google Falcon.
(Figura 11 aqui – o caminho correto, mas cheio de detalhe infernal que ninguém aguenta aprender)
Voltando ao ponto: tem um monte de detalhe sutil em microarquitetura de GPU que faz toda diferença: SIMD vs SIMT, framework de task scheduling, design de Memory Barrier, scalar core, interconexão NoC… Mas o mais importante é o equilíbrio entre facilidade de uso e performance. O arquiteto tem que garantir que o cliente não caia num abismo de performance só porque programou errado. Tem uma quantidade absurda de trabalho sujo nisso tudo.
No fim, o grande fosso da NVIDIA é prever o workload dos próximos 3~5 anos e ter capacidade full-stack. Nós mortais só temos a inteligência humana mesmo (risos). Depois da Apsara Conference eu tava zoando minha própria cara com isso.
E a previsão que fiz há seis meses… algumas coisas já aconteceram antes da hora. Daqui dois anos a gente confere.
(“Citizen Scientist: Previsão da Arquitetura de Grandes Modelos nos Próximos Cinco Anos?”)
O custo da inteligência “artificial” (a humana) é alto pra caralho. Dei volta completa de 20 anos: algoritmo → chip → rede → computação. Hoje estamos liderando em interconexão (protocolo e chip). Anos atrás a NVIDIA veio mostrar roadmap do BlueField-4… a gente já tinha implementado aquilo. Próxima geração vai esmagar CX10, fiquem tranquilos.
Em ScaleUP eu duvido que tenha mais de 5 pessoas no mundo que saibam mais que eu (sem falsa modéstia). Os trade-offs eu já mapeei tudo.
Faltando só framework-layer e treinar uns modelos menores mês que vem pra fechar o buraco. Em algoritmo eu nado de braçada desde OI, quantização, reinforcement learning distribuído na Cisco… matemática então nem se fala, fiz faculdade de matemática e engoli dezenas de disciplinas.
Eu vi o caminho em 2014. Ainda tô correndo atrás da álgebra que falta (risos).
4. Alguns conselhos (pros brothers chineses das placas nacionais)
Palavra dura, mas de coração: muitos chips domésticos ainda têm deficiência grave. Já derrubei vários em teste de aquisição centralizada porque achava cenário que cliente precisava e o chip entrava em avalanche de performance. Mas eu sou gente boa: ganhava o teste e ainda mandava o relatório de defeito pros caras da Huawei (risos). Hoje os produtos Datacom da Huawei estão bem maduros.
O diabo mora nos detalhes. Como a NVIDIA faz interconexão dentro do SM no CGA? Por que o MBarrier é daquele jeito? Como o Async Proxy simplifica software? Por que o **CuTE** Layout foi abstraído assim? Até detalhezinho do tcgen05 que embrulha commit dentro de pipeline.consumer.release e faz alloc/dealloc col-based no TMEM… tudo tem motivo, tudo tem trade-off.
Facilidade de uso não é só ecossistema. É saber por que aquilo funciona. Não tem atalho. Querer “ultrapassar na curva” geralmente termina capotando na curva.
A boa notícia: já estamos liderando em várias áreas locais. Vamos com calma, pé no chão, detalhe por detalhe.
“Porra, estrangeiro faz, chinês não faz? Chinês tem uma cabeça a menos que eles?”
Vamos fazer acontecer. Passo a passo. Sem pular etapa.