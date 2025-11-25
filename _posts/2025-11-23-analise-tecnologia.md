---
layout: post
title: Análise da Tecnologia das **GPUs** Nvida
author: Frank
categories:
    - artigo
    - Matemática
    - disciplina
tags:
    - algoritmos
    - C++
    - eng. software
    - estrutura de dados
    - orientação a objetos
    - programação
image: assets/images/tecnvida.webp
rating: 6
description: Um estudo da classe Maps destacando as melhorias implementadas na últimas versões do C++
date: 2025-11-23T14:39:47.039Z
preview: map é uma classe importante para otimização de algoritmos. Este artigo estuda o uso de Maps destacando seus métodos mais modernos.
lastmod: 2025-11-24T21:51:13.843Z
keywords:
    - algoritmos
    - CLang
    - Estrutura de dados
    - Maps
    - CPP Moderno
    - Algoritmos
    - GCC
published: false
draft: false
schema:
    type: Article
    headline: Maps Cache-Friendly em C++23 – std::flat_map, Localidade de Cache e Indexação Múltipla
    description: Análise profunda do std::flat_map introduzido no C++23, comparação prática com std::map, explicação detalhada sobre localidade de cache, invalidação de iteradores, algoritmos shift_left/shift_right, merge otimizado, implementação de map com duas chaves e modificação segura de chaves via node extraction.
    author:
        type: Person
        name: Frank Alcantara
    datePublished: 2025-11-23
    dateModified: 2025-11-23
    publisher:
        type: Organization
        name: frankalcantara.com
        logo:
            type: ImageObject
            url: https://frankalcantara.com/assets/images/logo.png
    image: https://frankalcantara.com/assets/images/tecnvida.webp
    keywords:
        - C++23
        - std::flat_map
        - std::map
        - localidade de cache
        - cache-friendly
        - estrutura de dados
        - containers associativos
        - performance C++
        - árvore rubro-negra
        - flat containers
        - shift_left
        - shift_right
        - node extraction
        - indexação múltipla
        - programação de sistemas
        - engenharia de software
    wordCount: 3780
    inLanguage: pt-BR
    license: https://creativecommons.org/licenses/by-sa/4.0/
    mainEntityOfPage:
        type: WebPage
        id: https://frankalcantara.com/2025/11/23/analise-da-tecnologia-das-gpus-nvida.html
slug: analise-da-tecnologia-das-gpus-nvida
---

Este texto é uma tradução, aumentada e comentada, para o português de uma tradução para o inglês feita por [Jukan](https://x.com/Jukanlosreve) de uma postagem no WeChat do blogueiro chinês [Zarbot](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Eu traduzi esta postagem, mas não tenho nenhuma associação com o autor, ou com o primeiro tradutor.

O texto de [Jukan](https://x.com/Jukanlosreve) é bastante técnico e detalhado. Indispensável para quem trabalha com alto desempenho. Eu levei um domingo inteiro lendo e acho que ainda não entendi tudo. Se você achar que o esforço vale a pena. Comece por [aqui](https://frankalcantara.com/limpe-seu-cerebro/). Este é um texto excelente para quem quiser testar sua capacidade de concentração e pensamento profundo. Mas, não é para qualquer um, precisa ter um mínimo de conhecimento de arquiteturas de CPUs/GPUs e programação paralela. Se não for o seu caso, não se preocupe, eu tenho muitos outros textos mais acessíveis.

>Eu vou colocar todos os meus comentários e complementos em blocos de destaque como este. Além, é claro de incluir todos os links que eu achar necessário para facilitar sua busca. Para que fique claro, deste ponto em diante, eu se refere ao autor original do texto, Zarbot.

**TL;DR**
 
Eu passei algumas semanas para organizar algumas operações GEMM no [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) e no [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) usando o CuteDSL. Eu observei a evolução do Ampere para o Hopper e depois para o Blackwell. Por coincidência, no fim de semana anterior ao último, eu participei do Summit de Computação Turing da Huawei e conversei com o Dr. Liao e outras pessoas da equipe Ascend. Depois disso, [Jensen Huang](https://grokipedia.com/page/Jensen_Huang) apresentou o quadro de desenvolvimento de engenharia [Vera Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) e [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) ao vivo na Keynote do GTC. Portanto, eu preparei uma análise abrangente e uma previsão da microarquitetura da nova geração (É só um palpite, um esforço de adivinhação não me culpe se eu estiver errado...).

>Já discuti essa arquitetura, focando na distribuição de tensão contínua [neste artigo](https://frankalcantara.com/nvidia-fabrica-de-ia/).

![](/assets/images/tradu1.webp)
**Figura 1**: A tabela apresenta o conjunto de instruções TCGen05 do Blackwell organizadas em dois grupos: instruções síncronas para gerenciamento de recursos e sincronização (alloc, dealloc, relinquish_alloc_permit, fence, wait e commit) e instruções assíncronas para operações computacionais e de movimentação de dados (mma para multiplicação de matrizes, cp para cópia, shift para deslocamento, ld para leitura e st para escrita). Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

>**GEMM**, **G**eneral **M**atrix **M**ultiply, representa a operação fundamental de multiplicação de matrizes expressa como:
>
>$$C = \alpha \cdot (A \times B) + \beta \cdot C$$
>
>na qual $A$ é uma matriz $m \times k$, $B$ é uma matriz $k \times n$, e $C$ é uma matriz $m \times n$. Os escalares $\alpha$ e $\beta$ são fatores de escala. Esta operação é a base computacional de praticamente todos os algoritmos de deep learning modernos, representando tipicamente mais de $90\%$ do tempo de execução em redes neurais.
>
>A complexidade computacional de GEMM é $O(m \cdot n \cdot k)$, mas a complexidade de acesso à memória é $O(m \cdot k + k \cdot n + m \cdot n)$. Esta discrepância cria o desafio fundamental: quanto maior a matriz, maior a razão `compute-to-memory-access`, permitindo melhor utilização do hardware.

>**CuteDSL**, Cute Domain-Specific Language,:é uma abstração algébrica desenvolvida pela Nvidia para expressar layouts de tensores e operações sobre eles de forma composicional. A abstração fundamental é o
conceito de Layout, definido algebricamente como uma função:
>
>$$\text{Layout} : \mathbb{Z}^n \rightarrow \mathbb{Z}$$
>
>que mapeia coordenadas multidimensionais lógicas para offsets lineares em memória. Um 
Layout é especificado por um par (Shape, Stride), onde:
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
>A álgebra do CuteDSL permite composição de layouts através de operações como:
>
>1. **Particionamento**: dividir um tensor em tiles menores
>2. **Composição**: combinar múltiplos layouts
>3. **Swizzling**: permutações complexas para otimizar acesso à memória
>
>O aspecto fundamental do CuteDSL é que ele abstrai completamente a complexidade do swizzling de memória necessário para evitar bank conflicts na shared memory. No Hopper e Blackwell, padrões de swizzle de 128 bits são necessários para maximizar throughput de acesso à memória, e o CuteDSL calcula automaticamente os índices corretos.

Na visão humilde deste observador, o maior fosso da Nvidia não é algo que possa ser explicado em uma ou duas frases. Há muita controvérsia em torno do ecossistema CUDA ou **SIMT**. Mas seu verdadeiro fosso reside precisamente em ter lidado com muito do "trabalho sujo" de forma limpa dentro da arquitetura inteira, combinado com capacidades full-stack de algoritmos a sistemas a chips. Isso também é um ponto de inspiração para muitos chips domésticos [chineses], especialmente nos detalhes sutis onde eles equilibram facilidade de uso/programabilidade vs. performance. Outro fator é o excelente timing deles em trazer arquiteturas para o mercado e sua execução de marketing. Como diz o ditado, "_Estar um passo à frente te faz pioneiro; estar meio passo à frente te faz um deus._"

Claro, toda arquitetura tem seus trade-offs e deficiências. A Nvidia não é um deus também. Em seguida, eu discutirei muitos problemas da Nvidia, como o Blackwell, o [Grace](https://www.nvidia.com/pt-br/data-center/grace-cpu/) e o recém-lançado [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/). Depois, assumindo que eu fosse um arquiteto para o Vera Rubin, eu discutirei direções evolutivas potenciais.

## 1. Discutindo a Evolução do Volta ao Blackwell

Começando com a introdução dos **Tensor Cores** no [Volta](https://www.nvidia.com/pt-br/data-center/v100/), a arquitetura **SIMT** tradicionalmente definida da Nvidia na verdade começou a sofrer um processo de disrupção. A migração completa da arquitetura só pode ser finalizada com a geração Rubin. Todo o processo se estendeu por dez anos, representando tanto iterações graduais de hardware quanto inovações progressivas de software.

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
>* **Desempenho (em cargas compatíveis)**: **SIMD** > **SIMT** > **SMT**
>
>Embora se diga que o "**SIMT** é um **SIMD** mais flexível, é importante notar a distinção técnica: internamente, a execução dos grupos de threads, chamados de *warps*, ainda ocorre de forma sincronizada, *lockstep*, similar ao **SIMD**. Pesquisadores frequentemente definem o **SIMT** mais precisamente como "**SIMD** com abstração de thread e máscara de execução.
>
>Além disso, enquanto a hierarquia que supomos acima seja útil, alguns arquitetos de sistemas consideram **SMT** e **SIMT** como filosofias opostas: o **SMT** busca maximizar as instruções por ciclo, **IPC** de poucas threads complexas, enquanto o **SIMT** sacrifica o **IPC** individual em favor da vazão total, throughput, de milhares de threads simples.
>
>**SIMT vs. SIMD**: ambos usam o *broadcast* de instruções para múltiplas unidades de execução, economizando hardware de controle, em ciclho fetch/decode*. No entanto, o modelo da NVIDIA introduz três diferenciais que o **SIMD** clássico não possui:
>
**1. Instrução Única, Múltiplos Conjuntos de Registradores**: o **SIMD** tradicional exige que o programador gerencie vetores curtos. O **SIMT** permite uma escrita escalar: o código é escrito para uma única thread, usando lógica padrão. Na prática, a **GPU** executa milhares de threads. Um Streaming Multiprocessor, **SM**, gerencia múltiplos núcleos, e todas os threads de um warp executam a mesma instrução simultaneamente, mas **cada thread possui seus próprios registradores** privados.
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
>>* **Memória Compartilhada**: Acessos concorrentes a bancos de memória diferentes na mesma memória compartilhada são rápidos. Se múltiplas threads acessarem o mesmo banco, ocorre conflito e serialização.
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

### 1.1 TensorCore

De uma perspectiva de hardware, olhando desde as instruções **FMA** mais antigas para o vetorizado DP4A, depois para a primeira geração TensorCore no Volta (SM70), e subsequentemente Ampere/Hopper/Blackwell, todos eles têm aumentado a escala da multiplicação de matrizes, melhorado a razão compute-to-memory access e suportado formatos de dados de precisão mais baixa.

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

$$D = A \times B + C$$

Onde $A$, $B$ e $C$ são números escalares, ou elementos correspondentes em um vetor **SIMD** comum. Neste acrônimo, a palavra chave é Fusionada. Em processadores antigos, para fazer $A \times B + C$, o hardware precisava de dois passos:
>
>1. Calcular $Intermediario = A \times B$, e arredondar o resultado.
>2. Calcular $Resultado = Intermediario + C$, e arredondar novamente.
>
>Na instrução **FMA**, a multiplicação e a adição acontecem em **um único passo** no hardware, com apenas um arredondamento final. Isso é importante por dois motivos principais:
>
>1. **Precisão**: como há apenas um arredondamento no final, o resultado é matematicamente mais preciso.
>2.  **Desempenho**: ela conta como duas operações de ponto flutuante (**FLOPs**), uma multiplicação e uma adição, mas é executada no tempo de apenas uma instrução. Isso dobra a capacidade de cálculo teórica do processador.
> Nós vimos `FMA <double>` no canto inferior esquerdo da **Figura 2**. Isso representa os núcleos CUDA tradicionais, não-Tensor Cores, operando com precisão dupla, FP64. Era assim que a maioria da supercomputação científica era feita antes da era da IA profunda.

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
>1. **HMMA (Volta/Turing):** **MMA Síncrono**. Um thread individual era responsável por segurar pedaços específicos da matriz em seus registradores para que o Tensor Core operasse.
>2. **MMA Assíncrono (Ampere):** O hardware começou a poder buscar os dados da memória enquanto calculava o **MMA** anterior.
>3. **WGMMA (Hopper/SM90):** **Warpgroup MMA**. A operação **MMA** tornou-se tão grande e complexa que não é mais um único thread ou um único Warp que a gerencia, mas um grupo de 128 threads, **Warpgroup** cooperando. Os dados para o **MMA**, no **Hopper** fluem diretamente da memória compartilhada para os Tensor Cores, sem a necessidade de microgerenciamento via registradores individuais das threads.

Olhando as mudanças na precisão numérica, como mostrado abaixo, acompanhadas pelas restrições de área do chip, a geração [Blackwell Ultra (B300)](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4) já começou a cortar a potência de computação para cálculos de alta precisão.

![](/assets/images/tradu3.webp)
**Figura 3**: Evolução dos formatos de dados suportados pelos Tensor Cores da Nvidia, desde o INT8 no Pascal até o FP4 com microscaling no Blackwell. Observe a tendência de adoção de formatos de menor precisão para maximizar throughput computacional em cargas de trabalho de deep learning. Esta imagem está na [postagem original](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21&poc_token=HE7LImmjpRK9PkW11_nsMI8ejGiPH0tZf6WKWLO6).

Espera-se que a geração [Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) dobre a escala dos Tensor Cores, estimada em $256 \times N \times 256 \text{ bits}$. Por outro lado, eu acho que veremos uma expansão adicional da **MMA** de 2-**CTA** (**C**ooperative **T**hread **A**rray) do Blackwell para uma instrução **MMA** conjunta de 4-CTA no Rubin. No entanto, haverão demandas adicionais para agendamento dentro do **CGA**, **C**ooperative **G**roup **A**rray).

Outro problema trazido pelo aumento na potência de computação é a mudança no caminho de suprimento de dados. Os TensorCores iniciais (Volta) começaram a reutilizar registradores do CUDA Core. Então, à medida que a escala dos TensorCores do Ampere se expandiu, considerando a pressão de registradores, o cp.async foi usado para contornar o L1 e reduzir o uso de RMEM. O Hopper então introduziu o TMA (Tensor Memory Accelerator), permitindo que os operandos fossem colocados diretamente no SMEM, e introduziu CGA e DSMEM (Distributed Shared Memory). No entanto, nesta fase, os resultados do Acumulador ainda estavam no RMEM para facilitar operações subsequentes de Epilogue, mas isso ainda requeria um mecanismo de barreira waitgroup. Finalmente, o Blackwell introduz o TMEM, essencialmente separando o TensorCore e o CUDA Core, enquanto também reutiliza o mecanismo MBarrier introduzido pelas operações assíncronas do TMA. Como mostrado na figura abaixo:

[](/Jukanlosreve/article/1992531045485531164/media/1992524238469906432)

Todo o processo levou cerca de 10 anos, do Volta começando com o que parecia um componente add-on do TensorCore, ao Blackwell introduzindo o TMEM que não depende de RMEM, alcançando basicamente uma separação totalmente assíncrona. Cada passo tem sido bem estável. (Referências: , )

1.2 Processamento Assíncrono O outro aspecto é o processamento assíncrono. Quando a geração Volta introduziu um PC (Program Counter) independente para cada Thread, na verdade marcou o início da execução assíncrona.

[](/Jukanlosreve/article/1992531045485531164/media/1992524819297124352)

A partir desse ponto, os threads podiam esperar por mensagens para realizar processamento assíncrono, abrindo uma janela para programação assíncrona em relação às arquiteturas alinhadas por PC tradicionais.

[](/Jukanlosreve/article/1992531045485531164/media/1992524941108064256)

A parte boa é que a Nvidia forneceu a abstração Cooperative Group no software. No entanto, os TensorCores ainda requeriam execução síncrona em todo o Warp. Então, começando com a introdução do cp.async no Ampere, o caminho de suprimento de dados de todo o programa efetivamente se tornou assíncrono, que é o conceito de "Async Thread" mencionado pela Nvidia.

[](/Jukanlosreve/article/1992531045485531164/media/1992525030966927360)

O Hopper foi um passo adiante ao introduzir o MBarrier. Pipelines assíncronos de software e Warp Specialization construídos em torno do MBarrier se tornaram populares. Ele introduziu o Async Proxy, distinguindo diferentes caminhos de acesso à memória através de General Proxy e Async Proxy. Para operações Async Proxy, geralmente há uma barreira de memória; o LD/ST (Load/Store) do General Proxy pode esperar por essa barreira para completar, permitindo que operações assíncronas do TMA sejam combinadas com o acesso à memória LD/ST **SIMT** original, garantindo requisitos de ordenação de memória.

[](/Jukanlosreve/article/1992531045485531164/media/1992525119626051584)

Claro, o Hopper tinha imperfeições. O WGMMA era uma solução temporária, ocupando uma grande quantidade de RMEM enquanto também requeria espera síncrona. Portanto, quando o Hopper foi lançado, foi explicitamente dito que o WGMMA do SM_90a não seria compatível com versões anteriores. Isso teve uma grande desvantagem.

[](/Jukanlosreve/article/1992531045485531164/media/1992525174865096704)

No Blackwell, o TensorCore também se tornou uma operação totalmente assíncrona, reutilizando a construção MBarrier. Assim, emitir TMA e instruções pode ser feito no nível de Thread. No entanto, a alocação e cópia de memória para o TMEM ainda requerem manuseio no nível Warp. Por outro lado, o mecanismo ClusterLaunchControl foi introduzido, fornecendo alguma capacidade de agendamento dinâmico.

[](/Jukanlosreve/article/1992531045485531164/media/1992525267232038912)

Podemos então construir padrões de processamento Warp Specialization mais complexos.

[](/Jukanlosreve/article/1992531045485531164/media/1992525374375575553)

1.3 Layout CuTe Isso também é uma abstração de software fantástica, especialmente em como esconde a complexidade do Swizzle no Hopper e no Blackwell. Além disso, de uma perspectiva algébrica, resolve cálculos complexos de fronteiras de Tile/Partition, tornando o código mais intuitivo—embora para aqueles que não são formados em álgebra, aprender o CuTe ainda apresenta uma curva de aprendizado íngreme. Eu comecei a discutir a álgebra de Layout CuTe no artigo abaixo: (Referência: )

No entanto, você precisa notar que na arquitetura dual die do Blackwell, ou mesmo na arquitetura 4-die do Rubin Ultra, e potencialmente em arquiteturas 3D-DRAM futuras, essa álgebra simplifica muitos problemas demais. Eu elaborarei o porquê nos capítulos posteriores. Claro, eu atualizarei este conteúdo ainda mais quando tiver tempo nos próximos meses.

1. Discutindo as Deficiências do Blackwell

Tendo dito algumas coisas boas, este capítulo discutirá algumas deficiências, principalmente para dissipar o misticismo.

2.1 O Problema SFU do B200 Enquanto escalava freneticamente a performance do TensorCore, uma grande quantidade de TMEM foi adicionada. Ao mesmo tempo, o DSMEM formado por algumas redes de interconexão GPC também ocupou muita área do die. O cancelamento do L2 Partitioning também consumiu área do die. Consequentemente, o número de SMs em um único die foi reduzido para 80. Infelizmente, a performance do SFU (Special Function Unit) pareado com o CUDA Core não foi aprimorada. Isso levou a operações GEMM parecendo muito mais fortes, mas gargalos aparecendo ao calcular Softmax na Attention.

[](/Jukanlosreve/article/1992531045485531164/media/1992525953730502657)

Claro, alguns podem dizer, "Sem problema, só use Linear Attention." De fato, mudanças recentes na Attention geraram alguma controvérsia. De um lado, há o GDN do Qwen-Next e o KDA do Kimi Linear. Do outro lado, o minmax M2 abandonou o Linear Attn. Outro caminho é o MoR do Google/DeepMind, e rumores sugerem que o Universal Transformer usado no GPT-5 parece ainda estar aprimorando a potência de computação dos blocos Attn. (Referência: " ) Enquanto isso, o DSA do DeepSeek-V3.2 e o NSA anterior seguiram o caminho do Sparse Attn.

Minha visão pessoal se alinha com a do DeepSeek: Linear Attn não resolve bem o gargalo de acesso à memória. A computação em si é fácil de escalar, mas o acesso à memória é difícil. Portanto, escolher Sparse Attn é o caminho correto. Outro aspecto é um artigo que eu li há algum tempo discutindo SDPA da perspectiva de Optimal Transport. Ou seja, o processo de cálculo forward do mecanismo de atenção—o processo de gerar pesos de atenção via a função Softmax—é completamente equivalente à solução exata de um problema de One-Sided Entropic Optimal Transport (EOT). Portanto, o Softmax é inevitável. (Referência: )

Baseado nessa perspectiva, minha visão é que as capacidades SFU devem combinar com a potência de computação do TensorCore. Felizmente, o B300 resolveu esse problema, ao custo de cortar a potência de computação para outras operações de alta precisão. Em relação a essa questão, eu sempre senti que B200 e GB200 não são plataformas que valem a pena investir pesadamente.

2.2 Estrutura de Instruções Complexa do Blackwell

Na verdade, começando pelo Hopper, a programação assíncrona se tornou muito complexa, e a introdução do TMEM pelo Blackwell adicionou ainda mais complexidade. Por exemplo, todo o conjunto de instruções TensorCore tcgen05 tem tanto instruções síncronas quanto assíncronas.

[](/Jukanlosreve/article/1992531045485531164/media/1992526777386962944)

Por outro lado, a granularidade de emissão de instruções difere—algumas são de granularidade thread, algumas de granularidade warp, e o cenário 2-SM também precisa ser considerado.

[](/Jukanlosreve/article/1992531045485531164/media/1992526910954622976)

É fácil cometer erros se a sincronização não for bem tratada. No entanto, a Nvidia introduziu muitas abstrações de Pipeline aqui, evitando bastantes erros. Combinado com o mecanismo de alocação de gerenciamento de memória do TMEM, usar alloc/dealloc reduz a complexidade de gerenciar o TMEM sob paralelismo multi-thread. De um ponto de vista de complexidade de gerenciamento, como mostrado na figura abaixo, Sch warp/TMA Warp e TC Warp podem todos alcançar processamento single-thread. Apenas o Epilogue Warp requer as coisas originais do **SIMT**. Uma vez entendido, não parece tão complexo, mas sempre tem que se manter em mente ao programar... Felizmente, tendo passado um longo tempo trabalhando em várias tarefas de programação assíncrona, lidar com isso não é tão difícil.

[](/Jukanlosreve/article/1992531045485531164/media/1992527018207166464)

2.3 Questões da CPU Embora o NVLink C2C tenha sido introduzido na geração Hopper, permitindo que o Grace se conecte diretamente com Hopper ou Blackwell via NVLink, o CPU do Grace tem bastantes questões. Na realidade, à medida que a potência de computação do Blackwell fica mais forte, o tempo de execução de muitos Kernels cai para o nível de microssegundo, criando um problema clássico de "Killer Microsecond". Para problemas de ns-level, espera síncrona basta. Para timeframes de ms-level, o custo de context switching é negligenciável. Mas quando se trata do nível us (microssegundo), isso representa um desafio significativo para processadores. Embora muitas otimizações de programação assíncrona tenham sido introduzidas, **CPUs** como o Grace ainda enfrentam muitos gargalos. Um é que a velocidade de Kernel Launch não é rápida o suficiente—embora alguém pudesse discutir que isso pode ser resolvido via cuda-graph ou persistent kernels, nem todos os workloads atendem a essas condições. Outro aspecto são defeitos na microarquitetura do Grace. Embora o Grace tenha usado o Neoverse V2 Core mais forte da ARM na época, seu design não adotou o Cache L2 de 2MB usado pelo V2, mas o cortou para 1MB. Em contraste, o AWS Graviton 4, que também usa V2 Cores, adotou um Cache L2 de 2MB. Atualmente, alguns clientes encontrando problemas de L1 ICache Miss no Grace em GB200 estão em grande parte relacionados a isso. Nós vemos que a promoção do Grace pela Nvidia basicamente discute aplicações relacionadas a HPC...

[](/Jukanlosreve/article/1992531045485531164/media/1992527206690779138)

Outro ponto enfatizado é o equilíbrio de maior largura de banda e capacidade de memória, escolhendo LPDDR5x, e estendendo ainda mais as capacidades de acesso à memória do Hopper e Blackwell via NVLink C2C. Adicionalmente, toda a rede on-chip é uma arquitetura Mesh. A latência de acesso L3 requer múltiplos hops no NOC (Network on Chip), o que tem um impacto significativo.

[](/Jukanlosreve/article/1992531045485531164/media/1992527328044564480)

Por outro lado, como o CX7 pareado com o GB200 não tem um Switch PCIe embutido, o tráfego ScaleOut RDMA deve atravessar toda a NOC do Grace e depois alcançar o Blackwell via NVLink C2C.

[](/Jukanlosreve/article/1992531045485531164/media/1992527390850129921)

Isso leva a bastantes problemas. Por exemplo, quando esse tráfego atravessa toda a NOC do Grace, com um Cache L2 menor, a penalidade de cache miss aumenta ainda mais. Chips and Cheese tem um teste mostrando que a latência do Grace é muito maior do que