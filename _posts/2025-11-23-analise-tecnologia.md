---
layout: post
title: Análise da Tecnologia das GPUs Nvida
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
lastmod: 2025-11-23T23:04:39.930Z
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

O testo de Zarbot é bastante técnico e detalhado, e eu o achei muito interessante. Na verdade, é indispensável para quem trabalha com alto desempenho. Eu levei um domingo inteiro lendo e acho que ainda não entendi tudo. Se você achar que o esforço vale a pena. Comece por [aqui](https://frankalcantara.com/limpe-seu-cerebro/). Este é um texto excelente para quem quiser testar sua capacidade de concentração e pensamento profundo.

>Eu vou colocar todos os meus comentários e complementos em blocos de destaque como este. Além, é claro de incluir todos os links que eu achar necessário para facilitar sua busca. Para que fique claro, deste ponto em diante, eu se refere ao autor original do texto, Zarbot.

**TL;DR**
 
Eu passei algumas semanas para organizar algumas operações GEMM no [Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) e no [Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) usando o CuteDSL. Eu observei a evolução do Ampere para o Hopper e depois para o Blackwell. Por coincidência, no fim de semana anterior ao último, eu participei do Summit de Computação Turing da Huawei e conversei com o Dr. Liao e outras pessoas da equipe Ascend. Depois disso, [Jensen Huang](https://grokipedia.com/page/Jensen_Huang) apresentou o quadro de desenvolvimento de engenharia [Vera Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) e [BlueField-4](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) ao vivo na Keynote do GTC. Portanto, eu preparei uma análise abrangente e uma previsão da microarquitetura da nova geração (É só um palpite, um esforço de adivinhação não me culpe se eu estiver errado...).

>Já discuti essa arquitetura, focando na distribuição de tensão contínua aqui.

![](/images/tradu1.webp)
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

Na visão humilde deste observador, o maior fosso da Nvidia não é algo que possa ser explicado em uma ou duas frases. Há muita controvérsia em torno do ecossistema CUDA ou SIMT. Mas seu verdadeiro fosso reside precisamente em ter lidado com muito do "trabalho sujo" de forma limpa dentro da arquitetura inteira, combinado com capacidades full-stack de algoritmos a sistemas a chips. Isso também é um ponto de inspiração para muitos chips domésticos [chineses], especialmente nos detalhes sutis onde eles equilibram facilidade de uso/programabilidade vs. performance. Outro fator é o excelente timing deles em trazer arquiteturas para o mercado e sua execução de marketing. Como diz o ditado, "Estar um passo à frente te faz pioneiro; estar meio passo à frente te faz um deus."

Claro, toda arquitetura tem seus trade-offs e deficiências. A Nvidia não é um deus também. Em seguida, eu discutirei muitos problemas da Nvidia, como o Blackwell, o Grace e o recém-lançado BlueField-4. Depois, assumindo que eu fosse um arquiteto para o Vera Rubin, eu discutirei direções evolutivas potenciais.

1. Discutindo a Evolução do Volta ao Blackwell

Começando com a introdução dos Tensor Cores no Volta, a arquitetura SIMT tradicionalmente definida da Nvidia na verdade começou a ser disruptada. A migração completa da arquitetura pode só ser finalizada com a geração Rubin. Todo o processo se estendeu por dez anos, representando tanto iterações graduais de hardware quanto inovações progressivas de software.

1.1 TensorCore De uma perspectiva de hardware, olhando das instruções FMA mais antigas para o vetorizado DP4A, depois para a primeira geração TensorCore no Volta (SM70), e subsequentemente Ampere/Hopper/Blackwell, todos eles têm aumentado a escala da multiplicação de matrizes, melhorado a razão compute-to-memory access e suportado formatos de dados de precisão mais baixa.

[](/Jukanlosreve/article/1992531045485531164/media/1992523463500980224)

Olhando as mudanças na precisão numérica, como mostrado abaixo, acompanhadas pelas restrições de área do chip, a geração Blackwell Ultra (B300) já começou a cortar a potência de computação para cálculos de alta precisão.

[](/Jukanlosreve/article/1992531045485531164/media/1992523881085812737)

Espera-se que a geração Rubin dobre ainda mais a escala dos TensorCores, estimada em 256 x N x 256 bits. Por outro lado, eu acho que veremos uma expansão adicional da MMA de 2-CTA do Blackwell para uma instrução MMA conjunta de 4-CTA no Rubin. No entanto, haverá demandas adicionais para agendamento dentro da CGA (Cooperative Group Array).

Outro problema trazido pelo aumento na potência de computação é a mudança no caminho de suprimento de dados. Os TensorCores iniciais (Volta) começaram a reutilizar registradores do CUDA Core. Então, à medida que a escala dos TensorCores do Ampere se expandiu, considerando a pressão de registradores, o cp.async foi usado para contornar o L1 e reduzir o uso de RMEM. O Hopper então introduziu o TMA (Tensor Memory Accelerator), permitindo que os operandos fossem colocados diretamente no SMEM, e introduziu CGA e DSMEM (Distributed Shared Memory). No entanto, nesta fase, os resultados do Acumulador ainda estavam no RMEM para facilitar operações subsequentes de Epilogue, mas isso ainda requeria um mecanismo de barreira waitgroup. Finalmente, o Blackwell introduz o TMEM, essencialmente separando o TensorCore e o CUDA Core, enquanto também reutiliza o mecanismo MBarrier introduzido pelas operações assíncronas do TMA. Como mostrado na figura abaixo:

[](/Jukanlosreve/article/1992531045485531164/media/1992524238469906432)

Todo o processo levou cerca de 10 anos, do Volta começando com o que parecia um componente add-on do TensorCore, ao Blackwell introduzindo o TMEM que não depende de RMEM, alcançando basicamente uma separação totalmente assíncrona. Cada passo tem sido bem estável. (Referências: , )

1.2 Processamento Assíncrono O outro aspecto é o processamento assíncrono. Quando a geração Volta introduziu um PC (Program Counter) independente para cada Thread, na verdade marcou o início da execução assíncrona.

[](/Jukanlosreve/article/1992531045485531164/media/1992524819297124352)

A partir desse ponto, as threads podiam esperar por mensagens para realizar processamento assíncrono, abrindo uma janela para programação assíncrona em relação às arquiteturas alinhadas por PC tradicionais.

[](/Jukanlosreve/article/1992531045485531164/media/1992524941108064256)

A parte boa é que a Nvidia forneceu a abstração Cooperative Group no software. No entanto, os TensorCores ainda requeriam execução síncrona em todo o Warp. Então, começando com a introdução do cp.async no Ampere, o caminho de suprimento de dados de todo o programa efetivamente se tornou assíncrono, que é o conceito de "Async Thread" mencionado pela Nvidia.

[](/Jukanlosreve/article/1992531045485531164/media/1992525030966927360)

O Hopper foi um passo adiante ao introduzir o MBarrier. Pipelines assíncronos de software e Warp Specialization construídos em torno do MBarrier se tornaram populares. Ele introduziu o Async Proxy, distinguindo diferentes caminhos de acesso à memória através de General Proxy e Async Proxy. Para operações Async Proxy, geralmente há uma barreira de memória; o LD/ST (Load/Store) do General Proxy pode esperar por essa barreira para completar, permitindo que operações assíncronas do TMA sejam combinadas com o acesso à memória LD/ST SIMT original, garantindo requisitos de ordenação de memória.

[](/Jukanlosreve/article/1992531045485531164/media/1992525119626051584)

Claro, o Hopper tinha imperfeições. O WGMMA era uma solução temporária, ocupando uma grande quantidade de RMEM enquanto também requeria espera síncrona. Portanto, quando o Hopper foi lançado, foi explicitamente dito que o WGMMA do SM_90a não seria compatível com versões anteriores. Isso teve uma grande desvantagem.

[](/Jukanlosreve/article/1992531045485531164/media/1992525174865096704)

No Blackwell, o TensorCore também se tornou uma operação totalmente assíncrona, reutilizando a construção MBarrier. Assim, emitir TMA e instruções pode ser feito no nível de Thread. No entanto, a alocação e cópia de memória para o TMEM ainda requerem manuseio no nível Warp. Por outro lado, o mecanismo ClusterLaunchControl foi introduzido, fornecendo alguma capacidade de agendamento dinâmico.

[](/Jukanlosreve/article/1992531045485531164/media/1992525267232038912)

Podemos então construir padrões de processamento Warp Specialization mais complexos.

[](/Jukanlosreve/article/1992531045485531164/media/1992525374375575553)

1.3 Layout CuTe Isso também é uma abstração de software fantástica, especialmente em como esconde a complexidade do Swizzle no Hopper e no Blackwell. Além disso, de uma perspectiva algébrica, resolve cálculos complexos de fronteiras de Tile/Partition, tornando o código mais intuitivo—embora para aqueles que não são formados em álgebra, aprender o CuTe ainda apresenta uma curva de aprendizado íngreme. Eu comecei a discutir a álgebra de Layout CuTe no artigo abaixo: (Referência: )

No entanto, você precisa notar que na arquitetura dual die do Blackwell, ou mesmo na arquitetura 4-die do Rubin Ultra, e potencialmente em arquiteturas 3D-DRAM futuras, essa álgebra simplifica muitos problemas demais. Eu elaborarei o porquê nos capítulos posteriores. Claro, eu atualizarei este conteúdo ainda mais quando tiver tempo nos próximos meses.

2. Discutindo as Deficiências do Blackwell

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

É fácil cometer erros se a sincronização não for bem tratada. No entanto, a Nvidia introduziu muitas abstrações de Pipeline aqui, evitando bastantes erros. Combinado com o mecanismo de alocação de gerenciamento de memória do TMEM, usar alloc/dealloc reduz a complexidade de gerenciar o TMEM sob paralelismo multi-thread. De um ponto de vista de complexidade de gerenciamento, como mostrado na figura abaixo, Sch warp/TMA Warp e TC Warp podem todos alcançar processamento single-thread. Apenas o Epilogue Warp requer as coisas originais do SIMT. Uma vez entendido, não parece tão complexo, mas sempre tem que se manter em mente ao programar... Felizmente, tendo passado um longo tempo trabalhando em várias tarefas de programação assíncrona, lidar com isso não é tão difícil.

[](/Jukanlosreve/article/1992531045485531164/media/1992527018207166464)

2.3 Questões da CPU Embora o NVLink C2C tenha sido introduzido na geração Hopper, permitindo que o Grace se conecte diretamente com Hopper ou Blackwell via NVLink, o CPU do Grace tem bastantes questões. Na realidade, à medida que a potência de computação do Blackwell fica mais forte, o tempo de execução de muitos Kernels cai para o nível de microssegundo, criando um problema clássico de "Killer Microsecond". Para problemas de ns-level, espera síncrona basta. Para timeframes de ms-level, o custo de context switching é negligenciável. Mas quando se trata do nível us (microssegundo), isso representa um desafio significativo para processadores. Embora muitas otimizações de programação assíncrona tenham sido introduzidas, CPUs como o Grace ainda enfrentam muitos gargalos. Um é que a velocidade de Kernel Launch não é rápida o suficiente—embora alguém pudesse discutir que isso pode ser resolvido via cuda-graph ou persistent kernels, nem todos os workloads atendem a essas condições. Outro aspecto são defeitos na microarquitetura do Grace. Embora o Grace tenha usado o Neoverse V2 Core mais forte da ARM na época, seu design não adotou o Cache L2 de 2MB usado pelo V2, mas o cortou para 1MB. Em contraste, o AWS Graviton 4, que também usa V2 Cores, adotou um Cache L2 de 2MB. Atualmente, alguns clientes encontrando problemas de L1 ICache Miss no Grace em GB200 estão em grande parte relacionados a isso. Nós vemos que a promoção do Grace pela Nvidia basicamente discute aplicações relacionadas a HPC...

[](/Jukanlosreve/article/1992531045485531164/media/1992527206690779138)

Outro ponto enfatizado é o equilíbrio de maior largura de banda e capacidade de memória, escolhendo LPDDR5x, e estendendo ainda mais as capacidades de acesso à memória do Hopper e Blackwell via NVLink C2C. Adicionalmente, toda a rede on-chip é uma arquitetura Mesh. A latência de acesso L3 requer múltiplos hops no NOC (Network on Chip), o que tem um impacto significativo.

[](/Jukanlosreve/article/1992531045485531164/media/1992527328044564480)

Por outro lado, como o CX7 pareado com o GB200 não tem um Switch PCIe embutido, o tráfego ScaleOut RDMA deve atravessar toda a NOC do Grace e depois alcançar o Blackwell via NVLink C2C.

[](/Jukanlosreve/article/1992531045485531164/media/1992527390850129921)

Isso leva a bastantes problemas. Por exemplo, quando esse tráfego atravessa toda a NOC do Grace, com um Cache L2 menor, a penalidade de cache miss aumenta ainda mais. Chips and Cheese tem um teste mostrando que a latência do Grace é muito maior do que