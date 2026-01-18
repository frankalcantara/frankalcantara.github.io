---
layout: post
title: Robótica para Iniciantes - Do Microcontrolador ao Controle de Motores
author: Frank
categories:
   - disciplina
   - Engenharia
   - Robótica
tags:
   - robótica
   - eletrônica
   - microcontroladores
   - STM32
   - motores elétricos
   - baterias
   - inteligência artificial
featured: false
rating: 5
description: Guia completo de robótica, cobrindo microcontroladores, gestão de energia e controle de motores para quem deseja aprender do zero com foco em projetos práticos.
date: 2026-01-17T18:48:01.923Z
preview: |
   A robótica é a maior, e talvez última oportunidade para o Brasil entrar no mundo da tecnologia de ponta. Nós perdemos a revolução do hardware, do software, da internet, da computação móvel, da inteligência artificial mas a robótica ainda está em seus primórdios. Este é o momento de agir.
lastmod: 2026-01-18T11:10:09.900Z
published: true
draft: 2026-01-17T18:48:04.606Z
slug: iniciando-robotica
image: assets/images/tecnvida.webp
keywords:
   - Aprendizado de robótica
   - baterias lipo
   - Controle de motores
   - guia de robótica
   - microcontroladores
   - Motores Elétricos
   - robótica para iniciantes
   - STM32
schema:
   type: Article
   name: Robótica para Iniciantes - Do Microcontrolador ao Controle de Motores
   description: Guia completo de robótica, cobrindo microcontroladores, gestão de energia e controle de motores para quem deseja aprender do zero com foco em projetos práticos.
   steps:
      - name: "Modulo "
        description: Introdução ao ambiente de desenvolvimento e a cpu
        fieldGroup: steps_group
      - name: Módulo 2
        description: Gestão de Energia e integradade de sinais
        fieldGroup: steps_group
      - name: "Módulo "
        description: transformar sinais elétricos em força mecânica precisa e controlada, dominando a física dos motores e a eletrônica de potência.
        fieldGroup: steps_group
      - name: Projeto Integrador
        description: Um projeto complexo com os conteúdos abordados nos módulos 1, 2 e 3
        fieldGroup: steps_group
      - name: Módulo 4
        description: o sistema de percepção ambiental e garantir que múltiplos módulos conversem através de barramentos industriais robustos.
        fieldGroup: steps_group
      - name: Módulo 5
        description: consolidar o hardware em placas profissionais e garantir que a eletrônica sobreviva ao mundo físico (vibração, calor e impacto).
        fieldGroup: steps_group
      - name: Projeto Integrador Final
        description: Crie um robô autônomo
        fieldGroup: steps_group
---

Eu escrevi dois posts no Linkedin sobre o que é necessário estudar para iniciar em robótica. Um com 10.000 visualizações e outro com pouco mais de 12.000 visualizações. Sem dúvida nenhuma, o tema interessa muita gente. Então decidi juntar tudo em um único post, expandir os tópicos e criar uma espécie de currículo para quem quer aprender robótica do zero.

O texto a seguir serve de guia para você aprender sozinho, mas também serve de base para um curso *in company*, uma pós-graduação e até mesmo para um curso de graduação. Só depende da sua iniciativa.

Optei por usar o STM32F407VET6 por causa do IP Arm Cortex-M4F com FPU *single-precision*, que é poderoso o suficiente para fazer controle de motores em FOC (*Field-Oriented Control*) em tempo real. Além disso, o chip tem uma variedade rica de periféricos que permitem explorar todos os conceitos necessários para robótica. Mas, há uma razão egoística: eu estou desenvolvendo um Hat para Raspberry Pi que usa este chip como co-processador de tempo real para processamento de audio totalmente aberto, [aqui](https://github.com/frankalcantara/RoboTZero/tree/main/AuralFocus). Então, eu preciso dominar o hardware e software deste microcontrolador, tudo que eu estudo tem aplicação direta no meu projeto.

Todo o material é pensado para ser desenvolvido em **bare-metal** (sem bibliotecas de alto nível), utilizando apenas registradores, timers e interrupções. Isso é essencial para entender como o hardware funciona. O objetivo é fugir do mundo maker e chegar ao nível profissional. Esta também é razão de eu não seguir a imensa maioria da internet no Brasil e ficar no trio arduino/raspberry pi/esp32. Esses kits são ótimos para prototipagem rápida, mas não ensinam o que é necessário para trabalhar com robótica do mundo real.

Espero que seja útil.

## Módulo 1: O Núcleo de Processamento (Cérebro)

**Objetivo**: compreender a unidade central de processamento e a execução de tarefas em tempo real.

* **Hardware**: estudo das arquiteturas **ARM Cortex-M4** ou **M7**. Estes IPs estão disponíveis nas famílias STM32 (NXP Kinetis ou Texas Instruments Tiva C). Seu foco deve ser na arquitetura, tipos de memória e periféricos internos. Todos os periféricos devem ser explorados fisicamente: GPIOs, ADCs, DACs, Timers, UART, I2C, SPI e PWM.

>Não use kits de desenvolvimento prontos. Principalmente não use o Raspiberry Pi ou outros kits que tenham sido desenvolvidos para o uso de sistemas operacionais. Compre uma placa simples (ex: STM32F407VET6, tiva) e monte seu próprio circuito de programação com um **ST-Link V2** ou **J-Link**. Aprenda a usar o [**OpenOCD**](https://openocd.org/) para programar e depurar seu microcontrolador via **SWD** (**S**erial **W**ire **D**ebug). Isso é essencial para entender como o hardware funciona.

* **Software**: programação em **C++**, manipulação de registros, configuração de Timers para contagem de tempo exata e tratamento de Interrupções. Seus primeiros códigos devem ser escritos sem o uso de bibliotecas de alto nível (HAL ou SDKs). Não use o Arduino IDE para este módulo. Nem nada parecido. Aprenda a manipular os registradores, memória e portas diretamente. Não use nenhum laço para fazer a CPU "esperar". Use interrupções e timers. Se a sua CPU parar para esperar, qualquer coisa, seu código estará errado. Esta limitação forçará a criação de código eficiente e o uso de máquinas de estados finitos.

___
> O uso indiscriminado de variáveis globais em sistemas embarcados cria "efeitos colaterais" imprevisíveis, dificulta a realização de testes unitários e aumenta o risco de condições de corrida (*race conditions*) em sistemas baseados em interrupções.
> 
> Para eliminar as globais, utilizaremos as seguintes técnicas:
>
>**A. Encapsulamento em Classes (Singleton ou Static Wrappers)**: em vez de uma variável solta para o status da bateria, criamos uma classe `PowerMonitor`. Os dados tornam-se membros privados, acessíveis apenas por métodos específicos.
>
>**B. Injeção de Dependência**: em vez de um módulo de motor olhar para uma variável global de sensores, **passamos uma referência do sensor para o motor no momento da inicialização**.
>
>**C. O Desafio das Interrupções (ISRs)**: as interrupções em microcontroladores ARM são funções C que não aceitam argumentos. Para evitar globais aqui, utilizamos o padrão **Bridge**:
>
>1. Criamos um ponteiro estático dentro da classe que aponta para a instância ativa (`instancePtr`).  
>
>2. A ISR chama esse ponteiro para executar o método da classe. Ponteiro seguro!
>
>Para o seu código esteja mais próximo dos códigos que encontramos em projetos reais, siga estas regras:
>
>1. **Local Static**: se uma variável só é usada dentro de uma função, mas precisa manter seu valor entre chamadas, declare-a como `static` dentro da função.  
>
>2. **Const-Correctness**: se um dado não deve ser alterado, use `const` em todos os lugares possíveis.  
>
>3. **Ponteiros Opacos**: esconda a implementação do hardware dentro de arquivos `.cpp`, expondo apenas o necessário no `.hpp`.
>
___

* **Aplicação de IA**: utilizar modelos de linguagem para explicar registradores específicos de *datasheets* extensos e gerar códigos base de inicialização de periféricos. Não confie nos pinos sugeridos automaticamente por ferramentas de IA. Existe um problema documentado de alucinação de pinos em alguns modelos. Seu objetivo é encontrar um ambiente (VSCode/PlatformIO/Assistente de IA) que se transforme na sua ferramenta fazendo seu aprendizado mais rápido.

### Projetos para o Módulo 1

Os sistemas abaixo foram pensados para serem desenvolvidos de forma puramente ***bare-metal*** (sem bibliotecas), utilizando apenas timers e interrupções e estão organizados do mais simples ao mais complexo. Cada sistema deve ser desenvolvido em um projeto separado, para que você possa focar em cada desafio individualmente.

#### I. Gerador de Sinais PWM Multiplexado

**Desafio**: criar 4 canais de PWM independentes em que a frequência e o ciclo de trabalho (*duty cycle*) sejam alterados via UART sem interromper o sinal.

**Periféricos**: Timers, GPIO (*Alternate Function*), UART (Interrupção de recepção).

**Conceito**: cálculo de *prescaler* e *autoreload* register para atingir frequências específicas. Utilize a IA para converter fórmulas de tempo em valores de *prescaler* e *autoreload* para os Timers. Mas, corrija os cálculos manualmente para entender o processo. A ideia é que a IA auxilie, mas que você compreenda o que está fazendo.

>Ou você entende o que está fazendo, ou você não entende nada. E o mercado de trabalho não perdoa ignorância.

#### II. Voltímetro Digital com Alarme Visual

**Desafio**: ler uma tensão analógica (0-3.3V) e, caso ultrapasse um limite, acionar um padrão de intermitência em um LED.

**Periféricos**: ADC (em modo de varredura ou disparo por timer), DAC (para espelhar a leitura), Timers e GPIOs.

**Conceito**: conversão de valores binários para tensão real e gestão de prioridade de interrupção (NVIC).

#### III. Escaneador de Barramento I2C/SPI

**Desafio**: criar um código que varra o barramento I2C ou SPI para identificar endereços de dispositivos conectados e retornar o ID do fabricante via serial.

**Periféricos**: I2C, SPI, UART.

**Conceito**: entendimento profundo de *start/stop bits*, acknowledge (ACK) e tempos de subida/descida do sinal no hardware.

#### IV. Cronômetro de Alta Precisão (Milissegundos)

**Desafio**: um sistema que conte o tempo entre dois eventos físicos (ex: passagem por dois sensores ópticos) usando interrupções externas.

**Periféricos**: External Interrupts (EXTI), Timers em modo Input Capture.

**Conceito**: medição de largura de pulso e tratamento de debounce de hardware/software sem usar delay().

#### V. Terminal de Eco com Buffer Circular

**Desafio**: implementar um buffer circular para gerenciar dados recebidos via UART. O sistema deve processar comandos de texto simples (ex: "LED_ON", "LED_OFF") enquanto mantém a comunicação ativa.

**Periféricos**: UART (TX/RX), Timers.

**Conceito**: Gerenciamento de ponteiros de memória e prevenção de buffer overflow em sistemas de tempo real.

## **Módulo 2: Gestão de Energia e Integridade de Sinais**

**Objetivo**: garantir a autonomia sistêmica e a preservação da integridade dos dados frente ao ruído eletromagnético gerado pelos atuadores.

* **Hardware**: estudo das químicas de células **LiPo** e *Li-ion*, focando em curvas de descarga e resistência interna. Projeto de circuitos de proteção (BMS) e topologias de conversores DC-DC (*Buck*, *Boost* e *Buck-Boost*). O foco em integridade de sinal exige o domínio de planos de terra (GND) separados para sinais analógicos e digitais, além do cálculo de capacitores de desacoplamento para filtragem de ruído de alta frequência.  

>**CUIDADO**: **mantenha um extintor (Extintor AVD (Dispersão Aquosa de Vermiculita), ou Classe D) e um balde de areia de praia perto da sua bancada.** As baterias de lítio são perigosas e podem incendiar se manuseadas incorretamente. Quando pegam fogo, água e extintores químicos são ineficazes. Nunca deixe uma bateria carregando sem supervisão. ***Lembre-se: manusear incorretamente é praticamente sinônimo de aprender. Você vai errar!!!***.

* **Software**: implementação de monitoramento de tensão e corrente via ADC. Para manter o determinismo, a leitura deve ser feita via DMA (*Direct Memory Access*), permitindo que os dados cheguem à memória sem intervenção da CPU. As rotinas de segurança (*Failsafes*) e o gerenciamento de energia devem ser implementados como Máquinas de Estados Finitos. 

>É terminantemente proibido o uso de `delay()`, `sleep()` ou laços de espera para aguardar a estabilização de tensões; deixe de ser preguiçoso, pense e encontre uma forma de utiliza os Timers do módulo anterior para cadenciar as amostragens.  

* **Aplicação de IA**: consulte as ferramentas de IA para modelagem de filtros digitais e cálculo de eficiência de conversores chaveados. **Não esqueça que modelos de IA podem alucinar sobre a capacidade de dissipação térmica de componentes e limites de corrente de trilhas de PCB**. Esta alucinação, eu ainda não vi, mas nas comunidades do **X.com**, esta semana, teve muito burburinho sobre isso. Para não cair em um erro de alucinação, valide as sugestões de componentes com cálculos de potência:

$$P_{total} = I_{rms}^{2} \times R_{dson} + P_{switching}$$

Utilize a IA para traduzir requisitos de autonomia (ex: "operar por 2 horas com pico de 10A") em especificações de capacidade de bateria e taxa de descarga (C-rate), mas sempre peça para o assistente explicar o que está fazendo, pegue uma calculadora e verifique os resultados com as leis da física.

### Projetos para o Módulo 2

Estes projetos devem ser executados seguindo as diretrizes de "tolerância zero" para laços de espera e uso obrigatório de acesso direto à memória (DMA).

>**AVISO DE SEGURANÇA MANDATÓRIO**: antes de iniciar qualquer um dos projetos abaixo, verifique a presença do **Extintor AVD ou Classe D** e do **balde de areia**. O manuseio de células LiPo/Li-ion exige atenção total. Não avance se a área de trabalho não estiver protegida.

#### **I. Monitor de Telemetria de Células via DMA**

**Desafio**: criar um sistema que monitore a tensão de uma bateria e a corrente de carga em tempo real sem interrupções da CPU.

**Hardware**: divisor de tensão resistivo (calculado para a faixa do ADC) e sensor de corrente (***Shunt*** ou *Efeito Hall*). Implementação de capacitores de desacoplamento nos pinos de alimentação do sensor.  

**Software**: configuração do ADC em modo contínuo com transferência de dados via DMA para um buffer circular. Uma Máquina de Estados Finitos (FSM) deve processar esses dados para detectar sub-tensão.  

**Objetivo**: garantir que a CPU esteja livre para outras tarefas enquanto a telemetria é atualizada em segundo plano.

#### **II. Estação de Teste de Eficiência de Conversor Buck**

**Desafio**: Calcular a eficiência real de um conversor DC-DC comparando a potência de entrada e saída.

**Hardware**: conversor *Buck* chaveado, resistores de carga para diferentes níveis de corrente.  

**Software**: leitura simultânea de dois canais ADC (entrada e saída). O cálculo de eficiência deve ser feito em uma tarefa disparada por um Timer a cada 100ms.  

**Aplicação de IA**: solicite ao assistente de IA o cálculo da eficiência teórica baseada no $R_{dson}$ dos MOSFETs do conversor e compare com os dados obtidos. Cálculo:  
  
  $$\eta = \frac{V_{out} \times I_{out}}{V_{in} \times I_{in}} \times 100$$

### **III. Sistema de Proteção Ativa (BMS via Software)**

**Desafio**: implementar um sistema de corte de carga (*Load Shedding*) baseado em limites térmicos e elétricos.

**Hardware**: MOSFET de potência atuando como chave geral, sensor de temperatura NTC.  

**Software**: Máquina de Estados Finitos com estados: STARTUP, NORMAL_OP, THERMAL_THROTTLING e EMERGENCY_SHUTDOWN. O uso de `sleep()` para debouncing é proibido; utilize Timers para validar leituras anômalas.  

**Objetivo**: proteger a integridade química da bateria através de lógica de software rápida e determinística.

### **IV. Analisador de Ruído e Integridade de Sinal**

**Desafio**: visualizar o impacto do ruído de motores no barramento de dados e testar a eficácia de filtros físicos.

**Hardware**: Um motor DC pequeno acionado por PWM (ruído) e um sensor analógico sensível. Teste em duas protoboards/PCBs: uma com planos de terra interligados incorretamente e outra com separação física de AGND (Analógico) e DGND (Digital).  

**Software**: Amostragem de alta velocidade via ADC+DMA para capturar transientes de ruído.  

**Aplicação de IA**: Use a IA para projetar um filtro digital de primeira ordem (passa-baixa) para limpar os dados coletados e implemente-o no código para comparar com o sinal bruto. Neste ponto, lembre-se, o objetivo é entender o impacto do design físico na qualidade do sinal. Ainda assim, você deve validar os resultados com cálculos manuais de frequência de corte:

$$f_c = \frac{1}{2 \pi R C}$$

### **V. Estimador de Autonomia Dinâmica (Time-to-Empty)**

**Desafio**: Desenvolver um algoritmo que preveja quanto tempo de vida resta à bateria com base no perfil de consumo atual.

**Hardware**: integração dos sensores de corrente e tensão dos projetos anteriores.  

**Software**: implementação de um integrador numérico (Contagem de Coulomb) para estimar o estado de carga (SoC). A Máquina de Estados Finitos deve atualizar a estimativa de tempo restante a cada mudança significativa de carga.  

**Aplicação de IA**: Peça para a IA explicar o ***Modelo de Peukert*** para baterias de descarga rápida e ajuste o seu algoritmo para considerar a queda de capacidade em altas correntes.  

**Validação**: Compare a estimativa da IA com a curva de descarga real observada nos seus Timers.

## **Módulo 3: Atuação e Dinâmica de Movimento**

**Objetivo**: transformar sinais elétricos em força mecânica precisa e controlada, dominando a física dos motores e a eletrônica de potência.

* **Hardware**: estudo de motores DC, servomotores, motores de passo (*steppers*) e motores **BLDC** (*Brushless*). Foco no funcionamento de **Pontes H**, drivers de corrente e a física da Força Eletromotriz (*Back-EMF*). Domínio de algoritmos de Controle Orientado a Campo (**FOC**) para obter torque constante e movimentos suaves.

**CUIDADO**: **Motores geram torque e inércia. Dedos, cabos e sensores podem ser esmagados ou cortados.** Sempre teste seus algoritmos de controle com a fonte de alimentação limitada em corrente. Motores BLDC, em especial, podem entrar em curto-circuito e explodir os MOSFETs do driver se a lógica de chaveamento estiver errada. Use óculos de proteção. 

>**Regra de Ouro: se você não queimou um MOSFET, não está tentando o suficiente.**

* **Software**: geração de sinais PWM via hardware com resolução avançada. Implementação de malhas de controle PID e ***# Transformadas de Park e Clarke*** para FOC. Seus códigos devem ser escritos em **C++** utilizando classes para representar os motores, mas sem bibliotecas externas.

É terminantemente proibido o uso de qualquer função que bloqueie o processamento para gerar pulsos de motor de passo. Se você usar um laço para contar pulsos, seu robô será lerdo e impreciso. Use Timers em modo Output Compare e Interrupts. O processador deve estar livre para calcular a próxima posição enquanto o periférico cuida do movimento físico.

* **Aplicação de IA**: utilize a IA para auxiliar na sintonização teórica dos ganhos **Kp**, **Ki** e **Kd** do seu **PID** e para explicar a matemática vetorial por trás do FOC. **Não confie cegamente nos parâmetros de malha sugeridos**. A IA não conhece o atrito real dos seus eixos nem a folga das suas engrenagens. Valide os cálculos de torque:

$$\tau = K_t \times I$$

Sempre peça para o assistente detalhar a lógica de controle vetorial, mas implemente bit a bit nos registradores do Timer.

### **Projetos para o Módulo 3**

#### **I. Driver de Motor de Passo de Alta Velocidade**

**Desafio**: acionar um motor de passo em altas rotações com rampas de aceleração e desaceleração senoidais, sem perder passos e sem travar a CPU.

**Periféricos**: Timers (Output Compare), GPIO.

**Conceito**: cálculo dinâmico da frequência do Timer em tempo real para criar rampas suaves.

#### **II. Controlador de Velocidade em Malha Fechada (PID)**

**Desafio**: manter a rotação de um motor DC constante, independentemente da carga aplicada ao eixo.

**Periféricos**: Timer (PWM), Timer (Encoder Mode), ADC.

**Conceito**: implementação de um controlador PID discreto. O tempo de amostragem ($T_s$) deve ser rigorosamente controlado por um Timer de hardware.

#### **III. Driver BLDC "Open-Loop" (Six-Step)**

**Desafio**: girar um motor **BLDC** utilizando a sequência de 6 passos, controlando a comutação manualmente via software.

**Periféricos**: 3 canais de PWM (com sinais complementares e *Dead-time* inserido via hardware), GPIO.

**Conceito**: entendimento do defasamento de 120 graus e a importância do *Dead-time* para evitar o curto-circuito da ponte.

## Projeto Integrador dos Módulos 1, 2 e 3: Driver BLDC com Field-Oriented Control (FOC) *sensorless* em Malha Fechada

O **F**ield-**O**riented **C**ontrol, **FOC**, é o que separa robôs de brinquedo de sistemas profissionais (drones de alta performance, braços robóticos industriais, veículos elétricos). Implementá-lo em **bare-metal** em um chip como o [STM32F407VET6](https://www.st.com/resource/en/datasheet/dm00037051.pdf) é totalmente viável e interessante.

No STM32F407VET6 o Cortex-M4F com FPU single-precision lida tranquilamente com as operações de ponto flutuante em taxas de $10$–$20 \text{ kHz}$ (*loop* de corrente típico). Você vai usar os timers avançados do chip para gerar os 3 PWMs complementares com *dead-time* automático, ADC + DMA para amostragem síncrona de correntes, e interrupções precisas para o *loop* de controle.

> Se você me conhece, sabe que estou projetando uma geometria nova para motores BLDC. Acredito que aqui está o futuro dos motores elétricos em robótica. Existem muitas oportunidades aqui e creio que tive uma boa ideia.

> **AVISO DE SEGURANÇA OBRIGATÓRIO**: Motores BLDC em FOC podem girar com torque brutal e velocidade alta. **Sempre limite a corrente da fonte de alimentação inicial (use uma fonte de bancada com limite de $2$–$3\text{ A}$)**. Use óculos de proteção. Os MOSFETs vão queimar se o *dead-time* estiver errado, fases invertidas ou corrente muito alta. Tenha MOSFETs sobressalentes. **Nunca teste em bateria LiPo sem BMS completo e limitação de corrente**.

**Desafio**: implementar um controle [FOC](https://www.ti.com/lit/ab/slaae96a/slaae96a.pdf?ts=1768659465805) completo (*sensorless*, baseado em observador de *back-EMF* ou com sensores Hall opcionais) para um motor BLDC pequeno, alcançando:
- controle preciso de velocidade (*setpoint* via UART ou potenciômetro).
- controle de torque (corrente $Iq$) com resposta rápida.
- rampa de aceleração/desaceleração suave.
- proteção contra *overcurrent* e *stall* via software.
- *loop* de corrente a $≥10 \text{ kHz}$ e *loop* de velocidade a $≥1 \text{ kHz}$.

O sistema deve iniciar em *open-loop* (*six-step* alinhado) para alinhar o rotor, transitar para ***FOC sensorless*** e manter rotação estável mesmo com variações de carga. Tudo sem bloquear a CPU. O *loop* principal deve ser uma Máquina de Estados Finitos disparada por interrupções.

**Hardware Necessário**:

- Motor BLDC trifásico pequeno (ex: 2212/2208 de drone, ou **GB2208 gimbal motor — KV 80–200** para facilitar *sensorless* em baixa velocidade).
- Ponte H trifásica completa:
  - 6 MOSFETs N-channel low-Rdson (ex: IRF3205 ou melhores como BSC028N06LS3, 60V/40A+).
  - 3 gate drivers half-bridge (ex: IR2184 ou FAN7382 — essencial para bootstrap correto).
  - Ou módulo ESC pronto low-level (ex: placas “BLDC driver 3-phase” chinesas com MOSFETs expostos para você controlar os gates diretamente).
- Sensores de corrente fase:
  - Opção 1 (recomendada inicial): 3 shunts de baixa resistência (0.001–0.005Ω, 5W) + amplificadores diferenciais (ex: INA240A2 ou 3x op-amp LM358 configurado como diff-amp).
  - Opção 2: 2–3 sensores Hall bidirecionais (ex: ACS712-20A ou melhor INA226 I2C — mas para FOC real, prefira shunts para amostragem síncrona).
- Divisor de tensão para medir Back-EMF (opcional para debug).
- Sensores Hall integrados no motor (se o motor tiver — muitos BLDC de drone não têm; use *sensorless* puro).
- Capacitores de bulk grandes no barramento DC (1000–2200uF 25V+) + capacitores cerâmicos 100nF perto dos MOSFETs.
- Resistor de bleed/pulled para bootstrap.

**Periféricos Usados**:

- Timer avançado (TIM1 ou TIM8): 3 canais PWM complementares + *dead-time* hardware (crucial!).
- ADC1/ADC2/ADC3 em modo injected + trigger por timer + DMA circular para amostragem síncrona das 2–3 correntes no centro do PWM low-side (técnica de current sensing padrão).
- Timer geral para *loop* de velocidade (Output Compare ou Update interrupt).
- UART para setpoint e telemetria (enviar ângulo elétrico, Iq/Id, velocidade real).
- GPIO + EXTI para zero-cross detection (se implementar hybrid sensorless).
- DMA para transferir amostras de corrente diretamente para buffer.

**Conceito Principal**:

1. **Geração de PWM**: SVM (*Space Vector Modulation*) via timer avançado. Calcule *duty cycles* para as 3 fases a partir de $Vα$/$Vβ$.

2. **Amostragem de corrente**: Sincronizada no centro do pulso PWM low-side para minimizar ruído.

3. **Transformadas**:
   - Clarke: $Iα$, $Iβ$ a partir de $Ia$, $Ib$ ($Ic = -Ia-Ib$).
   - Park: $Id$, $Iq$ a partir de $Iα$/$Iβ$ e $θ$ (ângulo elétrico).
   - Inverse Park: $Vd$/$Vq$ → $Vα$/$Vβ$.

4. **Controladores**: 2 $PI$ separados ($Id = 0$ para maximizar torque, $Iq$ = setpoint de torque/velocidade).

5. **Estimador de posição/velocidade** (*sensorless*):
   - Início: open-loop ramp para alinhar e acelerar até velocidade mínima.
   - Transição: observador PLL ou sliding-mode observer para estimar $θ$ e $ω$ a partir de *back-EMF* (calculado via $V$ fase reconstruído ou correntes).

6. **Loop de velocidade**: $PI$ externo que gera setpoint $Iq$.

Tudo em **ponto flutuante** usando a FPU do M4. Use funções matemáticas rápidas ($sin$/$cos$ via *lookup* table de $1024$ entradas para velocidade).

**Software** (bare-metal puro, C++ com classes):

- Crie classes: `MotorBLDC`, `FOCController`, `CurrentSense`, `SVM`, `PLLObserver`.
- FSM principal com estados: `ALIGN`, `OPEN_LOOP_RAMP`, `CLOSED_LOOP_FOC`, `FAULT`.
- *loop* de corrente em interrupção do timer PWM.
- *loop* de velocidade em interrupção mais lenta.
- **Proibido**: qualquer `delay()`, `sleep()` ou laço de espera. Use apenas interrupções e DMA.
- Telemetria: envie via UART a cada 100ms (velocidade, correntes, $θ$, erros $PI$).

**Aplicação de IA**:

- Peça à IA para gerar o código base das transformadas [Clarke/Park/Inverse](https://kaliasgoldmedal.yolasite.com/resources/MCDSP/UNIT5/Clarke_park.pdf) em C++ otimizado com FPU intrinsics (ex: `__fmul`, etc.).
- Use IA para derivar as equações discretas dos controladores $PI$ (com *anti-windup* e limite de saída).
- Solicite sintonização inicial de $Kp$/$Ki$ para os *loops* de corrente e velocidade baseada em parâmetros do motor ($Kt$, $L$, $R$, polos).
- Para o observador *sensorless*: peça explicação detalhada do PLL ou SMO, com equações discretas.
- **Sempre valide**: teste com motor bloqueado (torque zero), compare ângulo estimado com Hall real (se disponível), meça eficiência vs six-step. IA alucina constantes de motor ou ganhos instáveis — corrija com osciloscópio e testes reais. Use fórmulas clássicas:
  
  $$T = \frac{3}{2} p \lambda I_q$$

  (torque proporcional a $Iq$, com $Id=0$).

**Critérios de Sucesso**:

- Motor gira silenciosamente (sem o “chiado” do six-step). Se chegar aqui, saia para comemorar com a família.
- Mantém velocidade constante com carga aplicada/removida.
- Transição *open → closed loop* sem travar.
- Consumo de CPU $< 70\%$ a $16$–$20 \text{ kHz}$ *loop* de corrente.
- Telemetria mostra $Id \approx 0$ e $Iq$ proporcional ao torque.

Quem implementa FOC bare-metal entende eletrônica de potência e controle embarcado. Depois disso, você estará pronto para robôs de competição ou produtos reais.

>Notadamente os braços robóticos que permitem [backdriving](https://enriquedelsol.com/2017/12/05/backdrivability/) suave e controle preciso. **Se você dominar isso, estará à frente de 90% dos engenheiros de robótica no mercado**.

## **Módulo 4: Sensoriamento e Sistema Nervoso (Comunicação)**

**Objetivo**: dotar o sistema de percepção ambiental e garantir que múltiplos módulos conversem através de barramentos industriais robustos.

* **Hardware**: integração de Encoders magnéticos e ópticos, IMUs (Acelerômetro e Giroscópio) e sensores de distância (LiDAR ou Ultrassom). Domínio físico dos barramentos I2C, SPI e, obrigatoriamente, CAN Bus. Você deve aprender a projetar a terminação resistiva e entender a física do sinal diferencial.  

* **Software**: leitura de sensores complexos utilizando DMA para evitar sobrecarga da CPU. Implementação de protocolos de comunicação robustos com Checksum ou CRC. O processamento de dados de sensores deve ser assíncrono.

* **Aplicação de IA**: utilize IA para o desenvolvimento de filtros digitais de fusão sensorial (como o Filtro Complementar ou Kalman). Peça para a IA gerar a matriz de covariância teórica, mas valide-a experimentalmente. Lembre-se: IAs alucinam protocolos de comunicação que não existem. Verifique os endereços de registro do sensor diretamente no manual do fabricante.

### **Projetos para o Módulo 4**

#### **I. Unidade de Medição Inercial (IMU) via DMA**

**Desafio**: ler dados de 6 eixos (Acc/Giro) a 1kHz e calcular a inclinação do robô sem intervenção direta da CPU na coleta.

**Periféricos**: SPI (ou I2C), DMA, Timers.

**Conceito**: configuração do DMA para preencher um buffer de memória assim que o sensor sinalizar "Data Ready".

#### **II. Rede CAN Bus Multi-Nó**

**Desafio**: estabelecer comunicação entre duas placas distintas, enviando telemetria de um lado e comandos de motor do outro.

**Periféricos**: CAN Controller (bxCAN ou FDCAN), Transceivers físicos.

**Conceito**: arbitragem de barramento, identificadores de mensagens e tratamento de erros de rede.

#### **III. Gateway de Fusão Sensorial e Odometria Espacial**

**Desafio**: sincronizar dados de um LiDAR (UART), uma IMU (SPI) e Encoders de quadratura (Timers) em um único fluxo de dados transmitido via CAN Bus com carimbo de tempo (*Timestamp*).

**Periféricos**: UART (DMA), SPI (DMA), Timers (Encoder Mode), CAN Bus, Global Sync Timer.

**Conceito**: o desafio aqui é o sincronismo. Você deve criar um sistema em que os dados do LiDAR (distância), da IMU (ângulo) e dos Encoders (posição) se refiram ao mesmo instante temporal. Utilize um Timer de 32 bits de alta frequência como base de tempo global. Cada mensagem CAN enviada deve conter o dado do sensor e o valor deste Timer no momento exato da captura.

**Aplicação de IA**: Solicite à IA uma estrutura de pacotes eficiente para encapsular esses dados no limite de 8 bytes (ou 64 bytes se usar CAN-FD) do barramento CAN. Peça também para o assistente sugerir uma lógica de priorização de mensagens: qual dado é mais importante para o robô não colidir?

Como eu sugeri o **STM32F407VET6** para o **Projeto Integrador 1** vamos usá-lo novamente aqui para o Gateway de Fusão Sensorial e Odometria. Este chip possui uma variedade rica de periféricos que o tornam ideal para este desafio:

1. **Global Sync Timer (Base de Tempo)**:  

   * **Recurso**: TIM2 ou TIM5.  
   * **Justificativa**: Diferente dos outros timers de 16 bits, o TIM2 e o TIM5 são de **32 bits**. Isso permite que você tenha um carimbo de tempo (*timestamp*) de alta resolução que não sofrerá *overflow* rapidamente, o que é fundamental para sincronizar dados de sensores com frequências diferentes.  

2. **LiDAR (Interface UART)**:  

   * **Recurso**: USART1 ou USART6 (conectados ao barramento APB2 de alta velocidade).  
   * **Configuração**: Deve ser operado com o DMA2. O fluxo de dados do LiDAR costuma ser intenso, e o uso de interrupção por caractere sobrecarregaria a CPU desnecessariamente.  

3. **IMU (Interface SPI)**:  

   * **Recurso**: SPI1.  
   * **Configuração**: Também operado via DMA. Como o protocolo SPI é síncrono e atinge altas velocidades (até 42 Mbits/s neste chip), a transferência direta para a memória evita que o sistema "trave" esperando o fim da transmissão dos 6 eixos de dados.  

4. **Encoders (Odometria)**:  

   * **Recursos**: TIM3 e TIM4 em modo **Encoder Interface**.  
   * **Configuração**: Estes timers possuem hardware dedicado para decodificar sinais de quadratura (fases A e B), contando pulsos automaticamente sem intervenção do software.  

5. **Comunicação Externa (CAN Bus)**:  
   * **Recurso**: CAN1 (Master) e CAN2 (Slave).  
   * **Configuração**: O STM32F407 possui o periférico **bxCAN**, que suporta filtros de aceitação por hardware. Isso significa que o chip pode descartar mensagens irrelevantes no barramento sem que o software precise processá-las.

##### **Cálculo de Largura de Banda no Barramento**

Ao integrar esses sensores, você deve considerar a carga de dados. Se o seu LiDAR opera a uma taxa de $115200$ bps e a sua IMU a $1$ kHz, o cálculo da carga de interrupção (se você não usasse DMA) seria:

$$f_{total\_ints} = f_{uart\_char} + f_{imu\_sample} \times N_{bytes}$$

Com o uso de DMA, o impacto na CPU é reduzido drasticamente, limitando-se apenas ao processamento da lógica de fusão sensorial na Máquina de Estados Finitos.

## **Módulo 5: Design Profissional e Integração Mecatrônica**

**Objetivo**: consolidar o hardware em placas profissionais e garantir que a eletrônica sobreviva ao mundo físico (vibração, calor e impacto).

* **Hardware**: design avançado de PCBs utilizando [KiCad](https://www.kicad.org/). Foco em dissipação térmica, largura de trilhas para alta corrente e caminhos de retorno de corrente. Integração mecânica em CAD 3D (Fusion 360, SolidWorks ou OnShape) para garantir que conectores e dissipadores se encaixem no chassi.
  
* **Software**: arquitetura de firmware modular. O código deve ser organizado em camadas: Driver (Registradores), HAL Própria (Abstração Lógica) e App (Comportamento). Documentação das APIs internas é obrigatória.

>Breadboards (protoboards) são para crianças. Robôs profissionais usam PCBs com planos de terra sólidos e conectores com trava. **Se o seu projeto depende de "jumpers" coloridos, ele não é um robô, é um ninho de ratos esperando para falhar**.

* **Aplicação de IA**: use ferramentas de IA para revisar o seu layout de PCB em busca de erros de etiqueta ou sugestões de posicionamento de componentes para evitar interferência eletromagnética (EMI). Use a IA para gerar scripts de automação para exportar seus arquivos de fabricação (*Gerbers*).

### **Projetos para o Módulo 5**

#### **I. Projeto e Montagem de uma Power Shield Modular**

**Desafio**: projetar uma PCB que contenha o sistema de potência (Módulo 2) e os drivers de motor (Módulo 3) em um formato modular que se encaixe no seu microcontrolador.

**Ferramentas**: KiCad, OnShape.

**Conceito**: cálculo de largura de trilha para 10A e posicionamento de capacitores de *bulk* para evitar quedas de tensão no barramento lógico.

## **Projeto Integrador Final dos Módulos 1, 2, 3, 4 e 5:  O Robô Autônomo**

**Desafio**: montar o sistema completo, integrando os 4 módulos anteriores em um chassi funcional. O robô deve navegar um percurso simples, monitorando sua própria energia e corrigindo sua trajetória via sensores, tudo em tempo real.

**Conceito**: orquestração de múltiplas Máquinas de Estados Finitos rodando simultaneamente. Este é o teste definitivo de sua capacidade como engenheiro de robótica.

**Nota de Encerramento**: se você chegou aqui e seu robô funciona sem usar `delay()` e sem bibliotecas prontas, parabéns. Você agora entende o silício. O mercado de engenharia de elite é o seu lugar.

## **Lista de Materiais (BOM) \- Curso de Robótica Bare-Metal**

Este documento contém a lista consolidada de hardware para os 5 módulos do curso. Os preços são estimativas baseadas no mercado de 2025/2026.

### **1\. Hardware de Processamento e Base (Obrigatório)**

| Item | Descrição Técnica | Qtd | Est. (R$) |
| :---- | :---- | :---- | :---- |
| MCU Principal | STM32F407VET6 Black Pill / Core Board | 1 | 120,00 |
| Programador | ST-Link V2 (Clone ou Original) | 1 | 45,00 |
| Protoboard | Protoboard 830 pontos (Transparente ou Branca) | 3 | 90,00 |
| Jumpers | Kit Macho-Macho / Macho-Fêmea (20cm) | 1 | 40,00 |
| Instrumentação | Osciloscópio USB (ex: Hantek 6022BE ou DSO138) | 1 | 350,00 |
| **Analisador Lógico** | **USB 8 canais 24MHz (Compatível Saleae)** | **1** | **60,00** |
| Fonte de Bancada | Módulo Regulador 5V/3A ou Fonte Ajustável | 1 | 150,00 |
| **Subtotal** |  |  | **R$ 855,00** |

### **2\. Gestão de Energia e Segurança (Módulo 2\)**

| Item | Descrição Técnica | Qtd | Est. (R$) |
| :---- | :---- | :---- | :---- |
| Bateria | Célula Li-ion 18650 3.7V (Original) | 2 | 80,00 |
| Carregador | Módulo TP4056 com proteção | 3 | 15,00 |
| Sensor Corrente | INA219 (I2C) ou ACS712 5A | 2 | 50,00 |
| Conversores | Kit DC-DC (Buck/Boost/Buck-Boost) | 3 | 60,00 |
| MOSFETs | IRFZ44N / IRF3205 (N-Channel) | 10 | 50,00 |
| Segurança | Extintor AVD/Classe D e Balde de Areia | 1 | 250,00 |
| **Subtotal** |  |  | **R$ 505,00** |

### **3\. Atuação, Movimento e Sensoriamento (Módulos 3 e 4\)**

| Item | Descrição Técnica | Qtd | Est. (R$) |
| :---- | :---- | :---- | :---- |
| Motores | Motor DC (Gearbox), Stepper NEMA17, BLDC 2212 | 3 | 350,00 |
| Drivers | Ponte H L298N e Gate Drivers IR2101 | 2 | 100,00 |
| IMU | MPU6050 ou MPU9250 (SPI/I2C) | 1 | 45,00 |
| Comunicação | Transceptor CAN SN65HVD230 | 2 | 40,00 |
| Encoders | Disco Óptico ou Sensor Magnético AS5600 | 2 | 80,00 |
| **Subtotal** |  |  | **R$ 615,00** |

### **4\. Integração e Consumíveis (Módulo 5\)**

| Item | Descrição Técnica | Qtd | Est. (R$) |
| :---- | :---- | :---- | :---- |
| Fabricação PCB | Lote de 5-10 PCBs (JLCPCB/PCBWay) | 1 | 150,00 |
| Chassi | Base em Acrílico ou Impressão 3D | 1 | 120,00 |
| Componentes | Kit Resistores, Capacitores, LEDs, Botões | 1 | 100,00 |
| Soldagem | Estação de Solda, Estanho, Pinças | 1 | 220,00 |
| **Subtotal** |  |  | **R$ 590,00** |

#### **Resumo Financeiro Estimado em Hardware**

* **Investimento Inicial (Base \+ Segurança):** R$ 1.105,00  
* **Investimento por Módulo (Restante):** R$ 1.460,00  
* **TOTAL ESTIMADO:** **R$ 2.565,00**

>*Nota: Os valores podem variar dependendo do câmbio e do local da compra, quantidade e qualidade do fornecedor.*