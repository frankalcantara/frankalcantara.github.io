const text = 
{
    english: 
    {
        by: "by",
        version: "Version",
        documentation: "Documentation",
        commands: "Commands",
        clockspeed: "Clock Speed" + '<span class="tooltiptext clkExp"></span>',
        accumulator: "Accumulator" + '<span class="tooltiptext accumulatorExp"></span>',
        programcounter: "Program Counter",
        usecache: "Use Cache" + '<span class="tooltiptext cacheExp"></span>',
        usepipeline: "Use Pipeline" + '<span class="tooltiptext-up pipelineExp"></span>',
        curcode: "Code",
        loadtoram: "Load Code to RAM",
        runcode: "Run Code",
        ramname: "Random Access Memory (RAM)" + '<span class="tooltiptext-left ramExp"></span>',
        cpuname: "Central Processing Unit",
        commandlist: "Command List",
        name: "Name",
        value: "Value",
        clear: "Clear",
        input: "Input",
        output: "Output",
        cacheconsole: "Cache Console",
        commandname: "Command Name",
        syntax: "Syntax",
        description: "Description",
        codeplaceholder: "Code goes here...",
        nocache: "There is no Cache",
        custom: "Custom",
        addtwonum: "Add Two Numbers",
        sortthree: "Sort Three Numbers",
        countdown: "Countdown Timer",
        inputplaceholder: "Input: 1 2 3 ...",
        fetch: "Fetching" + '<span class="tooltiptext-up fetchExp"></span>',
        decode: "Decoding" + '<span class="tooltiptext-up decodeExp"></span>',
        execute: "Executing" + '<span class="tooltiptext-up executeExp"></span>',
        writeback: "Write Back" + '<span class="tooltiptext-up writebackExp"></span>',
        updatecache: "Updating Cache",
        invalidinputs: "Invalid value in input",
        missinghalt: "Error: Accessing variable as instruction, missing halt",
        haltcode: "Halt Code",
        texteditor: "Text Editor",
        executiontime: (start, finish) => { return `Execution time: ${(finish - start)/1000} seconds`; },
        consolebranch: (newPos) => { return `Jumping to position: ${newPos}`; },
        consoleignorebrp: () => { return "Ignoring BRP"; },
        consoleignorebrz: () => { return "Ignoring BRZ"; },
        consoleinput: (input) => { return `Getting input: ${input}`; },
        consoleoutput: (output) => { return `Writing output: ${output}` },
        consolehalt: () => { return "Ending program"; },
        consoleadd: (value) => { return `Adding ${value} to accumulator`; },
        consolesub: (value) => { return `Subtracting ${value} from accumulator`; },
        consolesta: (p, value) => { return `Storing at: ${p} value: ${value}`; },
        consolelda: (p) => { return `Loading from: ${p} to the accumulator`; },
        cachehit: (address) => { return `Cache hit on address: ${address}`; },
        cachemiss: (address) => { return `Cache miss on address: ${address}`; },
        manualhalt: "Code manually halted",
        fetching: (address) => { return `Fetched: ${address}`; },
        aboutus: "About us",
        //Tooltips
        accumulatorExp: "The accumulator is a register that holds the results of mathematical operations and memory reads.",
        pcExp: "The program counter (PC) is a register which the value corresponds to the current instruction position in the memory.",
        cirExp: "The current instruction register (CIR) contains the latest instruction address fetched from memory.",
        marExp: "The memory address register (MAR) contains the address of a memory location which is about to be written/read from memory",
        mdrExp: "The memory data register (MDR) contains the value that is about to be written/read from memory",
        clkExp: "Total of cycles per second (Fetch - Decode - Execute - Writeback)",
        ramExp: "The RAM (Random Access Memory) is a dynamic access memory in which the program instructions are stored, as well as the values used by the program.",
        fetchExp: "The fetch is a operation from the machine cycle that retrieves an address from memory.",
        decodeExp: "The decode is a operation from the machine cycle that decodes an address from memory transforming it to an instruction.",
        executeExp: "The execute is a operation from the machine cycle that executes an decoded instruction from memory.",
        writebackExp: "The writeback is a operation from the machine cycle that writes data to the registers.",
        cacheExp: "Cache memory is a faster, smaller memory compared to RAM. The processor stores instructions that can be used in it to reduce address lookup time.",
        pipelineExp: "Instruction pipelining is the parallelization of machine cycles, in which all cycles execute simultaneously as soon as they become available.",
    },
    portuguese: 
    {
        by: "por",
        version: "Versão",
        documentation: "Documentação",
        commands: "Comandos",
        clockspeed: "Velocidade de Clock" + '<span class="tooltiptext clkExp"></span>',
        accumulator: "Acumulador" + '<span class="tooltiptext accumulatorExp"></span>',
        programcounter: "Contador",
        usecache: "Usar Cache" + '<span class="tooltiptext cacheExp"></span>',
        usepipeline: "Usar Pipeline" + '<span class="tooltiptext-up pipelineExp"></span>',
        curcode: "Código",
        loadtoram: "Carregar código para a RAM",
        runcode: "Executar Código",
        ramname: "Memória de Acesso Aleatório (RAM)" + '<span class="tooltiptext-left ramExp"></span>',
        cpuname: "Unidade central de processamento",
        commandlist: "Lista de comandos",
        name: "Nome",
        value: "Valor",
        clear: "Limpar",
        input: "Entrada",
        output: "Saída",
        cacheconsole: "Console do Cache",
        commandname: "Nome",
        syntax: "Sintaxe",
        description: "Descrição",
        codeplaceholder: "Escreva o código aqui...",
        nocache: "Não possui Cache",
        custom: "Customizado",
        addtwonum: "Adicionar Dois Números",
        sortthree: "Organizar Três Números",
        countdown: "Temporizador",
        inputplaceholder: "Entrada: 1 2 3 ...",
        fetch: "Resgatando",
        decode: "Decodificando",
        execute: "Executando",
        writeback: "Retorno",
        updatecache: "Atualizando Cache",
        invalidinputs: "Valor indevido na entrada",
        missinghalt: "Erro: Acessando variável como instrução, faltando halt",
        haltcode: "Encerrar Código",
        texteditor: "Editor de Texto",
        executiontime: (start, finish) => { return `Tempo de execução: ${(finish - start)/1000} segundos`; },
        consolebranch: (newPos) => { return `Pulando para posição: ${newPos}`; },
        consoleignorebrp: () => { return "Ignorando BRP"; },
        consoleignorebrz: () => { return "Ignorando BRZ"; },
        consoleinput: (input) => { return `Obtendo entrada: ${input}`; },
        consoleoutput: (output) => { return `Escrevendo saída: ${output}`; },
        consolehalt: () => { return "Encerrando programa"; },
        consoleadd: (value) => { return `Adicionando ${value} ao acumulador`; },
        consolesub: (value) => { return `Subtraindo ${value} do acumulador`; },
        consolesta: (p, value) => { return `Guardando em: ${p} valor: ${value}`; },
        consolelda: (p) => { return `Carregando de: ${p} para o acumulador`; },
        cachehit: (address) => { return `Acerto de cache no endereço: ${address}`; },
        cachemiss: (address) => { return `Falha de cache no endereço: ${address}`; },
        manualhalt: "Código encerrado manualmente",
        fetching: (address) => { return `Resgatado: ${address}`; },
        aboutus: "Sobre nós",
        // Tooltips
        pcexplanation: "O Program Counter é um valor que corresponde a posição atual da instrução na memória.",
        accumulatorExp: "O acumulador é um registrador que mantém os resultados das operações matemáticas e leituras de memória.",
        pcExp: "O contador de programa (PC) é um registrador cujo valor corresponde à posição atual da instrução na memória.",
        cirExp: "O registrador de instrução atual (CIR) contém o endereço da última instrução buscada da memória.",
        marExp: "O registrador de endereço de memória (MAR) contém o endereço de um local de memória que está prestes a ser escrito/lido da memória",
        mdrExp: "O registrador de dados de memória (MDR) contém o valor que está prestes a ser escrito/lido da memória",
        clkExp: "Total de ciclos por segundo (Fetch - Decode - Execute - Writeback)",
        ramExp: "A memória RAM (Random Access Memory) é uma memória de acesso dinâmico no qual é guardado as instruções do programa escrito assim como os valores usados pelo programa.",
        fetchExp: "O fetch é uma operação do ciclo da máquina que recupera um endereço da memória.",
        decodeExp: "O decode é uma operação do ciclo da máquina que decodifica um endereço da memória, transformando-o em uma instrução.",
        executeExp: "O execute é uma operação do ciclo da máquina que executa uma instrução decodificada da memória.",
        writebackExp: "O writeback é uma operação do ciclo da máquina que escreve dados nos registradores.",
        cacheExp: "A memória Cache é uma memória de leitura rápida e espaço menor se comparada com a RAM, o processador guarda instruções que poderão ser utilizadas nela para diminuir o tempo de busca de endereços.",
        pipelineExp: "O pipelining de instrução é a paralelização dos ciclos de máquina, no qual todos executarão ao mesmo tempo assim que estiverem disponíveis.",
    }
}
