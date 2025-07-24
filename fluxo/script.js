// Configuração inicial do Mermaid
mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
    },
    themeVariables: {
        primaryColor: '#4f46e5',
        primaryTextColor: '#1f2937',
        primaryBorderColor: '#4f46e5',
        lineColor: '#6b7280',
        secondaryColor: '#f8fafc',
        tertiaryColor: '#f3f4f6'
    }
});

// Variáveis globais
let currentDiagram = '';
let parser = null;
let stepExecutor = null;
let isStepByStepMode = false;
let parseResult = null;

// Variáveis de controle de zoom
let currentZoom = 1.0;

// Variável de controle do console flip
let isShowingSyntax = true; // Iniciar mostrando sintaxe
let isInitializationComplete = false; // Controlar quando a inicialização terminou

// Variável do CodeMirror
let editorInstance = null;

// Elementos DOM - inicializados após DOM carregar
let diagramContainer, errorDisplay, executeAllBtn, executeStepBtn, resetBtn;
let nextStepBtn, prevStepBtn, stepControls, stepCounter, stepDisplay, variableInputs, zoomInBtn, zoomOutBtn, fitDiagramBtn;
let consoleOutput, currentStepInfo, exampleSelector, flipConsoleBtn, consoleTitle, syntaxHelp, toggleConsoleBtn;

// Flag para evitar renderização múltipla
let isRendering = false;

// Exemplos predefinidos
const examples = {
    basico: {
        nome: "Básico - Sequência Simples",
        codigo: `flowchart TD
    A[Início] --> B[/Ler nome\\]
    B --> C[\\Mostrar 'Olá, ' + nome/]
    C --> D[Fim]`
    },
    decisao: {
        nome: "Intermediário - Com Decisão",
        codigo: `flowchart TD
    A[Início] --> B[/Ler idade\\]
    B --> C{idade >= 18}
    C -->|Sim| D[\\Pode votar/]
    C -->|Não| E[\\Não pode votar/]
    D --> F[Fim]
    E --> F`
    },
    calculadora: {
        nome: "Avançado - Calculadora",
        codigo: `flowchart TD
    A[Início] --> B[/Ler num1\\]
    B --> C[/Ler num2\\]
    C --> D[/Ler operacao\\]
    D --> E{operacao == '+'}
    E -->|Sim| F[resultado = num1 + num2]
    E -->|Não| G{operacao == '-'}
    G -->|Sim| H[resultado = num1 - num2]
    G -->|Não| I{operacao == '*'}
    I -->|Sim| J[resultado = num1 * num2]
    I -->|Não| K{operacao == '/'}
    K -->|Sim| L[resultado = num1 / num2]
    K -->|Não| M[Operação inválida]
    F --> N[\\Mostrar resultado/]
    H --> N
    J --> N
    L --> N
    M --> N
    N --> O[Fim]`
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🔧 DOM carregado, inicializando...');
    
    // Inicializar elementos DOM
    initializeElements();
    
    // Verificar se elementos essenciais existem
    if (!diagramContainer || !consoleOutput) {
        console.error('❌ Elementos essenciais não encontrados');
        return;
    }
    
    console.log('✅ Elementos DOM inicializados');
    
    // Inicializar parser (com fallback se falhar)
    try {
        parser = new UnifiedFlowchartParser();
        console.log('✅ Parser inicializado');
    } catch (error) {
        console.error('❌ Erro ao inicializar parser:', error);
        console.log('⚠️ Continuando sem parser - carregamento de exemplos ainda funcionará');
        parser = null;
    }
    
    // Configurar event listeners
    setupEventListeners();
    
    // Aguardar carregamento completo dos scripts
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Inicializar CodeMirror e interface
    await initializeCodeMirrorEditor();
    initializeInterface();
    
    // Marcar inicialização como completa
    isInitializationComplete = true;
    
    logToConsole('🚀 Fluxograma Interativo carregado com sucesso!');
    logToConsole('📋 Selecione um exemplo acima ou digite seu próprio fluxograma');
    logToConsole('💡 Dica: Use o botão "👣 Passo a Passo" para aprender como o algoritmo funciona');
});

// Inicializar elementos DOM
function initializeElements() {
    // editor removido - usando apenas CodeMirror
    diagramContainer = document.getElementById('mermaid-diagram');
    errorDisplay = document.getElementById('error-display');
    executeAllBtn = document.getElementById('execute-all');
    executeStepBtn = document.getElementById('execute-step');
    resetBtn = document.getElementById('reset');
    nextStepBtn = document.getElementById('next-step');
    prevStepBtn = document.getElementById('prev-step');
    stepCounter = document.getElementById('step-counter');
    stepDisplay = document.getElementById('step-display');
    variableInputs = document.getElementById('input-variables');
    consoleOutput = document.getElementById('console-output');
    currentStepInfo = document.getElementById('current-step-info');
    exampleSelector = document.getElementById('example-selector');
    zoomInBtn = document.getElementById('zoom-in');
    zoomOutBtn = document.getElementById('zoom-out');
    fitDiagramBtn = document.getElementById('fit-diagram');
    flipConsoleBtn = document.getElementById('flip-console');
    consoleTitle = document.getElementById('console-title');
    syntaxHelp = document.getElementById('syntax-help');
    toggleConsoleBtn = document.getElementById('toggle-console');
}

// Função para debounce (evitar muitas renderizações)
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Configurar event listeners
function setupEventListeners() {
    console.log('🔧 Configurando event listeners...');
    
    // Note: CodeMirror configura seus próprios listeners via handleEditorChange
    
    if (executeAllBtn) executeAllBtn.addEventListener('click', executeAll);
    if (executeStepBtn) executeStepBtn.addEventListener('click', executeStepByStep);
    if (resetBtn) resetBtn.addEventListener('click', resetExecution);
    if (nextStepBtn) nextStepBtn.addEventListener('click', executeNextStep);
    if (prevStepBtn) prevStepBtn.addEventListener('click', executePreviousStep);

    if (zoomInBtn) zoomInBtn.addEventListener('click', zoomIn);
    if (zoomOutBtn) zoomOutBtn.addEventListener('click', zoomOut);
    if (fitDiagramBtn) fitDiagramBtn.addEventListener('click', fitDiagram);
    if (flipConsoleBtn) flipConsoleBtn.addEventListener('click', toggleConsoleView);
    if (toggleConsoleBtn) toggleConsoleBtn.addEventListener('click', toggleConsoleCollapse);
    
    // Click no header também deve alternar o console
    const consoleHeader = document.getElementById('console-header');
    if (consoleHeader) {
        consoleHeader.addEventListener('click', function(e) {
            // Não alternar se clicou em um botão
            if (!e.target.closest('button')) {
                toggleConsoleCollapse();
            }
        });
    }
    
    // Event listener para carregamento automático de exemplos
    if (exampleSelector) {
        exampleSelector.addEventListener('change', function() {
            console.log('🔧 Seletor mudou para:', this.value);
            const selectedValue = this.value;
            if (selectedValue) {
                console.log('📋 Carregando exemplo:', selectedValue);
                loadExample(selectedValue);
            } else {
                console.log('🔄 Resetando título');
                resetarTitulo();
            }
        });
        console.log('✅ Event listener do seletor configurado');
    }
    
    console.log('✅ Event listeners configurados');
}

// Inicializar CodeMirror
async function initializeCodeMirrorEditor() {
    console.log('🔧 Inicializando CodeMirror...');
    
    try {
        // Aguardar carregamento do SimpleMermaidEditor
        if (!window.simpleMermaidEditor) {
            throw new Error('SimpleMermaidEditor não carregado');
        }
        
        // Inicializar editor
        const success = await window.simpleMermaidEditor.initialize('codemirror-container');
        
        if (success) {
            editorInstance = window.simpleMermaidEditor;
            
            console.log('🔍 editorInstance atribuído:', !!editorInstance);
            console.log('🔍 editorInstance.setValue:', typeof editorInstance?.setValue);
            console.log('🔍 editorInstance.getValue:', typeof editorInstance?.getValue);
            console.log('🔍 editorInstance.editor:', !!editorInstance?.editor);
            
            // Configurar callback de mudança
            editorInstance.editor.on('change', handleEditorChange);
            
            console.log('✅ CodeMirror inicializado!');
            logToConsole('🎨 Editor CodeMirror ativo');
        } else {
            throw new Error('Falha na inicialização do CodeMirror');
        }
        
    } catch (error) {
        console.error('❌ ERRO FATAL ao inicializar CodeMirror:', error);
        throw error; // Sistema para se não conseguir carregar
    }
}

// Função para lidar com mudanças no editor (CodeMirror ou textarea)
function handleEditorChange(value) {
    // Usar debounce para evitar renderizações excessivas
    if (handleEditorChange.timeout) {
        clearTimeout(handleEditorChange.timeout);
    }
    
    handleEditorChange.timeout = setTimeout(() => {
        renderDiagram();
    }, 800);
}

// Funções de compatibilidade para obter e definir valor do editor
function getEditorValue() {
    if (editorInstance) {
        return editorInstance.getValue();
    }
    return '';
}

function setEditorValue(value) {
    console.log(`🔧 setEditorValue chamado com:`, value.substring(0, 50) + '...');
    console.log(`🔍 editorInstance:`, !!editorInstance);
    console.log(`🔍 editorInstance.setValue:`, typeof editorInstance?.setValue);
    
    if (editorInstance) {
        try {
            editorInstance.setValue(value);
            console.log(`✅ setValue executado com sucesso`);
        } catch (error) {
            console.error(`❌ Erro ao executar setValue:`, error);
        }
    } else {
        console.error(`❌ editorInstance não disponível`);
    }
}

function focusEditor() {
    if (editorInstance) {
        editorInstance.focus();
    }
}

// Inicializar interface
function initializeInterface() {
    console.log('🔧 Inicializando interface...');
    
    // Configurar editor inicial
    if (editorInstance) {
        editorInstance.setValue('');
        console.log('🎨 CodeMirror configurado');
    }
    
    // Mostrar mensagem no diagrama
    if (diagramContainer) {
        diagramContainer.innerHTML = '<div style="text-align: center; color: #6b7280; padding: 50px; font-size: 1.1rem;">\n            📈 Selecione um exemplo para começar<br>\n            <span style="font-size: 0.9rem; opacity: 0.8;">ou digite seu próprio fluxograma no editor</span>\n        </div>';
    }
    
    // Garantir que o estado inicial seja sintaxe
    initializeConsoleState();
    
    // Inicializar console como colapsado
    initializeConsoleCollapse();
    
    // Definir estado inicial como "não iniciado"
    setButtonStates('not-started');
    
    console.log('✅ Interface inicializada');
}

// Inicializar estado do console para sintaxe
function initializeConsoleState() {
    if (flipConsoleBtn && consoleTitle && consoleOutput && syntaxHelp) {
        // Forçar estado inicial: sintaxe visível
        isShowingSyntax = true;
        consoleTitle.textContent = 'Sintaxe para Criação de Fluxogramas';
        flipConsoleBtn.textContent = '💼';
        flipConsoleBtn.title = 'Voltar para o console de saída';
        
        consoleOutput.style.display = 'none';
        syntaxHelp.style.display = 'block';
        
        console.log('✅ Estado inicial definido: Sintaxe visível');
    } else {
        console.log('⚠️ Elementos do console flip não encontrados durante inicialização');
    }
}

// Inicializar console como colapsado
function initializeConsoleCollapse() {
    const outputConsole = document.querySelector('.output-console');
    const consoleContent = document.getElementById('console-content');
    const toggleBtn = document.getElementById('toggle-console');
    
    if (outputConsole && consoleContent && toggleBtn) {
        // Estado inicial: colapsado
        consoleContent.style.display = 'none';
        outputConsole.classList.add('collapsed');
        toggleBtn.textContent = '▲';
        toggleBtn.classList.add('rotated');
        toggleBtn.title = 'Expandir Console';
        
        console.log('✅ Console inicializado como colapsado');
    } else {
        console.log('⚠️ Elementos do console collapse não encontrados durante inicialização');
    }
}

// Carregar exemplo específico
function loadExample(exampleKey) {
    console.log(`📋 Carregando exemplo: ${exampleKey}`);
    console.log(`🔍 EditorInstance disponível:`, !!editorInstance);
    console.log(`🔍 Examples disponíveis:`, Object.keys(examples));
    
    const example = examples[exampleKey];
    if (!example) {
        console.error(`Exemplo '${exampleKey}' não encontrado`);
        return;
    }
    
    console.log(`📋 Exemplo encontrado:`, example.nome);
    console.log(`📝 Código a ser carregado:`, example.codigo.substring(0, 100) + '...');
    
    if (!editorInstance) {
        console.error('Editor não disponível');
        return;
    }
    
    console.log(`🔧 Chamando setEditorValue...`);
    // Carregar código no editor
    setEditorValue(example.codigo);
    
    logToConsole(`📋 Exemplo carregado: ${example.nome}`);
    
    // Mostrar indicação visual do exemplo carregado
    mostrarExemploCarregado(example.nome);
    
    // Renderizar após um pequeno delay
    setTimeout(() => {
        renderDiagram();
    }, 300);
}

// Função para mostrar qual exemplo está carregado
function mostrarExemploCarregado(nomeExemplo) {
    // Atualizar o título do painel do diagrama
    const panelHeader = document.querySelector('.diagram-panel .panel-header h2');
    if (panelHeader) {
        panelHeader.textContent = `Visualização do Fluxograma: ${nomeExemplo}`;
        logToConsole(`🔍 Título atualizado: ${nomeExemplo}`);
    }
}

// Função para resetar o título
function resetarTitulo() {
    const panelHeader = document.querySelector('.diagram-panel .panel-header h2');
    if (panelHeader) {
        panelHeader.textContent = 'Visualização do Fluxograma';
        logToConsole('🔄 Título resetado');
    }
}

// Renderizar diagrama Mermaid
async function renderDiagram() {
    // Evitar renderização múltipla simultânea
    if (isRendering) {
        console.log('⚠️ Renderização já em progresso, ignorando...');
        return;
    }
    
    if (!editorInstance || !diagramContainer) {
        console.error('❌ Elementos necessários não disponíveis para renderização');
        return;
    }
    
    isRendering = true;
    console.log('🔧 Iniciando renderização...');
    
    const code = getEditorValue().trim();
    
    if (!code) {
        diagramContainer.innerHTML = '<div style="text-align: center; color: #6b7280; padding: 50px;">Digite seu fluxograma no editor</div>';
        hideError();
        isRendering = false;
        
        // Se não há código e nenhum exemplo selecionado, resetar título
        if (!exampleSelector || exampleSelector.value === '') {
            resetarTitulo();
        }
        
        // Voltar ao estado "não iniciado" quando não há código
        setButtonStates('not-started');
        
        return;
    }
    
    try {
        console.log('🔧 Limpando container...');
        diagramContainer.innerHTML = '';
        
        console.log('🔧 Gerando ID único...');
        const diagramId = 'diagram-' + Date.now();
        
        console.log('🔧 Chamando mermaid.render()...');
        const { svg } = await mermaid.render(diagramId, code);
        
        console.log('✅ SVG gerado, inserindo no DOM...');
        diagramContainer.innerHTML = svg;

        // Resetar zoom e posição para o novo diagrama
        resetZoom(true);
        
        // Salvar diagrama atual
        currentDiagram = code;
        
        console.log('🔧 Testando parser...');
        // Parsear passos para execução - COM PROTEÇÃO
        try {
            if (parser && typeof parser.parse === 'function') {
                parseResult = parser.parse(code);
                logToConsole(`✅ Fluxograma parseado: ${parseResult.nodes.length} nós, ${parseResult.connections.length} conexões`);
                
                // Preparar campos de entrada
                prepareInputVariables();
            } else {
                console.log('⚠️ Parser não disponível - apenas renderizando diagrama');
                logToConsole('⚠️ Parser não disponível - funcionalidade de execução limitada');
                parseResult = null;
            }
        } catch (parseError) {
            console.warn('⚠️ Aviso no parsing:', parseError.message);
            logToConsole(`⚠️ Aviso no parsing: ${parseError.message}`);
            parseResult = null;
        }
        
        hideError();
        console.log('✅ Renderização concluída com sucesso');
        
        // Ativar botões quando há um fluxograma carregado
        setButtonStates('normal');
        
    } catch (error) {
        console.error('❌ Erro na renderização:', error);
        showError('Erro na sintaxe do fluxograma: ' + error.message);
        logToConsole(`❌ Erro de sintaxe: ${error.message}`);
    } finally {
        isRendering = false;
    }
}

// Mostrar erro
function showError(message) {
    if (errorDisplay) {
        errorDisplay.textContent = message;
        errorDisplay.style.display = 'block';
    }
    if (diagramContainer) {
        diagramContainer.style.display = 'none';
    }
}

// Esconder erro
function hideError() {
    if (errorDisplay) {
        errorDisplay.style.display = 'none';
    }
    if (diagramContainer) {
        diagramContainer.style.display = 'flex';
    }
}

// Preparar campos de entrada de variáveis
// Função para limpar texto da label
function cleanLabelText(text) {
    return text
        .replace(/^[\\\/]+/, '')  // Remove barras do início
        .replace(/[\\\/]+$/, '')  // Remove barras do final
        .replace(/^\[/, '')       // Remove [ do início
        .replace(/\]$/, '')       // Remove ] do final
        .trim();
}

function prepareInputVariables() {
    if (!variableInputs || !parseResult) {
        console.log('⚠️ Não é possível preparar campos de entrada');
        return;
    }
    
    console.log('🔧 Preparando campos de entrada...');
    
    const inputNodes = parseResult.nodes.filter(node => node.type === 'input');
    logToConsole(`🔍 Encontrados ${inputNodes.length} nós de entrada`);
    
    variableInputs.innerHTML = '';
    
    if (inputNodes.length === 0) {
        variableInputs.innerHTML = '<p style="color: #6b7280; font-style: italic; text-align: center; padding: 20px;">Nenhuma variável de entrada necessária</p>';
        return;
    }
    
    const title = document.createElement('h3');
    title.textContent = 'Variáveis de Entrada';
    variableInputs.appendChild(title);
    
    inputNodes.forEach(node => {
        // Utilizar o método do parser para consistência
        if (!parser) {
            console.error("Parser não inicializado, impossível extrair nome da variável.");
            return;
        }
        const varName = parser.extractVariableName(node.text);
        const cleanLabel = cleanLabelText(node.text);
        logToConsole(`⚙️ Criando campo para variável: ${varName}`);
        
        const inputDiv = document.createElement('div');
        inputDiv.className = 'variable-input';
        
        inputDiv.innerHTML = `
            <label for="var-${varName}">${cleanLabel}:</label>
            <input type="text" 
                   id="var-${varName}" 
                   data-variable="${varName}" 
                   placeholder="Digite o valor"
                   autocomplete="off">
        `;
        
        variableInputs.appendChild(inputDiv);
    });
    
    logToConsole(`✅ ${inputNodes.length} campos de entrada criados`);
    console.log('✅ Campos de entrada preparados');
}

// Executar fluxograma completo
async function executeAll() {
    if (!currentDiagram || !parseResult) {
        logToConsole('❌ Nenhum fluxograma válido para executar');
        return;
    }
    
    console.log('🚀 Executando fluxograma completo...');
    
    try {
        // Limpar resultados anteriores
        if (currentStepInfo) currentStepInfo.textContent = 'Iniciando execução...';
        
        logToConsole('🚀 === EXECUÇÃO COMPLETA ===');
        
        // NÃO preparar campos novamente se já existem valores
        const existingFields = document.querySelectorAll('#input-variables input');
        const hasValues = Array.from(existingFields).some(field => field.value.trim());
        
        if (!hasValues) {
            // Só preparar campos se não há valores
            prepareInputVariables();
            logToConsole('📝 Campos de entrada preparados');
        } else {
            logToConsole('📝 Mantendo valores existentes nos campos');
        }
        
        // Inicializar executor passo-a-passo
        stepExecutor = new StepByStepExecutor(parseResult, logToConsole);
        
        // Iniciar execução
        const started = await stepExecutor.start();
        
        if (!started) {
            logToConsole('❌ Falha ao iniciar execução');
            return;
        }
        
        // Executar em modo contínuo (velocidade total)
        await stepExecutor.runFullSpeed();
        
        // Atualizar interface com resultado final (sempre mostrar no painel de status)
        updateCurrentStepInfo();
        
        logToConsole('✅ Execução completa finalizada');
        
    } catch (error) {
        logToConsole(`❌ Erro na execução: ${error.message}`);
        console.error('Erro na execução:', error);
    }
}

// Executar passo-a-passo
async function executeStepByStep() {
    if (!currentDiagram || !parseResult) {
        logToConsole('❌ Nenhum fluxograma válido para executar');
        return;
    }
    
    console.log('👣 Iniciando execução passo-a-passo...');
    
    try {
        // Limpar resultados anteriores
        if (currentStepInfo) currentStepInfo.textContent = 'Preparando execução passo-a-passo...';
        
        // NÃO preparar campos novamente se já existem valores
        const existingFields = document.querySelectorAll('#input-variables input');
        const hasValues = Array.from(existingFields).some(field => field.value.trim());
        
        if (!hasValues) {
            // Só preparar campos se não há valores
            prepareInputVariables();
            logToConsole('📝 Campos de entrada preparados');
        } else {
            logToConsole('📝 Mantendo valores existentes nos campos');
        }
        
        // Inicializar executor passo-a-passo
        stepExecutor = new StepByStepExecutor(parseResult, logToConsole);
        
        // Iniciar execução
        const started = await stepExecutor.start();
        
        if (started) {
            isStepByStepMode = true;
            setButtonStates('step-by-step');
            updateStepCounter();
            updateCurrentStepInfo();
            
            logToConsole('👣 Modo passo-a-passo ativado');
            logToConsole('💡 Use o botão "Próximo Passo" para continuar');
        } else {
            logToConsole('❌ Falha ao iniciar execução passo-a-passo');
        }
        
    } catch (error) {
        logToConsole(`❌ Erro ao iniciar execução passo-a-passo: ${error.message}`);
        console.error('Erro passo-a-passo:', error);
    }
}

// Executar próximo passo
async function executeNextStep() {
    if (!stepExecutor || !isStepByStepMode) return;
    
    try {
        const success = await stepExecutor.executeNextStep();
        
        if (success) {
            updateStepCounter();
            updateCurrentStepInfo();
        } else {
            logToConsole('🏁 Execução passo-a-passo finalizada');
            isStepByStepMode = false;
            setButtonStates('normal');
        }
        
    } catch (error) {
        logToConsole(`❌ Erro no próximo passo: ${error.message}`);
    }
}

// Executar passo anterior
async function executePreviousStep() {
    if (!stepExecutor || !isStepByStepMode) return;
    
    try {
        const success = await stepExecutor.executePreviousStep();
        
        if (success) {
            updateStepCounter();
            updateCurrentStepInfo();
        }
        
    } catch (error) {
        logToConsole(`❌ Erro no passo anterior: ${error.message}`);
    }
}

// Resetar execução
function resetExecution() {
    console.log('🔄 Resetando execução...');
    
    if (stepExecutor) {
        stepExecutor.reset();
        stepExecutor = null;
    }
    
    // 🧹 LIMPAR console se estiver visível (dar controle ao usuário)
    if (!isShowingSyntax && consoleOutput) {
        consoleOutput.textContent = '';
        // Adicionar mensagem de reset no console
        const timestamp = new Date().toLocaleTimeString();
        const resetMessage = `[${timestamp}] 🔄 === CONSOLE LIMPO PELO RESET ===\n`;
        consoleOutput.textContent = resetMessage;
    }
    
    isStepByStepMode = false;
    setButtonStates('normal');
    
    if (variableInputs) variableInputs.innerHTML = '';
    if (currentStepInfo) currentStepInfo.textContent = 'Pronto para execução';
    
    logToConsole('🔄 Execução resetada - ready para nova execução!');
}

// Atualizar contador de passos
function updateStepCounter() {
    if (stepCounter && stepExecutor && isStepByStepMode) {
        const current = stepExecutor.getCurrentStepNumber();
        const total = stepExecutor.getTotalSteps();
        stepCounter.textContent = `${current}/${total}`;
    }
}

// Atualizar informações do passo atual
function updateCurrentStepInfo() {
    if (currentStepInfo && stepExecutor) {
        if (isStepByStepMode) {
            // Modo passo-a-passo: mostrar informação do passo atual
            currentStepInfo.textContent = stepExecutor.getCurrentStepInfo();
        } else {
            // Modo execução completa: mostrar resultado final
            const lastOutput = stepExecutor.getLastOutputResult();
            if (lastOutput !== null) {
                currentStepInfo.textContent = `🎆 Resultado: ${lastOutput}`;
            } else {
                currentStepInfo.textContent = '✅ Execução completa finalizada';
            }
        }
    }
}

// Configurar estados dos botões
function setButtonStates(state) {
    if (!executeAllBtn || !executeStepBtn || !resetBtn || !nextStepBtn || !prevStepBtn) return;
    
    // Remover todas as classes de estado primeiro
    const allButtons = [executeAllBtn, executeStepBtn, resetBtn, nextStepBtn, prevStepBtn];
    allButtons.forEach(btn => {
        btn.classList.remove('not-started');
    });
    
    if (stepDisplay) {
        stepDisplay.classList.remove('not-started', 'active');
    }
    
    switch (state) {
        case 'not-started':
            executeAllBtn.disabled = true;
            executeStepBtn.disabled = true;
            resetBtn.disabled = true;
            nextStepBtn.disabled = true;
            prevStepBtn.disabled = true;
            
            // Aplicar visual cinza a todos os botões
            allButtons.forEach(btn => {
                btn.classList.add('not-started');
            });
            
            if (stepCounter && stepDisplay) {
                stepCounter.textContent = '-/-';
                stepDisplay.classList.add('not-started');
            }
            break;
            
        case 'normal':
            executeAllBtn.disabled = false;
            executeStepBtn.disabled = false;
            resetBtn.disabled = false;
            nextStepBtn.disabled = true;
            prevStepBtn.disabled = true;
            if (stepCounter && stepDisplay) {
                stepCounter.textContent = '-/-';
                // stepDisplay volta ao estado padrão (sem classes)
            }
            break;
            
        case 'step-by-step':
            executeAllBtn.disabled = true;
            executeStepBtn.disabled = true;
            resetBtn.disabled = false;
            nextStepBtn.disabled = false;
            prevStepBtn.disabled = false;
            if (stepCounter && stepDisplay) {
                stepDisplay.classList.add('active');
            }
            break;
    }
}

// --- Funções de Controle do Diagrama (Zoom) ---

function updateDiagramTransform() {
    const svg = diagramContainer.querySelector('svg');
    if (svg) {
        svg.style.transform = `scale(${currentZoom})`;
        svg.style.transformOrigin = 'center';
    }
}

function zoomIn() {
    currentZoom += 0.1;
    updateDiagramTransform();
    logToConsole('🔍 Zoom aumentado');
}

function zoomOut() {
    if (currentZoom > 0.2) { // Prevenir zoom excessivo
        currentZoom -= 0.1;
        updateDiagramTransform();
        logToConsole('🔍 Zoom diminuído');
    }
}

function fitDiagram() {
    resetZoom(); // Chama a função com log
}

function resetZoom(silent = false) {
    const svg = diagramContainer.querySelector('svg');
    const container = diagramContainer.parentElement; // .diagram-container

    if (!svg || !container) {
        if (!silent) {
            logToConsole('⚠️ Não foi possível ajustar o diagrama (SVG ou container não encontrado).');
        }
        return;
    }

    // Get container dimensions (excluding padding for accuracy)
    const containerStyle = window.getComputedStyle(container);
    const containerWidth = container.clientWidth - parseFloat(containerStyle.paddingLeft) - parseFloat(containerStyle.paddingRight);
    const containerHeight = container.clientHeight - parseFloat(containerStyle.paddingTop) - parseFloat(containerStyle.paddingBottom);

    // Get SVG's real dimensions from its bounding box.
    const svgBBox = svg.getBBox();
    const svgWidth = svgBBox.width;
    const svgHeight = svgBBox.height;

    if (svgWidth <= 0 || svgHeight <= 0) {
        // If dimensions are invalid, just reset to 100% scale
        currentZoom = 1.0;
    } else {
        // Calculate scale ratios
        const scaleX = containerWidth / svgWidth;
        const scaleY = containerHeight / svgHeight;

        // Use the smaller ratio to ensure the whole diagram fits.
        // Cap at 1.0 to avoid zooming in on small diagrams.
        currentZoom = Math.min(scaleX, scaleY, 1.0);
    }

    // Apply the new zoom level and reset scroll
    updateDiagramTransform();
    container.scrollTop = 0;
    container.scrollLeft = 0;

    if (!silent) {
        logToConsole(`📐 Diagrama ajustado à tela (zoom: ${currentZoom.toFixed(2)}x)`);
    }
}

// Alternar entre console e ajuda de sintaxe
function toggleConsoleView() {
    if (!flipConsoleBtn || !consoleTitle || !consoleOutput || !syntaxHelp) {
        return;
    }
    
    isShowingSyntax = !isShowingSyntax;
    
    if (isShowingSyntax) {
        // Mostrar sintaxe
        consoleTitle.textContent = 'Sintaxe para Criação de Fluxogramas';
        flipConsoleBtn.textContent = '💼';
        flipConsoleBtn.title = 'Voltar para o console de saída';
        
        consoleOutput.style.display = 'none';
        syntaxHelp.style.display = 'block';
    } else {
        // Mostrar console - LIMPAR LOGS ANTERIORES
        consoleTitle.textContent = 'Console de Saída';
        flipConsoleBtn.textContent = '📖';
        flipConsoleBtn.title = 'Alternar entre console e sintaxe';
        
        // 🧹 LIMPAR console antes de mostrar
        consoleOutput.textContent = '';
        
        consoleOutput.style.display = 'block';
        syntaxHelp.style.display = 'none';
        
        // Log inicial indicando início da sessão
        const timestamp = new Date().toLocaleTimeString();
        const sessionStart = `[${timestamp}] 🚀 === NOVA SESSÃO DE LOGS INICIADA ===\n`;
        consoleOutput.textContent = sessionStart;
    }
}

// Toggle collapse/expand do console
function toggleConsoleCollapse() {
    const outputConsole = document.querySelector('.output-console');
    const consoleContent = document.getElementById('console-content');
    const toggleBtn = document.getElementById('toggle-console');
    
    if (!outputConsole || !consoleContent || !toggleBtn) {
        console.error('Elementos do console não encontrados');
        return;
    }
    
    const isCollapsed = consoleContent.style.display === 'none';
    
    if (isCollapsed) {
        // Expandir
        consoleContent.style.display = 'block';
        outputConsole.classList.remove('collapsed');
        outputConsole.classList.add('expanded');
        toggleBtn.textContent = '▼';
        toggleBtn.classList.remove('rotated');
        toggleBtn.title = 'Colapsar Console';
    } else {
        // Colapsar
        consoleContent.style.display = 'none';
        outputConsole.classList.remove('expanded');
        outputConsole.classList.add('collapsed');
        toggleBtn.textContent = '▲';
        toggleBtn.classList.add('rotated');
        toggleBtn.title = 'Expandir Console';
    }
}

// Log para console
function logToConsole(message) {
    // 🔍 NOVA LÓGICA: SÓ registrar logs quando console estiver visível
    // SEM ALTERNÂNCIA AUTOMÁTICA - console controlado apenas pelo usuário
    if (isShowingSyntax) {
        // Se está mostrando sintaxe, NÃO registrar logs e NÃO alternar automaticamente
        return; // Usuário decide quando ver console via botão flip
    }
    
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    
    // USAR APENAS o console original, NÃO o redefinido
    if (window.originalConsoleLog) {
        window.originalConsoleLog(logEntry);
    }
    
    if (consoleOutput) {
        consoleOutput.textContent += logEntry + '\n';
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
        
        // Limitar tamanho do console
        const lines = consoleOutput.textContent.split('\n');
        if (lines.length > 100) {
            consoleOutput.textContent = lines.slice(-100).join('\n');
        }
    }
}

// Substituir console.log e prompt para capturar execução
(function setupConsoleCapture() {
    // SALVAR o console.log original ANTES de redefinir
    window.originalConsoleLog = console.log;
    
    window.console.log = function(...args) {
        logToConsole(args.join(' '));
        // NÃO chamar originalLog aqui para evitar duplicação
    };
    
    window.prompt = function(message) {
        if (!parser) {
            console.error("Parser não inicializado, impossível extrair nome da variável para o prompt.");
            return '0'; // Retornar valor padrão se o parser não estiver disponível
        }
        const varName = parser.extractVariableName(message);
        const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
        
        if (inputElement && inputElement.value.trim()) {
            return inputElement.value.trim();
        }
        
        logToConsole(`⚠️ Prompt: ${message} (usando valor padrão)`);
        return '0';
    };    
})();

// === SEÇÃO REMOVIDA: NUMERAÇÃO DE LINHAS SIMPLES ===
// CodeMirror já possui numeração nativa

// === SEÇÃO REMOVIDA: SYNTAX HIGHLIGHTING ANTIGO ===
// CodeMirror já possui highlighting nativo

// === SEÇÃO REMOVIDA: FUNÇÃO DE TESTE MANUAL ===
// Não mais necessária com CodeMirror
