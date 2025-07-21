/**
 * Executor Passo-a-Passo para Fluxogramas
 * Permite execução controlada e educativa de algoritmos
 */
class StepByStepExecutor {
    constructor(parseResult, logFunction) {
        this.parseResult = parseResult;
        this.log = logFunction; // Função para logar no console da UI
        this.variables = {};
        this.currentStepId = null;
        this.executionStack = [];
        this.isRunning = false;
        this.isPaused = false;
        this.userInputs = {};
        this.decisionResults = {};
        this.outputResults = []; // Armazenar resultados de saída
        this.lastOutputResult = null; // Último resultado de saída para exibição
    }

    /**
     * Iniciar execução passo-a-passo
     */
    async start() {
        if (this.isRunning) {
            this.log('Execução já está em andamento');
            return false;
        }

        this.reset();
        this.log('=== Iniciando execução passo-a-passo ===');

        // Encontrar o nó inicial
        const startNode = this.parseResult.nodes.find(n => n.type === 'start');
        if (!startNode) {
            this.log('Erro: Nó de início não encontrado.');
            return false;
        }

        this.isRunning = true;
        this.currentStepId = startNode.id;
        
        // NÃO preparar variáveis de entrada aqui - já foi feito no script principal
        
        // Destacar primeiro nó
        this.highlightCurrentNode();
        
        return true;
    }

    /**
     * Resetar estado do executor
     */
    reset() {
        this.currentStepId = null;
        this.variables = {};
        this.executionStack = [];
        this.isRunning = false;
        this.isPaused = false;
        this.userInputs = {};
        this.decisionResults = {};
        this.outputResults = [];
        this.lastOutputResult = null;
        this.clearHighlights();
    }

    /**
     * Executar próximo passo - COM DEBUG DETALHADO
     */
    async executeNextStep() {
        this.log(`=== INICIANDO executeNextStep ===`);
        this.log(`isRunning: ${this.isRunning}`);
        this.log(`currentStepId: ${this.currentStepId}`);
        
        if (!this.isRunning || !this.currentStepId) {
            this.log('Execução finalizada ou não iniciada.');
            this.isRunning = false;
            return false;
        }

        const currentNode = this.parseResult.nodeMap.get(this.currentStepId);
        this.log(`Nó atual encontrado: ${currentNode ? 'SIM' : 'NÃO'}`);
        
        if (!currentNode) {
            this.log(`Erro: Nó ${this.currentStepId} não encontrado.`);
            this.log(`Nós disponíveis: ${Array.from(this.parseResult.nodeMap.keys()).join(', ')}`);
            return false;
        }

        this.log(`--- Executando: ${currentNode.text} (Tipo: ${currentNode.type}) ---`);
        
        // Executar nó baseado no tipo
        const executionResult = await this.executeNode(currentNode);
        this.log(`Resultado da execução: success=${executionResult.success}, decision=${executionResult.decision}`);
        
        if (!executionResult.success) {
            this.log('Execução interrompida por erro.');
            this.isRunning = false;
            return false;
        }

        // Salvar estado atual antes de avançar (para função "anterior")
        this.executionStack.push({
            stepId: this.currentStepId,
            variables: {...this.variables}
        });
        this.log(`Estado salvo. Pilha tem ${this.executionStack.length} itens`);
        this.log(`Variáveis atuais: ${JSON.stringify(this.variables)}`);

        // Encontrar o próximo nó
        this.log(`Chamando findNextNodeId(${this.currentStepId}, ${executionResult.decision})`);
        
        // Debug das conexões disponíveis
        const availableConnections = this.parseResult.connections.filter(c => c.from === this.currentStepId);
        this.log(`Conexões disponíveis de ${this.currentStepId}: ${JSON.stringify(availableConnections)}`);
        
        const nextNodeId = this.findNextNodeId(this.currentStepId, executionResult.decision);
        this.log(`Próximo nó retornado: ${nextNodeId}`);

        if (!nextNodeId) {
            this.log('\n=== Execução finalizada ===');
            this.isRunning = false;
            this.clearHighlights();
            return false;
        }

        // Avançar para o próximo nó
        this.log(`Mudando currentStepId de ${this.currentStepId} para ${nextNodeId}`);
        this.currentStepId = nextNodeId;

        // Destacar próximo nó
        this.highlightCurrentNode();
        this.log(`=== executeNextStep FINALIZADO COM SUCESSO ===`);
        return true;
    }

    /**
     * Executa o fluxograma continuamente, pausando apenas em nós de entrada.
     * Projetado para ser usado pelo botão "Executar Tudo".
     */
    async runFullSpeed() {
        if (!this.isRunning || !this.currentStepId) {
            this.log('Execução não iniciada ou já finalizada.');
            return;
        }

        this.log('Executando em modo contínuo...');

        // Loop principal de execução
        while (this.isRunning && this.currentStepId) {
            const currentNode = this.parseResult.nodeMap.get(this.currentStepId);

            // Se o nó atual for de entrada, verificar se há valor
            if (currentNode && currentNode.type === 'input') {
                const varName = this.extractVariableName(currentNode.text);
                const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
                
                if (!inputElement || !inputElement.value.trim()) {
                    this.log(`Pausando: Necessário valor para "${currentNode.text}"`);
                    this.highlightCurrentNode();
                    return; // Pausa e espera input do usuário
                }
            }

            // Executar o passo atual
            const canContinue = await this.executeNextStep();
            if (!canContinue) {
                return; // Execução terminou ou erro
            }

            // Pequeno delay para visualização
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    /**
     * Executar passo anterior
     */
    async executePreviousStep() {
        if (this.executionStack.length === 0) {
            this.log('Já está no primeiro passo, não é possível voltar.');
            return false;
        }

        // Restaurar estado anterior
        const previousState = this.executionStack.pop();
        this.currentStepId = previousState.stepId;
        this.variables = {...previousState.variables};

        this.log(`Voltando para: ${this.currentStepId}`);

        // Destacar nó atual
        this.highlightCurrentNode();

        return true;
    }

    /**
     * Executar um nó específico baseado no seu tipo
     * @param {Object} node - Nó a ser executado
     * @returns {Object} { success: boolean, decision?: boolean }
     */
    async executeNode(node) {
        switch (node.type) {
            case 'start':
                return this.executeStartNode(node);
            
            case 'input':
                return this.executeInputNode(node);
            
            case 'process':
                return this.executeProcessNode(node);
            
            case 'decision':
                return this.executeDecisionNode(node);
            
            case 'output':
                return this.executeOutputNode(node);
            
            case 'end':
                return this.executeEndNode(node);
            
            default:
                this.log(`Tipo de nó desconhecido: ${node.type}`);
                return { success: true };
        }
    }

    /**
     * Executar nó de início
     */
    executeStartNode(node) {
        this.log('Iniciando algoritmo...');
        return { success: true };
    }

    /**
     * Executar nó de entrada
     */
    executeInputNode(node) {
        this.log(`Executando nó de entrada: ${node.text}`);
        const varName = this.extractVariableName(node.text);
        this.log(`Variável extraída: ${varName}`);
        
        const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
        
        if (!inputElement) {
            this.log(`Erro: Campo de entrada para "${varName}" não encontrado`);
            return { success: false };
        }

        const value = inputElement.value.trim();
        this.log(`Valor lido do campo: "${value}"`);
        
        if (!value) {
            this.log(`Aviso: Valor para "${varName}" não fornecido. Usando 0.`);
            this.variables[varName] = 0;
        } else {
            // Tentar converter para número; se falhar, manter como string
            const numValue = parseFloat(value);
            if (!isNaN(numValue) && isFinite(numValue) && value === numValue.toString()) {
                this.variables[varName] = numValue;
            } else {
                this.variables[varName] = value; // Manter como string
            }
        }
        this.log(`Entrada processada: ${varName} = ${JSON.stringify(this.variables[varName])}`);
        return { success: true };
    }

    /**
     * Executar nó de processo
     */
    executeProcessNode(node) {
        this.log(`Processando: ${node.text}`);
        
        try {
            // Verificar se é uma atribuição (contém =)
            if (node.text.includes('=')) {
                const parts = node.text.split('=').map(p => p.trim());
                const varName = parts[0];
                const expression = parts[1];
                
                // Substituir variáveis na expressão
                const evaluatedExpression = this.substituteVariables(expression);
                
                // Executar a atribuição
                const context = this.variables;
                const result = new Function(...Object.keys(context), `return ${evaluatedExpression}`)(...Object.values(context));
                
                this.variables[varName] = result;
                this.log(`${varName} = ${result}`);
            } else {
                this.log(`Processo executado: ${node.text}`);
            }
            
            return { success: true };
        } catch (error) {
            this.log(`Erro ao executar processo: ${error.message}`);
            return { success: false };
        }
    }

    /**
     * Executar nó de decisão
     */
    executeDecisionNode(node) {
        this.log(`Executando nó de decisão: ${node.text}`);
        const condition = this.convertConditionToJS(node.text);
        this.log(`Condição convertida: ${condition}`);
        
        try {
            // Substituir variáveis na condição
            const evaluatedCondition = this.substituteVariables(condition);
            this.log(`Condição com variáveis substituídas: ${evaluatedCondition}`);
            this.log(`Variáveis disponíveis: ${JSON.stringify(this.variables)}`);
            
            // Usar eval diretamente para melhor compatibilidade com strings
            const result = eval(evaluatedCondition);
            
            this.decisionResults[node.id] = result;
            this.log(`Decisão: ${node.text}`);
            this.log(`Condição avaliada: ${evaluatedCondition} = ${result ? 'Verdadeiro' : 'Falso'}`);
            
            return { success: true, decision: result };
        } catch (error) {
            this.log(`Erro ao avaliar condição: ${error.message}`);
            this.log(`Condição original: "${node.text}"`);
            this.log(`Condição convertida: "${condition}"`);
            this.log(`Condição final: "${this.substituteVariables(condition)}"`);
            this.log(`Variáveis: ${JSON.stringify(this.variables)}`);
            return { success: false };
        }
    }

    /**
     * Executar nó de saída
     */
    executeOutputNode(node) {
        this.log(`Executando nó de saída: ${node.text}`);
        
        // Extrair o conteúdo a ser exibido, removendo sintaxe de trapézio e prefixos
        let outputContent = node.text
            .replace(/^(mostrar|escrever|exibir|output|print)\s+/i, '')  // Remove prefixos
            .replace(/^\[\\/, '')  // Remove [\  do início
            .replace(/\/\]$/, '')  // Remove /] do final
            .replace(/^\[\//, '')  // Remove [/  do início (entrada)
            .replace(/\\\]$/, '')  // Remove \] do final (entrada)
            .replace(/^\\/, '')   // Remove \ do início
            .replace(/\/$/, '')   // Remove / do final
            .trim();
        
        this.log(`Conteúdo extraído: "${outputContent}"`);
        
        // Substituir variáveis e mostrar
        const output = this.substituteVariables(outputContent);
        this.log(`Conteúdo com variáveis: "${output}"`);
        this.log(`Saída: ${output}`);
        
        // Armazenar resultado para exibição na interface
        this.lastOutputResult = output;
        this.outputResults.push({
            step: this.executionStack.length + 1,
            nodeText: node.text,
            result: output,
            timestamp: new Date().toLocaleTimeString()
        });
        
        return { success: true };
    }

    /**
     * Executar nó de fim
     */
    executeEndNode(node) {
        this.log('Algoritmo finalizado');
        return { success: true }; // Sucesso, mas não haverá próximo nó
    }

    /**
     * Preparar campos de entrada de variáveis
     */
    prepareInputVariables() {
        const inputNodes = this.parseResult.nodes.filter(node => node.type === 'input');
        // O container correto é 'input-variables'
        const variableInputsContainer = document.getElementById('input-variables');
        
        if (!variableInputsContainer) return;
        
        variableInputsContainer.innerHTML = '';
        
        if (inputNodes.length === 0) {
            variableInputsContainer.innerHTML = '<p style="color: #6b7280; font-style: italic; text-align: center; padding: 20px;">Nenhuma variável de entrada necessária</p>';
            return;
        }

        const title = document.createElement('h3');
        title.textContent = 'Variáveis de Entrada';
        variableInputsContainer.appendChild(title);

        for (const node of inputNodes) {
            const varName = this.extractVariableName(node.text);
            
            const inputDiv = document.createElement('div');
            inputDiv.className = 'variable-input';
            
            inputDiv.innerHTML = `
                <label for="var-${varName}">${node.text}:</label>
                <input type="text" 
                       id="var-${varName}" 
                       data-variable="${varName}" 
                       placeholder="Digite o valor"
                       autocomplete="off">
            `;
            
            variableInputsContainer.appendChild(inputDiv);
        }
    }

    /**
     * Destacar nó atual no diagrama
     */
    highlightCurrentNode() {
        // Limpar destaques anteriores
        this.clearHighlights();
        if (!this.currentStepId) return;
        
        // Adicionar destaque visual
        const diagramContainer = document.getElementById('mermaid-diagram');
        if (diagramContainer) {
            // Procurar por elementos que contêm o ID do nó
            const svgElements = diagramContainer.querySelectorAll(`[id*="${this.currentStepId}"]`);
            
            for (const element of svgElements) {
                if (element.id && element.id.includes(this.currentStepId)) {
                    // Verificar se é o elemento do nó e não uma conexão
                    // Elementos de nó geralmente têm classes como 'node' ou terminam com o ID exato
                    const isNodeElement = element.classList.contains('node') || 
                                        element.classList.contains('nodeLabel') ||
                                        element.id === this.currentStepId ||
                                        element.id.endsWith(`-${this.currentStepId}`) ||
                                        (element.tagName === 'g' && !element.id.includes('edge') && !element.id.includes('link'));
                    
                    if (isNodeElement) {
                        element.style.stroke = '#ff6b6b';
                        element.style.strokeWidth = '3px';
                        element.style.filter = 'drop-shadow(0 0 6px #ff6b6b)';
                        element.classList.add('current-step-highlight');
                    }
                }
            }
        }
    }

    /**
     * Limpar destaques visuais
     */
    clearHighlights() {
        const highlightedElements = document.querySelectorAll('.current-step-highlight');
        highlightedElements.forEach(element => {
            element.style.stroke = '';
            element.style.strokeWidth = '';
            element.style.filter = '';
            element.classList.remove('current-step-highlight');
        });
    }

    /**
     * Reverter último passo (simplificado)
     */
    revertLastStep() {
        // NOTA: Uma implementação completa de "passo anterior" exigiria
        // o armazenamento do estado (ex: valores de variáveis) a cada passo
        // e a restauração desse estado aqui. Por simplicidade, esta função
        // apenas reverte o ponteiro do passo, mas não o estado das variáveis.
        this.log('Estado revertido para o passo anterior');
    }

    /**
     * Extrair nome da variável do texto
     */
    extractVariableName(text) {
        const patterns = [
            /ler\s+(\w+)/i,
            /digite\s+(\w+)/i,
            /entrada\s+(\w+)/i,
            /input\s+(\w+)/i,
            /(\w+)$/ // Pega a última palavra como fallback
        ];

        for (const pattern of patterns) {
            const match = text.match(pattern);
            if (match) {
                return match[1].toLowerCase();
            }
        }

        return text.toLowerCase().replace(/\s/g, '_'); // Fallback mais seguro
    }

    /**
     * Substituir variáveis na expressão
     */
    substituteVariables(expression) {
        let result = expression;
        
        // Substituir variáveis conhecidas
        for (const [varName, value] of Object.entries(this.variables)) {
            const regex = new RegExp(`\\b${varName}\\b`, 'gi');
            
            // Se o valor é uma string, adicionar aspas; se é número, manter como está
            let valueToReplace;
            if (typeof value === 'string') {
                // Escapar aspas internas e envolver em aspas duplas
                valueToReplace = `"${value.replace(/"/g, '\\"')}"`;
            } else {
                valueToReplace = value;
            }
            
            result = result.replace(regex, valueToReplace);
        }

        return result;
    }

    /**
     * Converter condição do fluxograma para JavaScript - VERSÃO FINAL FUNCIONAL
     */
    convertConditionToJS(condition) {
        this.log(`Convertendo condição original: "${condition}"`);
        let jsCondition = condition.replace(/\?$/, '').trim();
        this.log(`Após remover '?': "${jsCondition}"`);

        // ESTRATÉGIA SIMPLES E FUNCIONAL: verificar se já contém operadores válidos
        if (jsCondition.includes('>=')) {
            this.log(`Operador >= encontrado, mantendo como está`);
        } else if (jsCondition.includes('<=')) {
            this.log(`Operador <= encontrado, mantendo como está`);
        } else if (jsCondition.includes('!=')) {
            jsCondition = jsCondition.replace(/\s*!=\s*/g, ' !== ');
            this.log(`Convertido != para !==`);
        } else if (jsCondition.includes('==')) {
            jsCondition = jsCondition.replace(/\s*==\s*/g, ' === ');
            this.log(`Convertido == para ===`);
        }
        
        // Conectores lógicos
        jsCondition = jsCondition.replace(/\s+e\s+/gi, ' && ');
        jsCondition = jsCondition.replace(/\s+ou\s+/gi, ' || ');
        jsCondition = jsCondition.replace(/\s+and\s+/gi, ' && ');
        jsCondition = jsCondition.replace(/\s+or\s+/gi, ' || ');

        // Normalizar espaços
        jsCondition = jsCondition.replace(/\s+/g, ' ').trim();

        this.log(`Condição JavaScript final: "${jsCondition}"`);
        return jsCondition;
    }

    /**
     * Encontra o ID do próximo nó a ser executado
     */
    findNextNodeId(currentId, decisionResult = null) {
        const node = this.parseResult.nodeMap.get(currentId);
        const outgoing = this.parseResult.connections.filter(c => c.from === currentId);

        this.log(`Buscando próximo nó de ${currentId}. Conexões encontradas: ${outgoing.length}`);

        if (outgoing.length === 0) {
            this.log(`Nenhuma conexão de saída encontrada para ${currentId}`);
            return null;
        }

        // Para nós de decisão
        if (node.type === 'decision') {
            this.log(`Processando decisão. Resultado: ${decisionResult ? 'Verdadeiro' : 'Falso'}`);
            
            // Procurar conexão apropriada baseada no resultado
            let targetConnection = null;
            
            if (decisionResult) {
                // Procurar por "Sim", "True", "Verdadeiro"
                targetConnection = outgoing.find(c => {
                    const label = c.label.toLowerCase();
                    return label.includes('sim') || label.includes('true') || label.includes('verdadeiro');
                });
            } else {
                // Procurar por "Não", "False", "Falso"
                targetConnection = outgoing.find(c => {
                    const label = c.label.toLowerCase();
                    return label.includes('não') || label.includes('nao') || label.includes('false') || label.includes('falso');
                });
            }
            
            if (targetConnection) {
                this.log(`Seguindo caminho: ${targetConnection.label} → ${targetConnection.to}`);
                return targetConnection.to;
            } else {
                this.log(`Caminho para decisão não encontrado. Usando primeira conexão disponível.`);
                return outgoing[0].to;
            }
        }

        // Para outros tipos de nós, usar a primeira conexão
        const nextId = outgoing[0].to;
        this.log(`Próximo nó: ${nextId}`);
        return nextId;
    }

    /**
     * Obter informações do passo atual
     */
    getCurrentStepInfo() {
        if (!this.currentStepId) {
            return 'Execução finalizada';
        }
        
        const currentNode = this.parseResult.nodeMap.get(this.currentStepId);
        
        if (!currentNode) {
            return 'Nó não encontrado';
        }
        
        // Se é um nó de saída e temos resultado, mostrar o resultado
        if (currentNode.type === 'output' && this.lastOutputResult !== null) {
            return `Resultado: ${this.lastOutputResult}`;
        }
        
        // Se é um nó de fim e temos resultados de saída, mostrar o último resultado
        if (currentNode.type === 'end' && this.outputResults.length > 0) {
            const lastResult = this.outputResults[this.outputResults.length - 1];
            return `Execução finalizada. Último resultado: ${lastResult.result}`;
        }
        
        return `Executando: ${currentNode.text}`;
    }

    /**
     * Obter número do passo atual
     */
    getCurrentStepNumber() {
        // O número de passos executados é o tamanho da pilha.
        // O passo atual é o próximo, então somamos 1.
        // No início (pilha vazia), estamos no passo 1.
        return this.executionStack.length + 1;
    }

    /**
     * Obter número total de passos
     */
    getTotalSteps() {
        return this.parseResult.nodes.length; // Uma aproximação
    }

    /**
     * Verificar se a execução está ativa
     */
    isExecutionActive() {
        return this.isRunning;
    }

    /**
     * Parar execução
     */
    stop() {
        this.isRunning = false;
        this.clearHighlights();
        this.log('Execução interrompida');
    }
    
    /**
     * Obter todos os resultados de saída
     */
    getAllOutputResults() {
        return this.outputResults;
    }
    
    /**
     * Obter último resultado de saída
     */
    getLastOutputResult() {
        return this.lastOutputResult;
    }
}

// Disponibilizar no escopo global
window.StepByStepExecutor = StepByStepExecutor;