// Executor passo a passo para fluxogramas
class StepExecutor {
    constructor(parser, consoleLogger) {
        this.parser = parser;
        this.logger = consoleLogger;
        this.currentNodeId = null;
        this.executionStack = [];
        this.variables = {};
        this.stepHistory = [];
        this.isExecuting = false;
    }

    // Iniciar execução passo a passo
    async startStepByStep() {
        this.reset();
        
        const nodes = this.parser.getNodes();
        const startNode = nodes.find(node => node.type === 'start') || nodes[0];
        
        if (!startNode) {
            this.logger('Nenhum nó de início encontrado');
            return false;
        }

        this.currentNodeId = startNode.id;
        this.isExecuting = true;
        this.logger('=== Iniciando execução passo a passo ===');
        
        // Preparar variáveis de entrada
        await this.prepareInputVariables();
        
        return true;
    }

    // Executar próximo passo
    async executeNextStep() {
        if (!this.isExecuting || !this.currentNodeId) {
            return false;
        }

        const nodes = this.parser.getNodes();
        const currentNode = nodes.find(node => node.id === this.currentNodeId);
        
        if (!currentNode) {
            this.logger('Nó atual não encontrado');
            return false;
        }

        this.logger(`Executando: ${currentNode.text}`);
        
        // Adicionar ao histórico
        this.stepHistory.push({
            nodeId: this.currentNodeId,
            node: currentNode,
            variables: { ...this.variables }
        });

        // Executar o nó atual
        const result = await this.executeNode(currentNode);
        
        // Determinar próximo nó
        const nextNodeId = this.getNextNode(currentNode, result);
        
        if (nextNodeId) {
            this.currentNodeId = nextNodeId;
        } else {
            this.isExecuting = false;
            this.logger('Execução finalizada');
        }

        return this.isExecuting;
    }

    // Voltar ao passo anterior
    goToPreviousStep() {
        if (this.stepHistory.length > 1) {
            // Remover o passo atual
            this.stepHistory.pop();
            
            // Voltar ao passo anterior
            const previousStep = this.stepHistory[this.stepHistory.length - 1];
            this.currentNodeId = previousStep.nodeId;
            this.variables = { ...previousStep.variables };
            
            this.logger(`Voltando para: ${previousStep.node.text}`);
            return true;
        }
        
        return false;
    }

    // Executar um nó específico
    async executeNode(node) {
        switch (node.type) {
            case 'start':
                this.logger('Iniciando execução...');
                return 'continue';

            case 'input':
                return await this.handleInputNode(node);

            case 'output':
                return this.handleOutputNode(node);

            case 'process':
                return this.handleProcessNode(node);

            case 'decision':
                return this.handleDecisionNode(node);

            case 'end':
                this.logger('Execução finalizada');
                this.isExecuting = false;
                return 'end';

            default:
                this.logger(`Tipo de nó desconhecido: ${node.type}`);
                return 'continue';
        }
    }

    // Lidar com nó de entrada
    async handleInputNode(node) {
        const varName = this.extractVariableName(node.text);
        
        if (varName) {
            const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
            
            if (inputElement && inputElement.value !== '') {
                const value = inputElement.value;
                this.variables[varName] = isNaN(value) ? value : parseFloat(value);
                this.logger(`${varName} = ${this.variables[varName]}`);
            } else {
                this.logger(`Aguardando entrada para: ${varName}`);
                // Destacar o campo de entrada
                if (inputElement) {
                    inputElement.focus();
                    inputElement.style.border = '2px solid #f59e0b';
                }
                return 'wait_input';
            }
        }
        
        return 'continue';
    }

    // Lidar com nó de saída
    handleOutputNode(node) {
        const output = this.processOutputText(node.text);
        this.logger(`Saída: ${output}`);
        return 'continue';
    }

    // Lidar com nó de processo
    handleProcessNode(node) {
        const result = this.processCalculation(node.text);
        this.logger(`Processamento: ${result}`);
        return 'continue';
    }

    // Lidar com nó de decisão
    handleDecisionNode(node) {
        const condition = this.evaluateCondition(node.text);
        this.logger(`Decisão: ${node.text} = ${condition ? 'Verdadeiro' : 'Falso'}`);
        return condition ? 'true' : 'false';
    }

    // Obter próximo nó
    getNextNode(currentNode, result) {
        const connections = this.parser.getConnections();
        
        if (currentNode.type === 'decision') {
            // Para decisões, procurar conexão baseada no resultado
            const connection = connections.find(conn => {
                if (conn.from !== currentNode.id) return false;
                
                const label = conn.label.toLowerCase();
                if (result === 'true') {
                    return label.includes('sim') || label.includes('yes') || label.includes('true') || label.includes('verdadeiro');
                } else {
                    return label.includes('não') || label.includes('no') || label.includes('false') || label.includes('falso');
                }
            });
            
            return connection ? connection.to : null;
        } else {
            // Para outros tipos, procurar primeira conexão
            const connection = connections.find(conn => conn.from === currentNode.id);
            return connection ? connection.to : null;
        }
    }

    // Preparar variáveis de entrada
    async prepareInputVariables() {
        const nodes = this.parser.getNodes();
        const inputNodes = nodes.filter(node => node.type === 'input');
        
        const variableInputs = document.getElementById('variable-inputs');
        variableInputs.innerHTML = '';
        
        inputNodes.forEach(node => {
            const varName = this.extractVariableName(node.text);
            if (varName) {
                const inputDiv = document.createElement('div');
                inputDiv.className = 'variable-input';
                
                inputDiv.innerHTML = `
                    <label for="var-${varName}">${node.text}:</label>
                    <input type="text" id="var-${varName}" data-variable="${varName}" 
                           placeholder="Digite o valor" onchange="this.style.border = ''">
                `;
                
                variableInputs.appendChild(inputDiv);
            }
        });
    }

    // Extrair nome da variável
    extractVariableName(text) {
        const patterns = [
            /ler\s+(\w+)/i,
            /input\s+(\w+)/i,
            /digite\s+(\w+)/i,
            /entrada\s+(\w+)/i,
            /(\w+)\s*=/,
            /(\w+)/
        ];

        for (let pattern of patterns) {
            const match = text.match(pattern);
            if (match && match[1]) {
                return match[1];
            }
        }

        return null;
    }

    // Processar texto de saída
    processOutputText(text) {
        return text.replace(/\b(\w+)\b/g, (match) => {
            return this.variables[match] !== undefined ? this.variables[match] : match;
        });
    }

    // Processar cálculo
    processCalculation(text) {
        const assignmentMatch = text.match(/(\w+)\s*=\s*(.+)/);
        if (assignmentMatch) {
            const [, varName, expression] = assignmentMatch;
            
            try {
                // Substituir variáveis na expressão
                const processedExpression = expression.replace(/\b(\w+)\b/g, (match) => {
                    return this.variables[match] !== undefined ? this.variables[match] : match;
                });
                
                // Avaliar expressão matemática simples
                if (/^[\d\s+\-*/().]+$/.test(processedExpression)) {
                    this.variables[varName] = eval(processedExpression);
                    return `${varName} = ${this.variables[varName]}`;
                }
            } catch (error) {
                this.logger(`Erro no cálculo: ${error.message}`);
            }
        }
        
        return text;
    }

    // Avaliar condição
    evaluateCondition(text) {
        try {
            let condition = text.replace(/\?$/, ''); // Remover ?
            
            // Substituir variáveis
            condition = condition.replace(/\b(\w+)\b/g, (match) => {
                return this.variables[match] !== undefined ? this.variables[match] : match;
            });
            
            // Converter operadores
            condition = condition
                .replace(/\s*>=\s*/g, ' >= ')
                .replace(/\s*<=\s*/g, ' <= ')
                .replace(/\s*>\s*/g, ' > ')
                .replace(/\s*<\s*/g, ' < ')
                .replace(/\s*==\s*/g, ' === ')
                .replace(/\s*=\s*/g, ' === ')
                .replace(/\s+e\s+/gi, ' && ')
                .replace(/\s+ou\s+/gi, ' || ');
            
            return eval(condition);
        } catch (error) {
            this.logger(`Erro na condição: ${error.message}`);
            return false;
        }
    }

    // Destacar nó atual
    highlightCurrentNode() {
        // Remover destaque anterior
        document.querySelectorAll('.current-step').forEach(el => {
            el.classList.remove('current-step');
        });
        
        if (this.currentNodeId) {
            // Tentar encontrar o elemento SVG correspondente
            const svgElements = document.querySelectorAll('#mermaid-diagram svg g.node');
            svgElements.forEach(element => {
                const textElement = element.querySelector('text');
                if (textElement && textElement.textContent.includes(this.currentNodeId)) {
                    element.classList.add('current-step');
                }
            });
        }
    }

    // Obter informações do passo atual
    getCurrentStepInfo() {
        if (!this.currentNodeId) return '';
        
        const nodes = this.parser.getNodes();
        const currentNode = nodes.find(node => node.id === this.currentNodeId);
        
        return currentNode ? `Executando: ${currentNode.text}` : '';
    }

    // Obter número do passo atual
    getCurrentStepNumber() {
        return this.stepHistory.length;
    }

    // Obter total de passos (estimativa)
    getTotalSteps() {
        return this.parser.getNodes().length;
    }

    // Resetar executor
    reset() {
        this.currentNodeId = null;
        this.executionStack = [];
        this.variables = {};
        this.stepHistory = [];
        this.isExecuting = false;
        
        // Remover destaques
        document.querySelectorAll('.current-step').forEach(el => {
            el.classList.remove('current-step');
        });
        
        // Limpar bordas dos inputs
        document.querySelectorAll('input[data-variable]').forEach(input => {
            input.style.border = '';
        });
    }

    // Verificar se está executando
    isRunning() {
        return this.isExecuting;
    }

    // Obter variáveis atuais
    getVariables() {
        return { ...this.variables };
    }
}

// Exportar para uso global
window.StepExecutor = StepExecutor;

