// Parser avançado para converter fluxogramas Mermaid em JavaScript executável
class FlowchartParser {
    constructor() {
        this.nodes = new Map();
        this.connections = [];
        this.variables = new Set();
        this.jsCode = '';
    }

    // Parsear o código Mermaid completo
    parse(mermaidCode) {
        this.reset();
        const lines = mermaidCode.split('\n').map(line => line.trim()).filter(line => line);
        
        for (let line of lines) {
            // Ignorar declaração do tipo de diagrama
            if (line.startsWith('flowchart') || line.startsWith('graph')) {
                continue;
            }
            
            this.parseLine(line);
        }
        
        return this.generateJavaScript();
    }

    // Resetar o parser
    reset() {
        this.nodes.clear();
        this.connections = [];
        this.variables.clear();
        this.jsCode = '';
    }

    // Parsear uma linha individual
    parseLine(line) {
        // Remover comentários
        line = line.replace(/%%.*$/, '').trim();
        if (!line) return;

        // Padrões de regex para diferentes elementos
        const patterns = {
            // Nó: A[Texto], A(Texto), A{Texto}
            node: /^([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/,
            // Conexão simples: A --> B
            simpleConnection: /^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)$/,
            // Conexão com label: A -->|label| B
            labeledConnection: /^([A-Za-z0-9_]+)\s*-->\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)$/
        };

        // Verificar se é um nó
        const nodeMatch = line.match(patterns.node);
        if (nodeMatch) {
            const [, id, text] = nodeMatch;
            this.addNode(id, text);
            return;
        }

        // Verificar se é uma conexão com label
        const labeledMatch = line.match(patterns.labeledConnection);
        if (labeledMatch) {
            const [, from, label, to] = labeledMatch;
            this.addConnection(from, to, label);
            return;
        }

        // Verificar se é uma conexão simples
        const simpleMatch = line.match(patterns.simpleConnection);
        if (simpleMatch) {
            const [, from, to] = simpleMatch;
            this.addConnection(from, to);
            return;
        }
    }

    // Adicionar um nó
    addNode(id, text) {
        const nodeType = this.determineNodeType(text);
        this.nodes.set(id, {
            id,
            text,
            type: nodeType,
            processed: false
        });

        // Extrair variáveis de nós de entrada
        if (nodeType === 'input') {
            const varName = this.extractVariableName(text);
            if (varName) {
                this.variables.add(varName);
            }
        }
    }

    // Adicionar uma conexão
    addConnection(from, to, label = '') {
        this.connections.push({ from, to, label });
    }

    // Determinar o tipo do nó
    determineNodeType(text) {
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('início') || lowerText.includes('inicio') || lowerText.includes('start')) {
            return 'start';
        }
        if (lowerText.includes('fim') || lowerText.includes('end')) {
            return 'end';
        }
        if (lowerText.includes('?') || lowerText.includes('se ') || lowerText.includes('if ')) {
            return 'decision';
        }
        if (lowerText.includes('ler ') || lowerText.includes('input') || lowerText.includes('digite') || lowerText.includes('entrada')) {
            return 'input';
        }
        if (lowerText.includes('escrever ') || lowerText.includes('mostrar ') || lowerText.includes('output') || lowerText.includes('print') || lowerText.includes('exibir')) {
            return 'output';
        }
        
        return 'process';
    }

    // Extrair nome da variável
    extractVariableName(text) {
        // Padrões para extrair variáveis
        const patterns = [
            /ler\s+(\w+)/i,
            /input\s+(\w+)/i,
            /digite\s+(\w+)/i,
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

    // Gerar código JavaScript
    generateJavaScript() {
        let js = '// Código JavaScript gerado automaticamente\n';
        js += 'async function executarFluxograma() {\n';
        
        // Declarar variáveis
        if (this.variables.size > 0) {
            js += '  // Declaração de variáveis\n';
            for (let varName of this.variables) {
                js += `  let ${varName};\n`;
            }
            js += '\n';
        }

        // Encontrar nó de início
        const startNode = this.findStartNode();
        if (startNode) {
            js += this.generateNodeCode(startNode.id, new Set());
        } else {
            js += '  console.log("Nenhum nó de início encontrado");\n';
        }

        js += '}\n\n';
        js += '// Executar o fluxograma\n';
        js += 'executarFluxograma().catch(console.error);';

        return js;
    }

    // Encontrar nó de início
    findStartNode() {
        for (let [id, node] of this.nodes) {
            if (node.type === 'start') {
                return node;
            }
        }
        // Se não encontrar nó de início, usar o primeiro nó
        return this.nodes.values().next().value;
    }

    // Gerar código para um nó específico
    generateNodeCode(nodeId, visited, indent = '  ') {
        if (visited.has(nodeId)) {
            return ''; // Evitar loops infinitos
        }
        visited.add(nodeId);

        const node = this.nodes.get(nodeId);
        if (!node) return '';

        let code = '';

        switch (node.type) {
            case 'start':
                code += `${indent}// Início: ${node.text}\n`;
                code += `${indent}console.log("Iniciando execução...");\n`;
                break;

            case 'input':
                const varName = this.extractVariableName(node.text);
                if (varName) {
                    code += `${indent}// Entrada: ${node.text}\n`;
                    code += `${indent}${varName} = await getInput("${node.text}");\n`;
                    code += `${indent}console.log("${varName} =", ${varName});\n`;
                }
                break;

            case 'output':
                code += `${indent}// Saída: ${node.text}\n`;
                code += `${indent}console.log(${this.processOutputText(node.text)});\n`;
                break;

            case 'process':
                code += `${indent}// Processo: ${node.text}\n`;
                const processCode = this.processCalculation(node.text);
                if (processCode !== node.text) {
                    code += `${indent}${processCode};\n`;
                }
                code += `${indent}console.log("Processando: ${node.text}");\n`;
                break;

            case 'decision':
                code += `${indent}// Decisão: ${node.text}\n`;
                const condition = this.processCondition(node.text);
                code += `${indent}if (${condition}) {\n`;
                
                // Encontrar conexões "Sim" e "Não"
                const yesConnection = this.connections.find(conn => 
                    conn.from === nodeId && (conn.label.toLowerCase().includes('sim') || conn.label.toLowerCase().includes('yes') || conn.label.toLowerCase().includes('true'))
                );
                const noConnection = this.connections.find(conn => 
                    conn.from === nodeId && (conn.label.toLowerCase().includes('não') || conn.label.toLowerCase().includes('no') || conn.label.toLowerCase().includes('false'))
                );

                if (yesConnection) {
                    code += `${indent}  console.log("Condição verdadeira: ${node.text}");\n`;
                    code += this.generateNodeCode(yesConnection.to, new Set(visited), indent + '  ');
                }
                
                code += `${indent}} else {\n`;
                
                if (noConnection) {
                    code += `${indent}  console.log("Condição falsa: ${node.text}");\n`;
                    code += this.generateNodeCode(noConnection.to, new Set(visited), indent + '  ');
                }
                
                code += `${indent}}\n`;
                return code; // Retornar aqui para evitar processar conexões normais

            case 'end':
                code += `${indent}// Fim: ${node.text}\n`;
                code += `${indent}console.log("Execução finalizada.");\n`;
                return code; // Não processar mais conexões após o fim
        }

        // Processar próximas conexões (exceto para decisões que já foram processadas)
        if (node.type !== 'decision') {
            const nextConnections = this.connections.filter(conn => conn.from === nodeId);
            for (let connection of nextConnections) {
                code += this.generateNodeCode(connection.to, new Set(visited), indent);
            }
        }

        return code;
    }

    // Processar texto de saída
    processOutputText(text) {
        // Substituir variáveis no texto
        return `"${text}".replace(/\\b(\\w+)\\b/g, (match) => {
            return typeof window[match] !== 'undefined' ? window[match] : match;
        })`;
    }

    // Processar cálculo
    processCalculation(text) {
        // Detectar atribuições simples
        const assignmentMatch = text.match(/(\w+)\s*=\s*(.+)/);
        if (assignmentMatch) {
            const [, varName, expression] = assignmentMatch;
            this.variables.add(varName);
            return `${varName} = ${expression}`;
        }

        return text;
    }

    // Processar condição
    processCondition(text) {
        // Converter condições em português para JavaScript
        let condition = text
            .replace(/\s*>=\s*/g, ' >= ')
            .replace(/\s*<=\s*/g, ' <= ')
            .replace(/\s*>\s*/g, ' > ')
            .replace(/\s*<\s*/g, ' < ')
            .replace(/\s*==\s*/g, ' === ')
            .replace(/\s*=\s*/g, ' === ')
            .replace(/\s+e\s+/gi, ' && ')
            .replace(/\s+ou\s+/gi, ' || ')
            .replace(/\s+and\s+/gi, ' && ')
            .replace(/\s+or\s+/gi, ' || ');

        // Remover o ponto de interrogação se existir
        condition = condition.replace(/\?$/, '');

        return condition;
    }

    // Obter lista de variáveis
    getVariables() {
        return Array.from(this.variables);
    }

    // Obter nós parseados
    getNodes() {
        return Array.from(this.nodes.values());
    }

    // Obter conexões
    getConnections() {
        return [...this.connections];
    }
}

// Função auxiliar para obter entrada do usuário
async function getInput(prompt) {
    return new Promise((resolve) => {
        // Tentar obter do input da interface
        const varName = prompt.match(/(\w+)/)?.[1] || 'input';
        const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
        
        if (inputElement && inputElement.value) {
            const value = inputElement.value;
            resolve(isNaN(value) ? value : parseFloat(value));
        } else {
            // Fallback para prompt do navegador
            const value = window.prompt(prompt);
            resolve(isNaN(value) ? value : parseFloat(value));
        }
    });
}

// Exportar para uso global
window.FlowchartParser = FlowchartParser;

