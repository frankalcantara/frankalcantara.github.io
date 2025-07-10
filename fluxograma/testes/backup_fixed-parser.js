// Parser corrigido para fluxogramas Mermaid
class FixedFlowchartParser {
    constructor() {
        this.nodes = new Map();
        this.connections = [];
        this.variables = new Set();
    }

    // Parsear o código Mermaid completo
    parse(mermaidCode) {
        this.reset();
        const lines = mermaidCode.split('\n').map(line => line.trim()).filter(line => line);
        
        console.log('Linhas para parsear:', lines);
        
        for (let line of lines) {
            // Ignorar declaração do tipo de diagrama
            if (line.startsWith('flowchart') || line.startsWith('graph')) {
                continue;
            }
            
            this.parseLine(line);
        }
        
        console.log('Nós parseados:', Array.from(this.nodes.values()));
        console.log('Conexões parseadas:', this.connections);
        
        return this.generateJavaScript();
    }

    // Resetar o parser
    reset() {
        this.nodes.clear();
        this.connections = [];
        this.variables.clear();
    }

    // Parsear uma linha individual
    parseLine(line) {
        // Remover comentários
        line = line.replace(/%%.*$/, '').trim();
        if (!line) return;

        console.log('Parseando linha:', line);

        // Padrões mais específicos para capturar nós em conexões
        const patterns = {
            // Conexão completa: A[Texto] --> B[Texto]
            fullConnection: /^([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]\s*-->\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/,
            // Conexão com label: A -->|label| B[Texto]
            labeledConnection: /^([A-Za-z0-9_]+)\s*-->\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/,
            // Conexão simples: A --> B[Texto]
            simpleConnection: /^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/,
            // Conexão apenas IDs: A --> B
            bareConnection: /^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)$/,
            // Nó isolado: A[Texto]
            isolatedNode: /^([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/
        };

        // Verificar conexão completa: A[Texto] --> B[Texto]
        const fullMatch = line.match(patterns.fullConnection);
        if (fullMatch) {
            const [, fromId, fromText, toId, toText] = fullMatch;
            console.log('Conexão completa encontrada:', { fromId, fromText, toId, toText });
            
            this.addNode(fromId, fromText);
            this.addNode(toId, toText);
            this.addConnection(fromId, toId);
            return;
        }

        // Verificar conexão com label: A -->|label| B[Texto]
        const labeledMatch = line.match(patterns.labeledConnection);
        if (labeledMatch) {
            const [, fromId, label, toId, toText] = labeledMatch;
            console.log('Conexão com label encontrada:', { fromId, label, toId, toText });
            
            this.addNode(toId, toText);
            this.addConnection(fromId, toId, label.trim());
            return;
        }

        // Verificar conexão simples: A --> B[Texto]
        const simpleMatch = line.match(patterns.simpleConnection);
        if (simpleMatch) {
            const [, fromId, toId, toText] = simpleMatch;
            console.log('Conexão simples encontrada:', { fromId, toId, toText });
            
            this.addNode(toId, toText);
            this.addConnection(fromId, toId);
            return;
        }

        // Verificar conexão apenas IDs: A --> B
        const bareMatch = line.match(patterns.bareConnection);
        if (bareMatch) {
            const [, fromId, toId] = bareMatch;
            console.log('Conexão bare encontrada:', { fromId, toId });
            
            this.addConnection(fromId, toId);
            return;
        }

        // Verificar nó isolado: A[Texto]
        const nodeMatch = line.match(patterns.isolatedNode);
        if (nodeMatch) {
            const [, id, text] = nodeMatch;
            console.log('Nó isolado encontrado:', { id, text });
            
            this.addNode(id, text);
            return;
        }

        console.log('Linha não reconhecida:', line);
    }

    // Adicionar um nó
    addNode(id, text) {
        if (this.nodes.has(id)) {
            return; // Nó já existe
        }

        const nodeType = this.determineNodeType(text);
        console.log(`Adicionando nó: ${id} [${text}] tipo: ${nodeType}`);
        
        this.nodes.set(id, {
            id,
            text,
            type: nodeType
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
            console.log('Nó de início encontrado:', startNode);
            js += this.generateNodeCode(startNode.id, new Set());
        } else {
            console.log('Nenhum nó de início encontrado. Nós disponíveis:', Array.from(this.nodes.values()));
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
        return `"${text}".replace(/\\b(\\w+)\\b/g, (match) => {
            return typeof window[match] !== 'undefined' ? window[match] : match;
        })`;
    }

    // Processar cálculo
    processCalculation(text) {
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
        let condition = text.trim();
        
        // Remover ? no final primeiro
        condition = condition.replace(/\?$/, '');
        
        // Processar operadores de comparação de forma mais cuidadosa
        condition = condition
            .replace(/\s*>=\s*/g, ' >= ')
            .replace(/\s*<=\s*/g, ' <= ')
            .replace(/\s*!=\s*/g, ' !== ')
            .replace(/\s*==\s*/g, ' === ')
            .replace(/\s*>\s*/g, ' > ')
            .replace(/\s*<\s*/g, ' < ')
            .replace(/\s+e\s+/gi, ' && ')
            .replace(/\s+ou\s+/gi, ' || ')
            .replace(/\s+and\s+/gi, ' && ')
            .replace(/\s+or\s+/gi, ' || ');
        
        // Garantir que não há operadores duplicados
        condition = condition
            .replace(/>\s*=\s*=\s*=/g, ' >= ')
            .replace(/<\s*=\s*=\s*=/g, ' <= ')
            .replace(/=\s*=\s*=\s*=/g, ' === ');
        
        return condition.trim();
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
        const varName = prompt.match(/(\w+)/)?.[1] || 'input';
        const inputElement = document.querySelector(`input[data-variable="${varName}"]`);
        
        if (inputElement && inputElement.value) {
            const value = inputElement.value;
            resolve(isNaN(value) ? value : parseFloat(value));
        } else {
            const value = window.prompt(prompt);
            resolve(isNaN(value) ? value : parseFloat(value));
        }
    });
}

// Exportar para uso global
window.FixedFlowchartParser = FixedFlowchartParser;

