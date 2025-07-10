/**
 * Parser Unificado para Fluxogramas Mermaid - CORRIGIDO
 * Converte sintaxe Mermaid em estrutura executável para ensino algorítmico
 */
class UnifiedFlowchartParser {
    constructor() {
        this.nodes = new Map();
        this.connections = [];
        this.variables = new Set();
        this.executionOrder = [];
    }

    /**
     * Parsear código Mermaid completo
     * @param {string} mermaidCode - Código do fluxograma em sintaxe Mermaid
     * @returns {Object} Estrutura parseada do fluxograma
     */
    parse(mermaidCode) {
        this.reset();

        if (!mermaidCode || !mermaidCode.trim()) {
            throw new Error('Código do fluxograma está vazio');
        }

        const lines = mermaidCode.split('\n')
            .map(line => line.trim())
            .filter(line => line && !line.startsWith('%%') && !line.startsWith('flowchart') && !line.startsWith('graph'));

        console.log('📋 Linhas a serem parseadas:', lines);

        // Padrões regex aprimorados para capturar o formato Mermaid
        // Formato: A[Texto] --> B[Texto] ou A[Texto] -->|Label| B[Texto]
        const fullConnectionPattern = /^([a-zA-Z0-9_]+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]\s*-->\s*(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_]+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]$/;
        
        // Formato: A --> B{Texto} ou A -->|Label| B{Texto} (destino com chaves)
        const connectionToBracePattern = /^([a-zA-Z0-9_]+)\s*-->\s*(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_]+)\s*\{([^\}]+)\}$/;
        
        // Formato: A{Texto} -->|Label| B[Texto] (origem com chaves) - CORRIGIDO
        const connectionFromBracePattern = /^([a-zA-Z0-9_]+)\s*-->\s*(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_]+)\s*[\[\(]([^\]\)]+)[\]\)]$/;
        
        // Formato: A --> B[Texto] ou A -->|Label| B[Texto] (destino com colchetes)
        const connectionToBracketPattern = /^([a-zA-Z0-9_]+)\s*-->\s*(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_]+)\s*\[([^\]]+)\]$/;
        
        // Formato: A --> B ou A -->|Label| B (nós já definidos)
        const simpleConnectionPattern = /^([a-zA-Z0-9_]+)\s*-->\s*(?:\|([^|]+)\|)?\s*([a-zA-Z0-9_]+)$/;
        
        // Formato: A[Texto] ou A{Texto} (definição de nó isolado)
        const nodePattern = /^([a-zA-Z0-9_]+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]$/;

        // Processar cada linha
        for (const line of lines) {
            console.log(`🔍 Processando linha: "${line}"`);
            
            // Tentar padrão completo primeiro (nó + conexão + nó)
            const fullMatch = line.match(fullConnectionPattern);
            if (fullMatch) {
                const [, fromId, fromText, label, toId, toText] = fullMatch;
                console.log(`✅ Padrão completo: ${fromId}[${fromText}] --> ${toId}[${toText}]`);
                
                this.addNode(fromId, fromText);
                this.addNode(toId, toText);
                this.addConnection(fromId, toId, label || '');
                continue;
            }

            // Tentar padrão de conexão para nó com chaves (B --> C{texto})
            const toBraceMatch = line.match(connectionToBracePattern);
            if (toBraceMatch) {
                const [, fromId, label, toId, toText] = toBraceMatch;
                console.log(`✅ Conexão para chaves: ${fromId} --> ${toId}{${toText}}`);
                
                // Adicionar nós se não existirem
                if (!this.nodes.has(fromId)) {
                    this.addNode(fromId, fromId);
                }
                this.addNode(toId, toText);
                this.addConnection(fromId, toId, label || '');
                continue;
            }

            // Tentar padrão de conexão para nó com colchetes (C -->|Sim| D[texto]) - NOVO
            const toBracketMatch = line.match(connectionToBracketPattern);
            if (toBracketMatch) {
                const [, fromId, label, toId, toText] = toBracketMatch;
                console.log(`✅ Conexão para colchetes: ${fromId} -->|${label || ''}| ${toId}[${toText}]`);
                
                // Adicionar nós se não existirem
                if (!this.nodes.has(fromId)) {
                    this.addNode(fromId, fromId);
                }
                this.addNode(toId, toText);
                this.addConnection(fromId, toId, label || '');
                continue;
            }

            // Tentar padrão de conexão de nó com chaves (C{texto} -->|Label| D[texto])
            const fromBraceMatch = line.match(connectionFromBracePattern);
            if (fromBraceMatch) {
                const [, fromId, label, toId, toText] = fromBraceMatch;
                console.log(`✅ Conexão de chaves: ${fromId} -->|${label || ''}| ${toId}[${toText}]`);
                
                if (!this.nodes.has(fromId)) {
                    this.addNode(fromId, fromId);
                }
                this.addNode(toId, toText);
                this.addConnection(fromId, toId, label || '');
                continue;
            }

            // Tentar padrão de conexão simples
            const simpleMatch = line.match(simpleConnectionPattern);
            if (simpleMatch) {
                const [, fromId, label, toId] = simpleMatch;
                console.log(`✅ Conexão simples: ${fromId} --> ${toId}`);
                
                // Adicionar nós se não existirem (usando ID como texto)
                if (!this.nodes.has(fromId)) {
                    this.addNode(fromId, fromId);
                }
                if (!this.nodes.has(toId)) {
                    this.addNode(toId, toId);
                }
                
                this.addConnection(fromId, toId, label || '');
                continue;
            }

            // Tentar padrão de nó isolado
            const nodeMatch = line.match(nodePattern);
            if (nodeMatch) {
                const [, id, text] = nodeMatch;
                console.log(`✅ Nó isolado: ${id}[${text}]`);
                this.addNode(id, text);
                continue;
            }

            console.log(`⚠️ Linha não reconhecida: "${line}"`);
        }

        console.log(`📊 Resultado do parsing: ${this.nodes.size} nós, ${this.connections.length} conexões`);
        console.log('📋 Nós encontrados:', Array.from(this.nodes.keys()));
        console.log('🔗 Conexões encontradas:', this.connections);

        // Construir ordem de execução
        this.buildExecutionOrder();

        return this.getParseResult();
    }

    /**
     * Resetar estado do parser
     */
    reset() {
        this.nodes.clear();
        this.connections = [];
        this.variables.clear();
        this.executionOrder = [];
    }

    /**
     * Adicionar nó ao grafo
     * @param {string} id - ID do nó
     * @param {string} text - Texto do nó
     */
    addNode(id, text) {
        if (!this.nodes.has(id)) {
            const nodeType = this.determineNodeType(text);
            const node = {
                id,
                text: text.trim(),
                type: nodeType,
                shape: this.getNodeShape(text)
            };

            this.nodes.set(id, node);
            console.log(`➕ Nó adicionado: ${id} (${nodeType}) - "${text}"`);

            // Extrair variáveis de nós de entrada
            if (nodeType === 'input') {
                const varName = this.extractVariableName(text);
                if (varName) {
                    this.variables.add(varName);
                    console.log(`📝 Variável extraída: ${varName}`);
                }
            }
        }
    }

    /**
     * Adicionar conexão entre nós
     * @param {string} fromId - ID do nó de origem
     * @param {string} toId - ID do nó de destino
     * @param {string} label - Rótulo da conexão
     */
    addConnection(fromId, toId, label) {
        const connection = {
            from: fromId,
            to: toId,
            label: label.trim(),
            condition: this.parseConditionLabel(label)
        };
        
        this.connections.push(connection);
        console.log(`🔗 Conexão adicionada: ${fromId} -> ${toId} (${label || 'sem label'})`);
    }

    /**
     * Determinar tipo do nó baseado no texto
     * @param {string} text - Texto do nó
     * @returns {string} Tipo do nó
     */
    determineNodeType(text) {
        const lowerText = text.toLowerCase();

        if (lowerText.includes('início') || lowerText.includes('inicio') || lowerText.includes('start')) {
            return 'start';
        }
        if (lowerText.includes('fim') || lowerText.includes('end')) {
            return 'end';
        }
        if (lowerText.includes('?') || text.includes('>=') || text.includes('<=') || text.includes('>') || text.includes('<') || text.includes('==')) {
            return 'decision';
        }
        if (lowerText.includes('ler ') || lowerText.includes('input') || lowerText.includes('digite') || lowerText.includes('entrada')) {
            return 'input';
        }
        if (lowerText.includes('mostrar ') || lowerText.includes('escrever ') || lowerText.includes('output') || 
            lowerText.includes('print') || lowerText.includes('exibir') || text.match(/pode votar/i) || text.match(/não pode votar/i)) {
            return 'output';
        }

        return 'process';
    }

    /**
     * Determinar forma do nó baseado no texto original
     * @param {string} text - Texto original com delimitadores
     * @returns {string} Forma do nó
     */
    getNodeShape(text) {
        return 'rectangle';
    }

    /**
     * Extrair nome da variável do texto de entrada
     * @param {string} text - Texto do nó de entrada
     * @returns {string|null} Nome da variável ou null
     */
    extractVariableName(text) {
        const patterns = [
            /ler\s+(\w+)/i,           // "Ler idade"
            /digite\s+(\w+)/i,        // "Digite nome"
            /entrada\s+(\w+)/i,       // "Entrada valor"
            /input\s+(\w+)/i,         // "Input number"
            /(\w+)$/                  // Última palavra
        ];

        for (const pattern of patterns) {
            const match = text.match(pattern);
            if (match) {
                return match[1].toLowerCase();
            }
        }

        return 'valor';
    }

    /**
     * Parsear rótulo de condição
     * @param {string} label - Rótulo da conexão
     * @returns {Object} Objeto com informações da condição
     */
    parseConditionLabel(label) {
        if (!label) return null;

        const lowerLabel = label.toLowerCase();
        
        return {
            value: label,
            isTrue: lowerLabel.includes('sim') || lowerLabel.includes('true') || lowerLabel.includes('verdadeiro'),
            isFalse: lowerLabel.includes('não') || lowerLabel.includes('nao') || lowerLabel.includes('false') || lowerLabel.includes('falso')
        };
    }

    /**
     * Construir ordem de execução baseada no grafo
     */
    buildExecutionOrder() {
        const startNode = Array.from(this.nodes.values()).find(node => node.type === 'start');
        if (!startNode) {
            throw new Error('Nenhum nó de início encontrado. Use um nó como [Início] para começar.');
        }

        console.log(`🎯 Nó inicial encontrado: ${startNode.id}`);
        
        const visited = new Set();
        const order = [];

        this.traverseGraph(startNode.id, visited, order);
        this.executionOrder = order;
        
        console.log(`📋 Ordem de execução: ${order.join(' -> ')}`);
    }

    /**
     * Atravessar grafo para construir ordem de execução
     * @param {string} nodeId - ID do nó atual
     * @param {Set} visited - Conjunto de nós visitados
     * @param {Array} order - Array da ordem de execução
     */
    traverseGraph(nodeId, visited, order) {
        if (visited.has(nodeId)) return;

        visited.add(nodeId);
        const node = this.nodes.get(nodeId);
        if (node) {
            order.push(nodeId);
        }

        // Encontrar próximos nós
        const outgoingConnections = this.connections.filter(conn => conn.from === nodeId);
        
        for (const connection of outgoingConnections) {
            this.traverseGraph(connection.to, visited, order);
        }
    }

    /**
     * Obter resultado do parsing
     * @returns {Object} Estrutura completa do fluxograma parseado
     */
    getParseResult() {
        return {
            nodes: Array.from(this.nodes.values()),
            connections: this.connections,
            variables: Array.from(this.variables),
            executionOrder: this.executionOrder,
            nodeMap: this.nodes
        };
    }

    /**
     * Gerar código JavaScript executável
     * @returns {string} Código JavaScript
     */
    generateJavaScript() {
        const parseResult = this.getParseResult();
        
        let jsCode = '// Código gerado automaticamente do fluxograma\n';
        jsCode += 'async function executarFluxograma() {\n';
        jsCode += '  try {\n';
        jsCode += '    logToConsole("=== Iniciando execução do fluxograma ===");\n\n';

        // Declarar variáveis
        if (parseResult.variables.length > 0) {
            jsCode += '    // Declaração de variáveis\n';
            for (const varName of parseResult.variables) {
                jsCode += `    let ${varName};\n`;
            }
            jsCode += '\n';
        }

        // Obter valores de entrada
        jsCode += '    // Obter valores de entrada\n';
        const inputNodes = parseResult.nodes.filter(node => node.type === 'input');
        for (const node of inputNodes) {
            const varName = this.extractVariableName(node.text);
            jsCode += `    const ${varName}Input = document.querySelector('input[data-variable="${varName}"]');\n`;
            jsCode += `    if (!${varName}Input || !${varName}Input.value.trim()) {\n`;
            jsCode += `      logToConsole('Erro: Valor para "${varName}" não fornecido');\n`;
            jsCode += `      return;\n`;
            jsCode += `    }\n`;
            jsCode += `    ${varName} = isNaN(${varName}Input.value) ? ${varName}Input.value.trim() : parseFloat(${varName}Input.value);\n`;
            jsCode += `    logToConsole('${varName} = ' + ${varName});\n\n`;
        }

        // Processar nós de decisão e saída
        const decisionNodes = parseResult.nodes.filter(node => node.type === 'decision');
        for (const node of decisionNodes) {
            const condition = this.convertConditionToJS(node.text);
            
            jsCode += `    // Decisão: ${node.text}\n`;
            jsCode += `    logToConsole('Verificando: ${node.text}');\n`;
            jsCode += `    if (${condition}) {\n`;
            jsCode += `      logToConsole('Resultado: Verdadeiro');\n`;
            
            // Encontrar conexão "Sim"
            const trueConnection = parseResult.connections.find(conn => 
                conn.from === node.id && conn.condition && conn.condition.isTrue
            );
            if (trueConnection) {
                const trueNode = parseResult.nodeMap.get(trueConnection.to);
                if (trueNode && trueNode.type === 'output') {
                    jsCode += `      logToConsole('${trueNode.text}');\n`;
                }
            }
            
            jsCode += `    } else {\n`;
            jsCode += `      logToConsole('Resultado: Falso');\n`;
            
            // Encontrar conexão "Não"
            const falseConnection = parseResult.connections.find(conn => 
                conn.from === node.id && conn.condition && conn.condition.isFalse
            );
            if (falseConnection) {
                const falseNode = parseResult.nodeMap.get(falseConnection.to);
                if (falseNode && falseNode.type === 'output') {
                    jsCode += `      logToConsole('${falseNode.text}');\n`;
                }
            }
            
            jsCode += `    }\n\n`;
        }

        jsCode += '    logToConsole("=== Execução finalizada ===");\n';
        jsCode += '  } catch (error) {\n';
        jsCode += '    logToConsole("Erro na execução: " + error.message);\n';
        jsCode += '  }\n';
        jsCode += '}\n\n';
        jsCode += 'executarFluxograma();';

        return jsCode;
    }

    /**
     * Converter condição do fluxograma para JavaScript
     * @param {string} condition - Condição em linguagem natural
     * @returns {string} Condição em JavaScript
     */
    convertConditionToJS(condition) {
        let jsCondition = condition.replace(/\?$/, '').trim();

        // Substituir operadores
        jsCondition = jsCondition
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

        return jsCondition;
    }

    /**
     * Obter nós do fluxograma
     * @returns {Array} Array de nós
     */
    getNodes() {
        return Array.from(this.nodes.values());
    }

    /**
     * Obter variáveis do fluxograma
     * @returns {Array} Array de variáveis
     */
    getVariables() {
        return Array.from(this.variables);
    }

    /**
     * Obter conexões do fluxograma
     * @returns {Array} Array de conexões
     */
    getConnections() {
        return this.connections;
    }
}

// Disponibilizar no escopo global
window.UnifiedFlowchartParser = UnifiedFlowchartParser;