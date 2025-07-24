// Parser simplificado e funcional para fluxogramas
class SimpleFlowchartParser {
    constructor() {
        this.nodes = [];
        this.connections = [];
        this.variables = new Set();
    }

    parse(mermaidCode) {
        this.nodes = [];
        this.connections = [];
        this.variables.clear();

        const lines = mermaidCode.split('\n').filter(line => line.trim());
        
        for (let line of lines) {
            line = line.trim();
            if (line.startsWith('flowchart') || line.startsWith('graph')) continue;
            
            this.parseLine(line);
        }

        return this.generateJavaScript();
    }

    parseLine(line) {
        // Padrão para conexões: A[texto] --> B[texto] ou A --> B{texto}
        const connectionMatch = line.match(/([A-Z]+)(?:\[([^\]]+)\])?\s*-->\s*(?:\|([^|]+)\|)?\s*([A-Z]+)(?:[\[\{]([^\]\}]+)[\]\}])?/);
        
        if (connectionMatch) {
            const [, fromId, fromText, label, toId, toText] = connectionMatch;
            
            // Adicionar nós se não existirem
            if (fromText && !this.nodes.find(n => n.id === fromId)) {
                this.nodes.push({
                    id: fromId,
                    text: fromText,
                    type: this.getNodeType(fromText)
                });
            }
            
            if (toText && !this.nodes.find(n => n.id === toId)) {
                this.nodes.push({
                    id: toId,
                    text: toText,
                    type: this.getNodeType(toText)
                });
            }
            
            this.connections.push({
                from: fromId,
                to: toId,
                label: label || ''
            });
        }
    }

    getNodeType(text) {
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('início') || lowerText.includes('inicio') || lowerText.includes('start')) {
            return 'start';
        }
        if (lowerText.includes('fim') || lowerText.includes('end')) {
            return 'end';
        }
        if (lowerText.includes('?')) {
            return 'decision';
        }
        if (lowerText.includes('ler ') || lowerText.includes('input')) {
            const varMatch = text.match(/ler\s+(\w+)/i);
            if (varMatch) {
                this.variables.add(varMatch[1]);
            }
            return 'input';
        }
        if (lowerText.includes('mostrar ') || lowerText.includes('output') || lowerText.includes('pode votar') || lowerText.includes('não pode votar')) {
            return 'output';
        }
        
        return 'process';
    }

    generateJavaScript() {
        let jsCode = '// Código gerado automaticamente\n';
        jsCode += 'async function executarFluxograma() {\n';
        jsCode += '  try {\n';
        
        // Encontrar nó de início
        const startNode = this.nodes.find(n => n.type === 'start');
        if (!startNode) {
            throw new Error('Nenhum nó de início encontrado');
        }
        
        // Gerar código para cada nó
        jsCode += '    logToConsole("=== Iniciando execução ===");\n';
        
        // Obter variáveis de entrada
        const inputNodes = this.nodes.filter(n => n.type === 'input');
        for (let node of inputNodes) {
            const varName = this.extractVariableName(node.text);
            jsCode += `    const ${varName}Input = document.querySelector('input[data-variable="${varName}"]');\n`;
            jsCode += `    if (!${varName}Input || !${varName}Input.value) {\n`;
            jsCode += `      logToConsole('Erro: Valor para ${varName} não fornecido');\n`;
            jsCode += `      return;\n`;
            jsCode += `    }\n`;
            jsCode += `    const ${varName} = parseFloat(${varName}Input.value) || ${varName}Input.value;\n`;
            jsCode += `    logToConsole('${varName} = ' + ${varName});\n`;
        }
        
        // Processar decisões
        const decisionNodes = this.nodes.filter(n => n.type === 'decision');
        for (let node of decisionNodes) {
            const condition = this.processCondition(node.text);
            const simConnections = this.connections.filter(c => c.from === node.id && c.label.toLowerCase().includes('sim'));
            const naoConnections = this.connections.filter(c => c.from === node.id && c.label.toLowerCase().includes('não'));
            
            jsCode += `    logToConsole('Executando: ${node.text}');\n`;
            jsCode += `    if (${condition}) {\n`;
            jsCode += `      logToConsole('Decisão: ${node.text} = Verdadeiro');\n`;
            
            if (simConnections.length > 0) {
                const nextNode = this.nodes.find(n => n.id === simConnections[0].to);
                if (nextNode && nextNode.type === 'output') {
                    jsCode += `      logToConsole('${nextNode.text}');\n`;
                }
            }
            
            jsCode += `    } else {\n`;
            jsCode += `      logToConsole('Decisão: ${node.text} = Falso');\n`;
            
            if (naoConnections.length > 0) {
                const nextNode = this.nodes.find(n => n.id === naoConnections[0].to);
                if (nextNode && nextNode.type === 'output') {
                    jsCode += `      logToConsole('${nextNode.text}');\n`;
                }
            }
            
            jsCode += `    }\n`;
        }
        
        jsCode += '    logToConsole("=== Execução finalizada ===");\n';
        jsCode += '  } catch (error) {\n';
        jsCode += '    logToConsole("Erro na execução: " + error.message);\n';
        jsCode += '  }\n';
        jsCode += '}\n\n';
        jsCode += 'executarFluxograma();';
        
        return jsCode;
    }

    extractVariableName(text) {
        const match = text.match(/ler\s+(\w+)/i);
        return match ? match[1] : 'valor';
    }

    processCondition(text) {
        let condition = text.replace(/\?$/, '').trim();
        
        // Substituir operadores
        condition = condition
            .replace(/>=/, ' >= ')
            .replace(/<=/, ' <= ')
            .replace(/!=/, ' !== ')
            .replace(/==/, ' === ')
            .replace(/([^><=!])=([^=])/g, '$1 === $2')
            .replace(/\s+/g, ' ')
            .trim();
        
        return condition;
    }

    getVariables() {
        return Array.from(this.variables);
    }

    getNodes() {
        return this.nodes;
    }
}

// Adicionar ao escopo global
window.SimpleFlowchartParser = SimpleFlowchartParser;

