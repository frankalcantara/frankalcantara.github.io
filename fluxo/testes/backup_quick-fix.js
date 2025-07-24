// Correção rápida para o parser de fluxogramas
function quickFixParser(mermaidCode) {
    const lines = mermaidCode.split('\n').map(line => line.trim()).filter(line => line);
    const nodes = [];
    const connections = [];
    
    for (let line of lines) {
        // Ignorar declaração do tipo de diagrama
        if (line.startsWith('flowchart') || line.startsWith('graph')) {
            continue;
        }
        
        // Parsear nós e conexões em uma linha
        // Exemplo: A[Início] --> B[Ler idade]
        const fullLineMatch = line.match(/^([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]\s*-->\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/);
        if (fullLineMatch) {
            const [, fromId, fromText, toId, toText] = fullLineMatch;
            
            // Adicionar nós se não existirem
            if (!nodes.find(n => n.id === fromId)) {
                nodes.push({
                    id: fromId,
                    text: fromText,
                    type: determineNodeType(fromText)
                });
            }
            if (!nodes.find(n => n.id === toId)) {
                nodes.push({
                    id: toId,
                    text: toText,
                    type: determineNodeType(toText)
                });
            }
            
            connections.push({ from: fromId, to: toId, label: '' });
            continue;
        }
        
        // Parsear conexões com labels
        // Exemplo: C -->|Sim| D[Pode votar]
        const labelConnectionMatch = line.match(/^([A-Za-z0-9_]+)\s*-->\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/);
        if (labelConnectionMatch) {
            const [, fromId, label, toId, toText] = labelConnectionMatch;
            
            if (!nodes.find(n => n.id === toId)) {
                nodes.push({
                    id: toId,
                    text: toText,
                    type: determineNodeType(toText)
                });
            }
            
            connections.push({ from: fromId, to: toId, label: label.trim() });
            continue;
        }
        
        // Parsear conexões simples
        // Exemplo: D --> F[Fim]
        const simpleConnectionMatch = line.match(/^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)[\[\(\{]([^\]\)\}]+)[\]\)\}]$/);
        if (simpleConnectionMatch) {
            const [, fromId, toId, toText] = simpleConnectionMatch;
            
            if (!nodes.find(n => n.id === toId)) {
                nodes.push({
                    id: toId,
                    text: toText,
                    type: determineNodeType(toText)
                });
            }
            
            connections.push({ from: fromId, to: toId, label: '' });
            continue;
        }
        
        // Parsear conexões sem texto de destino
        // Exemplo: D --> F
        const bareConnectionMatch = line.match(/^([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)$/);
        if (bareConnectionMatch) {
            const [, fromId, toId] = bareConnectionMatch;
            connections.push({ from: fromId, to: toId, label: '' });
            continue;
        }
    }
    
    return { nodes, connections };
}

function determineNodeType(text) {
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

// Função para executar passo a passo simplificada
function executeStepByStepSimple() {
    const code = document.getElementById('mermaid-editor').value;
    const { nodes, connections } = quickFixParser(code);
    
    console.log('Nós encontrados:', nodes);
    console.log('Conexões encontradas:', connections);
    
    const startNode = nodes.find(node => node.type === 'start');
    if (startNode) {
        logToConsole(`Nó de início encontrado: ${startNode.text}`);
        
        // Preparar variáveis de entrada
        const inputNodes = nodes.filter(node => node.type === 'input');
        const variableInputs = document.getElementById('variable-inputs');
        variableInputs.innerHTML = '';
        
        inputNodes.forEach(node => {
            const varName = node.text.match(/(\w+)/)?.[1] || 'var';
            const inputDiv = document.createElement('div');
            inputDiv.className = 'variable-input';
            
            inputDiv.innerHTML = `
                <label for="var-${varName}">${node.text}:</label>
                <input type="text" id="var-${varName}" data-variable="${varName}" placeholder="Digite o valor">
            `;
            
            variableInputs.appendChild(inputDiv);
        });
        
        // Mostrar controles passo a passo e configurar botões
        document.getElementById('step-controls').style.display = 'flex';
        document.getElementById('step-counter').textContent = 'Passo: 1/' + nodes.length;
        
        // Configurar estado dos botões para modo passo a passo
        if (typeof setStepByStepMode === 'function') {
            setStepByStepMode(true);
        }
        
    } else {
        logToConsole('Nenhum nó de início encontrado nos nós: ' + nodes.map(n => n.text).join(', '));
    }
}

// Adicionar ao escopo global
window.quickFixParser = quickFixParser;
window.executeStepByStepSimple = executeStepByStepSimple;

