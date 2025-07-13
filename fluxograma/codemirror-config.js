// Syntax Highlighting Simples para Mermaid - CORRIGIDO
// Abordagem: Overlay de highlighting sobre textarea existente

class SimpleHighlighter {
    constructor() {
        this.textarea = null;
        this.highlightLayer = null;
        this.isActive = false;
        this.updateTimeout = null;
    }
    
    // Inicializar highlighting
    initialize(textareaId) {
        console.log('🎨 Inicializando syntax highlighting simples...');
        
        try {
            this.textarea = document.getElementById(textareaId);
            if (!this.textarea) {
                throw new Error(`Textarea '${textareaId}' não encontrada`);
            }
            
            this.createHighlightLayer();
            this.bindEvents();
            this.updateHighlighting();
            
            this.isActive = true;
            console.log('✅ Syntax highlighting ativado!');
            
            return true;
            
        } catch (error) {
            console.warn('⚠️ Erro ao ativar highlighting:', error);
            return false;
        }
    }
    
    // Criar camada de highlighting
    createHighlightLayer() {
        const wrapper = this.textarea.parentElement;
        
        // Criar div de highlighting
        this.highlightLayer = document.createElement('div');
        this.highlightLayer.className = 'syntax-highlight-layer';
        
        // Copiar estilos do textarea de forma mais precisa
        const computedStyle = window.getComputedStyle(this.textarea);
        const textareaRect = this.textarea.getBoundingClientRect();
        const wrapperRect = wrapper.getBoundingClientRect();
        
        // Calcular offset da numeração de linhas com mais precisão
        const lineNumbers = wrapper.querySelector('.line-numbers');
        let leftOffset = 0;
        
        if (lineNumbers) {
            // Se há numeração, calcular sua largura
            const lineNumbersRect = lineNumbers.getBoundingClientRect();
            leftOffset = lineNumbersRect.width;
        } else {
            // Fallback: calcular pela diferença de posição
            leftOffset = textareaRect.left - wrapperRect.left;
        }
        
        this.highlightLayer.style.cssText = `
            position: absolute;
            top: 0;
            left: ${leftOffset}px;
            width: calc(100% - ${leftOffset}px);
            height: 100%;
            font-family: ${computedStyle.fontFamily};
            font-size: ${computedStyle.fontSize};
            line-height: ${computedStyle.lineHeight};
            padding: ${computedStyle.padding};
            margin: 0;
            border: none;
            border-radius: ${computedStyle.borderRadius};
            box-sizing: border-box;
            overflow: hidden;
            pointer-events: none;
            z-index: 1;
            background: transparent;
            color: transparent;
            white-space: pre;
            word-wrap: normal;
            overflow-wrap: normal;
            letter-spacing: ${computedStyle.letterSpacing};
            text-align: left;
        `;
        
        // Garantir que wrapper seja relativo
        if (window.getComputedStyle(wrapper).position === 'static') {
            wrapper.style.position = 'relative';
        }
        
        // Inserir antes do textarea
        wrapper.insertBefore(this.highlightLayer, this.textarea);
        
        // Garantir que textarea fique por cima mas transparente
        this.textarea.style.position = 'relative';
        this.textarea.style.zIndex = '2';
        this.textarea.style.background = 'transparent';
        this.textarea.style.color = 'transparent';
        this.textarea.style.caretColor = '#1f2937';
        
        // Adicionar classe ao wrapper para CSS adicional
        wrapper.classList.add('highlighting-active');
    }
    
    // Bind eventos
    bindEvents() {
        // Atualizar highlighting quando texto mudar
        this.textarea.addEventListener('input', () => {
            this.debounceUpdate();
        });
        
        // Sincronizar scroll
        this.textarea.addEventListener('scroll', () => {
            if (this.highlightLayer) {
                this.highlightLayer.scrollTop = this.textarea.scrollTop;
                this.highlightLayer.scrollLeft = this.textarea.scrollLeft;
            }
        });
        
        // Atualizar ao redimensionar e recalcular posição
        window.addEventListener('resize', () => {
            this.debounceUpdate();
            this.recalculatePosition();
        });
    }
    
    // Recalcular posição do overlay
    recalculatePosition() {
        if (!this.highlightLayer || !this.textarea) return;
        
        const wrapper = this.textarea.parentElement;
        const textareaRect = this.textarea.getBoundingClientRect();
        const wrapperRect = wrapper.getBoundingClientRect();
        
        // Calcular offset da numeração de linhas com mais precisão
        const lineNumbers = wrapper.querySelector('.line-numbers');
        let leftOffset = 0;
        
        if (lineNumbers) {
            // Se há numeração, calcular sua largura
            const lineNumbersRect = lineNumbers.getBoundingClientRect();
            leftOffset = lineNumbersRect.width;
        } else {
            // Fallback: calcular pela diferença de posição
            leftOffset = textareaRect.left - wrapperRect.left;
        }
        
        this.highlightLayer.style.left = `${leftOffset}px`;
        this.highlightLayer.style.width = `calc(100% - ${leftOffset}px)`;
    }
    
    // Debounce para atualizações
    debounceUpdate() {
        if (this.updateTimeout) {
            clearTimeout(this.updateTimeout);
        }
        
        this.updateTimeout = setTimeout(() => {
            this.updateHighlighting();
        }, 300);
    }
    
    // Atualizar highlighting
    updateHighlighting() {
        if (!this.highlightLayer || !this.textarea) return;
        
        const text = this.textarea.value;
        const highlightedText = this.highlightMermaidSyntax(text);
        
        this.highlightLayer.innerHTML = highlightedText;
        
        // Recalcular posição para garantir alinhamento
        this.recalculatePosition();
        
        // Sincronizar scroll
        this.highlightLayer.scrollTop = this.textarea.scrollTop;
        this.highlightLayer.scrollLeft = this.textarea.scrollLeft;
    }
    
    // Aplicar syntax highlighting - VERSÃO CORRIGIDA
    highlightMermaidSyntax(text) {
        if (!text) return '';
        
        // Processar linha por linha para manter quebras de linha
        const lines = text.split('\n');
        const highlightedLines = lines.map(line => {
            if (!line.trim()) return line; // Manter linhas vazias
            
            let highlighted = line
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // Aplicar highlighting em ordem de prioridade
            
            // 1. Comments %% (deve vir primeiro)
            if (line.includes('%%')) {
                highlighted = highlighted.replace(/%%(.*)$/, '<span class="line-comment">%%$1</span>');
                return highlighted; // Se é comentário, não processar mais
            }
            
            // 2. Keywords (flowchart, TD, etc.)
            highlighted = highlighted.replace(/\b(flowchart|graph|TD|TB|BT|RL|LR|subgraph|end)\b/g, '<span class="keyword">$1</span>');
            
            // 3. Connections com labels |texto|
            highlighted = highlighted.replace(/(-->)\s*\|([^|]+)\|/g, '<span class="operator">$1</span> |<span class="comment">$2</span>|');
            
            // 4. Connections simples
            highlighted = highlighted.replace(/(-->|---|\.\.>|\-\.-)/g, '<span class="operator">$1</span>');
            
            // 5. Node text [texto] e decision text {texto}
            highlighted = highlighted.replace(/\[([^\]]+)\]/g, '[<span class="string">$1</span>]');
            highlighted = highlighted.replace(/\{([^}]+)\}/g, '{<span class="string">$1</span>}');
            
            // 6. Node IDs (A, B, C, etc.) - mais específico
            highlighted = highlighted.replace(/^(\s*)([A-Z][A-Za-z0-9_]*)(\s+)/g, '$1<span class="variable">$2</span>$3');
            highlighted = highlighted.replace(/(\s)([A-Z][A-Za-z0-9_]*)(\s*(?:-->|\[|\{))/g, '$1<span class="variable">$2</span>$3');
            
            return highlighted;
        });
        
        return highlightedLines.join('\n');
    }
    
    // Verificar se está ativo
    isReady() {
        return this.isActive;
    }
    
    // Destruir highlighting
    destroy() {
        if (this.highlightLayer) {
            this.highlightLayer.remove();
            this.highlightLayer = null;
        }
        
        if (this.textarea) {
            this.textarea.style.background = '';
            this.textarea.style.position = '';
            this.textarea.style.zIndex = '';
            this.textarea.style.color = '';
            this.textarea.style.caretColor = '';
            
            // Remover classe do wrapper
            const wrapper = this.textarea.parentElement;
            if (wrapper) {
                wrapper.classList.remove('highlighting-active');
            }
        }
        
        this.isActive = false;
        console.log('🔄 Syntax highlighting removido');
    }
}

// Instância global
window.simpleHighlighter = new SimpleHighlighter();

// Funções de compatibilidade
window.initializeCodeMirror = function(textareaId, changeCallback) {
    const success = window.simpleHighlighter.initialize(textareaId);
    
    if (success && changeCallback) {
        // Bind callback se fornecido
        const textarea = document.getElementById(textareaId);
        if (textarea) {
            textarea.addEventListener('input', () => {
                changeCallback(textarea.value);
            });
        }
    }
    
    return Promise.resolve(success);
};

window.getEditorValue = function() {
    const textarea = document.getElementById('mermaid-editor');
    return textarea ? textarea.value : '';
};

window.setEditorValue = function(value) {
    const textarea = document.getElementById('mermaid-editor');
    if (textarea) {
        textarea.value = value;
        
        // Atualizar highlighting se ativo
        if (window.simpleHighlighter.isReady()) {
            window.simpleHighlighter.updateHighlighting();
        }
        
        // Atualizar numeração se disponível
        if (typeof updateLineNumbers === 'function') {
            updateLineNumbers();
        }
    }
};

window.focusEditor = function() {
    const textarea = document.getElementById('mermaid-editor');
    if (textarea) {
        textarea.focus();
    }
};

window.mermaidEditor = {
    getValue: () => window.getEditorValue(),
    setValue: (value) => window.setEditorValue(value),
    focus: () => window.focusEditor(),
    isReady: () => window.simpleHighlighter.isReady()
};

console.log('🎨 Simple Syntax Highlighter CORRIGIDO carregado');

// Teste de debugging
window.testHighlighting = function() {
    const testText = `flowchart TD
    A[Início] --> B[Ler nome]
    B --> C{idade >= 18}
    C -->|Sim| D[Pode votar]`;
    
    if (window.simpleHighlighter && window.simpleHighlighter.isReady()) {
        const highlighted = window.simpleHighlighter.highlightMermaidSyntax(testText);
        console.log('🇫🇧 Texto original:', testText);
        console.log('🎨 Texto com highlighting:', highlighted);
        return highlighted;
    } else {
        console.log('⚠️ Highlighter não está pronto');
        return null;
    }
};

console.log('🧪 Para testar highlighting, use: testHighlighting()');
