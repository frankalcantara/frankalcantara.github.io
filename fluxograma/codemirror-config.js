// Syntax Highlighting SIMPLES E FUNCIONAL para Mermaid
// Versão simplificada que FUNCIONA

class SimpleHighlighter {
    constructor() {
        this.textarea = null;
        this.highlightLayer = null;
        this.isActive = false;
    }
    
    initialize(textareaId) {
        console.log('🎨 Inicializando syntax highlighting SIMPLES...');
        
        try {
            this.textarea = document.getElementById(textareaId);
            if (!this.textarea) {
                throw new Error(`Textarea '${textareaId}' não encontrada`);
            }
            
            this.createSimpleHighlightLayer();
            this.bindEvents();
            this.updateHighlighting();
            
            this.isActive = true;
            console.log('✅ Syntax highlighting SIMPLES ativado!');
            
            return true;
            
        } catch (error) {
            console.warn('⚠️ Erro ao ativar highlighting:', error);
            return false;
        }
    }
    
    createSimpleHighlightLayer() {
        const wrapper = this.textarea.parentElement;
        
        // Remover highlighting anterior se existir
        const existingLayer = wrapper.querySelector('.syntax-highlight-layer');
        if (existingLayer) {
            existingLayer.remove();
        }
        
        // Criar div de highlighting
        this.highlightLayer = document.createElement('div');
        this.highlightLayer.className = 'syntax-highlight-layer';
        
        // Copiar estilos do textarea
        const textareaStyle = window.getComputedStyle(this.textarea);
        
        // Calcular posição da numeração
        const lineNumbers = wrapper.querySelector('.line-numbers');
        const leftOffset = lineNumbers ? lineNumbers.offsetWidth : 0;
        
        // Aplicar estilos à camada de highlighting
        Object.assign(this.highlightLayer.style, {
            position: 'absolute',
            top: '0',
            left: leftOffset + 'px',
            width: `calc(100% - ${leftOffset}px)`,
            height: '100%',
            fontFamily: textareaStyle.fontFamily,
            fontSize: textareaStyle.fontSize,
            lineHeight: textareaStyle.lineHeight,
            padding: textareaStyle.padding,
            margin: '0',
            border: 'none',
            boxSizing: 'border-box',
            overflow: 'hidden',
            pointerEvents: 'none',
            zIndex: '1',
            background: '#ffffff',
            color: '#1f2937',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word'
        });
        
        // Configurar wrapper
        wrapper.style.position = 'relative';
        
        // Inserir camada ANTES do textarea
        wrapper.insertBefore(this.highlightLayer, this.textarea);
        
        // Tornar textarea TOTALMENTE transparente
        Object.assign(this.textarea.style, {
            position: 'relative',
            zIndex: '2',
            background: 'transparent',
            backgroundColor: 'transparent',
            color: 'transparent',
            caretColor: '#000000',
            border: 'none',
            outline: 'none',
            textShadow: 'none',
            boxShadow: 'none'
        });
        
        // Adicionar classe e estilos CSS
        wrapper.classList.add('highlighting-active');
        this.addStyles();
    }
    
    addStyles() {
        // Remover estilos anteriores
        const existingStyle = document.getElementById('simple-highlighting-styles');
        if (existingStyle) {
            existingStyle.remove();
        }
        
        const style = document.createElement('style');
        style.id = 'simple-highlighting-styles';
        style.textContent = `
            /* FORÇAR transparência total do textarea */
            .highlighting-active textarea,
            .highlighting-active #mermaid-editor {
                color: transparent !important;
                background: transparent !important;
                background-color: transparent !important;
                caret-color: #000000 !important;
                text-shadow: none !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Garantir que highlight layer seja visível */
            .syntax-highlight-layer {
                pointer-events: none !important;
                z-index: 1 !important;
                background: #ffffff !important;
            }
            
            /* Cores de backup por classes - caso inline falhe */
            .mmd-keyword { color: #d73a49 !important; font-weight: bold !important; }
            .mmd-node-id { color: #6f42c1 !important; font-weight: 600 !important; }
            .mmd-node-text { color: #032f62 !important; font-weight: 500 !important; }
            .mmd-connection { color: #e36209 !important; font-weight: bold !important; }
            .mmd-bracket { color: #586069 !important; font-weight: bold !important; }
            .mmd-label { color: #22863a !important; font-style: italic !important; }
            .mmd-comment { color: #6a737d !important; font-style: italic !important; }
            .mmd-number { color: #005cc5 !important; }
            .mmd-operator { color: #d73a49 !important; font-weight: bold !important; }
        `;
        
        document.head.appendChild(style);
    }
    
    bindEvents() {
        this.textarea.addEventListener('input', () => this.updateHighlighting());
        this.textarea.addEventListener('scroll', () => this.syncScroll());
        window.addEventListener('resize', () => this.updateHighlighting());
    }
    
    syncScroll() {
        if (this.highlightLayer) {
            this.highlightLayer.scrollTop = this.textarea.scrollTop;
            this.highlightLayer.scrollLeft = this.textarea.scrollLeft;
        }
    }
    
    updateHighlighting() {
        if (!this.highlightLayer || !this.textarea) return;
        
        const text = this.textarea.value;
        const highlighted = this.applyHighlighting(text);
        
        this.highlightLayer.innerHTML = highlighted;
        this.syncScroll();
    }
    
    applyHighlighting(text) {
        if (!text) return '';
        
        return text.split('\n').map(line => {
            if (!line.trim()) return line;
            
            // Escapar HTML
            let result = line
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // Aplicar highlighting com CORES INLINE para garantir que funcionem
            
            // 1. Comentários primeiro
            if (result.includes('%%')) {
                return result.replace(/%%(.*)$/, '<span style="color: #6a737d; font-style: italic;">%%$1</span>');
            }
            
            // 2. Keywords
            result = result.replace(/\b(graph|flowchart|TD|TB|BT|RL|LR)\b/g, '<span style="color: #d73a49; font-weight: bold;">$1</span>');
            
            // 3. Conectores
            result = result.replace(/(==>|-->)/g, '<span style="color: #e36209; font-weight: bold;">$1</span>');
            
            // 4. Texto em colchetes
            result = result.replace(/\[([^\]]+)\]/g, '<span style="color: #586069; font-weight: bold;">[</span><span style="color: #032f62; font-weight: 500;">$1</span><span style="color: #586069; font-weight: bold;">]</span>');
            
            // 5. Texto em chaves
            result = result.replace(/\{([^}]+)\}/g, '<span style="color: #586069; font-weight: bold;">{</span><span style="color: #032f62; font-weight: 500;">$1</span><span style="color: #586069; font-weight: bold;">}</span>');
            
            // 6. IDs dos nós
            result = result.replace(/^(\s*)([A-Z][A-Za-z0-9_]*)(\s+)/g, '$1<span style="color: #6f42c1; font-weight: 600;">$2</span>$3');
            result = result.replace(/(<\/span>)(\s+)([A-Z][A-Za-z0-9_]*)(\s*(?=<span|\s*$))/g, '$1$2<span style="color: #6f42c1; font-weight: 600;">$3</span>$4');
            
            return result;
        }).join('\n');
    }
    
    isReady() {
        return this.isActive;
    }
    
    destroy() {
        if (this.highlightLayer) {
            this.highlightLayer.remove();
            this.highlightLayer = null;
        }
        
        if (this.textarea) {
            // Restaurar textarea
            Object.assign(this.textarea.style, {
                background: '',
                color: '',
                caretColor: '',
                border: '',
                outline: ''
            });
            
            const wrapper = this.textarea.parentElement;
            if (wrapper) {
                wrapper.classList.remove('highlighting-active');
            }
        }
        
        // Remover estilos
        const styleElement = document.getElementById('simple-highlighting-styles');
        if (styleElement) {
            styleElement.remove();
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
        
        if (window.simpleHighlighter.isReady()) {
            window.simpleHighlighter.updateHighlighting();
        }
        
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

console.log('🎨 Syntax Highlighter SIMPLES carregado');

// Função de debug
window.debugHighlighting = function() {
    const textarea = document.getElementById('mermaid-editor');
    const wrapper = textarea ? textarea.parentElement : null;
    const highlightLayer = wrapper ? wrapper.querySelector('.syntax-highlight-layer') : null;
    
    console.log('=== DEBUG HIGHLIGHTING ===');
    console.log('Textarea encontrada:', !!textarea);
    console.log('Wrapper encontrado:', !!wrapper);
    console.log('Highlight layer encontrada:', !!highlightLayer);
    
    if (textarea) {
        console.log('Textarea styles:', {
            color: textarea.style.color,
            background: textarea.style.background,
            zIndex: textarea.style.zIndex,
            position: textarea.style.position
        });
    }
    
    if (highlightLayer) {
        console.log('Highlight layer styles:', {
            zIndex: highlightLayer.style.zIndex,
            position: highlightLayer.style.position,
            left: highlightLayer.style.left,
            width: highlightLayer.style.width
        });
        console.log('Highlight layer content preview:', highlightLayer.innerHTML.substring(0, 100));
    }
    
    if (wrapper) {
        console.log('Wrapper classes:', wrapper.className);
        console.log('Wrapper position:', window.getComputedStyle(wrapper).position);
    }
    
    return {
        textarea: !!textarea,
        wrapper: !!wrapper, 
        highlightLayer: !!highlightLayer,
        isActive: window.simpleHighlighter.isReady()
    };
};

console.log('🔧 Para debug, use: debugHighlighting()');
