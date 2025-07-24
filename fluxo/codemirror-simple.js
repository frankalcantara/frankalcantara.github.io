// CodeMirror 5 Simples para Mermaid
// Implementação mais estável e funcional

class SimpleMermaidEditor {
    constructor() {
        this.editor = null;
        this.isReady = false;
    }

    async initialize(containerId) {
        console.log('🎨 Inicializando CodeMirror 5...');
        
        try {
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Container '${containerId}' não encontrado`);
            }

            // CodeMirror deve estar disponível - sem fallback
            if (typeof CodeMirror === 'undefined') {
                console.error('❌ CodeMirror não carregado - aplicação falhará');
                throw new Error('CodeMirror não carregado');
            }

            // Limpar container
            container.innerHTML = '';
            
            // Criar textarea vazio
            const textarea = document.createElement('textarea');
            textarea.value = '';
            container.appendChild(textarea);

            // Criar editor CodeMirror 5
            this.editor = CodeMirror.fromTextArea(textarea, {
                mode: 'javascript', // Usar modo JavaScript para highlighting básico
                theme: 'default',
                lineNumbers: true,
                lineWrapping: true,
                indentUnit: 4,
                tabSize: 4,
                autoCloseBrackets: true,
                matchBrackets: true,
                foldGutter: true,
                gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
                extraKeys: {
                    'Ctrl-Space': 'autocomplete',
                    'Tab': 'indentMore',
                    'Shift-Tab': 'indentLess'
                }
            });

            // Aplicar estilos customizados
            this.applyCustomStyles();
            
            // Aplicar highlighting personalizado
            this.setupMermaidHighlighting();

            this.isReady = true;
            console.log('✅ CodeMirror 5 inicializado com sucesso!');
            return true;

        } catch (error) {
            console.error('❌ ERRO FATAL ao inicializar CodeMirror 5:', error);
            throw error; // Propagar erro - sem fallback
        }
    }

    applyCustomStyles() {
        if (!this.editor) return;

        // Aplicar estilos ao container do CodeMirror
        const wrapper = this.editor.getWrapperElement();
        wrapper.style.height = '100%';
        wrapper.style.fontSize = '14px';
        wrapper.style.fontFamily = '"Courier New", monospace';
        
        // Redimensionar editor
        this.editor.setSize('100%', '100%');
    }

    setupMermaidHighlighting() {
        if (!this.editor) return;

        // Aplicar highlighting personalizado usando overlays
        this.editor.on('change', () => {
            this.applyMermaidHighlighting();
        });

        // Aplicar highlighting inicial
        setTimeout(() => this.applyMermaidHighlighting(), 100);
    }

    applyMermaidHighlighting() {
        if (!this.editor) return;

        try {
            // Definir overlay para Mermaid
            CodeMirror.defineMode('mermaid-overlay', (config, parserConfig) => {
                return CodeMirror.overlayMode(
                    CodeMirror.getMode(config, 'javascript'),
                    {
                        token: (stream, state) => {
                            // Keywords
                            if (stream.match(/\b(flowchart|graph|TD|TB|BT|RL|LR)\b/)) {
                                return 'keyword';
                            }
                            
                            // Arrows
                            if (stream.match(/(-->|==>|---)/)) {
                                return 'operator';
                            }
                            
                            // Brackets with content
                            if (stream.match(/\[([^\]]+)\]/)) {
                                return 'string';
                            }
                            
                            // Braces with content
                            if (stream.match(/\{([^}]+)\}/)) {
                                return 'string';
                            }
                            
                            // Labels
                            if (stream.match(/\|([^|]+)\|/)) {
                                return 'variable-2';
                            }
                            
                            // Comments
                            if (stream.match(/%%.*$/)) {
                                return 'comment';
                            }
                            
                            // Node IDs
                            if (stream.match(/\b([A-Z][A-Za-z0-9_]*)\b/)) {
                                return 'variable';
                            }
                            
                            stream.next();
                            return null;
                        }
                    }
                );
            });

            // Aplicar o modo
            this.editor.setOption('mode', 'mermaid-overlay');
            
        } catch (error) {
            console.warn('Erro ao aplicar highlighting:', error);
        }
    }

    // Fallback removido - CodeMirror ou erro

    getValue() {
        return this.editor ? this.editor.getValue() : '';
    }

    setValue(value) {
        if (this.editor) {
            this.editor.setValue(value);
        }
    }

    focus() {
        if (this.editor) {
            this.editor.focus();
        }
    }
}

// CSS para highlighting customizado
const mermaidHighlightCSS = `
<style id="mermaid-highlight-css">
.cm-keyword { color: #d73a49 !important; font-weight: bold !important; }
.cm-variable { color: #6f42c1 !important; font-weight: 600 !important; }
.cm-variable-2 { color: #22863a !important; font-style: italic !important; }
.cm-string { color: #032f62 !important; font-weight: 500 !important; }
.cm-operator { color: #e36209 !important; font-weight: bold !important; }
.cm-comment { color: #6a737d !important; font-style: italic !important; }
.cm-number { color: #005cc5 !important; font-weight: 500 !important; }

/* Estilos do container */
.CodeMirror {
    height: 100% !important;
    font-family: 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.CodeMirror-gutters {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

.CodeMirror-linenumber {
    color: #64748b !important;
}

.CodeMirror-cursor {
    border-left: 2px solid #4f46e5 !important;
}

.CodeMirror-selected {
    background: rgba(79, 70, 229, 0.2) !important;
}
</style>
`;

// Adicionar CSS
if (!document.getElementById('mermaid-highlight-css')) {
    document.head.insertAdjacentHTML('beforeend', mermaidHighlightCSS);
}

// Instância global
window.simpleMermaidEditor = new SimpleMermaidEditor();

// INICIALIZAÇÃO AUTOMÁTICA REMOVIDA
// Agora é controlada pelo script.js

console.log('🎨 Simple Mermaid Editor (v5) carregado');