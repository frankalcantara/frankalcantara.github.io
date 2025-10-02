# Kyrion Forms MVP

Um clone moderno do Google Forms com suporte avançado a **LaTeX** e **código**, desenvolvido com Material Design 3 e tecnologias web modernas.

## 🚀 Funcionalidades

### ✅ Módulo 1: Estrutura Base (Implementado)
- [x] Layout responsivo com Material Design 3
- [x] Sistema de roteamento SPA
- [x] Navegação lateral com menu mobile
- [x] Página inicial com dashboard
- [x] Estrutura modular de componentes

### ✅ Módulo 2: Sistema de Formulários (Implementado)
- [x] Criação e gerenciamento de formulários
- [x] Interface "Meus Formulários" com cards
- [x] Edição inline de título e descrição
- [x] Sistema de navegação entre páginas
- [x] Sincronização automática de dados entre páginas

### ✅ Módulo 3: Perguntas Múltipla Escolha (Implementado)
- [x] Sistema de questões múltipla escolha completo
- [x] Edição inline de enunciados com markdown
- [x] Sistema "outro item" para expansão automática
- [x] Registry-based state management
- [x] Event delegation para performance

### ✅ Módulo 4: Renderização Markdown (Implementado)
- [x] Integração completa do Marked.js
- [x] Suporte GitHub Flavored Markdown (GFM)
- [x] Renderização customizada de tabelas
- [x] Preservação de acentos e caracteres especiais
- [x] Responsividade com scroll horizontal

### 🔄 Próximos Módulos
- [ ] **Módulo 5**: Suporte LaTeX avançado (KaTeX)
- [ ] **Módulo 6**: Editor de Código (CodeMirror)
- [ ] **Módulo 7**: Responder Formulários
- [ ] **Módulo 8**: Persistência Local e Export/Import

## 🛠️ Tecnologias

### Frontend
- **HTML5 + CSS3 + JavaScript** (Vanilla, sem frameworks)
- **Material Design 3** via CDN
- **Material Web Components** para UI
- **Editor.js** para edição rica de texto
- **KaTeX** para renderização de LaTeX
- **CodeMirror 6** para edição de código

### Armazenamento
- **IndexedDB** (principal)
- **localStorage** (fallback)

### CDNs Utilizadas
```html
<!-- Material Design -->
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
<script type="module" src="https://unpkg.com/@material/web@1.0.0/all.js"></script>

<!-- Editor.js -->
<script src="https://cdn.jsdelivr.net/npm/@editorjs/editorjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/header@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/paragraph@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/list@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/code@latest"></script>

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>

<!-- CodeMirror 6 -->
<script type="module" src="https://cdn.skypack.dev/codemirror@6"></script>
```

## 📁 Estrutura do Projeto

```
kyrion-forms/
├── index.html              # Página principal
├── css/
│   └── main.css            # Estilos consolidados (Material Design + markdown)
├── js/
│   ├── main.js             # ✅ Aplicação principal ATIVA
│   ├── app.js              # Aplicação modular (não utilizada)
│   ├── router.js           # Sistema de roteamento SPA (não utilizada)
│   └── views/
│       └── home.js         # Página inicial (não utilizada)
└── docs/
    ├── README.md
    ├── UPDATES.md
    ├── MULTIPLE-CHOICE-*.md
    └── QUESTIONS-REGISTRY.md
```

## 🎯 Como Executar

### Método 1: Servidor Local (Recomendado)
```bash
# Navegue até a pasta do projeto
cd kyrion-forms

# Inicie um servidor HTTP local
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js (se tiver npx)
npx http-server

# Acesse: http://localhost:8000
```

### Método 2: Live Server (VS Code)
1. Instale a extensão "Live Server"
2. Clique com botão direito em `index.html`
3. Selecione "Open with Live Server"

### Método 3: Abrir Arquivo (Limitado)
⚠️ **Não recomendado**: Abrir diretamente no browser pode causar problemas de CORS com módulos ES6.

## 🎨 Material Design 3

O projeto utiliza o **Material Design 3** com:

### Color Tokens
- Sistema de cores adaptativo (claro/escuro)
- Tokens semânticos (`--md-sys-color-primary`, etc.)
- Suporte automático ao tema do sistema

### Componentes
- `md-filled-button`, `md-text-button`
- `md-icon-button`, `md-fab`
- `md-list`, `md-list-item`
- `md-circular-progress`
- `md-icon`

### Typography
- Fonte Roboto
- Escalas tipográficas MD3
- Hierarquia visual clara

## 🧩 Arquitetura

### Single Page Application (SPA)
- **Router**: Gerencia navegação entre views
- **BaseComponent**: Classe base para todos os componentes
- **App**: Controlador principal da aplicação

### Sistema de Componentes
```javascript
// Exemplo de componente
export default class MyView extends BaseComponent {
    async render() {
        return `<div>HTML content</div>`;
    }
    
    async mount() {
        // Configurar event listeners
        this.setupEventListeners();
    }
}
```

### Gerenciamento de Estado
- Estado global em `window.KyrionForms`
- Persistência automática (IndexedDB + localStorage)
- Auto-save a cada 30 segundos

## 📱 Responsividade

### Breakpoints
- **Desktop**: > 768px
- **Tablet**: 481px - 768px
- **Mobile**: ≤ 480px

### Adaptações Mobile
- Sidebar colapsável
- Touch-friendly buttons
- Simplified layouts
- Bottom navigation

## 🔧 Desenvolvimento

### Próximos Passos (Módulo 2)
1. **Sistema de Formulários**
   - Criar/editar formulários
   - Gerenciar perguntas
   - Preview em tempo real
   - Autosave

### Convenções de Código
- **ES6 Modules** para organização
- **Async/await** para operações assíncronas
- **Material Design tokens** para styling
- **Mobile-first** approach

### Debug
```javascript
// Acessar estado global
console.log(window.KyrionForms);

// Acessar instância da app
console.log(window.app);

// Verificar storage
window.app.storage.getAllForms().then(console.log);
```

## 🎯 Roadmap

### Fase 1: MVP (8 semanas) ✅ 4/8
- [x] **Semana 1**: Estrutura base + UI
- [x] **Semana 2**: Sistema de formulários
- [x] **Semana 3**: Múltipla escolha
- [x] **Semana 4**: Renderização Markdown (Marked.js)
- [ ] **Semana 5**: Suporte LaTeX avançado (KaTeX)
- [ ] **Semana 6**: Editor de código (CodeMirror)
- [ ] **Semana 7**: Responder formulários
- [ ] **Semana 8**: Persistência + export/import

### Fase 2: Melhorias
- Backend com Node.js + MongoDB
- Autenticação OAuth
- Compartilhamento de formulários
- Analytics avançadas
- PWA (offline support)

## 🐛 Issues Conhecidos

### Módulo 1
- Material Web Components podem demorar para carregar
- Algumas animações podem não funcionar em browsers antigos
- CORS issues se aberto diretamente como arquivo

### Soluções
- Usar servidor HTTP local
- Fallbacks para browsers antigos
- Loading states para componentes

## 📋 TODOs Imediatos

### Módulo 2 (Próxima Semana)
- [ ] Criar view `forms.js` (lista de formulários)
- [ ] Criar view `form-builder.js` (construtor)
- [ ] Implementar CRUD de formulários
- [ ] Sistema de drag & drop para questões
- [ ] Preview em tempo real

## 🎯 Funcionalidades Principais Implementadas

### 📝 **Sistema de Formulários**
- **Criação de formulários**: Interface intuitiva para criar novos formulários
- **Gerenciamento**: Página "Meus Formulários" com visualização em cards
- **Edição inline**: Título e descrição editáveis diretamente na interface
- **Sincronização automática**: Dados sincronizados entre todas as páginas

### 📋 **Sistema de Questões**
- **Múltipla escolha**: Sistema completo de questões com opções editáveis
- **Enunciados markdown**: Suporte completo a GitHub Flavored Markdown
- **Tabelas responsivas**: Renderização automática com scroll horizontal
- **"Outro item"**: Expansão automática de opções sem botões extras
- **Registry-based**: Gerenciamento de estado centralizado e eficiente

### 🎨 **Interface e UX**
- **Material Design 3**: Design system completo com tokens de cor
- **Responsivo**: Adaptação perfeita para desktop, tablet e mobile
- **Navegação SPA**: Transições suaves entre páginas
- **Sidebar inteligente**: Menu lateral com formulários em memória

### 📄 **Renderização de Conteúdo**
- **Markdown completo**: GitHub Flavored Markdown com tabelas
- **Acentos preservados**: Codificação UTF-8 correta para português
- **Tabelas estilizadas**: Headers em negrito, bordas e efeito zebra
- **Performance otimizada**: Event delegation e lazy loading

### ⚠️ Observação Importante - Foco do Desenvolvimento
**Contexto atual**: Interface de **criação de formulários** (form builder). A UX/UI é focada em **criadores de formulários** construindo **enunciados de questões**.

### Melhorias
- [ ] Testes unitários
- [ ] Error boundaries
- [ ] Performance optimization
- [ ] Accessibility (WCAG 2.1)
- [ ] Internacionalização (i18n)

## 🤝 Contribuição

1. Siga o plano de desenvolvimento modular
2. Mantenha consistência com Material Design 3
3. Teste em diferentes dispositivos
4. Documente novas funcionalidades

## 📄 Licença

Este é um projeto educacional/experimental. Consulte o proprietário para uso comercial.

---

**Status**: Módulos 1-4 completos ✅ | Próximo: Módulo 5 (Suporte LaTeX avançado)

**Última atualização**: 20/07/2025
